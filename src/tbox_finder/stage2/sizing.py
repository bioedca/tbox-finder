"""P3-06 sizing smoke — measure the Stage-2 step's real VRAM and wall cost, on one GPU.

**Why this exists.** P3-06's first two production submits were sized from
``reports/p1/lora_vram_smoke.json`` (P1-16), which reports 2.014 GiB peak at batch 8. That
report's own config block says what it measured: ``loss_is_placeholder: true``,
``data_is_synthetic: true``, ``is_science: false``, ``seq_len_nt: 350``, five timed steps —
a bare LoRA-wrapped **backbone** under ``placeholder_loss(hidden_states)``, with no heads and
no objective. The real seven-head step on real sequences allocated **13.97 GiB** and OOM'd
every point of job 1044. The number was not wrong; it was answering a different question.

So this harness measures **the shipped step**: it calls
:func:`tbox_finder.stage2.train.forward_backward` — the same function the trainer calls —
over the same :class:`~tbox_finder.stage2.train.Stage2SequenceDataset` and the same
:class:`~tbox_finder.stage2.losses.MultitaskLoss`, on **real rows from the real admitted
population**. It cannot drift from the trainer without the trainer changing too.

**What it measures, and why each part matters.**

* A **descending batch sweep**. The largest batch that fits is the answer the production run
  needs, and OOM is caught per point so one failure does not end the sweep.
* Both a **typical** and a **worst-case** batch. Sequences here run 59–552 tokens and the
  collator pads to the batch maximum, so a batch that happens to contain the longest row
  costs what a batch of *all* longest rows costs. Sizing on the median would reproduce this
  whole failure at a lower batch size — the worst case is the one that has to fit.
* **Several optimiser steps, not one.** AdamW allocates ``exp_avg``/``exp_avg_sq`` lazily on
  the first ``.step()``, so a one-step measurement understates steady-state peak; and a
  footprint that *grows* across steps (the open question job 1044 left) is only visible as a
  per-step series, which is why every step's peak is recorded rather than just the maximum.
* **Gradient-checkpointing effectiveness, measured not assumed.** The count of modules
  carrying the flag, plus an explicit on/off comparison at a batch size that fits both ways.
  "Enabled" that saves nothing is the failure mode that looks exactly like success.

The report is written even when every point OOMs — a sizing run that learns "batch 8 does not
fit" has succeeded at its job, so its gate asks whether the *measurement* is sound, never
whether the numbers are convenient.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import report_schema as RSCH
from tbox_finder.models.rna_backbone_registry import (
    PRODUCTION_BACKBONE,
    backbone_summary,
    resolve_backbone,
)
from tbox_finder.stage2 import heads as H
from tbox_finder.stage2 import losses as L
from tbox_finder.stage2 import train as T

__all__ = [
    "DEFAULT_BATCH_SWEEP",
    "DEFAULT_OUT",
    "CLAUSES_FIRST_REQUIRED_AT",
    "KNOWN_SCHEMAS",
    "LEGACY_SCHEMAS_WITHOUT_SKIP_CODE",
    "SCHEMA_VERSION",
    "SKIP_ON_ARM_DID_NOT_FIT",
    "SKIP_PORT_CANNOT_CHECKPOINT",
    "STEP",
    "build_report",
    "clauses_not_required_at",
    "derive_clauses",
    "measure_batch",
    "run_sizing",
    "validate_report",
]

#: Bumped 1 -> 2 with the clause set (CodeRabbit r2): `measured_the_requested_batch` was added
#: and `recommendation` became validator-re-derived. Job 1051's committed report is schema 1;
#: its numbers stand, but it does not carry the schema-2 clause and its recommendation predates
#: the computed-headroom shape.
#: Bumped 2 -> 3 at P3-17: the report gains a `backbone` block and the clause set gains
#: `checkpointing_skip_is_earned`, and a clause set is part of a report's shape
#: ([[new-gate-clause-invalidates-old-reports]]). Job 1051's committed report is schema **1**
#: — this note used to say schema 2, which `reports/p3/stage2_sizing.json` disproves; its
#: numbers stand and it is not regenerated.
#: Bumped 3 -> 4 at P3-17 review round 5: `provenance.env_lock` stopped being the production
#: constant and became the SUBJECT's lock, and the clause set gained
#: `provenance_env_lock_is_the_backbones` and a machine-readable
#: `gradient_checkpointing.comparison_skipped_code`. Job 1374's committed RNA-FM report is
#: schema 3: its GiB stand, but it carries the stale `provenance.env_lock` this bump exists to
#: stop, and it is NOT regenerated here (re-measuring VRAM needs an A4000 — see the dev-log
#: disclosure) ([[new-gate-clause-invalidates-old-reports]]).
SCHEMA_VERSION = "4"
STEP = "P3-06-sizing"
DEFAULT_OUT = "reports/p3/stage2_sizing.json"

#: Why the checkpointing on/off comparison was skipped, as a VALUE rather than as a sentence.
#: `checkpointing_skip_is_earned` used to accept the skip by looking for the substring "did not
#: fit" inside `comparison_skipped_reason`, which coupled a gate clause to the wording of a
#: human-facing message: rewording the sentence silently flipped the clause. The sentence stays
#: for the reader; the clause grades these.
SKIP_PORT_CANNOT_CHECKPOINT = "port_cannot_run_checkpointing"
SKIP_ON_ARM_DID_NOT_FIT = "on_arm_did_not_fit"

#: The schemas written BEFORE `comparison_skipped_code` existed, listed rather than compared.
#: `schema_version` is a string, and `"3" < "4"` only happens to work while both are one digit
#: — the same string-compare trap this branch already fixed once in the library-version check.
LEGACY_SCHEMAS_WITHOUT_SKIP_CODE = frozenset({"1", "2", "3"})

#: Every schema this validator knows, oldest first — listed, never string-compared.
KNOWN_SCHEMAS: tuple[str, ...] = ("1", "2", "3", "4")

#: The clauses a report FIRST carries at each schema. A report written before a clause existed
#: cannot have recorded it, and re-grading it under today's set reports a failure that is
#: really an age difference: `reports/p3/stage2_sizing.json` (schema 1, job 1051) grades
#: `checkpointing_skip_is_earned` FALSE under the schema-3 set purely because
#: `usable_on_this_backbone` did not exist when it was written
#: ([[new-gate-clause-invalidates-old-reports]]).
CLAUSES_FIRST_REQUIRED_AT: dict[str, frozenset[str]] = {
    "2": frozenset({"measured_the_requested_batch"}),
    "3": frozenset({"checkpointing_skip_is_earned"}),
    "4": frozenset(
        {"provenance_env_lock_is_the_backbones", "checkpointing_usability_agrees_with_the_port"}
    ),
}


def clauses_not_required_at(schema: str) -> frozenset[str]:
    """Clauses introduced AFTER ``schema``, which a report at that schema cannot carry."""
    return RSCH.clauses_not_required_at(
        schema, known=KNOWN_SCHEMAS, first_required_at=CLAUSES_FIRST_REQUIRED_AT
    )


RSCH.check_schema_tables(
    known=KNOWN_SCHEMAS,
    first_required_at=CLAUSES_FIRST_REQUIRED_AT,
    current=SCHEMA_VERSION,
    module=__name__,
)


#: Descending, so the sweep learns the ceiling before spending time under it. 8 is what job
#: 1044 tried and lost; 1 is the floor below which the arm is not viable on this card.
DEFAULT_BATCH_SWEEP: tuple[int, ...] = (8, 4, 2, 1)

#: Enough steps for AdamW's state to be allocated and for a per-step growth trend to show.
DEFAULT_STEPS = 6

#: Rows to draw from. Small — this measures a step, not a schedule.
DEFAULT_MAX_RECORDS = 200


def _gib(n_bytes: float) -> float:
    return round(float(n_bytes) / (1024**3), 4)


def _select_rows(dataset_parquet: str, *, max_records: int) -> tuple[list[Any], dict[str, Any]]:
    """Real rows from the real admitted population — never synthetic, never the whole file.

    Returns the rows plus the census, so the report can state which population it measured
    rather than leaving the reader to assume it was the training one.
    """
    rows = T.load_rows(dataset_parquet)
    admitted, census = T.select_rows(rows, rung=T.TRAIN_RUNG)
    picked = [rows[i] for i in admitted[:max_records]]
    return picked, census


def _worst_case_rows(rows: Sequence[Mapping[str, Any]], *, batch_size: int) -> list[Any]:
    """The ``batch_size`` LONGEST rows — the batch the collator pads everything else up to."""
    return sorted(rows, key=lambda r: int(r.get("n_tokens") or 0), reverse=True)[:batch_size]


def _typical_rows(rows: Sequence[Mapping[str, Any]], *, batch_size: int) -> list[Any]:
    """A median-length window — what a run spends most of its steps on."""
    ordered = sorted(rows, key=lambda r: int(r.get("n_tokens") or 0))
    mid = len(ordered) // 2
    lo = max(0, mid - batch_size // 2)
    return ordered[lo : lo + batch_size]


def measure_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    steps: int,
    gradient_checkpointing: bool,
    loss_config: L.Stage2LossConfig,
    base_model: Any = None,
    device: str | None = None,
    backbone: str = PRODUCTION_BACKBONE,
) -> dict[str, Any]:
    """Peak VRAM + step time for one (batch_size, regime), or a clean ``oom`` record.

    A fresh model and optimiser per call: carrying them between points would let one point's
    allocator state set the next point's peak, and the whole purpose here is a number that
    means something on its own.
    """
    import torch

    spec = H.load_head_spec()
    # The per-arm destinations are passed even though sizing writes NEITHER: since ADR-0002
    # A15 `Stage2TrainConfig.__post_init__` refuses a non-production backbone aimed at the
    # production checkpoint/report paths, and inheriting those defaults here would make the
    # comparator's sizing leg unrunnable for a reason that has nothing to do with sizing.
    cfg = T.Stage2TrainConfig(
        backbone=backbone,
        checkpoint_dir=T.default_checkpoint_dir(backbone),
        report_path=T.default_report_path(backbone),
        batch_size=batch_size,
        gradient_checkpointing=gradient_checkpointing,
        loss=loss_config,
        structure_head=loss_config.structure_enabled,
        device=device,
    )
    record: dict[str, Any] = {
        # WHICH model these GiB describe. A sizing number is meaningless without it, and the
        # two arms differ by 6.5x in parameters — a report that omitted this could be read as
        # sizing either one ([[size-a-run-from-the-protocols-own-report]]).
        "backbone": backbone,
        "batch_size": batch_size,
        "measured_batch_size": None,
        "gradient_checkpointing": gradient_checkpointing,
        "oom": False,
        "error": None,
        "n_steps": 0,
        "peak_vram_gib": None,
        "peak_vram_gib_per_step": [],
        "step_ms": [],
        "padded_tokens": None,
        "n_modules_with_checkpointing": None,
    }
    model = optimizer = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, wrap = T.build_model(cfg, base_model=base_model)
        record["n_modules_with_checkpointing"] = wrap.get("n_modules_with_checkpointing")
        target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(target)
        loss_fn = L.MultitaskLoss(loss_config)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)

        ds = T.Stage2SequenceDataset(rows, spec, loss_config=loss_config)
        batch = T.collate_stage2(
            [ds[i] for i in range(min(batch_size, len(ds)))],
            ignore_index=loss_config.ignore_index,
        )
        record["padded_tokens"] = int(batch["input_ids"].shape[1])
        # The requested batch is truncated to len(ds); recording only the request would let a
        # report say "batch 8 fits" when 3 rows were measured.
        record["measured_batch_size"] = int(batch["input_ids"].shape[0])
        batch = T._to_device(batch, target)

        # ⚠ Reset the peak counter per step, or the series is a RUNNING MAXIMUM and monotonic
        # by construction — which is what job 1051 actually recorded ([8.0847, 8.1827, 8.1827,
        # …]). A cumulative max can still reveal a leak, so that run's "no growth" conclusion
        # survives; but it cannot show a per-step footprint, and `growth_ratio` over it means
        # something weaker than its name suggests. The overall peak is tracked separately.
        overall_peak = 0
        for _ in range(steps):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            T.forward_backward(model, loss_fn, batch)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            # AdamW allocates exp_avg/exp_avg_sq on the FIRST step, so the optimiser step is
            # inside the measured region rather than after it.
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                step_peak = torch.cuda.max_memory_allocated()
                overall_peak = max(overall_peak, step_peak)
                record["peak_vram_gib_per_step"].append(_gib(step_peak))
            record["step_ms"].append(round(1000.0 * (time.perf_counter() - started), 2))
            record["n_steps"] += 1
        if torch.cuda.is_available():
            record["peak_vram_gib"] = _gib(overall_peak)
    except Exception as exc:  # noqa: BLE001 - an OOM here is a RESULT, not a crash
        is_oom = "OutOfMemoryError" in type(exc).__name__ or "out of memory" in str(exc).lower()
        record["oom"] = bool(is_oom)
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        if not is_oom:
            raise
    finally:
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return record


def _growth_ratio(series: Sequence[float]) -> float | None:
    """Last step's peak ÷ the first's — > 1 means the footprint is still climbing.

    Job 1044 left open whether its 13.97 GiB was a per-step cost or an accumulation. This is
    the number that answers it, so it is computed rather than left to a reader eyeballing a
    list.
    """
    usable = [v for v in series if isinstance(v, (int, float)) and v > 0]
    if len(usable) < 2:
        return None
    return round(usable[-1] / usable[0], 4)


def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """Re-derive the gate from recorded evidence.

    The gate asks whether the MEASUREMENT is sound — not whether the answer is convenient.
    "Batch 8 does not fit" is a successful sizing run.
    """
    meas = report.get("measurements") or []
    device = report.get("device") or {}
    population = report.get("population") or {}
    ckpt = report.get("gradient_checkpointing") or {}
    prov = report.get("provenance") or {}

    with_numbers = [m for m in meas if not m.get("oom") and m.get("peak_vram_gib")]
    clauses = {
        # It ran on the hardware the production run targets, not on a laptop or a CPU.
        "measured_on_cuda": bool(device.get("is_cuda")) and bool(device.get("name")),
        # The whole point: real rows, real objective, real step.
        "real_objective": (
            report.get("data_is_synthetic") is False
            and report.get("loss_is_placeholder") is False
            and bool(population.get("n_rows"))
            and bool(report.get("step_function") == "tbox_finder.stage2.train.forward_backward")
        ),
        # The sweep was attempted across several points and every one reached a DEFINITE
        # outcome — it ran steps, or it OOM'd. Deliberately not "at least one point
        # succeeded": an all-OOM sweep is a valid measurement (nothing fits on this card is
        # exactly the kind of answer this harness exists to produce), and a clause demanding
        # success would push the harness toward a convenient number. What it does refuse is a
        # point that neither ran nor failed — a measurement that silently did nothing.
        "swept": len(meas) >= 2
        and all(m.get("n_steps", 0) > 0 or m.get("oom") is True for m in meas),
        # Enough steps that AdamW state is allocated and a trend is visible.
        "enough_steps_for_optimizer_state": all(m.get("n_steps", 0) >= 3 for m in with_numbers),
        # A fitting point must have MEASURED the batch size it reports, or "batch 8 fits" can
        # mean "3 rows fit" — the request standing in for the measurement once again.
        "measured_the_requested_batch": all(
            m.get("measured_batch_size") == m.get("batch_size") for m in with_numbers
        ),
        # The worst case was actually exercised — sizing on the median is how this failed.
        "worst_case_measured": any(m.get("regime") == "worst_case" for m in meas),
        # Checkpointing effectiveness was measured, not assumed.
        "checkpointing_effect_measured": (
            ckpt.get("on_peak_gib") is not None and ckpt.get("off_peak_gib") is not None
        )
        or ckpt.get("comparison_skipped_reason") is not None,
        # ...and a SKIP has to be earned. The clause above accepts any non-null reason, so on
        # its own it lets a run opt out of the comparison by writing a sentence. This binds
        # the skip to the report's own recorded evidence: either the comparison ran, or the
        # port genuinely cannot run the 'on' arm, or the 'on' arm OOM'd — and the sweep's
        # measurements must agree with the usability flag rather than merely accompany it.
        # ⚠ The OOM disjunct grades `comparison_skipped_code`, NOT the wording of
        # `comparison_skipped_reason`. Grading the sentence made this clause turn on prose:
        # rewording the message flipped a gate. Schema-3 reports carry no code, so the legacy
        # substring is still accepted for them ALONE — never for a report this code writes.
        "checkpointing_skip_is_earned": (
            (ckpt.get("on_peak_gib") is not None and ckpt.get("off_peak_gib") is not None)
            or ckpt.get("usable_on_this_backbone") is False
            or ckpt.get("comparison_skipped_code") == SKIP_ON_ARM_DID_NOT_FIT
            or (
                str(report.get("schema_version")) in LEGACY_SCHEMAS_WITHOUT_SKIP_CODE
                and "did not fit" in str(ckpt.get("comparison_skipped_reason") or "")
            )
        )
        # ⚠ EVERY measurement must record the flag, and every one must agree with the
        # usability fact. The `for ... if m.get(...) is not None` form this replaces went
        # vacuously TRUE the moment `measure_batch` stopped recording the key — the sole
        # writer, unpinned, and an `all()` over an empty sequence is True
        # ([[clauses-must-guard-emptiness]]). The old `and not m.get("off_comparison")`
        # exclusion is gone with it: nothing in this repo ever wrote that key (the
        # off-comparison record is never appended to `measurements`), so it excluded nothing
        # while reading like a considered carve-out.
        and bool(meas)
        and all(m.get("gradient_checkpointing") is not None for m in meas)
        and all(
            bool(m.get("gradient_checkpointing")) is bool(ckpt.get("usable_on_this_backbone"))
            for m in meas
        ),
        # The clause that was missing when job 1374's report stamped the PRODUCTION lock over
        # an RNA-FM measurement. `provenance.env_lock` is resolved from the REQUESTED key and
        # `backbone.env_lock` from the RESOLVED spec, so a producer that goes back to stamping
        # a module constant makes the two disagree and flips this FALSE. Nothing graded either
        # field before, which is exactly why the wrong one survived a passing gate.
        # The skip clause above accepts `usable_on_this_backbone: False` as earning a skip, and
        # nothing checked that flag against the PORT FACT the same report carries three keys
        # away in `backbone.gradient_checkpointing_usable`. A report that disagreed with itself
        # — claiming the port cannot checkpoint while recording that it can — earned its skip
        # clean ([[gate-must-bind-to-upstream-evidence]]).
        "checkpointing_usability_agrees_with_the_port": (
            ckpt.get("usable_on_this_backbone") is not None
            and (report.get("backbone") or {}).get("gradient_checkpointing_usable") is not None
            and bool(ckpt.get("usable_on_this_backbone"))
            is bool((report.get("backbone") or {}).get("gradient_checkpointing_usable"))
        ),
        "provenance_env_lock_is_the_backbones": (
            bool(prov.get("env_lock"))
            and prov.get("env_lock") == (report.get("backbone") or {}).get("env_lock")
        ),
    }
    return {k: bool(v) for k, v in clauses.items()}


def validate_report(report: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    schema = str(report.get("schema_version"))
    if schema not in KNOWN_SCHEMAS:
        problems.append(
            f"schema_version {report.get('schema_version')!r} is not one of {KNOWN_SCHEMAS!r}"
        )
    if report.get("step") != STEP:
        problems.append(f"step {report.get('step')!r} != {STEP!r}")
    gate = report.get("gate")
    if not isinstance(gate, Mapping):
        problems.append("missing gate block")
        return problems
    recomputed = derive_clauses(report)
    recorded = gate.get("clauses")
    if not isinstance(recorded, Mapping):
        problems.append("gate.clauses is missing")
        return problems
    # A report is graded against the clause set of ITS OWN schema. At the current schema this
    # excuses nothing and the check is unchanged.
    expected_clauses = {
        name: value
        for name, value in recomputed.items()
        if name not in clauses_not_required_at(schema)
    }
    if set(recorded) != set(expected_clauses):
        problems.append(
            f"gate.clauses keys {sorted(recorded)} != re-derived {sorted(expected_clauses)}"
        )
    for name, value in expected_clauses.items():
        if bool(recorded.get(name)) != value:
            problems.append(f"gate.clauses[{name!r}] disagrees with re-derivation ({value})")
    if bool(gate.get("overall_pass")) != all(expected_clauses.values()):
        problems.append("gate.overall_pass disagrees with the re-derived clauses")
    # ⚠ Re-derive the RECOMMENDATION too, not just the clauses. It is the one field a reader
    # acts on — it chose batch_size=4 for the production sweep — and nothing checked that it
    # followed from `measurements`. A report edited by hand, or written by an older shape,
    # would ship a stale recommendation past a green gate.
    meas = report.get("measurements") or []
    fitting = [m for m in meas if not m.get("oom") and m.get("regime") == "worst_case"]
    expected = _recommend(
        fitting,
        max((m["batch_size"] for m in fitting), default=None),
        report.get("device") or {},
    )
    if report.get("recommendation") != expected:
        problems.append(
            f"recommendation {report.get('recommendation')!r} does not follow from the "
            f"measurements; re-derivation gives {expected!r}"
        )
    return problems


def _recommend(
    fitting: Sequence[Mapping[str, Any]], largest: int | None, device: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The recommended batch size, with headroom COMPUTED rather than asserted.

    An earlier version said "fits with headroom" without ever calculating any — the same
    species of claim as a gate clause that reads a flag instead of measuring an effect. If the
    numbers needed are absent the fields are ``None`` and the basis states only what was
    observed: that the batch ran without OOM.
    """
    if largest is None:
        return None
    peak = next((m.get("peak_vram_gib") for m in fitting if m.get("batch_size") == largest), None)
    total = device.get("total_memory_gib")
    headroom = (
        round(float(total) - float(peak), 4)
        if isinstance(peak, (int, float)) and isinstance(total, (int, float))
        else None
    )
    return {
        "batch_size": largest,
        "worst_case_peak_gib": peak,
        "device_total_gib": total,
        "headroom_gib": headroom,
        "basis": (
            "largest worst-case batch observed to run without OOM; headroom is measured and "
            "EXCLUDES DDP gradient buckets and NCCL buffers, which a single-GPU measurement "
            "does not exercise"
        ),
    }


def build_report(
    *,
    measurements: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    device: Mapping[str, Any],
    checkpointing: Mapping[str, Any],
    config: Mapping[str, Any],
    backbone: Mapping[str, Any],
) -> dict[str, Any]:
    fitting = [m for m in measurements if not m.get("oom") and m.get("regime") == "worst_case"]
    largest = max((m["batch_size"] for m in fitting), default=None)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "prd": "PRD §10.3 (the 16 GB A4000 budget)",
        "written_at": datetime.now(UTC).isoformat(),
        # The three flags whose ABSENCE from P1-16's report is why P3-06 was mis-sized.
        # Stated affirmatively here so a future reader can tell at a glance what this
        # measurement covers, without reading the code that produced it.
        "data_is_synthetic": False,
        "loss_is_placeholder": False,
        "step_function": "tbox_finder.stage2.train.forward_backward",
        "supersedes_for_stage2_sizing": "reports/p1/lora_vram_smoke.json (P1-16)",
        "config": dict(config),
        "backbone": dict(backbone),
        "population": dict(population),
        "device": dict(device),
        "gradient_checkpointing": dict(checkpointing),
        "measurements": [dict(m) for m in measurements],
        "largest_fitting_batch_worst_case": largest,
        "recommendation": _recommend(fitting, largest, device),
        "provenance": {
            # NOT `T.ENV_LOCK`. That constant names the PRODUCTION lock, so stamping it made
            # job 1374's RNA-FM sizing report claim an environment the run did not use, while
            # the `backbone` block in the same artifact recorded the right one. Resolve it from
            # the subject, and let `provenance_env_lock_is_the_backbones` grade the agreement.
            # `requested_key` is required, not defaulted: a missing subject must raise, not
            # silently fall back to production.
            "env_lock": T.env_lock_for(str(backbone["requested_key"])),
            "entrypoint": "tbox_finder.stage2.sizing",
        },
    }
    clauses = derive_clauses(report)
    report["gate"] = {
        "clauses": clauses,
        "overall_pass": all(clauses.values()),
        "failed": sorted(n for n, ok in clauses.items() if not ok),
    }
    return report


def run_sizing(
    *,
    dataset_parquet: str = T.DEFAULT_DATASET,
    batch_sweep: Sequence[int] = DEFAULT_BATCH_SWEEP,
    steps: int = DEFAULT_STEPS,
    max_records: int = DEFAULT_MAX_RECORDS,
    out_path: str = DEFAULT_OUT,
    base_model: Any = None,
    device: str | None = None,
    backbone: str = PRODUCTION_BACKBONE,
    log: Any = print,
) -> dict[str, Any]:
    """Run the sweep, write the report, return it. Raises if the report fails its own gate."""
    import torch

    rows, census = _select_rows(dataset_parquet, max_records=max_records)
    lengths = [int(r.get("n_tokens") or 0) for r in rows]
    population = {
        "dataset_parquet": dataset_parquet,
        "rung": T.TRAIN_RUNG,
        "n_rows": len(rows),
        "n_admitted_total": census.get("n_admitted"),
        "n_tokens_min": min(lengths) if lengths else None,
        "n_tokens_max": max(lengths) if lengths else None,
        "n_tokens_median": sorted(lengths)[len(lengths) // 2] if lengths else None,
    }
    device_block = {
        "is_cuda": bool(torch.cuda.is_available()),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_memory_gib": (
            _gib(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else None
        ),
        "capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
    }
    loss_config = L.Stage2LossConfig()

    # ⚠ Whether the sweep may enable checkpointing is a property of the PORT, not a choice.
    # This harness used to hardcode True, which is correct for the rotary production backbone
    # and fatal for RNA-FM: SLURM job 1370 died here, in the first measurement, because that
    # port adds its absolute position embeddings in place while checkpointing makes the
    # embedding output a leaf requiring grad. Sizing with a setting the arm cannot run would
    # measure a configuration that never trains.
    spec = resolve_backbone(backbone)
    checkpointing_usable = bool(spec.gradient_checkpointing_usable)

    measurements: list[dict[str, Any]] = []
    for batch_size in batch_sweep:
        for regime, picker in (("worst_case", _worst_case_rows), ("typical", _typical_rows)):
            chosen = picker(rows, batch_size=batch_size)
            log(f"── measuring batch={batch_size} regime={regime} ──")
            record = measure_batch(
                chosen,
                batch_size=batch_size,
                steps=steps,
                gradient_checkpointing=checkpointing_usable,
                loss_config=loss_config,
                base_model=base_model,
                device=device,
                backbone=backbone,
            )
            record["regime"] = regime
            record["growth_ratio"] = _growth_ratio(record["peak_vram_gib_per_step"])
            measurements.append(record)
            log(
                f"   peak={record['peak_vram_gib']} GiB oom={record['oom']} "
                f"tokens={record['padded_tokens']} growth={record['growth_ratio']}"
            )
            # Once the worst case fits, the smaller batches only confirm what is already
            # known — but keep sweeping so the report carries the whole curve.

    # Checkpointing effectiveness, at the smallest batch so both regimes have a chance to fit.
    smallest = min(batch_sweep)
    checkpointing: dict[str, Any] = {
        "batch_size": smallest,
        "comparison_skipped_reason": None,
        "comparison_skipped_code": None,
        # Recorded so an absent comparison is a STATED fact rather than a missing key. The
        # asymmetry between the two arms' sizing reports is real and has to be visible.
        "usable_on_this_backbone": checkpointing_usable,
        "usability_note": spec.gradient_checkpointing_note,
    }
    if not checkpointing_usable:
        checkpointing["comparison_skipped_reason"] = (
            f"gradient checkpointing is not usable on backbone {spec.key!r}, so there is no "
            f"'on' arm to compare against: {spec.gradient_checkpointing_note}"
        )
        checkpointing["comparison_skipped_code"] = SKIP_PORT_CANNOT_CHECKPOINT
        checkpointing["on_peak_gib"] = None
        checkpointing["off_peak_gib"] = None
    on = next(
        (
            m
            for m in measurements
            if m["batch_size"] == smallest and m["regime"] == "worst_case" and not m["oom"]
        ),
        None,
    )
    if not checkpointing_usable:
        pass  # already recorded above; there is no 'on' arm to measure on this port
    elif on is None:
        checkpointing["comparison_skipped_reason"] = (
            f"the batch-{smallest} worst case did not fit even WITH checkpointing, so an "
            "off-comparison would only OOM harder and measure nothing"
        )
        checkpointing["comparison_skipped_code"] = SKIP_ON_ARM_DID_NOT_FIT
        checkpointing["on_peak_gib"] = None
        checkpointing["off_peak_gib"] = None
    else:
        off = measure_batch(
            _worst_case_rows(rows, batch_size=smallest),
            batch_size=smallest,
            steps=max(3, steps // 2),
            gradient_checkpointing=False,
            loss_config=loss_config,
            base_model=base_model,
            device=device,
            # The SECOND call site. Threading the key into only the sweep would have left this
            # off-comparison measuring the production backbone against an RNA-FM "on" number,
            # i.e. a saving_ratio computed across two different models
            # ([[fixed-one-of-two-identical-things]]).
            backbone=backbone,
        )
        checkpointing["on_peak_gib"] = on.get("peak_vram_gib")
        checkpointing["off_peak_gib"] = off.get("peak_vram_gib")
        checkpointing["off_oom"] = off.get("oom")
        checkpointing["n_modules_with_flag"] = on.get("n_modules_with_checkpointing")
        if on.get("peak_vram_gib") and off.get("peak_vram_gib"):
            checkpointing["saving_ratio"] = round(off["peak_vram_gib"] / on["peak_vram_gib"], 4)
            # A flag that is "on" and saves nothing is the failure that looks like success.
            checkpointing["effective"] = checkpointing["saving_ratio"] > 1.05

    report = build_report(
        measurements=measurements,
        population=population,
        device=device_block,
        checkpointing=checkpointing,
        config={
            "batch_sweep": list(batch_sweep),
            "steps": steps,
            "max_records": max_records,
            "world_size": 1,
            "optimizer": "AdamW",
            "dtype": "bfloat16 (lora_harness.TRAIN_DTYPE)",
            "gradient_checkpointing": checkpointing_usable,
        },
        backbone={
            **backbone_summary(spec),
            # The measurement's own subject, so a reader never has to infer which model these
            # GiB describe — the two arms differ by ~6.5x in parameters.
            "requested_key": backbone,
        },
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"wrote {out_path}")

    problems = validate_report(report)
    if problems:
        raise RuntimeError(f"sizing report is malformed: {problems}")
    if not report["gate"]["overall_pass"]:
        raise RuntimeError(
            f"sizing gate FAILED (clauses: {', '.join(report['gate']['failed'])}). The gate "
            "grades the MEASUREMENT, not the answer — a failure here means the numbers cannot "
            "be trusted, not that the batch size is too big."
        )
    return report


def _run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-2 sizing smoke (P3-06)")
    parser.add_argument("--dataset", default=T.DEFAULT_DATASET)
    parser.add_argument("--batch-sweep", default=",".join(str(b) for b in DEFAULT_BATCH_SWEEP))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--backbone",
        default=PRODUCTION_BACKBONE,
        help=(
            "rna_backbone_registry allow-list key to size (ADR-0002 A15). The arms differ by "
            "~6.5x in parameters, so sizing one and training the other measures nothing."
        ),
    )
    args = parser.parse_args(argv)
    sweep = tuple(int(b) for b in str(args.batch_sweep).split(",") if b.strip())
    report = run_sizing(
        dataset_parquet=args.dataset,
        batch_sweep=sweep,
        steps=args.steps,
        max_records=args.max_records,
        out_path=args.out,
        backbone=args.backbone,
    )
    largest = report["largest_fitting_batch_worst_case"]
    print(f"SIZING_DONE largest_fitting_worst_case_batch={largest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_run())
