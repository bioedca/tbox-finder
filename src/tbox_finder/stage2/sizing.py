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

from tbox_finder.stage2 import heads as H
from tbox_finder.stage2 import losses as L
from tbox_finder.stage2 import train as T

__all__ = [
    "DEFAULT_BATCH_SWEEP",
    "DEFAULT_OUT",
    "SCHEMA_VERSION",
    "STEP",
    "build_report",
    "derive_clauses",
    "measure_batch",
    "run_sizing",
    "validate_report",
]

SCHEMA_VERSION = "1"
STEP = "P3-06-sizing"
DEFAULT_OUT = "reports/p3/stage2_sizing.json"

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
) -> dict[str, Any]:
    """Peak VRAM + step time for one (batch_size, regime), or a clean ``oom`` record.

    A fresh model and optimiser per call: carrying them between points would let one point's
    allocator state set the next point's peak, and the whole purpose here is a number that
    means something on its own.
    """
    import torch

    spec = H.load_head_spec()
    cfg = T.Stage2TrainConfig(
        batch_size=batch_size,
        gradient_checkpointing=gradient_checkpointing,
        loss=loss_config,
        structure_head=loss_config.structure_enabled,
        device=device,
    )
    record: dict[str, Any] = {
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
    }
    return {k: bool(v) for k, v in clauses.items()}


def validate_report(report: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {report.get('schema_version')!r} != {SCHEMA_VERSION!r}")
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
    if set(recorded) != set(recomputed):
        problems.append(f"gate.clauses keys {sorted(recorded)} != re-derived {sorted(recomputed)}")
    for name, value in recomputed.items():
        if bool(recorded.get(name)) != value:
            problems.append(f"gate.clauses[{name!r}] disagrees with re-derivation ({value})")
    if bool(gate.get("overall_pass")) != all(recomputed.values()):
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
        "population": dict(population),
        "device": dict(device),
        "gradient_checkpointing": dict(checkpointing),
        "measurements": [dict(m) for m in measurements],
        "largest_fitting_batch_worst_case": largest,
        "recommendation": _recommend(fitting, largest, device),
        "provenance": {"env_lock": T.ENV_LOCK, "entrypoint": "tbox_finder.stage2.sizing"},
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

    measurements: list[dict[str, Any]] = []
    for batch_size in batch_sweep:
        for regime, picker in (("worst_case", _worst_case_rows), ("typical", _typical_rows)):
            chosen = picker(rows, batch_size=batch_size)
            log(f"── measuring batch={batch_size} regime={regime} ──")
            record = measure_batch(
                chosen,
                batch_size=batch_size,
                steps=steps,
                gradient_checkpointing=True,
                loss_config=loss_config,
                base_model=base_model,
                device=device,
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
    checkpointing: dict[str, Any] = {"batch_size": smallest, "comparison_skipped_reason": None}
    on = next(
        (
            m
            for m in measurements
            if m["batch_size"] == smallest and m["regime"] == "worst_case" and not m["oom"]
        ),
        None,
    )
    if on is None:
        checkpointing["comparison_skipped_reason"] = (
            f"the batch-{smallest} worst case did not fit even WITH checkpointing, so an "
            "off-comparison would only OOM harder and measure nothing"
        )
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
    args = parser.parse_args(argv)
    sweep = tuple(int(b) for b in str(args.batch_sweep).split(",") if b.strip())
    report = run_sizing(
        dataset_parquet=args.dataset,
        batch_sweep=sweep,
        steps=args.steps,
        max_records=args.max_records,
        out_path=args.out,
    )
    largest = report["largest_fitting_batch_worst_case"]
    print(f"SIZING_DONE largest_fitting_worst_case_batch={largest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_run())
