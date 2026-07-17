"""P2-05 — Stage-1 training sizing: aggregate per-point runs into a footprint report.

**What this module is.** A pure aggregator over N ``train_stage1`` reports, one per
configuration point (world_size × batch_size × gradient_checkpointing). It computes the
four quantities imp.md P2-05 asks for — per-GPU VRAM headroom vs 16 GB, windows/sec/GPU,
DDP scaling efficiency, and the gradient-checkpointing throughput cost — and from them an
**illustrative** full-run wall-clock + GPU-hours extrapolation.

**Why an aggregator and not a benchmark loop.** The thing being timed must be the thing
that ships. A bespoke benchmark loop would measure a loop nobody trains with; instead
``train_stage1`` was instrumented in place (``steps.step_seconds`` /
``steps.batch_wait_seconds``) and this module only reads what the real entrypoint recorded.
Forking a second train step to time it would repeat the P2-02 focal-loss lesson, where two
copies of one function meant fixing one and shipping the bug in the other.

**Why one process per point.** A CUDA OOM poisons the context, so a batch sweep inside one
process cannot distinguish "batch 32 OOMs" from "batch 32 OOMs *because* batch 16 leaked".
P2-04 established the pattern; ``slurm/p2/sizing_smoke.sbatch`` drives it.

**⚠️ What this module deliberately does NOT do: freeze a budget.**
ADR-0003 D7 governs the training/sweep GPU-hour budget. Its determination rule is
``budget ≈ (sweep-grid cardinality × per-run GPU-hours) + (continued-pretraining pass)``,
and it says plainly: *"Number change → ADR sign-off (§7 item 2)."* Emitting a per-run
GPU-hours figure and multiplying it by the P2-06 grid **is that rule executing** — so
landing the product as the campaign bound would freeze an ADR-governed number without the
sign-off. D7 also declares the freeze point to be "first measurement … at **P1**", but no
ADR amendment ever recorded one, which makes this measurement the de-facto first freeze
(or a re-freeze — D7 makes that an ADR-sign-off change either way).

This module therefore follows the P1-14 precedent (``reports/p1/rinalmo_throughput.json``):
the extrapolation is emitted under ``illustrative_extrapolation`` with ``advisory_only``,
``binding: False`` and a ``binding_gate`` naming who may freeze it. The report states the
cost; it does not enact the bound. ``validate_report`` makes a report that claims otherwise
**invalid**, so the honesty is enforced rather than promised (the P1-15 ``forward_verified``
precedent).

**Scaling-efficiency semantics.** Under DDP each rank runs ``batch_size`` windows per step,
so per-GPU throughput is ``batch_size / step_seconds`` and aggregate throughput is
``world_size ×`` that. Efficiency at ``N`` is per-GPU throughput retained against the
``N=1`` point — the all-reduce is a per-step tax, so a value < 1 is expected and is exactly
the quantity PRD §10.3's "DDP×8 for throughput" claim rests on. ADR-0003 D6 pins
``Wall-clock = GPU-hours / G`` linearity **for the scan estimator only**; D7 pins no scaling
law, so nothing here assumes 8× from a 2-GPU point — the efficiency is measured at each
world size actually run, and ``extrapolate`` refuses a world size it has no point for.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"
STEP = "P2-05"
GENERATED_BY = "tbox_finder.train.sizing"
ADR = "ADR-0003"
PRD = "§10.3, §11, §15"

#: PRD §10.3 pins the card: "Fine-tuning footprint on 8× RTX A4000 (**16 GB**, Ampere
#: sm_86)". The budget number is the PRD's 16 GB, NOT the card's usable
#: ``total_vram_gib`` (~15.6 GiB) — they are different claims and the report carries both.
PRD_VRAM_BUDGET_GIB = 16.0

#: PRD §10.3 pins "DDP×8 for throughput" — the world size the footprint claim is about.
PRD_TARGET_WORLD_SIZE = 8

#: Steps discarded before timing. P2-04 refused to quote step times precisely because
#: single-step timings were warmup noise (allocator growth, cuDNN autotune, lazy kernel
#: load). Discarding a fixed prefix is what makes the remainder steady-state.
DEFAULT_WARMUP_STEPS = 5

#: Who is allowed to turn this measurement into a bound. Not this step.
BINDING_GATE = (
    "ADR-0003 D7 training/sweep GPU-hour budget freeze — needs ADR sign-off (§7 item 2). "
    "Not frozen by P2-05; this report is the measurement the freeze would cite."
)


# ═════════════════════════════════════════════════════════════════════════════
# Small numeric helpers (stdlib only — the aggregator runs in bare CI)
# ═════════════════════════════════════════════════════════════════════════════
def _is_real(v: Any) -> bool:
    """True for a non-bool finite real. ``isinstance(True, int)`` is True — reject it."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _is_pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy 'linear'/type-7). ``None`` if empty.

    Re-derived in stdlib rather than importing numpy's, so the golden expectations are
    pinned by an independent path (the P2-02/P2-03 precedent: grading a function against
    the library that produced it is a tautology).
    """
    xs = sorted(float(x) for x in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def steady_state(
    step_seconds: Sequence[float], *, warmup: int = DEFAULT_WARMUP_STEPS
) -> dict[str, Any]:
    """Summarise per-step timings after discarding ``warmup`` leading steps.

    Returns ``n_measured == 0`` and ``None`` statistics when the point has no steps left
    after the discard — a point that ran too few steps to be timed must read as *not
    measured*, never as a fast one. Callers must check ``n_measured``.
    """
    xs = [float(x) for x in step_seconds]
    kept = xs[max(0, int(warmup)) :]
    if not kept:
        return {
            "n_steps_total": len(xs),
            "warmup_discarded": min(max(0, int(warmup)), len(xs)),
            "n_measured": 0,
            "seconds_mean": None,
            "seconds_median": None,
            "seconds_p95": None,
            "seconds_min": None,
        }
    return {
        "n_steps_total": len(xs),
        "warmup_discarded": min(max(0, int(warmup)), len(xs)),
        "n_measured": len(kept),
        "seconds_mean": sum(kept) / len(kept),
        "seconds_median": percentile(kept, 50),
        "seconds_p95": percentile(kept, 95),
        "seconds_min": min(kept),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Point extraction
# ═════════════════════════════════════════════════════════════════════════════
def point_key(world_size: int, batch_size: int, gradient_checkpointing: bool) -> str:
    """Stable identity for a configuration point (also the per-point report filename)."""
    return f"ws{int(world_size)}_bs{int(batch_size)}_ckpt{int(bool(gradient_checkpointing))}"


def extract_point(
    report: Mapping[str, Any], *, warmup: int = DEFAULT_WARMUP_STEPS
) -> dict[str, Any]:
    """Reduce one ``train_stage1`` report to a sizing point.

    Reads only what ``train_stage1`` already records — nothing here is asserted by the
    sbatch, because a point that took its ``batch_size`` from the filename rather than the
    run would happily mislabel a run that Hydra overrode differently.
    """
    if not isinstance(report, Mapping):
        raise TypeError(f"report must be a mapping, got {type(report).__name__}")
    steps = report.get("steps")
    steps = steps if isinstance(steps, Mapping) else {}
    diagnostics = report.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    config = diagnostics.get("config")
    config = config if isinstance(config, Mapping) else {}
    ckpt = report.get("gradient_checkpointing")
    ckpt = ckpt if isinstance(ckpt, Mapping) else {}
    hardware = report.get("hardware")
    hardware = hardware if isinstance(hardware, Mapping) else {}

    world_size = steps.get("world_size")
    batch_size = config.get("batch_size")
    requested = ckpt.get("requested")
    step_seconds = steps.get("step_seconds")
    wait_seconds = steps.get("batch_wait_seconds")

    if not _is_pos_int(world_size):
        raise ValueError(f"steps.world_size must be a positive int, got {world_size!r}")
    if not _is_pos_int(batch_size):
        raise ValueError(
            f"diagnostics.config.batch_size must be a positive int, got {batch_size!r}"
        )
    if not isinstance(requested, bool):
        raise ValueError(f"gradient_checkpointing.requested must be a bool, got {requested!r}")
    if not isinstance(step_seconds, list) or not step_seconds:
        raise ValueError(
            "steps.step_seconds missing or empty — this report predates the P2-05 "
            "instrumentation, or the run timed no steps; it cannot be a sizing point."
        )

    timing = steady_state(step_seconds, warmup=warmup)
    wait = (
        steady_state(wait_seconds, warmup=warmup)
        if isinstance(wait_seconds, list) and wait_seconds
        else None
    )

    median = timing["seconds_median"]
    # A point with no steady-state steps yields no rate. `None`, never 0.0 or a guess —
    # a fabricated rate is indistinguishable from a measured one downstream.
    wps_per_gpu = (batch_size / median) if (_is_real(median) and median > 0) else None

    peak = report.get("peak_vram_gib")
    total = hardware.get("total_vram_gib")
    return {
        "key": point_key(world_size, batch_size, requested),
        "world_size": int(world_size),
        "batch_size": int(batch_size),
        "gradient_checkpointing": bool(requested),
        "timing": timing,
        "batch_wait": wait,
        "windows_per_sec_per_gpu": wps_per_gpu,
        "windows_per_sec_aggregate": (
            wps_per_gpu * int(world_size) if wps_per_gpu is not None else None
        ),
        "peak_vram_gib": float(peak) if _is_real(peak) else None,
        # PRD §10.3's budget is 16 GB; the card's usable total is a different number and
        # both margins matter. P1-16's lesson: on a 15.6 GiB card an over-16 GiB peak is
        # UNREACHABLE (the allocator OOMs first), so the "< 16 GB" boolean is nearly
        # unfailable and the informative fact is the margin. Report the margin.
        "vram_margin_gib_vs_prd_budget": (
            PRD_VRAM_BUDGET_GIB - float(peak) if _is_real(peak) else None
        ),
        "vram_margin_gib_vs_card": (
            float(total) - float(peak) if _is_real(peak) and _is_real(total) else None
        ),
        "vram_headroom_frac_of_card": (
            1.0 - (float(peak) / float(total))
            if _is_real(peak) and _is_real(total) and float(total) > 0
            else None
        ),
        "device_name": hardware.get("device_name"),
        "total_vram_gib": float(total) if _is_real(total) else None,
        "grads_finite": report.get("grads_finite"),
        "losses_finite": all(_is_real(x) for x in (steps.get("losses") or [])),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Derived analyses
# ═════════════════════════════════════════════════════════════════════════════
def scaling_efficiency(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """DDP scaling efficiency vs the world_size==1 point, at matched batch + checkpointing.

    Matched-config by construction: comparing ws=8 to a ws=1 point that ran a different
    batch size or checkpointing setting would attribute a batch effect to the all-reduce.
    Returns ``{}`` when no ws=1 baseline exists at that config — an unbaselined efficiency
    is not reported as 1.0.
    """
    out: dict[str, Any] = {}
    by_cfg: dict[tuple[int, bool], dict[int, Mapping[str, Any]]] = {}
    for p in points:
        cfg = (int(p["batch_size"]), bool(p["gradient_checkpointing"]))
        by_cfg.setdefault(cfg, {})[int(p["world_size"])] = p

    for (batch, ckpt), by_ws in sorted(by_cfg.items()):
        base = by_ws.get(1)
        if base is None or base.get("windows_per_sec_per_gpu") is None:
            continue
        base_rate = float(base["windows_per_sec_per_gpu"])
        if base_rate <= 0:
            continue
        series: dict[str, Any] = {}
        for ws, p in sorted(by_ws.items()):
            rate = p.get("windows_per_sec_per_gpu")
            if rate is None:
                continue
            series[str(ws)] = {
                "world_size": ws,
                "windows_per_sec_per_gpu": float(rate),
                "windows_per_sec_aggregate": float(rate) * ws,
                # Per-GPU throughput retained vs the 1-GPU baseline. The all-reduce is a
                # per-step tax, so < 1.0 is expected; > 1.0 would be noise or a mislabelled
                # point, not a superlinear speedup.
                "efficiency_vs_ws1": float(rate) / base_rate,
                "speedup_vs_ws1": (float(rate) * ws) / base_rate,
            }
        if len(series) > 1:
            out[f"batch{batch}_ckpt{int(ckpt)}"] = {
                "batch_size": batch,
                "gradient_checkpointing": ckpt,
                "baseline_windows_per_sec_per_gpu": base_rate,
                "by_world_size": series,
            }
    return out


def checkpointing_cost(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Throughput + VRAM cost of gradient checkpointing at matched (world_size, batch).

    This is the half P2-04 explicitly did not measure: it quantified the VRAM saving
    (batch 8: 4.424 → 0.961 GiB on a laptop 4060) but refused to quote step times because
    each batch ran in its own process and single-step timings were warmup noise. Here both
    sides are steady-state on the real A4000.
    """
    out: dict[str, Any] = {}
    by_cfg: dict[tuple[int, int], dict[bool, Mapping[str, Any]]] = {}
    for p in points:
        by_cfg.setdefault((int(p["world_size"]), int(p["batch_size"])), {})[
            bool(p["gradient_checkpointing"])
        ] = p

    for (ws, batch), by_ckpt in sorted(by_cfg.items()):
        on, off = by_ckpt.get(True), by_ckpt.get(False)
        if on is None or off is None:
            continue
        r_on, r_off = on.get("windows_per_sec_per_gpu"), off.get("windows_per_sec_per_gpu")
        v_on, v_off = on.get("peak_vram_gib"), off.get("peak_vram_gib")
        entry: dict[str, Any] = {"world_size": ws, "batch_size": batch}
        if r_on is not None and r_off is not None and float(r_on) > 0:
            entry["windows_per_sec_per_gpu_on"] = float(r_on)
            entry["windows_per_sec_per_gpu_off"] = float(r_off)
            # >1 ⇒ checkpointing is slower (recompute), the expected direction.
            entry["throughput_cost_factor"] = float(r_off) / float(r_on)
            entry["throughput_retained_frac"] = (
                float(r_on) / float(r_off) if float(r_off) > 0 else None
            )
        if v_on is not None and v_off is not None and float(v_on) > 0:
            entry["peak_vram_gib_on"] = float(v_on)
            entry["peak_vram_gib_off"] = float(v_off)
            entry["vram_saving_factor"] = float(v_off) / float(v_on)
        if len(entry) > 2:
            out[f"ws{ws}_batch{batch}"] = entry
    return out


def largest_fitting_batch(
    points: Sequence[Mapping[str, Any]], *, world_size: int = 1
) -> int | None:
    """Largest batch with a completed point at ``world_size``. ``None`` if none ran.

    "Fitting" means a point that ran to completion and recorded a peak — an OOM point
    produces no report at all, so absence is the evidence (the sbatch does not write a
    report on a failed point).
    """
    sizes = [
        int(p["batch_size"])
        for p in points
        if int(p["world_size"]) == int(world_size) and p.get("peak_vram_gib") is not None
    ]
    return max(sizes) if sizes else None


def extrapolate(
    *,
    windows_per_sec_per_gpu: float,
    windows_per_epoch: int,
    epochs: int,
    world_size: int,
) -> dict[str, Any]:
    """Illustrative full-run cost. **Not** a budget — see the module docstring / D7.

    ``GPU-hours = total_windows / rate / 3600`` and ``wall-clock = GPU-hours / G``, the
    ADR-0003 D6 estimator's shape. D6 pins that linearity for the *scan*; here the same
    algebra is applied to training with the DDP efficiency **already inside**
    ``windows_per_sec_per_gpu`` (it is measured at the world size being extrapolated), so
    no unpinned scaling law is assumed.
    """
    if not (_is_real(windows_per_sec_per_gpu) and float(windows_per_sec_per_gpu) > 0):
        raise ValueError(
            f"windows_per_sec_per_gpu must be a positive real, got {windows_per_sec_per_gpu!r}"
        )
    if not _is_pos_int(windows_per_epoch):
        raise ValueError(f"windows_per_epoch must be a positive int, got {windows_per_epoch!r}")
    if not _is_pos_int(epochs):
        raise ValueError(f"epochs must be a positive int, got {epochs!r}")
    if not _is_pos_int(world_size):
        raise ValueError(f"world_size must be a positive int, got {world_size!r}")

    total_windows = float(windows_per_epoch) * float(epochs)
    aggregate_rate = float(windows_per_sec_per_gpu) * float(world_size)
    wall_hours = total_windows / aggregate_rate / 3600.0
    return {
        "windows_per_epoch": int(windows_per_epoch),
        "epochs": int(epochs),
        "total_windows": total_windows,
        "world_size": int(world_size),
        "windows_per_sec_per_gpu": float(windows_per_sec_per_gpu),
        "windows_per_sec_aggregate": aggregate_rate,
        "wall_clock_hours": wall_hours,
        "gpu_hours": wall_hours * float(world_size),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════
def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """Re-derive every gate clause from the recorded evidence.

    ``overall_pass = all(clauses)`` catches a clause flipped FALSE but structurally cannot
    catch one fabricated TRUE — an all-true gate is self-consistent whatever the evidence
    says (the P1-15/P1-16 lesson, and the reason builder and validator share this one
    function rather than each computing its own truth).
    """
    if not isinstance(report, Mapping):
        return {}
    points = report.get("points")
    points = points if isinstance(points, list) else []
    extrap = report.get("illustrative_extrapolation")
    extrap = extrap if isinstance(extrap, Mapping) else {}

    measured = [p for p in points if isinstance(p, Mapping)]
    vram_vals = [p.get("peak_vram_gib") for p in measured if _is_real(p.get("peak_vram_gib"))]
    rates = [
        p.get("windows_per_sec_per_gpu")
        for p in measured
        if _is_real(p.get("windows_per_sec_per_gpu"))
    ]

    return {
        # The imp.md gate. Reported alongside the margin, never instead of it: on a
        # 15.6 GiB card a >16 GiB peak is unreachable, so a bare True here is close to
        # unfailable and means far less than `min_vram_margin_gib_vs_prd_budget` (P1-16).
        "vram_under_prd_budget": bool(vram_vals) and max(vram_vals) < PRD_VRAM_BUDGET_GIB,
        "points_measured": len(measured) > 0,
        "throughput_measured": len(rates) > 0,
        "all_points_grads_finite": bool(measured)
        and all(p.get("grads_finite") is True for p in measured),
        "all_points_losses_finite": bool(measured)
        and all(p.get("losses_finite") is True for p in measured),
        # The honesty invariants, enforced not promised: this step measures, it does not
        # freeze ADR-0003 D7's budget and it is not a science result.
        "extrapolation_marked_advisory": extrap.get("advisory_only") is True
        and extrap.get("binding") is False
        and isinstance(extrap.get("binding_gate"), str)
        and bool(extrap.get("binding_gate")),
        "not_a_science_result": report.get("is_science") is False,
        "budget_not_frozen": report.get("freezes_adr0003_d7_budget") is False,
    }


def build_report(
    *,
    points: Sequence[Mapping[str, Any]],
    windows_per_epoch: int | None,
    epochs: int,
    git_sha: str | None = None,
    warmup: int = DEFAULT_WARMUP_STEPS,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the P2-05 sizing report. Clauses are re-derived, never asserted."""
    pts = [dict(p) for p in points]
    scaling = scaling_efficiency(pts)
    ckpt_cost = checkpointing_cost(pts)

    vram_margins = [
        p["vram_margin_gib_vs_prd_budget"]
        for p in pts
        if _is_real(p.get("vram_margin_gib_vs_prd_budget"))
    ]
    peaks = [p["peak_vram_gib"] for p in pts if _is_real(p.get("peak_vram_gib"))]

    # Extrapolate from the PRD §10.3 target world size when a point exists for it. Never
    # from a smaller world size scaled up: D7 pins no scaling law, and inventing one is how
    # a budget acquires a number nobody measured.
    target = [
        p
        for p in pts
        if int(p["world_size"]) == PRD_TARGET_WORLD_SIZE
        and p.get("windows_per_sec_per_gpu") is not None
    ]
    extrap: dict[str, Any] = {
        "advisory_only": True,
        "binding": False,
        "binding_gate": BINDING_GATE,
        "assumptions": [
            "Rate is the steady-state median at the quoted world size — the DDP all-reduce "
            "cost is measured at that world size, not modelled from a smaller one.",
            "windows_per_epoch is len(Stage1WindowDataset) over the FULL nested_train fold "
            "(not the sizing slice), computed from the same data config the points ran.",
            "`step_seconds` includes the per-step full-parameter grad-finiteness scan "
            "(train_stage1.py), which syncs per parameter per step. Its own comment scopes "
            "it to 'smoke scale'; nobody has decided whether it stays at full-run scale, so "
            "these hours are an UPPER bound on that axis. P2-06 owns the decision.",
            "fp32: train_stage1 exposes no precision knob and ADR-0002 A7's determinism "
            "contract disables TF32/cudnn autotune, so this is NOT the bf16/TF32-native "
            "throughput PRD §10.3 names as a property of the sm_86 card.",
        ],
    }
    if target and _is_pos_int(windows_per_epoch):
        best = max(target, key=lambda p: float(p["windows_per_sec_per_gpu"]))
        extrap["basis_point"] = best["key"]
        extrap.update(
            extrapolate(
                windows_per_sec_per_gpu=float(best["windows_per_sec_per_gpu"]),
                windows_per_epoch=int(windows_per_epoch),
                epochs=int(epochs),
                world_size=PRD_TARGET_WORLD_SIZE,
            )
        )
    else:
        extrap["basis_point"] = None
        extrap["unavailable_reason"] = (
            f"no completed point at the PRD §10.3 target world_size={PRD_TARGET_WORLD_SIZE}"
            if not target
            else "windows_per_epoch unavailable"
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "prd": PRD,
        # A footprint measurement is not a T-box result. The model here is training on a
        # seeded slice for a handful of steps; nothing about it is a scientific claim.
        "is_science": False,
        "gate4_graded": False,
        # The single most important field in this file. See the module docstring.
        "freezes_adr0003_d7_budget": False,
        "prd_vram_budget_gib": PRD_VRAM_BUDGET_GIB,
        "prd_target_world_size": PRD_TARGET_WORLD_SIZE,
        "warmup_steps_discarded": int(warmup),
        "n_points": len(pts),
        "points": pts,
        "device_name": next((p.get("device_name") for p in pts if p.get("device_name")), None),
        "total_vram_gib": next(
            (p.get("total_vram_gib") for p in pts if p.get("total_vram_gib") is not None), None
        ),
        "max_peak_vram_gib": max(peaks) if peaks else None,
        "min_vram_margin_gib_vs_prd_budget": min(vram_margins) if vram_margins else None,
        "largest_fitting_batch_ws1": largest_fitting_batch(pts, world_size=1),
        "ddp_scaling": scaling,
        "gradient_checkpointing_cost": ckpt_cost,
        "grad_finiteness_scan_in_step_seconds": True,
        "illustrative_extrapolation": extrap,
        "provenance": {
            "git_sha": git_sha,
            "windows_per_epoch_full_fold": (
                int(windows_per_epoch) if _is_pos_int(windows_per_epoch) else None
            ),
            "epochs_assumed": int(epochs),
        },
        "notes": list(notes),
    }
    clauses = derive_clauses(report)
    report["gate"] = {**clauses, "overall_pass": all(clauses.values())}
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of problems with a P2-05 sizing report; empty ⇒ valid. Fails closed.

    Total by construction — never raises on a malformed report, it reports.
    """
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return [f"report must be a mapping, got {type(report).__name__}"]

    for key in ("schema_version", "step", "generated_by", "adr", "prd"):
        if not isinstance(report.get(key), str) or not report.get(key):
            problems.append(f"{key}: missing or not a non-empty string")
    if report.get("step") != STEP:
        problems.append(f"step: expected {STEP!r}, got {report.get('step')!r}")

    if report.get("is_science") is not False:
        problems.append("is_science: must be exactly False (a footprint measurement, not a result)")
    if report.get("gate4_graded") is not False:
        problems.append("gate4_graded: must be exactly False (GATE-4 is P2-14)")
    # ADR-0003 D7: a sizing report that claims to freeze the budget is invalid, full stop.
    # The claim is structural, so make the structure refuse it (P1-15's `forward_verified`
    # precedent — a validator that makes the over-claim impossible beats a promise not to).
    if report.get("freezes_adr0003_d7_budget") is not False:
        problems.append(
            "freezes_adr0003_d7_budget: must be exactly False — D7 makes the training/sweep "
            "budget number an ADR-sign-off change (§7 item 2); P2-05 measures, it does not freeze."
        )

    extrap = report.get("illustrative_extrapolation")
    if not isinstance(extrap, Mapping):
        problems.append("illustrative_extrapolation: missing or not a mapping")
    else:
        if extrap.get("advisory_only") is not True:
            problems.append("illustrative_extrapolation.advisory_only: must be exactly True")
        if extrap.get("binding") is not False:
            problems.append("illustrative_extrapolation.binding: must be exactly False")
        if not isinstance(extrap.get("binding_gate"), str) or not extrap.get("binding_gate"):
            problems.append("illustrative_extrapolation.binding_gate: missing gate attribution")
        for key in ("wall_clock_hours", "gpu_hours"):
            if key in extrap and not (_is_real(extrap[key]) and float(extrap[key]) > 0):
                problems.append(f"illustrative_extrapolation.{key}: must be a positive finite real")

    points = report.get("points")
    if not isinstance(points, list):
        problems.append("points: missing or not a list")
    else:
        for i, p in enumerate(points):
            if not isinstance(p, Mapping):
                problems.append(f"points[{i}]: not a mapping")
                continue
            for key in ("world_size", "batch_size"):
                if not _is_pos_int(p.get(key)):
                    problems.append(f"points[{i}].{key}: must be a positive int")
            if not isinstance(p.get("gradient_checkpointing"), bool):
                problems.append(f"points[{i}].gradient_checkpointing: must be a bool")
            rate = p.get("windows_per_sec_per_gpu")
            if rate is not None and not (_is_real(rate) and float(rate) > 0):
                problems.append(f"points[{i}].windows_per_sec_per_gpu: must be positive or None")
        if isinstance(report.get("n_points"), int) and report["n_points"] != len(points):
            problems.append(f"n_points: {report['n_points']} != len(points) {len(points)}")

    derived = derive_clauses(report)
    stored = report.get("gate")
    if not isinstance(stored, Mapping):
        problems.append("gate: missing or not a mapping")
    else:
        for name, value in derived.items():
            if name not in stored:
                problems.append(f"gate.{name}: missing")
            elif stored[name] is not value:
                problems.append(
                    f"gate.{name}: stored {stored[name]!r} != re-derived {value!r} "
                    "(a clause must follow from the recorded evidence, never be asserted)"
                )
        expected_overall = all(derived.values())
        if stored.get("overall_pass") is not expected_overall:
            problems.append(
                f"gate.overall_pass: stored {stored.get('overall_pass')!r} != "
                f"re-derived {expected_overall!r}"
            )
    return problems


# ═════════════════════════════════════════════════════════════════════════════
# Data property + CLI
# ═════════════════════════════════════════════════════════════════════════════
def full_fold_windows_per_epoch(data_config: Mapping[str, Any]) -> int:
    """``len(Stage1WindowDataset)`` over the FULL ``nested_train`` fold — the epoch size.

    The extrapolation denominator. Taken from the real datamodule over the real corpus,
    never from the sizing slice (which is ``max_records``-truncated) — a per-epoch window
    count read off a 512-record slice would under-state the full run by ~16×.

    ``data_config`` comes from a point report's ``class_counts_scope.data_config``, so the
    geometry that sizes the run is the same geometry the points were timed under.
    """
    from dataclasses import fields as dataclass_fields

    from tbox_finder.data.window_dataset import (
        Stage1DataConfig,
        Stage1WindowDataset,
        load_corpus_records,
    )

    # Filter to the dataclass's own fields: the recorded `data_config` is whatever
    # `asdict()` produced at the time, and a report from a future config gaining a field
    # must not crash the aggregator on an unexpected keyword.
    known = {f.name for f in dataclass_fields(Stage1DataConfig)}
    cfg = Stage1DataConfig(**{k: v for k, v in data_config.items() if k in known})
    records, _ = load_corpus_records(training_fold_only=True, window=cfg.window_nt)
    return len(Stage1WindowDataset(records, config=cfg))


def load_points(
    paths: Sequence[Path], *, warmup: int = DEFAULT_WARMUP_STEPS
) -> list[dict[str, Any]]:
    """Load + reduce per-point reports. A path that is not a valid point is fatal."""
    out: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        try:
            out.append(extract_point(report, warmup=warmup))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: not a usable sizing point — {exc}") from exc
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate P2-05 Stage-1 sizing points.")
    parser.add_argument(
        "--points", nargs="+", required=True, type=Path, help="per-point JSON reports"
    )
    parser.add_argument("--out", required=True, type=Path, help="aggregated report path")
    parser.add_argument("--epochs", type=int, required=True, help="epochs the full run assumes")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument(
        "--windows-per-epoch",
        type=int,
        default=None,
        help="full-fold epoch size; computed from the datamodule when omitted",
    )
    args = parser.parse_args(argv)

    points = load_points(args.points, warmup=args.warmup)
    if not points:
        raise SystemExit("no sizing points supplied")

    windows_per_epoch = args.windows_per_epoch
    if windows_per_epoch is None:
        raw = json.loads(Path(args.points[0]).read_text(encoding="utf-8"))
        scope = raw.get("class_counts_scope") or {}
        data_config = scope.get("data_config")
        if not isinstance(data_config, Mapping):
            raise SystemExit(
                "cannot compute windows_per_epoch: point report carries no "
                "class_counts_scope.data_config; pass --windows-per-epoch explicitly"
            )
        windows_per_epoch = full_fold_windows_per_epoch(data_config)

    git_sha = None
    try:  # provenance is best-effort here; the points carry the authoritative SHA
        import subprocess  # noqa: PLC0415

        git_sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
    except OSError:
        git_sha = None

    report = build_report(
        points=points,
        windows_per_epoch=windows_per_epoch,
        epochs=args.epochs,
        git_sha=git_sha,
        warmup=args.warmup,
    )
    problems = validate_report(report)
    if problems:
        raise SystemExit(
            "P2-05 sizing report failed its own validator:\n  " + "\n  ".join(problems)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate = report["gate"]
    print(f"SIZING_REPORT {args.out} overall_pass={gate['overall_pass']}")
    print(
        f"  max_peak_vram_gib={report['max_peak_vram_gib']} "
        f"min_margin_vs_16GB={report['min_vram_margin_gib_vs_prd_budget']}"
    )
    # A failing gate must not be a silent zero exit — but the report is written first, so
    # the evidence survives the failure (the P2-04 precedent).
    return 0 if gate["overall_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
