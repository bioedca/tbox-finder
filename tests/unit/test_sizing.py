"""P2-05 — unit tests for the Stage-1 sizing aggregator.

Torch-free: the aggregator is pure arithmetic over recorded JSON, so the whole thing runs
in bare CI. That is deliberate — the science-honesty invariants (ADR-0003 D7: this report
must not freeze the budget) are the part most worth guarding, and guarding them must not
depend on a GPU being present.

`overall_pass = all(clauses)` catches a clause flipped FALSE but structurally cannot catch
one fabricated TRUE, which is why the validator re-derives and why these tests sabotage
rather than read.

**Scope of the anti-tautology coverage, stated rather than claimed.** An earlier version of
this docstring asserted that *every* gate clause carried a test proving it bites. That was
false — a blanket claim about my own tests that a reviewer checked and disproved, which is
the §10.2 failure mode one level up from the code. What is actually true: the clauses whose
falsity is reachable from a report `build_report` can emit are exercised against evidence
constructed to violate them (`vram_under_prd_budget`, `all_points_grads_finite`,
`ddp_target_world_size_measured`, `extrapolation_basis_is_prd_pinned_config`). The
invariant clauses — `not_a_science_result`, `budget_not_frozen`,
`extrapolation_marked_advisory` — are **structurally constant** for any report the builder
produces, so their tests necessarily tamper with the report to prove the *validator*
rejects the lie; that is tamper-detection, not bite-detection, and it is the strongest
statement available for a field the builder hard-codes. Both kinds are needed; neither is
sufficient alone.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tbox_finder.train.sizing import (
    PRD_TARGET_WORLD_SIZE,
    PRD_VRAM_BUDGET_GIB,
    build_report,
    checkpointing_cost,
    derive_clauses,
    extract_point,
    extrapolate,
    largest_fitting_batch,
    percentile,
    point_key,
    scaling_efficiency,
    steady_state,
    validate_report,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixture builders — realistic point reports, NOT degenerate ones
# ═════════════════════════════════════════════════════════════════════════════
def make_point_report(
    *,
    world_size=1,
    batch_size=8,
    ckpt=True,
    step_seconds=None,
    batch_wait_seconds=None,
    peak_vram_gib=4.0,
    total_vram_gib=15.602,
    grads_finite=True,
    losses=None,
    data_config=None,
):
    """A minimal but structurally faithful `train_stage1` report.

    The timings deliberately VARY across steps (and the warmup prefix is deliberately
    slower than steady state, as a real run's is) — a constant fixture could not
    distinguish a working warmup discard from one that discards nothing, which is exactly
    the class of degenerate fixture that let a whole suite compare uniform to uniform.
    """
    if step_seconds is None:
        # 5 slow warmup steps, then 5 steady ones whose median is 0.20 but whose MEAN is
        # 0.256 — deliberately ASYMMETRIC (a right tail, as real step times have).
        # The earlier fixture was [0.18, 0.19, 0.20, 0.21, 0.22]: symmetric, so mean ==
        # median == 0.20 and swapping `percentile(kept, 50)` for the mean passed all 59
        # tests. A fixture that cannot distinguish the statistic under test from a wrong
        # one is the P2-03 degenerate-fixture lesson wearing a new costume.
        step_seconds = [1.0, 0.9, 0.8, 0.7, 0.6] + [0.18, 0.19, 0.20, 0.21, 0.50]
    if losses is None:
        losses = [1.6 - 0.001 * i for i in range(len(step_seconds))]
    steps = {
        "n_steps": len(losses),
        "losses": list(losses),
        "world_size": world_size,
        "step_seconds": list(step_seconds),
    }
    if batch_wait_seconds is not None:
        steps["batch_wait_seconds"] = list(batch_wait_seconds)
    return {
        "schema_version": "1",
        "step": "P2-04",
        "steps": steps,
        "gradient_checkpointing": {"requested": ckpt, "n_blocks": 16, "n_blocks_wrapped": 16},
        "diagnostics": {"config": {"batch_size": batch_size, "seed": 42}},
        "hardware": {
            "device": f"cuda:{world_size - 1}",
            "device_name": "NVIDIA RTX A4000",
            "total_vram_gib": total_vram_gib,
        },
        "peak_vram_gib": peak_vram_gib,
        "grads_finite": grads_finite,
        "class_counts_scope": {"data_config": data_config or {"window_nt": 1024, "stride_nt": 512}},
    }


def make_points(specs):
    """specs: iterable of dicts passed to make_point_report → extracted sizing points."""
    return [extract_point(make_point_report(**s)) for s in specs]


# ═════════════════════════════════════════════════════════════════════════════
# percentile — pinned against an INDEPENDENT closed form, not against numpy
# ═════════════════════════════════════════════════════════════════════════════
def test_percentile_matches_hand_derivation():
    # type-7: pos = (n-1)*q/100. n=5, q=50 → pos=2 → xs[2] == 3.0 exactly.
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    # n=4, q=50 → pos=1.5 → 2.0*(2-1.5) + 3.0*(1.5-1) = 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5, abs=1e-12, rel=0)
    # n=5, q=95 → pos=3.8 → xs[3]*(4-3.8) + xs[4]*(3.8-3) = 4*0.2 + 5*0.8 = 4.8
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == pytest.approx(4.8, abs=1e-12, rel=0)


def test_percentile_is_order_insensitive_and_handles_singletons():
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0
    assert percentile([7.5], 50) == 7.5
    assert percentile([], 50) is None


def test_percentile_would_catch_a_wrong_interpolation():
    """Anti-tautology: the n=4 median is the one case a naive mid-index gets wrong."""
    xs = [1.0, 2.0, 3.0, 4.0]
    naive_mid_index = sorted(xs)[len(xs) // 2]  # == 3.0, the WRONG answer
    assert percentile(xs, 50) != naive_mid_index


# ═════════════════════════════════════════════════════════════════════════════
# steady_state
# ═════════════════════════════════════════════════════════════════════════════
def test_steady_state_discards_warmup():
    xs = [9.0, 9.0, 9.0, 9.0, 9.0, 0.1, 0.2, 0.3]
    s = steady_state(xs, warmup=5)
    assert s["n_steps_total"] == 8
    assert s["warmup_discarded"] == 5
    assert s["n_measured"] == 3
    assert s["seconds_median"] == pytest.approx(0.2, abs=1e-12, rel=0)
    assert s["seconds_min"] == pytest.approx(0.1, abs=1e-12, rel=0)


def test_steady_state_warmup_discard_actually_bites():
    """The warmup steps are 45x the steady ones — not discarding them is unmissable."""
    xs = [9.0] * 5 + [0.2] * 3
    discarded = steady_state(xs, warmup=5)["seconds_median"]
    not_discarded = steady_state(xs, warmup=0)["seconds_median"]
    assert discarded == pytest.approx(0.2, abs=1e-12, rel=0)
    assert not_discarded > 1.0  # would silently triple every GPU-hour estimate


def test_steady_state_reports_not_measured_rather_than_a_fast_point():
    """A point with fewer steps than the warmup must read as unmeasured, never as fast."""
    s = steady_state([0.1, 0.2], warmup=5)
    assert s["n_measured"] == 0
    assert s["seconds_median"] is None
    assert s["seconds_mean"] is None


def test_steady_state_empty():
    s = steady_state([], warmup=5)
    assert s["n_measured"] == 0 and s["seconds_median"] is None


# ═════════════════════════════════════════════════════════════════════════════
# extract_point
# ═════════════════════════════════════════════════════════════════════════════
def test_extract_point_reads_identity_from_the_run_not_the_caller():
    p = extract_point(make_point_report(world_size=4, batch_size=16, ckpt=False))
    assert (p["world_size"], p["batch_size"], p["gradient_checkpointing"]) == (4, 16, False)
    assert p["key"] == point_key(4, 16, False) == "ws4_bs16_ckpt0"


def test_extract_point_computes_rate_from_steady_state_median():
    # steady median = 0.20 s, batch 8 → 40 windows/sec/GPU; aggregate = x world_size.
    p = extract_point(make_point_report(world_size=2, batch_size=8))
    assert p["windows_per_sec_per_gpu"] == pytest.approx(40.0, abs=1e-9, rel=0)
    assert p["windows_per_sec_aggregate"] == pytest.approx(80.0, abs=1e-9, rel=0)


def test_extract_point_vram_margins_are_against_both_the_prd_budget_and_the_card():
    p = extract_point(make_point_report(peak_vram_gib=4.0, total_vram_gib=15.602))
    assert p["vram_margin_gib_vs_prd_budget"] == pytest.approx(12.0, abs=1e-9, rel=0)
    assert p["vram_margin_gib_vs_card"] == pytest.approx(11.602, abs=1e-9, rel=0)
    assert p["vram_headroom_frac_of_card"] == pytest.approx(1 - 4.0 / 15.602, abs=1e-9, rel=0)


def test_extract_point_rejects_a_report_with_no_timing():
    """A pre-P2-05 report (P2-04's committed artifact) is not a sizing point."""
    r = make_point_report()
    del r["steps"]["step_seconds"]
    with pytest.raises(ValueError, match="step_seconds"):
        extract_point(r)


def test_extract_point_rejects_bool_world_size():
    """isinstance(True, int) is True — a bool must not read as world_size 1."""
    r = make_point_report()
    r["steps"]["world_size"] = True
    with pytest.raises(ValueError, match="world_size"):
        extract_point(r)


def test_extract_point_rejects_non_bool_checkpointing_flag():
    r = make_point_report()
    r["gradient_checkpointing"]["requested"] = 1  # truthy, but not a bool
    with pytest.raises(ValueError, match="requested"):
        extract_point(r)


def test_extract_point_unmeasured_point_yields_no_rate_rather_than_zero():
    r = make_point_report(step_seconds=[0.5, 0.5])  # fewer steps than the warmup
    p = extract_point(r)
    assert p["windows_per_sec_per_gpu"] is None
    assert p["windows_per_sec_aggregate"] is None


# ═════════════════════════════════════════════════════════════════════════════
# scaling_efficiency
# ═════════════════════════════════════════════════════════════════════════════
def _rate_steps(seconds):
    """A point whose steady-state MEDIAN is exactly `seconds` — and whose mean is not.

    Asymmetric on purpose (see make_point_report): a constant steady block would make
    mean == median, so every rate assertion would pass against a mean-based implementation
    and the choice of statistic would be untested.
    """
    return [9.0] * 5 + [seconds * 0.9, seconds * 0.95, seconds, seconds * 1.05, seconds * 3.0]


def test_scaling_efficiency_against_a_hand_computed_series():
    pts = make_points(
        [
            {"world_size": 1, "batch_size": 8, "step_seconds": _rate_steps(0.20)},  # 40/s
            {"world_size": 2, "batch_size": 8, "step_seconds": _rate_steps(0.25)},  # 32/s
            {"world_size": 8, "batch_size": 8, "step_seconds": _rate_steps(0.40)},  # 20/s
        ]
    )
    s = scaling_efficiency(pts)["batch8_ckpt1"]["by_world_size"]
    assert s["1"]["efficiency_vs_ws1"] == pytest.approx(1.0, abs=1e-9, rel=0)
    assert s["2"]["efficiency_vs_ws1"] == pytest.approx(32.0 / 40.0, abs=1e-9, rel=0)
    assert s["8"]["efficiency_vs_ws1"] == pytest.approx(20.0 / 40.0, abs=1e-9, rel=0)
    # Aggregate throughput still rises even as per-GPU efficiency falls — the two must not
    # be conflated; speedup is the aggregate ratio.
    assert s["8"]["windows_per_sec_aggregate"] == pytest.approx(160.0, abs=1e-9, rel=0)
    assert s["8"]["speedup_vs_ws1"] == pytest.approx(4.0, abs=1e-9, rel=0)


def test_scaling_efficiency_only_compares_matched_configs():
    """A ws=8 point at a different batch must not be baselined against ws=1 at batch 8."""
    pts = make_points(
        [
            {"world_size": 1, "batch_size": 8, "step_seconds": _rate_steps(0.20)},
            {"world_size": 8, "batch_size": 32, "step_seconds": _rate_steps(0.40)},
        ]
    )
    out = scaling_efficiency(pts)
    # batch8 has no ws>1 partner and batch32 has no ws=1 baseline → nothing reportable.
    assert out == {}


def test_scaling_efficiency_omits_series_with_no_ws1_baseline():
    pts = make_points(
        [
            {"world_size": 2, "batch_size": 8, "step_seconds": _rate_steps(0.25)},
            {"world_size": 8, "batch_size": 8, "step_seconds": _rate_steps(0.40)},
        ]
    )
    assert scaling_efficiency(pts) == {}  # never silently baselines at 1.0


def test_scaling_efficiency_separates_checkpointing_settings():
    pts = make_points(
        [
            {"world_size": 1, "batch_size": 8, "ckpt": True, "step_seconds": _rate_steps(0.20)},
            {"world_size": 2, "batch_size": 8, "ckpt": True, "step_seconds": _rate_steps(0.25)},
            {"world_size": 1, "batch_size": 8, "ckpt": False, "step_seconds": _rate_steps(0.10)},
            {"world_size": 2, "batch_size": 8, "ckpt": False, "step_seconds": _rate_steps(0.125)},
        ]
    )
    out = scaling_efficiency(pts)
    assert set(out) == {"batch8_ckpt1", "batch8_ckpt0"}
    assert out["batch8_ckpt1"]["baseline_windows_per_sec_per_gpu"] == pytest.approx(40.0)
    assert out["batch8_ckpt0"]["baseline_windows_per_sec_per_gpu"] == pytest.approx(80.0)


# ═════════════════════════════════════════════════════════════════════════════
# checkpointing_cost
# ═════════════════════════════════════════════════════════════════════════════
def test_checkpointing_cost_quantifies_the_trade_at_matched_config():
    pts = make_points(
        [
            {
                "batch_size": 8,
                "ckpt": True,
                "step_seconds": _rate_steps(0.40),
                "peak_vram_gib": 1.0,
            },
            {
                "batch_size": 8,
                "ckpt": False,
                "step_seconds": _rate_steps(0.20),
                "peak_vram_gib": 4.6,
            },
        ]
    )
    e = checkpointing_cost(pts)["ws1_batch8"]
    # ckpt ON is half the throughput → cost factor 2.0; and saves 4.6x the VRAM.
    assert e["throughput_cost_factor"] == pytest.approx(2.0, abs=1e-9, rel=0)
    assert e["throughput_retained_frac"] == pytest.approx(0.5, abs=1e-9, rel=0)
    assert e["vram_saving_factor"] == pytest.approx(4.6, abs=1e-9, rel=0)


def test_checkpointing_cost_needs_both_sides():
    pts = make_points([{"batch_size": 8, "ckpt": True}])
    assert checkpointing_cost(pts) == {}


# ═════════════════════════════════════════════════════════════════════════════
# largest_fitting_batch — absence-is-the-evidence
# ═════════════════════════════════════════════════════════════════════════════
def test_largest_fitting_batch_reads_absence_as_oom():
    pts = make_points([{"batch_size": 8}, {"batch_size": 16}])  # 32 OOM'd → no report
    assert largest_fitting_batch(pts, world_size=1) == 16


def test_largest_fitting_batch_none_when_nothing_ran():
    assert largest_fitting_batch([], world_size=1) is None


# ═════════════════════════════════════════════════════════════════════════════
# extrapolate
# ═════════════════════════════════════════════════════════════════════════════
def test_extrapolate_against_a_hand_computed_cost():
    # 100 windows/s/GPU x 8 GPUs = 800/s. 2,880,000 windows / 800 = 3600 s = 1.0 h wall,
    # 8.0 GPU-hours. Chosen so the arithmetic is checkable by eye.
    e = extrapolate(
        windows_per_sec_per_gpu=100.0, windows_per_epoch=288_000, epochs=10, world_size=8
    )
    assert e["total_windows"] == pytest.approx(2_880_000.0, abs=1e-6, rel=0)
    assert e["windows_per_sec_aggregate"] == pytest.approx(800.0, abs=1e-9, rel=0)
    assert e["wall_clock_hours"] == pytest.approx(1.0, abs=1e-9, rel=0)
    assert e["gpu_hours"] == pytest.approx(8.0, abs=1e-9, rel=0)


def test_extrapolate_gpu_hours_are_wall_clock_times_gpus():
    e = extrapolate(windows_per_sec_per_gpu=7.5, windows_per_epoch=1234, epochs=3, world_size=4)
    assert e["gpu_hours"] == pytest.approx(e["wall_clock_hours"] * 4, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"windows_per_sec_per_gpu": 0.0, "windows_per_epoch": 10, "epochs": 1, "world_size": 1},
        {"windows_per_sec_per_gpu": -1.0, "windows_per_epoch": 10, "epochs": 1, "world_size": 1},
        {
            "windows_per_sec_per_gpu": math.nan,
            "windows_per_epoch": 10,
            "epochs": 1,
            "world_size": 1,
        },
        {"windows_per_sec_per_gpu": 1.0, "windows_per_epoch": 0, "epochs": 1, "world_size": 1},
        {"windows_per_sec_per_gpu": 1.0, "windows_per_epoch": 10, "epochs": 0, "world_size": 1},
        {"windows_per_sec_per_gpu": 1.0, "windows_per_epoch": 10, "epochs": 1, "world_size": True},
    ],
)
def test_extrapolate_rejects_degenerate_inputs(kwargs):
    with pytest.raises(ValueError):
        extrapolate(**kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# build_report / validate_report — the honesty invariants
# ═════════════════════════════════════════════════════════════════════════════
def full_points():
    return make_points(
        [
            {"world_size": 1, "batch_size": 8, "ckpt": True, "step_seconds": _rate_steps(0.20)},
            {"world_size": 8, "batch_size": 8, "ckpt": True, "step_seconds": _rate_steps(0.40)},
        ]
    )


def test_build_report_round_trips_its_own_validator():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    assert validate_report(r) == []
    assert r["gate"]["overall_pass"] is True


def test_report_never_freezes_the_adr0003_d7_budget():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    assert r["freezes_adr0003_d7_budget"] is False
    assert r["is_science"] is False
    assert r["gate4_graded"] is False
    e = r["illustrative_extrapolation"]
    assert e["advisory_only"] is True and e["binding"] is False
    assert "ADR-0003 D7" in e["binding_gate"]


def test_validator_rejects_a_report_claiming_to_freeze_the_budget():
    """Anti-tautology: the D7 invariant is only real if claiming otherwise FAILS."""
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    r["freezes_adr0003_d7_budget"] = True
    problems = validate_report(r)
    assert any("freezes_adr0003_d7_budget" in p for p in problems)


def test_validator_rejects_an_extrapolation_marked_binding():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    r["illustrative_extrapolation"]["binding"] = True
    assert any("binding" in p for p in validate_report(r))


def test_validator_rejects_an_extrapolation_with_no_gate_attribution():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    r["illustrative_extrapolation"]["binding_gate"] = ""
    assert any("binding_gate" in p for p in validate_report(r))


def test_validator_catches_a_fabricated_true_clause():
    """`overall_pass = all(clauses)` cannot catch a clause fabricated TRUE — the
    re-derivation is what catches it. Prove the re-derivation bites."""
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    r["illustrative_extrapolation"]["advisory_only"] = False  # evidence now says False
    r["gate"]["extrapolation_marked_advisory"] = True  # but the clause still claims True
    problems = validate_report(r)
    assert any("extrapolation_marked_advisory" in p for p in problems)


def test_validator_catches_a_fabricated_overall_pass():
    """The evidence must make a clause HONESTLY False, then `overall_pass` is fabricated True.

    Tampering with the *stored* clause instead would be caught by the clause re-derivation
    and never exercise the `overall_pass` branch at all — the test would pass while
    asserting nothing about the invariant it names.
    """
    pts = make_points([{"grads_finite": False}])  # a genuinely failing clause
    r = build_report(points=pts, windows_per_epoch=288_000, epochs=10)
    assert r["gate"]["overall_pass"] is False  # the honest value
    r["gate"]["overall_pass"] = True  # forged
    problems = validate_report(r)
    assert any("overall_pass" in p for p in problems), problems


def test_validator_catches_a_clause_flipped_against_its_evidence():
    """The other direction: evidence says True, the stored clause claims False."""
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    r["gate"]["vram_under_prd_budget"] = False
    problems = validate_report(r)
    assert any("vram_under_prd_budget" in p for p in problems), problems


def test_vram_clause_fails_on_an_over_budget_peak():
    pts = make_points([{"peak_vram_gib": PRD_VRAM_BUDGET_GIB + 0.5, "total_vram_gib": 24.0}])
    r = build_report(points=pts, windows_per_epoch=1000, epochs=1)
    assert r["gate"]["vram_under_prd_budget"] is False
    assert r["gate"]["overall_pass"] is False
    assert validate_report(r) == []  # a FAILING gate is still a VALID report


def test_margin_is_reported_not_just_the_boolean():
    """P1-16's lesson: on a 15.6 GiB card the '< 16 GB' boolean is near-unfailable, so the
    margin is the informative number. It must exist and be signed correctly."""
    pts = make_points([{"peak_vram_gib": 1.0}])
    r = build_report(points=pts, windows_per_epoch=1000, epochs=1)
    assert r["min_vram_margin_gib_vs_prd_budget"] == pytest.approx(15.0, abs=1e-9, rel=0)
    assert r["max_peak_vram_gib"] == pytest.approx(1.0, abs=1e-9, rel=0)


def test_grads_not_finite_fails_the_gate():
    pts = make_points([{"grads_finite": False}])
    r = build_report(points=pts, windows_per_epoch=1000, epochs=1)
    assert r["gate"]["all_points_grads_finite"] is False
    assert r["gate"]["overall_pass"] is False


def test_extrapolation_refuses_a_world_size_it_has_no_point_for():
    """D7 pins no scaling law — a ws=8 budget must never be modelled from a ws=1 point."""
    pts = make_points([{"world_size": 1, "batch_size": 8}])
    r = build_report(points=pts, windows_per_epoch=288_000, epochs=10)
    e = r["illustrative_extrapolation"]
    assert e["basis_point"] is None
    assert "gpu_hours" not in e
    assert str(PRD_TARGET_WORLD_SIZE) in e["unavailable_reason"]


def test_extrapolation_basis_is_the_prd_pinned_config_not_the_fastest_point():
    """The load-bearing one: checkpointing-OFF is FASTER, so a speed-selected basis quotes a
    config PRD §10.3 does not pin under the field a reader reads as "the" full-run cost.

    Constructed so the two selection rules disagree: ckpt-off runs at 2x the ckpt-on rate.
    A basis chosen by speed picks ws8_bs8_ckpt0; the correct rule picks ws8_bs8_ckpt1.
    """
    pts = make_points(
        [
            {"world_size": 1, "batch_size": 8, "ckpt": True, "step_seconds": _rate_steps(0.40)},
            {"world_size": 8, "batch_size": 8, "ckpt": True, "step_seconds": _rate_steps(0.40)},
            {"world_size": 8, "batch_size": 8, "ckpt": False, "step_seconds": _rate_steps(0.20)},
        ]
    )
    r = build_report(points=pts, windows_per_epoch=288_000, epochs=10)
    e = r["illustrative_extrapolation"]
    assert e["basis_point"] == "ws8_bs8_ckpt1", "basis must follow the PRD §10.3 pin, not speed"
    assert e["gradient_checkpointing"] is True
    assert e["is_prd_pinned_config"] is True
    assert r["gate"]["extrapolation_basis_is_prd_pinned_config"] is True
    # The faster, unpinned config is still reported — but named as an alternative.
    alt = e["alternative_gradient_checkpointing_off"]
    assert alt["basis_point"] == "ws8_bs8_ckpt0"
    assert alt["is_prd_pinned_config"] is False
    assert (
        alt["gpu_hours"] < e["gpu_hours"]
    ), "ckpt off is faster; the fixture is meaningless otherwise"


def test_gate_rejects_an_extrapolation_built_on_the_unpinned_config():
    """Anti-tautology: prove the new clause bites."""
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    r["illustrative_extrapolation"]["is_prd_pinned_config"] = False
    problems = validate_report(r)
    assert any("extrapolation_basis_is_prd_pinned_config" in p for p in problems), problems


def test_alternative_block_is_absent_when_no_unpinned_point_ran():
    """No ckpt-off ws=8 point → the alternative states why, rather than inventing a number."""
    pts = make_points(
        [
            {"world_size": 1, "batch_size": 8, "ckpt": True, "step_seconds": _rate_steps(0.40)},
            {"world_size": 8, "batch_size": 8, "ckpt": True, "step_seconds": _rate_steps(0.40)},
        ]
    )
    r = build_report(points=pts, windows_per_epoch=288_000, epochs=10)
    alt = r["illustrative_extrapolation"]["alternative_gradient_checkpointing_off"]
    assert alt["basis_point"] is None
    assert "gpu_hours" not in alt
    assert "gradient_checkpointing=False" in alt["unavailable_reason"]


def test_extrapolation_uses_the_target_world_size_point_when_present():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    e = r["illustrative_extrapolation"]
    assert e["basis_point"] == "ws8_bs8_ckpt1"
    assert e["world_size"] == PRD_TARGET_WORLD_SIZE
    # ws8 steady median 0.40 s at batch 8 → 20/s/GPU → 160/s aggregate.
    assert e["windows_per_sec_aggregate"] == pytest.approx(160.0, abs=1e-9, rel=0)
    assert e["gpu_hours"] == pytest.approx(2_880_000 / 160.0 / 3600.0 * 8, rel=1e-12)


def test_report_discloses_the_grad_scan_is_inside_the_measured_step():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    assert r["grad_finiteness_scan_in_step_seconds"] is True
    assert any("grad-finiteness" in a for a in r["illustrative_extrapolation"]["assumptions"])


def test_report_discloses_fp32_is_not_the_bf16_native_throughput():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    assert any("fp32" in a and "TF32" in a for a in r["illustrative_extrapolation"]["assumptions"])


def test_validator_is_total_on_malformed_input():
    """Never raises; reports. A validator that crashes fails open on the worst inputs."""
    for bad in (None, [], "nope", 42, {"step": "P2-05"}):
        out = validate_report(bad)
        assert isinstance(out, list) and out


def test_derive_clauses_is_total_on_malformed_input():
    assert derive_clauses(None) == {}
    assert isinstance(derive_clauses({}), dict)


def test_validator_rejects_wrong_step():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    r["step"] = "P2-04"
    assert any("step" in p for p in validate_report(r))


def test_validator_rejects_n_points_disagreeing_with_points():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    r["n_points"] = 99
    assert any("n_points" in p for p in validate_report(r))


def test_report_is_json_serialisable():
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    assert json.loads(json.dumps(r))["step"] == "P2-05"


# ═════════════════════════════════════════════════════════════════════════════
# The COMMITTED measured report (anti-rot tier)
#
# Fail-closed on TBOX_REQUIRE_SIZING_SMOKE, armed in CI. P2-04's lesson: an env guard
# armed NOWHERE makes its own fail-closed branch dead code, and every committed-report
# test then skips green if the artifact vanishes — the anti-rot contract written and then
# not connected.
# ═════════════════════════════════════════════════════════════════════════════
_REPO = Path(__file__).resolve().parents[2]
_SIZING_REPORT = _REPO / "reports" / "p2" / "sizing_smoke.json"
_SIZING_POINTS = _REPO / "reports" / "p2" / "sizing"


def _fail_or_skip(var: str, reason: str) -> None:
    if os.environ.get(var) == "1":
        pytest.fail(f"{var}=1 but the tier is unrunnable: {reason}")
    pytest.skip(reason)


def _committed():
    if not _SIZING_REPORT.exists():
        _fail_or_skip("TBOX_REQUIRE_SIZING_SMOKE", f"no committed report at {_SIZING_REPORT}")
    return json.loads(_SIZING_REPORT.read_text())


def test_committed_report_validates():
    assert validate_report(_committed()) == []


def test_committed_report_clauses_re_derive():
    """Catches TAMPERING with the committed gate.

    Stated limit, not glossed: `build_report` SEALED that gate with `derive_clauses`, so
    this cannot catch a wrong derivation — it would seal and re-derive identically. What
    pins the derivation independently is the hand-built tier above, where each clause is
    asserted against evidence constructed to violate it. Neither tier suffices alone.
    """
    r = _committed()
    derived = derive_clauses(r)
    for name, value in derived.items():
        assert r["gate"][name] is value, name
    assert r["gate"]["overall_pass"] is all(derived.values())


def test_committed_report_was_measured_on_an_a4000_not_the_laptop():
    """The §10.3 budget is about a 16 GB A4000.

    P2-04 asserted a laptop RTX 4060 number as an A4000 property, and P2-03 shipped sample
    properties as corpus facts — a VRAM number is unattributable without its card, so pin
    the attribution rather than trust the filename.
    """
    r = _committed()
    assert "A4000" in (r["device_name"] or ""), r["device_name"]
    # The real card is ~15.602 GiB usable, which is LESS than the PRD's 16 GB budget —
    # the two are different claims and the report must not conflate them.
    assert 15.0 < r["total_vram_gib"] < 16.0
    assert r["prd_vram_budget_gib"] == PRD_VRAM_BUDGET_GIB


def test_committed_report_does_not_freeze_the_budget():
    """The ADR-0003 D7 invariant, checked on the artifact that actually ships."""
    r = _committed()
    assert r["freezes_adr0003_d7_budget"] is False
    assert r["is_science"] is False and r["gate4_graded"] is False
    e = r["illustrative_extrapolation"]
    assert (
        e["advisory_only"] is True and e["binding"] is False and "ADR-0003 D7" in e["binding_gate"]
    )


def test_committed_extrapolation_is_built_on_the_prd_pinned_config():
    """ckpt-OFF is faster and IS present in the committed points, so this is not vacuous."""
    r = _committed()
    e = r["illustrative_extrapolation"]
    assert e["gradient_checkpointing"] is True and e["is_prd_pinned_config"] is True
    alt = e["alternative_gradient_checkpointing_off"]
    assert alt["is_prd_pinned_config"] is False
    assert alt["gpu_hours"] < e["gpu_hours"], "the unpinned config is the faster one"


def test_committed_points_cover_the_ddp8_claim_and_the_oom_boundary():
    """The two facts the step exists to establish must actually be in the artifact."""
    if not _SIZING_POINTS.exists():
        _fail_or_skip("TBOX_REQUIRE_SIZING_SMOKE", f"no committed points at {_SIZING_POINTS}")
    keys = {p.stem for p in _SIZING_POINTS.glob("*.json")}
    assert "ws8_bs8_ckpt1" in keys, "no DDP x8 point — the PRD §10.3 claim is ungraded"
    # batch 32 WITHOUT checkpointing OOM'd natively on the A4000: absence IS the evidence
    # (the sbatch writes no report for a failed point), and its presence would mean the
    # OOM boundary moved.
    assert (
        "ws1_bs32_ckpt0" not in keys
    ), "batch 32 ckpt-off OOM'd on the A4000; a report means it no longer does"
    assert "ws1_bs32_ckpt1" in keys, "batch 32 WITH checkpointing fits — that is the §10.3 point"
    r = _committed()
    assert r["ddp_scaling"]["batch8_ckpt1"]["by_world_size"]["8"]["efficiency_vs_ws1"] > 0.5


def test_the_step_seconds_fixture_can_distinguish_median_from_mean():
    """Guards the guard: if mean == median on the fixture, every rate test is blind to the
    statistic it claims to pin. A reviewer proved the old symmetric fixture let
    `seconds_median = seconds_mean` pass all 59 tests."""
    s = steady_state(make_point_report()["steps"]["step_seconds"], warmup=5)
    assert s["seconds_median"] == pytest.approx(0.20, abs=1e-12, rel=0)
    assert (
        abs(s["seconds_mean"] - s["seconds_median"]) > 0.02
    ), "fixture is symmetric — a mean-vs-median swap would be undetectable"
    r = steady_state(_rate_steps(0.25), warmup=5)
    assert r["seconds_median"] == pytest.approx(0.25, abs=1e-12, rel=0)
    assert abs(r["seconds_mean"] - r["seconds_median"]) > 0.02


def test_gate_fails_when_no_ddp_target_point_was_measured():
    """The step exists to grade PRD §10.3's 'DDP×8 for throughput'.

    Reviewer-found, verified by execution: aggregating only ws=1 points previously gave a
    GREEN nine-clause gate with basis_point=None, no gpu_hours and ddp_scaling={} — a clause
    fabricated TRUE on absent evidence. One NCCL fault kills all four torchrun points
    together, so this is a live path, not a hypothetical.
    """
    pts = make_points([{"world_size": 1, "batch_size": 8, "ckpt": True}])
    r = build_report(points=pts, windows_per_epoch=288_000, epochs=10)
    assert r["illustrative_extrapolation"]["basis_point"] is None
    assert r["gate"]["ddp_target_world_size_measured"] is False
    assert (
        r["gate"]["extrapolation_basis_is_prd_pinned_config"] is False
    ), "a config label attached to no measurement must not certify the basis"
    assert r["gate"]["overall_pass"] is False
    assert validate_report(r) == []  # a FAILING gate is still a VALID report


def test_gate_passes_only_once_the_ddp_target_point_exists():
    """The complement — proves the new clauses are not simply always-False."""
    r = build_report(points=full_points(), windows_per_epoch=288_000, epochs=10)
    assert r["gate"]["ddp_target_world_size_measured"] is True
    assert r["gate"]["extrapolation_basis_is_prd_pinned_config"] is True
    assert r["gate"]["overall_pass"] is True
