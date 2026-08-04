"""P3-10 — GATE-2's P3 half: the in-distribution ECE gate and the D13 leave-clade-out read.

numpy-only, no torch, no pandas: this tier runs in CI's stack rather than ``importorskip``-ing
itself green. Everything under the ``TORCH TIER`` banner of ``calib/gate2.py`` is exercised by
``tests/ml/test_stage2_eval_smoke.py``'s sibling, not here.

What this file is trying to catch, in order of how badly it would hurt:

1. A gate that **passes on absent evidence.** Every clause is checked to be FALSE on an empty
   report, and each is then driven TRUE/FALSE independently — a clause that cannot go FALSE
   grades nothing ([[clauses-must-guard-emptiness]], [[gate-clauses-need-re-derivation]]).
2. A **leak** that turns an in-distribution number into a claimed OOD one. The order-level
   disjointness check is the one a row-level check cannot see, so it is sabotaged separately
   from the row-level one ([[sabotage-attribution-names-the-test]]).
3. A **cost knob certifying** — a truncated unit list producing a report that reads like a
   full grade ([[cost-knobs-can-certify]]).
4. A **quietly pinned** deployment prior or D13 drift bound. Every refusal here is paired
   with the identical-but-clean input validating clean, so a guard that refuses everything
   cannot pass ([[raises-test-needs-a-positive-control]]).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tbox_finder import coverage as COV
from tbox_finder import metrics as M
from tbox_finder import power as PW
from tbox_finder.calib import ece as ECE
from tbox_finder.calib import gate2 as G
from tbox_finder.calib import recalibrate as R
from tbox_finder.eval import resample as RS

# No `pytest.importorskip("numpy")` here: the `tbox_finder` imports above already pull
# numpy, so collection would fail in that block and a guard below it could never run.
# This module's dependency on numpy is hard, and the file docstring says so.

_REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Fixtures — deliberately ASYMMETRIC, so an inversion cannot preserve a count
# --------------------------------------------------------------------------- #
def _split_rows() -> list[dict]:
    """A miniature split table: 5 trained rows over 2 orders, 7 holdout rows over 3.

    The arm sizes and the per-unit sizes are all different, so a fold-sense inversion or a
    swapped population changes *which* rows are selected and not merely how many
    ([[symmetric-count-fixture-blind-to-inversion]]).
    """
    rows: list[dict] = []
    # trained: fold_random=train, not calib, nested_train=True — orders A and B
    for i, order in enumerate(["OrderA", "OrderA", "OrderA", "OrderB", "OrderB"]):
        rows.append(
            {
                "row_id": f"tr{i}",
                "is_tbox": 1,
                "cluster_id": float(100 + i),
                "calib": False,
                "fold_random": "train",
                "fold_basis": "parent_record",
                "nested_train": True,
                "is_designated_loo_holdout": False,
                "loo_order_unit": None,
                "resolved_order": order,
                "rna_sequence": "ACGU",
            }
        )
    # a parentless decoy admitted by the second disjunct only (nested_train is False)
    rows.append(
        {
            "row_id": "trdec",
            "is_tbox": 0,
            "cluster_id": None,
            "calib": False,
            "fold_random": "train",
            "fold_basis": "decoy_pool_random",
            "nested_train": False,
            "is_designated_loo_holdout": False,
            "loo_order_unit": None,
            "resolved_order": None,
            "rna_sequence": "ACGU",
        }
    )
    # a calib row on the train fold — excluded from training by the `not calib` clause
    rows.append(
        {
            "row_id": "cal0",
            "is_tbox": 1,
            "cluster_id": 200.0,
            "calib": True,
            "fold_random": "train",
            "fold_basis": "parent_record",
            "nested_train": True,
            "is_designated_loo_holdout": False,
            "loo_order_unit": None,
            "resolved_order": "OrderA",
            "rna_sequence": "ACGU",
        }
    )
    # holdout: 4 rows in OrderC, 2 in OrderD, 1 in OrderE — asymmetric on purpose
    plan = [("OrderC", 4), ("OrderD", 2), ("OrderE", 1)]
    n = 0
    for order, count in plan:
        for _ in range(count):
            rows.append(
                {
                    "row_id": f"ho{n}",
                    "is_tbox": 1,
                    "cluster_id": float(300 + n),
                    "calib": False,
                    "fold_random": "train" if n % 2 else "test",
                    "fold_basis": "parent_record",
                    "nested_train": False,
                    "is_designated_loo_holdout": True,
                    "loo_order_unit": order,
                    "resolved_order": order,
                    "rna_sequence": "ACGU",
                }
            )
            n += 1
    return rows


def _in_distribution_scores(n_calib: int = 260, n_test: int = 420, n_val: int = 95) -> dict:
    """Three unequal rungs whose logits are calibrated **by construction**.

    Labels are drawn ``y ~ Bernoulli(sigma(z))``, so ``sigma(z)`` is the true conditional
    mean and the honest reading is a small ECE with ``T`` near 1. Two properties follow that
    a hand-tuned fixture would have to be lucky to get:

    * the arms **overlap**, so the calib carve has misclassified rows and a temperature
      exists at all. A perfectly separated carve has none — the NLL minimiser is the
      ``beta -> inf`` limit and P3-07's fitter refuses it by design, which is exactly what
      P3-08 measured on the no-aux arm — so a cleanly separated fixture would exercise the
      refusal path and nothing else;
    * the gated clause is TRUE for a *reason*, not because a number was chosen to make it
      TRUE. The fixture asserts both, so a change that breaks either fails here rather than
      silently turning the "honest report passes every clause" control vacuous.
    """
    rng = __import__("random").Random(20260804)
    row_ids, logits, labels, rungs = [], [], [], []
    for rung, count in (("calib", n_calib), ("test", n_test), ("val", n_val)):
        for i in range(count):
            z = -3.2 + 6.4 * ((i + 0.5) / count)
            y = 1 if rng.random() < 1.0 / (1.0 + math.exp(-z)) else 0
            row_ids.append(f"{rung}-{i}")
            labels.append(y)
            logits.append(z)
            rungs.append(rung)
    for rung in ("calib", "test", "val"):
        wrong_side = sum(
            1
            for r, y, z in zip(rungs, labels, logits, strict=True)
            if r == rung and ((y == 1) != (z > 0.0))
        )
        assert wrong_side >= 2, f"{rung} separated cleanly: T would be unfittable, not measured"
    return {
        "row_ids": row_ids,
        "logits": logits,
        "labels": labels,
        "rungs": rungs,
        "meta": {"device": "cpu", "batch_size": 4},
        "load": None,
    }


def _blocks_for(scores: dict) -> dict[str, str]:
    """Two rows per homology cluster, so the block bootstrap has fewer blocks than rows."""
    return {rid: f"cluster:{i // 2}" for i, rid in enumerate(scores["row_ids"])}


def _report() -> dict:
    """A complete, honest, passing GATE-2 report built through the shipped assembler."""
    scores = _in_distribution_scores()
    gate = G.grade_in_distribution(
        scores=scores, blocks_by_row=_blocks_for(scores), n_boot=40, seed=7
    )
    shift = G.prior_shift_band_sweep(
        scores=scores,
        source_prior=float(gate["calibration"]["calib_prevalence"]),
        temperature=float(gate["calibration"]["temperature"]),
    )
    ood = _ood_block(n_units=2)
    scope = {
        "gate_rung": G.GATE_RUNG,
        "n_gate_rung_rows_in_split_table": gate["n"],
        "n_loo_holdout_rows_in_split_table": 7,
        "n_loo_holdout_rows_scored": 7,
        "n_loo_holdout_units_in_split_table": 2,
        "n_row_overlap": 0,
        "n_order_overlap": 0,
        "rescore_n_overlap": 3,
        "rescore_max_abs_delta": 1e-4,
    }
    return G.build_report(
        gate=gate,
        prior_shift=shift,
        ood=ood,
        scope=scope,
        scoring={"arm": "aux1.0_lr1e-4"},
        provenance={"git_sha": "deadbeef"},
        written_at="2026-08-04T00:00:00+00:00",
    )


def _phylum_scores(plan):
    """Units with per-unit sizes and phyla from ``plan`` — asymmetric on purpose."""
    rng = __import__("random").Random(17)
    row_ids, labels, logits, rows_by_id = [], [], [], {}
    for unit, phylum, count, centre in plan:
        for i in range(count):
            rid = f"{unit}-{i}"
            row_ids.append(rid)
            labels.append(1 if i % 6 else 0)
            logits.append(centre + 0.4 * rng.gauss(0.0, 1.0))
            rows_by_id[rid] = {"_unit": unit, "_phylum": phylum, "_block": f"c:{unit}:{i // 3}"}
    scores = {
        "row_ids": row_ids,
        "logits": logits,
        "labels": labels,
        "rungs": None,
        "meta": {},
        "load": None,
    }
    return scores, rows_by_id


def _ood_block(*, n_units: int) -> dict:
    """The OOD block, built by the SHIPPED producer rather than by hand.

    An earlier version assembled this dict literally and drifted: it was missing
    ``why_not_gated``, ``by_phylum``, ``adjudicable_fraction``, ``n_boot``, ``bootstrap_seed``
    and ``n_units_sub_min_n``, so no test resting on it could have been checking those. A
    fixture that stands in for a producer has to come from that producer — otherwise every
    new field the producer grows is a field the whole suite silently stops covering.
    """
    plan = [
        (f"Order{chr(65 + i)}", f"Phylum{'One' if i % 2 else 'Two'}", 40 + 7 * i, 2.6 - 0.3 * i)
        for i in range(n_units)
    ]
    scores, rows_by_id = _phylum_scores(plan)
    return G.grade_ood_units(
        scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=8, seed=3
    )


# --------------------------------------------------------------------------- #
# 1. Nothing is TRUE by omission
# --------------------------------------------------------------------------- #
def test_every_clause_is_false_on_an_empty_report() -> None:
    clauses = G.derive_clauses({})
    assert clauses, "derive_clauses returned nothing, so no clause could ever fail"
    true_on_nothing = sorted(name for name, ok in clauses.items() if ok)
    assert true_on_nothing == [], (
        "these clauses read TRUE from a report with no evidence at all, i.e. they are "
        f"satisfied by the absence of the thing they check: {true_on_nothing}"
    )


def test_every_clause_is_true_on_the_honest_report() -> None:
    """The positive control for the test above: a real, complete report passes every clause."""
    report = G.build_report(**_report_kwargs())
    false_clauses = sorted(name for name, ok in report["clauses"].items() if not ok)
    assert false_clauses == [], f"an honest complete report failed: {false_clauses}"
    assert report["overall_pass"] is True
    assert G.validate_report(report) == []


def _report_kwargs() -> dict:
    report = _report()
    return {
        "gate": report["gate"],
        "prior_shift": report["prior_shift"],
        "ood": report["ood"],
        "scope": report["scope"],
        "scoring": report["scoring"],
        "provenance": report["provenance"],
        "written_at": report["generated_at_utc"],
    }


def test_clause_key_set_is_stable_and_named() -> None:
    """The clause names are the report's public contract; drift must be a visible diff."""
    assert set(G.derive_clauses({})) == {
        "d13_drift_bound_left_unadjudicated",
        "deployment_prior_is_a_band_not_a_pin",
        "ece_estimator_matches_adr_d11",
        "gate_threshold_is_the_pinned_default",
        "graded_every_loo_holdout_unit",
        "graded_object_is_pre_prior_shift",
        "graded_rung_is_the_gate2_split",
        "in_distribution_ece_within_gate",
        "loo_holdout_orders_disjoint_from_training_orders",
        "loo_holdout_rows_disjoint_from_training_rows",
        "ood_estimator_distinct_from_the_in_distribution_one",
        "ood_min_n_floor_is_the_pinned_constant",
        "ood_reported_never_gated",
        "ood_resampled_at_block_granularity",
        "rescoring_reproduces_the_p3_08_overlap",
        "scored_every_designated_loo_holdout_row",
        "scored_every_row_of_the_gate_rung",
        "temperature_fitted_on_calib_only",
        "temperature_positive_and_converged",
    }


# --------------------------------------------------------------------------- #
# 2. The gated statistic
# --------------------------------------------------------------------------- #
def test_the_gate_threshold_is_the_pinned_blinded_frozen_default() -> None:
    assert G.ECE_GATE is PW.ECE_GATE
    assert G.ECE_GATE == 0.05
    assert G.ECE_N_BINS is M.ECE_N_BINS
    assert G.ECE_N_BINS == 15


def test_gate_boundary_grades_both_directions_at_the_pinned_value() -> None:
    report = _report()
    for ece, want in ((G.ECE_GATE - 1e-9, True), (G.ECE_GATE, True), (G.ECE_GATE + 1e-9, False)):
        report["gate"]["ece"] = ece
        assert G.derive_clauses(report)["in_distribution_ece_within_gate"] is want


def test_a_loosened_threshold_is_refused_even_when_the_number_would_pass() -> None:
    report = _report()
    clean = G.validate_report(report)
    assert clean == [], f"positive control failed: {clean}"
    report["gate"]["gate"] = 0.20
    report["gate"]["ece"] = 0.19
    report["gate"]["passes"] = True
    problems = G.validate_report(report)
    assert any("blinded-frozen" in p for p in problems), problems


def test_a_verdict_that_was_assigned_rather_than_derived_is_caught() -> None:
    report = _report()
    assert G.validate_report(report) == []
    report["gate"]["passes"] = not report["gate"]["passes"]
    problems = G.validate_report(report)
    assert any("assigned, not derived" in p for p in problems), problems


def test_the_graded_object_is_the_pre_prior_shift_named_posterior() -> None:
    report = _report()
    assert report["gate"]["graded_posterior_key"] == R.NAMED_POSTERIOR_KEY
    assert report["gate"]["prior_shift_applied"] is False
    # swapping in the OTHER posterior the stack produces must break the clause
    report["gate"]["graded_posterior_key"] = R.PRIOR_SHIFTED_POSTERIOR_KEY
    assert G.derive_clauses(report)["graded_object_is_pre_prior_shift"] is False
    report["gate"]["graded_posterior_key"] = R.NAMED_POSTERIOR_KEY
    report["gate"]["prior_shift_applied"] = True
    assert G.derive_clauses(report)["graded_object_is_pre_prior_shift"] is False


def test_the_estimator_config_is_the_one_adr_d11_pins() -> None:
    report = _report()
    for field, bad in (
        ("ece_n_bins", 10),
        ("ece_binning", "equal_width"),
        ("ece_debiased", False),
        ("estimator", "something_else"),
    ):
        mutated = _report()
        mutated["gate"][field] = bad
        assert G.derive_clauses(mutated)["ece_estimator_matches_adr_d11"] is False, field
    assert G.derive_clauses(report)["ece_estimator_matches_adr_d11"] is True


def test_the_gate_rung_is_the_split_adr_0004_a7_names() -> None:
    report = _report()
    assert report["gate"]["graded_rung"] == "test"
    report["gate"]["graded_rung"] = "val"
    assert G.derive_clauses(report)["graded_rung_is_the_gate2_split"] is False


# --------------------------------------------------------------------------- #
# 3. The fit — T on calib, never on a graded row
# --------------------------------------------------------------------------- #
def test_temperature_is_fitted_on_calib_alone_and_no_fit_row_is_graded() -> None:
    scores = _in_distribution_scores()
    gate = G.grade_in_distribution(
        scores=scores, blocks_by_row=_blocks_for(scores), n_boot=20, seed=7
    )
    cal = gate["calibration"]
    assert cal["fitted_on"] == R.CALIB_RUNG
    # `n_by_rung` is a census of every rung — the fit is what `n_fitted` counts, and it must
    # equal the calib census exactly. Asserting the census had ONE key would be a clause that
    # can only ever fail, which is a different bug from the one this test is for.
    assert cal["n_by_rung"][R.CALIB_RUNG] == cal["n_fitted"]
    assert cal["n_fitted"] == sum(1 for r in scores["rungs"] if r == "calib")
    assert cal["n_fit_rows_also_graded"] == 0
    # and the graded population is the test rung ITSELF, not merely the right size
    assert gate["n"] == sum(1 for r in scores["rungs"] if r == "test")


def test_a_fit_row_leaking_into_the_grade_turns_the_clause_false() -> None:
    report = _report()
    assert G.derive_clauses(report)["temperature_fitted_on_calib_only"] is True
    report["gate"]["calibration"]["n_fit_rows_also_graded"] = 1
    assert G.derive_clauses(report)["temperature_fitted_on_calib_only"] is False


def test_a_second_rung_in_the_fit_census_turns_the_clause_false() -> None:
    report = _report()
    report["gate"]["calibration"]["n_by_rung"] = {"calib": 20, "test": 1}
    assert G.derive_clauses(report)["temperature_fitted_on_calib_only"] is False


def test_grading_an_empty_rung_raises_rather_than_returning_a_pass() -> None:
    scores = _in_distribution_scores()
    scores = dict(scores, rungs=["calib" if r == "test" else r for r in scores["rungs"]])
    with pytest.raises(ValueError, match="a grade over nothing reads as a pass"):
        G.grade_in_distribution(scores=scores, blocks_by_row=_blocks_for(scores), n_boot=5)


def test_a_score_table_with_no_rungs_cannot_be_graded() -> None:
    scores = dict(_in_distribution_scores(), rungs=None)
    with pytest.raises(ValueError, match="carries no rungs"):
        G.grade_in_distribution(scores=scores, blocks_by_row={}, n_boot=5)


# --------------------------------------------------------------------------- #
# 4. The leave-clade-out population really is out of distribution
# --------------------------------------------------------------------------- #
def test_training_admission_reproduces_the_p3_06_predicate_clause_by_clause() -> None:
    """Each of the four clauses is flipped ALONE — a compound disjunct sabotaged as a whole
    would leave a broken half undetected ([[sabotage-attribution-names-the-test]])."""
    base = {
        "fold_random": "train",
        "calib": False,
        "nested_train": True,
        "fold_basis": "parent_record",
    }
    assert G.training_admission(base) is True
    assert G.training_admission(dict(base, fold_random="test")) is False
    assert G.training_admission(dict(base, calib=True)) is False
    # nested_train False alone excludes a corpus row …
    assert G.training_admission(dict(base, nested_train=False)) is False
    # … but the parentless-decoy disjunct admits it independently
    assert (
        G.training_admission(dict(base, nested_train=False, fold_basis="decoy_pool_random")) is True
    )


def test_a_nan_flag_is_not_a_true_flag() -> None:
    """``bool(nan) is True``; a row whose flag is ABSENT must not be admitted."""
    base = {
        "fold_random": "train",
        "calib": float("nan"),
        "nested_train": True,
        "fold_basis": "parent_record",
    }
    assert G.training_admission(base) is True, "a missing calib flag means not-calib"
    assert G.training_admission(dict(base, nested_train=float("nan"))) is False
    assert G.training_admission(dict(base, calib="True")) is False
    assert G.training_admission(dict(base, calib="0")) is True
    assert G.training_admission(dict(base, calib="1")) is False


def test_the_holdout_census_measures_row_and_order_disjointness() -> None:
    holdout, census = G.holdout_from_rows(_split_rows())
    assert census["n_row_overlap"] == 0
    assert census["n_order_overlap"] == 0
    assert census["n_units"] == 3
    assert census["n_holdout_rows"] == 7
    assert census["n_training_rows"] == 6  # 5 corpus + 1 parentless decoy, NOT the calib row
    # identity, not just count: the holdout is exactly the flagged rows
    assert {row["row_id"] for row in holdout} == {f"ho{i}" for i in range(7)}
    assert {row["_unit"] for row in holdout} == {"OrderC", "OrderD", "OrderE"}


def test_an_order_level_leak_is_visible_when_no_row_level_leak_is() -> None:
    """The failure a row check cannot see: the same ORDER on both sides, no shared row."""
    rows = _split_rows()
    for row in rows:
        if row["row_id"] == "tr0":
            row["resolved_order"] = "OrderC"  # a trained row from a held-out order
    _, census = G.holdout_from_rows(rows)
    assert census["n_row_overlap"] == 0, "the row-level check stays clean — that is the point"
    assert census["n_order_overlap"] == 1
    assert census["overlapping_orders"] == ["OrderC"]
    report = _report()
    report["scope"]["n_order_overlap"] = 1
    clauses = G.derive_clauses(report)
    assert clauses["loo_holdout_orders_disjoint_from_training_orders"] is False
    assert clauses["loo_holdout_rows_disjoint_from_training_rows"] is True


def test_a_row_level_leak_turns_its_own_clause_false() -> None:
    report = _report()
    report["scope"]["n_row_overlap"] = 1
    clauses = G.derive_clauses(report)
    assert clauses["loo_holdout_rows_disjoint_from_training_rows"] is False
    assert clauses["loo_holdout_orders_disjoint_from_training_orders"] is True


def test_a_holdout_row_with_no_unit_is_refused_not_dropped() -> None:
    rows = _split_rows()
    clean, _ = G.holdout_from_rows(rows)
    assert len(clean) == 7, "positive control: the unmutated table parses"
    for row in rows:
        if row["row_id"] == "ho0":
            row["loo_order_unit"] = None
    with pytest.raises(ValueError, match="carries no loo_order_unit"):
        G.holdout_from_rows(rows)


def test_a_stringified_null_unit_is_refused_too() -> None:
    rows = _split_rows()
    for row in rows:
        if row["row_id"] == "ho0":
            row["loo_order_unit"] = "None"
    with pytest.raises(ValueError, match="carries no loo_order_unit"):
        G.holdout_from_rows(rows)


# --------------------------------------------------------------------------- #
# 5. The OOD read is a second estimator, reported and never gated
# --------------------------------------------------------------------------- #
def test_the_ood_estimator_is_not_the_in_distribution_one() -> None:
    assert ECE.ESTIMATOR != ECE.IN_DISTRIBUTION_ESTIMATOR
    report = _report()
    assert G.derive_clauses(report)["ood_estimator_distinct_from_the_in_distribution_one"] is True
    report["ood"]["estimator"] = ECE.IN_DISTRIBUTION_ESTIMATOR
    assert G.derive_clauses(report)["ood_estimator_distinct_from_the_in_distribution_one"] is False


def test_no_ood_entry_may_be_gated() -> None:
    report = _report()
    assert G.derive_clauses(report)["ood_reported_never_gated"] is True
    first = sorted(report["ood"]["units"])[0]
    report["ood"]["units"][first]["gated"] = True
    assert G.derive_clauses(report)["ood_reported_never_gated"] is False
    problems = G.validate_report(report)
    assert all("ood.gated" not in p for p in problems), "the block-level flag is still False"
    report["ood"]["gated"] = True
    assert any("must be False" in p for p in G.validate_report(report))


def test_the_min_n_floor_is_the_pinned_constant_and_the_boundary_tracks_it() -> None:
    """Move the pin and both the estimator's admissibility and this clause must move with it.

    Asserting ``min_n == 20`` alone would survive a re-typed literal, and
    ``gate2.X is coverage.OOD_ECE_MIN_N`` would too — CPython caches small ints
    ([[pinned-constant-that-nothing-reads]]). So the pin is checked by *behaviour*: raise the
    floor above a unit's positive count and that unit must become inadmissible, and the
    clause must reject a report whose units were graded against the old floor.

    ``coverage.classify_order``'s ``floor=`` default is bound at def time and therefore does
    NOT follow a re-assigned module attribute — asserted here so the difference between the
    two read paths is written down rather than discovered.
    """
    report = _report()
    assert G.derive_clauses(report)["ood_min_n_floor_is_the_pinned_constant"] is True
    assert COV.classify_order(COV.OOD_ECE_MIN_N) == "adjudicable"
    assert COV.classify_order(COV.OOD_ECE_MIN_N - 1) == "sub_min_n_inconclusive"

    y = [1] * 25 + [0] * 6
    p = [0.9] * 25 + [0.2] * 6
    blocks = [f"cluster:{i // 2}" for i in range(31)]
    admissible_at_20 = ECE.ood_ece(y, p, blocks, block_key="cluster_id", n_boot=4, seed=1)
    assert admissible_at_20["admissible"] is True and admissible_at_20["min_n"] == 20

    # `ece.py` binds the name with `from ... import`, so the estimator reads ITS module
    # global while `gate2`'s clause reads `coverage`'s attribute. Both are moved together
    # here, and the equality below is what keeps the two read paths from drifting.
    assert ECE.OOD_ECE_MIN_N == COV.OOD_ECE_MIN_N
    original = COV.OOD_ECE_MIN_N
    try:
        COV.OOD_ECE_MIN_N = original + 10  # 30 > the unit's 25 positives
        ECE.OOD_ECE_MIN_N = original + 10
        moved = ECE.ood_ece(y, p, blocks, block_key="cluster_id", n_boot=4, seed=1)
        assert moved["admissible"] is False, "ood_ece did not re-read the pin at call time"
        assert moved["min_n"] == original + 10
        assert moved["ood_ece"] is None and moved["inadmissible_point"] is not None
        # the report's units were graded against the ORIGINAL pin, so the clause must fail
        assert G.derive_clauses(report)["ood_min_n_floor_is_the_pinned_constant"] is False
        # …and the def-time default does not follow the attribute, which is why the clause
        # is written against `COV.OOD_ECE_MIN_N` and not against `classify_order`'s default.
        assert COV.classify_order(original) == "adjudicable"
    finally:
        COV.OOD_ECE_MIN_N = original
        ECE.OOD_ECE_MIN_N = original
    assert COV.OOD_ECE_MIN_N == 20 == ECE.OOD_ECE_MIN_N
    assert G.derive_clauses(report)["ood_min_n_floor_is_the_pinned_constant"] is True


def test_a_sub_min_n_unit_yields_no_number_to_grade() -> None:
    y = [1] * 5 + [0] * 3
    p = [0.9, 0.8, 0.85, 0.95, 0.7, 0.2, 0.3, 0.1]
    blocks = [f"cluster:{i // 2}" for i in range(8)]
    out = ECE.ood_ece(y, p, blocks, block_key="cluster_id", n_boot=5, seed=1)
    assert out["admissible"] is False
    assert out["ood_ece"] is None
    assert out["inadmissible_point"] is not None
    assert out["ci"] is None
    assert COV.classify_order(out["n_positives"]) == "sub_min_n_inconclusive"


def test_the_ood_bootstrap_blocks_are_split_columns_never_record_ids() -> None:
    assert G.OOD_BLOCK_KEY in RS.BLOCK_GRANULARITY_COLUMNS
    assert G.OOD_UNIT_KEY in RS.BLOCK_GRANULARITY_COLUMNS
    report = _report()
    assert G.derive_clauses(report)["ood_resampled_at_block_granularity"] is True
    assert RS.RECORD_LEVEL_COLUMNS, (
        "no record-level column to sabotage, so the loop below grades nothing — the exact "
        "pass-on-absent-evidence shape this module exists to catch"
    )
    assert RS.BLOCK_GRANULARITY_COLUMNS, "no block-granularity column to accept either"
    for column in RS.RECORD_LEVEL_COLUMNS:
        mutated = _report()
        mutated["ood"]["unit_key"] = column
        assert G.derive_clauses(mutated)["ood_resampled_at_block_granularity"] is False
        with pytest.raises(ValueError, match="identifies a record, not a block"):
            RS.blocks_by_key([1, 2], ["a", "b"], key_name=column)


def test_the_ood_units_carry_their_phylum_and_are_regrouped_by_it() -> None:
    """PRD §12's ECE-vs-phylogenetic-distance read — descriptive, and never gated."""
    plan = [
        ("OrderA", "PhylumOne", 40, 3.0),
        ("OrderB", "PhylumOne", 31, 0.2),  # deliberately worse-calibrated and smaller
        ("OrderC", "PhylumTwo", 27, 2.6),
    ]
    scores, rows_by_id = _phylum_scores(plan)
    out = G.grade_ood_units(scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=6)
    assert {u["phylum"] for u in out["units"].values()} == {"PhylumOne", "PhylumTwo"}
    strat = out["by_phylum"]
    assert set(strat) == {"PhylumOne", "PhylumTwo"}
    assert strat["PhylumOne"]["n_units"] == 2 and strat["PhylumTwo"]["n_units"] == 1
    assert strat["PhylumOne"]["units"] == ["OrderA", "OrderB"]
    # identity, not just counts: the bounds are the member units' own values
    members = [out["units"][u]["ood_ece"] for u in strat["PhylumOne"]["units"]]
    assert strat["PhylumOne"]["min_ood_ece"] == pytest.approx(min(members))
    assert strat["PhylumOne"]["max_ood_ece"] == pytest.approx(max(members))
    assert strat["PhylumOne"]["n_records"] == 40 + 31
    assert strat["PhylumTwo"]["min_ood_ece"] == pytest.approx(out["units"]["OrderC"]["ood_ece"])
    # regrouping is not a second estimate: no phylum entry carries a CI or a verdict
    for entry in strat.values():
        assert "ci" not in entry and "passes" not in entry and "gated" not in entry
    assert out["gated"] is False


def test_a_sub_min_n_unit_is_excluded_from_the_phylum_regrouping() -> None:
    """An inadmissible unit's value supports no verdict, so it must not enter a summary."""
    plan = [("OrderBig", "PhylumOne", 40, 2.5), ("OrderTiny", "PhylumOne", 9, 2.5)]
    scores, rows_by_id = _phylum_scores(plan)
    out = G.grade_ood_units(scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=6)
    assert out["units"]["OrderTiny"]["admissible"] is False
    assert out["units"]["OrderTiny"]["inadmissible_point"] is not None
    assert out["by_phylum"]["PhylumOne"]["units"] == ["OrderBig"]
    assert out["by_phylum"]["PhylumOne"]["n_units"] == 1
    assert out["by_phylum"]["PhylumOne"]["n_records"] == 40


def test_a_unit_spanning_two_phyla_is_refused_not_averaged() -> None:
    plan = [("OrderA", "PhylumOne", 40, 2.5)]
    scores, rows_by_id = _phylum_scores(plan)
    clean = G.grade_ood_units(scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=6)
    assert clean["units"]["OrderA"]["phylum"] == "PhylumOne"
    rows_by_id["OrderA-0"] = dict(rows_by_id["OrderA-0"], _phylum="PhylumTwo")
    with pytest.raises(ValueError, match="more than one phylum"):
        G.grade_ood_units(scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=6)


def test_grade_ood_units_refuses_a_row_outside_the_named_population() -> None:
    scores = {
        "row_ids": ["ho0", "stranger"],
        "logits": [1.0, 2.0],
        "labels": [1, 1],
        "rungs": None,
        "meta": {},
        "load": None,
    }
    rows_by_id = {"ho0": {"_unit": "OrderC", "_block": "cluster:1"}}
    with pytest.raises(ValueError, match="not in the designated leave-one-order-out holdout"):
        G.grade_ood_units(scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=3)


# --------------------------------------------------------------------------- #
# 6. Completeness — a cost knob must not certify
# --------------------------------------------------------------------------- #
def test_truncating_the_unit_list_turns_the_completeness_clause_false() -> None:
    report = _report()
    assert G.derive_clauses(report)["graded_every_loo_holdout_unit"] is True
    report["ood"]["truncated_to_n_units"] = 1
    assert G.derive_clauses(report)["graded_every_loo_holdout_unit"] is False
    report["ood"]["truncated_to_n_units"] = None
    report["ood"]["n_units"] = report["scope"]["n_loo_holdout_units_in_split_table"] - 1
    assert G.derive_clauses(report)["graded_every_loo_holdout_unit"] is False


def test_a_short_scored_holdout_turns_its_completeness_clause_false() -> None:
    report = _report()
    assert G.derive_clauses(report)["scored_every_designated_loo_holdout_row"] is True
    report["scope"]["n_loo_holdout_rows_scored"] -= 1
    assert G.derive_clauses(report)["scored_every_designated_loo_holdout_row"] is False


def test_is_science_is_the_completeness_conjunction_not_an_assertion() -> None:
    """A cost-knobbed run must not land at the phase-exit path flagged as a full grade.

    Each completeness clause is broken ALONE, because `is_science` is their conjunction and a
    version that read only one of them would still look right on the other two.
    """
    complete = G.build_report(**_report_kwargs())
    assert complete["is_science"] is True
    assert G.validate_report(complete) == []
    assert set(G.COMPLETENESS_CLAUSES) <= set(complete["clauses"])

    breakers = {
        "graded_every_loo_holdout_unit": ("ood", {"truncated_to_n_units": 1}),
        "scored_every_designated_loo_holdout_row": ("scope", {"n_loo_holdout_rows_scored": 1}),
        "scored_every_row_of_the_gate_rung": ("scope", {"n_gate_rung_rows_in_split_table": 999}),
    }
    for clause, (block, patch) in breakers.items():
        kwargs = _report_kwargs()
        kwargs[block] = dict(kwargs[block], **patch)
        truncated = G.build_report(**kwargs)
        assert truncated["clauses"][clause] is False, clause
        assert truncated["is_science"] is False, f"{clause} broken but is_science stayed True"
        assert G.validate_report(truncated) == []
        truncated["is_science"] = True
        assert any("is_science" in p for p in G.validate_report(truncated)), clause


def test_a_short_gate_rung_turns_its_completeness_clause_false() -> None:
    report = _report()
    assert G.derive_clauses(report)["scored_every_row_of_the_gate_rung"] is True
    report["scope"]["n_gate_rung_rows_in_split_table"] = report["gate"]["n"] + 1
    assert G.derive_clauses(report)["scored_every_row_of_the_gate_rung"] is False


def test_max_units_actually_truncates_and_records_that_it_did() -> None:
    rng = __import__("random").Random(5)
    row_ids, labels, rows_by_id = [], [], {}
    for unit, count in (("OrderA", 30), ("OrderB", 25), ("OrderC", 24)):
        for i in range(count):
            rid = f"{unit}-{i}"
            row_ids.append(rid)
            labels.append(1 if i % 5 else 0)
            rows_by_id[rid] = {"_unit": unit, "_block": f"cluster:{unit}:{i // 3}"}
    logits = [rng.gauss(2.0, 1.0) for _ in row_ids]
    scores = {
        "row_ids": row_ids,
        "logits": logits,
        "labels": labels,
        "rungs": None,
        "meta": {},
        "load": None,
    }
    full = G.grade_ood_units(scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=4)
    cut = G.grade_ood_units(
        scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=4, max_units=2
    )
    assert full["n_units"] == 3 and full["truncated_to_n_units"] is None
    assert cut["n_units"] == 2 and cut["truncated_to_n_units"] == 2
    assert set(cut["units"]) == {"OrderA", "OrderB"}


# --------------------------------------------------------------------------- #
# 7. Nothing unpinned was quietly pinned
# --------------------------------------------------------------------------- #
def test_the_deployment_prior_is_reported_as_the_prd_band_not_a_scalar() -> None:
    scores = _in_distribution_scores()
    gate = G.grade_in_distribution(
        scores=scores, blocks_by_row=_blocks_for(scores), n_boot=10, seed=7
    )
    shift = G.prior_shift_band_sweep(
        scores=scores,
        source_prior=float(gate["calibration"]["calib_prevalence"]),
        temperature=float(gate["calibration"]["temperature"]),
    )
    assert shift["pinned_target_prior"] is None
    assert shift["gated"] is False
    assert shift["band_odds"] == [float(x) for x in R.DEPLOYMENT_PRIOR_ODDS_RANGE]
    endpoints = [pt for pt in shift["points"] if pt["is_band_endpoint"]]
    assert len(endpoints) == 2
    assert all(pt["target_prior_in_prd_band"] for pt in shift["points"])
    # the shift moved the posterior: at 10^4:1 the ECE must differ from the pre-shift one
    assert shift["points"][-1]["ece"] != pytest.approx(gate["ece"], abs=1e-9)
    # …and every reported value must still BE an ECE. `prior_shift` returns shifted LOGITS
    # on purpose (the correction is exactly additive there), so a read that forgot to turn
    # them back into a posterior produces a number above 1 that still looks like a metric.
    for pt in shift["points"]:
        assert 0.0 <= pt["ece"] <= 1.0, f"{pt['negatives_per_positive']}: {pt['ece']} is not an ECE"
        assert 0.0 <= pt["ece_plugin"] <= 1.0
    # monotone in the shift's magnitude on this positives-heavy rung: pushing the prior
    # further below the split's own prevalence can only make the posterior more wrong.
    eces = [pt["ece"] for pt in shift["points"]]
    assert eces == sorted(eces), f"the deployment ECE did not grow with the shift: {eces}"
    assert eces[0] > gate["ece"], "the un-shifted posterior must be the better-calibrated one"


def test_pinning_a_deployment_prior_is_refused_and_the_clean_report_is_not() -> None:
    report = _report()
    assert G.validate_report(report) == [], "positive control: the unpinned report is clean"
    report["prior_shift"]["pinned_target_prior"] = 1e-4
    problems = G.validate_report(report)
    assert any("may not pin a deployment prior" in p for p in problems), problems
    assert G.derive_clauses(report)["deployment_prior_is_a_band_not_a_pin"] is False


def test_pinning_the_d13_drift_bound_is_refused_and_the_clean_report_is_not() -> None:
    report = _report()
    assert G.validate_report(report) == []
    report["ood"]["d13_adjudication"]["condition_i_drift_bound"]["bound"] = 0.15
    problems = G.validate_report(report)
    assert any("drift bound is unpinned" in p for p in problems), problems
    assert G.derive_clauses(report)["d13_drift_bound_left_unadjudicated"] is False


def test_deciding_a_calibrated_negative_pass_from_two_conditions_is_refused() -> None:
    report = _report()
    assert G.validate_report(report) == []
    report["ood"]["d13_adjudication"]["calibrated_negative_pass"] = True
    problems = G.validate_report(report)
    assert any("two-of-three verdict" in p for p in problems), problems
    assert G.derive_clauses(report)["d13_drift_bound_left_unadjudicated"] is False


def test_the_drift_bound_clause_is_not_satisfied_by_deleting_the_block() -> None:
    """A report with no D13 block also has no pinned bound — and must NOT pass the clause."""
    report = _report()
    del report["ood"]["d13_adjudication"]
    assert G.derive_clauses(report)["d13_drift_bound_left_unadjudicated"] is False
    problems = G.validate_report(report)
    assert any("d13_adjudication" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# 8. The re-scoring control
# --------------------------------------------------------------------------- #
def test_the_rescoring_agreement_clause_fires_in_both_directions() -> None:
    report = _report()
    assert G.derive_clauses(report)["rescoring_reproduces_the_p3_08_overlap"] is True
    report["scope"]["rescore_max_abs_delta"] = G.RESCORE_AGREEMENT_TOL * 2
    assert G.derive_clauses(report)["rescoring_reproduces_the_p3_08_overlap"] is False
    report["scope"]["rescore_max_abs_delta"] = 0.0
    report["scope"]["rescore_n_overlap"] = 0
    assert G.derive_clauses(report)["rescoring_reproduces_the_p3_08_overlap"] is False, (
        "a zero-row overlap means the control never ran; a perfect delta over nothing "
        "must not certify"
    )


# --------------------------------------------------------------------------- #
# 9. Report plumbing
# --------------------------------------------------------------------------- #
def test_overall_pass_is_the_conjunction_and_a_forged_one_is_caught() -> None:
    """Checked on a report that FAILS a clause, because an all-TRUE one proves nothing.

    ``overall_pass = all(clauses)`` and ``overall_pass = True`` agree exactly when every
    clause is TRUE, so asserting the identity on the honest fixture is satisfied by the
    hardcoded version too. The discriminating case is a report with one clause FALSE, built
    through the real ``build_report`` so the conjunction is the shipped code's own.
    """
    kwargs = _report_kwargs()
    kwargs["scope"] = dict(kwargs["scope"], n_order_overlap=1)  # a real leak -> one clause FALSE
    failing = G.build_report(**kwargs)
    assert failing["clauses"]["loo_holdout_orders_disjoint_from_training_orders"] is False
    assert failing["overall_pass"] is False, (
        "build_report reported an overall pass while a clause was FALSE — the conjunction "
        "was not computed from the clauses"
    )
    assert G.validate_report(failing) == [], "an honestly-failing report is still consistent"

    passing = G.build_report(**_report_kwargs())
    assert passing["overall_pass"] is True
    assert passing["overall_pass"] == all(passing["clauses"].values())
    passing["overall_pass"] = False
    assert any("overall_pass" in p for p in G.validate_report(passing))


def test_a_written_clause_that_disagrees_with_its_re_derivation_is_caught() -> None:
    report = _report()
    name = "graded_rung_is_the_gate2_split"
    report["clauses"][name] = not report["clauses"][name]
    problems = G.validate_report(report)
    assert any(f"clauses.{name}" in p for p in problems), problems


def test_a_dropped_clause_is_caught_rather_than_ignored() -> None:
    report = _report()
    report["clauses"].pop("in_distribution_ece_within_gate")
    problems = G.validate_report(report)
    assert any("key set drifted" in p for p in problems), problems


def test_an_honestly_failing_gate_is_not_a_report_problem() -> None:
    """``validate_report`` checks consistency, not the verdict — a real fail must validate."""
    report = _report()
    report["gate"]["ece"] = 0.42
    report["gate"]["passes"] = False
    report["clauses"] = G.derive_clauses(report)
    report["overall_pass"] = all(report["clauses"].values())
    assert report["overall_pass"] is False
    assert G.validate_report(report) == []


def test_figure_data_carries_no_number_the_report_does_not() -> None:
    report = _report()
    fig = G.figure_data(report)
    assert fig["in_distribution"]["ece"] == report["gate"]["ece"]
    assert fig["gate"] == report["gate"]["gate"] == G.ECE_GATE
    assert fig["ood_min_n"] == report["ood"]["min_n"]
    assert [u["unit"] for u in fig["ood_units"]] == sorted(report["ood"]["units"])
    for entry in fig["ood_units"]:
        source = report["ood"]["units"][entry["unit"]]
        assert entry["ood_ece"] == source["ood_ece"]
        assert entry["n_positives"] == source["n_positives"]


def test_the_disclosures_carry_the_two_inherited_caveats() -> None:
    """Each caveat is matched on BOTH its provenance marker and its substance.

    Matching only the substance ("ONE calib row") survives a rewrite that keeps the phrase
    while dropping which step it came from; matching only the marker survives one that keeps
    the label over a gutted sentence. A sabotage that replaced the P3-08 entry's opening
    clause left "ONE calib row" untouched in the next concatenated line and this test stayed
    green, which is how the pair below got written.
    """
    report = _report()
    entries = report["disclosures"]
    joined = " ".join(entries)
    p3_08 = [e for e in entries if "INHERITED FROM P3-08" in e]
    p3_09 = [e for e in entries if "INHERITED FROM P3-09" in e]
    assert len(p3_08) == 1 and len(p3_09) == 1
    assert "ONE calib row" in p3_08[0], "the P3-08 single-row caveat must travel with the number"
    assert "degenerate-limit rule" in p3_08[0]
    assert "SATURATE" in p3_09[0], "the P3-09 debias-saturation caveat must travel with it too"
    assert "0.123" in p3_09[0], "the measured truth the estimator saturated against"
    assert "P5" in joined, "the FDR half of GATE-2 is not represented here and must say so"


def test_no_recorded_path_publishes_this_machines_layout(tmp_path) -> None:
    """Every path the report records must be repo-relative (CodeRabbit r1, major).

    The step ran from the main checkout with absolute paths into a linked worktree, so the
    committed artifact carried an OS user name and a `.claude/worktrees/...` layout into a
    **public** repo — and two reports generated on different machines would differ in their
    `inputs` key set without differing in content.
    """
    inside = _REPO / "reports" / "p3" / "stage2_scores.json"
    assert not G._recorded_path(inside).startswith("/"), G._recorded_path(inside)
    assert G._recorded_path(inside) == "reports/p3/stage2_scores.json"
    # the same file reached by an absolute path must normalise identically
    assert G._recorded_path(str(inside.resolve())) == G._recorded_path(inside)
    # a path genuinely outside the repo is returned unchanged rather than mangled
    outside = tmp_path / "elsewhere.json"
    assert G._recorded_path(outside) == str(outside)


def test_an_invalid_report_diverts_its_figure_data_too(tmp_path) -> None:
    """The figure-data path is what `plot_gate2_figures` consumes (CodeRabbit r1, minor).

    Leaving figure data derived from a *rejected* report at the consumer path publishes
    figures for a grade nothing accepted. Both artifacts must divert together.
    """
    report_path = tmp_path / "gate2_p3_ece.json"
    figure_path = tmp_path / "gate2_figure_data.json"
    assert G._output_path(report_path, valid=True) == report_path
    assert G._output_path(figure_path, valid=True) == figure_path
    bad_report = G._output_path(report_path, valid=False)
    bad_figure = G._output_path(figure_path, valid=False)
    assert bad_report.name.endswith(".invalid.json") and bad_report != report_path
    assert bad_figure.name.endswith(".invalid.json") and bad_figure != figure_path
    # and the two must divert IN STEP through the shipped writer, not merely be capable of
    # it — checking `_output_path` alone stays green when a call site stops using it, which
    # is exactly what a sabotage of the call site proved.
    report = _report()
    for valid in (True, False):
        out_dir = tmp_path / ("valid" if valid else "invalid")
        rp, fp = out_dir / "gate2_p3_ece.json", out_dir / "p3" / "gate2_figure_data.json"
        wrote_report, wrote_figure = G.write_outputs(
            report, report_path=rp, figure_data_path=fp, valid=valid
        )
        assert wrote_report.is_file() and wrote_figure.is_file()
        assert (wrote_report == rp) is valid, "report diverted out of step"
        assert (wrote_figure == fp) is valid, "figure data diverted out of step"
        assert not rp.is_file() or valid, "a rejected grade was left at the consumer path"
        assert not fp.is_file() or valid, "figure data for a rejected grade reached consumers"
        assert (
            json.loads(wrote_figure.read_text())["in_distribution"]["ece"] == report["gate"]["ece"]
        )


def test_a_diverted_rerun_removes_the_stale_canonical_artifacts(tmp_path) -> None:
    """Diverting is not enough on a RE-run (CodeRabbit app r1, major).

    A previously accepted report left at the consumer path is a *stale grade* that reads as
    current, sitting beside a fresh ``.invalid.json``. The absence of the canonical file is
    what tells a consumer the latest run was refused.
    """
    report = _report()
    rp, fp = tmp_path / "gate2_p3_ece.json", tmp_path / "gate2_figure_data.json"

    good_report, good_figure = G.write_outputs(
        report, report_path=rp, figure_data_path=fp, valid=True
    )
    assert (good_report, good_figure) == (rp, fp) and rp.is_file() and fp.is_file()
    first = json.loads(rp.read_text())["gate"]["ece"]

    stale = dict(report)
    stale["gate"] = dict(report["gate"], ece=0.999)
    bad_report, bad_figure = G.write_outputs(
        stale, report_path=rp, figure_data_path=fp, valid=False
    )
    assert bad_report.name.endswith(".invalid.json") and bad_report.is_file()
    assert bad_figure.name.endswith(".invalid.json") and bad_figure.is_file()
    assert not rp.exists(), f"a stale grade (ece={first}) survived a refused re-run at {rp}"
    assert not fp.exists(), "figure data for a superseded grade survived a refused re-run"
    assert json.loads(bad_report.read_text())["gate"]["ece"] == 0.999


def test_an_accepted_rerun_removes_the_stale_diverted_artifacts(tmp_path) -> None:
    """The mirror of the divert (CodeRabbit r3): a rejected run's `.invalid.json` must not
    survive a later accepted one, or a reader sees a rejected grade with older numbers
    sitting beside the accepted report."""
    report = _report()
    rp, fp = tmp_path / "gate2_p3_ece.json", tmp_path / "gate2_figure_data.json"

    rejected = dict(report)
    rejected["gate"] = dict(report["gate"], ece=0.999)
    bad_report, bad_figure = G.write_outputs(
        rejected, report_path=rp, figure_data_path=fp, valid=False
    )
    assert bad_report.is_file() and bad_figure.is_file() and not rp.exists()

    good_report, good_figure = G.write_outputs(
        report, report_path=rp, figure_data_path=fp, valid=True
    )
    assert (good_report, good_figure) == (rp, fp) and rp.is_file() and fp.is_file()
    assert not bad_report.exists(), "a rejected grade survived beside the accepted report"
    assert not bad_figure.exists(), "rejected figure data survived beside the accepted one"
    assert json.loads(rp.read_text())["gate"]["ece"] == report["gate"]["ece"]


def test_the_figure_caption_formatter_never_raises_on_an_absent_number() -> None:
    """A run that could not fit a temperature is exactly the one whose figure is wanted (r3).

    `figure_data` derives `ece`, `ece_plugin` and `temperature` through `.get(...)` chains, so
    all three can be `None`, and `f"{None:.4f}"` raises mid-render. Only the formatter is
    exercised here: `plot_figures` itself needs matplotlib, which is in neither the local
    pytest env nor CI's pinned install list, so an `importorskip` would skip everywhere. The
    render is verified by hand under the `viz` env as a CLAUDE.md §8.5 manual gate.
    """
    assert G._fmt(0.005661656) == "0.0057"
    assert G._fmt(0) == "0.0000"
    assert G._fmt(1.140627, ".2f") == "1.14"
    for absent in (None, float("nan"), float("inf"), float("-inf"), True, False, "0.5", [1]):
        assert G._fmt(absent) == "n/a", absent

    # the fields it guards really can be absent: a report with no calibration block
    stripped = _report()
    stripped["gate"] = {k: v for k, v in stripped["gate"].items() if k != "calibration"}
    fig = G.figure_data(stripped)
    assert fig["in_distribution"]["temperature"] is None
    assert G._fmt(fig["in_distribution"]["temperature"]) == "n/a"


def test_a_perfectly_calibrated_unit_is_not_treated_as_absent(tmp_path) -> None:
    """An OOD ECE of exactly 0.0 is falsy (CodeRabbit r5).

    `or 0.0` / `or float("nan")` would erase a perfectly calibrated unit from the panel and
    collapse its sort key onto that of a genuinely missing value — a real outcome rendered as
    no outcome. Only the projection and the ordering are checked here; the render itself
    needs matplotlib (see the formatter test).
    """
    report = _report()
    fig = G.figure_data(report)
    zero, missing = dict(fig["ood_units"][0]), dict(fig["ood_units"][0])
    zero.update(unit="OrderZero", ood_ece=0.0, admissible=True, inadmissible_point=None)
    missing.update(unit="OrderNone", ood_ece=None, admissible=False, inadmissible_point=None)
    fig["ood_units"] = [zero, missing] + fig["ood_units"]

    # the SHIPPED selector, not a re-implementation — a local copy stays green when the real
    # one regresses, which is exactly what a sabotage of this fix demonstrated.
    assert G.ood_point(zero) == 0.0, "a perfectly calibrated unit has a value, not an absence"
    assert G.ood_point(missing) is None
    assert G.ood_point(zero) is not None and G.ood_point(missing) is None
    ordered = sorted(fig["ood_units"], key=G._ood_sort_key)
    assert {u["unit"] for u in ordered} == {u["unit"] for u in fig["ood_units"]}
    assert "OrderZero" in {u["unit"] for u in ordered}
    # and an inadmissible unit still plots its inadmissible_point
    inadm = dict(zero, unit="OrderInadm", admissible=False, ood_ece=None, inadmissible_point=0.0)
    assert G.ood_point(inadm) == 0.0


def test_the_bin_size_label_never_claims_a_count_some_bins_lack() -> None:
    """Equal-MASS bins differ in size when n is not divisible by the bin count."""
    assert G._bin_size_label([{"n": 203}, {"n": 203}]) == "203 rows each"
    assert G._bin_size_label([{"n": 203}, {"n": 202}, {"n": 203}]) == "202-203 rows"
    assert G._bin_size_label([]) == "no rows"
    # the real report's own bins, whichever shape they have, must describe themselves
    rows = _report()["gate"]["reliability"]
    counts = {int(r["n"]) for r in rows}
    label = G._bin_size_label(rows)
    assert ("each" in label) is (len(counts) == 1)


def test_truncation_is_recorded_only_when_the_list_was_really_cut() -> None:
    """`--max-units 60` on a 30-unit holdout truncates nothing (CodeRabbit r2).

    Recording it anyway turns `graded_every_loo_holdout_unit` FALSE on a run that graded
    every unit — a completeness clause failing on a complete run.
    """
    plan = [("OrderA", "PhylumOne", 40, 2.6), ("OrderB", "PhylumOne", 31, 1.4)]
    scores, rows_by_id = _phylum_scores(plan)
    call = dict(scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=4)

    for max_units in (None, 2, 5, 99):
        out = G.grade_ood_units(**call, max_units=max_units)
        assert out["n_units"] == 2, max_units
        assert out["truncated_to_n_units"] is None, (
            f"--max-units {max_units} recorded a truncation of a 2-unit holdout that did "
            "not happen"
        )
    cut = G.grade_ood_units(**call, max_units=1)
    assert cut["n_units"] == 1 and cut["truncated_to_n_units"] == 1
    assert set(cut["units"]) == {"OrderA"}


def test_the_concentration_indices_break_ties_the_way_the_report_reads_them() -> None:
    """First maximum wins — `(value, index)` tuples would pick the LAST (CodeRabbit r2).

    A near-separated posterior makes ties at the maximum reachable: 11 of the real report's
    15 bins carry a debiased gap of exactly 0.0.
    """
    tied = [
        {"n": 1, "weight": 0.5, "debiased_gap": 0.2, "acc": 0.5, "p_min": 0.0, "p_max": 0.4},
        {"n": 1, "weight": 0.5, "debiased_gap": 0.2, "acc": 0.5, "p_min": 0.6, "p_max": 1.0},
    ]
    conc = G._bin_concentration(tied, 0.2)
    assert conc["top_bin_index"] == 0, "a tie at the maximum contribution picked the last bin"
    assert conc["widest_bin_index"] == 0, "a tie at the widest span picked the last bin"
    # and the unambiguous case still selects the real maximum, not merely index 0
    clear = [
        {"n": 1, "weight": 0.5, "debiased_gap": 0.0, "acc": 1.0, "p_min": 0.0, "p_max": 0.1},
        {"n": 1, "weight": 0.5, "debiased_gap": 0.4, "acc": 0.5, "p_min": 0.1, "p_max": 1.0},
    ]
    sharp = G._bin_concentration(clear, 0.2)
    assert sharp["top_bin_index"] == 1 and sharp["widest_bin_index"] == 1


def test_the_figure_projection_carries_the_not_gated_markers() -> None:
    """The figure data holds `gate: 0.05` and the OOD ECEs in one document (CodeRabbit r2).

    A script reading only that file must still be able to tell that the OOD numbers are not
    graded against the in-distribution threshold sitting beside them.
    """
    report = _report()
    fig = G.figure_data(report)
    assert fig["gate"] == G.ECE_GATE
    assert fig["in_distribution"]["passes"] is report["gate"]["passes"]
    assert fig["ood_gated"] is False
    for key in ("ood_why_not_gated", "ood_by_phylum_is", "ood_macro_average_is"):
        assert fig[key] == report["ood"][key.removeprefix("ood_")], key
        assert fig[key], f"{key} is empty, so the guard says nothing"
    assert "not gated" in fig["ood_why_not_gated"] or "never gated" in fig["ood_why_not_gated"]


def test_a_shortfall_in_replicates_says_why_it_happened() -> None:
    """A unit whose CI used fewer replicates than requested must explain it (CodeRabbit r2).

    `block_bootstrap` drops replicates whose statistic is non-finite; with few blocks a draw
    can repeat one block until the leave-one-out kernel is undefined. Recorded so an auditor
    can tell rejected resamples from a truncated bootstrap.
    """
    # A SINGLETON block is what makes a replicate degenerate: the kernel leaves out by row,
    # so only a draw taken entirely from a one-row block leaves nothing to compare against.
    # The real report's `Pseudonocardiales` has block sizes [1, 2, 49] and lost 7 of 200 —
    # exactly 1/27, the chance of drawing its singleton three times. A fixture without a
    # singleton drops nothing, `requested == survived`, and this test would pass while
    # recording the survived count instead of the requested one.
    rng = __import__("random").Random(3)
    n = 30
    labels = [1] * 25 + [0] * 5
    row_ids = [f"OrderA-{i}" for i in range(n)]
    logits = [2.4 + 0.5 * rng.gauss(0.0, 1.0) for _ in range(n)]
    block_of = ["c0"] * (n - 1) + ["c1"]  # one singleton
    rows_by_id = {
        rid: {"_unit": "OrderA", "_phylum": "PhylumOne", "_block": blk}
        for rid, blk in zip(row_ids, block_of, strict=True)
    }
    scores = {
        "row_ids": row_ids,
        "logits": logits,
        "labels": labels,
        "rungs": None,
        "meta": {},
        "load": None,
    }
    out = G.grade_ood_units(
        scores=scores, rows_by_id=rows_by_id, temperature=1.0, n_boot=40, seed=3
    )
    unit = out["units"]["OrderA"]
    survived = unit["ci"]["n_boot"]
    assert survived < 40, "the fixture must actually LOSE replicates or this tests nothing"
    assert unit["n_boot_requested"] == 40, "the requested count was replaced by the survivors"
    assert unit["n_boot_dropped"] == 40 - survived
    assert survived + unit["n_boot_dropped"] == 40, "the accounting must close"
    assert "one-row block" in unit["n_boot_drop_reason"]
    # the count that explains the drop is recorded beside it, and it is NOT the inherited
    # `block_census.n_singleton_blocks` (which counts cluster-less rows, a different thing)
    assert unit["n_blocks_of_size_one"] == 1, "the fixture's one singleton must be counted"
    assert unit["n_blocks"] == 2
    assert "n_singleton_blocks" in unit["n_boot_drop_reason"], (
        "the reason must name the field it is NOT, or a reader cross-referencing the scores "
        "file reads 0 singletons beside a non-zero drop and sees a contradiction"
    )


def test_the_two_schema_versions_are_declared_to_be_different_schemas() -> None:
    """`schema_version` and `provenance.schema_version` version different things (r2)."""
    from tbox_finder import provenance as PROV

    report = _report()
    scope = report["schema_version_scope"]
    assert "report" in scope.lower() and "provenance" in scope.lower()
    assert repr(PROV.SCHEMA_VERSION) in scope, "the scope note must quote the actual envelope"
    assert report["schema_version"] == G.SCHEMA_VERSION


def test_the_figure_data_subcommand_reproduces_what_grade_writes(tmp_path) -> None:
    """Re-deriving the projection from a committed report must be byte-identical.

    That equality is what lets a metadata-only fix regenerate the figure data in seconds
    instead of re-running a 40-minute leave-one-out bootstrap — so it is pinned, not assumed.
    """
    report = _report()
    rp, fp = tmp_path / "gate2_p3_ece.json", tmp_path / "gate2_figure_data.json"
    G.write_outputs(report, report_path=rp, figure_data_path=fp, valid=True)
    from_grade = fp.read_text()

    out = tmp_path / "re-derived.json"
    assert G.main(["figure-data", "--report", str(rp), "--out", str(out)]) == 0
    assert out.read_text() == from_grade, "the subcommand and `grade` disagree on the projection"


def test_the_report_identifies_its_step_adr_and_prd_sections() -> None:
    report = _report()
    assert report["step"] == "P3-10"
    assert "D11" in report["adr"] and "D13" in report["adr"]
    assert "§2.3" in report["prd"] and "§12" in report["prd"]
    for key, bad in (("step", "P3-09"), ("schema_version", "9"), ("adr", "")):
        mutated = _report()
        mutated[key] = bad
        assert any(p.startswith(key) for p in G.validate_report(mutated))


def test_the_bin_concentration_block_says_how_narrow_the_evidence_is() -> None:
    """A near-separated posterior makes most bins trivially correct; the report must say so.

    The block is reconstructed here from the report's own reliability table, so a version
    that computed it from something else — or copied a plausible number — disagrees.
    """
    report = _report()
    conc = report["gate"]["bin_concentration"]
    rel = report["gate"]["reliability"]
    assert conc["n_bins"] == len(rel) == G.ECE_N_BINS
    assert conc["gated"] is False

    contributions = [b["weight"] * b["debiased_gap"] for b in rel]
    assert conc["top_bin_contribution"] == pytest.approx(max(contributions))
    assert conc["top_bin_index"] == contributions.index(max(contributions))
    # the shares must reconstruct the gated ECE, not merely look plausible
    assert sum(contributions) == pytest.approx(report["gate"]["ece"], abs=1e-12)
    assert conc["top_bin_share_of_ece"] == pytest.approx(max(contributions) / report["gate"]["ece"])
    assert 0.0 < conc["top_bin_share_of_ece"] <= 1.0

    spans = [b["p_max"] - b["p_min"] for b in rel]
    assert conc["widest_bin_span"] == pytest.approx(max(spans))
    assert conc["widest_bin_index"] == spans.index(max(spans))
    assert conc["n_bins_with_saturated_accuracy"] == sum(1 for b in rel if b["acc"] in (0.0, 1.0))
    assert conc["n_bins_with_zero_debiased_gap"] == sum(1 for b in rel if b["debiased_gap"] == 0.0)


def test_the_concentration_caveat_reaches_the_disclosures() -> None:
    report = _report()
    entry = [e for e in report["disclosures"] if "BIN-CONCENTRATED" in e]
    assert len(entry) == 1, "the concentration caveat must travel with the number"
    share = report["gate"]["bin_concentration"]["top_bin_share_of_ece"]
    assert f"{100 * share:.1f}%" in entry[0], "the disclosure must quote the measured share"


def test_the_plug_in_ece_is_reported_beside_the_gated_debiased_one() -> None:
    """P3-09 measured the debias term saturating below a real error; the plug-in is the
    upward-biased companion that makes an optimistic gated value visible."""
    report = _report()
    assert isinstance(report["gate"]["ece_plugin"], float)
    assert report["gate"]["ece_plugin"] >= report["gate"]["ece"] - 1e-12
    assert math.isfinite(report["gate"]["ece_ci"]["point"])
