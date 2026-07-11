"""Unit tests for the P0-27 OOD-ECE min-N coverage simulation (`tbox_finder.coverage`).

The pinned ``OOD_ECE_MIN_N`` (ADR-0005 D13) decides which deployment corpora are
adjudicable versus inconclusive-by-rule, so it is load-bearing for the §2.3 rollup
shape: each predicate is locked here with a **clean-pass AND a boundary/fail** case
(the guard is proven to bite — CLAUDE.md §8.7/§10.3), and the "predominantly
inconclusive" determination is exercised in *both* directions (so it is not
hard-wired to a single verdict).

Layering matches the repo pattern: the pure predicates are stdlib-only and run in
the bare CI env; the real-table read, the seeded bootstrap, and the P0-26
cross-check are numpy/pandas-gated (``importorskip``) and run only where the
partition parquet is materialized. The full-corpus recompute is a manual
step-local gate (CLAUDE.md §8.5), not CI.
"""

import json
from pathlib import Path

import pytest

from tbox_finder import coverage, power

_SPLIT_TABLE = Path("data/interim/splits/split_assignments.parquet")
_POWER_REPORT = Path("data/processed/audits/power_budget_report.json")


# --------------------------------------------------------------------------- #
# Frozen floor + internal consistency with the recall floor (ADR-0005 D13/D18)
# --------------------------------------------------------------------------- #


def test_ood_ece_floor_pinned_at_20():
    assert coverage.OOD_ECE_MIN_N == 20


def test_ood_ece_floor_internally_consistent_with_recall_floor():
    # D13's admissibility floor coheres with the P0-26 recall floor (one number
    # across the eval contract). A drift here should fail loud.
    assert coverage.OOD_ECE_MIN_N == power.MIN_REAL_HOMOLOG_N


def test_named_zero_positive_clades_nonempty_and_include_archaea():
    assert coverage.NAMED_ZERO_POSITIVE_CLADES  # non-empty
    assert "Archaea" in coverage.NAMED_ZERO_POSITIVE_CLADES


# --------------------------------------------------------------------------- #
# orders_clearing_floor / adjudicable_fraction / classify_order
# --------------------------------------------------------------------------- #


def test_orders_clearing_floor_selects_and_sorts():
    counts = {"b": 20, "a": 50, "c": 19}
    assert coverage.orders_clearing_floor(counts, floor=20) == ["a", "b"]  # sorted, c excluded


def test_orders_clearing_floor_none_clear_when_floor_above_max():
    assert coverage.orders_clearing_floor({"a": 19, "b": 5}, floor=20) == []


def test_adjudicable_fraction_clean_and_boundary():
    counts = {"a": 20, "b": 19, "c": 100}
    assert coverage.adjudicable_fraction(counts, floor=20) == pytest.approx(2 / 3)
    # N == floor is adjudicable (>=), N == floor-1 is not — boundary bites.
    assert coverage.adjudicable_fraction({"x": 20}, floor=20) == 1.0
    assert coverage.adjudicable_fraction({"x": 19}, floor=20) == 0.0


def test_adjudicable_fraction_empty_is_zero():
    assert coverage.adjudicable_fraction({}, floor=20) == 0.0


def test_classify_order_boundary():
    assert coverage.classify_order(20, floor=20) == "adjudicable"
    assert coverage.classify_order(19, floor=20) == "sub_min_n_inconclusive"


# --------------------------------------------------------------------------- #
# floor_sweep_coverage — monotone non-increasing in the floor
# --------------------------------------------------------------------------- #


def test_floor_sweep_coverage_monotone_non_increasing():
    counts = {"a": 12, "b": 20, "c": 35, "d": 60, "e": 5}
    sweep = coverage.floor_sweep_coverage(counts, sweep=(10, 15, 20, 30, 50))
    n_clearing = [sweep[str(f)]["n_clearing"] for f in (10, 15, 20, 30, 50)]
    assert n_clearing == [4, 3, 3, 2, 1]  # decreasing as the floor rises
    assert all(a >= b for a, b in zip(n_clearing, n_clearing[1:], strict=False))
    assert sweep["10"]["n_orders"] == 5


# --------------------------------------------------------------------------- #
# verdict_vector_shape + modal-shape determination (both directions)
# --------------------------------------------------------------------------- #


def test_verdict_vector_shape_counts():
    counts = {"a": 20, "b": 19, "c": 100, "d": 3}  # 2 adjudicable, 2 sub-min-N
    shape = coverage.verdict_vector_shape(counts, n_named_zero_positive_clades=4, floor=20)
    assert shape["adjudicable"] == 2
    assert shape["sub_min_n_inconclusive"] == 2
    assert shape["zero_positive_inconclusive_named_clades"] == 4
    assert shape["inconclusive_by_rule_total"] == 6


def test_modal_shape_inconclusive_when_named_zero_clades_present():
    # The real case: named zero-positive superclades present → inconclusive is modal
    # even if adjudicable orders outnumber sub-min-N ones.
    counts = {f"o{i}": 100 for i in range(30)}  # 30 adjudicable, 0 sub-min-N
    shape = coverage.verdict_vector_shape(counts, n_named_zero_positive_clades=4, floor=20)
    assert coverage.modal_shape_is_inconclusive(shape) is True
    assert coverage.predicted_gate3_modal_shape(shape) == "discovery-predominantly-inconclusive"


def test_modal_shape_adjudicable_predominant_when_no_zero_clades_and_adjudicable_dominates():
    # Guard bites: with NO zero-positive superclade and adjudicable strictly dominating,
    # the determination flips → not hard-wired to "inconclusive".
    counts = {f"o{i}": 100 for i in range(30)}  # 30 adjudicable, 0 sub-min-N
    shape = coverage.verdict_vector_shape(counts, n_named_zero_positive_clades=0, floor=20)
    assert coverage.modal_shape_is_inconclusive(shape) is False
    assert coverage.predicted_gate3_modal_shape(shape) == "adjudicable-predominant"


def test_modal_shape_inconclusive_when_sub_min_n_dominates_even_without_zero_clades():
    counts = {"a": 100, **{f"s{i}": 1 for i in range(10)}}  # 1 adjudicable, 10 sub-min-N
    shape = coverage.verdict_vector_shape(counts, n_named_zero_positive_clades=0, floor=20)
    assert coverage.modal_shape_is_inconclusive(shape) is True


# --------------------------------------------------------------------------- #
# Real-partition tier (pandas/numpy-gated) — non-fabrication + P0-26 cross-check
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _SPLIT_TABLE.exists(), reason="split-assignment parquet not materialized")
def test_real_partition_43_orders_30_clear_matches_p0_26():
    pytest.importorskip("pandas")
    order_counts, order_phylum, phylum_counts = coverage._read_heldout_order_counts(_SPLIT_TABLE)
    # The signed P0-26 result: 43 resolved held-out orders, 30 clear the min-N floor.
    assert len(order_counts) == 43
    assert len(coverage.orders_clearing_floor(order_counts, coverage.OOD_ECE_MIN_N)) == 30
    # Measured absence of the named zero-positive superclades from the resolved phyla.
    present = {p for p in order_phylum.values() if p}
    assert not (present & set(coverage.NAMED_ZERO_POSITIVE_CLADES))
    # Only bacterial phyla carry positives (no archaeal phylum present).
    assert phylum_counts  # non-empty


@pytest.mark.skipif(
    not (_SPLIT_TABLE.exists() and _POWER_REPORT.exists()),
    reason="partition parquet or P0-26 report not materialized",
)
def test_per_order_n_matches_power_budget_report_exactly():
    pytest.importorskip("pandas")
    order_counts, _, _ = coverage._read_heldout_order_counts(_SPLIT_TABLE)
    cc = coverage._cross_check_power_report(order_counts, _POWER_REPORT)
    assert cc["available"] is True
    assert cc["per_order_n_matches"] is True
    assert cc["n_mismatched_orders"] == 0


@pytest.mark.skipif(not _SPLIT_TABLE.exists(), reason="split-assignment parquet not materialized")
def test_bootstrap_is_seed_reproducible():
    pytest.importorskip("numpy")
    order_counts, _, _ = coverage._read_heldout_order_counts(_SPLIT_TABLE)
    a = coverage._bootstrap_fraction_ci(order_counts, coverage.OOD_ECE_MIN_N, n_boot=200, seed=42)
    b = coverage._bootstrap_fraction_ci(order_counts, coverage.OOD_ECE_MIN_N, n_boot=200, seed=42)
    assert a == b  # deterministic under a fixed seed
    assert 0.0 <= a["fraction_lo"] <= a["fraction_mean"] <= a["fraction_hi"] <= 1.0


@pytest.mark.skipif(not _SPLIT_TABLE.exists(), reason="split-assignment parquet not materialized")
def test_build_report_has_gate_fields_and_inconclusive_modal_shape():
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    order_counts, order_phylum, phylum_counts = coverage._read_heldout_order_counts(_SPLIT_TABLE)
    boot = coverage._bootstrap_fraction_ci(
        order_counts, coverage.OOD_ECE_MIN_N, n_boot=200, seed=42
    )
    cc = coverage._cross_check_power_report(order_counts, _POWER_REPORT)
    report = coverage.build_report(order_counts, order_phylum, phylum_counts, boot, cc)
    # Validation-gate fields (imp.md P0-27): count of clearing orders, adjudicable +
    # inconclusive-by-rule fractions, the verdict-vector shape, and the modal outcome.
    assert report["n_orders_clearing_floor"] == 30
    assert report["n_orders_with_heldout"] == 43
    assert 0.0 <= report["adjudicable_fraction_among_positive_bearing_orders"] <= 1.0
    assert report["inconclusive_by_rule"]["n_sub_min_n_positive_bearing_orders"] == 13
    assert report["verdict_vector_shape"]["modal_shape_is_inconclusive"] is True
    assert report["predicted_gate3_modal_shape"] == "discovery-predominantly-inconclusive"
    assert report["internal_consistency"]["consistent"] is True
    # JSON-serializable with sorted keys (matches the on-disk contract).
    json.dumps(report, sort_keys=True)
