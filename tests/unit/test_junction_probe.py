"""Gate-logic tests for the junction probe (P2-10d′-b, ADR-0005 A7 pin 7).

These test the *rules*, not the measurement: that a malformed report fails closed, that no
clause can be vacuously TRUE when the evidence is what is missing, that the must-fire
power clause actually bites, and that the tier-2 model-side rule is the comparison A7 pins
rather than a threshold.
"""

from __future__ import annotations

import math

import pytest

from tbox_finder.eval.junction_probe import (
    NULL_BAND_SIGMA,
    JunctionProbeError,
    auroc_null_stderr,
    cv_auroc,
    deviation,
    junction_clauses,
    kmer_frequencies,
    model_side_pass,
)


def _report(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "auroc_null_unspliced_vs_unspliced": 0.505,
        "auroc_junction_control_vs_plain": 0.498,
        "auroc_decoy_vs_junction_control": 0.870,
        "null_stderr": 0.008,
        "n_plain": 2500,
        "n_null_reference": 2500,
        "n_control": 2500,
        "n_decoy": 2500,
        "arms_matched": True,
        "null_band_sigma": NULL_BAND_SIGMA,
    }
    base.update(over)
    return base


# ── the happy path ───────────────────────────────────────────────────────────────────
def test_a_clean_measurement_passes_every_clause() -> None:
    """MUST FIRE, or every failure test below is vacuous."""
    assert all(junction_clauses(_report()).values())


# ── fail closed ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, {}, [], "report", 3])
def test_a_missing_or_malformed_report_fails_closed(bad: object) -> None:
    clauses = junction_clauses(bad)  # type: ignore[arg-type]
    assert set(clauses) and not any(clauses.values())


@pytest.mark.parametrize(
    "key",
    [
        "auroc_null_unspliced_vs_unspliced",
        "auroc_junction_control_vs_plain",
        "auroc_decoy_vs_junction_control",
        "null_stderr",
        "n_plain",
        "n_control",
    ],
)
def test_every_clause_is_false_when_its_evidence_is_absent(key: str) -> None:
    """[[clauses-must-guard-emptiness]] — a clause derived from a missing quantity must
    not be TRUE exactly when the quantity is missing."""
    report = _report()
    report.pop(key)
    assert not any(junction_clauses(report).values())


def test_a_bool_is_not_accepted_as_a_count() -> None:
    """`isinstance(True, int)` is True in Python, so a bool leaking into a count field
    would sail through a naive numeric check ([[gate-clauses-need-re-derivation]])."""
    assert not any(junction_clauses(_report(n_control=True)).values())


def test_a_bool_is_not_accepted_as_an_auroc() -> None:
    assert not any(junction_clauses(_report(auroc_junction_control_vs_plain=True)).values())


@pytest.mark.parametrize("n", [0, -1])
def test_an_empty_arm_fails_closed(n: int) -> None:
    assert not any(junction_clauses(_report(n_decoy=n)).values())


def test_a_non_finite_stderr_fails_closed() -> None:
    assert not any(junction_clauses(_report(null_stderr=math.inf)).values())


# ── the must-fire power clause ───────────────────────────────────────────────────────
def test_a_powerless_probe_is_reported_as_powerless_not_as_clean() -> None:
    """The whole point. If the probe cannot separate a decoy-bearing window from its own
    matched control, its verdict on the junction is not evidence of absence — and the two
    situations produce identical junction numbers."""
    clauses = junction_clauses(_report(auroc_decoy_vs_junction_control=0.501))
    assert clauses["junction_within_null_band"] is True
    assert clauses["probe_can_discriminate"] is False
    assert not all(clauses.values())


def test_a_separable_junction_fails_the_band_clause() -> None:
    clauses = junction_clauses(_report(auroc_junction_control_vs_plain=0.83))
    assert clauses["probe_can_discriminate"] is True
    assert clauses["junction_within_null_band"] is False


def test_the_band_is_relative_to_the_MEASURED_null_not_to_0_5() -> None:
    """A junction arm at 0.53 passes when the null itself sits at 0.53, and fails when the
    null sits at 0.50 — the reference travels with the measurement."""
    assert junction_clauses(
        _report(auroc_null_unspliced_vs_unspliced=0.53, auroc_junction_control_vs_plain=0.53)
    )["junction_within_null_band"]
    assert not junction_clauses(
        _report(auroc_null_unspliced_vs_unspliced=0.50, auroc_junction_control_vs_plain=0.60)
    )["junction_within_null_band"]


def test_separability_below_chance_counts_as_separability() -> None:
    """AUROC is symmetric about 0.5: a 0.17 junction arm is as readable as 0.83, and a
    one-sided `auroc > threshold` check would call it clean."""
    assert not junction_clauses(_report(auroc_junction_control_vs_plain=0.17))[
        "junction_within_null_band"
    ]


def test_unmatched_arms_fail_even_when_the_aurocs_look_perfect() -> None:
    clauses = junction_clauses(_report(arms_matched=False))
    assert clauses["arms_are_matched"] is False
    assert not all(clauses.values())


def test_arms_matched_must_be_literally_true() -> None:
    """A truthy non-True (a count, a string) must not satisfy the clause."""
    for truthy in (1, "yes", [1]):
        assert junction_clauses(_report(arms_matched=truthy))["arms_are_matched"] is False


# ── helpers ──────────────────────────────────────────────────────────────────────────
def test_deviation_is_symmetric_about_chance() -> None:
    assert deviation(0.8) == pytest.approx(deviation(0.2))


def test_auroc_null_stderr_matches_hanley_mcneil() -> None:
    assert auroc_null_stderr(2500, 2500) == pytest.approx(
        math.sqrt(5001 / (12 * 2500 * 2500)), rel=1e-12
    )


@pytest.mark.parametrize("n_pos,n_neg", [(0, 10), (10, 0), (0, 0)])
def test_auroc_null_stderr_is_infinite_for_an_empty_arm(n_pos: int, n_neg: int) -> None:
    assert math.isinf(auroc_null_stderr(n_pos, n_neg))


def test_kmer_frequencies_normalise_and_drop_non_acgt() -> None:
    m = kmer_frequencies(["ACGTACGT", "NNNNNNNN"], k=2)
    assert m.shape == (2, 16)
    assert m[0].sum() == pytest.approx(1.0)
    # An all-N row has no countable k-mer; it must be all-zero, not renormalised noise.
    assert m[1].sum() == pytest.approx(0.0)


def test_kmer_frequencies_refuse_degenerate_input() -> None:
    with pytest.raises(JunctionProbeError):
        kmer_frequencies([], k=4)
    with pytest.raises(JunctionProbeError):
        kmer_frequencies(["ACGT"], k=0)


def test_cv_auroc_refuses_an_arm_too_small_to_cross_validate() -> None:
    with pytest.raises(JunctionProbeError, match="at least 2"):
        cv_auroc(["ACGT"], ["ACGT", "TGCA"], k=2, seed=1)


# ── tier 2: the post-run rule, pinned before the run produces the numbers ────────────
def test_model_side_pass_requires_the_decoy_to_beat_its_own_junction_null() -> None:
    assert model_side_pass(auroc_decoy_vs_control=0.84, auroc_control_vs_plain=0.51)
    assert not model_side_pass(auroc_decoy_vs_control=0.55, auroc_control_vs_plain=0.80)


def test_model_side_pass_is_not_satisfied_by_a_tie() -> None:
    assert not model_side_pass(auroc_decoy_vs_control=0.70, auroc_control_vs_plain=0.70)


def test_model_side_pass_reads_below_chance_separation_as_separation() -> None:
    """A junction arm at 0.20 is strongly readable; a one-sided comparison would score it
    as 0.20 < 0.65 and pass a run whose junction cue was the strongest signal present."""
    assert not model_side_pass(auroc_decoy_vs_control=0.65, auroc_control_vs_plain=0.20)
