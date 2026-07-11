"""Unit tests for union-prior loci-masking + the ADR-0006 D11 spare rule (P0-30).

Stdlib-only (no pandas/numpy) so they run in the bare CI ``test`` env. Every guard
has a clean-pass AND a failing case so it is proven to bite (CLAUDE.md §8.7/§10.3).
The load-bearing gate is ``test_spare_rule_tier2n_locus_excluded_from_mining``.
"""

from __future__ import annotations

import math

import pytest

from tbox_finder import masking


# --------------------------------------------------------------------------- #
# Locus geometry
# --------------------------------------------------------------------------- #
def test_normalize_locus_forward_and_reverse():
    assert masking.normalize_locus(10, 20) == (10, 20)
    assert masking.normalize_locus(20, 10) == (10, 20)  # reverse strand folded
    assert masking.normalize_locus(5, 5) == (5, 5)


def test_intervals_overlap():
    assert masking.intervals_overlap(1, 5, 4, 9)  # touching-inside
    assert masking.intervals_overlap(1, 5, 5, 9)  # single-point touch
    assert not masking.intervals_overlap(1, 5, 6, 9)  # disjoint
    assert masking.intervals_overlap(1, 100, 40, 50)  # nested


# --------------------------------------------------------------------------- #
# LocusIndex.is_masked — clean-pass AND leaky-fail
# --------------------------------------------------------------------------- #
def _index():
    # accession X: locus 1000-1100 (forward) and 5000-4900 (reverse strand)
    return masking.LocusIndex.from_records([("X", 1000, 1100), ("X", 5000, 4900), ("Y", 200, 300)])


def test_is_masked_overlapping_locus_is_masked():
    idx = _index()
    assert idx.is_masked("X", 1050, 1080)  # inside 1000-1100
    assert idx.is_masked("X", 1090, 1200)  # straddles the right edge
    assert idx.is_masked("X", 4950, 4950)  # inside the reverse-strand 4900-5000


def test_is_masked_clean_window_is_not_masked():
    idx = _index()
    assert not idx.is_masked("X", 2000, 2100)  # between the two X loci
    assert not idx.is_masked("X", 1200, 1300)  # right of the first locus, no flank


def test_is_masked_flank_brings_near_miss_into_masking():
    idx = _index()
    # 1120-1140 is 20 nt right of locus end 1100: clean at flank 0, masked at flank 50.
    assert not idx.is_masked("X", 1120, 1140, flank=0)
    assert idx.is_masked("X", 1120, 1140, flank=50)


def test_is_masked_no_or_unknown_accession_not_masked():
    idx = _index()
    assert not idx.is_masked(None, 1050, 1080)  # synthetic decoy, no coords
    assert not idx.is_masked(float("nan"), 1050, 1080)
    assert not idx.is_masked("Z", 1050, 1080)  # accession absent from the index


def test_is_masked_negative_flank_raises():
    with pytest.raises(ValueError):
        _index().is_masked("X", 1050, 1080, flank=-1)


def test_locus_index_merges_adjacent_intervals():
    # 100-200 and 201-300 are adjacent → one merged span; a query across the seam masks.
    idx = masking.LocusIndex.from_records([("A", 100, 200), ("A", 201, 300)])
    assert idx.n_intervals == 1
    assert idx.is_masked("A", 195, 205)


def test_mask_a_pool_removes_only_contaminated_records():
    # A coordinate-carrying pool (as P2 mined windows would be): the guard must
    # remove the record overlapping a known locus and keep the clean one.
    idx = _index()
    pool = [
        {"accession": "X", "start": 1050, "end": 1080},  # contaminated
        {"accession": "X", "start": 3000, "end": 3100},  # clean
        {"accession": None, "start": 1050, "end": 1080},  # synthetic → never masked
    ]
    kept = [r for r in pool if not idx.is_masked(r["accession"], r["start"], r["end"], flank=50)]
    assert len(kept) == 2
    assert {r["start"] for r in kept} == {3000, 1050}
    assert all(r["accession"] != "X" or r["start"] == 3000 for r in kept)


# --------------------------------------------------------------------------- #
# The spare rule — the load-bearing anti-mimicry gate
# --------------------------------------------------------------------------- #
def test_spare_rule_tier2n_locus_excluded_from_mining():
    # A CM-invisible Tier-2N locus fails every canonical-architecture predicate but
    # passes a model-independent disjunct (here R-scape covariation) → it MUST be
    # excluded from the mining pool so aggressive mining cannot train it away.
    assert masking.spare_rule_excludes_from_mining(
        relaxed_architecture=False,
        any_helix_rscape=True,
        downstream_aaRS_synteny=False,
    )


def test_spare_rule_each_disjunct_independently_excludes():
    assert masking.spare_rule_excludes_from_mining(relaxed_architecture=True)
    assert masking.spare_rule_excludes_from_mining(any_helix_rscape=True)
    assert masking.spare_rule_excludes_from_mining(downstream_aaRS_synteny=True)


def test_spare_rule_plain_negative_is_mineable():
    # No disjunct fires and no Stage-2 exists (P2 round) → the candidate is a genuine
    # negative, eligible for the mining pool.
    assert not masking.spare_rule_excludes_from_mining()
    assert not masking.spare_rule_excludes_from_mining(
        relaxed_architecture=False, any_helix_rscape=False, downstream_aaRS_synteny=False
    )


def test_spare_rule_stage2_posterior_p3_round():
    # P3 round: a high Stage-2 posterior also excludes; below threshold does not;
    # a posterior with no threshold (P2 round, no Stage-2) is inert.
    assert masking.spare_rule_excludes_from_mining(stage2_posterior=0.95, stage2_threshold=0.9)
    assert not masking.spare_rule_excludes_from_mining(stage2_posterior=0.80, stage2_threshold=0.9)
    assert not masking.spare_rule_excludes_from_mining(stage2_posterior=0.99)  # no threshold


def test_spare_rule_input_validation():
    with pytest.raises(TypeError):
        masking.spare_rule_excludes_from_mining(relaxed_architecture=1)  # not bool
    with pytest.raises(ValueError):
        masking.spare_rule_excludes_from_mining(stage2_posterior=1.5, stage2_threshold=0.9)
    with pytest.raises(TypeError):
        masking.spare_rule_excludes_from_mining(stage2_posterior=True, stage2_threshold=0.9)


# --------------------------------------------------------------------------- #
# Residual-contamination report
# --------------------------------------------------------------------------- #
def test_residual_contamination_report():
    rep = masking.residual_contamination_report(
        n_union_total=100,
        n_union_maskable=97,
        pool_mask_counts={"gc_background": 0, "leader_decoy": 2},
    )
    assert rep["union_denominator"] == 100
    assert rep["union_residual_no_coords"] == 3
    assert math.isclose(rep["residual_contamination_fraction"], 0.03)
    assert rep["total_pool_records_masked"] == 2


def test_residual_contamination_report_validation():
    with pytest.raises(ValueError):
        masking.residual_contamination_report(
            n_union_total=10, n_union_maskable=11, pool_mask_counts={}
        )
    with pytest.raises(ValueError):
        masking.residual_contamination_report(
            n_union_total=-1, n_union_maskable=0, pool_mask_counts={}
        )


def test_residual_contamination_empty_union_no_zero_division():
    rep = masking.residual_contamination_report(
        n_union_total=0, n_union_maskable=0, pool_mask_counts={}
    )
    assert rep["residual_contamination_fraction"] == 0.0
