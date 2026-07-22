"""P2-10c′-c-i — unit tests for the along-sequence candidate-caller.

The three boundary invariants that define the operator — a position is element-like at
``>=`` threshold (inclusive), runs bridge a gap of ``<=`` gap_merge (inclusive), a run is a
candidate at length ``>=`` min_span (inclusive) — each get a property test **plus** a
must-fire sabotage partner: an independent reference caller with that one decision flipped is
shown to produce a *different* answer on a crafted fixture, so the property test provably
bites (the reconcile / lora_config_exact anti-tautology lesson). A boundary test that no
wrong operator can fail is a tautology, not a guard.

Reference candidate spans are computed by ``_ref_candidates`` — a plain-Python caller with
the three decisions as knobs — never by calling the module under test. p_elem is controlled
exactly by setting the threshold equal to a value the fixture actually carries, so boundary
equality is exact by construction rather than up to a log/exp round-trip.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tbox_finder.infer import call as cc

# --------------------------------------------------------------------------------------
# fixture builders (stdlib / numpy only; independent of the module under test)
# --------------------------------------------------------------------------------------


def _log_probs_from_pelem(p_elem, *, dominant: int = 1) -> np.ndarray:
    """Build a finite ``(n, 8)`` log-array whose ``1 − exp(col0)`` reproduces ``p_elem``.

    Background mass is ``1 − p_elem``; the remainder is placed on class ``dominant`` (1..7)
    with a tiny finite floor on every other class, so the array is finite (the caller's
    finite-guard is satisfied) and ``dominant`` is the arg-max among the element classes.
    The rows need not sum to 1 — the caller reads only ``exp(col0)`` and the per-class
    arg-max, exactly as a real reconciled distribution would be consumed.
    """
    p_elem = np.asarray(p_elem, dtype=np.float64)
    n = p_elem.shape[0]
    lp = np.full((n, cc.NUM_CLASSES), np.log(1e-9), dtype=np.float64)
    lp[:, cc.BACKGROUND_INDEX] = np.log(np.clip(1.0 - p_elem, 1e-12, 1.0))
    lp[:, dominant] = np.log(np.clip(p_elem, 1e-12, 1.0))
    return lp


def _rand_p_elem(n: int, *, tag: str) -> np.ndarray:
    """Deterministic pseudo-random ``p_elem`` in [0, 1] — SHAKE-256, not numpy RNG.

    A ``numpy.random`` stream is not guaranteed stable across releases, and these values
    back exact reference-match assertions.
    """
    raw = hashlib.shake_256(tag.encode()).digest(n * 8)
    return np.array(
        [int.from_bytes(raw[i * 8 : (i + 1) * 8], "big") / 2.0**64 for i in range(n)],
        dtype=np.float64,
    )


def _ref_candidates(
    p_elem,
    *,
    threshold: float,
    min_span: int,
    gap_merge: int,
    ge: bool = True,
    gap_le: bool = True,
    span_ge: bool = True,
) -> list[tuple[int, int]]:
    """Independent reference caller returning ``[(start, end), ...]``.

    The three knobs ``ge`` / ``gap_le`` / ``span_ge`` select the boundary convention; the
    module under test must match ``ge=gap_le=span_ge=True``, and each sabotage flips exactly
    one of them to demonstrate the corresponding boundary decision is real.
    """
    p = list(p_elem)
    n = len(p)
    elem = [(v >= threshold) if ge else (v > threshold) for v in p]

    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if elem[i]:
            j = i
            while j < n and elem[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged:
            gap = start - merged[-1][1]
            bridge = (gap <= gap_merge) if gap_le else (gap < gap_merge)
            if bridge:
                merged[-1] = (merged[-1][0], end)
                continue
        merged.append((start, end))

    out: list[tuple[int, int]] = []
    for start, end in merged:
        keep = (end - start >= min_span) if span_ge else (end - start > min_span)
        if keep:
            out.append((start, end))
    return out


def _spans(candidates) -> list[tuple[int, int]]:
    return [(c.start, c.end) for c in candidates]


# --------------------------------------------------------------------------------------
# the element score
# --------------------------------------------------------------------------------------


def test_element_posterior_is_one_minus_background():
    p_target = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    lp = _log_probs_from_pelem(p_target)
    got = cc.element_posterior(lp)
    # 1 − exp(log(clip(1 − p))) reproduces p on [1e-12, 1 − 1e-12]; the endpoints clip.
    assert np.allclose(got, p_target, atol=1e-9)
    assert got.min() >= 0.0 and got.max() <= 1.0  # clipped to a valid score


def test_element_posterior_rejects_non_finite_and_wrong_shape():
    lp = _log_probs_from_pelem([0.5, 0.5])
    bad = lp.copy()
    bad[0, 0] = np.inf
    with pytest.raises(cc.CandidateError):
        cc.element_posterior(bad)
    with pytest.raises(cc.CandidateError):
        cc.element_posterior(np.zeros((3, cc.NUM_CLASSES + 1)))
    with pytest.raises(cc.CandidateError):
        cc.element_posterior(np.zeros((0, cc.NUM_CLASSES)))


# --------------------------------------------------------------------------------------
# reference match on random fixtures (the operator, end to end)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("min_span", [1, 3, 10])
@pytest.mark.parametrize("gap_merge", [0, 2, 8])
def test_matches_independent_reference_on_random_fixtures(threshold, min_span, gap_merge):
    lp = _log_probs_from_pelem(_rand_p_elem(300, tag=f"call|{threshold}|{min_span}|{gap_merge}"))
    p = cc.element_posterior(lp)  # the exact same array the caller computes internally
    got = _spans(
        cc.call_candidates(lp, threshold=threshold, min_span=min_span, gap_merge=gap_merge)
    )
    expected = _ref_candidates(p, threshold=threshold, min_span=min_span, gap_merge=gap_merge)
    assert got == expected


def test_the_random_fixture_is_expressive_not_degenerate():
    # Guards the reference-match test from passing vacuously on an all-empty / all-one call
    # set (the degenerate-fixture lesson). A mid fixture must yield several bounded runs.
    lp = _log_probs_from_pelem(_rand_p_elem(300, tag="expressive"))
    calls = cc.call_candidates(lp, threshold=0.5, min_span=3, gap_merge=2)
    assert 3 <= len(calls) <= 60
    assert any(c.length < 300 for c in calls)  # not one run spanning everything


# --------------------------------------------------------------------------------------
# boundary invariant 1 — threshold is inclusive (>=), sabotage-verified
# --------------------------------------------------------------------------------------


def test_threshold_boundary_is_inclusive_and_the_test_bites():
    # position 1 sits exactly at the threshold; only under `>=` is it element-like.
    lp = _log_probs_from_pelem([0.2, 0.6, 0.2])
    p = cc.element_posterior(lp)
    tau = float(p[1])  # threshold == a value the fixture carries → exact equality at pos 1
    got = _spans(cc.call_candidates(lp, threshold=tau, min_span=1, gap_merge=0))
    ref_ge = _ref_candidates(p, threshold=tau, min_span=1, gap_merge=0, ge=True)
    ref_gt = _ref_candidates(p, threshold=tau, min_span=1, gap_merge=0, ge=False)
    assert got == ref_ge == [(1, 2)]
    assert ref_ge != ref_gt  # MUST FIRE: a `>` operator would drop the boundary → []


# --------------------------------------------------------------------------------------
# boundary invariant 2 — gap-merge bridges a gap of <= gap_merge, sabotage-verified
# --------------------------------------------------------------------------------------


def test_gap_merge_is_inclusive_le_and_the_test_bites():
    # two element blocks separated by exactly 2 background positions.
    lp = _log_probs_from_pelem([0.9, 0.9, 0.1, 0.1, 0.9, 0.9])
    p = cc.element_posterior(lp)
    got = _spans(cc.call_candidates(lp, threshold=0.5, min_span=1, gap_merge=2))
    ref_le = _ref_candidates(p, threshold=0.5, min_span=1, gap_merge=2, gap_le=True)
    ref_lt = _ref_candidates(p, threshold=0.5, min_span=1, gap_merge=2, gap_le=False)
    assert got == ref_le == [(0, 6)]  # gap of exactly 2 is bridged into one candidate
    assert ref_lt == [(0, 2), (4, 6)]  # MUST FIRE: `<` would leave them split
    assert ref_le != ref_lt


def test_gap_merge_zero_bridges_nothing():
    lp = _log_probs_from_pelem([0.9, 0.9, 0.1, 0.9, 0.9])
    got = _spans(cc.call_candidates(lp, threshold=0.5, min_span=1, gap_merge=0))
    assert got == [(0, 2), (3, 5)]  # a 1-position gap is not bridged at gap_merge=0


# --------------------------------------------------------------------------------------
# boundary invariant 3 — min-span keeps a run of length >= min_span, sabotage-verified
# --------------------------------------------------------------------------------------


def test_min_span_is_inclusive_ge_and_the_test_bites():
    # a run of length exactly 3.
    lp = _log_probs_from_pelem([0.1, 0.9, 0.9, 0.9, 0.1])
    p = cc.element_posterior(lp)
    got = _spans(cc.call_candidates(lp, threshold=0.5, min_span=3, gap_merge=0))
    ref_ge = _ref_candidates(p, threshold=0.5, min_span=3, gap_merge=0, span_ge=True)
    ref_gt = _ref_candidates(p, threshold=0.5, min_span=3, gap_merge=0, span_ge=False)
    assert got == ref_ge == [(1, 4)]  # length exactly min_span is kept
    assert ref_gt == []  # MUST FIRE: `>` would drop the length-3 run
    assert ref_ge != ref_gt


# --------------------------------------------------------------------------------------
# degenerate sequences
# --------------------------------------------------------------------------------------


def test_all_background_yields_no_candidates():
    lp = _log_probs_from_pelem(np.full(50, 0.01))
    assert cc.call_candidates(lp, threshold=0.5, min_span=1, gap_merge=0) == []


def test_all_element_yields_one_spanning_candidate():
    lp = _log_probs_from_pelem(np.full(50, 0.99))
    calls = cc.call_candidates(lp, threshold=0.5, min_span=1, gap_merge=0)
    assert _spans(calls) == [(0, 50)]
    assert calls[0].length == 50


# --------------------------------------------------------------------------------------
# recall-favouring: NO required-element co-occurrence (the operator's defining choice)
# --------------------------------------------------------------------------------------


def test_single_element_run_is_called_without_co_occurrence():
    # a run driven by exactly ONE element class (Stem_I, index 1) and no other — a caller
    # that demanded canonical multi-element co-occurrence would reject it; the recall-
    # favouring operator (user decision 2026-07-22) calls it. This is the Tier-2N-protecting
    # behaviour: the flagship discovery class is the non-canonical T-box.
    lp = _log_probs_from_pelem(
        np.concatenate([np.full(10, 0.05), np.full(30, 0.95), np.full(10, 0.05)]), dominant=1
    )
    calls = cc.call_candidates(lp, threshold=0.5, min_span=5, gap_merge=0)
    assert _spans(calls) == [(10, 40)]
    assert calls[0].dominant_class == 1  # driven by a single element, still a candidate


def test_dominant_class_names_the_driving_element():
    for k in range(1, cc.NUM_CLASSES):
        lp = _log_probs_from_pelem(np.full(20, 0.9), dominant=k)
        calls = cc.call_candidates(lp, threshold=0.5, min_span=1, gap_merge=0)
        assert len(calls) == 1 and calls[0].dominant_class == k


# --------------------------------------------------------------------------------------
# contig ends flagged, never dropped (PRD §6 / ADR-0005 D3)
# --------------------------------------------------------------------------------------


def test_zero_flanked_candidate_is_kept_and_flagged():
    lp = _log_probs_from_pelem(np.full(30, 0.9))
    zf = np.zeros(30, dtype=bool)
    zf[:12] = True  # the first 12 positions were scored with contig-end pad context
    calls = cc.call_candidates(lp, zf, threshold=0.5, min_span=1, gap_merge=0)
    assert _spans(calls) == [(0, 30)]  # kept, not dropped
    assert calls[0].n_zero_flanked == 12  # flagged with the exact count

    # and with no zero_flanked supplied, the count is zero (interior-sequence default).
    calls_default = cc.call_candidates(lp, threshold=0.5, min_span=1, gap_merge=0)
    assert calls_default[0].n_zero_flanked == 0


def test_zero_flanked_shape_is_validated():
    lp = _log_probs_from_pelem(np.full(10, 0.9))
    with pytest.raises(cc.CandidateError):
        cc.call_candidates(lp, np.zeros(9, dtype=bool), threshold=0.5, min_span=1, gap_merge=0)


# --------------------------------------------------------------------------------------
# per-candidate strength stats
# --------------------------------------------------------------------------------------


def test_peak_and_mean_p_elem_are_reported():
    lp = _log_probs_from_pelem([0.1, 0.6, 0.9, 0.7, 0.1])
    calls = cc.call_candidates(lp, threshold=0.5, min_span=1, gap_merge=0)
    assert len(calls) == 1
    c = calls[0]
    assert c.start == 1 and c.end == 4
    assert c.peak_p_elem == pytest.approx(0.9, abs=1e-9)
    assert c.mean_p_elem == pytest.approx((0.6 + 0.9 + 0.7) / 3, abs=1e-9)


# --------------------------------------------------------------------------------------
# pin discipline — the three geometry knobs have NO default (cannot be taken by accident)
# --------------------------------------------------------------------------------------


def test_geometry_parameters_are_keyword_only_with_no_default():
    lp = _log_probs_from_pelem(np.full(10, 0.9))
    # omitting any of threshold / min_span / gap_merge is a TypeError, not a silent
    # production default (the min_sequences / PEFT-target_modules "no accidental value" rule).
    with pytest.raises(TypeError):
        cc.call_candidates(lp)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        cc.call_candidates(lp, threshold=0.5, min_span=1)  # type: ignore[call-arg]
    # and they are keyword-only: positional geometry is refused.
    with pytest.raises(TypeError):
        cc.call_candidates(lp, None, 0.5, 1, 0)  # type: ignore[misc]


def test_out_of_range_parameters_raise():
    lp = _log_probs_from_pelem(np.full(10, 0.9))
    for bad in (-0.01, 1.01):
        with pytest.raises(cc.CandidateError):
            cc.call_candidates(lp, threshold=bad, min_span=1, gap_merge=0)
    with pytest.raises(cc.CandidateError):
        cc.call_candidates(lp, threshold=0.5, min_span=0, gap_merge=0)
    with pytest.raises(cc.CandidateError):
        cc.call_candidates(lp, threshold=0.5, min_span=1, gap_merge=-1)


def test_call_is_deterministic():
    lp = _log_probs_from_pelem(_rand_p_elem(200, tag="determinism"))
    a = _spans(cc.call_candidates(lp, threshold=0.5, min_span=5, gap_merge=3))
    b = _spans(cc.call_candidates(lp, threshold=0.5, min_span=5, gap_merge=3))
    assert a == b
    assert a == sorted(a)  # ascending start order


# --------------------------------------------------------------------------------------
# the sweep — consistency with the single-point caller + grid shape
# --------------------------------------------------------------------------------------


def test_sweep_rows_match_per_point_calls_and_grid_shape():
    lp = _log_probs_from_pelem(_rand_p_elem(400, tag="sweep"))
    rows = cc.sweep_candidate_counts(lp)
    assert len(rows) == (
        len(cc.PROVISIONAL_THRESHOLD_GRID)
        * len(cc.PROVISIONAL_MIN_SPAN_GRID)
        * len(cc.PROVISIONAL_GAP_MERGE_GRID)
    )
    for row in rows:
        direct = cc.call_candidates(
            lp,
            threshold=row["threshold"],
            min_span=row["min_span"],
            gap_merge=row["gap_merge"],
        )
        assert row["n_candidates"] == len(direct)
        assert row["total_candidate_nt"] == sum(c.length for c in direct)
        assert row["n_zero_flanked_candidates"] == sum(1 for c in direct if c.n_zero_flanked > 0)


def test_sweep_validates_zero_flanked_shape():
    lp = _log_probs_from_pelem(np.full(20, 0.9))
    with pytest.raises(cc.CandidateError):
        cc.sweep_candidate_counts(lp, np.zeros(19, dtype=bool))


def test_sweep_counts_zero_flanked_candidates_per_point():
    # a fully zero-flanked short sequence: every called candidate touches pad context, so
    # each grid point's n_zero_flanked_candidates equals its n_candidates.
    lp = _log_probs_from_pelem(np.full(40, 0.95))
    zf = np.ones(40, dtype=bool)
    rows = cc.sweep_candidate_counts(lp, zf, thresholds=[0.5], min_spans=[1], gap_merges=[0])
    assert rows[0]["n_candidates"] == 1
    assert rows[0]["n_zero_flanked_candidates"] == 1


def test_provisional_grids_pin_nothing_and_are_nonempty():
    # they are sizing scaffolding, not a pinned operating point — just assert they exist and
    # are usable; ADR-0005 D3 owns the frozen production values, at the phase gate.
    assert len(cc.PROVISIONAL_THRESHOLD_GRID) >= 2
    assert all(0.0 <= t <= 1.0 for t in cc.PROVISIONAL_THRESHOLD_GRID)
    assert all(m >= 1 for m in cc.PROVISIONAL_MIN_SPAN_GRID)
    assert all(g >= 0 for g in cc.PROVISIONAL_GAP_MERGE_GRID)
