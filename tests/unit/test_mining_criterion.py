"""P2-10e — unit tests for the provisional mining-internal detection criterion.

Numpy tier (the criterion reuses ``infer/call.py``). The recovery rule is checked against
:func:`tbox_finder.infer.call.call_candidates` directly (so a drift in the wiring is caught,
not re-asserted), and the degenerate-criterion guard is checked on **identity** — a
right-count / wrong-members recovery is still degenerate — with the count-preserving failure
mode exercised explicitly (the symmetric-count-fixture lesson). ADR-0005 A9 Pin 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from tbox_finder.eval import mining_criterion as mc
from tbox_finder.infer.call import call_candidates

NUM_CLASSES = 8


def _log_probs_from_pelem(p_elem: np.ndarray) -> np.ndarray:
    """``(n, 8)`` log-posterior with ``1 − P(background) == p_elem`` per position.

    Background holds ``1 − p_elem``; the 7 element classes split ``p_elem`` evenly. Clipped
    off the exact 0/1 endpoints so the log is finite (``call.element_posterior`` re-clips).
    """
    p_elem = np.asarray(p_elem, dtype=np.float64)
    n = p_elem.shape[0]
    eps = 1e-9
    p_bg = np.clip(1.0 - p_elem, eps, 1.0 - eps)
    per_elem = np.clip(p_elem, eps, 1.0 - eps) / (NUM_CLASSES - 1)
    lp = np.empty((n, NUM_CLASSES), dtype=np.float64)
    lp[:, 0] = np.log(p_bg)
    lp[:, 1:] = np.log(per_elem)[:, None]
    return lp


def _with_run(n: int, run_start: int, run_len: int, *, hi: float = 0.99, lo: float = 0.01):
    """length-``n`` p_elem: ``hi`` inside ``[run_start, run_start+run_len)``, else ``lo``."""
    p = np.full(n, lo, dtype=np.float64)
    p[run_start : run_start + run_len] = hi
    return p


# --------------------------------------------------------------------------- #
# The pinned provisional values (ADR-0005 A9 Pin 3) — locked so a change is conscious.
# --------------------------------------------------------------------------- #
def test_provisional_criterion_pins_are_the_signed_values():
    assert mc.PROVISIONAL_THRESHOLD == 0.9
    assert mc.PROVISIONAL_MIN_SPAN == 50
    assert mc.PROVISIONAL_GAP_MERGE == 10


# --------------------------------------------------------------------------- #
# sequence_is_recovered ≡ call_candidates(...) >= 1 under the pinned triple.
# --------------------------------------------------------------------------- #
def test_recovered_iff_call_candidates_nonempty_under_the_pins():
    lp = _log_probs_from_pelem(_with_run(300, 100, 120))  # a 120-nt strong element run
    expected = (
        len(
            call_candidates(
                lp,
                None,
                threshold=mc.PROVISIONAL_THRESHOLD,
                min_span=mc.PROVISIONAL_MIN_SPAN,
                gap_merge=mc.PROVISIONAL_GAP_MERGE,
            )
        )
        >= 1
    )
    assert mc.sequence_is_recovered(lp) is expected is True


def test_all_background_is_not_recovered():
    lp = _log_probs_from_pelem(np.full(300, 0.01))
    assert mc.sequence_is_recovered(lp) is False


def test_run_shorter_than_min_span_is_not_recovered():
    # A 30-nt element run is below PROVISIONAL_MIN_SPAN (50) ⇒ no called locus ⇒ not recovered.
    lp = _log_probs_from_pelem(_with_run(300, 100, 30))
    assert mc.sequence_is_recovered(lp) is False
    # Sabotage partner: the very same run IS recovered once min_span drops below it, proving
    # the negative above is the min_span cut biting, not an all-background fixture.
    assert (
        len(call_candidates(lp, None, threshold=mc.PROVISIONAL_THRESHOLD, min_span=20, gap_merge=0))
        >= 1
    )


# --------------------------------------------------------------------------- #
# recovered_ids — identity over the probe id space.
# --------------------------------------------------------------------------- #
def test_recovered_ids_returns_exactly_the_recovered_keys():
    recovered_lp = _log_probs_from_pelem(_with_run(300, 50, 120))
    background_lp = _log_probs_from_pelem(np.full(300, 0.01))
    scored = {
        "v_hit_a": (recovered_lp, None),
        "v_hit_b": (recovered_lp, None),
        "v_miss_a": (background_lp, None),
        "v_miss_b": (background_lp, None),
    }
    assert mc.recovered_ids(scored) == {"v_hit_a", "v_hit_b"}


def test_recovered_ids_fixture_is_expressive_not_degenerate():
    # Guards the identity test above from passing vacuously on an all-hit / all-miss set.
    recovered_lp = _log_probs_from_pelem(_with_run(300, 50, 120))
    background_lp = _log_probs_from_pelem(np.full(300, 0.01))
    got = mc.recovered_ids({"a": (recovered_lp, None), "b": (background_lp, None)})
    assert 0 < len(got) < 2  # a proper, non-empty subset


# --------------------------------------------------------------------------- #
# Degenerate-criterion guard — identity-based, both endpoints, and the anti-tautology
# right-count/wrong-members case.
# --------------------------------------------------------------------------- #
def test_assess_all_recovered_is_degenerate_high():
    a = mc.assess_recovery({"x", "y", "z"}, {"x", "y", "z"})
    assert a["degenerate"] is True
    assert a["recall"] == 1.0
    assert "1.0" in a["degenerate_reason"]


def test_assess_none_recovered_is_degenerate_low():
    a = mc.assess_recovery({"x", "y", "z"}, set())
    assert a["degenerate"] is True
    assert a["recall"] == 0.0
    assert "recall is 0" in a["degenerate_reason"]


def test_assess_intermediate_is_non_degenerate():
    a = mc.assess_recovery({"a", "b", "c", "d"}, {"a", "b"})
    assert a["degenerate"] is False
    assert a["recall"] == 0.5
    assert a["n_recovered_in_probe"] == 2


def test_assess_is_identity_not_count():
    # Right count (3), wrong members (none in the probe set) ⇒ 0 recovered ⇒ degenerate.
    # A count-only assessment would read recall = 3/3 = 1.0 and (wrongly) call it "recovered".
    a = mc.assess_recovery({"a", "b", "c"}, {"x", "y", "z"})
    assert a["n_recovered_in_probe"] == 0
    assert a["degenerate"] is True
    assert a["recall"] == 0.0


def test_guard_raises_on_both_degenerate_endpoints():
    with pytest.raises(mc.ProvisionalCriterionError):
        mc.guard_non_degenerate({"a", "b"}, {"a", "b"})
    with pytest.raises(mc.ProvisionalCriterionError):
        mc.guard_non_degenerate({"a", "b"}, set())


def test_guard_returns_assessment_when_non_degenerate():
    assessment = mc.guard_non_degenerate({"a", "b", "c", "d"}, {"a", "b", "c"})
    assert assessment["degenerate"] is False
    assert assessment["recall"] == 0.75


def test_assess_empty_members_raises():
    with pytest.raises(mc.ProvisionalCriterionError):
        mc.assess_recovery(set(), {"x"})
