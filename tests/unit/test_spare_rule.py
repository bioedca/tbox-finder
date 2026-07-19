"""Unit tier — the P2-07 three-valued mining spare-rule gate + readiness gate.

Guards the failure mode the module exists for: the pinned predicate
:func:`tbox_finder.masking.spare_rule_excludes_from_mining` takes booleans that
default to ``False``, so with no disjunct backend available it excludes **nothing**
and the Tier-2N protection reads green while protecting zero candidates. Also
guards the subtler variant — a backend set consisting only of relaxed-architecture
detection, which is ``False`` on every Tier-2N locus by ADR-0006 D9 row 5 and so
provides no protection at all while looking fully instrumented.

Bare-CI tier: pure stdlib, no numpy/pandas/torch.
"""

from __future__ import annotations

import pytest

from tbox_finder.masking import spare_rule_excludes_from_mining
from tbox_finder.mining.hard_negative import MINEABLE_POOLS, RETAINED_LEADER_POOL
from tbox_finder.mining.spare_rule import (
    MIN_PROTECTIVE_DISJUNCTS_AVAILABLE,
    MODEL_INDEPENDENT_DISJUNCTS,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    TIER2N_PROTECTIVE_DISJUNCTS,
    SpareRuleEvidence,
    SpareRuleEvidenceError,
    is_mining_excluded,
    mining_round_readiness,
    spare_reason,
)

ALL_AVAILABLE = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, True)


def _evidence(**overrides: str) -> SpareRuleEvidence:
    """Evidence with every disjunct evaluated-and-failed unless overridden."""
    base = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, STATUS_FAILED)
    base.update(overrides)
    return SpareRuleEvidence(**base)


# --------------------------------------------------------------------------- #
# Each disjunct excludes (the ADR-0005 D14 OR)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("disjunct", MODEL_INDEPENDENT_DISJUNCTS)
def test_each_model_independent_disjunct_alone_excludes_from_mining(disjunct: str) -> None:
    assert is_mining_excluded(_evidence(**{disjunct: STATUS_PASSED})) is True


def test_all_disjuncts_evaluated_and_failed_is_minable() -> None:
    assert is_mining_excluded(_evidence()) is False
    assert spare_reason(_evidence()) == "minable"


def test_stage2_posterior_is_inert_at_p2_and_excludes_only_at_p3() -> None:
    """ADR-0005 D14 phase-conditioning: the Stage-2 disjunct needs a threshold."""
    evidence = _evidence(stage2_posterior=0.99)
    assert is_mining_excluded(evidence) is False
    assert is_mining_excluded(evidence, stage2_threshold=0.9) is True


# --------------------------------------------------------------------------- #
# The three-valued arm: the reason this module exists
# --------------------------------------------------------------------------- #
def test_unavailable_backend_spares_rather_than_mines() -> None:
    """An unrun backend leaves the OR undetermined → fail closed."""
    evidence = _evidence(any_helix_rscape=STATUS_UNAVAILABLE)
    assert is_mining_excluded(evidence) is True
    assert spare_reason(evidence) == "unavailable_backend:any_helix_rscape"


def test_default_evidence_is_unavailable_not_failed() -> None:
    """An unconfigured record must not read as 'all three checked and failed'."""
    evidence = SpareRuleEvidence()
    assert evidence.unavailable() == MODEL_INDEPENDENT_DISJUNCTS
    assert is_mining_excluded(evidence) is True


def test_a_passed_disjunct_outranks_an_unavailable_one() -> None:
    """Kleene OR: a decisive True wins regardless of unknowns elsewhere."""
    evidence = _evidence(any_helix_rscape=STATUS_PASSED, downstream_aaRS_synteny=STATUS_UNAVAILABLE)
    assert is_mining_excluded(evidence) is True
    assert spare_reason(evidence) == "passed:any_helix_rscape"


def test_the_pinned_boolean_predicate_would_mine_an_unevaluated_candidate() -> None:
    """Anti-tautology: pin the defect this module corrects, in the original API.

    Called with its own defaults — which is exactly what "no backend available"
    produces — the pinned predicate returns ``False`` (⇒ mine it). This test fails
    if that ever stops being true, i.e. if the wrapper's reason for existing
    silently disappears and the two layers become redundant.
    """
    assert spare_rule_excludes_from_mining() is False
    assert is_mining_excluded(SpareRuleEvidence()) is True


# --------------------------------------------------------------------------- #
# Status validation — a typo must not read as "not passed"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["PASSED", "pass", "true", "", "unknown"])
def test_unknown_status_string_is_rejected(bad: str) -> None:
    with pytest.raises(SpareRuleEvidenceError):
        SpareRuleEvidence(any_helix_rscape=bad)


@pytest.mark.parametrize("bad", [True, "0.9", -0.1, 1.1])
def test_malformed_stage2_posterior_is_rejected(bad: object) -> None:
    with pytest.raises(SpareRuleEvidenceError):
        SpareRuleEvidence(stage2_posterior=bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The readiness gate — availability is not fungible across disjuncts
# --------------------------------------------------------------------------- #
def test_round_is_ready_when_a_protective_disjunct_is_available() -> None:
    for disjunct in TIER2N_PROTECTIVE_DISJUNCTS:
        availability = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, False)
        availability[disjunct] = True
        report = mining_round_readiness(availability)
        assert report["ready"] is True
        assert report["refusal_reason"] is None


def test_architecture_only_backend_set_is_refused_despite_being_available() -> None:
    """The sharp case: instrumented-looking, but zero Tier-2N protection.

    ADR-0006 D9 row 5 defines Tier-2N as ``(c)✓ ∧ (a)✓ ∧ (b)✗`` — relaxed
    architecture is False on every Tier-2N locus, so it can never spare one.
    """
    availability = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, False)
    availability["relaxed_architecture"] = True
    report = mining_round_readiness(availability)
    assert report["ready"] is False
    assert report["available_disjuncts"] == ["relaxed_architecture"]
    assert report["n_protective_available"] == 0
    assert "zero" in str(report["refusal_reason"])


def test_no_backend_at_all_is_refused() -> None:
    report = mining_round_readiness(dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, False))
    assert report["ready"] is False
    assert report["n_protective_available"] == 0


def test_relaxed_architecture_is_not_a_protective_disjunct() -> None:
    """Pin the membership itself, so a well-meaning 'completeness' edit trips."""
    assert "relaxed_architecture" not in TIER2N_PROTECTIVE_DISJUNCTS
    assert set(TIER2N_PROTECTIVE_DISJUNCTS) == {"any_helix_rscape", "downstream_aaRS_synteny"}
    assert MIN_PROTECTIVE_DISJUNCTS_AVAILABLE >= 1


def test_availability_must_name_every_disjunct() -> None:
    """A forgotten backend must raise, not default to available or unavailable."""
    with pytest.raises(SpareRuleEvidenceError):
        mining_round_readiness({"any_helix_rscape": True})


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_non_boolean_availability_is_rejected(bad: object) -> None:
    availability: dict = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, True)
    availability["any_helix_rscape"] = bad
    with pytest.raises(SpareRuleEvidenceError):
        mining_round_readiness(availability)


# --------------------------------------------------------------------------- #
# ADR-0005 D14: the leader pool is RETAINED
# --------------------------------------------------------------------------- #
def test_leader_decoy_pool_is_retained_as_mineable() -> None:
    """The hardest, most-useful hard negatives — explicitly kept by ADR-0005 D14."""
    assert RETAINED_LEADER_POOL in MINEABLE_POOLS


def test_the_spare_rule_does_not_exclude_a_leader_decoy_wholesale() -> None:
    """Retention is per-candidate evidence, not a pool-level exemption.

    A leader decoy whose disjuncts were all evaluated and failed is minable — the
    spare rule must not blanket-spare the pool.
    """
    assert is_mining_excluded(_evidence()) is False
