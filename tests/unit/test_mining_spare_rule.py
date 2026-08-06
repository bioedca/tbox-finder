"""Unit tier — the P3 re-mining round: the fourth spare-rule disjunct (P3-15).

Three things are guarded here, in descending order of how expensive they would be
to get wrong:

1. **The fourth disjunct's unevaluated arm.** Before P3-15 the Stage-2 posterior was
   two-valued: ``None`` contributed nothing to the OR, so a candidate whose three
   model-independent disjuncts had all failed was **mined with Stage-2 never
   consulted**. That is the conflation :mod:`tbox_finder.mining.spare_rule` exists to
   prevent, one disjunct later, and the cost is a real Tier-2N T-box trained against.

2. **"Ready" does not mean "can mine".** Sparing is a disjunction, mining its
   negation — a conjunction. One unavailable backend caps the round's yield at 0 for
   every candidate while the shipped readiness gate still reads ``ready``. That state
   is not hypothetical: ADR-0005 A10's Phase-2 §7 deferral records it as the measured
   P2 state, with the round reporting success on 0 mined.

3. **No value is pinned.** D14 pins the rule; the §13.1 phase gate pins the Stage-2
   operating point.

Bare-CI tier: pure stdlib, no numpy/pandas/torch.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tbox_finder.eval.tier2n_probe import ProbeSet
from tbox_finder.masking import LocusIndex, spare_rule_excludes_from_mining
from tbox_finder.mining import mine_round as mine_round_module
from tbox_finder.mining.hard_negative import (
    MINEABLE_POOLS,
    OUTCOME_MINED,
    OUTCOME_SPARED,
    RETAINED_LEADER_POOL,
    HardNegativeMiningError,
    MiningCandidate,
    classify_candidate,
    mine_round,
)
from tbox_finder.mining.remine import (
    STAGE2_SUPPLY_AVAILABLE,
    RemineError,
    apply_remine_spare_rule,
    build_parser,
    build_remine_availability,
    build_remine_report,
    exclude_probe_members,
    load_probe_set,
    load_stage2_posteriors,
    max_possible_yield,
    no_remine_parameter_has_a_default,
    plan_remine_round,
    probe_member_ids,
    read_remine_manifest,
    remine_candidate_evidence,
    remine_problems,
)
from tbox_finder.mining.remine import main as remine_main
from tbox_finder.mining.spare_rule import (
    ALL_DISJUNCTS,
    MODEL_INDEPENDENT_DISJUNCTS,
    STAGE2_DISJUNCT,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    TIER2N_PROTECTIVE_DISJUNCTS,
    SpareRuleEvidence,
    is_mining_excluded,
    spare_reason,
    stage2_status,
    unavailable_disjuncts,
)

THRESHOLD = 0.9
EMPTY_MASK = LocusIndex.from_records([])


def _all_failed(**overrides: object) -> SpareRuleEvidence:
    """Evidence with every model-independent disjunct **run and failed**.

    This is the only shape from which a candidate can be mined, so it is the shape
    every fourth-disjunct assertion must start from: anything weaker is spared for a
    reason that has nothing to do with Stage-2, and the test would pass without the
    code under test running at all.
    """
    kwargs: dict[str, object] = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, STATUS_FAILED)
    kwargs.update(overrides)
    return SpareRuleEvidence(**kwargs)  # type: ignore[arg-type]


def _candidate(cid: str = "c1", **evidence_kwargs: object) -> MiningCandidate:
    return MiningCandidate(
        candidate_id=cid,
        pool="genomic_window",
        accession="GCA_000000001.1:c0",
        locus_start=100,
        locus_end=200,
        score=0.99,
        evidence=_all_failed(**evidence_kwargs),
    )


def _all_available() -> dict[str, bool]:
    return dict.fromkeys(ALL_DISJUNCTS, True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. The fourth disjunct's unevaluated arm — the defect P3-15 closes
# ═════════════════════════════════════════════════════════════════════════════
def test_an_unscored_candidate_is_spared_not_mined_at_a_p3_round() -> None:
    """The regression. Every model-independent disjunct failed; Stage-2 never ran.

    Mining this candidate would consult three of four disjuncts and call the answer
    complete. ADR-0005 A10 Pin 3 pins the direction verbatim: unscored ⇒ unavailable
    ⇒ spared, never mined.
    """
    evidence = _all_failed(stage2_posterior=None)
    assert stage2_status(evidence, stage2_threshold=THRESHOLD) == STATUS_UNAVAILABLE
    assert is_mining_excluded(evidence, stage2_threshold=THRESHOLD) is True
    assert spare_reason(evidence, stage2_threshold=THRESHOLD) == (
        f"unavailable_backend:{STAGE2_DISJUNCT}"
    )


def test_a_scored_low_candidate_is_still_minable() -> None:
    """The positive control for the test above.

    Without it, "spared" would be satisfied by a guard that spares *everything* — the
    same failure a bare ``pytest.raises`` has. A candidate Stage-2 actually scored and
    scored low must still reach the pool, or the fourth disjunct would have silently
    switched the round off rather than completed it.
    """
    evidence = _all_failed(stage2_posterior=0.01)
    assert stage2_status(evidence, stage2_threshold=THRESHOLD) == STATUS_FAILED
    assert is_mining_excluded(evidence, stage2_threshold=THRESHOLD) is False
    assert spare_reason(evidence, stage2_threshold=THRESHOLD) == "minable"


def test_a_high_posterior_spares_and_names_itself_in_the_reason() -> None:
    evidence = _all_failed(stage2_posterior=0.99)
    assert stage2_status(evidence, stage2_threshold=THRESHOLD) == STATUS_PASSED
    assert is_mining_excluded(evidence, stage2_threshold=THRESHOLD) is True
    assert spare_reason(evidence, stage2_threshold=THRESHOLD) == f"passed:{STAGE2_DISJUNCT}"


def test_the_threshold_boundary_agrees_with_the_pinned_predicate() -> None:
    """``>=`` is the pinned comparison (ADR-0005 D14), and the two readers must agree.

    ``is_mining_excluded`` **delegates** the OR to
    :func:`~tbox_finder.masking.spare_rule_excludes_from_mining`, so its verdict alone
    cannot see this module's own comparison: a boundary assertion phrased on the
    verdict stays green with ``stage2_status`` mis-written — measured, by sabotage
    (``>=`` → ``>`` left an earlier version of this test passing). The status is
    therefore read directly and cross-checked against the pinned predicate at three
    points around the threshold, which is exactly where the two could diverge and
    make ``spare_reason`` publish a cause the decision never used.
    """
    for posterior, expected in (
        (THRESHOLD - 1e-9, STATUS_FAILED),
        (THRESHOLD, STATUS_PASSED),
        (THRESHOLD + 1e-9, STATUS_PASSED),
    ):
        evidence = _all_failed(stage2_posterior=posterior)
        status = stage2_status(evidence, stage2_threshold=THRESHOLD)
        assert status == expected, posterior
        pinned = spare_rule_excludes_from_mining(
            stage2_posterior=posterior, stage2_threshold=THRESHOLD
        )
        assert (status == STATUS_PASSED) is pinned, posterior
        assert is_mining_excluded(evidence, stage2_threshold=THRESHOLD) is pinned, posterior


def test_the_disjunct_stays_inert_when_the_round_declares_no_stage2() -> None:
    """D14 phase-conditioning: with no threshold the P2 behaviour is bit-for-bit intact.

    Both arms are asserted — a high posterior does not spare, and an absent one does
    not add a fourth ``unavailable``. The second half is what keeps the P2 loop's
    reason strings and mined set unchanged by this step.
    """
    assert is_mining_excluded(_all_failed(stage2_posterior=0.99)) is False
    assert stage2_status(_all_failed(stage2_posterior=None)) is None
    assert unavailable_disjuncts(_all_failed(stage2_posterior=None)) == ()
    assert spare_reason(_all_failed()) == "minable"


def test_unavailable_disjuncts_is_the_single_derivation_behind_both_readers() -> None:
    """The decision and its published attribution must not be able to disagree.

    ``is_mining_excluded`` and ``spare_reason`` both read
    :func:`unavailable_disjuncts`; asserting the three agree on the same evidence pins
    that, so a future fork of one of them shows up here rather than as a report naming
    a cause that did not fire.
    """
    evidence = _all_failed(any_helix_rscape=STATUS_UNAVAILABLE, stage2_posterior=None)
    missing = unavailable_disjuncts(evidence, stage2_threshold=THRESHOLD)
    assert missing == ("any_helix_rscape", STAGE2_DISJUNCT)
    assert is_mining_excluded(evidence, stage2_threshold=THRESHOLD) is True
    assert spare_reason(evidence, stage2_threshold=THRESHOLD) == (
        "unavailable_backend:any_helix_rscape,high_stage2_posterior"
    )


def test_a_passed_model_independent_disjunct_outranks_an_unscored_stage2() -> None:
    """Kleene OR: a decisive pass wins, and the reason names the pass, not the gap."""
    evidence = _all_failed(any_helix_rscape=STATUS_PASSED, stage2_posterior=None)
    assert is_mining_excluded(evidence, stage2_threshold=THRESHOLD) is True
    assert spare_reason(evidence, stage2_threshold=THRESHOLD) == "passed:any_helix_rscape"


def test_the_stage2_disjunct_is_not_counted_as_model_independent() -> None:
    """Anti-circularity accounting: D14 calls the first three model-independent.

    Folding the project's own model into that tuple would let a report claim
    model-independent evidence for a candidate spared by the model itself.
    """
    assert STAGE2_DISJUNCT not in MODEL_INDEPENDENT_DISJUNCTS
    assert STAGE2_DISJUNCT not in TIER2N_PROTECTIVE_DISJUNCTS
    assert ALL_DISJUNCTS == MODEL_INDEPENDENT_DISJUNCTS + (STAGE2_DISJUNCT,)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Round-level: "ready" is not "can mine"
# ═════════════════════════════════════════════════════════════════════════════
def test_a_round_with_every_backend_can_mine() -> None:
    """The positive control for the whole yield gate.

    Every refusal assertion below is worthless without it: a preflight that refuses
    unconditionally would satisfy them all.
    """
    gate = max_possible_yield(_all_available(), stage2_threshold=THRESHOLD)
    assert gate["yield_producible"] is True
    assert gate["blocking_disjuncts"] == []
    assert gate["max_mined"] is None
    assert gate["reason"] is None


@pytest.mark.parametrize("missing", ALL_DISJUNCTS)
def test_one_missing_backend_caps_the_round_yield_at_zero(missing: str) -> None:
    """Each disjunct is asserted **alone**, because mining is a conjunction.

    A single all-unavailable fixture would pass against a gate that only checks one
    named backend; sabotaging any one member has to be visible on its own.
    """
    availability = _all_available()
    availability[missing] = False
    gate = max_possible_yield(availability, stage2_threshold=THRESHOLD)
    assert gate["yield_producible"] is False
    assert gate["blocking_disjuncts"] == [missing]
    assert gate["max_mined"] == 0
    assert missing in str(gate["reason"])


def test_a_p3_round_without_a_threshold_has_not_declared_stage2_live() -> None:
    gate = max_possible_yield(_all_available(), stage2_threshold=None)
    assert gate["yield_producible"] is False
    assert gate["stage2_declared_live"] is False
    assert gate["blocking_disjuncts"] == [STAGE2_DISJUNCT]


def test_the_measured_p2_state_is_ready_and_yields_nothing() -> None:
    """The exact state ADR-0005 A10's Phase-2 §7 deferral measured, as a test.

    Covariation available, relaxed-architecture and synteny absent: readiness passes
    (one protective backend), yield is provably 0, and the two gates disagree. This is
    what ``may_run`` exists to reconcile — and it is the state the repo is in today.
    """
    plan = plan_remine_round(
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        stage2_threshold=THRESHOLD,
    )
    assert plan["ready"] is True
    assert plan["yield_producible"] is False
    assert plan["may_run"] is False
    assert sorted(plan["yield"]["blocking_disjuncts"]) == [
        "downstream_aaRS_synteny",
        "relaxed_architecture",
    ]


def test_stage2_alone_cannot_make_a_round_ready() -> None:
    """The pathology the readiness gate exists to prevent, one disjunct later.

    If the Stage-2 backend counted as protective, a P3 round would go green on the
    model's own posterior with no model-independent evidence at all — and still mine
    nothing.
    """
    plan = plan_remine_round(
        rscape_installed=False,
        msa_supply_available=False,
        stage2_supply_available=True,
        stage2_threshold=THRESHOLD,
    )
    assert plan["ready"] is False
    assert plan["may_run"] is False


def test_a_fully_backed_round_may_run() -> None:
    plan = plan_remine_round(
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        relaxed_arch_available=True,
        synteny_available=True,
        stage2_threshold=THRESHOLD,
    )
    assert plan["ready"] is True and plan["yield_producible"] is True
    assert plan["may_run"] is True


def test_the_stage2_supply_flag_agrees_with_its_derivation_and_resolves_at_call_time() -> None:
    """The RUN blocker is data, and flipping it must reach a caller that omits it.

    Strengthened at P3-15′-b rather than inverted: the old ``is False`` could only catch
    drift in one direction, so a stale ``True`` (or, now, a stale ``False`` on a checkout
    that *can* evidence the supply) would pass. Pinning the constant against its own
    re-derivation catches both, and is the P3-15′-a discipline applied to the second flag.
    """
    from tbox_finder.mining.stage2_producer import derive_stage2_supply_available

    derived = derive_stage2_supply_available()
    assert STAGE2_SUPPLY_AVAILABLE is derived["available"], derived["reasons"]
    # The real backend state: `relaxed_arch_available`/`synteny_available` are left at
    # their False defaults because P3-15′-c and P3-15′-d have not landed. Forcing them
    # True (as this test did while the Stage-2 flag was the only blocker) would assert a
    # world in which every disjunct has a backend, which is not the one that ships.
    plan = plan_remine_round(
        rscape_installed=True,
        msa_supply_available=True,
        stage2_threshold=THRESHOLD,
    )
    # Omitted by the caller ⇒ resolved from the module constant at CALL time, which is the
    # property that makes the flip reach every caller without a signature change.
    assert plan["stage2_supply_available"] is STAGE2_SUPPLY_AVAILABLE
    assert plan["ready"] is True
    assert STAGE2_DISJUNCT not in plan["yield"]["blocking_disjuncts"]
    # Still refused — by the two backends that genuinely do not exist. P3-15′-b supplies a
    # posterior; it does not supply relaxed-architecture or synteny, and mining is a
    # conjunction. A change that makes this round `may_run` is a regression, not progress.
    assert plan["may_run"] is False
    assert plan["yield"]["blocking_disjuncts"] == [
        "relaxed_architecture",
        "downstream_aaRS_synteny",
    ]
    assert plan["yield"]["max_mined"] == 0


def test_availability_delegates_the_msa_producibility_rule() -> None:
    """ADR-0006 A2: R-scape installed is necessary, not sufficient."""
    installed_only = build_remine_availability(
        rscape_installed=True, msa_supply_available=False, stage2_supply_available=True
    )
    assert installed_only["any_helix_rscape"] is False
    both = build_remine_availability(
        rscape_installed=True, msa_supply_available=True, stage2_supply_available=True
    )
    assert both["any_helix_rscape"] is True
    assert set(both) == set(ALL_DISJUNCTS)


# ═════════════════════════════════════════════════════════════════════════════
# 3. The posterior join — fail-closed on every miss
# ═════════════════════════════════════════════════════════════════════════════
def test_an_id_absent_from_the_posterior_table_is_spared() -> None:
    """A dropped producer shard costs sensitivity, never a mined true T-box."""
    evidence = remine_candidate_evidence(
        "missing",
        covariation_status={"missing": STATUS_FAILED},
        stage2_posteriors={"other": 0.99},
    )
    assert evidence.stage2_posterior is None
    assert is_mining_excluded(evidence, stage2_threshold=THRESHOLD) is True


def test_a_present_id_carries_its_posterior_through() -> None:
    evidence = remine_candidate_evidence(
        "c1", covariation_status={"c1": STATUS_FAILED}, stage2_posteriors={"c1": 0.42}
    )
    assert evidence.stage2_posterior == pytest.approx(0.42)
    assert evidence.any_helix_rscape == STATUS_FAILED


def test_the_join_delegates_the_p2_evidence_builder() -> None:
    """The three model-independent statuses come from the P2 builder, not a fork."""
    evidence = remine_candidate_evidence(
        "c1", covariation_status=None, stage2_posteriors={"c1": 0.99}
    )
    assert evidence.any_helix_rscape == STATUS_UNAVAILABLE
    assert evidence.relaxed_architecture == STATUS_UNAVAILABLE
    assert evidence.downstream_aaRS_synteny == STATUS_UNAVAILABLE


def test_a_posterior_table_carrying_logits_is_refused(tmp_path: Path) -> None:
    """A logit in the posterior column would fail every ``>=`` and mine the spared."""
    path = tmp_path / "post.json"
    path.write_text(json.dumps({"posteriors": {"c1": 4.2}}), encoding="utf-8")
    with pytest.raises(RemineError, match=r"outside \[0, 1\]"):
        load_stage2_posteriors(path)


def test_a_wellformed_posterior_table_loads(tmp_path: Path) -> None:
    """Positive control for the refusal above — the same reader on clean input."""
    path = tmp_path / "post.json"
    path.write_text(json.dumps({"posteriors": {"c1": 0.42, "c2": 1.0}}), encoding="utf-8")
    assert load_stage2_posteriors(path) == {"c1": 0.42, "c2": 1.0}


@pytest.mark.parametrize("bad", [True, "0.9", None])
def test_a_nonnumeric_posterior_is_refused(tmp_path: Path, bad: object) -> None:
    path = tmp_path / "post.json"
    path.write_text(json.dumps({"posteriors": {"c1": bad}}), encoding="utf-8")
    with pytest.raises(RemineError, match="not a real number"):
        load_stage2_posteriors(path)


def test_the_manifest_reader_stamps_both_evidence_sources(tmp_path: Path) -> None:
    path = tmp_path / "fps.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "step": "P2-10e",
                "n_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "a",
                        "accession": "GCA_1:c0",
                        "locus_start": 1,
                        "locus_end": 9,
                        "score": 0.9,
                        "pool": "genomic_window",
                    },
                    {
                        "candidate_id": "b",
                        "accession": "GCA_1:c0",
                        "locus_start": 20,
                        "locus_end": 30,
                        "score": 0.8,
                        "pool": "genomic_window",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = read_remine_manifest(
        path,
        covariation_status={"a": STATUS_FAILED, "b": STATUS_FAILED},
        stage2_posteriors={"a": 0.99},
    )
    by_id = {c.candidate_id: c for c in candidates}
    assert by_id["a"].evidence.stage2_posterior == pytest.approx(0.99)
    assert by_id["b"].evidence.stage2_posterior is None
    assert by_id["a"].evidence.any_helix_rscape == STATUS_FAILED


def test_the_round_leg_runs_end_to_end_and_measures_the_structural_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leg ``stage1_remine.sbatch`` calls, **executed** — not merely shipped.

    Only the union-prior mask is substituted (it needs the DVC-tracked corpus parquet,
    absent in a worktree); every operator that decides an outcome is the shipped one,
    and the backends are declared exactly as the repo can declare them today —
    covariation and Stage-2, no relaxed-architecture, no synteny.

    The fixture is **asymmetric on purpose**: three candidates that must come back with
    three *different* spare reasons, so a count-only assertion cannot pass on the wrong
    three. And the outcome is the measured structural zero — ``n_mined == 0``, every
    candidate spared, including the one Stage-2 scored at 0.01 — with the attribution
    naming the two disjuncts that have no backend. A round with a real producer for
    every disjunct would mine ``mined-low``; this one cannot, and says why.
    """
    manifest = tmp_path / "fps.json"
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": cid,
                        "accession": "GCA_1:c0",
                        "locus_start": s,
                        "locus_end": s + 50,
                        "score": 0.9,
                        "pool": "genomic_window",
                    }
                    for cid, s in (
                        ("spared-high", 100),
                        ("mined-low", 300),
                        ("unscored", 500),
                        ("probe-member", 700),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {
                "status": dict.fromkeys(
                    ("spared-high", "mined-low", "unscored", "probe-member"), "failed"
                )
            }
        ),
        encoding="utf-8",
    )
    posteriors = tmp_path / "post.json"
    posteriors.write_text(
        json.dumps({"posteriors": {"spared-high": 0.99, "mined-low": 0.01, "probe-member": 0.01}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mine_round_module, "load_union_mask", lambda **_kw: EMPTY_MASK)

    report = apply_remine_spare_rule(
        manifest,
        status,
        posteriors,
        stage2_threshold=THRESHOLD,
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        probe_set=ProbeSet(natural=(), synthetic=("probe-member", "not-in-substrate")),
    )
    # The probe member is gone from the outcome entirely — not spared, ABSENT. Asserting
    # its identity (rather than a count) is what distinguishes "excluded nothing" from
    # "never excluded": a disjoint probe set makes those two indistinguishable, measured
    # by sabotage (removing the exclusion call left an earlier fixture green).
    assert "probe-member" not in report["reasons"]
    assert report["excluded_probe_member_ids"] == ["probe-member"]
    assert report["n_excluded_probe_members"] == 1
    assert report["n_probe_members_considered"] == 2
    assert report["n_mined"] == 0
    assert report["mined_ids"] == []
    assert sorted(report["spared_ids"]) == ["mined-low", "spared-high", "unscored"]
    # Three candidates, three distinct causes — the Stage-2 arm fires inside the leg.
    assert report["reasons"]["spared-high"] == f"passed:{STAGE2_DISJUNCT}"
    assert report["reasons"]["mined-low"] == (
        "unavailable_backend:relaxed_architecture,downstream_aaRS_synteny"
    )
    assert report["reasons"]["unscored"] == (
        f"unavailable_backend:relaxed_architecture,downstream_aaRS_synteny,{STAGE2_DISJUNCT}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# The round-level cross-checks in hard_negative.mine_round
# ═════════════════════════════════════════════════════════════════════════════
def test_a_posterior_without_a_declared_threshold_raises() -> None:
    """Evidence for a backend the round never declared must not be silently ignored."""
    availability = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, True)
    with pytest.raises(HardNegativeMiningError, match="declared no stage2_threshold"):
        mine_round([_candidate(stage2_posterior=0.99)], EMPTY_MASK, availability)


def test_a_posterior_with_no_declared_stage2_backend_raises() -> None:
    """The reverse contradiction — and the FAIL-OPEN one, found by review r1.

    A round that declares a threshold but no Stage-2 backend resolves a low posterior
    to ``failed``, so the candidate counts as "every disjunct ran and failed" and is
    mined on evidence for a backend the round said was absent. Reproduced before the
    guard existed: it was mined, and the reason read *"no disjunct passed and every
    disjunct was evaluated"* — the attribution asserting the very thing that was false.
    """
    availability = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, True)
    availability[STAGE2_DISJUNCT] = False
    with pytest.raises(HardNegativeMiningError, match="backend is unavailable this round"):
        mine_round(
            [_candidate(stage2_posterior=0.01)],
            EMPTY_MASK,
            availability,
            stage2_threshold=THRESHOLD,
        )


def test_the_same_round_is_accepted_once_both_are_declared() -> None:
    """Positive control for both guards: they refuse contradictions, not every round."""
    availability = dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, True)
    availability[STAGE2_DISJUNCT] = True
    report = mine_round(
        [_candidate(stage2_posterior=0.99)], EMPTY_MASK, availability, stage2_threshold=THRESHOLD
    )
    assert report["n_spared"] == 1 and report["n_mined"] == 0
    mined = mine_round(
        [_candidate(stage2_posterior=0.01)], EMPTY_MASK, availability, stage2_threshold=THRESHOLD
    )
    assert mined["n_mined"] == 1


def test_classify_candidate_mines_only_when_all_four_disjuncts_failed() -> None:
    outcome, _reason = classify_candidate(
        _candidate(stage2_posterior=0.01), EMPTY_MASK, stage2_threshold=THRESHOLD
    )
    assert outcome == OUTCOME_MINED
    outcome, _reason = classify_candidate(
        _candidate(stage2_posterior=None), EMPTY_MASK, stage2_threshold=THRESHOLD
    )
    assert outcome == OUTCOME_SPARED


# ═════════════════════════════════════════════════════════════════════════════
# ADR-0005 D14: the leader pool is retained; the Tier-2N probe is not mined
# ═════════════════════════════════════════════════════════════════════════════
def test_the_leader_decoy_pool_stays_mineable() -> None:
    """D14 retains the hardest, most-useful hard negatives.

    Dropping it would look like a harmless tightening and would cost exactly the
    5′UTR/tRNA-adjacent context a tRNA-sensing finder most needs to learn against.
    """
    assert RETAINED_LEADER_POOL in MINEABLE_POOLS


def test_probe_members_are_excluded_from_the_mining_substrate() -> None:
    """The instrument and the treatment must not share a record.

    A probe member mined as a hard negative would be trained against and then have
    its own recall measured — a drop would be indistinguishable from the round having
    trained the probe away, which is the halt signal.
    """
    probe = ProbeSet(natural=("nat1",), synthetic=("syn1", "syn2"))
    candidates = [_candidate("nat1"), _candidate("c-real"), _candidate("syn2")]
    keep, dropped = exclude_probe_members(candidates, probe)
    assert [c.candidate_id for c in keep] == ["c-real"]
    assert dropped == ["nat1", "syn2"]
    assert probe_member_ids(probe) == frozenset({"nat1", "syn1", "syn2"})


def test_exclusion_is_a_no_op_on_a_disjoint_substrate() -> None:
    """Positive control: the filter removes probe members, not candidates."""
    probe = ProbeSet(natural=(), synthetic=("syn1",))
    candidates = [_candidate("c1"), _candidate("c2")]
    keep, dropped = exclude_probe_members(candidates, probe)
    assert [c.candidate_id for c in keep] == ["c1", "c2"]
    assert dropped == []


# ═════════════════════════════════════════════════════════════════════════════
# The report + its self-certification clause set
# ═════════════════════════════════════════════════════════════════════════════
def _refused_report() -> dict[str, object]:
    plan = plan_remine_round(
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        stage2_threshold=THRESHOLD,
    )
    return build_remine_report(
        plan=plan, round_report=None, probe_trace=None, stage2_threshold=THRESHOLD
    )


def test_a_refused_round_reports_no_mined_count_and_self_checks_clean() -> None:
    """``None``, not ``0`` — a refused round and a round that found nothing differ."""
    report = _refused_report()
    assert report["may_run"] is False
    assert report["round"] is None and report["tier2n_probe"] is None
    assert report["stage2_threshold_pinned"] is False
    assert remine_problems(report) == []


def test_a_refused_round_carrying_a_mining_outcome_is_caught() -> None:
    report = _refused_report()
    report["round"] = {"n_mined": 7}
    assert "a refused round carries a mining outcome" in remine_problems(report)


def test_a_forged_may_run_is_caught_by_re_derivation() -> None:
    """The clause recomputes ``readiness × yield``; it never reads ``may_run`` back."""
    report = _refused_report()
    report["may_run"] = True
    problems = remine_problems(report)
    assert any("disagrees with readiness" in p for p in problems)


def test_a_pinned_threshold_claim_is_caught() -> None:
    report = _refused_report()
    report["stage2_threshold_pinned"] = True
    assert any("pins no value" in p for p in remine_problems(report))


# ═════════════════════════════════════════════════════════════════════════════
# The CLI legs the sbatch actually invokes
# ═════════════════════════════════════════════════════════════════════════════
def test_the_plan_leg_exits_nonzero_and_records_why(tmp_path: Path) -> None:
    """``slurm/p3/stage1_remine.sbatch`` leg (0) branches on this exit code."""
    out = tmp_path / "plan.json"
    rc = remine_main(["plan", "--stage2-threshold", "0.9", "--out", str(out)])
    assert rc == 1
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["may_run"] is False
    assert written["problems"] == []
    assert written["plan"]["yield"]["max_mined"] == 0


def test_the_apply_leg_writes_no_mining_outcome_on_a_refused_round(tmp_path: Path) -> None:
    """A round that provably cannot mine must not publish zeroes that read as a result.

    The paths are deliberately absent: if the leg reached them it would raise, so this
    also pins that the preflight runs **before** any I/O.
    """
    out = tmp_path / "round.json"
    rc = remine_main(
        [
            "apply-spare-rule",
            "--stage2-threshold",
            "0.9",
            "--manifest",
            str(tmp_path / "absent-manifest.json"),
            "--status-table",
            str(tmp_path / "absent-status.json"),
            "--posteriors",
            str(tmp_path / "absent-post.json"),
            "--probe-set",
            str(tmp_path / "absent-probe.json"),
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    assert json.loads(out.read_text(encoding="utf-8"))["round"] is None


def test_the_sbatch_flag_spellings_parse() -> None:
    """The tokens ``stage1_remine.sbatch`` composes must be the ones argparse accepts.

    A renamed flag would surface only as a dead job after the queue wait, which is the
    expensive way to find out; parsing the shipped spellings here is the cheap way.
    """
    parser = build_parser()
    args = parser.parse_args(
        [
            "plan",
            "--stage2-threshold",
            "0.9",
            "--rscape-installed",
            "--msa-supply-available",
            "--stage2-supply-available",
            "--relaxed-arch-available",
            "--synteny-available",
            "--out",
            "x.json",
        ]
    )
    assert args.cmd == "plan" and args.stage2_threshold == pytest.approx(0.9)
    with pytest.raises(SystemExit):
        parser.parse_args(["plan", "--out", "x.json"])  # threshold is required, no default


def test_the_union_mask_has_one_builder() -> None:
    """Promote, don't duplicate: both rounds mask through ``load_union_mask``.

    A second copy of the mask construction is free to drift, and a drift puts a known
    T-box locus into the negative pool — the failure PRD §9.1's masking clause exists
    to prevent. Read off the source so a re-forked copy fails here.
    """
    source = inspect.getsource(mine_round_module.apply_spare_rule)
    assert "load_union_mask" in source
    assert "LocusIndex.from_records" not in source
    assert "load_union_mask" in inspect.getsource(apply_remine_spare_rule)


# ═════════════════════════════════════════════════════════════════════════════
# The Tier-2N probe set must reach the round the sbatch actually invokes (review r1)
# ═════════════════════════════════════════════════════════════════════════════
def test_the_apply_leg_requires_a_probe_set() -> None:
    """Review r1: the first draft's optional ``probe_set`` was never wired to the CLI.

    So the exclusion never ran, and ``n_excluded_probe_members: 0`` read as *"no probe
    member was in the substrate"* rather than *"no exclusion happened"*. Both ends are
    now unrepresentable — the CLI flag is required and the function parameter has no
    default.
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "apply-spare-rule",
                "--stage2-threshold",
                "0.9",
                "--manifest",
                "m.json",
                "--status-table",
                "s.json",
                "--posteriors",
                "p.json",
                "--out",
                "o.json",
            ]
        )
    params = inspect.signature(apply_remine_spare_rule).parameters
    assert params["probe_set"].default is inspect.Parameter.empty


def test_an_empty_probe_set_is_refused_rather_than_silently_excluding_nothing(
    tmp_path: Path,
) -> None:
    """``reports/p2/tier2n_probe.json`` carries counts, not ids — a loader pointed at it
    would build an empty set and the exclusion would report 0 while excluding nothing."""
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({"probe_set_size": 45, "n_synthetic": 45}), encoding="utf-8")
    with pytest.raises(RemineError, match="probe set is empty"):
        load_probe_set(path)


def test_a_probe_set_with_ids_loads_both_arms(tmp_path: Path) -> None:
    """Positive control for the refusal above, on the same reader."""
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({"natural": ["n1"], "synthetic": ["s1", "s2"]}), encoding="utf-8")
    loaded = load_probe_set(path)
    assert probe_member_ids(loaded) == frozenset({"n1", "s1", "s2"})


def test_a_run_round_that_excluded_against_an_empty_probe_set_is_caught() -> None:
    """The 0-with-no-denominator state the review named, as a report clause."""
    plan = plan_remine_round(
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        relaxed_arch_available=True,
        synteny_available=True,
        stage2_threshold=THRESHOLD,
    )
    assert plan["may_run"] is True
    report = build_remine_report(
        plan=plan,
        round_report={
            "n_mined": 3,
            "n_probe_members_considered": 0,
            "n_excluded_probe_members": 0,
            "excluded_probe_member_ids": [],
        },
        probe_trace=None,
        stage2_threshold=THRESHOLD,
    )
    assert any("did not run against a non-empty probe set" in p for p in remine_problems(report))


def test_a_run_round_with_a_real_probe_denominator_self_checks_clean() -> None:
    """Positive control: the clause fires on a missing denominator, not on every round."""
    plan = plan_remine_round(
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        relaxed_arch_available=True,
        synteny_available=True,
        stage2_threshold=THRESHOLD,
    )
    report = build_remine_report(
        plan=plan,
        round_report={
            "n_mined": 3,
            "n_probe_members_considered": 45,
            "n_excluded_probe_members": 1,
            "excluded_probe_member_ids": ["syn1"],
        },
        probe_trace=None,
        excluded_probe_ids=["syn1"],
        stage2_threshold=THRESHOLD,
    )
    assert remine_problems(report) == []


def test_a_mismatched_exclusion_count_is_caught() -> None:
    plan = plan_remine_round(
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        relaxed_arch_available=True,
        synteny_available=True,
        stage2_threshold=THRESHOLD,
    )
    report = build_remine_report(
        plan=plan,
        round_report={
            "n_mined": 3,
            "n_probe_members_considered": 45,
            "n_excluded_probe_members": 2,
            "excluded_probe_member_ids": ["syn1"],
        },
        probe_trace=None,
        stage2_threshold=THRESHOLD,
    )
    assert any("disagrees with the excluded id list" in p for p in remine_problems(report))


def test_no_remine_parameter_carries_a_default() -> None:
    """P3-15 pins nothing — re-derived from the live signature, not a static list."""
    assert no_remine_parameter_has_a_default() is True

    def with_a_default(*, stage2_threshold: float = 0.9) -> None:  # pragma: no cover
        """A stand-in for the mistake: the operating point acquiring a default."""

    assert no_remine_parameter_has_a_default(with_a_default) is False


def test_a_none_default_does_not_slip_past_the_pin() -> None:
    """Review r2: ``None`` was treated as "no default", which was the hole itself.

    ``stage2_threshold: float | None = None`` gives the operating point a default while
    looking like an absent one — exactly what this pin exists to catch. The only
    parameter that legitimately needs a ``None`` sentinel is ``stage2_supply_available``,
    exempt by name, and the shipped signature stays green (the positive control).
    """

    def with_a_none_default(*, stage2_threshold: float | None = None) -> None:  # pragma: no cover
        """The operating point acquiring a default that reads as absent."""

    assert no_remine_parameter_has_a_default(with_a_none_default) is False
    assert no_remine_parameter_has_a_default() is True


def test_a_flat_posterior_mapping_is_accepted_and_a_bad_one_refused_by_name(
    tmp_path: Path,
) -> None:
    """Review r2: a flat mapping raised a bare ``KeyError``, not this module's refusal."""
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"c1": 0.5}), encoding="utf-8")
    assert load_stage2_posteriors(flat) == {"c1": 0.5}

    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"posteriors": {"c1": 0.5}}), encoding="utf-8")
    assert load_stage2_posteriors(wrapped) == {"c1": 0.5}

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(RemineError, match="expected a candidate_id"):
        load_stage2_posteriors(bad)


def test_a_top_level_exclusion_count_that_contradicts_the_round_is_caught() -> None:
    """Review r2: ``build_remine_report``'s ``excluded_probe_ids`` defaults to ``()``.

    A caller that omits it publishes ``n_excluded_probe_members: 0`` at the top level
    beside a non-empty round-level list — the same unreadable 0 the module refuses
    everywhere else, and nothing compared the two.
    """
    plan = plan_remine_round(
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        relaxed_arch_available=True,
        synteny_available=True,
        stage2_threshold=THRESHOLD,
    )
    round_report = {
        "n_mined": 3,
        "n_probe_members_considered": 45,
        "n_excluded_probe_members": 1,
        "excluded_probe_member_ids": ["syn1"],
    }
    forgotten = build_remine_report(
        plan=plan, round_report=round_report, probe_trace=None, stage2_threshold=THRESHOLD
    )
    assert any("disagree with the round" in p for p in remine_problems(forgotten))

    # Positive control: the same report with the ids forwarded self-checks clean.
    passed_through = build_remine_report(
        plan=plan,
        round_report=round_report,
        probe_trace=None,
        excluded_probe_ids=["syn1"],
        stage2_threshold=THRESHOLD,
    )
    assert remine_problems(passed_through) == []
