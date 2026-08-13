"""P3-15′-k — the round publication's re-derivations, and the committed artifact.

Two tiers, and the split matters. The tests below the divider read
``reports/p3/remine_round_report.json`` and can only ever catch a *changed artifact*;
they are blind to the code that wrote it ([[artifact-pinning-test-cannot-see-the-code]]).
Everything above the divider drives the functions with constructed inputs, so a
sabotaged operator is red whether or not the artifact is regenerated.

The fixtures are deliberately **asymmetric** — the two halves of the spared split carry
different counts, and the per-disjunct tallies are all distinct — because a fixture
where both halves are equal cannot tell a fold from its inverse
([[symmetric-count-fixture-blind-to-inversion]]), and one where every clause is TRUE
cannot tell ``all`` from ``any`` ([[all-true-fixture-cannot-test-a-conjunction]]).
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tbox_finder.mining.remine_publish import (
    GRID_SUPPLY_PREFIX,
    REQUIRED_SUPPLY_KEYS,
    PublicationError,
    build_publication,
    id_namespaces,
    outcome_fingerprint,
    parse_spare_reason,
    plan_leg_summary,
    probe_exclusion_diagnosis,
    publication_problems,
    retrain_disposition,
    spared_split,
    substrate_ids_from_manifest,
    threshold_sensitivity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_PATH = REPO_ROOT / "reports/p3/remine_round_report.json"

#: The bytes the P3-15′-k round published. Regenerating the round legitimately means
#: updating this in the same commit — deliberately, as a golden fixture does.
COMMITTED_PUBLICATION_SHA256 = "c8aa02cfc4ac56160904998d78e80c0394f616e7a27a255905432fe770371999"


# ═════════════════════════════════════════════════════════════════════════════
# Constructed fixtures — asymmetric on purpose
# ═════════════════════════════════════════════════════════════════════════════
def make_round(**overrides: Any) -> dict[str, Any]:
    """A 9-candidate round: 1 mined, 5 spared by a pass, 3 by an unavailable backend.

    Every count in it is distinct (1 / 5 / 3, and per-disjunct 3 / 2 / 1 / 1), so a
    swapped numerator, a folded half, or an ``all``→``any`` flip moves a number.
    """
    reasons = {
        "c-mined": "no disjunct passed and every disjunct was evaluated",
        "c-syn-1": "passed:downstream_aaRS_synteny",
        "c-syn-2": "passed:downstream_aaRS_synteny",
        "c-syn-arch": "passed:relaxed_architecture,downstream_aaRS_synteny",
        "c-arch": "passed:relaxed_architecture",
        "c-stage2": "passed:high_stage2_posterior",
        "c-unav-1": "unavailable_backend:any_helix_rscape",
        "c-unav-2": "unavailable_backend:any_helix_rscape,relaxed_architecture",
        "c-unav-3": "unavailable_backend:downstream_aaRS_synteny",
    }
    spared = [k for k in reasons if k != "c-mined"]
    round_block = {
        "n_candidates": 9,
        "n_mined": 1,
        "n_masked": 0,
        "n_spared": len(spared),
        "n_refused_no_coordinates": 0,
        "mined_ids": ["c-mined"],
        "spared_ids": spared,
        "reasons": reasons,
        "n_excluded_probe_members": 0,
        "excluded_probe_member_ids": [],
        "n_probe_members_considered": 3,
    }
    report = {
        "leg": "apply-spare-rule",
        "may_run": True,
        "stage2_threshold": 0.9,
        "stage2_threshold_pinned": False,
        "parent_checkpoint": None,
        "problems": [],
        "round": round_block,
    }
    report.update(overrides)
    return report


def make_plan(**overrides: Any) -> dict[str, Any]:
    derivation = {
        "available": True,
        "clauses": {"a": True, "b": True},
        "reasons": [],
        "repo_root": "/absolute/path/that/must/not/be/published",
    }
    plan = {
        "leg": "plan",
        "problems": [],
        "plan": {
            "availability": {
                "relaxed_architecture": True,
                "any_helix_rscape": True,
                "downstream_aaRS_synteny": True,
                "high_stage2_posterior": True,
            },
            "readiness": {"ready": True},
            "yield": {"yield_producible": True, "blocking_disjuncts": []},
            "msa_supply_derivation": copy.deepcopy(derivation),
            "stage2_supply_derivation": copy.deepcopy(derivation),
            "synteny_supply_derivation": copy.deepcopy(derivation),
            "relaxed_arch_supply_derivation": copy.deepcopy(derivation),
        },
    }
    plan.update(overrides)
    return plan


def make_grid_twin(threshold: float) -> dict[str, Any]:
    """A second grid point over the SAME candidates whose composition differs.

    Deliberately asymmetric: with two identical points the composition flag is invariant
    whatever the code does, so a test that flips it would be testing nothing
    ([[symmetric-count-fixture-blind-to-inversion]]). Same id set and same mined ids, so
    the same-round binding still holds.
    """
    twin = make_round(stage2_threshold=threshold)
    twin["round"]["reasons"]["c-stage2"] = "unavailable_backend:high_stage2_posterior"
    return twin


def make_supplies(grid: dict[str, Any]) -> dict[str, str]:
    """Every key the binding requires, plus one per grid point."""
    return {k: "f" * 64 for k in REQUIRED_SUPPLY_KEYS} | {
        f"{GRID_SUPPLY_PREFIX}{k}": "a" * 64 for k in grid
    }


def make_publication(**overrides: Any) -> dict[str, Any]:
    round_report = make_round()
    # Each grid point carries the threshold ITS run was given, so the labels below are
    # bound to their own bytes rather than asserted by the operator.
    grid = {"0.5": make_grid_twin(0.5), "0.9": round_report}
    pub = build_publication(
        round_report=round_report,
        plan_report=make_plan(),
        substrate_ids={f"sub:{i}" for i in range(9)},
        probe_ids={f"tier2n:{i}" for i in range(3)},
        grid=grid,
        supplies=make_supplies(grid),
        reproduction={
            "reproduced": True,
            "published": outcome_fingerprint(round_report),
            "replayed": outcome_fingerprint(round_report),
        },
        provenance={"rule": "test"},
    )
    pub.update(overrides)
    return pub


# ═════════════════════════════════════════════════════════════════════════════
# (1) The spared split
# ═════════════════════════════════════════════════════════════════════════════
def test_spared_split_names_which_half_each_candidate_landed_in() -> None:
    """The whole point of the split is that 938 spared has two opposite readings.

    Asserted by **identity** of the per-disjunct tally, not only by the two totals: at
    5 / 3 the totals already separate, but a tally that credited the wrong disjunct
    would keep both totals and still misreport which criterion protected the corpus.
    """
    split = spared_split(make_round())
    assert split["n_spared"] == 8
    assert split["n_spared_by_a_passing_disjunct"] == 5
    assert split["n_spared_by_unavailable_backend_only"] == 3
    assert split["passes_per_disjunct"] == {
        "relaxed_architecture": 2,
        "any_helix_rscape": 0,
        "downstream_aaRS_synteny": 3,
        "high_stage2_posterior": 1,
    }
    assert split["unavailable_per_disjunct"] == {
        "relaxed_architecture": 1,
        "any_helix_rscape": 2,
        "downstream_aaRS_synteny": 1,
        "high_stage2_posterior": 0,
    }


def test_the_stage2_alone_count_is_the_anti_circularity_coordinate() -> None:
    """``high_stage2_posterior`` counts as *alone* only when nothing else passed.

    The fixture carries a candidate whose reason names the Stage-2 disjunct **beside** a
    model-independent one. ``spare_reason`` never emits that shape, but a membership
    test (``STAGE2 in disjuncts``) instead of an identity test would count it, inflating
    the one number ADR-0005 D14's model-independence accounting has to be able to
    subtract. So the shape is fed in deliberately and must not be counted.
    """
    reasons = dict(make_round()["round"]["reasons"])
    reasons["c-mixed"] = "passed:relaxed_architecture,high_stage2_posterior"
    report = make_round()
    report["round"]["reasons"] = reasons
    report["round"]["spared_ids"] = [k for k in reasons if k != "c-mined"]
    report["round"]["n_spared"] = len(report["round"]["spared_ids"])

    split = spared_split(report)
    assert split["passes_per_disjunct"]["high_stage2_posterior"] == 2
    assert split["n_spared_by_stage2_posterior_alone"] == 1


def test_a_spared_candidate_with_no_reason_is_refused_not_dropped() -> None:
    report = make_round()
    report["round"]["reasons"].pop("c-arch")
    with pytest.raises(PublicationError, match="no entry in the reasons map"):
        spared_split(report)


def test_a_spared_count_that_disagrees_with_the_id_list_is_refused() -> None:
    report = make_round()
    report["round"]["n_spared"] = 99
    with pytest.raises(PublicationError, match="disagrees with"):
        spared_split(report)


def test_an_empty_reasons_map_makes_the_split_vacuous_and_is_refused() -> None:
    """[[clauses-must-guard-emptiness]] — an empty map splits 0 / 0 and reads as a result."""
    report = make_round()
    report["round"]["reasons"] = {}
    with pytest.raises(PublicationError, match="no per-candidate reasons"):
        spared_split(report)


def test_a_duplicate_spared_id_inflates_a_half_and_is_refused() -> None:
    """Found by review (CLI r1). The count check above cannot see it: a repeated id keeps
    ``n_spared == len(spared_ids)`` true while being counted once per occurrence, so one
    half of the published split and its per-disjunct tally both inflate
    ([[duplicate-key-merges-instead-of-colliding]]).
    """
    report = make_round()
    report["round"]["spared_ids"] = [*report["round"]["spared_ids"], "c-arch"]
    report["round"]["n_spared"] = len(report["round"]["spared_ids"])
    with pytest.raises(PublicationError, match="duplicate candidate id"):
        spared_split(report)


def test_a_duplicate_mined_id_overstates_the_round_and_is_refused() -> None:
    """The sibling of the finding above, on the list that IS a 3-candidate round's headline
    ([[fixed-one-of-two-identical-things]])."""
    pub = make_publication()
    pub["outcome"]["mined_ids"] = ["c-mined", "c-mined"]
    pub["outcome"]["n_mined"] = 2
    pub["outcome"]["n_candidates"] = 10
    problems = publication_problems(pub)
    assert any("mined_ids carries a duplicate" in p for p in problems), problems


def test_a_round_that_spared_nothing_splits_to_zeroes_rather_than_raising() -> None:
    """Found by review (CLI r2). A grid point may legitimately mine every candidate, and
    ``threshold_sensitivity`` calls this for every point — refusing the empty list made
    such a point unpublishable. The anti-vacuity intent stays on ``reasons``: a round that
    decided nothing at all still cannot produce a split.
    """
    report = make_round()
    report["round"]["spared_ids"] = []
    report["round"]["n_spared"] = 0
    split = spared_split(report)
    assert split["n_spared"] == 0
    assert split["n_spared_by_a_passing_disjunct"] == 0
    assert split["n_spared_by_unavailable_backend_only"] == 0
    assert set(split["passes_per_disjunct"].values()) == {0}

    report["round"]["reasons"] = {}
    with pytest.raises(PublicationError, match="no per-candidate reasons"):
        spared_split(report)


@pytest.mark.parametrize(
    "reason",
    [
        "passed:not_a_disjunct",
        "unavailable_backend:relaxed_architecture,typo",
        "spared_because_i_said_so",
        "passed:",
    ],
)
def test_an_unreadable_spare_reason_is_refused_rather_than_bucketed(reason: str) -> None:
    """A reason this repo cannot parse must not silently land in either half."""
    with pytest.raises(PublicationError):
        parse_spare_reason(reason)


def test_parse_spare_reason_reads_the_shipped_grammar() -> None:
    """The positive control for the refusals above: a guard refusing EVERYTHING would
    satisfy every ``pytest.raises`` in this file ([[raises-test-needs-a-positive-control]])."""
    assert parse_spare_reason("passed:relaxed_architecture") == (
        "passed",
        ("relaxed_architecture",),
    )
    assert parse_spare_reason("unavailable_backend:any_helix_rscape,relaxed_architecture") == (
        "unavailable",
        ("any_helix_rscape", "relaxed_architecture"),
    )


# ═════════════════════════════════════════════════════════════════════════════
# (2) The probe-exclusion zero
# ═════════════════════════════════════════════════════════════════════════════
def test_a_zero_from_disjoint_namespaces_is_labelled_differently_from_a_real_zero() -> None:
    """The two zeros must never be published as the same finding."""
    structural = probe_exclusion_diagnosis(
        probe_ids={"tier2n:A:p1", "tier2n:A:p2"},
        substrate_ids={"GCA_1:c1:0:1-2", "GCA_2:c1:0:1-2"},
        n_excluded=0,
    )
    assert structural["reading"] == "structural_namespace_disjoint"
    assert structural["shared_namespaces"] == []

    shared = probe_exclusion_diagnosis(
        probe_ids={"GCA_1:c1:0:9-9"},
        substrate_ids={"GCA_1:c1:0:1-2", "GCA_2:c1:0:1-2"},
        n_excluded=0,
    )
    assert shared["reading"] == "disjoint_within_a_shared_namespace"
    assert shared["shared_namespaces"] == ["GCA_1"]


def test_a_probe_member_actually_present_is_reported_as_a_measurement() -> None:
    diagnosis = probe_exclusion_diagnosis(
        probe_ids={"GCA_1:c1:0:1-2"},
        substrate_ids={"GCA_1:c1:0:1-2", "GCA_2:c1:0:1-2"},
        n_excluded=1,
    )
    assert diagnosis["reading"] == "members_excluded"
    assert diagnosis["excluded_ids"] == ["GCA_1:c1:0:1-2"]


def test_an_exclusion_count_that_does_not_describe_this_substrate_is_refused() -> None:
    """Binds the published count to the substrate it claims to describe."""
    with pytest.raises(PublicationError, match="does not describe this substrate"):
        probe_exclusion_diagnosis(
            probe_ids={"GCA_1:c1:0:1-2"},
            substrate_ids={"GCA_1:c1:0:1-2"},
            n_excluded=0,
        )


def test_an_empty_probe_set_or_substrate_is_refused() -> None:
    with pytest.raises(PublicationError, match="probe set is empty"):
        probe_exclusion_diagnosis(probe_ids=set(), substrate_ids={"a:1"}, n_excluded=0)
    with pytest.raises(PublicationError, match="substrate is empty"):
        probe_exclusion_diagnosis(probe_ids={"tier2n:1"}, substrate_ids=set(), n_excluded=0)


def test_id_namespaces_reads_the_prefix_and_is_sorted() -> None:
    assert id_namespaces(["b:1", "a:2", "a:3", "bare"]) == ["a", "b", "bare"]


# ═════════════════════════════════════════════════════════════════════════════
# (3) The retrain disposition
# ═════════════════════════════════════════════════════════════════════════════
def test_the_disposition_states_the_supersession_is_unmet_and_carries_the_parent_field() -> None:
    disposition = retrain_disposition(make_round())
    assert disposition["retrain_leg_run"] is False
    assert disposition["checkpoint_superseded"] is False
    assert disposition["prd_18_1_supersession_status"] == "UNMET"
    assert disposition["parent_checkpoint_in_round_report"] is None


def test_a_round_report_naming_a_parent_checkpoint_fails_the_publication_check() -> None:
    """The evidence for "no retrain" is the round report's own ``parent_checkpoint``."""
    pub = make_publication()
    pub["retrain"] = retrain_disposition(make_round(parent_checkpoint="checkpoints/p2.ckpt"))
    assert any("names a parent checkpoint" in p for p in publication_problems(pub))


# ═════════════════════════════════════════════════════════════════════════════
# The unpinned Stage-2 operating point
# ═════════════════════════════════════════════════════════════════════════════
def test_a_single_point_grid_is_refused_as_a_sensitivity() -> None:
    with pytest.raises(PublicationError, match="at least two grid points"):
        threshold_sensitivity({"0.9": make_round()}, published_threshold=0.9)


def test_the_published_threshold_must_be_a_grid_point() -> None:
    with pytest.raises(PublicationError, match="not one of the grid points"):
        threshold_sensitivity(
            {"0.5": make_round(stage2_threshold=0.5), "0.9": make_round()},
            published_threshold=0.99,
        )


def test_a_mislabelled_grid_point_is_refused_rather_than_published() -> None:
    """Found by review (app r1). ``--grid THRESHOLD=PATH`` takes the label from the
    operator, so without binding it to the report's own ``stage2_threshold`` a sensitivity
    table can be assembled from runs at other operating points — a caption, which is
    exactly what the ``supplies`` digests exist not to be.
    """
    with pytest.raises(PublicationError, match="was produced at stage2_threshold"):
        threshold_sensitivity(
            {"0.5": make_round(stage2_threshold=0.99), "0.9": make_round()},
            published_threshold=0.9,
        )


def test_a_grid_point_carrying_no_threshold_of_its_own_is_refused() -> None:
    with pytest.raises(PublicationError, match="carries no stage2_threshold of its own"):
        threshold_sensitivity(
            {"0.5": make_round(stage2_threshold=None), "0.9": make_round()},
            published_threshold=0.9,
        )


def test_a_grid_point_from_a_refused_round_is_refused() -> None:
    with pytest.raises(PublicationError, match="not a round leg's report that ran"):
        threshold_sensitivity(
            {"0.5": make_round(stage2_threshold=0.5, may_run=False), "0.9": make_round()},
            published_threshold=0.9,
        )


def test_a_grid_whose_mined_set_moves_is_reported_as_moving() -> None:
    """The clause that would make the round's headline threshold-dependent."""
    moved = make_round()
    moved["round"]["mined_ids"] = ["c-other"]
    sensitivity = threshold_sensitivity(
        {"0.5": make_round(stage2_threshold=0.5), "0.9": moved}, published_threshold=0.9
    )
    assert sensitivity["mined_set_invariant"] is False
    assert "MOVES ACROSS THE GRID" in sensitivity["reading"]


def test_an_invariant_mined_set_with_a_moving_split_says_both() -> None:
    moved = make_round()
    moved["round"]["reasons"]["c-stage2"] = "unavailable_backend:high_stage2_posterior"
    sensitivity = threshold_sensitivity(
        {"0.5": make_round(stage2_threshold=0.5), "0.9": moved}, published_threshold=0.9
    )
    assert sensitivity["mined_set_invariant"] is True
    assert sensitivity["spared_split_invariant"] is False
    assert sensitivity["grid"]["0.5"]["n_spared_by_a_passing_disjunct"] == 5
    assert sensitivity["grid"]["0.9"]["n_spared_by_a_passing_disjunct"] == 4


# ═════════════════════════════════════════════════════════════════════════════
# Leg (0)
# ═════════════════════════════════════════════════════════════════════════════
def test_the_plan_summary_strips_the_absolute_repo_root() -> None:
    """The reason the plan report is not committed: each derivation embeds a machine path."""
    summary = plan_leg_summary(make_plan())
    serialized = json.dumps(summary)
    assert "/absolute/path" not in serialized
    assert summary["may_run"] is True
    assert summary["supply_derivations"]["msa_supply_derivation"]["clauses"] == {
        "a": True,
        "b": True,
    }


def test_a_report_from_the_other_leg_is_refused_as_the_plan() -> None:
    with pytest.raises(PublicationError, match="not 'plan'"):
        plan_leg_summary(make_plan(leg="apply-spare-rule"))


def test_a_derivation_without_clauses_is_refused_as_unauditable() -> None:
    plan = make_plan()
    plan["plan"]["synteny_supply_derivation"]["clauses"] = {}
    with pytest.raises(PublicationError, match="no clauses"):
        plan_leg_summary(plan)


def test_a_non_boolean_clause_is_refused() -> None:
    """A clause carrying a path or a message is how the repo_root leak would come back."""
    plan = make_plan()
    plan["plan"]["msa_supply_derivation"]["clauses"]["a"] = "/some/path"
    with pytest.raises(PublicationError, match="are not booleans"):
        plan_leg_summary(plan)


# ═════════════════════════════════════════════════════════════════════════════
# The publication self-check — each clause broken ALONE
# ═════════════════════════════════════════════════════════════════════════════
def test_a_well_formed_publication_has_no_problems() -> None:
    """The negative control. Without it every clause test below could pass on a checker
    that returns a problem for everything ([[raises-test-needs-a-positive-control]])."""
    assert publication_problems(make_publication()) == []


def test_a_checker_that_flags_everything_would_fail_this_control() -> None:
    """States the control's purpose as an assertion, so deleting the one above is red."""
    pub = make_publication()
    assert publication_problems(pub) == []
    pub["leg"] = "plan"
    assert publication_problems(pub) != []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda p: p.__setitem__("leg", "plan"), "not the round leg"),
        (lambda p: p.__setitem__("may_run", False), "refused round"),
        (lambda p: p.__setitem__("stage2_threshold_pinned", True), "§13.1 owns the value"),
        (lambda p: p.__setitem__("round_report_problems", ["boom"]), "self-check failed"),
        (
            lambda p: p["spared_split"].__setitem__("n_spared_by_a_passing_disjunct", 4),
            "does not sum to",
        ),
        (lambda p: p["outcome"].__setitem__("n_mined", 2), "did not decide every candidate"),
        (lambda p: p["outcome"].__setitem__("mined_ids", []), "n_mined disagrees"),
        (
            lambda p: p["probe_exclusion"].__setitem__("n_substrate_candidates", 4),
            "do not describe the same manifest",
        ),
        (
            lambda p: p["stage2_sensitivity"]["grid"]["0.5"].__setitem__(
                "report_stage2_threshold", 0.77
            ),
            "the label does not describe its own bytes",
        ),
        (lambda p: p.__setitem__("stage2_threshold", None), "not a number"),
        (
            lambda p: p["probe_exclusion"].__setitem__("reading", "members_excluded"),
            "does not follow",
        ),
        (lambda p: p["retrain"].__setitem__("retrain_leg_run", True), "claims a retrain"),
        (
            lambda p: p["stage2_sensitivity"].__setitem__("mined_set_invariant", False),
            "disagrees with the grid",
        ),
        (lambda p: p["reproduction"].__setitem__("replayed", {}), "did not reproduce"),
        (lambda p: p["supplies"].__setitem__("covariation_status.json", "short"), "no sha256"),
        (lambda p: p["supplies"].__setitem__("covariation_status.json", "z" * 64), "no sha256"),
        (lambda p: p["supplies"].pop("union_prior.parquet"), "which the replay reads"),
        (lambda p: p["supplies"].pop(f"{GRID_SUPPLY_PREFIX}0.5"), "name no hashed report"),
        (
            lambda p: p["stage2_sensitivity"].__setitem__("spared_split_invariant", True),
            "disagrees with the grid's own sparing compositions",
        ),
        (
            lambda p: p["stage2_sensitivity"]["grid"]["0.5"].__setitem__(
                "candidate_id_set_sha256", "b" * 64
            ),
            "do not describe one round",
        ),
        (
            lambda p: p["spared_split"].__setitem__("passes_per_disjunct", {"x": 99}),
            "per-disjunct detail is unbound",
        ),
        (lambda p: p.__setitem__("may_run", "true"), "not a JSON boolean"),
        (lambda p: p["plan_leg"].__setitem__("ready", "true"), "are not JSON booleans"),
        (lambda p: p["plan_leg"].__setitem__("ready", False), "did not authorise"),
        (
            lambda p: p["plan_leg"]["availability"].__setitem__("any_helix_rscape", False),
            "unavailable backend",
        ),
        (
            lambda p: p["plan_leg"]["supply_derivations"]["msa_supply_derivation"].__setitem__(
                "available", False
            ),
            "cannot evidence",
        ),
        (lambda p: p["plan_leg"].__setitem__("problems", ["boom"]), "plan leg's self-check"),
        (
            lambda p: p["plan_leg"].__setitem__("blocking_disjuncts", ["any_helix_rscape"]),
            "blocking disjuncts",
        ),
    ],
)
def test_each_clause_fires_when_its_own_evidence_is_broken_alone(
    mutate: Any, expected: str
) -> None:
    """One mutation at a time. A publication where every field is right satisfies
    ``all(clauses)`` no matter how the clauses are wired, so each is broken on its own
    ([[all-true-fixture-cannot-test-a-conjunction]])."""
    pub = make_publication()
    mutate(pub)
    problems = publication_problems(pub)
    assert any(expected in p for p in problems), problems


def test_the_reproduction_must_be_about_this_publications_own_outcome() -> None:
    """Two fingerprints can agree with each other about a different round entirely
    ([[gate-must-bind-to-upstream-evidence]])."""
    other = make_round()
    other["round"]["n_mined"] = 7
    other["round"]["mined_ids"] = ["x", "y", "z", "p", "q", "r", "s"]
    pub = make_publication()
    pub["reproduction"] = {
        "reproduced": True,
        "published": outcome_fingerprint(other),
        "replayed": outcome_fingerprint(other),
    }
    problems = publication_problems(pub)
    assert any("disagrees with this publication's outcome" in p for p in problems), problems


def test_a_reproduced_flag_that_contradicts_its_fingerprints_is_caught() -> None:
    """``reproduced`` is re-derived from the two fingerprints, never trusted."""
    pub = make_publication()
    pub["reproduction"]["replayed"] = dict(pub["reproduction"]["replayed"], n_spared=1)
    pub["reproduction"]["reproduced"] = True
    problems = publication_problems(pub)
    assert any("disagrees with its own fingerprints" in p for p in problems), problems


def test_a_one_point_sensitivity_is_caught_by_the_checker_too_not_only_the_builder() -> None:
    """The builder refuses a one-point grid, so this clause is only reachable on a payload
    the builder did not write — a hand-edited or externally-supplied report. Deleting it
    left the whole suite green until this test existed, which is exactly the shape of a
    clause that can never fail ([[gate-clauses-need-re-derivation]]).
    """
    pub = make_publication()
    pub["stage2_sensitivity"]["grid"] = {"0.9": pub["stage2_sensitivity"]["grid"]["0.9"]}
    problems = publication_problems(pub)
    assert any("fewer than two grid points" in p for p in problems), problems


# ═════════════════════════════════════════════════════════════════════════════
# The substrate reader
# ═════════════════════════════════════════════════════════════════════════════
def test_a_duplicate_candidate_id_is_refused_rather_than_merged(tmp_path: Path) -> None:
    """[[duplicate-key-merges-instead-of-colliding]] — a set() would silently absorb it."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"candidates": [{"candidate_id": "a"}, {"candidate_id": "a"}]}),
        encoding="utf-8",
    )
    with pytest.raises(PublicationError, match="collapse to"):
        substrate_ids_from_manifest(path)


def test_an_empty_manifest_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"candidates": []}), encoding="utf-8")
    with pytest.raises(PublicationError, match="no candidate rows"):
        substrate_ids_from_manifest(path)


# ═════════════════════════════════════════════════════════════════════════════
# ── The committed artifact ────────────────────────────────────────────────────
# Everything below reads the published report. These cannot see the code that wrote
# it; they exist to pin what was published, not to test the operators above.
# ═════════════════════════════════════════════════════════════════════════════
def committed_publication() -> dict[str, Any]:
    return json.loads(PUBLICATION_PATH.read_text(encoding="utf-8"))


def test_the_published_round_passes_its_own_self_check_re_derived_here() -> None:
    """Re-runs the checker over the committed bytes rather than reading its ``problems``.

    A ``problems: []`` written into the artifact is the report's own conclusion; running
    the clause set again is what makes it evidence ([[gate-clauses-need-re-derivation]]).
    """
    pub = committed_publication()
    assert publication_problems(pub) == []
    assert pub["problems"] == []


def test_the_published_round_mined_three_of_941_and_retrained_nothing() -> None:
    pub = committed_publication()
    assert pub["leg"] == "apply-spare-rule"
    assert pub["outcome"] == {
        "n_candidates": 941,
        "n_mined": 3,
        "n_masked": 0,
        "n_spared": 938,
        "n_refused_no_coordinates": 0,
        "mined_ids": [
            "GCA_011056645.1:c228:8704:9345-9438",
            "GCA_015478585.1:c26:4608:5040-5154",
            "GCA_040905065.1:c256:4899:5781-5916",
        ],
    }
    assert pub["retrain"]["retrain_leg_run"] is False
    assert pub["retrain"]["prd_18_1_supersession_status"] == "UNMET"


def test_the_published_spared_split_is_recomputed_from_the_grid_point_it_names() -> None:
    """718 / 220 is a fact **at threshold 0.9** — the composition moves across the grid."""
    pub = committed_publication()
    assert pub["stage2_threshold"] == 0.9
    assert pub["stage2_threshold_pinned"] is False
    assert pub["spared_split"]["n_spared_by_a_passing_disjunct"] == 718
    assert pub["spared_split"]["n_spared_by_unavailable_backend_only"] == 220
    assert pub["spared_split"]["n_spared_by_stage2_posterior_alone"] == 79
    grid = pub["stage2_sensitivity"]["grid"]
    assert [grid[k]["n_spared_by_a_passing_disjunct"] for k in ("0.5", "0.9", "0.99")] == [
        724,
        718,
        700,
    ]
    assert pub["stage2_sensitivity"]["mined_set_invariant"] is True
    assert pub["stage2_sensitivity"]["spared_split_invariant"] is False


def test_the_published_probe_exclusion_zero_is_the_structural_one() -> None:
    pub = committed_publication()
    exclusion = pub["probe_exclusion"]
    assert exclusion["n_probe_members"] == 45
    assert exclusion["n_excluded"] == 0
    assert exclusion["reading"] == "structural_namespace_disjoint"
    assert exclusion["probe_id_namespaces"] == ["tier2n"]
    assert exclusion["shared_namespaces"] == []


def test_the_publication_names_no_absolute_path() -> None:
    """The plan leg's report is withheld for exactly this reason; the published one must
    not reintroduce it.

    ⚠ This checked two prefixes — ``/home/`` and ``/exports/`` — which is an allowlist
    wearing a general claim: a ``repo_root`` under ``/scratch/``, ``/Users/`` or ``/data/``
    passed it, and so would the ``/absolute/path/...`` this file's own plan fixture uses.
    Found by review (CLI r2). Asserted on the parsed values instead, so ANY POSIX absolute
    path fails; the provenance entries are repo-relative and stay green.
    """

    def strings(node: Any) -> Any:
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from strings(value)
        elif isinstance(node, list):
            for item in node:
                yield from strings(item)

    values = list(strings(committed_publication()))
    assert values, "the publication carries no strings, so this assertion is vacuous"
    assert [s for s in values if s.startswith("/") and len(s) > 1] == []


def test_the_published_report_is_pinned_by_digest() -> None:
    digest = hashlib.sha256(PUBLICATION_PATH.read_bytes()).hexdigest()
    assert digest == COMMITTED_PUBLICATION_SHA256


# ═════════════════════════════════════════════════════════════════════════════
# Review-found holes (app r1), each reproduced before it was closed
# ═════════════════════════════════════════════════════════════════════════════
def test_a_round_that_actually_excluded_a_probe_member_still_publishes() -> None:
    """``outcome.n_candidates`` is the POST-exclusion count.

    ``apply_remine_spare_rule`` calls ``exclude_probe_members`` before ``mine_round``, so
    comparing the manifest size directly to ``n_candidates`` held on this round only
    because 0 members were excluded — and would have refused every round where the
    exclusion did anything. Found by review; reproduced by construction.
    """
    report = make_round()
    report["round"]["n_excluded_probe_members"] = 1
    report["round"]["excluded_probe_member_ids"] = ["tier2n:0"]
    pub = build_publication(
        round_report=report,
        plan_report=make_plan(),
        substrate_ids={f"sub:{i}" for i in range(9)} | {"tier2n:0"},
        probe_ids={f"tier2n:{i}" for i in range(3)},
        grid={"0.5": make_grid_twin(0.5), "0.9": report},
        supplies=make_supplies({"0.5": None, "0.9": None}),
        reproduction={
            "reproduced": True,
            "published": outcome_fingerprint(report),
            "replayed": outcome_fingerprint(report),
        },
        provenance={},
    )
    assert pub["probe_exclusion"]["n_excluded"] == 1
    assert pub["probe_exclusion"]["reading"] == "members_excluded"
    assert publication_problems(pub) == []


def test_a_substrate_that_does_not_account_for_the_exclusion_is_still_caught() -> None:
    """The positive control for the fix above: relaxing the clause must not delete it."""
    pub = make_publication()
    pub["probe_exclusion"]["n_substrate_candidates"] = 20
    assert any("do not describe the same manifest" in p for p in publication_problems(pub))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("stage2_threshold", None),
        lambda p: p.__setitem__("stage2_threshold", "0.9"),
        lambda p: p["stage2_sensitivity"]["grid"].__setitem__("not-a-number", {}),
    ],
)
def test_the_checker_records_a_problem_rather_than_raising(mutate: Any) -> None:
    """``publication_problems`` runs on payloads it did not write — a committed report, a
    hand-edited one — and ``main`` calls it to decide whether to write at all. A
    ``float()`` on a non-numeric key or a ``None`` threshold escaped as an unhandled
    exception, i.e. a traceback where a recorded refusal was meant. Found by review.
    """
    pub = make_publication()
    mutate(pub)
    problems = publication_problems(pub)  # must not raise
    assert problems


# ═════════════════════════════════════════════════════════════════════════════
# Review-found holes (CLI r2) — the fail-open shapes Python's duck typing allows
# ═════════════════════════════════════════════════════════════════════════════
def test_a_repeated_disjunct_token_would_double_credit_a_criterion() -> None:
    """``passed:x,x`` credited criterion x twice for one candidate in
    ``passes_per_disjunct`` — the tally that says WHICH criterion protected the corpus."""
    with pytest.raises(PublicationError, match="names a disjunct twice"):
        parse_spare_reason("passed:relaxed_architecture,relaxed_architecture")


def test_an_empty_disjunct_token_is_refused_rather_than_silently_dropped() -> None:
    with pytest.raises(PublicationError, match="empty disjunct token"):
        parse_spare_reason("passed:,,relaxed_architecture")


def test_a_non_finite_grid_label_is_refused() -> None:
    """``abs(nan - x) > 1e-12`` is **False**, so a NaN label agreed with every report and
    slipped past the binding — and ``json.dumps`` would then write non-standard ``NaN``."""
    with pytest.raises(PublicationError, match="not a finite threshold"):
        threshold_sensitivity(
            {"nan": make_round(stage2_threshold=float("nan")), "0.9": make_round()},
            published_threshold=0.9,
        )


def test_grid_points_over_different_candidate_sets_are_refused() -> None:
    """Binding a label to its own threshold is not enough: two self-consistent points over
    DIFFERENT manifests would compare two rounds rather than one round at two thresholds.
    """
    other = make_round(stage2_threshold=0.5)
    other["round"]["reasons"] = {f"other-{k}": v for k, v in other["round"]["reasons"].items()}
    other["round"]["spared_ids"] = [f"other-{i}" for i in other["round"]["spared_ids"]]
    other["round"]["mined_ids"] = ["c-mined"]
    with pytest.raises(PublicationError, match="different candidate sets"):
        threshold_sensitivity({"0.5": other, "0.9": make_round()}, published_threshold=0.9)


def test_a_string_boolean_cannot_authorise_the_plan() -> None:
    """``bool("false")`` is True, so coercing an authorisation flag let a string authorise
    a round. Every flag that decides whether the round may publish must be a JSON bool."""
    plan = make_plan()
    plan["plan"]["readiness"]["ready"] = "true"
    with pytest.raises(PublicationError, match="not a JSON boolean"):
        plan_leg_summary(plan)

    availability = make_plan()
    availability["plan"]["availability"]["any_helix_rscape"] = "yes"
    with pytest.raises(PublicationError, match="not JSON booleans"):
        plan_leg_summary(availability)


def test_a_grid_whose_per_disjunct_composition_moves_is_not_invariant() -> None:
    """``spared_split_invariant`` compared ONE of the six split fields, so a grid whose
    per-disjunct tallies moved could be published as composition-invariant."""
    moved = make_round(stage2_threshold=0.5)
    # Same two aggregate halves, different per-disjunct credit: only the full comparison
    # sees this.
    moved["round"]["reasons"]["c-arch"] = "passed:downstream_aaRS_synteny"
    sensitivity = threshold_sensitivity(
        {"0.5": moved, "0.9": make_round()}, published_threshold=0.9
    )
    assert sensitivity["grid"]["0.5"]["n_spared_by_a_passing_disjunct"] == 5
    assert sensitivity["grid"]["0.9"]["n_spared_by_a_passing_disjunct"] == 5
    assert sensitivity["spared_split_invariant"] is False


def test_the_fingerprint_covers_every_outcome_category() -> None:
    """A replay that moved a candidate from ``masked`` to ``refused_no_coordinates`` — two
    different facts about the same locus — matched the fingerprint and read reproduced."""
    a = make_round()
    b = make_round()
    b["round"]["n_masked"] = 1
    b["round"]["n_refused_no_coordinates"] = -1 + b["round"]["n_refused_no_coordinates"] + 1
    b["round"]["n_masked"], b["round"]["n_refused_no_coordinates"] = 1, 0
    a["round"]["n_masked"], a["round"]["n_refused_no_coordinates"] = 0, 1
    assert outcome_fingerprint(a) != outcome_fingerprint(b)


def test_the_checker_does_not_raise_on_malformed_nested_data() -> None:
    """``.get`` on a ``None`` derivation or a non-object grid point raised
    ``AttributeError`` out of a function documented as RETURNING problems."""
    for mutate in (
        lambda p: p["plan_leg"]["supply_derivations"].__setitem__("msa_supply_derivation", None),
        lambda p: p["stage2_sensitivity"]["grid"].__setitem__("0.5", None),
        lambda p: p["stage2_sensitivity"]["grid"].__setitem__("0.5", ["not", "an", "object"]),
    ):
        pub = make_publication()
        mutate(pub)
        assert publication_problems(pub)  # must not raise


# ═════════════════════════════════════════════════════════════════════════════
# Review-found holes (CLI r3) — guards defeated at their own call sites
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("bad", [1.9, True, "2", None])
def test_the_exclusion_count_reaches_its_guard_uncoerced(bad: Any) -> None:
    """``int(1.9)`` is 1 and ``int(True)`` is 1, so coercing at the CALL SITE meant
    ``probe_exclusion_diagnosis``'s ``is_count`` guard never saw the original value — the
    hole that guard exists to close, reintroduced one frame up
    ([[fixed-one-of-two-identical-things]]). Found by review (CLI r3).
    """
    report = make_round()
    report["round"]["n_excluded_probe_members"] = bad
    with pytest.raises(PublicationError, match="not a non-negative integer"):
        build_publication(
            round_report=report,
            plan_report=make_plan(),
            substrate_ids={f"sub:{i}" for i in range(9)},
            probe_ids={f"tier2n:{i}" for i in range(3)},
            grid={"0.5": make_grid_twin(0.5), "0.9": report},
            supplies=make_supplies({"0.5": None, "0.9": None}),
            reproduction={},
            provenance={},
        )


def test_a_string_mined_id_list_is_refused_rather_than_iterated_per_character() -> None:
    """``"abc"`` iterates to three single-character ids whose count matches ``n_mined: 3``,
    so a grid point could publish character-level mined ids."""
    bad = make_round(stage2_threshold=0.5)
    bad["round"]["mined_ids"] = "abc"
    bad["round"]["n_mined"] = 3
    with pytest.raises(PublicationError, match="not a JSON array"):
        threshold_sensitivity({"0.5": bad, "0.9": make_round()}, published_threshold=0.9)


def test_a_non_integer_n_mined_is_named_as_such() -> None:
    """The combined condition reported the wrong cause: a non-integer ``n_mined`` produced
    an error text about distinct ids."""
    bad = make_round(stage2_threshold=0.5)
    bad["round"]["n_mined"] = 1.0
    with pytest.raises(PublicationError, match="not a non-negative integer"):
        threshold_sensitivity({"0.5": bad, "0.9": make_round()}, published_threshold=0.9)


@pytest.mark.parametrize("field", ["problems", "blocking_disjuncts"])
def test_a_string_plan_list_field_is_refused_not_split_into_characters(field: str) -> None:
    """``list("ab")`` is ``['a', 'b']``, so a string ``blocking_disjuncts`` would have been
    reported as two blocking disjuncts; a non-iterable raises ``TypeError`` out of a module
    contracted to refuse with ``PublicationError``."""
    plan = make_plan()
    if field == "problems":
        plan["problems"] = "boom"
    else:
        plan["plan"]["yield"]["blocking_disjuncts"] = "ab"
    with pytest.raises(PublicationError, match="not a JSON array"):
        plan_leg_summary(plan)


def test_a_boolean_where_a_count_belongs_does_not_satisfy_the_sum_clauses() -> None:
    """``isinstance(True, int)`` is True and ``(True, False)`` sums to 1, so a boolean in
    the split or in the four outcome counts satisfied the aggregate clauses."""
    pub = make_publication()
    pub["spared_split"]["n_spared_by_a_passing_disjunct"] = True
    pub["spared_split"]["n_spared_by_unavailable_backend_only"] = False
    pub["spared_split"]["n_spared"] = 1
    assert any("not a pair of counts" in p for p in publication_problems(pub))

    other = make_publication()
    other["outcome"]["n_masked"] = True
    assert any("not all integers" in p for p in publication_problems(other))


def test_the_committed_artifact_is_tracked_by_git_and_extracts_at_its_pin() -> None:
    """Review (CLI r3) filed *"Commit reports/p3/remine_round_report.json — Git does not
    track it"* as a major. It is tracked, so the finding was refused rather than applied;
    this test is what makes that refusal falsifiable instead of an assertion, and it is the
    same claim a prior step's review got wrong ([[review-findings-can-be-factually-wrong]]).
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "reports/p3/remine_round_report.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode == 0, tracked.stderr
    blob = subprocess.run(
        ["git", "show", "HEAD:reports/p3/remine_round_report.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    # Existence and non-emptiness only. Pinning HEAD's digest here would go red in the
    # window between regenerating the artifact and committing it — a test that fails on a
    # legitimate working state rather than on a defect
    # ([[regenerated-report-breaks-shape-lock-test]]). The digest pin lives on the working
    # tree file, above.
    assert blob.returncode == 0 and blob.stdout.strip()
