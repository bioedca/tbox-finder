"""Unit tests for the P3-15'-f criterion-(b) parameter re-derivation.

Four properties carry this file:

* **the conjunction is broken one member at a time** — a fixture where every disjunct
  fails satisfies "mined" no matter which member the code forgot to read, so each of
  the four is flipped ALONE ([[all-true-fixture-cannot-test-a-conjunction]]);
* **the decision rule is tested as a rule, not as its answer** — pinning
  ``sensitive_core`` would stay green under a rule that returns a constant, so the
  floors are asserted by *changing the grid* and watching the selection move
  ([[artifact-pinning-test-cannot-see-the-code]]);
* **each guard has a positive control** — a refusal that fires on everything passes
  ``pytest.raises`` exactly as happily as a correct one
  ([[raises-test-needs-a-positive-control]]);
* **the record rule is asserted by IDENTITY** — a record with one passing and one
  failing query is the only case distinguishing ANY from ALL, and counts alone are
  blind to the two being swapped ([[symmetric-count-fixture-blind-to-inversion]]).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder.mining import architecture_param_measure as apm
from tbox_finder.mining import architecture_param_recommend as rec
from tbox_finder.mining.spare_rule import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
)

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════
#: Two 2-pair helices with a ``UG`` register in the first flanked bulge.
SS_TWO_HELIX = "((..((....))..))"
ROW_WITH_UG = "GGUGGGAAAACCCUGC"
#: One helix, no flanked bulge — fails (b) at every setting in the sweep.
SS_ONE_HELIX = "((((....))))"
ROW_ONE_HELIX = "GGGGAAAACCCC"

LOOSE = apm.ParamTuple("loose", 1, 1, 2, 1, 10_000, 1, False)
CHOSEN = apm.ParamTuple("sensitive_core", 1, 2, 2, 2, 50, 2, False)


def stockholm(ss_cons: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f"{name:20s} {seq}\n" for name, seq in rows)
    return f"# STOCKHOLM 1.0\n{body}{'#=GC SS_cons':20s} {ss_cons}\n//\n"


def write_consensus(root: Path, slug: str, ss_cons: str, row: str, depth: int = 20) -> None:
    rows = [("candidate", row)] + [(f"h{i}", row) for i in range(depth - 1)]
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "msa.sto").write_text(stockholm(ss_cons, rows))


def evidence_row(a: str, c: str, posterior: float | None = 0.01) -> dict:
    return {"covariation_status": a, "synteny_status": c, "stage2_posterior": posterior}


def point(
    *,
    params: apm.ParamTuple = CHOSEN,
    fp_failed: int = 10,
    fp_passed: int = 90,
    fp_failed_with_a: int = 5,
    mined: int = 3,
    damage: int = 8,
    producible: int = 68,
) -> rec.PointResult:
    """A :class:`PointResult` with only the fields a given assertion reads."""
    return rec.PointResult(
        params=params,
        fp_failed=fp_failed,
        fp_passed=fp_passed,
        fp_failed_with_a_failed=fp_failed_with_a,
        mined_by_threshold={"stage2_not_declared": mined},
        control={
            "n_records_losing_a_and_b": damage,
            "n_records_losing_b": damage,
            "n_records_producible": producible,
            "n_queries_decided": 76,
            "n_queries_a_and_b_failed": damage,
            "share_records_losing_a_and_b_ci95": [0.0, 1.0],
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# The grid, and admissibility measured rather than re-encoded
# ═════════════════════════════════════════════════════════════════════════════
def test_grid_is_the_full_cross_product_of_the_shipped_axes():
    expected = (
        len(apm.MIN_NAMED_HELICES_SWEEP)
        * len(apm.MIN_HELIX_PAIRS_SWEEP)
        * len(apm.BULGE_MIN_NT_SWEEP)
        * len(apm.BULGE_MAX_NT_SWEEP)
        * len(apm.NCCA_PAIRING_NT_SWEEP)
        * 2
    )
    points = rec.grid_points()
    assert len(points) == expected
    assert (
        len({p.as_dict()["label"] if False else tuple(sorted(p.as_dict().items())) for p in points})
        == expected
    )


def test_grid_labels_are_distinct_so_a_point_cannot_shadow_another():
    labels = [p.label for p in rec.grid_points()]
    assert len(set(labels)) == len(labels)


def test_grid_carries_the_named_six_verbatim():
    """The recommendation is read against the two reports' settings, so they must be IN it."""
    grid = {tuple(sorted(p.as_dict().items())) for p in rec.grid_points()}
    for named in apm.default_tuples():
        assert tuple(sorted(named.as_dict().items())) in grid, named.label


def test_sweep_arm_returns_the_shipped_refusal_string_for_an_inadmissible_point(tmp_path):
    write_consensus(tmp_path, "s1", SS_TWO_HELIX, ROW_WITH_UG)
    items = apm.read_supply(tmp_path)
    illegal = apm.ParamTuple("illegal", 1, 2, 2, 1, 50, 3, False)  # bulge_min 1 < ncca 3
    outcome = rec.sweep_arm(items, illegal)
    assert isinstance(outcome, str)
    assert "ncca_pairing_nt" in outcome


def test_sweep_arm_returns_verdicts_for_an_admissible_point(tmp_path):
    """The positive control: the refusal above is about the SETTING, not about everything."""
    write_consensus(tmp_path, "s1", SS_TWO_HELIX, ROW_WITH_UG)
    items = apm.read_supply(tmp_path)
    outcome = rec.sweep_arm(items, LOOSE)
    assert isinstance(outcome, dict)
    assert set(outcome) == {"s1"}
    assert outcome["s1"] in (STATUS_PASSED, STATUS_FAILED)


def test_sweep_arm_agrees_with_the_shipped_candidate_state(tmp_path):
    """No second spelling of "what (b) says": the sweep must be candidate_state itself."""
    write_consensus(tmp_path, "s1", SS_TWO_HELIX, ROW_WITH_UG)
    write_consensus(tmp_path, "s2", SS_ONE_HELIX, ROW_ONE_HELIX)
    items = apm.read_supply(tmp_path)
    got = rec.sweep_arm(items, LOOSE)
    assert got == {i.slug: apm.candidate_state(i, LOOSE) for i in items}


# ═════════════════════════════════════════════════════════════════════════════
# The spare rule — a CONJUNCTION, broken one member at a time
# ═════════════════════════════════════════════════════════════════════════════
def test_all_four_disjuncts_failed_is_the_only_mined_case():
    by_id = {"x": evidence_row(STATUS_FAILED, STATUS_FAILED, 0.01)}
    assert rec.mined_ids(by_id, {"x": STATUS_FAILED}, stage2_threshold=0.5) == ["x"]


@pytest.mark.parametrize(
    "a,c,arch_state,posterior,why",
    [
        (STATUS_PASSED, STATUS_FAILED, STATUS_FAILED, 0.01, "criterion (a) passed"),
        (STATUS_FAILED, STATUS_PASSED, STATUS_FAILED, 0.01, "criterion (c) passed"),
        (STATUS_FAILED, STATUS_FAILED, STATUS_PASSED, 0.01, "criterion (b) passed"),
        (STATUS_FAILED, STATUS_FAILED, STATUS_FAILED, 0.99, "the Stage-2 posterior passed"),
        (STATUS_UNAVAILABLE, STATUS_FAILED, STATUS_FAILED, 0.01, "criterion (a) unavailable"),
        (STATUS_FAILED, STATUS_UNAVAILABLE, STATUS_FAILED, 0.01, "criterion (c) unavailable"),
        (STATUS_FAILED, STATUS_FAILED, STATUS_UNAVAILABLE, 0.01, "criterion (b) unavailable"),
        (STATUS_FAILED, STATUS_FAILED, STATUS_FAILED, None, "the posterior is missing"),
    ],
)
def test_flipping_exactly_one_member_spares_the_candidate(a, c, arch_state, posterior, why):
    """Each member ALONE — an all-failed fixture cannot see which member is unread."""
    by_id = {"x": evidence_row(a, c, posterior)}
    assert rec.mined_ids(by_id, {"x": arch_state}, stage2_threshold=0.5) == [], why


def test_a_candidate_absent_from_the_architecture_column_is_spared_not_mined():
    """The fail-open version of this default mines the whole corpus."""
    by_id = {"x": evidence_row(STATUS_FAILED, STATUS_FAILED, 0.01)}
    assert rec.mined_ids(by_id, {}, stage2_threshold=0.5) == []


def test_stage2_not_declared_removes_the_disjunct_rather_than_failing_it():
    """threshold None is D14's phase-conditioning, not 'the posterior failed'."""
    by_id = {"x": evidence_row(STATUS_FAILED, STATUS_FAILED, 0.99)}
    assert rec.mined_ids(by_id, {"x": STATUS_FAILED}, stage2_threshold=None) == ["x"]
    assert rec.mined_ids(by_id, {"x": STATUS_FAILED}, stage2_threshold=0.5) == []


def test_yield_ceiling_counts_the_a_and_c_intersection_not_either_alone():
    by_id = {
        "both": evidence_row(STATUS_FAILED, STATUS_FAILED),
        "only_a": evidence_row(STATUS_FAILED, STATUS_PASSED),
        "only_c": evidence_row(STATUS_PASSED, STATUS_FAILED),
        "neither": evidence_row(STATUS_PASSED, STATUS_PASSED),
    }
    out = rec.yield_ceiling(by_id, ids_with_consensus=list(by_id))
    assert out["n_criterion_a_failed"] == 2
    assert out["n_criterion_c_failed"] == 2
    assert out["n_failing_both_a_and_c"] == 1
    assert out["candidate_ids_at_the_ceiling"] == ["both"]
    assert out["by_stage2_threshold"]["stage2_not_declared"] == {
        "max_mined_if_b_failed_everywhere": 1,
        "min_mined_if_b_passed_everywhere": 0,
    }


def test_the_ceiling_excludes_a_candidate_with_no_consensus():
    """(b) reads 'unavailable' without an msa.sto, and unavailable SPARES."""
    by_id = {"both": evidence_row(STATUS_FAILED, STATUS_FAILED)}
    out = rec.yield_ceiling(by_id, ids_with_consensus=[])
    assert out["n_failing_both_a_and_c"] == 1
    assert out["n_failing_both_a_and_c_with_a_consensus"] == 0
    assert (
        out["by_stage2_threshold"]["stage2_not_declared"]["max_mined_if_b_failed_everywhere"] == 0
    )


# ═════════════════════════════════════════════════════════════════════════════
# Control damage — ANY-passes-spares, asserted by identity
# ═════════════════════════════════════════════════════════════════════════════
REC_A, REC_B, REC_C = "aaa:c0", "bbb:c0", "ccc:c0"
Q_A1, Q_A2 = f"{REC_A}:0:10-20", f"{REC_A}:0:30-40"
Q_B1, Q_B2 = f"{REC_B}:0:10-20", f"{REC_B}:0:30-40"
Q_C1 = f"{REC_C}:0:10-20"


def test_control_records_groups_queries_by_their_record():
    status = {Q_A1: STATUS_FAILED, Q_A2: STATUS_PASSED, Q_B1: STATUS_FAILED}
    assert rec.control_records(status) == {REC_A: [Q_A1, Q_A2], REC_B: [Q_B1]}


def test_one_surviving_query_spares_the_whole_record():
    """The ANY/ALL witness: record A has one failing and one passing query."""
    status = {Q_A1: STATUS_FAILED, Q_A2: STATUS_FAILED, Q_B1: STATUS_FAILED, Q_B2: STATUS_FAILED}
    records = rec.control_records(status)
    arch_by_query = {
        Q_A1: STATUS_FAILED,
        Q_A2: STATUS_PASSED,  # <- the survivor
        Q_B1: STATUS_FAILED,
        Q_B2: STATUS_FAILED,
    }
    out = rec.control_damage(status, records, arch_by_query)
    assert out["n_records_producible"] == 2
    assert out["n_records_losing_b"] == 1, "only record B loses (b) on every query"
    assert out["n_queries_b_failed"] == 3, "the query count sees three; the record rule sees one"


def test_the_conjunction_needs_criterion_a_to_have_failed_too():
    """A record whose queries fail (b) but PASS (a) is spared and must not be counted."""
    status = {Q_A1: STATUS_PASSED, Q_A2: STATUS_PASSED, Q_B1: STATUS_FAILED, Q_B2: STATUS_FAILED}
    records = rec.control_records(status)
    arch_by_query = dict.fromkeys([Q_A1, Q_A2, Q_B1, Q_B2], STATUS_FAILED)
    out = rec.control_damage(status, records, arch_by_query)
    assert out["n_records_losing_b"] == 2, "(b) alone fails on both records"
    assert out["n_records_losing_a_and_b"] == 1, "only record B also failed (a)"


def test_a_record_with_no_producible_query_leaves_the_denominator():
    status = {Q_A1: STATUS_FAILED, Q_C1: STATUS_UNAVAILABLE}
    records = rec.control_records(status)
    out = rec.control_damage(status, records, {Q_A1: STATUS_FAILED})
    assert out["n_records_producible"] == 1


def test_a_query_level_ci_is_refused_and_the_reason_travels_with_it():
    status = {Q_A1: STATUS_FAILED}
    out = rec.control_damage(status, rec.control_records(status), {Q_A1: STATUS_FAILED})
    assert out["query_level_ci95"] is None
    assert "pseudo-replicated" in out["why_no_query_ci"]
    assert out["share_records_losing_a_and_b_ci95"][0] < out["share_records_losing_a_and_b_ci95"][1]


def test_the_missing_criterion_c_on_the_control_is_disclosed_not_absorbed():
    status = {Q_A1: STATUS_FAILED}
    out = rec.control_damage(status, rec.control_records(status), {Q_A1: STATUS_FAILED})
    assert "UPPER" in out["criterion_c_not_in_this_conjunction"]


# ═════════════════════════════════════════════════════════════════════════════
# The decision rule
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "params,violates",
    [
        (apm.ParamTuple("x", 1, 1, 2, 2, 50, 2, False), "min_named_helices >= 2"),
        (apm.ParamTuple("x", 1, 2, 1, 2, 50, 2, False), "min_helix_pairs >= 2"),
        (apm.ParamTuple("x", 1, 2, 2, 1, 50, 1, False), "ncca_pairing_nt >= 2"),
        (apm.ParamTuple("x", 1, 2, 2, 2, 10_000, 2, False), "bulge_max_nt is bounded"),
    ],
)
def test_each_floor_is_violated_alone(params, violates):
    """One member broken at a time — the conjunction cannot hide a floor nobody reads."""
    assert not rec.clears_all_floors(params)
    assert violates not in rec.floors_passed(params)
    assert len(rec.floors_passed(params)) == len(rec.DECISION_FLOORS) - 1


def test_the_recommended_shape_clears_every_floor():
    """The positive control for the four parametrised refusals above."""
    assert rec.clears_all_floors(CHOSEN)
    assert len(rec.floors_passed(CHOSEN)) == len(rec.DECISION_FLOORS)


def test_the_rule_selects_the_most_permissive_survivor_not_the_first_one():
    permissive = point(params=CHOSEN, fp_failed=10)
    strict = point(params=apm.ParamTuple("strict", 1, 3, 2, 4, 50, 2, False), fp_failed=99)
    out = rec.apply_decision_rule([strict, permissive])
    assert out["chosen"] is permissive
    assert out["n_at_the_measured_minimum"] == 1


def test_the_rule_ignores_a_more_permissive_point_that_fails_a_floor():
    """Otherwise the floors are decoration: the argmin would walk straight past them."""
    below_floor = point(params=apm.ParamTuple("loose", 1, 1, 2, 1, 10_000, 1, False), fp_failed=1)
    out = rec.apply_decision_rule([below_floor, point(params=CHOSEN, fp_failed=10)])
    assert out["chosen"].params.as_dict() == CHOSEN.as_dict()
    assert out["n_clearing_every_floor"] == 1


def test_ties_resolve_toward_the_value_that_claims_less():
    wobble_on = point(params=apm.ParamTuple("w1", 1, 2, 2, 2, 50, 2, True), fp_failed=10)
    wobble_off = point(params=CHOSEN, fp_failed=10)
    out = rec.apply_decision_rule([wobble_on, wobble_off])
    assert out["chosen"].params.allow_wobble is False
    assert out["tied_with"] == [wobble_on.params.as_dict()]


def test_a_tie_the_break_order_cannot_resolve_is_refused():
    """Two settings differing only in a parameter NOT proved inert must stop the rule."""
    a = point(params=apm.ParamTuple("a", 1, 2, 2, 2, 50, 2, False), fp_failed=10)
    b = point(params=apm.ParamTuple("b", 1, 3, 2, 2, 50, 2, False), fp_failed=10)
    with pytest.raises(rec.RecommendError, match="tied"):
        rec.apply_decision_rule([a, b])


def test_a_tie_differing_in_a_non_inert_axis_is_refused_even_when_the_keys_differ():
    """The refusal must test the VOCABULARY, not a key collision.

    These two differ in `min_helix_pairs` (which TIE_BREAK_ORDER cannot speak to) *and* in
    `bulge_max_nt` (which it can). A key comparison separates them by `bulge_max_nt` alone
    and publishes a `min_helix_pairs` the rule never chose.
    """
    a = point(params=apm.ParamTuple("a", 1, 2, 2, 2, 20, 2, False), fp_failed=10)
    b = point(params=apm.ParamTuple("b", 1, 2, 3, 2, 50, 2, False), fp_failed=10)
    with pytest.raises(rec.RecommendError, match="min_helix_pairs"):
        rec.apply_decision_rule([a, b])


def test_a_tie_only_inside_the_tie_break_vocabulary_is_resolved_not_refused():
    """Positive control: differing ONLY in bulge_max_nt is exactly what the order exists for."""
    a = point(params=apm.ParamTuple("a", 1, 2, 2, 2, 50, 2, False), fp_failed=10)
    b = point(params=apm.ParamTuple("b", 1, 2, 2, 2, 20, 2, False), fp_failed=10)
    out = rec.apply_decision_rule([a, b])
    assert out["chosen"].params.bulge_max_nt == 20, "the narrower stated filter wins"


def test_the_tie_break_vocabulary_is_derived_from_the_order_not_restated():
    assert tuple(name for name, _ in rec.TIE_BREAK_ORDER) == rec.TIE_BREAK_VOCABULARY


def test_the_tie_refusal_does_not_fire_on_a_single_winner():
    """Positive control: the refusal above is about the tie, not about every input."""
    out = rec.apply_decision_rule([point(params=CHOSEN, fp_failed=10)])
    assert out["chosen"].params.as_dict() == CHOSEN.as_dict()


def test_a_grid_with_no_floor_clearing_point_is_refused_not_defaulted():
    below = point(params=apm.ParamTuple("loose", 1, 1, 2, 1, 10_000, 1, False), fp_failed=1)
    with pytest.raises(rec.RecommendError, match="no admissible grid point"):
        rec.apply_decision_rule([below])


def test_floor_sensitivity_reports_a_floor_that_changes_the_selection():
    chosen = point(params=CHOSEN, fp_failed=10)
    cheaper_below_floor = point(
        params=apm.ParamTuple("one_helix", 1, 1, 2, 2, 50, 2, False), fp_failed=1
    )
    out = rec.floor_sensitivity([chosen, cheaper_below_floor], chosen)
    assert out["min_named_helices >= 2"]["changes_the_selection"] is True
    assert out["min_helix_pairs >= 2"]["changes_the_selection"] is False


def test_floor_sensitivity_marks_a_changed_selection_that_measures_identically():
    """A floor whose removal moves the pick to a setting measuring exactly the same."""
    chosen = point(params=CHOSEN, fp_failed=10, mined=3, damage=8)
    twin_below_the_floor = point(
        params=apm.ParamTuple("twin", 1, 1, 2, 2, 50, 2, False),
        fp_failed=9,  # strictly more permissive, so the tie-break is not consulted
        mined=3,
        damage=8,
    )
    entry = rec.floor_sensitivity([chosen, twin_below_the_floor], chosen)["min_named_helices >= 2"]
    assert entry["changes_the_selection"] is True
    assert entry["measured_identically_to_the_chosen_setting"] is False, "fp_failed differs"


def test_a_floor_whose_removal_leaves_the_rule_undecided_says_so():
    """mhp 1 and mhp 2 measure identically here, so without the floor there is no answer.

    Reporting a tuple anyway would attribute to the rule a decision it refused to make —
    and the tuple reported would be whichever the grid enumerated first.
    """
    chosen = point(params=CHOSEN, fp_failed=10)
    tied_below_the_floor = point(
        params=apm.ParamTuple("twin", 1, 2, 1, 2, 50, 2, False), fp_failed=10
    )
    entry = rec.floor_sensitivity([chosen, tied_below_the_floor], chosen)["min_helix_pairs >= 2"]
    assert entry["selection_without_this_floor"] is None
    assert entry["changes_the_selection"] is None
    assert "tied" in entry["the_rule_cannot_decide_without_it"]


def test_floor_sensitivity_refuses_a_vacuous_identity_when_nothing_moved():
    """Comparing the chosen setting with itself is identical by construction, not evidence."""
    chosen = point(params=CHOSEN, fp_failed=10, mined=3, damage=8)
    sentinel_twin = point(
        params=apm.ParamTuple("twin", 1, 2, 2, 2, 10_000, 2, False),
        fp_failed=10,
        mined=3,
        damage=8,
    )
    entry = rec.floor_sensitivity([chosen, sentinel_twin], chosen)["bulge_max_nt is bounded"]
    assert (
        entry["changes_the_selection"] is False
    ), "the tie-break already prefers the bounded value"
    assert entry["measured_identically_to_the_chosen_setting"] is None


# ═════════════════════════════════════════════════════════════════════════════
# The frontier and the self-criticism block
# ═════════════════════════════════════════════════════════════════════════════
def test_the_frontier_keeps_the_cheapest_point_at_each_yield():
    cheap = point(params=CHOSEN, mined=5, damage=8)
    dear = point(params=apm.ParamTuple("dear", 1, 3, 2, 4, 50, 2, False), mined=5, damage=40)
    rows = rec.frontier([dear, cheap])
    assert len(rows) == 1
    assert rows[0]["fewest_control_records_losing_a_and_b"] == 8
    assert rows[0]["n_admissible_points_at_this_yield"] == 2


def test_the_restricted_frontier_drops_points_below_the_floors():
    inside = point(params=CHOSEN, mined=5, damage=20)
    outside = point(params=apm.ParamTuple("loose", 1, 1, 2, 1, 10_000, 1, False), mined=5, damage=1)
    assert rec.frontier([inside, outside])[0]["fewest_control_records_losing_a_and_b"] == 1
    restricted = rec.frontier([inside, outside], only_floor_clearing=True)
    assert restricted[0]["fewest_control_records_losing_a_and_b"] == 20
    assert restricted[0]["n_admissible_points_at_this_yield"] == 1


def test_an_empty_damage_denominator_is_refused_not_divided_by():
    """A control with no producible record must name its cause, not raise an arithmetic error."""
    status = {Q_A1: STATUS_UNAVAILABLE, Q_B1: STATUS_UNAVAILABLE}
    with pytest.raises(rec.RecommendError, match="no decided corpus"):
        rec.control_damage(status, rec.control_records(status), {})


def test_one_producible_record_is_enough_for_a_denominator():
    """Positive control: the refusal above is about emptiness, not about small n."""
    status = {Q_A1: STATUS_FAILED, Q_B1: STATUS_UNAVAILABLE}
    out = rec.control_damage(status, rec.control_records(status), {Q_A1: STATUS_FAILED})
    assert out["n_records_producible"] == 1


def test_the_bulge_floor_claim_is_measured_against_its_own_sentinel_twin():
    """The rationale says keeping the bound costs nothing; that must be a measurement."""
    chosen = point(params=CHOSEN, fp_failed=10, mined=3, damage=8)
    twin = point(
        params=apm.ParamTuple("twin", 1, 2, 2, 2, rec.BULGE_MAX_UNBOUNDED, 2, False),
        fp_failed=10,
        mined=3,
        damage=8,
    )
    out = rec.bulge_sentinel_comparison([chosen, twin], chosen)
    assert out["sentinel_twin_admissible"] is True
    assert out["measures_identically"] is True


def test_the_sentinel_comparison_reports_a_real_difference_as_one():
    """Positive control: 'measures identically' must be able to come out False."""
    chosen = point(params=CHOSEN, fp_failed=10, mined=3, damage=8)
    twin = point(
        params=apm.ParamTuple("twin", 1, 2, 2, 2, rec.BULGE_MAX_UNBOUNDED, 2, False),
        fp_failed=11,
        mined=4,
        damage=9,
    )
    assert rec.bulge_sentinel_comparison([chosen, twin], chosen)["measures_identically"] is False


def test_an_absent_sentinel_twin_is_reported_rather_than_assumed_identical():
    chosen = point(params=CHOSEN, fp_failed=10)
    out = rec.bulge_sentinel_comparison([chosen], chosen)
    assert out["sentinel_twin_admissible"] is False
    assert "measures_identically" not in out


def test_the_bulge_floor_rationale_cites_a_field_the_report_actually_carries():
    """The original rationale cited floor_sensitivity, whose field is null in exactly this case."""
    floor = next(f for f in rec.DECISION_FLOORS if f.key == "bulge_max_nt is bounded")
    assert "bulge_sentinel_comparison" in floor.rationale
    assert "floor_sensitivity" not in floor.rationale.split("NOT readable off")[0]


def test_dominating_alternatives_finds_a_strictly_better_floor_clearing_point():
    chosen = point(params=CHOSEN, mined=3, damage=8)
    better = point(params=apm.ParamTuple("b", 1, 2, 3, 3, 50, 2, False), mined=5, damage=8)
    out = rec.dominating_alternatives([chosen, better], chosen)
    assert out["n_settings"] == 1
    assert out["max_extra_mined"] == 2
    assert out["examples_by_yield"][0]["n_mined"] == 5


def test_dominating_alternatives_does_not_count_a_costlier_point():
    """The positive control: 'dominating' must mean BOTH coordinates, not just yield."""
    chosen = point(params=CHOSEN, mined=3, damage=8)
    costlier = point(params=apm.ParamTuple("b", 1, 2, 3, 3, 50, 2, False), mined=5, damage=9)
    out = rec.dominating_alternatives([chosen, costlier], chosen)
    assert out["n_settings"] == 0
    assert out["max_extra_mined"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# Anti-selectivity
# ═════════════════════════════════════════════════════════════════════════════
def test_anti_selectivity_reports_the_direction_the_control_falls_through():
    worse_on_the_control = rec.PointResult(
        params=CHOSEN,
        fp_failed=10,
        fp_passed=90,
        fp_failed_with_a_failed=10,
        mined_by_threshold={"stage2_not_declared": 3},
        control={
            "n_records_losing_a_and_b": 8,
            "n_records_producible": 68,
            "n_queries_decided": 100,
            "n_queries_a_and_b_failed": 50,
            "share_records_losing_a_and_b_ci95": [0.0, 1.0],
        },
    )
    out = rec.anti_selectivity(worse_on_the_control, n_fp_decided=100)
    assert out["control_share"] == 0.5
    assert out["fp_share"] == 0.1
    assert out["control_falls_through_more"] is True
    assert out["fp_minus_control_pp"] == pytest.approx(-40.0)
    assert out["intervals_disjoint"] is True


def test_anti_selectivity_is_not_hardwired_to_one_direction():
    """Positive control for the flag above — swap the arms and it must invert."""
    worse_on_the_fp_arm = rec.PointResult(
        params=CHOSEN,
        fp_failed=10,
        fp_passed=90,
        fp_failed_with_a_failed=50,
        mined_by_threshold={"stage2_not_declared": 3},
        control={
            "n_records_losing_a_and_b": 8,
            "n_records_producible": 68,
            "n_queries_decided": 100,
            "n_queries_a_and_b_failed": 10,
            "share_records_losing_a_and_b_ci95": [0.0, 1.0],
        },
    )
    out = rec.anti_selectivity(worse_on_the_fp_arm, n_fp_decided=100)
    assert out["control_falls_through_more"] is False
    assert out["fp_minus_control_pp"] == pytest.approx(40.0)


# ═════════════════════════════════════════════════════════════════════════════
# Binding the report to its evidence
# ═════════════════════════════════════════════════════════════════════════════
def test_a_partial_overlap_of_the_same_size_is_refused():
    """The failure a cardinality check cannot see: same n, different corpus."""
    status = {"a": STATUS_PASSED, "b": STATUS_FAILED, "c": STATUS_FAILED}
    with pytest.raises(rec.RecommendError, match="decided set"):
        rec.assert_supply_is_the_decided_set("FP", ["a", "b", "z"], status)


def test_the_decided_set_check_passes_on_the_real_thing():
    status = {"a": STATUS_PASSED, "b": STATUS_FAILED, "c": STATUS_UNAVAILABLE}
    rec.assert_supply_is_the_decided_set("FP", ["a", "b"], status)


def test_a_consensus_for_an_undecided_candidate_is_refused():
    status = {"a": STATUS_PASSED, "c": STATUS_UNAVAILABLE}
    with pytest.raises(rec.RecommendError, match="did not decide"):
        rec.assert_supply_is_the_decided_set("FP", ["a", "c"], status)


def test_a_duplicated_manifest_id_is_refused_by_the_coverage_check():
    """A set-equality check reads as exact while the row count is already wrong."""
    with pytest.raises(rec.RecommendError, match="duplicated candidate_id"):
        rec.assert_covers_manifest({"a": {}, "b": {}}, ["a", "b", "b"])


def test_distinct_manifest_ids_pass():
    """Positive control for the duplicate refusal."""
    rec.assert_manifest_ids_distinct(["a", "b", "c"])


def test_covering_the_manifest_refuses_both_directions():
    with pytest.raises(rec.RecommendError, match="missing"):
        rec.assert_covers_manifest({"a": {}}, ["a", "b"])
    with pytest.raises(rec.RecommendError, match="extra"):
        rec.assert_covers_manifest({"a": {}, "b": {}}, ["a"])
    rec.assert_covers_manifest({"a": {}, "b": {}}, ["a", "b"])


def test_a_duplicate_candidate_id_is_refused_rather_than_merged(tmp_path):
    """A dict comprehension would MERGE them and every derived count would reconcile."""
    p = tmp_path / "inputs.json"
    row = {"candidate_id": "x", **evidence_row(STATUS_FAILED, STATUS_FAILED)}
    p.write_text(json.dumps({"rows": [row, dict(row)]}))
    with pytest.raises(rec.RecommendError, match="duplicate candidate_id"):
        rec.load_spare_rule_inputs(p)


@pytest.mark.parametrize(
    "row,match",
    [
        (
            {"candidate_id": "x", "covariation_status": "nope", "synteny_status": "failed"},
            "expected",
        ),
        (
            {"candidate_id": "x", "covariation_status": "failed", "synteny_status": "nope"},
            "expected",
        ),
        (
            {
                "candidate_id": "x",
                "covariation_status": "failed",
                "synteny_status": "failed",
                "stage2_posterior": "high",
            },
            "non-numeric",
        ),
        ({"covariation_status": "failed", "synteny_status": "failed"}, "no candidate_id"),
        (
            {
                "candidate_id": "x",
                "covariation_status": "failed",
                "synteny_status": "failed",
                # bool subclasses int, so it reads as 1.0 and SPARES at every threshold.
                "stage2_posterior": True,
            },
            "non-numeric",
        ),
        (
            {
                "candidate_id": "x",
                "covariation_status": "failed",
                "synteny_status": "failed",
                "stage2_posterior": 12.5,
            },
            r"outside \[0, 1\]",
        ),
    ],
)
def test_a_malformed_spare_rule_input_row_is_refused(tmp_path, row, match):
    p = tmp_path / "inputs.json"
    p.write_text(json.dumps({"rows": [row]}))
    with pytest.raises(rec.RecommendError, match=match):
        rec.load_spare_rule_inputs(p)


def test_a_well_formed_spare_rule_input_loads(tmp_path):
    """Positive control for the four refusals above."""
    p = tmp_path / "inputs.json"
    p.write_text(
        json.dumps({"rows": [{"candidate_id": "x", **evidence_row(STATUS_FAILED, STATUS_PASSED)}]})
    )
    loaded = rec.load_spare_rule_inputs(p)
    assert set(loaded["by_id"]) == {"x"}


def test_build_inputs_refuses_a_column_that_does_not_cover_the_manifest(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"candidates": [{"candidate_id": "x"}, {"candidate_id": "y"}]}))
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {"x": STATUS_FAILED}}))
    syn = tmp_path / "syn.json"
    syn.write_text(json.dumps({"status": {"x": STATUS_FAILED, "y": STATUS_FAILED}}))
    post = tmp_path / "p.json"
    post.write_text(json.dumps({"posteriors": {"x": 0.1, "y": 0.2}}))
    with pytest.raises(rec.RecommendError, match="covariation table is missing"):
        rec.build_inputs(
            covariation_status_path=cov,
            synteny_status_path=syn,
            stage2_posteriors_path=post,
            fp_manifest_path=manifest,
        )


def test_build_inputs_refuses_a_column_carrying_an_id_the_manifest_does_not(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"candidates": [{"candidate_id": "x"}]}))
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {"x": STATUS_FAILED, "ghost": STATUS_FAILED}}))
    syn = tmp_path / "syn.json"
    syn.write_text(json.dumps({"status": {"x": STATUS_FAILED}}))
    post = tmp_path / "p.json"
    post.write_text(json.dumps({"posteriors": {"x": 0.1}}))
    with pytest.raises(rec.RecommendError, match="absent from the manifest"):
        rec.build_inputs(
            covariation_status_path=cov,
            synteny_status_path=syn,
            stage2_posteriors_path=post,
            fp_manifest_path=manifest,
        )


def test_build_inputs_joins_and_discloses_the_unanchored_column(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"candidates": [{"candidate_id": "x"}]}))
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {"x": STATUS_FAILED}}))
    syn = tmp_path / "syn.json"
    syn.write_text(json.dumps({"status": {"x": STATUS_PASSED}}))
    post = tmp_path / "p.json"
    post.write_text(json.dumps({"posteriors": {"x": 0.25}}))
    out = rec.build_inputs(
        covariation_status_path=cov,
        synteny_status_path=syn,
        stage2_posteriors_path=post,
        fp_manifest_path=manifest,
    )
    assert out["rows"] == [
        {
            "candidate_id": "x",
            "covariation_status": STATUS_FAILED,
            "synteny_status": STATUS_PASSED,
            "stage2_posterior": 0.25,
        }
    ]
    assert out["sources"]["covariation_status"]["anchored_by"]
    assert out["sources"]["stage2_posteriors"]["anchored_by"] is None
    assert "SENSITIVITY" in out["sources"]["stage2_posteriors"]["warning"]


# ═════════════════════════════════════════════════════════════════════════════
# The published contract
# ═════════════════════════════════════════════════════════════════════════════
def test_the_adr_string_names_the_delegation_the_recommendation_relies_on():
    """A4 is what lets a round supply these values at all; the report must cite it."""
    assert "A4" in rec.ADR
    assert "no defaults" in rec.ADR


def test_every_floor_carries_a_rationale_and_a_source():
    for floor in rec.DECISION_FLOORS:
        body = floor.as_dict()
        assert body["rationale"].strip()
        assert body["source"].startswith("ADR-")


def test_the_decision_rule_statement_names_what_it_refuses_to_do():
    """The rule's whole defence is that it is not an argmax; that must be IN the artifact."""
    assert "argmax" in rec.DECISION_RULE_STATEMENT
    assert "fitted" in rec.DECISION_RULE_STATEMENT


def test_the_tie_break_order_covers_only_parameters_it_can_justify():
    covered = {name for name, _ in rec.TIE_BREAK_ORDER}
    assert covered == {"allow_wobble", "stem_i_nt_threshold", "bulge_max_nt"}


# ═════════════════════════════════════════════════════════════════════════════
# End to end, on a synthetic corpus small enough to hand-check
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def tiny_corpus(tmp_path, monkeypatch):
    """Two FP candidates and two control queries, wired through every file the CLI reads."""
    fp_root = tmp_path / "fp"
    ctrl_root = tmp_path / "ctrl"
    fp_ids = ["GCA_1:c1:0:10-20", "GCA_2:c1:0:10-20"]
    from tbox_finder.mining.architecture_producer import candidate_msa_path

    for cid, (ss, row) in zip(
        fp_ids, [(SS_TWO_HELIX, ROW_WITH_UG), (SS_ONE_HELIX, ROW_ONE_HELIX)], strict=True
    ):
        write_consensus(fp_root, candidate_msa_path(fp_root, cid).parent.name, ss, row)
    ctrl_ids = [Q_A1, Q_A2]
    for cid, (ss, row) in zip(
        ctrl_ids, [(SS_TWO_HELIX, ROW_WITH_UG), (SS_ONE_HELIX, ROW_ONE_HELIX)], strict=True
    ):
        write_consensus(ctrl_root, candidate_msa_path(ctrl_root, cid).parent.name, ss, row)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"candidates": [{"candidate_id": c} for c in fp_ids]}))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "rows": [
                    {"candidate_id": c, **evidence_row(STATUS_FAILED, STATUS_FAILED)}
                    for c in fp_ids
                ]
            }
        )
    )
    ctrl_status = tmp_path / "ctrl_status.json"
    ctrl_status.write_text(json.dumps({"status": dict.fromkeys(ctrl_ids, STATUS_FAILED)}))

    fp_report = tmp_path / "fp_report.json"
    fp_report.write_text(
        json.dumps(
            {
                "supply": {
                    "supply_digest_sha256": apm.supply_digest(apm.read_supply(fp_root)),
                    "supply_origin": "two.amlab:$HOME/somewhere",
                }
            }
        )
    )
    ctrl_report = tmp_path / "ctrl_report.json"
    ctrl_report.write_text(
        json.dumps(
            {
                "supply": {
                    "supply_digest_sha256": apm.supply_digest(apm.read_supply(ctrl_root)),
                    "supply_origin": "two.amlab:$HOME/elsewhere",
                }
            }
        )
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"headline": {"reading": "carried forward"}}))
    return {
        "fp_msa_root": fp_root,
        "control_msa_root": ctrl_root,
        "fp_manifest_path": manifest,
        "spare_rule_inputs_path": inputs,
        "control_status_path": ctrl_status,
        "fp_report_path": fp_report,
        "control_report_path": ctrl_report,
        "comparison_report_path": comparison,
    }


def test_recommend_runs_end_to_end_and_recommends_a_floor_clearing_setting(tiny_corpus):
    body = rec.recommend(**tiny_corpus)
    assert body["pins_nothing"] is True
    assert rec.clears_all_floors(apm.ParamTuple(label="chosen", **body["recommendation"]["params"]))
    assert body["grid"]["n_admissible"] < body["grid"]["n_points_enumerated"]
    assert body["grid"]["n_refused_by_the_shipped_localizer"] > 0
    assert body["arms"]["comparison_headline_carried_forward"] == "carried forward"


def test_recommend_refuses_a_supply_the_committed_report_does_not_describe(tiny_corpus):
    """The digest is the only thing binding a cluster-only supply to a public report."""
    report = Path(tiny_corpus["fp_report_path"])
    report.write_text(json.dumps({"supply": {"supply_digest_sha256": "0" * 64}}))
    with pytest.raises(rec.RecommendError, match="different supply"):
        rec.recommend(**tiny_corpus)


def test_recommend_refuses_a_supply_origin_that_leaks_a_local_path(tiny_corpus):
    with pytest.raises(rec.RecommendError, match="local absolute path"):
        rec.recommend(**tiny_corpus, supply_origin="/home/someone/tbox-scratch/msa")


def test_recommend_accepts_a_cluster_shaped_supply_origin(tiny_corpus):
    """Positive control: the refusal above is about the SHAPE, not about the flag."""
    body = rec.recommend(**tiny_corpus, supply_origin="two.amlab:$HOME/tbox-scratch/msa")
    assert body["pins_nothing"] is True


def test_the_yield_at_the_recommendation_never_exceeds_the_ceiling(tiny_corpus):
    """The relation the whole argument rests on, asserted on real derived numbers."""
    body = rec.recommend(**tiny_corpus)
    ceiling = body["yield_ceiling"]["by_stage2_threshold"]["stage2_not_declared"]
    mined = body["recommendation"]["measured"]["round_yield_n_mined"]["stage2_not_declared"]
    assert mined <= ceiling["max_mined_if_b_failed_everywhere"]
    assert mined >= ceiling["min_mined_if_b_passed_everywhere"]
    assert len(body["recommendation"]["mined_candidate_ids"]) == mined


def test_every_frontier_yield_is_within_the_ceiling(tiny_corpus):
    body = rec.recommend(**tiny_corpus)
    ceiling = body["yield_ceiling"]["by_stage2_threshold"]["stage2_not_declared"][
        "max_mined_if_b_failed_everywhere"
    ]
    for row in body["frontier_all_admissible"]:
        assert row["n_mined"] <= ceiling
