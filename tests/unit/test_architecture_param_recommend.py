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
from tbox_finder.mining.architecture_param_control_compare import CompareError
from tbox_finder.mining.curated_control_sizing import wilson_interval
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
    fp_loci_decided: int = 90,
    fp_loci_failing_a_and_b: int = 5,
    mined: int = 3,
    damage: int = 8,
    b_only_extra: int = 2,
    producible: int = 68,
) -> rec.PointResult:
    """A :class:`PointResult` with only the fields a given assertion reads.

    ``damage`` is the (a)∧(b) record count; ``b_only_extra`` is how many further records
    lose (b) alone, so the two never arrive equal by default.
    """
    return rec.PointResult(
        params=params,
        fp_failed=fp_failed,
        fp_passed=fp_passed,
        fp_failed_with_a_failed=fp_failed_with_a,
        fp_loci_decided=fp_loci_decided,
        fp_loci_failing_a_and_b=fp_loci_failing_a_and_b,
        mined_by_threshold={"stage2_not_declared": mined},
        mined_loci_by_threshold={"stage2_not_declared": mined},
        control={
            # ⚠ (a)∧(b) and (b)-alone are DIFFERENT by default. Giving both the same
            # `damage` made every consumer of this helper blind to the two being
            # interchanged, which is the inversion the file header claims to defend
            # against ([[symmetric-count-fixture-blind-to-inversion]]).
            "n_records_losing_a_and_b": damage,
            "n_records_losing_b": damage + b_only_extra,
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
    # By VALUE, not by label: two points could carry distinct labels and the same seven
    # numbers. (`as_dict()` carries the parameters only — the label is not in it.)
    assert len({tuple(sorted(p.as_dict().items())) for p in points}) == expected


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
    by_id = {"GCA_1.1:c1:0:10-20": evidence_row(STATUS_FAILED, STATUS_FAILED, 0.01)}
    assert rec.mined_ids(by_id, {"GCA_1.1:c1:0:10-20": STATUS_FAILED}, stage2_threshold=0.5) == [
        "GCA_1.1:c1:0:10-20"
    ]


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
    by_id = {"GCA_1.1:c1:0:10-20": evidence_row(a, c, posterior)}
    assert rec.mined_ids(by_id, {"GCA_1.1:c1:0:10-20": arch_state}, stage2_threshold=0.5) == [], why


def test_a_candidate_absent_from_the_architecture_column_is_spared_not_mined():
    """The fail-open version of this default mines the whole corpus."""
    by_id = {"GCA_1.1:c1:0:10-20": evidence_row(STATUS_FAILED, STATUS_FAILED, 0.01)}
    assert rec.mined_ids(by_id, {}, stage2_threshold=0.5) == []


def test_stage2_not_declared_removes_the_disjunct_rather_than_failing_it():
    """threshold None is D14's phase-conditioning, not 'the posterior failed'."""
    by_id = {"GCA_1.1:c1:0:10-20": evidence_row(STATUS_FAILED, STATUS_FAILED, 0.99)}
    assert rec.mined_ids(by_id, {"GCA_1.1:c1:0:10-20": STATUS_FAILED}, stage2_threshold=None) == [
        "GCA_1.1:c1:0:10-20"
    ]
    assert rec.mined_ids(by_id, {"GCA_1.1:c1:0:10-20": STATUS_FAILED}, stage2_threshold=0.5) == []


def test_yield_ceiling_counts_the_a_and_c_intersection_not_either_alone():
    by_id = {
        "GCA_1.1:c1:0:10-20": evidence_row(STATUS_FAILED, STATUS_FAILED),
        "GCA_1.1:c1:0:30-40": evidence_row(STATUS_FAILED, STATUS_PASSED),
        "GCA_1.1:c1:0:50-60": evidence_row(STATUS_PASSED, STATUS_FAILED),
        "GCA_1.1:c1:0:70-80": evidence_row(STATUS_PASSED, STATUS_PASSED),
    }
    out = rec.yield_ceiling(by_id, ids_with_consensus=list(by_id))
    assert out["n_criterion_a_failed"] == 2
    assert out["n_criterion_c_failed"] == 2
    assert out["n_failing_both_a_and_c"] == 1
    assert out["candidate_ids_at_the_ceiling"] == ["GCA_1.1:c1:0:10-20"]
    assert out["by_stage2_threshold"]["stage2_not_declared"] == {
        "max_mined_if_b_failed_everywhere": 1,
        "max_mined_if_b_failed_everywhere_loci": 1,
        "min_mined_if_b_passed_everywhere": 0,
    }


def test_the_ceiling_excludes_a_candidate_with_no_consensus():
    """(b) reads 'unavailable' without an msa.sto, and unavailable SPARES."""
    by_id = {"GCA_1.1:c1:0:10-20": evidence_row(STATUS_FAILED, STATUS_FAILED)}
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


def control_manifest(tmp_path, ids, accession_of=None) -> Path:
    """A control manifest in the shape ``load_control_records`` reads."""
    path = tmp_path / "control_manifest.json"
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": q,
                        "accession": (accession_of or (lambda x: ":".join(x.split(":")[:2])))(q),
                    }
                    for q in ids
                ]
            }
        )
    )
    return path


def test_control_records_groups_queries_by_the_manifests_accession(tmp_path):
    status = {Q_A1: STATUS_FAILED, Q_A2: STATUS_PASSED, Q_B1: STATUS_FAILED}
    manifest = control_manifest(tmp_path, list(status))
    assert rec.control_records(status, manifest) == {REC_A: [Q_A1, Q_A2], REC_B: [Q_B1]}


def test_control_records_refuses_a_status_table_that_is_a_different_corpus(tmp_path):
    """The grouping and the scores must describe the same queries, not merely the same n."""
    manifest = control_manifest(tmp_path, [Q_A1, Q_A2])
    with pytest.raises(rec.RecommendError, match="different corpora"):
        rec.control_records({Q_A1: STATUS_FAILED, Q_B1: STATUS_FAILED}, manifest)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("faled", id="a_typo"),
        pytest.param(None, id="a_null"),
        pytest.param(1, id="a_number"),
        pytest.param("", id="an_empty_string"),
    ],
)
def test_control_records_refuses_a_status_value_it_cannot_spell(tmp_path, bad):
    """An unspellable status is not decided, so it silently leaves the 68-record rate.

    ``control_damage`` reads producibility as ``status[q] in DECIDED_STATES``: a typo does
    not raise, it removes that query from the denominator every share and interval on this
    arm is computed against. The FP arm's columns go through ``checked_status`` in the
    writer AND the reader; this arm did not.
    """
    status = {Q_A1: STATUS_FAILED, Q_A2: bad}
    with pytest.raises(rec.RecommendError, match="control status"):
        rec.control_records(status, control_manifest(tmp_path, [Q_A1, Q_A2]))


def test_the_control_status_check_admits_every_shipped_state(tmp_path):
    """Positive control: `unavailable` is a legitimate value, not a defect to refuse."""
    status = {Q_A1: STATUS_PASSED, Q_A2: STATUS_UNAVAILABLE, Q_B1: STATUS_FAILED}
    assert rec.control_records(status, control_manifest(tmp_path, list(status))) == {
        REC_A: [Q_A1, Q_A2],
        REC_B: [Q_B1],
    }


def test_control_records_names_the_status_table_not_the_manifest(tmp_path):
    """A refusal that sends the reader to the wrong file is one they will disbelieve.

    ``checked_status`` prints its first argument as where the bad value lives, and the bad
    value here comes from the status table.
    """
    manifest = control_manifest(tmp_path, [Q_A1, Q_A2])
    status_path = tmp_path / "ctrl_status_v0.json"
    with pytest.raises(rec.RecommendError, match="ctrl_status_v0.json"):
        rec.control_records({Q_A1: STATUS_FAILED, Q_A2: "faled"}, manifest, status_path=status_path)


def test_control_records_refuses_a_manifest_row_that_contradicts_its_own_id(tmp_path):
    """The shipped loader is the one that owns this refusal; delegating means inheriting it.

    ⚠ The exception type is NAMED. ``pytest.raises(Exception)`` would accept an
    ``AttributeError`` or a ``KeyError`` just as happily, so a regression turning the
    refusal into a crash would keep this green.
    """
    manifest = control_manifest(tmp_path, [Q_A1, Q_A2], accession_of=lambda q: "one:record")
    with pytest.raises((rec.RecommendError, CompareError), match="record|accession"):
        rec.control_records({Q_A1: STATUS_FAILED, Q_A2: STATUS_FAILED}, manifest)


def test_one_surviving_query_spares_the_whole_record(tmp_path):
    """The ANY/ALL witness: record A has one failing and one passing query."""
    status = {Q_A1: STATUS_FAILED, Q_A2: STATUS_FAILED, Q_B1: STATUS_FAILED, Q_B2: STATUS_FAILED}
    records = rec.control_records(status, control_manifest(tmp_path, list(status)))
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


def test_the_conjunction_needs_criterion_a_to_have_failed_too(tmp_path):
    """A record whose queries fail (b) but PASS (a) is spared and must not be counted."""
    status = {Q_A1: STATUS_PASSED, Q_A2: STATUS_PASSED, Q_B1: STATUS_FAILED, Q_B2: STATUS_FAILED}
    records = rec.control_records(status, control_manifest(tmp_path, list(status)))
    arch_by_query = dict.fromkeys([Q_A1, Q_A2, Q_B1, Q_B2], STATUS_FAILED)
    out = rec.control_damage(status, records, arch_by_query)
    assert out["n_records_losing_b"] == 2, "(b) alone fails on both records"
    assert out["n_records_losing_a_and_b"] == 1, "only record B also failed (a)"


def asymmetric_control(tmp_path):
    """One control corpus in which every counter of `control_damage` takes a DIFFERENT value.

    Built so that no two published quantities coincide — the shape a uniform fixture cannot
    have, and the reason the sabotages below bite:

    * record A — ``Q_A1`` fails (a) and (b); ``Q_A2`` fails (b) but PASSES (a)
      ⇒ A loses (b) on every query but is spared under the conjunction, so ANY/ALL and
        (b)-alone/(a)∧(b) are distinguishable in the same record;
    * record B — ``Q_B1`` fails both; ``Q_B2`` is UNAVAILABLE ⇒ not producible
      ⇒ B is damaged on its one decided query, and the query denominator must exclude the
        other, so ``n_queries_decided`` (4) differs from the query total (5);
    * record C — ``Q_C1`` fails (a) only, and PASSES (b) ⇒ C is producible and undamaged.

    Expected: 3 producible records · 1 losing (a)∧(b) · 2 losing (b) · 4 decided queries of
    5 · 3 failing (b) · 2 failing both — six quantities, no two equal by accident.
    """
    status = {
        Q_A1: STATUS_FAILED,
        Q_A2: STATUS_PASSED,
        Q_B1: STATUS_FAILED,
        Q_B2: STATUS_UNAVAILABLE,
        Q_C1: STATUS_FAILED,
    }
    arch = {
        Q_A1: STATUS_FAILED,
        Q_A2: STATUS_FAILED,
        Q_B1: STATUS_FAILED,
        Q_B2: STATUS_FAILED,
        Q_C1: STATUS_PASSED,
    }
    records = rec.control_records(status, control_manifest(tmp_path, list(status)))
    return status, records, arch


def test_the_record_damage_rule_is_ALL_of_the_producible_queries(tmp_path):
    """`all(` -> `any(` at the (a)∧(b) record rule left the whole suite green.

    This is the number the §7 choice was made against — "8 of 68 control records lose both
    protections" — so the quantifier it rests on must be pinned by a fixture that can tell
    ALL from ANY: record A has one query failing both and one passing (a)
    ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    status, records, arch = asymmetric_control(tmp_path)
    out = rec.control_damage(status, records, arch)
    assert out["n_records_producible"] == 3
    assert out["n_records_losing_a_and_b"] == 1, "A survives: its second query passes (a)"
    assert out["n_records_losing_b"] == 2, "A and B lose (b) on every producible query"


def test_the_query_damage_count_keeps_its_criterion_a_conjunct(tmp_path):
    """Dropping `and status[q] == STATUS_FAILED` republished the (b)-only count as (a)∧(b).

    Nothing asserted `n_queries_a_and_b_failed` against real output, so the published
    query-level count could silently become the larger (b)-only one.
    """
    status, records, arch = asymmetric_control(tmp_path)
    out = rec.control_damage(status, records, arch)
    assert out["n_queries_b_failed"] == 3
    assert out["n_queries_a_and_b_failed"] == 2, "Q_A2 fails (b) but passes (a)"
    assert out["n_queries_a_and_b_failed"] != out["n_queries_b_failed"]


def test_the_query_denominator_counts_only_producible_queries(tmp_path):
    """`n_queries_decided` must exclude an `unavailable` query of a producible record.

    Record B contributes one decided query and one spared one, so the denominator (4) and
    the raw query total (5) differ — a record-level fixture cannot show this.
    """
    status, records, arch = asymmetric_control(tmp_path)
    out = rec.control_damage(status, records, arch)
    assert sum(len(q) for q in records.values()) == 5
    assert out["n_queries_decided"] == 4


def test_each_published_share_divides_its_OWN_numerator(tmp_path):
    """Swapping the two `wilson_interval` calls, or the two numerators, stayed green.

    Both shares were only ever asserted where every count was equal, so nothing bound a
    share — or its interval — to the count it claims to summarise.
    """
    status, records, arch = asymmetric_control(tmp_path)
    out = rec.control_damage(status, records, arch)
    assert out["share_records_losing_a_and_b"] == round(1 / 3, 6)
    assert out["share_records_losing_b"] == round(2 / 3, 6)
    # ⚠ By IDENTITY against each interval's OWN numerator, not by containment: at n = 3
    # the two Wilson intervals are wide enough that EACH contains the other's point
    # estimate, so a containment test passes with the two swapped
    # ([[symmetric-count-fixture-blind-to-inversion]]).
    lo_ab, hi_ab = wilson_interval(1, 3)
    lo_b, hi_b = wilson_interval(2, 3)
    assert out["share_records_losing_a_and_b_ci95"] == [round(lo_ab, 6), round(hi_ab, 6)]
    assert out["share_records_losing_b_ci95"] == [round(lo_b, 6), round(hi_b, 6)]
    assert out["share_records_losing_a_and_b_ci95"] != out["share_records_losing_b_ci95"]


def test_a_record_with_no_producible_query_leaves_the_denominator(tmp_path):
    status = {Q_A1: STATUS_FAILED, Q_C1: STATUS_UNAVAILABLE}
    records = rec.control_records(status, control_manifest(tmp_path, list(status)))
    out = rec.control_damage(status, records, {Q_A1: STATUS_FAILED})
    assert out["n_records_producible"] == 1


def test_a_query_level_ci_is_refused_and_the_reason_travels_with_it(tmp_path):
    status = {Q_A1: STATUS_FAILED}
    out = rec.control_damage(
        status,
        rec.control_records(status, control_manifest(tmp_path, list(status))),
        {Q_A1: STATUS_FAILED},
    )
    assert out["query_level_ci95"] is None
    assert "pseudo-replicated" in out["why_no_query_ci"]
    assert out["share_records_losing_a_and_b_ci95"][0] < out["share_records_losing_a_and_b_ci95"][1]


def test_the_missing_criterion_c_on_the_control_is_disclosed_not_absorbed(tmp_path):
    status = {Q_A1: STATUS_FAILED}
    out = rec.control_damage(
        status,
        rec.control_records(status, control_manifest(tmp_path, list(status))),
        {Q_A1: STATUS_FAILED},
    )
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


def test_floor_sensitivity_does_not_call_a_differently_measuring_selection_identical():
    """A floor whose removal moves the pick to a setting that measures DIFFERENTLY.

    The twin is strictly more permissive (``fp_failed`` 9 against 10), so it wins on the
    metric itself and the tie-break is never consulted — and the identity field must then
    read False, not True. Named for what it asserts: the True branch is the test below,
    and a name promising one while asserting the other leaves a reader unable to tell
    which behaviour is pinned.
    """
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


def test_floor_sensitivity_marks_a_changed_selection_that_measures_identically():
    """The TRUE branch: the pick moves, and the setting it moves to measures the same.

    Reached through the bulge floor, the one floor whose parameter is in the tie-break
    vocabulary: the sentinel twin is inadmissible while the floor stands, and once it is
    dropped it ties on every measured coordinate and wins on ``stem_i_nt_threshold``. The
    three-valued field was otherwise pinned only at None and False — on the shipped grid
    it takes those two values alone — so nothing asserted that a genuine identity is
    reported as one ([[clauses-must-guard-emptiness]]).
    """
    chosen = point(
        params=apm.ParamTuple("chosen", 5, 2, 2, 2, 50, 2, False), fp_failed=10, mined=3, damage=8
    )
    sentinel_twin = point(
        params=apm.ParamTuple("twin", 1, 2, 2, 2, 10_000, 2, False),
        fp_failed=10,  # a tie, and the twin's smaller stem_i_nt_threshold wins the order
        mined=3,
        damage=8,
    )
    entry = rec.floor_sensitivity([chosen, sentinel_twin], chosen)["bulge_max_nt is bounded"]
    assert entry["selection_without_this_floor"]["bulge_max_nt"] == 10_000
    assert entry["changes_the_selection"] is True
    assert entry["measured_identically_to_the_chosen_setting"] is True


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
def test_the_frontier_keeps_the_cheapest_point_WHEREVER_it_sits_in_the_input():
    """The argmin, not the last-seen: with the cheap point FIRST the loop must keep it.

    Every earlier fixture listed its cheap point last, so replacing the incumbent
    comparison with "keep whatever came last" left the whole suite green — and the
    frontier is the block that tells the reader what the rule declined to take.
    """
    cheap = point(params=CHOSEN, mined=5, damage=8)
    dear = point(params=apm.ParamTuple("dear", 1, 3, 2, 4, 50, 2, False), mined=5, damage=40)
    for order in ([cheap, dear], [dear, cheap]):
        rows = rec.frontier(order)
        assert len(rows) == 1
        assert rows[0]["fewest_control_records_losing_a_and_b"] == 8, "the argmin, either way"
        assert rows[0]["example_params"] == CHOSEN.as_dict(), "and the row describes THAT point"


def test_the_published_fp_share_divides_failed_by_the_decided_total():
    """`share_failed_of_decided` — the FP arm's headline rate — was asserted nowhere.

    `as_dict` is the sole writer of the published field, and no test called it, so the
    denominator could become `fp_passed` alone and every report would still render.
    """
    body = point(fp_failed=10, fp_passed=90).as_dict()
    assert body["fp_arm"]["share_failed_of_decided"] == round(10 / 100, 6)
    assert body["fp_arm"]["share_failed_of_decided"] != round(10 / 90, 6)


def test_a_point_that_decided_nothing_reports_no_share_rather_than_dividing_by_zero():
    """Positive control for the guard beside it: the `else None` arm is reachable."""
    assert point(fp_failed=0, fp_passed=0).as_dict()["fp_arm"]["share_failed_of_decided"] is None


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


def test_an_empty_damage_denominator_is_refused_not_divided_by(tmp_path):
    """A control with no producible record must name its cause, not raise an arithmetic error."""
    status = {Q_A1: STATUS_UNAVAILABLE, Q_B1: STATUS_UNAVAILABLE}
    with pytest.raises(rec.RecommendError, match="no decided corpus"):
        rec.control_damage(
            status, rec.control_records(status, control_manifest(tmp_path, list(status))), {}
        )


def test_one_producible_record_is_enough_for_a_denominator(tmp_path):
    """Positive control: the refusal above is about emptiness, not about small n."""
    status = {Q_A1: STATUS_FAILED, Q_B1: STATUS_UNAVAILABLE}
    out = rec.control_damage(
        status,
        rec.control_records(status, control_manifest(tmp_path, list(status))),
        {Q_A1: STATUS_FAILED},
    )
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


def test_a_dominating_point_below_a_decision_floor_is_not_counted():
    """The block enumerates what the RULE declined, so it may only list admissible points.

    Both existing fixtures used a floor-clearing alternative, so `clears_all_floors` in
    the filter was never exercised: a setting the rule may not choose could be published
    as a better one it passed over. `mnh=1` is below the min_named_helices floor.
    """
    chosen = point(params=CHOSEN, mined=3, damage=8)
    below = point(params=apm.ParamTuple("loose", 1, 1, 2, 2, 50, 2, False), mined=9, damage=1)
    out = rec.dominating_alternatives([chosen, below], chosen)
    assert out["n_settings"] == 0, "it mines more for less, but the rule cannot take it"
    assert out["max_extra_mined"] == 0


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
    worse_on_the_control = point(
        damage=34, producible=68, fp_loci_failing_a_and_b=10, fp_loci_decided=100
    )
    out = rec.anti_selectivity(worse_on_the_control)
    assert out["control_share"] == 0.5
    assert out["fp_share"] == 0.1
    assert out["control_falls_through_more"] is True
    assert out["fp_minus_control_pp"] == pytest.approx(-40.0)
    assert out["intervals_disjoint"] is True


def test_anti_selectivity_is_stated_on_records_and_loci_not_on_rows():
    """The correction: neither row unit is independent, so neither may carry an interval."""
    r = point(damage=8, producible=68, fp_loci_failing_a_and_b=25, fp_loci_decided=199)
    out = rec.anti_selectivity(r)
    assert out["control_records_failing_a_and_b"] == "8/68"
    assert out["fp_loci_failing_a_and_b"] == "25/199"
    assert "RECORDS" in out["unit"] and "LOCI" in out["unit"]
    assert out["also_at_the_row_level_for_reference"]["no_interval_because"]


def test_anti_selectivity_is_not_hardwired_to_one_direction():
    """Positive control for the flag above — swap the arms and it must invert."""
    worse_on_the_fp_arm = point(
        damage=7, producible=68, fp_loci_failing_a_and_b=50, fp_loci_decided=100
    )
    out = rec.anti_selectivity(worse_on_the_fp_arm)
    assert out["control_falls_through_more"] is False
    assert out["fp_minus_control_pp"] > 0


# ═════════════════════════════════════════════════════════════════════════════
# The manifest's SHAPE — an input, not an invariant (review round 7)
# ═════════════════════════════════════════════════════════════════════════════
def write_json(tmp_path, name, payload) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_the_manifest_may_be_a_top_level_list_of_rows(tmp_path):
    """The shape both call sites MEANT to accept and neither did.

    ``manifest.get("candidates", manifest)`` reads as "the object's rows, or the rows
    themselves" — but a list has no ``.get``, so the second shape raised AttributeError,
    which is outside ``main``'s except tuple: a traceback where a refusal was intended.
    """
    p = write_json(tmp_path, "m.json", [{"candidate_id": "x"}, {"id": "y"}])
    assert rec.read_manifest_ids(p) == ["x", "y"]


def test_the_manifest_may_be_an_object_carrying_its_rows(tmp_path):
    """Positive control: the shipped shape still reads, so the guard is about SHAPE."""
    p = write_json(tmp_path, "m.json", {"candidates": [{"candidate_id": "x"}], "n_candidates": 1})
    assert rec.read_manifest_ids(p) == ["x"]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param({"n_candidates": 941}, "no candidate rows", id="object_without_candidates"),
        pytest.param({"candidates": []}, "no candidate rows", id="empty_candidate_list"),
        pytest.param(941, "no candidate rows", id="a_bare_scalar"),
        pytest.param([], "no candidate rows", id="an_empty_list"),
        pytest.param([{"candidate_id": "x"}, "y"], "not objects", id="a_bare_string_row"),
        pytest.param([{"score": 0.5}], "candidate_id", id="a_row_with_neither_id_key"),
    ],
)
def test_a_manifest_shape_that_cannot_be_read_is_refused_not_raised(tmp_path, payload, match):
    """Every one of these escaped as AttributeError, TypeError or ProducerError before.

    ``ProducerError`` is a ``RuntimeError``, so the row-with-no-id case escaped ``main``
    just as the shape cases did — the refusal has to be re-raised in this module's own
    error type, not merely delegated.
    """
    with pytest.raises(rec.RecommendError, match=match):
        rec.read_manifest_ids(write_json(tmp_path, "m.json", payload))


def test_an_unreadable_manifest_is_refused_with_its_path(tmp_path):
    with pytest.raises(rec.RecommendError, match="cannot read the manifest"):
        rec.read_manifest_ids(tmp_path / "absent.json")


def test_build_inputs_reads_the_manifest_through_the_shared_reader(tmp_path):
    """The wiring, not the helper: a list-shaped manifest must reach the join.

    Testing ``read_manifest_ids`` alone would stay green with either call site still
    holding its own copy of the old expression
    ([[artifact-pinning-test-cannot-see-the-code]]).
    """
    manifest = write_json(tmp_path, "m.json", [{"candidate_id": "x"}])
    out = rec.build_inputs(
        covariation_status_path=write_json(tmp_path, "cov.json", {"status": {"x": STATUS_FAILED}}),
        synteny_status_path=write_json(tmp_path, "syn.json", {"status": {"x": STATUS_PASSED}}),
        stage2_posteriors_path=write_json(tmp_path, "p.json", {"posteriors": {"x": 0.25}}),
        fp_manifest_path=manifest,
    )
    assert [row["candidate_id"] for row in out["rows"]] == ["x"]


def test_recommend_reads_the_manifest_through_the_shared_reader(tiny_corpus, tmp_path):
    """The second call site, wired the same way — the one a single fix would have missed.

    ([[fixed-one-of-two-identical-things]] — three review rounds running have found a fix
    landed on one of two identical call sites.)
    """
    rows = json.loads(Path(tiny_corpus["fp_manifest_path"]).read_text())["candidates"]
    as_a_list = write_json(tmp_path, "l.json", rows)
    body = rec.recommend(**{**tiny_corpus, "fp_manifest_path": as_a_list})
    assert body["pins_nothing"] is True


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


def test_a_supply_naming_one_candidate_twice_is_refused():
    """`set(supply_ids)` merged the repeat and the equality then read as exact.

    The same shape `assert_covers_manifest` refuses on the manifest side
    ([[duplicate-key-merges-instead-of-colliding]]): two consensuses for one candidate
    would be counted twice by every per-item loop while the set comparison agreed.
    """
    status = {"a": STATUS_PASSED, "b": STATUS_FAILED}
    with pytest.raises(rec.RecommendError, match="more than once"):
        rec.assert_supply_is_the_decided_set("FP", ["a", "b", "b"], status)


def test_two_candidates_claiming_one_slug_are_refused_not_merged(tmp_path):
    """A dict comprehension kept the LAST id and dropped the other, counts reconciling.

    The reachable trigger is a repeated id — `candidate_slug` appends a 12-hex digest of
    the exact id, so two DISTINCT ids cannot collide — and a repeat exercises the same
    merge branch the guard exists for ([[duplicate-key-merges-instead-of-colliding]]).
    """
    with pytest.raises(rec.RecommendError, match="claimed by two candidate ids"):
        rec.index_by_slug("FP", tmp_path, [Q_A1, Q_A1])


def test_the_slug_index_maps_distinct_candidates_to_distinct_slugs(tmp_path):
    """Positive control, and the property the guard assumes: the slug is 1:1 on ids."""
    index = rec.index_by_slug("FP", tmp_path, [Q_A1, Q_A2])
    assert sorted(index.values()) == sorted([Q_A1, Q_A2])
    assert len(index) == 2


def test_the_duplicate_refusal_does_not_fire_on_a_distinct_supply():
    """Positive control — and the ids arrive as a GENERATOR at one of the two call sites,
    so materialising them must not consume the supply before the comparison."""
    status = {"a": STATUS_PASSED, "b": STATUS_FAILED}
    rec.assert_supply_is_the_decided_set("FP", iter(["a", "b"]), status)


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


def posterior_inputs(tmp_path, literal: str) -> Path:
    """A one-row inputs file whose posterior is written as a raw JSON literal.

    ``json.dumps`` cannot express these two: an integer beyond float range comes back
    from ``json.loads`` as a Python ``int``, and ``1e400`` as ``inf``.
    """
    p = tmp_path / "inputs.json"
    p.write_text(
        '{"rows": [{"candidate_id": "x", "covariation_status": "failed", '
        f'"synteny_status": "failed", "stage2_posterior": {literal}}}]}}'
    )
    return p


def test_a_posterior_too_large_for_a_float_is_refused_not_raised(tmp_path):
    """`float(10**400)` raises OverflowError — an ArithmeticError `main` does not catch.

    JSON bounds no integer, so the column can carry one; the CLI would have ended in a
    traceback where every other malformed posterior gets `refused:`.
    """
    with pytest.raises(rec.RecommendError, match="too large to represent"):
        rec.load_spare_rule_inputs(posterior_inputs(tmp_path, "9" * 400))


def test_a_posterior_that_parses_to_infinity_is_refused_by_the_range_test(tmp_path):
    """Positive control for the branch above: `1e400` parses to `inf`, not to an int."""
    with pytest.raises(rec.RecommendError, match=r"outside \[0, 1\]"):
        rec.load_spare_rule_inputs(posterior_inputs(tmp_path, "1e400"))


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
    cov.write_text(json.dumps({"status": {"GCA_1.1:c1:0:10-20": STATUS_FAILED}}))
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
    syn.write_text(json.dumps({"status": {"GCA_1.1:c1:0:10-20": STATUS_FAILED}}))
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
    cid = "GCA_1.1:c1:0:10-20"
    manifest.write_text(json.dumps({"candidates": [{"candidate_id": cid}]}))
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {cid: STATUS_FAILED}}))
    syn = tmp_path / "syn.json"
    syn.write_text(json.dumps({"status": {cid: STATUS_PASSED}}))
    post = tmp_path / "p.json"
    post.write_text(json.dumps({"posteriors": {cid: 0.25}}))
    out = rec.build_inputs(
        covariation_status_path=cov,
        synteny_status_path=syn,
        stage2_posteriors_path=post,
        fp_manifest_path=manifest,
    )
    assert out["rows"] == [
        {
            "candidate_id": cid,
            "covariation_status": STATUS_FAILED,
            "synteny_status": STATUS_PASSED,
            "stage2_posterior": 0.25,
        }
    ]
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
        assert body["source"].strip()


def test_a_floor_that_is_an_assumption_says_so_rather_than_citing_an_ADR():
    """min_helix_pairs has no ADR clause behind it; the source must not imply one."""
    floor = next(f for f in rec.DECISION_FLOORS if f.key == "min_helix_pairs >= 2")
    assert "ASSUMPTION" in floor.source
    assert "NO concept of stacked-pair depth" in floor.rationale


def test_the_min_named_helices_floor_names_the_competing_reading():
    """The level is the §7 choice; a rationale that hid the other reading would decide it."""
    floor = next(f for f in rec.DECISION_FLOORS if f.key.startswith("min_named_helices"))
    assert "§7" in floor.rationale
    assert "named_elements_present" in floor.source
    assert set(rec.MIN_NAMED_HELICES_READINGS) >= {2, 3}


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
    ctrl_manifest = control_manifest(tmp_path, ctrl_ids)

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
    comparison.write_text(
        json.dumps(
            {
                "headline": {"reading": "carried forward"},
                # The comparison's own statement of WHICH two arms it read, bound to the
                # supplies here the same way the two measurement reports are.
                "arms": {
                    "fp": {"supply_digest_sha256": apm.supply_digest(apm.read_supply(fp_root))},
                    "control": {
                        "supply_digest_sha256": apm.supply_digest(apm.read_supply(ctrl_root))
                    },
                },
            }
        )
    )
    return {
        "fp_msa_root": fp_root,
        "control_msa_root": ctrl_root,
        "fp_manifest_path": manifest,
        "spare_rule_inputs_path": inputs,
        "control_status_path": ctrl_status,
        "control_manifest_path": ctrl_manifest,
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


def test_recommend_refuses_a_control_supply_that_is_not_criterion_as_decided_set(tiny_corpus):
    """The CONTROL call site of the decided-set guard, driven through `recommend`.

    The guard's unit tests only ever pass label "FP", and the fixture makes the FP arm's
    two id sets identical, so deleting either call site left the suite green. Here the
    control table calls one query `unavailable` while its consensus sits on disk.
    """
    p = Path(tiny_corpus["control_status_path"])
    status = json.loads(p.read_text())["status"]
    status[sorted(status)[0]] = STATUS_UNAVAILABLE
    p.write_text(json.dumps({"status": status}))
    with pytest.raises(rec.RecommendError, match="control supply is not criterion"):
        rec.recommend(**tiny_corpus)


def test_recommend_refuses_a_consensus_whose_slug_has_no_manifest_candidate(tiny_corpus, tmp_path):
    """The orphan check, driven through `recommend` — deleting it left the suite green.

    A consensus the manifest does not name means the supply and the manifest describe
    different corpora; the digests are re-stamped here so the *earlier* binding gate
    passes and this guard is the one under test.
    """
    from tbox_finder.mining.architecture_producer import candidate_msa_path

    ctrl_root = Path(tiny_corpus["control_msa_root"])
    write_consensus(
        ctrl_root,
        candidate_msa_path(ctrl_root, "zzz:c0:0:10-20").parent.name,
        SS_TWO_HELIX,
        ROW_WITH_UG,
    )
    digest = apm.supply_digest(apm.read_supply(ctrl_root))
    report = Path(tiny_corpus["control_report_path"])
    body = json.loads(report.read_text())
    body["supply"]["supply_digest_sha256"] = digest
    report.write_text(json.dumps(body))
    comparison = Path(tiny_corpus["comparison_report_path"])
    cmp_body = json.loads(comparison.read_text())
    cmp_body["arms"]["control"]["supply_digest_sha256"] = digest
    comparison.write_text(json.dumps(cmp_body))
    with pytest.raises(rec.RecommendError, match="resolve to no manifest"):
        rec.recommend(**tiny_corpus)


def test_recommend_names_the_control_status_file_in_a_status_refusal(tiny_corpus):
    """The WIRING of the path, not the helper: the kwarg is droppable in silence.

    ``control_records`` takes ``status_path`` keyword-only with a default, so removing it
    from the call site leaves the refusal firing — pointing at ``control_manifest.json``,
    a file whose contents are fine ([[artifact-pinning-test-cannot-see-the-code]]).
    """
    p = Path(tiny_corpus["control_status_path"])
    status = json.loads(p.read_text())["status"]
    status[sorted(status)[0]] = "faled"
    p.write_text(json.dumps({"status": status}))
    with pytest.raises(rec.RecommendError, match="ctrl_status.json"):
        rec.recommend(**tiny_corpus)


def test_recommend_refuses_a_comparison_report_about_other_supplies(tiny_corpus):
    """The headline is a READING of two arms; the file hash identifies only the file.

    Every other number in the payload is bound to the supplies on disk by digest. The
    carried-forward sentence was not, so a comparison of a different pair of arms could be
    quoted beside them ([[gate-must-bind-to-upstream-evidence]]).
    """
    comparison = json.loads(Path(tiny_corpus["comparison_report_path"]).read_text())
    comparison["arms"]["control"]["supply_digest_sha256"] = "0" * 64
    Path(tiny_corpus["comparison_report_path"]).write_text(json.dumps(comparison))
    with pytest.raises(rec.RecommendError, match="reading of different arms"):
        rec.recommend(**tiny_corpus)


@pytest.mark.parametrize(
    "headline",
    [
        pytest.param("the FP arm loses more", id="a_string"),
        pytest.param(["a"], id="a_list"),
        pytest.param(3, id="a_number"),
    ],
)
def test_recommend_refuses_a_comparison_report_whose_headline_is_not_an_object(
    tiny_corpus, headline
):
    """The round-8 guard bound `arms`; the field it protects was still an unguarded `.get`.

    `comparison.get("headline", {})` substitutes its default only when the key is ABSENT,
    so a present-but-not-an-object value reached `.get` and raised `AttributeError` —
    outside `main`'s except tuple, i.e. a traceback where a refusal was intended, the very
    escape `read_manifest_ids` was written to close.
    """
    p = Path(tiny_corpus["comparison_report_path"])
    body = json.loads(p.read_text())
    body["headline"] = headline
    p.write_text(json.dumps(body))
    with pytest.raises(rec.RecommendError, match="'headline' is"):
        rec.recommend(**tiny_corpus)


def test_a_headline_block_that_is_simply_absent_is_not_an_error(tiny_corpus):
    """Positive control: ABSENT stays permissive — only a wrong SHAPE is refused."""
    p = Path(tiny_corpus["comparison_report_path"])
    body = json.loads(p.read_text())
    del body["headline"]
    p.write_text(json.dumps(body))
    assert rec.recommend(**tiny_corpus)["arms"]["comparison_headline_carried_forward"] is None


@pytest.mark.parametrize("key", ["fp_report_path", "control_report_path"])
def test_recommend_refuses_a_measurement_report_that_is_not_an_object(tiny_corpus, key):
    """A report file holding a list or a scalar refuses instead of tracebacking.

    The digest binding is the ONLY thing tying a cluster-only supply to a committed public
    report, so it must fail by name.
    """
    Path(tiny_corpus[key]).write_text(json.dumps([{"supply": {}}]))
    with pytest.raises(rec.RecommendError, match="must be a JSON object"):
        rec.recommend(**tiny_corpus)


def test_recommend_refuses_a_report_whose_supply_block_is_null(tiny_corpus):
    """`.get("supply", {})` defaulted only for an ABSENT key; `null` reached `.get`."""
    Path(tiny_corpus["fp_report_path"]).write_text(json.dumps({"supply": "cluster"}))
    with pytest.raises(rec.RecommendError, match="'supply' is"):
        rec.recommend(**tiny_corpus)


def test_recommend_refuses_a_comparison_report_that_records_no_digest(tiny_corpus):
    """Absent binds nothing: read as "no disagreement found" it is a vacuous pass."""
    Path(tiny_corpus["comparison_report_path"]).write_text(
        json.dumps({"headline": {"reading": "carried forward"}})
    )
    with pytest.raises(rec.RecommendError, match="no 'arms' block"):
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


# ═════════════════════════════════════════════════════════════════════════════
# The locus — the FP arm's independent unit (review round 3)
# ═════════════════════════════════════════════════════════════════════════════
def test_two_tiling_windows_calling_the_same_span_are_one_locus():
    """The defect this exists to fix: 79 of 278 FP 'candidates' were second calls."""
    a = "GCA_1.1:c106:15360:16034-16111"
    b = "GCA_1.1:c106:15872:16034-16111"
    assert rec.locus_of(a) == rec.locus_of(b) == "GCA_1.1:c106:16034-16111"


def test_different_spans_on_one_contig_are_different_loci():
    """Positive control: locus_of must not collapse everything to the contig."""
    assert rec.locus_of("GCA_1.1:c106:15360:100-200") != rec.locus_of("GCA_1.1:c106:15360:300-400")


def test_an_id_that_is_not_locus_shaped_is_refused_not_guessed():
    with pytest.raises(rec.RecommendError, match="cannot be derived"):
        rec.locus_of("not:a:locus")


def test_the_decided_states_come_from_the_shipped_status_constants():
    assert rec.DECIDED_STATES == (STATUS_PASSED, STATUS_FAILED)


# ═════════════════════════════════════════════════════════════════════════════
# The §7 axis: every min_named_helices level, priced
# ═════════════════════════════════════════════════════════════════════════════
def test_every_min_named_helices_level_is_priced():
    grid = [
        point(params=apm.ParamTuple(f"m{m}", 1, m, 2, 2, 50, 2, False), fp_failed=10 * m, mined=m)
        for m in (2, 3, 4)
    ]
    out = rec.selection_by_min_named_helices(grid)
    levels = {r["min_named_helices"]: r for r in out["levels"]}
    assert set(levels) == set(rec.MIN_NAMED_HELICES_LEVELS)
    for m in (2, 3, 4):
        assert levels[m]["selection"]["min_named_helices"] == m
        assert levels[m]["round_yield_n_mined"] == m


def test_a_level_the_rule_cannot_decide_is_reported_not_skipped():
    tied = [
        point(params=apm.ParamTuple("a", 1, 2, 2, 2, 50, 2, False), fp_failed=10),
        point(params=apm.ParamTuple("b", 1, 2, 3, 2, 50, 2, False), fp_failed=10),
    ]
    levels = {r["min_named_helices"]: r for r in rec.selection_by_min_named_helices(tied)["levels"]}
    assert levels[2]["selection"] is None
    assert levels[2]["the_rule_cannot_decide_at_this_level"]


def test_the_two_supported_readings_are_both_carried():
    assert "plural" in rec.MIN_NAMED_HELICES_READINGS[2].lower()
    assert "canonical" in rec.MIN_NAMED_HELICES_READINGS[3].lower()
    assert "named_elements_present" in rec.MIN_NAMED_HELICES_READINGS[3]


# ═════════════════════════════════════════════════════════════════════════════
# Tie-break honesty, and the origin guard
# ═════════════════════════════════════════════════════════════════════════════
def test_a_single_valued_tie_break_axis_is_reported_as_unable_to_fire():
    grid = [
        point(params=apm.ParamTuple("a", 1, 2, 2, 2, 50, 2, False)),
        point(params=apm.ParamTuple("b", 1, 2, 2, 2, 20, 2, True)),
    ]
    out = rec.tie_break_exercised(grid)
    assert out["can_fire"]["stem_i_nt_threshold"] is False, "one value on the swept grid"
    assert out["can_fire"]["allow_wobble"] is True
    assert out["can_fire"]["bulge_max_nt"] is True


def test_an_origin_carried_from_another_report_clears_the_same_guard_as_the_flag():
    with pytest.raises(rec.RecommendError, match="local absolute path"):
        rec.checked_origin("fp", "/home/someone/tbox-scratch/msa")


def test_a_cluster_shaped_origin_is_carried_forward():
    """Positive control: the guard is about the shape, not about carrying origins at all."""
    assert rec.checked_origin("fp", "two.amlab:$HOME/tbox-scratch/msa").startswith("two.amlab")
    assert rec.checked_origin("fp", None) is None


def test_evaluate_point_counts_the_fp_arm_by_locus_not_by_manifest_row(tmp_path):
    """Two candidates from overlapping windows on one span must count once, not twice."""
    from tbox_finder.mining.architecture_producer import candidate_msa_path

    fp_root = tmp_path / "fp"
    twins = ["GCA_9.1:c7:1024:5000-5100", "GCA_9.1:c7:1536:5000-5100"]
    elsewhere = "GCA_9.1:c7:2048:9000-9100"
    for cid in [*twins, elsewhere]:
        write_consensus(
            fp_root, candidate_msa_path(fp_root, cid).parent.name, SS_ONE_HELIX, ROW_ONE_HELIX
        )
    items = apm.read_supply(fp_root)
    by_slug = {candidate_msa_path(fp_root, c).parent.name: c for c in [*twins, elsewhere]}
    # ⚠ ASYMMETRIC ON PURPOSE. `elsewhere` fails (b) but PASSES (a), so a version of
    # evaluate_point that drops the (a) conjunct — or reads (c) instead — counts 3 where
    # the correct one counts 2. An all-FAILED fixture cannot tell those three apart
    # ([[all-true-fixture-cannot-test-a-conjunction]]).
    by_id = {c: evidence_row(STATUS_FAILED, STATUS_FAILED) for c in twins}
    by_id[elsewhere] = evidence_row(STATUS_PASSED, STATUS_FAILED)

    ctrl_root = tmp_path / "ctrl"
    write_consensus(
        ctrl_root, candidate_msa_path(ctrl_root, Q_A1).parent.name, SS_TWO_HELIX, ROW_WITH_UG
    )
    ctrl_items = apm.read_supply(ctrl_root)
    ctrl_status = {Q_A1: STATUS_FAILED}
    result = rec.evaluate_point(
        CHOSEN,
        fp_items=items,
        fp_id_by_slug=by_slug,
        by_id=by_id,
        control_items=ctrl_items,
        control_id_by_slug={candidate_msa_path(ctrl_root, Q_A1).parent.name: Q_A1},
        control_status=ctrl_status,
        control_record_index={REC_A: [Q_A1]},
    )
    assert result.fp_failed + result.fp_passed == 3, "three manifest rows"
    assert result.fp_loci_decided == 2, "the twins are one locus, `elsewhere` another"
    assert result.fp_failed == 3, "(b) fails on all three"
    assert result.fp_failed_with_a_failed == 2, "only the two whose (a) also failed"
    assert result.fp_loci_failing_a_and_b == 1, "and those two are one locus"
    # ⚠ The THIRD locus-collapse site. `mined_by_threshold` counts manifest rows and
    # `mined_loci_by_threshold` collapses them; nothing asserted the second, so the
    # published `round_yield_n_mined_loci` could silently become the row count — the very
    # overstatement of "how much DNA" this locus unit exists to prevent.
    assert result.mined_by_threshold["stage2_not_declared"] == 2, "both twins are mined rows"
    assert result.mined_loci_by_threshold["stage2_not_declared"] == 1, "one piece of DNA"


# ═════════════════════════════════════════════════════════════════════════════
# Review round 4 — both units, and the anchors that must be checked
# ═════════════════════════════════════════════════════════════════════════════
def test_the_ceiling_is_published_in_loci_as_well_as_manifest_rows():
    """15 rows are 12 loci on the real supply; a row count alone overstates the DNA."""
    twins = ["GCA_9.1:c7:1024:5000-5100", "GCA_9.1:c7:1536:5000-5100"]
    by_id = {c: evidence_row(STATUS_FAILED, STATUS_FAILED) for c in twins}
    out = rec.yield_ceiling(by_id, ids_with_consensus=twins)
    band = out["by_stage2_threshold"]["stage2_not_declared"]
    assert band["max_mined_if_b_failed_everywhere"] == 2, "two manifest rows"
    assert band["max_mined_if_b_failed_everywhere_loci"] == 1, "one locus"
    assert out["distinct_loci_at_the_ceiling"] == 1


def test_an_anchor_is_claimed_only_when_the_digest_actually_matches(tmp_path):
    target = tmp_path / "table.json"
    target.write_text('{"status": {}}')
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"provenance": {"extra": {"external_inputs": {"x": "deadbeef"}}}}))
    assert (
        rec.anchor_of(target, report, ("provenance", "extra", "external_inputs", "x")) is None
    ), "a digest that does not match must not be reported as anchored"


def test_an_anchor_is_claimed_when_the_digest_does_match(tmp_path):
    """Positive control: anchor_of must be able to say yes, or it is a constant None."""
    target = tmp_path / "table.json"
    target.write_text('{"status": {}}')
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"provenance": {"extra": {"external_inputs": {"x": rec.sha256_of(target)}}}})
    )
    got = rec.anchor_of(target, report, ("provenance", "extra", "external_inputs", "x"))
    assert got and "records this exact digest" in got


def test_a_missing_anchor_report_is_not_treated_as_an_anchor(tmp_path):
    target = tmp_path / "table.json"
    target.write_text("{}")
    assert rec.anchor_of(target, tmp_path / "absent.json", ("anything",)) is None


def test_the_manifest_coverage_check_is_not_satisfied_by_a_matching_cardinality():
    """A len()==len() degradation passes every equal/missing/extra case a naive test tries."""
    with pytest.raises(rec.RecommendError, match="do not cover|extra"):
        rec.assert_covers_manifest({"a": {}, "z": {}}, ["a", "b"])


def test_intervals_disjoint_can_come_out_false():
    """The negative control the flag needs: overlapping arms must read False."""
    overlapping = point(damage=30, producible=68, fp_loci_failing_a_and_b=60, fp_loci_decided=199)
    assert rec.anti_selectivity(overlapping)["intervals_disjoint"] is False


# ═════════════════════════════════════════════════════════════════════════════
# Review round 5 — the WRITER validates too, with the reader's own rules
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "posterior,match",
    [
        (True, "non-numeric"),
        ("high", "non-numeric"),
        (12.5, r"outside \[0, 1\]"),
        (-0.1, r"outside \[0, 1\]"),
    ],
)
def test_build_inputs_refuses_a_bad_posterior_at_the_join(tmp_path, posterior, match):
    """`float(True)` is 1.0 — commit it and the loader can never see it again."""
    cid = "GCA_1.1:c1:0:10-20"
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"candidates": [{"candidate_id": cid}]}))
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {cid: STATUS_FAILED}}))
    syn = tmp_path / "syn.json"
    syn.write_text(json.dumps({"status": {cid: STATUS_FAILED}}))
    post = tmp_path / "p.json"
    post.write_text(json.dumps({"posteriors": {cid: posterior}}))
    with pytest.raises(rec.RecommendError, match=match):
        rec.build_inputs(
            covariation_status_path=cov,
            synteny_status_path=syn,
            stage2_posteriors_path=post,
            fp_manifest_path=manifest,
        )


def test_build_inputs_refuses_a_bad_status_at_the_join(tmp_path):
    cid = "GCA_1.1:c1:0:10-20"
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"candidates": [{"candidate_id": cid}]}))
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {cid: "probably"}}))
    syn = tmp_path / "syn.json"
    syn.write_text(json.dumps({"status": {cid: STATUS_FAILED}}))
    post = tmp_path / "p.json"
    post.write_text(json.dumps({"posteriors": {cid: 0.5}}))
    with pytest.raises(rec.RecommendError, match="expected one of"):
        rec.build_inputs(
            covariation_status_path=cov,
            synteny_status_path=syn,
            stage2_posteriors_path=post,
            fp_manifest_path=manifest,
        )


def test_the_writer_and_the_reader_apply_the_same_functions():
    """Two copies of the rule is how the writer came to have none of it."""
    import inspect

    writer = inspect.getsource(rec.build_inputs)
    reader = inspect.getsource(rec.load_spare_rule_inputs)
    for fn in ("checked_status", "checked_posterior"):
        assert fn in writer, f"build_inputs must go through {fn}"
        assert fn in reader, f"load_spare_rule_inputs must go through {fn}"


def test_an_id_with_a_fifth_field_is_refused_not_silently_merged():
    """A fifth field would drop the middle ones and merge two loci into one."""
    with pytest.raises(rec.RecommendError, match="not 4"):
        rec.locus_of("GCA_1.1:c1:0:extra:10-20")


def test_the_undecidable_axes_are_parameter_keys_only():
    """`^` would admit a vocabulary name that is not a parameter and KeyError in the guard."""
    rec_params = set(CHOSEN.as_dict())
    assert set(rec.TIE_BREAK_VOCABULARY) <= rec_params, "today they are all parameters"
    a = point(params=apm.ParamTuple("a", 1, 2, 2, 2, 50, 2, False), fp_failed=10)
    b = point(params=apm.ParamTuple("b", 1, 2, 3, 2, 20, 2, False), fp_failed=10)
    with pytest.raises(rec.RecommendError, match="min_helix_pairs"):
        rec.apply_decision_rule([a, b])
