"""Unit tests for the P3-15'-g-iv matched-control comparison.

Three properties carry this file:

* **the two arms are bound to each other, not merely tabulated together** — every
  refusal here is a way the comparison could contrast the wrong corpora while every
  count still reconciled;
* **the record rule is asserted by IDENTITY, never by a count** — a record with one
  passing and one failing query is the only case that distinguishes ANY from ALL,
  and a symmetric fixture is blind to the two being swapped
  ([[symmetric-count-fixture-blind-to-inversion]]);
* **every guard has a positive control** — a check that refuses everything passes
  ``pytest.raises`` just as happily as a correct one
  ([[raises-test-needs-a-positive-control]]).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder.mining import architecture_param_control_compare as cmp
from tbox_finder.mining import architecture_param_measure as apm
from tbox_finder.mining.covariation_producer import candidate_slug

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════
#: Two 2-pair helices with a ``UG`` register in the first flanked bulge — passes (b)
#: at a loose setting.  Hand-checked in tests/unit/test_architecture_param_measure.py.
SS_TWO_HELIX = "((..((....))..))"
ROW_WITH_UG = "GGUGGGAAAACCCUGC"
#: One helix, no flanked bulge — fails (b) at every setting in the sweep.
SS_ONE_HELIX = "((((....))))"
ROW_ONE_HELIX = "GGGGAAAACCCC"

#: One deliberately-loose setting, so the synthetic supply's pass/fail split is the
#: one the fixture's docstring claims rather than a strictness accident.
LOOSE = apm.ParamTuple("loosest_nonvacuous", 1, 1, 2, 1, 10_000, 1, False)


def stockholm(ss_cons: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f"{name:20s} {seq}\n" for name, seq in rows)
    return f"# STOCKHOLM 1.0\n{body}{'#=GC SS_cons':20s} {ss_cons}\n//\n"


def write_consensus(root: Path, slug: str, ss_cons: str, row: str, depth: int = 20) -> None:
    rows = [("candidate", row)] + [(f"h{i}", row) for i in range(depth - 1)]
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "msa.sto").write_text(stockholm(ss_cons, rows))


#: Four records, **asymmetric by construction**: one mixed pass/fail (the ANY/ALL
#: witness), one wholly failing, one with no consensus at all, and one PARTIALLY
#: producible — the last exists because "a record with a producible query" is an
#: ANY over its queries too, and a corpus where every record is wholly producible
#: or wholly not cannot tell that ``any`` from an ``all``.
REC_A, REC_B, REC_C, REC_D = "aaa:c0", "bbb:c0", "ccc:c0", "ddd:c0"
Q_A_PASS = f"{REC_A}:0:10-20"
Q_A_FAIL = f"{REC_A}:0:30-40"
Q_B_FAIL1 = f"{REC_B}:0:10-20"
Q_B_FAIL2 = f"{REC_B}:0:30-40"
Q_C_NONE = f"{REC_C}:0:10-20"
Q_D_PROD = f"{REC_D}:0:10-20"
Q_D_NONE = f"{REC_D}:0:30-40"
ALL_QUERIES = [Q_A_PASS, Q_A_FAIL, Q_B_FAIL1, Q_B_FAIL2, Q_C_NONE, Q_D_PROD, Q_D_NONE]

#: The grouping the manifest fixture must yield.
EXPECTED_BY_RECORD = {
    REC_A: [Q_A_PASS, Q_A_FAIL],
    REC_B: [Q_B_FAIL1, Q_B_FAIL2],
    REC_C: [Q_C_NONE],
    REC_D: [Q_D_PROD, Q_D_NONE],
}

#: A hand-built record map + (b) states for the :func:`compare_tuple` unit tests,
#: independent of any file on disk.
BY_RECORD = {REC_A: [Q_A_PASS, Q_A_FAIL], REC_B: [Q_B_FAIL1, Q_B_FAIL2], REC_C: [Q_C_NONE]}
STATES = {Q_A_PASS: "passed", Q_A_FAIL: "failed", Q_B_FAIL1: "failed", Q_B_FAIL2: "failed"}


def status_rows(status_by_id: dict[str, str], n_homologs: dict[str, int] | None = None) -> dict:
    """A producer status table in the shape ``covariation_producer merge`` writes."""
    homologs = n_homologs or {}
    rows = []
    for cid, state in status_by_id.items():
        nh = homologs.get(cid, 25 if state in ("passed", "failed") else 3)
        rows.append(
            {
                "candidate_id": cid,
                "status": state,
                "n_homologs": nh,
                "msa_depth": nh + 1 if state in ("passed", "failed") else 0,
            }
        )
    return {"schema_version": "1.0", "status": dict(status_by_id), "rows": rows}


@pytest.fixture
def control_files(tmp_path):
    """Manifest + status + msa supply for the four-record control corpus."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "candidates": [
                    {
                        "candidate_id": q,
                        "accession": cmp.record_of(q),
                        "locus_start": 1,
                        "locus_end": 2,
                    }
                    for q in ALL_QUERIES
                ],
            }
        )
    )
    by_id = {
        Q_A_PASS: "passed",
        Q_A_FAIL: "failed",
        Q_B_FAIL1: "failed",
        Q_B_FAIL2: "passed",
        Q_C_NONE: "unavailable",
        Q_D_PROD: "failed",
        Q_D_NONE: "unavailable",
    }
    status = tmp_path / "status.json"
    status.write_text(json.dumps(status_rows(by_id)))
    root = tmp_path / "msa"
    write_consensus(root, candidate_slug(Q_A_PASS), SS_TWO_HELIX, ROW_WITH_UG)
    write_consensus(root, candidate_slug(Q_A_FAIL), SS_ONE_HELIX, ROW_ONE_HELIX)
    write_consensus(root, candidate_slug(Q_B_FAIL1), SS_ONE_HELIX, ROW_ONE_HELIX)
    write_consensus(root, candidate_slug(Q_B_FAIL2), SS_ONE_HELIX, ROW_ONE_HELIX)
    write_consensus(root, candidate_slug(Q_D_PROD), SS_ONE_HELIX, ROW_ONE_HELIX)
    return manifest, status, root


def a_split(counts) -> dict:
    """Split a row's decided candidates across the two (a) strata, ASYMMETRICALLY.

    All the `failed` go to the (a)-`failed` arm where they fit, so the two strata
    never carry the same share and a test reading the wrong one gets a different
    number.
    """
    failed, passed = int(counts["failed"]), int(counts["passed"])
    a_failed_n = max(1, (failed + passed) // 3)
    in_a_failed = min(failed, a_failed_n)
    return {
        "failed": {
            "failed": in_a_failed,
            "passed": a_failed_n - in_a_failed,
            "unavailable": 0,
            "n": a_failed_n,
        },
        "passed": {
            "failed": failed - in_a_failed,
            "passed": passed - (a_failed_n - in_a_failed),
            "unavailable": 0,
            "n": failed + passed - a_failed_n,
        },
    }


def a_report(labels_params, *, n_manifest, n_measured, counts_by_label, step) -> dict:
    """A minimal measurement report in the shape ``architecture_param_measure`` writes."""
    return {
        "schema_version": "1.0",
        "step": step,
        "sweep": {"min_named_helices": [1, 2], "ncca_pairing_nt": [1]},
        "supply": {
            "n_candidates_in_manifest": n_manifest,
            "n_consensuses_measured": n_measured,
            "supply_digest_sha256": "0" * 64,
            "supply_origin": "synthetic",
            "alignment_depth": {"median": 20, "n": n_measured, "counts": {"20": n_measured}},
            "consensus_width": {"median": 16, "n": n_measured, "counts": {"16": n_measured}},
        },
        "joint": [
            {
                "label": label,
                "params": params.as_dict(),
                "counts": counts_by_label[label],
                # The (a) split, so the stratified block is exercised on a body this
                # test computes rather than only on the committed artifact — an
                # artifact test cannot see the code
                # ([[artifact-pinning-test-cannot-see-the-code]]).
                "by_covariation_status": a_split(counts_by_label[label]),
            }
            for label, params in labels_params
        ],
    }


@pytest.fixture
def reports(tmp_path, control_files):
    """A control report that DESCRIBES the fixture supply, and an FP report beside it.

    The control counts are computed from the fixture supply by the shipped
    :func:`~tbox_finder.mining.architecture_param_measure.candidate_state`, because
    :func:`compare` refuses a report whose counts do not match the supply — that is
    the artifact binding under test, and a hand-typed count would only be testing
    whether I could predict the localizer.  The FP counts are free numbers, chosen
    to differ per setting so no test can read the right value off the wrong row.
    """
    _, _, root = control_files
    items = apm.read_supply(root)
    labels_params = [(p.label, p) for p in apm.default_tuples()]

    control_counts = {}
    for label, params in labels_params:
        states = [apm.candidate_state(i, params) for i in items]
        control_counts[label] = {
            "passed": sum(1 for s in states if s == "passed"),
            "failed": sum(1 for s in states if s == "failed"),
            "unavailable": len(ALL_QUERIES) - len(items),
        }
    fp_counts = {
        label: {"passed": 30 - 5 * i, "failed": 10 + 5 * i, "unavailable": 60}
        for i, (label, _) in enumerate(labels_params)
    }

    control = tmp_path / "control_report.json"
    fp = tmp_path / "fp_report.json"
    control.write_text(
        json.dumps(
            a_report(
                labels_params,
                n_manifest=len(ALL_QUERIES),
                n_measured=5,
                counts_by_label=control_counts,
                step=apm.SUPPLY_ARMS["curated_control"].step,
            )
        )
    )
    fp.write_text(
        json.dumps(
            a_report(
                labels_params,
                n_manifest=100,
                n_measured=40,
                counts_by_label=fp_counts,
                step=apm.SUPPLY_ARMS["round0_fp"].step,
            )
        )
    )
    return control, fp


def run_compare(control_files, reports, **overrides):
    manifest, status, root = control_files
    control_report, fp_report = reports
    kwargs = dict(
        control_msa_root=root,
        control_manifest_path=manifest,
        control_status_path=status,
        control_report_path=control_report,
        fp_report_path=fp_report,
        detect_report_path=None,
    )
    kwargs.update(overrides)
    return cmp.compare(**kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# record_of — the grouping key, and why it is only ever a cross-check
# ═════════════════════════════════════════════════════════════════════════════
def test_record_of_takes_the_first_two_colon_fields():
    assert cmp.record_of("aaa:c0:781:1136-1267") == "aaa:c0"


def test_record_of_refuses_an_id_that_encodes_no_record():
    with pytest.raises(cmp.CompareError, match="not '<record>"):
        cmp.record_of("aaa")


def test_load_control_records_groups_queries_under_their_accession(control_files):
    manifest, _, _ = control_files
    by_record = cmp.load_control_records(manifest)
    assert by_record == EXPECTED_BY_RECORD


def test_load_control_records_refuses_a_duplicate_candidate_id(tmp_path):
    """Two queries collapsing onto one would shrink every denominator silently."""
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {"candidate_id": Q_A_PASS, "accession": REC_A},
                    {"candidate_id": Q_A_PASS, "accession": REC_A},
                ]
            }
        )
    )
    with pytest.raises(cmp.CompareError, match="duplicate candidate_id"):
        cmp.load_control_records(path)


def test_load_control_records_refuses_an_accession_that_contradicts_its_own_id(tmp_path):
    """The row's two statements about which record it is must agree."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"candidates": [{"candidate_id": Q_A_PASS, "accession": "zzz:c0"}]}))
    with pytest.raises(cmp.CompareError, match="but its row says"):
        cmp.load_control_records(path)


def test_load_control_records_refuses_a_row_missing_its_accession(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"candidates": [{"candidate_id": Q_A_PASS}]}))
    with pytest.raises(cmp.CompareError, match="missing 'candidate_id' or 'accession'"):
        cmp.load_control_records(path)


def test_load_control_records_refuses_an_empty_manifest(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"candidates": []}))
    with pytest.raises(cmp.CompareError, match="no candidates"):
        cmp.load_control_records(path)


# ═════════════════════════════════════════════════════════════════════════════
# load_status
# ═════════════════════════════════════════════════════════════════════════════
def test_load_status_returns_the_map_and_the_rows(control_files):
    _, status, _ = control_files
    mapping, rows = cmp.load_status(status)
    assert mapping[Q_A_PASS] == "passed"
    assert len(rows) == len(ALL_QUERIES)


def test_load_status_refuses_an_empty_status_map(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"status": {}, "rows": [{"candidate_id": "x", "status": "passed"}]}))
    with pytest.raises(cmp.CompareError, match="no 'status' map"):
        cmp.load_status(path)


def test_load_status_refuses_a_map_that_disagrees_with_its_own_rows(tmp_path):
    """A table whose two halves disagree cannot say what the round produced."""
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "status": {"x": "passed"},
                "rows": [
                    {"candidate_id": "x", "status": "failed", "n_homologs": 25, "msa_depth": 26}
                ],
            }
        )
    )
    with pytest.raises(cmp.CompareError, match="disagrees with its own 'rows'"):
        cmp.load_status(path)


def test_load_status_refuses_a_table_with_no_rows(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"status": {"x": "passed"}, "rows": []}))
    with pytest.raises(cmp.CompareError, match="no 'rows'"):
        cmp.load_status(path)


# ═════════════════════════════════════════════════════════════════════════════
# assert_grids_identical — the claim "the grid is identical" made checkable
# ═════════════════════════════════════════════════════════════════════════════
def test_identical_reports_are_accepted(reports):
    """The positive control: a guard that refused everything would also pass above."""
    control, fp = reports
    cmp.assert_grids_identical(json.loads(control.read_text()), json.loads(fp.read_text()))


def test_grids_differing_in_a_sweep_axis_are_refused(reports):
    control, fp = reports
    c, f = json.loads(control.read_text()), json.loads(fp.read_text())
    f["sweep"]["min_named_helices"] = [1, 2, 3]
    with pytest.raises(cmp.CompareError, match="different grids"):
        cmp.assert_grids_identical(c, f)


def test_a_relabelled_joint_row_is_refused(reports):
    control, fp = reports
    c, f = json.loads(control.read_text()), json.loads(fp.read_text())
    f["joint"][0]["label"] = "something_else"
    with pytest.raises(cmp.CompareError, match="in the control report and"):
        cmp.assert_grids_identical(c, f)


def test_the_same_label_carrying_different_parameters_is_refused(reports):
    """The sharpest failure: the table would contrast two different settings."""
    control, fp = reports
    c, f = json.loads(control.read_text()), json.loads(fp.read_text())
    f["joint"][0]["params"]["min_named_helices"] = 99
    with pytest.raises(cmp.CompareError, match="different parameters"):
        cmp.assert_grids_identical(c, f)


def test_a_report_without_a_sweep_block_is_refused(reports):
    control, fp = reports
    c, f = json.loads(control.read_text()), json.loads(fp.read_text())
    c.pop("sweep")
    with pytest.raises(cmp.CompareError, match="no 'sweep' block"):
        cmp.assert_grids_identical(c, f)


def test_a_report_with_no_joint_rows_is_refused(reports):
    control, fp = reports
    c, f = json.loads(control.read_text()), json.loads(fp.read_text())
    f["joint"] = []
    with pytest.raises(cmp.CompareError, match="no 'joint' rows"):
        cmp.assert_grids_identical(c, f)


def test_a_different_number_of_joint_rows_is_refused(reports):
    control, fp = reports
    c, f = json.loads(control.read_text()), json.loads(fp.read_text())
    f["joint"] = f["joint"] + [dict(f["joint"][0])]
    with pytest.raises(cmp.CompareError, match="not the same named settings"):
        cmp.assert_grids_identical(c, f)


# ═════════════════════════════════════════════════════════════════════════════
# The record rule — ANY vs ALL, asserted by identity
# ═════════════════════════════════════════════════════════════════════════════
def test_a_record_with_one_passing_query_is_spared_and_one_with_none_is_mined():
    """The only case that separates ANY from ALL, and it is asserted by NAME.

    ``REC_A`` has one passing and one failing query, ``REC_B`` two failing, ``REC_C``
    none producible.  Under ANY: 2 producible records, 1 spared, 1 mined.  A count
    alone would be satisfied by the inverse rule on a symmetric fixture, so the
    ``spared_all`` witness is checked too — it must be 0, because no record has all
    of its queries passing.
    """
    row = cmp.compare_tuple(
        params=LOOSE,
        label=LOOSE.label,
        states=STATES,
        by_record=BY_RECORD,
        fp_row={"counts": {"passed": 30, "failed": 10, "unavailable": 60}},
    )
    rec = row["control_record_level"]
    assert rec["n_records_producible"] == 2
    assert rec["n_spared_any"] == 1
    assert rec["n_spared_all"] == 0
    assert rec["n_mined"] == 1
    assert rec["share_mined"] == pytest.approx(0.5)


def test_a_record_whose_every_query_passes_counts_under_both_rules():
    """The positive control for the ALL arm: without it, ``n_spared_all`` could be
    hardwired to 0 and every test above would still pass."""
    states = dict(STATES, **{Q_A_FAIL: "passed"})
    row = cmp.compare_tuple(
        params=LOOSE,
        label=LOOSE.label,
        states=states,
        by_record=BY_RECORD,
        fp_row={"counts": {"passed": 30, "failed": 10, "unavailable": 60}},
    )
    assert row["control_record_level"]["n_spared_any"] == 1
    assert row["control_record_level"]["n_spared_all"] == 1


def test_a_record_with_no_producible_query_leaves_the_denominator():
    """ADR-0005 D14: no consensus ⇒ ``unavailable`` ⇒ spared, and not a (b) verdict."""
    row = cmp.compare_tuple(
        params=LOOSE,
        label=LOOSE.label,
        states=STATES,
        by_record=BY_RECORD,
        fp_row={"counts": {"passed": 30, "failed": 10, "unavailable": 60}},
    )
    assert REC_C in BY_RECORD
    assert row["control_record_level"]["n_records_producible"] == len(BY_RECORD) - 1


def test_the_query_level_block_reports_a_share_but_refuses_an_interval():
    """Pseudo-replication: the query-level CI must be absent, not merely wide."""
    row = cmp.compare_tuple(
        params=LOOSE,
        label=LOOSE.label,
        states=STATES,
        by_record=BY_RECORD,
        fp_row={"counts": {"passed": 30, "failed": 10, "unavailable": 60}},
    )
    q = row["control_query_level"]
    assert q["n_decided"] == 4
    assert q["failed"] == 3
    assert q["share_failed"] == pytest.approx(0.75)
    assert q["ci95"] is None
    assert "pseudo-replicated" in q["why_no_ci"]


# ═════════════════════════════════════════════════════════════════════════════
# The two-arm contrast
# ═════════════════════════════════════════════════════════════════════════════
def test_the_difference_is_fp_minus_control_and_keeps_its_sign():
    """A sign flip here would invert every reading of the table.

    The control fails 3 of 4 (0.75); the FP arm fails 10 of 40 (0.25). The FP arm
    fails LESS, so the difference must be negative.
    """
    row = cmp.compare_tuple(
        params=LOOSE,
        label=LOOSE.label,
        states=STATES,
        by_record=BY_RECORD,
        fp_row={"counts": {"passed": 30, "failed": 10, "unavailable": 60}},
    )
    assert row["fp_candidate_level"]["share_failed"] == pytest.approx(0.25)
    assert row["discrimination"]["fp_minus_control_query_share_failed"] == pytest.approx(-0.5)


def test_ci_containment_is_true_when_the_fp_point_falls_inside_and_false_outside():
    """Both directions, because a predicate stuck at one value passes half of them."""
    inside = cmp.compare_tuple(
        params=LOOSE,
        label=LOOSE.label,
        states=STATES,
        by_record=BY_RECORD,
        fp_row={"counts": {"passed": 20, "failed": 20, "unavailable": 0}},
    )
    assert inside["fp_candidate_level"]["share_failed"] == pytest.approx(0.5)
    assert inside["discrimination"]["control_record_ci_contains_fp_point"] is True

    outside = cmp.compare_tuple(
        params=LOOSE,
        label=LOOSE.label,
        states=STATES,
        by_record=BY_RECORD,
        fp_row={"counts": {"passed": 40, "failed": 0, "unavailable": 0}},
    )
    assert outside["fp_candidate_level"]["share_failed"] == pytest.approx(0.0)
    assert outside["discrimination"]["control_record_ci_contains_fp_point"] is False


def test_share_and_ci_is_a_wilson_interval_and_refuses_nothing_at_n_zero():
    got = cmp._share_and_ci(1, 2)
    assert got["share"] == pytest.approx(0.5)
    lo, hi = got["ci95"]
    assert lo < 0.5 < hi
    # Wilson, not Wald: at k=n the interval must stay strictly inside [0, 1] rather
    # than collapsing to a point, which is the failure Wald has here.
    saturated = cmp._share_and_ci(2, 2)
    assert saturated["share"] == pytest.approx(1.0)
    assert saturated["ci95"][0] > 0.0 and saturated["ci95"][1] <= 1.0
    empty = cmp._share_and_ci(0, 0)
    assert empty == {"n": 0, "k": 0, "share": None, "ci95": None}


# ═════════════════════════════════════════════════════════════════════════════
# The self-hit floor caveat — sized from the control's OWN rows
# ═════════════════════════════════════════════════════════════════════════════
def test_the_flip_band_is_min_sequences_minus_two_and_counts_only_unavailable_rows():
    """``msa_depth == n_homologs + 1`` ⇒ a self-hitting query would reach ``+2``.

    So the queries a self-hit would rescue are exactly the ``unavailable`` ones two
    below the floor.  The fixture puts one row in the band, one just outside it, and
    one *produced* row in the band — the last must NOT be counted, because it
    already has a consensus.
    """
    floor = cmp.MIN_REAL_HOMOLOG_N
    rows = [
        {"candidate_id": "a", "status": "unavailable", "n_homologs": floor - 2, "msa_depth": 0},
        {"candidate_id": "b", "status": "unavailable", "n_homologs": floor - 3, "msa_depth": 0},
        {"candidate_id": "c", "status": "passed", "n_homologs": floor - 2, "msa_depth": floor - 1},
        {"candidate_id": "d", "status": "failed", "n_homologs": floor + 5, "msa_depth": floor + 6},
    ]
    out = cmp.self_hit_floor_caveat(rows)
    assert out["msa_depth_equals_n_homologs_plus_one"] is True
    assert out["flip_band_n_homologs"] == floor - 2
    assert out["n_queries_that_would_flip"] == 1
    assert out["pp_of_all_queries"] == pytest.approx(25.0)
    assert out["direction"].startswith("downward")


def test_the_caveat_refuses_to_size_itself_when_the_depth_relation_does_not_hold():
    """The band is only derivable *because* depth is homologs+1; if it is not, the
    size must go to ``None`` rather than being reported off an assumption."""
    floor = cmp.MIN_REAL_HOMOLOG_N
    rows = [
        {"candidate_id": "a", "status": "passed", "n_homologs": floor, "msa_depth": floor + 7},
        {"candidate_id": "b", "status": "unavailable", "n_homologs": floor - 2, "msa_depth": 0},
    ]
    out = cmp.self_hit_floor_caveat(rows)
    assert out["msa_depth_equals_n_homologs_plus_one"] is False
    assert out["flip_band_n_homologs"] is None
    assert out["n_queries_that_would_flip"] is None
    assert out["pp_of_all_queries"] is None


def test_the_caveat_refuses_a_supply_with_no_produced_rows():
    with pytest.raises(cmp.CompareError, match="no produced rows"):
        cmp.self_hit_floor_caveat(
            [{"candidate_id": "a", "status": "unavailable", "n_homologs": 1, "msa_depth": 0}]
        )


# ═════════════════════════════════════════════════════════════════════════════
# compare() — the bindings that stop two different corpora being contrasted
# ═════════════════════════════════════════════════════════════════════════════
def test_compare_runs_end_to_end_on_the_fixture(control_files, reports):
    body = run_compare(control_files, reports)
    assert body["step"] == "P3-15'-g-iv"
    assert body["pins_nothing"] is True
    assert body["arms"]["grid_identical"] is True
    # 4 records: A and B wholly producible, D partially, C not at all. ANY over the
    # record's queries ⇒ 3; an `all` here would say 2.
    assert body["producibility"]["control_record_level"]["n_records"] == 4
    assert body["producibility"]["control_record_level"]["n_records_with_a_producible_query"] == 3
    assert body["producibility"]["control_query_level"]["n_producible"] == 5
    assert body["producibility"]["control_query_level"]["n_queries"] == len(ALL_QUERIES)
    assert len(body["by_parameter_tuple"]) == len(apm.default_tuples())


def test_compare_refuses_a_status_table_that_is_not_the_manifests_corpus(
    control_files, reports, tmp_path
):
    manifest, _, _ = control_files
    stray = tmp_path / "stray_status.json"
    by_id = {q: "passed" for q in ALL_QUERIES[:-1]}
    by_id["zzz:c0:0:1-2"] = "passed"  # an id the manifest does not carry
    stray.write_text(json.dumps(status_rows(by_id)))
    with pytest.raises(cmp.CompareError, match="not the same corpus"):
        run_compare(control_files, reports, control_status_path=stray)


def test_compare_refuses_a_supply_dir_that_matches_no_control_candidate(control_files, reports):
    _, _, root = control_files
    write_consensus(root, candidate_slug("not:a:control:query"), SS_ONE_HELIX, ROW_ONE_HELIX)
    with pytest.raises(cmp.CompareError, match="match no control candidate"):
        run_compare(control_files, reports)


def test_compare_refuses_when_the_supply_is_not_criterion_as_decided_set(
    control_files, reports, tmp_path
):
    """(b) reads the alignment (a) read (ADR-0006 A4): same SET, not same size."""
    manifest, _, root = control_files
    shifted = tmp_path / "shifted_status.json"
    # Same 4/1 split, but a different member is the unavailable one.
    by_id = {
        Q_A_PASS: "unavailable",
        Q_A_FAIL: "passed",
        Q_B_FAIL1: "failed",
        Q_B_FAIL2: "passed",
        Q_C_NONE: "passed",
        Q_D_PROD: "failed",
        Q_D_NONE: "unavailable",
    }
    shifted.write_text(json.dumps(status_rows(by_id)))
    with pytest.raises(cmp.CompareError, match="not the set"):
        run_compare(control_files, reports, control_status_path=shifted)


def test_compare_refuses_a_control_report_that_does_not_describe_the_supply(
    control_files, reports, tmp_path
):
    """The artifact-binding check: right shape, wrong file."""
    control, fp = reports
    stale = json.loads(control.read_text())
    stale["joint"][0]["counts"] = {"passed": 4, "failed": 0, "unavailable": 1}
    path = tmp_path / "stale_control.json"
    path.write_text(json.dumps(stale))
    with pytest.raises(cmp.CompareError, match="does not describe"):
        run_compare(control_files, reports, control_report_path=path)


def test_compare_refuses_tuples_the_reports_never_measured(control_files, reports):
    other = apm.ParamTuple("some_other_label", 1, 1, 2, 1, 10_000, 1, False)
    with pytest.raises(cmp.CompareError, match="not the ones the control"):
        run_compare(control_files, reports, tuples=[other])


def test_compare_refuses_an_absolute_supply_origin(control_files, reports):
    """The one field recorded verbatim in a PUBLIC report."""
    with pytest.raises(cmp.CompareError, match="local absolute path"):
        run_compare(control_files, reports, supply_origin="/home/someone/scratch/msa")


def test_compare_accepts_a_host_qualified_supply_origin(control_files, reports):
    """Positive control for the refusal above."""
    body = run_compare(control_files, reports, supply_origin="two.amlab:$HOME/scratch/msa")
    assert body["arms"]["control"]["supply_origin"] == "two.amlab:$HOME/scratch/msa"


# ═════════════════════════════════════════════════════════════════════════════
# The CLI
# ═════════════════════════════════════════════════════════════════════════════
def cli_args(control_files, reports, out):
    manifest, status, root = control_files
    control, fp = reports
    return [
        "compare",
        "--control-msa-root",
        str(root),
        "--control-manifest",
        str(manifest),
        "--control-status",
        str(status),
        "--control-report",
        str(control),
        "--fp-report",
        str(fp),
        "--detect-report",
        str(out.parent / "does-not-exist.json"),
        "--out",
        str(out),
    ]


def test_main_writes_the_report_and_exits_zero(control_files, reports, tmp_path):
    out = tmp_path / "out" / "comparison.json"
    assert cmp.main(cli_args(control_files, reports, out)) == 0
    body = json.loads(out.read_text())
    assert body["provenance"]["rule"] == "P3-15'-g-iv matched-control comparison"
    assert body["headline"]["n_settings"] == len(apm.default_tuples())


def test_main_exits_three_on_a_refusal_rather_than_raising(control_files, reports, tmp_path):
    out = tmp_path / "comparison.json"
    args = cli_args(control_files, reports, out)
    args[args.index("--control-status") + 1] = str(tmp_path / "missing.json")
    assert cmp.main(args) == 3
    assert not out.exists()


def absolute_path_leaks(payload) -> list[str]:
    """Every string in ``payload`` that starts with ``/``, with its JSON path.

    One definition, used by the scan and by its positive control: two copies would
    let the control keep passing against a scan it no longer resembles.
    """
    leaks: list[str] = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str) and node.startswith("/"):
            leaks.append(f"{path}: {node}")

    walk(payload)
    return leaks


def test_main_publishes_no_absolute_path_anywhere_in_the_payload(control_files, reports, tmp_path):
    """The P3-15'-f lesson: a home directory and an account name leaked into a
    PUBLIC report. The whole payload is walked, not the fields I remembered."""
    out = tmp_path / "comparison.json"
    assert cmp.main(cli_args(control_files, reports, out)) == 0
    assert absolute_path_leaks(json.loads(out.read_text())) == []


def test_the_absolute_path_scan_can_actually_fail():
    """Positive control for the scan above, running the SAME function — on P3-15'-f
    the equivalent control could not fail, because its subject never started with
    '/'."""
    assert absolute_path_leaks({"a": {"b": ["/home/someone/x"]}}) == ["/a/b[0]: /home/someone/x"]


# ═════════════════════════════════════════════════════════════════════════════
# The committed artifacts — read, not assumed
# ═════════════════════════════════════════════════════════════════════════════
#: Resolved from ``__file__``, not from the cwd — see the same note in
#: tests/unit/test_architecture_param_measure.py. A ``skipif`` on a cwd-relative
#: path turns a MISSING committed artifact into a skip, and a skip is not a pass.
REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON = REPO_ROOT / "reports/p3/architecture_parameter_control_comparison.json"
CONTROL_REPORT = REPO_ROOT / "reports/p3/architecture_parameter_measurement_control.json"
FP_REPORT = REPO_ROOT / "reports/p3/architecture_parameter_measurement.json"


def test_all_three_committed_reports_are_present_at_all():
    """The precondition the artifact tests below used to SKIP on."""
    for path in (COMPARISON, CONTROL_REPORT, FP_REPORT):
        assert path.is_absolute(), path
        assert path.is_file(), path


def test_the_committed_comparison_is_internally_consistent():
    """Every headline number must be recomputable from the report's own rows.

    ⚠ This test cannot see the code ([[artifact-pinning-test-cannot-see-the-code]]);
    it is a lock on the committed artifact, and the rules themselves are tested as
    functions above.
    """
    body = json.loads(COMPARISON.read_text())
    rows = body["by_parameter_tuple"]
    h = body["headline"]
    assert h["n_settings"] == len(rows)
    assert h["n_settings_where_control_ci_contains_the_fp_point"] == sum(
        1 for r in rows if r["discrimination"]["control_record_ci_contains_fp_point"]
    )
    # `or 0`, mirroring the production expression: the difference is None whenever an
    # arm decided nothing, and `None < 0` is a TypeError, not a False.
    assert h["n_settings_where_control_fails_more_than_fp"] == sum(
        1 for r in rows if (r["discrimination"]["fp_minus_control_query_share_failed"] or 0) < 0
    )
    for r in rows:
        rec = r["control_record_level"]
        assert rec["n_mined"] + rec["n_spared_any"] == rec["n_records_producible"]
        assert rec["n_spared_all"] <= rec["n_spared_any"]
        q = r["control_query_level"]
        assert q["passed"] + q["failed"] == q["n_decided"]
        assert q["ci95"] is None


def test_the_committed_reports_really_do_share_a_grid():
    """The claim the whole comparison rests on, checked against the real files."""
    cmp.assert_grids_identical(
        json.loads(CONTROL_REPORT.read_text()), json.loads(FP_REPORT.read_text())
    )


def test_the_committed_comparison_states_its_intervals_on_the_record_level_n():
    """The rule the step was given: never a CI on the 317 queries."""
    body = json.loads(COMPARISON.read_text())
    n_records = body["producibility"]["control_record_level"]["n_records_with_a_producible_query"]
    for r in body["by_parameter_tuple"]:
        assert r["control_record_level"]["n_records_producible"] == n_records
        assert r["control_record_level"]["share_mined_ci95"] is not None
        assert r["control_query_level"]["ci95"] is None


# ═════════════════════════════════════════════════════════════════════════════
# CodeRabbit round 1 — the emitted prose must describe the corpus it measured
# ═════════════════════════════════════════════════════════════════════════════
def test_the_pseudo_replication_note_describes_the_corpus_it_was_handed(control_files, reports):
    """The first draft said "149 of the 160 control records" on a 4-record fixture.

    A report that states a corpus it did not measure is the same defect as a report
    that names the wrong arm — and neither is visible from inside, because every
    count still reconciles.  The numbers are asserted against the FIXTURE's sizes.
    """
    body = run_compare(control_files, reports)
    note = body["by_parameter_tuple"][0]["control_query_level"]["why_no_ci"]
    assert "3 of the 4 control records" in note
    assert f"({len(ALL_QUERIES)} queries in all)" in note
    assert "149" not in note and "160" not in note and "317" not in note


def test_the_unproducible_note_counts_this_corpus_not_the_job_1264_one(control_files, reports):
    """4 records, 7 queries, 5 producible ⇒ 2 unproducible queries, 1 whole record."""
    body = run_compare(control_files, reports)
    note = body["limitations"]["the_control_measures_the_instrument_not_the_biology"]
    assert f"({len(ALL_QUERIES) - 5} of {len(ALL_QUERIES)} queries, 1 of 4 records)" in note
    assert "241" not in note and "92" not in note


def test_the_fp_interval_caveat_quotes_no_corpus_size(control_files, reports):
    """The 76-assemblies figure belongs to another report; it cannot be recomputed
    here, so it is stated qualitatively rather than as a literal that will go stale."""
    body = run_compare(control_files, reports)
    caveat = body["by_parameter_tuple"][0]["fp_candidate_level"]["ci_caveat"]
    assert "941" not in caveat and "76" not in caveat
    assert "fewer source assemblies" in caveat


def test_compare_refuses_an_fp_report_whose_manifest_is_empty(control_files, reports, tmp_path):
    """Dividing by it would raise ZeroDivisionError, which `main` does not catch."""
    control, fp = reports
    broken = json.loads(fp.read_text())
    broken["supply"]["n_candidates_in_manifest"] = 0
    path = tmp_path / "empty_fp.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(cmp.CompareError, match="no candidates"):
        run_compare(control_files, reports, fp_report_path=path)


def test_load_status_refuses_a_row_that_is_not_an_object(tmp_path):
    """A bare string entry raises TypeError, which escapes the exit-3 convention."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"status": {"x": "passed"}, "rows": ["not-a-row"]}))
    with pytest.raises(cmp.CompareError, match="are not "):
        cmp.load_status(path)


def test_load_status_refuses_a_row_missing_the_depth_fields(tmp_path):
    """`self_hit_floor_caveat` reads `n_homologs`/`msa_depth` off these same rows."""
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps({"status": {"x": "passed"}, "rows": [{"candidate_id": "x", "status": "passed"}]})
    )
    with pytest.raises(cmp.CompareError, match="msa_depth"):
        cmp.load_status(path)


def test_a_malformed_status_row_exits_three_rather_than_tracebacking(
    control_files, reports, tmp_path
):
    """The convention end-to-end, through the operator-supplied path."""
    out = tmp_path / "comparison.json"
    bad = tmp_path / "bad_status.json"
    bad.write_text(json.dumps({"status": {"x": "passed"}, "rows": ["nope"]}))
    args = cli_args(control_files, reports, out)
    args[args.index("--control-status") + 1] = str(bad)
    assert cmp.main(args) == 3
    assert not out.exists()


# ═════════════════════════════════════════════════════════════════════════════
# CodeRabbit round 2 (GitHub app) — the two guards the CLI round missed
# ═════════════════════════════════════════════════════════════════════════════
def test_a_joint_entry_that_is_not_an_object_is_refused(reports):
    """`c_row.get("label")` on a bare string raises AttributeError, which escapes
    the exit-3 convention — the SIBLING of `load_status`'s row guard."""
    control, fp = reports
    c, f = json.loads(control.read_text()), json.loads(fp.read_text())
    f["joint"][0] = "not-a-row"
    with pytest.raises(cmp.CompareError, match=r"entr\(ies\) that are not"):
        cmp.assert_grids_identical(c, f)


def test_the_joint_shape_guard_names_the_arm_it_found_the_defect_in(reports):
    """Both arms are guarded, not only the one that happened to be checked first."""
    control, fp = reports
    c, f = json.loads(control.read_text()), json.loads(fp.read_text())
    c["joint"][1] = ["also", "not", "a", "row"]
    with pytest.raises(cmp.CompareError, match="the control report"):
        cmp.assert_grids_identical(c, f)


def test_load_status_refuses_a_duplicate_candidate_id_in_its_rows(tmp_path):
    """A dict comprehension keeps the LAST row, so a duplicate silently overwrites a
    query's status while the map/rows agreement check still passes."""
    path = tmp_path / "s.json"
    row = {"candidate_id": "x", "status": "passed", "n_homologs": 25, "msa_depth": 26}
    path.write_text(
        json.dumps(
            {
                "status": {"x": "passed"},
                "rows": [row, {**row, "status": "failed"}],
            }
        )
    )
    with pytest.raises(cmp.CompareError, match="duplicate candidate_id"):
        cmp.load_status(path)


def test_the_duplicate_row_guard_lets_distinct_ids_through(tmp_path):
    """Positive control: a guard that refused every rows list would pass above."""
    path = tmp_path / "s.json"
    row = {"candidate_id": "x", "status": "passed", "n_homologs": 25, "msa_depth": 26}
    path.write_text(
        json.dumps(
            {
                "status": {"x": "passed", "y": "failed"},
                "rows": [row, {**row, "candidate_id": "y", "status": "failed"}],
            }
        )
    )
    mapping, rows = cmp.load_status(path)
    assert mapping == {"x": "passed", "y": "failed"}
    assert len(rows) == 2


def test_each_arms_ground_truth_is_read_off_the_registry_not_restated(control_files, reports):
    """`SupplyArm.ground_truth` had no production reader while the comparison emitted
    its own spelling of the same fact — two statements that can disagree."""
    body = run_compare(control_files, reports)
    assert body["arms"]["control"]["ground_truth"] == (
        apm.SUPPLY_ARMS["curated_control"].ground_truth
    )
    assert body["arms"]["fp"]["ground_truth"] == apm.SUPPLY_ARMS["round0_fp"].ground_truth


def test_compare_refuses_a_report_whose_step_no_arm_declares(control_files, reports, tmp_path):
    """The binding the ground_truth lookup buys: a report this module did not write."""
    control, fp = reports
    stranger = json.loads(control.read_text())
    stranger["step"] = "some-other-tool"
    path = tmp_path / "stranger.json"
    path.write_text(json.dumps(stranger))
    with pytest.raises(apm.MeasureError, match="no supply arm declares step"):
        run_compare(control_files, reports, control_report_path=path)


def test_compare_refuses_one_arm_read_against_itself(control_files, reports, tmp_path):
    """Two reports of the same ground_truth are not a control and a comparator."""
    control, fp = reports
    twin = json.loads(fp.read_text())
    twin["step"] = apm.SUPPLY_ARMS["round0_fp"].step
    counts = json.loads(control.read_text())["joint"]
    twin["joint"] = counts  # keep the grid identical so THIS refusal is the one that fires
    path = tmp_path / "twin.json"
    path.write_text(json.dumps(twin))
    with pytest.raises(cmp.CompareError, match="one arm read against itself"):
        run_compare(control_files, reports, control_report_path=path)


# ═════════════════════════════════════════════════════════════════════════════
# CodeRabbit round 3 — the value types, and the last traceback escape
# ═════════════════════════════════════════════════════════════════════════════
def test_load_status_refuses_a_null_depth_field(tmp_path):
    """`int(None)` raises TypeError inside `self_hit_floor_caveat` — the same escape
    the key-presence guard closes, one level down."""
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "status": {"x": "passed"},
                "rows": [
                    {"candidate_id": "x", "status": "passed", "n_homologs": None, "msa_depth": 26}
                ],
            }
        )
    )
    with pytest.raises(cmp.CompareError, match="non-integer"):
        cmp.load_status(path)


def test_load_status_refuses_a_boolean_homolog_count(tmp_path):
    """`bool` is an `int` subclass, so `True` would pass a naive isinstance check."""
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "status": {"x": "passed"},
                "rows": [
                    {"candidate_id": "x", "status": "passed", "n_homologs": True, "msa_depth": 26}
                ],
            }
        )
    )
    with pytest.raises(cmp.CompareError, match="non-integer"):
        cmp.load_status(path)


def test_the_value_type_guard_lets_real_integer_rows_through(tmp_path):
    """Positive control: a guard refusing every row would satisfy both tests above."""
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "status": {"x": "passed"},
                "rows": [
                    {"candidate_id": "x", "status": "passed", "n_homologs": 25, "msa_depth": 26}
                ],
            }
        )
    )
    mapping, rows = cmp.load_status(path)
    assert mapping == {"x": "passed"} and len(rows) == 1


def test_a_type_error_from_operator_json_still_exits_three(control_files, reports, tmp_path):
    """The stated convention, through a path that really does raise TypeError.

    ⚠ The first version of this test fed a null depth field — which the value-type
    guard above now refuses with CompareError before any ``int()`` runs, so it
    exercised the guard and not the tuple, and the sabotage that removes TypeError
    from the tuple stayed GREEN against it. ``int(None)`` on the FP report's
    ``n_consensuses_measured`` is a live TypeError with no guard in front of it.
    """
    out = tmp_path / "comparison.json"
    control, fp = reports
    broken = json.loads(fp.read_text())
    broken["supply"]["n_consensuses_measured"] = None
    path = tmp_path / "null_measured.json"
    path.write_text(json.dumps(broken))
    args = cli_args(control_files, reports, out)
    args[args.index("--fp-report") + 1] = str(path)
    assert cmp.main(args) == 3
    assert not out.exists()


def test_an_fp_row_that_decided_nothing_yields_a_none_difference_not_a_crash():
    """`share_failed` is None when an arm has no decided candidates, and `None < 0`
    is a TypeError rather than a False — so the headline's comparison and the
    committed-artifact test both have to spell it `or 0`."""
    row = cmp.compare_tuple(
        params=LOOSE,
        label=LOOSE.label,
        states=STATES,
        by_record=BY_RECORD,
        fp_row={"counts": {"passed": 0, "failed": 0, "unavailable": 40}},
    )
    assert row["fp_candidate_level"]["share_failed"] is None
    assert row["discrimination"]["fp_minus_control_query_share_failed"] is None
    assert row["discrimination"]["control_record_ci_contains_fp_point"] is None
    # The headline's own expression must survive it, and count it as "not below zero".
    assert ((row["discrimination"]["fp_minus_control_query_share_failed"] or 0) < 0) is False


# ═════════════════════════════════════════════════════════════════════════════
# The (a)-stratified contrast — the composition confound, closed by measurement
# ═════════════════════════════════════════════════════════════════════════════
def a_stratified_row(c_pass, c_fail, f_pass, f_fail):
    """Two `joint` rows carrying only what `stratify_by_criterion_a` reads."""

    def arm(failed, n):
        return {"failed": failed, "passed": n - failed, "unavailable": 0, "n": n}

    return (
        {"by_covariation_status": {"passed": arm(*c_pass), "failed": arm(*c_fail)}},
        {"by_covariation_status": {"passed": arm(*f_pass), "failed": arm(*f_fail)}},
    )


def test_the_stratified_contrast_reports_each_arm_inside_each_a_stratum():
    """Asymmetric on purpose: 1/10 vs 4/10 in one stratum, 6/10 vs 3/10 in the other,
    so a test reading the wrong stratum or the wrong arm gets a different number."""
    c_row, f_row = a_stratified_row((1, 10), (6, 10), (4, 10), (3, 10))
    out = cmp.stratify_by_criterion_a(c_row, f_row)
    assert out["criterion_a_passed"]["control"]["share_failed"] == pytest.approx(0.1)
    assert out["criterion_a_passed"]["fp"]["share_failed"] == pytest.approx(0.4)
    assert out["criterion_a_passed"]["fp_minus_control"] == pytest.approx(0.3)
    assert out["criterion_a_failed"]["control"]["share_failed"] == pytest.approx(0.6)
    assert out["criterion_a_failed"]["fp"]["share_failed"] == pytest.approx(0.3)
    assert out["criterion_a_failed"]["fp_minus_control"] == pytest.approx(-0.3)


def test_the_stratified_contrast_would_expose_a_simpsons_reversal():
    """The whole point: arms that agree overall can differ inside every stratum.

    Control 5/20 = 0.25 overall (1/10 and 4/10); FP 5/20 = 0.25 overall (4/10 and
    1/10). Identical unstratified, opposite within each stratum — and the block must
    say so rather than inheriting the overall agreement.
    """
    c_row, f_row = a_stratified_row((1, 10), (4, 10), (4, 10), (1, 10))
    out = cmp.stratify_by_criterion_a(c_row, f_row)
    assert out["criterion_a_passed"]["fp_minus_control"] == pytest.approx(0.3)
    assert out["criterion_a_failed"]["fp_minus_control"] == pytest.approx(-0.3)


def test_the_stratified_contrast_is_none_when_a_report_carries_no_a_split():
    """Absent, not empty: without a covariation status the stratification does not
    exist, and an empty block would read as "measured and found nothing"."""
    c_row, f_row = a_stratified_row((1, 10), (6, 10), (4, 10), (3, 10))
    assert cmp.stratify_by_criterion_a({}, f_row) is None
    assert cmp.stratify_by_criterion_a(c_row, {}) is None


def test_an_empty_a_stratum_gives_a_none_share_not_a_zero():
    c_row, f_row = a_stratified_row((0, 0), (6, 10), (4, 10), (3, 10))
    out = cmp.stratify_by_criterion_a(c_row, f_row)
    assert out["criterion_a_passed"]["control"]["share_failed"] is None
    assert out["criterion_a_passed"]["fp_minus_control"] is None


def test_the_committed_stratified_block_reconciles_with_the_unstratified_counts():
    """Each arm's two strata must sum to that arm's decided total in the same row."""
    body = json.loads(COMPARISON.read_text())
    for r in body["by_parameter_tuple"]:
        strat = r["stratified_by_criterion_a"]
        assert strat is not None
        c_failed = sum(strat[f"criterion_a_{s}"]["control"]["failed"] for s in ("passed", "failed"))
        c_n = sum(strat[f"criterion_a_{s}"]["control"]["n"] for s in ("passed", "failed"))
        assert c_failed == r["control_query_level"]["failed"]
        assert c_n == r["control_query_level"]["n_decided"]
        f_failed = sum(strat[f"criterion_a_{s}"]["fp"]["failed"] for s in ("passed", "failed"))
        f_n = sum(strat[f"criterion_a_{s}"]["fp"]["n"] for s in ("passed", "failed"))
        assert f_failed == r["fp_candidate_level"]["failed"]
        assert f_n == r["fp_candidate_level"]["n_decided"]


def test_compare_wires_the_stratified_block_into_every_computed_row(control_files, reports):
    """The wiring, on a body this test computes — not on the committed artifact.

    The committed-artifact version of this check stayed GREEN under a sabotage that
    removed the call, because the sabotage does not regenerate the report
    ([[artifact-pinning-test-cannot-see-the-code]]).
    """
    body = run_compare(control_files, reports)
    for row in body["by_parameter_tuple"]:
        strat = row["stratified_by_criterion_a"]
        assert strat is not None, row["label"]
        c_n = sum(strat[f"criterion_a_{s}"]["control"]["n"] for s in ("passed", "failed"))
        assert c_n == row["control_query_level"]["n_decided"]
        f_failed = sum(strat[f"criterion_a_{s}"]["fp"]["failed"] for s in ("passed", "failed"))
        assert f_failed == row["fp_candidate_level"]["failed"]


# ═════════════════════════════════════════════════════════════════════════════
# CodeRabbit round 6 — the THIRD site of one escape class, closed everywhere
# ═════════════════════════════════════════════════════════════════════════════
def test_supply_block_returns_the_object_and_refuses_a_non_object():
    """Positive control first: a guard that refused everything would pass below."""
    assert cmp.supply_block({"supply": {"a": 1}}, "control") == {"a": 1}
    for bad in ([], "supply", None, 3):
        with pytest.raises(cmp.CompareError, match="carries no 'supply' object"):
            cmp.supply_block({"supply": bad}, "control")
    with pytest.raises(cmp.CompareError, match="the fp report"):
        cmp.supply_block({}, "fp")


def test_distribution_block_drops_counts_and_refuses_a_non_object():
    got = cmp.distribution_block(
        {"supply": {"alignment_depth": {"median": 30, "counts": {"20": 1}}}},
        "control",
        "alignment_depth",
    )
    assert got == {"median": 30}
    with pytest.raises(cmp.CompareError, match="'supply.alignment_depth' is not an object"):
        cmp.distribution_block({"supply": {"alignment_depth": []}}, "control", "alignment_depth")


def _supply_block_line_span(lines: list[str]) -> range:
    """1-based line numbers of ``def supply_block``'s body, from its def to the next."""
    start = next(i for i, line in enumerate(lines, 1) if line.startswith("def supply_block"))
    end = next(
        (i for i, line in enumerate(lines, 1) if i > start and line.startswith("def ")),
        len(lines) + 1,
    )
    return range(start, end)


def test_no_supply_access_bypasses_the_validated_readers():
    """The point of the round-6 finding is the CLASS, not the one site it named.

    Two earlier rounds each closed one site of this escape and left a sibling, so
    this asserts the property directly against the module's own source
    ([[fixed-one-of-two-identical-things]]).
    """
    source = Path(cmp.__file__).read_text()
    # ⚠ The WHOLE file, both quote styles, and BOTH ACCESS FORMS. The first version
    # sliced off everything before `def supply_block`, so an access above the reader
    # was never looked at; the second caught only subscripts, so a future
    # `report.get("supply")` elsewhere would reintroduce the unvalidated path. The one
    # permitted getter is the reader's own, identified by line span rather than by
    # pattern — a whitelist keyed on the text would match a copy of it anywhere.
    lines = source.splitlines()
    reader_span = _supply_block_line_span(lines)
    offenders = [
        (n, line.strip())
        for n, line in enumerate(lines, 1)
        if (
            '["supply"]' in line
            or "['supply']" in line
            or (('.get("supply")' in line or ".get('supply')" in line) and n not in reader_span)
        )
    ]
    assert offenders == [], offenders


@pytest.mark.parametrize("arm", ["control", "fp"])
def test_a_report_with_a_non_object_supply_exits_three(control_files, reports, tmp_path, arm):
    """Both arms, because the guard is per-report and one of them was unreachable
    from the other's test."""
    control, fp = reports
    src = control if arm == "control" else fp
    broken = json.loads(src.read_text())
    broken["supply"] = ["not", "an", "object"]
    path = tmp_path / f"broken_{arm}.json"
    path.write_text(json.dumps(broken))
    out = tmp_path / "comparison.json"
    args = cli_args(control_files, reports, out)
    flag = "--control-report" if arm == "control" else "--fp-report"
    args[args.index(flag) + 1] = str(path)
    assert cmp.main(args) == 3
    assert not out.exists()


@pytest.mark.parametrize("block", ["alignment_depth", "consensus_width"])
def test_a_report_with_a_non_object_distribution_exits_three(
    control_files, reports, tmp_path, block
):
    control, fp = reports
    broken = json.loads(control.read_text())
    broken["supply"][block] = None
    path = tmp_path / f"broken_{block}.json"
    path.write_text(json.dumps(broken))
    out = tmp_path / "comparison.json"
    args = cli_args(control_files, reports, out)
    args[args.index("--control-report") + 1] = str(path)
    assert cmp.main(args) == 3
    assert not out.exists()


def test_the_source_scan_would_catch_an_access_placed_above_the_reader():
    """Positive control for the scan: it must find both quote styles, anywhere.

    Written against a synthetic source string rather than the module, because the
    module is (and must stay) clean — a positive control that can only pass when the
    subject is broken is no control at all.
    """

    def offenders(source: str) -> list[tuple[int, str]]:
        return [
            (n, line.strip())
            for n, line in enumerate(source.splitlines(), 1)
            if '["supply"]' in line or "['supply']" in line
        ]

    above_the_reader = 'x = report["supply"]\ndef supply_block():\n    pass\n'
    single_quoted = "def supply_block():\n    pass\ny = report['supply']\n"
    assert offenders(above_the_reader) == [(1, 'x = report["supply"]')]
    assert offenders(single_quoted) == [(3, "y = report['supply']")]
    assert offenders('supply = report.get("supply")\n') == []


def test_an_oserror_from_the_provenance_hashing_exits_three(
    control_files, reports, tmp_path, monkeypatch
):
    """`sha256_of` READS each external input, after `compare` has already returned.

    ⚠ That read is not reachable by input alone: every path in the hashing loop is
    also read INSIDE `compare`, so an unreadable file fails earlier and the earlier
    refusal masks this one. The reachable case is a TOCTOU — the file goes away, or
    loses its permissions, between the `is_file()` check and the read — so the
    OSError is injected at exactly that boundary rather than simulated with a
    scenario the CLI cannot actually reach.
    """
    out = tmp_path / "comparison.json"

    def boom(_path):
        raise OSError("Input/output error")

    monkeypatch.setattr(cmp, "sha256_of", boom)
    assert cmp.main(cli_args(control_files, reports, out)) == 3
    assert not out.exists()


def test_the_hashing_refusal_has_a_positive_control(control_files, reports, tmp_path):
    """Without this, a `main` that returned 3 unconditionally would pass above."""
    out = tmp_path / "comparison.json"
    assert cmp.main(cli_args(control_files, reports, out)) == 0
    assert out.is_file()


# ═════════════════════════════════════════════════════════════════════════════
# CodeRabbit round 10 — the supply_origin guard, at both spellings and both sites
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "origin",
    ["/home/someone/scratch/msa", "~/work/round_p3_15g_control", "C:\\Users\\someone\\msa"],
)
def test_every_local_path_spelling_of_supply_origin_is_refused(control_files, reports, origin):
    """A leading `/` was only the most obvious one; `~` expands to the same home
    directory and the Windows form carries the account name just as plainly."""
    with pytest.raises(cmp.CompareError, match="local absolute path"):
        run_compare(control_files, reports, supply_origin=origin)


@pytest.mark.parametrize(
    "origin",
    [
        "two.amlab:$HOME/tbox-scratch/round_p3_15g_control/msa (SLURM job 1264)",
        "two.amlab:$HOME/x",
        "cluster-scratch, round_p3_15g_control",
    ],
)
def test_host_qualified_supply_origins_are_still_accepted(control_files, reports, origin):
    """Positive control: the real value carries a colon and must not be caught."""
    body = run_compare(control_files, reports, supply_origin=origin)
    assert body["arms"]["control"]["supply_origin"] == origin


def test_the_source_scan_catches_a_get_form_outside_the_reader():
    """Positive control for the second access form, on synthetic sources.

    A `report.get("supply")` placed anywhere but inside `supply_block` reintroduces
    exactly the unvalidated path the reader exists to close, and the subscript-only
    scan could not see it.
    """

    def offenders(source: str) -> list[tuple[int, str]]:
        lines = source.splitlines()
        span = _supply_block_line_span(lines)
        return [
            (n, line.strip())
            for n, line in enumerate(lines, 1)
            if (
                '["supply"]' in line
                or "['supply']" in line
                or (('.get("supply")' in line or ".get('supply')" in line) and n not in span)
            )
        ]

    reader = (
        'def supply_block(report, arm):\n    supply = report.get("supply")\n    return supply\n'
    )
    assert offenders(reader) == []
    leaked = reader + 'def other(report):\n    return report.get("supply")\n'
    assert offenders(leaked) == [(5, 'return report.get("supply")')]
    single = reader + "def other(report):\n    return report.get('supply')\n"
    assert offenders(single) == [(5, "return report.get('supply')")]


@pytest.mark.parametrize("dropped", ["failed", "passed", "n"])
def test_a_stratum_missing_any_count_is_refused_not_defaulted(dropped):
    """`n` used to be read as `.get("n", 0)` while `failed` was read strictly, so a
    stratum missing `n` published `share_failed: None` — "measured and found nothing" —
    where the same defect in `failed` raised ([[clauses-must-guard-emptiness]])."""
    c_row, f_row = a_stratified_row((1, 10), (6, 10), (4, 10), (3, 10))
    del c_row["by_covariation_status"]["passed"][dropped]
    with pytest.raises(cmp.CompareError, match=f"missing \\['{dropped}'\\]"):
        cmp.stratify_by_criterion_a(c_row, f_row)


def test_a_stratum_whose_n_contradicts_its_counts_is_refused():
    """n is the sum of the two arms; a disagreeing n makes every share below wrong."""
    c_row, f_row = a_stratified_row((1, 10), (6, 10), (4, 10), (3, 10))
    c_row["by_covariation_status"]["passed"]["n"] = 99
    with pytest.raises(cmp.CompareError, match="but failed"):
        cmp.stratify_by_criterion_a(c_row, f_row)


def test_a_consistent_stratum_is_still_accepted():
    """Positive control: a guard refusing every stratum would satisfy both tests above."""
    c_row, f_row = a_stratified_row((1, 10), (6, 10), (4, 10), (3, 10))
    out = cmp.stratify_by_criterion_a(c_row, f_row)
    assert out["criterion_a_passed"]["control"]["n"] == 10
