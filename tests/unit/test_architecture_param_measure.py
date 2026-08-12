"""Unit tests for the P3-15'-f criterion-(b) parameter measurement.

Two properties carry this file, and both are about the measurement being *about*
the producer rather than about a committed file:

* every number must come out of the shipped
  :mod:`tbox_finder.mining.architecture` localizer, so the values chosen from the
  measurement are the values the producer will apply;
* every "this parameter decides nothing" claim must have a **positive control** —
  a case where the same function says it *does* decide something. A refusal that
  refuses everything, and an inertness proof that finds everything inert, are the
  same failure ([[raises-test-needs-a-positive-control]]).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder.mining import architecture as arch
from tbox_finder.mining import architecture_param_measure as apm

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures — synthetic Stockholm alignments with hand-checked structure
# ═════════════════════════════════════════════════════════════════════════════
#: Two 2-pair helices; flanked bulges at columns 2-3 and 12-13.  The first bulge's
#: residues in the candidate row are ``UG`` — a register of the ``UGGN`` acceptor
#: motif at ``ncca_pairing_nt=2``.
SS_TWO_HELIX = "((..((....))..))"
ROW_WITH_UG = "GGUGGGAAAACCCUGCC"[:16]
#: The same structure with bulge residues that match no ``UGGN`` register.
ROW_WITHOUT_MOTIF = "GGACGGAAAACCCCACC"[:16]
#: One helix, no flanked bulge (its only unpaired run is a hairpin loop).
SS_ONE_HELIX = "((((....))))"
ROW_ONE_HELIX = "GGGGAAAACCCC"
ROW_ONE_HELIX_ALT = "GGGGUUUUCCCC"


def stockholm(ss_cons: str, rows: list[tuple[str, str]]) -> str:
    """A minimal Stockholm alignment, in the layout ``parse_stockholm`` accepts."""
    body = "".join(f"{name:20s} {seq}\n" for name, seq in rows)
    return f"# STOCKHOLM 1.0\n{body}{'#=GC SS_cons':20s} {ss_cons}\n//\n"


def write_supply(root, spec: dict[str, tuple[str, str, int]]) -> None:
    """``{slug: (ss_cons, candidate_row, depth)}`` → ``<slug>/msa.sto`` under ``root``."""
    for slug, (ss_cons, row, depth) in spec.items():
        rows = [("candidate", row)] + [(f"h{i}", row) for i in range(depth - 1)]
        d = root / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "msa.sto").write_text(stockholm(ss_cons, rows))


def manifest_for(tmp_path, candidate_ids, extra=None):
    """A minimal FP-manifest file whose candidate ids slug to the supply's dirs."""
    rows = [{"candidate_id": cid, **(extra or {})} for cid in candidate_ids]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "1.0", "candidates": rows}))
    return path


@pytest.fixture
def supply(tmp_path):
    """Four consensuses, all at the depth floor, deliberately **asymmetric**.

    One passes (b), three fail it for two different reasons (no motif in the bulge;
    no second helix and no flanked bulge at all).  The arm sizes are 1/1/2 rather
    than balanced so that a test which reads a count off the *wrong* stratum gets a
    different number instead of a coincidentally equal one
    ([[symmetric-count-fixture-blind-to-inversion]] — the first version of this
    fixture was 1/1/1 and a real inversion sabotage stayed green against it).
    """
    from tbox_finder.mining.covariation_producer import candidate_slug

    ids = [
        "ACC.1:c1:0:10-20",
        "ACC.1:c2:0:10-20",
        "ACC.1:c3:0:10-20",
        "ACC.1:c4:0:10-20",
    ]
    root = tmp_path / "msa"
    write_supply(
        root,
        {
            candidate_slug(ids[0]): (SS_TWO_HELIX, ROW_WITH_UG, 20),
            candidate_slug(ids[1]): (SS_TWO_HELIX, ROW_WITHOUT_MOTIF, 20),
            candidate_slug(ids[2]): (SS_ONE_HELIX, ROW_ONE_HELIX, 20),
            candidate_slug(ids[3]): (SS_ONE_HELIX, ROW_ONE_HELIX_ALT, 20),
        },
    )
    return root, ids


# ═════════════════════════════════════════════════════════════════════════════
# The measurement reads the SHIPPED localizer, not a copy of it
# ═════════════════════════════════════════════════════════════════════════════
def test_the_localizer_primitives_are_the_shipped_objects_not_reimplementations():
    """A second implementation of "what a helix is" is the whole failure mode.

    Identity, not equality: a local ``def find_helices`` in the measurement module
    would satisfy any behavioural check written against the same fixtures while
    silently drifting from the producer the parameters are chosen for.
    """
    assert apm.find_helices is arch.find_helices
    assert apm.find_bulges is arch.find_bulges
    assert apm.degapped_span is arch.degapped_span
    assert apm.named_elements_status is arch.named_elements_status
    assert apm.ncca_bulge_status is arch.ncca_bulge_status
    assert apm.parse_stockholm is arch.parse_stockholm
    assert apm.architecture_status is arch.architecture_status


def test_helix_marginals_agree_with_a_direct_call_to_named_elements_status(supply):
    """The grid must be ``named_elements_status``' own answer, cell for cell."""
    root, _ = supply
    items = apm.read_supply(root)
    out = apm.helix_marginals(items, min_helix_pairs_values=(1, 2), min_named_helices_values=(1, 2))
    grid = out["named_elements_present_by_min_helix_pairs_then_min_named_helices"]
    for mhp in (1, 2):
        for mnh in (1, 2):
            expected = sum(
                arch.named_elements_status(item.pairs, min_named_helices=mnh, min_helix_pairs=mhp)[
                    0
                ]
                for item in items
            )
            assert grid[str(mhp)][str(mnh)]["n_pass"] == expected


def test_named_helices_of_two_separates_the_one_helix_consensus(supply):
    """A behavioural anchor: the grid is not uniformly saturated.

    Without this, a grid that returned "everything passes" would satisfy the
    agreement test above only because both sides were equally wrong.
    """
    root, _ = supply
    items = apm.read_supply(root)
    grid = apm.helix_marginals(items, min_helix_pairs_values=(2,), min_named_helices_values=(1, 2))[
        "named_elements_present_by_min_helix_pairs_then_min_named_helices"
    ]["2"]
    assert grid["1"]["n_pass"] == 4
    assert grid["2"]["n_pass"] == 2  # the two one-helix consensuses drop out
    assert grid["2"]["n_fail"] == 2


def test_every_reported_share_reconciles_against_the_counts_beside_it(supply):
    """Published rates drifted four times in P3-15'-c-ii by hand-copying."""
    root, _ = supply
    items = apm.read_supply(root)
    grid = apm.helix_marginals(
        items,
        min_helix_pairs_values=apm.MIN_HELIX_PAIRS_SWEEP,
        min_named_helices_values=apm.MIN_NAMED_HELICES_SWEEP,
    )["named_elements_present_by_min_helix_pairs_then_min_named_helices"]
    for row in grid.values():
        for cell in row.values():
            assert cell["n_pass"] + cell["n_fail"] == len(items)
            assert cell["share_pass"] == pytest.approx(cell["n_pass"] / len(items), abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# The bulge arm
# ═════════════════════════════════════════════════════════════════════════════
def test_bulge_sizes_are_candidate_residues_not_alignment_columns(tmp_path):
    """``find_bulges`` deliberately has no size filter; the size is degapped."""
    root = tmp_path / "msa"
    # The bulge spans two columns, one of which is a gap in the candidate's own row.
    write_supply(root, {"slug": (SS_TWO_HELIX, "GG-GGGAAAACCCUGCC"[:16], 20)})
    items = apm.read_supply(root)
    sizes = apm.bulge_marginals(
        items,
        bulge_min_nt_values=(1,),
        bulge_max_nt_values=(10,),
        ncca_pairing_nt_values=(1,),
        allow_wobble_values=(False,),
    )["bulge_residue_size"]["counts"]
    assert sizes.get("1") == 1, sizes  # two columns, one gap ⇒ one residue


def test_an_illegal_bulge_floor_is_recorded_as_refused_not_skipped(supply):
    """``bulge_min_nt < ncca_pairing_nt`` is a fail-OPEN gap the localizer refuses.

    Skipping the cell silently would leave the reader unable to tell a refused
    combination from one that was never swept.
    """
    root, _ = supply
    items = apm.read_supply(root)
    grid = apm.bulge_marginals(
        items,
        bulge_min_nt_values=(1,),
        bulge_max_nt_values=(10,),
        ncca_pairing_nt_values=(3,),
        allow_wobble_values=(False,),
    )["bulge_state_grid"]
    assert "refused" in grid["ncca=3;range=1-10"]
    assert "ncca_pairing_nt" in grid["ncca=3;range=1-10"]["refused"]


def test_bulge_state_counts_are_three_valued_and_sum_to_the_supply(supply):
    """``undetectable`` must stay distinct from ``absent`` — collapsing them is
    exactly what makes ``class_ii_relax`` vacuous."""
    root, _ = supply
    items = apm.read_supply(root)
    cell = apm.bulge_marginals(
        items,
        bulge_min_nt_values=(2,),
        bulge_max_nt_values=(10,),
        ncca_pairing_nt_values=(2,),
        allow_wobble_values=(False,),
    )["bulge_state_grid"]["ncca=2;range=2-10"]["allow_wobble=false"]
    assert set(cell) == set(arch.BULGE_STATES)
    assert sum(cell.values()) == len(items)
    assert cell["detected"] == 1  # only the UG-bearing consensus
    assert cell["absent"] == 1  # bulges present, none pairs the acceptor end
    assert cell["undetectable"] == 2  # the one-helix consensuses have no flanked bulge


# ═════════════════════════════════════════════════════════════════════════════
# Inertness — each claim with a positive control
# ═════════════════════════════════════════════════════════════════════════════
def test_allow_wobble_is_structurally_inert_for_the_ncca_acceptor():
    out = apm.wobble_inertness("NCCA")
    assert out["motif"] == "UGGN"
    assert out["inert"] is True
    assert out["positions_where_wobble_can_fire"] == []


def test_wobble_inertness_positive_control_an_acceptor_where_it_does_fire():
    """Without this, a ``wobble_inertness`` hardwired to ``True`` passes.

    ``CCG`` yields motif ``CGG``; the ``C`` position's acceptor base is ``G``,
    which **is** in ``WOBBLE_PAIRS``, so the flag is live there.
    """
    out = apm.wobble_inertness("CCG")
    assert out["motif"] == "CGG"
    assert out["inert"] is False
    assert [p["motif_base"] for p in out["positions_where_wobble_can_fire"]] == ["C"]


def test_the_wobble_inertness_claim_is_confirmed_by_the_localizer_itself(supply):
    """The structural proof and the shipped detector must agree, both ways.

    On ``NCCA`` the flag changes nothing; on the ``CCG`` control it flips a real
    verdict — so the proof is tracking the detector, not a coincidence.
    """
    root, _ = supply
    items = apm.read_supply(root)
    item = items[0]
    kw = {"bulge_size_range": (2, 10), "ncca_pairing_nt": 2}
    assert (
        arch.ncca_bulge_status(item.row, item.pairs, allow_wobble=False, **kw)[0]
        == arch.ncca_bulge_status(item.row, item.pairs, allow_wobble=True, **kw)[0]
    )
    ctrl = {**kw, "acceptor_3prime": "CCG"}
    assert arch.ncca_bulge_status(item.row, item.pairs, allow_wobble=False, **ctrl)[0] == "absent"
    assert arch.ncca_bulge_status(item.row, item.pairs, allow_wobble=True, **ctrl)[0] == "detected"


def test_stem_i_threshold_is_inert_on_a_manifest_carrying_neither_field():
    rows = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    out = apm.stem_i_threshold_inertness(rows)
    assert out["inert"] is True
    assert out["n_rows_that_differ"] == 0
    assert out["n_ultrashort_relax_true_at_either_threshold"] == 0


def test_stem_i_inertness_positive_control_a_row_that_carries_the_extent():
    """A supplied extent makes the threshold decide, so ``inert`` must go False."""
    rows = [{"candidate_id": "a", "stem_i_extent_nt": 40}]
    out = apm.stem_i_threshold_inertness(rows)
    assert out["inert"] is False
    assert out["n_rows_that_differ"] == 1
    assert out["n_with_stem_i_extent_nt"] == 1


def test_stem_i_inertness_positive_control_a_translational_regulatory_mode():
    """The mode arm fires regardless of the threshold — inert, but not *unused*.

    The rows agree at both thresholds (so ``n_rows_that_differ`` is 0), yet the
    relaxation is on, which is a different fact from "the relaxation never fired".
    Reporting only the disagreement count would call this inert.
    """
    rows = [{"candidate_id": "a", "regulatory_mode": "translational"}]
    out = apm.stem_i_threshold_inertness(rows)
    assert out["n_rows_that_differ"] == 0
    assert out["n_ultrashort_relax_true_at_either_threshold"] == 1
    assert out["inert"] is False


# ═════════════════════════════════════════════════════════════════════════════
# The joint outcome
# ═════════════════════════════════════════════════════════════════════════════
def _tuple(**kw):
    base = dict(
        label="t",
        stem_i_nt_threshold=1,
        min_named_helices=2,
        min_helix_pairs=2,
        bulge_min_nt=2,
        bulge_max_nt=10,
        ncca_pairing_nt=2,
        allow_wobble=False,
    )
    base.update(kw)
    return apm.ParamTuple(**base)


def test_candidates_without_a_consensus_are_counted_unavailable_not_dropped(supply):
    root, _ = supply
    items = apm.read_supply(root)
    out = apm.evaluate_tuple(items, _tuple(), n_without_consensus=97)
    # EXACT, not `>=`: with `>=` an implementation that wrongly marked every measured
    # consensus unavailable would report 101 and still pass, and the total and the sum
    # below both stay satisfied under that misclassification.
    assert out["counts"]["unavailable"] == 97
    assert out["counts"]["passed"] + out["counts"]["failed"] == len(items)
    assert out["n_candidates"] == len(items) + 97
    assert sum(out["counts"].values()) == out["n_candidates"]


def test_the_joint_outcome_goes_through_architecture_status_depth_floor(tmp_path):
    """A shallow alignment must read ``unavailable`` (⇒ spared), never ``failed``.

    This is the A2 Pin 2 floor, and it lives in ``architecture_status``. A joint
    measurement that recomputed ``named AND detected`` would score this consensus
    ``passed`` and never notice the floor exists.
    """
    root = tmp_path / "msa"
    write_supply(root, {"slug": (SS_TWO_HELIX, ROW_WITH_UG, 3)})
    items = apm.read_supply(root)
    assert apm.evaluate_tuple(items, _tuple(), n_without_consensus=0)["counts"] == {
        "passed": 0,
        "failed": 0,
        "unavailable": 1,
    }
    # Positive control: lower the floor and the same consensus is decided.
    decided = apm.evaluate_tuple(items, _tuple(), n_without_consensus=0, min_sequences=1)
    assert decided["counts"]["unavailable"] == 0
    assert decided["counts"]["passed"] == 1


def test_the_covariation_stratification_partitions_the_supply(supply):
    """Every measured consensus lands in exactly one (a)-status arm."""
    root, ids = supply
    from tbox_finder.mining.covariation_producer import candidate_slug

    by_slug = {
        candidate_slug(ids[0]): "passed",
        candidate_slug(ids[1]): "passed",
        candidate_slug(ids[2]): "failed",
        candidate_slug(ids[3]): "failed",
    }
    out = apm.evaluate_tuple(
        apm.read_supply(root), _tuple(), n_without_consensus=0, covariation_by_slug=by_slug
    )
    arms = out["by_covariation_status"]
    assert sum(a["n"] for a in arms.values()) == 4
    cons = out["control_and_consequence"]
    assert cons["n_a_passed"] == 2
    assert cons["n_b_agrees_with_a_passed"] == 1  # only the UG-bearing one passes (b)
    assert cons["share_b_agrees_with_a_passed"] == pytest.approx(0.5, abs=1e-6)
    # Read off the (a)-FAILED arm: both one-helix consensuses. The (a)-passed arm has
    # exactly ONE (b) failure, so a swap of the two arms changes this number.
    assert cons["n_failing_both_a_and_b"] == 2
    assert arms["passed"]["failed"] == 1


def test_default_tuples_are_all_legal_for_the_localizer(supply):
    """A default whose ``bulge_min_nt < ncca_pairing_nt`` would raise mid-report."""
    root, _ = supply
    items = apm.read_supply(root)
    for params in apm.default_tuples():
        assert params.bulge_min_nt >= params.ncca_pairing_nt, params.label
        assert params.bulge_max_nt >= params.bulge_min_nt, params.label
        assert 1 <= params.min_named_helices <= arch.MAX_NAMED_HELICES, params.label
        apm.evaluate_tuple(items, params, n_without_consensus=0)


def test_default_tuples_span_the_admissible_helix_range():
    """The sweep must reach both ends, or the report cannot show the trade."""
    helices = {p.min_named_helices for p in apm.default_tuples()}
    assert min(helices) == 1
    assert max(helices) == arch.MAX_NAMED_HELICES


# ═════════════════════════════════════════════════════════════════════════════
# Reading the supply
# ═════════════════════════════════════════════════════════════════════════════
def test_an_unparseable_consensus_is_refused_not_silently_dropped(tmp_path):
    root = tmp_path / "msa"
    write_supply(root, {"good": (SS_TWO_HELIX, ROW_WITH_UG, 20)})
    (root / "bad").mkdir()
    (root / "bad" / "msa.sto").write_text("# STOCKHOLM 1.0\nonly_a_row AAAA\n//\n")
    with pytest.raises(apm.MeasureError, match="could not be parsed"):
        apm.read_supply(root)


def test_an_empty_supply_root_is_refused(tmp_path):
    root = tmp_path / "msa"
    root.mkdir()
    with pytest.raises(apm.MeasureError, match="no <slug>/msa.sto"):
        apm.read_supply(root)


def test_supply_digest_binds_to_the_bytes_not_the_count(tmp_path):
    """A different round directory of the same cardinality must not look identical."""
    a, b = tmp_path / "a", tmp_path / "b"
    write_supply(a, {"s": (SS_TWO_HELIX, ROW_WITH_UG, 20)})
    write_supply(b, {"s": (SS_TWO_HELIX, ROW_WITHOUT_MOTIF, 20)})
    assert apm.supply_digest(apm.read_supply(a)) != apm.supply_digest(apm.read_supply(b))
    assert apm.supply_digest(apm.read_supply(a)) == apm.supply_digest(apm.read_supply(a))


def test_a_supply_dir_outside_the_manifest_is_refused(tmp_path, supply):
    """The supply and the manifest must be the same corpus.

    An internally flawless report about the wrong corpus is the failure mode
    [[gate-must-bind-to-upstream-evidence]] names.
    """
    root, ids = supply
    path = manifest_for(tmp_path, ids[:2])  # the last two consensuses have no manifest row
    with pytest.raises(apm.MeasureError, match="do not correspond to any manifest"):
        apm.measure(
            msa_root=root,
            manifest_path=path,
            covariation_status_path=None,
            positive_control_path=None,
        )


def test_a_manifest_with_no_candidates_is_refused(tmp_path, supply):
    root, _ = supply
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"candidates": []}))
    with pytest.raises(apm.MeasureError, match="no candidate rows"):
        apm.measure(
            msa_root=root,
            manifest_path=path,
            covariation_status_path=None,
            positive_control_path=None,
        )


# ═════════════════════════════════════════════════════════════════════════════
# The report as a whole
# ═════════════════════════════════════════════════════════════════════════════
def test_measure_pins_nothing_and_reconciles_its_own_denominators(tmp_path, supply):
    root, ids = supply
    body = apm.measure(
        msa_root=root,
        manifest_path=manifest_for(tmp_path, ids + ["ACC.1:c9:0:10-20"]),
        covariation_status_path=None,
        positive_control_path=None,
    )
    assert body["pins_nothing"] is True
    s = body["supply"]
    assert s["n_consensuses_measured"] == 4
    assert s["n_candidates_in_manifest"] == 5
    assert s["n_candidates_without_consensus"] == 1
    for entry in body["joint"]:
        assert entry["n_candidates"] == s["n_candidates_in_manifest"]
        assert sum(entry["counts"].values()) == s["n_candidates_in_manifest"]


def test_measure_cross_checks_the_covariation_decided_set_against_the_supply(tmp_path, supply):
    """(b) reads the same MSA (a) does, so the two sets must be equal."""
    from tbox_finder.mining.covariation_producer import candidate_slug

    root, ids = supply
    cov = tmp_path / "cov.json"
    decided = {cid: ("passed" if i % 2 == 0 else "failed") for i, cid in enumerate(ids)}
    cov.write_text(json.dumps({"status": decided}))
    body = apm.measure(
        msa_root=root,
        manifest_path=manifest_for(tmp_path, ids),
        covariation_status_path=cov,
        positive_control_path=None,
    )
    assert body["supply"]["covariation"]["decided_set_equals_supply"] is True
    assert body["supply"]["covariation"]["n_decided"] == 4
    # The join really is by slug, and could not have been by raw id: no candidate id
    # equals its own directory name. (The previous form here asserted only that
    # `candidate_slug(...)` returns a non-empty string, which no implementation fails.)
    assert all(candidate_slug(cid) != cid for cid in ids)

    # Positive control: a decided candidate with no consensus breaks the equality.
    cov.write_text(json.dumps({"status": {**decided, "ACC.1:c9:0:1-2": "passed"}}))
    body = apm.measure(
        msa_root=root,
        manifest_path=manifest_for(tmp_path, ids + ["ACC.1:c9:0:1-2"]),
        covariation_status_path=cov,
        positive_control_path=None,
    )
    assert body["supply"]["covariation"]["decided_set_equals_supply"] is False


def test_distribution_reports_an_empty_input_rather_than_raising():
    out = apm._distribution([])
    assert out["n"] == 0
    assert out["min"] is None
    assert out["median"] is None
    assert out["percentiles"] == {f"p{p}": None for p in apm.PERCENTILES}


def test_both_distribution_branches_emit_the_same_keys():
    """A reader doing ``dist["median"]`` must not have to know which branch ran.

    An empty helix or bulge set is reachable (a consensus with no flanked bulge), so
    the two shapes would both appear in one report. ⚠ The **nested** percentile keys
    count too: ``dist["percentiles"]["p50"]`` is as broken by a missing inner key as
    by a missing outer one, and the first version of this test only compared the
    top level.
    """
    empty, full = apm._distribution([]), apm._distribution([1, 2, 3])
    assert set(empty) == set(full)
    assert set(empty["percentiles"]) == set(full["percentiles"])
    assert all(v is None for v in empty["percentiles"].values())


def test_median_is_the_same_json_type_whatever_the_parity_of_n():
    """``statistics.median`` returns an int for odd n and a float for even n."""
    odd = apm._distribution([1, 2, 3])["median"]
    even = apm._distribution([1, 2, 3, 4])["median"]
    assert isinstance(odd, float) and isinstance(even, float)
    assert odd == 2.0 and even == 2.5


def test_the_cli_writes_a_report_with_provenance(tmp_path, supply, monkeypatch):
    root, ids = supply
    out = tmp_path / "report.json"
    monkeypatch.chdir(tmp_path)
    rc = apm.main(
        [
            "measure",
            "--msa-root",
            str(root),
            "--manifest",
            str(manifest_for(tmp_path, ids)),
            "--positive-control",
            str(tmp_path / "absent.sto"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    body = json.loads(out.read_text())
    assert body["provenance"]["inputs"], "provenance must hash the manifest it read"
    assert body["provenance"]["extra"]["external_inputs"]["supply_digest_sha256"] == (
        body["supply"]["supply_digest_sha256"]
    )
    # An absent control is RECORDED, not omitted: a report with no key reads as one
    # that was never meant to have a control.
    assert body["positive_control"]["available"] is False
    assert body["positive_control"]["reason"]


def test_the_cli_refuses_a_bad_supply_with_exit_3_not_a_traceback(tmp_path):
    empty = tmp_path / "msa"
    empty.mkdir()
    assert (
        apm.main(
            [
                "measure",
                "--msa-root",
                str(empty),
                "--manifest",
                str(manifest_for(tmp_path, ["ACC.1:c1:0:1-2"])),
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
        == 3
    )


def _string_values(node, trail="$"):
    """Every string leaf of a JSON payload, with the path that reached it."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _string_values(v, f"{trail}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _string_values(v, f"{trail}[{i}]")
    elif isinstance(node, str):
        yield trail, node


def test_no_absolute_filesystem_path_reaches_the_committed_report(tmp_path, supply):
    """This repo is PUBLIC — a verbatim path publishes a home directory and a username.

    Scanned over the **whole payload** rather than at the four known sites, because
    the failure mode is a *new* site added later ([[fixed-one-of-two-identical-things]]).
    The supply, the covariation table and the positive control are all staged outside
    the checkout here, which is exactly the case that leaked.
    """
    root, ids = supply
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {cid: "passed" for cid in ids}}))
    ctrl = tmp_path / "ctrl.sto"
    ctrl.write_text(stockholm(SS_TWO_HELIX, [("candidate", ROW_WITH_UG), ("h1", ROW_WITH_UG)]))
    out = tmp_path / "report.json"
    manifest = manifest_for(tmp_path, ids)
    assert (
        apm.main(
            [
                "measure",
                "--msa-root",
                str(root),
                "--manifest",
                str(manifest),
                "--covariation-status",
                str(cov),
                "--positive-control",
                str(ctrl),
                "--supply-origin",
                "two.amlab:$HOME/tbox-scratch/round/msa",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    body = json.loads(out.read_text())
    offenders = [(where, val) for where, val in _string_values(body) if val.startswith("/")]
    assert offenders == [], offenders
    # ...and the content binding survives the redaction, or the fix traded one defect
    # for a report that identifies nothing.
    assert body["supply"]["covariation"]["sha256"]
    assert body["provenance"]["extra"]["external_inputs"]["supply_digest_sha256"]
    # `supply_origin` is the one path-shaped field kept VERBATIM (it is operator-authored
    # provenance naming the cluster), so the scan above cannot be what protects it.
    assert body["supply"]["supply_origin"] == "two.amlab:$HOME/tbox-scratch/round/msa"


def test_an_input_outside_the_checkout_is_recorded_by_name_and_hash(tmp_path, supply):
    """It cannot go in ``provenance.inputs``: that field records the path it hashed."""
    root, ids = supply
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {cid: "passed" for cid in ids}}))
    out = tmp_path / "report.json"
    apm.main(
        [
            "measure",
            "--msa-root",
            str(root),
            "--manifest",
            str(manifest_for(tmp_path, ids)),
            "--covariation-status",
            str(cov),
            "--positive-control",
            str(tmp_path / "absent.sto"),
            "--out",
            str(out),
        ]
    )
    ext = json.loads(out.read_text())["provenance"]["extra"]["external_inputs"]
    assert ext["covariation_status"]["name"] == "cov.json"
    assert ext["covariation_status"]["sha256"] == apm._sha256_of(cov)


def test_is_inside_repo_separates_a_basename_collision_from_a_real_repo_path(tmp_path):
    """A basename that happens to exist in the checkout must not read as repo-relative.

    Without the separate predicate, an external ``/somewhere/else/PRD.md`` would be
    published as ``PRD.md`` *and* hashed as the checkout's own file.
    """
    outside = tmp_path / "conftest.py"
    outside.write_text("# not this repo's\n")
    assert apm.is_inside_repo(outside) is False
    assert apm.portable_path(outside) == "conftest.py"
    inside = Path("src/tbox_finder/mining/architecture_param_measure.py")
    assert apm.is_inside_repo(inside) is True
    assert apm.portable_path(inside) == inside.as_posix()


def test_a_covariation_file_with_no_status_map_is_refused(tmp_path, supply):
    """Degrading to ``{}`` leaves a report that still looks complete."""
    root, ids = supply
    bad = tmp_path / "cov.json"
    bad.write_text(json.dumps({"rows": []}))
    with pytest.raises(apm.MeasureError, match="no 'status' map"):
        apm.measure(
            msa_root=root,
            manifest_path=manifest_for(tmp_path, ids),
            covariation_status_path=bad,
            positive_control_path=None,
        )


def test_an_empty_covariation_status_map_is_refused(tmp_path, supply):
    root, ids = supply
    bad = tmp_path / "cov.json"
    bad.write_text(json.dumps({"status": {}}))
    with pytest.raises(apm.MeasureError, match="is empty"):
        apm.measure(
            msa_root=root,
            manifest_path=manifest_for(tmp_path, ids),
            covariation_status_path=bad,
            positive_control_path=None,
        )


def test_a_candidate_slug_collision_in_the_manifest_is_refused(tmp_path, supply, monkeypatch):
    """The missing-consensus count assumes ``candidate_id -> slug`` is injective.

    A collision makes one directory stand for two candidates, so
    ``n_candidates_without_consensus`` counts a phantom missing consensus and every
    joint row's ``unavailable`` inflates — while the arithmetic still reconciles.
    """
    root, ids = supply
    monkeypatch.setattr(apm, "candidate_slug", lambda _cid: "one_slug_for_everything")
    with pytest.raises(apm.MeasureError, match="slug collision"):
        apm.measure(
            msa_root=root,
            manifest_path=manifest_for(tmp_path, ids),
            covariation_status_path=None,
            positive_control_path=None,
        )


def test_a_manifest_without_collisions_is_not_refused(tmp_path, supply):
    """Positive control for the collision guard: it must not refuse everything."""
    root, ids = supply
    body = apm.measure(
        msa_root=root,
        manifest_path=manifest_for(tmp_path, ids),
        covariation_status_path=None,
        positive_control_path=None,
    )
    assert body["supply"]["n_candidates_without_consensus"] == 0


def test_the_positive_control_declares_its_own_n_of_one(tmp_path):
    """It bounds the choice in one direction and supports no rate — say so."""
    sto = tmp_path / "ctrl.sto"
    sto.write_text(stockholm(SS_TWO_HELIX, [("candidate", ROW_WITH_UG), ("h1", ROW_WITH_UG)]))
    out = apm.positive_control(
        sto,
        min_named_helices_values=(1, 2),
        min_helix_pairs_values=(2,),
        ncca_pairing_nt_values=(1, 2),
        bulge_max_nt=10,
    )
    assert out["n"] == 1
    assert "supports no rate" in out["caveat"]
    assert out["helix_stack_depths"] == [2, 2]
    assert out["bulge_state"]["ncca=2;range=2-10"] == "detected"
    assert out["named_elements_present"]["min_helix_pairs=2;min_named_helices=2"] is True


def test_an_absolute_supply_origin_is_refused_rather_than_redacted(tmp_path, supply):
    """``supply_origin`` is the one path-shaped field recorded verbatim.

    That makes it the one place a local absolute path could still reach a public
    report, so it is refused — redacting it would destroy the cluster path the field
    exists to carry, and silently keeping it would publish a home directory.
    """
    root, ids = supply
    with pytest.raises(apm.MeasureError, match="local absolute path"):
        apm.measure(
            msa_root=root,
            manifest_path=manifest_for(tmp_path, ids),
            covariation_status_path=None,
            positive_control_path=None,
            supply_origin="/home/someuser/tbox-scratch/round/msa",
        )
    # Positive control: the host-qualified form the field is for is accepted.
    body = apm.measure(
        msa_root=root,
        manifest_path=manifest_for(tmp_path, ids),
        covariation_status_path=None,
        positive_control_path=None,
        supply_origin="two.amlab:$HOME/tbox-scratch/round/msa",
    )
    assert body["supply"]["supply_origin"] == "two.amlab:$HOME/tbox-scratch/round/msa"


def test_a_repeated_candidate_id_in_the_manifest_is_refused(tmp_path, supply):
    """A set comprehension deduplicates *before* anything can look.

    The collapsed count flows into ``n_candidates_without_consensus`` and then into
    ``unavailable`` in every joint row — and the arithmetic still reconciles, which
    is what makes it invisible.
    """
    root, ids = supply
    path = tmp_path / "manifest.json"
    rows = [{"candidate_id": cid} for cid in ids] + [{"candidate_id": ids[0]}]
    path.write_text(json.dumps({"candidates": rows}))
    with pytest.raises(apm.MeasureError, match="duplicate candidate id"):
        apm.measure(
            msa_root=root,
            manifest_path=path,
            covariation_status_path=None,
            positive_control_path=None,
        )


def test_a_manifest_row_carrying_no_identifier_is_refused(tmp_path, supply):
    """Such a row stringifies to ``"None"`` and every one of them collapses to one."""
    root, ids = supply
    path = tmp_path / "manifest.json"
    rows = [{"candidate_id": cid} for cid in ids] + [{"accession": "X"}, {"accession": "Y"}]
    path.write_text(json.dumps({"candidates": rows}))
    with pytest.raises(apm.MeasureError, match="neither 'candidate_id' nor 'id'"):
        apm.measure(
            msa_root=root,
            manifest_path=path,
            covariation_status_path=None,
            positive_control_path=None,
        )


def test_an_id_only_manifest_row_is_accepted(tmp_path, supply):
    """Positive control: the ``id`` fallback must still work, or the guard over-refuses."""
    root, ids = supply
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"candidates": [{"id": cid} for cid in ids]}))
    body = apm.measure(
        msa_root=root,
        manifest_path=path,
        covariation_status_path=None,
        positive_control_path=None,
    )
    assert body["supply"]["n_candidates_in_manifest"] == 4


def test_an_unbalanced_consensus_joins_the_refusal_instead_of_escaping_it(tmp_path):
    """``pair_table`` raises on unbalanced brackets — the same "unusable file" class.

    Outside the ``try`` it escaped ``read_supply``'s own refusal, so the operator got
    a bare traceback instead of the message that names every offending file.
    """
    root = tmp_path / "msa"
    write_supply(root, {"good": (SS_TWO_HELIX, ROW_WITH_UG, 20)})
    (root / "bad").mkdir()
    (root / "bad" / "msa.sto").write_text(
        stockholm("((((....)))", [("candidate", ROW_ONE_HELIX[:11]), ("h1", ROW_ONE_HELIX[:11])])
    )
    with pytest.raises(apm.MeasureError, match="could not be parsed"):
        apm.read_supply(root)


def test_the_cli_refuses_a_missing_manifest_with_exit_3_not_a_traceback(tmp_path, supply):
    """``--manifest`` is read directly, so an OSError otherwise exits 1 with a trace."""
    root, _ = supply
    assert (
        apm.main(
            [
                "measure",
                "--msa-root",
                str(root),
                "--manifest",
                str(tmp_path / "does-not-exist.json"),
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
        == 3
    )


def test_the_cli_refuses_a_malformed_manifest_with_exit_3(tmp_path, supply):
    root, _ = supply
    bad = tmp_path / "manifest.json"
    bad.write_text("{not json")
    assert (
        apm.main(
            [
                "measure",
                "--msa-root",
                str(root),
                "--manifest",
                str(bad),
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
        == 3
    )


def test_the_positive_control_names_the_bulge_range_each_state_was_read_at(tmp_path):
    """Its low bound tracks ``ncca_pairing_nt``, so a bare ncca key is ambiguous."""
    sto = tmp_path / "ctrl.sto"
    sto.write_text(stockholm(SS_TWO_HELIX, [("candidate", ROW_WITH_UG), ("h1", ROW_WITH_UG)]))
    out = apm.positive_control(
        sto,
        min_named_helices_values=(2,),
        min_helix_pairs_values=(2,),
        ncca_pairing_nt_values=(1, 2),
        bulge_max_nt=10,
    )
    assert set(out["bulge_state"]) == {"ncca=1;range=1-10", "ncca=2;range=2-10"}
    assert "tracks ncca_pairing_nt" in out["bulge_min_nt_used"]


def test_a_manifest_object_without_a_candidates_key_is_refused(tmp_path, supply):
    """``manifest["candidates"]`` would raise ``KeyError``, which ``main`` does not catch."""
    root, _ = supply
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"schema_version": "1.0"}))
    with pytest.raises(apm.MeasureError, match="no candidate rows"):
        apm.measure(
            msa_root=root,
            manifest_path=bad,
            covariation_status_path=None,
            positive_control_path=None,
        )
    assert (
        apm.main(
            [
                "measure",
                "--msa-root",
                str(root),
                "--manifest",
                str(bad),
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
        == 3
    )


def test_a_manifest_row_that_is_not_an_object_is_refused(tmp_path, supply):
    """``row.get(...)`` on a bare string raises ``AttributeError`` — same escape."""
    root, ids = supply
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"candidates": [{"candidate_id": ids[0]}, ids[1]]}))
    with pytest.raises(apm.MeasureError, match="not objects"):
        apm.measure(
            msa_root=root,
            manifest_path=bad,
            covariation_status_path=None,
            positive_control_path=None,
        )


def test_a_non_string_covariation_status_value_is_refused(tmp_path, supply):
    """An unhashable value makes ``Counter(values())`` raise ``TypeError``.

    ``main`` does not catch ``TypeError``, so the CLI would exit 1 with a traceback
    through an operator-supplied path.
    """
    root, ids = supply
    bad = tmp_path / "cov.json"
    bad.write_text(json.dumps({"status": {ids[0]: ["passed"], ids[1]: "failed"}}))
    with pytest.raises(apm.MeasureError, match="not strings"):
        apm.measure(
            msa_root=root,
            manifest_path=manifest_for(tmp_path, ids),
            covariation_status_path=bad,
            positive_control_path=None,
        )


def test_a_well_formed_covariation_map_is_not_refused(tmp_path, supply):
    """Positive control for the value-type guard: it must not refuse every map."""
    root, ids = supply
    good = tmp_path / "cov.json"
    good.write_text(json.dumps({"status": {cid: "passed" for cid in ids}}))
    body = apm.measure(
        msa_root=root,
        manifest_path=manifest_for(tmp_path, ids),
        covariation_status_path=good,
        positive_control_path=None,
    )
    assert body["supply"]["covariation"]["n_decided"] == 4


def test_a_candidate_slug_collision_in_the_covariation_table_is_refused(
    tmp_path, supply, monkeypatch
):
    """The sibling of the manifest check — the covariation table is a separate input.

    Two ids collapsing onto one slug is last-write-wins, so a candidate is attributed
    to the wrong criterion-(a) arm; ``by_covariation_status``,
    ``control_and_consequence`` and ``helix_arm_on_covariation_passed`` all shift, and
    every count still reconciles.
    """
    root, ids = supply
    # The colliding pair appears ONLY in the covariation table, so the manifest's own
    # (already-tested) guard cannot be what fires.
    extra = ["EXTRA.1:c1:0:1-2", "EXTRA.2:c1:0:1-2"]
    cov = tmp_path / "cov.json"
    cov.write_text(
        json.dumps(
            {"status": {**{cid: "passed" for cid in ids}, **{e: "unavailable" for e in extra}}}
        )
    )
    manifest = manifest_for(tmp_path, ids)
    real_slug = apm.candidate_slug
    monkeypatch.setattr(
        apm, "candidate_slug", lambda cid: "collide" if cid in extra else real_slug(cid)
    )
    with pytest.raises(apm.MeasureError, match="collision.*in the covariation"):
        apm.measure(
            msa_root=root,
            manifest_path=manifest,
            covariation_status_path=cov,
            positive_control_path=None,
        )


def test_a_malformed_positive_control_is_refused_not_traced_back(tmp_path, supply):
    """``main`` does not catch ``IndexError``, so an unguarded parse exits 1."""
    root, ids = supply
    bad = tmp_path / "ctrl.sto"
    bad.write_text("# STOCKHOLM 1.0\n#=GC SS_cons          ((((....)))\n//\n")
    with pytest.raises(apm.MeasureError, match="positive control .* could not be parsed"):
        apm.measure(
            msa_root=root,
            manifest_path=manifest_for(tmp_path, ids),
            covariation_status_path=None,
            positive_control_path=bad,
        )


def test_an_unwritable_report_path_refuses_with_exit_3(tmp_path, supply):
    """``--out`` is operator-supplied too; the input paths already follow this rule."""
    root, ids = supply
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory\n")
    assert (
        apm.main(
            [
                "measure",
                "--msa-root",
                str(root),
                "--manifest",
                str(manifest_for(tmp_path, ids)),
                "--positive-control",
                str(tmp_path / "absent.sto"),
                "--out",
                str(blocker / "report.json"),
            ]
        )
        == 3
    )


def test_the_inertness_narrative_follows_the_counts_beside_it(tmp_path):
    """As a fixed string it asserted "neither field" on a manifest that supplies one."""
    neither = apm.stem_i_threshold_inertness([{"candidate_id": "a"}])
    assert "neither field" in neither["why"]
    supplied = apm.stem_i_threshold_inertness(
        [{"candidate_id": "a", "stem_i_extent_nt": 40}, {"candidate_id": "b"}]
    )
    assert "neither field" not in supplied["why"]
    assert "1 row(s) with stem_i_extent_nt" in supplied["why"]


def test_the_field_census_sees_every_row_not_only_the_first():
    """Heterogeneous rows would make the published field list under-report."""
    out = apm.stem_i_threshold_inertness(
        [{"candidate_id": "a"}, {"candidate_id": "b", "pool": "x", "score": 1}]
    )
    assert out["manifest_row_fields"] == ["candidate_id", "pool", "score"]


# ═════════════════════════════════════════════════════════════════════════════
# The P3-15'-g-iv supply-arm seam
# ═════════════════════════════════════════════════════════════════════════════
#: The committed FP report, whose four self-descriptions the default arm must
#: reproduce verbatim — that identity is what makes the seam byte-identical.
FP_REPORT = Path("reports/p3/architecture_parameter_measurement.json")


def test_resolve_arm_defaults_to_the_fp_arm_and_accepts_both_keys():
    assert apm.resolve_arm(None).key == apm.DEFAULT_ARM == "round0_fp"
    assert apm.resolve_arm("curated_control").key == "curated_control"


def test_resolve_arm_passes_a_supply_arm_through_unchanged():
    arm = apm.SUPPLY_ARMS["curated_control"]
    assert apm.resolve_arm(arm) is arm


def test_resolve_arm_refuses_an_unknown_arm_rather_than_defaulting():
    """A silent fallback would publish the FP arm's prose over another supply."""
    with pytest.raises(apm.MeasureError, match="unknown supply arm"):
        apm.resolve_arm("some_other_round")


def test_the_two_arms_describe_different_corpora():
    """Positive control for the refusal above: the strings must actually differ."""
    fp, control = apm.SUPPLY_ARMS["round0_fp"], apm.SUPPLY_ARMS["curated_control"]
    assert fp.step != control.step
    assert fp.disclosure != control.disclosure
    assert fp.covariation_note != control.covariation_note
    assert fp.provenance_rule != control.provenance_rule
    assert (fp.ground_truth, control.ground_truth) == ("unknown", "believed_positive")


@pytest.mark.skipif(not FP_REPORT.is_file(), reason="run from the repo root")
def test_the_default_arms_prose_is_exactly_what_the_committed_fp_report_carries():
    """The seam's whole safety claim, locked against the artifact it must not move.

    ⚠ This reads a committed file and so cannot see the code
    ([[artifact-pinning-test-cannot-see-the-code]]); it is here because the *only*
    thing that makes adding `--arm` safe is that the default reproduces these four
    strings, and that is a property of the committed report, not of the module.
    """
    fp = apm.SUPPLY_ARMS[apm.DEFAULT_ARM]
    body = json.loads(FP_REPORT.read_text())
    assert body["step"] == fp.step
    assert body["disclosure"] == fp.disclosure
    assert body["supply"]["covariation"]["control_note"] == fp.covariation_note
    assert body["provenance"]["rule"] == fp.provenance_rule


def test_covariation_stratified_note_is_the_arms_note_and_defaults_to_the_fp_one():
    assert apm.covariation_stratified_note() == apm.SUPPLY_ARMS["round0_fp"].covariation_note
    assert (
        apm.covariation_stratified_note("curated_control")
        == apm.SUPPLY_ARMS["curated_control"].covariation_note
    )


def test_only_the_self_description_changes_between_the_two_arms(supply, tmp_path):
    """The matched control is only matched if the arm changes NOTHING measurable.

    Both arms are run over the same supply and the two bodies diffed: the set of
    differing paths must be exactly the self-description, so a future edit that made
    an arm touch a count, a sweep axis or a distribution fails here rather than in a
    report nobody re-derives.
    """
    root, ids = supply
    manifest = manifest_for(tmp_path, ids)
    # A covariation table is supplied so `supply.covariation.control_note` — the
    # third arm-specific string — is actually reached; without it the diff below
    # would be silent about the one field a reader is most likely to quote.
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"status": {cid: "passed" for cid in ids}}))
    kwargs = dict(
        msa_root=root,
        manifest_path=manifest,
        covariation_status_path=cov,
        positive_control_path=None,
    )
    fp_body = apm.measure(**kwargs, arm="round0_fp")
    control_body = apm.measure(**kwargs, arm="curated_control")

    differing: list[str] = []

    def diff(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                diff(a.get(k), b.get(k), f"{path}/{k}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                diff(x, y, f"{path}[{i}]")
        elif a != b:
            differing.append(path)

    diff(fp_body, control_body)
    assert differing == [
        "/disclosure",
        "/positive_control/reason",
        "/step",
        "/supply/covariation/control_note",
    ]


def test_the_arm_changes_the_provenance_rule_the_cli_records(supply, tmp_path):
    root, ids = supply
    out = tmp_path / "control_report.json"
    rc = apm.main(
        [
            "measure",
            "--arm",
            "curated_control",
            "--msa-root",
            str(root),
            "--manifest",
            str(manifest_for(tmp_path, ids)),
            "--positive-control",
            str(tmp_path / "absent.sto"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    body = json.loads(out.read_text())
    assert body["step"] == "P3-15'-g-iv"
    assert body["provenance"]["rule"] == apm.SUPPLY_ARMS["curated_control"].provenance_rule
    assert "MATCHED positive control" in body["disclosure"]


def test_the_cli_refuses_an_arm_that_is_not_in_the_registry(supply, tmp_path):
    root, ids = supply
    with pytest.raises(SystemExit) as excinfo:
        apm.main(
            [
                "measure",
                "--arm",
                "not_an_arm",
                "--msa-root",
                str(root),
                "--manifest",
                str(manifest_for(tmp_path, ids)),
                "--out",
                str(tmp_path / "r.json"),
            ]
        )
    assert excinfo.value.code == 2


def test_candidate_state_answers_for_one_consensus_and_tracks_the_setting(supply):
    """The extracted per-candidate verdict, tested on its own rather than against
    ``evaluate_tuple`` — once the latter delegates, agreement is a tautology
    ([[promote-dont-duplicate-is-a-correctness-rule]])."""
    root, _ = supply
    items = {i.slug: i for i in apm.read_supply(root)}
    from tbox_finder.mining.covariation_producer import candidate_slug

    passing = items[candidate_slug("ACC.1:c1:0:10-20")]
    loose = apm.ParamTuple("loose", 1, 1, 2, 1, 10_000, 1, False)
    strict = apm.ParamTuple("strict", 1, 4, 5, 7, 20, 4, False)
    assert apm.candidate_state(passing, loose) == "passed"
    assert apm.candidate_state(passing, strict) == "failed"


def test_candidate_state_never_returns_unavailable(supply):
    """Its docstring's claim: a SupplyItem exists, so the missing-consensus arm is
    the caller's. A drift here would double-count `unavailable` in every row."""
    root, _ = supply
    for item in apm.read_supply(root):
        for params in apm.default_tuples():
            assert apm.candidate_state(item, params) in ("passed", "failed")
