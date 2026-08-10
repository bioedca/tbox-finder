"""Unit tests for ``mining.curated_control_sample`` (P3-15'-g-ii).

The load-bearing properties, and why each is tested the way it is:

* **The leakage gate is a conjunction**, so every clause is broken *alone*.  A fixture in
  which every clause is TRUE cannot tell a live clause from a hardcoded ``True``
  ([[all-true-fixture-cannot-test-a-conjunction]]), and a clause compared across two id
  namespaces would be vacuously TRUE exactly when the evidence is missing
  ([[namespace-mismatch-invisible-noop]] / [[clauses-must-guard-emptiness]]).
* **The matching is asserted maximal, not merely valid.**  Any assignment covers *some*
  orders; the fixture is built so a name-ordered greedy loses one and only an augmenting
  path recovers it.
* **The window is asserted by identity, not by length.**  A window of the right size at
  the wrong offset is the defect the sizing step's bounds check was added for, so the
  round-trip compares the re-sliced locus to the record's own bytes.
* **The minted ids are parsed back by the shipped parsers** (``mine_round``'s and
  ``covariation_producer``'s), not by a re-spelled regex here — a control whose ids the
  producer cannot read is a control that silently produces nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from tbox_finder.mining import curated_control_sample as ccs
from tbox_finder.mining.covariation_producer import candidate_slug, read_candidate_manifest
from tbox_finder.mining.mine_round import parse_window_name

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════
CTX = "ACGT" * 700  # 2,800 nt of clean context, like a real ±1,074 flank fetch


def record(rid: str, *, cluster: int, order: str | None, phylum: str = "Firmicutes") -> dict:
    return {
        "record_sha256": rid,
        "cluster_id": cluster,
        "order": order,
        "phylum": phylum,
        "genus": "G",
        "tbox_type": "Transcriptional",
    }


def wide_row(
    rid: str,
    *,
    seq: str = CTX,
    offset: int = 1074,
    length: int = 240,
    strand: int = 1,
    region_start: int = 5_000,
    clipped_start: bool = False,
    clipped_end: bool = False,
    accession: str = "NZ_CP000001.1",
) -> dict:
    return {
        "_record_sha256": rid,
        "context_seq": seq,
        "locus_offset": offset,
        "locus_length": length,
        "clipped_start": clipped_start,
        "clipped_end": clipped_end,
        "accession": accession,
        "strand": strand,
        "region_start": region_start,
    }


def _joined_row(rid: str, **kwargs) -> dict:
    """A wide joined row carrying **both** column sets — the sizing module's and this one's."""
    row = wide_row(rid, **kwargs)
    row.update(
        {
            "status": "ok",
            "type": "Transcriptional",
            "TaxId": 1234,
            "cluster_id": 1,
            "resolved_phylum": "Firmicutes",
            "resolved_order": "Lactobacillales",
            "resolved_genus": "Lactobacillus",
        }
    )
    return row


@dataclass(frozen=True)
class FakeCandidate:
    candidate_id: str
    accession: str
    locus_start: int
    locus_end: int
    score: float
    pool: str = "genomic_window"


def split_index(
    *,
    drawn_digests: list[str],
    trained_digests: list[str] | None = None,
    roles: dict[str, str] | None = None,
) -> dict:
    """A minimal split index in which every leakage clause is TRUE — the base to break."""
    trained_digests = trained_digests if trained_digests is not None else ["T1", "T2"]
    role_by_digest = {d: "heldout" for d in drawn_digests}
    role_by_digest.update({d: "train" for d in trained_digests})
    if roles:
        role_by_digest.update(roles)
    record_id_by_digest = {d: f"rid::{d}" for d in role_by_digest}
    trained_ids = {f"rid::{d}" for d in trained_digests}
    return {
        "n_rows": len(role_by_digest),
        "trained_record_ids": trained_ids,
        "trained_digests": set(trained_digests),
        "trained_clusters": {9001, 9002},
        "n_trained": len(trained_digests),
        "trained_role_ids": set(trained_ids),
        "flagged_record_ids": set(trained_ids),
        "record_id_by_digest": record_id_by_digest,
        "role_by_digest": role_by_digest,
    }


# ═════════════════════════════════════════════════════════════════════════════
# cluster_order_options
# ═════════════════════════════════════════════════════════════════════════════
def test_cluster_order_options_keys_by_cluster_then_order():
    options, exclusions = ccs.cluster_order_options(
        [
            record("b", cluster=1, order="Alpha"),
            record("a", cluster=1, order="Alpha"),
            record("c", cluster=1, order="Beta"),
        ]
    )
    assert set(options) == {1}
    assert set(options[1]) == {"Alpha", "Beta"}
    assert options[1]["Alpha"]["record_sha256"] == "a"  # smallest digest wins
    assert exclusions == {}


def test_cluster_order_options_excludes_and_counts_missing_ids():
    options, exclusions = ccs.cluster_order_options(
        [
            record("a", cluster=None, order="Alpha"),
            record("", cluster=2, order="Alpha"),
            record("b", cluster=3, order=None),
        ]
    )
    assert exclusions == {"no_cluster_id": 1, "no_record_id": 1}
    assert set(options) == {3}
    assert set(options[3]) == {None}  # an order-less cluster still yields a representative


# ═════════════════════════════════════════════════════════════════════════════
# maximum_order_coverage — asserted MAXIMAL, with a greedy that would fail
# ═════════════════════════════════════════════════════════════════════════════
def test_matching_is_maximal_where_a_name_ordered_greedy_would_not_be():
    """``Alpha`` can use two clusters, ``Zeta`` only one — the one ``Alpha`` sees first.

    A greedy that walks orders in name order and takes the first free cluster gives
    cluster 1 to ``Alpha`` and leaves ``Zeta`` uncovered.  Only an augmenting path bumps
    ``Alpha`` to cluster 2 and covers both.
    """
    records = [
        record("a1", cluster=1, order="Alpha"),
        record("a2", cluster=2, order="Alpha"),
        record("z1", cluster=1, order="Zeta"),
    ]
    options, _ = ccs.cluster_order_options(records)
    coverage = ccs.maximum_order_coverage(options)
    assert coverage["n_orders_coverable"] == 2
    assert coverage["unreachable_orders"] == {}
    assert coverage["order_to_cluster"]["Zeta"] == 1
    assert coverage["order_to_cluster"]["Alpha"] == 2


def test_matching_reports_an_order_that_no_assignment_can_reach():
    """Two orders, one shared cluster: exactly one is coverable and the other is named."""
    options, _ = ccs.cluster_order_options(
        [record("a1", cluster=1, order="Alpha"), record("z1", cluster=1, order="Zeta")]
    )
    coverage = ccs.maximum_order_coverage(options)
    assert coverage["n_orders_present"] == 2
    assert coverage["n_orders_coverable"] == 1
    assert set(coverage["unreachable_orders"]) == {"Zeta"}
    assert coverage["unreachable_orders"]["Zeta"]["clusters"] == [1]


def test_matching_ignores_the_none_order_bucket():
    options, _ = ccs.cluster_order_options(
        [record("a1", cluster=1, order=None), record("a2", cluster=2, order="Alpha")]
    )
    coverage = ccs.maximum_order_coverage(options)
    assert coverage["n_orders_present"] == 1
    assert coverage["order_to_cluster"] == {"Alpha": 2}


def test_matching_is_deterministic_under_input_permutation():
    records = [
        record("a1", cluster=1, order="Alpha"),
        record("a2", cluster=2, order="Alpha"),
        record("z1", cluster=1, order="Zeta"),
        record("m1", cluster=2, order="Mu"),
        record("m2", cluster=3, order="Mu"),
    ]
    first = ccs.maximum_order_coverage(ccs.cluster_order_options(records)[0])
    second = ccs.maximum_order_coverage(ccs.cluster_order_options(records[::-1])[0])
    assert first["order_to_cluster"] == second["order_to_cluster"]


# ═════════════════════════════════════════════════════════════════════════════
# cluster_representatives
# ═════════════════════════════════════════════════════════════════════════════
def test_representatives_are_one_per_cluster_and_honour_the_matching():
    records = [
        record("a1", cluster=1, order="Alpha"),
        record("a2", cluster=2, order="Alpha"),
        record("z1", cluster=1, order="Zeta"),
    ]
    reps, exclusions, coverage = ccs.cluster_representatives(records)
    assert exclusions == {}
    assert len({r["cluster_id"] for r in reps}) == len(reps) == 2
    by_cluster = {r["cluster_id"]: r for r in reps}
    assert by_cluster[1]["record_sha256"] == "z1"  # cluster 1 is Zeta's only chance
    assert by_cluster[2]["record_sha256"] == "a2"
    assert coverage["n_orders_coverable"] == 2


def test_unmatched_cluster_prefers_an_order_bearing_member():
    """An UNMATCHED cluster must not fall back to an order-less record and lose a stratum.

    The cluster has to be genuinely unmatched for the fallback to run at all: ``Alpha`` is
    covered by cluster 1, so cluster 2 goes to the fallback branch, where the smallest
    digest (``b0``) carries no order and the correct pick is ``b1``.  A fixture whose only
    cluster is matched exercises none of this and passes with the branch deleted.
    """
    records = [
        record("a1", cluster=1, order="Alpha"),
        record("b0", cluster=2, order=None),  # smallest digest in its cluster, no order
        record("b1", cluster=2, order="Alpha"),
    ]
    reps, _, coverage = ccs.cluster_representatives(records)
    assert coverage["order_to_cluster"] == {"Alpha": 1}  # cluster 2 is unmatched
    assert sorted(r["record_sha256"] for r in reps) == ["a1", "b1"]


def test_cluster_with_no_order_bearing_member_still_yields_a_representative():
    reps, _, coverage = ccs.cluster_representatives([record("x", cluster=7, order=None)])
    assert [r["record_sha256"] for r in reps] == ["x"]
    assert coverage["n_orders_coverable"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# order_stratified_draw
# ═════════════════════════════════════════════════════════════════════════════
def _draw_frame(n_orders: int = 4, per_order: int = 5) -> list[dict]:
    out = []
    cluster = 0
    for o in range(n_orders):
        for _j in range(per_order):
            cluster += 1
            out.append(record(f"r{cluster:04d}", cluster=cluster, order=f"Order{o}"))
    return out


def test_draw_is_one_per_cluster_and_equally_allocated():
    result = ccs.order_stratified_draw(_draw_frame(), k=8)
    drawn = result["drawn"]
    assert len(drawn) == 8
    assert len({r["cluster_id"] for r in drawn}) == 8
    assert result["n_orders_reached"] == 4
    assert result["max_records_per_order"] - result["min_records_per_order"] <= 1


def test_draw_reaches_every_order_before_any_order_repeats():
    result = ccs.order_stratified_draw(_draw_frame(n_orders=6, per_order=4), k=6)
    assert result["allocation_per_order"] == {f"Order{i}": 1 for i in range(6)}


def test_draw_refuses_a_k_larger_than_the_supply():
    with pytest.raises(ccs.ControlSampleError, match="cannot draw k="):
        ccs.order_stratified_draw(_draw_frame(n_orders=2, per_order=2), k=99)


def test_draw_refuses_a_non_positive_k():
    with pytest.raises(ccs.ControlSampleError, match="k must be >= 1"):
        ccs.order_stratified_draw(_draw_frame(), k=0)


def test_draw_is_invariant_to_input_order():
    frame = _draw_frame(n_orders=5, per_order=3)
    a = ccs.order_stratified_draw(frame, k=10)["drawn"]
    b = ccs.order_stratified_draw(frame[::-1], k=10)["drawn"]
    assert [r["record_sha256"] for r in a] == [r["record_sha256"] for r in b]


def test_draw_reports_the_coverage_ceiling_beside_what_it_reached():
    frame = [
        record("a1", cluster=1, order="Alpha"),
        record("z1", cluster=1, order="Zeta"),
        record("m1", cluster=2, order="Mu"),
    ]
    result = ccs.order_stratified_draw(frame, k=2)
    assert result["n_orders_present_in_frame"] == 3
    assert result["n_orders_coverable_one_per_cluster"] == 2
    assert set(result["orders_unreachable_one_per_cluster"]) == {"Zeta"}


# ═════════════════════════════════════════════════════════════════════════════
# leakage_clauses — every clause broken ALONE
# ═════════════════════════════════════════════════════════════════════════════
DRAWN = [record("D1", cluster=1, order="Alpha"), record("D2", cluster=2, order="Beta")]


def test_leakage_all_clauses_pass_on_a_clean_draw():
    out = ccs.leakage_clauses(DRAWN, index=split_index(drawn_digests=["D1", "D2"]))
    assert out["all_pass"] is True
    assert set(out["clauses"]) == set(ccs.REQUIRED_LEAKAGE_CLAUSES)
    assert out["n_drawn"] == 2 and out["n_drawn_clusters"] == 2


def test_leakage_fails_on_a_shared_record_digest_alone():
    index = split_index(drawn_digests=["D1", "D2"], trained_digests=["D2", "T1"])
    out = ccs.leakage_clauses(DRAWN, index=index)
    assert out["clauses"]["no_shared_record_digest"] is False
    assert out["all_pass"] is False


def test_leakage_fails_on_a_shared_record_id_alone():
    """The id clause must fire even when digests differ — it is a *separate* namespace."""
    index = split_index(drawn_digests=["D1", "D2"])
    index["trained_record_ids"] = {"rid::D1"}
    out = ccs.leakage_clauses(DRAWN, index=index)
    assert out["clauses"]["no_shared_record_id"] is False
    assert out["clauses"]["no_shared_record_digest"] is True  # only ONE clause moved
    assert out["all_pass"] is False


def test_leakage_fails_on_a_shared_cluster_alone():
    index = split_index(drawn_digests=["D1", "D2"])
    index["trained_clusters"] = {2}
    out = ccs.leakage_clauses(DRAWN, index=index)
    assert out["clauses"]["no_shared_cluster_id"] is False
    assert out["clauses"]["no_shared_record_digest"] is True
    assert out["all_pass"] is False


def test_leakage_fails_when_a_drawn_record_is_not_heldout():
    index = split_index(drawn_digests=["D1", "D2"], roles={"D2": "excluded_clade_crossing"})
    out = ccs.leakage_clauses(DRAWN, index=index)
    assert out["clauses"]["every_drawn_record_is_heldout"] is False
    assert out["all_pass"] is False


def test_leakage_fails_when_a_drawn_record_is_absent_from_the_split_table():
    index = split_index(drawn_digests=["D1", "D2"])
    del index["role_by_digest"]["D2"]
    out = ccs.leakage_clauses(DRAWN, index=index)
    assert out["clauses"]["every_drawn_record_resolved"] is False
    assert out["all_pass"] is False


def test_leakage_fails_when_the_nested_train_set_is_empty():
    """The emptiness guard: with no training records every intersection is empty, and a
    gate that reads that as "clean" passes hardest exactly when its evidence is gone."""
    index = split_index(drawn_digests=["D1", "D2"], trained_digests=[])
    out = ccs.leakage_clauses(DRAWN, index=index)
    assert out["clauses"]["nested_train_nonempty"] is False
    assert out["clauses"]["nested_train_flag_agrees_with_role"] is False
    assert out["all_pass"] is False


def test_leakage_fails_on_an_empty_draw():
    out = ccs.leakage_clauses([], index=split_index(drawn_digests=[]))
    assert out["clauses"]["drawn_nonempty"] is False
    assert out["all_pass"] is False


def test_leakage_fails_when_the_flag_and_the_role_disagree_by_membership_not_count():
    """Equal-sized but different sets: a count comparison would call this agreement."""
    index = split_index(drawn_digests=["D1", "D2"])
    index["flagged_record_ids"] = {f"rid::other{i}" for i in range(len(index["trained_role_ids"]))}
    out = ccs.leakage_clauses(DRAWN, index=index)
    assert len(index["flagged_record_ids"]) == len(index["trained_role_ids"])
    assert out["clauses"]["nested_train_flag_agrees_with_role"] is False
    assert out["all_pass"] is False


def test_leakage_refuses_a_non_boolean_clause(monkeypatch):
    """A truthy non-bool satisfies ``all()`` without having been evaluated as a test."""
    monkeypatch.setattr(ccs, "REQUIRED_LEAKAGE_CLAUSES", ("drawn_nonempty", "not_a_clause"))
    with pytest.raises(ccs.ControlSampleError, match="clause set is incomplete"):
        ccs.leakage_clauses(DRAWN, index=split_index(drawn_digests=["D1", "D2"]))


# ═════════════════════════════════════════════════════════════════════════════
# emit_window
# ═════════════════════════════════════════════════════════════════════════════
def test_window_round_trips_the_locus_by_identity():
    row = wide_row("R1")
    out = ccs.emit_window(row)
    assert out["ok"] is True
    assert len(out["window_seq"]) == ccs.WINDOW_NT
    lead, length, offset = out["lead"], out["locus_length"], out["locus_offset"]
    assert out["window_seq"][lead : lead + length] == CTX[offset : offset + length]
    assert out["window_start"] + lead == offset


def test_window_is_fully_interior_even_when_the_record_is_clipped():
    """The no-pad rule: the emitted window never runs past ``context_seq``, clipped or not."""
    row = wide_row("R2", seq=CTX[:1300], offset=0, length=200, clipped_start=True)
    out = ccs.emit_window(row)
    assert out["ok"] is True
    assert out["window_start"] >= 0
    assert out["window_start"] + ccs.WINDOW_NT <= 1300
    assert out["clipped_start"] is True  # the record's real flag is carried, not consulted


def test_window_refused_when_no_unpadded_window_exists():
    row = wide_row("R3", seq=CTX[:600], offset=100, length=200, clipped_start=True)
    out = ccs.emit_window(row)
    assert out == {"ok": False, "record_sha256": "R3", "reason": "no_unpadded_window"}


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"seq": ""}, "no_context_sequence"),
        ({"offset": -5}, "locus_coordinates_unusable"),
        ({"length": 0}, "locus_coordinates_unusable"),
        ({"offset": 2790, "length": 100}, "locus_past_context_end"),
        # offset 0 keeps the locus inside the context, so the >window check is what fires
        ({"offset": 0, "length": 2000}, "locus_longer_than_window"),
    ],
)
def test_window_named_refusals(kwargs, reason):
    assert ccs.emit_window(wide_row("R", **kwargs))["reason"] == reason


def test_negative_offset_is_refused_rather_than_slicing_from_the_end():
    """Python reads a negative start as an offset from the end — the wrong locus at the
    right length.  The guard is the sign check, before any slice."""
    out = ccs.emit_window(wide_row("R", offset=-240, length=240))
    assert out["ok"] is False and out["reason"] == "locus_coordinates_unusable"


def test_window_name_parses_with_the_shipped_round0_parser():
    out = ccs.emit_window(wide_row("R1"))
    accession, contig, start = parse_window_name(out["window_name"])
    assert accession == "R1"
    assert contig == ccs.CONTROL_CONTIG_INDEX
    assert start == out["window_start"]


def test_replicon_provenance_plus_and_minus_strand_differ():
    plus = ccs.emit_window(wide_row("R1", strand=1))["replicon"]
    minus = ccs.emit_window(wide_row("R1", strand=2))["replicon"]
    assert plus["forward_start"] != minus["forward_start"]
    assert plus["reason"] is None and minus["reason"] is None
    assert plus["forward_end"] - plus["forward_start"] == ccs.WINDOW_NT - 1


def test_replicon_provenance_names_missing_geometry_without_killing_the_window():
    row = wide_row("R1")
    row["strand"] = None
    out = ccs.emit_window(row)
    assert out["ok"] is True
    assert out["replicon"]["reason"] == "replicon_geometry_missing"


# ═════════════════════════════════════════════════════════════════════════════
# emit_windows / read_windows
# ═════════════════════════════════════════════════════════════════════════════
def test_emit_windows_counts_a_record_missing_from_the_frame():
    drawn = [record("R1", cluster=1, order="Alpha"), record("GONE", cluster=2, order="Beta")]
    windows, summary = ccs.emit_windows(drawn, {"R1": wide_row("R1")})
    assert summary["n_windows"] == 1
    assert summary["refusal_reasons"] == {"record_not_in_frame": 1}
    assert summary["refused_record_ids"] == ["GONE"]
    assert windows[0]["order"] == "Alpha"


def test_read_windows_refuses_mixed_widths(tmp_path):
    path = tmp_path / "w.jsonl"
    path.write_text(
        json.dumps({"window_nt": 1024, "window_seq": "A" * 1024})
        + "\n"
        + json.dumps({"window_nt": 512, "window_seq": "A" * 512})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ccs.ControlSampleError, match="mixes window widths"):
        ccs.read_windows(path)


def test_read_windows_refuses_a_declared_width_the_sequence_does_not_have(tmp_path):
    path = tmp_path / "w.jsonl"
    path.write_text(
        json.dumps({"window_nt": 1024, "window_seq": "A" * 900}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ccs.ControlSampleError, match="declares window_nt"):
        ccs.read_windows(path)


def test_write_then_read_windows_round_trips(tmp_path):
    windows = [ccs.emit_window(wide_row("R2")), ccs.emit_window(wide_row("R1"))]
    path = ccs.write_windows(windows, tmp_path / "w.jsonl")
    back = ccs.read_windows(path)
    assert [w["record_sha256"] for w in back] == ["R1", "R2"]  # sorted on write


# ═════════════════════════════════════════════════════════════════════════════
# detect_candidates / build_queries
# ═════════════════════════════════════════════════════════════════════════════
def _window(rid: str = "R1") -> dict:
    return ccs.emit_window(wide_row(rid))


def _candidate(window: dict, rel_start: int, rel_end: int, score: float = 0.99) -> FakeCandidate:
    start = window["window_start"]
    name = window["window_name"]
    return FakeCandidate(
        candidate_id=f"{name}:{start + rel_start}-{start + rel_end}",
        accession=f"{window['record_sha256']}:c{ccs.CONTROL_CONTIG_INDEX}",
        locus_start=start + rel_start,
        locus_end=start + rel_end,
        score=score,
    )


def test_detect_candidates_uses_the_injected_scanner_and_shapes_rows():
    window = _window()
    seen: dict = {}

    def fake_scan(checkpoint, pairs, *, device=None):
        seen["checkpoint"], seen["pairs"], seen["device"] = checkpoint, pairs, device
        return [_candidate(window, 100, 200)]

    rows = ccs.detect_candidates([window], checkpoint="CKPT", device="cpu", scan=fake_scan)
    assert seen["checkpoint"] == "CKPT" and seen["device"] == "cpu"
    assert seen["pairs"] == [(window["window_name"], window["window_seq"])]
    assert rows[0]["locus_end"] - rows[0]["locus_start"] == 100


def test_build_queries_slices_the_span_from_its_own_window():
    window = _window()
    built = ccs.build_queries([window], [vars(_candidate(window, 300, 420))])
    query = built["queries"][0]
    assert query["query_seq"] == window["window_seq"][300:420]
    assert query["query_nt"] == 120
    assert query["record_sha256"] == "R1"


def test_build_queries_measures_overlap_with_the_true_locus():
    window = _window()
    lead, length = window["lead"], window["locus_length"]
    on = vars(_candidate(window, lead, lead + length))
    off = vars(_candidate(window, 0, 40))
    built = ccs.build_queries([window], [on, off])
    by_id = {q["candidate_id"]: q for q in built["queries"]}
    assert by_id[on["candidate_id"]]["overlap_iou"] == 1.0
    assert by_id[on["candidate_id"]]["overlaps_true_locus"] is True
    assert by_id[off["candidate_id"]]["overlaps_true_locus"] is False


def test_build_queries_reports_the_dropout_rather_than_dropping_it():
    fired, silent = _window("R1"), _window("R2")
    built = ccs.build_queries([fired, silent], [vars(_candidate(fired, 10, 90))])
    assert built["dropout"]["n_records_with_a_query"] == 1
    assert built["dropout"]["n_records_dropped"] == 1
    assert built["dropout"]["dropped_record_ids"] == ["R2"]
    assert built["dropout"]["dropout_share"] == 0.5
    assert built["stage1_locus_recall"]["n_records"] == 2


def test_build_queries_refuses_a_span_outside_its_window():
    window = _window()
    built = ccs.build_queries([window], [vars(_candidate(window, 900, 1200))])
    assert built["queries"] == []
    assert built["dropout"]["refusal_reasons"] == {"span_outside_window": 1}


def test_build_queries_refuses_an_ambiguous_query():
    window = _window()
    window = dict(window, window_seq="R" * ccs.WINDOW_NT)
    built = ccs.build_queries([window], [vars(_candidate(window, 10, 90))])
    assert built["queries"] == []
    assert built["dropout"]["refusal_reasons"] == {"ambiguous_alphabet": 1}


def test_build_queries_refuses_a_candidate_whose_window_is_unknown():
    window = _window("R1")
    stray = vars(_candidate(_window("R2"), 10, 90))
    built = ccs.build_queries([window], [stray])
    assert built["dropout"]["refusal_reasons"] == {"candidate_window_unknown": 1}


def test_build_queries_refuses_two_windows_sharing_a_name():
    window = _window()
    with pytest.raises(ccs.ControlSampleError, match="share a window_name"):
        ccs.build_queries([window, dict(window)], [])


def test_stage1_recall_counts_records_not_spans():
    """Three spans on one record is recall 1/2, not 3/2 — the unit is the record."""
    fired, silent = _window("R1"), _window("R2")
    lead = fired["lead"]
    spans = [
        vars(_candidate(fired, lead, lead + 40)),
        vars(_candidate(fired, lead + 50, lead + 90)),
        vars(_candidate(fired, lead + 100, lead + 140)),
    ]
    built = ccs.build_queries([fired, silent], spans)
    assert len(built["queries"]) == 3
    assert built["stage1_locus_recall"]["n_recovered"] == 1
    assert built["stage1_locus_recall"]["recall"] == 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Emitted artifacts are readable by the producer's OWN readers
# ═════════════════════════════════════════════════════════════════════════════
def test_manifest_is_read_back_by_covariation_producer(tmp_path):
    window = _window()
    built = ccs.build_queries([window], [vars(_candidate(window, 100, 220))])
    path = ccs.write_manifest(built["queries"], tmp_path / "m.json")
    specs = read_candidate_manifest(path)
    assert len(specs) == 1
    assert specs[0].candidate_id == built["queries"][0]["candidate_id"]
    assert specs[0].locus_end - specs[0].locus_start == 120


def test_minted_ids_slug_without_collision():
    window = _window()
    a = _candidate(window, 100, 220).candidate_id
    b = _candidate(window, 100, 221).candidate_id
    assert candidate_slug(a) != candidate_slug(b)


def test_query_fasta_is_wrapped_and_keyed_by_candidate_id(tmp_path):
    window = _window()
    built = ccs.build_queries([window], [vars(_candidate(window, 100, 220))])
    path = ccs.write_query_fasta(built["queries"], tmp_path / "q.fa")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f">{built['queries'][0]['candidate_id']}"
    assert "".join(lines[1:]) == built["queries"][0]["query_seq"]
    assert all(len(line) <= 80 for line in lines[1:])


# ═════════════════════════════════════════════════════════════════════════════
# Reports + CLI
# ═════════════════════════════════════════════════════════════════════════════
def test_detection_triple_follows_the_criterion_module_rather_than_a_literal(monkeypatch):
    """Moved at the source, the report must move with it.

    Asserting equality against the shipped constants would pass just as happily if the
    triple were re-typed as ``0.9 / 50 / 10`` here — a value the report *claims* comes
    from ADR-0005 A9 Pin 3 while reading nothing ([[pinned-constant-that-nothing-reads]]).
    Perturbing the source is what makes the read observable.
    """
    from tbox_finder.eval import mining_criterion

    baseline = ccs._detection_triple()
    assert baseline["threshold"] == mining_criterion.PROVISIONAL_THRESHOLD
    assert baseline["min_span"] == mining_criterion.PROVISIONAL_MIN_SPAN
    assert baseline["gap_merge"] == mining_criterion.PROVISIONAL_GAP_MERGE

    monkeypatch.setattr(mining_criterion, "PROVISIONAL_THRESHOLD", 0.4242)
    monkeypatch.setattr(mining_criterion, "PROVISIONAL_MIN_SPAN", 7)
    monkeypatch.setattr(mining_criterion, "PROVISIONAL_GAP_MERGE", 3)
    moved = ccs._detection_triple()
    assert (moved["threshold"], moved["min_span"], moved["gap_merge"]) == (0.4242, 7, 3)


def test_window_bounds_guard_fires_on_a_lead_outside_the_honest_range(monkeypatch):
    """The interiority check is a live guard, not decoration — with a positive control.

    A guard that raises on *everything* also satisfies ``pytest.raises``, so the honest
    lead is exercised first and must NOT raise ([[raises-test-needs-a-positive-control]]).

    The sibling identity check (the re-sliced locus equalling the record's own bytes) is
    redundant *by construction* while ``start = locus_offset - lead``, and is kept to fail
    loudly against a future edit that decouples the two — so it is deliberately not
    exercised by a contrived call here.
    """
    assert ccs.emit_window(wide_row("R1"))["ok"] is True
    monkeypatch.setattr(ccs, "deterministic_lead", lambda lead_range, **kw: 2000)
    with pytest.raises(ccs.ControlSampleError, match="contract is broken"):
        ccs.emit_window(wide_row("R1"))


def test_eligible_records_refuses_two_rows_sharing_a_record_id():
    """A duplicate id merges instead of colliding, and every summed count still
    reconciles ([[duplicate-key-merges-instead-of-colliding]])."""
    rows = [_joined_row("R1"), _joined_row("R1")]
    with pytest.raises(ccs.ControlSampleError, match="appears twice"):
        ccs.eligible_records(rows)


def test_eligible_records_refuses_a_missing_window_column():
    row = _joined_row("R1")
    del row["region_start"]
    with pytest.raises(ccs.ControlSampleError, match="missing required column"):
        ccs.eligible_records([row])


def test_matchedness_baseline_is_read_from_the_sizing_report(tmp_path):
    sizing = tmp_path / "sizing.json"
    sizing.write_text(json.dumps({"matchedness": {"ks_d": 0.4242}}), encoding="utf-8")
    out = ccs._matchedness([100, 110], [100, 120], sizing_report=sizing)
    assert out["baseline_raw_curated_ks_d"] == 0.4242


def test_matchedness_names_an_unreadable_baseline_instead_of_inventing_one(tmp_path):
    out = ccs._matchedness([100], [100], sizing_report=tmp_path / "absent.json")
    assert out["baseline_raw_curated_ks_d"] is None
    assert out["baseline_reason"].startswith("unavailable:")


def test_cli_refuses_a_non_positive_k(capsys):
    assert ccs.main(["draw", "--k", "0"]) == 3
    assert "refused:" in capsys.readouterr().err


def test_cli_detect_refuses_an_absent_checkpoint(tmp_path, capsys):
    windows = tmp_path / "w.jsonl"
    ccs.write_windows([_window()], windows)
    code = ccs.main(
        [
            "detect",
            "--windows",
            str(windows),
            "--checkpoint",
            str(tmp_path / "absent.pt"),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 3
    assert "is not on disk" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_score_summary_rounds_and_reports_the_calling_margin():
    from tbox_finder.eval import mining_criterion

    out = ccs._score_summary([0.92820112657787251, 0.999993055806917])
    assert out["min"] == 0.928201  # rounded past the ~1e-8 CUDA reduction-order noise
    assert out["min_margin_over_threshold"] == round(
        0.92820112657787251 - mining_criterion.PROVISIONAL_THRESHOLD, ccs.SCORE_DECIMALS
    )
    assert out["min_margin_over_threshold"] > 1e-3  # the margin dwarfs the noise


def test_score_summary_is_stable_under_float_noise_at_the_reported_precision():
    """Two runs that differ only by CUDA reduction order must produce the same summary."""
    base = [0.9282011265778725, 0.9993126233291483, 0.9999930558069173]
    jittered = [0.9282010632537339, 0.9993126232439826, 0.9999930558151939]
    assert ccs._score_summary(base) == ccs._score_summary(jittered)


def test_report_bodies_carry_the_pins_nothing_header():
    header = ccs._header()
    assert header["pins_nothing"] is True
    assert header["step"] == "P3-15'-g-ii"
