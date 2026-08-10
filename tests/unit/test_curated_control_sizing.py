"""Unit tests for the P3-15'-g curated-control sizing measurement.

Every function that produces a number in the committed report is exercised here on
inputs whose right answer is known by hand, because the report itself cannot test
the code that wrote it ([[artifact-pinning-test-cannot-see-the-code]]).  Where a
number could plausibly be hardcoded, the fixture uses distinctive non-round values
so a constant cannot pass ([[pinned-constant-that-nothing-reads]]).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from tbox_finder.mining.curated_control_sizing import (
    DEFAULT_CONTEXT,
    DEFAULT_CONTROL_SEED,
    DEFAULT_CORPUS,
    DEFAULT_FP_MANIFEST,
    DEFAULT_HOST_MANIFEST,
    DEFAULT_MEASURE_REPORT,
    DEFAULT_SPLIT_TABLE,
    ENV_LOCK,
    PRODUCIBILITY_ANCHORS,
    SizingError,
    compute_envelope,
    existing_control_placement,
    expected_distinct_strata,
    fp_assemblies,
    fp_span_lengths,
    ks_statistic,
    partition_inputs,
    percentiles,
    power_table,
    query_supply,
    records_from_joined,
    size_report,
    span_matchedness,
    stratum_census,
    substrate_overlap,
    wilson_interval,
    wilson_width,
)
from tbox_finder.mining.homolog_msa import HomologMsaError, is_clean_nucleotide

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_REPORT = REPO_ROOT / "reports/p3/curated_control_sizing.json"


def _record(**overrides):
    """A usable held-out record; overrides make exactly one thing wrong at a time."""
    base = {
        "record_sha256": "a" * 64,
        "context_status": "ok",
        "locus_length": 8,
        "locus_seq": "ACGTACGT",
        "tbox_type": "Transcriptional",
        "phylum": "Firmicutes",
        "order": "Lactobacillales",
        "genus": "Lactobacillus",
        "cluster_id": 1,
        "host_taxid": 1301,
    }
    base.update(overrides)
    return base


MEASURE_REPORT = {
    "k_sample": 50,
    "homolog_depth": {"median": 20.0},
    "status_counts": {"passed": 16, "failed": 10, "unavailable": 24},
    # Deliberately non-round and mutually distinct, so a projection that reads the
    # wrong stage (or a hardcoded constant) cannot coincide with the right answer.
    "wall_seconds": {
        "total_per_candidate": {"mean": 17.5, "median": 9.25, "max": 113.75},
        "search": {"max": 111.5},
        "score": {"max": 0.25},
        "align": {"max": 47.5},
    },
}


# ── percentiles ──────────────────────────────────────────────────────────────
def test_percentiles_are_nearest_rank_not_interpolated():
    # Interpolation would give 2.5 for p50 of [1,2,3,4] — a value no record carries.
    summary = percentiles([1, 2, 3, 4], (50,))
    assert summary["p50"] == 2


def test_percentiles_report_n_min_max_mean():
    summary = percentiles([4, 1, 3], (50,))
    assert (summary["n"], summary["min"], summary["max"]) == (3, 1, 4)
    assert summary["mean"] == pytest.approx(8 / 3, abs=0.01)


def test_percentiles_on_empty_says_n_zero_rather_than_inventing_a_value():
    assert percentiles([]) == {"n": 0}


def test_percentiles_p95_is_high_and_p05_is_low():
    values = list(range(1, 101))
    summary = percentiles(values, (5, 95))
    assert summary["p05"] == 5 and summary["p95"] == 95


# ── KS ───────────────────────────────────────────────────────────────────────
def test_ks_of_disjoint_supports_is_one():
    assert ks_statistic([1, 2, 3], [100, 200, 300]) == 1.0


def test_ks_of_identical_samples_is_zero():
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0


def test_ks_of_partial_overlap_is_the_hand_computed_gap():
    # ECDF_a at 2 = 1.0 (both of a are <= 2); ECDF_b at 2 = 0.5 -> D = 0.5.
    assert ks_statistic([1, 2], [2, 3]) == 0.5


def test_ks_refuses_an_empty_sample():
    with pytest.raises(SizingError):
        ks_statistic([], [1, 2])


# ── Wilson ───────────────────────────────────────────────────────────────────
def test_wilson_width_shrinks_as_n_grows():
    assert wilson_width(400) < wilson_width(100) < wilson_width(25)


def test_wilson_is_widest_at_a_half():
    assert wilson_width(100, 0.5) > wilson_width(100, 0.9)


def test_wilson_stays_inside_the_unit_interval_at_the_boundary():
    lo, hi = wilson_interval(10, 10)
    assert 0.0 <= lo <= hi <= 1.0
    # Wald would put the upper bound at exactly 1.0 with zero width; Wilson does not.
    assert lo < 1.0


def test_wilson_refuses_a_zero_sample():
    with pytest.raises(SizingError):
        wilson_interval(0, 0)


# ── expected_distinct_strata ─────────────────────────────────────────────────
def test_drawing_one_record_hits_exactly_one_stratum():
    assert expected_distinct_strata([99, 1], 1) == pytest.approx(1.0)


def test_drawing_the_whole_frame_hits_every_stratum():
    assert expected_distinct_strata([5, 3, 2], 10) == pytest.approx(3.0)


def test_a_skewed_frame_hits_fewer_strata_than_a_balanced_one_at_the_same_k():
    skewed = expected_distinct_strata([970] + [10, 10, 10], 10)
    balanced = expected_distinct_strata([250, 250, 250, 250], 10)
    assert skewed < balanced


def test_expected_strata_is_monotone_in_k():
    sizes = [50, 30, 20]
    values = [expected_distinct_strata(sizes, k) for k in (1, 5, 20, 60)]
    assert values == sorted(values)


def test_expected_strata_refuses_a_draw_larger_than_the_frame():
    with pytest.raises(SizingError):
        expected_distinct_strata([2, 3], 6)


# ── query supply ─────────────────────────────────────────────────────────────
def test_a_clean_record_is_usable_and_its_length_is_measured_after_degapping():
    supply = query_supply([_record(locus_seq="AC-GU..ACGT", locus_length=11)])
    assert supply["n_usable"] == 1
    # degap_to_dna strips - and . and maps U->T: 8 nt survive, not the 11 declared.
    assert supply["usable_length_nt"]["max"] == 8


def test_a_non_ok_context_status_is_refused_and_the_status_is_named():
    supply = query_supply([_record(context_status="multi_hit")])
    assert supply["n_usable"] == 0
    assert supply["refusal_reasons"] == {"context_status:multi_hit": 1}


def test_a_clipped_locus_is_refused_rather_than_searched_truncated():
    supply = query_supply([_record(locus_seq="ACGT", locus_length=8)])
    assert supply["refusal_reasons"] == {"locus_truncated_by_context_clip": 1}


def test_an_ambiguity_code_is_refused_by_the_shipped_gate():
    # R/W/K/H are IUPAC ambiguity codes; search_homologs raises on exactly these.
    supply = query_supply([_record(locus_seq="ACGTACGR")])
    assert supply["refusal_reasons"] == {"ambiguous_alphabet": 1}


def test_supply_counts_reconcile_and_the_reasons_account_for_every_refusal():
    records = [
        _record(),
        _record(context_status="unavailable"),
        _record(locus_seq="ACGTACGR"),
        _record(locus_seq="ACG", locus_length=8),
    ]
    supply = query_supply(records)
    assert supply["n_usable"] + supply["n_refused"] == supply["n_records"] == 4
    assert sum(supply["refusal_reasons"].values()) == supply["n_refused"] == 3


def test_the_gate_is_the_shipped_one():
    # Binds this module's supply verdict to homolog_msa's own predicate: if that
    # predicate changes, this assertion changes with it rather than drifting.
    assert is_clean_nucleotide("ACGTN") and not is_clean_nucleotide("ACGU")
    assert not is_clean_nucleotide("") and not is_clean_nucleotide("acgt")
    # ⚠ And the binding itself, through query_supply: calling the predicate here
    # proves what the predicate does, NOT that the supply verdict consults it, so
    # the two could diverge with this test still green.
    for locus_seq in ("ACGTN", "ACGTR"):
        supply = query_supply([_record(locus_seq=locus_seq, locus_length=len(locus_seq))])
        assert (supply["n_usable"] == 1) is is_clean_nucleotide(locus_seq)


# ── matchedness ──────────────────────────────────────────────────────────────
def test_identical_query_populations_are_perfectly_matched():
    lengths = [100, 150, 200]
    out = span_matchedness(lengths, lengths)
    assert out["ks_d"] == 0.0
    assert out["share_curated_inside_fp_range"] == 1.0
    assert out["median_ratio_curated_over_fp"] == 1.0


def test_disjoint_query_populations_are_reported_as_disjoint():
    out = span_matchedness([500, 600], [50, 60])
    assert out["ks_d"] == 1.0
    assert out["n_curated_inside_fp_range"] == 0
    assert out["median_ratio_curated_over_fp"] == pytest.approx(10.0)


def test_matchedness_refuses_an_empty_population():
    with pytest.raises(SizingError):
        span_matchedness([], [1, 2])


# ── substrate ────────────────────────────────────────────────────────────────
def test_substrate_counts_records_and_distinct_taxids_separately():
    records = [_record(host_taxid=7), _record(host_taxid=7), _record(host_taxid=99)]
    out = substrate_overlap(
        records, [7, 7, 5], fp_assemblies=["GCA_1.1"], rep_assemblies=["GCA_1.1"]
    )
    assert out["n_records_whose_host_taxid_is_a_rep"] == 2
    assert out["n_distinct_curated_taxids_in_reps"] == 1
    assert out["n_distinct_curated_taxids"] == 2
    assert out["n_rep_taxids"] == 2  # 7 and 5, deduplicated


def test_self_hit_counts_the_intersection_not_the_fp_set():
    out = substrate_overlap(
        [_record()],
        [1301],
        fp_assemblies=["GCA_1.1", "GCA_2.1", "GCA_3.1"],
        rep_assemblies=["GCA_1.1", "GCA_9.1"],
    )
    assert out["fp_self_hit"]["n_fp_assemblies"] == 3
    assert out["fp_self_hit"]["n_fp_assemblies_in_searched_db"] == 1


# ── strata ───────────────────────────────────────────────────────────────────
def test_census_counts_strata_and_summarises_missing_genus():
    records = [
        _record(order="A", cluster_id=1),
        _record(order="A", cluster_id=2, genus=None),
        _record(order="B", cluster_id=3, phylum="Actinobacteria", tbox_type="Translational"),
    ]
    census = stratum_census(records, ks=[2])
    assert census["order"] == {"A": 2, "B": 1}
    assert census["phylum"] == {"Firmicutes": 2, "Actinobacteria": 1}
    assert census["tbox_type"] == {"Transcriptional": 2, "Translational": 1}
    assert census["genus"]["n_records_without_genus"] == 1
    assert census["clusters"]["n_clusters"] == 3


def test_one_per_cluster_feasibility_is_bounded_by_the_cluster_count():
    records = [_record(cluster_id=1), _record(cluster_id=1), _record(cluster_id=2)]
    census = stratum_census(records, ks=[2, 3])
    feasible = {row["k"]: row["one_per_cluster_feasible"] for row in census["draw"]}
    assert feasible == {2: True, 3: False}


def test_the_quota_check_counts_clusters_not_records():
    # One order, three records, but only ONE cluster: a per-order quota of 3 cannot be
    # met with independent records even though three rows exist.
    records = [_record(order="A", cluster_id=1) for _ in range(3)]
    census = stratum_census(records, ks=[3])
    row = census["draw"][0]
    assert row["equal_allocation_per_order"] == 3
    assert row["orders_with_enough_clusters_for_quota"] == 0


def test_largest_cluster_and_singletons_are_reported():
    records = [_record(cluster_id=1), _record(cluster_id=1), _record(cluster_id=2)]
    clusters = stratum_census(records, ks=[1])["clusters"]
    assert clusters["largest_cluster_size"] == 2
    assert clusters["n_singleton_clusters"] == 1


# ── power ────────────────────────────────────────────────────────────────────
def test_power_rows_cover_every_k_and_every_anchor():
    rows = power_table(ks=[10, 20])
    assert len(rows) == 2 * len(PRODUCIBILITY_ANCHORS)


def test_the_interval_is_computed_on_the_effective_n_not_on_k():
    rows = power_table(ks=[100], anchors=[("half", 0.5), ("all", 1.0)])
    by_anchor = {row["producibility_anchor"]: row for row in rows}
    assert by_anchor["half"]["effective_n"] == 50
    assert by_anchor["all"]["effective_n"] == 100
    # If the width were computed on k, both rows would carry the same number.
    assert by_anchor["half"]["wilson_width_95_at_p50"] > by_anchor["all"]["wilson_width_95_at_p50"]


def test_a_producible_share_that_rounds_to_zero_reports_no_interval():
    rows = power_table(ks=[1], anchors=[("none", 0.0)])
    assert rows[0]["effective_n"] == 0 and rows[0]["wilson_width_95_at_p50"] is None


# ── envelope ─────────────────────────────────────────────────────────────────
def test_worst_case_is_search_plus_the_bound_plus_score():
    env = compute_envelope(
        measure_report=MEASURE_REPORT, ks=[20], shard_size=20, align_timeout_s=600
    )
    assert env["worst_case_per_candidate_s"] == pytest.approx(111.5 + 600 + 0.25)


def test_array_width_rounds_up_for_a_non_divisible_k():
    env = compute_envelope(
        measure_report=MEASURE_REPORT, ks=[41], shard_size=20, align_timeout_s=600
    )
    assert env["per_k"][0]["array_width"] == 3


def test_expected_core_hours_scale_with_k():
    env = compute_envelope(
        measure_report=MEASURE_REPORT, ks=[200, 400], shard_size=20, align_timeout_s=600
    )
    small, large = env["per_k"]
    # abs tolerance, not exact: both numbers are rounded for publication, so exact
    # doubling is unavailable — but a projection that ignored k entirely would be
    # equal, and one that scaled on array_width would land elsewhere.
    assert large["expected_core_h"] == pytest.approx(2 * small["expected_core_h"], abs=2e-3)


def test_worst_case_core_hours_are_priced_on_k_not_on_padded_shards():
    # A --time request is per task, so the per-task wall is a FULL shard; a core-hour
    # total is billed on runtime, so it must not charge candidates that do not exist.
    env = compute_envelope(
        measure_report=MEASURE_REPORT, ks=[20, 21], shard_size=20, align_timeout_s=600
    )
    full, padded = env["per_k"]
    assert (full["array_width"], padded["array_width"]) == (1, 2)
    # The padded form would DOUBLE here (two full shards for 21 candidates).
    assert padded["worst_case_core_h"] == pytest.approx(
        full["worst_case_core_h"] * 21 / 20, abs=2e-2
    )
    # …while the per-task wall stays a full shard, unchanged between the two.
    assert padded["worst_case_wall_h_per_task"] == full["worst_case_wall_h_per_task"]


def test_the_two_core_hour_columns_price_the_same_candidate_count():
    env = compute_envelope(
        measure_report=MEASURE_REPORT, ks=[50], shard_size=20, align_timeout_s=600
    )
    row = env["per_k"][0]
    # Both are k * per-candidate * cpus / 3600, differing only in the per-candidate
    # figure, so their ratio is exactly the ratio of those two figures.
    assert row["worst_case_core_h"] / row["expected_core_h"] == pytest.approx(
        env["worst_case_per_candidate_s"]
        / MEASURE_REPORT["wall_seconds"]["total_per_candidate"]["mean"],
        rel=1e-3,
    )


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), 0.0, -1.0])
def test_a_bound_that_cannot_bind_is_refused(bound):
    # nan is the dangerous one: every comparison against it is False, so a bare
    # `<= 0` guard admits it and the recorded bound is inert ([[cost-knobs-can-certify]]).
    with pytest.raises(SizingError):
        compute_envelope(
            measure_report=MEASURE_REPORT, ks=[20], shard_size=20, align_timeout_s=bound
        )


def test_a_non_positive_shard_size_is_refused():
    with pytest.raises(SizingError):
        compute_envelope(measure_report=MEASURE_REPORT, ks=[20], shard_size=0, align_timeout_s=600)


def test_a_measure_report_without_wall_seconds_is_refused_not_defaulted():
    with pytest.raises(SizingError):
        compute_envelope(measure_report={}, ks=[20], shard_size=20, align_timeout_s=600)


def test_a_measure_report_missing_a_stage_is_refused():
    broken = {"wall_seconds": {"total_per_candidate": {"mean": 1.0}}}
    with pytest.raises(SizingError):
        compute_envelope(measure_report=broken, ks=[20], shard_size=20, align_timeout_s=600)


# ── the existing n=1 control ─────────────────────────────────────────────────
def test_a_seed_inside_the_frame_is_reported_as_inside():
    records = [_record(record_sha256="deadbeef")]
    assert existing_control_placement(records, "deadbeef")["in_this_frame"] is True


def test_a_seed_outside_the_frame_is_reported_as_outside():
    records = [_record(record_sha256="deadbeef")]
    placement = existing_control_placement(records, "0" * 64)
    assert placement["in_this_frame"] is False
    assert "NOT in the held-out carve" in placement["note"]


def test_an_absent_seed_is_unmeasured_rather_than_assumed():
    assert existing_control_placement([_record()], None)["in_this_frame"] is None


# ── the FP manifest ──────────────────────────────────────────────────────────
def test_fp_span_lengths_are_end_minus_start():
    manifest = {"candidates": [{"locus_start": 10, "locus_end": 47, "accession": "GCA_1.1:c0"}]}
    assert fp_span_lengths(manifest) == [37]


def test_fp_assembly_is_split_at_the_last_contig_marker():
    manifest = {"candidates": [{"locus_start": 0, "locus_end": 1, "accession": "GCA_1.1:c10"}]}
    assert fp_assemblies(manifest) == ["GCA_1.1"]


def test_an_accession_without_a_contig_marker_is_refused():
    with pytest.raises(SizingError):
        fp_assemblies({"candidates": [{"accession": "GCA_1.1"}]})


def test_a_manifest_without_candidates_is_refused():
    with pytest.raises(SizingError):
        fp_span_lengths({"n_candidates": 0})


def test_a_manifest_row_that_is_not_an_object_is_refused_as_a_shape_defect():
    # `match=` is load-bearing: without the shape guard a string row still raises, but
    # from the span lookup downstream ("no usable locus span") — so a bare
    # `pytest.raises(SizingError)` passes with the guard deleted, and the row shape
    # reaches `fp_assemblies`, where `.get` on a str escapes as an AttributeError
    # rather than the module's exit-3 refusal.
    with pytest.raises(SizingError, match="not an object"):
        fp_span_lengths({"candidates": ["GCA_1.1:c0:1-2"]})


def test_the_assembly_reader_refuses_a_non_object_row_too():
    with pytest.raises(SizingError, match="not an object"):
        fp_assemblies({"candidates": ["GCA_1.1:c0:1-2"]})


def test_a_manifest_row_without_a_span_is_refused():
    with pytest.raises(SizingError):
        fp_span_lengths({"candidates": [{"accession": "GCA_1.1:c0"}]})


# ── the whole body ───────────────────────────────────────────────────────────
def _body(records=None):
    return size_report(
        records=records if records is not None else [_record(), _record(cluster_id=2)],
        fp_manifest={
            "candidates": [
                {"locus_start": 0, "locus_end": 4, "accession": "GCA_1.1:c0"},
                {"locus_start": 0, "locus_end": 6, "accession": "GCA_2.1:c0"},
            ]
        },
        rep_taxids=[1301],
        rep_assemblies=["GCA_1.1", "GCA_2.1"],
        measure_report=MEASURE_REPORT,
        ks=[2],
        shard_size=2,
        align_timeout_s=600,
        seed_record_id="0" * 64,
    )


def test_the_body_carries_every_block_and_pins_nothing():
    body = _body()
    assert body["pins_nothing"] is True
    assert body["step"] == "P3-15'-g"
    for key in (
        "frame",
        "query_supply",
        "existing_positive_control",
        "matchedness",
        "substrate",
        "strata",
        "power",
        "envelope",
    ):
        assert key in body


def test_the_body_does_not_leak_the_internal_usable_list():
    # `_usable` is an internal hand-off between query_supply and the length measurement;
    # publishing it would put 8,709 locus sequences into a committed report.
    assert "_usable" not in _body()["query_supply"]


def test_an_empty_frame_is_refused_rather_than_sized():
    with pytest.raises(SizingError):
        _body(records=[])


def test_a_frame_whose_every_record_fails_the_gate_is_refused():
    # Named refusal, not just "something raised": with the guard removed the empty
    # usable list reaches the length measurement and `percentiles` returns {"n": 0},
    # i.e. a report that measures a supply of nothing and says so nowhere.
    with pytest.raises(SizingError, match="no curated record survived"):
        _body(records=[_record(context_status="unavailable")])


# ── the committed report ─────────────────────────────────────────────────────
def test_the_env_lock_this_report_stamps_itself_with_exists():
    # A provenance stamp naming a lockfile that is not there records `null` and the
    # §11 field is silently unmet.
    assert (REPO_ROOT / ENV_LOCK).is_file()


@pytest.mark.skipif(not COMMITTED_REPORT.is_file(), reason="report not generated in this tree")
def test_the_committed_report_publishes_no_absolute_path():
    # ⚠ This repo is PUBLIC. The scan walks the WHOLE payload rather than the fields
    # known to hold paths, because on P3-15'-f the four known sites were never the
    # risk — a fifth, added later, was.
    def walk(node, trail="$"):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from walk(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield from walk(value, f"{trail}[{i}]")
        elif isinstance(node, str):
            yield trail, node

    leaks = [
        (trail, value)
        for trail, value in walk(json.loads(COMMITTED_REPORT.read_text()))
        if value.startswith("/")
    ]
    assert leaks == []


@pytest.mark.skipif(not COMMITTED_REPORT.is_file(), reason="report not generated in this tree")
def test_the_committed_report_records_every_input_it_read():
    provenance = json.loads(COMMITTED_REPORT.read_text())["provenance"]
    # ⚠ `inputs` ONLY. Merging in the external basenames would let an out-of-repo
    # file with a colliding basename stand in for a missing repo default — the
    # assertion would pass while the input it names was never read.
    recorded = provenance["inputs"]
    # Every default input the CLI reads must leave a trace: a number in this report
    # that traces back to no recorded file is a number a reader cannot check.
    for path in (
        DEFAULT_CORPUS,
        DEFAULT_SPLIT_TABLE,
        DEFAULT_CONTEXT,
        DEFAULT_FP_MANIFEST,
        DEFAULT_HOST_MANIFEST,
        DEFAULT_MEASURE_REPORT,
        DEFAULT_CONTROL_SEED,
    ):
        assert path in recorded, path
    # Hex, not merely 64 characters: "g" * 64 is the right length and no digest.
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in recorded.values())
    # And the env stamp compared against the lockfile it claims to be — a truthiness
    # check passes on any hardcoded string ([[pinned-constant-that-nothing-reads]]).
    assert (
        provenance["env_lock_hash"]
        == hashlib.sha256((REPO_ROOT / ENV_LOCK).read_bytes()).hexdigest()
    )


@pytest.mark.skipif(not COMMITTED_REPORT.is_file(), reason="report not generated in this tree")
def test_the_committed_report_is_internally_consistent():
    body = json.loads(COMMITTED_REPORT.read_text())
    supply = body["query_supply"]
    assert supply["n_usable"] + supply["n_refused"] == supply["n_records"]
    assert sum(supply["refusal_reasons"].values()) == supply["n_refused"]
    assert supply["n_records"] == body["frame"]["n_records"]
    assert 0.0 <= body["matchedness"]["ks_d"] <= 1.0
    assert body["pins_nothing"] is True
    for row in body["strata"]["draw"]:
        assert row["expected_orders_hit_uniform"] <= row["orders_available"]
    for row in body["power"]["rows"]:
        assert row["effective_n"] <= row["k"]
    for row in body["envelope"]["per_k"]:
        assert row["worst_case_wall_h_per_task"] >= row["expected_wall_h_per_task"]
        assert row["array_width"] == math.ceil(row["k"] / row["shard_size"])


# ── the pandas boundary ──────────────────────────────────────────────────────
def _joined(**overrides):
    row = {
        "_record_sha256": "b" * 64,
        "status": "ok",
        "context_seq": "TTTACGTACGTTTT",
        "locus_offset": 3,
        "locus_length": 8,
        "type": "Transcriptional",
        "TaxId": 1301,
        "cluster_id": 7,
        "resolved_phylum": "Firmicutes",
        "resolved_order": "Lactobacillales",
        "resolved_genus": "Lactobacillus",
    }
    row.update(overrides)
    return row


def test_the_locus_is_carved_out_of_the_context_at_its_offset():
    record = records_from_joined([_joined()])[0]
    assert record["locus_seq"] == "ACGTACGT"
    assert (record["cluster_id"], record["host_taxid"]) == (7, 1301)


def test_a_missing_column_is_refused_by_name_not_filled_with_none():
    row = _joined()
    del row["resolved_genus"]
    with pytest.raises(SizingError, match="resolved_genus"):
        records_from_joined([row])


def test_the_column_check_reads_every_row_not_row_zero():
    bad = _joined()
    del bad["TaxId"]
    with pytest.raises(SizingError, match="row 1"):
        records_from_joined([_joined(), bad])


def test_a_nan_stratum_becomes_none_rather_than_a_stratum_called_nan():
    # pandas emits float('nan') for a missing object cell; nan is not None, so an
    # un-normalised value would survive every `is not None` filter and be counted.
    record = records_from_joined([_joined(resolved_genus=float("nan"))])[0]
    assert record["genus"] is None
    census = stratum_census([record], ks=[1])
    assert census["genus"]["n_distinct"] == 0
    assert census["genus"]["n_records_without_genus"] == 1


def test_a_nan_coordinate_yields_no_locus_sequence_rather_than_a_slice():
    record = records_from_joined([_joined(locus_offset=float("nan"))])[0]
    assert record["locus_seq"] is None and record["locus_length"] == 8
    assert query_supply([record])["refusal_reasons"] == {"no_locus_sequence": 1}


# ── malformed spans (a zero median divides) ──────────────────────────────────
def test_a_non_positive_fp_span_is_refused_at_the_reader():
    with pytest.raises(SizingError, match="non-positive locus span"):
        fp_span_lengths({"candidates": [{"locus_start": 40, "locus_end": 40, "accession": "x:c0"}]})


def test_matchedness_refuses_a_zero_median_rather_than_dividing_by_it():
    # ZeroDivisionError is not in `main`'s except tuple, so this would be a traceback
    # where every other malformed input exits 3 with a refusal.
    with pytest.raises(SizingError, match="median query length is not positive"):
        span_matchedness([100, 200], [0, 0, 0])


# ── the slice boundary (a wrong locus at the right length) ───────────────────
def test_a_negative_offset_is_refused_rather_than_slicing_from_the_end():
    # seq[-4:-4+8] returns 4 characters here, but with a longer tail it returns
    # exactly locus_length characters — a DIFFERENT locus that passes every
    # downstream length check and gets searched as though it were this record.
    row = _joined(context_seq="AAAACCCCGGGGTTTTAAAACCCC", locus_offset=-8, locus_length=8)
    record = records_from_joined([row])[0]
    assert record["locus_seq"] is None
    # And it is reported as a COORDINATE defect, not as a context clip: a negative
    # offset is a broken row, not a window that ran off a replicon end.
    assert query_supply([record])["refusal_reasons"] == {"locus_coordinates_unusable": 1}


def test_a_slice_running_past_the_end_of_the_context_is_refused_at_the_boundary():
    record = records_from_joined([_joined(locus_offset=10, locus_length=8)])[0]
    assert record["locus_seq"] is None


def test_a_zero_length_locus_is_refused():
    record = records_from_joined([_joined(locus_length=0)])[0]
    assert record["locus_seq"] is None and record["locus_coords_invalid"] is True


# ── provenance input partitioning (the branch the report cannot reach) ───────
def test_a_repo_input_is_recorded_repo_relative():
    repo_inputs, external = partition_inputs([("seed", REPO_ROOT / DEFAULT_CONTROL_SEED)])
    assert repo_inputs == [DEFAULT_CONTROL_SEED] and external == {}


def test_an_external_input_is_recorded_by_basename_AND_hash_never_by_path(tmp_path):
    staged = tmp_path / "staged.sto"
    staged.write_text("# STOCKHOLM 1.0\n", encoding="utf-8")
    repo_inputs, external = partition_inputs([("seed", staged)])
    assert repo_inputs == []
    assert external["seed"]["name"] == "staged.sto"
    # The hash is the point: a basename alone cannot be checked against anything.
    assert external["seed"]["sha256"] == hashlib.sha256(staged.read_bytes()).hexdigest()
    # And nothing in the entry may carry the absolute staging path.
    assert not any(str(value).startswith("/") for value in external["seed"].values())


def test_an_absent_input_is_skipped_rather_than_recorded_as_a_hashless_name(tmp_path):
    repo_inputs, external = partition_inputs([("missing", tmp_path / "not-there.json")])
    assert (repo_inputs, external) == ([], {})


def test_a_duplicated_representative_is_counted_once_as_a_genome():
    # `n_production_genomes` and the self-hit intersection must read ONE population:
    # a manifest with two rows for a representative would otherwise publish a genome
    # count larger than the set the other number was measured against.
    out = substrate_overlap(
        [_record()],
        [1301],
        fp_assemblies=["GCA_1.1"],
        rep_assemblies=["GCA_1.1", "GCA_1.1", "GCA_2.1"],
    )
    assert out["n_production_genomes"] == 2
    assert out["fp_self_hit"]["n_fp_assemblies_in_searched_db"] == 1


def test_an_unreadable_seed_alignment_exits_three_not_one(tmp_path, monkeypatch, capsys):
    # HomologMsaError subclasses RuntimeError, so without it in the except tuple an
    # empty or out-of-range seed alignment leaves the CLI's exit-3 refusal convention
    # for a traceback. Raised through the loader seam so the assertion is about the
    # HANDLER and needs neither the DVC parquet nor a parquet engine.
    import tbox_finder.mining.curated_control_sizing as mod

    def _raise(**_kwargs):
        raise HomologMsaError("no FASTA record found in the seed alignment")

    # Both parquet seams are stubbed: the rep-manifest read happens first, and the
    # local test env has no parquet engine — the point of this test is the HANDLER.
    monkeypatch.setattr(mod, "_load_rep_columns", lambda _path: ([], []))
    monkeypatch.setattr(mod, "load_frame", _raise)
    code = mod.main(
        [
            "size",
            "--k",
            "10",
            "--shard-size",
            "5",
            "--align-timeout-s",
            "600",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == 3
    assert "refused:" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


# ── the seed's identifier space (a False that could mean "type mismatch") ────
def test_a_seed_outside_the_carve_but_inside_the_corpus_is_a_real_placement():
    records = [_record(record_sha256="a" * 64)]
    placement = existing_control_placement(records, "b" * 64, corpus_record_ids={"b" * 64})
    assert placement["in_this_frame"] is False
    assert placement["in_full_corpus"] is True
    assert "non-heldout side" in placement["note"]


def test_a_seed_in_no_identifier_space_is_unmeasured_not_negative():
    # A Stockholm labelled "ACC/start-end" matches no record hash, so a bare False
    # would publish a TYPE MISMATCH as the measured result
    # ([[namespace-mismatch-invisible-noop]]).
    records = [_record(record_sha256="a" * 64)]
    placement = existing_control_placement(
        records, "CP000truncated/12-345", corpus_record_ids={"a" * 64}
    )
    assert placement["in_this_frame"] is None
    assert placement["in_full_corpus"] is False
    assert "different identifier space" in placement["note"]


def test_without_the_corpus_control_the_placement_still_reports_its_absence():
    placement = existing_control_placement([_record(record_sha256="a" * 64)], "b" * 64)
    assert placement["in_full_corpus"] is None


# ── the clipped-locus bucket (two facts that were reported as one) ───────────
def test_a_clipped_locus_is_reported_as_clipped_not_as_missing_context():
    record = records_from_joined([_joined(locus_offset=10, locus_length=8)])[0]
    assert record["locus_clipped"] is True
    assert query_supply([record])["refusal_reasons"] == {"locus_truncated_by_context_clip": 1}


def test_a_record_with_no_context_at_all_is_reported_as_missing_context():
    record = records_from_joined([_joined(context_seq=None)])[0]
    assert record["locus_clipped"] is False
    assert query_supply([record])["refusal_reasons"] == {"no_locus_sequence": 1}


# ── malformed measure-report sub-blocks ─────────────────────────────────────
@pytest.mark.parametrize("block", ["homolog_depth", "status_counts"])
def test_a_scalar_measure_report_block_is_refused_not_dict_converted(block):
    broken = {**MEASURE_REPORT, block: 3}
    with pytest.raises(SizingError, match=block):
        compute_envelope(measure_report=broken, ks=[20], shard_size=20, align_timeout_s=600)


def test_a_scalar_per_candidate_wall_block_is_refused():
    broken = {"wall_seconds": {**MEASURE_REPORT["wall_seconds"], "total_per_candidate": 5}}
    with pytest.raises(SizingError):
        compute_envelope(measure_report=broken, ks=[20], shard_size=20, align_timeout_s=600)


# ── the KS tail (a review claimed a defect here; it is measured, not argued) ──
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 2, 3], [1], 0.6667),  # unequal maxima — the case round 5 raised
        ([1], [1, 2, 3, 4], 0.75),
        ([5, 6], [1, 2, 3], 1.0),
        ([1, 1, 1], [1, 2], 0.5),
    ],
)
def test_ks_matches_a_brute_force_ecdf_when_the_supports_end_differently(a, b, expected):
    # When one sample is exhausted its ECDF is 1.0 and the other's only rises, so the
    # tail gap shrinks monotonically from the value the loop already recorded at the
    # exit point — the merge loop is complete, and these are the pinned values.
    assert ks_statistic(a, b) == expected


def test_ks_agrees_with_a_reference_implementation_on_the_whole_grid():
    def reference(a, b):
        points = sorted(set(a) | set(b))
        return max(
            abs(sum(1 for x in a if x <= t) / len(a) - sum(1 for x in b if x <= t) / len(b))
            for t in points
        )

    for a in ([1], [1, 2], [1, 2, 3], [2, 2, 5], [4, 9]):
        for b in ([1], [1, 4], [3], [1, 2, 3, 9], [2, 2]):
            assert ks_statistic(a, b) == round(reference(a, b), 4), (a, b)


def test_a_placement_without_the_corpus_control_does_not_claim_corpus_membership():
    placement = existing_control_placement([_record(record_sha256="a" * 64)], "b" * 64)
    assert placement["in_full_corpus"] is None
    assert "UNMEASURED" in placement["note"]
    assert "non-heldout side" not in placement["note"]


def test_an_unserializable_report_body_is_refused_not_traced_back(tmp_path, monkeypatch, capsys):
    import tbox_finder.mining.curated_control_sizing as mod

    monkeypatch.setattr(mod, "_load_rep_columns", lambda _path: ([], []))
    monkeypatch.setattr(mod, "load_frame", lambda **_kwargs: [])
    monkeypatch.setattr(mod, "corpus_record_ids", lambda _path: set())
    # A body carrying something json cannot encode (a set), reached after the whole
    # measurement has run — the traceback would land at the very last step.
    monkeypatch.setattr(mod, "size_report", lambda **_kwargs: {"strata": {"phylum": {1j: 2}}})
    code = mod.main(
        [
            "size",
            "--k",
            "10",
            "--shard-size",
            "5",
            "--align-timeout-s",
            "600",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == 3
    assert "not JSON-serializable" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
