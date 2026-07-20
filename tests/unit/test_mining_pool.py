"""P2-10b: coordinate recovery, namespace normalisation, and window geometry.

Every test here guards a way the P2-10b mask measurement could report a clean,
non-vacuous-looking number while measuring nothing:

* a coordinate that parses but addresses a different accession namespace,
* a coordinate invented for a record that has no replicon,
* a window whose forward-genome bounds are computed in the wrong frame,
* a ``NaN`` coordinate that passes an ``is not None`` guard and is then mined.

stdlib-only (no pandas/pyarrow) so it runs in bare CI alongside the pinned env.
"""

from __future__ import annotations

import pytest

from tbox_finder import masking
from tbox_finder.data.flank_context import STRAND_MINUS, STRAND_PLUS, forward_bounds
from tbox_finder.decoys import parse_structured_rna_locus
from tbox_finder.mining import pool as mining_pool
from tbox_finder.mining.hard_negative import (
    MINEABLE_POOLS,
    OUTCOME_MASKED,
    OUTCOME_MINED,
    OUTCOME_UNMASKABLE,
    MiningCandidate,
    classify_candidate,
)
from tbox_finder.mining.spare_rule import (
    MODEL_INDEPENDENT_DISJUNCTS,
    STATUS_FAILED,
    SpareRuleEvidence,
)


# --------------------------------------------------------------------------- #
# Rfam header → coordinates
# --------------------------------------------------------------------------- #
def test_a_plain_rfam_header_yields_forward_coordinates() -> None:
    parsed = parse_structured_rna_locus("SAM|RF00162|CBXI010000044.1/24675-24784")
    assert parsed == ("CBXI010000044.1", 24675, 24784, STRAND_PLUS)


def test_a_reverse_strand_header_keeps_its_orientation() -> None:
    """``start > end`` is the minus strand; masking folds it, so it is kept as written."""
    parsed = parse_structured_rna_locus("SAM|RF00162|ABFD02000009.1/97533-97428")
    assert parsed == ("ABFD02000009.1", 97533, 97428, STRAND_MINUS)
    assert masking.normalize_locus(97533, 97428) == (97428, 97533)


def test_a_description_containing_pipes_does_not_shift_the_locator_field() -> None:
    """The real trap: one staged header's free text embeds ``|``.

    ``header.split("|")[2]`` yields five fields for this record, so the locator
    must be taken from the first whitespace token before any pipe split.
    """
    header = "TPP|RF00059|AGNL01045717.1/7292-7390 ENA|AGNL01045717|marine metagenome"
    assert parse_structured_rna_locus(header) == ("AGNL01045717.1", 7292, 7390, STRAND_PLUS)


@pytest.mark.parametrize(
    ("header", "accession"),
    [
        ("tRNA|RF00005|NC_021162.1/16412073-16412145", "NC_021162.1"),
        ("tRNA|RF00005|NW_009526658.1/44125-44054", "NW_009526658.1"),
    ],
)
def test_refseq_accessions_are_admitted(header: str, accession: str) -> None:
    """18 tRNA records are RefSeq (``NC_``/``NW_``); an INSDC-only pattern drops them.

    Their coordinates are real, so refusing them would understate coverage and
    quietly discard genomic candidates.
    """
    parsed = parse_structured_rna_locus(header)
    assert parsed is not None and parsed[0] == accession


def test_a_non_replicon_identifier_is_refused_rather_than_given_coordinates() -> None:
    """An RNAcentral URS has an interval but no genome — a coordinate here is fiction.

    Returning ``(URS…, 1, 112)`` would put a non-null coordinate in the artifact
    that addresses no replicon and can never mask, i.e. exactly the invisible
    no-op P2-10b exists to remove. Refusing keeps it countable.
    """
    assert parse_structured_rna_locus("tRNA|RF00005|URS000080DE72_32630/1-112") is None


@pytest.mark.parametrize(
    "header",
    [
        "SAM|RF00162",  # too few pipe fields
        "SAM|RF00162|CBXI010000044.1",  # no interval
        "SAM|RF00162|CBXI010000044.1/abc-def",  # non-numeric interval
        "SAM|RF00162|CBXI010000044.1/0-100",  # 1-based frame: 0 is not a coordinate
        "",
    ],
)
def test_an_unparseable_header_returns_none_never_a_guess(header: str) -> None:
    assert parse_structured_rna_locus(header) is None


# --------------------------------------------------------------------------- #
# Accession-namespace normalisation — the measured silent-no-op
# --------------------------------------------------------------------------- #
def test_a_versioned_query_masks_against_an_unversioned_index() -> None:
    """The measured P2-10b defect: 0 of 2,750 coordinate-bearing Rfam accessions intersected as-is.

    The union prior stores accessions unversioned and every external coordinate
    source stores them versioned, so without normalisation this mask never fires
    and the pool reads clean.
    """
    index = masking.LocusIndex.from_records([("CBXI010000044", 24_000, 25_000)])
    assert index.is_masked("CBXI010000044.1", 24_675, 24_784, flank=50) is True
    assert index.is_masked("CBXI010000044", 24_675, 24_784, flank=50) is True


def test_two_versions_of_one_accession_pool_their_intervals() -> None:
    """Collapsing keys must merge intervals, not let the later version win."""
    index = masking.LocusIndex.from_records(
        [("CP001598.1", 100, 200), ("CP001598.2", 5_000, 5_100)]
    )
    assert index.n_accessions == 1
    assert index.is_masked("CP001598", 150, 160) is True
    assert index.is_masked("CP001598", 5_050, 5_060) is True


def test_the_namespace_report_separates_incompatible_from_clean() -> None:
    """A zero mask count is unreadable without this: no overlap, or no shared namespace?"""
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    compatible = masking.accession_namespace_report(["CP001598.1", "CP001598.2"], index)
    assert compatible["n_intersect_as_is"] == 0
    assert compatible["n_intersect_normalized"] == 1
    assert compatible["normalization_recovered"] == 1
    assert compatible["namespace_compatible"] is True

    foreign = masking.accession_namespace_report(["XX999999.1"], index)
    assert foreign["n_intersect_normalized"] == 0
    assert foreign["namespace_compatible"] is False


def test_a_pool_with_no_accessions_at_all_is_not_reported_compatible() -> None:
    """Emptiness must not be vacuously TRUE — that is the clause-guard rule."""
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    assert masking.accession_namespace_report([], index)["namespace_compatible"] is False
    assert masking.accession_namespace_report([None], index)["namespace_compatible"] is False


# --------------------------------------------------------------------------- #
# Forward-genome window geometry (the frame the 100% control validates)
# --------------------------------------------------------------------------- #
def test_plus_strand_bounds_index_from_the_region_start() -> None:
    assert forward_bounds(
        strand=STRAND_PLUS, region_start=1_001, region_len=500, offset=0, length=300
    ) == (1_001, 1_300)
    assert forward_bounds(
        strand=STRAND_PLUS, region_start=1_001, region_len=500, offset=200, length=300
    ) == (1_201, 1_500)


def test_minus_strand_bounds_index_from_the_last_fetched_base() -> None:
    """``context_seq`` is reverse-complemented server-side, so index 0 is the HIGH end.

    Anchoring to ``region_start`` instead scored 23,146/23,532 on the locus-centred
    control; this frame scores 23,532/23,532.
    """
    # region [1001, 1500] returned in full: index 0 ↔ forward 1500.
    assert forward_bounds(
        strand=STRAND_MINUS, region_start=1_001, region_len=500, offset=0, length=300
    ) == (1_201, 1_500)
    assert forward_bounds(
        strand=STRAND_MINUS, region_start=1_001, region_len=500, offset=200, length=300
    ) == (1_001, 1_300)


def test_a_clipped_minus_strand_region_uses_its_actual_length_not_the_requested_stop() -> None:
    """``plan_region`` cannot clamp the stop, so a truncated region skews the frame.

    All 345 residual misses of the ``region_stop``-anchored frame were clipped
    minus-strand rows; keying on ``region_start + region_len - 1`` removes them.
    """
    # 500 nt requested from 1001, only 400 returned → last forward base is 1400.
    assert forward_bounds(
        strand=STRAND_MINUS, region_start=1_001, region_len=400, offset=0, length=100
    ) == (1_301, 1_400)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(strand=3, region_start=1, region_len=10, offset=0, length=5),
        dict(strand=STRAND_PLUS, region_start=0, region_len=10, offset=0, length=5),
        dict(strand=STRAND_PLUS, region_start=1, region_len=10, offset=8, length=5),
        dict(strand=STRAND_PLUS, region_start=1, region_len=10, offset=-1, length=5),
        dict(strand=STRAND_PLUS, region_start=1, region_len=10, offset=0, length=0),
    ],
)
def test_forward_bounds_refuses_an_impossible_window(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        forward_bounds(**kwargs)


def test_the_two_strands_disagree_so_the_frame_is_not_a_free_parameter() -> None:
    """A guard against a 'fix' that makes both strands share one formula."""
    plus = forward_bounds(
        strand=STRAND_PLUS, region_start=1_001, region_len=500, offset=0, length=300
    )
    minus = forward_bounds(
        strand=STRAND_MINUS, region_start=1_001, region_len=500, offset=0, length=300
    )
    assert plus != minus


# --------------------------------------------------------------------------- #
# Window carving
# --------------------------------------------------------------------------- #
def _context_row(**kw: object) -> dict:
    base: dict = {
        "record_id": "rec1",
        "accession": "CP001598.1",
        "strand": STRAND_PLUS,
        "region_start": 1_001,
        "locus_offset": 400,
        "locus_length": 200,
        "lead_flank": 400,
        "trail_flank": 400,
        "status": "ok",
        "context_seq": "A" * 1_000,
    }
    base.update(kw)
    return base


def test_a_lead_window_is_carved_at_the_outer_edge_with_real_coordinates() -> None:
    window = mining_pool.carve_window(_context_row(), "lead", window_nt=300, margin_nt=50)
    assert window is not None
    assert (window["locus_start"], window["locus_end"]) == (1_001, 1_300)
    assert window["length"] == 300 == len(window["sequence"])
    assert window["is_designed_control"] is False
    assert window["source_record_id"] == "rec1"


def test_a_trail_window_is_carved_at_the_far_edge() -> None:
    window = mining_pool.carve_window(_context_row(), "trail", window_nt=300, margin_nt=50)
    assert window is not None
    assert (window["locus_start"], window["locus_end"]) == (1_701, 2_000)


def test_the_designed_control_is_the_locus_itself_and_is_flagged() -> None:
    control = mining_pool.carve_window(
        _context_row(), mining_pool.SIDE_LOCUS_CONTROL, window_nt=300, margin_nt=50
    )
    assert control is not None
    assert (control["locus_start"], control["locus_end"]) == (1_401, 1_600)
    assert control["is_designed_control"] is True


def test_a_flank_shorter_than_window_plus_margin_yields_no_window() -> None:
    """Never a truncated or invented window — the row is skipped."""
    assert (
        mining_pool.carve_window(_context_row(lead_flank=349), "lead", window_nt=300, margin_nt=50)
        is None
    )


@pytest.mark.parametrize("status", ["multi_hit", "unavailable", "bad_name", "not_found"])
def test_an_unanchored_row_yields_no_window(status: str) -> None:
    """Non-``ok`` rows carry sentinel offsets, not coordinates.

    The geometry is deliberately **valid** here so the status check is the sole
    discriminator. An earlier form also passed ``locus_offset=-1``, tripping the
    status guard and the sentinel guard at once — it stayed green under removal of
    either guard alone and only reddened when both were removed, i.e. it passed
    under the wrong implementation. The ml tier cannot cover this either: every
    real non-``ok`` row has an empty ``context_seq``, so ``region_len <= 0``
    refuses it whatever the status guard does.
    """
    assert (
        mining_pool.carve_window(_context_row(status=status), "lead", window_nt=300, margin_nt=50)
        is None
    )


def test_a_sentinel_locus_offset_yields_no_window() -> None:
    """The other guard, on its own: an anchored-looking row with a sentinel offset."""
    assert (
        mining_pool.carve_window(_context_row(locus_offset=-1), "lead", window_nt=300, margin_nt=50)
        is None
    )


def test_the_fixture_itself_carves_a_window_so_the_guards_are_the_discriminator() -> None:
    """Without this, both tests above could pass because the fixture never carves."""
    assert mining_pool.carve_window(_context_row(), "lead", window_nt=300, margin_nt=50) is not None


@pytest.mark.parametrize("bad", [None, float("nan"), "", "   "])
def test_a_missing_or_blank_accession_yields_no_window(bad: object) -> None:
    """``str(None)`` is ``"None"`` — a non-missing string that every guard accepts.

    Laundering a null this way would put a window carrying a coordinate that
    addresses no replicon into a MINEABLE pool, where it classifies as *minable*
    rather than refused.
    """
    assert (
        mining_pool.carve_window(_context_row(accession=bad), "lead", window_nt=300, margin_nt=50)
        is None
    )


def test_carve_pool_tags_controls_separately_from_the_natural_windows() -> None:
    rows = [_context_row(record_id=f"rec{i}") for i in range(10)]
    records = mining_pool.carve_pool(rows, seed=42, window_nt=300, margin_nt=50, n_controls=4)
    controls = [r for r in records if r["is_designed_control"]]
    natural = [r for r in records if not r["is_designed_control"]]
    assert len(controls) == 4
    assert len(natural) == 20  # lead + trail per row
    assert {r["side"] for r in natural} == {"lead", "trail"}


def test_carve_pool_is_deterministic_under_a_fixed_seed() -> None:
    rows = [_context_row(record_id=f"rec{i}") for i in range(10)]
    kw = dict(window_nt=300, margin_nt=50, n_controls=4)
    first = mining_pool.carve_pool(rows, seed=42, **kw)
    assert [r["candidate_id"] for r in first] == [
        r["candidate_id"] for r in mining_pool.carve_pool(rows, seed=42, **kw)
    ]


def test_the_control_draw_actually_depends_on_the_seed() -> None:
    """Determinism alone is satisfied by ignoring the seed entirely.

    Comparing two same-seed calls passes under ``anchored[:n]`` — and under a
    hardcoded RNG key — so the ``seed`` stamped into the provenance would be
    backed by no assertion at all (CLAUDE.md §8.3).
    """
    rows = [_context_row(record_id=f"rec{i}") for i in range(10)]
    kw = dict(window_nt=300, margin_nt=50, n_controls=4)
    ids = lambda recs: [r["candidate_id"] for r in recs if r["is_designed_control"]]  # noqa: E731
    assert ids(mining_pool.carve_pool(rows, seed=42, **kw)) != ids(
        mining_pool.carve_pool(rows, seed=43, **kw)
    )


def test_the_control_draw_is_not_a_head_slice() -> None:
    """A seeded sample must not coincide with taking the first N rows."""
    rows = [_context_row(record_id=f"rec{i}") for i in range(10)]
    picked = [
        r["candidate_id"]
        for r in mining_pool.carve_pool(rows, seed=42, window_nt=300, margin_nt=50, n_controls=4)
        if r["is_designed_control"]
    ]
    assert picked != [f"rec{i}:locus_control" for i in range(4)]


# --------------------------------------------------------------------------- #
# The non-vacuity gate
# --------------------------------------------------------------------------- #
def _summary(**kw: object) -> dict:
    base: dict = {
        "designed_control": {
            "n_records": 10,
            "n_masked_at_flank": 10,
            "n_overlapping_at_flank_0": 10,
            "n_exact_interval_match": 10,
        },
        "natural": {"n_records": 100, "n_masked_at_flank": 1},
        "accession_namespace": {"namespace_compatible": True},
    }
    base.update(kw)
    return base


def _gate(**kw: object) -> dict:
    return mining_pool.control_gate(_summary(**kw), n_controls_requested=10)


# --------------------------------------------------------------------------- #
# mask_pool — the function every gate number comes from
# --------------------------------------------------------------------------- #
def _window(**kw: object) -> dict:
    base: dict = {
        "accession": "CP001598.1",
        "locus_start": 1_000,
        "locus_end": 1_100,
        "is_designed_control": False,
    }
    base.update(kw)
    return base


def test_mask_pool_keeps_the_designed_and_natural_partitions_disjoint() -> None:
    """A control counted in the natural numerator inflates the reported contamination."""
    index = masking.LocusIndex.from_records([("CP001598", 1_050, 1_060)])
    summary = mining_pool.mask_pool(
        [
            _window(is_designed_control=True),
            _window(locus_start=900_000, locus_end=900_100),
        ],
        index,
        flank=50,
    )
    assert summary["designed_control"]["n_records"] == 1
    assert summary["natural"]["n_records"] == 1
    assert summary["designed_control"]["n_masked_at_flank"] == 1
    assert summary["natural"]["n_masked_at_flank"] == 0
    assert summary["natural"]["masked_fraction"] == 0.0


def test_mask_pool_separates_overlap_from_proximity() -> None:
    """ "Masked" at a flank means *near*; only flank 0 means *intersects*."""
    index = masking.LocusIndex.from_records([("CP001598", 1_130, 1_200)])
    summary = mining_pool.mask_pool([_window()], index, flank=50)
    assert summary["natural"]["n_masked_at_flank"] == 1  # 30 nt away, within flank
    assert summary["natural"]["n_overlapping_at_flank_0"] == 0


def test_mask_pool_refuses_a_non_positive_flank() -> None:
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    for bad in (0, -1):
        with pytest.raises(mining_pool.MiningPoolError, match="flank must be positive"):
            mining_pool.mask_pool([_window()], index, flank=bad)


def test_mask_pool_on_an_empty_pool_reports_zero_not_a_passing_fraction() -> None:
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    summary = mining_pool.mask_pool([], index, flank=50)
    assert summary["designed_control"]["n_records"] == 0
    assert summary["natural"]["n_records"] == 0
    assert mining_pool.control_gate(summary, n_controls_requested=0)["overall_pass"] is False


def test_the_gate_passes_when_every_control_masks() -> None:
    assert _gate()["overall_pass"] is True


def test_one_unmasked_control_fails_the_gate() -> None:
    """Controls overlap a known locus by construction: 9/10 is a frame defect."""
    gate = _gate(
        designed_control={
            "n_records": 10,
            "n_masked_at_flank": 9,
            "n_overlapping_at_flank_0": 9,
            "n_exact_interval_match": 9,
        }
    )
    assert gate["overall_pass"] is False
    assert gate["clauses"]["all_controls_masked"] is False


def test_a_control_that_only_masks_within_the_flank_fails_the_gate() -> None:
    """The sub-flank frame error: 100% "masked" while 0% actually overlap.

    A uniform frame shift smaller than ``flank_nt`` keeps every control inside the
    flank and so scores a perfect ``n_masked_at_flank``. Gating on the flank count
    alone would pass it, with the refuting number sitting unread in the same dict.
    """
    gate = _gate(
        designed_control={
            "n_records": 10,
            "n_masked_at_flank": 10,
            "n_overlapping_at_flank_0": 0,
            "n_exact_interval_match": 0,
        }
    )
    assert gate["overall_pass"] is False
    assert gate["clauses"]["all_controls_masked"] is True
    assert gate["clauses"]["all_controls_overlap"] is False


def test_a_zero_natural_rate_does_not_fail_the_gate() -> None:
    """No natural contamination is a legitimate result; a dead control is not."""
    gate = _gate(natural={"n_records": 100, "n_masked_at_flank": 0})
    assert gate["overall_pass"] is True


@pytest.mark.parametrize(
    "missing",
    [
        {"designed_control": {}},
        {"natural": {}},
        {"accession_namespace": {}},
        {
            "designed_control": {
                "n_records": 0,
                "n_masked_at_flank": 0,
                "n_overlapping_at_flank_0": 0,
            }
        },
        {"accession_namespace": {"namespace_compatible": False}},
    ],
)
def test_an_absent_or_empty_measurement_fails_rather_than_passing_vacuously(
    missing: dict,
) -> None:
    """0 == 0 makes ``all_controls_masked`` TRUE exactly when no control ran."""
    assert _gate(**missing)["overall_pass"] is False


def test_the_gate_is_total_on_a_completely_empty_summary() -> None:
    assert mining_pool.control_gate({})["overall_pass"] is False


# --------------------------------------------------------------------------- #
# MINEABLE_POOLS + the fail-open NaN guard
# --------------------------------------------------------------------------- #
def test_dinuc_shuffled_is_no_longer_mineable() -> None:
    """ADR-0005 A6: its only coordinates are its parents', so mining it self-masks."""
    assert "dinuc_shuffled" not in MINEABLE_POOLS
    assert "gc_background" not in MINEABLE_POOLS


def test_the_coordinate_bearing_substrate_is_mineable() -> None:
    assert mining_pool.POOL_GENOMIC_WINDOW in MINEABLE_POOLS


def test_a_nan_accession_is_refused_not_mined() -> None:
    """The fail-open direction: ``NaN is not None``, so an identity guard passes it.

    The candidate would then reach the mask, match nothing, and be classified
    ``mined`` — a real T-box locus admitted to the negative pool on the strength
    of a coordinate that was never evaluated.
    """
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    candidate = MiningCandidate(
        candidate_id="c1",
        pool="structured_rna",
        accession=float("nan"),  # type: ignore[arg-type]
        locus_start=150,
        locus_end=160,
        score=0.9,
    )
    outcome, _ = classify_candidate(candidate, index)
    assert outcome == OUTCOME_UNMASKABLE


@pytest.mark.parametrize("field", ["locus_start", "locus_end"])
def test_a_nan_coordinate_is_refused_not_mined(field: str) -> None:
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    kwargs: dict = {
        "candidate_id": "c1",
        "pool": "structured_rna",
        "accession": "CP001598.1",
        "locus_start": 150,
        "locus_end": 160,
        "score": 0.9,
    }
    kwargs[field] = float("nan")
    outcome, _ = classify_candidate(MiningCandidate(**kwargs), index)
    assert outcome == OUTCOME_UNMASKABLE


def test_a_fully_coordinated_candidate_off_every_known_locus_is_still_mined() -> None:
    """The guard must not refuse everything — that would be green for the wrong reason.

    Evidence is all-``failed`` so the spare rule does not intercept: with the
    default all-``unavailable`` evidence the candidate is (correctly) *spared*,
    which would make this test pass for a reason unrelated to the coordinate guard.
    """
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    evidence = SpareRuleEvidence(**dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, STATUS_FAILED))
    candidate = MiningCandidate(
        candidate_id="c1",
        pool="structured_rna",
        accession="CP001598.1",
        locus_start=900_000,
        locus_end=900_300,
        score=0.9,
        evidence=evidence,
    )
    assert classify_candidate(candidate, index)[0] == OUTCOME_MINED


def test_a_versioned_accession_masks_through_the_full_classify_path() -> None:
    """End-to-end: the namespace fix must reach ``classify_candidate``, not just the index.

    Without normalisation this candidate is ``mined`` — a known T-box locus
    admitted to the negative pool because its accession carried a ``.1``.
    """
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    evidence = SpareRuleEvidence(**dict.fromkeys(MODEL_INDEPENDENT_DISJUNCTS, STATUS_FAILED))
    candidate = MiningCandidate(
        candidate_id="c1",
        pool="structured_rna",
        accession="CP001598.1",
        locus_start=150,
        locus_end=160,
        score=0.9,
        evidence=evidence,
    )
    assert classify_candidate(candidate, index)[0] == OUTCOME_MASKED


def test_a_control_that_overlaps_but_does_not_reproduce_its_interval_fails() -> None:
    """The binding clause: overlap tolerates a shift of up to the locus length.

    Measured on the real pool, uniform shifts through ±160 nt kept 500/500
    controls both masked and overlapping while the reported natural contamination
    moved 0.184 % → 0.473 %. Exact reproduction has zero tolerance.
    """
    gate = _gate(
        designed_control={
            "n_records": 10,
            "n_masked_at_flank": 10,
            "n_overlapping_at_flank_0": 10,
            "n_exact_interval_match": 9,
        }
    )
    assert gate["overall_pass"] is False
    assert gate["clauses"]["all_controls_overlap"] is True
    assert gate["clauses"]["all_controls_exact"] is False


def test_a_partial_control_draw_fails_rather_than_grading_the_survivors() -> None:
    """``n_controls`` is a contract, not a request.

    ``carve_pool`` drops any row whose geometry does not fit, so 8 requested → 2
    emitted previously graded green on the 2 with the shortfall recorded nowhere.
    """
    gate = mining_pool.control_gate(_summary(), n_controls_requested=25)
    assert gate["overall_pass"] is False
    assert gate["clauses"]["all_requested_controls_carved"] is False
    assert gate["n_controls_requested"] == 25
    assert gate["n_control_windows"] == 10


def test_an_unstated_control_request_is_false_not_absent() -> None:
    """Omitting the request must not make the clause vacuously true."""
    gate = mining_pool.control_gate(_summary())
    assert gate["clauses"]["all_requested_controls_carved"] is False
    assert gate["overall_pass"] is False


def test_controls_present_is_false_when_the_control_block_is_missing() -> None:
    """The clause is redundant with the others, but a dead clause is untestable."""
    gate = mining_pool.control_gate(_summary(designed_control={}), n_controls_requested=0)
    assert gate["clauses"]["controls_present"] is False


def test_exact_interval_match_is_stricter_than_overlap() -> None:
    """Independent closed form for the primitive the gate now rests on."""
    index = masking.LocusIndex.from_records([("CP001598", 1_000, 1_200)])
    assert index.matches_interval_exactly("CP001598.1", 1_000, 1_200) is True
    assert index.matches_interval_exactly("CP001598.1", 1_200, 1_000) is True  # strand folded
    for shifted in ((1_001, 1_201), (999, 1_199), (1_000, 1_201)):
        assert index.is_masked("CP001598.1", *shifted, flank=0) is True
        assert index.matches_interval_exactly("CP001598.1", *shifted) is False
    assert index.matches_interval_exactly(None, 1_000, 1_200) is False
    assert index.matches_interval_exactly("XX999999", 1_000, 1_200) is False


def test_exact_match_uses_source_intervals_not_the_merged_spans() -> None:
    """Merging is lossy: two adjacent loci collapse and neither survives verbatim.

    Measured: 769 of 23,532 locus-centred windows fail an exact match against the
    merged spans and all 23,532 match against the source intervals.
    """
    index = masking.LocusIndex.from_records([("CP001598", 100, 200), ("CP001598", 201, 300)])
    assert index.n_intervals == 1  # merged into [100, 300]
    assert index.matches_interval_exactly("CP001598", 100, 200) is True
    assert index.matches_interval_exactly("CP001598", 201, 300) is True
    assert index.matches_interval_exactly("CP001598", 100, 300) is False


def test_the_namespace_report_publishes_a_normalised_denominator() -> None:
    """Comparing a normalised intersection to a raw count reads as a false shortfall."""
    index = masking.LocusIndex.from_records([("CP001598", 100, 200)])
    report = masking.accession_namespace_report(["CP001598.1", "CP001598.2"], index)
    assert report["n_distinct_accessions"] == 2  # raw
    assert report["n_distinct_normalized"] == 1  # the true denominator
    assert report["n_intersect_normalized"] == 1
    assert report["n_unaddressable_normalized"] == 0


def test_a_dinuc_decoy_records_the_parent_it_permutes() -> None:
    """ADR-0005 A6's leakage-adjacent link; untested on either leg before now."""
    from tbox_finder.decoys import POOL_DINUC, build_corpus_pools

    ids = [f"sha{i}" for i in range(4)]
    records = build_corpus_pools(
        [0.5] * 4,
        [20] * 4,
        ["ACGTACGTACGTACGTACGT"] * 4,
        seed=42,
        n_gc=2,
        n_dinuc_sources=4,
        dinuc_per_source=2,
        record_ids=ids,
    )
    dinuc = [r for r in records if r["pool"] == POOL_DINUC]
    assert len(dinuc) == 8
    for r in dinuc:
        parent_index = int(r["decoy_id"].split("_")[1])
        assert r["source_record_id"] == ids[parent_index]


def test_omitting_record_ids_leaves_the_parent_link_null_rather_than_guessed() -> None:
    from tbox_finder.decoys import POOL_DINUC, build_corpus_pools

    records = build_corpus_pools(
        [0.5] * 4,
        [20] * 4,
        ["ACGTACGTACGTACGTACGT"] * 4,
        seed=42,
        n_gc=2,
        n_dinuc_sources=4,
        dinuc_per_source=1,
    )
    assert all(r["source_record_id"] is None for r in records if r["pool"] == POOL_DINUC)


def test_mismatched_record_ids_are_refused_rather_than_silently_misaligned() -> None:
    from tbox_finder.decoys import build_corpus_pools

    with pytest.raises(ValueError, match="record_ids"):
        build_corpus_pools(
            [0.5] * 4,
            [20] * 4,
            ["ACGTACGTACGTACGTACGT"] * 4,
            seed=42,
            n_gc=2,
            n_dinuc_sources=4,
            dinuc_per_source=1,
            record_ids=["only-one"],
        )
