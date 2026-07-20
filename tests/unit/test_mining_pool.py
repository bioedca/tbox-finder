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
    """The measured P2-10b defect: 0 of 2,751 Rfam accessions intersected as-is.

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


def test_an_unanchored_row_yields_no_window() -> None:
    """Non-``ok`` rows carry sentinel offsets, not coordinates."""
    for status in ("multi_hit", "unavailable", "bad_name", "not_found"):
        assert (
            mining_pool.carve_window(
                _context_row(status=status, locus_offset=-1), "lead", window_nt=300, margin_nt=50
            )
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


# --------------------------------------------------------------------------- #
# The non-vacuity gate
# --------------------------------------------------------------------------- #
def _summary(**kw: object) -> dict:
    base: dict = {
        "designed_control": {
            "n_records": 10,
            "n_masked_at_flank": 10,
            "n_overlapping_at_flank_0": 10,
        },
        "natural": {"n_records": 100, "n_masked_at_flank": 1},
        "accession_namespace": {"namespace_compatible": True},
    }
    base.update(kw)
    return base


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
    assert mining_pool.control_gate(summary)["overall_pass"] is False


def test_the_gate_passes_when_every_control_masks() -> None:
    assert mining_pool.control_gate(_summary())["overall_pass"] is True


def test_one_unmasked_control_fails_the_gate() -> None:
    """Controls overlap a known locus by construction: 9/10 is a frame defect."""
    gate = mining_pool.control_gate(
        _summary(
            designed_control={
                "n_records": 10,
                "n_masked_at_flank": 9,
                "n_overlapping_at_flank_0": 9,
            }
        )
    )
    assert gate["overall_pass"] is False
    assert gate["clauses"]["all_controls_masked"] is False


def test_a_control_that_only_masks_within_the_flank_fails_the_gate() -> None:
    """The sub-flank frame error: 100% "masked" while 0% actually overlap.

    A uniform frame shift smaller than ``flank_nt`` keeps every control inside the
    flank and so scores a perfect ``n_masked_at_flank``. Gating on the flank count
    alone would pass it, with the refuting number sitting unread in the same dict.
    """
    gate = mining_pool.control_gate(
        _summary(
            designed_control={
                "n_records": 10,
                "n_masked_at_flank": 10,
                "n_overlapping_at_flank_0": 0,
            }
        )
    )
    assert gate["overall_pass"] is False
    assert gate["clauses"]["all_controls_masked"] is True
    assert gate["clauses"]["all_controls_overlap"] is False


def test_a_zero_natural_rate_does_not_fail_the_gate() -> None:
    """No natural contamination is a legitimate result; a dead control is not."""
    gate = mining_pool.control_gate(_summary(natural={"n_records": 100, "n_masked_at_flank": 0}))
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
    assert mining_pool.control_gate(_summary(**missing))["overall_pass"] is False


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
