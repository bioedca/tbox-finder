"""P2-00 — unit tests for the real-flank sourcing + exact-match anchoring module.

Two tiers:

* **bare (stdlib only)** — the pure parse/plan/anchor/status/geometry/digest
  helpers and the fail-closed ``validate_report``. These run in bare CI (no
  biopython, no pandas, no network), which is where the honesty invariants must
  be enforced.
* **committed-artifact** — the measured ``reports/p2/flank_context.json`` must
  itself validate (the ``test_nt_backbone::_PROVENANCE`` precedent).

The validator tests are deliberately *bite* tests: each mutates one field of an
otherwise-valid report and asserts ``validate_report`` complains. The headline
case is :func:`test_validate_rejects_fabricated_anchor_rate` — an all-consistent
report whose ``anchor_rate`` its own counts do not support (the P1-15/P1-16
durable lesson that ``all(clauses)`` cannot catch a clause fabricated TRUE).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tbox_finder.data import flank_context as fc

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_REPORT = _REPO_ROOT / "reports" / "p2" / "flank_context.json"


# --------------------------------------------------------------------------- #
# parse_locus_name
# --------------------------------------------------------------------------- #


def test_parse_plus_strand() -> None:
    assert fc.parse_locus_name("CP002213.1:100-400") == ("CP002213.1", 100, 400, fc.STRAND_PLUS)


def test_parse_minus_strand_normalises_bounds() -> None:
    acc, lo, hi, strand = fc.parse_locus_name("CT573213.2:2420974-2420503")
    assert (acc, lo, hi, strand) == ("CT573213.2", 2420503, 2420974, fc.STRAND_MINUS)
    assert lo <= hi, "bounds must be normalised to forward coordinates"


def test_parse_single_base_locus_is_plus_strand() -> None:
    assert fc.parse_locus_name("X:5-5") == ("X", 5, 5, fc.STRAND_PLUS)


def test_parse_underscored_and_versioned_accessions() -> None:
    assert fc.parse_locus_name("NZ_CP045032.1:1-2")[0] == "NZ_CP045032.1"
    assert fc.parse_locus_name("ACNB01000093.1:9-1")[0] == "ACNB01000093.1"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "garbage",
        "CP002213.1",
        "CP002213.1:100",
        "CP002213.1:abc-400",
        "CP002213.1:100-",
        "CP002213.1:100-400 extra",
        ":100-400",
        "CP002213.1:0-400",
        "CP002213.1:100-0",
    ],
)
def test_parse_rejects_unparseable_names(bad: str) -> None:
    """A name that cannot be parsed returns None — never a guessed coordinate."""
    assert fc.parse_locus_name(bad) is None


# --------------------------------------------------------------------------- #
# plan_region
# --------------------------------------------------------------------------- #


def test_plan_region_applies_pad_both_sides() -> None:
    assert fc.plan_region(2000, 2300, pad_nt=1024) == (976, 3324)


def test_plan_region_clamps_start_at_one() -> None:
    start, stop = fc.plan_region(10, 50, pad_nt=1024)
    assert start == 1, "must not request a coordinate below 1"
    assert stop == 50 + 1024


def test_plan_region_zero_pad_is_identity() -> None:
    assert fc.plan_region(10, 50, pad_nt=0) == (10, 50)


@pytest.mark.parametrize("kwargs", [{"lo": 0, "hi": 5}, {"lo": 10, "hi": 5}])
def test_plan_region_rejects_invalid_bounds(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        fc.plan_region(**kwargs)


def test_plan_region_rejects_negative_pad() -> None:
    with pytest.raises(ValueError):
        fc.plan_region(10, 50, pad_nt=-1)


# --------------------------------------------------------------------------- #
# revcomp
# --------------------------------------------------------------------------- #


def test_revcomp_basic() -> None:
    assert fc.revcomp("ACGT") == "ACGT"
    assert fc.revcomp("AAAC") == "GTTT"
    assert fc.revcomp("N") == "N"


def test_revcomp_is_an_involution() -> None:
    seq = "ACGTTGCANNACGT"
    assert fc.revcomp(fc.revcomp(seq)) == seq


# --------------------------------------------------------------------------- #
# anchor_offset — the load-bearing exact-match anchor
# --------------------------------------------------------------------------- #


def test_anchor_unique_hit_returns_offset() -> None:
    assert fc.anchor_offset("AAACGTAAA", "CGT") == (3, 1)


def test_anchor_zero_hits() -> None:
    assert fc.anchor_offset("AAAA", "CGT") == (-1, 0)


def test_anchor_multi_hit_is_ambiguous() -> None:
    """A locus occurring twice is ambiguous — no offset may be returned."""
    offset, hits = fc.anchor_offset("ACGTACGT", "ACGT")
    assert offset == -1
    assert hits > 1


def test_anchor_counts_overlapping_occurrences() -> None:
    """str.count() would report 2 for 'AA' in 'AAAA'; overlapping ones also count.

    A self-overlapping locus is still ambiguous, so the naive non-overlapping
    count must not be used.
    """
    offset, hits = fc.anchor_offset("AAAA", "AA")
    assert offset == -1
    assert hits > 1


@pytest.mark.parametrize("region,locus", [("", "ACGT"), ("ACGT", ""), ("", "")])
def test_anchor_empty_inputs_fail_closed(region: str, locus: str) -> None:
    assert fc.anchor_offset(region, locus) == (-1, 0)


def test_anchor_locus_at_region_start_and_end() -> None:
    assert fc.anchor_offset("CGTAAA", "CGT") == (0, 1)
    assert fc.anchor_offset("AAACGT", "CGT") == (3, 1)


# --------------------------------------------------------------------------- #
# status_of — the shared builder/validator derivation
# --------------------------------------------------------------------------- #


def test_status_of_all_branches() -> None:
    assert fc.status_of(parsed_ok=False, fetched=False, n_hits=0) == fc.STATUS_BAD_NAME
    assert fc.status_of(parsed_ok=False, fetched=True, n_hits=1) == fc.STATUS_BAD_NAME
    assert fc.status_of(parsed_ok=True, fetched=False, n_hits=0) == fc.STATUS_FETCH_FAILED
    assert fc.status_of(parsed_ok=True, fetched=True, n_hits=0) == fc.STATUS_NOT_FOUND
    assert fc.status_of(parsed_ok=True, fetched=True, n_hits=2) == fc.STATUS_MULTI_HIT
    assert fc.status_of(parsed_ok=True, fetched=True, n_hits=1) == fc.STATUS_OK


def test_status_values_are_exhaustive() -> None:
    assert set(fc.STATUS_VALUES) == {
        fc.STATUS_OK,
        fc.STATUS_BAD_NAME,
        fc.STATUS_FETCH_FAILED,
        fc.STATUS_NOT_FOUND,
        fc.STATUS_MULTI_HIT,
    }


# --------------------------------------------------------------------------- #
# build_context_record
# --------------------------------------------------------------------------- #


def _ok_record(pad: int = 4, **overrides: Any) -> dict[str, Any]:
    locus = "CGTACG"
    region = "A" * pad + locus + "T" * pad
    kwargs: dict[str, Any] = dict(
        record_id="r1",
        locus=locus,
        region=region,
        region_start=100,
        region_stop=200,
        accession="CP1.1",
        strand=fc.STRAND_PLUS,
        parsed_ok=True,
        pad_nt=pad,
    )
    kwargs.update(overrides)
    return fc.build_context_record(**kwargs)


def test_build_record_anchored_geometry() -> None:
    row = _ok_record(pad=4)
    assert row["status"] == fc.STATUS_OK
    assert row["locus_offset"] == 4
    assert row["lead_flank"] == 4
    assert row["trail_flank"] == 4
    assert row["locus_length"] == 6
    assert row["context_seq"] == "AAAACGTACGTTTT"
    assert row["clipped_start"] is False
    assert row["clipped_end"] is False


def test_build_record_geometry_is_internally_consistent() -> None:
    row = _ok_record(pad=7)
    seq = row["context_seq"]
    off, ln = row["locus_offset"], row["locus_length"]
    assert row["lead_flank"] + ln + row["trail_flank"] == len(seq)
    assert seq[off : off + ln] == "CGTACG", "offset must actually index the locus"


def test_build_record_flags_clipping_when_flank_short() -> None:
    """A side shorter than the requested pad means the region ran off a contig end."""
    row = fc.build_context_record(
        record_id="r",
        locus="CGT",
        region="A" + "CGT" + "T" * 50,
        region_start=1,
        region_stop=54,
        accession="X",
        strand=fc.STRAND_PLUS,
        parsed_ok=True,
        pad_nt=10,
    )
    assert row["status"] == fc.STATUS_OK
    assert row["clipped_start"] is True, "lead flank 1 < pad 10"
    assert row["clipped_end"] is False, "trail flank 50 >= pad 10"


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"parsed_ok": False}, fc.STATUS_BAD_NAME),
        ({"region": None}, fc.STATUS_FETCH_FAILED),
        ({"region": "TTTTTTTT"}, fc.STATUS_NOT_FOUND),
        ({"region": "CGTACGCGTACG"}, fc.STATUS_MULTI_HIT),
    ],
)
def test_build_record_non_anchored_statuses(overrides: dict[str, Any], expected: str) -> None:
    row = _ok_record(**overrides)
    assert row["status"] == expected


@pytest.mark.parametrize(
    "overrides",
    [{"parsed_ok": False}, {"region": None}, {"region": "TTTTTTTT"}, {"region": "CGTACGCGTACG"}],
)
def test_build_record_non_anchored_carries_no_geometry(overrides: dict[str, Any]) -> None:
    """A non-anchored row must not carry offsets that could be mistaken for real."""
    row = _ok_record(**overrides)
    assert row["locus_offset"] == -1
    assert row["lead_flank"] == -1
    assert row["trail_flank"] == -1
    assert row["context_seq"] == ""


def test_build_record_bad_name_beats_a_present_region() -> None:
    """parsed_ok=False dominates: we cannot trust a region fetched for a bad name."""
    assert _ok_record(parsed_ok=False)["status"] == fc.STATUS_BAD_NAME


# --------------------------------------------------------------------------- #
# derive_counts / derive_anchor_rate
# --------------------------------------------------------------------------- #


def _rows(**by_status: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for status, n in by_status.items():
        for i in range(n):
            out.append(
                {
                    "record_id": f"{status}-{i}",
                    "accession": "X",
                    "strand": 1,
                    "locus_offset": 0 if status == fc.STATUS_OK else -1,
                    "locus_length": 3,
                    "lead_flank": 0,
                    "trail_flank": 0,
                    "clipped_start": False,
                    "clipped_end": False,
                    "status": status,
                    "pad_nt": 1024,
                    "context_seq": "ACG" if status == fc.STATUS_OK else "",
                }
            )
    return out


def test_derive_counts_covers_every_status_key() -> None:
    counts = fc.derive_counts(_rows(ok=2, not_found=1))
    assert counts == {"ok": 2, "bad_name": 0, "fetch_failed": 0, "not_found": 1, "multi_hit": 0}


def test_derive_counts_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        fc.derive_counts([{"status": "wat"}])


def test_derive_anchor_rate() -> None:
    assert fc.derive_anchor_rate({"ok": 3, "not_found": 1}) == 0.75
    assert fc.derive_anchor_rate({"ok": 0, "not_found": 0}) == 0.0, "empty must not divide by zero"


# --------------------------------------------------------------------------- #
# context_digest
# --------------------------------------------------------------------------- #


def test_digest_is_row_order_independent() -> None:
    rows = _rows(ok=3, not_found=2)
    assert fc.context_digest(rows) == fc.context_digest(list(reversed(rows)))


def test_digest_changes_when_a_context_sequence_changes() -> None:
    rows = _rows(ok=2)
    before = fc.context_digest(rows)
    rows[0]["context_seq"] = "TTT"
    assert fc.context_digest(rows) != before


def test_digest_changes_when_an_offset_changes() -> None:
    rows = _rows(ok=2)
    before = fc.context_digest(rows)
    rows[0]["locus_offset"] = 99
    assert fc.context_digest(rows) != before


# --------------------------------------------------------------------------- #
# validate_report — fail-closed bite tests
# --------------------------------------------------------------------------- #


def _valid_report() -> dict[str, Any]:
    rows = _rows(ok=98, not_found=2)
    return fc.build_report(
        rows, window_nt=1024, accessed="2026-07-16", per_source={"s": {"ok": 98}}
    )


def test_valid_report_validates() -> None:
    assert fc.validate_report(_valid_report()) == []


def test_build_report_derives_rate_from_counts() -> None:
    report = _valid_report()
    assert report["n_anchored"] == 98
    assert report["anchor_rate"] == 0.98
    assert report["status_counts"]["not_found"] == 2


def test_build_report_derives_pad_from_the_rows_not_the_caller() -> None:
    """pad_nt must be EVIDENCE: build_report has no pad parameter to fabricate one."""
    import inspect

    assert "pad_nt" not in inspect.signature(fc.build_report).parameters
    assert _valid_report()["pad_nt"] == 1024


def test_build_report_refuses_rows_that_disagree_on_pad() -> None:
    rows = _rows(ok=98, not_found=2)
    rows[0]["pad_nt"] = 2048
    with pytest.raises(ValueError, match="disagree on pad_nt"):
        fc.build_report(rows, window_nt=1024, accessed="2026-07-16")


def test_build_report_refuses_rows_with_no_pad_stamp() -> None:
    """A legacy cache row carries no pad — the pad must not be guessed."""
    rows = _rows(ok=98, not_found=2)
    del rows[0]["pad_nt"]
    with pytest.raises(ValueError, match="pad_nt stamp"):
        fc.build_report(rows, window_nt=1024, accessed="2026-07-16")


def test_validate_rejects_non_mapping() -> None:
    assert fc.validate_report(["not", "a", "mapping"]) != []


def test_validate_rejects_fabricated_anchor_rate() -> None:
    """THE headline bite: a self-consistent-looking report whose own counts refute it.

    ``overall_pass = all(clauses)`` catches a clause flipped FALSE but structurally
    cannot catch one fabricated TRUE, so the rate is re-derived from the evidence.
    """
    report = _valid_report()
    report["anchor_rate"] = 1.0  # counts say 98/100
    problems = fc.validate_report(report)
    assert any("anchor_rate" in p and "re-derived" in p for p in problems), problems


def test_validate_rejects_fabricated_n_anchored() -> None:
    report = _valid_report()
    report["n_anchored"] = 100  # status_counts["ok"] == 98
    assert any("n_anchored" in p for p in fc.validate_report(report))


def test_validate_rejects_n_records_not_matching_counts() -> None:
    report = _valid_report()
    report["n_records"] = 99
    assert any("n_records" in p for p in fc.validate_report(report))


def test_validate_rejects_bool_as_count() -> None:
    """isinstance(True, int) is True in Python — a bool must not pass as a count."""
    report = _valid_report()
    report["status_counts"]["ok"] = True
    assert any(
        "status_counts['ok']" in p or "status_counts" in p for p in fc.validate_report(report)
    )


def test_validate_rejects_bool_as_n_records() -> None:
    report = _valid_report()
    report["n_records"] = True
    assert any("n_records" in p for p in fc.validate_report(report))


def test_validate_rejects_wrong_schema_version() -> None:
    report = _valid_report()
    report["schema_version"] = "0.9"
    assert any("schema_version" in p for p in fc.validate_report(report))


def test_validate_rejects_missing_status_counts() -> None:
    report = _valid_report()
    del report["status_counts"]
    assert any("status_counts" in p for p in fc.validate_report(report))


@pytest.mark.parametrize("bad_window", ["abc", [1024], {"a": 1}, None, 3.5])
def test_validate_never_raises_on_a_malformed_window_nt(bad_window: Any) -> None:
    """The validator is the fail-closed gate: it must RETURN problems, never raise.

    A gate that crashes on malformed evidence yields a traceback that reads as an
    infrastructure fault rather than a rejected report.
    """
    report = _valid_report()
    report["window_nt"] = bad_window
    problems = fc.validate_report(report)  # must not raise
    assert any("window_nt" in p for p in problems), problems


def test_validate_rejects_a_degraded_run() -> None:
    """A broken run must not certify itself — it routes to a §7 stop-and-ask."""
    report = fc.build_report(_rows(ok=50, not_found=50), window_nt=1024, accessed="2026-07-16")
    assert any("MIN_ANCHOR_RATE" in p for p in fc.validate_report(report))


def test_validate_rejects_an_infrastructure_failure() -> None:
    """An all-fetch_failed run is an outage, not a data property — never certified."""
    report = fc.build_report(_rows(ok=0, fetch_failed=100), window_nt=1024, accessed="2026-07-16")
    problems = fc.validate_report(report)
    assert any("MAX_FETCH_FAILED_RATE" in p for p in problems), problems


def test_validate_rejects_pad_smaller_than_window() -> None:
    """pad < window ⇒ the locus cannot reach every window phase (PRD §6)."""
    report = _valid_report()
    report["pad_nt"] = 256
    assert any("pad_nt" in p for p in fc.validate_report(report))


def test_validate_rejects_claimed_synthetic_flank() -> None:
    """§10.3: this module sources real flank; it may never certify a synthetic one."""
    report = _valid_report()
    report["synthetic_flank"] = True
    assert any("synthetic_flank" in p for p in fc.validate_report(report))


def test_validate_rejects_claimed_trusted_coordinates() -> None:
    report = _valid_report()
    report["coordinates_trusted"] = True
    assert any("coordinates_trusted" in p for p in fc.validate_report(report))


@pytest.mark.parametrize("field", ["db", "base_url", "accessed"])
def test_validate_requires_source_provenance(field: str) -> None:
    report = _valid_report()
    del report["source"][field]
    assert any(field in p for p in fc.validate_report(report))


def test_validate_rejects_missing_source_block() -> None:
    report = _valid_report()
    del report["source"]
    assert any("source" in p for p in fc.validate_report(report))


# --------------------------------------------------------------------------- #
# Contract pins
# --------------------------------------------------------------------------- #


def test_pad_is_at_least_the_window() -> None:
    """PRD §6: a locus must be placeable at ANY phase of the window."""
    assert fc.DEFAULT_PAD_NT >= fc.DEFAULT_WINDOW_NT


def test_window_matches_the_prd_pin() -> None:
    assert fc.DEFAULT_WINDOW_NT == 1024


def test_context_cols_are_stable() -> None:
    assert fc.CONTEXT_COLS[0] == "record_id"
    assert "context_seq" in fc.CONTEXT_COLS
    assert "status" in fc.CONTEXT_COLS


# --------------------------------------------------------------------------- #
# _should_retry — transient vs deterministic outcomes
# --------------------------------------------------------------------------- #


def test_only_transient_fetch_failure_is_retried() -> None:
    """A network blip must not be baked into the artifact as an anchor failure."""
    row = {"status": fc.STATUS_FETCH_FAILED, "pad_nt": 1024}
    assert fc._should_retry(row, retry_failed=True, pad_nt=1024) is True


@pytest.mark.parametrize(
    "status", [fc.STATUS_OK, fc.STATUS_NOT_FOUND, fc.STATUS_MULTI_HIT, fc.STATUS_BAD_NAME]
)
def test_deterministic_outcomes_are_never_retried(status: str) -> None:
    """These are properties of the record — re-fetching burns quota for the same answer."""
    row = {"status": status, "pad_nt": 1024}
    assert fc._should_retry(row, retry_failed=True, pad_nt=1024) is False


def test_retry_can_be_disabled() -> None:
    row = {"status": fc.STATUS_FETCH_FAILED, "pad_nt": 1024}
    assert fc._should_retry(row, retry_failed=False, pad_nt=1024) is False


def test_row_fetched_at_a_different_pad_is_always_refetched() -> None:
    """Every geometry field is a function of pad — reuse across pads is a wrong answer."""
    row = {"status": fc.STATUS_OK, "pad_nt": 1024}
    assert fc._should_retry(row, retry_failed=False, pad_nt=2048) is True
    assert fc._should_retry(row, retry_failed=True, pad_nt=2048) is True


def test_legacy_row_without_a_pad_stamp_is_refetched() -> None:
    """A pre-stamp cache row cannot be trusted at any pad — the safe direction."""
    assert fc._should_retry({"status": fc.STATUS_OK}, retry_failed=False, pad_nt=1024) is True


def test_should_retry_tolerates_a_status_less_row() -> None:
    """Total: a malformed row is re-fetched, not a crash."""
    assert fc._should_retry({}, retry_failed=True, pad_nt=1024) is True


# --------------------------------------------------------------------------- #
# RateLimiter / workers_for — the NCBI courtesy ceiling
# --------------------------------------------------------------------------- #


def test_ncbi_rate_ceilings_are_the_published_literals() -> None:
    """Pin the LITERAL published ceilings — NOT the module constants.

    ``assert workers_for(None) == int(RATE_LIMIT_NO_KEY)`` is a TAUTOLOGY: it
    compares the function to the very constant that produces it, so it passes at
    any value. That tautology let RATE_LIMIT_NO_KEY be silently changed to 100.0
    (a 33x overrun of NCBI's anonymous ceiling) while the suite stayed green.
    Hard-code what NCBI actually publishes so a bad constant cannot pass.
    """
    assert fc.RATE_LIMIT_NO_KEY == 3.0, "NCBI anonymous ceiling is 3 req/s"
    assert fc.RATE_LIMIT_WITH_KEY == 10.0, "NCBI api_key ceiling is 10 req/s"


def test_workers_match_the_applicable_ceiling() -> None:
    assert fc.workers_for(None) == 3
    assert fc.workers_for("some-key") == 10
    assert fc.workers_for(None) < fc.workers_for("some-key")


def test_rate_limiter_rejects_non_positive_rate() -> None:
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            fc.RateLimiter(bad)


def test_rate_limiter_spaces_serial_acquisitions() -> None:
    limiter = fc.RateLimiter(50.0)  # 20 ms apart — fast enough to keep the test cheap
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    elapsed = time.monotonic() - start
    # 5 acquisitions = 4 enforced gaps of 20 ms (the first is free).
    assert elapsed >= 4 * 0.02 * 0.9, f"limiter did not space acquisitions (elapsed {elapsed})"


def test_rate_limiter_holds_the_ceiling_under_concurrency() -> None:
    """The NCBI ceiling is a GLOBAL request rate, so threads must share the limiter."""
    rate = 50.0
    n = 20
    limiter = fc.RateLimiter(rate)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: limiter.acquire(), range(n)))
    elapsed = time.monotonic() - start
    observed = n / elapsed
    assert observed <= rate * 1.35, f"concurrent rate {observed:.1f}/s exceeded ceiling {rate}/s"


# --------------------------------------------------------------------------- #
# Committed artifact
# --------------------------------------------------------------------------- #


def test_committed_report_exists_and_validates() -> None:
    """The measured artifact must satisfy its own fail-closed validator."""
    assert _COMMITTED_REPORT.is_file(), f"missing measured report: {_COMMITTED_REPORT}"
    report = json.loads(_COMMITTED_REPORT.read_text(encoding="utf-8"))
    assert fc.validate_report(report) == []


def test_committed_report_is_the_full_corpus() -> None:
    report = json.loads(_COMMITTED_REPORT.read_text(encoding="utf-8"))
    assert report["n_records"] == 23535, "P2-00 covers every corpus record"
    assert report["step"] == "P2-00"
