"""P2-00 — real flanking genomic context for the Stage-1 training corpus.

PRD §6 pins Stage-1 training positives as **offset-augmented random window-phase
placements in a 1024-nt window** (stride 512). The corpus cannot supply that
geometry on its own: ``master_clean_v0.parquet`` stores each locus as
``FASTA_sequence`` — dense genomic DNA of only **104–550 nt** (median 281), whose
*native* flank is negligible (median 0 nt leading / 50 nt trailing; only 5 of
23,535 records reach 1024 nt). Filling the remainder synthetically is rejected:
the §9.1 GC-matched pool is ``gc_length_matched_0th_order`` (i.i.d. sequence), and
splicing a real locus into compositionally-flat background would hand the model a
trivially separable boundary cue — inflating GATE-4 *and* the GATE-1
generalization number, the exact artefact PRD §5 exists to defeat.

This module therefore **sources the real flank** from NCBI ``nuccore`` (PRD §10.2:
find publicly-available defensible data rather than fabricate), keyed on the
``Name`` column, which parses as ``accession:start-end`` for 100 % of records
(11,633 distinct accessions).

Why the coordinates are **not** trusted (the load-bearing design point):

* ``Name``'s coordinate span reconciles with ``len(FASTA_sequence)`` for only
  **61.7 %** of records. The disagreement is concentrated in
  ``RF00230_master.fa`` (5.9 % agreement) and is **always** ``span > seqlen``, by
  a bounded 1–45 nt (median 26) — i.e. the stored sequence is a *sub-extent* of
  the named span, never longer.
* So a region is fetched with generous padding and the stored ``FASTA_sequence``
  is located inside it by **exact string match**. A record is *anchored* iff its
  sequence occurs **exactly once** in the padded region. This is self-verifying
  (it does not rely on the suspect coordinates at all) and **fails closed** —
  zero hits or multiple hits are recorded as a non-anchored status with a reason,
  never guessed at (CLAUDE.md §10.3).

The anchor rate is **measured and reported**, never assumed; a material shortfall
is a CLAUDE.md §7 stop-and-ask, not something this module absorbs silently.

Padding: ``DEFAULT_PAD_NT = 1024`` ≥ the window, so a locus of any corpus length
can be placed at **any** phase of a 1024-nt window (true window-phase offset
augmentation, PRD §6). Records near a contig end yield short flank — recorded via
``clipped_start``/``clipped_end`` so the P2-01 datamodule can zero-flank and flag
them exactly as PRD §6 pins for contig ends.

Outputs (imp.md P2-00):
    data/interim/flank_context/context_v0.parquet    one row per corpus record
    data/interim/flank_context/fetch_cache.jsonl     resumable fetch log
    data/interim/flank_context/context_v0.provenance.json
    data/external/ncbi_flank/provenance.json         §10.2 source provenance
    reports/p2/flank_context.json                    measured anchor audit (git)

The heavy imports (Bio.Entrez / pandas) are lazy; the pure parse/anchor/plan/
digest/validate helpers import bare so the unit tests run in bare CI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from tbox_finder import ingest, provenance
from tbox_finder.labels import NAME_COL, WINDOW_SEQ_COL

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #

#: report schema version (bumped on any breaking field change).
SCHEMA_VERSION = "1.0"

#: ``Name`` parses as ``<accession>:<start>-<end>`` (1-based inclusive; end < start
#: ⇒ the locus is on the minus strand). Verified on 23,535/23,535 records.
NAME_RE = re.compile(r"^(\S+):(\d+)-(\d+)$")

#: NCBI E-utilities parameters (PRD §10.2 provenance).
NCBI_DB = "nuccore"
NCBI_RETTYPE = "fasta"
NCBI_RETMODE = "text"
NCBI_TOOL = "tbox-finder"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
#: NCBI GenBank records carry no single blanket licence; usage terms:
NCBI_TERMS_URL = "https://www.ncbi.nlm.nih.gov/home/about/policies/"

#: NCBI courtesy rate limits — 3 req/s anonymous, 10 req/s with an API key.
#: These are an external service's published ceiling, so they are frozen here in
#: CODE with no config override (the ``coverage.py::OOD_ECE_MIN_N`` precedent) —
#: nothing downstream may raise them. Their unit tests pin the LITERAL numbers
#: rather than these constants: a test asserting ``workers_for(None) ==
#: int(RATE_LIMIT_NO_KEY)`` is a tautology that passes at any value.
RATE_LIMIT_NO_KEY = 3.0
RATE_LIMIT_WITH_KEY = 10.0

#: Socket timeout. Bio.Entrez 1.87 calls ``urlopen(request)`` with **no** timeout
#: and exposes no per-call hook, so a stalled connection would hang a multi-hour
#: run forever with no diagnosis. Applied process-wide via ``socket``.
NCBI_TIMEOUT_S = 60.0

#: Operational sanity floors (frozen in code — no config may weaken them).
#: NOT scientific thresholds: no PRD/ADR pins an anchor rate, so these are
#: implementer choices (the P1-16 ``config.pinned=false`` precedent) that exist
#: only to stop a *broken run* certifying itself. A breach writes nothing and
#: routes to a CLAUDE.md §7 stop-and-ask — never a silently-degraded artifact.
MIN_ANCHOR_RATE = 0.95
MAX_FETCH_FAILED_RATE = 0.01

#: PRD §6 scan/train window geometry (echoed; the pin lives in
#: conf/model/caduceus_stage1.yaml + continued_pretrain.DEFAULT_WINDOW_NT).
DEFAULT_WINDOW_NT = 1_024

#: flank padding per side. >= DEFAULT_WINDOW_NT so a locus of ANY corpus length
#: (max 550 nt) can sit at any phase of a 1024-nt window (PRD §6 offset aug).
DEFAULT_PAD_NT = 1_024

#: strand encoding used by NCBI efetch (seq_start/seq_stop stay in forward coords).
STRAND_PLUS = 1
STRAND_MINUS = 2

#: per-record anchoring outcomes. Only ``STATUS_OK`` rows are trainable substrate.
STATUS_OK = "ok"
STATUS_BAD_NAME = "bad_name"
STATUS_FETCH_FAILED = "fetch_failed"
STATUS_NOT_FOUND = "not_found"
STATUS_MULTI_HIT = "multi_hit"
STATUS_VALUES: tuple[str, ...] = (
    STATUS_OK,
    STATUS_BAD_NAME,
    STATUS_FETCH_FAILED,
    STATUS_NOT_FOUND,
    STATUS_MULTI_HIT,
)

#: split-table columns (the git-LFS committed table is the identity anchor).
SPLIT_CORPUS_SHA_COL = "corpus_record_sha256"
SPLIT_SOURCE_COL = "source"
POSITIVE_SOURCE = "corpus"

#: master passthrough columns.
MASTER_SOURCE_COL = "source"

#: default artifact locations.
DEFAULT_MASTER = Path("data/processed/master_clean_v0.parquet")
DEFAULT_SPLIT_TABLE = Path("data/processed/splits/split_assignments.parquet")
DEFAULT_OUT_DIR = Path("data/interim/flank_context")
DEFAULT_EXTERNAL_DIR = Path("data/external/ncbi_flank")
DEFAULT_REPORT = Path("reports/p2/flank_context.json")
CONTEXT_NAME = "context_v0.parquet"
CACHE_NAME = "fetch_cache.jsonl"
PROVENANCE_NAME = "context_v0.provenance.json"

#: columns written to context_v0.parquet.
CONTEXT_COLS: tuple[str, ...] = (
    "record_id",
    "accession",
    "strand",
    "region_start",
    "region_stop",
    "locus_offset",
    "locus_length",
    "lead_flank",
    "trail_flank",
    "clipped_start",
    "clipped_end",
    "status",
    "pad_nt",
    "context_seq",
)

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


# --------------------------------------------------------------------------- #
# Pure helpers (bare-CI importable — no biopython, no pandas)
# --------------------------------------------------------------------------- #


def revcomp(seq: str) -> str:
    """Reverse-complement an uppercase ACGTN string."""
    return seq.translate(_COMPLEMENT)[::-1]


def parse_locus_name(name: str) -> tuple[str, int, int, int] | None:
    """Parse ``accession:start-end`` → ``(accession, lo, hi, strand)``.

    ``lo``/``hi`` are forward-coordinate 1-based inclusive bounds (``lo <= hi``);
    ``strand`` is :data:`STRAND_MINUS` when the name's end precedes its start.
    Returns ``None`` on any unparseable name (the caller records
    :data:`STATUS_BAD_NAME` — never a guess).
    """
    match = NAME_RE.match(name.strip()) if name else None
    if match is None:
        return None
    accession, raw_a, raw_b = match.group(1), match.group(2), match.group(3)
    a, b = int(raw_a), int(raw_b)
    if a <= 0 or b <= 0:
        return None
    strand = STRAND_MINUS if b < a else STRAND_PLUS
    return accession, min(a, b), max(a, b), strand


def plan_region(lo: int, hi: int, *, pad_nt: int = DEFAULT_PAD_NT) -> tuple[int, int]:
    """Return the 1-based inclusive ``(start, stop)`` region to fetch.

    Padding is applied in **forward** coordinates on both sides (NCBI's
    ``seq_start``/``seq_stop`` are always ascending regardless of ``strand``).
    The start clamps at 1; the stop has no upper clamp here because the contig
    length is unknown until the fetch returns — a short return is detected
    downstream as clipping, not as an error.
    """
    if pad_nt < 0:
        raise ValueError(f"pad_nt must be >= 0, got {pad_nt}")
    if lo < 1 or hi < lo:
        raise ValueError(f"invalid locus bounds: lo={lo} hi={hi}")
    return max(1, lo - pad_nt), hi + pad_nt


def anchor_offset(region: str, locus: str) -> tuple[int, int]:
    """Locate ``locus`` inside ``region`` by exact match → ``(offset, n_hits)``.

    ``offset`` is the 0-based index of the single occurrence, or ``-1`` when the
    match is not unique. **Fails closed**: an empty locus, an empty region, zero
    hits, or >1 hits all return a non-unique result for the caller to record with
    a reason. Overlapping occurrences are counted (``str.count`` would miss them),
    because a self-overlapping locus is still ambiguous.
    """
    if not region or not locus:
        return -1, 0
    hits = 0
    first = -1
    start = 0
    while True:
        idx = region.find(locus, start)
        if idx < 0:
            break
        if hits == 0:
            first = idx
        hits += 1
        if hits > 1:
            return -1, hits
        start = idx + 1
    return (first, hits) if hits == 1 else (-1, hits)


def status_of(*, parsed_ok: bool, fetched: bool, n_hits: int) -> str:
    """Derive the per-record anchoring status from evidence.

    Shared by the builder and the validator so a status can never be *asserted*
    independently of the evidence that produced it (the P1-15/P1-16 durable
    lesson: ``overall_pass = all(clauses)`` cannot catch a clause fabricated
    TRUE, so every clause is re-derived from its recorded evidence).
    """
    if not parsed_ok:
        return STATUS_BAD_NAME
    if not fetched:
        return STATUS_FETCH_FAILED
    if n_hits == 0:
        return STATUS_NOT_FOUND
    if n_hits > 1:
        return STATUS_MULTI_HIT
    return STATUS_OK


def build_context_record(
    *,
    record_id: str,
    locus: str,
    region: str | None,
    region_start: int,
    region_stop: int,
    accession: str,
    strand: int,
    parsed_ok: bool,
    pad_nt: int = DEFAULT_PAD_NT,
) -> dict[str, Any]:
    """Assemble one context row from fetch evidence (pure; no I/O).

    The returned ``status`` is derived via :func:`status_of`, and the geometry
    fields are populated **only** on the anchored path — a non-anchored row
    carries no offsets to be mistaken for real ones.
    """
    offset, n_hits = anchor_offset(region or "", locus)
    status = status_of(parsed_ok=parsed_ok, fetched=region is not None, n_hits=n_hits)
    row: dict[str, Any] = {
        "record_id": record_id,
        "accession": accession,
        "strand": int(strand),
        "region_start": int(region_start),
        "region_stop": int(region_stop),
        "locus_offset": -1,
        "locus_length": len(locus),
        "lead_flank": -1,
        "trail_flank": -1,
        "clipped_start": False,
        "clipped_end": False,
        "status": status,
        "pad_nt": int(pad_nt),
        "context_seq": "",
    }
    if status != STATUS_OK or region is None:
        return row
    trail = len(region) - offset - len(locus)
    row.update(
        locus_offset=offset,
        lead_flank=offset,
        trail_flank=trail,
        # A side shorter than the requested padding means the region ran off a
        # contig end (PRD §6: contig ends are zero-flanked and flagged).
        clipped_start=offset < pad_nt,
        clipped_end=trail < pad_nt,
        context_seq=region,
    )
    return row


def context_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Golden digest over the anchored context geometry (row-order independent).

    Hashes identity + geometry + the context sequence, sorted by ``record_id`` —
    the ``seg_smoke.smoke_digest`` / ``labels.labels_digest`` precedent.
    """
    ordered = sorted(records, key=lambda r: str(r["record_id"]))
    hashes = [
        ingest.record_hash(
            [
                str(r["record_id"]),
                str(r["accession"]),
                int(r["strand"]),
                int(r["locus_offset"]),
                int(r["locus_length"]),
                str(r["status"]),
                str(r["context_seq"]),
            ]
        )
        for r in ordered
    ]
    return ingest.records_digest(hashes)


# --------------------------------------------------------------------------- #
# Measured-report derivation + fail-closed validator
# --------------------------------------------------------------------------- #


def derive_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Per-status counts, re-derived from the rows (builder/validator shared)."""
    counts = dict.fromkeys(STATUS_VALUES, 0)
    for row in records:
        status = str(row.get("status", ""))
        if status not in counts:
            raise ValueError(f"unknown status {status!r} (allowed: {STATUS_VALUES})")
        counts[status] += 1
    return counts


def derive_pad_nt(records: Sequence[Mapping[str, Any]]) -> int:
    """Re-derive the fetch padding **from the rows**, not from a caller's claim.

    Every geometry field is a function of the pad the region was fetched at, so
    the report's ``pad_nt`` must come from the evidence. A caller-supplied pad was
    the module's one *asserted* clause — precisely the shape the P1-15/P1-16
    lesson warns about (``all(clauses)`` cannot catch a clause fabricated TRUE),
    since a re-run at a different ``--pad-nt`` would reuse cached rows and still
    stamp the new pad on the report. Raises if the rows are missing the stamp or
    disagree — never guesses.
    """
    if not records:
        raise ValueError("cannot derive pad_nt from zero records")
    missing = sum(1 for r in records if not _is_int(r.get("pad_nt")))
    if missing:
        raise ValueError(f"{missing} record(s) carry no int pad_nt stamp — cannot derive pad")
    pads = {int(r["pad_nt"]) for r in records}
    if len(pads) > 1:
        raise ValueError(f"records disagree on pad_nt: {sorted(pads)} — refusing to certify one")
    return pads.pop()


def derive_anchor_rate(counts: Mapping[str, int]) -> float:
    """Anchored fraction, re-derived from the per-status counts."""
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 0.0
    return int(counts[STATUS_OK]) / total


def _is_int(value: Any) -> bool:
    """True iff ``value`` is a real integer — a bool is NOT a count.

    ``isinstance(True, int)`` is True in Python, which let a bool masquerade as a
    count in P1-16 (CodeRabbit r1). Reject bools explicitly.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of schema/consistency problems (empty ⇒ valid). Never raises.

    Fails **closed**: every headline number is **re-derived** from the report's own
    recorded evidence (the per-status counts) rather than trusted, so a report
    cannot certify an anchor rate its counts do not support.
    """
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version != {SCHEMA_VERSION!r} (got {report.get('schema_version')!r})"
        )

    for key in ("pad_nt", "window_nt", "n_records"):
        if not _is_int(report.get(key)):
            problems.append(f"{key} is not an int (got {report.get(key)!r})")

    if (
        _is_int(report.get("pad_nt"))
        and _is_int(report.get("window_nt"))
        and int(report["pad_nt"]) < int(report["window_nt"])
    ):
        problems.append(
            "pad_nt < window_nt — a locus cannot then be placed at every window phase "
            "(PRD §6 offset augmentation)"
        )

    counts = report.get("status_counts")
    if not isinstance(counts, Mapping):
        problems.append("status_counts block missing or not a mapping")
        return problems
    for status in STATUS_VALUES:
        if not _is_int(counts.get(status)):
            problems.append(f"status_counts[{status!r}] is not an int (got {counts.get(status)!r})")
    if problems:
        return problems

    total = sum(int(counts[s]) for s in STATUS_VALUES)
    if _is_int(report.get("n_records")) and int(report["n_records"]) != total:
        problems.append(f"n_records {report['n_records']} != sum(status_counts) {total}")

    if not _is_int(report.get("n_anchored")):
        problems.append(f"n_anchored is not an int (got {report.get('n_anchored')!r})")
    elif int(report["n_anchored"]) != int(counts[STATUS_OK]):
        problems.append(
            f"n_anchored {report['n_anchored']} != status_counts['{STATUS_OK}'] {counts[STATUS_OK]}"
        )

    reported = report.get("anchor_rate")
    if not isinstance(reported, (int, float)) or isinstance(reported, bool):
        problems.append(f"anchor_rate is not a number (got {reported!r})")
    else:
        expected = derive_anchor_rate({s: int(counts[s]) for s in STATUS_VALUES})
        if abs(float(reported) - expected) > 1e-9:
            problems.append(f"anchor_rate {reported} != re-derived {expected}")

    # Operational floors: a broken run must not certify itself (§10.3).
    if (
        isinstance(reported, (int, float))
        and not isinstance(reported, bool)
        and float(reported) < MIN_ANCHOR_RATE
    ):
        problems.append(
            f"anchor_rate {float(reported):.4f} < MIN_ANCHOR_RATE {MIN_ANCHOR_RATE} — "
            "a run this degraded is a CLAUDE.md §7 stop-and-ask, not an artifact"
        )
    total_for_rate = sum(int(counts[s_]) for s_ in STATUS_VALUES)
    if total_for_rate > 0:
        failed_rate = int(counts[STATUS_FETCH_FAILED]) / total_for_rate
        if failed_rate > MAX_FETCH_FAILED_RATE:
            problems.append(
                f"fetch_failed rate {failed_rate:.4f} > MAX_FETCH_FAILED_RATE "
                f"{MAX_FETCH_FAILED_RATE} — infrastructure failure, not a data property"
            )

    # Honesty flags — this module sources real data; it may never claim otherwise.
    if report.get("synthetic_flank") is not False:
        problems.append("synthetic_flank must be False — flank is sourced, never generated (§10.3)")
    if report.get("coordinates_trusted") is not False:
        problems.append(
            "coordinates_trusted must be False — anchoring is by exact sequence match, "
            "the Name span disagrees with the stored sequence on 38.3% of records"
        )

    src = report.get("source")
    if not isinstance(src, Mapping):
        problems.append("source provenance block missing")
    else:
        for key in ("db", "base_url", "accessed"):
            if not src.get(key):
                problems.append(f"source.{key} missing (§10.2 provenance)")

    return problems


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    window_nt: int,
    accessed: str,
    per_source: Mapping[str, Mapping[str, int]] | None = None,
    fetch_errors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the measured anchor-audit report (all fields derived, none asserted).

    ``pad_nt`` is **re-derived from the rows** (:func:`derive_pad_nt`), never taken
    from the caller, so the report cannot certify a padding the evidence refutes.
    """
    counts = derive_counts(records)
    pad_nt = derive_pad_nt(records)
    anchored = [r for r in records if r["status"] == STATUS_OK]
    leads = sorted(int(r["lead_flank"]) for r in anchored)
    trails = sorted(int(r["trail_flank"]) for r in anchored)

    def _pct(vals: list[int], q: float) -> int:
        return int(vals[min(len(vals) - 1, max(0, int(q * (len(vals) - 1))))]) if vals else -1

    return {
        "schema_version": SCHEMA_VERSION,
        "step": "P2-00",
        "pad_nt": int(pad_nt),
        "window_nt": int(window_nt),
        "n_records": len(records),
        "n_anchored": len(anchored),
        "anchor_rate": derive_anchor_rate(counts),
        "status_counts": counts,
        "per_source_status": {k: dict(v) for k, v in (per_source or {}).items()},
        "flank_nt": {
            "lead_min": leads[0] if leads else -1,
            "lead_p50": _pct(leads, 0.5),
            "trail_min": trails[0] if trails else -1,
            "trail_p50": _pct(trails, 0.5),
        },
        "n_clipped_start": sum(1 for r in anchored if r["clipped_start"]),
        "n_clipped_end": sum(1 for r in anchored if r["clipped_end"]),
        "n_full_window_capable": sum(
            1
            for r in anchored
            if int(r["lead_flank"]) + int(r["locus_length"]) + int(r["trail_flank"]) >= window_nt
        ),
        "digest": context_digest(records),
        # A wholly-failed run must be able to say WHY rather than certify an
        # empty artifact (§10.3): the distinct fetch errors are carried here.
        "fetch_error_sample": sorted({str(e) for e in (fetch_errors or [])})[:5],
        # Honesty flags (§10.3) — validator-enforced.
        "synthetic_flank": False,
        "coordinates_trusted": False,
        "is_science": False,
        "source": {
            "db": NCBI_DB,
            "base_url": NCBI_BASE_URL,
            "rettype": NCBI_RETTYPE,
            "terms_url": NCBI_TERMS_URL,
            "accessed": accessed,
        },
    }


# --------------------------------------------------------------------------- #
# Fetch layer (lazy heavy imports)
# --------------------------------------------------------------------------- #


def _entrez(email: str, api_key: str | None):
    import socket  # lazy

    from Bio import Entrez  # lazy

    # Bio.Entrez 1.87 calls urlopen(request) with no timeout and exposes no
    # per-call hook, so a stalled socket would hang a multi-hour run forever.
    socket.setdefaulttimeout(NCBI_TIMEOUT_S)

    Entrez.email = email
    Entrez.tool = NCBI_TOOL
    if api_key:
        Entrez.api_key = api_key
    return Entrez


def fetch_region(
    entrez,
    accession: str,
    start: int,
    stop: int,
    strand: int,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 3,
    backoff: float = 1.5,
    errors: list[str] | None = None,
) -> str | None:
    """Fetch one strand-oriented region → uppercase DNA, or ``None`` on failure.

    ``strand=2`` makes NCBI return the reverse complement of the forward
    ``[start, stop]`` range, i.e. the locus read 5'→3' on the minus strand.
    Returns ``None`` (never a partial/guessed sequence) after ``retries``.

    The ``limiter`` is acquired **per request, inside the retry loop** — metering
    only the first attempt would let a retry storm issue up to ``retries`` times
    the NCBI ceiling. The last failure is appended to ``errors`` (never swallowed
    silently) so a wholly-failed run can say *why* instead of certifying an empty
    artifact.
    """
    last_error = ""
    for attempt in range(retries):
        if limiter is not None:
            limiter.acquire()
        try:
            with entrez.efetch(
                db=NCBI_DB,
                id=accession,
                rettype=NCBI_RETTYPE,
                retmode=NCBI_RETMODE,
                seq_start=str(start),
                seq_stop=str(stop),
                strand=str(strand),
            ) as handle:
                text = handle.read()
        except Exception as exc:  # noqa: BLE001 - network/HTTP/parse: retry then give up
            last_error = f"{type(exc).__name__}: {exc}"[:200]
            if attempt == retries - 1:
                break
            time.sleep(backoff * (attempt + 1))
            continue
        lines = [ln.strip() for ln in text.splitlines()]
        if not lines or not lines[0].startswith(">"):
            last_error = "response was not FASTA"
            break
        seq = "".join(lines[1:]).upper()
        if seq:
            return seq
        last_error = "empty sequence body"
        break
    if errors is not None and last_error:
        errors.append(last_error)
    return None


def _should_retry(cached: Mapping[str, Any], *, retry_failed: bool, pad_nt: int) -> bool:
    """True iff a cached row should be re-fetched rather than reused.

    Two independent reasons:

    * **Transient failure.** Only :data:`STATUS_FETCH_FAILED` is retried: it is a
      network/HTTP outcome, so baking it in would understate the anchor rate for a
      reason that has nothing to do with the data. The other non-anchored statuses
      are **deterministic** properties of the record (an unparseable name, a
      sequence genuinely absent from — or repeated within — the fetched region), so
      re-fetching them would burn NCBI quota to reproduce the same answer.
    * **Geometry mismatch.** Every geometry field (``locus_offset``, ``lead_flank``,
      ``trail_flank``, ``clipped_*``, ``region_*``) is a function of ``pad_nt``, so a
      row fetched at one pad must never be reused under another. A legacy row with
      no ``pad_nt`` stamp compares unequal and is re-fetched — the safe direction.
    """
    if retry_failed and str(cached.get("status")) == STATUS_FETCH_FAILED:
        return True
    return cached.get("pad_nt") != int(pad_nt)


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load the resumable fetch log (record_id → row). Corrupt lines are skipped."""
    cache: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return cache
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "record_id" in row:
                cache[str(row["record_id"])] = row
    return cache


class RateLimiter:
    """Thread-safe token spacer — at most ``per_second`` acquisitions per second.

    NCBI's courtesy ceiling is a **global** request rate (3/s anonymous, 10/s with
    an API key), not a per-connection one, so the limiter is shared by every
    worker thread. Fetches are latency-bound (~0.8 s round-trip), so a serial loop
    reaches only ~1.2 req/s — well under the ceiling. Bounded concurrency lets the
    pool sit *at* the ceiling without ever exceeding it.
    """

    def __init__(self, per_second: float) -> None:
        if per_second <= 0:
            raise ValueError(f"per_second must be > 0, got {per_second}")
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
                now = self._next_at
            self._next_at = max(now, self._next_at) + self._interval


def workers_for(api_key: str | None) -> int:
    """Concurrency matched to the applicable NCBI rate ceiling."""
    return int(RATE_LIMIT_WITH_KEY if api_key else RATE_LIMIT_NO_KEY)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def load_corpus(master_parquet: Path, split_table: Path):
    """Read master + the committed split table and return the id-keyed corpus rows.

    Identity is gated fail-loud: ``ingest.compute_record_hashes(master)`` must
    reproduce the committed table's ``corpus_record_sha256`` set exactly. The
    split table is git-LFS (always present), so this checks the join without
    depending on the DVC-tracked labels table.
    """
    import pandas as pd  # lazy

    master = pd.read_parquet(master_parquet)
    hashes = ingest.compute_record_hashes(master)
    if len(hashes) != len(master):
        raise ValueError(f"hash count {len(hashes)} != master rows {len(master)}")

    splits = pd.read_parquet(split_table, columns=[SPLIT_CORPUS_SHA_COL, SPLIT_SOURCE_COL])
    corpus_sha = set(
        splits.loc[splits[SPLIT_SOURCE_COL] == POSITIVE_SOURCE, SPLIT_CORPUS_SHA_COL]
        .dropna()
        .astype(str)
    )
    ours = set(hashes)
    if ours != corpus_sha:
        raise ValueError(
            "master record hashes do not match the committed split table's "
            f"corpus_record_sha256 (master-only {len(ours - corpus_sha)}, "
            f"table-only {len(corpus_sha - ours)}) — refusing to proceed (§10.3)"
        )

    master = master.assign(record_id=hashes)
    return master


def build_flank_context(
    *,
    master_parquet: Path = DEFAULT_MASTER,
    split_table: Path = DEFAULT_SPLIT_TABLE,
    out_dir: Path = DEFAULT_OUT_DIR,
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    report_path: Path = DEFAULT_REPORT,
    pad_nt: int = DEFAULT_PAD_NT,
    window_nt: int = DEFAULT_WINDOW_NT,
    email: str | None = None,
    api_key: str | None = None,
    limit: int | None = None,
    retry_failed: bool = True,
    env_lock: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch + exact-match-anchor real flanking context for every corpus record."""
    import pandas as pd  # lazy

    email = email or os.environ.get("NCBI_EMAIL") or ""
    if not email:
        raise ValueError(
            "an NCBI contact email is required (--email or NCBI_EMAIL) — NCBI "
            "E-utilities etiquette requires the caller to identify themselves"
        )
    api_key = api_key or os.environ.get("NCBI_API_KEY") or None

    out_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / CACHE_NAME

    master = load_corpus(master_parquet, split_table)
    if limit is not None:
        master = master.head(int(limit))

    cache = _read_cache(cache_path)
    entrez = _entrez(email, api_key)
    limiter = RateLimiter(RATE_LIMIT_WITH_KEY if api_key else RATE_LIMIT_NO_KEY)
    n_workers = workers_for(api_key)
    # list.append is atomic under the GIL, so worker threads may share this sink.
    fetch_errors: list[str] = []

    def _resolve(rid: str, name: str, locus: str) -> dict[str, Any]:
        """Fetch + anchor one record (runs on a worker thread)."""
        parsed = parse_locus_name(name)
        if parsed is None:
            return build_context_record(
                record_id=rid,
                locus=locus,
                region=None,
                region_start=-1,
                region_stop=-1,
                accession="",
                strand=STRAND_PLUS,
                parsed_ok=False,
                pad_nt=pad_nt,
            )
        accession, lo, hi, strand = parsed
        start, stop = plan_region(lo, hi, pad_nt=pad_nt)
        region = fetch_region(
            entrez, accession, start, stop, strand, limiter=limiter, errors=fetch_errors
        )
        return build_context_record(
            record_id=rid,
            locus=locus,
            region=region,
            region_start=start,
            region_stop=stop,
            accession=accession,
            strand=strand,
            parsed_ok=True,
            pad_nt=pad_nt,
        )

    by_id: dict[str, dict[str, Any]] = {}
    order: list[tuple[str, str]] = []  # (record_id, source) in master order
    todo: list[tuple[str, str, str]] = []  # (record_id, name, locus)
    for _, row in master.iterrows():
        rid = str(row["record_id"])
        order.append((rid, str(row.get(MASTER_SOURCE_COL))))
        if rid in cache and not _should_retry(cache[rid], retry_failed=retry_failed, pad_nt=pad_nt):
            by_id[rid] = cache[rid]
        else:
            todo.append((rid, str(row[NAME_COL]), str(row[WINDOW_SEQ_COL]).upper()))

    n_fetched = len(todo)
    if todo:
        # Results are appended to the resumable cache as they land, so an
        # interrupted run resumes instead of re-fetching (NCBI courtesy).
        with (
            cache_path.open("a", encoding="utf-8") as cache_fh,
            ThreadPoolExecutor(max_workers=n_workers) as pool,
        ):
            futures = {pool.submit(_resolve, rid, name, locus): rid for rid, name, locus in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                rec = future.result()
                by_id[str(rec["record_id"])] = rec
                cache_fh.write(json.dumps(rec, sort_keys=True) + "\n")
                cache_fh.flush()
                if done % 500 == 0 or done == len(todo):
                    print(f"  fetched {done}/{len(todo)}", file=sys.stderr, flush=True)

    records: list[dict[str, Any]] = []
    per_source: dict[str, dict[str, int]] = {}
    for rid, src in order:
        rec = by_id[rid]
        records.append(rec)
        bucket = per_source.setdefault(src, dict.fromkeys(STATUS_VALUES, 0))
        bucket[str(rec["status"])] += 1

    accessed = date.today().isoformat()
    report = build_report(
        records,
        window_nt=window_nt,
        accessed=accessed,
        per_source=per_source,
        fetch_errors=fetch_errors,
    )
    problems = validate_report(report)
    if problems:
        # Nothing is written: no parquet, no report, no provenance. A degraded run
        # must not leave behind an artifact that looks certified (§10.3). The
        # resumable cache survives, so a fixed re-run costs no NCBI quota.
        raise ValueError(
            "P2-00 flank-context report failed validation (nothing certified):\n  - "
            + "\n  - ".join(problems)
            + (f"\n  fetch errors seen: {report['fetch_error_sample']}" if fetch_errors else "")
        )

    frame = pd.DataFrame(records, columns=list(CONTEXT_COLS)).sort_values("record_id")
    context_path = out_dir / CONTEXT_NAME
    frame.to_parquet(context_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance.write_provenance(
        out_dir / PROVENANCE_NAME,
        rule="workflow/rules/data.smk :: p2_flank_context",
        script="src/tbox_finder/data/flank_context.py",
        seed=provenance.DEFAULT_SEED,
        inputs=[str(master_parquet), str(split_table)],
        outputs=[str(context_path), str(report_path)],
        env_lock=env_lock,
        adr="ADR-0004",
        extra={
            "digest": report["digest"],
            "n_records": report["n_records"],
            "n_anchored": report["n_anchored"],
            "anchor_rate": report["anchor_rate"],
            "pad_nt": report["pad_nt"],
        },
    )
    # §10.2 external-source provenance (source URL, db, terms, accessed date).
    provenance.write_provenance(
        external_dir / "provenance.json",
        rule="workflow/rules/data.smk :: p2_flank_context",
        script="src/tbox_finder/data/flank_context.py",
        seed=provenance.DEFAULT_SEED,
        inputs=[],
        outputs=[str(context_path)],
        env_lock=env_lock,
        adr="ADR-0004",
        extra={
            "base_url": NCBI_BASE_URL,
            "db": NCBI_DB,
            "rettype": NCBI_RETTYPE,
            "terms_url": NCBI_TERMS_URL,
            "accessed": accessed,
            "n_accessions": int(frame["accession"].nunique()),
            "n_requests": n_fetched,
            "note": (
                "Flank sourced from NCBI nuccore by accession; anchored to the corpus "
                "by exact sequence match, not by the stored coordinates (§10.3)."
            ),
        },
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P2-00 — source real flanking genomic context")
    p.add_argument("--master", default=str(DEFAULT_MASTER))
    p.add_argument("--split-table", default=str(DEFAULT_SPLIT_TABLE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--external-dir", default=str(DEFAULT_EXTERNAL_DIR))
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument("--pad-nt", type=int, default=DEFAULT_PAD_NT)
    p.add_argument("--window-nt", type=int, default=DEFAULT_WINDOW_NT)
    p.add_argument("--email", default=None, help="NCBI contact email (or NCBI_EMAIL)")
    p.add_argument("--api-key", default=None, help="NCBI API key (or NCBI_API_KEY) → 10 req/s")
    p.add_argument("--limit", type=int, default=None, help="probe: cap the record count")
    p.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="reuse cached transient fetch_failed rows instead of re-fetching them",
    )
    p.add_argument("--env-lock", default=None)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_flank_context(
        master_parquet=Path(args.master),
        split_table=Path(args.split_table),
        out_dir=Path(args.out_dir),
        external_dir=Path(args.external_dir),
        report_path=Path(args.report),
        pad_nt=args.pad_nt,
        window_nt=args.window_nt,
        email=args.email,
        api_key=args.api_key,
        limit=args.limit,
        retry_failed=not args.no_retry_failed,
        env_lock=args.env_lock,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
