"""pilot_fetch.py — the ρ-pilot LOCAL whole-genome fetch (ADR-0003 D6; P2-10c′-b).

P2-10c′-a selected a reproducible, phylum-stratified, divergence-spanning ~100-genome
manifest of GTDB R232 species representatives (``pilot_genomes_v0.parquet`` — an
**accession list**, no sequence). This module implements the **fetch half**: it pulls the
whole-genome nucleotide sequence for each manifest accession from NCBI and writes one
normalized FASTA per genome, so the SLURM scan sub-step (P2-10c′-c) can measure **ρ =
Stage-1 candidates / Mbp** — the pivot ADR-0003 D6 pins and nobody has measured, and which
sizes the 0.7–68 GB P2-10c′/P2-10e homolog-DB + negative-window fetch.

**No whole-genome fetcher existed.** The one in-repo NCBI client — ``data/flank_context.py``
— fetches *regions* of a single ``nuccore`` record keyed on ``accession:start-end``. A GTDB
manifest instead carries **assembly** accessions (``GC[AF]_<n>.<v>``), so this module adds
the missing **assembly → whole-genome resolution**:

    esearch(db="assembly", term="<acc>[Assembly Accession]")  → assembly UID
    esummary(db="assembly", id=<uid>)                         → the assembly's FtpPath
    GET  <FtpPath>/<basename>_genomic.fna.gz  → gunzip        → the whole genome (all contigs)

The assembly ``_genomic.fna.gz`` (the canonical NCBI genome download every genome tool uses)
was chosen over an ``elink(assembly→nuccore) + efetch`` path after a measured probe: **~50 %
of this manifest's GenBank (GCA) accessions are MAGs whose contigs are not elinked at all** —
only a sequence-free WGS *master* placeholder is linked — so ``elink+efetch`` cannot reach
their sequence, while the ``_genomic.fna.gz`` contains every contig for complete genomes and
MAGs alike. It is a plain HTTPS GET of a gzip stream (stdlib ``urllib`` + ``gzip`` — no env
amendment), and gzip's CRC makes a truncated download **raise** rather than silently yield a
partial genome, so ``total_bp`` (the ρ denominator) can never be understated.

The NCBI E-utilities *transport* (rate limiter, worker count, the ``Entrez`` client with its
socket-timeout + ``max_tries=1`` discipline, the transient-vs-permanent 4xx split) is
**reused verbatim** from ``flank_context`` rather than forked — a second copy would mean
fixing one and shipping the bug in the other (MEMORY: promote-don't-duplicate).

Honesty (CLAUDE.md §10.3). This step **fetches sequence and measures only genome size in
base pairs** — a direct property of the bytes NCBI returned. It counts **no** Stage-1
candidates, computes **no** ρ, and chooses **no** detection threshold (all of that is the
SLURM scan sub-step). ``total_bp`` is the Mbp *denominator* ρ will later divide into, not ρ
itself. So this module pins no ADR value and carries no scientific claim; its report asserts
those honesty flags and the validator enforces them.

Fail-closed, like ``flank_context``: every headline number in the report is **re-derived**
from the per-genome evidence rows (the P1-15/P1-16 lesson that ``all(clauses)`` cannot catch
a clause fabricated TRUE), a fabricated ``ok`` with zero base pairs is rejected, and a run
whose success rate or clade span has degraded below an operational floor writes **nothing**
and routes to a CLAUDE.md §7 stop-and-ask rather than certifying a broken pilot.

Outputs (imp.md P2-10c′-b):
    data/interim/pilot_genomes/<assembly_accession>.fna    one normalized FASTA per genome
    data/interim/pilot_genomes_fetch_log.jsonl             resumable per-genome fetch log
    data/interim/pilot_genomes.provenance.json             §11 provenance sidecar
    data/processed/audits/pilot_fetch_report.json          measured fetch audit (git-tracked)

The pure resolve/parse/summarize/status/derive/validate helpers import bare (stdlib +
``flank_context``'s stdlib-only transport), so the honesty invariants are unit-tested in
bare CI with no biopython, no pandas, and no network.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from tbox_finder import ingest, provenance

# Reuse the flank_context NCBI transport verbatim (promote-don't-duplicate): one rate
# limiter, one worker-count rule, one Entrez client (socket timeout + max_tries=1 so the
# limiter is the SOLE retry authority), one transient-vs-permanent 4xx split, one timeout.
from tbox_finder.data.flank_context import (
    _TRANSIENT_4XX,
    NCBI_BASE_URL,
    NCBI_TERMS_URL,
    NCBI_TIMEOUT_S,
    RATE_LIMIT_NO_KEY,
    RATE_LIMIT_WITH_KEY,
    RateLimiter,
    _entrez,
    workers_for,
)

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #

#: report schema version (bumped on any breaking field change).
SCHEMA_VERSION = "1.0"

#: NCBI E-utilities parameters (PRD §10.2 provenance).
ASSEMBLY_DB = "assembly"

#: The assembly genome download. NCBI serves each assembly's whole-genome FASTA at
#: ``<FtpPath>/<basename>_genomic.fna.gz`` (all replicons/contigs). ``FtpPath`` comes from
#: the assembly esummary; the ``ftp://`` host is served identically over HTTPS.
GENOME_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
GENOMIC_FNA_SUFFIX = "_genomic.fna.gz"
_FTP_KEY_GENBANK = "FtpPath_GenBank"
_FTP_KEY_REFSEQ = "FtpPath_RefSeq"

#: assembly-accession source prefixes → which FtpPath to use (its OWN source's assembly).
_REFSEQ_PREFIX = "GCF_"
_GENBANK_PREFIX = "GCA_"

#: FASTA line-wrap width for the normalized on-disk sequence (deterministic re-fetch).
FASTA_WRAP = 80

#: per-genome fetch outcomes. Only ``STATUS_OK`` rows carry sequence + a Mbp contribution.
#: ``fetch_failed`` (transient network / HTTP-5xx / 429 / truncated-gzip / empty body) is
#: **retried**; every other non-``ok`` status is **deterministic** and never retried:
#: ``no_assembly`` (the accession resolves to no assembly UID), ``no_ftp_path`` (the assembly
#: resolves but its esummary carries no download path — a suppressed/withdrawn assembly), and
#: ``unavailable`` (NCBI permanently rejects a request — a non-rate-limit 4xx, e.g. a 404 on
#: the genome file).
STATUS_OK = "ok"
STATUS_NO_ASSEMBLY = "no_assembly"
STATUS_NO_FTP_PATH = "no_ftp_path"
STATUS_FETCH_FAILED = "fetch_failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_VALUES: tuple[str, ...] = (
    STATUS_OK,
    STATUS_NO_ASSEMBLY,
    STATUS_NO_FTP_PATH,
    STATUS_FETCH_FAILED,
    STATUS_UNAVAILABLE,
)

#: Operational sanity floors (frozen in code — no config may weaken them). NOT scientific
#: thresholds: no PRD/ADR pins a fetch success rate, so these are implementer choices (the
#: ``flank_context.MIN_ANCHOR_RATE`` precedent) that exist only to stop a *broken run*
#: certifying itself. A breach writes nothing and routes to a CLAUDE.md §7 stop-and-ask.
#: ``MIN_PHYLA_OK`` mirrors ``pilot_genomes.DEFAULT_MIN_PHYLA`` so the divergence span the
#: selection guaranteed is not silently lost to fetch failures.
MIN_SUCCESS_RATE = 0.90
MIN_PHYLA_OK = 50

#: manifest columns consumed (written by ``pilot_genomes.build``).
MANIFEST_ACCESSION_COL = "assembly_accession"
MANIFEST_DOMAIN_COL = "domain"
MANIFEST_PHYLUM_COL = "phylum"

#: default artifact locations.
DEFAULT_MANIFEST = Path("data/processed/pilot/pilot_genomes_v0.parquet")
DEFAULT_GENOME_DIR = Path("data/interim/pilot_genomes")
DEFAULT_LOG = Path("data/interim/pilot_genomes_fetch_log.jsonl")
DEFAULT_PROVENANCE = Path("data/interim/pilot_genomes.provenance.json")
DEFAULT_REPORT = Path("data/processed/audits/pilot_fetch_report.json")

#: per-genome log/report row columns.
GENOME_COLS: tuple[str, ...] = (
    "assembly_accession",
    "domain",
    "phylum",
    "status",
    "assembly_uid",
    "source_url",
    "n_replicons",
    "total_bp",
    "seq_sha256",
    "fasta_path",
)


# --------------------------------------------------------------------------- #
# Pure helpers (bare-CI importable — no biopython, no pandas, no network)
# --------------------------------------------------------------------------- #


def parse_esearch_uids(record: Mapping[str, Any]) -> list[str]:
    """Extract the ``IdList`` UID strings from a parsed ``esearch`` record."""
    ids = record.get("IdList") if isinstance(record, Mapping) else None
    if not isinstance(ids, (list, tuple)):
        return []
    return [str(i) for i in ids]


def parse_assembly_docsum(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Pull the single ``DocumentSummary`` mapping from a parsed assembly ``esummary``.

    The v2 esummary structure is ``{"DocumentSummarySet": {"DocumentSummary": [ds, …]}}``.
    Returns ``None`` (never a guess) if the structure is malformed or empty.
    """
    if not isinstance(record, Mapping):
        return None
    dss = record.get("DocumentSummarySet")
    if not isinstance(dss, Mapping):
        return None
    ds = dss.get("DocumentSummary")
    if isinstance(ds, Sequence) and not isinstance(ds, (str, bytes)) and ds:
        first = ds[0]
        return first if isinstance(first, Mapping) else None
    return None


def assembly_ftp_url(docsum: Mapping[str, Any] | None, accession: str) -> str:
    """Build the whole-genome ``_genomic.fna.gz`` URL from an assembly docsum → ``""`` if none.

    A GenBank (``GCA_``) accession uses its own ``FtpPath_GenBank``; a RefSeq (``GCF_``) one
    uses ``FtpPath_RefSeq`` — the assembly's own source, never the cross-source equivalent
    (which may not exist / may differ). The ``ftp://`` path NCBI returns is served identically
    over HTTPS. Returns ``""`` when the assembly carries no download path (a
    suppressed/withdrawn assembly) — the caller records :data:`STATUS_NO_FTP_PATH`.
    """
    if not isinstance(docsum, Mapping):
        return ""
    key = _FTP_KEY_REFSEQ if accession.startswith(_REFSEQ_PREFIX) else _FTP_KEY_GENBANK
    ftp = str(docsum.get(key, "") or "").strip()
    if not ftp:
        return ""
    https = ftp.replace("ftp://", "https://", 1).rstrip("/")
    base = https.rsplit("/", 1)[-1]
    return f"{https}/{base}{GENOMIC_FNA_SUFFIX}"


def iter_fasta_records(text: str) -> list[tuple[str, str]]:
    """Split a multi-record FASTA into ``[(header, sequence)]`` (sequence uppercased).

    ``header`` is the defline with its leading ``>`` stripped and trailing whitespace
    removed; ``sequence`` is the concatenation of the record's sequence lines, uppercased
    with interior whitespace removed. Text before the first ``>`` is ignored. Pure — no I/O.
    """
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header = line[1:].strip()
            seq = []
        elif header is not None:
            seq.append(line.strip().upper())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def normalize_fasta(records: Sequence[tuple[str, str]], *, wrap: int = FASTA_WRAP) -> str:
    """Render parsed records to a deterministic normalized FASTA string.

    Headers verbatim; sequences uppercased and hard-wrapped at ``wrap`` columns. Records are
    emitted **in input order**, so an identical (accession-sorted) record list re-renders
    byte-for-byte — the property DVC content-hashing wants.
    """
    if wrap <= 0:
        raise ValueError(f"wrap must be positive, got {wrap}")
    out: list[str] = []
    for header, seq in records:
        out.append(f">{header}")
        if seq:
            out.extend(seq[i : i + wrap] for i in range(0, len(seq), wrap))
        else:
            out.append("")
    return "\n".join(out) + "\n" if out else ""


def fasta_total_bp(records: Sequence[tuple[str, str]]) -> int:
    """Total base pairs across records (the raw Mbp contribution — not ρ)."""
    return sum(len(seq) for _, seq in records)


def genome_status(
    *,
    esearch_ok: bool,
    uid_found: bool,
    esummary_ok: bool,
    ftp_found: bool,
    download_ok: bool,
    total_bp: int,
    permanent_fail: bool = False,
) -> str:
    """Derive a per-genome status from fetch evidence (builder/validator shared).

    The evidence chain mirrors the resolution order — esearch → esummary → download — and
    each transient failure short-circuits to :data:`STATUS_FETCH_FAILED` (which is retried)
    while each *definitive* outcome yields its own deterministic status (never retried).
    ``permanent_fail`` (a non-rate-limit 4xx anywhere in the chain) dominates.
    """
    if permanent_fail:
        return STATUS_UNAVAILABLE
    if not esearch_ok:
        return STATUS_FETCH_FAILED
    if not uid_found:
        return STATUS_NO_ASSEMBLY
    if not esummary_ok:
        return STATUS_FETCH_FAILED
    if not ftp_found:
        return STATUS_NO_FTP_PATH
    if not download_ok or total_bp <= 0:
        return STATUS_FETCH_FAILED
    return STATUS_OK


def _is_int(value: Any) -> bool:
    """True iff ``value`` is a real integer — a bool is NOT a count (``flank_context``)."""
    return isinstance(value, int) and not isinstance(value, bool)


def genome_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent digest over the per-genome fetch evidence.

    Hashes each row's identity + status + geometry + sequence hash, sorted by
    ``assembly_accession`` (the ``flank_context.context_digest`` precedent). The
    ``seq_sha256`` folds the actual fetched bytes into the digest, so a report cannot match
    while the sequences differ.
    """
    ordered = sorted(rows, key=lambda r: str(r["assembly_accession"]))
    hashes = [
        ingest.record_hash(
            [
                str(r["assembly_accession"]),
                str(r["status"]),
                int(r["n_replicons"]),
                int(r["total_bp"]),
                str(r.get("seq_sha256", "")),
            ]
        )
        for r in ordered
    ]
    return ingest.records_digest(hashes)


# --------------------------------------------------------------------------- #
# Measured-report derivation + fail-closed validator
# --------------------------------------------------------------------------- #


def derive_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Per-status counts, re-derived from the per-genome rows (builder/validator shared)."""
    counts = dict.fromkeys(STATUS_VALUES, 0)
    for row in rows:
        status = str(row.get("status", ""))
        if status not in counts:
            raise ValueError(f"unknown status {status!r} (allowed: {STATUS_VALUES})")
        counts[status] += 1
    return counts


def derive_success_rate(counts: Mapping[str, int]) -> float:
    """Fetched-ok fraction, re-derived from the per-status counts."""
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 0.0
    return int(counts[STATUS_OK]) / total


def _percentiles(values: Sequence[int]) -> dict[str, int]:
    """min / p50 / max over a value list (``-1`` sentinels on an empty list)."""
    if not values:
        return {"min": -1, "p50": -1, "max": -1}
    ordered = sorted(int(v) for v in values)
    p50 = ordered[min(len(ordered) - 1, max(0, int(0.5 * (len(ordered) - 1))))]
    return {"min": ordered[0], "p50": p50, "max": ordered[-1]}


def build_report(
    rows: Sequence[Mapping[str, Any]], *, accessed: str, errors: Sequence[str] | None = None
) -> dict[str, Any]:
    """Assemble the measured fetch-audit report (every field derived, none asserted)."""
    counts = derive_status_counts(rows)
    ok = [r for r in rows if str(r["status"]) == STATUS_OK]
    bps = [int(r["total_bp"]) for r in ok]
    reps = [int(r["n_replicons"]) for r in ok]
    total_bp = sum(bps)

    per_domain_bp: dict[str, int] = {}
    per_domain_ok: dict[str, int] = {}
    for r in ok:
        dom = str(r.get("domain", ""))
        per_domain_bp[dom] = per_domain_bp.get(dom, 0) + int(r["total_bp"])
        per_domain_ok[dom] = per_domain_ok.get(dom, 0) + 1
    phyla_ok = {str(r.get("phylum", "")) for r in ok if str(r.get("phylum", "")).strip()}

    return {
        "schema_version": SCHEMA_VERSION,
        "step": "P2-10c'-b",
        "n_genomes": len(rows),
        "n_ok": len(ok),
        "success_rate": derive_success_rate(counts),
        "status_counts": counts,
        "total_bp": total_bp,
        "total_mbp": total_bp / 1e6,
        "n_phyla_spanned_ok": len(phyla_ok),
        "genome_bp": _percentiles(bps),
        "replicons": {**_percentiles(reps), "total": sum(reps)},
        "per_domain_bp": dict(sorted(per_domain_bp.items())),
        "per_domain_ok": dict(sorted(per_domain_ok.items())),
        "digest": genome_digest(rows),
        "fetch_error_sample": sorted({str(e) for e in (errors or [])})[:5],
        # Honesty flags (§10.3) — validator-enforced. This step measures genome SIZE
        # only; it counts no candidates, computes no ρ, chooses no threshold.
        "rho_measured": False,
        "candidates_counted": False,
        "is_science": False,
        "sequences_synthetic": False,
        "source": {
            "assembly_db": ASSEMBLY_DB,
            "eutils_base_url": NCBI_BASE_URL,
            "genome_ftp_host": GENOME_FTP_HOST,
            "genomic_suffix": GENOMIC_FNA_SUFFIX,
            "terms_url": NCBI_TERMS_URL,
            "accessed": accessed,
        },
        "per_genome": [{k: row[k] for k in GENOME_COLS} for row in rows],
    }


def validate_report(report: Mapping[str, Any]) -> list[str]:  # noqa: C901 - one flat clause list
    """Return a list of schema/consistency problems (empty ⇒ valid). Never raises.

    Fails **closed**: every headline number is **re-derived** from the report's own
    per-genome rows, so a report cannot certify a success rate, a base-pair total, or an
    ``ok`` genome its evidence does not support (the P1-15/P1-16 lesson).
    """
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version != {SCHEMA_VERSION!r} (got {report.get('schema_version')!r})"
        )

    rows = report.get("per_genome")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        problems.append("per_genome block missing or not a list")
        return problems
    # Per-row shape + the fabrication guard: an ``ok`` genome MUST carry real sequence
    # evidence, and a non-``ok`` genome MUST NOT — a fabricated ok with zero bp is the
    # exact shape ``all(clauses)`` cannot catch, so it is caught here per row.
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            problems.append(f"per_genome[{idx}] is not a mapping")
            continue
        missing = [c for c in GENOME_COLS if c not in row]
        if missing:
            problems.append(f"per_genome[{idx}] missing columns {missing}")
            continue
        status = str(row["status"])
        if status not in STATUS_VALUES:
            problems.append(f"per_genome[{idx}] has unknown status {status!r}")
            continue
        for c in ("n_replicons", "total_bp"):
            if not _is_int(row[c]) or int(row[c]) < 0:
                problems.append(f"per_genome[{idx}].{c} is not a non-negative int ({row[c]!r})")
        if status == STATUS_OK:
            if not (_is_int(row["total_bp"]) and int(row["total_bp"]) > 0):
                problems.append(f"per_genome[{idx}] is ok but total_bp <= 0 ({row['total_bp']!r})")
            if not (_is_int(row["n_replicons"]) and int(row["n_replicons"]) > 0):
                problems.append(f"per_genome[{idx}] is ok but n_replicons <= 0")
            if not (isinstance(row["seq_sha256"], str) and len(row["seq_sha256"]) == 64):
                problems.append(f"per_genome[{idx}] is ok but seq_sha256 is not a 64-hex digest")
            if not (isinstance(row["fasta_path"], str) and row["fasta_path"].strip()):
                problems.append(f"per_genome[{idx}] is ok but fasta_path is empty")
        else:
            if _is_int(row["total_bp"]) and int(row["total_bp"]) != 0:
                problems.append(f"per_genome[{idx}] is {status} but total_bp != 0")
            if _is_int(row["n_replicons"]) and int(row["n_replicons"]) != 0:
                problems.append(f"per_genome[{idx}] is {status} but n_replicons != 0")
    if problems:
        return problems

    counts = report.get("status_counts")
    if not isinstance(counts, Mapping):
        problems.append("status_counts block missing or not a mapping")
        return problems
    unknown = set(counts) - set(STATUS_VALUES)
    if unknown:
        problems.append(f"status_counts has unknown status keys: {sorted(unknown)}")
    for status in STATUS_VALUES:
        if not _is_int(counts.get(status)):
            problems.append(f"status_counts[{status!r}] is not an int (got {counts.get(status)!r})")
        elif int(counts[status]) < 0:
            problems.append(f"status_counts[{status!r}] is negative ({counts[status]})")
    if problems:
        return problems

    # Re-derive the counts FROM the rows — the report's status_counts must agree.
    derived = derive_status_counts(rows)
    for status in STATUS_VALUES:
        if int(counts[status]) != derived[status]:
            problems.append(
                f"status_counts[{status!r}] {counts[status]} != re-derived {derived[status]}"
            )

    for key in ("n_genomes", "n_ok", "n_phyla_spanned_ok"):
        if not _is_int(report.get(key)):
            problems.append(f"{key} is not an int (got {report.get(key)!r})")
    if problems:
        return problems

    total = sum(int(counts[s]) for s in STATUS_VALUES)
    if int(report["n_genomes"]) != total:
        problems.append(f"n_genomes {report['n_genomes']} != sum(status_counts) {total}")
    if int(report["n_ok"]) != int(counts[STATUS_OK]):
        problems.append(
            f"n_ok {report['n_ok']} != status_counts['{STATUS_OK}'] {counts[STATUS_OK]}"
        )

    reported = report.get("success_rate")
    rate_is_finite = (
        isinstance(reported, (int, float))
        and not isinstance(reported, bool)
        and math.isfinite(float(reported))
    )
    if not isinstance(reported, (int, float)) or isinstance(reported, bool):
        problems.append(f"success_rate is not a number (got {reported!r})")
    elif not rate_is_finite:
        problems.append(f"success_rate is not finite (got {reported!r})")
    else:
        expected = derive_success_rate({s: int(counts[s]) for s in STATUS_VALUES})
        if abs(float(reported) - expected) > 1e-9:
            problems.append(f"success_rate {reported} != re-derived {expected}")
        elif float(reported) < MIN_SUCCESS_RATE:
            problems.append(
                f"success_rate {float(reported):.4f} < MIN_SUCCESS_RATE {MIN_SUCCESS_RATE} — "
                "too many pilot genomes failed to fetch; a run this degraded is a CLAUDE.md "
                "§7 stop-and-ask, not a certified pilot"
            )

    # total_bp / total_mbp re-derived from the ok rows.
    ok_bp = sum(int(r["total_bp"]) for r in rows if str(r["status"]) == STATUS_OK)
    if not _is_int(report.get("total_bp")) or int(report["total_bp"]) != ok_bp:
        problems.append(f"total_bp {report.get('total_bp')!r} != re-derived {ok_bp}")
    reported_mbp = report.get("total_mbp")
    if not isinstance(reported_mbp, (int, float)) or isinstance(reported_mbp, bool):
        problems.append(f"total_mbp is not a number (got {reported_mbp!r})")
    elif abs(float(reported_mbp) - ok_bp / 1e6) > 1e-6:
        problems.append(f"total_mbp {reported_mbp} != re-derived {ok_bp / 1e6}")

    # Divergence span must survive the fetch (the selection guaranteed it; failures erode it).
    derived_phyla = len(
        {
            str(r.get("phylum", ""))
            for r in rows
            if str(r["status"]) == STATUS_OK and str(r.get("phylum", "")).strip()
        }
    )
    if (
        _is_int(report.get("n_phyla_spanned_ok"))
        and int(report["n_phyla_spanned_ok"]) != derived_phyla
    ):
        problems.append(
            f"n_phyla_spanned_ok {report['n_phyla_spanned_ok']} != re-derived {derived_phyla}"
        )
    if derived_phyla < MIN_PHYLA_OK:
        problems.append(
            f"ok genomes span {derived_phyla} phyla < MIN_PHYLA_OK {MIN_PHYLA_OK} — the "
            "divergence-spanning property the selection guaranteed did not survive the fetch; "
            "ρ measured on it would not bound cross-clade variance (§7 stop-and-ask)"
        )

    if not (isinstance(report.get("digest"), str) and report["digest"].strip()):
        problems.append("digest missing or empty")
    elif report["digest"] != genome_digest(rows):
        problems.append("digest != re-derived genome_digest(per_genome)")

    # Honesty flags — this step measures genome SIZE only (§10.3 no-overclaim).
    for flag, why in (
        ("rho_measured", "ρ is measured by the SLURM scan sub-step, not here"),
        ("candidates_counted", "no Stage-1 candidate is counted here"),
        ("is_science", "this step fetches substrate; it establishes no scientific result"),
        ("sequences_synthetic", "sequence is sourced from NCBI, never generated"),
    ):
        if report.get(flag) is not False:
            problems.append(f"{flag} must be False — {why}")

    src = report.get("source")
    if not isinstance(src, Mapping):
        problems.append("source provenance block missing")
    else:
        for key, expected_val in (
            ("assembly_db", ASSEMBLY_DB),
            ("eutils_base_url", NCBI_BASE_URL),
            ("genome_ftp_host", GENOME_FTP_HOST),
            ("genomic_suffix", GENOMIC_FNA_SUFFIX),
            ("terms_url", NCBI_TERMS_URL),
        ):
            if src.get(key) != expected_val:
                problems.append(f"source.{key} != {expected_val!r} (got {src.get(key)!r})")
        if not (isinstance(src.get("accessed"), str) and src["accessed"].strip()):
            problems.append("source.accessed missing or empty (§10.2 provenance)")

    return problems


# --------------------------------------------------------------------------- #
# Fetch layer (lazy heavy imports)
# --------------------------------------------------------------------------- #


def _read_xml(handle) -> Any:
    """Parse an Entrez XML response stream, always closing the handle."""
    from Bio import Entrez  # lazy

    try:
        return Entrez.read(handle)
    finally:
        handle.close()


def _permanent(exc: Exception) -> bool:
    """True iff ``exc`` is an accession-specific (non-rate-limit) HTTP 4xx rejection."""
    code = getattr(exc, "code", None)
    return isinstance(code, int) and 400 <= code < 500 and code not in _TRANSIENT_4XX


def esearch_assembly_uid(
    entrez,
    accession: str,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 5,
    errors: list[str] | None = None,
) -> tuple[str | None, bool, bool]:
    """Resolve an assembly accession → ``(uid | None, esearch_ok, permanent_fail)``.

    ``esearch_ok`` is False only on a *transient* failure (retried); an ``esearch`` that
    succeeds but returns no id yields ``(None, True, False)`` → :data:`STATUS_NO_ASSEMBLY`.
    The ``[Assembly Accession]`` field pins the match to the accession, not a free-text hit.
    """
    term = f"{accession}[Assembly Accession]"
    last = ""
    for attempt in range(retries):
        if limiter is not None:
            limiter.acquire()
        try:
            record = _read_xml(entrez.esearch(db=ASSEMBLY_DB, term=term))
        except Exception as exc:  # noqa: BLE001 - network/HTTP/parse: retry then give up
            last = f"esearch {type(exc).__name__}: {exc}"[:200]
            if _permanent(exc):
                if errors is not None:
                    errors.append(last)
                return None, False, True
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
            continue
        uids = parse_esearch_uids(record)
        return (uids[0] if uids else None), True, False
    if errors is not None and last:
        errors.append(last)
    return None, False, False


def assembly_genome_url(
    entrez,
    uid: str,
    accession: str,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 5,
    errors: list[str] | None = None,
) -> tuple[str, bool, bool]:
    """Assembly UID → ``(genome_url, esummary_ok, permanent_fail)`` via ``esummary``.

    Returns ``("", True, False)`` when the assembly resolves but carries no FtpPath (a
    suppressed/withdrawn assembly) → :data:`STATUS_NO_FTP_PATH`. A transient esummary failure
    yields ``("", False, False)`` → retried :data:`STATUS_FETCH_FAILED`.
    """
    last = ""
    for attempt in range(retries):
        if limiter is not None:
            limiter.acquire()
        try:
            record = _read_xml(entrez.esummary(db=ASSEMBLY_DB, id=uid))
        except Exception as exc:  # noqa: BLE001
            last = f"esummary {type(exc).__name__}: {exc}"[:200]
            if _permanent(exc):
                if errors is not None:
                    errors.append(last)
                return "", False, True
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
            continue
        return assembly_ftp_url(parse_assembly_docsum(record), accession), True, False
    if errors is not None and last:
        errors.append(last)
    return "", False, False


def download_genome_fasta(
    url: str,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 5,
    errors: list[str] | None = None,
) -> tuple[list[tuple[str, str]] | None, bool]:
    """GET + gunzip the whole-genome ``_genomic.fna.gz`` → ``(records | None, permanent_fail)``.

    A single HTTPS GET of the assembly's genome FASTA (all replicons/contigs). gzip's CRC
    makes a **truncated** download raise on ``decompress`` — so this never returns a partial
    genome with an understated base-pair count (the completeness guarantee; review ncbi
    lens). ``None`` on any failure; the limiter meters the request.
    """
    if not url:
        return None, False
    last = ""
    for attempt in range(retries):
        if limiter is not None:
            limiter.acquire()
        try:
            with urllib.request.urlopen(
                url, timeout=NCBI_TIMEOUT_S
            ) as resp:  # noqa: S310 - https NCBI
                raw = resp.read()
            text = gzip.decompress(raw).decode("ascii", "replace")
        except Exception as exc:  # noqa: BLE001 - network/HTTP/gzip: retry then give up
            last = f"download {type(exc).__name__}: {exc}"[:200]
            if _permanent(exc):
                if errors is not None:
                    errors.append(last)
                return None, True
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
            continue
        records = iter_fasta_records(text)
        if records and fasta_total_bp(records) > 0:
            return records, False
        last = "download produced no sequence"
        if attempt == retries - 1:
            break
        time.sleep(1.5 * (attempt + 1))
    if errors is not None and last:
        errors.append(last)
    return None, False


def resolve_genome(
    entrez,
    *,
    accession: str,
    domain: str,
    phylum: str,
    genome_dir: Path,
    limiter: RateLimiter | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve + fetch one genome, write its normalized FASTA, return an evidence row.

    Runs on a worker thread. Writes ``<genome_dir>/<accession>.fna`` **only** on the
    :data:`STATUS_OK` path; a non-ok genome writes no file and carries zeroed geometry so a
    failure can never be mistaken for a fetched genome.
    """
    row: dict[str, Any] = {
        "assembly_accession": accession,
        "domain": domain,
        "phylum": phylum,
        "status": STATUS_FETCH_FAILED,
        "assembly_uid": "",
        "source_url": "",
        "n_replicons": 0,
        "total_bp": 0,
        "seq_sha256": "",
        "fasta_path": "",
    }

    uid, esearch_ok, perm = esearch_assembly_uid(entrez, accession, limiter=limiter, errors=errors)
    row["assembly_uid"] = uid or ""
    url = ""
    esummary_ok = False
    records: list[tuple[str, str]] | None = None
    total_bp = 0

    if esearch_ok and uid and not perm:
        url, esummary_ok, perm = assembly_genome_url(
            entrez, uid, accession, limiter=limiter, errors=errors
        )
        row["source_url"] = url
        if esummary_ok and url and not perm:
            records, perm = download_genome_fasta(url, limiter=limiter, errors=errors)
            if records is not None:
                total_bp = fasta_total_bp(records)

    status = genome_status(
        esearch_ok=esearch_ok,
        uid_found=bool(uid),
        esummary_ok=esummary_ok,
        ftp_found=bool(url),
        download_ok=records is not None,
        total_bp=total_bp,
        permanent_fail=perm,
    )
    row["status"] = status
    if status == STATUS_OK and records is not None:
        # Canonicalize replicon order (by defline, which begins with the unique nuccore
        # accession) so the on-disk FASTA bytes / seq_sha256 / genome_digest are invariant to
        # NCBI's FASTA record order — deterministic across re-fetches (review determinism
        # lens). total_bp (sum) and n_replicons (len) are order-independent.
        records = sorted(records, key=lambda hr: hr[0])
        text = normalize_fasta(records)
        fasta_path = genome_dir / f"{accession}.fna"
        fasta_path.write_text(text, encoding="utf-8")
        row.update(
            n_replicons=len(records),
            total_bp=total_bp,
            seq_sha256=provenance.sha256_file(fasta_path),
            fasta_path=str(fasta_path),
        )
    return row


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def load_manifest(manifest_parquet: Path) -> list[dict[str, str]]:
    """Read the pilot manifest → one ``{accession, domain, phylum}`` dict per genome.

    Fail-loud: a manifest whose accessions are not ``GC[AF]_<n>.<v>`` or are not unique is a
    corruption, not a genome list (the ``pilot_genomes.build`` guard, re-checked here so the
    fetcher never mines a malformed input, §10.3).
    """
    import pandas as pd  # lazy

    df = pd.read_parquet(
        manifest_parquet,
        columns=[MANIFEST_ACCESSION_COL, MANIFEST_DOMAIN_COL, MANIFEST_PHYLUM_COL],
    )
    rows = [
        {
            "accession": str(r[MANIFEST_ACCESSION_COL]),
            "domain": str(r[MANIFEST_DOMAIN_COL]),
            "phylum": str(r[MANIFEST_PHYLUM_COL]),
        }
        for _, r in df.iterrows()
    ]
    accs = [r["accession"] for r in rows]
    if not accs:
        raise ValueError(f"manifest {manifest_parquet} is empty — nothing to fetch")
    if len(set(accs)) != len(accs):
        raise ValueError(f"manifest {manifest_parquet} has duplicate accessions (§10.3)")
    bad = [a for a in accs if not (a.startswith((_GENBANK_PREFIX, _REFSEQ_PREFIX)) and "." in a)]
    if bad:
        raise ValueError(f"manifest has non-GC[AF]_<n>.<v> accessions: {bad[:5]}")
    return rows


def _load_log(path: Path) -> dict[str, dict[str, Any]]:
    """Load the resumable per-genome fetch log (accession → row). Corrupt lines skipped."""
    log: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return log
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "assembly_accession" in row:
                log[str(row["assembly_accession"])] = row
    return log


def _reusable(cached: Mapping[str, Any], genome_dir: Path) -> bool:
    """True iff a logged genome can be reused instead of re-fetched.

    A transient :data:`STATUS_FETCH_FAILED` is always re-fetched. A deterministic non-ok
    status is reused (re-fetching only re-hits the same dead accession). An ``ok`` genome is
    reused **only** if its FASTA is still on disk and its bytes still hash to the logged
    ``seq_sha256`` — a missing or mutated file is re-fetched (a git-LFS-pointer /
    truncated-file trap, MEMORY: git-lfs-pointers-in-ci).
    """
    if any(c not in cached for c in GENOME_COLS):
        return False
    status = str(cached.get("status"))
    if status not in STATUS_VALUES or status == STATUS_FETCH_FAILED:
        return False
    if status != STATUS_OK:
        return True
    fasta_path = genome_dir / f"{cached['assembly_accession']}.fna"
    if not fasta_path.is_file():
        return False
    try:
        return provenance.sha256_file(fasta_path) == str(cached.get("seq_sha256"))
    except OSError:
        return False


def build_pilot_fetch(
    *,
    manifest_parquet: Path = DEFAULT_MANIFEST,
    genome_dir: Path = DEFAULT_GENOME_DIR,
    log_path: Path = DEFAULT_LOG,
    report_path: Path = DEFAULT_REPORT,
    provenance_path: Path = DEFAULT_PROVENANCE,
    email: str | None = None,
    api_key: str | None = None,
    limit: int | None = None,
    env_lock: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch every manifest genome from NCBI, write per-genome FASTA + audit + provenance."""
    email = email or os.environ.get("NCBI_EMAIL") or ""
    if not email:
        raise ValueError(
            "an NCBI contact email is required (--email or NCBI_EMAIL) — NCBI E-utilities "
            "etiquette requires the caller to identify themselves"
        )
    api_key = api_key or os.environ.get("NCBI_API_KEY") or None

    genome_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_parquet)
    if limit is not None:
        manifest = manifest[: int(limit)]

    log = _load_log(log_path)
    entrez = _entrez(email, api_key)
    limiter = RateLimiter(RATE_LIMIT_WITH_KEY if api_key else RATE_LIMIT_NO_KEY)
    n_workers = workers_for(api_key)
    errors: list[str] = []

    by_acc: dict[str, dict[str, Any]] = {}
    todo: list[dict[str, str]] = []
    for rec in manifest:
        acc = rec["accession"]
        if acc in log and _reusable(log[acc], genome_dir):
            by_acc[acc] = {k: log[acc][k] for k in GENOME_COLS}
        else:
            todo.append(rec)

    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed  # lazy

        write_lock = threading.Lock()
        with (
            log_path.open("a", encoding="utf-8") as log_fh,
            ThreadPoolExecutor(max_workers=n_workers) as pool,
        ):
            futures = {
                pool.submit(
                    resolve_genome,
                    entrez,
                    accession=rec["accession"],
                    domain=rec["domain"],
                    phylum=rec["phylum"],
                    genome_dir=genome_dir,
                    limiter=limiter,
                    errors=errors,
                ): rec["accession"]
                for rec in todo
            }
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                by_acc[str(row["assembly_accession"])] = row
                # ThreadPoolExecutor may complete futures concurrently; serialize the append
                # so log lines never interleave.
                with write_lock:
                    log_fh.write(json.dumps(row, sort_keys=True) + "\n")
                    log_fh.flush()
                if done % 10 == 0 or done == len(todo):
                    print(f"  fetched {done}/{len(todo)}", file=sys.stderr, flush=True)

    rows = [by_acc[rec["accession"]] for rec in manifest]
    accessed = date.today().isoformat()
    report = build_report(rows, accessed=accessed, errors=errors)
    problems = validate_report(report)
    if problems:
        # Nothing certified: no report, no provenance. The resumable log + fetched FASTAs
        # survive on disk, so a fixed re-run costs no NCBI quota for the genomes already in.
        raise ValueError(
            "P2-10c'-b pilot-fetch report failed validation (nothing certified):\n  - "
            + "\n  - ".join(problems)
            + (f"\n  fetch errors seen: {report['fetch_error_sample']}" if errors else "")
        )

    # Orphan reconciliation (review determinism lens): the report's ``digest`` addresses
    # exactly the ``ok`` rows, but the DVC-tracked genome directory is never pruned — a stale
    # ``.fna`` from a prior run under a different manifest would ride along in the committed
    # artifact uncertified by the digest. Assert the on-disk ``.fna`` set equals the certified
    # ok-accession set; a mismatch is a CLAUDE.md §7 stop-and-ask (list the orphans for manual
    # cleanup — never rm a path here, MEMORY: sbatch-rm-of-tracked-path). Only enforced on a
    # full run: a ``--limit`` probe legitimately fetches a subset.
    if limit is None:
        ok_accs = {str(r["assembly_accession"]) for r in rows if str(r["status"]) == STATUS_OK}
        on_disk = {p.stem for p in genome_dir.glob("*.fna")}
        orphans = sorted(on_disk - ok_accs)
        missing = sorted(ok_accs - on_disk)
        if orphans or missing:
            raise ValueError(
                "P2-10c'-b pilot-fetch genome directory does not match the certified set "
                "(nothing certified):\n"
                f"  orphan .fna not in the ok set (remove them): {orphans[:10]}\n"
                f"  ok genomes missing a .fna on disk: {missing[:10]}"
            )

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance.write_provenance(
        provenance_path,
        rule="workflow/rules/data.smk :: fetch_pilot_genomes",
        script="src/tbox_finder/mining/pilot_fetch.py",
        seed=provenance.DEFAULT_SEED,
        inputs=[str(manifest_parquet)],
        # provenance hashes FILES; the genome set is a DVC-tracked *directory* whose content
        # is content-addressed by the report's ``digest`` (which folds in every genome's
        # seq_sha256). Trace the report file here; carry the dir + digest below.
        outputs=[str(report_path)],
        env_lock=env_lock,
        adr="ADR-0003",
        extra={
            "genome_dir": str(genome_dir),
            "digest": report["digest"],
            "n_genomes": report["n_genomes"],
            "n_ok": report["n_ok"],
            "success_rate": report["success_rate"],
            "total_bp": report["total_bp"],
            "total_mbp": report["total_mbp"],
            "eutils_base_url": NCBI_BASE_URL,
            "genome_ftp_host": GENOME_FTP_HOST,
            "terms_url": NCBI_TERMS_URL,
            "accessed": accessed,
            "note": (
                "Whole-genome sequence sourced from NCBI: assembly accession → assembly UID "
                "(esearch) → assembly FtpPath (esummary) → whole-genome _genomic.fna.gz "
                "(HTTPS GET + gunzip; all replicons/contigs). Measures genome SIZE (bp) only "
                "— no ρ, no candidate count, no detection threshold (§10.3). gzip integrity "
                "guarantees a complete download (a truncated stream raises)."
            ),
        },
    )
    print(
        f"fetched {report['n_ok']}/{report['n_genomes']} pilot genomes "
        f"({report['total_mbp']:.1f} Mbp total across {report['n_phyla_spanned_ok']} phyla)",
        file=sys.stderr,
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tbox_finder.mining.pilot_fetch",
        description="P2-10c'-b — LOCAL whole-genome fetch of the ρ-pilot manifest",
    )
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--genome-dir", default=str(DEFAULT_GENOME_DIR))
    p.add_argument("--log", default=str(DEFAULT_LOG))
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    p.add_argument("--email", default=None, help="NCBI contact email (or NCBI_EMAIL)")
    p.add_argument("--api-key", default=None, help="NCBI API key (or NCBI_API_KEY) → 10 req/s")
    p.add_argument("--limit", type=int, default=None, help="probe: cap the genome count")
    p.add_argument("--env-lock", default=None)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_pilot_fetch(
        manifest_parquet=Path(args.manifest),
        genome_dir=Path(args.genome_dir),
        log_path=Path(args.log),
        report_path=Path(args.report),
        provenance_path=Path(args.provenance),
        email=args.email,
        api_key=args.api_key,
        limit=args.limit,
        env_lock=args.env_lock,
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "per_genome"}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
