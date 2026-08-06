"""P3-15′-c-i (acquisition) — download the NCBI annotation the (c) supply measurement found.

The measurement half of this step (``mining/annotation_supply``, merged as ``2be5ff4``) probed
all 660 admissible hosts and wrote ``reports/p3/annotation_supply.json``: **339 annotated /
321 unannotated**, route ``mixed``, and — the part that makes this module small — every
annotated row already carries its ``gff_url`` **and** the ``gff_md5`` NCBI publishes for it.
So the acquisition needs **no URL discovery at all**: no ``esearch``, no ``esummary``, not even
a directory listing. It reads a committed report and GETs 339 files it already knows the
checksum of.

That committed md5 is the whole integrity story, and it is a better one than the genome fetch
got. ``pilot_fetch`` has no checksum path — it relies on gzip's CRC to catch truncation and
hashes the *normalized* FASTA it writes, which certifies its own output rather than NCBI's.
Here the bytes are stored **exactly as served** (``.gff.gz``, never decompressed, never
rewritten), so ``md5(file) == report.gff_md5`` stays checkable forever, by anyone, against a
value that was recorded before this module existed. :func:`verify_annotations` re-runs that
check offline.

What this module is *not*
-------------------------
It is not the (c) predicate. ADR-0006 D4 — first same-strand downstream CDS start within
500 bp encodes an aaRS / amino-acid-biosynthesis / transport / transamidation function — is
P3-15′-c-ii, together with its clade-matched false-pass rate and the symmetric false-FAIL /
per-clade-exclusion / pseudogene diagnostic D4 requires. **No threshold appears in this file**,
and ``500`` appears nowhere in it: an acquisition step that quietly pinned D4's window would be
pinning a blinded-frozen value outside the ADR that owns it.

The ceiling this acquisition buys, stated plainly
-------------------------------------------------
339 of 660 admissible hosts, and **655 of 941 candidates (69.6 %)**. The remaining 286
candidates sit on 28 unannotated hosts, where (c) reads ``unavailable`` and ADR-0005 D14
therefore **spares** rather than mines them. Raising that ceiling to 941 is the separate
annotator arm P3-15′-c commissioned (bakta/prokka over all 321 unannotated hosts), which needs
an ADR-0002 env amendment; this module touches no ``envs/*`` file and adds no dependency
beyond the standard library.

Refusals, and why the completeness clause is a *separate* clause
---------------------------------------------------------------
Every gate here is fail-closed and re-derived from the rows rather than read from a headline
([[gate-clauses-need-re-derivation]]). Two of them exist specifically because correctness
alone is not enough:

* ``--limit`` sets ``sweep_complete: False`` and :func:`validate_fetch_report` refuses on it.
  A cost knob that can hand a phase-exit path a green report is the ``--max-records 8`` shape
  ([[cost-knobs-can-certify]]) — every *fetched* file being perfect says nothing about the 331
  that were never requested.
* ``n_ok`` must equal the full target count **and** the target count must equal the supply
  report's own ``admissible_status_counts["annotated"]``. A report describing 8 flawless
  downloads must not be distinguishable from a complete one only by reading the numbers.

The verification control is powered, and the negative legs are what make it so. A run where
the md5 comparison silently never happened would otherwise report 339 clean files; so on a
pinned assembly the control asserts the real bytes **match**, a corrupted *expectation*
**fails**, and corrupted *payload* bytes **fail** — proving the comparison reads both sides,
not just one ([[matched-control-before-certifying]], [[control-matchedness-must-be-asserted]]).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from tbox_finder import provenance
from tbox_finder.data.flank_context import NCBI_TIMEOUT_S, RATE_LIMIT_NO_KEY, RateLimiter
from tbox_finder.mining import gff3
from tbox_finder.mining.annotation_supply import (
    ACCESSION_RE,
    GFF_SUFFIX,
    ROUTE_REFUSED,
    STATUS_ANNOTATED,
    AnnotationSupplyError,
    assembly_basename,
    build_allowlisted_opener,
    require_allowed_url,
)

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-c-i"
ADR = "ADR-0006"

DEFAULT_SUPPLY_REPORT = "reports/p3/annotation_supply.json"
DEFAULT_ANNOTATION_DIR = "data/interim/production_annotations"
DEFAULT_FETCH_REPORT = "reports/p3/annotation_fetch.json"
DEFAULT_PARSE_REPORT = "reports/p3/annotation_parse.json"

#: per-assembly outcomes. ``ok`` covers both a fresh download and a cache hit — the *how* is
#: ``from_cache``, because a cached file is md5-verified on every run and is therefore exactly
#: as trustworthy, while a status that conflated them would hide a run that downloaded nothing.
STATUS_OK = "ok"
STATUS_MD5_MISMATCH = "md5_mismatch"
STATUS_FETCH_FAILED = "fetch_failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_VALUES: tuple[str, ...] = (
    STATUS_OK,
    STATUS_MD5_MISMATCH,
    STATUS_FETCH_FAILED,
    STATUS_UNAVAILABLE,
)

#: the assembly the md5 machinery certifies itself on. Pinned rather than "whichever fetched
#: first", so the report names a fixed witness; and required to be *in* the target set, so a
#: shrunken target list is a refusal instead of a silently relocated control.
CONTROL_ACCESSION = "GCA_002790315.1"

USER_AGENT = "tbox-finder/P3-15c-i annotation acquisition (https://github.com/bioedca/tbox-finder)"
NCBI_TERMS_URL = "https://www.ncbi.nlm.nih.gov/home/about/policies/"
GENOME_FTP_HOST = "ftp.ncbi.nlm.nih.gov"

_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_BACKOFF_BASE_S = 1.5
_TRANSIENT_4XX = frozenset({401, 403, 408, 429})

#: every clause :func:`validate_fetch_report` must have *evaluated*. Compared as a set before
#: ``all(...)`` runs, because a clause that is simply absent contributes nothing to an ``all``
#: and passes ([[clauses-must-guard-emptiness]] — the P3-15′-b r1 major was exactly this).
REQUIRED_CLAUSES: frozenset[str] = frozenset(
    {
        "sweep_complete",
        "targets_match_supply_report",
        "rows_match_targets",
        "no_duplicate_accessions",
        "status_counts_rederive",
        "all_targets_ok",
        "every_ok_row_md5_matches",
        "bytes_total_rederives",
        "bytes_total_matches_supply_report",
        "path_binds_to_accession",
        "control_powered",
        "corpus_digest_rederives",
        "no_orphans",
    }
)


class AnnotationFetchError(RuntimeError):
    """A refusal: the acquisition cannot be made or cannot be trusted."""


# --------------------------------------------------------------------------- #
# Pure helpers (no network, no filesystem) — the unit-testable core
# --------------------------------------------------------------------------- #


def md5_hex(payload: bytes) -> str:
    """Lowercase hex md5 of ``payload``.

    md5 is not a security choice here — it is the algorithm NCBI publishes in
    ``md5checksums.txt``, so it is the only digest that can be compared against a value this
    repo did not compute itself. ``usedforsecurity=False`` says so to the linters.
    """
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def digest_matches(payload: bytes, expected_md5: str) -> bool:
    """The **one** md5 comparison this module makes — the acquisition's whole integrity test.

    Extracted rather than inlined so that :func:`verification_control` exercises the same
    comparator :func:`fetch_one` and :func:`parse_census` do. Inline, the control's negative
    legs reduce to "md5 has no collisions" — true, and completely uninformative about *this*
    code: they could not go False no matter what the acquisition did. Routed through here,
    a comparator that always answers True makes both negative legs fail and the control
    reports itself unpowered, which is the only version of the control that is worth running
    ([[control-matchedness-must-be-asserted]]).
    """
    return md5_hex(payload) == str(expected_md5).strip().lower()


def corpus_digest(pairs: Iterable[tuple[str, str]]) -> str:
    """A stable identity for "which bytes this report is about".

    sha256 over sorted ``accession:md5`` lines. Both the acquisition report and the offline
    census emit one, so a reader can tell **by comparison** that the census parsed the files
    the acquisition validated — rather than inferring it from two timestamps that happened to
    be close. Without it a census generated *before* its fetch (the ordering defect this fixes)
    describes a previous corpus state and nothing in either artifact says so.
    """
    lines = sorted(f"{accession}:{str(md5).strip().lower()}" for accession, md5 in pairs)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def destination_name(accession: str) -> str:
    """``GCA_000296795.1`` → ``GCA_000296795.1.gff.gz``.

    Flat and accession-keyed, mirroring ``data/interim/production_genomes/<accession>.fna``.
    The NCBI basename (``…_ASM29679v1``) is deliberately **not** in the filename: the assembly
    name is not stable across NCBI re-releases, and a consumer joining on the candidate
    manifest has the accession and nothing else.
    """
    text = str(accession)
    if not ACCESSION_RE.fullmatch(text):
        raise AnnotationFetchError(f"malformed assembly accession: {accession!r}")
    return f"{text}.gff.gz"


def check_url_binds_to_accession(accession: str, url: str) -> None:
    """Refuse a row whose ``gff_url`` names a different assembly than its ``accession``.

    The row is *labelled* with the accession but the bytes come from whatever directory the
    URL points at, so a mis-join files a real, well-formed GFF under the wrong assembly and
    never surfaces as a failure — it just makes (c) answer about the wrong genome
    ([[namespace-mismatch-invisible-noop]]). This is the same binding check PR #112 added at
    the measurement layer; the acquisition needs its own, because it consumes the report's
    two fields independently.
    """
    basename = assembly_basename(str(url)[: -len(GFF_SUFFIX)] + "_genomic.fna.gz")
    if not basename.startswith(f"{accession}_"):
        raise AnnotationFetchError(
            f"gff_url does not belong to {accession!r}: basename {basename!r}"
        )


def load_annotation_targets(
    supply_report: str | Path = DEFAULT_SUPPLY_REPORT,
) -> list[dict[str, Any]]:
    """The committed supply report → the acquisition's target list, or a refusal.

    Refuses rather than returning a short list when the report cannot certify what it is
    describing: a partial sweep, a refused route, an annotated row missing its URL or md5, a
    row count that disagrees with the report's own headline tally, a URL outside the NCBI
    allowlist, or a URL that names a different assembly than its row.
    """
    return annotation_targets(read_supply_report(supply_report))


def read_supply_report(supply_report: str | Path = DEFAULT_SUPPLY_REPORT) -> Mapping[str, Any]:
    """Parse the committed supply report **once**, with its envelope checks.

    Split out because a caller needing both the target list and a headline count used to parse
    the same file twice, and two reads of one path are not guaranteed to see the same bytes —
    the targets and the number they are checked against could describe different payloads.
    """
    path = Path(supply_report)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnnotationFetchError(f"supply report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnnotationFetchError(f"supply report is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise AnnotationFetchError(
            f"supply report root is {type(payload).__name__}, expected an object"
        )
    return payload


def strict_count(payload: Mapping[str, Any], key: str) -> int:
    """A headline count read as a **real** ``int`` or refused.

    ``int(payload.get(key) or 0)`` raises ``ValueError``/``TypeError`` on a malformed field, so
    it escapes as a traceback and exit 1 instead of the refusal and exit 3 this module
    promises; and it would coerce ``"339"`` or ``339.7`` into a count, the coerce-before-
    validate mistake P3-15′-b shipped where ``"0.5"`` merged as ``0.5`` and *certified*.
    ``bool`` is excluded explicitly — ``isinstance(True, int)`` is True.
    """
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AnnotationFetchError(f"supply report {key} is {value!r}, expected an int")
    return value


def annotation_targets(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An already-parsed supply report → the target list, or a refusal. Pure; no I/O."""
    if payload.get("sweep_complete") is not True:
        raise AnnotationFetchError(
            "supply report has sweep_complete != True — a partial sweep cannot enumerate the "
            "annotated hosts, so its target list would understate the corpus silently"
        )
    route = payload.get("route")
    if isinstance(route, Mapping) and route.get("route") == ROUTE_REFUSED:
        raise AnnotationFetchError(
            f"supply report route is {ROUTE_REFUSED!r}: {route.get('reasons')}"
        )

    rows = payload.get("per_assembly")
    if not isinstance(rows, list):
        raise AnnotationFetchError(
            f"supply report per_assembly is {type(rows).__name__}, expected a list"
        )

    counts = payload.get("admissible_status_counts")
    if not isinstance(counts, Mapping):
        raise AnnotationFetchError("supply report has no admissible_status_counts block")
    expected = counts.get(STATUS_ANNOTATED)
    if not isinstance(expected, int) or isinstance(expected, bool):
        raise AnnotationFetchError(
            f"admissible_status_counts[{STATUS_ANNOTATED!r}] is {expected!r}, expected an int"
        )

    targets: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AnnotationFetchError(
                f"per_assembly row is {type(row).__name__}, expected an object"
            )
        if row.get("status") != STATUS_ANNOTATED:
            continue
        accession = str(row.get("accession", ""))
        url = str(row.get("gff_url", ""))
        md5 = str(row.get("gff_md5", "")).lower()
        n_bytes = row.get("gff_bytes")
        if not url:
            raise AnnotationFetchError(f"{accession}: status is annotated but gff_url is empty")
        if not _MD5_RE.fullmatch(md5):
            raise AnnotationFetchError(
                f"{accession}: gff_md5 is not 32 lowercase hex digits: {md5!r}"
            )
        if not isinstance(n_bytes, int) or isinstance(n_bytes, bool) or n_bytes <= 0:
            raise AnnotationFetchError(
                f"{accession}: gff_bytes is {n_bytes!r}, expected a positive int"
            )
        # Allowlist FIRST. Behind the suffix check it is unreachable for every URL that does
        # not happen to end in ``_genomic.gff.gz`` — which quietly turned a test asserting
        # "a file:// URL is refused" into a test of the *suffix* rule, passing for the wrong
        # reason ([[raises-test-needs-a-positive-control]]). Order is the fix; matching on the
        # message is the other half.
        try:
            require_allowed_url(url)
        except AnnotationSupplyError as exc:
            raise AnnotationFetchError(f"{accession}: {exc}") from exc
        if not url.endswith(GFF_SUFFIX):
            raise AnnotationFetchError(
                f"{accession}: gff_url does not end in {GFF_SUFFIX!r}: {url!r}"
            )
        check_url_binds_to_accession(accession, url)
        targets.append(
            {"accession": accession, "gff_url": url, "gff_md5": md5, "expected_bytes": n_bytes}
        )

    if len(targets) != expected:
        raise AnnotationFetchError(
            f"supply report lists {len(targets)} annotated rows but its own "
            f"admissible_status_counts says {expected} — the report disagrees with itself"
        )
    seen = {t["accession"] for t in targets}
    if len(seen) != len(targets):
        raise AnnotationFetchError(
            f"supply report has duplicate annotated accessions "
            f"({len(targets)} rows, {len(seen)} unique)"
        )
    return targets


def verification_control(payload: bytes, expected_md5: str) -> dict[str, Any]:
    """Three legs that together prove the md5 comparison reads *both* of its sides.

    ``positive_matches`` — the real bytes hash to the recorded value. On its own this is
    satisfied by a comparison that always returns True.
    ``corrupt_expectation_fails`` — the same bytes against a mutated md5 must **not** match,
    which is what fails if the expectation is ignored.
    ``corrupt_payload_fails`` — mutated bytes against the real md5 must **not** match, which
    is what fails if the *payload* is ignored (a re-hash of the expectation, say).
    Each leg is broken alone in the tests, because either negative alone is satisfied by a
    degenerate comparison.
    """
    flipped = ("b" if expected_md5[0] != "b" else "c") + expected_md5[1:]
    return {
        "accession": CONTROL_ACCESSION,
        "n_bytes": len(payload),
        "observed_md5": md5_hex(payload),
        "expected_md5": expected_md5,
        "positive_matches": digest_matches(payload, expected_md5),
        "corrupt_expectation_fails": not digest_matches(payload, flipped),
        "corrupt_payload_fails": not digest_matches(payload + b"\x00", expected_md5),
    }


def control_is_powered(control: Mapping[str, Any]) -> bool:
    """All three legs, each read by name.

    A missing key is False, never absent-and-skipped: ``all(control.get(k) for k in ...)`` over
    a *declared* key list is the version that cannot be satisfied by an empty dict.
    """
    return all(
        control.get(leg) is True
        for leg in ("positive_matches", "corrupt_expectation_fails", "corrupt_payload_fails")
    )


def derive_status_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Re-derive the per-status tally from the rows themselves."""
    counts = dict.fromkeys(STATUS_VALUES, 0)
    for row in rows:
        status = str(row.get("status", ""))
        if status not in counts:
            raise AnnotationFetchError(f"unknown per-assembly status {status!r}")
        counts[status] += 1
    return counts


def build_fetch_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[Mapping[str, Any]],
    control: Mapping[str, Any],
    orphans: Sequence[str],
    n_annotated_in_supply: int,
    bytes_total_in_supply: int,
    sweep_complete: bool,
) -> dict[str, Any]:
    """Assemble the acquisition report. Every headline is derived here from ``rows``."""
    counts = derive_status_counts(rows)
    ok_rows = [r for r in rows if r.get("status") == STATUS_OK]
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "sweep_complete": bool(sweep_complete),
        "n_targets": len(targets),
        "n_annotated_in_supply_report": int(n_annotated_in_supply),
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_downloaded": sum(1 for r in ok_rows if r.get("from_cache") is False),
        "n_cached": sum(1 for r in ok_rows if r.get("from_cache") is True),
        "status_counts": counts,
        "bytes_total": sum(int(r.get("n_bytes") or 0) for r in ok_rows),
        "corpus_digest": corpus_digest(
            (str(r.get("accession", "")), str(r.get("observed_md5", ""))) for r in ok_rows
        ),
        "bytes_total_in_supply_report": int(bytes_total_in_supply),
        "control": dict(control),
        "orphans": sorted(orphans),
        "per_assembly": sorted(rows, key=lambda r: str(r.get("accession", ""))),
        "source": {
            "accessed": date.today().isoformat(),
            "genome_ftp_host": GENOME_FTP_HOST,
            "gff_suffix": GFF_SUFFIX,
            "terms_url": NCBI_TERMS_URL,
        },
    }


def validate_fetch_report(report: Mapping[str, Any]) -> list[str]:
    """Re-derive every clause from ``per_assembly``; return the problems, empty on a clean run.

    Nothing here trusts a headline the writer put in the report — each clause recomputes its
    fact from the rows, so a hand-edited or truncated report fails rather than certifying.
    """
    rows = report.get("per_assembly")
    if not isinstance(rows, list) or not all(isinstance(r, Mapping) for r in rows):
        return ["per_assembly is not a list of objects"]

    ok_rows = [r for r in rows if r.get("status") == STATUS_OK]
    accessions = [str(r.get("accession", "")) for r in rows]
    n_targets = report.get("n_targets")
    n_supply = report.get("n_annotated_in_supply_report")

    clauses: dict[str, bool] = {}
    clauses["sweep_complete"] = report.get("sweep_complete") is True
    clauses["targets_match_supply_report"] = (
        isinstance(n_targets, int)
        and not isinstance(n_targets, bool)
        and isinstance(n_supply, int)
        and not isinstance(n_supply, bool)
        and n_targets == n_supply
    )
    clauses["rows_match_targets"] = isinstance(n_targets, int) and len(rows) == n_targets
    clauses["no_duplicate_accessions"] = len(set(accessions)) == len(accessions)
    try:
        clauses["status_counts_rederive"] = derive_status_counts(rows) == report.get(
            "status_counts"
        )
    except AnnotationFetchError:
        clauses["status_counts_rederive"] = False
    clauses["all_targets_ok"] = isinstance(n_targets, int) and len(ok_rows) == n_targets
    clauses["every_ok_row_md5_matches"] = all(
        _MD5_RE.fullmatch(str(r.get("observed_md5", "")))
        and str(r.get("observed_md5")) == str(r.get("expected_md5"))
        for r in ok_rows
    )
    rederived_bytes = sum(int(r.get("n_bytes") or 0) for r in ok_rows)
    clauses["bytes_total_rederives"] = rederived_bytes == report.get("bytes_total")
    clauses["bytes_total_matches_supply_report"] = rederived_bytes == report.get(
        "bytes_total_in_supply_report"
    )
    try:
        clauses["path_binds_to_accession"] = all(
            Path(str(r.get("path", ""))).name == destination_name(str(r.get("accession", "")))
            for r in ok_rows
        )
    except AnnotationFetchError:
        clauses["path_binds_to_accession"] = False
    control = report.get("control")
    clauses["corpus_digest_rederives"] = report.get("corpus_digest") == corpus_digest(
        (str(r.get("accession", "")), str(r.get("observed_md5", ""))) for r in ok_rows
    )
    clauses["control_powered"] = isinstance(control, Mapping) and control_is_powered(control)
    clauses["no_orphans"] = report.get("orphans") == []

    missing = REQUIRED_CLAUSES - set(clauses)
    if missing:
        return [f"clause(s) never evaluated: {sorted(missing)}"]
    # The other direction, which the set-equality test alone could not assert: a clause this
    # function computes but nobody declared is never read by the comprehension below, so it
    # would be silently ignored no matter what it evaluated to.
    undeclared = set(clauses) - REQUIRED_CLAUSES
    if undeclared:
        return [f"clause(s) computed but never declared: {sorted(undeclared)}"]
    return [f"clause failed: {name}" for name in sorted(REQUIRED_CLAUSES) if not clauses[name]]


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def _permanent(exc: Exception) -> bool:
    """True iff ``exc`` is a URL-specific (non-rate-limit) HTTP 4xx rejection."""
    code = getattr(exc, "code", None)
    return isinstance(code, int) and 400 <= code < 500 and code not in _TRANSIENT_4XX


def _urlopen_bytes(url: str) -> bytes:
    req = urllib.request.Request(require_allowed_url(url), headers={"User-Agent": USER_AGENT})
    with build_allowlisted_opener().open(
        req, timeout=NCBI_TIMEOUT_S
    ) as resp:  # noqa: S310 - https NCBI
        return bytes(resp.read())


def download_gff(
    url: str,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 4,
    opener: Any = None,
    sleep: Any = None,
) -> tuple[bytes | None, str, bool]:
    """GET ``url`` → ``(payload | None, note, permanent_fail)``.

    Backoff is ``1.5 * (attempt + 1)`` — the schedule the rest of the repo uses. The rate
    limiter is **not** a substitute: it meters the steady state at 3 req/s, so without a
    backoff a 429 would be re-sent four times inside ~1.3 s, which is precisely what the
    limiter exists to avoid. A permanent 4xx does **not** sleep — 339 pointless waits turn a
    two-minute acquisition into a twenty-minute one.
    """
    wait = sleep if sleep is not None else time.sleep
    get = opener if opener is not None else _urlopen_bytes
    note = ""
    for attempt in range(retries):
        if limiter is not None:
            limiter.acquire()
        try:
            return bytes(get(url)), "", False
        except AnnotationSupplyError:
            raise
        except Exception as exc:  # noqa: BLE001 - network/HTTP: retry, then give up
            note = f"{type(exc).__name__}: {exc}"[:200]
            if _permanent(exc):
                return None, note, True
            if attempt == retries - 1:
                break
            wait(_BACKOFF_BASE_S * (attempt + 1))
    return None, note, False


def fetch_one(
    target: Mapping[str, Any],
    *,
    annotation_dir: str | Path,
    limiter: RateLimiter | None = None,
    opener: Any = None,
    sleep: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire one assembly's GFF → a per-assembly report row.

    A file already on disk is **re-hashed, not trusted**: a cache hit is only a cache hit when
    its md5 still matches the committed value, so an interrupted or corrupted earlier run
    re-downloads instead of certifying. Bytes are written only *after* the digest matches —
    a mismatching payload never reaches the annotation directory under a name a later step
    would read.
    """
    accession = str(target["accession"])
    url = str(target["gff_url"])
    expected = str(target["gff_md5"]).lower()
    dest = Path(annotation_dir) / destination_name(accession)
    check_url_binds_to_accession(accession, url)

    row: dict[str, Any] = {
        "accession": accession,
        "gff_url": url,
        "expected_md5": expected,
        "observed_md5": "",
        "n_bytes": 0,
        "path": str(dest),
        "from_cache": False,
        "note": "",
    }

    if dest.exists() and not force:
        payload = dest.read_bytes()
        if digest_matches(payload, expected):
            row.update(
                status=STATUS_OK,
                observed_md5=md5_hex(payload),
                n_bytes=len(payload),
                from_cache=True,
            )
            return row
        row["note"] = f"cached file md5 {md5_hex(payload)} != expected; re-downloading"

    payload, note, permanent = download_gff(url, limiter=limiter, opener=opener, sleep=sleep)
    if payload is None:
        row.update(
            status=STATUS_UNAVAILABLE if permanent else STATUS_FETCH_FAILED,
            note=(row["note"] + " | " + note).strip(" |"),
        )
        return row

    observed = md5_hex(payload)
    row.update(observed_md5=observed, n_bytes=len(payload))
    if not digest_matches(payload, expected):
        row.update(
            status=STATUS_MD5_MISMATCH,
            note=(row["note"] + f" | md5 {observed} != expected {expected}").strip(" |"),
        )
        return row

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    row["status"] = STATUS_OK
    return row


def find_orphans(annotation_dir: str | Path, expected_accessions: Iterable[str]) -> list[str]:
    """On-disk ``*.gff.gz`` files that no target claims.

    Reported, never deleted — the repo's standing rule for reconciliation. An orphan means the
    directory holds annotation from a different corpus than the report describes, which makes
    a later "(c) is unavailable for this host" ambiguous between *absent* and *stale*.
    """
    directory = Path(annotation_dir)
    if not directory.is_dir():
        return []
    wanted = {destination_name(str(a)) for a in expected_accessions}
    return sorted(p.name for p in directory.glob("*.gff.gz") if p.name not in wanted)


def fetch_annotations(
    *,
    supply_report: str | Path = DEFAULT_SUPPLY_REPORT,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    limiter: RateLimiter | None = None,
    workers: int = 4,
    opener: Any = None,
    sleep: Any = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire every annotated host's GFF and return the acquisition report."""
    payload = read_supply_report(supply_report)
    targets = annotation_targets(payload)
    n_annotated = strict_count(payload["admissible_status_counts"], STATUS_ANNOTATED)
    bytes_total_in_supply = strict_count(payload, "gff_bytes_total_known")

    if not any(t["accession"] == CONTROL_ACCESSION for t in targets):
        raise AnnotationFetchError(
            f"control assembly {CONTROL_ACCESSION} is not in the target set — the acquisition "
            "would run with an unpowered md5 control"
        )

    selected = targets if limit is None else targets[: max(0, int(limit))]
    sweep_complete = limit is None

    Path(annotation_dir).mkdir(parents=True, exist_ok=True)

    def _one(target: Mapping[str, Any]) -> dict[str, Any]:
        return fetch_one(
            target,
            annotation_dir=annotation_dir,
            limiter=limiter,
            opener=opener,
            sleep=sleep,
            force=force,
        )

    if workers > 1 and len(selected) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_one, selected))
    else:
        rows = [_one(t) for t in selected]

    control_row = next((r for r in rows if r["accession"] == CONTROL_ACCESSION), None)
    if control_row is None or control_row.get("status") != STATUS_OK:
        control = {
            "accession": CONTROL_ACCESSION,
            "positive_matches": False,
            "corrupt_expectation_fails": False,
            "corrupt_payload_fails": False,
            "note": "control assembly was not acquired in this run",
        }
    else:
        control = verification_control(
            Path(control_row["path"]).read_bytes(), str(control_row["expected_md5"])
        )

    return build_fetch_report(
        rows,
        targets=selected,
        control=control,
        orphans=find_orphans(annotation_dir, (t["accession"] for t in targets)),
        n_annotated_in_supply=n_annotated,
        bytes_total_in_supply=bytes_total_in_supply,
        sweep_complete=sweep_complete,
    )


# --------------------------------------------------------------------------- #
# Offline verification + parse census
# --------------------------------------------------------------------------- #


def parse_census(
    *,
    supply_report: str | Path = DEFAULT_SUPPLY_REPORT,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    fetch_report: str | Path = DEFAULT_FETCH_REPORT,
) -> dict[str, Any]:
    """Re-verify every acquired file offline **and** parse it — the report ``verify`` writes.

    Two independent things at once, on purpose. The md5 re-check proves the bytes on disk are
    still NCBI's; the parse proves those bytes are readable as the CDS coordinates D4 needs.
    A corpus that hashes correctly but parses to zero CDS is useless to (c), and a census that
    only counted files would call it healthy.
    """
    targets = load_annotation_targets(supply_report)
    directory = Path(annotation_dir)

    rows: list[dict[str, Any]] = []
    for target in targets:
        accession = str(target["accession"])
        path = directory / destination_name(accession)
        row: dict[str, Any] = {"accession": accession, "path": str(path)}
        if not path.exists():
            rows.append({**row, "ok": False, "note": "missing"})
            continue
        raw = path.read_bytes()
        if not digest_matches(raw, str(target["gff_md5"])):
            rows.append({**row, "ok": False, "note": f"md5 {md5_hex(raw)} != {target['gff_md5']}"})
            continue
        try:
            text = gff3.read_gff3_text(path)
            cds = gff3.parse_gff3_document(text)
        except (gff3.Gff3Error, OSError, UnicodeDecodeError) as exc:
            rows.append({**row, "ok": False, "note": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        rows.append(
            {
                **row,
                "ok": len(cds) > 0,
                "observed_md5": md5_hex(raw),
                "n_bytes": len(raw),
                "n_cds": len(cds),
                "n_cds_pseudo": sum(1 for c in cds if gff3.is_pseudo(c)),
                "n_cds_multisegment": sum(1 for c in cds if len(c.segments) > 1),
                "n_cds_resolved_strand": sum(1 for c in cds if c.strand in gff3.STRANDS_RESOLVED),
                "n_cds_with_identity_text": sum(1 for c in cds if gff3.gene_identity_text(c)),
                "n_seqids": len({c.seqid for c in cds}),
                "has_fasta_section": any(
                    line.strip().upper() == gff3.FASTA_DIRECTIVE for line in text.splitlines()
                ),
                "note": "" if cds else "parsed to zero CDS",
            }
        )

    ok_rows = [r for r in rows if r.get("ok") is True]
    # The corpus this census actually read, and the corpus the acquisition certified. Compared
    # rather than assumed: a census run BEFORE its fetch parses a previous state of the
    # directory and, on timestamps alone, looks indistinguishable from one run after.
    observed = corpus_digest((str(r["accession"]), str(r.get("observed_md5", ""))) for r in ok_rows)
    # Validate the fetch report before trusting anything in it. Reading ``corpus_digest``
    # alone lets a hand-edited report declare whatever the census just computed, and the
    # binding then certifies itself — the digest has to be re-derived from the rows, which is
    # what ``validate_fetch_report``'s ``corpus_digest_rederives`` clause does.
    declared = ""
    try:
        fetch_payload = read_supply_report(fetch_report)
    except AnnotationFetchError as exc:
        reason = f"the fetch report is unreadable: {exc}"
    else:
        # Every clause, not just the digest field. Reading ``corpus_digest`` out of an
        # unvalidated report lets a hand-edited one declare whatever the census just computed,
        # and the binding then certifies itself.
        problems = validate_fetch_report(fetch_payload)
        if problems:
            reason = f"the fetch report does not validate: {'; '.join(problems)}"
        else:
            declared = str(fetch_payload.get("corpus_digest", ""))
            reason = (
                "ok" if declared == observed else "the fetch report certifies a different corpus"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "corpus_digest": observed,
        "fetch_report_corpus_digest": declared,
        "corpus_matches_fetch_report": reason == "ok",
        "corpus_check_reason": reason,
        "n_targets": len(targets),
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "failures": [r for r in rows if r.get("ok") is not True],
        "totals": {
            "n_cds": sum(int(r.get("n_cds", 0)) for r in ok_rows),
            "n_cds_pseudo": sum(int(r.get("n_cds_pseudo", 0)) for r in ok_rows),
            "n_cds_multisegment": sum(int(r.get("n_cds_multisegment", 0)) for r in ok_rows),
            "n_cds_resolved_strand": sum(int(r.get("n_cds_resolved_strand", 0)) for r in ok_rows),
            "n_cds_with_identity_text": sum(
                int(r.get("n_cds_with_identity_text", 0)) for r in ok_rows
            ),
            "n_seqids": sum(int(r.get("n_seqids", 0)) for r in ok_rows),
            "n_with_fasta_section": sum(1 for r in ok_rows if r.get("has_fasta_section")),
        },
        "per_assembly": sorted(rows, key=lambda r: str(r.get("accession", ""))),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _write(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _cmd_fetch(args: argparse.Namespace) -> int:
    report = fetch_annotations(
        supply_report=args.supply_report,
        annotation_dir=args.annotation_dir,
        limiter=RateLimiter(RATE_LIMIT_NO_KEY),
        workers=args.workers,
        limit=args.limit,
        force=args.force,
    )
    problems = validate_fetch_report(report)
    report["provenance"] = provenance.build_provenance(
        rule="src/tbox_finder/mining/annotation_fetch.py :: fetch",
        script="src/tbox_finder/mining/annotation_fetch.py",
        inputs=[args.supply_report],
        adr=ADR,
        extra={
            "accessed": date.today().isoformat(),
            "annotation_dir": str(args.annotation_dir),
            "control_accession": CONTROL_ACCESSION,
            "gff_suffix": GFF_SUFFIX,
        },
    )
    report["validation_problems"] = problems
    out = _write(args.out, report)
    print(
        f"annotation-fetch: {report['n_ok']}/{report['n_targets']} ok "
        f"({report['n_downloaded']} downloaded, {report['n_cached']} cached), "
        f"{report['bytes_total']} B → {out}"
    )
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 3
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    report = parse_census(
        supply_report=args.supply_report,
        annotation_dir=args.annotation_dir,
        fetch_report=args.fetch_report,
    )
    report["provenance"] = provenance.build_provenance(
        rule="src/tbox_finder/mining/annotation_fetch.py :: verify",
        script="src/tbox_finder/mining/annotation_fetch.py",
        inputs=[args.supply_report, args.fetch_report],
        adr=ADR,
        extra={"annotation_dir": str(args.annotation_dir)},
    )
    out = _write(args.out, report)
    totals = report["totals"]
    print(
        f"annotation-verify: {report['n_ok']}/{report['n_targets']} ok | "
        f"{totals['n_cds']} CDS ({totals['n_cds_pseudo']} pseudo, "
        f"{totals['n_cds_multisegment']} multi-segment) → {out}"
    )
    if report["n_failed"]:
        print(f"REFUSED: {report['n_failed']} assemblies failed re-verification", file=sys.stderr)
        return 3
    if not report["corpus_matches_fetch_report"]:
        print(
            f"REFUSED: {report['corpus_check_reason']} "
            f"({report['corpus_digest'][:12]} vs "
            f"{report['fetch_report_corpus_digest'][:12] or '<absent>'})"
            " — run `verify` AFTER `fetch`",
            file=sys.stderr,
        )
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annotation-fetch",
        description="P3-15'-c-i — acquire the NCBI GFF annotation criterion (c) is read from.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download every annotated host's GFF, md5-verified")
    f.add_argument("--supply-report", default=DEFAULT_SUPPLY_REPORT)
    f.add_argument("--annotation-dir", default=DEFAULT_ANNOTATION_DIR)
    f.add_argument("--out", default=DEFAULT_FETCH_REPORT)
    f.add_argument("--workers", type=int, default=4)
    f.add_argument(
        "--force",
        action="store_true",
        help="re-download even when an on-disk file already matches its committed md5.",
    )
    f.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: fetch the first N targets. Sets sweep_complete=False, which the "
        "report validation REFUSES on — a cost knob may not certify a complete acquisition.",
    )
    f.set_defaults(func=_cmd_fetch)

    v = sub.add_parser("verify", help="offline: re-hash every acquired GFF and parse it")
    v.add_argument("--supply-report", default=DEFAULT_SUPPLY_REPORT)
    v.add_argument("--annotation-dir", default=DEFAULT_ANNOTATION_DIR)
    v.add_argument("--fetch-report", default=DEFAULT_FETCH_REPORT)
    v.add_argument("--out", default=DEFAULT_PARSE_REPORT)
    v.set_defaults(func=_cmd_verify)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (AnnotationFetchError, AnnotationSupplyError) as exc:
        print(f"annotation-fetch: REFUSED — {exc}", file=sys.stderr)
        return 3


__all__ = [
    "ADR",
    "CONTROL_ACCESSION",
    "DEFAULT_ANNOTATION_DIR",
    "DEFAULT_FETCH_REPORT",
    "DEFAULT_PARSE_REPORT",
    "DEFAULT_SUPPLY_REPORT",
    "REQUIRED_CLAUSES",
    "SCHEMA_VERSION",
    "STATUS_FETCH_FAILED",
    "STATUS_MD5_MISMATCH",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "STATUS_VALUES",
    "STEP",
    "USER_AGENT",
    "AnnotationFetchError",
    "build_fetch_report",
    "build_parser",
    "check_url_binds_to_accession",
    "control_is_powered",
    "corpus_digest",
    "derive_status_counts",
    "destination_name",
    "download_gff",
    "fetch_annotations",
    "fetch_one",
    "find_orphans",
    "annotation_targets",
    "load_annotation_targets",
    "main",
    "md5_hex",
    "parse_census",
    "read_supply_report",
    "strict_count",
    "validate_fetch_report",
    "verification_control",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
