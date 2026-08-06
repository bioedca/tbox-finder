"""P3-15′-c-i — is the (c) synteny backend supplyable from NCBI-served annotation?

ADR-0006 **D4** makes criterion (c) *"the first downstream same-strand CDS start within 500 bp
of the T-box element 3′ end encodes an aaRS / amino-acid-biosynthesis / transport /
transamidation function"*. Every word of that needs **CDS coordinates + a gene identity** for
the candidate's host assembly, and the repo has none: zero GFF/GBK/faa artifacts under
``data/``, zero annotation tooling in any ``envs/*.yml``.

`imp.md` P3-15′-c-i frames the route question as *"how many of the 660 admissible hosts are
``GCF_`` (RefSeq, annotated) vs ``GCA_`` (GenBank/MAG, frequently unannotated)? If the GCA
share is large, ``prodigal`` is required after all ⇒ an **ADR-0002 env amendment**."* — that
proxy is measured here (:func:`accession_prefix_tally`) **and is not the decision**. A
``GCA_`` assembly is very often annotated by its submitter or by PGAP, and NCBI then serves
its ``_genomic.gff.gz`` at the same ``FtpPath`` the genome came from. The decision therefore
rests on the **directly measured** per-assembly file manifest, not on the accession prefix.

Transport: **no new one.** Every admissible host's whole-genome download URL is already
recorded, per assembly, in the git-tracked ``data/processed/audits/production_fetch_report.json``
(``per_genome[].source_url``, written by ``mining/pilot_fetch.py`` at P2-10c′-fetch). The
annotation lives beside it — same directory, different suffix — so the probe is a **sibling
lookup on a committed URL**, and ``esearch``/``esummary`` are not called at all. One GET of
each assembly directory's ``md5checksums.txt`` returns that directory's **complete** file list
(plus the checksums a later acquisition step needs), so one request answers every suffix at
once rather than one HEAD per suffix.

**The route rule is pre-registered here, above the measurement** (CLAUDE.md §10.3): it is a
function of counts this module computes, so no number can be chosen after seeing the answer.
See :func:`derive_acquisition_route` and :data:`ROUTE_*`.

**Three-valued, fail-closed.** A probe outcome is ``annotated`` / ``unannotated`` / ``unknown``
— never two-valued. A transport failure is ``unknown``, and ``unknown`` is **not** folded into
``unannotated``: doing so would let a broken run (wrong host, captive portal, NCBI outage)
report "nothing is annotated" and thereby *certify* the expensive ``prodigal`` route, the
[[matched-control-before-certifying]] failure. :func:`derive_acquisition_route` refuses
outright while any host is ``unknown``.

**The probe carries its own control.** A run that 404s on everything and a run that resolves
everything are distinguishable only if the probe is exercised against a URL that **must**
resolve and a URL that **must not**. :func:`run_control` does both and
:func:`control_is_powered` reports whether the pair actually discriminated; a route derivation
on an unpowered control is refused.

Consumers
---------
``P3-15′-c-ii`` builds ``criterion_c`` on top of the acquired GFFs. This module stops at
*supply* — what can be acquired, at what cost, and by which route — and pins **no** predicate,
**no** threshold and **no** ADR value.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from tbox_finder import provenance

# Reuse the flank_context NCBI transport verbatim (promote-don't-duplicate), exactly as
# ``mining/pilot_fetch.py`` does: one rate limiter, one worker rule, one timeout, one
# transient-vs-permanent 4xx split. This module invents no rate policy of its own.
from tbox_finder.data.flank_context import (
    _TRANSIENT_4XX,
    NCBI_TIMEOUT_S,
    RATE_LIMIT_NO_KEY,
    RateLimiter,
)

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #

#: report schema version (bumped on any breaking field change).
SCHEMA_VERSION = "1.0"

#: the committed evidence this module reads. Both are **git-tracked and non-DVC**, so CI,
#: laptop and cluster get the same answer (the P3-15′-b derivation lesson).
DEFAULT_FETCH_REPORT = "data/processed/audits/production_fetch_report.json"

#: the suffix ``pilot_fetch`` downloaded; every sibling URL is built by replacing it.
GENOMIC_FNA_SUFFIX = "_genomic.fna.gz"

#: NCBI's per-assembly file manifest. One GET returns every file in the assembly directory.
MD5_MANIFEST_NAME = "md5checksums.txt"

#: the annotation artifacts (c) needs, by role.
#: ``GFF`` carries the CDS coordinates + strand + product name D4 reads;
#: ``FAA`` carries the protein sequences D4's Pfam/KO hypothetical/pseudogene fallback runs on.
GFF_SUFFIX = "_genomic.gff.gz"
PROTEIN_FAA_SUFFIX = "_protein.faa.gz"

#: ``_genomic.gbff.gz`` is served for **unannotated** assemblies too (a sequence-only flat
#: file), so its presence is *not* evidence of annotation and it is recorded for information
#: only — never read by :func:`classify_assembly`.
GENOMIC_GBFF_SUFFIX = "_genomic.gbff.gz"

#: per-assembly annotation outcomes. Three-valued on purpose (see the module docstring).
STATUS_ANNOTATED = "annotated"
STATUS_UNANNOTATED = "unannotated"
STATUS_UNKNOWN = "unknown"
STATUS_VALUES: tuple[str, ...] = (STATUS_ANNOTATED, STATUS_UNANNOTATED, STATUS_UNKNOWN)

#: pre-registered acquisition routes (see :func:`derive_acquisition_route`).
#: ``NCBI_GFF``   — every candidate-carrying host is annotated: the shipped fetcher's suffix
#:                  extension supplies (c) outright. **No ADR-0002 amendment.**
#: ``MIXED``      — some candidate-carrying hosts are unannotated. The round still runs: an
#:                  unannotated host makes (c) ``unavailable``, which *spares* its candidates
#:                  (ADR-0005 D14 fail-closed), so ``prodigal`` would raise the mining yield
#:                  rather than make the round possible. Calling for it is a §7 decision.
#: ``PRODIGAL``   — no candidate-carrying host is annotated: the NCBI route supplies nothing
#:                  and a gene caller is the only way to a non-zero (c) supply.
#: ``REFUSED``    — the measurement did not establish the answer (any ``unknown``, or an
#:                  unpowered control, or an incomplete sweep). Never a route to act on.
ROUTE_NCBI_GFF = "ncbi_gff"
ROUTE_MIXED = "mixed"
ROUTE_PRODIGAL = "prodigal_required"
ROUTE_REFUSED = "refused"

#: assembly-accession shape. ``pilot_fetch`` writes these; a malformed one is a refusal, not
#: a skip — a silently-dropped accession understates the denominator every fraction is read
#: against.
ACCESSION_RE = re.compile(r"^GC[AF]_\d{9}\.\d+$")

#: the control pair. The positive is a *fixed, named* assembly that must resolve **and** be
#: annotated; the negative is that same assembly's directory with its accession corrupted, so
#: it must **not** resolve. Any run in which both come back the same way has no power to tell
#: "unannotated" from "unreachable" and may not certify a route.
CONTROL_POSITIVE_ACCESSION = "GCF_000007185.1"
CONTROL_NEGATIVE_SUFFIX = "__tbox_finder_control_must_not_exist"

#: transient-retry backoff base, in seconds — the same ``1.5 * (attempt + 1)`` schedule
#: ``pilot_fetch`` uses. Operational, not scientific; private so the public surface stays
#: free of numbers (see ``test_module_pins_no_scientific_value``).
_BACKOFF_BASE_S = 1.5

#: the only host this module may contact, and the only schemes it may use. Every probe URL
#: is *derived* from a committed ``source_url``, so a ``file://`` or third-party value in
#: that report would otherwise become a local-file read or an off-host request the moment it
#: reached ``urlopen``. Enforced both where a URL is built and at the transport boundary,
#: because ``fetch_file_manifest``/``head_content_length`` are public and take a URL directly.
ALLOWED_URL_HOSTS: frozenset[str] = frozenset({"ftp.ncbi.nlm.nih.gov"})
ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"https"})

#: HTTP identification. NCBI asks that automated clients identify themselves.
USER_AGENT = "tbox-finder/P3-15c-i annotation-supply probe (https://github.com/bioedca/tbox-finder)"


class AnnotationSupplyError(RuntimeError):
    """A refusal: the measurement cannot be made or cannot be trusted."""


# --------------------------------------------------------------------------- #
# Pure helpers (no network) — the unit-testable core
# --------------------------------------------------------------------------- #


def require_allowed_url(url: str) -> str:
    """``url`` if it is an allowed NCBI URL, else a refusal.

    The guard is on *scheme and host*, not on a substring: ``"ftp.ncbi.nlm.nih.gov" in url``
    would accept ``https://evil.test/?x=ftp.ncbi.nlm.nih.gov``.
    """
    parsed = urllib.parse.urlsplit(str(url))
    if parsed.scheme not in ALLOWED_URL_SCHEMES or parsed.hostname not in ALLOWED_URL_HOSTS:
        raise AnnotationSupplyError(f"not an allowed NCBI URL: {url!r}")
    return str(url)


def sibling_url(source_url: str, suffix: str) -> str:
    """``…/<basename>_genomic.fna.gz`` → ``…/<basename><suffix>``.

    The assembly *basename* is the directory name NCBI uses for both the directory and every
    file inside it, so the sibling is a pure string substitution on a URL this repo already
    recorded — no listing, no e-utils, no guess. Raises on a URL that does not carry the
    expected suffix rather than emitting a plausible-looking wrong URL.
    """
    url = str(source_url).strip()
    if not url.endswith(GENOMIC_FNA_SUFFIX):
        raise AnnotationSupplyError(f"source_url does not end in {GENOMIC_FNA_SUFFIX!r}: {url!r}")
    return require_allowed_url(url)[: -len(GENOMIC_FNA_SUFFIX)] + suffix


def assembly_dir_url(source_url: str) -> str:
    """The assembly's directory URL (the parent of its ``_genomic.fna.gz``)."""
    url = str(source_url).strip()
    if "/" not in url:
        raise AnnotationSupplyError(f"source_url is not a URL: {url!r}")
    return require_allowed_url(url).rsplit("/", 1)[0]


def assembly_basename(source_url: str) -> str:
    """``GCA_000296795.1_ASM29679v1`` — the shared prefix of every file in the directory."""
    leaf = str(source_url).strip().rsplit("/", 1)[-1]
    if not leaf.endswith(GENOMIC_FNA_SUFFIX):
        raise AnnotationSupplyError(
            f"source_url does not end in {GENOMIC_FNA_SUFFIX!r}: {source_url!r}"
        )
    return leaf[: -len(GENOMIC_FNA_SUFFIX)]


def parse_md5_manifest(text: str) -> dict[str, str]:
    """NCBI ``md5checksums.txt`` → ``{filename: md5}``.

    Each line is ``<32-hex md5>  ./<filename>``. A line that does not match is **skipped
    silently only if blank**; any other malformed line raises, because a partially-parsed
    manifest that happens to omit the GFF line is indistinguishable from an unannotated
    assembly — the [[clauses-must-guard-emptiness]] shape.
    """
    out: dict[str, str] = {}
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.fullmatch(r"([0-9a-fA-F]{32})\s+\./(.+)", line)
        if m is None:
            raise AnnotationSupplyError(f"malformed md5checksums.txt line: {line[:120]!r}")
        out[m.group(2)] = m.group(1).lower()
    if not out:
        raise AnnotationSupplyError("md5checksums.txt parsed to zero entries")
    return out


def classify_assembly(filenames: Iterable[str], basename: str) -> str:
    """The assembly's file list → :data:`STATUS_ANNOTATED` / :data:`STATUS_UNANNOTATED`.

    Annotated **iff** the directory serves ``<basename>_genomic.gff.gz``. That file is the
    only one D4's rule can be evaluated from directly (CDS coordinates + strand + product),
    and NCBI writes it exactly when the assembly carries an annotation.

    ``_genomic.gbff.gz`` is deliberately **not** consulted: NCBI serves one for unannotated
    assemblies too, so keying on it would call an unannotated GenBank assembly annotated.

    This function never returns :data:`STATUS_UNKNOWN` — that state belongs to the transport
    (a directory whose listing could not be read), not to a listing that was read.
    """
    names = set(filenames)
    return STATUS_ANNOTATED if f"{basename}{GFF_SUFFIX}" in names else STATUS_UNANNOTATED


def accession_prefix_tally(accessions: Iterable[str]) -> dict[str, int]:
    """``{"GCA": n, "GCF": m}`` — the ``imp.md`` proxy, measured but not decisive."""
    tally = {"GCA": 0, "GCF": 0}
    for acc in accessions:
        if not ACCESSION_RE.fullmatch(str(acc)):
            raise AnnotationSupplyError(f"malformed assembly accession: {acc!r}")
        tally[str(acc)[:3]] += 1
    return tally


def load_source_urls(
    fetch_report: str | Path = DEFAULT_FETCH_REPORT,
) -> dict[str, str]:
    """The committed ``production_fetch_report.json`` → ``{accession: source_url}``.

    Only ``status == "ok"`` rows carry a genome, and only those can carry an annotation
    sibling. A row with an empty ``source_url`` is dropped **with** its accession absent from
    the mapping, so a caller asking for it gets a ``KeyError``, never a silent empty URL.
    """
    payload = json.loads(Path(fetch_report).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AnnotationSupplyError(f"{fetch_report}: top-level JSON is not an object")
    rows = payload.get("per_genome")
    if not isinstance(rows, list) or not rows:
        raise AnnotationSupplyError(
            f"{fetch_report}: per_genome is missing or empty — cannot resolve any URL"
        )
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AnnotationSupplyError(f"{fetch_report}: per_genome row is not a mapping")
        acc = str(row.get("assembly_accession", ""))
        url = str(row.get("source_url", "") or "")
        if str(row.get("status", "")) != "ok" or not url:
            continue
        if not ACCESSION_RE.fullmatch(acc):
            raise AnnotationSupplyError(f"{fetch_report}: malformed accession {acc!r}")
        if acc in out and out[acc] != url:
            raise AnnotationSupplyError(
                f"{fetch_report}: accession {acc} carries conflicting source_url values — "
                "one would be silently discarded and the sweep would report the surviving "
                "directory as this accession's evidence with no record of the conflict"
            )
        out[acc] = url
    if not out:
        raise AnnotationSupplyError(f"{fetch_report}: no ok row carries a source_url")
    return out


def candidate_host_accessions(fp_manifest: str | Path) -> list[str]:
    """The round's FP manifest → the sorted **assembly** accessions that carry a candidate.

    The manifest's ``accession`` field is ``<assembly>:c<contig_index>`` (the id shape
    ``homolog_msa.resolve_candidate_sequence`` parses), so the assembly is the part before the
    first ``:``. Reading the whole field as an accession is the mistake this function exists
    to make impossible: it inflates the host count from 76 to 228.
    """
    payload = json.loads(Path(fp_manifest).read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise AnnotationSupplyError(f"{fp_manifest}: no candidates")
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or "accession" not in row:
            raise AnnotationSupplyError(f"{fp_manifest}: candidate row carries no accession")
        acc = str(row["accession"]).split(":", 1)[0]
        if not ACCESSION_RE.fullmatch(acc):
            raise AnnotationSupplyError(f"{fp_manifest}: malformed host accession {acc!r}")
        out.add(acc)
    return sorted(out)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def _permanent(exc: Exception) -> bool:
    """True iff ``exc`` is a resource-specific (non-rate-limit) HTTP 4xx rejection."""
    code = getattr(exc, "code", None)
    return isinstance(code, int) and 400 <= code < 500 and code not in _TRANSIENT_4XX


def fetch_file_manifest(
    dir_url: str,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 4,
    opener: Any = None,
    sleep: Any = None,
) -> tuple[dict[str, str] | None, str]:
    """GET ``<dir_url>/md5checksums.txt`` → ``({filename: md5} | None, note)``.

    ``None`` means **unknown**, not empty: a 404 on the manifest is returned as ``None`` with
    a note, because an assembly directory that serves no manifest tells us nothing about
    whether it serves a GFF. Only a manifest that *parsed* licenses a classification.

    A *transient* failure (429 / 5xx / socket) backs off before the retry, on the same
    ``1.5 * (attempt + 1)`` schedule ``pilot_fetch`` uses. The rate limiter alone is not
    backoff — it meters the steady state at 3 req/s, so without this a 429 would be re-sent
    four times inside ~1.3 s, which is the behaviour the limiter exists to prevent.
    """
    url = require_allowed_url(f"{dir_url.rstrip('/')}/{MD5_MANIFEST_NAME}")
    get = opener if opener is not None else _urlopen_text
    wait = sleep if sleep is not None else time.sleep
    last = ""
    for attempt in range(retries):
        if limiter is not None:
            limiter.acquire()
        try:
            text = get(url)
        except AnnotationSupplyError:
            raise  # an allowlist violation is a refusal, not a transient network failure
        except Exception as exc:  # noqa: BLE001 - network/HTTP: retry then give up
            last = f"{type(exc).__name__}: {exc}"[:200]
            if _permanent(exc):
                return None, last
            if attempt == retries - 1:
                break
            wait(_BACKOFF_BASE_S * (attempt + 1))
            continue
        try:
            return parse_md5_manifest(text), ""
        except AnnotationSupplyError as exc:
            return None, f"parse: {exc}"[:200]
    return None, last or "exhausted retries"


class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-check the allowlist on every redirect hop.

    ``urlopen`` follows redirects, so validating only the URL we hand it leaves the guard one
    hop deep: a 302 can move the request to another host or scheme after the check has passed.
    Every hop is re-validated here, and a violation raises rather than being followed.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        require_allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


#: the single opener both transport helpers use, so neither can bypass the redirect guard.
_OPENER = urllib.request.build_opener(_AllowlistRedirectHandler())


def _urlopen_text(url: str) -> str:
    req = urllib.request.Request(require_allowed_url(url), headers={"User-Agent": USER_AGENT})
    with _OPENER.open(req, timeout=NCBI_TIMEOUT_S) as resp:  # noqa: S310 - https NCBI
        return resp.read().decode("utf-8", "replace")


def _urlhead_length(url: str) -> int:
    req = urllib.request.Request(
        require_allowed_url(url), method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    with _OPENER.open(req, timeout=NCBI_TIMEOUT_S) as resp:  # noqa: S310 - https NCBI
        return int(resp.headers.get("Content-Length") or -1)


def head_content_length(
    url: str,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 3,
    header: Any = None,
    sleep: Any = None,
) -> int | None:
    """``Content-Length`` of ``url``, or ``None`` when it could not be established.

    Feeds the acquisition cost estimate only. ``None`` is reported as
    ``n_gff_size_unknown`` and never silently counted as zero — an understated download
    budget is how a SLURM acquisition gets sized against a number nobody measured.

    Backs off between transient retries on the same schedule as
    :func:`fetch_file_manifest`.
    """
    if not url:
        return None
    require_allowed_url(url)
    get = header if header is not None else _urlhead_length
    wait = sleep if sleep is not None else time.sleep
    for attempt in range(retries):
        if limiter is not None:
            limiter.acquire()
        try:
            n = int(get(url))
        except AnnotationSupplyError:
            raise  # an allowlist violation is a refusal, not a transient network failure
        except Exception as exc:  # noqa: BLE001 - network/HTTP: retry then give up
            if _permanent(exc) or attempt == retries - 1:
                return None
            wait(_BACKOFF_BASE_S * (attempt + 1))
            continue
        return n if n >= 0 else None
    return None


def probe_assembly(
    accession: str,
    source_url: str,
    *,
    limiter: RateLimiter | None = None,
    opener: Any = None,
    sleep: Any = None,
) -> dict[str, Any]:
    """One assembly → its annotation-supply evidence row.

    The pair ``(accession, source_url)`` is checked to agree before anything is fetched. The
    row is *labelled* with ``accession`` but *classified* from whatever directory ``source_url``
    names, so a mis-joined pair yields well-formed evidence filed under the wrong assembly —
    it never surfaces as ``unknown`` and it moves the counts the route is derived from. A
    mismatch is a join error, not a per-host unknown, so it refuses.
    """
    basename = assembly_basename(source_url)
    if not basename.startswith(f"{accession}_"):
        raise AnnotationSupplyError(
            f"source_url basename {basename!r} does not belong to accession {accession!r}"
        )
    manifest, note = fetch_file_manifest(
        assembly_dir_url(source_url), limiter=limiter, opener=opener, sleep=sleep
    )
    if manifest is None:
        return {
            "accession": accession,
            "status": STATUS_UNKNOWN,
            "note": note,
            "gff_url": "",
            "gff_md5": "",
            "gff_bytes": None,
            "faa_present": False,
            "gbff_present": False,
            "n_files": 0,
        }
    status = classify_assembly(manifest, basename)
    gff_name = f"{basename}{GFF_SUFFIX}"
    return {
        "accession": accession,
        "status": status,
        "note": "",
        "gff_url": sibling_url(source_url, GFF_SUFFIX) if status == STATUS_ANNOTATED else "",
        "gff_md5": manifest.get(gff_name, ""),
        "gff_bytes": None,
        "faa_present": f"{basename}{PROTEIN_FAA_SUFFIX}" in manifest,
        "gbff_present": f"{basename}{GENOMIC_GBFF_SUFFIX}" in manifest,
        "n_files": len(manifest),
    }


# --------------------------------------------------------------------------- #
# The control
# --------------------------------------------------------------------------- #


def run_control(
    source_urls: Mapping[str, str],
    *,
    limiter: RateLimiter | None = None,
    opener: Any = None,
    sleep: Any = None,
) -> dict[str, Any]:
    """Probe one URL that must resolve-and-be-annotated and one that must not resolve.

    Without this pair, a run in which **every** request fails reports 660 ``unknown`` — which
    :func:`derive_acquisition_route` already refuses — but a run in which every request
    returns a *parseable manifest missing the GFF* would report 660 ``unannotated`` and
    certify :data:`ROUTE_PRODIGAL`. The positive leg makes that indistinguishable state
    detectable; the negative leg proves the probe can still say "no".
    """
    pos_url = source_urls.get(CONTROL_POSITIVE_ACCESSION, "")
    if not pos_url:
        raise AnnotationSupplyError(
            f"control positive {CONTROL_POSITIVE_ACCESSION} is absent from the fetch report — "
            "the control cannot run, so no route may be certified"
        )
    positive = probe_assembly(
        CONTROL_POSITIVE_ACCESSION, pos_url, limiter=limiter, opener=opener, sleep=sleep
    )
    neg_dir = assembly_dir_url(pos_url) + CONTROL_NEGATIVE_SUFFIX
    neg_manifest, neg_note = fetch_file_manifest(
        neg_dir, limiter=limiter, opener=opener, sleep=sleep
    )
    negative = {
        "url": f"{neg_dir}/{MD5_MANIFEST_NAME}",
        "resolved": neg_manifest is not None,
        "note": neg_note,
    }
    return {
        "positive_accession": CONTROL_POSITIVE_ACCESSION,
        "positive_status": positive["status"],
        "positive_n_files": positive["n_files"],
        "negative": negative,
    }


def control_is_powered(control: Mapping[str, Any]) -> bool:
    """The control discriminated iff the positive is annotated **and** the negative did not
    resolve. Both legs are required: a probe that resolves everything and a probe that
    resolves nothing each satisfy exactly one of them.
    """
    neg = control.get("negative")
    return bool(
        control.get("positive_status") == STATUS_ANNOTATED
        and isinstance(neg, Mapping)
        and neg.get("resolved") is False
    )


# --------------------------------------------------------------------------- #
# The pre-registered route rule
# --------------------------------------------------------------------------- #


def derive_acquisition_route(report: Mapping[str, Any]) -> dict[str, Any]:
    """The measured supply → the acquisition route, by the rule pinned in this module.

    Pre-registered, in this order (CLAUDE.md §10.3 — the rule is a function of counts, fixed
    before the measurement ran):

    1. **Refuse** if the control was not powered, if the sweep did not cover every requested
       assembly, or if any probed assembly is ``unknown``. An unresolved host is not an
       unannotated host, and a truncated sweep is not a measurement.
    2. :data:`ROUTE_NCBI_GFF` if **every** candidate-carrying host is ``annotated``.
    3. :data:`ROUTE_PRODIGAL` if **no** candidate-carrying host is ``annotated``.
    4. :data:`ROUTE_MIXED` otherwise.

    The verdict is keyed to the **candidate-carrying** hosts because those are the assemblies
    a P3-15 round must evaluate (c) on. The full admissible set is reported beside it because
    a later scan-scale (c) — and D4's per-clade (c)-exclusion diagnostic — needs all of them.
    """
    reasons: list[str] = []
    counts = report.get("candidate_host_status_counts")
    all_counts = report.get("admissible_status_counts")
    if not isinstance(counts, Mapping) or not isinstance(all_counts, Mapping):
        return {"route": ROUTE_REFUSED, "reasons": ["report carries no status counts"]}

    if not control_is_powered(report.get("control") or {}):
        reasons.append(
            "control not powered (positive must be annotated, negative must not resolve)"
        )
    if not bool(report.get("sweep_complete")):
        reasons.append("sweep incomplete — not every requested assembly was probed")
    n_cand = int(report.get("n_candidate_hosts", 0))
    n_cand_probed = int(report.get("n_candidate_hosts_probed", -1))
    if n_cand_probed != n_cand:
        reasons.append(
            f"{n_cand - n_cand_probed} candidate-carrying host(s) were not probed — "
            "an unprobed host is not an unannotated host"
        )
    n_unknown = int(all_counts.get(STATUS_UNKNOWN, 0))
    if n_unknown:
        reasons.append(f"{n_unknown} admissible host(s) unresolved — unknown is not unannotated")
    # The candidate counts get their OWN unknown check and their own total reconciliation. In a
    # report this module wrote, the candidate rows are a subset of the probed rows, so a
    # candidate `unknown` implies an admissible `unknown` — but this gate is handed a report,
    # and a gate that leans on an invariant enforced somewhere else is not a gate.
    n_cand_unknown = int(counts.get(STATUS_UNKNOWN, 0))
    if n_cand_unknown:
        reasons.append(
            f"{n_cand_unknown} candidate-carrying host(s) unresolved — "
            "unknown is not unannotated"
        )
    n_total = sum(int(counts.get(k, 0)) for k in STATUS_VALUES)
    if n_total != n_cand:
        reasons.append(
            f"candidate-host status counts sum to {n_total} but n_candidate_hosts is "
            f"{n_cand} — the annotated fraction would be read against the wrong denominator"
        )
    if reasons:
        return {"route": ROUTE_REFUSED, "reasons": reasons}

    n_annotated = int(counts.get(STATUS_ANNOTATED, 0))
    if n_total <= 0:
        return {"route": ROUTE_REFUSED, "reasons": ["zero candidate-carrying hosts"]}
    if n_annotated == n_total:
        route = ROUTE_NCBI_GFF
    elif n_annotated == 0:
        route = ROUTE_PRODIGAL
    else:
        route = ROUTE_MIXED
    return {
        "route": route,
        "reasons": [],
        "n_candidate_hosts_annotated": n_annotated,
        "n_candidate_hosts": n_total,
        "adr_0002_amendment_required": route == ROUTE_PRODIGAL,
    }


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #


def _status_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(STATUS_VALUES, 0)
    for row in rows:
        counts[str(row["status"])] += 1
    return counts


def measure_annotation_supply(
    *,
    admissible: Sequence[str],
    candidate_hosts: Sequence[str],
    source_urls: Mapping[str, str],
    limiter: RateLimiter | None = None,
    workers: int = 4,
    opener: Any = None,
    header: Any = None,
    sleep: Any = None,
    limit: int | None = None,
    measure_sizes: bool = True,
) -> dict[str, Any]:
    """Probe every admissible host and assemble the supply report.

    ``limit`` truncates the sweep for a smoke run. It sets ``sweep_complete: False``, which
    :func:`derive_acquisition_route` **refuses** on — a cost knob may not certify a route
    ([[cost-knobs-can-certify]]).

    ``measure_sizes`` adds a ``Content-Length`` pass over the annotated hosts' GFF URLs, so
    the acquisition ack is sized against measured bytes. It is **cost only**: a size that
    cannot be established is reported as ``n_gff_size_unknown`` and changes no route.
    """
    requested = list(admissible)
    for label, seq in (("admissible", requested), ("candidate_hosts", list(candidate_hosts))):
        if len(set(seq)) != len(seq):
            raise AnnotationSupplyError(
                f"the {label} set carries duplicate accessions — a duplicate is probed twice "
                "and inflates every denominator the route is read against (the mirror of the "
                "silently-dropped accession this module already refuses)"
            )
    not_admissible = sorted(set(candidate_hosts) - set(requested))
    if not_admissible:
        raise AnnotationSupplyError(
            f"{len(not_admissible)} candidate-carrying host(s) are not in the admissible set "
            f"(first: {not_admissible[:3]}) — they would be dropped from the candidate "
            "denominator the route is read against"
        )
    targets = requested[:limit] if limit is not None else requested
    missing = [a for a in targets if a not in source_urls]
    if missing:
        raise AnnotationSupplyError(
            f"{len(missing)} admissible host(s) carry no source_url in the fetch report "
            f"(first: {missing[:3]}) — the sweep would silently understate its denominator"
        )

    control = run_control(source_urls, limiter=limiter, opener=opener, sleep=sleep)

    def _one(acc: str) -> dict[str, Any]:
        return probe_assembly(acc, source_urls[acc], limiter=limiter, opener=opener, sleep=sleep)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_one, targets))
    else:
        rows = [_one(a) for a in targets]

    annotated = [r for r in rows if r["status"] == STATUS_ANNOTATED]
    if measure_sizes and annotated:

        def _size(row: dict[str, Any]) -> int | None:
            return head_content_length(
                str(row["gff_url"]), limiter=limiter, header=header, sleep=sleep
            )

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                sizes = list(pool.map(_size, annotated))
        else:
            sizes = [_size(r) for r in annotated]
        for row, n in zip(annotated, sizes, strict=True):
            row["gff_bytes"] = n

    known = [r["gff_bytes"] for r in annotated if isinstance(r.get("gff_bytes"), int)]

    by_acc = {r["accession"]: r for r in rows}
    cand_rows = [by_acc[a] for a in candidate_hosts if a in by_acc]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": "P3-15'-c-i",
        "n_admissible_requested": len(requested),
        "n_probed": len(rows),
        "sweep_complete": len(rows) == len(requested),
        "accession_prefix_tally_admissible": accession_prefix_tally(requested),
        "accession_prefix_tally_candidate_hosts": accession_prefix_tally(candidate_hosts),
        "admissible_status_counts": _status_counts(rows),
        "n_candidate_hosts": len(candidate_hosts),
        "n_candidate_hosts_probed": len(cand_rows),
        "candidate_host_status_counts": _status_counts(cand_rows),
        "n_faa_present_admissible": sum(1 for r in rows if r["faa_present"]),
        "gff_bytes_total_known": sum(known),
        "n_gff_size_known": len(known),
        "n_gff_size_unknown": len(annotated) - len(known),
        "control": control,
        "per_assembly": sorted(rows, key=lambda r: str(r["accession"])),
    }
    report["route"] = derive_acquisition_route(report)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cmd_measure(args: argparse.Namespace) -> int:
    source_urls = load_source_urls(args.fetch_report)
    from tbox_finder.mining.mine_round import load_admissible_accessions

    admissible = load_admissible_accessions()
    candidate_hosts = candidate_host_accessions(args.fp_manifest)
    outside = sorted(set(candidate_hosts) - set(admissible))
    if outside:
        raise AnnotationSupplyError(
            f"{len(outside)} candidate host(s) are not admissible (first: {outside[:3]})"
        )
    limiter = RateLimiter(RATE_LIMIT_NO_KEY)
    report = measure_annotation_supply(
        admissible=admissible,
        candidate_hosts=candidate_hosts,
        source_urls=source_urls,
        limiter=limiter,
        workers=args.workers,
        limit=args.limit,
        measure_sizes=args.measure_sizes,
    )
    report["provenance"] = provenance.build_provenance(
        rule="src/tbox_finder/mining/annotation_supply.py :: measure",
        script="src/tbox_finder/mining/annotation_supply.py",
        inputs=[args.fetch_report, args.fp_manifest],
        adr="ADR-0006",
        extra={
            "accessed": date.today().isoformat(),
            "genome_ftp_host": "ftp.ncbi.nlm.nih.gov",
            "manifest_name": MD5_MANIFEST_NAME,
            "gff_suffix": GFF_SUFFIX,
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    route = report["route"]
    print(
        f"annotation-supply: admissible {report['admissible_status_counts']} | "
        f"candidate hosts {report['candidate_host_status_counts']} | "
        f"route={route['route']} → {out}"
    )
    if route["route"] == ROUTE_REFUSED:
        print(f"REFUSED: {'; '.join(route['reasons'])}", file=sys.stderr)
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annotation-supply",
        description="P3-15'-c-i — measure the NCBI annotation supply for criterion (c).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="probe every admissible host → the supply report")
    m.add_argument("--fetch-report", default=DEFAULT_FETCH_REPORT)
    m.add_argument("--fp-manifest", default="data/processed/mining/round0_fp_manifest.json")
    m.add_argument("--out", default="reports/p3/annotation_supply.json")
    m.add_argument("--workers", type=int, default=4)
    m.add_argument(
        "--measure-sizes",
        dest="measure_sizes",
        action="store_true",
        default=True,
        help="HEAD each annotated host's GFF for its Content-Length (default: on).",
    )
    m.add_argument(
        "--no-measure-sizes",
        dest="measure_sizes",
        action="store_false",
        help="skip the size pass. Cost only — it changes no route, and the report then "
        "carries n_gff_size_unknown == n_annotated rather than an invented total.",
    )
    m.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: probe the first N admissible hosts. Sets sweep_complete=False, "
        "which the route derivation REFUSES on — a cost knob may not certify a route.",
    )
    m.set_defaults(func=_cmd_measure)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except AnnotationSupplyError as exc:
        print(f"annotation-supply: REFUSED — {exc}", file=sys.stderr)
        return 3


__all__ = [
    "ACCESSION_RE",
    "AnnotationSupplyError",
    "GFF_SUFFIX",
    "MD5_MANIFEST_NAME",
    "PROTEIN_FAA_SUFFIX",
    "ROUTE_MIXED",
    "ROUTE_NCBI_GFF",
    "ROUTE_PRODIGAL",
    "ROUTE_REFUSED",
    "SCHEMA_VERSION",
    "STATUS_ANNOTATED",
    "STATUS_UNANNOTATED",
    "STATUS_UNKNOWN",
    "STATUS_VALUES",
    "accession_prefix_tally",
    "assembly_basename",
    "assembly_dir_url",
    "candidate_host_accessions",
    "classify_assembly",
    "control_is_powered",
    "derive_acquisition_route",
    "fetch_file_manifest",
    "load_source_urls",
    "main",
    "measure_annotation_supply",
    "parse_md5_manifest",
    "probe_assembly",
    "run_control",
    "sibling_url",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
