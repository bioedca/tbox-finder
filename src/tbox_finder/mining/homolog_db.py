"""homolog_db.py — build the model-independent homolog-search target DB (P2-10c′-homologdb).

ADR-0006 **D7** pins the homolog set that populates each candidate's de-novo R-scape MSA to a
**model-independent sequence-homology search — nhmmer / BLAST / cmsearch-from-candidate — not
model-grouped membership**. Amendment **A1** names the target that search runs *against*: the
2,500 GTDB R232 species-representative genomes fetched at ``data/interim/production_genomes/``
(P2-10c′-fetch). D7 names the *method* and A1 the *identity*; **this module builds the searchable
target DB** — the last machinery blocker before a per-candidate ``(a)`` covariation verdict.

The candidate is the **query**; the 2,500-genome set is the **target sequence DB**. From the one
concatenated target FASTA this builds the index each named D7 route needs:

* ``makeblastdb`` → a nucleotide BLAST DB (``blastn`` searches it),
* ``makehmmerdb`` → the binary FM-index that makes ``nhmmer`` tractable over ~7 Gbp,

and it keeps the concatenated FASTA itself as the ``cmsearch``-from-candidate target (and the
plain-FASTA ``nhmmer`` fallback). Contig headers are rewritten to ``<accession>:c<contig_index>``
(the :func:`tbox_finder.mining.substrate_windows.window_name` contig scheme, minus the ``:start``)
so every hit's subject name traces back to its source genome — a same-genome self-hit is
identifiable, which the downstream homolog-set assembly (P6-07) needs.

**Two-independent-reads integrity, not "the file exists".** The concatenation re-measures every
genome's contig count and base total from the FASTA it reads *now* and asserts they equal the
git-frozen per-genome baseline recorded FASTA-independently at build-manifest time
(``production_windows_report.json``; P2-10c′-shardspec). A silently dropped or truncated genome
turns that equality FALSE (MEMORY: a control's matchedness must itself be asserted; measure a
matched baseline before certifying). ``blastdbcmd -info`` is then cross-checked against the same
measured totals, so a DB that indexed less than the whole target fails loud.

**A designed must-fire round-trip control.** Certifying the DB is more than "files are non-empty":
a real sub-sequence of the **first and last** (accession-sorted) indexed genome must be recovered
**at its own ``<accession>:c<ci>``** by ``blastn`` *and* ``nhmmer``. It is a sequence known to be in
the DB, so self-recovery MUST fire — a build that dropped a genome, corrupted the FM-index, or
mis-named a subject fails the control (MEMORY: never gate on a quantity whose legitimate value is
zero — use a designed control that must fire). Probing both ends catches a truncated concatenation.

**Honesty (CLAUDE.md §10.3).** This *builds an index and proves it searchable*. It runs **no**
genome scan, calls **no** ``cmsearch``, counts **no** candidates, assembles **no** homolog set, and
pins **no** D7 inclusion threshold — those E-value / bit-score cutoffs are **P6-frozen (ADR-0006
D17)** and out of scope here. A1 is signed; this is *execution* of a frozen pin, not a new decision.

Runs in ``envs/homology.yml`` (``tbox-homology`` on the cluster: hmmer 3.4 + blast 2.17.0 +
biopython; **no pandas/pyarrow**), so the frozen accession set and the per-genome integrity
baseline are read from the stdlib-JSON ``production_windows_report.json`` — never the parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tbox_finder import provenance

# Reuse (promote-don't-duplicate): the repo's contig-aware FASTA reader + normalized renderer.
# pilot_fetch is pandas-free at import (its manifest read defers pandas inside a function), so it
# imports cleanly in the minimal tbox-homology env. The genome-FASTA read + the target-name scheme
# are inlined here (a 3-line read + an f-string) rather than imported from substrate_windows, which
# pulls window_dataset (numpy/torch) — an import that env has no wheels for.
from tbox_finder.mining.pilot_fetch import FASTA_WRAP, iter_fasta_records, normalize_fasta

# ── Schema / provenance ──────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"
STEP = "P2-10c'-homologdb"
ADR = "ADR-0006"  # D7 (method) + A1 (target-DB identity); thresholds are P6-frozen (D17)

# ── Canonical artifact paths (one path per artifact) ─────────────────────────
GENOME_DIR = "data/interim/production_genomes"
BASELINE_REPORT = "data/processed/audits/production_windows_report.json"
DB_DIR = "data/interim/homolog_db"
TARGET_FASTA_NAME = "target.fna"  # concatenated, header-rewritten target (nhmmer/cmsearch target)
BLAST_PREFIX_NAME = "target"  # makeblastdb -out <DB_DIR>/target → target.n{hr,in,sq,...}
FMINDEX_NAME = "target.fm"  # makehmmerdb binary FM-index (nhmmer target)
REPORT = "data/processed/audits/homolog_db_report.json"
PROVENANCE = "data/interim/homolog_db.provenance.json"

RULE = "slurm/p2/build_homolog_db.sbatch :: python -m tbox_finder.mining.homolog_db build"
SCRIPT = "src/tbox_finder/mining/homolog_db.py"

# ── ADR-delegated pins (not implementer choices) ─────────────────────────────
#: ADR-0006 A1 pins the target set at 2,500 GTDB R232 species reps. The driver asserts the frozen
#: baseline carries exactly this many genomes — a truncated baseline report cannot certify a build.
EXPECTED_N_GENOMES = 2500
#: Pinned tool major.minor from envs/homology.yml (ADR-0002 A12). Asserted where the tools are USED
#: (the A11 lesson: a version pin asserted nowhere it runs is not asserted) — a silent env drift to
#: a different BLAST/HMMER build fails the run rather than certifying an artifact by the wrong tool.
EXPECTED_BLAST_VERSION_PREFIX = "2.17.0"
EXPECTED_HMMER_VERSION_PREFIX = "3.4"

# ── Round-trip control geometry ──────────────────────────────────────────────
PROBE_LEN = 500  # a real sub-sequence long enough for an unambiguous self-hit
MIN_SELF_IDENTITY = 99.0  # blastn pident floor for a self-recovery (a real subseq is ~100%)
MIN_SELF_LEN_FRAC = 0.90  # aligned length must cover ≥ 90 % of the probe
SELF_HIT_MAX_EVALUE = 1e-5  # both engines must place the self-hit at least this significant

#: The blastn -outfmt 6 column order this module requests and parses (kept in one place so the
#: request and the parser cannot drift).
BLAST_OUTFMT_COLUMNS = ("qseqid", "sseqid", "pident", "length", "evalue", "bitscore")

NOTES = (
    "P2-10c′-homologdb (ADR-0006 D7 + A1): build the model-independent homolog-search TARGET DB "
    "over the 2,500 GTDB R232 reps — concatenated target FASTA (headers rewritten to "
    "<accession>:c<contig_index>) + makeblastdb nucleotide DB + makehmmerdb FM-index, the targets "
    "blastn / nhmmer / cmsearch-from-candidate search a candidate against (D7). Integrity is a "
    "two-independent-reads check (re-measured per-genome contig/bp == the git-frozen "
    "production_windows_report baseline; blastdbcmd -info == the same measured totals) plus a "
    "must-fire round-trip control (first+last accession-sorted genome self-recovers at its own "
    "<accession>:c<ci> via blastn AND nhmmer). Builds an index and proves it searchable — no "
    "genome scan, no cmsearch, no candidate count, no homolog-set assembly, no D7 threshold "
    "(P6-frozen, D17). Pins no ADR value beyond A1's already-signed source+count."
)


class HomologDbError(RuntimeError):
    """Raised on a malformed genome, a broken integrity invariant, a version drift, or a failed
    round-trip self-recovery — every path that must refuse to certify the DB."""


class ToolTimeoutError(HomologDbError):
    """Raised by :func:`_run` when a tool exceeds its caller-supplied wall-clock bound.

    A distinct type, not a bare :class:`HomologDbError`, because the two mean different things to a
    fail-closed caller: a non-zero exit is *the tool answering "no"*, while a timeout is *no answer
    at all*. `covariation_producer.align_shard` maps this one to ``reason="align_timeout"`` and
    writes no MSA, so the score stage reads ``unavailable`` ⇒ **spared** (ADR-0005 D14).
    """


# ═════════════════════════════════════════════════════════════════════════════
# Frozen accession set + per-genome integrity baseline (stdlib JSON — no pandas)
# ═════════════════════════════════════════════════════════════════════════════
def load_baseline(
    report_path: str | Path = BASELINE_REPORT,
) -> tuple[dict[str, dict[str, int]], int, int]:
    """``({accession: {n_contigs, total_bp}}, total_bp, n_genomes)`` from the window-count report.

    This report is the FASTA-independent per-genome record written at build-manifest time
    (P2-10c′-shardspec); reading it here (a *second, independent* measurement basis) is what makes
    the concatenation's re-measurement a real integrity check rather than a self-comparison. It is
    git-tracked stdlib JSON, so it needs no pandas — the tbox-homology env has none.
    """
    try:
        data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise HomologDbError(f"baseline report {report_path} is unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HomologDbError(f"baseline report {report_path} is not valid JSON: {exc}") from exc
    per_genome_rows = data.get("per_genome")
    if not per_genome_rows:
        raise HomologDbError(f"baseline report {report_path} carries no per_genome rows")
    baseline: dict[str, dict[str, int]] = {}
    try:
        for row in per_genome_rows:
            acc = str(row["assembly_accession"])
            if acc in baseline:
                raise HomologDbError(f"duplicate accession {acc!r} in baseline report")
            baseline[acc] = {"n_contigs": int(row["n_contigs"]), "total_bp": int(row["total_bp"])}
        total_bp = int(data["total_bp"])
        n_genomes = int(data["n_genomes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HomologDbError(f"malformed baseline report {report_path}: {exc}") from exc
    if n_genomes != len(baseline):
        raise HomologDbError(
            f"baseline n_genomes ({n_genomes}) != per_genome rows ({len(baseline)})"
        )
    if sum(r["total_bp"] for r in baseline.values()) != total_bp:
        raise HomologDbError("baseline aggregate total_bp != Σ per-genome total_bp")
    return baseline, total_bp, n_genomes


# ═════════════════════════════════════════════════════════════════════════════
# Target FASTA concatenation + integrity
# ═════════════════════════════════════════════════════════════════════════════
def target_seq_name(accession: str, contig_index: int) -> str:
    """A globally-unique, whitespace-free target-sequence name (the blastn/nhmmer subject id).

    ``<accession>:c<contig_index>`` — the accession prefix is unique across genomes, the contig
    index unique within a genome. ``contig_index`` is the position in the genome's *full* record
    list (empty contigs consume an index but are not emitted), so it matches
    :func:`substrate_windows.window_name`'s contig numbering exactly.
    """
    return f"{accession}:c{contig_index}"


def read_genome_fasta(genome_dir: str | Path, accession: str) -> str:
    """Read one fetched genome FASTA (``<genome_dir>/<accession>.fna``); fail loud if absent.

    Mirrors :func:`substrate_windows.read_genome_fasta`; inlined to keep this module's import graph
    free of window_dataset (numpy/torch), which the tbox-homology env cannot import.
    """
    path = Path(genome_dir) / f"{accession}.fna"
    if not path.is_file():
        raise HomologDbError(
            f"genome FASTA missing for baseline accession {accession!r}: {path} — the indexed set "
            "must equal the git-frozen selection; a missing genome is a silent-shrink failure."
        )
    return path.read_text(encoding="utf-8")


def concatenate_target_fasta(
    genome_dir: str | Path, accessions: Sequence[str], out_fasta: str | Path
) -> dict[str, Any]:
    """Concatenate the genomes into one header-rewritten target FASTA; return the re-measured stats.

    Streams one genome at a time (never holds > one genome's records in memory). Empty contigs are
    skipped (they contribute no window and no searchable target), so the re-measured ``n_contigs`` /
    ``total_bp`` line up with the baseline, which was counted the same way. Returns
    ``{per_genome: {acc: {n_contigs, total_bp}}, n_sequences, total_bp}``.
    """
    out_fasta = Path(out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    per_genome: dict[str, dict[str, int]] = {}
    n_sequences = 0
    total_bp = 0
    with out_fasta.open("w", encoding="utf-8") as fh:
        for acc in accessions:
            records = iter_fasta_records(read_genome_fasta(genome_dir, acc))
            renamed: list[tuple[str, str]] = []
            g_contigs = 0
            g_bp = 0
            for ci, (header, seq) in enumerate(records):
                if not seq:
                    continue
                # Keep the original defline as the (space-separated) title for downstream
                # traceability; the first token — the rewritten name — is the subject id.
                renamed.append((f"{target_seq_name(acc, ci)} {header}", seq))
                g_contigs += 1
                g_bp += len(seq)
            if g_contigs == 0:
                raise HomologDbError(f"genome {acc} has no non-empty contig — cannot index it")
            fh.write(normalize_fasta(renamed, wrap=FASTA_WRAP))
            per_genome[acc] = {"n_contigs": g_contigs, "total_bp": g_bp}
            n_sequences += g_contigs
            total_bp += g_bp
    return {"per_genome": per_genome, "n_sequences": n_sequences, "total_bp": total_bp}


def verify_integrity(
    measured: dict[str, Any], baseline: dict[str, dict[str, int]], baseline_total_bp: int
) -> None:
    """Fail-closed: the re-measured target must equal the git-frozen baseline, genome by genome.

    Checks (any failure raises): the accession set matches exactly (no silent drop, no stray extra);
    every genome's ``n_contigs`` and ``total_bp`` match; the aggregate ``total_bp`` matches. This is
    the two-independent-reads guard — the baseline was measured FASTA-independently upstream, so a
    truncated/altered genome the concatenation reads now diverges here.
    """
    measured_pg: dict[str, dict[str, int]] = measured["per_genome"]
    m_set, b_set = set(measured_pg), set(baseline)
    if m_set != b_set:
        missing = sorted(b_set - m_set)[:3]
        extra = sorted(m_set - b_set)[:3]
        raise HomologDbError(
            f"indexed accession set != baseline: {len(b_set - m_set)} missing (e.g. {missing}), "
            f"{len(m_set - b_set)} extra (e.g. {extra})"
        )
    for acc in sorted(baseline):
        m, b = measured_pg[acc], baseline[acc]
        if m["n_contigs"] != b["n_contigs"] or m["total_bp"] != b["total_bp"]:
            raise HomologDbError(
                f"genome {acc} re-measured {m} != baseline {b} — a truncated or altered FASTA "
                "since build-manifest, or a stale genome dir."
            )
    if int(measured["total_bp"]) != int(baseline_total_bp):
        raise HomologDbError(
            f"aggregate total_bp re-measured {measured['total_bp']} != baseline {baseline_total_bp}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Tool version + tabular-output parsers (binary-free, unit-tested)
# ═════════════════════════════════════════════════════════════════════════════
def parse_makeblastdb_version(text: str) -> str:
    """Extract e.g. ``2.17.0`` from ``makeblastdb -version`` (the first ``makeblastdb: X`` line)."""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("makeblastdb:"):
            token = line.split(":", 1)[1].strip().split()[0]
            return token.rstrip("+")
    raise HomologDbError(f"could not parse makeblastdb version from: {text!r}")


def parse_hmmer_version(text: str) -> str:
    """Extract e.g. ``3.4`` from an HMMER banner line (``# HMMER 3.4 (...)`` or ``HMMER 3.4``)."""
    for line in text.splitlines():
        tokens = line.replace("#", " ").split()
        for i, tok in enumerate(tokens[:-1]):
            if tok.upper() == "HMMER":
                return tokens[i + 1]
    raise HomologDbError(f"could not parse HMMER version from: {text!r}")


def assert_tool_version(tool: str, actual: str, expected_prefix: str) -> None:
    """Raise unless ``actual`` starts with the pinned ``expected_prefix`` (ADR-0002 A12)."""
    if not actual.startswith(expected_prefix):
        raise HomologDbError(
            f"{tool} version {actual!r} != pinned {expected_prefix!r} (envs/homology.yml, A12) — "
            "the DB would be built by the wrong tool build; refusing to certify."
        )


def parse_blast_tabular(
    text: str, columns: Sequence[str] = BLAST_OUTFMT_COLUMNS
) -> list[dict[str, str]]:
    """Parse ``blastn -outfmt 6`` (tab-separated) into ``[{column: value}]`` rows; ``#`` skipped."""
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < len(columns):
            raise HomologDbError(
                f"blast tabular row has {len(fields)} fields, need {len(columns)}: {line!r}"
            )
        rows.append(dict(zip(columns, fields, strict=False)))
    return rows


def parse_nhmmer_tblout(text: str) -> list[dict[str, str]]:
    """Parse ``nhmmer --tblout`` into ``[{target, evalue, score}]`` (comment/``#`` lines skipped).

    The target name is field 0 and the E-value field 12 (0-indexed) in the fixed leading columns;
    the trailing free-text description never shifts them, so positional indexing is safe.
    """
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 15:
            raise HomologDbError(f"nhmmer tblout row has {len(fields)} fields (need ≥15): {line!r}")
        rows.append({"target": fields[0], "evalue": fields[12], "score": fields[13]})
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# The must-fire round-trip self-recovery control
# ═════════════════════════════════════════════════════════════════════════════
_ACGT = frozenset("ACGT")


def extract_probe(fasta_text: str, probe_len: int = PROBE_LEN) -> tuple[int, int, str]:
    """``(contig_index, offset, subsequence)`` for a deterministic **pure-ACGT** probe.

    ``contig_index`` is the position in the genome's *full* record list (matching
    :func:`target_seq_name`). The offset is deterministic (no RNG — the control is reproducible):
    a fixed mid-contig start, then scanned mid→end then head→mid in ``probe_len`` steps for the
    first window with **no** ambiguity codes / ``N``-runs. A quarter-point window can land in an
    assembly gap (a run of ``N``) or an ambiguity stretch that blastn/nhmmer would not self-recover
    — a *spurious* round-trip failure — so a clean, uniquely-alignable window is required. Raises if
    no contig yields a pure-ACGT ``probe_len`` window (a genome that degenerate should fail loud).
    """
    for ci, (_header, seq) in enumerate(iter_fasta_records(fasta_text)):
        if len(seq) < probe_len:
            continue
        last = len(seq) - probe_len
        start0 = min(max(len(seq) // 4, 0), last)
        # Non-overlapping probe_len steps (bounded work on a huge contig), deterministic order.
        for offset in [*range(start0, last + 1, probe_len), *range(0, start0, probe_len)]:
            window = seq[offset : offset + probe_len]
            if set(window) <= _ACGT:
                return ci, offset, window
    raise HomologDbError(f"no pure-ACGT contig window ≥ {probe_len} nt for probe extraction")


def assert_self_recovery(
    *,
    accession: str,
    contig_index: int,
    probe_len: int,
    blast_hits: Sequence[dict[str, str]],
    nhmmer_hits: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Assert a probe self-recovered at its own ``<accession>:c<ci>`` by both engines; give summary.

    The probe is a real sub-sequence of an indexed genome, so a correctly-built, searchable DB MUST
    return it at its own subject name at ~100 % identity. Absence (empty hits, wrong subject, or
    below-floor identity/E-value) is a build failure — this is the designed control that must fire.
    """
    want = target_seq_name(accession, contig_index)

    blast_ok = None
    for h in blast_hits:
        if (
            h.get("sseqid") == want
            and float(h["pident"]) >= MIN_SELF_IDENTITY
            and int(h["length"]) >= probe_len * MIN_SELF_LEN_FRAC
            and float(h["evalue"]) <= SELF_HIT_MAX_EVALUE
        ):
            blast_ok = h
            break
    if blast_ok is None:
        raise HomologDbError(
            f"blastn did not self-recover {want} (probe_len={probe_len}; need "
            f"pident≥{MIN_SELF_IDENTITY}, len≥{probe_len * MIN_SELF_LEN_FRAC:.0f}, "
            f"E≤{SELF_HIT_MAX_EVALUE}); hits={list(blast_hits)[:3]}"
        )

    nhmmer_ok = None
    for h in nhmmer_hits:
        if h.get("target") == want and float(h["evalue"]) <= SELF_HIT_MAX_EVALUE:
            nhmmer_ok = h
            break
    if nhmmer_ok is None:
        raise HomologDbError(
            f"nhmmer did not self-recover {want} (need E≤{SELF_HIT_MAX_EVALUE}); "
            f"hits={list(nhmmer_hits)[:3]}"
        )
    return {
        "accession": accession,
        "target": want,
        "probe_len": probe_len,
        "blastn": {
            "pident": blast_ok["pident"],
            "length": blast_ok["length"],
            "evalue": blast_ok["evalue"],
        },
        "nhmmer": {"evalue": nhmmer_ok["evalue"], "score": nhmmer_ok["score"]},
        "recovered": True,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Thin binary wrappers (cluster-only — exercised by the in-env smoke + the real build)
# ═════════════════════════════════════════════════════════════════════════════
def tool_path(name: str) -> str:
    """Resolve a required binary on PATH or fail loud (no override — the A11 ``shutil.which`` rule).

    A missing tool must never be silently substituted or skipped: the round-trip control would then
    have nothing to run and could read as "no hits" = a false green.
    """
    path = shutil.which(name)
    if path is None:
        raise HomologDbError(
            f"required binary {name!r} not on PATH — is the tbox-homology env active?"
        )
    return path


def _run(cmd: Sequence[str], *, timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run a checked subprocess, capturing text output; raise HomologDbError on non-zero exit.

    ``timeout_s`` bounds the call's wall clock; :class:`ToolTimeoutError` is raised if it is
    exceeded. ``None`` (the default) is the unbounded behaviour every caller had before
    P3-15'-e-ii, so the certified search/index paths run byte-identically — only the align stage
    passes a bound, and it passes one explicitly.
    """
    if timeout_s is not None and timeout_s <= 0:
        # A 0/negative bound would time out *every* invocation, and because a timeout is spared
        # rather than fatal (ADR-0005 D14) that failure is silent: a full round would report
        # `unavailable` for all 941 candidates and look like a clean, producible-nothing run.
        raise ValueError(f"timeout_s must be a positive number of seconds, got {timeout_s!r}")
    if timeout_s is None:
        proc = subprocess.run(
            cmd, capture_output=True, text=True
        )  # noqa: S603 (tool_path-resolved argv)
    else:
        proc = _run_bounded(cmd, timeout_s)
    if proc.returncode != 0:
        raise HomologDbError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc


def _run_bounded(cmd: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    """:func:`_run`'s bounded arm — the child runs in **its own process group** and a timeout
    ``SIGKILL``s the whole group.

    ``subprocess.run(timeout=...)`` reaps only the process it spawned. That is not enough here:
    ``mlocarna`` is a Perl driver that dispatches work to ``locarna`` children (``--threads`` is a
    real parallelism knob), so killing the driver alone would orphan those children and leave them
    burning the node's cores — the exact cost this bound exists to stop, minus the process holding
    the shard open. ``start_new_session=True`` makes the child a session/group leader so
    :func:`os.killpg` can take the whole tree down (POSIX; this pipeline is Linux-only, §9.2).
    """
    with subprocess.Popen(  # noqa: S603 (tool_path-resolved argv, no shell)
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as popen:
        try:
            stdout, stderr = popen.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_group(popen)
            stdout, stderr = popen.communicate()  # drain the pipes of the now-dead group
            raise ToolTimeoutError(
                f"command exceeded its {timeout_s:g}s bound and was killed: {' '.join(cmd)}"
            ) from None
    return subprocess.CompletedProcess(list(cmd), popen.returncode, stdout, stderr)


def _kill_process_group(popen: subprocess.Popen[str]) -> None:
    """``SIGKILL`` the timed-out child's whole process group, falling back to the child alone.

    The self-group check is a **safety interlock, not defensive noise**: this only kills a group
    because :func:`_run_bounded` put the child in a fresh one. If ``start_new_session=True`` were
    ever dropped, the child would share *our* process group and this ``killpg`` would SIGKILL the
    array task that called it — turning a spared candidate into a dead shard, the precise failure
    the bound exists to prevent. Refuse that and kill only the child.
    """
    try:
        pgid = os.getpgid(popen.pid)
    except (ProcessLookupError, PermissionError):
        popen.kill()  # already gone (or not ours) — the direct child is still ours to reap
        return
    if pgid == os.getpgrp():
        popen.kill()
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        popen.kill()


def run_makeblastdb(target_fasta: str | Path, prefix: str | Path, title: str) -> None:
    _run(
        [
            tool_path("makeblastdb"),
            "-in",
            str(target_fasta),
            "-dbtype",
            "nucl",
            "-out",
            str(prefix),
            "-title",
            title,
        ]
    )


def run_blastdbcmd_info(prefix: str | Path) -> tuple[int, int]:
    """``(n_sequences, total_bases)`` reported by ``blastdbcmd -db <prefix> -info``."""
    text = _run([tool_path("blastdbcmd"), "-db", str(prefix), "-info"]).stdout
    n_seq = n_bases = None
    for line in text.splitlines():
        s = line.strip()
        # e.g. "1,234 sequences; 5,678,900 total bases"
        if "sequences;" in s and "total bases" in s:
            left, right = s.split("sequences;", 1)
            n_seq = int(left.replace(",", "").strip())
            n_bases = int(right.replace("total bases", "").replace(",", "").strip())
            break
    if n_seq is None or n_bases is None:
        raise HomologDbError(f"could not parse blastdbcmd -info sequence/base counts from:\n{text}")
    return n_seq, n_bases


def run_makehmmerdb(target_fasta: str | Path, fm_out: str | Path) -> None:
    _run([tool_path("makehmmerdb"), str(target_fasta), str(fm_out)])


def run_blastn(query_fasta: str | Path, prefix: str | Path) -> list[dict[str, str]]:
    outfmt = "6 " + " ".join(BLAST_OUTFMT_COLUMNS)
    text = _run(
        [
            tool_path("blastn"),
            "-query",
            str(query_fasta),
            "-db",
            str(prefix),
            "-outfmt",
            outfmt,
            "-max_target_seqs",
            "5",
            "-evalue",
            str(SELF_HIT_MAX_EVALUE),
        ]
    ).stdout
    return parse_blast_tabular(text)


def run_nhmmer(
    query_fasta: str | Path, fm_index: str | Path, tblout: str | Path
) -> list[dict[str, str]]:
    tblout = Path(tblout)
    _run(
        [
            tool_path("nhmmer"),
            "--tblout",
            str(tblout),
            "--noali",
            "-E",
            str(SELF_HIT_MAX_EVALUE),
            "-o",
            "/dev/null",
            str(query_fasta),
            str(fm_index),
        ]
    )
    return parse_nhmmer_tblout(tblout.read_text(encoding="utf-8"))


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator (cluster-only) + report
# ═════════════════════════════════════════════════════════════════════════════
def build(
    *,
    genome_dir: str | Path = GENOME_DIR,
    baseline_report: str | Path = BASELINE_REPORT,
    out_dir: str | Path = DB_DIR,
    report_path: str | Path = REPORT,
    provenance_path: str | Path = PROVENANCE,
    env_lock: str | None = None,
    expected_n_genomes: int | None = EXPECTED_N_GENOMES,
    probe_len: int = PROBE_LEN,
) -> dict[str, Any]:
    """Concatenate → index (BLAST + FM) → integrity + round-trip control → report + provenance.

    Writes the report/provenance **only** after every gate passes (fail-closed): a failed run leaves
    no certified artifact, so the §9.3 artifact-based verify sees no green. Returns the report dict.
    """
    out_dir = Path(out_dir)
    target_fasta = out_dir / TARGET_FASTA_NAME
    blast_prefix = out_dir / BLAST_PREFIX_NAME
    fm_index = out_dir / FMINDEX_NAME

    baseline, baseline_total_bp, baseline_n = load_baseline(baseline_report)
    if expected_n_genomes is not None and baseline_n != expected_n_genomes:
        raise HomologDbError(
            f"baseline carries {baseline_n} genomes != ADR-0006 A1's pinned {expected_n_genomes}"
        )
    accessions = sorted(baseline)

    # Silent-shrink guard: the fetched genome dir must hold exactly the frozen accession set.
    on_disk = {p.stem for p in Path(genome_dir).glob("*.fna")}
    if on_disk != set(accessions):
        raise HomologDbError(
            f"genome dir accession set != baseline ({len(set(accessions) - on_disk)} missing, "
            f"{len(on_disk - set(accessions))} extra) — stage the full fetched set before indexing."
        )

    measured = concatenate_target_fasta(genome_dir, accessions, target_fasta)
    verify_integrity(measured, baseline, baseline_total_bp)

    # BLAST DB + cross-check it indexed the whole target.
    run_makeblastdb(
        target_fasta, blast_prefix, title=f"{STEP} homolog-search target (n={baseline_n})"
    )
    db_n_seq, db_n_bases = run_blastdbcmd_info(blast_prefix)
    if (db_n_seq, db_n_bases) != (measured["n_sequences"], measured["total_bp"]):
        raise HomologDbError(
            f"blastdbcmd -info ({db_n_seq} seq, {db_n_bases} bp) != concatenated target "
            f"({measured['n_sequences']} seq, {measured['total_bp']} bp) — the DB indexed a subset."
        )

    # nhmmer FM-index.
    run_makehmmerdb(target_fasta, fm_index)
    if not fm_index.is_file() or fm_index.stat().st_size == 0:
        raise HomologDbError(f"makehmmerdb produced no FM-index at {fm_index}")

    # Assert the pinned tool versions where they ran (A12). Both `-version`/`-h` exit 0 (verified
    # in-env), so the checked `_run` helper is the right transport — a non-zero here means a broken
    # binary and should fail the build, not silently yield an unparseable banner.
    blast_version = parse_makeblastdb_version(_run([tool_path("makeblastdb"), "-version"]).stdout)
    hmmer_version = parse_hmmer_version(_run([tool_path("nhmmer"), "-h"]).stdout)
    assert_tool_version("blast", blast_version, EXPECTED_BLAST_VERSION_PREFIX)
    assert_tool_version("hmmer", hmmer_version, EXPECTED_HMMER_VERSION_PREFIX)

    # Must-fire round-trip control: first + last accession-sorted genome self-recovers by both.
    probes: list[dict[str, Any]] = []
    # dict.fromkeys dedups: a 1-genome baseline has accessions[0] == accessions[-1] → one probe.
    for acc in dict.fromkeys((accessions[0], accessions[-1])):
        ci, offset, sub = extract_probe(read_genome_fasta(genome_dir, acc), probe_len)
        query = out_dir / f"probe_{acc}.fna"
        tbl = out_dir / f"probe_{acc}.tbl"
        query.write_text(f">probe_{acc}:c{ci}:{offset}\n{sub}\n", encoding="utf-8")
        try:
            blast_hits = run_blastn(query, blast_prefix)
            nhmmer_hits = run_nhmmer(query, fm_index, tbl)
            probes.append(
                assert_self_recovery(
                    accession=acc,
                    contig_index=ci,
                    probe_len=len(sub),
                    blast_hits=blast_hits,
                    nhmmer_hits=nhmmer_hits,
                )
            )
        finally:  # scratch files always cleaned, even if a self-recovery assertion raises
            query.unlink(missing_ok=True)
            tbl.unlink(missing_ok=True)

    report = build_report(
        baseline_n=baseline_n,
        measured=measured,
        db_counts=(db_n_seq, db_n_bases),
        fm_index_bytes=fm_index.stat().st_size,
        blast_version=blast_version,
        hmmer_version=hmmer_version,
        probes=probes,
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    # provenance HASHES its outputs (provenance.sha256_paths), so they must be the paths where the
    # files actually exist NOW — the node-local out_dir the driver wrote to, not the canonical
    # DB_DIR the sbatch promotes to AFTER this returns (that path does not exist yet; hashing it
    # would raise). build_report's string labels differ — they hash nothing, so they name DB_DIR.
    provenance.write_provenance(
        provenance_path,
        rule=RULE,
        script=SCRIPT,
        seed=provenance.DEFAULT_SEED,
        inputs=[str(baseline_report)],
        outputs=[str(target_fasta), str(report_path)],
        env_lock=env_lock,
        adr=ADR,
        extra={
            "step": STEP,
            "notes": NOTES,
            "genome_dir": str(genome_dir),
            "db_dir": str(out_dir),
            "n_genomes": baseline_n,
            "blast_version": blast_version,
            "hmmer_version": hmmer_version,
        },
    )
    return report


def build_report(
    *,
    baseline_n: int,
    measured: dict[str, Any],
    db_counts: tuple[int, int],
    fm_index_bytes: int,
    blast_version: str,
    hmmer_version: str,
    probes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """The committed audit: what was indexed, by which tool builds, and the round-trip evidence.

    The ``blastdb.prefix`` / ``fmindex.path`` / ``target_fasta`` fields are the DB's **canonical
    committed location** (``DB_DIR`` = ``data/interim/homolog_db/``) — where the sbatch **promotes**
    the DVC-tracked DB — **not** the transient node-local ``out_dir`` the build wrote to. This is
    deliberate: a reader of the committed report (or its downstream P6-07 consumer) needs the path
    the artifact rests at, so threading the ephemeral ``/tmp/$USER-$JOB/…`` ``out_dir`` here would
    record a path that no longer exists after the job. ``DB_DIR`` is the one true resting place.
    """
    if not probes:
        raise HomologDbError("no round-trip probe recovered — refusing to certify an unproven DB")
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "n_genomes": int(baseline_n),
        "n_sequences": int(measured["n_sequences"]),
        "total_bp": int(measured["total_bp"]),
        "blastdb": {
            "n_sequences": int(db_counts[0]),
            "total_bases": int(db_counts[1]),
            "prefix": f"{DB_DIR}/{BLAST_PREFIX_NAME}",
        },
        "fmindex": {"path": f"{DB_DIR}/{FMINDEX_NAME}", "bytes": int(fm_index_bytes)},
        "target_fasta": f"{DB_DIR}/{TARGET_FASTA_NAME}",
        "blast_version": blast_version,
        "hmmer_version": hmmer_version,
        "round_trip": {
            "probes": list(probes),
            "all_recovered": all(p["recovered"] for p in probes),
        },
        # Honesty flags (§10.3): this step builds+proves an index and nothing more.
        "builds_index_only": True,
        "runs_no_genome_scan": True,
        "runs_no_cmsearch": True,
        "counts_no_candidates": True,
        "assembles_no_homolog_set": True,
        "pins_no_inclusion_threshold": True,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbox_finder.mining.homolog_db")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser(
        "build", help="build the BLAST DB + nhmmer FM-index over the fetched genomes"
    )
    b.add_argument("--genome-dir", default=GENOME_DIR)
    b.add_argument("--baseline-report", default=BASELINE_REPORT)
    b.add_argument("--out-dir", default=DB_DIR)
    b.add_argument("--report", default=REPORT)
    b.add_argument("--provenance", default=PROVENANCE)
    b.add_argument("--env-lock", default=None)
    b.add_argument("--expected-n-genomes", type=int, default=EXPECTED_N_GENOMES)
    b.add_argument("--probe-len", type=int, default=PROBE_LEN)

    a = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.cmd == "build":
        report = build(
            genome_dir=a.genome_dir,
            baseline_report=a.baseline_report,
            out_dir=a.out_dir,
            report_path=a.report,
            provenance_path=a.provenance,
            env_lock=a.env_lock,
            expected_n_genomes=a.expected_n_genomes,
            probe_len=a.probe_len,
        )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "round_trip"}, indent=2, sort_keys=True
            )
        )
        print(
            f"round-trip: {report['round_trip']['all_recovered']} "
            f"({len(report['round_trip']['probes'])} probes)"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
