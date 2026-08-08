"""P2-10e-msa-producer — the per-candidate covariation PRODUCER (ADR-0005 **A10**).

This is the piece A9 under-priced. A9 Pin 2 costed the round's spare-rule leg at *"≈ 0 GPU-h
(torch-free numpy on the Leg-1 logits; ~2–3 h CPU wall)"* — a pricing that predates the covariation
producer. Populating each mined false positive's ``SpareRuleEvidence.any_helix_rscape`` is not free
numpy: it is the per-candidate 3-conda-env pipeline (:mod:`~tbox_finder.mining.homolog_msa`) —
nhmmer homolog-search vs the A1 target DB → mlocarna Sankoff align → R-scape covariation — at
~120–180 s/candidate. **A10** re-prices it as a NEW CPU cost centre and pins the orchestration as a
**decoupled SLURM array over false-positive shards** that runs *between* the round's scan leg and
its retrain leg, scores **every** false positive (no subsample), and hands a per-candidate
covariation status table (keyed by ``candidate_id``) to the retrain leg.

WHY A MODULE AND NOT ONE FUNCTION (the 3-env constraint)
--------------------------------------------------------
The certified chain spans three mutually-incompatible conda envs — ``tbox-homology``
(nhmmer/blastn), ``tbox-locarna`` (mlocarna), ``tbox-rscape`` (R-scape) — and a single Python
process cannot switch conda envs, so there is **no in-process ``candidate → status`` entry point
and cannot be one** (:mod:`homolog_msa` records the same fact). The producer is therefore a **bash
orchestrator** (``slurm/p2/mine_round.sbatch`` leg (c) / ``slurm/p2/mine_round_measure.sbatch``)
that activates each env once and calls one **shard-level** subcommand per stage —
:func:`search_shard` (``tbox-homology``), :func:`align_shard` (``tbox-locarna``),
:func:`score_shard` (``tbox-rscape``) —
each looping over the shard's candidates internally, so a shard costs **three** env activations, not
three per candidate. Each stage persists its per-candidate output under a stable per-candidate
work-directory, so the next stage (different env, possibly a different array task retry) reads what
the previous one wrote.

FAIL-CLOSED, EVERYWHERE (ADR-0006 A2 sparing semantics, untouched by A10)
------------------------------------------------------------------------
The producer never *mines*; it only *scores*. A candidate whose homolog set is below the
``min_sequences`` floor, whose mlocarna alignment carries no comparative consensus, or whose
R-scape run errors, resolves to :data:`~tbox_finder.mining.spare_rule.STATUS_UNAVAILABLE` — which
the spare rule reads as **spared, never silently mined**. A candidate that is *absent* from the
merged status table is likewise ``unavailable``
(:func:`tbox_finder.mining.mine_round.window_candidates_to_mining` resolves a missing id to
``unavailable``), so a dropped shard cannot fail *open*. Only a per-candidate **logic** error (a
coordinate out of its contig, a malformed span) is caught and recorded as not-producible; a
tool/subprocess/env failure (:class:`HomologDbError`) propagates and fails the whole shard loudly —
an env misconfiguration that silently marked every candidate ``unavailable`` would turn the round
into the exact no-op the readiness gate exists to prevent.

Transport is stdlib + the pandas/torch-free :mod:`homolog_msa` primitives (promote-don't-duplicate
— this module composes them, it never re-derives a search/align/score). :mod:`covariation` (and its
R-scape dependency) is reached only through :func:`homolog_msa.score_msa`, imported lazily inside
the score path, so :func:`search_shard` / :func:`align_shard` never drag R-scape into an env
lacking it.

PRD §9.1; ADR-0005 A9 → **A10**; ADR-0006 D7 / A1 / A2 / A3 / D17.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tbox_finder.mining.homolog_db import ToolTimeoutError, assert_usable_timeout
from tbox_finder.mining.homolog_msa import (
    BLAST_PREFIX,
    FM_INDEX,
    GENOME_DIR,
    HomologMsaError,
    align_candidate,
    resolve_candidate_sequence,
    score_msa,
    search_homologs,
    write_fasta_record,
)
from tbox_finder.power import MIN_REAL_HOMOLOG_N

SCHEMA_VERSION = "1.0"
STEP = "P2-10e-msa-producer"
ADR = "ADR-0005"  # A10 (this cost centre) + ADR-0006 D7/A1/A2/A3 (the route it scores)

#: The status-table row key. The wiring at ``mine_round.window_candidates_to_mining`` looks a
#: candidate's covariation status up by exactly this string, so producer and consumer share one key.
CANDIDATE_ID_KEY = "candidate_id"


class ProducerError(ValueError):
    """Raised on malformed producer input (bad shard row, duplicate candidate_id across shards)."""


# ═════════════════════════════════════════════════════════════════════════════
# The false-positive candidate spec + shard/manifest I/O (pure; stdlib JSON)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class CandidateSpec:
    """One mined false positive to score: coordinates only, exactly what a MiningCandidate carries.

    ``accession`` is the **contig-scoped** id ``<assembly>:c<contig_index>`` (the namespace
    ``mine_round.window_candidates_to_mining`` writes and the homolog DB / genome FASTAs share);
    ``locus_start``/``locus_end`` are 0-based half-open contig coordinates.
    """

    candidate_id: str
    accession: str
    locus_start: int
    locus_end: int

    def __post_init__(self) -> None:
        if not str(self.candidate_id):
            raise ProducerError("candidate_id must be non-empty")
        if not str(self.accession):
            raise ProducerError(f"candidate {self.candidate_id!r} has an empty accession")
        if not (0 <= int(self.locus_start) < int(self.locus_end)):
            raise ProducerError(
                f"candidate {self.candidate_id!r} span [{self.locus_start}, {self.locus_end}) "
                "must be 0-based half-open with start < end"
            )


def spec_to_dict(spec: CandidateSpec) -> dict[str, Any]:
    return {
        CANDIDATE_ID_KEY: spec.candidate_id,
        "accession": spec.accession,
        "locus_start": int(spec.locus_start),
        "locus_end": int(spec.locus_end),
    }


def spec_from_dict(row: dict[str, Any]) -> CandidateSpec:
    try:
        return CandidateSpec(
            candidate_id=str(row[CANDIDATE_ID_KEY]),
            accession=str(row["accession"]),
            locus_start=int(row["locus_start"]),
            locus_end=int(row["locus_end"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProducerError(f"malformed candidate row {row!r}: {exc}") from exc


def write_candidate_manifest(specs: Sequence[CandidateSpec], path: str | Path) -> Path:
    """Write the round's collected false-positive candidates (leg b → leg c handoff)."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "n_candidates": len(specs),
        "candidates": [spec_to_dict(s) for s in specs],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Path(path)


def read_candidate_manifest(path: str | Path) -> list[CandidateSpec]:
    """Read a candidate manifest or a single shard file (both carry a ``candidates`` list)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["candidates"] if isinstance(payload, dict) else payload
    return [spec_from_dict(r) for r in rows]


def select_sample(specs: Sequence[CandidateSpec], k: int) -> list[CandidateSpec]:
    """The first ``k`` candidates in a **deterministic** candidate-id order (A10 Phase-1 sample).

    Candidate ids begin with the accession, so a lexicographic sort is the *accession-sorted*
    order A10 Pin 1 names; ties break on the full id (contig + offsets), so the sample is fixed by
    the id set alone — no seed, reproducible across a re-run (§8.3).
    """
    if k < 1:
        raise ProducerError(f"sample size k must be >= 1, got {k}")
    return sorted(specs, key=lambda s: s.candidate_id)[:k]


def partition_shards(specs: Sequence[CandidateSpec], n_shards: int) -> list[list[CandidateSpec]]:
    """Split candidates into ``n_shards`` contiguous, deterministic shards (leg-c array tasks).

    Candidate-id-sorted then chunked by near-equal size, so shard *i* is a stable function of the id
    set and ``n_shards`` alone — an array task can pick its shard by ``SLURM_ARRAY_TASK_ID`` and a
    re-run reproduces the same partition.
    """
    if n_shards < 1:
        raise ProducerError(f"n_shards must be >= 1, got {n_shards}")
    ordered = sorted(specs, key=lambda s: s.candidate_id)
    n = len(ordered)
    base, extra = divmod(n, n_shards)
    shards: list[list[CandidateSpec]] = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < extra else 0)
        shards.append(ordered[start : start + size])
        start += size
    return shards


def write_shards(specs: Sequence[CandidateSpec], n_shards: int, out_dir: str | Path) -> list[Path]:
    """Write ``shard_<NNN>.json`` files (leg-c ``--array`` inputs); returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, shard in enumerate(partition_shards(specs, n_shards)):
        p = out / f"shard_{i:03d}.json"
        write_candidate_manifest(shard, p)
        paths.append(p)
    return paths


# ═════════════════════════════════════════════════════════════════════════════
# Per-candidate work-directory (stable across the three env stages)
# ═════════════════════════════════════════════════════════════════════════════
def candidate_slug(candidate_id: str) -> str:
    """A filesystem-safe, collision-free per-candidate directory name.

    Candidate ids carry ``:`` / ``-`` / ``.`` — legal on Linux but noisy — and two *distinct* ids
    can sanitise to the same string, so a 12-hex digest of the exact id is appended: the slug is a
    pure, deterministic function of the id and cannot collide two candidates onto one work-dir.
    """
    safe = "".join(c if c.isalnum() else "_" for c in candidate_id)[:64]
    digest = hashlib.sha1(candidate_id.encode("utf-8")).hexdigest()[
        :12
    ]  # noqa: S324 (non-crypto id)
    return f"{safe}_{digest}"


#: The per-candidate CM-free consensus this producer writes and BOTH the (a) scorer and the
#: criterion-(b) producer read (ADR-0006 A3/A4).  Defined HERE, beside the writer, and
#: imported by ``architecture_producer`` — a second literal would let a rename here make
#: every (b) candidate silently ``unavailable`` while the (a) leg kept working.
MSA_FILENAME = "msa.sto"


def candidate_workdir(workroot: str | Path, candidate_id: str) -> Path:
    return Path(workroot) / candidate_slug(candidate_id)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — SEARCH (runs in tbox-homology; never imports covariation/R-scape)
# ═════════════════════════════════════════════════════════════════════════════
def search_shard(
    specs: Sequence[CandidateSpec],
    *,
    workroot: str | Path,
    engine: str,
    evalue: float,
    max_target_seqs: int,
    min_pident: float,
    min_cov: float,
    min_sequences: int = MIN_REAL_HOMOLOG_N,
    db_prefix: str | Path = BLAST_PREFIX,
    fm_index: str | Path = FM_INDEX,
    genome_dir: str | Path = GENOME_DIR,
) -> list[dict[str, Any]]:
    """Resolve each candidate's nucleotides, search the A1 target DB, write ``candidate``+homologs.

    Per candidate, writes ``<workdir>/candidate.fa``, ``<workdir>/homologs.fa`` and a
    ``<workdir>/search.json`` report (the align/score stages read the latter — cross-env handoff).
    A per-candidate :class:`HomologMsaError` (out-of-contig span, unclean sequence, insufficient
    homologs signalled by ``sufficient=False``) is recorded as ``search_ok=False`` /
    ``sufficient=False`` (⇒ fail-closed ``unavailable`` downstream, ⇒ spared), never raised, so one
    malformed candidate cannot abort the shard. A :class:`HomologDbError` (tool absent, subprocess
    crash) is an env fault affecting *every* candidate and propagates — failing loud beats marking a
    whole shard ``unavailable`` and mining nothing.
    """
    rows: list[dict[str, Any]] = []
    for spec in specs:
        wd = candidate_workdir(workroot, spec.candidate_id)
        wd.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            cand_seq = resolve_candidate_sequence(
                genome_dir, spec.accession, spec.locus_start, spec.locus_end
            )
            write_fasta_record(spec.candidate_id, cand_seq, wd / "candidate.fa")
            report = search_homologs(
                candidate_fasta=wd / "candidate.fa",
                out_fasta=wd / "homologs.fa",
                engine=engine,
                evalue=evalue,
                max_target_seqs=max_target_seqs,
                min_pident=min_pident,
                min_cov=min_cov,
                prefix=db_prefix,
                fm_index=fm_index,
                genome_dir=genome_dir,
                min_sequences=min_sequences,
                tblout=(wd / "nhmmer.tblout") if engine == "nhmmer" else None,
            )
            report["search_ok"] = True
        except HomologMsaError as exc:
            report = {
                "search_ok": False,
                "sufficient": False,
                "n_homologs": 0,
                "n_records": 0,
                "error": str(exc),
            }
        report[CANDIDATE_ID_KEY] = spec.candidate_id
        report["search_wall_s"] = round(time.perf_counter() - t0, 4)
        (wd / "search.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(report)
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — ALIGN (runs in tbox-locarna; mlocarna comparative consensus)
# ═════════════════════════════════════════════════════════════════════════════
def _discard_msa(msa: Path) -> None:
    """Make "a failed align leaves no MSA" true **by construction** rather than by convention.

    Every non-success branch of :func:`align_shard` relies on the absence of ``msa.sto`` to reach
    ``unavailable`` ⇒ spared, but that absence was only ever an accident of the workroot being
    freshly created per array task. A stale MSA under a re-used workdir would be promoted by the
    sbatch (which tests ``[ -s .../msa.sto ]``, not this run's verdict) and scored as if this run
    had produced it — a *fabricated* consensus for a candidate whose alignment did not finish.
    """
    msa.unlink(missing_ok=True)


def align_shard(
    specs: Sequence[CandidateSpec],
    *,
    workroot: str | Path,
    cpu: int = 2,
    align_timeout_s: float,
) -> list[dict[str, Any]]:
    """mlocarna-align each candidate whose search stage found a **sufficient** homolog set.

    Reads ``<workdir>/search.json``; a candidate marked ``sufficient=False`` is skipped
    (``aligned=False``, ``reason=insufficient_homologs``) — no MSA is written, so the score stage
    reads ``unavailable`` for it (⇒ spared). An ``align_candidate`` :class:`HomologMsaError` (e.g.
    mlocarna emitted no ``#=GC SS_cons``) is recorded ``aligned=False``, not raised, for the same
    fail-closed reason. Writes ``<workdir>/msa.sto`` on success and ``<workdir>/align.json`` always.

    ``align_timeout_s`` is **keyword-required with no default** (the ADR-0006 A4 rule-parameter
    shape): it decides which candidates get a consensus at all, so every caller must name it and
    the round must record it. A candidate whose mlocarna exceeds it is recorded
    ``reason="align_timeout"`` and, like the other two spare branches, leaves no MSA behind ⇒
    ``unavailable`` ⇒ **spared**. This is the branch that exists because one candidate in job 1205
    ran 6h11m without finishing and took its whole 20-candidate shard down with the 12 h wall.
    """
    # Validate the bound ONCE, up front — not on the first sufficient candidate. A shard whose
    # bound is unusable must die before the run, not after however many alignments happened to
    # precede the first one that would have used it.
    assert_usable_timeout(align_timeout_s)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        wd = candidate_workdir(workroot, spec.candidate_id)
        search = _read_json(wd / "search.json")
        row: dict[str, Any] = {CANDIDATE_ID_KEY: spec.candidate_id}
        if not search.get("sufficient"):
            _discard_msa(wd / MSA_FILENAME)  # the THIRD spare branch — same absence, same reason
            row.update(aligned=False, reason="insufficient_homologs", align_wall_s=0.0)
            (wd / "align.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            rows.append(row)
            continue
        t0 = time.perf_counter()
        try:
            align_candidate(
                candidate_fasta=wd / "candidate.fa",
                homologs_fasta=wd / "homologs.fa",
                out_sto=wd / MSA_FILENAME,
                work_dir=wd / "align",
                cpu=cpu,
                timeout_s=align_timeout_s,
            )
            row["aligned"] = True
        except ToolTimeoutError as exc:
            row.update(aligned=False, reason="align_timeout", error=str(exc))
            _discard_msa(wd / MSA_FILENAME)
        except HomologMsaError as exc:
            row.update(aligned=False, reason="align_failed", error=str(exc))
            _discard_msa(wd / MSA_FILENAME)
        row["align_wall_s"] = round(time.perf_counter() - t0, 4)
        (wd / "align.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(row)
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — SCORE (runs in tbox-rscape; the pinned covariation spare-rule status)
# ═════════════════════════════════════════════════════════════════════════════
def _msa_depth(msa: Path) -> int:
    """Number of sequence rows in the candidate's MSA (0 if absent), for the depth measurement."""
    if not msa.is_file():
        return 0
    from tbox_finder.mining.msa_shuffle import read_pfam_alignment

    rows, _gc = read_pfam_alignment(msa)
    return len(rows)


def score_shard(
    specs: Sequence[CandidateSpec],
    *,
    workroot: str | Path,
    min_sequences: int = MIN_REAL_HOMOLOG_N,
    out_table: str | Path | None = None,
) -> dict[str, Any]:
    """Score each candidate's MSA → the pinned covariation status, and write the shard status table.

    Delegates to :func:`homolog_msa.score_msa` (== :func:`covariation.covariation_spare_status`:
    ``TOTAL_ACROSS_HELICES`` + ``min_sequences=20``, fail-closed to ``unavailable``). A candidate
    with no ``<workdir>/msa.sto`` (search insufficient or align failed) is scored ``None`` ⇒
    ``STATUS_UNAVAILABLE`` ⇒ spared. Rows carry the per-stage wall (search/align/score) and homolog
    depth A10 Pin 1 measures. The table's ``status`` map (``candidate_id → status``) is what the
    retrain leg's :func:`mine_round.window_candidates_to_mining` consumes.
    """
    rows: list[dict[str, Any]] = []
    for spec in specs:
        wd = candidate_workdir(workroot, spec.candidate_id)
        msa = wd / MSA_FILENAME
        search = _read_json(wd / "search.json")
        align = _read_json(wd / "align.json")
        t0 = time.perf_counter()
        status = score_msa(msa if msa.is_file() else None, min_sequences=min_sequences)
        score_wall_s = round(time.perf_counter() - t0, 4)
        rows.append(
            {
                CANDIDATE_ID_KEY: spec.candidate_id,
                "status": status,
                "n_homologs": int(search.get("n_homologs", 0)),
                "sufficient": bool(search.get("sufficient", False)),
                "aligned": bool(align.get("aligned", False)),
                "msa_depth": _msa_depth(msa),
                "search_wall_s": float(search.get("search_wall_s", 0.0)),
                "align_wall_s": float(align.get("align_wall_s", 0.0)),
                "score_wall_s": score_wall_s,
            }
        )
    table = _status_table(rows)
    if out_table is not None:
        Path(out_table).write_text(
            json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return table


def _status_table(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    status = {r[CANDIDATE_ID_KEY]: r["status"] for r in rows}
    if len(status) != len(rows):
        raise ProducerError("duplicate candidate_id within a single shard status table")
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "min_sequences": int(MIN_REAL_HOMOLOG_N),
        "n_candidates": len(rows),
        "status_counts": dict(sorted(Counter(status.values()).items())),
        "status": status,
        "rows": list(rows),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Merge — per-shard status tables → one round status table keyed by candidate_id
# ═════════════════════════════════════════════════════════════════════════════
def merge_status_tables(
    table_paths: Sequence[str | Path], *, out_path: str | Path | None = None
) -> dict[str, Any]:
    """Concatenate per-shard status tables into the round's status table (leg-c array → leg-d).

    A ``candidate_id`` appearing in two shards is a partition bug (shards must be disjoint) and
    raises — a silent last-writer-wins would let a mis-partition drop a candidate's real status back
    to whatever the other shard said. The merged ``status`` map is the sole object the retrain leg
    needs; ``rows`` are retained for the Phase-1 wall/depth measurement.
    """
    rows: list[dict[str, Any]] = []
    status: dict[str, str] = {}
    for path in table_paths:
        table = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in table["rows"]:
            cid = row[CANDIDATE_ID_KEY]
            if cid in status:
                raise ProducerError(
                    f"candidate_id {cid!r} appears in more than one shard status table — "
                    "shards must be disjoint (partition_shards bug or a re-used out path)"
                )
            status[cid] = row["status"]
            rows.append(row)
    merged = _status_table(rows)
    if out_path is not None:
        Path(out_path).write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return merged


def load_status_map(table_path: str | Path) -> dict[str, str]:
    """The ``candidate_id → status`` map from a merged (or single-shard) status table."""
    table = json.loads(Path(table_path).read_text(encoding="utf-8"))
    status = table.get("status")
    if not isinstance(status, dict):
        raise ProducerError(f"{table_path} has no 'status' map — not a producer status table")
    # Guard against a status table whose 'status' map silently disagrees with its own rows.
    row_status = {r[CANDIDATE_ID_KEY]: r["status"] for r in table.get("rows", [])}
    if row_status and row_status != {str(k): str(v) for k, v in status.items()}:
        raise ProducerError(f"{table_path}: 'status' map disagrees with 'rows' — corrupt table")
    return {str(k): str(v) for k, v in status.items()}


# ═════════════════════════════════════════════════════════════════════════════
# A10 Phase-1 measurement — N₀ + the per-candidate wall/depth distribution + envelope
# ═════════════════════════════════════════════════════════════════════════════
#: Candidate array widths the envelope is extrapolated over (Phase 2 pins the actual width).
DEFAULT_ARRAY_WIDTHS: tuple[int, ...] = (8, 16, 32, 48, 96)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    """min / median / mean / max / total of a sample (all zeros for an empty sample)."""
    xs = [float(v) for v in values]
    if not xs:
        return {"n": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0, "total": 0.0}
    return {
        "n": len(xs),
        "min": round(min(xs), 4),
        "median": round(statistics.median(xs), 4),
        "mean": round(statistics.fmean(xs), 4),
        "max": round(max(xs), 4),
        "total": round(sum(xs), 4),
    }


def measure_report(
    manifest_path: str | Path,
    status_table_path: str | Path,
    *,
    out_path: str | Path | None = None,
    array_widths: Sequence[int] = DEFAULT_ARRAY_WIDTHS,
) -> dict[str, Any]:
    """The A10 Pin 1 measurement report: **measured** N₀ + per-candidate wall/depth + envelope.

    ``N₀`` is the round's real false-positive count (the merged scan-shard manifest's
    ``n_candidates``); the per-stage wall (search / align / score) and homolog depth are read from
    the K-sample producer's status-table rows — recorded, not re-estimated. The extrapolated
    full-round envelope is ``N₀ × median-per-candidate-total`` single-thread core-hours, and, per
    candidate array width ``W``, ``(N₀ / W) × median-total`` wall-hours per task. **Phase 2 pins the
    actual array width** on these numbers (A10 Pin 2); this report is the input to that pin, not the
    pin itself.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    n0 = int(manifest.get("n_candidates", len(manifest.get("candidates", []))))
    table = json.loads(Path(status_table_path).read_text(encoding="utf-8"))
    rows = table.get("rows", [])

    search_wall = [float(r.get("search_wall_s", 0.0)) for r in rows]
    align_wall = [float(r.get("align_wall_s", 0.0)) for r in rows]
    score_wall = [float(r.get("score_wall_s", 0.0)) for r in rows]
    total_wall = [s + a + c for s, a, c in zip(search_wall, align_wall, score_wall, strict=True)]
    depth = [int(r.get("msa_depth", 0)) for r in rows]

    median_total_s = _distribution(total_wall)["median"]
    est_core_h = round(n0 * median_total_s / 3600.0, 2)
    envelope = {
        str(w): {
            "array_width": int(w),
            "est_core_h": est_core_h,
            "est_wall_h_per_task": round((n0 / int(w)) * median_total_s / 3600.0, 2),
        }
        for w in array_widths
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "phase": 1,
        "n0_false_positives": n0,
        "k_sample": len(rows),
        "status_counts": table.get("status_counts", {}),
        "wall_seconds": {
            "search": _distribution(search_wall),
            "align": _distribution(align_wall),
            "score": _distribution(score_wall),
            "total_per_candidate": _distribution(total_wall),
        },
        "homolog_depth": _distribution([float(d) for d in depth]),
        "extrapolated_envelope": envelope,
        "note": (
            "MEASURED (not estimated): N0 from the full-substrate scan, per-stage wall/depth from "
            "the K-sample producer run. Phase 2 pins the decoupled-array width on these numbers "
            "(A10 Pin 2) and flips MSA_SUPPLY_AVAILABLE."
        ),
    }
    if out_path is not None:
        Path(out_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


# ═════════════════════════════════════════════════════════════════════════════
# CLI — one subcommand per env stage (called by the sbatch after each conda activate)
# ═════════════════════════════════════════════════════════════════════════════
def _cmd_make_shards(args: argparse.Namespace) -> int:
    specs = read_candidate_manifest(args.manifest)
    paths = write_shards(specs, args.n_shards, args.out_dir)
    print(f"make-shards: {len(specs)} candidates → {len(paths)} shards under {args.out_dir}")
    return 0


def _cmd_select_sample(args: argparse.Namespace) -> int:
    specs = read_candidate_manifest(args.manifest)
    sample = select_sample(specs, args.k)
    write_candidate_manifest(sample, args.out)
    print(f"select-sample: {len(sample)}/{len(specs)} candidates → {args.out} (k={args.k})")
    return 0


def _cmd_search_shard(args: argparse.Namespace) -> int:
    specs = read_candidate_manifest(args.shard)
    rows = search_shard(
        specs,
        workroot=args.workroot,
        engine=args.engine,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
        min_pident=args.min_pident,
        min_cov=args.min_cov,
        min_sequences=args.min_sequences,
        db_prefix=args.db_prefix,
        fm_index=args.fm_index,
        genome_dir=args.genome_dir,
    )
    n_ok = sum(1 for r in rows if r.get("sufficient"))
    print(f"search-shard: {len(rows)} candidates, {n_ok} with a sufficient homolog set")
    return 0


def _usable_timeout_arg(text: str) -> float:
    """argparse type for ``--align-timeout-s``: a positive, FINITE number of seconds.

    ``type=float`` alone accepts ``nan`` and ``inf``. ``nan`` is the dangerous one — every
    comparison against it is False, so it slips past a ``<= 0`` check and
    ``communicate(timeout=nan)`` never fires: the bound would be declared, recorded in the round's
    provenance, and completely inert. Refusing it here means the shard dies at argparse, before the
    ~37 min search stage, rather than after it.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    try:
        assert_usable_timeout(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def _cmd_align_shard(args: argparse.Namespace) -> int:
    specs = read_candidate_manifest(args.shard)
    rows = align_shard(
        specs, workroot=args.workroot, cpu=args.cpu, align_timeout_s=args.align_timeout_s
    )
    n_aln = sum(1 for r in rows if r.get("aligned"))
    n_out = sum(1 for r in rows if r.get("reason") == "align_timeout")
    # Report the timed-out count on its own: it is the one non-success branch that is a *choice*
    # this run made (the bound), so a silent zero and a silent five must not read the same.
    print(f"align-shard: {n_aln}/{len(rows)} candidates aligned ({n_out} timed out ⇒ spared)")
    return 0


def _cmd_score_shard(args: argparse.Namespace) -> int:
    specs = read_candidate_manifest(args.shard)
    table = score_shard(
        specs, workroot=args.workroot, min_sequences=args.min_sequences, out_table=args.out_table
    )
    print(f"score-shard: {table['n_candidates']} candidates → {args.out_table}")
    print(f"  status_counts: {table['status_counts']}")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    merged = merge_status_tables(args.tables, out_path=args.out)
    print(f"merge: {merged['n_candidates']} candidates → {args.out}")
    print(f"  status_counts: {merged['status_counts']}")
    return 0


def _cmd_measure_report(args: argparse.Namespace) -> int:
    report = measure_report(args.manifest, args.status_table, out_path=args.out)
    print(
        f"measure-report: N0={report['n0_false_positives']} K={report['k_sample']} "
        f"median_total_s={report['wall_seconds']['total_per_candidate']['median']} → {args.out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbox_finder.mining.covariation_producer", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_mk = sub.add_parser("make-shards", help="partition a candidate manifest into array shards")
    p_mk.add_argument("--manifest", required=True)
    p_mk.add_argument("--n-shards", type=int, required=True)
    p_mk.add_argument("--out-dir", required=True)
    p_mk.set_defaults(func=_cmd_make_shards)

    p_ss = sub.add_parser("select-sample", help="deterministic first-K sample (A10 Phase-1)")
    p_ss.add_argument("--manifest", required=True)
    p_ss.add_argument("--k", type=int, required=True)
    p_ss.add_argument("--out", required=True)
    p_ss.set_defaults(func=_cmd_select_sample)

    p_search = sub.add_parser("search-shard", help="STAGE 1 (tbox-homology): resolve + search")
    p_search.add_argument("--shard", required=True)
    p_search.add_argument("--workroot", required=True)
    p_search.add_argument("--engine", choices=("blastn", "nhmmer"), required=True)
    # P6-frozen inclusion cutoffs (ADR-0006 D17): keyword-required, NO default (homolog_msa rule).
    p_search.add_argument("--evalue", type=float, required=True)
    p_search.add_argument("--max-target-seqs", type=int, required=True)
    p_search.add_argument("--min-pident", type=float, required=True)
    p_search.add_argument("--min-cov", type=float, required=True)
    p_search.add_argument("--min-sequences", type=int, default=MIN_REAL_HOMOLOG_N)
    p_search.add_argument("--db-prefix", default=BLAST_PREFIX)
    p_search.add_argument("--fm-index", default=FM_INDEX)
    p_search.add_argument("--genome-dir", default=GENOME_DIR)
    p_search.set_defaults(func=_cmd_search_shard)

    p_align = sub.add_parser("align-shard", help="STAGE 2 (tbox-locarna): mlocarna align")
    p_align.add_argument("--shard", required=True)
    p_align.add_argument("--workroot", required=True)
    p_align.add_argument("--cpu", type=int, default=2)
    # Keyword-required, NO default (the ADR-0006 A4 rule-parameter shape): the bound decides which
    # candidates produce a consensus, so a caller that forgets it must fail loudly at argparse
    # rather than silently inherit an unbounded align and re-run job 1205's 12 h shard death.
    p_align.add_argument(
        "--align-timeout-s",
        type=_usable_timeout_arg,
        required=True,
        help="per-candidate mlocarna wall-clock bound in seconds; exceeding it ⇒ unavailable "
        "⇒ spared (ADR-0005 D14). No default — the round must name it.",
    )
    p_align.set_defaults(func=_cmd_align_shard)

    p_score = sub.add_parser("score-shard", help="STAGE 3 (tbox-rscape): covariation status")
    p_score.add_argument("--shard", required=True)
    p_score.add_argument("--workroot", required=True)
    p_score.add_argument("--min-sequences", type=int, default=MIN_REAL_HOMOLOG_N)
    p_score.add_argument("--out-table", required=True)
    p_score.set_defaults(func=_cmd_score_shard)

    p_merge = sub.add_parser("merge", help="merge per-shard status tables → the round status table")
    p_merge.add_argument("--tables", nargs="+", required=True)
    p_merge.add_argument("--out", required=True)
    p_merge.set_defaults(func=_cmd_merge)

    p_meas = sub.add_parser("measure-report", help="A10 Phase-1: N0 + wall/depth + envelope report")
    p_meas.add_argument(
        "--manifest", required=True, help="the merged scan-shard FP manifest (→ N0)"
    )
    p_meas.add_argument("--status-table", required=True, help="the K-sample producer status table")
    p_meas.add_argument("--out", required=True)
    p_meas.set_defaults(func=_cmd_measure_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "ADR",
    "CANDIDATE_ID_KEY",
    "CandidateSpec",
    "ProducerError",
    "SCHEMA_VERSION",
    "STEP",
    "align_shard",
    "candidate_slug",
    "candidate_workdir",
    "load_status_map",
    "main",
    "measure_report",
    "merge_status_tables",
    "partition_shards",
    "read_candidate_manifest",
    "score_shard",
    "search_shard",
    "select_sample",
    "spec_from_dict",
    "spec_to_dict",
    "write_candidate_manifest",
    "write_shards",
]


if __name__ == "__main__":
    sys.exit(main())
