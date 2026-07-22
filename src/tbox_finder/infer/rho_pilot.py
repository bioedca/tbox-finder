"""P2-10c′-c-ii — the ρ-pilot SLURM scan driver: 100 pilot genomes + the Stage-1
checkpoint in, the ρ(τ, min_span, gap_merge) surface + throughput ``w`` out.

Where it sits
-------------
This is the last runnable sub-step of the "unblock mining properly" arc (ADR-0003 **D6**).
:mod:`tbox_finder.infer.scan` is the transport (tile → forward → reconcile → per-position
posteriors); :mod:`tbox_finder.infer.call` is the caller (posteriors → candidate loci, a
sweep over ``τ × min_span × gap_merge``). This module is the *orchestration* around them:
for each of the 100 divergent-clade pilot genomes (``data/interim/pilot_genomes/``,
287.76 Mbp fetched in P2-10c′-b) it reads the multi-contig FASTA, scans every contig with
the job-680 production checkpoint, sweeps candidate counts, and sums over contigs and
genomes to

    ρ(τ, min_span, gap_merge) = Σ candidates / T[Mbp]     (ADR-0003 D6)

reporting ρ as a **surface over the provisional grid**, not a single number (user decision
2026-07-22: "sweep ρ(τ), pin nothing"). It also measures ``w`` = Caduceus windows/sec/GPU
(ADR-0003 D6), the Stage-1 throughput that sizes the eventual genome scan.

The pin discipline (§10.3; ADR-0005 D3; ADR-0003 D6)
----------------------------------------------------
ρ is a **measured ops number**: ADR-0003 D6 says it "does not own any gate threshold that
touches the scientific claim", and ADR-0005 **D3 freezes the production Stage-1 threshold and
the locus geometry at the phase gate (§13.1), never at a sizing pilot** — and mining
(P2-10e) replaces this round-0 checkpoint, so any value frozen now is re-derived later. So
this driver **pins no ADR value and carries no scientific claim**: it reuses
:mod:`tbox_finder.infer.call`'s ``PROVISIONAL_*`` grids (labelled as binding nothing) and
records ρ as a function of them. The recall-favouring caller (global element score, no
required-element co-occurrence) makes ρ a **conservative upper bound** on the downstream
fetch — the safe direction to size a 0.7–68 GB download in.

Where the torch is, and where it is not
---------------------------------------
The GPU forward is behind :func:`tbox_finder.infer.scan.scan_sequence`, imported **lazily
inside the scan functions**; the *device is a plain string* (``"cuda:0"``) threaded through
to it. So the module's whole reporting/aggregation surface — FASTA parse, shard assignment,
the ρ-surface summation, ``build_report`` / ``validate_report`` — is ``numpy``/stdlib-only
and imports + unit-tests on the bare CI Tier-1 path, exactly like :mod:`.reconcile` and
:mod:`.call`. Only :func:`scan_genome` / :func:`scan_shard` touch a GPU.

The two-phase SLURM shape (``slurm/p2/scan_pilot.sbatch``)
---------------------------------------------------------
Genome scanning is embarrassingly parallel (no DDP, no gradient), so the sbatch splits the
allocated A4000s into one single-GPU ``scan-shard`` worker each (worker *i* → ``cuda:i``),
writes a per-shard partial to node-local ``/tmp``, then a single ``reduce`` globs the
partials and writes the one git-tracked report. Sharding is deterministic (LPT by genome
size) so every worker computes the same partition from the same genome directory.

PRD §5–§7 (window/throughput/scan), §9.1; ADR-0003 D6; ADR-0005 D3 + A3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder import ingest, provenance
from tbox_finder.infer.call import (
    PROVISIONAL_GAP_MERGE_GRID,
    PROVISIONAL_MIN_SPAN_GRID,
    PROVISIONAL_THRESHOLD_GRID,
    element_posterior,
    sweep_candidate_counts,
)

SCHEMA_VERSION = "1.0"
STEP = "P2-10c'-c-ii"

#: The ρ-pilot sweep grids (user decision 2026-07-22: "sweep ρ(τ), pin nothing"). Reused
#: verbatim from :mod:`tbox_finder.infer.call` — a forked copy would let the caller and the
#: reporter drift ([[promote-dont-duplicate-is-a-correctness-rule]]). They bind **no** ADR
#: value; ADR-0005 D3 freezes the production threshold + locus geometry at the phase gate.
PILOT_THRESHOLDS: tuple[float, ...] = PROVISIONAL_THRESHOLD_GRID
PILOT_MIN_SPANS: tuple[int, ...] = PROVISIONAL_MIN_SPAN_GRID
PILOT_GAP_MERGES: tuple[int, ...] = PROVISIONAL_GAP_MERGE_GRID

#: Windows per forward. Scan-time only; changes speed/peak-VRAM, never the reconciled result
#: (:mod:`.scan`). The measured throughput saturates by ~batch 16 on the A4000 (P2-04/P2-05
#: sizing), and inference ``no_grad`` VRAM is a fraction of training's, so 16 sits at the
#: throughput knee with ample headroom.
DEFAULT_SCAN_BATCH = 16

#: Frozen **operational** liveness floor — NOT an ADR pin, and loosening it is not a
#: threshold change. A working Stage-1 forward over 287.76 Mbp of real genomic DNA scores at
#: least one position ``>= 0.5`` element-like; a global max element posterior below it means
#: the forward is dead / the checkpoint mis-loaded — a broken run, reported as failing rather
#: than emitted as a (spuriously small) ρ (§10.3). It is a designed must-fire control whose
#: legitimate value on a live detector is emphatically not near zero
#: ([[namespace-mismatch-invisible-noop]], [[control-matchedness-must-be-asserted]]).
MIN_GLOBAL_MAX_P_ELEM = 0.5

#: The three per-grid-point count fields carried, aligned to the canonical grid order.
COUNT_FIELDS: tuple[str, ...] = (
    "n_candidates",
    "n_zero_flanked_candidates",
    "total_candidate_nt",
)

DEFAULT_GENOME_DIR = Path("data/interim/pilot_genomes")
DEFAULT_CHECKPOINT = Path("data/processed/checkpoints/stage1_production/stage1.pt")
DEFAULT_MANIFEST_REPORT = Path("data/processed/audits/pilot_fetch_report.json")
DEFAULT_REPORT = Path("data/processed/pilot/rho_pilot_report.json")
DEFAULT_PROVENANCE = Path("data/processed/pilot/rho_pilot_report.provenance.json")


class RhoPilotError(ValueError):
    """Raised on malformed scan input, a mis-shaped partial, or a report that fails to certify."""


# ═════════════════════════════════════════════════════════════════════════════
# FASTA — stdlib, uppercase-normalising (the ml-dna env has no biopython)
# ═════════════════════════════════════════════════════════════════════════════
def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse multi-contig FASTA text → ``[(contig_id, seq_upper), ...]``.

    ``contig_id`` is the first whitespace-delimited token after ``>``. The sequence is
    concatenated across wrapped lines and **upper-cased** — this is load-bearing:
    :func:`tbox_finder.data.window_dataset.encode_bases` keys an *uppercase-only* dict and
    maps every miss (lowercase ``a/c/g/t``, IUPAC ambiguity, ``N``) to the ``N`` token, so a
    soft-masked (lowercase) genome would be silently destroyed to all-``N`` without this
    ``.upper()``. NCBI ``_genomic.fna`` are conventionally uppercase, but normalising here
    makes the driver correct regardless of the source's masking convention.

    Empty records (a header with no sequence) are dropped — a zero-length contig has no
    window and :func:`tbox_finder.infer.scan.scan_sequence` rejects an empty sequence anyway.
    """
    contigs: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []

    def _flush() -> None:
        if header is not None:
            seq = "".join(chunks)
            if seq:
                contigs.append((header, seq))

    for line in text.splitlines():
        if line.startswith(">"):
            _flush()
            header = line[1:].split(None, 1)[0] if line[1:].strip() else ""
            chunks = []
        else:
            chunks.append(line.strip().upper())
    _flush()
    return contigs


def read_fasta_file(path: str | Path) -> list[tuple[str, str]]:
    """Read a ``.fna`` file → parsed, uppercase-normalised contigs (see :func:`parse_fasta`)."""
    return parse_fasta(Path(path).read_text(encoding="utf-8"))


def genome_accession(path: str | Path) -> str:
    """The assembly accession of a pilot ``.fna`` — its filename stem (``GCA_002683655.1``)."""
    return Path(path).stem


def discover_genomes(genome_dir: str | Path) -> list[Path]:
    """Every ``*.fna`` under ``genome_dir``, sorted by accession — the full pilot set."""
    return sorted(Path(genome_dir).glob("*.fna"))


# ═════════════════════════════════════════════════════════════════════════════
# Deterministic shard assignment — LPT by genome size (balances GPU wall)
# ═════════════════════════════════════════════════════════════════════════════
def assign_shards(sized: Sequence[tuple[str, int]], n_shards: int) -> list[list[str]]:
    """Partition ``(accession, size_bytes)`` items into ``n_shards`` balanced lists.

    Longest-processing-time greedy: sort by size descending (ties broken by accession for
    determinism), assign each genome to the currently-lightest shard. This balances GPU wall
    across workers far better than round-robin when genome sizes span 0.52–7.07 Mbp. It is a
    **pure partition** — every accession lands in exactly one shard and none is dropped or
    duplicated (asserted in tests) — and it is a deterministic function of the input set and
    ``n_shards`` alone, so every worker independently computes the same assignment from the
    same genome directory without any coordination.
    """
    if n_shards < 1:
        raise RhoPilotError(f"n_shards must be >= 1, got {n_shards}")
    if not sized:
        raise RhoPilotError("no genomes to shard")
    ordered = sorted(sized, key=lambda it: (-int(it[1]), str(it[0])))
    loads = [0] * n_shards
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    for acc, size in ordered:
        lightest = min(range(n_shards), key=lambda s: (loads[s], s))
        shards[lightest].append(acc)
        loads[lightest] += int(size)
    return shards


# ═════════════════════════════════════════════════════════════════════════════
# The canonical grid + the ρ-surface reduction (pure — the unit-tested core)
# ═════════════════════════════════════════════════════════════════════════════
def grid_order(
    thresholds: Sequence[float],
    min_spans: Sequence[int],
    gap_merges: Sequence[int],
) -> list[tuple[float, int, int]]:
    """The canonical ``(threshold, min_span, gap_merge)`` order — the sweep's own emission order.

    :func:`tbox_finder.infer.call.sweep_candidate_counts` emits rows in the loop order
    ``threshold`` (outer) → ``gap_merge`` (middle) → ``min_span`` (inner). This function
    reproduces exactly that order so per-contig sweep rows map positionally onto a fixed
    count vector; :func:`rows_to_count_arrays` asserts the alignment, so a future re-ordering
    of the sweep loop fails loud rather than silently transposing the surface.
    """
    return [
        (float(t), int(ms), int(gm)) for t in thresholds for gm in gap_merges for ms in min_spans
    ]


def rows_to_count_arrays(
    rows: Sequence[Mapping[str, Any]], grid: Sequence[tuple[float, int, int]]
) -> dict[str, list[int]]:
    """Extract a sweep's rows into count vectors aligned to ``grid`` (fail-closed on drift).

    Each of ``rows`` (a :func:`sweep_candidate_counts` output) must match ``grid`` positionally
    on ``(threshold, min_span, gap_merge)``; a mismatch raises rather than mis-aligning the
    surface. Returns one ``list[int]`` per field of :data:`COUNT_FIELDS`.
    """
    if len(rows) != len(grid):
        raise RhoPilotError(f"sweep emitted {len(rows)} rows but the grid has {len(grid)} points")
    out: dict[str, list[int]] = {f: [] for f in COUNT_FIELDS}
    for k, (row, point) in enumerate(zip(rows, grid, strict=True)):
        got = (float(row["threshold"]), int(row["min_span"]), int(row["gap_merge"]))
        if got != point:
            raise RhoPilotError(
                f"sweep row {k} is {got} but grid point {k} is {point} — sweep order drifted"
            )
        for f in COUNT_FIELDS:
            out[f].append(int(row[f]))
    return out


def axes_from_grid(
    grid: Sequence[Sequence[float]],
) -> tuple[list[float], list[int], list[int]]:
    """Recover ``(thresholds, min_spans, gap_merges)`` from a recorded canonical grid.

    The inverse of :func:`grid_order`: the axes are the first-appearance-ordered unique values
    of each column. The caller re-runs :func:`grid_order` on the result and checks it round-trips
    to ``grid``, so a grid that was **not** produced by :func:`grid_order` (a hand-edited or
    corrupt partial) is rejected rather than silently mis-reduced.
    """
    thresholds = list(dict.fromkeys(float(p[0]) for p in grid))
    min_spans = list(dict.fromkeys(int(p[1]) for p in grid))
    gap_merges = list(dict.fromkeys(int(p[2]) for p in grid))
    if grid_order(thresholds, min_spans, gap_merges) != [
        (float(p[0]), int(p[1]), int(p[2])) for p in grid
    ]:
        raise RhoPilotError("recorded grid is not a canonical threshold×gap_merge×min_span grid")
    return thresholds, min_spans, gap_merges


def sum_arrays(arrays: Sequence[Sequence[int]]) -> list[int]:
    """Element-wise sum of equal-length integer vectors — the ρ-surface accumulator.

    This is the whole reduction: candidate counts are additive over contigs (each contig is
    scanned independently, so a candidate never spans a contig boundary) and over genomes, so
    the global count at each grid point is the plain element-wise sum of the per-genome
    vectors. Kept a tiny pure function precisely so a unit test can pin it against a
    hand-summed oracle and a sabotage partner (sum-vs-max, transposed index) must diverge.
    """
    if not arrays:
        raise RhoPilotError("nothing to sum")
    width = len(arrays[0])
    total = [0] * width
    for a in arrays:
        if len(a) != width:
            raise RhoPilotError(f"ragged vectors: {len(a)} != {width}")
        for i, v in enumerate(a):
            total[i] += int(v)
    return total


def rho_surface(
    global_counts: Mapping[str, Sequence[int]],
    grid: Sequence[tuple[float, int, int]],
    total_mbp: float,
) -> list[dict[str, Any]]:
    """Build the ρ surface: one row per grid point with ρ = candidates / T[Mbp].

    ``rho_excl_zero_flanked_per_mbp`` divides the count with contig-end (zero-flanked)
    candidates removed, so the report shows ρ's sensitivity to excluding pad-touched calls
    without ever silently dropping them (§10.3).
    """
    if not (float(total_mbp) > 0.0):
        raise RhoPilotError(f"total_mbp must be > 0, got {total_mbp}")
    tm = float(total_mbp)
    surface: list[dict[str, Any]] = []
    for k, (t, ms, gm) in enumerate(grid):
        n = int(global_counts["n_candidates"][k])
        n_zf = int(global_counts["n_zero_flanked_candidates"][k])
        surface.append(
            {
                "threshold": float(t),
                "min_span": int(ms),
                "gap_merge": int(gm),
                "n_candidates": n,
                "rho_per_mbp": n / tm,
                "n_zero_flanked_candidates": n_zf,
                "rho_excl_zero_flanked_per_mbp": (n - n_zf) / tm,
                "total_candidate_nt": int(global_counts["total_candidate_nt"][k]),
            }
        )
    return surface


def permissive_corner_index(grid: Sequence[tuple[float, int, int]]) -> int:
    """Grid index of the most-permissive corner: min τ, min span, max gap-merge.

    This corner yields the most candidates, so its count is the designed must-fire liveness
    control — a working detector over 287 Mbp cannot legitimately return zero there.
    """
    thresholds = [p[0] for p in grid]
    min_spans = [p[1] for p in grid]
    gap_merges = [p[2] for p in grid]
    corner = (min(thresholds), min(min_spans), max(gap_merges))
    return list(grid).index(corner)


def per_genome_digest(per_genome: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over the per-genome scan evidence.

    Each genome hashes to its accession + total_bp + the three count vectors; the per-genome
    hashes are sorted before folding, so the digest depends on the evidence set, not on shard
    order or genome order (the :func:`tbox_finder.mining.pilot_fetch.genome_digest` precedent).
    """
    hashes = []
    for g in per_genome:
        payload = json.dumps(
            {
                "accession": str(g["assembly_accession"]),
                "total_bp": int(g["total_bp"]),
                **{f: [int(x) for x in g[f]] for f in COUNT_FIELDS},
            },
            sort_keys=True,
        )
        hashes.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return ingest.records_digest(sorted(hashes))


# ═════════════════════════════════════════════════════════════════════════════
# The GPU legs — scan one genome, scan a shard (torch lazily imported here)
# ═════════════════════════════════════════════════════════════════════════════
def scan_genome(
    model: Any,
    fasta_path: str | Path,
    *,
    device: Any,
    grid: Sequence[tuple[float, int, int]],
    thresholds: Sequence[float] = PILOT_THRESHOLDS,
    min_spans: Sequence[int] = PILOT_MIN_SPANS,
    gap_merges: Sequence[int] = PILOT_GAP_MERGES,
    batch_size: int = DEFAULT_SCAN_BATCH,
) -> dict[str, Any]:
    """Scan every contig of one genome → per-genome candidate-count vectors + evidence.

    For each contig: :func:`tbox_finder.infer.scan.scan_sequence` (tile → forward →
    reconcile) then :func:`sweep_candidate_counts` over the grid, accumulated element-wise.
    ``scan_seconds`` times **only** the GPU forward (not FASTA read or the CPU sweep), so
    ``n_windows / scan_seconds`` is an honest windows/sec/GPU throughput ``w``. ``max_p_elem``
    (global over the genome) is the liveness witness — a dead forward leaves it near zero.
    """
    from tbox_finder.data.window_dataset import tile_windows
    from tbox_finder.infer.scan import scan_sequence

    contigs = read_fasta_file(fasta_path)
    if not contigs:
        raise RhoPilotError(f"{fasta_path} has no non-empty contigs")

    acc = genome_accession(fasta_path)
    counts: dict[str, list[int]] = {f: [0] * len(grid) for f in COUNT_FIELDS}
    total_bp = 0
    n_windows = 0
    max_p_elem = 0.0
    scan_seconds = 0.0

    for _cid, seq in contigs:
        total_bp += len(seq)
        n_windows += len(tile_windows(len(seq)))
        t0 = time.perf_counter()
        recon = scan_sequence(model, seq, device=device, batch_size=batch_size)
        scan_seconds += time.perf_counter() - t0
        rows = sweep_candidate_counts(
            recon.log_probs,
            recon.zero_flanked,
            thresholds=thresholds,
            min_spans=min_spans,
            gap_merges=gap_merges,
        )
        arrays = rows_to_count_arrays(rows, grid)
        for f in COUNT_FIELDS:
            counts[f] = [a + b for a, b in zip(counts[f], arrays[f], strict=True)]
        p_elem = element_posterior(recon.log_probs)
        if p_elem.size:
            max_p_elem = max(max_p_elem, float(p_elem.max()))

    return {
        "assembly_accession": acc,
        "total_bp": int(total_bp),
        "n_contigs": len(contigs),
        "n_windows": int(n_windows),
        "scan_seconds": float(scan_seconds),
        "max_p_elem": float(max_p_elem),
        **{f: counts[f] for f in COUNT_FIELDS},
    }


def scan_shard(
    *,
    genome_dir: str | Path,
    shard: int,
    n_shards: int,
    gpu_index: int,
    checkpoint: str | Path,
    out_path: str | Path,
    thresholds: Sequence[float] = PILOT_THRESHOLDS,
    min_spans: Sequence[int] = PILOT_MIN_SPANS,
    gap_merges: Sequence[int] = PILOT_GAP_MERGES,
    batch_size: int = DEFAULT_SCAN_BATCH,
) -> dict[str, Any]:
    """Scan this shard's genomes on ``cuda:<gpu_index>`` and write a node-local partial JSON.

    Loads the checkpoint once (:func:`tbox_finder.infer.scan.load_stage1_checkpoint`, which
    fails closed on any key/shape mismatch — the invisible-mis-load guard is upstream), then
    scans its LPT slice of the pilot set. The partial carries the per-genome evidence and the
    shard's throughput; ``reduce`` sums the partials. No validation here — a partial is
    intermediate — but the sbatch writes a ``SHARD_OK`` marker only on the success path.
    """
    from tbox_finder.infer.scan import load_stage1_checkpoint

    if not (0 <= int(shard) < int(n_shards)):
        raise RhoPilotError(f"shard {shard} out of range [0, {n_shards})")
    genomes = discover_genomes(genome_dir)
    if not genomes:
        raise RhoPilotError(f"no *.fna genomes under {genome_dir}")
    sized = [(genome_accession(p), p.stat().st_size) for p in genomes]
    my_accs = set(assign_shards(sized, int(n_shards))[int(shard)])
    by_acc = {genome_accession(p): p for p in genomes}

    grid = grid_order(thresholds, min_spans, gap_merges)
    device = f"cuda:{int(gpu_index)}"
    model = load_stage1_checkpoint(checkpoint, device=device)

    wall0 = time.perf_counter()
    per_genome: list[dict[str, Any]] = []
    for acc in sorted(my_accs):
        per_genome.append(
            scan_genome(
                model,
                by_acc[acc],
                device=device,
                grid=grid,
                thresholds=thresholds,
                min_spans=min_spans,
                gap_merges=gap_merges,
                batch_size=batch_size,
            )
        )
    shard_wall = time.perf_counter() - wall0

    partial = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": "shard_partial",
        "shard": int(shard),
        "n_shards": int(n_shards),
        "device": device,
        "n_genomes": len(per_genome),
        "n_windows": sum(int(g["n_windows"]) for g in per_genome),
        "scan_seconds": sum(float(g["scan_seconds"]) for g in per_genome),
        "shard_wall_seconds": float(shard_wall),
        "grid": [list(p) for p in grid],
        "per_genome": per_genome,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return partial


# ═════════════════════════════════════════════════════════════════════════════
# Reduce — glob the partials, sum → the ρ surface, certify, write the report
# ═════════════════════════════════════════════════════════════════════════════
def _load_manifest(manifest_report: str | Path) -> tuple[set[str], int, float]:
    """The certified pilot set from the fetch report → (ok accessions, total_bp, total_mbp)."""
    rep = json.loads(Path(manifest_report).read_text(encoding="utf-8"))
    per = rep.get("per_genome")
    if not isinstance(per, list):
        raise RhoPilotError(f"{manifest_report} has no per_genome list")
    ok = {str(r["assembly_accession"]) for r in per if str(r.get("status")) == "ok"}
    return ok, int(rep["total_bp"]), float(rep["total_mbp"])


def build_report(
    per_genome: Sequence[Mapping[str, Any]],
    shard_meta: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float],
    min_spans: Sequence[int],
    gap_merges: Sequence[int],
    manifest_total_bp: int,
    manifest_total_mbp: float,
    manifest_ok: Sequence[str],
    checkpoint_sha256: str,
    accessed: str,
) -> dict[str, Any]:
    """Assemble the ρ-pilot report — every headline re-derived from the per-genome evidence."""
    grid = grid_order(thresholds, min_spans, gap_merges)
    per_sorted = sorted(per_genome, key=lambda g: str(g["assembly_accession"]))

    global_counts = {
        f: sum_arrays([[int(x) for x in g[f]] for g in per_sorted]) for f in COUNT_FIELDS
    }
    scanned_bp = sum(int(g["total_bp"]) for g in per_sorted)
    scanned_mbp = scanned_bp / 1e6
    surface = rho_surface(global_counts, grid, scanned_mbp)

    total_windows = sum(int(m["n_windows"]) for m in shard_meta)
    total_scan_seconds = sum(float(m["scan_seconds"]) for m in shard_meta)
    per_shard_w = [
        {
            "shard": int(m["shard"]),
            "device": str(m["device"]),
            "n_genomes": int(m["n_genomes"]),
            "n_windows": int(m["n_windows"]),
            "scan_seconds": float(m["scan_seconds"]),
            "shard_wall_seconds": float(m["shard_wall_seconds"]),
            "w_windows_per_sec_per_gpu": (
                int(m["n_windows"]) / float(m["scan_seconds"])
                if float(m["scan_seconds"]) > 0
                else 0.0
            ),
        }
        for m in sorted(shard_meta, key=lambda m: int(m["shard"]))
    ]
    global_max_p_elem = max((float(g["max_p_elem"]) for g in per_sorted), default=0.0)
    corner = permissive_corner_index(grid)

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "n_genomes": len(per_sorted),
        "n_shards": len(shard_meta),
        "scanned_bp": scanned_bp,
        "scanned_mbp": scanned_mbp,
        "manifest_total_bp": int(manifest_total_bp),
        "manifest_total_mbp": float(manifest_total_mbp),
        "grid_order": [list(p) for p in grid],
        "grids": {
            "thresholds": [float(t) for t in thresholds],
            "min_spans": [int(m) for m in min_spans],
            "gap_merges": [int(g) for g in gap_merges],
        },
        "rho_surface": surface,
        "throughput": {
            "total_windows": total_windows,
            "total_scan_seconds": total_scan_seconds,
            "w_windows_per_sec_per_gpu": (
                total_windows / total_scan_seconds if total_scan_seconds > 0 else 0.0
            ),
            "per_shard": per_shard_w,
        },
        "power": {
            "global_max_p_elem": global_max_p_elem,
            "min_global_max_p_elem_floor": MIN_GLOBAL_MAX_P_ELEM,
            "permissive_corner": {
                "grid_index": corner,
                "threshold": grid[corner][0],
                "min_span": grid[corner][1],
                "gap_merge": grid[corner][2],
                "n_candidates": int(global_counts["n_candidates"][corner]),
            },
        },
        "phi": {
            "measured": False,
            "reason": (
                "φ (ADR-0003 D6) is the both-strand carry-through fraction of the §6 "
                "strand-resolver — a Stage-2 quantity. Caduceus-PS is RC-equivariant, so one "
                "Stage-1 pass already covers both strands; φ is not derivable from a Stage-1 "
                "candidate scan and is measured at the pre-P5 sizing gate, not here (§10.3)."
            ),
        },
        "checkpoint": {
            "path": str(DEFAULT_CHECKPOINT),
            "sha256": str(checkpoint_sha256),
        },
        "geometry": {
            "window_nt": 1024,
            "stride_nt": 512,
            "scan_batch": DEFAULT_SCAN_BATCH,
        },
        "digest": per_genome_digest(per_sorted),
        # Honesty flags (§10.3), validator-enforced. This step DOES measure ρ and count
        # candidates, but pins no ADR value and asserts no scientific claim: ρ is an ops
        # number that sizes a fetch (ADR-0003 D6), and D3 freezes production values at the
        # phase gate. The provisional grids bind nothing.
        "rho_measured": True,
        "candidates_counted": True,
        "pins_no_adr_value": True,
        "is_science": False,
        "sequences_synthetic": False,
        "provisional_grids_bind_nothing": True,
        "manifest_ok_accessions": sorted(str(a) for a in manifest_ok),
        "source": {
            "genome_dir": str(DEFAULT_GENOME_DIR),
            "manifest_report": str(DEFAULT_MANIFEST_REPORT),
            "accessed": accessed,
        },
        "per_genome": [
            {
                "assembly_accession": str(g["assembly_accession"]),
                "total_bp": int(g["total_bp"]),
                "n_contigs": int(g["n_contigs"]),
                "n_windows": int(g["n_windows"]),
                "scan_seconds": float(g["scan_seconds"]),
                "max_p_elem": float(g["max_p_elem"]),
                **{f: [int(x) for x in g[f]] for f in COUNT_FIELDS},
            }
            for g in per_sorted
        ],
    }


def validate_report(report: Mapping[str, Any]) -> list[str]:  # noqa: C901 - one flat clause list
    """Re-derive every headline from the evidence; list failures — never raises; ``[]`` == valid.

    The clauses that matter: coverage (the scanned accession set and bp equal the certified
    manifest — no genome silently dropped, the masked-nothing guard); ρ-surface re-derivation
    (every ``n_candidates`` re-summed from ``per_genome``, every ρ re-divided by T); grid
    integrity; and the two designed must-fire liveness controls (a live global max element
    posterior and a non-empty permissive corner — whose legitimate values on a working
    detector over 287 Mbp are not zero, [[namespace-mismatch-invisible-noop]]).
    """
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version != {SCHEMA_VERSION!r} (got {report.get('schema_version')!r})"
        )
    if report.get("step") != STEP:
        problems.append(f"step != {STEP!r} (got {report.get('step')!r})")

    grids = report.get("grids")
    if not isinstance(grids, Mapping) or not all(
        isinstance(grids.get(k), Sequence) and grids.get(k)
        for k in ("thresholds", "min_spans", "gap_merges")
    ):
        return problems + ["grids block missing or incomplete"]
    grid = grid_order(grids["thresholds"], grids["min_spans"], grids["gap_merges"])
    if [list(p) for p in grid] != report.get("grid_order"):
        problems.append("grid_order does not match grids (thresholds×gap_merges×min_spans)")

    per = report.get("per_genome")
    if not isinstance(per, Sequence) or isinstance(per, (str, bytes)) or not per:
        return problems + ["per_genome missing or empty"]

    # ── per-genome shape + fabrication guard ────────────────────────────────
    for idx, g in enumerate(per):
        acc = str(g.get("assembly_accession", ""))
        if not (acc[:3] in ("GCA", "GCF") and "_" in acc):
            problems.append(f"per_genome[{idx}] accession {acc!r} is not a GC[AF]_ accession")
        if int(g.get("total_bp", 0)) <= 0:
            problems.append(f"per_genome[{idx}] {acc} total_bp <= 0")
        if int(g.get("n_contigs", 0)) <= 0 or int(g.get("n_windows", 0)) <= 0:
            problems.append(f"per_genome[{idx}] {acc} has no contigs/windows")
        if not (0.0 <= float(g.get("max_p_elem", -1.0)) <= 1.0):
            problems.append(f"per_genome[{idx}] {acc} max_p_elem out of [0, 1]")
        for f in COUNT_FIELDS:
            vec = g.get(f)
            if not isinstance(vec, Sequence) or len(vec) != len(grid):
                problems.append(f"per_genome[{idx}] {acc} {f} is not a length-{len(grid)} vector")
            elif any(int(v) < 0 for v in vec):
                problems.append(f"per_genome[{idx}] {acc} {f} has a negative count")
        nz = g.get("n_zero_flanked_candidates")
        nc = g.get("n_candidates")
        if (
            isinstance(nz, Sequence)
            and isinstance(nc, Sequence)
            and len(nz) == len(nc)
            and any(int(z) > int(c) for z, c in zip(nz, nc, strict=True))
        ):
            problems.append(f"per_genome[{idx}] {acc} zero-flanked count exceeds total")
    if problems:
        return problems

    # ── coverage: the scanned set == the certified manifest set (no silent drop) ──
    scanned = {str(g["assembly_accession"]) for g in per}
    manifest_ok = {str(a) for a in report.get("manifest_ok_accessions", [])}
    if not manifest_ok:
        problems.append("manifest_ok_accessions missing — cannot certify coverage")
    else:
        if scanned != manifest_ok:
            missing = sorted(manifest_ok - scanned)[:5]
            extra = sorted(scanned - manifest_ok)[:5]
            problems.append(f"scanned set != manifest ok set (missing {missing}, extra {extra})")
        if len(per) != len(manifest_ok) or int(report.get("n_genomes", -1)) != len(per):
            problems.append(
                f"n_genomes {report.get('n_genomes')} / |per_genome| {len(per)} "
                f"!= |manifest ok| {len(manifest_ok)}"
            )
    scanned_bp = sum(int(g["total_bp"]) for g in per)
    if scanned_bp != int(report.get("scanned_bp", -1)):
        problems.append(f"scanned_bp {report.get('scanned_bp')} != re-summed {scanned_bp}")
    if scanned_bp != int(report.get("manifest_total_bp", -1)):
        problems.append(
            f"scanned_bp {scanned_bp} != manifest_total_bp {report.get('manifest_total_bp')} "
            "— a contig or genome was dropped or double-counted (§10.3)"
        )

    # ── ρ-surface re-derivation from the evidence ───────────────────────────
    surface = report.get("rho_surface")
    if not isinstance(surface, Sequence) or len(surface) != len(grid):
        return problems + [f"rho_surface is not a length-{len(grid)} list"]
    global_counts = {f: sum_arrays([[int(x) for x in g[f]] for g in per]) for f in COUNT_FIELDS}
    scanned_mbp = scanned_bp / 1e6
    for k, ((t, ms, gm), row) in enumerate(zip(grid, surface, strict=True)):
        if (float(row["threshold"]), int(row["min_span"]), int(row["gap_merge"])) != (t, ms, gm):
            problems.append(f"rho_surface[{k}] grid point mismatch")
            continue
        n = int(global_counts["n_candidates"][k])
        if int(row["n_candidates"]) != n:
            problems.append(f"rho_surface[{k}] n_candidates {row['n_candidates']} != re-summed {n}")
        if abs(float(row["rho_per_mbp"]) - n / scanned_mbp) > 1e-9:
            problems.append(f"rho_surface[{k}] rho_per_mbp != n_candidates / T[Mbp]")
        n_zf = int(global_counts["n_zero_flanked_candidates"][k])
        if abs(float(row["rho_excl_zero_flanked_per_mbp"]) - (n - n_zf) / scanned_mbp) > 1e-9:
            problems.append(f"rho_surface[{k}] rho_excl_zero_flanked_per_mbp mis-derived")

    # ── the designed must-fire liveness controls ────────────────────────────
    power = report.get("power", {})
    gmax = float(power.get("global_max_p_elem", -1.0)) if isinstance(power, Mapping) else -1.0
    re_gmax = max((float(g["max_p_elem"]) for g in per), default=0.0)
    if abs(gmax - re_gmax) > 1e-9:
        problems.append(f"power.global_max_p_elem {gmax} != re-derived {re_gmax}")
    if gmax < MIN_GLOBAL_MAX_P_ELEM:
        problems.append(
            f"global_max_p_elem {gmax:.4f} < floor {MIN_GLOBAL_MAX_P_ELEM} — the Stage-1 forward "
            "produced no element-like signal anywhere in 287 Mbp; the checkpoint is dead or "
            "mis-loaded, not a small ρ (§10.3)"
        )
    corner = permissive_corner_index(grid)
    corner_n = int(global_counts["n_candidates"][corner])
    if corner_n <= 0:
        problems.append(
            f"permissive corner (τ={grid[corner][0]}, min_span={grid[corner][1]}, "
            f"gap_merge={grid[corner][2]}) has 0 candidates over 287 Mbp — a live detector "
            "cannot; this is a broken run, not a result (§9.3 stop, §10.3)"
        )

    # ── honesty flags (§10.3) ───────────────────────────────────────────────
    for flag, want in (
        ("rho_measured", True),
        ("candidates_counted", True),
        ("pins_no_adr_value", True),
        ("is_science", False),
        ("sequences_synthetic", False),
        ("provisional_grids_bind_nothing", True),
    ):
        if bool(report.get(flag)) is not want:
            problems.append(f"honesty flag {flag} must be {want}")

    if not (isinstance(report.get("digest"), str) and report["digest"].strip()):
        problems.append("digest missing or empty")
    elif report["digest"] != per_genome_digest(per):
        problems.append("digest != re-derived per_genome_digest")

    return problems


def load_partials(partial_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read + shape-check the shard partials; raise on a missing/duplicate/mis-typed shard."""
    if not partial_paths:
        raise RhoPilotError("no shard partials given to reduce")
    partials: list[dict[str, Any]] = []
    seen: set[int] = set()
    for p in partial_paths:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        if obj.get("kind") != "shard_partial" or obj.get("step") != STEP:
            raise RhoPilotError(f"{p} is not a {STEP} shard partial")
        sh = int(obj["shard"])
        if sh in seen:
            raise RhoPilotError(f"duplicate shard {sh} among partials")
        seen.add(sh)
        partials.append(obj)
    n_shards = {int(o["n_shards"]) for o in partials}
    if len(n_shards) != 1:
        raise RhoPilotError(f"partials disagree on n_shards: {sorted(n_shards)}")
    (expected,) = n_shards
    if seen != set(range(expected)):
        raise RhoPilotError(f"expected shards {sorted(range(expected))}, got {sorted(seen)}")
    grids = {json.dumps(o.get("grid"), sort_keys=True) for o in partials}
    if len(grids) != 1:
        raise RhoPilotError("shard partials disagree on the sweep grid — cannot reduce")
    return partials


def reduce_partials(
    partial_paths: Sequence[str | Path],
    *,
    manifest_report: str | Path = DEFAULT_MANIFEST_REPORT,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    report_path: str | Path = DEFAULT_REPORT,
    provenance_path: str | Path = DEFAULT_PROVENANCE,
    accessed: str,
    env_lock: str | None = None,
) -> dict[str, Any]:
    """Glob the shard partials → the certified ρ-pilot report + provenance sidecar.

    The sweep grid is taken from the partials themselves (:func:`axes_from_grid`), not
    re-assumed here, so reduce always matches whatever grid the shards actually scanned.
    Fail-closed: if :func:`validate_report` finds any problem the report is **not** written,
    so a broken scan never leaves a green artifact behind.
    """
    partials = load_partials(partial_paths)
    per_genome = [g for o in partials for g in o["per_genome"]]
    thresholds, min_spans, gap_merges = axes_from_grid(partials[0]["grid"])
    manifest_ok, manifest_total_bp, manifest_total_mbp = _load_manifest(manifest_report)

    report = build_report(
        per_genome,
        partials,
        thresholds=thresholds,
        min_spans=min_spans,
        gap_merges=gap_merges,
        manifest_total_bp=manifest_total_bp,
        manifest_total_mbp=manifest_total_mbp,
        manifest_ok=manifest_ok,
        checkpoint_sha256=provenance.sha256_file(checkpoint),
        accessed=accessed,
    )
    problems = validate_report(report)
    if problems:
        raise RhoPilotError(
            "ρ-pilot report failed validation (nothing certified):\n  - " + "\n  - ".join(problems)
        )

    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance.write_provenance(
        provenance_path,
        rule="slurm/p2/scan_pilot.sbatch :: tbox_finder.infer.rho_pilot reduce",
        script="src/tbox_finder/infer/rho_pilot.py",
        seed=provenance.DEFAULT_SEED,
        inputs=[str(checkpoint), str(manifest_report)],
        outputs=[str(rp)],
        env_lock=env_lock,
        adr="ADR-0003",
        extra={
            "step": STEP,
            "genome_dir": str(DEFAULT_GENOME_DIR),
            "n_genomes": report["n_genomes"],
            "scanned_bp": report["scanned_bp"],
            "scanned_mbp": report["scanned_mbp"],
            "digest": report["digest"],
            "w_windows_per_sec_per_gpu": report["throughput"]["w_windows_per_sec_per_gpu"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "accessed": accessed,
            "note": (
                "ρ = Stage-1 candidates / Mbp measured over the 100 pilot genomes with the "
                "job-680 production checkpoint (ADR-0003 D6). ρ is a measured ops number that "
                "sizes the homolog-DB + negative-window fetch; it pins no ADR value and asserts "
                "no scientific claim (ADR-0005 D3 freezes production values at the phase gate)."
            ),
        },
    )
    return report


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tbox_finder.infer.rho_pilot",
        description="P2-10c'-c-ii — the ρ-pilot scan driver (scan-shard | reduce)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan-shard", help="scan one shard of pilot genomes on one GPU")
    s.add_argument("--genome-dir", default=str(DEFAULT_GENOME_DIR))
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--n-shards", type=int, required=True)
    s.add_argument("--gpu-index", type=int, required=True)
    s.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    s.add_argument("--out", required=True, help="node-local partial JSON path")
    s.add_argument("--batch-size", type=int, default=DEFAULT_SCAN_BATCH)

    r = sub.add_parser("reduce", help="sum shard partials → the certified ρ-pilot report")
    r.add_argument("--partials", nargs="+", required=True)
    r.add_argument("--manifest-report", default=str(DEFAULT_MANIFEST_REPORT))
    r.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    r.add_argument("--report", default=str(DEFAULT_REPORT))
    r.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    r.add_argument("--accessed", required=True, help="ISO date the scan ran (provenance)")
    r.add_argument("--env-lock", default=None)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if args.cmd == "scan-shard":
        partial = scan_shard(
            genome_dir=Path(args.genome_dir),
            shard=args.shard,
            n_shards=args.n_shards,
            gpu_index=args.gpu_index,
            checkpoint=Path(args.checkpoint),
            out_path=Path(args.out),
            batch_size=args.batch_size,
        )
        print(
            f"shard {partial['shard']}/{partial['n_shards']} scanned "
            f"{partial['n_genomes']} genomes, {partial['n_windows']} windows "
            f"in {partial['scan_seconds']:.1f}s GPU",
            file=sys.stderr,
        )
        return 0
    if args.cmd == "reduce":
        report = reduce_partials(
            args.partials,
            manifest_report=Path(args.manifest_report),
            checkpoint=Path(args.checkpoint),
            report_path=Path(args.report),
            provenance_path=Path(args.provenance),
            accessed=args.accessed,
            env_lock=args.env_lock,
        )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("per_genome", "rho_surface")},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise RhoPilotError(f"unknown command {args.cmd!r}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
