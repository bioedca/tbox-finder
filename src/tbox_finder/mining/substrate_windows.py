"""substrate_windows.py — the A8 production tiling/shard-spec emitter (P2-10c′-shardspec).

ADR-0005 **A8** pins a model-independent, **sharded** ``cmsearch`` pre-scan over the fetched
GTDB/RefSeq negative substrate (the union-prior mask is fail-open off-catalogue). The GATE /
consumer of that scan already exists — :mod:`tbox_finder.mining.substrate_prescan` ships
:func:`~tbox_finder.mining.substrate_prescan.read_shard_spec`,
:func:`~tbox_finder.mining.substrate_prescan.build_shard_segment`, and the eight must-fire
clauses (i)–(viii) — and ``slurm/p2/substrate_prescan.sbatch`` asserts a per-shard
``$SUBSTRATE_DIR/shard_<tid>.json`` "produced by the P2-10c′ fetch/tiling step". **Nothing
produced it for a real run**: the only code that emitted those specs was the golden-fixture
minter (``scripts/make_substrate_prescan_control.py``), which *fabricates* locus-free
background windows over fake ``GCA_FIX*`` accessions. This module is the missing **PRODUCER**.

It does two things over the 2,500 genomes fetched at ``data/interim/production_genomes/<acc>.fna``
(P2-10c′-fetch):

1. ``build-manifest`` — tile every genome at the **frozen** 1024/512 scan geometry and record a
   **FASTA-independent per-genome window-count manifest** (parquet + audit report). This is
   (a) the clause-(i) ``expected`` basis A8 pin 4 requires ("derived from the fetched substrate's
   recorded per-genome window totals … NOT re-counted from the FASTA the scan reads") and
   (b) a valid clause-(viii) within-genome **shrink baseline** — its ``per_genome[].n_windows``
   is exactly what :func:`substrate_prescan.prior_genome_windows_from_fetch_report` reads (the
   committed *fetch* report carries only ``total_bp``, so it is not a valid baseline; this one is).

2. ``emit-specs`` — partition the selected genomes across ``NSHARDS`` shards by the
   deterministic LPT-by-work sharder and write one ``read_shard_spec``-shaped
   ``{shard, expected_production_windows, genome_windows, production}`` JSON per shard, with the
   invariant ``expected_production_windows == len(production) == Σ genome_windows`` (also
   enforced downstream by :func:`substrate_prescan.build_shard_segment`). ``expected`` is sourced
   from the frozen manifest (parquet); ``production`` from a live FASTA re-tile — two independent
   reads, so a scan-time transfer truncation of ``production`` turns clause (i) FALSE.

**Reuse, not fork (promote-don't-duplicate).** The tiling geometry is the canonical scan tiler
:func:`tbox_finder.data.window_dataset.tile_windows` (PRD §6; contig ≤ 1024 nt → one window,
no minimum-window drop, the trailing window anchored to the 3′ end); the FASTA reader is
:func:`tbox_finder.mining.pilot_fetch.iter_fasta_records`; the shard partitioner is
:func:`tbox_finder.infer.rho_pilot.assign_shards` (LPT, pure partition, used by the ρ-pilot scan);
the spec shape is validated against :func:`substrate_prescan.read_shard_spec`; provenance is
:mod:`tbox_finder.provenance`. Nothing here re-implements them.

**Honesty (CLAUDE.md §10.3).** Pure window **geometry** — it reads DNA and counts/emits 1024-nt
windows. It runs **no** ``cmsearch``, loads **no** model, asserts **no** contamination number,
and pins **no** threshold, **no** ρ, and **no** ADR value: A8 is already signed (2026-07-23),
so this is *execution* of a frozen pin, not a new decision. The scan itself — and the
``reports/p2/substrate_prescan.json`` gate report — are downstream at P2-10c′/P2-10e.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from tbox_finder import provenance
from tbox_finder.data.window_dataset import STRIDE_NT, WINDOW_NT, tile_windows
from tbox_finder.infer.rho_pilot import assign_shards
from tbox_finder.mining import substrate_prescan as sp
from tbox_finder.mining.pilot_fetch import iter_fasta_records

# ── Schema / provenance ──────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"
STEP = "P2-10c'-shardspec"

# ── Canonical artifact paths (one path per artifact) ─────────────────────────
MINING_DIR = "data/processed/mining"
GENOME_DIR = "data/interim/production_genomes"
SELECTION_MANIFEST = f"{MINING_DIR}/production_genomes_v0.parquet"
WINDOW_MANIFEST = f"{MINING_DIR}/production_windows_v0.parquet"
WINDOW_PROVENANCE = f"{MINING_DIR}/production_windows_v0.provenance.json"
WINDOW_REPORT = "data/processed/audits/production_windows_report.json"

# ── ADR provenance metadata ──────────────────────────────────────────────────
WINDOW_ADR = "ADR-0005"  # A8 (the pre-scan geometry + clauses); selection is ADR-0006 A1
WINDOW_RULE = "workflow/rules/data.smk :: emit_substrate_windows"
WINDOW_SCRIPT = "src/tbox_finder/mining/substrate_windows.py"

WINDOW_NOTES = (
    "P2-10c′-shardspec (ADR-0005 A8): the REAL production tiling/shard-spec emitter for the "
    "off-catalogue cmsearch substrate pre-scan. build-manifest tiles the 2,500 fetched genomes "
    "at the frozen 1024/512 scan geometry (window_dataset.tile_windows) and records a "
    "FASTA-independent per-genome window-count manifest — the clause-(i) `expected` basis and a "
    "valid clause-(viii) shrink baseline (per_genome[].n_windows, which the fetch report lacks). "
    "emit-specs partitions the selection across NSHARDS (rho_pilot.assign_shards, LPT) and writes "
    "read_shard_spec-shaped shard JSONs. Pure geometry: no cmsearch, no model, no threshold, no ρ, "
    "no contamination claim — the scan is downstream (P2-10e). Pins no ADR value beyond A8's "
    "already-signed geometry."
)


class SubstrateWindowsError(RuntimeError):
    """Raised on a malformed genome, a broken window invariant, or a missing input."""


def _geometry_is_consistent() -> bool:
    """The tiling geometry this module uses equals the gate's frozen geometry.

    :mod:`substrate_prescan` freezes ``WINDOW_NT``/``STRIDE_NT`` independently (its gate math
    reads them); the tiler here imports them from :mod:`window_dataset`. Both trace to PRD §6 /
    ADR-0005 D3, but a silent drift between the producer and the gate would emit specs the gate
    counts against a different geometry. A drift-guard test asserts this returns True.
    """
    return WINDOW_NT == sp.WINDOW_NT and STRIDE_NT == sp.STRIDE_NT


# ═════════════════════════════════════════════════════════════════════════════
# Tiling — provenance-carrying windows over a genome's contigs
# ═════════════════════════════════════════════════════════════════════════════
def window_name(accession: str, contig_index: int, start: int) -> str:
    """A unique, whitespace-free window name (clause (v) requires unique names).

    ``<accession>:c<contig_index>:<start>`` — the accession prefix makes names unique **across**
    genomes, ``(contig_index, start)`` unique **within** a genome; no whitespace, so the
    ``cmsearch`` hit→window join (on the exact target name) cannot collapse two windows.
    """
    return f"{accession}:c{contig_index}:{start}"


def iter_genome_windows(accession: str, fasta_text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(window_name, window_seq)`` for every 1024/512 tile of every contig of a genome.

    Contigs are read with :func:`pilot_fetch.iter_fasta_records` (contig-aware, uppercased). Each
    contig is tiled with the canonical :func:`window_dataset.tile_windows`, so a contig ≤ 1024 nt
    yields exactly one window (its own sequence, un-padded — ``cmsearch`` scans variable-length
    targets; zero-flank padding is a *model-tokenizer* concern, not a homology-search one), and
    the trailing window is anchored to the contig 3′ end. Empty contigs are skipped (a genome must
    not contribute a zero-length window; ``tile_windows`` rejects a non-positive length).
    """
    for ci, (_header, seq) in enumerate(iter_fasta_records(fasta_text)):
        if not seq:
            continue
        for start in tile_windows(len(seq), window=WINDOW_NT, stride=STRIDE_NT):
            yield window_name(accession, ci, start), seq[start : start + WINDOW_NT]


def count_genome_windows(fasta_text: str) -> tuple[int, int, int]:
    """``(n_contigs, total_bp, n_windows)`` for a genome's FASTA text — the manifest row basis.

    ``n_windows`` is re-derived as ``Σ_contigs len(tile_windows(len(contig)))`` (never read back
    from a prior count), so a manifest that claims a window total the geometry does not produce is
    caught by the re-derivation test (MEMORY: gate clauses need re-derivation).
    """
    records = iter_fasta_records(fasta_text)
    n_contigs = 0
    total_bp = 0
    n_windows = 0
    for _header, seq in records:
        if not seq:
            continue
        n_contigs += 1
        total_bp += len(seq)
        n_windows += len(tile_windows(len(seq), window=WINDOW_NT, stride=STRIDE_NT))
    return n_contigs, total_bp, n_windows


def read_genome_fasta(genome_dir: str | Path, accession: str) -> str:
    """Read one fetched genome FASTA (``<genome_dir>/<accession>.fna``); fail loud if absent."""
    path = Path(genome_dir) / f"{accession}.fna"
    if not path.is_file():
        raise SubstrateWindowsError(
            f"genome FASTA missing for selected accession {accession!r}: {path} — the scanned set "
            "must equal the git-frozen selection (ADR-0005 A8 clause viii); a missing genome is a "
            "silent-shrink failure, not a skip."
        )
    return path.read_text(encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# The per-genome window-count manifest (build-manifest)
# ═════════════════════════════════════════════════════════════════════════════
def load_selection_accessions(manifest: str | Path = SELECTION_MANIFEST) -> list[str]:
    """The git-frozen selected-genome accession set — reuses the gate's reader (one source)."""
    return sp.load_selection_accessions(manifest)


def build_window_rows(
    genome_dir: str | Path, accessions: Sequence[str]
) -> list[dict[str, int | str]]:
    """One row per selected genome: ``{assembly_accession, n_contigs, total_bp, n_windows}``.

    Streams one genome at a time (never holds > one genome's FASTA in memory), so the full 6.7 GB
    substrate is tiled in bounded memory. Rows are sorted by accession for a stable, diff-friendly
    manifest. Every selected accession must have a fetched FASTA (fail loud otherwise) — the
    manifest's accession set is the substrate the clause-(viii) set-equality is measured against.
    """
    rows: list[dict[str, int | str]] = []
    for acc in sorted(str(a) for a in accessions):
        n_contigs, total_bp, n_windows = count_genome_windows(read_genome_fasta(genome_dir, acc))
        if n_windows < 1:
            raise SubstrateWindowsError(
                f"genome {acc} tiled to 0 windows (n_contigs={n_contigs}, total_bp={total_bp}) — a "
                "fetched genome with no scannable sequence should never reach the substrate."
            )
        rows.append(
            {
                "assembly_accession": acc,
                "n_contigs": int(n_contigs),
                "total_bp": int(total_bp),
                "n_windows": int(n_windows),
            }
        )
    return rows


def _stats(values: Sequence[int]) -> dict[str, int]:
    """min / p50 (median, rounded) / max — the report's per-genome distribution summary."""
    return {
        "min": int(min(values)),
        "p50": int(round(statistics.median(values))),
        "max": int(max(values)),
    }


def build_window_report(rows: Sequence[dict[str, int | str]]) -> dict[str, object]:
    """The audit + clause-(viii) shrink baseline.

    Shape mirrors the fetch report's ``{... , per_genome: [{assembly_accession, n_windows, …}]}``
    so :func:`substrate_prescan.prior_genome_windows_from_fetch_report` can read this directly as a
    future ``--prior-fetch-report`` baseline. The global ``expected_production_windows`` (= Σ
    per-genome ``n_windows``) is the FASTA-independent denominator clause (i) cross-checks against.
    """
    if not rows:
        raise SubstrateWindowsError("no genome rows — refusing to certify an empty substrate")
    per_genome = sorted(rows, key=lambda r: str(r["assembly_accession"]))
    n_windows = [int(r["n_windows"]) for r in per_genome]
    total_bp = [int(r["total_bp"]) for r in per_genome]
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": WINDOW_ADR,
        "window_nt": WINDOW_NT,
        "stride_nt": STRIDE_NT,
        "n_genomes": len(per_genome),
        "total_bp": int(sum(total_bp)),
        "expected_production_windows": int(sum(n_windows)),
        "genome_windows_stats": _stats(n_windows),
        "genome_bp_stats": _stats(total_bp),
        "per_genome": per_genome,
    }


def write_window_manifest(rows: Sequence[dict[str, int | str]], out_parquet: str | Path) -> Path:
    """Write the per-genome window-count parquet (accession, n_contigs, total_bp, n_windows)."""
    import pandas as pd  # local import: keep the module light for the bare-import tier

    df = pd.DataFrame(
        sorted(rows, key=lambda r: str(r["assembly_accession"])),
        columns=["assembly_accession", "n_contigs", "total_bp", "n_windows"],
    )
    p = Path(out_parquet)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def build_manifest(
    *,
    genome_dir: str | Path = GENOME_DIR,
    selection_manifest: str | Path = SELECTION_MANIFEST,
    out_parquet: str | Path = WINDOW_MANIFEST,
    provenance_path: str | Path = WINDOW_PROVENANCE,
    report_path: str | Path = WINDOW_REPORT,
    env_lock: str | None = None,
) -> int:
    """Tile the fetched substrate → the per-genome window-count manifest + report + provenance.

    Returns the number of genomes recorded. Pure geometry; no scan, no model, no threshold.
    """
    if not _geometry_is_consistent():
        raise SubstrateWindowsError(
            "tiling geometry drift: window_dataset "
            f"({WINDOW_NT}/{STRIDE_NT}) != substrate_prescan ({sp.WINDOW_NT}/{sp.STRIDE_NT})"
        )
    accessions = load_selection_accessions(selection_manifest)
    rows = build_window_rows(genome_dir, accessions)
    report = build_window_report(rows)
    out_parquet = str(out_parquet)
    write_window_manifest(rows, out_parquet)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    provenance.write_provenance(
        provenance_path,
        rule=WINDOW_RULE,
        script=WINDOW_SCRIPT,
        seed=provenance.DEFAULT_SEED,
        inputs=[str(selection_manifest)],
        outputs=[out_parquet, str(report_path)],
        env_lock=env_lock,
        adr=WINDOW_ADR,
        extra={
            "step": STEP,
            "notes": WINDOW_NOTES,
            "genome_dir": str(genome_dir),
            "n_genomes": len(rows),
            "expected_production_windows": report["expected_production_windows"],
            "window_nt": WINDOW_NT,
            "stride_nt": STRIDE_NT,
        },
    )
    return len(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Shard-spec emission (emit-specs) — the read_shard_spec-shaped scan input
# ═════════════════════════════════════════════════════════════════════════════
def load_manifest_window_counts(manifest: str | Path = WINDOW_MANIFEST) -> dict[str, int]:
    """``{accession: n_windows}`` from the committed window manifest (the ``expected`` source)."""
    import pandas as pd  # local import: keep the module light for the bare-import tier

    df = pd.read_parquet(manifest, columns=["assembly_accession", "n_windows"])
    return {str(a): int(n) for a, n in zip(df["assembly_accession"], df["n_windows"], strict=True)}


def build_shard_spec(
    shard: int,
    accessions: Sequence[str],
    *,
    genome_dir: str | Path,
    expected_by_accession: dict[str, int],
) -> dict[str, object]:
    """One ``read_shard_spec``-shaped spec for a shard's genomes.

    ``expected_production_windows`` is summed from the **frozen manifest**
    (``expected_by_accession``); ``production`` is a **live FASTA re-tile**. The two must agree at
    emit time (a mismatch means the FASTA changed since build-manifest — fail loud), while remaining
    independent sources so a scan-time truncation of ``production`` is caught by clause (i).
    ``genome_windows[acc]`` is the live per-genome window count, and ``Σ genome_windows ==
    len(production)`` by construction (names are globally unique) — the exact invariant
    :func:`substrate_prescan.build_shard_segment` asserts.
    """
    production: dict[str, str] = {}
    genome_windows: dict[str, int] = {}
    for acc in accessions:
        gw = dict(iter_genome_windows(acc, read_genome_fasta(genome_dir, acc)))
        before = len(production)
        production.update(gw)
        if len(production) - before != len(gw):
            raise SubstrateWindowsError(f"window-name collision emitting shard {shard} at {acc}")
        genome_windows[acc] = len(gw)
    expected = int(sum(expected_by_accession[acc] for acc in accessions))
    if expected != len(production):
        raise SubstrateWindowsError(
            f"shard {shard}: manifest expected {expected} != live-tiled "
            f"{len(production)} windows — the FASTA changed since build-manifest, or is stale."
        )
    if int(sum(genome_windows.values())) != len(production):
        raise SubstrateWindowsError(f"shard {shard}: Σ genome_windows != len(production)")
    return {
        "shard": int(shard),
        "expected_production_windows": expected,
        "genome_windows": genome_windows,
        "production": production,
    }


def emit_shard_specs(
    *,
    genome_dir: str | Path = GENOME_DIR,
    window_manifest: str | Path = WINDOW_MANIFEST,
    selection_manifest: str | Path = SELECTION_MANIFEST,
    n_shards: int,
    out_dir: str | Path,
) -> list[Path]:
    """Partition the selection into ``n_shards`` shards and write ``shard_<tid>.json`` for each.

    Partitioning is the deterministic LPT-by-window-count sharder (:func:`rho_pilot.assign_shards`,
    keyed on ``n_windows`` — the true ``cmsearch`` work per genome). The **union of every shard's
    genomes equals the entire selection set** (assign_shards is a pure partition), which is exactly
    what clause (viii) ``set_ok`` requires. Each spec is round-tripped through
    :func:`substrate_prescan.read_shard_spec` before it is trusted.
    """
    if not _geometry_is_consistent():
        raise SubstrateWindowsError(
            "tiling geometry drift: window_dataset "
            f"({WINDOW_NT}/{STRIDE_NT}) != substrate_prescan ({sp.WINDOW_NT}/{sp.STRIDE_NT})"
        )
    accessions = load_selection_accessions(selection_manifest)
    expected_by_accession = load_manifest_window_counts(window_manifest)
    missing = [a for a in accessions if a not in expected_by_accession]
    if missing:
        raise SubstrateWindowsError(
            f"{len(missing)} selected genome(s) absent from the window manifest "
            f"(e.g. {missing[:3]}) — re-run build-manifest to cover the whole selection."
        )
    sized = [(acc, int(expected_by_accession[acc])) for acc in accessions]
    shards = assign_shards(sized, n_shards)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for tid, accs in enumerate(shards):
        spec = build_shard_spec(
            tid, accs, genome_dir=genome_dir, expected_by_accession=expected_by_accession
        )
        path = out_dir / f"shard_{tid}.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        sp.read_shard_spec(path)  # round-trip: refuse a spec the gate cannot read
        written.append(path)
    return written


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbox_finder.mining.substrate_windows")
    sub = p.add_subparsers(dest="cmd", required=True)

    bm = sub.add_parser(
        "build-manifest", help="tile the fetched substrate → per-genome window manifest"
    )
    bm.add_argument("--genome-dir", default=GENOME_DIR)
    bm.add_argument("--selection", default=SELECTION_MANIFEST)
    bm.add_argument("--out", default=WINDOW_MANIFEST)
    bm.add_argument("--provenance", default=WINDOW_PROVENANCE)
    bm.add_argument("--report", default=WINDOW_REPORT)
    bm.add_argument("--env-lock", default=None)

    es = sub.add_parser(
        "emit-specs", help="partition into shards → read_shard_spec JSONs (scan input)"
    )
    es.add_argument("--genome-dir", default=GENOME_DIR)
    es.add_argument("--manifest", default=WINDOW_MANIFEST)
    es.add_argument("--selection", default=SELECTION_MANIFEST)
    es.add_argument("--n-shards", type=int, required=True)
    es.add_argument("--out-dir", required=True)

    a = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.cmd == "build-manifest":
        n = build_manifest(
            genome_dir=a.genome_dir,
            selection_manifest=a.selection,
            out_parquet=a.out,
            provenance_path=a.provenance,
            report_path=a.report,
            env_lock=a.env_lock,
        )
        print(f"build-manifest: {n} genomes tiled → {a.out}")
        return 0
    if a.cmd == "emit-specs":
        paths = emit_shard_specs(
            genome_dir=a.genome_dir,
            window_manifest=a.manifest,
            selection_manifest=a.selection,
            n_shards=a.n_shards,
            out_dir=a.out_dir,
        )
        print(f"emit-specs: {len(paths)} shard specs → {a.out_dir}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
