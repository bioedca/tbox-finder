"""production_fetch.py — the ρ-sized production genome fetch (ADR-0006 A1; P2-10c′-fetch).

P2-10c′-g selected the pinned production substrate — 2,500 GTDB R232 species representatives
spanning all 197 phyla (``data/processed/mining/production_genomes_v0.parquet``, an
**accession list**, no sequence; ADR-0006 A1). This module implements the **fetch half**:
it pulls the whole-genome nucleotide sequence for each manifest accession from NCBI and
writes one normalized FASTA per genome, so the downstream homolog-DB build (ADR-0006 D7,
``envs/homology.yml``) and the ADR-0005 D14 negative-window supply have their substrate — the
"one acquisition, two blockers" set. The fetched substrate is gated by the ADR-0005 A8
cmsearch pre-scan **before** it enters the mining/negative pool (a separate downstream step).

**Reuse, not fork (promote-don't-duplicate).** The whole-genome fetcher
(:func:`~tbox_finder.mining.pilot_fetch.build_pilot_fetch`), its fail-closed report/validator,
the resumable NCBI transport, and the ``assembly accession → FtpPath → _genomic.fna.gz``
resolution are the P2-10c′-b pilot's, shared verbatim. Only the per-invocation **metadata**
(the report ``step`` label, ADR-0006 / the production rule / this module as provenance
``script``, a production provenance note) and the two **operational floors**, re-scoped for
2,500 genomes across 197 phyla, differ — the load-bearing fetch + certification logic lives
in one place (a forked copy would fix one and ship the bug in the other).

**Re-scoped floors (ADR-0006 A1 selection properties; implementer choices, not ADR pins).**
``MIN_PHYLA_OK`` is lifted from the pilot's 50 to **150**: the A1 manifest spans all 197 R232
phyla, so a fetch that dropped below 150 has lost a quarter of the divergence the homolog DB
exists to provide (D7's divergent-homolog recall concern) — a CLAUDE.md §7 stop-and-ask, not a
certified fetch. ``MIN_SUCCESS_RATE`` stays 0.90 (a 2,500-genome pull carries some suppressed /
withdrawn assemblies; 0.90 tolerates ≤250 deterministic failures without certifying a broken run).

Honesty (CLAUDE.md §10.3). Like the pilot, this step **fetches sequence and measures only
genome size in base pairs** — it counts no candidates, computes no ρ, builds no homolog DB,
carves no negatives, runs no cmsearch, and pins no D7/A8 inclusion threshold. So it carries no
scientific claim; the shared report asserts those honesty flags and the validator enforces them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tbox_finder.mining import pilot_fetch as pf

#: Canonical production artifact paths (mirror the pilot's ``data/interim`` / audits layout).
PRODUCTION_MANIFEST = Path("data/processed/mining/production_genomes_v0.parquet")
PRODUCTION_GENOME_DIR = Path("data/interim/production_genomes")
PRODUCTION_LOG = Path("data/interim/production_genomes_fetch_log.jsonl")
PRODUCTION_PROVENANCE = Path("data/interim/production_genomes.provenance.json")
PRODUCTION_REPORT = Path("data/processed/audits/production_fetch_report.json")

#: ADR-0006 A1 provenance metadata (the only per-invocation difference vs the pilot fetch).
PRODUCTION_STEP = "P2-10c'-fetch"
PRODUCTION_ADR = "ADR-0006"
#: The SLURM entry the fetch actually runs under (there is no Snakemake rule — the fetch is a
#: §9.3 submit-ack job, not a DAG product); provenance must name it, not the pilot rule.
PRODUCTION_RULE = (
    "slurm/p2/fetch_production_genomes.sbatch :: python -m tbox_finder.mining.production_fetch"
)
#: The module the sbatch invokes; provenance must record THIS, not the pilot module it
#: delegates to, or the artifact claims the wrong author (the P2-10c′-g CodeRabbit major).
PRODUCTION_SCRIPT = "src/tbox_finder/mining/production_fetch.py"

#: Re-scoped operational floors for the 2,500-rep / 197-phylum production fetch. Stricter than
#: the pilot's (50 / 0.90); NOT ADR pins (no PRD/ADR pins a fetch success rate — the
#: ``flank_context.MIN_ANCHOR_RATE`` precedent), so they carry no §7 sign-off, only a §7
#: stop-and-ask on breach. ``MIN_PHYLA_OK`` mirrors ``production_genomes.PRODUCTION_MIN_PHYLA``.
PRODUCTION_MIN_SUCCESS_RATE = 0.90
PRODUCTION_MIN_PHYLA_OK = 150

PRODUCTION_PROVENANCE_NOTE = (
    "P2-10c′-fetch (ADR-0006 A1): the PRODUCTION whole-genome fetch of the 2,500 GTDB R232 "
    "species-representative manifest — the model-independent homolog-search target DB "
    "(ADR-0006 D7) and the independent-genome source for ADR-0005 D14's 1,024-nt "
    "negative-window supply (P2-10e), one acquisition for two blockers. Whole-genome sequence "
    "sourced from NCBI: assembly accession → assembly UID (esearch) → assembly FtpPath "
    "(esummary) → whole-genome _genomic.fna.gz (HTTPS GET + gunzip; all replicons/contigs). "
    "Measures genome SIZE (bp) only — no ρ, no candidate count, no detection threshold, no "
    "homolog DB, no negatives, no cmsearch (§10.3). The fetched substrate is gated by the "
    "ADR-0005 A8 cmsearch pre-scan before it enters the mining/negative pool. gzip integrity "
    "guarantees a complete download (a truncated stream raises)."
)


def build(
    *,
    manifest_parquet: Path = PRODUCTION_MANIFEST,
    genome_dir: Path = PRODUCTION_GENOME_DIR,
    log_path: Path = PRODUCTION_LOG,
    report_path: Path = PRODUCTION_REPORT,
    provenance_path: Path = PRODUCTION_PROVENANCE,
    email: str | None = None,
    api_key: str | None = None,
    limit: int | None = None,
    env_lock: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch every production-manifest genome (delegates to :func:`pilot_fetch.build_pilot_fetch`).

    Delegation, not duplication: the fetcher, its resumable transport, and its fail-closed
    report/validator are the pilot's; only the ADR-0006 metadata + the re-scoped floors differ.
    """
    return pf.build_pilot_fetch(
        manifest_parquet=manifest_parquet,
        genome_dir=genome_dir,
        log_path=log_path,
        report_path=report_path,
        provenance_path=provenance_path,
        email=email,
        api_key=api_key,
        limit=limit,
        env_lock=env_lock,
        step=PRODUCTION_STEP,
        adr=PRODUCTION_ADR,
        rule_name=PRODUCTION_RULE,
        script=PRODUCTION_SCRIPT,
        provenance_note=PRODUCTION_PROVENANCE_NOTE,
        min_success_rate=PRODUCTION_MIN_SUCCESS_RATE,
        min_phyla_ok=PRODUCTION_MIN_PHYLA_OK,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tbox_finder.mining.production_fetch",
        description="P2-10c′-fetch — the ρ-sized production whole-genome fetch (ADR-0006 A1)",
    )
    p.add_argument("--manifest", default=str(PRODUCTION_MANIFEST))
    p.add_argument("--genome-dir", default=str(PRODUCTION_GENOME_DIR))
    p.add_argument("--log", default=str(PRODUCTION_LOG))
    p.add_argument("--report", default=str(PRODUCTION_REPORT))
    p.add_argument("--provenance", default=str(PRODUCTION_PROVENANCE))
    p.add_argument("--email", default=None, help="NCBI contact email (or NCBI_EMAIL)")
    p.add_argument("--api-key", default=None, help="NCBI API key (or NCBI_API_KEY) → 10 req/s")
    p.add_argument("--limit", type=int, default=None, help="probe: cap the genome count")
    p.add_argument("--env-lock", default=None)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = build(
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
