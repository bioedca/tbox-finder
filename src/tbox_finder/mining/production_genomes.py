"""production_genomes.py — the production genome-selection manifest (ADR-0006 A1; P2-10c′-g).

ADR-0006 **D7** pins the model-independent homolog-search *method* (nhmmer / BLAST /
cmsearch-from-candidate) but names **no target database**; **Amendment A1** pins it — and
the same independent-genome set is the source for **ADR-0005 D14**'s 1,024-nt
negative-window supply (P2-10e), the "one acquisition, two blockers" decision. The pinned
substrate is a reproducible, phylum-stratified selection of **~2,500 GTDB R232 species
representatives**.

This module implements the **SELECTION half only**: it turns the pinned R232
species-rep crosswalk (``sp_clusters_r232.tsv``) into an accession-list manifest. It
fetches **no** genomes (the ρ-sized SLURM fetch does that), builds **no** homolog DB,
carves **no** negatives, runs **no** cmsearch, and pins **no** D7/A8 inclusion threshold
and **no** ρ value — all downstream. It therefore carries no §10.3 fabrication surface and
pins no ADR value beyond the A1 source+count. The manifest is the **git-frozen input to
ADR-0005 A8 clause (viii)** (``genome_completeness`` — the fetched-and-scanned set must
equal exactly this selection).

**Reuse, not fork (promote-don't-duplicate).** The pure, unit-tested selector
(:func:`~tbox_finder.mining.pilot_genomes.select_pilot_genomes`) and the guarded builder
(:func:`~tbox_finder.mining.pilot_genomes.build`, with its four must-fire non-vacuity
guards) are shared with the P2-10c′-a pilot verbatim. Only the **defaults** (``n_target``
2,500, ``per_phylum_cap`` 20) and the **metadata** (ADR-0006, the production rule, the
production notes) differ — the load-bearing correctness logic lives in one place.

**Pinned selection (ADR-0006 A1).** ``n_target=2500``, ``per_phylum_cap=20``, ``seed=42``.
Over R232's 197 phyla (172 bacterial + 25 archaeal), the breadth-first round-robin yields
**2,500 reps spanning all 197 phyla — 2,109 bacteria + 391 archaea**. ``per_phylum_cap=20``
is the *smallest* cap that reaches 2,500 (``max_achievable`` = 2,589 at cap 20; 2,493 at
cap 19). Size is a **projection** — ~2,500 reps × the pilot-measured ~2.88 Mbp/rep ≈ ~7 GB
decompressed FASTA — matched to the **low measured ρ ≈ 0.25–0.59/Mbp** (job 687); the
actual bp/size is measured only at the fetch, never here. The count is a **revisable ops
target**: if P2-10e's round count ``N`` or D7's ``min_sequences`` homolog floor later
demand more depth, the manifest is re-selected at a higher ``n_target`` under the same
seeded algorithm.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from tbox_finder import provenance
from tbox_finder.mining import pilot_genomes as pg

#: Canonical artifact paths (one path per artifact — imp.md §"Canonical artifact paths").
PRODUCTION_DIR = "data/processed/mining"
PRODUCTION_MANIFEST = f"{PRODUCTION_DIR}/production_genomes_v0.parquet"
PRODUCTION_PROVENANCE = f"{PRODUCTION_DIR}/production_genomes_v0.provenance.json"
PRODUCTION_REPORT = "data/processed/audits/production_genomes_report.json"

#: ADR-0006 A1 provenance metadata (the only per-invocation difference vs the pilot).
PRODUCTION_ADR = "ADR-0006"
PRODUCTION_RULE = "workflow/rules/data.smk :: select_production_genomes"

#: ADR-0006 A1 pinned selection. ``per_phylum_cap=20`` is the smallest cap reaching the
#: 2,500 target over R232's 197 phyla (max_achievable 2,589 at 20; 2,493 at 19) — set at
#: encode from the frozen R232 crosswalk (sha256 96414cdc…), not fabricated.
PRODUCTION_N_TARGET = 2500
PRODUCTION_PER_PHYLUM_CAP = 20

#: Non-vacuity floors. All 197 R232 phyla are spanned (2,500 ≫ 197 ⇒ pass 1 covers every
#: phylum) and 391 archaeal reps are drawn; these floors are conservative must-fire guards
#: well above the pilot's (50 / 3). The exact full-span (197) + per-domain counts are
#: recorded in the report and asserted exactly in the unit tests.
PRODUCTION_MIN_PHYLA = 150
PRODUCTION_MIN_ARCHAEA = 100

PRODUCTION_NOTES = (
    "P2-10c′-g (ADR-0006 A1): the PRODUCTION genome-selection manifest — the "
    "model-independent homolog-search target DB (ADR-0006 D7) and the independent-genome "
    "source for ADR-0005 D14's 1,024-nt negative-window supply (P2-10e), one acquisition "
    "for two blockers. A reproducible, phylum-stratified selection of ~2,500 GTDB R232 "
    "species representatives, drawn round-robin across all 197 phyla (breadth-first; one "
    "rep per phylum per pass; per_phylum_cap=20 → 2,109 bacteria + 391 archaea). This "
    "manifest is an ACCESSION LIST only: it fetches no genomes (the ρ-sized SLURM fetch "
    "does that), builds no homolog DB, carves no negatives, runs no cmsearch, and pins no "
    "D7/A8 inclusion threshold and no ρ value — all downstream. It is the git-frozen input "
    "to ADR-0005 A8 clause (viii) (genome_completeness). Size PROJECTION ~7 GB decompressed "
    "FASTA (pilot-measured ~2.88 Mbp/rep × 2,500; bp/size measured at fetch, not here); "
    "sized to the low measured ρ ≈ 0.25–0.59/Mbp (job 687). Pins no ADR value beyond the "
    "A1 source+count and carries no discovery claim."
)


def build(
    *,
    crosswalk_path: str = pg.DEFAULT_CROSSWALK,
    out_parquet: str = PRODUCTION_MANIFEST,
    provenance_path: str = PRODUCTION_PROVENANCE,
    report_path: str = PRODUCTION_REPORT,
    seed: int = provenance.DEFAULT_SEED,
    n_target: int = PRODUCTION_N_TARGET,
    per_phylum_cap: int = PRODUCTION_PER_PHYLUM_CAP,
    min_phyla: int = PRODUCTION_MIN_PHYLA,
    min_archaea: int = PRODUCTION_MIN_ARCHAEA,
    env_lock: str | None = None,
) -> int:
    """Select + guard + write the production manifest (delegates to :func:`pilot_genomes.build`).

    Delegation, not duplication: the selector and every must-fire guard are the pilot's;
    only the defaults + ADR-0006 metadata differ.
    """
    return pg.build(
        crosswalk_path=crosswalk_path,
        out_parquet=out_parquet,
        provenance_path=provenance_path,
        report_path=report_path,
        seed=seed,
        n_target=n_target,
        per_phylum_cap=per_phylum_cap,
        min_phyla=min_phyla,
        min_archaea=min_archaea,
        env_lock=env_lock,
        adr=PRODUCTION_ADR,
        rule_name=PRODUCTION_RULE,
        notes=PRODUCTION_NOTES,
        kind="production",
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbox_finder.mining.production_genomes")
    p.add_argument("--crosswalk", default=pg.DEFAULT_CROSSWALK)
    p.add_argument("--out", default=PRODUCTION_MANIFEST)
    p.add_argument("--provenance", default=PRODUCTION_PROVENANCE)
    p.add_argument("--report", default=PRODUCTION_REPORT)
    p.add_argument("--seed", type=int, default=provenance.DEFAULT_SEED)
    p.add_argument("--n-target", type=int, default=PRODUCTION_N_TARGET)
    p.add_argument("--per-phylum-cap", type=int, default=PRODUCTION_PER_PHYLUM_CAP)
    p.add_argument("--min-phyla", type=int, default=PRODUCTION_MIN_PHYLA)
    p.add_argument("--min-archaea", type=int, default=PRODUCTION_MIN_ARCHAEA)
    p.add_argument("--env-lock", default=None)
    a = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    return build(
        crosswalk_path=a.crosswalk,
        out_parquet=a.out,
        provenance_path=a.provenance,
        report_path=a.report,
        seed=a.seed,
        n_target=a.n_target,
        per_phylum_cap=a.per_phylum_cap,
        min_phyla=a.min_phyla,
        min_archaea=a.min_archaea,
        env_lock=a.env_lock,
    )


if __name__ == "__main__":
    raise SystemExit(main())
