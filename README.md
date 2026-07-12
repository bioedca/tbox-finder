# tbox-finder

Genome-wide discovery of **T-box riboswitches** and expansion of their known
phylogenetic distribution — an open-science, publicly-versioned project.

A two-stage detector scans prokaryotic genomes/metagenomes and emits
per-nucleotide T-box structural-element annotations with **calibrated**
confidence. The scientific value is defensible, non-circular discovery, so
data-leakage control, calibration, and orthogonal validation are first-class.

> **Status:** **Phase 0 — Foundation: complete.** Next: Phase 1 (backbones & heads).
> Methodology decisions are pinned in `docs/decisions/` (ADRs); the released
> model/dataset cards will document intended use, splits, calibration, and limitations.

## Phase headlines

- **Phase 0 — Foundation (2026-07-12).** A leakage-controlled corpus of **23,535**
  curated T-box records is partitioned by structure-aware, RF00230-homology-clustered
  **leave-clade-out** splits (max held-out↔train consensus identity **< 0.70**),
  committed as a sequence-free split table and guarded by a **CI-blocking no-leakage
  test** over the real partition. The **non-circular evaluation contract** (GATE-1
  recall@matched-precision vs a `cmsearch` baseline; GATE-2 calibration ECE + discovery
  FDR; GATE-3 per-corpus→project rollup; GATE-4 per-nucleotide segmentation F1) and its
  blinded-frozen thresholds are pinned across **six seed ADRs** (ADR-0001…0006), with
  static decoy pools + union-prior masking and an eval-gate regression harness in place.
  *No detector has been trained yet — Phase 0 ships the foundation, not a discovery
  result.*

## Layout (PRD §16)

| Path | Purpose |
|---|---|
| `src/tbox_finder/` | Library (parsers, labels, metrics, splits). |
| `workflow/` | Snakemake local/CPU DAG (`rules/`, `profiles/slurm/`). |
| `conf/` | Hydra config groups (`model/`, `data/`, `optim/`). |
| `envs/` | Pinned conda-lock environments. |
| `slurm/` | Hand-authored `sbatch` jobs (GPU/heavy stages). |
| `data/{raw,external,interim,processed}/` | Data tiers (`raw`/`external` immutable). |
| `tests/{unit,golden,ml,fixtures}/` | Test layers. |
| `analyses/` | Quarto per-phase dev-logs. |
| `figures/`, `paper/`, `app/` | Figures, manuscript, Svelte discovery atlas. |
| `docker/` | CI reproducibility image. |
| `docs/decisions/` | Architecture Decision Records (ADRs). |

## License

Code: **MIT**. Model weights (at release): **CC-BY-4.0**. Curated-dataset
license: the most-restrictive license compatible with upstream sources, per
the P0 license-compatibility audit.
