# tbox-finder

Genome-wide discovery of **T-box riboswitches** and expansion of their known
phylogenetic distribution — an open-science, publicly-versioned project.

A two-stage detector scans prokaryotic genomes/metagenomes and emits
per-nucleotide T-box structural-element annotations with **calibrated**
confidence. The scientific value is defensible, non-circular discovery, so
data-leakage control, calibration, and orthogonal validation are first-class.

> **Status:** P0 — Foundation (repository scaffold). Methodology decisions are
> pinned in `docs/decisions/` (ADRs); the released model/dataset cards will
> document intended use, splits, calibration, and limitations.

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
