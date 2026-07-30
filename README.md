# tbox-finder

Genome-wide discovery of **T-box riboswitches** and expansion of their known
phylogenetic distribution — an open-science, publicly-versioned project.

A two-stage detector scans prokaryotic genomes/metagenomes and emits
per-nucleotide T-box structural-element annotations with **calibrated**
confidence. The scientific value is defensible, non-circular discovery, so
data-leakage control, calibration, and orthogonal validation are first-class.

> **Status:** **Phase 2 — Stage-1 training: in progress** (Phase 1 complete).
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

- **Phase 1 — Backbones & heads (2026-07-15).** Both backbones are validated and their
  transfer risk is retired. **Stage 1** (DNA, Caduceus-PS 7.73M): CUDA kernels verified on
  the `sm_86` A4000, RC-equivariance holds, and a per-nucleotide 8-class segmentation head
  clears the **binding transfer go/no-go** — per-nt F1 over the three core elements
  **0.9999** vs a background-only baseline of 0.0 — and **reproduces** on a seeded re-run
  within the pre-registered tolerance. **Stage 2** (RNA, RiNALMo-giga 650M): the
  `multimolecule` mirror's encoder is **bit-identical** to the official release (497/497
  tensors), and a LoRA fine-tune step (`r=16, α=32, dropout=0.05, all-linear`, bf16 +
  gradient checkpointing; 1.94% of params trainable) runs end-to-end on **one 16 GB A4000
  at 1.484 GiB peak** with FlashAttention-2 confirmed by forward on `sm_86`. The full
  transfer-fallback ladder (frozen-embedding probe, GTDB continued-pretraining,
  NT-multispecies) is **built but untriggered** — the go/no-go passed.
  *Still no detector and no discovery result: Phase 1 ships validated backbones, and its
  smoke runs are mechanics/expressivity probes, not generalization claims.*

- **Phase 2 — Stage-1 training (in progress).** Stage-1 **architecture ablations** are
  measured (`reports/p2/ablation_table.json`): on a disjoint 830-record / 469-cluster
  selection fold, a CRF head, a gated reverse-complement combination and both smaller
  backbones span 0.8890–0.9416 min-core-F1 against the promoted baseline's 0.9386, and
  **no arm is reported as separated**. The limit is run-to-run training variance, not fold
  size: an **identical-seed, identical-config re-run** of one arm moved 0.9242 → **0.7568**,
  the whole shift in one element (antiterminator per-nt F1 0.9242→0.7568, boundary IoU
  0.8590→0.6087) while its AUPRC barely moved (0.9843→0.9705) — the argmax boundary moved,
  not the representation. Cluster-blocked bootstrap CIs resample a *fixed* checkpoint, so
  they exclude that variance and understated it ~4×; separation is now judged only on
  replicate-widened intervals. The **throughput** half is unaffected (a property of the
  architecture, not of a trained checkpoint): **69.5 / 268.4 / 423.5** windows/s/GPU
  (1.0× / 3.86× / 6.10×) across the three pinned checkpoints.
  *Selection-fold measurements — **not** generalization results; GATE-4 is graded later in
  Phase 2 on the leave-one-order-out holdout. Stage-1 outputs here are uncalibrated
  (temperature scaling is a later Phase-2 step).*

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
