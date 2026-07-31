# tbox-finder

Genome-wide discovery of **T-box riboswitches** and expansion of their known
phylogenetic distribution — an open-science, publicly-versioned project.

A two-stage detector scans prokaryotic genomes/metagenomes and emits
per-nucleotide T-box structural-element annotations with **calibrated**
confidence. The scientific value is defensible, non-circular discovery, so
data-leakage control, calibration, and orthogonal validation are first-class.

> **Status:** **Phase 2 — Stage-1 training: complete.** Next: Phase 3 (Stage-2
> re-ranker + integration). Methodology decisions are pinned in
> `docs/decisions/` (ADRs); a Phase-2 model-card draft is at
> [`docs/model_card.md`](docs/model_card.md), and the released model/dataset
> cards will document intended use, splits, calibration, and limitations.

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

- **Phase 2 — Stage-1 training (2026-07-31).** **GATE-4 passes.** A per-nucleotide
  8-class Caduceus-PS segmenter reaches a **minimum per-element per-nt F1 of 0.952**
  over the three core elements {Stem I, Specifier, Antiterminator} against a
  pre-registered floor of **0.80**, with a homology-cluster-blocked bootstrap 95%
  interval of **[0.943, 0.960]** (1,029 clusters, 2,000 resamples) that clears the floor
  by its *lower* bound. Per element: Stem I 0.975, Antiterminator 0.970, Specifier 0.952
  (the minimum, and therefore the statistic); boundary IoU 0.951 / 0.941 / 0.908 through
  the deployed overlapping-window reconciliation operator. **The gate grades an
  evaluation twin, not the shipped checkpoint** — the shipped scanner trained on the
  entire in-distribution fold and has no held-out population, so a twin trained with the
  graded clusters withheld (8,303 → 7,099 records, 0 shared clusters and 0 shared
  records) was graded in its place; 0.952 is therefore a **conservative proxy**.
  *The number beside it that matters more for discovery:* on the **leave-one-order-out**
  holdout the same checkpoint falls to **~0.72** macro per-element F1 across 30 held-out
  orders. GATE-4 grades in-distribution segmentation quality — an explicitly labelled
  reference, **not** a generalization test; the generalization claim is graded at GATE-1
  in Phase 4. Two further Stage-1 checkpoints are retained: the **class-II-CM-naive
  anti-mimicry ablation** (scored Stage-1-only at GATE-1) and the GATE-4 twin.
  Calibration machinery is fitted (T = 0.9896) but **not shipped** and **not gated** —
  every Phase-2 number is computed on the uncalibrated posterior. The §9.1
  hard-negative-mining loop is machinery-complete, measured, and **deliberately not
  executed** (1 of its 3 spare-rule evidence backends exists, so the rule fails closed
  and a round would be a verified no-op) — deferred, not cancelled.

  Stage-1 **architecture ablations** are
  measured (`reports/p2/ablation_table.json`): on a disjoint 830-record / 469-cluster
  selection fold, arms are **replicated** and separation is judged on replicate-widened
  intervals (re-runs reproduce closely: spread **0.0001** over 3 replicates of the 471k
  backbone, 0.0000 over 2 of the 1.93M, 0.0006 over 2 of the baseline). The 1.93M backbone
  (0.9191, [0.8920, 0.9412]) is **not distinguishable** from the baseline (0.9386/0.9393,
  [0.9125, 0.9576]), while the smallest **471k** checkpoint separates **downward**
  (0.8890, [0.8581, 0.9047]) — the floor of the throughput-driven downward search.
  Measured forward-only scan rate: **69.5 / 268.4 / 423.5** windows/s/GPU
  (1.0× / 3.86× / 6.10×), so 1.93M buys 3.86× throughput at no resolvable accuracy cost.
  Two unreplicated arms are reported **without** a verdict — the CRF head, and a gated
  RC combination withheld for cause (its two runs disagreed 0.9242 vs 0.7568, entirely in
  one element whose AUPRC held but whose boundary IoU collapsed 0.8590→0.6087: a
  decision-threshold shift under uncalibrated outputs, unattributed since the runs were at
  different commits).
  *These ablation figures are selection-fold measurements — **not** generalization
  results, and not the gated statistic. Phase 2 ships a trained segmenter and one passing
  method gate; it ships no discovery result, no calibrated output, and no evidence about
  novel lineages.*

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
