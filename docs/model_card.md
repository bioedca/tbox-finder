# Model card — tbox-finder Stage-1 genomic scanner

> **DRAFT — Phase-2 exit (2026-07-31).** This card documents the Stage-1
> checkpoints as they exist at the end of Phase 2. It is **not** a release card:
> nothing has been pushed to the Hugging Face Hub, no Zenodo DOI is minted, and
> the two-stage system this Stage-1 model belongs to is **not yet built**
> (Stage 2 is Phase 3). The release card is authored at the Phase-7 gate
> (PRD §16, §17; CLAUDE.md §6.3) and will supersede this file.
>
> **No discovery result exists yet.** Phase 2 ships a trained segmenter and one
> passing method gate (GATE-4). The project's headline generalization claim
> (GATE-1) is graded in Phase 4, calibration (GATE-2) in Phase 3/5, and the
> discovery campaign (GATE-3) in Phases 5–6. Numbers in this card must not be
> read as evidence for any of those.

---

## 1. Model summary

`tbox-finder` Stage 1 is a **per-nucleotide, 8-class DNA segmenter** that labels
prokaryotic/archaeal genomic sequence with T-box riboswitch structural elements.
It is the high-recall first stage of a planned two-stage detector (Stage 2, an
RNA precision re-ranker, is Phase 3) and is not a standalone T-box caller.

| | |
|---|---|
| **Task** | per-nucleotide 8-class token classification over DNA |
| **Classes** | `background`, `Stem_I`, `Specifier`, `Stem_II` (subsumes the folded IIA/B pseudoknot extent), `Stem_III`, `Antiterminator_Tbox_seq`, `Terminator`, `Discriminator` |
| **Backbone** | Caduceus-PS `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16` @ revision `d89eeb853136ea64da7feb3d0c8e909771b17ae6` (7.73 M params, `d_model` 256, 16 layers, reverse-complement **equivariant**) |
| **Head** | per-position linear 8-class over the L × `d_model` hidden states (`use_crf=false`); RC hidden-state combination = **`concat`** (directionality-preserving; the averaged form is forbidden by ADR-0005 D15 because the §6 strand-resolver reads element *order*) |
| **Scan geometry** | window **1024 nt**, stride **512 nt**, both strands |
| **Reconciliation** | overlapping-window `log_mean_exp` of per-window log-softmax → arg-max, applied **before** locus construction (ADR-0005 D3 + A3) |
| **Output** | per-position 8-class logits (**uncalibrated** — see §6) |
| **License** | model weights **CC-BY-4.0**. Dataset license follows the Phase-0 compatibility audit and is release-gated on it. |

---

## 2. The three Stage-1 checkpoints (PRD §10.1 two-run split disclosure)

Phase 2 produced **three** Stage-1 artefacts. Only one is shipped. Conflating
them would misattribute a gate result, so each is named with its own fold, its
own grader, and its own release status.

| Checkpoint | Labels / fold | Graded by | Shipped? |
|---|---|---|---|
| **`stage1_production`** — the shipped scanner | full 8-class incl. class II via `TBDB001.cm`-derived structure; full ADR-0004 D5 `nested_train` fold (**8,303** records) | **GATE-2** (P3 ECE / P5 FDR), **GATE-1 arms a–c** (P4), and it runs the **P5 scan / GATE-3**. **It is not itself GATE-4-graded** — see the twin below. | **yes** |
| **`stage1_classII_naive`** — the anti-mimicry ablation | class-I-style / shared-element only: **`TBDB001.cm` withheld**, every class-II record's per-nt target zeroed to all-background (1,200 records, 193,587 element-nt withheld). Same fold, same negative curriculum, same hyper-parameters — `label_source=naive` is the single difference. | **only** the **GATE-1 class-II anti-mimicry sub-arm** (P4), **scored Stage-1-only** — never routed through the class-II-trained production Stage 2, which would confound the ablation (ADR-0005 D9) | no — retained for reproducibility |
| **`stage1_gate4_twin`** — the GATE-4 evaluation twin | identical to the production run except `exclude_gate4_eval=true`: the whole clusters of the GATE-4 graded population are withheld (**8,303 → 7,099** records; 1,204 records in 1,031 clusters removed) | **GATE-4** (P2), and nothing else | no — an evaluation instrument |

**Why a twin exists, and what it costs the reader.** The shipped production
scanner trained on the *entire* in-distribution fold, so it has **no**
in-distribution held-out population at all and cannot be graded on one. Rather
than grade it on data it had seen — or retrain the shipped model on less data —
a second checkpoint was trained with the graded clusters withheld and graded in
its place (ADR-0004 A6). The twin saw **strictly less** data than the shipped
model, so **GATE-4's 0.952 is a conservative proxy for the shipped scanner, not
a measurement of it.** Disjointness is proved rather than assumed: the graded
population shares **0** clusters and **0** records with the twin's training
fold, and the intersection test is positive-controlled
(|`gate4_eval` ∩ `nested_train`| = 1,029 of 1,029), so the zero is real
disjointness and not a namespace mismatch.

**Provenance.** Each checkpoint directory carries a `provenance.json` recording
its rule, script path, git SHA, env-lock hash, seed, and input/output SHA-256s
(PRD §11). All three are DVC-tracked on the cluster-scratch remote.

| Checkpoint | `stage1.pt` SHA-256 | git SHA | step |
|---|---|---|---|
| `stage1_production` | `09931a22…b380940` | `5e29197e` | P2-10d′-b |
| `stage1_classII_naive` | `42fc479e…674240` | `655c76cf` | P2-11 |
| `stage1_gate4_twin` | `0140a8a3…944a29` | `11289740` | P2-14 |

All three share env lock `envs/ml-dna.conda-lock.yml`
(`70b66801…3bb5003`) and seed **42**.

---

## 3. Intended use

**In scope.** Basic research: cataloguing a native bacterial/archaeal
gene-regulatory RNA. Stage 1 is intended to be run as the **high-recall first
pass** of the two-stage pipeline over prokaryotic/archaeal genomic contigs,
emitting candidate loci for a calibrated precision re-ranker and, downstream, a
model-independent orthogonal-validation pipeline (covariation, architecture,
synteny). Its outputs are hypotheses to be filtered, not annotations to be
published.

**Out of scope / not supported.**

- **Stage-1 output alone is not a T-box call.** It runs at a deliberately
  high-recall operating point; the false-positive rate is meant to be absorbed
  by Stage 2 and the orthogonal pipeline, neither of which exists yet.
- **Not a generalization claim.** See §5 — across held-out taxonomic orders
  per-element F1 falls to ~0.72. Applying this checkpoint to a lineage far from
  the ~90 %-Firmicutes training corpus is exactly the regime it is *weakest* in,
  and the regime the project has not yet gated.
- **Not calibrated.** The shipped logits are raw (§6). Do not threshold them as
  probabilities.
- **Not a tRNA finder**, not a general riboswitch finder, and not a claim of
  completeness of the T-box universe (PRD §2.2).

**Responsible-research posture.** T-boxes are native regulatory RNAs; this work
has no engineered-pathogen, gain-of-function, or dual-use design intent. The
principal foreseeable harm is **over-reading**: presenting an uncalibrated
Stage-1 hit, or an in-distribution metric, as confirmed biology in a novel
lineage. This card states the sensitivity bounds so downstream users do not.

---

## 4. Training data and splits

**Corpus.** Derived from the TBDB master table (23,535 curated T-box records)
through the Phase-0 ingest → label-derivation chain; the per-nucleotide targets
live in `data/processed/labels/labels_v0.parquet`
(`6caf46ea…a447e4`), with flanking genomic context in
`data/interim/flank_context/context_v0.parquet` (`f35f25cd…3297742`).

**Splits (ADR-0004 D2/D3/D5).** Structure-aware, **homology-clustered**
partitioning: records are clustered by consensus-column identity over
**per-class covariance-model alignments** (RF00230 for class I, `TBDB001.cm`
for class II, unioned), with identity **≥ 0.70** linking records into a single
cluster that is assigned **wholly to one fold**. The split ladder holds a
**leave-one-order-out** holdout (the generalization axis, graded at GATE-1/P4)
and, inside `nested_train`, an in-distribution genus-stratified reference used
for segmentation quality. The per-record assignment table is committed
(`data/processed/splits/split_assignments.parquet`, `6388ba19…cefba029`,
git-LFS) and a **CI-blocking no-leakage test** (ADR-0004 D7) asserts over that
real full-corpus partition — not a fixture — that no cluster or taxon spans a
fold boundary.

**Negatives (PRD §9.1; ADR-0005 A7).** Negatives are streamed at a **10/11**
share (realized 0.9083 over the run): mined genomic windows plus embedded
dinucleotide-shuffled and structured-RNA decoys, drawn uniformly from a
19,409-item pool. A decoy whose own parent record falls outside `nested_train`
is refused. Over 10 epochs the production run consumed 10,472 positive and
103,688 negative draws.

**Curriculum.** Focal cross-entropy (γ = 0.5) with inverse-frequency class
weighting (α = 0.5), random window-phase **offset augmentation**, balanced
phylum sampling, and rare-class oversampling with class-II augmentation
constrained to training-fold parents.

**Optimization.** lr 3e-4, weight decay 0.01, grad-clip 1.0, batch 8 per GPU,
10 epochs, **fp32** (TF32 and cuDNN autotune off, ADR-0002 A7), gradient
checkpointing on, 8× NVIDIA RTX A4000 (16 GB, `sm_86`) under DDP, torch
2.7.1+cu128. Seed 42, `PYTHONHASHSEED=0`. The hyper-parameters are the P2-06
Hydra `--multirun` sweep winner (`g0.5_lr3e-4_a0.5`), promoted on the validation
ladder — never on test.

**Known composition bias.** The corpus is ~90 % Firmicutes. The GATE-4 graded
population is **98.7 % Firmicutes** (1,185 / 1,201 records) — this is the
composition that *remains* after the leave-one-order-out holdout is removed, not
a sampling decision.

---

## 5. Evaluation

### 5.1 GATE-4 — PASS (the Phase-2 exit gate)

Graded on the **evaluation twin** (§2), on the in-distribution `gate4_eval`
population: 1,201 records / 1,029 clusters / 15 orders / 2,976,259 nucleotides /
5,102 reconciled windows. Predictions pass through the deployed
overlapping-window reconciliation operator, so boundary metrics are not artefacts
of the 512-nt tiling grid.

**Gated statistic** — the **minimum**, over the three core elements, of each
element's per-nucleotide class F1 (a homogeneous unit; deliberately a minimum,
not a mean, so a strong element cannot mask a weak one):

| | value |
|---|---|
| **min per-element per-nt F1** | **0.951776** |
| pre-registered floor (ADR-0004 D6) | 0.80 |
| cluster-blocked bootstrap 95 % CI | **[0.9426, 0.9605]** (1,029 blocks, 2,000 resamples) |
| verdict | **PASS** — cleared by the CI *lower* bound |

Per element: **Stem I 0.9749**, **Antiterminator 0.9697**, **Specifier 0.9518**
(the minimum, and therefore the statistic).

### 5.2 Reported, explicitly **non-gated**

| Read | Value |
|---|---|
| micro / macro per-nt F1 | 0.9929 / 0.9550 |
| non-core per-nt F1 | Discriminator 0.9739, Stem II 0.9543, Terminator 0.9388, **Stem III 0.8800** (weakest), background 0.9965 |
| boundary IoU | Stem I 0.9509, Discriminator 0.9492, Antiterminator 0.9412, Stem II 0.9127, Specifier 0.9080, Terminator 0.8846, **Stem III 0.7857** |
| Specifier exact-3-nt codon | **89.79 %** exact (1,046 / 1,165 scorable); 98.71 % overlapping truth; 91.85 % right-length |
| **leave-one-order-out**, macro over 30 held-out orders (21.0 M positions) | **Antiterminator 0.7646, Stem I 0.7214, Specifier 0.7154**; non-core Discriminator 0.8601, Stem II 0.7073, Stem III 0.6981, Terminator 0.6960. Per-order minima approach zero. |
| 9-PDB cross-source label-noise ceiling | **not estimable** |

**The across-order number is the one that matters for discovery.** GATE-4 grades
in-distribution segmentation quality — an explicitly labelled *reference*, not a
generalization test. The gap between **~0.95 in distribution and ~0.72 across
orders** is the quantity that bears on novel-clade discovery, and it is reported
here beside the passing gate rather than after it. That axis is graded at
GATE-1 (Phase 4); nothing in this card licenses a generalization claim.

**The label-noise ceiling is closed, not favourable.** ADR-0004 D6 permits
recalibrating the 0.80 floor as a documented function of a cross-source ceiling
*C* estimated from the 9 crystal depositions. **C is not estimable**:
Antiterminator and Specifier resolve to a residue extent in **0 of 9**
depositions (Stem I in 5, Stem II in 1), so no cross-source agreement over the
*gated* elements exists. The recalibration path is therefore **closed** and the
floor stands as pre-registered — an N ≤ 9 ceiling must not be allowed to lower
the bar in one direction only.

---

## 6. Calibration — machinery only, nothing shipped

Phase-2 calibration is a **machinery demonstration**, not a gate result.

- Single shared scalar temperature fitted by per-window multi-class NLL:
  **T = 0.989587** (converged, 24 iterations, 1,778,688 positions); NLL/position
  0.017839 → 0.017836.
- Fitted on one seeded whole-cluster half of the P2-06a `selection_val` rung and
  read out-of-sample on the other half (ADR-0005 A11). That rung is **not**
  ADR-0005 D11's "disjoint calibration split", which does not exist at Phase 2 —
  the `calib` column is carved from the training folds at P3-02.
- **No temperature is shipped or consumed by any gate.** All Phase-2 metrics,
  including GATE-4, are computed on the **uncalibrated** posterior. GATE-2's
  temperature is re-fitted at Phase 3 on the `calib` carve.
- The stack stops at the *named* posterior (temperature-scaled, **pre**
  prior-shift). The Saerens/Elkan deployment prior-shift is Phase 5.

**GATE-2 (in-distribution ECE ≤ 0.05) is ungraded at the time of writing.**

---

## 7. Limitations

1. **Uncalibrated outputs.** Logits are not probabilities. Any thresholding is
   the caller's responsibility until GATE-2 is graded.
2. **Weak across-order generalization, unquantified beyond the LOO read.**
   ~0.72 macro per-element F1 across 30 held-out orders, with per-order minima
   near zero. The recall-vs-phylogenetic-distance curve is Phase 4.
3. **Firmicutes-dominated training and evaluation.** ~90 % of the corpus and
   98.7 % of the GATE-4 graded population.
4. **The gate grades a twin, not the shipped model** (§2). Conservative in
   direction, but it is a proxy.
5. **Non-core elements are weaker and ungated.** Stem III is the floor
   (F1 0.880, IoU 0.786); Stem III and Discriminator additionally carry a
   label-noise caveat from sparse annotation.
6. **Human-reference pretraining.** The Caduceus backbone is pretrained on the
   human reference genome while the target domain is entirely
   prokaryotic/archaeal. The Phase-1 transfer go/no-go passed and the fallback
   ladder (frozen-embedding probe, GTDB continued-pretraining, NT-multispecies)
   was built but never triggered.
7. **Class-II signal is scarce** and is carried by oversampling plus
   `TBDB001.cm`-derived structure; the anti-mimicry question is answered by the
   separate naive ablation (§2), not by the production checkpoint.
8. **Hard-negative mining did not execute.** The Phase-2 mining loop is
   machinery-complete and measured but was **deliberately not run** (§8), so the
   shipped checkpoint carries no mined hard negatives beyond the Phase-0/§9.1
   seed pool.
9. **Stage 1 is half a system.** Every precision-facing property of
   `tbox-finder` — calibrated confidence, genome-scale FDR, orthogonal
   confirmation — lives downstream and does not exist yet.

---

## 8. What Phase 2 deliberately did **not** do

The §9.1 **hard-negative-mining loop** is fully implemented, certified, and
budgeted, and its round-0 measurement ran (N₀ = 941 false positives over 660
genomes; scan 23.87 windows/s/GPU; per-candidate covariation producer measured
at 9.01 s median). It was **not executed as a training round**, by a recorded
Phase-2 decision, because the spare rule that decides whether a false positive
may be mined requires **three** independent evidence disjuncts
(relaxed-architecture OR any-helix R-scape covariation OR downstream-aaRS
synteny) and only **one** — covariation — has a backend. The other two return
`unavailable`, and the rule **fails closed**: a candidate is minable only if all
three ran and failed. With 1 of 3 backends the yield is therefore structurally
**zero** for every candidate, and a mining round would be a verified no-op
rather than a source of hard negatives. The machinery is **deferred, not
cancelled** — the budget is pinned (ADR-0005 A10 Phase 2) and exercisable the
moment the remaining two backends exist.

This is disclosed rather than checked off: the Phase-2 exit checklist item
"hard-negative-mining loop executed" is **not** met, by decision.

---

## 9. Reproducibility

- **Code:** <https://github.com/bioedca/tbox-finder> (public from day 1).
- **Environment:** `envs/ml-dna.yml` + `envs/ml-dna.conda-lock.yml`
  (lock hash `70b66801…3bb5003`), plus a CI-built CUDA/PyTorch Docker image as
  the archival artefact.
- **Artefacts:** all three checkpoints and their `provenance.json` are
  DVC-tracked; training reports are committed under `reports/p2/`
  (`train_stage1_production.json`, `train_stage1_classII_naive.json`,
  `train_stage1_gate4_twin.json`, `gate4.json`, `ablation_table.json`,
  `calibration.json`, `sweep_selection.json`).
- **Decisions:** every threshold and method choice is pinned in
  `docs/decisions/ADR-0001…0006`; the per-step record is
  `docs/dev-log/phase2_2026-07-31.pdf`.
- **Determinism:** seed 42 and `PYTHONHASHSEED=0` throughout; the Phase-2
  ablation replicates reproduce to a spread of 0.0001–0.0006 across re-runs at
  the pinned seed. Note that this measures run-to-run determinism, **not** seed
  variance — every replicate holds seed 42.

## 10. Citation

Not yet citable. A Zenodo DOI and preprint are minted at the Phase-7 release
gate; this card will be replaced by the release card at that point.
