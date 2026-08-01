# ADR-0004 — Split & leakage policy

- **Status:** Accepted (user sign-off 2026-07-10; **Amendment A1** — D2 coverage denominator — user sign-off 2026-07-10, P0-22; **A3** synthetic class-II parents 2026-07-19; **A4** held-out-order negative-window rule (a1+b2) 2026-07-23; **A5** whole-genome host-order negative-admission rule (a2) 2026-07-26; **A7** the D11 disjoint calibration carve (P3-02) 2026-08-01; CLAUDE.md §7 item 2)
- **Date:** 2026-07-10
- **Deciders:** bioedca (project owner)
- **Phase:** P0 (seed ADR)
- **Delegated from:** PRD §8 (label-derivation single-label precedence), §9.2 (splits & leakage control), §2.3 + §12 (GATE-4 segmentation-quality gate)
- **Supersedes / superseded by:** none
- **Related:** ADR-0001 (architecture & aims — the non-circularity principle: a leaky split would let a homology artefact masquerade as generalization, D2/D3), ADR-0002 (environment & ML stack — the two Stage-1 checkpoints, production + class-II-CM-naive, whose shared training fold D5 constructs), ADR-0003 (cluster & scan ops — the GATE-4 readiness a compute-forced sweep-prune must not compromise, D7), ADR-0005 (non-circular eval — the GATE-1 arms and the pinned decoy prevalence / per-bin min-N these splits feed; pending P0-25), ADR-0006 (validation decision rule & tiering — the §13.3 de-novo covariation whose anti-CM-circularity is the discovery-side counterpart of D2's clustering-via-CM being split-only; pending P0-29)

This ADR pins the **split construction and leakage-control contract** on which the project's headline generalization claim (GATE-1) rests, plus the **single-label precedence rule** for the per-nucleotide segmentation target and the **GATE-4 segmentation-quality gate**. It owns four load-bearing guarantees: (1) the per-base label is **unambiguous** — a complete, measured all-pairs element-overlap precedence table resolves every overlap; (2) positives are partitioned so that **no homology cluster and no held-out taxon straddles the train/test boundary** (PRD §9.2; CLAUDE.md §8.2); (3) **one** production Stage-1 checkpoint is scored on **every** leave-clade-out and literature-anchor holdout without a scheme-A training fold leaking a scheme-B held-out clade; and (4) the **committed split-assignment table** the CI no-leakage test re-checks on every PR carries **variant→parent→fold provenance** so oversampling/augmentation on the anti-mimicry path cannot smuggle a held-out parent into training.

Two threshold values are pinned here under the PRD §2.3 precedence carve-out: the **structure-aware homology-clustering distance/coverage cut** (D2) and the **GATE-4 per-element per-nucleotide-class F1 floor** (D6, default ≥ 0.80). Both are **blinded-frozen at P0** with a documented magnitude rationale (§2.3): they may not change after P4 unblinding, and any pre-P4 recalibration still requires ADR-0004 re-sign-off (CLAUDE.md §7 item 2).

---

## Context

The project's headline methodological result is a **generalization claim** — higher recall than `cmsearch` at matched precision on phylogenetically held-out clades (GATE-1, PRD §2.3). That claim is only as trustworthy as the split that produces it: **homology leakage across the train/test boundary inflates exactly the number the paper reports** (PRD §9.2; CLAUDE.md §8.2). Three properties of the corpus make leakage control non-trivial and force the decisions below:

- **The positives are one homologous family, sequence-divergent but structurally conserved.** T-boxes are all RF00230 / class-II homologs; raw sequence-identity clustering under-clusters divergent-but-structurally-close homologs (two loci at ~55% raw identity but near-identical secondary structure would be split across folds), so the clustering distance must be **structure-aware** (PRD §9.2 [DOI:10.1038/s41576-021-00434-9]).
- **The catalogue is ~90% Firmicutes across 29 phyla** (P0-12: Firmicutes 90.01%), so the only statistically well-powered generalization holdout is **leave-one-order-out** (~31 orders with ≥20 positives; PRD §12), with class- and phylum-level holdout reported as corroborating stress tests. The split must support a **nested ladder** — random (genus-stratified), leave-clade-out, and the independent literature anchor — reported together.
- **~4% of records lack a clade label** (P0-15: 928/3.94% no order, 841/3.57% no class, 453/1.92% no phylum), which the leave-one-order-out split and the no-leakage test **cannot silently absorb**.

The per-nucleotide segmentation target (PRD §8) has its own leakage-adjacent hazard: T-box structural elements **overlap almost universally** (the specifier codon sits inside the Stem-I loop; the antiterminator and terminator are mutually exclusive conformations of the same RNA; the discriminator reads the tRNA NCCA end inside the antiterminator bulge), so a single-label softmax target needs an **explicit, biology-motivated precedence rule** with each overlap's prevalence **measured**, not asserted (CLAUDE.md §10.3; PRD §8).

The measured overlap prevalences below are produced by `scripts/measure_overlap_prevalence.py` over `data/processed/master_clean_v0.parquet` (P0-12; 23,535 records; DVC md5 `8356cf24`, file sha256 `1eb76591…b062dd2`) and recorded in `data/processed/audits/overlap_prevalence_report.json`, so the pinned table is reproducible.

---

## Decision

### D1. Single-label precedence rule + the complete measured all-pairs overlap table (locked)

The per-base target is a **single-label** 8-class vector (`background`, `Stem_I`, `Specifier`, `Stem_II`, `Stem_III`, `Antiterminator_Tbox_seq`, `Terminator`, `Discriminator`; PRD §8). Where two element extents overlap, the base is assigned by a **fixed total precedence order** (most-specific / on-conformation wins):

> **Discriminator ▸ Specifier ▸ Antiterminator_Tbox_seq ▸ Terminator ▸ Stem_II ▸ Stem_III ▸ Stem_I ▸ background**

A base takes the class of the **highest-precedence element whose extent covers it**; `background` only where no element covers it. This total order reproduces all three biology-motivated rules (below) and gives every residual overlap a defined, harmless resolution.

**The complete all-pairs overlap table** (all 21 element pairs; over records where *both* elements are annotated; inclusive-nt overlap; measured on the 23,535-record corpus). **Three overlaps are material; the other 18 pairs are effectively disjoint (≤ 0.13%).**

| Element A | Element B | records w/ both | overlap prev. | median bp | containment | precedence resolution |
|---|---|--:|--:|--:|---|---|
| **Stem_I** | **Specifier** | 23,122 | **100.00%** | 3 | Specifier ⊂ Stem_I in **100.0%** | Specifier carves out of Stem_I |
| **Antiterminator** | **Terminator** | 23,439 | **99.54%** | 11 | partial (Term⊂AT 0.26%, AT⊂Term 0.01%) | overlap → Antiterminator (on-conformation convention) |
| **Antiterminator** | **Discriminator** | 23,208 | **100.00%** | 4 | Discriminator ⊂ Antiterminator in **100.0%** | Discriminator carves out of Antiterminator |
| Stem_I | Antiterminator | 23,475 | 0.08% | — | — | (disjoint; residual → total order) |
| Specifier | Antiterminator | 23,142 | 0.13% | — | — | (disjoint; residual → total order) |
| Stem_III | Antiterminator | 23,384 | 0.04% | — | — | (disjoint) |
| Stem_III | Discriminator | 23,127 | 0.03% | — | — | (disjoint) |
| Terminator | Discriminator | 23,173 | 0.93% | 2 | — | Discriminator wins (total order) |
| Stem_I ∩ {Stem_II, Stem_III, Terminator, Discriminator} | — | ~21.7–23.4k | ≤ 0.00% | — | — | (disjoint) |
| Specifier ∩ {Stem_II, Stem_III, Terminator, Discriminator} | — | ~21.5–23.1k | ≤ 0.00% | — | — | (disjoint) |
| Stem_II ∩ {Stem_III, Antiterminator, Terminator, Discriminator} | — | ~21.7k | 0.00% | — | — | (disjoint) |

The three material rules, with biology:

1. **Specifier carves out of `Stem_I`** (more-specific wins). The specifier codon is nested in the Stem-I loop in **100.0% of co-present records** (23,122/23,122; median 3 bp — the annotated codon). This is the PRD's "**Specifier nested in Stem I in 98.3%**" figure read at corpus scale: 98.3% is the specifier **presence** rate (23,142/23,535); *conditional on both elements being annotated*, containment is total. [Specifier–anticodon coupling in the Stem-I loop: PMID:32882008; DOI:10.1093/nar/gkaa721.]
2. **Antiterminator∩Terminator → `Antiterminator_Tbox_seq`** by a fixed **on-conformation** convention. The antiterminator and terminator are **mutually exclusive conformations of the same RNA** (the dot-bracket encodes only one), so the base is assigned the antiterminator (the T-box-defining "read-through" conformation). Measured overlap **99.54%** (23,332/23,439), median **11 bp**. [PMID:25583497.]
3. **Discriminator∩`Antiterminator_Tbox_seq` → `Discriminator`** (more-specific wins). The discriminator base(s) read the tRNA NCCA acceptor end inside the antiterminator bulge; containment is **100.0%** (23,208/23,208), median 4 bp. [PMID:25583497; PMID:32882008.]

**Pseudoknot** (IIA/B) is **folded into `Stem_II` by class definition, not by precedence** — no standalone pseudoknot class is trained (its crossing pairs cannot be encoded in the nested dot-bracket that sources the labels, and no genome-scale per-record annotation exists; PRD §8). It is retained only as a PDB-fixture structural diagnostic.

The `Specifier` extent is the **annotated specifier codon** (`codon_start`/`codon_end`); P0-20/P0-21 may widen it to the specifier loop against the 9 crystal fixtures, which does **not** change the more-specific-wins rule. This precedence table is **encoded in the crystal-structure fixtures and unit-tested** in `tests/unit/test_label_derivation.py` (P0-20/P0-21), and its resolution is applied identically in both the production and the class-II-CM-naive label runs (the naive run additionally withholds `TBDB001.cm`-derived structure; PRD §8).

### D2. Structure-aware homology-clustered assignment — distance, method, threshold (locked; built in P0-22)

Positives are clustered on a **structure-aware distance**, and **whole clusters are assigned to a single fold — never split across train/val/test** [DOI:10.1038/s41576-021-00434-9]. Pinned specification:

- **Distance (structure-aware, not raw identity).** Align all positives to the class covariance models — `RF00230.cm` (class I) and `TBDB001.cm` (class II) via `cmalign` (Infernal ≥ 1.1.4; the P0-11a staged CMs) — and compute pairwise identity **over consensus (match-state) columns only**, so indels in variable loops do not dominate and structurally-homologous low-raw-identity pairs are recognised. **Distance `d = 1 − consensus-column identity`.** Using the CM alignment purely as a *distance metric for partitioning* is **not** a discovery circularity: clustering only decides which fold a sequence lands in; the anti-CM-circularity requirement lives at the §13.3 covariation step (de-novo MSA, ADR-0006), not here.
- **Clustering method.** **Single-linkage** agglomerative (transitive closure of the "within-threshold" relation), the conservative choice for leakage: any chain of near-neighbours collapses into one cluster, so no two within-threshold sequences can straddle the split. Applied within each of {class I, class II} and the union taken (a cluster is never merged across classes only if below threshold).
- **Threshold (the pinned cut).** Two positives join the same cluster when **`d ≤ 0.30` (consensus-column identity ≥ 0.70) AND alignment coverage ≥ 0.70 of the RF00230 model consensus span (`clen`)** — i.e. the number of co-occupied consensus columns is ≥ 0.70·`clen` (**Amendment A1**; originally worded "of the shorter member's consensus span", which was defective — see below). A sequence with sub-threshold CM coverage (e.g. a divergent/Tier-2N-like locus that aligns poorly to RF00230) forms its **own singleton cluster** — the safe behaviour (it cannot straddle by construction), and now the *realised* behaviour: a sequence covering < 0.70 of the model can form no edge (co-occupied ≤ its span < 0.70·`clen`).
- **ncRNA-low-identity rationale.** The 0.70 identity cut is deliberately **below** the ~0.80–0.90 protein redundancy-removal regime because (i) it is **structure-aware** (consensus-column) identity, which runs higher than raw identity for a given structural similarity, and (ii) structurally-conserved ncRNA homologs retain structure well below the raw identity at which a sequence-only cut would separate them, so a lower cut **over-merges**, the safe direction for leakage control [DOI:10.1038/s41576-021-00434-9]. The number is the *default*; its **adequacy** (not just construction-consistency) is enforced by the safety net below, since the no-leakage CI cannot police cut-tightness.
- **Adequacy safety net (pinned; reported by P0-22/P0-23).** (a) An **all-vs-all train↔test structure-distance histogram** — no train↔test positive pair may be closer than the cut (a visible separation gap). (b) A **tighter-cutoff re-cluster sensitivity sweep** over identity ∈ {0.60, 0.70, 0.80, 0.90}: the **leave-one-order-out headline metric must be stable** across the sweep; a material move is a **CLAUDE.md §7 stop-and-ask**, not a silent choice.

Any change to the `d ≤ 0.30` / coverage ≥ 0.70 cut requires ADR-0004 re-sign-off (§2.3 carve-out; blinded-frozen at P0).

### D3. Cluster–clade crossing forced rule + phylogenetic-independence diagnostic (locked)

Under any **leave-clade-out** scheme, a homology cluster can contain members from both the held-out clade and training clades. The rule (PRD §9.2):

- **Any cluster containing *any* held-out-clade member is assigned *in full* to the held-out fold.** Its training-clade members are then **excluded from training AND not scored as held-out positives** (they are neither train nor test) — so a near-homolog of a held-out sequence can never sit in training under a different taxon label, and the held-out recall is not inflated by a within-cluster training twin.
- **Phylogenetic-independence diagnostic (reported per scheme, P0-22/P0-23):** the **count + taxonomic spread of these clade-crossing clusters/records**. A large clade-crossing count means the held-out clade is not phylogenetically clean; the diagnostic makes that visible rather than silent.

### D4. Taxonomy-incomplete handling (locked; implemented P0-15)

Records lacking a clade label at the holdout rank are handled by a pre-registered, fail-closed policy (already realised by P0-15 `src/tbox_finder/taxonomy.py::replace_lineage`):

- **Re-derive each full lineage from its TaxId** against the frozen governing-release taxonomy (the pinned NCBI taxdump, P0-15 — *not* GTDB, since the corpus lineage columns are NCBI-named at a pre-2021 vintage and GTDB cannot resolve the environmental/metagenome/CPR TaxIds that dominate the residue); recovery rate reported per rank (P0-15: phylum +7.06%, class +8.44%, order +6.36%).
- **Still-incomplete residue is pre-registered as dropped from clade-holdout** (`dropped_from_clade_holdout`), kept **only** in the random (genus-stratified) split, reported per rank. The invariant `resolved_order` NULL ⟺ `dropped_from_clade_holdout` holds, so **a no-clade record can never silently enter a clade fold**.
- This is the D7 no-leakage test's **defined behaviour on records lacking a clade label**: such a record must be in the dropped/random-only bucket or the test **fails** (it cannot silently pass at the holdout rank).

### D5. Nested most-restrictive training-fold construction — one checkpoint, every holdout (locked; operationalised P0-22)

The production Stage-1 checkpoint is trained **once** on a **single most-restrictive nested training fold** whose complement is the **union** of every scheme-B/(c) holdout:

- The training fold **simultaneously excludes** (i) the leave-one-order-out held-out orders, (ii) the Actinobacteria phylum-holdout, and (iii) the literature-anchor clusters (the P0-16 Vitreschak-2008 non-Firmicutes set + the P0-17 additional class-II positives + the 18-record Actinobacteria/ILE/class-II set, arm (c)).
- Therefore **one checkpoint** is scored on every §9.2 scheme-B and scheme-(c) holdout **without** a scheme-A (random-split) training fold leaking a scheme-B held-out clade into training. Scheme-A (random, genus-stratified) is the detection-quality **reference** only; it is never the source of a generalization number.
- The **class-II-CM-naive ablation** (ADR-0002; graded only by the GATE-1 class-II anti-mimicry sub-arm, scored Stage-1-only) is trained on the **same nested training fold** with the same fold assignments, differing only in label source (withholds `TBDB001.cm`); so the two checkpoints share one leakage-controlled partition.

This nested construction is the single-checkpoint consequence of the split ladder; P0-22 builds the fold table that realises it, and D7's committed table records each record's fold-per-scheme.

### D6. GATE-4 — segmentation-quality gate (locked default; recalibration governed)

- **Gated quantity:** the **minimum per-element per-nucleotide-class F1 over the three core elements {Stem I, Specifier, Antiterminator}**, on the **in-distribution homology-clustered (genus-stratified) split** — an explicitly-labeled **segmentation-quality reference, not a generalization test** (PRD §2.3/§12). Per-nucleotide class F1 is a **homogeneous, commensurable unit → there is no cross-unit mean**; the gate is the **min over the three**, not an average.
- **Floor:** **≥ 0.80 (recalibratable binding default).** **Magnitude rationale (authored at P0, before any P4 result, per §2.3):** 0.80 per-nucleotide F1 on the three core elements is the "the segmenter genuinely localizes the defining T-box elements with boundary fidelity" bar — high enough that a model clearing it has learned element extents (not merely background vs foreground), and high enough to support the downstream §13.1 locus-construction and the §13.3(d) sequence-read specifier that the discovery pipeline depends on; yet low enough to tolerate the ~1–2 nt boundary ambiguity intrinsic to projecting TBDB dot-bracket annotations onto individual nucleotides. It is a **reference** gate on the in-distribution split, so 0.80 is achievable for a well-trained segmenter on the three *core* elements while the sparse/label-noisy classes are excluded (below).
- **Excluded from the gate (reported per-class only):** Stem II (S-turn, incl. the folded IIA/B-pk extent), Stem III, class-I-restricted Terminator, and Discriminator — the label-noise caveat for sparse Stem III / Discriminator. Boundary IoU is reported per element.
- **Reported (non-gated) sanity checks:** **Specifier exact-3-nt-codon detection** and the **9-PDB cross-source label-noise ceiling** (P0-21). The N ≤ 9 crystal ceiling has **no CI** and is a *different* label source, so it **must not one-directionally lower** the 0.80 bar.
- **δ / recalibration governance.** The 0.80 default may be recalibrated **only** if the P0-21 9-PDB label-noise ceiling `C` (reported non-gated) demonstrates the cross-source annotation itself caps achievable per-nt F1 below the floor; in that case the floor may be reset to a **documented function of `C`** (e.g. `min(0.80, C − δ)` with δ stated), **by ADR-0004 re-sign-off**. Absent that, the floor stays 0.80. **Blinded-frozen at P0:** the floor may not change after P4 unblinding, and any pre-P4 recalibration still needs ADR-0004 sign-off (§2.3; CLAUDE.md §7 item 2). GATE-4 is graded at **P2** (ADR-0002 production checkpoint); a capacity-driven GATE-4 failure first attempts the ADR-0002 backbone fallbacks (a method-gate failure, PRD §2.3 branch 3), never an auto-route to the GATE-1-failure deliverable.

### D7. Committed split-table no-leakage CI, incl. variant→parent→fold provenance (locked; test built P0-24)

A compact **per-record split-assignment table** (`record_id`, `cluster_id`, lineage-by-rank, fold-per-scheme, and for each augmented/synthetic variant its `parent_record_id`; **no sequences**; hash-linked to `master_clean_v0.parquet`) is committed to git/LFS so `tests/ml/test_no_leakage.py` re-checks the **real ~23,535-record partition** (not a smoke fixture) on **every PR** (CLAUDE.md §8.2; a bounded, deliberate carve-out from the no-data-in-repo rule). The test asserts, **for every scheme in §9.2**:

- **Holdout-unit separation + cluster non-splitting** — no cluster is split across folds, and the scheme's holdout unit (the cluster, and the taxon at the scheme's holdout rank) does not straddle the train/test boundary. This is **not** non-spanning at *every* taxonomic rank (which would contradict leave-one-order-out, where lower ranks *within* a held-out order are intentionally held out together).
- **Variant→parent→fold provenance** — every augmented/synthetic-class-II variant **inherits its parent record's fold** (all class-II augmentation is constrained to training-fold parents; PRD §8/§11), so oversampling/augmentation on the headline anti-mimicry path cannot leak a held-out parent into training.
- **Defined behaviour on no-clade records** (D4) — a record with no assignment at the holdout rank **cannot silently pass**; it must be in the dropped/random-only bucket or the test fails.
- The **18-record Actinobacteria/ILE/class-II set**, the **P0-17 additional independent class-II positives**, and the **P0-16 independent literature anchor** (arm c) are all included so none leaks into training.

The test is **CI-blocking**; per CLAUDE.md §8.5 (broadened) **any data/label/cluster-pipeline change** — not only split-logic edits — re-runs the full-corpus check. **DOME reporting fields** [DOI:10.1093/gigascience/giae094] (train/test redundancy + partition strategy) are declared in the eval report.

---

## Consequences

- **P0-20/P0-21 (label derivation + fixtures):** implement the D1 total precedence order + the measured table; unit-test against the 9 PDB extents + hand-checked fixture + the naive-run-withholds-`TBDB001.cm` assertion.
- **P0-22/P0-23 (split construction + table):** implement D2 clustering + D3 clade-crossing + D5 nested fold; emit the D7 committed split-assignment table + the D2 adequacy histogram + sensitivity sweep + the D3 independence diagnostic.
- **P0-24 (no-leakage test):** implement D7 over the committed table.
- **P2 (training/eval):** GATE-4 graded per D6; the single production checkpoint + the naive ablation trained on the D5 nested fold.
- **Reproducibility:** the D1 table is regenerated by `scripts/measure_overlap_prevalence.py`; re-run it on any change to the P0-12 ingest and diff `overlap_prevalence_report.json`.

## Cross-reference impact list

<!-- Heading normalized 2026-07-12 from "Related documents" to match the five
     sibling ADRs (content unchanged); the Phase-0-exit bullet was appended. -->

- **PRD:** §8 (label derivation & precedence — D1), §9.1 (masking denominator the splits share), §9.2 (splits & leakage — D2/D3/D4/D5/D7), §2.3 + §12 (GATE-4 — D6), §7.1 (the arm-(c) anchor + class-II sets D5/D7 hold out), §11 (training-fold-constrained augmentation — D7).
- **ADRs:** ADR-0001 (non-circularity), ADR-0002 (the two Stage-1 checkpoints D5 trains), ADR-0003 (D7 GATE-4-readiness reference), ADR-0005 (GATE-1 arms / decoy prevalence / min-N — pending P0-25), ADR-0006 (de-novo covariation anti-circularity — pending P0-29).
- **CLAUDE.md:** §2.3 precedence carve-out (the two pinned defaults), §8.2 (CI no-leakage), §8.5 (broadened re-run trigger), §7 items 2/4 (ADR sign-off; gate stop-and-ask), §10.3 (measured-not-fabricated prevalence).
- **Cards / paper (release-bound):** dataset card (partition strategy + train/test redundancy + clade-crossing counts), `paper/manuscript.qmd` (the split-ladder methods paragraph + GATE-4 result).
- **Phase-0 exit (2026-07-12):** the D2/D3/D5 leave-clade-out split design + the D7 CI-blocking no-leakage test are the subject of the `paper/manuscript.qmd` §Non-circular evaluation design paragraph + Figure 1 caption (held-out↔train consensus identity **< 0.70**, observed max 0.699) and the `README.md` P0 headline; no decision or value changed.
- **Phase-2 exit (2026-07-31):** **D6 is graded and PASSES** — min per-element per-nt-class F1 over {Stem I, Specifier, Antiterminator} = **0.951776** vs the **0.80** default floor, cluster-blocked bootstrap 95 % CI **[0.9426, 0.9605]** (1,029 blocks / 2,000 resamples), cleared by the lower bound; per element 0.9749 / 0.9518 / 0.9697, the Specifier minimum being the statistic (`reports/p2/gate4.json`). **D6's δ-recalibration path is CLOSED, not exercised:** the cross-source ceiling *C* is **not estimable** — Antiterminator and Specifier resolve to a residue extent in **0 of the 9** depositions — so the floor stands exactly as pre-registered and no δ was applied in either direction. **A6's graded population and eval twin are the realized form of D5's two-checkpoint design:** the shipped full-fold production scanner has no in-distribution holdout, so the twin (`exclude_gate4_eval=true`, 8,303 → 7,099 records, 1,031 clusters withheld) was graded in its place, with 0 shared clusters and 0 shared records against a positive-controlled intersection test — making 0.952 a **conservative proxy** for the shipped model. **D7 stayed green in CI** across every Phase-2 data/label touch. Both D5 checkpoints exist and are DVC-tracked (production + the class-II-CM-naive ablation, the latter withholding `TBDB001.cm` over 1,200 records / 193,587 element-nt and scored Stage-1-only at GATE-1/P4). Subject of `paper/manuscript.qmd` §sec-gate4 + the new **Figure 2** caption, the `README.md` Phase-2 headline, and `docs/model_card.md` §2/§5 (the two-run split + per-checkpoint GATE attribution PRD §10.1 requires). No decision or value changed.

## Amendments

### A1 — D2 coverage denominator: the RF00230 **model** consensus, not the shorter member's span (P0-22; user sign-off 2026-07-10)

**Trigger.** The first real split build (P0-22) exposed an internal inconsistency in D2. The pinned coverage clause read "coverage ≥ 0.70 **of the shorter member's consensus span**", but the same bullet's rationale requires that "a locus that aligns poorly to RF00230 forms its **own singleton cluster**." These conflict for short sequences: a locus aligning to only 1 of the 224 RF00230 consensus columns has a *shorter-member span* of 1, so it is trivially 100 %-covered and — over that 1 column, at whatever nucleotide it carries — links (identity 1.0, coverage 1.0) to **thousands** of other positives.

**Measured effect (the degeneracy).** Under the literal shorter-member reading, 13 such low-occupancy hub sequences (consensus span 1–3; node degree up to **8,642**) single-linkage-bridged the corpus into one cluster of **20,941 / 23,569 records (88.9 %) spanning 66 orders and 29 phyla**, at *every* sweep cut (0.60–0.90). The cluster–clade-crossing forced rule (D3) then excluded 13,279 records, leaving a nested training fold of **676 (2.9 %)** — unusable, and directly contradicting the sign-off rationale that 0.70 "avoids the multi-order mega-cluster risk." The distances themselves were sound (median pairwise consensus identity 0.46; a Bacillales↔Actinobacteria pair 0.48; only 0.1 % of pairs ≥ the cut); the pathology was purely the coverage denominator.

**Amendment.** Coverage is measured against the **RF00230 model consensus span (`clen`)**: two positives link only when their **co-occupied consensus columns ≥ 0.70·`clen`** (and identity ≥ 0.70). A sequence covering < 0.70 of the model can form **no** edge (co-occupied ≤ its span < 0.70·`clen`) → forced singleton, realising the pinned "aligns-poorly-to-RF00230 → singleton" behaviour. The pinned *number* (0.70) is unchanged; only the denominator (model `clen`, not shorter-member span) changes. **Measured result:** class-I largest cluster **20,941 → 1,238 (5.5 %)** (9,474 clusters); class-II largest 824 (67.7 %, a genuinely tight *ileS* family, mostly held-out Actinobacteria); histogram inside-cut = 0 preserved.

**Scope of re-validation.** `src/tbox_finder/splits.py` (coverage denominator + a forced-singleton unit test), `conf/data/splits.yaml`, and the D2 threshold bullet above. Single-linkage, the 0.70 identity/coverage numbers, and every other D-decision are **unchanged**. This is an ADR-0004 §2.3 re-sign-off of a delegated pinned value (CLAUDE.md §7 item 2), not a scope change.

### A2 — GATE-4 0.80 magnitude rationale co-authored + blinded-freeze confirmed (P0-28; user sign-off 2026-07-11)

- **Status:** **Accepted (user sign-off 2026-07-11; CLAUDE.md §7 item 2), "accept both as drafted".** Confirms (does not change) the D6 GATE-4 per-nt per-element F1 floor of **≥ 0.80** and records its SESOI + blinded-freeze alongside the ADR-0005 gate defaults, so the whole delegated-default set is documented in one P0-28 pass.

**No value changes.** The D6 floor stays **0.80**; its magnitude rationale was already authored in D6 at P0-19. This amendment (i) co-locates that 0.80 rationale in the P0-28 record (`analyses/gate_default_rationales.qmd`, rendered from `src/tbox_finder/power.py::magnitude_rationale('gate4_f1_floor')`; the code constant `GATE4_F1_FLOOR == 0.80` is asserted equal to this ADR value by `tests/unit/test_magnitude_rationale.py`), and (ii) restates the blinded-freeze.

**Rationale (SESOI).** 0.80 per-nt F1 on the 3 core elements is the smallest segmentation quality that **(i)** demonstrates learned element *extents* (not merely background-vs-foreground), and **(ii)** supports the downstream §13.1 locus construction + §13.3(d) sequence-read specifier the discovery pipeline depends on, while **(iii)** tolerating the ~1–2 nt boundary ambiguity intrinsic to projecting TBDB dot-bracket annotations onto individual nucleotides. It is a **reference** gate on the in-distribution split; the N ≤ 9 PDB cross-source label-noise ceiling `C` (P0-21, reported non-gated, no CI) **must not one-directionally lower** it (D6). This is a project-internal SESOI on a bespoke per-nt F1 (empirically anchored by the P0-21 ceiling), so it carries no external numeric cite.

**Blinded-freeze.** 0.80 is **blinded-frozen at P0** (authored before any P4 result): no post-P4 change; a pre-P4 recalibration needs ADR-0004 re-sign-off, and D6 already scopes the *only* permitted recalibration (a documented function of `C` if the P0-21 ceiling demonstrably caps achievable per-nt F1 below the floor).

## Sign-off

- **User sign-off:** ☑ recorded 2026-07-10 (bioedca), CLAUDE.md §7 item 2. The D2 structure-aware clustering cut was selected as **`d ≤ 0.30` (consensus-column identity ≥ 0.70) + coverage ≥ 0.70** (single-linkage) over the tighter (0.80/0.80) and looser (0.60/0.70) alternatives, on the rationale that it over-merges slightly (the safe leakage direction) while avoiding the multi-order mega-cluster risk a lower cut invites; the re-cluster sensitivity sweep {0.60–0.90} + train↔test distance histogram (D2 adequacy net) backstop the choice. ADR accepted as drafted.
- **Amendment A1 sign-off:** ☑ recorded 2026-07-10 (bioedca), CLAUDE.md §7 item 2 / §2.3 re-sign-off. Coverage denominator = RF00230 model consensus `clen` (not shorter-member span); pinned 0.70 unchanged. Selected over a separate min-occupancy eligibility gate (equivalent effect, two knobs) and over revisiting the linkage method (single-linkage retained). Chosen on the P0-22 measured evidence above.
- **Amendment A2 sign-off:** ☑ recorded 2026-07-11 (bioedca), CLAUDE.md §7 item 2. The GATE-4 0.80 floor is **unchanged**; A2 co-authors its SESOI into the P0-28 record and restates the blinded-freeze alongside the ADR-0005 defaults. "Accept both as drafted."
- **Amendment A6 sign-off:** ☑ recorded 2026-07-30 (bioedca), CLAUDE.md §7 items 1+2, via AskUserQuestion on the measured fork. Population = **the eval twin on scheme-A `test` inside `nested_train`** (1,201 records / 1,029 clusters), selected over (B) the twin-on-`selection_val` variant — same GPU cost but the fold the sweep selected on, and 97.6 % Firmicutes at 469 blocks — over (C) a no-retrain a-fortiori grade on the LOO holdout, which gates the generalization population D6 does not gate and risks mis-triggering the PRD §2.3 branch-3 backbone fallbacks on a false failure, and over (D) declaring GATE-4 ungradeable at P2. Ship = **the full-fold P2-10d′-b production checkpoint** (the twin is graded, never shipped). No pinned value changes; the 0.80 floor and its blinded-freeze are untouched.
- **Amendment A5 sign-off:** ☑ recorded 2026-07-26 (bioedca), CLAUDE.md §7 items 1+2. Bridge = GTDB R232 metadata `ncbi_taxid` (over a live esummary re-fetch — deterministic + reproducible + same pinned release); admissibility = blacklist-held-out (admit training + no-corpus-positive hosts; over the whitelist-training-only reading which discarded the 197-phylum breadth); loosening fallback OFF. No pinned value changes; A5 extends the negative-admission contract to the whole-genome substrate a1 could not reach.

### A3 — evaluation-only synthetic class-II variants may be parented on held-out records (P2-08; user sign-off 2026-07-19)

- **Status:** **Accepted (user sign-off 2026-07-19, "both approved"; CLAUDE.md §7 item 2).**

**Trigger.** P2-08 measured that D7's "all class-II augmentation is constrained to training-fold parents" makes the ADR-0005 D9 construction-powered recovery set unbuildable. Only **22** class-II records sit in `nested_train`, spanning **20 clusters / 4 orders**, and **zero** are Actinobacteria (all 1,160 Actinobacteria class-II corpus records are held out). Because D9 grades with a *block-resampled* floor (PRD §2.3: resampling at the homology-cluster / held-out-order level), emitting N variants from 22 parents raises the record count without raising the block count — "above min-N by construction" would have been satisfied on a quantity the gate does not resample. The D9 within-Actinobacteria-memorization control was likewise unbuildable, its contrast arm being empty.

**Amendment.** D7's training-fold-parent constraint is **scoped to *training* augmentation** (the PRD §11 oversampling path). An **evaluation-only** synthetic class-II recovery set may be parented on **held-out** class-II records. D7's rationale — augmentation must not leak a held-out parent into training — is preserved by a *stronger* mechanism: instead of making every variant training-safe, an eval-only variant is **never training-eligible at all**. Parents are restricted to `source == "corpus"`, so the **18-record blind set is excluded as a parent** and the natural arm stays independent of the synthetic one.

**Consequent contract repair (the conflict this exposed).** `tests/ml/test_no_leakage.py::external_positive_leaks` treated every non-`corpus` source as an independent external positive requiring `is_anchor_heldout` and `fold_random is None`. A D7-conforming variant inherits its parent's fold, and the eligible parents carry `fold_random` ∈ {train 1108, val 28, test 24} — so **every correct variant was a violation by construction**: the two pinned contracts were in direct conflict. The predicate conflated two categories and had no clause for a third (a *derived* row). Resolution:

- `splits.EXTERNAL_POSITIVE_SOURCES = ("anchor", "blind")` — the external-positive predicate is scoped to these.
- `splits.DERIVED_SOURCES = ("synthetic_classII",)` — derived rows carry their own, **strictly stronger** predicate `synthetic_variant_leaks`: real variant (`parent_record_id != record_id`); parent resolvable in-table and `source == "corpus"`, `klass == "II"`, not `nested_train`, `nested_role != "train"`; and the variant itself out of training. D7 fold inheritance across all six `FOLD_SCHEME_COLUMNS` still applies, enforced by `variant_parent_fold_mismatches`.
- A new `unknown_sources` predicate closes the `source` vocabulary, so re-scoping to named allowlists cannot silently exempt an unrecognised value from *every* check.
- `external_cluster_training_twins` is scoped the same way: a derived row shares its parent's cluster by construction, so counting it would re-flag the parent's own cluster.

**Structural, not conventional.** `splits.append_variant_rows` reads every fold, lineage, and cluster value **off the parent row**; the caller supplies only `variant_id` + `parent_record_id`. A caller cannot hand a variant a fold that differs from its parent's because a caller cannot hand it a fold at all.

**Measured result.** 2,344 variants / **1,172 parents / 103 clusters / 25 orders** — both block units clear `MIN_REAL_HOMOLOG_N = 20`. *(CodeRabbit r1 corrected this from a reported 26: `block_counts` was counting the one eligible parent with a null `resolved_order` as a distinct order block, inflating the graded quantity by one. Nulls are no longer blocks.)* Committed table 23,569 → **25,913** rows (23,535 corpus + 34 external + 2,344 synthetic); `n_clusters` unchanged at 9,603 (variants inherit parent clusters). **`nested_train` membership is unchanged** — variants are held-out-parented — so the 8,303 / 7,472 fold sizes and the P2-06 selection-val carve are untouched; that is precisely why this is safe where a training-fold-parented append would not have been.

**Scope of re-validation.** `src/tbox_finder/splits.py` (`append_variant_rows`, the `--variants` writer path, the `n_external` / `n_synthetic_variants` provenance split), `tests/ml/test_no_leakage.py` (predicate re-scope + adversarial fixtures + re-baselined counts), `tests/unit/test_split_table_schema.py`, the committed table + its provenance JSON. No other D-decision changes; D7's inheritance requirement is unchanged and now non-vacuous for the first time.

### A4 — the held-out-order **negative-window** rule: parent-record fold is the definition, made CI-visible (a1 + b2; P2-10c′-f; user sign-off 2026-07-23)

- **Status:** **Accepted (user sign-off 2026-07-23, "a1 + b2, loosening fallback OFF"; CLAUDE.md §7 item 1 — the §9.2 held-out-order-negative Q-question — and §7 item 2 — this ADR amendment).**

**Trigger.** P2-10c′ introduces a mined/background **negative** substrate (the `genomic_window` pool, and later an independent-genome pool) carved from the flank of corpus records. D3/D4/D5 pin the leave-one-order-out holdout for **positives**; they were silent on the mirror question for negatives — *may a background window drawn beside (or from a genome hosting) a held-out-order T-box enter training under a negative label?* The concern is directional and load-bearing: the immediate genomic neighbourhood of the very loci GATE-1/GATE-4 grade would then sit in training, inflating exactly the generalization number the paper reports (the D2/D3 rationale, "homology leakage across the train/test boundary inflates exactly the number the paper reports"). Measured on the committed 1024-nt `genomic_window` pool, **64.2 % (29,216/45,488)** of natural windows have a parent **out of the D5 nested training fold** — leave-one-order-out held out, D3 clade-crossing-excluded, or D4 dropped — and are refused by this rule [`data/processed/audits/mining_pool_report.json`, `parent_fold.n_natural_windows_out_of_fold`; the loader's own `reports/p2/negative_injection.json` records `n_refused_parent_out_of_fold = 27,960` on the masked/control-filtered stream]. (The narrower designated-LOO-holdout *subset* — a strict part of that out-of-fold set, since the fold complement also carries D3/D4 records — was 37.1 % on the earlier P2-10b pool.)

**The rule already exists and is fail-closed by default — this amendment pins its *definition* and makes it *CI-visible*, changing no threshold value.** What shipped at P2-10d′-a (user decision 2026-07-20): every window stamps `source_record_id` (the corpus record whose flank it was carved beside) and `parent_nested_train` (whether that parent is in the D5 nested-most-restrictive training fold — which *inherits* the LOO holdout, the D3 clade-crossing exclusion, and the D4 dropped bucket, so it is **at least as strict** as the positive side, not merely equivalent). `admit_pool_rows` / `load_negative_records` default `require_parent_nested_train=True` and refuse any window whose parent is out-of-fold (`parent_not_nested_train`) or unresolved (`parent_fold_unknown`), the pandas-3 NaN fail-open guarded by `is_missing` [`data/negatives.py`].

- **a1 — the definition.** Admissibility keys on the **parent corpus record's fold**. For the current flank-carved `genomic_window` substrate this **equals a host-genome→order exclusion by the carving geometry** — `carve_pool` emits windows only from catalogued-locus flanks, so every window's parent record *is* a catalogued locus and "host genome" and "parent locus" coincide; a genome hosting a held-out-order T-box therefore has a held-out-order parent. **This geometry dependency is documented, not assumed away.** It can diverge two ways: (i) a **future non-flank / whole-genome background pool** carves windows not anchored to a catalogued locus, breaking the coincidence; (ii) a taxonomy-source disagreement between the parent's `resolved_order` and a host-accession lookup. The explicit **host-accession→`resolved_order`** exclusion (**Option a2**, belt-and-braces) is **deferred to the point a whole-genome background pool is introduced** — no `host_accession→order` map exists in the repo today, so a2 is new provenance owed then, not now.
- **b2 — CI visibility.** The guarantee is a runtime property of the loader; the committed per-record split table the CI §8.2 gate reads has **no negative rows**, so a `require_parent_nested_train=False` caller (or a future consumer bypassing `admit_pool_rows`) would loosen it invisibly ([[ci-leakage-gate-blind-to-runtime-augmentation]]). A **fail-closed clause** is added to `tests/ml/test_no_leakage.py`: it asserts every shipped negative-loader entry point defaults `require_parent_nested_train=True`, drives a pool built over the **real committed partition**'s parents through the shipped stamp + admit path (including a delegating loader, not only `admit_pool_rows`), and asserts **by identity** — the admitted windows' parents are *exactly* the in-fold set (from **asymmetrically-sized** in-fold and held-out-order groups, so a fold-sense inversion cannot pass on a lucky symmetric count) — plus a **must-fire non-degeneracy companion** (the split→parent join resolves every parent, a designated LOO holdout order genuinely appears among the parents, and loosening the rule re-admits exactly the refused windows). A broken/empty join or a loosened default turns the gate **RED** instead of green-with-nothing-refused ([[namespace-mismatch-invisible-noop]]). Both the count-based companion *and* the identity assertion are necessary: a namespace-mismatched join stamps *every* parent out-of-fold and mimics total discrimination in the admission report alone (only the stamp's `n_unresolved_parent` distinguishes it), while a fold-sense inversion refuses the *wrong* windows with numerically identical counts (only the identity assertion distinguishes it).

**The loosening fallback (`require_parent_nested_train=False`) stays OFF.** Flipping it to reclaim the refused out-of-fold windows (≈ 64 % of the natural pool) is the one option that *weakens* leakage control. The **compositional-leak mechanism** that motivates keeping it off — that a held-out order's genomes carry order-specific GC/codon/k-mer signal a scanner could key on — is carried here as an **explicit, unquantified hypothesis** (§10.1: not asserted as a measured biological fact, no ≥2-source claim made); it is a reason to preserve the fail-closed default, not a magnitude. Were the fallback ever taken, it must ship with the pre-registered three-outcome disclosure control (a same-artifact power-precondition + chance-referenced verdict + coverage guard) drafted in the P2-10c′ masking source — not encoded here, because the fallback is not taken.

**Cross-reference impact — ADR-0005 D14 is named affected-by-consumption.** This is **not orthogonal to ADR-0005 D14**: `admit_pool_rows` applies "the §9.1 / D14 / §9.2 admission rules" to the same negative/mining pool, and P2-10e's mining loop — whose spare rule and Tier-2N halt/rollback **are** D14 — draws from that pool, so any change to which windows are order-eligible changes the pool D14 mines. Per the repo's amendment convention (ADR-0005 A7 lists ADR-0006 D12 in its cross-reference impact even while "unaffected"), **ADR-0005 D14 is named the negative/mining-pool-composition owner, affected-by-consumption**, while D14's spare rule and Tier-2N probe are **untouched**. The off-catalogue locus-mask question is the separately-signed ADR-0005 **A8** (P2-10c′-d), which is **locus-level only and explicitly does not address this host-order question**; A4 and A8 are complementary, not overlapping.

**No value changes; scope-clarifying.** No pinned threshold moves (D2's cut, D6's floor, the fold sizes are all unchanged); A4 pins the *definition* of an already-shipped, already-fail-closed negative-side rule and adds its CI witness. `nested_train` membership, the 8,303/7,472 fold sizes, and every positive-side D-decision are unchanged.

**Scope of re-validation.** `tests/ml/test_no_leakage.py` (the new `held_out_order_negative_clauses` gate + its unit bite tier + the committed-table b2 tier; the P0-24 predicates are untouched), and this ADR. PRD §9.2 carries a supplement note (PRD > ADR). No code in `data/negatives.py` / `mining/pool.py` changes — the rule they already enforce is what A4 documents and CI now witnesses.

### A6 — the GATE-4 **graded population** and the two-run split: D6 is graded on an eval twin, not on the shipped checkpoint (P2-14; user sign-off 2026-07-30)

- **Status:** **Accepted (user sign-off 2026-07-30, bioedca; CLAUDE.md §7 items 1+2 — AskUserQuestion: population = "eval twin on scheme-A test", ship = "full-fold P2-09/P2-10d′-b checkpoint").** Extends **D6** with the population it gates on. **Pins no new numeric value and moves no gate:** the statistic (min per-element per-nt-class F1 over the three core elements), the floor (**0.80**), the exclusions, and the reported-non-gated set are all unchanged.

**Trigger.** D6 gates *"the in-distribution homology-clustered (genus-stratified) split"*. P2-14 discovered that **no such population is held out from the checkpoint D6 grades**. The shipped production scanner trained on the **full** D5 fold (`exclude_selection_val=false`, 8,303 records; `reports/p2/train_stage1_production.json` records `selection_val_excluded: false`, `n_records: 8303`, `eval_requested: false`, `gate4_graded: false`). Measured on the committed split table, every candidate in-distribution population fails:

| candidate | n (corpus rows) | why it is not gradeable |
|---|---|---|
| `fold_random == "test"` (scheme A) | 2,353 | **1,204 (51.2 %)** are `nested_train` — inside the training stream |
| …minus the training fold | 1,149 | **796** are `nested_role == "heldout"` (786 designated leave-one-order-out + 10 Actinobacteria/anchor) — the *generalization* arm, which D6 does not gate and P4 owns |
| …minus those as well | 353 | 271 `excluded_clade_crossing` + 82 no-clade `dropped`; `splits.py:734/742` designates **both** never-scored (D3/D4) |
| `selection_val` (P2-06a inner rung) | 830 | inside the training stream **and** the fold the P2-06 sweep selected γ/lr/α on |

**Amendment.**

1. **The graded population is `gate4_eval`:** corpus records with `nested_train == True` **and** `fold_random == "test"`, closed over whole homology clusters. Measured on the committed table: **1,201 records / 1,029 clusters / 154 genera / 15 orders / 2,976,259 nt** after the ≥1024-nt interior-window filter (3 records excluded and counted, never padded). It is scheme A — random, genus-stratified, whole-cluster — intersected with the training fold, so it is in-distribution by construction, carries **0** designated-LOO records, and was **never** used for model selection.
2. **The graded artifact is the GATE-4 eval twin:** the production protocol (P2-06 winner config, 10:1 §9.1 negative mix, class-II oversampling, seed 42, fp32, DDP×8) with `exclude_gate4_eval=true`, which withholds the 1,031 `gate4_eval` clusters — **8,303 → 7,099** training records. The carve is cluster-closed and costs nothing unscored: the withheld records are exactly the graded ones.
3. **The shipped checkpoint is unchanged and is *not* graded on this population.** It saw 1,204 more records, so the twin's grade is a **conservative proxy** for it, and the direction is stated wherever the number appears. This is the **two-run split** PRD §10.1 requires the model card to disclose; the twin is DVC-tracked, retained for reproducibility, and never shipped.
4. **`eval/gate4.py` refuses to grade a checkpoint that cannot prove it is a twin.** The gate re-derives `fold_scope == "gate4_twin_train"` and `n_gate4_eval_excluded > 0` from the twin's **own training report** (the measured exclusion, not the requested flag) and binds that report to the checkpoint bytes by sha256 through the provenance sidecar. A checkpoint trained without the flag fails the clause; so does one whose flag was set but whose carve reached nothing.
5. **The leave-one-order-out read stays REPORTED and NON-GATED at P2-14** (8,621 records / 30 orders / 21.0M nt, held out from both checkpoints), macro-averaged across held-out orders per ADR-0005 D5. It is the PRD §12:241 generalization headline, graded at **GATE-1 / P4**. Two in-repo comments (`conf/train/stage1.yaml`, `slurm/p2/train_production.sbatch`) called it *"P2-14's grading set"*; that is code-comment drift against D6's own words and is corrected in this step.

**What is *not* amended.** D6's statistic, its 0.80 floor and blinded-freeze (A2), the excluded non-core classes, the reported non-gated set, and the δ-recalibration rule. On δ: the P0-21 ceiling `C` is **not estimable** — 0 of the 9 depositions resolve Specifier or Antiterminator to a residue extent (`tests/fixtures/pdb_element_extents/`, re-derived by `eval.gate4.pdb_label_noise_ceiling`) — so D6's recalibration path is **closed** and the floor stands at 0.80 whatever GATE-4 measures. D5's nested fold, D7's committed table and its schema are **untouched**: A6 adds no column and reads only `fold_random`, `nested_train` and `cluster_id`.

**Cross-reference impact.** **P2-14** (`src/tbox_finder/eval/gate4.py`, `window_dataset.{gate4_eval_cluster_ids,load_gate4_eval_records,gate4_eval_problems,load_loo_holdout_records}`, `train_stage1.exclude_gate4_eval`, `conf/train/stage1.yaml`, `slurm/p2/train_gate4_twin.sbatch`, `workflow/rules/eval.smk`, `reports/p2/gate4.json`); **P2-15** — the phase-exit gate `dvc push`es **both** checkpoints and the model card must carry the two-run split + the conservative direction; **PRD §10.1** — this is the disclosure that row already requires; **ADR-0002** — the production/naive checkpoint map is unchanged, the twin is a third, unshipped artifact; **ADR-0003** — the twin is **≈2.0 h wall / ≈16.1 GPU-h** (scaled from the shipped run's measured 14,270 steps / 2.36 h / 18.9 GPU-h by 7,099/8,303), a new line in the P2 budget; **P4** — the LOO population is untouched and unspent by this step.

### A5 — the whole-genome background-pool **host-order** negative-admission rule (Option a2; P2-10c′-a2; user sign-off 2026-07-26)

- **Status:** **Accepted (user sign-off 2026-07-26; CLAUDE.md §7 item 1 — the GTDB↔NCBI host-order-reconciliation Q-question — and §7 item 2 — this ADR amendment).** Decisions: bridge = GTDB R232 metadata `ncbi_taxid`; admissibility = blacklist-held-out (admit training + no-corpus-positive hosts); loosening fallback OFF.

**Trigger.** P2-10c′ introduced the fetched **whole-genome** negative substrate ADR-0006 A1 pins — 2,500 GTDB R232 species reps, tiled 1024/512 into **13,481,953** windows (`production_windows_v0.parquet`). A4 shipped the flank-carved negative rule (**a1**) keyed on each window's **parent corpus record** and *explicitly deferred* the whole-genome case: "a future non-flank / whole-genome background pool carves windows not anchored to a catalogued locus, breaking the coincidence … the explicit **host-accession→`resolved_order`** exclusion (**Option a2**) is **deferred to the point a whole-genome background pool is introduced** — no `host_accession→order` map exists in the repo today, so a2 is new provenance owed then, not now" [A4]. That point arrived. A fetched window's host is a `GCA_/GCF_` assembly accession (the `<accession>` prefix of its `<accession>:c<ci>:<start>` id), **not** a corpus `record_id`; the shipped a1 path (`mining/pool.py::load_parent_folds`/`stamp_parent_folds` → `data/negatives.py::admit_pool_rows`, all keyed on the corpus `record_id` of `source_record_id`, `source == "corpus"` only) has **no key** for it — so every fetched window was counted `n_unresolved_parent` and refused `parent_fold_unknown`. Fail-closed, but the pool was **unusable**: P2-10e had no negative substrate (the flank-carved a1 pool tops out at ≈0.25 % injectable — imp.md P2-10d).

**The gap this closes.** The D5 leave-one-order-out holdout is defined over the corpus's **NCBI pre-2021** `resolved_order` names — **30 designated held-out orders** + the Actinobacteria **phylum-holdout** (`member_heldout = (order ∈ heldout_orders) or (resolved_phylum == HOLDOUT_PHYLUM)`, `splits.py`; the class-holdout unit is **subsumed** — holding out the whole Actinobacteria phylum removes all its classes). The fetched genomes carry **GTDB R232** taxonomy only (`gtdb_taxonomy`; **no** NCBI order, **no** NCBI taxid, **no** fold), and GTDB reclassifies extensively, so a GTDB order name is **not** interchangeable with an NCBI `resolved_order` name. Admitting a fetched window as a training negative without resolving its host to the corpus namespace would risk seating the genomic neighbourhood of a **held-out-order** T-box in training under a negative label — the exact directional leakage A4/D2/D3 guard.

**A5 — the rule (a2).** A window from the whole-genome / non-flank pool is admissible as a training negative **iff its host genome resolves to a corpus-namespace `resolved_order` (and phylum/class) that is not a designated D5 holdout unit**, computed as:

1. **Bridge (new provenance) — GTDB R232 metadata.** Persist an **NCBI taxid** per production genome from the **already URL+MD5-pinned** GTDB metadata (`bac120_metadata_r232.tsv.gz` / `ar53_metadata_r232.tsv.gz`, `taxonomy.py::FILES`, MD5-verified on fetch by `taxonomy.ensure_file` — a new `data/external` checksummed download) joined on `gtdb_accession` → its `ncbi_taxid`; then resolve that taxid to an NCBI order/phylum/class via the **already-pinned** NCBI taxdump (`taxdmp_2026-07-01.zip`) and the **identical** `taxonomy.read_taxdump` + `resolve_row` + vintage `reconcile_name` path (same corpus vintage vocab) that produced the corpus `resolved_order`. The resolved order lands in the **same namespace** as the corpus holdout set. **No hand-built GTDB↔NCBI rename map** — reconcile via the taxdump's own synonym/merged records (the D4 anti-pattern avoided). Deterministic, same pinned R232 release, LOCAL, no live-API. Implemented in `src/tbox_finder/mining/host_order.py`; the committed, sequence-free `data/processed/mining/production_host_orders_v0.parquet` (keyed on `assembly_accession`) is the table the §8.2 CI reads, committed to git like `split_assignments.parquet`.
2. **Exclusion (the mirror of a1) — blacklist held-out.** Refuse the window **iff** its host's resolved NCBI lineage intersects **any designated D5 holdout unit** — one of the 30 leave-one-order-out held-out orders (derived from `is_designated_loo_holdout`) or the Actinobacteria phylum-holdout (`splits.HOLDOUT_PHYLUM`, cross-checked against the committed partition so a drift turns RED). **Admit** iff the host resolves entirely to **training** taxa **or** to a taxon with **no corpus positives at all** — independent-negative territory the discovery set exists to supply (ADR-0006 A1); such a genome hosts no held-out *catalogued* positive and so cannot leak one. *(The stricter whitelist reading — admit only hosts whose order carries a training positive — was rejected: it refuses the ~majority of the 197-phylum breadth for no leakage benefit.)*
3. **Fail-closed (the mirror of D4).** **Refuse** (never admit) any genome whose host order is **unresolvable** — a MAG/environmental taxid with no formal order rank, an unresolved name, or a missing taxid — and any window whose host is **absent from the host-order table** (a broken join). Mirrors D4's "no-clade record → dropped/refused"; the leakage-safe direction, at the cost of some independent negatives.

**As-built (measured 2026-07-26, LOCAL).** Over the 2,500 production genomes: **660 admissible** (3,630,683 windows) / 1,840 refused = **53 held-out-order + 11 Actinobacteria-phylum + 1,776 unresolvable** (fail-closed; **100 %** GTDB-metadata join coverage — every genome got an `ncbi_taxid`, and the 1,776 are genuine no-formal-order MAG/environmental/CPR hosts). Admissible **window** fraction **26.9 %** (3,630,683 / 13,481,953) — the leakage-safe P2-10e substrate, vs a1's ≈0.25 %. Non-degeneracy holds on real data: **18 of the 30** designated held-out orders genuinely appear among the fetched hosts (`data/processed/audits/production_host_orders_report.json`).

**The loosening fallback stays OFF** (mirror of A4). Admitting held-out-host-order windows to reclaim negatives is the one option that *weakens* leakage control; the compositional-leak motivation (an order's genomes carrying order-specific GC/codon/k-mer signal) is carried as an explicit **unquantified hypothesis** (§10.1: no ≥2-source claim), a reason to keep the default fail-closed, not a magnitude.

**CI visibility (mirror of b2).** `tests/ml/test_no_leakage.py` gains a **host-order** fail-closed gate (`host_order_negative_clauses` + unit bite tier + a committed-table tier): it drives fetched-genome windows whose hosts resolve to a **designated held-out order** through the shipped `host_order.load_host_folds`/`stamp_host_folds` path and asserts they are **refused by identity** (admitted set == the admitted-host set; from **asymmetrically-sized** held-out and training host groups, so a fold-/sense-inversion cannot pass on a symmetric count — [[symmetric-count-fixture-blind-to-inversion]]), plus must-fire companions: a designated held-out order genuinely appears among the hosts; loosening the rule re-admits **exactly** the refused windows; a broken taxid→order join turns the gate **RED**, not green-with-nothing-refused ([[namespace-mismatch-invisible-noop]]).

**Cross-reference impact.** ADR-0005 **D14** is again named **affected-by-consumption** (a2 changes which fetched windows are order-eligible for the pool D14 mines; D14's spare rule + Tier-2N probe untouched). ADR-0005 **A8** (locus-level cmsearch masking) is **complementary and non-overlapping** — A8 removes off-catalogue T-box *loci* from windows; A5 removes held-out *host-order* windows; neither addresses the other's question (A8's own text disclaims "any taxon/order-level negative exclusion"). ADR-0006 **A1** is the pool's *source* pin (source+count only; touches no labeling/fold rule); A5 is the labeling/eligibility rule A1 deferred.

**No pinned value changes; scope-extending.** No D2 cut, D6 floor, or fold size moves. A5 **extends** the negative-admission contract to a substrate a1 could not reach (whole-genome, no corpus parent) and introduces the **first order-level negative exclusion** (which A8 explicitly disclaimed). `nested_train` membership and every positive-side D-decision are unchanged.

**Scope of re-validation.** The bridge module (`mining/host_order.py`: persist per-genome NCBI taxid via the GTDB metadata join + resolve host order via the existing taxdump path; the genome-accession-keyed admit path `load_host_folds`/`stamp_host_folds`, default-armed), rule `build_production_host_orders`, the committed host-order table + provenance + report, `tests/unit/test_host_order.py`, the `tests/ml/test_no_leakage.py` host-order gate, this ADR, and a PRD §9.2 supplement note (PRD > ADR). The a1/b2 clauses and the P0-24 predicates are **untouched**. New provenance: the per-genome host-order artifact (from the two GTDB metadata files — a new checksummed `data/external` fetch, §5.2 immutable-fetch) + its committed host-order resolution.

### A7 — the D11 **disjoint calibration split**: a `calib` carve from inside the D5 training fold, genus-stratified whole-cluster, prevalence-matched on the negative side (P3-02; user sign-off 2026-08-01)

- **Status:** **Accepted (user sign-off 2026-08-01, bioedca; CLAUDE.md §7 item 2 — AskUserQuestion: negative rule = "prevalence-matched", carve size = "0.10 → 859 records", placement = "committed column + CI re-derivation", all three taken as drafted).** Extends **D7** (the committed split-table schema) and adds the
  fold-vocabulary entry ADR-0005 **D11** presumes and **A11** explicitly deferred to this
  step. **Pins no gate and moves no gate:** GATE-2's gated object (the pre-prior-shift
  named posterior), its threshold (**ECE ≤ 0.05**), its estimator (15 equal-mass debiased
  bins) and its **P3-exit** grading point are all unchanged (ADR-0005 D11).

**Trigger.** ADR-0005 D11 pins the recalibration stack as *train → temperature-scale on a
disjoint calibration split → prior-shift*, but **ADR-0004's fold vocabulary contains no
calibration fold** and the committed table has no `calib` column. A11 (P2-13) recorded this
absence explicitly, fitted its non-gated Stage-1 `T` on a seeded half of the P2-06a
`selection_val` rung as a stopgap, shipped the disclosure *"`selection_val` is the fold the
P2-06 sweep **selected on** — it is **NOT** D11's disjoint calibration split"*, and named
**P3-02** as the step that carves the real one. GATE-2's `T` is re-fitted here; **no `T`
from P2-13 is inherited.**

---

#### A7.1 — The eligible pool (measured)

`calib` is drawn from **corpus records that are in the D5 nested training fold *and* in the
scheme-A `train` fold *and* outside the P2-06a `selection_val` clusters**:

```
pool = source == "corpus"
     ∧ nested_train == True                       # D5: inside every scheme's training fold
     ∧ fold_random == "train"                     # scheme A: outside the in-distribution val/test
     ∧ cluster_id ∉ selection_val_cluster_ids(…)  # outside the P2-06 model-selection rung
```

| population | records | clusters | orders | phyla |
|---|--:|--:|--:|--:|
| corpus ∧ `nested_train` (D5 fold) | 8,303 | 4,775 | 29 | 16 |
| …∧ `fold_random == "train"` | 5,654 | 2,810 | 26 | 15 |
| …∧ ∉ `selection_val` — **the eligible pool** | **5,034** | **2,526** | **25** | **14** |

The `nested_train` fold splits `fold_random` as **train 5,654 / val 1,445 / test 1,204**.

**Three exclusions, each load-bearing:**

1. **`fold_random == "test"` is excluded** — this is the in-distribution split GATE-2's ECE
   is graded on (ADR-0005 D11; `imp.md` P3-10). PRD §12's requirement that the calibration
   split and the graded split *"must not overlap or the ECE gate is inflated"* is satisfied
   **structurally**: `fold_random` is whole-cluster assigned (measured: **0** clusters span
   more than one non-null `fold_random`), so excluding the value excludes the cluster.
2. **`gate4_eval` needs no separate exclusion** — ADR-0004 **A6** defines it as
   `nested_train ∧ fold_random == "test"` (1,204 records / 1,031 clusters, re-derived here
   with the shipped helper), which exclusion (1) already removes in full. Measured:
   pool-minus-`selection_val` and pool-minus-`selection_val`-minus-`gate4_eval` are the
   **same 5,034 records** — the intersection is empty by construction, not by luck.
3. **`selection_val` is excluded** — it is the fold the P2-06 sweep selected γ/lr/α on, and
   A11 already rejected reusing it as *"the read becomes in-sample"*. It re-derives to
   **831 records / 469 clusters** here (A11 reports 830 after `window_dataset`'s ≥1024-nt
   interior-window filter — the carve is identical, the filter is downstream).

**Robust to P3-03.** `imp.md` leaves Stage-2's train-eligibility policy to P3-03, and A6
showed that scheme-A `test` is 51.2 % `nested_train` — so whether Stage-2's in-distribution
test population ends up being all of `fold_random == "test"` (3,045 rows) or only its
non-`nested_train` part (1,741 rows), a pool restricted to `fold_random == "train"` is
disjoint from **both**. The carve does not pre-empt P3-03.

---

#### A7.2 — The draw: genus-stratified, whole-cluster, seeded (the pinned rule)

> **`CALIB_CARVE_SEED = 20260801`, `CALIB_CARVE_FRACTION = 0.10`, stratum = `resolved_genus`.**

**Algorithm (pinned, deterministic in content not iteration order — CLAUDE.md §8.3).**
Each cluster is assigned one stratum = its first non-null `resolved_genus`
(a null-genus cluster gets the explicit `"__unassigned__"` stratum, never a silent drop).
Strata are visited in **sorted** order; within a stratum, cluster ids are **sorted**, then
permuted by a single `np.random.default_rng(CALIB_CARVE_SEED)`, and whole clusters are taken
until that stratum's own record target `fraction × stratum_records` is reached.

**Why stratified and not the uniform `selection_val_cluster_ids` draw — measured, not asserted.**
The eligible pool is **98.5 % Firmicutes** and **56.7 % Bacillales**; only **2 of its 25
orders** carry ≥ 20 records. A uniform whole-cluster draw collapses onto that mode:

| draw | records | clusters | **orders** | **phyla** |
|---|--:|--:|--:|--:|
| uniform (the `selection_val` rule), fraction 0.10 | 505 | 241 | **6** | **4** |
| uniform, fraction 0.20 | 1,008 | 513 | 12 | 7 |
| **genus-stratified, fraction 0.10 (pinned)** | **859** | **431** | **23** | **13** |
| genus-stratified, fraction 0.15 | 1,095 | 573 | 23 | 13 |

A 6-order calibration set for a model whose deployment claim is phylogenetic breadth would
fit `T` on essentially two orders. Genus stratification is **not a new device**: scheme A is
*"random (**genus-stratified**)"* in PRD §9.2 and `splits.assign_random_folds` already builds
it that way, so `calib` is drawn the same way the split it must mirror was drawn.
The realized fraction (**17.1 %**, 859/5,034) exceeds the nominal 0.10 because small strata
cannot subdivide a whole cluster — a stratum of one 3-record cluster contributes 3 records or
0. The realized number is reported, never back-solved for.

**Cost.** Stage-2 training loses **859** of the 5,034 eligible positives (the D5 fold goes
8,303 → 7,444 for Stage 2). This is the intrinsic price of D11's word *"disjoint"*.

---

#### A7.3 — Measured leakage invariants (all clean under the pinned draw)

| invariant | measured |
|---|--:|
| `calib` ∩ `fold_random ∈ {val, test}` | **0** |
| `calib` ∩ `selection_val` clusters | **0** |
| `calib` ∩ `gate4_eval` clusters (A6) | **0** |
| `calib` ∩ `is_designated_loo_holdout` | **0** |
| `calib` ∩ `clade_crossing_cluster` | **0** |
| clusters split across the calib/train boundary | **0** |
| training-stream (`nested_train`) cluster-mates of a calib cluster left outside `calib` | **0** |
| non-`corpus` rows falling inside a calib cluster | **0** (all 881 rows in the 431 clusters are `corpus`) |

**The one residue, named rather than absorbed.** 22 corpus records are cluster-mates of a
calib cluster but sit outside `calib`. **All 22 are `nested_role == "dropped"`** — the D4
taxonomy-incomplete bucket, which is never trained and never scored. `calib` is therefore
defined as a **record-level predicate over the eligible pool**, and cluster-closure is
asserted in the operative form: *no cluster has a member in `calib` and a member in the
training stream or in any scored population*. Admitting the 22 would put never-scored records
into a calibration fit; excluding them leaks nothing, because 0 of them are `nested_train`.

---

#### A7.4 — The negative side: prevalence-matched (the decision `imp.md`/P3-01 flagged)

The committed split table has **no negative rows** (verified: `source` ∈
{corpus, synthetic_classII, blind, anchor}) — the A4/b2 blind spot. But GATE-2's head is
**binary**, so a positives-only `calib` cannot fit `T` at a meaningful prevalence. P3-01 left
this open deliberately: its 5,007 parentless decoys carry a **null** `nested_train`, because
*"whether a parentless decoy may enter the nested training fold is a P3-03 sampling policy,
and a `True` here would be a policy decision disguised as a data field."*

**Measured supply and the mismatch it creates:**

| calib negative rule | negatives | calib prevalence | vs. in-distribution test |
|---|--:|--:|--:|
| (a) inherit-only (dinuc decoys whose parent is a calib record) | 64 | **0.931** | 0.773 — **badly off** |
| (b) **prevalence-matched (pinned)** | **253** | **0.7725** | **0.7727 — matched** |
| (c) all 4,028 train-fold parentless decoys | 4,092 | 0.174 | 0.773 — badly off the other way |

**The pinned rule (b).** `calib`'s negatives are drawn to reproduce the **in-distribution
test split's own prevalence**, which is what D11 grades the ECE at (*"at the in-distribution
split's own prevalence"*, PRD §12). Measured on the P3-01 Stage-2 dataset, the
`fold_random == "test"` population is **2,353 positives / 692 negatives = 0.7727**. For 859
calib positives that target needs ≈**253** negatives, supplied by two routes in a fixed order:

1. Dinucleotide-shuffled decoys whose parent record is in `calib` — they inherit `calib`
   from their parent, exactly as P3-01 already has them inherit every other scheme
   (ADR-0004 D7 variant→parent→fold, applied to the negative side). This is
   **inheritance, not a draw**; it is not optional and it is not re-drawable.
2. Parentless decoys drawn from the **4,028 whose `fold_random == "train"`** at
   `DECOY_CALIB_RATE = 0.0469`, by a deterministic keyed hash on the decoy id
   (`stage2.dataset.decoy_calib`, sharing `DECOY_FOLD_SEED = 20260731` under a distinct
   `":calib:"` domain prefix). **A parentless decoy with `fold_random ∈ {val, test}`
   returns False unconditionally** — so the carve is *structurally* unable to reach the
   graded split, and the 979 val/test parentless decoys are untouchable.

   **Why a second hash and not a 4-way widening of `decoy_fold`.** Re-partitioning the
   existing 0.80/0.10/0.10 mass into four outcomes would move decoys **across** the
   train/val/test boundary, silently changing a partition P3-01 already committed. A
   second, independent draw restricted to the train portion realises the same admissible
   set without disturbing `fold_random` at all.

**As built (measured, not tuned).** The realised calibration set is **859 positives / 230
negatives = prevalence 0.7888**, against the test split's **0.7727** — a **+1.6 pp**
overshoot. The keyed hash drew **168** parentless decoys where the rate's expectation is
188.9 (−1.6 σ of the binomial), and **62** rather than 64 decoys inherited, because two of
the eligible dinucleotide shuffles are union-prior-masked out of the dataset. `DECOY_CALIB_RATE`
is **not** re-tuned to close that 1.6 pp: the rate is the pinned quantity and the realised
prevalence is a measurement (CLAUDE.md §10.3). For scale, the rejected inherit-only rule
sits **15.4 pp** off. Both numbers ship as report constants in
`data/processed/audits/stage2_dataset_report.json → calib` (`prevalence`,
`in_distribution_test_prevalence`), so a future drift is visible rather than inferred.

This makes the decision **explicit and measured**, which is precisely what P3-01 asked for:
a parentless decoy's `calib` membership is a stated rule with a rate, not a join default, and
its `nested_train` stays **null** — A7 does **not** resolve P3-03's sampling policy, it only
says which decoys the calibration *fit* may see.

**Disclosed, not hidden.** `calib` is drawn from inside `nested_train`, so its clade support
is narrower than the test split's by construction: **23 orders / 13 phyla** vs the test
split's **48 orders / 12 phyla** (the extra orders are the held-out clades `nested_train`
excludes — they *cannot* be in a training-fold carve). At phylum level the two are close
(Firmicutes **0.962** vs **0.941**); at order level calib is Bacillales 0.491 /
Clostridiales 0.469 vs test 0.364 / 0.278. This is reported as a calibration-transfer caveat
in the P3-10 GATE-2 artifact, **not** corrected by reweighting (which would be an
unpinned estimator).

---

#### A7.5 — Where `calib` lives: a committed column **plus** a re-derivation identity clause

`calib` (`bool`, non-null on every row) is **appended last to
`splits.FOLD_SCHEME_COLUMNS`** — and therefore lands in `COMMITTED_TABLE_COLUMNS`
immediately after `nested_role`, which splices that tuple. The **committed table's** schema
version goes **1.0 → 1.1**, recorded as `extra.table_schema_version`
(`splits.COMMITTED_TABLE_SCHEMA_VERSION`); the provenance sidecar's own top-level
`schema_version` is a *different* field, shared by every artifact in the repo, and is
**not** touched.

- **Appended last** so `FOLD_SCHEME_COLUMNS.index(...)` lookups (e.g.
  `window_dataset.record_order`) keep their positions — a reordering would silently
  mis-key the leave-one-order-out macro-average.
- **In `FOLD_SCHEME_COLUMNS`** so the existing `variant_parent_fold_mismatches` /
  `synthetic_variant_leaks` predicates cover `calib` inheritance for the 2,344 synthetic
  variants for free. All 2,344 are held-out-parented (A3), so all inherit `calib = False`;
  the 34 externals are `calib = False` (`fold_random` is null for them).
- **In `BOOL_FLAG_COLUMNS`** so the dtype is asserted — `bool("False") is True` would blind
  every predicate that reads it.

**Why a committed column and not a derived predicate** (the `selection_val` / `gate4_eval`
pattern): D7's whole point is that the CI re-checks the **real** partition off a committed
artifact. A derived-only `calib` would make the calibration fold a function of code that can
drift between the fitter and the checker — [[promote-dont-duplicate-is-a-correctness-rule]].

**Why the column is nevertheless not trusted on its own.** The no-leakage test **re-derives**
the carve from `cluster_id` / `nested_train` / `fold_random` / `resolved_genus` + the pinned
seed and fraction, and asserts **set identity** with the committed column — it never reads
back the boolean and calls that a pass ([[gate-clauses-need-re-derivation]]). Because two
near-equal partitions make a mis-wired carve invisible to every *count*, the assertion is on
the **identity of the cluster set**, mirroring A11's *"the validator re-cuts the fit half"*
discipline and A4/A5's must-fire companion pattern.

**New `tests/ml/test_no_leakage.py` clause set `_CALIB_CARVE_CLAUSES`** (fail-closed, each
independently sabotage-tested):

| clause | what it refuses |
|---|---|
| `calib_is_nonempty` | a carve that reached nothing (the vacuous-green failure mode) |
| `calib_cluster_set_matches_rederivation` | a committed column that disagrees with the pinned rule |
| `calib_never_in_val_or_test` | any calib record in the graded in-distribution split |
| `calib_inside_nested_train` | a calib record outside the D5 training fold |
| `calib_disjoint_from_selection_val` | reuse of the P2-06 model-selection rung |
| `calib_disjoint_from_gate4_eval` | overlap with the A6 GATE-4 graded population |
| `calib_clusters_do_not_straddle` | a cluster with members in both `calib` and the training stream |
| `calib_variants_inherit_parent` | a synthetic variant with a `calib` differing from its parent's |
| `loosening_admits_the_refused` | the must-fire companion: dropping the pool restriction re-admits exactly the refused rows, so a namespace/dtype no-op cannot read as a clean pass ([[namespace-mismatch-invisible-noop]]) |

---

#### A7.5b — A fork this step exposed: Stage 1's fold tuple was a hand-typed copy

`data/window_dataset.py` carried its **own** literal `FOLD_SCHEME_COLUMNS`, commented
*"the six fold-per-scheme columns (splits.FOLD_SCHEME_COLUMNS)"*. Adding `calib` upstream
put it one column behind **and nothing failed** — the only consumer that would have
noticed is a subset check. That is the forked-helper failure mode exactly
([[promote-dont-duplicate-is-a-correctness-rule]]): the copy silently stops meaning what
its comment says.

Stage 1 is nonetheless **right not to carry `calib`** — it is a Stage-2 calibration fold,
and widening `CorpusRecord.folds` to hold it would change `negatives.NEGATIVE_FOLDS`, every
`zip(..., strict=True)` over the pair, and the shape of every committed Stage-1 report, for
a flag Stage 1 never reads. So the fix is not to lengthen it but to **derive** it:

```python
STAGE2_ONLY_FOLD_COLUMNS = frozenset({"calib"})
FOLD_SCHEME_COLUMNS = tuple(
    c for c in splits.FOLD_SCHEME_COLUMNS if c not in STAGE2_ONLY_FOLD_COLUMNS
)
```

A new **§9.2 scheme** added upstream now flows into Stage 1 automatically, while the one
deliberate omission is *named* and asserted
(`tests/unit/test_window_dataset.py::test_stage1_fold_columns_derive_from_splits_minus_the_named_stage2_carve`,
which also re-checks `len(NEGATIVE_FOLDS) == len(FOLD_SCHEME_COLUMNS)`). Stage 1's tuple is
unchanged in content and order, so no Stage-1 artifact, checkpoint, or report is affected.

#### A7.5c — The clause set is verified by sabotage, not by reading

Each A7 clause was mutated on the **real committed partition** and confirmed to flip:
clearing one `calib` bit, admitting one extra training record, moving a calib record's
`fold_random` to `test`, splitting a calib cluster, blanking the column, carving the whole
pool, and admitting a `selection_val` record — seven targeted sabotages, each biting the
intended clause. `loosening_admits_the_refused` is verified non-vacuous the other way: it
is TRUE at baseline because dropping the `fold_random == "train"` conjunct grows the pool
**5,034 → 6,366**, so the conjunct demonstrably refuses records rather than describing a
state that does not occur ([[namespace-mismatch-invisible-noop]]).

`disjoint_from_selection_val` and `disjoint_from_gate4_eval` are **structurally implied**
by the pool definition and cannot fail while `cluster_set_matches_rederivation` holds; both
were nevertheless shown reachable under a direct mutation, and they are retained as guards
against a future change to the pool definition rather than as independent evidence.

#### A7.6 — What is **not** amended

D1–D6 in full; D7's *contents* guarantee (no sequences), its variant→parent→fold rule and
its CI-blocking status (all preserved, one column wider); D2's `d ≤ 0.30` / coverage ≥ 0.70
cut and D6's 0.80 floor (both blinded-frozen); A4/A5's negative-admission rules (untouched —
A7 governs the *calibration fit's* negative supply, not training admission); ADR-0005 D11's
gated object, threshold, estimator and P3-exit grading point; A11's Stage-1 `T`, which stays
**non-gated and un-inherited**.

#### A7.7 — Cross-reference impact list

- **P3-02** (this step): `src/tbox_finder/splits.py` (`carve_calibration_split`,
  `COMMITTED_TABLE_COLUMNS`, `FOLD_SCHEME_COLUMNS`, `BOOL_FLAG_COLUMNS`, `write_table`,
  `append_variant_rows`), the regenerated git-LFS `split_assignments.parquet` +
  `schema_version` 1.1 provenance, `tests/ml/test_no_leakage.py`,
  `tests/unit/test_split_table_schema.py` (the `==` column-list assertion).
- **P3-01 artifact**: `stage2/dataset.py::SPLIT_CARRIED_COLUMNS` gains `calib` and
  `decoy_fold` gains the 4-way outcome ⇒ `stage2_dataset.parquet` is **re-run**, and its
  golden digest (`tests/fixtures/stage2_dataset/expected.sha256`) is re-baselined in the same
  commit — a deliberate consequence, recorded, not a side effect.
- **P3-03**: train-eligibility must exclude `calib` (and the parentless decoys' `nested_train`
  stays null — A7 resolves nothing about it).
- **P3-07** (`calibration/recalibrate.py`): fits `T` on `calib`, **never** on test.
- **P3-10 / GATE-2**: grades ECE on the in-distribution test split; carries the A7.4
  clade-support caveat as a reported constant.
- **ADR-0005 D11/A11**: the "disjoint calibration split" A11 deferred now exists; A11's
  disclosure (iii) *"GATE-2's T is re-fitted at P3-02/P3 on the `calib` carve"* is discharged.
- **`imp.md` P3-02** names `src/tbox_finder/data/splits.py::carve_calibration_split`; the
  shipped module is `src/tbox_finder/splits.py` (there is no `data/splits.py`) — **path drift
  in the roadmap, corrected there.**
- **Cards / paper**: dataset card (partition strategy gains the calibration fold),
  `paper/manuscript.qmd` calibration paragraph at P3-10.

#### A7.8 — §10.1 evidence gate

A7 pins no biological or statistical *fact*; it pins a partition. The one methodological
claim it leans on — that post-hoc temperature scaling is fitted on a **held-out** split
disjoint from the graded one — is already cited by D11 (arXiv:1706.04599, Guo et al. 2017)
and is restated, not re-derived. The genus-stratification choice is justified by a
**measurement in this document** (6 orders vs 23), not by an appeal to literature. Every
count above is reproducible from the two committed artifacts by the pinned algorithm.
