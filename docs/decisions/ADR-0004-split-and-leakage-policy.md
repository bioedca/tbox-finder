# ADR-0004 — Split & leakage policy

- **Status:** Accepted (user sign-off 2026-07-10; **Amendment A1** — D2 coverage denominator — user sign-off 2026-07-10, P0-22; CLAUDE.md §7 item 2)
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

## Related documents

- **PRD:** §8 (label derivation & precedence — D1), §9.1 (masking denominator the splits share), §9.2 (splits & leakage — D2/D3/D4/D5/D7), §2.3 + §12 (GATE-4 — D6), §7.1 (the arm-(c) anchor + class-II sets D5/D7 hold out), §11 (training-fold-constrained augmentation — D7).
- **ADRs:** ADR-0001 (non-circularity), ADR-0002 (the two Stage-1 checkpoints D5 trains), ADR-0003 (D7 GATE-4-readiness reference), ADR-0005 (GATE-1 arms / decoy prevalence / min-N — pending P0-25), ADR-0006 (de-novo covariation anti-circularity — pending P0-29).
- **CLAUDE.md:** §2.3 precedence carve-out (the two pinned defaults), §8.2 (CI no-leakage), §8.5 (broadened re-run trigger), §7 items 2/4 (ADR sign-off; gate stop-and-ask), §10.3 (measured-not-fabricated prevalence).
- **Cards / paper (release-bound):** dataset card (partition strategy + train/test redundancy + clade-crossing counts), `paper/manuscript.qmd` (the split-ladder methods paragraph + GATE-4 result).

## Amendments

### A1 — D2 coverage denominator: the RF00230 **model** consensus, not the shorter member's span (P0-22; user sign-off 2026-07-10)

**Trigger.** The first real split build (P0-22) exposed an internal inconsistency in D2. The pinned coverage clause read "coverage ≥ 0.70 **of the shorter member's consensus span**", but the same bullet's rationale requires that "a locus that aligns poorly to RF00230 forms its **own singleton cluster**." These conflict for short sequences: a locus aligning to only 1 of the 224 RF00230 consensus columns has a *shorter-member span* of 1, so it is trivially 100 %-covered and — over that 1 column, at whatever nucleotide it carries — links (identity 1.0, coverage 1.0) to **thousands** of other positives.

**Measured effect (the degeneracy).** Under the literal shorter-member reading, 13 such low-occupancy hub sequences (consensus span 1–3; node degree up to **8,642**) single-linkage-bridged the corpus into one cluster of **20,941 / 23,569 records (88.9 %) spanning 66 orders and 29 phyla**, at *every* sweep cut (0.60–0.90). The cluster–clade-crossing forced rule (D3) then excluded 13,279 records, leaving a nested training fold of **676 (2.9 %)** — unusable, and directly contradicting the sign-off rationale that 0.70 "avoids the multi-order mega-cluster risk." The distances themselves were sound (median pairwise consensus identity 0.46; a Bacillales↔Actinobacteria pair 0.48; only 0.1 % of pairs ≥ the cut); the pathology was purely the coverage denominator.

**Amendment.** Coverage is measured against the **RF00230 model consensus span (`clen`)**: two positives link only when their **co-occupied consensus columns ≥ 0.70·`clen`** (and identity ≥ 0.70). A sequence covering < 0.70 of the model can form **no** edge (co-occupied ≤ its span < 0.70·`clen`) → forced singleton, realising the pinned "aligns-poorly-to-RF00230 → singleton" behaviour. The pinned *number* (0.70) is unchanged; only the denominator (model `clen`, not shorter-member span) changes. **Measured result:** class-I largest cluster **20,941 → 1,238 (5.5 %)** (9,474 clusters); class-II largest 824 (67.7 %, a genuinely tight *ileS* family, mostly held-out Actinobacteria); histogram inside-cut = 0 preserved.

**Scope of re-validation.** `src/tbox_finder/splits.py` (coverage denominator + a forced-singleton unit test), `conf/data/splits.yaml`, and the D2 threshold bullet above. Single-linkage, the 0.70 identity/coverage numbers, and every other D-decision are **unchanged**. This is an ADR-0004 §2.3 re-sign-off of a delegated pinned value (CLAUDE.md §7 item 2), not a scope change.

## Sign-off

- **User sign-off:** ☑ recorded 2026-07-10 (bioedca), CLAUDE.md §7 item 2. The D2 structure-aware clustering cut was selected as **`d ≤ 0.30` (consensus-column identity ≥ 0.70) + coverage ≥ 0.70** (single-linkage) over the tighter (0.80/0.80) and looser (0.60/0.70) alternatives, on the rationale that it over-merges slightly (the safe leakage direction) while avoiding the multi-order mega-cluster risk a lower cut invites; the re-cluster sensitivity sweep {0.60–0.90} + train↔test distance histogram (D2 adequacy net) backstop the choice. ADR accepted as drafted.
- **Amendment A1 sign-off:** ☑ recorded 2026-07-10 (bioedca), CLAUDE.md §7 item 2 / §2.3 re-sign-off. Coverage denominator = RF00230 model consensus `clen` (not shorter-member span); pinned 0.70 unchanged. Selected over a separate min-occupancy eligibility gate (equivalent effect, two knobs) and over revisiting the linkage method (single-linkage retained). Chosen on the P0-22 measured evidence above.
