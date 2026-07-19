# ADR-0005 — Non-circular evaluation design

- **Status:** Accepted (user sign-off 2026-07-11; CLAUDE.md §7 item 2)
- **Date:** 2026-07-11
- **Deciders:** bioedca (project owner)
- **Phase:** P0 (seed ADR)
- **Delegated from:** PRD §2.3 (formal acceptance gates — GATE-1, GATE-2, and the gate-precedence / blinded-freeze policy), §5 (the circularity problem & the five non-circular mechanisms), §6 (Stage-1↔Stage-2 integration + overlapping-window reconciliation + strand resolver), §9.1 (four negative/decoy classes + mining protection), §11 (recalibration stack order, aux-loss weighting), §12 (evaluation & validation framework — ECE, FDR, OOD-drift rule), §13.1 (scan decision machinery)
- **Supersedes / superseded by:** none
- **Related:** ADR-0001 (architecture & aims — the non-circularity principle this ADR operationalizes end-to-end), ADR-0002 (environment & ML stack — the two Stage-1 checkpoints [production + class-II-CM-naive] whose gate attribution D9 fixes; the RiNALMo parity gate whose swap-trigger margins D17 pins), ADR-0003 (cluster & scan ops — the governing GTDB release, minimum-viable scan, and per-corpus GPU/wall ceilings the GATE-1/GATE-3 scan runs against), ADR-0004 (split & leakage policy — the homology-cluster / leave-clade-out / literature-anchor partition and the variant→parent→fold provenance every GATE-1 arm is resampled and blocked on; the GATE-4 0.80 floor, a sibling delegated default), ADR-0006 (validation decision rule & tiering — the §13.3 confirmed-novel rule + the model-independent homolog-search thresholds that make the open-scan precision orthogonal; **pending P0-29**)

This ADR pins the **evaluation contract for the project's central methodological result — the generalization claim (GATE-1, "beat `cmsearch` at matched precision on held-out clades and on synthetic divergence") — and the calibration/FDR gate (GATE-2)** that decides whether the system is sound. It is the document that operationalizes PRD §5, the **#1 design constraint**: both the training labels and the naïve ground truth are CM-derived, so without a pre-registered non-circular design a model can at best re-learn what the CM already sees. Every decision below is a **rule**; the numeric **values** the PRD delegates here are stated as **defaults** under the PRD §2.3 precedence carve-out (the ADR-pinned value is authoritative; a recalibration still requires ADR re-sign-off, CLAUDE.md §7 item 2) and are **blinded-frozen at P0** — they may not change after P4 unblinding. Three numeric values are **deferred by design** and pinned by later P0 steps that amend this ADR: the **per-bin / per-arm minimum-real-homolog N** (P0-26), the **OOD-ECE min-N admissibility floor** (P0-27), and the **authored magnitude-rationale text** for each gate default (P0-28); this step pins the rules those steps fill in.

---

## Context

The headline result is a **generalization claim** and the **precondition for the flagship distribution-expansion result** (GATE-3). Its trustworthiness rests entirely on the evaluation being non-circular (PRD §5): a CM-trained, CM-scored model cannot demonstrate "finds what CMs miss" unless the benchmark is explicitly constructed to reach beyond the CM. PRD §5 defeats the circularity with five pre-registered mechanisms — (1) held-out-clade extrapolation, (2) synthetic-divergence stress, (3) class-II-CM-naive anti-mimicry, (4) PU framing for model-unique open-scan hits, (5) orthogonal model-independent confirmation — and PRD §2.3/§12/§13.1 delegate the operationalizing **thresholds and decision rules** to this ADR. The split that feeds all of this (homology-cluster / leave-one-order-out / literature-anchor, with variant→parent→fold provenance) is pinned in ADR-0004; the discovery-side orthogonality (the CM-free de-novo covariation and model-independent homolog search) is pinned in ADR-0006. This ADR sits between them and owns the **grading**.

Two facts drive the shape of the decisions. **First**, the catalogue is ~90 % Firmicutes across 29 phyla (P0-12), so only **leave-one-order-out** is statistically well-powered (≈31 orders with ≥20 positives; PRD §12) and the gated statistic must be **macro-averaged across held-out orders** (micro-averaging would let Firmicutes dominate) with **block-level** (cluster / held-out-order) resampling carrying the phylogenetic-exchangeability caveat into every CI. **Second**, the leave-clade-out positives are themselves CM-derived and therefore cannot contain T-boxes below `cmsearch`'s gathering cutoff — the **genuinely CM-invisible region is reached only by the synthetic-divergence arm and the §13 discovery campaign**, so those two arms carry an outsized evidential load and need their own power budget and structural-realism controls.

**The low-identity claim region is literature-sourced, not chosen for convenience.** The synthetic-divergence and lowest-identity-bin arms make their claim precisely where sequence-based homology search is documented to fail. According to PubMed, Freyhult, Bollback & Gardner (2007) benchmarked BLAST, FASTA, HMMer and Infernal on curated ncRNA sets and found the most popular (sequence-based) methods are often the least accurate on divergent homologs — the "genomic dark matter" region that motivates structure-aware search [PMID:17151342; DOI:10.1101/gr.5890907 (accessed 2026-07-10)]; Menzel, Gorodkin & Stadler (2009) independently concluded that homology search beyond the reach of BLAST "is not at all a routine task" and depends on curated structural alignments [PMID:19861422; DOI:10.1261/rna.1556009 (accessed 2026-07-10)]. These two agreeing sources ground the **fixed** low-identity bin edges in D1 (the edges isolate the region both papers identify as where sequence homology collapses); they are cited for the qualitative divergence-sensitivity finding — no specific numeric cutoff is transcribed from either paper's tables.

---

## Decision

### D1. Pre-registered identity bins (fixed edges, Freyhult-sourced) + controlled-divergence levels (locked rule; default values)

Held-out positives are binned by **% nucleotide identity to the nearest seed (training-fold) member** into **fixed, pre-registered edges — never quantile bins** (quantile bins would make the claim region data-dependent, PRD §5):

> **default edges: `<50` | `50–70` | `70–85` | `85–100` %** identity.

The **"lowest-identity bins" = the bottom two** (`<50` and `50–70`, i.e. the `<70 %` region). The edges are **fixed for the life of the project** and positioned to isolate the low-identity region where sequence-based homology search is documented to lose sensitivity and where the CM-vs-model gap is expected to open [PMID:17151342 / DOI:10.1101/gr.5890907; corroborated PMID:19861422 / DOI:10.1261/rna.1556009 — accessed 2026-07-10]. A cell with fewer than the **per-bin min-N** (D18, value at P0-26) is **pooled upward per the sub-min-N pooling rule** (adjacent-bin pooling toward higher identity, disclosed) or **reported-not-gated**, never silently passed.

**Controlled-divergence synthetic variants** (mechanism 2) are generated at a **fixed number of target-identity levels** — **default: one level centered in each of the four bins plus one deep `<40 %` stress level (5 levels total: ≈ {40, 55, 65, 78, 92} % identity)** — by structure-preserving mutation along the CM consensus, with **each variant's parent provenance pinned** (so ADR-0004's variant→parent→fold no-leakage guard applies). The level set and count are frozen here; recall-vs-divergence is plotted for the model and for `cmsearch` on identical inputs.

### D2. Canonical `cmsearch` operating point (locked)

The head-to-head baseline operating point is the **Rfam RF00230 GA (gathering) threshold for class I** and the **published `TBDB001.cm` class-II threshold for class II** (PRD §4/§2.3). A **full E-value sweep is reported as a supplementary PR curve** (not the gated point). `cmsearch`'s **build/gathering cutoff is reported next to the matched-precision threshold** on every GATE-1 figure, so the CM-invisible region (below that cutoff, reachable only by mechanism 2 + §13) is explicit. Infernal ≥ 1.1.4; a filter-off (`--max` / `--nohmm`) arm is reported per bin to attribute failures to the HMM pre-filter vs the CYK/structure stage (PRD §5, [PMID:24008419]).

### D3. Scan decision machinery — Stage-1 threshold, locus-construction, Stage-2 operating point (locked rules; values frozen at the phase gate)

Pre-registered as **rules** here; the *values* are frozen at the phase gate (§13.1) and never tuned on the test set:

- **Stage-1 threshold** = the **most-liberal (highest-recall) value that retains a pinned per-locus recall floor** (default **≥ 99 %**, measured on individually-windowed leave-one-order-out positives).
- **Locus-construction rule** (an explicit spec): the **overlapping-window logit-reconciliation operator** — at window 1024 / stride 512 every interior nucleotide is covered by ≥ 2 windows; the per-position 8-class logits from all covering windows are **averaged in log-sum-exp space then arg-maxed into one per-position prediction** *before* along-sequence element merging (a seam-free operator, so the Stage-1 recall floor holds under overlapping tiling and boundary IoU is not a 512-grid artifact) — then per-class-vs-global threshold scope, minimum span, gap-merge distance, **recall-favoring** required-element co-occurrence (mandating canonical elements would re-impose the §5/§13.3 bias), and flank size. **Training positives are offset-augmented** (random window-phase offsets) for phase robustness; contig ends are zero-flanked and flagged.
- **Stage-2 operating point** = the **most-liberal calibrated posterior with empirical false-discovery proportion ≤ 10 % (point estimate)** on the consensus null (GATE-2 then certifies this FDR on its bootstrap-CI upper bound, D12).

Reported metrics always distinguish **Stage-1-only**, **two-stage**, and **two-stage+orthogonal**.

### D4. GATE-1 two-part effect-size bar + min-N-conditioned fallback (locked rule; default values; blinded-frozen)

GATE-1 recall (vs `cmsearch` at **matched precision**, D7) passes an arm iff:

> **point estimate ≥ +10 pp recall AND block-resampled bootstrap-CI lower bound > +5 pp** (the pinned positive floor).

The weaker clause **"point ≥ +10 pp AND CI lower bound > 0"** is admissible **only as an explicit, disclosed min-N-conditioned fallback** (invoked when an arm's real-positive N is below the D18 per-arm min-N so the tighter CI cannot be estimated — never as a convenience loosening). Recall is graded on the **two-stage** system with **Stage-1-only reported alongside** (exception: the class-II anti-mimicry sub-arm is Stage-1-only, D9).

The **+10 pp / +5 pp** values are **defaults** under the precedence carve-out and **blinded-frozen at P0**: each carries a **documented magnitude rationale** — a smallest-effect-of-interest / effect-size argument or a power calculation at the D7 pinned decoy prevalence and the D18 per-bin min-N — **authored before any P4 result at P0-28** and frozen; any *pre-P4* change needs ADR re-sign-off (CLAUDE.md §7 item 2); no *post-P4* change is permitted.

### D5. GATE-1 resampling & averaging — block-level + macro-average (locked)

All GATE-1 (and per-order ECE) confidence intervals are **resampled at the homology-cluster / held-out-order (block) level, not the record level**, because the positives are phylogenetically correlated (ADR-0004; PRD §12) — this carries the exchangeability caveat into the gate CI. The gated leave-clade-out statistic is **macro-averaged across held-out orders** (not micro, which the ~90 %-Firmicutes corpus would dominate); the **per-order distribution is reported alongside**. Where N is small, the model-vs-`cmsearch` recall difference is additionally read with **small-N-appropriate inference** (an exact/permutation test or a Bayesian recall-difference posterior) beside the bootstrap CI (PRD §5).

### D6. GATE-1 headline-certifying arms + AND-of-powered-arms combination rule (locked)

The two-part bar (D4) is **required on**:

1. the **lowest-identity bins pooled** (bottom two, D1), **and**
2. a **pooled non-Firmicutes-order subgroup and/or the independent literature anchor** (arm (c), PRD §7.1) — so *beyond-Firmicutes* generalization is **certified**, and

3. **recovery of held-out class II from the class-II-CM-naive checkpoint** (the anti-mimicry sub-arm, D9).

**Combination rule = logical AND of the powered arms.** GATE-1 passes iff **every arm that is powered** (clears its D18 min-N) passes its bar. An arm below min-N is **"reported-not-gated"** — removed from the AND and **disclosed**, never silently counted as a pass. **Floor:** a GATE-1 pass **requires** arm (1) powered-and-passing **AND** at least one of {non-Firmicutes-order subgroup, arm (c)} powered-and-passing. If the **anti-mimicry sub-arm (3)** falls below the D9 min-graded-evidence floor, GATE-1 is **reported-not-gated for anti-mimicry → CLAUDE.md §7 stop-and-ask** (PRD §2.3 branch 2), not a silent waiver. The **synthetic-divergence arm being powered is a pre-registered precondition for the §1 CM-invisibility claim** (D8); unmet, that claim rests on the §13 campaign, disclosed. *(These GATE-1 arms (a)/(b)/(c) are distinct from the ADR-0004 §9.2 split-ladder (a)/(b)/(c).)*

### D7. GATE-1 closed-benchmark precision — single pinned decoy prevalence + sweep + arm-(c) basis (locked rule; default value; blinded-frozen)

GATE-1 precision is computed on the **closed leave-clade-out + synthetic-divergence benchmark** using the **four §9.1 decoy classes as labeled negatives at a single pinned prevalence** (so precision is well-defined and the §5-mechanism-4 PU framing never enters it, D10):

> **default pinned benchmark decoy prevalence: 100 : 1 (decoy : positive)** — deliberately **distinct from** the ~10 : 1 training seed ratio (§9.1) and from the ~10³–10⁴ : 1 genome-scale prevalence.

The **+10 pp gap is additionally reported as a prevalence-sensitivity sweep** spanning **10 : 1 → 10² : 1 → 10³ : 1 → 10⁴ : 1** (toward genome scale), so the gap's prevalence-robustness is visible. Because the comparison is **matched-precision** (model recall read at the threshold where model precision equals `cmsearch` precision, both scoring the identical negative pool), the pinned prevalence sets the operating point, not the fairness of the comparison. **Arm (c)'s precision** uses the **§9.1 decoy classes over the anchor's host contigs at the same pinned prevalence**, so the beyond-Firmicutes gated arm has a defined precision denominator. **Arm-(c) sourcing fallback:** if the P0-16 literature anchor / P0-17 additional class-II set falls below the D18 min-N, the P0-26 audit invokes the **arm-(c) sourcing-fallback trigger** (source more independent positives, or drop arm (c) from the AND and rest the beyond-Firmicutes certification on the non-Firmicutes-order subgroup) as a **§7 stop-and-ask**. The pinned prevalence is a **blinded-frozen default** (magnitude rationale at P0-28).

### D8. Synthetic-divergence arm — generator validity, structural realism, model-side leakage control, power budget (locked)

Before synthetic bins count toward GATE-1, **all** of the following hold (PRD §5 mechanism 2):

- **Generator-validity (structural-realism, not detectability alone):** synthetic variants at identity *X* must match **real** low-identity homologs at *X* on **both** (i) `cmsearch` detectability **and** (ii) a **structural-realism axis** — compensatory-vs-disruptive mutation distribution, MFE, and base-pair distance to the consensus. The per-bin `cmsearch` failure mode (HMM-filter pre-discard vs CYK) is reported with the D2 filter-off arm.
- **Model-side leakage control:** each variant's parent provenance is pinned (ADR-0004 variant→parent→fold) and GATE-1 **requires no train-vs-test recall gap at matched divergence** (or a held-out-clade checkpoint matching production recall) as an **acceptance condition** — so consensus-anchored generation cannot smuggle in "easiness" correlated with the CM-derived training signal.
- **Power budget → gated-vs-reported-not-gated:** P0 audits, **per leave-one-order-out round and pooled**, the per-%-identity-bin N of real held-out positives and the count of real low-identity homologs; **below the D18 minimum-real-homolog N the synthetic arm is reported-not-gated** and GATE-1 rests on the real bins (under their min-N floor) + the §13 campaign. A **powered synthetic arm is the pre-registered precondition for the §1 CM-invisibility claim** (D6).

### D9. Class-II anti-mimicry sub-arm — Stage-1-only, construction-powered, min-graded-evidence floor, within-phylum control (locked)

Anti-mimicry (mechanism 3) tests whether the **class-II-CM-naive Stage-1 checkpoint** (trained with class-I-style / shared-element labels only, withholding `TBDB001.cm` — ADR-0002/§8/§10.1) still recovers held-out translational T-boxes:

- **Scored Stage-1-only** on the naive checkpoint — **never routed through the production Stage-2** (TBDB-class-II-trained), which would rescue class-II hits and confound the test.
- Graded by a **recovery metric carrying the same block-resampled min-N floor + CI as the recall arm (D4/D5)**, plus a **within-phylum-homology control** that separates anti-mimicry from within-Actinobacteria memorization.
- The **18-record single-phylum natural set (PRD §7.1) is at chronic risk of sub-min-N**, so the sub-arm is **augmented by a construction-powered synthetic-class-II recovery measure** (above min-N by construction, built in P2) **and** by the **P0-sourced additional independent non-Actinobacteria class-II positives** (P0-17).
- A GATE-1 pass requires a **pinned minimum of graded anti-mimicry evidence** (the D18 min-graded-evidence floor); below it, **reported-not-gated for anti-mimicry → §7 stop-and-ask** (D6), never a silent waiver.

### D10. §5-mechanism-4 PU-framing scope — open-scan only (locked)

The **Positive-Unlabeled framing applies only to the open §13 discovery scan, never to GATE-1's closed benchmark.** On the discovery scan, CM-negatives are treated as **unlabeled, not true-negative**: a **model-unique hit is not auto-scored a false positive**, and its **precision is established only by orthogonal evidence** (§13 / ADR-0006, P6) — never by "another CM agrees." This framing **never enters GATE-1's closed-benchmark precision denominator** (there the four §9.1 decoys are labeled negatives at the D7 pinned prevalence). This scoping is what keeps GATE-1 precision computable while leaving the open scan free to surface CM-invisible novelty.

### D11. GATE-2 in-distribution ECE — named posterior, binned estimator, recalibration stack (locked rule; default value; gated at P3 exit)

The **recalibration stack order is pinned**: **train → temperature-scale on a disjoint calibration split [arXiv:1706.04599] → prior-shift to the deployment prior via a Saerens/Elkan log-odds correction**. GATE-2's calibration sanity gate is measured on the **named posterior = the temperature-scaled posterior *before* the deployment prior-shift, at the in-distribution split's own prevalence** — a **machinery-failure check**, not a deployment-prevalence check (a prior-shifted posterior is miscalibrated by construction at benchmark prevalence, so the **prior-shifted / deployment-prevalence ECE is reported separately, non-gated**).

> **default gate: in-distribution ECE ≤ 0.05, gated at P3 exit** (catching a calibration-machinery failure before the P5 scan; the FDR half of GATE-2 is graded at P5, D12).

**Binned-ECE estimator (pinned):** **equal-mass (equal-frequency) bins, default 15**, with a **debiasing correction**, on the positive-class posterior. Equal-mass over equal-width because equal-width bins are unstable in the sparse high-confidence region. The 0.05 default is **blinded-frozen** with its P0-28 magnitude rationale.

### D12. GATE-2 genome-scale FDR — FDP CI-upper-bound ≤ 10 % on the consensus null (locked rule; default value; gated at P5)

Genome-scale FDR uses **three nulls** (distinct from §9.1's four training-negative classes): a **structured-RNA-retaining target-decoy run as the PRIMARY estimator** [PMID:17327847]; a **dinucleotide-shuffle null as a validated optimistic lower bound** (must first reproduce `cmsearch` decoy hit-rates); and a **reversed-sequence** null (reverse, **not** reverse-complement — RC is useless against an RC-equivariant scanner) as a composition-exact control.

> **default gate: false-discovery *proportion* with bootstrap-CI upper bound ≤ 10 %** of the **pre-orthogonal candidate table** (denominator = post-Stage-2 candidate count) on the **consensus (most-conservative / highest-of-three) null**.

Gated on the **CI upper bound** because the max-of-three handles between-null but not within-null sampling error. The **primary target-decoy null additionally carries an empirical-calibration check** (held-out known-negatives, or agreement with the reversed control within tolerance) symmetric with the shuffle-null check, **before GATE-2 is graded**. **Expected false hits / Mb / genome is reported as a separate density** (not the gated quantity); retained recall on divergent loci is reported at the operating point. The 10 % default is **blinded-frozen** (P0-28 rationale). **Conformal prediction enters only as a reported held-out-clade empirical-coverage diagnostic** [arXiv:2107.07511] — both operating points are set by the D3 recall-floor and FDP ≤ 10 % rules (not a conformal α), and clade-holdout breaks exchangeability, voiding the nominal guarantee.

### D13. OOD-ECE / drift decision rule + calibrated-negative-PASS conditions (locked rule; min-N floor deferred to P0-27)

A leave-clade-out / OOD ECE ≤ 0.05 is likely infeasible under clade shift, so **OOD ECE is reported, not gated**, and adjudicated by a **decision rule**:

- Estimated with a **small-N-robust OOD estimator** — a proper-scoring-rule decomposition or a smoothed/kernel calibration estimator, with bootstrap CIs — **distinct from the D11 in-distribution binned estimator** — subject to a **pinned min-N admissibility floor (value at P0-27)**.
- A per-corpus result is a **"calibrated-negative PASS" only if (i)** its nearest-relative leave-clade-out ECE (with CIs) meets a **pinned drift bound**, **(ii)** it clears **min-N**, **and (iii)** it clears a **corpus-specific *detection-power floor*** — an extrapolated recall@matched-precision at the corpus's phylogenetic distance and/or a **synthetic-Tier-2N spike-in recovery** test — so a well-calibrated-but-*blind* model cannot earn a bounded-distribution claim. Otherwise **"sensitivity-bounded / inconclusive negative *by rule*."**
- A **sub-min-N or zero-positive corpus** (e.g. Archaea/DPANN, where no OOD ECE is computable) is inconclusive **unless** stated **nearest-relative extrapolations** (an ECE-vs-phylogenetic-distance *and* a recall/power-vs-phylogenetic-distance regression, both with **prediction intervals**) clear **both** the drift bound **and** the detection-power floor. Near-zero-prior corpora carry their **own deployment prior + per-corpus FDR null** (PRD §7.2). Per-corpus verdicts roll up to one GATE-3 outcome per the PRD §2.3 project-level rollup (pinned in ADR-0006).

### D14. Hard-negative mining spare-rule + Tier-2N probe (locked; phase-conditioned)

All known T-box loci (**the full union prior + the run's own training positives + flank**) are **masked from every negative/decoy pool** (matching the §13.3 clade-level null's masking reference), and residual contamination against that union denominator is reported. Because a CM-missed **non-canonical (Tier-2N)** T-box is unknown to masking and fails any canonical predicate, the **mining-exclusion rule is a spare rule, not keyed to canonical architecture**: a candidate is excluded from the hard-negative-mining pool if it passes **relaxed-architecture detection OR any-helix R-scape covariation OR downstream-aaRS synteny** (the three *model-independent* disjuncts, sufficient for the **P2 Stage-1 mining loop** where no Stage-2 exists yet) **OR**, at the **P3 re-mining round once Stage-2 exists**, a **high Stage-2 posterior**. The **5′UTR / tRNA-adjacent leader pool is retained** (the hardest, most-useful hard negatives). A **Tier-2N probe set** (non-canonical + synthetic-Tier-2N positives) is evaluated **each mining round**, and a per-round **recall drop on it halts/rolls back** the iteration — so aggressive mining cannot directionally train the production scanner to reject the flagship Tier-2N class; the worst case is directionally-bounded Tier-2N sensitivity, not an invalid generalization claim.

### D15. Strand resolver + strand-robustness diagnostic (locked)

Caduceus-PS is strand-agnostic, so a **strand-resolver** orients each locus from the **predicted element order** (Specifier/Stem I → antiterminator/terminator), populating the §13.1 strand column. The **RC hidden-state combination ablation is constrained to a directionality-preserving (non-averaged) form** (an order-destroying average would defeat the resolver). **Ambiguous loci** (single confident element, or scrambled order) are **flagged low-order-confidence and carried through on both strands**; the resolved strand is corroborated by **strand-specific model-independent signals** (R-scape covariation and R2DT architecture pass only on the correct strand — §13.2/§13.3(c)). A **strand-robustness diagnostic** (fraction of confirmed loci tier-invariant to strand re-resolution) is reported, so a mis-resolution degrades to a **bounded false negative on divergent loci, never a false-novelty claim**. The low-order-confidence / both-strand-carry-through fraction is reported at the sizing gate (concentrated on divergent loci).

### D16. Stage-2 multi-task aux-loss weighting + with/without-aux GATE-2 check (locked rule; default method)

Stage-2 trains a binary T-box head + boundary-refinement + regulatory-mode + auxiliary specifier/amino-acid/tRNA-family heads with a **pinned weighting method**:

> **default: fixed manual task weights** (binary head weighted to dominate), with **uncertainty-weighting (Kendall-style) as the pre-registered alternative** if the fixed weights underperform on the validation ladder.

A **with/without-aux check** confirms the aux heads **do not degrade the calibrated primary head's GATE-2 grade**; the aux-loss weight is a Hydra `--multirun` sweep axis (PRD §11), promoted via the validation ladder (never the test set).

### D17. RiNALMo → RNA-FM swap trigger margins (locked; numeric defaults; blinded-frozen)

The RiNALMo Stage-2 is swapped for RNA-FM only under the PRD §10.2 conditions; the two numeric margins the PRD delegates here are pinned as **defaults for sign-off**:

- **(c) [P3–P4] calibration margin:** swap if RiNALMo's **post-calibration leave-clade-out ECE exceeds RNA-FM's by > 0.02 (absolute ECE)**, sustained across the held-out-order distribution (block-resampled).
- **(d) [P4] discovery-metric margin:** swap if RiNALMo transfers worse on the primary discovery metric — **leave-clade-out recall@matched-precision by > 3 pp, or AUPRC by > 0.03** — even if ECE matches.

The pre-P3 RiNALMo forward-throughput probe (condition (b)) is **advisory-only**; the binding latency decision is frozen at the P5 sizing gate (ADR-0003), and a crude "clearly-hopeless" pre-P3 reading is a §7 stop-and-ask, not an auto-switch. **On a swap, the substituted RNA-FM Stage-2 must clear the *absolute* GATE-1 (+10 pp point / +5 pp CI-floor, D4) and GATE-2 (in-distribution ECE ≤ 0.05, D11) thresholds and re-pass the P4→P5 go/no-go on its own P3/P4 numbers before entering the GATE-3 scan** (PRD §2.3/§10.2) — the shipped backbone's gates are never inherited from RiNALMo. The 0.02 / 3 pp / 0.03 margins are **blinded-frozen defaults** (recalibration → ADR re-sign-off).

### D18. Delegations — values pinned by later P0 steps / elsewhere (locked map)

- **Per-bin / per-arm minimum-real-homolog N** (D1/D4/D6/D8/D9) — the floor below which an arm is reported-not-gated and the weak-clause fallback is admissible — **pinned by P0-26** (GATE-1 power-budget audit), which **amends this ADR** and renders the min-N-reachability verdict + arm-(c) sourcing-fallback trigger for the P0-16/P0-17 sets.
- **OOD-ECE min-N admissibility floor** (D13) — **pinned by P0-27** (OOD-ECE min-N coverage simulation), which amends this ADR.
- **Authored magnitude-rationale text** for each blinded-frozen default (+10 pp / +5 pp — D4; benchmark decoy prevalence — D7; ECE ≤ 0.05 — D11; FDR ≤ 10 % — D12; the RNA-FM margins — D17) — **authored at P0-28** and blinded-frozen; this ADR pins the defaults and the freeze policy, P0-28 pins the *why*.
- **Governing GTDB release + union novelty prior** (used by every GATE-1/GATE-3 novelty determination) — the release is pinned in **ADR-0003 (Amendment A1)** and the union prior is **frozen at P0** (P0-14); this ADR consumes them, does not re-pin them.
- **GATE-4 segmentation floor (≥ 0.80)** and the **split/precedence/variant-provenance contract** — pinned in **ADR-0004**; **§13.3 confirmed-novel rule + model-independent homolog-search thresholds + project-level rollup** — pinned in **ADR-0006** (pending P0-29).

---

## Consequences

- **The generalization claim is falsifiable and non-circular by construction.** Every GATE-1 number is block-resampled, macro-averaged, matched-precision, and certified beyond Firmicutes; the CM-invisible region is reached only by the powered synthetic arm and the §13 campaign, both explicitly gated or explicitly disclosed-as-resting-on-§13.
- **Under-power is a first-class, non-silent outcome.** The AND-of-powered-arms rule + reported-not-gated + the §7 stop-and-ask (D6/D8/D9) mean an arm can never be quietly dropped or loosened to manufacture a pass; a min-N shortfall routes to PRD §2.3 branch 2 (re-power or the resource-paper path), not to a downgraded claim.
- **Two numeric values and all magnitude rationales are deliberately deferred** (D18) so the min-N floor is measured (P0-26/P0-27) and the rationales are authored blinded (P0-28) — but the *rules* that consume them are locked now, so those steps fill in numbers into a fixed frame.
- **Calibration is a machinery gate, not a deployment claim** (D11): the named-posterior ECE catches a broken calibration head before the scan, while the honest OOD/deployment miscalibration is reported and adjudicated by the drift rule (D13), so a well-calibrated-but-blind model cannot earn a bounded-distribution claim.
- **Risk owned:** the fixed bin edges (D1) and the pinned benchmark prevalence (D7) are judgment calls frozen before results; if P0-26 finds the lowest-identity bins are chronically sub-min-N, the synthetic arm carries the CM-invisibility claim and its generator-validity budget (D8) becomes the load-bearing control — a dependency this ADR makes explicit rather than hiding.

## Related documents

- **PRD.md** §2.3, §5, §6, §9.1, §11, §12, §13.1 (delegating sections).
- **ADR-0001** (aims / non-circularity principle), **ADR-0002** (two Stage-1 checkpoints; RiNALMo parity gate), **ADR-0003** (governing GTDB release; minimum-viable scan; compute ceilings), **ADR-0004** (split & leakage policy; GATE-4; variant→parent→fold), **ADR-0006** (§13.3 rule; model-independent homolog search; rollup — pending P0-29).
- **Amended by:** P0-26 (min-N), P0-27 (OOD-ECE min-N floor), P0-28 (magnitude-rationale text).

## Cross-reference impact list

- **P0-26** consumes D1/D4/D6/D7/D8/D9/D18 and amends this ADR with the per-bin/per-arm min-N.
- **P0-27** consumes D13/D18 and amends this ADR with the OOD-ECE min-N admissibility floor.
- **P0-28** consumes D4/D7/D11/D12/D17/D18 and authors the blinded magnitude rationales.
- **P0-30** (decoy prevalence + mining spare rule) consumes D7/D14; **P2** (hard-negative mining, class-II augmentation) consumes D8/D9/D14; **P3** (ECE) consumes D11/D13; **P4** (GATE-1) consumes D1–D10/D15–D17; **P5** (FDR) consumes D3/D12; **P6** (orthogonal validation, PU precision) consumes D10.
- **Phase-0 exit (2026-07-12):** the blinded-frozen GATE-1 (recall@matched-precision vs `cmsearch`, +10 pp/+5 pp bar) and GATE-2 (ECE ≤ 0.05, FDR CI-upper ≤ 10 %) design is described (values unchanged, no result asserted) in the `paper/manuscript.qmd` §Non-circular evaluation design paragraph and the `README.md` P0 headline; consolidated in `docs/dev-log/phase0_2026-07-12.pdf`.

## Sign-off

**Accepted — user sign-off 2026-07-11 (CLAUDE.md §7 item 2, P0-25), "accept as drafted."** The scientific-evidence gate for the D1 fixed identity-bin edges was cleared with two independent agreeing sources [PMID:17151342 / DOI:10.1101/gr.5890907; PMID:19861422 / DOI:10.1261/rna.1556009, accessed 2026-07-10]. All numeric defaults below are now **blinded-frozen at P0**: they may not change after P4 unblinding, and any pre-P4 recalibration requires ADR-0005 re-sign-off.

**Blinded-frozen numeric defaults:** D1 bin edges `<50|50–70|70–85|85–100 %` + 5 divergence levels; D3 Stage-1 recall floor ≥ 99 %; D4 **+10 pp / +5 pp**; D7 benchmark decoy prevalence **100:1** + sweep 10:1→10⁴:1; D11 **ECE ≤ 0.05**, 15 equal-mass debiased bins; D12 **FDR CI-upper ≤ 10 %**; D16 fixed aux weights; D17 **ECE +0.02 / recall +3 pp / AUPRC +0.03** swap margins. The min-N floor (P0-26), OOD-ECE min-N admissibility floor (P0-27), and authored magnitude rationales (P0-28) are pinned by later P0 steps that amend this ADR (D18).

---

## Amendment A1 — Minimum-real-homolog N pinned + GATE-1 power-budget audit (P0-26, 2026-07-11)

- **Status:** Accepted (user sign-off 2026-07-11; CLAUDE.md §7 item 2 — the D18 min-N delegation). Pins D18; consumed by D1/D4/D6/D8/D9.

**Pinned value.** `MIN_REAL_HOMOLOG_N = **20**` real held-out positives **per cell** — each D1 identity bin and each D6 headline-certifying arm. Below it a cell is **reported-not-gated** (removed from the D6 AND-of-powered-arms, disclosed) and the D4 weak clause (point ≥ +10 pp AND block-resampled CI lower bound > 0) is admissible; **at or above it** the strong D4 bar (CI lower bound > +5 pp) applies. **Blinded-frozen at P0** — a change needs ADR-0005 re-sign-off.

**Rationale.** (i) One internally-consistent floor: 20 equals the split's already-pinned well-powered leave-one-order-out unit (`conf/data/splits.yaml min_heldout_positives = 20`) and PRD §12 ("31 orders with ≥ 20 positives"; "the ~20-positive floor is statistically unstable"). (ii) It is an **estimability floor, not a power-to-pass floor**: at N < 20 the per-positive recall granularity 1/N > 5 pp exceeds the D4 strong-bar positive floor, so the tighter CI cannot resolve it (D4's "the tighter CI cannot be estimated"); at N ≥ 20 the strong CI is estimable — whether a *powered* arm then *passes* is a separate P4 question. (iii) N ≥ 20 keeps the normal-approximation binomial interval non-degenerate for central recall (extreme recall uses the D5 exact/Wilson / permutation inference). A **companion block count** is reported per arm (D5 resamples at the homology-cluster / held-out-order block level; < 2 blocks → not block-resamplable) — **reported, not a second pinned floor**.

**Audit result** (`src/tbox_finder/power.py`; `data/processed/audits/power_budget_report.json`; over the real 23,569-record P0-22 partition; identity = the ADR-0004 D2 consensus-column metric, `coverage_cut = 0.70`):

- **Held-out N = 8,749.** Every held-out positive has nearest-training identity **< 0.70 (max 0.699)** by whole-cluster holdout, so every one falls in the bottom two D1 bins — `<50`: 85 · `50–70`: 6,793 · `70–85`: **0** · `85–100`: **0** · no-coverage-adequate-neighbour (‑1): 1,871. **The 70–85 and 85–100 bins are provably empty for the leave-clade-out arm** (reachable only by the D8 synthetic-divergence arm + the §13 campaign).
- **Low-identity homolog count:** 6,878 measurable `< 70 %` + 1,871 no-adequate-neighbour = 8,749. **Per leave-one-order-out round:** 43 held-out orders, **30 ≥ min-N**.
- **Reachability of the headline-certifying cells:** (1) lowest-identity bins pooled — N = 8,749 / 43 blocks → **POWERED**; (2) non-Firmicutes-order subgroup — N = **1,635** / 34 blocks (Actinobacteria 1,230 · Tenericutes 202 · Chloroflexi 117 · Proteobacteria 31 · Deinococcus-Thermus 28 · Synergistetes 27) → **POWERED**; upper bins 70–100 % — N = 0 → **structurally empty**; arm (c) literature anchor (P0-16) — N = **16** (5 coordinate-novel + 11 corpus-overlap) → **below min-N**; class-II anti-mimicry independent-of-Actinobacteria natural set (P0-17: 0 + blind non-Actino: 0) — N = **0** → **below min-N**.

**Determinations.**

1. **Arm-(c) sourcing-fallback (D7 §7 stop-and-ask; user sign-off 2026-07-11).** The P0-16 anchor's N = 16 < 20 → **arm (c) is dropped from the D6 mandatory AND**; the **beyond-Firmicutes certification rests on the powered non-Firmicutes-order subgroup** (N = 1,635 / 34 blocks). The D6 floor is satisfied (arm 1 powered **AND** the non-Firmicutes-order subgroup powered). The 16 re-derived anchor loci are **reported as corroborating, non-gating** model-independent evidence, not sourced-up further.
2. **Class-II anti-mimicry (D9).** The natural independent-of-Actinobacteria class-II set is N = 0 < 20 → **reported-not-gated**; the sub-arm **rests on the P2 construction-powered synthetic-class-II recovery set** (above min-N by construction) **+ the D9 within-phylum-homology control**, exactly as D9 pre-registered. (The total held-out class-II N ≈ 1,190 is Actinobacteria-dominated — single-phylum, so within-Actinobacteria memorization-confounded — and the ~12 non-Actinobacteria held-out class-II corpus records are CM/DB-derived P0-17 *leads*, not independent positives.)
3. **Synthetic-divergence arm (D8).** Remains the **pre-registered precondition for the §1 CM-invisibility claim**: the real leave-clade-out bins reach only to 0.699 identity, so the deep-divergence (`< 40 %`) cells are supplied only by the P2/P4 synthetic arm and the §13 campaign — unmet, that claim rests on §13, disclosed.

**Cross-reference impact:** P0-28 uses the pinned per-bin/per-arm min-N in its D4 magnitude-rationale power calculations; P4 reads the powered-vs-reported-not-gated arm set (arm (c) dropped from the AND; beyond-Firmicutes on the non-Firmicutes-order subgroup); P2 builds the construction-powered synthetic-class-II recovery set the anti-mimicry sub-arm now depends on.

---

## Amendment A2 — OOD-ECE min-N admissibility floor pinned + coverage simulation (P0-27, 2026-07-11)

- **Status:** **Accepted (user sign-off 2026-07-11; CLAUDE.md §7 item 2 — the D18 OOD-ECE-min-N delegation), "accept 20 as drafted."** Pins D18 (the second deferred value); consumed by D13; feeds the ADR-0006 rollup (P0-29). Committed atomically with the P0-27 simulation artifacts.

**Pinned value.** `OOD_ECE_MIN_N = **20**` real positives in a deployment corpus's nearest-relative leave-clade-out **calibration unit** (the leave-one-order-out benchmark unit). Below it the D13 small-N-robust OOD-ECE estimator is **inadmissible** → the corpus is **"sensitivity-bounded / inconclusive negative *by rule*"** (never a silent calibrated-negative PASS). The floor is an **admissibility gate on condition (ii) of the D13 three-part PASS** (drift bound ∧ min-N ∧ detection-power floor); conditions (i)/(iii) are separate P3/P4 quantities. **Blinded-frozen at P0** — a change needs ADR-0005 re-sign-off. **Frozen in code** (`src/tbox_finder/coverage.py::OOD_ECE_MIN_N`, no CLI/config override) so no committed report can contradict the pin (Amendment A1 precedent).

**Rationale.** (i) **One internally-consistent floor:** 20 equals the recall floor `MIN_REAL_HOMOLOG_N` (Amendment A1), the split's `min_heldout_positives = 20`, and PRD §12 ("31 orders with ≥20 positives"; the ~20-positive per-order ECE is "statistically unstable"). (ii) **Estimator-matched:** the D13 OOD estimator is a *small-N-robust* proper-scoring-rule / kernel calibration estimator with bootstrap CIs — chosen precisely so N = 20 is admissible (bootstrap CI reported, not per-bin filled), distinct from the D11 in-distribution 15-equal-mass-bin estimator that would need far more; below 20 the per-order calibration estimate is PRD-documented unstable → inadmissible. (iii) **Conservative by construction:** a *higher* floor only makes *more* corpora inconclusive-by-rule (anti-overclaiming), and this is an *estimability* floor, not a power-to-pass floor — clearing it admits an OOD-ECE estimate, it does not manufacture a PASS. A more conservative reviewer floor (30/50) is reported in the coverage sweep so the sensitivity is visible; 20 is pinned for internal consistency.

**Simulation result** (`src/tbox_finder/coverage.py`; `data/processed/audits/ood_ece_coverage_report.json`; seeded block bootstrap over held-out orders, seed 42, B = 2000; over the real 23,569-record P0-22 partition):

- **43 held-out leave-one-order-out units; 30 clear the floor.** Adjudicable fraction **among positive-bearing orders = 0.698** (block-bootstrap 95 % CI **0.558–0.837**; clearing-count CI 24–36) — the PRD §12 quantity. **13 sub-min-N orders** are inconclusive-by-rule. Floor sweep {10 → 32 · 15 → 31 · 20 → 30 · 30 → 22 · 50 → 16} clearing (fraction 0.744 → 0.372).
- **Adjudicable calibration footprint = 7 bacterial phyla, 0 archaeal** (Actinobacteria, Chloroflexi, Deinococcus-Thermus, Firmicutes, Proteobacteria, Synergistetes, Tenericutes), **hyper-concentrated**: Firmicutes = 81.2 % of corpus held-out positives, the single largest order (Lactobacillales) = 61.1 % of all held-out. The named zero-positive scan superclades **{Archaea, DPANN, CPR/Patescibacteria, candidate phyla}** are **measured-absent** from the partition's resolved phyla ("0/23,535 TBDB records are archaeal", PRD §7.2 — the catalogue's silence, ascertainment-limited).
- **Non-fabrication guard:** per-order N is **byte-identical to the signed P0-26 `power_budget_report.json`** (0 mismatched orders). Internal consistency `OOD_ECE_MIN_N == MIN_REAL_HOMOLOG_N == 20`.

**Determinations.**

1. **"Discovery-predominantly-inconclusive" is a pre-registered modal GATE-3 shape (§2.3), quantified ex ante.** The adjudicable calibration footprint spans only **7 bacterial phyla**, while the §13 scan (PRD §7.2) targets the whole prokaryotic tree — including the named zero-positive superclades (all Archaea/DPANN, CPR/Patescibacteria, candidate phyla) and every T-box-free bacterial phylum, each **inconclusive-by-rule ex ante** (no leave-clade-out ECE is computable). So the modal per-corpus verdict is inconclusive-by-rule *before any scan runs* — the terminal outcome is a distinct pre-registered state, not a failure (§2.3).
2. **Two-scope adjudicable fraction.** The 0.698 figure is scoped to positive-bearing orders (PRD §12). The **scan-wide** adjudicable fraction is far lower and **not pinnable here** (the GTDB corpus enumeration is deferred to ADR-0003); it is bounded above by nearest-relative reach of the 7-phylum footprint and is dominated by the zero-positive complement.
3. **Anti-under-scope guard.** Because the floor is an admissibility gate, a **budget-forced under-scope cannot masquerade as a calibrated-negative PASS** (a PASS additionally requires the D13 drift bound *and* the corpus-specific detection-power floor — separate P3/P4 quantities, not this gate).

**Cross-reference impact:** P0-29 authors the ADR-0006 project-level rollup function that consumes this verdict-vector shape; P3 (ECE) reads the D13 OOD estimator's admissibility floor; P6 (GATE-3 tiering / orthogonal validation) reads the ex-ante adjudicable-fraction expectations. The floor does **not** alter the GATE-1 recall arms (Amendment A1) — it governs GATE-2/GATE-3 deployment calibration only.

---

## Amendment A3 — Authored gate-default magnitude rationales + blinded-freeze declaration (P0-28, 2026-07-11)

- **Status:** **Accepted (user sign-off 2026-07-11; CLAUDE.md §7 item 2 — the D18 authored-magnitude-rationale delegation, the third and last deferred value), "accept both as drafted".** Pins D18; records the SESOI / power argument for each blinded-frozen default of D4, D7, D11, D12, D17 (and cross-references the ADR-0004 D6 0.80). Committed atomically with `analyses/gate_default_rationales.qmd` + `src/tbox_finder/power.py::magnitude_rationale` + `tests/unit/test_magnitude_rationale.py`.

**What this amendment pins.** ADR-0005 pinned the *rules and the default values* (D4/D7/D11/D12/D17) and declared them blinded-frozen; it deferred the **authored magnitude-rationale *text*** to P0-28 (D18). This amendment supplies that text — a smallest-effect-of-interest (SESOI) / effect-size argument or a power calculation for **each** delegated default — **authored 2026-07-11, before any P4 result exists**, and declares the freeze. The full derivations live in `analyses/gate_default_rationales.qmd`, rendered verbatim from `src/tbox_finder/power.py::magnitude_rationale` (single source of truth; a unit test asserts the code constants match these ADR-pinned values byte-for-byte).

**Scientific-evidence gate (§10.1) — cleared, ≥2 agreeing method sources per high-stakes convention** (identifiers verified against PubMed / the publisher of record; no numeric cutoff transcribed from any paper — §10.3):

- **SESOI / equivalence-bound method:** Lakens 2017 [DOI:10.1177/1948550617697177]; Lakens, Scheel & Isager 2018 [DOI:10.1177/2515245918770963] (accessed 2026-07-11).
- **Binned-ECE / well-calibration:** Guo et al 2017 [arXiv:1706.04599]; Naeini, Cooper & Hauskrecht 2015 [PMID:25927013] (accessed 2026-07-11).
- **Discovery-stage FDR:** Benjamini & Hochberg 1995 [DOI:10.1111/j.2517-6161.1995.tb02031.x]; Storey & Tibshirani 2003 [DOI:10.1073/pnas.1530509100]; Elias & Gygi 2007 [PMID:17327847, already D12] (accessed 2026-07-11).
- **Low-identity field context (recall SESOI):** Freyhult et al 2007 [DOI:10.1101/gr.5890907]; Menzel et al 2009 [DOI:10.1261/rna.1556009] (already D1).

**Authored rationales (compact; full text in the QMD).**

1. **D4 recall point bar (+10 pp).** *SESOI + power.* The smallest recall gain over `cmsearch` at matched precision that is scientifically material **and** robustly separated from noise at the pinned min-N: +10 pp = **two per-positive granularities** (2 × 1/20), so a passing arm's point estimate sits a full CI-floor-width above the +5 pp non-trivial-improvement boundary. In the bottom-two identity bins where sequence homology search is documented to lose sensitivity (D1 sources), +10 pp = recovering ≥1-in-10 more divergent T-boxes than the CM.
2. **D4 recall CI floor (+5 pp).** *Power (estimability).* The finest recall-difference lower bound the block-resampled CI can resolve at min-N = 20 (1/20 = 5 pp per positive). This is the **same 1/N ≥ 5 pp argument that pinned `MIN_REAL_HOMOLOG_N` (Amendment A1)** — the recall bar and the min-N floor are **one internally-consistent construction**, not two independent picks. Below min-N the D4 weak clause (CI lower > 0) is the disclosed fallback.
3. **D11 in-distribution ECE (≤ 0.05).** *SESOI + power.* A ≤ 5 % confidence-accuracy gap on the temperature-scaled named posterior (a machinery-failure budget). Two backings: **(i)** it is half the D12 FDR budget (0.05 / 0.10), so a GATE-2-passing calibration head cannot by itself exhaust the downstream FDR budget; **(ii)** with 15 equal-mass debiased bins the binned-ECE estimator resolves 0.05 above its own noise floor at realistic calibration-set N, yet 0.05 is tight enough to catch a broken head. Post-hoc temperature scaling routinely reaches this range → achievable-yet-meaningful.
4. **D12 genome-scale FDP (≤ 0.10).** *SESOI.* A **discovery-stage** FDR on the *pre-orthogonal* candidate table — the candidates then pass ADR-0006 orthogonal Tier-1/2/2N validation (the terminal FP control), so the right error rate here is a *discovery* FDR (FDR/q-value is the established genome-wide-discovery criterion, "more liberal" than a terminal 1–5 %). 10 % is the SESOI on the candidate table: tight enough that orthogonal validation isn't swamped (E[false] = 0.10 × table size, reported as a density), loose enough to retain the divergent-loci recall that is the model's advantage. Gated on the CI upper bound.
5. **D7 benchmark decoy prevalence (100 : 1).** *Design (not SESOI/power).* Sits ~1 decade above the ~10 : 1 training seed ratio and 1–2 decades below the ~10³–10⁴ : 1 genome scale; because the comparison is matched-precision, prevalence sets only the operating point (the +10 pp gap is reported over a 10 : 1 → 10⁴ : 1 sweep). Documented here per the D18 delegation.
6. **D17 swap margins (0.02 ECE / 3 pp recall / 0.03 AUPRC).** *SESOI on the backbone-choice decision.* 0.02 ECE = 40 % of the D11 gate (a meaningful fraction of the calibration budget), **sustained** across the held-out-order distribution; 3 pp = the smallest *sustained* recall deficit worth switching backbones over (a single-arm point difference is already governed by the +5 pp floor); 0.03 AUPRC = a comparable threshold-free ranking deficit.
7. **ADR-0004 D6 GATE-4 floor (≥ 0.80).** Co-authored into ADR-0004 (Amendment A2, this session): the smallest per-nt F1 on the 3 core elements that demonstrates learned element *extents* and supports §13.1 locus construction + §13.3(d) specifier read, while tolerating the ~1–2 nt dot-bracket-projection boundary ambiguity; the N ≤ 9 PDB label-noise ceiling C (P0-21, non-gated) must not one-directionally lower it.

**Honest caveat (recorded, not hidden).** The naïve normal-approximation minimum-detectable-effect at a 0.5 baseline and N = 20 is ≈ 22 pp — larger than the +10 pp bar. This is by design: min-N = 20 makes the +5 pp CI floor **estimable**, not the exactly-min-N arm **powered to pass**. GATE-1's gated inference is the block-resampled bootstrap CI + the D5 small-N exact/permutation test, and the powered arms pool far above N = 20 (lowest-identity pool N = 8,749 / 43 blocks; non-Firmicutes-order subgroup N = 1,635 / 34 blocks — Amendment A1). An arm clearing min-N but not the bar is reported-not-gated / §7 stop-and-ask, never a silent pass (D6).

**Blinded-freeze declaration.** All rationales were authored **2026-07-11, before any P4 benchmark result exists**. **No post-P4 change is permitted**; a **pre-P4 recalibration requires ADR re-sign-off** (CLAUDE.md §7 item 2), recorded as a further amendment. This closes the D18 "authored magnitude-rationale text" delegation — the third and final deferred value.

**Cross-reference impact:** P4 (GATE-1) and P5 (FDR) inherit these frozen defaults and may not change them post-unblinding; the ADR-0004 D6 0.80 rationale is co-authored there (Amendment A2). No rule or data artifact changes — this amendment adds the *why* behind numbers already pinned by ADR-0005 / ADR-0004.

---

## Amendment A3 — D3 locus rule: the reconciliation operator normalises per window before averaging (P2-03, 2026-07-16)

- **Status:** **Accepted (user sign-off 2026-07-16; CLAUDE.md §7 item 2), "(A) Normalise per window — amend ADR."** Disambiguates D3's locus-construction rule (it pins no new *value*); consumed by P2-03 (`infer/reconcile.py`), P2-14 (GATE-4), and the P5 scan. Committed atomically with the P2-03 operator + its unit/golden gates.

**The ambiguity.** D3 pins "the per-position 8-class **logits** from all covering windows are **averaged in log-sum-exp space then arg-maxed** into one per-position prediction" (identically in PRD §6). That sentence admits **two coherent operators**:

- **(B) literal** — `argmax_c log( mean_w exp( logits[w,p,c] ) )` — a soft-max pooling over windows, applied to the raw logits.
- **(A) implemented** — `argmax_c log( mean_w softmax(logits[w,p])_c )` — the arithmetic mean of the per-window **posteriors**, computed in log space.

They are **not notational variants**. Measured at P2-03 on the golden fixture's real pinned-tiling geometries (window 1024 / stride 512 over nine real sampled P2-00 context lengths, 1,380–2,556 nt): **3,418 of 8,931 multi-covered positions — 38.3% — receive a different class** under (A) vs (B). (Synthetic near-uniform logits maximise the disagreement; a trained head's peaked logits agree far more often. The number establishes the choice is **material**, not that it is this large in production.)

**Pinned form.** **(A).** `log_probs[p] = log( mean over covering windows w of softmax(logits[w,p]) )`, computed as a coverage-normalised log-sum-exp of per-window `log_softmax` values, then `argmax`, applied **before** any along-sequence element merging. **Frozen in code** (`src/tbox_finder/infer/reconcile.py`, no config override), with `diagnostics()` reporting `pinned_by = "ADR-0005 D3 + A3"` (the A1/A2 freeze-in-code precedent).

**Rationale.**

1. **(B) weights each window by an unconstrained nuisance quantity — the discriminating argument.** `exp(logit)` is an *unnormalised* score, so averaging it weights window `w` by its partition function `Z_w = Σ_c exp(logits[w,p,c])`. **Nothing in the training objective constrains `Z_w`**: the softmax inside the cross-entropy is invariant to a constant shift of a position's logits, so the model is never trained to control that offset. Under (B), two windows predicting the **identical posterior** contribute unequally whenever their logit offsets differ — for a reason the model never claimed. Under (A) they contribute equally, by construction.
2. **(A) is the standard ensemble average** of member predictive distributions, and the reconciliation *is* an ensemble over covering contexts.
3. **The D3 consumer wants a posterior.** The Stage-1 threshold is "the most-liberal value retaining a per-locus recall floor" and the §11 recalibration/ECE stack grades a calibrated posterior; (A) emits one directly (`Σ_c exp(log_probs) = 1`). *Recorded honestly: this argument is **weaker than it looks** and did not decide the matter — (B)'s output can also be softmaxed over `c`, yielding a distribution with the same arg-max. It is a convenience argument; rationale 1 is the discriminating one.*

**What this amendment does NOT change.** The **coverage normalisation** (`− log |W(p)|`) is common to both readings and was never at issue: it is what "**averaged**" means as against "summed", and D3's own stated purpose — seam-freeness, "boundary IoU is not a 512-grid artifact" — *requires* it, since without it an interior nucleotide (coverage 2–3) would outscore a tail nucleotide (coverage 1) on identical evidence. The **arg-max-last** ordering, the **before-merging** placement, the **zero-flank-and-flag** contig-end rule, and the 1024/512 tiling are implemented exactly as D3/PRD §6 pin them. No numeric gate value moves.

**Cross-reference impact:**
- **P2-03** — `infer/reconcile.py::reconcile_windows` implements (A); golden digest `08f088c9…` is (A)'s. A unit test pins the operator identity against **external literals**, and the three behavioural claims in `diagnostics()` are **re-derived from measured behaviour**, not echoed (the P1-15/P1-16 self-certification lesson).
- **P2-14 / GATE-4** — per-nucleotide F1 and boundary IoU are graded on (A)'s arg-max.
- **P5 scan** — the §13.1 candidate table derives from (A); the D3 Stage-1 threshold is set on (A)'s posterior at the phase gate (unchanged **as a rule**).
- **D12/GATE-2, D4/GATE-1, the strand resolver** — all consume locus calls *downstream* of this operator; unaffected as rules.
- No data artifact is invalidated: no Stage-1 checkpoint exists yet (P2-04), so nothing was reconciled under the other reading.

---

## Amendment A4 — Tier-2N probe eligibility requires a length-matched excision control (P2-07, 2026-07-19)

- **Status:** **PROPOSED — awaiting user sign-off (CLAUDE.md §7 item 2).** Not in force until signed. Amends **D14** (the Tier-2N probe set); consumed by P2-09 (round-0 mining), P5 (synthetic-Tier-2N spike-in recovery / §12 detection-power floor), P6.
- **Note on numbering:** this file already carries **two** amendments labelled `A3` (P0-28 and P2-03) — a pre-existing collision flagged for reconciliation at the P2 exit gate. This amendment takes `A4` to avoid compounding it.

**What D14 leaves open.** D14 pins that a Tier-2N probe set is evaluated each mining round and that a recall drop halts/rolls back the iteration. It does not say how a synthetic construct **earns** Tier-2N membership. P2-07 built that generator and found the obvious operationalisation — "the covariance model misses it" — is confounded twice over, measured rather than assumed.

**Measured (P2-07, `RF00230.cm --cut_ga`, INFERNAL 1.1.5; `reports/p2/tier2n_probe.json`).**

| quantity | value |
|---|---|
| unablated corpus records missed (n=500, seed 42) | **27.6 %** (362/500 detected) |
| — independently reproduced (n=300, seed 20260719) | 27.0 % (219/300 detected) |
| real element ablations missed (n=599) | **66.7 %** |
| **length-matched random excisions missed (n=599)** | **78.8 %** |

1. **Divergence confound.** Better than a quarter of real, architecturally-canonical TBDB T-boxes are already missed at the GA cutoff, because RF00230's `GA` is 93.00 bits and the corpus was built with a different model. Labelling a CM miss as Tier-2N would grade *sequence divergence* — the separate GATE-1 divergence arm (D6/D8) — as *architectural* novelty.
2. **Excision-length confound, the decisive one.** A covariance model is an alignment, so removing nucleotides degrades it regardless of *which* ones. Excising an equal-length segment from a **random non-element position breaks detection MORE often (78.8 %) than removing the actual Stem II or Terminator (66.7 %)**. Element excision is therefore *less* destructive than the null it must beat: on this evidence a parent/variant discordance carries **no architecture-specific signal at all**, and a probe set built from it would measure excision.

**Pinned rule.** A synthetic construct is **Tier-2N-probe-eligible** only as a measured **triple**:

| leg | required verdict | confound removed |
|---|---|---|
| parent | CM-**detected** | divergence |
| variant (element ablated) | CM-**missed** | — the effect under test |
| length-matched control (equal-length excision at a non-overlapping position) | CM-**detected** | excision length |

Every emitted variant **carries its own control**, generated with it, so a control cannot be omitted; an absent control arm leaves every variant ineligible rather than degrading to the confounded two-leg filter. An unmeasured leg is **never** read as a miss. Discards are reported per cause and the per-family split is reported alongside the pooled count, so a family resting on the min-N floor stays visible. The D18/A1 floor `MIN_REAL_HOMOLOG_N = 20` applies to the probe set unchanged.

**Effect at P2-07:** 599 emitted → 45 eligible (21 class-II, 24 stem-II; both clear min-N independently) = 161 parent-already-missed + 200 ablation-did-not-break + **193 length-confounded** + 45 + 0 unmeasured. The two-leg filter would have admitted **238**, of which 193 — **81 %** — are length artifacts.

**Scope limits, disclosed rather than buried.**
- The construct's architecture families are the **two** that cleared the CLAUDE.md §10.1 ≥2-independent-source bar (class-II platform swap; joint Stem II + IIA/B deletion). Its diversity is a **lower bound**, not a sample of the natural non-canonical space.
- Attribution is to the **excised element via a CM proxy**, *not* to ADR-0006 D9's relaxed-architecture detector **(b)**, which has no in-repo backend until P6-01/P6-11. The in-repo element-level `stem2_structure_only.cm` was tested and is **not** one (1/300 parents detected even at E ≤ 1000). The triple is **pre-registered for re-derivation against the real (b) detector at P6**, and the probe set is provisional until then.
- The **natural** Tier-2N arm is **N = 0 by construction** — the corpus is 100 % CM-derived and so cannot contain a CM-invisible locus, and no published non-canonical architecture is CM-invisible. Reported at zero, in the D6/D9 reported-not-gated spirit, never dropped from the accounting.
- **No published RF00230 false-negative rate exists** (§10.1 gate: `NO DIRECT EVIDENCE FOUND`; only the *procedural* fact that TBDB needed a second class-II CM, PMID:32882008). Every Tier-2N recall figure this project reports is therefore **first-party, measured in-repo**.

**Cross-reference impact:**
- **P2-07** — `src/tbox_finder/synth/tier2n.py` implements the triple and freezes the measured baselines in `MEASURED_CONFOUND_BASELINES`; `src/tbox_finder/eval/tier2n_probe.py` admits only eligible variants; `src/tbox_finder/infernal.py` is the single `cmsearch` caller. Unit tests pin each leg and were verified **by sabotage**.
- **P2-09** — round-0 mining reads the probe set through `build_probe_set`; a sub-min-N set is `ROUND_INADMISSIBLE`, never a silent pass.
- **P5** — the synthetic-Tier-2N spike-in recovery (§12 detection-power floor) inherits the triple and the provisional status.
- **P6** — re-derives eligibility against the real (b) architecture detector; if the two disagree materially, the P6 result supersedes and this amendment is re-signed.
- **ADR-0006 D9** — unchanged as a rule; this amendment supplies the operational proxy D9's row-5 routing needs before its (b) backend exists.
- No gate **value** moves; `MIN_REAL_HOMOLOG_N` is untouched.
