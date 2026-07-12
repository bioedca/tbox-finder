# ADR-0001 — Architecture & aims

- **Status:** Accepted (user sign-off recorded 2026-07-08, CLAUDE.md §7 item 2)
- **Date:** 2026-07-08
- **Deciders:** bioedca (project owner)
- **Phase:** P0 (seed ADR, authored first)
- **Delegated from:** PRD §1, §2 (esp. §2.3), §5, §6, §18.3
- **Supersedes / superseded by:** none
- **Related:** ADR-0002 (environment & ML stack), ADR-0003 (cluster & scan ops), ADR-0004 (split & leakage policy), ADR-0005 (non-circular eval design), ADR-0006 (validation decision rule & tiering)

This is the **contract ADR**. It is authored first so every later seed ADR and gate step inherits the locked precedence and terminal-outcome map. It pins **architecture, aims, claim & gate precedence, the precedence carve-out, the three distinct failure branches, the project-level rollup and its inconclusive terminal outcome, the intended-use posture, and release-license gating**. It **does not** pin threshold *values* — those are delegated to ADR-0002…0006 (§18.3) and, per the carve-out below, are authoritative there.

---

## Context

The entire known T-box catalogue was assembled with covariance-model (CM) homology search (Infernal `cmsearch` over Rfam RF00230 + a bespoke class-II CM); TBDB curates 23,535 sequences across 3,632 bacterial species [PMID:32882008; DOI:10.1093/nar/gkaa721]. A CM encodes one consensus secondary structure and loses sensitivity on divergent/remote homologs [DOI:10.1101/gr.5890907], so the apparent phylogenetic distribution is bounded by the seed alignment's cultured-Gram-positive sampling bias [DOI:10.1261/rna.819308; DOI:10.1093/nar/gkaa1047].

Because both training labels **and** the naïve ground truth are CM-derived, the project-defining risk is **circularity** (PRD §5): a model trained on CM-detectable labels and scored against CM-derived truth can at best re-learn what the CM already sees. Two facts therefore shape every decision that follows: (1) the project's value is **scientifically-defensible discovery**, and (2) the headline claim is a **generalization claim**. Leakage control, calibration, and orthogonal (model-independent) validation are first-class, not afterthoughts.

This ADR locks the invariants that the rest of the project — and every downstream gate decision — depends on.

---

## Decision

### D1. Two-stage architecture (locked)

`tbox-finder` is a **hybrid two-stage** system: a high-recall genome scanner feeding a high-precision RNA re-ranker (PRD §6).

- **Stage 1 — Caduceus-PS (DNA, single-nucleotide, RC-equivariant).** Tiles contigs into overlapping windows (**W = 1024 nt, stride 512**; both strands handled by RC-equivariance), emits **per-nucleotide segmentation logits over the normative 8 classes** `{background, Stem_I, Specifier, Stem_II, Stem_III, Antiterminator_Tbox_seq, Terminator, Discriminator}` (`Stem_II` subsumes the IIA/B pseudoknot extent, §8), runs at a **low (high-recall) threshold**, and merges contiguous element calls into a candidate locus ± flank. Overlapping-window logits are **averaged (log-sum-exp) then arg-maxed into one per-position prediction before locus construction** (seam-free operator; the value-level rule lives in ADR-0005). The default checkpoint is **Caduceus-PS `seqlen-131k_d_model-256_n_layer-16`** scanned at 1024/512; final size is a Phase-2 ablation.
- **Stage 2 — RiNALMo (RNA, 650M) precision re-ranker.** Ingests the resolved locus (± flank), transcribed T→U — **sequence only**; predicted/annotated structure is used solely as the §8/§11 auxiliary consistency **target**, never as a Stage-2 input. Heads: (a) calibrated binary T-box vs not, (b) boundary refinement, (c) regulatory-mode class, (d) AUX specifier codon / cognate amino acid / tRNA family.
- **Division of labor.** Stage 1 maximizes recall on divergent loci; Stage 2 is the calibrated precision filter that makes genome-scale FDR tractable. Reported metrics always distinguish **Stage-1-only**, **two-stage**, and **two-stage + orthogonal**. Stage-1 threshold, locus-construction rule, and Stage-2 operating point are **pre-registered as rules in ADR-0005** (values frozen at the phase gate).
- **Strand handoff.** Caduceus-PS is strand-agnostic; a **strand-resolver** orients each locus from predicted element order; ambiguous loci are flagged low-order-confidence and carried on both strands, with strand corroborated by model-independent signals (R-scape covariation, R2DT architecture, §13.3 synteny) — a mis-resolution degrades to a bounded false negative, never a false-novelty claim.
- **Two Stage-1 checkpoints exist** (full map pinned in PRD §10.1): the **production scanner** (P2-trained, graded GATE-4/GATE-2, runs GATE-1 arms a–c, drives the P5 scan/GATE-3, shipped) and the **class-II-CM-naive anti-mimicry ablation** (graded **only** by the GATE-1 class-II anti-mimicry sub-arm, scored Stage-1-only).
- **Ablations / fallbacks (locked as permitted).** Stage 2 may be ablated to (i) RNA-FM (cheaper) or (ii) a covariance-model + learned-calibration confirmer, if RiNALMo transfer underperforms (numeric triggers in ADR-0002/§10.2). A Stage-1 backbone fallback (NT-multispecies / GTDB continued-pretraining) is permitted only via branch B3 below.

### D2. Aims (locked)

- **Primary aim:** **expand the phylogenetic distribution of T-boxes** — recover divergent/structurally non-canonical T-boxes CMs miss, into under-sampled/sparsely-recorded lineages (Gram-negatives, candidate phyla, uncultured MAG lineages; Archaea as a high-novelty stretch).
- **Secondary aims:** (i) characterize specifier↔anticodon (tRNA) coupling and coevolution on the discovered set; (ii) catalogue novel structural/functional variants; (iii) deliver a reusable, openly-released method generalizable to other tRNA-sensing / structured cis-regulatory RNAs — the **cross-family demonstration is a stretch** proof-of-concept (§14.5); the **reusable method itself is committed**.
- **Goals G-A…G-E** (PRD §2.1) are adopted verbatim as the project's goal set: two-stage detector with calibrated per-nucleotide annotations (G-A); higher recall at matched precision than `cmsearch` on held-out clades + synthetic divergence (G-B); genome-scale discovery campaign with orthogonally-validated novel T-boxes (G-C); Stage-2 auxiliary multi-task head as a **training signal / regularizer** that is display-only ("model prediction (non-validated)") and never supplies evidence-path specifiers (G-D); open release (G-E).

### D3. Claim & gate precedence (locked)

- The **headline methodological result is the generalization claim (GATE-1)** — higher recall than `cmsearch` at matched precision on phylogenetically held-out clades and on synthetic divergence. GATE-1 is **both the central claim and the precondition** for the **flagship scientific application**, the distribution-expansion result (**GATE-3**).
- **GATE-1, GATE-2, GATE-4 are mandatory method gates** that decide whether the system is sound.
- **GATE-3 is split:** (i) a **mandatory method sub-gate** — **scan-plan execution** (the pre-registered minimum-viable scan ran to spec) + genome-scale FDR control + a per-clade multiplicity statement + the §13.3 rule applied faithfully — and (ii) a **non-blocking breadth floor** reported as-found.
- A rigorous, calibrated, leakage- and FDR-controlled scan that legitimately finds the distribution **bounded** is an **accepted, publishable "calibrated-negative PASS"** — licensed by the OOD-ECE/drift decision rule **and** a corpus-specific detection-power floor (§12), so a well-calibrated-but-blind model **cannot** earn a bounded-distribution claim. A calibrated-negative PASS is **not a failure**.
- **Threshold values are delegated** to the P0 seed ADRs (§18.3); empirical confirmation happens at the named phase gate. GATE-1/2/4 firm-looking numbers (+10 pp point / +5 pp CI-floor, ECE ≤ 0.05, FDR ≤ 10%, F1 ≥ 0.80) are **defaults**, read the same way as GATE-4's explicit "default ≥ 0.80" — subject to the carve-out in D4.

### D4. Precedence carve-out (written verbatim)

The following is reproduced **verbatim** from PRD §2.3 and is the one place a lower-ranked ADR overrides the PRD (CLAUDE.md precedence carve-out), for delegated threshold *values* only:

> **Precedence carve-out.** For any threshold the PRD delegates to a seed ADR, the **ADR-pinned value is authoritative** and the number stated here is the **default/intent** — this carve-out overrides the global PRD > ADR ordering for delegated threshold *values* only (a recalibration changing a number still requires ADR sign-off, CLAUDE.md §7 item 2). Accordingly the firm-looking GATE-1/GATE-2 numbers (+10 pp point / +5 pp CI-floor, ECE ≤ 0.05, FDR ≤ 10%) are themselves **defaults** subject to ADR recalibration, read the same way as GATE-4's explicit "default ≥ 0.80". Each delegated default carries a **documented P0 magnitude rationale** — a smallest-effect-of-interest / effect-size argument or a power calculation at the pinned decoy prevalence and per-bin min-N, authored *before* any P4 result — and is **blinded-frozen at P0**: it may not change after P4 unblinding, and any *pre*-P4 recalibration still needs ADR sign-off (CLAUDE.md §7 item 2).

**Binding consequence.** Every delegated threshold value is **authoritative in its owning ADR** (ADR-0004 for GATE-4; ADR-0005 for GATE-1/GATE-2; ADR-0006 for the GATE-3 breadth floor + rollup) and is **blinded-frozen at P0** — it may not change after P4 unblinding, and any pre-P4 recalibration requires ADR sign-off.

### D5. GATE-1-failure terminal deliverable + P4→P5 go/no-go (locked)

- Because GATE-1 is the precondition for the GATE-3 flagship, an **outright** GATE-1 failure (the fully-fallback'd two-stage system still does not beat `cmsearch`) removes **both** the headline method claim **and** the calibrated-negative fallback — a "bounded-distribution" claim is untrustworthy from a method not shown to beat `cmsearch`.
- **Pre-registered terminal deliverable:** a **calibrated, leakage-controlled neural segmenter that matches but does not exceed `cmsearch`**, contributed honestly as a parity/negative method result, with the **recall-vs-divergence curve** and **class-II-CM-naive recovery** (§5 mechanisms 2–3) as the scientific content.
- **P4→P5 go/no-go:** a pre-registered gate binds the expensive genome-scale scan to a GATE-1 pass. An outright GATE-1 failure **halts/de-scopes P5**; an underpowered GATE-1 triggers branch B2 (D6).
- **Swapped-backbone integrity:** if Stage-2 was swapped to RNA-FM (§10.2), its P3/P4 numbers are **re-evaluated against the absolute GATE-1 (+10 pp point / +5 pp CI-floor) / GATE-2 (in-distribution ECE ≤ 0.05) check and the P4→P5 go/no-go is re-run on the swapped backbone** before the scan.

### D6. The three distinct failure branches (locked, §2.3)

These are **three distinct branches, not one**; a step must route to the correct branch and never collapse them:

- **B1 — Outright GATE-1 failure.** The fully-fallback'd system does not beat `cmsearch`. → the **parity-segmenter terminal deliverable** of D5. Present the pre-registered fallback (gate name + threshold + observed value + recommended action) and stop-and-ask (CLAUDE.md §7 item 4 / §8.5); do **not** silently downgrade a tier or loosen a threshold.
- **B2 — Underpowered / inconclusive GATE-1.** Arms are **reported-not-gated for min-N reasons** rather than failing the bar. → **halt P5 and CLAUDE.md §7 stop-and-ask** — distinct from B1. Options carried are **re-powering** (source more positives / a stronger synthetic power budget) or the **method + calibration-coverage resource-paper** path (D7). **Not** the parity-segmenter.
- **B3 — Method-gate failure at GATE-4 (P2 segmentation) or GATE-2 (calibration/FDR).** A **capacity / machinery** failure, **not** a detection-parity conclusion. → a **separate pre-registered terminal outcome** (a sub-parity but honestly-contributed segmenter) that **first attempts the ADR-0002 NT-multispecies / GTDB continued-pretraining backbone fallbacks** (a capacity failure is *not* auto-routed to B1 and *is* permitted an alternative backbone) via a CLAUDE.md §7 stop-and-ask **before any terminal pivot**.

### D7. Project-level rollup + the discovery-predominantly-inconclusive terminal outcome (locked; rollup function pinned in ADR-0006)

- A calibrated-negative PASS is a **per-corpus** verdict. A corpus clearing neither the power+ECE floor nor min-N is a **"sensitivity-bounded / inconclusive negative"** (§12). Because the P0/P3 coverage simulation predicts *ex ante* that some high-novelty corpora (Archaea/DPANN, sub-min-N CPR) will be inconclusive-by-rule, a **project-level rollup function** (its exact form **pinned in ADR-0006**) maps the per-corpus verdict vector {positive / calibrated-negative PASS / inconclusive} to **one GATE-3 outcome**:
  - **positive** — if the breadth floor is met;
  - **project-level calibrated-negative PASS** — iff **every** breadth-relevant, min-N-clearing corpus is a per-corpus PASS;
  - **discovery-predominantly-inconclusive** — otherwise; its pre-registered deliverable is the **method + calibration-coverage resource paper** reporting the adjudicable fraction.
- The **discovery-predominantly-inconclusive** state is a **distinct, pre-registered terminal outcome — not a failure.**

### D8. Data-ethics / intended-use posture (locked, §2.2)

tbox-finder is a **basic-research tool for cataloguing a native bacterial/archaeal gene-regulatory RNA**. It has **no engineered-pathogen, gain-of-function, or dual-use design intent**. The released model/dataset cards carry an **intended-use + limitations statement** (calibration, splits, sensitivity bounds) so downstream users do not over-read predictions as confirmed biology. Any G-D-predicted field displayed in the atlas is labelled **"model prediction (non-validated)"** and never used as evidence-path (validation) input.

### D9. Release-license gating (locked, PRD header + §2.1 G-E)

- **Code: MIT** (PRD header, pinned). The repository is public from day 1 (CLAUDE.md §4). Committing the physical `LICENSE` file is an **administrative add of the already-decided MIT license** (not yet committed as of P0-02 — the copyright-holder legal name is the only open item; tracked as a separate step) — **not** an unresolved license choice.
- **Model weights: CC-BY-4.0** (PRD header, pinned).
- **Curated-dataset license: delegated to the P0 license-compatibility audit** — the most-restrictive license compatible with the upstream sources (TBDB / Rfam / NCBI / GTDB), **defaulting to releasing own-derived fields** (coordinates, labels, predictions, splits, calibration) **under CC-BY-4.0** plus a **checksummed fetch/reconstruct script** for any encumbered upstream source. The audit gates the dataset card (§18.1 P0; §18.2 upstream-license risk).
- No secrets, tokens, W&B keys, or large data blobs in the public repo.
- Release (HF Hub model + dataset cards, Zenodo DOI) is **release-time only** and gated on complete cards (intended use, training data, splits, eval, calibration, limitations, license).

### D10. Delegations to the sibling seed ADRs (no values pinned here)

Per §18.3, this ADR **delegates** and does not restate values: **ADR-0002** (environment/ML stack, sm_86 pins, RNA-FM re-clear, Caduceus transfer go/no-go + backbone fallbacks); **ADR-0003** (cluster/scan ops, corpus priority + compute ceiling); **ADR-0004** (structure-aware homology-clustered splits, overlap precedence, GATE-4 = min per-element per-nt-class F1 over the 3 core elements ≥ 0.80 default); **ADR-0005** (non-circular eval: `cmsearch` operating point, GATE-1 two-part bar + block resampling + macro-average + AND-of-powered-arms, GATE-2 FDR/ECE machinery, synthetic-arm generator-validity, class-II anti-mimicry sub-arm); **ADR-0006** (validation decision rule & tiering, the per-corpus→GATE-3 rollup function, model-independent homolog-set assembly, clade-level null).

---

## Consequences

- **Positive.** Every later ADR and gate step inherits a single locked precedence and terminal-outcome map; the three failure branches cannot be silently collapsed; "bounded distribution" is a first-class publishable outcome guarded against the well-calibrated-but-blind trap; intended-use and licensing posture are fixed before any release artifact.
- **Costs / constraints.** The two-stage architecture and the anti-mimicry (class-II-CM-naive) checkpoint impose a second Stage-1 training run and a Stage-1-only grading path. The blinded-freeze rule forbids post-P4 threshold tuning, trading late flexibility for pre-registration integrity.
- **This ADR pins no numbers.** All threshold values remain delegated to ADR-0002…0006 and authoritative there (D4).

## Cross-reference impact list

- **PRD:** §1 (thesis/aims), §2.1 (G-A…G-E), §2.2 (non-goals + intended-use), §2.3 (gates, precedence carve-out, three branches, rollup), §5 (circularity mechanisms 1–5), §6 (two-stage architecture), §18.3 (ADR index).
- **Sibling ADRs:** ADR-0002 (backbone fallbacks / RNA-FM re-clear feed B3 + D5), ADR-0003 (scan ops / compute ceiling feed GATE-3 scan-plan execution), ADR-0004 (splits + GATE-4 default), ADR-0005 (GATE-1/GATE-2 machinery + defaults + blinded freeze), ADR-0006 (GATE-3 breadth floor + rollup function + tiering).
- **imp.md:** P0-03 (this step); referenced by ADR-0002…0006 authoring steps and every downstream gate decision.
- **CLAUDE.md:** §2.3 precedence carve-out, §7 items 2/4 (ADR sign-off, gate-failure stop-and-ask), §8.5 (step-local gates), §10.3 (no fabricated metrics).
- **Cards / paper (release-bound):** model card (two-checkpoint split, per-checkpoint GATE attribution, CC-BY-4.0), dataset card (license per P0 audit), `paper/manuscript.qmd` (aims + terminal-outcome map).
- **Phase-0 exit (2026-07-12):** the two-stage, non-circular aims are consolidated (no decision changed) into `docs/dev-log/phase0_2026-07-12.pdf`, the `README.md` P0 headline, and the `paper/manuscript.qmd` §Non-circular evaluation design paragraph (GATE-1…GATE-4 framing + orthogonal-validation requirement).

## Sign-off

- **User sign-off:** ☑ recorded 2026-07-08 (bioedca), CLAUDE.md §7 item 2.
