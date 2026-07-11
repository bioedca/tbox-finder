# ADR-0006 — Validation decision rule & tiering

- **Status:** **Accepted (user sign-off 2026-07-11; CLAUDE.md §7 item 2, P0-29 — "accept as drafted"; the D6/D7/D8 thresholds-frozen-at-P6 deferral endorsed).**
- **Date:** 2026-07-11
- **Deciders:** bioedca (project owner)
- **Phase:** P0 (seed ADR)
- **Delegated from:** PRD §13.3 (the confirmed-novel decision rule + tiering), §13.2 (orthogonal in-silico validation pipeline), §7.2 (assembly grades + union novelty prior + archaeal posture), §2.3 (project-level rollup), §12 (calibrated-negative-PASS conditions)
- **Supersedes / superseded by:** none
- **Related:** ADR-0001 (D7 — the three project-level GATE-3 outcome strings this ADR's rollup targets), ADR-0003 (scan-plan execution — the GATE-3 method sub-gate), ADR-0004 (D1 element precedence — reused by the (b) named-element predicate; the leakage-controlled split the held-out canonical predicate-freezing set is drawn from), ADR-0005 (D3 scan machinery, D7 decoy prevalence, D9 class-II anti-mimicry, D12 GATE-2 FDR, D13 OOD-ECE/drift → the per-corpus verdict this rollup consumes, D14 mining spare-rule → this ADR's discovery-side masking counterpart)

This ADR pins the project's **anti-circularity validation contract**: the pre-registered rule that decides whether a Stage-2 candidate is a **confirmed novel T-box**, the **tier** it earns, and how per-corpus verdicts **roll up to one GATE-3 outcome**. Its load-bearing property is **orthogonality**: a model-unique hit is confirmed **only by model-independent evidence** — R-scape covariation on a **CM-free de-novo alignment built neither from the model's own covariation nor by force-fitting the RF00230 CM** [DOI:10.1371/journal.pcbi.1011262], correct synteny, and (for canonical hits) specifier–tRNA consistency read from the sequence — **never** by "another CM agrees." The flagship class it exists to protect is **Tier-2N** (CM-invisible non-canonical architecture), which by construction fails every RF00230-templated predicate and would be discarded by any rule keyed to the CM.

Per the **PRD §2.3 precedence carve-out**, the delegated numeric defaults pinned here (the **500 bp** synteny window, the **E[false novel clades] < 0.5** bound) are authoritative over the PRD's stated intent; values that require empirical calibration on the held-out canonical set (the short-Stem-I nt threshold, the RF00230 alignability cutoff, the homolog-search thresholds) are pinned here **as rules with defaults frozen at the P6 phase gate**, following the ADR-0005 "rule now, value at the phase gate" precedent. Any change to a pinned value needs ADR-0006 re-sign-off (CLAUDE.md §7 item 2).

---

## Context

The corpus is ~90% Firmicutes and entirely CM-derived (23,535 TBDB records; §5). A validation rule that leans on RF00230 alignment, canonical architecture, or "documented" ultrashort/class-II status would re-encode exactly the ascertainment bias the project exists to overcome, and would silently reject the CM-invisible novelty (**Tier-2N**) that is the §1 sub-claim. Two forces therefore shape every decision below: **(i) each criterion must be estimable from model-independent evidence**, and **(ii) each relaxation must be keyed to an operational, unit-testable predicate — not to prior-literature documentation** — so a genuinely novel variant qualifies rather than being routed only to the experimental backlog.

**Scientific-evidence gate (CLAUDE.md §10.1) — two high-stakes facts, each ≥2 agreeing peer-reviewed sources, cleared 2026-07-11:**

- **The 500 bp downstream-gene window (D4) is a deliberately recall-favoring pad.** T-box transcriptional leaders and the riboswitch core are short: *B. subtilis proBA/proI* T-box leader transcripts are **269–270 nt** [PMID:21233158; DOI:10.1099/mic.0.047357-0 (accessed 2026-07-11)], and the T-box riboswitch **core is ~180 nt** (Stem I ~100 nt) [PMID:28621923; DOI:10.1021/acs.biochem.7b00284 (accessed 2026-07-11)], consistent with the TBDB leader-length distribution [DOI:10.1093/nar/gkaa721]. A 500 bp window is therefore ~1.5× the observed Firmicutes max (~335 bp) and ~2.8× the p99 (~180 bp) — generous by design. **Specificity is supplied by the gene-identity requirement, not the distance**; a tighter empirical percentile would inherit the Firmicutes bias. **No numeric cutoff is transcribed from any paper** (CLAUDE.md §10.3) — 500 bp is the project's own recall-favoring engineering choice, with the empirical p95/p99 ≈ 119/180 bp reported as a sensitivity check.

- **The archaeal-occurrence posture (D16) holds an archaeal T-box as not a-priori-impossible.** Riboswitches — including TPP — genuinely reach Archaea: TPP "occurs in plants, bacteria, fungi, and **archaea**" [PMID:36594112; DOI:10.1002/wrna.1774 (accessed 2026-07-11)], and TPP/FMN/Lys/guanidine/c-di-AMP riboswitches have been identified across Archaeal genomes [PMID:31020937; DOI:10.2174/1386207322666190425143301 (accessed 2026-07-11)]. So Archaea are scanned as a pre-registered **stretch** target through full §13 validation, and the archaeal absence in the CM catalogue (0/23,535 TBDB records) is treated as **ascertainment-limited detection history**, not solely a biological prior. A **residual biological possibility is held open** (not rebutted by the ascertainment argument): the T-box switch may not operate under archaeal factor-dependent termination / leaderless translation as it does under the bacterial intrinsic-terminator + SD-sequestration mechanisms — a **mechanistic-compatibility** question re-verified at PRD §20.
  - **Citation correction (flagged, not propagated).** PRD §7.2 lists **PMID:36008760** among the archaeal-occurrence sources; that paper (Ashniev et al. 2022, *BMC Genomics* [DOI:10.1186/s12864-022-08796-y]) is a **bacterial** T-box histidine-regulation study (Gram-positive Firmicutes/Actinobacteria) and does **not** support archaeal riboswitch occurrence. This ADR carries the two genuinely-archaeal sources above; PMID:36008760 is retained only as a **union-prior literature-occurrence** source (bacterial T-box distribution), not for the archaeal posture. Recommend correcting the PRD §7.2 citation list.

The verdict vocabulary this ADR routes into is already fixed upstream: the **per-corpus verdict vector** `{positive / calibrated-negative PASS / inconclusive}` (PRD §2.3; per-corpus PASS conditions in ADR-0005 D13), and the three **project-level GATE-3 outcomes** `positive` / `project-level calibrated-negative PASS` / `discovery-predominantly-inconclusive` (ADR-0001 D7; PRD §2.3). ADR-0005 delegated the **rollup function** and the **§13.3 rule** here (D18 delegation map); P0-27 pre-computed the ex-ante verdict-vector *shape* (`src/tbox_finder/coverage.py`). This ADR authors the rules over that vocabulary.

---

## Decision

### D1. The confirmed-novel decision rule — (a) ∧ (b) ∧ (c) core, (d) supporting (locked rule)

A Stage-2 candidate in a **novel** lineage (host GTDB-Tk placement in a clade with **no prior record under the union prior**; §7.2, D11) is **confirmed** iff criteria **(a) covariation**, **(b) architecture**, and **(c) synteny** all hold. **(d) specifier–tRNA consistency** is *supporting*: it distinguishes Tier-1 from Tier-2 but is never required for confirmation. **Tier-2N** is a pre-registered confirmed-novel class that **relaxes (b)** to `a ∧ c`. The four criteria (D2–D5), the short-Stem-I/class-II predicate (D6), and the tier routing (D9) are frozen as **unit-testable predicates** (specifications in the *Predicate specifications* section below; tests land at P6 on the held-out canonical set drawn from the ADR-0004 leakage-controlled split). The **per-class joint `a ∧ b ∧ c` false-pass rate** (including the class-II `a ∧ c` Tier-2N path) is estimated on the §9.1 5′UTR/tRNA-adjacent leader decoys + clade-matched random leaders and **reported as a property of the rule** (feeding the D12 multiplicity statement).

### D2. Criterion (a) — covariation, union anchored in Stem I (locked rule; default value)

Pass = **≥ 2 helix-level significant covarying pairs (R-scape E ≤ 0.05, helix-level aggregation) within the Stem-I + antiterminator region, with ≥ 1 in Stem I**, computed on the **CM-free de-novo MSA of the candidate's own model-independent homolog set** (D7) — never the model's own covariation, never force-fit to RF00230.

> **default (a) covariation: ≥2 helix-level pairs at E ≤ 0.05, ≥1 in Stem I** (union region = Stem I + antiterminator)

- **Ultrashort-Stem-I / class-II carve-out.** Where the **model-independent short-Stem-I/class-II predicate (D6)** holds, (a) may **instead** be met by **≥ 2 significant pairs in any conserved helix with ≥ 1 in the specifier-bearing region** (with a minimum-covariation-power caveat reported). This keeps a variant that passes relaxed-(b) from being silently rejected by an unrelaxed (a).
- **Antiterminator covariation is supporting, not required** — the antiterminator is a short (~14 nt), highly-conserved motif with low R-scape power; requiring it would reject genuine T-boxes and depress GATE-3 recall. Its architecture is enforced by (b) instead.
- **`(a) holds` ≝ canonical-(a) OR carve-out-(a).** Both the anchoring region and the E-threshold are fixed (not data-dependent), preserving a data-independent claim region.

### D3. Criterion (b) — architecture, operational named-element predicate (locked rule)

Pass = a **named-element-presence predicate**: the expected helices are present **and** an NCCA-pairing bulge is detected (R2DT 2.0 / `cmalign` on the host cross-check; §13.2). The predicate reuses the ADR-0004 D1 element vocabulary and is **frozen on the held-out canonical set and unit-tested at P6**. It is **never keyed to a `cmalign` bit-score vs RF00230** (which would re-impose the CM ceiling). Two **separate, individually-justified relaxations**:

1. **Ultrashort-Stem-I variants** (by the same **D6** model-independent short-Stem-I predicate — *not* prior-literature documentation) relax the **expected-helix-presence** requirement, because ultrashort Stem-I variants lean on Stem II/pseudoknot binding energy [PMID:29581302].
2. **Class-II variants** relax only the **detection *confidence* of the NCCA bulge**: the antiterminator–NCCA contact is **biologically present in class II** [PMID:25583497; PMID:39097611], but class-I-template R2DT/`cmalign` bias makes its canonical-geometry detection unreliable — so the bulge remains the target **where detectable** rather than being treated as absent.

R-scape (a) remains the model-independent structural anchor even under both (b) relaxations.

### D4. Criterion (c) — synteny, same-strand downstream gene within 500 bp + gene-identity (locked rule; **default value; blinded-frozen**)

Pass = the first downstream **same-strand** CDS start (strand from the §6 resolver) within **500 bp** of the T-box element 3′ end encodes an **aaRS / amino-acid-biosynthesis / transport / transamidation** function. The 3′ end is the **terminator-inclusive** boundary for transcriptional (class I) T-boxes; **class-II** T-boxes have no terminator and abut/overlap the start codon (distance ≈ 0, handled as a separate case).

> **default (c) synteny window: 500 bp** downstream, same strand — recall-favoring pad (~1.5× Firmicutes max 335 bp); **specificity from the gene-identity requirement, not the distance**. Empirical p95/p99 ≈ 119/180 bp reported as a sensitivity check. Blinded-frozen at P0; change → ADR-0006 re-sign-off.

- **Annotation robustness (MAG-critical).** The gene-identity test is **not** left to the primary product name. Where the first downstream ORF is annotated *hypothetical* or is **pseudogenized**, targeted **aaRS / amino-acid-biosynthesis Pfam/KO HMM profiles** are run on it; a **tandem / intervening-ORF carve-out** extends the window past a downstream leader or sub-threshold ORF (§7.1 tandem loci).
- **Symmetric diagnostics.** A **false-FAIL rate + per-clade (c)-exclusion + pseudogene diagnostic** is reported **symmetric with the false-pass rate**, so the annotation-driven recall cap in incompletely-annotated CPR/DPANN/MAG lineages is measured, not silent. The (c) — and joint `a ∧ b ∧ c` — **false-pass rate is estimated on *clade-matched* random leaders + the §9.1 decoys**, so it already encodes per-clade aaRS/biosynthesis gene density (feeds D12).

### D5. Criterion (d) — specifier–tRNA consistency, supporting + completeness-aware → indeterminate-d (locked rule)

(d) holds = the specifier codon matches its amino acid **AND** a cognate-anticodon tRNA is encoded in-genome. The specifier is **read from the sequence** (specifier loop via `cmalign`/R2DT — or the **CM-free de-novo MSA for sub-alignability-cutoff Tier-2N loci**, D8 — + genetic code + tRNAscan-SE), **never the Stage-2 G-D aux head**, preserving model-independence.

- **Completeness-aware.** An in-genome cognate-tRNA absence is a specifier–tRNA **mismatch only on an AQ/complete host**. On an **incomplete assembly** (an unassembled tRNA is a technical, not biological, absence) the hit is routed to the **Indeterminate-d** state (D9) — **not** demoted to a Tier-2 "reprogrammed specificity" claim (which would inflate the §14.3 orphan catalogue).
- The §14.1 specifier↔anticodon coupling analysis reads this **same sequence-derived specifier** and is **restricted to AQ/complete hosts**; Tier-2N specifier reads (localized from the de-novo MSA) are flagged **low-confidence**.

### D6. Model-independent short-Stem-I / class-II predicate (locked rule; **default value frozen at P6**)

A single predicate gates **both** the (a) and (b) carve-outs, so relaxations are consistent and not literature-gated:

- Pass = **Stem-I helix extent below a pinned nt threshold** (measured from the de-novo structure, model-independent) **OR** a **class-II regulatory-mode signal** (translational mode: no terminator hairpin + SD-sequestration architecture; §3).
- Keyed to **structure/mode, *not* prior-literature "documentation"** — a genuinely novel ultrashort variant qualifies rather than being routed only to the §14.4 backlog.

> **default short-Stem-I nt threshold: frozen at P6** on the held-out canonical Stem-I extent distribution (the value that admits characterized ultrashort variants while excluding the canonical ~100 nt Stem-I core; no number is fabricated at P0 — §10.3). Rule pinned now; value at the phase gate (ADR-0005 precedent).

A **Stem-I exclusion diagnostic** (the GATE-3 exclusion diagnostic) reports the fraction of model-unique non-canonical candidates excluded **solely** by the Stem-I covariation requirement, so residual structural bias is quantified.

### D7. Model-independent homolog-set assembly + CM-free de-novo alignment (locked rule; **default thresholds frozen at P6**)

The homolog set populating each candidate's R-scape MSA is assembled by a **model-independent sequence-homology search — nhmmer / BLAST / cmsearch-from-candidate — not by model-grouped membership**, so the sole model-independent anchor is verifiably orthogonal at the step that most matters. The alignment is built by a **CM-free de-novo structure-aware aligner** (e.g. LocARNA, or a covariance model built *from the candidates*), **never from RF00230 and never from the model's covariation** (anti-circularity, §5 mechanism 5).

> **default homolog-search inclusion thresholds: frozen at P6** — the nhmmer / BLAST / cmsearch-from-candidate E-value / bit-score cutoffs that maximize divergent-homolog recall while holding the alignment's covariation power; pinned as a rule now, value at the phase gate. The **divergence/recall ceiling on the most-divergent homologs is documented** [DOI:10.1101/gr.5890907], since sequence-homology search itself collapses on divergent ncRNA.

### D8. RF00230 alignability cutoff + canonical-holdout invariance check (locked rule; **default value frozen at P6**)

For any candidate below a **pinned per-candidate RF00230 alignability (bit-score) cutoff** — notably the flagship CM-invisible **Tier-2N** loci — the `cmalign`-vs-RF00230 localization path is **abandoned entirely** for the D7 de-novo route (specifier localization, D5, then uses the de-novo MSA and is flagged low-confidence). A **canonical-holdout invariance check** (R-scape calls invariant to alignment source on held-out canonical T-boxes) is reported as a **GATE-3 method field** *before* de-novo covariation is trusted on novel loci.

> **default RF00230 alignability bit-score cutoff: frozen at P6** on the held-out canonical alignability distribution (the bit-score below which `cmalign`-vs-RF00230 Stem-I coverage degrades); rule now, value at the phase gate.

### D9. Tier set — complete + mutually exclusive routing (locked rule)

Confirmation status is decided by the truth of `(a, b, c, d)` (with (a) per D2's union, (b) per D3's relaxations), the host completeness (D5), and — for the residual case — whether R-scape-significant covariation lies **outside** Stem I. The routing is **exhaustive and mutually exclusive** (proof: the leaves below partition on `(c) ∈ {T,F}` → `(a,b) ∈ {TT,TF,FT,FF}` → `(d) ∈ {pass, mismatch@AQ, indeterminate@MQ}` and the covariation-locus sub-case; every candidate lands in exactly one leaf):

| # | (c) | (a) | (b) | (d) / residual | → Route | Confirmed? | Catalogue |
|---|---|---|---|---|---|---|---|
| 1 | ✗ | — | — | — | §14.4 backlog / segregated atlas layer | no | no |
| 2 | ✓ | ✓ | ✓ | (d) pass | **Tier-1** (high-confidence canonical) | **yes** | yes |
| 3 | ✓ | ✓ | ✓ | (d) mismatch **on AQ/complete host** | **Tier-2** (non-canonical specificity) | **yes** | yes |
| 4 | ✓ | ✓ | ✓ | (d) **indeterminate** (unassembled tRNA, incomplete host) | **Indeterminate-d** (MQ-only) | **yes** | yes |
| 5 | ✓ | ✓ | ✗ | — | **Tier-2N** (non-canonical architecture — flagship CM-invisible) | **yes** | yes |
| 6 | ✓ | ✗ | ✓ | low covariation power | **Tier-3** (covariation-inconclusive) | no | no (segregated layer) |
| 7 | ✓ | ✗ | ✗ | R-scape-significant covariation **outside** Stem I | **Unclassified** (structurally-plausible, Stem-I-covariation-inconclusive) | no | no (§14.4 backlog) |
| 8 | ✓ | ✗ | ✗ | no significant covariation | unconfirmed candidate | no | no (backlog) |

- **Tier-2N** (row 5) is a **confirmed-novel** class: it enters the confirmed catalogue, counts toward GATE-3 breadth, **and** is additionally flagged flagship-novel for §14.4 experimental follow-up (D14).
- **Indeterminate-d** (row 4) is a+b+c-confirmed and floor-eligible (breadth-eligible), necessarily **MQ-only** (it arises only on incomplete assemblies), with (d) labelled **"specificity-indeterminate"** — never a Tier-2 reprogrammed-specificity claim, and excluded from §14.1.
- **Tier-3** (row 6) and **Unclassified** (row 7) are **never** folded into the confirmed catalogue; they appear only as a segregated atlas layer + a labelled "candidate, not confirmed" table.

### D10. Strand corroboration via model-independent (a)/(b) (locked rule)

The §6-resolver strand is **corroborated by strand-specific model-independent signals**: R-scape covariation (a) and R2DT architecture (b) pass **only on the correct strand**, in addition to (c) synteny (same-strand downstream gene). A **strand-robustness diagnostic** (fraction of confirmed loci tier-invariant to strand re-resolution) is reported. A mis-resolution therefore degrades to a **bounded false negative on divergent loci, never a false novelty claim**.

### D11. Negative/decoy masking keyed to the full union prior + the spare rule (locked rule; **phase-conditioned**) — consumed by P0-30

The discovery-side counterpart of ADR-0005 D14. All known T-box loci are **masked from every negative/decoy pool** using the **full union prior** (`data/processed/priors/union_prior.parquet`; the masking denominator = `accession` + `locus_start`/`locus_end` where present — TBDB and RF00230-only loci carry locus coordinates; literature clades carry none) **+ the run's own training positives + a flank**, matching the D12 clade-level null's masking reference. The **residual contamination rate against that union denominator is reported**.

Because a CM-missed **Tier-2N** T-box is by definition unknown to masking **and** fails any canonical-architecture predicate, the mining-exclusion rule is **not keyed to canonical architecture** but to a **spare rule** — a candidate is excluded from the hard-negative-mining pool if it passes:

> **spare rule (excludes from mining):** relaxed-architecture detection **OR** any-helix R-scape covariation **OR** downstream-aaRS synteny — the three **model-independent** disjuncts (sufficient for the P2 Stage-1 mining loop where no Stage-2 yet exists) — **OR**, at the **P3 re-mining round once Stage-2 exists**, a high Stage-2 posterior.

The 5′UTR/tRNA-adjacent leader pool (the hardest, most-useful hard negatives) is **retained**. A **Tier-2N probe set** (non-canonical + synthetic-Tier-2N positives) is evaluated each mining round; a per-round **recall drop halts/rolls back** the iteration (ADR-0005 D14). Worst case = **directionally-bounded discovery sensitivity on Tier-2N, not an invalid generalization claim**.

### D12. Per-clade multiplicity bound + real-gene-preserving clade-level null (locked rule; **default value; blinded-frozen**)

The per-candidate `a ∧ b ∧ c` false-pass rate (D4, clade-matched → already carries per-clade gene density) is lifted to a **multiplicity-aware per-clade statement** across all scanned clades:

> **default multiplicity bound: E[false novel clades] < 0.5**, reported as a GATE-3 breadth-floor field. Blinded-frozen at P0; change → ADR-0006 re-sign-off.

**Clade-level empirical null — two complementary controls:**
- **Primary — real-gene-preserving null.** Real genomes with **all union-prior T-box loci masked** (D11), or gene-label-permuted genomes, run through the **identical `a ∧ b ∧ c` pipeline**, so the **synteny channel (c) is actually exercised**; count clades clearing the breadth floor by chance.
- **Secondary — reversed/shuffled-genome run.** A **labelled (a)/(b)-only control** validating the structural channel only: reversing/shuffling destroys ORFs, so (c) cannot fire, and its near-zero pass rate **must not** be read as whole-rule reassurance.

### D13. Tier-2N per-phylum establishment (locked rule)

Per-hit **floor-eligibility** (D14) is separated from **per-phylum establishment**. A CM-invisible **Tier-2N-only phylum** is **creditable toward breadth**, with the D12 `E[false novel clades] < 0.5` bound as the spurious-phylum guard. **Per-phylum establishment** (a phylum counted toward the §2.3 breadth floor of "≥3 previously-unrecorded phyla, each ≥2 independent genera + ≥1 validation-Tier-{1,2,2N} or Indeterminate-d") requires the per-hit eligibility (D14) satisfied by **≥2 independent genera** within that phylum. This lets a flagship Tier-2N phylum count for breadth while the multiplicity bound guards against a spurious "novel phylum."

### D14. Assembly-grade × validation-tier eligibility matrix + backlog prioritization (locked rule)

Validation tiers (1/2/2N/3, Indeterminate-d) are **orthogonal** to the §7.2 assembly grades (AQ/MQ/LQ). Headline **breadth-floor eligibility**:

| | Tier-1/2/2N or Indeterminate-d (validated) | Tier-3 (inconclusive) |
|---|---|---|
| **AQ** (isolate genomes only) | floor-eligible (single assembly) | §14.4 backlog |
| **MQ** (all floor-passing MAGs) | floor-eligible **iff ≥2 independent assemblies** (flagged MAG-derived) | §14.4 backlog |
| **LQ** (below floor / contig-break) | corroborating only | excluded |

- **Indeterminate-d is MQ-only** — an AQ/complete host with an absent cognate tRNA is a genuine mismatch → Tier-2, not indeterminate (D5).
- **§14.4 backlog prioritization score** (from already-computed quantities): `Stage-2 posterior × novelty rank × synteny strength × in-genome cognate-tRNA presence × homolog count`, with a **forwarding cap logged**. Tier-2N enters the confirmed catalogue **and** is additionally forwarded flagship-novel; Tier-3 + Unclassified appear only in the segregated "candidate, not confirmed" table.

### D15. Per-corpus → GATE-3 project-level rollup function (locked rule) — consumes the P0-27 verdict-vector shape

Each scanned corpus yields a **per-corpus verdict** in the vocabulary `{positive / calibrated-negative PASS / inconclusive}` (ADR-0005 D13): a corpus is `calibrated-negative PASS` **only if** (i) its nearest-relative leave-clade-out OOD-ECE (with CIs) meets the pinned drift bound, (ii) it clears **`OOD_ECE_MIN_N = 20`** (`src/tbox_finder/coverage.py::OOD_ECE_MIN_N`; ADR-0005 Amendment A2), **and** (iii) it clears the corpus-specific **detection-power floor** (extrapolated recall@matched-precision + synthetic-Tier-2N spike-in recovery; §12); else `inconclusive` (sensitivity-bounded / inconclusive negative *by rule*, which also covers sub-min-N / zero-positive corpora). A corpus is `positive` if it contributes ≥1 confirmed floor-eligible hit (D9/D14).

The **rollup function** maps the per-corpus verdict vector to **one** of the three ADR-0001 D7 / PRD §2.3 project-level GATE-3 outcomes:

> **rollup (frozen, unit-testable — spec in *Predicate specifications*):**
> 1. **`positive`** — iff the §2.3 breadth floor is met (Tier-1 + Tier-2 + Tier-2N + Indeterminate-d hits spanning ≥3 previously-unrecorded phyla, per D13/D14) **and** the GATE-3 method sub-gate passes (ADR-0003 scan-plan execution + ADR-0005 D12 FDR + the D12 `E[false novel clades] < 0.5` statement).
> 2. **`project-level calibrated-negative PASS`** — else, **iff *every* breadth-relevant, min-N-clearing corpus is a per-corpus `calibrated-negative PASS`** (no corpus is `inconclusive` among the breadth-relevant min-N-clearing set).
> 3. **`discovery-predominantly-inconclusive`** — otherwise. A **distinct, pre-registered terminal outcome — not a failure**; its deliverable is the **method + calibration-coverage resource paper** reporting the **adjudicable fraction** (the §2.3 header-venue line).

**Consumed shape.** The rollup ingests the ex-ante verdict-vector shape produced by `coverage.py::verdict_vector_shape(...)` (four-key dict: `adjudicable` / `sub_min_n_inconclusive` / `zero_positive_inconclusive_named_clades` / `inconclusive_by_rule_total`). `coverage.py::predicted_gate3_modal_shape(...)` is the **ex-ante binary predictor** (`discovery-predominantly-inconclusive` | `adjudicable-predominant`) — **not** this three-valued rollup; the P0-27 sim already predicts `discovery-predominantly-inconclusive` as the modal shape (footprint = 7 bacterial / 0 archaeal phyla), so outcome 3 is genuinely pre-registered ex ante, not a surprise (§2.3).

### D16. Archaeal / near-zero-prior corpus validation posture (locked rule)

Archaea (incl. DPANN) and other near-zero-prior corpora are scanned as pre-registered **stretch** targets: any hit is gated through the **full §13 / D1–D15 validation** before any claim, using the corpus's **own deployment prior + per-corpus FDR null** (not the Bacteria-derived prior; §7.2). A near-zero-prior corpus earns `calibrated-negative PASS` only under the D15 (i)–(iii) conditions; a sub-min-N / zero-positive corpus (no computable leave-clade-out ECE — e.g. Archaea/DPANN) is `inconclusive` **unless** stated nearest-relative extrapolations (ECE-vs-distance **and** recall/power-vs-distance regressions, both with prediction intervals) clear **both** the drift bound **and** the detection-power floor. The archaeal absence is scoped to what the OOD evidence supports (**ascertainment-limited detection history**, per the Context evidence gate), and the **residual mechanistic-compatibility possibility is held open** and re-verified at §20 — never asserted as a confirmed biological negative.

### D17. Delegations — values frozen at the phase gate / consumed elsewhere (locked map)

- **Short-Stem-I nt threshold** (D6) — **frozen at P6** on the held-out canonical Stem-I extent distribution.
- **RF00230 alignability bit-score cutoff** (D8) — **frozen at P6** on the held-out canonical alignability distribution.
- **Homolog-search inclusion thresholds** (D7, nhmmer/BLAST/cmsearch-from-candidate) — **frozen at P6**.
- **The (a)/(b)/(c)/(d) + D6 short-Stem-I + D9 routing + D15 rollup predicates** — **unit tests land at P6** on the held-out canonical set.
- **Masking spare rule** (D11) — **consumed by P0-30** (§9.1 static decoy/negative construction + union-prior loci-masking + residual-contamination report).
- **Rollup shape** (D15) — **consumes** `coverage.py::verdict_vector_shape` / `OOD_ECE_MIN_N` / `predicted_gate3_modal_shape` (P0-27).

---

## Predicate specifications (frozen; unit tests land at P6)

These are the frozen, unit-testable rules the P6 test suite (`tests/unit/test_validation_rule.py`, `tests/ml/test_gate3_rollup.py`) will implement. Pure-logic signatures over model-independent inputs; the P6-frozen numeric thresholds enter as parameters (defaults per D6/D7/D8).

```
short_stem_i_or_class_ii(stem_i_extent_nt, regulatory_mode, *, stem_i_nt_threshold) -> bool
    # D6. True iff stem_i_extent_nt < stem_i_nt_threshold OR regulatory_mode == "translational".

criterion_a(covarying_pairs, *, stem_i_nt_threshold, class_ii_carveout: bool) -> bool
    # D2. covarying_pairs: list of (helix, region, E). Significant = E <= 0.05.
    # canonical: >=2 significant in {StemI, antiterminator} with >=1 in StemI.
    # carve-out (iff class_ii_carveout): >=2 significant in any conserved helix with >=1 in specifier-bearing region.
    # returns canonical OR carve-out.

criterion_b(named_elements_present: bool, ncca_bulge_detected: bool,
            *, ultrashort_relax: bool, class_ii_relax: bool) -> bool
    # D3. base: named_elements_present AND ncca_bulge_detected.
    # ultrashort_relax: drop the expected-helix-presence requirement.
    # class_ii_relax: NCCA bulge required only where detectable (confidence-relaxed).
    # never a function of a cmalign-vs-RF00230 bit-score.

criterion_c(downstream_gene_fn, downstream_gene_distance_bp, strand_same: bool,
            *, window_bp=500) -> bool
    # D4. True iff strand_same AND distance <= window_bp AND
    #      downstream_gene_fn in {aaRS, aa-biosynthesis, transport, transamidation}
    #      (product name OR Pfam/KO HMM hit on hypothetical/pseudogene; tandem carve-out extends window).
    # class II: distance ≈ 0 (abuts/overlaps start codon).

criterion_d(specifier_codon, cognate_aa, trna_encoded: bool, host_grade) -> {"pass","mismatch","indeterminate"}
    # D5. specifier read from sequence, not the G-D head.
    # pass: codon->cognate_aa AND trna_encoded.
    # tRNA absent on AQ/complete host  -> "mismatch".
    # tRNA absent on incomplete (MQ) host -> "indeterminate".

route_tier(a: bool, b: bool, c: bool, d: {"pass","mismatch","indeterminate"},
           covariation_outside_stem_i: bool, host_grade)
    -> {"Tier-1","Tier-2","Tier-2N","Tier-3","Indeterminate-d","Unclassified","unconfirmed"}
    # D9. Implements the routing table (exhaustive + mutually exclusive). Enforces Indeterminate-d MQ-only.

per_corpus_verdict(drift_ok: bool, min_n_ok: bool, power_floor_ok: bool, has_confirmed_hit: bool)
    -> {"positive","calibrated-negative PASS","inconclusive"}
    # D15/ADR-0005 D13. positive iff has_confirmed_hit;
    # else "calibrated-negative PASS" iff (drift_ok AND min_n_ok AND power_floor_ok); else "inconclusive".

gate3_rollup(per_corpus_verdicts: dict[str, verdict], breadth_floor_met: bool,
             method_subgate_pass: bool, breadth_relevant_min_n_corpora: set[str])
    -> {"positive","project-level calibrated-negative PASS","discovery-predominantly-inconclusive"}
    # D15. positive iff (breadth_floor_met AND method_subgate_pass);
    # else "project-level calibrated-negative PASS" iff every corpus in breadth_relevant_min_n_corpora
    #      has verdict == "calibrated-negative PASS";
    # else "discovery-predominantly-inconclusive".
```

---

## Consequences

- **The claim is orthogonal at the step that matters.** Confirmation rests on model-independent covariation (D2/D7), architecture (D3), and synteny (D4) — never on "another CM agrees" (ADR-0005 D10 PU-framing). The de-novo route (D8) means the flagship Tier-2N loci are never force-fit to RF00230.
- **The flagship novelty survives mining and validation.** The D11 spare rule keeps aggressive hard-negative mining from training the scanner to reject Tier-2N; D9/D13 give Tier-2N a confirmed-catalogue seat and per-phylum breadth credit; the D12 multiplicity bound is the spurious-phylum guard.
- **The negative is honest and bounded.** D15's three-valued rollup makes `discovery-predominantly-inconclusive` a pre-registered terminal outcome (predicted ex ante by P0-27), so a budget- or coverage-limited scan cannot masquerade as a calibrated-negative — and a calibrated-negative PASS still requires the §12 detection-power floor, not calibration alone.
- **Recall-favoring where bias hides novelty; specific where it counts.** The 500 bp window (D4), the structure/mode-keyed carve-outs (D6), and the supporting-not-required (a)-antiterminator and (d) all lean toward recall; specificity comes from gene-identity (D4), the E ≤ 0.05 covariation anchor (D2), and the clade-level nulls (D12).
- **Every P0-frozen value is either evidence-gated or deferred honestly.** 500 bp and `E[false novel clades] < 0.5` are pinned now; the three data-calibrated thresholds are pinned as rules with values frozen at P6 — no number is fabricated at P0 (§10.3).

## Related documents

- **PRD.md** §13.3 (decision rule + tiering), §13.2 (orthogonal pipeline), §7.2 (assembly grades + union prior + archaeal posture), §2.3 (rollup + breadth floor), §12 (calibrated-negative-PASS conditions), §20 (mechanistic-compatibility re-verification).
- **ADR-0001** D7 (project-level GATE-3 outcome strings), **ADR-0003** (scan-plan execution = GATE-3 method sub-gate), **ADR-0004** (D1 element precedence; leakage-controlled split), **ADR-0005** (D3/D7/D9/D12/D13/D14 — the eval contract this validation ADR consumes).
- **Amended by:** *(none yet — thresholds in D6/D7/D8 are frozen at P6, not by an ADR amendment)*.

## Cross-reference impact list

- **P0-30** consumes **D11** (masking spare rule + union-prior denominator) for the §9.1 static decoy/negative construction + residual-contamination report.
- **P6** consumes **D1–D9** (implements the frozen predicates as `tests/unit/test_validation_rule.py`), **D15** (`tests/ml/test_gate3_rollup.py`), and **freezes** the D6/D7/D8 numeric thresholds on the held-out canonical set.
- **P5** consumes **D12** (the multiplicity statement + clade-level nulls feed the GATE-3 method sub-gate alongside ADR-0005 D12 FDR).
- **P0-27** (`src/tbox_finder/coverage.py`) is consumed by **D15** (verdict-vector shape / `OOD_ECE_MIN_N` / modal-shape predictor).

## Sign-off

**Accepted — user sign-off 2026-07-11 (CLAUDE.md §7 item 2, P0-29), "accept as drafted"** — including endorsement of the D6/D7/D8 data-calibrated thresholds pinned as rules with numeric values frozen at the P6 phase gate (no P0 fabrication, §10.3). The PRD §7.2 archaeal-citation correction below was applied to PRD.md in the same session.

**Scientific-evidence gate (CLAUDE.md §10.1) — CLEARED, both high-stakes items ≥2 agreeing peer-reviewed sources (accessed 2026-07-11):**
- **500 bp downstream-gene window** — T-box leaders/core are ~180–290 nt: Brill et al. 2011 [PMID:21233158; DOI:10.1099/mic.0.047357-0] + Fang et al. 2017 [PMID:28621923; DOI:10.1021/acs.biochem.7b00284] + TBDB [DOI:10.1093/nar/gkaa721]. No numeric cutoff transcribed from any paper (§10.3).
- **Archaeal-occurrence posture** — riboswitches (incl. TPP) reach Archaea: Wakchaure & Ganguly 2023 [PMID:36594112; DOI:10.1002/wrna.1774] + Gupta & Swati 2019 [PMID:31020937; DOI:10.2174/1386207322666190425143301]. **Flagged:** PRD §7.2's PMID:36008760 is a bacterial (not archaeal) T-box paper — carried here only as a union-prior source; recommend the PRD citation be corrected.

**Blinded-frozen numeric defaults (P0):** (c) synteny window **500 bp** (D4); multiplicity bound **E[false novel clades] < 0.5** (D12). **Frozen-at-P6 rules (values not fabricated at P0):** short-Stem-I nt threshold (D6), RF00230 alignability bit-score cutoff (D8), homolog-search inclusion thresholds (D7).
