# `pdb_element_extents/` — crystal-structure label-derivation fixtures (P0-21)

Ground-truth fixtures for `tests/unit/test_label_derivation.py`, the label-integrity
guard for every downstream training step (PRD §8; ADR-0004 D1/D6).

## Files

- **`pdb_element_extents.json`** — the 9 crystal/cryo-EM T-box entries, each with the
  per-element residue extents **transcribed verbatim** from the tboxevo crystal-derived
  files (`tboxevo/data/external/{cocrystal_ss_cons/anchor_<ID>.json, secondarystructure/<ID>/<ID>-2D.dbn}`).
  Coords are 1-based inclusive in the leader-strand local frame (the frame
  `src/tbox_finder/labels.py` paints in).
- **`synthetic_handchecked.json`** — the hand-checked synthetic fixture carrying the
  **exhaustive** precedence-overlap coverage (all three material overlaps + residual
  precedence pairs), with hand-authored expected label strings.

## Why the split (Option B, user sign-off 2026-07-10)

tboxevo records the **real crystal-derived secondary structure** (dot-bracket) plus a
Stem-I `ss_cons` / apical slice per entry (and a Stem-II slice for 6UFM) — but **no
file maps crystal residues to the Specifier codon / Antiterminator / Terminator /
Discriminator extents**, and **6 of the 9 depositions are Stem-I-only fragments**. So:

- the **9 real depositions** pin precedence correctness on the elements they physically
  resolve — Stem_I (5 entries), Stem_II (6UFM) — and validate class-II routing +
  naive-withholding on the 3 real Actinobacteria *ileS* class-II crystals;
- the **hand-checked synthetic fixture** carries the exhaustive all-overlap precedence
  coverage.

No functional-element boundary is invented (CLAUDE.md §10.3): un-resolved elements are
absent, not guessed.

## 9-PDB cross-source label-noise ceiling `C` (ADR-0004 D6, reported non-gated)

`C` governs whether the GATE-4 0.80 per-nt-F1 floor may be δ-recalibrated. **Finding:**
none of the 9 depositions resolves Specifier / Antiterminator / Terminator /
Discriminator to a residue extent (only Stem_I ×5 + Stem_II ×1), so a numeric per-nt
cross-source F1 ceiling over the full 8-class target is **not estimable from these
depositions alone** (and a within-fixture agreement would be circular). One cross-source
boundary-noise instance is observed (WCBNVLFN apical/Stem_I ±pad overhang, 1 nt).
**Conclusion:** `C` shows no sub-floor cap on the resolvable classes, so per ADR-0004 D6
the floor **stays 0.80** (the N≤9 crystal ceiling must not one-directionally lower the
bar). A rigorous numeric `C` would need the deferred literature-grounded per-element
annotation aligned to matched corpus records.

## Entry summary

| Entry | Organism / gene | Class | Scope | Resolved extent(s) |
|---|---|---|---|---|
| 4LCK | *O. iheyensis* glyQ | I | Stem-I only | Stem_I (1,102) |
| 4MGN | *G. kaustophilus* glyQS | I | Stem-I only | Stem_I (1,86) |
| 2KZL | *B. subtilis* tyrS | I | Stem-I/Specifier-domain (NMR) | Stem_I (1,55) |
| 6POM | *B. subtilis* glyQS | I | full-length | none (apical diagnostic only) |
| UCKOPSB | Thermoactinomycetaceae trpS | I | classical Stem-I leader | Stem_I (1,93) |
| WCBNVLFN | *D. putei* asp/serC | I | classical Stem-I leader | Stem_I (1,95) |
| 6UFG | *M. tuberculosis* ileS | II | ultrashort translational | none (prose only) |
| 6UFH | *M. tuberculosis* ileS | II | ultrashort translational | none (prose only) |
| 6UFM | *N. farcinica* ileS | II | ultrashort translational | Stem_II (30,77) |
