# Dataset-license-compatibility audit (P0-18)

**Scope.** Per-source license audit for the **curated dataset** tbox-finder releases at
preprint (HF Hub dataset card + Zenodo DOI). Gates the dataset card and the P7 release
(PRD header; §2.1 G-E; §16/§17; ADR-0001 §D9 *Release-license gating*).

**Fixed, not audited here** (PRD header / ADR-0001 §D9): **Code = MIT**; **model weights =
CC-BY-4.0**. This audit resolves only the **curated-dataset** license — "the most-restrictive
license compatible with the upstream sources, defaulting to release of own-derived fields
under CC-BY-4.0 plus a checksummed fetch/reconstruct script for any encumbered upstream
source."

**Machine-readable companion.** `scripts/reconstruct_encumbered.py` holds the same registry
as `SOURCES` (single source of truth) and the checksummed fetch/reconstruct stubs for the
encumbered sources. `python scripts/reconstruct_encumbered.py --list` prints the verdict
table; `--json` dumps the registry; `--reconstruct <source_id>` builds a (plan-only) fetch
plan from the committed provenance pins.

All upstream licenses were **verified against the authoritative upstream on 2026-07-10**
(GitHub repo/About sidebar for the MIT sources; wwPDB usage policy for PDB; EMBL-EBI for
Rfam; the GTDB downloads page; NCBI data-usage policies), or carried from the source's own
committed `data/external/**/provenance.json`.

---

## Per-source verdict table

Verdict key — **compatible**: permissive or public-domain, safe to redistribute in a
CC-BY-4.0 dataset (with attribution/notice); **encumbered**: share-alike or proprietary,
**not embedded** in the release → materialized on the user's machine by
`scripts/reconstruct_encumbered.py`.

| `source_id` | Source / role | Upstream license (SPDX) | In release? | Verdict | Route |
|---|---|---|---|---|---|
| `tbdb_master` | TBDB `Master_tboxes.csv` — primary training corpus | MIT | derived fields | **compatible** | redistribute-with-attribution |
| `rfam_rf00230` | Rfam RF00230 class-I CM / baseline | CC0-1.0 | yes | **compatible** | redistribute-with-attribution |
| `tbox_scan_cm` | tbox-scan `TBDB001.cm` class-II CM | MIT | yes | **compatible** | redistribute-with-attribution |
| `tboxevo_cms` | tboxevo sub-element CMs (`idtm_*`, `stem2_*`) | MIT | yes | **compatible** | redistribute-with-attribution |
| `pdb_crystals` | 9 PDB crystals (2KZL/4LCK/4MGN/6POM/6UFG/…) → label fixtures | CC0-1.0 | yes | **compatible** | redistribute-with-attribution |
| `vitreschak_supplement` | Vitreschak 2008 (RNA) supplement — anchor localizer | proprietary (© RNA Society/CSHL) | **no** | **encumbered** | reconstruct-from-source |
| `gtdb` | GTDB R232 taxonomy → novelty-prior projection + P6 placement | CC-BY-SA-4.0 | **no** | **encumbered** | reconstruct-from-source |
| `ncbi_refseq` | NCBI RefSeq genomes — scanned corpus + anchor-leader source | public domain | derived fields | **compatible** | redistribute-with-attribution |
| `ncbi_taxonomy` | NCBI Taxonomy taxdump — TaxId lineage re-placement (P0-15) | public domain | derived fields | **compatible** | redistribute-with-attribution |
| `anchor_p0_16` | GATE-1 independent anchor (P0-16; own-derived from RefSeq) | CC-BY-4.0 (own) | yes | **compatible** | redistribute-with-attribution |
| `classII_p0_17` | Additional class-II positives (P0-17; own, VERIFIED-EMPTY) | CC-BY-4.0 (own) | yes | **compatible** | redistribute-with-attribution |

**Two encumbered sources → reconstruct:** `gtdb` (CC-BY-SA-4.0 share-alike) and
`vitreschak_supplement` (proprietary). Everything else is permissive (MIT) or public domain
(CC0 / US-Gov), and the P0-16/P0-17 sets are this project's own work.

### Why the two encumbered sources still do not block the CC-BY-4.0 release

- **`gtdb` (CC-BY-SA-4.0).** Share-alike would force any dataset *embedding* the GTDB
  compilation to CC-BY-SA-4.0 — incompatible with the permissive CC-BY-4.0 default and with
  the CC-BY-4.0 model weights. Resolution: the release **does not embed** the GTDB name
  compilation. GTDB is used only for the union novelty prior (P0-14) and P6 placement; the
  released dataset ships own-derived **has-prior / no-prior verdicts as booleans (facts, not
  a copyrightable compilation)**, and the GTDB-**projected** columns are reconstructed by
  `reconstruct_gtdb()` (re-fetch pinned R232 tables → re-apply `src/tbox_finder/priors.py`).
  Note the corpus lineage-by-rank columns are **NCBI-named** (re-derived from the public-domain
  NCBI taxdump, P0-15), **not** GTDB — so the split-assignment table (P0-23) carries no GTDB
  content.
- **`vitreschak_supplement` (proprietary).** The copyrighted supplement is **never
  redistributed**. The released GATE-1 anchor leaders are re-derived from **NCBI RefSeq
  (public domain)** via `src/tbox_finder/anchors.py::source_anchor` (P0-16), so nothing
  encumbered ships. `reconstruct_vitreschak()` re-fetches the supplement (checksummed) only
  to reproduce the re-derivation.

---

## Resolved curated-dataset license

> **Q-question (release-scope; CLAUDE.md §7 item 1) — RESOLVED, user sign-off 2026-07-10:
> Option A.** The audit found **no source that blocks the CC-BY-4.0 own-derived default**.
> The alternative (Option B) is retained below for the record.

**Resolved — Option A (ADR-0001 §D9 default): `CC-BY-4.0`, own-derived fields + reconstruct
script for the two encumbered sources.**
The curated dataset is released under **CC-BY-4.0** covering this project's **own-derived
fields** (window/element coordinates, per-nucleotide labels, model predictions, split
assignments, calibration tables, the RefSeq-re-derived anchor, hash-links). Redistributed
permissive-upstream content (TBDB/tbox-scan/tboxevo **MIT**, Rfam/PDB **CC0**, NCBI
**public domain**) travels with its attribution/notice. The two **encumbered** sources
(`gtdb`, `vitreschak_supplement`) are **not embedded**; `scripts/reconstruct_encumbered.py`
materializes their derived columns from source on the user's machine. Consistent with the
CC-BY-4.0 model weights and ADR-0001 §D9.

*Trade-off.* Maximally permissive and reusable (standard for an ML dataset); the only cost is
that a user reproducing the GTDB-projected novelty-prior columns runs one checksummed fetch
step (`--reconstruct gtdb`).

**Alternative — Option B: embed the GTDB-projected columns and release the whole dataset under
`CC-BY-SA-4.0`** (the most-restrictive *compatible* license). Simpler single-file release, no
reconstruct step — but share-alike "infects" the whole dataset (downstream derivatives must
also be CC-BY-SA-4.0), reduces reuse, and is inconsistent with the CC-BY-4.0 weights. **Not
recommended.**

---

## Provenance

License-verification sources (accessed 2026-07-10): `github.com/mpiersonsmela/tbox`
(MIT — TBDB), `github.com/jamarchand/tbox-scan` (MIT), `wwpdb.org/about/usage-policies`
(PDB CC0-1.0, 2021 policy), `rfam.org` / EMBL-EBI licensing (Rfam CC0-1.0),
`gtdb.ecogenomic.org/downloads` (GTDB CC-BY-SA-4.0), NCBI data-usage policies (RefSeq +
Taxonomy public domain). Source-carried licenses: `data/external/gtdb/provenance.json`,
`data/external/gate1_anchor/provenance.json`, `data/external/refs/provenance.json`,
`data/external/ncbi_taxonomy/provenance.json`.
