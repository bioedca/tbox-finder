"""taxonomy.py — governing GTDB release pin (P0-13) + TaxId lineage re-placement (P0-15).

One **governing GTDB release** binds every downstream taxonomic determination in the
project: the union novelty prior and all "novel-lineage" calls (PRD §7.2), the split
clade labels (§9.2), and the P6 GTDB-Tk host-placement DB (§13.2). Pinning it wrong
silently shifts every novelty call, so the release is frozen here — once — with full
provenance and a hard count gate.

**P0-15 (TaxId lineage re-placement).** ~4% of the 23,535-record corpus lacks a clade
label (453 no phylum, 841 no class, 928 no order — PRD §9.2), which the leave-one-order-out
headline split and the no-leakage test cannot silently absorb. Every such record carries
an NCBI ``TaxId``, so :func:`replace_lineage` re-derives its lineage-by-rank from a
**frozen, checksum-pinned NCBI Taxonomy snapshot** (dated ``taxdump_archive`` zip) and
fills only the missing ranks (fill-only — the curated 96% is never overwritten). The
corpus's existing labels are NCBI names at a pre-2021 vintage (``Firmicutes``, not the
modern ``Bacillota``) and the split groups on those exact strings, so recovered labels are
reconciled to the corpus vintage using the **taxdump's own synonym / equivalent-name
records** (no hand-built rename map to go stale). NCBI — not GTDB — is the re-derivation
source here because the ``TaxId`` is a native NCBI id, the split labels are NCBI-named, and
GTDB is genome-keyed (it cannot resolve the environmental / metagenome / CPR TaxIds that
dominate the residue). GTDB R232 remains governing for novelty (P0-14) + P6 placement; NCBI
is the frozen source for corpus-lineage recovery — complementary, not conflicting. Any
**still-incomplete residue** (organisms with no formal NCBI rank — metagenomes, uncultured
MAGs, CPR) is flagged ``dropped_from_clade_holdout`` (kept only in the random split), so a
no-clade record can never silently enter a clade fold. The recovery rate is reported per
rank; total accounting sums to 23,535 (fail-loud, §10.3).

**Pinned release: GTDB R232 (Release 11-RS232, 2026-04-15).** This is the release the
reusable tboxevo taxonomy maps were built against (tboxevo ADR-0001), and its contingency
is satisfied — an official GTDB-Tk reference package (``gtdbtk_r232_data.tar.gz``,
GTDB-Tk >= 2.7.0) is published — so the PRD §7.2 r220 fallback is *not* invoked. Because
placement (P6) uses the same release, no release-to-release crosswalk (§13.2) is needed.
Pin approved by the user 2026-07-09 (ADR-0003 Amendment A1; CLAUDE.md §7 items 1/2).

This module fetches + MD5-verifies the **species-representative crosswalk**
(``sp_clusters_r232.tsv``: one row per GTDB species cluster → representative genome +
GTDB taxonomy), whose row count **is** the species-rep count, and asserts it equals the
release's published value (bac120 189,801 + ar53 10,122 = 199,923; GTDB stats/r232,
accessed 2026-07-09) — fail-loud on any drift (CLAUDE.md §10.3). The full genome→lineage
taxonomy tables + metadata (consumed by P0-14/15/22, P6) are **pinned** here by URL +
authoritative MD5 but fetched on demand by those steps (they are ~120 MB and re-fetchable;
``data/external/`` is immutable + re-fetched by checksummed rules, never DVC-tracked —
CLAUDE.md §5.2). Provenance (source URL, release, accessed date, license, checksums,
counts) is written to ``data/external/gtdb/provenance.json`` (PRD §7.2; §10.2; §11).

Stdlib-only (plus :mod:`tbox_finder.provenance`, itself stdlib-only), so it imports in a
bare CI test env without pulling the data/ML stack, and the pure counting logic
(:func:`count_species_reps`) is unit-tested on a committed fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tbox_finder.provenance import SCHEMA_VERSION, git_sha, write_provenance

#: Canonical committed home for the GTDB pin provenance (the large crosswalk TSVs
#: staged alongside stay gitignored + re-fetched; only provenance.json is committed).
GTDB_DIR = Path("data/external/gtdb")

#: The date every GTDB fact below (release stats, MD5SUM.txt, license) was verified.
ACCESSED_DATE = "2026-07-09"

# --- Pinned governing release (PRD §7.2; ADR-0003 A1; user sign-off 2026-07-09) --------
GTDB_RELEASE = "R232"
GTDB_RELEASE_LABEL = "Release 11-RS232"
GTDB_RELEASE_DATE = "2026-04-15"
GTDB_BASE_URL = "https://data.gtdb.ecogenomic.org/releases/release232/232.0"
#: GTDB data license (gtdb.ecogenomic.org/downloads, accessed 2026-07-09).
GTDB_LICENSE = "CC-BY-SA-4.0 (GTDB data; https://gtdb.ecogenomic.org/downloads)"

#: The GTDB-Tk reference package whose existence makes R232 (not r220) the governing pin.
GTDBTK_PACKAGE = "gtdbtk_r232_data.tar.gz"
GTDBTK_MIN_VERSION = "2.7.0"

#: Published species-representative (species-cluster) counts — GTDB stats/r232, accessed
#: 2026-07-09; independently re-verified by :func:`count_species_reps` on sp_clusters.
SPECIES_REPS_BAC120 = 189_801
SPECIES_REPS_AR53 = 10_122
SPECIES_REPS_TOTAL = 199_923

#: The staged crosswalk whose row count is the species-rep count gate.
SP_CLUSTERS_NAME = "sp_clusters_r232.tsv"

_CHUNK = 1 << 20
_DOMAIN_BACTERIA = "d__Bacteria"
_DOMAIN_ARCHAEA = "d__Archaea"
#: sp_clusters columns (GTDB): rep genome, GTDB species, GTDB taxonomy, ANI..., members.
_TAXONOMY_COL = 2


@dataclass(frozen=True)
class GtdbFile:
    """One pinned GTDB release file — either staged here or fetched on demand."""

    name: str  # filename under GTDB_DIR
    url_path: str  # path relative to GTDB_BASE_URL
    md5: str  # authoritative MD5 (GTDB MD5SUM.txt, accessed ACCESSED_DATE)
    staged: bool  # True = downloaded + MD5-verified here; False = pinned for on-demand fetch
    role: str  # what the file is / who consumes it

    @property
    def url(self) -> str:
        return f"{GTDB_BASE_URL}/{self.url_path}"


# --- The R232 file manifest -----------------------------------------------------------
# MD5s are GTDB's own (release232/232.0/MD5SUM.txt, accessed 2026-07-09). The bac120
# taxonomy MD5 also matches the value tboxevo independently pinned — a cross-check that
# these maps are the same governing release.
FILES: tuple[GtdbFile, ...] = (
    GtdbFile(
        name=SP_CLUSTERS_NAME,
        url_path="auxillary_files/sp_clusters_r232.tsv",
        md5="7e4b2fc21135b6173d2b91ce4f290879",
        staged=True,
        role=(
            "Species-representative crosswalk — one row per GTDB species cluster "
            "(representative genome → GTDB taxonomy); row count == species-rep count "
            "(the P0-13 gate). Frames all §7.2 novelty determinations."
        ),
    ),
    GtdbFile(
        name="bac120_taxonomy_r232.tsv",
        url_path="bac120_taxonomy_r232.tsv",
        md5="4d42e137959485d57e8589bdf1b4c347",
        staged=False,
        role=(
            "Bacterial genome → 7-rank GTDB lineage crosswalk (all genomes). "
            "Fetched on demand by P0-14 (NCBI→GTDB projection), P0-15 (TaxId "
            "re-placement), P0-22 (clade labels)."
        ),
    ),
    GtdbFile(
        name="ar53_taxonomy_r232.tsv",
        url_path="ar53_taxonomy_r232.tsv",
        md5="c8e9aa537cf9c5e86e3fbf5972fd01d6",
        staged=False,
        role="Archaeal genome → 7-rank GTDB lineage crosswalk (§7.2 archaeal stretch).",
    ),
    GtdbFile(
        name="bac120_metadata_r232.tsv.gz",
        url_path="bac120_metadata_r232.tsv.gz",
        md5="0ba4237077b65cfc5556e1d8be797485",
        staged=False,
        role="Bacterial full metadata (gtdb_representative flag, GTDB-Tk placement inputs; P6).",
    ),
    GtdbFile(
        name="ar53_metadata_r232.tsv.gz",
        url_path="ar53_metadata_r232.tsv.gz",
        md5="0f51fe6f01b4ccbf0d9bab6c07df87af",
        staged=False,
        role="Archaeal full metadata (P6).",
    ),
)


# --- P0-15: pinned NCBI Taxonomy snapshot (TaxId → lineage re-placement) ---------------
#: Canonical committed home for the taxdump pin provenance (the dated zip staged alongside
#: stays gitignored + re-fetched; only provenance.json is committed — mirrors GTDB_DIR).
NCBI_TAXONOMY_DIR = Path("data/external/ncbi_taxonomy")

#: A dated snapshot from NCBI's **immutable** ``taxdump_archive`` (the dated zips never
#: change, unlike the rolling ``taxdump.tar.gz``), pinned for reproducible re-derivation.
#: 2026-07-01 is the most recent archived snapshot at/near the project accessed-date, so it
#: carries the most-complete lineage; being NCBI-native it resolves every corpus TaxId that
#: has a formal rank. NCBI publishes no ``.md5`` companion for the dated archive files, so
#: the checksums below are our own — computed on download (2026-07-09) and fail-loud
#: re-verified on every fetch (CLAUDE.md §10.3).
TAXDUMP_SNAPSHOT = "taxdmp_2026-07-01.zip"
TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump_archive/taxdmp_2026-07-01.zip"
TAXDUMP_MD5 = "f024526caf1d1183de84ab38edd24fbd"
TAXDUMP_SHA256 = "7afc2d0707abd481baff334bb5c345e9fbd36f4f0d1d912d2660ac18920def05"
#: NCBI Taxonomy is a US-Government work in the public domain (NCBI data-usage policies,
#: https://www.ncbi.nlm.nih.gov/home/about/policies/; accessed 2026-07-09).
TAXDUMP_LICENSE = "NCBI Taxonomy — public domain (US Gov work; NCBI data-usage policies)"

#: The lineage-by-rank columns re-derived per record (the corpus's own rank vocabulary).
LINEAGE_RANKS: tuple[str, ...] = ("phylum", "class", "order", "family", "genus")
#: The split ranks that carry a per-rank ``dropped_from_<rank>_holdout`` flag (PRD §9.2
#: reports residue "per rank"); the headline leave-one-order-out holdout is the ``order``.
HOLDOUT_RANKS: tuple[str, ...] = ("phylum", "class", "order")
HEADLINE_HOLDOUT_RANK = "order"

#: Per-rank provenance of a resolved label (which source filled it).
SOURCE_CORPUS = "corpus"  # kept from the curated corpus (fill-only never overwrites)
SOURCE_RECOVERED = "taxid_recovered"  # re-derived from the TaxId via the frozen taxdump
SOURCE_UNRESOLVED = "unresolved"  # no formal NCBI rank → residue (dropped from clade-holdout)

#: NCBI ``names.dmp`` scientific-name class (the fallback when no vintage synonym matches).
_SCI_NAME_CLASS = "scientific name"
#: Default corpus inputs (overridable on the CLI); ``ingested`` supplies the row-aligned
#: ``record_sha256`` hash-link that master_clean_v0 does not itself carry (PRD §9.2).
CORPUS_PARQUET = Path("data/processed/master_clean_v0.parquet")
INGESTED_PARQUET = Path("data/interim/master_tboxes_ingested.parquet")
INTERIM_DIR = Path("data/interim")
AUDIT_DIR = Path("data/processed/audits")
LINEAGE_REPLACED_NAME = "lineage_replaced.parquet"
LINEAGE_REPLACED_PROVENANCE = "lineage_replaced.provenance.json"
LINEAGE_AUDIT_NAME = "lineage_replacement_report.json"


def _stream_download(url: str, dest: Path) -> tuple[str, str, int]:
    """Stream ``url`` to ``dest``, returning ``(md5_hex, sha256_hex, n_bytes)``."""
    # MD5 mirrors GTDB's published MD5SUM.txt (integrity parity, not a security hash).
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    nbytes = 0
    req = urllib.request.Request(url, headers={"User-Agent": "tbox-finder/P0-13"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    # url is a pinned https GTDB release path (module constant), never user input.
    with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:
        for chunk in iter(lambda: resp.read(_CHUNK), b""):
            md5.update(chunk)
            sha.update(chunk)
            fh.write(chunk)
            nbytes += len(chunk)
    return md5.hexdigest(), sha.hexdigest(), nbytes


def _md5_file(path: Path) -> str:
    """Streamed MD5 of an existing file (idempotent-download re-verification)."""
    # MD5 mirrors GTDB's published MD5SUM.txt (integrity parity, not a security hash).
    md5 = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            sha.update(chunk)
    return sha.hexdigest()


def ensure_file(f: GtdbFile, dest_dir: Path) -> tuple[str, int]:
    """Ensure ``f`` is present + MD5-correct under ``dest_dir``; return ``(sha256, bytes)``.

    Re-uses an already-downloaded copy whose MD5 matches (idempotent); otherwise
    downloads it. Fail-loud if the fetched/existing bytes do not match the pinned MD5
    (CLAUDE.md §10.3 — a wrong crosswalk must never be pinned silently).
    """
    dest = dest_dir / f.name
    if dest.is_file() and _md5_file(dest) == f.md5:
        return _sha256_file(dest), dest.stat().st_size
    got_md5, got_sha, nbytes = _stream_download(f.url, dest)
    if got_md5 != f.md5:
        raise ValueError(
            f"GTDB checksum mismatch for {f.name}: got MD5 {got_md5} != pinned {f.md5} "
            f"(url {f.url})"
        )
    return got_sha, nbytes


def count_species_reps(sp_clusters_path: str | Path) -> dict[str, int]:
    """Count GTDB species representatives per domain from an ``sp_clusters`` TSV.

    Each data row is one species cluster (one representative genome), so the row count
    is the species-rep count; the domain is read from the leading ``d__`` of the
    ``GTDB taxonomy`` column. Returns ``{"total", "bacteria", "archaea", "other"}``.
    Streams the file (it is ~50 MB) and skips the header. ``other`` (non-bac/ar rows)
    should be 0 for a valid GTDB release file.
    """
    total = bacteria = archaea = other = 0
    with Path(sp_clusters_path).open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header.startswith("Representative genome"):
            raise ValueError(
                f"unexpected sp_clusters header (not a GTDB species-cluster file): {header[:60]!r}"
            )
        for line in fh:
            if not line.strip():
                continue
            fields = line.split("\t")
            taxonomy = fields[_TAXONOMY_COL] if len(fields) > _TAXONOMY_COL else ""
            total += 1
            if taxonomy.startswith(_DOMAIN_BACTERIA):
                bacteria += 1
            elif taxonomy.startswith(_DOMAIN_ARCHAEA):
                archaea += 1
            else:
                other += 1
    return {"total": total, "bacteria": bacteria, "archaea": archaea, "other": other}


def assert_published_counts(counts: Mapping[str, int]) -> None:
    """Fail-loud unless the counted species reps equal the release's published values."""
    expected = {
        "total": SPECIES_REPS_TOTAL,
        "bacteria": SPECIES_REPS_BAC120,
        "archaea": SPECIES_REPS_AR53,
        "other": 0,
    }
    if dict(counts) != expected:
        raise ValueError(
            "species-rep count gate FAILED (CLAUDE.md §10.3): counted "
            f"{dict(counts)} != published {expected} for GTDB {GTDB_RELEASE}. "
            "Do NOT pin — reconcile against the release before proceeding."
        )


def build_manifest(entries: list[dict], counts: Mapping[str, int] | None) -> dict:
    """Assemble the ``provenance.json`` record for the pinned GTDB release."""
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            f"P0-13: pinned governing GTDB release {GTDB_RELEASE} ({GTDB_RELEASE_LABEL}, "
            f"{GTDB_RELEASE_DATE}). Species-representative crosswalk staged + count-gated; "
            "genome→lineage taxonomy + metadata pinned for on-demand fetch. Binds §7.2 "
            "novelty, §9.2 clade labels, §13.2 P6 placement DB. CLAUDE.md §5.2/§10.2/§11."
        ),
        "rule": "workflow/rules/data.smk :: pin_gtdb_release",
        "script": "src/tbox_finder/taxonomy.py",
        "prd": "§7.2, §13.2",
        "adr": "ADR-0003 (Amendment A1)",
        "git_sha": git_sha(),
        "accessed_date": ACCESSED_DATE,
        "gtdb_release": GTDB_RELEASE,
        "gtdb_release_label": GTDB_RELEASE_LABEL,
        "gtdb_release_date": GTDB_RELEASE_DATE,
        "gtdb_base_url": GTDB_BASE_URL,
        "gtdb_license": GTDB_LICENSE,
        "gtdbtk_reference_package": GTDBTK_PACKAGE,
        "gtdbtk_min_version": GTDBTK_MIN_VERSION,
        "r220_fallback_invoked": False,
        "species_reps_published": {
            "bacteria": SPECIES_REPS_BAC120,
            "archaea": SPECIES_REPS_AR53,
            "total": SPECIES_REPS_TOTAL,
        },
        "species_reps_counted": dict(counts) if counts is not None else None,
        "files": entries,
    }


def pin_release(
    dest_dir: str | Path = GTDB_DIR,
    *,
    files: Iterable[GtdbFile] = FILES,
    download: bool = True,
) -> Path:
    """Pin GTDB ``R232``: stage + count-gate the species-rep crosswalk, write provenance.

    For each ``staged`` file: fetch (or re-use a checksum-matching copy) into ``dest_dir``
    and MD5-verify it. For each pinned-only file: record its URL + authoritative MD5 for
    on-demand fetch. Then count species reps from the staged ``sp_clusters`` crosswalk and
    assert the total matches the release's published value (fail-loud). Returns the
    ``provenance.json`` path. ``download=False`` skips fetching (records the pins only) —
    used by tests, which never touch the network.
    """
    dst_dir = Path(dest_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    sp_clusters_path: Path | None = None
    for f in files:
        entry: dict = {
            "filename": f.name,
            "url": f.url,
            "md5": f.md5,
            "staged": f.staged,
            "role": f.role,
            "license": GTDB_LICENSE,
            "accessed_date": ACCESSED_DATE,
        }
        if f.staged and download:
            sha256, nbytes = ensure_file(f, dst_dir)
            entry["sha256"] = sha256
            entry["bytes"] = nbytes
            entry["md5_verified"] = True
            if f.name == SP_CLUSTERS_NAME:
                sp_clusters_path = dst_dir / f.name
        entries.append(entry)

    counts: dict[str, int] | None = None
    if sp_clusters_path is not None:
        counts = count_species_reps(sp_clusters_path)
        assert_published_counts(counts)

    manifest = build_manifest(entries, counts)
    manifest_path = dst_dir / "provenance.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return manifest_path


# ======================================================================================
# P0-15 — TaxId → lineage re-derivation from the frozen NCBI taxdump
# ======================================================================================


def _iter_dmp(lines: Iterable[str]) -> Iterator[list[str]]:
    r"""Yield trimmed field-lists from NCBI ``.dmp`` rows (``a\t|\tb\t|\t…\t|``)."""
    for raw in lines:
        row = raw.rstrip("\n")
        if row.endswith("\t|"):
            row = row[:-2]
        yield [field.strip() for field in row.split("\t|\t")]


def parse_nodes(lines: Iterable[str]) -> tuple[dict[int, int], dict[int, str]]:
    """Parse ``nodes.dmp`` → ``(parent[taxid], rank[taxid])``."""
    parent: dict[int, int] = {}
    rank: dict[int, str] = {}
    for row in _iter_dmp(lines):
        taxid = int(row[0])
        parent[taxid] = int(row[1])
        rank[taxid] = row[2]
    return parent, rank


def parse_merged(lines: Iterable[str]) -> dict[int, int]:
    """Parse ``merged.dmp`` → ``{old_taxid: new_taxid}`` (retired-id redirects)."""
    return {int(row[0]): int(row[1]) for row in _iter_dmp(lines)}


def parse_names(
    lines: Iterable[str], keep: set[int] | None = None
) -> dict[int, dict[str, list[str]]]:
    """Parse ``names.dmp`` → ``{taxid: {name_class: [names]}}``.

    ``keep`` restricts to those taxids (only the ancestors reconciliation actually needs)
    so the ~290 MB file need not be held whole. **All** name classes are retained — the
    vintage reconciliation searches synonyms + equivalent names, not just the sci name.
    """
    out: dict[int, dict[str, list[str]]] = {}
    for row in _iter_dmp(lines):
        taxid = int(row[0])
        if keep is not None and taxid not in keep:
            continue
        out.setdefault(taxid, {}).setdefault(row[3], []).append(row[1])
    return out


def resolve_merged(taxid: int, merged: Mapping[int, int]) -> int:
    """Redirect a retired TaxId through ``merged.dmp`` (identity if not merged)."""
    return merged.get(int(taxid), int(taxid))


def lineage_by_rank(
    taxid: int,
    parent: Mapping[int, int],
    rank: Mapping[int, str],
    merged: Mapping[int, int],
    ranks: Sequence[str] = LINEAGE_RANKS,
) -> dict[str, int]:
    """Walk parents to root; return ``{rank_name: ancestor_taxid}`` for the target ranks.

    Ranks the lineage does not carry — a common case, since CPR / environmental / MAG
    taxa lack a formal order or even phylum in NCBI itself — are simply absent from the
    result (they become the residue). Cycle-guarded and root-terminating.
    """
    out: dict[str, int] = {}
    seen: set[int] = set()
    want = set(ranks)
    cur = resolve_merged(taxid, merged)
    while cur and cur not in seen:
        seen.add(cur)
        rk = rank.get(cur)
        if rk in want and rk not in out:
            out[rk] = cur
        nxt = parent.get(cur)
        if nxt is None or nxt == cur:
            break
        cur = nxt
    return out


def reconcile_name(
    taxid: int,
    names_by_taxid: Mapping[int, Mapping[str, Sequence[str]]],
    vocab_at_rank: Iterable[str],
) -> tuple[str | None, str]:
    """Pick the taxid's name matching the corpus vintage vocab, else its scientific name.

    Searches every name class (synonym, equivalent name, …) for a string the corpus
    already uses at this rank — so modern ``Bacillota`` reconciles back to the corpus's
    ``Firmicutes`` when NCBI records that as a synonym. Returns ``(name, source)`` with
    ``source ∈ {"vocab-synonym", "scientific"}``; ``name`` is ``None`` only if the taxid
    has no scientific name (never expected for a real node).
    """
    vocab = vocab_at_rank if isinstance(vocab_at_rank, (set, frozenset)) else set(vocab_at_rank)
    entry = names_by_taxid.get(taxid, {})
    for names in entry.values():
        for name in names:
            if name in vocab:
                return name, "vocab-synonym"
    sci = entry.get(_SCI_NAME_CLASS)
    return (sci[0] if sci else None), "scientific"


def resolve_row(
    existing: Mapping[str, str | None],
    lineage: Mapping[str, int],
    names_by_taxid: Mapping[int, Mapping[str, Sequence[str]]],
    vocab: Mapping[str, Iterable[str]],
    ranks: Sequence[str] = LINEAGE_RANKS,
) -> dict[str, object]:
    """Fill-only lineage resolution for one record.

    For each rank: keep the curated corpus label if present (``source=corpus`` — the 96%
    is never overwritten); else recover it from the TaxId lineage, reconciled to the
    corpus vintage (``source=taxid_recovered``); else leave it unresolved
    (``source=unresolved`` → residue). Returns ``resolved_<rank>`` + ``<rank>_source`` for
    every rank.
    """
    out: dict[str, object] = {}
    for rk in ranks:
        cur = existing.get(rk)
        if cur is not None:
            out[f"resolved_{rk}"] = cur
            out[f"{rk}_source"] = SOURCE_CORPUS
            continue
        anc = lineage.get(rk)
        name: str | None = None
        if anc is not None:
            name, _how = reconcile_name(anc, names_by_taxid, vocab.get(rk, ()))
        if name is not None:
            out[f"resolved_{rk}"] = name
            out[f"{rk}_source"] = SOURCE_RECOVERED
        else:
            out[f"resolved_{rk}"] = None
            out[f"{rk}_source"] = SOURCE_UNRESOLVED
    return out


def fetch_taxdump(dest_dir: str | Path = NCBI_TAXONOMY_DIR, *, download: bool = True) -> Path:
    """Ensure the pinned taxdump zip is present + MD5-correct under ``dest_dir``.

    With ``download=True`` (production) the pin is enforced: a checksum-matching staged
    copy is re-used, otherwise the file is (re-)downloaded and fail-loud MD5-verified
    against the pin (CLAUDE.md §10.3 — a wrong taxonomy snapshot must never drive the
    split silently). With ``download=False`` (offline / tests) an already-present file is
    trusted as-is (no network, no pin check — the caller supplied it deliberately); an
    absent file errors. Returns the zip path.
    """
    dst = Path(dest_dir)
    dst.mkdir(parents=True, exist_ok=True)
    zip_path = dst / TAXDUMP_SNAPSHOT
    if not download:
        if zip_path.is_file():
            return zip_path
        raise FileNotFoundError(f"taxdump {zip_path} absent and download=False")
    if zip_path.is_file() and _md5_file(zip_path) == TAXDUMP_MD5:
        return zip_path
    got_md5, _sha, _n = _stream_download(TAXDUMP_URL, zip_path)
    if got_md5 != TAXDUMP_MD5:
        raise ValueError(
            f"NCBI taxdump checksum mismatch: got MD5 {got_md5} != pinned {TAXDUMP_MD5} "
            f"(url {TAXDUMP_URL})"
        )
    return zip_path


def _zip_lines(zf: zipfile.ZipFile, name: str) -> Iterator[str]:
    """Yield decoded text lines from ``name`` inside an open taxdump zip."""
    with zf.open(name) as fh:
        yield from io.TextIOWrapper(fh, encoding="utf-8")


def read_taxdump(
    zip_path: str | Path, incomplete_taxids: Iterable[int], ranks: Sequence[str] = LINEAGE_RANKS
) -> tuple[dict[int, dict[str, int]], dict[int, dict[str, list[str]]]]:
    """From the taxdump zip, build ``{taxid: lineage_by_rank}`` for the incomplete TaxIds
    plus the ``names.dmp`` sub-table for the ancestor taxids those lineages reference."""
    incomplete = {int(t) for t in incomplete_taxids}
    with zipfile.ZipFile(zip_path) as zf:
        merged = parse_merged(_zip_lines(zf, "merged.dmp"))
        parent, rank = parse_nodes(_zip_lines(zf, "nodes.dmp"))
        lineages = {t: lineage_by_rank(t, parent, rank, merged, ranks) for t in incomplete}
        needed = {anc for lin in lineages.values() for anc in lin.values()}
        names_by_taxid = parse_names(_zip_lines(zf, "names.dmp"), keep=needed)
    return lineages, names_by_taxid


@dataclass(frozen=True)
class ReplaceLineageOutputs:
    """Paths + in-memory report returned by :func:`replace_lineage`."""

    parquet: Path
    audit: Path
    provenance: Path
    taxdump_provenance: Path
    report: dict


def build_taxdump_provenance(zip_path: Path) -> dict:
    """The committed pin manifest for the frozen NCBI taxdump (mirrors the GTDB pin)."""
    size = zip_path.stat().st_size if Path(zip_path).is_file() else None
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "P0-15: pinned NCBI Taxonomy snapshot for TaxId → lineage re-placement of the "
            "~4% clade-incomplete corpus positives (PRD §9.2). NCBI — not GTDB — is the "
            "re-derivation source: the TaxId is a native NCBI id and the split labels are "
            "NCBI-named at a pre-2021 vintage; GTDB stays governing for novelty (P0-14) + "
            "P6 placement. Immutable dated taxdump_archive zip; MD5-verified on fetch."
        ),
        "rule": "workflow/rules/data.smk :: replace_taxid_lineage",
        "script": "src/tbox_finder/taxonomy.py",
        "prd": "§9.2, §12",
        "adr": "PRD §9.2 (ADR-0004 pending, P0-19)",
        "git_sha": git_sha(),
        "accessed_date": ACCESSED_DATE,
        "snapshot": TAXDUMP_SNAPSHOT,
        "url": TAXDUMP_URL,
        "md5": TAXDUMP_MD5,
        "sha256": TAXDUMP_SHA256,
        "bytes": size,
        "license": TAXDUMP_LICENSE,
        "files_used": ["nodes.dmp", "names.dmp", "merged.dmp"],
    }


def build_recovery_report(out_df, corpus_missing, vocab, ranks=LINEAGE_RANKS) -> dict:
    """Assemble + fail-loud-check the per-rank recovery accounting from the resolved frame.

    ``out_df`` is the resolved DataFrame; ``corpus_missing`` maps each rank to its
    pre-recovery isna count (the independent cross-check). Asserts, for every rank,
    ``recovered + still_incomplete == pre_missing`` and
    ``present + recovered + still_incomplete == n`` (CLAUDE.md §10.3). Also lists the
    genuinely-new taxa a recovery introduced (recovered names absent from the corpus
    vocab — e.g. new Candidatus orders), for transparency.
    """
    n = len(out_df)
    per_rank: dict[str, dict] = {}
    new_taxa: dict[str, dict[str, int]] = {}
    for r in ranks:
        src = out_df[f"{r}_source"]
        present = int((src == SOURCE_CORPUS).sum())
        recovered = int((src == SOURCE_RECOVERED).sum())
        still = int((src == SOURCE_UNRESOLVED).sum())
        pre_missing = int(corpus_missing[r])
        if recovered + still != pre_missing:
            raise ValueError(
                f"accounting FAILED (§10.3) at rank {r}: recovered {recovered} + "
                f"still_incomplete {still} != pre_missing {pre_missing}"
            )
        if present + recovered + still != n:
            raise ValueError(
                f"accounting FAILED (§10.3) at rank {r}: present {present} + recovered "
                f"{recovered} + still {still} != n {n}"
            )
        per_rank[r] = {
            "present_from_corpus": present,
            "pre_missing": pre_missing,
            "recovered": recovered,
            "still_incomplete": still,
            "recovery_rate": round(recovered / pre_missing, 4) if pre_missing else None,
        }
        rec_mask = src == SOURCE_RECOVERED
        rvocab = set(vocab.get(r, ()))
        counts: dict[str, int] = {}
        for name in out_df.loc[rec_mask, f"resolved_{r}"]:
            if name is not None and name not in rvocab:
                counts[name] = counts.get(name, 0) + 1
        if counts:
            new_taxa[r] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    dropped = {r: int((out_df[f"{r}_source"] == SOURCE_UNRESOLVED).sum()) for r in HOLDOUT_RANKS}
    return {
        "n_records": n,
        "per_rank": per_rank,
        "dropped_from_clade_holdout": dropped[HEADLINE_HOLDOUT_RANK],
        "dropped_per_rank": dropped,
        "new_taxa_introduced": new_taxa,
        "accounting_ok": True,
    }


def replace_lineage(
    corpus_path: str | Path = CORPUS_PARQUET,
    ingested_path: str | Path = INGESTED_PARQUET,
    *,
    taxdump_dir: str | Path = NCBI_TAXONOMY_DIR,
    interim_dir: str | Path = INTERIM_DIR,
    audit_dir: str | Path = AUDIT_DIR,
    download: bool = True,
    ranks: Sequence[str] = LINEAGE_RANKS,
) -> ReplaceLineageOutputs:
    """Re-derive lineage-by-rank for the taxonomy-incomplete positives (P0-15; PRD §9.2).

    Fill-only: keeps every present corpus label, recovers only missing ranks from the
    frozen NCBI taxdump (vintage-reconciled), flags the still-incomplete residue
    ``dropped_from_clade_holdout``. Writes ``lineage_replaced.parquet`` (keyed by the
    row-aligned ``record_sha256``), a per-rank recovery audit, an artifact provenance, and
    the taxdump pin provenance. Returns their paths + the report.
    """
    import pandas as pd  # lazy — keeps the module import stdlib-only for the unit tests

    corpus_path = Path(corpus_path)
    interim_dir = Path(interim_dir)
    audit_dir = Path(audit_dir)
    corpus = pd.read_parquet(corpus_path)
    n = len(corpus)

    # Row-aligned record_sha256 hash-link (master_clean_v0 carries no hash itself; PRD §9.2).
    ingested = pd.read_parquet(ingested_path, columns=["record_sha256", "TaxId"])
    if len(ingested) != n:
        raise ValueError(
            f"ingested ({len(ingested)}) / corpus ({n}) length mismatch — cannot attach "
            "record_sha256"
        )
    if not (
        ingested["TaxId"].reset_index(drop=True) == corpus["TaxId"].reset_index(drop=True)
    ).all():
        raise ValueError(
            "ingested/clean parquet row-misalignment (TaxId) — cannot attach record_sha256"
        )

    # Corpus vintage vocab per rank (the reconciliation target) + the incomplete TaxIds.
    vocab = {r: frozenset(corpus[r].dropna().astype(str)) for r in ranks}
    corpus_missing = {r: int(corpus[r].isna().sum()) for r in ranks}
    incomplete_mask = pd.Series(False, index=corpus.index)
    for r in ranks:
        incomplete_mask = incomplete_mask | corpus[r].isna()
    incomplete_taxids = {int(t) for t in corpus.loc[incomplete_mask, "TaxId"].unique()}

    zip_path = fetch_taxdump(taxdump_dir, download=download)
    lineages, names_by_taxid = read_taxdump(zip_path, incomplete_taxids, ranks)

    existing_records = corpus[list(ranks)].to_dict("records")
    taxids = corpus["TaxId"].astype(int).tolist()
    rows = []
    for existing_row, taxid in zip(existing_records, taxids, strict=True):
        existing = {r: (v if pd.notna(v) else None) for r, v in existing_row.items()}
        rows.append(resolve_row(existing, lineages.get(taxid, {}), names_by_taxid, vocab, ranks))
    out = pd.DataFrame.from_records(rows)
    out.insert(0, "record_sha256", ingested["record_sha256"].to_numpy())
    out.insert(1, "TaxId", corpus["TaxId"].to_numpy())
    for r in HOLDOUT_RANKS:
        out[f"dropped_from_{r}_holdout"] = out[f"{r}_source"] == SOURCE_UNRESOLVED
    out["dropped_from_clade_holdout"] = out[f"dropped_from_{HEADLINE_HOLDOUT_RANK}_holdout"]

    report = build_recovery_report(out, corpus_missing, vocab, ranks)

    interim_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = interim_dir / LINEAGE_REPLACED_NAME
    out.to_parquet(out_parquet, index=False)
    out_audit = audit_dir / LINEAGE_AUDIT_NAME
    out_audit.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    tax_prov_path = Path(taxdump_dir) / "provenance.json"
    tax_prov_path.parent.mkdir(parents=True, exist_ok=True)
    tax_prov_path.write_text(
        json.dumps(build_taxdump_provenance(zip_path), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )

    out_provenance = interim_dir / LINEAGE_REPLACED_PROVENANCE
    write_provenance(
        out_provenance,
        rule="workflow/rules/data.smk :: replace_taxid_lineage",
        script="src/tbox_finder/taxonomy.py",
        inputs=[corpus_path, ingested_path, tax_prov_path],
        outputs=[out_parquet, out_audit],
        env_lock="envs/data.conda-lock.yml",
        adr="PRD §9.2 (ADR-0004 pending, P0-19)",
        extra={
            "taxdump_snapshot": TAXDUMP_SNAPSHOT,
            "n_records": n,
            "dropped_from_clade_holdout": report["dropped_from_clade_holdout"],
            "order_recovery_rate": report["per_rank"]["order"]["recovery_rate"],
        },
    )
    return ReplaceLineageOutputs(out_parquet, out_audit, out_provenance, tax_prov_path, report)


# ======================================================================================
# CLI dispatch — ``pin`` (P0-13, default) / ``replace-lineage`` (P0-15)
# ======================================================================================


def _run_pin(argv: list[str] | None) -> int:
    """P0-13 CLI: ``python -m tbox_finder.taxonomy [pin] --dest-dir data/external/gtdb``."""
    parser = argparse.ArgumentParser(
        prog="tbox_finder.taxonomy pin",
        description="Pin the governing GTDB release + stage the species-rep crosswalk (P0-13).",
    )
    parser.add_argument("--dest-dir", default=str(GTDB_DIR), help="staging destination")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="record the pins only; do not fetch or run the count gate (offline)",
    )
    args = parser.parse_args(argv)
    out = pin_release(args.dest_dir, download=not args.no_download)
    manifest = json.loads(out.read_text())
    counted = manifest.get("species_reps_counted")
    tail = f" ({counted['total']} species reps)" if counted else " (pins only, no fetch)"
    print(f"pinned GTDB {GTDB_RELEASE} -> {out}{tail}")
    return 0


def _run_replace_lineage(argv: list[str] | None) -> int:
    """P0-15 CLI: ``python -m tbox_finder.taxonomy replace-lineage --interim-dir …``."""
    parser = argparse.ArgumentParser(
        prog="tbox_finder.taxonomy replace-lineage",
        description="Re-derive lineage-by-rank for taxonomy-incomplete positives (P0-15).",
    )
    parser.add_argument("--corpus", default=str(CORPUS_PARQUET))
    parser.add_argument("--ingested", default=str(INGESTED_PARQUET))
    parser.add_argument("--taxdump-dir", default=str(NCBI_TAXONOMY_DIR))
    parser.add_argument("--interim-dir", default=str(INTERIM_DIR))
    parser.add_argument("--audit-dir", default=str(AUDIT_DIR))
    parser.add_argument(
        "--no-download", action="store_true", help="require an already-present taxdump zip"
    )
    args = parser.parse_args(argv)
    result = replace_lineage(
        args.corpus,
        args.ingested,
        taxdump_dir=args.taxdump_dir,
        interim_dir=args.interim_dir,
        audit_dir=args.audit_dir,
        download=not args.no_download,
    )
    order = result.report["per_rank"]["order"]
    print(
        f"re-placed lineage -> {result.parquet} "
        f"(order recovered {order['recovered']}/{order['pre_missing']}, "
        f"{result.report['dropped_from_clade_holdout']} dropped_from_clade_holdout)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI dispatch. ``replace-lineage`` → P0-15; anything else → the P0-13 pin (default)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "replace-lineage":
        return _run_replace_lineage(args[1:])
    if args and args[0] == "pin":
        return _run_pin(args[1:])
    return _run_pin(args)


if __name__ == "__main__":
    raise SystemExit(main())
