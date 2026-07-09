"""taxonomy.py — pin the governing GTDB release + stage the species-rep crosswalk (P0-13).

One **governing GTDB release** binds every downstream taxonomic determination in the
project: the union novelty prior and all "novel-lineage" calls (PRD §7.2), the split
clade labels (§9.2), and the P6 GTDB-Tk host-placement DB (§13.2). Pinning it wrong
silently shifts every novelty call, so the release is frozen here — once — with full
provenance and a hard count gate.

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
import json
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from tbox_finder.provenance import SCHEMA_VERSION, git_sha

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


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m tbox_finder.taxonomy --dest-dir data/external/gtdb``."""
    parser = argparse.ArgumentParser(
        description="Pin the governing GTDB release + stage the species-rep crosswalk (P0-13)."
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


if __name__ == "__main__":
    raise SystemExit(main())
