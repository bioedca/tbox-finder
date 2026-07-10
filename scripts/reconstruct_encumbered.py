#!/usr/bin/env python3
"""Dataset license registry + reconstruct stubs for encumbered sources — tbox-finder P0-18.

This is the machine-readable companion to ``docs/license_audit.md`` and the release-time
tool that keeps the curated dataset **own-derived + permissively licensed** (ADR-0001 §D9;
PRD header / §2.1 G-E). It carries:

1. ``SOURCES`` — every upstream source feeding the training corpus / curated dataset, each
   with its **verified** upstream license (SPDX where one exists), whether the release
   **redistributes** it, and a compatibility **verdict** against the default
   **CC-BY-4.0 own-derived** release: ``compatible`` (permissive/public-domain →
   redistribute-with-attribution) or ``encumbered`` (share-alike or proprietary →
   reconstruct-from-source, never embedded in the release).
2. A ``reconstruct_*`` **stub** per encumbered source. The released curated dataset ships
   only own-derived fields (coordinates, labels, predictions, splits, calibration) + the
   RefSeq-re-derived anchor; the columns that *derive from* an encumbered source (the
   GTDB-projected novelty-prior labels) are **not embedded** — this script re-fetches the
   encumbered source from its pinned, checksummed provenance and delegates the actual
   column reconstruction to the in-repo module that already produces it. Network fetch is
   **opt-in** (``--download``); by default the functions read the committed
   ``data/external/**/provenance.json`` pins and return a fetch *plan* only (no network),
   which is what the P0-18 unit test exercises. Full materialization runs at P7 release.

The two encumbered sources are **GTDB** (CC-BY-SA-4.0 — share-alike is incompatible with a
permissive CC-BY-4.0 redistribution) and the **Vitreschak 2008 RNA supplement**
(© RNA Society / CSHL Press — proprietary; the released GATE-1 anchor leaders are re-derived
from NCBI RefSeq, which is public domain, so the supplement itself is never redistributed).

Network (``--download`` only): the pinned public HTTPS endpoints in the committed
provenance. Read-only; no secret is embedded or logged. Stdlib only.

Usage:
    python scripts/reconstruct_encumbered.py --list             # human table
    python scripts/reconstruct_encumbered.py --json             # registry as JSON
    python scripts/reconstruct_encumbered.py --reconstruct gtdb # fetch plan (no network)
    python scripts/reconstruct_encumbered.py --reconstruct gtdb --download --dest /tmp/gtdb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Verdicts.
COMPATIBLE = "compatible"  # permissive / public-domain → safe to redistribute (with attribution)
ENCUMBERED = "encumbered"  # share-alike or proprietary → reconstruct-from-source, never embedded

# Routes.
REDISTRIBUTE = "redistribute-with-attribution"
RECONSTRUCT = "reconstruct-from-source"

# The resolved curated-dataset license (ADR-0001 §D9 default; user sign-off 2026-07-10).
# Q-question RESOLVED — Option A: own-derived fields under CC-BY-4.0 + this reconstruct
# script for the encumbered sources (Option B, embed-everything-under-CC-BY-SA-4.0, declined).
RELEASE_LICENSE = "CC-BY-4.0"


@dataclass(frozen=True)
class Source:
    """One upstream source + its license-compatibility verdict for the curated-dataset release."""

    source_id: str
    display_name: str
    role: str
    upstream_license: str  # human-readable license string (as verified upstream)
    license_spdx: str  # SPDX id, "public-domain", or "proprietary"
    redistributed_in_release: bool  # is upstream content (not just own-derived facts) shipped?
    verdict: str  # COMPATIBLE | ENCUMBERED
    route: str  # REDISTRIBUTE | RECONSTRUCT
    provenance_url: str
    accessed: str  # YYYY-MM-DD (license verified against the authoritative upstream)
    notes: str = ""


SOURCES: tuple[Source, ...] = (
    Source(
        source_id="tbdb_master",
        display_name="TBDB — Master_tboxes.csv (primary training corpus)",
        role="23,535-record annotation table → cleaned corpus + own-derived labels/splits",
        upstream_license="MIT (github.com/mpiersonsmela/tbox — Pierson Smela et al. 2020)",
        license_spdx="MIT",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://github.com/mpiersonsmela/tbox",
        accessed="2026-07-10",
        notes=(
            "MIT is permissive → the cleaned corpus + own-derived fields may be relicensed "
            "CC-BY-4.0 with the MIT notice + attribution retained. The local tboxdb-master "
            "archive omits the LICENSE file; the MIT license verified against the upstream repo."
        ),
    ),
    Source(
        source_id="rfam_rf00230",
        display_name="Rfam RF00230 (class-I covariance model / baseline)",
        role="cmsearch GATE-1 baseline CM + RF00230-derived reference FASTA + masking",
        upstream_license="CC0-1.0 (Rfam / EMBL-EBI — all Rfam data are CC0 public-domain)",
        license_spdx="CC0-1.0",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://rfam.org/",
        accessed="2026-07-10",
        notes="CC0 public-domain dedication — no restriction; cite as good scientific practice.",
    ),
    Source(
        source_id="tbox_scan_cm",
        display_name="tbox-scan — TBDB001.cm (class-II covariance model)",
        role="class-II baseline CM (the anti-mimicry ablation withholds it from labels)",
        upstream_license="MIT (github.com/jamarchand/tbox-scan — 2020)",
        license_spdx="MIT",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://github.com/jamarchand/tbox-scan",
        accessed="2026-07-10",
        notes="MIT (verified against the upstream repo's About sidebar / LICENSE).",
    ),
    Source(
        source_id="tboxevo_cms",
        display_name="tboxevo sub-element covariance models (idtm_*, stem2_*)",
        role="structural sub-element CMs (predecessor project; GATE-1 baselines)",
        upstream_license="MIT (tboxevo predecessor; built from TBDB/Rfam-derived seeds)",
        license_spdx="MIT",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://github.com/bioedca/tbox-finder",
        accessed="2026-07-10",
        notes="Own-lineage derivative of MIT (TBDB) / CC0 (Rfam) seed alignments; MIT.",
    ),
    Source(
        source_id="pdb_crystals",
        display_name="PDB crystal structures (2KZL, 4LCK, 4MGN, 6POM, 6UFG, …; 9 entries)",
        role="structural ground truth → per-element extents for the label-derivation fixtures",
        upstream_license="CC0-1.0 (wwPDB — PDB archive data adopted CC0 in 2021)",
        license_spdx="CC0-1.0",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://www.wwpdb.org/about/usage-policies",
        accessed="2026-07-10",
        notes="CC0 public-domain dedication; attribute the structure depositors as good practice.",
    ),
    Source(
        source_id="vitreschak_supplement",
        display_name="Vitreschak et al. 2008 (RNA) supplement — Supp_Fig1/Table/Legends (.doc)",
        role="literature localizer for the independent GATE-1 anchor (arm c) re-derivation",
        upstream_license="© 2008 RNA Society / Cold Spring Harbor Laboratory Press (proprietary)",
        license_spdx="proprietary",
        redistributed_in_release=False,
        verdict=ENCUMBERED,
        route=RECONSTRUCT,
        provenance_url="https://rnajournal.cshlp.org/content/suppl/2008/03/21/14.4.717.DC1/",
        accessed="2026-07-10",
        notes=(
            "Copyrighted supplement — NOT redistributed in-repo/in-dataset. The released "
            "anchor leaders are re-derived from NCBI RefSeq (public domain) via "
            "src/tbox_finder/anchors.py::source_anchor, so nothing encumbered ships. This "
            "route re-fetches the supplement (checksummed) to reproduce the re-derivation."
        ),
    ),
    Source(
        source_id="gtdb",
        display_name="GTDB R232 taxonomy + species-rep tables",
        role="governing taxonomy for the union novelty prior (NCBI→GTDB projection) + P6 placement",
        upstream_license="CC-BY-SA-4.0 (GTDB — https://gtdb.ecogenomic.org/downloads)",
        license_spdx="CC-BY-SA-4.0",
        redistributed_in_release=False,
        verdict=ENCUMBERED,
        route=RECONSTRUCT,
        provenance_url="https://gtdb.ecogenomic.org/downloads",
        accessed="2026-07-10",
        notes=(
            "Share-alike (CC-BY-SA-4.0) is incompatible with a permissive CC-BY-4.0 "
            "redistribution: embedding the GTDB compilation would force the whole dataset "
            "to CC-BY-SA-4.0. The GTDB-projected novelty-prior columns are therefore NOT "
            "embedded — this route re-fetches the pinned R232 tables (P0-13) and the "
            "projection is re-applied by src/tbox_finder/priors.py. Own-derived has-prior/"
            "no-prior verdicts ship as booleans (facts), not the GTDB name compilation."
        ),
    ),
    Source(
        source_id="ncbi_refseq",
        display_name="NCBI RefSeq (prokaryotic genomes)",
        role="scanned discovery corpus (streamed, discarded) + re-derived anchor-leader source",
        upstream_license="Public domain (US Government work; NCBI data-usage policies)",
        license_spdx="public-domain",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://www.ncbi.nlm.nih.gov/refseq/",
        accessed="2026-07-10",
        notes=(
            "Public domain; anchor leaders derived from it are freely shippable. "
            "Scanned genomes are not in the dataset (ADR-0003)."
        ),
    ),
    Source(
        source_id="ncbi_taxonomy",
        display_name="NCBI Taxonomy (taxdump)",
        role="TaxId lineage re-placement (P0-15) — corpus-vintage NCBI lineage columns",
        upstream_license="Public domain (US Government work; NCBI data-usage policies)",
        license_spdx="public-domain",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://www.ncbi.nlm.nih.gov/taxonomy",
        accessed="2026-07-10",
        notes="Public domain; the re-placed lineage-by-rank columns derived from it are shippable.",
    ),
    Source(
        source_id="anchor_p0_16",
        display_name="GATE-1 independent anchor set (P0-16; own-derived)",
        role="phylogenetically-independent GATE-1 arm (c) positives (re-derived from RefSeq)",
        upstream_license="CC-BY-4.0 (this project's own derived work, from public-domain RefSeq)",
        license_spdx="CC-BY-4.0",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://github.com/bioedca/tbox-finder",
        accessed="2026-07-10",
        notes="Own work; released under the curated-dataset license.",
    ),
    Source(
        source_id="classII_p0_17",
        display_name="Additional class-II positives set (P0-17; own-derived, VERIFIED-EMPTY)",
        role="class-II anti-mimicry sub-arm positives (natural non-Actinobacteria set is empty)",
        upstream_license="CC-BY-4.0 (this project's own work; leads catalogued by-reference)",
        license_spdx="CC-BY-4.0",
        redistributed_in_release=True,
        verdict=COMPATIBLE,
        route=REDISTRIBUTE,
        provenance_url="https://github.com/bioedca/tbox-finder",
        accessed="2026-07-10",
        notes=(
            "Own work; empty FASTA. The 40 leads are TBDB-derived (MIT) and "
            "referenced, not embedded."
        ),
    ),
)


def encumbered_sources() -> list[Source]:
    """Every source whose upstream license forces the reconstruct-from-source route."""
    return [s for s in SOURCES if s.verdict == ENCUMBERED]


def source_by_id(source_id: str) -> Source:
    for s in SOURCES:
        if s.source_id == source_id:
            return s
    raise KeyError(f"unknown source_id: {source_id!r} (known: {[s.source_id for s in SOURCES]})")


def _load_provenance(rel_path: str) -> dict:
    """Read a committed ``data/external/**/provenance.json`` pin (no network)."""
    path = REPO_ROOT / rel_path
    if not path.is_file():
        raise FileNotFoundError(
            f"provenance pin not found: {path} — needed to build the reconstruct plan"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = (
        hashlib.md5()
    )  # noqa: S324 — provenance-verification digest (GTDB publishes MD5), not security
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_verify(
    url: str, dest: Path, *, md5: str | None = None, sha256: str | None = None
) -> Path:
    """Fetch ``url`` → ``dest`` and fail-loud on a checksum mismatch (§10.3)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 — pinned public HTTPS endpoint
    if md5 is not None:
        got = _md5(dest)
        if got != md5:
            raise ValueError(f"MD5 mismatch for {url}: expected {md5}, got {got}")
    if sha256 is not None:
        got = _sha256(dest)
        if got != sha256:
            raise ValueError(f"SHA-256 mismatch for {url}: expected {sha256}, got {got}")
    return dest


def reconstruct_gtdb(dest: Path | None = None, *, download: bool = False) -> dict:
    """Re-fetch the pinned GTDB R232 taxonomy tables so the NCBI→GTDB novelty-prior projection
    can be re-applied (src/tbox_finder/priors.py). Plan-only unless ``download=True``.

    The GTDB compilation (CC-BY-SA-4.0) is never embedded in the release; this materializes
    the encumbered-derived columns from source on the user's machine.
    """
    prov = _load_provenance("data/external/gtdb/provenance.json")
    # The taxonomy crosswalks drive the projection; the species-rep table is auxiliary.
    wanted_roles = ("taxonomy", "crosswalk", "lineage")
    files = [
        f
        for f in prov.get("files", [])
        if any(
            k in (f.get("role", "").lower() + " " + f.get("filename", "").lower())
            for k in wanted_roles
        )
        or "taxonomy" in f.get("filename", "").lower()
    ]
    if not files:  # fall back to the whole manifest rather than silently reconstruct nothing
        files = prov.get("files", [])
    plan = {
        "source_id": "gtdb",
        "license": prov.get("gtdb_license"),
        "release": prov.get("gtdb_release"),
        "base_url": prov.get("gtdb_base_url"),
        "reconstruction_module": "src/tbox_finder/priors.py (NCBI→GTDB projection; P0-14)",
        "files": [
            {
                "filename": f.get("filename"),
                "url": f.get("url"),
                "md5": f.get("md5"),
                "sha256": f.get("sha256"),
            }
            for f in files
        ],
        "downloaded": [],
    }
    if download:
        if dest is None:
            raise ValueError("--dest is required with --download")
        for f in files:
            out = Path(dest) / f["filename"]
            _download_verify(f["url"], out, md5=f.get("md5"))
            plan["downloaded"].append(str(out))
    return plan


def reconstruct_vitreschak(dest: Path | None = None, *, download: bool = False) -> dict:
    """Re-fetch the © RNA Society/CSHL Vitreschak-2008 supplement (checksummed) so the GATE-1
    anchor leaders can be re-derived from RefSeq. Plan-only unless ``download=True``.

    The supplement is proprietary and never redistributed; only the RefSeq-re-derived anchor
    ships. This route reproduces the re-derivation (src/tbox_finder/anchors.py::source_anchor).
    """
    prov = _load_provenance("data/external/gate1_anchor/provenance.json")
    src = prov.get("extra", {}).get("source", {})
    base = src.get("supplement_base")
    shas = src.get("supplement_sha256", {})
    plan = {
        "source_id": "vitreschak_supplement",
        "license": src.get("license"),
        "supplement_base": base,
        "reconstruction_module": "src/tbox_finder/anchors.py::source_anchor (RefSeq; P0-16)",
        "files": [
            {"filename": name, "url": (base or "") + name, "sha256": sha}
            for name, sha in shas.items()
        ],
        "downloaded": [],
    }
    if download:
        if dest is None:
            raise ValueError("--dest is required with --download")
        for f in plan["files"]:
            out = Path(dest) / f["filename"]
            _download_verify(f["url"], out, sha256=f.get("sha256"))
            plan["downloaded"].append(str(out))
    return plan


RECONSTRUCTORS = {
    "gtdb": reconstruct_gtdb,
    "vitreschak_supplement": reconstruct_vitreschak,
}


def _print_table() -> None:
    print(f"Resolved curated-dataset license (own-derived fields): {RELEASE_LICENSE}")
    print(f"Encumbered → reconstruct: {[s.source_id for s in encumbered_sources()]}")
    print()
    hdr = f"{'source_id':<22} {'verdict':<11} {'route':<28} {'license_spdx'}"
    print(hdr)
    print("-" * len(hdr))
    for s in SOURCES:
        print(f"{s.source_id:<22} {s.verdict:<11} {s.route:<28} {s.license_spdx}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list", action="store_true", help="print the per-source verdict table")
    ap.add_argument("--json", action="store_true", help="dump the source registry as JSON")
    ap.add_argument(
        "--reconstruct",
        metavar="SOURCE_ID",
        help="build a reconstruct plan for an encumbered source",
    )
    ap.add_argument(
        "--download", action="store_true", help="actually fetch (checksummed) instead of plan-only"
    )
    ap.add_argument("--dest", type=Path, help="destination dir for --download")
    args = ap.parse_args(argv)

    if args.json:
        print(
            json.dumps(
                {"release_license": RELEASE_LICENSE, "sources": [asdict(s) for s in SOURCES]},
                indent=2,
            )
        )
        return 0
    if args.reconstruct:
        fn = RECONSTRUCTORS.get(args.reconstruct)
        if fn is None:
            print(
                f"'{args.reconstruct}' is not an encumbered source; "
                f"reconstruct targets: {list(RECONSTRUCTORS)}",
                file=sys.stderr,
            )
            return 2
        plan = fn(args.dest, download=args.download)
        print(json.dumps(plan, indent=2))
        return 0
    # default: the table
    _print_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
