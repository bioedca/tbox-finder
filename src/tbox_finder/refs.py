"""refs.py — stage the §7.1 immutable reference assets into ``data/external/refs/``.

The project's genome-scan masking (P0-30), the GATE-1 reference covariance models
(P4), and the P6 orthogonal validation all consume a *fixed* set of reference
assets: the two class covariance models (Rfam class-I ``RF00230.cm`` and the
class-II ``TBDB001.cm``), a handful of tboxevo-built structural sub-element CMs
(``idtm_*.cm`` / ``stem2_*.cm``), the TBDB "master" reference FASTAs, and the
single natural class-II blind set (``heldout_blind_set.fasta`` — 18 Actinobacteria
/ ILE records). PRD §7.1 pins these as *immutable raw inputs*, and CLAUDE.md §5.2
says ``data/external/`` is staged by a **checksummed** rule and never hand-edited.

This module is that checksummed staging rule. Each asset is pinned by its expected
SHA-256; :func:`stage_refs` copies the source, re-verifies the copy against the
pinned hash (fail-loud on any mismatch — CLAUDE.md §10.3), and writes a
``provenance.json`` manifest carrying, per asset, its source path, version,
accessed date, and license (PRD §7.1; CLAUDE.md §10.2/§11).

The source assets live in immutable sibling checkouts on the laptop
(``tboxdb-master/``, ``tbox-scan-master/``, ``tboxevo/`` — the ``sources_root``);
they are **not** present on a fresh clone or the cluster checkout, so staging is a
one-time LOCAL operation and the staged copies (git-LFS for ``*.cm`` / ``*.fa`` /
``*.fasta``; PRD §9.2 / .gitattributes) are what travel with the repo. The staged
state is verified independently by ``tests/unit/test_refs_staging.py``, which is
LFS-pointer-aware so it passes in CI where LFS content is not smudged.

Stdlib-only (plus :mod:`tbox_finder.provenance`, itself stdlib-only), so it imports
in a bare CI test env without pulling the data/ML stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tbox_finder.provenance import SCHEMA_VERSION, git_sha

#: Canonical committed location for the staged reference assets (PRD §7.1 / §16).
REFS_DIR = Path("data/external/refs")

#: The date the immutable sources were staged (provenance ``accessed_date``).
ACCESSED_DATE = "2026-07-09"

#: First line of a Git LFS v1 pointer file (spec: git-lfs/git-lfs docs/spec.md).
_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"

_CHUNK = 1 << 20


@dataclass(frozen=True)
class RefAsset:
    """One staged reference asset + the provenance PRD §7.1 / §10.2 requires."""

    dest: str  # filename under REFS_DIR
    source: str  # path relative to ``sources_root``
    sha256: str  # pinned expected content hash (SHA-256 hexdigest)
    kind: str  # "cm" | "fasta"
    upstream: str  # origin database / family / build
    version: str  # version string (Rfam release, Infernal build, tboxevo build)
    license: str  # license of the source distribution
    note: str  # short human description (role downstream)

    @property
    def lfs(self) -> bool:
        """``*.cm`` / ``*.fa`` / ``*.fasta`` are git-LFS-tracked (.gitattributes)."""
        return self.dest.endswith((".cm", ".fa", ".fasta"))


# --- The §7.1 reference-asset manifest ------------------------------------------
# SHA-256 values are pinned from the immutable sources (verified 2026-07-09). A
# copy whose hash drifts from these is refused — the assets are immutable inputs.
_TBOXDB_MIT = "MIT (tboxdb — Pierson Smela, Marchand & Jordan 2020; see tboxdb LICENSE)"
_TBOXSCAN_MIT = "MIT (tbox-scan — jamarchand 2020; see tbox-scan LICENSE)"
_TBOXEVO_DERIVED = (
    "Derived, this project's own (tboxevo predecessor); built from "
    "TBDB/RF00230-derived seed alignments — upstream MIT (TBDB) / CC0 (Rfam)"
)

MANIFEST: tuple[RefAsset, ...] = (
    # --- class covariance models (baselines) ---
    RefAsset(
        dest="RF00230.cm",
        source="tboxdb-master/tboxdb-master/pipeline/RF00230.cm",
        sha256="082616efe0616479dc9b9f565b5bcab7648d7ea1d629430940319f247af52cea",
        kind="cm",
        upstream="Rfam RF00230 (T-box leader), EMBL-EBI; vendored in tboxdb",
        version="Rfam RF00230 / INFERNAL1/a [1.1 | October 2013]",
        license="CC0-1.0 (Rfam RF00230, upstream); redistributed via " + _TBOXDB_MIT,
        note="Class-I T-box covariance model (cmsearch GATE-1 baseline; P0-30 masking, P4, P6).",
    ),
    RefAsset(
        dest="TBDB001.cm",
        source="tbox-scan-master/tbox-scan-master/tboxscan/data/TBDB001.cm",
        sha256="00a821f23bfaf96c1e454022bc0a8fcbe2a9150abadd00bca6abb9298d9a0d1c",
        kind="cm",
        upstream="tbox-scan / TBDB (tbdb.io) class-II model (NAME seed_ILE)",
        version="TBDB001 / INFERNAL1/a [1.1.2 | July 2016]",
        license=_TBOXSCAN_MIT,
        note="Class-II T-box covariance model (baseline; P0-30 masking, P4, P6).",
    ),
    # --- tboxevo structural sub-element CMs (apical / Stem II seed alignments) ---
    RefAsset(
        dest="idtm_seq_aware.cm",
        source="tboxevo/data/processed/cms/idtm_seq_aware.cm",
        sha256="9210b9619377e5f18a6c8350198ee2caa8bf1018e1c3f3954c0bca0b7b67ca70",
        kind="cm",
        upstream="tboxevo idtm apical seed alignment (sequence-aware variant)",
        version="tboxevo build / INFERNAL1/a [1.1.5 | Sep 2023]",
        license=_TBOXEVO_DERIVED,
        note="Structural sub-element model — apical/specifier region (sequence-aware).",
    ),
    RefAsset(
        dest="idtm_structure_only.cm",
        source="tboxevo/data/processed/cms/idtm_structure_only.cm",
        sha256="800f949fe149e498d347493fdee8bd736a04110aca1bba1ab2f401460fd64fcf",
        kind="cm",
        upstream="tboxevo idtm apical seed alignment (structure-only)",
        version="tboxevo build / INFERNAL1/a [1.1.5 | Sep 2023]",
        license=_TBOXEVO_DERIVED,
        note="Structural sub-element model — apical/specifier region (structure-only).",
    ),
    RefAsset(
        dest="idtm_structure_only.amendment2.cm",
        source="tboxevo/data/processed/cms/idtm_structure_only.amendment2.cm",
        sha256="d93f7a5b5803cbe5bbb7555b81ce778acfdcc88a87ca7fb53c3e34d758857d20",
        kind="cm",
        upstream="tboxevo idtm apical seed alignment (structure-only, amendment 2)",
        version="tboxevo build / INFERNAL1/a [1.1.5 | Sep 2023]",
        license=_TBOXEVO_DERIVED,
        note="Structural sub-element CM — apical region (structure-only, amendment 2).",
    ),
    RefAsset(
        dest="stem2_structure_only.cm",
        source="tboxevo/data/processed/cms/stem2_structure_only.cm",
        sha256="6ee7106e37369a5ab3092ad9cecac30a8122d3aea6f91be76a61ba4d03eab409",
        kind="cm",
        upstream="tboxevo Stem II apical seed alignment (structure-only)",
        version="tboxevo build / INFERNAL1/a [1.1.5 | Sep 2023]",
        license=_TBOXEVO_DERIVED,
        note="Structural sub-element model — Stem II apical region (structure-only).",
    ),
    # --- TBDB "master" reference FASTAs (RF00230-derived / GC-norm / gecont3 / Vitreschak) ---
    RefAsset(
        dest="RF00230_master.fa",
        source="tboxdb-master/tboxdb-master/master_datasets/MASTER/RF00230_master.fa",
        sha256="9d343159073d90f66f92e929bbb9a9644b0d82327a86b16e3df07e2c9166158c",
        kind="fasta",
        upstream="TBDB master dataset — RF00230-derived (12,287 seqs)",
        version="TBDB master_datasets/MASTER (2020)",
        license=_TBOXDB_MIT,
        note="Reference FASTA — RF00230-derived component of the §7.1 master reference set.",
    ),
    RefAsset(
        dest="RFGC3V_master.fa",
        source="tboxdb-master/tboxdb-master/master_datasets/MASTER/RFGC3V_master.fa",
        sha256="5b91e48216663023b601913570079a7465de94235de08ed15f244dcaddbba1c2",
        kind="fasta",
        upstream="TBDB master dataset — combined RF00230+GC+gecont3+Vitreschak (11,228 seqs)",
        version="TBDB master_datasets/MASTER (2020)",
        license=_TBOXDB_MIT,
        note="Reference FASTA — combined RF00230/GC-norm/gecont3/Vitreschak master set (§7.1).",
    ),
    RefAsset(
        dest="Vitreschak_master.fa",
        source="tboxdb-master/tboxdb-master/master_datasets/MASTER/Vitreschak_master.fa",
        sha256="1f66d7b90f1f55de24efca3b84c611e19aad8a57ff6315f32b72dc113818e559",
        kind="fasta",
        upstream="TBDB master dataset — Vitreschak 2008 literature set (698 seqs)",
        version="TBDB master_datasets/MASTER (2020)",
        license=_TBOXDB_MIT,
        note="Reference FASTA — Vitreschak 2008 literature-occurrence component (§7.1).",
    ),
    RefAsset(
        dest="gecont3_master.fa",
        source="tboxdb-master/tboxdb-master/master_datasets/MASTER/gecont3_master.fa",
        sha256="00aedc38db4c0b7e09fbd77d50fa7edb57ff8af9ae8589e56a98b9075e077efa",
        kind="fasta",
        upstream="TBDB master dataset — gecont3 gene-context set (2,426 seqs)",
        version="TBDB master_datasets/MASTER (2020)",
        license=_TBOXDB_MIT,
        note="Reference FASTA — gecont3 gene-context component of the §7.1 master reference set.",
    ),
    # --- the single natural class-II blind set (18 Actinobacteria / ILE records) ---
    RefAsset(
        dest="heldout_blind_set.fasta",
        source="tboxevo/data/processed/cms/heldout_blind_set.fasta",
        sha256="3c26f5aee39fb4c6f303a92c14b4f25ae4f57302db36090ddc92e80bfbf780f1",
        kind="fasta",
        upstream=(
            "tboxevo; TBDB-derived; 18 Actinobacteria / ILE / class-II records, "
            "genus-guided by Vitreschak 2008 + Battaglia 2019 (PRD §7.1)"
        ),
        version="tboxevo build (18 records)",
        license="TBDB-derived; " + _TBOXDB_MIT,
        note="Class-II blind set — single-phylum, NOT phylo-independent; §9.2 no-leakage.",
    ),
)


def content_sha256(path: str | Path) -> str:
    """SHA-256 of an asset's *content*, whether a real file or a Git LFS pointer.

    A committed ``*.cm`` / ``*.fa`` is a real file in the laptop working tree but an
    LFS *pointer* on a CI checkout that did not smudge LFS (``actions/checkout`` with
    ``lfs: false``). A v1 pointer's ``oid sha256:<hex>`` **is** the content hash
    (git-lfs spec), so either representation yields the same digest — letting the
    staging test verify checksums in both places.
    """
    p = Path(path)
    data = p.read_bytes()
    if data.startswith(_LFS_POINTER_MAGIC):
        for line in data.decode("utf-8").splitlines():
            if line.startswith("oid sha256:"):
                return line.split("oid sha256:", 1)[1].strip()
        raise ValueError(f"Git LFS pointer without an 'oid sha256:' line: {p}")
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_streamed(path: Path) -> str:
    """Streamed SHA-256 of a real file (staging path — sources are never pointers)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(dest_dir: Path, assets: Iterable[RefAsset]) -> dict:
    """Assemble the ``provenance.json`` record for the staged assets."""
    entries = []
    for a in assets:
        staged = dest_dir / a.dest
        entries.append(
            {
                "filename": a.dest,
                "kind": a.kind,
                "lfs": a.lfs,
                "sha256": a.sha256,
                "bytes": staged.stat().st_size,
                "source_path": a.source,
                "upstream": a.upstream,
                "version": a.version,
                "license": a.license,
                "accessed_date": ACCESSED_DATE,
                "note": a.note,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "P0-11a: staged PRD §7.1 immutable reference assets — class covariance "
            "models, tboxevo structural sub-element CMs, TBDB master reference FASTAs, "
            "and the class-II blind set. data/external/ is immutable (CLAUDE.md §5.2)."
        ),
        "rule": "workflow/rules/data.smk :: stage_reference_assets",
        "script": "src/tbox_finder/refs.py",
        "prd": "§7.1",
        "git_sha": git_sha(),
        "accessed_date": ACCESSED_DATE,
        "assets": entries,
    }


def stage_refs(
    sources_root: str | Path,
    dest_dir: str | Path = REFS_DIR,
    *,
    assets: Iterable[RefAsset] = MANIFEST,
) -> Path:
    """Copy each §7.1 source asset into ``dest_dir``, verifying pinned checksums.

    For every asset: the source must exist and hash to its pinned SHA-256, the copy
    is re-hashed and must match (fail-loud on any drift — CLAUDE.md §10.3), then a
    ``provenance.json`` manifest is written. Returns the manifest path.
    """
    src_root = Path(sources_root)
    dst_dir = Path(dest_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    materialized = list(assets)
    for a in materialized:
        src = src_root / a.source
        if not src.is_file():
            raise FileNotFoundError(f"§7.1 source asset missing: {src}")
        src_hash = _sha256_streamed(src)
        if src_hash != a.sha256:
            raise ValueError(
                f"source checksum mismatch for {a.dest}: {src_hash} != pinned {a.sha256}"
            )
        dst = dst_dir / a.dest
        shutil.copyfile(src, dst)
        dst_hash = _sha256_streamed(dst)
        if dst_hash != a.sha256:
            raise ValueError(
                f"staged checksum mismatch for {a.dest}: {dst_hash} != pinned {a.sha256}"
            )
    manifest = build_manifest(dst_dir, materialized)
    manifest_path = dst_dir / "provenance.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m tbox_finder.refs --sources-root <dir>``."""
    parser = argparse.ArgumentParser(description="Stage the §7.1 reference assets (P0-11a).")
    parser.add_argument(
        "--sources-root",
        default="..",
        help="dir holding the immutable tboxdb-master/tbox-scan-master/tboxevo checkouts",
    )
    parser.add_argument("--dest-dir", default=str(REFS_DIR), help="staging destination")
    args = parser.parse_args(argv)
    out = stage_refs(args.sources_root, args.dest_dir)
    print(f"staged {len(MANIFEST)} reference assets -> {out.parent} (manifest: {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
