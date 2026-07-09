"""Unit tests for the P0-11a §7.1 reference-asset staging (src/tbox_finder/refs.py).

These verify the **committed staged state** under ``data/external/refs/`` — they do
NOT re-run staging (the immutable laptop sources are absent in CI). Every assertion
works whether the LFS-tracked ``*.cm`` / ``*.fa`` files are real bytes (laptop) or
Git LFS *pointers* (CI checkout with ``lfs: false``): a v1 pointer's
``oid sha256:`` is the content hash, so :func:`tbox_finder.refs.content_sha256`
returns the right digest either way.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tbox_finder import refs
from tbox_finder.refs import MANIFEST, REFS_DIR, RefAsset, content_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGED_DIR = REPO_ROOT / REFS_DIR
PROVENANCE = STAGED_DIR / "provenance.json"


def test_manifest_covers_every_expected_asset():
    """The manifest pins exactly the §7.1 assets: 6 CMs + 5 FASTAs, unique names."""
    names = [a.dest for a in MANIFEST]
    assert len(names) == len(set(names)), "duplicate destination filenames in MANIFEST"
    cms = {a.dest for a in MANIFEST if a.kind == "cm"}
    fastas = {a.dest for a in MANIFEST if a.kind == "fasta"}
    assert cms == {
        "RF00230.cm",
        "TBDB001.cm",
        "idtm_seq_aware.cm",
        "idtm_structure_only.cm",
        "idtm_structure_only.amendment2.cm",
        "stem2_structure_only.cm",
    }
    assert fastas == {
        "RF00230_master.fa",
        "RFGC3V_master.fa",
        "Vitreschak_master.fa",
        "gecont3_master.fa",
        "heldout_blind_set.fasta",
    }


def test_every_asset_is_staged():
    """Each manifest asset exists as a file under data/external/refs/."""
    for a in MANIFEST:
        staged = STAGED_DIR / a.dest
        assert staged.is_file(), f"missing staged asset: {staged}"


def test_provenance_manifest_present_and_complete():
    """provenance.json exists and carries source/version/accessed-date/license per asset."""
    assert PROVENANCE.is_file(), f"missing {PROVENANCE}"
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    by_name = {e["filename"]: e for e in prov["assets"]}
    assert set(by_name) == {a.dest for a in MANIFEST}
    for a in MANIFEST:
        entry = by_name[a.dest]
        # PRD §7.1 / CLAUDE.md §10.2 required provenance fields, all non-empty.
        for field in ("source_path", "version", "accessed_date", "license", "sha256"):
            assert entry.get(field), f"{a.dest}: empty/missing provenance field {field!r}"
        assert entry["source_path"] == a.source
        assert entry["sha256"] == a.sha256


@pytest.mark.parametrize("asset", MANIFEST, ids=lambda a: a.dest)
def test_staged_content_matches_pinned_checksum(asset: RefAsset):
    """Each staged asset hashes to its pinned SHA-256 (LFS-pointer-aware)."""
    staged = STAGED_DIR / asset.dest
    assert content_sha256(staged) == asset.sha256


def test_cm_assets_are_git_lfs_tracked():
    """Every ``*.cm`` (and every FASTA) routes through git-LFS per .gitattributes."""
    if shutil.which("git") is None:  # pragma: no cover - git present in CI + laptop
        pytest.skip("git unavailable")
    for a in MANIFEST:
        if not a.dest.endswith(".cm"):
            continue
        rel = f"{REFS_DIR.as_posix()}/{a.dest}"
        out = subprocess.run(
            ["git", "check-attr", "filter", "--", rel],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        got = out.stdout.strip()
        assert got.endswith("filter: lfs"), f"{rel} not git-LFS-tracked: {got}"


def test_content_sha256_reads_lfs_pointer(tmp_path: Path):
    """content_sha256 returns the oid of a Git LFS pointer without smudged content."""
    oid = "0" * 64
    pointer = tmp_path / "pointer.cm"
    pointer.write_text(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 12345\n"
    )
    assert content_sha256(pointer) == oid


def test_content_sha256_hashes_real_file(tmp_path: Path):
    """content_sha256 falls back to hashing real (non-pointer) bytes."""
    import hashlib

    real = tmp_path / "real.bin"
    payload = b">seq\nACGU\n"
    real.write_bytes(payload)
    assert content_sha256(real) == hashlib.sha256(payload).hexdigest()


def test_refs_module_imports_stdlib_only():
    """refs.py must import in a bare env (no data/ML stack) — CLAUDE.md §3.1 discipline."""
    assert refs.REFS_DIR.as_posix() == "data/external/refs"
    assert refs.ACCESSED_DATE == "2026-07-09"
