"""Unit tests for the P0-13 governing-GTDB-release pin (src/tbox_finder/taxonomy.py).

These verify the **pin contract** — the pinned release id, the published species-rep
counts, the file manifest (staged vs on-demand-fetch split + authoritative MD5s), and
the pure species-rep counting logic exercised on a committed **real** sp_clusters
fixture. They never touch the network: the 49 MB ``sp_clusters_r232.tsv`` is gitignored
+ re-fetched by the rule (CLAUDE.md §5.2), so CI does not have it, and the offline
``pin_release(download=False)`` path is what CI exercises.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tbox_finder import taxonomy
from tbox_finder.taxonomy import (
    FILES,
    GTDB_BASE_URL,
    SP_CLUSTERS_NAME,
    SPECIES_REPS_AR53,
    SPECIES_REPS_BAC120,
    SPECIES_REPS_TOTAL,
    assert_published_counts,
    count_species_reps,
    pin_release,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/gtdb_sp_clusters/sp_clusters_sample.tsv"
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")


def test_governing_release_is_r232():
    """The pinned governing release is R232 (tboxevo-matching; user sign-off 2026-07-09)."""
    assert taxonomy.GTDB_RELEASE == "R232"
    assert taxonomy.GTDB_RELEASE_LABEL == "Release 11-RS232"
    assert taxonomy.GTDB_RELEASE_DATE == "2026-04-15"
    assert taxonomy.ACCESSED_DATE == "2026-07-09"
    assert taxonomy.GTDB_DIR.as_posix() == "data/external/gtdb"
    # Contingency that makes R232 (not r220) the pin: an available GTDB-Tk package.
    assert taxonomy.GTDBTK_PACKAGE == "gtdbtk_r232_data.tar.gz"
    assert taxonomy.GTDBTK_MIN_VERSION == "2.7.0"
    assert "CC-BY-SA-4.0" in taxonomy.GTDB_LICENSE


def test_published_species_rep_counts_sum():
    """Published bac120 + ar53 species reps sum to the pinned total (GTDB stats/r232)."""
    assert SPECIES_REPS_BAC120 == 189_801
    assert SPECIES_REPS_AR53 == 10_122
    assert SPECIES_REPS_TOTAL == 199_923
    assert SPECIES_REPS_BAC120 + SPECIES_REPS_AR53 == SPECIES_REPS_TOTAL


def test_file_manifest_staged_vs_pinned():
    """Exactly the species-rep crosswalk is staged; taxonomy + metadata are pinned-only."""
    by_name = {f.name: f for f in FILES}
    assert len(by_name) == len(FILES), "duplicate filenames in FILES"
    staged = {f.name for f in FILES if f.staged}
    assert staged == {SP_CLUSTERS_NAME}, "only sp_clusters is staged (count-gate artifact)"
    # The genome→lineage taxonomy + metadata are pinned for on-demand fetch (P0-14/15/22/P6).
    assert {f.name for f in FILES if not f.staged} == {
        "bac120_taxonomy_r232.tsv",
        "ar53_taxonomy_r232.tsv",
        "bac120_metadata_r232.tsv.gz",
        "ar53_metadata_r232.tsv.gz",
    }


@pytest.mark.parametrize("f", FILES, ids=lambda f: f.name)
def test_every_file_has_authoritative_md5_and_gtdb_url(f):
    """Each pinned file carries a 32-hex MD5 and a URL under the R232 release directory."""
    assert _MD5_RE.match(f.md5), f"{f.name}: MD5 not 32-hex: {f.md5!r}"
    assert f.url.startswith(GTDB_BASE_URL + "/")
    assert f.role, f"{f.name}: empty role"


def test_count_species_reps_on_real_fixture():
    """count_species_reps splits domains from real GTDB sp_clusters rows (3 bac + 2 ar)."""
    assert FIXTURE.is_file(), f"missing fixture: {FIXTURE}"
    assert count_species_reps(FIXTURE) == {
        "total": 5,
        "bacteria": 3,
        "archaea": 2,
        "other": 0,
    }


def test_count_species_reps_rejects_non_gtdb_header(tmp_path: Path):
    """A file whose header is not a GTDB species-cluster header is refused (fail-loud)."""
    bad = tmp_path / "bad.tsv"
    bad.write_text("not\ta\tgtdb\tfile\nrow1\trow1\td__Bacteria;p__x\n")
    with pytest.raises(ValueError, match="unexpected sp_clusters header"):
        count_species_reps(bad)


def test_assert_published_counts_gate():
    """The count gate passes on the exact published counts and fails otherwise (§10.3)."""
    assert_published_counts(
        {
            "total": SPECIES_REPS_TOTAL,
            "bacteria": SPECIES_REPS_BAC120,
            "archaea": SPECIES_REPS_AR53,
            "other": 0,
        }
    )
    with pytest.raises(ValueError, match="species-rep count gate FAILED"):
        assert_published_counts(
            {"total": 199_922, "bacteria": 189_800, "archaea": 10_122, "other": 0}
        )


def test_pin_release_offline_writes_provenance(tmp_path: Path):
    """pin_release(download=False) records the pins + counts=None without any network I/O."""
    out = pin_release(tmp_path, download=False)
    assert out == tmp_path / "provenance.json"
    prov = json.loads(out.read_text(encoding="utf-8"))
    assert prov["gtdb_release"] == "R232"
    assert prov["gtdb_release_date"] == "2026-04-15"
    assert prov["r220_fallback_invoked"] is False
    assert prov["gtdbtk_reference_package"] == "gtdbtk_r232_data.tar.gz"
    assert prov["gtdb_license"].startswith("CC-BY-SA-4.0")
    assert prov["species_reps_published"] == {
        "bacteria": SPECIES_REPS_BAC120,
        "archaea": SPECIES_REPS_AR53,
        "total": SPECIES_REPS_TOTAL,
    }
    # download=False → no fetch, so no verified count is recorded.
    assert prov["species_reps_counted"] is None
    # No bytes were written for the staged crosswalk (nothing fetched).
    assert not (tmp_path / SP_CLUSTERS_NAME).exists()
    # Every file is pinned with the §10.2 provenance fields.
    by_name = {e["filename"]: e for e in prov["files"]}
    assert set(by_name) == {f.name for f in FILES}
    for f in FILES:
        entry = by_name[f.name]
        for field in ("url", "md5", "staged", "role", "license", "accessed_date"):
            assert field in entry, f"{f.name}: missing provenance field {field!r}"
        assert entry["md5"] == f.md5


def test_taxonomy_imports_without_data_ml_stack():
    """taxonomy.py must import in a bare env (no data/ML stack) — CLAUDE.md §3.1 discipline."""
    heavy = "{'numpy', 'pandas', 'polars', 'torch', 'transformers', 'snakemake', 'dvc', 'Bio'}"
    code = (
        "import sys\n"
        "import tbox_finder.taxonomy  # noqa: F401\n"
        f"hit = {heavy} & set(sys.modules)\n"
        "assert not hit, f'taxonomy.py pulled heavy deps: {hit}'\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
