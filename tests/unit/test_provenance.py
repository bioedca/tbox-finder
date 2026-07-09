"""P0-10: the provenance.json writer (CLAUDE.md §11).

Locks the per-artifact provenance convention: every derived artifact carries a
``provenance.json`` recording its rule, script, git SHA, env-lock hash, seed, and
input + output content hashes. The final test is the load-bearing gate — it builds
a record from *real* repo files and asserts all six §11 fields are populated with
real values (a real 40-hex commit SHA, a real 64-hex env-lock hash, real input +
output hashes), which is exactly what P0-10's validation gate requires.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from tbox_finder import provenance as prov

REPO_ROOT = Path(__file__).resolve().parents[2]

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# ---- content hashing -------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    payload = b"tbox-finder provenance\n\x00\x01\x02" * 100_000  # > 1 MiB, exercises chunking
    p.write_bytes(payload)
    assert prov.sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        prov.sha256_file(tmp_path / "does-not-exist")


def test_sha256_paths_keys_and_values(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("alpha")
    b.write_text("beta")
    got = prov.sha256_paths([a, b])
    assert set(got) == {str(a), str(b)}
    assert got[str(a)] == hashlib.sha256(b"alpha").hexdigest()
    assert all(SHA256_RE.match(v) for v in got.values())


def test_sha256_paths_empty():
    assert prov.sha256_paths([]) == {}


# ---- git SHA ---------------------------------------------------------------


def test_git_sha_in_repo():
    sha = prov.git_sha(REPO_ROOT)
    # In CI/dev the repo has commits → a 40-hex SHA; tolerate the graceful fallback.
    assert GIT_SHA_RE.match(sha) or sha == prov.GIT_SHA_UNKNOWN


def test_git_sha_non_repo_is_unknown(tmp_path):
    assert prov.git_sha(tmp_path) == prov.GIT_SHA_UNKNOWN


# ---- env-lock hash ---------------------------------------------------------


def test_env_lock_hash_matches_file(tmp_path):
    lock = tmp_path / "fake.conda-lock.yml"
    lock.write_text("name: fake\n")
    assert prov.env_lock_hash(lock) == hashlib.sha256(b"name: fake\n").hexdigest()


# ---- build_provenance ------------------------------------------------------


def test_build_provenance_has_all_required_fields(tmp_path):
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    inp.write_text("in")
    out.write_text("out")
    lock = tmp_path / "env.conda-lock.yml"
    lock.write_text("deps")

    rec = prov.build_provenance(
        rule="workflow/rules/data.smk :: ingest",
        script="scripts/ingest.py",
        seed=7,
        inputs=[inp],
        outputs=[out],
        env_lock=lock,
        adr="ADR-0003",
        repo_root=REPO_ROOT,
    )

    for field in prov.REQUIRED_FIELDS:
        assert field in rec, f"missing required §11 field: {field}"
    assert rec["rule"] == "workflow/rules/data.smk :: ingest"
    assert rec["script"] == "scripts/ingest.py"
    assert rec["seed"] == 7
    assert rec["adr"] == "ADR-0003"
    assert SHA256_RE.match(rec["env_lock_hash"])
    assert SHA256_RE.match(rec["inputs"][str(inp)])
    assert SHA256_RE.match(rec["outputs"][str(out)])
    assert rec["schema_version"] == prov.SCHEMA_VERSION
    # timestamp is ISO-8601 and parseable
    datetime.fromisoformat(rec["generated_at_utc"])


def test_build_provenance_defaults():
    rec = prov.build_provenance(rule="r", script="s.py")
    assert rec["seed"] == prov.DEFAULT_SEED
    assert rec["inputs"] == {}
    assert rec["outputs"] == {}
    assert rec["env_lock_hash"] is None
    assert rec["adr"] is None
    assert "extra" not in rec


def test_build_provenance_seed_coerced_to_int():
    rec = prov.build_provenance(rule="r", script="s.py", seed="13")
    assert rec["seed"] == 13
    assert isinstance(rec["seed"], int)


def test_build_provenance_extra_folded_in():
    rec = prov.build_provenance(rule="r", script="s.py", extra={"tool": "cmsearch 1.1.5"})
    assert rec["extra"] == {"tool": "cmsearch 1.1.5"}


# ---- write_provenance ------------------------------------------------------


def test_write_provenance_roundtrip(tmp_path):
    out = tmp_path / "nested" / "dir" / "provenance.json"  # parents must be created
    written = prov.write_provenance(
        out,
        rule="r",
        script="s.py",
        seed=1,
        repo_root=REPO_ROOT,
    )
    assert written == out
    assert out.is_file()
    loaded = json.loads(out.read_text())
    for field in prov.REQUIRED_FIELDS:
        assert field in loaded
    # keys are sorted for a stable diff
    assert list(loaded) == sorted(loaded)


# ---- load-bearing gate: real repo files ------------------------------------


def test_provenance_carries_all_stamps_on_real_repo_files(tmp_path):
    """P0-10 gate: a provenance.json built from real repo files carries a real
    git SHA, a real env-lock hash, and real input + output content hashes."""
    env_lock = REPO_ROOT / "envs" / "data.conda-lock.yml"
    an_input = REPO_ROOT / "pyproject.toml"
    an_output = REPO_ROOT / "README.md"
    for f in (env_lock, an_input, an_output):
        if not f.is_file():
            pytest.skip(f"expected repo file absent: {f}")

    out = tmp_path / "provenance.json"
    prov.write_provenance(
        out,
        rule="workflow/rules/setup.smk :: dvc_init",
        script="src/tbox_finder/provenance.py",
        seed=prov.DEFAULT_SEED,
        inputs=[an_input],
        outputs=[an_output],
        env_lock=env_lock,
        adr="ADR-0003",
        repo_root=REPO_ROOT,
    )
    rec = json.loads(out.read_text())

    assert GIT_SHA_RE.match(rec["git_sha"]), "git SHA must be a real 40-hex commit"
    assert SHA256_RE.match(rec["env_lock_hash"]), "env-lock hash must be real sha256"
    assert rec["inputs"] and all(SHA256_RE.match(v) for v in rec["inputs"].values())
    assert rec["outputs"] and all(SHA256_RE.match(v) for v in rec["outputs"].values())
    assert rec["env_lock_hash"] == hashlib.sha256(env_lock.read_bytes()).hexdigest()
