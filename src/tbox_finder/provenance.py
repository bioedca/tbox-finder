"""provenance.py — the per-artifact ``provenance.json`` writer (CLAUDE.md §11).

Every derived artifact in this project carries a ``provenance.json`` sidecar so a
DVC-tracked table or checkpoint can be traced back to *exactly* the code, commit,
environment, seed, and inputs that produced it (CLAUDE.md §11; PRD §16). This is
the canonical writer for that convention — the tbox-finder counterpart of the
``provenance.json`` map described for tboxevo, with one deliberate upgrade: the
env-lock hash is **actually computed** here (in tboxevo it was left an unwired
``null`` stub), because the P0-10 validation gate requires it.

A provenance record carries the six fields CLAUDE.md §11 mandates —
``rule``, ``script``, ``git_sha``, ``env_lock_hash``, ``seed``, and the input +
output hashes — plus a ``schema_version``, a UTC timestamp, and an optional ``adr``
cross-reference. All content hashes are SHA-256 hexdigests (64 chars), files
streamed in 1 MiB chunks (the tboxevo file-hash idiom).

Stdlib-only, so it imports in any env (including a bare CI test env) without
pulling the ML/data stack.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Bump when the on-disk provenance schema changes in a non-additive way.
SCHEMA_VERSION = "1.0"

#: File-hash read chunk (1 MiB), matching the tboxevo streaming idiom.
_CHUNK = 1 << 20

#: Sentinel recorded when the git SHA cannot be determined (no repo / no git).
GIT_SHA_UNKNOWN = "unknown"

#: Project-wide default seed (kept in step with the Hydra/Snakemake seed policy,
#: CLAUDE.md §8.3). Callers pass their own seed; this is only the fallback.
DEFAULT_SEED = 42

#: The six field names CLAUDE.md §11 requires every provenance record to carry.
REQUIRED_FIELDS = ("rule", "script", "git_sha", "env_lock_hash", "seed", "inputs", "outputs")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hexdigest of ``path``, streamed in 1 MiB chunks.

    Raises ``FileNotFoundError`` if the file is absent — a provenance record must
    never silently hash a missing input/output (CLAUDE.md §10.3, fail loud).
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_paths(paths: Iterable[str | Path]) -> dict[str, str]:
    """Map ``{path-as-given: sha256}`` for an iterable of file paths.

    The key is the path exactly as passed (relative stays relative), so the record
    reads the way the rule referenced the file; the value is its content hash.
    """
    return {str(p): sha256_file(p) for p in paths}


def git_sha(repo_root: str | Path | None = None) -> str:
    """Current commit SHA via ``git rev-parse HEAD``.

    Returns :data:`GIT_SHA_UNKNOWN` if git is unavailable or ``repo_root`` is not a
    repository, rather than raising — provenance capture must not abort a pipeline.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return GIT_SHA_UNKNOWN
    return out.stdout.strip() or GIT_SHA_UNKNOWN


def env_lock_hash(lock_path: str | Path) -> str:
    """SHA-256 of a ``conda-lock`` lockfile — the environment reproducibility stamp.

    The lockfile pins exact package builds (ADR-0002), so its hash identifies the
    resolved environment an artifact was produced in.
    """
    return sha256_file(lock_path)


def build_provenance(
    *,
    rule: str,
    script: str | Path,
    seed: int = DEFAULT_SEED,
    inputs: Iterable[str | Path] | None = None,
    outputs: Iterable[str | Path] | None = None,
    env_lock: str | Path | None = None,
    adr: str | None = None,
    repo_root: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance record (CLAUDE.md §11) as a plain dict.

    Args:
        rule: the producing entry — a Snakemake rule or module entry
            (e.g. ``"workflow/rules/data.smk :: ingest_master"``).
        script: path to the script/module that ran.
        seed: the stochastic seed used (CLAUDE.md §8.3); default :data:`DEFAULT_SEED`.
        inputs: input file paths — each hashed into ``inputs``.
        outputs: output file paths — each hashed into ``outputs``.
        env_lock: the ``conda-lock`` lockfile for the producing env; hashed into
            ``env_lock_hash``. ``None`` records ``None`` (e.g. an env-free step).
        adr: optional ADR id this artifact's method is pinned by (e.g. ``"ADR-0003"``).
        repo_root: directory to resolve the git SHA from (defaults to CWD).
        extra: optional extra key/values folded in under ``extra`` (e.g. tool versions).

    Returns:
        A JSON-serializable dict carrying every :data:`REQUIRED_FIELDS` field.
    """
    prov: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rule": rule,
        "script": str(script),
        "git_sha": git_sha(repo_root),
        "env_lock_hash": env_lock_hash(env_lock) if env_lock is not None else None,
        "seed": int(seed),
        "inputs": sha256_paths(inputs or []),
        "outputs": sha256_paths(outputs or []),
        "adr": adr,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    if extra:
        prov["extra"] = dict(extra)
    return prov


def write_provenance(out_path: str | Path, **kwargs: Any) -> Path:
    """Build a provenance record and write it as pretty JSON to ``out_path``.

    Keyword args are forwarded to :func:`build_provenance`. Parent directories are
    created as needed. Keys are sorted for a stable, diff-friendly layout. Returns
    the path written.
    """
    prov = build_provenance(**kwargs)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    return p
