"""hub.py — the release-time Hugging Face Hub push/pull helper (CLAUDE.md §6.3).

At preprint/paper release the trained model and curated dataset are pushed to the HF
Hub with model + dataset cards (CLAUDE.md §6.3; PRD §16/§17). This module is the
scaffold for that step — thin, tested wrappers over ``huggingface_hub`` that:

  * authenticate **only** from the environment (``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``),
    never from a committed secret — the repo is public (CLAUDE.md §4). The token is
    read for a single API call and is never logged, written to disk, or persisted;
  * **defer** the ``huggingface_hub`` import into each function, so
    ``import tbox_finder.hub`` succeeds in a bare env (the CI test env, or any
    non-``ml`` env) with the dependency absent — the same bare-env-import discipline
    as ``provenance.py``.

Nothing here is wired into the CPU DAG or the training loop: HF push is release-time
only (CLAUDE.md §6.3). Pinned client: ``huggingface_hub`` 0.35.3 (ADR-0002).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huggingface_hub import HfApi

#: Env vars huggingface_hub itself honors, in precedence order (``HF_TOKEN`` wins).
#: We only ever READ these — a token is never written to disk or logged.
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")

#: Repo types the Hub accepts for our two release artifacts (weights vs. dataset).
REPO_TYPE_MODEL = "model"
REPO_TYPE_DATASET = "dataset"


def resolve_token() -> str | None:
    """Return the HF token from the environment, or ``None`` if unset.

    Reads :data:`TOKEN_ENV_VARS` in precedence order. The value is returned to the
    caller for a single API call and is never logged, written, or committed
    (CLAUDE.md §4). ``None`` means "no token" — a public read still works; a push
    then fails loudly at the API layer, which is the intended behavior.
    """
    for var in TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return None


def _require_hf_api(token: str | None) -> HfApi:
    """Deferred-import an :class:`~huggingface_hub.HfApi` bound to ``token``.

    Imported here (not at module scope) so this module loads without
    ``huggingface_hub`` installed — the provenance-style bare-env import discipline
    (CLAUDE.md §11). Raises a clear :class:`ImportError` if the dependency is absent.
    The token is passed explicitly (and is not persisted by ``HfApi``); ``None`` lets
    ``huggingface_hub`` fall back to its own env/stored-login resolution.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - only hit in a bare env
        raise ImportError(
            "huggingface_hub is required for HF Hub push/pull; activate the `ml-dna` "
            "or `ml-rna` conda env (ADR-0002)."
        ) from exc
    return HfApi(token=token)


def push_folder(
    repo_id: str,
    folder: str | Path,
    *,
    repo_type: str = REPO_TYPE_MODEL,
    path_in_repo: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> Any:
    """Create ``repo_id`` if needed, then upload ``folder`` to it (release-time).

    Args:
        repo_id: ``"<owner>/<name>"`` on the Hub.
        folder: local directory to upload (must exist).
        repo_type: :data:`REPO_TYPE_MODEL` (default) or :data:`REPO_TYPE_DATASET`.
        path_in_repo: subpath within the repo to upload into (root if ``None``).
        private: create the repo private (default public — release artifacts).
        commit_message: upload commit message; a default is derived if ``None``.
        token: explicit token; falls back to :func:`resolve_token` (env only).
        allow_patterns / ignore_patterns: forwarded file globs.

    Returns:
        The ``huggingface_hub`` ``CommitInfo`` for the upload commit.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"upload folder does not exist: {folder}")
    api = _require_hf_api(token or resolve_token())
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        commit_message=commit_message or f"Upload {repo_type} {repo_id}",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def pull_snapshot(
    repo_id: str,
    *,
    repo_type: str = REPO_TYPE_MODEL,
    revision: str | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> str:
    """Download a full repo snapshot and return the local path.

    Deferred-imports ``snapshot_download``. ``token`` falls back to
    :func:`resolve_token` (env only); a public repo needs none.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - only hit in a bare env
        raise ImportError(
            "huggingface_hub is required for HF Hub push/pull; activate the `ml-dna` "
            "or `ml-rna` conda env (ADR-0002)."
        ) from exc
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir) if local_dir is not None else None,
        token=token or resolve_token(),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def push_model(repo_id: str, folder: str | Path, **kwargs: Any) -> Any:
    """Convenience wrapper: :func:`push_folder` with ``repo_type="model"``."""
    return push_folder(repo_id, folder, repo_type=REPO_TYPE_MODEL, **kwargs)


def push_dataset(repo_id: str, folder: str | Path, **kwargs: Any) -> Any:
    """Convenience wrapper: :func:`push_folder` with ``repo_type="dataset"``."""
    return push_folder(repo_id, folder, repo_type=REPO_TYPE_DATASET, **kwargs)
