"""Closed allow-list of the pinned Caduceus-PS checkpoints (ADR-0002 D2 + **A14**).

**Why a registry and why torch-free.** Until P2-12 the checkpoint identity lived as two bare
module constants in :mod:`tbox_finder.models.caduceus_backbone` (``REPO_ID`` / ``REVISION``),
``load_caduceus_ps`` took **no** repo argument, and ``_require_pinned_revision`` rejected every
revision but the one. That is the right posture for a *production* pin, but it made PRD §10.1's
pre-registered **size / effective-context ablation** unreachable — and unreachable *silently*:
a Hydra ``model.repo_id=…`` override composes cleanly, is dropped by
``train_stage1._cfg_from_mapping`` (it iterates the dataclass fields), and the run report then
echoes the code constants. The "ablation" would train the pinned production model while its gate
passed — the P2-11 ``label_source`` footgun class (CLAUDE.md §10.3).

ADR-0002 **A14** therefore *adds* two ablation-only pins and *bumps nothing*: selection goes
through the closed allow-list below, an unknown key or an unpinned revision **raises**, and the
parameter count is **measured at load** and checked against the recorded expectation rather than
asserted from the PRD.

This module carries **no ``torch`` / ``transformers`` dependency** (the loaders that consume it
do). That is deliberate and load-bearing twice over:

* ``Stage1TrainConfig.__post_init__`` can reject an unknown ``backbone`` at **compose** time on
  the login node — before ``sbatch``, before a GPU node, before a backbone download — instead of
  dying late after the model load (the ~20 GPU-h exposure class of ADR-0005 A9/A10);
* the **bare CI tier** and the laptop's torch-free ``tbox-finder-data`` env can both lock the
  allow-list, exactly as :mod:`tbox_finder.models.rc_combine` locks the RC-combination contract.

**Provenance of the numbers.** Repo ids, revisions and file manifests verified against the
Hugging Face API on 2026-07-29; all three repos ship the same ``trust_remote_code`` modeling
files (``configuration_caduceus.py``, ``modeling_caduceus.py``, ``modeling_rcps.py``), so A14 is
an add with **no env change and no new lockfile** (D2/D3 untouched). Every
``expected_param_count`` below was **measured** by loading the checkpoint's ``AutoModel`` on CPU
in ``tbox-ml-dna`` on 2026-07-29 and summing ``p.numel()`` — the production value reproduces the
independently code-pinned ``EXPECTED_PARAM_COUNT`` 7,725,312 to the integer, which is what
validates the method for the two new ones. None of them is copied from the PRD.

**Two PRD §10.1 wording corrections recorded here rather than silently absorbed** (ADR-0002 A14):

1. the real repo ids carry an ``_lr-8e-3`` suffix the PRD omits — the ids below are authoritative;
2. both smaller checkpoints are ``seqlen-1k`` pretrained, so this axis moves parameter count
   **and** pretraining context *together*. It is a joint **size×context** arm, not a pure size
   sweep, and :attr:`CaduceusCheckpoint.pretrain_seq_len` is carried per-entry so the ablation
   table can say so rather than attribute a delta to size alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: Length of a git commit SHA — the only revision form a pin may take (never ``main``).
_SHA_LEN = 40


@dataclass(frozen=True)
class CaduceusCheckpoint:
    """One pinned Caduceus-PS checkpoint: identity + the facts a run must re-derive.

    ``expected_param_count`` is an *expectation to check a measurement against*, never a value
    to report in place of one. ``d_model`` is likewise checked against the loaded config: the
    ~471k entry is ``d_model`` **118**, not 256, so a segmentation head sized to the module
    default would silently mismatch the ``2*d_model`` hidden state.
    """

    key: str
    repo_id: str
    revision: str
    d_model: int
    n_layer: int
    #: Pretraining context length of the checkpoint (**not** the scan window, which stays at the
    #: ADR-0005 D3+A3 1024/512 geometry for every leg). Caduceus is a Mamba stack with no
    #: positional table, so this is a pretraining-distribution fact, not an architectural cap.
    pretrain_seq_len: int
    #: Measured at load (CPU, ``tbox-ml-dna``, 2026-07-29); the gate compares, never asserts.
    expected_param_count: int
    role: str
    #: Approximate size as PRD §10.1 words it, for the ablation table's human-readable column.
    prd_label: str


#: The production Stage-1 backbone (PRD §6/§10.1; ADR-0002 D2). **Unchanged by A14.**
PRODUCTION_CHECKPOINT = "caduceus-ps-131k-d256-l16"

#: Closed allow-list (ADR-0002 D2 + A14). Adding an entry is an ADR change, not a config change.
CHECKPOINTS: Mapping[str, CaduceusCheckpoint] = MappingProxyType(
    {
        PRODUCTION_CHECKPOINT: CaduceusCheckpoint(
            key=PRODUCTION_CHECKPOINT,
            repo_id="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
            revision="d89eeb853136ea64da7feb3d0c8e909771b17ae6",
            d_model=256,
            n_layer=16,
            pretrain_seq_len=131_072,
            expected_param_count=7_725_312,
            role="production",
            prd_label="7.73M",
        ),
        "caduceus-ps-1k-d256-l4": CaduceusCheckpoint(
            key="caduceus-ps-1k-d256-l4",
            repo_id="kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
            revision="2de36509b76b4395a97bfa39acafeb6f17b537c5",
            d_model=256,
            n_layer=4,
            pretrain_seq_len=1_024,
            expected_param_count=1_934_592,
            role="ablation",
            prd_label="~1.93M",
        ),
        "caduceus-ps-1k-d118-l4": CaduceusCheckpoint(
            key="caduceus-ps-1k-d118-l4",
            repo_id="kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
            revision="51d88e326c0fadb0dcc63efdd7345119ce672ce5",
            d_model=118,
            n_layer=4,
            pretrain_seq_len=1_024,
            expected_param_count=470_702,
            role="ablation",
            prd_label="~471k",
        ),
    }
)

#: Stable ordering for tables and CLI help — production first, then descending size.
CHECKPOINT_KEYS: tuple[str, ...] = tuple(CHECKPOINTS)


def resolve_checkpoint(key: str) -> CaduceusCheckpoint:
    """Resolve an allow-list key → its :class:`CaduceusCheckpoint`, or raise.

    Fail-closed by construction: the allow-list is the *only* way to name a checkpoint, so an
    unknown or misspelled key cannot degrade to "load the default" — which is exactly how an
    ablation ends up training the production model and reporting a green gate.
    """
    if not isinstance(key, str):
        raise ValueError(f"backbone must be a string key; got {key!r} ({type(key).__name__})")
    spec = CHECKPOINTS.get(key)
    if spec is None:
        raise ValueError(
            f"unknown backbone {key!r}; the ADR-0002 A14 allow-list is closed: {CHECKPOINT_KEYS}. "
            "Adding a checkpoint is an ADR amendment (D2/D8 — pins are added, never bumped "
            "silently), not a config override."
        )
    return spec


def require_pinned_revision(key: str, revision: str) -> CaduceusCheckpoint:
    """Reject any revision other than ``key``'s pinned one; return the resolved checkpoint.

    ``trust_remote_code=True`` **executes** the Hub modeling code at ``revision`` on load, so an
    arbitrary revision is arbitrary remote-code execution. Re-pinning is a code change plus
    re-sign-off (ADR-0002 D2), never a runtime argument — the per-checkpoint generalisation of
    the original single-pin rule, never a relaxation of it.
    """
    spec = resolve_checkpoint(key)
    if revision != spec.revision:
        raise ValueError(
            f"revision must be the code-pinned revision {spec.revision!r} for backbone "
            f"{spec.key!r} ({spec.repo_id}); loading remote code (trust_remote_code=True) from "
            f"another revision is not allowed, got {revision!r}"
        )
    return spec


def checkpoint_for_repo_id(repo_id: str) -> CaduceusCheckpoint | None:
    """The allow-list entry whose ``repo_id`` matches, or ``None``.

    Used to **re-derive** the backbone axis of a report written before ``backbone`` was a config
    field (the 36 committed P2-06 sweep points record only ``backbone.repo_id``/``revision``).
    Re-deriving from the report's own recorded evidence is the rule; defaulting a missing axis to
    "production" on faith is what would let a differently-trained point join the table unnoticed.
    """
    for spec in CHECKPOINTS.values():
        if spec.repo_id == repo_id:
            return spec
    return None


def checkpoint_summary(spec: CaduceusCheckpoint) -> dict[str, object]:
    """The JSON-safe identity block a run report echoes for its **resolved** checkpoint."""
    return {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "d_model": int(spec.d_model),
        "n_layer": int(spec.n_layer),
        "pretrain_seq_len": int(spec.pretrain_seq_len),
        "expected_param_count": int(spec.expected_param_count),
        "role": spec.role,
        "prd_label": spec.prd_label,
    }
