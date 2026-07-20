"""P2-10a — the Stage-1 scanner: a checkpoint and an arbitrary sequence in, per-position
posteriors out.

Everything downstream of a trained Stage-1 model needs this and, until now, nothing in the
repo provided it: there was no ``torch.load``/``load_state_dict`` anywhere in ``src/``, so
``data/processed/checkpoints/stage1_production/stage1.pt`` (P2-09) could not be *used*, only
written. The P2-10 hard-negative mining rounds need it to score the §9.1 negative pools,
GATE-4 (P2-14) needs it to scan the held-out fold, and the P5 genome scan needs it at scale.

What this module is, and what it deliberately is not
----------------------------------------------------
It is the **transport**: tile → encode → forward → hand to the pinned operator. The
reduction itself is *not* here — :func:`tbox_finder.infer.reconcile.reconcile_windows` owns
it (ADR-0005 D3 + Amendment A3), and this module calls it rather than re-deriving anything.
Along-sequence locus construction (threshold, minimum span, gap-merge) is still a later
step; :func:`scan_sequence` stops at the reconciled per-position distribution.

One loop, two encoding policies
-------------------------------
:func:`scan_encoded_windows` is the *promoted* tile→forward→reconcile loop. Before this
step it existed only inside ``train_stage1.evaluate_selection_val``, and that function's own
docstring states why it must not be forked: evaluating through the deployed operator is what
makes "a config selected here is selected under the arithmetic it will be deployed under"
true. A second copy in the scanner would let the config *selected* drift from the arithmetic
*deployed* — the precise failure that docstring exists to prevent. So the trainer now
delegates here.

What is **not** shared is the encoder, on purpose:

* ``window_dataset.encode_eval_window`` (the trainer's) **raises** when a window would run
  off the record's context. That is a live guard — filter 4 of ``load_selection_val_records``
  guarantees it cannot happen, and a boundary invented mid-context would be a silent
  falsehood, not a contig end.
* :func:`encode_scan_window` (this module's) **pads**, because a scanner is handed sequences
  whose ends genuinely *are* ends — Rfam decoys are ~100–200 nt against a 1024-nt window.

Collapsing those two into one permissive encoder would delete the trainer's guard, so the
seam is drawn at the loop, not the encoder. Both encoders agree on the padding *token*:
``PAD_TOKEN_ID`` (4), which is what training's :func:`~tbox_finder.data.window_dataset.carve_window`
writes and what the backbone's own ``config.pad_token_id`` is — a scan must not present the
model a pad convention it never saw in training.

Contig ends are flagged, not hidden
-----------------------------------
PRD §6 / ADR-0005 D3: *contig ends are zero-flanked and flagged*. ``reconcile_windows``
already implements both halves — it never averages an out-of-bounds position's logits, and
it sets ``Reconciled.zero_flanked`` for every real position whose window ran off an end. This
module's job is only to pass honest ``starts`` so that geometry is computable. For a sequence
shorter than one window, every position is zero-flanked; the caller decides what that means
for downstream use, and nothing here silently drops it.

``torch`` is imported **lazily inside functions**, so this module imports in the bare CI tier
and its pure geometry/encoding surface is unit-testable without a GPU (``Stage1Segmenter``'s
backbone forward requires CUDA — ADR-0002 A2 C2 — but ``logits_from_hidden`` does not, which
is the seam the stub-model tests use).

PRD §6, §9.1; ADR-0005 D3 + A3, D15.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder.data.window_dataset import (
    PAD_TOKEN_ID,
    STRIDE_NT,
    WINDOW_NT,
    encode_bases,
    tile_windows,
)
from tbox_finder.infer.reconcile import NUM_CLASSES, Reconciled, reconcile_windows

SCHEMA_VERSION = "1.0"
STEP = "P2-10a"

#: The P2-09 round-0 production checkpoint. A bare ``state_dict`` — ``torch.save`` was called
#: on ``segmenter.state_dict()`` (``train_stage1.py``), so it carries **no architecture
#: metadata**. :func:`load_stage1_checkpoint` must therefore re-supply the geometry, and the
#: strict key/shape check is what stands between a wrong guess and a silently-wrong model.
DEFAULT_CHECKPOINT = Path("data/processed/checkpoints/stage1_production/stage1.pt")

#: Windows per forward. Scan-time only; it changes speed and peak VRAM, never the result —
#: reconciliation is over the full window set regardless of how it was batched.
DEFAULT_SCAN_BATCH = 8


class ScanError(ValueError):
    """Raised on malformed scan input or a checkpoint that does not fit the architecture."""


# ═════════════════════════════════════════════════════════════════════════════
# Encoding — pad-aware, contig ends are real ends
# ═════════════════════════════════════════════════════════════════════════════
def encode_scan_window(seq: str, start: int, *, window: int = WINDOW_NT) -> np.ndarray:
    """Encode ``[start, start + window)`` of ``seq`` to token ids, padding past either end.

    The scan counterpart of ``window_dataset.encode_eval_window``. Where that one *raises*
    on an overrun (its caller guarantees the window is interior), this one pads with
    ``PAD_TOKEN_ID`` — matching training's ``carve_window``, which fills the whole window
    with ``PAD_TOKEN_ID`` before writing the real slice into it, and matching the backbone's
    ``config.pad_token_id``.

    A window that covers **no** real position is an error rather than an all-pad window: it
    would contribute a distribution over nothing, and ``reconcile_windows`` rejects it too.
    Raising here names the offending offset while it is still cheap to attribute.
    """
    if window <= 0:
        raise ScanError(f"window must be positive, got {window}")
    seq_len = len(seq)
    if seq_len <= 0:
        raise ScanError("cannot scan an empty sequence")
    start = int(start)

    lo, hi = max(start, 0), min(start + window, seq_len)
    if hi <= lo:
        raise ScanError(
            f"scan window [{start}, {start + window}) covers no position of "
            f"[0, {seq_len}) — it would score no DNA at all"
        )

    ids = np.full(window, PAD_TOKEN_ID, dtype=np.int16)
    ids[lo - start : hi - start] = encode_bases(seq[lo:hi])
    return ids


def scan_window_ids(
    seq: str,
    *,
    window: int = WINDOW_NT,
    stride: int = STRIDE_NT,
) -> tuple[np.ndarray, list[int]]:
    """Tile ``seq`` at the pinned scan geometry → ``((n_windows, window) int16, starts)``.

    ``tile_windows`` is tail-anchored, so a sequence longer than one window is tiled
    entirely in-bounds and nothing is padded; a shorter one yields the single window
    ``[0, window)``, whose tail is padded and which ``reconcile_windows`` will flag.
    """
    starts = tile_windows(len(seq), window=window, stride=stride)
    ids = np.stack([encode_scan_window(seq, s, window=window) for s in starts])
    return ids, starts


# ═════════════════════════════════════════════════════════════════════════════
# The promoted loop — forward every window, hand the logits to the pinned operator
# ═════════════════════════════════════════════════════════════════════════════
def scan_encoded_windows(
    model: Any,
    window_ids: Any,
    starts: Sequence[int],
    seq_len: int,
    *,
    device: Any,
    batch_size: int = DEFAULT_SCAN_BATCH,
) -> Reconciled:
    """Forward pre-encoded windows and reconcile them (ADR-0005 D3 + A3).

    This is the single implementation of the tile→forward→reconcile loop; both
    :func:`scan_sequence` and ``train_stage1.evaluate_selection_val`` route through it, so
    the arithmetic a config is *selected* under and the arithmetic it is *deployed* under
    cannot drift apart.

    ``window_ids`` is ``(n_windows, window)`` of token ids — whatever encoder the caller's
    honesty policy demands (see the module docstring). ``starts`` are the matching offsets;
    ``seq_len`` the underlying sequence length. Windows are forwarded in ``batch_size``
    chunks, which affects speed and peak VRAM but never the result.

    ⚠ ``model.eval()`` + ``torch.no_grad()``, restoring whatever mode the model was already
    in. Restoring the *observed* state rather than a hardcoded one is what makes this safe to
    nest inside a caller that has already switched to eval (the trainer does): the inner
    restore is then a no-op instead of switching training back on mid-loop. ``eval()`` is
    load-bearing on its own terms — ``dropout`` is a swept axis, and scoring with dropout
    live would inject noise into the very comparison a sweep is making. ``no_grad`` rather
    than ``inference_mode`` follows the trainer's precedent: context7 (``/pytorch/pytorch``)
    is explicit that ``inference_mode`` tensors cannot re-enter autograd, and this runs
    inside a training entrypoint.
    """
    import torch  # lazy — keeps the bare CI tier importable

    ids = np.asarray(window_ids)
    if ids.ndim != 2:
        raise ScanError(f"window_ids must be (n_windows, window)-shaped, got shape={ids.shape}")
    if ids.shape[0] != len(starts):
        raise ScanError(
            f"starts carries {len(starts)} offsets but window_ids carries {ids.shape[0]} windows"
        )
    if batch_size <= 0:
        raise ScanError(f"batch_size must be positive, got {batch_size}")
    if ids.shape[0] == 0:
        # `scan_sequence` cannot reach this (`tile_windows` always emits >= 1 window), but
        # this function is public and the trainer calls it directly. Without the guard the
        # batch loop simply never runs and `torch.cat([])` dies with "expected a non-empty
        # list of Tensors" — an error naming neither the caller's mistake nor this module.
        raise ScanError("window_ids carries no windows; there is nothing to score")

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        chunks = []
        with torch.no_grad():
            for i in range(0, ids.shape[0], batch_size):
                batch = torch.from_numpy(ids[i : i + batch_size].astype(np.int64))
                chunks.append(model(input_ids=batch.to(device)).float().cpu())
        logits = torch.cat(chunks, dim=0)
    finally:
        if was_training:
            model.train()

    if logits.shape[-1] != NUM_CLASSES:
        raise ScanError(
            f"model emitted {logits.shape[-1]} classes, expected {NUM_CLASSES} "
            f"(labels.CLASS_ORDER)"
        )
    return reconcile_windows(logits, np.asarray(starts), seq_len)


def scan_sequence(
    model: Any,
    seq: str,
    *,
    device: Any,
    window: int = WINDOW_NT,
    stride: int = STRIDE_NT,
    batch_size: int = DEFAULT_SCAN_BATCH,
) -> Reconciled:
    """Scan one sequence end-to-end → per-position reconciled posteriors.

    The scanner entry point: tiles at the pinned geometry, pads contig ends, forwards, and
    reconciles. ``Reconciled.log_probs`` is an ``(len(seq), 8)`` proper distribution — what
    the Stage-1 threshold (ADR-0005 D3) and the §11 recalibration/ECE stack consume — and
    ``Reconciled.zero_flanked`` marks every position whose context included synthetic pad.
    """
    ids, starts = scan_window_ids(seq, window=window, stride=stride)
    return scan_encoded_windows(model, ids, starts, len(seq), device=device, batch_size=batch_size)


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint loading — the architecture is re-supplied, then verified key-by-key
# ═════════════════════════════════════════════════════════════════════════════
def load_stage1_checkpoint(
    path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | None = None,
    rc_combine: str | None = None,
    use_crf: bool = False,
    dropout: float = 0.0,
    backbone: Any = None,
) -> Any:
    """Rebuild the Stage-1 segmenter and load ``path`` into it, verifying every key.

    ``train_stage1`` saves ``segmenter.state_dict()`` — weights only, **no architecture
    metadata** — so the geometry (``rc_combine`` mode, CRF head or linear) has to be
    re-supplied here and could in principle be guessed wrong. The defence is that a wrong
    guess is *structurally* detectable and is therefore made fatal:

    * a wrong ``rc_combine`` mode changes ``RCCombine.output_dim`` and so the head's weight
      **shapes** → ``load_state_dict`` raises on the size mismatch (this happens even at
      ``strict=False``);
    * ``use_crf=True`` against a linear-head checkpoint adds CRF parameters → missing keys.

    So the load runs at ``strict=False`` and this function raises on any non-empty
    missing/unexpected list itself. That is not weaker than ``strict=True`` — it is the same
    hard failure (shape mismatches still raise inside ``load_state_dict``) plus the actual
    key lists in the message, which is what makes a mismatch diagnosable rather than merely
    fatal. ``dropout`` is deliberately *not* guarded: it adds no parameters, and the model is
    returned in ``.eval()`` where dropout is inert.

    ``rc_combine`` defaults to the code-pinned ``DEFAULT_RC_COMBINE`` ("concat"), which is
    what ``conf/model/caduceus_stage1.yaml`` ships and what P2-09 trained under. ``backbone``
    may be passed to reuse an already-loaded one; otherwise the pinned-revision Caduceus-PS
    is loaded (construction works on CPU — only a *forward* needs CUDA).
    """
    import torch  # lazy

    from tbox_finder.models.caduceus_backbone import load_caduceus_ps
    from tbox_finder.models.rc_combine import DEFAULT_RC_COMBINE
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter

    ckpt = Path(path)
    if not ckpt.is_file():
        raise ScanError(f"no Stage-1 checkpoint at {ckpt}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if backbone is None:
        backbone = load_caduceus_ps(device=device)

    segmenter = Stage1Segmenter(
        backbone=backbone,
        rc_combine=DEFAULT_RC_COMBINE if rc_combine is None else rc_combine,
        use_crf=use_crf,
        dropout=dropout,
    ).to(device)

    # weights_only=True is the torch>=2.6 default (context7 /pytorch/pytorch, serialization
    # notes) and is correct for a pure state_dict; naming it keeps the intent legible and the
    # behaviour stable if the default ever moves again.
    state = torch.load(ckpt, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise ScanError(
            f"{ckpt} does not hold a state_dict (got {type(state).__name__}); "
            f"train_stage1 saves segmenter.state_dict()"
        )

    incompatible = segmenter.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise ScanError(
            f"{ckpt} does not fit the rebuilt Stage1Segmenter "
            f"(rc_combine={rc_combine or DEFAULT_RC_COMBINE!r}, use_crf={use_crf}): "
            f"{len(missing)} missing key(s) {missing[:5]}, "
            f"{len(unexpected)} unexpected key(s) {unexpected[:5]} — "
            f"the checkpoint carries no architecture metadata, so a wrong geometry here "
            f"would otherwise load a silently-wrong model"
        )

    segmenter.eval()
    return segmenter


def checkpoint_key_report(
    path: str | Path = DEFAULT_CHECKPOINT,
    *,
    rc_combine: str | None = None,
    use_crf: bool = False,
) -> dict[str, Any]:
    """Key-level fit of a checkpoint against a freshly built segmenter, as data.

    The reporting counterpart of :func:`load_stage1_checkpoint`'s guard: it returns the
    counts instead of raising, so a validation gate can record ``n_missing``/``n_unexpected``
    rather than merely observing that nothing blew up. Built on CPU with ``backbone=None``
    (the head + RC-combine are the parameters the architecture guess actually controls, and
    they need no CUDA and no HF download) and never returns the weights.

    ⚠ **Shapes are compared, not just key names**, because a key-set comparison is not
    sufficient in general. It happens to be sufficient for the ``concat``/``gate`` pair —
    ``RCCombine`` registers ``gate_logit`` only in ``gate`` mode
    (``models/stage1_segmenter.py``), so that mismatch shows up as a missing key *as well as*
    as a shape difference on ``head.classifier.weight`` (``(8, 512)`` vs ``(8, 256)``);
    measured against the P2-09 checkpoint, ``rc_combine="gate"`` yields
    ``n_head_missing=1`` **and** ``n_shape_mismatch=1``. The mismatches key names cannot
    catch are the ones that keep the *same* parameter set and change only its extent: a
    checkpoint from a different-``d_model`` backbone, or from a head with a different class
    count. Those differ solely in shape, and without this comparison they would load as a
    clean 0/0 fit — the vacuous green this function exists to make impossible.
    """
    import torch  # lazy

    from tbox_finder.models.rc_combine import DEFAULT_RC_COMBINE
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter

    ckpt = Path(path)
    if not ckpt.is_file():
        raise ScanError(f"no Stage-1 checkpoint at {ckpt}")

    mode = DEFAULT_RC_COMBINE if rc_combine is None else rc_combine
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    # backbone=None keeps this cheap: the head + RC-combine are the parameters whose shapes
    # the architecture guess actually controls, and they are constructible without CUDA.
    skeleton = Stage1Segmenter(backbone=None, rc_combine=mode, use_crf=use_crf)
    own_state = skeleton.state_dict()
    own, have = set(own_state), set(state)
    head_missing = sorted(own - have)
    head_unexpected = sorted(k for k in have - own if not k.startswith("backbone."))
    shape_mismatch = sorted(
        f"{k}: checkpoint{tuple(state[k].shape)} vs model{tuple(own_state[k].shape)}"
        for k in own & have
        if tuple(state[k].shape) != tuple(own_state[k].shape)
    )
    return {
        "checkpoint": str(ckpt),
        "rc_combine": mode,
        "use_crf": bool(use_crf),
        "n_checkpoint_keys": len(have),
        "n_head_missing": len(head_missing),
        "n_head_unexpected": len(head_unexpected),
        "n_shape_mismatch": len(shape_mismatch),
        "head_missing": head_missing,
        "head_unexpected": head_unexpected,
        "shape_mismatch": shape_mismatch,
        "fits": not (head_missing or head_unexpected or shape_mismatch),
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
    }


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SCAN_BATCH",
    "ScanError",
    "checkpoint_key_report",
    "encode_scan_window",
    "load_stage1_checkpoint",
    "scan_encoded_windows",
    "scan_sequence",
    "scan_window_ids",
]
