"""P1-09 — Segmentation-head fallback: frozen-embedding + per-position linear/CRF probe.

The Stage-1 default is an **end-to-end** fine-tune of the P1-04
:class:`~tbox_finder.models.stage1_segmenter.Stage1Segmenter` (backbone + head trained
together; the P1-07 binding go/no-go). PRD §10.1 pins a **fallback** for the case where
that end-to-end fine-tune underperforms:

    *"Head … a per-position token-classification head (Linear, optionally + CRF for
    boundary coherence) over the L×d_model hidden states. Fallback if end-to-end
    fine-tuning underperforms: frozen-embedding + per-position linear/CRF probe."*
    — PRD §10.1; §18.2 mitigation *"build head + frozen-embedding fallback; validate on
    fixtures early (P1)."*

This module **builds and smoke-tests that fallback path** so P2 has it ready if the
end-to-end fine-tune underperforms. It **freezes** the Caduceus-PS backbone, caches its
per-position hidden states once, and trains **only** the P1-03
:class:`~tbox_finder.models.seg_head.SegmentationHead` (Linear by default, optional
in-repo linear-chain CRF) on those frozen embeddings — i.e. the head *in probe mode*.
It is the per-**position** (per-nucleotide) 8-class counterpart of the P1-05
window-level binary linear probe (:mod:`tbox_finder.probes.frozen_linear_probe`).

**Non-gated by construction.** The P1-07 **GO** verdict means this fallback is not
*used* for P2 training, but PRD §18.2 requires the path be *built + smoke-tested* now.
Neither PRD §10.1/§18.2 nor ADR-0002 pins any numeric threshold on the fallback build:
the deliverable is *"produces a valid length-L 8-class segmentation and a **reported**
per-nt F1"* — a **fallback baseline that is measured and disclosed, never a pass/fail
gate**. (ADR-0002 D7's directional go/no-go thresholds belong to the *end-to-end* smoke
P1-07, not to this frozen-embedding probe.)

**Honesty (§10.3).** Like P1-07, the probe trains **and** grades on the same 300
leave-clade-out held-out loci (``data.eval_split == "fine_tune_set"``). This is a
*frozen-embedding learnability / expressivity* measurement — "can a linear/CRF head over
the frozen human-pretrained embeddings paint prokaryotic T-box elements?" — **not** a
within-smoke generalization test. True generalization is graded at P2 on the full split;
the loci are held out from the P2 training corpus (§9.2), so this measurement does not
contaminate P2, but a high per-nt F1 here reflects frozen-feature *fit*, not transfer
generalization, and must not be read as such.

**Single-env torch.** The pinned input is *the P1-03 seg head (a torch ``nn.Module``,
Linear + optional in-repo CRF) in probe mode*, so both stages run under ``tbox-ml-dna``:
``--stage extract`` (frozen backbone forward → per-position embeddings ``.npz`` cache;
CUDA, ADR-0002 A2 C2) then ``--stage probe`` (train the head on the cache → report; CPU
is sufficient — a small ``Linear`` + CRF). This resolves the step's provisional two-env
/ scikit-learn hedge to single-env because scikit-learn cannot express the CRF variant
and the head is torch. All heavy imports are lazy so the module imports in a bare env.

Reuses the P1-06 loader, tokenizer, determinism and metrics from
:mod:`tbox_finder.train.stage1_smoke` (same smoke set, same per-nt-class metric unit) and
the P1-04 backbone-free frozen-embedding entry
(:meth:`~tbox_finder.models.stage1_segmenter.Stage1Segmenter.logits_from_hidden` /
``decode_from_hidden`` / ``loss_from_hidden``).

Rule/Entry: :func:`run_seg_probe`.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import provenance
from tbox_finder.labels import CLASS_ORDER, CORE_ELEMENTS
from tbox_finder.models.caduceus_backbone import D_MODEL, PRETRAINING_DOMAIN, REPO_ID, REVISION

# Reuse the P1-06 loader / tokenizer / determinism / focal loss / metrics so the fallback
# probe consumes the **same** smoke set with the **same** positional contract and grades
# the **same** commensurable per-nt-class unit (no re-derivation).
from tbox_finder.train import stage1_smoke as smoke

# --------------------------------------------------------------------------------------
# Constants (frozen in code, not config).
# --------------------------------------------------------------------------------------
SCHEMA_VERSION = "1"
STEP = "P1-09"
GENERATED_BY = "src/tbox_finder/probes/frozen_seg_probe.py"
#: This fallback build is pinned in the PRD, not a distinct ADR decision; ADR-0002 is
#: referenced for the frozen-backbone + directionality-preserving RC-combine context.
PRD_SECTIONS = ("§10.1", "§18.2")
ADR = "ADR-0002"

#: Frozen Caduceus-PS per-position hidden width == 2*d_model (fwd‖rc channels), single-
#: sourced from :data:`tbox_finder.models.caduceus_backbone.D_MODEL`.
EMB_DIM: int = 2 * D_MODEL

#: Segmentation classes (ADR-0004 D1), single-sourced from :data:`labels.CLASS_ORDER`.
NUM_CLASSES: int = len(CLASS_ORDER)

#: labels.npy pad / seg-head cross-entropy ignore value (ADR-0005). Local literal so the
#: module imports bare; a drift-guard test asserts it equals ``stage1_smoke.IGNORE_INDEX``.
IGNORE_INDEX = -100

# Defaults (repo-relative; overridable on the CLI).
DEFAULT_WINDOWS = "data/interim/p1_seg_smoke/windows.parquet"
DEFAULT_LABELS = "data/interim/p1_seg_smoke/labels.npy"
DEFAULT_EMB_CACHE = "data/interim/probes/frozen_seg_embeddings.npz"
DEFAULT_OUT = "reports/p1/frozen_seg_probe.json"

# Probe hyper-parameters (frozen defaults; deterministic §8.3). Only the head trains — the
# backbone is frozen — so more epochs than the P1-07 end-to-end smoke are cheap (a tiny
# Linear over cached embeddings, CPU-fine). NON-gated: exact values are not load-bearing.
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1.0e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRAD_CLIP = 1.0
#: Focal-loss γ for the Linear probe (PRD §11 pins per-nt focal CE for the dense-background
#: regime; γ=0 recovers plain CE). The CRF probe (``use_crf``) uses the CRF NLL instead.
DEFAULT_GAMMA = 2.0
DEFAULT_RC_COMBINE = "concat"  # directionality-preserving identity (P1-04); "mean" is code-rejected

ENV_LOCK = "envs/ml-dna.conda-lock.yml"


# --------------------------------------------------------------------------------------
# Run configuration (a plain dataclass so ``run_seg_probe`` is callable without a CLI).
# --------------------------------------------------------------------------------------
@dataclass
class ProbeConfig:
    """Fully-resolved knobs for the frozen-embedding seg-head probe (§8.3 determinism)."""

    seed: int = DEFAULT_SEED
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip: float = DEFAULT_GRAD_CLIP
    gamma: float = DEFAULT_GAMMA
    rc_combine: str = DEFAULT_RC_COMBINE
    use_crf: bool = False
    dropout: float = 0.0  # 0.0 ⇒ deterministic inference
    device: str | None = None  # None ⇒ cuda if available else cpu (extract needs cuda)
    windows_parquet: str = DEFAULT_WINDOWS
    labels_npy: str = DEFAULT_LABELS
    emb_cache: str = DEFAULT_EMB_CACHE
    report_path: str = DEFAULT_OUT
    reuse_embeddings: bool = False

    def sanitized_knobs(self) -> dict[str, Any]:
        """The subset of knobs that belong in the report/provenance (no paths)."""
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "gamma": self.gamma,
            "rc_combine": self.rc_combine,
            "use_crf": self.use_crf,
            "dropout": self.dropout,
        }


# --------------------------------------------------------------------------------------
# Pure stdlib helpers (bare-testable — no numpy / torch).
# --------------------------------------------------------------------------------------
def _sanitize(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` so the JSON is strict (no NaN/Inf)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _finite_number(v: Any) -> bool:
    """True iff ``v`` is a real, finite number (bools rejected)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _prob01(v: Any) -> bool:
    """True iff ``v`` is a finite number in the closed unit interval."""
    return _finite_number(v) and 0.0 <= float(v) <= 1.0


def _nullable_prob01(v: Any) -> bool:
    """True iff ``v`` is ``None`` (an absent-class NaN, sanitized) or a finite prob in [0,1]."""
    return v is None or _prob01(v)


def _bad_bool(value: Any, expected: bool) -> bool:
    """True iff ``value`` is not the exact boolean ``expected`` (int 0/1 rejected)."""
    return not (isinstance(value, bool) and value is expected)


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of schema problems for a frozen-seg-probe report (empty ⇒ valid).

    Never raises. Fails **closed** on the anti-fabrication / honesty invariants (§10.3):
    a per-nt F1 outside [0, 1], a missing / extra class in the 8-class table, the wrong
    core-element set, a report that claims to be a binding gate, a non-frozen backbone,
    a broken §9.2 held-out flag or §10.3 ``eval_split`` disclosure, a decoded segmentation
    that is not a valid 8-class path, or a stale backbone revision. There is **no**
    verdict/threshold check — this fallback is non-gated by construction (PRD §10.1).
    """
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if report.get("step") != STEP:
        errors.append(f"step must be {STEP!r}")
    if _bad_bool(report.get("measured"), True):
        errors.append("measured must be boolean True")
    if _bad_bool(report.get("binding"), False):
        errors.append("binding must be boolean False (non-gated fallback baseline, PRD §10.1)")
    if _bad_bool(report.get("frozen_backbone"), True):
        errors.append("frozen_backbone must be boolean True (the backbone is not fine-tuned)")

    backbone = report.get("backbone")
    if not isinstance(backbone, dict):
        errors.append("backbone block missing")
    elif backbone.get("revision") != REVISION:
        errors.append("backbone.revision must be the pinned Caduceus-PS commit")

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics block missing")
    else:
        per_class = metrics.get("per_class_f1")
        if not isinstance(per_class, dict) or set(per_class) != set(CLASS_ORDER):
            errors.append(f"metrics.per_class_f1 must have exactly the 8 classes {CLASS_ORDER}")
        else:
            for name, val in per_class.items():
                if not _nullable_prob01(val):
                    errors.append(f"metrics.per_class_f1[{name}] must be null or in [0, 1]")
        per_elem = metrics.get("per_element_f1")
        if not isinstance(per_elem, dict) or set(per_elem) != set(CORE_ELEMENTS):
            errors.append(
                f"metrics.per_element_f1 must have exactly the core elements {CORE_ELEMENTS}"
            )
        else:
            for name, val in per_elem.items():
                if not _nullable_prob01(val):
                    errors.append(f"metrics.per_element_f1[{name}] must be null or in [0, 1]")
        for key in ("min_core_f1", "macro_f1", "micro_f1"):
            if not _nullable_prob01(metrics.get(key)):
                errors.append(f"metrics.{key} must be null or a finite number in [0, 1]")

    seg = report.get("segmentation_validity")
    if not isinstance(seg, dict):
        errors.append("segmentation_validity block missing")
    else:
        # The step's validation gate: every window decodes to a valid length-L 8-class path.
        if _bad_bool(seg.get("all_windows_valid_8class"), True):
            errors.append("segmentation_validity.all_windows_valid_8class must be boolean True")
        lo, hi = seg.get("min_pred_class"), seg.get("max_pred_class")
        if not (isinstance(lo, int) and not isinstance(lo, bool) and lo >= 0):
            errors.append("segmentation_validity.min_pred_class must be an int >= 0")
        if not (isinstance(hi, int) and not isinstance(hi, bool) and hi < NUM_CLASSES):
            errors.append(f"segmentation_validity.max_pred_class must be an int < {NUM_CLASSES}")

    data = report.get("data")
    if not isinstance(data, dict):
        errors.append("data block missing")
    else:
        if _bad_bool(data.get("held_out_only"), True):
            errors.append("data.held_out_only must be boolean True (§9.2 leave-clade-out fold)")
        if data.get("eval_split") != "fine_tune_set":
            errors.append("data.eval_split must be 'fine_tune_set' (§10.3 honest disclosure)")

    gate = report.get("gate")
    if not isinstance(gate, dict):
        errors.append("gate block missing")
    else:
        if _bad_bool(gate.get("binding"), False):
            errors.append("gate.binding must be boolean False")
        if _bad_bool(gate.get("gated"), False):
            errors.append("gate.gated must be boolean False (non-gated fallback baseline)")
    return errors


# --------------------------------------------------------------------------------------
# Stage A — frozen per-position embedding extraction (lazy torch/transformers; GPU).
# --------------------------------------------------------------------------------------
def extract_seg_embeddings(
    windows: Sequence[tuple[str, Sequence[int], int]],
    ids_of,
    *,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    log=lambda _m: None,
):
    """Frozen Caduceus-PS **per-position** hidden states for each smoke window.

    Loads the code-pinned backbone in eval mode, freezes every parameter, and runs a
    single-sequence (batch=1) forward per window under ``torch.no_grad()`` — so each
    window's hidden state covers exactly its real positions (no pad-token contamination).
    Deterministic: no dropout in eval, no sampling, TF32/cudnn off (§8.3).

    Returns ``(hidden_flat, labels_flat, lengths)``:
      - ``hidden_flat`` ``(sum_L, EMB_DIM)`` float32 — every window's valid positions,
        concatenated;
      - ``labels_flat`` ``(sum_L,)`` int16 — the matching per-nt class indices (no pad);
      - ``lengths`` ``(N,)`` int32 — per-window valid length, to re-slice ``hidden_flat``.
    """
    import numpy as np  # lazy
    import torch  # lazy

    from tbox_finder.models.caduceus_backbone import _hidden_states, load_caduceus_ps

    smoke.set_determinism(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_caduceus_ps(device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    n = len(windows)
    hidden_parts: list[Any] = []
    labels_parts: list[list[int]] = []
    lengths: list[int] = []
    with torch.no_grad():
        for i, (seq, labels, valid_len) in enumerate(windows):
            x = torch.tensor([ids_of(seq)], device=device)  # (1, L)
            h = _hidden_states(model(input_ids=x))  # (1, L, EMB_DIM)
            h = h.squeeze(0).float().cpu().numpy()  # (L, EMB_DIM)
            if h.shape != (valid_len, EMB_DIM):
                raise ValueError(
                    f"window {i}: hidden {h.shape} != (valid_len {valid_len}, {EMB_DIM})"
                )
            hidden_parts.append(h)
            labels_parts.append([int(v) for v in labels])
            lengths.append(int(valid_len))
            if (i + 1) % 50 == 0 or (i + 1) == n:
                log(f"  extracted {i + 1}/{n} windows")

    hidden_flat = np.concatenate(hidden_parts, axis=0).astype(np.float32)
    labels_flat = np.array([v for row in labels_parts for v in row], dtype=np.int16)
    return hidden_flat, labels_flat, np.array(lengths, dtype=np.int32)


def run_extract(
    *,
    windows_parquet: str | Path = DEFAULT_WINDOWS,
    labels_npy: str | Path = DEFAULT_LABELS,
    emb_cache: str | Path = DEFAULT_EMB_CACHE,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    log=print,
) -> Path:
    """Stage A: load the P1-06 smoke set, extract frozen embeddings, write the ``.npz`` cache."""
    import numpy as np  # lazy

    from tbox_finder.models.caduceus_backbone import load_tokenizer

    windows = smoke.load_smoke_windows(windows_parquet, labels_npy)
    ids_of = smoke._make_ids_of(load_tokenizer())
    log(f"[{STEP}] extracting frozen embeddings for {len(windows)} held-out windows")
    hidden_flat, labels_flat, lengths = extract_seg_embeddings(
        windows, ids_of, device=device, seed=seed, log=log
    )

    # Metadata pinned into the cache so run_probe validates + reports what was ACTUALLY
    # embedded (revision / emb-dim / seed / source hashes) — a stale cache is detectable.
    extract_inputs = {}
    for path in (windows_parquet, labels_npy):
        try:
            extract_inputs[str(path)] = provenance.sha256_file(path)
        except Exception:
            extract_inputs[str(path)] = "unavailable"
    meta = {
        "revision": REVISION,
        "backbone": REPO_ID,
        "emb_dim": EMB_DIM,
        "seed": int(seed),
        "pooling": "per-position (no pooling)",
        "extract_inputs": extract_inputs,
    }

    emb_cache = Path(emb_cache)
    emb_cache.parent.mkdir(parents=True, exist_ok=True)
    # Only numeric + fixed-width Unicode arrays, so the cache loads with allow_pickle=False.
    np.savez(
        emb_cache,
        hidden_flat=hidden_flat,
        labels_flat=labels_flat,
        lengths=lengths,
        seed=np.int64(seed),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    log(
        f"[{STEP}] wrote embeddings cache: {emb_cache} "
        f"(hidden {hidden_flat.shape}, {len(lengths)} windows)"
    )
    return emb_cache


def _unpack_windows(hidden_flat, labels_flat, lengths):
    """Re-slice the flat cache into ``[(hidden_i (L,EMB_DIM), labels_i (L,)), ...]``."""
    windows = []
    off = 0
    for length in lengths:
        length = int(length)
        windows.append((hidden_flat[off : off + length], labels_flat[off : off + length]))
        off += length
    if off != hidden_flat.shape[0]:
        raise ValueError(f"lengths sum {off} != hidden rows {hidden_flat.shape[0]} (corrupt cache)")
    return windows


# --------------------------------------------------------------------------------------
# Stage B — train the P1-03 head on the frozen cache; decode + grade (lazy torch; CPU).
# --------------------------------------------------------------------------------------
def build_frozen_probe(cfg: ProbeConfig):
    """The P1-04 segmenter with **no backbone** — only the RC-combine + P1-03 head train."""
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter

    # backbone=None ⇒ forward(input_ids=…) is disabled; we drive the head via the frozen-
    # embedding entry (logits_from_hidden / decode_from_hidden / loss_from_hidden). With
    # rc_combine="concat" the RC-combine is a parameter-free identity, so the ONLY trainable
    # parameters are the P1-03 SegmentationHead (Linear + optional CRF) — the head "in probe
    # mode" over frozen embeddings.
    return Stage1Segmenter(
        backbone=None,
        rc_combine=cfg.rc_combine,
        use_crf=cfg.use_crf,
        dropout=cfg.dropout,
    )


def train_frozen_head(segmenter, windows, cfg: ProbeConfig, *, log=print):
    """Train the seg head on frozen per-position embeddings (backbone frozen).

    Batch=1 per window (variable length, no pad/mask, deterministic). The Linear probe
    uses per-nt **focal** cross-entropy (γ=``cfg.gamma``; PRD §11, avoids collapse to the
    dominant background class); the CRF probe uses the CRF negative log-likelihood.
    """
    import torch  # lazy

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    segmenter = segmenter.to(device)
    segmenter.train()
    params = [p for p in segmenter.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("frozen probe has no trainable parameters (expected the seg head)")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        order = smoke._epoch_order(len(windows), cfg.seed, epoch)
        running = 0.0
        for idx in order:
            hidden_np, labels_np = windows[idx]
            h = torch.as_tensor(hidden_np, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.as_tensor(labels_np, dtype=torch.long, device=device).unsqueeze(0)
            if cfg.use_crf:
                loss = segmenter.loss_from_hidden(h, y)  # CRF NLL
            else:
                logits = segmenter.logits_from_hidden(h)  # (1, L, 8)
                loss = smoke.focal_cross_entropy(logits, y, gamma=cfg.gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            running += float(loss.detach())
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epochs:
            log(f"  epoch {epoch + 1}/{cfg.epochs} loss={running / max(len(windows), 1):.4f}")
    return segmenter


def evaluate_frozen_head(segmenter, windows) -> tuple[list[int], list[int], dict[str, int]]:
    """Decode per-nt predictions over every window; return flat ``(y_true, y_pred)`` + validity.

    Each window decodes (CRF Viterbi else argmax) to a length-L path of class indices; we
    assert every prediction is a valid 8-class index and the path length matches L (the
    step's validation gate), then flatten to the ``metrics`` input contract.
    """
    import torch  # lazy

    device = next(segmenter.parameters()).device
    segmenter.eval()
    true_rows: list[list[int]] = []
    pred_rows: list[list[int]] = []
    all_valid = True
    lo, hi = NUM_CLASSES, -1
    with torch.no_grad():
        for hidden_np, labels_np in windows:
            h = torch.as_tensor(hidden_np, dtype=torch.float32, device=device).unsqueeze(0)
            pred = segmenter.decode_from_hidden(h)[0]  # length-L class path
            labels = [int(v) for v in labels_np.tolist()]
            if len(pred) != len(labels) or any(not (0 <= int(p) < NUM_CLASSES) for p in pred):
                all_valid = False
            for p in pred:
                lo = min(lo, int(p))
                hi = max(hi, int(p))
            true_rows.append(labels)
            pred_rows.append([int(p) for p in pred])
    y_true, y_pred = smoke.flatten_valid(true_rows, pred_rows)
    validity = {
        "all_windows_valid_8class": bool(all_valid),
        "n_windows_decoded": len(pred_rows),
        "min_pred_class": int(lo) if hi >= 0 else 0,
        "max_pred_class": int(hi) if hi >= 0 else 0,
    }
    return y_true, y_pred, validity


def _env_block() -> dict[str, Any]:
    """Best-effort runtime env fingerprint (never raises)."""
    block: dict[str, Any] = {
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    for name in ("numpy", "torch", "transformers"):
        try:
            mod = __import__(name)
            block[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            block[name] = None
    return block


def build_seg_report(
    *,
    metrics_block: Mapping[str, Any],
    validity: Mapping[str, Any],
    config: Mapping[str, Any],
    n_windows: int,
    n_valid_positions: int,
    windows_parquet: str | Path,
    labels_npy: str | Path,
    emb_meta: Mapping[str, Any],
    measured: bool = True,
) -> dict[str, Any]:
    """Assemble the ``frozen_seg_probe.json`` fallback-baseline report (non-gated)."""
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "prd_sections": list(PRD_SECTIONS),
        "adr": ADR,
        "measured": bool(measured),
        "binding": False,  # non-gated fallback baseline (PRD §10.1)
        "frozen_backbone": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "backbone": {
            "repo_id": REPO_ID,
            # Sourced from the CACHE metadata (what actually produced the embeddings), NOT
            # the module constant — so validate_report's `revision == REVISION` check bites
            # at runtime: a revision-less / foreign cache yields None here → the validator
            # fires (fail-closed), instead of the report tautologically self-certifying the
            # pin it was written from.
            "revision": emb_meta.get("revision"),
            "pretraining_domain": PRETRAINING_DOMAIN,
        },
        "config": dict(config),
        "data": {
            "windows_parquet": str(windows_parquet),
            "labels_npy": str(labels_npy),
            "n_windows": int(n_windows),
            "n_valid_positions": int(n_valid_positions),
            "held_out_only": True,  # P1-06 smoke set is the leave-clade-out held-out fold (§9.2)
            # HONESTY (§10.3): trains AND grades on the same 300 held-out loci — a
            # frozen-embedding learnability/expressivity measurement, NOT a within-smoke
            # generalization test. Generalization is graded at P2 on the full split; the
            # loci are held out from the P2 corpus (§9.2) so this does not contaminate P2.
            "eval_split": "fine_tune_set",
        },
        "metrics": dict(metrics_block),
        "segmentation_validity": dict(validity),
        "gate": {
            "binding": False,
            "gated": False,
            "kind": "fallback_baseline",
            "note": (
                "Non-gated frozen-embedding + per-position linear/CRF seg-head fallback "
                "(PRD §10.1/§18.2). Per-nt F1 is REPORTED as a fallback baseline; there is "
                "no pass/fail threshold. ADR-0002 D7's go/no-go thresholds belong to the "
                "end-to-end P1-07 smoke, not to this frozen-embedding probe."
            ),
        },
        "embedding": {
            "backbone": emb_meta.get("backbone", REPO_ID),
            # No REVISION default: the embedding block must reflect the cache's true
            # provenance, not the pin (mirrors the backbone block above).
            "revision": emb_meta.get("revision"),
            "hidden_dim": EMB_DIM,
            "pooling": emb_meta.get("pooling", "per-position (no pooling)"),
            "frozen": True,
            "extract_seed": int(emb_meta.get("seed", config.get("seed", DEFAULT_SEED))),
        },
    }
    return _sanitize(report)


def run_probe(
    *,
    emb_cache: str | Path = DEFAULT_EMB_CACHE,
    out: str | Path = DEFAULT_OUT,
    windows_parquet: str | Path = DEFAULT_WINDOWS,
    labels_npy: str | Path = DEFAULT_LABELS,
    cfg: ProbeConfig | None = None,
    log=print,
) -> dict[str, Any]:
    """Stage B: load the frozen cache, train the head, decode + grade, write the report."""
    import numpy as np  # lazy

    cfg = cfg or ProbeConfig()
    smoke.set_determinism(cfg.seed)

    emb_cache = Path(emb_cache)
    data = np.load(emb_cache, allow_pickle=False)
    hidden_flat = data["hidden_flat"]
    labels_flat = data["labels_flat"]
    lengths = data["lengths"]

    meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
    cache_dim = int(meta.get("emb_dim", hidden_flat.shape[1]))
    if cache_dim != EMB_DIM or int(hidden_flat.shape[1]) != EMB_DIM:
        raise ValueError(
            f"embedding cache dim mismatch: meta={cache_dim}, array={hidden_flat.shape[1]}, "
            f"expected {EMB_DIM} — re-run --stage extract"
        )
    if meta.get("revision") not in (None, REVISION):
        raise ValueError(
            f"embedding cache revision {meta.get('revision')!r} != pinned {REVISION!r} — "
            "re-run --stage extract against the pinned backbone"
        )

    windows = _unpack_windows(hidden_flat, labels_flat, lengths)
    n_valid = int(sum(int(x) for x in lengths))
    log(f"[{STEP}] loaded {len(windows)} windows, {n_valid} valid positions from {emb_cache}")

    segmenter = build_frozen_probe(cfg)
    log(
        f"[{STEP}] training seg head (use_crf={cfg.use_crf}, rc_combine={cfg.rc_combine}) "
        f"on frozen embeddings for {cfg.epochs} epochs"
    )
    train_frozen_head(segmenter, windows, cfg, log=log)
    y_true, y_pred, validity = evaluate_frozen_head(segmenter, windows)

    metrics_block = smoke.compute_metrics(y_true, y_pred)
    report = build_seg_report(
        metrics_block=metrics_block,
        validity=validity,
        config=cfg.sanitized_knobs(),
        n_windows=len(windows),
        n_valid_positions=n_valid,
        windows_parquet=windows_parquet,
        labels_npy=labels_npy,
        emb_meta=meta,
    )

    report["env"] = _env_block()
    inputs = {}
    for path in (windows_parquet, labels_npy, emb_cache):
        try:
            inputs[str(path)] = provenance.sha256_file(path)
        except Exception:
            inputs[str(path)] = "unavailable"
    report["provenance"] = {
        "git_sha": provenance.git_sha(),
        "probe_seed": int(cfg.seed),
        "extract_seed": int(meta.get("seed", cfg.seed)),
        "env_lock": _safe_env_lock(ENV_LOCK),
        "inputs": inputs,
        "extract_inputs": meta.get("extract_inputs", {}),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    problems = validate_report(report)
    if problems:
        sys.stderr.write(json.dumps({"schema_errors": problems}, indent=2) + "\n")
        raise RuntimeError("frozen_seg_probe report failed self-validation: " + "; ".join(problems))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    log(
        f"[{STEP}] wrote report → {out}  min_core_f1={metrics_block['min_core_f1']}  "
        f"macro_f1={metrics_block['macro_f1']}  valid_seg={validity['all_windows_valid_8class']}"
    )
    return report


def _safe_env_lock(lock_path: str) -> str:
    try:
        return provenance.env_lock_hash(lock_path)
    except Exception:
        return "unavailable"


def run_seg_probe(cfg: ProbeConfig | None = None, *, log=print) -> dict[str, Any]:
    """End-to-end fallback probe (Stage A extract → Stage B probe) — the pinned entry.

    Extracts frozen per-position embeddings (unless ``cfg.reuse_embeddings`` and the cache
    exists), trains the P1-03 head on them, decodes + grades per-nt F1, and writes the
    non-gated ``frozen_seg_probe.json`` fallback-baseline report. Single-env torch
    (``tbox-ml-dna``): extraction needs CUDA (ADR-0002 A2 C2); the head probe is CPU-fine.
    """
    cfg = cfg or ProbeConfig()
    if not (cfg.reuse_embeddings and Path(cfg.emb_cache).exists()):
        run_extract(
            windows_parquet=cfg.windows_parquet,
            labels_npy=cfg.labels_npy,
            emb_cache=cfg.emb_cache,
            device=cfg.device,
            seed=cfg.seed,
            log=log,
        )
    return run_probe(
        emb_cache=cfg.emb_cache,
        out=cfg.report_path,
        windows_parquet=cfg.windows_parquet,
        labels_npy=cfg.labels_npy,
        cfg=cfg,
        log=log,
    )


# --------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P1-09 frozen-embedding seg-head fallback probe")
    p.add_argument(
        "--stage",
        choices=("extract", "probe", "all"),
        default="all",
        help="extract (GPU forward → cache) | probe (train head → report) | all",
    )
    p.add_argument("--windows", default=DEFAULT_WINDOWS)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--emb-cache", default=DEFAULT_EMB_CACHE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--use-crf", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default=None)
    p.add_argument("--reuse-embeddings", action="store_true")
    return p


def _cfg_from_args(args) -> ProbeConfig:
    return ProbeConfig(
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        use_crf=args.use_crf,
        device=args.device,
        windows_parquet=args.windows,
        labels_npy=args.labels,
        emb_cache=args.emb_cache,
        report_path=args.out,
        reuse_embeddings=args.reuse_embeddings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _cfg_from_args(args)
    if args.stage == "extract":
        run_extract(
            windows_parquet=cfg.windows_parquet,
            labels_npy=cfg.labels_npy,
            emb_cache=cfg.emb_cache,
            device=cfg.device,
            seed=cfg.seed,
        )
        return 0
    if args.stage == "probe":
        report = run_probe(
            emb_cache=cfg.emb_cache,
            out=cfg.report_path,
            windows_parquet=cfg.windows_parquet,
            labels_npy=cfg.labels_npy,
            cfg=cfg,
        )
    else:
        report = run_seg_probe(cfg)
    # Non-gated: a valid measured report always exits 0 (the per-nt F1 is a reported baseline).
    print(
        json.dumps(
            {
                "metrics": {
                    "min_core_f1": report["metrics"]["min_core_f1"],
                    "macro_f1": report["metrics"]["macro_f1"],
                },
                "segmentation_validity": report["segmentation_validity"],
                "out": str(cfg.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
