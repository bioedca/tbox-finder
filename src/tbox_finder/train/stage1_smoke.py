"""P1-07 — Transfer go/no-go part (ii, **BINDING**): per-nucleotide 8-class
segmentation smoke fine-tune + F1-over-the-3-core-elements.

The public Caduceus-PS checkpoints are **human-reference pretrained**, while the
discovery scan is fully prokaryotic/archaeal/MAG (PRD §10.1; ADR-0002 D7). This module
answers the **binding** half of the two-part Stage-1 transfer go/no-go: a small-scale
per-nucleotide fine-tune of the P1-04 :class:`~tbox_finder.models.stage1_segmenter.Stage1Segmenter`
(Caduceus-PS backbone → RC-combine → 8-class seg head) on the **P1-06 held-out
prokaryotic smoke set** (``data/interim/p1_seg_smoke/{windows.parquet,labels.npy}`` —
300 leave-clade-out loci, §9.2), grading per-nucleotide **F1 over the three core
elements** {Stem I, Specifier, Antiterminator} (the ADR-0004 D6 min-not-mean unit) and
adjudicating **GO / NO-GO**.

**The go/no-go rule (ADR-0002 D7).** *Go* iff the segmentation smoke recovers the three
core elements at per-nt F1 **clearly above a background-only baseline** on held-out
prokaryotic loci — a directional "the backbone can learn prokaryotic T-box boundaries"
signal, **not** the GATE-4 bar (the 0.80 floor is graded at P2 on the full split, not
here). A **background-only** predictor (predict class 0 everywhere) scores per-element F1
= 0 on every core element, so its min-core-F1 baseline is **0.0**; "clearly above" is
operationalized here as a directional smoke threshold (:data:`DEFAULT_GO_THRESHOLD`) with
a **borderline band** below it that routes to a CLAUDE.md §7 stop-and-ask — because
ADR-0002 D7 makes a borderline read a stop-and-ask and pins the exact small-N smoke
threshold to be *set with the P1 fixture* (it is a go/no-go, **not** a blinded-frozen
gate). These directional thresholds are therefore **provisional pending the §7 verdict
sign-off** (recorded in the ADR-0002 amendment), unlike the 0.80 GATE-4 floor which is
frozen in :mod:`tbox_finder.power`/:mod:`tbox_finder.metrics` and never weakened here.

**This part is BINDING** (the P1-05 frozen-embedding pre-filter is advisory only): a
binary probe can pass while the embeddings lack the positional structure segmentation
needs, or fail while full fine-tuning succeeds. A **NO-GO** triggers the pinned ADR-0002
fallback ladder — full fine-tuning is already the default here → GTDB continued-pretraining
(P1-09) → NT-multispecies backbone (P1-10).

**Objective (PRD §11).** Per-nucleotide **focal** cross-entropy (γ≈2, Lin et al. 2017
[arXiv:1708.02002]) for the dense-background regime — plain CE lets the tiny smoke collapse
to all-background and manufacture a spurious NO-GO. **Deterministic** (explicit seeds,
TF32/cudnn off; §8.3) so the seeded re-run in **P1-08** reproduces the go/no-go within
tolerance.

Compute: SLURM ``gpu`` A4000 (the packaged Mamba forward hard-requires ``selective_scan_cuda``,
ADR-0002 A2 C2 → GPU only; there is no CPU forward). Rule/Entry: :func:`main` (Hydra) →
:func:`run_smoke`.

**Two-tier importability.** All heavy imports (``torch``, ``transformers``, ``hydra``,
``wandb``, ``numpy``, ``pandas``) are lazy inside functions, so the go/no-go adjudication
+ report machinery import and run in a bare env (the CI Tier-1 path).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import metrics as M
from tbox_finder import provenance
from tbox_finder.labels import CLASS_INDEX, CORE_ELEMENTS
from tbox_finder.models.caduceus_backbone import (
    PRETRAINING_DOMAIN,
    REPO_ID,
    REVISION,
)
from tbox_finder.power import GATE4_F1_FLOOR

# --------------------------------------------------------------------------------------
# Constants (gate-critical values single-sourced from code, never weakened by config).
# --------------------------------------------------------------------------------------
SCHEMA_VERSION = "1"
STEP = "P1-07"
GENERATED_BY = "src/tbox_finder/train/stage1_smoke.py"
ADR = "ADR-0002"
ENV_LOCK = "envs/ml-dna.conda-lock.yml"

#: labels.npy pad / seg-head cross-entropy ignore value (ADR-0005). Kept as a local
#: literal so this module imports bare (importing it from ``data.seg_smoke`` would pull
#: numpy); a drift guard test asserts it equals ``seg_smoke.IGNORE_INDEX``.
IGNORE_INDEX = -100

_BASE_TOKENS = "ACGTN"  # Caduceus char vocab bases we map explicitly

#: The three GATE-4 core elements graded by the smoke, as class indices — single-sourced
#: from :data:`tbox_finder.labels.CORE_ELEMENTS` (== metrics.CORE_ELEMENT_INDICES).
CORE_ELEMENT_NAMES: tuple[str, ...] = CORE_ELEMENTS
CORE_ELEMENT_INDICES: tuple[int, ...] = tuple(CLASS_INDEX[e] for e in CORE_ELEMENTS)

VERDICTS = ("GO", "borderline", "NO-GO")

#: The concrete background-only baseline: a predictor that emits class 0 (background)
#: everywhere never predicts a core element, so its per-element F1 is 0 on each of the
#: three core classes and its **min-core-F1 baseline is exactly 0.0** (ADR-0002 D7's
#: "background-only baseline"). GO requires the smoke's min-core-F1 to be *clearly above*
#: this.
BACKGROUND_ONLY_MIN_F1 = 0.0

#: DIRECTIONAL go/no-go smoke thresholds (ADR-0002 D7). NOT the 0.80 GATE-4 bar — that is
#: graded at P2 on the full split. ``GO`` iff min-core-F1 ≥ :data:`DEFAULT_GO_THRESHOLD`
#: (a clear signal that all three core elements are recovered well above the 0.0
#: background-only baseline, yet below the 0.80 GATE-4 floor which P2 owns); ``NO-GO`` iff
#: min-core-F1 ≤ :data:`DEFAULT_NOGO_THRESHOLD` (at/near background); the band between is
#: ``borderline`` → CLAUDE.md §7 stop-and-ask. These are **provisional** directional
#: values set *with the P1 fixture* per D7 (a go/no-go, not a blinded-frozen gate),
#: finalized at the §7 verdict sign-off + the ADR-0002 amendment; overridable on the CLI.
DEFAULT_GO_THRESHOLD = 0.30
DEFAULT_NOGO_THRESHOLD = 0.10

# Defaults (repo-relative; overridable via Hydra / CLI).
DEFAULT_WINDOWS = "data/interim/p1_seg_smoke/windows.parquet"
DEFAULT_LABELS = "data/interim/p1_seg_smoke/labels.npy"
DEFAULT_CHECKPOINT_DIR = "checkpoints/p1/seg_smoke"
DEFAULT_REPORT = "reports/p1/seg_smoke_gonogo.json"
DEFAULT_PREFILTER_REPORT = "reports/p1/prefilter_separability.json"

# windows.parquet columns we consume (self-contained: raw DNA + per-nt labels).
WINDOW_SEQ_COL = "FASTA_sequence"
RECORD_ID_COL = "record_id"
LENGTH_COL = "tbox_length"


# --------------------------------------------------------------------------------------
# Run configuration (a plain dataclass so ``run_smoke`` is callable without Hydra —
# ``main`` builds one from the composed OmegaConf; tests build one directly).
# --------------------------------------------------------------------------------------
@dataclass
class SmokeConfig:
    """Fully-resolved run knobs for :func:`run_smoke`. Seeds + determinism are §8.3."""

    seed: int = 42
    epochs: int = 20
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    gamma: float = 2.0  # focal-loss γ (PRD §11)
    rc_combine: str = "concat"  # directionality-preserving (P1-04); "mean" is code-rejected
    use_crf: bool = False
    dropout: float = 0.0  # 0.0 ⇒ deterministic inference (P1-08 repro)
    batch_size: int = 1  # variable-length windows; batch=1 avoids pad/mask (deterministic)
    device: str | None = None  # None ⇒ cuda if available else cpu (forward needs cuda)
    windows_parquet: str = DEFAULT_WINDOWS
    labels_npy: str = DEFAULT_LABELS
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    report_path: str = DEFAULT_REPORT
    prefilter_report: str = DEFAULT_PREFILTER_REPORT
    go_threshold: float = DEFAULT_GO_THRESHOLD
    nogo_threshold: float = DEFAULT_NOGO_THRESHOLD
    save_checkpoint: bool = True
    wandb_mode: str = "offline"  # "offline" | "online" | "disabled"
    wandb_project: str = "tbox-finder"
    wandb_entity: str | None = None
    wandb_dir: str = "wandb"

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
            "batch_size": self.batch_size,
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


def _bad_bool(value: Any, expected: bool) -> bool:
    """True iff ``value`` is not the exact boolean ``expected`` (int 0/1 rejected)."""
    return not (isinstance(value, bool) and value is expected)


def strip_pad(
    true_row: Sequence[int], pred_row: Sequence[int], *, ignore_index: int = IGNORE_INDEX
) -> tuple[list[int], list[int]]:
    """Return ``(true, pred)`` keeping only positions where ``true != ignore_index``.

    The gate metric (:func:`tbox_finder.metrics.gate4_core_min_f1`) does **not** mask the
    ``-100`` pad — a pad position predicted as a core class would count as a false
    positive and wrongly depress F1 — so pad **must** be stripped before flattening.
    Raises :class:`ValueError` on a length mismatch.
    """
    if len(true_row) != len(pred_row):
        raise ValueError(f"true/pred length mismatch: {len(true_row)} != {len(pred_row)}")
    keep_true: list[int] = []
    keep_pred: list[int] = []
    for t, p in zip(true_row, pred_row, strict=True):
        if int(t) != ignore_index:
            keep_true.append(int(t))
            keep_pred.append(int(p))
    return keep_true, keep_pred


def flatten_valid(
    true_rows: Sequence[Sequence[int]],
    pred_rows: Sequence[Sequence[int]],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[int], list[int]]:
    """Flatten per-window ``(true, pred)`` rows into two 1-D arrays, stripping pad.

    ``true_rows[i]`` and ``pred_rows[i]`` are the per-nucleotide class indices of window
    *i* (equal length per row). Returns the concatenated ``(y_true, y_pred)`` over all
    windows' non-pad positions — the exact input contract of
    :func:`tbox_finder.metrics.gate4_core_min_f1`.
    """
    if len(true_rows) != len(pred_rows):
        raise ValueError(f"row-count mismatch: {len(true_rows)} != {len(pred_rows)}")
    y_true: list[int] = []
    y_pred: list[int] = []
    for t_row, p_row in zip(true_rows, pred_rows, strict=True):
        kt, kp = strip_pad(t_row, p_row, ignore_index=ignore_index)
        y_true.extend(kt)
        y_pred.extend(kp)
    return y_true, y_pred


def core_element_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Per-nt F1 for each of the three core elements, keyed by element name.

    A thin wrapper over :func:`tbox_finder.metrics.per_nt_class_f1` so the transfer harness
    and GATE-4 grade the same commensurable per-nt-class unit (no re-derivation).
    """
    return {
        name: M.per_nt_class_f1(y_true, y_pred, CLASS_INDEX[name]) for name in CORE_ELEMENT_NAMES
    }


def background_only_min_f1(y_true: Sequence[int]) -> float:
    """The min-core-F1 a **background-only** predictor (class 0 everywhere) achieves.

    This is ADR-0002 D7's concrete "background-only baseline": for any window with core
    elements present in truth it is exactly ``0.0`` (each core F1 = 0 when never
    predicted). Computed via the same gate metric so the baseline and the model share one
    definition.
    """
    y_pred = [0] * len(y_true)
    return M.gate4_core_min_f1(y_true, y_pred)["min_f1"]


def classify_gonogo(
    min_core_f1: Any,
    *,
    go_threshold: float = DEFAULT_GO_THRESHOLD,
    nogo_threshold: float = DEFAULT_NOGO_THRESHOLD,
    baseline: float = BACKGROUND_ONLY_MIN_F1,
) -> str:
    """Operationalize the ADR-0002 D7 go/no-go rule into a verdict (fail-closed).

    - ``GO`` — ``min_core_f1 >= go_threshold`` **and** clearly above the background-only
      baseline: all three core elements are recovered at per-nt F1 well above the 0.0
      baseline (a directional "the backbone learns prokaryotic T-box boundaries" signal).
    - ``NO-GO`` — ``min_core_f1 <= nogo_threshold`` (at/near the background-only baseline),
      or **unmeasurable** (a core element absent from both truth and prediction → NaN →
      fail-closed, §10.3): the smoke did not recover the core elements.
    - ``borderline`` — between the two thresholds (weak/ambiguous transfer): per ADR-0002
      D7 a borderline read is a **CLAUDE.md §7 stop-and-ask**, never an auto-decision.

    The thresholds are DIRECTIONAL smoke values (D7), **not** the 0.80 GATE-4 bar; they are
    provisional pending the §7 verdict sign-off.
    """
    if not _finite_number(min_core_f1):
        return "NO-GO"  # a core element unmeasurable ⇒ fail-closed (§10.3)
    v = float(min_core_f1)
    if v <= float(nogo_threshold) or v <= float(baseline):
        return "NO-GO"
    if v >= float(go_threshold):
        return "GO"
    return "borderline"


def build_report(
    *,
    metrics_block: Mapping[str, Any],
    baseline_min_f1: float,
    config: Mapping[str, Any],
    n_windows: int,
    n_valid_positions: int,
    go_threshold: float,
    nogo_threshold: float,
    windows_parquet: str | Path,
    labels_npy: str | Path,
    prefilter: Mapping[str, Any] | None,
    wandb_run_id: str | None,
    measured: bool = True,
) -> dict[str, Any]:
    """Assemble the ``seg_smoke_gonogo.json`` report (verdict follows from the metrics)."""
    min_core_f1 = metrics_block.get("min_core_f1")
    verdict = classify_gonogo(min_core_f1, go_threshold=go_threshold, nogo_threshold=nogo_threshold)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "measured": bool(measured),
        "binding": True,  # this is the BINDING half of the ADR-0002 D7 go/no-go
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "backbone": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "pretraining_domain": PRETRAINING_DOMAIN,
        },
        "config": dict(config),
        "data": {
            "windows_parquet": str(windows_parquet),
            "labels_npy": str(labels_npy),
            "n_windows": int(n_windows),
            "n_valid_positions": int(n_valid_positions),
            "held_out_only": True,  # P1-06 smoke set is leave-clade-out held-out fold (§9.2)
            # HONESTY (§10.3): the smoke fine-tunes AND grades on the same 300 held-out
            # loci — a *learnability / expressivity* transfer probe (ADR-0002 D7: "can the
            # backbone learn prokaryotic T-box boundaries"), NOT a within-smoke
            # generalization test. True generalization is graded at P2 on the full split
            # (D7 defers it). The loci are held-out from the P2 training corpus (§9.2), so
            # the GO does not contaminate P2; but a high F1 here reflects fit, not transfer
            # generalization, and must not be read as such.
            "eval_split": "fine_tune_set",
        },
        "metrics": dict(metrics_block),
        "baseline": {"kind": "background_only", "min_core_f1": float(baseline_min_f1)},
        "gate": {
            "criterion": (
                "per-nt F1 over the 3 core elements {Stem_I, Specifier, "
                "Antiterminator_Tbox_seq} clearly above the background-only baseline "
                "(ADR-0002 D7) — NOT the 0.80 GATE-4 floor (graded at P2 on the full split)"
            ),
            "gate4_floor_reference": GATE4_F1_FLOOR,  # for context; NOT the smoke bar
            "go_threshold": float(go_threshold),
            "nogo_threshold": float(nogo_threshold),
            "thresholds_provisional": True,  # pending §7 verdict sign-off + ADR-0002 amend
            "min_core_f1": min_core_f1,
            "baseline_min_core_f1": float(baseline_min_f1),
            "verdict": verdict,
            "binding": True,
        },
        "advisory_prefilter": prefilter,
        "wandb_run_id": wandb_run_id,
    }
    return _sanitize(report)


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of schema/consistency problems (empty ⇒ valid). Never raises.

    Fails **closed**: the verdict must follow from the metrics + thresholds, the report
    must declare itself binding, and the held-out-only flag must hold (§9.2). A
    hand-edited gate that contradicts the numbers is rejected (§10.3, §8.7).
    """
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if report.get("step") != STEP:
        errors.append(f"step must be {STEP!r}")
    if _bad_bool(report.get("measured"), True):
        errors.append("measured must be boolean True")
    if _bad_bool(report.get("binding"), True):
        errors.append("binding must be boolean True (the ADR-0002 D7 binding go/no-go)")

    backbone = report.get("backbone")
    if not isinstance(backbone, dict):
        errors.append("backbone block missing")
    else:
        if backbone.get("revision") != REVISION:
            errors.append("backbone.revision must be the pinned Caduceus-PS commit")

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics block missing")
    else:
        per = metrics.get("per_element_f1")
        if not isinstance(per, dict) or set(per) != set(CORE_ELEMENT_NAMES):
            errors.append(
                f"metrics.per_element_f1 must have exactly the core elements {CORE_ELEMENT_NAMES}"
            )
        # F1s may be null (a core element absent from truth+pred ⇒ NaN ⇒ sanitized None).
        for name, val in (per or {}).items() if isinstance(per, dict) else []:
            if val is not None and not _prob01(val):
                errors.append(f"metrics.per_element_f1[{name}] must be null or in [0, 1]")
        mcf = metrics.get("min_core_f1")
        if mcf is not None and not _prob01(mcf):
            errors.append("metrics.min_core_f1 must be null or in [0, 1]")

    baseline = report.get("baseline")
    if not isinstance(baseline, dict):
        errors.append("baseline block missing")
    elif baseline.get("min_core_f1") != BACKGROUND_ONLY_MIN_F1:
        errors.append("baseline.min_core_f1 must be 0.0 (a background-only predictor)")

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
        if gate.get("verdict") not in VERDICTS:
            errors.append(f"gate.verdict must be one of {VERDICTS}")
        if _bad_bool(gate.get("binding"), True):
            errors.append("gate.binding must be boolean True")
        for key in ("go_threshold", "nogo_threshold"):
            if not _prob01(gate.get(key)):
                errors.append(f"gate.{key} must be a finite number in [0, 1]")
        # Internal consistency: the verdict must follow from min_core_f1 + the thresholds.
        if _prob01(gate.get("go_threshold")) and _prob01(gate.get("nogo_threshold")):
            expected = classify_gonogo(
                gate.get("min_core_f1"),
                go_threshold=float(gate["go_threshold"]),
                nogo_threshold=float(gate["nogo_threshold"]),
            )
            if gate.get("verdict") != expected:
                errors.append(f"gate.verdict inconsistent with min_core_f1 (expected {expected!r})")
    return errors


# --------------------------------------------------------------------------------------
# Torch tier — data loading, focal loss, fine-tune, evaluation (lazy imports; GPU).
# --------------------------------------------------------------------------------------
def _epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """A deterministic, ``PYTHONHASHSEED``-independent per-epoch shuffle of ``range(n)``."""
    rng = random.Random((int(seed) * 1_000_003) ^ (int(epoch) + 1))
    order = list(range(n))
    rng.shuffle(order)
    return order


def set_determinism(seed: int) -> None:
    """Seed every RNG + disable nondeterministic fast paths (§8.3; P1-08 repro)."""
    import numpy as np  # lazy
    import torch  # lazy

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # warn_only: the Mamba selective-scan kernel has no deterministic variant registered;
    # we still pin every seed + disable TF32/cudnn autotune, which is what P1-08 re-runs.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_ids_of(tokenizer):
    """Build the char→id map (reused verbatim from the P1-05 extraction path)."""
    id_map: dict[str, int] = {}
    for base in _BASE_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(base)
        if tid is not None and tid != tokenizer.unk_token_id:
            id_map[base] = int(tid)
    unk = tokenizer.unk_token_id
    if unk is None:
        unk = id_map["A"]

    def ids_of(seq: str) -> list[int]:
        return [id_map.get(c, unk) for c in seq.upper()]

    return ids_of


def load_smoke_windows(
    windows_parquet: str | Path, labels_npy: str | Path
) -> list[tuple[str, list[int], int]]:
    """Load the P1-06 smoke set as ``[(dna_seq, label_indices, valid_len), ...]``.

    ``windows.parquet`` is self-contained (raw ``FASTA_sequence`` + per-nt labels); we
    re-establish the load-bearing positional contract — ``labels.npy`` row *i* ↔
    ``windows`` row *i* **after sorting both by ``record_id`` ascending** — and take exactly
    the first ``tbox_length`` label entries per row (the valid, non-pad positions). Each
    returned window carries **no pad** (``valid_len == len(dna_seq) == len(label_indices)``),
    so the downstream flatten never sees ``-100``. Fails loud on any length disagreement.
    """
    import numpy as np  # lazy
    import pandas as pd  # lazy

    df = pd.read_parquet(windows_parquet).sort_values(RECORD_ID_COL).reset_index(drop=True)
    labels = np.load(labels_npy, allow_pickle=False)  # (n, max_len) int16, pad -100
    if labels.shape[0] != len(df):
        raise ValueError(f"labels rows {labels.shape[0]} != windows rows {len(df)}")

    windows: list[tuple[str, list[int], int]] = []
    for i, row in df.iterrows():
        seq = str(row[WINDOW_SEQ_COL])
        length = int(row[LENGTH_COL])
        if length != len(seq):
            raise ValueError(f"row {i}: tbox_length {length} != len(FASTA_sequence) {len(seq)}")
        lab = labels[i]
        valid = lab[lab != IGNORE_INDEX]
        if valid.shape[0] != length:
            raise ValueError(
                f"row {i}: {valid.shape[0]} valid labels != tbox_length {length} "
                "(labels.npy / windows.parquet not co-sorted by record_id)"
            )
        windows.append((seq, [int(v) for v in valid.tolist()], length))
    return windows


def focal_cross_entropy(logits, targets, *, gamma: float, ignore_index: int = IGNORE_INDEX):
    """Focal cross-entropy for per-nucleotide segmentation (Lin et al. 2017; PRD §11).

    ``logits`` is ``(B, L, C)``, ``targets`` is ``(B, L)`` of class indices (``ignore_index``
    marks pad). Reduces to the **mean over non-ignored tokens** (not ``(B*L)``, which would
    dilute by padding). At ignored positions ``F.cross_entropy(reduction="none")`` returns
    0 → ``p_t = exp(-0) = 1`` → the focal weight ``(1 - p_t)^γ`` is 0, so pad contributes
    nothing. γ = 0 recovers plain cross-entropy.
    """
    import torch  # lazy
    import torch.nn.functional as F  # lazy

    logits_t = logits.transpose(1, 2)  # (B, C, L) for F.cross_entropy
    ce = F.cross_entropy(logits_t, targets, ignore_index=ignore_index, reduction="none")  # (B, L)
    pt = torch.exp(-ce)
    focal = (1.0 - pt).clamp_min(0.0).pow(gamma) * ce
    valid = targets != ignore_index
    denom = valid.sum().clamp_min(1)
    return focal.sum() / denom


def build_segmenter(cfg: SmokeConfig):
    """Load the code-pinned Caduceus-PS backbone and wrap it in the P1-04 segmenter."""
    import torch  # lazy

    from tbox_finder.models.caduceus_backbone import load_caduceus_ps, load_tokenizer
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_caduceus_ps(device=device)  # pinned revision; rejects any other
    segmenter = Stage1Segmenter(
        backbone=backbone,
        rc_combine=cfg.rc_combine,
        use_crf=cfg.use_crf,
        dropout=cfg.dropout,
    ).to(device)
    tokenizer = load_tokenizer()
    return segmenter, tokenizer, device


def train_smoke(segmenter, windows, ids_of, cfg: SmokeConfig, *, log=print):
    """Full fine-tune (backbone + head) with focal loss (ADR-0002 D7 fallback 1)."""
    import torch  # lazy

    device = next(segmenter.parameters()).device
    segmenter.train()
    for p in segmenter.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(segmenter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        order = _epoch_order(len(windows), cfg.seed, epoch)
        running = 0.0
        for idx in order:
            seq, labels, _ = windows[idx]
            x = torch.tensor([ids_of(seq)], device=device)  # (1, L)
            y = torch.tensor([labels], device=device)  # (1, L)
            logits = segmenter(input_ids=x)  # (1, L, 8) — GPU forward
            loss = focal_cross_entropy(logits, y, gamma=cfg.gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(segmenter.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.detach())
        log(f"  epoch {epoch + 1}/{cfg.epochs} focal_loss={running / max(len(windows), 1):.4f}")
    return segmenter


def evaluate_smoke(segmenter, windows, ids_of) -> tuple[list[int], list[int]]:
    """Decode per-nt predictions over every held-out window → flat ``(y_true, y_pred)``.

    Each window contributes exactly its ``valid_len`` positions (no pad), so the flatten is
    the direct input to :func:`tbox_finder.metrics.gate4_core_min_f1`.
    """
    import torch  # lazy

    device = next(segmenter.parameters()).device
    segmenter.eval()
    true_rows: list[list[int]] = []
    pred_rows: list[list[int]] = []
    with torch.no_grad():
        for seq, labels, _ in windows:
            x = torch.tensor([ids_of(seq)], device=device)
            logits = segmenter(input_ids=x)  # (1, L, 8)
            pred = logits.argmax(dim=-1).squeeze(0).tolist()  # length L
            true_rows.append(list(labels))
            pred_rows.append([int(v) for v in pred])
    return flatten_valid(true_rows, pred_rows)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Grade a flattened per-nt prediction — the core-element gate + reported extras."""
    gate = M.gate4_core_min_f1(y_true, y_pred)  # {per_element_f1, min_f1, floor, passes}
    return {
        "per_element_f1": gate["per_element_f1"],  # the 3 core elements {name: f1}
        "min_core_f1": gate["min_f1"],
        "macro_f1": M.macro_f1(y_true, y_pred),
        "micro_f1": M.micro_f1(y_true, y_pred),
        "per_class_f1": M.per_nt_f1_by_class(y_true, y_pred),  # all 8 classes {name: f1}
    }


def _load_prefilter(path: str | Path) -> dict[str, Any] | None:
    """The P1-05 advisory pre-filter verdict (non-binding input to the adjudication)."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        rep = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(rep, dict):  # a valid-but-non-object advisory JSON must not crash the run
        return None
    gate = rep.get("gate")
    gate = gate if isinstance(gate, dict) else {}
    metrics = rep.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "path": str(path),
        "verdict": gate.get("verdict"),
        "binding": False,
        "auroc": metrics.get("auroc"),
    }


def _init_wandb(cfg: SmokeConfig):
    """Best-effort offline W&B init; returns ``(run, run_id)`` or ``(None, None)``."""
    if cfg.wandb_mode == "disabled":
        return None, None
    try:
        import wandb  # lazy
    except ImportError:
        return None, None
    try:
        run = wandb.init(
            mode=cfg.wandb_mode,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            dir=cfg.wandb_dir,
            job_type="p1-07-seg-smoke",
            config=cfg.sanitized_knobs(),
            reinit=True,
        )
        return run, run.id
    except Exception:  # noqa: BLE001 — W&B must never break a training run
        return None, None


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_smoke(cfg: SmokeConfig, *, log=print) -> dict[str, Any]:
    """Run the full go/no-go smoke: seed → load → fine-tune → grade → report.

    Writes the checkpoint (+ provenance) and the ``seg_smoke_gonogo.json`` report, and
    returns the report dict. The verdict is **measured**; the borderline/NO-GO branch is a
    CLAUDE.md §7 stop-and-ask handled by the caller (this function never loosens a
    threshold or fabricates a number, §10.3).
    """
    # Fail loud on knobs this smoke does not wire, so a stray override can't silently
    # no-op while being recorded in provenance / W&B (§10.3 truthful provenance).
    if cfg.use_crf:
        raise ValueError(
            "use_crf is not supported by the P1-07 smoke: PRD §11 pins per-nt focal "
            "cross-entropy and this harness trains on emission logits + argmax-decodes "
            "(the CRF NLL/Viterbi path is a P2 ablation, unwired here)."
        )
    if cfg.batch_size != 1:
        raise ValueError(
            f"batch_size must be 1 (variable-length windows are trained one at a time to "
            f"avoid pad/mask and stay deterministic; got {cfg.batch_size})."
        )
    if not (0.0 <= cfg.nogo_threshold <= cfg.go_threshold <= 1.0):
        raise ValueError(
            f"go_threshold ({cfg.go_threshold}) and nogo_threshold ({cfg.nogo_threshold}) "
            "must both be in [0, 1] with go_threshold >= nogo_threshold — a malformed "
            "override must never produce a nonsensical verdict (§10.3)."
        )
    set_determinism(cfg.seed)
    log(f"[{STEP}] loading smoke set {cfg.windows_parquet} + {cfg.labels_npy}")
    windows = load_smoke_windows(cfg.windows_parquet, cfg.labels_npy)
    n_valid = sum(w[2] for w in windows)
    log(f"[{STEP}] {len(windows)} held-out windows, {n_valid} valid positions")

    segmenter, tokenizer, device = build_segmenter(cfg)
    ids_of = _make_ids_of(tokenizer)
    log(f"[{STEP}] backbone {REPO_ID}@{REVISION[:8]} on {device}; fine-tune {cfg.epochs} epochs")

    run, wandb_run_id = _init_wandb(cfg)
    train_smoke(segmenter, windows, ids_of, cfg, log=log)
    y_true, y_pred = evaluate_smoke(segmenter, windows, ids_of)

    metrics_block = compute_metrics(y_true, y_pred)
    baseline = background_only_min_f1(y_true)
    prefilter = _load_prefilter(cfg.prefilter_report)

    report = build_report(
        metrics_block=metrics_block,
        baseline_min_f1=baseline,
        config=cfg.sanitized_knobs(),
        n_windows=len(windows),
        n_valid_positions=n_valid,
        go_threshold=cfg.go_threshold,
        nogo_threshold=cfg.nogo_threshold,
        windows_parquet=cfg.windows_parquet,
        labels_npy=cfg.labels_npy,
        prefilter=prefilter,
        wandb_run_id=wandb_run_id,
    )

    problems = validate_report(report)
    if problems:
        raise RuntimeError("seg_smoke report failed self-validation: " + "; ".join(problems))

    if run is not None:
        try:
            run.log(
                {
                    "min_core_f1": metrics_block["min_core_f1"],
                    **{f"f1/{k}": v for k, v in metrics_block["per_element_f1"].items()},
                }
            )
            run.summary["verdict"] = report["gate"]["verdict"]
            run.finish()
        except Exception:  # noqa: BLE001
            pass

    _write_outputs(cfg, segmenter, report, log=log)
    verdict = report["gate"]["verdict"]
    mcf = metrics_block["min_core_f1"]
    log(f"[{STEP}] min_core_f1={mcf} baseline={baseline} -> verdict={verdict}")
    return report


def _write_outputs(cfg: SmokeConfig, segmenter, report: Mapping[str, Any], *, log=print) -> None:
    """Persist the report (git-tracked) + checkpoint & provenance (DVC-tracked)."""
    import torch  # lazy

    rp = Path(cfg.report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"[{STEP}] wrote report → {rp}")

    if not cfg.save_checkpoint:
        return
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "seg_smoke.pt"
    torch.save(segmenter.state_dict(), ckpt)
    provenance.write_provenance(
        ckpt_dir / "provenance.json",
        rule="src/tbox_finder/train/stage1_smoke.py::run_smoke",
        script=GENERATED_BY,
        seed=cfg.seed,
        inputs=[cfg.windows_parquet, cfg.labels_npy],
        outputs=[ckpt],
        env_lock=ENV_LOCK if Path(ENV_LOCK).is_file() else None,
        adr=ADR,
        extra={
            "step": STEP,
            "backbone_revision": REVISION,
            "verdict": report["gate"]["verdict"],
            "min_core_f1": report["gate"]["min_core_f1"],
        },
    )
    log(f"[{STEP}] wrote checkpoint → {ckpt}")


# --------------------------------------------------------------------------------------
# Entry points — Hydra (`main`) for the sbatch, argparse (`_run`) for local sizing.
# --------------------------------------------------------------------------------------
def _cfg_from_mapping(cfg: Mapping[str, Any]) -> SmokeConfig:
    """Build a :class:`SmokeConfig` from a (possibly nested) composed config mapping."""
    tracking = dict(cfg.get("tracking", {}) or {})
    optim = dict(cfg.get("optim", {}) or {})
    model = dict(cfg.get("model", {}) or {})
    # rc_combine is single-sourced to the model group (caduceus_stage1.yaml: {mode: concat}),
    # the P1-04 knob's canonical home — so `model.rc_combine.mode=gate` is an honored override.
    # Fall back to a top-level scalar / the SmokeConfig default when the group is absent.
    rc = model.get("rc_combine", cfg.get("rc_combine"))
    if isinstance(rc, Mapping):
        rc = rc.get("mode")
    return SmokeConfig(
        seed=int(cfg.get("seed", 42)),
        epochs=int(cfg.get("epochs", 20)),
        lr=float(optim.get("lr", cfg.get("lr", 3.0e-4))),
        weight_decay=float(optim.get("weight_decay", cfg.get("weight_decay", 0.01))),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        gamma=float(cfg.get("gamma", 2.0)),
        rc_combine=str(rc or "concat"),
        use_crf=bool(cfg.get("use_crf", False)),
        dropout=float(cfg.get("dropout", 0.0)),
        batch_size=int(cfg.get("batch_size", 1)),
        device=cfg.get("device"),
        windows_parquet=str(cfg.get("windows_parquet", DEFAULT_WINDOWS)),
        labels_npy=str(cfg.get("labels_npy", DEFAULT_LABELS)),
        checkpoint_dir=str(cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)),
        report_path=str(cfg.get("report_path", DEFAULT_REPORT)),
        prefilter_report=str(cfg.get("prefilter_report", DEFAULT_PREFILTER_REPORT)),
        go_threshold=float(cfg.get("go_threshold", DEFAULT_GO_THRESHOLD)),
        nogo_threshold=float(cfg.get("nogo_threshold", DEFAULT_NOGO_THRESHOLD)),
        save_checkpoint=bool(cfg.get("save_checkpoint", True)),
        wandb_mode=str(tracking.get("mode", cfg.get("wandb_mode", "offline"))),
        wandb_project=str(tracking.get("project", cfg.get("wandb_project", "tbox-finder"))),
        wandb_entity=tracking.get("entity", cfg.get("wandb_entity")),
        wandb_dir=str(tracking.get("dir", cfg.get("wandb_dir", "wandb"))),
    )


def main() -> None:
    """Hydra entry (the sbatch target). Composes ``conf/train/stage1_smoke.yaml``."""
    import hydra  # lazy
    from omegaconf import OmegaConf  # lazy

    @hydra.main(version_base=None, config_path="../../../conf", config_name="train/stage1_smoke")
    def _entry(cfg) -> None:
        resolved = OmegaConf.to_container(cfg, resolve=True)
        run_smoke(_cfg_from_mapping(resolved))

    _entry()


def _run(argv: list[str] | None = None) -> int:
    """Minimal argparse CLI — used for the local sizing smoke (no Hydra composition)."""
    parser = argparse.ArgumentParser(description="P1-07 Stage-1 segmentation smoke go/no-go")
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = SmokeConfig(
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        windows_parquet=args.windows,
        labels_npy=args.labels,
        report_path=args.report,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoint=not args.no_checkpoint,
        wandb_mode=args.wandb_mode,
    )
    run_smoke(cfg)
    # A NO-GO / borderline is a valid MEASURED outcome (the CLAUDE.md §7 stop-and-ask +
    # ADR-0002 fallback ladder own the verdict), not a CLI error; run_smoke already raised
    # if the report failed self-validation. So a completed run exits 0 regardless of verdict.
    return 0


if __name__ == "__main__":
    # Hydra when invoked bare (the sbatch path); argparse when given CLI flags.
    if any(a.startswith("--") for a in sys.argv[1:]):
        sys.exit(_run())
    main()
