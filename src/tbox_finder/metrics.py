"""PRD §2 gate-metric implementations (P0-31) — the eval harness P2/P3/P4 will use.

This module is the **metric kernel library** behind the acceptance gates (PRD §2):

- **GATE-4 (ADR-0004 D6 → P2):** per-nucleotide, one-vs-rest class F1, gated as the
  **minimum over the three core elements** {Stem I, Specifier, Antiterminator} — a
  homogeneous, commensurable unit, so **no cross-unit mean** (D6); default floor
  ``GATE4_F1_FLOOR = 0.80``. Boundary IoU is reported per element (non-gated).
- **GATE-1 (ADR-0005 D2/D4/D7 → P4):** recall vs a ``cmsearch`` baseline **at matched
  precision** (model recall read at the threshold where model precision equals the
  baseline's, both scoring the identical negative pool at the pinned decoy prevalence),
  plus AUPRC (Area Under the Precision-Recall curve, not AUROC [DOI:10.1371/journal.pone.0118432]).
  Pass bar: point ≥ +10 pp **AND** block-resampled CI lower > +5 pp (D4).
- **GATE-2 (ADR-0005 D11 → P3):** in-distribution **binned-ECE** on the positive-class
  posterior — **15 equal-mass bins, debiased** — gated ``≤ ECE_GATE = 0.05``.
- **D5 aggregation:** GATE-1 / per-order ECE CIs are resampled at the homology-cluster /
  held-out-order **block** level (never per-record) and **macro-averaged across held-out
  orders** (not micro, which the ~90 %-Firmicutes corpus would dominate).

Structure (mirrors ``power.py`` / ``coverage.py``): **pure, stdlib-only metric kernels**
— no numpy / scikit-learn / pandas import anywhere — so the eval harness runs in *any*
env, including the bare CI ``test`` env, and every metric is unit-testable against a
hand-computed value (the §8.7 / §10.3 anti-fabrication guard). ``tests/ml/test_eval_gate.py``
additionally **cross-checks** ``average_precision`` / the ECE binning against the pinned
``scikit-learn 1.9.0`` / ``numpy`` reference under the ``data`` env, proving the stdlib
kernels equal the canonical library implementation.

Pins are single-sourced from :mod:`tbox_finder.power` (ADR-0004/0005 defaults) — this
module never re-declares a gate number, only imports and asserts against it, so the
contract cannot drift. This scaffold **verifies the harness itself, not model quality**
(CLAUDE.md §8.4): it never fabricates a metric or a performance number (§10.3).

Three items the ADRs deliberately leave to the implementer (do **not** read them as
ADR-pinned; see the per-function notes): the **ECE debiasing term** (D11 pins
equal-mass / 15 / debiased, not the exact correction — refs [arXiv:1706.04599;
PMID:25927013]), the **AUPRC step-vs-interpolated** convention (D-refs pin "not AUROC",
not the estimator — this module matches scikit-learn's step estimator), and the
**boundary-IoU band** (pinned "reported per element", no band/threshold — this module
reports the extent IoU). Certifying any of these to a specific formula is a P3 / ADR
concern (CLAUDE.md §7), not a P0-31 assertion.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from tbox_finder.labels import CLASS_INDEX, CLASS_ORDER, CORE_ELEMENTS
from tbox_finder.power import (
    ECE_GATE,
    GATE4_F1_FLOOR,
    RECALL_CI_FLOOR_PP,
    RECALL_POINT_BAR_PP,
)

# --------------------------------------------------------------------------- #
# Pins (single-sourced from power.py where they already exist; new here where not)
# --------------------------------------------------------------------------- #
#: Binned-ECE estimator resolution (ADR-0005 D11): 15 **equal-mass** bins, debiased.
ECE_N_BINS = 15

#: The three GATE-4 core elements, as softmax class indices (ADR-0004 D6). The gate is
#: the **min** of their per-nt F1 — never a mean (commensurable per-nt-class unit).
CORE_ELEMENT_INDICES: tuple[int, ...] = tuple(CLASS_INDEX[e] for e in CORE_ELEMENTS)

__all__ = [
    "ECE_N_BINS",
    "CORE_ELEMENT_INDICES",
    "prf",
    "per_nt_class_f1",
    "per_nt_f1_by_class",
    "macro_f1",
    "micro_f1",
    "gate4_core_min_f1",
    "gate4_pass",
    "element_extent_iou",
    "boundary_iou_by_element",
    "average_precision",
    "precision_recall_at_threshold",
    "baseline_operating_point",
    "recall_at_matched_precision",
    "recall_gap_pp",
    "gate1_recall_bar",
    "binned_ece",
    "gate2_ece_pass",
    "macro_average",
    "block_bootstrap_ci",
]


# --------------------------------------------------------------------------- #
# Small stdlib helpers
# --------------------------------------------------------------------------- #
def _safe_div(num: float, den: float) -> float:
    """``num/den``; NaN when the denominator is 0 (metric undefined, not 0)."""
    return num / den if den else float("nan")


def _array_split(items: list, n_parts: int) -> list[list]:
    """Deterministic ``numpy.array_split`` analogue: split ``items`` into ``n_parts``
    contiguous groups whose sizes differ by at most one (the first ``len % n_parts``
    groups get the extra element). Realizes **equal-mass** binning by equal-count
    slicing of a stable-sorted sequence, so ties are handled deterministically."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    n = len(items)
    base, extra = divmod(n, n_parts)
    out: list[list] = []
    start = 0
    for i in range(n_parts):
        size = base + (1 if i < extra else 0)
        out.append(items[start : start + size])
        start += size
    return out


# --------------------------------------------------------------------------- #
# Per-nucleotide class F1 (GATE-4 gated unit) + macro/micro (reported)
# --------------------------------------------------------------------------- #
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from confusion counts. Each is NaN when undefined
    (no predicted positives → precision NaN; no true positives → recall NaN; the
    class absent from both truth and prediction → F1 NaN)."""
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return precision, recall, f1


def _class_confusion(
    y_true: Sequence[int], y_pred: Sequence[int], class_index: int
) -> tuple[int, int, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if p == class_index and t == class_index:
            tp += 1
        elif p == class_index and t != class_index:
            fp += 1
        elif p != class_index and t == class_index:
            fn += 1
    return tp, fp, fn


def per_nt_class_f1(y_true: Sequence[int], y_pred: Sequence[int], class_index: int) -> float:
    """One-vs-rest **per-nucleotide** F1 for a single class (ADR-0004 D6). NaN when the
    class is absent from both truth and prediction (F1 undefined — not silently 0)."""
    tp, fp, fn = _class_confusion(y_true, y_pred, class_index)
    return prf(tp, fp, fn)[2]


def per_nt_f1_by_class(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str] = CLASS_ORDER,
) -> dict[str, float]:
    """Per-nt F1 for every named class → ``{class_name: f1}`` (reported per class)."""
    return {name: per_nt_class_f1(y_true, y_pred, CLASS_INDEX[name]) for name in classes}


def macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str] = CLASS_ORDER,
) -> float:
    """Macro-average per-nt F1 over ``classes`` (reported, non-gated; PRD §2). Classes
    whose F1 is undefined (absent from both) are dropped from the average, mirroring the
    ``macro`` reporting convention — the **GATE-4 gate never uses this** (it is a min,
    not a mean; see :func:`gate4_core_min_f1`)."""
    vals = [per_nt_class_f1(y_true, y_pred, CLASS_INDEX[c]) for c in classes]
    defined = [v for v in vals if not math.isnan(v)]
    return _safe_div(sum(defined), len(defined))


def micro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str] = CLASS_ORDER,
) -> float:
    """Micro-average per-nt F1 over ``classes`` (reported, non-gated; PRD §2): pool the
    TP/FP/FN across classes, then one F1 from the pooled counts."""
    tp = fp = fn = 0
    for c in classes:
        a, b, d = _class_confusion(y_true, y_pred, CLASS_INDEX[c])
        tp += a
        fp += b
        fn += d
    return prf(tp, fp, fn)[2]


def gate4_core_min_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """GATE-4 gated statistic (ADR-0004 D6): the **minimum per-nt F1 over the three core
    elements** {Stem I, Specifier, Antiterminator} — explicitly a **min, not a mean**
    (per-nt-class F1 is a commensurable unit, so there is no cross-unit average). Returns
    the per-element F1s, the ``min_f1``, and ``passes`` (``min_f1 ≥ GATE4_F1_FLOOR``). If
    any core F1 is undefined (NaN), ``min_f1`` is NaN and ``passes`` is False — an
    unmeasurable core element cannot silently certify the gate."""
    per_element = {
        name: per_nt_class_f1(y_true, y_pred, CLASS_INDEX[name]) for name in CORE_ELEMENTS
    }
    vals = list(per_element.values())
    min_f1 = float("nan") if any(math.isnan(v) for v in vals) else min(vals)
    return {
        "per_element_f1": per_element,
        "min_f1": min_f1,
        "floor": GATE4_F1_FLOOR,
        "passes": (not math.isnan(min_f1)) and min_f1 >= GATE4_F1_FLOOR,
    }


def gate4_pass(min_core_f1: float) -> bool:
    """GATE-4 predicate: ``min_core_f1 ≥ GATE4_F1_FLOOR`` (ADR-0004 D6). NaN → False."""
    return (not math.isnan(min_core_f1)) and min_core_f1 >= GATE4_F1_FLOOR


# --------------------------------------------------------------------------- #
# Boundary / extent IoU (reported per element, non-gated)
# --------------------------------------------------------------------------- #
def element_extent_iou(y_true: Sequence[int], y_pred: Sequence[int], class_index: int) -> float:
    """Per-element extent IoU at nucleotide resolution: ``|{i: true==c} ∩ {i: pred==c}| /
    |{i: true==c} ∪ {i: pred==c}|``. NaN when the element is absent from both (union
    empty). ADR-0004 D6 pins *"boundary IoU reported per element"* but **no band or
    threshold** — this is the full-extent IoU; a boundary-band variant
    [arXiv:2103.16562] is a non-pinned P2 reporting refinement."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    inter = union = 0
    for t, p in zip(y_true, y_pred, strict=True):
        in_t = t == class_index
        in_p = p == class_index
        if in_t or in_p:
            union += 1
            if in_t and in_p:
                inter += 1
    return _safe_div(inter, union)


def boundary_iou_by_element(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str] = CLASS_ORDER,
) -> dict[str, float]:
    """Extent IoU per named class → ``{class_name: iou}`` (reported; ADR-0004 D6)."""
    return {name: element_extent_iou(y_true, y_pred, CLASS_INDEX[name]) for name in classes}


# --------------------------------------------------------------------------- #
# Detection metrics: AUPRC + recall @ matched precision (GATE-1)
# --------------------------------------------------------------------------- #
def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Average precision (AUPRC), the **step** estimator ``AP = Σ_n (R_n − R_{n−1}) P_n``
    — identical to ``sklearn.metrics.average_precision_score`` (verified in
    ``tests/ml/test_eval_gate.py`` against the pinned scikit-learn 1.9.0). Not
    trapezoid-interpolated and **not** AUROC (PRD §2; ADR-0005 D17). ``y_true`` is
    0/1 (1 = positive). NaN when there are no positives.

    Tied scores are grouped at one threshold (as scikit-learn does), so the point is
    emitted once per distinct score with cumulative TP/FP through the tie group.
    """
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must be the same length")
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0:
        return float("nan")
    # sort by score desc; ties broken deterministically by truth then original index so
    # the tie *group* boundary is well-defined (grouping is on the score value only).
    order = sorted(range(len(y_score)), key=lambda i: (-y_score[i], -y_true[i], i))
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    m = len(order)
    while i < m:
        s = y_score[order[i]]
        j = i
        while j < m and y_score[order[j]] == s:
            if y_true[order[j]] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def precision_recall_at_threshold(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float
) -> tuple[float, float]:
    """Precision and recall of the hard call ``score ≥ threshold`` (1 = positive)."""
    tp = fp = fn = 0
    for y, s in zip(y_true, y_score, strict=True):
        called = s >= threshold
        if called and y == 1:
            tp += 1
        elif called and y != 1:
            fp += 1
        elif (not called) and y == 1:
            fn += 1
    return _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)


def baseline_operating_point(
    y_true: Sequence[int], baseline_hit: Sequence[int]
) -> tuple[float, float]:
    """Precision and recall of a fixed binary baseline (e.g. ``cmsearch`` at its RF00230
    GA / published class-II threshold, ADR-0005 D2). ``baseline_hit`` is 0/1."""
    tp = fp = fn = 0
    for y, h in zip(y_true, baseline_hit, strict=True):
        if h and y == 1:
            tp += 1
        elif h and y != 1:
            fp += 1
        elif (not h) and y == 1:
            fn += 1
    return _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)


def recall_at_matched_precision(
    y_true: Sequence[int], y_score: Sequence[float], target_precision: float
) -> dict:
    """Model recall at **matched precision** (ADR-0005 D7): the maximum recall over all
    score thresholds whose precision ≥ ``target_precision`` (the baseline's precision on
    the identical negative pool). Returns ``{threshold, precision, recall, matched}``;
    ``matched`` is False (recall 0.0) if the model never reaches the target precision."""
    best = {
        "threshold": None,
        "precision": float("nan"),
        "recall": 0.0,
        "matched": False,
    }
    for t in sorted(set(y_score), reverse=True):
        p, r = precision_recall_at_threshold(y_true, y_score, t)
        if not math.isnan(p) and p >= target_precision and r >= best["recall"]:
            best = {"threshold": t, "precision": p, "recall": r, "matched": True}
    return best


def recall_gap_pp(model_recall: float, baseline_recall: float) -> float:
    """Recall difference in **percentage points** (model − baseline), the GATE-1 headline
    quantity (ADR-0005 D4)."""
    return (model_recall - baseline_recall) * 100.0


def gate1_recall_bar(
    point_gap_pp: float, ci_lower_pp: float, *, min_n_fallback: bool = False
) -> bool:
    """GATE-1 two-part effect-size bar (ADR-0005 D4). Strong (default): point estimate
    ``≥ +RECALL_POINT_BAR_PP`` (=+10 pp) **AND** block-resampled CI lower bound
    ``> +RECALL_CI_FLOOR_PP`` (=+5 pp). The weaker ``point ≥ +10 AND CI lower > 0`` clause
    is admissible **only** as an explicitly disclosed min-N-conditioned fallback
    (``min_n_fallback=True``)."""
    if point_gap_pp < RECALL_POINT_BAR_PP:
        return False
    return ci_lower_pp > (0.0 if min_n_fallback else RECALL_CI_FLOOR_PP)


# --------------------------------------------------------------------------- #
# Calibration: binned ECE (GATE-2)
# --------------------------------------------------------------------------- #
def binned_ece(
    y_true: Sequence[int],
    p_pos: Sequence[float],
    n_bins: int = ECE_N_BINS,
    *,
    debias: bool = True,
) -> float:
    """Binned Expected Calibration Error on the **positive-class posterior** (ADR-0005
    D11): **equal-mass** bins (default 15), each contributing ``(n_b/N)·|acc_b − conf_b|``.

    Equal-mass binning is realized by slicing the probability-sorted samples into
    equal-count groups (:func:`_array_split`) — stable and tie-deterministic — chosen over
    equal-width because equal-width bins are unstable in the sparse high-confidence region
    (D11).

    **Debiasing (``debias=True``, the D11 default) — implementer choice, NOT an
    ADR-pinned formula.** D11 pins *equal-mass / 15 / debiased* but leaves the exact
    correction open (refs Guo 2017 [arXiv:1706.04599], Naeini 2015 [PMID:25927013]).
    Plug-in ECE is biased upward in finite samples: even a perfectly-calibrated bin shows
    a nonzero ``|acc_b − conf_b|`` from sampling noise. This module subtracts, per bin,
    that expected noise floor — ``σ_b·√(2/π)`` with ``σ_b = √(conf_b(1−conf_b)/n_b)`` (the
    mean of a half-normal, the leading-order ``E|acc_b − conf_b|`` under the calibrated
    null) — flooring each debiased gap at 0. Final certification of the debiasing term is
    a P3-exit / ADR concern (CLAUDE.md §7); ``debias=False`` gives the raw plug-in ECE.
    """
    if len(y_true) != len(p_pos):
        raise ValueError("y_true and p_pos must be the same length")
    n = len(p_pos)
    if n == 0:
        return float("nan")
    order = sorted(range(n), key=lambda k: p_pos[k])
    ece = 0.0
    for b in _array_split(order, n_bins):
        if not b:
            continue
        m = len(b)
        conf = sum(p_pos[k] for k in b) / m
        acc = sum(1 for k in b if y_true[k] == 1) / m
        gap = abs(acc - conf)
        if debias:
            sigma = math.sqrt(max(conf * (1.0 - conf), 0.0) / m)
            gap = max(0.0, gap - sigma * math.sqrt(2.0 / math.pi))
        ece += (m / n) * gap
    return ece


def gate2_ece_pass(ece: float) -> bool:
    """GATE-2 in-distribution calibration predicate: ``ece ≤ ECE_GATE`` (=0.05, ADR-0005
    D11). NaN → False."""
    return (not math.isnan(ece)) and ece <= ECE_GATE


# --------------------------------------------------------------------------- #
# Aggregation: macro-average over held-out orders + block-level bootstrap (D5)
# --------------------------------------------------------------------------- #
def macro_average(per_order_values: Sequence[float]) -> float:
    """Macro-average over held-out orders — equal weight per order, **not** record-pooled
    (ADR-0005 D5), because the ~90 %-Firmicutes corpus would otherwise dominate a micro
    average. Undefined (NaN) per-order values are dropped; NaN if none remain."""
    defined = [v for v in per_order_values if not math.isnan(v)]
    return _safe_div(sum(defined), len(defined))


def block_bootstrap_ci(
    blocks: Sequence,
    statistic: Callable[[list], float],
    *,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile CI for ``statistic`` resampled at the **block** (homology-cluster /
    held-out-order) level, never per-record (ADR-0005 D5) — the phylogenetic exchangeability
    unit. Each ``blocks[i]`` is one block's data; a bootstrap replicate draws ``len(blocks)``
    blocks with replacement (seeded, so reproducible — CLAUDE.md §8.3), concatenates them,
    and applies ``statistic``. Returns ``{point, lower, upper, ci_level, n_boot, n_blocks}``
    over the non-NaN replicates; fewer than 2 blocks → not block-resamplable (NaN CI,
    ADR-0005 Amendment A1)."""
    blocks = list(blocks)
    n_blocks = len(blocks)
    point = statistic([x for blk in blocks for x in blk]) if n_blocks else float("nan")
    if n_blocks < 2:
        return {
            "point": point,
            "lower": float("nan"),
            "upper": float("nan"),
            "ci_level": ci_level,
            "n_boot": 0,
            "n_blocks": n_blocks,
        }
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        drawn = [blocks[rng.randrange(n_blocks)] for _ in range(n_blocks)]
        val = statistic([x for blk in drawn for x in blk])
        if not math.isnan(val):
            reps.append(val)
    reps.sort()
    alpha = (1.0 - ci_level) / 2.0
    lower = _percentile(reps, alpha)
    upper = _percentile(reps, 1.0 - alpha)
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "ci_level": ci_level,
        "n_boot": len(reps),
        "n_blocks": n_blocks,
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy default ``'linear'``) over a pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
