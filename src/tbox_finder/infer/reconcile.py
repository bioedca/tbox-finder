"""P2-03 — the frozen overlapping-window logit-reconciliation operator.

At the pinned scan tiling (window 1024 nt / stride 512 nt, PRD §6) every interior
nucleotide is scored by >= 2 windows, each with different flanking context. ADR-0005 D3
pins how those competing predictions collapse into one per-position call:

    the per-position 8-class logits from all covering windows are **averaged in
    log-sum-exp space then arg-maxed** into one per-position prediction, *before*
    along-sequence element merging.

This module is that operator and nothing else: it stops at the per-position arg-max.
Along-sequence locus construction (threshold scope, minimum span, gap-merge distance,
required-element co-occurrence, flank size — the rest of the D3 rule) is a later step.

Why the operator exists
-----------------------
It is the **seam-free** reduction. Without it, a per-window arg-max would leave the
per-locus recall floor (>= 99 %, ADR-0005 D3) and the GATE-4 boundary IoU dependent on
where the 512-nt grid happened to fall — an element straddling a window edge would be
scored from truncated context on one side. Reconciling first makes the grid invisible.

The pinned form (ADR-0005 D3 + Amendment A3)
--------------------------------------------
For position ``p`` covered by windows ``W(p)``::

    log_probs[p] = log( mean_{w in W(p)} softmax(logits[w, p])  )
                 = logsumexp_{w in W(p)} log_softmax(logits[w, p]) - log |W(p)|
    prediction[p] = argmax_c log_probs[p, c]

D3's sentence — "the *logits* ... averaged in log-sum-exp space then arg-maxed" — admits
**two coherent operators**, and they are not notational variants: averaging the raw logits
in exp-space (soft-max pooling over windows) and averaging the per-window *posteriors*
disagree on **38.3 %** of multi-covered positions on this step's golden geometries.
**Amendment A3 (user sign-off 2026-07-16) pins the posterior form implemented here**; the
reasoning lives in the ADR and is not re-argued here. The short version: ``exp(logit)`` is
an unnormalised score, so pooling it weights window ``w`` by its partition function
``Z_w`` — a quantity the training objective never constrains, because the softmax in the
cross-entropy is invariant to a constant shift of a position's logits. Two windows
predicting the *same* distribution must contribute equally; under A3's form they do.

Two further properties, each a unit test:

1. **Coverage normalisation** (``- log |W(p)|``) is common to both readings and was never
   at issue: it is what "averaged" means as against "summed". A bare log-sum-exp grows
   with the number of covering windows, so a position covered twice would outscore an
   identical position covered once — re-introducing exactly the 512-grid seam the operator
   exists to remove. With it, coverage-invariance is *bit-exact*.
2. **arg-max last**, on the reconciled distribution — as pinned. The result is a proper
   distribution (``exp(log_probs).sum(axis=1) == 1``), which the Stage-1 threshold (D3)
   and the §11 recalibration/ECE stack consume directly.

:func:`diagnostics` records the operator identity with ``pinned=True`` and cites D3 + A3.

Contig ends
-----------
PRD §6 / ADR-0005 D3: *contig ends are zero-flanked and flagged.* A window that runs off
either end of the sequence is zero-padded by the caller; the model still emits logits at
those pad positions, but they describe no DNA. This module therefore

* **never averages a pad position's logits** — only the in-bounds slice of each window
  contributes (so pad logits may be arbitrary, even NaN, without touching the result); and
* **flags** every real position covered by such a window in ``Reconciled.zero_flanked``,
  because the model's context for it included synthetic zeros.

At the pinned tiling this fires only for sequences shorter than one window:
:func:`tbox_finder.data.window_dataset.tile_windows` is tail-anchored, so a sequence
longer than the window is tiled entirely in-bounds and nothing is flagged.

This module is ``numpy``-only and torch-free, so it runs in the bare CI tier; a CPU
``torch.Tensor`` of logits is accepted directly via ``numpy.asarray``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from tbox_finder.labels import CLASS_ORDER

SCHEMA_VERSION = "1.0"
STEP = "P2-03"

#: Number of per-nucleotide segmentation classes (PRD §8; `labels.CLASS_ORDER` is source).
NUM_CLASSES = len(CLASS_ORDER)

#: Pinned scan tiling (PRD §6). Mirrored from `data.window_dataset`; drift-guarded by test.
WINDOW_NT = 1_024
STRIDE_NT = 512

#: The frozen operator identity. Changing any of these changes the science.
#: D3 pins the reduction; **A3** (user sign-off 2026-07-16) disambiguates it to the
#: per-window-normalised form — see the module docstring. Frozen in code with no config
#: override, on the A1/A2 precedent, so no report can contradict the pin.
OPERATOR = "log_mean_exp_of_per_window_log_softmax_then_argmax"
OPERATOR_PINNED = True
OPERATOR_PINNED_BY = "ADR-0005 D3 + A3 (locus-construction rule); PRD §6, §13.1"


@dataclass(frozen=True, eq=False)
class Reconciled:
    """Per-position reconciliation of every covering window, over ``[0, seq_len)``.

    Attributes
    ----------
    log_probs:
        ``(seq_len, NUM_CLASSES)`` float64. ``log`` of the coverage-averaged posterior;
        ``exp(log_probs).sum(axis=1)`` is 1 to floating-point tolerance.
    prediction:
        ``(seq_len,)`` int16 class index — ``argmax(log_probs, axis=1)``. Ties resolve to
        the lowest class index (``numpy.argmax`` semantics); at the `labels.CLASS_ORDER`
        ordering that favours ``background`` (index 0), the conservative call.
    coverage:
        ``(seq_len,)`` int32 count of windows that scored the position. Always >= 1.
    zero_flanked:
        ``(seq_len,)`` bool — the position was scored by at least one window that ran off
        a contig end and was zero-padded there (PRD §6 zero-flank-and-flag).
    """

    log_probs: np.ndarray
    prediction: np.ndarray
    coverage: np.ndarray
    zero_flanked: np.ndarray
    seq_len: int
    window: int
    n_windows: int


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable ``log(sum(exp(x)))`` along ``axis``.

    Shift-and-exponentiate: the maximum is factored out so no ``exp`` overflows. Written
    here rather than taken from ``scipy.special`` because this module is imported on the
    bare Tier-1 CI path, whose install is ``pytest``/``pyarrow``/``numpy``/``pandas``/
    ``scikit-learn`` — scipy arrives only incidentally, as a scikit-learn transitive
    dependency, and depending on it for four lines would make that accident load-bearing.
    No other numpy log-sum-exp exists in the repo (``models/seg_head.py`` has a torch one,
    inside the CRF forward algorithm).

    A non-finite maximum is shifted by 0 instead, so the degenerate rows return their
    correct limits (all ``-inf`` -> ``-inf``; any ``+inf`` -> ``+inf``) rather than the
    ``inf - inf = nan`` the naive shift produces.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True))
    out = m_safe + total
    return out if keepdims else np.squeeze(out, axis=axis)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log(softmax(x))`` along ``axis``.

    ``over`` is silenced because the subtraction legitimately saturates to ``-inf`` for
    absurd inputs (``|logit| ~ 1e308``); :func:`reconcile_windows` fails closed on the
    resulting non-finite reduction rather than letting a warning stand in for a guard.
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return x - logsumexp(x, axis=axis, keepdims=True)


def reconcile_windows(
    window_logits: Any,
    starts: Any,
    seq_len: int,
) -> Reconciled:
    """Reconcile per-window logits into one per-position prediction (ADR-0005 D3).

    Parameters
    ----------
    window_logits:
        ``(n_windows, window, NUM_CLASSES)`` array-like of per-nucleotide class logits —
        raw model output, *not* pre-normalised (this function normalises per window). A
        CPU ``torch.Tensor`` is accepted. Logits at positions outside ``[0, seq_len)``
        (a zero-padded contig end) are ignored and need not be finite.
    starts:
        ``(n_windows,)`` of window start offsets, as returned by
        :func:`tbox_finder.data.window_dataset.tile_windows`. Window ``k`` covers the
        half-open span ``[starts[k], starts[k] + window)``; offsets may be negative or
        overrun ``seq_len`` (a zero-flanked contig end). Order is irrelevant: the
        reduction is canonicalised by start offset, so for **distinct** offsets — which is
        all ``tile_windows`` ever emits — a permuted input reconciles **bit-identically**.
        Duplicate offsets carrying *different* logits tie-break on input position, so they
        are invariant only to floating-point tolerance (summation order); duplicate offsets
        carrying *identical* logits stay bit-exact, which is the coverage-invariance case
        that matters.
    seq_len:
        Length of the underlying sequence. Every position in ``[0, seq_len)`` must be
        covered by at least one window; a gap is an error, never a silent ``background``.

    Returns
    -------
    Reconciled

    Raises
    ------
    ValueError
        On any malformed input: bad shape or class count, ``starts``/``n_windows``
        mismatch, non-positive ``seq_len``/``window``, a non-finite contributing logit, a
        window entirely outside the sequence, or an uncovered position.
    """
    logits = np.asarray(window_logits, dtype=np.float64)
    if logits.ndim != 3:
        raise ValueError(
            f"window_logits must be (n_windows, window, {NUM_CLASSES})-shaped, "
            f"got ndim={logits.ndim} shape={logits.shape}"
        )
    n_windows, window, n_classes = logits.shape
    if n_classes != NUM_CLASSES:
        raise ValueError(
            f"window_logits must carry {NUM_CLASSES} classes "
            f"(labels.CLASS_ORDER), got {n_classes}"
        )
    if n_windows <= 0:
        raise ValueError("window_logits must carry at least one window, got 0")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    seq_len_int = _as_index(seq_len, "seq_len")
    if seq_len_int <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len_int}")

    # Read `starts` WITHOUT a numpy round-trip when the caller passed a Python sequence:
    # `np.asarray([True, 5])` promotes to int64, silently turning the bool into 1 before
    # `_as_index` can reject it. (An ndarray the caller *built* that way has already lost
    # the bool at their call site; a bool-dtype ndarray still round-trips to Python bools
    # via `.tolist()` and is rejected.)
    if isinstance(starts, np.ndarray):
        raw_starts: list[Any] = starts.reshape(-1).tolist()
    elif hasattr(starts, "__iter__") and not isinstance(starts, (str, bytes)):
        raw_starts = list(starts)
    else:
        raw_starts = [starts]
    start_list = [_as_index(s, "starts") for s in raw_starts]
    if len(start_list) != n_windows:
        raise ValueError(
            f"starts carries {len(start_list)} offsets but window_logits carries "
            f"{n_windows} windows"
        )

    # Canonical reduction order (by start offset) so a permuted caller input reconciles
    # bit-identically, not merely to floating-point tolerance.
    order = sorted(range(n_windows), key=lambda k: (start_list[k], k))

    # Slice to the in-bounds span *before* any arithmetic: a zero-padded contig end
    # carries logits that describe no DNA, so they must not reach the reduction at all
    # (not even as a NaN that a later mask would have to chase).
    spans: list[tuple[int, int, np.ndarray, bool]] = []
    for k in order:
        start = start_list[k]
        lo = max(start, 0)
        hi = min(start + window, seq_len_int)
        if hi <= lo:
            raise ValueError(
                f"window {k} at start={start} (width {window}) covers no position of "
                f"[0, {seq_len_int})"
            )
        contribution = logits[k, lo - start : hi - start]
        if not np.all(np.isfinite(contribution)):
            raise ValueError(
                f"window {k} carries a non-finite logit at an in-bounds position "
                f"(pad positions outside [0, seq_len) are exempt)"
            )
        # Per-window log-softmax: the reduction must weight a window by the distribution
        # it predicts, never by an arbitrary constant offset of its logits (see docstring).
        padded = start < 0 or start + window > seq_len_int
        spans.append((lo, hi, log_softmax(contribution, axis=-1), padded))

    coverage = np.zeros(seq_len_int, dtype=np.int64)
    zero_flanked = np.zeros(seq_len_int, dtype=bool)
    running_max = np.full((seq_len_int, NUM_CLASSES), -np.inf, dtype=np.float64)

    for lo, hi, log_p, padded in spans:
        np.maximum(running_max[lo:hi], log_p, out=running_max[lo:hi])
        coverage[lo:hi] += 1
        if padded:
            zero_flanked[lo:hi] = True

    uncovered = int(np.count_nonzero(coverage == 0))
    if uncovered:
        first = int(np.flatnonzero(coverage == 0)[0])
        raise ValueError(
            f"{uncovered} of {seq_len_int} positions are covered by no window "
            f"(first at {first}); the tiling does not span the sequence"
        )

    shifted_sum = np.zeros((seq_len_int, NUM_CLASSES), dtype=np.float64)
    for lo, hi, log_p, _ in spans:
        with np.errstate(invalid="ignore"):
            shifted_sum[lo:hi] += np.exp(log_p - running_max[lo:hi])

    # log(mean_w exp(log_p_w)) == max + log(sum(exp(x - max)) / n). Dividing *inside* the
    # log (rather than subtracting log(n) afterwards) makes coverage-invariance exact: at
    # n identical contributions the ratio is exactly 1.0, log(1.0) is exactly 0.0, and the
    # result is bit-identical to the single-window value.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = running_max + np.log(shifted_sum / coverage[:, None])

    # Fail closed on a non-finite reconciliation. `np.argmax` treats NaN as the maximum, so
    # a single NaN does not merely propagate — it silently *selects a class*. Reachable
    # only from absurd input (|logit| ~ 1e308 overflows the log-softmax shift to -inf, and
    # -inf - -inf is NaN), which no trained head produces; but the whole point of the guard
    # is that a wrong call must never leave this function looking like a real one (§10.3).
    if not np.all(np.isfinite(log_probs)):
        bad = int(np.count_nonzero(~np.isfinite(log_probs).all(axis=1)))
        raise ValueError(
            f"reconciliation produced a non-finite log-probability at {bad} of "
            f"{seq_len_int} positions; the input logits are outside the range this "
            f"operator can reduce (argmax would silently select the NaN)"
        )
    prediction = np.argmax(log_probs, axis=1).astype(np.int16)

    return Reconciled(
        log_probs=log_probs,
        prediction=prediction,
        coverage=coverage.astype(np.int32),
        zero_flanked=zero_flanked,
        seq_len=seq_len_int,
        window=int(window),
        n_windows=int(n_windows),
    )


def diagnostics() -> dict[str, Any]:
    """Machine-readable record of the frozen operator (provenance, §11)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "operator": OPERATOR,
        "pinned": OPERATOR_PINNED,
        "pinned_by": OPERATOR_PINNED_BY,
        "num_classes": NUM_CLASSES,
        "class_order": list(CLASS_ORDER),
        "window_nt": WINDOW_NT,
        "stride_nt": STRIDE_NT,
        "normalises_per_window": True,
        "coverage_normalised": True,
        "ignores_pad_positions": True,
    }


def _as_index(value: Any, field: str) -> int:
    """Coerce to ``int`` accepting numpy integers, rejecting ``bool`` and non-integers.

    ``isinstance(True, int)`` is True and ``isinstance(np.int64(1), int)`` is False, so
    neither a bare ``isinstance`` check nor ``int()`` is safe here (P1-16 / P2-02 lesson).
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be an integer, got bool {value!r}")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
