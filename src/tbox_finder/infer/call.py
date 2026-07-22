"""P2-10c′-c-i — the along-sequence candidate-caller: reconciled per-position posteriors
in, called Stage-1 candidate loci out.

Where it sits
-------------
:func:`tbox_finder.infer.scan.scan_sequence` stops at the reconciled per-position
distribution (the ADR-0005 D3 + A3 operator in :mod:`tbox_finder.infer.reconcile`); its own
docstring names *along-sequence locus construction (threshold, minimum span, gap-merge)* as
"a later step". This module is that step, in the **minimal, recall-favouring form the ρ-pilot
needs** — nothing more.

What it is for, and the pin discipline it observes
--------------------------------------------------
The immediate consumer is the ρ-pilot (ADR-0003 **D6**): scan ~100 divergent-clade genomes
with the production Stage-1 checkpoint and count **Stage-1 candidates / Mbp = ρ**, the
unmeasured pivot that sizes the shared GTDB/RefSeq homolog-DB + negative-window fetch
(0.7–68 GB). ρ is a **measured ops number, not a scientific-claim gate** (ADR-0003 D6: it
"does not own any gate threshold that touches the scientific claim").

ADR-0005 **D3** pins the Stage-1 threshold and the locus-construction rule as *rules* whose
**values are frozen at the phase gate (§13.1), never at a sizing pilot** — and mining (P2-10e)
will replace this round-0 checkpoint, so any value frozen now would be re-derived later. This
module therefore **pins no value**: ``threshold``, ``min_span`` and ``gap_merge`` are
keyword-only with **no defaults** (a production value cannot be taken by accident — the
P2-10c ``min_sequences`` / PEFT-``target_modules`` lesson), and the ρ-pilot reports ρ as a
**function** of them (a sweep — user decision 2026-07-22), not a single number. The provisional
sweep grids (:data:`PROVISIONAL_THRESHOLD_GRID` etc.) are labelled as such and bind nothing.

The operator (user decision 2026-07-22: "minimal recall-favouring")
-------------------------------------------------------------------
For a reconciled ``log_probs`` of shape ``(seq_len, 8)`` (class 0 = ``background``, 1..7 the
T-box elements, :data:`tbox_finder.labels.CLASS_ORDER`):

1. **Element score**, global scope: ``p_elem[i] = 1 − exp(log_probs[i, background])`` — the
   posterior mass on *any* element. Global rather than per-class, and with **no
   required-element co-occurrence**, because D3 warns that "mandating canonical elements
   would re-impose the §5/§13.3 bias"; the flagship discovery class is the *non-canonical*
   Tier-2N T-box, so a sizing caller that demanded canonical architecture would under-count
   exactly what the project exists to find. This is the most-liberal (highest-recall) reading,
   which makes ρ a **conservative upper bound** on the fetch — the safe direction to size in.
2. **Binarise** at ``τ``: ``elem[i] = p_elem[i] >= threshold``.
3. **Gap-merge**: two element runs separated by ``<= gap_merge`` background positions are one
   candidate (``gap_merge = 0`` merges nothing — consecutive distinct runs are always ≥ 1 apart).
4. **Minimum span**: a merged run of ``>= min_span`` positions is a candidate; shorter ones drop.

Each surviving run is one :class:`Candidate`. ``ρ = (Σ candidates over all genomes) / T[Mbp]``.

Contig ends are flagged, not dropped
------------------------------------
PRD §6 / ADR-0005 D3: contig ends are zero-flanked *and flagged*. A short replicon (the pilot
genomes carry a median of 105 replicons, some < 1024 nt) is scored with synthetic pad context;
:class:`Reconciled` marks those positions in ``zero_flanked``. Recall-favouring means such a
candidate is **kept and counted**, but its ``n_zero_flanked`` is recorded so the ρ report can
show ρ's sensitivity to excluding zero-flanked candidates — never a silent drop (§10.3).

``numpy``-only and torch-free, so it imports and unit-tests on the bare CI Tier-1 path, exactly
like :mod:`tbox_finder.infer.reconcile`. PRD §6, §13.1; ADR-0005 D3 + A3; ADR-0003 D6.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tbox_finder.labels import CLASS_INDEX, CLASS_ORDER

SCHEMA_VERSION = "1.0"
STEP = "P2-10c′-c"

#: Number of per-nucleotide segmentation classes (PRD §8; ``labels.CLASS_ORDER`` is source).
NUM_CLASSES = len(CLASS_ORDER)

#: Softmax index of the ``background`` class — the position whose complement is the element
#: score. Read from :data:`tbox_finder.labels.CLASS_INDEX` rather than hardcoded ``0`` so a
#: re-ordering of ``CLASS_ORDER`` cannot silently point the caller at an element class.
BACKGROUND_INDEX = CLASS_INDEX["background"]

#: Provisional ρ-pilot sweep grids (user decision 2026-07-22: "sweep ρ(τ), pin nothing").
#: These bind **no** ADR value — ADR-0005 D3 freezes the production Stage-1 threshold and the
#: locus geometry at the phase gate (§13.1). They exist only to shape the ρ(τ, min_span,
#: gap_merge) surface the pilot reports; -c-ii may widen or refine them at will.
PROVISIONAL_THRESHOLD_GRID: tuple[float, ...] = (0.5, 0.7, 0.9, 0.95, 0.99)
PROVISIONAL_MIN_SPAN_GRID: tuple[int, ...] = (20, 50, 100)
PROVISIONAL_GAP_MERGE_GRID: tuple[int, ...] = (0, 10, 50)


class CandidateError(ValueError):
    """Raised on malformed caller input or an out-of-range threshold / span / gap."""


@dataclass(frozen=True)
class Candidate:
    """One called Stage-1 candidate locus over ``[start, end)`` of a scanned sequence.

    Attributes
    ----------
    start, end:
        Half-open span, ``0 <= start < end <= seq_len``.
    length:
        ``end - start`` — the merged element run length in nucleotides.
    peak_p_elem, mean_p_elem:
        Max and mean of ``1 − P(background)`` over the run — the strength of the call,
        reported (not gated on) so the ρ table can show how ρ moves with a strength cut.
    n_zero_flanked:
        Count of positions in the run scored by at least one zero-padded (contig-end) window.
        ``> 0`` marks a call whose context included synthetic pad — kept and counted
        (recall-favouring), flagged so it can be excluded downstream if wanted.
    dominant_class:
        Index (1..7) of the element class with the greatest mean posterior over the run — the
        element that most drives the call. Descriptive only; the caller mandates no class.
    """

    start: int
    end: int
    length: int
    peak_p_elem: float
    mean_p_elem: float
    n_zero_flanked: int
    dominant_class: int


def element_posterior(log_probs: Any) -> np.ndarray:
    """``(seq_len,)`` float64 of ``1 − P(background)`` — the global element score.

    ``log_probs`` is the reconciled ``(seq_len, NUM_CLASSES)`` log-posterior
    (:class:`tbox_finder.infer.reconcile.Reconciled.log_probs`); a CPU ``torch.Tensor``
    converts through ``numpy.asarray``. The result is clipped to ``[0, 1]`` so a
    ``P(background)`` that reconciles to ``1 + 1e-16`` cannot yield a spuriously negative
    score — the clip only ever touches floating-point dust, never a real distinction.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[1] != NUM_CLASSES:
        raise CandidateError(
            f"log_probs must be (seq_len, {NUM_CLASSES})-shaped, got shape={lp.shape}"
        )
    if lp.shape[0] == 0:
        raise CandidateError("log_probs carries no positions; there is nothing to call")
    if not np.all(np.isfinite(lp)):
        raise CandidateError(
            "log_probs carries a non-finite value; a reconciled distribution is finite "
            "(reconcile_windows fails closed on non-finite input) — this is not one"
        )
    p_bg = np.exp(lp[:, BACKGROUND_INDEX])
    return np.clip(1.0 - p_bg, 0.0, 1.0)


def _merge_runs(runs: list[tuple[int, int]], gap_merge: int) -> list[tuple[int, int]]:
    """Merge consecutive ``[start, end)`` runs separated by ``<= gap_merge`` positions.

    ``runs`` are the maximal True runs, already in ascending, non-overlapping order. The gap
    between run ``A=[s1,e1)`` and the next ``B=[s2,e2)`` is ``s2 - e1`` background positions
    (always ``>= 1`` for distinct runs), so ``gap_merge = 0`` merges nothing and ``gap_merge
    = g`` bridges any background gap of length ``<= g``.
    """
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= gap_merge:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans of every maximal True run in a 1-D bool array.

    Sentinel-padded rising/falling-edge detection: with a ``False`` sentinel on each end, a
    ``+1`` diff marks a run start and a ``-1`` diff a run end, both already in original
    coordinates. ``np.diff`` / ``np.flatnonzero`` are ABI-stable across the numpy 2.x the
    repo pins (``ml-dna`` 2.5.1), so no version-sensitive API is on this path.
    """
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    # rising/falling edges pair up exactly (a sentinel-padded bool run has equal counts),
    # so strict=True is a free correctness assertion, not a constraint.
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def call_candidates(
    log_probs: Any,
    zero_flanked: Any = None,
    *,
    threshold: float,
    min_span: int,
    gap_merge: int,
) -> list[Candidate]:
    """Call Stage-1 candidate loci from reconciled per-position posteriors (the D3 operator).

    Parameters
    ----------
    log_probs:
        Reconciled ``(seq_len, NUM_CLASSES)`` log-posterior — ``Reconciled.log_probs``. The
        scan path calls ``call_candidates(r.log_probs, r.zero_flanked, ...)``.
    zero_flanked:
        Optional ``(seq_len,)`` bool — ``Reconciled.zero_flanked``. ``None`` is treated as
        all-False (no contig-end padding), which is the interior-sequence case.
    threshold:
        ``τ`` in ``[0, 1]``. A position is element-like iff ``1 − P(background) >= τ``.
        Keyword-only, no default — this is a swept sizing parameter, never a pinned value.
    min_span:
        Minimum merged-run length in nucleotides (``>= 1``); shorter runs are not candidates.
    gap_merge:
        Maximum background gap in nucleotides to bridge between element runs (``>= 0``).

    Returns
    -------
    list[Candidate]
        Called loci in ascending ``start`` order; empty if none clear the operator.
    """
    p_elem = element_posterior(log_probs)
    seq_len = p_elem.shape[0]

    if not (0.0 <= float(threshold) <= 1.0):
        raise CandidateError(f"threshold must be in [0, 1], got {threshold}")
    if int(min_span) < 1:
        raise CandidateError(f"min_span must be >= 1, got {min_span}")
    if int(gap_merge) < 0:
        raise CandidateError(f"gap_merge must be >= 0, got {gap_merge}")
    min_span = int(min_span)
    gap_merge = int(gap_merge)

    if zero_flanked is None:
        zf = np.zeros(seq_len, dtype=bool)
    else:
        zf = np.asarray(zero_flanked, dtype=bool)
        if zf.shape != (seq_len,):
            raise CandidateError(
                f"zero_flanked must be (seq_len,)=({seq_len},)-shaped, got shape={zf.shape}"
            )

    lp = np.asarray(log_probs, dtype=np.float64)
    elem = p_elem >= float(threshold)
    runs = _merge_runs(_true_runs(elem), gap_merge)

    candidates: list[Candidate] = []
    for start, end in runs:
        if end - start < min_span:
            continue
        run_p = p_elem[start:end]
        # Mean posterior per class over the run; argmax among the element classes (1..7)
        # names the driving element. Descriptive only — nothing is gated on it.
        run_class_post = np.exp(lp[start:end]).mean(axis=0)
        dominant = 1 + int(np.argmax(run_class_post[1:]))
        candidates.append(
            Candidate(
                start=int(start),
                end=int(end),
                length=int(end - start),
                peak_p_elem=float(run_p.max()),
                mean_p_elem=float(run_p.mean()),
                n_zero_flanked=int(np.count_nonzero(zf[start:end])),
                dominant_class=dominant,
            )
        )
    return candidates


def sweep_candidate_counts(
    log_probs: Any,
    zero_flanked: Any = None,
    *,
    thresholds: Sequence[float] = PROVISIONAL_THRESHOLD_GRID,
    min_spans: Sequence[int] = PROVISIONAL_MIN_SPAN_GRID,
    gap_merges: Sequence[int] = PROVISIONAL_GAP_MERGE_GRID,
) -> list[dict[str, Any]]:
    """ρ-pilot sweep: candidate count per ``(threshold, min_span, gap_merge)`` grid point.

    The GPU forward is independent of ``τ / min_span / gap_merge``, so a scan forwards each
    sequence once (in -c-ii's sbatch) and calls this on the reconciled posteriors: the element
    score is computed once per threshold and the cheap merge/span filter is re-run per
    ``(min_span, gap_merge)``. Returns one row per grid point — the raw material of the
    ρ(τ, min_span, gap_merge) surface, summed over genomes and divided by T[Mbp] to give ρ.

    Each row also carries ``n_zero_flanked_candidates`` (candidates whose context included
    contig-end pad) so ρ's sensitivity to excluding them is visible per grid point.
    """
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for min_span in min_spans:
            for gap_merge in gap_merges:
                calls = call_candidates(
                    log_probs,
                    zero_flanked,
                    threshold=threshold,
                    min_span=min_span,
                    gap_merge=gap_merge,
                )
                rows.append(
                    {
                        "threshold": float(threshold),
                        "min_span": int(min_span),
                        "gap_merge": int(gap_merge),
                        "n_candidates": len(calls),
                        "n_zero_flanked_candidates": sum(1 for c in calls if c.n_zero_flanked > 0),
                        "total_candidate_nt": sum(c.length for c in calls),
                    }
                )
    return rows


__all__ = [
    "BACKGROUND_INDEX",
    "PROVISIONAL_GAP_MERGE_GRID",
    "PROVISIONAL_MIN_SPAN_GRID",
    "PROVISIONAL_THRESHOLD_GRID",
    "Candidate",
    "CandidateError",
    "NUM_CLASSES",
    "SCHEMA_VERSION",
    "STEP",
    "call_candidates",
    "element_posterior",
    "sweep_candidate_counts",
]
