"""P3-12 — the ADR-0005 D3 locus-construction rule: reconciled per-position posteriors in,
candidate loci ± flank out.

Where it sits
-------------
:func:`tbox_finder.infer.reconcile.reconcile_windows` (P2-03, the operator ADR-0005 **A3**
froze in code) collapses every covering window into one per-position distribution;
:func:`tbox_finder.infer.call.call_candidates` (P2-10c′-c-i) merges that into candidate spans
under a *global* threshold with no co-occurrence and no flank — the minimal recall-favouring
form the ρ-pilot needed. This module is D3's **full** locus-construction rule, the last
Stage-1 step before the P3-13 strand-resolver and the Stage-2 handoff (PRD §6, §13.1).

D3 [ADR-0005:46] lists exactly five knobs:

===========================================  =====================================
knob                                         where it lives
===========================================  =====================================
per-class-vs-global threshold **scope**      here (:func:`element_assignment`)
minimum span                                 ``call.candidates_from_mask``
gap-merge distance                           ``call.candidates_from_mask``
required-element **co-occurrence**           here
flank size                                   here
===========================================  =====================================

Composition, not a fork
-----------------------
The merge is **not** re-derived. :func:`construct_loci` builds the scope-specific mask and
hands it to :func:`tbox_finder.infer.call.candidates_from_mask` — the single mask →
gap-merge → min-span → :class:`~tbox_finder.infer.call.Candidate` site in the repo, which
``call_candidates`` also calls. D3 pins **one** locus operator; a second copy of the merge
would mean fixing one and shipping the bug in the other, the rule
:mod:`tbox_finder.eval.mining_criterion` already observes for the caller as a whole and the
one P3-11 applied to the reconciliation operator. Each surviving
:class:`~tbox_finder.infer.call.Candidate` is carried **inside** its :class:`Locus` rather
than copied field-by-field, so the core span and its strength summaries have exactly one
representation and cannot drift.

The co-occurrence rule is a COUNT, never a named set
----------------------------------------------------
D3 requires "**recall-favoring** required-element co-occurrence" and warns *in the same
clause* that "mandating canonical elements would re-impose the §5/§13.3 bias". The flagship
discovery class is the **non-canonical, CM-invisible Tier-2N** locus, so a rule demanding
Stem I *and* a Specifier *and* an antiterminator would reject exactly what the project exists
to find. This module therefore exposes ``min_distinct_elements`` — *how many* distinct element
classes a locus must contain — and exposes **no** parameter that can name *which*. The
predicate is class-identity-blind by construction, which is a testable property rather than a
promise: permuting which element class is which leaves the surviving locus set unchanged
(``test_co_occurrence_is_class_identity_blind``).

**The recall floor.** At ``min_distinct_elements <= 1`` the rule discards nothing: every
merged run contains at least one masked position, and every masked position is assigned some
element class, so ``n_distinct_elements >= 1`` always. Under ``threshold_scope="global"`` the
mask is bit-identical to ``call_candidates``'s, so the cores are the *same spans* — the
Stage-1 per-locus recall floor D3 defines is preserved **exactly**, not approximately
(``test_global_scope_at_k1_loses_nothing_the_caller_found``). Every value above 1 trades
recall for specificity and is therefore a phase-gate quantity, not a default.

Pin discipline: this module pins NO value
------------------------------------------
ADR-0005 D3 freezes the locus-rule *values* at the §13.1 phase gate, "never tuned on the test
set", and A9 Pin 3 keeps the mining-internal triple explicitly non-binding on that freeze.
Accordingly **every rule parameter here is keyword-only with no default** — scope, threshold,
minimum span, gap-merge, co-occurrence count and flank alike. A production value cannot be
taken by accident, and there is no constant in this module for a later reader to mistake for
a frozen one. :func:`no_rule_parameter_has_a_default` re-derives that from the live signature
so the discipline is checked, not merely stated.

Nothing is gated on a strength value
-------------------------------------
Selection reads posteriors only through the threshold. ``peak_p_elem`` / ``mean_p_elem`` ride
along on the carried :class:`~tbox_finder.infer.call.Candidate` as *reported* quantities:
P3-11 measured ``mean_p_elem`` swinging 0.6453 → 0.9997 with the 512-window phase while the
per-position call was exactly invariant, so a locus rule thresholding on mean strength would
re-import the very 512-grid dependence D3's reconciliation operator exists to remove.

Kept and flagged, never silently dropped
-----------------------------------------
Recall-favouring extends to the diagnostics. A locus whose context included contig-end pad is
**counted**, with ``n_zero_flanked_span`` recording it (PRD §6 / D3: contig ends are
zero-flanked *and flagged*). Where ``coverage`` is supplied, ``n_single_covered_span`` records
positions scored by only one window — the single-coverage terminus P3-11 measured as
phase-dependent and left downstream-invisible because
:class:`~tbox_finder.infer.call.Candidate` has no coverage field. :class:`Locus` is a new
record, so carrying it here costs no P2 schema change. It matters most on short MAG contigs,
where the singly-covered 1,024 nt is a large fraction of the sequence.

``numpy``-only and torch-free, so it imports and unit-tests on the bare CI Tier-1 path,
exactly like :mod:`tbox_finder.infer.call` and :mod:`tbox_finder.infer.reconcile`.
PRD §6, §13.1, §5/§13.3; ADR-0005 D3 + A3 + A9 Pin 3.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tbox_finder.infer.call import (
    ELEMENT_INDICES,
    NUM_ELEMENT_CLASSES,
    Candidate,
    CandidateError,
    candidates_from_mask,
    element_posterior,
    zero_flanked_array,
)
from tbox_finder.labels import CLASS_ORDER

SCHEMA_VERSION = "1.0"
STEP = "P3-12"

#: The two D3 threshold **scopes**. ``"global"`` thresholds ``1 − P(background)`` — posterior
#: mass on *any* element, the ``call_candidates`` reading. ``"per_class"`` thresholds each
#: element class against its own ``τ_c``, which is what lets a scope choice express "call this
#: class more liberally than that one". D3 pins the *choice* at the §13.1 phase gate; this
#: module implements both and defaults to neither.
THRESHOLD_SCOPES: tuple[str, ...] = ("global", "per_class")

ThresholdScope = Literal["global", "per_class"]

#: Element-class **names**, in ``ELEMENT_INDICES`` order — the exact key set a ``per_class``
#: threshold mapping must carry. Derived from :data:`tbox_finder.labels.CLASS_ORDER` via the
#: caller's own index array, never re-typed, so a re-ordering of ``CLASS_ORDER`` moves both
#: together. ``background`` is absent by construction: it is not a thresholdable element.
ELEMENT_CLASS_NAMES: tuple[str, ...] = tuple(CLASS_ORDER[i] for i in ELEMENT_INDICES)

#: Sentinel in :func:`element_assignment` for "this position is not element-like". Negative so
#: it can never collide with a class index, and so ``assignment >= 0`` *is* the element mask.
NOT_ELEMENT = -1

#: The five D3 knobs as they appear on :func:`construct_loci` (``min_span`` and ``gap_merge``
#: are forwarded to the composed merge, so six names). Every one is keyword-only with **no
#: default** — :func:`no_rule_parameter_has_a_default` re-derives that from the live signature.
RULE_PARAMETERS: tuple[str, ...] = (
    "threshold_scope",
    "threshold",
    "min_span",
    "gap_merge",
    "min_distinct_elements",
    "flank",
)


class LocusError(ValueError):
    """Raised on a malformed locus-rule parameter or input.

    Distinct from :class:`tbox_finder.infer.call.CandidateError`, which the composed merge
    still raises for its own inputs (``log_probs`` shape/finiteness, ``min_span``,
    ``gap_merge``, ``zero_flanked`` shape) — the two are not merged into one type precisely so
    a failure is attributable to the layer that refused it.
    """


@dataclass(frozen=True)
class Locus:
    """One candidate locus: a called element core, its flanked span, and its co-occurrence.

    Attributes
    ----------
    candidate:
        The core call — the merged element run, as
        :class:`tbox_finder.infer.call.Candidate`. Carried whole rather than copied out, so
        ``candidate.start`` / ``candidate.end`` are the **only** representation of the core
        span and ``peak_p_elem`` / ``mean_p_elem`` / ``n_zero_flanked`` / ``dominant_class``
        keep their P2 meanings unchanged (all core-scoped, all reported, none gated on).
    start, end, length:
        The **flanked** half-open span handed downstream, ``0 <= start <= candidate.start <
        candidate.end <= end <= seq_len``, and ``end - start``. Coordinates are in the frame of
        the sequence that was scanned; strand and 5′→3′ orientation are P3-13's, not this
        rule's, so the two shortfall fields below are named left/right rather than 5′/3′.
    flank:
        The flank requested, in nucleotides, on each side.
    flank_short_left, flank_short_right:
        Nucleotides of requested flank that did not exist because the sequence ended —
        ``0`` when the full flank was available. Recorded rather than allowed to be inferred
        from the span, so a locus truncated at a contig edge is visible downstream.
    element_classes:
        Distinct element-class indices present in the **core**, ascending. "Present" means the
        class was the assignment at ≥ 1 core position (:func:`element_assignment`); bridged
        gap positions are assigned :data:`NOT_ELEMENT` and contribute nothing.
    n_distinct_elements:
        ``len(element_classes)`` — the quantity the co-occurrence rule thresholds. A count,
        never an identity: see the module docstring.
    n_zero_flanked_span:
        Positions in the **flanked** span scored by at least one zero-padded (contig-end)
        window. The core-only count is ``candidate.n_zero_flanked``; the two are deliberately
        differently named because they measure different intervals.
    n_single_covered_span:
        Positions in the flanked span covered by exactly one window, or ``None`` when no
        ``coverage`` array was supplied. ``None`` means **not measured** — it does not mean
        zero. Tail-anchored tiling leaves the first and last window of a contig singly
        covered, which P3-11 measured as the one phase-dependent region.
    overlaps_neighbour:
        True when this locus's flanked span intersects an adjacent locus's. Flanking runs
        *after* gap-merging, so two cores further apart than ``gap_merge`` can still overlap
        once padded; they are reported separately (never re-merged, which would undo the
        gap-merge decision) and the intersection is flagged rather than left to surprise a
        downstream consumer counting candidates.
    """

    candidate: Candidate
    start: int
    end: int
    length: int
    flank: int
    flank_short_left: int
    flank_short_right: int
    element_classes: tuple[int, ...]
    n_distinct_elements: int
    n_zero_flanked_span: int
    n_single_covered_span: int | None
    overlaps_neighbour: bool


def _validate_threshold(
    threshold_scope: str, threshold: Any
) -> tuple[float | None, np.ndarray | None]:
    """Resolve ``(scope, threshold)`` into either a scalar τ or a per-element τ vector.

    Fail-closed on every mismatch: a scalar under ``per_class`` (which class would it be?), a
    mapping under ``global``, a partial mapping (an unlisted class would be silently
    uncallable — a recall loss recorded nowhere), an unknown key, or ``background`` as a key
    (it is the complement of the element mass, not an element to threshold).
    """
    if threshold_scope not in THRESHOLD_SCOPES:
        raise LocusError(
            f"threshold_scope must be one of {THRESHOLD_SCOPES}, got {threshold_scope!r}"
        )

    if threshold_scope == "global":
        if isinstance(threshold, Mapping):
            raise LocusError(
                "threshold_scope='global' takes a single τ applied to 1 − P(background); a "
                "mapping is the per_class form — pass threshold_scope='per_class' to use it"
            )
        tau = float(threshold)
        if not (0.0 <= tau <= 1.0):
            raise LocusError(f"threshold must be in [0, 1], got {threshold}")
        return tau, None

    if not isinstance(threshold, Mapping):
        raise LocusError(
            "threshold_scope='per_class' takes a mapping {element class name: τ_c} covering "
            f"all {NUM_ELEMENT_CLASSES} element classes, got {type(threshold).__name__}"
        )
    keys = set(threshold)
    expected = set(ELEMENT_CLASS_NAMES)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise LocusError(
            "a per_class threshold mapping must carry exactly the element classes "
            f"{list(ELEMENT_CLASS_NAMES)}; missing={missing}, unknown={unknown}. A partial "
            "mapping is refused rather than defaulted: an unlisted class would become "
            "uncallable, a recall loss with nothing recording it"
        )
    tau_vec = np.array([float(threshold[name]) for name in ELEMENT_CLASS_NAMES], dtype=np.float64)
    in_range = (tau_vec >= 0.0) & (tau_vec <= 1.0)
    if not np.all(in_range):
        bad = {
            name: float(tau_vec[j]) for j, name in enumerate(ELEMENT_CLASS_NAMES) if not in_range[j]
        }
        raise LocusError(f"every per_class threshold must be in [0, 1]; out of range: {bad}")
    return None, tau_vec


def element_assignment(
    log_probs: Any,
    *,
    threshold_scope: ThresholdScope,
    threshold: float | Mapping[str, float],
) -> np.ndarray:
    """``(seq_len,)`` int64 — the element class assigned to each position, or :data:`NOT_ELEMENT`.

    This is D3's **threshold-scope** knob, and the single place the scope choice is expressed.
    ``assignment >= 0`` is the element mask handed to
    :func:`tbox_finder.infer.call.candidates_from_mask`; the class values are what the
    co-occurrence count is taken over.

    * ``threshold_scope="global"`` — a position is element-like iff
      ``1 − P(background) >= τ``, bit-identical to
      :func:`tbox_finder.infer.call.call_candidates` (it calls the same
      :func:`~tbox_finder.infer.call.element_posterior`), and the assigned class is the
      arg-max over the element classes. Posterior mass on *any* element, so a position split
      across two element classes still clears τ.
    * ``threshold_scope="per_class"`` — a position is element-like iff **some** element class
      has ``P(c) >= τ_c``, and the assigned class is the arg-max among those that cleared.
      Strictly a different mask, not a re-parameterisation of the global one: a single high
      ``τ_c`` suppresses one class's positions while leaving every other class untouched,
      which the global scope cannot express at any τ.

    Ties are broken toward the lower class index (``np.argmax``'s first-max rule) —
    deterministic, and it touches only ``element_classes`` / ``dominant_class``, never whether
    a position is element-like.

    Raises
    ------
    LocusError
        On an unknown scope, or a threshold whose form does not match it.
    tbox_finder.infer.call.CandidateError
        On a ``log_probs`` that is not a finite, non-empty ``(seq_len, NUM_CLASSES)`` array
        (raised by the shared :func:`~tbox_finder.infer.call.element_posterior`).
    """
    tau, tau_vec = _validate_threshold(threshold_scope, threshold)

    p_elem = element_posterior(log_probs)  # also validates shape / finiteness / non-emptiness
    lp = np.asarray(log_probs, dtype=np.float64)
    elem_probs = np.exp(lp[:, ELEMENT_INDICES])

    if tau_vec is None:  # global scope — _validate_threshold returned the scalar τ
        selected = p_elem >= float(tau)
        scored = elem_probs
    else:
        cleared = elem_probs >= tau_vec[None, :]
        selected = cleared.any(axis=1)
        # -inf on the classes that did not clear, so the arg-max can only name a class that
        # actually passed its own τ_c. Rows where nothing cleared are all -inf; argmax returns
        # 0 there and is discarded by ``selected`` below.
        scored = np.where(cleared, elem_probs, -np.inf)

    winner = ELEMENT_INDICES[np.argmax(scored, axis=1)]
    return np.where(selected, winner, NOT_ELEMENT).astype(np.int64)


def _coverage_array(coverage: Any, seq_len: int) -> np.ndarray | None:
    """Normalise an optional ``Reconciled.coverage`` to a ``(seq_len,)`` integer array."""
    if coverage is None:
        return None
    cov = np.asarray(coverage)
    if cov.dtype.kind not in "iu":
        raise LocusError(
            f"coverage must be an integer array (windows per position), got dtype={cov.dtype}"
        )
    if cov.shape != (seq_len,):
        raise LocusError(f"coverage must be (seq_len,)=({seq_len},)-shaped, got shape={cov.shape}")
    if np.any(cov < 1):
        raise LocusError(
            "coverage carries a position covered by no window; reconcile_windows guarantees "
            "coverage >= 1 everywhere, so this is not a reconciled coverage array"
        )
    return cov


def construct_loci(
    log_probs: Any,
    zero_flanked: Any = None,
    coverage: Any = None,
    *,
    threshold_scope: ThresholdScope,
    threshold: float | Mapping[str, float],
    min_span: int,
    gap_merge: int,
    min_distinct_elements: int,
    flank: int,
) -> list[Locus]:
    """Construct candidate loci from reconciled per-position posteriors — D3's locus rule.

    The full five-knob rule: threshold under the chosen **scope**, gap-merge, minimum span
    (all three via the shared :func:`tbox_finder.infer.call.candidates_from_mask`), then
    required-element **co-occurrence**, then **flank**.

    Parameters
    ----------
    log_probs:
        Reconciled ``(seq_len, NUM_CLASSES)`` log-posterior —
        :attr:`tbox_finder.infer.reconcile.Reconciled.log_probs`.
    zero_flanked:
        Optional ``(seq_len,)`` bool — ``Reconciled.zero_flanked``. ``None`` is all-False.
    coverage:
        Optional ``(seq_len,)`` int — ``Reconciled.coverage``. Supplying it populates
        ``Locus.n_single_covered_span``; omitting it leaves that field ``None`` (**not
        measured**, which is not the same as zero).
    threshold_scope:
        ``"global"`` or ``"per_class"`` — see :func:`element_assignment`.
    threshold:
        ``τ`` in ``[0, 1]`` for ``"global"``; a mapping ``{element class name: τ_c}`` covering
        all element classes for ``"per_class"``.
    min_span:
        Minimum merged-run length in nucleotides (``>= 1``), applied to the **core**, before
        any flank is added — so the flank can never rescue a run that failed the span cut.
    gap_merge:
        Maximum background gap in nucleotides bridged between element runs (``>= 0``).
    min_distinct_elements:
        The co-occurrence count: a locus is kept iff at least this many **distinct** element
        classes appear in its core, ``0 <= k <= NUM_ELEMENT_CLASSES``. ``k <= 1`` keeps
        everything the merge produced (the recall floor); higher values trade recall for
        specificity. **No class can be named** — see the module docstring.
    flank:
        Nucleotides added on each side of the core (``>= 0``), clipped at the sequence ends
        with the shortfall recorded.

    Returns
    -------
    list[Locus]
        Loci in ascending core-``start`` order; empty if none clear the rule.

    Raises
    ------
    LocusError
        On a bad scope/threshold pairing, a co-occurrence count outside
        ``[0, NUM_ELEMENT_CLASSES]``, a negative flank, or a malformed ``coverage``.
    tbox_finder.infer.call.CandidateError
        On a malformed ``log_probs`` / ``zero_flanked`` / ``min_span`` / ``gap_merge``
        (raised by the composed merge, which owns those).

    Notes
    -----
    Ordering is **merge → co-occurrence → flank**, and it is load-bearing. Co-occurrence is
    read over the *core*, not the flanked span, so widening the flank can never change which
    loci survive — the flank is context handed to Stage-2, not evidence. Flanking last means
    two cores that gap-merge kept apart stay apart; if their padded spans intersect, that is
    reported via ``overlaps_neighbour`` rather than silently re-merged.
    """
    assignment = element_assignment(log_probs, threshold_scope=threshold_scope, threshold=threshold)
    seq_len = int(assignment.shape[0])

    if int(min_distinct_elements) < 0 or int(min_distinct_elements) > NUM_ELEMENT_CLASSES:
        raise LocusError(
            f"min_distinct_elements must be in [0, {NUM_ELEMENT_CLASSES}], got "
            f"{min_distinct_elements}"
        )
    if int(flank) < 0:
        raise LocusError(f"flank must be >= 0, got {flank}")
    min_distinct_elements = int(min_distinct_elements)
    flank = int(flank)

    zf = zero_flanked_array(zero_flanked, seq_len)
    cov = _coverage_array(coverage, seq_len)

    candidates = candidates_from_mask(
        log_probs,
        assignment >= 0,
        zero_flanked,
        min_span=min_span,
        gap_merge=gap_merge,
    )

    kept: list[tuple[Candidate, tuple[int, ...], int, int, int, int]] = []
    for cand in candidates:
        core = assignment[cand.start : cand.end]
        present = tuple(int(c) for c in np.unique(core) if int(c) != NOT_ELEMENT)
        if len(present) < min_distinct_elements:
            continue
        start = max(0, cand.start - flank)
        end = min(seq_len, cand.end + flank)
        kept.append(
            (
                cand,
                present,
                start,
                end,
                flank - (cand.start - start),
                flank - (end - cand.end),
            )
        )

    loci: list[Locus] = []
    for i, (cand, present, start, end, short_left, short_right) in enumerate(kept):
        overlaps = (i > 0 and start < kept[i - 1][3]) or (
            i + 1 < len(kept) and kept[i + 1][2] < end
        )
        loci.append(
            Locus(
                candidate=cand,
                start=int(start),
                end=int(end),
                length=int(end - start),
                flank=flank,
                flank_short_left=int(short_left),
                flank_short_right=int(short_right),
                element_classes=present,
                n_distinct_elements=len(present),
                n_zero_flanked_span=int(np.count_nonzero(zf[start:end])),
                n_single_covered_span=(
                    None if cov is None else int(np.count_nonzero(cov[start:end] == 1))
                ),
                overlaps_neighbour=bool(overlaps),
            )
        )
    return loci


def no_rule_parameter_has_a_default(func: Any = None) -> bool:
    """True iff every D3 locus-rule knob on :func:`construct_loci` is keyword-only, no default.

    ADR-0005 D3 freezes the locus-rule values at the §13.1 phase gate. A default here would be
    a de-facto frozen value that no ADR signed off and that a caller could take by accident —
    the ``min_sequences`` / PEFT-``target_modules`` lesson :mod:`tbox_finder.infer.call` cites
    for the same reason. Re-derived from the live signature rather than asserted in prose, so
    adding a default to any knob makes this return False.

    The array-like inputs (``log_probs``, ``zero_flanked``, ``coverage``) are data, not rule
    values, and are excluded: the latter two default to ``None`` meaning "not supplied".

    ``func`` exists so the predicate can be pointed at a stub that *does* carry a default and
    shown to return False — a check that only ever ran against the compliant signature would
    be satisfied by a predicate hardcoded to True.
    """
    params = inspect.signature(construct_loci if func is None else func).parameters
    keyword_only = {
        name: p for name, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    # Compared as a set, not a subset: a knob added later without a matching entry here would
    # otherwise be waved through unchecked, which is how a default slips in unnoticed.
    if set(keyword_only) != set(RULE_PARAMETERS):
        return False
    return all(p.default is inspect.Parameter.empty for p in keyword_only.values())


__all__ = [
    "ELEMENT_CLASS_NAMES",
    "NOT_ELEMENT",
    "RULE_PARAMETERS",
    "SCHEMA_VERSION",
    "STEP",
    "THRESHOLD_SCOPES",
    "CandidateError",
    "Locus",
    "LocusError",
    "ThresholdScope",
    "construct_loci",
    "element_assignment",
    "no_rule_parameter_has_a_default",
]
