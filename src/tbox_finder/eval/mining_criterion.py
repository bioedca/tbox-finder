"""P2-10e — the **provisional** mining-internal Stage-1 detection criterion.

Why this module exists (ADR-0005 Amendment A9, Pin 3)
-----------------------------------------------------
The P2-10e hard-negative-mining rounds are gated each round by the Tier-2N probe's
per-round recall (``eval/tier2n_probe.py``). :func:`~tbox_finder.eval.tier2n_probe.probe_recall`
needs one input the harness does not produce on its own — ``recovered_ids``, the set
of probe members the current Stage-1 checkpoint still *detects*. Producing it requires
a **Stage-1 detection criterion**: a posterior threshold plus along-sequence
locus-construction (minimum span, gap-merge). :func:`tbox_finder.infer.scan.scan_sequence`
stops at the reconciled per-position distribution and names that step "a later step";
:func:`tbox_finder.infer.call.call_candidates` is that step — but it deliberately pins
**no value** (its ``threshold`` / ``min_span`` / ``gap_merge`` are keyword-only with no
defaults), because ADR-0005 **D3** freezes the *production* Stage-1 threshold + locus
geometry only at the §13.1 phase gate, which is genuinely **downstream** of these mining
rounds. So the gate P2-10e needs to measure cannot use a D3-frozen value, because D3 is
not yet frozen.

Amendment A9 Pin 3 resolves this circular ordering with a **provisional** criterion:

* a threshold + min-span + gap-merge **pinned here** and used **only** to compute the
  per-round Tier-2N recall (and to collect the model's false positives on the negative
  substrate — the same detection operator, applied to a different sequence set);
* held **identical across all N mining rounds** (cross-round-constant), so its absolute
  value **cancels** in ``round_decision``'s best-round-so-far *relative* comparison — the
  halt measures the model's change between rounds, not the criterion's value;
* declared **explicitly non-binding** on D3's §13.1 phase-gate freeze and the P5
  production operating point. This is **not** the D3 value and does not become it.

Reuse, not fork (``infer/call.py``)
-----------------------------------
The locus-construction operator itself is **not re-implemented here**. This module reuses
:func:`tbox_finder.infer.call.call_candidates` verbatim and only supplies the pinned
provisional triple. Forking the operator would mean fixing one copy and shipping the bug
in the other (the promote-don't-duplicate rule); "recovered" is defined as exactly
"``call_candidates`` returns ≥ 1 candidate under the provisional criterion".

The degenerate-criterion guard
------------------------------
A cross-round-constant value cancels **only if it is not degenerate**. Two degeneracies
would silently void the gate:

* recall ≈ **1.0** — the criterion recovers *every* probe member regardless of the model,
  so recall is pinned at 1.0 every round and the drop is always 0: the halt can never
  fire and the gate is vacuous;
* recall ≈ **0** — the criterion recovers *no* probe member, so the model's ability to
  detect the flagship class is unmeasurable and any later "recovery" is noise.

So before the first round's decision is trusted, :func:`guard_non_degenerate` asserts the
provisional criterion resolves the probe set at a **strictly intermediate** recall — some
but not all probe members recovered on the round-0 checkpoint (an identity check,
``0 < |members ∩ recovered| < |members|``, not a bare count). A degenerate round-0 recall
is a CLAUDE.md §7 stop: adjust the provisional criterion (still held constant across
rounds) rather than trust a vacuous gate.

``numpy``-only and torch-free (it consumes the reconciled arrays, not the model), so it
imports and unit-tests on the bare-CI path exactly like ``infer/call.py`` and
``infer/reconcile.py``. PRD §9.1, §13.1; ADR-0005 D3 + A9; ADR-0006 D9.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tbox_finder.infer.call import call_candidates

SCHEMA_VERSION = "1.0"
STEP = "P2-10e"

# --------------------------------------------------------------------------- #
# The provisional detection criterion (ADR-0005 A9 Pin 3) — cross-round-constant,
# non-binding, NOT the ADR-0005 D3 phase-gate value.
# --------------------------------------------------------------------------- #
#: Posterior threshold on the element score ``1 − P(background)``. A central point of
#: ``infer/call.py``'s provisional ρ-pilot grid (``PROVISIONAL_THRESHOLD_GRID``), chosen
#: to be discriminating rather than saturating. Its *value* is non-load-bearing: it is
#: held identical across every mining round, so it cancels in ``round_decision``'s
#: relative comparison, and :func:`guard_non_degenerate` refuses the first round unless it
#: resolves the probe set at a non-degenerate recall.
PROVISIONAL_THRESHOLD: float = 0.9

#: Minimum merged element-run length (nt) for a called locus. A T-box leader is
#: ~150–350 nt, so 50 nt is comfortably below a real locus yet excludes single-position
#: flickers. Provisional and cross-round-constant (see :data:`PROVISIONAL_THRESHOLD`).
PROVISIONAL_MIN_SPAN: int = 50

#: Maximum background gap (nt) bridged between element runs of one called locus.
#: Provisional and cross-round-constant (see :data:`PROVISIONAL_THRESHOLD`).
PROVISIONAL_GAP_MERGE: int = 10


class ProvisionalCriterionError(ValueError):
    """Raised when the provisional criterion is degenerate on the round-0 probe set."""


def sequence_is_recovered(log_probs: Any, zero_flanked: Any = None) -> bool:
    """Whether the provisional criterion calls at least one locus on a scanned sequence.

    ``log_probs`` is the reconciled ``(seq_len, NUM_CLASSES)`` log-posterior
    (:class:`tbox_finder.infer.reconcile.Reconciled.log_probs`); ``zero_flanked`` its
    matching ``(seq_len,)`` contig-end mask (``None`` ⇒ all-interior). "Recovered" is
    exactly one called candidate under the pinned provisional triple — the operator is
    :func:`tbox_finder.infer.call.call_candidates`, reused, never re-derived.
    """
    candidates = call_candidates(
        log_probs,
        zero_flanked,
        threshold=PROVISIONAL_THRESHOLD,
        min_span=PROVISIONAL_MIN_SPAN,
        gap_merge=PROVISIONAL_GAP_MERGE,
    )
    return len(candidates) >= 1


def recovered_ids(scored: Mapping[str, tuple[Any, Any]]) -> set[str]:
    """Set of ids the provisional criterion recovers, from ``{id: (log_probs, zero_flanked)}``.

    The value returned drops straight into
    :func:`tbox_finder.eval.tier2n_probe.probe_recall` as its ``recovered_ids`` argument;
    the keys must therefore be the probe set's own id space (``Tier2NVariant.variant_id``
    for the synthetic arm). An id maps to ``(log_probs, zero_flanked)`` — the reconciled
    posteriors of that member's sequence under the current checkpoint. The scan itself
    (checkpoint → posteriors) is the caller's torch-side job; this stays torch-free so the
    recovery rule is unit-testable on committed numpy fixtures.
    """
    out: set[str] = set()
    for seq_id, payload in scored.items():
        log_probs, zero_flanked = payload
        if sequence_is_recovered(log_probs, zero_flanked):
            out.add(str(seq_id))
    return out


def assess_recovery(members: set[str], recovered: set[str]) -> dict[str, Any]:
    """Assess whether the provisional criterion is degenerate on a probe member set.

    ``members`` are the probe-set ids (``set(probe_set.natural) | set(probe_set.synthetic)``);
    ``recovered`` the ids the criterion recovered this round. Degeneracy is decided on the
    **identity** of the recovered probe members — ``n_recovered = |members ∩ recovered|``,
    not the size of ``recovered`` (which may contain ids outside the probe set) — so a
    criterion that recovers the right *count* of the *wrong* members is still caught.

    Non-degenerate ⇔ ``0 < n_recovered < n_probe`` (some but not all recovered). Both
    endpoints are degenerate: ``n_recovered == n_probe`` (recall pinned at 1.0 ⇒ the halt
    can never fire) and ``n_recovered == 0`` (the model's flagship-class detection is
    unmeasurable). Raises on an empty member set — a recall of ``0/0`` is not a measurement.
    """
    n_probe = len(members)
    if n_probe == 0:
        raise ProvisionalCriterionError(
            "cannot assess the provisional criterion on an empty probe member set"
        )
    hit = members & recovered
    n_recovered = len(hit)
    recall = n_recovered / n_probe

    if n_recovered == 0:
        degenerate, reason = True, (
            "recall is 0: the provisional criterion recovers no probe member on this "
            "checkpoint, so the model's flagship-class detection is unmeasurable and any "
            "later recovery would be noise"
        )
    elif n_recovered == n_probe:
        degenerate, reason = True, (
            "recall is 1.0: the provisional criterion recovers every probe member "
            "regardless of the model, so the per-round drop is pinned at 0 and the "
            "Tier-2N halt can never fire — the gate would be vacuous"
        )
    else:
        degenerate, reason = False, None

    return {
        "n_probe": n_probe,
        "n_recovered_in_probe": n_recovered,
        "recall": recall,
        "degenerate": degenerate,
        "degenerate_reason": reason,
        "threshold": PROVISIONAL_THRESHOLD,
        "min_span": PROVISIONAL_MIN_SPAN,
        "gap_merge": PROVISIONAL_GAP_MERGE,
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
    }


def guard_non_degenerate(members: set[str], recovered: set[str]) -> dict[str, Any]:
    """Raise unless the provisional criterion is non-degenerate on the round-0 probe set.

    The driver calls this **before** trusting the first round's decision. On a degenerate
    round-0 recall it raises :class:`ProvisionalCriterionError` (a §7 stop — adjust the
    provisional criterion, held constant across rounds, rather than run a vacuous gate);
    otherwise it returns the :func:`assess_recovery` report.
    """
    assessment = assess_recovery(members, recovered)
    if assessment["degenerate"]:
        raise ProvisionalCriterionError(
            "provisional detection criterion is degenerate on the round-0 probe set "
            f"({assessment['degenerate_reason']}); "
            f"recovered {assessment['n_recovered_in_probe']}/{assessment['n_probe']} at "
            f"threshold={PROVISIONAL_THRESHOLD}, min_span={PROVISIONAL_MIN_SPAN}, "
            f"gap_merge={PROVISIONAL_GAP_MERGE} — adjust the (cross-round-constant) "
            "provisional criterion, do not trust the relative halt on top of it"
        )
    return assessment


__all__ = [
    "PROVISIONAL_GAP_MERGE",
    "PROVISIONAL_MIN_SPAN",
    "PROVISIONAL_THRESHOLD",
    "SCHEMA_VERSION",
    "STEP",
    "ProvisionalCriterionError",
    "assess_recovery",
    "guard_non_degenerate",
    "recovered_ids",
    "sequence_is_recovered",
]
