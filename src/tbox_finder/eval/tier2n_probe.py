"""The Tier-2N probe set + per-round recall halt/rollback rule (P2-07).

ADR-0005 D14: *"A **Tier-2N probe set** (non-canonical + synthetic-Tier-2N
positives) is evaluated **each mining round**, and a per-round **recall drop on it
halts/rolls back** the iteration"* — so aggressive hard-negative mining cannot
directionally train the production scanner to reject the flagship class. The worst
case is then a *directionally-bounded* Tier-2N sensitivity, not an invalid
generalization claim.

Probe-set composition, and why the natural arm is empty
-------------------------------------------------------
The probe set is the union of two arms:

* the **natural** arm — real non-canonical (Tier-2N) positives; and
* the **synthetic** arm — :mod:`tbox_finder.synth.tier2n` output.

The natural arm is **empty at P2-07, by construction rather than by oversight**.
The corpus is 100 % TBDB/CM-derived, so it cannot contain a CM-invisible locus;
neither the corpus nor the committed split table carries a tier column to select
on; and the literature documents no genuinely CM-invisible T-box architecture —
by definition, since that is the class this project exists to discover. The arm is
therefore reported at N = 0 and **disclosed**, in the same "reported-not-gated"
spirit as ADR-0005 D6/D9, never quietly dropped from the accounting.

That places the whole min-N burden on the synthetic arm, which is why
:mod:`tbox_finder.synth.tier2n` makes probe eligibility a **measured discordant
pair** (parent CM-detected, variant CM-missed) rather than a count the generator
chooses. Without that, "probe-set size ≥ min-N" would gate a knob, not evidence.

Halt / rollback
---------------
Recall is measured on the probe set each round and compared against the
**best round so far**, not merely the previous round — otherwise a slow monotone
bleed of a few points per round never trips a
previous-round comparison while still destroying the class over an iteration.

Pure stdlib. PRD §9.1, §12; ADR-0005 D14; ADR-0006 D9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.synth.tier2n import Tier2NVariant

#: The probe-set floor, imported from the ADR-0005 Amendment A1 pin.
TIER2N_PROBE_MIN_N = MIN_REAL_HOMOLOG_N

#: Absolute per-round recall drop, relative to the best round so far, that halts
#: the mining iteration and rolls back to the best checkpoint.
#:
#: Set to one probe-positive's worth of recall at the pinned floor
#: (``1 / MIN_REAL_HOMOLOG_N`` = 5 pp), matching the same ``1/N`` granularity
#: argument that pinned ``MIN_REAL_HOMOLOG_N`` and the ADR-0005 D4 +5 pp CI floor
#: — the smallest drop the probe set can actually resolve. A tighter bound would
#: be unmeasurable noise; a looser one would let a real regression pass.
#: Frozen: no CLI/config override.
TIER2N_RECALL_DROP_HALT = 1.0 / MIN_REAL_HOMOLOG_N

ROUND_CONTINUE = "continue"
ROUND_HALT_ROLLBACK = "halt_rollback"
ROUND_INADMISSIBLE = "inadmissible"


class Tier2NProbeError(ValueError):
    """Raised on a malformed probe set or round history."""


@dataclass(frozen=True)
class ProbeSet:
    """The evaluated Tier-2N probe set for a mining round."""

    natural: tuple[str, ...]
    synthetic: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.natural) + len(self.synthetic)

    def meets_min_n(self) -> bool:
        """Whether the probe set clears the ADR-0005 A1 floor.

        Guarded on non-emptiness so the clause cannot read true off an absent set.
        """
        return self.size > 0 and self.size >= TIER2N_PROBE_MIN_N


def build_probe_set(
    variants: list[Tier2NVariant],
    natural_ids: tuple[str, ...] = (),
) -> ProbeSet:
    """Assemble the probe set from the synthetic generator output + a natural arm.

    Only **probe-eligible** variants (measured discordant pairs) enter the
    synthetic arm; an emitted-but-unmeasured variant is excluded, so an unrun
    ``cmsearch`` shrinks the probe set toward failing min-N rather than inflating
    it toward passing.
    """
    synthetic = tuple(sorted(v.variant_id for v in variants if v.is_probe_eligible()))
    return ProbeSet(natural=tuple(sorted(natural_ids)), synthetic=synthetic)


def probe_recall(probe_set: ProbeSet, recovered_ids: set[str]) -> float:
    """Fraction of the probe set the scanner still recovers this round.

    Raises on an empty probe set: a recall of ``0/0`` would otherwise be reported
    as a number (or a vacuous 1.0) for a measurement that never happened.
    """
    if probe_set.size == 0:
        raise Tier2NProbeError("cannot compute recall on an empty probe set")
    members = set(probe_set.natural) | set(probe_set.synthetic)
    return len(members & set(recovered_ids)) / len(members)


def round_decision(
    probe_set: ProbeSet,
    recall_this_round: float,
    recall_history: list[float],
) -> dict[str, Any]:
    """Decide whether the mining iteration continues, or halts and rolls back.

    The comparison baseline is the **best** recall observed so far, not the
    previous round. Returns a report whose clauses are re-derived from the
    arguments rather than accumulated by the caller.
    """
    if not 0.0 <= recall_this_round <= 1.0:
        raise Tier2NProbeError(f"recall must be in [0, 1], got {recall_this_round}")
    for value in recall_history:
        if not 0.0 <= value <= 1.0:
            raise Tier2NProbeError(f"recall history contains {value}, outside [0, 1]")

    admissible = probe_set.meets_min_n()
    best_prior = max(recall_history) if recall_history else None
    drop = None if best_prior is None else best_prior - recall_this_round
    breached = bool(drop is not None and drop >= TIER2N_RECALL_DROP_HALT)

    if not admissible:
        decision = ROUND_INADMISSIBLE
    elif breached:
        decision = ROUND_HALT_ROLLBACK
    else:
        decision = ROUND_CONTINUE

    return {
        "decision": decision,
        "probe_set_size": probe_set.size,
        "n_natural": len(probe_set.natural),
        "n_synthetic": len(probe_set.synthetic),
        "tier2n_probe_min_n": TIER2N_PROBE_MIN_N,
        "probe_set_meets_min_n": admissible,
        "recall_this_round": recall_this_round,
        "best_prior_recall": best_prior,
        "recall_drop_vs_best": drop,
        "halt_threshold": TIER2N_RECALL_DROP_HALT,
        "halt_threshold_breached": breached,
        "baseline_rule": "best round so far (not previous round) — catches slow monotone bleed",
        "natural_arm_disclosure": (
            "the natural Tier-2N arm is empty by construction: the corpus is "
            "100% CM-derived and so cannot contain a CM-invisible locus; reported "
            "at N=0 rather than dropped from the accounting"
        ),
    }
