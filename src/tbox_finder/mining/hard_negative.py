"""Collect Stage-1 false positives from the §9.1 pools as hard negatives (P2-07).

PRD §9.1's mined curriculum: sample windows at a moderate ~10:1 seed ratio, then
**iteratively mine Stage-1 false positives** — from the Rfam structured-RNA pool,
the 5′UTR/tRNA-adjacent leader pool, and the dinucleotide-shuffle survivors — as
hard negatives for the next round.

Two guards stand between a Stage-1 false positive and the mined pool, in order:

1. **Union-prior locus masking.** All known T-box loci (TBDB + RF00230-only hits +
   curated literature occurrences, plus the run's own training positives, plus a
   flank) are masked from every negative/decoy pool, so a known T-box is never
   mined as a negative. Delegated to :class:`tbox_finder.masking.LocusIndex`.
   A candidate **without coordinates cannot be masked** — it is refused rather
   than passed through, because silently unmaskable candidates would make the
   mask the no-op it measured as at P0 (``total_pool_records_masked = 0``).
2. **The spare rule**, three-valued (:mod:`tbox_finder.mining.spare_rule`), which
   protects a *non-canonical* (Tier-2N) locus that masking cannot know about.

The **5′UTR / tRNA-adjacent leader pool is retained** — ADR-0005 D14 keeps it
explicitly, as the hardest and most useful hard-negative context for a
tRNA-sensing finder. It is never excluded wholesale; only per-candidate evidence
may spare an individual member.

``pandas`` is imported lazily; the pure functions here are stdlib-only so the unit
tier runs in bare CI. PRD §9.1; ADR-0005 D14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tbox_finder.masking import DEFAULT_FLANK_NT, LocusIndex
from tbox_finder.mining.spare_rule import (
    SpareRuleEvidence,
    is_mining_excluded,
    mining_round_readiness,
    spare_reason,
)

#: Pools that PRD §9.1 mines Stage-1 false positives from.
MINEABLE_POOLS: tuple[str, ...] = ("structured_rna", "leader_decoy", "dinuc_shuffled")

#: The 5′UTR / tRNA-adjacent leader pool, retained by ADR-0005 D14. Named here so
#: a test can assert it is *in* :data:`MINEABLE_POOLS` — a regression that dropped
#: it would otherwise look like a harmless tightening.
RETAINED_LEADER_POOL = "leader_decoy"

#: Outcome vocabulary for a mining decision.
OUTCOME_MINED = "mined"
OUTCOME_MASKED = "masked_known_locus"
OUTCOME_SPARED = "spared_by_rule"
OUTCOME_UNMASKABLE = "refused_no_coordinates"


class HardNegativeMiningError(ValueError):
    """Raised when a round is run without a ready backend set, or on bad input."""


@dataclass(frozen=True)
class MiningCandidate:
    """A Stage-1 false positive proposed for the hard-negative pool."""

    candidate_id: str
    pool: str
    accession: str | None
    locus_start: int | None
    locus_end: int | None
    score: float
    evidence: SpareRuleEvidence = SpareRuleEvidence()

    def has_coordinates(self) -> bool:
        return (
            self.accession is not None
            and self.locus_start is not None
            and self.locus_end is not None
        )


def classify_candidate(
    candidate: MiningCandidate,
    mask: LocusIndex | None,
    *,
    stage2_threshold: float | None = None,
    flank: int = DEFAULT_FLANK_NT,
) -> tuple[str, str]:
    """Return ``(outcome, reason)`` for one candidate.

    Order matters: an uncoordinated candidate is refused **before** the spare rule
    runs, so a candidate that masking could not evaluate never reaches the pool on
    the strength of spare-rule evidence alone.

    ``flank`` defaults to :data:`tbox_finder.masking.DEFAULT_FLANK_NT`, matching the
    PRD §9.1 "known loci **+ a flank**" rule; passing ``0`` would silently shrink
    the mask to bare-overlap and let a locus edge through.
    """
    if not candidate.has_coordinates():
        return OUTCOME_UNMASKABLE, "candidate carries no accession/locus coordinates"
    if mask is not None and mask.is_masked(
        str(candidate.accession),
        int(candidate.locus_start),
        int(candidate.locus_end),
        flank=flank,
    ):
        return OUTCOME_MASKED, "overlaps a known T-box locus in the union prior (+flank)"
    if is_mining_excluded(candidate.evidence, stage2_threshold=stage2_threshold):
        return OUTCOME_SPARED, spare_reason(candidate.evidence, stage2_threshold=stage2_threshold)
    return OUTCOME_MINED, "no disjunct passed and every disjunct was evaluated"


def mine_round(
    candidates: list[MiningCandidate],
    mask: LocusIndex | None,
    backend_availability: dict[str, bool],
    *,
    stage2_threshold: float | None = None,
    flank: int = DEFAULT_FLANK_NT,
) -> dict[str, Any]:
    """Run one mining round, refusing outright if the backend set is not ready.

    The readiness gate is checked **first** and raises: a round with no protective
    disjunct available would spare every candidate and emit a report full of
    zeroes that reads exactly like "there was nothing to mine".
    """
    readiness = mining_round_readiness(backend_availability)
    if not readiness["ready"]:
        raise HardNegativeMiningError(f"mining round refused — {readiness['refusal_reason']}")

    outcomes: dict[str, list[str]] = {
        OUTCOME_MINED: [],
        OUTCOME_MASKED: [],
        OUTCOME_SPARED: [],
        OUTCOME_UNMASKABLE: [],
    }
    reasons: dict[str, str] = {}
    for candidate in candidates:
        outcome, reason = classify_candidate(
            candidate, mask, stage2_threshold=stage2_threshold, flank=flank
        )
        outcomes[outcome].append(candidate.candidate_id)
        reasons[candidate.candidate_id] = reason

    per_pool: dict[str, dict[str, int]] = {}
    for pool in sorted({c.pool for c in candidates}):
        members = [c for c in candidates if c.pool == pool]
        per_pool[pool] = {
            "n_candidates": len(members),
            "n_mined": sum(1 for c in members if c.candidate_id in set(outcomes[OUTCOME_MINED])),
        }

    return {
        "readiness": readiness,
        "n_candidates": len(candidates),
        "n_mined": len(outcomes[OUTCOME_MINED]),
        "n_masked": len(outcomes[OUTCOME_MASKED]),
        "n_spared": len(outcomes[OUTCOME_SPARED]),
        "n_refused_no_coordinates": len(outcomes[OUTCOME_UNMASKABLE]),
        "mined_ids": sorted(outcomes[OUTCOME_MINED]),
        "spared_ids": sorted(outcomes[OUTCOME_SPARED]),
        "reasons": reasons,
        "per_pool": per_pool,
        "leader_pool_retained": RETAINED_LEADER_POOL in MINEABLE_POOLS,
        "mineable_pools": list(MINEABLE_POOLS),
    }
