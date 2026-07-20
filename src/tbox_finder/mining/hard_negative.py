"""Collect Stage-1 false positives from the §9.1 pools as hard negatives (P2-07).

PRD §9.1's mined curriculum: sample windows at a moderate ~10:1 seed ratio, then
**iteratively mine Stage-1 false positives** as hard negatives for the next round.
The pools that can supply such a candidate are :data:`MINEABLE_POOLS` — mining
needs genomic coordinates the mask can evaluate, which is a stronger requirement
than being a §9.1 negative (see that constant for what it excludes and why).

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

from tbox_finder import masking
from tbox_finder.masking import DEFAULT_FLANK_NT, LocusIndex
from tbox_finder.mining.spare_rule import (
    MODEL_INDEPENDENT_DISJUNCTS,
    STATUS_UNAVAILABLE,
    SpareRuleEvidence,
    is_mining_excluded,
    mining_round_readiness,
    spare_reason,
)

#: Pools that PRD §9.1 mines Stage-1 false positives from.
#:
#: ``dinuc_shuffled`` was removed at P2-10b (ADR-0005 **A6**). Mining a candidate
#: requires coordinates that are *independent* of the training positives, and a
#: dinucleotide shuffle's only possible coordinates are the positive it permutes:
#: carrying them masks 100 % of the pool against its own parents, which turns the
#: mask-non-vacuity measurement into a tautology, while leaving them null makes
#: ``refused_no_coordinates`` — the signal that something is *wrong* — the pool's
#: permanent steady state. It remains a §9.1 training negative; only its
#: *mineability* is withdrawn, and its parent link is now recorded explicitly as
#: ``source_record_id``.
#:
#: ``genomic_window`` (:mod:`tbox_finder.mining.pool`) replaces it as the
#: coordinate-bearing substrate. ``gc_background`` is absent for the same reason
#: it always was: it is emitted i.i.d. at a matched GC and exists in no genome.
MINEABLE_POOLS: tuple[str, ...] = ("structured_rna", "leader_decoy", "genomic_window")

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

    def __post_init__(self) -> None:
        # PRD §9.1 names the pools that may be mined. An unrecognised pool would
        # otherwise flow through classification and appear in ``per_pool`` as if
        # it were sanctioned — including pools deliberately excluded from mining.
        if self.pool not in MINEABLE_POOLS:
            raise HardNegativeMiningError(
                f"unknown mining pool {self.pool!r}; expected one of {MINEABLE_POOLS}"
            )

    def has_coordinates(self) -> bool:
        """True iff all three coordinate fields are present **and not NaN/NA**.

        The ``is not None`` test this replaced was fail-open: a pandas round-trip
        through a nullable column yields ``NaN``/``pd.NA``, which passes an
        identity check, reaches :meth:`~tbox_finder.masking.LocusIndex.is_masked`,
        matches nothing (that method treats a missing accession as unmaskable),
        and so classifies the candidate ``mined`` rather than
        ``refused_no_coordinates``. Sharing the mask's own missing test is what
        keeps the guard and the mask from disagreeing (P2-10b).
        """
        return not (
            masking.is_missing(self.accession)
            or masking.is_missing(self.locus_start)
            or masking.is_missing(self.locus_end)
        )


def classify_candidate(
    candidate: MiningCandidate,
    mask: LocusIndex,
    *,
    stage2_threshold: float | None = None,
    flank: int = DEFAULT_FLANK_NT,
) -> tuple[str, str]:
    """Return ``(outcome, reason)`` for one candidate.

    Order matters: an uncoordinated candidate is refused **before** the spare rule
    runs, so a candidate that masking could not evaluate never reaches the pool on
    the strength of spare-rule evidence alone.

    The mask is **required**, and ``flank`` must be positive. Both were previously
    optional, which made the two ways of disabling masking indistinguishable from
    running it: ``mask=None`` skipped the union-prior check entirely and ``flank=0``
    shrank it to bare overlap, each letting a known T-box locus into the negative
    pool while the round reported a clean pass. This is the P0 failure the mask was
    written for (``total_pool_records_masked = 0``), so it is now unrepresentable —
    a caller that genuinely wants no masking passes an empty
    :class:`~tbox_finder.masking.LocusIndex` and says so.
    """
    if mask is None:
        raise HardNegativeMiningError(
            "a union-prior LocusIndex is required; pass an empty LocusIndex to mask nothing"
        )
    if flank <= 0:
        raise HardNegativeMiningError(
            f"flank must be positive (PRD §9.1 masks known loci + a flank); got {flank}"
        )
    if not candidate.has_coordinates():
        return OUTCOME_UNMASKABLE, "candidate carries no accession/locus coordinates"
    if mask.is_masked(
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
    mask: LocusIndex,
    backend_availability: dict[str, bool],
    *,
    stage2_threshold: float | None = None,
    flank: int = DEFAULT_FLANK_NT,
) -> dict[str, Any]:
    """Run one mining round, refusing outright if the backend set is not ready.

    The readiness gate is checked **first** and raises: a round with no protective
    disjunct available would spare every candidate and emit a report full of
    zeroes that reads exactly like "there was nothing to mine".

    Candidate evidence is then cross-checked against the round's declared backend
    availability. A candidate claiming a ``passed``/``failed`` verdict for a
    disjunct whose backend is unavailable this round is a contradiction, and the
    ``failed`` direction is actively dangerous — it is precisely what turns the
    fail-closed rule back into a fail-open one, since three ``failed`` verdicts
    make a candidate minable no matter what ran. Raising here keeps the round's
    own availability declaration and its per-candidate evidence from disagreeing
    silently.
    """
    readiness = mining_round_readiness(backend_availability)
    if not readiness["ready"]:
        raise HardNegativeMiningError(f"mining round refused — {readiness['refusal_reason']}")

    for candidate in candidates:
        for disjunct in MODEL_INDEPENDENT_DISJUNCTS:
            if backend_availability[disjunct]:
                continue
            status = candidate.evidence.status_of(disjunct)
            if status != STATUS_UNAVAILABLE:
                raise HardNegativeMiningError(
                    f"candidate {candidate.candidate_id!r} carries {status!r} evidence for "
                    f"{disjunct!r}, but that backend is unavailable this round"
                )

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
