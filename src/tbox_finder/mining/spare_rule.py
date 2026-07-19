"""The mining spare-rule gate under **incomplete disjunct evidence** (P2-07).

ADR-0005 D14 (mirrored by ADR-0006 D11 and PRD §9.1) pins the hard-negative
mining-exclusion rule: a candidate is **excluded from the mining pool** if it
passes **relaxed-architecture detection (b) OR any-helix R-scape covariation (a)
OR downstream-aaRS synteny (c)** — the three *model-independent* disjuncts,
sufficient for the P2 Stage-1 mining loop where no Stage-2 exists yet — **or**, at
the P3 re-mining round, a high Stage-2 posterior. The rule is deliberately *not*
keyed to canonical architecture so that a CM-missed **Tier-2N** locus, which fails
every canonical predicate, is still protected from being mined away.

Why this module exists on top of :func:`tbox_finder.masking.spare_rule_excludes_from_mining`
------------------------------------------------------------------------------------------
The pinned predicate takes three **booleans defaulting to ``False``**. That signature
silently conflates *"the backend ran and the candidate failed this disjunct"* with
*"no backend exists, so nothing was ever evaluated"*. At P2-07 the second case is
**every case**: none of the three disjuncts has an executable backend in-repo yet
(no ``cmsearch``-driven relaxed-architecture caller, no R-scape — absent from every
``envs/*.yml`` and scheduled for P6-01 — and no neighbouring-CDS coordinates or
strand from which ADR-0006 D4's "first same-strand downstream CDS within 500 bp"
could be computed). Called with its defaults, the pinned predicate therefore
excludes **nothing**, and the guard that exists to protect the flagship class reads
green while protecting zero candidates.

This module closes that hole in two places and **delegates the decision itself**
to the pinned predicate — it never re-implements the OR (one rule, one
implementation; a forked copy means fixing one and shipping the bug in the other).

1. **Per candidate — three-valued evidence, unknown resolved conservatively.**
   Each disjunct carries ``passed`` / ``failed`` / ``unavailable``. The decision is
   Kleene strong three-valued OR with ``unknown`` resolved *toward sparing*:

   ==========================  ==================================
   evidence                    verdict
   ==========================  ==================================
   any disjunct ``passed``     **excluded** (decisive; the OR is satisfied)
   else any ``unavailable``    **excluded** (fail-closed: an unrun backend
                               cannot be assumed to have failed)
   else (all ``failed``)       minable
   ==========================  ==================================

2. **Per round — a readiness gate on which backends exist at all.**
   Availability is not fungible across the three disjuncts. ADR-0006 D9 row 5
   defines **Tier-2N** as ``(c)✓ ∧ (a)✓ ∧ (b)✗`` — architecture detection *fails*
   on a Tier-2N locus **by definition**. Therefore the relaxed-architecture
   disjunct (b) can **never** spare a Tier-2N candidate, and a mining round whose
   only available backend is (b) has **zero** Tier-2N protection while appearing
   fully instrumented. :func:`mining_round_readiness` requires at least one of the
   two *protective* disjuncts — (a) covariation or (c) synteny — to be available
   before a round may run, and refuses the round otherwise rather than letting it
   degrade into a no-op that still reports success.

Pure stdlib: no I/O, no third-party imports. PRD §9.1; ADR-0005 D14; ADR-0006
D9/D11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tbox_finder.masking import spare_rule_excludes_from_mining

# --------------------------------------------------------------------------- #
# Disjunct evidence vocabulary (frozen; no CLI/config override)
# --------------------------------------------------------------------------- #
#: The backend ran and the candidate **satisfies** this disjunct.
STATUS_PASSED = "passed"
#: The backend ran and the candidate **does not** satisfy this disjunct.
STATUS_FAILED = "failed"
#: No backend was available, so this disjunct was **never evaluated**. Distinct
#: from :data:`STATUS_FAILED` — conflating the two is the failure mode this
#: module exists to prevent.
STATUS_UNAVAILABLE = "unavailable"

#: The complete, closed status vocabulary. Validated on construction so a typo
#: cannot silently read as "not passed".
DISJUNCT_STATUSES: tuple[str, ...] = (STATUS_PASSED, STATUS_FAILED, STATUS_UNAVAILABLE)

#: The three **model-independent** disjuncts of the P2 spare rule, in the
#: ADR-0006 D9 lettering: (b) relaxed architecture, (a) covariation, (c) synteny.
MODEL_INDEPENDENT_DISJUNCTS: tuple[str, ...] = (
    "relaxed_architecture",
    "any_helix_rscape",
    "downstream_aaRS_synteny",
)

#: The subset of disjuncts that can actually spare a **Tier-2N** candidate.
#:
#: ADR-0006 D9 row 5 routes Tier-2N as ``(c)✓ ∧ (a)✓ ∧ (b)✗``: the architecture
#: disjunct is FALSE on every Tier-2N locus by definition, so it contributes
#: nothing to the protection this rule exists to provide. Only (a) and (c) do.
#: Frozen here — no CLI/config override — so no committed round report can claim
#: Tier-2N protection from an architecture-only backend set.
TIER2N_PROTECTIVE_DISJUNCTS: tuple[str, ...] = (
    "any_helix_rscape",
    "downstream_aaRS_synteny",
)

#: Minimum number of :data:`TIER2N_PROTECTIVE_DISJUNCTS` that must be available
#: before a mining round may run. One is enough to give the rule a non-vacuous
#: protective path; zero makes the round a silent no-op.
MIN_PROTECTIVE_DISJUNCTS_AVAILABLE = 1


class SpareRuleEvidenceError(ValueError):
    """Raised when disjunct evidence is malformed (unknown status, bad posterior)."""


@dataclass(frozen=True)
class SpareRuleEvidence:
    """Per-candidate evidence for the three model-independent disjuncts.

    Each field is one of :data:`DISJUNCT_STATUSES`. The default is
    :data:`STATUS_UNAVAILABLE`, **not** ``failed`` — an unconfigured evidence
    record must read as "nothing was evaluated" (⇒ spared), never as "every
    disjunct was checked and all three failed" (⇒ minable).

    ``stage2_posterior`` is inert at P2 (no Stage-2 exists); it excludes only when
    a ``stage2_threshold`` is supplied at the P3 re-mining round (ADR-0005 D14).
    """

    relaxed_architecture: str = STATUS_UNAVAILABLE
    any_helix_rscape: str = STATUS_UNAVAILABLE
    downstream_aaRS_synteny: str = STATUS_UNAVAILABLE
    stage2_posterior: float | None = None

    def __post_init__(self) -> None:
        for name in MODEL_INDEPENDENT_DISJUNCTS:
            value = getattr(self, name)
            if value not in DISJUNCT_STATUSES:
                raise SpareRuleEvidenceError(
                    f"{name} must be one of {DISJUNCT_STATUSES}, got {value!r}"
                )
        post = self.stage2_posterior
        if post is None:
            return
        if isinstance(post, bool) or not isinstance(post, (int, float)):
            raise SpareRuleEvidenceError(
                f"stage2_posterior must be a real number or None, got {type(post).__name__}"
            )
        if not 0.0 <= float(post) <= 1.0:
            raise SpareRuleEvidenceError(f"stage2_posterior must be in [0, 1], got {post}")

    def status_of(self, disjunct: str) -> str:
        """Status for one of :data:`MODEL_INDEPENDENT_DISJUNCTS`."""
        if disjunct not in MODEL_INDEPENDENT_DISJUNCTS:
            raise SpareRuleEvidenceError(
                f"unknown disjunct {disjunct!r}; expected one of {MODEL_INDEPENDENT_DISJUNCTS}"
            )
        return str(getattr(self, disjunct))

    def passed(self) -> tuple[str, ...]:
        """Disjuncts whose backend ran and which the candidate satisfies."""
        return tuple(d for d in MODEL_INDEPENDENT_DISJUNCTS if self.status_of(d) == STATUS_PASSED)

    def unavailable(self) -> tuple[str, ...]:
        """Disjuncts that were never evaluated because no backend existed."""
        return tuple(
            d for d in MODEL_INDEPENDENT_DISJUNCTS if self.status_of(d) == STATUS_UNAVAILABLE
        )


def is_mining_excluded(
    evidence: SpareRuleEvidence,
    *,
    stage2_threshold: float | None = None,
) -> bool:
    """Return ``True`` iff the candidate must be **kept out of** the mining pool.

    Implements the Kleene three-valued resolution documented in the module
    docstring. The two-valued decision is **delegated** to the pinned predicate
    :func:`tbox_finder.masking.spare_rule_excludes_from_mining` — this function
    supplies it with ``passed → True`` / ``failed → False`` and adds only the
    ``unavailable`` arm, so there is exactly one implementation of the OR itself.

    A candidate is spared when any disjunct passed (the OR is satisfied) **or**
    when the decision is undetermined because some backend never ran.
    """
    decided = spare_rule_excludes_from_mining(
        relaxed_architecture=evidence.status_of("relaxed_architecture") == STATUS_PASSED,
        any_helix_rscape=evidence.status_of("any_helix_rscape") == STATUS_PASSED,
        downstream_aaRS_synteny=evidence.status_of("downstream_aaRS_synteny") == STATUS_PASSED,
        stage2_posterior=evidence.stage2_posterior,
        stage2_threshold=stage2_threshold,
    )
    if decided:
        return True
    # No disjunct passed. Only "every disjunct actually ran and failed" licenses
    # mining; an unrun backend leaves the OR undetermined → fail closed.
    return bool(evidence.unavailable())


def spare_reason(
    evidence: SpareRuleEvidence,
    *,
    stage2_threshold: float | None = None,
) -> str:
    """Attribution string for why a candidate was spared, or ``"minable"``.

    Round reports gate on :func:`is_mining_excluded`; this exists so a spared
    candidate's *cause* is auditable — in particular so a round spared entirely by
    ``unavailable_backend`` cannot be mistaken for one spared by real evidence.
    """
    passed = evidence.passed()
    if passed:
        return "passed:" + ",".join(passed)
    if (
        evidence.stage2_posterior is not None
        and stage2_threshold is not None
        and float(evidence.stage2_posterior) >= float(stage2_threshold)
    ):
        return "passed:stage2_posterior"
    missing = evidence.unavailable()
    if missing:
        return "unavailable_backend:" + ",".join(missing)
    return "minable"


def mining_round_readiness(available: dict[str, bool]) -> dict[str, Any]:
    """Decide whether a mining round may run, given which backends exist.

    ``available`` maps each of :data:`MODEL_INDEPENDENT_DISJUNCTS` to whether its
    backend can be executed this round. Every disjunct must be named explicitly —
    a missing key raises rather than defaulting, so a forgotten backend cannot be
    silently read as unavailable *or* as available.

    The round is ready iff at least :data:`MIN_PROTECTIVE_DISJUNCTS_AVAILABLE` of
    :data:`TIER2N_PROTECTIVE_DISJUNCTS` are available. Availability of the
    relaxed-architecture disjunct alone is explicitly **not** sufficient: it is
    ``False`` on every Tier-2N locus by ADR-0006 D9 row 5, so it cannot spare the
    class the rule protects.
    """
    missing_keys = [d for d in MODEL_INDEPENDENT_DISJUNCTS if d not in available]
    if missing_keys:
        raise SpareRuleEvidenceError(
            f"availability must name every disjunct; missing {missing_keys}"
        )
    for name, value in available.items():
        if not isinstance(value, bool):
            raise SpareRuleEvidenceError(
                f"availability[{name!r}] must be bool, got {type(value).__name__}"
            )

    available_disjuncts = tuple(d for d in MODEL_INDEPENDENT_DISJUNCTS if available[d])
    protective_available = tuple(d for d in TIER2N_PROTECTIVE_DISJUNCTS if available[d])
    n_protective = len(protective_available)
    ready = n_protective >= MIN_PROTECTIVE_DISJUNCTS_AVAILABLE

    if ready:
        refusal = None
    elif available_disjuncts:
        refusal = (
            "only non-protective disjuncts are available "
            f"({list(available_disjuncts)}); relaxed_architecture is False on every "
            "Tier-2N locus (ADR-0006 D9 row 5), so this round would mine with zero "
            "Tier-2N protection"
        )
    else:
        refusal = (
            "no model-independent disjunct backend is available; " "every candidate would be spared"
        )

    return {
        "ready": ready,
        "available_disjuncts": list(available_disjuncts),
        "protective_disjuncts_available": list(protective_available),
        "n_protective_available": n_protective,
        "min_protective_required": MIN_PROTECTIVE_DISJUNCTS_AVAILABLE,
        "tier2n_protective_disjuncts": list(TIER2N_PROTECTIVE_DISJUNCTS),
        "refusal_reason": refusal,
    }
