"""P3-15′-k — publish the P3 re-mining round's decision, with its own limits re-derived.

:mod:`tbox_finder.mining.remine` decides the round; this module **publishes** it. The
two are separate because the round report answers *"what did the spare rule do?"* and
the publication has to answer three further questions that a reader would otherwise
have to infer from silence — and each of the three has a zero or a count that reads as
something else when stated bare:

**1. "938 spared" reproduces the ambiguity ADR-0005 A10 flagged.** A10's Phase-2 §7
note recorded a round where ``mining_round_readiness`` said *ready* and the yield was
structurally 0 — 941/941 spared because no backend had run. A P3 round that spares 938
looks identical in that summary field. The two states are only distinguishable in the
per-candidate ``reasons`` map, which separates *a disjunct actually passed* from
*the backend was unavailable*, so :func:`spared_split` re-derives the split from that
map rather than restating ``n_spared``.

**2. ``n_excluded_probe_members: 0`` has three causes and one number.** The probe set
was empty (refused upstream by ``remine.load_probe_set``); the substrate and the probe
ids do not share a namespace, so no member *could* be present; or the exclusion ran
against overlapping namespaces and genuinely removed nothing. Only the third is a
measurement of the round. :func:`probe_exclusion_diagnosis` measures which one holds
instead of asserting it, and refuses the degenerate inputs that would make the zero
uninterpretable.

**3. The round decided and did not retrain.** The §7 decision of 2026-08-12 was
*decision-only, no retrain* — so no checkpoint superseded the P2 production one and
PRD §18.1's *"``stage1_production.ckpt`` supersedes the P2 checkpoint"* stays UNMET.
A round report that carries a mining outcome and nothing about legs (2)/(3) reads as
though the arc completed. :func:`retrain_disposition` states it, and
:func:`publication_problems` refuses a publication whose round report carries a
``parent_checkpoint`` — the field a retraining round would fill.

Two further things the publication carries because the round report cannot:

* **The Stage-2 threshold sensitivity.** ADR-0005 D14 pins the *rule*; the §13.1 phase
  gate pins the *operating point*, so this round runs at an **unpinned** value and
  ``stage2_threshold_pinned`` stays false. Publishing one threshold's numbers alone
  would leave a reader unable to tell an insensitive result from a tuned one, so the
  grid is carried and :func:`threshold_sensitivity` measures what moves across it.
  The mined set does not move; the *sparing composition* does, which is why the split
  in (1) is reported **with its threshold attached**.
* **A binding to the supplies.** ``remine`` writes no digest of the four evidence
  tables it read, so a publication could name any four files. :func:`reproduce_round`
  re-runs the shipped ``apply_remine_spare_rule`` over the recorded paths and refuses
  unless it reproduces the round report's outcome — the publication's supply digests
  are then facts about the bytes that produced these numbers, not a caption.

PRD §9.1, §18.1; ADR-0005 D14/A10/A1; ADR-0006 D11/A2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder.mining.remine import LEG_ROUND, load_probe_set, probe_member_ids
from tbox_finder.mining.spare_rule import ALL_DISJUNCTS, STAGE2_DISJUNCT

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-k"

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The grammar ``spare_rule.spare_reason`` emits. Parsed rather than pattern-matched
#: loosely: an unrecognised prefix means this module and the rule have drifted, and a
#: silent "not a pass, so count it as unavailable" would move candidates between the two
#: halves of the very split this module exists to publish.
REASON_PASSED_PREFIX = "passed:"
REASON_UNAVAILABLE_PREFIX = "unavailable_backend:"
REASON_MINABLE = "minable"

DEFAULT_FP_MANIFEST = "data/processed/mining/round0_fp_manifest.json"
DEFAULT_PROBE_IDS = "reports/p2/tier2n_probe_ids.json"
DEFAULT_OUT = "reports/p3/remine_round_report.json"

#: The §7 call this round was run under (TODO/dev-log 2026-08-12, AskUserQuestion).
SEVEN_DECISION = (
    "CLAUDE.md §7 decision 2026-08-12 (user): DECISION-ONLY ROUND, NO RETRAIN — run the "
    "round's decision legs (0)+(1) and publish the report as the honest answer to "
    "ADR-0005 A10's Phase-2 note; the DDP retrain leg (2) is not run and P3-15′-j (the "
    "mined-pool builder) is deferred, not scheduled."
)


class PublicationError(ValueError):
    """Raised on a round report this module cannot publish without overclaiming."""


# ═════════════════════════════════════════════════════════════════════════════
# (1) Why the spared were spared — re-derived from the per-candidate reasons
# ═════════════════════════════════════════════════════════════════════════════
def _round_block(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """The round report's ``round`` block, or the block itself if one was passed."""
    if not isinstance(report, Mapping):
        raise PublicationError(f"expected a round report object, got {type(report)}")
    block = report.get("round", report)
    if not isinstance(block, Mapping):
        raise PublicationError("the report's 'round' block is not an object")
    return block


def parse_spare_reason(reason: Any) -> tuple[str, tuple[str, ...]]:
    """``reason`` → ``(kind, disjuncts)`` for ``kind`` in ``passed``/``unavailable``.

    Every disjunct token is checked against :data:`ALL_DISJUNCTS`. A token this repo
    does not know is refused rather than bucketed: the two buckets are the published
    headline, and a typo'd or renamed disjunct silently landing in one of them would
    move the number without moving anything visible.
    """
    if not isinstance(reason, str):
        raise PublicationError(f"spare reason {reason!r} is not a string")
    if reason.startswith(REASON_PASSED_PREFIX):
        kind, body = "passed", reason[len(REASON_PASSED_PREFIX) :]
    elif reason.startswith(REASON_UNAVAILABLE_PREFIX):
        kind, body = "unavailable", reason[len(REASON_UNAVAILABLE_PREFIX) :]
    else:
        raise PublicationError(
            f"spare reason {reason!r} is not one of the shipped spare-rule attributions "
            f"({REASON_PASSED_PREFIX!r}…, {REASON_UNAVAILABLE_PREFIX!r}…); a spared "
            "candidate whose cause cannot be read must not be counted into either half"
        )
    disjuncts = tuple(tok for tok in body.split(",") if tok)
    if not disjuncts:
        raise PublicationError(f"spare reason {reason!r} names no disjunct")
    unknown = [d for d in disjuncts if d not in ALL_DISJUNCTS]
    if unknown:
        raise PublicationError(f"spare reason {reason!r} names unknown disjunct(s) {unknown}")
    return kind, disjuncts


def spared_split(report: Mapping[str, Any]) -> dict[str, Any]:
    """Split the spared candidates by **cause**, recomputed from ``reasons``.

    ``n_spared`` alone cannot distinguish the round ADR-0005 A10 recorded (every
    candidate spared because no backend ran) from a round where orthogonal criteria
    actually passed, and those are opposite scientific readings of the same number.

    ``n_spared_by_stage2_posterior_alone`` is the anti-circularity coordinate.
    ``spare_reason`` emits ``passed:high_stage2_posterior`` **only** when no
    model-independent disjunct passed, so that attribution is exactly the set of
    candidates protected by the project's own model and nothing else. Reported
    separately because ADR-0005 D14 calls the other three model-independent and this
    one is not, and a reader must be able to subtract it.
    """
    block = _round_block(report)
    reasons = block.get("reasons")
    spared = block.get("spared_ids")
    if not isinstance(reasons, Mapping) or not reasons:
        raise PublicationError("the round carries no per-candidate reasons map to split")
    if not isinstance(spared, list) or not spared:
        raise PublicationError(
            "the round carries no spared_ids, so a split of the spared would be vacuous"
        )
    recorded = block.get("n_spared")
    if recorded != len(spared):
        raise PublicationError(f"n_spared={recorded!r} disagrees with {len(spared)} spared_ids")

    passed_ids: list[str] = []
    unavailable_ids: list[str] = []
    passes: dict[str, int] = dict.fromkeys(ALL_DISJUNCTS, 0)
    missing: dict[str, int] = dict.fromkeys(ALL_DISJUNCTS, 0)
    stage2_alone = 0
    for candidate_id in spared:
        if candidate_id not in reasons:
            raise PublicationError(
                f"spared candidate {candidate_id!r} has no entry in the reasons map, so its "
                "cause cannot be published"
            )
        kind, disjuncts = parse_spare_reason(reasons[candidate_id])
        if kind == "passed":
            passed_ids.append(str(candidate_id))
            for disjunct in disjuncts:
                passes[disjunct] += 1
            if disjuncts == (STAGE2_DISJUNCT,):
                stage2_alone += 1
        else:
            unavailable_ids.append(str(candidate_id))
            for disjunct in disjuncts:
                missing[disjunct] += 1
    return {
        "n_spared": len(spared),
        "n_spared_by_a_passing_disjunct": len(passed_ids),
        "n_spared_by_unavailable_backend_only": len(unavailable_ids),
        "n_spared_by_stage2_posterior_alone": stage2_alone,
        "passes_per_disjunct": passes,
        "unavailable_per_disjunct": missing,
    }


# ═════════════════════════════════════════════════════════════════════════════
# (2) What the probe-exclusion zero actually measures
# ═════════════════════════════════════════════════════════════════════════════
def id_namespaces(ids: Sequence[str] | set[str]) -> list[str]:
    """The distinct ``<prefix>:`` tokens of an id set — its namespace, sorted.

    ``tier2n:CLASS_II_PLATFORM_SWAP:p00007`` and
    ``GCA_011056645.1:c228:8704:9345-9438`` are drawn from different id spaces, and
    that — not an empty probe set, not a filtered-out member — is why the round's
    exclusion count is 0. Measuring the namespaces makes the claim falsifiable: a
    substrate that ever shares one moves this list.
    """
    return sorted({str(i).split(":", 1)[0] for i in ids})


def probe_exclusion_diagnosis(
    *,
    probe_ids: set[str],
    substrate_ids: set[str],
    n_excluded: int,
) -> dict[str, Any]:
    """Classify the round's probe-exclusion count against what it could have been.

    Refuses both degenerate inputs outright. An empty probe set makes the exclusion a
    guaranteed no-op (``remine.load_probe_set`` already refuses it, and this is the
    second reader that must not accept it), and an empty substrate makes the zero a
    statement about nothing.
    """
    if not probe_ids:
        raise PublicationError(
            "the probe set is empty, so its exclusion count is a guaranteed 0 and measures "
            "nothing (ADR-0005 D14 per-round probe)"
        )
    if not substrate_ids:
        raise PublicationError(
            "the mining substrate is empty, so a 0 exclusion is a statement about nothing"
        )
    overlap = sorted(probe_ids & substrate_ids)
    if len(overlap) != int(n_excluded):
        raise PublicationError(
            f"the round excluded {n_excluded} probe member(s) but {len(overlap)} of the "
            "probe ids are present in the substrate; the report's exclusion count does not "
            "describe this substrate"
        )
    probe_ns = id_namespaces(probe_ids)
    substrate_ns = id_namespaces(substrate_ids)
    shared_ns = sorted(set(probe_ns) & set(substrate_ns))
    if overlap:
        reading = "members_excluded"
    elif shared_ns:
        reading = "disjoint_within_a_shared_namespace"
    else:
        reading = "structural_namespace_disjoint"
    return {
        "n_probe_members": len(probe_ids),
        "n_substrate_candidates": len(substrate_ids),
        "n_excluded": int(n_excluded),
        "excluded_ids": overlap,
        "probe_id_namespaces": probe_ns,
        "substrate_id_namespaces": substrate_ns,
        "shared_namespaces": shared_ns,
        "reading": reading,
        "note": (
            "a 0 here is NOT evidence the round left the Tier-2N probe intact under a "
            "shared substrate; on this round the two id spaces do not intersect at all, so "
            "no member could have been mined and the exclusion is structurally vacuous"
            if reading == "structural_namespace_disjoint"
            else "the probe ids and the substrate share a namespace, so this count is a "
            "measurement of the round rather than of the id spaces"
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Leg (0) — the structural preflight, carried into the publication rather than trusted
# ═════════════════════════════════════════════════════════════════════════════
#: Derivation keys carried into the publication. ``repo_root`` is deliberately dropped:
#: it is an absolute path on whichever machine ran the leg, and the plan report is
#: therefore NOT committed — only this sanitized summary is. ``reasons`` is dropped for
#: the same reason (its strings name files).
_DERIVATION_KEYS = (
    "msa_supply_derivation",
    "stage2_supply_derivation",
    "synteny_supply_derivation",
    "relaxed_arch_supply_derivation",
)


def plan_leg_summary(plan_report: Mapping[str, Any]) -> dict[str, Any]:
    """Leg (0)'s verdict, re-derived, with its machine-specific paths stripped.

    The two legs run as separate processes (on the cluster, as separate jobs), so the
    round leg cannot inherit the preflight's verdict and the publication should not
    inherit it either: ``may_run`` is recomputed here from the plan's own
    ``readiness × yield``, and :func:`publication_problems` then checks that the
    availability map leg (0) decided on is the one leg (1) mined under. A round whose
    two legs disagreed about which backends existed would otherwise publish leg (1)'s
    numbers under leg (0)'s authorisation.
    """
    if not isinstance(plan_report, Mapping):
        raise PublicationError("the plan leg's report is not an object")
    if plan_report.get("leg") != "plan":
        raise PublicationError(f"the plan report's leg is {plan_report.get('leg')!r}, not 'plan'")
    plan = plan_report.get("plan")
    if not isinstance(plan, Mapping):
        raise PublicationError("the plan report carries no plan block")
    readiness = plan.get("readiness")
    yield_gate = plan.get("yield")
    if not isinstance(readiness, Mapping) or not isinstance(yield_gate, Mapping):
        raise PublicationError("the plan block carries no readiness/yield gate")
    derivations: dict[str, Any] = {}
    for key in _DERIVATION_KEYS:
        block = plan.get(key)
        if not isinstance(block, Mapping):
            raise PublicationError(f"the plan report carries no {key}")
        clauses = block.get("clauses")
        if not isinstance(clauses, Mapping) or not clauses:
            raise PublicationError(f"{key} carries no clauses, so its verdict is unauditable")
        non_bool = sorted(k for k, v in clauses.items() if not isinstance(v, bool))
        if non_bool:
            raise PublicationError(f"{key} clauses {non_bool} are not booleans")
        derivations[key] = {"available": bool(block.get("available")), "clauses": dict(clauses)}
    return {
        "leg": "plan",
        "availability": dict(plan.get("availability") or {}),
        "ready": bool(readiness.get("ready")),
        "yield_producible": bool(yield_gate.get("yield_producible")),
        "may_run": bool(readiness.get("ready")) and bool(yield_gate.get("yield_producible")),
        "blocking_disjuncts": list(yield_gate.get("blocking_disjuncts") or []),
        "supply_derivations": derivations,
        "problems": list(plan_report.get("problems", [])),
        "report_not_committed_because": (
            "the plan leg's report embeds each derivation's absolute repo_root, so the file "
            "stays on scratch and only this path-free summary is published; its bytes are "
            "pinned by digest in 'supplies'"
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# (3) What the round did NOT do
# ═════════════════════════════════════════════════════════════════════════════
def retrain_disposition(report: Mapping[str, Any]) -> dict[str, Any]:
    """State, with its evidence, that legs (2)/(3) did not run.

    The evidence is the round report's own ``parent_checkpoint``: a retraining round
    records the checkpoint it re-mined *from*, so a decision-only round must carry
    ``None`` there. A publication claiming "no retrain" beside a report that names a
    parent is refused by :func:`publication_problems` rather than published with a
    caveat.
    """
    parent = report.get("parent_checkpoint")
    return {
        "retrain_leg_run": False,
        "parent_checkpoint_in_round_report": parent,
        "checkpoint_superseded": False,
        "canonical_stage1_checkpoint": "the P2 production checkpoint — unchanged by this round",
        "prd_18_1_supersession_status": "UNMET",
        "prd_18_1_note": (
            "PRD §18.1 states that stage1_production.ckpt supersedes the P2 checkpoint. This "
            "round decided and declined to retrain, so nothing superseded it; P3-16, P4 GATE-1 "
            "and the P5 scan continue to consume the P2 checkpoint. Reconciling PRD §18.1 with "
            "ADR-0005 is owed and needs user sign-off (CLAUDE.md §7 item 2)."
        ),
        "decision": SEVEN_DECISION,
    }


# ═════════════════════════════════════════════════════════════════════════════
# The Stage-2 operating point — unpinned, so its sensitivity is published
# ═════════════════════════════════════════════════════════════════════════════
def threshold_sensitivity(
    grid: Mapping[str, Mapping[str, Any]], *, published_threshold: float
) -> dict[str, Any]:
    """Measure what moves across the Stage-2 grid, and what does not.

    Two points are the minimum: a single-point "sensitivity" is the published number
    wearing a second name. The published threshold must itself be a grid point, or the
    headline is not one of the things this measured.
    """
    if len(grid) < 2:
        raise PublicationError(
            f"a sensitivity needs at least two grid points, got {len(grid)}; one point "
            "restates the headline rather than testing it"
        )
    points: dict[str, Any] = {}
    mined_sets: list[tuple[str, ...]] = []
    for key, report in sorted(grid.items(), key=lambda kv: float(kv[0])):
        block = _round_block(report)
        mined = tuple(sorted(str(i) for i in block.get("mined_ids", [])))
        split = spared_split(report)
        points[key] = {
            "n_mined": block.get("n_mined"),
            "mined_ids": list(mined),
            "n_spared_by_a_passing_disjunct": split["n_spared_by_a_passing_disjunct"],
            "n_spared_by_unavailable_backend_only": split["n_spared_by_unavailable_backend_only"],
            "n_spared_by_stage2_posterior_alone": split["n_spared_by_stage2_posterior_alone"],
        }
        mined_sets.append(mined)
    if not any(abs(float(k) - float(published_threshold)) < 1e-12 for k in grid):
        raise PublicationError(
            f"the published threshold {published_threshold} is not one of the grid points "
            f"{sorted(grid)}; the headline would sit outside its own sensitivity"
        )
    invariant = len(set(mined_sets)) == 1
    split_invariant = len({p["n_spared_by_a_passing_disjunct"] for p in points.values()}) == 1
    return {
        "grid": points,
        "mined_set_invariant": invariant,
        "spared_split_invariant": split_invariant,
        "reading": (
            "the mined set does not move across the grid, so the unpinned Stage-2 operating "
            "point does not decide this round's yield; the SPARING COMPOSITION does move, so "
            "the split below is reported with its threshold attached"
            if invariant and not split_invariant
            else (
                "the mined set does not move across the grid, and neither does the sparing "
                "composition"
                if invariant
                else "THE MINED SET MOVES ACROSS THE GRID — the unpinned Stage-2 operating "
                "point decides this round's yield and the value cannot be left to §13.1 "
                "without changing the result (CLAUDE.md §7)"
            )
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Binding the publication to the bytes that produced it
# ═════════════════════════════════════════════════════════════════════════════
def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome_fingerprint(report: Mapping[str, Any]) -> dict[str, Any]:
    """The four fields two runs of the same round must agree on, in a comparable shape."""
    block = _round_block(report)
    return {
        "n_candidates": block.get("n_candidates"),
        "n_mined": block.get("n_mined"),
        "n_spared": block.get("n_spared"),
        "mined_ids": sorted(str(i) for i in block.get("mined_ids", [])),
        "reasons_sha256": hashlib.sha256(
            json.dumps(block.get("reasons"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def reproduce_round(
    round_report: Mapping[str, Any],
    *,
    fp_manifest: str | Path,
    status_table: str | Path,
    posteriors: str | Path,
    relaxed_arch_status_table: str | Path,
    synteny_status_table: str | Path,
    probe_set_path: str | Path,
    stage2_threshold: float,
    union_prior: str | Path,
    corpus_parquet: str | Path,
) -> dict[str, Any]:
    """Re-run the shipped round leg over the recorded supplies and compare outcomes.

    ``remine`` records no digest of the four evidence tables it read, so without this a
    publication's ``supplies`` block would be a caption — any four paths could be named
    beside any numbers. Calling the shipped
    :func:`~tbox_finder.mining.remine.apply_remine_spare_rule` (never a second
    implementation of it) makes the digests facts about the bytes these numbers came
    from ([[gate-must-bind-to-upstream-evidence]]).
    """
    from tbox_finder.mining.remine import apply_remine_spare_rule

    replay = apply_remine_spare_rule(
        fp_manifest,
        status_table,
        posteriors,
        stage2_threshold=float(stage2_threshold),
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        probe_set=load_probe_set(probe_set_path),
        synteny_status_table=synteny_status_table,
        relaxed_arch_status_table=relaxed_arch_status_table,
        relaxed_arch_available=True,
        synteny_available=True,
        union_prior=union_prior,
        corpus_parquet=corpus_parquet,
    )
    published = outcome_fingerprint(round_report)
    replayed = outcome_fingerprint(replay)
    return {
        "reproduced": published == replayed,
        "published": published,
        "replayed": replayed,
    }


# ═════════════════════════════════════════════════════════════════════════════
# The publication and its self-check
# ═════════════════════════════════════════════════════════════════════════════
def build_publication(
    *,
    round_report: Mapping[str, Any],
    plan_report: Mapping[str, Any],
    substrate_ids: set[str],
    probe_ids: set[str],
    grid: Mapping[str, Mapping[str, Any]],
    supplies: Mapping[str, str],
    reproduction: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the published round report from re-derived measurements only."""
    block = _round_block(round_report)
    threshold = round_report.get("stage2_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise PublicationError(f"the round report's stage2_threshold is {threshold!r}")
    split = spared_split(round_report)
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": "ADR-0005 D14/A10/A1; ADR-0006 D11/A2; PRD §9.1, §18.1",
        "leg": round_report.get("leg"),
        "may_run": round_report.get("may_run"),
        "stage2_threshold": float(threshold),
        "stage2_threshold_pinned": bool(round_report.get("stage2_threshold_pinned", False)),
        "plan_leg": plan_leg_summary(plan_report),
        "outcome": {
            "n_candidates": block.get("n_candidates"),
            "n_mined": block.get("n_mined"),
            "n_masked": block.get("n_masked"),
            "n_spared": block.get("n_spared"),
            "n_refused_no_coordinates": block.get("n_refused_no_coordinates"),
            "mined_ids": sorted(str(i) for i in block.get("mined_ids", [])),
        },
        "spared_split": split,
        "probe_exclusion": probe_exclusion_diagnosis(
            probe_ids=probe_ids,
            substrate_ids=substrate_ids,
            n_excluded=int(block.get("n_excluded_probe_members", -1)),
        ),
        "retrain": retrain_disposition(round_report),
        "stage2_sensitivity": threshold_sensitivity(grid, published_threshold=float(threshold)),
        "supplies": dict(supplies),
        "reproduction": dict(reproduction),
        "round_report_problems": list(round_report.get("problems", [])),
        "provenance": dict(provenance),
    }


def publication_problems(pub: Mapping[str, Any]) -> list[str]:
    """Re-derive every claim the publication makes from the evidence it carries.

    Nothing here reads back a summary field the publication also wrote. The one
    exception is ``reproduction.reproduced``, which is a *measurement* taken against
    inputs this function is not given — so it is re-derived from the two fingerprints
    the measurement recorded, not trusted as a boolean.
    """
    problems: list[str] = []

    if pub.get("leg") != LEG_ROUND:
        problems.append(f"leg={pub.get('leg')!r} — this is not the round leg's report")
    if not pub.get("may_run"):
        problems.append("may_run is false — a refused round has no outcome to publish")
    if pub.get("stage2_threshold_pinned"):
        problems.append("stage2_threshold_pinned is true — §13.1 owns the value, not this round")
    if pub.get("round_report_problems"):
        problems.append(f"the round report self-check failed: {pub['round_report_problems']}")

    # Leg (0) authorised a round; leg (1) mined one. They ran as separate processes, so
    # the publication must show they decided under the SAME backend set — otherwise
    # leg (1)'s numbers are published under an authorisation for a different round.
    plan_leg = pub.get("plan_leg")
    if not isinstance(plan_leg, Mapping):
        problems.append("publication carries no plan-leg (structural preflight) summary")
    else:
        if plan_leg.get("problems"):
            problems.append(f"the plan leg's self-check failed: {plan_leg['problems']}")
        derived_may_run = bool(plan_leg.get("ready")) and bool(plan_leg.get("yield_producible"))
        if bool(plan_leg.get("may_run")) != derived_may_run:
            problems.append("the plan leg's may_run disagrees with its readiness × yield")
        if not derived_may_run:
            problems.append("the plan leg did not authorise this round")
        if plan_leg.get("blocking_disjuncts"):
            problems.append(
                f"the plan leg records blocking disjuncts {plan_leg['blocking_disjuncts']} yet a "
                "mining outcome was published"
            )
        availability = plan_leg.get("availability")
        if not isinstance(availability, Mapping) or sorted(availability) != sorted(ALL_DISJUNCTS):
            problems.append(
                "the plan leg's availability map does not name every disjunct "
                f"{list(ALL_DISJUNCTS)}"
            )
        elif not all(bool(v) for v in availability.values()):
            problems.append(
                "the plan leg ran with an unavailable backend, so every candidate on that "
                "disjunct was spared fail-closed and the yield is structurally capped"
            )
        derivations = plan_leg.get("supply_derivations")
        if not isinstance(derivations, Mapping) or sorted(derivations) != sorted(_DERIVATION_KEYS):
            problems.append("the plan leg carries no per-supply derivation for every supply")
        else:
            unevidenced = sorted(k for k, v in derivations.items() if not v.get("available"))
            if unevidenced:
                problems.append(
                    f"the round declared supplies its own checkout cannot evidence: {unevidenced}"
                )
            hollow = sorted(k for k, v in derivations.items() if not v.get("clauses"))
            if hollow:
                problems.append(f"supply derivations {hollow} carry no clauses to audit")

    outcome = pub.get("outcome")
    split = pub.get("spared_split")
    if not isinstance(outcome, Mapping) or not isinstance(split, Mapping):
        return problems + ["publication carries no outcome/spared_split block"]

    # The split must account for every spared candidate, and for no others.
    halves = (
        split.get("n_spared_by_a_passing_disjunct"),
        split.get("n_spared_by_unavailable_backend_only"),
    )
    if not all(isinstance(h, int) for h in halves):
        problems.append(f"the spared split is not a pair of counts: {halves!r}")
    elif sum(halves) != split.get("n_spared"):
        problems.append(
            f"the spared split {halves} does not sum to n_spared={split.get('n_spared')!r}"
        )
    if split.get("n_spared") != outcome.get("n_spared"):
        problems.append("spared_split.n_spared disagrees with the outcome's n_spared")
    # ...and the round must account for every candidate. A completeness clause beside the
    # correctness ones: a truncated run reproduces every ratio above on a subset
    # ([[cost-knobs-can-certify]]).
    parts = [
        outcome.get(k) for k in ("n_mined", "n_masked", "n_spared", "n_refused_no_coordinates")
    ]
    if not all(isinstance(p, int) for p in parts):
        problems.append(f"the outcome's four counts are not all integers: {parts!r}")
    elif sum(parts) != outcome.get("n_candidates"):
        problems.append(
            f"the outcome's four counts sum to {sum(parts)}, not "
            f"n_candidates={outcome.get('n_candidates')!r} — the round did not decide every "
            "candidate it claims to have read"
        )
    mined = outcome.get("mined_ids")
    if not isinstance(mined, list) or len(mined) != outcome.get("n_mined"):
        problems.append("n_mined disagrees with the mined id list")

    exclusion = pub.get("probe_exclusion")
    if not isinstance(exclusion, Mapping):
        problems.append("publication carries no probe-exclusion diagnosis")
    else:
        if not exclusion.get("n_probe_members"):
            problems.append("the probe exclusion ran against an empty probe set")
        if exclusion.get("n_substrate_candidates") != outcome.get("n_candidates"):
            problems.append(
                "the probe exclusion was diagnosed against a different substrate than the "
                "round decided"
            )
        # The reading must follow from the namespaces the diagnosis recorded, not from a
        # word written beside them.
        shared = exclusion.get("shared_namespaces")
        excluded = exclusion.get("excluded_ids")
        if isinstance(shared, list) and isinstance(excluded, list):
            derived = (
                "members_excluded"
                if excluded
                else (
                    "disjoint_within_a_shared_namespace"
                    if shared
                    else "structural_namespace_disjoint"
                )
            )
            if exclusion.get("reading") != derived:
                problems.append(
                    f"probe-exclusion reading {exclusion.get('reading')!r} does not follow "
                    f"from its own namespaces (derived {derived!r})"
                )

    retrain = pub.get("retrain")
    if not isinstance(retrain, Mapping):
        problems.append("publication carries no retrain disposition")
    else:
        if retrain.get("retrain_leg_run") or retrain.get("checkpoint_superseded"):
            problems.append(
                "the publication claims a retrain — P3-15′-k is the decision-only round"
            )
        if retrain.get("parent_checkpoint_in_round_report") is not None:
            problems.append(
                "the round report names a parent checkpoint, so it is not a decision-only "
                "round's report"
            )

    sensitivity = pub.get("stage2_sensitivity")
    if not isinstance(sensitivity, Mapping) or not isinstance(sensitivity.get("grid"), Mapping):
        problems.append("publication carries no Stage-2 sensitivity grid")
    else:
        grid = sensitivity["grid"]
        if len(grid) < 2:
            problems.append("the Stage-2 sensitivity has fewer than two grid points")
        derived_invariant = len({tuple(p.get("mined_ids", [])) for p in grid.values()}) == 1
        if bool(sensitivity.get("mined_set_invariant")) != derived_invariant:
            problems.append("mined_set_invariant disagrees with the grid's own mined id sets")
        published = [
            p
            for k, p in grid.items()
            if abs(float(k) - float(pub.get("stage2_threshold", "nan"))) < 1e-12
        ]
        if not published:
            problems.append("the published threshold is not one of the sensitivity grid points")
        elif published[0].get("mined_ids") != outcome.get("mined_ids"):
            problems.append(
                "the published threshold's grid point disagrees with the published outcome"
            )

    reproduction = pub.get("reproduction")
    if not isinstance(reproduction, Mapping):
        problems.append("publication carries no reproduction check against its supplies")
    else:
        derived = reproduction.get("published") == reproduction.get("replayed")
        if not derived:
            problems.append(
                "re-running the round leg over the recorded supplies did not reproduce the "
                "published outcome — the supply digests do not describe these numbers"
            )
        if bool(reproduction.get("reproduced")) != derived:
            problems.append("reproduction.reproduced disagrees with its own fingerprints")
        # ...and the fingerprint that was compared must be THIS publication's outcome. Without
        # this the reproduction could agree with itself about some other round entirely
        # ([[gate-must-bind-to-upstream-evidence]]).
        fingerprint = reproduction.get("published")
        if isinstance(fingerprint, Mapping):
            mismatched = sorted(
                k
                for k in ("n_candidates", "n_mined", "n_spared", "mined_ids")
                if fingerprint.get(k) != outcome.get(k)
            )
            if mismatched:
                problems.append(
                    f"the reproduction's published fingerprint disagrees with this "
                    f"publication's outcome on {mismatched}"
                )
        else:
            problems.append("the reproduction records no published fingerprint")

    supplies = pub.get("supplies")
    if not isinstance(supplies, Mapping) or not supplies:
        problems.append("publication names no supplies")
    else:
        bad = sorted(k for k, v in supplies.items() if not (isinstance(v, str) and len(v) == 64))
        if bad:
            problems.append(f"supply entries {bad} carry no sha256 digest")
    return problems


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def substrate_ids_from_manifest(path: str | Path) -> set[str]:
    """The mining substrate's candidate ids — the same manifest the round decided."""
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise PublicationError(f"{path}: expected an FP manifest object")
    rows = payload.get("candidates", payload.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise PublicationError(f"{path}: the FP manifest carries no candidate rows")
    ids = {str(r["candidate_id"]) for r in rows if isinstance(r, Mapping) and "candidate_id" in r}
    if len(ids) != len(rows):
        raise PublicationError(
            f"{path}: {len(rows)} rows collapse to {len(ids)} distinct candidate_ids — a "
            "duplicate id would merge two candidates into one substrate member"
        )
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tbox_finder.mining.remine_publish",
        description="Publish the P3-15 re-mining round report (P3-15′-k, decision-only).",
    )
    parser.add_argument("--round-report", required=True, help="the apply-spare-rule leg's report")
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="THRESHOLD=PATH",
        help="a Stage-2 sensitivity grid point; repeat (at least twice, including the "
        "published threshold)",
    )
    parser.add_argument("--manifest", default=DEFAULT_FP_MANIFEST)
    parser.add_argument("--probe-set", default=DEFAULT_PROBE_IDS)
    parser.add_argument("--status-table", required=True, help="covariation (a) status table")
    parser.add_argument("--posteriors", required=True, help="Stage-2 posterior table")
    parser.add_argument("--relaxed-arch-status", required=True, help="criterion (b) status table")
    parser.add_argument("--synteny-status", required=True, help="criterion (c) status table")
    parser.add_argument("--union-prior", default="data/processed/priors/union_prior.parquet")
    parser.add_argument("--corpus", default="data/processed/master_clean_v0.parquet")
    parser.add_argument(
        "--plan-report",
        required=True,
        help="leg (0)'s report; its verdict is republished path-free and its bytes pinned",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser


def _parse_grid(entries: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    grid: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if "=" not in entry:
            raise PublicationError(f"--grid entry {entry!r} is not THRESHOLD=PATH")
        key, path = entry.split("=", 1)
        key = str(float(key))
        if key in grid:
            raise PublicationError(f"--grid names threshold {key} twice")
        grid[key] = read_json(path)
    return grid


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    round_report = read_json(args.round_report)
    grid = _parse_grid(args.grid)

    supplies = {
        "covariation_status.json": sha256_file(args.status_table),
        "architecture_status.json": sha256_file(args.relaxed_arch_status),
        "synteny_status.json": sha256_file(args.synteny_status),
        "stage2_posteriors.json": sha256_file(args.posteriors),
        "round0_fp_manifest.json": sha256_file(args.manifest),
        "tier2n_probe_ids.json": sha256_file(args.probe_set),
        "remine_plan.json": sha256_file(args.plan_report),
    }

    reproduction = reproduce_round(
        round_report,
        fp_manifest=args.manifest,
        status_table=args.status_table,
        posteriors=args.posteriors,
        relaxed_arch_status_table=args.relaxed_arch_status,
        synteny_status_table=args.synteny_status,
        probe_set_path=args.probe_set,
        stage2_threshold=float(round_report["stage2_threshold"]),
        union_prior=args.union_prior,
        corpus_parquet=args.corpus,
    )

    pub = build_publication(
        round_report=round_report,
        plan_report=read_json(args.plan_report),
        substrate_ids=substrate_ids_from_manifest(args.manifest),
        probe_ids=set(probe_member_ids(load_probe_set(args.probe_set))),
        grid=grid,
        supplies=supplies,
        reproduction=reproduction,
        provenance={
            "rule": "src/tbox_finder/mining/remine_publish.py::main",
            "round_leg": "python -m tbox_finder.mining.remine apply-spare-rule",
            "plan_leg": "python -m tbox_finder.mining.remine plan",
            "compute": "LOCAL",
            "retrain_leg_run": False,
            "sbatch_not_used": (
                "slurm/p3/stage1_remine.sbatch runs leg (2) unconditionally on n_mined > 0 and "
                "has no decide-only mode, so the two CLI legs were invoked directly"
            ),
        },
    )
    problems = publication_problems(pub)
    pub["problems"] = problems
    if problems:
        # Refuse BEFORE the write. A published report carrying its own failed self-check is
        # a report a reader has to notice; not writing it is the fail-closed direction.
        print(f"remine publication self-check FAILED: {problems}", file=sys.stderr)
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pub, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"published {args.out} — n_mined={pub['outcome']['n_mined']} "
        f"spared {pub['spared_split']['n_spared_by_a_passing_disjunct']} by a passing disjunct / "
        f"{pub['spared_split']['n_spared_by_unavailable_backend_only']} by unavailable backend"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
