"""The P3 hard-negative **re-mining** round — the fourth spare-rule disjunct (P3-15).

ADR-0005 D14 (mirrored by ADR-0006 D11 and PRD §9.1) phase-conditions the mining
spare rule: three *model-independent* disjuncts are *"sufficient for the P2 Stage-1
mining loop where no Stage-2 exists yet"*, **or**, at the **P3 re-mining round once
Stage-2 exists**, a **high Stage-2 posterior**. This module is that round.

It is the P3 sibling of :mod:`tbox_finder.mining.mine_round` and it **imports**
rather than re-implements every operator that round already owns — the FP manifest
reader, the evidence builder, the union-prior mask, the readiness gate, the
Tier-2N probe decision. A second copy of a spare rule ADR-0005 D14 pins as *one*
rule is the failure this module is written to avoid, not a convenience it forgoes.

What P3-15 adds
---------------

**1. The fourth disjunct is accounted like the other three.** The pinned predicate
(:func:`tbox_finder.masking.spare_rule_excludes_from_mining`) has carried
``stage2_posterior``/``stage2_threshold`` since P2-07, but two-valued: a candidate
the Stage-2 producer never scored contributed *nothing* to the OR, so three failed
model-independent disjuncts made it **minable with Stage-2 never consulted**.
:func:`tbox_finder.mining.spare_rule.stage2_status` closes that (P3-15); this module
is the caller that supplies the evidence.

**2. A round cannot report success while mining nothing.** The shipped round-level
gate (:func:`~tbox_finder.mining.spare_rule.mining_round_readiness`) asks whether at
least one *protective* backend exists. That is necessary and **not sufficient**:
sparing needs one disjunct to pass, but **mining needs every disjunct to have run
and failed**, so a single unavailable backend drives the round's maximum possible
yield to exactly zero while readiness still reads ``ready``. ADR-0005 A10's
Phase-2 §7 note recorded that state at P2 as measured fact (*"``mining_round_readiness``
return ``ready = True`` but yield is structurally 0"*). :func:`max_possible_yield`
turns it into a computed refusal instead of a silent no-op, and
:func:`plan_remine_round` is what ``slurm/p3/stage1_remine.sbatch`` exits on —
before any GPU time.

Adding a *fourth* disjunct cannot relieve that: it is one more conjunct in the
mining condition. Whatever the Stage-2 producer scores, a round missing the
relaxed-architecture or synteny backend mines **0** candidates.

**3. Nothing is pinned.** Every rule parameter — the Stage-2 operating point that
becomes ``stage2_threshold`` above all — is keyword-only with **no default**, and
the CLI marks it ``required=True``. D14 pins the *rule*; the *value* is frozen at
the §13.1 phase gate, mirroring ADR-0005 D3 and the P3-12/P3-13/P3-14 discipline.
:func:`no_remine_parameter_has_a_default` re-derives that from the live signature.

The Stage-2 posterior **producer** — the leg that turns a coordinate-only
:class:`~tbox_finder.mining.hard_negative.MiningCandidate` into a scored payload —
is the analogue of :mod:`tbox_finder.mining.covariation_producer` and is gated by
:data:`STAGE2_SUPPLY_AVAILABLE`, exactly as the covariation supply is gated by
``mine_round.MSA_SUPPLY_AVAILABLE``. Recorded as data so the RUN blocker is
inspectable rather than buried in prose.

PRD §9.1; ADR-0005 D14/A9/A10; ADR-0006 D9 row 5, D11, A2.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder.eval.tier2n_probe import ProbeSet
from tbox_finder.mining.hard_negative import MiningCandidate
from tbox_finder.mining.mine_round import (
    build_round_availability,
    candidate_evidence,
    read_fp_manifest,
)
from tbox_finder.mining.spare_rule import (
    ALL_DISJUNCTS,
    MODEL_INDEPENDENT_DISJUNCTS,
    STAGE2_DISJUNCT,
    SpareRuleEvidence,
    mining_round_readiness,
)

SCHEMA_VERSION = "1.0"
STEP = "P3-15"

#: Whether the per-candidate **Stage-2 posterior supply** exists yet — the leg that
#: resolves a coordinate-only mining candidate to nucleotides, transcribes it, and
#: scores it through the calibrated Stage-2 re-ranker. It does not: P3-14 shipped
#: the contig-keyed two-stage harness, not a candidate-keyed producer. This is the
#: single flag the unblock step flips, the mirror of
#: ``mine_round.MSA_SUPPLY_AVAILABLE``. Recorded as data so a reader can see the
#: RUN blocker without reading prose.
STAGE2_SUPPLY_AVAILABLE = False


class RemineError(ValueError):
    """Raised on malformed re-mining input, or on a round that cannot mine."""


# ═════════════════════════════════════════════════════════════════════════════
# Backend availability — the three model-independent disjuncts + Stage-2
# ═════════════════════════════════════════════════════════════════════════════
def build_remine_availability(
    *,
    rscape_installed: bool,
    msa_supply_available: bool,
    stage2_supply_available: bool,
    relaxed_arch_available: bool = False,
    synteny_available: bool = False,
) -> dict[str, bool]:
    """The P3 availability map: the P2 three, **delegated**, plus the Stage-2 key.

    The three model-independent entries are produced by
    :func:`~tbox_finder.mining.mine_round.build_round_availability` — not recomputed
    here — so the MSA-producibility rule ADR-0006 A2's scope-guard correction pins
    (``any_helix_rscape = rscape_installed AND msa_supply_available``) has exactly
    one implementation.

    ``high_stage2_posterior`` is added as a **separate** key rather than folded into
    :data:`~tbox_finder.mining.spare_rule.MODEL_INDEPENDENT_DISJUNCTS`: D14 calls the
    first three model-independent and this one is the project's own model, a
    distinction the anti-circularity accounting depends on. It is likewise **not**
    added to ``TIER2N_PROTECTIVE_DISJUNCTS`` — were it, a P3 round could satisfy the
    readiness gate on the model's own posterior alone, which is precisely the
    "ready but mining nothing" no-op :func:`max_possible_yield` exists to refuse.
    """
    availability = build_round_availability(
        rscape_installed=rscape_installed,
        msa_supply_available=msa_supply_available,
        relaxed_arch_available=relaxed_arch_available,
        synteny_available=synteny_available,
    )
    availability[STAGE2_DISJUNCT] = bool(stage2_supply_available)
    return availability


# ═════════════════════════════════════════════════════════════════════════════
# The structural preflight — can this round mine ANY candidate at all?
# ═════════════════════════════════════════════════════════════════════════════
def max_possible_yield(
    availability: Mapping[str, bool],
    *,
    stage2_threshold: float | None,
) -> dict[str, Any]:
    """Upper-bound this round's mined count from the backend set alone.

    Sparing is a **disjunction** (one passing disjunct is enough); mining is its
    negation, so it is a **conjunction** — every disjunct must have run and failed.
    A disjunct whose backend is unavailable this round can only carry
    ``unavailable`` (``hard_negative.mine_round`` raises on any other status for an
    undeclared backend), and ``unavailable`` spares. Therefore **one** unavailable
    backend caps the round's yield at exactly 0, for every candidate, whatever the
    other backends score.

    This is a statement about the rule, not about the data, so it is decidable
    before a single window is scanned — which is the point: the P2 rounds were
    ``ready`` and mined 0 (ADR-0005 A10 Phase-2 §7 deferral), and the round reported
    that as success. ``max_mined`` is ``0`` when provably zero and ``None`` when not
    bounded here (never a fabricated positive count).

    A P3 round that supplies no ``stage2_threshold`` has not declared Stage-2 live,
    so the fourth disjunct is unevaluated for every candidate — also a hard 0.
    """
    blocking: list[str] = [d for d in MODEL_INDEPENDENT_DISJUNCTS if not availability.get(d, False)]
    # Two independent ways the fourth disjunct goes unevaluated, and both are a hard
    # zero: the round never declared Stage-2 live, or it did and has no producer.
    no_threshold = stage2_threshold is None
    no_producer = not availability.get(STAGE2_DISJUNCT, False)
    if no_threshold or no_producer:
        blocking.append(STAGE2_DISJUNCT)

    producible = not blocking
    if producible:
        reason = None
    else:
        reason = (
            "mining requires EVERY disjunct to have run and failed, but "
            f"{blocking} {'has' if len(blocking) == 1 else 'have'} no backend this round; "
            "every candidate is spared fail-closed, so the round's maximum yield is 0 "
            "(ADR-0005 D14 spare rule; ADR-0006 A2 fail-closed sparing)"
        )
    return {
        "yield_producible": producible,
        "blocking_disjuncts": blocking,
        "max_mined": None if producible else 0,
        "all_disjuncts": list(ALL_DISJUNCTS),
        "stage2_declared_live": stage2_threshold is not None,
        "reason": reason,
    }


def plan_remine_round(
    *,
    rscape_installed: bool,
    msa_supply_available: bool,
    stage2_supply_available: bool | None = None,
    stage2_threshold: float | None,
    relaxed_arch_available: bool = False,
    synteny_available: bool = False,
) -> dict[str, Any]:
    """Decide whether the P3 re-mining round may run — before any GPU time.

    ``may_run`` is the conjunction of the two gates, which answer different
    questions and neither of which implies the other:

    * ``readiness`` (shipped, ADR-recorded) — is there a *protective* backend, so
      the rule can spare a Tier-2N locus at all?
    * ``yield`` (P3-15) — can the rule mine anything, i.e. did **every** disjunct
      run?

    ``stage2_supply_available`` defaults to :data:`STAGE2_SUPPLY_AVAILABLE`, resolved
    through a ``None`` sentinel at **call** time rather than bound at import, so
    flipping the flag at the unblock step reaches every caller including direct ones.
    """
    if stage2_supply_available is None:
        stage2_supply_available = STAGE2_SUPPLY_AVAILABLE
    availability = build_remine_availability(
        rscape_installed=rscape_installed,
        msa_supply_available=msa_supply_available,
        stage2_supply_available=stage2_supply_available,
        relaxed_arch_available=relaxed_arch_available,
        synteny_available=synteny_available,
    )
    readiness = mining_round_readiness(availability)
    yield_gate = max_possible_yield(availability, stage2_threshold=stage2_threshold)
    return {
        "availability": availability,
        "readiness": readiness,
        "yield": yield_gate,
        "ready": bool(readiness["ready"]),
        "yield_producible": bool(yield_gate["yield_producible"]),
        "may_run": bool(readiness["ready"]) and bool(yield_gate["yield_producible"]),
        "stage2_supply_available": bool(stage2_supply_available),
        "msa_supply_available": bool(msa_supply_available),
        "rscape_installed": bool(rscape_installed),
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Per-candidate evidence — the Stage-2 posterior join, fail-closed on a miss
# ═════════════════════════════════════════════════════════════════════════════
def remine_candidate_evidence(
    candidate_id: str,
    *,
    covariation_status: Mapping[str, str] | None,
    stage2_posteriors: Mapping[str, float] | None,
) -> SpareRuleEvidence:
    """The candidate's P3 evidence: the P2 builder's result **plus** its posterior.

    The three model-independent statuses are produced by
    :func:`~tbox_finder.mining.mine_round.candidate_evidence` and then extended with
    ``dataclasses.replace`` — the P2 builder is called, never forked, so its
    fail-closed "an id absent from the map resolves to ``unavailable``" rule is the
    same rule here.

    The Stage-2 join carries the same discipline: an id **absent** from
    ``stage2_posteriors`` yields ``None``, which
    :func:`~tbox_finder.mining.spare_rule.stage2_status` reads as ``unavailable`` ⇒
    the candidate is **spared**. A dropped producer shard therefore costs
    sensitivity, never a mined true T-box.
    """
    base = candidate_evidence(candidate_id, covariation_status)
    if stage2_posteriors is None:
        return base
    posterior = stage2_posteriors.get(candidate_id)
    if posterior is None:
        return base
    return dataclasses.replace(base, stage2_posterior=float(posterior))


def load_stage2_posteriors(path: str | Path) -> dict[str, float]:
    """Read the producer's ``candidate_id → calibrated Stage-2 posterior`` table.

    The value is the **named** (pre-prior-shift, temperature-scaled) posterior —
    ``two_stage``'s ``stage2_named_posterior`` column, ADR-0005 D11's gated object —
    because that is the quantity the Stage-2 operating point is defined on. Values
    are range-validated here so a logit written into the posterior column raises at
    the join rather than silently failing every ``>=`` comparison and mining the
    candidates it was supposed to spare.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["posteriors"] if isinstance(payload, dict) else payload
    if not isinstance(rows, Mapping):
        raise RemineError(f"{path}: expected a candidate_id → posterior mapping, got {type(rows)}")
    out: dict[str, float] = {}
    for key, value in rows.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RemineError(f"{path}: posterior for {key!r} is not a real number ({value!r})")
        if not 0.0 <= float(value) <= 1.0:
            raise RemineError(
                f"{path}: posterior for {key!r} is {value}, outside [0, 1] — a calibrated "
                "posterior was expected (ADR-0005 D11 named posterior), not a logit"
            )
        out[str(key)] = float(value)
    return out


def read_remine_manifest(
    path: str | Path,
    *,
    covariation_status: Mapping[str, str] | None = None,
    stage2_posteriors: Mapping[str, float] | None = None,
) -> list[MiningCandidate]:
    """The round's false positives, stamped with **both** produced evidence sources.

    Delegates the manifest parse to
    :func:`~tbox_finder.mining.mine_round.read_fp_manifest` and then re-stamps each
    candidate's evidence through :func:`remine_candidate_evidence`, so the P2 reader
    stays the single parser of the manifest shape.
    """
    candidates = read_fp_manifest(path, covariation_status=covariation_status)
    return [
        dataclasses.replace(
            c,
            evidence=remine_candidate_evidence(
                c.candidate_id,
                covariation_status=covariation_status,
                stage2_posteriors=stage2_posteriors,
            ),
        )
        for c in candidates
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Tier-2N probe members are never mining candidates
# ═════════════════════════════════════════════════════════════════════════════
def probe_member_ids(probe_set: ProbeSet) -> frozenset[str]:
    """Every id in the Tier-2N probe set — both arms."""
    return frozenset(probe_set.natural) | frozenset(probe_set.synthetic)


def load_probe_set(path: str | Path) -> ProbeSet:
    """Read the round's Tier-2N probe set from an explicit ``natural``/``synthetic`` id list.

    Deliberately **not** read from ``reports/p2/tier2n_probe.json``: that artifact
    carries counts (``probe_set_size``, ``n_synthetic``), not member ids, so a loader
    pointed at it would build an **empty** :class:`ProbeSet` and the exclusion below
    would run clean while excluding nothing — a filter that reports success and
    filters nothing is the shape this module exists to refuse.

    An empty probe set is therefore refused outright rather than accepted: with no
    members, ``exclude_probe_members`` is a guaranteed no-op and its ``0`` becomes
    indistinguishable from a real measurement.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RemineError(f"{path}: expected an object with 'natural'/'synthetic' id lists")
    natural = tuple(str(x) for x in payload.get("natural", ()))
    synthetic = tuple(str(x) for x in payload.get("synthetic", ()))
    probe_set = ProbeSet(natural=natural, synthetic=synthetic)
    if probe_set.size == 0:
        raise RemineError(
            f"{path}: the probe set is empty, so excluding its members from the mining "
            "substrate would be a guaranteed no-op reporting 0 exclusions; supply the "
            "round's real Tier-2N probe ids (ADR-0005 D14 per-round probe)"
        )
    return probe_set


def exclude_probe_members(
    candidates: Sequence[MiningCandidate], probe_set: ProbeSet
) -> tuple[list[MiningCandidate], list[str]]:
    """Split the candidates into ``(mineable_substrate, excluded_probe_ids)``.

    The Tier-2N probe is the instrument that decides whether a round degraded the
    flagship class (ADR-0005 D14 per-round halt/rollback). A probe member mined as a
    hard negative would train the scanner against the very locus the next round then
    measures recall on — the measurement and the treatment would share a record, and
    a recall drop would be indistinguishable from the probe having been trained away.

    Returns the excluded ids rather than only a count, so a round report names which
    members were removed instead of asserting that none were present.
    """
    members = probe_member_ids(probe_set)
    keep = [c for c in candidates if c.candidate_id not in members]
    dropped = sorted(c.candidate_id for c in candidates if c.candidate_id in members)
    return keep, dropped


# ═════════════════════════════════════════════════════════════════════════════
# The round itself — four-disjunct evidence → mined/spared outcome
# ═════════════════════════════════════════════════════════════════════════════
def apply_remine_spare_rule(
    fp_manifest: str | Path,
    status_table: str | Path,
    posteriors: str | Path,
    *,
    stage2_threshold: float,
    rscape_installed: bool,
    msa_supply_available: bool,
    stage2_supply_available: bool,
    probe_set: ProbeSet,
    relaxed_arch_available: bool = False,
    synteny_available: bool = False,
    union_prior: str | Path = "data/processed/priors/union_prior.parquet",
    corpus_parquet: str | Path = "data/processed/master_clean_v0.parquet",
) -> dict[str, Any]:
    """The P3 round's spare-rule leg: produced evidence → mined/spared outcome.

    The P2 leg's shape (:func:`~tbox_finder.mining.mine_round.apply_spare_rule`) with
    two additions — the Stage-2 posterior join, and the Tier-2N probe members removed
    from the substrate before the rule runs. The union-prior mask is built by the
    **promoted** :func:`~tbox_finder.mining.mine_round.load_union_mask`, so both rounds
    mask against one arithmetic.

    ``probe_set`` is **required, not optional**. Review found the first draft's optional
    parameter unwired from the CLI the sbatch actually invokes, so the exclusion never
    ran and the report's ``n_excluded_probe_members: 0`` read as *"no probe member was in
    the substrate"* rather than *"no exclusion happened"*. Making it required makes that
    state unrepresentable, and the report carries ``n_probe_members_considered`` beside
    the exclusion count so a 0 is readable against a non-zero denominator.

    :func:`~tbox_finder.mining.hard_negative.mine_round` still owns the refusal and the
    evidence↔availability cross-check; this function never decides mineability itself.
    """
    from tbox_finder.mining.covariation_producer import load_status_map
    from tbox_finder.mining.hard_negative import mine_round as run_mine_round
    from tbox_finder.mining.mine_round import load_union_mask

    availability = build_remine_availability(
        rscape_installed=rscape_installed,
        msa_supply_available=msa_supply_available,
        stage2_supply_available=stage2_supply_available,
        relaxed_arch_available=relaxed_arch_available,
        synteny_available=synteny_available,
    )
    candidates = read_remine_manifest(
        fp_manifest,
        covariation_status=load_status_map(status_table),
        stage2_posteriors=load_stage2_posteriors(posteriors),
    )
    candidates, excluded = exclude_probe_members(candidates, probe_set)

    mask = load_union_mask(union_prior=union_prior, corpus_parquet=corpus_parquet)
    report = run_mine_round(
        candidates, mask, availability, stage2_threshold=float(stage2_threshold)
    )
    report["schema_version"] = SCHEMA_VERSION
    report["step"] = STEP
    report["n_excluded_probe_members"] = len(excluded)
    report["excluded_probe_member_ids"] = list(excluded)
    # The denominator: a 0 exclusion against a non-zero considered count is a
    # measurement; a 0 against a 0 would be a no-op wearing the same number.
    report["n_probe_members_considered"] = len(probe_member_ids(probe_set))
    return report


# ═════════════════════════════════════════════════════════════════════════════
# The round report + its self-certification clause set
# ═════════════════════════════════════════════════════════════════════════════
def build_remine_report(
    *,
    plan: Mapping[str, Any],
    round_report: Mapping[str, Any] | None,
    probe_trace: Mapping[str, Any] | None,
    excluded_probe_ids: Sequence[str] = (),
    stage2_threshold: float | None,
    parent_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Assemble the P3 re-mining round report.

    ``round_report`` and ``probe_trace`` are ``None`` when the plan refused — a
    refused round has no mined count and no probe recall, and writing ``0``/``0.0``
    for them would be indistinguishable from a round that ran and found nothing.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": "ADR-0005 D14 (Stage-2 disjunct, P3-phased); ADR-0006 D11/A2",
        "plan": dict(plan),
        "may_run": bool(plan.get("may_run", False)),
        "stage2_threshold": None if stage2_threshold is None else float(stage2_threshold),
        "stage2_threshold_pinned": False,
        "parent_checkpoint": parent_checkpoint,
        "round": None if round_report is None else dict(round_report),
        "tier2n_probe": None if probe_trace is None else dict(probe_trace),
        "n_excluded_probe_members": len(tuple(excluded_probe_ids)),
        "excluded_probe_member_ids": sorted(excluded_probe_ids),
    }


def remine_problems(report: Mapping[str, Any]) -> list[str]:
    """Re-derive the report's own claims from its recorded evidence.

    Every clause is recomputed from what the report carries, never read back from a
    summary field it also wrote — a clause that reads its own conclusion can only
    catch a value flipped false, never one written true.
    """
    problems: list[str] = []
    plan = report.get("plan")
    if not isinstance(plan, Mapping):
        return ["report carries no plan block"]

    readiness = plan.get("readiness")
    yield_gate = plan.get("yield")
    if not isinstance(readiness, Mapping) or not isinstance(yield_gate, Mapping):
        return ["plan block carries no readiness/yield gate"]

    derived_may_run = bool(readiness.get("ready")) and bool(yield_gate.get("yield_producible"))
    if bool(report.get("may_run")) != derived_may_run:
        problems.append(
            f"may_run={report.get('may_run')} disagrees with readiness×yield={derived_may_run}"
        )

    availability = plan.get("availability")
    if isinstance(availability, Mapping):
        blocking = sorted(d for d in ALL_DISJUNCTS if not availability.get(d, False))
        recorded = sorted(str(d) for d in yield_gate.get("blocking_disjuncts", []))
        # A round that declared no threshold blocks the Stage-2 disjunct on that
        # ground alone, so the availability-derived set is a subset, not an equality.
        if not set(blocking) <= set(recorded):
            problems.append(f"blocking_disjuncts {recorded} omits unavailable backends {blocking}")
        if derived_may_run and blocking:
            problems.append(f"may_run is True while {blocking} have no backend")

    if report.get("stage2_threshold_pinned"):
        problems.append("stage2_threshold_pinned is True — P3-15 pins no value (ADR-0005 D3/D14)")

    if not bool(report.get("may_run")):
        if report.get("round") is not None:
            problems.append("a refused round carries a mining outcome")
        if report.get("tier2n_probe") is not None:
            problems.append("a refused round carries a Tier-2N probe trace")
        return problems

    # A round that ran must show the Tier-2N exclusion actually ran. Without the
    # denominator, `n_excluded_probe_members: 0` reads as "no probe member was in the
    # substrate" when it may mean "no exclusion happened" — the state review found in
    # this module's first draft, where the CLI never passed a probe set at all.
    round_report = report.get("round")
    if not isinstance(round_report, Mapping):
        problems.append("a round that may run carries no mining outcome")
        return problems
    considered = round_report.get("n_probe_members_considered")
    if not isinstance(considered, int) or considered <= 0:
        problems.append(
            f"n_probe_members_considered={considered!r} — the Tier-2N exclusion did not run "
            "against a non-empty probe set, so its exclusion count measures nothing"
        )
    excluded_ids = round_report.get("excluded_probe_member_ids")
    if not isinstance(excluded_ids, list) or len(excluded_ids) != round_report.get(
        "n_excluded_probe_members"
    ):
        problems.append("n_excluded_probe_members disagrees with the excluded id list")
    return problems


def no_remine_parameter_has_a_default(func: Any = None) -> bool:
    """True iff every keyword-only rule parameter of ``func`` has **no** default.

    P3-15 pins no value: the Stage-2 operating point that becomes
    ``stage2_threshold`` is frozen at the §13.1 phase gate, not here. Re-derived
    from the live signature so adding a default fails this rather than passing a
    hand-maintained list. ``*_available`` flags are backend-presence facts, not rule
    parameters, and are exempt by name.
    """
    target = plan_remine_round if func is None else func
    exempt = {"stage2_supply_available", "relaxed_arch_available", "synteny_available"}
    for name, param in inspect.signature(target).parameters.items():
        if param.kind is not inspect.Parameter.KEYWORD_ONLY or name in exempt:
            continue
        if param.default is not inspect.Parameter.empty and param.default is not None:
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def write_json(path: str | Path, payload: Any) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Path(path)


def _bool_flag(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    parser.add_argument(f"--{name}", action="store_true", help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tbox_finder.mining.remine",
        description="P3 hard-negative re-mining round (ADR-0005 D14 Stage-2 disjunct).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _round_args(p: argparse.ArgumentParser) -> None:
        # No default anywhere: D14 pins the rule, the §13.1 phase gate pins the value.
        p.add_argument(
            "--stage2-threshold",
            type=float,
            required=True,
            help=(
                "the high-Stage-2-posterior operating point "
                "(no default; §13.1 freezes the value)"
            ),
        )
        _bool_flag(p, "rscape-installed", "the pinned R-scape build is present")
        _bool_flag(
            p, "msa-supply-available", "the per-candidate MSA supply exists (ADR-0006 D7/A1)"
        )
        _bool_flag(
            p, "stage2-supply-available", "the per-candidate Stage-2 posterior supply exists"
        )
        _bool_flag(p, "relaxed-arch-available", "a relaxed-architecture backend exists")
        _bool_flag(p, "synteny-available", "a downstream-aaRS synteny backend exists")
        p.add_argument("--out", required=True, help="where to write the round report")

    plan = sub.add_parser("plan", help="decide whether the round may run, and write the reason")
    _round_args(plan)

    apply_ = sub.add_parser(
        "apply-spare-rule", help="produced evidence → the round's mined/spared outcome"
    )
    _round_args(apply_)
    apply_.add_argument("--manifest", required=True, help="the round's FP manifest")
    apply_.add_argument("--status-table", required=True, help="merged covariation-status table")
    apply_.add_argument("--posteriors", required=True, help="candidate_id → Stage-2 posterior")
    apply_.add_argument(
        "--probe-set",
        required=True,
        help="Tier-2N probe ids ({natural, synthetic}); its members are removed from the substrate",
    )
    apply_.add_argument("--union-prior", default="data/processed/priors/union_prior.parquet")
    apply_.add_argument("--corpus", default="data/processed/master_clean_v0.parquet")
    return parser


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = plan_remine_round(
        rscape_installed=bool(args.rscape_installed),
        msa_supply_available=bool(args.msa_supply_available),
        stage2_supply_available=bool(args.stage2_supply_available),
        stage2_threshold=float(args.stage2_threshold),
        relaxed_arch_available=bool(args.relaxed_arch_available),
        synteny_available=bool(args.synteny_available),
    )
    report = build_remine_report(
        plan=plan,
        round_report=None,
        probe_trace=None,
        stage2_threshold=float(args.stage2_threshold),
    )
    problems = remine_problems(report)
    report["problems"] = problems
    write_json(args.out, report)
    if problems:
        print(f"remine report self-check FAILED: {problems}")
        return 2
    if not plan["may_run"]:
        print("re-mining round REFUSED — no GPU time spent")
        if not plan["ready"]:
            print(f"  readiness: {plan['readiness']['refusal_reason']}")
        if not plan["yield_producible"]:
            print(f"  yield: {plan['yield']['reason']}")
        return 1
    print("re-mining round may run")
    return 0


def _cmd_apply_spare_rule(args: argparse.Namespace) -> int:
    plan = plan_remine_round(
        rscape_installed=bool(args.rscape_installed),
        msa_supply_available=bool(args.msa_supply_available),
        stage2_supply_available=bool(args.stage2_supply_available),
        stage2_threshold=float(args.stage2_threshold),
        relaxed_arch_available=bool(args.relaxed_arch_available),
        synteny_available=bool(args.synteny_available),
    )
    if not plan["may_run"]:
        # The preflight is not advisory: a round that provably cannot mine must not
        # produce a mined/spared outcome at all, or the zeroes read as a real result.
        report = build_remine_report(
            plan=plan,
            round_report=None,
            probe_trace=None,
            stage2_threshold=float(args.stage2_threshold),
        )
        report["problems"] = remine_problems(report)
        write_json(args.out, report)
        print("re-mining round REFUSED at the preflight — no mining outcome written")
        return 1
    probe_set = load_probe_set(args.probe_set)
    round_report = apply_remine_spare_rule(
        args.manifest,
        args.status_table,
        args.posteriors,
        stage2_threshold=float(args.stage2_threshold),
        probe_set=probe_set,
        rscape_installed=bool(args.rscape_installed),
        msa_supply_available=bool(args.msa_supply_available),
        stage2_supply_available=bool(args.stage2_supply_available),
        relaxed_arch_available=bool(args.relaxed_arch_available),
        synteny_available=bool(args.synteny_available),
        union_prior=args.union_prior,
        corpus_parquet=args.corpus,
    )
    report = build_remine_report(
        plan=plan,
        round_report=round_report,
        probe_trace=None,
        excluded_probe_ids=round_report["excluded_probe_member_ids"],
        stage2_threshold=float(args.stage2_threshold),
    )
    problems = remine_problems(report)
    report["problems"] = problems
    write_json(args.out, report)
    if problems:
        print(f"remine report self-check FAILED: {problems}")
        return 2
    print(f"n_mined={round_report['n_mined']} n_spared={round_report['n_spared']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "plan":
        return _cmd_plan(args)
    if args.cmd == "apply-spare-rule":
        return _cmd_apply_spare_rule(args)
    raise RemineError(f"unknown subcommand {args.cmd!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
