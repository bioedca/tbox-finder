"""P2-10e — the hard-negative-mining **round driver** (scan → FP-collect → spare → retrain).

One §9.1 mining round, composed from existing entrypoints — this module never re-derives
the operators it stands on:

* scan a checkpoint over a sequence → :func:`tbox_finder.infer.scan.scan_sequence`;
* posteriors → called loci → :func:`tbox_finder.infer.call.call_candidates`, under the
  **provisional cross-round-constant criterion** pinned in
  :mod:`tbox_finder.eval.mining_criterion` (ADR-0005 A9 Pin 3);
* union-prior masking + the three-valued spare rule + the round readiness gate →
  :mod:`tbox_finder.mining.hard_negative` / :mod:`tbox_finder.mining.spare_rule`;
* the per-round Tier-2N halt/rollback → :mod:`tbox_finder.eval.tier2n_probe`.

Readiness first, and *honestly* (the ADR-0006 A2 scope-guard correction)
------------------------------------------------------------------------
:func:`tbox_finder.mining.spare_rule.mining_round_readiness` refuses a round unless at
least one **protective** spare-rule backend — covariation-(a) or synteny-(c) — is
available; relaxed-architecture-(b) alone is ``False`` on every Tier-2N locus (ADR-0006 D9
row 5) and cannot spare the class the rule protects.
:func:`tbox_finder.mining.hard_negative.mine_round` checks this **first and raises** when unmet.

ADR-0006 A2's scope guard says the covariation pins "do not block a P2-10e submit … an
unscored (a)-leg → unavailable → spared, never silently mined". That is true of the
**per-candidate** sparing, but it misses the **round-level** gate: the covariation leg is
"available" only if per-candidate MSAs can actually be produced (the D7/A1 homolog-search +
CM-free de-novo alignment). :func:`covariation.backend_available` answers the narrower
question "is R-scape *installed*". If a round declared covariation available merely because
R-scape is installed, readiness would pass and the round would run as a **no-op** — no MSAs
⇒ every candidate ``unavailable`` ⇒ every candidate spared ⇒ zero mined — yet report
success, the exact degradation the readiness gate exists to prevent (but does not catch,
because it keys on ``backend_available``). So :func:`build_round_availability` gates
``any_helix_rscape`` on **MSA-producibility**, not on R-scape being installed.

The consequence, recorded honestly: at P2 the per-candidate homolog-search MSA supply
(ADR-0006 D7 / A1) is **not built**, and no synteny backend exists, so
:data:`MSA_SUPPLY_AVAILABLE` is ``False``, no protective backend is available, and every
round is **refused** at readiness. The N=4 RUN is therefore blocked until that supply lands
(a downstream step); this driver is the correct fail-closed machinery, and
``slurm/p2/mine_round.sbatch`` exits at the readiness preflight before spending any GPU time.

``numpy``-tier (via :mod:`tbox_finder.eval.mining_criterion` → ``infer.call``) but
**torch-free at import**: the checkpoint scan imports torch lazily inside the two scan
functions, so the readiness/adapter/decision surface unit-tests without a GPU. PRD §9.1;
ADR-0005 D3 + D14 + A9; ADR-0006 D9 / D11 + A2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tbox_finder.eval import mining_criterion
from tbox_finder.eval.tier2n_probe import ProbeSet, probe_recall, round_decision
from tbox_finder.mining.hard_negative import MiningCandidate
from tbox_finder.mining.host_order import HostOrderError, host_accession
from tbox_finder.mining.spare_rule import (
    SpareRuleEvidence,
    mining_round_readiness,
)

SCHEMA_VERSION = "1.0"
STEP = "P2-10e"

#: The genomic-window mining pool the a2 substrate FPs belong to (``mining/pool.py``).
GENOMIC_WINDOW_POOL = "genomic_window"

#: Whether the per-candidate covariation MSA supply (ADR-0006 D7 / A1: homolog-DB search +
#: CM-free de-novo alignment) exists **yet**. It does not — the homolog-search *target DB*
#: was built (P2-10c′-homologdb) but the per-candidate search + alignment that turns a
#: candidate into an MSA is downstream infrastructure. This is the single flag the MSA-supply
#: unblock step flips to ``True``; until then no covariation verdict is producible and — with
#: no synteny backend either — every round is refused at readiness. Recorded as data so the
#: RUN blocker is inspectable, not buried in prose.
MSA_SUPPLY_AVAILABLE = False


class MineRoundError(ValueError):
    """Raised on malformed round-driver input (bad window id, empty probe set)."""


# ═════════════════════════════════════════════════════════════════════════════
# Round readiness — the honest, MSA-producibility-gated availability map
# ═════════════════════════════════════════════════════════════════════════════
def build_round_availability(
    *,
    rscape_installed: bool,
    msa_supply_available: bool,
    relaxed_arch_available: bool = False,
    synteny_available: bool = False,
) -> dict[str, bool]:
    """Build the honest ``mining_round_readiness`` availability map.

    ``any_helix_rscape`` is available **iff** R-scape is installed *and* the per-candidate
    MSA supply exists — R-scape installed is necessary but not sufficient (a round with no
    MSAs would spare every candidate and mine nothing; ADR-0006 A2 scope-guard correction).
    This is deliberately stricter than
    :func:`tbox_finder.mining.covariation.round_backend_availability`, which keys
    ``any_helix_rscape`` on :func:`~tbox_finder.mining.covariation.backend_available` alone.
    """
    return {
        "relaxed_architecture": bool(relaxed_arch_available),
        "any_helix_rscape": bool(rscape_installed) and bool(msa_supply_available),
        "downstream_aaRS_synteny": bool(synteny_available),
    }


def plan_round(
    *,
    rscape_installed: bool,
    msa_supply_available: bool | None = None,
    relaxed_arch_available: bool = False,
    synteny_available: bool = False,
) -> dict[str, Any]:
    """Decide whether a round may run, and record why not — before any GPU time.

    Returns ``{availability, readiness, ready, ...}``. ``ready`` is ``False`` at P2 (no
    protective backend), and the driver's caller (``mine_round.sbatch``) exits on it rather
    than scanning 3.63M windows only to have :func:`hard_negative.mine_round` refuse.

    ``msa_supply_available`` defaults to the module flag :data:`MSA_SUPPLY_AVAILABLE`, resolved
    via a ``None`` sentinel at **call** time (not bound at import) so that flipping the flag at
    the unblock step reaches every caller, including direct ones.
    """
    if msa_supply_available is None:
        msa_supply_available = MSA_SUPPLY_AVAILABLE
    availability = build_round_availability(
        rscape_installed=rscape_installed,
        msa_supply_available=msa_supply_available,
        relaxed_arch_available=relaxed_arch_available,
        synteny_available=synteny_available,
    )
    readiness = mining_round_readiness(availability)
    return {
        "availability": availability,
        "readiness": readiness,
        "ready": bool(readiness["ready"]),
        "msa_supply_available": bool(msa_supply_available),
        "rscape_installed": bool(rscape_installed),
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
    }


# ═════════════════════════════════════════════════════════════════════════════
# FP-collect — called loci (window coordinates) → MiningCandidate (genome coordinates)
# ═════════════════════════════════════════════════════════════════════════════
def parse_window_name(window_name: str) -> tuple[str, int, int]:
    """``"<accession>:c<contig_index>:<start>"`` → ``(accession, contig_index, start)``.

    The accession is recovered by :func:`tbox_finder.mining.host_order.host_accession` (the
    canonical parser, reused not forked); the contig index and window start are read from the
    ``:c<ci>:<start>`` tail :func:`tbox_finder.mining.substrate_windows.window_name` writes.
    """
    try:
        accession = host_accession(window_name)
    except HostOrderError as exc:
        raise MineRoundError(f"window id {window_name!r} did not parse: {exc}") from exc
    tail = str(window_name)[len(accession) :]
    # tail is ":c<contig_index>:<start>"
    if not tail.startswith(":c"):
        raise MineRoundError(f"window id {window_name!r} lacks a ':c<contig>:<start>' tail")
    parts = tail[2:].split(":")
    if len(parts) != 2:
        raise MineRoundError(f"window id {window_name!r} does not parse to contig+start")
    try:
        return accession, int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise MineRoundError(f"window id {window_name!r} has a non-integer contig/start") from exc


def window_candidates_to_mining(
    candidates: Sequence[Any],
    *,
    window_name: str,
) -> list[MiningCandidate]:
    """Map called loci (window-relative spans) to genome-coordinate :class:`MiningCandidate`.

    Each :class:`tbox_finder.infer.call.Candidate` carries a ``[start, end)`` span **relative
    to the scanned 1024-nt window**; genome coordinates are ``window_start + span``. Every
    resulting candidate is a false positive by construction — the a2 substrate is a
    host-order-admissible negative window masked of all known T-boxes — so it enters with the
    default all-``unavailable`` :class:`SpareRuleEvidence` (the spare rule decides its fate).

    ``MiningCandidate.accession`` is set to the **contig-scoped** id ``<accession>:c<ci>`` —
    the same contig-id namespace the homolog-DB build uses — not the bare assembly accession.
    Window offsets are per-contig (each contig is 0-based), so keying the union-prior mask on
    the assembly accession alone would collapse two distinct loci at the same offset on
    different contigs of one assembly onto the same coordinates (masking/mining keys on
    ``(accession, locus_start, locus_end)``); the contig-scoped key keeps them distinct.
    """
    accession, contig_index, window_start = parse_window_name(window_name)
    contig_id = f"{accession}:c{contig_index}"
    out: list[MiningCandidate] = []
    for cand in candidates:
        genome_start = window_start + int(cand.start)
        genome_end = window_start + int(cand.end)
        out.append(
            MiningCandidate(
                candidate_id=f"{window_name}:{genome_start}-{genome_end}",
                pool=GENOMIC_WINDOW_POOL,
                accession=contig_id,
                locus_start=genome_start,
                locus_end=genome_end,
                score=float(cand.peak_p_elem),
                evidence=SpareRuleEvidence(),
            )
        )
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Tier-2N gate — probe recall (provisional criterion) → round decision + degenerate guard
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_probe_round(
    probe_set: ProbeSet,
    recovered: set[str],
    recall_history: list[float],
    *,
    round_index: int,
) -> dict[str, Any]:
    """Compute this round's Tier-2N recall and the halt/rollback decision.

    ``recovered`` is the set of probe ids the current checkpoint recovers under the
    provisional criterion (:func:`mining_criterion.recovered_ids`). On **round 0**
    (``round_index == 0`` / empty ``recall_history``) the degenerate-criterion guard is
    enforced first: a round-0 recall of ≈0 or ≈1 makes the cross-round-constant criterion's
    value load-bearing and raises :class:`mining_criterion.ProvisionalCriterionError` (a §7
    stop). Then :func:`tbox_finder.eval.tier2n_probe.round_decision` grades the recall against
    the best round so far.
    """
    members = set(probe_set.natural) | set(probe_set.synthetic)
    recall = probe_recall(probe_set, recovered)

    degenerate_guard: dict[str, Any] | None = None
    if round_index == 0 or not recall_history:
        degenerate_guard = mining_criterion.guard_non_degenerate(members, recovered)

    decision = round_decision(probe_set, recall, recall_history)
    return {
        "round_index": int(round_index),
        "recall_this_round": recall,
        "decision": decision,
        "degenerate_guard": degenerate_guard,
        "criterion": {
            "threshold": mining_criterion.PROVISIONAL_THRESHOLD,
            "min_span": mining_criterion.PROVISIONAL_MIN_SPAN,
            "gap_merge": mining_criterion.PROVISIONAL_GAP_MERGE,
            "note": "provisional, cross-round-constant, non-binding on ADR-0005 D3 (A9 Pin 3)",
        },
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Scan primitives (ready-path; torch lazily imported) — reuse scan + call
# ═════════════════════════════════════════════════════════════════════════════
def scan_probe_variants(
    checkpoint_path: str | Path,
    variants: Sequence[Any],
    *,
    device: Any = None,
) -> set[str]:
    """Scan each Tier-2N probe variant's sequence → the set of recovered ``variant_id``.

    Reuses :func:`tbox_finder.infer.scan.load_stage1_checkpoint` + ``scan_sequence`` and the
    provisional criterion. Torch is imported lazily by ``scan``. Ready-path: run when the
    round is ready (this is downstream of the readiness refusal at P2).
    """
    from tbox_finder.infer.scan import load_stage1_checkpoint, scan_sequence

    model = load_stage1_checkpoint(checkpoint_path, device=device)
    dev = device if device is not None else next(model.parameters()).device
    scored: dict[str, tuple[Any, Any]] = {}
    for variant in variants:
        reconciled = scan_sequence(model, variant.sequence, device=dev)
        scored[str(variant.variant_id)] = (reconciled.log_probs, reconciled.zero_flanked)
    return mining_criterion.recovered_ids(scored)


def scan_substrate_windows(
    checkpoint_path: str | Path,
    windows: Sequence[tuple[str, str]],
    *,
    device: Any = None,
) -> list[MiningCandidate]:
    """Scan ``(window_name, sequence)`` a2 substrate windows → false-positive MiningCandidates.

    Reuses ``scan`` + :func:`tbox_finder.infer.call.call_candidates` under the provisional
    criterion, then :func:`window_candidates_to_mining` for the coordinate mapping. Torch is
    imported lazily. Ready-path (downstream of the P2 readiness refusal).
    """
    from tbox_finder.infer.call import call_candidates
    from tbox_finder.infer.scan import load_stage1_checkpoint, scan_sequence

    model = load_stage1_checkpoint(checkpoint_path, device=device)
    dev = device if device is not None else next(model.parameters()).device
    out: list[MiningCandidate] = []
    for window_name, seq in windows:
        reconciled = scan_sequence(model, seq, device=dev)
        candidates = call_candidates(
            reconciled.log_probs,
            reconciled.zero_flanked,
            threshold=mining_criterion.PROVISIONAL_THRESHOLD,
            min_span=mining_criterion.PROVISIONAL_MIN_SPAN,
            gap_merge=mining_criterion.PROVISIONAL_GAP_MERGE,
        )
        out.extend(window_candidates_to_mining(candidates, window_name=window_name))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# CLI — the sbatch's readiness preflight (`plan`); exits nonzero when a round is refused
# ═════════════════════════════════════════════════════════════════════════════
def _parse_bool(value: str) -> bool:
    """Strict bool parser for CLI overrides — an unrecognized value is an error, not False.

    A silent ``"treu" → False`` would quietly force a refused plan on a typo, hiding operator
    intent (the whole point of an explicit override).
    """
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _cmd_plan(args: argparse.Namespace) -> int:
    from tbox_finder.mining.covariation import backend_available

    rscape_installed = (
        args.rscape_installed if args.rscape_installed is not None else (backend_available())
    )
    plan = plan_round(
        rscape_installed=bool(rscape_installed),
        msa_supply_available=bool(args.msa_supply_available),
        relaxed_arch_available=bool(args.relaxed_arch_available),
        synteny_available=bool(args.synteny_available),
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not plan["ready"]:
        # A refused round is not an error in the crash sense — it is the honest,
        # expected P2 outcome. Exit code 3 lets the sbatch branch on it explicitly
        # (distinct from a 1/2 staging/gate failure) and skip the GPU legs.
        print(
            "mining round REFUSED at readiness — " f"{plan['readiness']['refusal_reason']}",
            file=sys.stderr,
        )
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbox_finder.mining.mine_round")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="readiness preflight (refuses when no protective backend)")
    plan.add_argument(
        "--rscape-installed",
        type=_parse_bool,
        default=None,
        help="override R-scape presence (default: probe covariation.backend_available())",
    )
    plan.add_argument(
        "--msa-supply-available",
        action="store_true",
        default=MSA_SUPPLY_AVAILABLE,
        help=(
            "declare the ADR-0006 D7/A1 per-candidate MSA supply available; absent ⇒ the "
            "module default MSA_SUPPLY_AVAILABLE, so flipping that flag at the unblock step "
            "unblocks the preflight without a CLI change"
        ),
    )
    plan.add_argument("--relaxed-arch-available", action="store_true")
    plan.add_argument("--synteny-available", action="store_true")
    plan.add_argument("--out", default=None, help="write the plan JSON here")
    plan.set_defaults(func=_cmd_plan)

    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "GENOMIC_WINDOW_POOL",
    "MSA_SUPPLY_AVAILABLE",
    "MineRoundError",
    "SCHEMA_VERSION",
    "STEP",
    "build_round_availability",
    "evaluate_probe_round",
    "main",
    "parse_window_name",
    "plan_round",
    "scan_probe_variants",
    "scan_substrate_windows",
    "window_candidates_to_mining",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
