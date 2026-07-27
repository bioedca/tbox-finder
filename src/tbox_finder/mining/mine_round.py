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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder.eval import mining_criterion
from tbox_finder.eval.tier2n_probe import ProbeSet, probe_recall, round_decision
from tbox_finder.mining.hard_negative import MiningCandidate
from tbox_finder.mining.host_order import HostOrderError, host_accession
from tbox_finder.mining.spare_rule import (
    STATUS_UNAVAILABLE,
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
        contig_index, window_start = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise MineRoundError(f"window id {window_name!r} has a non-integer contig/start") from exc
    if contig_index < 0 or window_start < 0:
        raise MineRoundError(f"window id {window_name!r} has a negative contig/start")
    return accession, contig_index, window_start


def candidate_evidence(
    candidate_id: str, covariation_status: Mapping[str, str] | None
) -> SpareRuleEvidence:
    """The candidate's :class:`SpareRuleEvidence`, given the round's covariation status table.

    Two modes, one fail-closed default:

    * ``covariation_status is None`` (the scan/collect legs, before the producer has run) → the
      default all-``unavailable`` evidence: nothing has been evaluated, so every disjunct is
      ``unavailable`` ⇒ the candidate is spared, never silently mined.
    * a status ``Mapping`` (the retrain leg, after the ADR-0005 A10 producer wrote the merged
      covariation-status table) → the candidate's ``any_helix_rscape`` disjunct is set to its
      **produced** status. A ``candidate_id`` **absent** from the map resolves to
      :data:`STATUS_UNAVAILABLE` — a dropped shard, a candidate the producer never scored, cannot
      fail *open* (it is spared, not mined). The other two disjuncts stay ``unavailable`` (no
      relaxed-architecture / synteny backend exists at P2); :class:`SpareRuleEvidence` validates
      the status string, so a corrupt value raises rather than reading as "not passed".
    """
    if covariation_status is None:
        return SpareRuleEvidence()
    return SpareRuleEvidence(
        any_helix_rscape=str(covariation_status.get(candidate_id, STATUS_UNAVAILABLE))
    )


def window_candidates_to_mining(
    candidates: Sequence[Any],
    *,
    window_name: str,
    covariation_status: Mapping[str, str] | None = None,
) -> list[MiningCandidate]:
    """Map called loci (window-relative spans) to genome-coordinate :class:`MiningCandidate`.

    Each :class:`tbox_finder.infer.call.Candidate` carries a ``[start, end)`` span **relative
    to the scanned 1024-nt window**; genome coordinates are ``window_start + span``. Every
    resulting candidate is a false positive by construction — the a2 substrate is a
    host-order-admissible negative window masked of all known T-boxes.

    ``covariation_status`` is the wiring the ADR-0005 A10 producer step adds: at scan/collect time
    it is ``None`` and every candidate carries the default all-``unavailable`` evidence (the spare
    rule then spares them all); at retrain time the round's merged covariation-status table
    (``candidate_id → status``, from
    :func:`tbox_finder.mining.covariation_producer.merge_status_tables`) is passed in and each
    candidate's ``any_helix_rscape`` disjunct carries its **real** produced status — see
    :func:`candidate_evidence` for the fail-closed absent-id rule.

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
        candidate_id = f"{window_name}:{genome_start}-{genome_end}"
        out.append(
            MiningCandidate(
                candidate_id=candidate_id,
                pool=GENOMIC_WINDOW_POOL,
                accession=contig_id,
                locus_start=genome_start,
                locus_end=genome_end,
                score=float(cand.peak_p_elem),
                evidence=candidate_evidence(candidate_id, covariation_status),
            )
        )
    return out


# ═════════════════════════════════════════════════════════════════════════════
# FP-manifest I/O + the leg-(d) spare-rule application (covariation status → mined/spared)
# ═════════════════════════════════════════════════════════════════════════════
def write_fp_manifest(candidates: Sequence[MiningCandidate], path: str | Path) -> Path:
    """Persist collected false-positive :class:`MiningCandidate` s (leg (b) → legs (c)/(d) handoff).

    Written in the :mod:`tbox_finder.mining.covariation_producer` manifest shape — a ``candidates``
    list keyed by ``candidate_id``/``accession``/``locus_start``/``locus_end`` — so the producer
    (leg (c)) reads it with :func:`~tbox_finder.mining.covariation_producer.read_candidate_manifest`
    directly; ``score``/``pool`` ride along as extra keys the producer ignores and the retrain leg
    (:func:`read_fp_manifest`) reads back to reconstitute the candidate.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "n_candidates": len(candidates),
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "accession": c.accession,
                "locus_start": int(c.locus_start),
                "locus_end": int(c.locus_end),
                "score": float(c.score),
                "pool": c.pool,
            }
            for c in candidates
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Path(path)


def read_fp_manifest(
    path: str | Path, *, covariation_status: Mapping[str, str] | None = None
) -> list[MiningCandidate]:
    """Reconstitute the leg-(b) false positives, stamping each with its produced covariation status.

    The retrain leg passes the round's merged covariation-status table as ``covariation_status``;
    :func:`candidate_evidence` resolves an id absent from that map to ``unavailable`` (fail-closed).
    With ``covariation_status=None`` this round-trips the manifest with default all-``unavailable``
    evidence (what the collect leg wrote).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["candidates"] if isinstance(payload, dict) else payload
    out: list[MiningCandidate] = []
    for r in rows:
        cid = str(r["candidate_id"])
        out.append(
            MiningCandidate(
                candidate_id=cid,
                pool=str(r.get("pool", GENOMIC_WINDOW_POOL)),
                accession=str(r["accession"]),
                locus_start=int(r["locus_start"]),
                locus_end=int(r["locus_end"]),
                score=float(r.get("score", 0.0)),
                evidence=candidate_evidence(cid, covariation_status),
            )
        )
    return out


def load_admissible_accessions(
    host_order_table: str | Path | None = None,
    split_table: str | Path | None = None,
) -> list[str]:
    """The a2 host-order-admissible genome accessions (leg-(a) scan substrate), sorted.

    Reuses :func:`tbox_finder.mining.host_order.load_host_folds` (the same a2 rule the §8.2 CI
    reads), keeping only the ``admitted`` hosts — the 660-genome negative substrate. Defaults are
    the module's committed-table defaults; pandas is imported lazily by ``load_host_folds``.
    """
    from tbox_finder.mining.host_order import load_host_folds

    kwargs: dict[str, Any] = {}
    if host_order_table is not None:
        kwargs["host_order_table"] = host_order_table
    if split_table is not None:
        kwargs["split_table"] = split_table
    verdicts, _heldout = load_host_folds(**kwargs)
    return sorted(acc for acc, (admissible, _reason) in verdicts.items() if admissible)


def apply_spare_rule(
    fp_manifest: str | Path,
    status_table: str | Path,
    *,
    rscape_installed: bool,
    msa_supply_available: bool,
    relaxed_arch_available: bool = False,
    synteny_available: bool = False,
    union_prior: str | Path = "data/processed/priors/union_prior.parquet",
    corpus_parquet: str | Path = "data/processed/master_clean_v0.parquet",
) -> dict[str, Any]:
    """Leg (d): apply the produced covariation status to the round's FPs → the mining outcome.

    Reloads the false positives with their merged covariation status
    (:func:`read_fp_manifest` + :func:`~tbox_finder.mining.covariation_producer.load_status_map`),
    builds the union-prior + own-positives mask exactly as
    :mod:`tbox_finder.mining.pool` does, and delegates to
    :func:`tbox_finder.mining.hard_negative.mine_round` under this round's backend-availability map.
    ``mine_round`` refuses outright if no protective backend is available and raises if a candidate
    carries evidence for an unavailable backend — so the availability declared here and the produced
    status must agree. Returns the round report (``mined_ids``/``spared_ids``/``per_pool``); the
    mined ids are the hard negatives the DDP retrain then trains against.
    """
    from tbox_finder import masking
    from tbox_finder.mining.covariation_producer import load_status_map
    from tbox_finder.mining.hard_negative import mine_round as run_mine_round

    status_map = load_status_map(status_table)
    candidates = read_fp_manifest(fp_manifest, covariation_status=status_map)
    availability = build_round_availability(
        rscape_installed=rscape_installed,
        msa_supply_available=msa_supply_available,
        relaxed_arch_available=relaxed_arch_available,
        synteny_available=synteny_available,
    )
    union_loci, _n_union, _n_dropped = masking.load_union_loci(union_prior)
    own_loci = masking.load_own_positive_loci(corpus_parquet)
    mask = masking.LocusIndex.from_records(list(union_loci) + list(own_loci))
    report = run_mine_round(candidates, mask, availability)
    report["schema_version"] = SCHEMA_VERSION
    report["step"] = STEP
    report["status_counts"] = _status_counts(status_map)
    return report


def _status_counts(status_map: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in status_map.values():
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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


def _scan_windows(
    model: Any,
    windows: Any,
    dev: Any,
    *,
    covariation_status: Mapping[str, str] | None = None,
) -> list[MiningCandidate]:
    """Scan an iterable of ``(window_name, sequence)`` with a **preloaded** model → candidates.

    Shared by :func:`scan_substrate_windows` (one window list) and :func:`scan_admissible_genomes`
    (a genome slice, tiled on the fly) so the checkpoint is loaded once, not once per genome. Reuses
    ``scan_sequence`` + the D3 operator ``call_candidates`` under the provisional criterion.
    """
    from tbox_finder.infer.call import call_candidates
    from tbox_finder.infer.scan import scan_sequence

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
        out.extend(
            window_candidates_to_mining(
                candidates, window_name=window_name, covariation_status=covariation_status
            )
        )
    return out


def scan_substrate_windows(
    checkpoint_path: str | Path,
    windows: Sequence[tuple[str, str]],
    *,
    device: Any = None,
    covariation_status: Mapping[str, str] | None = None,
) -> list[MiningCandidate]:
    """Scan ``(window_name, sequence)`` a2 substrate windows → false-positive MiningCandidates.

    Reuses ``scan`` + :func:`tbox_finder.infer.call.call_candidates` under the provisional
    criterion, then :func:`window_candidates_to_mining` for the coordinate mapping. Torch is
    imported lazily. Ready-path (downstream of the P2 readiness refusal).
    """
    from tbox_finder.infer.scan import load_stage1_checkpoint

    model = load_stage1_checkpoint(checkpoint_path, device=device)
    dev = device if device is not None else next(model.parameters()).device
    return _scan_windows(model, windows, dev, covariation_status=covariation_status)


def scan_admissible_genomes(
    checkpoint_path: str | Path,
    accessions: Sequence[str],
    *,
    genome_dir: str | Path = "data/interim/production_genomes",
    device: Any = None,
) -> list[MiningCandidate]:
    """Tile + scan a slice of the a2 host-order-admissible genomes → false-positive candidates.

    Leg (a) of a mining round. The 660 admissible accessions (:func:`load_admissible_accessions`)
    are sharded across GPUs by the sbatch; each shard calls this with its accession slice. Reuses
    :func:`tbox_finder.mining.substrate_windows.iter_genome_windows` (canonical 1024/512 tiling) so
    the scanned geometry is identical to the shard-spec emitter's, and streams genome-by-genome so a
    slice holds one genome's windows at a time. The checkpoint is loaded once for the whole slice.
    """
    from tbox_finder.infer.scan import load_stage1_checkpoint
    from tbox_finder.mining.substrate_windows import iter_genome_windows, read_genome_fasta

    model = load_stage1_checkpoint(checkpoint_path, device=device)
    dev = device if device is not None else next(model.parameters()).device
    out: list[MiningCandidate] = []
    for acc in accessions:
        windows = iter_genome_windows(str(acc), read_genome_fasta(genome_dir, str(acc)))
        out.extend(_scan_windows(model, windows, dev))
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


def _cmd_admissible(args: argparse.Namespace) -> int:
    accessions = load_admissible_accessions(
        host_order_table=args.host_order_table, split_table=args.split_table
    )
    Path(args.out).write_text("\n".join(accessions) + "\n", encoding="utf-8")
    print(f"admissible-accessions: {len(accessions)} a2 hosts → {args.out}")
    return 0


def _cmd_scan_shard(args: argparse.Namespace) -> int:
    accessions = [
        line.strip()
        for line in Path(args.accessions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        accessions = accessions[: args.limit]
    candidates = scan_admissible_genomes(
        args.checkpoint, accessions, genome_dir=args.genome_dir, device=args.device
    )
    write_fp_manifest(candidates, args.out_manifest)
    print(
        f"scan-shard: {len(accessions)} genomes → {len(candidates)} false-positive "
        f"candidates → {args.out_manifest}"
    )
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    """Merge per-GPU/per-slice FP manifests into the round's single candidate manifest (leg b)."""
    merged: list[MiningCandidate] = []
    seen: set[str] = set()
    for path in args.manifests:
        for cand in read_fp_manifest(path):
            if cand.candidate_id in seen:
                raise MineRoundError(
                    f"candidate_id {cand.candidate_id!r} appears in more than one scan-shard "
                    "manifest — genome slices must be disjoint (a re-used slice or double-count)"
                )
            seen.add(cand.candidate_id)
            merged.append(cand)
    write_fp_manifest(merged, args.out)
    print(f"collect: {len(args.manifests)} manifests → {len(merged)} candidates → {args.out}")
    return 0


def _cmd_apply_spare_rule(args: argparse.Namespace) -> int:
    from tbox_finder.mining.covariation import backend_available

    rscape_installed = (
        args.rscape_installed if args.rscape_installed is not None else backend_available()
    )
    report = apply_spare_rule(
        args.manifest,
        args.status_table,
        rscape_installed=bool(rscape_installed),
        msa_supply_available=bool(args.msa_supply_available),
        relaxed_arch_available=bool(args.relaxed_arch_available),
        synteny_available=bool(args.synteny_available),
        union_prior=args.union_prior,
        corpus_parquet=args.corpus,
    )
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"apply-spare-rule: n_candidates={report['n_candidates']} "
        f"mined={report['n_mined']} spared={report['n_spared']} "
        f"masked={report['n_masked']} refused={report['n_refused_no_coordinates']} → {args.out}"
    )
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

    adm = sub.add_parser("admissible-accessions", help="write the a2 admissible host accessions")
    adm.add_argument("--out", required=True)
    adm.add_argument("--host-order-table", default=None)
    adm.add_argument("--split-table", default=None)
    adm.set_defaults(func=_cmd_admissible)

    scan = sub.add_parser("scan-shard", help="LEG (a): scan a genome slice → FP candidate manifest")
    scan.add_argument("--checkpoint", required=True)
    scan.add_argument("--accessions", required=True, help="one accession per line (a scan slice)")
    scan.add_argument("--out-manifest", required=True)
    scan.add_argument("--genome-dir", default="data/interim/production_genomes")
    scan.add_argument("--device", default=None, help="torch device (default: the model's device)")
    scan.add_argument(
        "--limit", type=int, default=None, help="scan only the first N (sizing smoke)"
    )
    scan.set_defaults(func=_cmd_scan_shard)

    coll = sub.add_parser("collect", help="LEG (b): merge scan-shard FP manifests → one manifest")
    coll.add_argument("--manifests", nargs="+", required=True)
    coll.add_argument("--out", required=True)
    coll.set_defaults(func=_cmd_collect)

    apply = sub.add_parser("apply-spare-rule", help="LEG (d): status table → mined/spared outcome")
    apply.add_argument("--manifest", required=True, help="the round FP manifest (leg b)")
    apply.add_argument("--status-table", required=True, help="the merged covariation status table")
    apply.add_argument("--out", required=True)
    apply.add_argument("--rscape-installed", type=_parse_bool, default=None)
    apply.add_argument("--msa-supply-available", action="store_true", default=MSA_SUPPLY_AVAILABLE)
    apply.add_argument("--relaxed-arch-available", action="store_true")
    apply.add_argument("--synteny-available", action="store_true")
    apply.add_argument("--union-prior", default="data/processed/priors/union_prior.parquet")
    apply.add_argument("--corpus", default="data/processed/master_clean_v0.parquet")
    apply.set_defaults(func=_cmd_apply_spare_rule)

    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "GENOMIC_WINDOW_POOL",
    "MSA_SUPPLY_AVAILABLE",
    "MineRoundError",
    "SCHEMA_VERSION",
    "STEP",
    "apply_spare_rule",
    "build_round_availability",
    "candidate_evidence",
    "evaluate_probe_round",
    "load_admissible_accessions",
    "main",
    "parse_window_name",
    "plan_round",
    "read_fp_manifest",
    "scan_admissible_genomes",
    "scan_probe_variants",
    "scan_substrate_windows",
    "window_candidates_to_mining",
    "write_fp_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
