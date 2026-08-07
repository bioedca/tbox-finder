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

Where the supply actually stands (P3-15′-a, 2026-08-05)
------------------------------------------------------
The covariation-(a) supply **exists**, so :data:`MSA_SUPPLY_AVAILABLE` is ``True``: the
D7/A1 homolog-search target DB is built and DVC-versioned (SLURM job 741); the per-candidate
search → mlocarna alignment → R-scape leg is :mod:`tbox_finder.mining.covariation_producer`
and is certified covariation-*producible* — not merely R-scape-installed — by a must-fire
matched control on a real divergent class-II T-box (job 766; ADR-0002's certification gate,
ADR-0006 A3); and :func:`apply_spare_rule` routes the produced status onto each candidate's
evidence. Readiness therefore passes on the protective (a)-backend alone.

That does **not** mean a round can mine anything. Sparing is a disjunction, so mining is a
conjunction: relaxed-architecture-(b) and synteny-(c) still have no backend, so
:func:`tbox_finder.mining.remine.max_possible_yield` bounds every round's yield at 0 and
refuses. The two gates answering differently — ``ready`` ``True``, ``yield_producible``
``False`` — is the **correct** state until (b) and (c) land, not an inconsistency.

The constant is a *declaration*, so it is checked rather than trusted:
:func:`derive_msa_supply_available` re-derives the same fact from the shipped evidence and
:func:`plan_round` publishes the derivation beside the claim, so the constant cannot drift
from reality in either direction — a checkout missing the DB pointer or the certification
makes the CLI **refuse** (exit 4), and a constant left ``False`` while the supply exists
fails the unit pin.

``numpy``-tier (via :mod:`tbox_finder.eval.mining_criterion` → ``infer.call``) but
**torch-free at import**: the checkpoint scan imports torch lazily inside the two scan
functions, so the readiness/adapter/decision surface unit-tests without a GPU. PRD §9.1;
ADR-0005 D3 + D14 + A9; ADR-0006 D9 / D11 + A2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
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
#: CM-free de-novo alignment) exists. It does, since **P3-15′-a**: target DB built and
#: versioned (job 741), producer certified covariation-producible (job 766), producer wired
#: into the round. Kept as an explicit constant rather than a filesystem probe at import so
#: it stays greppable, import-cheap and overridable per run — and kept **honest** by
#: :func:`derive_msa_supply_available`, which re-derives the same fact from the shipped
#: evidence and is what the unit pin and the CLI preflight compare it against.
MSA_SUPPLY_AVAILABLE = True

#: The **criterion-(c)** analogue, flipped by P3-15′-c-ii: the 339 md5-verified NCBI GFFs are
#: on disk (P3-15′-c-i), ``mining/synteny.py`` implements ADR-0006 D4's predicate and
#: ``mining/synteny_producer.py`` writes the per-candidate ``candidate_id → status`` table the
#: round reads.  Kept honest the same way ``MSA_SUPPLY_AVAILABLE`` is — by
#: :func:`tbox_finder.mining.synteny_producer.derive_synteny_supply_available`, which
#: re-derives the fact from disk and is what the unit pin and the CLI preflight compare it
#: against, so the constant cannot drift from reality in **either** direction.
#:
#: ⚠ **This module's own P2 CLI does not read it, and that asymmetry is deliberate.** The P3
#: round (``remine``) carries the two-way ``--synteny-available`` / ``--no-synteny-available``
#: pair defaulted from this constant plus the exit-4 unevidenced preflight; the P2 CLI keeps
#: its bare ``store_true``, so a P2 round declares the backend only when explicitly asked.
#: The drift is one-directional — P2 can under-declare (refuse / spare more) but never
#: over-declare — which is the same fail-closed shape recorded for
#: ``slurm/p3/stage1_remine.sbatch``'s ``MSA_SUPPLY_AVAILABLE`` env var at P3-15′-a.
SYNTENY_SUPPLY_AVAILABLE = True

#: The git-tracked evidence :func:`derive_msa_supply_available` re-derives the supply from.
#: All three ride any checkout (including the cluster's and CI's) — none is DVC- or
#: LFS-shaped — so the derivation answers the same in every environment.
HOMOLOG_DB_DVC = "data/interim/homolog_db.dvc"
HOMOLOG_MSA_PROVENANCE = "data/interim/homolog_msa/provenance.json"
CERTIFIED_MSA = "data/interim/homolog_msa/certified_positive.sto"

#: ``src/tbox_finder/mining/mine_round.py`` → the repo root (the project runs from a checkout
#: with ``PYTHONPATH=src`` or an editable install; both keep this file inside the tree).
REPO_ROOT = Path(__file__).resolve().parents[3]


class MineRoundError(ValueError):
    """Raised on malformed round-driver input (bad window id, empty probe set)."""


# ═════════════════════════════════════════════════════════════════════════════
# The validator half of MSA_SUPPLY_AVAILABLE — re-derive the supply from evidence
# ═════════════════════════════════════════════════════════════════════════════
#: The DVC directory-pointer shape (``dvc add`` writes exactly this). Read with strict
#: regexes rather than a YAML parse: no ``envs/*.yml`` puts ``pyyaml`` in the CI tier, and a
#: pointer that does not match this shape exactly must fail **closed** rather than be
#: interpreted generously — the whole point of the clause is that a missing/rewritten DB
#: cannot read as a present one.
_DVC_MD5 = re.compile(r"^\s*-\s*md5:\s*([0-9a-f]{32}\.dir)\s*$", re.MULTILINE)
_DVC_PATH = re.compile(r"^\s*path:\s*(\S+)\s*$", re.MULTILINE)
_DVC_NFILES = re.compile(r"^\s*nfiles:\s*(\d+)\s*$", re.MULTILINE)
_DVC_SIZE = re.compile(r"^\s*size:\s*(\d+)\s*$", re.MULTILINE)

#: The producer entry points a round's covariation leg actually calls — the per-candidate
#: homolog search, the CM-free de-novo alignment, the R-scape scoring, and the two the round
#: driver itself imports (:func:`apply_spare_rule`). Checked by attribute on the imported
#: module, so a producer reduced to a stub file cannot satisfy the clause.
PRODUCER_ENTRY_POINTS = (
    "search_shard",
    "align_shard",
    "score_shard",
    "merge_status_tables",
    "load_status_map",
)

#: The matched-control dimensions the certification must report as matched, **named** rather
#: than iterated off whatever keys the record happens to carry: a clause read from the
#: evidence's own key set is vacuously satisfied exactly when the evidence is missing, so an
#: emptied ``matched_control`` would certify "all matched" by having nothing to be unmatched.
REQUIRED_MATCHED_CONTROL_FLAGS = (
    "composition_matched",
    "depth_matched",
    "ss_cons_matched",
    "width_matched",
)


def read_dvc_dir_pointer(path: str | Path) -> dict[str, Any] | None:
    """A DVC **directory** pointer → ``{md5, path, nfiles, size}``, or ``None`` if malformed.

    ``None`` on anything unexpected — unreadable file, no/multiple ``md5`` lines, a
    non-``.dir`` hash, a missing field. Callers treat ``None`` as "not versioned".
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        # UnicodeDecodeError as well as OSError (CodeRabbit CLI r3, both reproduced by
        # execution): a pointer carrying non-UTF-8 bytes is malformed, not a crash.
        return None
    md5s = _DVC_MD5.findall(text)
    paths = _DVC_PATH.findall(text)
    nfiles = _DVC_NFILES.findall(text)
    sizes = _DVC_SIZE.findall(text)
    if not (len(md5s) == len(paths) == len(nfiles) == len(sizes) == 1):
        return None
    try:
        # `\d+` guarantees digits, not that ``int`` accepts them: CPython caps str→int at
        # 4300 digits and raises ValueError past it (measured). Malformed ⇒ ``None``.
        n_files, size = int(nfiles[0]), int(sizes[0])
    except ValueError:
        return None
    return {
        "md5": md5s[0],
        "path": paths[0],
        "nfiles": n_files,
        "size": size,
    }


def _read_json_or_none(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha256(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def derive_msa_supply_available(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Re-derive, from the shipped evidence, whether the covariation-(a) MSA supply exists.

    :data:`MSA_SUPPLY_AVAILABLE` is a hand-written declaration; this is the independent
    re-derivation the declaration is checked against, so the constant cannot drift from
    reality in **either** direction (a stale ``False`` fails the unit pin; a ``True`` on a
    checkout that cannot evidence it refuses the CLI preflight). It is deliberately *not*
    what :func:`build_round_availability` reads — a run must be able to declare the supply
    unavailable on a machine that has not staged it, which is a strictly narrower claim.

    Six clauses, every one fail-closed on a missing or malformed artifact, and each
    independently breakable (that is what makes ``all(...)`` testable at all):

    ``target_db_versioned``
        ``data/interim/homolog_db.dvc`` is a well-formed non-empty directory pointer — the
        D7/A1 homolog-search target DB is built and version-pinned (job 741).
    ``producer_present``
        :mod:`tbox_finder.mining.covariation_producer` imports and exposes every entry in
        :data:`PRODUCER_ENTRY_POINTS` — the per-candidate search → align → score chain.
    ``producer_status_wired``
        :func:`candidate_evidence` actually **stamps** a produced status onto the candidate's
        ``any_helix_rscape`` instead of defaulting it to ``unavailable``. Measured by calling
        it, not by reading a signature: a producer that ships but whose output the round drops
        on the floor is the no-op this whole gate exists to refuse.
    ``certification_green``
        the job-766 provenance records the must-fire matched control passing *as a control*:
        the real class-II seed ``passed``, its column-shuffled twin ``failed`` (an
        ``unavailable`` twin is "no power", which certifies nothing), the alignment
        composition/depth/width/SS_cons matched, and the depth at or above the ADR-0006 A2
        Pin 2 floor read from :data:`~tbox_finder.power.MIN_REAL_HOMOLOG_N` rather than
        re-typed.
    ``certification_matches_versioned_db``
        that certification was produced against the **same** DB version the pointer names
        today — a DB rebuilt after certification would otherwise inherit a green it never
        earned.
    ``certified_msa_intact``
        the committed certified MSA still hashes to what the run recorded.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    clauses: dict[str, bool] = {}
    reasons: list[str] = []

    def _fail(clause: str, why: str) -> bool:
        reasons.append(f"{clause}: {why}")
        return False

    # ── clause 1: the target DB is built and version-pinned ──────────────────
    pointer = read_dvc_dir_pointer(root / HOMOLOG_DB_DVC)
    if pointer is None:
        clauses["target_db_versioned"] = _fail(
            "target_db_versioned", f"{HOMOLOG_DB_DVC} is missing or not a DVC directory pointer"
        )
    elif pointer["path"] != "homolog_db":
        clauses["target_db_versioned"] = _fail(
            "target_db_versioned", f"pointer names {pointer['path']!r}, expected 'homolog_db'"
        )
    elif pointer["nfiles"] < 1 or pointer["size"] < 1:
        clauses["target_db_versioned"] = _fail(
            "target_db_versioned",
            f"pointer is empty (nfiles={pointer['nfiles']}, size={pointer['size']})",
        )
    else:
        clauses["target_db_versioned"] = True

    # ── clause 2: the producer ships ─────────────────────────────────────────
    try:
        from tbox_finder.mining import covariation_producer
    except Exception as exc:  # noqa: BLE001 - a broken producer is a FAILED clause, not a crash
        # Deliberately broader than ImportError (CodeRabbit CLI r2, reproduced by execution):
        # a module-level failure anywhere in the producer or its transitive imports —
        # RuntimeError, OSError, AttributeError — would otherwise propagate out of a function
        # documented as fail-closed on every clause, aborting `plan` with a traceback instead
        # of the exit-4 refusal. A producer that cannot even import is exactly the state this
        # clause exists to report.
        covariation_producer = None
        clauses["producer_present"] = _fail("producer_present", f"import failed: {exc!r}")
    if covariation_producer is not None:
        missing = [e for e in PRODUCER_ENTRY_POINTS if not hasattr(covariation_producer, e)]
        clauses["producer_present"] = (
            True
            if not missing
            else _fail("producer_present", f"covariation_producer lacks {missing}")
        )

    # ── clause 3: the round stamps what the producer produced ────────────────
    from tbox_finder.mining.spare_rule import STATUS_PASSED

    probe_id = "__msa_supply_probe__"
    stamped = candidate_evidence(probe_id, {probe_id: STATUS_PASSED}).any_helix_rscape
    clauses["producer_status_wired"] = (
        True
        if stamped == STATUS_PASSED
        else _fail(
            "producer_status_wired",
            f"a produced {STATUS_PASSED!r} status reached the candidate as {stamped!r} — "
            "the producer's output is not composed into the round",
        )
    )

    # ── clauses 4-5: the must-fire certification, and that it names this DB ──
    provenance = _read_json_or_none(root / HOMOLOG_MSA_PROVENANCE)
    extra = provenance.get("extra", {}) if isinstance(provenance, Mapping) else {}
    if not isinstance(extra, Mapping) or not extra:
        clauses["certification_green"] = _fail(
            "certification_green", f"{HOMOLOG_MSA_PROVENANCE} is missing or malformed"
        )
        clauses["certification_matches_versioned_db"] = _fail(
            "certification_matches_versioned_db", f"{HOMOLOG_MSA_PROVENANCE} carries no inputs"
        )
    else:
        from tbox_finder.power import MIN_REAL_HOMOLOG_N

        control = extra.get("matched_control")
        control = control if isinstance(control, Mapping) else {}
        floor = extra.get("min_sequences_floor")
        depth = extra.get("n_records")
        failures: list[str] = []
        if extra.get("positive_status") != "passed":
            failures.append(f"positive_status={extra.get('positive_status')!r} (want 'passed')")
        if extra.get("shuffled_status") != "failed":
            failures.append(
                f"shuffled_status={extra.get('shuffled_status')!r} (want 'failed' — an "
                "'unavailable' twin is no power, which certifies nothing)"
            )
        unmatched = [k for k in REQUIRED_MATCHED_CONTROL_FLAGS if not control.get(k)]
        if unmatched:
            failures.append(f"matched_control unmet (missing counts as unmatched): {unmatched}")
        if not isinstance(floor, int) or floor < MIN_REAL_HOMOLOG_N:
            failures.append(f"min_sequences_floor={floor!r} below MIN_REAL_HOMOLOG_N")
        if not isinstance(depth, int) or not isinstance(floor, int) or depth < floor:
            failures.append(f"certified depth n_records={depth!r} below the floor {floor!r}")
        clauses["certification_green"] = (
            True if not failures else _fail("certification_green", "; ".join(failures))
        )

        inputs = provenance.get("inputs") if isinstance(provenance, Mapping) else None
        inputs = inputs if isinstance(inputs, Mapping) else {}
        certified_against = next(
            (v for k, v in inputs.items() if str(k).startswith("data/interim/homolog_db")), None
        )
        if pointer is None:
            clauses["certification_matches_versioned_db"] = _fail(
                "certification_matches_versioned_db", "no readable DB pointer to compare against"
            )
        elif certified_against != pointer["md5"]:
            clauses["certification_matches_versioned_db"] = _fail(
                "certification_matches_versioned_db",
                f"certified against {certified_against!r}, pointer names {pointer['md5']!r} — "
                "the target DB changed after certification",
            )
        else:
            clauses["certification_matches_versioned_db"] = True

    # ── clause 6: the certified MSA is the one that was certified ────────────
    outputs = provenance.get("outputs") if isinstance(provenance, Mapping) else None
    outputs = outputs if isinstance(outputs, Mapping) else {}
    recorded = outputs.get(CERTIFIED_MSA)
    observed = _sha256(root / CERTIFIED_MSA)
    if observed is None:
        clauses["certified_msa_intact"] = _fail("certified_msa_intact", f"{CERTIFIED_MSA} missing")
    elif not recorded or recorded != observed:
        clauses["certified_msa_intact"] = _fail(
            "certified_msa_intact", f"sha256 {observed} != recorded {recorded!r}"
        )
    else:
        clauses["certified_msa_intact"] = True

    return {
        "available": all(clauses.values()),
        "clauses": clauses,
        "reasons": reasons,
        "repo_root": str(root),
    }


def msa_supply_declaration_unevidenced(plan: Mapping[str, Any]) -> bool:
    """Does this plan declare a supply its own checkout cannot evidence?

    The asymmetry is deliberate. Declaring the supply **unavailable** while the evidence
    says otherwise is a conservative under-claim (the round refuses; nothing is mined on a
    machine that may not have staged the DB). Declaring it **available** without the
    evidence is the fail-open direction — it is what would let a round pass readiness,
    spend the GPU legs and mine on a covariation leg that cannot run.
    """
    derivation = plan.get("msa_supply_derivation") or {}
    return bool(plan.get("msa_supply_available")) and not bool(derivation.get("available"))


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
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Decide whether a round may run, and record why not — before any GPU time.

    Returns ``{availability, readiness, ready, msa_supply_derivation, ...}``. Since P3-15′-a
    ``ready`` is ``True`` on the covariation-(a) backend alone; the driver's caller
    (``mine_round.sbatch``) exits on it rather than scanning 3.63M windows only to have
    :func:`hard_negative.mine_round` refuse. **Readiness is not permission to mine** — see
    :func:`tbox_finder.mining.remine.max_possible_yield`, which still bounds the round's
    yield at 0 while (b) and (c) have no backend.

    ``msa_supply_available`` defaults to the module flag :data:`MSA_SUPPLY_AVAILABLE`, resolved
    via a ``None`` sentinel at **call** time (not bound at import) so that flipping the flag at
    the unblock step reaches every caller, including direct ones.

    The plan carries :func:`derive_msa_supply_available`'s clause-by-clause re-derivation
    beside the declaration, so the artifact a round is verified from records the *evidence*
    for its availability claim rather than only the claim.
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
        "msa_supply_derivation": derive_msa_supply_available(repo_root=repo_root),
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
    candidate_id: str,
    covariation_status: Mapping[str, str] | None,
    synteny_status: Mapping[str, str] | None = None,
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
      fail *open* (it is spared, not mined).

    ``synteny_status`` is the criterion-(c) analogue P3-15′-c-ii added
    (:func:`tbox_finder.mining.synteny_producer.load_status_map`), read under the **same**
    fail-closed absent-id rule.  ``relaxed_architecture`` is the one disjunct still without a
    backend (P3-15′-d), so it stays ``unavailable``.  :class:`SpareRuleEvidence` validates
    every status string, so a corrupt value raises rather than reading as "not passed".
    """
    return SpareRuleEvidence(
        any_helix_rscape=(
            STATUS_UNAVAILABLE
            if covariation_status is None
            else str(covariation_status.get(candidate_id, STATUS_UNAVAILABLE))
        ),
        downstream_aaRS_synteny=(
            STATUS_UNAVAILABLE
            if synteny_status is None
            else str(synteny_status.get(candidate_id, STATUS_UNAVAILABLE))
        ),
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
    path: str | Path,
    *,
    covariation_status: Mapping[str, str] | None = None,
    synteny_status: Mapping[str, str] | None = None,
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
                evidence=candidate_evidence(cid, covariation_status, synteny_status),
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


def load_union_mask(
    *,
    union_prior: str | Path = "data/processed/priors/union_prior.parquet",
    corpus_parquet: str | Path = "data/processed/master_clean_v0.parquet",
) -> Any:
    """The PRD §9.1 mining mask: the full union prior **plus** the run's own positives.

    Promoted out of :func:`apply_spare_rule` at P3-15 so the P2 round and the P3
    re-mining round build the mask with one arithmetic. Two copies of a masking rule
    are free to drift, and a drift here puts a known T-box into the negative pool.
    """
    from tbox_finder import masking

    union_loci, _n_union, _n_dropped = masking.load_union_loci(union_prior)
    own_loci = masking.load_own_positive_loci(corpus_parquet)
    return masking.LocusIndex.from_records(list(union_loci) + list(own_loci))


def apply_spare_rule(
    fp_manifest: str | Path,
    status_table: str | Path,
    *,
    rscape_installed: bool,
    msa_supply_available: bool,
    synteny_status_table: str | Path | None = None,
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
    from tbox_finder.mining.covariation_producer import load_status_map
    from tbox_finder.mining.hard_negative import mine_round as run_mine_round
    from tbox_finder.mining.synteny_producer import ProducerError as SyntenyProducerError
    from tbox_finder.mining.synteny_producer import load_status_map as load_synteny_status_map

    # Declaring the (c) backend available with no status table is the SILENT form of a
    # refusal: every candidate's synteny disjunct reads ``unavailable``, all are spared, and
    # the round reports a zero yield indistinguishable from an honest one.
    if synteny_available and synteny_status_table is None:
        raise ValueError(
            "synteny_available=True but no synteny status table was supplied; the (c) "
            "disjunct would read 'unavailable' for every candidate and spare them all"
        )
    if not synteny_available and synteny_status_table is not None:
        raise ValueError(
            "a synteny status table was supplied but synteny_available=False; "
            "hard_negative.mine_round refuses produced evidence for an undeclared backend"
        )

    try:
        synteny_status_map = (
            load_synteny_status_map(synteny_status_table)
            if synteny_status_table is not None
            else None
        )
    except (
        SyntenyProducerError,
        OSError,
        ValueError,
        AttributeError,
        KeyError,
        TypeError,
    ) as exc:
        # A data fault this round must REPORT: the caller branches on the return code, and an
        # unhandled ProducerError bypasses that path entirely.
        raise ValueError(f"synteny status table unusable: {exc}") from exc

    status_map = load_status_map(status_table)
    candidates = read_fp_manifest(
        fp_manifest, covariation_status=status_map, synteny_status=synteny_status_map
    )
    availability = build_round_availability(
        rscape_installed=rscape_installed,
        msa_supply_available=msa_supply_available,
        relaxed_arch_available=relaxed_arch_available,
        synteny_available=synteny_available,
    )
    mask = load_union_mask(union_prior=union_prior, corpus_parquet=corpus_parquet)
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
# Scan throughput instrumentation — persistent, timeout-surviving win/s + FP-rate
# ═════════════════════════════════════════════════════════════════════════════
def _safe_ratio(numerator: float, denominator: float | None) -> float | None:
    """``numerator / denominator``, or ``None`` when the denominator is 0 / unmeasured.

    The throughput ratios (windows/s, FPs/window) are ``None`` — not ``0`` or ``inf`` — before any
    window or wall time has accrued, so an early snapshot reads as "not yet measured" rather than a
    fabricated rate (a first flush at wall ``0`` would otherwise divide by zero).
    """
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


class ScanThroughputLog:
    """A persistent, timeout-surviving accountant for the LEG-(a) genome scan.

    Accumulates windows scanned, false positives found, and wall time so a snapshot yields the two
    numbers job 793's 6 h TIMEOUT could not measure: **win/s/GPU** (``windows_per_s``, the ADR-0005
    A9 throughput the whole-node job over-estimated ~3×) and the **partial N₀ rate**
    (``candidates_per_window``, FPs/window). It is written incrementally to a **persistent**
    ``reports/`` path (never node-local ``/tmp``) so a wall-clock kill leaves the last snapshot on
    disk — the instrumentation gap the TIMEOUT exposed (job 793 auto-cleaned its ``/tmp`` evidence
    on SIGTERM). Pure: the clock is injected by the caller, so the accounting is deterministic
    under test.
    """

    def __init__(self, *, step: str = STEP, schema_version: str = SCHEMA_VERSION) -> None:
        self.step = step
        self.schema_version = schema_version
        self.windows_scanned = 0
        self.candidates_found = 0
        self.genomes_completed = 0
        self.per_genome: list[dict[str, Any]] = []
        self.current_accession: str | None = None
        self._current_windows = 0
        self._current_candidates = 0
        self._start: float | None = None
        self._genome_start: float | None = None

    def begin(self, now: float) -> None:
        """Stamp the scan start (idempotent — the first call wins, so wall_s is from t0)."""
        if self._start is None:
            self._start = now

    def begin_genome(self, accession: str, now: float) -> None:
        self.begin(now)
        self.current_accession = accession
        self._current_windows = 0
        self._current_candidates = 0
        self._genome_start = now

    def record_window(self, n_candidates: int) -> None:
        self.windows_scanned += 1
        self._current_windows += 1
        self.candidates_found += int(n_candidates)
        self._current_candidates += int(n_candidates)

    def end_genome(self, now: float) -> None:
        """Close the current genome into ``per_genome`` (a no-op if none is open).

        The window count is whatever this genome contributed — partial when a window cap or an
        interruption cut it short — so the record is honestly labelled, never inflated to the tile
        count of a genome that did not finish.
        """
        if self.current_accession is None:
            return
        wall = None if self._genome_start is None else max(0.0, now - self._genome_start)
        self.per_genome.append(
            {
                "accession": self.current_accession,
                "n_windows": self._current_windows,
                "n_candidates": self._current_candidates,
                "wall_s": wall,
                "windows_per_s": _safe_ratio(self._current_windows, wall),
            }
        )
        self.genomes_completed += 1
        self.current_accession = None
        self._current_windows = 0
        self._current_candidates = 0
        self._genome_start = None

    def snapshot(
        self, now: float, *, complete: bool = False, note: str | None = None
    ) -> dict[str, Any]:
        """The measurement, derived from the recorded counts + the injected clock."""
        wall = 0.0 if self._start is None else max(0.0, now - self._start)
        return {
            "schema_version": self.schema_version,
            "step": self.step,
            "complete": bool(complete),
            "note": note,
            "windows_scanned": self.windows_scanned,
            "candidates_found": self.candidates_found,
            "genomes_completed": self.genomes_completed,
            "wall_s": wall,
            "windows_per_s": _safe_ratio(self.windows_scanned, wall),
            "candidates_per_window": _safe_ratio(self.candidates_found, self.windows_scanned),
            "in_progress_accession": self.current_accession,
            "in_progress_windows": self._current_windows,
            "per_genome": list(self.per_genome),
        }

    def write(
        self, path: str | Path, now: float, *, complete: bool = False, note: str | None = None
    ) -> Path:
        """Persist the snapshot **atomically** (write a sibling ``.tmp`` → ``os.replace``).

        Atomicity is load-bearing: a SIGTERM landing mid-write must not truncate the last good
        snapshot, so the new bytes are staged and swapped in with a single ``os.replace`` — the
        reader ever sees either the previous complete snapshot or the new one, never a half-file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        payload = json.dumps(
            self.snapshot(now, complete=complete, note=note), indent=2, sort_keys=True
        )
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, p)
        return p


def run_measured_scan(
    genome_windows: Iterable[tuple[str, Iterable[tuple[str, str]]]],
    scan_window: Callable[[str, str], Sequence[MiningCandidate]],
    *,
    progress_out: str | Path | None = None,
    flush_every_windows: int = 500,
    max_windows: int | None = None,
    now: Callable[[], float] = time.monotonic,
    log: ScanThroughputLog | None = None,
) -> tuple[list[MiningCandidate], ScanThroughputLog]:
    """Drive a genome-streamed scan with persistent throughput accounting.

    ``genome_windows`` yields ``(accession, windows)`` where ``windows`` is any iterable of
    ``(window_name, sequence)``; ``scan_window`` maps one window to its (possibly empty) list of
    false-positive :class:`MiningCandidate` s. The throughput log is flushed to ``progress_out``
    at the start, every ``flush_every_windows`` windows, at each genome boundary, and — in a
    ``finally`` — once more at the end (or at a mid-scan interruption), so a wall-clock kill leaves
    the last snapshot. With ``max_windows`` set the scan stops cleanly after that many windows (a
    self-terminating throughput sample that finishes inside a short wall); the stop is intentional,
    so the final snapshot is marked ``complete``. The clock is injected (``now``) so the accounting
    is deterministic under test.
    """
    if flush_every_windows < 1:
        raise MineRoundError(f"flush_every_windows must be >= 1, got {flush_every_windows}")
    if max_windows is not None and max_windows < 1:
        raise MineRoundError(f"max_windows must be >= 1 when set, got {max_windows}")

    log = log if log is not None else ScanThroughputLog()
    out: list[MiningCandidate] = []
    log.begin(now())
    if progress_out is not None:
        log.write(progress_out, now())  # an initial snapshot: an immediate stall is still visible
    completed = False
    reached_cap = False
    try:
        for accession, windows in genome_windows:
            log.begin_genome(str(accession), now())
            hit_cap = False
            for window_name, seq in windows:
                candidates = scan_window(window_name, seq)
                out.extend(candidates)
                log.record_window(len(candidates))
                if progress_out is not None and log.windows_scanned % flush_every_windows == 0:
                    log.write(progress_out, now())
                if max_windows is not None and log.windows_scanned >= max_windows:
                    hit_cap = True
                    break
            log.end_genome(now())
            if progress_out is not None:
                log.write(progress_out, now())
            if hit_cap:
                reached_cap = True
                break
        completed = True
    finally:
        if progress_out is not None:
            note = "window_cap" if reached_cap else ("completed" if completed else "interrupted")
            log.write(progress_out, now(), complete=completed, note=note)
    return out, log


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


def _scan_one_window(
    model: Any,
    dev: Any,
    window_name: str,
    seq: str,
    *,
    covariation_status: Mapping[str, str] | None = None,
) -> list[MiningCandidate]:
    """Scan one ``(window_name, sequence)`` with a **preloaded** model → its FP candidates.

    The single per-window scan step, shared verbatim by :func:`_scan_windows` (the substrate-window
    path) and :func:`scan_admissible_genomes`'s measured driver (promote-don't-duplicate: one call
    site for ``scan_sequence`` + the D3 operator ``call_candidates`` under the provisional
    criterion, so both paths call loci identically).
    """
    from tbox_finder.infer.call import call_candidates
    from tbox_finder.infer.scan import scan_sequence

    reconciled = scan_sequence(model, seq, device=dev)
    candidates = call_candidates(
        reconciled.log_probs,
        reconciled.zero_flanked,
        threshold=mining_criterion.PROVISIONAL_THRESHOLD,
        min_span=mining_criterion.PROVISIONAL_MIN_SPAN,
        gap_merge=mining_criterion.PROVISIONAL_GAP_MERGE,
    )
    return window_candidates_to_mining(
        candidates, window_name=window_name, covariation_status=covariation_status
    )


def _scan_windows(
    model: Any,
    windows: Any,
    dev: Any,
    *,
    covariation_status: Mapping[str, str] | None = None,
) -> list[MiningCandidate]:
    """Scan an iterable of ``(window_name, sequence)`` with a **preloaded** model → candidates.

    Used by :func:`scan_substrate_windows` (one window list); the checkpoint is loaded once by the
    caller, not once per window. Delegates each window to :func:`_scan_one_window`.
    """
    out: list[MiningCandidate] = []
    for window_name, seq in windows:
        out.extend(
            _scan_one_window(model, dev, window_name, seq, covariation_status=covariation_status)
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
    progress_out: str | Path | None = None,
    flush_every_windows: int = 500,
    max_windows: int | None = None,
) -> list[MiningCandidate]:
    """Tile + scan a slice of the a2 host-order-admissible genomes → false-positive candidates.

    Leg (a) of a mining round. The 660 admissible accessions (:func:`load_admissible_accessions`)
    are sharded across GPUs by the sbatch; each shard calls this with its accession slice. Reuses
    :func:`tbox_finder.mining.substrate_windows.iter_genome_windows` (canonical 1024/512 tiling) so
    the scanned geometry is identical to the shard-spec emitter's, and streams genome-by-genome so a
    slice holds one genome's windows at a time. The checkpoint is loaded once for the whole slice.

    ``progress_out`` / ``flush_every_windows`` / ``max_windows`` thread the
    :func:`run_measured_scan` throughput instrumentation through leg (a): with ``progress_out`` set,
    win/s/GPU + the partial FP-rate are flushed to that **persistent** path (per genome + every
    ``flush_every_windows`` windows) so a wall-clock kill still yields a measurement;
    ``max_windows`` bounds the scan to a self-terminating throughput sample (the A10 Phase-1 probe).
    """
    from tbox_finder.infer.scan import load_stage1_checkpoint
    from tbox_finder.mining.substrate_windows import iter_genome_windows, read_genome_fasta

    model = load_stage1_checkpoint(checkpoint_path, device=device)
    dev = device if device is not None else next(model.parameters()).device

    def scan_window(window_name: str, seq: str) -> list[MiningCandidate]:
        return _scan_one_window(model, dev, window_name, seq)

    genome_windows = (
        (acc, iter_genome_windows(str(acc), read_genome_fasta(genome_dir, str(acc))))
        for acc in accessions
    )
    candidates, _log = run_measured_scan(
        genome_windows,
        scan_window,
        progress_out=progress_out,
        flush_every_windows=flush_every_windows,
        max_windows=max_windows,
    )
    return candidates


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
    unevidenced = msa_supply_declaration_unevidenced(plan)
    if unevidenced:
        # The artifact must not be able to read as a good plan. The §9.3 artifact verify gates
        # on `ready == true` in this very file, so returning 4 *after* writing a plan that says
        # `ready: true` would leave the stale artifact to pass a check the run just failed — a
        # guard reporting after the side effect it exists to prevent. The write is kept (rather
        # than skipped) so a plan from an EARLIER run cannot survive at this path either, and
        # `readiness` keeps the readiness gate's own untouched verdict: nothing is misreported,
        # the top-level "may this round proceed" is simply False, and it names what overrode it.
        plan = {
            **plan,
            "ready": False,
            "ready_overridden_by": "msa_supply_declaration_unevidenced",
        }
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    if unevidenced:
        # Its own exit code: 3 is the *expected* refusal the sbatch converts to a clean exit,
        # so routing this through 3 would turn a misconfigured checkout into a silent
        # "no round today". This is a staging fault.
        print(
            "FATAL: this round declares the covariation MSA supply available, but this "
            "checkout cannot evidence it — "
            f"{'; '.join(plan['msa_supply_derivation']['reasons'])}",
            file=sys.stderr,
        )
        return 4
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


def _install_sigterm_flush() -> None:
    """Convert SLURM's wall-clock SIGTERM into a ``SystemExit`` so ``finally`` blocks run.

    SLURM sends SIGTERM (then SIGKILL after ``KillWait``) when a job hits its ``--time`` wall. The
    default SIGTERM disposition terminates the process **without** unwinding the stack, so
    :func:`run_measured_scan`'s ``finally``-flush would not fire and the last flushed progress
    snapshot (up to ``flush_every_windows`` windows old) would be the newest evidence. Raising
    ``SystemExit`` instead lets the ``finally`` write one last snapshot — pinning the exact
    in-progress genome when the wall hit — before the process exits (within ``KillWait``). This is
    the belt to the periodic-flush suspenders; both write only to the persistent ``reports/`` path.
    """

    def _handler(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)


def _cmd_scan_shard(args: argparse.Namespace) -> int:
    if args.progress_out:
        _install_sigterm_flush()
    accessions = [
        line.strip()
        for line in Path(args.accessions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        accessions = accessions[: args.limit]
    candidates = scan_admissible_genomes(
        args.checkpoint,
        accessions,
        genome_dir=args.genome_dir,
        device=args.device,
        progress_out=args.progress_out,
        flush_every_windows=args.flush_every_windows,
        max_windows=args.max_windows,
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
    # The same evidence gate the preflight applies, at the leg that actually mines: this is
    # the call that turns candidates into hard negatives, and it is reached on a different
    # job (mine_round_retrain.sbatch) from the one that ran `plan`, so it cannot inherit the
    # preflight's verdict. Declared-available + unevidenced is fatal (4), never a quiet round.
    derivation = derive_msa_supply_available()
    if msa_supply_declaration_unevidenced(
        {"msa_supply_available": args.msa_supply_available, "msa_supply_derivation": derivation}
    ):
        print(
            "FATAL: --msa-supply-available declared, but this checkout cannot evidence the "
            f"covariation MSA supply — {'; '.join(derivation['reasons'])}",
            file=sys.stderr,
        )
        return 4
    report = apply_spare_rule(
        args.manifest,
        args.status_table,
        rscape_installed=bool(rscape_installed),
        msa_supply_available=bool(args.msa_supply_available),
        synteny_status_table=args.synteny_status,
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


def _add_msa_supply_flags(parser: argparse.ArgumentParser) -> None:
    """The MSA-supply declaration, in **both** directions.

    The default is the module flag, read at parser-build (i.e. ``main()``) time so flipping
    :data:`MSA_SUPPLY_AVAILABLE` reaches the CLI without a CLI change. Since P3-15′-a that
    default is ``True``, which is exactly why the negative flag exists: a bare ``store_true``
    would leave the declaration one-way and give a checkout that has **not** staged the
    homolog DB no way to say so — it would have to discover the gap the expensive way, in the
    producer array, after the GPU scan legs. Mutually exclusive, so a script cannot assert
    both and silently get whichever argparse saw last.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--msa-supply-available",
        dest="msa_supply_available",
        action="store_true",
        default=MSA_SUPPLY_AVAILABLE,
        help=(
            "declare the ADR-0006 D7/A1 per-candidate MSA supply available; absent ⇒ the "
            f"module default MSA_SUPPLY_AVAILABLE (currently {MSA_SUPPLY_AVAILABLE})"
        ),
    )
    group.add_argument(
        "--no-msa-supply-available",
        dest="msa_supply_available",
        action="store_false",
        default=argparse.SUPPRESS,
        help=(
            "declare it UNAVAILABLE on this machine (e.g. the homolog DB is not staged here) "
            "— the conservative direction: the round refuses instead of mining"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """The P2 CLI parser, extracted so it can be inspected without running a round.

    ⚠ Review found the test that checked ``--synteny-status`` was reachable doing so by
    parsing ``[..., "--synteny-status", "t.json", "--help"]`` and asserting exit 0.  argparse
    handles ``--help`` **before** it rejects an unknown option, so that exits 0 even for a
    flag that does not exist — verified by execution on a parser with no such option at all.
    The assertion was vacuous; a structural check needs the parser object.
    """
    parser = argparse.ArgumentParser(prog="tbox_finder.mining.mine_round")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="readiness preflight (refuses when no protective backend)")
    plan.add_argument(
        "--rscape-installed",
        type=_parse_bool,
        default=None,
        help="override R-scape presence (default: probe covariation.backend_available())",
    )
    _add_msa_supply_flags(plan)
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
        "--limit", type=int, default=None, help="scan only the first N genomes (sizing smoke/probe)"
    )
    scan.add_argument(
        "--progress-out",
        default=None,
        help=(
            "persistent throughput-snapshot path (write under reports/, NEVER /tmp): win/s/GPU + "
            "FP-rate, flushed at start + per genome + every --flush-every-windows windows + on "
            "exit; survives a wall-clock SIGTERM so a timed-out scan still yields a measurement"
        ),
    )
    scan.add_argument(
        "--flush-every-windows",
        type=int,
        default=500,
        help="flush the throughput snapshot every N windows (default 500)",
    )
    scan.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="stop cleanly after N windows — a self-terminating throughput sample (A10 probe)",
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
    _add_msa_supply_flags(apply)
    apply.add_argument("--relaxed-arch-available", action="store_true")
    apply.add_argument("--synteny-available", action="store_true")
    apply.add_argument(
        "--synteny-status",
        default=None,
        help=(
            "candidate_id → criterion-(c) status (synteny_producer merge output); "
            "REQUIRED whenever the round declares --synteny-available"
        ),
    )
    apply.add_argument("--union-prior", default="data/processed/priors/union_prior.parquet")
    apply.add_argument("--corpus", default="data/processed/master_clean_v0.parquet")
    apply.set_defaults(func=_cmd_apply_spare_rule)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


__all__ = [
    "CERTIFIED_MSA",
    "GENOMIC_WINDOW_POOL",
    "HOMOLOG_DB_DVC",
    "HOMOLOG_MSA_PROVENANCE",
    "MSA_SUPPLY_AVAILABLE",
    "MineRoundError",
    "PRODUCER_ENTRY_POINTS",
    "REQUIRED_MATCHED_CONTROL_FLAGS",
    "REPO_ROOT",
    "SCHEMA_VERSION",
    "STEP",
    "ScanThroughputLog",
    "apply_spare_rule",
    "build_round_availability",
    "candidate_evidence",
    "derive_msa_supply_available",
    "evaluate_probe_round",
    "load_admissible_accessions",
    "main",
    "msa_supply_declaration_unevidenced",
    "parse_window_name",
    "plan_round",
    "read_dvc_dir_pointer",
    "read_fp_manifest",
    "run_measured_scan",
    "scan_admissible_genomes",
    "scan_probe_variants",
    "scan_substrate_windows",
    "window_candidates_to_mining",
    "write_fp_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
