"""P3-15'-g-iii — the matched control's PRODUCER RUN: its array sizing and its submit plan.

P3-15'-g-ii produced the control's queries: **317** Stage-1 re-detected spans over **160**
held-out curated records, matched to the 941 round-0 false-positive candidates the
criterion-(b) sweep read (KS *D* 0.084 against the raw-element arm's 0.965).  This step
turns those queries into a `nhmmer` + `mlocarna` + R-scape producer run — the same
instrument, on the same target DB, at the same cutoffs — and this module is the part of it
that must be **derived rather than typed**: the array width, the per-task worst case, and
the exact token list of the two commands the §9.3 ack authorizes.

**Why a report and not a paragraph.**  A submit-ack is judged on numbers, and every number
in it here comes from a committed artifact re-read at runtime: the array width from the
manifest's own candidate count through the *shipped* :func:`~tbox_finder.mining.
covariation_producer.partition_shards`, the per-candidate worst case from P3-15'-g's
sizing report, the query/FP length medians from P3-15'-g-ii's detect report, the wall limit
and the D17 cutoffs parsed out of the sbatch's own bytes.  A width retyped into a chat
message is a number nothing can re-derive and nothing can invalidate
([[pinned-constant-that-nothing-reads]], [[hydra-struct-mode-override-needs-yaml-key]]).

**What the plan refuses.**  :func:`plan_clauses` is a versioned, enumerated clause set,
each clause emptiness-guarded, and the plan is refused unless every one passes.  Three of
them are the ones that would otherwise be discovered on the cluster after a queue wait:
the shipped shard validator must accept **every** shard against the query FASTA (the exact
call each array task makes, not a re-spelling of it), the sbatch must actually carry the
sequence route, and its wall limit must cover the worst-case task.  One more is the
mistake this step is most likely to make quietly: the control's ``ROUND_DIR`` must **not**
be the false-positive supply's, or the control's consensuses would be promoted into the
directory the P3-15'-f sweep reads and the two supplies would be one.

**This module pins nothing** (``pins_nothing: true``).  It sizes and it refuses; the
submit itself is a CLAUDE.md §9.3 stop, and criterion (b)'s parameter choice is downstream
of the run's result.

Run (LOCAL, sub-second)::

    PYTHONPATH=src python -m tbox_finder.mining.curated_control_run plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder.mining.architecture_param_measure import portable_path
from tbox_finder.mining.covariation_producer import (
    CandidateSpec,
    ProducerError,
    load_query_fasta,
    partition_shards,
    read_candidate_manifest,
    resolve_shard_queries,
)
from tbox_finder.mining.curated_control_sizing import ENV_LOCK, partition_inputs
from tbox_finder.provenance import build_provenance

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-g-iii"
ADR = (
    "ADR-0005 A10 (the per-candidate producer cost centre and its decoupled-array "
    "orchestration); ADR-0006 A2 (min_sequences = 20) / A3 (mlocarna) / A4 (criterion "
    "(b)'s instrument); D17 (the P6-frozen search cutoffs this run does not touch)"
)

#: The P3-15'-g-ii artifacts this run consumes, and the P3-15'-g sizing it is priced from.
DEFAULT_MANIFEST = "data/processed/mining/curated_control_manifest_v0.json"
DEFAULT_QUERY_FASTA = "data/processed/mining/curated_control_queries_v0.fasta"
DEFAULT_SIZING_REPORT = "reports/p3/curated_control_sizing.json"
DEFAULT_DETECT_REPORT = "reports/p3/curated_control_detect.json"

#: The producer array itself — the SAME sbatch the full-corpus FP supply ran (jobs
#: 1205/1254), taking the sequence route through its optional ``QUERY_FASTA`` export.  A
#: sibling script would have had to duplicate the promotion loop, the marker discipline and
#: the D17 cutoffs, and "the control ran the same instrument" would then mean "a copy of
#: it" ([[promote-dont-duplicate-is-a-correctness-rule]]).
DEFAULT_SBATCH = "slurm/p2/mine_round_producer.sbatch"

#: job 766's certified `nhmmer` search config, committed.  The sbatch's own cutoffs are
#: checked against it rather than against a retyped list: this is what makes "the same
#: instrument" a claim with a witness.
CERTIFIED_SEARCH_REPORT = "reports/p2/homolog_msa_search_report.json"

#: The control's OWN scratch round directory.  ⚠ NOT ``round_p3_15_supply``: that is where
#: the 941 FP candidates' consensuses live, and the P3-15'-f sweep reads every ``msa.sto``
#: under it, so promoting the control's consensuses there would silently merge the two
#: supplies and every rate measured afterwards would be over a mixture.
DEFAULT_ROUND_DIR = "$HOME/tbox-scratch/round_p3_15g_control"

#: The FP supply's round directory — named here only so the clause below can refuse it.
FP_SUPPLY_ROUND_DIR = "$HOME/tbox-scratch/round_p3_15_supply"

#: Candidates per array task.  P3-15'-g's envelope is tabulated at this shard size and its
#: worst-case per-task wall is ``shard_size × worst_case_per_candidate_s``; changing it
#: changes that bound, so it is a CLI argument that defaults to the sized value rather than
#: a literal buried in the arithmetic.
DEFAULT_SHARD_SIZE = 20

#: The clause set is versioned and enumerated: ``all_pass`` is over exactly these keys, a
#: missing key is a hard failure, and adding a clause bumps the version so an older report
#: re-validates FALSE rather than silently skipping the new check
#: ([[new-gate-clause-invalidates-old-reports]], [[gate-clauses-need-re-derivation]]).
CLAUSE_SCHEMA_VERSION = "1.1"  # 1.1 adds query_fasta_export_is_the_staged_copy
REQUIRED_PLAN_CLAUSES: tuple[str, ...] = (
    "manifest_nonempty",
    "query_fasta_nonempty",
    "every_manifest_candidate_has_a_query",
    "every_query_length_matches_its_span",
    "shipped_validator_accepts_every_shard",
    "shards_partition_the_manifest",
    "array_width_covers_every_shard",
    "round_dir_is_not_the_fp_supply",
    "align_timeout_matches_the_sized_bound",
    "query_fasta_export_is_the_staged_copy",
    "sbatch_carries_the_sequence_route",
    "sbatch_cutoffs_match_the_certified_config",
    "worst_case_task_fits_the_wall_limit",
)


class RunPlanError(ValueError):
    """The run cannot be planned from the artifacts as given."""


def _sha256_of(path: str | Path) -> str:
    """sha256 of one file, chunked — the query FASTA is small but the helper is not sized to it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# Reading the committed inputs (no number in this module is typed twice)
# ═════════════════════════════════════════════════════════════════════════════
def _float_or_none(text: str) -> float | None:
    """``float(text)`` or ``None`` — an unparseable exported value must FAIL its clause.

    ``float("600s")`` raises, and an exception here would abort the plan with a traceback
    instead of a refused clause naming what disagreed.
    """
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _require(payload: Mapping[str, Any], *keys: str) -> Any:
    """``payload[k0][k1]…`` or a refusal naming the path that is missing.

    A ``KeyError`` escaping into the CLI would name one key with no indication of which
    committed artifact changed shape under it.
    """
    node: Any = payload
    for i, key in enumerate(keys):
        if not isinstance(node, Mapping) or key not in node:
            raise RunPlanError(f"missing key {'.'.join(keys[: i + 1])} — the input changed shape")
        node = node[key]
    return node


def read_sizing(path: str | Path = DEFAULT_SIZING_REPORT) -> dict[str, Any]:
    """P3-15'-g's envelope: the per-candidate worst case, the mean, and the sized bound.

    ``worst_case_per_candidate_s`` is read, not recomputed: its components (the FP corpus's
    per-stage search and score maxima) are recorded in ADR-0005 A10's Phase-2 pin, not in
    this report, so re-deriving it here would mean retyping two numbers from ADR prose and
    calling the result a derivation.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    envelope = _require(payload, "envelope")
    return {
        "worst_case_per_candidate_s": float(_require(envelope, "worst_case_per_candidate_s")),
        "expected_per_candidate_s": float(
            _require(envelope, "measured_from", "per_candidate_wall_s", "mean")
        ),
        "align_timeout_s": float(_require(envelope, "align_timeout_s")),
        "cpus_per_task": int(_require(envelope, "cpus_per_task")),
        "expected_wall_caveat": str(_require(envelope, "caveat")),
        "measured_from": str(_require(envelope, "measured_from", "corpus")),
    }


def read_query_lengths(path: str | Path = DEFAULT_DETECT_REPORT) -> dict[str, Any]:
    """The control's query length distribution and the FP one it is matched against."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matched = _require(payload, "matchedness_vs_fp_candidates")
    return {
        "query_length_nt": dict(_require(matched, "curated_length_nt")),
        "fp_length_nt": dict(_require(matched, "fp_length_nt")),
        "n_records_with_a_query": int(_require(payload, "dropout", "n_records_with_a_query")),
        "n_records_scanned": int(_require(payload, "dropout", "n_windows_scanned")),
    }


#: ``NAME="value"`` as the sbatch writes its constants (one per line, or several separated
#: by ``;`` on the CTRL_ line).
_SBATCH_ASSIGNMENT = re.compile(r'(?m)(?:^|;)\s*(?P<name>[A-Z_][A-Z0-9_]*)="(?P<value>[^"]*)"')
#: ``#SBATCH --time=HH:MM:SS`` (the only wall form this repo's sbatch files use).
_SBATCH_TIME = re.compile(r"(?m)^#SBATCH\s+--time=(\d+):(\d{2}):(\d{2})\s*$")

#: The five sbatch constants the D17 search cutoffs live in — ONE list, read by the guard
#: and by the block that indexes them. Two spellings drift: a guard naming three of them
#: lets the other two escape as a bare ``KeyError`` naming one key and no artifact, which
#: is exactly the diagnostic :func:`_require` exists to prevent.
SBATCH_CUTOFF_KEYS: tuple[str, ...] = (
    "CTRL_ENGINE",
    "CTRL_EVALUE",
    "CTRL_MAX_TARGET_SEQS",
    "CTRL_MIN_PIDENT",
    "CTRL_MIN_COV",
)


def read_sbatch(path: str | Path = DEFAULT_SBATCH) -> dict[str, Any]:
    """The wall limit, the D17 search cutoffs and the sequence-route markers, from its bytes.

    Parsed rather than restated: the sbatch is the artifact that runs, so a plan that
    describes a *different* wall or *different* cutoffs than the file carries would be
    describing a run that does not exist ([[verify-the-line-you-ship]]).
    """
    text = Path(path).read_text(encoding="utf-8")
    time_match = _SBATCH_TIME.search(text)
    if time_match is None:
        raise RunPlanError(f"{portable_path(path)}: no `#SBATCH --time=HH:MM:SS` header found")
    hours, minutes, seconds = (int(g) for g in time_match.groups())
    assigned = {m.group("name"): m.group("value") for m in _SBATCH_ASSIGNMENT.finditer(text)}
    missing = [k for k in SBATCH_CUTOFF_KEYS if k not in assigned]
    if missing:
        raise RunPlanError(
            f"{portable_path(path)}: cutoffs {missing} not found — the producer's search "
            "config is no longer where this plan reads it from"
        )
    return {
        "path": portable_path(path),
        "wall_limit_h": round(hours + minutes / 60.0 + seconds / 3600.0, 4),
        "cutoffs": {
            "engine": assigned["CTRL_ENGINE"],
            "evalue": float(assigned["CTRL_EVALUE"]),
            "max_target_seqs": int(assigned["CTRL_MAX_TARGET_SEQS"]),
            "min_pident": float(assigned["CTRL_MIN_PIDENT"]),
            "min_cov": float(assigned["CTRL_MIN_COV"]),
        },
        # Two independent halves of the seam: the guard that refuses a missing FASTA before
        # the search stage, and the argument that actually reaches the producer. Either one
        # alone would let a plan be written against an sbatch that cannot run it.
        "has_query_fasta_guard": "QUERY_FASTA" in assigned,
        "passes_query_fasta_argument": "SEARCH_QUERY_ARGS[@]" in text and "--query-fasta" in text,
    }


def read_certified_cutoffs(path: str | Path = CERTIFIED_SEARCH_REPORT) -> dict[str, Any]:
    """job 766's certified search config — the witness the sbatch's cutoffs are read against."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cutoffs = _require(payload, "cutoffs")
    return {
        "engine": str(_require(payload, "engine")),
        "evalue": float(_require(cutoffs, "evalue")),
        "max_target_seqs": int(_require(cutoffs, "max_target_seqs")),
        "min_pident": float(_require(cutoffs, "min_pident")),
        "min_cov": float(_require(cutoffs, "min_cov")),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The array layout and its envelope
# ═════════════════════════════════════════════════════════════════════════════
def shard_layout(specs: Sequence[CandidateSpec], *, shard_size: int) -> dict[str, Any]:
    """Partition the manifest exactly as the run will, and report the layout it produces.

    ``n_shards`` is ``ceil(n / shard_size)``; the sizes come from the **shipped**
    :func:`partition_shards`, so ``max_shard_size`` — the number the worst-case wall is
    computed on — is the real size of the real largest shard rather than the requested
    ``shard_size`` (they differ whenever ``n`` is not a multiple of it).
    """
    if shard_size < 1:
        raise RunPlanError(f"shard_size must be >= 1, got {shard_size}")
    if not specs:
        raise RunPlanError("the manifest carries no candidates — there is nothing to shard")
    n_shards = math.ceil(len(specs) / shard_size)
    shards = partition_shards(specs, n_shards)
    sizes = [len(s) for s in shards]
    return {
        "n_candidates": len(specs),
        "requested_shard_size": int(shard_size),
        "n_shards": n_shards,
        "array_spec": f"0-{n_shards - 1}",
        "shard_sizes": {"min": min(sizes), "max": max(sizes), "total": sum(sizes)},
        "shards": shards,
    }


def envelope(layout: Mapping[str, Any], sizing: Mapping[str, Any]) -> dict[str, Any]:
    """The per-task and whole-array cost, bounded and expected, on the sized per-candidate wall.

    The **bound** is what a submit is sized on: ``max_shard_size × worst_case_per_candidate_s``
    per task, which contains the ``ALIGN_TIMEOUT_S`` bound by construction and therefore does
    not extrapolate anything.  The **expectation** multiplies the FP corpus's measured mean
    onto curated queries and inherits P3-15'-g's own caveat about that population difference,
    carried here verbatim rather than paraphrased.
    """
    max_shard = int(_require(layout, "shard_sizes", "max"))
    n_shards = int(_require(layout, "n_shards"))
    cpus = int(_require(sizing, "cpus_per_task"))
    worst_wall_h = max_shard * float(sizing["worst_case_per_candidate_s"]) / 3600.0
    expected_wall_h = max_shard * float(sizing["expected_per_candidate_s"]) / 3600.0
    return {
        "cpus_per_task": cpus,
        "worst_case_wall_h_per_task": round(worst_wall_h, 4),
        "worst_case_core_h": round(worst_wall_h * n_shards * cpus, 2),
        "expected_wall_h_per_task": round(expected_wall_h, 4),
        "expected_core_h": round(expected_wall_h * n_shards * cpus, 2),
        "gpu_h": 0.0,
        "expected_wall_caveat": str(sizing["expected_wall_caveat"]),
        "bound_is_extrapolation_free": (
            "the worst case is max_shard_size x (search_max + ALIGN_TIMEOUT_S + score_max); "
            "the ALIGN_TIMEOUT_S term is a bound this run enforces, not a measurement of it"
        ),
    }


def min_cov_reach(lengths: Mapping[str, Any], min_cov: float) -> dict[str, Any]:
    """What ``--min-cov`` demands of a hit at the two populations' median query lengths.

    ``min_cov`` is applied against *query* length
    (:func:`~tbox_finder.mining.homolog_msa.select_nhmmer_homologs`), so the same cutoff is
    a different absolute demand on a different query population — the reason P3-15'-g-iii
    was told to re-check it against the measured 104 nt query median rather than the curated
    element's 279.  Stated as nucleotides at each median so the comparison is visible
    instead of asserted.
    """
    query_median = float(_require(lengths, "query_length_nt", "p50"))
    fp_median = float(_require(lengths, "fp_length_nt", "p50"))
    return {
        "min_cov": float(min_cov),
        "query_median_nt": query_median,
        "fp_median_nt": fp_median,
        "required_covered_nt_at_query_median": round(min_cov * query_median, 2),
        "required_covered_nt_at_fp_median": round(min_cov * fp_median, 2),
        "note": (
            "min_cov is a fraction of QUERY length, so the absolute demand tracks the query "
            "population; these two medians are 6 nt apart, which is what the P3-15'-g-ii "
            "re-detect was for. The cutoff is left at the D17-frozen value: changing it for "
            "the control would make it a different instrument from the corpus it calibrates."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The submit plan — the exact token list the §9.3 ack authorizes
# ═════════════════════════════════════════════════════════════════════════════
def submit_plan(
    layout: Mapping[str, Any],
    sizing: Mapping[str, Any],
    *,
    manifest: str,
    query_fasta: str,
    query_fasta_sha256: str,
    sbatch: str,
    round_dir: str,
) -> dict[str, Any]:
    """The commands the run consists of, as argv lists derived from the layout.

    **Staging comes first, and it is not a formality.**  ``*.fasta`` is git-LFS-tracked
    (``.gitattributes``) and the cluster has **no git-lfs binary at all** (measured
    2026-08-11: ``git lfs version`` is not a git command there), so the checkout holds a
    ~130-byte pointer where the sequences should be — a non-empty file that passes every
    ``-s`` test — and §9.3 step 3's ``git reset --hard`` restores that pointer on **every**
    sync.  The sequences are therefore copied to the round's own scratch directory, outside
    the checkout, where no later sync can revert them and where they cannot dirty the
    ``git status`` a provenance gate reads.  The copy is verified by digest, because a
    silently truncated or stale staged file is the one failure the sbatch's guards cannot
    see ([[git-lfs-pointers-in-ci]]).

    Then ``make-shards`` on the login node (a 317-row JSON partition, not compute); the array
    reads ``$ROUND_DIR/shards/shard_NNN.json``, and the sbatch's own ``[ -s "$SHARD" ]``
    guard catches a width/shard mismatch per task, before any search runs.

    ``$HOME`` is left **unexpanded** and single-quoted in the rsync destination: it is the
    *remote* shell that must expand it, and an unquoted ``$HOME`` would be expanded by the
    local one — naming the laptop's home directory, which does not exist on the cluster.
    """
    n_shards = int(_require(layout, "n_shards"))
    shard_dir = f"{round_dir}/shards"
    staged_query_fasta = f"{round_dir}/{Path(query_fasta).name}"
    make_shards = [
        "python",
        "-m",
        "tbox_finder.mining.covariation_producer",
        "make-shards",
        "--manifest",
        manifest,
        "--n-shards",
        str(n_shards),
        "--out-dir",
        shard_dir,
    ]
    exports = ",".join(
        [
            "ALL",
            f"SHARD_DIR={shard_dir}",
            f"ROUND_DIR={round_dir}",
            f"ALIGN_TIMEOUT_S={sizing['align_timeout_s']:g}",
            f"QUERY_FASTA={staged_query_fasta}",
        ]
    )
    sbatch_argv = [
        "sbatch",
        f"--array={layout['array_spec']}",
        f"--export={exports}",
        sbatch,
    ]
    return {
        "round_dir": round_dir,
        "shard_dir": shard_dir,
        "prep_env": "tbox-homology",
        "staging": {
            "why": (
                "*.fasta is git-LFS-tracked and the cluster has NO git-lfs binary (measured "
                "2026-08-11), so its checkout holds a ~130-byte pointer that passes every "
                "non-empty check, and `git reset --hard` restores that pointer on every §9.3 "
                "sync. The sequences are staged OUTSIDE the checkout so no later sync reverts "
                "them and nothing dirties the repo's git status."
            ),
            "staged_query_fasta": staged_query_fasta,
            "query_fasta_sha256": query_fasta_sha256,
            "mkdir_command": f"mkdir -p {round_dir}",
            "copy_command": f"rsync -avz {query_fasta} two:'{staged_query_fasta}'",
            "verify_command": (
                f"sha256sum {staged_query_fasta}  # must equal query_fasta_sha256 above"
            ),
        },
        "make_shards_argv": make_shards,
        "sbatch_argv": sbatch_argv,
        "make_shards_command": " ".join(make_shards),
        "sbatch_command": " ".join(sbatch_argv),
        "verification": {
            "note": "artifact-based (sacct off, CLAUDE.md §9.3 step 8)",
            "expect_shard_ok_markers": n_shards,
            "shard_ok_glob": f"{round_dir}/status/shard_*.SHARD_OK",
            "expect_zero_byte_err": "reports/p2/mine_round_producer_<jobid>_*.err",
            "consensus_root": f"{round_dir}/msa/<candidate-slug>/msa.sto",
            "why_manual": (
                "this is a STANDALONE producer array with no retrain leg (d), and leg (d) is "
                "what normally counts the SHARD_OK markers against the shard files and fails "
                "loud on a gap — here that count is the operator's own §9.3 step-8 check"
            ),
        },
        "then": (
            "architecture_param_measure.measure --msa-root <round_dir>/msa over the resulting "
            "consensuses, read against P3-15'-f's identical grid; the read side needs no new "
            "code and is a separate step"
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The clause set — every one re-derived, every one emptiness-guarded
# ═════════════════════════════════════════════════════════════════════════════
def exported_value(submit: Mapping[str, Any], name: str) -> str | None:
    """The value ``name`` carries in the submit plan's ``--export`` token, or ``None``.

    Read back out of the token list rather than off the variable that built it: what the run
    receives is the ``--export`` string, and a clause that consults the *request* instead is
    true by construction on every real run — vacuous exactly where it is supposed to bite
    ([[gate-clauses-need-re-derivation]]).
    """
    for token in submit.get("sbatch_argv", []):
        if not str(token).startswith("--export="):
            continue
        for item in str(token)[len("--export=") :].split(","):
            key, sep, value = item.partition("=")
            if sep and key == name:
                return value
    return None


def plan_clauses(
    specs: Sequence[CandidateSpec],
    *,
    query_fasta: str | Path,
    layout: Mapping[str, Any],
    sizing: Mapping[str, Any],
    sbatch: Mapping[str, Any],
    certified: Mapping[str, Any],
    env: Mapping[str, Any],
    submit: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-derive every precondition of the run; ``all_pass`` is over the enumerated set.

    Each clause is computed from the artifacts, never read back from the configuration that
    requested it: a clause that consults the request is vacuously true exactly when the
    evidence is missing ([[clauses-must-guard-emptiness]]).  The two emptiness guards come
    first for that reason — with no candidates and no queries, every "for all" clause below
    is trivially true and the plan would certify an empty run.  The three run-shape clauses
    read their subject out of the ``--export`` token the run actually receives
    (:func:`exported_value`), not off the arguments that built it.
    """
    available = load_query_fasta(query_fasta)
    ids = [s.candidate_id for s in specs]
    with_query = [cid for cid in ids if cid in available]
    right_length = [
        s.candidate_id
        for s in specs
        if s.candidate_id in available
        and len(available[s.candidate_id]) == s.locus_end - s.locus_start
    ]
    shards = list(layout["shards"])
    sharded_ids = [s.candidate_id for shard in shards for s in shard]

    # The exact call every array task makes, on every shard, before any search runs. Not a
    # re-spelling of the rule: if this raises, the run dies at the same place for the same
    # reason, and the ack is refused instead of the queue wait being spent to discover it.
    validator_ok = True
    validator_error: str | None = None
    try:
        for shard in shards:
            resolve_shard_queries(shard, query_fasta)
    except ProducerError as exc:
        validator_ok = False
        validator_error = str(exc)

    exported_round_dir = exported_value(submit, "ROUND_DIR") or ""
    exported_timeout = exported_value(submit, "ALIGN_TIMEOUT_S")
    exported_query = exported_value(submit, "QUERY_FASTA")
    staged_query = str(submit.get("staging", {}).get("staged_query_fasta", ""))

    clauses = {
        "manifest_nonempty": len(specs) > 0,
        "query_fasta_nonempty": len(available) > 0,
        "every_manifest_candidate_has_a_query": bool(specs) and len(with_query) == len(specs),
        "every_query_length_matches_its_span": bool(specs) and len(right_length) == len(specs),
        "shipped_validator_accepts_every_shard": bool(shards) and validator_ok,
        "shards_partition_the_manifest": bool(shards)
        and sorted(sharded_ids) == sorted(ids)
        and len(set(sharded_ids)) == len(ids),
        "array_width_covers_every_shard": layout["array_spec"] == f"0-{len(shards) - 1}"
        and int(layout["n_shards"]) == len(shards),
        # A prefix test, not equality: `<fp>/msa` is a different string and the same supply.
        # Read off the EXPORTED value — that is the directory the array writes into.
        "round_dir_is_not_the_fp_supply": bool(exported_round_dir)
        and not (
            exported_round_dir == FP_SUPPLY_ROUND_DIR
            or exported_round_dir.startswith(FP_SUPPLY_ROUND_DIR.rstrip("/") + "/")
            or FP_SUPPLY_ROUND_DIR.startswith(exported_round_dir.rstrip("/") + "/")
        ),
        "align_timeout_matches_the_sized_bound": exported_timeout is not None
        and _float_or_none(exported_timeout) == float(sizing["align_timeout_s"]),
        # The repo path is a git-LFS POINTER on the cluster; only the staged copy carries the
        # sequences, so exporting the repo path would search 130 bytes of metadata.
        "query_fasta_export_is_the_staged_copy": bool(staged_query)
        and exported_query == staged_query,
        "sbatch_carries_the_sequence_route": bool(sbatch["has_query_fasta_guard"])
        and bool(sbatch["passes_query_fasta_argument"]),
        "sbatch_cutoffs_match_the_certified_config": dict(sbatch["cutoffs"]) == dict(certified),
        "worst_case_task_fits_the_wall_limit": float(env["worst_case_wall_h_per_task"])
        < float(sbatch["wall_limit_h"]),
    }
    missing = [name for name in REQUIRED_PLAN_CLAUSES if name not in clauses]
    if missing:
        raise RunPlanError(f"clause set is incomplete — missing {missing}")
    return {
        "clause_schema_version": CLAUSE_SCHEMA_VERSION,
        "clauses": {name: bool(clauses[name]) for name in REQUIRED_PLAN_CLAUSES},
        "all_pass": all(bool(clauses[name]) for name in REQUIRED_PLAN_CLAUSES),
        "measured": {
            "n_candidates": len(specs),
            "n_query_records": len(available),
            "n_with_a_query": len(with_query),
            "n_with_matching_length": len(right_length),
            "n_shards": len(shards),
            "validator_error": validator_error,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# The report
# ═════════════════════════════════════════════════════════════════════════════
def plan(
    *,
    manifest: str | Path = DEFAULT_MANIFEST,
    query_fasta: str | Path = DEFAULT_QUERY_FASTA,
    sizing_report: str | Path = DEFAULT_SIZING_REPORT,
    detect_report: str | Path = DEFAULT_DETECT_REPORT,
    sbatch: str | Path = DEFAULT_SBATCH,
    certified_report: str | Path = CERTIFIED_SEARCH_REPORT,
    round_dir: str = DEFAULT_ROUND_DIR,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    """The whole plan: layout, envelope, submit tokens, and the clause set that gates them."""
    specs = read_candidate_manifest(manifest)
    sizing = read_sizing(sizing_report)
    lengths = read_query_lengths(detect_report)
    sbatch_facts = read_sbatch(sbatch)
    certified = read_certified_cutoffs(certified_report)
    layout = shard_layout(specs, shard_size=shard_size)
    env = envelope(layout, sizing)
    query_digest = _sha256_of(query_fasta)
    # Built BEFORE the clauses, because three of them read what the run actually receives
    # out of its `--export` token rather than off the arguments that produced it.
    submit = submit_plan(
        layout,
        sizing,
        manifest=portable_path(manifest),
        query_fasta=portable_path(query_fasta),
        query_fasta_sha256=query_digest,
        sbatch=sbatch_facts["path"],
        round_dir=round_dir,
    )
    clauses = plan_clauses(
        specs,
        query_fasta=query_fasta,
        layout=layout,
        sizing=sizing,
        sbatch=sbatch_facts,
        certified=certified,
        env=env,
        submit=submit,
    )
    if not clauses["all_pass"]:
        failed = sorted(n for n, ok in clauses["clauses"].items() if not ok)
        raise RunPlanError(
            "the run is not submittable — failing clause(s): "
            + ", ".join(failed)
            + (
                f" ({clauses['measured']['validator_error']})"
                if clauses["measured"].get("validator_error")
                else ""
            )
        )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "pins_nothing": True,
        "supply": {
            "manifest": portable_path(manifest),
            "query_fasta": portable_path(query_fasta),
            "n_queries": int(layout["n_candidates"]),
            "n_records_with_a_query": lengths["n_records_with_a_query"],
            "n_records_scanned": lengths["n_records_scanned"],
            "independent_observations_note": (
                "the producer's unit is the QUERY, but independent observations are bounded "
                "by the RECORDS behind them (one draw per ADR-0004 cluster); any CI from this "
                "control is stated on the record-level n, never on the query count "
                "(§7 ruling, 2026-08-10)"
            ),
            "query_length_nt": lengths["query_length_nt"],
            "fp_length_nt": lengths["fp_length_nt"],
        },
        "instrument": {
            "sbatch": sbatch_facts["path"],
            "route": "sequence (QUERY_FASTA) — only the query differs from the FP supply run",
            "cutoffs": sbatch_facts["cutoffs"],
            "cutoffs_witness": portable_path(certified_report),
            "align_timeout_s": sizing["align_timeout_s"],
            "min_cov_reach": min_cov_reach(lengths, sbatch_facts["cutoffs"]["min_cov"]),
            "min_sequences_floor_note": (
                "ADR-0006 A2's min_sequences = 20 is unchanged; a curated query does not "
                "self-hit the searched DB (P3-15'-g measured 76/76 FP assemblies in it), so "
                "the control's depth is one sequence lower at the same homolog set"
            ),
        },
        "array": {k: v for k, v in layout.items() if k != "shards"},
        "envelope": env,
        "wall_limit_h": sbatch_facts["wall_limit_h"],
        "submit": submit,
        "preflight": clauses,
        "sized_from": {
            "sizing_report": portable_path(sizing_report),
            "detect_report": portable_path(detect_report),
            "measured_from": sizing["measured_from"],
        },
    }
    repo_inputs, external = partition_inputs(
        [
            ("manifest", manifest),
            ("query_fasta", query_fasta),
            ("sizing_report", sizing_report),
            ("detect_report", detect_report),
            ("sbatch", sbatch),
            ("certified_report", certified_report),
        ]
    )
    body["provenance"] = build_provenance(
        rule="P3-15'-g-iii curated-control producer run plan",
        script=portable_path(__file__),
        inputs=repo_inputs,
        outputs=[],
        env_lock=ENV_LOCK,
        adr=ADR,
        extra={"schema_version": SCHEMA_VERSION, "external_inputs": external},
    )
    # The staged copy's digest and the provenance entry are computed independently, over the
    # same file, and must agree: the staging block is what an operator checks the cluster copy
    # against, so a plan that published one digest while recording another would authorize
    # verifying the wrong bytes. Refused rather than reconciled.
    recorded = body["provenance"]["inputs"].get(portable_path(query_fasta)) or (
        external.get("query_fasta", {}).get("sha256")
    )
    if recorded != query_digest:
        raise RunPlanError(
            "the staged query FASTA digest and the provenance entry disagree "
            f"({query_digest} vs {recorded}) — the plan cannot name the bytes to verify"
        )
    return body


DEFAULT_OUT = "reports/p3/curated_control_run_plan.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curated_control_run",
        description="P3-15'-g-iii: size the matched control's producer array and derive "
        "the exact submit token list the §9.3 ack authorizes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan", help="the array layout + envelope + submit plan")
    p_plan.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p_plan.add_argument("--query-fasta", default=DEFAULT_QUERY_FASTA)
    p_plan.add_argument("--sizing-report", default=DEFAULT_SIZING_REPORT)
    p_plan.add_argument("--detect-report", default=DEFAULT_DETECT_REPORT)
    p_plan.add_argument("--sbatch", default=DEFAULT_SBATCH)
    p_plan.add_argument("--certified-report", default=CERTIFIED_SEARCH_REPORT)
    p_plan.add_argument("--round-dir", default=DEFAULT_ROUND_DIR)
    p_plan.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    p_plan.add_argument("--out", default=DEFAULT_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        body = plan(
            manifest=args.manifest,
            query_fasta=args.query_fasta,
            sizing_report=args.sizing_report,
            detect_report=args.detect_report,
            sbatch=args.sbatch,
            certified_report=args.certified_report,
            round_dir=args.round_dir,
            shard_size=args.shard_size,
        )
    except (RunPlanError, ProducerError, ValueError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"plan: {body['supply']['n_queries']} queries "
        f"({body['supply']['n_records_with_a_query']} records) → "
        f"{body['array']['n_shards']} shards, array {body['array']['array_spec']}, "
        f"worst case {body['envelope']['worst_case_wall_h_per_task']} h/task "
        f"({body['envelope']['worst_case_core_h']} core-h) -> {out}"
    )
    return 0


__all__ = [
    "ADR",
    "CLAUSE_SCHEMA_VERSION",
    "DEFAULT_ROUND_DIR",
    "DEFAULT_SHARD_SIZE",
    "FP_SUPPLY_ROUND_DIR",
    "REQUIRED_PLAN_CLAUSES",
    "RunPlanError",
    "SCHEMA_VERSION",
    "STEP",
    "envelope",
    "main",
    "min_cov_reach",
    "plan",
    "plan_clauses",
    "read_certified_cutoffs",
    "read_query_lengths",
    "read_sbatch",
    "read_sizing",
    "shard_layout",
    "submit_plan",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
