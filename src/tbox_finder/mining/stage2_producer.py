"""P3-15′-b — the per-candidate **Stage-2 posterior producer**.

The mining spare rule (ADR-0005 D14) spares a candidate when **any** disjunct passes,
so a round mines only on the *conjunction* of failures. Its fourth disjunct,
``high_stage2_posterior``, has had a pinned predicate since P2-07 and no backend: the
round-0 FP manifest carries **coordinates**, and
:func:`~tbox_finder.mining.remine.load_stage2_posteriors` wants ``candidate_id →
posterior``. This module is the leg between them, and the mirror of
:mod:`tbox_finder.mining.covariation_producer` for the covariation-(a) disjunct.

**It composes; it re-derives nothing.**

.. code-block:: text

    homolog_msa.resolve_candidate_sequence   coordinates → contig nucleotides
      → infer.handoff.transcribe_to_rna      PRD §6 alphabet handoff (T→U, ± strand)
      → stage2.eval.score_rows               the P3-06 production arm's raw binary logit
      → calib.recalibrate.calibrated_posterior   σ(z/T) — the ADR-0005 D11 named posterior

Four facts shape what it emits.

**1. The posterior is the D11 *named* object — temperature-scaled, PRE prior-shift.**
``calibrated_posterior`` is called with **no priors**, so ``prior_shift_applied`` is
False and ``stack_applied`` is ``["train", "temperature_scale"]``. The prior-shifted
sibling is explicitly non-gated (GATE-2 grades ``graded_posterior_key ==
"named_posterior"``), and at benchmark prevalence it is miscalibrated by construction.
The array is read back through ``payload[payload["gated_posterior_key"]]`` rather than a
literal key, so the D11 indirection has one implementation.

**2. The temperature is READ, never re-typed.** :func:`read_temperature` pulls
``gate.calibration.temperature`` out of ``reports/gate2_p3_ece.json``. A hardcoded copy
in ``src/`` would be a second home for a number GATE-2 owns, free to go stale with
nothing failing ([[pinned-constant-that-nothing-reads]]).

**3. The span is the BARE locus — there is no flank knob, deliberately.**
``stage2/dataset.py`` *raises* on a non-zero flank ("a positives-only flank is a
separable shortcut, PRD §5"), so the Stage-2 head was trained, and its temperature
fitted, on bare loci. A flanked payload is off-distribution and its posterior is
uncalibrated; exposing a flag would let a run decalibrate every value it publishes
while every gate stayed green.

**4. Both strands are always scored; which one is EMITTED is a required argument.**
The manifest carries no strand, and Stage-1 is Caduceus-PS (RC-equivariant), so a
Stage-1 window asserts no orientation at all. Measured on the real 941 (2026-08-05,
laptop A-series run): the minus strand scores higher on **458 of 941**, and at a 0.9
operating point **119 candidates pass only on the minus strand** — not marginally
(``+ = 7e-6`` vs ``− = 0.9997``). Under a plus-only reading those 119 come back
``failed``, i.e. a *satisfied mining conjunct*, on loci Stage-2 calls near-certain
T-boxes. ``--strand-policy`` is therefore ``required=True`` with **no default**, exactly
as ``--stage2-threshold`` is: D14 pins the rule, the round supplies the value. Both
per-strand posteriors ride in the report so the emitted number is auditable rather than
trusted.

**What it will NOT do.** An unresolvable or unscored candidate is **omitted** from the
table, never written as ``0.0``: absent ⇒ ``stage2_posterior is None`` ⇒
``STATUS_UNAVAILABLE`` ⇒ **spared**, while a literal zero would resolve to ``failed``
and mine it. And because ``load_stage2_posteriors`` accepts ``{}`` without complaint and
the sbatch's ``[ -s ]`` passes on a two-byte ``{}``, a truncated array would otherwise
certify as a clean round of nothing ([[cost-knobs-can-certify]]) — so the **denominator
travels with the numerator** and :func:`merge_posterior_tables` refuses below
``--min-coverage``.

PRD §6, §9.1, §11; ADR-0005 D14 (the P3 disjunct), D11/A11 (the named posterior).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder.calib.recalibrate import calibrated_posterior
from tbox_finder.decoys import dinucleotide_shuffle
from tbox_finder.infer.handoff import transcribe_to_rna
from tbox_finder.integration.two_stage import read_temperature
from tbox_finder.mining.covariation_producer import (
    CandidateSpec,
    read_candidate_manifest,
    write_shards,
)
from tbox_finder.mining.homolog_msa import (
    GENOME_DIR,
    HomologMsaError,
    resolve_candidate_sequence,
)
from tbox_finder.stage2 import tokenizer as TOK
from tbox_finder.stage2.eval import (
    DEFAULT_CKPT_ROOT,
    DEFAULT_SWEEP_DIR,
    production_arm_config,
    repo_relative,
)

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-b"
ADR = "ADR-0005 D14 (the P3 Stage-2 disjunct); D11/A11 (the named posterior)"
ENV_LOCK = "envs/ml-rna.conda-lock.yml"

#: ``src/tbox_finder/mining/stage2_producer.py`` → the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_FP_MANIFEST = "data/processed/mining/round0_fp_manifest.json"
DEFAULT_GATE2_REPORT = "reports/gate2_p3_ece.json"

#: The committed designed-control evidence :func:`derive_stage2_supply_available`
#: reads. Git-tracked, neither DVC- nor LFS-shaped, so CI, the laptop and the cluster
#: all answer the same — the P3-15′-a discipline.
CONTROL_REPORT = "reports/p3/stage2_producer_control.json"

#: The wrapper key ``remine.load_stage2_posteriors`` unwraps. The wrapper form is used
#: rather than a flat mapping because the flat form validates **every** top-level key as
#: a posterior, so no metadata (``n_scored``, the temperature, the load record) could
#: ride along — it would refuse ``schema_version`` by name.
POSTERIORS_KEY = "posteriors"

STRAND_PLUS = "+"
STRAND_MINUS = "-"
BOTH_STRANDS = (STRAND_PLUS, STRAND_MINUS)

#: Emit the posterior of the as-scanned (forward-tiled) orientation only.
POLICY_AS_SCANNED = "as_scanned"
#: Emit ``max`` over the two orientations — the 2026-08-05 §7 decision (bioedca).
POLICY_MAX_OVER_STRANDS = "max_over_strands"
STRAND_POLICIES = (POLICY_AS_SCANNED, POLICY_MAX_OVER_STRANDS)

#: The public chain :func:`derive_stage2_supply_available` requires to be present.
#: Named exhaustively rather than sampled: a clause that checks a *subset* is satisfied
#: by a stub carrying exactly that subset.
PRODUCER_ENTRY_POINTS = (
    "build_rows",
    "score_shard",
    "merge_posterior_tables",
    "run_control",
)

#: Every clause :func:`derive_stage2_supply_available` must report a verdict for.
#:
#: ``all(clauses.values())`` is True over an **empty** map, and a clause whose branch never
#: ran is simply absent — so a clause that disappears reads exactly like a clause that
#: passed. Reproduced by execution before this existed: with ``production_arm_config()``
#: returning ``None`` while ``sweep_fingerprint`` succeeded, neither the ``except`` branch
#: nor the ``if`` branch set ``production_arm_on_record``, and the derivation returned
#: ``available: True`` on five of six clauses. Naming the set exhaustively — the same
#: discipline :data:`REQUIRED_CONTROL_FLAGS` already applies one level down — turns a
#: silently-skipped clause into a refusal ([[clauses-must-guard-emptiness]]).
SUPPLY_CLAUSES = (
    "gate2_calibration_wellformed",
    "production_arm_on_record",
    "producer_present",
    "producer_posterior_wired",
    "control_green",
    "control_matches_this_calibration",
)

#: The designed control's floors. **Measured first on the real checkpoint, then frozen**
#: (CLAUDE.md §10.3) — never chosen to make a run pass. They are gate-control thresholds,
#: not a science operating point: ``STAGE2_THRESHOLD`` itself stays unpinned until the
#: §13.1 phase gate, which is why nothing here is named that.
CONTROL_MIN_POSITIVE = 0.90
CONTROL_MAX_SHUFFLE = 0.10
CONTROL_MIN_MARGIN = 0.50

#: The control's matchedness dimensions, **named** rather than iterated off whatever keys
#: the record happens to carry: a clause read from the evidence's own key set is
#: vacuously satisfied exactly when the evidence is missing
#: ([[clauses-must-guard-emptiness]]). A "null" that is a copy of the positive, or of a
#: different length/composition, is *no power* — and no power reads as no signal
#: ([[control-matchedness-must-be-asserted]]).
REQUIRED_CONTROL_FLAGS = (
    "shuffle_differs_from_positive",
    "dinucleotide_composition_matched",
    "length_matched",
)


class Stage2ProducerError(ValueError):
    """Raised on malformed producer input, or on a produced table that cannot certify."""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 (CPU, torch-free) — coordinates → the two RNA payloads per candidate
# ═════════════════════════════════════════════════════════════════════════════
def build_rows(
    specs: Sequence[CandidateSpec],
    *,
    genome_dir: str | Path = GENOME_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """``specs`` → ``(rows, unresolved)``, two scoring rows per resolved candidate.

    ``rows`` are ``score_rows``-shaped (``row_id`` + ``rna_sequence``) with the strand
    carried alongside; ``row_id`` is ``f"{candidate_id}|{strand}"`` because ``score_rows``
    keys its output dict on ``str(row_id)`` and a duplicate id silently collapses to the
    last-scored logit with the list length still matching.

    The two exception classes are treated **differently on purpose**, mirroring the
    covariation producer's split. :class:`HomologMsaError` is a property of *this
    candidate* (contig index out of range, span off the contig end, a non-ACGTN span) —
    recorded, the candidate omitted, the shard continues, and the omission costs
    sensitivity because an absent id is spared. :class:`HomologDbError` (a missing genome
    FASTA) and :class:`HandoffError` (an out-of-alphabet payload) are properties of the
    *checkout* and are allowed to propagate: widening the ``except`` to cover them would
    turn an un-materialised DVC pull into a whole shard of ``unavailable``, which is
    silently spared and reads exactly like a clean run.
    """
    # A duplicate WITHIN one shard is invisible to `merge_posterior_tables`, which only
    # refuses a candidate claimed by two shards. Left unchecked the two rows share a
    # `row_id`, `score_rows` keys its output on `str(row_id)`, and both resolve to the
    # last-scored logit — with `len(scored) == len(rows)` still holding, so the shear is
    # silent ([[duplicate-key-merges-instead-of-colliding]]). Coverage would eventually
    # drop below the floor and refuse, but it would name the wrong cause, and only while
    # `--min-coverage` is 1.0.
    seen: set[str] = set()
    for spec in specs:
        if spec.candidate_id in seen:
            raise Stage2ProducerError(
                f"candidate_id {spec.candidate_id!r} appears twice in one shard — the two "
                "rows would share a row_id and collapse to a single logit"
            )
        seen.add(spec.candidate_id)

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    for spec in specs:
        try:
            # `strand` left at its "plus" default: orientation belongs to
            # `transcribe_to_rna`, which validates it. Passing "minus" here as well
            # would reverse-complement twice.
            dna = resolve_candidate_sequence(
                genome_dir, spec.accession, spec.locus_start, spec.locus_end
            )
        except HomologMsaError as exc:
            unresolved.append({"candidate_id": spec.candidate_id, "error": str(exc)})
            continue
        # `score_rows` calls the tokenizer's bare `encode`, which does NOT enforce the
        # context window; a payload past it would be silently truncated into a different
        # sequence than the one named. Round-0 spans are 50-296 nt so this cannot bite
        # today — which is the reason to assert it now rather than after a wider round.
        TOK.assert_within_context(dna, row_id=spec.candidate_id)
        for strand in BOTH_STRANDS:
            rows.append(
                {
                    "row_id": f"{spec.candidate_id}|{strand}",
                    "rna_sequence": transcribe_to_rna(dna, strand=strand),
                    "candidate_id": spec.candidate_id,
                    "strand": strand,
                    "n_nt": len(dna),
                }
            )
    return rows, unresolved


def emit_posterior(per_strand: Mapping[str, float], *, strand_policy: str) -> float:
    """The single posterior published for a candidate, given both orientations.

    ``strand_policy`` is validated here rather than trusted, so an unknown policy is a
    refusal and never a silent fallback to one of the two real ones.
    """
    if strand_policy not in STRAND_POLICIES:
        raise Stage2ProducerError(
            f"strand_policy must be one of {list(STRAND_POLICIES)}, got {strand_policy!r}"
        )
    missing = [s for s in BOTH_STRANDS if s not in per_strand]
    if missing:
        raise Stage2ProducerError(
            f"both strands must be scored before a policy is applied; missing {missing}"
        )
    if strand_policy == POLICY_AS_SCANNED:
        return float(per_strand[STRAND_PLUS])
    return float(max(per_strand[STRAND_PLUS], per_strand[STRAND_MINUS]))


def posterior_kind(strand_policy: str) -> str:
    """The human-readable name of the object the table publishes, per policy.

    ``max_over_strands`` is a *selection* over two draws of the calibrated object, so it
    is not literally the value GATE-2 graded. Saying so in the artifact is the whole
    point: a reader must not read the column as an ECE-graded posterior when it is a
    max of two.
    """
    if strand_policy == POLICY_AS_SCANNED:
        return (
            "named_posterior (temperature-scaled, PRE prior-shift) — ADR-0005 D11, "
            "of the as-scanned (forward-tiled) orientation"
        )
    return (
        "max over both orientations of the named_posterior (temperature-scaled, PRE "
        "prior-shift) — ADR-0005 D11. A max of two draws is a SELECTION on the graded "
        "object: GATE-2's ECE describes a single scoring, not this maximum."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 (GPU) — score one shard
# ═════════════════════════════════════════════════════════════════════════════
def resolve_production_arm(
    *,
    checkpoint_root: str | Path = DEFAULT_CKPT_ROOT,
    sweep_dir: str | Path = DEFAULT_SWEEP_DIR,
) -> dict[str, Any]:
    """The P3-06 production arm, resolved through ``discover_arms``/``select_arm_pair``.

    The arm name is never hard-coded: ``production_arm_config()`` reads ``conf/`` and the
    ``(aux_weight, lr)`` of each trained arm comes from the report that arm's own run
    wrote, so a sweep/config divergence raises instead of scoring the wrong weights.
    """
    from tbox_finder.stage2.eval import discover_arms, select_arm_pair

    arms = discover_arms(checkpoint_root, sweep_dir=sweep_dir)
    with_aux, _no_aux = select_arm_pair(arms, production=production_arm_config())
    return dict(arms[with_aux])


def score_shard(
    specs: Sequence[CandidateSpec],
    *,
    temperature: float,
    strand_policy: str,
    genome_dir: str | Path = GENOME_DIR,
    checkpoint_root: str | Path = DEFAULT_CKPT_ROOT,
    sweep_dir: str | Path = DEFAULT_SWEEP_DIR,
    batch_size: int = 4,
    device: str | None = None,
) -> dict[str, Any]:
    """Resolve → transcribe → score → calibrate one shard; return its table payload.

    Torch is imported inside :func:`~tbox_finder.stage2.eval.load_stage2_checkpoint`, so
    this module stays importable on the bare CI path; only *calling* this needs the GPU
    stack.
    """
    if strand_policy not in STRAND_POLICIES:
        raise Stage2ProducerError(
            f"strand_policy must be one of {list(STRAND_POLICIES)}, got {strand_policy!r}"
        )
    from tbox_finder.stage2.eval import load_stage2_checkpoint, score_rows

    rows, unresolved = build_rows(specs, genome_dir=genome_dir)
    arm = resolve_production_arm(checkpoint_root=checkpoint_root, sweep_dir=sweep_dir)
    model, load_record = load_stage2_checkpoint(
        arm["checkpoint_path"],
        device=device,
        attn_implementation=arm["attn_implementation"],
    )
    scored = score_rows(model, rows, batch_size=batch_size, device=device)

    posteriors, strand_posteriors = score_to_posteriors(
        rows, scored, temperature=temperature, strand_policy=strand_policy
    )
    return build_table(
        posteriors,
        strand_posteriors=strand_posteriors,
        unresolved=unresolved,
        n_candidates=len(specs),
        temperature=temperature,
        strand_policy=strand_policy,
        arm=arm,
        load_record=load_record,
        batch_size=batch_size,
        device=device,
    )


def score_to_posteriors(
    rows: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
    strand_policy: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """``score_rows`` output → ``(candidate_id → emitted posterior, per-strand map)``.

    ``score_rows`` returns one record per input row **in caller order**, which is what
    lets the zip be ``strict=True``; a length drift is a refusal, not a silent shear.
    """
    logits = np.asarray([r["tbox_logit"] for r in scored], dtype=np.float64)
    if logits.size != len(rows):
        raise Stage2ProducerError(
            f"scored {logits.size} rows for {len(rows)} payloads — the join would shear"
        )
    payload = calibrated_posterior(logits, temperature=float(temperature))
    # The D11 indirection: read the array through the key the calibrator NAMES as gated,
    # never a literal "named_posterior", so a move of the gated object cannot silently
    # publish the ungated one.
    named = payload[payload["gated_posterior_key"]]

    strand_posteriors: dict[str, dict[str, float]] = {}
    for row, value in zip(rows, named, strict=True):
        strand_posteriors.setdefault(row["candidate_id"], {})[row["strand"]] = float(value)
    posteriors = {
        cid: emit_posterior(per, strand_policy=strand_policy)
        for cid, per in strand_posteriors.items()
    }
    return posteriors, strand_posteriors


def build_table(
    posteriors: Mapping[str, float],
    *,
    strand_posteriors: Mapping[str, Mapping[str, float]],
    unresolved: Sequence[Mapping[str, str]],
    n_candidates: int,
    temperature: float,
    strand_policy: str,
    arm: Mapping[str, Any],
    load_record: Mapping[str, Any] | None,
    batch_size: int,
    device: str | None,
) -> dict[str, Any]:
    """The wrapper-form table, with its denominator beside its numerator."""
    validate_posteriors(posteriors)
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "arm": arm.get("arm"),
        "checkpoint_dir": arm.get("checkpoint_dir"),
        "attn_implementation": arm.get("attn_implementation"),
        "temperature": float(temperature),
        "temperature_source": DEFAULT_GATE2_REPORT,
        "strand_policy": strand_policy,
        "posterior_kind": posterior_kind(strand_policy),
        "prior_shift_applied": False,
        "stack_applied": ["train", "temperature_scale"],
        "n_candidates": int(n_candidates),
        "n_scored": len(posteriors),
        "n_unresolved": len(unresolved),
        "unresolved": [dict(u) for u in unresolved],
        "coverage": (len(posteriors) / n_candidates) if n_candidates else 0.0,
        "batch_size": int(batch_size),
        "device": device,
        "load": dict(load_record) if load_record is not None else None,
        "strand_posteriors": {k: dict(v) for k, v in strand_posteriors.items()},
        POSTERIORS_KEY: dict(posteriors),
    }


def validate_posteriors(posteriors: Mapping[str, float]) -> None:
    """Refuse anything the consumer would refuse — before it is written, not after.

    ``load_stage2_posteriors``'s ``[0, 1]`` clause catches a raw logit only when
    ``|z| > 1``; a logit of ``0.4`` sails through as a "posterior". Checking here as
    well is not redundant: this is where the value is still attributable to a shard.
    """
    for cid, value in posteriors.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Stage2ProducerError(f"posterior for {cid!r} is not a real number ({value!r})")
        if not np.isfinite(float(value)):
            raise Stage2ProducerError(f"posterior for {cid!r} is not finite ({value!r})")
        if not 0.0 <= float(value) <= 1.0:
            raise Stage2ProducerError(
                f"posterior for {cid!r} is {value}, outside [0, 1] — a calibrated posterior "
                "was expected (ADR-0005 D11 named posterior), not a logit"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Merge — and the completeness denominator the consumer does not carry
# ═════════════════════════════════════════════════════════════════════════════
def merge_posterior_tables(
    paths: Sequence[str | Path],
    *,
    n_candidates: int,
    min_coverage: float = 1.0,
) -> dict[str, Any]:
    """Merge per-shard tables into the round's single table, or refuse.

    Two refusals, both of which a downstream reader could not make for itself:

    * a ``candidate_id`` present in more than one shard — shards partition the manifest,
      so an overlap means the sharding drifted and one of the two values is being
      silently discarded;
    * coverage below ``min_coverage``. This is the load-bearing one.
      ``load_stage2_posteriors`` accepts ``{}`` without complaint and the sbatch's
      ``[ -s ]`` passes on a two-byte ``{}``, so a run that scored 3 of 941 would publish
      a table in which 938 candidates resolve to ``unavailable`` ⇒ **spared** ⇒ a round
      that reports success having decided almost nothing.
    """
    if not paths:
        raise Stage2ProducerError("no shard tables to merge")
    if n_candidates <= 0:
        raise Stage2ProducerError(
            f"n_candidates must be positive, got {n_candidates} — without a denominator "
            "coverage is unmeasurable and the merge cannot refuse a truncated run"
        )
    merged: dict[str, float] = {}
    strand_posteriors: dict[str, dict[str, float]] = {}
    unresolved: list[dict[str, str]] = []
    heads: list[dict[str, Any]] = []
    for path in paths:
        table = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(table, Mapping) or POSTERIORS_KEY not in table:
            raise Stage2ProducerError(f"{path}: not a producer table (no {POSTERIORS_KEY!r} key)")
        rows = table[POSTERIORS_KEY]
        if not isinstance(rows, Mapping):
            # Checking only that the key EXISTS let a list/str/null reach `.items()`,
            # raising AttributeError/TypeError — which `_cmd_merge` does not catch, so the
            # operator got a traceback and a generic exit code instead of the named refusal
            # and exit 3 the sbatch branches on.
            raise Stage2ProducerError(
                f"{path}: {POSTERIORS_KEY!r} is {type(rows).__name__}, not a "
                "candidate_id → posterior mapping"
            )
        for cid, value in rows.items():
            if cid in merged:
                raise Stage2ProducerError(
                    f"candidate_id {cid!r} appears in more than one shard table — shards "
                    "must partition the manifest, so one of the two values would be lost"
                )
            if not _is_real_number(value):
                # Validated BEFORE coercion. `validate_posteriors` runs on the merged map,
                # so coercing first launders the very types it rejects by name: `true`
                # becomes 1.0 and "0.5" becomes 0.5, both certifying. And `float(None)` /
                # `float("abc")` raise TypeError/ValueError, which `_cmd_merge` does not
                # catch — a traceback and a generic exit code instead of the named refusal
                # and exit 3 the sbatch branches on.
                raise Stage2ProducerError(
                    f"{path}: posterior for {cid!r} is not a real number ({value!r})"
                )
            merged[cid] = float(value)
        # The same named refusal the `posteriors` payload gets. Without it a shard
        # carrying `"strand_posteriors": [...]` raises AttributeError on `.items()`, and
        # `"unresolved": {...}` (or a list of strings) raises ValueError in `dict(u)` —
        # neither is a Stage2ProducerError, so `_cmd_merge` lets it out as a traceback
        # with a generic exit code instead of exit 3.
        shard_strands = table.get("strand_posteriors") or {}
        if not isinstance(shard_strands, Mapping):
            raise Stage2ProducerError(
                f"{path}: 'strand_posteriors' is {type(shard_strands).__name__}, not a "
                "candidate_id → per-strand mapping"
            )
        for cid, per in shard_strands.items():
            if not isinstance(per, Mapping):
                raise Stage2ProducerError(
                    f"{path}: strand_posteriors[{cid!r}] is {type(per).__name__}, not a "
                    "strand → posterior mapping"
                )
            strand_posteriors[cid] = dict(per)

        shard_unresolved = table.get("unresolved") or []
        if isinstance(shard_unresolved, (str, bytes, Mapping)) or not isinstance(
            shard_unresolved, Sequence
        ):
            raise Stage2ProducerError(
                f"{path}: 'unresolved' is {type(shard_unresolved).__name__}, not a list of "
                "records"
            )
        for entry in shard_unresolved:
            if not isinstance(entry, Mapping):
                raise Stage2ProducerError(
                    f"{path}: an 'unresolved' entry is {type(entry).__name__}, not a record"
                )
            unresolved.append(dict(entry))
        heads.append(table)

    _require_uniform(heads, "temperature")
    _require_uniform(heads, "strand_policy")
    _require_uniform(heads, "arm")
    validate_posteriors(merged)

    head = heads[0]
    coverage = len(merged) / n_candidates
    table = {
        **{k: v for k, v in head.items() if k not in {POSTERIORS_KEY, "strand_posteriors"}},
        "n_candidates": int(n_candidates),
        "n_scored": len(merged),
        "n_unresolved": len(unresolved),
        "unresolved": unresolved,
        "coverage": coverage,
        "n_shards": len(heads),
        "min_coverage": float(min_coverage),
        "strand_posteriors": strand_posteriors,
        POSTERIORS_KEY: merged,
    }
    if coverage < min_coverage:
        raise Stage2ProducerError(
            f"coverage {coverage:.6f} ({len(merged)}/{n_candidates}) is below the required "
            f"{min_coverage} — an unscored candidate is SPARED, so a truncated table would "
            "publish a round that decided almost nothing as a clean one"
        )
    return table


def _require_uniform(tables: Sequence[Mapping[str, Any]], key: str) -> None:
    values = {json.dumps(t.get(key), sort_keys=True) for t in tables}
    if len(values) > 1:
        raise Stage2ProducerError(
            f"shard tables disagree on {key!r}: {sorted(values)} — they were not produced "
            "by one run and must not be merged into one published table"
        )


# ═════════════════════════════════════════════════════════════════════════════
# The designed control that must fire
# ═════════════════════════════════════════════════════════════════════════════
def read_control_positive(path: str | Path, *, name: str) -> str:
    """A real T-box's nucleotides, taken from the committed tbdb sample **by sequence**.

    Selected on the record's ``Name``, but the sequence is the record's own
    ``FASTA_sequence`` — never a ``Name``-derived coordinate slice, which matches the
    FASTA length on only 61.7 % of tbdb records ([[tbdb-name-coords-untrustworthy]]).
    """
    import csv

    csv.field_size_limit(10**7)
    with open(path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("Name") == name:
                sequence = (record.get("FASTA_sequence") or "").strip().upper()
                if not sequence:
                    raise Stage2ProducerError(f"{path}: record {name!r} carries no FASTA_sequence")
                return sequence
    raise Stage2ProducerError(f"{path}: no record named {name!r}")


def _dinucleotides(seq: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(len(seq) - 1):
        counts[seq[i : i + 2]] = counts.get(seq[i : i + 2], 0) + 1
    return counts


def control_flags(positive: str, shuffled: str) -> dict[str, bool]:
    """Is ``shuffled`` a *matched null* for ``positive``, or is it no control at all?

    Split out of :func:`run_control` deliberately: this is the half that decides whether
    the separation number means anything, and it is pure, so it must be testable without
    a GPU. Folded into the scoring path it could only be exercised on a machine with the
    checkpoint — i.e. never in CI, which is where a control quietly losing its power
    would go unnoticed ([[control-matchedness-must-be-asserted]]).
    """
    return {
        "shuffle_differs_from_positive": shuffled != positive,
        "dinucleotide_composition_matched": _dinucleotides(shuffled) == _dinucleotides(positive),
        "length_matched": len(shuffled) == len(positive),
    }


def run_control(
    positive_sequence: str,
    *,
    temperature: float,
    seed: int,
    checkpoint_root: str | Path = DEFAULT_CKPT_ROOT,
    sweep_dir: str | Path = DEFAULT_SWEEP_DIR,
    device: str | None = None,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Score a known-positive T-box against its own dinucleotide shuffle.

    The shuffle is the *matched* null: identical length, identical dinucleotide (hence
    mononucleotide) composition, identical first and last symbol. That matchedness is
    **asserted and published**, not assumed — a null that is a copy of the positive
    would make "no power" indistinguishable from "no signal", and the gate would read
    green while measuring nothing ([[control-matchedness-must-be-asserted]]).

    Returns the record whether or not it separates, and reports ``green`` as data; the
    caller decides the exit code. A control that refused before writing would leave no
    evidence of the failure it exists to surface.
    """
    from tbox_finder.stage2.eval import load_stage2_checkpoint, score_rows

    shuffled = dinucleotide_shuffle(positive_sequence, random.Random(seed))
    rows = [
        {
            "row_id": "control_positive",
            "rna_sequence": transcribe_to_rna(positive_sequence, strand=STRAND_PLUS),
        },
        {
            "row_id": "control_shuffle",
            "rna_sequence": transcribe_to_rna(shuffled, strand=STRAND_PLUS),
        },
    ]
    arm = resolve_production_arm(checkpoint_root=checkpoint_root, sweep_dir=sweep_dir)
    model, load_record = load_stage2_checkpoint(
        arm["checkpoint_path"],
        device=device,
        attn_implementation=arm["attn_implementation"],
    )
    scored = score_rows(model, rows, batch_size=batch_size, device=device)
    logits = np.asarray([r["tbox_logit"] for r in scored], dtype=np.float64)
    if logits.size != len(rows):
        # The same length guard `score_to_posteriors` applies before its join. Without it
        # the indexing below raises IndexError, which `_cmd_control` does not convert into
        # the named refusal — a traceback instead of a control that failed.
        raise Stage2ProducerError(
            f"scored {logits.size} rows for {len(rows)} control payloads — a control needs "
            "both arms, and a short return would silently compare something else"
        )
    payload = calibrated_posterior(logits, temperature=float(temperature))
    named = payload[payload["gated_posterior_key"]]
    positive_posterior, shuffle_posterior = float(named[0]), float(named[1])

    # `build_table` types the load record as optional, so this must not assume otherwise —
    # an unguarded `.get` raises AttributeError, which `_cmd_control` turns into a
    # traceback rather than a control that failed. And a control with no checkpoint hashes
    # cannot be tied to the bytes it was earned against, which is exactly what
    # `control_matches_this_calibration` reads, so it is refused by name rather than
    # written as a null that would fail that clause one layer later with a vaguer reason.
    record = load_record or {}
    missing_hashes = [k for k in ("adapter_sha256", "heads_sha256") if not record.get(k)]
    if missing_hashes:
        raise Stage2ProducerError(
            f"the checkpoint loader reported no {missing_hashes} — a control that cannot "
            "name the bytes it was earned against is not evidence for any checkpoint"
        )

    flags = control_flags(positive_sequence, shuffled)
    green = (
        positive_posterior >= CONTROL_MIN_POSITIVE
        and shuffle_posterior <= CONTROL_MAX_SHUFFLE
        and (positive_posterior - shuffle_posterior) >= CONTROL_MIN_MARGIN
        and all(flags[k] for k in REQUIRED_CONTROL_FLAGS)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "arm": arm["arm"],
        "sweep_fingerprint": sweep_fingerprint(arm["arm"], sweep_dir=sweep_dir),
        "temperature": float(temperature),
        "temperature_source": DEFAULT_GATE2_REPORT,
        "seed": int(seed),
        "n_control_rows": len(rows),
        "positive_sequence_sha256": hashlib.sha256(positive_sequence.encode()).hexdigest(),
        "shuffle_sequence_sha256": hashlib.sha256(shuffled.encode()).hexdigest(),
        "n_nt": len(positive_sequence),
        "positive_posterior": positive_posterior,
        "shuffle_posterior": shuffle_posterior,
        "margin": positive_posterior - shuffle_posterior,
        "thresholds": {
            "min_positive": CONTROL_MIN_POSITIVE,
            "max_shuffle": CONTROL_MAX_SHUFFLE,
            "min_margin": CONTROL_MIN_MARGIN,
        },
        "flags": flags,
        "adapter_sha256": record["adapter_sha256"],
        "heads_sha256": record["heads_sha256"],
        "green": green,
    }


def arms_matching_config(
    production: Mapping[str, float], *, sweep_dir: str | Path = DEFAULT_SWEEP_DIR
) -> list[str]:
    """Every arm whose own run report matches ``production``'s ``(aux_weight, lr)``.

    Reads the git-tracked sweep reports directly rather than through
    ``eval.discover_arms``, which requires the DVC-tracked checkpoint ROOT to exist and is
    therefore False-in-CI. Returned as a list so a configuration matching two arms (or
    none) is visible to the caller instead of silently resolving to one.
    """
    matches = []
    for path in sorted(Path(sweep_dir).glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report.get("config") or {}
        if (config.get("loss") or {}).get("aux_weight") == production.get(
            "aux_weight"
        ) and config.get("lr") == production.get("lr"):
            matches.append(path.stem)
    return matches


def sweep_fingerprint(arm: str, *, sweep_dir: str | Path = DEFAULT_SWEEP_DIR) -> dict[str, Any]:
    """The git-tracked identity of an arm's training run.

    The checkpoint bytes are DVC-tracked and therefore absent in CI, so the derivation
    cannot hash them. This is the substitute that *is* present in every checkout: a
    re-trained arm writes a new run report, so its ``saved_val_total`` /
    ``saved_from_epoch`` move and a control certified against the old weights stops
    matching.
    """
    report = json.loads(Path(sweep_dir, f"{arm}.json").read_text(encoding="utf-8"))
    config = report.get("config") or {}
    legacy = report.get("legacy") or {}
    wrap = report.get("wrap") or {}
    return {
        "arm": arm,
        "aux_weight": (config.get("loss") or {}).get("aux_weight"),
        "lr": config.get("lr"),
        "saved_val_total": legacy.get("saved_val_total"),
        "saved_from_epoch": legacy.get("saved_from_epoch"),
        "attn_implementation": wrap.get("attn_implementation"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The supply derivation — the constant's independent re-derivation
# ═════════════════════════════════════════════════════════════════════════════
def derive_stage2_supply_available(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Re-derive, from the shipped evidence, whether the Stage-2 posterior supply exists.

    :data:`~tbox_finder.mining.remine.STAGE2_SUPPLY_AVAILABLE` is a hand-written
    declaration; this is the independent re-derivation it is checked against, so the
    constant cannot drift in **either** direction — the P3-15′-a discipline, and the
    reason the unit pin asserts agreement rather than a literal ``True``.

    Every artifact read is **git-tracked and neither DVC- nor LFS-shaped**, so CI, the
    laptop and the cluster answer identically. That rules out the obvious clause "the
    checkpoint is on disk": the checkpoints are DVC-tracked, so such a clause would be
    False in CI and the pin would fail there and nowhere else. Its job is done instead by
    ``control_matches_this_calibration``, which ties the committed control to the arm's
    committed run report.

    Six clauses, each fail-closed and each independently breakable:

    ``gate2_calibration_wellformed``
        the GATE-2 report carries a positive temperature, grades
        ``named_posterior``, and did **not** apply the prior shift — i.e. the object
        this producer emits is the object D11 pins and GATE-2 graded.
    ``production_arm_on_record``
        the arm ``conf/`` names is the arm GATE-2 scored, and its own run report is
        present and agrees on ``(aux_weight, lr)``.
    ``producer_present``
        this module imports and exposes every entry in :data:`PRODUCER_ENTRY_POINTS`.
    ``producer_posterior_wired``
        a produced posterior actually **reaches** the candidate's evidence. Measured by
        *calling* ``remine.remine_candidate_evidence``, not by reading a signature: a
        producer whose output the round drops on the floor is exactly the no-op this
        gate exists to refuse.
    ``control_green``
        the committed designed control separated a real T-box from its matched
        dinucleotide shuffle, **and** reports its null as matched and not a copy.
    ``control_matches_this_calibration``
        that control was scored at the temperature the GATE-2 report carries *today*,
        against the arm whose run report is on disk *today*. A re-fit calibration or a
        re-trained arm must not inherit a green it never earned.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    clauses: dict[str, bool] = {}
    reasons: list[str] = []

    def _fail(clause: str, why: str) -> bool:
        reasons.append(f"{clause}: {why}")
        return False

    # ── clause 1: the calibration this producer rides on ─────────────────────
    gate2 = _read_json_or_none(root / DEFAULT_GATE2_REPORT)
    gate = gate2.get("gate") if isinstance(gate2, Mapping) else None
    gate = gate if isinstance(gate, Mapping) else {}
    calibration = gate.get("calibration") if isinstance(gate.get("calibration"), Mapping) else {}
    temperature = calibration.get("temperature")
    failures: list[str] = []
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        failures.append(f"gate.calibration.temperature is {temperature!r}, not a real number")
    elif not float(temperature) > 0.0:
        failures.append(f"gate.calibration.temperature is {temperature!r}, not positive")
    if gate.get("graded_posterior_key") != "named_posterior":
        failures.append(
            f"gate.graded_posterior_key is {gate.get('graded_posterior_key')!r}, not "
            "'named_posterior' — this producer emits the named object"
        )
    if gate.get("prior_shift_applied") is not False:
        failures.append(
            f"gate.prior_shift_applied is {gate.get('prior_shift_applied')!r}; D11 names the "
            "PRE-prior-shift posterior"
        )
    clauses["gate2_calibration_wellformed"] = (
        True if not failures else _fail("gate2_calibration_wellformed", "; ".join(failures))
    )

    # ── clause 2: the arm conf names is the arm GATE-2 scored ────────────────
    scoring = gate2.get("scoring") if isinstance(gate2, Mapping) else None
    scored_arm = scoring.get("arm") if isinstance(scoring, Mapping) else None
    try:
        production = production_arm_config()
        fingerprint = sweep_fingerprint(str(scored_arm), sweep_dir=root / DEFAULT_SWEEP_DIR)
        # The arm NAME conf/ would select, not merely the (aux_weight, lr) of the arm
        # GATE-2 happens to name: without this the clause compared a pair to itself and
        # could not notice that conf/ selects a *different* arm than the one graded.
        selected = arms_matching_config(production, sweep_dir=root / DEFAULT_SWEEP_DIR)
    except Exception as exc:  # noqa: BLE001 - unreadable evidence is a FAILED clause
        fingerprint = None
        production = None
        selected = None
        clauses["production_arm_on_record"] = _fail(
            "production_arm_on_record", f"could not read the arm's own run report: {exc!r}"
        )
    if fingerprint is not None and production is not None:
        mismatched = [
            f"{key}={fingerprint.get(key)!r} != conf {production.get(key)!r}"
            for key in ("aux_weight", "lr")
            if fingerprint.get(key) != production.get(key)
        ]
        if mismatched:
            clauses["production_arm_on_record"] = _fail(
                "production_arm_on_record",
                f"arm {scored_arm!r} disagrees with conf/: {'; '.join(mismatched)}",
            )
        elif selected != [scored_arm]:
            clauses["production_arm_on_record"] = _fail(
                "production_arm_on_record",
                f"conf/ selects {selected!r} but GATE-2 scored {scored_arm!r} — the graded "
                "arm and the shipped configuration name different checkpoints",
            )
        else:
            clauses["production_arm_on_record"] = True

    # ── clause 3: the producer ships ─────────────────────────────────────────
    try:
        from tbox_finder.mining import stage2_producer
    except Exception as exc:  # noqa: BLE001 - a broken producer is a FAILED clause
        # Deliberately broader than ImportError: a module-level RuntimeError/OSError
        # anywhere in the transitive chain would otherwise propagate out of a function
        # documented as fail-closed on every clause.
        stage2_producer = None
        clauses["producer_present"] = _fail("producer_present", f"import failed: {exc!r}")
    if stage2_producer is not None:
        missing = [e for e in PRODUCER_ENTRY_POINTS if not hasattr(stage2_producer, e)]
        clauses["producer_present"] = (
            True if not missing else _fail("producer_present", f"stage2_producer lacks {missing}")
        )

    # ── clause 4: a produced posterior reaches the candidate ─────────────────
    probe_id = "__stage2_supply_probe__"
    probe_value = 0.875
    try:
        from tbox_finder.mining.remine import remine_candidate_evidence

        stamped = remine_candidate_evidence(
            probe_id, covariation_status=None, stage2_posteriors={probe_id: probe_value}
        ).stage2_posterior
    except Exception as exc:  # noqa: BLE001 - an unreachable consumer is a FAILED clause
        # Same rule as clauses 2 and 3: a consumer that cannot even be imported is the
        # state this clause exists to report, not a traceback out of a function whose
        # contract is "fail-closed on every clause".
        clauses["producer_posterior_wired"] = _fail(
            "producer_posterior_wired", f"the round's evidence builder is unreachable: {exc!r}"
        )
    else:
        clauses["producer_posterior_wired"] = (
            True
            if stamped == probe_value
            else _fail(
                "producer_posterior_wired",
                f"a produced posterior of {probe_value} reached the candidate as "
                f"{stamped!r} — the producer's output is not composed into the round",
            )
        )

    # ── clauses 5-6: the must-fire control, and that it names this calibration ─
    control = _read_json_or_none(root / CONTROL_REPORT)
    if not isinstance(control, Mapping) or not control:
        clauses["control_green"] = _fail(
            "control_green", f"{CONTROL_REPORT} is missing or malformed"
        )
        clauses["control_matches_this_calibration"] = _fail(
            "control_matches_this_calibration", f"{CONTROL_REPORT} is missing or malformed"
        )
    else:
        flags = control.get("flags") if isinstance(control.get("flags"), Mapping) else {}
        control_failures: list[str] = []
        positive = control.get("positive_posterior")
        shuffle = control.get("shuffle_posterior")
        if not _is_real_number(positive) or float(positive) < CONTROL_MIN_POSITIVE:
            control_failures.append(f"positive_posterior={positive!r} below {CONTROL_MIN_POSITIVE}")
        if not _is_real_number(shuffle) or float(shuffle) > CONTROL_MAX_SHUFFLE:
            control_failures.append(f"shuffle_posterior={shuffle!r} above {CONTROL_MAX_SHUFFLE}")
        if (
            _is_real_number(positive)
            and _is_real_number(shuffle)
            and (float(positive) - float(shuffle)) < CONTROL_MIN_MARGIN
        ):
            control_failures.append(
                f"margin {float(positive) - float(shuffle)} below {CONTROL_MIN_MARGIN} — two "
                "thresholds without a separation can both be met by a flat scorer"
            )
        unmatched = [k for k in REQUIRED_CONTROL_FLAGS if not flags.get(k)]
        if unmatched:
            control_failures.append(
                f"the null is not a matched control (missing counts as unmatched): {unmatched}"
            )
        if control.get("green") is not True:
            control_failures.append(f"green={control.get('green')!r}")
        if not isinstance(control.get("n_control_rows"), int) or control["n_control_rows"] < 2:
            control_failures.append(
                f"n_control_rows={control.get('n_control_rows')!r} — a control needs both arms"
            )
        clauses["control_green"] = (
            True if not control_failures else _fail("control_green", "; ".join(control_failures))
        )

        tie_failures: list[str] = []
        if not isinstance(temperature, (int, float)) or control.get("temperature") != temperature:
            tie_failures.append(
                f"control scored at T={control.get('temperature')!r}, GATE-2 carries "
                f"{temperature!r} — the calibration was re-fitted after the control"
            )
        if control.get("arm") != scored_arm:
            tie_failures.append(f"control arm {control.get('arm')!r} != GATE-2 arm {scored_arm!r}")
        if fingerprint is None or control.get("sweep_fingerprint") != fingerprint:
            tie_failures.append(
                f"control fingerprint {control.get('sweep_fingerprint')!r} != on-disk "
                f"{fingerprint!r} — the arm was re-trained after the control"
            )
        for key in ("adapter_sha256", "heads_sha256"):
            if not control.get(key):
                tie_failures.append(f"control carries no {key}")
        clauses["control_matches_this_calibration"] = (
            True
            if not tie_failures
            else _fail("control_matches_this_calibration", "; ".join(tie_failures))
        )

    # A clause whose branch never ran is ABSENT, and `all(...)` skips what is not there,
    # so an unreported clause would read as a passing one. Filled from the named set rather
    # than from whatever the body happened to write.
    for clause in SUPPLY_CLAUSES:
        if clause not in clauses:
            clauses[clause] = _fail(clause, "the clause did not report a verdict")

    return {
        "available": all(clauses.values()),
        "clauses": {name: clauses[name] for name in SUPPLY_CLAUSES},
        "reasons": reasons,
        "repo_root": str(root),
    }


def _is_real_number(value: Any) -> bool:
    """A real number, ``bool`` excluded — the same rule :func:`validate_posteriors` uses.

    ``isinstance(True, int)`` is True, so a control record carrying
    ``"positive_posterior": true`` / ``"shuffle_posterior": false`` clears both thresholds
    AND the margin (1.0 - 0.0) and certifies a supply on two booleans. The two readers of
    this same quantity must not disagree about what counts as a number.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_json_or_none(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def write_json(path: str | Path, payload: Any) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Path(path)


def _add_strand_policy(parser: argparse.ArgumentParser) -> None:
    """``--strand-policy``: **required, no default**.

    The manifest carries no strand and Stage-1 is RC-equivariant, so the emitted object
    genuinely differs between the two readings — measured on the real 941, 119 candidates
    change their verdict at a 0.9 operating point. A default here would let the value
    that decides which loci get mined be supplied by nobody, which is precisely what
    ``--stage2-threshold``'s ``required=True`` refuses one layer up.
    """
    parser.add_argument(
        "--strand-policy",
        required=True,
        choices=list(STRAND_POLICIES),
        help=(
            "which orientation's posterior is published (no default; both are always "
            "scored and both ride in the report)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tbox_finder.mining.stage2_producer",
        description="P3-15′-b per-candidate Stage-2 posterior producer (ADR-0005 D14/D11).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    shards = sub.add_parser("make-shards", help="partition the FP manifest into shard files")
    shards.add_argument("--manifest", default=DEFAULT_FP_MANIFEST)
    shards.add_argument("--n-shards", type=int, required=True)
    shards.add_argument("--out-dir", required=True)

    score = sub.add_parser("score-shard", help="resolve → transcribe → score → calibrate")
    score.add_argument("--shard", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--genome-dir", default=GENOME_DIR)
    score.add_argument("--checkpoint-root", default=DEFAULT_CKPT_ROOT)
    score.add_argument("--sweep-dir", default=DEFAULT_SWEEP_DIR)
    score.add_argument("--gate2-report", default=DEFAULT_GATE2_REPORT)
    score.add_argument("--batch-size", type=int, default=4)
    score.add_argument("--device", default=None)
    _add_strand_policy(score)

    merge = sub.add_parser("merge", help="shard tables → the round's stage2_posteriors.json")
    merge.add_argument("--tables", nargs="+", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--manifest", default=DEFAULT_FP_MANIFEST)
    merge.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help=(
            "refuse below this scored fraction (default 1.0 — every candidate resolved; "
            "an unscored candidate is SPARED, so a gap is silent)"
        ),
    )

    control = sub.add_parser("control", help="the designed control that must fire")
    control.add_argument("--out", default=CONTROL_REPORT)
    control.add_argument("--positive-csv", required=True)
    control.add_argument("--positive-name", required=True)
    control.add_argument("--seed", type=int, required=True)
    control.add_argument("--checkpoint-root", default=DEFAULT_CKPT_ROOT)
    control.add_argument("--sweep-dir", default=DEFAULT_SWEEP_DIR)
    control.add_argument("--gate2-report", default=DEFAULT_GATE2_REPORT)
    control.add_argument("--device", default=None)
    return parser


def _cmd_make_shards(args: argparse.Namespace) -> int:
    specs = read_candidate_manifest(args.manifest)
    written = write_shards(specs, args.n_shards, args.out_dir)
    print(f"{len(specs)} candidates → {len(written)} shards under {args.out_dir}")
    return 0


def _cmd_score_shard(args: argparse.Namespace) -> int:
    specs = read_candidate_manifest(args.shard)
    table = score_shard(
        specs,
        temperature=read_temperature(args.gate2_report),
        strand_policy=args.strand_policy,
        genome_dir=args.genome_dir,
        checkpoint_root=args.checkpoint_root,
        sweep_dir=args.sweep_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    write_json(args.out, table)
    print(
        f"scored {table['n_scored']}/{table['n_candidates']} "
        f"({table['n_unresolved']} unresolved) → {args.out}"
    )
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    n_candidates = len(read_candidate_manifest(args.manifest))
    try:
        table = merge_posterior_tables(
            args.tables, n_candidates=n_candidates, min_coverage=args.min_coverage
        )
    except Stage2ProducerError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 3
    write_json(args.out, table)
    _write_provenance(args.out, table, tables=args.tables, manifest=args.manifest)
    print(
        f"merged {table['n_shards']} shards → {table['n_scored']}/{n_candidates} "
        f"(coverage {table['coverage']:.4f}) → {args.out}"
    )
    return 0


def _cmd_control(args: argparse.Namespace) -> int:
    record = run_control(
        read_control_positive(args.positive_csv, name=args.positive_name),
        temperature=read_temperature(args.gate2_report),
        seed=args.seed,
        checkpoint_root=args.checkpoint_root,
        sweep_dir=args.sweep_dir,
        device=args.device,
    )
    record["positive_csv"] = repo_relative(args.positive_csv)
    record["positive_name"] = args.positive_name
    # Written BEFORE the verdict is acted on: a control that refused without leaving a
    # record would destroy the evidence of the one failure it exists to surface.
    write_json(args.out, record)
    print(
        f"control: positive={record['positive_posterior']:.6f} "
        f"shuffle={record['shuffle_posterior']:.6f} margin={record['margin']:.6f} "
        f"green={record['green']}"
    )
    if not record["green"]:
        print(
            "FATAL: the designed control did not fire — the producer cannot certify",
            file=sys.stderr,
        )
        return 3
    return 0


def _write_provenance(
    out: str | Path, table: Mapping[str, Any], *, tables: Sequence[str], manifest: str
) -> None:
    from tbox_finder.provenance import build_provenance

    # `inputs` only: `build_provenance` hashes its `outputs`, and the merged table is
    # written by the caller — passing it would hash a path at the END of a long run.
    record = build_provenance(
        rule="stage2_producer::merge",
        script="src/tbox_finder/mining/stage2_producer.py",
        inputs=[manifest, *tables],
        env_lock=ENV_LOCK,
        adr=ADR,
        extra={
            "step": STEP,
            "arm": table.get("arm"),
            "temperature": table.get("temperature"),
            "temperature_source": table.get("temperature_source"),
            "strand_policy": table.get("strand_policy"),
            "posterior_kind": table.get("posterior_kind"),
            "n_scored": table.get("n_scored"),
            "n_candidates": table.get("n_candidates"),
            "coverage": table.get("coverage"),
            "load": table.get("load"),
        },
    )
    write_json(Path(out).with_name("provenance.json"), record)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "make-shards": _cmd_make_shards,
        "score-shard": _cmd_score_shard,
        "merge": _cmd_merge,
        "control": _cmd_control,
    }[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
