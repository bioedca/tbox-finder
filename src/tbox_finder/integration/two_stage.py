"""P3-14 — the end-to-end two-stage harness: contig in, calibrated candidate table + the
ADR-0005 D15 strand-robustness diagnostic out.

What this module is
-------------------
The single composition site for the PRD §6 path::

    Stage-1 window logits
      → reconcile_windows          (P2-03, the operator ADR-0005 A3 froze in code)
      → construct_loci             (P3-12, D3's five locus knobs)
      → resolve_strands            (P3-13, D15's element-order rule)
      → handoff_loci               (P3-13, T→U, sequence-only, both strands when ambiguous)
      → Stage-2 binary logit       (P3-06 LoRA checkpoint)
      → temperature-scale          (P3-07 stack; the GATE-2 *named* posterior)
      → candidate table            (§13.1 columns, incl. strand + low-order-confidence)

**Nothing here re-derives an operator.** Every arrow above is an existing, tested entry point
called by name; this module supplies the plumbing, the table schema and the diagnostic. That is
the same reading P3-11 applied to the reconciliation operator and P3-12 applied to the merge:
ADR-0005 pins *one* of each, and a second copy means fixing one and shipping the bug in the
other. The one composition that could not be borrowed — "score the strand the resolver did
*not* choose" — is expressed as
``dataclasses.replace(call, strands=BOTH_STRANDS)`` fed back through the shipped
:func:`~tbox_finder.infer.handoff.handoff_loci`, so even the counterfactual payloads come out of
the production transcriber.

Torch-free by construction, and that is load-bearing
----------------------------------------------------
The module imports ``numpy`` only. The two model legs are **CLI subcommands** that import torch
lazily and write a committed artifact each (``stage1`` → per-window logits, ``stage2`` → binary
logits); the ``run`` leg — the one CI executes and the one the golden regression locks — replays
those artifacts and is pure numpy. CI installs no torch at all, so a harness that needed it
could not be regression-tested; and the two legs need *different, mutually incompatible* envs
anyway (``ml-dna`` carries ``mamba-ssm`` for Caduceus, ``ml-rna`` carries ``multimolecule`` for
RiNALMo), so a single in-process end-to-end call is not merely inconvenient, it is impossible.
The split is the same 3-env shape ``mining/covariation_producer.py`` already uses.

Replay is **fail-closed and content-addressed**: every payload is keyed by
:func:`payload_key`, a hash over the contig, the locus index, the strand, the span and the RNA
itself. A change anywhere upstream changes the key, and a missing key raises
:class:`TwoStageError` naming it rather than scoring the payload with a stale neighbour's
number. A replay that silently substituted a score would be a fabricated result (§10.3), and a
lookup that *could not miss* would be a golden that locks nothing.

This module pins NO value
-------------------------
ADR-0005 D3 freezes the Stage-1 threshold, all five locus knobs **and the Stage-2 operating
point** at the §13.1 phase gate (P5-01); D15 freezes the strand rule's ``min_order_margin``
there too; the temperature is a *fitted* quantity (P3-10 fitted ``T`` on the D11 calibration
carve) and inventing one would be a fabricated calibration. Accordingly **every knob on
:func:`run_two_stage` is keyword-only with no default**, including ``source_prior`` /
``target_prior``, which must be passed as an explicit ``None`` to decline the §11 prior-shift
rather than declining it by omission. :func:`no_rule_parameter_has_a_default` re-derives that
from the live signature.

The strand-robustness diagnostic, and what it does NOT measure
--------------------------------------------------------------
ADR-0005 D15 specifies "a **strand-robustness diagnostic** (fraction of confirmed loci
*tier*-invariant to strand re-resolution)". The §13.3 tiers do not exist until P6 — there is no
orthogonal-validation pipeline to be invariant *of* at P3 — so what is measured here is the
**two-stage-confirmation** analogue: for every locus, Stage-2 scores *both* strands, and the
locus is *confirmation-invariant* when the calibrated verdict is the same either way. The report
carries ``tier_invariance: null`` with that reason beside the number it does compute, so the
substitution is a disclosure rather than a rename; re-reading it against real tiers is owed at
P6.

Two liveness reads ship with it, because "invariant" and "blind" produce the same 1.0. If
Stage-2 scored a sequence and its reverse complement identically the diagnostic would read a
perfect 1.0 while measuring nothing at all — the shape that let a mask run clean while masking
nothing. So ``max_abs_posterior_delta`` and ``n_verdict_disagreements`` are reported alongside,
and the golden asserts both are non-zero on the fixture: a designed control that **must** fire.

Where the fixture's ground truth is available (the committed contigs are real genomic context
minted from records of known orientation) the report also counts
``n_confirmed_on_wrong_strand_only`` — the one outcome D15 says must not happen, since a
mis-resolution is required to degrade to a bounded false negative and never to a false-novelty
claim.

Not an evaluation
-----------------
Everything this module writes is a **harness regression artifact** measured on a 20-contig
fixture: ``is_science: false``, ``gated: false``. The two-stage-vs-Stage-1-only precision
comparison is P3-16 and the genome-scale scan is P5; no number produced here is a performance
claim, and the report says so in ``disclosures``.

Stage-2 identity and the shard record (P3-16′-remint, ADR-0005 A15 Pin 6)
-------------------------------------------------------------------------
A Stage-2 logit is a function of more than its payload. It also depends on the checkpoint,
which is ``checkpoint_root`` plus ``arm``: P3-17 added an RNA-FM arm under the same name,
``aux1.0_lr1e-4``, so the arm name alone does not identify it. It depends on the backbone and
its revision, the dtype, the adapter **and** head weights, and the attention backend. It even
depends on the **batch** the payload was scored in: A15 records 147 of 4,148 payloads shared
by both v1 arms receiving different logits. So the ``stage2`` leg writes a schema-``1.1``
artifact that names every one of those. Its ``env_lock`` is derived from the backbone through
the ADR-0002 A15 allow-list rather than stamped as a constant, because the comparator runs
under a different lock.

``--shard I/N`` scores one contiguous slice of the row-id-sorted manifest and records which
slice. The torch-free ``merge-stage2`` leg reassembles the slices, and refuses unless every
shard is the same measurement, the indices are exactly ``0..N-1``, and each shard holds exactly
the rows its index assigns. A merged artifact records its ``shard_layout``: a slice is batched
differently from the whole manifest, so its logits can differ from an unsharded run's.

PRD §6, §12, §13.1; ADR-0005 D3 + A3 + D15 + A15.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from tbox_finder.calib.recalibrate import calibrated_posterior
from tbox_finder.eval.gate4 import json_safe
from tbox_finder.infer.call import rule_parameters_have_no_default
from tbox_finder.infer.handoff import Handoff, HandoffResult, handoff_loci
from tbox_finder.infer.locus import Locus, ThresholdScope, construct_loci
from tbox_finder.infer.reconcile import Reconciled, reconcile_windows
from tbox_finder.infer.strand import BOTH_STRANDS, STRAND_MINUS, STRAND_PLUS, StrandCall
from tbox_finder.infer.strand import resolve_strands as _resolve_strands
from tbox_finder.ingest import record_hash, records_digest
from tbox_finder.labels import CLASS_ORDER

# Torch-free by its own contract: the registry carries the multimolecule class as a string.
from tbox_finder.models import rna_backbone_registry as BR

SCHEMA_VERSION = "1.0"
STEP = "P3-14"
GENERATED_BY = "src/tbox_finder/integration/two_stage.py"
ADR = "ADR-0005 D3 (+A3 reconciliation operator), D15 (strand resolver + robustness diagnostic)"
PRD = "PRD §6, §12, §13.1"

#: The env the **replay** leg runs in — the only leg CI executes and the one that writes the
#: committed report. The Stage-1 leg runs in ``envs/ml-dna.yml``. The Stage-2 leg runs under
#: the lock of the backbone it scores with, which :func:`stage2_env_lock` derives and the
#: artifact records.
ENV_LOCK = "envs/data.conda-lock.yml"

#: Schema of what the ``stage2`` and ``merge-stage2`` legs write. It is kept apart from
#: :data:`SCHEMA_VERSION` on purpose. The harness report, the candidate table and the payload
#: manifest did not change shape, and bumping their version would move the committed golden
#: for a change it does not contain. ``1.0`` Stage-2 files carry none of the identity fields
#: below; that covers P3-14's fixture and P3-16's v1 interim scores. :func:`read_stage2` still
#: reads them, but they cannot be merged.
STAGE2_SCHEMA_VERSION = "1.1"

#: What makes two Stage-2 scorings **the same measurement** (ADR-0005 A15 Pin 6). Each one
#: moves the logits on its own. ``checkpoint_root`` is listed beside ``arm`` because P3-17's
#: RNA-FM arm shares the name ``aux1.0_lr1e-4``. ``batch_size`` is listed because scores
#: depend on batch composition. ``payloads_sha256`` is listed because a shard is defined
#: relative to one manifest. :func:`merge_stage2_shards` refuses any disagreement, and refuses
#: a ``None`` in any of these fields: equality over all-missing values certifies nothing.
STAGE2_IDENTITY_FIELDS: tuple[str, ...] = (
    "arm",
    "checkpoint_root",
    "checkpoint_dir",
    "backbone",
    "revision",
    "dtype",
    "adapter_sha256",
    "heads_sha256",
    "attn_implementation",
    "batch_size",
    "payloads_sha256",
    "env_lock",
)

#: Run-record fields every shard must also agree on before a merge. They are not identity in
#: the A15 sense, but a merge mixing two of them would be one artifact describing two runs.
#: ``device`` belongs here because the kernels, and so the last bits of a logit, are the
#: device's.
STAGE2_RUN_FIELDS: tuple[str, ...] = (
    "schema_version",
    "step",
    "arm_selected_by",
    "base_model_name_or_path",
    "n_lora_b_nonzero",
    "attn_implementation_matches_training",
    "device",
)

#: Every key of a schema-``1.1`` Stage-2 artifact, whether it is unsharded, one shard, or
#: merged. ``shard`` is ``None`` unless the file is a single shard. ``shard_layout`` is
#: ``None`` unless the file is a merge. So a merged file has the same keys as an unsharded
#: one.
STAGE2_ARTIFACT_KEYS: tuple[str, ...] = (
    *STAGE2_IDENTITY_FIELDS,
    *STAGE2_RUN_FIELDS,
    "shard",
    "shard_layout",
    "n_payloads",
    "logits",
    "n_tokens",
)

#: Every keyword-only knob on :func:`run_two_stage`. All of them are frozen elsewhere — the six
#: locus knobs and the Stage-2 operating point by ADR-0005 D3 at the §13.1 phase gate,
#: ``min_order_margin`` by D15 at the same gate, ``temperature`` by the P3-10 fit on the D11
#: calibration carve, and the two priors by whatever deployment prior a P5 corpus declares
#: (PRD §11 pins none). Compared as a **set** by
#: :func:`no_rule_parameter_has_a_default`, so a knob added later without an entry here fails
#: the inventory instead of arriving with a default nobody signed off.
RULE_PARAMETERS: tuple[str, ...] = (
    "threshold_scope",
    "threshold",
    "min_span",
    "gap_merge",
    "min_distinct_elements",
    "flank",
    "min_order_margin",
    "temperature",
    "stage2_operating_point",
    "source_prior",
    "target_prior",
)

#: Candidate-table columns, in order. One row per **emitted** payload — so an ambiguous locus
#: contributes two rows, which is D15's both-strand carry-through made visible in the §13.1
#: table rather than hidden behind a flag.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "contig_id",
    "contig_length",
    "locus_index",
    "payload_key",
    "core_start",
    "core_end",
    "core_length",
    "span_start",
    "span_end",
    "span_length",
    "flank",
    "flank_short_left",
    "flank_short_right",
    "element_classes",
    "n_distinct_elements",
    "dominant_class",
    "peak_p_elem",
    "mean_p_elem",
    "n_zero_flanked_core",
    "n_zero_flanked_span",
    "n_single_covered_span",
    "overlaps_neighbour",
    "strand",
    "strands_carried",
    "low_order_confidence",
    "strand_reason",
    "order_margin",
    "n_concordant",
    "n_discordant",
    "n_tied_position",
    "n_observed_elements",
    "rna_length",
    "n_soft_masked",
    "n_ambiguous",
    "rna_sha256",
    "stage2_logit",
    "stage2_named_posterior",
    "stage2_prior_shifted_posterior",
    "confirmed",
    "contig_n_outside_alphabet",
)

#: Float quantum for the golden digest. Floats are hashed as ``round(x * DIGEST_QUANTUM)``
#: rather than by ``repr``: the table's floats come out of ``exp``/``log`` over committed
#: inputs, and the last ulp of those is a property of the host's libm, not of the harness. Two
#: machines must agree on the committed hash or the golden is a machine-identity test. The cost
#: is a stated tolerance — a change below 1e-6 does not move the digest — and
#: ``test_digest_moves_on_a_shift_just_above_the_stated_tolerance`` pins that the tolerance is
#: not wider than advertised.
DIGEST_QUANTUM = 1_000_000

#: Log-probability tolerance the ``stage1`` mint leg holds its stored logits to, against what
#: ``scan_encoded_windows`` returns for the same contig. It is **not** zero and cannot be: the
#: fixture stores float16 to keep the committed archive at ~510 kB, so the reconciled posterior
#: differs in the last few digits by construction. **Measured over all 20 fixture contigs: max
#: |Δ log-prob| = 4.11e-3, with 0 per-position prediction mismatches** — the posterior moves, the
#: *call* does not. This bound sits ~12× above that and far below anything a locus threshold can
#: see, so it catches a real divergence (a wrong checkpoint, a wrong tiling) while tolerating the
#: quantisation. An earlier docstring claimed the check was "bit-identical"; it was not, and
#: CodeRabbit r1 was right to say so.
#:
#: **Stage-1 is not run-to-run bit-identical, and it was measured.** Re-running the ``stage1``
#: leg on the identical contigs changes **91 of 327,680** stored logits by up to 3.91e-3
#: (0.028 %) — while moving **zero** per-position predictions, so the *call* is stable and the
#: posterior is not. That alone is why the model output is committed and replayed rather than
#: re-run: a golden that re-ran Stage-1 would be flaky by construction.
#:
#: **Stage-2 is a different and narrower story, and the distinction matters.** Re-running the
#: ``stage2`` leg over the same manifest reproduces its artifact **byte-identically** (measured,
#: twice). What is *not* invariant is batch position: the same RNA reached
#: ``eval.score_rows`` from two contigs, landed at two places in its length-sorted batching, and
#: differed by up to 0.0292 of logit — which is what
#: ``determinism.max_abs_duplicate_logit_delta`` records. So Stage-2 is reproducible for a fixed
#: work list and sensitive to how that work list is batched, and a re-batched scoring of the
#: same sequences is not guaranteed to agree with this fixture.
MAX_MINT_LOG_PROB_DELTA = 0.05

#: Contig-record keys. ``truth_*`` are optional and present only because the committed fixture
#: was minted from records of known orientation; production scanning has none, and every read
#: that consumes them degrades to ``None`` rather than to a default.
CONTIG_REQUIRED_KEYS: tuple[str, ...] = ("contig_id", "sequence")
CONTIG_TRUTH_KEYS: tuple[str, ...] = (
    "truth_strand",
    "truth_start",
    "truth_end",
    "truth_source_record_id",
    "truth_transform",
)


class TwoStageError(ValueError):
    """A harness composition failure — a malformed contig record, or a missing replay score.

    Distinct from the operator errors it composes (``CandidateError`` / ``LocusError`` /
    ``StrandError`` / ``HandoffError``), all of which also subclass ``ValueError``: those name a
    bad *input to an operator*, this names a bad *composition*.
    """


# ══════════════════════════════════════════════════════════════════════════════════════
# Payload identity
# ══════════════════════════════════════════════════════════════════════════════════════
def payload_key(contig_id: str, payload: Handoff) -> str:
    """The content address of one Stage-2 payload — the replay join key.

    Hashes the contig, the locus index, the strand, the span **and the RNA itself** through
    :func:`tbox_finder.ingest.record_hash`, whose length-prefixed framing already guarantees no
    cell can collide with the separator.

    Including the RNA is the point: a change to the locus rule, the flank, the strand call or
    the transcriber moves the key, so a replayed run cannot pair a *new* payload with an *old*
    score. Including ``contig_id`` is also deliberate even though two contigs can legitimately
    produce the identical RNA — a contig and its reverse complement should, when the resolver
    works — because scoring it from both keeps the artifact's rows one-to-one with the payloads
    and turns the coincidence into a measurable determinism read rather than a silent dedup.
    """
    return record_hash(
        [
            "payload_key/v1",
            str(contig_id),
            int(payload.locus_index),
            str(payload.strand),
            int(payload.start),
            int(payload.end),
            str(payload.rna),
        ]
    )


def no_rule_parameter_has_a_default(func: Any = None) -> bool:
    """True iff every knob on :func:`run_two_stage` is keyword-only with no default.

    Same discipline, same shared implementation
    (:func:`tbox_finder.infer.call.rule_parameters_have_no_default`) as
    :func:`tbox_finder.infer.locus.no_rule_parameter_has_a_default` (D3) and
    :func:`tbox_finder.infer.strand.no_rule_parameter_has_a_default` (D15). ``func`` exists so
    the predicate can be pointed at a stub that *does* carry a default and shown to return
    False.
    """
    return rule_parameters_have_no_default(run_two_stage if func is None else func, RULE_PARAMETERS)


# ══════════════════════════════════════════════════════════════════════════════════════
# Records
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ContigRun:
    """Everything one contig produced, before Stage-2 is joined in.

    Attributes
    ----------
    contig_id, sequence:
        As supplied. ``sequence`` is the frame every coordinate below is in.
    reconciled:
        The :class:`~tbox_finder.infer.reconcile.Reconciled` the operator returned.
    loci:
        Ascending by core start, as :func:`~tbox_finder.infer.locus.construct_loci` returns.
    calls:
        Index-parallel to ``loci``.
    emitted:
        The production handoff — one payload per (locus, *carried* strand).
    both:
        The counterfactual handoff — one payload per (locus, strand) for **both** strands of
        **every** locus, produced by widening ``strands`` on each call and re-entering the same
        shipped :func:`~tbox_finder.infer.handoff.handoff_loci`. ``emitted.payloads`` is a
        subset of it by content, which is why a single Stage-2 scoring pass over ``both``
        serves the table and the diagnostic alike.
    truth:
        The contig record's optional ``truth_*`` keys, carried verbatim. Present only because
        the committed fixture was minted from records of known orientation; a production scan
        supplies none and every consumer degrades to ``None`` rather than to a default. It is
        deliberately **not** an input to any call above — nothing the harness decides may read
        it.
    """

    contig_id: str
    sequence: str
    reconciled: Reconciled
    loci: tuple[Locus, ...]
    calls: tuple[StrandCall, ...]
    emitted: HandoffResult
    both: HandoffResult
    truth: Mapping[str, Any]


@dataclass(frozen=True)
class TwoStageResult:
    """The harness output: the candidate table, its digest, and the diagnostic report."""

    rows: tuple[dict[str, Any], ...]
    digest: str
    report: dict[str, Any]
    runs: tuple[ContigRun, ...]


# ══════════════════════════════════════════════════════════════════════════════════════
# Stage-1 side
# ══════════════════════════════════════════════════════════════════════════════════════
def normalise_contig(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one contig record and fill the optional truth keys with ``None``.

    Refuses an empty sequence and a non-string id rather than letting either reach an operator,
    where the message would name the operator's contract instead of the harness's.
    """
    for key in CONTIG_REQUIRED_KEYS:
        if key not in record:
            raise TwoStageError(f"contig record is missing required key {key!r}")
    contig_id = record["contig_id"]
    if not isinstance(contig_id, str) or not contig_id:
        raise TwoStageError(f"contig_id must be a non-empty string, got {contig_id!r}")
    sequence = record["sequence"]
    if not isinstance(sequence, str) or not sequence:
        raise TwoStageError(f"contig {contig_id!r} carries no sequence")
    out: dict[str, Any] = {"contig_id": contig_id, "sequence": sequence}
    for key in CONTIG_TRUTH_KEYS:
        out[key] = record.get(key)
    truth_strand = out["truth_strand"]
    if truth_strand is not None and truth_strand not in (STRAND_PLUS, STRAND_MINUS):
        raise TwoStageError(
            f"contig {contig_id!r} declares truth_strand={truth_strand!r}; the only orientations "
            f"a locus can be on are {STRAND_PLUS!r}, {STRAND_MINUS!r}, or None (unknown)"
        )
    return out


def normalise_contigs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate a whole contig **set** — each record, and the uniqueness of their ids.

    ``contig_id`` is a key three times over: :func:`payload_key` hashes it, the diagnostic keys
    its per-locus posteriors on ``"<contig_id>\x00<locus_index>"``, and the Stage-1 archive is
    looked up by it. A duplicate therefore does not fail — it **merges**. Two contigs are
    reconciled against one contig's window logits, one contig's locus 0 overwrites the other's,
    and ``strand_robustness`` reads one contig's posteriors while attributing them to both.

    Nothing downstream can catch it, which is why the guard is here: ``n_loci`` and
    ``n_scored_payloads`` are both summed per run, so ``n_scored == 2 × n_loci`` still holds and
    :func:`strand_robustness_problems` returns clean. Reproduced before fixing — two records
    sharing an id produced two rows, an empty problem list and a plausible report. That is the
    one composition failure that yielded a number instead of a :class:`TwoStageError`, which is
    exactly what this module's replay contract says must not happen (CodeRabbit r3).

    The mint script has its own id-uniqueness guard, but that only protects the *committed*
    fixture; every other contig set reaches the harness through here.
    """
    contigs = [normalise_contig(record) for record in records]
    seen: set[str] = set()
    for contig in contigs:
        contig_id = contig["contig_id"]
        if contig_id in seen:
            raise TwoStageError(
                f"contig_id {contig_id!r} appears more than once; it keys the payload hash, the "
                "diagnostic's per-locus map and the Stage-1 lookup, so a duplicate silently "
                "merges two contigs rather than scoring each"
            )
        seen.add(contig_id)
    return contigs


def reconcile_contig(entry: Mapping[str, Any], seq_len: int) -> Reconciled:
    """A committed ``{"logits", "starts"}`` entry → :class:`Reconciled`, via the pinned operator.

    The golden replays *window* logits rather than a stored posterior precisely so the
    reconciliation operator ADR-0005 A3 froze runs inside the regression, instead of being
    assumed correct upstream of it.
    """
    for key in ("logits", "starts"):
        if key not in entry:
            raise TwoStageError(f"stage-1 entry is missing required key {key!r}")
    logits = np.asarray(entry["logits"], dtype=np.float64)
    starts = np.asarray(entry["starts"])
    return reconcile_windows(logits, starts, seq_len)


def run_contig(
    contig: Mapping[str, Any],
    reconciled: Reconciled,
    *,
    threshold_scope: ThresholdScope,
    threshold: float | Mapping[str, float],
    min_span: int,
    gap_merge: int,
    min_distinct_elements: int,
    flank: int,
    min_order_margin: int,
) -> ContigRun:
    """One contig through the whole Stage-1 → handoff chain.

    The strand scope/threshold are the **same objects** passed to ``construct_loci``: D15's rule
    votes over the per-position element assignment the loci were called from, and resolving
    against a differently-thresholded assignment would orient a locus by evidence it was not
    called on.
    """
    record = normalise_contig(contig)
    sequence = record["sequence"]
    loci = construct_loci(
        reconciled.log_probs,
        reconciled.zero_flanked,
        reconciled.coverage,
        threshold_scope=threshold_scope,
        threshold=threshold,
        min_span=min_span,
        gap_merge=gap_merge,
        min_distinct_elements=min_distinct_elements,
        flank=flank,
    )
    calls = _resolve_strands(
        reconciled.log_probs,
        loci,
        threshold_scope=threshold_scope,
        threshold=threshold,
        min_order_margin=min_order_margin,
    )
    emitted = handoff_loci(sequence, loci, calls)
    # The counterfactual: the same calls, carried on both strands. `StrandCall` is a plain
    # frozen dataclass, so `replace` states exactly that and nothing else — `strand`,
    # `low_order_confidence` and every vote count stay as measured, and the payloads still come
    # out of the shipped transcriber rather than a second copy of it.
    both = handoff_loci(sequence, loci, [replace(call, strands=BOTH_STRANDS) for call in calls])
    return ContigRun(
        contig_id=record["contig_id"],
        sequence=sequence,
        reconciled=reconciled,
        loci=tuple(loci),
        calls=tuple(calls),
        emitted=emitted,
        both=both,
        truth={key: record[key] for key in CONTIG_TRUTH_KEYS},
    )


def payload_manifest(runs: Sequence[ContigRun]) -> list[dict[str, Any]]:
    """The Stage-2 work list: every (locus, strand) payload of every contig, keyed and ordered.

    Emitted over ``both`` rather than ``emitted`` so one Stage-2 pass serves both the candidate
    table and the counterfactual half of the diagnostic. Ordered by (contig, locus, strand) so
    the manifest — and therefore the batches the scorer forms from it — is deterministic. The
    contig sort is applied **here** rather than assumed of the caller: ``handoff_loci`` fixes
    locus and strand order, but nothing fixed contig order, so an unsorted input produced a
    manifest and a digest that did not match the ordering this docstring claims (CodeRabbit r2).
    """
    manifest: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda run: run.contig_id):
        for payload in run.both.payloads:
            manifest.append(
                {
                    "row_id": payload_key(run.contig_id, payload),
                    "contig_id": run.contig_id,
                    "locus_index": int(payload.locus_index),
                    "strand": payload.strand,
                    "start": int(payload.start),
                    "end": int(payload.end),
                    "rna_sequence": payload.rna,
                }
            )
    return manifest


# ══════════════════════════════════════════════════════════════════════════════════════
# Stage-2 side
# ══════════════════════════════════════════════════════════════════════════════════════
def lookup_logit(stage2_logits: Mapping[str, float], key: str, *, what: str) -> float:
    """Fail-closed replay lookup.

    A missing key means the payload the harness just built is not the payload that was scored —
    the rule changed, the flank changed, the transcriber changed. Raising names it; returning a
    neighbour's number, a zero, or a ``nan`` would put a fabricated score in a candidate table
    (§10.3), and every one of those reads downstream exactly like a measurement.
    """
    if key not in stage2_logits:
        raise TwoStageError(
            f"no Stage-2 logit for {what} (payload_key={key}); the payload this run built is "
            "not one the committed scores cover — re-run the `stage2` leg rather than scoring "
            "it with another payload's number"
        )
    value = float(stage2_logits[key])
    if not np.isfinite(value):
        raise TwoStageError(f"Stage-2 logit for {what} is not finite ({value!r})")
    return value


def posteriors_for(
    logits: Sequence[float],
    *,
    temperature: float,
    source_prior: float | None,
    target_prior: float | None,
) -> dict[str, Any]:
    """Delegate to the P3-07 stack. Not re-implemented here, and not re-ordered here.

    :func:`tbox_finder.calib.recalibrate.calibrated_posterior` owns the pinned order
    (temperature-scale → optional Saerens/Elkan prior-shift) and owns the refusal of a
    half-specified prior pair. It also names which of the two posteriors GATE-2 grades, via
    ``gated_posterior_key`` — the harness reads that key rather than assuming it, so the column
    the candidate table calls ``stage2_named_posterior`` is the object the gate names and not a
    second opinion about which one that is.
    """
    return calibrated_posterior(
        np.asarray(logits, dtype=np.float64),
        temperature=temperature,
        source_prior=source_prior,
        target_prior=target_prior,
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# The candidate table
# ══════════════════════════════════════════════════════════════════════════════════════
def _class_name(index: int | None) -> str | None:
    if index is None or index < 0 or index >= len(CLASS_ORDER):
        return None
    return CLASS_ORDER[index]


def _quantise(value: Any) -> Any:
    """Floats → an integer on the :data:`DIGEST_QUANTUM` grid; everything else untouched."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float | np.floating):
        if not np.isfinite(float(value)):
            raise TwoStageError(f"a candidate-table cell is not finite ({value!r})")
        return int(round(float(value) * DIGEST_QUANTUM))
    if isinstance(value, np.integer):
        return int(value)
    return value


def candidate_table_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """The whole-table content digest — the value committed as ``expected.sha256``.

    House convention (:func:`tbox_finder.ingest.record_hash` per row →
    :func:`tbox_finder.ingest.records_digest` over the ordered row hashes), with every float
    first put on the :data:`DIGEST_QUANTUM` grid. Row order is the table's own and is *not*
    sorted away: the order is (contig, locus, strand) and a run that emitted the right rows in
    the wrong order is a defect the digest should catch.
    """
    hashes = [record_hash([_quantise(row[column]) for column in CANDIDATE_COLUMNS]) for row in rows]
    return records_digest(hashes)


def build_rows(
    runs: Sequence[ContigRun],
    *,
    stage2_logits: Mapping[str, float],
    temperature: float,
    stage2_operating_point: float,
    source_prior: float | None,
    target_prior: float | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """The §13.1 candidate table, plus the per-(locus, strand) posterior map the diagnostic reads.

    Returns ``(rows, both_posteriors)``. ``rows`` covers the **emitted** payloads only — what a
    scan would actually hand on. ``both_posteriors`` covers every (locus, strand) pair, keyed
    ``"<contig_id>\\x00<locus_index>"`` → ``{strand: named_posterior}``, and is the
    counterfactual the strand-robustness diagnostic is computed from.
    """
    # One calibration call over every payload in the run, so the temperature and the prior-shift
    # are applied exactly once and identically to the table and to the counterfactual.
    manifest = payload_manifest(runs)
    raw = [
        lookup_logit(
            stage2_logits,
            entry["row_id"],
            what=f"{entry['contig_id']} locus {entry['locus_index']} strand {entry['strand']}",
        )
        for entry in manifest
    ]
    payload = posteriors_for(
        raw, temperature=temperature, source_prior=source_prior, target_prior=target_prior
    )
    named = np.asarray(payload[payload["gated_posterior_key"]], dtype=np.float64)
    shifted = payload["prior_shifted_posterior"]
    shifted_arr = None if shifted is None else np.asarray(shifted, dtype=np.float64)

    by_key: dict[str, dict[str, Any]] = {}
    both_posteriors: dict[str, dict[str, float]] = {}
    for position, entry in enumerate(manifest):
        by_key[entry["row_id"]] = {
            "logit": raw[position],
            "named": float(named[position]),
            "shifted": None if shifted_arr is None else float(shifted_arr[position]),
        }
        locus_id = f"{entry['contig_id']}\x00{entry['locus_index']}"
        both_posteriors.setdefault(locus_id, {})[entry["strand"]] = float(named[position])

    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda run: run.contig_id):
        for handoff in run.emitted.payloads:
            key = payload_key(run.contig_id, handoff)
            scored = by_key[key]  # emitted ⊂ both by content, so this cannot miss
            locus = run.loci[handoff.locus_index]
            call = run.calls[handoff.locus_index]
            rows.append(
                {
                    "contig_id": run.contig_id,
                    "contig_length": len(run.sequence),
                    "locus_index": int(handoff.locus_index),
                    "payload_key": key,
                    "core_start": int(locus.candidate.start),
                    "core_end": int(locus.candidate.end),
                    "core_length": int(locus.candidate.length),
                    "span_start": int(locus.start),
                    "span_end": int(locus.end),
                    "span_length": int(locus.length),
                    "flank": int(locus.flank),
                    "flank_short_left": int(locus.flank_short_left),
                    "flank_short_right": int(locus.flank_short_right),
                    "element_classes": ",".join(
                        str(_class_name(index)) for index in locus.element_classes
                    ),
                    "n_distinct_elements": int(locus.n_distinct_elements),
                    "dominant_class": _class_name(int(locus.candidate.dominant_class)),
                    "peak_p_elem": float(locus.candidate.peak_p_elem),
                    "mean_p_elem": float(locus.candidate.mean_p_elem),
                    "n_zero_flanked_core": int(locus.candidate.n_zero_flanked),
                    "n_zero_flanked_span": int(locus.n_zero_flanked_span),
                    "n_single_covered_span": (
                        None
                        if locus.n_single_covered_span is None
                        else int(locus.n_single_covered_span)
                    ),
                    "overlaps_neighbour": bool(locus.overlaps_neighbour),
                    "strand": handoff.strand,
                    "strands_carried": ",".join(call.strands),
                    "low_order_confidence": bool(handoff.low_order_confidence),
                    "strand_reason": call.reason,
                    "order_margin": int(call.order_margin),
                    "n_concordant": int(call.n_concordant),
                    "n_discordant": int(call.n_discordant),
                    "n_tied_position": int(call.n_tied_position),
                    "n_observed_elements": int(call.n_distinct_elements),
                    "rna_length": int(handoff.length),
                    "n_soft_masked": int(handoff.n_soft_masked),
                    "n_ambiguous": int(handoff.n_ambiguous),
                    "rna_sha256": hashlib.sha256(handoff.rna.encode("ascii")).hexdigest(),
                    "stage2_logit": scored["logit"],
                    "stage2_named_posterior": scored["named"],
                    "stage2_prior_shifted_posterior": scored["shifted"],
                    "confirmed": bool(scored["named"] >= stage2_operating_point),
                    "contig_n_outside_alphabet": int(run.emitted.n_outside_alphabet),
                }
            )
    return rows, both_posteriors


# ══════════════════════════════════════════════════════════════════════════════════════
# The diagnostic
# ══════════════════════════════════════════════════════════════════════════════════════
def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def strand_robustness(
    runs: Sequence[ContigRun],
    both_posteriors: Mapping[str, Mapping[str, float]],
    *,
    stage2_operating_point: float,
) -> dict[str, Any]:
    """ADR-0005 D15's diagnostic, at the only scope P3 can measure it.

    A locus is **confirmed** when the calibrated posterior of at least one strand it *carries*
    reaches the operating point — "at least one" because D15 carries both strands for an
    ambiguous locus precisely so it survives on whichever is real. It is
    **confirmation-invariant** when the verdict is the same on ``+`` and on ``-``: had the
    resolver chosen the other way, the locus would still have been confirmed (or still not).
    The reported fraction is invariant-and-confirmed over confirmed.

    ``tier_invariance`` is ``None``: §13.3's tiers are a P6 object and there is nothing at P3 to
    be tier-invariant of. The substitution is stated rather than performed silently.
    """
    n_loci = 0
    n_confirmed = 0
    n_confirmed_invariant = 0
    n_disagreements = 0
    n_low_order = 0
    reasons: dict[str, int] = {}
    deltas: list[float] = []
    n_confirmed_plus = 0
    n_confirmed_minus = 0

    n_truth_loci = 0
    n_strand_correct = 0
    n_strand_incorrect = 0
    n_strand_ambiguous = 0
    n_wrong_strand_only = 0

    for run in runs:
        truth_strand = run.truth.get("truth_strand")
        truth_start, truth_end = run.truth.get("truth_start"), run.truth.get("truth_end")
        truth_span = (
            None if truth_start is None or truth_end is None else (int(truth_start), int(truth_end))
        )
        for index, (locus, call) in enumerate(zip(run.loci, run.calls, strict=True)):
            n_loci += 1
            reasons[call.reason] = reasons.get(call.reason, 0) + 1
            if call.low_order_confidence:
                n_low_order += 1
            per_strand = both_posteriors[f"{run.contig_id}\x00{index}"]
            plus = per_strand[STRAND_PLUS]
            minus = per_strand[STRAND_MINUS]
            confirmed_plus = plus >= stage2_operating_point
            confirmed_minus = minus >= stage2_operating_point
            n_confirmed_plus += int(confirmed_plus)
            n_confirmed_minus += int(confirmed_minus)
            deltas.append(abs(plus - minus))
            if confirmed_plus != confirmed_minus:
                n_disagreements += 1
            confirmed_by_strand = {STRAND_PLUS: confirmed_plus, STRAND_MINUS: confirmed_minus}
            # What the harness would actually hand on: a resolved locus emits one strand, an
            # ambiguous one emits both. The counterfactual verdicts above exist for the
            # invariance read; only these reach a candidate table.
            emitted_confirmed = {strand: confirmed_by_strand[strand] for strand in call.strands}
            confirmed = any(emitted_confirmed.values())
            if confirmed:
                n_confirmed += 1
                if confirmed_plus == confirmed_minus:
                    n_confirmed_invariant += 1
            if (
                truth_strand is not None
                and truth_span is not None
                and _overlaps(locus.candidate.start, locus.candidate.end, *truth_span)
            ):
                n_truth_loci += 1
                if call.strand is None:
                    n_strand_ambiguous += 1
                elif call.strand == truth_strand:
                    n_strand_correct += 1
                else:
                    n_strand_incorrect += 1
                other = STRAND_MINUS if truth_strand == STRAND_PLUS else STRAND_PLUS
                # Restricted to the EMITTED strands, not to both counterfactuals. A false
                # novelty needs the wrong strand to actually reach Stage-2 as a candidate; a
                # locus whose *unemitted* counterfactual happens to score well is not one, and
                # counting it would overstate the outcome D15 says must never happen
                # (CodeRabbit r2).
                if emitted_confirmed.get(other, False) and not emitted_confirmed.get(
                    truth_strand, False
                ):
                    n_wrong_strand_only += 1

    return {
        "definition": (
            "fraction of confirmed loci whose two-stage confirmation verdict is unchanged when "
            "the strand is re-resolved to the opposite orientation"
        ),
        "tier_invariance": None,
        "tier_invariance_reason": (
            "ADR-0005 D15 states the diagnostic over §13.3 confirmation tiers; the orthogonal "
            "validation pipeline that assigns them is P6, so at P3 there is no tier to be "
            "invariant of. Confirmation invariance is the measurable analogue and is reported "
            "in its place; re-reading this against real tiers is owed at P6."
        ),
        "confirmation_invariance": (
            None if n_confirmed == 0 else n_confirmed_invariant / n_confirmed
        ),
        "n_loci": n_loci,
        "n_confirmed_loci": n_confirmed,
        "n_confirmed_invariant": n_confirmed_invariant,
        "n_verdict_disagreements": n_disagreements,
        "n_confirmed_on_plus": n_confirmed_plus,
        "n_confirmed_on_minus": n_confirmed_minus,
        "max_abs_posterior_delta": None if not deltas else float(max(deltas)),
        "median_abs_posterior_delta": None if not deltas else float(np.median(deltas)),
        "n_low_order_confidence": n_low_order,
        "low_order_confidence_fraction": None if n_loci == 0 else n_low_order / n_loci,
        # A zero here means D15's both-strand carry-through emitted nothing on this input, so
        # `low_order_confidence_fraction: 0.0` is a property of the input and NOT evidence that
        # the recall-favouring branch works. Stated as its own boolean rather than left to be
        # inferred from a zero, which is the shape that lets a mask run clean while masking
        # nothing ([[namespace-mismatch-invisible-noop]]); the branch is covered synthetically
        # in tests/unit/test_two_stage.py.
        "ambiguity_path_exercised": n_low_order > 0,
        "strand_call_reasons": dict(sorted(reasons.items())),
        "truth": {
            "n_loci_overlapping_truth": n_truth_loci,
            "n_strand_calls_correct": n_strand_correct,
            "n_strand_calls_incorrect": n_strand_incorrect,
            "n_strand_calls_ambiguous": n_strand_ambiguous,
            "n_confirmed_on_wrong_strand_only": n_wrong_strand_only,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# The report
# ══════════════════════════════════════════════════════════════════════════════════════
def duplicate_rna_agreement(
    runs: Sequence[ContigRun], stage2_logits: Mapping[str, float]
) -> dict[str, Any]:
    """How far apart Stage-2's logits are for payloads carrying the *identical* RNA.

    Two things fall out of one measurement, which is why the payload key deliberately does not
    dedup by sequence:

    * **The fwd/RC round trip.** A contig and its reverse complement carry the same locus, so a
      working resolver + handoff must hand Stage-2 byte-identical RNA from both. The count of
      RNAs seen more than once is that round trip, measured rather than asserted.
    * **Scorer determinism.** RiNALMo runs in bf16 under flash-attention and batches by token
      length, so the *same* sequence in a different batch position is not guaranteed to give
      exactly the same logit. The spread is reported rather than assumed to be zero; a
      committed golden built on top of it should say how big it was.
    """
    groups: dict[str, list[float]] = {}
    for run in runs:
        for handoff in run.both.payloads:
            key = payload_key(run.contig_id, handoff)
            if key in stage2_logits:
                groups.setdefault(handoff.rna, []).append(float(stage2_logits[key]))
    repeated = [values for values in groups.values() if len(values) > 1]
    spreads = [max(values) - min(values) for values in repeated]
    return {
        "n_distinct_rna_payloads": len(groups),
        "n_rna_scored_more_than_once": len(repeated),
        "max_abs_duplicate_logit_delta": None if not spreads else float(max(spreads)),
        "n_duplicate_groups_disagreeing": sum(1 for spread in spreads if spread > 0.0),
    }


def build_report(
    runs: Sequence[ContigRun],
    rows: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
    digest: str,
    *,
    threshold_scope: ThresholdScope,
    threshold: float | Mapping[str, float],
    min_span: int,
    gap_merge: int,
    min_distinct_elements: int,
    flank: int,
    min_order_margin: int,
    temperature: float,
    stage2_operating_point: float,
    source_prior: float | None,
    target_prior: float | None,
    determinism: Mapping[str, Any],
    stage1_source: str | None = None,
    stage2_source: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Assemble ``reports/strand_robustness.json``.

    ``is_science`` is **false** and ``gated`` is **false**, and both are load-bearing rather than
    boilerplate: this is a harness regression measured on a committed fixture, not an evaluation.
    The two-stage-vs-Stage-1-only precision comparison is P3-16 and the genome-scale scan is P5.
    ``generated_at_utc`` is injectable so a test can assemble a byte-stable report.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "prd": PRD,
        "env_lock": ENV_LOCK,
        "generated_at_utc": (
            datetime.now(UTC).isoformat(timespec="seconds")
            if generated_at_utc is None
            else generated_at_utc
        ),
        "is_science": False,
        "gated": False,
        "scope": {
            "n_contigs": len(runs),
            "n_loci": sum(len(run.loci) for run in runs),
            "n_emitted_payloads": sum(len(run.emitted.payloads) for run in runs),
            "n_scored_payloads": sum(len(run.both.payloads) for run in runs),
            "contig_ids": [run.contig_id for run in runs],
            "total_nt": sum(len(run.sequence) for run in runs),
            "stage1_source": stage1_source,
            "stage2_source": stage2_source,
        },
        "rule": {
            "threshold_scope": threshold_scope,
            "threshold": threshold,
            "min_span": min_span,
            "gap_merge": gap_merge,
            "min_distinct_elements": min_distinct_elements,
            "flank": flank,
            "min_order_margin": min_order_margin,
            "temperature": temperature,
            "stage2_operating_point": stage2_operating_point,
            "source_prior": source_prior,
            "target_prior": target_prior,
            "pinned": False,
            "pinned_note": (
                "ADR-0005 D3 freezes the Stage-1 threshold, the five locus knobs and the "
                "Stage-2 operating point at the §13.1 phase gate (P5-01); D15 freezes "
                "min_order_margin there. These are the values this fixture run used, recorded "
                "so the artifact says which rule produced it — they pin nothing."
            ),
        },
        "strand_robustness": dict(diagnostic),
        "handoff": {
            "n_contigs_with_outside_alphabet": sum(
                1 for run in runs if run.emitted.n_outside_alphabet
            ),
            "total_outside_alphabet": sum(run.emitted.n_outside_alphabet for run in runs),
            "total_outside_alphabet_in_spans": sum(
                run.emitted.n_outside_alphabet_in_spans for run in runs
            ),
        },
        "candidate_table": {
            "n_rows": len(rows),
            "columns": list(CANDIDATE_COLUMNS),
            "digest": digest,
            "digest_quantum": DIGEST_QUANTUM,
            "n_confirmed_rows": sum(1 for row in rows if row["confirmed"]),
        },
        "determinism": dict(determinism),
        "disclosures": [
            "is_science=false: a harness regression on a committed fixture, not an evaluation. "
            "P3-16 is the two-stage-vs-Stage-1-only precision comparison; P5 is the scan.",
            "tier_invariance is null — §13.3 tiers are a P6 object; the reported number is "
            "two-stage-confirmation invariance, the measurable analogue.",
            "Every rule value is provisional (pinned=false); ADR-0005 D3/D15 freeze them at the "
            "§13.1 phase gate (P5-01).",
            "max_abs_posterior_delta and n_verdict_disagreements are the diagnostic's liveness "
            "controls: a strand-blind Stage-2 would report a perfect 1.0 invariance while "
            "measuring nothing, and both of these read zero in that case.",
            "Stage-1 is not run-to-run bit-identical: re-running its leg on the identical "
            "contigs moves 91 of 327,680 stored logits by up to 3.91e-3, changing no "
            "per-position prediction. That is why the model output is committed and replayed "
            "rather than re-run.",
            "Stage-2 IS run-to-run reproducible for a fixed work list (its artifact regenerates "
            "byte-identically), but it is batch-position-sensitive: "
            "determinism.max_abs_duplicate_logit_delta records 0.0292 between two scorings of "
            "byte-identical RNA that landed at different places in the length-sorted batching. "
            "A re-batched scoring of the same sequences need not agree with this fixture.",
            "The truth block is available only because the fixture contigs were minted from "
            "corpus records of known orientation; no truth reaches any harness decision.",
            "ambiguity_path_exercised says whether D15's both-strand carry-through emitted "
            "anything on this input. When it is false, low_order_confidence_fraction=0.0 is a "
            "property of the input and not evidence the branch works.",
        ],
    }


def strand_robustness_problems(report: Mapping[str, Any]) -> list[str]:
    """Re-derive every clause of a written report; ``[]`` means it is internally consistent.

    The house pattern (``eval/gate4.py::gate4_problems``): a report that only ever validated at
    write time rots the moment the schema moves. Every clause here is a *re-derivation* from
    other fields of the same report, not a restatement of what the writer intended.
    """
    problems: list[str] = []
    for key in (
        "schema_version",
        "step",
        "generated_by",
        "adr",
        "env_lock",
        "generated_at_utc",
        "is_science",
        "gated",
        "scope",
        "rule",
        "strand_robustness",
        "candidate_table",
        "disclosures",
    ):
        if key not in report:
            problems.append(f"missing top-level key {key!r}")
    if problems:
        return problems
    if report["step"] != STEP:
        problems.append(f"step is {report['step']!r}, expected {STEP!r}")
    if report["is_science"] is not False:
        problems.append("is_science must be false: this is a harness regression, not an eval")
    if report["gated"] is not False:
        problems.append("gated must be false: no PRD §2.3 gate is graded here")

    rule = report["rule"]
    if rule.get("pinned") is not False:
        problems.append("rule.pinned must be false — ADR-0005 D3/D15 freeze these at P5-01")
    for knob in RULE_PARAMETERS:
        if knob not in rule:
            problems.append(f"rule is missing knob {knob!r}")

    diagnostic = report["strand_robustness"]
    if diagnostic.get("tier_invariance") is not None:
        problems.append("strand_robustness.tier_invariance must be null at P3 — §13.3 tiers are P6")
    if not diagnostic.get("tier_invariance_reason"):
        problems.append("strand_robustness.tier_invariance_reason must state the substitution")
    n_confirmed = diagnostic.get("n_confirmed_loci")
    n_invariant = diagnostic.get("n_confirmed_invariant")
    fraction = diagnostic.get("confirmation_invariance")
    if isinstance(n_confirmed, int) and isinstance(n_invariant, int):
        if n_invariant > n_confirmed:
            problems.append(
                f"n_confirmed_invariant ({n_invariant}) exceeds n_confirmed_loci ({n_confirmed})"
            )
        if n_confirmed == 0:
            if fraction is not None:
                problems.append("confirmation_invariance must be null when no locus is confirmed")
        elif fraction is None or abs(fraction - n_invariant / n_confirmed) > 1e-12:
            problems.append(
                f"confirmation_invariance {fraction!r} does not re-derive from "
                f"{n_invariant}/{n_confirmed}"
            )
    n_loci = diagnostic.get("n_loci")
    if isinstance(n_loci, int):
        if isinstance(n_confirmed, int) and n_confirmed > n_loci:
            problems.append(f"n_confirmed_loci ({n_confirmed}) exceeds n_loci ({n_loci})")
        low = diagnostic.get("n_low_order_confidence")
        if isinstance(low, int) and low > n_loci:
            problems.append(f"n_low_order_confidence ({low}) exceeds n_loci ({n_loci})")
        reasons = diagnostic.get("strand_call_reasons")
        if isinstance(reasons, Mapping) and sum(reasons.values()) != n_loci:
            problems.append(
                f"strand_call_reasons sums to {sum(reasons.values())}, not n_loci ({n_loci})"
            )
        if n_loci != report["scope"].get("n_loci"):
            problems.append("strand_robustness.n_loci disagrees with scope.n_loci")

    table = report["candidate_table"]
    if table.get("columns") != list(CANDIDATE_COLUMNS):
        problems.append("candidate_table.columns is not the current CANDIDATE_COLUMNS")
    if table.get("digest_quantum") != DIGEST_QUANTUM:
        problems.append("candidate_table.digest_quantum is not the current DIGEST_QUANTUM")
    digest = table.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        problems.append(f"candidate_table.digest is not a sha256 hexdigest ({digest!r})")
    n_rows = table.get("n_rows")
    scope = report["scope"]
    if isinstance(n_rows, int) and n_rows != scope.get("n_emitted_payloads"):
        problems.append(
            f"candidate_table.n_rows ({n_rows}) != scope.n_emitted_payloads "
            f"({scope.get('n_emitted_payloads')}) — the table is one row per emitted payload"
        )
    low = diagnostic.get("n_low_order_confidence")
    if (
        isinstance(n_loci, int)
        and isinstance(low, int)
        and scope.get("n_emitted_payloads") != n_loci + low
    ):
        problems.append(
            f"scope.n_emitted_payloads ({scope.get('n_emitted_payloads')}) != n_loci + "
            f"n_low_order_confidence ({n_loci} + {low}); a resolved locus emits one payload and "
            "an ambiguous one emits two (D15 both-strand carry-through)"
        )
    if isinstance(low, int) and diagnostic.get("ambiguity_path_exercised") != (low > 0):
        problems.append(
            "strand_robustness.ambiguity_path_exercised does not re-derive from "
            "n_low_order_confidence"
        )
    n_scored, n_emitted = scope.get("n_scored_payloads"), scope.get("n_emitted_payloads")
    if isinstance(n_scored, int) and isinstance(n_emitted, int):
        if n_scored < n_emitted:
            problems.append(
                f"scope.n_scored_payloads ({n_scored}) < n_emitted_payloads ({n_emitted}); the "
                "counterfactual set is a superset of the emitted one by construction"
            )
        if isinstance(n_loci, int) and n_scored != 2 * n_loci:
            problems.append(
                f"scope.n_scored_payloads ({n_scored}) != 2 × n_loci ({n_loci}); every locus is "
                "scored on both strands"
            )
    return problems


# ══════════════════════════════════════════════════════════════════════════════════════
# The composed entry point
# ══════════════════════════════════════════════════════════════════════════════════════
def run_two_stage(
    contigs: Sequence[Mapping[str, Any]],
    stage1: Mapping[str, Mapping[str, Any]],
    stage2_logits: Mapping[str, float],
    *,
    threshold_scope: ThresholdScope,
    threshold: float | Mapping[str, float],
    min_span: int,
    gap_merge: int,
    min_distinct_elements: int,
    flank: int,
    min_order_margin: int,
    temperature: float,
    stage2_operating_point: float,
    source_prior: float | None,
    target_prior: float | None,
) -> TwoStageResult:
    """The whole PRD §6 path over a contig set, from replayed model output.

    ``stage1`` maps ``contig_id`` → ``{"logits": (n_windows, window, 8), "starts": (n_windows,)}``
    — *window* logits, so the pinned reconciliation operator runs inside this call rather than
    upstream of it. ``stage2_logits`` maps :func:`payload_key` → the raw binary logit; both are
    looked up fail-closed.

    Every knob is keyword-only with no default (:func:`no_rule_parameter_has_a_default`),
    ``source_prior``/``target_prior`` included: declining the §11 prior-shift is an explicit
    ``None``, never an omission.
    """
    runs: list[ContigRun] = []
    for record in normalise_contigs(contigs):
        contig_id = record["contig_id"]
        if contig_id not in stage1:
            raise TwoStageError(f"no Stage-1 window logits for contig {contig_id!r}")
        reconciled = reconcile_contig(stage1[contig_id], len(record["sequence"]))
        runs.append(
            run_contig(
                record,
                reconciled,
                threshold_scope=threshold_scope,
                threshold=threshold,
                min_span=min_span,
                gap_merge=gap_merge,
                min_distinct_elements=min_distinct_elements,
                flank=flank,
                min_order_margin=min_order_margin,
            )
        )
    rows, both_posteriors = build_rows(
        runs,
        stage2_logits=stage2_logits,
        temperature=temperature,
        stage2_operating_point=stage2_operating_point,
        source_prior=source_prior,
        target_prior=target_prior,
    )
    diagnostic = strand_robustness(
        runs, both_posteriors, stage2_operating_point=stage2_operating_point
    )
    digest = candidate_table_digest(rows)
    report = build_report(
        runs,
        rows,
        diagnostic,
        digest,
        determinism=duplicate_rna_agreement(runs, stage2_logits),
        threshold_scope=threshold_scope,
        threshold=threshold,
        min_span=min_span,
        gap_merge=gap_merge,
        min_distinct_elements=min_distinct_elements,
        flank=flank,
        min_order_margin=min_order_margin,
        temperature=temperature,
        stage2_operating_point=stage2_operating_point,
        source_prior=source_prior,
        target_prior=target_prior,
    )
    return TwoStageResult(rows=tuple(rows), digest=digest, report=report, runs=tuple(runs))


# ══════════════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════════════
def read_contigs(path: str | Path) -> list[dict[str, Any]]:
    """Read the committed contig set. JSON, not FASTA, and deliberately.

    ``.gitattributes`` routes ``*.fa`` / ``*.fasta`` through git-LFS, and a checkout without
    ``git lfs`` leaves a 132-byte pointer that every existence check passes — the failure mode
    that has bitten this repo in CI *and* on the cluster. A fixture CI must read is therefore a
    plain-text blob under ``tests/fixtures/**``, which ``.gitattributes`` also pins to ``-text``
    so no platform renormalises a byte of it.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    contigs = payload["contigs"] if isinstance(payload, Mapping) else payload
    return normalise_contigs(contigs)


def read_stage1(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read committed per-window Stage-1 logits (``<contig>/logits``, ``<contig>/starts``)."""
    entries: dict[str, dict[str, Any]] = {}
    with np.load(path) as archive:
        for name in archive.files:
            contig_id, _, field = name.rpartition("/")
            if field not in ("logits", "starts"):
                raise TwoStageError(f"unexpected array {name!r} in {path}")
            entries.setdefault(contig_id, {})[field] = archive[name]
    for contig_id, entry in entries.items():
        missing = {"logits", "starts"} - set(entry)
        if missing:
            raise TwoStageError(f"contig {contig_id!r} is missing {sorted(missing)} in {path}")
    return entries


def read_stage2(path: str | Path) -> dict[str, float]:
    """Read committed Stage-2 binary logits, keyed by :func:`payload_key`.

    Reads a schema-``1.0`` file (no identity fields), an unsharded ``1.1`` file and a merged
    ``1.1`` file alike: all three keep the logits under ``logits``. It **refuses** a single
    shard, meaning any file whose ``shard`` is not ``None``. A shard covers only the rows its
    index was assigned. A replay would still fail on the first missing key, but the message
    would name a payload rather than the actual cause, which is that the shards were never
    merged.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "logits" in payload:
        shard = payload.get("shard")
        if shard is not None:
            where = (
                f"shard {shard.get('index')!r} of {shard.get('count')!r}"
                if isinstance(shard, Mapping)
                else f"a shard ({shard!r})"
            )
            raise TwoStageError(
                f"{path} is {where} of a sharded Stage-2 scoring, not a complete one; "
                "reassemble the shards with `two_stage merge-stage2` and read the merged file"
            )
        logits = payload["logits"]
    else:
        logits = payload
    return {str(key): float(value) for key, value in logits.items()}


def read_temperature(path: str | Path) -> float:
    """Read the fitted ``T`` out of the GATE-2 report, rather than re-typing it.

    P3-10 fitted the temperature on the D11 calibration carve and there is no standalone
    parameter file — the value lives at ``gate.calibration.temperature`` inside
    ``reports/gate2_p3_ece.json``. A hand-copied constant here would be a second home for a
    number the gate owns, free to go stale with nothing failing
    ([[pinned-constant-that-nothing-reads]]); deriving it means a re-fit moves both together.
    """
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    try:
        value = report["gate"]["calibration"]["temperature"]
    except (KeyError, TypeError) as error:  # pragma: no cover - defensive
        raise TwoStageError(
            f"{path} carries no gate.calibration.temperature; this is not a GATE-2 report"
        ) from error
    temperature = float(value)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise TwoStageError(f"{path} carries a non-positive temperature ({temperature!r})")
    return temperature


def write_json(path: str | Path, payload: Any) -> None:
    """Atomic ``tmp`` → ``replace``, ``indent=2, sort_keys=True``, ``allow_nan=False``.

    The house write (``eval/gate4.py::main``). ``allow_nan=False`` after :func:`json_safe` is
    what keeps a bare ``NaN`` token — which Python reads back and every other JSON consumer
    rejects — out of a committed artifact.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out)


# ══════════════════════════════════════════════════════════════════════════════════════
# Stage-2 artifact identity and the shard record (P3-16′-remint; ADR-0005 A15 Pin 6)
# ══════════════════════════════════════════════════════════════════════════════════════
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_SHARD_SPEC = re.compile(r"\s*(\d+)\s*/\s*(\d+)\s*")


def _is_count(value: Any) -> bool:
    """An ``int`` that is not a ``bool``: ``True == 1`` passes every equality a count meets."""
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_logit(value: Any, *, what: str) -> float:
    """A real, finite, non-``bool`` number, or a refusal.

    ``float(True)`` is ``1.0``, which is a plausible logit
    ([[bool-equals-one-passes-config-gates]]). A ``NaN`` is turned into ``null`` by
    :func:`write_json`, and ``float(None)`` then fails at replay, far from the cause.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TwoStageError(f"{what} is not a number ({value!r})")
    number = float(value)
    if not np.isfinite(number):
        raise TwoStageError(f"{what} is not finite ({number!r})")
    return number


def require_repo_relative(text: Any, *, what: str) -> str:
    """Refuse a recorded path that is absolute, home-relative, or climbs out with ``..``.

    The Stage-2 identity fields are copied verbatim into reports that live in a public repo. An
    absolute path leaks this machine's layout, and a ``..`` path does not resolve from the
    repository root. Both are refused rather than rewritten, because rewriting would record a
    path nobody passed. The check is lexical (no filesystem access) so it holds for a path
    produced on another machine.
    """
    if not isinstance(text, str) or not text:
        raise TwoStageError(f"{what} must be a non-empty repo-relative path, got {text!r}")
    pure = PurePosixPath(text)
    if pure.is_absolute() or text.startswith("~") or ".." in pure.parts:
        raise TwoStageError(
            f"{what} is {text!r}; a committed Stage-2 artifact records repo-relative paths, "
            "never an absolute, home-relative or '..' one"
        )
    return text


def manifest_row_ids(manifest: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every ``row_id`` of a payload manifest, in manifest order, after validation.

    Each id must be a non-empty string with no newline (:func:`row_ids_sha256` frames ids by
    newline), and ids must be **unique**. A duplicate id does not fail by itself: it collapses
    two payloads into one logit, and a shard boundary falling between the two copies would
    assign one id to two shards ([[duplicate-key-merges-instead-of-colliding]]).
    """
    ids: list[str] = []
    seen: set[str] = set()
    for position, entry in enumerate(manifest):
        if not isinstance(entry, Mapping):
            raise TwoStageError(f"payload manifest entry {position} is not an object")
        row_id = entry.get("row_id")
        if not isinstance(row_id, str) or not row_id or "\n" in row_id:
            raise TwoStageError(
                f"payload manifest entry {position} carries no usable row_id ({row_id!r}); it "
                "must be a non-empty string with no newline"
            )
        if row_id in seen:
            raise TwoStageError(
                f"row_id {row_id!r} appears more than once in the payload manifest; a "
                "duplicate collapses two payloads into one logit and can straddle a shard "
                "boundary"
            )
        seen.add(row_id)
        ids.append(row_id)
    return ids


def row_ids_sha256(row_ids: Iterable[str]) -> str:
    """sha256 of ``"\\n".join(sorted(row_ids)) + "\\n"``: order-free, framed by newline."""
    return hashlib.sha256(("\n".join(sorted(row_ids)) + "\n").encode("utf-8")).hexdigest()


def shard_bounds(n_rows: int, index: int, count: int) -> tuple[int, int]:
    """``[start, stop)`` of shard ``index`` of ``count`` over ``n_rows`` sorted rows.

    The chunks are contiguous, and their sizes differ by at most one. The first
    ``n_rows % count`` chunks hold the extra row. Together the ``count`` chunks partition
    ``[0, n_rows)``, so if every index is present and each shard holds exactly its own chunk,
    the union is the whole manifest.
    """
    if not _is_count(count) or count < 1:
        raise TwoStageError(f"shard count must be an integer >= 1, got {count!r}")
    if not _is_count(index) or not 0 <= index < count:
        raise TwoStageError(f"shard index must be an integer in [0, {count}), got {index!r}")
    if not _is_count(n_rows) or n_rows < 0:
        raise TwoStageError(f"n_rows must be a non-negative integer, got {n_rows!r}")
    base, extra = divmod(n_rows, count)
    start = index * base + min(index, extra)
    return start, start + base + (1 if index < extra else 0)


def shard_manifest(
    manifest: Sequence[Mapping[str, Any]], index: int, count: int
) -> list[dict[str, Any]]:
    """Shard ``index`` of ``count``: a contiguous slice of the manifest sorted by ``row_id``.

    The slice depends on the **set** of rows and on nothing else. Input order does not matter,
    because the manifest is sorted by ``row_id`` here rather than trusted to arrive sorted. The
    merge re-derives every shard's assignment through this same function, so a shard holding
    rows its index was not assigned is refused.
    """
    start, stop = shard_bounds(len(manifest), index, count)
    manifest_row_ids(manifest)
    ordered = sorted(manifest, key=lambda entry: entry["row_id"])
    return [dict(entry) for entry in ordered[start:stop]]


def shard_contigs(
    contigs: Sequence[Mapping[str, Any]], index: int, count: int
) -> list[dict[str, Any]]:
    """Stage-1 shard ``index`` of ``count``: a contiguous slice of the contigs sorted by id.

    The Stage-1 twin of :func:`shard_manifest` (P3-16″-hardneg, the A17 census: ~820k one-window
    contigs are ~20 GPU-h per arm on one card). The slice depends on the **set** of contig ids
    alone, and :func:`merge_stage1_shards` re-derives every shard's assignment through this same
    function, so a shard file holding contigs its index was not assigned is refused.
    """
    start, stop = shard_bounds(len(contigs), index, count)
    ordered = sorted(normalise_contigs(contigs), key=lambda contig: contig["contig_id"])
    return [dict(contig) for contig in ordered[start:stop]]


def stage1_npz_contigs(path: str | Path) -> list[str]:
    """The contig ids a Stage-1 ``.npz`` holds, read off its member names without loading them."""
    with np.load(path) as archive:
        names = list(archive.files)
    by_contig: dict[str, set[str]] = {}
    for name in names:
        contig_id, _, field = name.rpartition("/")
        if field not in ("logits", "starts") or not contig_id:
            raise TwoStageError(f"unexpected array {name!r} in {path}")
        by_contig.setdefault(contig_id, set()).add(field)
    torn = sorted(cid for cid, fields in by_contig.items() if fields != {"logits", "starts"})
    if torn:
        raise TwoStageError(f"{path}: contig(s) {torn[:3]} lack logits or starts")
    return sorted(by_contig)


def merge_stage1_problems(
    contigs: Sequence[Mapping[str, Any]], shard_contig_ids: Sequence[Sequence[str]]
) -> list[str]:
    """Why ``shard_contig_ids`` (shard files in index order) do not reassemble ``contigs``.

    Shard ``i`` of ``N = len(shard_contig_ids)`` must hold exactly :func:`shard_contigs`'s slice
    ``i`` — not merely some disjoint part of the set. Checking only the union would pass a
    shard re-run against a different contig file or cut at a different ``N``.
    """
    count = len(shard_contig_ids)
    if count < 1:
        return ["no Stage-1 shard was given"]
    problems: list[str] = []
    for index, held in enumerate(shard_contig_ids):
        want = [contig["contig_id"] for contig in shard_contigs(contigs, index, count)]
        if sorted(held) != want:
            extra = sorted(set(held) - set(want))
            missing = sorted(set(want) - set(held))
            problems.append(
                f"shard {index}/{count} holds {len(held)} contig(s), not its assigned "
                f"{len(want)}: {len(extra)} unassigned (e.g. {extra[:3]}), {len(missing)} "
                f"missing (e.g. {missing[:3]})"
            )
    return problems


def parse_shard(text: str) -> tuple[int, int]:
    """``--shard I/N`` → ``(I, N)``, refusing anything but ``0 <= I < N``, ``N >= 1``."""
    match = _SHARD_SPEC.fullmatch(text)
    if match is None:
        raise argparse.ArgumentTypeError(f"--shard must read I/N (e.g. 0/4), got {text!r}")
    index, count = int(match.group(1)), int(match.group(2))
    if count < 1 or index >= count:
        raise argparse.ArgumentTypeError(f"--shard {text!r}: need N >= 1 and 0 <= I < N")
    return index, count


def stage2_env_lock(backbone: Any) -> str:
    """The conda lock a Stage-2 arm on ``backbone`` runs under, derived from the allow-list.

    ``backbone`` may be the allow-list key the loader records (``rinalmo-giga`` / ``rnafm``) or
    the Hugging Face repo id (``multimolecule/rinalmo-giga`` / ``multimolecule/rnafm``). The
    two namespaces cannot collide. The mapping is :attr:`RnaBackbone.env_lock` (ADR-0002 A15),
    the same field ``stage2.train.env_lock_for`` stamps into a training run's provenance, so
    the scoring artifact and the run that trained the weights cannot name different
    environments. Before A15 this leg wrote ``envs/ml-rna.conda-lock.yml`` for every arm.
    That would have named the wrong environment for the RNA-FM comparator, which the
    ``ml-rna`` lock cannot even load. An unknown backbone is refused, not defaulted.
    """
    spec = None
    if isinstance(backbone, str):
        spec = BR.BACKBONES.get(backbone) or BR.backbone_for_repo_id(backbone)
    if spec is None:
        raise TwoStageError(
            f"backbone {backbone!r} is not in the ADR-0002 A15 allow-list (keys "
            f"{BR.BACKBONE_KEYS}, repo ids {tuple(b.repo_id for b in BR.BACKBONES.values())}); "
            "its env lock cannot be derived, and a defaulted one would name an environment "
            "the run did not use"
        )
    return spec.env_lock


def parse_payload_manifest(raw: bytes, *, where: str) -> list[dict[str, Any]]:
    """A payload-manifest file's **bytes** → its validated ``payloads`` list.

    The function takes bytes, not a parsed object, so that the ``payloads_sha256`` beside a
    scoring is computed over the same bytes the rows were parsed from.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TwoStageError(f"{where} is not a JSON payload manifest: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("payloads"), list):
        raise TwoStageError(f"{where} carries no 'payloads' list; it is not a payload manifest")
    manifest = payload["payloads"]
    manifest_row_ids(manifest)
    return manifest


def stage2_identity(
    *,
    arm: str,
    arm_selected_by: str,
    checkpoint_root: str,
    checkpoint_dir: str,
    record: Mapping[str, Any],
    trained_under: str,
    device: str,
    batch_size: int,
    payloads_sha256: str,
) -> dict[str, Any]:
    """The identity and run block of a Stage-2 artifact (ADR-0005 A15 Pin 6).

    ``record`` is what ``stage2.eval.load_stage2_checkpoint`` returned. ``backbone``,
    ``revision``, ``dtype``, ``adapter_sha256`` and ``heads_sha256`` are copied from it, and
    are therefore measured off the loaded checkpoint, not taken from the command line.
    Torch-free and pure, so the leg can call it **after the load and before scoring**: a run
    whose identity cannot be stated fails before it spends the GPU time.

    Refusals:

    * any identity field ``None`` or empty: an artifact with a missing identity field cannot be
      told apart from another run;
    * any of the three digests not a sha256 hexdigest;
    * ``batch_size`` not an integer ``>= 1``;
    * a recorded path that is absolute or contains ``..``;
    * ``checkpoint_dir`` not equal to ``checkpoint_root/arm``. Those three fields are recorded
      separately, and this check stops them from describing three different things;
    * a recorded ``base_model_name_or_path`` that resolves to a backbone other than
      ``backbone``. The loader enforces the same rule; this repeats it on the artifact
      itself.
    """
    backbone = record.get("backbone")
    identity: dict[str, Any] = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "step": STEP,
        "arm": arm,
        "arm_selected_by": arm_selected_by,
        "checkpoint_root": require_repo_relative(checkpoint_root, what="checkpoint_root"),
        "checkpoint_dir": require_repo_relative(checkpoint_dir, what="checkpoint_dir"),
        "backbone": backbone,
        "base_model_name_or_path": record.get("base_model_name_or_path"),
        "revision": record.get("revision"),
        "dtype": record.get("dtype"),
        "adapter_sha256": record.get("adapter_sha256"),
        "heads_sha256": record.get("heads_sha256"),
        "n_lora_b_nonzero": record.get("n_lora_b_nonzero"),
        "attn_implementation": trained_under,
        "attn_implementation_matches_training": record.get("attn_implementation") == trained_under,
        "device": str(device),
        "env_lock": stage2_env_lock(backbone),
        "batch_size": batch_size,
        "payloads_sha256": payloads_sha256,
    }
    for field in STAGE2_IDENTITY_FIELDS:
        value = identity[field]
        if value is None or (isinstance(value, str) and not value):
            raise TwoStageError(
                f"Stage-2 identity field {field!r} is {value!r}; ADR-0005 A15 Pin 6 requires "
                "every identity field to be recorded, and a missing one cannot tell this run "
                "apart from another"
            )
    for field in ("adapter_sha256", "heads_sha256", "payloads_sha256"):
        if not isinstance(identity[field], str) or not _SHA256_HEX.fullmatch(identity[field]):
            raise TwoStageError(f"{field} is not a sha256 hexdigest ({identity[field]!r})")
    if not _is_count(batch_size) or batch_size < 1:
        raise TwoStageError(f"batch_size must be an integer >= 1, got {batch_size!r}")
    expected_dir = PurePosixPath(identity["checkpoint_root"]) / str(arm)
    if PurePosixPath(identity["checkpoint_dir"]) != expected_dir:
        raise TwoStageError(
            f"checkpoint_dir {identity['checkpoint_dir']!r} is not checkpoint_root/arm "
            f"({str(expected_dir)!r}); the three identity fields would name different "
            "checkpoints"
        )
    base = identity["base_model_name_or_path"]
    if base is not None:
        derived = BR.backbone_for_repo_id(base)
        if derived is None or derived.key != backbone:
            raise TwoStageError(
                f"the checkpoint records base model {base!r}, which is not backbone "
                f"{backbone!r}; the adapter and the recorded identity disagree"
            )
    return identity


def stage2_artifact(
    identity: Mapping[str, Any],
    *,
    payloads: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    shard: tuple[int, int] | None,
) -> dict[str, Any]:
    """Assemble the schema-``1.1`` artifact the ``stage2`` leg writes.

    ``payloads`` is the work list actually scored: the whole manifest, or one
    :func:`shard_manifest` slice. ``scored`` is ``stage2.eval.score_rows``' output. The scored
    row ids must be exactly the work list's, so an artifact cannot state a ``shard`` record or
    ``n_payloads`` that its logits do not match. ``shard_layout`` is ``None`` here, because a
    layout belongs only to a merge.
    """
    ids = manifest_row_ids(payloads)
    logits: dict[str, float] = {}
    n_tokens: dict[str, int] = {}
    for row in scored:
        row_id = str(row.get("row_id"))
        if row_id in logits:
            raise TwoStageError(f"the scorer returned row {row_id!r} twice")
        logits[row_id] = _finite_logit(row.get("tbox_logit"), what=f"logit of row {row_id!r}")
        tokens = row.get("n_tokens")
        if not _is_count(tokens) or tokens < 1:
            raise TwoStageError(f"row {row_id!r} has no positive integer n_tokens ({tokens!r})")
        n_tokens[row_id] = tokens
    if set(logits) != set(ids):
        raise TwoStageError(
            f"the scorer returned {len(logits)} rows for a work list of {len(ids)}: "
            f"{len(set(ids) - set(logits))} unscored, {len(set(logits) - set(ids))} foreign"
        )
    shard_record = None
    if shard is not None:
        index, count = shard
        shard_bounds(len(ids), index, count)  # validates the (index, count) pair itself
        shard_record = {
            "index": index,
            "count": count,
            "n_payloads": len(ids),
            "row_ids_sha256": row_ids_sha256(ids),
        }
    artifact = {
        **{key: identity[key] for key in (*STAGE2_IDENTITY_FIELDS, *STAGE2_RUN_FIELDS)},
        "shard": shard_record,
        "shard_layout": None,
        "n_payloads": len(ids),
        "logits": logits,
        "n_tokens": n_tokens,
    }
    return artifact


def merge_stage2_shards(manifest_raw: bytes, shard_files: Sequence[bytes]) -> dict[str, Any]:
    """Reassemble ``--shard I/N`` artifacts into one Stage-2 artifact, or refuse.

    Takes the payload manifest's **bytes** and each shard file's **bytes**. Every digest it
    checks or records is then computed from the same bytes it parsed. Refuses unless:

    * every shard is a schema-``1.1`` *shard* (``shard`` set, ``shard_layout`` ``None``) that
      carries every :data:`STAGE2_ARTIFACT_KEYS` field and no ``None`` identity field;
    * every shard's ``payloads_sha256`` equals the sha256 of the given manifest, so each shard
      was cut from **this** manifest;
    * every :data:`STAGE2_IDENTITY_FIELDS` and :data:`STAGE2_RUN_FIELDS` value is identical
      across shards. The comparison is by JSON text, so ``1`` vs ``1.0`` vs ``true`` disagree;
    * every shard states the same ``count``, and the indices are exactly ``{0..count-1}``, with
      none missing and none twice;
    * each shard's ``logits`` and ``n_tokens`` keys equal the rows :func:`shard_manifest`
      assigns its index, and its ``n_payloads`` and ``row_ids_sha256`` re-derive from them.

    No separate "union == manifest" clause is needed, because it cannot fail once the checks
    above pass: :func:`shard_bounds` partitions the sorted ids, every index is present, and
    each shard holds exactly its own chunk. ``test_shards_partition_the_manifest`` pins that
    partition directly.

    The result has the keys of an unsharded artifact (:data:`STAGE2_ARTIFACT_KEYS`), with
    ``shard: None`` and a ``shard_layout`` recording ``{index, n_payloads, row_ids_sha256,
    file_sha256}`` for every shard. The layout is kept because a slice is batched differently
    from the whole manifest, so the merged logits need not equal an unsharded run's.
    """
    manifest = parse_payload_manifest(manifest_raw, where="the payload manifest")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if not shard_files:
        raise TwoStageError("no shard files were given to merge")

    shards: list[tuple[Mapping[str, Any], str]] = []
    for position, raw in enumerate(shard_files):
        where = f"shard file #{position}"
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise TwoStageError(f"{where} is not JSON: {error}") from error
        if not isinstance(payload, Mapping):
            raise TwoStageError(f"{where} is not a JSON object")
        if payload.get("schema_version") != STAGE2_SCHEMA_VERSION:
            raise TwoStageError(
                f"{where} is schema {payload.get('schema_version')!r}, not "
                f"{STAGE2_SCHEMA_VERSION!r}; only a shard written by the 1.1 `stage2 --shard` "
                "leg records the identity a merge must check"
            )
        missing = [key for key in STAGE2_ARTIFACT_KEYS if key not in payload]
        if missing:
            raise TwoStageError(f"{where} is missing {missing}")
        for field in STAGE2_IDENTITY_FIELDS:
            if payload[field] is None:
                raise TwoStageError(
                    f"{where} records identity field {field!r} as null; a merge cannot "
                    "certify shards as one measurement on a field none of them states"
                )
        record = payload["shard"]
        if not isinstance(record, Mapping):
            raise TwoStageError(
                f"{where} has shard={record!r}; it is not one shard of a sharded scoring"
            )
        if payload["shard_layout"] is not None:
            raise TwoStageError(f"{where} is already a merged artifact, not a shard")
        for key in ("index", "count", "n_payloads"):
            if not _is_count(record.get(key)):
                raise TwoStageError(f"{where} shard.{key} is not an integer ({record.get(key)!r})")
        if payload["payloads_sha256"] != manifest_sha:
            raise TwoStageError(
                f"{where} (shard {record['index']}) was scored from payload manifest "
                f"{payload['payloads_sha256']}, not from the given one ({manifest_sha}); its "
                "rows were cut from a different work list"
            )
        shards.append((payload, hashlib.sha256(raw).hexdigest()))

    reference = shards[0][0]
    for field in (*STAGE2_IDENTITY_FIELDS, *STAGE2_RUN_FIELDS):
        seen = {json.dumps(payload[field], sort_keys=True) for payload, _ in shards}
        if len(seen) > 1:
            raise TwoStageError(
                f"the shards disagree on {field!r} ({sorted(seen)}); they are not one "
                "measurement, and merging them would publish one artifact describing several "
                "runs (ADR-0005 A15 Pin 6)"
            )

    counts = {payload["shard"]["count"] for payload, _ in shards}
    if len(counts) != 1:
        raise TwoStageError(f"the shards state different shard counts {sorted(counts)}")
    count = counts.pop()
    indices = [payload["shard"]["index"] for payload, _ in shards]
    duplicated = sorted({index for index in indices if indices.count(index) > 1})
    if duplicated:
        raise TwoStageError(f"shard index(es) {duplicated} were given more than once")
    absent = sorted(set(range(count)) - set(indices))
    if absent:
        raise TwoStageError(f"shard index(es) {absent} of {count} are missing")
    stray = sorted(set(indices) - set(range(count)))
    if stray:
        raise TwoStageError(f"shard index(es) {stray} lie outside [0, {count})")

    logits: dict[str, float] = {}
    n_tokens: dict[str, int] = {}
    layout: list[dict[str, Any]] = []
    for payload, file_sha in sorted(shards, key=lambda item: item[0]["shard"]["index"]):
        record = payload["shard"]
        index = record["index"]
        expected = [entry["row_id"] for entry in shard_manifest(manifest, index, count)]
        for name in ("logits", "n_tokens"):
            got = payload[name]
            if not isinstance(got, Mapping) or set(got) != set(expected):
                held = set(got) if isinstance(got, Mapping) else set()
                raise TwoStageError(
                    f"shard {index} of {count}: its {name} keys are not the rows its index "
                    f"assigns ({len(set(expected) - held)} assigned rows absent, "
                    f"{len(held - set(expected))} rows not assigned to it)"
                )
        if record["n_payloads"] != len(expected) or payload["n_payloads"] != len(expected):
            raise TwoStageError(
                f"shard {index} of {count} states n_payloads {record['n_payloads']!r} / "
                f"{payload['n_payloads']!r}, but its index assigns {len(expected)} rows"
            )
        digest = row_ids_sha256(expected)
        if record.get("row_ids_sha256") != digest:
            raise TwoStageError(
                f"shard {index} of {count} states row_ids_sha256 "
                f"{record.get('row_ids_sha256')!r}, which does not re-derive ({digest})"
            )
        for row_id in expected:
            logits[row_id] = _finite_logit(
                payload["logits"][row_id], what=f"shard {index} logit of row {row_id!r}"
            )
            tokens = payload["n_tokens"][row_id]
            if not _is_count(tokens) or tokens < 1:
                raise TwoStageError(f"shard {index} row {row_id!r} n_tokens is {tokens!r}")
            n_tokens[row_id] = tokens
        layout.append(
            {
                "index": index,
                "n_payloads": len(expected),
                "row_ids_sha256": digest,
                "file_sha256": file_sha,
            }
        )

    return {
        **{key: reference[key] for key in (*STAGE2_IDENTITY_FIELDS, *STAGE2_RUN_FIELDS)},
        "shard": None,
        "shard_layout": {"count": count, "shards": layout},
        "n_payloads": len(logits),
        "logits": logits,
        "n_tokens": n_tokens,
    }


def union_payload_manifests(manifests: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict]:
    """The deduplicated union of several payload manifests over one benchmark, by ``row_id``.

    A τ sweep's work lists share most rows, and a row's id is its content address
    (:func:`payload_key`), so scoring the union once gives every τ's replay each logit it needs.
    P3-16's v1 union was assembled by hand; this is that step, made a leg. A ``row_id`` carried
    with two different payloads is refused: one logit would then stand for two inputs. Rows come
    back sorted by ``row_id``, so the union does not depend on the order of its inputs.
    """
    union: dict[str, dict] = {}
    for manifest in manifests:
        manifest_row_ids(manifest)
        for row in manifest:
            row_id = row["row_id"]
            if row_id in union and union[row_id] != dict(row):
                raise TwoStageError(
                    f"row_id {row_id!r} is carried with two different payloads; the manifests "
                    "were not cut from one benchmark under one transcriber"
                )
            union.setdefault(row_id, dict(row))
    if not union:
        raise TwoStageError("the union of the given manifests is empty; there is nothing to score")
    return [union[row_id] for row_id in sorted(union)]


def refuse_v1_output(path: str | Path) -> None:
    """Refuse to write over a v1 artifact (ADR-0005 A15 Pin 5: v1 stays byte-unchanged).

    A15 Pin 5 states that "no v2 artifact is written to a v1 path". The fixed paths P3-17a
    hash-binds are ``stage2.cm_confirmer.CANONICAL_SOURCES``, read from there rather than
    re-typed here ([[promote-dont-duplicate-is-a-correctness-rule]]). Any
    ``data/interim/p3_16/v1_*`` file is also refused, which covers the production arm and the
    other τ reads P3-17a does not bind. Paths are compared by their trailing components after
    ``resolve()``, so the guard holds from the main checkout, from a worktree, and on the
    cluster.
    """
    from tbox_finder.stage2.cm_confirmer import CANONICAL_SOURCES  # lazy: torch-free, light

    resolved = Path(path).resolve()
    for key, value in CANONICAL_SOURCES.items():
        tail = PurePosixPath(value).parts
        if resolved.parts[-len(tail) :] == tail:
            raise TwoStageError(
                f"refusing to write {path}: it is the v1 artifact P3-17a binds as {key!r} "
                f"({value}); ADR-0005 A15 Pin 5 keeps every v1 byte unchanged and writes v2 to "
                "v2 paths"
            )
    if resolved.name.startswith("v1_") and resolved.parent.name == "p3_16":
        raise TwoStageError(
            f"refusing to write {path}: data/interim/p3_16/v1_* is the withdrawn v1 benchmark's "
            "record (ADR-0005 A15 Pin 5); write v2 artifacts to v2_* paths"
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════════════
def parse_threshold(text: str) -> float | dict[str, float]:
    """``--threshold`` as either a scalar or a per-class JSON mapping.

    ``element_assignment`` refuses a scalar under ``threshold_scope="per_class"`` and refuses a
    mapping under ``"global"``, so a CLI that could only parse a float made one of D3's two
    declared scopes unreachable — every ``per_class`` invocation failed after parsing
    (CodeRabbit r2). The scope choice is a §13.1 phase-gate freeze, so both must be expressible.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            mapping = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise argparse.ArgumentTypeError(f"--threshold is not valid JSON: {error}") from error
        if not isinstance(mapping, dict):
            raise argparse.ArgumentTypeError("--threshold JSON must be an object")
        return {str(key): float(value) for key, value in mapping.items()}
    try:
        return float(stripped)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--threshold is not a number: {text!r}") from error


def _rule_args(parser: argparse.ArgumentParser) -> None:
    """The rule knobs, every one ``required=True``.

    ``argparse`` defaults are defaults: an omitted ``--threshold`` that silently became 0.9
    would be a frozen value nobody signed off, reported as the value the caller asked for. Same
    discipline as the keyword-only-no-default signature, one layer out.
    """
    parser.add_argument("--threshold-scope", required=True, choices=["global", "per_class"])
    parser.add_argument(
        "--threshold",
        type=parse_threshold,
        required=True,
        help=(
            "a scalar for threshold_scope=global, or a JSON object of "
            "element-class name -> tau for threshold_scope=per_class"
        ),
    )
    parser.add_argument("--min-span", type=int, required=True)
    parser.add_argument("--gap-merge", type=int, required=True)
    parser.add_argument("--min-distinct-elements", type=int, required=True)
    parser.add_argument("--flank", type=int, required=True)
    parser.add_argument("--min-order-margin", type=int, required=True)


def _cmd_stage1(args: argparse.Namespace) -> int:
    """LEG 1 (env ``ml-dna``, torch): contigs → per-window Stage-1 logits.

    The forward loop here is small and explicit rather than borrowed, because
    :func:`tbox_finder.infer.scan.scan_encoded_windows` returns a *reconciled* posterior and the
    fixture needs the logits that go **into** the operator — storing its output instead would
    put the pinned operator outside the regression it exists to protect. To keep that from
    becoming an unchecked second copy, the leg reconciles its own stored logits and compares the
    result to what ``scan_encoded_windows`` returns for the same contig, on **two** counts: the
    per-position prediction must match **exactly**, and the posterior must agree to within
    :data:`MAX_MINT_LOG_PROB_DELTA`. Not "bit-identical" — the stored logits are float16, so the
    posterior cannot be, and saying so would have been an unmeasured claim in the one place that
    exists to check one.
    """
    refuse_v1_output(args.out)  # before the checkpoint load, not after the scan (A15 Pin 5)

    import torch  # lazy: this leg only

    from tbox_finder.infer import scan

    contigs = read_contigs(args.contigs)
    if args.shard is not None:
        contigs = shard_contigs(contigs, *args.shard)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = scan.load_stage1_checkpoint(args.checkpoint, device=device)
    arrays: dict[str, np.ndarray] = {}
    worst_delta = 0.0
    for contig in contigs:
        sequence = contig["sequence"]
        window_ids, starts = scan.scan_window_ids(sequence)
        chunks: list[np.ndarray] = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for begin in range(0, window_ids.shape[0], args.batch_size):
                    batch = window_ids[begin : begin + args.batch_size].astype(np.int64)
                    out = model(input_ids=torch.from_numpy(batch).to(device))
                    chunks.append(out.float().cpu().numpy())
        finally:
            model.train(was_training)
        logits = np.concatenate(chunks, axis=0).astype(args.dtype)
        reference = scan.scan_encoded_windows(
            model, window_ids, starts, len(sequence), device=device, batch_size=args.batch_size
        )
        mine = reconcile_windows(logits.astype(np.float64), np.asarray(starts), len(sequence))
        if not np.array_equal(mine.prediction, reference.prediction):
            raise TwoStageError(
                f"contig {contig['contig_id']!r}: reconciling this leg's stored logits does not "
                "reproduce scan_encoded_windows' prediction — the stored fixture would not be "
                "the shipped scanner's output"
            )
        delta = float(np.max(np.abs(mine.log_probs - reference.log_probs)))
        worst_delta = max(worst_delta, delta)
        if delta > args.max_log_prob_delta:
            raise TwoStageError(
                f"contig {contig['contig_id']!r}: the stored logits reconcile to a posterior "
                f"{delta:.3e} from scan_encoded_windows', above the "
                f"{args.max_log_prob_delta} tolerance; that is more than {args.dtype} "
                "quantisation and the fixture would not be the scanner's output"
            )
        arrays[f"{contig['contig_id']}/logits"] = logits
        arrays[f"{contig['contig_id']}/starts"] = np.asarray(starts, dtype=np.int32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(
        f"wrote {args.out}: {len(contigs)} contigs, dtype={args.dtype}, "
        f"max |delta log-prob| vs scan_encoded_windows = {worst_delta:.3e} "
        f"(tolerance {args.max_log_prob_delta}), prediction mismatches = 0"
    )
    return 0


def _cmd_merge_stage1(args: argparse.Namespace) -> int:
    """LEG 1b (torch-free): ``stage1 --shard I/N`` archives → one Stage-1 ``.npz``, or a refusal.

    Every check runs before an array is loaded: each shard must hold exactly its
    :func:`shard_contigs` slice of ``--contigs`` (given in index order), so the merged archive
    covers the contig set once each. The merged archive has the unsharded leg's layout, so
    ``payloads`` and ``run`` read it unchanged.
    """
    refuse_v1_output(args.out)
    out = Path(args.out).resolve()
    if any(Path(shard).resolve() == out for shard in args.shards):
        raise TwoStageError("--out is one of the --shards; the merge would overwrite its input")
    contigs = read_contigs(args.contigs)
    held = [stage1_npz_contigs(shard) for shard in args.shards]
    problems = merge_stage1_problems(contigs, held)
    if problems:
        raise TwoStageError(
            "the Stage-1 shards do not reassemble --contigs: " + "; ".join(problems)
        )
    arrays: dict[str, np.ndarray] = {}
    for shard in args.shards:
        with np.load(shard) as archive:
            for name in archive.files:
                arrays[name] = archive[name]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"wrote {args.out}: {len(contigs)} contigs from {len(args.shards)} Stage-1 shard(s)")
    return 0


def _cmd_payloads(args: argparse.Namespace) -> int:
    """LEG 2 (torch-free): contigs + Stage-1 logits + the rule → the Stage-2 work list."""
    refuse_v1_output(args.out)
    contigs = read_contigs(args.contigs)
    stage1 = read_stage1(args.stage1)
    runs = []
    for contig in contigs:
        contig_id = contig["contig_id"]
        if contig_id not in stage1:
            raise TwoStageError(f"no Stage-1 window logits for contig {contig_id!r}")
        reconciled = reconcile_contig(stage1[contig_id], len(contig["sequence"]))
        runs.append(
            run_contig(
                contig,
                reconciled,
                threshold_scope=args.threshold_scope,
                threshold=args.threshold,
                min_span=args.min_span,
                gap_merge=args.gap_merge,
                min_distinct_elements=args.min_distinct_elements,
                flank=args.flank,
                min_order_margin=args.min_order_margin,
            )
        )
    manifest = payload_manifest(runs)
    write_json(args.out, {"schema_version": SCHEMA_VERSION, "step": STEP, "payloads": manifest})
    print(f"wrote {args.out}: {len(manifest)} payloads over {len(runs)} contigs")
    return 0


def _cmd_stage2(args: argparse.Namespace) -> int:
    """LEG 3 (env ``ml-rna``, torch): the work list → one calibrated-stack-ready binary logit each.

    ``attn_implementation`` is taken from the arm's own run report, the way
    ``calib/gate2.py::score_loo_holdout`` does: RiNALMo was fine-tuned under
    ``flash_attention_2``, the checkpoint carries bf16 weights, and scoring it under a backend
    it was not trained under is a silent numerical change (transformers' own docs: FA2 supports
    fp16/bf16 only). Whether the two matched is recorded in the artifact, not assumed.

    ``flash_attn`` registers no CPU kernel, so a CPU run under that backend does not degrade —
    it raises deep inside the forward. The leg refuses **before** loading 2.5 GB of weights, and
    refuses rather than silently falling back to ``eager``: a fallback would score the fixture
    under a backend the checkpoint was not trained under and mint that into a committed golden.

    **Identity and shards (P3-16′-remint, ADR-0005 A15 Pin 6).** The artifact is schema
    ``1.1``. It records the identity :func:`stage2_identity` builds from the loader's own
    record, the batch size, the sha256 of the payload-manifest **bytes**, and an ``env_lock``
    derived from the backbone. ``--shard I/N`` scores only the :func:`shard_manifest` slice
    ``I`` and records ``shard: {index, count, n_payloads, row_ids_sha256}``. ``merge-stage2``
    reassembles the slices. Every check that does not need the model runs **before** the
    2.5 GB load: the v1-path guard, the out-vs-payloads collision, the batch size, the manifest
    and its shard, and the recorded paths. The identity check runs after the load and before
    scoring.
    """
    refuse_v1_output(args.out)
    if Path(args.out).resolve() == Path(args.payloads).resolve():
        raise TwoStageError("--out is the payload manifest itself; the scores would overwrite it")
    if not _is_count(args.batch_size) or args.batch_size < 1:
        raise TwoStageError(f"--batch-size must be an integer >= 1, got {args.batch_size!r}")

    import torch  # lazy: this leg only

    from tbox_finder.stage2 import eval as stage2_eval

    raw = Path(args.payloads).read_bytes()
    payloads_sha256 = hashlib.sha256(raw).hexdigest()
    manifest = parse_payload_manifest(raw, where=str(args.payloads))
    work = manifest if args.shard is None else shard_manifest(manifest, *args.shard)
    arms = stage2_eval.discover_arms(checkpoint_root=args.checkpoint_root, sweep_dir=args.sweep_dir)
    production = stage2_eval.production_arm_config()
    matching = [
        name
        for name, spec in sorted(arms.items())
        if spec["aux_weight"] == production["aux_weight"] and spec["lr"] == production["lr"]
    ]
    arm_name = args.arm or (matching[0] if matching else None)
    if arm_name is None or arm_name not in arms:
        raise TwoStageError(
            f"no discovered arm matches the production config {production!r}; "
            f"found {sorted(arms)}"
            if args.arm is None
            else f"--arm {args.arm!r} is not among the discovered arms {sorted(arms)}"
        )
    spec = arms[arm_name]
    # Relativised by the same function that relativised `checkpoint_dir`, so the two agree,
    # then refused if that left anything a public repo cannot carry.
    checkpoint_root = require_repo_relative(
        stage2_eval.repo_relative(args.checkpoint_root), what="--checkpoint-root"
    )
    require_repo_relative(spec["checkpoint_dir"], what=f"arm {arm_name} checkpoint_dir")
    trained_under = spec.get("attn_implementation")
    if trained_under is None:
        raise TwoStageError(
            f"arm {arm_name} records no attn_implementation; the committed logits must state the "
            "backend the checkpoint was trained under rather than leave it to be assumed"
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if trained_under == "flash_attention_2" and not str(device).startswith("cuda"):
        raise TwoStageError(
            f"arm {arm_name} was trained under flash_attention_2, which has no CPU kernel, and "
            f"the resolved device is {device!r}. Refused rather than switched to eager: the "
            "committed logits must come from the backend the checkpoint was trained under"
        )
    model, record = stage2_eval.load_stage2_checkpoint(
        spec["checkpoint_path"], attn_implementation=trained_under, device=device
    )
    identity = stage2_identity(
        arm=arm_name,
        arm_selected_by="stage2.eval.production_arm_config()" if args.arm is None else "cli",
        checkpoint_root=checkpoint_root,
        checkpoint_dir=spec["checkpoint_dir"],
        record=record,
        trained_under=trained_under,
        device=str(device),
        batch_size=args.batch_size,
        payloads_sha256=payloads_sha256,
    )
    scored = stage2_eval.score_rows(model, work, batch_size=args.batch_size)
    write_json(args.out, stage2_artifact(identity, payloads=work, scored=scored, shard=args.shard))
    where = "" if args.shard is None else f" (shard {args.shard[0]}/{args.shard[1]})"
    print(
        f"wrote {args.out}: {len(scored)} of {len(manifest)} payloads{where} scored by arm "
        f"{arm_name} ({identity['backbone']}, env {identity['env_lock']})"
    )
    return 0


def _cmd_merge_stage2(args: argparse.Namespace) -> int:
    """LEG 3b (torch-free): ``--shard I/N`` artifacts → one Stage-2 artifact, or a refusal.

    Every refusal is :func:`merge_stage2_shards`'. This leg also refuses an ``--out`` that is
    one of its own inputs, and checks that **before** reading anything
    ([[two-outputs-one-path-destroys-the-first]]). It also refuses a v1 path (A15 Pin 5).
    """
    out = Path(args.out).resolve()
    inputs = [Path(args.payloads).resolve(), *(Path(path).resolve() for path in args.shards)]
    if out in inputs:
        raise TwoStageError(f"--out {args.out} is one of the merge's own inputs")
    refuse_v1_output(args.out)
    merged = merge_stage2_shards(
        Path(args.payloads).read_bytes(), [Path(path).read_bytes() for path in args.shards]
    )
    write_json(args.out, merged)
    print(
        f"wrote {args.out}: {merged['n_payloads']} payloads merged from "
        f"{merged['shard_layout']['count']} shards (arm {merged['arm']}, {merged['backbone']})"
    )
    return 0


def _cmd_payloads_union(args: argparse.Namespace) -> int:
    """LEG 2b (torch-free): several ``payloads`` manifests → the one work list Stage-2 scores.

    Refuses an ``--out`` that is one of its inputs before reading anything, and a v1 path
    (A15 Pin 5). Each input is recorded, as passed, with the sha256 of the bytes it was parsed
    from; an absolute or ``..`` input path is refused rather than recorded.
    """
    for path in args.manifests:
        require_repo_relative(path, what="--manifests entry")
    out = Path(args.out).resolve()
    if out in {Path(path).resolve() for path in args.manifests}:
        raise TwoStageError(f"--out {args.out} is one of the union's own inputs")
    refuse_v1_output(args.out)
    raws = [(path, Path(path).read_bytes()) for path in args.manifests]
    union = union_payload_manifests([parse_payload_manifest(raw, where=path) for path, raw in raws])
    write_json(
        args.out,
        {
            "schema_version": SCHEMA_VERSION,
            "step": STEP,
            "union_of": [
                {"path": path, "sha256": hashlib.sha256(raw).hexdigest()} for path, raw in raws
            ],
            "payloads": union,
        },
    )
    print(f"wrote {args.out}: {len(union)} payloads, the union of {len(raws)} manifests")
    return 0


def repo_relative(path: str | Path) -> str:
    """A committed artifact records repo-relative paths, never a machine's absolute ones.

    An absolute path in a public report leaks the author's username and directory layout and
    means nothing to anyone else — the exact finding CodeRabbit raised on PR #103, and again on
    #108 for ``scripts/mint_two_stage_fixture.py``, which is why this is public and shared
    rather than copied there. Refused rather than trimmed when the path is outside the repo:
    silently rewriting it would put a path in the report that does not resolve from the repo
    root.
    """
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[3]
    try:
        return str(resolved.relative_to(root))
    except ValueError as error:
        raise TwoStageError(
            f"{path} resolves outside the repository ({resolved}); a committed report records "
            "repo-relative paths, and an absolute one would leak this machine's layout"
        ) from error


def _cmd_run(args: argparse.Namespace) -> int:
    """LEG 4 (torch-free — the leg CI runs): replay both artifacts → table + report + digest."""
    # Both outputs, before the replay: A15 Pin 5 keeps every v1 artifact byte-unchanged, and
    # this leg writes two paths rather than one ([[two-outputs-one-path-destroys-the-first]]).
    refuse_v1_output(args.report)
    refuse_v1_output(args.table)
    temperature = (
        read_temperature(args.temperature_from)
        if args.temperature_from is not None
        else args.temperature
    )
    result = run_two_stage(
        read_contigs(args.contigs),
        read_stage1(args.stage1),
        read_stage2(args.stage2),
        threshold_scope=args.threshold_scope,
        threshold=args.threshold,
        min_span=args.min_span,
        gap_merge=args.gap_merge,
        min_distinct_elements=args.min_distinct_elements,
        flank=args.flank,
        min_order_margin=args.min_order_margin,
        temperature=temperature,
        stage2_operating_point=args.stage2_operating_point,
        source_prior=args.source_prior,
        target_prior=args.target_prior,
    )
    report = dict(result.report)
    report["scope"] = dict(report["scope"])
    report["scope"]["stage1_source"] = repo_relative(args.stage1)
    report["scope"]["stage2_source"] = repo_relative(args.stage2)
    report["rule"] = dict(report["rule"])
    report["rule"]["temperature_source"] = (
        repo_relative(args.temperature_from) if args.temperature_from is not None else "cli"
    )
    problems = strand_robustness_problems(report)
    if problems:
        for problem in problems:
            print(f"REPORT PROBLEM: {problem}")
        return 1
    print(f"candidate_table digest: {result.digest}")
    # Compared BEFORE anything is written. The expectation exists to protect the committed
    # golden, and the earlier ordering wrote over it and then reported the mismatch — the guard
    # destroying the artifact it guards (CodeRabbit r2).
    if args.expect_digest and args.expect_digest != result.digest:
        print(f"DIGEST MISMATCH: expected {args.expect_digest}")
        print("refused to write; the committed artifacts are untouched")
        return 1
    write_json(args.report, report)
    write_json(
        args.table,
        {
            "schema_version": SCHEMA_VERSION,
            "step": STEP,
            "columns": list(CANDIDATE_COLUMNS),
            "digest": result.digest,
            "rows": list(result.rows),
        },
    )
    print(f"wrote {args.report} and {args.table}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("stage1", help="torch/ml-dna: contigs -> per-window Stage-1 logits")
    one.add_argument("--contigs", required=True)
    one.add_argument("--checkpoint", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--device", default=None)
    one.add_argument("--batch-size", type=int, default=8)
    one.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    one.add_argument("--max-log-prob-delta", type=float, default=MAX_MINT_LOG_PROB_DELTA)
    one.add_argument(
        "--shard",
        type=parse_shard,
        default=None,
        help="I/N: score only slice I of N of the id-sorted contigs; reassemble with merge-stage1",
    )
    one.set_defaults(func=_cmd_stage1)

    one_b = sub.add_parser(
        "merge-stage1", help="torch-free: stage1 --shard archives -> one Stage-1 archive"
    )
    one_b.add_argument("--contigs", required=True, help="the contig file every shard was cut from")
    one_b.add_argument("--shards", nargs="+", required=True, help="the shard archives, I = 0..N-1")
    one_b.add_argument("--out", required=True)
    one_b.set_defaults(func=_cmd_merge_stage1)

    two = sub.add_parser("payloads", help="torch-free: contigs + logits -> Stage-2 work list")
    two.add_argument("--contigs", required=True)
    two.add_argument("--stage1", required=True)
    two.add_argument("--out", required=True)
    _rule_args(two)
    two.set_defaults(func=_cmd_payloads)

    three = sub.add_parser("stage2", help="torch/ml-rna: work list -> binary logits")
    three.add_argument("--payloads", required=True)
    three.add_argument("--checkpoint-root", required=True)
    three.add_argument("--sweep-dir", required=True)
    three.add_argument("--out", required=True)
    three.add_argument("--arm", default=None)
    three.add_argument("--device", default=None)
    three.add_argument("--batch-size", type=int, default=4)
    three.add_argument(
        "--shard",
        type=parse_shard,
        default=None,
        help=(
            "I/N: score only slice I of N of the row-id-sorted manifest and record which; "
            "reassemble with merge-stage2"
        ),
    )
    three.set_defaults(func=_cmd_stage2)

    merge = sub.add_parser(
        "merge-stage2", help="torch-free: stage2 --shard artifacts -> one stage2 artifact"
    )
    merge.add_argument("--payloads", required=True, help="the manifest every shard was cut from")
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=_cmd_merge_stage2)

    union = sub.add_parser(
        "payloads-union", help="torch-free: several payload manifests -> one Stage-2 work list"
    )
    union.add_argument("--manifests", nargs="+", required=True)
    union.add_argument("--out", required=True)
    union.set_defaults(func=_cmd_payloads_union)

    four = sub.add_parser("run", help="torch-free: replay both -> candidate table + diagnostic")
    four.add_argument("--contigs", required=True)
    four.add_argument("--stage1", required=True)
    four.add_argument("--stage2", required=True)
    four.add_argument("--report", required=True)
    four.add_argument("--table", required=True)
    four.add_argument("--expect-digest", default=None)
    _rule_args(four)
    temperature = four.add_mutually_exclusive_group(required=True)
    temperature.add_argument("--temperature", type=float, default=None)
    temperature.add_argument(
        "--temperature-from",
        default=None,
        help="a GATE-2 report to read gate.calibration.temperature from (P3-10)",
    )
    four.add_argument("--stage2-operating-point", type=float, required=True)
    four.add_argument("--source-prior", type=float, default=None)
    four.add_argument("--target-prior", type=float, default=None)
    four.set_defaults(func=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())


__all__ = [
    "ADR",
    "CANDIDATE_COLUMNS",
    "CONTIG_REQUIRED_KEYS",
    "CONTIG_TRUTH_KEYS",
    "DIGEST_QUANTUM",
    "ENV_LOCK",
    "GENERATED_BY",
    "MAX_MINT_LOG_PROB_DELTA",
    "PRD",
    "RULE_PARAMETERS",
    "SCHEMA_VERSION",
    "STAGE2_ARTIFACT_KEYS",
    "STAGE2_IDENTITY_FIELDS",
    "STAGE2_RUN_FIELDS",
    "STAGE2_SCHEMA_VERSION",
    "STEP",
    "ContigRun",
    "TwoStageError",
    "TwoStageResult",
    "build_parser",
    "build_report",
    "build_rows",
    "candidate_table_digest",
    "duplicate_rna_agreement",
    "lookup_logit",
    "main",
    "manifest_row_ids",
    "merge_stage2_shards",
    "union_payload_manifests",
    "no_rule_parameter_has_a_default",
    "normalise_contig",
    "normalise_contigs",
    "parse_payload_manifest",
    "parse_shard",
    "parse_threshold",
    "payload_key",
    "payload_manifest",
    "posteriors_for",
    "read_contigs",
    "read_stage1",
    "read_stage2",
    "read_temperature",
    "refuse_v1_output",
    "repo_relative",
    "reconcile_contig",
    "require_repo_relative",
    "row_ids_sha256",
    "run_contig",
    "run_two_stage",
    "shard_bounds",
    "shard_manifest",
    "stage2_artifact",
    "stage2_env_lock",
    "stage2_identity",
    "strand_robustness",
    "strand_robustness_problems",
    "write_json",
]
