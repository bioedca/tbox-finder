"""P3-16 — two-stage vs Stage-1-only, the P3 exit gate: **AUPRC gated, matched recall reported**.

What is being compared, and why it is a fair comparison
-------------------------------------------------------
PRD §18.1's P3 exit gate is *"Two-stage beats Stage-1-only precision"*, i.e. whether the
Stage-2 re-ranker buys precision over the Stage-1 scanner alone. Both systems here share
**one** candidate-generation stage — the same checkpoint, the same ADR-0005 D3 locus rule,
the same emitted loci — and differ **only** in the score used to rank a candidate:

* **Stage-1-only** ranks by ``peak_p_elem`` — the maximum of ``1 − P(background)`` over the
  locus run, i.e. the strength of the Stage-1 call (uncalibrated, ADR-0005 A11 Pin 4: every
  Stage-1 number in this repo is on the uncalibrated posterior). **Peak, not mean**: P3-11
  measured ``mean_p_elem`` swinging 0.6453 → 0.9997 with the 512-nt window phase on one
  unchanged locus, so ranking on it would partly rank window alignment.
* **Two-stage** ranks by ``stage2_named_posterior``, the GATE-2 *named* posterior
  (temperature-scaled, pre prior-shift; P3-07 stack, P3-10's fitted ``T``).

Sharing the generator is what makes the comparison answer the gate's question rather than a
different one. It also fixes both systems' recall ceiling to Stage-1's, so a matched recall
is always reachable by both and the gate can never be won by an arm that simply calls more.

Precision at matched recall is the mirror of GATE-1's recall at matched precision
---------------------------------------------------------------------------------
:func:`tbox_finder.metrics.precision_at_matched_recall` is the kernel — max precision over
the thresholds an arm can actually reach, subject to ``recall >= R*``. ``R*`` is the recall
the **deployed** two-stage system achieves at its operating point, so the question the
number answers is: *at the sensitivity the shipped two-stage system runs at, what would
Stage 1 alone have cost in precision?*

What is gated, and why it is not the matched-recall number (ADR-0005 A13)
-------------------------------------------------------------------------
``imp.md``'s P3-16 block asks for precision at matched recall. **Measured, that quantity
cannot carry a P3 verdict**, and the reason is in ADR-0005 itself: D3 freezes the Stage-1
threshold *and* the Stage-2 operating point at the §13.1 phase gate — **P5-01, two phases
after this gate**. Read at the values this run used, the gated arm's matched-recall gain is
**−0.039 pp** at Stage-1 τ = 0.5 and **+6.34 / +6.18 pp** at τ = 0.7 / 0.9: the same system,
three answers, none of them at a pinned point. Picking the passing one would be exactly the
threshold-shopping CLAUDE.md §8.5 forbids.

The gated statistic is therefore **AUPRC at the D7 pinned decoy prevalence** — PRD §12's
named primary imbalance-aware metric, threshold-free, and so independent of both unfrozen
knobs. The full matched-recall comparison (point, block-bootstrap CI, a six-point recall
grid, and the Stage-1-threshold sweep with each point bound to its own report's digest) is
**retained and reported, gated on none of it**.

⚠ **AUPRC is not prevalence-invariant the way matched-recall precision is.** For a fixed pair
of thresholds the *sign* of a precision difference is invariant under any reweighting
``lambda`` (precision is monotone in ``fp/tp``), which is why the reported matched-recall
verdict is checked for exactly that invariance. Two PR **curves**, however, can cross, so an
AUPRC ordering can in principle move with prevalence: the gated value is computed at D7's
pinned 100 : 1 and the whole 10 : 1 → 10⁴ : 1 sweep is published beside it rather than
assumed to agree. ⚠ D7's own scoping note (``conf/data/decoys.yaml``) says the 100 : 1 pin is
*"applied in P4, NOT here"*; it is used here because ``imp.md``'s P3-16 block asks for it,
with the benchmark's own composition stated alongside.

This benchmark cannot physically hold 100 decoys per positive, so each sampled negative is
**reweighted** to stand for ``lambda = ratio · n_pos / n_neg`` population negatives
(:func:`prevalence_lambda`, :func:`reweighted_precision`,
:func:`~tbox_finder.metrics.average_precision_reweighted`).

The two Stage-1 arms, and which one the gate grades
----------------------------------------------------
The shipped P2-10d′-b scanner trained on the **whole** in-distribution fold (8,303
``nested_train`` records, ``exclude_selection_val=false``), so it has no in-distribution
held-out population at all — ``docs/model_card.md`` and ``reports/p2/gate4.json`` say so,
and GATE-4 had to grade a **twin** (``exclude_gate4_eval=true``) for the same reason.

Per the §7 decision of 2026-08-13 this module reports **both** arms on the identical item
set and grades the twin:

* ``twin`` — **gated**. Held out every one of the benchmark's positives (measured: 0 of
  1,201 seen), and Stage-2 held them out too (the benchmark is drawn from the ``test`` rung
  and Stage-2 admits only ``train``). This is the project's first end-to-end
  leakage-controlled in-distribution measurement.
* ``production`` — **reported, not gated**. The shipped scanner, scored on the same items,
  with its Stage-1-in-sample status carried per item. Its bias direction is known and
  stated: an in-sample Stage-1 flatters the *Stage-1-only* arm, so a pass there is
  conservative and a fail there is uninterpretable.

The gated verdict is re-derived by :func:`precision_problems` from the arm's own measured
exposure counts, not from the string ``"twin"`` — a report that graded the wrong arm, or
graded the twin on items the twin had seen, is refused rather than published.

Negatives: all four §9.1 pools, with per-pool exposure carried
---------------------------------------------------------------
Per the same §7 decision the gated denominator is all four §9.1 decoy classes as PRD §2.3
specifies. Two facts about them are measured, not assumed, and are disclosures rather than
exclusions: **all 2,999 ``structured_rna`` decoys were embedded in both Stage-1 trainings**
(ADR-0005 A7 pin 3), so that pool's contribution partly measures memorisation; and
``leader_decoy`` — §9.1's hardest class — contributes **one** item, and all 8 of its records
are T-box-derived with 2 exact substrings of training positives (A7 pin 5, re-sourcing
deferred to P2-10b′). The per-pool false-positive breakdown is in the report so neither has
to be taken on trust.

What this benchmark can and cannot resolve, said out loud
---------------------------------------------------------
In distribution, **Stage-1 alone is already ~99.7 % precise**: on 692 §9.1 decoys presented
as scan windows the gated arm's Stage-1 emits a candidate for **6** and retains **4** as
false positives at the matched-recall point; the two-stage system retains **4** as well, of
a partly different composition (Stage-2 rejects the ``leader_decoy``, admits one more
``structured_rna``). A four-versus-four comparison has essentially no power — the
matched-recall CI is ±17 pp — and no amount of care in the estimator changes that. The
precision headroom Stage-2 exists to buy is a **genome-scale** quantity (P5's FDR) and a
**held-out-clade** quantity (P4's GATE-1); this gate is an in-distribution reference, and it
is reported as one.

⚠ An earlier construction of this benchmark presented each item as its own excised contig
and yielded **1** discriminating negative of 692. Rebuilding in scan geometry (locus in real
±context; decoys spliced into real mined host windows by the shipped ``embed_decoy_rows``)
raised that to 4–6. It did not manufacture headroom that is not there, and the failed
attempt is recorded rather than dropped.

PRD §2.3, §6, §12, §18.1; ADR-0005 D3, D5, D7, A7, A11 Pin 4, A12, **A13**; ADR-0004 A6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder.eval.gate4 import json_safe
from tbox_finder.eval.resample import DEFAULT_N_BOOT, block_bootstrap
from tbox_finder.integration.two_stage import repo_relative, write_json
from tbox_finder.metrics import (
    average_precision,
    average_precision_reweighted,
    precision_at_matched_recall,
    precision_recall_at_threshold,
)
from tbox_finder.power import DECOY_PREVALENCE
from tbox_finder.provenance import build_provenance

# 1.1 renames the completeness clause `verdict_invariant_across_prevalence` to
# `matched_recall_verdict_invariant_across_prevalence` and adds
# `benchmark.scope.host_pool.overlap_with_training_folds` plus
# `stage1_threshold_sensitivity.base` / `.brackets_base` — a published-contract change, so the
# version moves with it rather than leaving two incompatible shapes under one number.
SCHEMA_VERSION = "1.1"
STEP = "P3-16"
GENERATED_BY = "src/tbox_finder/integration/precision.py"
ADR = (
    "ADR-0005 D3 (locus rule + Stage-2 operating point), D5 (block-level CI), "
    "D7 (benchmark decoy prevalence + sweep), A7 (which decoy pools were trained on), "
    "A11 Pin 4 (Stage-1 read on the uncalibrated posterior), A12 (the P2 checkpoint is "
    "canonical); ADR-0004 A6 (eval twin / two-run split)"
)
PRD = "PRD §2.3, §6, §12, §18.1"
ENV_LOCK = "envs/data.conda-lock.yml"

#: Score for an item the shared candidate stage emitted **nothing** for. Both posteriors
#: live in [0, 1], so this is below every reachable score — and it is never offered as a
#: threshold (see ``candidates=`` on :func:`~tbox_finder.metrics.precision_at_matched_recall`),
#: because a re-ranker cannot call a candidate that was never generated.
UNCALLED_SCORE = -1.0

#: The arm whose numbers the P3 exit gate is graded on (§7 decision 2026-08-13). The name is
#: a *label*; :func:`precision_problems` re-derives that this arm actually held the graded
#: population out, so a rename cannot move the grade.
GATED_ARM = "twin"

#: ADR-0005 D7's sweep, "10 : 1 → 10² : 1 → 10³ : 1 → 10⁴ : 1 (toward genome scale)".
PREVALENCE_SWEEP: tuple[int, ...] = (10, 100, 1_000, 10_000)

#: The matched-recall grid reported beside the gated point, so the verdict is visibly not a
#: one-point artefact. Points above an arm's reachable ceiling are recorded as unreachable.
RECALL_GRID: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

SYSTEMS: tuple[str, ...] = ("stage1_only", "two_stage")

#: Candidate-table columns each system ranks by.
SYSTEM_COLUMN: dict[str, str] = {
    "stage1_only": "peak_p_elem",
    "two_stage": "stage2_named_posterior",
}


class PrecisionError(ValueError):
    """A malformed input or an impossible composition — never a silently degraded number."""


# ══════════════════════════════════════════════════════════════════════════════════════
# Item folding: candidate-table rows (locus × strand) → one row per benchmark item
# ══════════════════════════════════════════════════════════════════════════════════════
def item_scores(
    items: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Fold the candidate table onto the benchmark's items.

    An item is *called* by a system when the shared candidate stage emitted at least one
    locus for it and that locus scores at or above the system's threshold; the item's score
    is therefore the **maximum** over its emitted (locus, strand) rows. Max — not mean, not
    first — because detection is existential: one confident locus in a decoy is a false
    positive however many unconfident ones sit beside it.

    Refuses a table row whose ``contig_id`` is not a benchmark item: the candidate table and
    the item manifest must describe the same run, and a stray id means they do not.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        contig_id = item["contig_id"]
        if contig_id in by_id:
            raise PrecisionError(f"duplicate benchmark item {contig_id!r}")
        label = item["label"]
        if label not in (0, 1):
            raise PrecisionError(f"item {contig_id!r} carries label {label!r}, not 0 or 1")
        by_id[contig_id] = {
            "contig_id": contig_id,
            "label": int(label),
            "pool": str(item["pool"]),
            "block": str(item["block"]),
            "seen_by": dict(item["seen_by"]),
            "n_rows": 0,
            **{system: UNCALLED_SCORE for system in SYSTEMS},
        }
    for row in rows:
        contig_id = row["contig_id"]
        entry = by_id.get(contig_id)
        if entry is None:
            raise PrecisionError(
                f"candidate table row names contig {contig_id!r}, which is not a benchmark "
                "item — the table and the item manifest are not from the same run"
            )
        entry["n_rows"] += 1
        for system in SYSTEMS:
            value = row[SYSTEM_COLUMN[system]]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                raise PrecisionError(
                    f"contig {contig_id!r}: {SYSTEM_COLUMN[system]} is {value!r}; a missing "
                    "score would silently rank the candidate last rather than fail"
                )
            entry[system] = max(entry[system], float(value))
    return [by_id[key] for key in sorted(by_id)]


def fold_arms(
    benchmark: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Both arms' candidate tables → the committed **per-item score table**.

    Two legs, for the same reason P3-14 has four: the candidate tables are ~4,600 rows × 40
    columns each and are produced from GPU output that no CI can re-derive, while the graded
    computation is a few hundred lines of arithmetic that CI *must* be able to re-run. This
    leg folds the model output down to what the grade actually reads — one row per benchmark
    item per arm — and that reduced table is what gets committed and replayed.
    """
    items = benchmark["items"]
    per_arm = {arm: item_scores(items, rows) for arm, rows in sorted(tables.items())}
    order = [item["contig_id"] for item in items]
    by_arm_id = {arm: {row["contig_id"]: row for row in scored} for arm, scored in per_arm.items()}
    folded = []
    for contig_id in sorted(order):
        first = next(iter(by_arm_id))
        base = by_arm_id[first][contig_id]
        folded.append(
            {
                "contig_id": contig_id,
                "label": base["label"],
                "pool": base["pool"],
                "block": base["block"],
                "seen_by": base["seen_by"],
                "arms": {
                    arm: {
                        "n_rows": by_arm_id[arm][contig_id]["n_rows"],
                        **{system: by_arm_id[arm][contig_id][system] for system in SYSTEMS},
                    }
                    for arm in sorted(by_arm_id)
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "benchmark": {
            "scope": benchmark["scope"],
            "source": {
                key: value
                for key, value in benchmark.get("source", {}).items()
                if key != "embedded_by_pool"
            },
            "embedded_by_pool": benchmark.get("source", {}).get("embedded_by_pool"),
        },
        "arms": sorted(by_arm_id),
        "items": folded,
    }


def arm_items(folded: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    """One arm's view of the folded table, in the shape :func:`arm_metrics` reads.

    Refuses an arm the table does not carry rather than returning an empty list, which would
    be scored as a benchmark on which nothing was called."""
    if arm not in folded["arms"]:
        raise PrecisionError(f"the folded score table carries no arm {arm!r}")
    out = []
    for item in folded["items"]:
        if arm not in item["arms"]:
            raise PrecisionError(f"item {item['contig_id']!r} carries no scores for arm {arm!r}")
        entry = item["arms"][arm]
        out.append(
            {
                "contig_id": item["contig_id"],
                "label": int(item["label"]),
                "pool": str(item["pool"]),
                "block": str(item["block"]),
                "seen_by": item["seen_by"],
                "n_rows": int(entry["n_rows"]),
                **{system: float(entry[system]) for system in SYSTEMS},
            }
        )
    return out


def arrays(scored: Sequence[Mapping[str, Any]], system: str) -> tuple[list[int], list[float]]:
    """``(y_true, y_score)`` for one system, in the item order given."""
    return [int(item["label"]) for item in scored], [float(item[system]) for item in scored]


def reachable_scores(scored: Sequence[Mapping[str, Any]], system: str) -> list[float]:
    """The thresholds a system can actually operate at — the scores of items the shared
    candidate stage emitted something for. An uncalled item's sentinel is excluded, so it
    can never become a threshold at which the item is called."""
    return sorted({float(item[system]) for item in scored if item["n_rows"] > 0})


# ══════════════════════════════════════════════════════════════════════════════════════
# Prevalence reweighting (ADR-0005 D7)
# ══════════════════════════════════════════════════════════════════════════════════════
def prevalence_lambda(n_positives: int, n_negatives: int, decoy_ratio: float) -> float:
    """How many population negatives one sampled negative stands for at ``decoy_ratio``
    decoys per positive. ``lambda = ratio * n_pos / n_neg``."""
    if n_negatives <= 0:
        raise PrecisionError("cannot reweight a benchmark with no negatives")
    if n_positives <= 0:
        raise PrecisionError("cannot reweight a benchmark with no positives")
    if decoy_ratio <= 0:
        raise PrecisionError(f"decoy_ratio must be positive, got {decoy_ratio!r}")
    return decoy_ratio * n_positives / n_negatives


def reweighted_precision(tp: int, fp: int, lam: float) -> float:
    """``tp / (tp + lambda*fp)`` — precision at a reweighted decoy prevalence. NaN when the
    threshold calls nothing (0/0), which is the same undefined precision the unweighted
    kernel reports rather than a fabricated 0 or 1."""
    denominator = tp + lam * fp
    return float("nan") if denominator <= 0 else tp / denominator


def confusion_at(y_true: Sequence[int], y_score: Sequence[float], threshold: float) -> dict:
    tp = fp = fn = 0
    for label, score in zip(y_true, y_score, strict=True):
        called = score >= threshold
        if called and label == 1:
            tp += 1
        elif called:
            fp += 1
        elif label == 1:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn}


# ══════════════════════════════════════════════════════════════════════════════════════
# The vectorised twin of the kernel — bootstrap only, and checked against the kernel
# ══════════════════════════════════════════════════════════════════════════════════════
def _fast_precision_at_matched_recall(
    labels: np.ndarray, scores: np.ndarray, target_recall: float, *, lam: float
) -> float:
    """The bootstrap's inner loop: max reweighted precision subject to ``recall >= R*``.

    A vectorised twin of :func:`~tbox_finder.metrics.precision_at_matched_recall` exists
    only because a 2,000-replicate block bootstrap over a ~1,900-item benchmark is ~10⁹
    stdlib operations. It is **not** a second definition: the published point estimates all
    come from the stdlib kernel, and ``test_fast_twin_matches_the_stdlib_kernel`` asserts
    the two agree on the real benchmark at every grid point and every sweep ratio, so a
    divergence is a red test rather than a quietly different CI.

    Uncalled items (sentinel score) are never selected because the sentinel is excluded from
    the threshold set, exactly as ``candidates=`` does in the kernel.
    """
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y != 1)
    # One point per distinct score (the tie group's cumulative counts), sentinel excluded.
    last = np.r_[s[1:] != s[:-1], True] & (s > UNCALLED_SCORE)
    tp, fp = tp[last], fp[last]
    recall = tp / n_pos
    admissible = recall >= target_recall
    if not admissible.any():
        return float("nan")
    denominator = tp + lam * fp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(denominator > 0, tp / denominator, np.nan)
    admitted = precision[admissible]
    if not np.isfinite(admitted).any():
        return float("nan")
    return float(np.nanmax(admitted))


def _fast_average_precision(labels: np.ndarray, scores: np.ndarray, *, lam: float) -> float:
    """Vectorised twin of :func:`~tbox_finder.metrics.average_precision_reweighted`.

    Same standing as :func:`_fast_precision_at_matched_recall`: bootstrap-only, never the
    published point estimate, and checked against the stdlib kernel on the real benchmark at
    every sweep ratio by ``test_fast_auprc_twin_matches_the_stdlib_kernel``. Ties are grouped
    on the score value exactly as the kernel does, so the two agree on tied inputs and not
    merely on distinct ones.
    """
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.lexsort((-labels, -scores))
    s = scores[order]
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y != 1)
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[last], fp[last]
    recall = tp / n_pos
    precision = tp / (tp + lam * fp)
    delta = np.diff(recall, prepend=0.0)
    return float(np.sum(delta * precision))


# ══════════════════════════════════════════════════════════════════════════════════════
# One arm
# ══════════════════════════════════════════════════════════════════════════════════════
def arm_metrics(
    scored: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    stage2_operating_point: float,
    decoy_prevalence: int,
    prevalence_sweep: Sequence[int],
    recall_grid: Sequence[float],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Every number for one Stage-1 checkpoint's arm."""
    y_true, s_two = arrays(scored, "two_stage")
    _, s_one = arrays(scored, "stage1_only")
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    n_candidate_items = sum(1 for item in scored if item["n_rows"] > 0)
    n_positive_candidates = sum(1 for item in scored if item["n_rows"] > 0 and item["label"] == 1)
    ceiling_recall = n_positive_candidates / n_pos if n_pos else float("nan")

    # The deployed operating point fixes R*: "at the sensitivity the shipped two-stage
    # system runs at, what would Stage 1 alone have cost in precision?"
    op_precision, op_recall = precision_recall_at_threshold(y_true, s_two, stage2_operating_point)
    target_recall = op_recall

    candidates = {system: reachable_scores(scored, system) for system in SYSTEMS}
    selected = {
        system: precision_at_matched_recall(
            y_true,
            arrays(scored, system)[1],
            target_recall,
            candidates=candidates[system],
        )
        for system in SYSTEMS
    }
    confusion = {
        system: (
            confusion_at(y_true, arrays(scored, system)[1], selected[system]["threshold"])
            if selected[system]["matched"]
            else {"tp": 0, "fp": 0, "fn": n_pos}
        )
        for system in SYSTEMS
    }

    # Prevalence reads. The benchmark's own composition first, then D7's pin and sweep.
    observed_ratio = n_neg / n_pos if n_pos else float("nan")
    ratios: dict[str, float] = {"observed": observed_ratio}
    for ratio in sorted({int(decoy_prevalence), *(int(r) for r in prevalence_sweep)}):
        ratios[f"{ratio}:1"] = float(ratio)
    prevalence: dict[str, Any] = {}
    for name, ratio in ratios.items():
        lam = prevalence_lambda(n_pos, n_neg, ratio)
        point = {
            system: reweighted_precision(confusion[system]["tp"], confusion[system]["fp"], lam)
            for system in SYSTEMS
        }
        auprc = {
            system: average_precision_reweighted(
                y_true, arrays(scored, system)[1], decoy_weight=lam
            )
            for system in SYSTEMS
        }
        gain = (point["two_stage"] - point["stage1_only"]) * 100.0
        auprc_gain = (auprc["two_stage"] - auprc["stage1_only"]) * 100.0
        prevalence[name] = {
            "decoy_ratio": ratio,
            "lambda": lam,
            "precision": point,
            "gain_pp": gain,
            "auprc": auprc,
            "auprc_gain_pp": auprc_gain,
            # REPORTED, not gated (ADR-0005 A13): a threshold-dependent verdict at P3 would
            # rest on the Stage-1 threshold and the Stage-2 operating point that D3 does not
            # freeze until the §13.1 phase gate at P5-01.
            "two_stage_beats_stage1_only": bool(
                not math.isnan(gain) and point["two_stage"] > point["stage1_only"]
            ),
            "two_stage_auprc_beats_stage1_only": bool(
                not math.isnan(auprc_gain) and auprc["two_stage"] > auprc["stage1_only"]
            ),
        }

    # The gated read is at D7's pinned prevalence; the verdict is invariant across the sweep
    # and a clause checks that rather than leaving it as a claim in this docstring.
    gated_key = f"{int(decoy_prevalence)}:1"
    gated = prevalence[gated_key]
    # ADR-0005 A13: the gated statistic is **AUPRC at the D7 pinned prevalence** — PRD §12's
    # named primary imbalance-aware metric, and the only form of this comparison that does not
    # read the system at operating points D3 leaves unfrozen until P5-01. The matched-recall
    # comparison is retained in full and REPORTED, non-gated.
    passes = bool(gated["two_stage_auprc_beats_stage1_only"])

    # Block-level CI on the gain, at the *fixed* point-estimate R* (ADR-0005 D5). Holding
    # R* fixed keeps the interval about the precision gap; letting each replicate re-pick
    # its own R* would fold the operating point's sampling error into the same number.
    lam_gated = prevalence_lambda(n_pos, n_neg, float(int(decoy_prevalence)))
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for item in scored:
        by_block.setdefault(item["block"], []).append(item)

    def _cols(sample: list, system: str) -> tuple[np.ndarray, np.ndarray]:
        labels = np.fromiter((int(i["label"]) for i in sample), dtype=np.int64, count=len(sample))
        scores = np.fromiter(
            (float(i[system]) for i in sample), dtype=np.float64, count=len(sample)
        )
        return labels, scores

    def gain_statistic(sample: list) -> float:
        labels, two = _cols(sample, "two_stage")
        _, one = _cols(sample, "stage1_only")
        p_two = _fast_precision_at_matched_recall(labels, two, target_recall, lam=lam_gated)
        p_one = _fast_precision_at_matched_recall(labels, one, target_recall, lam=lam_gated)
        return (p_two - p_one) * 100.0

    def auprc_gain_statistic(sample: list) -> float:
        """The **gated** statistic's replicate (ADR-0005 A13)."""
        labels, two = _cols(sample, "two_stage")
        _, one = _cols(sample, "stage1_only")
        return (
            _fast_average_precision(labels, two, lam=lam_gated)
            - _fast_average_precision(labels, one, lam=lam_gated)
        ) * 100.0

    blocks = [by_block[key] for key in sorted(by_block)]
    ci = block_bootstrap(blocks, gain_statistic, n_boot=n_boot, seed=seed)
    auprc_ci = block_bootstrap(blocks, auprc_gain_statistic, n_boot=n_boot, seed=seed)

    grid = []
    for target in sorted(recall_grid):
        point = {}
        for system in SYSTEMS:
            match = precision_at_matched_recall(
                y_true, arrays(scored, system)[1], target, candidates=candidates[system]
            )
            hit = (
                confusion_at(y_true, arrays(scored, system)[1], match["threshold"])
                if match["matched"]
                else {"tp": 0, "fp": 0, "fn": n_pos}
            )
            point[system] = {
                "matched": match["matched"],
                "threshold": match["threshold"],
                "recall": match["recall"],
                "precision_observed": match["precision"],
                "precision_at_pin": reweighted_precision(hit["tp"], hit["fp"], lam_gated),
            }
        both = point["two_stage"]["matched"] and point["stage1_only"]["matched"]
        grid.append(
            {
                "target_recall": target,
                "reachable": bool(target <= ceiling_recall),
                **point,
                "two_stage_beats_stage1_only": bool(
                    both
                    and point["two_stage"]["precision_at_pin"]
                    > point["stage1_only"]["precision_at_pin"]
                ),
            }
        )

    # Where each arm's false positives come from, so the two disclosed pools (memorised
    # structured_rna, the single T-box-derived leader_decoy) can be read rather than trusted.
    per_pool: dict[str, dict[str, Any]] = {}
    for item in scored:
        if item["label"] == 1:
            continue
        entry = per_pool.setdefault(
            item["pool"],
            {
                "n": 0,
                "n_seen_by_arm": 0,
                **{f"fp_{system}": 0 for system in SYSTEMS},
            },
        )
        entry["n"] += 1
        entry["n_seen_by_arm"] += int(bool(item["seen_by"][arm]))
        for system in SYSTEMS:
            if selected[system]["matched"] and item[system] >= selected[system]["threshold"]:
                entry[f"fp_{system}"] += 1

    return {
        "arm": arm,
        "gated": arm == GATED_ARM,
        "exposure": {
            "n_positives_seen_by_arm": sum(
                1 for item in scored if item["label"] == 1 and item["seen_by"][arm]
            ),
            "n_negatives_seen_by_arm": sum(
                1 for item in scored if item["label"] == 0 and item["seen_by"][arm]
            ),
        },
        "population": {
            "n_items": len(scored),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "n_blocks": len(by_block),
            "n_candidate_items": n_candidate_items,
            "n_positive_candidates": n_positive_candidates,
            "ceiling_recall": ceiling_recall,
            "observed_decoy_ratio": observed_ratio,
        },
        "operating_point": {
            "stage2_operating_point": stage2_operating_point,
            "two_stage_precision": op_precision,
            "two_stage_recall": op_recall,
            "target_recall": target_recall,
        },
        "matched_recall": {system: {**selected[system], **confusion[system]} for system in SYSTEMS},
        "prevalence": prevalence,
        "gated_prevalence_key": gated_key,
        "gain_pp": gated["gain_pp"],
        "gain_ci": ci,
        "auprc_gain_pp": gated["auprc_gain_pp"],
        "auprc_gain_ci": auprc_ci,
        "auprc": {
            system: average_precision(y_true, arrays(scored, system)[1]) for system in SYSTEMS
        },
        # AUPRC twice, because the two readings answer different questions and publishing
        # only one would hide the difference. The first ranks EVERY benchmark item, so its
        # terminal segment is the shared Stage-1 ceiling — identical for both systems, which
        # is why the comparison stays fair. The second ranks only the items the shared
        # candidate stage emitted something for, i.e. the re-ranking problem the two systems
        # actually differ on.
        "auprc_candidates_only": {
            system: average_precision(
                [int(item["label"]) for item in scored if item["n_rows"] > 0],
                [float(item[system]) for item in scored if item["n_rows"] > 0],
            )
            for system in SYSTEMS
        },
        "recall_grid": grid,
        "per_pool": per_pool,
        "passes": passes,
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# The report
# ══════════════════════════════════════════════════════════════════════════════════════
def precision_gain(
    folded: Mapping[str, Any],
    *,
    stage2_operating_point: float,
    decoy_prevalence: int,
    prevalence_sweep: Sequence[int],
    recall_grid: Sequence[float],
    n_boot: int,
    seed: int,
    sources: Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """The P3-16 artifact: both arms measured on one item set, the twin's arm gated.

    ``folded`` is :func:`fold_arms`' committed per-item score table. The gated arm must be
    present, or there is no grade to publish — an absent arm must not degrade to a report
    that grades whatever else it found.
    """
    benchmark = folded["benchmark"]
    if GATED_ARM not in folded["arms"]:
        raise PrecisionError(
            f"the folded score table carries no arm {GATED_ARM!r}; the P3 exit gate is graded "
            f"on it (§7 decision 2026-08-13), so a report without it has no verdict to make"
        )
    arms = {
        arm: arm_metrics(
            arm_items(folded, arm),
            arm,
            stage2_operating_point=stage2_operating_point,
            decoy_prevalence=decoy_prevalence,
            prevalence_sweep=prevalence_sweep,
            recall_grid=recall_grid,
            n_boot=n_boot,
            seed=seed,
        )
        for arm in sorted(folded["arms"])
    }
    scope = dict(benchmark["scope"])
    completeness = {
        "all_benchmark_items_scored": all(
            arm["population"]["n_items"] == scope["n_items"] for arm in arms.values()
        ),
        "gated_arm_present": GATED_ARM in arms,
        "gated_arm_held_out_every_positive": (
            arms[GATED_ARM]["exposure"]["n_positives_seen_by_arm"] == 0
        ),
        "both_systems_reached_the_target_recall": all(
            arms[GATED_ARM]["matched_recall"][system]["matched"] for system in SYSTEMS
        ),
        # A degenerate operating point makes every clause above pass while the gate compares
        # two systems at recall 0, where "max precision" is whichever arm happens to own the
        # single highest-scoring item. Refused rather than published.
        "target_recall_is_positive": arms[GATED_ARM]["operating_point"]["target_recall"] > 0.0,
        "candidate_stage_emitted_positives": (
            arms[GATED_ARM]["population"]["n_positive_candidates"] > 0
        ),
        # Named for the verdict it READS. `two_stage_beats_stage1_only` is the matched-recall
        # comparison, whose sign is λ-invariant by construction (every sweep point reweights
        # one fixed confusion), so this is a consistency check on the sweep — NOT a statement
        # about the gated AUPRC verdict, which ADR-0005 A13 Pin 3 declines to assert
        # invariant because two PR curves can cross. The old name claimed the second thing.
        "matched_recall_verdict_invariant_across_prevalence": len(
            {
                point["two_stage_beats_stage1_only"]
                for point in arms[GATED_ARM]["prevalence"].values()
            }
        )
        == 1,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "prd": PRD,
        "env_lock": ENV_LOCK,
        "generated_at_utc": generated_at_utc
        or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "benchmark": {
            "scope": scope,
            "source": benchmark.get("source", {}),
            "embedded_by_pool": benchmark.get("embedded_by_pool"),
        },
        "rule": {
            "stage2_operating_point": stage2_operating_point,
            "decoy_prevalence": decoy_prevalence,
            "prevalence_sweep": list(prevalence_sweep),
            "recall_grid": list(recall_grid),
            "n_boot": n_boot,
            "seed": seed,
            # ADR-0005 D3 freezes the Stage-1 threshold and the Stage-2 operating point at
            # the §13.1 phase gate (P5-01). Nothing here is that freeze.
            "pinned": False,
        },
        "gated_arm": GATED_ARM,
        "arms": arms,
        "completeness": completeness,
        "disclosures": disclosures(benchmark, arms),
        "sources": dict(sources or {}),
    }
    gated_point = arms[GATED_ARM]["prevalence"][arms[GATED_ARM]["gated_prevalence_key"]]
    report["gate"] = {
        "definition": (
            "PRD §18.1 P3 exit — two-stage beats Stage-1-only precision on the "
            "in-distribution benchmark, graded (ADR-0005 A13) on the THRESHOLD-FREE "
            "statistic: AUPRC at the D7 pinned decoy prevalence, PRD §12's primary "
            "imbalance-aware metric"
        ),
        "statistic": "auprc_at_pinned_prevalence",
        "arm": GATED_ARM,
        "decoy_prevalence": decoy_prevalence,
        "auprc": gated_point["auprc"],
        "gain_pp": arms[GATED_ARM]["auprc_gain_pp"],
        "gain_ci": arms[GATED_ARM]["auprc_gain_ci"],
        "passes": arms[GATED_ARM]["passes"],
        # Retained in full, REPORTED and NOT gated: ADR-0005 D3 freezes the Stage-1
        # threshold and the Stage-2 operating point at the §13.1 phase gate (P5-01), so a
        # P3 verdict read at either would rest on a value this phase cannot pin.
        "reported_not_gated": {
            "matched_recall": {
                "target_recall": arms[GATED_ARM]["operating_point"]["target_recall"],
                "precision": gated_point["precision"],
                "gain_pp": arms[GATED_ARM]["gain_pp"],
                "gain_ci": arms[GATED_ARM]["gain_ci"],
                "two_stage_beats_stage1_only": gated_point["two_stage_beats_stage1_only"],
            },
            "why": (
                "ADR-0005 D3 freezes the Stage-1 threshold and the Stage-2 operating point "
                "at P5-01, two phases after this gate; the matched-recall read is therefore "
                "reported at the values this run used and gated on neither"
            ),
        },
    }
    # Derived, never asserted: a truncated or half-measured run must not be able to land at
    # the phase-exit path flagged as real science ([[cost-knobs-can-certify]]).
    report["is_science"] = bool(all(completeness.values()))
    report["gated"] = report["is_science"]
    return json_safe(report)


def disclosures(benchmark: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What a reader must be told, phrased against measured values rather than adjectives."""
    scope = benchmark["scope"]
    seen = scope["seen_by_counts"]
    # Every quantity a disclosure states is read off the payload it is describing. A fixed
    # numeral here would survive a regeneration on different inputs and publish a false count
    # — the same defect, one sentence over, that round 1 caught in the geometry clause
    # (CodeRabbit, PR #133 round 2).
    admitted = {
        name: arm["population"]["n_candidate_items"] - arm["population"]["n_positive_candidates"]
        for name, arm in arms.items()
    }
    retained = {name: arm["matched_recall"]["two_stage"]["fp"] for name, arm in arms.items()}
    other_arms = ", ".join(
        f"{name} {admitted[name]}/{retained[name]}" for name in sorted(arms) if name != GATED_ARM
    )
    host_overlap = scope["host_pool"]["overlap_with_training_folds"]
    host_others = ", ".join(
        f"{name} {count}"
        for name, count in sorted(host_overlap["by_arm"].items())
        if name != GATED_ARM
    )
    out = [
        (
            f"The gated arm is the GATE-4 eval twin, not the shipped scanner. The shipped "
            f"P2-10d′-b checkpoint trained on {seen['production']['positives']} of the "
            f"{scope['n_positives']} benchmark positives (its training fold is the whole "
            f"in-distribution fold), so it has no in-distribution held-out population; the "
            f"twin saw {seen['twin']['positives']}. §7 decision 2026-08-13; ADR-0004 A6's "
            "two-run split is the same instrument GATE-4 used."
        ),
        (
            "The production arm's numbers are REPORTED, NOT GATED. Its Stage-1 is in-sample "
            "on every positive, which flatters the Stage-1-only system, so a pass there is "
            "conservative and a fail there is uninterpretable."
        ),
        (
            f"All four §9.1 decoy pools are in the gated denominator (§7 decision). "
            f"{seen['twin']['negatives']} of the {scope['n_negatives']} negatives were "
            "embedded in both Stage-1 trainings (ADR-0005 A7 pin 3 embeds every "
            "structured_rna decoy), so that pool's contribution partly measures memorisation."
        ),
        (
            f"The negatives' HOST windows are not unseen DNA either. Hosts are mined windows "
            f"whose mining-pool parent is out of fold, which constrains the PARENT and not "
            f"the window: {host_overlap['by_arm'].get(GATED_ARM)} of {scope['n_negatives']} "
            f"negatives share a verbatim {host_overlap['k']}-mer of host DNA with the gated "
            f"arm's own Stage-1 training fold ({host_others} for the arms reported beside "
            "it). The spliced decoy is excluded from that count; decoy-level exposure is the "
            "line above."
        ),
        (
            f"§9.1's hardest negative class contributes "
            f"{scope['negatives_by_pool'].get('leader_decoy', 0)} item(s) to THIS benchmark, "
            "drawn from an UPSTREAM pool of 8 leader decoys — a §9.1 pool size, not a count "
            "of benchmark items — all 8 of which are T-box-derived, 2 being exact substrings "
            "of training positives (A7 pin 5); re-sourcing is deferred to P2-10b′."
        ),
        (
            "The ADR-0005 D7 100:1 prevalence is reached by REWEIGHTING the sampled "
            "negatives, not by materialising them — this benchmark's own composition is "
            f"{arms[GATED_ARM]['population']['observed_decoy_ratio']:.3f} decoys per "
            "positive. conf/data/decoys.yaml scopes the D7 pin to P4; it is reported here "
            "because imp.md's P3-16 block asks for it, and the verdict is invariant across "
            "the whole 10:1 → 10⁴:1 sweep."
        ),
        (
            f"Benchmark items are {scope['geometry']}: a "
            "positive is its gate4_eval locus carved in real ±genomic context, a negative is "
            "a §9.1 decoy spliced into a real mined host window by the shipped "
            "embed_decoy_rows. An earlier build presented each item as its own excised "
            "contig and Stage-1 admitted a single decoy across the whole pool; in this "
            f"geometry the gated arm admits {admitted[GATED_ARM]} of {scope['n_negatives']} "
            f"and retains {retained[GATED_ARM]} at matched recall ({other_arms} for the "
            "arms reported beside it). This is an in-distribution reference, not a "
            "genome-scale scan simulation (P5)."
        ),
        (
            "ADR-0005 D3 freezes the Stage-1 threshold, the locus knobs and the Stage-2 "
            "operating point at the §13.1 phase gate (P5-01). Nothing here pins any of them; "
            "the Stage-1 threshold is swept and the sweep is reported beside the gate."
        ),
    ]
    return out


def stage1_threshold_sensitivity(
    points: Sequence[Mapping[str, Any]],
    *,
    base_threshold: float | None,
    base_passes: bool,
) -> dict[str, Any]:
    """Bind each swept Stage-1 threshold to the report that measured it.

    ADR-0005 D3 leaves the Stage-1 threshold unpinned until the §13.1 phase gate, so the
    gate must be shown not to turn on the value this run happened to use. Each point carries
    the threshold **read out of its own report's sources**, not a label supplied on the
    command line: a caption would let ``--sensitivity 0.5=<a 0.9 report>`` publish a table no
    run produced (the PR #131 finding, in its own words). A point whose declared label and
    measured threshold disagree is recorded as a mismatch and makes the block refuse.

    The swept points are **other** reports, so the threshold *this* report was measured at
    would otherwise be the one value the invariance read excluded — the annex could show a
    verdict invariant across 0.7 and 0.9 while the verdict is published at 0.5 (CodeRabbit,
    PR #133 round 2). It is carried here as ``base``, read from the caller's own gate rather
    than duplicated into a second file (a self-referential point would make the invariance
    read a tautology), ``verdict_invariant`` spans base + points, and ``brackets_base``
    states plainly whether the sweep encloses the base value or only sits to one side of it.
    """
    rows = []
    for point in points:
        measured = point["report"].get("sources", {}).get("stage1_threshold")
        rows.append(
            {
                "declared_stage1_threshold": point["declared"],
                "measured_stage1_threshold": measured,
                "label_matches_report": measured == point["declared"],
                "report_sha256": point["sha256"],
                "auprc": point["report"]["gate"]["auprc"],
                "auprc_gain_pp": point["report"]["gate"]["gain_pp"],
                "passes": bool(point["report"]["gate"]["passes"]),
                # The reported (non-gated) matched-recall read at this threshold, carried so
                # the sweep shows BOTH statistics moving — the gated one and the one that
                # motivated ADR-0005 A13 in the first place.
                "matched_recall": point["report"]["gate"]["reported_not_gated"]["matched_recall"],
            }
        )
    rows.sort(key=lambda row: row["measured_stage1_threshold"] or -1.0)
    measured = [
        row["measured_stage1_threshold"]
        for row in rows
        if row["measured_stage1_threshold"] is not None
    ]
    return {
        "points": rows,
        "n_points": len(rows),
        "base": {
            # The threshold the enclosing report was measured at, and that report's own
            # verdict. Source is named so a reader never mistakes it for a swept file.
            "stage1_threshold": base_threshold,
            "passes": bool(base_passes),
            "source": "this report",
        },
        "brackets_base": bool(
            measured
            and base_threshold is not None
            and min(measured) <= base_threshold <= max(measured)
        ),
        "all_labels_match": all(row["label_matches_report"] for row in rows),
        "verdict_invariant": len({*(row["passes"] for row in rows), bool(base_passes)}) <= 1,
    }


def precision_problems(report: Mapping[str, Any]) -> list[str]:
    """Re-derive every load-bearing clause from the report's own numbers.

    Deliberately does **not** trust: the arm label (the exposure counts decide which arm may
    be gated), ``passes`` (re-derived from the two precisions), ``is_science`` (re-derived
    from the completeness clauses), or the prevalence-invariance claim in the module
    docstring (re-derived from the sweep). Returns [] on a clean report.
    """
    problems: list[str] = []

    def want(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    # Every helper below RETURNS rather than raises on a payload it did not write: a
    # traceback out of the validator is indistinguishable from a crashed run (the PR #131
    # finding, restated by CodeRabbit on this PR's own new clauses).
    def number(node: Any, *path: str) -> float | None:
        for key in path:
            if not isinstance(node, Mapping):
                return None
            node = node.get(key)
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            return None
        try:
            return float(node)
        except OverflowError:
            # A Python int has no width limit; `float()` on one past 1e308 raises, which
            # would take the validator down on a value that is simply not a rate or a count
            # (CodeRabbit, PR #133 round 5).
            return None

    for key in ("schema_version", "step", "generated_by", "adr", "prd", "gate", "arms"):
        want(key in report, f"missing top-level key {key!r}")
    if problems:
        return problems
    # …and the coarse SHAPE of every block read below, so a truncated payload comes back as a
    # problem rather than a traceback. "A checker must RETURN a problem on a payload it did
    # not write, never raise: a traceback from the validator is indistinguishable from a
    # crashed run" (the PR #131 finding, restated on this PR's own new clauses).
    for key in ("gate", "arms", "completeness"):
        want(isinstance(report.get(key), Mapping), f"{key!r} is not a block; it cannot be read")
    want(
        isinstance(report.get("benchmark"), Mapping)
        and isinstance(report["benchmark"].get("scope"), Mapping),
        "benchmark.scope is missing or is not a block; the manifest cannot be read",
    )
    if problems:
        return problems

    want(report["step"] == STEP, f"step is {report['step']!r}, expected {STEP!r}")
    arms = report["arms"]
    gated_arm = report.get("gated_arm")
    want(
        isinstance(gated_arm, str) and gated_arm in arms,
        f"gated_arm {gated_arm!r} has no measured arm",
    )
    if not isinstance(gated_arm, str) or gated_arm not in arms:
        return problems
    gated = arms[gated_arm]
    # The nested blocks every clause below indexes. Checked once here so a truncated payload
    # is a problem, not a traceback — the same reason the top-level keys are (PR #131).
    for arm_name, arm in arms.items():
        want(isinstance(arm, Mapping), f"arm {arm_name!r} is not a block; it cannot be read")
        if not isinstance(arm, Mapping):
            continue
        for key in ("population", "prevalence", "per_pool", "matched_recall", "exposure"):
            want(
                isinstance(arm.get(key), Mapping) and bool(arm[key]),
                f"arm {arm_name!r} carries no {key!r} block; it cannot be read",
            )
        want(
            isinstance(arm.get("exposure"), Mapping)
            and "n_positives_seen_by_arm" in arm["exposure"],
            f"arm {arm_name!r} exposure carries no n_positives_seen_by_arm",
        )
        for system in SYSTEMS:
            want(
                isinstance(arm.get("matched_recall"), Mapping)
                and isinstance(arm["matched_recall"].get(system), Mapping),
                f"arm {arm_name!r} carries no matched-recall confusion for {system!r}",
            )
    for key in ("n_items", "n_positives", "n_negatives"):
        want(
            key in report["benchmark"]["scope"],
            f"the benchmark manifest carries no {key!r}",
        )
    for arm_name, arm in arms.items():
        if isinstance(arm.get("population"), Mapping):
            for key in ("n_items", "n_positives", "n_negatives", "n_blocks", "ceiling_recall"):
                want(
                    key in arm["population"],
                    f"arm {arm_name!r} population carries no {key!r}",
                )
    if problems:
        return problems

    # The grade may only rest on an arm that actually held the graded positives out. This is
    # the clause the whole §7 decision turns on, so it is re-derived from the exposure counts
    # rather than read off the arm's name.
    want(
        gated["exposure"]["n_positives_seen_by_arm"] == 0,
        f"the gated arm {gated_arm!r} trained on "
        f"{gated['exposure']['n_positives_seen_by_arm']} of the benchmark's positives; a "
        "gate graded on training data is not a measurement",
    )
    for arm_name, arm in arms.items():
        want(
            bool(arm["gated"]) == (arm_name == gated_arm),
            f"arm {arm_name!r} carries gated={arm['gated']!r} but gated_arm is {gated_arm!r}",
        )

    scope = report["benchmark"]["scope"]
    for arm_name, arm in arms.items():
        population = arm["population"]
        want(
            population["n_items"] == scope["n_items"],
            f"arm {arm_name!r} scored {population['n_items']} of the benchmark's "
            f"{scope['n_items']} items — a truncated run cannot certify the gate",
        )
        counted = (number(population, "n_positives"), number(population, "n_negatives"))
        want(
            None not in counted and sum(counted) == number(population, "n_items"),
            f"arm {arm_name!r}: positives + negatives != items",
        )
        blocks = number(population, "n_blocks")
        want(
            blocks is not None and blocks > 1,
            f"arm {arm_name!r} has {population.get('n_blocks')!r} resampling block(s); a "
            "block-level CI needs at least two (ADR-0005 D5/A1)",
        )
        ceiling = number(population, "ceiling_recall")
        want(
            ceiling is not None and 0.0 <= ceiling <= 1.0,
            f"arm {arm_name!r}: ceiling_recall {population.get('ceiling_recall')!r} is not a "
            "rate",
        )
        for system in SYSTEMS:
            auprc = number(arm.get("auprc"), system)
            want(
                auprc is not None and (math.isnan(auprc) or 0.0 <= auprc <= 1.0),
                f"arm {arm_name!r}: {system} AUPRC "
                f"{(arm.get('auprc') or {}).get(system)!r} is not in [0, 1]",
            )

        # The per-pool breakdown must account for exactly the false positives the gated
        # thresholds produced — otherwise the disclosure about memorised pools is decorative.
        def pool_total(field: str, pools: Mapping[str, Any] = arm["per_pool"]) -> float | None:
            values = [number(pool, field) for pool in pools.values()]
            return None if None in values else sum(values)

        for system in SYSTEMS:
            total = pool_total(f"fp_{system}")
            recorded = number(arm.get("matched_recall", {}).get(system), "fp")
            want(
                total is not None and total == recorded,
                f"arm {arm_name!r}: per-pool {system} false positives sum to {total}, but "
                f"the matched-recall confusion records {recorded!r}",
            )
        want(
            pool_total("n") is not None and pool_total("n") == number(population, "n_negatives"),
            f"arm {arm_name!r}: the per-pool negative counts do not sum to n_negatives",
        )

    # The verdict, re-derived from the GATED statistic (ADR-0005 A13: AUPRC at the D7 pinned
    # prevalence). Deliberately not from the matched-recall precisions, which are reported.
    key = gated["gated_prevalence_key"]
    # A checker must RETURN a problem on a payload it did not write, never raise: a traceback
    # from the validator is indistinguishable from a crashed run (the PR #131 finding).
    if key not in gated.get("prevalence", {}):
        problems.append(
            f"the gated arm names prevalence point {key!r} but carries no such point; the "
            "gated statistic has nowhere to be read from"
        )
        return problems
    point = gated["prevalence"][key]
    if "auprc" not in point:
        problems.append(f"prevalence point {key!r} carries no AUPRC — the gated statistic")
        return problems
    auprc = point["auprc"]
    two, one = number(auprc, "two_stage"), number(auprc, "stage1_only")
    derived = bool(
        two is not None
        and one is not None
        and not math.isnan(two)
        and not math.isnan(one)
        and two > one
    )
    want(
        bool(gated["passes"]) == derived,
        f"the gated arm reports passes={gated['passes']!r}, but its own AUPRCs "
        f"({auprc['two_stage']!r} vs {auprc['stage1_only']!r}) re-derive to {derived!r}",
    )
    want(
        report["gate"].get("statistic") == "auprc_at_pinned_prevalence",
        "the gate does not name the ADR-0005 A13 statistic; a report graded on something "
        "else must not be published under this schema",
    )
    want(
        report["gate"].get("auprc") == auprc,
        "gate.auprc does not match the gated arm's AUPRC at the pinned prevalence",
    )
    # The matched-recall comparison must still be present and still be marked non-gated:
    # dropping it would hide the read that fails, and promoting it would re-introduce the
    # dependence on operating points D3 does not freeze until P5-01.
    reported_block = report["gate"].get("reported_not_gated")
    reported = reported_block.get("matched_recall") if isinstance(reported_block, Mapping) else None
    want(
        isinstance(reported, Mapping) and "gain_pp" in reported,
        "the gate does not carry the matched-recall comparison as a reported, non-gated read",
    )
    want(
        bool(report["gate"].get("passes")) == bool(gated.get("passes")),
        "gate.passes disagrees with the gated arm's own verdict",
    )

    # ── Every published MAGNITUDE re-derived, not only the boolean verdict ────────────
    # The clauses above pin the AUPRC pair and re-derive `passes` from it. Nothing re-derived
    # the numbers a reader actually quotes — the headline gain, its interval, the block count,
    # the sweep's per-point gains — so each of them could be rewritten in the artifact and the
    # clause set would return [] ([[gate-clauses-need-re-derivation]]). A magnitude that no
    # clause recomputes is a fabricated value the report presents as a measured one.
    def close(observed: Any, derived: float | None) -> bool:
        # Through `number`, which is the one place the int-with-no-float case is handled.
        value = number({"v": observed}, "v")
        if derived is None or value is None:
            return False
        return math.isclose(value, derived, rel_tol=1e-9, abs_tol=1e-9)

    def gain_of(node: Any) -> float | None:
        two = number(node, "auprc", "two_stage")
        one = number(node, "auprc", "stage1_only")
        return None if two is None or one is None else (two - one) * 100.0

    gate = report["gate"] if isinstance(report.get("gate"), Mapping) else {}
    gate_ci = gate.get("gain_ci") if isinstance(gate.get("gain_ci"), Mapping) else {}
    population = gated.get("population") if isinstance(gated.get("population"), Mapping) else {}

    want(
        close(gate.get("gain_pp"), gain_of(gate)),
        f"gate.gain_pp {gate.get('gain_pp')!r} is not the difference of the AUPRCs it "
        f"is published beside ({gain_of(gate)!r})",
    )
    want(
        bool(gate_ci) and gate_ci == gated.get("auprc_gain_ci"),
        "gate.gain_ci is not the gated arm's own AUPRC-gain interval; the published interval "
        "must be the one the block bootstrap produced for the statistic being graded",
    )
    want(
        gate_ci.get("n_blocks") == population.get("n_blocks"),
        f"gate.gain_ci resamples {gate_ci.get('n_blocks')!r} blocks but the "
        f"gated arm scored {population.get('n_blocks')!r}",
    )
    want(
        gated["gated_prevalence_key"] == f"{gate.get('decoy_prevalence')}:1",
        f"the gate names prevalence {gate.get('decoy_prevalence')!r} but the gated arm "
        f"reads its statistic from {gated['gated_prevalence_key']!r}",
    )
    # `--n-boot` is a cost knob and no clause read it. At B = 1 the "95 % interval" collapses
    # to one resample and need not contain its own point estimate, and the report still
    # certified ([[cost-knobs-can-certify]]). Both published intervals are checked: the gated
    # AUPRC one and the matched-recall one A13 keeps beside it.
    for label, interval in (
        ("gate.gain_ci", gate_ci),
        ("the reported matched-recall gain_ci", gated.get("gain_ci")),
    ):
        lower = number(interval, "lower")
        centre = number(interval, "point")
        upper = number(interval, "upper")
        want(
            None not in (lower, centre, upper) and lower <= centre <= upper,
            f"{label} = [{lower!r}, {upper!r}] does not contain its own point estimate "
            f"{centre!r}",
        )
        level = number(interval, "ci_level")
        budget = number(interval, "n_boot")
        tail = None if level is None else (1.0 - level) / 2.0
        want(
            tail is not None and budget is not None and tail > 0.0 and budget * tail >= 1.0,
            f"{label}: {budget!r} resamples cannot resolve a {level!r} interval's tails — the "
            "percentile would be an endpoint of the resample set, not a quantile of it",
        )
    want(
        close(gate_ci.get("point"), number(gate, "gain_pp")),
        f"the interval's point estimate {gate_ci.get('point')!r} is not the "
        f"gain {gate.get('gain_pp')!r} it is published beside; the CI would then be "
        "resampled at a different statistic than the one graded",
    )
    # …and the reweighting the gated point was computed at is re-derived, not taken on trust.
    n_pos = number(population, "n_positives")
    n_neg = number(population, "n_negatives")
    prevalence = number(gate, "decoy_prevalence")
    derived_lambda = (
        None
        if None in (n_pos, n_neg, prevalence) or n_neg <= 0 or n_pos <= 0 or prevalence <= 0
        else prevalence_lambda(int(n_pos), int(n_neg), prevalence)
    )
    want(
        close(point.get("lambda"), derived_lambda),
        f"the gated prevalence point records lambda {point.get('lambda')!r}, which is not "
        f"{gate.get('decoy_prevalence')!r} decoys per positive over this benchmark's "
        "own composition",
    )
    # `reported` was already checked to be a mapping carrying gain_pp, above.
    if isinstance(reported, Mapping):
        want(
            reported.get("gain_pp") == gated["gain_pp"]
            and reported.get("gain_ci") == gated["gain_ci"],
            "the reported matched-recall read is not the gated arm's own; a second copy of "
            "the failing number could drift from the one that was measured",
        )
    want(
        close(gated.get("auprc_gain_pp"), gain_of(point)),
        f"the gated arm's auprc_gain_pp {gated.get('auprc_gain_pp')!r} is not the gain at the "
        f"prevalence point it is graded on ({gain_of(point)!r})",
    )
    for arm_name, arm in arms.items():
        for name, node in (arm.get("prevalence") or {}).items():
            if not isinstance(node, Mapping) or "auprc" not in node or "auprc_gain_pp" not in node:
                continue  # a separate clause above already refuses the omission
            want(
                close(node["auprc_gain_pp"], gain_of(node)),
                f"arm {arm_name!r} prevalence point {name!r}: auprc_gain_pp "
                f"{node['auprc_gain_pp']!r} is not the difference of its own AUPRCs "
                f"({gain_of(node)!r})",
            )

    # ── The graded population re-derived against the benchmark manifest ──────────────
    # `n_items` alone let a relabelled corpus certify while the report published the
    # manifest's composition ([[gate-must-bind-to-upstream-evidence]]).
    # A selection RULE is not a measurement of what the drawn windows contain. The host pool
    # published "clean for BOTH Stage-1 checkpoints" for a benchmark in which 115 of 692
    # negatives carry host DNA the production arm trained on (PR #133 round 2).
    host_pool = scope.get("host_pool", {})
    overlap = host_pool.get("overlap_with_training_folds")
    want(
        isinstance(overlap, Mapping) and isinstance(overlap.get("by_arm"), Mapping),
        "the benchmark's host pool carries no measured overlap with the Stage-1 training "
        "folds; a selection rule is not a measurement of what the drawn windows contain",
    )
    if isinstance(overlap, Mapping) and isinstance(overlap.get("by_arm"), Mapping):
        by_arm = overlap["by_arm"]
        want(
            set(by_arm) == set(arms),
            f"the host-overlap measurement covers {sorted(by_arm)!r}, not the arms this "
            f"report grades ({sorted(arms)!r})",
        )
        counts = [
            int(value)
            for value in by_arm.values()
            if isinstance(value, int) and not isinstance(value, bool)
        ] or [0]
        union = overlap.get("n_seen_by_any_arm")
        want(
            isinstance(union, bool) is False
            and isinstance(union, int)
            and max(counts) <= union <= sum(counts),
            f"the host-overlap union {union!r} is not between the largest single arm "
            f"({max(counts)}) and the sum of the arms ({sum(counts)}); it cannot be the "
            "union it names",
        )
        want(
            overlap.get("n_negatives") == scope["n_negatives"],
            f"the host-overlap measurement was taken over {overlap.get('n_negatives')!r} "
            f"negatives, not this benchmark's {scope['n_negatives']!r}",
        )

    by_pool = scope.get("negatives_by_pool") or {}
    pool_total = sum(value for value in by_pool.values() if isinstance(value, (int, float)))
    want(
        pool_total == scope.get("n_negatives"),
        f"the benchmark manifest's per-pool negatives sum to {pool_total}, not "
        f"its own n_negatives {scope.get('n_negatives')!r}",
    )
    want(
        scope.get("n_blocks") == population.get("n_blocks"),
        f"the manifest records {scope.get('n_blocks')!r} blocks but the gated arm resampled "
        f"{population.get('n_blocks')!r}",
    )
    for arm_name, arm in arms.items():
        arm_population = arm.get("population") if isinstance(arm.get("population"), Mapping) else {}
        arm_pools = arm.get("per_pool") if isinstance(arm.get("per_pool"), Mapping) else {}
        for field in ("n_positives", "n_negatives"):
            want(
                arm_population.get(field) == scope.get(field),
                f"arm {arm_name!r} graded {arm_population.get(field)!r} {field[2:]} but the "
                f"benchmark manifest declares {scope.get(field)!r}",
            )
        for pool, count in by_pool.items():
            scored = arm_pools.get(pool)
            scored = scored.get("n") if isinstance(scored, Mapping) else None
            want(
                scored == count,
                f"arm {arm_name!r} scored {scored!r} items from pool {pool!r}; the manifest "
                f"declares {count!r}",
            )

    # Prevalence-invariance, re-derived from the sweep rather than asserted in prose.
    verdicts = {
        name: bool(point["two_stage_beats_stage1_only"])
        for name, point in gated["prevalence"].items()
    }
    want(
        len(set(verdicts.values())) == 1,
        "the REPORTED matched-recall verdict is not invariant across the decoy-prevalence "
        f"sweep ({verdicts!r}); precision at matched recall is provably lambda-invariant, so "
        "this means the sweep and the selection disagree",
    )
    # AUPRC is NOT provably lambda-invariant (two PR curves can cross), so its sweep is
    # checked for presence and reported, never asserted invariant.
    for name, node in gated["prevalence"].items():
        want(
            "auprc" in node and "auprc_gain_pp" in node,
            f"prevalence point {name!r} carries no AUPRC; the gated statistic must be "
            "readable at every swept prevalence, not only at the pinned one",
        )
    want(
        len(gated["prevalence"]) >= len(PREVALENCE_SWEEP) + 1,
        "the prevalence sweep is shorter than ADR-0005 D7's 10:1 → 10⁴:1 plus the observed "
        "composition; an invariance clause over one point is vacuous",
    )

    sensitivity = report.get("stage1_threshold_sensitivity")
    if sensitivity is not None and not isinstance(sensitivity, Mapping):
        problems.append(
            "stage1_threshold_sensitivity is present but is not a block; it cannot be read"
        )
    elif sensitivity is not None:
        rows = [row for row in (sensitivity.get("points") or []) if isinstance(row, Mapping)]
        base_node = sensitivity.get("base") if isinstance(sensitivity.get("base"), Mapping) else {}
        want(
            number(sensitivity, "n_points") is not None and sensitivity["n_points"] >= 2,
            "the Stage-1 threshold sensitivity has fewer than two points; an invariance "
            "read over one point is vacuous",
        )
        want(
            bool(sensitivity.get("all_labels_match")),
            "a Stage-1 threshold sensitivity point's label disagrees with the threshold its "
            "own report records — the table would be a caption, not a measurement",
        )
        want(
            bool(sensitivity.get("verdict_invariant")),
            "the gate's verdict changes across the Stage-1 threshold sweep; ADR-0005 D3 "
            "leaves that value unpinned until P5-01, so the gate would rest on it",
        )
        # The swept points are other reports. Unless the annex also carries the threshold
        # THIS report was measured at, `verdict_invariant` can be true over a set that
        # excludes the only threshold the verdict is actually published at.
        sources = report.get("sources") if isinstance(report.get("sources"), Mapping) else {}
        want(
            base_node.get("stage1_threshold") == sources.get("stage1_threshold"),
            "the Stage-1 threshold sensitivity does not carry the threshold this report was "
            f"measured at ({sources.get('stage1_threshold')!r}), so its "
            "invariance read need not cover the point the gate is published at",
        )
        want(
            bool(base_node.get("passes")) == bool(gate.get("passes")),
            "the Stage-1 threshold sensitivity's base verdict disagrees with the gate it is "
            "an annex to",
        )
        # …and the three summary fields above are re-derived from the rows they summarise.
        # Read as written they are self-certifying: the same call that built the table also
        # wrote the verdicts about it ([[gate-clauses-need-re-derivation]]).
        want(
            sensitivity.get("n_points") == len(rows),
            f"the sensitivity annex claims {sensitivity.get('n_points')!r} points and carries "
            f"{len(rows)}",
        )
        want(
            all(
                bool(row.get("label_matches_report"))
                == (row.get("declared_stage1_threshold") == row.get("measured_stage1_threshold"))
                for row in rows
            ),
            "a sensitivity point's label_matches_report is not the comparison it names",
        )
        want(
            bool(sensitivity.get("all_labels_match"))
            == all(bool(row.get("label_matches_report")) for row in rows),
            "the annex's all_labels_match is not the conjunction of its own rows",
        )
        want(
            bool(sensitivity.get("verdict_invariant"))
            == (
                len({*(bool(row.get("passes")) for row in rows), bool(base_node.get("passes"))})
                <= 1
            ),
            "the annex's verdict_invariant is not re-derivable from its rows and its base",
        )
        swept = [
            row.get("measured_stage1_threshold")
            for row in rows
            if isinstance(row.get("measured_stage1_threshold"), (int, float))
        ]
        # Through `number`, not straight off the block: a non-numeric threshold inside an
        # otherwise well-formed `base` made the comparison below raise, which the
        # block-replacement sweep could not reach (CodeRabbit, PR #133 round 4).
        base_tau = number(base_node, "stage1_threshold")
        want(
            bool(sensitivity.get("brackets_base"))
            == bool(swept and base_tau is not None and min(swept) <= base_tau <= max(swept)),
            "the annex's brackets_base is not the comparison it names",
        )

    completeness = report.get("completeness", {})
    want(bool(completeness), "the report carries no completeness clause set")
    # Only the conjunction, not "is_science must be truthy". The presence of the clause set
    # is already checked above, and demanding truthiness here would refuse a report that
    # correctly derived is_science = False from a FALSE clause — i.e. fire on exactly the
    # honest incomplete run the field exists to mark (CodeRabbit, PR #133).
    want(
        isinstance(report.get("is_science"), bool),
        f"is_science is {report.get('is_science')!r}, not a boolean; a truthy stand-in would "
        "satisfy the conjunction clause below without being the conjunction",
    )
    want(
        bool(report.get("is_science")) == all(bool(v) for v in completeness.values()),
        "is_science is not the conjunction of the completeness clauses",
    )
    # A gate artifact with no disclosures is not a publishable one: the arm choice, the
    # memorised pools, the host overlap and the power floor are all carried there and nothing
    # else in the report states them.
    disclosures_block = report.get("disclosures")
    want(
        isinstance(disclosures_block, list)
        and bool(disclosures_block)
        and all(isinstance(line, str) and line.strip() for line in disclosures_block),
        "the report carries no disclosures; the arm choice, the memorised pools and the host "
        "overlap are stated nowhere else",
    )
    want(
        bool(report.get("gated")) == bool(report.get("is_science")),
        "gated and is_science disagree",
    )

    # A committed public artifact never carries this machine's layout. Not an allowlist of
    # two prefixes — any absolute path at all (the P3-15′-k review finding).
    for path in _absolute_paths(report):
        problems.append(f"the report carries an absolute path: {path!r}")
    return problems


def _absolute_paths(node: Any, seen: list[str] | None = None) -> list[str]:
    out = [] if seen is None else seen
    if isinstance(node, str):
        if node.startswith("/") and "/" in node[1:]:
            out.append(node)
    elif isinstance(node, Mapping):
        for key, value in node.items():
            _absolute_paths(key, out)
            _absolute_paths(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _absolute_paths(value, out)
    return out


# ══════════════════════════════════════════════════════════════════════════════════════
# I/O + CLI
# ══════════════════════════════════════════════════════════════════════════════════════
def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_table(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if "rows" not in payload:
        raise PrecisionError(f"{path}: not a candidate table (no 'rows' key)")
    return list(payload["rows"])


def _cmd_fold(args: argparse.Namespace) -> int:
    """LEG 1 (torch-free, but reads GPU-derived tables): candidate tables → the item table."""
    folded = fold_arms(
        read_json(args.benchmark),
        {"twin": read_table(args.twin_table), "production": read_table(args.production_table)},
    )
    folded["source"] = {
        "benchmark": repo_relative(args.benchmark),
        "twin_table": repo_relative(args.twin_table),
        "production_table": repo_relative(args.production_table),
        "twin_checkpoint": args.twin_checkpoint,
        "production_checkpoint": args.production_checkpoint,
        "stage1_threshold": args.stage1_threshold,
    }
    write_json(args.out, folded)
    print(
        f"wrote {args.out}: {len(folded['items'])} items × {len(folded['arms'])} arms "
        f"({', '.join(folded['arms'])})"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """LEG 2 (torch-free, CI-runnable): the item table → the graded report."""
    folded = read_json(args.items)
    sources = {**folded.get("source", {}), "items": repo_relative(args.items)}
    report = precision_gain(
        folded,
        stage2_operating_point=args.stage2_operating_point,
        decoy_prevalence=args.decoy_prevalence,
        prevalence_sweep=PREVALENCE_SWEEP,
        recall_grid=RECALL_GRID,
        n_boot=args.n_boot,
        seed=args.seed,
        sources=sources,
    )
    if args.sensitivity:
        points = []
        for spec in args.sensitivity:
            declared, _, path = spec.partition("=")
            if not path:
                raise PrecisionError(
                    f"--sensitivity expects '<threshold>=<report.json>', got {spec!r}"
                )
            raw = Path(path).read_bytes()
            points.append(
                {
                    "declared": float(declared),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "report": json.loads(raw.decode("utf-8")),
                }
            )
        report["stage1_threshold_sensitivity"] = json_safe(
            stage1_threshold_sensitivity(
                points,
                base_threshold=sources.get("stage1_threshold"),
                base_passes=bool(report["gate"]["passes"]),
            )
        )
    report["provenance"] = build_provenance(
        rule="workflow/rules/integration.smk :: precision_comparison",
        script=GENERATED_BY,
        inputs=[args.items],
        env_lock=ENV_LOCK,
        adr="ADR-0005",
        seed=args.seed,
        # Declared NAMES, never hashed paths: `build_provenance` hashes everything in
        # `outputs`, and this report does not exist when its own provenance is built
        # ([[build-provenance-hashes-its-outputs]], the P3-10 failure).
        extra={"declared_outputs": [repo_relative(args.out)]},
    )
    problems = precision_problems(report)
    report["problems"] = problems
    out = Path(args.out)
    # `is_science` is DERIVED from the completeness clauses, but until now nothing consumed
    # it: a truncated or half-measured run wrote the graded artifact to the phase-exit path
    # and exited 0, marked ``is_science: false`` in a field no gate read
    # ([[cost-knobs-can-certify]]). The report is still written — to ``.invalid.json``, where
    # a reader cannot mistake it for the accepted one.
    incomplete = sorted(name for name, ok in report.get("completeness", {}).items() if not ok)
    if problems or incomplete:
        # Divert AND remove the canonical path, so its absence is the signal — a stale
        # accepted report sitting beside a fresh .invalid.json reads as current (P3-10 r1).
        invalid = out.with_suffix(".invalid.json")
        write_json(invalid, report)
        out.unlink(missing_ok=True)
        if problems:
            print(f"REFUSED: {len(problems)} problem(s); wrote {invalid}")
            for problem in problems:
                print(f"  - {problem}")
        if incomplete:
            print(f"REFUSED: is_science is false; wrote {invalid}")
            for name in incomplete:
                print(f"  - completeness clause FALSE: {name}")
        return 3
    write_json(out, report)
    gate = report["gate"]
    matched = gate["reported_not_gated"]["matched_recall"]
    print(
        f"wrote {out}: gate {'PASSES' if gate['passes'] else 'FAILS'} on arm {gate['arm']} — "
        f"AUPRC@{gate['decoy_prevalence']}:1 two-stage {gate['auprc']['two_stage']:.6f} vs "
        f"Stage-1-only {gate['auprc']['stage1_only']:.6f} "
        f"(gain {gate['gain_pp']:+.4f} pp, 95% CI "
        f"[{gate['gain_ci']['lower']:.4f}, {gate['gain_ci']['upper']:.4f}]); "
        f"reported-not-gated matched recall {matched['target_recall']:.6f}: "
        f"{matched['precision']['two_stage']:.6f} vs "
        f"{matched['precision']['stage1_only']:.6f} ({matched['gain_pp']:+.3f} pp)"
    )
    return 0 if gate["passes"] else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fold = sub.add_parser("fold", help="candidate tables -> the committed per-item score table")
    fold.add_argument("--benchmark", required=True)
    fold.add_argument("--twin-table", required=True)
    fold.add_argument("--production-table", required=True)
    fold.add_argument("--out", required=True)
    fold.add_argument("--twin-checkpoint", required=True)
    fold.add_argument("--production-checkpoint", required=True)
    fold.add_argument("--stage1-threshold", type=float, required=True)
    fold.set_defaults(func=_cmd_fold)

    report = sub.add_parser("report", help="the item score table -> the graded P3-16 report")
    report.add_argument("--items", required=True)
    report.add_argument("--out", required=True)
    # No default: ADR-0005 D3 freezes the Stage-2 operating point at the §13.1 phase gate,
    # so this module may not supply one by omission (the `run_two_stage` discipline).
    report.add_argument("--stage2-operating-point", type=float, required=True)
    report.add_argument("--decoy-prevalence", type=int, default=DECOY_PREVALENCE)
    report.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    report.add_argument("--seed", type=int, default=42)
    report.add_argument(
        "--sensitivity",
        action="append",
        default=[],
        metavar="THRESHOLD=REPORT.json",
        help=(
            "a Stage-1-threshold sensitivity point: the report produced at that "
            "threshold. Each point is bound to its own report's recorded threshold "
            "and digest, so a mislabelled pair is refused rather than published."
        ),
    )
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
