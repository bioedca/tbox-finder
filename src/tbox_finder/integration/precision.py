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

The v2 re-grade — schema 1.2 (ADR-0005 A15, P3-16′-remint)
---------------------------------------------------------
Every number quoted above describes **v1**, whose grade A15 withdraws: its pass was carried
by decoys whose *hosts* hold RF00230-detectable sequence. v1's bytes stay unchanged as the
record of that grade, and the 1.1 clause set keeps passing on them. A benchmark manifest that
declares A15 Pin 4's gated subset (manifest schema ≥ 1.1, per-item ``gated``) is graded under
**schema 1.2**:

* **The leader is reported beside the gate.** ``leader_decoy`` items are minted and scored,
  then filtered out *at the* :func:`precision_gain` *call site*, before :func:`arm_metrics`.
  :func:`arm_metrics` keeps its signature and semantics, because P3-17a's committed
  precision block re-derives through it bit-for-bit. Their scores are published in a
  ``reported_not_gated`` block, and the four completeness clauses compare against the
  manifest's ``*_gated`` counts.
* **No power is recorded as such (Pin 5).** If no gated negative carries a Stage-1
  candidate in the gated arm, the report says ``verdict_class: "underpowered"``. The verdict
  itself stays exactly what A13 computes: with no negative interleaved, both systems' PR
  curves are the same, and the tie fails A13's strict ``>``.
* **The host-overlap invariant is a conjunction (Pin 6).** It holds when the gated arm's
  measured overlap is 0, the screen refused at least one window on the k-mer rule, every
  must-fire control fired, and the selection names the screen. A bare ``== 0`` is not
  enough. The screen and post-mint reports, the Stage-2 identity (bound to the GATE-2
  temperature source) and every interim input's sha256 are recorded and re-hashed
  (:func:`precision_input_problems`).
* **Flank familiarity is measured (Pin 7(b)).** A baseline scores each item by the fraction
  of its 32-mers present in the arm's own training DNA. It is graded with the A13 statistic
  and published beside both systems. If it rivals Stage-1, the report says the gate is
  confounded by construction.

PRD §2.3, §6, §12, §18.1; ADR-0005 D3, D5, D7, A7, A11 Pin 4, A12, **A13**, **A15**;
ADR-0004 A6.
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

from tbox_finder import refs
from tbox_finder.eval.gate4 import json_safe
from tbox_finder.eval.resample import DEFAULT_N_BOOT, block_bootstrap
from tbox_finder.integration.two_stage import (
    read_temperature,
    refuse_v1_output,
    repo_relative,
    write_json,
)
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
#
# 1.2 (ADR-0005 A15 Pin 6) is the v2 re-grade: the gated subset, the power clause, the
# familiarity baseline, the Stage-2 identity, and the screen and post-mint bindings. Every
# 1.2 clause is gated on `schema_version >= 1.2`, so the committed 1.1 bytes (the withdrawn
# v1 grade) keep validating under the clause set they were published under.
SCHEMA_VERSION = "1.2"
#: The pre-A15 schema. A manifest that declares no gated subset is still graded under it,
#: so v1 re-derives exactly. ``test_the_v1_path_reproduces_the_committed_v1_report`` holds
#: it to the committed bytes.
V1_SCHEMA_VERSION = "1.1"
#: The report schema at which the A15 clause set applies.
GATED_REPORT_SCHEMA: tuple[int, ...] = (1, 2)
#: The benchmark-manifest schema at which the minter declares A15 Pin 4's gated subset
#: (``scripts/mint_p3_16_benchmark.py``: per-item ``gated`` and the ``*_gated`` scope
#: counts). The REPORT schema follows the MANIFEST. A manifest that declares the subset is
#: graded under 1.2, and one that predates it under 1.1.
GATED_MANIFEST_SCHEMA: tuple[int, ...] = (1, 1)
STEP = "P3-16"
GENERATED_BY = "src/tbox_finder/integration/precision.py"
ADR = (
    "ADR-0005 D3 (locus rule + Stage-2 operating point), D5 (block-level CI), "
    "D7 (benchmark decoy prevalence + sweep), A7 (which decoy pools were trained on), "
    "A11 Pin 4 (Stage-1 read on the uncalibrated posterior), A12 (the P2 checkpoint is "
    "canonical); ADR-0004 A6 (eval twin / two-run split)"
)
ADR_V2 = (
    f"{ADR}; ADR-0005 A15 (screened re-mint, leader_decoy reported beside the gate, power "
    "clause, familiarity baseline, Stage-2 identity)"
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

# ── ADR-0005 A15 (schema 1.2) ─────────────────────────────────────────────────────────
#: The one pool A15 Pin 4 reports beside the gate. All of its §9.1 records are T-box-derived
#: (A7 Pin 5), so a benchmark item drawn from it is a label contradiction, not a negative.
NON_GATED_POOL = "leader_decoy"
#: Each full-benchmark scope count → the manifest's gated-subset count that the 1.2
#: completeness clauses compare against (A15 Pin 4).
GATED_SCOPE_FIELDS: dict[str, str] = {
    "n_items": "n_items_gated",
    "n_negatives": "n_negatives_gated",
    "negatives_by_pool": "negatives_by_pool_gated",
    "n_blocks": "n_blocks_gated",
}
#: A15 Pin 5's sentence, verbatim. It is written when the gated arm has no gated negative
#: with a Stage-1 candidate.
UNDERPOWERED_STATEMENT = (
    "no gated negative reached Stage-2; the benchmark has no power to test Stage-2"
)
#: A15 Pin 7(b): what the report must say when the familiarity baseline rivals Stage-1.
CONFOUNDED_STATEMENT = "the gate is confounded by construction"
#: A15 Pin 2's content threshold. A screen run at any other value is not the signed screen.
SCREEN_THRESHOLD_BITS = 10.0
#: A screened mint's ``scope.host_pool.selection`` must name the screen it applied.
SCREEN_SELECTION_TAG = "ADR-0005 A15"
#: Fields that identify the ONE Stage-2 checkpoint both arms were re-ranked by. They must be
#: present and identical across arms (A15 Pin 6). ``--checkpoint-root`` / ``--arm`` are
#: pinned because P3-17's RNA-FM arm shares the name ``aux1.0_lr1e-4``, so the backbone and
#: the adapter/heads digests are what tell the two apart.
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
    "env_lock",
)
#: The fields the Stage-2 identity must share with the GATE-2 report the temperature is read
#: from, so the logits and the ``T`` applied to them come from one checkpoint.
TEMPERATURE_BINDING_FIELDS: tuple[str, ...] = ("arm", "adapter_sha256", "heads_sha256")
#: The ``sources`` keys whose files a 1.2 report must record in ``interim_sha256``.
V2_SOURCE_FILES: tuple[str, ...] = (
    "benchmark",
    "twin_table",
    "production_table",
    "twin_stage1",
    "production_stage1",
    "twin_stage2",
    "production_stage2",
    "twin_payloads",
    "production_payloads",
    "host_screen",
    "postmint",
    "familiarity",
)
#: ``--rule-knobs``: the ADR-0005 D3 locus knobs the candidate tables were built under.
#: They are recorded, never defaulted, and none is a D3 freeze.
RULE_KNOB_NAMES: tuple[str, ...] = (
    "threshold",
    "min_span",
    "gap_merge",
    "min_distinct_elements",
    "flank",
    "min_order_margin",
)
#: Committed (git) prefixes. A file under one must be present wherever the report is
#: checked, CI included. Every other recorded input is interim and is re-hashed only when
#: present (the ``cm_confirmer.validate_report(require_all_inputs=...)`` split).
COMMITTED_PREFIXES: tuple[str, ...] = ("reports/",)
#: The repository the recorded repo-relative paths resolve against. It is the module's own
#: checkout, the root ``two_stage.repo_relative`` records against, and it is read at call
#: time so a test can point it at a staged tree.
REPO_ROOT = Path(__file__).resolve().parents[3]


class PrecisionError(ValueError):
    """A malformed input or an impossible composition — never a silently degraded number."""


def schema_tuple(value: Any) -> tuple[int, ...] | None:
    """``"1.2"`` → ``(1, 2)``; ``None`` for anything that is not a dotted run of integers.

    It returns rather than raises, because the validator reads it off payloads it did not
    write. A malformed version therefore selects no schema at all, and every clause that
    needs one refuses.
    """
    if not isinstance(value, str) or not value:
        return None
    parts = value.split(".")
    if not all(part.isascii() and part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def manifest_declares_gated_subset(benchmark: Mapping[str, Any]) -> bool:
    """Whether a benchmark MANIFEST declares A15 Pin 4's gated subset (schema ≥ 1.1)."""
    version = schema_tuple(benchmark.get("schema_version"))
    return version is not None and version >= GATED_MANIFEST_SCHEMA


def folded_declares_gated_subset(folded: Mapping[str, Any]) -> bool:
    """Whether a folded score table was built from a manifest that declares the gated subset.

    Read off ``benchmark.manifest_schema_version``, which :func:`fold_arms` copies from the
    manifest, and **not** off the folded table's own label. The report schema follows the
    evidence it was built from.
    """
    block = folded.get("benchmark")
    version = (
        schema_tuple(block.get("manifest_schema_version")) if isinstance(block, Mapping) else None
    )
    return version is not None and version >= GATED_MANIFEST_SCHEMA


def item_is_gated(item: Mapping[str, Any]) -> bool:
    """A15 Pin 4's per-item flag. An item from a manifest that predates it defaults to True."""
    return bool(item.get("gated", True))


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

    **The gated subset (ADR-0005 A15 Pin 4).** A manifest that declares it (schema ≥ 1.1)
    must give every item a strict-bool ``gated``, which is carried onto the folded item.
    The folded table is then labelled 1.2 and records ``benchmark.manifest_schema_version``.
    An older manifest folds exactly as before: schema 1.1, no ``gated`` key, every item
    gated by default. That keeps a v1 re-fold byte-identical to the committed items. An
    older manifest that nevertheless carries ``gated`` is refused. Its schema does not
    define the flag, so grading it under either clause set would be a guess.
    """
    items = benchmark["items"]
    declares = manifest_declares_gated_subset(benchmark)
    gated_flags: dict[str, bool] = {}
    for item in items:
        if declares:
            flag = item.get("gated")
            if not isinstance(flag, bool):
                raise PrecisionError(
                    f"item {item.get('contig_id')!r} carries gated={flag!r}; a manifest of "
                    f"schema {benchmark.get('schema_version')!r} declares ADR-0005 A15 Pin 4's "
                    "gated subset, so every item must carry a strict bool"
                )
            gated_flags[str(item["contig_id"])] = flag
        elif "gated" in item:
            raise PrecisionError(
                f"item {item.get('contig_id')!r} carries a gated flag, but the manifest's "
                f"schema {benchmark.get('schema_version')!r} predates ADR-0005 A15's gated "
                "subset; it cannot be graded under either clause set without a guess"
            )
    per_arm = {arm: item_scores(items, rows) for arm, rows in sorted(tables.items())}
    order = [item["contig_id"] for item in items]
    by_arm_id = {arm: {row["contig_id"]: row for row in scored} for arm, scored in per_arm.items()}
    folded = []
    for contig_id in sorted(order):
        first = next(iter(by_arm_id))
        base = by_arm_id[first][contig_id]
        entry = {
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
        if declares:
            entry["gated"] = gated_flags[contig_id]
        folded.append(entry)
    out = {
        "schema_version": SCHEMA_VERSION if declares else V1_SCHEMA_VERSION,
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
    if declares:
        out["benchmark"]["manifest_schema_version"] = benchmark["schema_version"]
    return out


# ══════════════════════════════════════════════════════════════════════════════════════
# The v2 fold inputs (ADR-0005 A15 Pin 6): each recorded with its sha256, none re-typed
# ══════════════════════════════════════════════════════════════════════════════════════
def work_list_coverage(
    payload: Mapping[str, Any], work_list: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """How many of the graded work list's payloads the Stage-2 file holds a logit for.

    The Stage-2 pass need not score the graded threshold's work list itself. P3-16 scored
    the UNION manifest once and served the whole τ sweep from it (measured on v1: the twin's
    7,446 logits cover all 4,828 τ = 0.5 payloads), so the identity's ``payloads_sha256`` is
    that manifest's digest. The binding that matters is coverage: every payload of the work
    list the candidate tables were built from must have been scored.
    """
    logits = payload.get("logits")
    keys = set(logits) if isinstance(logits, Mapping) else set()
    rows = [str(row.get("row_id")) for row in work_list]
    return {"n_rows": len(rows), "n_scored": sum(1 for row in rows if row in keys)}


def stage2_identity_record(
    payload: Mapping[str, Any],
    *,
    path: str,
    sha256: str,
    work_list: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """One arm's Stage-2 logits file → the identity block the report binds (A15 Pin 6).

    It copies what ``two_stage stage2`` (or ``merge-stage2``) wrote and derives nothing.
    A missing field is recorded as ``None``, and :func:`precision_problems` then refuses it.
    It is never filled from a constant here. ``shard`` is the per-shard marker: ``None`` on
    an unsharded or merged file. A non-null value means the file holds ONE shard's logits,
    which do not cover the payload manifest. ``work_list`` is the graded threshold's payload
    manifest; it adds :func:`work_list_coverage`.
    """
    record: dict[str, Any] = {field: payload.get(field) for field in STAGE2_IDENTITY_FIELDS}
    record.update(
        {
            "payloads_sha256": payload.get("payloads_sha256"),
            "n_payloads": payload.get("n_payloads"),
            "shard": payload.get("shard"),
            "shard_layout": payload.get("shard_layout"),
            "schema_version": payload.get("schema_version"),
            "path": path,
            "sha256": sha256,
            "work_list_coverage": (
                None if work_list is None else work_list_coverage(payload, work_list)
            ),
        }
    )
    return record


def temperature_source_record(
    report: Mapping[str, Any], *, path: str, sha256: str, temperature: float
) -> dict:
    """The GATE-2 report the Stage-2 ``T`` came from, with the identity of the arm it graded.

    ``temperature`` is :func:`~tbox_finder.integration.two_stage.read_temperature`'s read of
    the same file: P3-16 applied the temperature that way, so the recorded value is the one
    applied, not a second parse that could disagree with it.
    """
    scoring = report.get("scoring") if isinstance(report.get("scoring"), Mapping) else {}
    load = scoring.get("load") if isinstance(scoring.get("load"), Mapping) else {}
    return {
        "path": path,
        "sha256": sha256,
        "arm": scoring.get("arm"),
        "checkpoint_dir": load.get("checkpoint_dir"),
        "adapter_sha256": load.get("adapter_sha256"),
        "heads_sha256": load.get("heads_sha256"),
        "revision": load.get("revision"),
        "temperature": float(temperature),
    }


def screen_record(screen: Mapping[str, Any], *, path: str, sha256: str) -> dict:
    """``reports/p3/host_screen.json`` → the counts and verdicts the 1.2 clauses read.

    It copies and never re-derives: the screen report validates itself
    (``host_screen.validate_report``), and this report binds to it by digest. A missing
    field is recorded as ``None`` so a clause refuses it rather than reading an absence as
    a zero.
    """

    def block(name: str) -> Mapping[str, Any]:
        value = screen.get(name)
        return value if isinstance(value, Mapping) else {}

    fold, content, admitted = block("fold_screen"), block("content_screen"), block("admitted")
    controls = screen.get("controls")
    problems = screen.get("problems")
    return {
        "path": path,
        "sha256": sha256,
        "schema_version": screen.get("schema_version"),
        "step": screen.get("step"),
        "n_problems": len(problems) if isinstance(problems, list) else None,
        "controls_all_fired": screen.get("controls_all_fired"),
        "n_controls": len(controls) if isinstance(controls, list) else None,
        "n_controls_fired": (
            sum(1 for c in controls if isinstance(c, Mapping) and c.get("fired") is True)
            if isinstance(controls, list)
            else None
        ),
        "k": block("screened_dna").get("k"),
        "threshold_bits": content.get("threshold_bits"),
        "motif": content.get("motif"),
        "pool_sha256": block("pool").get("sha256"),
        "n_candidates": block("pool").get("n_candidates"),
        "n_refused_kmer": fold.get("n_refused_kmer"),
        "n_refused_coord": fold.get("n_refused_coord"),
        "n_refused_fold": fold.get("n_refused"),
        "n_survivors": fold.get("n_survivors"),
        "n_refused_cm": content.get("n_refused_cm"),
        "n_refused_motif": content.get("n_refused_motif"),
        "n_refused_content": content.get("n_refused"),
        "n_admitted": admitted.get("n"),
        "admitted_sha256": admitted.get("sha256"),
        "v1_composition": block("v1_audit").get("composition"),
    }


def postmint_record(postmint: Mapping[str, Any], *, path: str, sha256: str) -> dict:
    """``reports/p3/benchmark_v2_postmint_screen.json`` → its violation count (A15 Pin 3).

    Carries the **threshold** and its **calibration** beside the count, not just the count.
    ``n_violations == 0`` on its own certifies nothing about the gate: it reads the same at the
    signed per-model thresholds, at a forgotten 10.0, and at an absurd 1000.0. A15 Pin 6 makes
    the post-mint verdict a conjunction for that reason, so the fold has to fold the conjuncts
    forward rather than the headline ([[gate-clauses-need-re-derivation]]).
    """
    problems = postmint.get("problems")
    bench = postmint.get("benchmark")
    calibration = postmint.get("calibration")
    residue = postmint.get("noise_residue")
    return {
        "path": path,
        "sha256": sha256,
        "schema_version": postmint.get("schema_version"),
        "step": postmint.get("step"),
        "n_problems": len(problems) if isinstance(problems, list) else None,
        "n_violations": postmint.get("n_violations"),
        "n_exempt": postmint.get("n_exempt"),
        "threshold_bits": postmint.get("threshold_bits"),
        "search_threshold_bits": postmint.get("search_threshold_bits"),
        "n_noise_residue": (
            postmint.get("n_noise_residue")
            if postmint.get("n_noise_residue") is not None
            else (len(residue) if isinstance(residue, list) else None)
        ),
        "calibration_sha256": (
            calibration.get("sha256") if isinstance(calibration, Mapping) else None
        ),
        "null_max_bits": (
            calibration.get("null_max_bits") if isinstance(calibration, Mapping) else None
        ),
        "benchmark_sha256": bench.get("sha256") if isinstance(bench, Mapping) else None,
    }


def attach_payload_counts(
    folded: dict[str, Any], arm: str, payloads: Sequence[Mapping[str, Any]]
) -> None:
    """Set ``n_payloads`` on every folded item for ``arm``, from that arm's Stage-2 work list.

    This is A15 Pin 5's second power count: an item reached Stage-2 when the payload manifest
    holds at least one payload for its contig. A payload naming a contig the benchmark does
    not hold is refused, for the same reason :func:`item_scores` refuses a stray table row.
    """
    counts: dict[str, int] = {}
    for payload in payloads:
        counts[str(payload["contig_id"])] = counts.get(str(payload["contig_id"]), 0) + 1
    known = {item["contig_id"] for item in folded["items"]}
    stray = sorted(set(counts) - known)
    if stray:
        raise PrecisionError(
            f"the {arm} payload manifest names {len(stray)} contig(s) the benchmark does not "
            f"hold (e.g. {stray[:3]}); the manifest and the benchmark are not from one run"
        )
    for item in folded["items"]:
        if arm not in item["arms"]:
            raise PrecisionError(f"item {item['contig_id']!r} carries no scores for arm {arm!r}")
        item["arms"][arm]["n_payloads"] = counts.get(item["contig_id"], 0)


def attach_familiarity(
    folded: dict[str, Any], familiarity: Mapping[str, Any], *, benchmark_sha256: str
) -> dict:
    """Copy the per-item flank-familiarity scores onto the folded items (A15 Pin 7(b)).

    The familiarity file must name THIS benchmark by digest and score every item for every
    arm with a finite fraction in [0, 1]. Anything else is refused, never filled. Returns
    the metadata block the report records beside the baseline.
    """
    bench = familiarity.get("benchmark")
    recorded = bench.get("sha256") if isinstance(bench, Mapping) else None
    if recorded != benchmark_sha256:
        raise PrecisionError(
            f"the familiarity file scores benchmark {recorded!r}, not this one "
            f"({benchmark_sha256!r}); its scores describe a different item set"
        )
    scores = familiarity.get("items")
    if not isinstance(scores, Mapping):
        raise PrecisionError("the familiarity file carries no per-item 'items' block")
    ids = {item["contig_id"] for item in folded["items"]}
    if set(scores) != ids:
        raise PrecisionError(
            f"the familiarity file scores {len(set(scores) & ids)} of the benchmark's "
            f"{len(ids)} items and {len(set(scores) - ids)} it does not hold"
        )
    for item in folded["items"]:
        entry = scores[item["contig_id"]]
        copied: dict[str, float] = {}
        for arm in folded["arms"]:
            value = entry.get(arm) if isinstance(entry, Mapping) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise PrecisionError(
                    f"item {item['contig_id']!r}: familiarity for arm {arm!r} is {value!r}, not "
                    "a fraction in [0, 1]"
                )
            copied[arm] = float(value)
        item["familiarity"] = copied
    arms_meta = familiarity.get("arms") if isinstance(familiarity.get("arms"), Mapping) else {}
    return {
        "schema_version": familiarity.get("schema_version"),
        "step": familiarity.get("step"),
        "k": familiarity.get("k"),
        "benchmark_sha256": recorded,
        "arms": {arm: arms_meta.get(arm) for arm in sorted(folded["arms"])},
    }


def parse_rule_knobs(text: str) -> dict[str, float | int]:
    """``threshold=0.5,min_span=50,…`` → the recorded D3 locus knobs.

    Exactly :data:`RULE_KNOB_NAMES`, each once. An unknown, duplicated or missing name is
    refused, because a recorded knob set that silently lacks one would let a reader assume
    a default. Integers stay integers, and every other value must be a finite float. A
    per-class threshold mapping (``two_stage --threshold '{…}'``) is not expressible here
    and is refused rather than flattened.
    """
    knobs: dict[str, float | int] = {}
    for part in (piece.strip() for piece in text.split(",")):
        name, sep, raw = part.partition("=")
        name, raw = name.strip(), raw.strip()
        if not sep or not name or not raw:
            raise PrecisionError(f"--rule-knobs entry {part!r} is not '<name>=<value>'")
        if name not in RULE_KNOB_NAMES:
            raise PrecisionError(f"--rule-knobs names unknown knob {name!r}")
        if name in knobs:
            raise PrecisionError(f"--rule-knobs names {name!r} twice")
        if raw.lstrip("-").isascii() and raw.lstrip("-").isdigit():
            knobs[name] = int(raw)
            continue
        try:
            value = float(raw)
        except ValueError as error:
            raise PrecisionError(f"--rule-knobs {name}={raw!r} is not a number") from error
        if not math.isfinite(value):
            raise PrecisionError(f"--rule-knobs {name}={raw!r} is not finite")
        knobs[name] = value
    missing = [name for name in RULE_KNOB_NAMES if name not in knobs]
    if missing:
        raise PrecisionError(f"--rule-knobs is missing {missing}")
    return {name: knobs[name] for name in RULE_KNOB_NAMES}


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

    **Schema 1.2 (ADR-0005 A15)** applies when the table was folded from a manifest that
    declares the gated subset (:func:`folded_declares_gated_subset`). The non-gated items,
    i.e. the ``leader_decoy`` item, are then filtered out HERE, before :func:`arm_metrics`
    sees them (Pin 4). :func:`arm_metrics` is unchanged, so P3-17a's committed precision
    block still re-derives through it bit-for-bit. A table that does not declare the subset
    is graded exactly as before, under 1.1. A table that carries gated-subset content without
    declaring it is refused, so a stripped version label cannot downgrade a v2 benchmark to
    the 1.1 clause set.
    """
    benchmark = folded["benchmark"]
    if GATED_ARM not in folded["arms"]:
        raise PrecisionError(
            f"the folded score table carries no arm {GATED_ARM!r}; the P3 exit gate is graded "
            f"on it (§7 decision 2026-08-13), so a report without it has no verdict to make"
        )
    gated_schema = folded_declares_gated_subset(folded)
    graded = folded
    if gated_schema:
        unflagged = [
            item["contig_id"] for item in folded["items"] if not isinstance(item.get("gated"), bool)
        ]
        if unflagged:
            raise PrecisionError(
                f"{len(unflagged)} item(s) carry no strict-bool gated flag (e.g. {unflagged[:3]}) "
                "in a table folded from a manifest that declares ADR-0005 A15's gated subset"
            )
        # ADR-0005 A15 Pin 4, AT THE CALL SITE: the leader is minted and scored, and reported
        # beside the gate, never inside it. Filtering here rather than inside `arm_metrics`
        # keeps that function's semantics, and so P3-17a's re-derivation, untouched.
        graded = {**folded, "items": [item for item in folded["items"] if item["gated"]]}
    elif any("gated" in item for item in folded["items"]) or any(
        key in benchmark["scope"] for key in GATED_SCOPE_FIELDS.values()
    ):
        raise PrecisionError(
            "the folded table carries ADR-0005 A15 gated-subset content but does not declare "
            "the manifest schema that defines it; it cannot be graded under the 1.1 clause set"
        )
    arms = {
        arm: arm_metrics(
            arm_items(graded, arm),
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
    # The completeness clause compares against the manifest's GATED count under 1.2 (A15 Pin
    # 4): the denominator the gate grades is the gated subset, not every minted item.
    expected_items = scope.get(GATED_SCOPE_FIELDS["n_items"]) if gated_schema else scope["n_items"]
    completeness = {
        "all_benchmark_items_scored": all(
            arm["population"]["n_items"] == expected_items for arm in arms.values()
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
        "schema_version": SCHEMA_VERSION if gated_schema else V1_SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR_V2 if gated_schema else ADR,
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
        "disclosures": [] if gated_schema else disclosures(benchmark, arms),
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
    if gated_schema:
        report.update(
            gated_schema_blocks(folded, arms, decoy_prevalence=decoy_prevalence, scope=scope)
        )
        # A15 Pin 5: the class is RECORDED beside the verdict and never substitutes for it.
        # `gate.passes` above is exactly A13's strict `>`, and nothing below touches it.
        report["gate"]["verdict_class"] = report["power"]["verdict_class"]
        if report["power"]["verdict_class"] == "underpowered":
            report["gate"]["underpowered_statement"] = UNDERPOWERED_STATEMENT
        report["disclosures"] = disclosures_v2(benchmark, arms, report)
    # Derived, never asserted: a truncated or half-measured run must not be able to land at
    # the phase-exit path flagged as real science ([[cost-knobs-can-certify]]).
    report["is_science"] = bool(all(completeness.values()))
    report["gated"] = report["is_science"]
    return json_safe(report)


# ══════════════════════════════════════════════════════════════════════════════════════
# The schema-1.2 blocks (ADR-0005 A15)
# ══════════════════════════════════════════════════════════════════════════════════════
def non_gated_scores(folded: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Both systems' scores on every item reported beside the gate, per arm (A15 Pin 4).

    The leader is minted and scored. Publishing its scores here, rather than dropping them,
    is what makes "reported beside the gate" a measurement and not an omission.
    """
    out: dict[str, list[dict[str, Any]]] = {arm: [] for arm in sorted(folded["arms"])}
    for item in folded["items"]:
        if item_is_gated(item):
            continue
        for arm in sorted(folded["arms"]):
            entry = item["arms"][arm]
            out[arm].append(
                {
                    "contig_id": item["contig_id"],
                    "label": int(item["label"]),
                    "pool": str(item["pool"]),
                    "n_rows": int(entry["n_rows"]),
                    **{system: float(entry[system]) for system in SYSTEMS},
                }
            )
    return out


def power_block(folded: Mapping[str, Any], *, gated_arm: str = GATED_ARM) -> dict[str, Any]:
    """A15 Pin 5's score-blind power counts, per arm, over the GATED negatives.

    The counts are the number of gated negatives with at least one Stage-1 candidate
    (``n_rows > 0``), and with at least one Stage-2 payload (``n_payloads > 0``, from the
    arm's work list). ``verdict_class`` is ``"underpowered"`` iff the GATED arm's first
    count is 0. With no gated negative interleaved among the called positives, both systems
    produce one PR curve: each called-positive tie group has precision exactly 1, the last
    group is shared, and the gain is a tie that A13's strict ``>`` fails. The class records
    WHY the gate fails and never changes the verdict. A ``None`` payload count means the work
    list was never attached, and the 1.2 clause set refuses it rather than reading it as 0.
    """
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted(folded["arms"]):
        negatives = [
            item for item in folded["items"] if item_is_gated(item) and int(item["label"]) == 0
        ]
        counts = [item["arms"][arm].get("n_payloads") for item in negatives]
        attached = all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for count in counts
        )
        by_arm[arm] = {
            "n_gated_negatives": len(negatives),
            "n_gated_negatives_with_stage1_candidate": sum(
                1 for item in negatives if int(item["arms"][arm]["n_rows"]) > 0
            ),
            "n_gated_negatives_with_stage2_payload": (
                sum(1 for count in counts if count > 0) if attached else None
            ),
        }
    gated = by_arm[gated_arm]
    reached = gated["n_gated_negatives_with_stage1_candidate"]
    underpowered = reached == 0
    return {
        "rule": (
            "ADR-0005 A15 Pin 5 — score-blind: gated negatives with >= 1 Stage-1 candidate, and "
            "with >= 1 Stage-2 payload; underpowered iff the gated arm's first count is 0"
        ),
        "gated_arm": gated_arm,
        "by_arm": by_arm,
        "verdict_class": "underpowered" if underpowered else "powered",
        "sentence": (
            UNDERPOWERED_STATEMENT
            if underpowered
            else (
                f"{reached} of {gated['n_gated_negatives']} gated negatives carry a Stage-1 "
                f"candidate in the gated arm ({gated_arm}); the benchmark can test Stage-2 on "
                "them"
            )
        ),
    }


def familiarity_baseline(
    folded: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]], *, decoy_prevalence: int
) -> dict[str, Any]:
    """A15 Pin 7(b): the flank-familiarity baseline, graded with the A13 statistic.

    Each gated item is scored by the fraction of its canonical 32-mers present in that
    arm's own Stage-1 training DNA. The score is ranked with
    :func:`~tbox_finder.metrics.average_precision_reweighted` at D7's pinned prevalence,
    with the same ``λ = ratio · n_pos / n_neg`` over the same gated population that
    :func:`arm_metrics` used for the two systems, so the three AUPRCs are directly
    comparable. ``rivals_stage1`` is ``familiarity >= Stage-1-only``. When it holds on the
    gated arm, the report states the gate is confounded by construction.
    """
    gated_items = [item for item in folded["items"] if item_is_gated(item)]
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted(folded["arms"]):
        labels: list[int] = []
        scores: list[float] = []
        for item in gated_items:
            block = item.get("familiarity")
            value = block.get(arm) if isinstance(block, Mapping) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise PrecisionError(
                    f"item {item['contig_id']!r} carries no familiarity score for arm {arm!r}; "
                    "the A15 Pin 7(b) baseline cannot be graded on a partial item set"
                )
            labels.append(int(item["label"]))
            scores.append(float(value))
        n_pos = sum(labels)
        lam = prevalence_lambda(n_pos, len(labels) - n_pos, float(int(decoy_prevalence)))
        auprc = average_precision_reweighted(labels, scores, decoy_weight=lam)
        point = arms[arm]["prevalence"][arms[arm]["gated_prevalence_key"]]
        stage1 = float(point["auprc"]["stage1_only"])
        by_arm[arm] = {
            "n_items": len(labels),
            "lambda": lam,
            "auprc": auprc,
            "stage1_only_auprc": stage1,
            "two_stage_auprc": float(point["auprc"]["two_stage"]),
            "minus_stage1_pp": (auprc - stage1) * 100.0,
            "rivals_stage1": bool(not math.isnan(auprc) and auprc >= stage1),
        }
    return {
        "statistic": "auprc_at_pinned_prevalence",
        "decoy_prevalence": decoy_prevalence,
        "score": (
            "fraction of the item contig's canonical 32-mers present in the arm's own Stage-1 "
            "training DNA (fold context_seq, the admitted mined windows, the embedded decoy "
            "inserts) — ADR-0005 A15 Pin 7(b)"
        ),
        "gated_arm": GATED_ARM,
        "by_arm": by_arm,
        "gated_arm_rivals_stage1": by_arm[GATED_ARM]["rivals_stage1"],
        "source": folded.get("familiarity"),
    }


def gated_schema_blocks(
    folded: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    *,
    decoy_prevalence: int,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Every block a 1.2 report adds, each copied or computed from the folded table.

    Blocks the fold did not attach (Stage-2 identity, screen, post-mint) are written as
    ``None``. They are never synthesised: :func:`precision_problems` refuses the absence, so
    a half-assembled run diverts to ``.invalid.json`` instead of publishing.
    """
    host_pool = scope.get("host_pool") if isinstance(scope.get("host_pool"), Mapping) else {}
    screen = folded.get("screen")
    return {
        "reported_not_gated": {
            "rule": (
                "ADR-0005 A15 Pin 4 — leader_decoy is minted and scored, and reported beside "
                "the gate; it is filtered out of the gated denominator at the precision-report "
                "call site, before arm_metrics"
            ),
            "non_gated_items": non_gated_scores(folded),
        },
        "power": power_block(folded),
        "familiarity_baseline": familiarity_baseline(
            folded, arms, decoy_prevalence=decoy_prevalence
        ),
        "stage2_identity": folded.get("stage2"),
        "screen": screen,
        "postmint": folded.get("postmint"),
        "composition": {
            "v2_drawn_hosts": host_pool.get("composition"),
            "v1_hosts": screen.get("v1_composition") if isinstance(screen, Mapping) else None,
        },
    }


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


def _fmt_counts(counts: Any, limit: int = 6) -> str:
    """``{"lead": 400, "trail": 292}`` → ``"lead 400, trail 292"``. The largest come first,
    and a long tail is summarised by its size rather than dropped."""
    if not isinstance(counts, Mapping) or not counts:
        return "not recorded"
    rows = sorted(counts.items(), key=lambda kv: (-kv[1] if isinstance(kv[1], int) else 0, kv[0]))
    shown = ", ".join(f"{key} {value}" for key, value in rows[:limit])
    return shown if len(rows) <= limit else f"{shown}, … (+{len(rows) - limit} more)"


def _fmt_composition(composition: Any) -> str:
    if not isinstance(composition, Mapping) or not composition:
        return "not recorded"
    parts = []
    for key in sorted(composition):
        value = composition[key]
        parts.append(
            f"{key}: {_fmt_counts(value)}" if isinstance(value, Mapping) else f"{key} {value}"
        )
    return "; ".join(parts)


def disclosures_v2(
    benchmark: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    report: Mapping[str, Any],
) -> list[str]:
    """The v2 report's disclosures: A15 Pin 7 (a)–(e), the power clause, and the gated subset.

    As in :func:`disclosures`, every quantity is read off the payload it describes. There
    is no fixed numeral, so a regeneration on different inputs cannot publish a stale
    count. A block the run did not attach is phrased as "not recorded" rather than raising:
    the clause set refuses the absence, and the disclosure must not hide it behind a crash.
    """
    scope = benchmark["scope"]
    seen = scope.get("seen_by_counts") if isinstance(scope.get("seen_by_counts"), Mapping) else {}
    gated = arms[GATED_ARM]
    gated_pools = scope.get(GATED_SCOPE_FIELDS["negatives_by_pool"])
    gated_pools = gated_pools if isinstance(gated_pools, Mapping) else {}
    n_neg_gated = gated["population"]["n_negatives"]
    host_pool = scope.get("host_pool") if isinstance(scope.get("host_pool"), Mapping) else {}
    overlap = host_pool.get("overlap_with_training_folds")
    by_arm = overlap.get("by_arm") if isinstance(overlap, Mapping) else None
    screen = report.get("screen") if isinstance(report.get("screen"), Mapping) else {}
    power = report["power"]
    fam = report["familiarity_baseline"]["by_arm"][GATED_ARM]
    leader = report["reported_not_gated"]["non_gated_items"].get(GATED_ARM) or []
    threshold = screen.get("threshold_bits", "not recorded")
    motif = screen.get("motif", "not recorded")
    k = (report["familiarity_baseline"].get("source") or {}).get("k", "not recorded")

    def seen_in_pool(arm: str, pool: str) -> str:
        entry = arms[arm]["per_pool"].get(pool)
        return "0 of 0" if entry is None else f"{entry['n_seen_by_arm']} of {entry['n']}"

    admitted = {
        name: arm["population"]["n_candidate_items"] - arm["population"]["n_positive_candidates"]
        for name, arm in arms.items()
    }
    retained = {name: arm["matched_recall"]["two_stage"]["fp"] for name, arm in arms.items()}
    other_arms = ", ".join(
        f"{name} {admitted[name]}/{retained[name]}" for name in sorted(arms) if name != GATED_ARM
    )
    leader_read = (
        "; ".join(
            (
                f"{entry['contig_id']} uncalled on the gated arm (no Stage-1 candidate)"
                if entry["n_rows"] == 0
                else (
                    f"{entry['contig_id']} two-stage {entry['two_stage']:.6f}, Stage-1-only "
                    f"{entry['stage1_only']:.6f} on the gated arm"
                )
            )
            for entry in leader
        )
        or "not present (no non-gated item was minted)"
    )
    dup = scope.get("insert_exposure")
    dup = (
        dup.get("n_test_inserts_duplicating_stage2_training_decoys")
        if isinstance(dup, Mapping)
        else None
    )
    dup_read = (
        f"{dup} test inserts duplicate a Stage-2 training decoy"
        if isinstance(dup, int) and not isinstance(dup, bool)
        else (
            "test inserts duplicating Stage-2 training decoys are NOT counted by this benchmark "
            "manifest (booked, not measured here)"
        )
    )
    familiar = (
        f"Flank familiarity (ADR-0005 A15 Pin 7(b)): scoring each gated item by the fraction of "
        f"its {k}-mers present in the gated arm's own Stage-1 training DNA reaches AUPRC "
        f"{fam['auprc']:.6f} at {report['gate']['decoy_prevalence']}:1, against Stage-1-only "
        f"{fam['stage1_only_auprc']:.6f} and two-stage {fam['two_stage_auprc']:.6f} "
        f"({fam['minus_stage1_pp']:+.3f} pp vs Stage-1-only). Every gated host is screened "
        "32-mer-disjoint from training DNA while the positives' flanks mostly are not, and the "
        "twin's training negatives include flank windows of the gated positives (ADR-0004 A4 "
        "admits on parent_nested_train, not on each arm's fold)."
        if isinstance(fam.get("auprc"), (int, float))
        and isinstance(fam.get("minus_stage1_pp"), (int, float))
        else "Flank familiarity (ADR-0005 A15 Pin 7(b)): the baseline AUPRC is undefined."
    )
    familiar += (
        f" The baseline rivals Stage-1, so {CONFOUNDED_STATEMENT}; a fix needs a twin retrain."
        if fam["rivals_stage1"]
        else " It does not rival Stage-1."
    )
    if power["verdict_class"] == "underpowered":
        power_line = (
            f"{UNDERPOWERED_STATEMENT}: "
            f"{power['by_arm'][GATED_ARM]['n_gated_negatives_with_stage1_candidate']} of "
            f"{power['by_arm'][GATED_ARM]['n_gated_negatives']} gated negatives carry a Stage-1 "
            "candidate in the gated arm. The benchmark holds no negative either system can "
            "discriminate — this is NOT 'two-stage did not beat Stage-1-only'. The A13 verdict "
            "is recorded unchanged (a tie fails the strict >)."
        )
    else:
        payload = power["by_arm"][GATED_ARM]["n_gated_negatives_with_stage2_payload"]
        power_line = (
            f"{power['sentence']}; {payload if payload is not None else 'an unrecorded number'} "
            "of them reach Stage-2 as a payload. ADR-0005 A13 Pin 4's power floor is re-published "
            "with v2's numbers."
        )
    return [
        (
            f"The gated arm is the GATE-4 eval twin, not the shipped scanner. The shipped "
            f"P2-10d′-b checkpoint trained on {seen.get('production', {}).get('positives')} of "
            f"the {scope.get('n_positives')} benchmark positives; the twin saw "
            f"{seen.get(GATED_ARM, {}).get('positives')}. §7 decision 2026-08-13; ADR-0004 A6's "
            "two-run split is the same instrument GATE-4 used."
        ),
        (
            "The production arm's numbers are REPORTED, NOT GATED. Its Stage-1 is in-sample "
            "on every positive, which flatters the Stage-1-only system, so a pass there is "
            "conservative and a fail there is uninterpretable."
        ),
        (
            f"The gated denominator holds {n_neg_gated} negatives from {len(gated_pools)} §9.1 "
            f"pool(s) ({_fmt_counts(gated_pools)}). The "
            f"{(scope.get('negatives_by_pool') or {}).get(NON_GATED_POOL, 0)} {NON_GATED_POOL} "
            "item(s) are minted, scored and reported BESIDE the gate (ADR-0005 A15 Pin 4; every "
            "leader_decoy record is T-box-derived, A7 Pin 5). §9.1's class 4 has no genuine "
            "records (P2-10b′ deferred); the non-delivery is disclosed, not absorbed (A7)."
        ),
        (
            "This grade supersedes the withdrawn v1 grade (reports/two_stage_precision.json, kept "
            "byte-unchanged as its record; ADR-0005 A15 Pin 5). v1's pass was carried by decoys "
            "whose hosts hold RF00230-detectable sequence."
        ),
        (
            f"Host screen (ADR-0005 A15 Pins 1–2): of {screen.get('n_candidates', 'an unrecorded')}"
            f" candidate host windows, {screen.get('n_refused_kmer', 'an unrecorded number')} were "
            "refused for sharing an N-free canonical 32-mer with the screened training DNA and "
            f"{screen.get('n_refused_coord', 'an unrecorded number')} for a same-accession "
            f"training-interval overlap; {screen.get('n_refused_content', 'an unrecorded number')}"
            f" survivors were refused for an RF00230/TBDB001 hit >= {threshold} bits or the "
            f"motif {motif}, leaving {screen.get('n_admitted', 'an unrecorded number')} to draw "
            f"from; {screen.get('n_controls_fired', '?')} of {screen.get('n_controls', '?')} "
            "must-fire controls fired. The drawn hosts' measured overlap with each arm's Stage-1 "
            f"fold is {_fmt_counts(by_arm)}. The refusal count lower-bounds T-box content in the "
            "pool; a passing screen is NOT a claim the pool is clean (sub-threshold or "
            "CM-invisible content is unseen)."
        ),
        (
            f"CM circularity and scope (ADR-0005 A15 Pin 7(a)): after the content screen a gated "
            f"negative MEANS below {threshold} bits on RF00230/TBDB001 and free of {motif}, so a "
            "CM contestant is favoured by construction (P3-17a's confirmer; GATE-1's cmsearch "
            "baseline), and a learned model that finds a real but CM-invisible T-box in a gated "
            "host is scored as a false positive. This benchmark grades Stage-1-vs-two-stage "
            "integration on an in-distribution, CM-screened negative set; it is never evidence "
            "about novel-element recall, FDR or held-out-clade generalization, and it is not "
            "quoted beside a GATE-1 or P5 claim without this sentence."
        ),
        familiar,
        (
            f"Insert-level exposure (ADR-0005 A15 Pin 7(c)): structured_rna inserts seen in "
            f"Stage-1 training — gated arm {seen_in_pool(GATED_ARM, 'structured_rna')}"
            + "".join(
                f", {name} {seen_in_pool(name, 'structured_rna')}"
                for name in sorted(arms)
                if name != GATED_ARM
            )
            + "; dinuc_shuffled inserts seen by production "
            + (
                seen_in_pool("production", "dinuc_shuffled")
                if "production" in arms
                else "not measured (no production arm)"
            )
            + f"; {dup_read}."
        ),
        (
            "The shipped Stage-2 trained on T-box-derived leader_decoy records labelled negative "
            "(A7 Pin 5's contradiction, outside A7's Stage-1 scope; ADR-0005 A15 Pin 7(d)). The "
            f"leader's beside-gate read is {leader_read}, and it is read in that light."
        ),
        (
            "Composition shift (ADR-0005 A15 Pin 7(e)): v2's drawn hosts — "
            f"{_fmt_composition(report['composition']['v2_drawn_hosts'])} — against v1's hosts "
            f"— {_fmt_composition(report['composition']['v1_hosts'])}."
        ),
        power_line,
        (
            "The ADR-0005 D7 100:1 prevalence is reached by REWEIGHTING the sampled "
            "negatives, not by materialising them — the gated population's own composition is "
            f"{gated['population']['observed_decoy_ratio']:.3f} decoys per positive. "
            "conf/data/decoys.yaml scopes the D7 pin to P4; it is reported here because imp.md's "
            "P3-16 block asks for it, and the matched-recall verdict is checked invariant across "
            "the whole 10:1 → 10⁴:1 sweep."
        ),
        (
            f"Benchmark items are {scope.get('geometry')}: a positive is its gate4_eval locus "
            "carved in real ±genomic context, a negative is a §9.1 decoy spliced into a screened "
            f"real mined host window by the shipped embed_decoy_rows. The gated arm admits "
            f"{admitted[GATED_ARM]} of {n_neg_gated} gated negatives and retains "
            f"{retained[GATED_ARM]} at matched recall ({other_arms} for the arms reported beside "
            "it). This is an in-distribution reference, not a genome-scale scan simulation (P5)."
        ),
        (
            "ADR-0005 D3 freezes the Stage-1 threshold, the locus knobs and the Stage-2 "
            "operating point at the §13.1 phase gate (P5-01). Nothing here pins any of them; "
            "the knobs this run used are recorded under sources.rule_knobs."
        ),
    ]


def stage1_threshold_sensitivity(
    points: Sequence[Mapping[str, Any]],
    *,
    base_threshold: float | None,
    base_passes: bool,
    bind_benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind each swept Stage-1 threshold to the report that measured it.

    ``bind_benchmark`` is the schema-1.2 rebinding of A13 Pin 2's "failing read" clause to
    v2's OWN τ reads, whatever their sign (ADR-0005 A15 Pin 5). It is
    ``{"benchmark_sha256": <this report's benchmark digest>, "matched_recall": <this report's
    reported read>}``. Each row then also carries the benchmark digest its own report was
    graded on, and ``base`` carries this report's matched-recall read. A v1 τ annex, graded
    on ``benchmark_v1``, therefore cannot stand in a v2 sweep, and the reported read at the
    base τ is bound to the gate it annexes. Omitted (the 1.1 path), the block is exactly
    what it always was.

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
        if bind_benchmark is not None:
            rows[-1]["benchmark_sha256"] = (
                point["report"].get("sources", {}).get("benchmark_sha256")
            )
    rows.sort(key=lambda row: row["measured_stage1_threshold"] or -1.0)
    measured = [
        row["measured_stage1_threshold"]
        for row in rows
        if row["measured_stage1_threshold"] is not None
    ]
    base: dict[str, Any] = {
        # The threshold the enclosing report was measured at, and that report's own
        # verdict. Source is named so a reader never mistakes it for a swept file.
        "stage1_threshold": base_threshold,
        "passes": bool(base_passes),
        "source": "this report",
    }
    if bind_benchmark is not None:
        base["benchmark_sha256"] = bind_benchmark.get("benchmark_sha256")
        base["matched_recall"] = bind_benchmark.get("matched_recall")
    return {
        "points": rows,
        "n_points": len(rows),
        "base": base,
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

    **Schema 1.2 (ADR-0005 A15).** A report whose ``schema_version`` parses to ≥ 1.2 is
    graded against the manifest's GATED subset: the four completeness quantities (``n_items``,
    ``n_negatives``, ``negatives_by_pool``, ``n_blocks``) are read from their ``*_gated``
    counterparts. The host-overlap bound stays the full minted negative count, because the
    minter measured it over every negative. :func:`_gated_schema_problems` then adds the A15
    clause set. A pre-1.2 report keeps exactly the clause set it was published under; the
    one exception refuses a pre-1.2 report whose manifest declares a gated subset, which
    no 1.1 report can carry, so the committed v1 bytes are untouched by it.
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
        if isinstance(arm, Mapping) and isinstance(arm.get("population"), Mapping):
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

    # `raw_scope` is the manifest as minted; `scope` is the population the gate GRADES. They
    # are one object below schema 1.2. At 1.2 the four completeness quantities are the
    # manifest's `*_gated` counts (ADR-0005 A15 Pin 4), so every clause below that binds the
    # graded population to the manifest binds it to the gated subset, unchanged in form.
    raw_scope = report["benchmark"]["scope"]
    version = schema_tuple(report.get("schema_version"))
    gated_schema = version is not None and version >= GATED_REPORT_SCHEMA
    if gated_schema:
        scope_problems, scope = _graded_scope(raw_scope)
        problems.extend(scope_problems)
    else:
        scope = raw_scope
        # The one clause outside the ≥ 1.2 gate, and it cannot touch a published 1.1 report:
        # no 1.1 manifest carries these fields. Without it, a v2 benchmark relabelled 1.1
        # would be graded with the leader inside the denominator and none of A15's clauses.
        declared = sorted(key for key in GATED_SCOPE_FIELDS.values() if key in raw_scope)
        want(
            not declared,
            f"the benchmark declares ADR-0005 A15's gated subset ({declared}) but the report "
            f"is graded under schema {report.get('schema_version')!r}; a gated benchmark is "
            "graded under the 1.2 clause set or not at all",
        )
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
                f"arm {arm_name!r}: {system} AUPRC {auprc!r} is not in [0, 1]",
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
    # from the validator is indistinguishable from a crashed run (the PR #131 finding). Every
    # hop is shape-checked, not only the last: `prevalence`, the point, and its AUPRC block
    # can each be a scalar in a payload this function did not produce (round 6).
    sweep = gated.get("prevalence")
    if not isinstance(sweep, Mapping) or not isinstance(key, str) or key not in sweep:
        problems.append(
            f"the gated arm names prevalence point {key!r} but carries no such point; the "
            "gated statistic has nowhere to be read from"
        )
        return problems
    point = sweep[key]
    if not isinstance(point, Mapping) or not isinstance(point.get("auprc"), Mapping):
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
        f"the gated arm reports passes={gated.get('passes')!r}, but its own AUPRCs "
        f"({two!r} vs {one!r}) re-derive to {derived!r}",
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
    host_pool = scope.get("host_pool") if isinstance(scope.get("host_pool"), Mapping) else {}
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
        # Each count is a number of NEGATIVES, so it is bounded by the pool it counts. A
        # filter that silently drops a malformed value would let one through unexamined, and
        # nothing bounded the counts from above: 999 of 692 passed (CodeRabbit, PR #133
        # round 7). The overlap is a published claim now, so it is bounded like one.
        # `number()` would accept a FRACTIONAL n_negatives and make it the limit, so the
        # clause below would be comparing counts against a bound that is not a count. Any
        # such manifest is already refused by four other clauses (the per-pool sum and both
        # arms' populations disagree with it), so this is the clause standing on its own
        # rather than on its neighbours (CodeRabbit, PR #133 round 8).
        # Bounded by the FULL minted negative count, not the gated one: the minter measures
        # the host overlap over every negative it spliced (A15 Pin 4 leaves that untouched).
        raw_limit = raw_scope.get("n_negatives")
        limit = (
            raw_limit if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) else None
        )

        def is_count(value: Any) -> bool:
            return (
                not isinstance(value, bool)
                and isinstance(value, int)
                and limit is not None
                and 0 <= value <= limit
            )

        want(
            bool(by_arm) and all(is_count(value) for value in by_arm.values()),
            f"a host-overlap arm count is not a count of this benchmark's "
            f"{raw_scope.get('n_negatives')!r} negatives: {by_arm!r}",
        )
        union = overlap.get("n_seen_by_any_arm")
        want(
            is_count(union),
            f"the host-overlap union {union!r} is not a count of this benchmark's "
            f"{raw_scope.get('n_negatives')!r} negatives",
        )
        counts = [value for value in by_arm.values() if is_count(value)] or [0]
        want(
            is_count(union) and max(counts) <= union <= sum(counts),
            f"the host-overlap union {union!r} is not between the largest single arm "
            f"({max(counts)}) and the sum of the arms ({sum(counts)}); it cannot be the "
            "union it names",
        )
        want(
            overlap.get("n_negatives") == raw_scope["n_negatives"],
            f"the host-overlap measurement was taken over {overlap.get('n_negatives')!r} "
            f"negatives, not this benchmark's {raw_scope['n_negatives']!r}",
        )

    by_pool = (
        scope.get("negatives_by_pool")
        if isinstance(scope.get("negatives_by_pool"), Mapping)
        else {}
    )
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
        name: bool(node.get("two_stage_beats_stage1_only")) if isinstance(node, Mapping) else None
        for name, node in sweep.items()
    }
    want(
        len(set(verdicts.values())) == 1,
        "the REPORTED matched-recall verdict is not invariant across the decoy-prevalence "
        f"sweep ({verdicts!r}); precision at matched recall is provably lambda-invariant, so "
        "this means the sweep and the selection disagree",
    )
    # AUPRC is NOT provably lambda-invariant (two PR curves can cross), so its sweep is
    # checked for presence and reported, never asserted invariant.
    for name, node in sweep.items():
        want(
            isinstance(node, Mapping) and "auprc" in node and "auprc_gain_pp" in node,
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
        # falls through to the refusal below
        problems.append(
            "stage1_threshold_sensitivity is present but is not a block; it cannot be read"
        )
    elif sensitivity is not None:
        points_block = sensitivity.get("points")
        rows = (
            [row for row in points_block if isinstance(row, Mapping)]
            if isinstance(points_block, list)
            else []
        )
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
    if gated_schema:
        problems.extend(_gated_schema_problems(report, raw_scope))

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


# ── The schema-1.2 clause set (ADR-0005 A15). Every helper RETURNS, never raises, on a
# payload it did not write (the PR #131 contract the 1.1 clauses keep). ────────────────
def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _block(node: Any, key: str) -> Mapping[str, Any]:
    value = node.get(key) if isinstance(node, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _real(node: Any, *path: str) -> float | None:
    """A finite-or-NaN real at ``path``; ``None`` for a bool, a non-number or an
    unrepresentable int (``float(10**400)`` raises, so it is caught, not trusted)."""
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    try:
        return float(node)
    except OverflowError:
        return None


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _graded_scope(raw: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """The manifest view a 1.2 report's completeness clauses compare against (A15 Pin 4).

    Each ``*_gated`` count replaces its full-benchmark counterpart. A missing or malformed one
    is a problem, and the full value is left in place, so every clause that reads it
    disagrees with the gated population and fires too. It is never read as zero.
    """
    view = dict(raw)
    problems: list[str] = []
    for full, gated in GATED_SCOPE_FIELDS.items():
        value = raw.get(gated)
        ok = (
            isinstance(value, Mapping)
            and all(isinstance(key, str) and _is_count(count) for key, count in value.items())
            if full == "negatives_by_pool"
            else _is_count(value)
        )
        if ok:
            view[full] = dict(value) if isinstance(value, Mapping) else value
        else:
            problems.append(
                f"benchmark.scope.{gated} is {value!r}; a schema-1.2 report grades the gated "
                "subset the manifest declares (ADR-0005 A15 Pin 4), and none is declared"
            )
    return problems, view


def _gated_schema_problems(report: Mapping[str, Any], scope: Mapping[str, Any]) -> list[str]:
    """ADR-0005 A15's clauses, each re-derived from the report's own numbers.

    ``scope`` is the RAW manifest scope (the full minted counts). Each clause is breakable
    alone, with every other field consistent, so ``all(...)`` cannot hide one. The tests
    break each one in isolation ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    problems: list[str] = []

    def want(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    def close(observed: Any, derived: float | None) -> bool:
        value = _real({"v": observed}, "v")
        if value is None or derived is None or math.isnan(value) or math.isnan(derived):
            return False
        return math.isclose(value, derived, rel_tol=1e-9, abs_tol=1e-9)

    arms = {name: arm for name, arm in report["arms"].items() if isinstance(arm, Mapping)}
    gated_arm = report["gated_arm"]
    gate = _block(report, "gate")
    sources = _block(report, "sources")
    interim = _block(sources, "interim_sha256")
    host_pool = _block(scope, "host_pool")
    screen = _block(report, "screen")
    disclosures_block = report.get("disclosures")
    lines = (
        [line for line in disclosures_block if isinstance(line, str)]
        if isinstance(disclosures_block, list)
        else []
    )

    # ── (1) The non-gated set is EXACTLY the leader pool, in the manifest … ────────────
    full_pools = scope.get("negatives_by_pool")
    gated_pools = scope.get(GATED_SCOPE_FIELDS["negatives_by_pool"])
    n_leader = full_pools.get(NON_GATED_POOL, 0) if isinstance(full_pools, Mapping) else None
    want(
        isinstance(full_pools, Mapping)
        and isinstance(gated_pools, Mapping)
        and dict(gated_pools)
        == {pool: count for pool, count in full_pools.items() if pool != NON_GATED_POOL},
        f"the manifest's gated pools {gated_pools!r} are not its pools minus {NON_GATED_POOL!r} "
        f"({full_pools!r}); A15 Pin 4 reports exactly that pool beside the gate and no other",
    )
    counts = {key: scope.get(key) for key in ("n_items", "n_negatives", "n_blocks")}
    gated_counts = {key: scope.get(GATED_SCOPE_FIELDS[key]) for key in counts}
    want(
        _is_count(n_leader)
        and all(_is_count(v) for v in (*counts.values(), *gated_counts.values()))
        and counts["n_items"] - n_leader == gated_counts["n_items"]
        and counts["n_negatives"] - n_leader == gated_counts["n_negatives"]
        and gated_counts["n_blocks"] <= counts["n_blocks"],
        f"the manifest's gated counts {gated_counts!r} are not its full counts {counts!r} less "
        f"the {n_leader!r} {NON_GATED_POOL} item(s)",
    )
    n_non_gated = (
        counts["n_items"] - gated_counts["n_items"]
        if _is_count(counts["n_items"]) and _is_count(gated_counts["n_items"])
        else None
    )

    # ── … and in the report, with both its scores present per arm ───────────────────────
    non_gated = _block(report, "reported_not_gated").get("non_gated_items")
    want(
        isinstance(non_gated, Mapping) and set(non_gated) == set(arms),
        "reported_not_gated.non_gated_items does not carry every arm; A15 Pin 4 requires "
        "the leader's scores beside the gate on each",
    )
    if isinstance(non_gated, Mapping):
        ids_by_arm: dict[str, list[str]] = {}
        for arm_name, rows in non_gated.items():
            if not isinstance(rows, list) or not all(isinstance(r, Mapping) for r in rows):
                problems.append(f"reported_not_gated.non_gated_items[{arm_name!r}] is not a list")
                continue
            want(
                all(row.get("pool") == NON_GATED_POOL for row in rows),
                f"arm {arm_name!r} reports non-gated items from "
                f"{sorted({str(row.get('pool')) for row in rows})}; only {NON_GATED_POOL!r} is "
                "reported beside the gate",
            )
            want(
                n_non_gated is not None and len(rows) == n_non_gated,
                f"arm {arm_name!r} reports {len(rows)} non-gated item(s) but the manifest "
                f"leaves {n_non_gated!r} out of the gated subset",
            )
            ids_by_arm[str(arm_name)] = sorted(str(row.get("contig_id")) for row in rows)
            for row in rows:
                n_rows = row.get("n_rows")
                one, two = _real(row, "stage1_only"), _real(row, "two_stage")
                if not _is_count(n_rows) or one is None or two is None:
                    scored = False
                elif n_rows == 0:
                    scored = one == UNCALLED_SCORE and two == UNCALLED_SCORE
                else:
                    scored = 0.0 <= one <= 1.0 and 0.0 <= two <= 1.0
                want(
                    scored and row.get("label") == 0,
                    f"arm {arm_name!r}: non-gated item {row.get('contig_id')!r} does not carry "
                    "both systems' scores consistent with its candidate count",
                )
        want(
            len({tuple(ids) for ids in ids_by_arm.values()}) <= 1,
            "the arms report different non-gated items",
        )

    # ── (2) Power (A15 Pin 5), re-derived from each arm's own gated population ─────────
    power = _block(report, "power")
    by_arm_power = _block(power, "by_arm")
    want(
        set(by_arm_power) == set(arms),
        f"the power block covers {sorted(by_arm_power)!r}, not the graded arms {sorted(arms)!r}",
    )
    for arm_name, arm in arms.items():
        entry = _block(by_arm_power, arm_name)
        population = _block(arm, "population")
        neg = population.get("n_negatives")
        cand = (
            population.get("n_candidate_items") - population.get("n_positive_candidates")
            if _is_count(population.get("n_candidate_items"))
            and _is_count(population.get("n_positive_candidates"))
            else None
        )
        n_gated, with_one, with_two = (
            entry.get("n_gated_negatives"),
            entry.get("n_gated_negatives_with_stage1_candidate"),
            entry.get("n_gated_negatives_with_stage2_payload"),
        )
        want(
            _is_count(n_gated) and n_gated == neg,
            f"arm {arm_name!r}: the power block counts {n_gated!r} gated negatives but the arm "
            f"graded {neg!r}",
        )
        want(
            _is_count(with_one) and with_one == cand,
            f"arm {arm_name!r}: the power block counts {with_one!r} gated negatives with a "
            f"Stage-1 candidate, but the arm's own population re-derives {cand!r}",
        )
        want(
            _is_count(with_two)
            and _is_count(with_one)
            and _is_count(n_gated)
            and with_one <= with_two <= n_gated,
            f"arm {arm_name!r}: {with_two!r} gated negatives reached Stage-2 as a payload, which "
            f"is not between the {with_one!r} with a candidate (every emitted locus is a payload) "
            f"and the {n_gated!r} gated negatives",
        )
    reached = _block(by_arm_power, gated_arm).get("n_gated_negatives_with_stage1_candidate")
    expected_class = (
        None if not _is_count(reached) else ("underpowered" if reached == 0 else "powered")
    )
    want(
        expected_class is not None and power.get("verdict_class") == expected_class,
        f"power.verdict_class is {power.get('verdict_class')!r}, but the gated arm's count of "
        f"gated negatives with a Stage-1 candidate ({reached!r}) re-derives {expected_class!r}",
    )
    want(
        power.get("gated_arm") == gated_arm,
        f"the power block names gated arm {power.get('gated_arm')!r}, not {gated_arm!r}",
    )
    want(
        gate.get("verdict_class") == power.get("verdict_class"),
        "gate.verdict_class does not mirror the power block's",
    )
    if expected_class == "underpowered":
        want(
            gate.get("underpowered_statement") == UNDERPOWERED_STATEMENT
            and power.get("sentence") == UNDERPOWERED_STATEMENT,
            "an underpowered gate does not carry A15 Pin 5's sentence verbatim",
        )
    else:
        want(
            gate.get("underpowered_statement") is None,
            "a powered gate carries the underpowered statement",
        )

    # ── (3) Stage-2 identity (A15 Pin 6): present, identical across arms, = the T source ─
    identity = report.get("stage2_identity")
    temperature = _block(sources, "temperature_source")
    want(
        isinstance(identity, Mapping) and set(identity) == set(arms),
        "stage2_identity does not cover every graded arm; the Stage-2 checkpoint each arm was "
        "re-ranked by must be recorded (ADR-0005 A15 Pin 6)",
    )
    records = (
        {name: rec for name, rec in identity.items() if isinstance(rec, Mapping)}
        if isinstance(identity, Mapping)
        else {}
    )
    for arm_name, rec in records.items():
        absent = [field for field in STAGE2_IDENTITY_FIELDS if not _present(rec.get(field))]
        want(not absent, f"arm {arm_name!r}: stage2_identity records no {absent}")
        want(
            rec.get("shard") is None,
            f"arm {arm_name!r}: the Stage-2 file is shard {rec.get('shard')!r}, one slice of the "
            "work list, not a whole pass (merge the shards first)",
        )
        want(
            _present(rec.get("payloads_sha256")),
            f"arm {arm_name!r}: stage2_identity records no payloads_sha256; the manifest the "
            "Stage-2 pass scored is unnamed",
        )
        coverage = _block(rec, "work_list_coverage")
        n_rows, n_scored = coverage.get("n_rows"), coverage.get("n_scored")
        want(
            _is_count(n_rows) and n_rows > 0 and n_scored == n_rows,
            f"arm {arm_name!r}: the Stage-2 logits score {n_scored!r} of the {n_rows!r} payloads "
            "in the graded work list; every payload the candidate tables were built from must "
            "have been scored",
        )
    first = next(iter(records.values()), None)
    want(
        first is not None
        and all(
            rec.get(field) == first.get(field)
            for rec in records.values()
            for field in STAGE2_IDENTITY_FIELDS
        ),
        "the arms were re-ranked by different Stage-2 identities; both arms must share one "
        "checkpoint, backbone, revision, dtype, adapter, heads and batch size",
    )
    want(
        first is not None
        and all(
            _present(temperature.get(field)) and first.get(field) == temperature.get(field)
            for field in TEMPERATURE_BINDING_FIELDS
        ),
        "the Stage-2 identity is not the arm the temperature source graded "
        f"({ {field: temperature.get(field) for field in TEMPERATURE_BINDING_FIELDS} }); the T "
        "applied would belong to a different checkpoint",
    )
    t_value = _real(temperature, "temperature")
    want(
        t_value is not None and math.isfinite(t_value) and t_value > 0.0,
        f"the temperature source records T = {temperature.get('temperature')!r}, not a "
        "positive finite temperature",
    )

    # ── (4) Every v2 input recorded with its sha256, and each block bound to its entry ──
    for key in V2_SOURCE_FILES:
        rel = sources.get(key)
        want(
            isinstance(rel, str) and isinstance(interim.get(rel), str) and bool(interim.get(rel)),
            f"sources.{key} ({rel!r}) is not recorded in sources.interim_sha256",
        )
    file_blocks: list[tuple[str, Any, Any]] = [
        (f"stage2_identity[{name!r}]", records.get(name), sources.get(f"{name}_stage2"))
        for name in sorted(arms)
    ]
    file_blocks += [
        ("screen", report.get("screen"), sources.get("host_screen")),
        ("postmint", report.get("postmint"), sources.get("postmint")),
        (
            "familiarity_baseline.source",
            _block(report, "familiarity_baseline").get("source"),
            sources.get("familiarity"),
        ),
        ("sources.temperature_source", sources.get("temperature_source"), None),
    ]
    for label, node, expected_path in file_blocks:
        path = node.get("path") if isinstance(node, Mapping) else None
        want(
            isinstance(path, str)
            and (expected_path is None or path == expected_path)
            and _present(interim.get(path))
            and node.get("sha256") == interim.get(path),
            f"{label} is not bound to a hash-recorded input (path {path!r}); a block that "
            "names no digest could describe any file",
        )
    want(
        _present(sources.get("benchmark_sha256"))
        and isinstance(sources.get("benchmark"), str)
        and sources.get("benchmark_sha256") == interim.get(sources.get("benchmark")),
        "sources.benchmark_sha256 is not the recorded digest of sources.benchmark",
    )

    # ── (5) The host-overlap conjunction (A15 Pin 6) — a bare `== 0` is not the test ────
    overlap = _block(host_pool, "overlap_with_training_folds")
    gated_overlap = _block(overlap, "by_arm").get(gated_arm)
    want(
        _is_count(gated_overlap) and gated_overlap == 0,
        f"conjunct 1/4: the gated arm's measured host overlap is {gated_overlap!r}, not 0",
    )
    want(
        _is_count(screen.get("n_refused_kmer")) and screen.get("n_refused_kmer") > 0,
        f"conjunct 2/4: the screen refused {screen.get('n_refused_kmer')!r} windows on the "
        "k-mer rule; a screen that refuses nothing cannot be told from one that never ran",
    )
    n_controls, n_fired = screen.get("n_controls"), screen.get("n_controls_fired")
    want(
        screen.get("controls_all_fired") is True
        and _is_count(n_controls)
        and n_controls > 0
        and n_fired == n_controls,
        f"conjunct 3/4: the screen's must-fire controls did not all fire "
        f"(controls_all_fired={screen.get('controls_all_fired')!r}, {n_fired!r} of "
        f"{n_controls!r})",
    )
    selection = host_pool.get("selection")
    want(
        isinstance(selection, str) and SCREEN_SELECTION_TAG in selection,
        f"conjunct 4/4: the host selection {selection!r} does not name the "
        f"{SCREEN_SELECTION_TAG} screen",
    )
    minted_under = _block(host_pool, "screen").get("sha256")
    want(
        _present(minted_under) and screen.get("sha256") == minted_under,
        f"the screen report read here ({screen.get('sha256')!r}) is not the one the benchmark "
        f"was minted under ({minted_under!r})",
    )
    want(
        screen.get("n_problems") == 0,
        f"the screen report carries {screen.get('n_problems')!r} problem(s)",
    )
    threshold = screen.get("threshold_bits")
    want(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and threshold == SCREEN_THRESHOLD_BITS,
        f"the screen ran at {threshold!r} bits, not ADR-0005 A15 Pin 2's "
        f"{SCREEN_THRESHOLD_BITS}",
    )

    # ── (6) The post-mint assertion (A15 Pin 3) ─────────────────────────────────────────
    postmint = _block(report, "postmint")
    want(
        _is_count(postmint.get("n_violations")) and postmint.get("n_violations") == 0,
        f"the post-mint screen records {postmint.get('n_violations')!r} violation(s); A15 Pin 3 "
        "requires 0 before any score is read, and its only remedy is a §7 stop",
    )
    want(
        postmint.get("n_problems") == 0,
        f"the post-mint screen report carries {postmint.get('n_problems')!r} problem(s)",
    )
    # A15 Pin 6: the verdict is a conjunction. `n_violations == 0` reads identically at the
    # signed thresholds, at a forgotten 10.0 and at an absurd 1000.0, so the gate also checks
    # WHICH thresholds produced it and that each one cleared its own measured matched-null
    # floor. Re-derived here from the folded record rather than imported from host_screen, so
    # the two modules have to agree rather than share one value.
    pinned = postmint.get("threshold_bits")
    floor = postmint.get("null_max_bits")
    want(
        isinstance(pinned, Mapping) and bool(pinned),
        f"the post-mint record carries no per-model threshold_bits ({pinned!r}); A15 Pin 3's "
        "verdict cannot be read without the threshold it was taken at",
    )
    want(
        bool(postmint.get("calibration_sha256")),
        "the post-mint record names no calibration artifact; a re-signed threshold with no "
        "measured floor beside it is a loosened gate (CLAUDE.md §8.5)",
    )
    # Not `if isinstance(floor, Mapping)`: a missing calibration block would then skip the
    # whole floor comparison and the gate would pass on no evidence at all — the clause would
    # be vacuously satisfied exactly when the evidence is absent
    # ([[clauses-must-guard-emptiness]]). A missing floor is a FAILURE, not a skip.
    if isinstance(pinned, Mapping) and pinned:
        floor = floor if isinstance(floor, Mapping) else {}
        for model in sorted(pinned):
            bits, measured = pinned.get(model), floor.get(model)
            want(
                isinstance(bits, (int, float))
                and not isinstance(bits, bool)
                and isinstance(measured, (int, float))
                and not isinstance(measured, bool)
                # Compared without float() coercion: float(10**400) raises OverflowError, and a
                # validator that raises on a retyped node is not total. int/float comparison in
                # Python is exact and cannot overflow; NaN compares False, which fails closed.
                and bits > measured,
                f"the post-mint assertion ran {model} at {bits!r} bits against a measured "
                f"matched-null maximum of {measured!r}: inside the noise floor",
            )
            want(
                isinstance(bits, (int, float))
                and not isinstance(bits, bool)
                and bits >= SCREEN_THRESHOLD_BITS,
                f"the post-mint assertion ran {model} at {bits!r} bits, below Pin 2's screen "
                f"({SCREEN_THRESHOLD_BITS}): the assertion would be stronger than its own screen",
            )

    # ── (7) The familiarity baseline (A15 Pin 7(b)), re-derived against each arm ────────
    baseline = _block(report, "familiarity_baseline")
    by_arm_fam = _block(baseline, "by_arm")
    want(
        set(by_arm_fam) == set(arms),
        f"the familiarity baseline covers {sorted(by_arm_fam)!r}, not {sorted(arms)!r}",
    )
    for arm_name, arm in arms.items():
        entry = _block(by_arm_fam, arm_name)
        key = arm.get("gated_prevalence_key")
        point = _block(_block(arm, "prevalence"), key) if isinstance(key, str) else {}
        auprc, stage1 = _real(entry, "auprc"), _real(entry, "stage1_only_auprc")
        want(
            auprc is not None and 0.0 <= auprc <= 1.0,
            f"arm {arm_name!r}: familiarity AUPRC {entry.get('auprc')!r} is not in [0, 1]",
        )
        want(
            stage1 is not None
            and stage1 == _real(point, "auprc", "stage1_only")
            and _real(entry, "two_stage_auprc") == _real(point, "auprc", "two_stage"),
            f"arm {arm_name!r}: the familiarity baseline is published beside AUPRCs that are not "
            "the arm's own at the pinned prevalence",
        )
        want(
            close(entry.get("lambda"), _real(point, "lambda"))
            and entry.get("n_items") == _block(arm, "population").get("n_items"),
            f"arm {arm_name!r}: the familiarity baseline was not graded over the arm's own "
            "gated population at the pinned reweighting",
        )
        want(
            auprc is not None
            and stage1 is not None
            and close(entry.get("minus_stage1_pp"), (auprc - stage1) * 100.0),
            f"arm {arm_name!r}: familiarity minus_stage1_pp {entry.get('minus_stage1_pp')!r} is "
            "not the difference of the AUPRCs it is published beside",
        )
        want(
            isinstance(entry.get("rivals_stage1"), bool)
            and auprc is not None
            and stage1 is not None
            and entry.get("rivals_stage1") == (auprc >= stage1),
            f"arm {arm_name!r}: rivals_stage1 {entry.get('rivals_stage1')!r} is not the "
            "comparison it names",
        )
    rivals = _block(by_arm_fam, gated_arm).get("rivals_stage1")
    want(
        baseline.get("gated_arm_rivals_stage1") is rivals and isinstance(rivals, bool),
        "familiarity_baseline.gated_arm_rivals_stage1 is not the gated arm's own comparison",
    )
    want(
        any(CONFOUNDED_STATEMENT in line for line in lines) == bool(rivals),
        (
            f"the gated arm's familiarity baseline rivals Stage-1 but no disclosure says "
            f"{CONFOUNDED_STATEMENT!r} (A15 Pin 7(b))"
            if rivals
            else f"a disclosure claims {CONFOUNDED_STATEMENT!r} but the baseline does not rival "
            "Stage-1"
        ),
    )

    # ── (8) A13 Pin 2's failing-read clause, rebound to v2's own τ reads (A15 Pin 5) ────
    sensitivity = report.get("stage1_threshold_sensitivity")
    if isinstance(sensitivity, Mapping):
        bench = sources.get("benchmark_sha256")
        rows = sensitivity.get("points")
        rows = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
        want(
            _present(bench) and all(row.get("benchmark_sha256") == bench for row in rows),
            "a Stage-1 threshold sensitivity point was graded on a different benchmark than this "
            "report; A15 Pin 5 rebinds the τ reads to v2's own, whatever their sign",
        )
        reported = _block(gate, "reported_not_gated").get("matched_recall")
        want(
            isinstance(reported, Mapping)
            and _block(sensitivity, "base").get("matched_recall") == reported,
            "the sensitivity annex's base does not carry this report's own matched-recall read",
        )

    # ── (9) Composition (A15 Pin 7(e)) is the minted one beside the screen's v1 audit ───
    composition = _block(report, "composition")
    v2_hosts, v1_hosts = composition.get("v2_drawn_hosts"), composition.get("v1_hosts")
    want(
        isinstance(v2_hosts, Mapping)
        and bool(v2_hosts)
        and v2_hosts == host_pool.get("composition"),
        "composition.v2_drawn_hosts is not the minted scope.host_pool.composition",
    )
    want(
        isinstance(v1_hosts, Mapping)
        and bool(v1_hosts)
        and v1_hosts == screen.get("v1_composition"),
        "composition.v1_hosts is not the screen report's v1_audit composition",
    )
    return problems


# ══════════════════════════════════════════════════════════════════════════════════════
# File binding (schema 1.2): the report against the bytes it names
# ══════════════════════════════════════════════════════════════════════════════════════
def _relative_problem(rel: Any) -> str | None:
    """Why ``rel`` is not a repo-relative path, or ``None`` if it is."""
    if not isinstance(rel, str) or not rel.strip():
        return f"{rel!r} is not a path"
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        return f"{rel!r} is not repo-relative"
    return None


def precision_input_problems(
    report: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    require_all_inputs: bool = False,
    items: Mapping[str, Any] | None = None,
) -> list[str]:
    """Problems binding a schema-1.2 report to the files it names; ``[]`` means bound.

    It complements :func:`precision_problems`, which reads only the report's own numbers.
    It follows the ``cm_confirmer.validate_report(require_all_inputs=...)`` split:

    * every ``sources.interim_sha256`` entry is re-hashed when its file exists
      (:func:`~tbox_finder.refs.content_sha256`, so an LFS pointer checks through its own
      ``oid``). A committed file (:data:`COMMITTED_PREFIXES`: the screen and post-mint
      reports, the GATE-2 temperature source) must exist wherever the report is checked.
      An interim file is checked only when present, unless ``require_all_inputs``;
    * the item table (``sources.items``, committed) must hash to the digest provenance
      recorded for it;
    * the power counts, the familiarity baseline and the non-gated scores are re-derived
      from that item table with the same functions the producer used. ``items`` passes the
      table in memory, and is otherwise read from ``sources.items``.

    A pre-1.2 report records no interim digests, so this returns ``[]`` for it. v1's
    bindings are its committed items file and P3-17a's fixed-path ``CANONICAL_SOURCES``.
    Never raises on a payload it did not write.
    """
    if not isinstance(report, Mapping):
        return ["the report is not a mapping"]
    version = schema_tuple(report.get("schema_version"))
    if version is None or version < GATED_REPORT_SCHEMA:
        return []
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    problems: list[str] = []
    sources = _block(report, "sources")
    interim = sources.get("interim_sha256")
    if not isinstance(interim, Mapping) or not interim:
        return ["sources.interim_sha256 is missing or empty; no input can be re-hashed"]

    def check(rel: Any, digest: Any, what: str) -> bool:
        """Re-hash one file; returns whether it was present and matched."""
        why = _relative_problem(rel)
        if why is not None:
            problems.append(f"{what}: {why}")
            return False
        path = root / rel
        if not path.is_file():
            if require_all_inputs or str(rel).startswith(COMMITTED_PREFIXES):
                problems.append(f"{what} {rel} is absent")
            return False
        try:
            actual = refs.content_sha256(path)
        except (OSError, ValueError) as error:
            problems.append(f"{what} {rel} cannot be hashed: {error!r}")
            return False
        if actual != digest:
            problems.append(f"{what} {rel} no longer hashes to the recorded {str(digest)[:12]}…")
            return False
        return True

    present: dict[str, bool] = {}
    for rel, digest in sorted(interim.items(), key=lambda kv: str(kv[0])):
        present[str(rel)] = check(rel, digest, "input")

    # The Stage-2 work-list coverage, recomputed from the two files when both are on disk
    # (interim, so not in CI) and both still hash to what the fold recorded.
    identity = report.get("stage2_identity")
    for arm, record in (identity.items() if isinstance(identity, Mapping) else []):
        stage2_rel, payload_rel = sources.get(f"{arm}_stage2"), sources.get(f"{arm}_payloads")
        if not (present.get(str(stage2_rel)) and present.get(str(payload_rel))):
            continue
        try:
            derived = work_list_coverage(
                read_json(root / stage2_rel), read_json(root / payload_rel)["payloads"]
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            problems.append(
                f"arm {arm!r}: the Stage-2 work-list coverage cannot be re-derived: {error!r}"
            )
            continue
        if not isinstance(record, Mapping) or record.get("work_list_coverage") != derived:
            problems.append(
                f"arm {arm!r}: the recorded Stage-2 work-list coverage does not re-derive from "
                f"{stage2_rel} and {payload_rel} ({derived!r})"
            )

    items_rel = sources.get("items")
    provenance_inputs = _block(report, "provenance").get("inputs")
    recorded = provenance_inputs.get(items_rel) if isinstance(provenance_inputs, Mapping) else None
    if not isinstance(items_rel, str) or not _present(recorded):
        problems.append(
            f"the item table {items_rel!r} carries no provenance digest; the graded rows are "
            "not hash-bound"
        )
    else:
        items_present = check(items_rel, recorded, "item table")
        if items is None and items_present:
            try:
                items = read_json(root / items_rel)
            except (OSError, ValueError) as error:
                return [*problems, f"the item table {items_rel} cannot be read: {error!r}"]
    if items is None:
        return problems

    # Re-derived from the item table with the producer's own functions. Re-running the
    # bootstrap is not needed: `precision_problems` already re-derives every published
    # magnitude from the report, so only what the report cannot carry is re-read here.
    try:
        derived_power = power_block(items)
        derived_fam = familiarity_baseline(
            items, report["arms"], decoy_prevalence=int(report["gate"]["decoy_prevalence"])
        )
        derived_non_gated = json_safe(non_gated_scores(items))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return [*problems, f"the item table cannot be re-derived: {error!r}"]
    if json_safe(derived_power["by_arm"]) != _block(report, "power").get("by_arm"):
        problems.append("the power counts do not re-derive from the item table")
    if json_safe(derived_non_gated) != _block(report, "reported_not_gated").get("non_gated_items"):
        problems.append("the non-gated scores do not re-derive from the item table")
    published = _block(_block(report, "familiarity_baseline"), "by_arm")
    for arm, entry in derived_fam["by_arm"].items():
        observed = _real(_block(published, arm), "auprc")
        if observed is None or not math.isclose(
            observed, entry["auprc"], rel_tol=1e-9, abs_tol=1e-12
        ):
            problems.append(
                f"arm {arm!r}: the familiarity AUPRC {observed!r} does not re-derive from the "
                f"item table ({entry['auprc']!r})"
            )
    return problems


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


#: ``fold`` arguments that only a gated-subset (v2) manifest takes. All of them are required
#: for one, and every one is refused for an older manifest, so a v1 re-fold stays exactly
#: what it was.
V2_FOLD_ARGS: tuple[str, ...] = (
    "twin_stage1",
    "production_stage1",
    "twin_stage2",
    "production_stage2",
    "twin_payloads",
    "production_payloads",
    "temperature_source",
    "host_screen",
    "postmint",
    "familiarity",
    "rule_knobs",
)


def refuse_table_scored_by_another_run(
    rows: Sequence[Mapping[str, Any]], stage2: Mapping[str, Any], *, arm: str, path: str
) -> None:
    """Refuse when a candidate table's ``stage2_logit`` is not what ``stage2`` holds for its key.

    Compared exactly, not within a tolerance: both numbers are the same float round-tripped
    through the same JSON writer, so any difference means a different scoring run rather than
    arithmetic drift. A row whose ``payload_key`` the Stage-2 file does not carry is also a
    refusal — the fail-open reading would be to skip it.
    """
    logits = stage2.get("logits")
    logits = logits if isinstance(logits, Mapping) else {}
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        key, recorded = row.get("payload_key"), row.get("stage2_logit")
        if key is None or recorded is None:
            continue
        if key not in logits:
            raise PrecisionError(
                f"the {arm} candidate table has payload {str(key)[:12]}… that {path} does not "
                "score; its rows were produced by a different Stage-2 run"
            )
        if logits[key] != recorded:
            raise PrecisionError(
                f"the {arm} candidate table scores payload {str(key)[:12]}… at {recorded!r}, "
                f"but {path} holds {logits[key]!r}: the table was ranked by a different "
                "Stage-2 run than the one this report records as its scoring identity"
            )


def attach_v2_inputs(folded: dict[str, Any], args: argparse.Namespace) -> None:
    """Record every v2 input on the folded table with its repo-relative path and sha256.

    It reads, copies and hashes, and never re-types a value (ADR-0005 A15 Pin 6). The files
    are: both arms' Stage-1 logits (hashed only), Stage-2 logits (identity + shard layout)
    and payload manifests (per-item payload counts → power); the GATE-2 temperature source;
    the host-screen and post-mint reports; the familiarity scores; and the D3 locus knobs.
    Cross-file refusals (each a different run's file standing in) are raised here, before
    anything is written.
    """
    digests: dict[str, str] = {}

    def bind(path: str) -> tuple[str, str]:
        rel = repo_relative(path)
        digests[rel] = refs.content_sha256(path)
        return rel, digests[rel]

    source = folded["source"]
    _, bench_sha = bind(args.benchmark)
    bind(args.twin_table)
    bind(args.production_table)
    folded["stage2"] = {}
    for arm in ("production", "twin"):
        source[f"{arm}_stage1"] = bind(getattr(args, f"{arm}_stage1"))[0]
        rel, _ = bind(getattr(args, f"{arm}_payloads"))
        source[f"{arm}_payloads"] = rel
        work_list = read_json(getattr(args, f"{arm}_payloads"))["payloads"]
        attach_payload_counts(folded, arm, work_list)
        rel, digest = bind(getattr(args, f"{arm}_stage2"))
        source[f"{arm}_stage2"] = rel
        stage2 = read_json(getattr(args, f"{arm}_stage2"))
        folded["stage2"][arm] = stage2_identity_record(
            stage2, path=rel, sha256=digest, work_list=work_list
        )
        # The candidate table carries the Stage-2 logit each row was ranked on; this file is
        # the Stage-2 run being recorded as the scoring identity. Row-ID coverage and input
        # hashing both pass when they are DIFFERENT runs — a work list can be re-scored while
        # its table stays on disk — and the report would then publish one run's identity beside
        # another run's scores. Re-derived per row rather than assumed
        # ([[gate-must-bind-to-upstream-evidence]]).
        refuse_table_scored_by_another_run(
            read_table(getattr(args, f"{arm}_table")), stage2, arm=arm, path=rel
        )
    rel, digest = bind(args.temperature_source)
    source["temperature_source"] = temperature_source_record(
        read_json(args.temperature_source),
        path=rel,
        sha256=digest,
        temperature=read_temperature(args.temperature_source),
    )
    rel, digest = bind(args.host_screen)
    source["host_screen"] = rel
    folded["screen"] = screen_record(read_json(args.host_screen), path=rel, sha256=digest)
    rel, digest = bind(args.postmint)
    source["postmint"] = rel
    folded["postmint"] = postmint_record(read_json(args.postmint), path=rel, sha256=digest)
    recorded = folded["postmint"]["benchmark_sha256"]
    if recorded is not None and recorded != bench_sha:
        raise PrecisionError(
            f"the post-mint screen asserts on benchmark {recorded!r}, not this one ({bench_sha!r})"
        )
    rel, digest = bind(args.familiarity)
    source["familiarity"] = rel
    folded["familiarity"] = {
        "path": rel,
        "sha256": digest,
        **attach_familiarity(folded, read_json(args.familiarity), benchmark_sha256=bench_sha),
    }
    knobs = parse_rule_knobs(args.rule_knobs)
    if knobs["threshold"] != args.stage1_threshold:
        raise PrecisionError(
            f"--rule-knobs threshold={knobs['threshold']!r} but --stage1-threshold is "
            f"{args.stage1_threshold!r}; the candidate tables were built under one of them"
        )
    source["rule_knobs"] = knobs
    source["benchmark_sha256"] = bench_sha
    source["interim_sha256"] = dict(sorted(digests.items()))


def _cmd_fold(args: argparse.Namespace) -> int:
    """LEG 1 (torch-free, but reads GPU-derived tables): candidate tables → the item table."""
    # A15 Pin 5: no v2 artifact is written to a v1 path. Checked HERE, before the fold reads
    # anything, not at the write: a guard that fires after the work is done has already paid
    # for the overwrite it is meant to prevent ([[guard-runs-after-what-it-guards]]). The same
    # guard already protects five two_stage entrypoints; these were its unguarded siblings.
    refuse_v1_output(args.out)
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
    given = {name: getattr(args, name, None) for name in V2_FOLD_ARGS}
    if folded_declares_gated_subset(folded):
        missing = sorted(
            f"--{name.replace('_', '-')}" for name, value in given.items() if not value
        )
        if missing:
            raise PrecisionError(
                f"the manifest declares ADR-0005 A15's gated subset, so the v2 fold needs "
                f"{missing}; a 1.2 report binds every one of them"
            )
        attach_v2_inputs(folded, args)
    elif any(given.values()):
        raise PrecisionError(
            f"v2 inputs {sorted(name for name, value in given.items() if value)} were given for "
            "a manifest that declares no gated subset; a v1 fold takes none of them"
        )
    write_json(args.out, folded)
    print(
        f"wrote {args.out}: {len(folded['items'])} items × {len(folded['arms'])} arms "
        f"({', '.join(folded['arms'])}), schema {folded['schema_version']}"
    )
    return 0


def route_report(
    report: dict[str, Any], out: str | Path, *, extra_problems: Sequence[str] = ()
) -> int:
    """Validate, then write to the canonical path or divert; returns the exit code.

    The exit codes are unchanged: **3** for a refused report (any problem, or any
    completeness clause FALSE, written to ``.invalid.json`` with the canonical path
    removed), **4** for a valid report whose gate fails, **0** for a pass. An underpowered
    tie (ADR-0005 A15 Pin 5) is a VALID report that fails. It lands on the canonical path
    and returns 4, because "no power" is a measured outcome and not a broken run.
    """
    problems = [*precision_problems(report), *extra_problems]
    report["problems"] = problems
    out = Path(out)
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
        + (f"; verdict_class {gate['verdict_class']}" if "verdict_class" in gate else "")
    )
    return 0 if gate["passes"] else 4


def _cmd_report(args: argparse.Namespace) -> int:
    """LEG 2 (torch-free, CI-runnable): the item table → the graded report."""
    refuse_v1_output(args.out)  # A15 Pin 5, before the bootstrap, not at the write
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
    gated_schema = report["schema_version"] == SCHEMA_VERSION
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
                # A15 Pin 5: under 1.2 the τ reads are bound to THIS benchmark and the base
                # carries this report's own matched-recall read, whatever its sign.
                bind_benchmark=(
                    {
                        "benchmark_sha256": sources.get("benchmark_sha256"),
                        "matched_recall": report["gate"]["reported_not_gated"]["matched_recall"],
                    }
                    if gated_schema
                    else None
                ),
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
    # Schema 1.2 also re-hashes every file the report names, while they are still on disk
    # (interim files are absent from a CI checkout, so this is the run that can see them).
    extra = precision_input_problems(report, items=folded) if gated_schema else []
    return route_report(report, args.out, extra_problems=extra)


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
    # ADR-0005 A15 Pin 6 — the v2 inputs. Required together for a gated-subset manifest and
    # refused for an older one (`_cmd_fold`), so neither path can take a partial set.
    v2 = fold.add_argument_group("v2 inputs (ADR-0005 A15; gated-subset manifests only)")
    for arm in ("twin", "production"):
        v2.add_argument(f"--{arm}-stage1", help=f"{arm} Stage-1 logits (.npz); hashed only")
        v2.add_argument(f"--{arm}-stage2", help=f"{arm} Stage-2 logits JSON (identity + shards)")
        v2.add_argument(f"--{arm}-payloads", help=f"{arm} Stage-2 payload manifest (power)")
    v2.add_argument("--temperature-source", help="the GATE-2 report T was read from")
    v2.add_argument("--host-screen", help="reports/p3/host_screen.json")
    v2.add_argument("--postmint", help="reports/p3/benchmark_v2_postmint_screen.json")
    v2.add_argument("--familiarity", help="per-item flank-familiarity scores (A15 Pin 7(b))")
    v2.add_argument(
        "--rule-knobs",
        help="the D3 locus knobs the tables were built under, e.g. "
        "threshold=0.5,min_span=50,gap_merge=10,min_distinct_elements=2,flank=50,"
        "min_order_margin=1",
    )
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
