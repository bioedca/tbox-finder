"""GATE-4 — the Stage-1 segmentation-quality gate (P2-14; PRD §2.3/§12, ADR-0004 D6 + A6).

**The gated statistic** is the *minimum*, over the three core elements {Stem I, Specifier,
Antiterminator}, of each element's **per-nucleotide-class F1** — a homogeneous,
commensurable unit, so there is deliberately **no cross-unit mean** — against the
recalibratable binding default **>= 0.80** (ADR-0004 D6, floor single-sourced from
``power.GATE4_F1_FLOOR``).

**The population** is the one thing D6 did not, on its own, resolve for the checkpoint that
exists. D6 gates *"the in-distribution homology-clustered (genus-stratified) split"*, and
the shipped P2-10d′-b production checkpoint trained on **all 8,303** records of the ADR-0004
D5 fold (``exclude_selection_val=false``) — so no in-distribution population was withheld
from it, and `selection_val` (the P2-06a rung the sweep also selected on) is inside its
training stream. ADR-0004 **A6** resolves this: the graded population is
``window_dataset.load_gate4_eval_records`` — scheme-A ``test`` **inside** ``nested_train``,
1,201 records / 1,029 clusters / 154 genera / 15 orders — and the graded artifact is a
**GATE-4 eval twin** trained with ``exclude_gate4_eval=true``. The shipped checkpoint stays
the full-fold one (it saw strictly more data, so the twin's grade is a **conservative
proxy**); that is the two-run split PRD §10.1 requires the model card to disclose.

This module therefore **refuses to grade a checkpoint that cannot prove it is a twin**: the
gate reads the twin's own training report and re-derives ``fold_scope ==
"gate4_twin_train"`` and ``n_gate4_eval_excluded > 0`` from it, and binds the report to the
checkpoint bytes through the provenance sidecar's sha256. A gate that trusted a filename
here would certify a train-on-train number that looks exactly like a real one (§10.3).

**Reported alongside, explicitly NON-gated** (ADR-0004 D6): leave-one-order-out per-element
F1 macro-averaged across held-out orders (ADR-0005 D5 — micro would be dominated by the
~90 % Firmicutes corpus), Specifier exact-3-nt-codon detection, the 9-PDB cross-source
label-noise ceiling, boundary IoU per element, and all four non-core classes (Stem II incl.
the folded IIA/B-pk extent, Stem III, Terminator, Discriminator).

**Posterior.** Uncalibrated (ADR-0005 A11 Pin 4). P2-13's T = 0.9896 is a machinery
demonstration and is not shipped; grading the tempered posterior would be a later signed
decision, not a drift.

**Operator.** Predictions are reconciled by the frozen ADR-0005 D3 + A3 operator via
``infer.scan.scan_encoded_windows`` — the same loop the scanner deploys — so boundary IoU
is not a 512-grid artifact (PRD §6, the reason P2-03 blocks this step).

Entry: ``python -m tbox_finder.eval.gate4 run`` → :func:`compute_gate4`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder import metrics as M
from tbox_finder.labels import CLASS_INDEX, CLASS_ORDER, CORE_ELEMENTS
from tbox_finder.power import GATE4_F1_FLOOR

SCHEMA_VERSION = "1"
STEP = "P2-14"
GENERATED_BY = "src/tbox_finder/eval/gate4.py"
ADR = (
    "ADR-0004 D6 (gated statistic + floor) + A6 (graded population, eval twin, two-run "
    "split); ADR-0005 D3+A3 (reconciliation operator, consumed); ADR-0005 D5 (block-level "
    "resampling + macro-over-orders); ADR-0005 A11 Pin 4 (graded on the uncalibrated "
    "posterior)"
)
ENV_LOCK = "envs/ml-dna.conda-lock.yml"

DEFAULT_REPORT = Path("reports/p2/gate4.json")
DEFAULT_CHECKPOINT = Path("data/processed/checkpoints/stage1_gate4_twin/stage1.pt")
DEFAULT_TRAIN_REPORT = Path("reports/p2/train_stage1_gate4_twin.json")
DEFAULT_CHECKPOINT_PROVENANCE = Path("data/processed/checkpoints/stage1_gate4_twin/provenance.json")
#: The P0-21 crystal extents. The only in-repo home of the 9 depositions; nothing in
#: ``src/`` owned them before this step, and the ceiling was named in ``power.py`` but never
#: computed.
DEFAULT_PDB_FIXTURE = Path("tests/fixtures/pdb_element_extents/pdb_element_extents.json")

#: The frozen scan/eval geometry (ADR-0005 D3 + A3) — held, never swept here.
WINDOW_NT = 1024
STRIDE_NT = 512
DEFAULT_BATCH_SIZE = 8
#: Block-bootstrap replicates (ADR-0005 D5), matching the sweep's shipped value so the
#: GATE-4 CI and the selection CIs are the same estimator at the same resolution.
DEFAULT_N_BOOT = 2000

#: The posterior GATE-4 is graded on (ADR-0005 A11 Pin 4). Frozen here, no CLI override.
POSTERIOR = "uncalibrated"
#: The training fold scope a gradeable checkpoint must have been trained under (A6).
REQUIRED_TWIN_FOLD_SCOPE = "gate4_twin_train"

#: The four classes D6 excludes from the gate and reports per-class anyway. Derived from
#: the pinned class order minus background minus the core, so a class added upstream lands
#: here automatically instead of vanishing from the report.
NON_CORE_CLASSES: tuple[str, ...] = tuple(
    c for c in CLASS_ORDER if c not in CORE_ELEMENTS and c != "background"
)


# ══════════════════════════════════════════════════════════════════════════════════════
# Scoring — the promoted tile → forward → reconcile → confusions loop
# ══════════════════════════════════════════════════════════════════════════════════════
def score_records(
    model: Any,
    device: Any,
    records: Sequence[Any],
    *,
    block_key: Callable[[Any], Any],
    window: int = WINDOW_NT,
    stride: int = STRIDE_NT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    collect_positions: bool = True,
    collect_probs: bool = True,
) -> dict[str, Any]:
    """Score records under the **deployed** reconciliation operator; group by ``block_key``.

    The single tile → forward → reconcile → per-block-confusion loop, promoted here out of
    ``train_stage1.evaluate_selection_val`` (which now delegates to it). Two callers
    forking this loop is how the arithmetic a config is *selected* under drifts from the
    arithmetic it is *graded* and *deployed* under — the property
    ``infer.scan.scan_encoded_windows`` already exists to protect, one level down.

    ``block_key`` is the resampling unit: ``cluster_id`` for the in-distribution gate
    (ADR-0005 D5's homology-cluster block), the taxonomic **order** for the leave-one-order-
    out read (D5's macro-over-held-out-orders half).

    ``collect_positions``/``collect_probs`` exist for size, not taste: the LOO population is
    ~21.0M nucleotides, where pooled Python position lists and an ``(n, 8)`` posterior are
    hundreds of MB and buy nothing — every quantity that read needs is exact from the
    ``(8, 3)`` confusion vectors. Switching them off changes *what is retained*, never how a
    position is scored.

    ⚠ ``model.eval()`` + ``torch.no_grad()``, restoring the observed mode (the
    ``evaluate_selection_val`` contract, kept verbatim: dropout is a swept axis, so scoring
    with it live would add noise to the comparison).
    """
    import torch

    from tbox_finder.data.window_dataset import context_labels, encode_eval_window, tile_windows
    from tbox_finder.infer.scan import scan_encoded_windows
    from tbox_finder.train.train_stage1 import NUM_CLASSES, class_confusions

    was_training = bool(getattr(model, "training", False))
    model.eval()
    per_block: dict[Any, list[Any]] = {}
    per_record: list[tuple[np.ndarray, np.ndarray]] = []
    prob_chunks: list[np.ndarray] = []
    n_windows = 0
    n_positions = 0
    try:
        with torch.no_grad():
            for rec in records:
                seq_len = len(rec.context_seq)
                starts = tile_windows(seq_len, window=window, stride=stride)
                # The STRICT encoder: every loader feeding this function filters
                # `len(context) >= window`, so an overrun would invent a boundary mid-context
                # rather than find a contig end (PRD §6).
                ids = np.stack([encode_eval_window(rec, s, window=window) for s in starts])
                out = scan_encoded_windows(
                    model, ids, starts, seq_len, device=device, batch_size=batch_size
                )
                n_windows += int(out.n_windows)
                y_true = context_labels(rec)
                y_pred = np.asarray(out.prediction)
                n_positions += int(len(y_true))
                per_block.setdefault(block_key(rec), []).append(
                    class_confusions(y_true, y_pred, n_classes=NUM_CLASSES)
                )
                if collect_positions:
                    per_record.append((y_true, y_pred))
                if collect_probs:
                    prob_chunks.append(np.exp(np.asarray(out.log_probs)))
    finally:
        if was_training:
            model.train()

    return {
        "per_block": per_block,
        "per_record": per_record if collect_positions else None,
        "probs": np.concatenate(prob_chunks, axis=0) if collect_probs and prob_chunks else None,
        "n_windows": n_windows,
        "n_positions": n_positions,
    }


def same_number(a: float | None, b: float | None) -> bool:
    """True iff two route-derived scores are the same number, **NaN included**.

    ``nan == nan`` is False and ``nan`` is the honest value for an element no position
    exercises, so a naive equality check would read two agreeing routes as disagreeing
    exactly on the undefined elements — the case a consistency check most needs to pass
    quietly. ``None`` (an absent score) matches only ``None``.
    """
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def cross_route_disagreements(
    *,
    pooled_f1: Mapping[str, float],
    pooled_iou: Mapping[str, float],
    confusion_f1: Mapping[str, float],
    confusion_iou: Mapping[str, float],
    pooled_min_f1: float,
    confusion_min_f1: float,
) -> list[str]:
    """Name every score the two exact routes disagree on — empty when they agree.

    Each reported quantity has TWO routes in this file: the pooled ``metrics`` kernels walking
    every position, and the ``*_from_confusion`` re-association of the per-block ``(8, 3)``
    sufficient statistics the block bootstrap and the LOO read run on. The identity is exact,
    not an approximation, so any disagreement is a defect in one of them — and a caller that
    does not check would ship a report carrying ``gated.min_f1`` (pooled) and
    ``gated.block_bootstrap_ci.point`` (confusions) as two different values of one statistic.

    Pure and total: no model, no I/O, NaN-safe (see :func:`same_number`), so the check itself
    is testable rather than reachable only through a GPU forward pass.
    """
    return [
        f"{label}: pooled={pooled!r} confusions={via!r}"
        for label, pooled, via in (
            *((f"F1[{n}]", pooled_f1[n], confusion_f1[n]) for n in CLASS_ORDER),
            *((f"IoU[{n}]", pooled_iou[n], confusion_iou[n]) for n in CLASS_ORDER),
            ("gate4.min_f1", pooled_min_f1, confusion_min_f1),
        )
        if not same_number(pooled, via)
    ]


def summed_confusions(blocks: Sequence[Any]) -> Any:
    """Sum a list of ``(8, 3)`` confusion vectors into one. ``None`` on an empty list —
    never a zero matrix, which would score every class as an honest NaN and hide that
    nothing was measured at all."""
    if not len(blocks):
        return None
    return np.sum(np.stack([np.asarray(b) for b in blocks]), axis=0)


def f1_by_class_from_confusion(total: Any) -> dict[str, float]:
    """Per-class per-nt F1 from one summed ``(8, 3)`` confusion vector.

    Exactly ``metrics.per_nt_class_f1`` for every class — the confusion route is a
    re-association of the same counts, not an approximation — computed through the pinned
    ``metrics.prf`` kernel so there is no second F1 formula in the tree.
    """
    return {name: M.prf(*(int(x) for x in total[CLASS_INDEX[name]]))[2] for name in CLASS_ORDER}


def iou_by_class_from_confusion(total: Any) -> dict[str, float]:
    """Per-class extent IoU from one summed ``(8, 3)`` confusion vector
    (``metrics.iou_from_confusion``; identical to ``metrics.element_extent_iou``)."""
    return {
        name: M.iou_from_confusion(*(int(x) for x in total[CLASS_INDEX[name]]))
        for name in CLASS_ORDER
    }


def per_block_element_f1(per_block: Mapping[Any, Sequence[Any]]) -> dict[str, dict[str, float]]:
    """``block -> {class: per-nt F1}``, each from that block's own summed confusions."""
    out: dict[str, dict[str, float]] = {}
    for key in sorted(per_block, key=str):
        total = summed_confusions(per_block[key])
        if total is None:  # pragma: no cover — a key exists only because a record made it
            continue
        out[str(key)] = f1_by_class_from_confusion(total)
    return out


def macro_over_blocks(
    per_block_f1: Mapping[str, Mapping[str, float]], element: str
) -> dict[str, Any]:
    """Macro-average one element's per-nt F1 **across blocks** (ADR-0005 D5).

    Blocks whose F1 is undefined (the element absent from both truth and prediction in that
    block) are **excluded and counted**, not folded in as 0.0: an order that contains no
    Terminator has no Terminator F1, and averaging a fabricated 0 would report the corpus's
    class composition as a model failure. ``macro`` is ``None`` when no block defined it.
    """
    values = [
        float(v[element])
        for v in per_block_f1.values()
        if element in v and not math.isnan(float(v[element]))
    ]
    return {
        "macro": M.macro_average(values) if values else None,
        "n_blocks_defined": len(values),
        "n_blocks_undefined": len(per_block_f1) - len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# The 9-PDB cross-source label-noise ceiling (reported, non-gated; ADR-0004 D6 δ-governance)
# ══════════════════════════════════════════════════════════════════════════════════════
def pdb_label_noise_ceiling(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive whether a numeric cross-source ceiling ``C`` is estimable from the 9 PDB
    depositions, and therefore whether D6's δ-recalibration path is open.

    D6 permits recalibrating the 0.80 floor **only** if ``C`` demonstrates that the
    cross-source annotation itself caps achievable per-nt F1 below the floor. ``C`` is a
    per-nucleotide F1 between two label sources, so it can only be computed over classes
    that *both* sources resolve to residue extents. This re-derives that from the fixture's
    own ``cross_source_resolved_classes`` map — counting depositions per class — rather than
    restating the fixture's prose conclusion, so a future deposition that resolves a core
    element flips ``estimable`` to True here on its own and re-opens the question.

    Returns the resolved-deposition count per class, which core elements are unresolved,
    ``estimable``, and the δ-governance consequence.
    """
    block = fixture.get("label_noise_ceiling")
    if not isinstance(block, Mapping):
        raise ValueError(
            "PDB fixture carries no `label_noise_ceiling` block — the ADR-0004 D6 ceiling "
            "cannot be re-derived, and asserting 'not estimable' without the evidence is "
            "the same fabrication as asserting a number (§10.3)"
        )
    resolved = block.get("cross_source_resolved_classes")
    if not isinstance(resolved, Mapping) or not resolved:
        raise ValueError("PDB fixture's `cross_source_resolved_classes` map is missing or empty")

    counts = {str(k): len(v or []) for k, v in resolved.items()}
    unresolved_core = sorted(name for name in CORE_ELEMENTS if counts.get(name, 0) == 0)
    entries = fixture.get("entries")
    n_entries = len(entries) if isinstance(entries, (list, dict)) else None
    estimable = not unresolved_core
    # The prose reports the SAME numbers the block above re-derives — the deposition count
    # from the fixture, the floor from `power.GATE4_F1_FLOOR`. Writing "0 of the 9" / "0.80"
    # as literals would have this sentence keep asserting 9 depositions and a 0.80 floor
    # after a tenth deposition lands or ADR-0004 re-signs δ, which is a fabricated number in
    # a shipped report (CLAUDE.md §10.3) and would undo the point of re-deriving `estimable`
    # from the evidence rather than restating the fixture's prose. CodeRabbit, r1.
    depositions = f"the {n_entries}" if n_entries is not None else "the depositions supplied"
    small_n = f"N<={n_entries}" if n_entries is not None else "small-N"
    return {
        "gated": False,
        "n_depositions": n_entries,
        "resolved_depositions_by_class": dict(sorted(counts.items())),
        "core_elements": list(CORE_ELEMENTS),
        "unresolved_core_elements": unresolved_core,
        "estimable": estimable,
        "ceiling_c": None,
        "floor_unchanged": GATE4_F1_FLOOR,
        "delta_governance": (
            "C is NOT estimable from these depositions: "
            + ", ".join(unresolved_core)
            + f" are resolved to a residue extent in 0 of {depositions}, so no cross-source "
            "per-nt F1 over the gated elements exists. D6's recalibration path (floor := a "
            f"documented function of C) is therefore CLOSED and the floor stays "
            f"{GATE4_F1_FLOOR} — the {small_n} crystal ceiling must not one-directionally "
            "lower the bar."
            if not estimable
            else (
                "every core element is resolved in >=1 deposition, so a numeric C is now "
                "estimable and D6's δ-recalibration path is open — this requires ADR-0004 "
                "re-sign-off before any floor changes (CLAUDE.md §7 item 2)."
            )
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════════════
def json_safe(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` so the written report is **valid JSON**.

    ``json.dumps`` emits bare ``NaN`` / ``Infinity`` tokens by default. Python's own
    ``json.loads`` reads them back, which is exactly why this goes unnoticed — but they are
    not JSON, and every other consumer of a public phase-exit artifact (``jq``, JavaScript,
    R's ``jsonlite``, Go, Rust) rejects the whole file.

    GATE-4 produces NaN **by design**, so a real run very likely writes one: an element no
    position exercises scores NaN so an unmeasurable element cannot silently certify the gate,
    ``metrics.iou_from_confusion`` is NaN on an empty union (Stem III / Discriminator can
    easily be absent from 1,201 records), and ``block_bootstrap_ci`` is NaN below two blocks.

    Nothing re-validates the *written* file — :func:`gate4_problems` runs on the in-memory
    report inside :func:`compute_gate4`, before this — so ``NaN → null`` costs the gate
    nothing and buys a parseable deliverable. The ``eval/backbone_throughput.py`` idiom.
    CodeRabbit, r3.
    """
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):  # np.float64 subclasses float, so this catches it too
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.generic):  # np.int64 / np.bool_ do NOT subclass int / bool
        return json_safe(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    # Anything else is returned untouched so `json.dumps` raises on it. Stringifying an
    # unexpected type would put a plausible-looking value in a shipped report (§10.3); a
    # TypeError names the key instead.
    return obj


def _portable_path(path: str | Path) -> str:
    return str(Path(path).as_posix())


def file_sha256(path: str | Path) -> str:
    """Streamed sha256 of a file (the checkpoint is ~31 MB; do not slurp it)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_output_sha256(
    checkpoint_provenance: Mapping[str, Any] | None,
    checkpoint_path: str | Path | None,
) -> str | None:
    """The sidecar's recorded digest **for the graded checkpoint** — ``None`` if unresolvable.

    ``provenance.sha256_paths`` keys ``outputs`` by the path as the producing rule referenced
    it, so a sidecar recording more than one artifact must be **looked up, not iterated**:
    taking the first value binds ``checkpoint_identity_verified`` to whichever artifact
    happens to come first in insertion order. Today every training sidecar records exactly
    one output, so this is latent — and the existing clause makes a mismatch fail LOUDLY
    rather than pass, so the live cost would be a spurious refusal after a ~16 GPU-h run, not
    a fabricated pass. But a clause whose evidence is "some output of some sidecar" is not
    evidence of the thing it names. Total by contract, like its caller: an unresolvable or
    ambiguous lookup returns ``None`` and :func:`gate4_problems` does the judging.
    CodeRabbit, r3.
    """
    outputs = (
        checkpoint_provenance.get("outputs") if isinstance(checkpoint_provenance, Mapping) else None
    )
    if not isinstance(outputs, Mapping) or not outputs:
        return None
    if checkpoint_path is not None:
        want = Path(str(checkpoint_path))
        for key, value in outputs.items():
            if Path(str(key)) == want:
                return str(value)
        by_name = [str(v) for k, v in outputs.items() if Path(str(k)).name == want.name]
        if len(by_name) > 1:
            return None  # ambiguous basename — refuse rather than pick one
        if by_name:
            return by_name[0]
    # No path to match on (or a single-output sidecar spelled differently): the one entry is
    # unambiguous, and a wrong one still has to survive the sha256 comparison downstream.
    return str(next(iter(outputs.values()))) if len(outputs) == 1 else None


def twin_evidence(
    train_report: Mapping[str, Any],
    *,
    checkpoint_sha256: str | None = None,
    checkpoint_provenance: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract — never assert — the evidence that the graded checkpoint is a GATE-4 twin.

    Reads the twin's **own** training report for the fold scope and the measured exclusion
    count, and (when a provenance sidecar is supplied) binds that report to the checkpoint
    *bytes* by sha256. Every field is copied out of the artifacts; the clauses in
    :func:`gate4_problems` do the judging, so a missing field surfaces as a failed clause
    rather than a defaulted pass ([[gate-clauses-need-re-derivation]]).
    """
    scope = train_report.get("class_counts_scope")
    scope = scope if isinstance(scope, Mapping) else {}
    cfg = train_report.get("diagnostics")
    cfg = cfg.get("config") if isinstance(cfg, Mapping) else None
    cfg = cfg if isinstance(cfg, Mapping) else {}
    prov_sha = provenance_output_sha256(checkpoint_provenance, checkpoint_path)
    return {
        "fold_scope": scope.get("fold_scope"),
        "exclude_gate4_eval_config": cfg.get("exclude_gate4_eval"),
        "exclude_gate4_eval_measured": scope.get("exclude_gate4_eval"),
        "n_gate4_eval_excluded": scope.get("n_gate4_eval_excluded"),
        "n_gate4_eval_clusters": scope.get("n_gate4_eval_clusters"),
        "n_training_records": scope.get("n_records"),
        "label_source": scope.get("label_source"),
        "checkpoint_sha256": checkpoint_sha256,
        "provenance_output_sha256": prov_sha,
        "checkpoint_identity_verified": bool(
            checkpoint_sha256 is not None and prov_sha is not None and checkpoint_sha256 == prov_sha
        ),
    }


#: The SHIPPED P2-10d′-b production checkpoint's training-fold size. Unlike every other
#: number in the disclosures this one describes a **different artifact** — it is recorded in
#: `reports/p2/train_stage1_production.json` and pinned by ADR-0004 A6 — so no key of *this*
#: run re-derives it and it is a named constant rather than a derivation.
SHIPPED_CHECKPOINT_N_TRAIN = 8303


def disclosures(
    *,
    twin: Mapping[str, Any],
    scope: Mapping[str, Any],
    non_gated: Mapping[str, Any],
) -> list[str]:
    """The PRD §10.1 / ADR-0004 A6 disclosures, **derived from the evidence they describe**.

    These sentences ship inside ``gate4.json`` and are the only place a reader learns that the
    graded artifact is not the shipped one. Writing the twin's record count, the population's
    phylum share, the deposition count and the floor as string literals — as this block first
    did — makes them keep asserting one run's numbers under another's the moment the fold is
    re-cut, the fixture grows, or ADR-0004 re-signs δ, and a stale number in a shipped report
    reads exactly like a live one (CLAUDE.md §10.3). It also contradicted ``build_report``'s
    own contract, *"pure: every number is passed in already measured"*. Each sentence now
    degrades to a named refusal when its evidence is absent rather than inventing a value.
    CodeRabbit, r2 — the same finding as r1's δ-governance sentence, one block over.
    """
    n_twin_train = twin.get("n_training_records")
    twin_size = (
        f"The twin trained on {n_twin_train} records — strictly fewer — so this number is a "
        "CONSERVATIVE proxy for the shipped model, not a measurement of it."
        if isinstance(n_twin_train, int)
        else (
            "The twin's training-fold size is NOT re-derivable from this report "
            "(twin.n_training_records is missing), so the CONSERVATIVE direction of the "
            "two-run split cannot be checked from this sentence — read the twin's own "
            "training report."
        )
    )

    phylum_counts = scope.get("phylum_counts")
    n_scope = scope.get("n_records")
    if (
        isinstance(phylum_counts, Mapping)
        and phylum_counts
        and isinstance(n_scope, int)
        and n_scope
    ):
        # Ties broken by name so the sentence is deterministic across runs (CLAUDE.md §8.3).
        top_phylum, top_n = max(phylum_counts.items(), key=lambda kv: (kv[1], str(kv[0])))
        composition = (
            f"The graded population is {100.0 * top_n / n_scope:.1f}% {top_phylum} "
            f"({top_n}/{n_scope} records), which is the corpus composition left after the "
            "leave-one-order-out holdout is removed, not a sampling choice. The point "
            "estimate is pooled MICRO over positions; the cluster-blocked CI is the D5 "
            "estimator."
        )
    else:
        composition = (
            "The graded population's phylum composition is NOT re-derivable from this report "
            "(scope.phylum_counts is missing or empty) — do not read a composition claim off "
            "this sentence. The point estimate is pooled MICRO over positions; the "
            "cluster-blocked CI is the D5 estimator."
        )

    ceiling = non_gated.get("pdb_label_noise_ceiling")
    ceiling = ceiling if isinstance(ceiling, Mapping) else {}
    unresolved = ceiling.get("unresolved_core_elements") or []
    n_dep = ceiling.get("n_depositions")
    depositions = f"{n_dep} depositions" if n_dep is not None else "the depositions supplied"
    if ceiling.get("estimable") is False and unresolved:
        ceiling_note = (
            f"The cross-source label-noise ceiling C is NOT estimable ({', '.join(unresolved)} "
            f"resolve to a residue extent in 0 of {depositions}), so D6's δ-recalibration path "
            f"is closed and the {GATE4_F1_FLOOR} floor stands as written."
        )
    elif ceiling.get("estimable") is True:
        ceiling_note = (
            f"The cross-source label-noise ceiling C is now ESTIMABLE from {depositions}, so "
            "D6's δ-recalibration path has re-opened. The floor here is still "
            f"{GATE4_F1_FLOOR}: changing it requires ADR-0004 re-sign-off (CLAUDE.md §7 "
            "item 2), never this report."
        )
    else:
        ceiling_note = (
            "The cross-source label-noise ceiling is NOT reported here "
            "(reported_non_gated.pdb_label_noise_ceiling is missing or carries no verdict), so "
            f"nothing in this run bears on D6's δ-recalibration path and the {GATE4_F1_FLOOR} "
            "floor stands as written."
        )

    return [
        "GATE-4 grades the eval TWIN, not the shipped checkpoint: the shipped P2-10d'-b "
        f"production checkpoint trained on all {SHIPPED_CHECKPOINT_N_TRAIN} D5-fold records, "
        "this population included, so it has no in-distribution held-out population at all "
        f"(ADR-0004 A6). {twin_size} PRD §10.1's two-run split disclosure.",
        composition,
        "The leave-one-order-out read is REPORTED, NEVER GATED here (ADR-0004 D6 gates the "
        "in-distribution reference). It is the population PRD §12:241 reports as the "
        "generalization headline and PRD §2.3 grades at GATE-1 / P4.",
        "Graded on the UNCALIBRATED posterior (ADR-0005 A11 Pin 4); P2-13's T = 0.9896 is a "
        "machinery demonstration and is not shipped.",
        ceiling_note,
    ]


def build_report(
    *,
    checkpoint: str | Path,
    twin: Mapping[str, Any],
    scope: Mapping[str, Any],
    gated: Mapping[str, Any],
    non_gated: Mapping[str, Any],
    loo: Mapping[str, Any] | None,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the GATE-4 report. Pure: every number is passed in already measured."""
    from tbox_finder.infer.reconcile import OPERATOR

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "env_lock": ENV_LOCK,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "is_science": True,
        "gated": True,
        "checkpoint": _portable_path(checkpoint),
        "posterior": POSTERIOR,
        "operator": OPERATOR,
        "geometry": dict(geometry),
        "twin": dict(twin),
        "scope": dict(scope),
        "gate": dict(gated),
        "reported_non_gated": dict(non_gated),
        "loo_holdout": dict(loo) if loo is not None else None,
        "disclosures": disclosures(twin=twin, scope=scope, non_gated=non_gated),
    }


def gate4_problems(report: Mapping[str, Any]) -> list[str]:
    """Re-derive every GATE-4 clause from ``report``; ``[]`` when clean.

    Total and evidence-guarded. The clauses that matter most are the ones an absent-evidence
    bug would fabricate TRUE — a gate read off a *missing* twin block, or a ``passes`` copied
    rather than recomputed — so each is guarded on the evidence being present and the gated
    statistic is **recomputed from the per-element F1s** rather than trusted
    ([[gate-clauses-need-re-derivation]]).
    """
    # Local, like every other `window_dataset` use here: the module pulls pandas, and this
    # validator is imported by tests that must run without it.
    from tbox_finder.data.window_dataset import (
        A6_N_POSITIONS,
        A6_N_TWIN_EXCLUDED_CLUSTERS,
        A6_N_TWIN_EXCLUDED_RECORDS,
        A6_N_TWIN_TRAIN_RECORDS,
    )

    problems: list[str] = []

    if report.get("posterior") != POSTERIOR:
        problems.append(
            f"posterior: must be {POSTERIOR!r} (ADR-0005 A11 Pin 4), got "
            f"{report.get('posterior')!r} — grading a tempered posterior needs a signed decision"
        )
    try:
        from tbox_finder.infer.reconcile import OPERATOR
    except Exception:  # pragma: no cover - import guard only
        OPERATOR = None
    if OPERATOR is not None and report.get("operator") != OPERATOR:
        problems.append(
            f"operator: must be the pinned ADR-0005 D3+A3 operator {OPERATOR!r}, got "
            f"{report.get('operator')!r} — boundary IoU on unreconciled tiles is a "
            "512-grid artifact (PRD §6)"
        )

    # ── The twin clauses: is this checkpoint even gradeable on this fold? ──
    twin = report.get("twin")
    if not isinstance(twin, Mapping) or not twin:
        problems.append(
            "twin: block missing — without the training report's own fold scope there is no "
            "evidence the graded checkpoint withheld the graded fold, and a train-on-train "
            "GATE-4 number is indistinguishable from a real one"
        )
    else:
        if twin.get("fold_scope") != REQUIRED_TWIN_FOLD_SCOPE:
            problems.append(
                f"twin.fold_scope = {twin.get('fold_scope')!r}, must be "
                f"{REQUIRED_TWIN_FOLD_SCOPE!r} — the shipped full-fold checkpoint trained on "
                "this population (ADR-0004 A6)"
            )
        if twin.get("exclude_gate4_eval_measured") is not True:
            problems.append(
                "twin.exclude_gate4_eval_measured is not True — the loader did not report "
                "withholding the graded fold"
            )
        n_ex = twin.get("n_gate4_eval_excluded")
        if not isinstance(n_ex, int) or isinstance(n_ex, bool) or n_ex <= 0:
            problems.append(
                f"twin.n_gate4_eval_excluded: must be a positive int, got {n_ex!r} — a flag "
                "that was set but excluded NOTHING is exactly how this clause fabricates TRUE"
            )
        # …and a positive count cannot tell 1,204 from 900. ADR-0004 A6 pins the carve by
        # measurement (8,303 → 7,099, withholding 1,204 records in 1,031 clusters), so the
        # twin's own numbers are checked against it, not merely against zero.
        for key, pinned in (
            ("n_gate4_eval_excluded", A6_N_TWIN_EXCLUDED_RECORDS),
            ("n_gate4_eval_clusters", A6_N_TWIN_EXCLUDED_CLUSTERS),
            ("n_training_records", A6_N_TWIN_TRAIN_RECORDS),
        ):
            got = twin.get(key)
            if got != pinned:
                problems.append(
                    f"twin.{key} = {got!r}, but ADR-0004 A6 pins it at {pinned} — the twin "
                    "carved a different fold than the signed one, so its grade is not the "
                    "one D6 defines"
                )
        if twin.get("checkpoint_identity_verified") is not True:
            problems.append(
                "twin.checkpoint_identity_verified is not True — the graded checkpoint's "
                "sha256 does not match the provenance sidecar the training run wrote, so the "
                "report proving the fold was withheld may describe a DIFFERENT checkpoint"
            )
        if twin.get("label_source") not in (None, "production"):
            problems.append(
                f"twin.label_source = {twin.get('label_source')!r} — GATE-4 grades the "
                "PRODUCTION scanner; the class-II-CM-naive ablation is graded only at P4 "
                "(PRD §10.1)"
            )

    # ── The graded population's own invariants ──
    scope = report.get("scope")
    if not isinstance(scope, Mapping) or not scope:
        problems.append("scope: block missing — the graded population is unidentifiable")
    else:
        from tbox_finder.data.window_dataset import gate4_eval_problems

        problems.extend(f"scope.{p}" for p in gate4_eval_problems(scope))

        # ── Completeness: a truncated run must never certify ──
        # `--max-records` / `--loo-max-records` / `--no-loo` are cost knobs, and every
        # clause above survives truncation unharmed: a 8-record prefix of a disjoint
        # population is still disjoint, still `gate4_eval`, still >= 2 blocks. So the
        # cheap smoke would assemble a report whose `passes` is computed over a
        # population that is not D6's, write it to the phase-exit path, and read there
        # exactly like a full grade (§10.3). These clauses are what make `build_report`'s
        # hardcoded `is_science: True` honest: an incomplete run raises in
        # `compute_gate4` and no file is written at all.
        if scope.get("full_fold") is not True:
            problems.append(
                f"scope.full_fold = {scope.get('full_fold')!r}, must be True — the minimum "
                "over a truncated population is not the ADR-0004 D6 statistic, and a report "
                "that certified it would sit at the phase-exit path indistinguishable from a "
                "full grade"
            )
        if scope.get("max_records") is not None:
            problems.append(
                f"scope.max_records = {scope.get('max_records')!r}, must be None — a "
                "record-capped run is a mechanics smoke, never a graded result"
            )

    # ── The gated statistic, RE-DERIVED ──
    gate = report.get("gate")
    if not isinstance(gate, Mapping) or not gate:
        return problems + ["gate: block missing — there is no gated statistic to check"]

    per_element = gate.get("per_element_f1")
    if not isinstance(per_element, Mapping) or not per_element:
        problems.append("gate.per_element_f1: block missing — min_f1 cannot be re-derived")
    else:
        missing = [e for e in CORE_ELEMENTS if e not in per_element]
        if missing:
            problems.append(
                f"gate.per_element_f1: missing core element(s) {missing} — the min is over "
                "all three or it is not the D6 statistic"
            )
        else:
            # `float(per_element[e])` raises on None / a dict / a non-numeric string, and this
            # function's contract is TOTAL — it returns problems, never raises. A validator
            # that dies reports nothing, which is strictly worse than one that reports the
            # malformed value, and it dies exactly on the corrupt artifacts it exists to
            # catch (the same defect fixed in P2-13's calibration validator). Non-numeric
            # per-element F1s become problems and the re-derivation is skipped rather than
            # run on coerced garbage. CodeRabbit, r2.
            vals: list[float] = []
            bad = []
            for e in CORE_ELEMENTS:
                v = per_element[e]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    bad.append(f"{e}={v!r}")
                else:
                    vals.append(float(v))
            if bad:
                problems.append(
                    "gate.per_element_f1: non-numeric core element(s) "
                    f"{', '.join(bad)} — min_f1 cannot be re-derived from them"
                )
            got = gate.get("min_f1")
            if not isinstance(got, (int, float)) or isinstance(got, bool):
                problems.append(f"gate.min_f1: must be a number, got {got!r}")
            elif not bad:
                expected = float("nan") if any(math.isnan(v) for v in vals) else min(vals)
                if math.isnan(expected) != math.isnan(float(got)) or (
                    not math.isnan(expected) and abs(float(got) - expected) > 1e-12
                ):
                    per_core = dict(zip(CORE_ELEMENTS, vals, strict=True))
                    problems.append(
                        f"gate.min_f1 = {got!r} but the minimum over {per_core} is "
                        f"{expected!r} — the gated statistic is a MIN, not a mean (ADR-0004 D6)"
                    )
            floor = gate.get("floor")
            if floor != GATE4_F1_FLOOR:
                problems.append(
                    f"gate.floor = {floor!r}, must be power.GATE4_F1_FLOOR = {GATE4_F1_FLOOR} "
                    "— the floor is blinded-frozen at P0 and single-sourced (ADR-0004 D6/A2)"
                )
            elif isinstance(got, (int, float)) and not isinstance(got, bool):
                expected_pass = M.gate4_pass(float(got))
                if gate.get("passes") is not expected_pass:
                    problems.append(
                        f"gate.passes = {gate.get('passes')!r} but min_f1 {got!r} against the "
                        f"{GATE4_F1_FLOOR} floor re-derives {expected_pass!r}"
                    )

    n_positions = gate.get("n_positions")
    if not isinstance(n_positions, int) or isinstance(n_positions, bool) or n_positions <= 0:
        problems.append(f"gate.n_positions: must be a positive int, got {n_positions!r}")
    elif n_positions != A6_N_POSITIONS:
        problems.append(
            f"gate.n_positions = {n_positions}, but ADR-0004 A6 pins the graded population at "
            f"{A6_N_POSITIONS} nt — the same record count over a different nucleotide extent "
            "is a different measurement"
        )
    # The last cost knob that could certify. `--n-boot 50` yields a far noisier interval that
    # is not the ADR-0005 D5 estimator and reads, at the phase-exit path, exactly like the
    # full one; `--window`/`--stride` would likewise re-tile the reconciliation the frozen
    # ADR-0005 D3+A3 operator is defined over. Same class as the r5 truncation clauses.
    geometry = report.get("geometry")
    got_geometry = (
        (geometry.get("window_nt"), geometry.get("stride_nt"), geometry.get("n_boot"))
        if isinstance(geometry, Mapping)
        else None
    )
    if got_geometry != (WINDOW_NT, STRIDE_NT, DEFAULT_N_BOOT):
        problems.append(
            f"geometry = {geometry!r}, must be the frozen scan geometry at the shipped CI "
            f"resolution ({WINDOW_NT}/{STRIDE_NT}/{DEFAULT_N_BOOT}) — a reduced --n-boot is "
            "not the ADR-0005 D5 estimator, and a retiled window is not the D3 operator"
        )

    ci = gate.get("block_bootstrap_ci")
    if not isinstance(ci, Mapping) or not ci:
        problems.append("gate.block_bootstrap_ci: block missing (ADR-0005 D5)")
    elif isinstance(scope, Mapping) and ci.get("n_blocks") != scope.get("n_blocks"):
        problems.append(
            f"gate.block_bootstrap_ci.n_blocks = {ci.get('n_blocks')!r} but the scored "
            f"population has {scope.get('n_blocks')!r} blocks — the CI does not describe "
            "the eval"
        )

    # ── The reported, non-gated companions D6 requires ALONGSIDE the gate ──
    non_gated = report.get("reported_non_gated")
    if not isinstance(non_gated, Mapping) or not non_gated:
        problems.append("reported_non_gated: block missing — D6 requires them alongside the gate")
    else:
        for key in (
            "per_nt_f1_by_class",
            "boundary_iou_by_element",
            "non_core_classes",
            "specifier_exact_codon",
            "pdb_label_noise_ceiling",
        ):
            if not non_gated.get(key):
                problems.append(f"reported_non_gated.{key}: missing — D6 requires it reported")
        ceiling = non_gated.get("pdb_label_noise_ceiling")
        if isinstance(ceiling, Mapping) and ceiling.get("estimable") is True:
            problems.append(
                "reported_non_gated.pdb_label_noise_ceiling.estimable is True — a numeric C "
                "would re-open D6's δ-recalibration path, which needs ADR-0004 re-sign-off "
                "before this report may be read as a pass/fail against 0.80"
            )
        non_core = non_gated.get("non_core_classes")
        if isinstance(non_core, Mapping):
            leaked = sorted(set(non_core) & set(CORE_ELEMENTS))
            if leaked:
                problems.append(
                    f"reported_non_gated.non_core_classes contains core element(s) {leaked} — "
                    "the gate's own elements are not 'excluded from the gate' (D6)"
                )

    # The absence guard matters more than the value guard: `--no-loo` sets this block to
    # None, and a clause written as `isinstance(loo, Mapping) and loo and ...` is vacuously
    # TRUE exactly when the evidence D6 requires alongside the gate failed to materialise
    # ([[clauses-must-guard-emptiness]]).
    loo = report.get("loo_holdout")
    if not isinstance(loo, Mapping) or not loo:
        problems.append(
            "loo_holdout: block missing — D6 requires the leave-one-order-out read REPORTED "
            "alongside the gate, so a --no-loo run cannot certify"
        )
    else:
        if loo.get("gated") is not False:
            problems.append(
                "loo_holdout.gated must be False — the leave-one-order-out read is REPORTED at "
                "P2-14 and gated at GATE-1 / P4 (ADR-0004 D6; PRD §2.3)"
            )
        loo_scope = loo.get("scope")
        if not isinstance(loo_scope, Mapping) or loo_scope.get("full_fold") is not True:
            problems.append(
                "loo_holdout.scope.full_fold is not True — a --loo-max-records run reports a "
                "macro over a truncated order set, which is not the ADR-0005 D5 read"
            )
    return problems


# ══════════════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════════════
def compute_gate4(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    train_report_path: str | Path = DEFAULT_TRAIN_REPORT,
    checkpoint_provenance_path: str | Path = DEFAULT_CHECKPOINT_PROVENANCE,
    pdb_fixture_path: str | Path = DEFAULT_PDB_FIXTURE,
    window: int = WINDOW_NT,
    stride: int = STRIDE_NT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
    device: str | None = None,
    max_records: int | None = None,
    run_loo: bool = True,
    loo_max_records: int | None = None,
) -> dict[str, Any]:
    """Grade GATE-4 on the eval twin and assemble the full P2-14 report.

    Raises before any forward pass if the graded population fails its own invariants, and
    raises after assembly if any clause in :func:`gate4_problems` fails — a report that
    cannot certify itself is never written (the ``train_stage1`` contract).
    """
    from tbox_finder.data.window_dataset import (
        gate4_eval_problems,
        load_gate4_eval_records,
        load_loo_holdout_records,
        record_order,
    )
    from tbox_finder.infer.scan import load_stage1_checkpoint
    from tbox_finder.train.train_stage1 import min_core_f1_from_confusions

    records, scope = load_gate4_eval_records(window=window)
    problems = gate4_eval_problems(scope)
    if problems:
        raise ValueError(
            "the GATE-4 graded population failed its own invariants:\n  " + "\n  ".join(problems)
        )
    if max_records is not None:
        records = records[:max_records]
        scope = {
            **scope,
            "n_records_fold": scope["n_records"],
            "n_blocks_fold": scope["n_blocks"],
            "n_records": len(records),
            "n_blocks": len({r.cluster_id for r in records}),
            "full_fold": False,
            "max_records": int(max_records),
        }
    else:
        scope = {**scope, "full_fold": True, "max_records": None}

    train_report = json.loads(Path(train_report_path).read_text())
    prov_path = Path(checkpoint_provenance_path)
    provenance = json.loads(prov_path.read_text()) if prov_path.exists() else None
    twin = twin_evidence(
        train_report,
        checkpoint_sha256=file_sha256(checkpoint),
        checkpoint_provenance=provenance,
        checkpoint_path=checkpoint,
    )

    # `load_stage1_checkpoint` returns the segmenter itself, already `.eval()`, having
    # refused any architecture guess that does not fit the saved keys (scan.py:308).
    model = load_stage1_checkpoint(checkpoint, device=device)
    dev = next(model.parameters()).device

    scored = score_records(
        model,
        dev,
        records,
        block_key=lambda r: r.cluster_id,
        window=window,
        stride=stride,
        batch_size=batch_size,
    )
    y_true = [int(x) for arrays in scored["per_record"] for x in arrays[0]]
    y_pred = [int(x) for arrays in scored["per_record"] for x in arrays[1]]

    gate4 = M.gate4_core_min_f1(y_true, y_pred)
    probs = scored["probs"]
    auprc = {
        name: M.average_precision([1 if t == i else 0 for t in y_true], probs[:, i].tolist())
        for i, name in enumerate(CLASS_ORDER)
    }
    blocks = [v for _, v in sorted(scored["per_block"].items())]
    ci = M.block_bootstrap_ci(blocks, min_core_f1_from_confusions, n_boot=n_boot, seed=seed)
    if ci["n_blocks"] != scope["n_blocks"]:
        raise ValueError(
            f"block-count disagreement: scope.n_blocks={scope['n_blocks']} but the bootstrap "
            f"resampled {ci['n_blocks']} — the scope does not describe the eval"
        )

    per_class_f1 = M.per_nt_f1_by_class(y_true, y_pred)
    iou = M.boundary_iou_by_element(y_true, y_pred)

    # ── Cross-route consistency: pooled positions vs summed block confusions ──────────
    # Every number above has TWO exact routes in this file: the pooled `metrics` kernels
    # walking ~3.0M positions, and the `*_from_confusion` re-association of the per-block
    # `(8, 3)` sufficient statistics that the block bootstrap and the LOO read run on. The
    # identity is exact, not an approximation — so a disagreement is a defect in one of the
    # two, and until now nothing here would have seen it: the report would simply carry
    # `gated.min_f1` (pooled) and `gated.block_bootstrap_ci.point` (confusions) as two
    # contradictory headline numbers for the same statistic. `test_confusion_route_
    # reproduces_the_pinned_f1_and_iou_kernels` pins the identity on a fixture; this pins it
    # on the run that actually ships (CLAUDE.md §10.3 — re-derive, never trust). CodeRabbit, r1.
    total = summed_confusions([c for blks in scored["per_block"].values() for c in blks])
    if total is None:
        raise ValueError(
            f"no per-block confusions accumulated over {len(records)} records — the gated "
            "statistic would be computed on a population the bootstrap never saw"
        )
    disagreements = cross_route_disagreements(
        pooled_f1=per_class_f1,
        pooled_iou=iou,
        confusion_f1=f1_by_class_from_confusion(total),
        confusion_iou=iou_by_class_from_confusion(total),
        pooled_min_f1=gate4["min_f1"],
        confusion_min_f1=ci["point"],
    )
    if disagreements:
        raise ValueError(
            "the pooled-position and block-confusion metric routes disagree, so the report "
            "would carry two different values for the same statistic:\n  "
            + "\n  ".join(disagreements)
        )

    codon = M.specifier_exact_codon_detection(
        [(t.tolist(), p.tolist()) for t, p in scored["per_record"]]
    )
    ceiling = pdb_label_noise_ceiling(json.loads(Path(pdb_fixture_path).read_text()))

    gated = {
        "population": scope["fold_scope"],
        "statistic": "min over {Stem_I, Specifier, Antiterminator_Tbox_seq} of per-nt class F1",
        "per_element_f1": gate4["per_element_f1"],
        "min_f1": gate4["min_f1"],
        "floor": gate4["floor"],
        "passes": gate4["passes"],
        "block_bootstrap_ci": ci,
        "n_records": len(records),
        "n_positions": scored["n_positions"],
        "n_windows_forwarded": scored["n_windows"],
    }
    non_gated = {
        "per_nt_f1_by_class": per_class_f1,
        "boundary_iou_by_element": iou,
        "auprc_by_class": auprc,
        "macro_f1": M.macro_f1(y_true, y_pred),
        "micro_f1": M.micro_f1(y_true, y_pred),
        "non_core_classes": {
            name: {"per_nt_f1": per_class_f1[name], "boundary_iou": iou[name]}
            for name in NON_CORE_CLASSES
        },
        "specifier_exact_codon": codon,
        "pdb_label_noise_ceiling": ceiling,
    }

    loo_block: dict[str, Any] | None = None
    if run_loo:
        loo_records, loo_scope = load_loo_holdout_records(window=window)
        if loo_max_records is not None:
            loo_records = loo_records[:loo_max_records]
        loo_scored = score_records(
            model,
            dev,
            loo_records,
            block_key=record_order,
            window=window,
            stride=stride,
            batch_size=batch_size,
            # ~21.0M nucleotides: confusion vectors carry every quantity this read needs.
            collect_positions=False,
            collect_probs=False,
        )
        by_order = per_block_element_f1(loo_scored["per_block"])
        total = summed_confusions(
            [c for blocks_ in loo_scored["per_block"].values() for c in blocks_]
        )
        loo_block = {
            "gated": False,
            "scope": {
                **loo_scope,
                "n_records_scored": len(loo_records),
                "full_fold": loo_max_records is None,
            },
            "block_unit": "resolved_order",
            "macro_per_element_f1": {
                element: macro_over_blocks(by_order, element) for element in CORE_ELEMENTS
            },
            "macro_non_core_f1": {
                element: macro_over_blocks(by_order, element) for element in NON_CORE_CLASSES
            },
            "per_order_f1": by_order,
            "per_order_min_core_f1": {
                key: min_core_f1_from_confusions(blocks_)
                for key, blocks_ in sorted(
                    loo_scored["per_block"].items(), key=lambda kv: str(kv[0])
                )
            },
            "pooled_micro_per_nt_f1": (
                f1_by_class_from_confusion(total) if total is not None else None
            ),
            "pooled_micro_boundary_iou": (
                iou_by_class_from_confusion(total) if total is not None else None
            ),
            "n_positions": loo_scored["n_positions"],
            "n_windows_forwarded": loo_scored["n_windows"],
            "note": (
                "REPORTED, NOT GATED at P2-14 (ADR-0004 D6 gates the in-distribution "
                "reference). Macro across held-out orders per ADR-0005 D5; the pooled micro "
                "row is reported beside it because the corpus is ~90% Firmicutes and the two "
                "differ for that reason, not because of a model property."
            ),
        }

    report = build_report(
        checkpoint=checkpoint,
        twin=twin,
        scope=scope,
        gated=gated,
        non_gated=non_gated,
        loo=loo_block,
        geometry={"window_nt": int(window), "stride_nt": int(stride), "n_boot": int(n_boot)},
    )
    failures = gate4_problems(report)
    if failures:
        raise ValueError("GATE-4 report failed its own clauses:\n  " + "\n  ".join(failures))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbox_finder.eval.gate4", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="grade GATE-4 on the eval twin and write the report")
    run.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    run.add_argument("--train-report", default=str(DEFAULT_TRAIN_REPORT))
    run.add_argument("--checkpoint-provenance", default=str(DEFAULT_CHECKPOINT_PROVENANCE))
    run.add_argument("--pdb-fixture", default=str(DEFAULT_PDB_FIXTURE))
    run.add_argument("--out", default=str(DEFAULT_REPORT))
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", default=None)
    run.add_argument("--max-records", type=int, default=None)
    run.add_argument("--loo-max-records", type=int, default=None)
    run.add_argument(
        "--no-loo",
        action="store_true",
        help="skip the non-gated leave-one-order-out read (the ~21.0M-nt leg)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compute_gate4(
        checkpoint=args.checkpoint,
        train_report_path=args.train_report,
        checkpoint_provenance_path=args.checkpoint_provenance,
        pdb_fixture_path=args.pdb_fixture,
        batch_size=args.batch_size,
        n_boot=args.n_boot,
        seed=args.seed,
        device=args.device,
        max_records=args.max_records,
        run_loo=not args.no_loo,
        loo_max_records=args.loo_max_records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    # `allow_nan=False` AFTER `json_safe` is the belt, not the fix: the sanitiser has already
    # replaced every non-finite float, so this can only fire if it missed one — in which case
    # the run fails loudly instead of writing a file no non-Python reader can parse.
    tmp.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(out)
    gate = report["gate"]
    print(
        f"GATE-4 {'PASS' if gate['passes'] else 'FAIL'} "
        f"min_core_f1={gate['min_f1']:.6f} floor={gate['floor']} "
        f"({gate['n_records']} records / {report['scope']['n_blocks']} blocks) -> {out}"
    )
    return 0 if gate["passes"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
