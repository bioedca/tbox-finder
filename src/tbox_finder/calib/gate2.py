"""P3-10 — GATE-2, the P3 half: the in-distribution ECE gate on the **named** posterior.

PRD §2.3 (GATE-2) · §12 (calibration) · §18.1 · §9.2 (splits) ·
ADR-0005 **D11** (named posterior = temperature-scaled *before* the deployment
prior-shift, graded at the in-distribution split's own prevalence; 15 equal-mass
debiased bins; **ECE ≤ 0.05**, blinded-frozen, gated at P3 exit) · **D13** +
**Amendment A2** (a *distinct* small-N-robust OOD estimator, bootstrap CIs,
``OOD_ECE_MIN_N = 20``) · ADR-0004 **D5** (the nested leave-one-order-out fold) +
**A7** (``fold_random == "test"`` is the split GATE-2 is graded on; ``calib`` is
where ``T`` is fitted and nowhere else).

Everything above the ``TORCH TIER`` banner is numpy-only and grades in bare CI.

Five things here are load-bearing.

**1. Exactly one number is gated, and it is the one D11 names.** The gated statistic
is the binned ECE of the *named* posterior — ``σ(z/T)``, temperature-scaled, **before**
any prior-shift — on the ``test`` rung, at that rung's own prevalence. The
prior-shifted / deployment-prevalence ECE and every leave-clade-out ECE are
**reported and never gated**; D11 calls the first a machinery-failure check and
D13 says in as many words that an OOD ECE ≤ 0.05 is likely infeasible under clade
shift. :func:`derive_clauses` therefore asserts ``gated is False`` on every OOD
entry, so a later edit cannot quietly promote one into the gate.

**2. The leave-clade-out set is a different population from the gate rung, and its
disjointness is measured, not assumed.** The OOD read is computed over the rows
ADR-0004 D5 designates as the leave-one-order-out holdout — a population the
production checkpoint never trained on. That is the whole basis of calling the
number "OOD", so two clauses re-derive it from the split table: **zero** row overlap
with P3-06's admitted training set, and **zero** *order* overlap between the 30
holdout units and the orders training rows come from. A leak in either direction
turns an in-distribution number into a fraudulent drift measurement, and neither
is visible from the scores file alone (it carries no taxonomy — see
:func:`loo_holdout_rows`).

**3. The deployment prior is a band, not a number, and this step does not pin one.**
PRD §11 and ADR-0005 give the genome-scale prevalence as ``~10³–10⁴ : 1`` in prose;
**nothing** in ``src/``, ``conf/`` or any ADR encodes a scalar. Pinning one would be
a new blinded-frozen default needing ADR-0005 re-sign-off (CLAUDE.md §7 item 2), so
the prior-shifted ECE is reported as a **sweep across the band's own endpoints** and
:func:`validate_report` **refuses** a report that pins a single target prior. This is
the same shape as P3-07's refusal to default ``π_deploy`` and P3-08's refusal to
pin ``τ``.

**4. D13's condition (i) has no pinned bound, so no PASS is adjudicated here.** D13
makes a corpus a "calibrated-negative PASS" only under (i) a drift bound ∧ (ii)
min-N ∧ (iii) a detection-power floor. Amendment A2 pinned (ii) — and only (ii);
D18's delegation map lists no step that pins the drift bound, and it appears
nowhere in ``PRD.md``, ``ADR-0005``, ``ADR-0006``, ``conf/`` or ``src/``. So this
report answers (ii) per unit, records (i) as **unpinned** and (iii) as a P4 quantity,
and leaves ``calibrated_negative_pass`` as ``None`` with the reason attached. A
verdict computed from two of three conditions would read exactly like a verdict
computed from three.

**5. Cost knobs must not be able to certify.** ``--max-units`` and ``--n-boot`` make a
cheap run possible, and every *rule-shaped* clause above survives truncation
unharmed — a 3-unit prefix is still disjoint, still block-resampled, still ungated.
So the completeness clauses (``scored_every_row_of_the_gate_rung``,
``scored_every_designated_loo_holdout_row``, ``graded_every_loo_holdout_unit``) are
derived against the **split table's own counts**, and a truncated run turns them
FALSE ([[cost-knobs-can-certify]]).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import coverage as COV
from tbox_finder import metrics as M
from tbox_finder import power as PW
from tbox_finder import provenance as PROV
from tbox_finder.calib import ece as ECE
from tbox_finder.calib import recalibrate as R
from tbox_finder.eval import resample as RS

__all__ = [
    "ADR",
    "BOOTSTRAP_SEED",
    "COMPLETENESS_CLAUSES",
    "DEFAULT_FIGURE_DATA",
    "DEFAULT_LOO_SCORES",
    "DEFAULT_N_BOOT",
    "DEFAULT_OOD_N_BOOT",
    "DEFAULT_REPORT",
    "DEFAULT_SCORES",
    "ECE_GATE",
    "ECE_N_BINS",
    "ENV_LOCK",
    "FIGURES_DIR",
    "GATE_RUNG",
    "GENERATED_BY",
    "OOD_BLOCK_KEY",
    "OOD_UNIT_KEY",
    "PRD",
    "RESCORE_AGREEMENT_TOL",
    "SCHEMA_VERSION",
    "STEP",
    "build_parser",
    "build_report",
    "derive_clauses",
    "figure_data",
    "grade_in_distribution",
    "grade_ood_units",
    "holdout_from_rows",
    "loo_holdout_rows",
    "main",
    "plot_figures",
    "prior_shift_band_sweep",
    "score_loo_holdout",
    "validate_report",
    "write_outputs",
]

SCHEMA_VERSION = "1"
STEP = "P3-10"
GENERATED_BY = "src/tbox_finder/calib/gate2.py"
PRD = "PRD §2.3 (GATE-2), §12, §18.1, §9.2"
ADR = (
    "ADR-0005 D11 (named posterior = temperature-scaled PRE prior-shift; 15 equal-mass "
    "debiased bins; in-distribution ECE <= 0.05, blinded-frozen, gated at P3 exit) + D13 & "
    "Amendment A2 (distinct small-N-robust OOD estimator, bootstrap CIs, OOD_ECE_MIN_N=20); "
    "ADR-0004 D5 (nested leave-one-order-out fold) + A7 (test rung is the GATE-2 split)"
)
#: The score producer runs under the Stage-2 inference env; the grading tier is numpy-only.
ENV_LOCK = "envs/ml-rna.conda-lock.yml"

DEFAULT_DATASET = "data/processed/stage2_dataset.parquet"
DEFAULT_SCORES = "reports/p3/stage2_scores.json"
DEFAULT_LOO_SCORES = "reports/p3/stage2_scores_loo.json"
DEFAULT_REPORT = "reports/gate2_p3_ece.json"
DEFAULT_FIGURE_DATA = "reports/p3/gate2_figure_data.json"
FIGURES_DIR = "figures/calib"

#: ADR-0004 A7: GATE-2 is graded on the ``test`` rung. Imported, never re-typed.
GATE_RUNG = "test"
#: D11's estimator config, both imported from their frozen homes.
ECE_N_BINS = M.ECE_N_BINS
ECE_GATE = PW.ECE_GATE

#: PRD §12 resamples at the homology-cluster level *within* a held-out order …
OOD_BLOCK_KEY = "cluster_id"
#: … and macro-averages *across* held-out orders, which are themselves the blocks.
OOD_UNIT_KEY = "loo_order_unit"

#: The clauses that say the run covered its whole declared population. `is_science` is
#: their conjunction, so a cost-knobbed run cannot present as a full grade.
COMPLETENESS_CLAUSES: tuple[str, ...] = (
    "scored_every_row_of_the_gate_rung",
    "scored_every_designated_loo_holdout_row",
    "graded_every_loo_holdout_unit",
)

DEFAULT_N_BOOT = 2000
DEFAULT_OOD_N_BOOT = ECE.DEFAULT_OOD_N_BOOT
BOOTSTRAP_SEED = 20260804

#: bf16 + flash-attention reductions are not bit-reproducible across a different batch
#: composition, so the re-scoring control is an agreement bound, not an equality.
RESCORE_AGREEMENT_TOL = 5e-2

_ROW_ID = "row_id"
_LABEL = "is_tbox"
_SEQUENCE = "rna_sequence"
_CLUSTER = "cluster_id"
_CALIB = "calib"
_FOLD_RANDOM = "fold_random"
_FOLD_BASIS = "fold_basis"
_NESTED_TRAIN = "nested_train"
_LOO_HOLDOUT = "is_designated_loo_holdout"
_LOO_UNIT = "loo_order_unit"
_RESOLVED_ORDER = "resolved_order"
_RESOLVED_PHYLUM = "resolved_phylum"

_DECOY_POOL_RANDOM = "decoy_pool_random"

_UNPINNED_DRIFT_BOUND = (
    "ADR-0005 D13 condition (i) compares the leave-clade-out ECE to a *pinned drift "
    "bound*. No such value exists: D18's delegation map assigns only the min-N floor "
    "(pinned by Amendment A2), and no number of that shape appears in PRD.md, ADR-0005, "
    "ADR-0006, conf/ or src/. Pinning one is a new blinded-frozen default requiring "
    "ADR-0005 sign-off (CLAUDE.md §7 item 2), so condition (i) is left unadjudicated "
    "and no calibrated-negative PASS is derived from the two conditions that do have "
    "values."
)
_UNPINNED_DEPLOYMENT_PRIOR = (
    "The genome-scale deployment prevalence is prose only — '~10^3-10^4 : 1' at PRD.md "
    "§11 and ADR-0005 D7's contrast line — and nothing in src/, conf/ or any ADR encodes "
    "a scalar. The prior-shifted ECE is therefore reported across the band's endpoints "
    "and no single target prior is pinned; pinning one would be a new blinded-frozen "
    "default (CLAUDE.md §7 item 2)."
)


# --------------------------------------------------------------------------- #
# Split-table joins — the scores files carry no taxonomy and no clustering
# --------------------------------------------------------------------------- #
def _recorded_path(path: str | Path) -> str:
    """A path as the repository sees it — never as this machine does.

    Every path here lands in a committed artifact in a **public** repo, where an absolute
    path publishes the OS user name and the local directory layout and contributes nothing
    a reader can use; the sha256 beside it is the identity evidence, the string is only a
    locator. It bites hardest under a linked worktree, where the inputs are reached by
    absolute path from the main checkout — exactly how this ran. Delegates to P3-08's
    :func:`stage2.eval.repo_relative`, which already relativises against **both** roots,
    rather than adding a fourth copy of that logic to the repo.
    """
    from tbox_finder.stage2 import eval as E

    return E.repo_relative(path)


def _is_true(value: Any) -> bool:
    """``True`` only for a genuine boolean truth, never for ``nan`` or ``"False"``.

    ``bool(float("nan")) is True`` and pandas delivers nulls as NaN, so the naive test
    admits exactly the rows whose flag is *absent* ([[pandas-3-nan-truthy-in-training-env]]).
    Delegates to P3-07's shared parse rather than re-deriving the vocabulary.
    """
    from tbox_finder import masking

    if masking.is_missing(value):
        return False
    return bool(masking.bool_or_none(value))


def training_admission(row: Mapping[str, Any]) -> bool:
    """P3-06's Stage-2 train-eligibility predicate, re-derived from the split table.

    ``fold_random == "train"`` **and not** ``calib`` **and** (``nested_train`` **or** the
    row is a parentless decoy). Re-derived rather than read from a recorded count because
    the two disjointness clauses this feeds are the entire evidence that the leave-clade-out
    population is out of distribution ([[gate-clauses-need-re-derivation]]).
    """
    if str(row.get(_FOLD_RANDOM)) != "train":
        return False
    if _is_true(row.get(_CALIB)):
        return False
    return _is_true(row.get(_NESTED_TRAIN)) or str(row.get(_FOLD_BASIS)) == _DECOY_POOL_RANDOM


def _read_split_table(
    dataset: str | Path, *, with_sequences: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every row of the Stage-2 dataset with the columns this step joins on.

    ``with_sequences`` is off by default so the grading tier — which never needs a base —
    does not carry 30,542 RNA sequences through a report build.
    """
    import pandas as pd

    path = Path(dataset)
    columns = [
        _ROW_ID,
        _LABEL,
        _CLUSTER,
        _CALIB,
        _FOLD_RANDOM,
        _FOLD_BASIS,
        _NESTED_TRAIN,
        _LOO_HOLDOUT,
        _LOO_UNIT,
        _RESOLVED_ORDER,
        _RESOLVED_PHYLUM,
    ]
    if with_sequences:
        columns.append(_SEQUENCE)
    frame = pd.read_parquet(path, columns=columns)
    rows = frame.to_dict("records")
    for row in rows:
        row[_ROW_ID] = str(row[_ROW_ID])
    meta = {
        "path": _recorded_path(path),
        "sha256": PROV.sha256_file(path),
        "n_rows": len(rows),
    }
    return rows, meta


def loo_holdout_rows(
    dataset: str | Path = DEFAULT_DATASET, *, with_sequences: bool = False
) -> tuple[list[dict[str, Any]], dict]:
    """The ADR-0004 D5 designated leave-one-order-out holdout, plus its disjointness census.

    Returns the holdout rows — each carrying ``_block`` (its homology cluster, or itself
    when it has none) and ``_unit`` (its held-out order) — and a census that *measures*,
    rather than asserts, the two properties that make these rows OOD: no row of the holdout
    was admitted to Stage-2 training, and no *order* the holdout covers contributed a
    training row either. A leak of the second kind would leave every row-level check clean
    while the model had in fact seen the clade.
    """
    rows, meta = _read_split_table(dataset, with_sequences=with_sequences)
    holdout, census = holdout_from_rows(rows)
    census["dataset"] = meta
    return holdout, census


def holdout_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """The pure half of :func:`loo_holdout_rows` — split-table rows in, holdout + census out.

    Separated so the disjointness arithmetic, which is the entire evidence that the OOD
    number is OOD, is testable without a parquet on disk.
    """
    from tbox_finder.stage2 import eval as E

    trained = [row for row in rows if training_admission(row)]
    holdout = [dict(row) for row in rows if _is_true(row.get(_LOO_HOLDOUT))]

    trained_ids = {row[_ROW_ID] for row in trained}
    holdout_ids = {row[_ROW_ID] for row in holdout}
    trained_orders = {
        str(row[_RESOLVED_ORDER]) for row in trained if not _is_missing(row[_RESOLVED_ORDER])
    }
    holdout_orders = {
        str(row[_RESOLVED_ORDER]) for row in holdout if not _is_missing(row[_RESOLVED_ORDER])
    }

    keys, block_census = E.block_keys(holdout)
    for row, key in zip(holdout, keys, strict=True):
        unit = row.get(_LOO_UNIT)
        if _is_missing(unit):
            raise ValueError(
                f"row {row[_ROW_ID]!r} is flagged {_LOO_HOLDOUT} but carries no {_LOO_UNIT} — "
                "a holdout row with no unit cannot be assigned to a leave-clade-out fold, and "
                "silently dropping it would shrink a per-unit N that a min-N flag is read from"
            )
        row["_block"] = key
        row["_unit"] = str(unit)
        phylum = row.get(_RESOLVED_PHYLUM)
        row["_phylum"] = None if _is_missing(phylum) else str(phylum)

    census = {
        "n_training_rows": len(trained),
        "n_holdout_rows": len(holdout),
        "n_row_overlap": len(trained_ids & holdout_ids),
        "n_training_orders": len(trained_orders),
        "n_holdout_orders": len(holdout_orders),
        "n_order_overlap": len(trained_orders & holdout_orders),
        "overlapping_orders": sorted(trained_orders & holdout_orders),
        "n_units": len({row["_unit"] for row in holdout}),
        "block_census": block_census,
    }
    return holdout, census


def _is_missing(value: Any) -> bool:
    from tbox_finder import masking

    return masking.is_missing(value) or masking.is_null_token(value)


def gate_rung_row_ids(dataset: str | Path = DEFAULT_DATASET) -> set[str]:
    """Every ``row_id`` the split table puts on the GATE-2 rung — the completeness denominator."""
    from tbox_finder.stage2 import eval as E

    rows, _ = _read_split_table(dataset)
    rungs = E.rungs_for_rows(rows)
    return {row[_ROW_ID] for row, rung in zip(rows, rungs, strict=True) if rung == GATE_RUNG}


# --------------------------------------------------------------------------- #
# Score-file loading
# --------------------------------------------------------------------------- #
def load_scores(path: str | Path, arm: str) -> dict[str, Any]:
    """One arm's columnar score table, checked for the parallel-array invariant."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or arm not in arms:
        raise ValueError(
            f"{path}: no arm {arm!r} (has {sorted(arms) if isinstance(arms, Mapping) else arms!r})"
        )
    row_ids = [str(v) for v in payload["row_ids"]]
    logits = [float(v) for v in arms[arm]["logits"]]
    labels = [int(v) for v in payload["labels"]]
    if not (len(row_ids) == len(logits) == len(labels)):
        raise ValueError(
            f"{path}: parallel arrays disagree — row_ids={len(row_ids)}, "
            f"logits={len(logits)}, labels={len(labels)}"
        )
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"{path}: row_ids are not unique, so a join back would fan out")
    return {
        "row_ids": row_ids,
        "logits": logits,
        "labels": labels,
        "rungs": [str(v) for v in payload.get("rungs", [])] or None,
        "meta": {k: v for k, v in payload.items() if k not in {"arms", "row_ids", "labels"}},
        "load": arms[arm].get("load"),
    }


# --------------------------------------------------------------------------- #
# The gated read
# --------------------------------------------------------------------------- #
def _blocks(items: Sequence[Any], keys: Sequence[str]) -> list[list[Any]]:
    """Group ``items`` by block key through P3-09's resampler.

    Routed through :func:`resample.blocks_by_key` rather than a local ``dict`` so the
    record-level-column refusal and the null-label refusal apply here too; the keys handed
    in are ``stage2.eval.block_keys``' derived strings, which are cluster ids where a row
    has one and the row itself where it does not.
    """
    return RS.blocks_by_key(items, keys, key_name=OOD_BLOCK_KEY)


def grade_in_distribution(
    *,
    scores: Mapping[str, Any],
    blocks_by_row: Mapping[str, str],
    n_bins: int = ECE_N_BINS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fit ``T`` on ``calib`` alone, then grade the **named** posterior on the ``test`` rung.

    The fit is delegated to P3-07's :func:`recalibrate.temperature_scale`, which selects the
    ``calib`` rows itself: no argument here can admit a graded row into the fit, and its
    emptiness / unknown-token / single-class refusals carry through unchanged.
    """
    rungs = scores["rungs"]
    if rungs is None:
        raise ValueError("the in-distribution score file carries no rungs, so T has no split")
    fit = R.temperature_scale(scores["logits"], scores["labels"], rung=rungs)
    payload = R.calibrated_posterior(scores["logits"], temperature=fit.temperature)
    posterior = [float(v) for v in payload[R.NAMED_POSTERIOR_KEY]]

    idx = [i for i, rung in enumerate(rungs) if rung == GATE_RUNG]
    if not idx:
        raise ValueError(
            f"no rows on the {GATE_RUNG!r} rung — a grade over nothing reads as a pass"
        )
    calib_ids = {scores["row_ids"][i] for i, rung in enumerate(rungs) if rung == R.CALIB_RUNG}
    graded_ids = {scores["row_ids"][i] for i in idx}

    y = [int(scores["labels"][i]) for i in idx]
    p = [posterior[i] for i in idx]
    keys = [blocks_by_row[scores["row_ids"][i]] for i in idx]

    ece = M.binned_ece(y, p, n_bins, debias=True)
    ece_plugin = M.binned_ece(y, p, n_bins, debias=False)
    reliability = M.reliability_bins(y, p, n_bins)
    pairs = list(zip(y, p, strict=True))
    ci = M.block_bootstrap_ci(
        _blocks(pairs, keys),
        lambda sample: M.binned_ece([a for a, _ in sample], [b for _, b in sample], n_bins),
        n_boot=n_boot,
        seed=seed,
    )
    n_pos = int(sum(y))
    return {
        "graded_rung": GATE_RUNG,
        "graded_posterior_key": payload["gated_posterior_key"],
        "graded_object": "named_posterior (temperature-scaled, PRE prior-shift) — ADR-0005 D11",
        "prior_shift_applied": bool(payload["prior_shift_applied"]),
        "stack_order": list(payload["stack_order"]),
        "stack_applied": list(payload["stack_applied"]),
        "n": len(idx),
        "n_positive": n_pos,
        "n_negative": len(idx) - n_pos,
        "prevalence": n_pos / len(idx),
        "ece": ece,
        "ece_plugin": ece_plugin,
        "ece_ci": ci,
        "ece_n_bins": int(n_bins),
        "ece_binning": "equal_mass",
        "ece_debiased": True,
        "estimator": ECE.IN_DISTRIBUTION_ESTIMATOR,
        "gate": ECE_GATE,
        "passes": bool(M.gate2_ece_pass(ece)),
        "n_blocks": len(set(keys)),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "reliability": reliability,
        "bin_concentration": _bin_concentration(reliability, ece),
        "calibration": {
            "temperature": float(fit.temperature),
            "beta": float(fit.beta),
            "fitted_on": fit.fitted_on,
            "n_fitted": int(fit.n_fitted),
            "n_total": int(fit.n_total),
            "n_by_rung": dict(fit.n_by_rung),
            "converged": bool(fit.fit.converged),
            "n_iterations": int(fit.fit.n_iterations),
            "nll_per_position_initial": float(fit.fit.nll_per_position_initial),
            "nll_per_position_final": float(fit.fit.nll_per_position_final),
            "calib_prevalence": _prevalence(scores, rungs, R.CALIB_RUNG),
            "n_fit_rows_also_graded": len(calib_ids & graded_ids),
        },
    }


def _bin_size_label(rows: Sequence[Mapping[str, Any]]) -> str:
    """ "N rows each" only when every bin really has N.

    Equal-**mass** binning splits n rows into :data:`ECE_N_BINS` groups; when n is not
    divisible by the bin count the groups differ in size, so quoting one bin's count for all
    of them states a number some bins do not have. (n = 3,045 happens to divide by 15 today —
    which is exactly the kind of coincidence a label should not depend on.)
    """
    counts = sorted({int(row["n"]) for row in rows})
    if not counts:
        return "no rows"
    return f"{counts[0]} rows each" if len(counts) == 1 else f"{counts[0]}-{counts[-1]} rows"


def _bin_concentration(reliability: Sequence[Mapping[str, Any]], ece: float) -> dict[str, Any]:
    """How much of the gated ECE comes from how few bins — the margin's own evidence base.

    A near-separated posterior puts almost every row at p ~= 0 or p ~= 1, where an
    equal-mass binning produces bins that are trivially correct (accuracy exactly 0 or 1,
    zero gap) while ONE bin absorbs the entire transition region. The headline ECE is then
    small largely because 14 bins had nothing to get wrong, and its whole magnitude rests on
    the remaining one. That is the same shape as P3-08's single misclassified calib row: the
    gate passes honestly, and a reader must still be able to see how concentrated the
    evidence is. Reported, never gated — no threshold is pinned on any of these.
    """
    # `max` over the VALUES, then `.index()` — the FIRST maximum wins. Comparing
    # `(value, index)` tuples would break ties by the largest index instead, and a
    # near-separated posterior makes ties at the maximum reachable (11 of 15 bins here carry
    # a debiased gap of exactly 0.0), so the two orders genuinely disagree.
    contributions = [float(b["weight"]) * float(b["debiased_gap"]) for b in reliability]
    spans = [float(b["p_max"]) - float(b["p_min"]) for b in reliability]
    top = (
        (max(contributions), contributions.index(max(contributions)))
        if contributions
        else (0.0, -1)
    )
    widest = (max(spans), spans.index(max(spans))) if spans else (0.0, -1)
    return {
        "n_bins": len(reliability),
        "top_bin_index": top[1],
        "top_bin_contribution": top[0],
        "top_bin_share_of_ece": (top[0] / ece) if ece else None,
        "n_bins_with_zero_debiased_gap": sum(
            1 for b in reliability if float(b["debiased_gap"]) == 0.0
        ),
        "n_bins_with_saturated_accuracy": sum(
            1 for b in reliability if float(b["acc"]) in (0.0, 1.0)
        ),
        "widest_bin_index": widest[1],
        "widest_bin_span": widest[0],
        "gated": False,
        "note": (
            "descriptive only — no threshold is pinned on bin concentration anywhere; it "
            "states how much of the reported ECE rests on how few bins"
        ),
    }


def _prevalence(scores: Mapping[str, Any], rungs: Sequence[str], rung: str) -> float | None:
    picked = [int(scores["labels"][i]) for i, r in enumerate(rungs) if r == rung]
    return (sum(picked) / len(picked)) if picked else None


# --------------------------------------------------------------------------- #
# The prior-shifted read — a band, never a pin
# --------------------------------------------------------------------------- #
def prior_shift_band_sweep(
    *,
    scores: Mapping[str, Any],
    source_prior: float,
    n_bins: int = ECE_N_BINS,
    temperature: float,
) -> dict[str, Any]:
    """Deployment-prevalence ECE across the PRD band's endpoints — reported, never gated.

    The band is ``DEPLOYMENT_PRIOR_ODDS_RANGE`` (``10^3``–``10^4`` negatives per positive)
    read from P3-07, plus its geometric interior point, so the sweep brackets the whole
    prose range without any of its points being *the* deployment prior. This posterior is
    miscalibrated at benchmark prevalence **by construction** — that is exactly why D11
    gates the pre-shift object instead — so a large ECE here is the expected reading, not a
    finding.
    """
    rungs = scores["rungs"]
    idx = [i for i, rung in enumerate(rungs) if rung == GATE_RUNG]
    y = [int(scores["labels"][i]) for i in idx]
    raw = [float(scores["logits"][i]) for i in idx]

    low, high = R.DEPLOYMENT_PRIOR_ODDS_RANGE
    odds_grid = [float(low), float(math.sqrt(low * high)), float(high)]
    points = []
    for odds in odds_grid:
        target = R.prior_from_odds_ratio(odds)
        # Route through the SHIPPED stack producer, not a hand-composed shift. `prior_shift`
        # returns shifted **logits** — deliberately, so the correction stays exactly additive
        # — and feeding those to `binned_ece` as if they were probabilities silently produces
        # an "ECE" above 1. `calibrated_posterior` applies temperature and the shift in the
        # D11 order and hands back the named posterior for that stage under its own key.
        payload = R.calibrated_posterior(
            raw, temperature=temperature, source_prior=source_prior, target_prior=target
        )
        p = [float(v) for v in payload[R.PRIOR_SHIFTED_POSTERIOR_KEY]]
        if not all(0.0 <= v <= 1.0 for v in p):
            raise ValueError(
                "the prior-shifted read is not a probability — an ECE computed on it would "
                "be meaningless and could exceed 1"
            )
        points.append(
            {
                "negatives_per_positive": odds,
                "target_prior": target,
                "target_prior_in_prd_band": bool(R.deployment_prior_in_prd_band(target)),
                "log_odds_shift": R.log_odds_shift(source_prior=source_prior, target_prior=target),
                "ece": M.binned_ece(y, p, n_bins, debias=True),
                "ece_plugin": M.binned_ece(y, p, n_bins, debias=False),
                "is_band_endpoint": odds in (float(low), float(high)),
            }
        )
    return {
        "gated": False,
        "why_not_gated": (
            "ADR-0005 D11 / PRD §12: a prior-shifted posterior is miscalibrated at benchmark "
            "prevalence by construction, so the deployment-prevalence ECE is reported and the "
            "PRE-shift named posterior is what GATE-2 grades"
        ),
        "source_prior": float(source_prior),
        "source_prior_is": "the calib rung's own prevalence — the split T was fitted on",
        "pinned_target_prior": None,
        "unpinned_reason": _UNPINNED_DEPLOYMENT_PRIOR,
        "band_odds": [float(low), float(high)],
        "graded_rung": GATE_RUNG,
        "n": len(idx),
        "points": points,
    }


# --------------------------------------------------------------------------- #
# The OOD / leave-clade-out read
# --------------------------------------------------------------------------- #
def grade_ood_units(
    *,
    scores: Mapping[str, Any],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    temperature: float,
    n_boot: int = DEFAULT_OOD_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    max_units: int | None = None,
) -> dict[str, Any]:
    """Per-held-out-order OOD ECE through D13's estimator, with the A2 min-N flag.

    One unit = one leave-one-order-out holdout order = D13's "nearest-relative leave-clade-out
    calibration unit". Within a unit the exchangeability block is the homology cluster
    (PRD §12); *across* units the block is the unit itself, which is what the macro-average's
    CI resamples — PRD §12 macro-averages across held-out orders precisely because the
    ~90 %-Firmicutes corpus would otherwise let one order speak for the tree.
    """
    posterior = R.calibrated_posterior(scores["logits"], temperature=temperature)
    p_all = [float(v) for v in posterior[R.NAMED_POSTERIOR_KEY]]

    by_unit: dict[str, dict[str, list]] = {}
    for i, row_id in enumerate(scores["row_ids"]):
        row = rows_by_id.get(row_id)
        if row is None:
            raise ValueError(
                f"scored row {row_id!r} is not in the designated leave-one-order-out holdout — "
                "the OOD read would then be measuring a population it does not name"
            )
        bucket = by_unit.setdefault(row["_unit"], {"y": [], "p": [], "blocks": [], "phyla": set()})
        bucket["y"].append(int(scores["labels"][i]))
        bucket["p"].append(p_all[i])
        bucket["blocks"].append(row["_block"])
        if row.get("_phylum") is not None:
            bucket["phyla"].add(row["_phylum"])

    unit_names = sorted(by_unit)
    truncated_to = None
    # Only when the list was REALLY cut: `--max-units 60` on a 30-unit holdout truncates
    # nothing, and recording it anyway turns `graded_every_loo_holdout_unit` FALSE on a run
    # that graded every unit.
    if max_units is not None and int(max_units) < len(unit_names):
        truncated_to = int(max_units)
        unit_names = unit_names[:truncated_to]

    units: dict[str, Any] = {}
    for name in unit_names:
        bucket = by_unit[name]
        out = ECE.ood_ece(
            bucket["y"],
            bucket["p"],
            bucket["blocks"],
            block_key=OOD_BLOCK_KEY,
            n_boot=n_boot,
            seed=seed,
        )
        out["unit"] = name
        out["unit_key"] = OOD_UNIT_KEY
        # `ci.n_boot` counts the replicates that SURVIVED; `block_bootstrap` drops any whose
        # statistic is non-finite. The leave-one-out kernel excludes by ROW, so a replicate
        # is undefined exactly when every row in it shares one uid — i.e. when the draw came
        # entirely from a SINGLETON block. That is reachable whenever a unit has both few
        # blocks and a singleton: `Pseudonocardiales` has sizes [1, 2, 49] and lost 7 of 200,
        # which is 1/27 = the chance of drawing its singleton three times. A shortfall here
        # is therefore rejected resamples, not a truncated bootstrap, and an auditor must be
        # able to tell those apart without inferring it from a count.
        out["n_boot_requested"] = int(n_boot)
        survived = (out.get("ci") or {}).get("n_boot")
        out["n_boot_dropped"] = None if survived is None else max(0, int(n_boot) - int(survived))
        out["n_boot_drop_reason"] = (
            "replicates whose leave-one-out statistic was non-finite were dropped by "
            "eval.resample.block_bootstrap: the kernel leaves out by row, so a replicate "
            "drawn entirely from a singleton block leaves no row with a distinct-uid "
            "neighbour. Rejected resamples, not a truncated bootstrap."
        )
        out["admissibility_class"] = COV.classify_order(out["n_positives"])
        phyla = sorted(bucket["phyla"])
        if len(phyla) > 1:
            raise ValueError(
                f"held-out order {name!r} resolves to more than one phylum ({phyla}) — the "
                "phylum stratification below would then average across a taxonomy fault"
            )
        out["phylum"] = phyla[0] if phyla else None
        units[name] = out

    admissible = [u for u in units.values() if u["admissible"]]
    macro = None
    if admissible:
        values = [(u["unit"], float(u["ood_ece"])) for u in admissible]
        macro_blocks = RS.blocks_by_key(values, [v[0] for v in values], key_name=OOD_UNIT_KEY)
        macro = RS.block_bootstrap(
            macro_blocks,
            lambda sample: sum(v for _, v in sample) / len(sample) if sample else float("nan"),
            n_boot=n_boot,
            seed=seed,
        )

    return {
        "gated": False,
        "why_not_gated": (
            "ADR-0005 D13: an OOD / leave-clade-out ECE <= 0.05 is likely infeasible under "
            "clade shift, so it is reported and adjudicated by the drift decision rule, never "
            "gated. GATE-2 grades the D11 in-distribution number only."
        ),
        "estimator": ECE.ESTIMATOR,
        "distinct_from": ECE.IN_DISTRIBUTION_ESTIMATOR,
        "unit_key": OOD_UNIT_KEY,
        "block_key": OOD_BLOCK_KEY,
        "min_n": COV.OOD_ECE_MIN_N,
        "n_units": len(units),
        "n_units_admissible": len(admissible),
        "n_units_sub_min_n": len(units) - len(admissible),
        "adjudicable_fraction": (len(admissible) / len(units)) if units else None,
        "macro_average": macro,
        "by_phylum": _stratify_by_phylum(units),
        "by_phylum_is": (
            "the admissible units' OOD ECEs grouped by the phylum of the held-out order — "
            "PRD §12's ECE-vs-phylogenetic-distance read, REPORTED and not gated. Descriptive "
            "only: these are the same per-unit numbers regrouped, not a separate estimate, and "
            "no phylum-level bound is pinned anywhere"
        ),
        "macro_average_is": (
            "unweighted mean of the admissible units' OOD ECEs, block-resampled with the "
            "held-out order as the block (PRD §12 macro-average, not micro)"
        ),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "truncated_to_n_units": truncated_to,
        "d13_adjudication": {
            "condition_i_drift_bound": {
                "pinned": False,
                "bound": None,
                "verdict": None,
                "reason": _UNPINNED_DRIFT_BOUND,
            },
            "condition_ii_min_n": {
                "pinned": True,
                "floor": COV.OOD_ECE_MIN_N,
                "source": "ADR-0005 Amendment A2; tbox_finder.coverage.OOD_ECE_MIN_N",
                "per_unit": {name: units[name]["admissibility_class"] for name in units},
            },
            "condition_iii_detection_power_floor": {
                "pinned": False,
                "verdict": None,
                "reason": (
                    "the corpus-specific detection-power floor is an extrapolated "
                    "recall@matched-precision / synthetic-Tier-2N recovery quantity (PRD §12), "
                    "produced at P4 — it is not computable from a calibration report"
                ),
            },
            "calibrated_negative_pass": None,
            "calibrated_negative_pass_reason": (
                "D13 makes a corpus a calibrated-negative PASS only under (i) AND (ii) AND "
                "(iii). Only (ii) has a pinned value, so no verdict is derived here: a PASS "
                "computed from two of three conditions would be indistinguishable from one "
                "computed from three."
            ),
        },
        "units": units,
    }


def _stratify_by_phylum(units: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Regroup the ADMISSIBLE units' OOD ECEs by the phylum of the held-out order.

    Descriptive, and deliberately so: it is the same per-unit estimates rearranged, not a
    pooled phylum-level estimate (pooling would need the rows re-blocked and re-estimated,
    and would let the largest order speak for its phylum). Inadmissible units are excluded
    because their value is an ``inadmissible_point`` the ADR says supports no verdict.
    """
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for unit in units.values():
        if not unit.get("admissible"):
            continue
        buckets.setdefault(str(unit.get("phylum")), []).append(unit)
    out: dict[str, Any] = {}
    for phylum, members in sorted(buckets.items()):
        values = sorted(float(u["ood_ece"]) for u in members)
        mid = len(values) // 2
        out[phylum] = {
            "n_units": len(members),
            "n_records": sum(int(u["n_records"]) for u in members),
            "n_positives": sum(int(u["n_positives"]) for u in members),
            "min_ood_ece": values[0],
            "max_ood_ece": values[-1],
            "median_ood_ece": (
                values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
            ),
            "mean_ood_ece": sum(values) / len(values),
            "units": sorted(str(u["unit"]) for u in members),
        }
    return out


# --------------------------------------------------------------------------- #
# Report assembly, clause derivation, validation
# --------------------------------------------------------------------------- #
def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """Re-derive every GATE-2 clause from the report's **own** numbers.

    Each clause is guarded on its evidence being present, because the failure mode that
    matters is a clause an *absent* block would fabricate TRUE ([[gate-clauses-need-re-derivation]],
    [[clauses-must-guard-emptiness]]).
    """
    ind = report.get("gate") if isinstance(report.get("gate"), Mapping) else {}
    ood = report.get("ood") if isinstance(report.get("ood"), Mapping) else {}
    shift = report.get("prior_shift") if isinstance(report.get("prior_shift"), Mapping) else {}
    scope = report.get("scope") if isinstance(report.get("scope"), Mapping) else {}
    cal = ind.get("calibration") if isinstance(ind.get("calibration"), Mapping) else {}
    units = ood.get("units") if isinstance(ood.get("units"), Mapping) else {}

    ece = ind.get("ece")
    has_ece = isinstance(ece, (int, float)) and not isinstance(ece, bool) and math.isfinite(ece)

    return {
        # ── the gated statistic ──
        "in_distribution_ece_within_gate": bool(has_ece and M.gate2_ece_pass(float(ece))),
        "gate_threshold_is_the_pinned_default": ind.get("gate") == ECE_GATE,
        "ece_estimator_matches_adr_d11": (
            ind.get("ece_n_bins") == ECE_N_BINS
            and ind.get("ece_binning") == "equal_mass"
            and ind.get("ece_debiased") is True
            and ind.get("estimator") == ECE.IN_DISTRIBUTION_ESTIMATOR
        ),
        "graded_object_is_pre_prior_shift": (
            ind.get("graded_posterior_key") == R.NAMED_POSTERIOR_KEY
            and ind.get("prior_shift_applied") is False
        ),
        "graded_rung_is_the_gate2_split": ind.get("graded_rung") == GATE_RUNG,
        # ── the fit ──
        # `n_by_rung` is a census of EVERY rung, not of the fit, so "the only key is calib"
        # would never hold and would be a clause that can only fail. What must hold is that
        # the fit consumed exactly the calib rows and not one row it later grades.
        "temperature_fitted_on_calib_only": (
            cal.get("fitted_on") == R.CALIB_RUNG
            and isinstance(cal.get("n_by_rung"), Mapping)
            and cal.get("n_fitted") == cal["n_by_rung"].get(R.CALIB_RUNG)
            and isinstance(cal.get("n_fitted"), int)
            and cal["n_fitted"] > 0
            and cal.get("n_fit_rows_also_graded") == 0
        ),
        "temperature_positive_and_converged": (
            isinstance(cal.get("temperature"), (int, float))
            and float(cal.get("temperature", 0.0)) > 0.0
            and cal.get("converged") is True
        ),
        # ── completeness: a cost knob must not certify ──
        "scored_every_row_of_the_gate_rung": (
            isinstance(scope.get("n_gate_rung_rows_in_split_table"), int)
            and ind.get("n") == scope["n_gate_rung_rows_in_split_table"]
        ),
        "scored_every_designated_loo_holdout_row": (
            isinstance(scope.get("n_loo_holdout_rows_in_split_table"), int)
            and scope.get("n_loo_holdout_rows_scored") == scope["n_loo_holdout_rows_in_split_table"]
        ),
        "graded_every_loo_holdout_unit": (
            isinstance(scope.get("n_loo_holdout_units_in_split_table"), int)
            and ood.get("n_units") == scope["n_loo_holdout_units_in_split_table"]
            and ood.get("truncated_to_n_units") is None
        ),
        # ── the OOD population really is out of distribution ──
        "loo_holdout_rows_disjoint_from_training_rows": scope.get("n_row_overlap") == 0,
        "loo_holdout_orders_disjoint_from_training_orders": scope.get("n_order_overlap") == 0,
        # ── the OOD read is a second estimator, reported and never gated ──
        "ood_estimator_distinct_from_the_in_distribution_one": (
            ood.get("estimator") == ECE.ESTIMATOR
            and ood.get("distinct_from") == ECE.IN_DISTRIBUTION_ESTIMATOR
            and ood.get("estimator") != ECE.IN_DISTRIBUTION_ESTIMATOR
        ),
        "ood_min_n_floor_is_the_pinned_constant": (
            ood.get("min_n") == COV.OOD_ECE_MIN_N
            and bool(units)
            and all(u.get("min_n") == COV.OOD_ECE_MIN_N for u in units.values())
        ),
        "ood_reported_never_gated": (
            ood.get("gated") is False
            and bool(units)
            and all(u.get("gated") is False for u in units.values())
        ),
        "ood_resampled_at_block_granularity": (
            bool(units)
            and all(u.get("block_key") in RS.BLOCK_GRANULARITY_COLUMNS for u in units.values())
            and ood.get("unit_key") in RS.BLOCK_GRANULARITY_COLUMNS
        ),
        # ── nothing unpinned was quietly pinned ──
        "deployment_prior_is_a_band_not_a_pin": (
            shift.get("pinned_target_prior") is None
            and isinstance(shift.get("points"), list)
            and sum(1 for pt in shift["points"] if pt.get("is_band_endpoint")) == 2
            and shift.get("gated") is False
        ),
        "d13_drift_bound_left_unadjudicated": _drift_bound_unadjudicated(ood),
        # ── the re-scored overlap reproduces P3-08 ──
        "rescoring_reproduces_the_p3_08_overlap": (
            isinstance(scope.get("rescore_max_abs_delta"), (int, float))
            and isinstance(scope.get("rescore_n_overlap"), int)
            and scope["rescore_n_overlap"] > 0
            and float(scope["rescore_max_abs_delta"]) <= RESCORE_AGREEMENT_TOL
        ),
    }


def _drift_bound_unadjudicated(ood: Mapping[str, Any]) -> bool:
    """TRUE only when the report *has* a D13 block and it declines both unpinned conditions.

    Written as a positive check on present evidence rather than ``bound is None``: a report
    with no ``d13_adjudication`` at all also has no pinned bound, and would otherwise satisfy
    the clause by omission ([[clauses-must-guard-emptiness]]).
    """
    block = ood.get("d13_adjudication")
    if not isinstance(block, Mapping):
        return False
    cond_i = block.get("condition_i_drift_bound")
    cond_iii = block.get("condition_iii_detection_power_floor")
    if not isinstance(cond_i, Mapping) or not isinstance(cond_iii, Mapping):
        return False
    return (
        cond_i.get("pinned") is False
        and cond_i.get("bound") is None
        and cond_i.get("verdict") is None
        and bool(cond_i.get("reason"))
        and cond_iii.get("pinned") is False
        and block.get("calibrated_negative_pass") is None
        and bool(block.get("calibrated_negative_pass_reason"))
    )


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Problems with ``report``; ``[]`` when clean. An honestly-failing gate is not a problem.

    Re-derives the clauses and reports a *disagreement* with what was written, so a report
    whose ``overall_pass`` was assigned rather than computed is caught. Also refuses the two
    values this step is not allowed to pin.
    """
    problems: list[str] = []

    for key, want in (
        ("schema_version", SCHEMA_VERSION),
        ("step", STEP),
        ("generated_by", GENERATED_BY),
        ("adr", ADR),
        ("prd", PRD),
    ):
        if report.get(key) != want:
            problems.append(f"{key}: expected {want!r}, got {report.get(key)!r}")

    gate = report.get("gate")
    if not isinstance(gate, Mapping) or not gate:
        problems.append("gate: block missing — there is no gated statistic to check")
    else:
        ece = gate.get("ece")
        if not isinstance(ece, (int, float)) or isinstance(ece, bool) or not math.isfinite(ece):
            problems.append(f"gate.ece: {ece!r} is not a finite number, so nothing was graded")
        else:
            recomputed = bool(M.gate2_ece_pass(float(ece)))
            if gate.get("passes") != recomputed:
                problems.append(
                    f"gate.passes = {gate.get('passes')!r} but gate2_ece_pass({ece!r}) = "
                    f"{recomputed!r} — the verdict was assigned, not derived"
                )
        if gate.get("gate") != ECE_GATE:
            problems.append(
                f"gate.gate = {gate.get('gate')!r}, must be the blinded-frozen ADR-0005 D11 "
                f"default {ECE_GATE!r}; loosening it needs ADR sign-off (CLAUDE.md §7 item 2)"
            )

    shift = report.get("prior_shift")
    if not isinstance(shift, Mapping) or not shift:
        problems.append("prior_shift: block missing — the non-gated deployment read is absent")
    elif shift.get("pinned_target_prior") is not None:
        problems.append(
            f"prior_shift.pinned_target_prior = {shift.get('pinned_target_prior')!r}: this step "
            "may not pin a deployment prior — the PRD gives a band in prose and a scalar is a "
            "new blinded-frozen default needing ADR-0005 sign-off (CLAUDE.md §7 item 2)"
        )

    ood = report.get("ood")
    if not isinstance(ood, Mapping) or not ood:
        problems.append("ood: block missing — the D13 leave-clade-out read is absent")
    else:
        block = ood.get("d13_adjudication")
        if not isinstance(block, Mapping):
            problems.append("ood.d13_adjudication: block missing")
        else:
            cond_i = block.get("condition_i_drift_bound")
            if isinstance(cond_i, Mapping) and cond_i.get("bound") is not None:
                problems.append(
                    f"ood.d13_adjudication.condition_i_drift_bound.bound = "
                    f"{cond_i.get('bound')!r}: D13's drift bound is unpinned and pinning one "
                    "here would be a new blinded-frozen default (CLAUDE.md §7 item 2)"
                )
            if block.get("calibrated_negative_pass") is not None:
                problems.append(
                    "ood.d13_adjudication.calibrated_negative_pass was decided, but conditions "
                    "(i) and (iii) have no values — a two-of-three verdict is not D13's"
                )
        if ood.get("gated") is not False:
            problems.append(f"ood.gated = {ood.get('gated')!r}, must be False (ADR-0005 D13)")

    written = report.get("clauses")
    recomputed = derive_clauses(report)
    if not isinstance(written, Mapping):
        problems.append("clauses: block missing — nothing pins what the gate checked")
    else:
        if set(written) != set(recomputed):
            problems.append(
                "clauses: key set drifted — written "
                f"{sorted(set(written) - set(recomputed))!r} extra, "
                f"{sorted(set(recomputed) - set(written))!r} missing"
            )
        for name in sorted(set(written) & set(recomputed)):
            if bool(written[name]) != recomputed[name]:
                problems.append(
                    f"clauses.{name} = {written[name]!r} but re-derives to {recomputed[name]!r}"
                )
    expected_science = all(recomputed.get(name) for name in COMPLETENESS_CLAUSES)
    if report.get("is_science") != expected_science:
        problems.append(
            f"is_science = {report.get('is_science')!r} but the completeness clauses give "
            f"{expected_science!r} — a truncated run must not present as a full grade"
        )
    if report.get("overall_pass") != all(recomputed.values()):
        problems.append(
            f"overall_pass = {report.get('overall_pass')!r} but the re-derived clauses give "
            f"{all(recomputed.values())!r}"
        )
    return problems


def build_report(
    *,
    gate: Mapping[str, Any],
    prior_shift: Mapping[str, Any],
    ood: Mapping[str, Any],
    scope: Mapping[str, Any],
    scoring: Mapping[str, Any],
    provenance: Mapping[str, Any],
    written_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the GATE-2 report; every number is passed in already measured."""
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_version_scope": (
            "`schema_version` versions THIS REPORT'S BODY; `provenance.schema_version` "
            f"versions the provenance-record envelope (tbox_finder.provenance."
            f"SCHEMA_VERSION = {PROV.SCHEMA_VERSION!r}) and moves independently. They are "
            "different schemas and are not expected to match."
        ),
        "step": STEP,
        "generated_by": GENERATED_BY,
        "prd": PRD,
        "adr": ADR,
        "env_lock": ENV_LOCK,
        "generated_at_utc": written_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "gated": True,
        "gate": dict(gate),
        "prior_shift": dict(prior_shift),
        "ood": dict(ood),
        "scope": dict(scope),
        "scoring": dict(scoring),
        "provenance": dict(provenance),
        "disclosures": disclosures(gate=gate, ood=ood),
    }
    clauses = derive_clauses(report)
    report["clauses"] = clauses
    report["overall_pass"] = all(clauses.values())
    # DERIVED, never asserted. GATE-4's report can hardcode `is_science: True` because its
    # incomplete path raises and writes no file at all; this one always writes, so a
    # `--max-units` smoke would otherwise land a truncated grade at the phase-exit path
    # flagged as real science ([[cost-knobs-can-certify]]).
    report["is_science"] = all(clauses[name] for name in COMPLETENESS_CLAUSES)
    report["note"] = (
        "`gate.passes` is GATE-2's P3 half: the ADR-0005 D11 in-distribution ECE of the named "
        "posterior against the blinded-frozen 0.05 default. `overall_pass` is stricter — it is "
        "`gate.passes` AND every machinery, completeness and non-gated-ness clause. The FDR half "
        "of GATE-2 is graded at P5 (D12) and is not represented here."
    )
    return report


def disclosures(*, gate: Mapping[str, Any], ood: Mapping[str, Any]) -> list[str]:
    """The caveats a reader of this number must carry, stated where the number is."""
    out = [
        "GATE-2's P3 half grades the PRE-prior-shift named posterior only; the FDR half "
        "(ADR-0005 D12, FDP CI-upper-bound <= 10%) is graded at P5 and is absent here.",
        _UNPINNED_DEPLOYMENT_PRIOR,
        _UNPINNED_DRIFT_BOUND,
        "INHERITED FROM P3-08: the production arm's temperature is fittable only because "
        "ONE calib row of 1,089 is misclassified at the decision boundary; that row is the "
        "entire signal for beta. The no-aux control's calib carve is perfectly separated and "
        "has no temperature at all. ADR-0005 D11 has no degenerate-limit rule, and drafting "
        "one is a P3-exit item.",
        "INHERITED FROM P3-09: the D11 debiasing term was measured to SATURATE at ~0.060 "
        "against a known truth of 0.123 on genuinely miscalibrated data at n in {20, 50, 200} "
        "— i.e. it can under-state a real calibration error at small n. The plug-in ECE is "
        "reported beside the gated value for exactly this reason; certifying the debias term "
        "is a P3-exit / ADR concern flagged in `metrics.binned_ece`'s own docstring.",
        "The leave-clade-out units are ~92% positive (their negatives are the dinucleotide-"
        "shuffled decoys carved from the same held-out records), so each unit's OOD ECE is "
        "dominated by the positive arm and is not a deployment-prevalence read.",
    ]
    conc = (
        gate.get("bin_concentration") if isinstance(gate.get("bin_concentration"), Mapping) else {}
    )
    if isinstance(conc.get("top_bin_share_of_ece"), (int, float)):
        share = 100.0 * float(conc["top_bin_share_of_ece"])
        out.append(
            f"THE GATED MARGIN IS BIN-CONCENTRATED: {share:.1f}% "
            f"of the reported ECE comes from ONE of {conc.get('n_bins')} equal-mass bins, and "
            f"{conc.get('n_bins_with_saturated_accuracy')} bins have accuracy exactly 0 or 1 "
            f"because the posterior is near-separated (the widest bin spans "
            f"{float(conc.get('widest_bin_span', 0.0)):.4f} of the probability axis, i.e. it "
            "absorbs the whole transition region). The pass is honest, and its evidence base "
            "is narrow — the same shape as the single misclassified calib row above."
        )
    calib = gate.get("calibration") if isinstance(gate.get("calibration"), Mapping) else {}
    if isinstance(calib.get("temperature"), (int, float)):
        out.append(
            f"The graded posterior is sigma(z / T) with T = {float(calib['temperature']):.6f}, "
            f"fitted on {calib.get('n_fitted')!r} calib rows and applied unchanged to both the "
            "in-distribution and the leave-clade-out populations."
        )
    macro = ood.get("macro_average")
    if isinstance(macro, Mapping) and isinstance(macro.get("point"), (int, float)):
        out.append(
            f"The leave-clade-out macro-average OOD ECE is {float(macro['point']):.4f} over "
            f"{ood.get('n_units_admissible')!r} admissible held-out orders — REPORTED, not "
            "gated, and not comparable to the 0.05 in-distribution gate."
        )
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figure_data(report: Mapping[str, Any]) -> dict[str, Any]:
    """The plot-ready projection of the report — figures never recompute a number."""
    gate = report.get("gate", {})
    ood = report.get("ood", {})
    units = ood.get("units", {}) if isinstance(ood.get("units"), Mapping) else {}
    return {
        "step": STEP,
        "generated_by": GENERATED_BY,
        "generated_at_utc": report.get("generated_at_utc"),
        "gate": ECE_GATE,
        "reliability": gate.get("reliability", []),
        "in_distribution": {
            "ece": gate.get("ece"),
            "ece_plugin": gate.get("ece_plugin"),
            "ece_ci": gate.get("ece_ci"),
            "n": gate.get("n"),
            "prevalence": gate.get("prevalence"),
            "temperature": (gate.get("calibration") or {}).get("temperature"),
            "passes": gate.get("passes"),
            "bin_concentration": gate.get("bin_concentration"),
        },
        "ood_units": [
            {
                "unit": name,
                "ood_ece": units[name].get("ood_ece"),
                "inadmissible_point": units[name].get("inadmissible_point"),
                "admissible": units[name].get("admissible"),
                "n_positives": units[name].get("n_positives"),
                "n_records": units[name].get("n_records"),
                "phylum": units[name].get("phylum"),
                "ci": units[name].get("ci"),
            }
            for name in sorted(units)
        ],
        "ood_macro_average": ood.get("macro_average"),
        "ood_by_phylum": ood.get("by_phylum"),
        "ood_min_n": ood.get("min_n"),
        # A figure script that reads only this file otherwise has NO field saying the OOD
        # numbers are ungated, while `gate` (0.05) and `in_distribution.passes` sit in the
        # same document — so it could draw the in-distribution threshold across an OOD panel
        # and render a reported quantity as a gate failure. The guards travel with the data.
        "ood_gated": ood.get("gated"),
        "ood_why_not_gated": ood.get("why_not_gated"),
        "ood_by_phylum_is": ood.get("by_phylum_is"),
        "ood_macro_average_is": ood.get("macro_average_is"),
    }


def _fmt(value: Any, spec: str = ".4f") -> str:
    """Format a number for a figure caption, or say it is absent — never raise mid-render.

    A report that could not fit a temperature carries ``None`` here, and that is precisely
    the run whose figure someone wants to look at; a ``TypeError`` at render time turns a
    diagnostic into a crash.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return format(float(value), spec)
    return "n/a"


def plot_figures(
    *,
    figure_data_path: str | Path = DEFAULT_FIGURE_DATA,
    figures_dir: str | Path = FIGURES_DIR,
) -> int:
    """Render the reliability diagram and the per-held-out-order OOD panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(figure_data_path).read_text(encoding="utf-8"))
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ind = data["in_distribution"]

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5", label="perfect calibration")
    rows = [row for row in data["reliability"] if row.get("n")]
    if rows:
        # Each bin is drawn as a SEGMENT over the probability range it actually covers, and
        # the bins are NOT connected to each other. On a near-separated posterior an
        # equal-mass binning puts one bin across almost the whole axis and the rest in a
        # pile at each end; joining their centres with a line draws a smooth curve through
        # territory where nothing was measured, which reads as a well-sampled reliability
        # diagram rather than as the two-point-plus-one-bridge object it is.
        for i, row in enumerate(rows):
            ax.plot(
                [row["p_min"], row["p_max"]],
                [row["acc"], row["acc"]],
                color="tab:blue",
                linewidth=1.6,
                solid_capstyle="butt",
                alpha=0.85,
                label=f"bin (span of p, {_bin_size_label(rows)})" if i == 0 else None,
            )
        ax.scatter(
            [row["conf"] for row in rows],
            [row["acc"] for row in rows],
            s=16,
            color="tab:blue",
            zorder=3,
            label=f"bin (mean p, observed freq) — n={ind['n']}",
        )
        # Saturated bins overplot exactly on top of one another at the two corners, so a
        # reader counts one marker where there are eleven. Label each cluster with how many
        # bins it holds; the clustering is derived from the data, not assumed.
        clusters: dict[tuple[float, float], int] = {}
        for row in rows:
            key = (round(float(row["conf"]), 2), round(float(row["acc"]), 2))
            clusters[key] = clusters.get(key, 0) + 1
        for (cx, cy), count in clusters.items():
            if count < 2:
                continue
            ax.annotate(
                f"x{count} bins",
                (cx, cy),
                textcoords="offset points",
                xytext=(8 if cx < 0.5 else -8, 10 if cy < 0.5 else -14),
                ha="left" if cx < 0.5 else "right",
                fontsize=7.5,
                color="tab:blue",
            )
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed frequency of positives")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=7.5)
    ci = ind.get("ece_ci") or {}
    # `figure_data` derives these from `.get(...)` chains, so any of them can be None on a
    # report that failed to fit — exactly the run whose figure someone would want to look at.
    # The reads below already guard with `isinstance`; these now match.
    subtitle = (
        f"ECE = {_fmt(ind.get('ece'))} (gate {data['gate']}) · "
        f"plug-in {_fmt(ind.get('ece_plugin'))} · T = {_fmt(ind.get('temperature'))}"
    )
    conc = ind.get("bin_concentration") or {}
    if isinstance(conc.get("top_bin_share_of_ece"), (int, float)):
        subtitle += (
            f"\n{100 * float(conc['top_bin_share_of_ece']):.0f}% of the ECE from 1 of "
            f"{conc.get('n_bins')} bins · {conc.get('n_bins_with_saturated_accuracy')} bins "
            "at accuracy 0 or 1 (near-separated posterior)"
        )
    # Both ends, not just `lower`: the guard was asymmetric and the format string reads both.
    if all(
        isinstance(ci.get(end), (int, float)) and math.isfinite(float(ci[end]))
        for end in ("lower", "upper")
    ):
        subtitle += f"\nblock-bootstrap 95% CI [{ci['lower']:.4f}, {ci['upper']:.4f}]"
    fig.suptitle(
        f"GATE-2 in-distribution reliability — {ECE_N_BINS} equal-mass bins, debiased\n{subtitle}",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / "gate2_reliability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    units = sorted(
        data["ood_units"],
        key=lambda u: -((u["ood_ece"] if u["admissible"] else u["inadmissible_point"]) or 0.0),
    )
    fig2, ax2 = plt.subplots(figsize=(6.8, max(3.6, 0.24 * len(units) + 2.6)))
    ypos = list(range(len(units)))
    values = [
        (u["ood_ece"] if u["admissible"] else u["inadmissible_point"]) or float("nan")
        for u in units
    ]
    lower, upper = [], []
    for u, value in zip(units, values, strict=True):
        ci_u = u.get("ci") or {}
        lo, hi = ci_u.get("lower"), ci_u.get("upper")
        ok = isinstance(lo, (int, float)) and isinstance(hi, (int, float))
        lower.append(max(0.0, value - float(lo)) if ok and math.isfinite(float(lo)) else 0.0)
        upper.append(max(0.0, float(hi) - value) if ok and math.isfinite(float(hi)) else 0.0)
    ax2.errorbar(
        values, ypos, xerr=[lower, upper], fmt="none", ecolor="0.4", elinewidth=1, capsize=2
    )
    # Admissible and inadmissible points are drawn with DIFFERENT MARKERS, not just
    # different colours: an inadmissible unit's value is an `inadmissible_point`, which the
    # ADR says supports no verdict, and a reader must not be able to mistake it for a
    # graded one at a glance (or in greyscale).
    keep = [i for i, u in enumerate(units) if u["admissible"]]
    drop = [i for i, u in enumerate(units) if not u["admissible"]]
    if keep:
        ax2.scatter(
            [values[i] for i in keep],
            [ypos[i] for i in keep],
            s=20,
            c="tab:blue",
            zorder=3,
            label="admissible (>= min-N)",
        )
    if drop:
        ax2.scatter(
            [values[i] for i in drop],
            [ypos[i] for i in drop],
            s=26,
            marker="x",
            c="0.55",
            zorder=3,
            label=f"sub-min-N (< {data['ood_min_n']} positives) — inadmissible by rule",
        )
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(
        [f"{u['unit']} · {u.get('phylum') or '?'}  (n+={u['n_positives']})" for u in units],
        fontsize=7,
    )
    ax2.invert_yaxis()
    ax2.set_xlabel("leave-clade-out OOD ECE — REPORTED, not gated")
    macro = data.get("ood_macro_average") or {}
    if isinstance(macro.get("point"), (int, float)):
        ax2.axvline(
            float(macro["point"]),
            color="tab:red",
            linewidth=1,
            linestyle=":",
            label=f"macro-average {macro['point']:.4f} (admissible units)",
        )
    # ABOVE the axes: the units are sorted by drift, so the largest values sit in the upper
    # right and an in-axes legend lands on them, while below the axes it lands on the label.
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=1,
        fontsize=7,
        framealpha=0.0,
        handletextpad=0.4,
    )
    fig2.suptitle(
        "GATE-2 leave-one-order-out calibration drift — NON-GATED (ADR-0005 D13)\n"
        "beta-kernel leave-one-out estimator, cluster-blocked bootstrap CIs",
        fontsize=9,
    )
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    out2 = out_dir / "gate2_ood_by_order.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out2}")
    return 0


# --------------------------------------------------------------------------- #
# TORCH TIER — scoring the leave-clade-out holdout (lazy; nothing above needs it)
# --------------------------------------------------------------------------- #
def score_loo_holdout(
    *,
    dataset: str | Path = DEFAULT_DATASET,
    checkpoint_root: str | Path | None = None,
    sweep_dir: str | Path | None = None,
    out_path: str | Path = DEFAULT_LOO_SCORES,
    batch_size: int = 4,
    device: str | None = None,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Score the D5 leave-one-order-out holdout with the **production** arm.

    The checkpoint loader and the scorer are P3-08's, imported rather than rebuilt — that
    keeps the adapter-really-loaded verification, the ``(n_tokens, row_id)`` batch ordering
    and the tokenizer adapter on one implementation
    ([[promote-dont-duplicate-is-a-correctness-rule]]).
    Only the *row set* differs, and it is the one thing this function decides.
    """
    import torch

    from tbox_finder.stage2 import eval as E

    rows, census = loo_holdout_rows(dataset, with_sequences=True)
    if census["n_row_overlap"] or census["n_order_overlap"]:
        raise ValueError(
            "the designated leave-one-order-out holdout is not disjoint from Stage-2 training "
            f"({census['n_row_overlap']} shared rows, {census['n_order_overlap']} shared orders: "
            f"{census['overlapping_orders']!r}) — scoring it would produce an in-distribution "
            "number labelled OOD"
        )

    arms = E.discover_arms(
        checkpoint_root or E.DEFAULT_CKPT_ROOT, sweep_dir=sweep_dir or E.DEFAULT_SWEEP_DIR
    )
    production = E.production_arm_config()
    arm, _ = E.select_arm_pair(arms, production=production)
    trained_under = arms[arm].get("attn_implementation")
    backend = attn_implementation or trained_under
    if backend is None:
        raise RuntimeError(
            f"arm {arm}'s run report records no attention backend, so scoring cannot reproduce "
            "the numerics it trained under"
        )
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, record = E.load_stage2_checkpoint(
        arms[arm]["checkpoint_path"], attn_implementation=backend, device=target
    )
    record["attn_implementation_trained_under"] = trained_under
    record["attn_implementation_matches_training"] = bool(backend == trained_under)
    scored = E.score_rows(model, rows, batch_size=batch_size, device=target)
    del model
    if target != "cpu":
        torch.cuda.empty_cache()

    payload = {
        "step": STEP,
        "generated_by": f"{__name__}.score_loo_holdout",
        "population": (
            "ADR-0004 D5 designated leave-one-order-out holdout " "(is_designated_loo_holdout)"
        ),
        "dataset": census["dataset"]["path"],
        "dataset_sha256": census["dataset"]["sha256"],
        "device": target,
        "batch_size": int(batch_size),
        "row_ids": [str(row[_ROW_ID]) for row in rows],
        "labels": [int(bool(row[_LABEL])) for row in rows],
        "units": [row["_unit"] for row in rows],
        "blocks": [row["_block"] for row in rows],
        "disjointness": census,
        "arms": {arm: {"logits": [float(s["tbox_logit"]) for s in scored], "load": record}},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} — {len(rows)} rows, arm {arm}, device {target}")
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="P3-10 GATE-2 in-distribution ECE")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score-loo", help="score the leave-one-order-out holdout (GPU)")
    score.add_argument("--dataset", default=DEFAULT_DATASET)
    score.add_argument("--checkpoint-root", default=None)
    score.add_argument("--sweep-dir", default=None)
    score.add_argument("--out", default=DEFAULT_LOO_SCORES)
    score.add_argument("--batch-size", type=int, default=4)
    score.add_argument("--device", default=None)
    score.add_argument("--attn-implementation", default=None)

    grade = sub.add_parser("grade", help="build reports/gate2_p3_ece.json")
    grade.add_argument("--dataset", default=DEFAULT_DATASET)
    grade.add_argument("--scores", default=DEFAULT_SCORES)
    grade.add_argument("--loo-scores", default=DEFAULT_LOO_SCORES)
    grade.add_argument("--report", default=DEFAULT_REPORT)
    grade.add_argument("--figure-data", default=DEFAULT_FIGURE_DATA)
    grade.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    grade.add_argument("--ood-n-boot", type=int, default=DEFAULT_OOD_N_BOOT)
    grade.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    grade.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="SMOKE ONLY — truncates the held-out orders; the completeness clause then fails",
    )

    project = sub.add_parser(
        "figure-data", help="re-derive the figure-data JSON from a committed report"
    )
    project.add_argument("--report", default=DEFAULT_REPORT)
    project.add_argument("--out", default=DEFAULT_FIGURE_DATA)

    plot = sub.add_parser("plot-figures", help="render the figures from the figure-data JSON")
    plot.add_argument("--figure-data", default=DEFAULT_FIGURE_DATA)
    plot.add_argument("--figures-dir", default=FIGURES_DIR)
    return parser


def write_outputs(
    report: Mapping[str, Any],
    *,
    report_path: str | Path,
    figure_data_path: str | Path,
    valid: bool,
) -> tuple[Path, Path]:
    """Write the report and its figure data, diverting **both** when the report is invalid.

    They divert together on purpose. ``plot_gate2_figures`` consumes exactly
    ``figure_data_path``, so writing figure data derived from a *rejected* report to that
    path would publish figures for a grade nothing accepted — beside a report that was
    correctly diverted, which reads as a rendering bug rather than a refused result. Inside
    the workflow Snakemake deletes a failed job's outputs and masks it; a direct CLI run
    does not. Kept as one function so the *pairing* is testable: checking ``_output_path``
    alone passes even when a call site stops using it.
    """
    written: list[Path] = []
    for canonical, payload in (
        (Path(report_path), dict(report)),
        (Path(figure_data_path), figure_data(report)),
    ):
        target = _output_path(canonical, valid=valid)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not valid and canonical.exists():
            # Diverting is not enough on a RE-run. A previously accepted artifact would stay
            # at the consumer path with older numbers and older provenance, so a reader of
            # `DEFAULT_REPORT` gets a stale grade sitting beside a fresh `.invalid.json` —
            # the exact outcome "never written to the path a consumer reads" exists to
            # prevent. Remove the canonical file so the absence is the signal.
            canonical.unlink()
        if valid:
            # The mirror case, and the same ambiguity in the other direction: a previous
            # run's diverted artifact left beside a freshly accepted one shows a reader a
            # rejected grade with older numbers next to the accepted report.
            _output_path(canonical, valid=False).unlink(missing_ok=True)
        written.append(target)
    return written[0], written[1]


def _output_path(report_path: str | Path, *, valid: bool) -> Path:
    """An invalid report is diverted, never written to the path a consumer reads."""
    path = Path(report_path)
    return path if valid else path.with_suffix(".invalid.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "score-loo":
        score_loo_holdout(
            dataset=args.dataset,
            checkpoint_root=args.checkpoint_root,
            sweep_dir=args.sweep_dir,
            out_path=args.out,
            batch_size=args.batch_size,
            device=args.device,
            attn_implementation=args.attn_implementation,
        )
        return 0

    if args.command == "figure-data":
        # `figure_data` is a pure projection of the report, so re-deriving it from a
        # committed report is byte-identical to what `grade` wrote — and costs seconds
        # instead of re-running the leave-one-out bootstrap. A test pins that equality.
        source = json.loads(Path(args.report).read_text(encoding="utf-8"))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(figure_data(source), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {out} (re-derived from {_recorded_path(args.report)})")
        return 0

    if args.command == "plot-figures":
        return plot_figures(figure_data_path=args.figure_data, figures_dir=args.figures_dir)

    from tbox_finder.stage2 import eval as E

    split_rows, dataset_meta = _read_split_table(args.dataset)
    production = E.production_arm_config()
    arms = E.discover_arms(E.DEFAULT_CKPT_ROOT, sweep_dir=E.DEFAULT_SWEEP_DIR)
    arm, _ = E.select_arm_pair(arms, production=production)

    in_dist = load_scores(args.scores, arm)
    loo = load_scores(args.loo_scores, arm)

    keys, _ = E.block_keys(split_rows)
    blocks_by_row = {row[_ROW_ID]: key for row, key in zip(split_rows, keys, strict=True)}

    holdout, census = loo_holdout_rows(args.dataset)
    rows_by_id = {row[_ROW_ID]: row for row in holdout}

    gate = grade_in_distribution(
        scores=in_dist,
        blocks_by_row=blocks_by_row,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    temperature = float(gate["calibration"]["temperature"])
    shift = prior_shift_band_sweep(
        scores=in_dist,
        source_prior=float(gate["calibration"]["calib_prevalence"]),
        temperature=temperature,
    )
    ood = grade_ood_units(
        scores=loo,
        rows_by_id=rows_by_id,
        temperature=temperature,
        n_boot=args.ood_n_boot,
        seed=args.seed,
        max_units=args.max_units,
    )

    overlap = set(in_dist["row_ids"]) & set(loo["row_ids"])
    in_by_id = dict(zip(in_dist["row_ids"], in_dist["logits"], strict=True))
    loo_by_id = dict(zip(loo["row_ids"], loo["logits"], strict=True))
    deltas = [abs(in_by_id[r] - loo_by_id[r]) for r in sorted(overlap)]

    scope = {
        "gate_rung": GATE_RUNG,
        "n_gate_rung_rows_in_split_table": len(gate_rung_row_ids(args.dataset)),
        "n_loo_holdout_rows_in_split_table": census["n_holdout_rows"],
        "n_loo_holdout_rows_scored": len(loo["row_ids"]),
        "n_loo_holdout_units_in_split_table": census["n_units"],
        "n_row_overlap": census["n_row_overlap"],
        "n_order_overlap": census["n_order_overlap"],
        "overlapping_orders": census["overlapping_orders"],
        "n_training_rows": census["n_training_rows"],
        "n_training_orders": census["n_training_orders"],
        "n_holdout_orders": census["n_holdout_orders"],
        "rescore_n_overlap": len(overlap),
        "rescore_max_abs_delta": max(deltas) if deltas else None,
        "rescore_mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "rescore_tolerance": RESCORE_AGREEMENT_TOL,
        "rescore_note": (
            "the same rows scored in two different row sets: bf16 + flash-attention reductions "
            "depend on batch composition, so this is an agreement bound, not an equality"
        ),
        "max_units": args.max_units,
        "dataset": dataset_meta,
    }
    scoring = {
        "arm": arm,
        "in_distribution_scores": _recorded_path(args.scores),
        "loo_scores": _recorded_path(args.loo_scores),
        "in_distribution_device": in_dist["meta"].get("device"),
        "loo_device": loo["meta"].get("device"),
        "batch_size": loo["meta"].get("batch_size"),
        "n_boot": int(args.n_boot),
        "ood_n_boot": int(args.ood_n_boot),
        "bootstrap_seed": int(args.seed),
        "load": loo["load"],
    }
    # `outputs` is deliberately EMPTY. `build_provenance` *hashes* every path it is given,
    # and neither output exists yet — the report is what this call is being embedded into,
    # so hashing it here is both impossible and self-referential. The declared paths go in
    # as names under `extra` instead; the content hash of a committed report is `git`'s job.
    prov = PROV.build_provenance(
        rule="workflow/rules/calibration.smk :: gate2_ece",
        script=GENERATED_BY,
        seed=int(args.seed),
        inputs=[args.dataset, args.scores, args.loo_scores],
        env_lock=ENV_LOCK,
        adr=ADR,
        extra={"declared_outputs": [_recorded_path(args.report), _recorded_path(args.figure_data)]},
    )
    # `build_provenance` keys `inputs` by the string it was handed, and it must be handed a
    # path it can actually OPEN — which under a linked worktree means an absolute one. So the
    # keys are normalised here, AFTER hashing: the sha256 stays the file's own, only the
    # locator loses this machine's home directory and checkout layout.
    prov["inputs"] = {_recorded_path(k): v for k, v in prov["inputs"].items()}

    report = build_report(
        gate=gate, prior_shift=shift, ood=ood, scope=scope, scoring=scoring, provenance=prov
    )
    problems = validate_report(report)
    out, _ = write_outputs(
        report, report_path=args.report, figure_data_path=args.figure_data, valid=not problems
    )

    print(f"wrote {out}")
    print(
        f"GATE-2 in-distribution ECE = {gate['ece']:.6f} "
        f"(gate {ECE_GATE}) — passes={gate['passes']}"
    )
    print(f"overall_pass = {report['overall_pass']}")
    for name, ok in sorted(report["clauses"].items()):
        if not ok:
            print(f"  FALSE clause: {name}")
    for problem in problems:
        print(f"  REPORT PROBLEM: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
