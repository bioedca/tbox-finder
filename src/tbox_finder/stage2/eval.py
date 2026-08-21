"""P3-08 — the with/without-aux ablation on the **calibrated** Stage-2 binary head.

PRD §11 (with/without-aux check) · §12 (calibration) · §2.3 GATE-2 ·
ADR-0005 D11 (named posterior, 15 equal-mass debiased bins, ECE ≤ 0.05) + D16
(aux weighting; "the aux heads do not degrade the calibrated primary head's GATE-2
grade") · ADR-0004 A7 (``fold_random == "test"`` is the split GATE-2 is graded on;
``calib`` is where ``T`` is fitted and nowhere else).

This module also ships **the Stage-2 score producer**, which P3-07 deferred here.
Before it, nothing in this repo read a Stage-2 checkpoint back: ``stage2_heads.pt``
and ``lora_adapter/`` had write sites and zero readers, and
``stage2.train.evaluate`` thresholds ``tbox_logit`` at zero (``train.py:1161``) and
keeps only a scalar accuracy — so per-row binary logits, the input the whole
calibration stack eats, did not exist. Everything below the ``TORCH TIER`` banner
is that producer; everything above it is pure numpy and grades in bare CI.

Three things about this file are load-bearing and easy to get wrong:

**1. The gate's threshold never existed, and the §7 stop it raised is SETTLED.**
``imp.md`` grades "beyond a pre-registered tolerance"; no such tolerance is pinned
in ``PRD.md``, in any ADR, in ``conf/`` or in ``src/``. D16's own sentence admits
two readings — an *absolute* one (the with-aux arm must still clear the D11 gate,
ECE ≤ 0.05: no new number needed) and a *delta* one (with-aux must not be worse
than no-aux by more than τ: τ undefined). :func:`compare_arms` computes both and
reports the exact τ-window on which they disagree. Picking τ is a new
blinded-frozen default amending ADR-0005 D16 — CLAUDE.md §7 item 2, a user
decision, not a config value. ``ablation.tolerance`` is therefore ``None`` and
**User sign-off (AskUserQuestion, 2026-08-03) settled it on the absolute
reading**, so no τ is pinned and no ADR is amended. Both readings are still
computed and the τ-window on which they would disagree is still published, because
a decision is only auditable beside the alternative it rejected.
``ablation.reading_delta.tolerance`` stays ``None`` and the validator **refuses** any
report that pins one; ``ablation.verdict`` is *derived* from the governing reading's
evidence by :func:`verdict_from_absolute_reading`, never assigned.

**2. AUPRC is rank-based, so temperature scaling cannot move it — but the
posterior can still lose the ranking.** ``z ↦ z/T`` is strictly monotone for
``T > 0``, so average precision on the scaled logits is *exactly* AP on the raw
ones, and that identity is a gate clause rather than a comment. The posterior is a
different story: at this model's separation ``σ(z/T)`` saturates to exactly 1.0 for
many rows, which manufactures ties that the logits never had. So AUPRC is graded on
**logits**, ECE on the **posterior**, and the saturation count is recorded — a
report that graded ranking on a saturated posterior would silently understate it.

**3. "The adapter loaded" is measured, never assumed.**
``PeftModel.from_pretrained`` *warns* on missing adapter keys, and PEFT initialises
``lora_B`` to zero — so an adapter that silently fails to load is a mathematical
no-op, i.e. the bare pre-trained backbone, scoring away and producing a complete,
plausible, entirely meaningless report. :func:`load_stage2_checkpoint` therefore
compares every adapter and head tensor **bit-exactly against the file on disk** and
counts non-zero ``lora_B`` blocks; both are gate clauses.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder import metrics as M
from tbox_finder import power as PW
from tbox_finder import provenance as PROV
from tbox_finder.calib import recalibrate as R
from tbox_finder.stage2 import train as T

__all__ = [
    "ArmScores",
    "aux_ablation_check",
    "block_keys",
    "build_report",
    "compare_arms",
    "derive_clauses",
    "discover_arms",
    "grade_arm",
    "load_stage2_checkpoint",
    "production_arm_config",
    "repo_relative",
    "ranking_preserved",
    "reconcile_cached_scores",
    "verdict_from_absolute_reading",
    "rungs_for_rows",
    "score_rows",
    "select_arm_pair",
    "validate_report",
]

SCHEMA_VERSION = "1"
STEP = "P3-08"
ENTRYPOINT = "tbox_finder.stage2.eval"
RULE = f"{ENTRYPOINT}::aux_ablation_check"

# Paths: taken from the producer that wrote them (P3-06), never retyped here — a
# second copy of `lora_adapter` would go stale with nothing failing.
ENV_LOCK = T.ENV_LOCK
DEFAULT_DATASET = T.DEFAULT_DATASET
DEFAULT_CKPT_ROOT = T.DEFAULT_CKPT_DIR
ADAPTER_SUBDIR = T.ADAPTER_SUBDIR
HEADS_STATE_NAME = T.HEADS_STATE_NAME

DEFAULT_REPORT = "reports/stage2_aux_ablation.json"
#: Per-row logits, written beside the report. Not a cache convenience: these ARE the
#: score producer's product, they cost ~6 GPU-minutes an arm to regenerate, and P3-09
#: (OOD ECE) and P3-10 (the GATE-2 grade) both need exactly this table. Writing them
#: means a grading change is re-run in seconds against the same measured scores rather
#: than re-scoring and quietly getting slightly different bf16 numbers.
DEFAULT_SCORES = "reports/p3/stage2_scores.json"
DEFAULT_SWEEP_DIR = "reports/p3/sweep"
LOSS_CONF = "conf/loss/stage2.yaml"
OPTIM_CONF = "conf/optim/stage2.yaml"

#: PRD §11 — the config ladder is climbed on validation, **never** on the test set.
SELECT_RUNG = "val"
#: ADR-0004 A7 / ADR-0005 D11 — the in-distribution split GATE-2's ECE is graded on.
GRADE_RUNG = "test"
#: ADR-0005 D11 — ``T`` is fitted here and on nothing else.
CALIB_RUNG = R.CALIB_RUNG
#: The three rungs this step reads. ``train`` is deliberately absent: the arms were
#: fitted on it, so every number computed there is in-sample and means nothing.
SCORED_RUNGS: tuple[str, ...] = (CALIB_RUNG, SELECT_RUNG, GRADE_RUNG)

#: ADR-0005 D11, via the one implementation (`metrics`); not re-pinned here.
ECE_N_BINS = M.ECE_N_BINS
ECE_GATE = PW.ECE_GATE

DEFAULT_N_BOOT = 2000
BOOTSTRAP_SEED = 20260803

#: ADR-0005 D17's RiNALMo→RNA-FM **backbone-swap** trigger margins. They are the only
#: pre-registered numbers in this repo shaped like "a ΔECE / ΔAUPRC between two arms",
#: which makes them the obvious candidate a §7 sign-off might adopt for D16 — and
#: exactly for that reason they are carried as a **labelled non-governing reference**,
#: never applied. `tests/unit/test_aux_ablation_check.py` re-reads them out of the ADR
#: so this copy cannot drift away from the decision it quotes.
D17_REFERENCE_MARGINS: dict[str, float] = {"ece": 0.02, "auprc": 0.03}
#: **The governing reading of ADR-0005 D16, settled by user sign-off (AskUserQuestion,
#: 2026-08-03).** D16 says the aux heads must not degrade "the calibrated primary head's
#: GATE-2 grade", and the GATE-2 grade (D11) is a pass/fail: ECE <= 0.05. So the check is
#: whether the with-aux arm still HOLDS that grade — which needs no new number and no ADR
#: amendment. The delta against the no-aux arm stays **reported, not gated**; the
#: never-pre-registered tau is therefore not needed and is still refused by the validator.
GOVERNING_READING = "absolute"
GOVERNING_READING_SOURCE = (
    "ADR-0005 D16 read against D11; user sign-off (AskUserQuestion, 2026-08-03) — "
    "the absolute reading governs, the delta is reported not gated, no tolerance is pinned"
)
#: Verdicts `compare_arms` may emit. Each is DERIVED from the governing reading's own
#: evidence; none is assignable by a caller.
VERDICT_HOLDS = "aux_does_not_degrade_the_gate2_grade"
VERDICT_DEGRADES = "aux_degrades_the_gate2_grade"
VERDICT_UNDECIDABLE = "undecidable_with_aux_arm_has_no_ece"
VERDICTS = (VERDICT_HOLDS, VERDICT_DEGRADES, VERDICT_UNDECIDABLE)

D17_REFERENCE_SOURCE = (
    "ADR-0005 D17 — RiNALMo→RNA-FM backbone-swap trigger; NOT the D16 aux-ablation bar"
)

_ROW_ID = "row_id"
_LABEL = "is_tbox"
_SEQUENCE = "rna_sequence"
_CLUSTER = "cluster_id"
_CALIB = "calib"
_FOLD_RANDOM = "fold_random"


# --------------------------------------------------------------------------- #
# Row-side derivations (pure)
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _candidate_roots() -> tuple[Path, ...]:
    """This checkout, and the main checkout when this one is a linked worktree.

    Both are needed because a worktree under ``.claude/worktrees/`` has its own root
    while the DVC-materialised inputs (checkpoints, the dataset parquet) live in the
    main checkout — so relativising against only ``__file__``'s root silently leaves
    every input path absolute, which is how the developer-path leak survived its first
    fix.
    """
    roots = [_REPO_ROOT]
    try:
        import subprocess

        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            check=False,
        ).stdout.strip()
        if common:
            roots.append(Path(common).resolve().parent)
    except OSError:  # pragma: no cover - git absent
        pass
    return tuple(dict.fromkeys(roots))


def repo_relative(path: str | Path) -> str:
    """A path as the repository sees it — never as this laptop does.

    Every path this module records lands in a committed artifact in a **public** repo,
    where an absolute path contributes nothing a reader can use and permanently
    publishes the OS user name and the local directory layout
    ([[committing-real-tool-output-fixtures]]). The sha256 beside each path is the
    identity evidence; the string is only a locator. A path outside the repo is
    returned unchanged rather than mangled into `../../..`.
    """
    resolved = Path(path).resolve()
    for root in _candidate_roots():
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(path)


def rungs_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Per-row rung, re-derived through P3-07's own :func:`recalibrate.rung_labels`.

    Delegating rather than re-deriving is the point: ``assign_rung`` refuses a
    ``calib`` row that is not on the ``train`` fold (the P3-02 disjointness
    invariant), refuses an unparseable ``calib`` flag instead of taking
    ``bool(nan)``, and refuses an unknown ``fold_random`` token rather than
    filtering it away. A local re-implementation would have to re-earn all four.
    """
    return R.rung_labels(
        calib=[row.get(_CALIB) for row in rows],
        fold_random=[row.get(_FOLD_RANDOM) for row in rows],
    )


def block_keys(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Bootstrap block per row: its homology cluster, or itself when it has none.

    PRD §12 resamples at the homology-cluster level. The decoys have no cluster —
    they are *built*, not carved out of the corpus, so they share no ancestry with
    anything and each is its own block.

    ``cluster_id`` is a float column and the cluster-less rows carry NaN, so the
    naive ``set(col)`` would fold every unrelated decoy into **one** enormous block
    and silently destroy the resampling ([[nulls-inflate-block-counts]]). The census
    returned beside the keys reports both populations separately so a reader can see
    which happened.
    """
    keys: list[str] = []
    n_clustered = 0
    n_singleton = 0
    for row in rows:
        raw = row.get(_CLUSTER)
        # Normalise FIRST, then branch. `isinstance(raw, float)` is true for np.float64
        # (which this column happens to be today) and false for np.float32 — and under
        # that reading a float32 NaN is "clustered", keyed `cluster:nan`, and every
        # unrelated decoy collapses into one block: precisely the failure the docstring
        # above is about. A non-numeric id is a schema fault and is refused by name
        # rather than quietly taking the cluster-less path.
        numeric: float | None = None
        if raw is not None:
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"row {row.get(_ROW_ID)!r}: {_CLUSTER} is {raw!r}, which is neither a "
                    "number nor absent, so it can be neither a block key nor a singleton"
                ) from exc
            if math.isnan(numeric):
                numeric = None
        if numeric is not None:
            keys.append(f"cluster:{int(numeric) if numeric.is_integer() else numeric}")
            n_clustered += 1
        else:
            row_id = row.get(_ROW_ID)
            if row_id is None:
                raise ValueError("a cluster-less row carries no row_id, so it has no block")
            keys.append(f"row:{row_id}")
            n_singleton += 1
    census = {
        "n_rows": len(keys),
        "n_rows_with_cluster": n_clustered,
        "n_rows_without_cluster": n_singleton,
        "n_blocks": len(set(keys)),
        "n_blocks_from_clusters": len({k for k in keys if k.startswith("cluster:")}),
        "n_singleton_blocks": n_singleton,
    }
    return keys, census


@dataclass(frozen=True)
class ArmScores:
    """One arm's per-row binary-head logits over :data:`SCORED_RUNGS`.

    ``logits`` are the **raw** head outputs — un-scaled, un-sigmoided. Everything
    downstream derives from them, so the calibration stack is applied exactly once,
    in one place, in the ADR-0005 D11 order.
    """

    arm: str
    row_ids: tuple[str, ...]
    logits: np.ndarray
    labels: np.ndarray
    rungs: tuple[str, ...]
    blocks: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.row_ids)
        for name, seq in (
            ("logits", self.logits),
            ("labels", self.labels),
            ("rungs", self.rungs),
            ("blocks", self.blocks),
        ):
            if len(seq) != n:
                raise ValueError(f"{self.arm}: {name} has {len(seq)} entries but row_ids has {n}")
        if n == 0:
            raise ValueError(f"{self.arm}: no rows were scored")
        if len(set(self.row_ids)) != n:
            n_dups = n - len(set(self.row_ids))
            raise ValueError(f"{self.arm}: row_ids are not unique ({n_dups} duplicated)")
        unknown = sorted(set(self.rungs) - set(R.RUNG_VOCABULARY))
        if unknown:
            raise ValueError(f"{self.arm}: unknown rung token(s) {unknown}")
        # These are constructed from external JSON by `reconcile_cached_scores`, so the
        # logits are only as well-formed as the file. A nested list still satisfies the
        # length check above (a (n, k) array has len n), and NaN/Infinity would flow
        # straight into the fit — where `temperature_scale` would refuse them, but far
        # from the file that carried them.
        logits = np.asarray(self.logits)
        if logits.ndim != 1:
            raise ValueError(
                f"{self.arm}: logits must be 1-D per-row values, got shape {logits.shape}"
            )
        if not np.all(np.isfinite(logits)):
            n_bad = int(np.count_nonzero(~np.isfinite(logits)))
            raise ValueError(f"{self.arm}: {n_bad} of {logits.size} logits are not finite")

    def index_of(self, rung: str) -> np.ndarray:
        return np.asarray([i for i, r in enumerate(self.rungs) if r == rung], dtype=np.int64)

    def census(self) -> dict[str, int]:
        return {rung: int(sum(1 for r in self.rungs if r == rung)) for rung in R.RUNG_VOCABULARY}


def _blockwise(values: Sequence[Any], blocks: Sequence[str]) -> list[list[Any]]:
    """Group ``values`` by block, preserving first-seen block order (determinism)."""
    grouped: dict[str, list[Any]] = {}
    for value, block in zip(values, blocks, strict=True):
        grouped.setdefault(block, []).append(value)
    return list(grouped.values())


def _saturation(posterior: np.ndarray) -> dict[str, int]:
    return {
        "n_at_one": int(np.count_nonzero(posterior >= 1.0)),
        "n_at_zero": int(np.count_nonzero(posterior <= 0.0)),
    }


# --------------------------------------------------------------------------- #
# Grading one arm (pure)
# --------------------------------------------------------------------------- #


def ranking_preserved(
    z_raw: Any, z_scaled: Any, auprc: float, auprc_scaled: float
) -> tuple[bool, int, float]:
    """Did ``z -> z/T`` preserve the ranking? Returns ``(preserved, n_ties, ap_delta)``.

    A free function, and unit-tested as one, because the interesting case cannot be
    reached through :func:`grade_arm`: scaling can collapse two distinct logits onto one
    float64 **without** moving average precision at all (when the two rows share a
    label). Comparing only the AP values calls that preserved; it is not — ranking
    information was destroyed, and the next arm it happens to could lose real ordering.
    Both conditions are required, and both counts are returned so a False verdict says
    *which* one fired.
    """
    n_ties = int(len(set(np.asarray(z_raw).tolist())) - len(set(np.asarray(z_scaled).tolist())))
    ap_delta = float(abs(auprc - auprc_scaled))
    return (n_ties == 0 and ap_delta == 0.0), n_ties, ap_delta


def separation_census(scores: ArmScores, rung: str) -> dict[str, Any]:
    """How separable a rung is at the decision boundary — the fact that kills the fit.

    A temperature exists only when the calib set contains at least one row the head
    puts on the wrong side of zero. With every row already its own arg-max the exact
    minimiser of the NLL is ``beta -> inf`` (``T -> 0``) and there is nothing to
    estimate; ``fit_temperature`` says so rather than returning a bracket endpoint.
    This census is what turns that refusal from an error message into evidence — it
    reports *how far* from fittable the set is, and how big the margins are.
    """
    idx = scores.index_of(rung)
    if idx.size == 0:
        # Uniform key set, with `None` — not 0 — for everything that was not measured.
        # An empty rung is reachable (grade_arm records calib_separation even when the
        # fit refused for an EMPTY calib rung), so a consumer reading
        # `is_perfectly_separated` must not hit a KeyError; and
        # `n_misclassified_at_zero: 0` would read as a measured result, which is the
        # exact substitution this module refuses to make for `ece`.
        return {
            "n": 0,
            "n_positive": 0,
            "n_misclassified_at_zero": None,
            "accuracy_at_zero": None,
            "is_perfectly_separated": None,
            "min_abs_logit": None,
            "median_abs_logit": None,
            "max_abs_logit": None,
            "min_logit": None,
            "max_logit": None,
        }
    z = np.asarray(scores.logits, dtype=np.float64)[idx]
    y = np.asarray(scores.labels)[idx]
    predicted = (z > 0).astype(np.int64)
    wrong = int(np.count_nonzero(predicted != y))
    margin = np.abs(z)
    return {
        "n": int(idx.size),
        "n_positive": int(np.count_nonzero(y == 1)),
        "n_misclassified_at_zero": wrong,
        "accuracy_at_zero": float(1.0 - wrong / idx.size),
        "is_perfectly_separated": wrong == 0,
        "min_abs_logit": float(margin.min()),
        "median_abs_logit": float(np.median(margin)),
        "max_abs_logit": float(margin.max()),
        "min_logit": float(z.min()),
        "max_logit": float(z.max()),
    }


def _classify_refusal(exc: Exception) -> str:
    """Name the refusal so a reader (and a clause) can tell the three cases apart."""
    text = str(exc)
    if "arg-max" in text or "beta -> inf" in text:
        return "perfect_separation_beta_to_infinity"
    if "single-class" in text:
        return "single_class_calib_rung"
    # Parenthesised deliberately (r4): `and` binds tighter than `or`, so the unbracketed
    # form reads as `"no rows" in text or (...)` and would label ANY "no rows" refusal —
    # including one about a graded rung — as an empty *calib* rung. The branch name
    # claims calib in both readings, so both readings must require it.
    if ("no rows" in text or "zero" in text) and "calib" in text:
        return "empty_calib_rung"
    return "other"


def _uncalibrated_grade(
    scores: ArmScores,
    rung: str,
    *,
    n_bins: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """What survives when no temperature exists — and nothing that does not.

    ``ece`` is ``None``, not a number. The gated quantity is the ECE of the
    *temperature-scaled* posterior; without ``T`` that object does not exist, and
    putting the ``T = 1`` value in its place would be a fabricated metric wearing the
    gated key's name (§10.3). Average precision **is** reported, because it is
    computed from the ranking alone and a monotone rescaling could not have changed it.
    """
    idx = scores.index_of(rung)
    if idx.size == 0:
        raise ValueError(
            f"{scores.arm}: no rows on the {rung!r} rung — a grade computed on nothing "
            f"reads exactly like a passing one (census {scores.census()})"
        )
    y = [int(v) for v in np.asarray(scores.labels)[idx]]
    z = np.asarray(scores.logits, dtype=np.float64)[idx]
    blocks = [scores.blocks[i] for i in idx]
    auprc = M.average_precision(y, [float(v) for v in z])
    rank_pairs = list(zip(y, [float(v) for v in z], strict=True))
    auprc_ci = M.block_bootstrap_ci(
        _blockwise(rank_pairs, blocks),
        lambda sample: M.average_precision([a for a, _ in sample], [b for _, b in sample]),
        n_boot=n_boot,
        seed=seed,
    )
    p_raw = np.asarray(R.posterior_from_logits(z), dtype=np.float64)
    n_pos = int(sum(y))
    return {
        "n": int(idx.size),
        "n_positive": n_pos,
        "n_negative": int(idx.size) - n_pos,
        "prevalence": n_pos / int(idx.size),
        "ece": None,
        "ece_gate_pass": None,
        "ece_unavailable_reason": (
            "no temperature could be fitted on the calib rung, so the ADR-0005 D11 named "
            "posterior (temperature-scaled, pre-prior-shift) does not exist for this arm"
        ),
        "ece_n_bins": int(n_bins),
        "ece_binning": "equal_mass",
        "ece_debiased": True,
        "ece_gate": ECE_GATE,
        "auprc": auprc,
        "auprc_scaled_logits": auprc,
        "auprc_on_posterior": None,
        # No temperature exists, so no scaling was applied and no tie could be created
        # by one. Recorded explicitly, with the same keys as the fitted path, so a
        # consumer reads a number rather than a missing key it has to interpret.
        "auprc_rank_invariant": True,
        "n_ties_created_by_scaling": 0,
        "auprc_scaling_abs_delta": 0.0,
        "no_scaling_applied": True,
        "auprc_baseline_prevalence": n_pos / int(idx.size),
        "auprc_ci": auprc_ci,
        "graded_on": "RAW logits — the named posterior does not exist for this arm",
        "separation": separation_census(scores, rung),
        "uncalibrated_diagnostic": {
            "note": (
                "sigma(z) at T=1. NOT the GATE-2 object and NOT comparable to the 0.05 "
                "gate; recorded so the miscalibration that the missing T would have "
                "corrected is at least visible."
            ),
            "ece_at_T1": M.binned_ece(y, [float(v) for v in p_raw], n_bins, debias=True),
            "posterior_saturation": _saturation(p_raw),
        },
        "block_census": {"n_blocks": len(set(blocks)), "n_rows": int(idx.size)},
    }


def grade_arm(
    scores: ArmScores,
    *,
    graded_rungs: Sequence[str] = (SELECT_RUNG, GRADE_RUNG),
    n_bins: int = ECE_N_BINS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fit ``T`` on ``calib``, then grade the **named posterior** on each graded rung.

    "Fitted on calib and nowhere else" is structural, not a convention: this hands
    :func:`recalibrate.temperature_scale` the whole table plus the per-row rung and
    lets *it* select — no argument of this function can admit a graded row into the
    fit. Its emptiness, unknown-token and single-class refusals therefore all apply.

    **When the fit refuses, the refusal is recorded and the ECE is ``None``.** It is
    never replaced by ``T = 1``, and no number is put where the gated quantity would
    have gone: a substituted temperature is indistinguishable from a measured one
    (§10.3), and "ECE is 0.03 at T=1" answers a question GATE-2 did not ask. What can
    still be computed without a temperature *is* — average precision is rank-based, so
    it neither needs ``T`` nor would move under one — and it is reported under
    ``uncalibrated`` so nothing downstream can mistake it for the named posterior.
    """
    fit = None
    refusal: dict[str, Any] | None = None
    try:
        fit = R.temperature_scale(scores.logits, scores.labels, rung=scores.rungs)
    except (R.TemperatureFitError, ValueError) as exc:
        refusal = {
            "fitted": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "classification": _classify_refusal(exc),
        }

    if fit is None:
        return {
            "arm": scores.arm,
            "calibration": {
                "fitted": False,
                "temperature": None,
                "fitted_on": CALIB_RUNG,
                "refusal": refusal,
                "calib_separation": separation_census(scores, CALIB_RUNG),
            },
            "stack": {
                "stack_order": list(R.STACK_ORDER),
                "stack_applied": ["train"],
                "gated_posterior_key": R.NAMED_POSTERIOR_KEY,
                "prior_shift_applied": False,
                "named_posterior_exists": False,
            },
            "scored_census": scores.census(),
            "grades": {
                rung: _uncalibrated_grade(scores, rung, n_bins=n_bins, n_boot=n_boot, seed=seed)
                for rung in graded_rungs
            },
        }

    payload = R.calibrated_posterior(scores.logits, temperature=fit.temperature)
    posterior = np.asarray(payload[R.NAMED_POSTERIOR_KEY], dtype=np.float64)
    scaled_logits = np.asarray(scores.logits, dtype=np.float64) / float(fit.temperature)

    grades: dict[str, Any] = {}
    for rung in graded_rungs:
        idx = scores.index_of(rung)
        if idx.size == 0:
            raise ValueError(
                f"{scores.arm}: no rows on the {rung!r} rung — a grade computed on nothing "
                f"reads exactly like a passing one (census {scores.census()})"
            )
        y = [int(v) for v in np.asarray(scores.labels)[idx]]
        p = posterior[idx]
        z_raw = np.asarray(scores.logits, dtype=np.float64)[idx]
        z_scaled = scaled_logits[idx]
        blocks = [scores.blocks[i] for i in idx]

        # AUPRC on LOGITS (see the module docstring, point 2): rank-identical to the
        # calibrated ranking, and immune to the posterior saturation recorded below.
        auprc = M.average_precision(y, [float(v) for v in z_raw])
        auprc_scaled = M.average_precision(y, [float(v) for v in z_scaled])
        auprc_posterior = M.average_precision(y, [float(v) for v in p])
        # `z -> z/T` is strictly monotone in exact arithmetic but only NON-decreasing in
        # float64: two distinct logits can round onto one scaled value, and a new tie
        # changes average precision's tie handling in the last bits. Comparing the two AP
        # values alone would therefore let a rounding artifact flip a gate clause. State
        # the invariant directly instead — scaling created no ties — and record both the
        # tie count and the AP difference, so if it ever does fire it is diagnosable
        # rather than merely red.
        rank_ok, n_ties_created, auprc_scaling_abs_delta = ranking_preserved(
            z_raw, z_scaled, auprc, auprc_scaled
        )

        ece = M.binned_ece(y, [float(v) for v in p], n_bins, debias=True)
        ece_plugin = M.binned_ece(y, [float(v) for v in p], n_bins, debias=False)

        pairs = list(zip(y, [float(v) for v in p], strict=True))
        ece_ci = M.block_bootstrap_ci(
            _blockwise(pairs, blocks),
            lambda sample: M.binned_ece(
                [a for a, _ in sample], [b for _, b in sample], n_bins, debias=True
            ),
            n_boot=n_boot,
            seed=seed,
        )
        rank_pairs = list(zip(y, [float(v) for v in z_raw], strict=True))
        auprc_ci = M.block_bootstrap_ci(
            _blockwise(rank_pairs, blocks),
            lambda sample: M.average_precision([a for a, _ in sample], [b for _, b in sample]),
            n_boot=n_boot,
            seed=seed,
        )

        n_pos = int(sum(y))
        grades[rung] = {
            "n": int(idx.size),
            "n_positive": n_pos,
            "n_negative": int(idx.size) - n_pos,
            "prevalence": n_pos / int(idx.size),
            "ece": ece,
            "ece_plugin": ece_plugin,
            "ece_n_bins": int(n_bins),
            "ece_binning": "equal_mass",
            "ece_debiased": True,
            "ece_gate": ECE_GATE,
            "ece_gate_pass": bool(M.gate2_ece_pass(ece)),
            "ece_ci": ece_ci,
            "auprc": auprc,
            "auprc_scaled_logits": auprc_scaled,
            "auprc_on_posterior": auprc_posterior,
            "auprc_rank_invariant": rank_ok,
            "n_ties_created_by_scaling": n_ties_created,
            "auprc_scaling_abs_delta": auprc_scaling_abs_delta,
            "auprc_baseline_prevalence": n_pos / int(idx.size),
            "auprc_ci": auprc_ci,
            "posterior_saturation": _saturation(p),
            "graded_on": "named_posterior (temperature-scaled, PRE prior-shift)",
            "separation": separation_census(scores, rung),
            "block_census": {
                "n_blocks": len(set(blocks)),
                "n_rows": int(idx.size),
            },
            "reliability": M.reliability_bins(y, [float(v) for v in p], n_bins),
        }

    return {
        "arm": scores.arm,
        "calibration": {
            "fitted": True,
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
            "calib_separation": separation_census(scores, CALIB_RUNG),
        },
        "stack": {
            "stack_order": list(payload["stack_order"]),
            "stack_applied": list(payload["stack_applied"]),
            "gated_posterior_key": payload["gated_posterior_key"],
            "prior_shift_applied": bool(payload["prior_shift_applied"]),
            "named_posterior_exists": True,
        },
        "scored_census": scores.census(),
        "grades": grades,
    }


# --------------------------------------------------------------------------- #
# The ablation comparison (pure) — both readings of D16, neither resolved
# --------------------------------------------------------------------------- #
def compare_arms(
    with_aux: Mapping[str, Any],
    no_aux: Mapping[str, Any],
    *,
    grade_rung: str = GRADE_RUNG,
    with_aux_config: Mapping[str, Any] | None = None,
    no_aux_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The D16 check, computed under **both** readings of its one qualitative sentence.

    *Absolute reading* — "do not degrade the calibrated primary head's GATE-2 grade"
    means the with-aux arm must still **hold** that grade: ECE ≤ 0.05 (D11). Fully
    determined; no new number.

    *Delta reading* — ``imp.md``'s: with-aux must not be worse than no-aux "beyond a
    pre-registered tolerance" τ. τ **was never pre-registered anywhere in this repo**,
    so no verdict is emitted; instead the exact window on which the two readings
    disagree is reported. If the absolute reading passes and with-aux is ``d`` worse,
    every τ < d fails the delta reading while the absolute one passes — so ``d`` *is*
    the divergence, and it is the number a §7 sign-off needs in front of it
    ([[ambiguous-adr-prose-is-a-stop]]).
    """
    w = with_aux["grades"][grade_rung]
    n = no_aux["grades"][grade_rung]

    # The ECE half of the comparison exists only if BOTH arms have a named posterior.
    # When either fit refused there is no calibrated object to difference, and the
    # honest delta is `None` — not zero, which would read as "the arms agree".
    # The two readings have DIFFERENT evidence requirements, and conflating them was a
    # review finding worth the name: the absolute reading asks only whether the with-aux
    # arm still holds the D11 grade, so it is answerable whenever *that* arm has an ECE —
    # even when the control has none and the delta is undefined. Gating it on the
    # comparison's availability threw away a verdict this run actually has.
    with_aux_ece_available = w.get("ece") is not None
    ece_available = with_aux_ece_available and n.get("ece") is not None
    delta_ece = (w["ece"] - n["ece"]) if ece_available else None
    delta_auprc = w["auprc"] - n["auprc"]  # < 0 ⇒ with-aux RANKS worse

    absolute_passes = bool(w["ece_gate_pass"]) if with_aux_ece_available else None
    # A degradation is only a degradation if with-aux is the worse arm; a with-aux arm
    # that is better makes every non-negative τ pass, so the divergence window is empty.
    ece_degradation = max(0.0, delta_ece) if ece_available else None
    auprc_degradation = max(0.0, -delta_auprc)

    return {
        "with_aux_arm": with_aux["arm"],
        "no_aux_arm": no_aux["arm"],
        "graded_on_rung": grade_rung,
        "graded_object": "named_posterior (temperature-scaled, PRE prior-shift) — ADR-0005 D11",
        "matched_lr": (
            None
            if with_aux_config is None or no_aux_config is None
            else bool(_isclose(with_aux_config.get("lr"), no_aux_config.get("lr")))
        ),
        "ece_comparison_available": ece_available,
        "with_aux": {
            "ece": w.get("ece"),
            "auprc": w["auprc"],
            "ece_ci": w.get("ece_ci"),
            "auprc_ci": w.get("auprc_ci"),
        },
        "no_aux": {
            "ece": n.get("ece"),
            "auprc": n["auprc"],
            "ece_ci": n.get("ece_ci"),
            "auprc_ci": n.get("auprc_ci"),
        },
        "delta_ece": delta_ece,
        "delta_auprc": delta_auprc,
        "reading_absolute": {
            "rule": "with-aux still holds the D11 GATE-2 grade: in-distribution ECE <= gate",
            "source": "ADR-0005 D16 read against D11; needs no new number",
            "observed_ece": w.get("ece"),
            "gate": ECE_GATE,
            "passes": absolute_passes,
            "depends_only_on_the_with_aux_arm": True,
            "unavailable_reason": (
                None if with_aux_ece_available else w.get("ece_unavailable_reason")
            ),
        },
        "reading_delta": {
            "rule": "with-aux is not worse than no-aux by more than tolerance tau",
            "source": "imp.md P3-08 validation gate ('beyond a pre-registered tolerance')",
            "tolerance": None,
            "tolerance_status": "NEVER PRE-REGISTERED — not in PRD.md, any ADR, conf/ or src/",
            "observed_ece_degradation": ece_degradation,
            "ece_axis_decidable": ece_available,
            "observed_auprc_degradation": auprc_degradation,
            "verdict": "unpinned",
        },
        "divergence": {
            "readings_disagree_for_ece_tolerance_below": (
                ece_degradation if (ece_available and absolute_passes) else None
            ),
            "readings_disagree_for_auprc_tolerance_below": auprc_degradation,
            "note": _divergence_note(
                ece_available=ece_available,
                absolute_passes=absolute_passes,
                ece_degradation=ece_degradation,
            ),
        },
        "reference_margins_not_governing": {
            "source": D17_REFERENCE_SOURCE,
            **D17_REFERENCE_MARGINS,
            "would_pass_ece": (
                None if ece_degradation is None else ece_degradation <= D17_REFERENCE_MARGINS["ece"]
            ),
            "would_pass_auprc": auprc_degradation <= D17_REFERENCE_MARGINS["auprc"],
        },
        "governing_reading": GOVERNING_READING,
        "governing_reading_source": GOVERNING_READING_SOURCE,
        # Derived, never assigned. The verdict is a function of the governing reading's
        # own recorded evidence, so a report cannot carry a verdict its numbers do not
        # support — and `validate_report` re-derives it rather than trusting it.
        "verdict": verdict_from_absolute_reading(absolute_passes),
    }


def verdict_from_absolute_reading(absolute_passes: bool | None) -> str:
    """The D16 verdict, derived from the governing (absolute) reading and nothing else.

    A free function so :func:`validate_report` can re-derive it instead of believing the
    string in the file ([[gate-clauses-need-re-derivation]]). ``None`` means the with-aux
    arm has no ECE at all, which is genuinely undecidable — not a failure and not a pass.
    """
    if absolute_passes is None:
        return VERDICT_UNDECIDABLE
    return VERDICT_HOLDS if absolute_passes else VERDICT_DEGRADES


def _divergence_note(
    *, ece_available: bool, absolute_passes: bool | None, ece_degradation: float | None
) -> str:
    """Say, in one sentence, which of the four states the ECE axis is actually in.

    Kept as a function rather than a nested conditional because the *undecidable* case
    has to come first and be unmistakable: with no temperature there is no calibrated
    object, so neither reading of D16 has an answer on this axis — which is a different
    thing from the absolute reading failing, and must never be reported as one.
    """
    if not ece_available:
        return (
            "no ECE exists for at least one arm, so BOTH readings of D16 are unanswerable "
            "on the calibration axis; only the ranking axis is decidable"
        )
    if not absolute_passes:
        return "the absolute reading FAILS, so the delta reading is moot for ECE"
    if ece_degradation and ece_degradation > 0.0:
        return (
            "the absolute reading passes and with-aux is worse by this much, so every tau "
            "below it would fail the delta reading while the absolute one passes"
        )
    return "no tau >= 0 separates the two readings on ECE"


def _isclose(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Arm discovery + the production arm (derived, never hand-typed)
# --------------------------------------------------------------------------- #
def _yaml_scalar(path: str | Path, key: str) -> Any:
    """One top-level scalar out of a Hydra config, without importing Hydra.

    ``yaml.safe_load`` rather than a regex, because these files carry comment blocks
    and a ``defaults:`` list that a line-matcher would happily mis-read.
    """
    import yaml  # lazy: CLI path only, so the numpy grading tier stays CI-importable

    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    if not isinstance(doc, Mapping) or key not in doc:
        raise ValueError(f"{path}: no top-level {key!r}")
    return doc[key]


def production_arm_config(
    *, loss_conf: str | Path = LOSS_CONF, optim_conf: str | Path = OPTIM_CONF
) -> dict[str, float]:
    """The shipped ``(aux_weight, lr)`` — read out of ``conf/``, not written down here.

    The production arm is whichever P3-06 point was trained at the config the repo
    actually ships. Hard-coding ``"aux1.0_lr1e-4"`` would keep reading as correct
    forever after somebody edited ``conf/loss/stage2.yaml``
    ([[pinned-constant-that-nothing-reads]]); deriving it means such an edit makes
    :func:`select_arm_pair` fail to find a matching arm, loudly.

    ``float()`` rather than trusting the YAML type: ``1.0e-4`` resolves to a float
    under PyYAML but a bare ``1e-4`` would come back a *string*, and a silently
    string-typed lr would match no arm for a reason that looks nothing like the cause.
    """
    return {
        "aux_weight": float(_yaml_scalar(loss_conf, "aux_weight")),
        "lr": float(_yaml_scalar(optim_conf, "lr")),
    }


def discover_arms(
    checkpoint_root: str | Path = DEFAULT_CKPT_ROOT,
    *,
    sweep_dir: str | Path = DEFAULT_SWEEP_DIR,
) -> dict[str, dict[str, Any]]:
    """Every trained arm on disk, keyed by directory name, with its recorded config.

    The ``(aux_weight, lr)`` of an arm is read from the report **that arm's own run
    wrote**, never parsed back out of its directory name: the name is a shell string
    the sbatch composed (``KEY="aux${AUX_WEIGHT}_lr${LR}"``) and re-deriving a float
    from it would be a second, unpinned encoding of the same fact.
    """
    root = Path(checkpoint_root)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint root {root} does not exist")
    arms: dict[str, dict[str, Any]] = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        report_path = Path(sweep_dir) / f"{arm_dir.name}.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"arm {arm_dir.name} has a checkpoint but no run report at {report_path}; "
                "its aux_weight/lr are unknown and it cannot be placed in the ablation"
            )
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)
        config = report.get("config") or {}
        loss = config.get("loss") or {}
        legacy = report.get("legacy") or {}
        wrap = report.get("wrap") or {}
        arms[arm_dir.name] = {
            "arm": arm_dir.name,
            # Two fields, because they have two jobs. `checkpoint_path` is what gets
            # OPENED and must stay usable from wherever this runs (the DVC-materialised
            # checkpoints live in the main checkout, not in a linked worktree);
            # `checkpoint_dir` is what gets RECORDED into a committed, public artifact
            # and must not carry a developer's absolute path. Relativising the one the
            # loader opens is how the first version of this fix broke the run outright.
            "checkpoint_path": str(Path(arm_dir).resolve()),
            "checkpoint_dir": repo_relative(arm_dir),
            "run_report": repo_relative(report_path),
            "aux_weight": float(loss["aux_weight"]),
            "lr": float(config["lr"]),
            # The attention backend the arm was TRAINED under. Scoring reproduces it
            # rather than re-resolving: `select_attention_backend` answers "what is best
            # on this machine", which on a different card is a different backend and so
            # different numerics — and multimolecule's eager path is not merely slower,
            # it raises on a bf16 base with PEFT's float32-autocast adapters, a fault
            # the flash_attention_2 path the six arms trained under never reaches.
            "attn_implementation": wrap.get("attn_implementation"),
            # P3-06 saved final-epoch weights, not best-epoch: `best_val_total`
            # describes weights that were discarded. The comparable number is the
            # one that belongs to the checkpoint on disk.
            "saved_val_total": legacy.get("saved_val_total"),
            "saved_from_epoch": legacy.get("saved_from_epoch"),
        }
    if not arms:
        raise FileNotFoundError(f"no arm directories under {root}")
    return arms


def select_arm_pair(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    production: Mapping[str, float],
) -> tuple[str, str]:
    """``(with_aux_arm, no_aux_arm)`` — the production point and its matched-lr control.

    Matched on ``lr`` on purpose: the ablation asks what the *aux heads* do, so the
    only thing allowed to differ between the two arms is ``aux_weight``. An unmatched
    comparison would confound the aux effect with a learning-rate effect and the
    resulting delta would answer a question nobody asked.
    """
    with_aux = [
        name
        for name, cfg in arms.items()
        if _isclose(cfg["aux_weight"], production["aux_weight"])
        and _isclose(cfg["lr"], production["lr"])
    ]
    if len(with_aux) != 1:
        raise ValueError(
            f"the shipped config (aux_weight={production['aux_weight']}, lr={production['lr']}) "
            f"matches {len(with_aux)} trained arms {sorted(with_aux)} — expected exactly one; "
            "conf/ and the P3-06 sweep have diverged"
        )
    if _isclose(production["aux_weight"], 0.0):
        raise ValueError(
            "the shipped config is itself the no-aux arm (aux_weight=0), so there is no "
            "with/without-aux contrast to grade"
        )
    no_aux = [
        name
        for name, cfg in arms.items()
        if _isclose(cfg["aux_weight"], 0.0) and _isclose(cfg["lr"], production["lr"])
    ]
    if len(no_aux) != 1:
        raise ValueError(
            f"expected exactly one aux_weight=0 arm at lr={production['lr']}, found "
            f"{sorted(no_aux)}; the ablation needs a learning-rate-matched control"
        )
    return with_aux[0], no_aux[0]


# --------------------------------------------------------------------------- #
# Report: clauses / validator / builder
# --------------------------------------------------------------------------- #
def _pos_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """Re-derive every clause from the report's own recorded evidence.

    ``all(clauses)`` catches a clause flipped false but never one fabricated true, so
    nothing here reads back a *requested* setting ([[gate-clauses-need-re-derivation]]).
    Two of these are **completeness** clauses rather than correctness ones: every
    rule-shaped clause below survives a run truncated to a handful of rows, so
    "the fit used the whole calib carve" and "both arms were scored on every rung"
    have to be asserted separately ([[cost-knobs-can-certify]]).

    Note what is **not** a clause: the ablation verdict. Its threshold was never
    pre-registered, so a clause asserting it would be asserting a number this repo
    does not have.
    """
    arms = report.get("arms") or {}
    ablation = report.get("ablation") or {}
    dataset = report.get("dataset") or {}
    clauses: dict[str, bool] = {}

    arm_blocks = [a for a in arms.values() if isinstance(a, Mapping)]

    # -- the score producer actually produced scores from the trained weights ------- #
    loads = [(a.get("load") or {}) for a in arm_blocks]
    clauses["adapter_weights_verified_against_file"] = bool(loads) and all(
        _pos_int(ld.get("n_adapter_tensors_in_file"))
        and ld.get("n_adapter_tensors_matched") == ld.get("n_adapter_tensors_in_file")
        and ld.get("n_adapter_tensors_mismatched") == 0
        and ld.get("n_module_adapter_tensors_absent_from_file") == 0
        and ld.get("n_module_adapter_tensors") == ld.get("n_adapter_tensors_in_file")
        for ld in loads
    )
    # PEFT initialises lora_B to zero, so an adapter that silently failed to load is the
    # bare backbone. Counting non-zero B blocks measures the effect; reading a return
    # code would not ([[in-process-no-ops-look-like-compliance]]).
    clauses["adapter_is_live_not_identity"] = bool(loads) and all(
        _pos_int(ld.get("n_lora_b_nonzero"))
        and ld.get("n_lora_b_nonzero") == ld.get("n_lora_b_tensors")
        for ld in loads
    )
    clauses["head_weights_verified_against_file"] = bool(loads) and all(
        _pos_int(ld.get("n_head_tensors_in_file"))
        and ld.get("n_head_tensors_matched") == ld.get("n_head_tensors_in_file")
        for ld in loads
    )
    # Same weights, different attention kernel, different last bits — and on this
    # backbone the eager path does not merely perturb the numerics, it raises. Scoring
    # under a backend the arm never trained under is a silent numerics change.
    clauses["scored_under_the_training_attention_backend"] = bool(loads) and all(
        ld.get("attn_implementation_matches_training") is True for ld in loads
    )

    # -- the calibration stack ------------------------------------------------------ #
    calibs = [(a.get("calibration") or {}) for a in arm_blocks]
    stacks = [(a.get("stack") or {}) for a in arm_blocks]
    graded_for_ece = [((a.get("grades") or {}).get(GRADE_RUNG) or {}) for a in arm_blocks]
    clauses["temperature_fitted_on_calib_only"] = bool(calibs) and all(
        c.get("fitted_on") == CALIB_RUNG
        and _pos_int(c.get("n_fitted"))
        and c.get("n_fitted") == (c.get("n_by_rung") or {}).get(CALIB_RUNG)
        for c in calibs
    )
    clauses["temperature_positive_and_converged"] = bool(calibs) and all(
        c.get("fitted") is True
        and _finite(c.get("temperature"))
        and float(c.get("temperature", 0.0)) > 0.0
        and bool(c.get("converged"))
        for c in calibs
    )
    # The gated object itself. Split out from the fit clause because "no T could be
    # fitted" and "T was fitted but on the wrong rows" are different findings that
    # want different fixes, and folding them together would report either as the other.
    clauses["in_distribution_ece_is_computable"] = bool(graded_for_ece) and all(
        g.get("ece") is not None for g in graded_for_ece
    )
    clauses["graded_object_is_pre_prior_shift"] = bool(stacks) and all(
        s.get("gated_posterior_key") == R.NAMED_POSTERIOR_KEY
        and s.get("prior_shift_applied") is False
        and s.get("named_posterior_exists") is True
        and list(s.get("stack_applied") or []) == ["train", "temperature_scale"]
        for s in stacks
    )

    # -- the grades ----------------------------------------------------------------- #
    graded = [((a.get("grades") or {}).get(GRADE_RUNG) or {}) for a in arm_blocks]
    clauses["ece_estimator_matches_adr_d11"] = bool(graded) and all(
        g.get("ece_n_bins") == ECE_N_BINS
        and g.get("ece_binning") == "equal_mass"
        and g.get("ece_debiased") is True
        for g in graded
    )
    clauses["auprc_is_rank_invariant_under_scaling"] = bool(graded) and all(
        g.get("auprc_rank_invariant") is True for g in graded
    )
    clauses["graded_rung_is_the_gate2_split"] = (
        (ablation.get("graded_on_rung") == GRADE_RUNG)
        and bool(graded)
        and all(_pos_int(g.get("n")) for g in graded)
    )

    # -- completeness (the clauses a truncated run would otherwise satisfy) --------- #
    expected = dataset.get("rung_census") or {}
    clauses["scored_every_row_of_every_scored_rung"] = (
        bool(arm_blocks)
        and bool(expected)
        and all(
            all(
                (a.get("scored_census") or {}).get(rung) == expected.get(rung)
                for rung in SCORED_RUNGS
            )
            for a in arm_blocks
        )
    )
    clauses["both_ablation_arms_present"] = (
        ablation.get("with_aux_arm") in arms
        and ablation.get("no_aux_arm") in arms
        and ablation.get("with_aux_arm") != ablation.get("no_aux_arm")
    )
    clauses["ablation_arms_are_lr_matched"] = ablation.get("matched_lr") is True
    with_aux_block = arms.get(str(ablation.get("with_aux_arm")), {}) or {}
    no_aux_block = arms.get(str(ablation.get("no_aux_arm")), {}) or {}
    clauses["ablation_contrast_is_aux_weight"] = bool(
        _finite(with_aux_block.get("aux_weight"))
        and _finite(no_aux_block.get("aux_weight"))
        and float(with_aux_block["aux_weight"]) > 0.0
        and float(no_aux_block["aux_weight"]) == 0.0
    )

    return clauses


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Structural problems with ``report``; empty means well-formed and **self-consistent**.

    Deliberately *not* "and passing" (review r2): this checks that the recorded clauses
    match the ones re-derived from the report's own evidence and that ``overall_pass``
    agrees with them. A report whose gate is honestly ``false`` — as the shipped one is
    — is fully valid. Conflating the two would make a truthful failing report look
    malformed and invite someone to "fix" it.
    """
    problems: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {report.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    if report.get("step") != STEP:
        problems.append(f"step {report.get('step')!r} != {STEP!r}")
    for block in ("dataset", "arms", "ablation", "provenance", "gate"):
        if not isinstance(report.get(block), Mapping):
            problems.append(f"missing or non-mapping block {block!r}")
    ablation = report.get("ablation")
    if isinstance(ablation, Mapping):
        if (ablation.get("reading_delta") or {}).get("tolerance") is not None:
            problems.append(
                "ablation.reading_delta.tolerance is set, but no tolerance is pre-registered "
                "in PRD.md, any ADR, conf/ or src/ — pinning one needs CLAUDE.md §7 sign-off"
            )
        expected = verdict_from_absolute_reading(
            (ablation.get("reading_absolute") or {}).get("passes")
        )
        if ablation.get("verdict") != expected:
            problems.append(
                f"ablation.verdict {ablation.get('verdict')!r} but the governing "
                f"(absolute) reading's own evidence gives {expected!r}; the verdict is "
                "derived, not assignable"
            )
        if ablation.get("governing_reading") != GOVERNING_READING:
            problems.append(
                f"ablation.governing_reading {ablation.get('governing_reading')!r} != "
                f"{GOVERNING_READING!r} (ADR-0005 D16, user sign-off 2026-08-03)"
            )
    gate = report.get("gate")
    if not isinstance(gate, Mapping):
        return problems
    recomputed = derive_clauses(report)
    recorded = gate.get("clauses")
    if not isinstance(recorded, Mapping):
        problems.append("gate.clauses is missing")
        return problems
    if set(recorded) != set(recomputed):
        problems.append(f"gate.clauses keys {sorted(recorded)} != re-derived {sorted(recomputed)}")
    for name, value in recomputed.items():
        if bool(recorded.get(name)) != value:
            problems.append(
                f"gate.clauses[{name!r}]={recorded.get(name)!r} but re-derivation says {value}"
            )
    if bool(gate.get("overall_pass")) != all(recomputed.values()):
        problems.append(
            f"gate.overall_pass={gate.get('overall_pass')!r} but the re-derived clauses give "
            f"{all(recomputed.values())}"
        )
    return problems


def build_report(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    ablation: Mapping[str, Any],
    dataset: Mapping[str, Any],
    device: Mapping[str, Any] | None = None,
    scoring: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    written_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the report, then let :func:`derive_clauses` grade it from its own numbers."""
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "entrypoint": ENTRYPOINT,
        "generated_by": RULE,
        "written_at": written_at,
        "prd": [
            "§11 with/without-aux check",
            "§12 calibration (named posterior, pre-prior-shift)",
            "§2.3 GATE-2",
        ],
        "adr": [
            "ADR-0005 D11 (named posterior; 15 equal-mass debiased bins; ECE <= 0.05)",
            "ADR-0005 D16 (aux weighting + the with/without-aux check)",
            "ADR-0004 A7 (test is GATE-2's split; calib is where T is fitted)",
        ],
        "dataset": dict(dataset),
        "device": dict(device or {}),
        "scoring": dict(scoring or {}),
        "arms": {name: dict(block) for name, block in arms.items()},
        "ablation": dict(ablation),
        "provenance": dict(provenance or {}),
    }
    clauses = derive_clauses(report)
    report["gate"] = {
        "clauses": clauses,
        "overall_pass": all(clauses.values()),
        "note": (
            "these clauses grade the MACHINERY — that the trained weights really loaded, "
            "that T was fitted on calib alone, that the graded object is the pre-prior-shift "
            "named posterior, and that nothing was truncated. The ablation VERDICT is "
            "separate and lives in `ablation.verdict`: it is derived from the governing "
            "(absolute) reading of ADR-0005 D16, which asks only whether the with-aux arm "
            "still holds the D11 grade. A false overall_pass here therefore does NOT mean "
            "the aux heads degraded anything — on this run it means the no-aux CONTROL is "
            "uncalibratable (its calib carve is perfectly separated), an accepted and "
            "recorded outcome (user sign-off, 2026-08-03), with the degenerate-limit rule "
            "to be pinned in ADR-0005 D11 at the P3-exit gate."
        ),
    }
    return report


def aux_ablation_check(
    arm_scores: Mapping[str, ArmScores],
    *,
    arm_configs: Mapping[str, Mapping[str, Any]],
    with_aux_arm: str,
    no_aux_arm: str,
    dataset: Mapping[str, Any],
    load_records: Mapping[str, Mapping[str, Any]] | None = None,
    device: Mapping[str, Any] | None = None,
    scoring: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    written_at: str | None = None,
    grade_rung: str = GRADE_RUNG,
    n_bins: int = ECE_N_BINS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """**The imp.md P3-08 entry.** Grade every arm, compare the ablation pair, report.

    Pure: it takes already-produced per-row logits, so the whole gate — calibration,
    ECE, AUPRC, the clause set, the validator — runs in bare CI with no torch, no GPU
    and no 2.5 GB backbone. :func:`score_rows` is what turns checkpoints into the
    :class:`ArmScores` this eats.
    """
    for name in (with_aux_arm, no_aux_arm):
        if name not in arm_scores:
            raise KeyError(f"ablation arm {name!r} was not scored (have {sorted(arm_scores)})")

    graded: dict[str, dict[str, Any]] = {}
    for name, scores in arm_scores.items():
        block = grade_arm(
            scores,
            graded_rungs=(SELECT_RUNG, grade_rung) if grade_rung != SELECT_RUNG else (grade_rung,),
            n_bins=n_bins,
            n_boot=n_boot,
            seed=seed,
        )
        cfg = dict(arm_configs.get(name) or {})
        block["aux_weight"] = cfg.get("aux_weight")
        block["lr"] = cfg.get("lr")
        block["checkpoint_dir"] = cfg.get("checkpoint_dir")
        block["saved_val_total"] = cfg.get("saved_val_total")
        block["saved_from_epoch"] = cfg.get("saved_from_epoch")
        block["load"] = dict((load_records or {}).get(name) or {})
        graded[name] = block

    ablation = compare_arms(
        graded[with_aux_arm],
        graded[no_aux_arm],
        grade_rung=grade_rung,
        with_aux_config=arm_configs.get(with_aux_arm),
        no_aux_config=arm_configs.get(no_aux_arm),
    )

    return build_report(
        arms=graded,
        ablation=ablation,
        dataset=dataset,
        device=device,
        scoring=scoring,
        provenance=provenance,
        written_at=written_at,
    )


# =========================================================================== #
# TORCH TIER — the Stage-2 score producer (lazy imports; no GPU needed above)
# =========================================================================== #
_ADAPTER_NAME_RE = re.compile(r"\.default\.")


def _normalise_adapter_key(key: str) -> str:
    """Module parameter name → the key spelling used inside ``adapter_model.safetensors``.

    PEFT stores adapters on disk without the adapter name and inserts ``default`` into
    the key when it loads them, so the two spellings have to be brought together before
    they can be compared. Removing the segment is the cheap direction; the caller
    asserts the mapping stayed injective so a module genuinely named ``default``
    cannot collapse two parameters into one and hide a mismatch.
    """
    return _ADAPTER_NAME_RE.sub(".", key)


def load_stage2_checkpoint(
    checkpoint_dir: str | Path,
    *,
    spec: Any = None,
    revision: str | None = None,
    dtype: str | None = None,
    attn_implementation: str | None = None,
    device: str | None = None,
    structure_head: bool = False,
    pairing_proj_dim: int | None = None,
    base_model: Any = None,
    backbone: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """A P3-06 checkpoint → a ready-to-score :class:`Stage2Model`, weights **verified**.

    Returns ``(model, record)``. ``record`` is the evidence
    :func:`derive_clauses` grades: how many adapter and head tensors the files hold,
    how many of them the assembled module reproduces **bit-exactly**, and how many
    ``lora_B`` blocks are non-zero.

    That last count is the one that matters. ``PeftModel.from_pretrained`` only
    *warns* about missing adapter keys, and PEFT initialises ``lora_B`` to zero — so
    a failed adapter load leaves a model that is mathematically the untuned
    backbone, runs perfectly, and produces a full report of meaningless numbers. A
    return code cannot tell those apart; the weights can
    ([[in-process-no-ops-look-like-compliance]]).

    ``base_model`` lets a test drive the whole path through a tiny same-architecture
    RiNALMo instead of the 2.5 GB checkpoint, exactly as ``train.build_model`` does.
    """
    import torch  # lazy
    from peft import PeftModel  # lazy
    from safetensors.torch import load_file  # lazy

    from tbox_finder.models import rna_backbone_registry as BR
    from tbox_finder.stage2 import heads as H
    from tbox_finder.stage2.model import DEFAULT_PAIRING_PROJ_DIM, Stage2Model
    from tbox_finder.train import lora_harness as LH

    ckpt = Path(checkpoint_dir)
    adapter_dir = ckpt / ADAPTER_SUBDIR
    heads_path = ckpt / HEADS_STATE_NAME
    for path in (adapter_dir, heads_path):
        if not path.exists():
            raise FileNotFoundError(f"{ckpt} is not a Stage-2 checkpoint: {path} is missing")
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"{adapter_dir} holds no adapter_model.safetensors")

    with open(adapter_dir / "adapter_config.json", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    # A LoRA checkpoint stores adapters and heads and **not the backbone**, so the
    # weights it actually scores with are half read off disk and half whatever base
    # this process loaded. Pointing it at a different backbone is not an error anyone
    # would notice at runtime — the shapes match, the adapter applies cleanly, and the
    # logits are simply somebody else's.
    # ⚠ Since ADR-0002 A15 there are TWO pinned Stage-2 backbones, so this cannot compare
    # against a single constant any more. The backbone is **re-derived from the checkpoint's
    # own evidence** — the base model its adapter_config records — against the closed
    # allow-list, rather than defaulting a missing axis to "production" on faith
    # ([[gate-must-bind-to-upstream-evidence]]). A recorded base outside the allow-list still
    # raises, and so does a `backbone=` argument that contradicts what the checkpoint says:
    # being *asked* for the wrong base is the same defect as inheriting it.
    recorded_base = adapter_config.get("base_model_name_or_path")
    derived = BR.backbone_for_repo_id(recorded_base) if recorded_base else None
    if recorded_base and derived is None:
        raise ValueError(
            f"{adapter_dir} was trained against base model {recorded_base!r}, which is not in "
            f"the ADR-0002 A15 allow-list {BR.BACKBONE_KEYS}; the adapter would be applied to "
            "a backbone this repo does not pin"
        )
    if backbone is not None and derived is not None and backbone != derived.key:
        raise ValueError(
            f"{adapter_dir} records base model {recorded_base!r} (backbone {derived.key!r}) "
            f"but backbone={backbone!r} was requested; the adapter would be applied to a "
            "backbone it never saw"
        )
    resolved_backbone = derived.key if derived is not None else (backbone or BR.PRODUCTION_BACKBONE)

    resolved_dtype = dtype or LH.TRAIN_DTYPE
    # The revision that will actually be loaded, resolved from the SAME spec the loader uses
    # rather than from a module constant — otherwise a comparator checkpoint's record would
    # report the production checkpoint's revision.
    resolved_revision = revision or BR.resolve_backbone(resolved_backbone).revision
    if base_model is None:
        base_model = LH.load_rna_backbone(
            backbone=resolved_backbone,
            revision=revision,
            dtype=resolved_dtype,
            attn_implementation=attn_implementation,
            device=None,
        )

    # PeftModel.from_pretrained reads the SAVED adapter_config.json, so the LoRA rank,
    # alpha and target-module set come from the run that trained these weights rather
    # than from today's lora_config_kwargs(). A fresh wrap would silently leave any
    # site the two configs disagree about at its zero initialisation.
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)

    file_adapter = load_file(str(adapter_file))
    module_adapter_raw = {
        name: param for name, param in peft_model.named_parameters() if "lora_" in name
    }
    module_adapter = {_normalise_adapter_key(k): v for k, v in module_adapter_raw.items()}
    if len(module_adapter) != len(module_adapter_raw):
        raise RuntimeError(
            "normalising the PEFT adapter-name segment collapsed two distinct parameters; "
            "a module is literally named 'default' and the comparison would be blind"
        )

    matched = 0
    mismatched: list[str] = []
    missing: list[str] = []
    for key, saved in file_adapter.items():
        live = module_adapter.get(key)
        if live is None:
            missing.append(key)
            continue
        if live.shape != saved.shape or not torch.equal(
            live.detach().to("cpu", dtype=saved.dtype), saved
        ):
            mismatched.append(key)
        else:
            matched += 1

    # The reverse direction, and it is not symmetric bookkeeping: the loop above walks
    # the FILE, so a parameter the file simply does not carry is never visited — it
    # stays at PEFT's initialisation, and for lora_A that initialisation is *random*,
    # not zero. Walking the module too is what makes "these are the trained weights" a
    # statement about the whole adapter rather than about the subset that happened to
    # be written.
    absent_from_file = sorted(set(module_adapter) - set(file_adapter))

    lora_b_keys = [k for k in file_adapter if ".lora_B." in k]
    n_b_nonzero = sum(
        1
        for k in lora_b_keys
        if (live := module_adapter.get(k)) is not None and bool(torch.any(live.detach() != 0))
    )

    resolved_pairing = (
        DEFAULT_PAIRING_PROJ_DIM if pairing_proj_dim is None else int(pairing_proj_dim)
    )
    model = Stage2Model(
        spec if spec is not None else H.load_head_spec(),
        backbone=peft_model,
        # Scoring runs under eval(), where dropout is identity anyway; 0.0 makes that
        # explicit rather than carrying the training probability into an inference path.
        dropout=0.0,
        boundary_use_crf=False,
        structure_head=structure_head,
        pairing_proj_dim=resolved_pairing,
    )

    head_state = torch.load(heads_path, map_location="cpu", weights_only=True)
    by_attr: dict[str, dict[str, Any]] = {}
    for key, tensor in head_state.items():
        attr, _, rest = key.partition(".")
        if not rest:
            raise ValueError(f"{heads_path}: head key {key!r} has no module-local suffix")
        by_attr.setdefault(attr, {})[rest] = tensor
    live_heads = model.head_modules
    if set(by_attr) != set(live_heads):
        raise ValueError(
            f"{heads_path} carries heads {sorted(by_attr)} but this model has "
            f"{sorted(live_heads)} — the checkpoint and the head spec disagree"
        )
    for attr, sub_state in by_attr.items():
        # strict=True: a missing or unexpected head key is a refusal, not a warning.
        live_heads[attr].load_state_dict(sub_state, strict=True)

    head_matched = 0
    head_mismatched: list[str] = []
    live_head_params = dict(model.named_parameters())
    live_head_buffers = dict(model.named_buffers())
    for key, saved in head_state.items():
        live = live_head_params.get(key, live_head_buffers.get(key))
        if (
            live is None
            or live.shape != saved.shape
            or not torch.equal(live.detach().to("cpu", dtype=saved.dtype), saved)
        ):
            head_mismatched.append(key)
        else:
            head_matched += 1

    model.eval()
    if device is not None:
        model = model.to(device)

    record = {
        "checkpoint_dir": repo_relative(ckpt),
        "adapter_dir": repo_relative(adapter_dir),
        "heads_path": repo_relative(heads_path),
        "adapter_sha256": PROV.sha256_file(adapter_file),
        "heads_sha256": PROV.sha256_file(heads_path),
        "adapter_config": {
            key: adapter_config.get(key)
            for key in ("peft_type", "r", "lora_alpha", "lora_dropout", "target_modules", "bias")
        },
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        # Which allow-list entry the base was re-derived to be (ADR-0002 A15). Recorded
        # because `base_model_name_or_path` alone is a repo id, and every consumer that wants
        # to know "was this the shipped arm or the comparator" would otherwise have to
        # re-derive it independently and could disagree.
        "backbone": resolved_backbone,
        "revision": resolved_revision,
        "dtype": resolved_dtype,
        "attn_implementation": attn_implementation,
        "n_adapter_tensors_in_file": len(file_adapter),
        "n_adapter_tensors_matched": matched,
        "n_adapter_tensors_mismatched": len(mismatched),
        "n_adapter_tensors_missing_from_module": len(missing),
        "n_module_adapter_tensors": len(module_adapter),
        "n_module_adapter_tensors_absent_from_file": len(absent_from_file),
        "adapter_mismatched_keys": sorted(mismatched)[:10],
        "adapter_missing_keys": sorted(missing)[:10],
        "adapter_keys_absent_from_file": absent_from_file[:10],
        "n_lora_b_tensors": len(lora_b_keys),
        "n_lora_b_nonzero": n_b_nonzero,
        "n_head_tensors_in_file": len(head_state),
        "n_head_tensors_matched": head_matched,
        "head_mismatched_keys": sorted(head_mismatched)[:10],
        "structure_head": bool(structure_head),
        "pairing_proj_dim": resolved_pairing,
    }
    if mismatched or missing or absent_from_file:
        raise RuntimeError(
            f"{adapter_dir}: {len(mismatched)} adapter tensor(s) differ from the file, "
            f"{len(missing)} are absent from the assembled module and "
            f"{len(absent_from_file)} live adapter tensor(s) have no entry in the file "
            f"(so they still hold PEFT's initialisation) — refusing to score with weights "
            f"that are not the ones that were trained. Record: {record}"
        )
    if head_mismatched:
        raise RuntimeError(
            f"{heads_path}: {len(head_mismatched)} head tensor(s) differ from the file — "
            f"refusing to score. Record: {record}"
        )
    if lora_b_keys and n_b_nonzero == 0:
        raise RuntimeError(
            f"{adapter_dir}: every lora_B block is zero, so the adapter is the identity and "
            "this model is the untuned backbone wearing a checkpoint's name"
        )
    return model, record


def score_rows(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 4,
    device: Any = None,
    sequence_field: str = _SEQUENCE,
) -> list[dict[str, Any]]:
    """Per-row raw binary-head logits — the artifact the calibration stack eats.

    Rows are batched in ``(n_tokens, row_id)`` order rather than input order. That is
    a determinism decision as much as a speed one: padding a 71-token row up to a
    551-token neighbour wastes most of the batch, and *which* rows share a batch
    perturbs bf16 reductions at the last bits. Sorting on a key that depends only on
    the row set makes the batch composition — and so the logits — reproducible for a
    given set of rows and batch size.
    """
    import torch  # lazy

    from tbox_finder.stage2 import tokenizer as TOK

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    encoded: list[tuple[int, str, list[int]]] = []
    for row in rows:
        row_id = row.get(_ROW_ID)
        if row_id is None:
            raise ValueError("a row carries no row_id; its logit could not be joined back")
        sequence = row.get(sequence_field)
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"row {row_id}: {sequence_field!r} is missing or empty")
        ids = TOK.encode(sequence)
        encoded.append((len(ids), str(row_id), ids))
    encoded.sort(key=lambda item: (item[0], item[1]))

    out: dict[str, dict[str, Any]] = {}
    target = device if device is not None else next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size]
            width = max(n for n, _, _ in chunk)
            input_ids = torch.full((len(chunk), width), TOK.PAD_ID, dtype=torch.long, device=target)
            attention_mask = torch.zeros((len(chunk), width), dtype=torch.long, device=target)
            for i, (n, _, ids) in enumerate(chunk):
                input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=target)
                attention_mask[i, :n] = 1
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["tbox_logit"]
            logits = logits.detach().float().reshape(-1).cpu()
            for i, (n, row_id, _) in enumerate(chunk):
                out[row_id] = {
                    "row_id": row_id,
                    "tbox_logit": float(logits[i]),
                    "n_tokens": int(n),
                }
    # Restore the caller's row order: a producer that silently returns rows in its own
    # internal order invites a positional join against the labels.
    return [out[str(row[_ROW_ID])] for row in rows]


def reconcile_cached_scores(
    cached: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    dataset_meta: Mapping[str, Any],
    wanted: Sequence[str],
) -> tuple[dict[str, ArmScores], dict[str, Any], dict[str, Any]]:
    """Previously-written per-row logits -> :class:`ArmScores`, or a refusal.

    Extracted from ``main`` so its refusals are reachable without a GPU. Re-grading a
    stale cache is the one failure this shortcut can introduce and the one that would
    look most like success: seconds instead of minutes, a complete report, and numbers
    belonging to a different row set. So the cache must agree with this run on the
    dataset digest *and* on the exact row-id sequence, and must hold every arm asked
    for; each disagreement is refused by name rather than reconciled.
    """
    if cached.get("dataset_sha256") != dataset_meta["sha256"]:
        raise ValueError(
            f"the cached scores were produced from dataset sha256 "
            f"{cached.get('dataset_sha256')!r}, but this run reads "
            f"{dataset_meta['sha256']!r}"
        )
    cached_ids = [str(v) for v in cached.get("row_ids") or []]
    current_ids = [str(row[_ROW_ID]) for row in rows]
    if cached_ids != current_ids:
        raise ValueError(
            f"the cached scores cover {len(cached_ids)} rows and this run covers "
            f"{len(current_ids)}, or they are not the same rows in the same order"
        )
    arm_scores: dict[str, ArmScores] = {}
    load_records: dict[str, Any] = {}
    labels = np.asarray([int(bool(row[_LABEL])) for row in rows], dtype=np.int64)
    rungs = tuple(row["_rung"] for row in rows)
    blocks = tuple(row["_block"] for row in rows)
    for name in wanted:
        arm = (cached.get("arms") or {}).get(name)
        if arm is None:
            raise KeyError(f"the cached scores hold nothing for arm {name!r}")
        arm_scores[name] = ArmScores(
            arm=name,
            row_ids=tuple(cached_ids),
            logits=np.asarray(arm["logits"], dtype=np.float64),
            labels=labels,
            rungs=rungs,
            blocks=blocks,
        )
        load_records[name] = arm.get("load") or {}
    scoring = {
        "device": cached.get("device"),
        "batch_size": cached.get("batch_size"),
        "regraded_from_cached_scores": True,
    }
    return arm_scores, load_records, scoring


def scored_rows(
    dataset: str | Path = DEFAULT_DATASET,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows this step reads, plus the census the completeness clause is graded on.

    ``train`` is dropped on purpose: the arms were fitted on it, so every number it
    could produce is in-sample. What is left is ``calib`` (where ``T`` is fitted),
    ``val`` (the PRD §11 ladder) and ``test`` (the ADR-0004 A7 GATE-2 split).
    """
    all_rows = T.load_rows(dataset)
    all_rungs = rungs_for_rows(all_rows)
    rows = [row for row, rung in zip(all_rows, all_rungs, strict=True) if rung in SCORED_RUNGS]
    rungs = [rung for rung in all_rungs if rung in SCORED_RUNGS]
    census = {rung: sum(1 for r in all_rungs if r == rung) for rung in R.RUNG_VOCABULARY}
    blocks, block_census = block_keys(rows)
    meta = {
        "path": repo_relative(dataset),
        "sha256": PROV.sha256_file(dataset),
        "n_rows_total": len(all_rows),
        "n_rows_scored": len(rows),
        "rung_census": {rung: census[rung] for rung in SCORED_RUNGS},
        "full_rung_census": census,
        "scored_rungs": list(SCORED_RUNGS),
        "block_census": block_census,
    }
    for row, rung, block in zip(rows, rungs, blocks, strict=True):
        row["_rung"] = rung
        row["_block"] = block
    return rows, meta


def main(argv: Sequence[str] | None = None) -> int:
    """Score the ablation arms and write ``reports/stage2_aux_ablation.json``."""
    import argparse
    import datetime as _dt

    parser = argparse.ArgumentParser(description="P3-08 with/without-aux ablation")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--sweep-dir", default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument(
        "--scores-from",
        default=None,
        help="re-grade previously written per-row logits instead of re-scoring on the GPU",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument(
        "--all-arms",
        action="store_true",
        help="score every trained arm, not just the ablation pair (reported, non-gated)",
    )
    parser.add_argument(
        "--max-rows-per-rung",
        type=int,
        default=None,
        help="SMOKE ONLY — truncates each rung; the completeness clause then fails by design",
    )
    args = parser.parse_args(argv)

    import torch  # lazy

    rows, dataset_meta = scored_rows(args.dataset)
    if args.max_rows_per_rung is not None:
        kept: dict[str, int] = {}
        truncated = []
        for row in rows:
            rung = row["_rung"]
            if kept.get(rung, 0) < args.max_rows_per_rung:
                kept[rung] = kept.get(rung, 0) + 1
                truncated.append(row)
        rows = truncated
        dataset_meta["smoke_truncated_to_per_rung"] = args.max_rows_per_rung

    arms = discover_arms(args.checkpoint_root, sweep_dir=args.sweep_dir)
    production = production_arm_config()
    with_aux_arm, no_aux_arm = select_arm_pair(arms, production=production)
    wanted = sorted(arms) if args.all_arms else [with_aux_arm, no_aux_arm]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    arm_scores: dict[str, ArmScores] = {}
    load_records: dict[str, dict[str, Any]] = {}

    if args.scores_from:
        cached = json.loads(Path(args.scores_from).read_text(encoding="utf-8"))
        arm_scores, load_records, cached_scoring = reconcile_cached_scores(
            cached, rows=rows, dataset_meta=dataset_meta, wanted=wanted
        )
        device = cached_scoring.get("device") or device
        # The recorded batch size must describe the run that PRODUCED these logits, not
        # the flag this process happened to be invoked with. Batch composition perturbs
        # the bf16 reductions (see `score_rows`), so a mismatched number would document
        # a batching that never touched them.
        if cached_scoring.get("batch_size") is not None:
            args.batch_size = int(cached_scoring["batch_size"])
    for name in [] if args.scores_from else wanted:
        trained_under = arms[name].get("attn_implementation")
        backend = args.attn_implementation or trained_under
        if backend is None:
            raise RuntimeError(
                f"arm {name}'s run report records no attention backend, so scoring cannot "
                "reproduce the numerics it trained under; pass --attn-implementation "
                "explicitly and say so in the dev-log"
            )
        model, record = load_stage2_checkpoint(
            arms[name]["checkpoint_path"],
            attn_implementation=backend,
            device=device,
        )
        record["attn_implementation_trained_under"] = trained_under
        record["attn_implementation_matches_training"] = bool(backend == trained_under)
        scored = score_rows(model, rows, batch_size=args.batch_size, device=device)
        arm_scores[name] = ArmScores(
            arm=name,
            row_ids=tuple(str(row[_ROW_ID]) for row in rows),
            logits=np.asarray([s["tbox_logit"] for s in scored], dtype=np.float64),
            labels=np.asarray([int(bool(row[_LABEL])) for row in rows], dtype=np.int64),
            rungs=tuple(row["_rung"] for row in rows),
            blocks=tuple(row["_block"] for row in rows),
        )
        load_records[name] = record
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    if not args.scores_from:
        scores_out = Path(args.scores)
        scores_out.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "step": STEP,
                    "generated_by": ENTRYPOINT,
                    "dataset": dataset_meta["path"],
                    "dataset_sha256": dataset_meta["sha256"],
                    "device": device,
                    "batch_size": args.batch_size,
                    "row_ids": [str(row[_ROW_ID]) for row in rows],
                    "rungs": [row["_rung"] for row in rows],
                    "labels": [int(bool(row[_LABEL])) for row in rows],
                    "arms": {
                        name: {
                            "logits": [float(v) for v in arm_scores[name].logits],
                            "load": load_records[name],
                        }
                        for name in arm_scores
                    },
                },
                handle,
                default=str,
            )
            handle.write("\n")
        print(f"wrote {scores_out}")

    report = aux_ablation_check(
        arm_scores,
        arm_configs=arms,
        with_aux_arm=with_aux_arm,
        no_aux_arm=no_aux_arm,
        dataset=dataset_meta,
        load_records=load_records,
        device=T.device_record(),
        scoring={
            "batch_size": args.batch_size,
            "regraded_from_cached_scores": bool(args.scores_from),
            "scores_sidecar": repo_relative(args.scores_from or args.scores),
            "device": device,
            "inference_mode": True,
            "n_boot": args.n_boot,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "arms_scored": wanted,
        },
        provenance={
            "git_sha": PROV.git_sha(),
            "env_lock": ENV_LOCK,
            "env_lock_hash": PROV.env_lock_hash(ENV_LOCK),
            "entrypoint": ENTRYPOINT,
        },
        written_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        n_boot=args.n_boot,
    )

    problems = validate_report(report)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    print(f"wrote {out}")
    ab = report["ablation"]

    def _fmt(value: Any) -> str:
        return "UNAVAILABLE" if value is None else f"{value:+.6f}"

    for label, key in (("with-aux", "with_aux"), ("no-aux  ", "no_aux")):
        print(
            f"  {label} {ab[key + '_arm']}: ECE {_fmt(ab[key]['ece'])}  "
            f"AUPRC {ab[key]['auprc']:.6f}"
        )
    print(f"  delta ECE {_fmt(ab['delta_ece'])}   delta AUPRC {_fmt(ab['delta_auprc'])}")
    if not ab["ece_comparison_available"]:
        print("  ** the ECE axis is UNDECIDABLE: no temperature could be fitted **")
    print(f"  gate.overall_pass = {report['gate']['overall_pass']}")
    if problems:
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
