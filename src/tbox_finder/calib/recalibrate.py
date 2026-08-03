"""The P3-07 recalibration stack — temperature-scale on the disjoint ``calib`` split, then
the Saerens/Elkan prior-shift (PRD §11/§12; ADR-0005 D11).

D11 pins the **order** — ``train -> temperature_scale -> prior_shift`` — and, more
consequentially, pins *which posterior is graded*: GATE-2's in-distribution ECE ≤ 0.05 is
measured on the **named posterior**, the temperature-scaled posterior **before** the
deployment prior-shift, at the in-distribution split's own prevalence. The prior-shifted
posterior is miscalibrated at benchmark prevalence *by construction*, so it is reported
non-gated. This module therefore never returns "the" posterior: :func:`calibrated_posterior`
returns **both**, under keys that say which is which, plus ``gated_posterior_key`` naming the
one GATE-2 grades. A caller cannot pick the wrong object by accident, and a report built from
this dict carries the distinction into the artifact.

**The temperature fit is not reimplemented here.** ``calib.temperature.fit_temperature``
(P2-13, ADR-0005 A11 Pin 2/3) already solves the exact convex 1-D problem by bisection on
``β = 1/T``, is seedless and optimiser-free, and refuses the two degenerate limits by name
instead of returning a bracket endpoint. Stage 2's binary head is **one** logit
(``stage2/model.py::Stage2Model.forward -> "tbox_logit"``, shape ``(B,)``, raw), and stacking
it as ``[0, z]`` makes that multi-class fit *identically* the binary one:
``softmax([0, βz])[1] = σ(βz)``, so ``Σ_i [logsumexp(0, βz_i) − βz_i·y_i]`` is exactly the
binary cross-entropy at logit ``βz``. That is the same reduction ``stage2/losses.py`` already
uses to route the binary term through the audited focal-CE kernel. A second temperature fit
would mean fixing one copy and shipping the bug in the other, so there is one fit and this
module adapts its *inputs* — the non-tautological guard on the shared kernel is the
hand-derived closed-form root in ``tests/unit/test_recalibrate.py``, not agreement with the
thing being delegated to.

**"T is fitted on ``calib``, never on test" is enforced structurally, not asserted after the
fact.** :func:`temperature_scale` takes a per-row rung label and fits on the ``calib`` rows it
selects itself; it cannot be handed a graded row to fit on, because it never fits on a row it
did not select by name. It refuses an empty selection (a mask whose legitimate value is never
zero is the one that must raise — a vacuously-clean filter reads exactly like a working one),
refuses any rung token outside the pinned vocabulary (so a stringified null or a typo cannot
silently evaporate a row), and records ``n_by_rung`` over *every* rung so what was excluded is
auditable rather than assumed. :func:`assign_rung` re-derives the P3-02 invariant — a ``calib``
row lives in the ``fold_random == "train"`` rung — at the point of use rather than trusting it
upstream.

**No deployment prior is pinned here, and none is invented.** PRD.md:188 and ADR-0005:80 give
the genome-scale prevalence only as prose, ``~10³–10⁴:1`` (negatives per positive); no number
for it exists anywhere in the repo. Pinning a scalar π_deploy would be a new blinded-frozen
default requiring ADR sign-off (CLAUDE.md §7 item 2), and defaulting one silently would put a
fabricated quantity into every downstream posterior (§10.3). So :func:`prior_shift` takes
**both** priors as required keyword arguments with no defaults, :func:`calibrated_posterior`
refuses a half-specified shift, and :data:`DEPLOYMENT_PRIOR_RANGE` is exposed only as the
PRD-quoted band a supplied prior can be *reported* against
(:func:`deployment_prior_in_prd_band`) — never as a value the code supplies on its own. The
prior the scan actually deploys against is P5 business.

Structure: pinned vocabulary and the PRD band; pure numeric helpers; the two stack stages; the
producer that exposes both posteriors. numpy-only (no scipy, no sklearn, no torch), so the
unit tier runs in CI as-is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tbox_finder import masking
from tbox_finder.calib.temperature import (
    BETA_BRACKET,
    FIT_MAX_ITER,
    FIT_TOL,
    STACK_ORDER,
    TemperatureFit,
    TemperatureFitError,
    apply_temperature,
    fit_temperature,
)

__all__ = [
    "CALIB_RUNG",
    "DEPLOYMENT_PRIOR_ODDS_RANGE",
    "DEPLOYMENT_PRIOR_RANGE",
    "GATED_POSTERIOR_KEY",
    "GRADED_RUNGS",
    "NAMED_POSTERIOR_KEY",
    "PRIOR_SHIFTED_POSTERIOR_KEY",
    "RUNG_VOCABULARY",
    "STACK_ORDER",
    "TemperatureFitError",
    "TemperatureScaleResult",
    "apply_temperature",
    "assign_rung",
    "calibrated_posterior",
    "deployment_prior_in_prd_band",
    "log_odds_shift",
    "odds_ratio_from_prior",
    "posterior_from_logits",
    "prior_from_odds_ratio",
    "prior_shift",
    "rung_labels",
    "temperature_scale",
]
# ``TemperatureFitError`` and ``apply_temperature`` are re-exported so running one stack does
# not mean importing two calibration modules. They are the same objects, not wrappers — there
# is one temperature implementation and this module adapts its inputs, never its arithmetic.

STEP = "P3-07"
GENERATED_BY = "src/tbox_finder/calib/recalibrate.py"
ADR = "ADR-0005 D11 (stack order + named posterior); PRD §11/§12"

# --------------------------------------------------------------------------- #
# Pinned vocabulary (PRD §9.2 scheme A + the P3-02 carve) and the PRD prior band
# --------------------------------------------------------------------------- #

#: The rung ``T`` is fitted on — the P3-02 disjoint calibration carve (``calib`` column of
#: ``data/processed/splits/split_assignments.parquet``, a boolean overlay on the ``train``
#: rung, whole-cluster-assigned and asserted disjoint from val/test at build time).
CALIB_RUNG = "calib"

#: The rungs GATE-2 and GATE-4 grade on. Named so the refusal message can say *why* a row is
#: not fittable, rather than reporting an anonymous count.
GRADED_RUNGS: tuple[str, ...] = ("val", "test")

#: The closed vocabulary :func:`temperature_scale` accepts. Closed on purpose: an unknown
#: token is an error, not a row to skip. A rung column that arrives carrying ``"None"`` or
#: ``"Train"`` would otherwise be filtered out silently and the fit would run clean on a
#: quietly smaller set ([[stringified-null-survives-missing-checks]]).
RUNG_VOCABULARY: tuple[str, ...] = ("calib", "train", "val", "test")

#: PRD.md:188 / ADR-0005:80 state the genome-scale deployment prevalence as ``~10³–10⁴:1``
#: **negatives per positive** — the same units as ADR-0005 D7's pinned ``100:1`` benchmark
#: decoy prevalence and §9.1's ``~10:1`` training seed ratio. It is a **prose band**, not a
#: pinned scalar: nothing in the repo encodes a deployment prior, and this module does not
#: create one (see the module docstring).
DEPLOYMENT_PRIOR_ODDS_RANGE: tuple[float, float] = (1e3, 1e4)

#: The same band as a positive-class prior, ``π = 1/(1 + negatives_per_positive)``, ascending.
#: Reporting-only: :func:`deployment_prior_in_prd_band` reads it, no function defaults to it.
DEPLOYMENT_PRIOR_RANGE: tuple[float, float] = (
    1.0 / (1.0 + DEPLOYMENT_PRIOR_ODDS_RANGE[1]),
    1.0 / (1.0 + DEPLOYMENT_PRIOR_ODDS_RANGE[0]),
)

#: Keys of the :func:`calibrated_posterior` payload. ``GATED_POSTERIOR_KEY`` is the D11 named
#: posterior — the temperature-scaled, **pre**-prior-shift object GATE-2 grades.
NAMED_POSTERIOR_KEY = "named_posterior"
PRIOR_SHIFTED_POSTERIOR_KEY = "prior_shifted_posterior"
GATED_POSTERIOR_KEY = NAMED_POSTERIOR_KEY


@dataclass(frozen=True)
class TemperatureScaleResult:
    """A temperature certified to have been fitted on the ``calib`` rung and nothing else.

    ``n_by_rung`` is the census of the *whole* input, as sorted ``(rung, count)`` pairs — the
    excluded rows are counted, not discarded silently, so a reader can tell "fitted on the
    859 calib rows of a 30,542-row table" from "fitted on the 859 rows someone handed me".
    ``fit`` is the underlying :class:`~tbox_finder.calib.temperature.TemperatureFit`, carrying
    the stationarity residual so convergence can be re-derived rather than trusted.
    """

    temperature: float
    beta: float
    fitted_on: str
    n_fitted: int
    n_total: int
    n_by_rung: tuple[tuple[str, int], ...]
    fit: TemperatureFit

    def counts(self) -> dict[str, int]:
        """``n_by_rung`` as a mapping (the dataclass stores pairs so it stays hashable)."""
        return dict(self.n_by_rung)


# --------------------------------------------------------------------------- #
# Pure numeric helpers
# --------------------------------------------------------------------------- #
def _require_prior(value: Any, name: str) -> float:
    """A class prior must be a real number strictly inside ``(0, 1)``.

    Strictly: ``π = 0`` or ``π = 1`` sends the log-odds to ``∓inf``, and a shift of ``inf``
    is not a correction, it is a constant posterior wearing one. ``bool`` is rejected before
    the range test because ``bool`` is an ``int`` subclass and ``True`` would otherwise read
    as the prior ``1.0``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValueError(f"{name} must be a real number in (0, 1), got {value!r}")
    prior = float(value)
    if not math.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError(f"{name} must be finite and strictly inside (0, 1), got {prior!r}")
    return prior


def prior_from_odds_ratio(negatives_per_positive: float) -> float:
    """Positive-class prior ``π = 1/(1 + r)`` from an ``r : 1`` negative:positive ratio.

    The repo states every prevalence as a negative:positive ratio (ADR-0005 D7's ``100:1``
    benchmark, §9.1's ``~10:1`` training seed, PRD.md:188's ``~10³–10⁴:1`` genome scale), and
    every formula here wants a prior. One conversion, so the two never drift.
    """
    if isinstance(negatives_per_positive, bool) or not isinstance(
        negatives_per_positive, (int, float, np.floating, np.integer)
    ):
        raise ValueError(
            f"negatives_per_positive must be a real number, got {negatives_per_positive!r}"
        )
    ratio = float(negatives_per_positive)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"negatives_per_positive must be finite and > 0, got {ratio!r}")
    return 1.0 / (1.0 + ratio)


def odds_ratio_from_prior(prior: float) -> float:
    """Inverse of :func:`prior_from_odds_ratio`: ``r = (1 − π)/π``."""
    p = _require_prior(prior, "prior")
    return (1.0 - p) / p


def deployment_prior_in_prd_band(prior: float) -> bool:
    """Does ``prior`` sit inside the PRD/ADR-quoted ``~10³–10⁴:1`` genome-scale band?

    **Reporting only.** A ``False`` is not a refusal — the deployment prior a corpus actually
    carries is a P5 quantity, and PRD §7.2 explicitly contemplates near-zero-prior corpora
    (Archaea/DPANN) with their *own* prior outside this Bacteria-derived band. This predicate
    exists so an artifact can say which side of the quoted band its prior fell on instead of
    leaving a reader to eyeball it.
    """
    p = _require_prior(prior, "prior")
    lo, hi = DEPLOYMENT_PRIOR_RANGE
    return lo <= p <= hi


def _as_logits(logits: Any, *, name: str = "logits") -> np.ndarray:
    """1-D float64 view of a raw binary logit vector, non-finite refused."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError(f"{name} must be 1-D (one raw logit per row), got shape {z.shape}")
    if z.size and not np.isfinite(z).all():
        raise ValueError(f"{name} carries a non-finite value")
    return z


def posterior_from_logits(logits: Any) -> np.ndarray:
    """``σ(z)`` computed branch-wise so neither tail overflows.

    ``exp(-z)`` overflows for very negative ``z`` and ``exp(z)`` for very positive ``z``; each
    branch below uses only the exponential that underflows harmlessly toward 0.
    """
    z = _as_logits(logits)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_odds_shift(*, source_prior: float, target_prior: float) -> float:
    """The Saerens/Elkan additive log-odds correction ``logit(π_target) − logit(π_source)``.

    Direction is source → target: the classifier's posterior is calibrated at ``source_prior``
    (for the D11 stack, the prevalence of the ``calib`` split ``T`` was fitted on), and the
    correction moves it to ``target_prior``. Both are required and neither is defaulted, so
    the direction cannot be inverted by omission.
    """
    p_s = _require_prior(source_prior, "source_prior")
    p_t = _require_prior(target_prior, "target_prior")
    return math.log(p_t / (1.0 - p_t)) - math.log(p_s / (1.0 - p_s))


# --------------------------------------------------------------------------- #
# Stack stage 1 — temperature scaling, fitted on `calib` and nowhere else
# --------------------------------------------------------------------------- #
def assign_rung(*, calib: Any, fold_random: Any) -> str:
    """The rung of one row, from its ``calib`` flag and its scheme-A ``fold_random`` value.

    The single derivation of the rung label, so the ``calib``-overlay-on-``train`` shape lives
    in one place. Fail-closed in three directions, each of which is a real failure mode rather
    than a hypothetical:

    * a **missing** ``calib`` or ``fold_random`` raises — tested with
      :func:`tbox_finder.masking.is_missing`, because under pandas 3 a null string cell
      arrives as ``NaN``, which is *truthy*, so ``bool(value)`` would read a missing calib
      flag as ``True`` and a missing fold as the string ``"nan"``;
    * a ``calib`` row outside the ``train`` rung raises — that is the P3-02 invariant
      (``splits._assert_calib_disjoint``), re-derived here at the point where violating it
      would corrupt a gate rather than trusted from upstream;
    * an unrecognised ``fold_random`` raises, instead of being bucketed as "other".
    """
    if masking.is_missing(calib):
        raise ValueError("calib is missing — a row with no calibration flag has no rung")
    fold = masking.row_text(fold_random).strip()
    if not fold:
        raise ValueError("fold_random is missing — a row with no scheme-A fold has no rung")
    in_calib = bool(calib)
    if in_calib:
        if fold != "train":
            raise ValueError(
                f"calib row carries fold_random={fold!r}, but the P3-02 carve is taken from "
                "the training folds only (splits._assert_calib_disjoint); a calib row in a "
                "graded fold is leakage, not a rung"
            )
        return CALIB_RUNG
    if fold not in RUNG_VOCABULARY:
        raise ValueError(f"unrecognised fold_random {fold!r}; known: {sorted(RUNG_VOCABULARY)}")
    if fold == CALIB_RUNG:
        raise ValueError(
            "fold_random == 'calib' with calib False — calib is a boolean overlay on the "
            "'train' rung, never a fourth fold_random value"
        )
    return fold


def rung_labels(*, calib: Sequence[Any], fold_random: Sequence[Any]) -> list[str]:
    """:func:`assign_rung` over two equal-length columns, erroring with the offending index."""
    if len(calib) != len(fold_random):
        raise ValueError(
            f"calib carries {len(calib)} rows but fold_random carries {len(fold_random)}"
        )
    out: list[str] = []
    for i, (c, f) in enumerate(zip(calib, fold_random, strict=True)):
        try:
            out.append(assign_rung(calib=c, fold_random=f))
        except ValueError as exc:
            raise ValueError(f"row {i}: {exc}") from exc
    return out


def _rung_census(rungs: Sequence[str]) -> tuple[tuple[str, int], ...]:
    counts = {rung: 0 for rung in RUNG_VOCABULARY}
    for rung in rungs:
        counts[rung] += 1
    return tuple(sorted(counts.items()))


def temperature_scale(
    logits: Any,
    labels: Any,
    *,
    rung: Sequence[Any],
    bracket: tuple[float, float] = BETA_BRACKET,
    tol: float = FIT_TOL,
    max_iter: int = FIT_MAX_ITER,
    grad_tol: float | None = None,
) -> TemperatureScaleResult:
    """Fit one shared scalar ``T`` on the ``calib`` rows of ``(logits, labels)`` (D11 stage 2).

    ``logits`` is the Stage-2 binary head's raw output, one logit per row
    (``stage2/model.py`` ``"tbox_logit"``, shape ``(B,)``); ``labels`` the matching 0/1
    targets; ``rung`` the per-row label from :func:`rung_labels`. The whole table is passed in
    and this function does the selecting, which is what makes "fitted on ``calib``, never on
    test" a property of the code rather than a claim about the caller.

    The fit itself is :func:`tbox_finder.calib.temperature.fit_temperature` on the ``[0, z]``
    stack — exact, since ``softmax([0, βz])[1] = σ(βz)`` makes its multi-class NLL identically
    the binary cross-entropy at ``βz``. Its degenerate-limit refusals therefore carry over
    unchanged: a perfectly-separated calibration set (``β → ∞``, ``T → 0``) and a
    worse-than-uniform one (``β → 0``, ``T → ∞``) both raise
    :class:`~tbox_finder.calib.temperature.TemperatureFitError` by name instead of returning a
    bracket endpoint dressed as a temperature.

    Raises ``ValueError`` when the selection is empty or single-class — the two ways this can
    run "successfully" on nothing. An all-``False`` calib column is exactly as clean-looking as
    a working filter, and a single-class calibration set has no calibration to measure.
    """
    z = _as_logits(logits)
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {y.shape}")
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"logits carries {z.shape[0]} rows but labels carries {y.shape[0]}")
    if len(rung) != z.shape[0]:
        raise ValueError(f"logits carries {z.shape[0]} rows but rung carries {len(rung)}")
    y = y.astype(np.int64, copy=False)
    if y.size and (int(y.min()) < 0 or int(y.max()) > 1):
        raise ValueError(
            f"labels must be binary 0/1, got range [{int(y.min())}, {int(y.max())}] — the "
            "Stage-2 binary head is one logit, not a multi-class score"
        )

    rungs = [masking.row_text(r).strip() for r in rung]
    unknown = sorted({r for r in rungs if r not in RUNG_VOCABULARY})
    if unknown:
        raise ValueError(
            f"unrecognised rung label(s) {unknown}; known: {sorted(RUNG_VOCABULARY)}. An "
            "unknown token is refused rather than filtered: a row that silently vanishes "
            "shrinks the calibration set while every count still looks clean"
        )

    keep = np.array([r == CALIB_RUNG for r in rungs], dtype=bool)
    n_fitted = int(keep.sum())
    if n_fitted == 0:
        census = dict(_rung_census(rungs))
        n_graded = sum(census[r] for r in GRADED_RUNGS)
        raise ValueError(
            f"no rows on the {CALIB_RUNG!r} rung out of {len(rungs)} — an empty calibration "
            f"split is not a fit (census: {census}). GATE-2's T is fitted on the P3-02 carve; "
            "if it is empty the carve or the join is wrong, and falling back to the remainder "
            f"would fit T on the {n_graded} rows of the graded rungs {list(GRADED_RUNGS)}"
        )
    y_fit = y[keep]
    n_pos = int(y_fit.sum())
    if n_pos == 0 or n_pos == n_fitted:
        raise ValueError(
            f"the {CALIB_RUNG!r} rung is single-class ({n_pos} positive of {n_fitted}) — "
            "there is no calibration to fit on a set with only one outcome"
        )

    z_fit = z[keep]
    stacked = np.column_stack((np.zeros_like(z_fit), z_fit))
    fit = fit_temperature(
        stacked,
        y_fit,
        bracket=bracket,
        tol=tol,
        max_iter=max_iter,
        grad_tol=grad_tol,
    )
    return TemperatureScaleResult(
        temperature=fit.temperature,
        beta=fit.beta,
        fitted_on=CALIB_RUNG,
        n_fitted=n_fitted,
        n_total=len(rungs),
        n_by_rung=_rung_census(rungs),
        fit=fit,
    )


# --------------------------------------------------------------------------- #
# Stack stage 3 — the Saerens/Elkan prior-shift
# --------------------------------------------------------------------------- #
def prior_shift(logits: Any, *, source_prior: float, target_prior: float) -> np.ndarray:
    """Shift calibrated **logits** from ``source_prior`` to ``target_prior`` (D11 stage 3).

    The Saerens/Elkan correction in the log-odds form ADR-0005 D11 names [Saerens, Latinne &
    Decaestecker, *Neural Computation* 14(1):21–41, DOI:10.1162/089976602753284446 (accessed
    2026-08-03); Elkan, *The foundations of cost-sensitive learning*, IJCAI 2001, Theorem 1]:
    under a pure prior shift the class-conditional densities are unchanged, so the likelihood
    ratio is unchanged and the whole correction is the additive constant
    ``logit(π_t) − logit(π_s)``::

        z' = z + log(π_t/(1−π_t)) − log(π_s/(1−π_s))

    **Scope, stated because the two halves of that paper are not interchangeable.** This is
    the **known-target-prior closed form**. Saerens et al.'s headline contribution is the *EM
    procedure* for the case where π_target is *unknown* and must be estimated from unlabelled
    deployment data; that estimator is not implemented here and is not what D11 asks for —
    D11 shifts "to the deployment prior", i.e. to a prior supplied from outside. If P5 ever
    needs π_deploy estimated rather than stated, that is the EM half of the same paper and a
    separate piece of machinery, not a parameter of this one.

    Equivalently on the posterior, ``p' = p·r / (p·r + (1−p))`` with
    ``r = [π_t/(1−π_t)]·[(1−π_s)/π_s]``. Returning **logits** rather than posteriors is
    deliberate: the correction is exactly additive there, so it composes with a second shift
    and inverts exactly, while the posterior form round-trips only up to float error.

    Being an additive constant, it is strictly monotone in ``z`` — it moves every posterior
    but reorders none, which is why the operating point is set on the ranking and the prior
    shift changes only what the score *means*. It is **not** the gated object: a prior-shifted
    posterior is miscalibrated at benchmark prevalence by construction, so GATE-2 grades the
    pre-shift named posterior and this one is reported non-gated (D11).
    """
    z = _as_logits(logits)
    return z + log_odds_shift(source_prior=source_prior, target_prior=target_prior)


# --------------------------------------------------------------------------- #
# The producer — both posteriors, and which one is gated
# --------------------------------------------------------------------------- #
def calibrated_posterior(
    logits: Any,
    *,
    temperature: float,
    source_prior: float | None = None,
    target_prior: float | None = None,
) -> dict[str, Any]:
    """The D11 stack applied to raw Stage-2 binary logits, exposing **both** posteriors.

    Returns ``named_posterior`` — ``σ(z/T)``, the temperature-scaled, pre-prior-shift object
    GATE-2's in-distribution ECE ≤ 0.05 is graded on — and ``prior_shifted_posterior``, the
    Saerens/Elkan-corrected one, which is ``None`` unless both priors are given.
    ``gated_posterior_key`` names the former inside the payload so a downstream report cannot
    quietly grade the wrong one.

    Both priors or neither: a half-specified shift raises rather than being completed with a
    default, because the only defaultable value would be a deployment prior the repo does not
    have (the PRD gives ``~10³–10⁴:1`` as prose, and inventing a scalar from it is exactly the
    fabrication CLAUDE.md §10.3 forbids).
    """
    if (source_prior is None) != (target_prior is None):
        missing = "target_prior" if target_prior is None else "source_prior"
        raise ValueError(
            f"prior_shift needs both source_prior and target_prior; {missing} is missing. "
            "Neither is defaulted: the deployment prior is PRD prose (~10^3-10^4:1), not a "
            "value this module may invent"
        )
    tempered = apply_temperature(_as_logits(logits), temperature)
    named = posterior_from_logits(tempered)

    payload: dict[str, Any] = {
        "stack_order": list(STACK_ORDER),
        "temperature": float(temperature),
        NAMED_POSTERIOR_KEY: named,
        PRIOR_SHIFTED_POSTERIOR_KEY: None,
        "gated_posterior_key": GATED_POSTERIOR_KEY,
        "prior_shift_applied": False,
        "log_odds_shift": None,
        "source_prior": None,
        "target_prior": None,
        "target_prior_in_prd_band": None,
        "n_rows": int(named.shape[0]),
    }
    if source_prior is None:
        payload["stack_applied"] = ["train", "temperature_scale"]
        return payload

    shift = log_odds_shift(source_prior=source_prior, target_prior=target_prior)
    payload[PRIOR_SHIFTED_POSTERIOR_KEY] = posterior_from_logits(tempered + shift)
    payload["prior_shift_applied"] = True
    payload["log_odds_shift"] = shift
    payload["source_prior"] = _require_prior(source_prior, "source_prior")
    payload["target_prior"] = _require_prior(target_prior, "target_prior")
    payload["target_prior_in_prd_band"] = deployment_prior_in_prd_band(target_prior)
    payload["stack_applied"] = list(STACK_ORDER)
    return payload
