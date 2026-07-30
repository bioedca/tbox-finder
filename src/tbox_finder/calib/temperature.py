"""Stage-1 temperature scaling + the non-gated P2-13 reliability read (ADR-0005 D11 + A11).

What this module is, stated precisely because the words *calibration* and *ECE* name a
**gated** quantity elsewhere in this project and this is not it:

* **D11 pins** the recalibration stack order (*train → temperature-scale on a disjoint
  calibration split → prior-shift*), the binned-ECE estimator (15 equal-mass bins,
  debiased), and the gated object (the *named posterior*, pre-prior-shift) — graded at
  **P3 exit**, on the **P3-02 ``calib`` carve**, which does not exist yet.
* **A11 pins**, for Stage 1: the axis (the per-nucleotide 8-class logit vector), the form
  (**one shared scalar T**, the method of the arXiv:1706.04599 paper D11 cites), the fit
  (an exact **convex** 1-D solve in ``β = 1/T`` — no seed, no optimiser, no torch), where
  T sits (**before** the frozen D3+A3 reconciliation operator, i.e. the deployment order),
  and the P2 fold (cluster-disjoint halves of the P2-06a ``selection_val`` rung: fit on
  half A, read on half B).
* Therefore: **nothing here is a calibration claim, and no T fitted here is shipped.**
  ``selection_val`` is the fold the P2-06 sweep *selected* on, so it is not D11's disjoint
  split; GATE-2's T is re-fitted at P3-02/P3. The report carries ``gated: false`` and every
  disclosure as a **constant**, not as prose someone has to remember to repeat.

Two consequences worth reading before trusting a number out of here:

1. **A single T is not automatically GATE-4-neutral.** Scaling by ``T > 0`` preserves each
   *window's* own arg-max exactly (it is strictly monotone), but the D3+A3 operator averages
   *probabilities* across overlapping windows, so re-tempering the summands can change which
   class wins that average wherever coverage >= 2. The read therefore **measures**
   ``n_positions_argmax_changed`` (and the per-class F1 / boundary-IoU delta) instead of
   asserting invariance. P2-14 grades GATE-4 on the **uncalibrated** posterior.
2. **The fit objective and the read are deliberately different objects.** The fit minimises
   the **per-window** NLL (that is the convex one); the read scores the **reconciled**
   posterior (that is the deployed one). Both NLLs are reported.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder import metrics as M
from tbox_finder.labels import CLASS_ORDER, CORE_ELEMENTS

SCHEMA_VERSION = "1"
STEP = "P2-13"
GENERATED_BY = "src/tbox_finder/calib/temperature.py"
ADR = "ADR-0005 D11 + A11; ADR-0005 D3+A3 (operator, consumed)"
ENV_LOCK = "envs/ml-dna.conda-lock.yml"

DEFAULT_REPORT = Path("reports/p2/calibration.json")
DEFAULT_FIGURE_DATA = Path("reports/p2/calibration_figure_data.json")
FIGURES_DIR = Path("figures/calib")

#: The frozen scan/eval geometry (ADR-0005 D3 + A3) — held, never swept here.
WINDOW_NT = 1024
STRIDE_NT = 512
DEFAULT_BATCH_SIZE = 8

#: A11 Pin 5: the seeded whole-cluster halving of the ``selection_val`` rung. The seed is
#: **not** ``SELECTION_VAL_SEED`` — reusing that seed would make the calibration halves a
#: deterministic function of the same permutation that carved the rung, which is a
#: correlation with no upside and an argument to have to make later.
CALIB_SPLIT_SEED = 20260730
CALIB_FIT_FRACTION = 0.5

#: A11 Pin 3: the bracket the convex root-find works inside, in ``β = 1/T``. The endpoints
#: are the degenerate limits, and hitting one is an **error**, never a returned bound:
#: ``β → 0`` (T → ∞) is a worse-than-uniform fold, ``β → ∞`` (T → 0) is per-position perfect
#: accuracy, and in both cases the minimiser does not exist inside the bracket.
BETA_BRACKET = (1e-3, 1e3)
FIT_TOL = 1e-10
FIT_MAX_ITER = 200
#: Rows per chunk in the NLL/gradient reduction — bounds peak memory on a ~1e6-row fit
#: without changing the result (the reduction is a plain sum over chunks).
FIT_CHUNK_ROWS = 200_000

#: Bootstrap replicates for the cluster-blocked ECE CI. Lower than ``block_bootstrap_ci``'s
#: 2000 default **for a measured reason**: the estimator is the frozen stdlib
#: ``binned_ece`` (~0.6 s at 6e5 positions), and 8 classes x 2 conditions x 2000 replicates
#: is ~5 h of pure-Python sorting for a non-gated read. 200 replicates give a coarse
#: percentile CI — disclosed as such, and reported in the artifact as ``n_boot``.
DEFAULT_N_BOOT = 200

#: Shipped as report constants so the scope travels with the numbers (§10.3).
DISCLOSURES: tuple[str, ...] = (
    "NON-GATED. GATE-2's in-distribution ECE <= 0.05 is graded at P3 exit (PRD.md:56/:248, "
    "ADR-0005 D11) on the P3-02 `calib` carve. No number in this report is a gate result.",
    "The fold is the P2-06a `selection_val` inner rung — the fold the P2-06 sweep SELECTED "
    "on. It is NOT D11's 'disjoint calibration split', which does not exist at P2: the "
    "`calib` column is carved from the training folds at P3-02.",
    "T is fitted on one seeded whole-cluster half of that rung and the reliability is read "
    "on the other half, so the read is out-of-sample for T; at ~235 clusters per half the "
    "cluster-blocked CI is wide, and the ECE CI is a 200-replicate percentile interval.",
    "No T from this step is shipped or consumed by any gate. GATE-2's T is re-fitted at "
    "P3-02/P3 on the `calib` carve.",
    "The prior-shift stage is refused here: this artifact stops at the NAMED posterior "
    "(temperature-scaled, pre-prior-shift). The Saerens/Elkan correction is P5.",
    "The fit objective is the PER-WINDOW multi-class NLL (the convex one); the reliability "
    "is read on the RECONCILED posterior (the deployed one). Both NLLs are reported.",
    "One shared scalar T preserves each window's arg-max but NOT necessarily the reconciled "
    "arg-max where coverage >= 2, so `n_positions_argmax_changed` is measured. P2-14 grades "
    "GATE-4 on the uncalibrated posterior.",
    "The per-nt class mix is background-dominated, so a pooled-NLL fit is dominated by "
    "background; the per-class one-vs-rest reads are reported separately for that reason, "
    "with no cross-class mean (the core-element summary is the WORST/max, mirroring GATE-4's "
    "min-over-core discipline).",
    "NOT bit-reproducible run-to-run: the fit and the halving are exactly deterministic, but "
    "the GPU forward is not (the Mamba/CUDA kernels reduce in a run-dependent order). "
    "MEASURED over two full re-runs of this artifact at the same commit and inputs: T, every "
    "count (arg-max changes, coverage, cluster halves) and gate4_core_min_f1 are IDENTICAL; "
    "every per-class ECE agrees to <= 3e-8 relative; the largest disagreement anywhere in the "
    "read is 4.5e-6 relative on a bin edge, and `temperature.grad_at_solution` — a cancelling "
    "sum over ~1.8e6 terms — differs by 2.7e-4 relative. Re-derive, do not diff for equality.",
)

STACK_ORDER: tuple[str, ...] = ("train", "temperature_scale", "prior_shift")


class TemperatureFitError(ValueError):
    """The temperature fit could not be certified — never a silently-returned fallback."""


@dataclass(frozen=True)
class TemperatureFit:
    """A certified single-scalar temperature fit.

    ``temperature`` is ``1/beta``. ``grad_at_solution`` is the per-position derivative of the
    NLL with respect to ``beta`` at the returned point: it is the residual of the stationarity
    condition, and a caller that wants to re-derive convergence rather than trust a boolean
    checks it against ``grad_tol``.
    """

    temperature: float
    beta: float
    n_positions: int
    n_classes: int
    n_iterations: int
    converged: bool
    nll_per_position_initial: float
    nll_per_position_final: float
    grad_at_solution: float
    grad_tol: float
    bracket: tuple[float, float]
    tol: float


# --------------------------------------------------------------------------- #
# The convex 1-D fit
# --------------------------------------------------------------------------- #
def _as_fit_arrays(logits: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(logits)
    y = np.asarray(labels)
    if z.ndim != 2:
        raise TemperatureFitError(f"logits must be (n_positions, n_classes), got shape {z.shape}")
    if y.ndim != 1:
        raise TemperatureFitError(f"labels must be (n_positions,), got shape {y.shape}")
    if z.shape[0] != y.shape[0]:
        raise TemperatureFitError(
            f"logits carries {z.shape[0]} positions but labels carries {y.shape[0]}"
        )
    if z.shape[0] == 0:
        raise TemperatureFitError("no positions to fit — an empty calibration set is not a fit")
    if z.shape[1] < 2:
        raise TemperatureFitError(f"need >= 2 classes to temperature-scale, got {z.shape[1]}")
    if not np.isfinite(z).all():
        raise TemperatureFitError("logits carry a non-finite value")
    y = y.astype(np.int64, copy=False)
    if int(y.min()) < 0 or int(y.max()) >= z.shape[1]:
        raise TemperatureFitError(
            f"labels out of range: [{int(y.min())}, {int(y.max())}] for {z.shape[1]} classes"
        )
    return z, y


def _nll_and_grad(z: np.ndarray, y: np.ndarray, beta: float) -> tuple[float, float]:
    """Total NLL and ``dNLL/dbeta`` at ``beta`` for ``p = softmax(beta * z)``.

    ``NLL(β) = Σ_i [ logsumexp_c(β·z_ic) − β·z_{i,y_i} ]`` — convex in ``β`` (log-sum-exp
    composed with a linear map, less a linear term) — and
    ``dNLL/dβ = Σ_i [ Σ_c p_ic·z_ic − z_{i,y_i} ]``, which is therefore monotone increasing.
    Reduced in row chunks so a ~1e6-row fit does not materialise a second copy of the logits;
    the reduction is a plain sum, so chunking does not change the value beyond float
    associativity within a fixed chunk size.
    """
    n = z.shape[0]
    nll = 0.0
    grad = 0.0
    for lo in range(0, n, FIT_CHUNK_ROWS):
        hi = min(lo + FIT_CHUNK_ROWS, n)
        zc = z[lo:hi].astype(np.float64, copy=False)
        yc = y[lo:hi]
        scaled = beta * zc
        mx = scaled.max(axis=1, keepdims=True)
        ex = np.exp(scaled - mx)
        denom = ex.sum(axis=1, keepdims=True)
        lse = (mx[:, 0] + np.log(denom[:, 0])).sum()
        true_logit = np.take_along_axis(zc, yc[:, None], axis=1)[:, 0]
        nll += float(lse - beta * true_logit.sum())
        p = ex / denom
        grad += float((p * zc).sum() - true_logit.sum())
    return nll, grad


def fit_temperature(
    logits: Any,
    labels: Any,
    *,
    bracket: tuple[float, float] = BETA_BRACKET,
    tol: float = FIT_TOL,
    max_iter: int = FIT_MAX_ITER,
    grad_tol: float | None = None,
) -> TemperatureFit:
    """Fit one shared scalar temperature ``T`` by exact convex minimisation (A11 Pin 2/3).

    ``logits`` is ``(n_positions, n_classes)`` of **raw** model output, ``labels`` the
    matching integer class indices. The multi-class NLL is convex in ``β = 1/T`` with a
    monotone derivative, so the minimiser is the unique root of that derivative and is found
    by **bisection inside a pinned bracket** — deterministic, seedless, optimiser-free, and
    numpy-only, which is why the unit tier can run it in bare CI (no torch there).

    Raises :class:`TemperatureFitError` — never returns a bracket endpoint dressed as a
    solution — when the derivative does not change sign inside ``bracket``. That is the
    degenerate case, and it has two directions worth telling apart: a fold the model is
    perfect on drives ``β → ∞`` (``T → 0``), a fold it is worse-than-uniform on drives
    ``β → 0`` (``T → ∞``). Both are reported by name.
    """
    z, y = _as_fit_arrays(logits, labels)
    lo, hi = float(bracket[0]), float(bracket[1])
    if not (0.0 < lo < hi):
        raise TemperatureFitError(f"bracket must satisfy 0 < lo < hi, got {bracket!r}")
    if tol <= 0.0:
        raise TemperatureFitError(f"tol must be positive, got {tol!r}")
    if max_iter <= 0:
        raise TemperatureFitError(f"max_iter must be positive, got {max_iter!r}")
    n = int(z.shape[0])
    if grad_tol is None:
        # Scale-free: the gradient is a sum over positions, so its natural yardstick is
        # per-position. 1e-6 nats per unit β per position is far below any decision this
        # number feeds, and does not tighten with N (which an absolute tolerance would).
        grad_tol = 1e-6 * n

    nll_initial, _ = _nll_and_grad(z, y, 1.0)
    _, g_lo = _nll_and_grad(z, y, lo)
    _, g_hi = _nll_and_grad(z, y, hi)

    # The β → ∞ limit of dNLL/dβ, computed EXACTLY rather than sampled at ``hi``:
    # ``Σ_i (max_c z_ic − z_{i,y_i})`` ≥ 0, and it is 0 **iff every position's arg-max is
    # already its own label**. In that case the NLL decreases monotonically for all β and the
    # minimiser does not exist — but the sampled gradient at large β *underflows to exactly
    # 0.0*, which the ``|grad| <= grad_tol`` stopping rule would read as stationarity and
    # certify an arbitrary interior point as "the" temperature. Sampling cannot tell a
    # converged root from an underflowed tail; this limit can.
    margin_sum = 0.0
    for lo_i in range(0, z.shape[0], FIT_CHUNK_ROWS):
        hi_i = min(lo_i + FIT_CHUNK_ROWS, z.shape[0])
        zc = z[lo_i:hi_i].astype(np.float64, copy=False)
        yc = y[lo_i:hi_i]
        margin_sum += float(
            (zc.max(axis=1) - np.take_along_axis(zc, yc[:, None], axis=1)[:, 0]).sum()
        )

    if g_lo > 0.0:
        raise TemperatureFitError(
            f"NLL is already increasing at beta={lo:g} (T={1 / lo:g}): the minimiser lies "
            f"outside the bracket at T > {1 / lo:g}. This fold is worse than uniform under "
            f"the model — dNLL/dbeta({lo:g})={g_lo:.6g} > 0 over {n} positions."
        )
    if margin_sum <= 0.0:
        raise TemperatureFitError(
            f"every one of the {n} positions is already its own arg-max, so NLL(beta) "
            f"decreases monotonically and the minimiser is the near-perfect-accuracy limit "
            f"beta -> inf (T -> 0). There is no temperature to fit on this set."
        )
    if g_hi <= 0.0:
        raise TemperatureFitError(
            f"NLL is still decreasing at beta={hi:g} (T={1 / hi:g}): the minimiser lies "
            f"outside the bracket at T < {1 / hi:g}. This is the degenerate near-perfect-"
            f"accuracy direction (T -> 0 sharpens every position onto its own arg-max) — "
            f"dNLL/dbeta({hi:g})={g_hi:.6g} <= 0 over {n} positions, against an exact "
            f"beta -> inf limit of {margin_sum:.6g}."
        )

    iterations = 0
    converged = False
    beta = 0.5 * (lo + hi)
    grad = float("nan")
    while iterations < max_iter:
        iterations += 1
        beta = 0.5 * (lo + hi)
        _, grad = _nll_and_grad(z, y, beta)
        if abs(grad) <= grad_tol:
            converged = True
            break
        if (hi - lo) <= tol * max(1.0, beta):
            # ⚠ Width alone must NOT certify convergence. It used to, and that made the two
            # exits disagree: a width-terminated fit returned `converged=True` carrying a
            # gradient larger than `grad_tol`, which is precisely what `validate_report`
            # rejects — the module could emit a report its own validator refuses (the P2-12
            # `build_report`-vs-`validate_report` shape, one layer in). If the bracket
            # collapses before the stationarity residual is reached, no T is certified.
            raise TemperatureFitError(
                f"bracket collapsed to width {hi - lo:.3g} at beta={beta:.6g} while "
                f"|dNLL/dbeta|={abs(grad):.6g} still exceeds grad_tol {grad_tol:.3g}: the "
                "stationarity residual is not attainable at this scale, so no T is certified"
            )
        if grad > 0.0:
            hi = beta
        else:
            lo = beta
    if not converged:
        raise TemperatureFitError(
            f"bisection did not converge in {max_iter} iterations: bracket width "
            f"{hi - lo:.3g}, |dNLL/dbeta|={abs(grad):.6g} > {grad_tol:.3g}"
        )

    nll_final, _ = _nll_and_grad(z, y, beta)
    if not math.isfinite(nll_final) or nll_final > nll_initial:
        # A converged minimiser of a convex objective cannot score worse than beta=1. If it
        # does, the arithmetic — not the data — is wrong, and a fabricated T is worse than
        # no T (§10.3).
        raise TemperatureFitError(
            f"fit did not improve the objective it minimises: NLL(beta=1)={nll_initial:.6g} "
            f"-> NLL(beta={beta:.6g})={nll_final:.6g}"
        )
    return TemperatureFit(
        temperature=1.0 / beta,
        beta=beta,
        n_positions=n,
        n_classes=int(z.shape[1]),
        n_iterations=iterations,
        converged=True,
        nll_per_position_initial=nll_initial / n,
        nll_per_position_final=nll_final / n,
        grad_at_solution=grad,
        grad_tol=float(grad_tol),
        bracket=(float(bracket[0]), float(bracket[1])),
        tol=float(tol),
    )


def apply_temperature(logits: Any, temperature: float) -> np.ndarray:
    """``logits / T`` (A11 Pin 2), as float64. ``T`` must be finite and > 0."""
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TemperatureFitError(f"temperature must be a real number, got {temperature!r}")
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise TemperatureFitError(f"temperature must be finite and positive, got {t!r}")
    return np.asarray(logits, dtype=np.float64) / t


# --------------------------------------------------------------------------- #
# The P2 fold: a seeded whole-cluster halving of the selection_val rung (A11 Pin 5)
# --------------------------------------------------------------------------- #
def calibration_fit_cluster_ids(
    cluster_sizes: Mapping[int, int],
    *,
    fraction: float = CALIB_FIT_FRACTION,
    seed: int = CALIB_SPLIT_SEED,
) -> frozenset[int]:
    """The whole clusters T is **fitted** on; the complement is the read half.

    ``cluster_sizes`` maps ``cluster_id -> record count`` over the ``selection_val`` rung.
    Whole clusters are drawn in a seeded permutation until ``fraction`` of the rung's
    *records* is reached — so no homology cluster straddles the fit/read boundary, the same
    rule PRD §9.2 imposes on every other fold boundary in this project, applied one level in.
    Deterministic in the mapping's *content*, not its iteration order (ids are sorted before
    the permutation), so two callers building the mapping differently get the same halves.

    Mirrors :func:`window_dataset.selection_val_cluster_ids` deliberately: same shape, same
    guards, same seeded-permutation semantics. It is a *different* rule (a 50/50 halving of
    the rung, not a 10 % carve out of the training fold) with a different seed, so it is a
    sibling rather than a caller — but any future change to one should be read against the
    other.
    """
    if not isinstance(fraction, float) or not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be a float in (0, 1), got {fraction!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {seed!r}")
    ids = sorted(int(c) for c in cluster_sizes)
    if not ids:
        raise ValueError("cluster_sizes is empty — there is no selection_val rung to halve")
    sizes = {int(c): int(cluster_sizes[c]) for c in cluster_sizes}
    if any(n <= 0 for n in sizes.values()):
        raise ValueError("cluster_sizes carries a non-positive count")
    target = fraction * sum(sizes.values())

    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    held = 0
    for idx in rng.permutation(len(ids)):
        if held >= target:
            break
        cid = ids[int(idx)]
        chosen.add(cid)
        held += sizes[cid]
    if not chosen or len(chosen) == len(ids):
        raise ValueError(
            f"halving degenerate: {len(chosen)} of {len(ids)} clusters chosen — an empty "
            "half, or the whole rung, is not a fit/read split"
        )
    return frozenset(chosen)


# --------------------------------------------------------------------------- #
# The reliability read (per-class one-vs-rest, through the FROZEN estimator)
# --------------------------------------------------------------------------- #
def per_class_reliability(
    y_true: Sequence[int],
    probs: Any,
    *,
    n_bins: int = M.ECE_N_BINS,
) -> dict[str, dict[str, Any]]:
    """Per-class one-vs-rest ECE + reliability table for a per-position posterior matrix.

    ``probs`` is ``(n_positions, n_classes)``. For each class the positive-class posterior is
    that column and the label is ``1`` where the truth is that class — the reduction A11
    Pin 1 pins, because D11's estimator is defined *"on the positive-class posterior"* and a
    per-nt 8-class softmax has no single such column. The estimator itself is
    :func:`metrics.binned_ece` (15 equal-mass, debiased) — called, never re-implemented — and
    the reliability rows come from the same :func:`metrics.reliability_bins` it sums over, so
    the plotted curve and the reported number cannot disagree.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be (n_positions, n_classes), got shape {p.shape}")
    if len(y_true) != p.shape[0]:
        raise ValueError(f"y_true carries {len(y_true)} positions but probs carries {p.shape[0]}")
    if p.shape[1] != len(CLASS_ORDER):
        raise ValueError(f"probs carries {p.shape[1]} classes, expected {len(CLASS_ORDER)}")
    # The estimator sorts by posterior, so a NaN or an out-of-range value does not fail — it
    # sorts somewhere arbitrary and returns a number that looks like an ECE (§10.3). The check
    # lives HERE rather than inside `metrics.binned_ece` on purpose: that kernel is frozen and
    # golden-locked (tests/ml/test_eval_gate.py), and a per-call scan of its Sequence inputs
    # would run once per class per bootstrap replicate. One vectorised check per read is O(N)
    # in numpy and covers every value this module ever hands it.
    if not np.isfinite(p).all() or float(p.min()) < 0.0 or float(p.max()) > 1.0:
        raise ValueError(
            "probs must be finite and in [0, 1] — got "
            f"min={float(np.nanmin(p))!r} max={float(np.nanmax(p))!r} "
            f"n_nonfinite={int((~np.isfinite(p)).sum())}"
        )
    truth = np.asarray(y_true, dtype=np.int64)
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(CLASS_ORDER):
        y_bin = (truth == i).astype(np.int64).tolist()
        p_col = p[:, i].tolist()
        rows = M.reliability_bins(y_bin, p_col, n_bins)
        out[name] = {
            "n_positive": int(sum(y_bin)),
            "prevalence": (sum(y_bin) / len(y_bin)) if y_bin else float("nan"),
            "ece": M.binned_ece(y_bin, p_col, n_bins),
            "ece_plugin": M.binned_ece(y_bin, p_col, n_bins, debias=False),
            "reliability": rows,
        }
    return out


def _ece_over_pairs(pairs: list[tuple[int, float]]) -> float:
    """``binned_ece`` over ``(y, p)`` pairs — the statistic ``block_bootstrap_ci`` resamples."""
    if not pairs:
        return float("nan")
    y = [int(a) for a, _ in pairs]
    p = [float(b) for _, b in pairs]
    return M.binned_ece(y, p)


def ece_block_ci(
    blocks_by_cluster: Mapping[int, list[tuple[int, float]]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = CALIB_SPLIT_SEED,
) -> dict[str, Any]:
    """Cluster-blocked percentile CI for the one-vs-rest ECE (ADR-0005 D5 block level)."""
    blocks = [blocks_by_cluster[k] for k in sorted(blocks_by_cluster)]
    return M.block_bootstrap_ci(blocks, _ece_over_pairs, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- #
# The run: forward the checkpoint, fit on half A, read on half B
# --------------------------------------------------------------------------- #
def _forward_window_logits(
    model: Any,
    window_ids: np.ndarray,
    *,
    device: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Forward pre-encoded windows and return **raw per-window logits** (no reconciliation).

    ⚠ This deliberately stops one step short of ``infer.scan.scan_encoded_windows``, which is
    the promoted forward **+ reconcile** loop: the convex fit needs the logits *before* the
    D3+A3 average (A11 Pin 3/4), and that function returns only the reconciled output. To keep
    the duplication from becoming a divergence, :func:`run_calibration` cross-checks one
    record's ``reconcile_windows(self-forward)`` against ``scan_encoded_windows`` and records
    the max absolute difference in the report (``operator_agreement``) — a measured clause,
    not a promise. Tiling and encoding are the shared ``window_dataset`` helpers, unchanged.
    """
    import torch

    if window_ids.ndim != 2:
        raise ValueError(f"window_ids must be (n_windows, window), got shape {window_ids.shape}")
    if window_ids.shape[0] == 0:
        raise ValueError("window_ids carries no windows; there is nothing to forward")
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        chunks = []
        with torch.no_grad():
            for i in range(0, window_ids.shape[0], batch_size):
                batch = torch.from_numpy(window_ids[i : i + batch_size].astype(np.int64))
                chunks.append(model(input_ids=batch.to(device)).float().cpu())
        return torch.cat(chunks, dim=0).numpy()
    finally:
        if was_training:
            model.train()


def _portable_path(path: Path) -> str:
    """A repo-relative path when the file sits under the checkout, else the absolute one.

    The artifact is committed to a **public** repo, so recording ``/home/<user>/…`` would ship
    a developer-specific path that no reviewer can resolve and that names a local account for
    no provenance gain. Falls back to the absolute path when the input genuinely lives outside
    the checkout — an out-of-tree input is a fact worth recording, not one worth hiding.

    Both the running checkout **and** its main checkout are tried, because this step runs from
    a `.claude/worktrees/<concern>` worktree (CLAUDE.md §5.0) while the DVC-tracked inputs are
    materialised only in the main checkout: relativising against the worktree alone would fall
    back to absolute for every real input and the guard would never fire where it matters.
    """
    resolved = path.resolve()
    checkout = Path(__file__).resolve().parents[3]
    roots = [checkout]
    parts = checkout.parts
    if ".claude" in parts:
        idx = parts.index(".claude")
        if parts[idx : idx + 2] == (".claude", "worktrees"):
            roots.append(Path(*parts[:idx]))
    for root in roots:
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(resolved)


def _reconcile(window_logits: np.ndarray, starts: Sequence[int], seq_len: int) -> Any:
    """The frozen ADR-0005 D3 + A3 operator, on possibly-tempered logits."""
    from tbox_finder.infer.reconcile import reconcile_windows

    return reconcile_windows(window_logits, np.asarray(starts), seq_len)


def run_calibration(
    *,
    checkpoint: str | Path | None = None,
    context_parquet: str | Path | None = None,
    labels_parquet: str | Path | None = None,
    split_table: str | Path | None = None,
    window_nt: int = WINDOW_NT,
    stride_nt: int = STRIDE_NT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fraction: float = CALIB_FIT_FRACTION,
    seed: int = CALIB_SPLIT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    max_records: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Fit T on the ``selection_val`` fit half and read reliability on the read half.

    Heavy imports (torch, the loaders' pandas/pyarrow) are inside; the module imports in the
    bare-CI tier without them.
    """
    from tbox_finder.data.window_dataset import (
        context_labels,
        encode_eval_window,
        load_selection_val_records,
        selection_val_problems,
        tile_windows,
    )
    from tbox_finder.infer.scan import (
        DEFAULT_CHECKPOINT,
        load_stage1_checkpoint,
        scan_encoded_windows,
    )

    loader_kwargs: dict[str, Any] = {"window": window_nt}
    if context_parquet is not None:
        loader_kwargs["context_parquet"] = context_parquet
    if labels_parquet is not None:
        loader_kwargs["labels_parquet"] = labels_parquet
    if split_table is not None:
        loader_kwargs["split_table"] = split_table
    records, fold_scope = load_selection_val_records(**loader_kwargs)
    problems = selection_val_problems(fold_scope)
    if problems:
        # Same refusal the trainer makes: a rung that cannot prove itself disjoint from the
        # training fold would calibrate on memorised records and look exactly like a result.
        raise ValueError(
            "P2-06a selection-val fold failed its own invariants:\n  " + "\n  ".join(problems)
        )
    if max_records is not None:
        records = records[:max_records]

    cluster_sizes: dict[int, int] = {}
    for rec in records:
        cluster_sizes[int(rec.cluster_id)] = cluster_sizes.get(int(rec.cluster_id), 0) + 1
    fit_ids = calibration_fit_cluster_ids(cluster_sizes, fraction=fraction, seed=seed)
    fit_records = [r for r in records if int(r.cluster_id) in fit_ids]
    read_records = [r for r in records if int(r.cluster_id) not in fit_ids]
    if not fit_records or not read_records:
        raise ValueError(
            f"empty half: {len(fit_records)} fit / {len(read_records)} read records — "
            "the halving did not produce two scorable halves"
        )

    ckpt = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
    model = load_stage1_checkpoint(ckpt, device=device)
    dev = next(model.parameters()).device

    # ── forward each half once, keeping raw per-window logits ──
    def _forward_half(recs: list[Any]) -> list[dict[str, Any]]:
        out = []
        for rec in recs:
            seq_len = len(rec.context_seq)
            starts = tile_windows(seq_len, window=window_nt, stride=stride_nt)
            ids = np.stack([encode_eval_window(rec, s, window=window_nt) for s in starts])
            out.append(
                {
                    "record": rec,
                    "starts": starts,
                    "seq_len": seq_len,
                    "logits": _forward_window_logits(model, ids, device=dev, batch_size=batch_size),
                    "ids": ids,
                    "y_true": context_labels(rec),
                }
            )
        return out

    fit_fw = _forward_half(fit_records)
    read_fw = _forward_half(read_records)

    # ── the promoted-loop cross-check: our forward + the frozen operator vs the shared
    #    forward+reconcile implementation, on the first read record (bit-exact expected) ──
    probe = read_fw[0]
    shared = scan_encoded_windows(
        model, probe["ids"], probe["starts"], probe["seq_len"], device=dev, batch_size=batch_size
    )
    mine = _reconcile(probe["logits"], probe["starts"], probe["seq_len"])
    max_abs_diff = float(np.max(np.abs(np.asarray(shared.log_probs) - np.asarray(mine.log_probs))))
    operator_agreement = {
        "record_id": probe["record"].record_id,
        "n_windows": int(mine.n_windows),
        "max_abs_log_prob_diff": max_abs_diff,
        "bit_exact": max_abs_diff == 0.0,
        "compared_against": "infer.scan.scan_encoded_windows (the promoted forward+reconcile)",
    }

    # ── the fit: per-window positions of the fit half (A11 Pin 3) ──
    fit_logits = np.concatenate([f["logits"].reshape(-1, len(CLASS_ORDER)) for f in fit_fw])
    fit_labels = np.concatenate(
        [np.concatenate([f["y_true"][s : s + window_nt] for s in f["starts"]]) for f in fit_fw]
    )
    fit = fit_temperature(fit_logits, fit_labels)

    # ── the read: reconciled posteriors, uncalibrated vs tempered (deployment order) ──
    def _read(temperature: float) -> dict[str, Any]:
        y_all: list[int] = []
        pred_all: list[int] = []
        prob_rows: list[np.ndarray] = []
        cluster_of_position: list[int] = []
        n_coverage_ge_2 = 0
        for f in read_fw:
            logits = f["logits"] if temperature == 1.0 else f["logits"] / temperature
            rec_out = _reconcile(logits, f["starts"], f["seq_len"])
            prob_rows.append(np.exp(np.asarray(rec_out.log_probs)))
            y_all.extend(int(x) for x in f["y_true"])
            pred_all.extend(int(x) for x in np.asarray(rec_out.prediction))
            cluster_of_position.extend([int(f["record"].cluster_id)] * f["seq_len"])
            n_coverage_ge_2 += int(np.sum(np.asarray(rec_out.coverage) >= 2))
        probs = np.concatenate(prob_rows, axis=0)
        per_class = per_class_reliability(y_all, probs)
        for i, name in enumerate(CLASS_ORDER):
            blocks: dict[int, list[tuple[int, float]]] = {}
            col = probs[:, i].tolist()
            for pos, cid in enumerate(cluster_of_position):
                blocks.setdefault(cid, []).append((1 if y_all[pos] == i else 0, col[pos]))
            per_class[name]["ece_ci"] = ece_block_ci(blocks, n_boot=n_boot, seed=seed)
        return {
            "temperature": temperature,
            "per_class": per_class,
            "per_nt_f1_by_class": M.per_nt_f1_by_class(y_all, pred_all),
            "boundary_iou_by_element": M.boundary_iou_by_element(y_all, pred_all),
            "gate4_core_min_f1": M.gate4_core_min_f1(y_all, pred_all)["min_f1"],
            "core_worst_ece": max(per_class[e]["ece"] for e in CORE_ELEMENTS),
            "n_positions": len(y_all),
            "n_positions_coverage_ge_2": n_coverage_ge_2,
            "_prediction": pred_all,
        }

    pre = _read(1.0)
    post = _read(fit.temperature)
    changed = sum(1 for a, b in zip(pre["_prediction"], post["_prediction"], strict=True) if a != b)
    n_pos_read = int(pre["n_positions"])
    del pre["_prediction"], post["_prediction"]
    # Coverage is a property of the tiling, not of T, so the two conditions must agree — and
    # a disagreement would mean the read halves scored different geometry.
    if pre["n_positions_coverage_ge_2"] != post["n_positions_coverage_ge_2"]:
        raise ValueError(
            "coverage differs between the uncalibrated and tempered reads "
            f"({pre['n_positions_coverage_ge_2']} vs {post['n_positions_coverage_ge_2']}) — "
            "the two reads did not score the same tiling"
        )
    coverage_ge_2 = int(pre["n_positions_coverage_ge_2"])

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "env_lock": ENV_LOCK,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "is_science": False,
        "gated": False,
        "checkpoint": _portable_path(ckpt),
        "geometry": {"window_nt": int(window_nt), "stride_nt": int(stride_nt)},
        "stack_order": {
            "pinned": list(STACK_ORDER),
            "applied": ["train", "temperature_scale"],
            "prior_shift_applied": False,
            "named_posterior": True,
            "prior_shift_deferred_to": "P5 (deployment prior; Saerens/Elkan log-odds)",
        },
        "split": {
            "fold": "selection_val",
            "is_d11_disjoint_calibration_split": False,
            "d11_split_owner": "imp.md P3-02 (`calib` column, carved from the training folds)",
            "seed": int(seed),
            "fraction": float(fraction),
            "n_records": len(records),
            "n_records_fit": len(fit_records),
            "n_records_read": len(read_records),
            "n_clusters": len(cluster_sizes),
            "n_clusters_fit": len(fit_ids),
            "n_clusters_read": len(cluster_sizes) - len(fit_ids),
            "fit_cluster_ids": sorted(int(c) for c in fit_ids),
            "read_cluster_ids": sorted(c for c in cluster_sizes if c not in fit_ids),
            "cluster_sizes": {str(k): int(v) for k, v in sorted(cluster_sizes.items())},
            "full_fold": max_records is None,
            "max_records": max_records,
            "fold_scope_leakage": fold_scope.get("leakage"),
        },
        "temperature": {**asdict(fit), "fitted_on": "selection_val fit half (per-window logits)"},
        "operator_agreement": operator_agreement,
        "read": {
            "n_positions": n_pos_read,
            "n_records": len(read_records),
            "n_blocks": len(cluster_sizes) - len(fit_ids),
            "n_boot": int(n_boot),
            "n_positions_coverage_ge_2": coverage_ge_2,
            "n_positions_argmax_changed": int(changed),
            "argmax_change_fraction": (changed / n_pos_read) if n_pos_read else float("nan"),
            "uncalibrated": pre,
            "tempered": post,
        },
        "disclosures": list(DISCLOSURES),
    }


# --------------------------------------------------------------------------- #
# Report validation (total — returns problems, never raises)
# --------------------------------------------------------------------------- #
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of problems; empty ⇒ valid.

    Every clause is **re-derived from the recorded evidence** rather than read back from a
    flag the writer set: the halves are re-cut from the recorded ``cluster_sizes`` (which
    catches the fit/read **inversion** that equal-sized halves make invisible to a count),
    each reported ECE is re-summed from its own reliability rows, and the temperature's
    stationarity residual is checked against its tolerance.
    """
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return [f"report must be a mapping, got {type(report).__name__}"]
    for key in ("schema_version", "step", "generated_by", "adr", "env_lock"):
        if not isinstance(report.get(key), str) or not report.get(key):
            problems.append(f"{key}: missing or not a non-empty string")
    if report.get("is_science") is not False:
        problems.append("is_science: P2-13 is machinery + a non-gated read, not a claim (§10.3)")
    if report.get("gated") is not False:
        problems.append("gated: must be False — GATE-2's ECE is graded at P3 exit (D11)")
    disclosures = report.get("disclosures")
    if not isinstance(disclosures, list) or len(disclosures) < len(DISCLOSURES):
        problems.append(
            f"disclosures: expected the {len(DISCLOSURES)} shipped constants, got "
            f"{len(disclosures) if isinstance(disclosures, list) else 'none'}"
        )

    stack = report.get("stack_order")
    if not isinstance(stack, Mapping):
        problems.append("stack_order: missing — D11's order is what this step records")
    else:
        if list(stack.get("pinned") or []) != list(STACK_ORDER):
            problems.append(f"stack_order.pinned: must be {list(STACK_ORDER)} (D11)")
        if stack.get("prior_shift_applied") is not False:
            problems.append(
                "stack_order.prior_shift_applied: the P2 artifact must stop at the "
                "named posterior (pre-prior-shift)"
            )
        if list(stack.get("applied") or []) != ["train", "temperature_scale"]:
            problems.append("stack_order.applied: expected ['train', 'temperature_scale']")

    split = report.get("split")
    if not isinstance(split, Mapping):
        problems.append("split: missing")
    else:
        if split.get("is_d11_disjoint_calibration_split") is not False:
            problems.append(
                "split.is_d11_disjoint_calibration_split: selection_val is the fold the sweep "
                "selected on; claiming it is D11's disjoint split would be false"
            )
        sizes_raw = split.get("cluster_sizes")
        fit_ids = split.get("fit_cluster_ids")
        read_ids = split.get("read_cluster_ids")
        if not isinstance(sizes_raw, Mapping) or not sizes_raw:
            problems.append("split.cluster_sizes: missing — the halves cannot be re-derived")
        elif not isinstance(fit_ids, list) or not isinstance(read_ids, list):
            problems.append("split.fit_cluster_ids / read_cluster_ids: missing or not lists")
        else:
            sizes = {int(k): int(v) for k, v in sizes_raw.items()}
            try:
                expected = calibration_fit_cluster_ids(
                    sizes,
                    fraction=float(split.get("fraction", CALIB_FIT_FRACTION)),
                    seed=int(split.get("seed", CALIB_SPLIT_SEED)),
                )
            except ValueError as exc:
                problems.append(f"split: recorded cluster_sizes do not re-cut: {exc}")
            else:
                if set(int(c) for c in fit_ids) != set(int(c) for c in expected):
                    problems.append(
                        "split.fit_cluster_ids: does NOT re-derive from the recorded "
                        "cluster_sizes at the recorded seed/fraction — the fit and read halves "
                        "may be inverted (which would make the read in-sample for T)"
                    )
                if set(int(c) for c in read_ids) != set(sizes) - set(int(c) for c in expected):
                    problems.append(
                        "split.read_cluster_ids: is not the exact complement of the fit half"
                    )
            if set(int(c) for c in fit_ids) & set(int(c) for c in read_ids):
                problems.append("split: fit and read halves share a cluster — not disjoint")

    temp = report.get("temperature")
    if not isinstance(temp, Mapping):
        problems.append("temperature: missing")
    else:
        t = temp.get("temperature")
        if not _finite(t) or float(t) <= 0.0:
            problems.append(f"temperature.temperature: must be finite and > 0, got {t!r}")
        if temp.get("converged") is not True:
            problems.append("temperature.converged: an unconverged fit is not a fit")
        beta, grad, gtol = temp.get("beta"), temp.get("grad_at_solution"), temp.get("grad_tol")
        if _finite(t) and _finite(beta) and abs(float(beta) * float(t) - 1.0) > 1e-9:
            problems.append("temperature: beta and temperature are not reciprocals")
        if not (_finite(grad) and _finite(gtol)) or abs(float(grad)) > float(gtol):
            problems.append(
                f"temperature.grad_at_solution: |{grad!r}| exceeds grad_tol {gtol!r} — the "
                "stationarity condition the fit claims is not satisfied"
            )
        n0, n1 = temp.get("nll_per_position_initial"), temp.get("nll_per_position_final")
        if not (_finite(n0) and _finite(n1)) or float(n1) > float(n0):
            problems.append(
                f"temperature: the fit did not lower its own objective ({n0!r} -> {n1!r})"
            )

    agree = report.get("operator_agreement")
    if not isinstance(agree, Mapping):
        problems.append("operator_agreement: missing — the self-forward is uncross-checked")
    elif agree.get("bit_exact") is not True or agree.get("max_abs_log_prob_diff") != 0.0:
        problems.append(
            "operator_agreement: this module's forward + frozen operator did NOT reproduce "
            f"infer.scan.scan_encoded_windows bit-exactly (max_abs_log_prob_diff="
            f"{agree.get('max_abs_log_prob_diff')!r})"
        )

    read = report.get("read")
    if not isinstance(read, Mapping):
        return [*problems, "read: missing"]
    for key in ("n_positions", "n_records", "n_blocks", "n_boot", "n_positions_argmax_changed"):
        v = read.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            problems.append(f"read.{key}: missing or not a non-negative int")
    if isinstance(split, Mapping) and read.get("n_blocks") != split.get("n_clusters_read"):
        problems.append("read.n_blocks: does not equal split.n_clusters_read")
    for cond in ("uncalibrated", "tempered"):
        block = read.get(cond)
        if not isinstance(block, Mapping):
            problems.append(f"read.{cond}: missing")
            continue
        per_class = block.get("per_class")
        if not isinstance(per_class, Mapping) or set(per_class) != set(CLASS_ORDER):
            problems.append(f"read.{cond}.per_class: must carry exactly the 8 CLASS_ORDER keys")
            continue
        for name, entry in per_class.items():
            if not isinstance(entry, Mapping):
                problems.append(f"read.{cond}.per_class.{name}: not a mapping")
                continue
            ece = entry.get("ece")
            if not _finite(ece) or not 0.0 <= float(ece) <= 1.0:
                problems.append(f"read.{cond}.per_class.{name}.ece: not a probability-scale value")
            rows = entry.get("reliability")
            if not isinstance(rows, list) or not rows:
                problems.append(f"read.{cond}.per_class.{name}.reliability: missing bin table")
                continue
            # ⚠ Shape-check the rows BEFORE arithmetic on them. This function's contract is
            # "total — returns problems, never raises", and a validator that dies on a
            # malformed artifact reports nothing, which is strictly worse than one that
            # reports a problem (the P2-12 round-4 lesson, same failure one layer in).
            if not all(
                isinstance(r, Mapping)
                and _finite(r.get("weight"))
                and _finite(r.get("debiased_gap"))
                for r in rows
            ):
                problems.append(
                    f"read.{cond}.per_class.{name}.reliability: every row must carry a finite "
                    "'weight' and 'debiased_gap'"
                )
                continue
            weight = sum(float(r["weight"]) for r in rows)
            if abs(weight - 1.0) > 1e-9:
                problems.append(
                    f"read.{cond}.per_class.{name}.reliability: bin weights sum to {weight!r}"
                )
            resummed = sum(float(r["weight"]) * float(r["debiased_gap"]) for r in rows)
            if _finite(ece) and abs(resummed - float(ece)) > 1e-9:
                problems.append(
                    f"read.{cond}.per_class.{name}: reported ece {ece!r} does not re-sum from "
                    f"its own reliability rows ({resummed!r})"
                )
        core_eces = [
            per_class[e].get("ece") for e in CORE_ELEMENTS if isinstance(per_class.get(e), Mapping)
        ]
        if len(core_eces) != len(CORE_ELEMENTS) or not all(_finite(v) for v in core_eces):
            problems.append(
                f"read.{cond}.per_class: a core element's ece is missing or not finite, so "
                "core_worst_ece cannot be re-derived"
            )
        elif _finite(block.get("core_worst_ece")):
            worst = max(float(v) for v in core_eces)
            if abs(float(block["core_worst_ece"]) - worst) > 1e-12:
                problems.append(
                    f"read.{cond}.core_worst_ece: {block['core_worst_ece']!r} is not the max over "
                    f"the three core elements ({worst!r})"
                )
        else:
            problems.append(f"read.{cond}.core_worst_ece: missing or not finite")
    return problems


def read_is_out_of_sample(report: Mapping[str, Any]) -> bool:
    """Re-cut the halves from the report's own ``cluster_sizes`` and confirm the read half is
    the complement of the half T was fitted on.

    This is the **inversion** guard, and it is why the clause re-derives instead of reading a
    boolean: two ~equal halves make a swapped fit/read wiring invisible to every count in the
    artifact, while turning the read in-sample for T — the one property the P2 design exists
    to keep ([[symmetric-count-fixture-blind-to-inversion]]).
    """
    split = report.get("split") if isinstance(report, Mapping) else None
    if not isinstance(split, Mapping):
        return False
    sizes_raw, fit_ids, read_ids = (
        split.get("cluster_sizes"),
        split.get("fit_cluster_ids"),
        split.get("read_cluster_ids"),
    )
    if not isinstance(sizes_raw, Mapping) or not sizes_raw:
        return False
    if not isinstance(fit_ids, list) or not isinstance(read_ids, list):
        return False
    try:
        sizes = {int(k): int(v) for k, v in sizes_raw.items()}
        expected = calibration_fit_cluster_ids(
            sizes,
            fraction=float(split.get("fraction", CALIB_FIT_FRACTION)),
            seed=int(split.get("seed", CALIB_SPLIT_SEED)),
        )
    except (TypeError, ValueError):
        return False
    fit_set, read_set = {int(c) for c in fit_ids}, {int(c) for c in read_ids}
    return (
        fit_set == {int(c) for c in expected}
        and read_set == set(sizes) - fit_set
        and not (fit_set & read_set)
    )


def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """The step's gate, re-derived from the report (never asserted alongside it)."""
    problems = validate_report(report)
    read = report.get("read") if isinstance(report, Mapping) else None
    temp = report.get("temperature") if isinstance(report, Mapping) else None
    clauses = {
        "report_valid": not problems,
        "temperature_certified": bool(isinstance(temp, Mapping) and temp.get("converged") is True),
        "read_out_of_sample_for_T": read_is_out_of_sample(report),
        "operator_agrees_bit_exactly": bool(
            isinstance(report.get("operator_agreement"), Mapping)
            and report["operator_agreement"].get("bit_exact") is True
        ),
        "not_gated": report.get("gated") is False if isinstance(report, Mapping) else False,
        "prior_shift_refused": bool(
            isinstance(report.get("stack_order"), Mapping)
            and report["stack_order"].get("prior_shift_applied") is False
        ),
        "argmax_effect_measured": bool(
            isinstance(read, Mapping) and isinstance(read.get("n_positions_argmax_changed"), int)
        ),
    }
    clauses["overall_pass"] = all(clauses.values())
    return clauses


# --------------------------------------------------------------------------- #
# Figure data + rendering (numeric env writes JSON; viz env renders the PNG)
# --------------------------------------------------------------------------- #
def figure_data(report: Mapping[str, Any]) -> dict[str, Any]:
    """The compact payload the reliability diagram is drawn from.

    Split from rendering because ``matplotlib`` lives **only** in ``envs/viz.yml``, while this
    step's numeric half runs in ``ml-dna`` (torch, no matplotlib) — the repo's standing
    pattern (``splits.py``, ``power.py``, ``coverage.py``).
    """
    read = report["read"]
    classes = [*CORE_ELEMENTS, "background"]
    return {
        "step": STEP,
        "generated_by": GENERATED_BY,
        "temperature": report["temperature"]["temperature"],
        "n_positions": read["n_positions"],
        "n_blocks": read["n_blocks"],
        "gated": False,
        "classes": classes,
        "curves": {
            name: {
                cond: {
                    "conf": [r["conf"] for r in read[cond]["per_class"][name]["reliability"]],
                    "acc": [r["acc"] for r in read[cond]["per_class"][name]["reliability"]],
                    "weight": [r["weight"] for r in read[cond]["per_class"][name]["reliability"]],
                    "ece": read[cond]["per_class"][name]["ece"],
                }
                for cond in ("uncalibrated", "tempered")
            }
            for name in classes
        },
    }


def plot_figures(
    *,
    figure_data_path: str | Path = DEFAULT_FIGURE_DATA,
    figures_dir: str | Path = FIGURES_DIR,
) -> int:
    """Render the per-class reliability diagram from ``figure_data_path`` (viz env)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(figure_data_path).read_text())
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = data["classes"]
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 4.1), squeeze=False)
    for ax, name in zip(axes[0], classes, strict=True):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
        for cond, marker in (("uncalibrated", "o"), ("tempered", "s")):
            curve = data["curves"][name][cond]
            ax.plot(
                curve["conf"],
                curve["acc"],
                marker=marker,
                markersize=4,
                linewidth=1.2,
                label=f"{cond} (ECE {curve['ece']:.4f})",
            )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("mean posterior (bin)")
        ax.set_ylabel("observed positive rate")
        # symlog both ways: for a rare class the interesting bins sit at 1e-6..1e-2 and a
        # linear axis collapses them onto the origin, while plain log cannot show an
        # exactly-zero bin (which is the common case for the sparse classes).
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Stage-1 per-class reliability, one-vs-rest, 15 equal-mass debiased bins — "
        f"T={data['temperature']:.4f}, selection_val read half "
        f"({data['n_positions']:,} nt / {data['n_blocks']} clusters) — NON-GATED (P2-13)",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / "reliability_selection_val.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tbox_finder.calib.temperature",
        description=(
            "Fit the Stage-1 shared scalar temperature on the selection_val fit half and read "
            "the NON-GATED per-class reliability on the read half (P2-13; ADR-0005 D11 + A11)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fit T + read reliability (needs torch + the checkpoint)")
    run.add_argument("--checkpoint", default=None, help="Stage-1 checkpoint (.pt)")
    run.add_argument("--context", default=None, help="context_v0.parquet")
    run.add_argument("--labels", default=None, help="labels_v0.parquet")
    run.add_argument("--split-table", default=None, help="split_assignments.parquet")
    run.add_argument("--out", default=str(DEFAULT_REPORT), help="report JSON path")
    run.add_argument("--figure-data", default=str(DEFAULT_FIGURE_DATA))
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--fraction", type=float, default=CALIB_FIT_FRACTION)
    run.add_argument("--seed", type=int, default=CALIB_SPLIT_SEED)
    run.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    run.add_argument("--max-records", type=int, default=None, help="smoke only; marks full_fold")
    run.add_argument("--device", default=None)

    plot = sub.add_parser("plot-figures", help="render the reliability diagram (viz env)")
    plot.add_argument("--figure-data", default=str(DEFAULT_FIGURE_DATA))
    plot.add_argument("--figures-dir", default=str(FIGURES_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plot-figures":
        return plot_figures(figure_data_path=args.figure_data, figures_dir=args.figures_dir)

    report = run_calibration(
        checkpoint=args.checkpoint,
        context_parquet=args.context,
        labels_parquet=args.labels,
        split_table=args.split_table,
        batch_size=args.batch_size,
        fraction=args.fraction,
        seed=args.seed,
        n_boot=args.n_boot,
        max_records=args.max_records,
        device=args.device,
    )
    problems = validate_report(report)
    clauses = derive_clauses(report)
    report["gate"] = clauses
    out = Path(args.out)
    if problems:
        # An invalid report is diverted, never written to the step's path: a malformed
        # artifact at the real name is indistinguishable from a good one downstream.
        out = out.with_suffix(".invalid.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fd = Path(args.figure_data)
    fd.parent.mkdir(parents=True, exist_ok=True)
    fd.write_text(json.dumps(figure_data(report), indent=2, sort_keys=True) + "\n")

    temp = report["temperature"]["temperature"]
    read = report["read"]
    print(f"wrote {out}")
    print(
        f"T = {temp:.6f}  (fitted on {report['split']['n_records_fit']} records / "
        f"{report['split']['n_clusters_fit']} clusters)"
    )
    for cond in ("uncalibrated", "tempered"):
        worst = read[cond]["core_worst_ece"]
        print(
            f"  {cond:>13}: core-worst ECE {worst:.6f}  "
            f"gate4_core_min_f1 {read[cond]['gate4_core_min_f1']:.6f}"
        )
    print(
        f"  arg-max changed at {read['n_positions_argmax_changed']:,} / "
        f"{read['n_positions']:,} positions "
        f"({read['argmax_change_fraction'] * 100:.4f} %); "
        f"coverage>=2 at {read['n_positions_coverage_ge_2']:,}"
    )
    if problems:
        print("REPORT INVALID:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"gate: overall_pass={clauses['overall_pass']}")
    return 0 if clauses["overall_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
