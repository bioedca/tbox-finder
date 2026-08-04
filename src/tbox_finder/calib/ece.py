"""P3-09 — the two calibration-error estimators (PRD §12; ADR-0005 D11, D13, A2).

ADR-0005 asks for **two distinct** estimators, and the distinction is the point:

``binned_ece`` — **in-distribution, GATE-2-gated (D11).**
    15 equal-mass debiased bins on the positive-class posterior, graded at P3 exit
    against ``ECE ≤ 0.05``. Shipped at P0-31 and living in :mod:`tbox_finder.metrics`;
    **re-exported here**, not re-implemented, so imp.md's ``{binned_ece, ood_ece}`` pair
    is importable from one place without a second copy of the bin arithmetic
    ([[promote-dont-duplicate-is-a-correctness-rule]]).

``ood_ece`` — **out-of-distribution / leave-clade-out, reported and never gated (D13).**
    Built here. A leave-clade-out ECE ≤ 0.05 is likely infeasible under clade shift, so
    D13 reports it and adjudicates it by a decision rule, subject to the
    ``OOD_ECE_MIN_N = 20`` admissibility floor pinned in ADR-0005 Amendment A2.

**Why a second estimator at all, rather than the binned one on fewer rows.** The OOD unit
is a single held-out order — A2 pins its floor at *20 positives* — and at that N the D11
estimator is not merely noisy, it is *biased in a direction that flatters the model*:
``ECE_bin`` is systematically biased, the bias grows as the evaluation set shrinks and as
the bin count rises, and it is **worst for well-calibrated models** [arXiv:2012.08668]
(accessed 2026-08-04); histogram binning also needs ``O(B/ε²)`` samples where scaling-based
estimation needs ``O(1/ε²)`` [arXiv:1909.10155] (accessed 2026-08-04). Fifteen equal-mass
bins over 20 records is 1–2 records per bin: every ``acc_b`` is 0 or 1 and the estimate is
mostly binning noise. A2 says as much — the D13 estimator is "chosen precisely so N = 20 is
admissible (bootstrap CI reported, **not per-bin filled**)".

**The estimator shipped here.** A leave-one-out **Beta-kernel** (Nadaraya–Watson) estimate
of ``ĝ(p) = E[y | p]``, from which both of D13's admissible forms are read off the *same*
``ĝ`` — so this implementation satisfies the "proper-scoring-rule decomposition **or**
smoothed/kernel calibration estimator" disjunction without choosing one of them on the
ADR's behalf:

- the **smoothed/kernel calibration error** ``L1 = mean_i |ĝ_{-i}(p_i) − p_i|`` — the
  reported ``ood_ece``, in the same units as the binned ECE so the two are comparable;
- the **proper-scoring-rule (Brier/Bregman) decomposition** ``Brier = calibration +
  refinement`` with ``calibration = mean (p_i − ĝ_{-i})²`` and ``refinement =
  mean ĝ_{-i}(1 − ĝ_{-i})``, reported beside it with its residual.

Kernel density estimates of the calibration error are consistent and low-bias and are
explicitly intended to "be applied to small subsets of data" [arXiv:2210.07810] (accessed
2026-08-04); the corresponding proper-calibration-error estimators are consistent and
asymptotically unbiased [arXiv:2312.08589] (accessed 2026-08-04). The kernel is a **Beta**
kernel because posteriors pile up at 0 and 1 and beta-kernel estimators are *free of
boundary bias*, non-negative, and attain MISE ``O(n^{-4/5})`` [DOI:10.1016/S0167-9473(99)00010-9]
(accessed 2026-08-04) — a Gaussian kernel would smear mass off the ends of the interval
exactly where a near-saturated Stage-2 head puts most of its rows.

**Measured bias, and why the reported statistic is the plug-in.** Nothing is unbiased at
N = 20, so the direction of each candidate's bias was measured rather than assumed. The
sweep below is 40 seeded replicates per cell (``random.Random(2026)``, uniform posteriors)
against a known truth; ``tests/unit/test_ece.py::test_estimator_bias_direction_simulation``
runs the same simulation at 8 replicates and pins the directional facts that survive there:

=====================  ==========  ==========  ==========
estimator              n=20        n=50        n=200
=====================  ==========  ==========  ==========
**perfectly calibrated — truth 0.000**
-----------------------------------------------------------
kernel plug-in         0.132       0.098       0.051
kernel + debias        0.053       0.039       0.017
binned + debias (D11)  0.068       0.048       0.024
binned plug-in         0.280       0.175       0.085
**squashed logits, T = 2.5 — truth 0.123**
-----------------------------------------------------------
kernel plug-in         0.153       0.106       0.100
kernel + debias        0.077       0.051       0.063
binned + debias (D11)  0.065       0.062       0.060
binned plug-in         0.337       0.220       0.146
=====================  ==========  ==========  ==========

Two things fall out, and they decide the design.

*A binned estimator cannot be used at the D13 unit.* Fifteen equal-mass bins over 20 rows
reads **0.280 on perfectly calibrated data** — 5.6× the GATE-2 threshold, entirely from
binning noise. That is the concrete form of [arXiv:2012.08668]'s result and the reason A2
specifies an estimator that needs no per-bin fills.

*Debiasing buys a lower floor by destroying the magnitude.* Both debiased columns separate
the two regimes, but on genuinely miscalibrated data they saturate at ~0.06 **regardless of
N**, against a truth of 0.123 — they converge to about half the real drift. The per-point
half-normal correction (the exact analogue of the one ``metrics.binned_ece`` applies per
bin) does it too: 0.063 at n=200. D13 compares the reported OOD ECE against a *pinned drift
bound*, so the **magnitude** is the quantity, not merely the ordering — and only the kernel
plug-in approaches the truth as N grows (0.153 → 0.106 → 0.100). So ``ood_ece`` reports the
plug-in, and the debiasing correction is deliberately **not** applied.

The price is a larger small-sample floor (0.132 at n = 20 where the truth is 0), i.e. a bias
that is *upward* and decays with N. Upward is the conservative direction here: D13 grants a
corpus a "calibrated-negative PASS" only when its OOD ECE *meets* a drift bound, so
over-stating drift makes a PASS harder to earn, never easier — the anti-overclaiming posture
A2 argues for its own floor. It is also why the CI matters more than the point estimate near
N = 20, and why this number is reported and never gated.

*(The same table says something about the gated D11 estimator that this step neither
resolves nor changes: its debiased reading saturates at ~0.060 against a truth of 0.123. Its
docstring already flags "final certification of the debiasing term is a P3-exit / ADR
concern" — this is measured evidence for that pending decision. No D11 behaviour moves here,
and GATE-2 is graded at P3-10 on the estimator as it stands.)*

**Not ADR-pinned, and flagged as such.** D13 pins the estimator *family*, the bootstrap
CIs and the min-N floor. It does not pin a bandwidth, a bandwidth-selection rule, the
``L1``-vs-``L2`` exponent, or the plug-in-vs-debiased choice argued above, and A2 does not
either. Those are implementer choices, recorded in the payload (``bandwidth``,
``bandwidth_grid``, ``bandwidth_criterion``, ``estimator``) so a reviewer reads them off
the artifact rather than the source — the same standing as ``metrics.binned_ece``'s
debiasing term, and certifying any of them to a specific formula is a P3-exit / ADR
concern (CLAUDE.md §7), not an assertion this step makes.

**Cost.** The kernel estimate is ``O(n²)`` per bandwidth (``O(9n²)`` with selection, and
``n_boot`` further evaluations for the CI), against ``O(n log n)`` for the binned one. That
is affordable because the D13 unit is *one held-out clade* — A2's floor is 20 positives and
the leave-one-order-out units run to a few hundred records. It is **not** the estimator to
point at a full test split: at n≈3,000 one selected estimate is ~90 s and a 200-replicate CI
is ~30 min. The in-distribution split is D11's binned estimator's job anyway.

Pure stdlib (``math`` only), like :mod:`tbox_finder.metrics`, so the whole calibration
tier runs in the bare CI env rather than ``importorskip``-ing itself green.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any

from tbox_finder.coverage import OOD_ECE_MIN_N
from tbox_finder.eval.resample import block_bootstrap, blocks_by_key
from tbox_finder.metrics import ECE_N_BINS, binned_ece, reliability_bins

__all__ = [
    "ADR",
    "BANDWIDTH_CRITERION",
    "BANDWIDTH_GRID",
    "DEFAULT_OOD_N_BOOT",
    "ECE_N_BINS",
    "ESTIMATOR",
    "GENERATED_BY",
    "IN_DISTRIBUTION_ESTIMATOR",
    "OOD_ECE_MIN_N",
    "STEP",
    "binned_ece",
    "brier_decomposition",
    "kernel_conditional_mean",
    "ood_ece",
    "reliability_bins",
    "select_bandwidth",
]

STEP = "P3-09"
GENERATED_BY = "src/tbox_finder/calib/ece.py"
ADR = (
    "ADR-0005 D11 (in-distribution binned estimator, GATE-2) + D13 & Amendment A2 "
    "(distinct small-N-robust OOD estimator, bootstrap CIs, OOD_ECE_MIN_N=20); PRD §12"
)

#: Name of the OOD estimator, recorded in every payload. ``beta_kernel`` = Beta-kernel
#: Nadaraya–Watson conditional mean; ``loo`` = leave-one-source-record-out; ``l1`` = the
#: ``L_1`` calibration error read off it.
ESTIMATOR = "beta_kernel_loo_l1"

#: The D11 estimator this one must stay **distinct** from (D13). Recorded in the payload
#: so the distinctness requirement is auditable from the artifact, not just the ADR.
IN_DISTRIBUTION_ESTIMATOR = "tbox_finder.metrics.binned_ece"

#: Candidate bandwidths for the Beta kernel, a 1–2–5 decade ladder spanning three decades.
#: ``h`` sets the kernel's concentration (``≈1/h``): at ``h = 1`` the kernel is nearly
#: uniform and ``ĝ`` collapses to the global positive rate; at ``h = 0.002`` it is sharply
#: local. Exact decimal literals rather than a computed geometric series, so the grid — and
#: therefore any selected bandwidth — is bit-identical across platforms.
BANDWIDTH_GRID: tuple[float, ...] = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

#: How a bandwidth is chosen when the caller does not pin one: maximise the leave-one-out
#: **Bernoulli log-likelihood** of the observed labels under ``ĝ_{-i}``. That criterion is
#: itself a strictly proper scoring rule, it targets exactly the conditional mean this
#: estimator reports, and it is deterministic — no resampling, no seed. Ties resolve to the
#: **smallest** bandwidth on the grid (a strict improvement is required to move), which is
#: the degenerate single-class case where every bandwidth scores identically.
BANDWIDTH_CRITERION = "loo_bernoulli_log_likelihood"

#: Bootstrap replicates for the OOD CI. Deliberately below
#: ``eval.resample.DEFAULT_N_BOOT`` (2000): the kernel statistic is ``O(n²)`` per replicate
#: where the binned one is ``O(n log n)``, and the OOD unit is ~20–200 records. Same
#: precedent and same reason as ``calib.temperature.DEFAULT_N_BOOT``.
DEFAULT_OOD_N_BOOT = 200

#: Probability clamp. Keeps ``log(p)``/``log(1−p)`` finite for the posteriors of exactly
#: 0.0 and 1.0 that a saturated sigmoid produces, without moving any interior value.
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Input validation — in the caller's dtype, before any coercion
# --------------------------------------------------------------------------- #
def _check_inputs(y_true: Sequence[Any], p_pos: Sequence[Any]) -> tuple[list[int], list[float]]:
    """Validate and coerce ``(y, p)``, refusing in the **original** dtype.

    The order matters. ``int(y)`` truncates toward zero, so validating after coercion
    accepts ``y = 0.6`` as a negative and reports a bit-identical, fully "successful" fit
    on a target the caller wrote as a positive — the P3-07 review-r1 defect, in the same
    shape. Labels must therefore be exactly 0 or 1 *before* coercion, and posteriors must
    be finite and inside ``[0, 1]`` (the Beta kernel's support; an out-of-range posterior
    is a domain error here, not a rounding artifact).
    """
    if len(y_true) != len(p_pos):
        raise ValueError(f"y_true and p_pos must be the same length: {len(y_true)} != {len(p_pos)}")
    y_out: list[int] = []
    p_out: list[float] = []
    for i, (y, p) in enumerate(zip(y_true, p_pos, strict=True)):
        if isinstance(y, bool) or y in (0, 1):
            y_out.append(int(y))
        else:
            raise ValueError(
                f"y_true[{i}]={y!r} is not a binary label — expected exactly 0 or 1 "
                "(checked before coercion: int() would truncate 0.6 to a negative)"
            )
        p_f = float(p)
        if not math.isfinite(p_f) or p_f < 0.0 or p_f > 1.0:
            raise ValueError(
                f"p_pos[{i}]={p!r} is not a finite posterior in [0, 1] — the Beta kernel "
                "is supported on [0, 1]"
            )
        p_out.append(p_f)
    return y_out, p_out


def _clamp01(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


# --------------------------------------------------------------------------- #
# The kernel conditional mean
# --------------------------------------------------------------------------- #
def kernel_conditional_mean(
    y_true: Sequence[Any],
    p_pos: Sequence[Any],
    bandwidth: float,
    *,
    uids: Sequence[Hashable] | None = None,
) -> list[float]:
    """Leave-one-out Beta-kernel estimate of ``ĝ(p_i) = E[y | p = p_i]``, one value per row.

    Weight of sample ``j`` at query ``p_i`` is the Beta density
    ``Beta(p_j; a=p_i/h + 1, b=(1−p_i)/h + 1)`` — Chen's beta-kernel smoother
    [DOI:10.1016/S0167-9473(99)00010-9], the 1-simplex case of the Dirichlet kernel used by
    the ``L_p`` calibration-error estimator [arXiv:2210.07810]. Only the ``p_j``-dependent
    part is needed: the normaliser ``B(a, b)`` is constant in ``j`` and cancels in the
    Nadaraya–Watson ratio, so the log-weight is
    ``(p_i/h)·log p_j + ((1−p_i)/h)·log(1−p_j)`` and no ``lgamma`` is evaluated. Weights are
    exponentiated after subtracting their maximum, so the denominator is ``≥ 1`` and cannot
    underflow to zero even when the query sits at a boundary and every neighbour is at the
    other one.

    **Leave-one-out is by source record, not by position** (``uids``, defaulting to the row
    index). Including ``j = i`` would let each point vote for its own conditional mean —
    an ``O(1/n)`` bias straight toward "perfectly calibrated" that is largest exactly where
    D13 operates, at small ``n``. Excluding by *uid* rather than by index additionally makes
    the estimate invariant to a bootstrap replicate drawing the same block twice: a
    duplicated row would otherwise be its own nearest neighbour carrying its own label at
    maximal weight, so the resampled distribution would sit systematically below the point
    estimate and the CI would be narrow in the flattering direction.

    Rows whose neighbour set is empty (every other entry is a copy of the same source
    record) get ``NaN`` — the replicate is then undefined and
    :func:`tbox_finder.eval.resample.block_bootstrap` drops it, reporting a reduced
    ``n_boot``, rather than quietly averaging over whatever survived.
    """
    if bandwidth <= 0.0 or not math.isfinite(bandwidth):
        raise ValueError(f"bandwidth={bandwidth!r} must be finite and > 0")
    y, p = _check_inputs(y_true, p_pos)
    n = len(y)
    if n < 2:
        raise ValueError(
            f"kernel_conditional_mean needs at least 2 rows for a leave-one-out estimate, got {n}"
        )
    ids: Sequence[Hashable] = range(n) if uids is None else uids
    if len(ids) != n:
        raise ValueError(f"uids must have one entry per row: {len(ids)} != {n}")

    log_p = [math.log(_clamp01(v)) for v in p]
    log_1mp = [math.log(1.0 - _clamp01(v)) for v in p]
    inv_h = 1.0 / bandwidth

    out: list[float] = []
    for i in range(n):
        a = p[i] * inv_h
        b = (1.0 - p[i]) * inv_h
        logw: list[float] = []
        keep: list[int] = []
        for j in range(n):
            if ids[j] == ids[i]:
                continue
            keep.append(j)
            logw.append(a * log_p[j] + b * log_1mp[j])
        if not keep:
            out.append(float("nan"))
            continue
        top = max(logw)
        num = 0.0
        den = 0.0
        for j, lw in zip(keep, logw, strict=True):
            w = math.exp(lw - top)
            den += w
            if y[j] == 1:
                num += w
        out.append(num / den)
    return out


def select_bandwidth(
    y_true: Sequence[Any],
    p_pos: Sequence[Any],
    *,
    grid: Sequence[float] = BANDWIDTH_GRID,
    uids: Sequence[Hashable] | None = None,
) -> tuple[float, float]:
    """Choose a bandwidth by maximising the leave-one-out Bernoulli log-likelihood
    (:data:`BANDWIDTH_CRITERION`). Returns ``(bandwidth, log_likelihood)``.

    Deterministic: a full scan of ``grid`` in the given order, keeping the first strict
    improvement, so ties go to the earliest (smallest) bandwidth. A grid entry whose
    ``ĝ`` is undefined for any row scores ``-inf`` and can never win.
    """
    if not grid:
        raise ValueError("grid must be non-empty")
    y, _ = _check_inputs(y_true, p_pos)
    best_h = float(grid[0])
    best_ll = -math.inf
    for h in grid:
        g = kernel_conditional_mean(y_true, p_pos, float(h), uids=uids)
        ll = 0.0
        for y_i, g_i in zip(y, g, strict=True):
            if math.isnan(g_i):
                ll = -math.inf
                break
            g_c = _clamp01(g_i)
            ll += math.log(g_c) if y_i == 1 else math.log(1.0 - g_c)
        if ll > best_ll:
            best_ll = ll
            best_h = float(h)
    return best_h, best_ll


# --------------------------------------------------------------------------- #
# The two reads off one ĝ
# --------------------------------------------------------------------------- #
def brier_decomposition(
    y_true: Sequence[Any],
    p_pos: Sequence[Any],
    g_hat: Sequence[float],
) -> dict[str, float]:
    """The proper-scoring-rule (Bregman) decomposition of the Brier score — D13's other
    admissible form, read off the same ``ĝ`` as the kernel calibration error.

    With the **true** ``g(p) = E[y|p]`` the identity is exact:
    ``E[(p−y)²] = E[(p−g)²] + E[g(1−g)]``, because the cross term
    ``E[(p−g)(g−y)] = E[(p−g)·E[(g−y)|p]]`` vanishes. With an *estimated* ``ĝ`` it holds only
    up to estimation error, so ``residual = brier − (calibration + refinement)`` is reported
    rather than assumed away: it is a diagnostic on the kernel fit, and a large one means the
    reported ``calibration`` term is not trustworthy. Never asserted to be zero on real data.
    """
    y, p = _check_inputs(y_true, p_pos)
    if len(g_hat) != len(y):
        raise ValueError(f"g_hat must have one entry per row: {len(g_hat)} != {len(y)}")
    n = len(y)
    if n == 0:
        nan = float("nan")
        return {"brier": nan, "calibration": nan, "refinement": nan, "residual": nan}
    brier = sum((p_i - y_i) ** 2 for y_i, p_i in zip(y, p, strict=True)) / n
    calibration = sum((p_i - g_i) ** 2 for p_i, g_i in zip(p, g_hat, strict=True)) / n
    refinement = sum(g_i * (1.0 - g_i) for g_i in g_hat) / n
    return {
        "brier": brier,
        "calibration": calibration,
        "refinement": refinement,
        "residual": brier - (calibration + refinement),
    }


def _l1_calibration_error(
    y_true: Sequence[Any],
    p_pos: Sequence[Any],
    bandwidth: float,
    *,
    uids: Sequence[Hashable] | None = None,
) -> float:
    """``mean_i |ĝ_{-i}(p_i) − p_i|`` — the reported OOD statistic. NaN if any ``ĝ`` is."""
    _, p = _check_inputs(y_true, p_pos)
    g = kernel_conditional_mean(y_true, p_pos, bandwidth, uids=uids)
    total = 0.0
    for p_i, g_i in zip(p, g, strict=True):
        if math.isnan(g_i):
            return float("nan")
        total += abs(g_i - p_i)
    return total / len(p) if p else float("nan")


# --------------------------------------------------------------------------- #
# The public OOD estimator
# --------------------------------------------------------------------------- #
def ood_ece(
    y_true: Sequence[Any],
    p_pos: Sequence[Any],
    block_labels: Sequence[Any],
    *,
    block_key: str = "cluster_id",
    bandwidth: float | None = None,
    n_boot: int = DEFAULT_OOD_N_BOOT,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """The ADR-0005 D13 small-N-robust OOD / leave-clade-out calibration error, with a
    block-resampled bootstrap CI and the Amendment-A2 min-N admissibility flag.

    ``block_labels`` is **required** and is checked against the block-granularity column
    allowlist by :func:`tbox_finder.eval.resample.blocks_by_key`. There is deliberately no
    "no blocks" path: a record-level bootstrap over homologs returns an interval far
    narrower than the data support, and PRD §12 requires every CI here to be resampled at
    the cluster / held-out-order level. Within one held-out order the exchangeability unit
    is the homology cluster, which is why ``block_key`` defaults to ``cluster_id``.

    **Admissibility.** ``OOD_ECE_MIN_N = 20`` **real positives** in the calibration unit is
    an ADR-0005 A2 admissibility gate, imported from
    :data:`tbox_finder.coverage.OOD_ECE_MIN_N` rather than re-typed, and with no keyword to
    override it — A2 froze it in code specifically so no committed report can contradict the
    pin. Below the floor the estimate is **inadmissible**: ``ood_ece`` is ``None`` and no CI
    is computed, so a consumer reading ``ood_ece`` cannot grade a number the ADR says does
    not support a verdict, and the corpus routes to D13's "sensitivity-bounded /
    inconclusive negative *by rule*". The value is still recorded, under
    ``inadmissible_point``, because withholding a measurement and hiding it are different
    things — it just cannot be mistaken for an admissible one. Clearing the floor admits an
    estimate; it does not make it a PASS (conditions (i) and (iii) of D13 are separate
    P3/P4 quantities).

    Returns the payload described in this module's docstring: the estimate and its
    admissibility, the census (``n_records`` / ``n_positives`` / ``n_blocks``), the CI, the
    Brier decomposition, and the estimator settings that are implementer choices rather
    than ADR pins.
    """
    y, p = _check_inputs(y_true, p_pos)
    if len(block_labels) != len(y):
        raise ValueError(
            f"block_labels must have one entry per row: {len(block_labels)} != {len(y)}"
        )
    n = len(y)
    if n < 2:
        raise ValueError(f"ood_ece needs at least 2 rows for a leave-one-out estimate, got {n}")

    uids = list(range(n))
    rows = [(y[i], p[i], uids[i]) for i in range(n)]
    blocks = blocks_by_key(rows, block_labels, key_name=block_key)

    if bandwidth is None:
        h, h_ll = select_bandwidth(y, p, uids=uids)
        selected = True
    else:
        h = float(bandwidth)
        if h <= 0.0 or not math.isfinite(h):
            raise ValueError(f"bandwidth={bandwidth!r} must be finite and > 0")
        _, h_ll = select_bandwidth(y, p, grid=(h,), uids=uids)
        selected = False

    g_hat = kernel_conditional_mean(y, p, h, uids=uids)
    point = _l1_calibration_error(y, p, h, uids=uids)
    n_positives = sum(y)
    admissible = n_positives >= OOD_ECE_MIN_N

    def _statistic(pairs: list[tuple[int, float, Hashable]]) -> float:
        if not pairs:
            return float("nan")
        return _l1_calibration_error(
            [r[0] for r in pairs],
            [r[1] for r in pairs],
            h,  # fixed at the full-sample choice: replicates measure the error's sampling
            uids=[r[2] for r in pairs],  # variability, not the selector's
        )

    ci = (
        block_bootstrap(blocks, _statistic, n_boot=n_boot, ci_level=ci_level, seed=seed)
        if admissible
        else None
    )

    return {
        "estimator": ESTIMATOR,
        "distinct_from": IN_DISTRIBUTION_ESTIMATOR,
        "ood_ece": point if admissible else None,
        "inadmissible_point": None if admissible else point,
        "admissible": admissible,
        "inadmissible_reason": (
            None
            if admissible
            else (
                f"n_positives={n_positives} < OOD_ECE_MIN_N={OOD_ECE_MIN_N} — the D13 "
                "estimator is inadmissible below the ADR-0005 A2 floor; the corpus is "
                "sensitivity-bounded / inconclusive negative by rule"
            )
        ),
        "min_n": OOD_ECE_MIN_N,
        "n_records": n,
        "n_positives": n_positives,
        "n_blocks": len(blocks),
        "block_key": block_key,
        "ci": ci,
        "bandwidth": h,
        "bandwidth_selected": selected,
        "bandwidth_grid": list(BANDWIDTH_GRID),
        "bandwidth_criterion": BANDWIDTH_CRITERION,
        "bandwidth_loo_log_likelihood": h_ll,
        "brier_decomposition": brier_decomposition(y, p, g_hat),
        "gated": False,  # D13: OOD ECE is reported, never gated (GATE-2 is D11's, in-dist)
        "adr": ADR,
    }
