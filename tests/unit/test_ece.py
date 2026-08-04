"""P3-09 — the two calibration-error estimators (:mod:`tbox_finder.calib.ece`).

The module re-exports the D11 binned estimator and *builds* the D13 small-N-robust OOD one,
so the evidence here is split accordingly:

- the OOD estimator's core is pinned by **closed-form** expectations — constructions whose
  answer is derivable on paper and, in the primary case, independent of the bandwidth — not
  by a golden number this code produced (a self-generated expectation certifies nothing,
  §10.3);
- every refusal is paired with the corresponding clean input succeeding
  ([[raises-test-needs-a-positive-control]]);
- the ADR-pinned min-N floor is checked by *flipping the constant*, so a hand-typed copy of
  the pin could not pass ([[pinned-constant-that-nothing-reads]]);
- D13's "distinct from the D11 estimator" requirement is made executable by a seeded
  simulation against a **known truth**, rather than asserted in prose.

Pure stdlib — runs in the bare CI env.
"""

from __future__ import annotations

import math
import random

import pytest

from tbox_finder import coverage
from tbox_finder import metrics as M
from tbox_finder.calib import ece


# ========================================================================== #
# Closed-form: constant posterior => uniform kernel weights => exact answer
# ========================================================================== #
def test_constant_posterior_gives_the_hand_derived_value() -> None:
    """With every ``p_i`` equal, all kernel weights are equal, so
    ``ĝ_{-i} = (S − y_i)/(n − 1)`` and

        mean_i |ĝ_{-i} − p| = (1/n)·Σ_i (S − y_i)/(n−1) − p = S/n − p

    because ``Σ_i (S − y_i) = nS − S`` and every ``ĝ`` here exceeds ``p``. With n=100,
    S=80, p=0.5 the answer is **exactly 0.30** — and, because the weights cancel, it is the
    same for every bandwidth. That independence is what makes this a specification of the
    estimator rather than a snapshot of one configuration.
    """
    n, k, p_val = 100, 80, 0.5
    y = [1] * k + [0] * (n - k)
    p = [p_val] * n
    for h in ece.BANDWIDTH_GRID:
        assert ece._l1_calibration_error(y, p, h) == pytest.approx(0.30, abs=1e-12)

    out = ece.ood_ece(y, p, [f"c{i % 10}" for i in range(n)], n_boot=40)
    assert out["ood_ece"] == pytest.approx(0.30, abs=1e-12)
    assert out["n_records"] == 100 and out["n_positives"] == 80 and out["n_blocks"] == 10


def test_a_perfectly_matched_constant_posterior_reads_the_loo_floor() -> None:
    """Same construction with ``p = S/n`` — a *perfectly calibrated* sample — where the
    answer is the estimator's irreducible leave-one-out floor, also in closed form.

    Now ``ĝ_{-i}`` straddles ``p``: a positive row's neighbours are one positive short, a
    negative row's are one short of a negative, so the absolute values no longer cancel and

        mean_i |ĝ_{-i} − p| = 2·k·(n−k) / (n²·(n−1))

    which is **0.003232…** at n=100, k=80 — small, positive, and ``O(1/n)``. This is the
    small-sample floor the module docstring's table quantifies, in its exactly-derivable
    form: the estimator does not read 0 on perfectly calibrated data and must not be
    expected to. Bandwidth-independent for the same reason as the test above.
    """
    n, k = 100, 80
    y = [1] * k + [0] * (n - k)
    p = [k / n] * n
    floor = 2 * k * (n - k) / (n * n * (n - 1))
    assert floor == pytest.approx(0.0032323232323232323, abs=1e-15)
    for h in ece.BANDWIDTH_GRID:
        assert ece._l1_calibration_error(y, p, h) == pytest.approx(floor, abs=1e-12)


def test_two_separated_groups_decompose_into_two_closed_forms() -> None:
    """Two posterior levels far apart, with a narrow kernel, are independent problems: the
    cross-weights are ``exp(-O(1/h))`` and vanish. Group A (n=40, 30 ones, p=0.1) contributes
    ``|30/40 − 0.1| = 0.65`` per row up to the LOO correction, group B (n=60, 6 ones, p=0.9)
    contributes ``|6/60 − 0.9| = 0.80``; the exact LOO value is ``S/n − p`` per group as
    above, so the pooled answer is ``(40·0.65 + 60·0.80)/100 = 0.74``."""
    y = [1] * 30 + [0] * 10 + [1] * 6 + [0] * 54
    p = [0.1] * 40 + [0.9] * 60
    assert ece._l1_calibration_error(y, p, 0.002) == pytest.approx(0.74, abs=1e-6)


# ========================================================================== #
# Leave-one-out is by SOURCE RECORD — the property the bootstrap depends on
# ========================================================================== #
def test_duplicating_every_row_leaves_the_estimate_bit_identical() -> None:
    """Under uid-based exclusion, doubling the dataset doubles every kernel weight
    uniformly, so ``ĝ`` — a weighted *mean* — is unchanged **exactly**.

    This is the property a bootstrap replicate relies on: a block drawn twice must not let
    each of its rows become its own nearest neighbour carrying its own label. Position-based
    exclusion fails this test, and fails it in the flattering direction.
    """
    rng = random.Random(5)
    n = 60
    p = [rng.random() for _ in range(n)]
    y = [1 if rng.random() < v else 0 for v in p]
    uids = list(range(n))

    g_once = ece.kernel_conditional_mean(y, p, 0.05)
    g_twice = ece.kernel_conditional_mean(y + y, p + p, 0.05, uids=uids + uids)
    assert g_twice[:n] == pytest.approx(g_once, abs=1e-12)
    assert g_twice[n:] == pytest.approx(g_once, abs=1e-12)
    # The statistic itself is invariant to machine precision.
    once = ece._l1_calibration_error(y, p, 0.05)
    twice = ece._l1_calibration_error(y + y, p + p, 0.05, uids=uids + uids)
    assert twice == pytest.approx(once, abs=1e-12)


def test_position_based_exclusion_breaks_the_duplication_invariance() -> None:
    """The matched control for the test above ([[matched-control-before-certifying]]).

    Excluding by *position* instead of uid leaves each row's twin in its own neighbour set,
    at the maximum kernel weight, carrying its own label. Then the duplicated dataset no
    longer reproduces the original estimate — which is exactly the property a bootstrap
    replicate needs, since a replicate draws blocks with replacement. Measured direction on
    this fixture: **downward** (0.11910 vs 0.11986), i.e. toward "better calibrated", so the
    resampled distribution would sit below the point estimate and the interval would be
    narrow in the flattering direction. Direction is reported; the assertion is on the
    invariance being broken, which is true regardless of sign.
    """
    rng = random.Random(6)
    n = 60
    p = [rng.random() for _ in range(n)]
    y = [1 if rng.random() < v else 0 for v in p]

    original = ece._l1_calibration_error(y, p, 0.05)
    by_uid = ece._l1_calibration_error(y + y, p + p, 0.05, uids=list(range(n)) * 2)
    by_position = ece._l1_calibration_error(y + y, p + p, 0.05, uids=list(range(2 * n)))

    assert by_uid == pytest.approx(original, abs=1e-12)
    assert by_position != pytest.approx(
        original, abs=1e-6
    ), "uid exclusion is a no-op on this fixture — the control proves nothing"
    assert by_position < original  # measured direction, and the one that matters


def test_a_row_with_no_admissible_neighbour_is_nan_not_a_guess() -> None:
    g = ece.kernel_conditional_mean([1, 0], [0.4, 0.6], 0.05, uids=["same", "same"])
    assert all(math.isnan(v) for v in g)
    assert math.isnan(ece._l1_calibration_error([1, 0], [0.4, 0.6], 0.05, uids=["s", "s"]))
    # Positive control: distinct uids on the identical data give finite values.
    assert all(not math.isnan(v) for v in ece.kernel_conditional_mean([1, 0], [0.4, 0.6], 0.05))


# ========================================================================== #
# The ADR-0005 A2 min-N admissibility floor
# ========================================================================== #
def _rows(n_pos: int, n_neg: int) -> tuple[list[int], list[float], list[str]]:
    y = [1] * n_pos + [0] * n_neg
    p = [0.7] * n_pos + [0.3] * n_neg
    blocks = [f"c{i % 4}" for i in range(n_pos + n_neg)]
    return y, p, blocks


def test_below_the_floor_the_estimate_is_withheld_not_reported() -> None:
    y, p, b = _rows(coverage.OOD_ECE_MIN_N - 1, 30)
    out = ece.ood_ece(y, p, b, n_boot=20)
    assert out["admissible"] is False
    assert out["ood_ece"] is None, "an inadmissible estimate must not occupy the graded key"
    assert out["inadmissible_point"] is not None, "…but it must still be recorded"
    assert out["ci"] is None
    assert "sensitivity-bounded" in out["inadmissible_reason"]
    assert out["n_positives"] == coverage.OOD_ECE_MIN_N - 1


def test_at_the_floor_the_estimate_is_admitted() -> None:
    y, p, b = _rows(coverage.OOD_ECE_MIN_N, 30)
    out = ece.ood_ece(y, p, b, n_boot=20)
    assert out["admissible"] is True
    assert out["ood_ece"] is not None and out["inadmissible_point"] is None
    assert out["inadmissible_reason"] is None
    assert out["ci"]["n_blocks"] == 4 and out["ci"]["n_boot"] == 20
    assert out["ci"]["lower"] <= out["ci"]["point"] <= out["ci"]["upper"]


def test_the_floor_is_the_frozen_constant_not_a_local_copy(monkeypatch) -> None:
    """Flip the pin and the boundary must move with it. A hand-typed 20 in this module
    would keep both tests above green while silently ignoring an ADR amendment."""
    assert ece.OOD_ECE_MIN_N is coverage.OOD_ECE_MIN_N == 20
    monkeypatch.setattr(ece, "OOD_ECE_MIN_N", 40)
    y, p, b = _rows(30, 30)  # comfortably over the real pin, under the flipped one
    assert ece.ood_ece(y, p, b, n_boot=10)["admissible"] is False
    assert "OOD_ECE_MIN_N=40" in ece.ood_ece(y, p, b, n_boot=10)["inadmissible_reason"]


def test_admissibility_counts_positives_not_rows() -> None:
    """A2 pins the floor at 20 *real positives* in the calibration unit. A 500-row unit
    carrying 19 positives is still inadmissible — counting rows would silently admit exactly
    the sparse-positive corpora D13 exists to refuse."""
    y, p, b = _rows(19, 481)
    out = ece.ood_ece(y, p, b, n_boot=5)
    assert out["n_records"] == 500 and out["n_positives"] == 19
    assert out["admissible"] is False


def test_there_is_no_keyword_to_override_the_floor() -> None:
    import inspect

    params = set(inspect.signature(ece.ood_ece).parameters)
    assert not params & {"min_n", "min_positives", "floor", "ood_ece_min_n"}, (
        "A2 froze the floor in code with no CLI/config override so that no committed report "
        "can contradict the pin — a keyword here would be that override"
    )


# ========================================================================== #
# Input validation — in the caller's dtype, before coercion
# ========================================================================== #
@pytest.mark.parametrize("bad_y", [0.6, 0.4, 2, -1, float("nan"), "1"])
def test_non_binary_labels_are_refused_before_truncation(bad_y: object) -> None:
    with pytest.raises(ValueError, match="not a binary label"):
        ece.ood_ece([1, bad_y, 0], [0.5, 0.5, 0.5], ["a", "b", "c"])
    # Positive control: the identical call with a real label succeeds.
    assert ece.ood_ece([1, 1, 0], [0.5, 0.5, 0.5], ["a", "b", "c"], n_boot=5)["n_records"] == 3


@pytest.mark.parametrize("ok_y", [True, False, 0, 1, 0.0, 1.0])
def test_well_formed_label_spellings_are_accepted(ok_y: object) -> None:
    out = ece.ood_ece([ok_y, 1, 0], [0.5, 0.5, 0.5], ["a", "b", "c"], n_boot=5)
    assert out["n_records"] == 3


@pytest.mark.parametrize("bad_p", [-0.01, 1.01, float("nan"), float("inf")])
def test_posteriors_outside_the_kernel_support_are_refused(bad_p: float) -> None:
    with pytest.raises(ValueError, match=r"finite posterior in \[0, 1\]"):
        ece.ood_ece([1, 0, 1], [0.5, bad_p, 0.5], ["a", "b", "c"])


def test_saturated_posteriors_at_the_boundary_do_not_blow_up() -> None:
    """Exactly 0.0 and 1.0 are what a saturated sigmoid produces, and ``log(0)`` is where a
    naive kernel dies. The clamp must keep every value finite without moving interiors."""
    y = [1] * 12 + [0] * 12
    p = [1.0] * 12 + [0.0] * 12
    g = ece.kernel_conditional_mean(y, p, 0.05)
    assert all(math.isfinite(v) for v in g)
    # Two pure, maximally separated groups: each row's neighbours are its own group, so
    # ĝ is 1.0 / 0.0 and matches p. The estimator reads ~0 rather than raising or NaN-ing.
    assert g[:12] == pytest.approx([1.0] * 12, abs=1e-12)
    assert g[12:] == pytest.approx([0.0] * 12, abs=1e-12)
    assert ece._l1_calibration_error(y, p, 0.05) == pytest.approx(0.0, abs=1e-9)
    # The clamp is confined to the boundary: it is the identity on every interior value,
    # including ones far smaller than any real posterior.
    for interior in (1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9):
        assert ece._clamp01(interior) == interior
    assert 0.0 < ece._clamp01(0.0) < 1e-9 and 1.0 - 1e-9 < ece._clamp01(1.0) < 1.0


def test_length_and_size_guards() -> None:
    """Each guard is matched on the message that names **its own** function.

    A bare ``match="at least 2 rows"`` here was green with the ``ood_ece`` guard deleted:
    the call fell through to ``kernel_conditional_mean``, whose guard raises a message
    containing the same words. ``pytest.raises`` asserts *that* something raised, and a
    deeper guard satisfies it just as well ([[raises-test-needs-a-positive-control]]) —
    so the two are now told apart by the function each message names.
    """
    with pytest.raises(ValueError, match="same length"):
        ece.ood_ece([1, 0], [0.5], ["a", "b"])
    with pytest.raises(ValueError, match="block_labels must have one entry per row"):
        ece.ood_ece([1, 0], [0.5, 0.5], ["a"])
    with pytest.raises(ValueError, match=r"^ood_ece needs at least 2 rows"):
        ece.ood_ece([1], [0.5], ["a"])
    with pytest.raises(ValueError, match=r"^kernel_conditional_mean needs at least 2 rows"):
        ece.kernel_conditional_mean([1], [0.5], 0.05)
    for bad_h in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="must be finite and > 0"):
            ece.kernel_conditional_mean([1, 0], [0.5, 0.5], bad_h)


def test_block_labels_are_required_and_validated() -> None:
    import inspect

    sig = inspect.signature(ece.ood_ece)
    assert sig.parameters["block_labels"].default is inspect.Parameter.empty, (
        "there must be no 'no blocks' path — PRD §12 requires every CI here to be "
        "block-resampled"
    )
    # The allowlist refusal in eval.resample is reachable through this entry point.
    with pytest.raises(ValueError, match="identifies a record, not a block"):
        ece.ood_ece([1, 0], [0.6, 0.4], ["r1", "r2"], block_key="record_id")


# ========================================================================== #
# Bandwidth selection
# ========================================================================== #
def test_selection_is_deterministic_and_ties_go_to_the_smallest() -> None:
    # Constant posteriors => every bandwidth scores identically => the first (smallest) wins.
    y, p = [1] * 8 + [0] * 4, [0.5] * 12
    h, ll = ece.select_bandwidth(y, p)
    assert h == ece.BANDWIDTH_GRID[0] == min(ece.BANDWIDTH_GRID)
    assert math.isfinite(ll)
    assert ece.select_bandwidth(y, p) == (h, ll)  # no RNG anywhere in the criterion


def test_selection_prefers_a_narrow_kernel_when_the_signal_is_local() -> None:
    """Two well-separated, internally pure groups: a narrow kernel predicts every label
    perfectly, a near-uniform one predicts the global rate. The criterion must pick narrow."""
    y = [1] * 25 + [0] * 25
    p = [0.9] * 25 + [0.1] * 25
    h, _ = ece.select_bandwidth(y, p)
    assert h <= 0.05
    # And the criterion actually discriminates: the widest grid point scores strictly worse.
    _, ll_narrow = ece.select_bandwidth(y, p, grid=(h,))
    _, ll_wide = ece.select_bandwidth(y, p, grid=(1.0,))
    assert ll_narrow > ll_wide


def test_a_pinned_bandwidth_is_recorded_as_not_selected() -> None:
    y, p, b = _rows(25, 25)
    out = ece.ood_ece(y, p, b, bandwidth=0.02, n_boot=5)
    assert out["bandwidth"] == 0.02 and out["bandwidth_selected"] is False
    auto = ece.ood_ece(y, p, b, n_boot=5)
    assert auto["bandwidth_selected"] is True
    assert auto["bandwidth"] in ece.BANDWIDTH_GRID
    with pytest.raises(ValueError, match="grid must be non-empty"):
        ece.select_bandwidth([1, 0, 1], [0.6, 0.4, 0.6], grid=())


# ========================================================================== #
# The proper-scoring-rule (Brier) decomposition — D13's other admissible form
# ========================================================================== #
def test_brier_total_is_exact_and_the_decomposition_closes_when_ghat_is_right() -> None:
    """``brier`` itself needs no estimator: with p ≡ 0.5 it is exactly 0.25. The
    decomposition identity ``Brier = calibration + refinement`` is exact for the *true*
    conditional mean, and the constant-posterior construction gives ``ĝ`` essentially
    exactly, so the residual must vanish here — while remaining a *reported* diagnostic on
    real data, where it does not."""
    n, k = 100, 80
    y = [1] * k + [0] * (n - k)
    p = [0.5] * n
    g = ece.kernel_conditional_mean(y, p, 0.05)
    d = ece.brier_decomposition(y, p, g)
    assert d["brier"] == pytest.approx(0.25, abs=1e-12)
    assert d["residual"] == pytest.approx(0.0, abs=1e-9)
    assert d["calibration"] + d["refinement"] == pytest.approx(d["brier"], abs=1e-9)
    assert 0.0 < d["calibration"] < d["brier"] and 0.0 < d["refinement"] < d["brier"]


def test_brier_decomposition_rejects_a_mismatched_ghat() -> None:
    with pytest.raises(ValueError, match="one entry per row"):
        ece.brier_decomposition([1, 0], [0.5, 0.5], [0.5])


# ========================================================================== #
# D13's "distinct estimator" requirement, made executable
# ========================================================================== #
#: Replicates per simulation cell. 8 keeps the tier fast; the module docstring's table is
#: the same simulation at 40 (raise this to reproduce it).
_SIM_REPS = 8
_SIM_SEED = 2026
#: E|g(p) − p| when a calibrated posterior has its logits divided by T = 2.5 and the true
#: posterior is uniform on (0, 1). Computed by deterministic midpoint quadrature over
#: 2,000,000 points: 0.122603.
_SQUASH_TRUTH = 0.122603


def _draw_calibrated(n: int, rng: random.Random) -> tuple[list[int], list[float]]:
    p = [rng.random() for _ in range(n)]
    return [1 if rng.random() < v else 0 for v in p], p


def _draw_squashed(n: int, rng: random.Random, t: float = 2.5) -> tuple[list[int], list[float]]:
    """Labels drawn from the *true* posterior, reported at ``σ(logit(p)/T)`` — a known
    miscalibration of magnitude :data:`_SQUASH_TRUTH`."""
    truth = [rng.random() for _ in range(n)]
    y = [1 if rng.random() < v else 0 for v in truth]
    return y, [1.0 / (1.0 + math.exp(-math.log(v / (1.0 - v)) / t)) for v in truth]


def _sweep(draw, n: int) -> tuple[float, float, float]:
    rng = random.Random(_SIM_SEED)
    kernel, binned, binned_plugin = [], [], []
    for _ in range(_SIM_REPS):
        y, p = draw(n, rng)
        h, _ = ece.select_bandwidth(y, p)
        kernel.append(ece._l1_calibration_error(y, p, h))
        binned.append(M.binned_ece(y, p))
        binned_plugin.append(M.binned_ece(y, p, debias=False))
    m = len(kernel)
    return sum(kernel) / m, sum(binned) / m, sum(binned_plugin) / m


def test_estimator_bias_direction_simulation() -> None:
    """The measured facts that justify D13 asking for a *distinct* OOD estimator, and that
    justify this one reporting the plug-in. Seeded, so these are pins, not observations.

    Deliberately **not** asserted: that the kernel estimator is uniformly closer to the
    truth. At 8 replicates that ordering inverts at n=50 (it holds at 40 — see the module
    docstring), and a test should pin what its own sample supports.
    """
    for n in (20, 50, 200):
        k_cal, b_cal, bp_cal = _sweep(_draw_calibrated, n)
        k_mis, b_mis, bp_mis = _sweep(_draw_squashed, n)

        # (1) Nothing is unbiased at these N: on perfectly calibrated data every estimator
        #     reports a strictly positive floor. An OOD number near the floor is noise.
        assert k_cal > 0.0 and b_cal > 0.0 and bp_cal > 0.0

        # (2) A *plug-in* binned estimator is unusable at the D13 unit: at n=20 it reads
        #     several times the 0.05 GATE-2 threshold on perfectly calibrated data.
        if n == 20:
            assert bp_cal > 0.15, f"plug-in binned reads {bp_cal:.3f} on calibrated data"

        # (3) The debiased binned estimator under-states a *known* miscalibration at every
        #     N — it saturates at roughly half the truth instead of converging to it.
        assert b_mis < 0.6 * _SQUASH_TRUTH < _SQUASH_TRUTH

        # (4) The kernel plug-in reads higher than the debiased binned estimator in both
        #     regimes — the conservative direction for a drift-bound comparison.
        assert k_mis > b_mis and k_cal > b_cal

    # (5) The kernel floor decays with N (0.13 -> 0.05 at 40 reps), so the small-sample bias
    #     is a finite-sample effect and not a constant offset.
    assert _sweep(_draw_calibrated, 20)[0] > 1.5 * _sweep(_draw_calibrated, 200)[0]


def test_the_two_estimators_are_not_the_same_function() -> None:
    """D13 requires the OOD estimator to be **distinct** from D11's. Pinned structurally as
    well as numerically: the OOD entry point must not be the binned one under a new name."""
    assert ece.ood_ece is not M.binned_ece
    assert ece.binned_ece is M.binned_ece, "the D11 estimator is re-exported, not re-written"
    assert ece.ESTIMATOR != ece.IN_DISTRIBUTION_ESTIMATOR
    y, p, b = _rows(30, 30)
    out = ece.ood_ece(y, p, b, n_boot=10)
    assert out["distinct_from"] == "tbox_finder.metrics.binned_ece"
    assert out["ood_ece"] != pytest.approx(M.binned_ece(y, p), abs=1e-6)


def test_the_reported_point_and_the_ci_point_are_the_same_number() -> None:
    """One number, not two that agree to a few ulps. ``ood_ece`` computes ``ĝ`` in the
    resampler's own row order (the concatenation of the sorted blocks), so the reported
    estimate and ``ci["point"]`` are **bit-identical** rather than two float summations of
    the same multiset — which is what a reviewer asking "which of these is authoritative?"
    would otherwise be looking at."""
    y, p, b = _varied()
    out = ece.ood_ece(y, p, b, n_boot=10)
    assert out["ood_ece"] == out["ci"]["point"]
    # And the estimate itself does not depend on the caller's row order.
    order = sorted(range(len(y)), key=lambda i: (b[i], -p[i]))
    shuffled = ece.ood_ece(
        [y[i] for i in order], [p[i] for i in order], [b[i] for i in order], n_boot=10
    )
    assert shuffled["ood_ece"] == pytest.approx(out["ood_ece"], abs=1e-12)


def test_the_payload_records_the_choices_that_are_not_adr_pinned() -> None:
    y, p, b = _rows(25, 25)
    out = ece.ood_ece(y, p, b, n_boot=10)
    for key in (
        "estimator",
        "bandwidth",
        "bandwidth_grid",
        "bandwidth_criterion",
        "bandwidth_loo_log_likelihood",
        "min_n",
        "n_records",
        "n_positives",
        "n_blocks",
        "block_key",
        "brier_decomposition",
        "adr",
    ):
        assert key in out, key
    assert out["gated"] is False, "D13: the OOD ECE is reported, never gated"
    assert out["min_n"] == coverage.OOD_ECE_MIN_N
    assert out["bandwidth_grid"] == list(ece.BANDWIDTH_GRID)
    assert "D13" in out["adr"] and "D11" in out["adr"]


def _varied(seed: int = 3, n: int = 60) -> tuple[list[int], list[float], list[str]]:
    """A unit with *varied* posteriors. ``_rows`` deliberately has two pure levels, which
    makes its estimate invariant to block composition — a degenerate CI that would let a
    broken resampler look seeded. Anything asserting an interval needs this fixture instead
    ([[symmetric-count-fixture-blind-to-inversion]])."""
    rng = random.Random(seed)
    p = [round(rng.random(), 4) for _ in range(n)]
    y = [1 if rng.random() < v else 0 for v in p]
    return y, p, [f"c{i % 6}" for i in range(n)]


def test_the_ci_is_seeded_and_block_resampled() -> None:
    y, p, b = _varied()
    assert sum(y) >= coverage.OOD_ECE_MIN_N, "fixture must clear the floor to have a CI"
    a = ece.ood_ece(y, p, b, n_boot=25, seed=11)
    assert a["ci"] == ece.ood_ece(y, p, b, n_boot=25, seed=11)["ci"]  # CLAUDE.md §8.3
    d = ece.ood_ece(y, p, b, n_boot=25, seed=12)
    assert (d["ci"]["lower"], d["ci"]["upper"]) != (a["ci"]["lower"], a["ci"]["upper"])
    assert a["ci"]["n_blocks"] == 6 == len(set(b))
    assert a["ci"]["lower"] < a["ci"]["point"] < a["ci"]["upper"], "a degenerate interval"
    # One block => not resamplable, per ADR-0005 A1; the point survives, the interval does not.
    single = ece.ood_ece(y, p, ["only"] * len(y), n_boot=25)
    assert single["ci"]["n_blocks"] == 1 and math.isnan(single["ci"]["lower"])
    assert single["ci"]["point"] == pytest.approx(a["ci"]["point"])
