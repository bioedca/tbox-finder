"""P2-13 — the Stage-1 temperature-scaling machinery (ADR-0005 D11 + Amendment A11).

**Torch-free by construction**, which is the point: CI's test env installs numpy/pandas/
scikit-learn and **no torch**, so a gate that needed torch would `importorskip` GREEN and
certify nothing. Everything here runs on numpy alone — the fit is an exact convex 1-D solve
(A11 Pin 3) precisely so it can be tested where the suite actually runs.

What each tier locks:

* **the fit** — recovers a *known* temperature from synthetically mis-calibrated logits
  (the only honest check of a calibration fitter: manufacture the miscalibration, then see
  whether the fitter undoes it), is bit-deterministic, lowers its own objective, and
  **raises** in both degenerate directions rather than returning a bracket endpoint;
* **the halving** — whole-cluster, seeded, complementary, and asserted by **identity** at an
  **asymmetric** fraction: two ~equal halves make a swapped fit/read wiring invisible to any
  count, so a count-only test would pass under the one failure that matters
  ([[symmetric-count-fixture-blind-to-inversion]]);
* **the estimator seam** — ``metrics.reliability_bins`` and ``metrics.binned_ece`` are one
  implementation, and the reliability rows re-sum to the reported ECE;
* **the validator** — every clause is shown to **bite** on a mutated report, including the
  inversion, a forged bit-exactness claim, and an applied prior-shift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbox_finder import metrics as M
from tbox_finder.calib import temperature as T
from tbox_finder.labels import CLASS_ORDER, CORE_ELEMENTS


# --------------------------------------------------------------------------- #
# Fixtures: logits with a KNOWN mis-calibration
# --------------------------------------------------------------------------- #
def _synthetic_logits(
    n: int = 4000, n_classes: int = 8, *, sharpen: float = 2.5, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, float]:
    """Draw labels from a well-calibrated generator, then over-sharpen the logits.

    ``base`` logits define ``p = softmax(base)``; labels are sampled **from that
    distribution**, so ``base`` is calibrated by construction. The model under test is
    handed ``base * sharpen`` — over-confident by exactly ``sharpen`` — so the temperature
    that restores calibration is ``T = sharpen`` and the fit has a ground truth to hit.

    ⚠ Each row gets its **own** random logit vector: keying a generator on the class index
    alone would emit an identical row every time and the suite would compare uniform to
    uniform ([[degenerate-fixture-generators]]).
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.6, size=(n, n_classes))
    p = np.exp(base - base.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(n_classes, p=row) for row in p], dtype=np.int64)
    return base * sharpen, labels, sharpen


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #
def test_fit_recovers_a_known_temperature() -> None:
    logits, labels, true_t = _synthetic_logits()
    fit = T.fit_temperature(logits, labels)
    assert fit.converged is True
    # ~4k rows, so the MLE has sampling error; 12 % is comfortably inside it and far tighter
    # than the failure mode (a fitter that returns 1.0, i.e. "no correction at all").
    assert fit.temperature == pytest.approx(true_t, rel=0.12)
    assert abs(fit.temperature - 1.0) > 0.5, "a fitter that returns ~1 has not fitted anything"
    assert fit.beta == pytest.approx(1.0 / fit.temperature, rel=1e-12)


def test_fit_lowers_its_own_objective_and_reports_stationarity() -> None:
    logits, labels, _ = _synthetic_logits()
    fit = T.fit_temperature(logits, labels)
    assert fit.nll_per_position_final < fit.nll_per_position_initial
    assert abs(fit.grad_at_solution) <= fit.grad_tol
    # the reported per-position NLLs are the objective, re-derivable from the same kernel
    total_final, grad = T._nll_and_grad(np.asarray(logits), np.asarray(labels), fit.beta)
    assert total_final / fit.n_positions == pytest.approx(fit.nll_per_position_final, rel=1e-12)
    assert grad == pytest.approx(fit.grad_at_solution, rel=1e-9, abs=1e-9)


def test_fit_restores_calibration_it_did_not_have() -> None:
    """The behavioural claim: post-T ECE on the *positive-class* axis beats pre-T ECE."""
    logits, labels, _ = _synthetic_logits(n=6000, seed=11)
    fit = T.fit_temperature(logits, labels)
    scaled = T.apply_temperature(logits, fit.temperature)

    def _ece(z: np.ndarray) -> float:
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        # class 0 one-vs-rest, the axis A11 Pin 1 reduces to
        y = (np.asarray(labels) == 0).astype(int).tolist()
        return M.binned_ece(y, p[:, 0].tolist())

    before, after = _ece(np.asarray(logits, dtype=float)), _ece(scaled)
    assert after < before, f"temperature scaling did not improve ECE ({before} -> {after})"


def test_fit_is_deterministic_bitwise() -> None:
    logits, labels, _ = _synthetic_logits(n=1500, seed=3)
    a = T.fit_temperature(logits, labels)
    b = T.fit_temperature(logits, labels)
    assert a.beta == b.beta and a.temperature == b.temperature
    assert a.n_iterations == b.n_iterations
    # and invariant to row order: the objective is a sum over positions
    order = np.random.default_rng(0).permutation(len(labels))
    c = T.fit_temperature(np.asarray(logits)[order], np.asarray(labels)[order])
    assert c.temperature == pytest.approx(a.temperature, rel=1e-6)


def test_fit_refuses_the_perfect_accuracy_limit() -> None:
    """Per-position perfect accuracy drives T -> 0; the fitter must say so, not return a point.

    ⚠ Regression guard for a real defect this test found: at large ``beta`` the sampled
    gradient **underflows to exactly 0.0**, so the ``|grad| <= grad_tol`` stopping rule read
    the flat tail as stationarity and certified an arbitrary interior temperature. The fix is
    the exact ``beta -> inf`` limit ``Σ(max_c z − z_label)``, which is 0 here and cannot
    underflow. Sabotaging that check must turn this test RED.
    """
    n, k = 200, 8
    logits = np.full((n, k), -20.0)
    labels = np.arange(n) % k
    logits[np.arange(n), labels] = 20.0  # every row's own class wins by a mile
    # ⚠ matched on the EXACT-limit message, not on "either degeneracy message": the two
    # guards overlap on this input (the sampled gradient underflows to 0.0, which the
    # ``g_hi <= 0`` branch also refuses), and a regex accepting both would stay green with
    # the exact guard deleted — a test that certifies a disjunct it never exercises
    # ([[sabotage-attribution-must-name-the-test]] — sabotage compound disjuncts separately).
    with pytest.raises(T.TemperatureFitError, match="already its own arg-max"):
        T.fit_temperature(logits, labels)


def test_fit_refuses_a_minimiser_below_the_bracket() -> None:
    """Near-perfect (not perfect) accuracy: the root exists but sits outside T >= 1e-3."""
    n, k = 400, 8
    rng = np.random.default_rng(2)
    labels = rng.integers(0, k, size=n)
    # Vanishing margins: the optimum sits near beta ~ ln(k)/margin ~ 1e5, far above the
    # bracket's 1e3, so the fit must refuse rather than return 1e3 as if it were a solution.
    logits = rng.normal(0.0, 1e-6, size=(n, k))
    logits[np.arange(n), labels] += 1e-4
    logits[0] = -logits[0]  # one wrong position, so the exact beta->inf limit is > 0
    # the sampled-gradient branch, matched on its own message (the exact-limit guard cannot
    # fire here: one position is misclassified, so the beta->inf limit is strictly positive)
    with pytest.raises(T.TemperatureFitError, match="still decreasing at beta"):
        T.fit_temperature(logits, labels)


def test_fit_refuses_the_worse_than_uniform_limit() -> None:
    """A fold the model is anti-correlated on drives T -> inf; also an error, not a bound."""
    n, k = 300, 8
    logits = np.zeros((n, k))
    labels = np.arange(n) % k
    # put a large logit on a class that is never the label, and a large negative on the label
    logits[np.arange(n), labels] = -30.0
    logits[np.arange(n), (labels + 1) % k] = 30.0
    with pytest.raises(T.TemperatureFitError, match="worse than uniform|already increasing"):
        T.fit_temperature(logits, labels)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bracket": (2.0, 1.0)}, "bracket must satisfy"),
        ({"bracket": (0.0, 1.0)}, "bracket must satisfy"),
        ({"tol": 0.0}, "tol must be positive"),
        ({"max_iter": 0}, "max_iter must be positive"),
    ],
)
def test_fit_rejects_malformed_solver_settings(kwargs: dict, match: str) -> None:
    logits, labels, _ = _synthetic_logits(n=200)
    with pytest.raises(T.TemperatureFitError, match=match):
        T.fit_temperature(logits, labels, **kwargs)


@pytest.mark.parametrize(
    ("logits", "labels", "match"),
    [
        (np.zeros((4, 8)), np.zeros(3, dtype=int), "positions but labels"),
        (np.zeros((0, 8)), np.zeros(0, dtype=int), "empty calibration set"),
        (np.zeros((4, 1)), np.zeros(4, dtype=int), ">= 2 classes"),
        (np.zeros(4), np.zeros(4, dtype=int), "logits must be"),
        (np.zeros((4, 8)), np.zeros((4, 2), dtype=int), "labels must be"),
        (np.full((4, 8), np.nan), np.zeros(4, dtype=int), "non-finite"),
        (np.zeros((4, 8)), np.full(4, 9), "labels out of range"),
        (np.zeros((4, 8)), np.full(4, -1), "labels out of range"),
    ],
)
def test_fit_rejects_malformed_inputs(logits, labels, match: str) -> None:
    with pytest.raises(T.TemperatureFitError, match=match):
        T.fit_temperature(logits, labels)


def test_fit_refuses_a_width_terminated_bracket_rather_than_certifying_it() -> None:
    """A bracket that collapses before the stationarity residual is reached is NOT converged.

    Regression guard (CodeRabbit r1): the loop used to accept ``width <= tol`` as convergence,
    so it could return ``converged=True`` carrying a gradient larger than ``grad_tol`` — a fit
    whose own ``validate_report`` clause rejects it. Forced here by an absurdly tight
    ``grad_tol`` with a loose width ``tol``.
    """
    logits, labels, _ = _synthetic_logits(n=800, seed=13)
    with pytest.raises(T.TemperatureFitError, match="bracket collapsed to width"):
        T.fit_temperature(logits, labels, tol=1e-2, grad_tol=1e-30)


def test_per_class_reliability_rejects_a_non_posterior() -> None:
    """NaN / out-of-range inputs sort somewhere arbitrary and return a plausible-looking ECE."""
    n, k = 40, len(CLASS_ORDER)
    y = [i % k for i in range(n)]
    probs = np.full((n, k), 1.0 / k)
    good = T.per_class_reliability(y, probs)
    assert set(good) == set(CLASS_ORDER)
    for corrupt, needle in (
        (np.nan, "n_nonfinite"),
        (1.5, "max="),
        (-0.5, "min="),
        (np.inf, "n_nonfinite"),
    ):
        bad = probs.copy()
        bad[3, 2] = corrupt
        with pytest.raises(ValueError, match=needle):
            T.per_class_reliability(y, bad)


def test_portable_path_relativises_in_tree_inputs_only() -> None:
    from pathlib import Path

    checkout = Path(T.__file__).resolve().parents[3]
    rel = "data/processed/checkpoints/stage1_production/stage1.pt"
    assert T._portable_path(checkout / rel) == rel
    assert not T._portable_path(checkout / rel).startswith("/")  # no home dir in a public file
    # a worktree's inputs live in the MAIN checkout (DVC data is materialised only there), so
    # relativising against the worktree alone would fall back to absolute for every real input
    parts = checkout.parts
    if ".claude" in parts and parts[parts.index(".claude") : parts.index(".claude") + 2] == (
        ".claude",
        "worktrees",
    ):
        main_checkout = Path(*parts[: parts.index(".claude")])
        assert T._portable_path(main_checkout / rel) == rel
    outside = Path("/opt/elsewhere/stage1.pt")
    assert T._portable_path(outside) == "/opt/elsewhere/stage1.pt"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "2"])
def test_apply_temperature_rejects_a_non_temperature(bad) -> None:
    with pytest.raises(T.TemperatureFitError):
        T.apply_temperature(np.zeros((2, 8)), bad)


def test_apply_temperature_is_a_plain_division() -> None:
    z = np.array([[1.0, -2.0, 4.0]])
    assert np.array_equal(T.apply_temperature(z, 2.0), z / 2.0)
    # and it is arg-max preserving per row (the A11 Pin 4 half that IS invariant)
    rng = np.random.default_rng(5)
    z = rng.normal(size=(500, 8))
    for t in (0.3, 1.0, 3.7):
        assert np.array_equal(np.argmax(T.apply_temperature(z, t), 1), np.argmax(z, 1))


# --------------------------------------------------------------------------- #
# The whole-cluster halving (A11 Pin 5)
# --------------------------------------------------------------------------- #
def test_halving_is_whole_cluster_seeded_and_identified() -> None:
    # ASYMMETRIC sizes + fraction, so an inversion (fit<->read) cannot hide behind a count
    sizes = {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 17: 8}
    fit = T.calibration_fit_cluster_ids(sizes, fraction=0.3, seed=T.CALIB_SPLIT_SEED)
    again = T.calibration_fit_cluster_ids(sizes, fraction=0.3, seed=T.CALIB_SPLIT_SEED)
    assert fit == again, "the halving is not deterministic at a fixed seed"
    # IDENTITY, not size: this is the assertion an inversion fails
    assert fit == frozenset({12, 15, 17}), f"halving identity changed: {sorted(fit)}"
    assert sum(sizes[c] for c in fit) >= 0.3 * sum(sizes.values())
    # insertion order must not matter — the rule keys on content
    shuffled = dict(reversed(list(sizes.items())))
    assert T.calibration_fit_cluster_ids(shuffled, fraction=0.3, seed=T.CALIB_SPLIT_SEED) == fit
    # a different seed must pick a different bucket (else the seed is decorative)
    assert T.calibration_fit_cluster_ids(sizes, fraction=0.3, seed=1) != fit


def test_halving_at_the_pinned_fraction_splits_and_complements() -> None:
    sizes = {c: (c % 5) + 1 for c in range(200, 260)}
    fit = T.calibration_fit_cluster_ids(sizes)  # pinned 0.5 / CALIB_SPLIT_SEED
    read = set(sizes) - set(fit)
    assert fit and read and not (set(fit) & read)
    assert set(fit) | read == set(sizes)


@pytest.mark.parametrize(
    ("sizes", "kwargs", "match"),
    [
        ({}, {}, "cluster_sizes is empty"),
        ({1: 0}, {}, "non-positive count"),
        ({1: 2}, {"fraction": 0.5}, "halving degenerate"),
        ({1: 1, 2: 1}, {"fraction": 1.0}, "fraction must be a float"),
        ({1: 1, 2: 1}, {"fraction": 0.0}, "fraction must be a float"),
        ({1: 1, 2: 1}, {"seed": True}, "seed must be an int"),
    ],
)
def test_halving_fails_closed(sizes: dict, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        T.calibration_fit_cluster_ids(sizes, **kwargs)


# --------------------------------------------------------------------------- #
# The estimator seam: one binning, two consumers
# --------------------------------------------------------------------------- #
def test_reliability_bins_and_binned_ece_are_one_implementation() -> None:
    rng = np.random.default_rng(19)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int).tolist()
    rows = M.reliability_bins(y, p.tolist())
    assert len(rows) == M.ECE_N_BINS == 15  # ADR-0005 D11
    sizes = {r["n"] for r in rows}
    assert max(sizes) - min(sizes) <= 1, f"bins are not equal-mass: {sizes}"
    assert sum(r["weight"] for r in rows) == pytest.approx(1.0, abs=1e-12)
    assert all(rows[i]["conf"] <= rows[i + 1]["conf"] for i in range(len(rows) - 1))
    for key, debias in (("debiased_gap", True), ("gap", False)):
        resummed = sum(r["weight"] * r[key] for r in rows)
        assert resummed == pytest.approx(M.binned_ece(y, p.tolist(), debias=debias), abs=1e-15)
    assert M.reliability_bins([], []) == []
    assert math.isnan(M.binned_ece([], []))
    with pytest.raises(ValueError, match="same length"):
        M.reliability_bins([1], [0.5, 0.5])


def test_per_class_reliability_covers_every_class_and_re_sums() -> None:
    rng = np.random.default_rng(23)
    n, k = 900, len(CLASS_ORDER)
    probs = rng.random((n, k))
    probs /= probs.sum(axis=1, keepdims=True)
    y = rng.integers(0, k, size=n).tolist()
    out = T.per_class_reliability(y, probs)
    assert set(out) == set(CLASS_ORDER)
    for name, entry in out.items():
        resummed = sum(r["weight"] * r["debiased_gap"] for r in entry["reliability"])
        assert resummed == pytest.approx(entry["ece"], abs=1e-12), name
        assert entry["n_positive"] == sum(1 for t in y if CLASS_ORDER[t] == name)
        assert 0.0 <= entry["ece"] <= 1.0
    with pytest.raises(ValueError, match="positions but probs"):
        T.per_class_reliability(y[:-1], probs)
    with pytest.raises(ValueError, match="classes, expected"):
        T.per_class_reliability(y, probs[:, :3])


def test_ece_block_ci_resamples_clusters_not_positions() -> None:
    rng = np.random.default_rng(29)
    blocks = {
        cid: [(int(rng.random() < 0.3), float(rng.random())) for _ in range(40)]
        for cid in range(12)
    }
    ci = T.ece_block_ci(blocks, n_boot=50)
    assert ci["n_blocks"] == 12 and ci["n_boot"] == 50
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    single = T.ece_block_ci({0: blocks[0]}, n_boot=50)
    assert math.isnan(single["lower"]) and math.isnan(single["upper"])  # <2 blocks: no CI


# --------------------------------------------------------------------------- #
# The report contract — every clause shown to bite
# --------------------------------------------------------------------------- #
def _valid_report() -> dict:
    """A minimal report that validates, built the way the module builds one."""
    sizes = {c: 1 + (c % 3) for c in range(100, 130)}
    fit_ids = sorted(int(c) for c in T.calibration_fit_cluster_ids(sizes))
    read_ids = sorted(c for c in sizes if c not in set(fit_ids))
    rng = np.random.default_rng(31)
    n, k = 600, len(CLASS_ORDER)
    probs = rng.random((n, k))
    probs /= probs.sum(axis=1, keepdims=True)
    y = rng.integers(0, k, size=n).tolist()

    def _cond(temp: float) -> dict:
        per_class = T.per_class_reliability(y, probs)
        return {
            "temperature": temp,
            "per_class": per_class,
            "per_nt_f1_by_class": {},
            "boundary_iou_by_element": {},
            "gate4_core_min_f1": 0.5,
            "core_worst_ece": max(per_class[e]["ece"] for e in CORE_ELEMENTS),
            "n_positions": n,
            "n_positions_coverage_ge_2": 10,
        }

    return {
        "schema_version": T.SCHEMA_VERSION,
        "step": T.STEP,
        "generated_by": T.GENERATED_BY,
        "adr": T.ADR,
        "env_lock": T.ENV_LOCK,
        "generated_at_utc": "2026-07-30T00:00:00+00:00",
        "is_science": False,
        "gated": False,
        "stack_order": {
            "pinned": list(T.STACK_ORDER),
            "applied": ["train", "temperature_scale"],
            "prior_shift_applied": False,
            "named_posterior": True,
        },
        "split": {
            "fold": "selection_val",
            "is_d11_disjoint_calibration_split": False,
            "seed": T.CALIB_SPLIT_SEED,
            "fraction": T.CALIB_FIT_FRACTION,
            "n_clusters": len(sizes),
            "n_clusters_fit": len(fit_ids),
            "n_clusters_read": len(read_ids),
            "fit_cluster_ids": fit_ids,
            "read_cluster_ids": read_ids,
            "cluster_sizes": {str(c): v for c, v in sizes.items()},
        },
        "temperature": {
            "temperature": 2.0,
            "beta": 0.5,
            "n_positions": 1000,
            "n_classes": 8,
            "n_iterations": 40,
            "converged": True,
            "nll_per_position_initial": 1.2,
            "nll_per_position_final": 1.0,
            "grad_at_solution": 1e-7,
            "grad_tol": 1e-3,
            "bracket": list(T.BETA_BRACKET),
            "tol": T.FIT_TOL,
        },
        "operator_agreement": {"bit_exact": True, "max_abs_log_prob_diff": 0.0},
        "read": {
            "n_positions": n,
            "n_records": 20,
            "n_blocks": len(read_ids),
            "n_boot": T.DEFAULT_N_BOOT,
            "n_positions_argmax_changed": 3,
            "argmax_change_fraction": 3 / n,
            "n_positions_coverage_ge_2": 10,
            "uncalibrated": _cond(1.0),
            "tempered": _cond(2.0),
        },
        "disclosures": list(T.DISCLOSURES),
    }


def test_valid_report_validates_and_passes_its_gate() -> None:
    report = _valid_report()
    assert T.validate_report(report) == []
    clauses = T.derive_clauses(report)
    assert clauses["overall_pass"] is True
    assert T.read_is_out_of_sample(report) is True


@pytest.mark.parametrize(
    ("mutate", "expect"),
    [
        (lambda r: r.update(gated=True), "gated"),
        (lambda r: r.update(is_science=True), "is_science"),
        (lambda r: r.update(disclosures=[]), "disclosures"),
        (lambda r: r.update(schema_version=""), "schema_version"),
        (lambda r: r["stack_order"].update(prior_shift_applied=True), "prior_shift_applied"),
        (lambda r: r["stack_order"].update(applied=["train"]), "stack_order.applied"),
        (lambda r: r["stack_order"].update(pinned=["train"]), "stack_order.pinned"),
        (lambda r: r["split"].update(is_d11_disjoint_calibration_split=True), "D11"),
        (lambda r: r["temperature"].update(temperature=-1.0), "finite and > 0"),
        (lambda r: r["temperature"].update(converged=False), "unconverged"),
        (lambda r: r["temperature"].update(beta=0.9), "reciprocals"),
        (lambda r: r["temperature"].update(grad_at_solution=1.0), "grad_at_solution"),
        (lambda r: r["temperature"].update(nll_per_position_final=9.9), "lower its own"),
        (lambda r: r["operator_agreement"].update(bit_exact=False), "operator_agreement"),
        (lambda r: r["operator_agreement"].update(max_abs_log_prob_diff=1e-9), "bit-exactly"),
        (lambda r: r["read"].update(n_blocks=999), "n_blocks"),
        (lambda r: r["read"].update(n_positions_argmax_changed=None), "argmax_changed"),
        (lambda r: r["read"]["tempered"].update(core_worst_ece=0.0), "core_worst_ece"),
        (lambda r: r["read"]["tempered"]["per_class"].pop("background"), "CLASS_ORDER keys"),
        (
            lambda r: r["read"]["tempered"]["per_class"][CORE_ELEMENTS[0]].update(ece=0.99),
            "re-sum",
        ),
        (
            lambda r: r["read"]["uncalibrated"]["per_class"]["background"].update(reliability=[]),
            "reliability",
        ),
    ],
)
def test_every_validator_clause_bites(mutate, expect: str) -> None:
    report = _valid_report()
    assert T.validate_report(report) == []  # the mutation is what breaks it, not the fixture
    mutate(report)
    problems = T.validate_report(report)
    assert any(expect in p for p in problems), f"expected {expect!r} in {problems}"
    assert T.derive_clauses(report)["overall_pass"] is False


def test_inverted_halves_are_caught_by_identity_not_by_count() -> None:
    """The failure a symmetric count cannot see: fit and read swapped."""
    report = _valid_report()
    fit_ids = list(report["split"]["fit_cluster_ids"])
    read_ids = list(report["split"]["read_cluster_ids"])
    report["split"]["fit_cluster_ids"] = read_ids
    report["split"]["read_cluster_ids"] = fit_ids
    # counts are untouched by the swap — an n_clusters_* check would pass
    assert report["split"]["n_clusters_fit"] == len(fit_ids)
    assert T.read_is_out_of_sample(report) is False
    problems = T.validate_report(report)
    assert any("inverted" in p for p in problems), problems
    assert T.derive_clauses(report)["read_out_of_sample_for_T"] is False


def test_overlapping_halves_are_caught() -> None:
    report = _valid_report()
    report["split"]["read_cluster_ids"] = [
        *report["split"]["read_cluster_ids"],
        report["split"]["fit_cluster_ids"][0],
    ]
    problems = T.validate_report(report)
    assert any("complement" in p or "not disjoint" in p for p in problems), problems


def test_validator_is_total_on_junk() -> None:
    for junk in (None, [], "report", 3):
        problems = T.validate_report(junk)  # type: ignore[arg-type]
        assert problems and isinstance(problems, list)
        assert T.derive_clauses(junk if isinstance(junk, dict) else {})["overall_pass"] is False


@pytest.mark.parametrize(
    ("mutate", "expect"),
    [
        # each of these used to RAISE inside the validator (TypeError / KeyError), and a
        # validator that dies reports nothing — strictly worse than one that reports a problem
        (
            lambda r: r["read"]["tempered"]["per_class"]["background"].update(reliability=["x"]),
            "row",
        ),
        (
            lambda r: r["read"]["uncalibrated"]["per_class"]["Stem_I"]["reliability"][0].pop(
                "weight"
            ),
            "row",
        ),
        (
            lambda r: r["read"]["tempered"]["per_class"]["Stem_I"]["reliability"][0].update(
                debiased_gap="big"
            ),
            "row",
        ),
        (lambda r: r["read"]["uncalibrated"]["per_class"].update(Stem_I=None), "not a mapping"),
        (
            lambda r: r["read"]["tempered"]["per_class"]["Specifier"].pop("ece"),
            "core element's ece",
        ),
    ],
)
def test_validator_never_raises_on_a_malformed_report(mutate, expect: str) -> None:
    report = _valid_report()
    mutate(report)
    problems = T.validate_report(report)  # must RETURN, not raise
    assert any(expect in p for p in problems), f"expected {expect!r} in {problems}"
    assert T.derive_clauses(report)["overall_pass"] is False


def test_figure_data_carries_the_plotted_curves_only() -> None:
    data = T.figure_data(_valid_report())
    assert data["gated"] is False
    assert data["classes"][:3] == list(CORE_ELEMENTS)
    for name in data["classes"]:
        for cond in ("uncalibrated", "tempered"):
            curve = data["curves"][name][cond]
            assert len(curve["conf"]) == len(curve["acc"]) == len(curve["weight"])
            assert curve["ece"] == pytest.approx(
                sum(w * g for w, g in zip(curve["weight"], _gaps(curve), strict=True)), abs=1e-12
            )


def _gaps(curve: dict) -> list[float]:
    """|acc - conf| per bin, debiased the way the estimator does — a re-derivation of the
    plotted curve's own ECE from the points that are plotted."""
    out = []
    for conf, acc, w in zip(curve["conf"], curve["acc"], curve["weight"], strict=True):
        n = w * 600  # the fixture's n_positions
        sigma = math.sqrt(max(conf * (1.0 - conf), 0.0) / n)
        out.append(max(0.0, abs(acc - conf) - sigma * math.sqrt(2.0 / math.pi)))
    return out


def test_cli_parser_exposes_run_and_plot() -> None:
    parser = T.build_parser()
    run = parser.parse_args(["run", "--max-records", "4", "--n-boot", "5"])
    assert run.command == "run" and run.max_records == 4 and run.n_boot == 5
    plot = parser.parse_args(["plot-figures"])
    assert plot.command == "plot-figures" and plot.figures_dir == str(T.FIGURES_DIR)


def test_pins_are_not_restated_locally() -> None:
    """The estimator resolution and the gate value are imported, never re-declared here."""
    from pathlib import Path

    from tbox_finder import power

    assert M.ECE_N_BINS == 15
    assert M.ECE_GATE == power.ECE_GATE == 0.05
    src = Path(T.__file__ or "")
    assert src.name == "temperature.py"
    text = src.read_text(encoding="utf-8")
    for forbidden in ("ECE_N_BINS =", "ECE_GATE ="):
        assert forbidden not in text, f"{forbidden!r} must live in metrics/power, not calib"


def test_disclosures_carry_the_scope_that_must_travel() -> None:
    """The scope is shipped as constants precisely so it cannot be forgotten in prose."""
    assert isinstance(T.DISCLOSURES, tuple) and len(T.DISCLOSURES) >= 8
    assert all(isinstance(d, str) and d.strip() for d in T.DISCLOSURES)
    blob = " ".join(T.DISCLOSURES)
    for must_say in ("NON-GATED", "P3 exit", "P3-02", "selection_val", "prior-shift", "arg-max"):
        assert must_say in blob, f"the disclosures no longer say {must_say!r}"
