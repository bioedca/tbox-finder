"""P3-07 — the Stage-2 recalibration stack (PRD §11/§12; ADR-0005 D11).

**Torch-free and scipy-free by construction.** CI's test env installs numpy/pandas/
scikit-learn and no torch; the whole stack is closed-form or an exact convex 1-D solve, so
every tier here runs where the suite actually runs rather than ``importorskip``-ing green.

What each tier locks, and why that tier is the one that could bite:

* **the fit against hand-derived roots** — ``temperature_scale`` delegates to
  ``calib.temperature.fit_temperature``, so "the two agree" would be a tautology
  ([[promote-dont-duplicate-is-a-correctness-rule]]). The primary check is therefore an
  **independently derived closed form**: for ``z = [+a, +a, −a]``, ``y = [1, 0, 0]`` the
  binary NLL's stationarity condition ``Σ z_i(σ(βz_i) − y_i) = 0`` reduces to
  ``3σ(βa) = 2``, so ``β* = ln2/a`` exactly. A second case with different ``a`` and a
  different root (``4σ = 3`` ⇒ ``β* = ln3/a``) pins the scale dependence, a ternary search
  in the *test* re-finds the minimum by a different algorithm (minimising the value, not
  rooting the derivative), and a manufactured-miscalibration draw recovers a known ``T``;
* **the calib-only guarantee** — the imp.md gate is "``T`` fit only on ``calib``, never
  test". A count-only check is blind to a fold-sense inversion, so the fixture uses
  **asymmetric** rung sizes and asserts **identity** (the ``T`` from the mixed table equals
  the ``T`` from the intended rows fitted alone, and *differs* from the swapped-sense one)
  ([[symmetric-count-fixture-blind-to-inversion]]). Every refusal is paired with the
  identical clean input succeeding, so a guard that refused everything would fail here
  ([[raises-test-needs-a-positive-control]]);
* **the emptiness direction** — an all-``False`` calib column filters clean and fits on
  nothing, which reads exactly like a working filter ([[clauses-must-guard-emptiness]],
  [[namespace-mismatch-invisible-noop]]). It must raise, and it must say what it saw;
* **the pandas-3 direction** — a missing ``calib`` cell arrives as ``NaN`` in the training
  env, and ``NaN`` is **truthy**, so ``bool(cell)`` would silently promote a row with no
  flag into the calibration set ([[pandas-3-nan-truthy-in-training-env]]). NaN is asserted
  as a literal, in both columns;
* **the prior-shift closed form** — hand-computed log-odds on a known prevalence swing,
  the posterior-form equivalence, exact invertibility, and the band predicate shown to
  *discriminate* (ADR-0005 D7's own ``100:1`` benchmark prevalence must fall outside the
  PRD's ``10³–10⁴:1`` deployment band, or the predicate certifies nothing);
* **the no-fabricated-prior contract** — the module must be unable to supply a deployment
  prior on its own, so a half-specified shift raises and neither prior has a default
  ([[pinned-constant-that-nothing-reads]] in reverse: the band is read, never defaulted to).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbox_finder.calib import recalibrate as R
from tbox_finder.calib import temperature as T


# --------------------------------------------------------------------------- #
# Helpers — independent arithmetic, never the module's own
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    """Plain-math sigmoid, so the expectations are not the module's implementation."""
    return 1.0 / (1.0 + math.exp(-x))


def _binary_nll(z: np.ndarray, y: np.ndarray, beta: float) -> float:
    """``Σ_i [softplus(βz_i) − βz_i·y_i]`` — the objective, written out longhand."""
    total = 0.0
    for zi, yi in zip(z.tolist(), y.tolist(), strict=True):
        s = beta * zi
        total += math.log1p(math.exp(-abs(s))) + max(s, 0.0) - s * yi
    return total


def _ternary_min(z: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Minimise the convex NLL by ternary search — a different algorithm from the module's.

    The module bisects the *derivative* to find its root; this narrows a bracket on the
    *value*. Agreement between the two is therefore evidence about the answer, not about a
    shared implementation.
    """
    for _ in range(400):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _binary_nll(z, y, m1) < _binary_nll(z, y, m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def _table(spec: list[tuple[str, float, int]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``[(rung, logit, label), ...]`` → the three parallel columns the API takes."""
    rungs = [row[0] for row in spec]
    z = np.array([row[1] for row in spec], dtype=np.float64)
    y = np.array([row[2] for row in spec], dtype=np.int64)
    return z, y, rungs


#: A **non-separable** general-purpose fixture. Separability is not a detail here: the fit
#: legitimately *refuses* a perfectly-separated set (``β → ∞``, ``T → 0``), so a fixture whose
#: labels agree with the sign of every logit tests the refusal path and nothing else. Each arm
#: below therefore carries at least one pair of equal logits with opposite labels.
_NONSEP: list[tuple[float, int]] = [
    (-2.0, 0),
    (-1.0, 1),
    (-0.5, 0),
    (0.5, 1),
    (1.0, 0),
    (2.0, 1),
    (0.5, 0),
    (-0.5, 1),
]

#: Deliberately **asymmetric** rung sizes (7 calib / 4 train / 3 val / 5 test): with equal
#: arms a swapped fold sense would refuse the right *count* of the *wrong* rows and every
#: count-based assertion would still pass. The calib and test arms are each independently
#: fittable, so the inversion test can actually run both ways round.
_MIXED: list[tuple[str, float, int]] = [
    ("calib", 1.0, 1),
    ("calib", 1.0, 0),
    ("calib", -1.0, 0),
    ("calib", 2.0, 1),
    ("calib", -0.5, 0),
    ("calib", 0.25, 1),
    ("calib", -2.0, 0),
    ("train", 3.0, 1),
    ("train", -3.0, 0),
    ("train", 0.1, 1),
    ("train", 0.2, 0),
    ("val", 4.0, 1),
    ("val", -4.0, 0),
    ("val", 0.3, 1),
    ("test", 5.0, 1),
    ("test", -5.0, 0),
    ("test", 0.4, 1),
    ("test", 0.4, 0),
    ("test", 1.5, 1),
]

#: The smallest non-separable calib table — the hand-derived ``β* = ln2`` fixture reused
#: wherever a test needs *a* successful fit rather than a specific one.
_TINY: list[tuple[str, float, int]] = [("calib", 1.0, 1), ("calib", 1.0, 0), ("calib", -1.0, 0)]


# --------------------------------------------------------------------------- #
# The fit — hand-derived closed forms first, delegation second
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "a,ratio,expected_numerator",
    [
        (1.0, "3s=2", math.log(2.0)),  # z=[+a,+a,-a], y=[1,0,0] -> 3σ(βa)=2 -> βa=ln2
        (2.0, "3s=2", math.log(2.0)),  # same root, different scale: β* = ln2 / a
    ],
)
def test_fit_recovers_the_hand_derived_root_three_point(a, ratio, expected_numerator):
    """``dNLL/dβ = a[3σ(βa) − 2]`` ⇒ ``β* = ln2/a``, ``T* = a/ln2`` — derived, not delegated."""
    z, y, rungs = _table([("calib", a, 1), ("calib", a, 0), ("calib", -a, 0)])
    result = R.temperature_scale(z, y, rung=rungs, grad_tol=1e-8)
    assert result.beta == pytest.approx(expected_numerator / a, abs=1e-6)
    assert result.temperature == pytest.approx(a / expected_numerator, abs=1e-6)
    assert result.fitted_on == R.CALIB_RUNG
    assert result.n_fitted == 3


def test_fit_recovers_a_second_hand_derived_root_four_point():
    """``z=[+2,+2,+2,−2]``, ``y=[1,1,0,0]`` ⇒ ``4σ(2β)=3`` ⇒ ``β* = ln3/2``.

    A second root at a second scale: a fitter that happened to return ``ln2/a`` for any
    input would pass the case above and fail here.
    """
    z, y, rungs = _table(
        [("calib", 2.0, 1), ("calib", 2.0, 1), ("calib", 2.0, 0), ("calib", -2.0, 0)]
    )
    result = R.temperature_scale(z, y, rung=rungs, grad_tol=1e-8)
    assert result.beta == pytest.approx(math.log(3.0) / 2.0, abs=1e-6)
    assert result.temperature == pytest.approx(2.0 / math.log(3.0), abs=1e-6)


def test_fit_agrees_with_an_independent_minimiser_of_the_same_objective():
    """Ternary search on the NLL value re-finds the root the module bisects the derivative for."""
    z, y, rungs = _table([("calib", v, lab) for v, lab in _NONSEP])
    result = R.temperature_scale(z, y, rung=rungs, grad_tol=1e-9)
    assert result.beta == pytest.approx(_ternary_min(z, y, 1e-3, 1e3), abs=1e-5)


def test_fit_recovers_a_manufactured_miscalibration():
    """Draw ``y ~ Bernoulli(σ(base))``, hand the fitter ``base·sharpen``, expect ``T ≈ sharpen``.

    The only honest end-to-end check of a calibration fitter: manufacture the miscalibration,
    then see whether it is undone. Statistical, so the tolerance is loose — but a fitter
    returning ``T ≡ 1`` (a no-op) or ``T ≡ 1/sharpen`` (an inverted parameterisation) misses
    by far more than the tolerance.
    """
    rng = np.random.default_rng(11)
    n = 20_000
    sharpen = 2.5
    base = rng.normal(0.0, 1.8, size=n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-base))).astype(np.int64)
    result = R.temperature_scale(base * sharpen, y, rung=["calib"] * n)
    assert result.temperature == pytest.approx(sharpen, rel=0.05)


def test_fit_lowers_the_objective_it_minimises():
    z, y, rungs = _table([("calib", v, lab) for v, lab in _NONSEP])
    result = R.temperature_scale(z, y, rung=rungs)
    assert _binary_nll(z, y, result.beta) < _binary_nll(z, y, 1.0)
    assert result.fit.nll_per_position_final <= result.fit.nll_per_position_initial


def test_fit_is_bit_deterministic():
    z, y, rungs = _table([("calib", v, lab) for v, lab in _NONSEP])
    first = R.temperature_scale(z, y, rung=rungs)
    second = R.temperature_scale(z, y, rung=rungs)
    assert first.temperature == second.temperature
    assert first.beta == second.beta


def test_the_one_logit_stack_is_the_delegation_and_it_is_exact():
    """Consistency check, explicitly **secondary** to the closed forms above.

    ``softmax([0, βz])[1] = σ(βz)``, so the multi-class fit on the ``[0, z]`` stack *is* the
    binary fit. This asserts the adapter builds that stack correctly; it says nothing about
    whether the shared fit is right, which is what the hand-derived roots are for.
    """
    z, y, rungs = _table([("calib", v, lab) for v, lab in _NONSEP])
    mine = R.temperature_scale(z, y, rung=rungs)
    theirs = T.fit_temperature(np.column_stack((np.zeros_like(z), z)), y)
    assert mine.temperature == theirs.temperature
    assert mine.beta == theirs.beta


@pytest.mark.parametrize(
    "spec,reason",
    [
        ([("calib", 3.0, 1), ("calib", -3.0, 0)], "perfectly separated -> beta -> inf"),
        ([("calib", 5.0, 1), ("calib", 6.0, 1), ("calib", -5.0, 0)], "separated, T -> 0"),
    ],
)
def test_degenerate_calibration_sets_raise_rather_than_return_a_bracket_endpoint(spec, reason):
    """The A11 refusals carry through the adapter unchanged — no T is invented for them."""
    z, y, rungs = _table(spec)
    with pytest.raises(T.TemperatureFitError):
        R.temperature_scale(z, y, rung=rungs)


# --------------------------------------------------------------------------- #
# The calib-only guarantee — identity, not counts; asymmetric, not balanced
# --------------------------------------------------------------------------- #
def test_t_is_fitted_on_the_calib_rows_and_on_no_others():
    """Identity: the mixed-table ``T`` equals the ``T`` from those same rows fitted alone."""
    z, y, rungs = _table(_MIXED)
    mixed = R.temperature_scale(z, y, rung=rungs)

    calib_spec = [row for row in _MIXED if row[0] == R.CALIB_RUNG]
    z_only, y_only, rungs_only = _table(calib_spec)
    alone = R.temperature_scale(z_only, y_only, rung=rungs_only)

    assert mixed.temperature == alone.temperature  # exact: same arithmetic, same rows
    assert mixed.n_fitted == alone.n_fitted == len(calib_spec) == 7
    assert mixed.n_total == len(_MIXED) == 19


def test_the_filter_is_not_vacuous_the_other_rows_would_move_t():
    """Fitting the whole table gives a *different* ``T`` — so the selection did something."""
    z, y, rungs = _table(_MIXED)
    selected = R.temperature_scale(z, y, rung=rungs)
    everything = R.temperature_scale(z, y, rung=[R.CALIB_RUNG] * len(rungs))
    assert selected.temperature != everything.temperature


def test_swapping_the_calib_and_test_senses_changes_the_fit():
    """The inversion a count-only assertion cannot see.

    Relabel calib↔test and the fit moves — proving the selector reads the rung *sense*, not
    merely some 7 rows. With balanced arms this test would pass under the inversion.
    """
    z, y, rungs = _table(_MIXED)
    straight = R.temperature_scale(z, y, rung=rungs)
    swapped_rungs = [
        "test" if r == R.CALIB_RUNG else (R.CALIB_RUNG if r == "test" else r) for r in rungs
    ]
    swapped = R.temperature_scale(z, y, rung=swapped_rungs)
    assert swapped.n_fitted == 5  # the test arm's size, not the calib arm's
    assert swapped.temperature != straight.temperature
    # ...and it is the *test* rows it fitted, asserted by identity rather than by count.
    test_only = [(R.CALIB_RUNG, v, lab) for r, v, lab in _MIXED if r == "test"]
    z_t, y_t, rungs_t = _table(test_only)
    assert swapped.temperature == R.temperature_scale(z_t, y_t, rung=rungs_t).temperature


def test_the_census_counts_every_rung_including_the_empty_ones():
    z, y, rungs = _table(_MIXED)
    counts = R.temperature_scale(z, y, rung=rungs).counts()
    assert counts == {"calib": 7, "train": 4, "test": 5, "val": 3}
    assert set(counts) == set(R.RUNG_VOCABULARY)  # a 0-row rung is reported, not absent

    calib_only = [row for row in _MIXED if row[0] == R.CALIB_RUNG]
    z2, y2, rungs2 = _table(calib_only)
    assert R.temperature_scale(z2, y2, rung=rungs2).counts() == {
        "calib": 7,
        "train": 0,
        "val": 0,
        "test": 0,
    }


def test_an_empty_calib_rung_raises_and_names_what_it_saw():
    """The all-``False`` calib column: filters clean, fits on nothing, must not look fine.

    The message is asserted in three parts, not one: the phrase, the **census** (so a
    silently-wrong census cannot hide behind a correct headline), and the count of rows on
    ``GRADED_RUNGS`` — which is the only place that constant is read, so asserting only the
    first clause would leave it a pinned constant nothing checks.
    """
    spec = [row for row in _MIXED if row[0] != R.CALIB_RUNG]
    z, y, rungs = _table(spec)
    with pytest.raises(ValueError) as excinfo:
        R.temperature_scale(z, y, rung=rungs)
    message = str(excinfo.value)
    assert "no rows on the 'calib' rung out of 12" in message
    assert "{'calib': 0, 'test': 5, 'train': 4, 'val': 3}" in message
    assert "the 8 rows of the graded rungs ['val', 'test']" in message  # 5 test + 3 val

    # Positive control: the identical rows, with the (non-separable) train arm relabelled
    # calib, fit fine — so the refusal above is about emptiness, not about these rows.
    rescued = ["calib" if i < 4 else r for i, r in enumerate(rungs)]
    assert R.temperature_scale(z, y, rung=rescued).n_fitted == 4


@pytest.mark.parametrize(
    "labels,described",
    [
        ([1, 1], "all-positive"),
        ([0, 0], "all-negative"),
    ],
)
def test_a_single_class_calib_rung_raises(labels, described):
    """**Both** halves of the compound guard, sabotaged separately.

    ``n_pos == 0 or n_pos == n_fitted`` is a disjunct, and a test that exercises only the
    all-positive half leaves the other free to be deleted
    ([[sabotage-attribution-names-the-test]]). The all-negative half is the realistic one: a
    P3-02 join that drops the positive class yields a calib carve of pure negatives, on which
    the fit **converges cleanly** and returns a fully certified `T` if the guard is gone.
    """
    z, y, rungs = _table([("calib", -1.0, labels[0]), ("calib", 2.0, labels[1])])
    with pytest.raises(ValueError, match="single-class"):
        R.temperature_scale(z, y, rung=rungs)

    # Positive control: opposite labels on the same rung and it is a fit again.
    z2, y2, rungs2 = _table([*_TINY, ("test", -1.0, 0)])
    assert R.temperature_scale(z2, y2, rung=rungs2).n_fitted == 3


@pytest.mark.parametrize("token", ["None", "nan", "calibration", "Calib", "CALIB", "", "  "])
def test_an_unrecognised_rung_token_is_refused_not_filtered(token):
    """A token nobody recognises must not evaporate a row and leave the counts looking clean."""
    spec = list(_MIXED)
    spec[0] = (token, 1.0, 1)
    z, y, rungs = _table(spec)
    with pytest.raises(ValueError, match="unrecognised rung label"):
        R.temperature_scale(z, y, rung=rungs)

    # Positive control: the same table with that row labelled properly fits.
    spec[0] = (R.CALIB_RUNG, 1.0, 1)
    z, y, rungs = _table(spec)
    assert R.temperature_scale(z, y, rung=rungs).n_fitted == 7


@pytest.mark.parametrize(
    "bad,expected",
    [
        ({"labels": np.array([0, 1, 2])}, "labels must be binary 0/1"),
        ({"logits": np.zeros((3, 2))}, "logits must be 1-D"),
        ({"labels": np.array([[0], [1], [0]])}, "labels must be 1-D"),
    ],
)
def test_shape_and_domain_violations_raise(bad, expected):
    """``match=`` on **this module's own** message, not a bare ``ValueError``.

    A bare `pytest.raises(ValueError)` passes here even with both of these guards deleted,
    because the delegated multi-class fitter raises its own (differently-worded, and for the
    2-D case *wrongly-scoped*) error further in. Matching the text is what makes the guard
    this module owns the thing under test.
    """
    z, y, rungs = _table(_TINY)
    kwargs = {"logits": z, "labels": y}
    kwargs.update(bad)
    with pytest.raises(ValueError, match=expected):
        R.temperature_scale(kwargs["logits"], kwargs["labels"], rung=rungs)

    # Positive control: the untouched arguments succeed.
    assert R.temperature_scale(z, y, rung=rungs).n_fitted == 3


@pytest.mark.parametrize(
    "labels,expected",
    [
        (np.array([1.0, 0.6, 0.0]), "must be integral"),
        (np.array([1.0, 1.7, 0.0]), "must be integral"),
        (np.array([1.0, -0.9, 0.0]), "must be integral"),
        (np.array([1.0, np.nan, 0.0]), "non-finite"),
        (np.array([1.0, np.inf, 0.0]), "non-finite"),
        (np.array(["1", "0", "1"]), "boolean or numeric"),
        (np.array([None, 1, 0], dtype=object), "boolean or numeric"),
    ],
)
def test_a_non_integral_label_is_refused_before_the_int64_cast(labels, expected):
    """``astype(np.int64)`` truncates toward zero, so a range test placed *after* it is blind.

    A target written as ``0.6`` becomes ``0`` and the fit then runs — successfully, and
    bit-identically to the truncated targets — on values the caller never wrote; a ``NaN``
    label becomes the int64 sentinel and surfaces as a misleading ``range [-9.2e18, 1]``.
    The domain check therefore happens in the original dtype.
    """
    z = np.array([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match=expected):
        R.temperature_scale(z, labels, rung=["calib"] * 3)

    # Positive control: the same logits with the integral targets the caller meant.
    assert R.temperature_scale(z, np.array([1.0, 0.0, 0.0]), rung=["calib"] * 3).n_fitted == 3


@pytest.mark.parametrize("labels", [[True, False, False], np.array([1, 0, 0], dtype=np.int8)])
def test_boolean_and_integer_label_dtypes_are_accepted(labels):
    """The refusal above must not also refuse the dtypes parquet actually delivers."""
    z = np.array([1.0, 1.0, -1.0])
    result = R.temperature_scale(z, labels, rung=["calib"] * 3, grad_tol=1e-8)
    # ...and land on the same hand-derived root the float path does: T* = 1/ln2.
    assert result.temperature == pytest.approx(1.0 / math.log(2.0), abs=1e-6)


def test_length_disagreements_raise():
    z, y, rungs = _table(_TINY)
    with pytest.raises(ValueError, match="rung carries"):
        R.temperature_scale(z, y, rung=rungs[:2])
    with pytest.raises(ValueError, match="labels carries"):
        R.temperature_scale(z, y[:2], rung=rungs)


def test_non_finite_logits_are_refused():
    z, y, rungs = _table(_TINY)
    z_bad = z.copy()
    z_bad[1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        R.temperature_scale(z_bad, y, rung=rungs)
    assert R.temperature_scale(z, y, rung=rungs).n_fitted == 3


# --------------------------------------------------------------------------- #
# assign_rung / rung_labels — the P3-02 invariant, re-derived where it matters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "calib,fold,expected",
    [
        (True, "train", "calib"),
        (np.True_, "train", "calib"),
        (False, "train", "train"),
        (False, "val", "val"),
        (False, "test", "test"),
        (np.False_, "test", "test"),
    ],
)
def test_assign_rung_maps_the_calib_overlay_onto_the_scheme_a_folds(calib, fold, expected):
    assert R.assign_rung(calib=calib, fold_random=fold) == expected


@pytest.mark.parametrize("fold", ["val", "test"])
def test_a_calib_row_outside_the_training_fold_raises(fold):
    """The P3-02 invariant (``splits._assert_calib_disjoint``), re-derived at the fit site."""
    with pytest.raises(ValueError, match="P3-02 carve"):
        R.assign_rung(calib=True, fold_random=fold)
    assert R.assign_rung(calib=True, fold_random="train") == "calib"  # positive control


@pytest.mark.parametrize("missing", [None, float("nan"), np.nan])
def test_a_missing_calib_flag_raises_rather_than_reading_nan_as_true(missing):
    """``NaN`` is **truthy**: ``bool(float('nan')) is True``, asserted here as a literal.

    Under the training env's pandas 3 a null cell arrives as ``NaN``, so a guard written as
    ``bool(cell)`` would promote a row with no calibration flag straight into the set ``T`` is
    fitted on ([[pandas-3-nan-truthy-in-training-env]]).
    """
    assert bool(float("nan")) is True  # the bite, stated outright
    with pytest.raises(ValueError, match="calib is missing"):
        R.assign_rung(calib=missing, fold_random="train")
    assert R.assign_rung(calib=True, fold_random="train") == "calib"  # positive control


@pytest.mark.parametrize("token", ["False", "false", "0", "FALSE"])
def test_a_stringified_false_calib_flag_is_read_as_false_not_as_truthy(token):
    """``bool("False")`` is ``True`` — the second half of the same fail-open as NaN.

    The NaN guard closes the *missing* spelling; this closes the *rendered* one. A
    string-typed `calib` column (an object-dtype widening, a re-materialised join) carrying
    an explicit ``"False"`` would otherwise be promoted into the set ``T`` is fitted on, with
    every count still looking clean. Read through ``masking.bool_or_none``, the same parse
    ``stage2.train`` uses for the fold columns.
    """
    assert bool(token) is True  # the bite, stated outright
    assert R.assign_rung(calib=token, fold_random="train") == "train"


@pytest.mark.parametrize("token", ["True", "true", "1", "TRUE"])
def test_a_stringified_true_calib_flag_is_still_the_calib_rung(token):
    """Positive control for the refusal above — the parse is strict, not blanket-rejecting."""
    assert R.assign_rung(calib=token, fold_random="train") == "calib"


@pytest.mark.parametrize("bad", ["yes", "no", "maybe", "0.5", [0], (0,), 0.5, -1, 2])
def test_a_non_boolean_calib_flag_raises_rather_than_being_coerced(bad):
    """Anything outside the boolean vocabulary is an error, never a silent ``True``.

    ``0.5``, ``-1``, ``2``, ``[0]`` and ``(0,)`` are all truthy-or-falsy under bare
    ``bool()`` and none of them is a calibration flag; a column carrying them is a schema
    fault, and a fault must not resolve to a rung.
    """
    with pytest.raises(ValueError, match="neither missing nor boolean"):
        R.assign_rung(calib=bad, fold_random="train")
    assert R.assign_rung(calib=True, fold_random="train") == "calib"  # positive control


def test_a_string_typed_calib_column_does_not_inflate_the_fit():
    """End to end: three genuine calib rows in a stringified column stay three.

    Before the strict parse this returned ``['calib'] * 6`` and `temperature_scale` reported
    `n_fitted=6` — three *training* rows fitted as if they were the carve, with the census
    agreeing. The identity assertion is the fitted `T`: it must equal the `T` from the three
    intended rows alone, and differ from the six-row one.
    """
    calib = ["True", "True", "True", "False", "False", "False"]
    fold = ["train"] * 6
    z = np.array([1.0, 1.0, -1.0, 3.0, -3.0, 0.2])
    y = np.array([1, 0, 0, 1, 0, 1])

    labels = R.rung_labels(calib=calib, fold_random=fold)
    assert labels == ["calib", "calib", "calib", "train", "train", "train"]

    got = R.temperature_scale(z, y, rung=labels)
    assert got.n_fitted == 3
    intended = R.temperature_scale(z[:3], y[:3], rung=["calib"] * 3)
    assert got.temperature == intended.temperature
    inflated = R.temperature_scale(z, y, rung=["calib"] * 6)
    assert got.temperature != inflated.temperature


@pytest.mark.parametrize("missing", [None, float("nan"), "", "   "])
def test_a_missing_fold_raises(missing):
    with pytest.raises(ValueError, match="fold_random is missing"):
        R.assign_rung(calib=False, fold_random=missing)


@pytest.mark.parametrize("fold", ["Train", "selection_val", "None", "nan", "0"])
def test_an_unrecognised_fold_raises(fold):
    with pytest.raises(ValueError, match="unrecognised fold_random"):
        R.assign_rung(calib=False, fold_random=fold)


def test_calib_is_never_a_fourth_fold_random_value():
    with pytest.raises(ValueError, match="boolean overlay"):
        R.assign_rung(calib=False, fold_random="calib")


def test_rung_labels_reports_the_offending_row_index():
    with pytest.raises(ValueError, match=r"row 2: .*unrecognised fold_random"):
        R.rung_labels(calib=[False, False, False], fold_random=["train", "val", "Train"])
    assert R.rung_labels(calib=[True, False], fold_random=["train", "test"]) == ["calib", "test"]


def test_rung_labels_refuses_ragged_columns():
    with pytest.raises(ValueError, match="fold_random carries"):
        R.rung_labels(calib=[True, False], fold_random=["train"])


def test_rung_labels_feeds_temperature_scale_end_to_end():
    """The intended call shape: derive rungs from the two real columns, then fit."""
    calib = [True, True, True, False, False, False]
    fold = ["train", "train", "train", "train", "val", "test"]
    z = np.array([1.0, 1.0, -1.0, 9.0, -9.0, 9.0])
    y = np.array([1, 0, 0, 1, 0, 1])
    result = R.temperature_scale(z, y, rung=R.rung_labels(calib=calib, fold_random=fold))
    assert result.n_fitted == 3
    assert result.counts() == {"calib": 3, "train": 1, "val": 1, "test": 1}


# --------------------------------------------------------------------------- #
# Monotonicity — temperature scaling and the prior shift both preserve the ranking
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0, 1.4426950408889634, 3.7])
def test_temperature_scaling_is_strictly_monotone_in_the_raw_logit(temperature):
    z = np.array([-6.0, -2.3, -0.4, 0.0, 0.15, 1.9, 4.4, 11.0])
    payload = R.calibrated_posterior(z, temperature=temperature)
    p = payload[R.NAMED_POSTERIOR_KEY]
    assert np.all(np.diff(p) > 0.0)
    assert list(np.argsort(p)) == list(np.argsort(z))


def test_the_prior_shift_moves_every_posterior_but_reorders_none():
    z = np.array([-6.0, -2.3, -0.4, 0.0, 0.15, 1.9, 4.4])
    payload = R.calibrated_posterior(
        z, temperature=1.3, source_prior=0.8, target_prior=R.prior_from_odds_ratio(1e3)
    )
    named = payload[R.NAMED_POSTERIOR_KEY]
    shifted = payload[R.PRIOR_SHIFTED_POSTERIOR_KEY]
    assert np.all(shifted < named)  # the target prior is far lower, so every posterior drops
    assert list(np.argsort(shifted)) == list(np.argsort(named)) == list(np.argsort(z))


@pytest.mark.parametrize(
    "bad", [0.0, -0.0, -0.1, float("nan"), float("inf"), float("-inf"), True, "1.0", None]
)
def test_a_degenerate_temperature_is_refused(bad):
    """``T`` must be finite and strictly positive — ``T = 0`` divides, ``T < 0`` inverts the
    ranking, and ``bool`` would read ``True`` as ``T = 1``."""
    z = np.array([-1.0, 0.0, 1.0])
    with pytest.raises((T.TemperatureFitError, ValueError)):
        R.calibrated_posterior(z, temperature=bad)
    assert R.calibrated_posterior(z, temperature=1.0)["n_rows"] == 3  # positive control


def test_the_posterior_does_not_overflow_in_either_tail():
    z = np.array([-800.0, -40.0, 0.0, 40.0, 800.0])
    with np.errstate(over="raise", invalid="raise"):
        p = R.posterior_from_logits(z)
    assert np.isfinite(p).all()
    assert p[0] == 0.0
    assert p[-1] == 1.0
    assert p[2] == pytest.approx(0.5, abs=1e-15)


# --------------------------------------------------------------------------- #
# The prior shift — closed form, equivalence, invertibility
# --------------------------------------------------------------------------- #
def test_log_odds_shift_matches_the_hand_computed_value():
    """``π_s=0.5``, ``π_t=0.001`` ⇒ ``Δ = ln(0.001/0.999) − 0 = −ln(999)``."""
    shift = R.log_odds_shift(source_prior=0.5, target_prior=0.001)
    assert shift == pytest.approx(-math.log(999.0), rel=0, abs=1e-12)
    assert shift == pytest.approx(-6.906754778648554, rel=0, abs=1e-12)


@pytest.mark.parametrize("r_source,r_target", [(1.0, 1e3), (10.0, 1e4), (100.0, 1e3), (1e4, 10.0)])
def test_log_odds_shift_of_two_odds_ratios_is_exactly_log_of_their_quotient(r_source, r_target):
    """``prior_from_odds_ratio(r)`` has odds exactly ``1/r``, so ``Δ = ln(r_s) − ln(r_t)``."""
    shift = R.log_odds_shift(
        source_prior=R.prior_from_odds_ratio(r_source),
        target_prior=R.prior_from_odds_ratio(r_target),
    )
    assert shift == pytest.approx(math.log(r_source / r_target), rel=0, abs=1e-12)


def test_prior_shift_equals_the_posterior_form_odds_ratio_correction():
    """``p' = p·r/(p·r + 1 − p)`` with ``r = odds_t/odds_s`` — the same rule, written the
    other way, computed here with plain floats rather than the module's arithmetic."""
    z = np.array([-3.0, -0.7, 0.0, 0.9, 2.5])
    p_s, p_t = 0.75, 0.002
    r = (p_t / (1.0 - p_t)) / (p_s / (1.0 - p_s))
    expected = [(_sigmoid(v) * r) / (_sigmoid(v) * r + 1.0 - _sigmoid(v)) for v in z.tolist()]
    got = R.posterior_from_logits(R.prior_shift(z, source_prior=p_s, target_prior=p_t))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def test_prior_shift_is_exactly_invertible_and_the_identity_shift_is_a_no_op():
    z = np.array([-4.0, -0.25, 0.0, 1.1, 7.3])
    there = R.prior_shift(z, source_prior=0.8, target_prior=1e-4)
    back = R.prior_shift(there, source_prior=1e-4, target_prior=0.8)
    np.testing.assert_allclose(back, z, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        R.prior_shift(z, source_prior=0.3, target_prior=0.3), z, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_prior": 0.0, "target_prior": 0.5},
        {"source_prior": 1.0, "target_prior": 0.5},
        {"source_prior": 0.5, "target_prior": 0.0},
        {"source_prior": 0.5, "target_prior": 1.0},
        {"source_prior": -0.1, "target_prior": 0.5},
        {"source_prior": 0.5, "target_prior": 1.5},
        {"source_prior": float("nan"), "target_prior": 0.5},
        {"source_prior": 0.5, "target_prior": float("inf")},
        {"source_prior": True, "target_prior": 0.5},
        {"source_prior": "0.5", "target_prior": 0.5},
        {"source_prior": None, "target_prior": 0.5},
    ],
)
def test_a_degenerate_prior_is_refused(kwargs):
    z = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        R.prior_shift(z, **kwargs)
    # Positive control: identical call with both priors legal.
    assert R.prior_shift(z, source_prior=0.5, target_prior=0.25).shape == (3,)


# --------------------------------------------------------------------------- #
# Priors, odds ratios and the PRD band
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio", [1e-3, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6])
def test_prior_and_odds_ratio_round_trip(ratio):
    assert R.odds_ratio_from_prior(R.prior_from_odds_ratio(ratio)) == pytest.approx(
        ratio, rel=1e-12
    )


def test_prior_from_odds_ratio_is_the_stated_formula():
    assert R.prior_from_odds_ratio(1e3) == pytest.approx(1.0 / 1001.0, rel=0, abs=1e-15)
    assert R.prior_from_odds_ratio(1e4) == pytest.approx(1.0 / 10001.0, rel=0, abs=1e-15)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "10", None])
def test_a_degenerate_odds_ratio_is_refused(bad):
    with pytest.raises(ValueError):
        R.prior_from_odds_ratio(bad)
    assert R.prior_from_odds_ratio(10.0) == pytest.approx(1.0 / 11.0)


def test_the_prd_band_is_the_quoted_range_and_it_discriminates():
    """Both endpoints of PRD.md:188's ``~10³–10⁴:1`` are inside; the neighbours are not.

    ADR-0005 D7's pinned ``100:1`` benchmark decoy prevalence and §9.1's ``~10:1`` training
    seed ratio must both fall **outside** — a band that admitted them would certify nothing,
    since it is precisely the distinction ADR-0005:80 draws.
    """
    assert R.DEPLOYMENT_PRIOR_ODDS_RANGE == (1e3, 1e4)
    assert R.DEPLOYMENT_PRIOR_RANGE == (1.0 / 10001.0, 1.0 / 1001.0)
    assert R.deployment_prior_in_prd_band(R.prior_from_odds_ratio(1e3)) is True
    assert R.deployment_prior_in_prd_band(R.prior_from_odds_ratio(1e4)) is True
    assert R.deployment_prior_in_prd_band(R.prior_from_odds_ratio(3e3)) is True
    assert R.deployment_prior_in_prd_band(R.prior_from_odds_ratio(100.0)) is False  # D7
    assert R.deployment_prior_in_prd_band(R.prior_from_odds_ratio(10.0)) is False  # §9.1
    assert R.deployment_prior_in_prd_band(R.prior_from_odds_ratio(1e5)) is False


# --------------------------------------------------------------------------- #
# The producer — both posteriors, the gated one named, no invented prior
# --------------------------------------------------------------------------- #
def test_the_named_posterior_is_the_pre_prior_shift_object_and_is_the_gated_one():
    """D11: GATE-2 grades the temperature-scaled posterior **before** the prior shift."""
    z = np.array([-2.0, 0.0, 1.5])
    temperature = 2.0
    payload = R.calibrated_posterior(
        z, temperature=temperature, source_prior=0.5, target_prior=1e-3
    )
    assert payload["gated_posterior_key"] == R.NAMED_POSTERIOR_KEY
    hand = [_sigmoid(v / temperature) for v in z.tolist()]
    np.testing.assert_allclose(payload[R.NAMED_POSTERIOR_KEY], hand, rtol=1e-12, atol=0.0)
    # ...and the gated object is *not* the shifted one, which is the confusion D11 exists for.
    assert not np.allclose(
        payload[payload["gated_posterior_key"]], payload[R.PRIOR_SHIFTED_POSTERIOR_KEY]
    )


def test_without_priors_the_stack_stops_at_the_named_posterior():
    z = np.array([-2.0, 0.0, 1.5])
    payload = R.calibrated_posterior(z, temperature=1.7)
    assert payload[R.PRIOR_SHIFTED_POSTERIOR_KEY] is None
    assert payload["prior_shift_applied"] is False
    assert payload["log_odds_shift"] is None
    assert payload["target_prior_in_prd_band"] is None
    assert payload["stack_applied"] == ["train", "temperature_scale"]
    assert payload["stack_order"] == list(T.STACK_ORDER)


def test_with_both_priors_the_full_pinned_stack_is_recorded_as_applied():
    z = np.array([-2.0, 0.0, 1.5])
    payload = R.calibrated_posterior(
        z, temperature=1.1, source_prior=0.75, target_prior=R.prior_from_odds_ratio(1e4)
    )
    assert payload["prior_shift_applied"] is True
    assert payload["stack_applied"] == list(T.STACK_ORDER)
    assert payload["log_odds_shift"] == pytest.approx(
        R.log_odds_shift(source_prior=0.75, target_prior=R.prior_from_odds_ratio(1e4))
    )
    assert payload["target_prior_in_prd_band"] is True


def test_the_payload_echoes_its_inputs_unswapped_and_carries_its_own_provenance():
    """The recorded priors and ``T`` must be the ones actually used, in the right slots.

    Without this, swapping the two prior assignments produces an internally contradictory
    artifact — ``source_prior`` holding the deployment prior, ``target_prior`` the calib
    prevalence, and ``target_prior_in_prd_band`` computed from the *real* argument so it
    still reads ``True`` — and the whole suite stays green. The provenance scalars are
    asserted for the same reason: a constant nothing reads is a constant nothing checks.
    """
    z = np.array([-2.0, 0.0, 1.5])
    p_s, p_t = 0.75, R.prior_from_odds_ratio(1e4)
    payload = R.calibrated_posterior(z, temperature=1.1, source_prior=p_s, target_prior=p_t)

    assert payload["source_prior"] == p_s
    assert payload["target_prior"] == p_t
    assert payload["source_prior"] != payload["target_prior"]  # asymmetric on purpose
    assert payload["temperature"] == 1.1
    assert payload["n_rows"] == z.shape[0] == 3
    # The shift is derived from the recorded pair, so a swap could not stay self-consistent.
    assert payload["log_odds_shift"] == pytest.approx(
        R.log_odds_shift(
            source_prior=payload["source_prior"], target_prior=payload["target_prior"]
        ),
        rel=0,
        abs=1e-15,
    )
    assert payload["step"] == "P3-07"
    assert payload["generated_by"] == "src/tbox_finder/calib/recalibrate.py"
    assert "D11" in payload["adr"]


@pytest.mark.parametrize("kwargs", [{"source_prior": 0.5}, {"target_prior": 0.001}])
def test_a_half_specified_prior_shift_raises_rather_than_defaulting_one(kwargs):
    """The module must be unable to invent a deployment prior (§10.3).

    The PRD gives the genome-scale prevalence only as prose (``~10³–10⁴:1``); a default here
    would put a fabricated quantity into every downstream posterior while looking like a
    convenience.
    """
    z = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="needs both source_prior and target_prior"):
        R.calibrated_posterior(z, temperature=1.0, **kwargs)
    # Positive control: supplying both is fine, supplying neither is fine.
    assert R.calibrated_posterior(z, temperature=1.0)["prior_shift_applied"] is False
    assert (
        R.calibrated_posterior(z, temperature=1.0, source_prior=0.5, target_prior=0.001)[
            "prior_shift_applied"
        ]
        is True
    )


def test_neither_prior_has_a_default():
    """Structural, not behavioural: ``prior_shift`` cannot be called without both priors.

    If someone later gives either one a default, this fails — which is the point: the
    deployment prior is the value that must never arrive by omission.
    """
    z = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(TypeError):
        R.prior_shift(z)
    with pytest.raises(TypeError):
        R.prior_shift(z, source_prior=0.5)
    with pytest.raises(TypeError):
        R.log_odds_shift(source_prior=0.5)


def test_the_stack_order_is_the_pinned_one_and_is_not_a_second_copy():
    """One ``STACK_ORDER``, imported — a second literal would drift from ADR-0005 D11."""
    assert R.STACK_ORDER is T.STACK_ORDER
    assert R.STACK_ORDER == ("train", "temperature_scale", "prior_shift")
