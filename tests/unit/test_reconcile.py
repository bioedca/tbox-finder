"""P2-03 — unit tests for the frozen overlapping-window logit-reconciliation operator.

The three properties that make the operator *the* seam-free reduction pinned by
ADR-0005 D3 + A3 each get a test **plus** an anti-tautology partner proving the test bites —
that a plausible wrong operator (bare log-sum-exp; raw-logit averaging; per-window
arg-max) actually fails it. A property test that no wrong implementation can fail is a
tautology, not a guard (P1-15 `lora_config_exact` / P2-01 RC-golden lesson).

Reference values are derived in stdlib ``math`` with explicit loops — never by calling
the module under test.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from tbox_finder import labels as labels_mod
from tbox_finder.data import window_dataset as wd
from tbox_finder.infer import reconcile as rc

# --------------------------------------------------------------------------------------
# deterministic logit helpers (stdlib-only; no numpy RNG stream dependency)
# --------------------------------------------------------------------------------------


def _logits(n_windows: int, window: int, *, tag: str = "t", scale: float = 4.0) -> np.ndarray:
    """Deterministic pseudo-random logits, reproducible across numpy/platform versions.

    SHAKE-256 (FIPS 202), not `numpy.random` — a `Generator` stream is not guaranteed
    stable across numpy releases, and these values back bit-exactness assertions.

    The draw must be **independent per class**. An earlier draft keyed a cheap polynomial
    hash on ``f"{tag}|{w}|{j}|{c}"``, where the class index is the last character, so the
    eight class logits at a position landed within ~1e-8 of each other. `log_softmax`
    annihilates a per-position constant exactly, so every position softmaxed to a uniform
    0.125 and the whole suite compared uniform distributions to uniform distributions —
    passing against operators that were arbitrarily wrong.
    `test_the_logit_fixture_is_expressive_not_uniform` fails if that ever recurs.
    """
    out = np.empty((n_windows, window, rc.NUM_CLASSES), dtype=np.float64)
    for w in range(n_windows):
        raw = hashlib.shake_256(f"{tag}|{w}".encode()).digest(window * rc.NUM_CLASSES * 8)
        for j in range(window):
            for c in range(rc.NUM_CLASSES):
                off = (j * rc.NUM_CLASSES + c) * 8
                unit = int.from_bytes(raw[off : off + 8], "big") / 2.0**64
                out[w, j, c] = scale * (2.0 * unit - 1.0)
    return out


def _stdlib_log_softmax(row: list[float]) -> list[float]:
    m = max(row)
    total = sum(math.exp(v - m) for v in row)
    return [v - m - math.log(total) for v in row]


def _reference_reconcile(
    logits: np.ndarray, starts: list[int], seq_len: int
) -> tuple[list[list[float]], list[int], list[int], list[bool]]:
    """Independent stdlib re-derivation: log(mean of per-window posteriors), then argmax.

    Deliberately shares nothing with :mod:`tbox_finder.infer.reconcile` — no numpy
    reduction, no shared helper. Plain per-position loops over Python floats.
    """
    window = logits.shape[1]
    log_probs: list[list[float]] = []
    coverage: list[int] = []
    flagged: list[bool] = []
    for p in range(seq_len):
        probs = [0.0] * rc.NUM_CLASSES
        n = 0
        flag = False
        for k, start in enumerate(starts):
            if not (start <= p < start + window):
                continue
            n += 1
            if start < 0 or start + window > seq_len:
                flag = True
            row = [float(v) for v in logits[k, p - start]]
            for c, lp in enumerate(_stdlib_log_softmax(row)):
                probs[c] += math.exp(lp)
        log_probs.append([math.log(v / n) for v in probs])
        coverage.append(n)
        flagged.append(flag)
    prediction = [max(range(rc.NUM_CLASSES), key=lambda c: row[c]) for row in log_probs]
    return log_probs, prediction, coverage, flagged


# --------------------------------------------------------------------------------------
# the fixture must be able to express what the tests claim to measure
# --------------------------------------------------------------------------------------


def test_the_logit_fixture_is_expressive_not_uniform() -> None:
    """Guard the whole suite: near-equal class logits make every assertion below vacuous.

    `log_softmax` removes any per-position constant exactly, so a generator whose eight
    class logits at a position are nearly equal yields a uniform posterior everywhere.
    Comparisons of uniform-to-uniform pass against an operator that is arbitrarily wrong,
    including one that misaligns whole windows. This test pins the property that makes the
    rest of the file mean something.
    """
    logits = _logits(3, 64, tag="expressive")
    per_position_spread = logits.max(axis=2) - logits.min(axis=2)
    assert float(per_position_spread.min()) > 1.0, "classes are near-degenerate at some position"
    assert float(per_position_spread.mean()) > 4.0

    probs = np.exp(rc.log_softmax(logits, axis=-1))
    assert float(probs.max()) > 0.75, "no position can express a confident call"
    assert float(probs.min()) < 0.01, "no position can express a confidently-rejected class"
    # the argmax must actually vary across classes, not sit on one index
    assert len(set(np.argmax(logits, axis=2).ravel().tolist())) >= rc.NUM_CLASSES - 1


def test_the_logit_fixture_is_deterministic_and_varies_by_key() -> None:
    assert np.array_equal(_logits(2, 8, tag="k"), _logits(2, 8, tag="k"))
    assert not np.array_equal(_logits(2, 8, tag="k"), _logits(2, 8, tag="j"))
    a = _logits(2, 8, tag="k")
    assert not np.array_equal(a[0], a[1]), "windows must differ, or coverage tests are vacuous"


# --------------------------------------------------------------------------------------
# frozen constants / drift guards
# --------------------------------------------------------------------------------------


def test_num_classes_tracks_the_canonical_label_vocabulary() -> None:
    assert rc.NUM_CLASSES == len(labels_mod.CLASS_ORDER) == 8


def test_tiling_constants_do_not_drift_from_the_datamodule() -> None:
    """The reconciler and the datamodule must agree on the PRD §6 pinned tiling."""
    assert rc.WINDOW_NT == wd.WINDOW_NT == 1_024
    assert rc.STRIDE_NT == wd.STRIDE_NT == 512
    assert rc.STRIDE_NT < rc.WINDOW_NT  # stride < window => interior coverage >= 2


def test_operator_identity_is_recorded_as_pinned() -> None:
    """Pinned against EXTERNAL literals, not against the constants that produce them.

    `assert d["operator"] == rc.OPERATOR` would be a tautology — the dict is built from
    that constant, so it passes at any value, including one that no longer describes the
    operator (the P1-15 `lora_config_exact` class of defect). The literals below are
    written out so that changing the operator's identity forces this test to be edited.
    """
    d = rc.diagnostics()
    assert d["operator"] == "log_mean_exp_of_per_window_log_softmax_then_argmax"
    assert d["pinned"] is True
    assert d["pinned_by"] == "ADR-0005 D3 + A3 (locus-construction rule); PRD §6, §13.1"
    assert d["class_order"] == [
        "background",
        "Stem_I",
        "Specifier",
        "Stem_II",
        "Stem_III",
        "Antiterminator_Tbox_seq",
        "Terminator",
        "Discriminator",
    ]
    assert d["num_classes"] == 8
    assert d["window_nt"] == 1_024 and d["stride_nt"] == 512
    assert d["normalises_per_window"] is True
    assert d["coverage_normalised"] is True
    assert d["ignores_pad_positions"] is True
    # ...and the reported vocabulary must still be the repo's canonical one.
    assert d["class_order"] == list(labels_mod.CLASS_ORDER)


def test_diagnostics_claims_match_what_the_operator_actually_does() -> None:
    """The three behavioural claims in `diagnostics()` are re-derived, not echoed.

    A flag that says `coverage_normalised: True` is worth nothing unless the operator is
    actually coverage-normalised (P1-16: an all-true gate is self-consistent whatever the
    evidence says). Each claim below is verified against measured behaviour.
    """
    d = rc.diagnostics()
    window = 16
    single = _logits(1, window, tag="diag")

    # coverage_normalised: duplicating an identical window must not move the score
    base = rc.reconcile_windows(single, [0], window)
    doubled = rc.reconcile_windows(np.repeat(single, 2, axis=0), [0, 0], window)
    assert d["coverage_normalised"] is np.array_equal(doubled.log_probs, base.log_probs)

    # normalises_per_window: a constant shift of one window must not move the score
    pair = _logits(2, window, tag="diag2")
    shifted = pair.copy()
    shifted[1] += 3.0
    ref = rc.reconcile_windows(pair, [0, 8], window + 8)
    got = rc.reconcile_windows(shifted, [0, 8], window + 8)
    assert d["normalises_per_window"] is bool(np.allclose(ref.log_probs, got.log_probs, atol=1e-12))

    # ignores_pad_positions: poisoning an out-of-bounds position must not move the score
    poisoned = single.copy()
    poisoned[0, window - 4 :] = np.nan
    clean = rc.reconcile_windows(single, [0], window - 4)
    dirty = rc.reconcile_windows(poisoned, [0], window - 4)
    assert d["ignores_pad_positions"] is np.array_equal(clean.log_probs, dirty.log_probs)


# --------------------------------------------------------------------------------------
# log-sum-exp / log-softmax primitives
# --------------------------------------------------------------------------------------


def test_logsumexp_matches_stdlib_on_ordinary_values() -> None:
    x = np.array([0.5, -2.0, 3.25, 1.0])
    assert rc.logsumexp(x) == pytest.approx(math.log(sum(math.exp(v) for v in x)), abs=1e-12)


def test_logsumexp_is_stable_where_the_naive_form_overflows() -> None:
    x = np.array([1000.0, 1001.0, 999.0])
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(x)).all()  # the naive form really does overflow here
    expected = 1001.0 + math.log(math.exp(-1.0) + 1.0 + math.exp(-2.0))
    assert rc.logsumexp(x) == pytest.approx(expected, abs=1e-9)


def test_logsumexp_is_stable_where_the_naive_form_underflows() -> None:
    x = np.array([-1000.0, -1001.0])
    assert (np.exp(x) == 0.0).all()  # naive: log(0) = -inf
    assert rc.logsumexp(x) == pytest.approx(-1000.0 + math.log(1.0 + math.exp(-1.0)), abs=1e-9)


def test_logsumexp_returns_the_correct_limit_on_degenerate_rows() -> None:
    """The naive shift gives `inf - inf = nan` here; the limits are well defined."""
    assert rc.logsumexp(np.array([-np.inf, -np.inf])) == -np.inf
    assert rc.logsumexp(np.array([np.inf, 0.0])) == np.inf
    assert rc.logsumexp(np.array([-np.inf, 0.0])) == pytest.approx(0.0, abs=1e-12)
    rows = rc.logsumexp(np.array([[-np.inf, -np.inf], [0.0, 0.0]]), axis=1)
    assert rows[0] == -np.inf and rows[1] == pytest.approx(math.log(2.0), abs=1e-12)


def test_reconcile_refuses_to_emit_a_non_finite_reconciliation() -> None:
    """`np.argmax` picks NaN as the maximum, so a NaN silently *selects a class*.

    Before the guard, |logit| ~ 1e308 overflowed the log-softmax shift to -inf, `-inf -
    -inf` produced NaN, and the operator returned class 1 for a position whose true argmax
    was class 0 — a wrong call indistinguishable from a real one.
    """
    logits = np.zeros((1, 2, rc.NUM_CLASSES))
    logits[0, 0, 0] = 1e308
    logits[0, 0, 1] = -1e308
    assert np.isfinite(logits).all()  # passes the input guard: these ARE finite
    with pytest.raises(ValueError, match="non-finite log-probability"):
        rc.reconcile_windows(logits, [0], 2)


def test_ordinary_extreme_logits_still_reconcile() -> None:
    """Anti-tautology partner: the guard must not reject logits a real head can emit."""
    logits = np.zeros((1, 2, rc.NUM_CLASSES))
    logits[0, 0, 0] = 500.0  # far beyond any trained head, still reducible
    logits[0, 0, 1] = -500.0
    out = rc.reconcile_windows(logits, [0], 2)
    assert np.isfinite(out.log_probs).all()
    assert int(out.prediction[0]) == 0


def test_logsumexp_keepdims_and_axis() -> None:
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert rc.logsumexp(x, axis=1, keepdims=True).shape == (3, 1)
    assert rc.logsumexp(x, axis=1).shape == (3,)
    assert rc.logsumexp(x, axis=0).shape == (4,)


def test_log_softmax_normalises_and_is_shift_invariant() -> None:
    x = np.array([[1.0, -3.0, 0.5, 2.0, 0.0, -1.0, 4.0, 0.25]])
    lp = rc.log_softmax(x)
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rc.log_softmax(x + 17.0), lp, atol=1e-12)
    ref = _stdlib_log_softmax(list(x[0]))
    np.testing.assert_allclose(lp[0], ref, atol=1e-12)


# --------------------------------------------------------------------------------------
# property 1 — seam-free: reconciled == single-window on non-overlapping input
# --------------------------------------------------------------------------------------


def test_seam_free_non_overlapping_input_reduces_to_the_single_window() -> None:
    """imp.md gate: reconciled == single-window prediction when nothing overlaps."""
    window, n = 32, 3
    logits = _logits(n, window, tag="seamfree")
    starts = [0, 32, 64]
    out = rc.reconcile_windows(logits, starts, window * n)

    expected = rc.log_softmax(logits, axis=-1).reshape(window * n, rc.NUM_CLASSES)
    # bit-exact, not merely close: at coverage 1 the reduction must be the identity.
    assert np.array_equal(out.log_probs, expected)
    assert np.array_equal(out.coverage, np.ones(window * n, dtype=np.int32))
    assert np.array_equal(out.prediction, np.argmax(expected, axis=1).astype(np.int16))


def test_seam_free_holds_at_the_real_1024_nt_window_width() -> None:
    """Same property at the pinned 1024-nt width. (The pinned *stride* is 512 and never
    produces coverage 1 in the interior — this case abuts two windows to isolate the
    coverage-1 identity at production width.)"""
    logits = _logits(2, rc.WINDOW_NT, tag="realseam")
    out = rc.reconcile_windows(logits, [0, rc.WINDOW_NT], 2 * rc.WINDOW_NT)
    assert set(out.coverage.tolist()) == {1}
    assert np.array_equal(
        out.log_probs, rc.log_softmax(logits, axis=-1).reshape(2 * rc.WINDOW_NT, 8)
    )


# --------------------------------------------------------------------------------------
# property 2 — coverage-count invariance (the reason boundary IoU is not a grid artifact)
# --------------------------------------------------------------------------------------


def test_coverage_invariance_identical_windows_reconcile_to_the_single_window_value() -> None:
    """A position covered n times by *agreeing* windows must score as if covered once.

    This is the ``- log |W(p)|`` normalisation. Without it the reconciled logits grow
    with coverage, so the 512-grid seam (coverage 1 at the tail vs 2+ in the interior)
    would show up in every downstream threshold — the artifact PRD §6 exists to remove.
    """
    window = 24
    single = _logits(1, window, tag="cov")
    base = rc.reconcile_windows(single, [0], window)
    for n in (2, 3, 5):
        stacked = np.repeat(single, n, axis=0)
        out = rc.reconcile_windows(stacked, [0] * n, window)
        assert np.array_equal(out.coverage, np.full(window, n, dtype=np.int32))
        assert np.array_equal(out.log_probs, base.log_probs), f"coverage {n} shifted the score"


def test_coverage_invariance_bites_a_bare_logsumexp_operator() -> None:
    """Anti-tautology: the un-normalised reduction really does fail the test above."""
    window = 24
    single = _logits(1, window, tag="cov")
    lp = rc.log_softmax(single, axis=-1)
    bare_cov1 = rc.logsumexp(lp[:1], axis=0)  # log-sum-exp WITHOUT / coverage
    bare_cov2 = rc.logsumexp(np.repeat(lp, 2, axis=0), axis=0)
    assert not np.allclose(bare_cov1, bare_cov2)
    np.testing.assert_allclose(bare_cov2 - bare_cov1, math.log(2.0), atol=1e-12)


def test_agreeing_windows_leave_no_seam_across_a_real_coverage_boundary() -> None:
    """Two overlapping windows that agree => the overlap scores like the non-overlap."""
    window, seq_len = 64, 96
    per_position = _logits(1, seq_len, tag="agree")[0]  # one logit vector per position
    logits = np.stack([per_position[s : s + window] for s in (0, 32)])
    out = rc.reconcile_windows(logits, [0, 32], seq_len)
    assert sorted(set(out.coverage.tolist())) == [1, 2]
    expected = rc.log_softmax(per_position, axis=-1)
    # every position — coverage 1 and coverage 2 alike — recovers the agreed distribution
    assert np.array_equal(out.log_probs, expected)


# --------------------------------------------------------------------------------------
# property 3 — order invariance / associativity of the LSE reduction
# --------------------------------------------------------------------------------------


def test_reduction_is_bit_exactly_invariant_to_window_order() -> None:
    seq_len = 2556  # a real corpus context length; tail-anchored tiling, coverage 1..3
    starts = wd.tile_windows(seq_len)
    logits = _logits(len(starts), rc.WINDOW_NT, tag="order")
    ref = rc.reconcile_windows(logits, starts, seq_len)

    for perm in ([3, 1, 0, 2], [2, 3, 1, 0], list(reversed(range(len(starts))))):
        out = rc.reconcile_windows(logits[perm], [starts[i] for i in perm], seq_len)
        assert np.array_equal(out.log_probs, ref.log_probs)
        assert np.array_equal(out.prediction, ref.prediction)
        assert np.array_equal(out.coverage, ref.coverage)


def test_reduction_is_deterministic_across_repeated_calls() -> None:
    starts = wd.tile_windows(1600)
    logits = _logits(len(starts), rc.WINDOW_NT, tag="det")
    a = rc.reconcile_windows(logits, starts, 1600)
    b = rc.reconcile_windows(logits, starts, 1600)
    assert np.array_equal(a.log_probs, b.log_probs)
    assert np.array_equal(a.prediction, b.prediction)


# --------------------------------------------------------------------------------------
# per-window scale invariance (what the per-window log-softmax buys)
# --------------------------------------------------------------------------------------


def test_a_constant_shift_of_one_window_does_not_move_the_reconciliation() -> None:
    """A softmax discards a per-window constant, so the ensemble must too.

    NOT vacuous by softmax shift-invariance: the shift is applied to ONE of two windows
    *before* they are averaged. The partner test below shows a raw-logit reduction —
    the plausible alternative reading of the D3 rule — is moved by exactly this shift.
    """
    window, seq_len = 48, 72
    logits = _logits(2, window, tag="shift")
    starts = [0, 24]
    ref = rc.reconcile_windows(logits, starts, seq_len)

    shifted = logits.copy()
    shifted[1] += 6.25  # every class of window 1: same predicted distribution
    out = rc.reconcile_windows(shifted, starts, seq_len)
    np.testing.assert_allclose(out.log_probs, ref.log_probs, atol=1e-12)
    assert np.array_equal(out.prediction, ref.prediction)


def test_shift_invariance_bites_a_raw_logit_averaging_operator() -> None:
    """Anti-tautology: averaging raw logits in exp-space IS moved by the same shift."""
    window = 48
    logits = _logits(2, window, tag="shift")
    overlap = slice(24, 48)

    def raw_lse_mean(arr: np.ndarray) -> np.ndarray:
        stacked = np.stack([arr[0, overlap], arr[1, 0:24]])
        return rc.logsumexp(stacked, axis=0) - math.log(2.0)

    shifted = logits.copy()
    shifted[1] += 6.25
    before, after = raw_lse_mean(logits), raw_lse_mean(shifted)
    assert not np.allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------------------
# the output is a proper distribution / prediction consistency
# --------------------------------------------------------------------------------------


def test_reconciled_log_probs_are_a_normalised_distribution() -> None:
    starts = wd.tile_windows(1800)
    logits = _logits(len(starts), rc.WINDOW_NT, tag="dist")
    out = rc.reconcile_windows(logits, starts, 1800)
    np.testing.assert_allclose(np.exp(out.log_probs).sum(axis=1), 1.0, atol=1e-12)
    assert (out.log_probs <= 0.0).all()


def test_prediction_is_the_argmax_of_the_reconciled_log_probs() -> None:
    starts = wd.tile_windows(1500)
    logits = _logits(len(starts), rc.WINDOW_NT, tag="pred")
    out = rc.reconcile_windows(logits, starts, 1500)
    assert np.array_equal(out.prediction, np.argmax(out.log_probs, axis=1).astype(np.int16))
    assert out.prediction.dtype == np.int16
    assert out.prediction.min() >= 0 and out.prediction.max() < rc.NUM_CLASSES


def test_argmax_ties_resolve_to_background() -> None:
    """Tie-break is documented as numpy's first-index rule; index 0 is `background`."""
    assert labels_mod.CLASS_ORDER[0] == "background"
    flat = np.zeros((1, 4, rc.NUM_CLASSES), dtype=np.float64)
    out = rc.reconcile_windows(flat, [0], 4)
    assert np.array_equal(out.prediction, np.zeros(4, dtype=np.int16))


def test_matches_an_independent_stdlib_reimplementation() -> None:
    """The whole operator, re-derived in stdlib `math` with explicit per-position loops."""
    window, stride, seq_len = 40, 16, 100
    starts = wd.tile_windows(seq_len, window=window, stride=stride)
    logits = _logits(len(starts), window, tag="indep")
    out = rc.reconcile_windows(logits, starts, seq_len)

    ref_lp, ref_pred, ref_cov, ref_flag = _reference_reconcile(logits, starts, seq_len)
    np.testing.assert_allclose(out.log_probs, np.asarray(ref_lp), atol=1e-12)
    assert out.prediction.tolist() == ref_pred
    assert out.coverage.tolist() == ref_cov
    assert out.zero_flanked.tolist() == ref_flag


@pytest.mark.parametrize(
    ("starts", "seq_len"),
    [
        ([-8, 8], 40),  # left overhang only
        ([-8, 16], 40),  # both ends overhang
        ([-31, -16, 0, 16], 48),  # deep left overhang; window almost entirely off-contig
        ([0, 24], 50),  # right overhang only
    ],
)
def test_overhanging_windows_reconcile_to_the_independent_values_not_just_the_flag(
    starts: list[int], seq_len: int
) -> None:
    """Numeric coverage of the zero-flank slice arithmetic — NOT only `zero_flanked`.

    `logits[k, lo - start : hi - start]` has a non-zero left offset ONLY when `start < 0`.
    `tile_windows` never emits a negative start, so nothing else in this suite exercises
    that offset: an operator that dropped it (reading `logits[k, 0 : hi - lo]`, i.e.
    silently misaligning every left-overhanging window by `-start` nucleotides) passed all
    48 of these tests before this case existed — verified by sabotage, not by reading. The
    stdlib reference derives the same mapping independently, so it pins the values.
    """
    window = 32
    logits = _logits(len(starts), window, tag=f"ovh{starts[0]}{seq_len}")
    out = rc.reconcile_windows(logits, starts, seq_len)

    ref_lp, ref_pred, ref_cov, ref_flag = _reference_reconcile(logits, starts, seq_len)
    np.testing.assert_allclose(out.log_probs, np.asarray(ref_lp), atol=1e-12)
    assert out.prediction.tolist() == ref_pred
    assert out.coverage.tolist() == ref_cov
    assert out.zero_flanked.tolist() == ref_flag
    assert any(ref_flag), "this case must actually zero-flank something, or it guards nothing"


def test_the_left_overhang_offset_is_load_bearing() -> None:
    """Anti-tautology partner: prove the case above BITES the exact sabotage it targets.

    Re-implements the operator with the one defect at issue — reading an overhanging
    window from index 0 rather than from ``-start`` — and asserts the stdlib reference
    (which the test above compares against) rejects it. Without this, the test above could
    pass against a broken operator if the reference shared the same misreading.
    """
    window, seq_len, start = 32, 40, -8
    starts = [start, 8]
    logits = _logits(2, window, tag="loadbearing")
    ref_lp, _, _, _ = _reference_reconcile(logits, starts, seq_len)

    # the sabotaged reduction: every overhanging window read from its index 0
    sabotaged = np.full((seq_len, rc.NUM_CLASSES), 0.0)
    acc = np.full((seq_len, rc.NUM_CLASSES), -np.inf)
    cov = np.zeros(seq_len, dtype=np.int64)
    pieces = []
    for k, s in enumerate(starts):
        lo, hi = max(s, 0), min(s + window, seq_len)
        piece = rc.log_softmax(logits[k, 0 : hi - lo], axis=-1)  # SABOTAGE: no `- start`
        pieces.append((lo, hi, piece))
        np.maximum(acc[lo:hi], piece, out=acc[lo:hi])
        cov[lo:hi] += 1
    total = np.zeros((seq_len, rc.NUM_CLASSES))
    for lo, hi, piece in pieces:
        total[lo:hi] += np.exp(piece - acc[lo:hi])
    sabotaged = acc + np.log(total / cov[:, None])

    assert not np.allclose(sabotaged, np.asarray(ref_lp), atol=1e-6), (
        "the reference does not distinguish the misaligned read — the overhang test above "
        "would pass against a broken operator"
    )


# --------------------------------------------------------------------------------------
# contig ends: zero-flank + flag, and pad logits never enter the average
# --------------------------------------------------------------------------------------


def test_short_contig_is_zero_flanked_and_every_position_flagged() -> None:
    """`tile_windows` returns one overhanging window when seq_len <= window (PRD §6)."""
    seq_len = 366  # the shortest real context in the P2-00 corpus sample
    starts = wd.tile_windows(seq_len)
    assert starts == [0]
    logits = _logits(1, rc.WINDOW_NT, tag="short")
    out = rc.reconcile_windows(logits, starts, seq_len)
    assert out.log_probs.shape == (seq_len, rc.NUM_CLASSES)
    assert bool(out.zero_flanked.all())
    assert np.array_equal(out.coverage, np.ones(seq_len, dtype=np.int32))


def test_tail_anchored_tiling_of_a_long_contig_flags_nothing() -> None:
    """A sequence longer than the window is tiled entirely in-bounds => no zero-flank."""
    for seq_len in (1025, 1536, 2048, 2556):
        starts = wd.tile_windows(seq_len)
        assert starts[-1] + rc.WINDOW_NT == seq_len  # tail-anchored: no overhang
        logits = _logits(len(starts), rc.WINDOW_NT, tag=f"long{seq_len}")
        out = rc.reconcile_windows(logits, starts, seq_len)
        assert not bool(out.zero_flanked.any()), seq_len
        assert int(out.coverage.min()) >= 1


def test_only_the_positions_of_an_overhanging_window_are_flagged() -> None:
    window, seq_len = 32, 50
    logits = _logits(2, window, tag="ovh")
    out = rc.reconcile_windows(logits, [0, 24], seq_len)  # window 1 spans [24, 56) -> pad
    assert not out.zero_flanked[:24].any()
    assert out.zero_flanked[24:].all()


def test_a_window_starting_before_the_contig_is_flagged_too() -> None:
    """A left overhang flags exactly its own span — the flag is not smeared contig-wide."""
    window, seq_len = 32, 40
    logits = _logits(2, window, tag="neg")
    # window 0 spans [-8, 24) -> zero-padded on the left; window 1 spans [8, 40) -> exact.
    out = rc.reconcile_windows(logits, [-8, 8], seq_len)
    assert out.zero_flanked[:24].all()
    assert not out.zero_flanked[24:].any()
    assert int(out.coverage.max()) == 2  # [8, 24) is covered by both


def test_both_overhangs_together_flag_the_whole_contig() -> None:
    window, seq_len = 32, 40
    logits = _logits(2, window, tag="neg2")
    # window 0 spans [-8, 24); window 1 spans [16, 48) -> the pair overhangs both ends.
    out = rc.reconcile_windows(logits, [-8, 16], seq_len)
    assert bool(out.zero_flanked.all())


def test_pad_position_logits_never_enter_the_average() -> None:
    """Logits emitted over synthetic zero-pad describe no DNA and must be ignored."""
    seq_len = 300
    starts = wd.tile_windows(seq_len)
    logits = _logits(1, rc.WINDOW_NT, tag="pad")
    ref = rc.reconcile_windows(logits, starts, seq_len)

    for poison in (np.nan, np.inf, -np.inf, 1e300):
        polluted = logits.copy()
        polluted[0, seq_len:] = poison
        out = rc.reconcile_windows(polluted, starts, seq_len)
        assert np.array_equal(out.log_probs, ref.log_probs), poison
        assert np.isfinite(out.log_probs).all()


def test_pad_positions_are_actually_present_in_that_fixture() -> None:
    """Guard the test above from silently testing nothing if the geometry changes."""
    assert wd.WINDOW_NT > 300


# --------------------------------------------------------------------------------------
# fail-closed validation — every clause proven to bite
# --------------------------------------------------------------------------------------


def test_rejects_non_3d_logits() -> None:
    with pytest.raises(ValueError, match="ndim"):
        rc.reconcile_windows(np.zeros((4, rc.NUM_CLASSES)), [0], 4)


def test_rejects_a_wrong_class_count() -> None:
    with pytest.raises(ValueError, match="8 classes"):
        rc.reconcile_windows(np.zeros((1, 4, 7)), [0], 4)


def test_rejects_an_empty_window_stack() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        rc.reconcile_windows(np.zeros((0, 4, rc.NUM_CLASSES)), [], 4)


def test_rejects_a_starts_length_mismatch() -> None:
    with pytest.raises(ValueError, match="offsets but window_logits"):
        rc.reconcile_windows(np.zeros((2, 4, rc.NUM_CLASSES)), [0], 8)


@pytest.mark.parametrize("bad", [0, -5])
def test_rejects_a_non_positive_seq_len(bad: int) -> None:
    with pytest.raises(ValueError, match="seq_len must be positive"):
        rc.reconcile_windows(np.zeros((1, 4, rc.NUM_CLASSES)), [0], bad)


@pytest.mark.parametrize("bad", [True, False, 1.5, "4", None])
def test_rejects_a_non_integer_seq_len(bad: object) -> None:
    with pytest.raises(ValueError, match="seq_len must be"):
        rc.reconcile_windows(np.zeros((1, 4, rc.NUM_CLASSES)), [0], bad)


def test_accepts_numpy_integer_seq_len_and_starts() -> None:
    """`operator.index` path: np.int64 is not an `int` to `isinstance` (P2-02 lesson)."""
    logits = _logits(1, 4, tag="npint")
    out = rc.reconcile_windows(logits, np.asarray([np.int64(0)]), np.int64(4))
    assert out.seq_len == 4 and out.n_windows == 1


def test_rejects_a_boolean_start_offset() -> None:
    with pytest.raises(ValueError, match="starts must be an integer, got bool"):
        rc.reconcile_windows(np.zeros((1, 4, rc.NUM_CLASSES)), [True], 4)


def test_rejects_a_boolean_start_mixed_with_ints() -> None:
    """`np.asarray([True, 5])` promotes to int64, silently turning the bool into 1.

    A numpy round-trip on the caller's list would defeat the guard above without any test
    noticing, because the pure-`[True]` case coincidentally survives promotion as bool.
    """
    with pytest.raises(ValueError, match="starts must be an integer, got bool"):
        rc.reconcile_windows(np.zeros((2, 4, rc.NUM_CLASSES)), [True, 4], 8)
    with pytest.raises(ValueError, match="starts must be an integer, got bool"):
        rc.reconcile_windows(np.zeros((2, 4, rc.NUM_CLASSES)), [0, False], 8)


def test_rejects_a_boolean_dtype_start_array() -> None:
    with pytest.raises(ValueError, match="starts must be an integer, got bool"):
        rc.reconcile_windows(np.zeros((1, 4, rc.NUM_CLASSES)), np.array([False]), 4)


def test_rejects_a_zero_width_window() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        rc.reconcile_windows(np.zeros((1, 0, rc.NUM_CLASSES)), [0], 4)


def test_rejects_a_non_finite_in_bounds_logit() -> None:
    logits = _logits(1, 4, tag="nan")
    logits[0, 2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite logit"):
        rc.reconcile_windows(logits, [0], 4)


def test_rejects_a_window_entirely_outside_the_sequence() -> None:
    logits = _logits(2, 4, tag="oob")
    with pytest.raises(ValueError, match="covers no position"):
        rc.reconcile_windows(logits, [0, 40], 4)


def test_rejects_an_uncovered_position_rather_than_calling_it_background() -> None:
    """A tiling gap is an error: an unscored nucleotide must never become a silent call."""
    logits = _logits(2, 4, tag="gap")
    with pytest.raises(ValueError, match="covered by no window"):
        rc.reconcile_windows(logits, [0, 8], 12)


def test_the_gap_check_does_not_fire_on_a_valid_tiling() -> None:
    """Anti-tautology partner: the same geometry, gap closed, reconciles fine."""
    logits = _logits(3, 4, tag="gap")
    out = rc.reconcile_windows(logits, [0, 4, 8], 12)
    assert int(out.coverage.min()) == 1


# --------------------------------------------------------------------------------------
# array-like handoff (the scan feeds torch logits; the module stays torch-free)
# --------------------------------------------------------------------------------------


def test_accepts_a_plain_nested_list() -> None:
    logits = _logits(1, 3, tag="list").tolist()
    out = rc.reconcile_windows(logits, [0], 3)
    assert out.log_probs.shape == (3, rc.NUM_CLASSES)


def test_accepts_a_float32_array_without_losing_the_reduction() -> None:
    logits = _logits(2, 16, tag="f32")
    starts, seq_len = [0, 8], 24
    out32 = rc.reconcile_windows(logits.astype(np.float32), starts, seq_len)
    out64 = rc.reconcile_windows(logits, starts, seq_len)
    np.testing.assert_allclose(out32.log_probs, out64.log_probs, atol=1e-6)
    assert out32.log_probs.dtype == np.float64  # promoted, not silently truncated


def test_accepts_a_cpu_torch_tensor() -> None:
    """The scan produces torch logits; `numpy.asarray` must bridge without a torch import.

    torch is referenced only inside the test body — a torch symbol in a decorator would
    be evaluated at *collection* time and break the bare CI tier (P2-02 lesson).
    """
    torch = pytest.importorskip("torch")
    logits = _logits(2, 16, tag="torch")
    starts, seq_len = [0, 8], 24
    ref = rc.reconcile_windows(logits, starts, seq_len)
    out = rc.reconcile_windows(torch.from_numpy(logits), starts, seq_len)
    assert np.array_equal(out.log_probs, ref.log_probs)
    assert np.array_equal(out.prediction, ref.prediction)


# --------------------------------------------------------------------------------------
# the operator vs the naive alternative it replaces
# --------------------------------------------------------------------------------------


def test_reconciliation_can_overturn_a_per_window_argmax() -> None:
    """The operator is not cosmetic: it must be able to change the call at a seam.

    A window that sees an element in truncated context can call it confidently wrong;
    the covering windows outvote it. If this never happened, reconciliation would buy
    nothing and the GATE-4 boundary IoU really would be a 512-grid artifact.
    """
    window, seq_len = 8, 12
    logits = np.zeros((2, window, rc.NUM_CLASSES), dtype=np.float64)
    # window 0 (spans [0,8)) weakly prefers class 1 in the overlap [4,8)
    logits[0, :, 1] = 0.5
    # window 1 (spans [4,12)) strongly prefers class 2 over its whole span
    logits[1, :, 2] = 4.0
    out = rc.reconcile_windows(logits, [0, 4], seq_len)

    per_window_argmax_w0 = int(np.argmax(logits[0, 4]))
    assert per_window_argmax_w0 == 1  # window 0, alone, would call class 1 at position 4
    assert int(out.prediction[4]) == 2  # reconciled: window 1's confidence wins
    assert int(out.prediction[0]) == 1  # outside the overlap window 0 still stands
