"""P2-10a — the Stage-1 scanner's torch-free surface: encoding, geometry, pad honesty.

Everything here runs in the bare CI tier. The forward pass itself is `tests/ml/`; what is
testable without torch is exactly the part where a scanner can lie — the padding convention,
the tiling geometry, and whether a contig end is flagged as one.

The closed-form test at the bottom is the one that matters. It does **not** assert "the
scanner agrees with `reconcile_windows`" — that would be a tautology, since the scanner's
whole job is to call it. It pins the reconciled posterior against a value computed here from
first principles, so a wrong `starts` vector or a coverage-weighting bug has somewhere to
show up.

It does **not** cover a pad leak, and its own docstring says so: pads here are always
out-of-bounds, and `reconcile_windows` slices those away before any arithmetic (measured —
sabotaging `PAD_TOKEN_ID` to the `N` id failed three other tests in this file and left every
parametrisation of the closed form green). The `seq_len` wiring that could actually let a pad
into the reduction lives in `scan_sequence`, which this tier cannot call — no torch in bare
CI — so it is covered in `tests/ml/test_scan_checkpoint.py` instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbox_finder.data.window_dataset import (
    BASE_TO_ID,
    PAD_TOKEN_ID,
    STRIDE_NT,
    WINDOW_NT,
    encode_bases,
    tile_windows,
)
from tbox_finder.infer import scan as scan_mod
from tbox_finder.infer.reconcile import NUM_CLASSES, reconcile_windows
from tbox_finder.infer.scan import (
    ScanError,
    encode_scan_window,
    scan_window_ids,
)


def _seq(n: int, *, seed: int = 0) -> str:
    """A deterministic ACGT string of length ``n`` (no numpy RNG dependence in the pin)."""
    rng = np.random.default_rng(seed)
    return "".join("ACGT"[i] for i in rng.integers(0, 4, size=n))


# ═════════════════════════════════════════════════════════════════════════════
# Padding convention — the scanner must not invent a token the model never saw
# ═════════════════════════════════════════════════════════════════════════════
def test_pad_token_matches_training_carve_window_convention():
    """The scan pad must be the SAME id training padded with, not merely 'some pad'.

    `carve_window` fills the window with `PAD_TOKEN_ID` before writing real bases into it,
    and the backbone's own `config.pad_token_id` is 4. A scanner padding with, say, the `N`
    id would be presenting the model a convention it never saw at train time — and the
    result would still look perfectly well-formed.
    """
    assert PAD_TOKEN_ID == 4
    assert PAD_TOKEN_ID not in BASE_TO_ID.values()

    short = _seq(100)
    ids = encode_scan_window(short, 0, window=WINDOW_NT)
    assert ids.shape == (WINDOW_NT,)
    np.testing.assert_array_equal(ids[:100], encode_bases(short))
    assert np.all(ids[100:] == PAD_TOKEN_ID)


def test_window_fully_inside_sequence_emits_no_pad():
    """An interior window is all real DNA — a pad here would be a fabricated boundary."""
    seq = _seq(4000)
    ids = encode_scan_window(seq, 1024, window=WINDOW_NT)
    assert not np.any(ids == PAD_TOKEN_ID)
    np.testing.assert_array_equal(ids, encode_bases(seq[1024 : 1024 + WINDOW_NT]))


def test_negative_start_pads_on_the_left_not_the_right():
    """A 5' overrun pads the HEAD of the window. Padding the tail instead would shift every
    real base off its true offset while keeping the pad count identical — an error the
    length check alone cannot see."""
    seq = _seq(300)
    ids = encode_scan_window(seq, -50, window=200)
    assert np.all(ids[:50] == PAD_TOKEN_ID)
    np.testing.assert_array_equal(ids[50:], encode_bases(seq[:150]))


def test_window_covering_no_real_position_is_refused():
    """An all-pad window scores no DNA; it must not silently contribute a distribution."""
    seq = _seq(100)
    with pytest.raises(ScanError, match="covers no position"):
        encode_scan_window(seq, 500, window=WINDOW_NT)
    with pytest.raises(ScanError, match="covers no position"):
        encode_scan_window(seq, -WINDOW_NT, window=WINDOW_NT)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_window_is_refused(bad):
    with pytest.raises(ScanError, match="window must be positive"):
        encode_scan_window(_seq(50), 0, window=bad)


def test_empty_sequence_is_refused():
    with pytest.raises(ScanError, match="empty sequence"):
        encode_scan_window("", 0, window=WINDOW_NT)


# ═════════════════════════════════════════════════════════════════════════════
# Geometry — the scanner inherits the pinned tiling, it does not re-derive one
# ═════════════════════════════════════════════════════════════════════════════
def test_scan_geometry_is_the_pinned_one_not_a_local_copy():
    """Drift guard: the scanner's defaults ARE `window_dataset`'s, by identity of value.

    P2-03 pinned 1024/512 (PRD §6). A scanner that quietly defaulted to a different stride
    would change coverage — and therefore the reconciled posterior — everywhere.
    """
    assert (scan_mod.WINDOW_NT, scan_mod.STRIDE_NT) == (WINDOW_NT, STRIDE_NT) == (1024, 512)


def test_window_ids_match_tile_windows_offsets():
    seq = _seq(3000)
    ids, starts = scan_window_ids(seq)
    assert starts == tile_windows(len(seq), window=WINDOW_NT, stride=STRIDE_NT)
    assert ids.shape == (len(starts), WINDOW_NT)
    for row, s in zip(ids, starts, strict=True):
        np.testing.assert_array_equal(row, encode_scan_window(seq, s, window=WINDOW_NT))


def test_long_sequence_is_tiled_entirely_in_bounds():
    """`tile_windows` is tail-anchored, so nothing past one window's length is ever padded."""
    ids, starts = scan_window_ids(_seq(5000))
    assert not np.any(ids == PAD_TOKEN_ID)
    assert min(starts) == 0
    assert max(starts) + WINDOW_NT == 5000


def test_short_sequence_yields_one_padded_window():
    """Rfam decoys are ~100-200 nt against a 1024-nt window — the common scan case."""
    ids, starts = scan_window_ids(_seq(150))
    assert starts == [0]
    assert int(np.count_nonzero(ids == PAD_TOKEN_ID)) == WINDOW_NT - 150


# ═════════════════════════════════════════════════════════════════════════════
# Zero-flank-and-flag (PRD §6) — carried end-to-end, not just available
# ═════════════════════════════════════════════════════════════════════════════
def _uniform_logits(n_windows: int, window: int) -> np.ndarray:
    return np.zeros((n_windows, window, NUM_CLASSES), dtype=np.float64)


def test_interior_positions_are_not_zero_flagged_but_short_sequences_are():
    """The flag must track real geometry: a 5000-nt scan flags nothing, a 150-nt scan flags
    everything. A flag that fired everywhere (or nowhere) would be equally 'consistent'."""
    _, long_starts = scan_window_ids(_seq(5000))
    long_out = reconcile_windows(
        _uniform_logits(len(long_starts), WINDOW_NT), np.asarray(long_starts), 5000
    )
    assert not long_out.zero_flanked.any()

    _, short_starts = scan_window_ids(_seq(150))
    short_out = reconcile_windows(
        _uniform_logits(len(short_starts), WINDOW_NT), np.asarray(short_starts), 150
    )
    assert short_out.zero_flanked.all()
    assert short_out.log_probs.shape == (150, NUM_CLASSES)


def test_every_position_is_covered_at_the_pinned_geometry():
    """A gap would reconcile to nothing; `reconcile_windows` raises, but the geometry that
    feeds it is this module's responsibility."""
    for n in (1, 150, 1023, 1024, 1025, 1536, 3000, 5000):
        _, starts = scan_window_ids(_seq(n))
        out = reconcile_windows(_uniform_logits(len(starts), WINDOW_NT), np.asarray(starts), n)
        assert out.coverage.min() >= 1
        assert out.log_probs.shape == (n, NUM_CLASSES)


# ═════════════════════════════════════════════════════════════════════════════
# The independent closed form
# ═════════════════════════════════════════════════════════════════════════════
def _analytic_logit(token_id: int, cls: int) -> float:
    """A deterministic, class-asymmetric logit function of the token id alone.

    Asymmetry matters: a function constant across classes would softmax to uniform and the
    comparison would hold no matter what the scanner did with the windows (this repo has
    shipped exactly that mistake before — a fixture whose classes were all equal).
    """
    return ((token_id * 31 + cls * 17) % 13) / 4.0 - 1.5


def _analytic_window_logits(window_ids: np.ndarray) -> np.ndarray:
    """Stand-in for a model: logits depend only on the token at each position."""
    n, w = window_ids.shape
    out = np.empty((n, w, NUM_CLASSES), dtype=np.float64)
    for i in range(n):
        for j in range(w):
            for c in range(NUM_CLASSES):
                out[i, j, c] = _analytic_logit(int(window_ids[i, j]), c)
    return out


def _hand_log_softmax(logits: list[float]) -> list[float]:
    """log_softmax in plain Python — no numpy, no `reconcile` helper, no shared code."""
    m = max(logits)
    denom = math.log(sum(math.exp(x - m) for x in logits)) + m
    return [x - denom for x in logits]


@pytest.mark.parametrize("seq_len", [150, 1024, 1600, 3000])
def test_reconciled_posterior_matches_a_hand_computed_closed_form(seq_len):
    """Pin the scanner's output against arithmetic done here from first principles.

    Construction: the stub's logits depend only on the token id, so every window covering
    position ``p`` predicts the *same* distribution there. The coverage-averaged posterior
    must therefore equal that single per-window distribution exactly — independently of how
    many windows covered ``p``.

    Verified by sabotage to bite on:

    * a wrong `starts` vector — positions align to the wrong tokens (S5, off-by-one);
    * a lost coverage normalisation — a bare log-sum-exp grows with coverage, so the
      doubly-covered interior would diverge from the singly-covered edges;
    * arg-max taken per window instead of on the reconciled distribution.

    ⚠ It does **not** bite on the pad token's *value*: `scan_window_ids` only ever pads
    out-of-bounds positions, and `reconcile_windows` slices those away before any
    arithmetic, so swapping `PAD_TOKEN_ID` for another id leaves this value untouched
    (measured — sabotage S1 failed three other tests and left every parametrisation of this
    one green). The pad convention is pinned by
    `test_pad_token_matches_training_carve_window_convention`, and the `seq_len` wiring that
    *could* let a pad in is covered in `tests/ml/test_scan_checkpoint.py`, which drives
    `scan_sequence` itself rather than calling `reconcile_windows` directly as this tier
    must (no torch in bare CI).

    None of it routes through `reconcile_windows`' own arithmetic to decide what is correct.
    """
    seq = _seq(seq_len, seed=7)
    ids, starts = scan_window_ids(seq)
    out = reconcile_windows(_analytic_window_logits(ids), np.asarray(starts), seq_len)

    # Independent expectation, computed per position from the base alone.
    for p in (0, seq_len // 3, seq_len // 2, seq_len - 1):
        token = int(encode_bases(seq[p])[0])
        expected = _hand_log_softmax([_analytic_logit(token, c) for c in range(NUM_CLASSES)])
        np.testing.assert_allclose(out.log_probs[p], expected, rtol=0, atol=1e-12)
        assert int(out.prediction[p]) == int(np.argmax(expected))

    # A proper distribution, and coverage genuinely varied across the sequence where the
    # geometry says it should (otherwise the coverage-invariance claim above is untested).
    np.testing.assert_allclose(np.exp(out.log_probs).sum(axis=1), 1.0, rtol=0, atol=1e-12)
    if seq_len > WINDOW_NT:
        assert out.coverage.max() > out.coverage.min()


def test_the_closed_form_fixture_is_actually_discriminative():
    """Guard the guard: if `_analytic_logit` were degenerate the test above would compare
    uniform to uniform and pass against any implementation."""
    for token in sorted({*BASE_TO_ID.values(), PAD_TOKEN_ID}):
        row = [_analytic_logit(token, c) for c in range(NUM_CLASSES)]
        assert len(set(row)) > 1, f"token {token} produced class-constant logits {row}"

    # And the PAD row must differ from every real base's, or a pad leak would be invisible.
    pad_row = [_analytic_logit(PAD_TOKEN_ID, c) for c in range(NUM_CLASSES)]
    for base, token in BASE_TO_ID.items():
        assert [_analytic_logit(token, c) for c in range(NUM_CLASSES)] != pad_row, (
            f"base {base} is indistinguishable from PAD under the fixture — a pad leak "
            f"would not move the reconciled value"
        )
