"""P1-04 — tests for the Stage-1 segmenter's directionality-preserving RC-combination.

The load-bearing constraint (PRD §6/§10.1; ADR-0002): the RC-combination that feeds the
seg head must be **directionality-preserving (non-averaged)**, because the §6 strand-
resolver derives strand + 5′→3′ orientation from the **predicted element order** (Specifier
/ Stem I → antiterminator) and an order-destroying average would defeat it.

Two tiers (mirroring ``tests/unit/test_seg_head.py``):

- **Bare stdlib tier** (runs in CI, no torch): the mode contract lives in the torch-free
  :mod:`tbox_finder.models.rc_combine`, so CI locks it — ``concat`` is the default, the
  allowed set is exactly the non-averaged forms, and an order-destroying *mean* is rejected.
- **Torch tier** (``skipif`` skips it in bare CI; runs under ``tbox-ml-dna`` on CPU for the
  §8.5 manual gate): the :class:`RCCombine` shapes, the **directionality property** proven to
  *bite* (allowed combos are NOT invariant under the fwd/RC channel-half swap; a reference
  symmetric mean IS invariant → order collapses), and the **end-to-end order-recoverability**
  — a ``concat`` segmenter decodes a *strand-dependent* element order, whereas a mean-combined
  reference decodes identically for a window and its channel-swapped (opposite-strand) form.
"""

from __future__ import annotations

import pytest

from tbox_finder.models.caduceus_backbone import D_MODEL
from tbox_finder.models.rc_combine import (
    ALLOWED_RC_COMBINE,
    DEFAULT_RC_COMBINE,
    FORBIDDEN_RC_COMBINE,
    is_directionality_preserving,
    normalize_rc_combine,
)

# The RC-concatenated Caduceus-PS hidden width (2*d_model = 512), computed torch-free.
_INPUT_DIM = 2 * D_MODEL


# ========================================================================== #
# Bare stdlib tier — the RC-combination mode contract (no torch, runs in CI)
# ========================================================================== #
def test_default_is_concat_a_non_averaged_form():
    assert DEFAULT_RC_COMBINE == "concat"
    assert DEFAULT_RC_COMBINE in ALLOWED_RC_COMBINE


def test_allowed_set_is_exactly_the_non_averaged_forms():
    assert ALLOWED_RC_COMBINE == ("concat", "gate")
    for mode in ALLOWED_RC_COMBINE:
        assert is_directionality_preserving(mode)


def test_mean_and_average_are_rejected_as_order_destroying():
    # The load-bearing guard: an order-destroying average is never selectable (PRD §6/§10.1).
    assert set(FORBIDDEN_RC_COMBINE) == {"mean", "average", "avg"}
    for mode in FORBIDDEN_RC_COMBINE:
        assert not is_directionality_preserving(mode)
        with pytest.raises(ValueError, match="order-destroying"):
            normalize_rc_combine(mode)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown rc_combine"):
        normalize_rc_combine("bilinear")


def test_normalize_is_case_and_whitespace_insensitive():
    assert normalize_rc_combine("  CONCAT ") == "concat"
    assert normalize_rc_combine("Gate") == "gate"
    with pytest.raises(ValueError, match="order-destroying"):
        normalize_rc_combine("  MEAN ")


# ========================================================================== #
# Torch tier — RCCombine / Stage1Segmenter (skips in bare CI)
# ========================================================================== #
try:
    import torch  # noqa: E402

    from tbox_finder.models.stage1_segmenter import (  # noqa: E402
        GATE_INIT_LOGIT,
        RCCombine,
        Stage1Segmenter,
        swap_strand_channels,
    )

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in the bare CI env
    torch = None
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch not installed (bare CI) — combine tier runs under tbox-ml-dna"
)


def _asymmetric_hidden(batch: int = 2, length: int = 6):
    """A hidden state whose forward and RC channel-halves DIFFER (so the fwd/RC swap bites)."""
    torch.manual_seed(0)
    h = torch.randn(batch, length, _INPUT_DIM)
    # Force asymmetry: zero the RC half so h != swap(h) unambiguously.
    h[..., D_MODEL:] = 0.0
    return h


def _mean_ref(hidden):
    """The FORBIDDEN order-destroying combine: the symmetric mean of the two strand-halves.

    Defined only here (never in the shipped module) to demonstrate *why* it is excluded — it
    is invariant under :func:`swap_strand_channels`, so it collapses element order.
    """
    fwd, rc = torch.chunk(hidden, 2, dim=-1)
    return 0.5 * (fwd + rc)


@requires_torch
class TestRCCombineTorch:
    # --- shapes + basic contract ----------------------------------------- #
    def test_concat_is_identity_full_width(self):
        combine = RCCombine(D_MODEL, mode="concat")
        assert combine.output_dim == _INPUT_DIM
        h = _asymmetric_hidden()
        out = combine(h)
        assert tuple(out.shape) == (2, 6, _INPUT_DIM)
        assert torch.equal(out, h)  # concat = identity (injective → order-preserving)

    def test_gate_reduces_to_d_model(self):
        combine = RCCombine(D_MODEL, mode="gate")
        assert combine.output_dim == D_MODEL
        out = combine(_asymmetric_hidden())
        assert tuple(out.shape) == (2, 6, D_MODEL)

    def test_gate_init_logit_is_nonzero_asymmetric(self):
        # g = sigmoid(GATE_INIT_LOGIT) must be != 0.5 (else the gate would be a mean at init).
        assert GATE_INIT_LOGIT != 0.0
        combine = RCCombine(D_MODEL, mode="gate")
        g = torch.sigmoid(combine.gate_logit)
        assert torch.all(g != 0.5)

    def test_rccombine_rejects_mean(self):
        with pytest.raises(ValueError, match="order-destroying"):
            RCCombine(D_MODEL, mode="mean")

    def test_rccombine_rejects_wrong_width(self):
        combine = RCCombine(D_MODEL, mode="concat")
        with pytest.raises(ValueError, match="2\\*d_model"):
            combine(torch.randn(1, 4, _INPUT_DIM + 1))

    # --- the directionality property (the crux of the P1-04 gate) -------- #
    @pytest.mark.parametrize("mode", ["concat", "gate"])
    def test_allowed_combine_is_not_invariant_under_strand_swap(self, mode):
        # Directionality-preserving ⟺ the combine can tell the two strand-halves apart, i.e.
        # it is NOT invariant under swapping the fwd/RC channels — so element order survives.
        combine = RCCombine(D_MODEL, mode=mode)
        h = _asymmetric_hidden()
        assert not torch.allclose(combine(h), combine(swap_strand_channels(h)))

    def test_mean_reference_is_invariant_under_strand_swap(self):
        # The contrast: a symmetric mean IS invariant under the fwd/RC swap — its output cannot
        # distinguish a window from its opposite strand, collapsing order. This is why it is
        # excluded from the allowed modes (and is not even constructible via RCCombine).
        h = _asymmetric_hidden()
        assert torch.allclose(_mean_ref(h), _mean_ref(swap_strand_channels(h)))

    def test_swap_strand_channels_is_an_involution(self):
        h = _asymmetric_hidden()
        assert torch.equal(swap_strand_channels(swap_strand_channels(h)), h)

    def test_swap_rejects_odd_width(self):
        with pytest.raises(ValueError, match="even"):
            swap_strand_channels(torch.randn(1, 3, _INPUT_DIM + 1))


@requires_torch
class TestStage1SegmenterTorch:
    def test_default_mode_and_head_width_concat(self):
        seg = Stage1Segmenter(rc_combine="concat")
        assert seg.rc_combine_mode == "concat"
        assert seg.head.input_dim == _INPUT_DIM  # head consumes the full RC-concatenated state

    def test_gate_mode_head_width_d_model(self):
        seg = Stage1Segmenter(rc_combine="gate")
        assert seg.rc_combine_mode == "gate"
        assert seg.head.input_dim == D_MODEL

    def test_segmenter_default_is_concat(self):
        assert Stage1Segmenter().rc_combine_mode == DEFAULT_RC_COMBINE

    def test_segmenter_rejects_mean(self):
        with pytest.raises(ValueError, match="order-destroying"):
            Stage1Segmenter(rc_combine="mean")

    def test_logits_shape_is_B_L_8(self):
        seg = Stage1Segmenter(rc_combine="concat")
        logits = seg.logits_from_hidden(_asymmetric_hidden(3, 11))
        assert tuple(logits.shape) == (3, 11, 8)

    @pytest.mark.parametrize("mode", ["concat", "gate"])
    def test_logits_are_strand_dependent(self, mode):
        # End-to-end: the 8-class logits (hence the decoded element order) differ for a window
        # and its channel-swapped (opposite-strand) form → order is recoverable downstream.
        seg = Stage1Segmenter(rc_combine=mode)
        h = _asymmetric_hidden()
        assert not torch.allclose(
            seg.logits_from_hidden(h), seg.logits_from_hidden(swap_strand_channels(h))
        )

    def test_concat_decodes_strand_dependent_element_order(self):
        # Concrete: a concat segmenter with a controlled head decodes an ORDERED element path
        # (Specifier[2] region upstream, Antiterminator[5] region downstream) for the forward
        # channels, and a DIFFERENT path for the channel-swapped (opposite-strand) input — so
        # the §6 strand-resolver can read the order. Classes: background=0, Specifier=2,
        # Antiterminator_Tbox_seq=5 (ADR-0004 CLASS_ORDER).
        seg = Stage1Segmenter(rc_combine="concat")
        with torch.no_grad():
            seg.head.classifier.weight.zero_()
            seg.head.classifier.bias.zero_()
            seg.head.classifier.weight[2, 0] = 1.0  # forward channel 0 → Specifier
            seg.head.classifier.weight[5, 1] = 1.0  # forward channel 1 → Antiterminator
        h = torch.zeros(1, 6, _INPUT_DIM)
        h[0, 0:2, 0] = 5.0  # Specifier marker, positions 0–1 (upstream)
        h[0, 3:5, 1] = 5.0  # Antiterminator marker, positions 3–4 (downstream)

        fwd_path = seg.decode_from_hidden(h)[0]
        assert fwd_path == [2, 2, 0, 5, 5, 0]  # Specifier → Antiterminator: a readable order
        assert fwd_path != fwd_path[::-1]  # non-palindromic ⇒ order is present

        # The opposite-strand (channel-swapped) input moves the markers to the RC half, which
        # the controlled (forward-reading) head ignores → an all-background path, ≠ fwd_path.
        swapped_path = seg.decode_from_hidden(swap_strand_channels(h))[0]
        assert swapped_path == [0, 0, 0, 0, 0, 0]
        assert swapped_path != fwd_path

    def test_mean_reference_collapses_element_order(self):
        # The contrast at the head: because a symmetric mean is swap-invariant, ANY head on top
        # of it yields identical logits — hence identical decoded order — for a window and its
        # opposite strand. This is exactly the collapse the §6 strand-resolver cannot tolerate.
        head = Stage1Segmenter(rc_combine="gate").head  # a d_model-wide head
        h = _asymmetric_hidden()
        logits_fwd = head(_mean_ref(h))
        logits_swap = head(_mean_ref(swap_strand_channels(h)))
        assert torch.allclose(logits_fwd, logits_swap)
        assert head.decode(_mean_ref(h)) == head.decode(_mean_ref(swap_strand_channels(h)))

    def test_forward_without_backbone_raises(self):
        seg = Stage1Segmenter(backbone=None, rc_combine="concat")
        with pytest.raises(RuntimeError, match="no backbone"):
            seg.forward(input_ids=torch.zeros(1, 8, dtype=torch.long))
