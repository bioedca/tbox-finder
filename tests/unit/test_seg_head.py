"""P1-03 — tests for the Stage-1 per-position 8-class segmentation head.

Two tiers (mirroring ``tests/unit/test_caduceus_backbone.py``):

- **Bare stdlib tier** (runs in CI, no torch): the head's class-index contract is
  single-sourced from :mod:`tbox_finder.labels`, so this tier locks that source to
  the exact ADR-0004 D1 8-class order/mapping and checks the ``2*d_model`` width
  constant — the invariants the head depends on, torch-free.
- **Torch tier** (the ``TestSegHeadTorch`` class; ``skipif`` skips it in bare CI,
  runs under ``tbox-ml-dna`` on CPU for the §8.5 manual gate): the
  ``SegmentationHead`` / ``LinearChainCRF`` forward shape, CRF Viterbi validity,
  the transition layer proven to *bite* (CRF==argmax when transitions vanish; a
  stay-in-tag-0 bias overrides emissions), masked decoding, and the width guard.
"""

from __future__ import annotations

import pytest

from tbox_finder.labels import CLASS_INDEX, CLASS_ORDER
from tbox_finder.models.caduceus_backbone import D_MODEL

# The ADR-0004 D1 normative class order, written out as a literal so a silent
# reordering of the single source (labels.CLASS_ORDER) is caught here.
_ADR0004_CLASS_ORDER = (
    "background",
    "Stem_I",
    "Specifier",
    "Stem_II",
    "Stem_III",
    "Antiterminator_Tbox_seq",
    "Terminator",
    "Discriminator",
)

# The RC-concatenated Caduceus-PS hidden width (2*d_model = 512), computed
# torch-free from the backbone constant so the bare tier and helpers can use it.
_INPUT_DIM = 2 * D_MODEL


# ========================================================================== #
# Bare stdlib tier — the head's class-index contract (no torch, runs in CI)
# ========================================================================== #
def test_class_order_matches_adr0004_literal():
    assert CLASS_ORDER == _ADR0004_CLASS_ORDER
    assert len(CLASS_ORDER) == 8


def test_class_index_is_position_in_order():
    for i, name in enumerate(_ADR0004_CLASS_ORDER):
        assert CLASS_INDEX[name] == i


def test_caduceus_hidden_width_is_two_d_model():
    # The head's default input width is the RC-concatenated Caduceus-PS hidden
    # state 2*d_model = 512; asserted torch-free against the backbone constant.
    assert D_MODEL == 256
    assert _INPUT_DIM == 512


# ========================================================================== #
# Torch tier — SegmentationHead + LinearChainCRF (skips in bare CI)
# ========================================================================== #
try:
    import torch  # noqa: E402

    from tbox_finder.models.seg_head import (  # noqa: E402
        CADUCEUS_PS_HIDDEN_DIM,
        NUM_CLASSES,
        LinearChainCRF,
        SegmentationHead,
    )

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in the bare CI env
    torch = None
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch not installed (bare CI) — head tier runs under tbox-ml-dna"
)


def _hidden(batch: int, length: int, dim: int = _INPUT_DIM):
    torch.manual_seed(0)
    return torch.randn(batch, length, dim)


@requires_torch
class TestSegHeadTorch:
    # --- head basics ----------------------------------------------------- #
    def test_head_constants_match_adr0004(self):
        head = SegmentationHead(_INPUT_DIM)
        assert NUM_CLASSES == 8
        assert head.num_classes == 8
        assert head.class_order == _ADR0004_CLASS_ORDER
        assert CADUCEUS_PS_HIDDEN_DIM == 512

    @pytest.mark.parametrize("use_crf", [False, True])
    def test_forward_logits_shape_is_B_L_8(self, use_crf):
        head = SegmentationHead(_INPUT_DIM, use_crf=use_crf)
        logits = head(_hidden(3, 17))
        assert tuple(logits.shape) == (3, 17, 8)

    def test_forward_rejects_wrong_input_width(self):
        head = SegmentationHead(_INPUT_DIM)
        with pytest.raises(ValueError, match="input_dim"):
            head(_hidden(2, 5, dim=_INPUT_DIM + 1))

    def test_head_rejects_bad_construction(self):
        with pytest.raises(ValueError):
            SegmentationHead(0)
        with pytest.raises(ValueError):
            SegmentationHead(_INPUT_DIM, dropout=1.0)

    # --- decoding: argmax (no CRF) --------------------------------------- #
    def test_argmax_decode_matches_argmax_and_is_valid(self):
        head = SegmentationHead(_INPUT_DIM, use_crf=False)
        hidden = _hidden(4, 11)
        paths = head.decode(hidden)
        expected = head(hidden).argmax(dim=-1)
        assert len(paths) == 4
        for b, path in enumerate(paths):
            assert len(path) == 11
            assert all(0 <= c < 8 for c in path)
            assert path == expected[b].tolist()

    # --- decoding: CRF Viterbi ------------------------------------------- #
    def test_crf_viterbi_returns_length_L_valid_paths(self):
        head = SegmentationHead(_INPUT_DIM, use_crf=True)
        paths = head.decode(_hidden(3, 13))
        assert len(paths) == 3
        for path in paths:
            assert len(path) == 13
            assert all(0 <= c < 8 for c in path)

    def test_crf_viterbi_length_one(self):
        head = SegmentationHead(_INPUT_DIM, use_crf=True)
        paths = head.decode(_hidden(2, 1))
        assert [len(p) for p in paths] == [1, 1]
        assert all(0 <= p[0] < 8 for p in paths)

    def test_crf_equals_argmax_when_transitions_vanish(self):
        # With zero start/end/transition scores the CRF cannot prefer any path
        # shape, so Viterbi must reduce to per-position argmax of the emissions.
        head = SegmentationHead(_INPUT_DIM, use_crf=True)
        with torch.no_grad():
            head.crf.start_transitions.zero_()
            head.crf.end_transitions.zero_()
            head.crf.transitions.zero_()
        hidden = _hidden(3, 9)
        crf_paths = head.decode(hidden)
        argmax = head(hidden).argmax(dim=-1)
        for b, path in enumerate(crf_paths):
            assert path == argmax[b].tolist()

    def test_crf_transition_bias_overrides_emissions(self):
        # A strong stay-in-tag-0 bias must force the whole decoded path to class
        # 0, regardless of the emission argmax — proving the transition layer bites.
        head = SegmentationHead(_INPUT_DIM, use_crf=True)
        with torch.no_grad():
            head.crf.start_transitions.zero_()
            head.crf.end_transitions.zero_()
            head.crf.transitions.fill_(-1e4)
            head.crf.transitions[0, 0] = 1e4
            head.crf.start_transitions[0] = 1e4
        hidden = _hidden(2, 8)
        for path in head.decode(hidden):
            assert path == [0] * 8

    # --- CRF loss -------------------------------------------------------- #
    def test_crf_nll_is_finite_nonnegative(self):
        head = SegmentationHead(_INPUT_DIM, use_crf=True)
        hidden = _hidden(4, 7)
        tags = torch.randint(0, 8, (4, 7))
        loss = head.loss(hidden, tags)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        # partition >= any single-path score ⇒ NLL >= 0 (allow tiny fp slack)
        assert loss.item() >= -1e-5

    def test_crf_partition_ge_gold_score_per_sequence(self):
        head = SegmentationHead(_INPUT_DIM, use_crf=True)
        hidden = _hidden(5, 6)
        tags = torch.randint(0, 8, (5, 6))
        per_seq_nll = head.loss(hidden, tags, reduction="none")
        assert per_seq_nll.shape == (5,)
        assert bool((per_seq_nll >= -1e-4).all())

    def test_masked_cross_entropy_loss_finite(self):
        head = SegmentationHead(_INPUT_DIM, use_crf=False)
        hidden = _hidden(3, 10)
        tags = torch.randint(0, 8, (3, 10))
        mask = torch.ones(3, 10, dtype=torch.bool)
        mask[0, 7:] = False  # left-aligned padding
        loss = head.loss(hidden, tags, mask=mask)
        assert torch.isfinite(loss) and loss.ndim == 0

    # --- masking respected in decode ------------------------------------- #
    @pytest.mark.parametrize("use_crf", [False, True])
    def test_decode_truncates_to_mask_lengths(self, use_crf):
        head = SegmentationHead(_INPUT_DIM, use_crf=use_crf)
        hidden = _hidden(2, 6)
        mask = torch.tensor(
            [
                [True, True, True, True, False, False],
                [True, True, True, True, True, True],
            ]
        )
        paths = head.decode(hidden, mask=mask)
        assert [len(p) for p in paths] == [4, 6]
        for path in paths:
            assert all(0 <= c < 8 for c in path)

    def test_crf_rejects_non_left_aligned_mask(self):
        crf = LinearChainCRF(8)
        emissions = _hidden(1, 4, dim=8)
        bad_mask = torch.tensor([[False, True, True, True]])
        with pytest.raises(ValueError, match="left-aligned"):
            crf.viterbi_decode(emissions, mask=bad_mask)

    # --- standalone CRF sanity ------------------------------------------- #
    def test_linear_chain_crf_rejects_tiny_tagset(self):
        with pytest.raises(ValueError, match="num_tags"):
            LinearChainCRF(1)

    def test_decode_is_deterministic(self):
        head = SegmentationHead(_INPUT_DIM, use_crf=True)
        hidden = _hidden(3, 12)
        assert head.decode(hidden) == head.decode(hidden)
