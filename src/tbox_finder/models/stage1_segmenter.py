"""Stage-1 segmenter (P1-04): Caduceus-PS backbone → RC-combination → seg head.

Wires the P1-02 Caduceus-PS backbone and the P1-03 per-position 8-class
:class:`~tbox_finder.models.seg_head.SegmentationHead` into a single Stage-1
segmenter (:class:`Stage1Segmenter`), with an ``rc_combine`` knob (:class:`RCCombine`)
that governs how the backbone's **reverse-complement-concatenated** hidden state feeds
the head.

The directionality constraint (load-bearing)
--------------------------------------------
Caduceus-PS is RC-equivariant: a single forward gives a per-position hidden state of
width ``2*d_model`` — the **forward-strand channels concatenated with the
reverse-complement-strand channels** (:func:`caduceus_backbone._hidden_states` →
``(B, L, 2*d_model)``; the model-card RC transform is ``f(rc(x)) == f(x).flip((-2, -1))``).
Because detection is strand-agnostic, the §6 **strand-resolver derives strand + 5′→3′
orientation from the predicted element order** (Specifier / Stem I → antiterminator /
terminator). An **order-destroying average** of the two strand-halves — a combination
that is *symmetric* under swapping the forward and RC channel halves — collapses that
distinction (its output cannot tell a window from its reverse complement), so it would
**defeat strand resolution**. PRD §6 / §10.1 therefore **constrain the RC-combination
ablation to a directionality-preserving (non-averaged) form**, pinned in ADR-0002.

So the ``rc_combine`` knob is bounded to non-averaged forms (a P2 ablation dimension,
PRD §11 Sweeps: *RC-combination*):

* ``"concat"`` — **default**. Identity: feed the full ``2*d_model`` ``(fwd || rc)`` hidden
  state to the head (head ``input_dim == CADUCEUS_PS_HIDDEN_DIM == 2*d_model``). Injective,
  so no strand information is lost — directionality-preserving by construction.
* ``"gate"`` — a learned **directional** per-channel gate ``g⊙fwd + (1-g)⊙rc`` reducing
  ``2*d_model → d_model``, with ``g = sigmoid(gate_logit)`` initialised **asymmetric**
  (``g ≠ 0.5``) so the two strand-halves are weighted differently by construction. It is a
  *non-averaged* form: only the hard-coded symmetric mean (``g ≡ 0.5``) is excluded, since a
  symmetric average can never represent strand order.

An order-destroying **mean** (``0.5*(fwd + rc)``) is **not a selectable mode** — it is
rejected at construction and can never be the default (a unit test locks this).

**Compute.** The ``rc_combine`` + head are small (a gate / linear), CPU-fine — the unit
test exercises them on synthetic hidden states with **no backbone forward**. The backbone
forward (``Stage1Segmenter.forward(input_ids=…)``) needs CUDA (ADR-0002 A2 C2) and is
exercised at the P1-07 fine-tune smoke, not here. Torch is imported at module top (this
*is* a ``torch`` model, mirroring :mod:`tbox_finder.models.seg_head`); the package
``__init__`` stays lazy so the *package* still imports bare.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tbox_finder.models.caduceus_backbone import D_MODEL, _hidden_states
from tbox_finder.models.rc_combine import (
    ALLOWED_RC_COMBINE,
    DEFAULT_RC_COMBINE,
    FORBIDDEN_RC_COMBINE,
    is_directionality_preserving,
    normalize_rc_combine,
)
from tbox_finder.models.seg_head import SegmentationHead

# Re-export the torch-free mode contract (defined in tbox_finder.models.rc_combine so bare
# CI can lock it) for callers that reach it via this module.
__all__ = [
    "ALLOWED_RC_COMBINE",
    "FORBIDDEN_RC_COMBINE",
    "DEFAULT_RC_COMBINE",
    "GATE_INIT_LOGIT",
    "normalize_rc_combine",
    "is_directionality_preserving",
    "swap_strand_channels",
    "RCCombine",
    "Stage1Segmenter",
]

#: Initial gate logit for the ``"gate"`` mode — **nonzero** so ``g = sigmoid(0.5) ≈ 0.62``
#: at init (``≠ 0.5``): the forward and RC channel-halves are weighted asymmetrically by
#: construction, i.e. directionality-preserving from the first step (learnable thereafter).
GATE_INIT_LOGIT: float = 0.5


def swap_strand_channels(hidden: Tensor) -> Tensor:
    """Swap the forward-strand and RC-strand channel halves of a ``(…, 2*d_model)`` state.

    The channel-space action of "read the opposite strand": splits the last dim into the two
    equal halves ``(fwd, rc)`` and returns ``(rc, fwd)`` concatenated. A **directionality-
    preserving** combination is *not* invariant under this swap (element order survives, so
    the strand-resolver can read it); an **order-destroying** symmetric average *is* invariant
    under it (order collapses). Used as the directionality probe in the P1-04 unit gate and
    available for the P2 RC-combination ablation diagnostics.
    """
    if hidden.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even (2*d_model), got {hidden.shape[-1]}")
    fwd, rc = torch.chunk(hidden, 2, dim=-1)
    return torch.cat((rc, fwd), dim=-1)


class RCCombine(nn.Module):
    """Combine the ``2*d_model`` RC-concatenated Caduceus-PS hidden state for the seg head.

    Bounded to **directionality-preserving (non-averaged)** forms (:data:`ALLOWED_RC_COMBINE`),
    per PRD §6/§10.1 (ADR-0002). ``forward`` maps ``(B, L, 2*d_model) → (B, L, output_dim)``.

    Args:
        d_model: per-strand hidden width (Caduceus-PS ``d_model`` = 256; the input is twice
            this). Defaults to :data:`~tbox_finder.models.caduceus_backbone.D_MODEL`.
        mode: one of :data:`ALLOWED_RC_COMBINE`. ``"concat"`` (default) keeps ``output_dim ==
            2*d_model``; ``"gate"`` reduces to ``d_model`` with a learned directional gate. An
            order-destroying ``"mean"`` is rejected (:func:`normalize_rc_combine`).
    """

    def __init__(self, d_model: int = D_MODEL, *, mode: str = DEFAULT_RC_COMBINE) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.d_model = int(d_model)
        self.in_dim = 2 * self.d_model
        self.mode = normalize_rc_combine(mode)
        if self.mode == "gate":
            self.output_dim = self.d_model
            # Per-channel directional gate logit; g = sigmoid(gate_logit). Initialised nonzero
            # (GATE_INIT_LOGIT) so g ≠ 0.5 → the fwd/RC halves are weighted asymmetrically by
            # construction (directionality-preserving at init; learnable thereafter).
            self.gate_logit = nn.Parameter(torch.full((self.d_model,), GATE_INIT_LOGIT))
        else:  # "concat"
            self.output_dim = self.in_dim

    @property
    def directionality_preserving(self) -> bool:
        """Always True — the module only admits non-averaged modes (never constructs a mean)."""
        return True

    def forward(self, hidden: Tensor) -> Tensor:
        """``(B, L, 2*d_model) → (B, L, output_dim)`` per the selected non-averaged mode."""
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be (B, L, 2*d_model), got {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.in_dim:
            raise ValueError(f"hidden last dim {hidden.shape[-1]} != 2*d_model ({self.in_dim})")
        if self.mode == "concat":
            return hidden
        # "gate": convex per-channel mix of the two strand-halves with a LEARNED, asymmetric
        # gate — a non-averaged (directionality-preserving) form (only g ≡ 0.5 would be a mean).
        fwd, rc = torch.chunk(hidden, 2, dim=-1)
        g = torch.sigmoid(self.gate_logit)
        return g * fwd + (1.0 - g) * rc


class Stage1Segmenter(nn.Module):
    """Stage-1 segmenter: Caduceus-PS backbone → :class:`RCCombine` → :class:`SegmentationHead`.

    ``forward(input_ids=…)`` runs the backbone (CUDA; ADR-0002 A2 C2), combines the RC-
    concatenated hidden state via the non-averaged ``rc_combine`` knob, and emits per-position
    8-class logits ``(B, L, 8)`` over :data:`~tbox_finder.labels.CLASS_ORDER`. The
    backbone-free path (:meth:`logits_from_hidden` / :meth:`decode_from_hidden` /
    :meth:`loss_from_hidden`) drives the RC-combine + head directly on hidden states — the
    CPU-testable surface (the P1-04 unit gate) and the frozen-embedding probe entry (P1-05/P1-08).

    Args:
        backbone: the Caduceus-PS model (``load_caduceus_ps``); may be ``None`` to build only
            the RC-combine + head (the CPU/probe path). ``forward(input_ids=…)`` then raises.
        rc_combine: RC-combination mode (:data:`ALLOWED_RC_COMBINE`; default ``"concat"``).
        use_crf / dropout: forwarded to :class:`SegmentationHead`.
        d_model: per-strand hidden width (default :data:`D_MODEL`).
    """

    def __init__(
        self,
        backbone: nn.Module | None = None,
        *,
        rc_combine: str = DEFAULT_RC_COMBINE,
        use_crf: bool = False,
        dropout: float = 0.0,
        d_model: int = D_MODEL,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.rc_combine = RCCombine(d_model, mode=rc_combine)
        self.head = SegmentationHead(self.rc_combine.output_dim, use_crf=use_crf, dropout=dropout)

    @property
    def rc_combine_mode(self) -> str:
        """The active RC-combination mode (canonicalised, always non-averaged)."""
        return self.rc_combine.mode

    def logits_from_hidden(self, hidden_states: Tensor) -> Tensor:
        """RC-combine + head on backbone hidden states → per-position logits ``(B, L, 8)``."""
        return self.head(self.rc_combine(hidden_states))

    def decode_from_hidden(
        self, hidden_states: Tensor, mask: Tensor | None = None
    ) -> list[list[int]]:
        """Best per-position class path per sequence (CRF Viterbi else argmax; see head)."""
        return self.head.decode(self.rc_combine(hidden_states), mask=mask)

    def loss_from_hidden(
        self,
        hidden_states: Tensor,
        tags: Tensor,
        mask: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Segmentation loss (CRF NLL else masked cross-entropy; see head)."""
        return self.head.loss(self.rc_combine(hidden_states), tags, mask=mask, reduction=reduction)

    def forward(self, *, input_ids: Tensor, **backbone_kwargs) -> Tensor:
        """Backbone → RC-combine → head → per-position logits ``(B, L, 8)`` (needs CUDA)."""
        if self.backbone is None:
            raise RuntimeError(
                "Stage1Segmenter has no backbone; pass one to run forward(input_ids=…), or use "
                "logits_from_hidden(hidden_states) with precomputed backbone hidden states."
            )
        hidden = _hidden_states(self.backbone(input_ids=input_ids, **backbone_kwargs))
        return self.logits_from_hidden(hidden)
