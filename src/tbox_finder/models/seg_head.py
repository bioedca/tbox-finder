"""Stage-1 per-position 8-class segmentation head (P1-03).

A per-nucleotide token-classification head over the Caduceus-PS hidden states:
a ``Linear(input_dim -> 8)`` projection producing per-position logits over the
**normative 8-class scheme** (ADR-0004 D1), with an **optional** in-repo
linear-chain CRF transition layer for boundary coherence.

Design notes
------------
* **Class order is single-sourced** from :mod:`tbox_finder.labels`
  (``CLASS_ORDER`` / ``CLASS_INDEX``) — the same vocabulary the label-derivation
  pipeline (P0-20/P0-21) paints and the eval metrics (:mod:`tbox_finder.metrics`)
  score. This module never redeclares the class identifiers or their order, so a
  head logit index always maps to the ADR-0004 identifier at the same position.
* **The CRF is implemented in-repo** (no external ``torchcrf`` / ``pytorch-crf``
  dependency), per the P1-03 note — a standard batch-first linear-chain CRF
  (forward algorithm for the partition function, Viterbi for decoding). Boundary
  coherence is **optional** so P2 can ablate ``Linear``-only vs ``Linear``+CRF.
* **Torch is imported at module top** — unlike the :mod:`tbox_finder.models`
  package ``__init__`` (which keeps heavy imports lazy so the *package* imports
  cleanly in a bare CI env), this submodule *is* a ``torch`` model and requires
  ``torch`` to import. The unit test guards with ``pytest.importorskip("torch")``
  so bare CI skips the head tier and it runs under ``tbox-ml-dna`` for the §8.5
  manual gate (CPU is sufficient — this is a small ``Linear`` + CRF).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tbox_finder.labels import CLASS_ORDER
from tbox_finder.models.caduceus_backbone import D_MODEL

#: Number of segmentation classes (ADR-0004 D1). The head is 8-class by
#: construction; ``Terminator`` is class-I-only and its absence is handled by the
#: loss mask, not a separate head (P1-03 note).
NUM_CLASSES: int = len(CLASS_ORDER)

#: Expected backbone hidden width for the default Caduceus-PS checkpoint:
#: ``2 * d_model`` — the RC-concatenated per-position hidden state
#: (see :func:`tbox_finder.models.caduceus_backbone._hidden_states`, ``(B, L, 2*d_model)``).
#: Single-sourced from the backbone ``D_MODEL`` so the two cannot drift.
CADUCEUS_PS_HIDDEN_DIM: int = 2 * D_MODEL


# --------------------------------------------------------------------------- #
# In-repo linear-chain CRF (no external dependency)
# --------------------------------------------------------------------------- #
class LinearChainCRF(nn.Module):
    """Batch-first linear-chain conditional random field.

    Learns three parameter groups over ``num_tags`` tags:

    * ``start_transitions`` ``(num_tags,)`` — score of a sequence *starting* in tag i;
    * ``end_transitions`` ``(num_tags,)`` — score of a sequence *ending* in tag i;
    * ``transitions`` ``(num_tags, num_tags)`` — ``transitions[i, j]`` is the score of
      moving from tag i to tag j.

    ``forward`` returns the **mean/sum negative log-likelihood** of the gold tag
    sequence (a loss); :meth:`viterbi_decode` returns the single best tag path per
    sequence. Both accept an optional ``(B, L)`` boolean ``mask`` (left-aligned:
    real tokens first, padding after; the first timestep must be valid). Adapted
    from the standard forward / Viterbi algorithms (Lafferty, McCallum & Pereira
    2001; the batch-first formulation mirrors the widely-used ``pytorch-crf``).
    """

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        if num_tags < 2:
            raise ValueError(f"num_tags must be >= 2, got {num_tags}")
        self.num_tags = int(num_tags)
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise all transition scores ~ U(-0.1, 0.1)."""
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: Tensor,
        tags: Tensor,
        mask: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Negative log-likelihood of ``tags`` under the CRF.

        Args:
            emissions: ``(B, L, num_tags)`` per-position emission logits.
            tags: ``(B, L)`` gold class indices in ``[0, num_tags)``.
            mask: optional ``(B, L)`` bool; ``None`` ⇒ all valid.
            reduction: ``"mean"`` | ``"sum"`` | ``"none"`` (per-sequence).
        """
        self._validate(emissions, tags=tags, mask=mask)
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        mask = mask.bool()

        numerator = self._score(emissions, tags, mask)  # (B,)
        denominator = self._partition(emissions, mask)  # (B,)
        nll = denominator - numerator  # (B,), always >= 0 (partition >= any path)

        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            return nll.mean()
        raise ValueError(f"unknown reduction {reduction!r}")

    def _score(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        """Un-normalised score of the gold path, ``(B,)``."""
        batch_size, seq_len, _ = emissions.shape
        idx = torch.arange(batch_size, device=emissions.device)
        mask_f = mask.to(emissions.dtype)

        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[idx, 0, tags[:, 0]]
        for t in range(1, seq_len):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[idx, t, tags[:, t]]
            score = score + (trans + emit) * mask_f[:, t]

        # end transition at each sequence's last valid position
        last_idx = mask.long().sum(dim=1) - 1  # (B,)
        last_tags = tags[idx, last_idx]
        score = score + self.end_transitions[last_tags]
        return score

    def _partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """Log partition function via the forward algorithm, ``(B,)``."""
        seq_len = emissions.shape[1]
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)
        for t in range(1, seq_len):
            broadcast_alpha = alpha.unsqueeze(2)  # (B, C, 1)
            broadcast_emit = emissions[:, t].unsqueeze(1)  # (B, 1, C)
            inner = broadcast_alpha + self.transitions.unsqueeze(0) + broadcast_emit
            next_alpha = torch.logsumexp(inner, dim=1)  # (B, C)
            keep = mask[:, t].unsqueeze(1)  # (B, 1)
            alpha = torch.where(keep, next_alpha, alpha)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    @torch.no_grad()
    def viterbi_decode(self, emissions: Tensor, mask: Tensor | None = None) -> list[list[int]]:
        """Best tag path per sequence (length = number of valid tokens)."""
        self._validate(emissions, mask=mask)
        batch_size, seq_len, _ = emissions.shape
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)
        mask = mask.bool()

        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)
        history: list[Tensor] = []
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # (B, C, 1)
            broadcast_emit = emissions[:, t].unsqueeze(1)  # (B, 1, C)
            candidate = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emit
            next_score, best_prev = candidate.max(dim=1)  # (B, C), (B, C)
            keep = mask[:, t].unsqueeze(1)
            score = torch.where(keep, next_score, score)
            history.append(best_prev)
        score = score + self.end_transitions.unsqueeze(0)

        seq_lengths = mask.long().sum(dim=1)  # (B,)
        best_paths: list[list[int]] = []
        for b in range(batch_size):
            length = int(seq_lengths[b].item())
            best_last = int(score[b].argmax(dim=0).item())
            path = [best_last]
            # history[t-1] is the best predecessor entering position t; walk back
            # over the (length-1) transitions among this sequence's valid tokens.
            for hist in reversed(history[: length - 1]):
                best_last = int(hist[b, best_last].item())
                path.insert(0, best_last)
            best_paths.append(path)
        return best_paths

    def _validate(
        self,
        emissions: Tensor,
        *,
        tags: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(f"emissions must be (B, L, num_tags), got {tuple(emissions.shape)}")
        if emissions.shape[2] != self.num_tags:
            raise ValueError(f"emissions last dim {emissions.shape[2]} != num_tags {self.num_tags}")
        if emissions.shape[1] < 1:
            raise ValueError("emissions must have length L >= 1")
        if tags is not None and tags.shape != emissions.shape[:2]:
            raise ValueError(
                f"tags shape {tuple(tags.shape)} != emissions[:2] {tuple(emissions.shape[:2])}"
            )
        if mask is not None:
            if mask.shape != emissions.shape[:2]:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != emissions[:2] {tuple(emissions.shape[:2])}"
                )
            if not bool(mask[:, 0].all()):
                raise ValueError("mask must be left-aligned: the first timestep must be valid")
            if mask.shape[1] > 1:
                # No valid token may follow a padding token: the forward /
                # Viterbi algorithms treat mask.sum() as a contiguous prefix
                # length, so an interior gap (e.g. [True, False, True]) would
                # silently corrupt the score / decoded path.
                reopened = (~mask[:, :-1].bool()) & mask[:, 1:].bool()
                if bool(reopened.any()):
                    raise ValueError(
                        "mask must be left-aligned: no valid tokens may follow padding"
                    )


# --------------------------------------------------------------------------- #
# The 8-class segmentation head
# --------------------------------------------------------------------------- #
class SegmentationHead(nn.Module):
    """Per-position 8-class token-classification head over backbone hidden states.

    ``forward(hidden_states)`` returns per-position logits ``(B, L, 8)`` over the
    ADR-0004 :data:`~tbox_finder.labels.CLASS_ORDER`. With ``use_crf=True`` a
    :class:`LinearChainCRF` provides boundary-coherent training loss and Viterbi
    decoding; with ``use_crf=False`` the loss is masked cross-entropy and decoding
    is per-position ``argmax``.

    Args:
        input_dim: width of the per-position backbone hidden state. For the default
            Caduceus-PS checkpoint this is :data:`CADUCEUS_PS_HIDDEN_DIM` (``2*d_model``).
        use_crf: attach the optional linear-chain CRF transition layer.
        dropout: dropout applied to the hidden states before the linear projection.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        use_crf: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.input_dim = int(input_dim)
        self.use_crf = bool(use_crf)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.input_dim, NUM_CLASSES)
        self.crf: LinearChainCRF | None = LinearChainCRF(NUM_CLASSES) if self.use_crf else None

    @property
    def num_classes(self) -> int:
        """Number of output classes (always 8; ADR-0004 D1)."""
        return NUM_CLASSES

    @property
    def class_order(self) -> tuple[str, ...]:
        """The ADR-0004 class identifiers, index-aligned to the logit dimension."""
        return CLASS_ORDER

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Per-position logits ``(B, L, 8)`` from hidden states ``(B, L, input_dim)``."""
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be (B, L, input_dim), got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.input_dim:
            raise ValueError(
                f"hidden_states last dim {hidden_states.shape[-1]} != input_dim {self.input_dim}"
            )
        return self.classifier(self.dropout(hidden_states))

    def loss(
        self,
        hidden_states: Tensor,
        tags: Tensor,
        mask: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Segmentation loss.

        With the CRF, the per-sequence negative log-likelihood; otherwise masked
        cross-entropy (``ignore_index`` on masked-out positions). ``reduction`` is
        ``"mean"`` | ``"sum"`` | ``"none"``. Note ``reduction="none"`` shape differs
        by mode: per-sequence ``(B,)`` with the CRF, per-token ``(B, L)`` without it.
        """
        logits = self(hidden_states)
        if self.crf is not None:
            return self.crf(logits, tags, mask=mask, reduction=reduction)
        return _masked_cross_entropy(logits, tags, mask, reduction)

    @torch.no_grad()
    def decode(self, hidden_states: Tensor, mask: Tensor | None = None) -> list[list[int]]:
        """Best per-position class path per sequence (length = #valid tokens).

        CRF Viterbi when ``use_crf`` else per-position ``argmax``. Every returned
        index is a valid class in ``[0, 8)``. Inference is done in ``eval`` mode
        (dropout disabled) regardless of the module's current training state, then
        the previous state is restored, so decoding is deterministic.
        """
        was_training = self.training
        self.eval()
        try:
            logits = self(hidden_states)
            if self.crf is not None:
                return self.crf.viterbi_decode(logits, mask=mask)
            return _argmax_decode(logits, mask)
        finally:
            self.train(was_training)


# --------------------------------------------------------------------------- #
# Non-CRF helpers
# --------------------------------------------------------------------------- #
def _masked_cross_entropy(
    logits: Tensor, tags: Tensor, mask: Tensor | None, reduction: str
) -> Tensor:
    """Masked cross-entropy over ``(B, L, C)`` logits and ``(B, L)`` targets."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")
    # F.cross_entropy K-dim form wants (N, C, d1); transpose L into the last dim.
    logits_t = logits.transpose(1, 2)  # (B, C, L)
    target = tags
    if mask is not None:
        target = tags.clone()
        target[~mask.bool()] = -100  # ignore_index
    return F.cross_entropy(logits_t, target, ignore_index=-100, reduction=reduction)


def _argmax_decode(logits: Tensor, mask: Tensor | None) -> list[list[int]]:
    """Per-position ``argmax`` path per sequence, truncated to valid tokens."""
    preds = logits.argmax(dim=-1)  # (B, L)
    batch_size = preds.shape[0]
    if mask is None:
        return [preds[b].tolist() for b in range(batch_size)]
    mask = mask.bool()
    lengths = mask.long().sum(dim=1)
    return [preds[b, : int(lengths[b])].tolist() for b in range(batch_size)]
