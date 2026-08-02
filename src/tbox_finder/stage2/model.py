"""Stage-2 RiNALMo-giga multi-head re-ranker (P3-03) — sequence-only, LoRA-adapted.

:class:`Stage2Model` puts the PRD §10.2 heads on top of the parity-confirmed
``multimolecule/rinalmo-giga`` encoder (ADR-0002 D5/A9, pinned revision):

======  =========================================  ===================================
head    what it predicts (PRD §10.2 / §8)          output
======  =========================================  ===================================
(a)     binary T-box, calibrated at P3-07          ``tbox_logit``            ``(B,)``
(b)     per-base boundary refinement               ``boundary_logits``  ``(B, L, 8)``
(c)     regulatory mode (transcriptional/          ``regulatory_mode_logits`` ``(B, 2)``
        translational)
(d)     G-D auxiliary multi-task: specifier        ``specifier_codon_logits`` ``(B, 64)``
        codon / cognate amino acid / tRNA          ``cognate_aa_logits``      ``(B, n)``
        family                                     ``trna_family_logits``     ``(B, n)``
======  =========================================  ===================================

Head widths come from :class:`~tbox_finder.stage2.heads.Stage2HeadSpec`, which derives
them rather than declaring them — see that module for why (the PRD and the ADRs name
the heads but pin no cardinality).

Sequence-only, structurally (the load-bearing prohibition)
----------------------------------------------------------
PRD §10.2: *"RiNALMo ingests* **sequence only** *(no structure-input channel)."* PRD §6
adds that predicted/annotated structure is *"used solely as the §8/§11 auxiliary
structure-consistency* **target**\\ *, not a Stage-2 input"*, and ADR-0002 D5 repeats it.
:meth:`Stage2Model.forward` therefore takes **exactly** ``input_ids`` and
``attention_mask`` and declares no ``**kwargs``: passing a dot-bracket string is a
``TypeError`` at the call site, not a convention someone can drift past. A unit test
locks the signature, because a prohibition enforced only by review is enforced by nobody.

Token axis vs nucleotide axis
-----------------------------
``RnaTokenizer`` wraps every sequence in ``<cls> … <eos>``, so the encoder emits
``L + 2`` positions while ``label_string`` and ``pairing_dotbracket`` have ``L``. Head
(b) is trained against the per-nucleotide axis, so :meth:`Stage2Model.strip_special_tokens`
removes both flanking tokens and returns ``(B, L, H)`` index-aligned to the label string,
plus the nucleotide mask the P3-04 loss needs. It reproduces ``multimolecule``'s own
``BasePredictionHead.remove_special_tokens`` contract — drop position 0, zero the EOS at
each row's own last valid index, drop the final column — so padded positions stay zero
and rows of different lengths coexist in one batch without a gather.

LoRA composition: the backbone is wrapped, the heads are not
------------------------------------------------------------
:func:`build_stage2_model` wraps **the encoder alone** with
:func:`tbox_finder.train.lora_harness.build_peft_model` (the P1-15 entry, which owns the
PRD §10.3 pins ``r=16 / α=32 / dropout=0.05 / target_modules="all-linear"``) and attaches
the heads outside that wrapper. Two reasons this order is not cosmetic: ``"all-linear"``
adapts *every* ``nn.Linear`` it can reach, so heads inside the wrapper would be
LoRA-adapted instead of trained; and ``build_peft_model``'s frozen-base measurement —
"no trainable parameter lacking ``lora_`` in its name" — would be tripped by the heads
and stop meaning anything. Wrapping first keeps both properties, and
:func:`build_stage2_model` re-measures the split rather than asserting it.

The LoRA scalars are deliberately **not** re-declared here; re-typing them would fork the
§10.3 contract into two places that can disagree.

Torch is imported at module top (this *is* a torch model, mirroring
:mod:`tbox_finder.models.stage1_segmenter`); the package ``__init__`` stays bare so
:mod:`tbox_finder.stage2.dataset` still imports in the ``data`` env.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tbox_finder.eval.rinalmo_parity import D_MODEL, REVISION
from tbox_finder.models.seg_head import NUM_CLASSES, SegmentationHead
from tbox_finder.stage2 import tokenizer as tok
from tbox_finder.stage2.heads import (
    AMINO_ACID_FIELD,
    BOUNDARY_FIELD,
    CODON_FIELD,
    REGULATORY_MODE_FIELD,
    TRNA_FAMILY_FIELD,
    Stage2HeadSpec,
)

__all__ = ["HEAD_OUTPUT_KEYS", "OUTPUT_KEYS", "Stage2Model", "build_stage2_model"]

#: The logit keys :meth:`Stage2Model.forward` returns, in PRD §10.2 head order (a–d).
HEAD_OUTPUT_KEYS: tuple[str, ...] = (
    "tbox_logit",
    "boundary_logits",
    "regulatory_mode_logits",
    "specifier_codon_logits",
    "cognate_aa_logits",
    "trna_family_logits",
)

#: Everything :meth:`Stage2Model.forward` returns: the head logits plus the nucleotide
#: mask, which the P3-04 loss needs to drop padded positions from the per-nt term.
OUTPUT_KEYS: tuple[str, ...] = (*HEAD_OUTPUT_KEYS, "nucleotide_mask")


def _hidden_size(backbone: nn.Module) -> int:
    """The backbone's per-position width, read off the live module.

    Walks the PEFT wrapper if there is one. Never defaulted: a head built for a width the
    backbone does not produce is the Stage-1 ``build_model`` trap, and it surfaces as a
    shape error deep in a training run rather than at construction.
    """
    candidates = [backbone]
    unwrap = getattr(backbone, "get_base_model", None)
    if callable(unwrap):
        candidates.append(unwrap())
    candidates.append(getattr(backbone, "base_model", None))
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        hidden = getattr(config, "hidden_size", None)
        if hidden is not None:
            return int(hidden)
    raise ValueError(
        "cannot read hidden_size off the backbone (no .config.hidden_size on it, its PEFT "
        "base model, or its .base_model) — pass d_model= explicitly if this is intentional"
    )


class Stage2Model(nn.Module):
    """RiNALMo-giga encoder → four PRD §10.2 heads. Sequence in, logits out.

    ``forward(input_ids=…, attention_mask=…)`` runs the backbone and returns a dict over
    :data:`OUTPUT_KEYS`. The backbone-free surface (:meth:`heads_from_hidden`) drives the
    heads directly on precomputed hidden states — the CPU-testable path, and the one a
    frozen-embedding probe would use.

    Args:
        spec: the head vocabularies (:class:`~tbox_finder.stage2.heads.Stage2HeadSpec`).
        backbone: the RiNALMo encoder, normally already LoRA-wrapped by
            :func:`build_stage2_model`. May be ``None`` to build the heads alone (the
            CPU/probe path); :meth:`forward` then raises.
        d_model: per-position hidden width. Read off ``backbone`` when one is given —
            passing a conflicting value is an error, not an override.
        dropout: applied to the pooled representation before heads (a), (c) and (d), and
            forwarded to head (b).
        boundary_use_crf: attach the :class:`~tbox_finder.models.seg_head.LinearChainCRF`
            transition layer to head (b) for boundary-coherent loss/decoding.
    """

    def __init__(
        self,
        spec: Stage2HeadSpec,
        *,
        backbone: nn.Module | None = None,
        d_model: int | None = None,
        dropout: float = 0.0,
        boundary_use_crf: bool = False,
    ) -> None:
        super().__init__()
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        boundary_width = spec.size(BOUNDARY_FIELD)
        if boundary_width != NUM_CLASSES:
            # SegmentationHead's width is the ADR-0004 D1 class count; a spec carrying a
            # different boundary vocabulary would index into someone else's logit axis.
            raise ValueError(
                f"spec boundary vocabulary has {boundary_width} classes but the shared "
                f"SegmentationHead emits {NUM_CLASSES} (ADR-0004 D1) — they must agree"
            )

        resolved = self._resolve_d_model(backbone, d_model)
        self.spec = spec
        self.backbone = backbone
        self.d_model = resolved
        self.pool_dropout = nn.Dropout(float(dropout))

        # (a) binary T-box. One logit; the calibration is a separate P3-07 temperature,
        # never baked in here (PRD §12 pins the stack order).
        self.tbox_head = nn.Linear(resolved, 1)
        # (b) per-base boundary refinement — the SAME 8-class head Stage 1 uses, reused
        # rather than reimplemented so the two cannot drift apart.
        self.boundary_head = SegmentationHead(
            resolved, use_crf=boundary_use_crf, dropout=float(dropout)
        )
        # (c) regulatory mode.
        self.regulatory_mode_head = nn.Linear(resolved, spec.size(REGULATORY_MODE_FIELD))
        # (d) the G-D auxiliary multi-task head — a regularizer / atlas-display signal,
        # explicitly NOT the §13.3(d) evidence-path specifier (imp.md P3-03).
        self.specifier_codon_head = nn.Linear(resolved, spec.size(CODON_FIELD))
        self.cognate_aa_head = nn.Linear(resolved, spec.size(AMINO_ACID_FIELD))
        self.trna_family_head = nn.Linear(resolved, spec.size(TRNA_FAMILY_FIELD))

    @staticmethod
    def _resolve_d_model(backbone: nn.Module | None, d_model: int | None) -> int:
        if backbone is None:
            return int(d_model) if d_model is not None else D_MODEL
        measured = _hidden_size(backbone)
        if d_model is not None and int(d_model) != measured:
            raise ValueError(
                f"d_model={d_model} conflicts with the backbone's hidden_size={measured}; "
                f"omit d_model and let it be measured"
            )
        return measured

    # -- properties ----------------------------------------------------------- #
    @property
    def head_modules(self) -> dict[str, nn.Module]:
        """The six head modules, keyed by attribute name — the LoRA-exclusion set."""
        return {
            "tbox_head": self.tbox_head,
            "boundary_head": self.boundary_head,
            "regulatory_mode_head": self.regulatory_mode_head,
            "specifier_codon_head": self.specifier_codon_head,
            "cognate_aa_head": self.cognate_aa_head,
            "trna_family_head": self.trna_family_head,
        }

    @property
    def head_dtype(self) -> torch.dtype:
        """The heads' parameter dtype; hidden states are cast to it before the heads run."""
        return self.tbox_head.weight.dtype

    # -- the backbone-free surface -------------------------------------------- #
    @staticmethod
    def strip_special_tokens(
        hidden_states: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``(B, T, H), (B, T) → (B, T-2, H), (B, T-2)``: drop ``<cls>`` and each row's ``<eos>``.

        Reproduces ``multimolecule``'s ``BasePredictionHead.remove_special_tokens``: shift
        off position 0, zero the EOS at each row's own last valid index, then drop the
        trailing column. Padding stays zero and every row's nucleotides land at indices
        ``0 … L_i-1``, so the result is index-aligned to that row's ``label_string``
        without any per-row gather.
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be (B, T, H), got {tuple(hidden_states.shape)}")
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be (B, T), got {tuple(attention_mask.shape)}")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"attention_mask {tuple(attention_mask.shape)} does not match hidden_states "
                f"{tuple(hidden_states.shape[:2])}"
            )
        n_special = tok.N_FLANKING_SPECIAL_TOKENS
        if hidden_states.shape[1] <= n_special:
            raise ValueError(
                f"token axis is {hidden_states.shape[1]}, which leaves no nucleotide after "
                f"removing {n_special} special tokens"
            )
        lengths = attention_mask.sum(dim=-1)
        if int(lengths.min()) <= n_special:
            # A row of <cls><eos> (or shorter) has no nucleotide; without this guard its
            # EOS index would be computed from an empty span and silently mis-zeroed.
            raise ValueError(
                f"every row needs > {n_special} unmasked tokens (cls + >=1 nucleotide + eos); "
                f"shortest row has {int(lengths.min())}"
            )

        hidden = hidden_states[:, 1:, :]
        mask = attention_mask[:, 1:]
        eos_index = mask.sum(dim=-1) - 1
        positions = torch.arange(mask.shape[-1], device=mask.device)
        keep = positions.unsqueeze(0) != eos_index.unsqueeze(1)
        nucleotide_mask = (mask.bool() & keep)[:, :-1]
        hidden = hidden[:, :-1, :] * nucleotide_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, nucleotide_mask

    @staticmethod
    def pool(nucleotide_hidden: Tensor, nucleotide_mask: Tensor) -> Tensor:
        """Masked mean over nucleotide positions → ``(B, H)``, the sequence-level summary.

        Mean-pooling rather than a ``<cls>`` read because there is no pooler to read from:
        the checkpoint carries none and ``load_rinalmo_backbone`` passes
        ``add_pooling_layer=False`` so PEFT is never handed a randomly-initialised module
        to adapt. Masked so a padded batch gives each row the same vector it would get
        alone — the padding must not dilute the mean.
        """
        weights = nucleotide_mask.unsqueeze(-1).to(nucleotide_hidden.dtype)
        total = (nucleotide_hidden * weights).sum(dim=1)
        return total / weights.sum(dim=1).clamp(min=1.0)

    def heads_from_hidden(self, hidden_states: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """Heads (a)–(d) on precomputed backbone hidden states — no backbone forward.

        ``hidden_states`` is the raw encoder output over the **token** axis (``L + 2``);
        special-token removal happens here so every caller gets the same alignment.
        """
        nucleotide_hidden, nucleotide_mask = self.strip_special_tokens(
            hidden_states, attention_mask
        )
        # Cast once, here: the encoder runs in bf16 (PRD §10.3) while the heads stay in
        # their own dtype, and one explicit cast beats a dtype error inside each head.
        nucleotide_hidden = nucleotide_hidden.to(self.head_dtype)
        pooled = self.pool_dropout(self.pool(nucleotide_hidden, nucleotide_mask))
        return {
            "tbox_logit": self.tbox_head(pooled).squeeze(-1),
            "boundary_logits": self.boundary_head(nucleotide_hidden),
            "regulatory_mode_logits": self.regulatory_mode_head(pooled),
            "specifier_codon_logits": self.specifier_codon_head(pooled),
            "cognate_aa_logits": self.cognate_aa_head(pooled),
            "trna_family_logits": self.trna_family_head(pooled),
            "nucleotide_mask": nucleotide_mask,
        }

    # -- the full forward ------------------------------------------------------ #
    def forward(
        self, *, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Backbone → heads (a)–(d). **Sequence only** — the signature admits nothing else.

        There is no ``**kwargs`` here on purpose (PRD §6/§10.2, ADR-0002 D5): a structure
        channel cannot be smuggled in as an extra keyword, it is a ``TypeError``.
        """
        if self.backbone is None:
            raise RuntimeError(
                "Stage2Model has no backbone; pass one (see build_stage2_model), or use "
                "heads_from_hidden(hidden_states, attention_mask) with precomputed states."
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(tok.PAD_ID).to(torch.long)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]
        return self.heads_from_hidden(hidden_states, attention_mask)


def build_stage2_model(
    spec: Stage2HeadSpec,
    *,
    base_model: nn.Module | None = None,
    revision: str = REVISION,
    dtype: str | None = None,
    attn_implementation: str | None = None,
    gradient_checkpointing: bool = True,
    device: str | None = None,
    dropout: float = 0.0,
    boundary_use_crf: bool = False,
) -> tuple[Stage2Model, dict[str, Any]]:
    """LoRA-wrap the pinned RiNALMo encoder, then attach the heads. Returns ``(model, info)``.

    ``base_model=None`` loads the pinned checkpoint; passing a tiny same-architecture
    ``RiNALMoModel`` lets a test exercise the composition without the 2.5 GB download.

    ``info`` is ``build_peft_model``'s measured wrap record plus a ``stage2_heads`` block
    that **re-measures** the head/backbone split off the assembled model rather than
    restating the intent: how many head parameters there are, that all of them are
    trainable, and that none of them sits under the PEFT wrapper. A clause that merely
    echoes what the code asked for cannot fail.
    """
    from tbox_finder.train import lora_harness  # lazy: pulls peft/multimolecule

    peft_kwargs: dict[str, Any] = {
        "base_model": base_model,
        "revision": revision,
        "attn_implementation": attn_implementation,
        "gradient_checkpointing": gradient_checkpointing,
        "device": device,
    }
    if dtype is not None:
        peft_kwargs["dtype"] = dtype
    peft_model, info = lora_harness.build_peft_model(**peft_kwargs)

    model = Stage2Model(
        spec,
        backbone=peft_model,
        dropout=dropout,
        boundary_use_crf=boundary_use_crf,
    )

    head_param_names = {
        f"{attr}.{name}"
        for attr, module in model.head_modules.items()
        for name, _ in module.named_parameters()
    }
    captured = sorted(name for name in head_param_names if "lora_" in name)
    frozen_heads = sorted(
        f"{attr}.{name}"
        for attr, module in model.head_modules.items()
        for name, param in module.named_parameters()
        if not param.requires_grad
    )
    info = {
        **info,
        "stage2_heads": {
            "d_model": model.d_model,
            "head_sizes": spec.head_sizes,
            "output_keys": list(OUTPUT_KEYS),
            "n_head_parameters": sum(
                p.numel() for module in model.head_modules.values() for p in module.parameters()
            ),
            "n_head_modules": len(model.head_modules),
            "heads_lora_adapted": captured,
            "heads_frozen": frozen_heads,
            "heads_outside_peft_wrapper": not captured,
            "all_heads_trainable": not frozen_heads,
        },
    }
    return model, info
