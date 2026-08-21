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
(e)     **optional** per-base pairing partner      ``structure_logits``
        (P3-05, off by default)                    ``(B, L, 1 + L)``
======  =========================================  ===================================

Head widths come from :class:`~tbox_finder.stage2.heads.Stage2HeadSpec`, which derives
them rather than declaring them — see that module for why (the PRD and the ADRs name
the heads but pin no cardinality). Head (e) is the exception: its class axis *is* the
sequence's own position axis, so its width is the batch's padded length, not a
vocabulary (:class:`PairingHead`).

Head (e) is opt-in (``structure_head=False``)
---------------------------------------------
PRD §8 calls the pairing-partner label *"optional (Stage-2)"* and PRD §11 *"Optional
structure-consistency auxiliary loss from dot-bracket pairing"*; ADR-0005 D16's head
enumeration does not name it. So it is **not built unless asked for**, and the default
model is exactly the six-head D16 objective P3-03/P3-04 shipped. Turning it on is the
optional arm P3-08 folds into the aux ablation, not a change to the pinned default.

Sequence-only, structurally (the load-bearing prohibition)
----------------------------------------------------------
PRD §10.2: *"RiNALMo ingests* **sequence only** *(no structure-input channel)."* PRD §6
adds that predicted/annotated structure is *"used solely as the §8/§11 auxiliary
structure-consistency* **target**\\ *, not a Stage-2 input"*, and ADR-0002 D5 repeats it.
:meth:`Stage2Model.forward` therefore takes **exactly** ``input_ids`` and
``attention_mask`` and declares no ``**kwargs``: passing a dot-bracket string is a
``TypeError`` at the call site, not a convention someone can drift past. A unit test
locks the signature, because a prohibition enforced only by review is enforced by nobody.

Head (e) does not weaken that: it **emits** structure and consumes none. Its only inputs
are the hidden states the encoder produced from the sequence and the nucleotide mask, so
the direction of travel is model → dot-bracket, which is precisely what PRD §6 permits
("used solely as the … **target**"). Both public entries — :meth:`Stage2Model.forward`
*and* :meth:`Stage2Model.heads_from_hidden` — are signature-locked by the same ``ast``
test, so structure cannot enter through the backbone-free path either.

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

from tbox_finder.eval.rinalmo_parity import D_MODEL
from tbox_finder.models.rna_backbone_registry import PRODUCTION_BACKBONE
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

__all__ = [
    "ALL_HEAD_OUTPUT_KEYS",
    "DEFAULT_PAIRING_PROJ_DIM",
    "HEAD_OUTPUT_KEYS",
    "OPTIONAL_HEAD_OUTPUT_KEYS",
    "OUTPUT_KEYS",
    "STRUCTURE_OUTPUT_KEY",
    "PairingHead",
    "Stage2Model",
    "build_stage2_model",
]

#: The logit keys :meth:`Stage2Model.forward` **always** returns, in PRD §10.2 head
#: order (a–d).
HEAD_OUTPUT_KEYS: tuple[str, ...] = (
    "tbox_logit",
    "boundary_logits",
    "regulatory_mode_logits",
    "specifier_codon_logits",
    "cognate_aa_logits",
    "trna_family_logits",
)

#: Head (e)'s key — present only when ``structure_head=True`` (PRD §8/§11 "optional").
STRUCTURE_OUTPUT_KEY = "structure_logits"

#: The logit keys an opt-in head contributes. Kept separate from
#: :data:`HEAD_OUTPUT_KEYS` so "what a default model returns" stays a constant a test can
#: pin, while :attr:`Stage2Model.output_keys` reports what *this* model returns.
OPTIONAL_HEAD_OUTPUT_KEYS: tuple[str, ...] = (STRUCTURE_OUTPUT_KEY,)

#: Every head key that can appear, in PRD head order (a–e). The P3-04/P3-05 loss module
#: names its term→key map by hand (it must stay torch-free) and is drift-guarded against
#: this tuple.
ALL_HEAD_OUTPUT_KEYS: tuple[str, ...] = (*HEAD_OUTPUT_KEYS, *OPTIONAL_HEAD_OUTPUT_KEYS)

#: Everything a **default** :meth:`Stage2Model.forward` returns: the six always-on head
#: logits plus the nucleotide mask, which the P3-04 loss needs to drop padded positions
#: from the per-nt terms.
OUTPUT_KEYS: tuple[str, ...] = (*HEAD_OUTPUT_KEYS, "nucleotide_mask")

#: Head (e)'s inner projection width. An implementer default, not an ADR-pinned number:
#: no ADR or PRD line pins a cardinality for the structure-consistency head. It sets the
#: rank of the pairing score matrix, not its shape, so it is a capacity knob and not part
#: of any contract.
DEFAULT_PAIRING_PROJ_DIM = 64


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


class PairingHead(nn.Module):
    """Head (e): per-base pairing-partner logits ``(B, L, 1 + L)`` (PRD §8, P3-05).

    The class axis is ``1 + L``: **class 0 is "unpaired"** and class ``j + 1`` is "paired
    with position ``j``". Putting the unpaired class *first* rather than last is
    load-bearing — it makes a target index independent of the batch's padded length, so a
    row encoded alone and the same row encoded inside a longer batch carry identical
    targets. With the sentinel at the end, ``L_pad`` would leak into the label.

    **Scores are symmetric by construction.** Base pairing is a symmetric relation, so
    the raw bilinear score matrix is averaged with its own transpose before the unpaired
    column is prepended. That does not force the *predictions* to be symmetric (the
    softmax is still per-row), but it removes the model's ability to score ``i→j``
    differently from ``j→i`` — the cheapest form of the consistency this head is named
    for, and one that costs a single transpose.

    Two positions are forbidden outright rather than left to be learned: the diagonal (a
    base cannot pair with itself) and every padded column (a partner that is padding is
    not a partner). Both are filled with ``finfo(dtype).min`` rather than ``-inf``, which
    keeps a fully-masked row's ``logsumexp`` finite; the unpaired class is never masked,
    so no row is ever fully masked in practice either.

    **Cost is quadratic in length**, which is why P3-06 must size for it: the logits alone
    are ``B × L × (1+L)``, i.e. ~33 MB in fp32 at ``B=8, L=1022`` (RiNALMo's context —
    :data:`~tbox_finder.stage2.tokenizer.MAX_NUCLEOTIDE_TOKENS`), before the cross-entropy
    intermediates. The projections themselves are cheap: ``2 · d_model · proj_dim + d_model``
    parameters.

    Args:
        d_model: per-position hidden width of the states this head reads.
        proj_dim: rank of the bilinear score (:data:`DEFAULT_PAIRING_PROJ_DIM`).
        dropout: applied to the hidden states before the projections.
    """

    def __init__(
        self,
        d_model: int,
        *,
        proj_dim: int = DEFAULT_PAIRING_PROJ_DIM,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if int(proj_dim) <= 0:
            raise ValueError(f"proj_dim must be positive, got {proj_dim}")
        self.d_model = int(d_model)
        self.proj_dim = int(proj_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.left = nn.Linear(self.d_model, self.proj_dim)
        self.right = nn.Linear(self.d_model, self.proj_dim)
        # One logit per position for "this base is unpaired" — a genuine class, not a
        # threshold on the pairing scores: the dot-bracket supervises it directly.
        self.unpaired = nn.Linear(self.d_model, 1)

    def forward(self, nucleotide_hidden: Tensor, nucleotide_mask: Tensor) -> Tensor:
        """``(B, L, H), (B, L) → (B, L, 1 + L)``. Hidden states in, pairing logits out."""
        if nucleotide_hidden.dim() != 3:
            raise ValueError(
                f"nucleotide_hidden must be (B, L, H), got {tuple(nucleotide_hidden.shape)}"
            )
        if nucleotide_mask.shape != nucleotide_hidden.shape[:2]:
            raise ValueError(
                f"nucleotide_mask {tuple(nucleotide_mask.shape)} != nucleotide_hidden[:2] "
                f"{tuple(nucleotide_hidden.shape[:2])}"
            )
        hidden = self.dropout(nucleotide_hidden)
        scores = torch.matmul(self.left(hidden), self.right(hidden).transpose(1, 2))
        scores = scores * (self.proj_dim**-0.5)
        scores = 0.5 * (scores + scores.transpose(1, 2))

        length = scores.shape[1]
        self_pair = torch.eye(length, dtype=torch.bool, device=scores.device)
        padded_partner = ~nucleotide_mask.bool().unsqueeze(1)  # (B, 1, L) over partners j
        scores = scores.masked_fill(self_pair | padded_partner, torch.finfo(scores.dtype).min)
        return torch.cat([self.unpaired(hidden), scores], dim=-1)


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
            forwarded to heads (b) and (e).
        boundary_use_crf: attach the :class:`~tbox_finder.models.seg_head.LinearChainCRF`
            transition layer to head (b) for boundary-coherent loss/decoding.
        structure_head: attach head (e), the optional PRD §8/§11 pairing-partner head
            (:class:`PairingHead`). **Off by default** — see the module docstring. Must
            agree with ``Stage2LossConfig.structure_enabled``: an attached head whose term
            is disabled would sit in the optimiser receiving no gradient, and
            :func:`~tbox_finder.stage2.losses.multitask_loss` refuses that pairing.
        pairing_proj_dim: head (e)'s bilinear rank (:data:`DEFAULT_PAIRING_PROJ_DIM`).
    """

    def __init__(
        self,
        spec: Stage2HeadSpec,
        *,
        backbone: nn.Module | None = None,
        d_model: int | None = None,
        dropout: float = 0.0,
        boundary_use_crf: bool = False,
        structure_head: bool = False,
        pairing_proj_dim: int = DEFAULT_PAIRING_PROJ_DIM,
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
        # (e) the optional pairing-partner head — built only when asked for, so the
        # default model is exactly the six-head ADR-0005 D16 objective. Assigned as None
        # rather than left unset so `structure_head` is always a readable attribute.
        self.structure_head: PairingHead | None = (
            PairingHead(resolved, proj_dim=int(pairing_proj_dim), dropout=float(dropout))
            if structure_head
            else None
        )

    @staticmethod
    def _resolve_d_model(backbone: nn.Module | None, d_model: int | None) -> int:
        if backbone is None:
            # No module to measure. An explicit `d_model` wins; otherwise fall back to the
            # PRODUCTION backbone's width, and say so — since ADR-0002 A15 there are two
            # widths in the allow-list (1280 vs RNA-FM's 640), so this default is a
            # *production* assumption, not a universal one. Every real construction goes
            # through `build_stage2_model`, which always hands over a live backbone and
            # therefore always measures; this branch is for bare unit fixtures.
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
        """The live head modules, keyed by attribute name — the LoRA-exclusion set.

        Head (e) appears only when it was attached, so ``build_stage2_model``'s measured
        head/backbone split counts what this model actually has rather than what the
        default has.
        """
        modules: dict[str, nn.Module] = {
            "tbox_head": self.tbox_head,
            "boundary_head": self.boundary_head,
            "regulatory_mode_head": self.regulatory_mode_head,
            "specifier_codon_head": self.specifier_codon_head,
            "cognate_aa_head": self.cognate_aa_head,
            "trna_family_head": self.trna_family_head,
        }
        if self.structure_head is not None:
            modules["structure_head"] = self.structure_head
        return modules

    @property
    def has_structure_head(self) -> bool:
        """Was the optional PRD §8/§11 head (e) attached?"""
        return self.structure_head is not None

    @property
    def output_keys(self) -> tuple[str, ...]:
        """The keys :meth:`forward` returns for **this** model, head order then the mask."""
        head_keys = HEAD_OUTPUT_KEYS + (
            OPTIONAL_HEAD_OUTPUT_KEYS if self.has_structure_head else ()
        )
        return (*head_keys, "nucleotide_mask")

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
        outputs = {
            "tbox_logit": self.tbox_head(pooled).squeeze(-1),
            "boundary_logits": self.boundary_head(nucleotide_hidden),
            "regulatory_mode_logits": self.regulatory_mode_head(pooled),
            "specifier_codon_logits": self.specifier_codon_head(pooled),
            "cognate_aa_logits": self.cognate_aa_head(pooled),
            "trna_family_logits": self.trna_family_head(pooled),
        }
        if self.structure_head is not None:
            outputs[STRUCTURE_OUTPUT_KEY] = self.structure_head(nucleotide_hidden, nucleotide_mask)
        outputs["nucleotide_mask"] = nucleotide_mask
        return outputs

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
    backbone: str = PRODUCTION_BACKBONE,
    base_model: nn.Module | None = None,
    revision: str | None = None,
    dtype: str | None = None,
    attn_implementation: str | None = None,
    gradient_checkpointing: bool = True,
    device: str | None = None,
    dropout: float = 0.0,
    boundary_use_crf: bool = False,
    structure_head: bool = False,
    pairing_proj_dim: int = DEFAULT_PAIRING_PROJ_DIM,
) -> tuple[Stage2Model, dict[str, Any]]:
    """LoRA-wrap a pinned RNA encoder, then attach the heads. Returns ``(model, info)``.

    ``backbone`` is an :mod:`~tbox_finder.models.rna_backbone_registry` allow-list key and
    defaults to the **production** one (``rinalmo-giga``), so every pre-A15 caller is
    unchanged; ADR-0002 A15 / PRD §10.2's D6 comparator passes ``"rnafm"``. The heads are
    sized from the backbone's *measured* ``hidden_size`` (1280 vs 640), never from a constant,
    so swapping the key cannot leave a head built for the other model's width.

    ``base_model=None`` loads the pinned checkpoint; passing a tiny same-architecture model
    lets a test exercise the composition without the multi-GB download.

    ``info`` is ``build_peft_model``'s measured wrap record plus a ``stage2_heads`` block
    that **re-measures** the head/backbone split off the assembled model rather than
    restating the intent: how many head parameters there are, that all of them are
    trainable, and that none of them sits under the PEFT wrapper. A clause that merely
    echoes what the code asked for cannot fail.
    """
    from tbox_finder.train import lora_harness  # lazy: pulls peft/multimolecule

    peft_kwargs: dict[str, Any] = {
        "base_model": base_model,
        "backbone": backbone,
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
        structure_head=structure_head,
        pairing_proj_dim=pairing_proj_dim,
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
            # The LIVE keys, not the default tuple: a report that echoed OUTPUT_KEYS
            # would say "six heads" for a seven-head model.
            "output_keys": list(model.output_keys),
            "structure_head": model.has_structure_head,
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
