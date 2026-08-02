"""Stage-2 multi-task objective (P3-04).

PRD §11 pins Stage 2 as *"binary + multi-task aux losses with a **pinned weighting method**
(fixed weights or uncertainty/GradNorm-style balancing, set in ADR-0005), plus a
with/without-aux check"*. **ADR-0005 D16** sets that method:

    default: fixed manual task weights (binary head weighted to dominate), with
    uncertainty-weighting (Kendall-style) as the pre-registered alternative if the fixed
    weights underperform on the validation ladder.

Both are implemented here; ``fixed`` is the default and the only one a bare config reaches.
D16 pins the **method**, not the numbers — the per-term weights are implementer defaults on
the PRD §11 ``aux-loss weight`` sweep axis, and :meth:`Stage2LossConfig.diagnostics` says so
per field rather than letting a swept default read as a decision.

The six terms are the PRD §10.2 heads that :meth:`~tbox_finder.stage2.model.Stage2Model.forward`
returns: (a) the binary T-box logit, (b) the per-nucleotide boundary logits, (c) the
regulatory-mode logits, and (d) the three G-D auxiliaries (specifier codon, cognate amino
acid, tRNA family).

Design notes
------------
* **Masked, not dense.** Every head-(c)/(d) target is absent on all 7,007 decoy rows of
  ``stage2_dataset.parquet``, and 202 further rows carry an unmappable specifier codon (9 a
  rendered-null tRNA family). ``Stage2HeadSpec.encode`` already emits
  :data:`~tbox_finder.stage2.heads.IGNORE_INDEX` for each, so a term must reduce over its
  *supervised* elements only. Two consequences are load-bearing: torch's own
  ``F.cross_entropy(reduction="mean")`` returns **nan** when every target in the batch is
  ignored (measured on the pinned torch 2.7.1, not inferred), and a term that is silently
  vacuous looks exactly like a term that is doing nothing wrong. So every term routes
  through :func:`~tbox_finder.train.objective.focal_cross_entropy`, whose all-ignored batch
  reduces to ``0.0``, and every term reports its own ``n_supervised`` beside its value —
  a zero with a count of 0 is a *different* fact from a zero with a count of 512.
* **Nothing is forked.** The focal/class-weight/ignore arithmetic — including the
  ``clamp_min(eps)`` that keeps ``0 < γ < 1`` from producing silent nan gradients — lives in
  ``train.objective`` and is *called*, never copied. The binary head reaches the same
  arithmetic through an exact two-logit restatement (see :func:`_binary_term`) rather than a
  second implementation of the same guards.
* **Per-term reduction is fixed at the masked mean**, deliberately not a knob. A term's
  weight is only interpretable if every term is the same kind of number; the Stage-1 CRF
  wiring learned this the other way round (a per-sequence NLL against a per-token CE differs
  by ~3 orders of magnitude, making the weight meaningless). Head (b) is a per-nucleotide
  mean and the other five are per-record means; both are O(1) mean cross-entropies.
* **Torch is imported lazily inside functions**, so this module — and the whole weighting
  scheme, the dominance rule, and the config validation — imports and tests in the bare CI
  env, where torch is absent. The tensor tier runs under ``tbox-ml-rna``.
* **Parameter-free by construction.** Under ``weighting="uncertainty"`` the learnable
  log-variances belong to the caller (build them with :func:`uncertainty_log_variances` and
  hand them to the optimiser), exactly as ``Stage1Loss`` consumes the ``SegmentationHead``'s
  CRF rather than owning it. That keeps this a plain callable, not an ``nn.Module``.
* **The boundary CRF is refused, not silently ignored.** ``SegmentationHead(use_crf=True)``
  adds transition parameters that a cross-entropy term would leave with zero gradient — a
  model that looks trained and has a dead CRF. P3-06 owns the ``use_crf`` decision (P3-03
  flagged it), so passing a CRF here raises rather than quietly computing the CE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from tbox_finder.stage2.heads import IGNORE_INDEX, VOCAB_FIELDS
from tbox_finder.train.objective import (
    finite_float,
    focal_cross_entropy,
    inverse_frequency_weights,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from torch import Tensor

__all__ = [
    "AUX_TERMS",
    "BINARY_TERM",
    "IGNORE_INDEX",
    "MultitaskLoss",
    "NUCLEOTIDE_MASK_KEY",
    "PER_NUCLEOTIDE_TERMS",
    "Stage2LossConfig",
    "TERMS",
    "TERM_TO_FIELD",
    "TERM_TO_LOGITS",
    "WEIGHTING_METHODS",
    "multitask_loss",
    "uncertainty_log_variances",
]

# --------------------------------------------------------------------------- #
# The six terms, and how they name themselves elsewhere
# --------------------------------------------------------------------------- #
#: PRD §10.2 head (a) — the primary. D16 weights it to dominate; it is the head GATE-2
#: calibrates and the one the with/without-aux check (P3-08) protects.
BINARY_TERM = "binary"

#: Heads (b)–(d): everything ``aux_weight`` scales, and everything the P3-08 no-aux arm
#: (``aux_weight=0``) removes. Ordered as PRD §10.2 lists the heads.
AUX_TERMS: tuple[str, ...] = (
    "boundary",
    "regulatory_mode",
    "specifier_codon",
    "cognate_aa",
    "trna_family",
)

#: All six, primary first.
TERMS: tuple[str, ...] = (BINARY_TERM, *AUX_TERMS)

#: Terms whose logits carry a length axis. Everything else is one prediction per record.
PER_NUCLEOTIDE_TERMS: tuple[str, ...] = ("boundary",)

#: The ``Stage2Model.forward`` output key each term reads. Hand-written because the mapping
#: is not a transform of either name (``boundary`` ↔ ``boundary_logits`` but ``binary`` ↔
#: ``tbox_logit``), and drift-guarded in the torch tier against ``model.HEAD_OUTPUT_KEYS`` —
#: importing that here would drag torch into this bare-importable module.
TERM_TO_LOGITS: Mapping[str, str] = {
    "binary": "tbox_logit",
    "boundary": "boundary_logits",
    "regulatory_mode": "regulatory_mode_logits",
    "specifier_codon": "specifier_codon_logits",
    "cognate_aa": "cognate_aa_logits",
    "trna_family": "trna_family_logits",
}

#: The ``stage2_dataset.parquet`` column each aux term supervises, i.e. the
#: ``Stage2HeadSpec`` vocabulary field whose ``encode`` produced its targets. Guarded
#: against :data:`~tbox_finder.stage2.heads.VOCAB_FIELDS` by a unit test, so a head added or
#: reordered upstream fails here loudly instead of silently mis-pairing a target with a
#: logit axis of the same width. The binary term has no vocabulary field — its target is the
#: record's own class, not a categorical spelling.
TERM_TO_FIELD: Mapping[str, str] = {
    "boundary": "label_string",
    "regulatory_mode": "regulatory_mode",
    "specifier_codon": "specifier_codon",
    "cognate_aa": "cognate_aa",
    "trna_family": "trna_family",
}

if tuple(TERM_TO_FIELD[term] for term in AUX_TERMS) != VOCAB_FIELDS:  # pragma: no cover
    raise RuntimeError(
        "the aux terms no longer pair 1:1 with Stage2HeadSpec's vocabulary fields: "
        f"{tuple(TERM_TO_FIELD[term] for term in AUX_TERMS)} != {VOCAB_FIELDS}. Raised at "
        "import rather than left to a test, because a term paired with the wrong field "
        "supervises a head with another head's targets and every loss stays finite."
    )

#: The ``forward`` output that says which positions of head (b) carry a nucleotide.
NUCLEOTIDE_MASK_KEY = "nucleotide_mask"

#: ADR-0005 D16's two named methods, and nothing else. A closed allow-list: a typo cannot
#: silently select a third behaviour, and adding a method is an ADR amendment (§7 item 2),
#: not a config value.
WEIGHTING_METHODS: tuple[str, ...] = ("fixed", "uncertainty")

#: D16's default.
DEFAULT_WEIGHTING = "fixed"

#: Head (b) is the dense-background per-nucleotide regime PRD §11 names focal loss for, so
#: it starts at Lin et al.'s γ=2; the record-level heads start at plain cross-entropy. Both
#: are implementer defaults on a swept axis — no ADR pins either.
DEFAULT_BOUNDARY_GAMMA = 2.0
DEFAULT_GAMMA = 0.0


def _term_error(term: str) -> ValueError:
    return ValueError(f"unknown term {term!r}; expected one of {TERMS}")


# --------------------------------------------------------------------------- #
# Configuration — pure tier, validated at construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage2LossConfig:
    """Stage-2 multi-task objective configuration.

    **What is pinned and what is not.** ADR-0005 D16 pins the *weighting method*
    (:attr:`weighting`) and the qualitative rule that the binary head dominates. It pins no
    numeric weight — PRD §11 makes the aux-loss weight a Hydra ``--multirun`` sweep axis, so
    every scalar below is an implementer default on a swept axis. :meth:`diagnostics` records
    that split per field rather than shipping a table of numbers that all read as decisions.

    **Effective weight.** The binary term is weighted by :attr:`binary_weight`; each aux term
    by ``aux_weight × <term>_weight``. :attr:`aux_weight` is therefore the single swept axis
    PRD §11 names, and ``aux_weight=0`` is exactly the no-aux arm P3-08 consumes — it removes
    all five aux terms and leaves the objective the binary head alone.

    **Dominance is enforced, not merely intended.** D16's "binary head weighted to dominate"
    is qualitative; it is encoded here as ``binary_weight ≥ Σ effective aux weights`` and
    checked at construction. The weaker reading (binary ≥ each aux individually) would let
    five aux terms at 0.99 out-gradient a binary at 1.0 and still call it dominance, which is
    not what D16 can mean. Both margins are reported by :meth:`dominance`; only the sum rule
    raises. Raising the cap is an ADR-0005 amendment (CLAUDE.md §7 item 2), not a config
    value — which is why this refuses on the login node rather than reporting after the run.

    Attributes:
        weighting: ADR-0005 D16 method, one of :data:`WEIGHTING_METHODS`.
        aux_weight: global multiplier on all five aux terms; the PRD §11 sweep axis.
            Under ``weighting="uncertainty"`` only ``0.0`` (drop the aux tasks) and ``1.0``
            (weight them by learned uncertainty) are meaningful; anything between is a
            hybrid of the two D16 methods and is refused.
        binary_weight: weight on head (a).
        boundary_weight: base weight on head (b), before ``aux_weight``.
        regulatory_mode_weight: base weight on head (c), before ``aux_weight``.
        specifier_codon_weight: base weight on head (d1), before ``aux_weight``.
        cognate_aa_weight: base weight on head (d2), before ``aux_weight``.
        trna_family_weight: base weight on head (d3), before ``aux_weight``.
        binary_gamma: focal focusing parameter for head (a); ``0`` ⇒ cross-entropy.
        boundary_gamma: focal focusing parameter for head (b).
        aux_gamma: focal focusing parameter shared by heads (c) and (d1–d3).
        binary_class_weight_alpha: inverse-frequency exponent for head (a)'s two classes.
        boundary_class_weight_alpha: inverse-frequency exponent for head (b)'s 8 classes.
        aux_class_weight_alpha: inverse-frequency exponent for heads (c)/(d1–d3). PRD §8
            flags the specifier amino-acid distribution as skewed (Trp/Leu/Ile common,
            Lys/Glu/Gln rare) and routes it to class-imbalance handling; this is that knob.
        ignore_index: target sentinel for an unsupervised element.
    """

    weighting: str = DEFAULT_WEIGHTING
    aux_weight: float = 1.0
    binary_weight: float = 1.0
    boundary_weight: float = 0.3
    regulatory_mode_weight: float = 0.1
    specifier_codon_weight: float = 0.1
    cognate_aa_weight: float = 0.1
    trna_family_weight: float = 0.1
    binary_gamma: float = DEFAULT_GAMMA
    boundary_gamma: float = DEFAULT_BOUNDARY_GAMMA
    aux_gamma: float = DEFAULT_GAMMA
    binary_class_weight_alpha: float = 0.0
    boundary_class_weight_alpha: float = 0.0
    aux_class_weight_alpha: float = 0.0
    ignore_index: int = IGNORE_INDEX

    def __post_init__(self) -> None:
        if self.weighting not in WEIGHTING_METHODS:
            raise ValueError(
                f"weighting must be one of {WEIGHTING_METHODS} (ADR-0005 D16 names exactly "
                f"these two); got {self.weighting!r}. A third method is an ADR amendment, "
                "not a config value."
            )
        for field in fields(self):
            if field.name in ("weighting", "ignore_index"):
                continue
            finite_float(getattr(self, field.name), field.name)
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise ValueError(f"ignore_index must be an int, got {self.ignore_index!r}")
        if self.weighting == "uncertainty" and float(self.aux_weight) not in (0.0, 1.0):
            raise ValueError(
                "under weighting='uncertainty' the per-task weights are learned, so "
                f"aux_weight must be 0.0 (drop the aux tasks) or 1.0; got {self.aux_weight}. "
                "Any value between scales a learned weight by a manual one, which is a "
                "hybrid of ADR-0005 D16's two methods rather than either of them."
            )
        if self.weighting == "fixed":
            margins = self.dominance()
            if not margins["holds"]:
                raise ValueError(
                    "ADR-0005 D16 weights the binary head to dominate, but "
                    f"binary_weight={self.binary_weight:g} < Σ effective aux weights="
                    f"{margins['aux_total']:g} (aux_weight={self.aux_weight:g}). Lower "
                    f"aux_weight to ≤ {self.max_aux_weight:g} or raise binary_weight; "
                    "changing the rule itself needs an ADR-0005 amendment (CLAUDE.md §7)."
                )

    # -- weights ------------------------------------------------------------ #
    def base_weight(self, term: str) -> float:
        """The declared weight on ``term``, before :attr:`aux_weight` is applied."""
        if term not in TERMS:
            raise _term_error(term)
        return float(getattr(self, "binary_weight" if term == BINARY_TERM else f"{term}_weight"))

    def effective_weights(self) -> dict[str, float]:
        """``term -> weight actually multiplying that term's loss`` under ``fixed``.

        Reported under ``uncertainty`` too, where it is what the run *would* have used —
        :func:`multitask_loss` does not apply it there beyond the ``aux_weight`` on/off gate.
        """
        aux = float(self.aux_weight)
        return {
            term: (self.base_weight(term) if term == BINARY_TERM else aux * self.base_weight(term))
            for term in TERMS
        }

    def dominance(self) -> dict[str, Any]:
        """Both readings of D16's "binary head weighted to dominate", measured.

        ``sum_margin`` is the enforced one (binary minus the aux total); ``max_margin`` is
        the weaker per-term reading, reported so the difference between the two is visible
        rather than argued.
        """
        effective = self.effective_weights()
        binary = effective[BINARY_TERM]
        aux = [effective[term] for term in AUX_TERMS]
        aux_total = sum(aux)
        return {
            "rule": "binary_weight >= sum(effective aux weights)",
            "binary": binary,
            "aux_total": aux_total,
            "aux_max": max(aux) if aux else 0.0,
            "sum_margin": binary - aux_total,
            "max_margin": binary - (max(aux) if aux else 0.0),
            "holds": binary >= aux_total,
        }

    @property
    def max_aux_weight(self) -> float:
        """Largest :attr:`aux_weight` the dominance rule admits at these base weights."""
        base_total = sum(self.base_weight(term) for term in AUX_TERMS)
        if base_total <= 0.0:
            return float("inf")
        return float(self.binary_weight) / base_total

    # -- per-term hyperparameters ------------------------------------------- #
    def gamma(self, term: str) -> float:
        """Focal focusing parameter for ``term``."""
        if term == BINARY_TERM:
            return float(self.binary_gamma)
        if term == "boundary":
            return float(self.boundary_gamma)
        if term in AUX_TERMS:
            return float(self.aux_gamma)
        raise _term_error(term)

    def class_weight_alpha(self, term: str) -> float:
        """Inverse-frequency exponent for ``term``; ``0`` ⇒ unweighted."""
        if term == BINARY_TERM:
            return float(self.binary_class_weight_alpha)
        if term == "boundary":
            return float(self.boundary_class_weight_alpha)
        if term in AUX_TERMS:
            return float(self.aux_class_weight_alpha)
        raise _term_error(term)

    def diagnostics(self) -> dict[str, Any]:
        """Reported view of the configuration, with what is ADR-pinned marked as such."""
        return {
            "weighting": self.weighting,
            "weighting_methods": list(WEIGHTING_METHODS),
            "terms": list(TERMS),
            "aux_terms": list(AUX_TERMS),
            "aux_weight": float(self.aux_weight),
            "base_weights": {term: self.base_weight(term) for term in TERMS},
            "effective_weights": self.effective_weights(),
            "gamma": {term: self.gamma(term) for term in TERMS},
            "class_weight_alpha": {term: self.class_weight_alpha(term) for term in TERMS},
            "dominance": self.dominance(),
            "max_aux_weight": self.max_aux_weight,
            "ignore_index": int(self.ignore_index),
            # D16 pins the METHOD and the dominance rule; PRD §11 makes every scalar a swept
            # axis. Recording one flat "pinned: true/false" would misstate both halves.
            "pinned": {
                "weighting_method": "ADR-0005 D16",
                "dominance_rule": "ADR-0005 D16",
                "weight_values": False,
                "gamma_values": False,
                "class_weight_alpha_values": False,
            },
        }


# --------------------------------------------------------------------------- #
# Tensor tier — the six terms
# --------------------------------------------------------------------------- #
def _check_supervised_targets(target: Any, *, term: str, ignore_index: int) -> None:
    """Reject a float/bool target tensor before it silently indexes the wrong class."""
    import torch  # lazy

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"targets[{term!r}] must be a Tensor, got {type(target).__name__}")
    if target.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise ValueError(
            f"targets[{term!r}] must be an integer tensor of class indices (or "
            f"{ignore_index} where unsupervised); got dtype {target.dtype}. A float target "
            "would be truncated toward zero and land on class 0."
        )


def _binary_term(
    logits: Tensor,
    target: Tensor,
    *,
    gamma: float,
    weight: Sequence[float] | None,
    ignore_index: int,
) -> Tensor:
    """Head (a)'s masked focal BCE, expressed exactly as a 2-class cross-entropy.

    Softmax over the pair ``[0, z]`` is ``[1-σ(z), σ(z)]``, so cross-entropy against a
    ``{0, 1}`` target **is** ``binary_cross_entropy_with_logits(z, y)`` — identically, not
    approximately, and with the same gradient ``σ(z) - y`` (the fabricated ``0`` channel is
    a constant: ``zeros_like`` carries no grad). A unit test pins that equality against
    torch's own BCE as an external reference.

    Restating it this way is not cleverness for its own sake: it means head (a) inherits the
    *one* audited implementation of the ignore-index handling, the ``Σ w_y`` weighted-mean
    denominator, the all-ignored-reduces-to-0 convention, and the ``clamp_min(eps)`` that
    stops ``0 < γ < 1`` from producing a silent nan gradient. A separate binary focal loss
    would be a second place for each of those to be got wrong.

    ``weight``, when given, is ``[w_negative, w_positive]``.
    """
    import torch  # lazy

    if logits.dim() != 1:
        raise ValueError(f"tbox_logit must be (B,), got {tuple(logits.shape)}")
    if target.shape != logits.shape:
        raise ValueError(f"binary target {tuple(target.shape)} != tbox_logit {tuple(logits.shape)}")
    supervised = target != ignore_index
    if bool(((target < 0) | (target > 1))[supervised].any()):
        raise ValueError(
            "binary targets must be 0, 1, or the ignore sentinel; got other class indices"
        )
    pair = torch.stack([torch.zeros_like(logits), logits], dim=-1).unsqueeze(1)  # (B, 1, 2)
    return focal_cross_entropy(
        pair,
        target.unsqueeze(1),
        gamma=gamma,
        weight=weight,
        ignore_index=ignore_index,
        reduction="mean",
    )


def _boundary_term(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    gamma: float,
    weight: Sequence[float] | None,
    ignore_index: int,
) -> Tensor:
    """Head (b)'s masked focal cross-entropy over the padded nucleotide axis.

    Cross-checks the two independent signals the batch carries about which positions are
    real. ``nucleotide_mask`` says where a nucleotide is; the target says where a *label* is.
    The check is deliberately one-directional: a padded position carrying a real class is
    corruption (the model would be trained on a label for a base that does not exist), while
    a real position carrying ``ignore_index`` is the ordinary decoy case — all 7,007 decoy
    rows are unlabelled end to end. Enforcing the reverse direction too would refuse every
    negative in the corpus.
    """
    if logits.dim() != 3:
        raise ValueError(f"boundary_logits must be (B, L, C), got {tuple(logits.shape)}")
    if target.shape != logits.shape[:2]:
        raise ValueError(
            f"boundary target {tuple(target.shape)} != boundary_logits[:2] "
            f"{tuple(logits.shape[:2])}"
        )
    if mask.shape != logits.shape[:2]:
        raise ValueError(
            f"{NUCLEOTIDE_MASK_KEY} {tuple(mask.shape)} != boundary_logits[:2] "
            f"{tuple(logits.shape[:2])}"
        )
    padded = ~mask.bool()
    if bool((target[padded] != ignore_index).any()):
        raise ValueError(
            f"a position masked out by {NUCLEOTIDE_MASK_KEY} carries a real boundary class; "
            "the collator must set ignore_index at every padded position, or the head would "
            "be trained on a label for a base that does not exist."
        )
    return focal_cross_entropy(
        logits,
        target,
        gamma=gamma,
        weight=weight,
        ignore_index=ignore_index,
        reduction="mean",
    )


def _record_term(
    logits: Tensor,
    target: Tensor,
    *,
    term: str,
    gamma: float,
    weight: Sequence[float] | None,
    ignore_index: int,
) -> Tensor:
    """One masked focal cross-entropy per record, for heads (c) and (d1–d3)."""
    if logits.dim() != 2:
        raise ValueError(f"{term} logits must be (B, C), got {tuple(logits.shape)}")
    if target.shape != logits.shape[:1]:
        raise ValueError(
            f"{term} target {tuple(target.shape)} != logits[:1] {tuple(logits.shape[:1])}"
        )
    return focal_cross_entropy(
        logits.unsqueeze(1),
        target.unsqueeze(1),
        gamma=gamma,
        weight=weight,
        ignore_index=ignore_index,
        reduction="mean",
    )


def uncertainty_log_variances(
    *,
    terms: Sequence[str] = TERMS,
    device: Any = None,
    dtype: Any = None,
) -> Tensor:
    """The learnable log-variances for ADR-0005 D16's pre-registered alternative.

    Returns an ``nn.Parameter`` of shape ``(len(terms),)`` initialised to zero, i.e.
    ``s_i = log σ_i² = 0`` ⇒ ``σ_i = 1`` ⇒ every task starts at unit weight and contributes
    nothing to the regulariser. The caller owns it and must hand it to the optimiser — this
    module stays parameter-free (the ``Stage1Loss``/``LinearChainCRF`` division of labour).

    Index ``i`` is ``terms[i]``; :func:`multitask_loss` reads it in :data:`TERMS` order.
    """
    import torch  # lazy
    from torch import nn  # lazy

    if tuple(terms) != TERMS:
        raise ValueError(f"terms must be exactly {TERMS} in that order; got {tuple(terms)}")
    return nn.Parameter(torch.zeros(len(terms), device=device, dtype=dtype))


def multitask_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    config: Stage2LossConfig | None = None,
    class_weights: Mapping[str, Sequence[float]] | None = None,
    log_variances: Tensor | None = None,
    boundary_crf: Any = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Combine the six Stage-2 head losses under the ADR-0005 D16 weighting method.

    Args:
        outputs: what ``Stage2Model.forward`` returned — the six logit tensors keyed by
            :data:`TERM_TO_LOGITS` plus :data:`NUCLEOTIDE_MASK_KEY`.
        targets: ``term -> target tensor``, keyed by :data:`TERMS`. Every key is required and
            an unknown key raises: a mis-keyed target would otherwise train nothing while
            every loss stayed finite and plausible. Shapes are ``(B,)`` for every term except
            ``boundary``, which is ``(B, L)``. Unsupervised elements carry
            ``config.ignore_index``.
        config: defaults to :class:`Stage2LossConfig` — D16's fixed manual weights.
        class_weights: optional ``term -> per-class weights`` (length ``C`` for that head;
            ``[w_negative, w_positive]`` for ``binary``). Derive them from measured counts
            with :class:`MultitaskLoss` rather than assuming them.
        log_variances: required iff ``config.weighting == "uncertainty"``, forbidden
            otherwise; build it with :func:`uncertainty_log_variances`.
        boundary_crf: must be ``None``. Present so that turning on ``use_crf`` at P3-06
            fails here rather than training a CRF whose transitions never receive a gradient.

    Returns:
        ``(total, components)``. ``components`` carries each term's unweighted value, its
        weight, its ``n_supervised`` count, the terms skipped for want of any supervision in
        this batch, and — under ``uncertainty`` — the learned weights and the regulariser.

    Note:
        **Uncertainty weighting**, when selected, is the log-variance form
        ``Σ_i [ exp(-s_i) · L_i + s_i / 2 ]`` with ``s_i = log σ_i²``, i.e. Kendall, Gal &
        Cipolla's homoscedastic-uncertainty weighting (arXiv:1705.07115) in the
        classification parameterisation ``σ⁻² L + log σ``
        [DOI:10.1002/mp.15870 eq. 4; DOI:10.1049/cmu2.12793 eq. 17 (both accessed
        2026-08-02)] — every Stage-2 term here is a cross-entropy, so the classification
        coefficient is the applicable one rather than the regression form's extra ½ on the
        data term. Parameterising by ``s`` rather than ``σ`` avoids the division by zero the
        ``1/σ²`` form invites [DOI:10.1155/2022/6653598 eq. 12].

        A term with **no** supervision in the batch is dropped from the total entirely,
        regulariser included. Keeping it would hand the optimiser a free ``s_i/2`` to
        minimise with no data term opposing it, driving ``σ_i → 0`` and that task's weight
        to infinity on the strength of a batch that taught it nothing.
    """
    import torch  # lazy

    cfg = config or Stage2LossConfig()
    if boundary_crf is not None:
        raise NotImplementedError(
            "the Stage-2 boundary CRF term is not implemented: P3-04 ships the "
            "cross-entropy boundary term only, and P3-06 owns the SegmentationHead "
            "use_crf decision P3-03 flagged. Refusing rather than computing the CE, which "
            "would leave the CRF's transition parameters with no gradient at all."
        )

    missing_targets = [term for term in TERMS if term not in targets]
    if missing_targets:
        raise ValueError(f"targets is missing terms {missing_targets}; expected all of {TERMS}")
    unknown_targets = [key for key in targets if key not in TERMS]
    if unknown_targets:
        raise ValueError(
            f"targets carries unknown keys {unknown_targets}; expected exactly {TERMS}. A "
            "mis-keyed target trains nothing while every loss stays finite."
        )
    missing_outputs = [
        key for key in (*TERM_TO_LOGITS.values(), NUCLEOTIDE_MASK_KEY) if key not in outputs
    ]
    if missing_outputs:
        raise ValueError(f"outputs is missing {missing_outputs} (see Stage2Model.OUTPUT_KEYS)")

    if cfg.weighting == "uncertainty":
        if log_variances is None:
            raise ValueError(
                "weighting='uncertainty' needs log_variances; build them with "
                "uncertainty_log_variances() and add them to the optimiser (this module "
                "owns no parameters)."
            )
        if log_variances.shape != (len(TERMS),):
            raise ValueError(
                f"log_variances must have shape ({len(TERMS)},) in TERMS order, got "
                f"{tuple(log_variances.shape)}"
            )
    elif log_variances is not None:
        raise ValueError(
            f"log_variances was supplied but weighting is {cfg.weighting!r}; they would be "
            "silently ignored, leaving parameters in the optimiser that never move."
        )

    known_weight_keys = set(TERMS)
    if class_weights is not None:
        unknown = [key for key in class_weights if key not in known_weight_keys]
        if unknown:
            raise ValueError(f"class_weights carries unknown terms {unknown}")

    ignore_index = int(cfg.ignore_index)
    effective = cfg.effective_weights()
    mask = outputs[NUCLEOTIDE_MASK_KEY]

    values: dict[str, Tensor] = {}
    n_supervised: dict[str, int] = {}
    for term in TERMS:
        logits = outputs[TERM_TO_LOGITS[term]]
        target = targets[term]
        _check_supervised_targets(target, term=term, ignore_index=ignore_index)
        weight = None if class_weights is None else class_weights.get(term)
        gamma = cfg.gamma(term)
        if term == BINARY_TERM:
            value = _binary_term(
                logits, target, gamma=gamma, weight=weight, ignore_index=ignore_index
            )
        elif term in PER_NUCLEOTIDE_TERMS:
            value = _boundary_term(
                logits, target, mask, gamma=gamma, weight=weight, ignore_index=ignore_index
            )
        else:
            value = _record_term(
                logits,
                target,
                term=term,
                gamma=gamma,
                weight=weight,
                ignore_index=ignore_index,
            )
        values[term] = value
        n_supervised[term] = int((target != ignore_index).sum().item())

    # A term contributes only where it has both supervision and a non-zero weight. Both
    # exclusions are recorded: a 0.0 term value means something different in each case, and
    # in neither case is it "the head is doing well".
    skipped_unsupervised = [term for term in TERMS if n_supervised[term] == 0]
    if cfg.weighting == "uncertainty":
        included = [
            term
            for term in TERMS
            if n_supervised[term] > 0 and (term == BINARY_TERM or float(cfg.aux_weight) > 0.0)
        ]
    else:
        included = [term for term in TERMS if n_supervised[term] > 0 and effective[term] != 0.0]

    # Multiplying by 0 rather than torch.zeros(): a batch in which every term is excluded
    # (no supervision anywhere, or aux_weight=0 with an unsupervised binary target) must
    # still return a tensor attached to the graph, or the caller's backward() raises on a
    # leaf-free scalar. The value is exactly 0.0 either way.
    zero = values[BINARY_TERM] * 0.0
    components: dict[str, Any] = {
        "terms": values,
        "n_supervised": n_supervised,
        "weighting": cfg.weighting,
        "skipped_unsupervised": skipped_unsupervised,
        "included": included,
    }

    if cfg.weighting == "fixed":
        total = zero
        for term in included:
            total = total + effective[term] * values[term]
        components["weights"] = dict(effective)
        components["weighted"] = {term: effective[term] * values[term] for term in included}
    else:
        total = zero
        regulariser = zero
        learned: dict[str, Tensor] = {}
        weighted: dict[str, Tensor] = {}
        for index, term in enumerate(TERMS):
            if term not in included:
                continue
            s = log_variances[index]
            precision = torch.exp(-s)
            weighted[term] = precision * values[term]
            regulariser = regulariser + 0.5 * s
            total = total + weighted[term] + 0.5 * s
            learned[term] = precision
        components["weights"] = learned
        components["weighted"] = weighted
        components["regulariser"] = regulariser

    components["total"] = total
    return total, components


# --------------------------------------------------------------------------- #
# The composed objective
# --------------------------------------------------------------------------- #
class MultitaskLoss:
    """The Stage-2 objective: six masked terms combined by ADR-0005 D16's pinned method.

    Parameter-free, like ``Stage1Loss``. Under ``weighting="uncertainty"`` the caller owns
    the log-variances (:func:`uncertainty_log_variances`) and passes them per call.

    Class weights are **derived from measured counts, never assumed**: pass ``class_counts``
    for any term whose ``class_weight_alpha`` is positive, and construction fails if they are
    missing rather than quietly training unweighted under a config that says otherwise.

    Example:
        >>> loss_fn = MultitaskLoss()                                     # doctest: +SKIP
        >>> total, parts = loss_fn(model(**batch), batch["targets"])      # doctest: +SKIP
    """

    def __init__(
        self,
        config: Stage2LossConfig | None = None,
        *,
        class_counts: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.config = config or Stage2LossConfig()
        self.class_counts = (
            None
            if class_counts is None
            else {term: tuple(int(n) for n in counts) for term, counts in class_counts.items()}
        )
        if self.class_counts is not None:
            unknown = [term for term in self.class_counts if term not in TERMS]
            if unknown:
                raise ValueError(f"class_counts carries unknown terms {unknown}")
        weights: dict[str, tuple[float, ...]] = {}
        for term in TERMS:
            alpha = self.config.class_weight_alpha(term)
            if alpha <= 0.0:
                continue
            counts = None if self.class_counts is None else self.class_counts.get(term)
            if counts is None:
                raise ValueError(
                    f"class_weight_alpha for {term!r} is {alpha:g} but no class_counts["
                    f"{term!r}] was supplied: inverse-frequency weights are derived from the "
                    "measured training-stream frequencies, never assumed (CLAUDE.md §10.3)."
                )
            weights[term] = inverse_frequency_weights(counts, alpha=alpha)
        self.weights: dict[str, tuple[float, ...]] | None = weights or None

    def __call__(
        self,
        outputs: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        log_variances: Tensor | None = None,
        boundary_crf: Any = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute the total loss and its components. See :func:`multitask_loss`."""
        return multitask_loss(
            outputs,
            targets,
            config=self.config,
            class_weights=self.weights,
            log_variances=log_variances,
            boundary_crf=boundary_crf,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Reported (never gated) view of this objective, for the run report and W&B.

        Carries the derived class weights only where counts were actually supplied — this
        reports what was measured and fabricates nothing where nothing was.
        """
        out = self.config.diagnostics()
        out["class_weights"] = (
            None if self.weights is None else {term: list(w) for term, w in self.weights.items()}
        )
        out["class_counts"] = (
            None
            if self.class_counts is None
            else {term: list(counts) for term, counts in self.class_counts.items()}
        )
        return out
