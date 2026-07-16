"""Stage-1 per-nucleotide segmentation objective (P2-02).

PRD §11 pins the Stage-1 objective as *per-nucleotide cross-entropy with **focal loss**
(γ≈2, tuned) **or** inverse-frequency weighting for the dense-background regime*
[arXiv:1708.02002], plus an *optional CRF transition term* for boundary coherence. This
module implements all three pieces behind :class:`Stage1Loss`.

Design notes
------------
* **"or", not "and".** PRD §11 offers focal loss **or** inverse-frequency weighting as
  alternative correctives for the same imbalance. They compose numerically, so the
  default expresses exactly one of them — focal ``γ=2`` with **no** class weighting
  (``class_weight_alpha=0``). Lin et al. report that the two interact and that their
  class-weight α is best decreased as γ rises (arXiv:1708.02002 §4, Table 1(b)), so
  stacking both at full strength is not what the focal-loss result licenses. P2-01's
  sampler is a third corrective on the same imbalance, and P2-01 measured how such terms
  compound. :func:`loss_mass_share` is the diagnostic that makes the *consequence*
  visible; P2-06 sweeps γ / α against it.
* **Nothing here is ADR-pinned.** PRD §11 delegates γ to a sweep (P2-06) and ADR-0005
  pins only the *Stage-2* aux-loss weighting (D16); no ADR pins a Stage-1 γ, class-weight
  exponent, or CRF weight. Every value in :class:`Stage1LossConfig` is therefore an
  **implementer default on a swept axis**, not a decision — recorded as such rather than
  presented as pinned (the P1-15/P1-16 ``config.pinned=false`` precedent).
* **Torch is imported lazily inside functions** so ``tbox_finder.train`` stays importable
  in the bare CI env (see the package ``__init__``); the pure tier
  (:func:`class_weights_from_counts`, :func:`loss_mass_share`) is stdlib-only and runs in
  bare CI, while the tensor tier runs under ``tbox-ml-dna``.
* **The CRF is not owned here.** ``SegmentationHead(use_crf=True)`` owns the
  :class:`~tbox_finder.models.seg_head.LinearChainCRF` parameters (P1-03); this module
  consumes the module the caller passes, which keeps :class:`Stage1Loss` parameter-free
  and lets it stay a plain callable rather than an ``nn.Module``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tbox_finder.labels import CLASS_ORDER, CORE_ELEMENTS

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from torch import Tensor

#: Ignored-target sentinel. Declared as a literal (not imported from ``data.window_dataset``,
#: which would pull numpy into this bare-importable module) with a drift-guard test
#: asserting equality with ``seg_smoke.IGNORE_INDEX`` — the repo-wide pattern.
IGNORE_INDEX = -100

#: Number of segmentation classes, single-sourced from the ADR-0004 D1 vocabulary.
NUM_CLASSES = len(CLASS_ORDER)

#: Focal-loss focusing parameter. PRD §11 says "γ≈2, **tuned**" and names γ a P2-06 sweep
#: axis; arXiv:1708.02002 §4 reports γ=2 best in their sweep. An implementer default on a
#: swept axis — **not** an ADR-pinned value.
DEFAULT_GAMMA = 2.0

#: Inverse-frequency exponent. ``0.0`` ⇒ no class weighting, which is what makes the
#: default read as PRD §11's "focal **or** inverse-frequency" rather than both at once.
#: ``1.0`` ⇒ full inverse-frequency (every class takes an equal share of the loss mass).
DEFAULT_CLASS_WEIGHT_ALPHA = 0.0

#: Weight on the optional CRF transition term, relative to the (per-token) focal CE.
DEFAULT_CRF_WEIGHT = 1.0


def _finite_float(value: Any, name: str, *, minimum: float = 0.0) -> float:
    """Coerce to float and reject NaN / ±inf as well as out-of-range values.

    A bare ``x < 0`` test **passes** for NaN (every comparison with NaN is False), so a
    ``gamma=nan`` arriving from a sweep config would sail through validation and turn the
    whole loss — and then every parameter — silently NaN. Non-finite hyperparameters are
    never meaningful here, so they are rejected at the boundary (CLAUDE.md §10.3: fail
    loudly rather than emit a wrong number).
    """
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out}")
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}, got {out}")
    return out


__all__ = [
    "DEFAULT_CLASS_WEIGHT_ALPHA",
    "DEFAULT_CRF_WEIGHT",
    "DEFAULT_GAMMA",
    "IGNORE_INDEX",
    "NUM_CLASSES",
    "Stage1Loss",
    "Stage1LossConfig",
    "class_weights_from_counts",
    "core_mass_share",
    "focal_cross_entropy",
    "left_align_for_crf",
    "loss_mass_share",
]


# --------------------------------------------------------------------------- #
# Pure tier — class-frequency statistics → weights, and the consequence diagnostic
# --------------------------------------------------------------------------- #
def _counts_in_class_order(counts: Mapping[str, int] | Sequence[int]) -> tuple[int, ...]:
    """Normalise a count mapping/sequence into a tuple in :data:`CLASS_ORDER` order."""
    if isinstance(counts, Mapping):
        missing = [name for name in CLASS_ORDER if name not in counts]
        if missing:
            raise ValueError(f"class counts missing classes: {missing}")
        extra = [key for key in counts if key not in CLASS_ORDER]
        if extra:
            raise ValueError(f"class counts carry unknown classes: {extra}")
        ordered = [counts[name] for name in CLASS_ORDER]
    else:
        ordered = list(counts)
        if len(ordered) != NUM_CLASSES:
            raise ValueError(f"class counts must have {NUM_CLASSES} entries, got {len(ordered)}")
    out: list[int] = []
    for name, value in zip(CLASS_ORDER, ordered, strict=True):
        # ``bool`` is an ``int`` subclass and implements ``__index__``, so it must be
        # rejected explicitly — a True/False slipping through would silently count as 1/0
        # (the P1-15/P1-16 bool-as-count defect class).
        if isinstance(value, bool):
            raise TypeError(f"count for {name!r} must be an int, got bool")
        # ``operator.index`` is the integer protocol, so numpy integers (what np.bincount
        # returns — the natural way a caller tallies these) are accepted, while np.bool_,
        # floats and strings raise. A bare ``isinstance(value, int)`` would reject np.int64
        # and force every caller to hand-convert.
        try:
            n = operator.index(value)
        except TypeError:
            raise TypeError(
                f"count for {name!r} must be an integer, got {type(value).__name__}"
            ) from None
        if n < 0:
            raise ValueError(f"count for {name!r} must be >= 0, got {n}")
        out.append(n)
    return tuple(out)


def class_weights_from_counts(
    counts: Mapping[str, int] | Sequence[int],
    *,
    alpha: float = 1.0,
) -> tuple[float, ...]:
    """Inverse-frequency class weights ``w_c ∝ (N / n_c) ** alpha``, mean-normalised to 1.

    ``counts`` is the per-class **nucleotide** count over the training stream, either a
    mapping keyed by :data:`CLASS_ORDER` name or a sequence in that order. Returns the
    weight vector in :data:`CLASS_ORDER` order, suitable for the ``weight`` argument of
    :func:`focal_cross_entropy` (and of ``torch.nn.functional.cross_entropy``).

    ``alpha`` interpolates the corrective: ``0`` ⇒ all-ones (no weighting); ``1`` ⇒ full
    inverse-frequency, under which every class contributes an equal share of the loss
    mass (see :func:`loss_mass_share`). Intermediate values temper it.

    Weights are normalised to **mean 1** for interpretability. Under the ``"mean"``
    reduction this normalisation is a no-op on the loss value — the weighted mean divides
    by ``Σ w``, so a global rescale of ``w`` cancels — but it matters for ``"sum"`` and it
    keeps a printed weight vector readable against the unweighted baseline of 1.0.

    Raises:
        ValueError: if any class has a zero count (the inverse frequency is undefined —
            reported rather than silently clamped to a finite stand-in, CLAUDE.md §10.3),
            or if ``alpha`` is negative.
    """
    alpha = _finite_float(alpha, "alpha")
    ordered = _counts_in_class_order(counts)
    total = sum(ordered)
    if total <= 0:
        raise ValueError("class counts are all zero — no training stream to weight against")
    zero = [name for name, n in zip(CLASS_ORDER, ordered, strict=True) if n == 0]
    if zero:
        raise ValueError(
            f"classes {zero} have zero nucleotides: the inverse frequency is undefined. "
            "Supply counts from a stream that covers every class, or drop the class."
        )
    raw = [(total / n) ** alpha for n in ordered]
    mean_raw = sum(raw) / len(raw)
    return tuple(w / mean_raw for w in raw)


def loss_mass_share(
    counts: Mapping[str, int] | Sequence[int],
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Expected share of the weighted loss mass each class takes: ``n_c w_c / Σ n_j w_j``.

    This is the **consequence** of a weighting choice, which is the thing worth checking —
    an exponent read in isolation says little, whereas the mass share says how much of the
    loss each class commands. P2-01 learned this on the sampler side: the ``alpha`` looked
    innocuous and only the resulting concentration revealed the pathology, so the guard
    was pinned to the consequence rather than the exponent.

    **Scope, stated rather than glossed:** this is the *static* prior fixed by the counts
    and the class weights. It deliberately excludes the focal modulator ``(1-p_t)^γ``,
    which depends on the model's current predictions and so has no value outside a
    training run — γ does not appear in the formula above and this function is
    γ-independent by construction. Read it as "what the weighting alone does to the class
    balance", not as a measured gradient attribution of the trained objective.

    With ``weights=None`` (unweighted), the share is just the class frequency. With
    ``alpha=1`` weights from :func:`class_weights_from_counts`, every class returns
    ``1/8`` by construction — which is what inverse-frequency weighting *means*, and also
    why it is worth looking at: ADR-0004 D6 excludes Stem_II, Stem_III, Terminator and
    Discriminator from GATE-4 as label-noise-prone, so full inverse-frequency hands 4/8 of
    the loss mass to the four classes the gate explicitly declines to grade.
    """
    ordered = _counts_in_class_order(counts)
    if weights is None:
        w = [1.0] * NUM_CLASSES
    else:
        if len(weights) != NUM_CLASSES:
            raise ValueError(f"weights must have {NUM_CLASSES} entries, got {len(weights)}")
        w = [_finite_float(x, f"weight[{i}]") for i, x in enumerate(weights)]
    mass = [n * x for n, x in zip(ordered, w, strict=True)]
    total = sum(mass)
    if total <= 0:
        raise ValueError("total weighted mass is zero — nothing would contribute a gradient")
    return {name: m / total for name, m in zip(CLASS_ORDER, mass, strict=True)}


def core_mass_share(
    counts: Mapping[str, int] | Sequence[int],
    weights: Sequence[float] | None = None,
) -> float:
    """Combined loss-mass share of the three GATE-4 core elements (ADR-0004 D6).

    GATE-4 grades the **minimum** per-nucleotide F1 over {Stem_I, Specifier,
    Antiterminator_Tbox_seq}; this reports how much of the loss mass those three command
    under a given weighting. Reported, never gated — no ADR pins a floor for it.
    """
    share = loss_mass_share(counts, weights)
    return sum(share[name] for name in CORE_ELEMENTS)


# --------------------------------------------------------------------------- #
# Tensor tier — focal cross-entropy
# --------------------------------------------------------------------------- #
def focal_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float,
    weight: Tensor | Sequence[float] | None = None,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> Tensor:
    """Focal cross-entropy with optional per-class weights (Lin et al. 2017; PRD §11).

    ``FL(p_t) = -w_{y} (1 - p_t)^γ log(p_t)`` per position, over ``logits`` ``(B, L, C)``
    and ``targets`` ``(B, L)`` of class indices, with ``ignore_index`` marking positions
    that carry no DNA. Generalises ``train.stage1_smoke.focal_cross_entropy`` (P1-06) by
    adding ``weight`` / ``reduction``; with ``weight=None, reduction="mean"`` it is
    numerically that function, and a test pins the equivalence.

    Two subtleties make the weighted form easy to get wrong, so both are handled here
    explicitly rather than by passing ``weight`` through to ``F.cross_entropy``:

    * **The modulator must be built from the *unweighted* CE.** ``p_t = exp(-CE)`` only
      recovers the true class probability when ``CE = -log p_t``. Handing ``weight`` to
      ``F.cross_entropy(reduction="none")`` returns ``-w_y log p_t``, so ``exp(-CE)``
      would be ``p_t ** w_y`` — the focal term ``(1 - p_t)^γ`` would then be silently
      wrong by a class-dependent amount. The class weight is therefore applied as a
      separate gather, *after* the modulator is computed.
    * **The ``"mean"`` denominator is ``Σ w_y``, not the valid-token count.** This
      matches ``F.cross_entropy(weight=…, reduction="mean")``, which normalises by the
      total weight of the non-ignored targets (verified against the pinned torch 2.7.1,
      not inferred from the docs). Dividing by the token count instead would disagree
      with torch by a factor of ``mean(w_y)`` — and would break the γ=0 ≡ weighted-CE
      equivalence that this module's test asserts against torch as an external reference.

    At ignored positions ``F.cross_entropy(reduction="none")`` returns 0, and the gathered
    weight is forced to 0, so they contribute to neither numerator nor denominator.

    Args:
        logits: ``(B, L, C)`` per-position logits.
        targets: ``(B, L)`` class indices, or ``ignore_index``.
        gamma: focusing parameter; ``0`` recovers (weighted) cross-entropy.
        weight: optional per-class weights, length ``C``, in :data:`CLASS_ORDER` order.
        ignore_index: target value to exclude.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"`` (per-token ``(B, L)``).
    """
    import torch  # lazy — keeps tbox_finder.train bare-importable
    import torch.nn.functional as F  # lazy

    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")
    gamma = _finite_float(gamma, "gamma")
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B, L, C), got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} != logits[:2] {tuple(logits.shape[:2])}"
        )

    logits_t = logits.transpose(1, 2)  # (B, C, L) for F.cross_entropy's K-dim form
    ce = F.cross_entropy(logits_t, targets, ignore_index=ignore_index, reduction="none")  # (B, L)
    pt = torch.exp(-ce)
    # clamp_min(eps), not clamp_min(0): a confidently-correct position drives ce to
    # *exactly* 0.0 (measured on this exact call path with the pinned torch 2.7.1: at a
    # correct-class logit gap of 18 in both bf16 and fp32 — the K-dim kernel accumulates
    # bf16 in fp32 — and 38 in fp64), so 1-pt is exactly 0. pow's backward is
    # gamma * x**(gamma-1) * grad_out, which for 0 < gamma < 1 is inf at x=0, and grad_out
    # there is ce = 0, so inf * 0 = **nan** — silently, since the forward loss stays
    # finite. The nan then floods every upstream parameter. gamma < 1 is squarely on the
    # swept axis (Lin et al.'s own grid includes 0.5), so this must not be left to bite
    # P2-06. Clamping the base kills the inf while leaving the forward value bit-identical
    # (the clamp only engages where ce ~ 0, and the product is ~0 there either way) and
    # the gradients bit-identical at gamma = 0, 1, 2 — both verified in the unit tests.
    focal = (1.0 - pt).clamp_min(torch.finfo(ce.dtype).eps).pow(gamma) * ce
    valid = targets != ignore_index

    if weight is None:
        per_token = focal * valid
        denom = valid.sum().to(per_token.dtype)
    else:
        w = torch.as_tensor(weight, dtype=logits.dtype, device=logits.device)
        if w.dim() != 1 or w.numel() != logits.shape[2]:
            raise ValueError(
                f"weight must be a 1-D tensor of length C={logits.shape[2]}, "
                f"got shape {tuple(w.shape)}"
            )
        if not bool(torch.isfinite(w).all()):
            raise ValueError("weight entries must be finite")
        if bool((w < 0).any()):
            raise ValueError("weight entries must be non-negative")
        # Map every ignored target to class 0 before gathering: clamp_min(0) alone only
        # rescues *negative* sentinels, and ignore_index is caller-configurable, so a
        # non-negative sentinel (say 255) would index past the end of `w`. The multiply
        # by `valid` then zeroes whatever the ignored positions gathered.
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        w_t = w[safe_targets] * valid
        per_token = focal * w_t
        denom = w_t.sum()

    if reduction == "none":
        return per_token
    if reduction == "sum":
        return per_token.sum()
    # An all-ignored batch reduces to 0.0 rather than nan (the P1-06 clamp_min convention).
    return per_token.sum() / denom.clamp_min(torch.finfo(per_token.dtype).tiny)


# --------------------------------------------------------------------------- #
# Tensor tier — CRF wiring
# --------------------------------------------------------------------------- #
def left_align_for_crf(
    emissions: Tensor,
    tags: Tensor,
    mask: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Tensor, Tensor, Tensor]:
    """Roll each sequence so its contiguous run of real nucleotides starts at ``t=0``.

    :class:`~tbox_finder.models.seg_head.LinearChainCRF` requires a **left-aligned** mask
    (its ``_validate`` raises unless ``mask[:, 0]`` is all-True and no valid token follows
    padding) — its forward/Viterbi algorithms read ``mask.sum()`` as a contiguous prefix
    length. A P2-01 window zero-flanked at a **contig start** violates exactly that: the
    ignored positions sit at the *front*. This is not hypothetical — over the 8,303
    training-fold windows, **87 (1.05%) are zero-flanked at the start**, so wiring
    ``real_mask`` straight into the CRF raises on ~1 window in 95.

    The roll is exact rather than a fudge, on two measured facts: zero-flanking only ever
    occurs at a contig start/end, so the real run is a single **contiguous** block (0 of
    8,303 windows had an interior gap), and a circular roll preserves the order *within*
    that block. The wrapped-around pad lands in the tail where the mask zeroes it, and the
    CRF's end-transition is scored at ``mask.sum()-1`` — the last real nucleotide. The
    contiguity precondition is re-checked here per batch and raises if it ever fails,
    rather than trusting the measurement to hold forever.

    Also clamps ignored tags into range: the CRF has **no** ``ignore_index`` and indexes
    ``transitions[tags[:, t-1], tags[:, t]]`` at every timestep, so a raw ``-100`` would
    index out of bounds. Clamped tags sit only under masked-out positions, which
    contribute nothing to the score or the partition.

    Returns:
        ``(emissions, tags, mask)`` rolled so every sequence is left-aligned.
    """
    import torch  # lazy

    if emissions.dim() != 3:
        raise ValueError(f"emissions must be (B, L, C), got {tuple(emissions.shape)}")
    if tags.shape != emissions.shape[:2] or mask.shape != emissions.shape[:2]:
        raise ValueError(
            f"tags {tuple(tags.shape)} / mask {tuple(mask.shape)} must both be "
            f"{tuple(emissions.shape[:2])}"
        )
    mask = mask.bool()
    b, length, _ = emissions.shape

    n_valid = mask.sum(dim=1)  # (B,)
    if bool((n_valid == 0).any()):
        raise ValueError("every sequence must have at least one unmasked position")

    positions = torch.arange(length, device=mask.device).unsqueeze(0)  # (1, L)
    # First/last valid index per sequence: masked-out positions are pushed to +L / -1 so
    # they can never win the min / max.
    first = torch.where(mask, positions, torch.full_like(positions, length)).min(dim=1).values
    last = torch.where(mask, positions, torch.full_like(positions, -1)).max(dim=1).values
    if not bool((last - first + 1 == n_valid).all()):
        bad = int(((last - first + 1) != n_valid).nonzero()[0, 0].item())
        raise ValueError(
            f"sequence {bad} has a non-contiguous run of real nucleotides; the CRF models a "
            "single chain and cannot span an interior gap. P2-01 zero-flanks only at contig "
            "ends, so this indicates upstream corruption rather than a case to paper over."
        )

    # Circular roll by -first: gather index (i + first) % L.
    idx = (positions + first.unsqueeze(1)) % length  # (B, L)
    rolled_emissions = emissions.gather(1, idx.unsqueeze(-1).expand(-1, -1, emissions.shape[2]))
    rolled_tags = tags.gather(1, idx)
    rolled_mask = mask.gather(1, idx)

    # Post-condition: the roll must have produced exactly a left-aligned prefix.
    expected = positions < n_valid.unsqueeze(1)
    if not bool((rolled_mask == expected).all()):
        raise ValueError("internal error: roll did not left-align the mask")

    safe_tags = torch.where(rolled_tags == ignore_index, torch.zeros_like(rolled_tags), rolled_tags)
    if bool(((safe_tags < 0) | (safe_tags >= emissions.shape[2])).any()):
        raise ValueError("tags carry an out-of-range class index after ignore_index clamping")
    del b  # only shape-checked
    return rolled_emissions, safe_tags, rolled_mask


# --------------------------------------------------------------------------- #
# The composed Stage-1 objective
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage1LossConfig:
    """Stage-1 objective configuration.

    **No field here is ADR-pinned.** PRD §11 names γ a swept axis (P2-06) and the CRF a
    head-ablation axis (P2-12); ADR-0005 pins only the Stage-2 aux weighting (D16). These
    are implementer defaults on swept axes, recorded as such.

    Attributes:
        gamma: focal focusing parameter; ``0`` recovers cross-entropy.
        class_weight_alpha: inverse-frequency exponent; ``0`` ⇒ unweighted. Non-zero
            requires ``class_counts`` at construction.
        use_crf: add the optional CRF transition term (the caller supplies the module).
        crf_weight: weight on the per-token CRF NLL relative to the focal CE.
        ignore_index: target sentinel for positions carrying no DNA.
    """

    gamma: float = DEFAULT_GAMMA
    class_weight_alpha: float = DEFAULT_CLASS_WEIGHT_ALPHA
    use_crf: bool = False
    crf_weight: float = DEFAULT_CRF_WEIGHT
    ignore_index: int = IGNORE_INDEX

    def __post_init__(self) -> None:
        _finite_float(self.gamma, "gamma")
        _finite_float(self.class_weight_alpha, "class_weight_alpha")
        _finite_float(self.crf_weight, "crf_weight")


class Stage1Loss:
    """The Stage-1 per-nucleotide objective: focal CE (+ optional class weights, + optional CRF).

    Parameter-free by construction — the CRF's learnable transitions belong to
    ``SegmentationHead(use_crf=True)`` (P1-03), which the caller passes to
    :meth:`__call__`. That keeps this a plain callable rather than an ``nn.Module``, so
    the module needs no torch at import time.

    Example:
        >>> loss_fn = Stage1Loss()                       # focal γ=2, unweighted  # doctest: +SKIP
        >>> loss = loss_fn(logits, batch["labels"])                              # doctest: +SKIP
    """

    def __init__(
        self,
        config: Stage1LossConfig | None = None,
        *,
        class_counts: Mapping[str, int] | Sequence[int] | None = None,
    ) -> None:
        self.config = config or Stage1LossConfig()
        self.class_counts = None if class_counts is None else _counts_in_class_order(class_counts)
        alpha = float(self.config.class_weight_alpha)
        if alpha > 0.0:
            if self.class_counts is None:
                raise ValueError(
                    "class_weight_alpha > 0 needs class_counts: inverse-frequency weights are "
                    "derived from the measured training-stream class frequencies, never "
                    "assumed (CLAUDE.md §10.3)."
                )
            self.weights: tuple[float, ...] | None = class_weights_from_counts(
                self.class_counts, alpha=alpha
            )
        else:
            self.weights = None

    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        crf: Any = None,
        real_mask: Tensor | None = None,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute the Stage-1 loss over ``logits`` ``(B, L, C)`` and ``targets`` ``(B, L)``.

        Args:
            logits: per-position logits from ``SegmentationHead.forward``.
            targets: per-position class indices; ``ignore_index`` at zero-flanked positions.
            crf: the ``LinearChainCRF`` from the head; required iff ``config.use_crf``.
            real_mask: optional ``(B, L)`` bool marking real DNA. Defaults to
                ``targets != ignore_index``, which P2-01 guarantees is the same set.
            return_components: also return the individual terms (for W&B logging).
        """
        import torch  # lazy

        cfg = self.config
        ce = focal_cross_entropy(
            logits,
            targets,
            gamma=cfg.gamma,
            weight=self.weights,
            ignore_index=cfg.ignore_index,
            reduction="mean",
        )
        components: dict[str, Tensor] = {"focal_ce": ce}
        total = ce

        if cfg.use_crf:
            if crf is None:
                raise ValueError(
                    "config.use_crf is True but no crf module was supplied; pass the "
                    "SegmentationHead's .crf (the head owns the transition parameters)."
                )
            expected_mask = targets != cfg.ignore_index
            if real_mask is None:
                mask = expected_mask
            else:
                mask = real_mask.bool()
                # P2-01 emits IGNORE_INDEX at exactly the non-real positions, so the two
                # must agree. Check rather than trust: a mask marking an ignored target as
                # real would hand the CRF a -100 tag, which left_align_for_crf clamps to
                # class 0 — silently training the chain on a fabricated `background` label
                # at a position that carries no DNA. Cross-checking the datamodule's two
                # signals against each other costs nothing and turns a silent corruption
                # into an error (CLAUDE.md §10.3).
                if mask.shape != expected_mask.shape or not bool(
                    torch.equal(mask, expected_mask.to(mask.device))
                ):
                    raise ValueError(
                        "real_mask disagrees with (targets != ignore_index); the datamodule "
                        "must mark exactly the ignored positions as non-real."
                    )
            aligned_emissions, aligned_tags, aligned_mask = left_align_for_crf(
                logits, targets, mask, ignore_index=cfg.ignore_index
            )
            n_valid = aligned_mask.sum().to(logits.dtype)
            # Per-token NLL: the CRF's own reduction is per-*sequence*, whose magnitude
            # scales with L (~1024 here), so an un-normalised term would swamp the
            # per-token CE by ~3 orders of magnitude and make crf_weight uninterpretable.
            crf_nll = crf(aligned_emissions, aligned_tags, mask=aligned_mask, reduction="sum")
            crf_term = crf_nll / n_valid.clamp_min(torch.finfo(logits.dtype).tiny)
            components["crf_nll_per_token"] = crf_term
            total = total + float(cfg.crf_weight) * crf_term
        elif crf is not None:
            raise ValueError("a crf module was supplied but config.use_crf is False")

        components["total"] = total
        if return_components:
            return total, components
        return total

    def diagnostics(self) -> dict[str, Any]:
        """Reported (never gated) view of what this configuration does to the loss mass.

        Empty of mass statistics when no ``class_counts`` were supplied — this reports
        what was measured, and fabricates nothing when nothing was measured.
        """
        out: dict[str, Any] = {
            "gamma": float(self.config.gamma),
            "class_weight_alpha": float(self.config.class_weight_alpha),
            "use_crf": bool(self.config.use_crf),
            "crf_weight": float(self.config.crf_weight),
            "class_order": list(CLASS_ORDER),
            "weights": None if self.weights is None else list(self.weights),
            # Every value above is an implementer default on a swept axis (PRD §11 →
            # P2-06 for γ/α, P2-12 for the CRF); no ADR pins any of them.
            "pinned": False,
        }
        if self.class_counts is not None:
            out["class_counts"] = dict(zip(CLASS_ORDER, self.class_counts, strict=True))
            out["loss_mass_share"] = loss_mass_share(self.class_counts, self.weights)
            out["core_mass_share"] = core_mass_share(self.class_counts, self.weights)
            out["core_elements"] = list(CORE_ELEMENTS)
        return out
