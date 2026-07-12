"""RC-combination mode contract (P1-04) — torch-free so bare CI can lock it.

The Stage-1 segmenter combines the Caduceus-PS ``2*d_model`` RC-concatenated hidden state
(forward-strand channels ``||`` reverse-complement-strand channels) before the seg head.
PRD §6/§10.1 (ADR-0002) **constrain that combination to a directionality-preserving
(non-averaged) form**: the §6 strand-resolver derives strand + 5′→3′ orientation from the
**predicted element order** (Specifier / Stem I → antiterminator / terminator), and an
order-destroying *average* of the two strand-halves — a form symmetric under swapping the
forward and RC channels — collapses that order, defeating strand resolution.

These pure predicates/constants carry no ``torch`` dependency (unlike the ``nn.Module``
combinations in :mod:`tbox_finder.models.stage1_segmenter`), so the **bare CI test tier**
can lock the load-bearing contract — *mean is rejected, ``concat`` is the default, the
allowed set is exactly the non-averaged forms* — without the GPU env.
"""

from __future__ import annotations

#: Directionality-preserving (non-averaged) RC-combination modes — the only forms PRD
#: §6/§10.1 admit. ``"concat"`` feeds the full ``(fwd || rc)`` hidden state to the head
#: (injective); ``"gate"`` is a learned directional per-channel gate. Both are *not*
#: invariant under the forward↔RC channel-half swap, so element order survives.
ALLOWED_RC_COMBINE: tuple[str, ...] = ("concat", "gate")

#: Order-destroying (RC-invariant) names — a symmetric average of the two strand-halves.
#: Rejected: such a form is invariant under swapping the fwd/RC channels, so it collapses the
#: element-order signal the §6 strand-resolver reads. Never selectable, never the default.
FORBIDDEN_RC_COMBINE: tuple[str, ...] = ("mean", "average", "avg")

#: Default RC-combination mode (PRD §10.1: the head consumes the full RC-concatenated hidden
#: state — directionality-preserving by construction; ADR-0002).
DEFAULT_RC_COMBINE: str = "concat"


def normalize_rc_combine(mode: str) -> str:
    """Return the canonical ``rc_combine`` mode, rejecting order-destroying / unknown forms.

    Raises ``ValueError`` for a symmetric-average form (:data:`FORBIDDEN_RC_COMBINE`), with a
    message pointing at the PRD §6/§10.1 directionality constraint, and for any mode outside
    :data:`ALLOWED_RC_COMBINE`. Case- and whitespace-insensitive.
    """
    m = str(mode).strip().lower()
    if m in FORBIDDEN_RC_COMBINE:
        raise ValueError(
            f"rc_combine {mode!r} is an order-destroying (RC-invariant) average and is "
            "disallowed: the §6 strand-resolver derives orientation from predicted element "
            "order, which a symmetric fwd/RC average collapses (PRD §6/§10.1; ADR-0002). "
            f"Use a directionality-preserving form: {ALLOWED_RC_COMBINE}."
        )
    if m not in ALLOWED_RC_COMBINE:
        raise ValueError(f"unknown rc_combine mode {mode!r}; allowed: {ALLOWED_RC_COMBINE}")
    return m


def is_directionality_preserving(mode: str) -> bool:
    """True iff ``mode`` is an allowed non-averaged RC-combination (never raises)."""
    return str(mode).strip().lower() in ALLOWED_RC_COMBINE
