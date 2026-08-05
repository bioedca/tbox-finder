"""P3-13 — the ADR-0005 D15 strand-resolver: predicted element order in, strand + orientation out.

Why a resolver exists at all
----------------------------
Stage 1 is **Caduceus-PS**, which is reverse-complement *equivariant*: a locus and its RC score
identically, so detection is strand-agnostic and the scan alone cannot say which strand a
candidate lies on (PRD §6). The orientation is instead read off the **predicted element order**
— a T-box leader runs 5′ Stem I (carrying the specifier) → … → the 3′
antiterminator/discriminator domain → terminator, so a locus whose predicted elements appear in
that order along the scanned sequence is on the ``+`` strand and one whose elements appear in
the *reverse* of it is on the ``-`` strand. This is the module that populates the §13.1 strand
column and its low-order-confidence flag.

That is also why ADR-0005 D15 constrains the §10.1 RC hidden-state combination to a
**directionality-preserving (non-averaged)** form, a constraint
:mod:`tbox_finder.models.rc_combine` already refuses to violate: an averaged combination is
invariant under the forward↔RC swap, which would destroy exactly the element order this module
reads.

The canonical order is MEASURED, not typed — and ``CLASS_ORDER`` is not it
--------------------------------------------------------------------------
:data:`CANONICAL_ELEMENT_ORDER` is re-derived from
``tests/fixtures/element_order/element_order_prevalence.json``, the pairwise measurement
``scripts/measure_element_order.py`` takes over all 23,535 records of the curated corpus
(``test_canonical_order_is_the_measured_order``). The tempting shortcut —
:data:`tbox_finder.labels.CLASS_ORDER` — is **wrong for this purpose**: it is the *softmax
index* order, and it places ``Discriminator`` last, whereas the corpus places it 5′ of the
antiterminator's midpoint in 99.99 % of the 23,208 records annotating both (the ~3-nt
discriminator sits inside the 5′ half of the ~30-nt antiterminator extent — the nesting PRD §8
documents). Borrowing it would have mis-ranked one element in seven, silently.

Every adjacency in the measured order rests on ≥ 99.7 % of the records annotating both members;
the weakest is ``Stem_I`` < ``Specifier`` at 0.997232 over 23,122 records, which is the
specifier loop sitting in the 3′ portion of Stem I rather than at its centre.

**Scientific-evidence gate (CLAUDE.md §10.1), high-stakes, ≥ 2 independent agreeing sources**
(accessed 2026-08-05): Zhang & Ferré-D'Amaré 2013 — "Stem I contains the specifier trinucleotide
… 3′ to stem I is the antiterminator domain" [PMID:23892783]; Suddala & Zhang 2019 — "a 5′ Stem
I domain … and a 3′ antiterminator/antisequestrator (or discriminator) domain" [PMID:31206978;
DOI:10.1002/iub.2098]; Zhang 2020 — "the Stem I, Stem II, and Discriminator domains, which
collectively compose the T-box riboswitches" [PMID:32633085; DOI:10.1002/wrna.1600]. The first
two already carry in-repo citations in PRD §3. Note what the literature does *not* say: it
places the discriminator **within** the 3′ antiterminator domain and defines no order between
the two, so ``Discriminator`` < ``Antiterminator_Tbox_seq`` here is a statement about the
corpus's annotation *extents*, not a claim about domain order — and it is load-bearing only in
the rare locus predicting one without the other.

The decision rule is a rank-concordance sign, and it pins no value
-------------------------------------------------------------------
Each element class observed in the locus **core** is reduced to the median position of its
predicted nucleotides — a midpoint for a contiguous run, and robust to ``Stem_I`` being split
into two runs around the ``Specifier`` nested inside it. Every pair of observed classes then
votes: **concordant** if their observed order matches :data:`CANONICAL_ELEMENT_ORDER`,
**discordant** if it matches the reverse. ``order_margin = n_concordant − n_discordant`` is the
Kendall-τ numerator over that comparison, and the strand is ``+`` when it reaches
``min_order_margin``, ``-`` when it reaches it negatively, ambiguous otherwise.

D15 names two ambiguous cases and the rule reproduces both without a tuned constant: a **single
confident element** yields no pair at all, and a **scrambled order** cancels concordant against
discordant. At ``min_order_margin=1`` the rule is exactly the sign of τ and has *no* free
parameter; higher values are the stricter reading of "scrambled". Which one production uses is
a §13.1 phase-gate freeze, so — as in :mod:`tbox_finder.infer.locus` — ``min_order_margin`` is
**keyword-only with no default** and this module pins no value
(:func:`no_rule_parameter_has_a_default`). The margin and both vote counts ride along on every
:class:`StrandCall`, so a later ADR can freeze a threshold against measured margins rather than
against an argument.

Ambiguity is recall-favouring: BOTH strands are carried
--------------------------------------------------------
An ambiguous locus is **not dropped**. It is flagged ``low_order_confidence`` and its
``strands`` field carries ``("+", "-")``, so the Stage-2 handoff emits an RNA for each and the
locus survives on whichever strand is real (PRD §6; ADR-0005 D15). A mis-resolution therefore
degrades to a **bounded false negative on divergent loci, never a false-novelty claim**; the
resolved strand is in any case corroborated downstream by strand-specific *model-independent*
signals (R-scape covariation and R2DT architecture pass only on the correct strand — §13.2 /
§13.3(c)).

``numpy``-only and torch-free, so it imports and unit-tests on the bare CI Tier-1 path, exactly
like :mod:`tbox_finder.infer.call`, :mod:`tbox_finder.infer.reconcile` and
:mod:`tbox_finder.infer.locus`.
PRD §6, §13.1; ADR-0005 D15.
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tbox_finder.infer.call import require_integer
from tbox_finder.infer.locus import (
    ELEMENT_CLASS_NAMES,
    NOT_ELEMENT,
    Locus,
    ThresholdScope,
    element_assignment,
)
from tbox_finder.labels import CLASS_INDEX

SCHEMA_VERSION = "1.0"
STEP = "P3-13"

#: Strand of the scanned sequence a locus reads along. ``"+"`` means the locus runs 5′→3′ along
#: the sequence **as scanned**; ``"-"`` means it runs 5′→3′ along that sequence's reverse
#: complement. Never an absolute genomic strand: it is relative to whatever was handed to the
#: scanner, and :mod:`tbox_finder.infer.handoff` reverse-complements on exactly this reading.
STRAND_PLUS = "+"
STRAND_MINUS = "-"

#: What an ambiguous locus carries — both strands, in a fixed order so the handoff it drives is
#: deterministic (PRD §6: "both strands are carried through for it").
BOTH_STRANDS: tuple[str, str] = (STRAND_PLUS, STRAND_MINUS)

#: The 5′→3′ element order, **derived** from the corpus measurement in
#: ``tests/fixtures/element_order/element_order_prevalence.json`` (all 23,535 records;
#: ``scripts/measure_element_order.py``), and checked against it by
#: ``test_canonical_order_is_the_measured_order`` so a re-measurement that moved an element
#: would fail rather than leave a stale constant here. See the module docstring for why
#: ``labels.CLASS_ORDER`` is *not* this order.
CANONICAL_ELEMENT_ORDER: tuple[str, ...] = (
    "Stem_I",
    "Specifier",
    "Stem_II",
    "Stem_III",
    "Discriminator",
    "Antiterminator_Tbox_seq",
    "Terminator",
)

#: The single rule parameter on :func:`resolve_strand`. Keyword-only with no default — ADR-0005
#: D15's ambiguity rule is frozen at the §13.1 phase gate, exactly like D3's locus values.
RULE_PARAMETERS: tuple[str, ...] = ("min_order_margin",)

#: Why a call came out the way it did. ``"resolved"`` is the only non-ambiguous value.
REASON_RESOLVED = "resolved"
REASON_SINGLE_ELEMENT = "single_element"
REASON_NO_COMPARABLE_PAIRS = "no_comparable_pairs"
REASON_MARGIN_BELOW_THRESHOLD = "margin_below_threshold"


class StrandError(ValueError):
    """Raised on a malformed strand-rule parameter or input.

    Distinct from :class:`tbox_finder.infer.locus.LocusError` and
    :class:`tbox_finder.infer.call.CandidateError` so a refusal is attributable to the layer
    that made it.
    """


def build_element_rank(
    order: Sequence[str] = CANONICAL_ELEMENT_ORDER,
    class_names: Sequence[str] = ELEMENT_CLASS_NAMES,
) -> dict[int, int]:
    """``{class index: 5′→3′ rank}`` from an element order, refusing an incomplete one.

    Fails closed when ``order`` is not exactly the element-class set: an element missing from
    the rank contributes **no pair** to the concordance vote, so its positions would silently
    stop informing orientation — a recall loss on the strand column with nothing recording it.
    An extra or misspelled name is refused for the same reason (it would rank a class that
    cannot occur, and mask the absence of one that can).

    The defaults are the shipped pairing; the parameters exist so the refusal can be exercised
    on a deliberately incomplete order rather than only on the compliant one.
    """
    missing = sorted(set(class_names) - set(order))
    unknown = sorted(set(order) - set(class_names))
    if missing or unknown:
        raise StrandError(
            "the canonical element order must rank exactly the element classes "
            f"{list(class_names)}; missing={missing}, unknown={unknown}. An unranked class "
            "would contribute no ordering evidence and silently weaken every strand call"
        )
    if len(set(order)) != len(order):
        raise StrandError(f"the canonical element order repeats a class: {list(order)}")
    return {CLASS_INDEX[name]: rank for rank, name in enumerate(order)}


#: class index → 5′→3′ rank. Built (not typed) from :data:`CANONICAL_ELEMENT_ORDER`, and
#: validated at import against the element-class set :mod:`tbox_finder.infer.locus` derives from
#: ``CLASS_ORDER``, so adding a ninth class without ranking it stops the scanner instead of
#: quietly degrading every strand call that class appears in.
ELEMENT_RANK: dict[int, int] = build_element_rank()


@dataclass(frozen=True)
class StrandCall:
    """The strand decision for one locus, with the vote that produced it.

    Attributes
    ----------
    strand:
        ``"+"`` / ``"-"``, or ``None`` when the order is under-determined. ``None`` is not a
        failure — it is D15's ambiguous case, and ``strands`` still carries work downstream.
    strands:
        The strands carried through to Stage-2: one element when resolved, ``("+", "-")`` when
        ambiguous. This, not ``strand``, is what the handoff iterates.
    low_order_confidence:
        ``True`` exactly when ``strand is None`` — the §13.1 flag. Kept as its own field
        because it is a *column* in the candidate table, not an inference a consumer should
        have to re-make from a null.
    reason:
        Which branch decided it: ``"resolved"``, ``"single_element"`` (no pair to vote),
        ``"no_comparable_pairs"`` (every observed pair shares a median position), or
        ``"margin_below_threshold"`` (votes cancelled, or fell short of ``min_order_margin``).
    n_concordant, n_discordant, n_tied_position:
        The pair votes. Concordant = observed order agrees with
        :data:`CANONICAL_ELEMENT_ORDER`; discordant = it agrees with the reverse; tied = the
        two classes have the same median position, so the pair casts no vote.
    order_margin:
        ``n_concordant − n_discordant``. Signed: positive favours ``+``. Reported, so a later
        phase-gate freeze can be argued from a distribution of measured margins.
    min_order_margin:
        The threshold this call was made under, echoed so a stored call says which rule
        produced it rather than leaving it to be inferred from a config.
    observed_order:
        Element-class indices ordered by median position within the core, ascending — the
        observed order the vote was taken over. Ties broken by class index, deterministically.
    element_median_positions:
        ``(class index, median position)`` pairs, positions relative to the **core** start.
        Reported so a surprising call can be read without re-running the assignment.
    n_distinct_elements:
        Number of element classes observed in the core. ``< 2`` is D15's "single confident
        element" ambiguity.
    """

    strand: str | None
    strands: tuple[str, ...]
    low_order_confidence: bool
    reason: str
    n_concordant: int
    n_discordant: int
    n_tied_position: int
    order_margin: int
    min_order_margin: int
    observed_order: tuple[int, ...]
    element_median_positions: tuple[tuple[int, float], ...]
    n_distinct_elements: int


def _core_assignment(locus: Locus, assignment: Any) -> np.ndarray:
    """The per-position element assignment over ``locus``'s **core**, validated.

    The core, not the flanked span: the flank is context handed to Stage-2, not evidence — the
    same split :func:`tbox_finder.infer.locus.construct_loci` applies to co-occurrence, so
    widening the flank cannot move a strand call either.
    """
    arr = np.asarray(assignment)
    if arr.ndim != 1:
        raise StrandError(f"assignment must be a 1-D (seq_len,) array, got shape={arr.shape}")
    if arr.dtype.kind not in "iu":
        raise StrandError(
            f"assignment must be an integer class array (element_assignment's output), "
            f"got dtype={arr.dtype}"
        )
    start, end = locus.candidate.start, locus.candidate.end
    if end > arr.shape[0]:
        raise StrandError(
            f"locus core [{start}, {end}) runs past the assignment ({arr.shape[0]} positions); "
            "this assignment is not the one the locus was called from"
        )
    core = arr[start:end]

    # Every label must be a ranked element class or NOT_ELEMENT. Both failure modes here are
    # fail-OPEN without this: an unknown label mixed with a known one reaches ``ELEMENT_RANK[...]``
    # and raises a bare ``KeyError`` from inside the vote, while a core carrying *only* unknown
    # labels never looks a rank up at all and returns a ``StrandCall`` naming a class that does
    # not exist — silently, as an ordinary ambiguous locus.
    #
    # The realistic way to get here is not a corrupt array but a plausible substitution:
    # ``Reconciled.prediction`` is the arg-max over all NUM_CLASSES (the quantity ``eval/gate4``
    # grades boundary IoU on), so it carries ``background`` as 0 — which is not an element and
    # has no rank. Passing it instead of ``element_assignment``'s output currently dies with
    # ``KeyError: 0``. Named explicitly, because the two arrays are the same shape and dtype.
    unknown = sorted({int(c) for c in np.unique(core)} - set(ELEMENT_RANK) - {NOT_ELEMENT})
    if unknown:
        raise StrandError(
            f"assignment carries label(s) {unknown} that are neither NOT_ELEMENT "
            f"({NOT_ELEMENT}) nor a ranked element class {sorted(ELEMENT_RANK)}. If "
            f"{CLASS_INDEX['background']} appears, this is most likely Reconciled.prediction "
            "(the arg-max over all classes, which encodes background) where "
            "element_assignment's output was wanted — the two are the same shape and dtype"
        )
    return core


def resolve_strand(locus: Locus, assignment: Any, *, min_order_margin: int) -> StrandCall:
    """Resolve a locus's strand from its predicted element order (ADR-0005 D15).

    Parameters
    ----------
    locus:
        A :class:`tbox_finder.infer.locus.Locus` from ``construct_loci``.
    assignment:
        The ``(seq_len,)`` per-position element assignment
        :func:`tbox_finder.infer.locus.element_assignment` produced for the **same** sequence
        under the **same** threshold scope and value the loci were built with.
        :func:`resolve_strands` exists so a caller does not have to keep those in step by hand.
    min_order_margin:
        Minimum ``|n_concordant − n_discordant|`` required to name a strand; ``>= 1``. ``1`` is
        the parameter-free sign-of-τ rule. No default: the value is a §13.1 phase-gate freeze.

    Returns
    -------
    StrandCall
        Resolved or ambiguous — never dropped. See :class:`StrandCall`.

    Raises
    ------
    StrandError
        On a ``min_order_margin`` below 1 or not a whole number, or an ``assignment`` that is
        not the integer class array this locus was called from.
    """
    min_order_margin = require_integer("min_order_margin", min_order_margin, StrandError)
    if min_order_margin < 1:
        raise StrandError(
            f"min_order_margin must be >= 1, got {min_order_margin}; at 0 a locus whose "
            "concordant and discordant votes cancel exactly — D15's scrambled-order case — "
            "would be assigned a strand rather than carried on both"
        )

    core = _core_assignment(locus, assignment)

    positions: list[tuple[int, float]] = []
    for cls in np.unique(core):
        cls = int(cls)
        if cls == NOT_ELEMENT:
            continue
        positions.append((cls, float(np.median(np.flatnonzero(core == cls)))))
    # Ascending median position; class index breaks a tie so the order is deterministic.
    positions.sort(key=lambda item: (item[1], item[0]))
    observed_order = tuple(cls for cls, _ in positions)

    n_concordant = n_discordant = n_tied = 0
    for (cls_a, pos_a), (cls_b, pos_b) in itertools.combinations(positions, 2):
        if pos_a == pos_b:
            n_tied += 1
            continue
        # Both classes are ranked: build_element_rank refuses a partial order at import.
        forward = (pos_a < pos_b) == (ELEMENT_RANK[cls_a] < ELEMENT_RANK[cls_b])
        if forward:
            n_concordant += 1
        else:
            n_discordant += 1

    margin = n_concordant - n_discordant
    n_distinct = len(positions)

    if n_distinct < 2:
        strand, reason = None, REASON_SINGLE_ELEMENT
    elif n_concordant + n_discordant == 0:
        strand, reason = None, REASON_NO_COMPARABLE_PAIRS
    elif margin >= min_order_margin:
        strand, reason = STRAND_PLUS, REASON_RESOLVED
    elif -margin >= min_order_margin:
        strand, reason = STRAND_MINUS, REASON_RESOLVED
    else:
        strand, reason = None, REASON_MARGIN_BELOW_THRESHOLD

    return StrandCall(
        strand=strand,
        strands=BOTH_STRANDS if strand is None else (strand,),
        low_order_confidence=strand is None,
        reason=reason,
        n_concordant=n_concordant,
        n_discordant=n_discordant,
        n_tied_position=n_tied,
        order_margin=margin,
        min_order_margin=min_order_margin,
        observed_order=observed_order,
        element_median_positions=tuple(positions),
        n_distinct_elements=n_distinct,
    )


def resolve_strands(
    log_probs: Any,
    loci: Sequence[Locus],
    *,
    threshold_scope: ThresholdScope,
    threshold: float | Mapping[str, float],
    min_order_margin: int,
) -> list[StrandCall]:
    """Resolve every locus in one pass — the recommended entry point.

    The per-position assignment is computed **once**, from the same
    :func:`tbox_finder.infer.locus.element_assignment` and the same scope/threshold that
    ``construct_loci`` was given. Grading element order under a *different* mask than the one
    that built the loci is silent and plausible-looking (the spans still exist, the classes
    still exist, only the evidence changed), so this function exists to make the consistent
    path the easy one.
    """
    assignment = element_assignment(log_probs, threshold_scope=threshold_scope, threshold=threshold)
    return [resolve_strand(locus, assignment, min_order_margin=min_order_margin) for locus in loci]


def no_rule_parameter_has_a_default(func: Any = None) -> bool:
    """True iff every D15 strand-rule knob on :func:`resolve_strand` is keyword-only, no default.

    The same discipline :func:`tbox_finder.infer.locus.no_rule_parameter_has_a_default` enforces
    for D3, and for the same reason: a default here would be a de-facto frozen ambiguity rule
    that no ADR signed off and that a caller could take by accident. Re-derived from the live
    signature, and compared as a **set** so a knob added later without a matching
    :data:`RULE_PARAMETERS` entry fails the inventory instead of being waved through.

    ``func`` exists so the predicate can be pointed at a stub that *does* carry a default and
    shown to return False.
    """
    params = inspect.signature(resolve_strand if func is None else func).parameters
    keyword_only = {
        name: p for name, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    if set(keyword_only) != set(RULE_PARAMETERS):
        return False
    return all(p.default is inspect.Parameter.empty for p in keyword_only.values())


__all__ = [
    "BOTH_STRANDS",
    "CANONICAL_ELEMENT_ORDER",
    "ELEMENT_RANK",
    "REASON_MARGIN_BELOW_THRESHOLD",
    "REASON_NO_COMPARABLE_PAIRS",
    "REASON_RESOLVED",
    "REASON_SINGLE_ELEMENT",
    "RULE_PARAMETERS",
    "SCHEMA_VERSION",
    "STEP",
    "STRAND_MINUS",
    "STRAND_PLUS",
    "StrandCall",
    "StrandError",
    "build_element_rank",
    "no_rule_parameter_has_a_default",
    "resolve_strand",
    "resolve_strands",
]
