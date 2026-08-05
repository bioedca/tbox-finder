"""P3-13 — the ADR-0005 D15 strand-resolver and the PRD §6 alphabet handoff.

The imp.md P3-13 gate is three claims, and each has its own section below:

* **element order → orientation** — a locus whose predicted elements run in the measured 5′→3′
  order is called ``+``; the same locus read on the other strand is called ``-``;
* **ambiguous → both-strand carry-through** — a single confident element, or a scrambled
  order, is flagged ``low_order_confidence`` and yields a payload on *each* strand;
* **exact T→U transcription** — hand-checked, length-preserving, and delegated to the shipped
  ``stage2.tokenizer.transcribe`` rather than re-derived.

Two more sections carry the load the gate does not name. ``CANONICAL_ELEMENT_ORDER`` is checked
against the committed corpus measurement (so the constant cannot go stale, and cannot quietly
become ``labels.CLASS_ORDER``, which disagrees with it), and the reverse complement is checked
over the **full IUPAC alphabet** against the ``ACGTN``-only behaviour the rest of the repo has —
because on this path that behaviour is a silently wrong base handed to Stage-2.

Fixtures are built from explicit per-position distributions and every layout is **verified to
produce the assignment it claims** (``_assert_layout_realised``): P3-12 lost a debugging pass to
an all-zero logit tensor that log-softmaxes to a uniform distribution, where ``1 − P(background)
= 0.875`` everywhere and the whole contig comes back as one locus. A degenerate oracle reads
exactly like an operator defect.

``numpy`` is a hard dependency of the modules under test, so it is imported directly rather than
``importorskip``-ed — this file must not skip green in the bare CI tier.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tbox_finder.infer.call import ELEMENT_INDICES, NUM_CLASSES
from tbox_finder.infer.handoff import (
    IUPAC_COMPLEMENT,
    IUPAC_COMPLEMENT_UPPER,
    Handoff,
    HandoffError,
    handoff_is_sequence_only,
    handoff_loci,
    reverse_complement,
    transcribe_to_rna,
)
from tbox_finder.infer.locus import ELEMENT_CLASS_NAMES, construct_loci, element_assignment
from tbox_finder.infer.strand import (
    BOTH_STRANDS,
    CANONICAL_ELEMENT_ORDER,
    ELEMENT_RANK,
    REASON_MARGIN_BELOW_THRESHOLD,
    REASON_NO_COMPARABLE_PAIRS,
    REASON_RESOLVED,
    REASON_SINGLE_ELEMENT,
    RULE_PARAMETERS,
    STRAND_MINUS,
    STRAND_PLUS,
    StrandError,
    build_element_rank,
    no_rule_parameter_has_a_default,
    resolve_strand,
    resolve_strands,
)
from tbox_finder.labels import CLASS_INDEX, CLASS_ORDER
from tbox_finder.stage2 import tokenizer as rna_tokenizer

_REPO = Path(__file__).resolve().parents[2]
_MEASUREMENT = _REPO / "tests" / "fixtures" / "element_order" / "element_order_prevalence.json"
_ORDER_SCRIPT = _REPO / "scripts" / "measure_element_order.py"

# The threshold pairing every fixture below is called under. Test-local: ADR-0005 D3 freezes the
# production scope/τ at the §13.1 phase gate, and nothing here is a claim about that value.
_SCOPE = "global"
_TAU = 0.5


def _load_order_script():
    spec = importlib.util.spec_from_file_location("measure_element_order", _ORDER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #
def _log_probs_from_layout(seq_len: int, layout: list[tuple[str, int, int]]) -> np.ndarray:
    """Reconciled-shaped ``(seq_len, NUM_CLASSES)`` log-posteriors for an element layout.

    ``layout`` entries are ``(class name, start, end)`` half-open and painted **in order**, so
    a later entry nested inside an earlier one overwrites it — which is how the corpus's real
    geometry looks (the Specifier carves out of Stem I; the Discriminator out of the
    antiterminator, PRD §8).

    Background positions get ``P(background) = 0.99`` and element positions ``P(class) = 0.90``
    with ``P(background) = 0.02``, so ``1 − P(background)`` is 0.01 against 0.98 — far either
    side of :data:`_TAU`, and the arg-max at an element position is unambiguous.
    """
    n_elem = len(ELEMENT_INDICES)
    probs = np.full((seq_len, NUM_CLASSES), 0.01 / n_elem, dtype=np.float64)
    probs[:, CLASS_INDEX["background"]] = 0.99

    for name, start, end in layout:
        probs[start:end, :] = 0.08 / (n_elem - 1)
        probs[start:end, CLASS_INDEX["background"]] = 0.02
        probs[start:end, CLASS_INDEX[name]] = 0.90

    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    return np.log(probs)


def _assert_layout_realised(log_probs: np.ndarray, layout: list[tuple[str, int, int]]) -> None:
    """The fixture really produces the classes it claims — not a uniform-distribution artifact.

    Without this, a builder bug that made every position element-like (or every element the
    same class) would be indistinguishable from a resolver that ignores its input.
    """
    assignment = element_assignment(log_probs, threshold_scope=_SCOPE, threshold=_TAU)
    expected = np.full(log_probs.shape[0], -1, dtype=np.int64)
    for name, start, end in layout:
        expected[start:end] = CLASS_INDEX[name]
    np.testing.assert_array_equal(assignment, expected)


def _loci_and_calls(log_probs, *, min_order_margin=1, flank=0, gap_merge=12):
    loci = construct_loci(
        log_probs,
        threshold_scope=_SCOPE,
        threshold=_TAU,
        min_span=1,
        gap_merge=gap_merge,
        min_distinct_elements=0,
        flank=flank,
    )
    calls = resolve_strands(
        log_probs,
        loci,
        threshold_scope=_SCOPE,
        threshold=_TAU,
        min_order_margin=min_order_margin,
    )
    return loci, calls


#: A full canonical class-I geometry, 5′→3′, with the two real nestings: the Specifier inside
#: Stem I and the Discriminator inside the antiterminator. Element runs are separated by 5-nt
#: background gaps, which ``gap_merge=12`` bridges into one core.
CANONICAL_LAYOUT: list[tuple[str, int, int]] = [
    ("Stem_I", 0, 40),
    ("Specifier", 20, 26),
    ("Stem_II", 45, 60),
    ("Stem_III", 65, 80),
    ("Antiterminator_Tbox_seq", 85, 100),
    ("Discriminator", 86, 89),
    ("Terminator", 105, 130),
]
CANONICAL_SEQ_LEN = 160


# --------------------------------------------------------------------------- #
# 1. the canonical order is MEASURED, and it is not CLASS_ORDER
# --------------------------------------------------------------------------- #
def test_canonical_order_is_the_measured_order():
    """The shipped constant is re-derived from the committed 23,535-record measurement.

    A hand-typed rank is free to go stale against the corpus it claims to describe; this
    re-runs the script's own Copeland derivation over the committed pairwise matrix.
    """
    report = json.loads(_MEASUREMENT.read_text())
    derived = _load_order_script().derive_order(report)
    assert derived == CANONICAL_ELEMENT_ORDER
    assert tuple(report["derived_order_5p_to_3p"]) == CANONICAL_ELEMENT_ORDER


def test_measurement_covers_the_whole_corpus_and_every_element():
    report = json.loads(_MEASUREMENT.read_text())
    assert report["n_records"] == 23535
    assert {e["element"] for e in report["elements"]} == set(ELEMENT_CLASS_NAMES)
    # Every adjacency in the shipped order rests on a strong majority, not a coin flip.
    frac = {(p["A"], p["B"]): p["frac_a_before_b"] for p in report["pairs"] if p["n_both"] > 0}
    for a, b in zip(CANONICAL_ELEMENT_ORDER, CANONICAL_ELEMENT_ORDER[1:], strict=False):
        support = frac.get((a, b))
        if support is None:  # the pair is recorded in the other direction
            support = 1.0 - frac[(b, a)]
        assert support > 0.99, f"{a} < {b} rests on only {support}"


def test_derive_order_refuses_anything_short_of_a_strict_total_order():
    """The derivation's three refusals, none of which the real corpus exercises.

    The committed report happens to give a clean tournament, so removing any of these guards
    changes nothing on it — which is exactly why they need their own fixtures. A silently
    arbitrary tie-break would put an *unmeasured* ordering into the strand-resolver.
    """
    derive_order = _load_order_script().derive_order

    def report(pairs):
        return {
            "elements": [{"element": n} for n in ("A", "B", "C")],
            "pairs": pairs,
        }

    def pair(a, b, ab, ba):
        return {"A": a, "B": b, "n_both": ab + ba, "n_a_before_b": ab, "n_b_before_a": ba}

    # Positive control: a clean tournament resolves.
    clean = report([pair("A", "B", 10, 0), pair("A", "C", 10, 0), pair("B", "C", 10, 0)])
    assert derive_order(clean) == ("A", "B", "C")

    with pytest.raises(ValueError, match="exactly tied"):
        derive_order(report([pair("A", "B", 5, 5), pair("A", "C", 10, 0), pair("B", "C", 10, 0)]))

    with pytest.raises(ValueError, match="no record annotates both"):
        derive_order(report([pair("A", "B", 0, 0), pair("A", "C", 10, 0), pair("B", "C", 10, 0)]))

    # A rock-paper-scissors cycle: every pair decides, no total order exists.
    with pytest.raises(ValueError, match="strict total order"):
        derive_order(report([pair("A", "B", 10, 0), pair("B", "C", 10, 0), pair("A", "C", 0, 10)]))


def test_canonical_order_is_not_class_order_and_differs_at_the_discriminator():
    """``labels.CLASS_ORDER`` is a softmax index order — borrowing it mis-ranks one element.

    This is the whole reason the rank is measured. ``CLASS_ORDER`` puts ``Discriminator`` last;
    the corpus puts it 5′ of the antiterminator's midpoint in ~99.99 % of the records
    annotating both, because the ~3-nt discriminator sits in the 5′ half of the ~30-nt
    antiterminator extent.
    """
    class_order_elements = tuple(n for n in CLASS_ORDER if n != "background")
    assert class_order_elements != CANONICAL_ELEMENT_ORDER
    assert CANONICAL_ELEMENT_ORDER.index("Discriminator") < CANONICAL_ELEMENT_ORDER.index(
        "Antiterminator_Tbox_seq"
    )
    assert class_order_elements.index("Discriminator") > class_order_elements.index(
        "Antiterminator_Tbox_seq"
    )
    # …and the measurement is what says so.
    report = json.loads(_MEASUREMENT.read_text())
    pair = next(
        p
        for p in report["pairs"]
        if {p["A"], p["B"]} == {"Antiterminator_Tbox_seq", "Discriminator"}
    )
    assert pair["A"] == "Antiterminator_Tbox_seq"
    assert pair["frac_a_before_b"] < 0.001
    assert pair["n_both"] > 20000


def test_element_rank_covers_exactly_the_element_classes():
    assert set(ELEMENT_RANK) == {int(i) for i in ELEMENT_INDICES}
    assert sorted(ELEMENT_RANK.values()) == list(range(len(ELEMENT_CLASS_NAMES)))
    assert "background" not in CANONICAL_ELEMENT_ORDER


def test_build_element_rank_accepts_the_shipped_pairing():
    """Positive control for the three refusals below — the same input must succeed."""
    rank = build_element_rank()
    assert rank[CLASS_INDEX["Stem_I"]] == 0
    assert rank[CLASS_INDEX["Terminator"]] == len(ELEMENT_CLASS_NAMES) - 1


def test_build_element_rank_refuses_an_incomplete_order():
    with pytest.raises(StrandError, match="missing="):
        build_element_rank(CANONICAL_ELEMENT_ORDER[:-1])


def test_build_element_rank_refuses_an_unknown_class():
    with pytest.raises(StrandError, match="unknown="):
        build_element_rank((*CANONICAL_ELEMENT_ORDER, "Stem_IV"))


def test_build_element_rank_refuses_a_repeated_class():
    repeated = ("Stem_I", *CANONICAL_ELEMENT_ORDER[1:-1], "Stem_I")
    with pytest.raises(StrandError):
        build_element_rank(repeated)


# --------------------------------------------------------------------------- #
# 2. element order → orientation
# --------------------------------------------------------------------------- #
def test_canonical_order_resolves_to_the_plus_strand():
    log_probs = _log_probs_from_layout(CANONICAL_SEQ_LEN, CANONICAL_LAYOUT)
    _assert_layout_realised(log_probs, CANONICAL_LAYOUT)

    loci, calls = _loci_and_calls(log_probs)
    assert len(loci) == 1
    call = calls[0]
    assert call.strand == STRAND_PLUS
    assert call.strands == (STRAND_PLUS,)
    assert call.low_order_confidence is False
    assert call.reason == REASON_RESOLVED
    assert call.n_distinct_elements == 7
    # All 21 pairs vote, and all 21 agree with the canonical rank.
    assert (call.n_concordant, call.n_discordant, call.n_tied_position) == (21, 0, 0)
    assert call.order_margin == 21
    assert call.observed_order == tuple(CLASS_INDEX[name] for name in CANONICAL_ELEMENT_ORDER)


def test_reverse_complement_of_the_locus_resolves_to_the_minus_strand():
    """Caduceus-PS is RC-equivariant: scanning the RC gives the row-reversed posteriors.

    So the *same* locus, presented on the other strand, must come back ``-`` with an equal and
    opposite margin. This is the property the resolver exists to supply, and it is asserted on
    the reversal of the real fixture rather than on a second hand-built layout.
    """
    log_probs = _log_probs_from_layout(CANONICAL_SEQ_LEN, CANONICAL_LAYOUT)
    rc_log_probs = log_probs[::-1].copy()

    _, calls = _loci_and_calls(rc_log_probs)
    call = calls[0]
    assert call.strand == STRAND_MINUS
    assert call.strands == (STRAND_MINUS,)
    assert call.low_order_confidence is False
    assert (call.n_concordant, call.n_discordant) == (0, 21)
    assert call.order_margin == -21
    assert call.observed_order == tuple(
        CLASS_INDEX[name] for name in reversed(CANONICAL_ELEMENT_ORDER)
    )


def test_two_element_locus_resolves_on_a_margin_of_one():
    layout = [("Stem_I", 10, 40), ("Terminator", 50, 70)]
    log_probs = _log_probs_from_layout(100, layout)
    _assert_layout_realised(log_probs, layout)

    _, calls = _loci_and_calls(log_probs)
    assert calls[0].strand == STRAND_PLUS
    assert calls[0].order_margin == 1


def test_strand_is_read_from_the_core_not_the_flank():
    """Widening the flank cannot move a strand call — the flank is context, not evidence.

    The same split ``construct_loci`` applies to co-occurrence. If orientation were read over
    the flanked span, a locus near a neighbouring element would be oriented by evidence that is
    not its own.

    The fixture is built so that reading the flanked span **would** change the answer: a
    single-element locus (ambiguous by D15) sits 30 nt from an unrelated element, so a flank of
    40 pulls that element into the span and would resolve the locus on a neighbour's evidence.
    A fixture whose flank is pure background cannot test this at all — every median shifts by
    the same offset and the order is unchanged, so the assertion passes whichever span is read.
    """
    layout = [("Stem_I", 20, 50), ("Terminator", 80, 110)]
    log_probs = _log_probs_from_layout(200, layout)
    _assert_layout_realised(log_probs, layout)

    narrow_loci, narrow = _loci_and_calls(log_probs, flank=0)
    wide_loci, wide = _loci_and_calls(log_probs, flank=40)
    assert len(narrow_loci) == len(wide_loci) == 2, "the two elements must stay separate loci"
    # The wide flank really does reach the neighbour — otherwise this proves nothing.
    assert wide_loci[0].end > narrow_loci[1].start
    assert narrow[0].strand is None and narrow[0].reason == REASON_SINGLE_ELEMENT
    assert narrow == wide

    # And the same invariance on the fully canonical locus, where the flank is background.
    canonical = _log_probs_from_layout(CANONICAL_SEQ_LEN, CANONICAL_LAYOUT)
    _, c_narrow = _loci_and_calls(canonical, flank=0)
    _, c_wide = _loci_and_calls(canonical, flank=25)
    assert c_narrow[0] == c_wide[0]


def test_median_position_survives_stem_i_being_split_around_the_specifier():
    """The Specifier nests *inside* Stem I, so Stem I is two runs — the statistic must cope.

    A ``last-occurrence`` statistic would place Stem I *after* the Specifier here and cast a
    discordant vote on a perfectly canonical locus; the shipped median does not. Both are
    computed so the choice is demonstrated, not asserted.
    """
    layout = [("Stem_I", 0, 40), ("Specifier", 30, 36)]
    log_probs = _log_probs_from_layout(60, layout)
    _assert_layout_realised(log_probs, layout)

    _, calls = _loci_and_calls(log_probs)
    call = calls[0]
    assert call.strand == STRAND_PLUS
    positions = dict(call.element_median_positions)
    assert positions[CLASS_INDEX["Stem_I"]] < positions[CLASS_INDEX["Specifier"]]

    assignment = element_assignment(log_probs, threshold_scope=_SCOPE, threshold=_TAU)
    last_stem_i = np.flatnonzero(assignment == CLASS_INDEX["Stem_I"]).max()
    last_specifier = np.flatnonzero(assignment == CLASS_INDEX["Specifier"]).max()
    assert last_stem_i > last_specifier, "the fixture must actually exercise the split"


def test_resolve_strands_matches_per_locus_resolution_on_the_same_assignment():
    """The batch entry is the same rule, computed once — not a second implementation."""
    log_probs = _log_probs_from_layout(CANONICAL_SEQ_LEN, CANONICAL_LAYOUT)
    loci, batched = _loci_and_calls(log_probs)
    assignment = element_assignment(log_probs, threshold_scope=_SCOPE, threshold=_TAU)
    one_by_one = [resolve_strand(locus, assignment, min_order_margin=1) for locus in loci]
    assert batched == one_by_one


def test_two_loci_are_resolved_independently():
    """A minus-strand locus beside a plus-strand one must not contaminate it."""
    layout = [
        ("Stem_I", 10, 40),
        ("Terminator", 50, 70),
        ("Terminator", 200, 220),
        ("Stem_I", 230, 260),
    ]
    log_probs = _log_probs_from_layout(300, layout)
    _assert_layout_realised(log_probs, layout)

    loci, calls = _loci_and_calls(log_probs)
    assert len(loci) == 2
    assert [c.strand for c in calls] == [STRAND_PLUS, STRAND_MINUS]


# --------------------------------------------------------------------------- #
# 3. ambiguous → flagged and carried on BOTH strands
# --------------------------------------------------------------------------- #
def test_single_confident_element_is_ambiguous_and_carried_on_both_strands():
    """D15's first ambiguous case: one element casts no ordering vote at all."""
    layout = [("Stem_I", 20, 60)]
    log_probs = _log_probs_from_layout(100, layout)
    _assert_layout_realised(log_probs, layout)

    _, calls = _loci_and_calls(log_probs)
    call = calls[0]
    assert call.strand is None
    assert call.strands == BOTH_STRANDS
    assert call.low_order_confidence is True
    assert call.reason == REASON_SINGLE_ELEMENT
    assert call.n_distinct_elements == 1
    assert (call.n_concordant, call.n_discordant) == (0, 0)


def test_scrambled_order_is_ambiguous_and_carried_on_both_strands():
    """D15's second ambiguous case: concordant and discordant votes cancel exactly.

    Observed order Specifier → Stem_III → Stem_I → Stem_II is 3 of 6 pairs inverted against the
    canonical rank, so the margin is 0 and no strand is named.
    """
    layout = [
        ("Specifier", 0, 10),
        ("Stem_III", 20, 30),
        ("Stem_I", 40, 50),
        ("Stem_II", 60, 70),
    ]
    log_probs = _log_probs_from_layout(90, layout)
    _assert_layout_realised(log_probs, layout)

    _, calls = _loci_and_calls(log_probs)
    call = calls[0]
    assert call.n_distinct_elements == 4
    assert (call.n_concordant, call.n_discordant) == (3, 3)
    assert call.order_margin == 0
    assert call.strand is None
    assert call.strands == BOTH_STRANDS
    assert call.low_order_confidence is True
    assert call.reason == REASON_MARGIN_BELOW_THRESHOLD


def test_coincident_medians_cast_no_vote_and_are_counted():
    """Two classes sharing a median position are tied, not silently ordered by class index."""
    layout = [("Stem_I", 20, 31), ("Specifier", 25, 26)]
    log_probs = _log_probs_from_layout(60, layout)
    _assert_layout_realised(log_probs, layout)

    _, calls = _loci_and_calls(log_probs)
    call = calls[0]
    positions = dict(call.element_median_positions)
    assert positions[CLASS_INDEX["Stem_I"]] == positions[CLASS_INDEX["Specifier"]]
    assert (call.n_concordant, call.n_discordant, call.n_tied_position) == (0, 0, 1)
    assert call.strand is None
    assert call.reason == REASON_NO_COMPARABLE_PAIRS
    assert call.strands == BOTH_STRANDS


def test_a_higher_margin_threshold_moves_a_weak_call_to_ambiguous():
    """``min_order_margin`` is the stricter reading of "scrambled", and it bites.

    The same locus resolves at 1 and is carried on both strands at 2 — which is why the value
    is a §13.1 phase-gate freeze and not a default in the module.
    """
    layout = [("Stem_I", 10, 40), ("Terminator", 50, 70)]
    log_probs = _log_probs_from_layout(100, layout)

    _, at_one = _loci_and_calls(log_probs, min_order_margin=1)
    _, at_two = _loci_and_calls(log_probs, min_order_margin=2)
    assert at_one[0].strand == STRAND_PLUS
    assert at_two[0].strand is None
    assert at_two[0].strands == BOTH_STRANDS
    assert at_two[0].reason == REASON_MARGIN_BELOW_THRESHOLD
    # The evidence is identical; only the rule moved.
    assert at_one[0].order_margin == at_two[0].order_margin == 1
    assert (at_one[0].min_order_margin, at_two[0].min_order_margin) == (1, 2)


def test_low_order_confidence_is_exactly_the_unresolved_case():
    """The §13.1 flag and a null strand are one fact, never two that can disagree."""
    layouts = [
        [("Stem_I", 20, 60)],
        [("Stem_I", 10, 40), ("Terminator", 50, 70)],
        [("Specifier", 0, 10), ("Stem_III", 20, 30), ("Stem_I", 40, 50), ("Stem_II", 60, 70)],
    ]
    for layout in layouts:
        log_probs = _log_probs_from_layout(90, layout)
        _, calls = _loci_and_calls(log_probs)
        for call in calls:
            assert call.low_order_confidence == (call.strand is None)
            assert call.strands == (BOTH_STRANDS if call.strand is None else (call.strand,))
            assert (call.reason == REASON_RESOLVED) == (call.strand is not None)


# --------------------------------------------------------------------------- #
# 4. the rule pins no value
# --------------------------------------------------------------------------- #
def test_no_rule_parameter_has_a_default():
    assert no_rule_parameter_has_a_default() is True


def test_no_rule_parameter_has_a_default_rejects_a_defaulted_stub():
    def stub(locus, assignment, *, min_order_margin: int = 1):  # pragma: no cover - signature
        raise AssertionError

    assert no_rule_parameter_has_a_default(stub) is False


def test_no_rule_parameter_has_a_default_rejects_an_unlisted_knob():
    """A knob added later without a :data:`RULE_PARAMETERS` entry fails the inventory.

    The set comparison — not a subset — is the half that catches a *new* parameter, which has
    no default for the other half of the predicate to find.
    """

    def stub(locus, assignment, *, min_order_margin: int, tie_break: str):  # pragma: no cover
        raise AssertionError

    assert no_rule_parameter_has_a_default(stub) is False


def test_no_rule_parameter_has_a_default_rejects_a_stale_inventory():
    """The other direction: a listed knob the function no longer has.

    A one-way containment check would pass here, leaving ``RULE_PARAMETERS`` free to describe a
    signature that no longer exists — which is how the inventory stops being evidence.
    """

    def stub(locus, assignment):  # pragma: no cover - signature only
        raise AssertionError

    assert RULE_PARAMETERS, "the inventory must be non-empty for this to mean anything"
    assert no_rule_parameter_has_a_default(stub) is False


def test_rule_parameters_matches_the_live_signature():
    keyword_only = {
        name
        for name, p in inspect.signature(resolve_strand).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert keyword_only == set(RULE_PARAMETERS)


def test_min_order_margin_below_one_is_refused():
    log_probs = _log_probs_from_layout(90, [("Stem_I", 10, 40), ("Terminator", 50, 70)])
    loci = construct_loci(
        log_probs,
        threshold_scope=_SCOPE,
        threshold=_TAU,
        min_span=1,
        gap_merge=12,
        min_distinct_elements=0,
        flank=0,
    )
    assignment = element_assignment(log_probs, threshold_scope=_SCOPE, threshold=_TAU)
    for bad in (0, -1):
        with pytest.raises(StrandError, match="min_order_margin must be >= 1"):
            resolve_strand(loci[0], assignment, min_order_margin=bad)
    # Positive control: the identical call at the smallest legal value succeeds.
    assert resolve_strand(loci[0], assignment, min_order_margin=1).strand == STRAND_PLUS


def test_a_fractional_margin_is_refused_rather_than_truncated():
    """``int(1.9) == 1`` would silently loosen the ambiguity rule the caller asked for."""
    log_probs = _log_probs_from_layout(90, [("Stem_I", 10, 40), ("Terminator", 50, 70)])
    loci = construct_loci(
        log_probs,
        threshold_scope=_SCOPE,
        threshold=_TAU,
        min_span=1,
        gap_merge=12,
        min_distinct_elements=0,
        flank=0,
    )
    assignment = element_assignment(log_probs, threshold_scope=_SCOPE, threshold=_TAU)
    with pytest.raises(StrandError, match="whole number"):
        resolve_strand(loci[0], assignment, min_order_margin=1.9)
    with pytest.raises(StrandError, match="boolean"):
        resolve_strand(loci[0], assignment, min_order_margin=True)
    # An integral float is a legitimate YAML/JSON value and converts without changing.
    assert resolve_strand(loci[0], assignment, min_order_margin=1.0).min_order_margin == 1


def test_a_mismatched_assignment_is_refused():
    """Grading order under an assignment the locus was not called from is silent otherwise."""
    log_probs = _log_probs_from_layout(90, [("Stem_I", 10, 40), ("Terminator", 50, 70)])
    loci = construct_loci(
        log_probs,
        threshold_scope=_SCOPE,
        threshold=_TAU,
        min_span=1,
        gap_merge=12,
        min_distinct_elements=0,
        flank=0,
    )
    with pytest.raises(StrandError, match="runs past"):
        resolve_strand(loci[0], np.zeros(5, dtype=np.int64), min_order_margin=1)
    with pytest.raises(StrandError, match="1-D"):
        resolve_strand(loci[0], np.zeros((90, 2), dtype=np.int64), min_order_margin=1)
    with pytest.raises(StrandError, match="integer class array"):
        resolve_strand(loci[0], np.zeros(90, dtype=np.float64), min_order_margin=1)


# --------------------------------------------------------------------------- #
# 5. the reverse complement — IUPAC-complete and fail-closed
# --------------------------------------------------------------------------- #
def test_complement_table_is_an_involution():
    for base, comp in IUPAC_COMPLEMENT_UPPER.items():
        assert IUPAC_COMPLEMENT_UPPER[comp] == base, f"{base} -> {comp} does not round-trip"


def test_complement_table_covers_the_iupac_dna_alphabet():
    assert set("ACGTRYSWKMBDHVN") <= set(IUPAC_COMPLEMENT_UPPER)
    assert set(IUPAC_COMPLEMENT) == set(IUPAC_COMPLEMENT_UPPER) | {
        k.lower() for k in IUPAC_COMPLEMENT_UPPER
    }


def test_uracil_is_refused_rather_than_admitted_non_involutively():
    """Admitting ``U`` would break the round trip, so this path refuses it.

    ``T`` and ``U`` both complement to ``A``, so ``A`` cannot complement back to both: a table
    carrying ``U`` silently rewrites ``U`` to ``T`` across a double reverse-complement. The
    input here is scanned genomic **DNA**; RNA arriving means a caller has already transcribed
    and is about to transcribe again, which is worth an exception.
    """
    assert "U" not in IUPAC_COMPLEMENT_UPPER
    assert "u" not in IUPAC_COMPLEMENT
    with pytest.raises(HandoffError, match="outside the IUPAC nucleotide alphabet"):
        reverse_complement("ACGU")
    with pytest.raises(HandoffError, match="outside the IUPAC"):
        transcribe_to_rna("ACGU", strand=STRAND_MINUS)
    # Positive control: the DNA spelling of the same sequence is accepted.
    assert transcribe_to_rna("ACGT", strand=STRAND_MINUS) == "ACGU"


def test_reverse_complement_hand_checked():
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("AAACCCGGGTTT") == "AAACCCGGGTTT"
    assert reverse_complement("") == ""


def test_reverse_complement_is_its_own_inverse_over_the_full_alphabet():
    seq = "".join(sorted(IUPAC_COMPLEMENT_UPPER)) + "".join(
        sorted(k.lower() for k in IUPAC_COMPLEMENT_UPPER)
    )
    assert reverse_complement(reverse_complement(seq)) == seq


def test_reverse_complement_preserves_soft_masking():
    """A soft-masked repeat stays soft-masked through the strand flip."""
    assert reverse_complement("atGC") == "GCat"
    assert reverse_complement("acgt") == "acgt"


def test_reverse_complement_differs_from_the_acgtn_only_behaviour_elsewhere_in_the_repo():
    """The IUPAC table is not decoration: the repo's other RCs get this input wrong.

    ``anchors.revcomp`` / ``flank_context.revcomp`` / ``homolog_msa._revcomp`` are
    ``str.maketrans`` tables over ``ACGTN`` (+ lower case in one), which leave an ambiguity
    code **uncomplemented** — a silently wrong base on the minus strand handed to Stage-2.
    ``window_dataset.reverse_complement`` maps it to ``N`` instead, destroying it.
    """
    acgtn_only = str.maketrans("ACGTNacgtn", "TGCANtgcan")

    def naive_pass_through(seq):
        return seq.translate(acgtn_only)[::-1]

    def naive_to_n(seq):
        table = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        return "".join(table.get(ch, "N") for ch in reversed(seq))

    seq = "ARYSWKMBDHVGC"
    ours = reverse_complement(seq)
    assert ours == "GCBDHVKMWSRYT"
    assert ours != naive_pass_through(seq)
    assert ours != naive_to_n(seq)
    # And the naive tables really are wrong rather than merely different.
    assert naive_pass_through(seq) == "GCVHDBMKWSYRT"
    assert naive_to_n(seq) == "GCNNNNNNNNNNT"


def test_reverse_complement_fails_closed_on_a_non_nucleotide():
    with pytest.raises(HandoffError, match="outside the IUPAC nucleotide alphabet"):
        reverse_complement("ACGTX")
    with pytest.raises(HandoffError):
        reverse_complement("ACGT ACGT")
    # Positive control: the identical string without the offending character succeeds.
    assert reverse_complement("ACGT") == "ACGT"


# --------------------------------------------------------------------------- #
# 6. exact T→U transcription, delegated not forked
# --------------------------------------------------------------------------- #
def test_transcription_is_exact_on_the_plus_strand():
    assert transcribe_to_rna("ATGCTTTA", strand=STRAND_PLUS) == "AUGCUUUA"
    assert transcribe_to_rna("acgt", strand=STRAND_PLUS) == "ACGU"


def test_transcription_is_exact_on_the_minus_strand():
    """Reverse-complement first, then T→U — the RNA reads 5′→3′ in the locus's orientation."""
    assert transcribe_to_rna("ATGCTTTA", strand=STRAND_MINUS) == "UAAAGCAU"
    # Hand-check: RC("ATGCTTTA") = "TAAAGCAT" -> T→U = "UAAAGCAU".
    assert reverse_complement("ATGCTTTA") == "TAAAGCAT"


def test_transcription_delegates_to_the_shipped_stage2_tokenizer(monkeypatch):
    """A second ``.replace("T", "U")`` here would let inference and training drift apart.

    ``stage2.tokenizer.transcribe`` is the function the Stage-2 training table was built with,
    and the Stage-2 dataset builder transcribes the training corpus through it — so the loci
    scored at inference must go through the *same* function, not one that happens to agree
    today.

    Comparing outputs cannot show that: a copy-pasted fork returns identical strings and the
    assertion stays green while the two drift apart at the next edit. So the delegation itself
    is observed — the shipped module calls ``rna_tokenizer.transcribe`` as a module attribute
    (never a ``from``-import, which would be unpatchable), and substituting it here must change
    what ``transcribe_to_rna`` returns.
    """
    for seq in ("ATGC", "acgtn", "ARYSWKMBDHVGC", ""):
        assert transcribe_to_rna(seq, strand=STRAND_PLUS) == rna_tokenizer.transcribe(seq)

    monkeypatch.setattr(rna_tokenizer, "transcribe", lambda s: f"<delegated:{s}>")
    assert transcribe_to_rna("ATGC", strand=STRAND_PLUS) == "<delegated:ATGC>"
    # The minus strand orients first and delegates the alphabet change second.
    assert transcribe_to_rna("ATGC", strand=STRAND_MINUS) == "<delegated:GCAT>"


def test_transcription_is_length_preserving():
    for seq in ("ATGC", "acgtnACGTN", "ARYSWKMBDHV"):
        for strand in (STRAND_PLUS, STRAND_MINUS):
            assert len(transcribe_to_rna(seq, strand=strand)) == len(seq)


def test_transcription_leaves_no_thymine():
    seq = "".join(sorted(IUPAC_COMPLEMENT_UPPER)) + "acgt"
    for strand in (STRAND_PLUS, STRAND_MINUS):
        assert "T" not in transcribe_to_rna(seq, strand=strand)
        assert "t" not in transcribe_to_rna(seq, strand=strand)


def test_transcription_refuses_an_unknown_strand_and_an_unknown_base():
    with pytest.raises(HandoffError, match="strand must be"):
        transcribe_to_rna("ACGT", strand="?")
    with pytest.raises(HandoffError, match="strand must be"):
        transcribe_to_rna("ACGT", strand="plus")
    # The plus-strand path validates the alphabet too — resolving to '+' is not a way in.
    with pytest.raises(HandoffError, match="outside the IUPAC"):
        transcribe_to_rna("ACGTX", strand=STRAND_PLUS)


# --------------------------------------------------------------------------- #
# 7. the Stage-2 payloads
# --------------------------------------------------------------------------- #
def _canonical_sequence(rng_seed: int = 0) -> str:
    rng = np.random.default_rng(rng_seed)
    return "".join(rng.choice(list("ACGT"), size=CANONICAL_SEQ_LEN))


def test_a_resolved_locus_yields_one_payload_on_its_own_strand():
    log_probs = _log_probs_from_layout(CANONICAL_SEQ_LEN, CANONICAL_LAYOUT)
    loci, calls = _loci_and_calls(log_probs, flank=10)
    sequence = _canonical_sequence()

    payloads = handoff_loci(sequence, loci, calls)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload.strand == STRAND_PLUS
    assert payload.low_order_confidence is False
    assert payload.locus_index == 0
    assert (payload.start, payload.end) == (loci[0].start, loci[0].end)
    assert payload.length == loci[0].length == len(payload.rna)
    assert payload.rna == rna_tokenizer.transcribe(sequence[loci[0].start : loci[0].end])


def test_an_ambiguous_locus_yields_a_payload_on_each_strand():
    """D15's both-strand carry-through, at the point it actually costs something.

    The two payloads share a span and are reverse complements of one another, so whichever
    strand is real, Stage-2 sees it.
    """
    layout = [("Stem_I", 20, 60)]
    log_probs = _log_probs_from_layout(100, layout)
    loci, calls = _loci_and_calls(log_probs, flank=5)
    sequence = "".join(np.random.default_rng(1).choice(list("ACGT"), size=100))

    payloads = handoff_loci(sequence, loci, calls)
    assert [p.strand for p in payloads] == list(BOTH_STRANDS)
    assert all(p.low_order_confidence for p in payloads)
    assert all(p.locus_index == 0 for p in payloads)
    assert payloads[0].start == payloads[1].start and payloads[0].end == payloads[1].end
    span = sequence[loci[0].start : loci[0].end]
    assert payloads[0].rna == rna_tokenizer.transcribe(span)
    assert payloads[1].rna == rna_tokenizer.transcribe(reverse_complement(span))
    # The two payloads really are the two strands of one span.
    assert payloads[1].rna == payloads[0].rna.translate(str.maketrans("ACGU", "UGCA"))[::-1]


def test_payload_counts_soft_masked_and_ambiguous_positions():
    """Recall-favouring: an ``N``-rich or soft-masked span is flagged, never dropped."""
    layout = [("Stem_I", 10, 30), ("Terminator", 35, 50)]
    log_probs = _log_probs_from_layout(60, layout)
    loci, calls = _loci_and_calls(log_probs, flank=0)
    start = loci[0].start

    sequence = list("A" * 60)
    for i in range(start, start + 4):
        sequence[i] = "a"
    for i in range(start + 10, start + 13):
        sequence[i] = "N"
    sequence[start + 20] = "R"
    payloads = handoff_loci("".join(sequence), loci, calls)

    assert payloads[0].n_soft_masked == 4
    assert payloads[0].n_ambiguous == 4  # three N + one R
    # A span with neither reports zero rather than None — this one is measured.
    clean = handoff_loci("A" * 60, loci, calls)
    assert (clean[0].n_soft_masked, clean[0].n_ambiguous) == (0, 0)


def test_handoff_is_sequence_only():
    """PRD §6: RiNALMo ingests sequence only; structure is an auxiliary target, not an input."""
    assert handoff_is_sequence_only(Handoff) is True
    assert not any(
        token in f for f in Handoff.__dataclass_fields__ for token in ("structure", "pairing")
    )


def test_handoff_is_sequence_only_rejects_a_structure_carrying_record():
    """Negative control — the predicate must be able to say False."""

    @dataclass(frozen=True)
    class WithStructure:
        rna: str
        dot_bracket: str

    assert handoff_is_sequence_only(WithStructure) is False


def test_handoff_refuses_a_locus_call_length_mismatch():
    log_probs = _log_probs_from_layout(CANONICAL_SEQ_LEN, CANONICAL_LAYOUT)
    loci, calls = _loci_and_calls(log_probs)
    sequence = _canonical_sequence()
    with pytest.raises(HandoffError, match="paired by position"):
        handoff_loci(sequence, loci, [])
    # Positive control: the matched pair succeeds on the identical inputs.
    assert handoff_loci(sequence, loci, calls)


def test_handoff_refuses_a_sequence_the_loci_do_not_fit():
    log_probs = _log_probs_from_layout(CANONICAL_SEQ_LEN, CANONICAL_LAYOUT)
    loci, calls = _loci_and_calls(log_probs)
    with pytest.raises(HandoffError, match="not the sequence the locus was called from"):
        handoff_loci("ACGT" * 5, loci, calls)


def test_end_to_end_minus_strand_locus_hands_over_the_forward_reading():
    """The whole path on a locus that is genuinely on the minus strand.

    Reconciled posteriors → loci → strand → RNA. The payload must read 5′→3′ *in the locus's
    own orientation*, which for a ``-`` locus is the reverse complement of the scanned span —
    the thing Stage-2 has to see for a re-rank to mean anything.
    """
    log_probs = _log_probs_from_layout(CANONICAL_SEQ_LEN, CANONICAL_LAYOUT)[::-1].copy()
    loci, calls = _loci_and_calls(log_probs, flank=8)
    sequence = _canonical_sequence(7)

    payloads = handoff_loci(sequence, loci, calls)
    assert len(payloads) == 1
    assert calls[0].strand == STRAND_MINUS
    span = sequence[loci[0].start : loci[0].end]
    assert payloads[0].rna == rna_tokenizer.transcribe(reverse_complement(span))
    assert payloads[0].rna != rna_tokenizer.transcribe(span)
    assert len(payloads[0].rna) == loci[0].length
