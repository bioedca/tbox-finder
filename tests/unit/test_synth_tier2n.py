"""Unit tier — the P2-07 synthetic non-canonical (Tier-2N) generator.

Guards three failure modes:

* an **unvalidated architecture bin** creeping in (only the two families that
  cleared the CLAUDE.md §10.1 ≥2-independent-source gate may be emitted, and the
  two affirmatively-contradicted departures must have no bin at all);
* a **fabricated probe set** — probe eligibility must be a *measured* triple
  (parent CM-detected, variant CM-missed, length-matched control CM-detected),
  never a count the generator picks; an unmeasured leg must not count as a CM
  miss, and omitting the control arm must not degrade to the confounded pair
  filter (random equal-length excision misses 78.8 %, MORE than the 66.7 % of real
  element ablations — so without the control leg the filter measures excision, not
  architecture);
* a **degenerate emission** — an "ablated" variant that is actually an unchanged
  copy of its parent, which would silently make the whole construction a no-op.

Bare-CI tier: pure stdlib, no numpy/pandas/torch.
"""

from __future__ import annotations

import pytest

from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.synth import tier2n
from tbox_finder.synth.tier2n import (
    FAMILY_ABLATED_ELEMENTS,
    FAMILY_CITATIONS,
    FAMILY_CLASS_II,
    FAMILY_STEM_II_PK,
    OBLIGATE_ELEMENTS,
    TIER2N_PROBE_MIN_N,
    VALIDATED_FAMILIES,
    Tier2NGeneratorError,
    Tier2NVariant,
    ablate,
    build_report,
    classify_pairs,
    generate,
    length_matched_control,
)

# A parent long enough that both ablations remove a visible chunk. Element extents
# are 0-based half-open offsets into ``FASTA_sequence``, matching ELEMENT_COORDS.
_SEQ = (
    "AUGCAUGCAUGCAUGCAUGCAUGCAUGC"  # 0-28   Stem I
    "GGGCCCAAAUUUGGGCCCAAAUUUGGG"  # 28-55  Stem II region
    "CCCGGGAAAUUUCCCGGGAAAUUUCCC"  # 55-82  Stem III + antiterminator
    "UUUUUUUAAAAAAAGGGGGGGCCCCCC"  # 82-109 Terminator
)


def _parent(record_id: str, *, type_: str = "Transcriptional") -> dict[str, object]:
    return {
        "record_id": record_id,
        "FASTA_sequence": _SEQ,
        "Type": type_,
        "s1_start": 0,
        "s1_end": 28,
        "stem2_region_start": 28,
        "stem2_region_end": 55,
        "stem3_start": 55,
        "stem3_end": 68,
        "antiterm_start": 68,
        "antiterm_end": 82,
        "term_start": 82,
        "term_end": 109,
        "codon_start": 10,
        "codon_end": 13,
    }


def _parents(n: int) -> list[dict[str, object]]:
    return [_parent(f"rec{i:03d}") for i in range(n)]


# --------------------------------------------------------------------------- #
# Only literature-grounded families exist
# --------------------------------------------------------------------------- #
def test_only_the_two_evidence_gated_families_are_offered() -> None:
    assert set(VALIDATED_FAMILIES) == {FAMILY_CLASS_II, FAMILY_STEM_II_PK}


def test_every_family_carries_at_least_two_independent_citations() -> None:
    """The §10.1 bar, pinned in code so a new bin cannot ship uncited."""
    for family in VALIDATED_FAMILIES:
        assert len(FAMILY_CITATIONS[family]) >= 2, family
        assert all(c.startswith(("PMID:", "DOI:")) for c in FAMILY_CITATIONS[family])


def test_an_unvalidated_family_is_refused() -> None:
    with pytest.raises(Tier2NGeneratorError):
        generate(_parents(3), seed=1, families=("ULTRASHORT_STEM_I",))


@pytest.mark.parametrize("element", ["Stem_I", "Stem_III", "Antiterminator_Tbox_seq"])
def test_no_family_ablates_an_obligate_element(element: str) -> None:
    """'No Stem III' and 'no Stem I' are affirmatively contradicted — no bin exists."""
    assert element in OBLIGATE_ELEMENTS
    for family in VALIDATED_FAMILIES:
        assert element not in FAMILY_ABLATED_ELEMENTS[family]


def test_an_unknown_family_is_refused_by_ablate() -> None:
    with pytest.raises(Tier2NGeneratorError, match="unknown family"):
        ablate(_parent("rec000"), "STEM_I_DELETION")


def test_ablating_an_obligate_element_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the obligate-element guard itself, not the unknown-family branch.

    An earlier version of this test passed a family name that was simply not
    registered, so it raised on ``unknown family`` and the obligate-element guard
    below it was never executed — the test passed while asserting nothing about
    the invariant in its own name. Registering a fake family that targets Stem I
    is what actually reaches the guard.
    """
    monkeypatch.setitem(tier2n.FAMILY_ABLATED_ELEMENTS, "FAKE_STEM_I_DELETION", ("Stem_I",))
    with pytest.raises(Tier2NGeneratorError, match="obligate element"):
        ablate(_parent("rec000"), "FAKE_STEM_I_DELETION")


def test_ablate_has_no_bypass_for_the_obligate_guard() -> None:
    """The removed ``allow_forbidden`` escape hatch must not come back silently."""
    import inspect

    assert "allow_forbidden" not in inspect.signature(ablate).parameters


# --------------------------------------------------------------------------- #
# The ablation actually perturbs — no unchanged copies
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", VALIDATED_FAMILIES)
def test_ablation_removes_the_targeted_extent(family: str) -> None:
    parent = _parent("rec000")
    out = ablate(parent, family)
    assert out != _SEQ
    assert len(out) < len(_SEQ)


def test_stem_ii_ablation_removes_exactly_the_stem2_region() -> None:
    """Independent closed-form pin, not a re-run of the implementation."""
    parent = _parent("rec000")
    assert ablate(parent, FAMILY_STEM_II_PK) == _SEQ[:28] + _SEQ[55:]


def test_class_ii_ablation_removes_exactly_the_terminator() -> None:
    parent = _parent("rec000")
    assert ablate(parent, FAMILY_CLASS_II) == _SEQ[:82]


def test_a_parent_missing_the_targeted_extent_is_refused_not_copied() -> None:
    """A silently-unablated 'variant' would corrupt the discordant-pair measure."""
    parent = _parent("rec000")
    parent["stem2_region_start"] = None
    parent["stem2_region_end"] = None
    with pytest.raises(Tier2NGeneratorError):
        ablate(parent, FAMILY_STEM_II_PK)


def test_no_emitted_variant_equals_its_parent() -> None:
    variants = generate(_parents(12), seed=7)
    assert variants
    for variant in variants:
        assert variant.sequence != variant.parent_sequence


def test_class_ii_family_skips_parents_that_are_already_class_ii() -> None:
    parents = [_parent("rec000", type_="Translational")]
    variants = generate(parents, seed=3, families=(FAMILY_CLASS_II,))
    assert variants == []


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_generation_is_deterministic_under_a_fixed_seed() -> None:
    a = generate(_parents(20), seed=1234)
    b = generate(_parents(20), seed=1234)
    assert [v.variant_id for v in a] == [v.variant_id for v in b]
    assert [v.sequence for v in a] == [v.sequence for v in b]


def test_different_seeds_reorder_the_emission() -> None:
    """Expressiveness: the seed must actually do something.

    Guards the degenerate-generator family — a keying bug that made every draw
    identical would leave the determinism test above passing while the seed was
    silently inert.
    """
    a = [v.variant_id for v in generate(_parents(40), seed=1, max_per_family=8)]
    b = [v.variant_id for v in generate(_parents(40), seed=999, max_per_family=8)]
    assert a != b


# --------------------------------------------------------------------------- #
# Probe eligibility is MEASURED, not chosen
# --------------------------------------------------------------------------- #
def _variant(**kw: object) -> Tier2NVariant:
    base: dict = {
        "variant_id": "v1",
        "parent_record_id": "p1",
        "family": FAMILY_STEM_II_PK,
        "ablated_elements": ("Stem_II",),
        "sequence": "AAAA",
        "parent_sequence": "AAAACCCC",
        "control_sequence": "CCCC",
    }
    base.update(kw)
    return Tier2NVariant(**base)


def _triple(parent: bool | None, variant: bool | None, control: bool | None) -> Tier2NVariant:
    return _variant(
        cm_detected_parent=parent, cm_detected_variant=variant, cm_detected_control=control
    )


def test_probe_eligible_only_for_the_full_triple() -> None:
    assert _triple(True, False, True).is_probe_eligible()
    # parent already missed → the ablation explains nothing
    assert not _triple(False, False, True).is_probe_eligible()
    # ablation did not break detection
    assert not _triple(True, True, True).is_probe_eligible()


def test_a_length_confounded_variant_is_not_probe_eligible() -> None:
    """The confound that invalidated the pair-only construction.

    Parent detected and variant missed — the pair filter's definition of a hit —
    but the length-matched control is ALSO missed, so the detection loss is
    explained by excising that many nucleotides at all, not by which element went.
    Measured in-repo: random equal-length excision misses 79.1%, MORE than the
    66.7% of real element ablations.
    """
    assert not _triple(True, False, False).is_probe_eligible()


@pytest.mark.parametrize(
    ("parent", "variant", "control"),
    [
        (None, False, True),
        (True, None, True),
        (True, False, None),
        (None, None, None),
    ],
)
def test_an_unmeasured_leg_is_never_probe_eligible(parent, variant, control) -> None:
    """An unrun cmsearch must shrink the probe set, never inflate it.

    Asymmetry worth stating precisely, because only one leg's ``None`` guard is
    load-bearing (established by sabotage, not by reading):

    * **variant leg** — the guard is essential. Without it ``None`` reaches
      ``not bool(None)`` → ``True``, i.e. "never measured" silently reads as "the
      CM missed it", and an absent backend could manufacture a probe set of any
      size. Deleting the guard fails this test.
    * **parent / control legs** — ``None`` already falls through to ``False`` via
      ``bool(None)``, so those clauses are defensive rather than load-bearing;
      deleting them changes no outcome. They are kept for explicitness, and this
      test pins the *behaviour*, not the mechanism.
    """
    assert not _triple(parent, variant, control).is_probe_eligible()


def test_classify_pairs_leaves_unmapped_variants_unmeasured() -> None:
    variants = [_variant(variant_id="v1", parent_record_id="p1")]
    out = classify_pairs(variants, parent_detected={}, variant_detected={})
    assert out[0].cm_detected_parent is None
    assert out[0].cm_detected_variant is None
    assert out[0].cm_detected_control is None
    assert not out[0].is_probe_eligible()


def test_omitting_the_control_arm_yields_an_empty_probe_set() -> None:
    """A skipped control arm must not silently degrade to the confounded filter."""
    variants = [_variant(variant_id="v1", parent_record_id="p1")]
    out = classify_pairs(
        variants,
        parent_detected={"p1": True},
        variant_detected={"v1": False},
        control_detected=None,
    )
    assert out[0].cm_detected_control is None
    assert not out[0].is_probe_eligible()
    assert build_report(out, seed=1)["probe_set_meets_min_n"] is False


# --------------------------------------------------------------------------- #
# The min-N gate
# --------------------------------------------------------------------------- #
def test_probe_min_n_is_the_adr_pin_not_a_local_literal() -> None:
    assert TIER2N_PROBE_MIN_N == MIN_REAL_HOMOLOG_N


def test_report_min_n_clause_is_false_on_an_empty_probe_set() -> None:
    """A clause derived from config rather than evidence is vacuously true when
    the evidence is absent — assert the absence branch's gate, not just its fields."""
    report = build_report([], seed=1)
    assert report["n_probe_eligible"] == 0
    assert report["probe_set_meets_min_n"] is False


def test_report_min_n_clause_tracks_the_measured_eligible_count() -> None:
    eligible = [
        _variant(
            variant_id=f"v{i}",
            cm_detected_parent=True,
            cm_detected_variant=False,
            cm_detected_control=True,
        )
        for i in range(MIN_REAL_HOMOLOG_N)
    ]
    assert build_report(eligible, seed=1)["probe_set_meets_min_n"] is True
    assert build_report(eligible[:-1], seed=1)["probe_set_meets_min_n"] is False


def test_report_counts_every_discard_reason_visibly() -> None:
    """Each discard must be counted under its own cause, not silently dropped."""
    variants = [
        _triple(True, False, True),  # eligible
        _triple(False, False, True),  # parent already missed
        _triple(True, True, True),  # ablation did not break detection
        _triple(True, False, False),  # length-confounded
        _triple(None, None, None),  # unmeasured
    ]
    for i, v in enumerate(variants):
        variants[i] = Tier2NVariant(**{**v.__dict__, "variant_id": f"v{i}"})
    report = build_report(variants, seed=1)
    assert report["n_emitted"] == 5
    assert report["n_probe_eligible"] == 1
    assert report["n_discarded_parent_already_cm_missed"] == 1
    assert report["n_discarded_ablation_did_not_break_detection"] == 1
    assert report["n_discarded_length_confounded"] == 1
    assert report["n_unmeasured"] == 1


def test_report_exposes_per_family_min_n_so_a_floor_sitting_family_is_visible() -> None:
    """The pooled total can clear min-N while a family sits on the floor."""
    variants = [
        _variant(
            variant_id=f"a{i}",
            family=FAMILY_CLASS_II,
            cm_detected_parent=True,
            cm_detected_variant=False,
            cm_detected_control=True,
        )
        for i in range(MIN_REAL_HOMOLOG_N)
    ] + [
        _variant(
            variant_id=f"b{i}",
            family=FAMILY_STEM_II_PK,
            cm_detected_parent=True,
            cm_detected_variant=False,
            cm_detected_control=True,
        )
        for i in range(3)
    ]
    report = build_report(variants, seed=1)
    assert report["probe_set_meets_min_n"] is True
    assert report["per_family"][FAMILY_CLASS_II]["meets_min_n_alone"] is True
    assert report["per_family"][FAMILY_STEM_II_PK]["meets_min_n_alone"] is False


# --------------------------------------------------------------------------- #
# The length-matched control construct itself
# --------------------------------------------------------------------------- #
def test_control_removes_the_same_length_as_the_ablation() -> None:
    parent = _parent("rec000")
    for family in VALIDATED_FAMILIES:
        removed = len(_SEQ) - len(ablate(parent, family))
        control = length_matched_control(parent, family, n_removed=removed, seed=5)
        assert len(control) == len(_SEQ) - removed


def test_control_excision_does_not_overlap_the_ablated_extent() -> None:
    """Otherwise the control would remove part of the element it controls for."""
    parent = _parent("rec000")
    # Stem II occupies [28, 55); a control that avoided it must retain that slice.
    removed = len(_SEQ) - len(ablate(parent, FAMILY_STEM_II_PK))
    control = length_matched_control(parent, FAMILY_STEM_II_PK, n_removed=removed, seed=5)
    assert _SEQ[28:55] in control


def test_control_is_deterministic_and_seed_sensitive() -> None:
    parent = _parent("rec000")
    removed = len(_SEQ) - len(ablate(parent, FAMILY_STEM_II_PK))
    a = length_matched_control(parent, FAMILY_STEM_II_PK, n_removed=removed, seed=1)
    b = length_matched_control(parent, FAMILY_STEM_II_PK, n_removed=removed, seed=1)
    assert a == b
    seeds = {
        length_matched_control(parent, FAMILY_STEM_II_PK, n_removed=removed, seed=s)
        for s in range(24)
    }
    assert len(seeds) > 1, "the control window must actually vary with the seed"


def test_control_differs_from_the_real_ablation() -> None:
    parent = _parent("rec000")
    for family in VALIDATED_FAMILIES:
        variant = ablate(parent, family)
        removed = len(_SEQ) - len(variant)
        control = length_matched_control(parent, family, n_removed=removed, seed=5)
        assert control != variant


def test_every_generated_variant_carries_a_control() -> None:
    """A variant without a control can never be eligible — none may be emitted."""
    variants = generate(_parents(12), seed=7)
    assert variants
    for variant in variants:
        assert variant.control_sequence
        assert len(variant.control_sequence) == len(variant.sequence)
        assert variant.control_id == f"ctl:{variant.variant_id}"


@pytest.mark.parametrize("bad", [0, -1, 10_000])
def test_control_rejects_an_impossible_excision_length(bad: int) -> None:
    with pytest.raises(Tier2NGeneratorError):
        length_matched_control(_parent("rec000"), FAMILY_STEM_II_PK, n_removed=bad, seed=1)
