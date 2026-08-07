"""Unit tests for ADR-0006 D3's criterion-(b) predicate and its de-novo localizer.

Every test here is written so that **breaking the thing it names** turns it red — the
sabotage campaign for this step checks each one individually, because a suite that only
fails as a block proves *some* test bit, not which.

Two disciplines this file follows deliberately:

* **Relations are tested through the function, never through a committed artifact.**  An
  assertion over ``reports/p3/architecture_freeze.json`` cannot see a producer that
  writes the file correctly for the wrong reason — the bytes on disk were written by the
  code being sabotaged.  Where a property is a relation ("the detector recovers the
  curated bulge"), it is exercised on inputs built here.
* **Every refusal carries a positive control.**  A guard that raises on *everything* also
  satisfies ``pytest.raises``, so each refusal is paired with the nearest input that must
  be accepted.
"""

from __future__ import annotations

import pytest

from tbox_finder.labels import CLASS_ORDER
from tbox_finder.mining import architecture
from tbox_finder.mining.spare_rule import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
)

# A minimal but REAL architecture: helix A1 (4 pairs), a 7-nt bulge whose 5' half is the
# acceptor-pairing motif, helix A2 (5 pairs) inside it, then A1 closes. Built by hand from
# the curated shape measured on master_clean_v0, not copied from a fixture file.
#            0123456789...
ANTITERM_SEQ = "GCGGUGGCACCGCGAGUUCCCUUCUCGCCCGC"
ANTITERM_SS = "((((.......(((((.......)))))))))"


def sto(seqs: list[tuple[str, str]], ss_cons: str, *, wrap: int | None = None) -> str:
    """A Stockholm alignment, optionally wrapped into interleaved blocks."""
    width = len(ss_cons)
    if wrap is None:
        wrap = width
    lines = ["# STOCKHOLM 1.0"]
    for start in range(0, width, wrap):
        for name, seq in seqs:
            lines.append(f"{name} {seq[start : start + wrap]}")
        lines.append(f"#=GC SS_cons {ss_cons[start : start + wrap]}")
    lines.append("//")
    return "\n".join(lines) + "\n"


def localization(**over) -> architecture.Localization:
    base = {
        "candidate_id": "c1",
        "named_elements_present": True,
        "bulge_state": architecture.BULGE_DETECTED,
        "ultrashort_relax": False,
        "class_ii_relax": False,
        "n_sequences": 24,
    }
    base.update(over)
    return architecture.Localization(**base)


# ═════════════════════════════════════════════════════════════════════════════
class TestAcceptorMotifIsDerived:
    """The one sequence fact in this module is DERIVED from the tRNA 3' end, not typed in."""

    def test_the_motif_is_the_antiparallel_complement_of_NCCA(self) -> None:
        assert architecture.acceptor_pairing_motif() == "UGGN"

    def test_it_tracks_the_acceptor_string_rather_than_being_hardcoded(self) -> None:
        # If UGGN were a literal, these would not follow.
        assert architecture.acceptor_pairing_motif("CCA") == "UGG"
        assert architecture.acceptor_pairing_motif("A") == "U"
        assert architecture.acceptor_pairing_motif("GGC") == "GCC"

    def test_a_non_rna_acceptor_refuses(self) -> None:
        with pytest.raises(architecture.ArchitectureError, match="non-RNA"):
            architecture.acceptor_pairing_motif("CCX")

    def test_positive_control_a_dna_spelling_is_accepted_and_folded(self) -> None:
        assert architecture.acceptor_pairing_motif("NCCT") == "AGGN"

    def test_an_empty_acceptor_refuses(self) -> None:
        with pytest.raises(architecture.ArchitectureError, match="empty"):
            architecture.acceptor_pairing_motif("")


class TestStockholmParsing:
    def test_interleaved_blocks_are_concatenated_not_truncated(self) -> None:
        wrapped = sto([("s1", ANTITERM_SEQ), ("s2", ANTITERM_SEQ)], ANTITERM_SS, wrap=10)
        parsed = architecture.parse_stockholm(wrapped)
        assert parsed.width == len(ANTITERM_SS)
        assert parsed.row("s1") == ANTITERM_SEQ
        assert parsed.n_sequences == 2

    def test_an_alignment_without_ss_cons_refuses(self) -> None:
        text = "# STOCKHOLM 1.0\ns1 " + ANTITERM_SEQ + "\n//\n"
        with pytest.raises(architecture.ArchitectureError, match="SS_cons"):
            architecture.parse_stockholm(text)

    def test_duplicate_row_names_refuse_rather_than_collapsing(self) -> None:
        """`names` keeps both while the dict merges, so n_sequences would OVERSTATE the
        depth the ADR-0006 A2 floor is checked against."""
        text = sto([("s1", ANTITERM_SEQ), ("s1", ANTITERM_SEQ)], ANTITERM_SS)
        # The delegate accumulates rows BY NAME, so two rows sharing one concatenate into a
        # double-width row and the ragged guard fires first. Either refusal is correct; what
        # must never happen is a ConsensusStructure whose n_sequences exceeds its real rows.
        with pytest.raises(architecture.ArchitectureError, match="duplicate row names|ragged"):
            architecture.parse_stockholm(text)

    def test_n_sequences_cannot_exceed_the_rows_actually_present(self) -> None:
        """The invariant behind the A2 depth floor: `n_sequences` is `len(sequences)` by
        construction, so a duplicate row name cannot inflate the depth
        `architecture_status` gates on. Rows sharing a name concatenate into ONE entry."""
        text = "# STOCKHOLM 1.0\ns1 ACGU\ns1 ACGU\n#=GC SS_cons ((((((((\n//\n"
        parsed = architecture.parse_stockholm(text)
        assert parsed.n_sequences == len(parsed.sequences) == 1
        assert parsed.row("s1") == "ACGUACGU"

    def test_a_ragged_alignment_refuses(self) -> None:
        text = sto([("s1", ANTITERM_SEQ), ("s2", ANTITERM_SEQ[:-4])], ANTITERM_SS)
        with pytest.raises(architecture.ArchitectureError, match="ragged"):
            architecture.parse_stockholm(text)

    def test_positive_control_a_well_formed_alignment_parses(self) -> None:
        parsed = architecture.parse_stockholm(sto([("s1", ANTITERM_SEQ)], ANTITERM_SS))
        assert parsed.names == ("s1",) and parsed.ss_cons == ANTITERM_SS

    def test_row_may_be_addressed_by_name_or_index_and_they_agree(self) -> None:
        parsed = architecture.parse_stockholm(
            sto([("a", ANTITERM_SEQ), ("b", ANTITERM_SEQ)], ANTITERM_SS)
        )
        assert parsed.row(0) == parsed.row("a")
        with pytest.raises(architecture.ArchitectureError, match="out of range"):
            parsed.row(9)
        with pytest.raises(architecture.ArchitectureError, match="no row named"):
            parsed.row("nope")


class TestPairTable:
    def test_wuss_and_plain_brackets_give_the_same_pairing(self) -> None:
        """The curated corpus is WUSS, the mlocarna consensus is plain — one parser, one answer."""
        plain = architecture.pair_table("((..))")
        wuss = architecture.pair_table("<<__>>")
        assert plain == wuss == [5, 4, -1, -1, 1, 0]

    def test_bracket_families_do_not_cross_pair(self) -> None:
        pairs = architecture.pair_table("(<)>")
        assert pairs == [2, 3, 0, 1]

    def test_an_unbalanced_structure_refuses(self) -> None:
        with pytest.raises(architecture.ArchitectureError, match="unbalanced"):
            architecture.pair_table("((.)")
        with pytest.raises(architecture.ArchitectureError, match="unbalanced"):
            architecture.pair_table(".)")

    def test_positive_control_a_balanced_structure_parses(self) -> None:
        assert architecture.pair_table("(.)") == [2, -1, 0]

    def test_unknown_annotation_is_unpaired_not_an_error(self) -> None:
        """A pseudoknot letter must shrink what is claimed, never crash the producer array."""
        assert architecture.pair_table("(AA)") == [3, -1, -1, 0]


class TestHelices:
    def test_a_maximal_stack_is_one_helix_not_four(self) -> None:
        helices = architecture.find_helices(architecture.pair_table("((((....))))"), min_pairs=1)
        assert len(helices) == 1
        assert (helices[0].left_start, helices[0].right_end, helices[0].n_pairs) == (0, 11, 4)

    def test_min_pairs_excludes_shallow_stacks_and_the_bound_is_inclusive(self) -> None:
        pairs = architecture.pair_table("((...))..(....)")
        assert len(architecture.find_helices(pairs, min_pairs=1)) == 2
        assert len(architecture.find_helices(pairs, min_pairs=2)) == 1
        assert len(architecture.find_helices(pairs, min_pairs=3)) == 0

    def test_min_pairs_below_one_refuses(self) -> None:
        with pytest.raises(architecture.ArchitectureError, match="min_pairs"):
            architecture.find_helices([-1], min_pairs=0)


class TestBulges:
    def test_a_hairpin_loop_is_not_a_bulge(self) -> None:
        """Its flanks close each other; admitting it offers every stem-loop's loop as a bulge."""
        assert architecture.find_bulges(architecture.pair_table("((....))")) == []

    def test_the_real_antiterminator_bulge_is_found_between_the_two_helices(self) -> None:
        pairs = architecture.pair_table(ANTITERM_SS)
        bulges = architecture.find_bulges(pairs)
        assert len(bulges) == 1
        bulge = bulges[0]
        assert ANTITERM_SEQ[bulge.start : bulge.end + 1] == "UGGCACC"
        assert bulge.size == 7

    def test_a_terminal_tail_is_not_a_bulge(self) -> None:
        assert architecture.find_bulges(architecture.pair_table("..((..))..")) == []

    def test_find_bulges_does_not_filter_by_size(self) -> None:
        """Sizing in COLUMNS was the measured defect: a 16-column run is 8 residues in the
        candidate's own row, so the size test belongs where the sequence is known."""
        pairs = architecture.pair_table("((" + "." * 40 + "))")
        assert len(architecture.find_bulges(pairs)) == 0  # that is a hairpin loop
        pairs2 = architecture.pair_table("((" + "." * 40 + "((..))" + "))")
        assert [b.size for b in architecture.find_bulges(pairs2)] == [40]


class TestNccaBulgeIsThreeValued:
    """``absent`` vs ``undetectable`` is the distinction that keeps ``class_ii_relax`` honest."""

    def test_the_curated_bulge_is_detected(self) -> None:
        state, detail = architecture.ncca_bulge_status(
            ANTITERM_SEQ,
            architecture.pair_table(ANTITERM_SS),
            bulge_size_range=(5, 9),
            ncca_pairing_nt=4,
        )
        assert state == architecture.BULGE_DETECTED
        assert detail["matched"] == "UGGC"

    def test_a_resolvable_bulge_without_the_motif_is_ABSENT_not_undetectable(self) -> None:
        seq = ANTITERM_SEQ[:4] + "AAAAAAA" + ANTITERM_SEQ[11:]
        state, _ = architecture.ncca_bulge_status(
            seq,
            architecture.pair_table(ANTITERM_SS),
            bulge_size_range=(5, 9),
            ncca_pairing_nt=4,
        )
        assert state == architecture.BULGE_ABSENT

    def test_no_admissible_bulge_at_all_is_UNDETECTABLE_not_absent(self) -> None:
        state, _ = architecture.ncca_bulge_status(
            ANTITERM_SEQ,
            architecture.pair_table(ANTITERM_SS),
            bulge_size_range=(20, 30),
            ncca_pairing_nt=4,
        )
        assert state == architecture.BULGE_UNDETECTABLE

    def test_size_is_measured_in_residues_not_alignment_columns(self) -> None:
        """The gapped row's bulge is 11 columns but 7 residues; a column-width filter misses it."""
        gapped_seq = ANTITERM_SEQ[:4] + "UGG--C--ACC" + ANTITERM_SEQ[11:]
        gapped_ss = ANTITERM_SS[:4] + "..........." + ANTITERM_SS[11:]
        state, detail = architecture.ncca_bulge_status(
            gapped_seq,
            architecture.pair_table(gapped_ss),
            bulge_size_range=(7, 7),
            ncca_pairing_nt=4,
        )
        assert state == architecture.BULGE_DETECTED
        assert detail["bulge_size"] == 11  # columns
        assert detail["matched"] == "UGGC"

    def test_the_register_is_searched_across_the_whole_bulge(self) -> None:
        """Fauzi 2008: the pairing register is not constrained to the first four bases."""
        seq = ANTITERM_SEQ[:4] + "ACCUGGC" + ANTITERM_SEQ[11:]
        state, detail = architecture.ncca_bulge_status(
            seq, architecture.pair_table(ANTITERM_SS), bulge_size_range=(5, 9), ncca_pairing_nt=4
        )
        assert state == architecture.BULGE_DETECTED and detail["matched_offset"] == 3

    def test_wobble_is_off_unless_asked_for_and_tests_the_REAL_interface(self) -> None:
        """⚠ The first version of this test locked in an inverted rule.

        ``motif_base`` is already the Watson-Crick complement of the acceptor base, so
        comparing the candidate base against *it* is not a pairing question. The wobble is
        between the bulge base and the **acceptor** base.

        Concretely: motif position 1 is ``U`` (it pairs acceptor ``A76``). A real wobble
        there is bulge ``G`` against acceptor ``A``... which is not a wobble either — so
        take position 2, motif ``G``, acceptor ``C75``: no wobble. The reachable case on
        this motif is a wildcard-free position whose acceptor base is ``U`` or ``G``. With
        acceptor ``CCA`` reversed to motif ``UGG``, position 0's acceptor base is ``A``.
        So the honest test uses an acceptor that *has* a wobble-capable partner: acceptor
        ``G`` -> motif ``C`` -> a bulge ``U`` wobble-pairs the acceptor ``G``.
        """
        pairs = architecture.pair_table(ANTITERM_SS)
        # acceptor "GG" -> motif "CC"; bulge "UU" wobble-pairs acceptor "GG", and must NOT
        # be accepted as a plain Watson-Crick match.
        seq = ANTITERM_SEQ[:4] + "UUAGAUG" + ANTITERM_SEQ[11:]
        strict, _ = architecture.ncca_bulge_status(
            seq, pairs, bulge_size_range=(5, 9), ncca_pairing_nt=2, acceptor_3prime="GG"
        )
        loose, _ = architecture.ncca_bulge_status(
            seq,
            pairs,
            bulge_size_range=(5, 9),
            ncca_pairing_nt=2,
            acceptor_3prime="GG",
            allow_wobble=True,
        )
        assert strict == architecture.BULGE_ABSENT
        assert loose == architecture.BULGE_DETECTED

    def test_the_wobble_arm_compares_against_the_ACCEPTOR_base(self) -> None:
        """Unit-level pin on the direction, independent of any bulge geometry."""
        # motif "C" <- acceptor "G". A bulge U wobble-pairs the acceptor G.
        assert architecture._pairs_with("U", "C", allow_wobble=True) is True
        assert architecture._pairs_with("U", "C", allow_wobble=False) is False
        # motif "G" <- acceptor "C". A bulge U does NOT pair C, by wobble or otherwise.
        assert architecture._pairs_with("U", "G", allow_wobble=True) is False

    def test_the_all_wildcard_register_is_dropped_not_matched(self) -> None:
        """At ncca_pairing_nt=1 the motif UGGN yields registers U, G, G and **N**. Keeping
        N would make every admissibly-sized bulge read `detected` regardless of sequence —
        a vacuous detector. A bulge of only C/A residues must therefore still be ABSENT."""
        seq = ANTITERM_SEQ[:4] + "CACACAC" + ANTITERM_SEQ[11:]
        state, _ = architecture.ncca_bulge_status(
            seq,
            architecture.pair_table(ANTITERM_SS),
            bulge_size_range=(5, 9),
            ncca_pairing_nt=1,
        )
        assert state == architecture.BULGE_ABSENT

    def test_a_motif_with_ONLY_wildcards_refuses_outright(self) -> None:
        with pytest.raises(architecture.ArchitectureError, match="all-wildcard"):
            architecture.ncca_bulge_status(
                ANTITERM_SEQ,
                architecture.pair_table(ANTITERM_SS),
                bulge_size_range=(5, 9),
                ncca_pairing_nt=1,
                acceptor_3prime="N",
            )

    def test_positive_control_a_constrained_register_still_matches_at_one(self) -> None:
        state, _ = architecture.ncca_bulge_status(
            ANTITERM_SEQ,
            architecture.pair_table(ANTITERM_SS),
            bulge_size_range=(5, 9),
            ncca_pairing_nt=1,
        )
        assert state == architecture.BULGE_DETECTED

    def test_an_ambiguity_code_breaks_contiguity_rather_than_vanishing(self) -> None:
        """Deleting IUPAC codes as if they were gaps FABRICATES a contiguous match — a
        false positive, the one direction this module must not get wrong."""
        seq = ANTITERM_SEQ[:4] + "UGNGCAC" + ANTITERM_SEQ[11:]
        state, _ = architecture.ncca_bulge_status(
            seq,
            architecture.pair_table(ANTITERM_SS),
            bulge_size_range=(5, 9),
            ncca_pairing_nt=3,
        )
        assert state == architecture.BULGE_ABSENT

    def test_positive_control_a_gap_DOES_vanish(self) -> None:
        gapped_seq = ANTITERM_SEQ[:4] + "UG-GCAC" + ANTITERM_SEQ[11:]
        state, _ = architecture.ncca_bulge_status(
            gapped_seq,
            architecture.pair_table(ANTITERM_SS),
            bulge_size_range=(5, 9),
            ncca_pairing_nt=3,
        )
        assert state == architecture.BULGE_DETECTED

    def test_a_range_that_cannot_hold_the_motif_refuses(self) -> None:
        """Silently reading 'undetectable' for every candidate would spare the whole corpus."""
        with pytest.raises(architecture.ArchitectureError, match="shorter than"):
            architecture.ncca_bulge_status(
                ANTITERM_SEQ,
                architecture.pair_table(ANTITERM_SS),
                bulge_size_range=(1, 3),
                ncca_pairing_nt=4,
            )

    def test_ncca_pairing_nt_out_of_range_refuses(self) -> None:
        with pytest.raises(architecture.ArchitectureError, match="ncca_pairing_nt"):
            architecture.ncca_bulge_status(
                ANTITERM_SEQ,
                architecture.pair_table(ANTITERM_SS),
                bulge_size_range=(5, 9),
                ncca_pairing_nt=5,
            )


class TestNamedElementsReuseTheD1Vocabulary:
    def test_the_expected_set_is_derived_from_CLASS_ORDER(self) -> None:
        """D3 says (b) reuses the ADR-0004 D1 vocabulary; a re-typed copy would drift from it."""
        assert set(architecture.EXPECTED_HELIX_ELEMENTS) <= set(CLASS_ORDER)
        assert architecture.EXPECTED_HELIX_ELEMENTS == (
            "Stem_I",
            "Stem_II",
            "Stem_III",
            "Antiterminator_Tbox_seq",
        )
        assert len(architecture.EXPECTED_HELIX_ELEMENTS) == architecture.MAX_NAMED_HELICES

    def test_asking_for_more_helices_than_D1_names_refuses(self) -> None:
        with pytest.raises(architecture.ArchitectureError, match="Tier-2N"):
            architecture.named_elements_status(
                architecture.pair_table(ANTITERM_SS),
                min_named_helices=architecture.MAX_NAMED_HELICES + 1,
                min_helix_pairs=1,
            )

    def test_positive_control_the_maximum_is_accepted(self) -> None:
        present, _ = architecture.named_elements_status(
            architecture.pair_table("(.)(.)(.)(.)"),
            min_named_helices=architecture.MAX_NAMED_HELICES,
            min_helix_pairs=1,
        )
        assert present is True

    def test_the_count_is_compared_not_merely_reported(self) -> None:
        pairs = architecture.pair_table(ANTITERM_SS)  # two helices
        assert architecture.named_elements_status(pairs, min_named_helices=2, min_helix_pairs=1)[0]
        assert not architecture.named_elements_status(
            pairs, min_named_helices=3, min_helix_pairs=1
        )[0]


class TestD6Predicate:
    def test_short_stem_i_fires_the_carveout(self) -> None:
        assert architecture.short_stem_i_or_class_ii(30, "transcriptional", stem_i_nt_threshold=60)

    def test_a_canonical_stem_i_does_not(self) -> None:
        assert not architecture.short_stem_i_or_class_ii(
            96, "transcriptional", stem_i_nt_threshold=60
        )

    def test_the_threshold_is_strict_not_inclusive(self) -> None:
        assert not architecture.short_stem_i_or_class_ii(60, None, stem_i_nt_threshold=60)
        assert architecture.short_stem_i_or_class_ii(59, None, stem_i_nt_threshold=60)

    def test_translational_mode_fires_it_regardless_of_extent(self) -> None:
        assert architecture.short_stem_i_or_class_ii(200, "Translational", stem_i_nt_threshold=60)

    def test_an_unknown_extent_does_not_satisfy_the_threshold_arm(self) -> None:
        """Absence of a measurement is not evidence of shortness."""
        assert not architecture.short_stem_i_or_class_ii(None, None, stem_i_nt_threshold=60)

    def test_it_has_no_default_threshold(self) -> None:
        with pytest.raises(TypeError):
            architecture.short_stem_i_or_class_ii(30, None)  # type: ignore[call-arg]


class TestCriterionB:
    """D3's frozen predicate, tested against the ADR's own text."""

    def test_base_case_is_the_conjunction(self) -> None:
        assert architecture.criterion_b(True, True, ultrashort_relax=False, class_ii_relax=False)
        assert not architecture.criterion_b(
            True, False, ultrashort_relax=False, class_ii_relax=False
        )
        assert not architecture.criterion_b(
            False, True, ultrashort_relax=False, class_ii_relax=False
        )

    def test_ultrashort_relax_drops_ONLY_the_helix_requirement(self) -> None:
        assert architecture.criterion_b(False, True, ultrashort_relax=True, class_ii_relax=False)
        assert not architecture.criterion_b(
            False, False, ultrashort_relax=True, class_ii_relax=False
        )

    def test_class_ii_relax_drops_ONLY_the_bulge_requirement(self) -> None:
        assert architecture.criterion_b(True, False, ultrashort_relax=False, class_ii_relax=True)
        assert not architecture.criterion_b(
            False, False, ultrashort_relax=False, class_ii_relax=True
        )

    def test_both_relaxations_are_vacuous_which_is_why_the_status_layer_guards_it(self) -> None:
        """The ADR's literal text. Preserved here so the P6 freeze tests D3, not our fix."""
        assert architecture.criterion_b(False, False, ultrashort_relax=True, class_ii_relax=True)

    def test_the_relaxations_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            architecture.criterion_b(True, True, True, True)  # type: ignore[misc]


class TestArchitectureStatus:
    def test_a_detected_bulge_and_present_helices_pass(self) -> None:
        status, _ = architecture.architecture_status(localization(), min_sequences=20)
        assert status == STATUS_PASSED

    def test_an_absent_bulge_fails_and_is_not_excused_by_class_ii_relax(self) -> None:
        """D3 relaxes the bulge's DETECTION CONFIDENCE, not the requirement."""
        loc = localization(bulge_state=architecture.BULGE_ABSENT, class_ii_relax=True)
        status, detail = architecture.architecture_status(loc, min_sequences=20)
        assert status == STATUS_FAILED
        assert detail["class_ii_relax_effective"] is False
        assert detail["class_ii_relax_declared"] is True

    def test_an_undetectable_bulge_IS_excused_by_class_ii_relax(self) -> None:
        loc = localization(bulge_state=architecture.BULGE_UNDETECTABLE, class_ii_relax=True)
        status, detail = architecture.architecture_status(loc, min_sequences=20)
        assert status == STATUS_PASSED
        assert detail["class_ii_relax_effective"] is True

    def test_an_undetectable_bulge_without_the_relaxation_fails(self) -> None:
        loc = localization(bulge_state=architecture.BULGE_UNDETECTABLE, class_ii_relax=False)
        assert architecture.architecture_status(loc, min_sequences=20)[0] == STATUS_FAILED

    def test_no_localization_is_unavailable_never_failed(self) -> None:
        status, detail = architecture.architecture_status(None, min_sequences=20)
        assert status == STATUS_UNAVAILABLE
        assert "no per-candidate consensus" in detail["reason"]

    def test_below_the_A2_depth_floor_is_unavailable_never_failed(self) -> None:
        status, detail = architecture.architecture_status(
            localization(n_sequences=19), min_sequences=20
        )
        assert status == STATUS_UNAVAILABLE and "below" in detail["reason"]

    def test_the_depth_floor_is_inclusive_at_the_boundary(self) -> None:
        assert (
            architecture.architecture_status(localization(n_sequences=20), min_sequences=20)[0]
            == STATUS_PASSED
        )

    def test_the_vacuous_pass_guard_routes_to_unavailable_not_passed(self) -> None:
        """Both relaxations active with NEITHER element observed: D3's text says True, but a
        (b) that is never False makes D9 row 5's Tier-2N unreachable."""
        loc = localization(
            named_elements_present=False,
            bulge_state=architecture.BULGE_UNDETECTABLE,
            ultrashort_relax=True,
            class_ii_relax=True,
        )
        status, detail = architecture.architecture_status(loc, min_sequences=20)
        assert status == STATUS_UNAVAILABLE
        assert "Tier-2N" in detail["reason"]

    def test_the_guard_does_NOT_fire_when_one_element_was_observed(self) -> None:
        """A positive control for the guard: sabotaging it into always-firing must go red."""
        loc = localization(
            named_elements_present=True,
            bulge_state=architecture.BULGE_UNDETECTABLE,
            ultrashort_relax=True,
            class_ii_relax=True,
        )
        assert architecture.architecture_status(loc, min_sequences=20)[0] == STATUS_PASSED

    def test_ultrashort_relax_alone_cannot_pass_on_an_absent_bulge(self) -> None:
        loc = localization(
            named_elements_present=False,
            bulge_state=architecture.BULGE_ABSENT,
            ultrashort_relax=True,
        )
        assert architecture.architecture_status(loc, min_sequences=20)[0] == STATUS_FAILED


class TestLocalizeEndToEnd:
    def test_a_canonical_antiterminator_localizes_to_passed(self) -> None:
        consensus = architecture.parse_stockholm(
            sto([(f"s{i}", ANTITERM_SEQ) for i in range(24)], ANTITERM_SS)
        )
        loc = architecture.localize(
            "c1",
            consensus,
            stem_i_extent_nt=96,
            regulatory_mode="transcriptional",
            stem_i_nt_threshold=60,
            class_ii=False,
            min_named_helices=2,
            min_helix_pairs=1,
            bulge_size_range=(5, 9),
            ncca_pairing_nt=4,
        )
        assert loc.named_elements_present is True
        assert loc.bulge_state == architecture.BULGE_DETECTED
        assert loc.ultrashort_relax is False
        assert architecture.architecture_status(loc, min_sequences=20)[0] == STATUS_PASSED

    def test_the_candidates_OWN_row_is_read_not_the_first_row(self) -> None:
        """The consensus is the alignment's; the sequence read through it is this locus's."""
        broken = ANTITERM_SEQ[:4] + "AAAAAAA" + ANTITERM_SEQ[11:]
        consensus = architecture.parse_stockholm(
            sto([("good", ANTITERM_SEQ), ("bad", broken)], ANTITERM_SS)
        )
        good = architecture.localize(
            "c",
            consensus,
            row="good",
            stem_i_extent_nt=96,
            regulatory_mode=None,
            stem_i_nt_threshold=60,
            class_ii=False,
            min_named_helices=2,
            min_helix_pairs=1,
            bulge_size_range=(5, 9),
            ncca_pairing_nt=4,
        )
        bad = architecture.localize(
            "c",
            consensus,
            row="bad",
            stem_i_extent_nt=96,
            regulatory_mode=None,
            stem_i_nt_threshold=60,
            class_ii=False,
            min_named_helices=2,
            min_helix_pairs=1,
            bulge_size_range=(5, 9),
            ncca_pairing_nt=4,
        )
        assert good.bulge_state == architecture.BULGE_DETECTED
        assert bad.bulge_state == architecture.BULGE_ABSENT

    def test_localize_has_no_defaulted_rule_parameters(self) -> None:
        """A4 ships every value keyword-required; a default would decide what gets mined."""
        import inspect

        exempt = {"candidate_id", "consensus", "row", "allow_wobble"}
        defaulted = [
            name
            for name, p in inspect.signature(architecture.localize).parameters.items()
            if name not in exempt and p.default is not inspect.Parameter.empty
        ]
        assert defaulted == []


class TestNoCmalignOnThisPath:
    """ADR-0006 D3 forbids keying (b) to a cmalign-vs-RF00230 bit-score; A4 strengthens it."""

    def test_the_module_names_neither_cmalign_nor_rf00230_in_EXECUTABLE_code(self) -> None:
        """Walked as an AST, not grepped as text.

        ⚠ A source-text grep was the first attempt and it was **wrong** — it fired on the
        docstrings, which D3 *requires* to name both instruments in order to record that
        neither is on the path.  Prose that explains the prohibition is not a violation of
        it.  The AST distinguishes the two: docstrings and comments are dropped, and what
        remains is every identifier, attribute and runtime string literal the module can
        actually act on.
        """
        import ast
        from pathlib import Path

        tree = ast.parse(Path(architecture.__file__).read_text(encoding="utf-8"))
        # Drop every docstring node so only executable code survives.
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        tokens: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in docstrings:
                    tokens.append(node.value)
            elif isinstance(node, ast.Name):
                tokens.append(node.id)
            elif isinstance(node, ast.Attribute):
                tokens.append(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                tokens.append(getattr(node, "module", "") or "")
                tokens.extend(a.name for a in node.names)
        haystack = "\n".join(tokens).lower()
        for forbidden in ("cmalign", "rf00230", "rfam", "infernal"):
            assert forbidden not in haystack, f"{forbidden} reached criterion (b)'s code path"

    def test_positive_control_the_scan_can_see_an_identifier_when_one_exists(self) -> None:
        """Without this, a scan that silently collected nothing would read as a clean pass."""
        import ast

        tree = ast.parse("x = cmalign_score\n")
        names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
        assert "cmalign_score" in names

    def test_the_predicate_signature_takes_no_score(self) -> None:
        import inspect

        params = set(inspect.signature(architecture.criterion_b).parameters)
        assert params == {
            "named_elements_present",
            "ncca_bulge_detected",
            "ultrashort_relax",
            "class_ii_relax",
        }


class TestRoundTwoGuards:
    """CodeRabbit r2's predicate-side findings, one test per finding."""

    def test_an_unrecognized_bulge_state_refuses_at_construction(self) -> None:
        """`architecture_status` tests two states by equality, so anything else would fall
        through to a DECIDED NEGATIVE (failed ⇒ minable) rather than a refusal."""
        with pytest.raises(architecture.ArchitectureError, match="bulge_state"):
            architecture.Localization(
                candidate_id="c1",
                named_elements_present=True,
                bulge_state="detceted",  # typo
                ultrashort_relax=False,
                class_ii_relax=False,
                n_sequences=24,
            )

    @pytest.mark.parametrize("state", architecture.BULGE_STATES)
    def test_positive_control_every_real_state_constructs(self, state: str) -> None:
        assert localization(bulge_state=state).bulge_state == state

    def test_a_target_restricts_the_scan_to_ONE_bulge(self) -> None:
        """Two bulges, only the second carries the motif. Targeting the first must report
        ABSENT — scanning both would report DETECTED and credit the wrong bulge."""
        # Two copies of the real antiterminator shape: an unpaired run flanked by two
        # DIFFERENT helices. (A run whose flanks close each other is a hairpin loop, which
        # `find_bulges` correctly refuses — the first draft of this fixture made that
        # mistake and produced zero bulges.)
        blank = ANTITERM_SEQ[:4] + "AAACAAA" + ANTITERM_SEQ[11:]
        seq = blank + ANTITERM_SEQ
        ss = ANTITERM_SS + ANTITERM_SS
        pairs = architecture.pair_table(ss)
        bulges = architecture.find_bulges(pairs)
        assert len(bulges) == 2
        first, second = bulges
        scan_all, _ = architecture.ncca_bulge_status(
            seq, pairs, bulge_size_range=(7, 7), ncca_pairing_nt=4
        )
        only_first, _ = architecture.ncca_bulge_status(
            seq, pairs, bulge_size_range=(7, 7), ncca_pairing_nt=4, target=first
        )
        only_second, detail = architecture.ncca_bulge_status(
            seq, pairs, bulge_size_range=(7, 7), ncca_pairing_nt=4, target=second
        )
        assert scan_all == architecture.BULGE_DETECTED
        assert only_first == architecture.BULGE_ABSENT
        assert only_second == architecture.BULGE_DETECTED
        assert detail["scanned"] == "one located bulge"

    def test_the_default_still_scans_every_bulge(self) -> None:
        """Production does not know which bulge is the antiterminator's — finding out is
        the job — so `target=None` must keep the full scan."""
        _, detail = architecture.ncca_bulge_status(
            ANTITERM_SEQ,
            architecture.pair_table(ANTITERM_SS),
            bulge_size_range=(5, 9),
            ncca_pairing_nt=4,
        )
        assert detail["scanned"] == "every flanked bulge"


class TestReadErrorsBecomeArchitectureErrors:
    """The upstream conversion that keeps one bad file from losing a whole shard.

    ⚠ Located here, not in the producer, because sabotage showed the producer's widened
    ``except`` is redundant: narrowing it back left the behaviour green, since
    ``parse_stockholm`` had already converted. The test belongs on the line that carries
    the guarantee, not the line that merely restates it.
    """

    def test_non_utf8_bytes_convert(self, tmp_path) -> None:
        path = tmp_path / "bad.sto"
        path.write_bytes(b"# STOCKHOLM 1.0\n\xff\xfe not utf-8\n")
        with pytest.raises(architecture.ArchitectureError, match="cannot read"):
            architecture.parse_stockholm(path)

    def test_a_missing_file_converts(self, tmp_path) -> None:
        with pytest.raises(architecture.ArchitectureError, match="cannot read"):
            architecture.parse_stockholm(tmp_path / "absent.sto")

    def test_an_unreadable_file_converts(self, tmp_path) -> None:
        import os

        path = tmp_path / "locked.sto"
        path.write_text("# STOCKHOLM 1.0\n", encoding="utf-8")
        os.chmod(path, 0o000)
        try:
            if os.access(path, os.R_OK):  # pragma: no cover - running as root
                pytest.skip("cannot make a file unreadable as this user")
            with pytest.raises(architecture.ArchitectureError, match="cannot read"):
                architecture.parse_stockholm(path)
        finally:
            os.chmod(path, 0o644)

    def test_positive_control_a_readable_file_parses(self, tmp_path) -> None:
        path = tmp_path / "ok.sto"
        path.write_text(sto([("s1", ANTITERM_SEQ)], ANTITERM_SS), encoding="utf-8")
        assert architecture.parse_stockholm(path).n_sequences == 1
