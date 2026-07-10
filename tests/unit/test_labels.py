"""Unit tests for the pure-logic label-derivation primitives (P0-20).

Stdlib-only — no pandas — so they run in the bare CI ``test`` env. They lock the
ADR-0004 D1 precedence resolution, the class-I-only terminator rule, and the
class-II-CM-naive withholding on hand-constructed rows. The crystal-structure /
hand-checked golden assertions (9 PDB extents) land in P0-21's
``tests/unit/test_label_derivation.py``.
"""

from __future__ import annotations

import math

from tbox_finder import labels


def _class_i_row() -> dict:
    """A synthetic class-I record exercising all three material ADR-0004 overlaps."""
    return {
        "type": "Transcriptional",
        "s1_start": 1,
        "s1_end": 50,
        "codon_start": 20,
        "codon_end": 22,
        "stem2_region_start": None,
        "stem2_region_end": None,
        "stem3_start": None,
        "stem3_end": None,
        "antiterm_start": 60,
        "antiterm_end": 90,
        "term_start": 70,
        "term_end": 100,
        "discrim_start": 75,
        "discrim_end": 78,
        "tbox_length": 100,
        "FASTA_sequence": "N" * 100,
        "Name": "test_class_i",
    }


# --------------------------------------------------------------------------- #
# _is_absent / element_extent
# --------------------------------------------------------------------------- #
def test_is_absent_sentinels_and_values():
    for absent in (None, float("nan"), math.nan, "", "abc", 0, -1, -2, 0.0, -1.0, 0.9):
        assert labels._is_absent(absent), absent
    for present in (1, 1.0, 5, 100, "42", 42.0):
        assert not labels._is_absent(present), present


def test_element_extent_present_absent_and_minus_strand():
    row = _class_i_row()
    assert labels.element_extent(row, "Stem_I") == (1, 50)
    assert labels.element_extent(row, "Specifier") == (20, 22)
    assert labels.element_extent(row, "Stem_II") is None  # both endpoints absent
    # minus-strand (start > end) normalizes to (min, max)
    row["antiterm_start"], row["antiterm_end"] = 90, 60
    assert labels.element_extent(row, "Antiterminator_Tbox_seq") == (60, 90)
    # a single absent endpoint → absent
    row["term_end"] = 0
    assert labels.element_extent(row, "Terminator") is None


# --------------------------------------------------------------------------- #
# Precedence painting (ADR-0004 D1)
# --------------------------------------------------------------------------- #
def test_precedence_resolution_class_i():
    codes = labels.derive_label_codes(_class_i_row(), window_length=100, naive=False)
    # spans the full window, single-label (every char a known class code)
    assert len(codes) == 100
    assert set(codes) <= set(labels.CODE_TO_CLASS)
    # Stem_I 1-50 with Specifier carved out at 20-22 (Specifier ▸ Stem_I)
    assert codes[0:19] == "1" * 19
    assert codes[19:22] == "SSS"
    assert codes[22:50] == "1" * 28
    # background between elements
    assert codes[50:59] == "." * 9
    # Antiterminator wins over Terminator where they overlap (Antiterminator ▸ Terminator)
    assert codes[59:74] == "A" * 15  # 60-74 antiterminator
    # Discriminator carves out of Antiterminator (Discriminator ▸ Antiterminator)
    assert codes[74:78] == "DDDD"  # 75-78 discriminator
    assert codes[78:90] == "A" * 12  # 79-90 antiterminator
    # Terminator survives only beyond the antiterminator (91-100)
    assert codes[90:100] == "T" * 10


def test_specifier_is_more_specific_than_stem_i():
    row = _class_i_row()
    codes = labels.derive_label_codes(row, window_length=100, naive=False)
    s0, e0 = row["codon_start"], row["codon_end"]
    assert all(codes[p - 1] == labels.CLASS_CODE["Specifier"] for p in range(s0, e0 + 1))


# --------------------------------------------------------------------------- #
# Class-II: no terminator, and class-II-CM-naive withholding
# --------------------------------------------------------------------------- #
def test_class_ii_terminator_suppressed():
    row = _class_i_row()
    row["type"] = "Translational"  # class II — no terminator hairpin (PMID:25583497)
    codes = labels.derive_label_codes(row, window_length=100, naive=False)
    assert "T" not in codes  # terminator never painted for class II
    # the region the terminator would have occupied past the antiterminator is background
    assert codes[90:100] == "." * 10
    # shared elements still painted
    assert "A" in codes and "S" in codes and "D" in codes


def test_naive_withholds_class_ii_cm_structure():
    row = _class_i_row()
    row["type"] = "Translational"
    naive = labels.derive_label_codes(row, window_length=100, naive=True)
    # class-II record → all TBDB001.cm-derived structure withheld → all background
    assert naive == "." * 100


def test_naive_equals_production_for_class_i():
    row = _class_i_row()
    prod = labels.derive_label_codes(row, window_length=100, naive=False)
    naive = labels.derive_label_codes(row, window_length=100, naive=True)
    assert prod == naive  # class-I labels are identical between the two runs


# --------------------------------------------------------------------------- #
# label_source / class flag / helpers
# --------------------------------------------------------------------------- #
def test_label_source_and_class_flag():
    assert labels.label_source({"type": "Translational"}) == labels.LABEL_SOURCE_CLASS_II
    assert labels.label_source({"type": "Transcriptional"}) == labels.LABEL_SOURCE_CLASS_I
    assert labels.label_source({}) == labels.LABEL_SOURCE_CLASS_I  # unknown → class-I default
    assert labels.is_class_ii({"type": "Translational"})
    assert not labels.is_class_ii({"type": "Transcriptional"})
    assert labels.class_of({"type": "Translational"}) == "II"
    assert labels.class_of({"type": "Transcriptional"}) == "I"
    assert labels.class_of({}) == "?"


def test_class_vocabulary_is_consistent():
    assert len(labels.CLASS_ORDER) == 8
    assert labels.CLASS_ORDER[0] == "background"
    assert set(labels.CLASS_CODE) == set(labels.CLASS_ORDER)
    assert len(set(labels.CLASS_CODE.values())) == 8  # codes are unique
    # precedence covers every non-background class exactly once
    assert set(labels.PRECEDENCE_HIGH_TO_LOW) == set(labels.CLASS_ORDER) - {"background"}
    assert len(labels.PRECEDENCE_HIGH_TO_LOW) == 7


def test_label_string_to_indices_roundtrip():
    codes = labels.derive_label_codes(_class_i_row(), window_length=100, naive=False)
    idx = labels.label_string_to_indices(codes)
    assert len(idx) == 100
    assert all(0 <= i < 8 for i in idx)
    # rebuild the code string from indices and compare
    rebuilt = "".join(labels.CLASS_CODE[labels.CLASS_ORDER[i]] for i in idx)
    assert rebuilt == codes


def test_labels_digest_is_deterministic():
    prod = ["1SS1..A", "....AAAD"]
    naive = ["1SS1..A", "........"]
    assert labels.labels_digest(prod, naive) == labels.labels_digest(prod, naive)
    # order-sensitive: swapping records changes the digest
    assert labels.labels_digest(prod, naive) != labels.labels_digest(prod[::-1], naive[::-1])
