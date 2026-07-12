"""P1-06 leakage + selection/label unit gates for the segmentation smoke set.

Load-bearing guarantee (imp.md P1-06): **every** smoke locus is held-out-fold
(``nested_role == "heldout" & source == "corpus"``). The pure tiers run in bare CI;
the committed-fixture tier (``importorskip pandas``) asserts the same on the real
slice. Bite tests confirm each guard fails-closed on an injected violation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tbox_finder.data import seg_smoke as ss
from tbox_finder.labels import ELEMENT_COORDS

_FIXTURE_CSV = (
    Path(__file__).resolve().parents[1] / "fixtures" / "p1_seg_smoke" / "heldout_slice.csv"
)


def _rec(
    rid: str,
    *,
    source: str = "corpus",
    nested_role: str = "heldout",
    ttype: str = "Cyclic",
    window: int = 40,
    elements: dict[str, tuple[int, int]] | None = None,
    label_string: str | None = None,
) -> dict:
    """A self-consistent smoke record; ``label_string`` defaults to the P0 derivation."""
    row: dict = {
        ss.RECORD_ID_COL: rid,
        ss.SOURCE_COL: source,
        ss.NESTED_ROLE_COL: nested_role,
        ss.KLASS_COL: "II" if ttype == "Translational" else "I",
        ss.CLUSTER_COL: 1,
        ss.ORDER_COL: "OrderA",
        "Name": rid,
        ss.PHYLUM_COL: "Firmicutes",
        ss.MASTER_ORDER_COL: "OrderA",
        ss.CLASS_TYPE_COL: ttype,
        ss.WINDOW_LEN_COL: window,
    }
    for col in ss.COORD_COLS:
        row.setdefault(col, None)
    for elem, (start, end) in (elements or {}).items():
        s_col, e_col = ELEMENT_COORDS[elem]
        row[s_col], row[e_col] = start, end
    row[ss.WINDOW_SEQ_COL] = "N" * window
    row[ss.LABEL_STRING_COL] = (
        ss.derive_label_codes(row, window_length=window, naive=False)
        if label_string is None
        else label_string
    )
    return row


def _covering_pool() -> list[dict]:
    """A small held-out pool that together covers all 8 classes."""
    return [
        _rec("a1", elements={"Stem_I": (1, 40), "Specifier": (10, 12)}),  # Stem_I + S carve
        _rec("b2", elements={"Antiterminator_Tbox_seq": (5, 25), "Terminator": (18, 30)}),
        _rec("c3", elements={"Stem_II": (2, 12), "Stem_III": (14, 24)}),
        _rec("d4", elements={"Antiterminator_Tbox_seq": (5, 35), "Discriminator": (30, 34)}),
    ]


# --------------------------------------------------------------------------- #
# predicate + selection (pure, bare CI)
# --------------------------------------------------------------------------- #
def test_is_heldout_corpus_predicate() -> None:
    assert ss.is_heldout_corpus("corpus", "heldout") is True
    assert ss.is_heldout_corpus("blind", "heldout") is False
    assert ss.is_heldout_corpus("corpus", "train") is False


def test_select_subset_deterministic_and_sorted() -> None:
    pool = [_rec(f"{i:02x}", elements={"Stem_I": (1, 40)}) for i in range(12)]
    a = ss.select_smoke_subset(pool, max_loci=5, min_per_class=1, min_class_ii=0)
    b = ss.select_smoke_subset(list(reversed(pool)), max_loci=5, min_per_class=1, min_class_ii=0)
    ids_a = [r[ss.RECORD_ID_COL] for r in a]
    assert ids_a == [r[ss.RECORD_ID_COL] for r in b]  # input-order independent
    assert ids_a == sorted(ids_a)  # sorted by record_id
    assert len(a) == 5


def test_select_covers_all_element_classes() -> None:
    chosen = ss.select_smoke_subset(_covering_pool(), max_loci=10, min_per_class=1, min_class_ii=0)
    cov = ss.coverage_report(chosen)
    assert cov["all_eight_classes_present"] is True


def test_smoke_digest_row_order_independent() -> None:
    pool = _covering_pool()
    assert ss.smoke_digest(pool) == ss.smoke_digest(list(reversed(pool)))


# --------------------------------------------------------------------------- #
# leakage guard + validator bite tests (pure, bare CI)
# --------------------------------------------------------------------------- #
def test_leakage_report_100pct_heldout() -> None:
    pool = _covering_pool()
    leak = ss.leakage_report(pool)
    assert leak["positives_heldout_only"] is True
    assert leak["n_heldout_corpus"] == leak["n_records"] == len(pool)


def test_validate_flags_non_heldout_record() -> None:
    pool = _covering_pool() + [_rec("z9", nested_role="train", elements={"Stem_I": (1, 40)})]
    problems = ss.validate_smoke(pool)
    assert any("not held-out corpus" in p for p in problems)
    assert ss.leakage_report(pool)["positives_heldout_only"] is False


def test_validate_flags_external_source() -> None:
    pool = _covering_pool() + [_rec("z8", source="blind", elements={"Stem_I": (1, 40)})]
    assert any("not held-out corpus" in p for p in ss.validate_smoke(pool))


def test_validate_flags_terminator_on_class_ii() -> None:
    # A Translational (class-II) record whose label carries a Terminator code.
    bad = _rec("t1", ttype="Translational", window=6, label_string="..TT..")
    assert any("class-II" in p for p in ss.validate_smoke(_covering_pool() + [bad]))


def test_validate_flags_length_mismatch() -> None:
    bad = _rec("m1", window=40, label_string="." * 39)  # 39 != seq/window 40
    assert any("length" in p for p in ss.validate_smoke(_covering_pool() + [bad]))


def test_validate_flags_specifier_carveout_violation() -> None:
    # Stem_I + Specifier both annotated, but the label shows no 'S' → carve-out lost.
    bad = _rec(
        "cv1",
        window=40,
        elements={"Stem_I": (1, 40), "Specifier": (10, 12)},
        label_string="1" * 40,  # all Stem_I, Specifier swallowed
    )
    assert ss.coverage_report([bad])["specifier_carveout_violations"] == 1
    assert any("carve-out" in p for p in ss.validate_smoke(_covering_pool() + [bad]))


def test_validate_flags_antiterm_term_overlap_violation() -> None:
    # A Terminator code sits inside the antiterminator extent → on-conformation lost.
    bad = _rec(
        "at1",
        window=40,
        elements={"Antiterminator_Tbox_seq": (5, 25)},
        label_string="." * 9 + "T" + "." * 30,  # 'T' at pos 10, inside [5, 25]
    )
    assert ss.coverage_report([bad])["antiterm_term_violations"] == 1
    assert any("antiterminator extent" in p for p in ss.validate_smoke(_covering_pool() + [bad]))


def test_validate_flags_invalid_label_code_without_crashing() -> None:
    # An out-of-alphabet code must be reported (fail-closed), not raise KeyError.
    bad = _rec("q1", window=4, label_string="..X.")
    problems = ss.validate_smoke([bad])  # must not raise
    assert any("invalid label code" in p for p in problems)


def test_validate_flags_missing_class_coverage() -> None:
    # Only Stem_I present anywhere → the other element classes are missing.
    only_stem_i = [_rec(f"s{i}", elements={"Stem_I": (1, 40)}) for i in range(3)]
    assert any("missing 8-class coverage" in p for p in ss.validate_smoke(only_stem_i))


def test_validate_clean_pool_passes() -> None:
    assert ss.validate_smoke(_covering_pool()) == []


def test_assert_labels_consistent_bites_on_drift() -> None:
    # label_string that disagrees with derive_label_codes on the same coords.
    drifted = _rec("x1", window=8, elements={"Stem_I": (1, 8)}, label_string="." * 8)
    with pytest.raises(ValueError, match="drift"):
        ss.assert_labels_consistent([drifted])


# --------------------------------------------------------------------------- #
# padding / ignore-index (numpy tier)
# --------------------------------------------------------------------------- #
def test_labels_to_matrix_pad_and_values() -> None:
    np = pytest.importorskip("numpy")
    from tbox_finder.labels import label_string_to_indices

    pool = [
        _rec("p1", window=10, elements={"Stem_I": (1, 10)}),
        _rec("p2", window=6, elements={"Specifier": (2, 4)}),
    ]
    mat = ss.labels_to_matrix(pool)
    assert mat.dtype == np.int16
    assert mat.shape == (2, 10)
    ordered = sorted(pool, key=lambda r: r[ss.RECORD_ID_COL])
    for i, r in enumerate(ordered):
        idx = label_string_to_indices(r[ss.LABEL_STRING_COL])
        assert list(mat[i, : len(idx)]) == idx
        assert (mat[i, len(idx) :] == ss.IGNORE_INDEX).all()  # padded with -100
    assert ss.IGNORE_INDEX == -100


# --------------------------------------------------------------------------- #
# committed-fixture tier — the real held-out slice (pandas)
# --------------------------------------------------------------------------- #
def test_fixture_every_record_is_heldout_fold() -> None:
    pytest.importorskip("pandas")
    records = ss.load_fixture_csv(_FIXTURE_CSV)
    assert records, "fixture is empty"
    assert all(ss.is_heldout_corpus(r[ss.SOURCE_COL], r[ss.NESTED_ROLE_COL]) for r in records)
    leak = ss.leakage_report(records)
    assert leak["positives_heldout_only"] is True
    assert leak["n_heldout_corpus"] == leak["n_records"] == len(records)
    assert ss.validate_smoke(records) == []
