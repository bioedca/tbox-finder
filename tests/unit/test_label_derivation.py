"""P0-21 — label-integrity guard: P0-20 derivation vs the 9 crystal fixtures.

Asserts that ``src/tbox_finder/labels.py`` (P0-20) reproduces:

1. the **9 crystal-structure-derived per-element extents** (Stem_I ×5 + Stem_II ×1
   physically resolved by the depositions; the rest un-resolved, see the fixture README);
2. the **hand-checked synthetic fixture** carrying the exhaustive ADR-0004 D1 precedence
   coverage (all three material overlaps + residual precedence pairs); and
3. the **class-II-CM-naive withholding** — the naive run withholds ALL
   ``TBDB001.cm``-derived structure on the 3 real Actinobacteria *ileS* class-II crystals
   (PRD §8; ADR-0004 D1; the naive-run-withholds-``TBDB001.cm`` assertion of imp.md P0-21).

Fixtures are real crystal-derived extents (CLAUDE.md §8.7 — never mocks). The pure-logic
tier is stdlib-only (no pandas) so it runs in the bare CI ``test`` env; one end-to-end
tier is ``pandas``-gated (``importorskip``), mirroring the repo's 2-tier convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder import labels

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "pdb_element_extents"
EXPECTED_PDB_ENTRIES = {
    "4LCK",
    "4MGN",
    "2KZL",
    "6POM",
    "UCKOPSB",
    "WCBNVLFN",
    "6UFG",
    "6UFH",
    "6UFM",
}
REAL_CLASS_II_ILES_CRYSTALS = {"6UFG", "6UFH", "6UFM"}  # Actinobacteria ileS (TBDB001.cm)


# --------------------------------------------------------------------------- #
# Fixture loading + helpers (stdlib only)
# --------------------------------------------------------------------------- #
def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


PDB = _load("pdb_element_extents.json")
SYNTH = _load("synthetic_handchecked.json")
PDB_ENTRIES = {e["name"]: e for e in PDB["entries"]}
SYNTH_ENTRIES = {e["name"]: e for e in SYNTH["entries"]}
ALL_ENTRIES = [*PDB["entries"], *SYNTH["entries"]]


def _expand_runs(runs: list[list]) -> str:
    """Expand a run-length list ``[[code, count], ...]`` to the label string."""
    return "".join(code * count for code, count in runs)


def _row_from_entry(entry: dict, *, with_sequence: bool = False) -> dict:
    """Build a labels.py input row from a fixture entry's extents.

    Only the resolved elements set their coordinate columns; absent elements are left
    unset (``row.get`` → ``None`` → ``labels._is_absent`` True).
    """
    window = entry["window_length"]
    row: dict = {"type": entry["regulatory_mode"], "tbox_length": window, "Name": entry["name"]}
    for element, (start_col, end_col) in labels.ELEMENT_COORDS.items():
        if element in entry["extents"]:
            start, end = entry["extents"][element]
            row[start_col], row[end_col] = start, end
    if with_sequence:
        row["FASTA_sequence"] = "N" * window
    return row


# --------------------------------------------------------------------------- #
# Fixture integrity (the fixtures themselves are self-consistent)
# --------------------------------------------------------------------------- #
def test_pdb_manifest_has_the_nine_entries():
    assert set(PDB_ENTRIES) == EXPECTED_PDB_ENTRIES
    assert len(PDB["entries"]) == 9


@pytest.mark.parametrize("entry", ALL_ENTRIES, ids=lambda e: e["name"])
def test_expected_runs_span_the_window_and_are_single_label(entry):
    prod = _expand_runs(entry["expected_runs"])
    naive = _expand_runs(entry["expected_naive_runs"])
    window = entry["window_length"]
    assert len(prod) == window, entry["name"]
    assert len(naive) == window, entry["name"]
    # every char is a known 8-class code (single-label vocabulary)
    assert set(prod) <= set(labels.CODE_TO_CLASS), entry["name"]
    assert set(naive) <= set(labels.CODE_TO_CLASS), entry["name"]


@pytest.mark.parametrize("entry", PDB["entries"], ids=lambda e: e["name"])
def test_pdb_extents_are_in_frame(entry):
    """Every recorded crystal extent + apical diagnostic lies within the leader window."""
    window = entry["window_length"]
    for element, (start, end) in entry["extents"].items():
        assert 1 <= start <= end <= window, f"{entry['name']}:{element}"
    apical = entry.get("apical_diagnostic")
    if apical is not None:
        a_start, a_end = apical
        assert 1 <= a_start <= a_end <= window, f"{entry['name']}:apical"


# --------------------------------------------------------------------------- #
# THE label-integrity assertions: P0-20 output == the fixtures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry", ALL_ENTRIES, ids=lambda e: e["name"])
def test_derive_label_codes_matches_fixture(entry):
    """derive_label_codes reproduces the crystal-derived / hand-checked label vector."""
    row = _row_from_entry(entry)
    window = entry["window_length"]
    prod = labels.derive_label_codes(row, window_length=window, naive=False)
    naive = labels.derive_label_codes(row, window_length=window, naive=True)
    assert prod == _expand_runs(entry["expected_runs"]), entry["name"]
    assert naive == _expand_runs(entry["expected_naive_runs"]), entry["name"]


def test_synthetic_fixture_exercises_every_material_overlap():
    """The class-I synthetic fixture resolves all three material ADR-0004 overlaps."""
    entry = SYNTH_ENTRIES["synthetic_class_I_all_elements"]
    codes = labels.derive_label_codes(
        _row_from_entry(entry), window_length=entry["window_length"], naive=False
    )
    # Specifier carves out of Stem_I (codon 8-10)
    assert codes[7:10] == "SSS"
    assert codes[0:7] == "1111111" and codes[10:15] == "11111"
    # Stem_II wins over Stem_I (16-20) and over Stem_III (24-25)
    assert codes[15:25] == "2" * 10
    assert codes[25:30] == "3" * 5  # Stem_III standalone 26-30
    # Antiterminator wins over Terminator (40-45); Discriminator wins over both (43-46)
    assert codes[30:42] == "A" * 12  # 31-42 antiterminator
    assert codes[42:46] == "DDDD"  # 43-46 discriminator (over antiterm 43-45 + term 46)
    assert codes[46:55] == "T" * 9  # 47-55 terminator standalone
    assert codes[55:60] == "." * 5  # background


# --------------------------------------------------------------------------- #
# Class-II-CM-naive withholding — the naive-run-withholds-TBDB001.cm assertion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(REAL_CLASS_II_ILES_CRYSTALS))
def test_naive_withholds_tbdb001_structure_on_real_classII_crystals(name):
    """On the 3 real ileS class-II crystals: class-II CM source + naive all-background."""
    entry = PDB_ENTRIES[name]
    row = _row_from_entry(entry)
    assert labels.is_class_ii(row)
    assert labels.label_source(row) == labels.LABEL_SOURCE_CLASS_II
    naive = labels.derive_label_codes(row, window_length=entry["window_length"], naive=True)
    # NO TBDB001.cm-derived structure survives into the naive target
    assert set(naive) == {labels.CLASS_CODE["background"]}, name


def test_6UFM_production_paints_stem_ii_but_naive_withholds_it():
    """6UFM is the load-bearing case: production paints Stem_II, naive withholds it."""
    entry = PDB_ENTRIES["6UFM"]
    row = _row_from_entry(entry)
    window = entry["window_length"]
    prod = labels.derive_label_codes(row, window_length=window, naive=False)
    naive = labels.derive_label_codes(row, window_length=window, naive=True)
    assert labels.CLASS_CODE["Stem_II"] in prod  # production has the crystal Stem_II
    assert set(naive) == {labels.CLASS_CODE["background"]}  # naive withholds all structure


def test_class_i_crystals_have_identical_production_and_naive_labels():
    for entry in PDB["entries"]:
        if entry["class"] != "I":
            continue
        row = _row_from_entry(entry)
        window = entry["window_length"]
        prod = labels.derive_label_codes(row, window_length=window, naive=False)
        naive = labels.derive_label_codes(row, window_length=window, naive=True)
        assert prod == naive, entry["name"]


def test_class_ii_synthetic_suppresses_terminator_and_naive_withholds():
    entry = SYNTH_ENTRIES["synthetic_class_II_all_elements"]
    row = _row_from_entry(entry)
    window = entry["window_length"]
    prod = labels.derive_label_codes(row, window_length=window, naive=False)
    naive = labels.derive_label_codes(row, window_length=window, naive=True)
    assert labels.CLASS_CODE["Terminator"] not in prod  # class II has no terminator
    assert labels.CLASS_CODE["Antiterminator_Tbox_seq"] in prod  # shared element kept
    assert set(naive) == {labels.CLASS_CODE["background"]}


# --------------------------------------------------------------------------- #
# Cross-source label-noise ceiling C (ADR-0004 D6) — the reported artifact is honest
# --------------------------------------------------------------------------- #
def test_label_noise_ceiling_reports_unresolved_functional_classes():
    ceiling = PDB["label_noise_ceiling"]
    resolved = ceiling["cross_source_resolved_classes"]
    # the depositions resolve only Stem_I + Stem_II to residue extents; the four
    # overlap-defining elements are unresolved (so C cannot lower the 0.80 floor)
    assert set(resolved["Stem_I"]) == {"4LCK", "4MGN", "2KZL", "UCKOPSB", "WCBNVLFN"}
    assert resolved["Stem_II"] == ["6UFM"]
    for element in ("Specifier", "Antiterminator_Tbox_seq", "Terminator", "Discriminator"):
        assert resolved[element] == [], element
    # the reported extents match what the manifest actually encodes (no drift)
    for name in resolved["Stem_I"]:
        assert "Stem_I" in PDB_ENTRIES[name]["extents"]
    assert "Stem_II" in PDB_ENTRIES["6UFM"]["extents"]


# --------------------------------------------------------------------------- #
# End-to-end tier (pandas-gated) — the orchestration path wires the flags correctly
# --------------------------------------------------------------------------- #
def test_derive_labels_end_to_end_on_the_fixture_frame():
    pd = pytest.importorskip("pandas")
    rows = [_row_from_entry(e, with_sequence=True) for e in ALL_ENTRIES]
    df = pd.DataFrame(rows)
    labels_df, report, prod_codes, naive_codes = labels.derive_labels(df)

    assert len(labels_df) == len(ALL_ENTRIES)
    by_name = {r["name"]: r for _, r in labels_df.iterrows()}
    for entry in ALL_ENTRIES:
        out = by_name[entry["name"]]
        assert out["label_string"] == _expand_runs(entry["expected_runs"]), entry["name"]
        assert out["naive_label_string"] == _expand_runs(entry["expected_naive_runs"]), entry[
            "name"
        ]
        is_ii = entry["class"] == "II"
        assert bool(out["class_ii_cm_naive"]) is is_ii, entry["name"]
        assert out["label_source"] == (
            labels.LABEL_SOURCE_CLASS_II if is_ii else labels.LABEL_SOURCE_CLASS_I
        ), entry["name"]
    # no record dropped; the 3 real ileS class-II crystals are withheld in the naive run
    assert report["records_dropped"] == 0
    assert report["class_ii_cm_naive_withheld_records"] == sum(
        1 for e in ALL_ENTRIES if e["class"] == "II"
    )
