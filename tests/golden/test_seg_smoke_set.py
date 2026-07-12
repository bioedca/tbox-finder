"""Golden regression for the P1-06 held-out per-nt 8-class segmentation smoke set.

Two tiers, mirroring the P1-05 convention:

* **stdlib tier** (runs in bare CI, no pandas): re-reads the committed fixture slice
  with the ``csv`` module and recomputes the golden digest from ``(record_id,
  FASTA_sequence, label_string)`` — locking the digest independent of pandas/pyarrow.
* **pandas tier** (``importorskip``): rebuilds record dicts via the module loader and
  asserts the digest, that the reused labels *are* the P0 ``derive_label_codes``
  output, full 8-class coverage, and the ADR-0004 overlap conventions.

The fixture (``tests/fixtures/p1_seg_smoke/heldout_slice.csv``) is a real held-out
slice with coordinate columns, so the whole test reproduces without the
DVC-tracked master/labels tables (the ``ingest_sample`` reuse-a-slice precedent).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tbox_finder import ingest
from tbox_finder.data import seg_smoke as ss

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "p1_seg_smoke"
_FIXTURE_CSV = _FIXTURE_DIR / "heldout_slice.csv"
_EXPECTED = _FIXTURE_DIR / "expected.sha256"


def _stdlib_digest() -> str:
    """Recompute the smoke digest from the fixture CSV using only the stdlib."""
    with _FIXTURE_CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: r[ss.RECORD_ID_COL])
    per_record = [
        ingest.record_hash([r[ss.RECORD_ID_COL], r[ss.WINDOW_SEQ_COL], r[ss.LABEL_STRING_COL]])
        for r in rows
    ]
    return ingest.records_digest(per_record)


# --------------------------------------------------------------------------- #
# stdlib tier — runs everywhere (no pandas)
# --------------------------------------------------------------------------- #
def test_fixture_present() -> None:
    assert _FIXTURE_CSV.is_file()
    assert _EXPECTED.is_file()
    assert len(_EXPECTED.read_text().strip()) == 64


def test_seg_smoke_golden_digest_matches_stdlib() -> None:
    assert _stdlib_digest() == _EXPECTED.read_text().strip()


def test_fixture_has_all_eight_class_codes_stdlib() -> None:
    """Every 8-class single-char code appears somewhere in the committed slice."""
    with _FIXTURE_CSV.open(newline="") as fh:
        codes = {ch for row in csv.DictReader(fh) for ch in row[ss.LABEL_STRING_COL]}
    from tbox_finder.labels import CLASS_CODE, CLASS_ORDER

    assert codes == {CLASS_CODE[c] for c in CLASS_ORDER}


# --------------------------------------------------------------------------- #
# pandas tier — full rebuild via the module loader
# --------------------------------------------------------------------------- #
def test_seg_smoke_golden_digest_matches_module() -> None:
    pytest.importorskip("pandas")
    records = ss.load_fixture_csv(_FIXTURE_CSV)
    assert ss.smoke_digest(records) == _EXPECTED.read_text().strip()


def test_seg_smoke_labels_are_the_p0_derivation() -> None:
    """The reused ``label_string`` must equal ``derive_label_codes`` on the coords."""
    pytest.importorskip("pandas")
    records = ss.load_fixture_csv(_FIXTURE_CSV)
    ss.assert_labels_consistent(records)  # raises on any drift


def test_seg_smoke_coverage_and_adr0004_conventions() -> None:
    pytest.importorskip("pandas")
    records = ss.load_fixture_csv(_FIXTURE_CSV)
    cov = ss.coverage_report(records)
    assert cov["all_eight_classes_present"] is True
    assert cov["terminator_on_class_ii"] == 0  # ADR-0004 D1: Terminator is class-I only
    assert cov["specifier_carveout_violations"] == 0  # Specifier carves out of Stem_I
    assert cov["antiterm_term_violations"] == 0  # antiterm ∩ term → Antiterminator
    assert cov["n_class_ii"] >= 1  # non-vacuous class-II presence
    assert ss.validate_smoke(records) == []
