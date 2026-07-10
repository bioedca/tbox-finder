"""Golden-file regression for the 8-class label derivation (P0-20; CLAUDE.md §8.1).

``tests/fixtures/labels_sample/`` holds a committed ``expected.sha256`` — the
whole-artifact digest of the per-record (production, class-II-CM-naive) label pair
derived from the 100-record ``tests/fixtures/ingest_sample/`` slice (reused so no
second data blob is committed; the labels derive from the ingest output, so tracking
the ingest fixture is correct). This re-runs clean → ``derive_labels`` on the slice
and diffs ``labels_digest`` against the committed value, catching any drift in the
ADR-0004 D1 precedence, the class-I-only terminator rule, or the class-II-CM-naive
withholding. It also re-asserts the P0-20 gate: every per-nt vector spans its window
and is single-label, and no record is dropped.

Guarded by ``importorskip('pandas')`` — runs in the pinned ``data`` conda env and
skips (green) in the bare CI ``test`` env, where ``tests/unit/test_labels.py``
provides stdlib-only coverage of the derivation primitives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tbox_finder import ingest, labels

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "labels_sample"
_EXPECTED = _FIXTURE_DIR / "expected.sha256"
_INGEST_CSV = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ingest_sample" / "Master_tboxes_sample.csv"
)


def test_fixture_present() -> None:
    # stdlib-only guard so the fixture cannot silently disappear (runs everywhere)
    assert _EXPECTED.is_file()
    assert _INGEST_CSV.is_file()
    assert len(_EXPECTED.read_text().strip()) == 64


def test_labels_golden_digest_matches() -> None:
    pytest.importorskip("pandas")
    raw = ingest.read_raw(_INGEST_CSV)
    ingest.assert_parse_gate(raw, expected_records=100, expected_raw_cols=107)
    clean = ingest.clean(raw, expect_records=100, expect_named_cols=None)

    labels_df, report, prod_codes, naive_codes = labels.derive_labels(clean)

    # P0-20 gate: no record dropped
    assert len(labels_df) == 100 == report["n_records"]
    assert report["records_dropped"] == 0
    # P0-20 gate: every per-nt vector spans the full window and is single-label
    for prod, naive, window in zip(
        prod_codes, naive_codes, labels_df["window_length"], strict=True
    ):
        assert len(prod) == window and len(naive) == window
        assert set(prod) <= set(labels.CODE_TO_CLASS)
        assert set(naive) <= set(labels.CODE_TO_CLASS)
    # P0-20 gate: the class-II-CM-naive flag is present and correct, and naive withholds
    for _, r in labels_df.iterrows():
        is_ii = r["label_source"] == labels.LABEL_SOURCE_CLASS_II
        assert r["class_ii_cm_naive"] == is_ii
        if is_ii:
            assert set(r["naive_label_string"]) <= {labels.CLASS_CODE["background"]}

    digest = labels.labels_digest(prod_codes, naive_codes)
    assert digest == _EXPECTED.read_text().strip()
