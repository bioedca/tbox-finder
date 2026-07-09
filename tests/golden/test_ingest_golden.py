"""Golden-file regression for the Master_tboxes ingest stage (P0-12; CLAUDE.md §8.1).

``tests/fixtures/ingest_sample/`` holds a 100-record real-data slice of
``Master_tboxes.csv`` (the faithful leading-comma export format) plus a committed
``expected.sha256``. This test re-runs the cleaning contract on the fixture and
diffs the whole-artifact digest (``records_digest`` over the ordered per-record
hashes) against the committed value, so any drift in the parse → drop-index →
sentinel-normalise pipeline is caught.

Guarded by ``importorskip('pandas')`` — it runs in the pinned ``data`` conda env
(and any env with pandas) and skips (green) in the bare CI ``test`` env, where the
stdlib hash-contract golden in ``tests/unit/test_ingest.py`` provides coverage.
No parquet engine is needed (CSV in, hashes out), so ``pyarrow`` is not required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tbox_finder import ingest

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingest_sample"
_FIXTURE_CSV = _FIXTURE_DIR / "Master_tboxes_sample.csv"
_EXPECTED = _FIXTURE_DIR / "expected.sha256"


def test_fixture_present() -> None:
    # stdlib-only guard so the fixture cannot silently disappear (runs everywhere)
    assert _FIXTURE_CSV.is_file()
    assert _EXPECTED.is_file()
    assert len(_EXPECTED.read_text().strip()) == 64


def test_ingest_golden_digest_matches() -> None:
    pytest.importorskip("pandas")
    raw = ingest.read_raw(_FIXTURE_CSV)
    # the fixture is a 100-record slice with the leading unnamed index column
    ingest.assert_parse_gate(raw, expected_records=100, expected_raw_cols=107)
    clean = ingest.clean(raw, expect_records=100, expect_named_cols=None)
    assert clean.shape == (100, 106)
    digest = ingest.records_digest(ingest.compute_record_hashes(clean))
    assert digest == _EXPECTED.read_text().strip()
