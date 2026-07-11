"""Golden-file regression for the §9.1 corpus-derived decoy pools (P0-30; CLAUDE.md §8.1).

``tests/fixtures/decoys_sample/`` holds a committed ``expected.sha256`` — the
whole-artifact digest of the seeded GC+length-matched-background + dinucleotide-
shuffled pools built from the 100-record ``tests/fixtures/ingest_sample/`` slice
(reused so no second data blob is committed). Only the two **corpus-derived** pools
enter the golden fixture, so it is deterministic and network-free (the Rfam
structured-RNA + tboxevo leader pools are staged, checksummed refs, exercised by
the full ``build``). This re-runs clean → ``build_corpus_pools`` on the slice and
diffs ``decoys_digest`` against the committed value, catching any drift in the
Altschul-Erikson shuffle, the GC-matched generator, or the pool-construction order.

Guarded by ``importorskip('pandas')`` — runs in the pinned ``data`` conda env and
skips (green) in the bare CI ``test`` env, where ``tests/unit/test_decoys.py``
covers the generators stdlib-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tbox_finder import decoys, ingest

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "decoys_sample"
_EXPECTED = _FIXTURE_DIR / "expected.sha256"
_INGEST_CSV = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ingest_sample" / "Master_tboxes_sample.csv"
)

# Fixed golden params (small, deterministic).
_GOLDEN = dict(seed=42, n_gc=50, n_dinuc_sources=40, dinuc_per_source=2)


def _build_golden_records():
    raw = ingest.read_raw(_INGEST_CSV)
    ingest.assert_parse_gate(raw, expected_records=100, expected_raw_cols=107)
    clean = ingest.clean(raw, expect_records=100, expect_named_cols=None)
    return decoys.build_corpus_pools(
        clean[decoys.GC_COL].astype(float).tolist(),
        clean[decoys.LEN_COL].astype(int).tolist(),
        clean[decoys.SEQ_COL].astype(str).tolist(),
        **_GOLDEN,
    )


def test_fixture_present() -> None:
    # stdlib-only guard so the fixture cannot silently disappear (runs everywhere)
    assert _EXPECTED.is_file()
    assert _INGEST_CSV.is_file()
    assert len(_EXPECTED.read_text().strip()) == 64


def test_decoys_golden_digest_matches() -> None:
    pytest.importorskip("pandas")
    records = _build_golden_records()
    # P0-30 gate: both corpus-derived pools built at the expected sizes
    n_gc = sum(r["pool"] == decoys.POOL_GC for r in records)
    n_dinuc = sum(r["pool"] == decoys.POOL_DINUC for r in records)
    assert n_gc == _GOLDEN["n_gc"]
    assert n_dinuc == _GOLDEN["n_dinuc_sources"] * _GOLDEN["dinuc_per_source"]
    # P0-30 gate: each dinuc record preserves its source's dinucleotide composition
    for r in records:
        if r["pool"] == decoys.POOL_DINUC:
            assert sum(decoys.dinucleotide_counts(r["sequence"]).values()) == r["length"] - 1
        assert r["length"] == len(r["sequence"])
    assert decoys.decoys_digest(records) == _EXPECTED.read_text().strip()
