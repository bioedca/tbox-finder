"""Golden-file regression for the P3-01 Stage-2 supervised dataset (CLAUDE.md §8.1).

``tests/fixtures/stage2_dataset/`` holds a committed ``expected.sha256`` — the
whole-artifact digest of the dataset assembled from the 100-record
``tests/fixtures/ingest_sample/`` slice (reused, so no second data blob is committed),
its re-derived P0-20 labels, the **real committed** ADR-0004 split table (git-LFS, so it
is present in CI), and the two deterministic corpus-derived decoy pools. It catches drift
in the T→U transcription, the structure projection and its consensus-gap rule, the fold
inheritance, the column order, or the digest itself.

The fixture slice predates the corpus de-duplication that produced
``master_clean_v0.parquet``, so **6 of its 100 records have no row in the committed split
table**. The build is therefore run with ``require_full_join=False`` and the refused
count is asserted to be exactly 6 — the relaxation is pinned, not silent, and a change in
join coverage turns this test red rather than quietly shrinking the dataset.

Guarded by ``importorskip('pandas')``, like every other golden here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tbox_finder import decoys as decoys_mod
from tbox_finder import ingest, labels
from tbox_finder.stage2 import dataset as ds

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "stage2_dataset"
_EXPECTED = _FIXTURE_DIR / "expected.sha256"
_INGEST_CSV = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ingest_sample" / "Master_tboxes_sample.csv"
)
_SPLIT_TABLE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "splits"
    / "split_assignments.parquet"
)

#: Fixed, small, network-free decoy params (mirrors ``tests/golden/test_decoys_golden.py``).
_DECOY_PARAMS = dict(seed=42, n_gc=50, n_dinuc_sources=40, dinuc_per_source=2)

#: The fixture slice's join coverage against the committed split table. Pinned so a
#: change in either the slice or the corpus partition is visible here. The 4 refused
#: decoys are the dinucleotide shuffles whose parent is one of the 6 unjoined records —
#: ADR-0004 D7's fail-closed rule biting on real data inside the golden, so this fixture
#: also proves the rule is reachable rather than merely coded.
_EXPECTED_JOINED = 94
_EXPECTED_REFUSED = 6
_EXPECTED_DECOYS = 126
_EXPECTED_DECOYS_REFUSED = 4
_EXPECTED_ANCHORED = 79


def _build():
    import pandas as pd

    raw = ingest.read_raw(_INGEST_CSV)
    ingest.assert_parse_gate(raw, expected_records=100, expected_raw_cols=107)
    clean = ingest.clean(raw, expect_records=100, expect_named_cols=None)
    record_ids = ingest.compute_record_hashes(clean)

    labels_df, _report, _prod, _naive = labels.derive_labels(clean)

    decoy_records = decoys_mod.build_corpus_pools(
        clean[decoys_mod.GC_COL].astype(float).tolist(),
        clean[decoys_mod.LEN_COL].astype(int).tolist(),
        clean[decoys_mod.SEQ_COL].astype(str).tolist(),
        record_ids=record_ids,
        **_DECOY_PARAMS,
    )
    decoys_df = pd.DataFrame(decoy_records)
    # `build_corpus_pools` predates masking (applied by the full `build`); these pools are
    # corpus-derived and coordinate-free, so nothing in them is maskable.
    decoys_df["masked"] = False
    decoys_df["mask_reason"] = None

    splits_df = pd.read_parquet(_SPLIT_TABLE)
    return ds.build_dataset(
        corpus=clean,
        labels=labels_df,
        splits=splits_df,
        decoys=decoys_df,
        require_full_join=False,
    )


def test_fixture_present() -> None:
    # stdlib-only guard so the fixture cannot silently disappear (runs everywhere)
    assert _EXPECTED.is_file()
    assert _INGEST_CSV.is_file()
    assert len(_EXPECTED.read_text().strip()) == 64


def test_split_table_is_smudged_not_an_lfs_pointer() -> None:
    """A git-lfs pointer is ~132 bytes and would make this golden silently unbuildable."""
    assert _SPLIT_TABLE.is_file()
    assert _SPLIT_TABLE.stat().st_size > 1_000_000


def test_stage2_dataset_golden_digest_matches() -> None:
    pytest.importorskip("pandas")
    frame, report = _build()

    # P3-01 gate: the join coverage the fixture is pinned at
    assert report["n_positives"] == _EXPECTED_JOINED
    assert report["n_corpus_refused_unjoined"] == _EXPECTED_REFUSED
    assert report["n_decoys_refused_no_parent"] == _EXPECTED_DECOYS_REFUSED
    assert report["n_negatives"] == _EXPECTED_DECOYS
    assert report["n_rows"] == _EXPECTED_JOINED + _EXPECTED_DECOYS
    # every built decoy is either admitted with a fold or refused — none vanishes
    built = (
        _DECOY_PARAMS["n_gc"] + _DECOY_PARAMS["n_dinuc_sources"] * _DECOY_PARAMS["dinuc_per_source"]
    )
    assert built == _EXPECTED_DECOYS + _EXPECTED_DECOYS_REFUSED

    # P3-01 gate: schema, transcription, context bound, fold + parentage on every row
    assert list(frame.columns) == list(ds.OUTPUT_COLUMNS)
    assert not frame["rna_sequence"].str.contains("T").any()
    assert (frame["n_tokens"] <= 1024).all()
    assert frame["fold_random"].notna().all()
    assert frame["parent_record_id"].notna().all()
    assert set(frame["fold_basis"]) <= set(ds.FOLD_BASES)

    # P3-01 gate: every anchored pairing target is co-extensive and well-formed
    anchored = frame[frame["pairing_status"] == ds.PAIRING_ANCHORED]
    assert len(anchored) == _EXPECTED_ANCHORED, "structure-target coverage drifted"
    for db, n in zip(anchored["pairing_dotbracket"], anchored["seq_length"], strict=True):
        assert len(db) == n
        ds.dot_bracket_to_partners(db)  # raises if unbalanced

    assert ds.dataset_digest(frame) == _EXPECTED.read_text().strip()
