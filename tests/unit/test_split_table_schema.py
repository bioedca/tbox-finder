"""Unit tests for P0-23 — the committed (git/LFS) split-assignment table.

Guards the carve-out contract (ADR-0004; PRD §9.2/§16) that lets the no-leakage CI
(§8.2) read the *real* ~23.5k-record partition on every PR: the committed table is
sequence-free, hash-linked to ``master_clean_v0.parquet``, carries one fold per §9.2
scheme + the variant→parent provenance column, and its provenance declares the DOME
redundancy + partition-strategy fields.

Tiers:
- schema-constant (runs bare, stdlib only): the closed allowlist carries the
  identifier + fold + lineage columns and no sequence-bearing column; ``_is_hex64``;
- pandas tier (``importorskip pandas``): ``build_split_table`` projection +
  ``validate_table_schema`` fail-loud invariants on synthetic frames;
- DOME tier (stdlib): ``dome_reporting_fields`` copies the report numbers verbatim;
- provenance-sidecar tier (stdlib; skips if absent): the committed provenance records
  the whole-file corpus hash-link + the DOME fields;
- committed-parquet tier (pandas; skips on a Git-LFS pointer / in CI): the real table
  passes the schema gate, has no sequences, and its hash-link resolves to the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder import splits

_REPO = Path(__file__).resolve().parents[2]
_COMMITTED = _REPO / "data" / "processed" / "splits" / "split_assignments.parquet"
_PROVENANCE = _REPO / "data" / "processed" / "splits" / "split_assignments.provenance.json"
_CORPUS = _REPO / "data" / "processed" / "master_clean_v0.parquet"

_HEX64 = "a" * 64
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(path: Path) -> bool:
    """True iff ``path`` is a Git-LFS pointer stub (unsmudged content, e.g. in CI)."""
    with path.open("rb") as fh:
        return fh.read(len(_LFS_MAGIC)) == _LFS_MAGIC


# --------------------------------------------------------------------------- #
# schema-constant tier (bare)
# --------------------------------------------------------------------------- #
def test_allowlist_carries_identifier_and_provenance_columns():
    for col in ("record_id", "parent_record_id", "corpus_record_sha256", "source", "cluster_id"):
        assert col in splits.COMMITTED_TABLE_COLUMNS


def test_allowlist_carries_every_lineage_and_fold_scheme_column():
    for col in (*splits.LINEAGE_COLUMNS, *splits.FOLD_SCHEME_COLUMNS):
        assert col in splits.COMMITTED_TABLE_COLUMNS


def test_allowlist_excludes_every_sequence_column():
    assert not (splits.SEQUENCE_COLUMN_DENYLIST & set(splits.COMMITTED_TABLE_COLUMNS))


def test_allowlist_is_ordered_and_deduplicated():
    cols = list(splits.COMMITTED_TABLE_COLUMNS)
    assert cols[0] == "record_id"
    assert len(cols) == len(set(cols))


def test_is_hex64_accepts_lowercase_sha256_only():
    assert splits._is_hex64(_HEX64)
    assert not splits._is_hex64("A" * 64)  # uppercase
    assert not splits._is_hex64("a" * 63)  # short
    assert not splits._is_hex64("g" * 64)  # non-hex
    assert not splits._is_hex64(None)
    assert not splits._is_hex64(123)


def test_dome_reference_carries_the_pmid():
    assert "PMID:39661723" in splits.DOME_REFERENCE


# --------------------------------------------------------------------------- #
# pandas tier — build_split_table + validate_table_schema
# --------------------------------------------------------------------------- #
def _interim_frame():
    """A minimal but well-formed DVC-interim frame (2 corpus + 1 external)."""
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "record_sha256": [_HEX64, "b" * 64, ""],  # external → empty hash-link
            "seq_name": [_HEX64, "b" * 64, "anchor:DR_ILES"],
            "source": ["corpus", "corpus", "anchor"],
            "klass": ["I", "II", "II"],
            "cluster_id": [0, 1, 2],
            "resolved_phylum": ["Actinobacteria", "Firmicutes", "Chloroflexota"],
            "resolved_class": ["Actinobacteria", "Bacilli", None],
            "resolved_order": ["Frankiales", "Bacillales", None],
            "resolved_genus": ["Frankia", "Bacillus", None],
            "fold_random": ["train", "val", "train"],
            "loo_order_unit": ["Frankiales", "Bacillales", None],
            "class_holdout_unit": ["Actinobacteria", None, None],
            "phylum_holdout_unit": ["Actinobacteria", None, None],
            "nested_train": [True, False, False],
            "nested_role": ["train", "heldout", "heldout"],
            "is_designated_loo_holdout": [False, True, False],
            "is_anchor_heldout": [False, False, True],
            "clade_crossing_cluster": [False, False, False],
            "dropped_from_clade_holdout": [False, False, False],
            "parent_record_id": [_HEX64, "b" * 64, ""],  # interim mirrors record_sha256
        }
    )


def test_build_split_table_projects_onto_committed_schema():
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame(), corpus_sha256=_HEX64)
    assert list(table.columns) == list(splits.COMMITTED_TABLE_COLUMNS)
    # seq_name → record_id (unique, sequence-free).
    assert list(table["record_id"]) == [_HEX64, "b" * 64, "anchor:DR_ILES"]
    # Every base record self-references (P2 variants overwrite when appended).
    assert (table["parent_record_id"] == table["record_id"]).all()
    # Corpus rows keep their 64-hex hash-link; the external's empty link → NA.
    assert table.loc[2, "corpus_record_sha256"] is None or table["corpus_record_sha256"].isna()[2]
    assert table.loc[0, "corpus_record_sha256"] == _HEX64


def test_validate_table_schema_accepts_a_well_formed_table():
    pytest.importorskip("pandas")
    splits.validate_table_schema(splits.build_split_table(_interim_frame()))


def test_validate_table_schema_rejects_a_missing_required_column():
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame()).drop(columns=["cluster_id"])
    with pytest.raises(ValueError, match="schema mismatch"):
        splits.validate_table_schema(table)


def test_validate_table_schema_rejects_a_sequence_column():
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame())
    table["FASTA_sequence"] = "ACGU"  # a forbidden sequence column
    with pytest.raises(ValueError, match="schema mismatch|sequence-bearing"):
        splits.validate_table_schema(table)


def test_validate_table_schema_rejects_duplicate_record_id():
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame())
    table.loc[1, "record_id"] = table.loc[0, "record_id"]
    with pytest.raises(ValueError, match="record_id is not unique"):
        splits.validate_table_schema(table)


def test_validate_table_schema_rejects_orphan_parent():
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame())
    table.loc[0, "parent_record_id"] = "no-such-record"
    with pytest.raises(ValueError, match="not resolvable"):
        splits.validate_table_schema(table)


def test_validate_table_schema_rejects_corpus_row_without_hash_link():
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame())
    table.loc[0, "corpus_record_sha256"] = "not-a-hash"  # a corpus row must be 64-hex
    with pytest.raises(ValueError, match="corpus_record_sha256 hash-link"):
        splits.validate_table_schema(table)


# --------------------------------------------------------------------------- #
# DOME tier (bare) — numbers copied verbatim from the construction report
# --------------------------------------------------------------------------- #
def test_dome_reporting_fields_copies_report_numbers_verbatim():
    report = {
        "identity_cut": 0.70,
        "coverage_cut": 0.70,
        "sweep_identities": [0.6, 0.7, 0.8, 0.9],
        "histogram": {"max_identity": 0.6994818449020386, "n_inside_cut": 0, "n_heldout": 8749},
        "diagnostics": {"clade_crossing": {"order": {"n_crossing_clusters": 129}}},
    }
    dome = splits.dome_reporting_fields(report)
    red = dome["redundancy_between_partitions"]
    assert red["identity_cut"] == 0.70
    assert red["max_heldout_to_train_identity"] == 0.6994818449020386
    assert red["n_heldout_at_or_above_cut"] == 0  # no held-out record ≥ cut vs train
    strat = dome["partition_strategy"]
    assert set(strat["schemes"]) == {
        "fold_random",
        "loo_order_unit",
        "class_holdout_unit",
        "phylum_holdout_unit",
        "nested_role",
    }
    assert strat["whole_cluster_assignment"] is True
    assert strat["clade_crossing_diagnostic"]["order"]["n_crossing_clusters"] == 129


# --------------------------------------------------------------------------- #
# committed provenance-sidecar tier (bare; skips if the sidecar is absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _PROVENANCE.exists(), reason="committed provenance not present")
def test_committed_provenance_records_corpus_hash_link_and_dome():
    prov = json.loads(_PROVENANCE.read_text())
    for field in ("rule", "script", "git_sha", "env_lock_hash", "seed", "inputs", "outputs"):
        assert field in prov
    # Whole-file hash-link to the DVC corpus is recorded as an input hash.
    corpus_key = "data/processed/master_clean_v0.parquet"
    assert corpus_key in prov["inputs"]
    assert splits._is_hex64(prov["inputs"][corpus_key])
    # DOME reporting fields present (PRD §16).
    dome = prov["extra"]["dome"]
    assert "PMID:39661723" in dome["reference"]
    assert "redundancy_between_partitions" in dome
    assert "partition_strategy" in dome
    assert dome["redundancy_between_partitions"]["n_heldout_at_or_above_cut"] == 0


# --------------------------------------------------------------------------- #
# committed-parquet tier (pandas; skips on a Git-LFS pointer / in CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _COMMITTED.exists() or _is_lfs_pointer(_COMMITTED),
    reason="committed table absent or an unsmudged Git-LFS pointer (CI)",
)
def test_committed_table_passes_the_schema_gate_and_carries_no_sequences():
    pd = pytest.importorskip("pandas")
    table = pd.read_parquet(_COMMITTED)
    splits.validate_table_schema(table)  # closed allowlist ⇒ structurally sequence-free
    assert not (splits.SEQUENCE_COLUMN_DENYLIST & set(table.columns))
    # Row count agrees with the recorded provenance (the real full-corpus partition).
    if _PROVENANCE.exists():
        prov = json.loads(_PROVENANCE.read_text())
        assert len(table) == prov["extra"]["n_records"]


@pytest.mark.skipif(
    not _COMMITTED.exists() or _is_lfs_pointer(_COMMITTED) or not _CORPUS.exists(),
    reason="committed table or DVC corpus absent / a Git-LFS pointer (CI)",
)
def test_committed_table_hash_link_resolves_to_the_live_corpus():
    """The provenance-recorded corpus hash equals the live corpus file's hash."""
    from tbox_finder import provenance

    prov = json.loads(_PROVENANCE.read_text())
    recorded = prov["inputs"]["data/processed/master_clean_v0.parquet"]
    assert recorded == provenance.sha256_file(_CORPUS)
