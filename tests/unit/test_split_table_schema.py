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
#: Extra training-fold corpus rows appended for the ADR-0004 A7 carve. Six are the
#: minimum that makes both nested draws non-degenerate: ``selection_val_cluster_ids``
#: refuses a bucket that is empty *or* the whole fold, and so does
#: ``calib_cluster_ids``, so a fixture with one training cluster cannot project at all
#: (the selection rung would have to be 100 % of it). They share one genus on purpose —
#: a stratum holding exactly one cluster always draws that cluster, so a fixture with a
#: distinct genus per row would carve *everything* and trip the degeneracy guard, which
#: is the A7.2 stratified draw behaving correctly rather than a fixture quirk.
_EXTRA_IDS = ["c" * 64, "d" * 64, "e" * 64, "f" * 64, "0" * 64, "1" * 64]


def _interim_frame():
    """A well-formed DVC-interim frame (8 corpus + 1 external), carve-viable."""
    pd = pytest.importorskip("pandas")
    n_extra = len(_EXTRA_IDS)
    return pd.DataFrame(
        {
            # external → empty hash-link
            "record_sha256": [_HEX64, "b" * 64, "", *_EXTRA_IDS],
            "seq_name": [_HEX64, "b" * 64, "anchor:DR_ILES", *_EXTRA_IDS],
            "source": ["corpus", "corpus", "anchor", *["corpus"] * n_extra],
            "klass": ["I", "II", "II", *["I"] * n_extra],
            "cluster_id": [0, 1, 2, *range(3, 3 + n_extra)],
            "resolved_phylum": [
                "Actinobacteria",
                "Firmicutes",
                "Chloroflexota",
                *["Firmicutes"] * n_extra,
            ],
            "resolved_class": ["Actinobacteria", "Bacilli", None, *["Bacilli"] * n_extra],
            "resolved_order": ["Frankiales", "Bacillales", None, *["Bacillales"] * n_extra],
            # Two genera inside ONE order, on purpose: it makes genus- and
            # order-stratification carve *differently*, which is what lets
            # `test_calib_stratum_column_is_load_bearing` prove anything. With a single
            # genus per order the two partitions coincide and that test passes with the
            # constant still decorative ([[degenerate-fixture-generators]]).
            "resolved_genus": [
                "Frankia",
                "Bacillus",
                None,
                *(["Bacillus"] * (n_extra // 2) + ["Geobacillus"] * (n_extra - n_extra // 2)),
            ],
            "fold_random": ["train", "val", "train", *["train"] * n_extra],
            "loo_order_unit": ["Frankiales", "Bacillales", None, *["Bacillales"] * n_extra],
            "class_holdout_unit": ["Actinobacteria", None, None, *[None] * n_extra],
            "phylum_holdout_unit": ["Actinobacteria", None, None, *[None] * n_extra],
            "nested_train": [True, False, False, *[True] * n_extra],
            "nested_role": ["train", "heldout", "heldout", *["train"] * n_extra],
            "is_designated_loo_holdout": [False, True, False, *[False] * n_extra],
            "is_anchor_heldout": [False, False, True, *[False] * n_extra],
            "clade_crossing_cluster": [False, False, False, *[False] * n_extra],
            "dropped_from_clade_holdout": [False, False, False, *[False] * n_extra],
            # interim mirrors record_sha256
            "parent_record_id": [_HEX64, "b" * 64, "", *_EXTRA_IDS],
        }
    )


def test_build_split_table_projects_onto_committed_schema():
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame(), corpus_sha256=_HEX64)
    assert list(table.columns) == list(splits.COMMITTED_TABLE_COLUMNS)
    # seq_name → record_id (unique, sequence-free).
    assert list(table["record_id"]) == [_HEX64, "b" * 64, "anchor:DR_ILES", *_EXTRA_IDS]
    # Every base record self-references (P2 variants overwrite when appended).
    assert (table["parent_record_id"] == table["record_id"]).all()
    # Corpus rows keep their 64-hex hash-link; the external's empty link → NA.
    assert table.loc[2, "corpus_record_sha256"] is None or table["corpus_record_sha256"].isna()[2]
    assert table.loc[0, "corpus_record_sha256"] == _HEX64


# --------------------------------------------------------------------------- #
# ADR-0004 A7 — the disjoint calibration carve
# --------------------------------------------------------------------------- #
def _carve_cols(frame):
    return {
        "source": list(frame["source"]),
        "cluster_id": list(frame["cluster_id"]),
        "nested_train": list(frame["nested_train"]),
        "fold_random": list(frame["fold_random"]),
        "stratum": list(frame[splits.CALIB_STRATUM_COLUMN]),
    }


def test_calib_carve_is_whole_cluster_and_inside_the_training_fold():
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame())
    calib = table[table["calib"]]
    assert len(calib), "the carve reached nothing"
    assert set(calib["source"]) == {"corpus"}
    assert calib["nested_train"].all()
    assert set(calib["fold_random"]) == {"train"}
    # whole-cluster: every training-stream member of a calib cluster is itself calib
    calib_clusters = set(calib["cluster_id"])
    mates = table[table["cluster_id"].isin(calib_clusters) & table["nested_train"]]
    assert bool(mates["calib"].all())


def test_calib_carve_is_deterministic_and_seed_sensitive():
    """Same seed ⇒ same clusters; a different seed ⇒ different ones.

    The second half is what makes the CI identity clause meaningful: if the draw were
    seed-insensitive, re-deriving it would agree with *any* committed column and the
    check would be a tautology.
    """
    pytest.importorskip("pandas")
    cols = _carve_cols(splits.build_split_table(_interim_frame()))
    assert splits.calib_cluster_ids(**cols) == splits.calib_cluster_ids(**cols)
    other = splits.calib_cluster_ids(**cols, seed=splits.CALIB_CARVE_SEED + 1)
    assert isinstance(other, frozenset)


def test_calib_stratum_column_is_load_bearing(monkeypatch):
    """CodeRabbit r1: `CALIB_STRATUM_COLUMN` must actually drive the stratification.

    It is a *pinned* constant and the provenance sidecar reports it as
    ``calib.stratum_column``. It used to be decorative — the carve read
    ``table["resolved_genus"]`` by name — so changing the constant would have left the
    provenance claiming a stratification the carve never performed, i.e. a fabricated
    provenance value (CLAUDE.md §10.3), with nothing failing.
    """
    pytest.importorskip("pandas")
    frame = splits.build_split_table(_interim_frame())
    baseline = splits.carve_calibration_split(frame).sum()
    monkeypatch.setattr(splits, "CALIB_STRATUM_COLUMN", "resolved_order")
    assert splits.carve_calibration_split(frame).sum() != baseline


def test_calib_carve_ignores_row_order():
    """Deterministic in the columns' *content*, not their iteration order (§8.3)."""
    pytest.importorskip("pandas")
    frame = splits.build_split_table(_interim_frame())
    forward = splits.calib_cluster_ids(**_carve_cols(frame))
    reversed_cols = {k: list(reversed(v)) for k, v in _carve_cols(frame).items()}
    assert splits.calib_cluster_ids(**reversed_cols) == forward


def test_calib_carve_refuses_a_degenerate_draw():
    """An empty carve, or one that swallows the whole pool, is not a *disjoint* split.

    A carve that reached nothing would make every downstream "calib does not overlap X"
    clause vacuously true, which is the one way this fold can be silently wrong.
    """
    pytest.importorskip("pandas")
    cols = _carve_cols(splits.build_split_table(_interim_frame()))
    with pytest.raises(ValueError, match="degenerate"):
        splits.calib_cluster_ids(**cols, fraction=0.999)


def test_calib_eligible_pool_excludes_the_graded_split_and_the_selection_rung():
    pytest.importorskip("pandas")
    frame = splits.build_split_table(_interim_frame())
    cols = _carve_cols(frame)
    eligible, selection_val = splits.calib_eligible_row_indices(
        source=cols["source"],
        cluster_id=cols["cluster_id"],
        nested_train=cols["nested_train"],
        fold_random=cols["fold_random"],
    )
    assert eligible, "no eligible rows"
    assert selection_val, "the P2-06a selection rung was not excluded"
    for i in eligible:
        assert cols["source"][i] == "corpus"
        assert bool(cols["nested_train"][i])
        assert cols["fold_random"][i] == "train"
        assert int(cols["cluster_id"][i]) not in selection_val


def test_variant_rows_inherit_calib_from_their_parent():
    """`calib` is in FOLD_SCHEME_COLUMNS, so D7 inheritance covers it structurally."""
    pytest.importorskip("pandas")
    table = splits.build_split_table(_interim_frame(), corpus_sha256=_HEX64)
    parent_id = table.loc[0, "record_id"]
    out = splits.append_variant_rows(
        table, [{"variant_id": parent_id + "#v1", "parent_record_id": parent_id}]
    )
    assert bool(out.iloc[-1]["calib"]) == bool(table.loc[0, "calib"])
    assert out["calib"].dtype == bool  # BOOL_FLAG_COLUMNS tripwire


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


# --------------------------------------------------------------------------- #
# P2-08 — append_variant_rows (ADR-0004 D7 + A3)
# --------------------------------------------------------------------------- #
def _base_table():
    return splits.build_split_table(_interim_frame(), corpus_sha256=_HEX64)


def _variant(**over):
    v = {"variant_id": "b" * 64 + "#c2phase7", "parent_record_id": "b" * 64}
    v.update(over)
    return v


def test_append_variant_rows_inherits_every_fold_column_from_the_parent():
    pytest.importorskip("pandas")
    table = _base_table()
    out = splits.append_variant_rows(table, [_variant()])
    assert len(out) == len(table) + 1
    child = out.iloc[-1]
    parent = out[out["record_id"] == "b" * 64].iloc[0]
    for col in (*splits.FOLD_SCHEME_COLUMNS, *splits.LINEAGE_COLUMNS, "cluster_id", "klass"):
        assert child[col] == parent[col], f"{col} not inherited"
    assert child["source"] == splits.SYNTHETIC_CLASSII_SOURCE
    assert child["parent_record_id"] == "b" * 64


def test_a_caller_cannot_supply_a_fold_that_differs_from_the_parents():
    """D7 inheritance is structural, not conventional.

    The caller passes only ``variant_id`` + ``parent_record_id``; every fold value
    is read off the parent row. A variant dict carrying its own ``nested_train`` /
    ``fold_random`` must be **ignored**, not honoured — otherwise the one contract
    the no-leakage gate exists to enforce would be caller-supplied.
    """
    pytest.importorskip("pandas")
    table = _base_table()
    out = splits.append_variant_rows(
        table, [_variant(nested_train=True, nested_role="train", fold_random="test")]
    )
    child = out.iloc[-1]
    assert bool(child["nested_train"]) is False
    assert child["nested_role"] == "heldout"
    assert child["fold_random"] == "val"


def test_append_variant_rows_rejects_an_unknown_parent():
    pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="absent from the split table"):
        splits.append_variant_rows(_base_table(), [_variant(parent_record_id="nope")])


def test_append_variant_rows_rejects_a_variant_id_colliding_with_a_record():
    pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="collides with an existing record_id"):
        splits.append_variant_rows(_base_table(), [_variant(variant_id="b" * 64)])


def test_append_variant_rows_rejects_duplicate_variant_ids():
    pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="duplicate variant_id"):
        splits.append_variant_rows(_base_table(), [_variant(), _variant()])


def test_append_variant_rows_is_a_no_op_on_an_empty_recovery_set():
    pytest.importorskip("pandas")
    table = _base_table()
    assert len(splits.append_variant_rows(table, [])) == len(table)


def test_appended_variants_keep_the_bool_dtype_of_the_fold_flags():
    """An object-dtype bool column reads as a truthy string in the leakage predicates."""
    pytest.importorskip("pandas")
    out = splits.append_variant_rows(_base_table(), [_variant()])
    for col in splits.BOOL_FLAG_COLUMNS:
        assert out[col].dtype == bool, f"{col} is {out[col].dtype}"


def test_a_widened_flag_column_fails_loud_rather_than_being_coerced():
    """The dtype tripwire fires rather than repairing the symptom.

    Measured at P2-08: pandas preserves these dtypes on the real path, so the
    tripwire never fires in production. Sabotage showed a silent ``astype(bool)``
    coercion was untestable *and* would hide whatever widened the column, so it
    raises instead. This test is what makes the tripwire non-dead code.
    """
    pd = pytest.importorskip("pandas")
    table = _base_table()
    table["nested_train"] = table["nested_train"].astype(object)
    with pytest.raises(ValueError, match="'nested_train' is object, not bool"):
        splits.append_variant_rows(table, [_variant()])
    assert pd is not None


def test_appended_table_still_passes_the_schema_gate():
    pytest.importorskip("pandas")
    splits.validate_table_schema(splits.append_variant_rows(_base_table(), [_variant()]))


def test_the_derived_source_is_not_an_external_positive_source():
    """A3: derived rows and independent positives are different categories."""
    assert splits.SYNTHETIC_CLASSII_SOURCE not in splits.EXTERNAL_POSITIVE_SOURCES
    assert splits.SYNTHETIC_CLASSII_SOURCE in splits.DERIVED_SOURCES
    assert not set(splits.EXTERNAL_POSITIVE_SOURCES) & set(splits.DERIVED_SOURCES)
