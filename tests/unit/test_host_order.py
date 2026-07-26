"""Unit tests for the a2 whole-genome host-order negative-admission path (ADR-0004 A5; P2-10c′-a2).

Every guard is sabotage-verified: each test isolates one invariant so a regression fails the
test **named** for it (grep ``^FAILED``), per the standing test discipline. The a2 rule is a
leakage control — the headline generalization claim depends on it — so the identity assertions
here check *which* windows are refused, not merely *how many*
([[symmetric-count-fixture-blind-to-inversion]]).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from tbox_finder import splits
from tbox_finder.mining import host_order as ho
from tbox_finder.mining.substrate_windows import window_name

pd = pytest.importorskip("pandas")


# --------------------------------------------------------------------------- #
# host_accession — round-trips substrate_windows.window_name; fails loud otherwise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "acc,ci,start",
    [("GCA_000220375.1", 0, 0), ("GCF_964020195.1", 12, 512), ("GCA_900320995.1", 3, 1024)],
)
def test_host_accession_roundtrips_window_name(acc, ci, start):
    assert ho.host_accession(window_name(acc, ci, start)) == acc


@pytest.mark.parametrize(
    "bad",
    [
        "GCA_000220375.1",  # no geometry suffix at all
        "GCA_1:cX:5",  # non-digit contig index
        "GCA_1:c0:x",  # non-digit start
        ":c0:5",  # empty accession head
        "GCA_1:c0",  # missing the start field
    ],
)
def test_host_accession_rejects_malformed(bad):
    with pytest.raises(ho.HostOrderError):
        ho.host_accession(bad)


# --------------------------------------------------------------------------- #
# classify_host — the a2 rule; every branch + the pandas-NaN missing sentinel
# --------------------------------------------------------------------------- #
_HELD = ho.HeldoutUnits(
    orders=frozenset({"Frankiales", "Bifidobacteriales"}),
    phyla=frozenset({"Actinobacteria"}),
    classes=frozenset({"Actinobacteria", "Coriobacteriia"}),
)


def test_classify_admits_a_training_order():
    assert ho.classify_host("Lactobacillales", "Firmicutes", "Bacilli", _HELD) == (
        True,
        ho.ADMITTED,
    )


def test_classify_admits_an_order_with_no_corpus_positive():
    # An order absent from the corpus (not held out, not training) is independent-negative
    # territory the discovery set exists to supply — admitted.
    assert ho.classify_host("Aquificales", "Aquificae", "Aquificae", _HELD) == (True, ho.ADMITTED)


def test_classify_refuses_a_designated_heldout_order():
    assert ho.classify_host("Frankiales", "Actinobacteria", "Actinobacteria", _HELD) == (
        False,
        ho.REFUSED_HELDOUT_ORDER,
    )


def test_classify_refuses_heldout_phylum_when_order_not_designated():
    # An Actinobacteria order that is NOT one of the 30 LOO orders is still refused by the
    # phylum-holdout rule — the belt the ADR's three-rank language guarantees.
    assert ho.classify_host("Acidimicrobiales", "Actinobacteria", "Acidimicrobiia", _HELD) == (
        False,
        ho.REFUSED_HELDOUT_PHYLUM,
    )


def test_classify_fails_closed_on_unresolvable_order():
    assert ho.classify_host(None, "Proteobacteria", "Gammaproteobacteria", _HELD) == (
        False,
        ho.REFUSED_UNRESOLVABLE,
    )


def test_classify_treats_nan_order_as_unresolvable_not_the_string_nan():
    # A NaN order read back from parquet is truthy under pandas 3; it must fail closed, never
    # be admitted as the present-looking name "nan" ([[pandas-3-nan-truthy-in-training-env]]).
    assert ho.classify_host(float("nan"), "Proteobacteria", None, _HELD) == (
        False,
        ho.REFUSED_UNRESOLVABLE,
    )


def test_classify_refuses_a_designated_class_outside_a_held_phylum():
    held = ho.HeldoutUnits(orders=frozenset(), phyla=frozenset(), classes=frozenset({"Clostridia"}))
    assert ho.classify_host("Clostridiales", "Firmicutes", "Clostridia", held) == (
        False,
        ho.REFUSED_HELDOUT_CLASS,
    )


# --------------------------------------------------------------------------- #
# derive_heldout_units — the blacklist read off the committed split table
# --------------------------------------------------------------------------- #
def _split_fixture(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "split_assignments.parquet"
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)
    return path


_TRAIN_ROW = {
    "source": "corpus",
    "resolved_order": "Lactobacillales",
    "resolved_phylum": "Firmicutes",
    "resolved_class": "Bacilli",
    "nested_train": True,
    "is_designated_loo_holdout": False,
}
_HELD_ORDER_ROW = {
    "source": "corpus",
    "resolved_order": "Frankiales",
    "resolved_phylum": "Actinobacteria",
    "resolved_class": "Actinobacteria",
    "nested_train": False,
    "is_designated_loo_holdout": True,
}
_HELD_PHYLUM_ROW = {  # Actinobacteria order that is NOT designated LOO → held by phylum only
    "source": "corpus",
    "resolved_order": "Acidimicrobiales",
    "resolved_phylum": "Actinobacteria",
    "resolved_class": "Acidimicrobiia",
    "nested_train": False,
    "is_designated_loo_holdout": False,
}


def test_derive_heldout_units_reads_orders_phylum_classes(tmp_path):
    rows = [
        _TRAIN_ROW,
        _HELD_ORDER_ROW,
        dict(_HELD_ORDER_ROW, resolved_order="Bifidobacteriales", resolved_class="Actinobacteria"),
        _HELD_PHYLUM_ROW,
        # an external row whose order equals a designated held-out order must NOT count
        # (source != corpus): only corpus records designate the LOO holdout.
        {
            "source": "anchor",
            "resolved_order": "Frankiales",
            "resolved_phylum": "Actinobacteria",
            "resolved_class": "Actinobacteria",
            "nested_train": False,
            "is_designated_loo_holdout": False,
        },
    ]
    h = ho.derive_heldout_units(_split_fixture(tmp_path, rows))
    assert h.orders == frozenset({"Frankiales", "Bifidobacteriales"})
    assert h.phyla == frozenset({splits.HOLDOUT_PHYLUM})
    assert h.classes == frozenset({"Actinobacteria", "Acidimicrobiia"})


def test_derive_heldout_units_refuses_an_empty_holdout_set(tmp_path):
    # No is_designated_loo_holdout True → a blacklist that would admit every host. RED.
    rows = [_TRAIN_ROW, dict(_TRAIN_ROW, resolved_order="Bacillales")]
    with pytest.raises(ho.HostOrderError, match="ZERO designated held-out orders"):
        ho.derive_heldout_units(_split_fixture(tmp_path, rows))


def test_derive_heldout_units_fails_loud_on_holdout_phylum_drift(tmp_path):
    # A nested_train corpus record in the held-out phylum means splits.HOLDOUT_PHYLUM has
    # drifted from the committed partition — must not silently under-refuse.
    rows = [
        _HELD_ORDER_ROW,
        dict(_TRAIN_ROW, resolved_phylum="Actinobacteria", nested_train=True),
    ]
    with pytest.raises(ho.HostOrderError, match="drifted from the committed partition"):
        ho.derive_heldout_units(_split_fixture(tmp_path, rows))


def test_derive_heldout_units_rejects_an_lfs_pointer(tmp_path):
    ptr = tmp_path / "split_assignments.parquet"
    ptr.write_bytes(ho._LFS_POINTER_MAGIC + b"\noid sha256:deadbeef\nsize 123\n")
    with pytest.raises(ho.HostOrderError, match="Git-LFS pointer"):
        ho.derive_heldout_units(ptr)


# --------------------------------------------------------------------------- #
# load_host_taxids — stdlib gz stream; missing / non-integer taxid → None
# --------------------------------------------------------------------------- #
def _metadata_fixture(tmp_path: Path, name: str, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / name
    header = ["accession", "filler", "ncbi_taxid"]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for acc, taxid in rows:
            fh.write("\t".join([acc, "x", taxid]) + "\n")
    return path


def test_load_host_taxids_parses_integers_and_flags_missing(tmp_path):
    meta = _metadata_fixture(
        tmp_path,
        "meta.tsv.gz",
        [
            ("GB_GCA_1.1", "12345"),
            ("RS_GCF_2.1", "none"),  # non-integer → None (→ unresolvable → fail-closed)
            ("GB_GCA_3.1", ""),  # blank → None
            ("GB_GCA_4.1", "678"),  # present but NOT in wanted → excluded
        ],
    )
    got = ho.load_host_taxids([meta], ["GB_GCA_1.1", "RS_GCF_2.1", "GB_GCA_3.1"])
    assert got == {"GB_GCA_1.1": 12345, "RS_GCF_2.1": None, "GB_GCA_3.1": None}
    assert "GB_GCA_4.1" not in got  # not requested


def test_load_host_taxids_fails_loud_on_missing_column(tmp_path):
    path = tmp_path / "bad.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("accession\tfiller\n")  # no ncbi_taxid column
        fh.write("GB_GCA_1.1\tx\n")
    with pytest.raises(ho.HostOrderError, match="ncbi_taxid"):
        ho.load_host_taxids([path], ["GB_GCA_1.1"])


# --------------------------------------------------------------------------- #
# stamp_host_folds — admissibility is data on the window; asymmetric IDENTITY
# --------------------------------------------------------------------------- #
def _records_from(accessions: list[str]) -> list[dict]:
    # one window per host via its window_name id (no assembly_accession column) so the
    # host-accession recovery path is exercised, not just a direct column read.
    return [{"window_name": window_name(acc, 0, 0)} for acc in accessions]


def test_stamp_host_folds_refuses_exactly_the_heldout_hosts_by_identity():
    # ASYMMETRIC groups (3 refused, 5 admitted) so a fold/sense inversion cannot pass on a
    # symmetric count; assert the refused SET equals the held-out host set, not its size.
    heldout_hosts = ["GCA_h1.1", "GCA_h2.1", "GCA_h3.1"]
    training_hosts = ["GCA_t1.1", "GCA_t2.1", "GCA_t3.1", "GCA_t4.1", "GCA_t5.1"]
    admissible = {h: (False, ho.REFUSED_HELDOUT_ORDER) for h in heldout_hosts}
    admissible.update({t: (True, ho.ADMITTED) for t in training_hosts})
    records = _records_from(heldout_hosts + training_hosts)
    evidence = ho.stamp_host_folds(records, admissible)

    refused = {ho.host_accession(r["window_name"]) for r in records if not r[ho.HOST_FOLD_COL]}
    admitted = {ho.host_accession(r["window_name"]) for r in records if r[ho.HOST_FOLD_COL]}
    assert refused == set(heldout_hosts)  # identity, not count
    assert admitted == set(training_hosts)
    assert evidence["n_admissible"] == 5
    assert evidence["n_refused"] == 3
    assert evidence["n_host_unknown"] == 0


def test_stamp_host_folds_turns_red_on_a_broken_join():
    # A host absent from the admission map is a broken join, not a silent admit: it must be
    # refused refused_host_unknown and counted ([[namespace-mismatch-invisible-noop]]).
    records = _records_from(["GCA_known.1", "GCA_orphan.1"])
    admissible = {"GCA_known.1": (True, ho.ADMITTED)}
    evidence = ho.stamp_host_folds(records, admissible)
    orphan = next(r for r in records if ho.host_accession(r["window_name"]) == "GCA_orphan.1")
    assert orphan[ho.HOST_FOLD_COL] is False
    assert orphan[ho.HOST_REASON_COL] == ho.REFUSED_HOST_UNKNOWN
    assert evidence["n_host_unknown"] == 1


def test_stamp_host_folds_prefers_an_explicit_accession_column():
    rec = {"assembly_accession": "GCA_x.1", "window_name": window_name("GCA_WRONG.1", 0, 0)}
    ho.stamp_host_folds([rec], {"GCA_x.1": (True, ho.ADMITTED)})
    assert rec[ho.HOST_FOLD_COL] is True  # keyed on the column, not the (mismatched) id


# --------------------------------------------------------------------------- #
# Real committed-table tier — skips gracefully when the smudged table is absent
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[2]
_SPLIT = _REPO / splits.COMMITTED_TABLE
_HOST_TABLE = _REPO / ho.HOST_ORDER_TABLE


def _is_lfs_pointer(path: Path) -> bool:
    return (
        path.is_file() and path.open("rb").read(len(ho._LFS_POINTER_MAGIC)) == ho._LFS_POINTER_MAGIC
    )


@pytest.mark.skipif(
    not _SPLIT.exists() or _is_lfs_pointer(_SPLIT), reason="committed split table absent/unsmudged"
)
def test_committed_split_yields_the_30_designated_orders_and_actinobacteria():
    h = ho.derive_heldout_units(_SPLIT)
    assert len(h.orders) == 30
    assert h.phyla == frozenset({"Actinobacteria"})
    assert h.orders.isdisjoint({"Bacillales", "Clostridiales"})  # the 2 train-reserved orders


@pytest.mark.skipif(not _HOST_TABLE.exists(), reason="committed host-order table absent")
def test_committed_host_table_load_matches_its_report():
    import json

    verdicts, heldout = ho.load_host_folds(_HOST_TABLE, _SPLIT)
    assert len(verdicts) == 2500
    n_admissible = sum(1 for adm, _ in verdicts.values() if adm)
    report = json.loads((_REPO / ho.HOST_ORDER_REPORT).read_text())
    assert n_admissible == report["n_admissible_genomes"] == 660
    # a designated held-out order genuinely appears among the hosts (non-degeneracy).
    assert report["n_heldout_orders_present_among_hosts"] > 0
