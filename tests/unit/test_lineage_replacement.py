"""Unit tests for P0-15 TaxId → lineage re-placement (``tbox_finder.taxonomy``).

Two tiers (mirrors P0-13/P0-14): the parse / lineage-walk / vintage-reconcile logic is
pure stdlib and tested in the bare CI env; the end-to-end :func:`replace_lineage`
orchestrator (parquet I/O + a fixture taxdump zip) runs behind an ``importorskip("pandas")``
guard. The fixtures under ``tests/fixtures/ncbi_taxonomy/`` are a hand-built miniature
taxonomy exercising: a full lineage, a vintage synonym (Bacillota→Firmicutes), a merged
TaxId redirect, a CPR taxon with no formal order (residue), an environmental TaxId with no
rank at all (residue), and a genuinely-new recovered order.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tbox_finder import taxonomy as tx

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ncbi_taxonomy"


def _read(name: str) -> list[str]:
    return (FIXTURES / name).read_text().splitlines()


# --------------------------------------------------------------------------- pure tier


def test_module_imports_without_pandas():
    """taxonomy imports stdlib-only (pandas is lazy inside replace_lineage)."""
    import sys

    # importing the module must not have pulled pandas in
    assert "tbox_finder.taxonomy" in sys.modules
    src = (Path(tx.__file__)).read_text()
    assert "import pandas" not in src.split("def replace_lineage")[0]


def test_parse_nodes_names_merged():
    parent, rank = tx.parse_nodes(_read("nodes.dmp"))
    assert parent[1423] == 1386 and rank[1423] == "species"
    assert rank[1239] == "phylum" and rank[1385] == "order"
    merged = tx.parse_merged(_read("merged.dmp"))
    assert merged == {99999: 1423}
    names = tx.parse_names(_read("names.dmp"))
    assert names[1239]["scientific name"] == ["Bacillota"]
    assert names[1239]["synonym"] == ["Firmicutes"]


def test_parse_names_keep_filters():
    names = tx.parse_names(_read("names.dmp"), keep={1239})
    assert set(names) == {1239}


def test_lineage_by_rank_full_and_partial():
    parent, rank = tx.parse_nodes(_read("nodes.dmp"))
    merged = tx.parse_merged(_read("merged.dmp"))
    full = tx.lineage_by_rank(1423, parent, rank, merged)
    assert full == {
        "phylum": 1239,
        "class": 91061,
        "order": 1385,
        "family": 186817,
        "genus": 1386,
    }
    # CPR taxon: only a phylum, no class/order/family/genus
    assert tx.lineage_by_rank(2093855, parent, rank, merged) == {"phylum": 200918}
    # environmental TaxId: no target rank at all
    assert tx.lineage_by_rank(408170, parent, rank, merged) == {}


def test_lineage_by_rank_follows_merged_redirect():
    parent, rank = tx.parse_nodes(_read("nodes.dmp"))
    merged = tx.parse_merged(_read("merged.dmp"))
    # 99999 is retired → 1423; must resolve to the full Bacillus subtilis lineage
    assert tx.lineage_by_rank(99999, parent, rank, merged)["order"] == 1385


def test_reconcile_name_prefers_corpus_vintage_synonym():
    names = tx.parse_names(_read("names.dmp"))
    # corpus uses the pre-2021 name "Firmicutes" → reconcile modern Bacillota back to it
    assert tx.reconcile_name(1239, names, {"Firmicutes"}) == ("Firmicutes", "vocab-synonym")
    # no vintage match → fall back to the scientific name
    assert tx.reconcile_name(1239, names, set()) == ("Bacillota", "scientific")
    # CPR synonym reconciliation
    assert tx.reconcile_name(200918, names, {"Candidatus Atribacteria"})[0] == (
        "Candidatus Atribacteria"
    )


def test_resolve_row_fill_only_and_residue():
    parent, rank = tx.parse_nodes(_read("nodes.dmp"))
    merged = tx.parse_merged(_read("merged.dmp"))
    names = tx.parse_names(_read("names.dmp"))
    vocab = {"phylum": {"Firmicutes"}, "class": {"Bacilli"}, "order": {"Bacillales"}}
    lin = tx.lineage_by_rank(1423, parent, rank, merged)

    # present label is kept verbatim (fill-only), missing rank is recovered
    row = tx.resolve_row(
        {"phylum": "Firmicutes", "class": "Bacilli", "order": None, "family": None, "genus": None},
        lin,
        names,
        vocab,
    )
    assert row["phylum_source"] == tx.SOURCE_CORPUS and row["resolved_phylum"] == "Firmicutes"
    assert row["order_source"] == tx.SOURCE_RECOVERED and row["resolved_order"] == "Bacillales"

    # a taxon with no formal order → unresolved (residue), never a silent fold placement
    cpr = tx.lineage_by_rank(2093855, parent, rank, merged)
    row2 = tx.resolve_row(
        {"phylum": None, "class": None, "order": None, "family": None, "genus": None},
        cpr,
        names,
        {"phylum": {"Candidatus Atribacteria"}},
    )
    assert row2["resolved_phylum"] == "Candidatus Atribacteria"
    assert row2["resolved_order"] is None and row2["order_source"] == tx.SOURCE_UNRESOLVED


# ----------------------------------------------------------------- orchestrator tier


@pytest.fixture()
def taxdump_zip(tmp_path: Path) -> Path:
    """A fixture taxdump zip (nodes/names/merged) matching the pinned snapshot name."""
    zp = tmp_path / tx.TAXDUMP_SNAPSHOT
    with zipfile.ZipFile(zp, "w") as zf:
        for dmp in ("nodes.dmp", "names.dmp", "merged.dmp"):
            zf.write(FIXTURES / dmp, arcname=dmp)
    return zp


@pytest.fixture()
def corpus_and_ingested(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    # (phylum, class, order, family, genus, TaxId) — see fixture tree in test docstring
    rows = [
        ("Firmicutes", "Bacilli", "Bacillales", "Bacillaceae", "Bacillus", 1423),  # complete
        ("Firmicutes", "Bacilli", None, None, None, 1423),  # miss order → recovered
        (None, None, None, None, None, 1423),  # miss all → phylum via synonym
        ("Firmicutes", "Bacilli", None, None, None, 99999),  # merged TaxId → recovered
        ("Candidatus Atribacteria", None, None, None, None, 2093855),  # CPR, no order → residue
        (None, None, None, None, None, 2093855),  # CPR miss phylum → synonym; order residue
        (None, None, None, None, None, 408170),  # environmental → full residue
        ("Firmicutes", "Bacilli", None, None, None, 2606029),  # new order recovered
    ]
    cols = ["phylum", "class", "order", "family", "genus", "TaxId"]
    corpus = pd.DataFrame(rows, columns=cols)
    corpus_path = tmp_path / "master_clean_v0.parquet"
    corpus.to_parquet(corpus_path, index=False)
    ing = pd.DataFrame(
        {"record_sha256": [f"sha{i:03d}" for i in range(len(corpus))], "TaxId": corpus["TaxId"]}
    )
    ing_path = tmp_path / "master_tboxes_ingested.parquet"
    ing.to_parquet(ing_path, index=False)
    return corpus_path, ing_path


def test_replace_lineage_end_to_end(tmp_path, taxdump_zip, corpus_and_ingested):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    corpus_path, ing_path = corpus_and_ingested
    res = tx.replace_lineage(
        corpus_path,
        ing_path,
        taxdump_dir=taxdump_zip.parent,
        interim_dir=tmp_path / "interim",
        audit_dir=tmp_path / "audits",
        download=False,
    )
    out = pd.read_parquet(res.parquet)
    rep = res.report

    # accounting sums to N; per-rank recovered + still == pre_missing (fail-loud would raise)
    assert rep["accounting_ok"] is True
    assert rep["n_records"] == len(out) == 8
    assert rep["per_rank"]["order"] == {
        "present_from_corpus": 1,
        "pre_missing": 7,
        "recovered": 4,
        "still_incomplete": 3,
        "recovery_rate": round(4 / 7, 4),
    }
    assert rep["per_rank"]["phylum"]["recovered"] == 2
    assert rep["per_rank"]["class"]["recovered"] == 1

    # hash-link present, unique, no nulls
    assert out["record_sha256"].is_unique and out["record_sha256"].notna().all()

    # THE load-bearing invariant: every dropped record is unresolved at that holdout rank —
    # a no-clade record can never silently enter a clade fold (PRD §9.2 / CLAUDE.md §8.2)
    assert (out.loc[out["dropped_from_clade_holdout"], "resolved_order"].isna()).all()
    assert (
        out.loc[out["dropped_from_order_holdout"], "order_source"] == tx.SOURCE_UNRESOLVED
    ).all()
    assert int(out["dropped_from_clade_holdout"].sum()) == rep["dropped_from_clade_holdout"] == 3

    # fill-only: a present corpus label is kept exactly
    assert out.loc[0, "resolved_order"] == "Bacillales" and out.loc[0, "order_source"] == "corpus"
    # merged-TaxId record recovered its order
    assert out.loc[3, "resolved_order"] == "Bacillales" and out.loc[3, "order_source"] == (
        "taxid_recovered"
    )
    # vintage synonym: recovered phylum matches the corpus's pre-2021 name, not "Bacillota"
    assert out.loc[2, "resolved_phylum"] == "Firmicutes"
    assert out.loc[5, "resolved_phylum"] == "Candidatus Atribacteria"
    # environmental record: residue at every holdout rank
    assert bool(out.loc[6, "dropped_from_phylum_holdout"]) is True
    assert pd.isna(out.loc[6, "resolved_phylum"])

    # genuinely-new recovered order is surfaced in the audit (transparency)
    assert rep["new_taxa_introduced"]["order"] == {"Candidatus Izemoplasmatales": 1}

    # provenance artifacts written
    assert res.provenance.is_file() and res.taxdump_provenance.is_file()


def test_replace_lineage_rejects_misaligned_ingested(tmp_path, taxdump_zip):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    corpus = pd.DataFrame(
        {
            "phylum": ["Firmicutes"],
            "class": ["Bacilli"],
            "order": ["Bacillales"],
            "family": ["Bacillaceae"],
            "genus": ["Bacillus"],
            "TaxId": [1423],
        }
    )
    cpath = tmp_path / "c.parquet"
    corpus.to_parquet(cpath, index=False)
    # TaxId disagrees at row 0 → must fail-loud (cannot attach record_sha256)
    ing = pd.DataFrame({"record_sha256": ["x"], "TaxId": [999]})
    ipath = tmp_path / "i.parquet"
    ing.to_parquet(ipath, index=False)
    with pytest.raises(ValueError, match="row-misalignment"):
        tx.replace_lineage(
            cpath,
            ipath,
            taxdump_dir=taxdump_zip.parent,
            interim_dir=tmp_path / "o",
            audit_dir=tmp_path / "a",
            download=False,
        )
