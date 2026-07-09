"""Unit + integration tests for the P0-14 union-prior reconciliation (priors.py).

The pure NCBI→GTDB projection logic is stdlib-only (runs in the bare CI ``test`` env);
the parquet-join reconciliation is gated behind ``importorskip("pandas")``. Two
load-bearing properties are asserted per the P0-14 validation gate:
  * the conservative *unprojectable-but-known → has-prior* rule (§7.2:163), and
  * NCBI→GTDB renames/splits **cannot create false novelty** — a known-T-box lineage is
    never placed on the no-prior list through a renaming artifact.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tbox_finder import priors

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "priors"
BAC_FIXTURE = FIXTURE_DIR / "gtdb_taxonomy_sample.tsv"
AR_FIXTURE = FIXTURE_DIR / "ar53_taxonomy_sample.tsv"


def _fixture_universe() -> dict[str, set[str]]:
    with BAC_FIXTURE.open() as fh_b, AR_FIXTURE.open() as fh_a:
        return priors.load_clade_universe(list(fh_b) + list(fh_a))


# --- fixture presence (stdlib) --------------------------------------------------------
def test_fixtures_present() -> None:
    assert BAC_FIXTURE.is_file() and AR_FIXTURE.is_file()


# --- lineage parsing + universe (stdlib) ----------------------------------------------
def test_parse_gtdb_lineage() -> None:
    parsed = priors.parse_gtdb_lineage(
        "d__Bacteria;p__Bacillota;c__Bacilli;o__Bacillales;f__;g__;s__"
    )
    assert parsed["domain"] == "Bacteria"
    assert parsed["phylum"] == "Bacillota"
    assert parsed["class"] == "Bacilli"
    assert parsed["order"] == "Bacillales"
    # Empty ranks (f__/g__/s__) are omitted, not stored as "".
    assert "family" not in parsed and "species" not in parsed


def test_load_clade_universe() -> None:
    universe = _fixture_universe()
    assert {"Bacillota", "Bacillota_I", "Chloroflexota", "Desulfobacterota"} <= universe["phylum"]
    assert "Aquificota" in universe["phylum"]  # a no-prior phylum
    assert "Halobacteriota" in universe["phylum"]  # archaeal, from ar53
    assert "Clostridia" in universe["class"]


# --- projection (stdlib) --------------------------------------------------------------
def test_project_firmicutes_credits_all_bacillota() -> None:
    universe = _fixture_universe()["phylum"]
    result = priors.project_ncbi_phylum("Firmicutes", None, universe_phyla=universe)
    assert result.projectable and result.method == "name_projection"
    # Prefix match credits BOTH R232 Firmicutes-derived phyla (robust to GTDB suffixes).
    assert result.credited_phyla == frozenset({"Bacillota", "Bacillota_I"})


def test_project_proteobacteria_class_routed() -> None:
    universe = _fixture_universe()["phylum"]
    delta = priors.project_ncbi_phylum(
        "Proteobacteria", "Deltaproteobacteria", universe_phyla=universe
    )
    assert delta.method == "name_projection_class_routed"
    # PRD §7.2:163 delta targets; only those present in the release are credited.
    assert delta.credited_phyla == frozenset({"Desulfobacterota", "Myxococcota"})
    alpha = priors.project_ncbi_phylum(
        "Proteobacteria", "Alphaproteobacteria", universe_phyla=universe
    )
    assert alpha.credited_phyla == frozenset({"Pseudomonadota"})


def test_project_vitreschak_clades() -> None:
    universe = _fixture_universe()["phylum"]
    cases = {
        "Chloroflexi": "Chloroflexota",
        "Deinococcus-Thermus": "Deinococcota",
        "Dictyoglomi": "Dictyoglomota",
    }
    for ncbi, gtdb in cases.items():
        result = priors.project_ncbi_phylum(ncbi, None, universe_phyla=universe)
        assert result.projectable and result.credited_phyla == frozenset({gtdb}), ncbi


def test_project_cpr_to_patescibacteriota() -> None:
    universe = _fixture_universe()["phylum"]
    result = priors.project_ncbi_phylum(
        "Candidatus Saccharibacteria", None, universe_phyla=universe
    )
    assert result.credited_phyla == frozenset({"Patescibacteriota"})


def test_project_unprojectable_kinds() -> None:
    universe = _fixture_universe()["phylum"]
    euk = priors.project_ncbi_phylum("Arthropoda", None, universe_phyla=universe)
    assert not euk.projectable and euk.method == "eukaryotic_nonbacterial"
    null = priors.project_ncbi_phylum(None, None, universe_phyla=universe)
    assert not null.projectable and null.method == "null_lineage"
    unmapped = priors.project_ncbi_phylum(
        "candidate division KD3-62", None, universe_phyla=universe
    )
    assert not unmapped.projectable and unmapped.method == "unmapped_ncbi_phylum"


# --- no-false-novelty + conservative rule (stdlib) ------------------------------------
def test_renames_do_not_create_false_novelty() -> None:
    """Firmicutes/Proteobacteria renames put the GTDB targets on has-prior, not no-prior."""
    universe = _fixture_universe()["phylum"]
    credited = [
        priors.project_ncbi_phylum("Firmicutes", None, universe_phyla=universe).credited_phyla,
        priors.project_ncbi_phylum(
            "Proteobacteria", "Deltaproteobacteria", universe_phyla=universe
        ).credited_phyla,
        priors.project_ncbi_phylum(
            "Proteobacteria", "Alphaproteobacteria", universe_phyla=universe
        ).credited_phyla,
    ]
    has_prior, no_prior = priors.derive_prior_phyla(credited, frozenset(), universe)
    for gtdb in ("Bacillota", "Bacillota_I", "Desulfobacterota", "Myxococcota", "Pseudomonadota"):
        assert gtdb in has_prior, gtdb
        assert gtdb not in no_prior, gtdb


def test_assert_no_false_novelty_raises_when_known_missing() -> None:
    universe = _fixture_universe()["phylum"]
    # has-prior missing Chloroflexota (a known-T-box phylum) must fail loud (§10.3).
    with pytest.raises(ValueError, match="no-false-novelty gate FAILED"):
        priors.assert_no_false_novelty({"Bacillota"}, universe, known_bases=("Chloroflexota",))
    # …but passes when the known base is credited.
    priors.assert_no_false_novelty({"Chloroflexota"}, universe, known_bases=("Chloroflexota",))


def test_conservative_literature_credit_unprojectable_but_known() -> None:
    """Clade-level literature occurrences (no placeable genome) are still has-prior."""
    universe = _fixture_universe()["phylum"]
    lit = priors.credited_literature_phyla(universe)
    # The 3 confirmed clades are credited; withheld Dictyoglomota is NOT (single-source).
    assert lit == frozenset({"Desulfobacterota", "Deinococcota", "Chloroflexota"})
    # With no per-record genome credits, the literature clades alone make their phyla has-prior.
    has_prior, no_prior = priors.derive_prior_phyla([], lit, universe)
    for gtdb in ("Desulfobacterota", "Deinococcota", "Chloroflexota"):
        assert gtdb in has_prior and gtdb not in no_prior


def test_literature_artifact_reflects_evidence_gate() -> None:
    confirmed = [lc for lc in priors.LITERATURE_CLADES if not lc.withheld]
    withheld = [lc for lc in priors.LITERATURE_CLADES if lc.withheld]
    assert len(confirmed) == 3 and len(withheld) == 1
    by_clade = {lc.ncbi_clade: lc for lc in priors.LITERATURE_CLADES}
    assert by_clade["delta-proteobacteria"].gtdb_phylum == "Desulfobacterota"
    for lc in confirmed:
        assert lc.evidence_status == "confirmed_2plus_independent"
        assert len(lc.sources) >= 2  # the §10.1 ≥2-source bar
    assert withheld[0].ncbi_clade == "Dictyoglomi"
    assert withheld[0].evidence_status == "single_source"
    assert withheld[0].sources == (priors.VITRESCHAK_DOI,)


def test_parse_fasta_loci() -> None:
    lines = [
        ">CP001814.1:1988355-1988036 extra tokens\n",
        "ACGU\n",
        ">MNIY01000150.1:4366-4812\n",
        ">NOCOORDS.1\n",
    ]
    loci = priors.parse_fasta_loci(lines)
    assert loci[0] == ("CP001814", 1988355, 1988036)  # versionless accession + coords
    assert loci[1] == ("MNIY01000150", 4366, 4812)
    assert loci[2] == ("NOCOORDS", None, None)


# --- import purity (bare CI env) ------------------------------------------------------
def test_priors_imports_without_data_stack() -> None:
    """`import tbox_finder.priors` must not pull pandas/numpy (bare CI `test` env)."""
    code = (
        "import sys, tbox_finder.priors as p;"
        "assert 'pandas' not in sys.modules, 'pandas imported at module load';"
        "assert 'numpy' not in sys.modules, 'numpy imported at module load';"
        "assert p.VITRESCHAK_DOI"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# --- end-to-end reconciliation (pandas-gated) -----------------------------------------
def _write_corpus(pd, path: Path) -> None:
    rows = [
        # (phylum, class, order, family, genus, organism, type, source, acc, start, end, taxid)
        (
            "Firmicutes",
            "Bacilli",
            "Bacillales",
            "Bacillaceae",
            "Bacillus",
            "Bacillus subtilis",
            "Transcriptional",
            "classI.fa",
            "CP0001",
            100,
            200,
            1423,
        ),
        (
            "Actinobacteria",
            "Actinobacteria",
            "Mycobacteriales",
            "Mycobacteriaceae",
            "Mycobacterium",
            "Mycobacterium tuberculosis",
            "Translational",
            "actinobacteria_ILE.fa",
            "CP0002",
            300,
            400,
            1773,
        ),
        (
            "Chloroflexi",
            "Chloroflexia",
            "Chloroflexales",
            "Chloroflexaceae",
            "Chloroflexus",
            "Chloroflexus aurantiacus",
            "Transcriptional",
            "RF00230_master.fa",
            "CP0003",
            500,
            600,
            1108,
        ),
        (
            "Deinococcus-Thermus",
            "Deinococci",
            "Deinococcales",
            "Deinococcaceae",
            "Deinococcus",
            "Deinococcus radiodurans",
            "Transcriptional",
            "Vitreschak_master.fa",
            "CP0004",
            700,
            800,
            1299,
        ),
        (
            "Dictyoglomi",
            "Dictyoglomia",
            "Dictyoglomales",
            "Dictyoglomaceae",
            "Dictyoglomus",
            "Dictyoglomus thermophilum",
            "Transcriptional",
            "Marks_fulltbox.fa",
            "CP0005",
            900,
            1000,
            14,
        ),
        (
            "Proteobacteria",
            "Deltaproteobacteria",
            "Desulfuromonadales",
            "Geobacteraceae",
            "Geobacter",
            "Geobacter sulfurreducens",
            "Transcriptional",
            "Marks_fulltbox.fa",
            "CP0006",
            1100,
            1200,
            35554,
        ),
        (
            "Proteobacteria",
            "Alphaproteobacteria",
            "Rhodobacterales",
            "Rhodobacteraceae",
            "Sulfitobacter",
            "Sulfitobacter sp.",
            "Transcriptional",
            "gecont3.fa",
            "CP0007",
            1300,
            1400,
            60136,
        ),
        (
            "Synergistetes",
            "Synergistia",
            "Synergistales",
            "Synergistaceae",
            "Synergistes",
            "Synergistes sp.",
            "Transcriptional",
            "RIBEX_dedup.fa",
            "CP0008",
            1500,
            1600,
            1907,
        ),
        (
            "Arthropoda",
            "Insecta",
            "Diptera",
            "Drosophilidae",
            "Drosophila",
            "Drosophila melanogaster",
            "Transcriptional",
            "misannotation.fa",
            "CP0009",
            1700,
            1800,
            7227,
        ),
        (
            None,
            None,
            None,
            None,
            None,
            "unclassified bacterium",
            "Transcriptional",
            "RIBEX_dedup.fa",
            "CP0010",
            1900,
            2000,
            12345,
        ),
    ]
    cols = [
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "GBSeq_organism",
        "type",
        "source",
        "accession_name",
        "locus_start",
        "locus_end",
        "TaxId",
    ]
    pd.DataFrame(rows, columns=cols).to_parquet(path, index=False)


def test_reconcile_end_to_end(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    gtdb_dir = tmp_path / "gtdb"
    gtdb_dir.mkdir()
    (gtdb_dir / "bac120_taxonomy_r232.tsv").write_text(BAC_FIXTURE.read_text())
    (gtdb_dir / "ar53_taxonomy_r232.tsv").write_text(AR_FIXTURE.read_text())
    (gtdb_dir / "provenance.json").write_text("{}")

    corpus = tmp_path / "master_clean_v0.parquet"
    _write_corpus(pd, corpus)
    rf00230 = tmp_path / "RF00230_master.fa"
    rf00230.write_text(
        ">CP0001.1:100-200\nACGU\n"  # already in corpus -> not RF00230-only
        ">ZZ999.1:5-50\nACGU\n"  # residual -> rf00230_only_locus
        ">ZZ888.2:10-99\nACGU\n"  # residual -> rf00230_only_locus
    )

    out = priors.reconcile_union_prior(
        corpus_parquet=corpus,
        rf00230_fasta=rf00230,
        gtdb_dir=gtdb_dir,
        priors_dir=tmp_path / "priors",
        audit_dir=tmp_path / "audits",
        download=False,
    )

    prior_df = pd.read_parquet(out.union_prior)
    audit = json.loads(out.audit_report.read_text())

    # Accounting: 10 corpus + 2 RF00230-only + 3 literature = 15 records.
    assert len(prior_df) == 15
    assert set(prior_df["record_kind"]) == {"tbdb_locus", "rf00230_only_locus", "literature_clade"}
    assert audit["totals"] == {
        "union_prior_records": 15,
        "tbdb_locus": 10,
        "rf00230_only_locus": 2,
        "rf00230_only_accessions": 2,
        "literature_clade": 3,
    }

    # Every taxonomic record is projected OR audited-unprojectable, and they sum to total.
    proj = audit["projection"]
    assert proj["projectable_taxonomic"] + proj["unprojectable_taxonomic"] == 10
    assert proj["unprojectable_taxonomic"] == 2  # Arthropoda (eukaryotic) + null lineage
    assert proj["unprojectable_fraction_taxonomic"] == pytest.approx(0.2)

    # No-false-novelty: known-T-box GTDB phyla are has-prior, absent from no-prior.
    has_prior = set(audit["has_prior_phyla"])
    no_prior = set(audit["no_prior_phyla"])
    for gtdb in (
        "Bacillota",
        "Bacillota_I",
        "Actinomycetota",
        "Chloroflexota",
        "Deinococcota",
        "Dictyoglomota",
        "Desulfobacterota",
        "Myxococcota",
        "Pseudomonadota",
        "Synergistota",
    ):
        assert gtdb in has_prior and gtdb not in no_prior, gtdb
    # Genuinely-unrecorded phyla are on the no-prior (novel-eligible) list.
    for gtdb in ("Aquificota", "Acidobacteriota", "Patescibacteriota", "Halobacteriota"):
        assert gtdb in no_prior and gtdb not in has_prior, gtdb

    # Dictyoglomi: withheld from the curated set but has-prior via the corpus record.
    assert "Dictyoglomota" in has_prior
    withheld = audit["single_source_flagged_withheld"]
    assert len(withheld) == 1 and withheld[0]["gtdb_phylum"] == "Dictyoglomota"
    assert withheld[0]["has_prior_via_corpus"] is True
    assert {c["gtdb_phylum"] for c in audit["literature_occurrence_artifact"]} == {
        "Desulfobacterota",
        "Deinococcota",
        "Chloroflexota",
    }

    # Provenance carries the six §11 stamps.
    prov = json.loads(out.provenance.read_text())
    for field in ("rule", "script", "git_sha", "env_lock_hash", "seed", "inputs", "outputs"):
        assert field in prov
