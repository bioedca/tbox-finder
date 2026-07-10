"""P0-17 — additional independent non-Actinobacteria class-II positives.

The verified outcome is the EMPTY set (the peer-reviewed literature documents class-II
T-boxes only in Actinobacteria), so these tests lock the *withhold* contract rather than a
positive set:

* **Tier 1 (bare stdlib)** — the ≥2-source evidence chain, the Actinobacteria predicate,
  and the audit builder (0 positives + lead catalogue). Runs in the CI ``test`` env with
  no pandas.
* **Import purity** — ``import tbox_finder.anchors`` must not pull in pandas / Bio.
* **Tier 2 (pandas-gated)** — ``source_classII`` end-to-end against a tiny synthetic corpus
  parquet (offline): asserts 0 positives + empty FASTA, that Actinobacteria and
  transcriptional rows are excluded from the leads, that NO lead is ever emitted as a
  positive (the anti-circularity guard), and the six provenance fields.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tbox_finder import anchors

# --------------------------------------------------------------------------- #
# Tier 1 — pure logic + pinned evidence chain (no pandas / no network)
# --------------------------------------------------------------------------- #


def test_is_actinobacteria_matches_ncbi_and_gtdb_spellings():
    for name in ("Actinobacteria", "Actinomycetota", "Actinobacteriota", " actinobacteria "):
        assert anchors._is_actinobacteria(name) is True
    for name in ("Firmicutes", "Deinococcus-Thermus", "Chloroflexi", "", "(unresolved)"):
        assert anchors._is_actinobacteria(name) is False


def test_classII_sources_meet_two_source_high_stakes_bar():
    # HIGH-STAKES biological label (§10.1) → the negative rests on ≥2 independent sources.
    assert len(anchors.CLASSII_SOURCES) >= 2
    pmids = {s["pmid"] for s in anchors.CLASSII_SOURCES}
    # the load-bearing, mutually-independent peer-reviewed sources for the negative
    assert {"18359782", "19258532", "32882008", "25583497"} <= pmids
    for s in anchors.CLASSII_SOURCES:
        for field in ("pmid", "doi", "citation", "role"):
            assert s.get(field), f"source missing {field}: {s}"


def test_classII_determination_and_leads_disclaimer_are_honest():
    det = anchors.CLASSII_DETERMINATION.lower()
    assert "empty" in det and "actinobacteria" in det
    assert "10.2" in anchors.CLASSII_DETERMINATION or "10.3" in anchors.CLASSII_DETERMINATION
    dis = anchors.CLASSII_LEADS_DISCLAIMER.lower()
    assert "not " in dis and "lead" in dis and "circularity" in dis


def test_build_classII_audit_zero_positives_and_lead_catalogue():
    leads = [
        {"ncbi_phylum": "Deinococcus-Thermus", "organism": "A"},
        {"ncbi_phylum": "Deinococcus-Thermus", "organism": "B"},
        {"ncbi_phylum": "Chloroflexi", "organism": "C"},
        {"ncbi_phylum": "Firmicutes", "organism": "D"},
        {"ncbi_phylum": "(unresolved)", "organism": "E"},
    ]
    audit = anchors._build_classII_audit([], leads)
    assert audit["step"] == "P0-17"
    assert audit["raw_positive_count"] == 0
    assert audit["positives"] == []
    assert audit["counts_by_phylum"] == {}  # no positives → no positive phylum breakdown
    assert audit["leads"]["count"] == 5
    assert audit["leads"]["counts_by_ncbi_phylum"] == {
        "Deinococcus-Thermus": 2,
        "Chloroflexi": 1,
        "Firmicutes": 1,
        "(unresolved)": 1,
    }
    # highest-phylogenetic-spread leads flagged for P2 (closest ileS-paradigm parallels)
    assert audit["leads"]["highest_spread_priority_phyla"] == ["Chloroflexi", "Deinococcus-Thermus"]
    assert audit["leakage_report"]["n_positives"] == 0
    assert audit["leakage_report"]["n_added_to_no_leakage_test"] == 0
    assert len(audit["sources"]) >= 2


def test_classII_import_is_stdlib_only():
    """anchors (incl. the P0-17 additions) must bare-import with no pandas / Biopython."""
    code = (
        "import sys; import tbox_finder.anchors; "
        "assert 'pandas' not in sys.modules, 'pandas leaked at import'; "
        "assert 'Bio' not in sys.modules, 'Biopython leaked at import'; "
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# Tier 2 — source_classII end-to-end (offline; pandas-gated)
# --------------------------------------------------------------------------- #

_CORPUS_COLS = [
    "type",
    "phylum",
    "genus",
    "GBSeq_organism",
    "accession_name",
    "amino_acid_top",
    "refine_codon_top",
    "downstream_protein",
    "downstream_protein_EC",
    "Regulation",
    "term_sequence",
]


def _write_corpus(pd, path: Path) -> None:
    """A tiny corpus: 2 Actinobacteria class-II (excluded from leads), 3 non-Actino class-II
    (leads), 1 non-Actino *transcriptional* (excluded — not class II), 1 null-phylum class-II
    (lead, '(unresolved)'). Columns kept short so the row stays lint-clean."""
    pd.DataFrame(
        {
            "type": ["Translational"] * 5 + ["Transcriptional", "Translational"],
            "phylum": [
                "Actinobacteria",
                "Actinomycetota",
                "Deinococcus-Thermus",
                "Chloroflexi",
                "Firmicutes",
                "Deinococcus-Thermus",
                None,
            ],
            "genus": [
                "Streptomyces",
                "Mycobacterium",
                "Meiothermus",
                "Anaerolinea",
                "Paenibacillus",
                "Thermus",
                "unknown",
            ],
            "GBSeq_organism": [
                "Strep sp",
                "Myco sp",
                "Meiothermus ruber",
                "Anaerolinea thermophila",
                "Paeni sp",
                "Thermus thermophilus",
                "gut metagenome",
            ],
            "accession_name": [
                "CP000001",
                "CP000002",
                "CP005385",
                "AP012029",
                "CP000003",
                "CP000004",
                "MG000005",
            ],
            "amino_acid_top": ["ILE", "ILE", "VAL", None, "SER", "LEU", None],
            "refine_codon_top": ["AUC", "AUC", "GUC", None, "UCC", "CUC", None],
            "downstream_protein": ["ileS", "ileS", "valS", "leuS", "serS", "leuS", "hyp"],
            "downstream_protein_EC": [
                "6.1.1.5",
                "6.1.1.5",
                "6.1.1.9",
                None,
                "6.1.1.11",
                None,
                None,
            ],
            "Regulation": ["Unknown"] * 5 + ["Transcriptional", "Unknown"],
            "term_sequence": ["ACGT", "ACGT", "ACGT", None, "ACGT", "ACGT", None],
        }
    ).to_parquet(path)


def test_source_classII_withholds_positives_and_catalogues_leads(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    corpus = tmp_path / "corpus.parquet"
    _write_corpus(pd, corpus)

    out = anchors.source_classII(out_dir=tmp_path / "classII", corpus_parquet=corpus)
    audit = out.audit

    # verified empty positive set — the §10.2/§10.3 withhold
    assert audit["raw_positive_count"] == 0
    assert audit["positives"] == []
    fasta = (tmp_path / "classII" / anchors.CLASSII_FASTA).read_text()
    assert fasta.count(">") == 0
    assert fasta == ""  # 0 records → genuinely empty (git-LFS leaves empty files empty)

    # leads = the 4 NON-Actinobacteria class-II rows; Actinobacteria + transcriptional excluded
    assert audit["leads"]["count"] == 4
    assert audit["leads"]["counts_by_ncbi_phylum"] == {
        "Deinococcus-Thermus": 1,
        "Chloroflexi": 1,
        "Firmicutes": 1,
        "(unresolved)": 1,
    }
    assert audit["leads"]["highest_spread_priority_phyla"] == ["Chloroflexi", "Deinococcus-Thermus"]
    phyla = {r["ncbi_phylum"] for r in audit["leads"]["records"]}
    assert not any(anchors._is_actinobacteria(p) for p in phyla)  # no Actino leaked in
    # lead records carry the CM/DB-derived provenance (Regulation='Unknown', terminator annotated)
    dt = next(r for r in audit["leads"]["records"] if r["ncbi_phylum"] == "Deinococcus-Thermus")
    assert dt["tbdb_regulation"] == "Unknown"
    assert dt["tbdb_terminator_annotated"] is True
    assert dt["accession"] == "CP005385"

    # nothing enters the P0-24 no-leakage test
    assert audit["leakage_report"]["n_added_to_no_leakage_test"] == 0

    # provenance — the six §11 fields + the evidence chain
    prov = json.loads((tmp_path / "classII" / anchors.CLASSII_PROVENANCE).read_text())
    for field in ("rule", "script", "git_sha", "env_lock_hash", "seed", "inputs", "outputs"):
        assert field in prov
    assert prov["extra"]["raw_positive_count"] == 0
    assert len(prov["extra"]["sources"]) >= 2


def test_source_classII_never_emits_a_lead_as_positive(tmp_path):
    """Anti-circularity guard: no CM/DB-derived lead may become a labeled positive, no matter
    how many non-Actinobacteria class-II rows the corpus carries (§10.3)."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    # 20 non-Actinobacteria class-II rows — all leads, none a positive
    rows = [
        (
            "Translational",
            "Deinococcus-Thermus",
            f"Genus{i}",
            f"Organism {i}",
            f"CP{i:06d}",
            "VAL",
            "GUC",
            "valyl-tRNA synthetase",
            "6.1.1.9",
            "Unknown",
            "ACGT",
        )
        for i in range(20)
    ]
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame(rows, columns=_CORPUS_COLS).to_parquet(corpus)

    out = anchors.source_classII(out_dir=tmp_path / "classII", corpus_parquet=corpus)
    assert out.audit["raw_positive_count"] == 0
    assert out.audit["positives"] == []
    assert out.audit["leads"]["count"] == 20
    assert (tmp_path / "classII" / anchors.CLASSII_FASTA).read_text() == ""
