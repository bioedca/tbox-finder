"""P0-16 — independent non-Firmicutes GATE-1 anchor (arm c).

Two tiers mirroring ``test_priors``/``test_lineage_replacement``:

* **Tier 1 (bare stdlib)** — the pure re-derivation logic (Fig S1 parsing, segment
  localization, leader extraction, leakage classification). Runs in the CI ``test``
  env with no pandas / no Biopython.
* **Import purity** — ``import tbox_finder.anchors`` must not pull in pandas / Bio.
* **Tier 2 (pandas-gated)** — the ``source_anchor`` orchestrator end-to-end against a
  planted synthetic genome + injected Fig S1 text + a tiny corpus parquet (offline),
  asserting the FASTA, the six provenance fields, and the leakage classification.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tbox_finder import anchors

# --------------------------------------------------------------------------- #
# Tier 1 — pure logic (no pandas / no network)
# --------------------------------------------------------------------------- #


def test_parse_figs1_row_splits_on_elisions_and_strips_gaps():
    row = "TGAAGAGG--AG-(31)--CGACGCC-(116)-AAAAAGGG"
    segments, elisions = anchors.parse_figs1_row(row)
    assert segments == ["TGAAGAGGAG", "CGACGCC", "AAAAAGGG"]
    assert elisions == [31, 116]


def test_parse_figs1_row_folds_u_to_t_and_drops_noise():
    segments, elisions = anchors.parse_figs1_row("acgu-(5)-NNnACGT")
    assert segments == ["ACGT", "ACGT"]
    assert elisions == [5]


def test_segment_offsets_accumulate_lengths_plus_elisions():
    segs = ["A" * 20, "C" * 30, "G" * 40]
    assert anchors._segment_offsets(segs, [31, 116]) == [0, 51, 197]  # 20+31, +30+116


def test_revcomp():
    assert anchors.revcomp("ACGTN") == "NACGT"


def _plant(prefix: str, seg1: str, mid: str, seg2: str, suffix: str):
    leader = seg1 + mid + seg2
    return prefix + leader + suffix, leader, len(prefix)


def test_localize_leader_plus_strand_exact():
    seg1 = "TGAAGAGGAGTAGTAGTCCG"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGC"
    mid = "CGACGCCAGAGAGCCGGTGCTTGG"
    genome, leader, start = _plant("GATTACAGATTACAGATTACA", seg1, mid, seg2, "CCCGGGTTTAAA")
    hit = anchors.localize_leader(genome, [seg1, seg2], [len(mid)])
    assert hit is not None
    assert hit["strand"] == "+"
    assert hit["start"] == start
    assert hit["leader"] == leader
    assert hit["mismatches"] == 0
    assert hit["exact_offsets"] is True


def test_localize_leader_minus_strand_returns_forward_coords():
    seg1 = "TGAAGAGGAGTAGTAGTCCG"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGC"
    mid = "CGACGCCAGAGAGCCGGTGCTTGG"
    _, leader, _ = _plant("", seg1, mid, seg2, "")
    # ASYMMETRIC flanks (24 vs 6): the reverse-complement-space start (== len(suffix) == 6)
    # differs from the forward-genome start (== len(prefix) == 24), so this planting can
    # tell the two frames apart. A symmetric planting hides the bug (both equal len(flank)).
    prefix = "GATTACAGATTACAGATTACGGGT"  # 24
    suffix = "CCCGGG"  # 6
    genome = prefix + anchors.revcomp(leader) + suffix
    hit = anchors.localize_leader(genome, [seg1, seg2], [len(mid)])
    assert hit is not None
    assert hit["strand"] == "-"
    assert hit["leader"] == leader  # returned in sense orientation
    # coordinates are FORWARD-genome (not reverse-complement-space) for both strands, so
    # they line up with the forward-coordinate corpus in classify_leakage:
    assert hit["start"] == len(prefix)
    assert hit["end"] == len(prefix) + len(leader)
    assert hit["start"] < hit["end"]  # forward half-open — never start > end
    # the strong invariant tying the coords to the sense leader:
    assert anchors.revcomp(genome[hit["start"] : hit["end"]]) == leader


def test_localize_leader_tolerates_a_few_mismatches():
    seg1 = "TGAAGAGGAGTAGTAGTCCGACAT"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGCTT"
    mid = "CGACGCCAGAGAGCCGGTGCTTGG"
    genome, leader, start = _plant("GATTACAGATTACA", seg1, mid, seg2, "CCCGGG")
    # introduce one mismatch inside seg1 in the genome (approximate re-derivation)
    pos = start + 18
    genome = genome[:pos] + ("A" if genome[pos] != "A" else "C") + genome[pos + 1 :]
    hit = anchors.localize_leader(genome, [seg1, seg2], [len(mid)])
    assert hit is not None
    assert hit["mismatches"] == 1
    assert hit["exact_offsets"] is False
    assert hit["start"] == start


def test_localize_leader_returns_none_when_absent():
    seg1 = "TGAAGAGGAGTAGTAGTCCG"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGC"
    genome = "GATTACA" * 40  # neither segment present
    assert anchors.localize_leader(genome, [seg1, seg2], [30]) is None


def test_localize_requires_both_segments_at_consistent_offset():
    # seg1 present, seg2 present, but at the WRONG gap → must not localize
    seg1 = "TGAAGAGGAGTAGTAGTCCG"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGC"
    genome = "AA" + seg1 + "G" * 500 + seg2 + "TT"  # gap 500, elision says 24
    assert anchors.localize_leader(genome, [seg1, seg2], [24]) is None


def test_localize_leader_boundary_tracks_shifted_last_segment():
    # A +1 indel between the two usable segments: the second segment sits one base past its
    # nominal offset. The leader boundary must track the ACTUAL matched extremes (F2), so the
    # start comes from seg1's true position (not the anchor's nominal back-projection).
    seg1 = "TGAAGAGGAGTAGTAGTCCGACGT"  # 24
    seg2 = "AAAAAGGGTGGTACCGCGAGCGCTT"  # 25 (longer → the anchor)
    mid = "CGACGCCAGAGAGCCGGTGCTTGG"  # 24 nominal elision
    prefix = "GATTACAGAT"  # 10
    genome = prefix + seg1 + mid + "A" + seg2 + "CCCGGG"  # 'A' = the +1 insertion
    hit = anchors.localize_leader(genome, [seg1, seg2], [len(mid)])
    assert hit is not None
    assert hit["start"] == len(prefix)  # true seg1 start — a nominal frame would report +1
    assert hit["end"] == len(prefix) + len(seg1) + len(mid) + 1 + len(seg2)  # true seg2 end
    assert hit["exact_offsets"] is False  # a shifted offset is not exact
    assert genome[hit["start"] : hit["end"]] == hit["leader"]


def test_localize_leader_flags_ambiguous_when_two_locations():
    # two identical, well-separated copies of the leader → two equally-good (exact) genomic
    # locations; the tie must be surfaced as ambiguous, not resolved by iteration order (F3).
    seg1 = "TGAAGAGGAGTAGTAGTCCG"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGC"
    mid = "CGACGCCAGAGAGCCGGTGCTTGG"
    _, leader, _ = _plant("", seg1, mid, seg2, "")
    genome = "GATTACA" + leader + ("TTTTGGGGCCCCAAAA" * 5) + leader + "AAATTT"
    hit = anchors.localize_leader(genome, [seg1, seg2], [len(mid)])
    assert hit is not None
    assert hit["ambiguous"] is True
    assert hit["n_tied_locations"] == 2


def test_localize_leader_unambiguous_when_single_location():
    seg1 = "TGAAGAGGAGTAGTAGTCCG"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGC"
    mid = "CGACGCCAGAGAGCCGGTGCTTGG"
    genome, _, _ = _plant("GATTACAGATTACA", seg1, mid, seg2, "CCCGGG")
    hit = anchors.localize_leader(genome, [seg1, seg2], [len(mid)])
    assert hit is not None
    assert hit["ambiguous"] is False
    assert hit["n_tied_locations"] == 1


def test_classify_leakage_overlap_and_novel_and_minus_strand():
    accs = {"NC_TEST", "XX_TEST"}
    corpus = [
        {"accession": "XX_TEST", "start": 1000, "end": 1300, "organism": "x", "phylum": "p"},
        {"accession": "XX_TEST", "start": 9000, "end": 8700, "organism": "x", "phylum": "p"},
    ]
    # overlaps the first record
    r = anchors.classify_leakage(accs, 1100, 1400, corpus)
    assert r["coord_overlap"] is True
    # overlaps the minus-strand (start>end) record — must normalize
    r2 = anchors.classify_leakage(accs, 8800, 8900, corpus)
    assert r2["coord_overlap"] is True
    # no overlap
    r3 = anchors.classify_leakage(accs, 5000, 5100, corpus)
    assert r3["coord_overlap"] is False
    # accession not in the anchor's replicon set → ignored
    r4 = anchors.classify_leakage({"OTHER"}, 1100, 1400, corpus)
    assert r4["coord_overlap"] is False


def test_build_audit_uncoordinated_not_asserted_novel():
    # F1: a locus with a same-replicon corpus record that lacks coordinates has overlap that
    # can be neither confirmed nor ruled out — it must NOT be counted coordinate-novel.
    loci = [
        {
            "gtdb_phylum": "Deinococcota",
            "organism": "A",
            "match_quality": "exact",
            "corpus_coord_overlap": True,
            "corpus_overlap_records": [{"coord": "overlap"}],
        },
        {
            "gtdb_phylum": "Deinococcota",
            "organism": "B",
            "match_quality": "exact",
            "corpus_coord_overlap": False,
            "corpus_overlap_records": [{"coord": "no-coords"}],
        },
        {
            "gtdb_phylum": "Deinococcota",
            "organism": "C",
            "match_quality": "exact",
            "corpus_coord_overlap": False,
            "corpus_overlap_records": [],
        },
    ]
    lr = anchors._build_audit(loci, [], {}, {})["leakage_report"]
    assert lr["n_corpus_coord_overlap"] == 1
    assert lr["n_uncoordinated_same_replicon"] == 1  # B — same-replicon record, no coords
    assert lr["n_coordinate_novel"] == 1  # C — no same-replicon record at all
    assert lr["n_no_confirmed_coord_overlap"] == 2  # uncoordinated + novel (B not asserted novel)


def test_parse_figs1_selects_rows_and_ignores_prose():
    text = "\n".join(
        [
            "some prose header that is not a locus row",
            "GSU_LEUA    TGAAGAGGAG--TAGTAGTCCG-(31)-AAAAAGGGTGGTACCGCG",
            "DR_ILES     CGTGAGGCC--AGTAGCCTCGGACA-(91)-AAGTTGGGTGGTACCACG",
            "short X",
        ]
    )
    loci = anchors.parse_figs1(text)
    assert set(loci) == {"GSU_LEUA", "DR_ILES"}
    assert loci["GSU_LEUA"]["abbr"] == "GSU"
    assert loci["GSU_LEUA"]["gene"] == "LEUA"
    assert loci["DR_ILES"]["elisions"] == [91]


def test_hosts_are_all_confirmed_non_firmicutes_and_dictyoglomi_withheld():
    # every anchor host maps to a confirmed non-Firmicutes GTDB phylum
    assert {h.gtdb_phylum for h in anchors.HOSTS.values()} == {
        "Desulfobacterota",
        "Deinococcota",
        "Chloroflexota",
    }
    for h in anchors.HOSTS.values():
        assert len(h.sources) >= 2  # ≥2-source high-stakes bar (CLAUDE.md §10.1)
    # single-source Dictyoglomi is withheld, not an anchor host
    assert "DTH" in anchors.WITHHELD
    assert all(h.abbr != "DTH" for h in anchors.HOSTS.values())


def test_import_is_stdlib_only():
    """anchors must bare-import in the CI test env (no pandas / no Biopython)."""
    code = (
        "import sys; import tbox_finder.anchors; "
        "assert 'pandas' not in sys.modules, 'pandas leaked at import'; "
        "assert 'Bio' not in sys.modules, 'Biopython leaked at import'; "
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# Tier 2 — orchestrator end-to-end (offline; pandas-gated)
# --------------------------------------------------------------------------- #


def _write_corpus(pd, path: Path) -> None:
    # the planted leader sits at genome position 20..95 (prefix len 20); one corpus record
    # overlaps it (30..200), one is elsewhere on the same replicon (minus-strand encoded)
    pd.DataFrame(
        {
            "accession_name": ["XX_TEST", "XX_TEST"],
            "locus_start": [30, 9000],
            "locus_end": [200, 8700],
            "GBSeq_organism": ["Testella exampla STR", "Testella exampla STR"],
            "phylum": ["Deinococcus-Thermus", "Deinococcus-Thermus"],
            "genus": ["Testella", "Testella"],
        }
    ).to_parquet(path)


def test_source_anchor_end_to_end(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    # a single synthetic confirmed host, so no real genomes are fetched
    host = anchors.Host(
        "TST",
        "Testella exampla STR",
        "Deinococcus-Thermus",
        "Deinococcota",
        "GCF_TEST",
        "Complete Genome",
        ("DOI:10.1261/rna.819308", "DOI:10.1128/MMBR.00026-08"),
    )
    monkeypatch.setattr(anchors, "HOSTS", {"TST": host})

    seg1 = "TGAAGAGGAGTAGTAGTCCGAC"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGCTT"
    mid = "CGACGCCAGAGAGCCGGTGCTTGGCATG"
    genome, leader, start = _plant("GATTACAGATTACAGATTAC", seg1, mid, seg2, "CCCGGGTTTAAA")

    anchor_dir = tmp_path / "gate1_anchor"
    cache = anchor_dir / anchors.CACHE_SUBDIR
    cache.mkdir(parents=True)
    (cache / "GCF_TEST.acc.json").write_text(
        json.dumps({"refseq": ["NC_TEST.1"], "genbank": ["XX_TEST.1"]})
    )
    (cache / "NC_TEST.1.fasta").write_text(genome + "\n")

    corpus = tmp_path / "corpus.parquet"
    _write_corpus(pd, corpus)

    figs1_text = f"TST_LEUA    {seg1}-({len(mid)})-{seg2}"
    out = anchors.source_anchor(
        anchor_dir=anchor_dir,
        corpus_parquet=corpus,
        download=False,
        figs1_text=figs1_text,
    )

    # FASTA carries the re-derived sense leader with the elided middle filled from genome
    fasta = (anchor_dir / anchors.ANCHOR_FASTA).read_text()
    assert fasta.count(">") == 1
    seq = "".join(ln for ln in fasta.splitlines() if not ln.startswith(">"))
    assert seq == leader
    assert "TST_LEUA" in fasta and "Deinococcota" in fasta and "Vitreschak2008" in fasta

    audit = out.audit
    assert audit["raw_record_count"] == 1
    assert audit["counts_by_gtdb_phylum"] == {"Deinococcota": 1}
    assert audit["targeting"]["re_derived"] == 1
    # planted overlap record 30-200 straddles the leader at 20-95 → coord overlap
    assert audit["leakage_report"]["n_corpus_coord_overlap"] == 1
    assert "DTH" in audit["withheld"]
    assert start >= 0  # planted offset sanity

    # provenance carries the six required §11 fields + the source/independence stamps
    prov = json.loads((anchor_dir / anchors.ANCHOR_PROVENANCE).read_text())
    for field in ("rule", "script", "git_sha", "env_lock_hash", "seed", "inputs", "outputs"):
        assert field in prov
    assert prov["extra"]["source"]["doi"] == anchors.SOURCE_DOI
    assert "RF00230" in prov["extra"]["independence"]


def test_source_anchor_minus_strand_leakage_caught(tmp_path, monkeypatch):
    """A minus-strand corpus-present locus must be flagged coord-overlap end-to-end.

    Regression for the reverse-complement-vs-forward coordinate-frame bug: minus-strand
    anchor loci were compared against the (forward-coordinate) corpus in reverse-complement
    space, so a locus already in the training corpus was mis-scored 'coordinate-novel' and
    would have escaped the P0-24 holdout (a §8.2 leakage miss). Here the corpus row overlaps
    only the FORWARD leader interval (beyond the reverse-complement-space interval), so it is
    caught after the fix but was missed before it.
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    host = anchors.Host(
        "TST",
        "Testella exampla STR",
        "Deinococcus-Thermus",
        "Deinococcota",
        "GCF_TEST",
        "Complete Genome",
        ("DOI:10.1261/rna.819308", "DOI:10.1128/MMBR.00026-08"),
    )
    monkeypatch.setattr(anchors, "HOSTS", {"TST": host})

    seg1 = "TGAAGAGGAGTAGTAGTCCGAC"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGCTT"
    mid = "CGACGCCAGAGAGCCGGTGCTTGGCATG"
    _, leader, _ = _plant("", seg1, mid, seg2, "")
    prefix = "GATTACAGATTACAGATTACGGGTACGTGA"  # 30 (asymmetric vs the 6-nt suffix)
    suffix = "CCCGGG"  # revcomp-palindrome, len 6
    genome = prefix + anchors.revcomp(leader) + suffix
    f_start, f_end = len(prefix), len(prefix) + len(leader)  # forward leader interval
    # Corpus row overlapping ONLY the far end of the forward interval — past the reverse-
    # complement-space interval [len(suffix), len(suffix)+len(leader)) — so it overlaps the
    # forward frame but NOT the buggy reverse frame. Encoded minus-strand (start > end) to
    # also exercise corpus-side normalization.
    row_hi, row_lo = f_end + 5, f_end - 15

    anchor_dir = tmp_path / "gate1_anchor"
    cache = anchor_dir / anchors.CACHE_SUBDIR
    cache.mkdir(parents=True)
    (cache / "GCF_TEST.acc.json").write_text(
        json.dumps({"refseq": ["NC_TEST.1"], "genbank": ["XX_TEST.1"]})
    )
    (cache / "NC_TEST.1.fasta").write_text(genome + "\n")

    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame(
        {
            "accession_name": ["XX_TEST"],
            "locus_start": [row_hi],
            "locus_end": [row_lo],
            "GBSeq_organism": ["Testella exampla STR"],
            "phylum": ["Deinococcus-Thermus"],
            "genus": ["Testella"],
        }
    ).to_parquet(corpus)

    figs1_text = f"TST_LEUA    {seg1}-({len(mid)})-{seg2}"
    out = anchors.source_anchor(
        anchor_dir=anchor_dir,
        corpus_parquet=corpus,
        download=False,
        figs1_text=figs1_text,
    )
    audit = out.audit
    locus = audit["loci"][0]
    assert locus["strand"] == "-"
    # forward coordinates recorded (revcomp invariant); never start > end
    assert locus["start_0based"] == f_start
    assert locus["end_0based_excl"] == f_end
    assert anchors.revcomp(genome[locus["start_0based"] : locus["end_0based_excl"]]) == leader
    # the corpus-present minus-strand locus is caught as overlap (missed before the fix)
    assert locus["corpus_coord_overlap"] is True
    assert audit["leakage_report"]["n_corpus_coord_overlap"] == 1


def test_source_anchor_raises_when_no_loci_localize(tmp_path, monkeypatch):
    """F4: publishing empty FASTA/report/provenance is a silent failure — fail loud instead."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    host = anchors.Host(
        "TST",
        "Testella exampla STR",
        "Deinococcus-Thermus",
        "Deinococcota",
        "GCF_TEST",
        "Complete Genome",
        ("DOI:10.1261/rna.819308", "DOI:10.1128/MMBR.00026-08"),
    )
    monkeypatch.setattr(anchors, "HOSTS", {"TST": host})

    cache = tmp_path / "gate1_anchor" / anchors.CACHE_SUBDIR
    cache.mkdir(parents=True)
    (cache / "GCF_TEST.acc.json").write_text(
        json.dumps({"refseq": ["NC_TEST.1"], "genbank": ["XX_TEST.1"]})
    )
    (cache / "NC_TEST.1.fasta").write_text("ACGT" * 300 + "\n")  # no anchor segments present

    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame(
        {
            "accession_name": ["XX_TEST"],
            "locus_start": [1],
            "locus_end": [2],
            "GBSeq_organism": ["Testella exampla STR"],
            "phylum": ["Deinococcus-Thermus"],
            "genus": ["Testella"],
        }
    ).to_parquet(corpus)

    seg1 = "TGAAGAGGAGTAGTAGTCCGAC"
    seg2 = "AAAAAGGGTGGTACCGCGAGCGCTT"
    figs1_text = f"TST_LEUA    {seg1}-(24)-{seg2}"
    with pytest.raises(RuntimeError, match="refusing to publish empty"):
        anchors.source_anchor(
            anchor_dir=tmp_path / "gate1_anchor",
            corpus_parquet=corpus,
            download=False,
            figs1_text=figs1_text,
        )


def test_offline_missing_genome_cache_fails_loud(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    host = anchors.HOSTS["GSU"]
    monkeypatch.setattr(anchors, "HOSTS", {"GSU": host})
    corpus = tmp_path / "c.parquet"
    pytest.importorskip("pandas").DataFrame(
        {
            "accession_name": ["X"],
            "locus_start": [1],
            "locus_end": [2],
            "GBSeq_organism": ["y"],
            "phylum": ["p"],
            "genus": ["g"],
        }
    ).to_parquet(corpus)
    with pytest.raises(FileNotFoundError):
        anchors.source_anchor(
            anchor_dir=tmp_path / "a",
            corpus_parquet=corpus,
            download=False,
            figs1_text="GSU_LEUA    TGAAGAGGAGTAGTAGTCCGAC-(30)-AAAAAGGGTGGTACCGCGAGCGC",
        )
