"""Unit tests for the per-candidate homolog-search + CM-free de-novo MSA producer (P2-10e-msa).

Covers the pure logic and the tool-argv *contracts* (the tool chain itself is certified end-to-end
by ``slurm/p2/certify_homolog_msa.sbatch`` on the cluster, not here). Stdlib-only so it runs in the
bare-CI unit tier: the binary-shelling paths are exercised via monkeypatched seams, and the only
R-scape-touching call tested (``score_msa(None)``) returns before any binary. Asserts identity,
not counts (symmetric-count-fixture-blind-to-inversion), and checks controls fire under sabotage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder.mining import homolog_msa as hm
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED, STATUS_UNAVAILABLE
from tbox_finder.power import MIN_REAL_HOMOLOG_N

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_STO = REPO_ROOT / "tests" / "fixtures" / "rscape" / "classII_sub.sto"


# ── coordinate / strand mapping ──────────────────────────────────────────────
def test_blast_hit_to_range_plus_and_minus() -> None:
    assert hm.blast_hit_to_range({"sseqid": "acc:c0", "sstart": "100", "send": "342"}) == (
        "acc:c0",
        100,
        342,
        "plus",
    )
    # blastn reports send < sstart on the minus strand → lo/hi normalised, strand flipped
    assert hm.blast_hit_to_range({"sseqid": "acc:c1", "sstart": "900", "send": "700"}) == (
        "acc:c1",
        700,
        900,
        "minus",
    )


def test_nhmmer_hit_to_range_strand_symbols() -> None:
    assert hm.nhmmer_hit_to_range(
        {"target": "acc:c0", "alifrom": "10", "alito": "230", "strand": "+"}
    ) == ("acc:c0", 10, 230, "plus")
    assert hm.nhmmer_hit_to_range(
        {"target": "acc:c2", "alifrom": "500", "alito": "300", "strand": "-"}
    ) == ("acc:c2", 300, 500, "minus")


def test_blast_coverage() -> None:
    assert hm.blast_coverage({"length": "110"}, 220) == pytest.approx(0.5)
    assert hm.blast_coverage({"length": "110"}, 0) == 0.0


# ── homolog-set selection: filter + dedup-by-subject keeping lowest-E, order preserved ──────────
def test_select_blast_homologs_identity_dedup_and_filters() -> None:
    hits = [
        # subj A: two hits — the SECOND is lower-E, so it must be the kept representative
        {
            "sseqid": "A:c0",
            "pident": "88",
            "length": "200",
            "evalue": "1e-9",
            "sstart": "1",
            "send": "200",
        },
        {
            "sseqid": "A:c0",
            "pident": "90",
            "length": "210",
            "evalue": "1e-30",
            "sstart": "5",
            "send": "214",
        },
        # subj B: below the pident floor → dropped
        {
            "sseqid": "B:c0",
            "pident": "55",
            "length": "200",
            "evalue": "1e-8",
            "sstart": "1",
            "send": "200",
        },
        # subj C: below the coverage floor → dropped (len 40 / 220 = 0.18)
        {
            "sseqid": "C:c0",
            "pident": "95",
            "length": "40",
            "evalue": "1e-8",
            "sstart": "1",
            "send": "40",
        },
        # subj D: minus-strand keeper
        {
            "sseqid": "D:c1",
            "pident": "70",
            "length": "180",
            "evalue": "1e-6",
            "sstart": "900",
            "send": "721",
        },
    ]
    got = hm.select_blast_homologs(hits, query_len=220, min_pident=60.0, min_cov=0.5)
    # IDENTITY (not count): A kept at its lower-E coords, D minus-strand; B/C dropped; A before D.
    assert got == [("A:c0", 5, 214, "plus"), ("D:c1", 721, 900, "minus")]


def test_select_nhmmer_homologs_identity_and_coverage() -> None:
    hits = [
        {"target": "A:c0", "alifrom": "1", "alito": "220", "strand": "+", "evalue": "1e-20"},
        {
            "target": "A:c0",
            "alifrom": "3",
            "alito": "210",
            "strand": "+",
            "evalue": "1e-5",
        },  # higher E, dropped
        {
            "target": "B:c0",
            "alifrom": "1",
            "alito": "60",
            "strand": "-",
            "evalue": "1e-9",
        },  # cov 60/220 < 0.5
    ]
    got = hm.select_nhmmer_homologs(hits, query_len=220, min_cov=0.5)
    assert got == [("A:c0", 1, 220, "plus")]


# ── tabular parsers ──────────────────────────────────────────────────────────
def test_parse_blast_hits_and_reject_short_row() -> None:
    text = "q\tA:c0\t90.0\t200\t1e-30\t350\t5\t214\n# a comment\n\n"
    rows = hm.parse_blast_hits(text)
    assert rows == [
        {
            "qseqid": "q",
            "sseqid": "A:c0",
            "pident": "90.0",
            "length": "200",
            "evalue": "1e-30",
            "bitscore": "350",
            "sstart": "5",
            "send": "214",
        }
    ]
    with pytest.raises(hm.HomologMsaError):
        hm.parse_blast_hits("q\tA:c0\t90.0\n")  # too few fields


def test_parse_nhmmer_hits_positional_and_reject_short_row() -> None:
    # 16 whitespace fields; description trails and must not shift the fixed columns.
    line = "A:c0 - query - 1 220 10 230 8 232 5000 + 1e-20 88.5 0.1 some description here"
    rows = hm.parse_nhmmer_hits(line + "\n#comment\n")
    assert rows == [
        {
            "target": "A:c0",
            "alifrom": "10",
            "alito": "230",
            "strand": "+",
            "evalue": "1e-20",
            "score": "88.5",
        }
    ]
    with pytest.raises(hm.HomologMsaError):
        hm.parse_nhmmer_hits("A:c0 - query - 1 220\n")


# ── RNAfold structure parse + SS_cons validation ─────────────────────────────
def test_parse_rnafold_structure_extracts_dot_bracket() -> None:
    out = ">candidate\nGGGAAACCC\n(((...))) ( -1.20)\n"
    assert hm.parse_rnafold_structure(out, expected_len=9) == "(((...)))"


def test_parse_rnafold_structure_raises_when_no_length_match() -> None:
    out = ">candidate\nGGGAAACCC\n(((...))) ( -1.20)\n"
    with pytest.raises(hm.HomologMsaError):
        hm.parse_rnafold_structure(out, expected_len=42)


def test_assert_balanced_structure() -> None:
    hm.assert_balanced_structure("<<<...>>>")
    hm.assert_balanced_structure("((..)).<>")
    with pytest.raises(hm.HomologMsaError):
        hm.assert_balanced_structure("((..)")  # unclosed
    with pytest.raises(hm.HomologMsaError):
        hm.assert_balanced_structure(")(")  # close before open
    with pytest.raises(hm.HomologMsaError):
        hm.assert_balanced_structure("((.x.))")  # illegal char


def test_build_candidate_stockholm_shape_and_length_guard() -> None:
    sto = hm.build_candidate_stockholm("cand", "GGGAAACCC", "(((...)))")
    assert sto.count("# STOCKHOLM 1.0") == 1
    assert sto.rstrip().endswith("//")
    assert "#=GC SS_cons (((...)))" in sto
    assert "cand" in sto.split("\n")[2]  # the sequence row
    with pytest.raises(hm.HomologMsaError):
        hm.build_candidate_stockholm("cand", "GGGAAACCC", "(((...))")  # length mismatch


def test_homolog_record_name_is_unique_and_whitespace_free() -> None:
    a = hm.homolog_record_name("acc x:c0", 5, 214, "plus")
    b = hm.homolog_record_name("acc x:c0", 721, 900, "minus")
    assert " " not in a and " " not in b
    assert a != b
    assert a == "acc/5-214:p"  # first token only (whitespace stripped)


# ── sequence helpers ─────────────────────────────────────────────────────────
def test_degap_to_dna() -> None:
    assert hm.degap_to_dna("ac-gu.NN") == "ACGTNN"


def test_read_stockholm_sequence_from_committed_fixture() -> None:
    name, seq = hm.read_stockholm_sequence(SEED_STO, index=0)
    assert seq and set(seq) <= set("ACGTN")  # degapped DNA
    assert "-" not in seq and "." not in seq
    with pytest.raises(hm.HomologMsaError):
        hm.read_stockholm_sequence(SEED_STO, index=10_000)


def test_read_single_sequence(tmp_path: Path) -> None:
    fa = tmp_path / "c.fa"
    fa.write_text(">c desc\nACGT\nACGT\n", encoding="utf-8")
    # read_single_sequence returns the raw header (callers apply _safe_seq_name where a
    # whitespace-free id is needed); the sequence is concatenated across wrapped lines.
    assert hm.read_single_sequence(fa) == ("c desc", "ACGTACGT")


# ── argv contracts (tool_path stubbed to identity so no binary is required) ──────────────────────
@pytest.fixture()
def _stub_tool_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hm, "tool_path", lambda name: name)


def test_blastn_argv(_stub_tool_path: None) -> None:
    argv = hm.blastn_argv("q.fa", "db/target", evalue=1e-3, max_target_seqs=500)
    assert argv[0] == "blastn"
    assert argv[1:5] == ["-query", "q.fa", "-db", "db/target"]
    assert "-outfmt" in argv and "6 " + " ".join(hm.BLAST_OUTFMT_COLUMNS) in argv
    assert "-max_target_seqs" in argv and "500" in argv
    assert argv[argv.index("-evalue") + 1] == repr(1e-3)
    assert argv[-2:] == ["-dust", "no"]


def test_nhmmer_argv(_stub_tool_path: None) -> None:
    argv = hm.nhmmer_argv("q.fa", "db/target.fm", "out.tbl", evalue=1e-3)
    assert argv[0] == "nhmmer"
    assert "--tblout" in argv and "out.tbl" in argv
    assert argv[argv.index("-E") + 1] == repr(1e-3)
    assert argv[-2:] == ["q.fa", "db/target.fm"]


def test_parse_subject_id() -> None:
    # accessions carry a dot-version but no ":c"; the last :c<int> is the contig separator
    assert hm.parse_subject_id("GCA_000220375.1:c0") == ("GCA_000220375.1", 0)
    assert hm.parse_subject_id("GCF_964020195.1:c37") == ("GCF_964020195.1", 37)
    with pytest.raises(hm.HomologMsaError):
        hm.parse_subject_id("GCA_000220375.1")  # no contig suffix


def test_revcomp() -> None:
    assert hm._revcomp("ACGTN") == "NACGT"
    assert hm._revcomp("AATTCG") == "CGAATT"


def test_extract_homolog_sequence_from_genome_plus_and_minus(tmp_path: Path) -> None:
    # genome FASTA with two contigs; <accession>:c<ci> maps to record[ci] by iter order
    (tmp_path / "GCA_9.1.fna").write_text(
        ">contig_a\nAAAACCCCGGGGTTTT\n>contig_b\nACGTACGTACGT\n", encoding="utf-8"
    )
    # plus strand, contig 0, 1-based inclusive [5..8] → CCCC
    assert hm.extract_homolog_sequence(tmp_path, "GCA_9.1:c0", 5, 8, "plus") == "CCCC"
    # minus strand → reverse-complement of the plus subrange [1..4]=AAAA → TTTT
    assert hm.extract_homolog_sequence(tmp_path, "GCA_9.1:c0", 1, 4, "minus") == "TTTT"
    # contig 1
    assert hm.extract_homolog_sequence(tmp_path, "GCA_9.1:c1", 1, 4, "plus") == "ACGT"
    hm._genome_records.cache_clear()  # don't leak the tmp genome into other tests
    with pytest.raises(hm.HomologMsaError):
        hm.extract_homolog_sequence(tmp_path, "GCA_9.1:c9", 1, 4, "plus")  # contig index OOB
    hm._genome_records.cache_clear()
    with pytest.raises(hm.HomologMsaError):
        hm.extract_homolog_sequence(tmp_path, "GCA_9.1:c0", 5, 99, "plus")  # range OOB
    hm._genome_records.cache_clear()


def test_cmbuild_and_cmalign_argv(_stub_tool_path: None) -> None:
    assert hm.cmbuild_argv("c.cm", "c.sto", name="candidate") == [
        "cmbuild",
        "-F",
        "-n",
        "candidate",
        "c.cm",
        "c.sto",
    ]
    argv = hm.cmalign_argv("c.cm", "h.fa", "out.sto", cpu=8)
    assert argv[:3] == ["cmalign", "--outformat", "Pfam"]
    assert "--noprob" in argv
    assert argv[-3:] == ["out.sto", "c.cm", "h.fa"] or argv[-2:] == ["c.cm", "h.fa"]
    assert argv[argv.index("-o") + 1] == "out.sto"


def test_rnafold_argv(_stub_tool_path: None) -> None:
    assert hm.rnafold_argv() == ["RNAfold", "--noPS"]


def test_fmt_evalue_rejects_nonpositive_and_bool() -> None:
    assert hm._fmt_evalue(1e-3) == repr(1e-3)
    for bad in (0, -1.0, True):
        with pytest.raises(hm.HomologMsaError):
            hm._fmt_evalue(bad)


# ── search assembly (search + extraction stubbed) ────────────────────────────
def _candidate_fa(tmp_path: Path, seq: str = "ACGT" * 20) -> Path:
    fa = tmp_path / "seed.fa"
    fa.write_text(f">classII_seed\n{seq}\n", encoding="utf-8")
    return fa


def test_search_homologs_assembles_candidate_first_and_flags_sufficiency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fa = _candidate_fa(tmp_path)
    # two subjects returned by the (stubbed) search; both clear filters
    hits = [
        {
            "sseqid": "A:c0",
            "pident": "90",
            "length": "80",
            "evalue": "1e-9",
            "sstart": "1",
            "send": "80",
        },
        {
            "sseqid": "B:c0",
            "pident": "80",
            "length": "80",
            "evalue": "1e-8",
            "sstart": "1",
            "send": "80",
        },
    ]
    monkeypatch.setattr(hm, "run_blastn_search", lambda *a, **k: hits)
    monkeypatch.setattr(hm, "extract_homolog_sequence", lambda *a, **k: "ACGT" * 20)
    out_fa = tmp_path / "homologs.fa"
    report = hm.search_homologs(
        candidate_fasta=fa,
        out_fasta=out_fa,
        engine="blastn",
        evalue=1e-3,
        max_target_seqs=500,
        min_pident=60.0,
        min_cov=0.5,
        min_sequences=3,
    )
    records = hm.iter_fasta_records(out_fa.read_text(encoding="utf-8"))
    assert records[0][0] == "candidate"  # candidate is always first
    # <accession>:c<ci> subject ids carry no whitespace/pipe, so they survive _safe_seq_name intact
    assert [n for n, _ in records[1:]] == ["A:c0/1-80:p", "B:c0/1-80:p"]
    assert report["n_records"] == 3 and report["n_homologs"] == 2
    assert report["sufficient"] is True  # 3 >= min_sequences 3


def test_search_homologs_fail_closed_below_min_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fa = _candidate_fa(tmp_path)
    monkeypatch.setattr(
        hm,
        "run_blastn_search",
        lambda *a, **k: [
            {
                "sseqid": "A:c0",
                "pident": "90",
                "length": "80",
                "evalue": "1e-9",
                "sstart": "1",
                "send": "80",
            }
        ],
    )
    monkeypatch.setattr(hm, "extract_homolog_sequence", lambda *a, **k: "ACGT" * 20)
    report = hm.search_homologs(
        candidate_fasta=fa,
        out_fasta=tmp_path / "h.fa",
        engine="blastn",
        evalue=1e-3,
        max_target_seqs=500,
        min_pident=60.0,
        min_cov=0.5,
        min_sequences=MIN_REAL_HOMOLOG_N,  # 20 — only 2 records → not producible
    )
    assert report["sufficient"] is False
    assert report["min_sequences"] == MIN_REAL_HOMOLOG_N


def test_search_homologs_drops_non_acgt_extracted_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fa = _candidate_fa(tmp_path)
    monkeypatch.setattr(
        hm,
        "run_blastn_search",
        lambda *a, **k: [
            {
                "sseqid": "A:c0",
                "pident": "90",
                "length": "80",
                "evalue": "1e-9",
                "sstart": "1",
                "send": "80",
            },
            {
                "sseqid": "B:c0",
                "pident": "90",
                "length": "80",
                "evalue": "1e-8",
                "sstart": "1",
                "send": "80",
            },
        ],
    )
    # B extracts garbage (not ACGT/N) → excluded from the FASTA
    monkeypatch.setattr(
        hm,
        "extract_homolog_sequence",
        lambda prefix, sseqid, lo, hi, strand: "ACGT" * 20 if sseqid == "A:c0" else "XYZ!!",
    )
    report = hm.search_homologs(
        candidate_fasta=fa,
        out_fasta=tmp_path / "h.fa",
        engine="blastn",
        evalue=1e-3,
        max_target_seqs=500,
        min_pident=60.0,
        min_cov=0.5,
        min_sequences=2,
    )
    assert report["n_homologs"] == 1  # only A survived


# ── fail-closed scoring + the must-fire matched-control certification ────────────────────────────
def test_score_msa_none_is_unavailable_without_binaries() -> None:
    # covariation_spare_status(None) short-circuits before touching R-scape → safe in bare CI.
    assert hm.score_msa(None) == STATUS_UNAVAILABLE


def _tiny_pfam(rows: list[tuple[str, str]], ss_cons: str) -> str:
    from tbox_finder.mining.msa_shuffle import write_pfam_alignment

    return write_pfam_alignment(rows, {"SS_cons": ss_cons})


def _write_positive(tmp_path: Path) -> Path:
    # 4 rows x 6 cols with real per-column variation so the shuffle changes it.
    rows = [
        ("s1", "ACGUAC"),
        ("s2", "AGGUUC"),
        ("s3", "ACGCAC"),
        ("s4", "AUGUAG"),
    ]
    p = tmp_path / "positive.sto"
    p.write_text(_tiny_pfam(rows, "<<..>>"), encoding="utf-8")
    return p


def _score_by_name(mapping: dict[str, str]):
    def _fake(msa, *, min_sequences=MIN_REAL_HOMOLOG_N):  # noqa: ANN001
        return mapping[Path(msa).name]

    return _fake


def test_certify_msa_certified_when_positive_fires_and_shuffle_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pos = _write_positive(tmp_path)
    monkeypatch.setattr(
        hm,
        "score_msa",
        _score_by_name({"positive.sto": STATUS_PASSED, "shuffled.sto": STATUS_FAILED}),
    )
    report_path = tmp_path / "certify.json"
    report = hm.certify_msa(positive_sto=pos, report_path=report_path, shuffle_seed=7)
    assert report["certified"] is True
    assert report["positive_status"] == STATUS_PASSED
    assert report["shuffled_status"] == STATUS_FAILED
    assert report["matched_control"]["composition_matched"] is True
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk["certified"] is True


def test_certify_msa_raises_when_positive_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pos = _write_positive(tmp_path)
    monkeypatch.setattr(
        hm,
        "score_msa",
        _score_by_name({"positive.sto": STATUS_FAILED, "shuffled.sto": STATUS_FAILED}),
    )
    with pytest.raises(hm.HomologMsaError):
        hm.certify_msa(positive_sto=pos, report_path=tmp_path / "c.json", shuffle_seed=7)
    # the report is still written (records the failure), but certified is False
    assert json.loads((tmp_path / "c.json").read_text())["certified"] is False


def test_certify_msa_raises_when_shuffle_also_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A degenerate always-PASS backend must NOT certify (control-matchedness-must-be-asserted).
    pos = _write_positive(tmp_path)
    monkeypatch.setattr(
        hm,
        "score_msa",
        _score_by_name({"positive.sto": STATUS_PASSED, "shuffled.sto": STATUS_PASSED}),
    )
    with pytest.raises(hm.HomologMsaError):
        hm.certify_msa(positive_sto=pos, report_path=tmp_path / "c.json", shuffle_seed=7)


def test_certify_msa_raises_when_shuffle_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "can't tell" (R-scape timed out/errored on the matched shuffle) is NOT a demonstrated
    # negative: UNAVAILABLE conflates "no power" with "no signal" and must NOT certify — the
    # negative leg requires STATUS_FAILED, not merely != PASSED (matched-control lesson).
    pos = _write_positive(tmp_path)
    monkeypatch.setattr(
        hm,
        "score_msa",
        _score_by_name({"positive.sto": STATUS_PASSED, "shuffled.sto": STATUS_UNAVAILABLE}),
    )
    with pytest.raises(hm.HomologMsaError):
        hm.certify_msa(positive_sto=pos, report_path=tmp_path / "c.json", shuffle_seed=7)
    assert json.loads((tmp_path / "c.json").read_text())["certified"] is False


def test_assert_matched_control_rejects_mismatches(tmp_path: Path) -> None:
    pos = _write_positive(tmp_path)
    # a good matched shuffle passes
    from tbox_finder.mining.msa_shuffle import (
        read_pfam_alignment,
        shuffle_alignment_columns,
        write_pfam_alignment,
    )

    rows, gc = read_pfam_alignment(pos)
    good = tmp_path / "shuffled.sto"
    good.write_text(
        write_pfam_alignment(shuffle_alignment_columns(rows, seed=1), gc), encoding="utf-8"
    )
    summary = hm.assert_matched_control(pos, good)
    assert summary["composition_matched"] and summary["ss_cons_matched"]

    # a control with a DIFFERENT SS_cons is not matched → raises
    bad_ss = tmp_path / "bad_ss.sto"
    bad_ss.write_text(write_pfam_alignment(rows, {"SS_cons": "......"}), encoding="utf-8")
    with pytest.raises(hm.HomologMsaError):
        hm.assert_matched_control(pos, bad_ss)

    # a control with different composition (a base changed) is not matched → raises
    mutated = [(n, s.replace("A", "G", 1)) for n, s in rows]
    bad_comp = tmp_path / "bad_comp.sto"
    bad_comp.write_text(write_pfam_alignment(mutated, gc), encoding="utf-8")
    with pytest.raises(hm.HomologMsaError):
        hm.assert_matched_control(pos, bad_comp)
