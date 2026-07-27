"""Unit tests for the per-candidate homolog-search + CM-free de-novo MSA producer (P2-10e-msa).

Covers the pure logic and the tool-argv *contracts* (the tool chain itself is certified end-to-end
by ``slurm/p2/certify_homolog_msa.sbatch`` on the cluster, not here). Stdlib-only so it runs in the
bare-CI unit tier: the binary-shelling paths are exercised via monkeypatched seams, and the only
R-scape-touching call tested (``score_msa(None)``) returns before any binary. Asserts identity,
not counts (symmetric-count-fixture-blind-to-inversion), and checks controls fire under sabotage.
"""

from __future__ import annotations

import json
import re
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


# ── mlocarna argv + version guard + align orchestration (ADR-0006 A3 / ADR-0002 A13) ─────────────
def test_mlocarna_argv(_stub_tool_path: None) -> None:
    argv = hm.mlocarna_argv("in.fa", "work/tgt", threads=2)
    assert argv[0] == "mlocarna"
    assert "--stockholm" in argv
    # --consensus-structure alifold is load-bearing: the default `none` writes NO #=GC SS_cons line
    assert argv[argv.index("--consensus-structure") + 1] == "alifold"
    assert argv[argv.index("--threads") + 1] == "2"
    assert argv[argv.index("--tgtdir") + 1] == "work/tgt"
    assert argv[-1] == "in.fa"  # the positional multi-FASTA trails the options (Getopt::Long)


def test_assert_mlocarna_version_accepts_pin_and_rejects_lookalike() -> None:
    # banner is injectable so the match logic is unit-testable without the binary
    assert hm.assert_mlocarna_version(banner="LocARNA 2.0.1\n") == "LocARNA 2.0.1"
    # a look-alike bump (2.0.10) must NOT satisfy the 2.0.1 pin (the \b guard), nor a wrong tool/ver
    for bad in ("LocARNA 2.0.10", "LocARNA 1.9.0", "locarna 2.0.1", "", "mlocarna 2.0.1"):
        with pytest.raises(hm.HomologMsaError):
            hm.assert_mlocarna_version(banner=bad)
    assert hm.PINNED_LOCARNA_VERSION == "2.0.1"  # the pin envs/locarna.yml + the sbatch agree on


def test_pinned_locarna_version_matches_lockfile() -> None:
    """Tie the code pin to the solved env: PINNED_LOCARNA_VERSION must equal the `locarna`
    version in envs/locarna.conda-lock.yml (bumping one without the other fails here)."""
    lock = (REPO_ROOT / "envs" / "locarna.conda-lock.yml").read_text(encoding="utf-8")
    assert re.search(
        rf"^- name: locarna\n\s+version: '?{re.escape(hm.PINNED_LOCARNA_VERSION)}'?\s*$",
        lock,
        re.MULTILINE,
    ), "PINNED_LOCARNA_VERSION must match the locarna version in envs/locarna.conda-lock.yml"


def test_assert_candidate_in_homologs_present_and_absent(tmp_path: Path) -> None:
    cand = tmp_path / "cand.fa"
    cand.write_text(">c\nACGTACGTACGT\n", encoding="utf-8")
    # search_homologs writes the candidate as record 0; degap+U→T normalisation must match it
    ok = tmp_path / "ok.fa"
    ok.write_text(">candidate\nACGU-ACGTACGT\n>h1\nACGTACGTAAAA\n", encoding="utf-8")
    hm._assert_candidate_in_homologs(cand, ok)  # present after degap → no raise
    bad = tmp_path / "bad.fa"
    bad.write_text(">h1\nTTTTGGGGCCCC\n>h2\nAAAACCCCGGGG\n", encoding="utf-8")
    with pytest.raises(hm.HomologMsaError):
        hm._assert_candidate_in_homologs(cand, bad)  # candidate absent → mismatched search/align


def _stub_align_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cand = tmp_path / "cand.fa"
    cand.write_text(">c\nACGTACGTACGT\n", encoding="utf-8")
    homologs = tmp_path / "homologs.fa"
    homologs.write_text(">candidate\nACGTACGTACGT\n>h1\nACGTACGTAAAA\n", encoding="utf-8")
    return cand, homologs


def test_align_candidate_copies_result_stk_and_runs_ss_cons_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cand, homologs = _stub_align_inputs(tmp_path)
    monkeypatch.setattr(hm, "assert_mlocarna_version", lambda *a, **k: "LocARNA 2.0.1")
    monkeypatch.setattr(hm, "tool_path", lambda name: name)  # no mlocarna binary in bare CI

    # stub `_run(...)` to write the <tgtdir>/results/result.stk that mlocarna would produce
    def _fake_run(argv, *a, **k):  # noqa: ANN001, ANN002, ANN003
        tgt = Path(argv[argv.index("--tgtdir") + 1])
        (tgt / "results").mkdir(parents=True, exist_ok=True)
        (tgt / "results" / "result.stk").write_text(
            "# STOCKHOLM 1.0\n\ncandidate   ACGUACGUACGU\nh1          ACGUACGUAAAA\n"
            "#=GC SS_cons <<<....>>>.\n//\n",
            encoding="utf-8",
        )
        return None

    monkeypatch.setattr(hm, "_run", _fake_run)
    out = tmp_path / "out.sto"
    result = hm.align_candidate(
        candidate_fasta=cand, homologs_fasta=homologs, out_sto=out, work_dir=tmp_path / "w", cpu=2
    )
    assert result == out
    text = out.read_text(encoding="utf-8")
    assert "# STOCKHOLM 1.0" in text and "#=GC SS_cons" in text  # copied verbatim + guard passed


def test_align_candidate_raises_when_mlocarna_writes_no_stockholm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cand, homologs = _stub_align_inputs(tmp_path)
    monkeypatch.setattr(hm, "assert_mlocarna_version", lambda *a, **k: "LocARNA 2.0.1")
    monkeypatch.setattr(hm, "tool_path", lambda name: name)  # no mlocarna binary in bare CI
    monkeypatch.setattr(hm, "_run", lambda *a, **k: None)  # produces no result.stk
    with pytest.raises(hm.HomologMsaError):
        hm.align_candidate(
            candidate_fasta=cand,
            homologs_fasta=homologs,
            out_sto=tmp_path / "out.sto",
            work_dir=tmp_path / "w",
            cpu=2,
        )


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
    # sensitive gapped task, not megablast (A13; job-748 megablast found 0 homologs)
    assert argv[argv.index("-task") + 1] == "blastn"
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


# --------------------------------------------------------------------------- #
# resolve_candidate_sequence — a MiningCandidate's OWN nucleotides (0-based half-open)
# --------------------------------------------------------------------------- #
def test_resolve_candidate_sequence_half_open_plus_and_minus(tmp_path: Path):
    # contig c0 = 8 nt "AACCGGTT"; c1 = "ACGT".
    (tmp_path / "GCA_9.1.fna").write_text(">c0\nAACCGGTT\n>c1\nACGT\n", encoding="utf-8")
    # 0-based half-open [2, 6) of c0 → "CCGG" (deliberately NOT the 1-based-inclusive convention
    # extract_homolog_sequence uses — the candidate coords come from window_start + cand.start).
    assert hm.resolve_candidate_sequence(tmp_path, "GCA_9.1:c0", 2, 6) == "CCGG"
    assert hm.resolve_candidate_sequence(tmp_path, "GCA_9.1:c1", 0, 4) == "ACGT"
    # minus strand reverse-complements the same span.
    assert hm.resolve_candidate_sequence(tmp_path, "GCA_9.1:c1", 0, 4, strand="minus") == "ACGT"[
        ::-1
    ].translate(str.maketrans("ACGT", "TGCA"))


def test_resolve_candidate_sequence_rejects_out_of_bounds(tmp_path: Path):
    (tmp_path / "GCA_9.2.fna").write_text(">c0\nAACCGGTT\n", encoding="utf-8")
    with pytest.raises(hm.HomologMsaError):
        hm.resolve_candidate_sequence(tmp_path, "GCA_9.2:c0", 4, 99)  # end past contig
    with pytest.raises(hm.HomologMsaError):
        hm.resolve_candidate_sequence(tmp_path, "GCA_9.2:c9", 0, 4)  # contig index OOB
    with pytest.raises(hm.HomologMsaError):
        hm.resolve_candidate_sequence(tmp_path, "GCA_9.2:c0", 5, 5)  # empty half-open span
