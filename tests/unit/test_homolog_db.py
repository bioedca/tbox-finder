"""Unit tests for the homolog-search target-DB builder (ADR-0006 D7 + A1, P2-10c'-homologdb).

The DB build runs external binaries (``makeblastdb`` / ``blastdbcmd`` / ``makehmmerdb`` / ``blastn``
/ ``nhmmer``) that no CI/local env carries, so those calls are proven by an **in-env end-to-end run
on the cluster** (a 3-genome toy through the real binaries: both probes self-recover, ``blastdbcmd
-info`` cross-checks, versions assert). What CI locks here is the **binary-free logic** the build's
correctness hinges on: the two-independent-reads integrity guard, the tabular-output parsers, the
version asserts, and — load-bearing — the **must-fire round-trip control** predicate.

Every guard has a MUST-FIRE partner: a *dropped/truncated genome* makes integrity raise (not merely
a healthy path passes), and a self-recovery hit at the **wrong genome** (the count-preserving
inversion, [[symmetric-count-fixture-blind-to-inversion]]) is refused by identity, not counted as
success. Bare-CI tier: stdlib only — the module imports without pandas/numpy/torch.
"""

from __future__ import annotations

import json

import pytest

from tbox_finder.mining import homolog_db as hdb


# ── fixtures ─────────────────────────────────────────────────────────────────
def _genome_fasta(contigs: list[str]) -> str:
    """A multi-contig FASTA; an empty-string contig emits an empty record (to test skip logic)."""
    return "".join(f">ctg_{i} desc with spaces {i}\n{seq}\n" for i, seq in enumerate(contigs))


def _write_genome(root, acc: str, contigs: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{acc}.fna").write_text(_genome_fasta(contigs))


def _seq(n: int, salt: int) -> str:
    import random

    rng = random.Random(salt)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _baseline_dict(per_genome: list[dict]) -> dict:
    return {
        "schema_version": "1.0",
        "step": "toy",
        "n_genomes": len(per_genome),
        "total_bp": sum(int(r["total_bp"]) for r in per_genome),
        "per_genome": per_genome,
    }


def _row(acc: str, n_contigs: int, total_bp: int) -> dict:
    return {"assembly_accession": acc, "n_contigs": n_contigs, "total_bp": total_bp, "n_windows": 0}


# ── target-sequence naming ───────────────────────────────────────────────────
def test_target_seq_name() -> None:
    assert hdb.target_seq_name("GCA_000220375.1", 0) == "GCA_000220375.1:c0"
    assert hdb.target_seq_name("GCA_9.9", 7) == "GCA_9.9:c7"


# ── baseline loader ──────────────────────────────────────────────────────────
def test_load_baseline_parses(tmp_path) -> None:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(_baseline_dict([_row("A", 2, 100), _row("B", 1, 50)])))
    baseline, total_bp, n = hdb.load_baseline(p)
    assert n == 2 and total_bp == 150
    assert baseline == {
        "A": {"n_contigs": 2, "total_bp": 100},
        "B": {"n_contigs": 1, "total_bp": 50},
    }


def test_load_baseline_rejects_missing_per_genome(tmp_path) -> None:
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"n_genomes": 0, "total_bp": 0}))
    with pytest.raises(hdb.HomologDbError):
        hdb.load_baseline(p)


def test_load_baseline_rejects_duplicate_accession(tmp_path) -> None:
    p = tmp_path / "b.json"
    p.write_text(json.dumps(_baseline_dict([_row("A", 1, 10), _row("A", 1, 10)])))
    with pytest.raises(hdb.HomologDbError):
        hdb.load_baseline(p)


def test_load_baseline_rejects_aggregate_mismatch(tmp_path) -> None:
    # MUST-FIRE: a total_bp that disagrees with Σ per-genome is a corrupt baseline.
    p = tmp_path / "b.json"
    d = _baseline_dict([_row("A", 1, 10), _row("B", 1, 20)])
    d["total_bp"] = 999
    p.write_text(json.dumps(d))
    with pytest.raises(hdb.HomologDbError):
        hdb.load_baseline(p)


def test_load_baseline_rejects_malformed_row(tmp_path) -> None:
    # A non-integer total_bp must surface as HomologDbError, not a bare ValueError.
    p = tmp_path / "b.json"
    d = _baseline_dict([_row("A", 1, 10)])
    d["per_genome"][0]["total_bp"] = "not-an-int"
    p.write_text(json.dumps(d))
    with pytest.raises(hdb.HomologDbError):
        hdb.load_baseline(p)


def test_load_baseline_rejects_invalid_json(tmp_path) -> None:
    p = tmp_path / "b.json"
    p.write_text("{ this is not json")
    with pytest.raises(hdb.HomologDbError):
        hdb.load_baseline(p)


def test_load_baseline_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(hdb.HomologDbError):
        hdb.load_baseline(tmp_path / "does_not_exist.json")


# ── concatenation + header rewrite ───────────────────────────────────────────
def test_concatenate_rewrites_headers_skips_empty(tmp_path) -> None:
    gdir = tmp_path / "genomes"
    # Contig 0 empty (skipped, but consumes index 0), contig 1 = 40 nt, contig 2 = 20 nt.
    _write_genome(gdir, "GCA_A.1", ["", _seq(40, 1), _seq(20, 2)])
    _write_genome(gdir, "GCA_B.1", [_seq(30, 3)])
    out = tmp_path / "target.fna"
    measured = hdb.concatenate_target_fasta(gdir, ["GCA_A.1", "GCA_B.1"], out)

    assert measured["per_genome"]["GCA_A.1"] == {"n_contigs": 2, "total_bp": 60}
    assert measured["per_genome"]["GCA_B.1"] == {"n_contigs": 1, "total_bp": 30}
    assert measured["n_sequences"] == 3 and measured["total_bp"] == 90

    headers = [ln[1:].split()[0] for ln in out.read_text().splitlines() if ln.startswith(">")]
    # Empty contig 0 is skipped but its index is consumed → A's contigs are c1, c2 (not c0, c1).
    assert headers == ["GCA_A.1:c1", "GCA_A.1:c2", "GCA_B.1:c0"]


def test_concatenate_raises_on_all_empty(tmp_path) -> None:
    gdir = tmp_path / "genomes"
    _write_genome(gdir, "GCA_E.1", ["", ""])
    with pytest.raises(hdb.HomologDbError):
        hdb.concatenate_target_fasta(gdir, ["GCA_E.1"], tmp_path / "t.fna")


def test_concatenate_raises_on_missing_genome(tmp_path) -> None:
    gdir = tmp_path / "genomes"
    gdir.mkdir()
    with pytest.raises(hdb.HomologDbError):
        hdb.concatenate_target_fasta(gdir, ["GCA_absent.1"], tmp_path / "t.fna")


# ── integrity: the two-independent-reads guard (silent-shrink MUST-FIRE) ─────
def _measured(per_genome: dict[str, dict[str, int]]) -> dict:
    return {
        "per_genome": per_genome,
        "n_sequences": sum(v["n_contigs"] for v in per_genome.values()),
        "total_bp": sum(v["total_bp"] for v in per_genome.values()),
    }


def test_verify_integrity_passes_on_match() -> None:
    base = {"A": {"n_contigs": 1, "total_bp": 100}, "B": {"n_contigs": 2, "total_bp": 200}}
    hdb.verify_integrity(_measured(dict(base)), base, 300)  # no raise


def test_verify_integrity_bites_on_dropped_genome() -> None:
    base = {"A": {"n_contigs": 1, "total_bp": 100}, "B": {"n_contigs": 2, "total_bp": 200}}
    # Silent shrink: B was dropped from the indexed set.
    with pytest.raises(hdb.HomologDbError):
        hdb.verify_integrity(_measured({"A": {"n_contigs": 1, "total_bp": 100}}), base, 300)


def test_verify_integrity_bites_on_extra_genome() -> None:
    base = {"A": {"n_contigs": 1, "total_bp": 100}}
    extra = {"A": {"n_contigs": 1, "total_bp": 100}, "X": {"n_contigs": 1, "total_bp": 5}}
    with pytest.raises(hdb.HomologDbError):
        hdb.verify_integrity(_measured(extra), base, 100)


def test_verify_integrity_bites_on_truncated_bp() -> None:
    base = {"A": {"n_contigs": 1, "total_bp": 100}}
    # A was truncated to 60 bp since build-manifest.
    with pytest.raises(hdb.HomologDbError):
        hdb.verify_integrity(_measured({"A": {"n_contigs": 1, "total_bp": 60}}), base, 100)


def test_verify_integrity_bites_on_aggregate_mismatch() -> None:
    base = {"A": {"n_contigs": 1, "total_bp": 100}}
    m = _measured({"A": {"n_contigs": 1, "total_bp": 100}})
    with pytest.raises(hdb.HomologDbError):
        hdb.verify_integrity(m, base, 999)  # baseline aggregate disagrees


# ── version parse + assert (A12 pin, asserted where used) ────────────────────
def test_parse_makeblastdb_version() -> None:
    text = "makeblastdb: 2.17.0+\n Package: blast 2.17.0, build Aug 11 2025 09:46:06"
    assert hdb.parse_makeblastdb_version(text) == "2.17.0"


def test_parse_hmmer_version() -> None:
    text = (
        "# nhmmer :: search a DNA model, alignment, or sequence against a DNA database\n"
        "# HMMER 3.4 (Aug 2023); http://hmmer.org/\n"
    )
    assert hdb.parse_hmmer_version(text) == "3.4"


def test_assert_tool_version_drift_raises() -> None:
    hdb.assert_tool_version("blast", "2.17.0", "2.17.0")  # no raise
    with pytest.raises(hdb.HomologDbError):
        hdb.assert_tool_version("blast", "2.16.0", "2.17.0")
    with pytest.raises(hdb.HomologDbError):
        hdb.assert_tool_version("hmmer", "3.3.2", "3.4")


# ── tabular-output parsers (real formats captured in-env) ────────────────────
def test_parse_blast_tabular() -> None:
    text = (
        "probe_A\tGCA_A.1:c0\t100.000\t500\t0.0\t924\n"
        "probe_A\tGCA_B.1:c0\t72.100\t120\t3e-08\t95\n"
    )
    rows = hdb.parse_blast_tabular(text)
    assert rows[0] == {
        "qseqid": "probe_A",
        "sseqid": "GCA_A.1:c0",
        "pident": "100.000",
        "length": "500",
        "evalue": "0.0",
        "bitscore": "924",
    }
    assert len(rows) == 2


def test_parse_blast_tabular_rejects_short_row() -> None:
    with pytest.raises(hdb.HomologDbError):
        hdb.parse_blast_tabular("probe_A\tGCA_A.1:c0\t100.0\n")  # only 3 fields


def test_parse_nhmmer_tblout() -> None:
    # 16-column nhmmer --tblout data line: target(0) ... E-value(12) score(13) bias(14) desc(15+)
    line = (
        "GCA_A.1:c0 - probe_A - 1 500 301 800 301 800 1200 + "
        "1.8e-161 526.1 0.0 synthetic toy genome"
    )
    rows = hdb.parse_nhmmer_tblout("# comment\n" + line + "\n")
    assert rows == [{"target": "GCA_A.1:c0", "evalue": "1.8e-161", "score": "526.1"}]


# ── probe extraction ─────────────────────────────────────────────────────────
def test_extract_probe_picks_first_long_contig() -> None:
    # A short contig at index 0, a long one at index 1 → probe's contig index must be 1.
    fasta = _genome_fasta([_seq(100, 9), _seq(600, 10)])
    ci, offset, sub = hdb.extract_probe(fasta, probe_len=500)
    assert ci == 1 and len(sub) == 500
    assert offset == min(max(600 // 4, 0), 600 - 500)  # deterministic mid-contig, no RNG


def test_extract_probe_raises_when_no_long_contig() -> None:
    with pytest.raises(hdb.HomologDbError):
        hdb.extract_probe(_genome_fasta([_seq(100, 1)]), probe_len=500)


def test_extract_probe_skips_N_window() -> None:
    # The quarter-point window (offset 375) is a 500-nt N-gap; the next stride window (875) is
    # clean. A fixed quarter-point would have returned an N-run probe that cannot self-recover.
    seq = _seq(375, 7) + "N" * 500 + _seq(625, 8)  # len 1500
    ci, offset, sub = hdb.extract_probe(_genome_fasta([seq]), probe_len=500)
    assert ci == 0 and offset == 875 and set(sub) <= set("ACGT")


def test_extract_probe_raises_when_all_N() -> None:
    # A long-enough contig with no pure-ACGT window must fail loud, not return a dead probe.
    with pytest.raises(hdb.HomologDbError):
        hdb.extract_probe(_genome_fasta(["N" * 600]), probe_len=500)


# ── the must-fire round-trip control predicate ───────────────────────────────
def _blast_hit(sseqid: str, pident="100.000", length="500", evalue="0.0") -> dict:
    return {
        "qseqid": "probe",
        "sseqid": sseqid,
        "pident": pident,
        "length": length,
        "evalue": evalue,
        "bitscore": "924",
    }


def _nhmmer_hit(target: str, evalue="1e-100") -> dict:
    return {"target": target, "evalue": evalue, "score": "500.0"}


def test_assert_self_recovery_passes() -> None:
    out = hdb.assert_self_recovery(
        accession="GCA_A.1",
        contig_index=0,
        probe_len=500,
        blast_hits=[_blast_hit("GCA_A.1:c0")],
        nhmmer_hits=[_nhmmer_hit("GCA_A.1:c0")],
    )
    assert out["recovered"] is True and out["target"] == "GCA_A.1:c0"


def test_self_recovery_bites_on_wrong_blast_genome() -> None:
    # INVERSION (count-preserving), BLAST leg isolated: exactly one strong blastn hit but at the
    # WRONG genome, while nhmmer is correct — only the blastn identity guard can refuse it. Both
    # legs wrong would let the nhmmer check mask a broken blast check ([[symmetric-count-...]]).
    with pytest.raises(hdb.HomologDbError):
        hdb.assert_self_recovery(
            accession="GCA_A.1",
            contig_index=0,
            probe_len=500,
            blast_hits=[_blast_hit("GCA_B.1:c0")],
            nhmmer_hits=[_nhmmer_hit("GCA_A.1:c0")],
        )


def test_self_recovery_bites_on_low_identity() -> None:
    with pytest.raises(hdb.HomologDbError):
        hdb.assert_self_recovery(
            accession="GCA_A.1",
            contig_index=0,
            probe_len=500,
            blast_hits=[_blast_hit("GCA_A.1:c0", pident="90.0")],
            nhmmer_hits=[_nhmmer_hit("GCA_A.1:c0")],
        )


def test_self_recovery_bites_on_high_evalue() -> None:
    with pytest.raises(hdb.HomologDbError):
        hdb.assert_self_recovery(
            accession="GCA_A.1",
            contig_index=0,
            probe_len=500,
            blast_hits=[_blast_hit("GCA_A.1:c0", evalue="1.0")],
            nhmmer_hits=[_nhmmer_hit("GCA_A.1:c0")],
        )


def test_self_recovery_bites_on_empty_blast() -> None:
    with pytest.raises(hdb.HomologDbError):
        hdb.assert_self_recovery(
            accession="GCA_A.1",
            contig_index=0,
            probe_len=500,
            blast_hits=[],
            nhmmer_hits=[_nhmmer_hit("GCA_A.1:c0")],
        )


def test_self_recovery_bites_on_missing_nhmmer() -> None:
    with pytest.raises(hdb.HomologDbError):
        hdb.assert_self_recovery(
            accession="GCA_A.1",
            contig_index=0,
            probe_len=500,
            blast_hits=[_blast_hit("GCA_A.1:c0")],
            nhmmer_hits=[_nhmmer_hit("GCA_B.1:c0")],
        )


# ── report ───────────────────────────────────────────────────────────────────
def test_build_report_flags_and_shape() -> None:
    probe = {
        "accession": "GCA_A.1",
        "target": "GCA_A.1:c0",
        "probe_len": 500,
        "recovered": True,
        "blastn": {"pident": "100.000", "length": "500", "evalue": "0.0"},
        "nhmmer": {"evalue": "1e-100", "score": "500.0"},
    }
    rep = hdb.build_report(
        baseline_n=2500,
        measured=_measured({"A": {"n_contigs": 1, "total_bp": 100}}),
        db_counts=(1, 100),
        fm_index_bytes=4096,
        blast_version="2.17.0",
        hmmer_version="3.4",
        probes=[probe],
    )
    assert rep["n_genomes"] == 2500 and rep["round_trip"]["all_recovered"] is True
    for flag in (
        "builds_index_only",
        "runs_no_genome_scan",
        "runs_no_cmsearch",
        "counts_no_candidates",
        "assembles_no_homolog_set",
        "pins_no_inclusion_threshold",
    ):
        assert rep[flag] is True


def test_build_report_rejects_no_probes() -> None:
    with pytest.raises(hdb.HomologDbError):
        hdb.build_report(
            baseline_n=1,
            measured=_measured({"A": {"n_contigs": 1, "total_bp": 1}}),
            db_counts=(1, 1),
            fm_index_bytes=1,
            blast_version="2.17.0",
            hmmer_version="3.4",
            probes=[],
        )
