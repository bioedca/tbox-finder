"""P2-10e-msa-producer — unit tests for the per-candidate covariation producer (ADR-0005 A10).

Bare-CI tier: the binary-shelling stages (nhmmer/mlocarna/R-scape) are validated end-to-end by
`slurm/p2/mine_round_measure.sbatch`; here we test the pure transport A10 stands on — the shard/
manifest I/O, the collision-free per-candidate work-dir, the deterministic sample/partition, the
disjoint-merge guard, the status-table shape, the fail-closed search branch (a logic error is
recorded, not raised; an env fault propagates), the None-MSA → unavailable score, and the Phase-1
measurement summary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tbox_finder.mining import covariation_producer as cp
from tbox_finder.mining.homolog_db import HomologDbError
from tbox_finder.mining.homolog_msa import HomologMsaError
from tbox_finder.mining.spare_rule import STATUS_UNAVAILABLE


def _spec(cid: str = "GCA_1.1:c0:0-30", acc: str = "GCA_1.1:c0", lo: int = 0, hi: int = 30):
    return cp.CandidateSpec(candidate_id=cid, accession=acc, locus_start=lo, locus_end=hi)


# --------------------------------------------------------------------------- #
# CandidateSpec — 0-based half-open span validation
# --------------------------------------------------------------------------- #
def test_candidate_spec_rejects_bad_span():
    with pytest.raises(cp.ProducerError):
        cp.CandidateSpec(candidate_id="c", accession="GCA_1.1:c0", locus_start=30, locus_end=30)
    with pytest.raises(cp.ProducerError):
        cp.CandidateSpec(candidate_id="c", accession="GCA_1.1:c0", locus_start=30, locus_end=10)
    with pytest.raises(cp.ProducerError):
        cp.CandidateSpec(candidate_id="", accession="GCA_1.1:c0", locus_start=0, locus_end=10)


def test_spec_dict_roundtrip_ignores_extra_keys():
    row = cp.spec_to_dict(_spec())
    row["score"] = 0.97  # the FP manifest carries score/pool; the producer ignores them
    row["pool"] = "genomic_window"
    back = cp.spec_from_dict(row)
    assert (back.candidate_id, back.accession, back.locus_start, back.locus_end) == (
        "GCA_1.1:c0:0-30",
        "GCA_1.1:c0",
        0,
        30,
    )


def test_manifest_roundtrip(tmp_path: Path):
    specs = [_spec("GCA_1.1:c0:0-30"), _spec("GCA_2.1:c1:5-40", "GCA_2.1:c1", 5, 40)]
    path = tmp_path / "m.json"
    cp.write_candidate_manifest(specs, path)
    back = cp.read_candidate_manifest(path)
    assert [s.candidate_id for s in back] == [s.candidate_id for s in specs]


# --------------------------------------------------------------------------- #
# candidate_slug — deterministic, sanitised, collision-free
# --------------------------------------------------------------------------- #
def test_candidate_slug_is_safe_deterministic_and_distinct():
    a = cp.candidate_slug("GCA_1.1:c0:5-30")
    assert a == cp.candidate_slug("GCA_1.1:c0:5-30")  # deterministic
    assert ":" not in a and "/" not in a  # filesystem-safe
    # Two ids that sanitise to the same prefix must not collide (the digest disambiguates).
    assert cp.candidate_slug("a:b") != cp.candidate_slug("a_b")


# --------------------------------------------------------------------------- #
# select_sample / partition_shards — deterministic, disjoint, complete
# --------------------------------------------------------------------------- #
def test_select_sample_is_first_k_sorted():
    specs = [_spec(f"GCA_{i}.1:c0:0-30") for i in (3, 1, 2, 0)]
    sample = cp.select_sample(specs, 2)
    assert [s.candidate_id for s in sample] == ["GCA_0.1:c0:0-30", "GCA_1.1:c0:0-30"]
    with pytest.raises(cp.ProducerError):
        cp.select_sample(specs, 0)


def test_partition_shards_disjoint_and_complete():
    specs = [_spec(f"GCA_{i:02d}.1:c0:0-30") for i in range(10)]
    shards = cp.partition_shards(specs, 3)
    assert len(shards) == 3
    sizes = sorted(len(s) for s in shards)
    assert sizes == [3, 3, 4]  # near-equal
    seen = [c.candidate_id for shard in shards for c in shard]
    assert sorted(seen) == sorted(s.candidate_id for s in specs)  # partition, no loss/dup


# --------------------------------------------------------------------------- #
# merge_status_tables — disjoint shards, duplicate id is a partition bug
# --------------------------------------------------------------------------- #
def _status_table(pairs):
    return cp._status_table([{cp.CANDIDATE_ID_KEY: cid, "status": st} for cid, st in pairs])


def test_merge_status_tables_concatenates(tmp_path: Path):
    (tmp_path / "s0.json").write_text(__import__("json").dumps(_status_table([("a", "passed")])))
    (tmp_path / "s1.json").write_text(
        __import__("json").dumps(_status_table([("b", "failed"), ("c", STATUS_UNAVAILABLE)]))
    )
    merged = cp.merge_status_tables([tmp_path / "s0.json", tmp_path / "s1.json"])
    assert merged["status"] == {"a": "passed", "b": "failed", "c": STATUS_UNAVAILABLE}
    assert merged["status_counts"]["passed"] == 1


def test_merge_rejects_duplicate_candidate_id(tmp_path: Path):
    import json

    (tmp_path / "s0.json").write_text(json.dumps(_status_table([("dup", "passed")])))
    (tmp_path / "s1.json").write_text(json.dumps(_status_table([("dup", "failed")])))
    with pytest.raises(cp.ProducerError):
        cp.merge_status_tables([tmp_path / "s0.json", tmp_path / "s1.json"])


def test_load_status_map_guards_rows_status_disagreement(tmp_path: Path):
    import json

    table = _status_table([("a", "passed"), ("b", "failed")])
    good = tmp_path / "good.json"
    good.write_text(json.dumps(table))
    assert cp.load_status_map(good) == {"a": "passed", "b": "failed"}
    # Corrupt the 'status' map so it disagrees with 'rows' → must raise, not silently trust either.
    table["status"]["a"] = "failed"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(table))
    with pytest.raises(cp.ProducerError):
        cp.load_status_map(bad)


# --------------------------------------------------------------------------- #
# search_shard — fail-closed on a logic error, fail-LOUD on an env fault
# --------------------------------------------------------------------------- #
def _genome(tmp_path: Path) -> Path:
    d = tmp_path / "genomes"
    d.mkdir()
    (d / "GCA_1.1.fna").write_text(
        ">c0\n" + "ACGT" * 20 + "\n", encoding="utf-8"
    )  # one 80-nt contig
    return d


def test_search_shard_records_out_of_contig_candidate_as_not_producible(tmp_path: Path):
    gdir = _genome(tmp_path)
    # Contig index c9 does not exist ⇒ resolve raises HomologMsaError ⇒ caught ⇒ search_ok=False.
    bad = cp.CandidateSpec(
        candidate_id="GCA_1.1:c9:0-30", accession="GCA_1.1:c9", locus_start=0, locus_end=30
    )
    rows = cp.search_shard(
        [bad],
        workroot=tmp_path / "work",
        engine="nhmmer",
        evalue=100.0,
        max_target_seqs=500,
        min_pident=40.0,
        min_cov=0.5,
        genome_dir=gdir,
    )
    assert rows[0]["search_ok"] is False and rows[0]["sufficient"] is False


def test_search_shard_propagates_env_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gdir = _genome(tmp_path)
    good = cp.CandidateSpec(
        candidate_id="GCA_1.1:c0:0-30", accession="GCA_1.1:c0", locus_start=0, locus_end=30
    )

    def _missing_tool(name: str) -> str:
        raise HomologDbError(f"{name} not on PATH")

    # A missing binary is an env fault affecting EVERY candidate — it must propagate, not be
    # swallowed as a per-candidate 'unavailable' that would mine nothing.
    monkeypatch.setattr("tbox_finder.mining.homolog_msa.tool_path", _missing_tool)
    with pytest.raises(HomologDbError):
        cp.search_shard(
            [good],
            workroot=tmp_path / "work",
            engine="nhmmer",
            evalue=100.0,
            max_target_seqs=500,
            min_pident=40.0,
            min_cov=0.5,
            genome_dir=gdir,
        )


# --------------------------------------------------------------------------- #
# align_shard — both fail-closed branches (no MSA written ⇒ score reads unavailable ⇒ spared)
# --------------------------------------------------------------------------- #
def test_align_shard_skips_insufficient_homologs(tmp_path: Path):
    spec = _spec("GCA_1.1:c0:0-30")
    wd = cp.candidate_workdir(tmp_path / "work", spec.candidate_id)
    wd.mkdir(parents=True, exist_ok=True)
    # The search stage found too few homologs → align_shard skips WITHOUT calling mlocarna.
    (wd / "search.json").write_text('{"sufficient": false}', encoding="utf-8")
    rows = cp.align_shard([spec], workroot=tmp_path / "work")
    assert rows[0]["aligned"] is False and rows[0]["reason"] == "insufficient_homologs"
    assert not (wd / "msa.sto").exists()  # no MSA ⇒ the score stage reads unavailable ⇒ spared


def test_align_shard_records_align_failure_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = _spec("GCA_1.1:c0:0-30")
    wd = cp.candidate_workdir(tmp_path / "work", spec.candidate_id)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "search.json").write_text('{"sufficient": true}', encoding="utf-8")

    def _boom(**_kwargs):
        raise HomologMsaError("mlocarna emitted no #=GC SS_cons")

    # A per-candidate alignment failure is RECORDED aligned=False, never raised — the same
    # fail-closed reason the score stage then reads as unavailable (⇒ spared), never mined.
    monkeypatch.setattr(cp, "align_candidate", _boom)
    rows = cp.align_shard([spec], workroot=tmp_path / "work")
    assert rows[0]["aligned"] is False and rows[0]["reason"] == "align_failed"
    assert not (wd / "msa.sto").exists()


# --------------------------------------------------------------------------- #
# score_shard — a candidate with no MSA scores 'unavailable' (⇒ spared), no R-scape needed
# --------------------------------------------------------------------------- #
def test_score_shard_no_msa_is_unavailable(tmp_path: Path):
    spec = _spec("GCA_1.1:c0:0-30")
    # No search.json / msa.sto written → score_msa(None) → STATUS_UNAVAILABLE (fail-closed).
    table = cp.score_shard([spec], workroot=tmp_path / "work", out_table=tmp_path / "st.json")
    assert table["status"][spec.candidate_id] == STATUS_UNAVAILABLE
    assert (tmp_path / "st.json").is_file()


# --------------------------------------------------------------------------- #
# measure_report — N0 from the manifest, per-stage wall + envelope from the sample
# --------------------------------------------------------------------------- #
def test_measure_report_summarises_n0_and_wall(tmp_path: Path):
    import json

    manifest = tmp_path / "fp.json"
    cp.write_candidate_manifest([_spec(f"GCA_{i}.1:c0:0-30") for i in range(120)], manifest)
    rows = [
        {
            cp.CANDIDATE_ID_KEY: "a",
            "status": "failed",
            "search_wall_s": 2.0,
            "align_wall_s": 100.0,
            "score_wall_s": 8.0,
            "msa_depth": 24,
        },
        {
            cp.CANDIDATE_ID_KEY: "b",
            "status": STATUS_UNAVAILABLE,
            "search_wall_s": 1.0,
            "align_wall_s": 0.0,
            "score_wall_s": 0.0,
            "msa_depth": 0,
        },
    ]
    status = tmp_path / "st.json"
    status.write_text(json.dumps(cp._status_table(rows)))
    rep = cp.measure_report(manifest, status, out_path=tmp_path / "rep.json", array_widths=(8, 48))
    assert rep["n0_false_positives"] == 120
    assert rep["k_sample"] == 2
    # total-per-candidate median of {110.0, 1.0} = 55.5; core-h = 120*55.5/3600.
    assert rep["wall_seconds"]["total_per_candidate"]["median"] == 55.5
    assert rep["extrapolated_envelope"]["48"]["array_width"] == 48
    assert rep["extrapolated_envelope"]["8"]["est_core_h"] == round(120 * 55.5 / 3600.0, 2)
