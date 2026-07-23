"""ML-behavioral gate for the A8 substrate pre-scan (ADR-0005 A8, P2-10c'-d).

Unlike ``tests/unit`` (synthetic in-memory scans), this gate is anchored on a **real** golden
fixture minted by ``scripts/make_substrate_prescan_control.py mint-fixture`` — real T-box
records from ``RF00230_master.fa`` embedded in background, scanned by a real local ``cmsearch``
(§8.7: real fixtures, real pipeline, no mocking). CI parses that committed real ``--tblout``
output; a ``TBOX_REQUIRE_INFERNAL`` variant re-runs ``cmsearch`` live where the binary exists.
The committed fixture is a git-tracked tier — its absence is a hard failure, not a skip
([[regenerated-report-breaks-shape-lock-test]] discipline).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tbox_finder import infernal
from tbox_finder.mining import substrate_prescan as sp

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests/fixtures/substrate_prescan"


def _load(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _n_shards() -> int:
    return int(_load("golden_report.json")["n_shards"])


def test_fixture_present():
    # git-tracked golden fixture — absence is a hard failure (a skip could rot into no coverage)
    for name in ("golden_report.json", "selection_accessions.json"):
        assert (FIX / name).exists(), f"missing committed fixture {name}"
    for shard in range(_n_shards()):
        for sub in (
            f"control_shard_{shard}.json",
            f"shard_specs/shard_{shard}.json",
            f"tblout/shard_{shard}.tblout",
        ):
            assert (FIX / sub).exists(), f"missing committed fixture {sub}"


def test_golden_report_certifies_and_separated():
    report = _load("golden_report.json")
    selected = _load("selection_accessions.json")
    # the REAL cmsearch run produced a certifying report over the real spike/null control
    assert report["overall_pass"] is True
    assert sp.validate_report(report, selected_accessions=selected) == []
    # and it genuinely separated: every spike detected, no matched-null detected
    assert (
        report["substrate_removal_rate"] == 0.0
    )  # production is locus-free background (admissible)
    for seg in report["segments"]:
        assert (
            seg["arms"]["spike"]["n_removed"] == seg["arms"]["spike"]["n_windows"] >= sp.MIN_SPIKE_N
        )
        assert seg["arms"]["null"]["n_removed"] == 0
        assert seg["cm_sha256"] == sp.RF00230_CM_SHA256


def test_golden_sabotage_wrong_cm_fails_validate():
    report = _load("golden_report.json")
    selected = _load("selection_accessions.json")
    report["segments"][0]["cm_sha256"] = "0" * 64  # a degraded/swapped production CM
    problems = sp.validate_report(report, selected_accessions=selected)
    assert any("cm_identity" in p for p in problems), problems


def test_golden_sabotage_dropped_genome_fails_validate():
    report = _load("golden_report.json")
    selected = _load("selection_accessions.json") + ["GCA_SILENTLY_DROPPED.1"]
    problems = sp.validate_report(report, selected_accessions=selected)
    assert any("genome_completeness" in p for p in problems), problems


def test_generator_matchedness_holds_and_sabotage_fires():
    arms = _load("control_shard_0.json")["arms"]
    spike_meta = sp.arm_metadata(arms["spike"])
    null_meta = sp.arm_metadata(arms["null"])
    assert sp.arms_matched(spike_meta, null_meta)  # equal count/length/composition by construction
    assert len(arms["spike"]) >= sp.MIN_SPIKE_N
    # sabotage: truncate one null window → length multiset differs → NOT matched
    perturbed = dict(arms["null"])
    k = sorted(perturbed)[0]
    perturbed[k] = perturbed[k][:-7]
    assert not sp.arms_matched(spike_meta, sp.arm_metadata(perturbed))
    # a spike and null window differ (the null is a shuffle, not a copy)
    assert arms["spike"][sorted(arms["spike"])[0]] != arms["null"][sorted(arms["null"])[0]]


@pytest.mark.parametrize("shard", range(_n_shards()))
def test_shard_rebuilds_from_real_tblout(shard):
    # exercise the REAL parse+join path on committed real cmsearch output, EVERY shard, even in
    # bare CI (shard 1 carries 20 of the 40 recovered spikes — CodeRabbit PR #76).
    report = _load("golden_report.json")
    arms = _load(f"control_shard_{shard}.json")["arms"]
    spec = _load(f"shard_specs/shard_{shard}.json")
    tblout_text = (FIX / f"tblout/shard_{shard}.tblout").read_text(encoding="utf-8")
    hits = infernal.parse_tblout(tblout_text)
    seg = sp.build_shard_segment(
        shard,
        arm_windows={
            "production": spec["production"],
            "spike": arms["spike"],
            "null": arms["null"],
        },
        hits=hits,
        tblout_text=tblout_text,
        cm_sha256=sp.RF00230_CM_SHA256,
        expected_production_windows=spec["expected_production_windows"],
        score_threshold=report["score_threshold"],
        shard_ok=True,
        genome_windows=spec["genome_windows"],
    )
    g = next(s for s in report["segments"] if s["shard"] == shard)
    for key in (
        "n_hits_reported",
        "n_hits_joined",
        "invocation_id",
        "n_windows_scanned",
        "n_distinct_names",
    ):
        assert seg[key] == g[key], key
    for arm in sp.ARMS:
        assert seg["arms"][arm]["n_removed"] == g["arms"][arm]["n_removed"], arm
    # shard 1's spikes must be genuinely recovered too (not just shard 0's)
    assert seg["arms"]["spike"]["n_removed"] == seg["arms"]["spike"]["n_windows"] >= sp.MIN_SPIKE_N


def test_live_cmsearch_separates_spike_from_null(tmp_path):
    """Re-run cmsearch live on the committed control+production (only where cmsearch exists)."""
    if not infernal.cmsearch_available():
        if os.environ.get("TBOX_REQUIRE_INFERNAL") == "1":
            pytest.fail("TBOX_REQUIRE_INFERNAL=1 but cmsearch is not on PATH")
        pytest.skip("cmsearch not available (bare CI) — the golden fixture carries the real result")
    report = _load("golden_report.json")
    arms = _load("control_shard_0.json")["arms"]
    spec = _load("shard_specs/shard_0.json")
    merged = {**spec["production"], **arms["spike"], **arms["null"]}
    fasta = infernal.write_fasta(merged, tmp_path / "shard0.fna")
    tblout = tmp_path / "shard0.tblout"
    hits = infernal.run_cmsearch(infernal.RF00230_CM, fasta, tblout, cut_ga=False, cpu=2)
    seg = sp.build_shard_segment(
        0,
        arm_windows={
            "production": spec["production"],
            "spike": arms["spike"],
            "null": arms["null"],
        },
        hits=hits,
        tblout_text=tblout.read_text(encoding="utf-8"),
        cm_sha256=sp.RF00230_CM_SHA256,
        expected_production_windows=spec["expected_production_windows"],
        score_threshold=report["score_threshold"],
        shard_ok=True,
        genome_windows=spec["genome_windows"],
    )
    # the live detector must separate: all spikes recovered, no matched null recovered
    assert seg["arms"]["spike"]["n_removed"] == seg["arms"]["spike"]["n_windows"]
    assert seg["arms"]["null"]["n_removed"] == 0
