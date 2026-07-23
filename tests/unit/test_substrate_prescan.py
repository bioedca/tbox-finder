"""Unit tests for the off-catalogue cmsearch substrate pre-scan gate (ADR-0005 A8, P2-10c'-d).

Every one of the 8 REQUIRED clauses has a MUST-FIRE partner: a healthy scenario certifies, then
one recorded count is perturbed and the clause named for the invariant is asserted to flip FALSE
(the [[sabotage-attribution-names-the-test]] discipline — a red gate proves *some* clause bit,
so each test targets one clause). The join + parse path is exercised on real
``cmsearch --tblout`` text via :func:`tbox_finder.infernal.parse_tblout`. Bare-CI tier: stdlib
only, no ``cmsearch`` binary.
"""

from __future__ import annotations

import pytest

from tbox_finder.infernal import CmsearchHit
from tbox_finder.mining import substrate_prescan as sp

THRESH = 30.0


def _seq(key: str, n: int) -> str:
    import hashlib

    raw = hashlib.shake_256(key.encode()).digest(n)
    return "".join("ACGT"[b & 3] for b in raw)


def _shard(
    shard: int,
    *,
    n_spike: int = sp.MIN_SPIKE_N,
    n_null: int = sp.MIN_NULL_N,
    n_prod: int = 4,
    spike_detected: bool = True,
    null_detected: bool = False,
    extra_hits: list[CmsearchHit] | None = None,
) -> dict:
    """A healthy (or deliberately perturbed) single-shard segment built via the real join path."""
    production = {f"prod_{shard}_{i}": _seq(f"p{shard}{i}", 260) for i in range(n_prod)}
    spike = {f"spike_{shard}_{i}": _seq(f"s{shard}{i}", 300) for i in range(n_spike)}
    # matched null: same length + composition (a reversal preserves both), locus destroyed
    null = {f"null_{shard}_{i}": _seq(f"s{shard}{i}", 300)[::-1] for i in range(n_null)}
    genome_windows = {f"GCA_T{shard:02d}{i:03d}.1": 1 for i in range(n_prod)}

    hits: list[CmsearchHit] = []
    if spike_detected:
        hits += [CmsearchHit(target=n, score=110.0, evalue=1e-25) for n in spike]
    if null_detected:
        hits += [CmsearchHit(target=n, score=110.0, evalue=1e-25) for n in null]
    hits += extra_hits or []
    tblout_text = "\n".join(f"{h.target} score={h.score}" for h in hits) or "# empty\n"

    return sp.build_shard_segment(
        shard,
        arm_windows={"production": production, "spike": spike, "null": null},
        hits=hits,
        tblout_text=tblout_text,
        cm_sha256=sp.RF00230_CM_SHA256,
        expected_production_windows=n_prod,
        score_threshold=THRESH,
        shard_ok=True,
        genome_windows=genome_windows,
    )


def _healthy() -> tuple[list[dict], list[str]]:
    segs = [_shard(0), _shard(1)]
    selected = sorted({a for s in segs for a in s["genomes"]})
    return segs, selected


def _clauses(segs, selected, *, prior=None):
    return sp.derive_clauses(
        segs, n_shards_expected=len(segs), selected_accessions=selected, prior_genome_windows=prior
    )


# ── the healthy scenario certifies, and is expressive (not trivial-vs-trivial) ──────────────
def test_healthy_certifies_and_is_expressive():
    segs, selected = _healthy()
    cl = _clauses(segs, selected)
    assert all(cl[k] for k in sp.REQUIRED_CLAUSES), cl
    # expressiveness: the detector actually separated (else the whole gate is vacuous)
    for s in segs:
        assert s["arms"]["spike"]["n_removed"] == sp.MIN_SPIKE_N
        assert s["arms"]["null"]["n_removed"] == 0
        assert s["arms"]["production"]["n_removed"] == 0
    report = sp.build_report(
        segs,
        n_shards_expected=2,
        selected_accessions=selected,
        prior_genome_windows=None,
        score_threshold=THRESH,
        substrate_scanned=True,
        accessed="2026-07-23",
    )
    assert report["overall_pass"] is True
    assert sp.validate_report(report, selected_accessions=selected) == []
    assert report["substrate_removal_rate"] == 0.0  # admissible low removal is NOT a failure


# ── (i) production denominator — strict loaded==expected + emptiness ─────────────────────────
def test_i_consumed_ne_expected_must_fire():
    segs, selected = _healthy()
    segs[0]["expected_production_windows"] += 1  # a dropped/truncated production window
    assert _clauses(segs, selected)["production_denominator_consistent"] is False


def test_i_emptiness_must_fire():
    segs, selected = _healthy()
    # expected > 0 but consumed 0: a shard that wrote no production windows yet reads healthy
    segs[0]["arms"]["production"]["n_windows"] = 0
    segs[0]["genome_windows"] = {}
    segs[0]["genomes"] = []
    # keep selected referencing shard-0 genomes so the emptiness (not coverage) is what bites
    assert _clauses(segs, selected)["production_denominator_consistent"] is False


# ── (ii) detector live in every shard — per-shard spike removal > 0 ──────────────────────────
def test_ii_dead_shard_spike_must_fire():
    segs = [_shard(0), _shard(1, spike_detected=False)]  # one dead shard among many
    selected = sorted({a for s in segs for a in s["genomes"]})
    assert _clauses(segs, selected)["detector_live_every_shard"] is False


# ── (iii) power arms min-N + matchedness ────────────────────────────────────────────────────
def test_iii_below_min_spike_must_fire():
    segs = [_shard(0, n_spike=sp.MIN_SPIKE_N - 1), _shard(1)]
    selected = sorted({a for s in segs for a in s["genomes"]})
    assert _clauses(segs, selected)["power_arms_min_n"] is False


def test_iii_unmatched_null_length_must_fire():
    segs, selected = _healthy()
    # perturb the null arm's length fingerprint only (composition/count untouched here)
    segs[0]["arms"]["null"]["length_multiset_sha"] = "deadbeef"
    assert _clauses(segs, selected)["power_arms_min_n"] is False


# ── (iv) STRICT spike-vs-null separation ────────────────────────────────────────────────────
def test_iv_copy_spike_into_null_must_fire():
    # the P2-10c lesson: a null that is the spike (detected identically) certifies matchedness but
    # not separation — removal(spike) − removal(null) = 0 ≤ MARGIN.
    segs = [_shard(0, null_detected=True), _shard(1, null_detected=True)]
    selected = sorted({a for s in segs for a in s["genomes"]})
    cl = _clauses(segs, selected)
    assert cl["spike_null_separation"] is False
    assert cl["detector_live_every_shard"] is True  # the spike arm IS live — only (iv) bites


def test_iv_dead_shard_both_arms_must_fire():
    segs = [_shard(0), _shard(1, spike_detected=False)]  # 0 in both arms → 0 ≤ MARGIN
    selected = sorted({a for s in segs for a in s["genomes"]})
    assert _clauses(segs, selected)["spike_null_separation"] is False


# ── (v) hit→window mapping totality + unique naming + name-mismatch ─────────────────────────
def test_v_duplicate_name_must_fire():
    segs, selected = _healthy()
    segs[0]["n_distinct_names"] -= 1  # two windows collapsed to one name
    assert _clauses(segs, selected)["hit_window_mapping_total"] is False


def test_v_unjoined_hit_must_fire():
    segs, selected = _healthy()
    segs[0]["n_hits_reported"] += 1  # a reported hit that maps to no scanned window
    assert _clauses(segs, selected)["hit_window_mapping_total"] is False


def test_v_cross_arm_name_collision_fires_in_build():
    # clause (v)(a) is non-vacuous: a window name shared across two arms makes n_windows_scanned
    # (submitted, counted per arm) exceed n_distinct_names (merged keys) → mapping clause FALSE.
    spike = {f"spike_0_{i}": _seq(f"s{i}", 300) for i in range(sp.MIN_SPIKE_N)}
    null = {f"null_0_{i}": _seq(f"s{i}", 300)[::-1] for i in range(sp.MIN_NULL_N)}
    # production reuses a spike name — a naming-scheme collision the merge would silently hide
    production = {"spike_0_0": _seq("collide", 260)}
    seg = sp.build_shard_segment(
        0,
        arm_windows={"production": production, "spike": spike, "null": null},
        hits=[],
        tblout_text="# empty\n",
        cm_sha256=sp.RF00230_CM_SHA256,
        expected_production_windows=1,
        score_threshold=THRESH,
        shard_ok=True,
        genome_windows={"GCA_T00000.1": 1},
    )
    assert seg["n_windows_scanned"] > seg["n_distinct_names"]  # collision visible, not merged away
    assert _clauses([seg], ["GCA_T00000.1"])["hit_window_mapping_total"] is False


def test_v_name_mismatch_deflates_join_in_build():
    # the real build path: mutate a window's name after cmsearch named the hit → the hit no longer
    # joins, n_hits_reported > n_hits_joined, so the mapping clause fires (the .N-suffix guard)
    production = {"prod_0_0": _seq("p", 260)}
    spike = {f"spike_0_{i}": _seq(f"s{i}", 300) for i in range(sp.MIN_SPIKE_N)}
    null = {f"null_0_{i}": _seq(f"s{i}", 300)[::-1] for i in range(sp.MIN_NULL_N)}
    hits = [CmsearchHit(target=n, score=110.0, evalue=1e-25) for n in spike]
    # rename one spike window (as if an accession version suffix were normalised on one side)
    renamed = dict(spike)
    victim = "spike_0_0"
    renamed["spike_0_0_RENAMED"] = renamed.pop(victim)
    seg = sp.build_shard_segment(
        0,
        arm_windows={"production": production, "spike": renamed, "null": null},
        hits=hits,
        tblout_text="x",
        cm_sha256=sp.RF00230_CM_SHA256,
        expected_production_windows=1,
        score_threshold=THRESH,
        shard_ok=True,
        genome_windows={"GCA_T00000.1": 1},
    )
    assert seg["n_hits_reported"] == sp.MIN_SPIKE_N
    assert seg["n_hits_joined"] == sp.MIN_SPIKE_N - 1  # the renamed window's hit did not join
    assert _clauses([seg], ["GCA_T00000.1"])["hit_window_mapping_total"] is False


# ── (vi) shard completeness + one-tblout co-invocation ──────────────────────────────────────
def test_vi_missing_shard_must_fire():
    segs, selected = _healthy()
    # only 1 of 2 shards recorded (a lost array task) — aggregate still looks internally consistent
    assert (
        sp.derive_clauses(
            segs[:1],
            n_shards_expected=2,
            selected_accessions=segs[0]["genomes"],
            prior_genome_windows=None,
        )["shard_completeness_coinvocation"]
        is False
    )


def test_vi_shard_ok_false_must_fire():
    segs, selected = _healthy()
    segs[0]["shard_ok"] = False
    assert _clauses(segs, selected)["shard_completeness_coinvocation"] is False


def test_vi_two_tblout_files_must_fire():
    segs, selected = _healthy()
    segs[0]["n_tblout_files"] = 2  # production + control were split into separate cmsearch calls
    assert _clauses(segs, selected)["shard_completeness_coinvocation"] is False


def test_vi_split_invocation_id_must_fire():
    segs, selected = _healthy()
    segs[0]["arms"]["spike"]["invocation_id"] = "not-the-production-invocation"
    assert _clauses(segs, selected)["shard_completeness_coinvocation"] is False


# ── (vii) CM identity — fires independently of co-invocation ─────────────────────────────────
def test_vii_wrong_cm_must_fire():
    segs, selected = _healthy()
    segs[0]["cm_sha256"] = "0" * 64  # a truncated/swapped/wrong-path production CM
    cl = _clauses(segs, selected)
    assert cl["cm_identity"] is False
    # crucially, co-invocation still passes (all arms share one live invocation) — vii is the
    # independent backstop that catches a degraded CM under a shared invocation.
    assert cl["shard_completeness_coinvocation"] is True


# ── (viii) genome completeness — git-frozen set + no within-genome shrink ────────────────────
def test_viii_dropped_genome_must_fire():
    segs, selected = _healthy()
    selected = selected + ["GCA_MISSING.1"]  # frozen manifest names a genome the scan never covered
    assert _clauses(segs, selected)["genome_completeness"] is False


def test_viii_within_genome_shrink_must_fire():
    segs, selected = _healthy()
    prior = {a: 5 for a in selected}  # prior report had 5 windows/genome; the scan has 1 → shrink
    assert _clauses(segs, selected, prior=prior)["genome_completeness"] is False


def test_viii_maiden_fetch_ok_when_set_matches():
    segs, selected = _healthy()
    assert _clauses(segs, selected, prior=None)["genome_completeness"] is True


# ── the clause-set-completeness gate + overall_pass integrity ────────────────────────────────
def test_missing_clause_key_is_hard_fail():
    segs, selected = _healthy()
    report = sp.build_report(
        segs,
        n_shards_expected=2,
        selected_accessions=selected,
        prior_genome_windows=None,
        score_threshold=THRESH,
        substrate_scanned=True,
        accessed="2026-07-23",
    )
    del report["clauses"]["cm_identity"]  # a builder that skipped a clause
    problems = sp.validate_report(report, selected_accessions=selected)
    assert any("cm_identity" in p and "MISSING" in p for p in problems)


def test_clause_schema_version_bump_invalidates_old_report():
    segs, selected = _healthy()
    report = sp.build_report(
        segs,
        n_shards_expected=2,
        selected_accessions=selected,
        prior_genome_windows=None,
        score_threshold=THRESH,
        substrate_scanned=True,
        accessed="2026-07-23",
    )
    report["clause_schema_version"] = sp.CLAUSE_SCHEMA_VERSION + 99
    assert any(
        "clause_schema_version" in p
        for p in sp.validate_report(report, selected_accessions=selected)
    )


def test_overall_pass_must_equal_rederived_all():
    segs, selected = _healthy()
    segs[0]["cm_sha256"] = "0" * 64  # a real failure
    report = sp.build_report(
        segs,
        n_shards_expected=2,
        selected_accessions=selected,
        prior_genome_windows=None,
        score_threshold=THRESH,
        substrate_scanned=True,
        accessed="2026-07-23",
    )
    assert report["overall_pass"] is False  # build_report already reflects the failure
    report["overall_pass"] = True  # forge a green verdict over a red clause set
    assert any(
        "overall_pass" in p for p in sp.validate_report(report, selected_accessions=selected)
    )


def test_removal_rate_is_resummed_not_readback():
    segs, selected = _healthy()
    report = sp.build_report(
        segs,
        n_shards_expected=2,
        selected_accessions=selected,
        prior_genome_windows=None,
        score_threshold=THRESH,
        substrate_scanned=True,
        accessed="2026-07-23",
    )
    report["n_production_windows"] = report["n_production_windows"] + 100  # a fabricated headline
    assert any(
        "n_production_windows" in p
        for p in sp.validate_report(report, selected_accessions=selected)
    )


def test_empty_segments_certifies_nothing():
    assert sp.derive_clauses(
        [], n_shards_expected=0, selected_accessions=[], prior_genome_windows=None
    ) == {k: False for k in sp.REQUIRED_CLAUSES}
    report = {
        "schema_version": sp.SCHEMA_VERSION,
        "clause_schema_version": sp.CLAUSE_SCHEMA_VERSION,
        "step": sp.STEP,
        "segments": [],
    }
    assert any(
        "segments missing or empty" in p for p in sp.validate_report(report, selected_accessions=[])
    )


def test_pinned_cm_sha256_matches_committed_cm():
    # the CM-identity anchor must equal the sha256 recorded in the tier2n probe report;
    # anchor the path to the repo root (not the process CWD) — CodeRabbit PR #76
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    probe = json.loads((repo / "reports/p2/tier2n_probe.json").read_text())
    assert probe["cm_sha256"] == sp.RF00230_CM_SHA256


def test_reduce_over_segments_certifies_and_refuses_broken(tmp_path):
    # the cluster reduce entry: per-shard segment JSONs → certified report on disk; a broken
    # segment raises rather than leaving a healthy-looking artifact (§10.3).
    import json

    segs, selected = _healthy()
    paths = []
    for s in segs:
        p = tmp_path / f"seg_{s['shard']}.json"
        p.write_text(json.dumps(s))
        paths.append(p)
    out = tmp_path / "report.json"
    report = sp.reduce(
        paths,
        n_shards_expected=2,
        selected_accessions=selected,
        prior_genome_windows=None,
        score_threshold=THRESH,
        accessed="2026-07-23",
        out_report=out,
        out_provenance=tmp_path / "report.provenance.json",
    )
    assert report["overall_pass"] is True and out.exists()
    # a wrong CM in one segment must make reduce REFUSE to write a certified report
    segs[0]["cm_sha256"] = "0" * 64
    (tmp_path / "seg_0.json").write_text(json.dumps(segs[0]))
    with pytest.raises(sp.SubstratePrescanError):
        sp.reduce(
            paths,
            n_shards_expected=2,
            selected_accessions=selected,
            prior_genome_windows=None,
            score_threshold=THRESH,
            accessed="2026-07-23",
            out_report=tmp_path / "report2.json",
            out_provenance=tmp_path / "report2.provenance.json",
        )


def test_score_threshold_is_keyword_required_no_default():
    # §10.3: no numeric detection threshold is pinned; the recall-favouring cutoff is supplied
    with pytest.raises(TypeError):
        sp.build_shard_segment(  # type: ignore[call-arg]
            0,
            arm_windows={"production": {}, "spike": {}, "null": {}},
            hits=[],
            tblout_text="x",
            cm_sha256=sp.RF00230_CM_SHA256,
            expected_production_windows=0,
            shard_ok=True,
            genome_windows={},
        )
