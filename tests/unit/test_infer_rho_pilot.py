"""Unit tests for :mod:`tbox_finder.infer.rho_pilot` — the ρ-pilot scan driver's pure core.

The GPU legs (:func:`~tbox_finder.infer.rho_pilot.scan_genome` /
:func:`~tbox_finder.infer.rho_pilot.scan_shard`) need CUDA (ADR-0002 A2 C2) and run on the
cluster; what is tested here is the ``numpy``/stdlib aggregation + reporting surface that
turns per-contig sweep rows into the certified ρ surface. Because
:func:`tbox_finder.infer.call.sweep_candidate_counts` is itself ``numpy``-only, these tests
drive the **real caller → real reduction** chain end-to-end (a synthetic ``log_probs`` stands
in for the GPU forward), so they pin the arithmetic the sbatch deploys, not a mock of it.

Discipline (mirrors ``test_infer_call.py``):
* deterministic fixtures via :mod:`hashlib` ``shake_256`` (numpy RNG streams are not
  release-stable and these back exact-equality assertions);
* every reduction/derivation invariant carries an **independent reference with a flippable
  knob** whose flipped form is shown to diverge on a crafted fixture — a test no wrong
  implementation can fail is a tautology ([[tests-can-specify-the-bug]]);
* anti-degeneracy guards on the fixtures ([[degenerate-fixture-generators]]);
* ``validate_report`` clauses are shown to **fire** under a targeted mutation, not merely to
  pass on a good report ([[gate-clauses-need-re-derivation]],
  [[control-matchedness-must-be-asserted]]).
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from tbox_finder.infer import rho_pilot as rp
from tbox_finder.infer.call import BACKGROUND_INDEX, NUM_CLASSES, sweep_candidate_counts

# Small non-degenerate grids (8 points) — enough that the surface varies across cells.
THRESHOLDS = (0.5, 0.9)
MIN_SPANS = (20, 50)
GAP_MERGES = (0, 10)
GRID = rp.grid_order(THRESHOLDS, MIN_SPANS, GAP_MERGES)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fixtures (shake_256, NOT numpy RNG)
# ─────────────────────────────────────────────────────────────────────────────
def _rand_p_elem(seed: str, length: int) -> np.ndarray:
    """A deterministic ``[0, 1)`` element-score vector with real structure (some runs high)."""
    raw = hashlib.shake_256(seed.encode()).digest(length)
    base = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 255.0
    # Bias a middle stretch upward so a permissive threshold sees genuine candidate runs.
    lo, hi = length // 4, length // 2
    base[lo:hi] = 0.6 + 0.4 * base[lo:hi]
    return base


def _log_probs_from_pelem(p_elem: np.ndarray, dominant: int = 1) -> np.ndarray:
    """``(len, 8)`` log-posterior with ``P(background) = 1 − p_elem`` and the rest on one class."""
    p = np.clip(np.asarray(p_elem, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    probs = np.full((p.shape[0], NUM_CLASSES), 1e-12)
    probs[:, BACKGROUND_INDEX] = 1.0 - p
    probs[:, dominant] = p
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


def _fake_genome(accession: str, contig_lengths, *, seed: str, zero_flank_tail: bool = False):
    """A torch-free stand-in for :func:`scan_genome` — real sweep, same output shape.

    Builds a synthetic ``log_probs`` per contig, runs the **real**
    :func:`sweep_candidate_counts`, and accumulates exactly as ``scan_genome`` does, so the
    per-genome dict is identical in shape and in arithmetic to the GPU path's output.
    """
    counts = {f: [0] * len(GRID) for f in rp.COUNT_FIELDS}
    total_bp = 0
    max_p_elem = 0.0
    for ci, length in enumerate(contig_lengths):
        p_elem = _rand_p_elem(f"{seed}:{accession}:{ci}", length)
        lp = _log_probs_from_pelem(p_elem)
        zf = np.zeros(length, dtype=bool)
        if zero_flank_tail:
            zf[-max(1, length // 8) :] = True
        rows = sweep_candidate_counts(
            lp, zf, thresholds=THRESHOLDS, min_spans=MIN_SPANS, gap_merges=GAP_MERGES
        )
        arrays = rp.rows_to_count_arrays(rows, GRID)
        for f in rp.COUNT_FIELDS:
            counts[f] = [a + b for a, b in zip(counts[f], arrays[f], strict=True)]
        total_bp += length
        max_p_elem = max(max_p_elem, float(rp.element_posterior(lp).max()))
    return {
        "assembly_accession": accession,
        "total_bp": total_bp,
        "n_contigs": len(contig_lengths),
        "n_windows": sum(max(1, math.ceil(max(0, L - 1024) / 512) + 1) for L in contig_lengths),
        "scan_seconds": 1.0 + 0.001 * total_bp,
        "max_p_elem": max_p_elem,
        **counts,
    }


def _fixture_report():
    """Three fake genomes across two shards → a valid, non-degenerate report."""
    genomes = [
        _fake_genome("GCA_000000001.1", [1200, 400], seed="s", zero_flank_tail=True),
        _fake_genome("GCA_000000002.1", [2000, 1500, 300], seed="s"),
        _fake_genome("GCF_000000003.1", [900], seed="s"),
    ]
    manifest_ok = [g["assembly_accession"] for g in genomes]
    manifest_total_bp = sum(g["total_bp"] for g in genomes)
    shard_meta = [
        {
            "shard": 0,
            "n_shards": 2,
            "device": "cuda:0",
            "n_genomes": 2,
            "n_windows": genomes[0]["n_windows"] + genomes[1]["n_windows"],
            "scan_seconds": genomes[0]["scan_seconds"] + genomes[1]["scan_seconds"],
            "shard_wall_seconds": 99.0,
        },
        {
            "shard": 1,
            "n_shards": 2,
            "device": "cuda:1",
            "n_genomes": 1,
            "n_windows": genomes[2]["n_windows"],
            "scan_seconds": genomes[2]["scan_seconds"],
            "shard_wall_seconds": 50.0,
        },
    ]
    report = rp.build_report(
        genomes,
        shard_meta,
        thresholds=THRESHOLDS,
        min_spans=MIN_SPANS,
        gap_merges=GAP_MERGES,
        manifest_total_bp=manifest_total_bp,
        manifest_total_mbp=manifest_total_bp / 1e6,
        manifest_ok=manifest_ok,
        checkpoint_sha256="0" * 64,
        accessed="2026-07-22",
    )
    return report, genomes, shard_meta


# ─────────────────────────────────────────────────────────────────────────────
# FASTA
# ─────────────────────────────────────────────────────────────────────────────
def test_parse_fasta_multicontig_and_uppercases():
    text = ">c1 desc here\nacGT\nNnnn\n>c2\nGGCC\n"
    contigs = rp.parse_fasta(text)
    assert contigs == [("c1", "ACGTNNNN"), ("c2", "GGCC")]  # id is first token; seq upper-cased


def test_parse_fasta_uppercasing_is_load_bearing():
    # A soft-masked (lowercase) contig MUST come back uppercase, else encode_bases maps it to N.
    ((cid, seq),) = rp.parse_fasta(">x\nacgtacgt\n")
    assert seq == "ACGTACGT"
    assert seq != "acgtacgt"  # MUST FIRE — the .upper() is the whole point


def test_parse_fasta_drops_empty_records():
    assert rp.parse_fasta(">empty\n>real\nAC\n") == [("real", "AC")]


# ─────────────────────────────────────────────────────────────────────────────
# assign_shards — partition, determinism, balance (+ sabotage partner)
# ─────────────────────────────────────────────────────────────────────────────
def _sized(n):
    return [
        (f"g{i}", int.from_bytes(hashlib.shake_256(f"sz{i}".encode()).digest(3), "big"))
        for i in range(n)
    ]


def test_assign_shards_is_a_partition():
    sized = _sized(37)
    shards = rp.assign_shards(sized, 4)
    flat = [a for s in shards for a in s]
    assert sorted(flat) == sorted(a for a, _ in sized)  # every genome exactly once
    assert len(flat) == len(set(flat)) == 37  # none duplicated


def test_assign_shards_is_deterministic():
    sized = _sized(20)
    assert rp.assign_shards(sized, 3) == rp.assign_shards(list(reversed(sized)), 3)


def _ref_assign(sized, n_shards, *, to_lightest: bool):
    """Reference LPT with a flippable target: lightest (correct) vs heaviest (the sabotage)."""
    ordered = sorted(sized, key=lambda it: (-int(it[1]), str(it[0])))
    loads = [0] * n_shards
    shards = [[] for _ in range(n_shards)]
    for acc, size in ordered:
        pick = (min if to_lightest else max)(range(n_shards), key=lambda s: (loads[s], s))
        shards[pick].append(acc)
        loads[pick] += int(size)
    return shards, loads


def test_assign_shards_balances_and_the_heaviest_target_does_not():
    sized = _sized(40)
    got = rp.assign_shards(sized, 4)
    ref_light, loads_light = _ref_assign(sized, 4, to_lightest=True)
    ref_heavy, loads_heavy = _ref_assign(sized, 4, to_lightest=False)
    assert got == ref_light  # matches the lightest-bin rule
    spread_light = max(loads_light) - min(loads_light)
    spread_heavy = max(loads_heavy) - min(loads_heavy)
    assert spread_light < spread_heavy  # MUST FIRE — assigning to the heaviest bin unbalances


def test_assign_shards_rejects_bad_args():
    with pytest.raises(rp.RhoPilotError):
        rp.assign_shards([], 2)
    with pytest.raises(rp.RhoPilotError):
        rp.assign_shards(_sized(3), 0)


# ─────────────────────────────────────────────────────────────────────────────
# grid_order + rows_to_count_arrays — alignment to the real sweep, drift is fatal
# ─────────────────────────────────────────────────────────────────────────────
def test_grid_order_matches_sweep_row_order():
    lp = _log_probs_from_pelem(_rand_p_elem("g", 1500))
    rows = sweep_candidate_counts(
        lp, thresholds=THRESHOLDS, min_spans=MIN_SPANS, gap_merges=GAP_MERGES
    )
    sweep_points = [(float(r["threshold"]), int(r["min_span"]), int(r["gap_merge"])) for r in rows]
    assert sweep_points == GRID  # the canonical order IS the sweep's emission order


def test_rows_to_count_arrays_detects_order_drift():
    lp = _log_probs_from_pelem(_rand_p_elem("d", 1500))
    rows = sweep_candidate_counts(
        lp, thresholds=THRESHOLDS, min_spans=MIN_SPANS, gap_merges=GAP_MERGES
    )
    rp.rows_to_count_arrays(rows, GRID)  # aligned → fine
    shuffled = [rows[i] for i in [1, 0, 2, 3, 4, 5, 6, 7]]
    with pytest.raises(rp.RhoPilotError):  # MUST FIRE — a transposed sweep must not silently align
        rp.rows_to_count_arrays(shuffled, GRID)


# ─────────────────────────────────────────────────────────────────────────────
# sum_arrays — the reduction, pinned against a hand oracle (+ sum-vs-max sabotage)
# ─────────────────────────────────────────────────────────────────────────────
def test_sum_arrays_is_elementwise_sum_not_max():
    a = [1, 5, 2]
    b = [4, 0, 3]
    c = [2, 2, 2]
    assert rp.sum_arrays([a, b, c]) == [7, 7, 7]  # hand-summed oracle
    ref_max = [max(col) for col in zip(a, b, c, strict=True)]
    assert rp.sum_arrays([a, b, c]) != ref_max  # MUST FIRE — a max-reduce diverges on this fixture


def test_sum_arrays_rejects_ragged_and_empty():
    with pytest.raises(rp.RhoPilotError):
        rp.sum_arrays([])
    with pytest.raises(rp.RhoPilotError):
        rp.sum_arrays([[1, 2], [1, 2, 3]])


# ─────────────────────────────────────────────────────────────────────────────
# rho_surface — ρ = candidates / T[Mbp] (+ the /bp vs /Mbp must-fire)
# ─────────────────────────────────────────────────────────────────────────────
def test_rho_surface_divides_by_mbp_not_bp():
    counts = {
        "n_candidates": [10] * len(GRID),
        "n_zero_flanked_candidates": [2] * len(GRID),
        "total_candidate_nt": [1000] * len(GRID),
    }
    total_bp = 2_000_000
    surface = rp.rho_surface(counts, GRID, total_bp / 1e6)
    assert surface[0]["rho_per_mbp"] == pytest.approx(10 / 2.0)  # 10 candidates / 2 Mbp = 5.0
    assert surface[0]["rho_per_mbp"] != pytest.approx(10 / total_bp)  # MUST FIRE — /bp is 1e6× off
    # zero-flanked-excluded ρ uses (n − n_zf):
    assert surface[0]["rho_excl_zero_flanked_per_mbp"] == pytest.approx((10 - 2) / 2.0)


def test_rho_surface_rejects_nonpositive_total():
    with pytest.raises(rp.RhoPilotError):
        rp.rho_surface({f: [0] for f in rp.COUNT_FIELDS}, GRID[:1], 0.0)


def test_permissive_corner_is_min_tau_min_span_max_gap():
    idx = rp.permissive_corner_index(GRID)
    assert GRID[idx] == (0.5, 20, 10)


# ─────────────────────────────────────────────────────────────────────────────
# per_genome_digest — order-independent (naive records_digest is not)
# ─────────────────────────────────────────────────────────────────────────────
def test_per_genome_digest_is_order_independent():
    _, genomes, _ = _fixture_report()
    assert rp.per_genome_digest(genomes) == rp.per_genome_digest(list(reversed(genomes)))


def test_per_genome_digest_changes_when_a_count_changes():
    _, genomes, _ = _fixture_report()
    d0 = rp.per_genome_digest(genomes)
    bumped = [dict(g) for g in genomes]
    bumped[0] = dict(bumped[0])
    bumped[0]["n_candidates"] = [c + 1 for c in bumped[0]["n_candidates"]]
    assert rp.per_genome_digest(bumped) != d0  # MUST FIRE — evidence is folded into the digest


# ─────────────────────────────────────────────────────────────────────────────
# Fixture sanity — non-degenerate (else the derivation tests are vacuous)
# ─────────────────────────────────────────────────────────────────────────────
def test_fixture_is_expressive_not_degenerate():
    report, _, _ = _fixture_report()
    surface = report["rho_surface"]
    counts = [r["n_candidates"] for r in surface]
    assert max(counts) > 0
    assert len(set(counts)) > 1  # the surface actually varies across grid points
    assert report["power"]["permissive_corner"]["n_candidates"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# build_report → validate_report — a good report is clean; each clause must fire
# ─────────────────────────────────────────────────────────────────────────────
def test_valid_report_passes_validation():
    report, _, _ = _fixture_report()
    assert rp.validate_report(report) == []


def test_scanned_bp_must_equal_manifest_total():
    report, _, _ = _fixture_report()
    report["scanned_bp"] = report["scanned_bp"] + 1
    problems = rp.validate_report(report)
    assert any("scanned_bp" in p for p in problems)  # MUST FIRE — coverage/no-drop guard


def test_dropping_a_genome_fails_coverage():
    report, _, _ = _fixture_report()
    report["per_genome"] = report["per_genome"][:-1]
    problems = rp.validate_report(report)
    assert any("manifest" in p or "n_genomes" in p for p in problems)  # MUST FIRE


def test_corrupted_surface_count_fails_rederivation():
    report, _, _ = _fixture_report()
    report["rho_surface"][0]["n_candidates"] += 5
    problems = rp.validate_report(report)
    assert any("n_candidates" in p and "re-summed" in p for p in problems)  # MUST FIRE


def test_corrupted_rho_value_fails_rederivation():
    report, _, _ = _fixture_report()
    report["rho_surface"][0]["rho_per_mbp"] *= 2.0
    problems = rp.validate_report(report)
    assert any("rho_per_mbp" in p for p in problems)  # MUST FIRE


def test_dead_forward_fails_the_power_floor():
    report, genomes, shard_meta = _fixture_report()
    # A checkpoint that produced no element signal: every position ~background.
    for g in report["per_genome"]:
        g["max_p_elem"] = 0.1
    report["power"]["global_max_p_elem"] = 0.1
    problems = rp.validate_report(report)
    assert any("global_max_p_elem" in p and "floor" in p for p in problems)  # MUST FIRE


def test_empty_permissive_corner_fails():
    # Build genomes whose permissive-corner (k=index) count is 0 but signal is otherwise live.
    report, _, _ = _fixture_report()
    corner = rp.permissive_corner_index(GRID)
    for g in report["per_genome"]:
        g["n_candidates"][corner] = 0
        g["n_zero_flanked_candidates"][corner] = 0
    # keep the surface + digest consistent so ONLY the corner clause is isolated
    report["rho_surface"][corner]["n_candidates"] = 0
    report["rho_surface"][corner]["rho_per_mbp"] = 0.0
    report["rho_surface"][corner]["n_zero_flanked_candidates"] = 0
    report["rho_surface"][corner]["rho_excl_zero_flanked_per_mbp"] = 0.0
    report["digest"] = rp.per_genome_digest(report["per_genome"])
    problems = rp.validate_report(report)
    assert any("permissive corner" in p for p in problems)  # MUST FIRE


def test_honesty_flag_flip_fails():
    report, _, _ = _fixture_report()
    report["is_science"] = True
    assert any("is_science" in p for p in rp.validate_report(report))  # MUST FIRE


def test_corrupted_digest_fails():
    report, _, _ = _fixture_report()
    report["digest"] = "deadbeef"
    assert any("digest" in p for p in rp.validate_report(report))  # MUST FIRE


def test_zero_flanked_exceeding_total_fails():
    report, _, _ = _fixture_report()
    report["per_genome"][0]["n_zero_flanked_candidates"][0] = (
        report["per_genome"][0]["n_candidates"][0] + 1
    )
    assert any("zero-flanked" in p for p in rp.validate_report(report))  # MUST FIRE


# ─────────────────────────────────────────────────────────────────────────────
# load_partials — shape/coverage of the shard set
# ─────────────────────────────────────────────────────────────────────────────
def test_load_partials_requires_the_full_shard_range(tmp_path):
    _, genomes, shard_meta = _fixture_report()
    # write only shard 0 of a 2-shard run
    p0 = tmp_path / "shard_0.json"
    part0 = {
        "kind": "shard_partial",
        "step": rp.STEP,
        "shard": 0,
        "n_shards": 2,
        "device": "cuda:0",
        "n_genomes": 1,
        "n_windows": 1,
        "scan_seconds": 1.0,
        "shard_wall_seconds": 1.0,
        "grid": [list(p) for p in GRID],
        "per_genome": [genomes[0]],
    }
    p0.write_text(json.dumps(part0))
    with pytest.raises(rp.RhoPilotError):  # MUST FIRE — a missing shard must not reduce
        rp.load_partials([p0])


def test_load_partials_rejects_duplicate_shard(tmp_path):
    _, genomes, _ = _fixture_report()
    paths = []
    for i in range(2):
        p = tmp_path / f"dup_{i}.json"
        p.write_text(
            json.dumps(
                {
                    "kind": "shard_partial",
                    "step": rp.STEP,
                    "shard": 0,
                    "n_shards": 1,
                    "device": "cuda:0",
                    "n_genomes": 1,
                    "n_windows": 1,
                    "scan_seconds": 1.0,
                    "shard_wall_seconds": 1.0,
                    "grid": [list(g) for g in GRID],
                    "per_genome": [genomes[0]],
                }
            )
        )
        paths.append(p)
    with pytest.raises(rp.RhoPilotError):
        rp.load_partials(paths)


def test_reduce_end_to_end_writes_certified_report(tmp_path):
    """The full reduce path: partials + a manifest report on disk → a validated report file."""
    _, genomes, shard_meta = _fixture_report()
    manifest_ok = [g["assembly_accession"] for g in genomes]
    manifest_total_bp = sum(g["total_bp"] for g in genomes)
    # a minimal pilot_fetch-style manifest report
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "total_bp": manifest_total_bp,
                "total_mbp": manifest_total_bp / 1e6,
                "per_genome": [{"assembly_accession": a, "status": "ok"} for a in manifest_ok],
            }
        )
    )
    # a fake checkpoint file (reduce sha256s it, never loads it)
    ckpt = tmp_path / "stage1.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    # two shard partials covering all three genomes
    parts = []
    for i, gs in enumerate(([genomes[0], genomes[1]], [genomes[2]])):
        p = tmp_path / f"shard_{i}.json"
        p.write_text(
            json.dumps(
                {
                    "kind": "shard_partial",
                    "step": rp.STEP,
                    "shard": i,
                    "n_shards": 2,
                    "device": f"cuda:{i}",
                    "n_genomes": len(gs),
                    "n_windows": sum(g["n_windows"] for g in gs),
                    "scan_seconds": sum(g["scan_seconds"] for g in gs),
                    "shard_wall_seconds": 10.0,
                    "grid": [list(g) for g in GRID],
                    "per_genome": gs,
                }
            )
        )
        parts.append(p)
    out = tmp_path / "rho_pilot_report.json"
    prov = tmp_path / "rho_pilot_report.provenance.json"
    report = rp.reduce_partials(
        parts,
        manifest_report=manifest,
        checkpoint=ckpt,
        report_path=out,
        provenance_path=prov,
        accessed="2026-07-22",
    )
    assert out.is_file() and prov.is_file()
    on_disk = json.loads(out.read_text())
    assert rp.validate_report(on_disk) == []  # the written report certifies
    assert report["n_genomes"] == 3
    assert report["scanned_bp"] == manifest_total_bp
    assert report["checkpoint"]["sha256"] == hashlib.sha256(b"not-a-real-checkpoint").hexdigest()


def test_reduce_refuses_to_write_a_broken_report(tmp_path):
    """A manifest that disagrees on total_bp must abort reduce with nothing written."""
    _, genomes, _ = _fixture_report()
    manifest_ok = [g["assembly_accession"] for g in genomes]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "total_bp": sum(g["total_bp"] for g in genomes) + 999,  # WRONG on purpose
                "total_mbp": 1.0,
                "per_genome": [{"assembly_accession": a, "status": "ok"} for a in manifest_ok],
            }
        )
    )
    ckpt = tmp_path / "stage1.pt"
    ckpt.write_bytes(b"x")
    p = tmp_path / "shard_0.json"
    p.write_text(
        json.dumps(
            {
                "kind": "shard_partial",
                "step": rp.STEP,
                "shard": 0,
                "n_shards": 1,
                "device": "cuda:0",
                "n_genomes": 3,
                "n_windows": 9,
                "scan_seconds": 3.0,
                "shard_wall_seconds": 3.0,
                "grid": [list(g) for g in GRID],
                "per_genome": genomes,
            }
        )
    )
    out = tmp_path / "rho_pilot_report.json"
    with pytest.raises(rp.RhoPilotError):  # MUST FIRE — fail-closed
        rp.reduce_partials(
            [p],
            manifest_report=manifest,
            checkpoint=ckpt,
            report_path=out,
            provenance_path=tmp_path / "prov.json",
            accessed="2026-07-22",
        )
    assert not out.exists()  # nothing certified
