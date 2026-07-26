"""Unit tests for the production tiling/shard-spec emitter (ADR-0005 A8, P2-10c'-shardspec).

The emitter PRODUCES what :mod:`tbox_finder.mining.substrate_prescan` (the gate) consumes, so the
load-bearing invariants are shared with the gate: the tiling geometry equals the gate's frozen
geometry; a shard spec satisfies :func:`substrate_prescan.build_shard_segment`'s
``Σ genome_windows == len(production) == expected_production_windows``; the union of shards equals
the git-frozen selection (clause viii ``set_ok``); window names are unique + whitespace-free
(clause v); and the window-count report is a valid clause-(viii) shrink baseline. Each invariant
has a MUST-FIRE partner — a malformed input is asserted to raise / flip, not merely a healthy path
(the [[sabotage-attribution-names-the-test]] discipline). Bare-CI tier: stdlib + numpy (via the
canonical tiler) for the pure-geometry tests; the manifest round-trips use pandas/pyarrow.
"""

from __future__ import annotations

import json

import pytest

from tbox_finder.data.window_dataset import tile_windows
from tbox_finder.mining import substrate_prescan as sp
from tbox_finder.mining import substrate_windows as sw


def _contig(n: int, salt: str = "") -> str:
    """A deterministic ``n``-nt ACGT contig (identity is irrelevant to window *geometry*)."""
    base = "ACGT" if not salt else "".join("ACGT"[(ord(salt[0]) + i) & 3] for i in range(4))
    return (base * (n // 4 + 1))[:n]


def _fasta(lengths, salt: str = "") -> str:
    return (
        "\n".join(
            f">ctg_{i} {salt}desc with spaces {i}\n{_contig(n, salt + str(i))}"
            for i, n in enumerate(lengths)
        )
        + "\n"
    )


def _write_genomes(root, mapping) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for acc, lengths in mapping.items():
        (root / f"{acc}.fna").write_text(_fasta(lengths, salt=acc))


# ── geometry drift guard ─────────────────────────────────────────────────────
def test_geometry_matches_gate() -> None:
    assert sw._geometry_is_consistent()
    assert (sw.WINDOW_NT, sw.STRIDE_NT) == (sp.WINDOW_NT, sp.STRIDE_NT) == (1024, 512)


# ── tiling reuses the canonical scan tiler (not a re-implementation) ──────────
def test_tiler_reuses_canonical_offsets() -> None:
    for n in (300, 1024, 1025, 1536, 2048, 3072, 5000):
        starts = [
            int(nm.rsplit(":", 1)[1]) for nm, _ in sw.iter_genome_windows("GCA_X.1", _fasta([n]))
        ]
        assert starts == tile_windows(n, window=1024, stride=512)


@pytest.mark.parametrize(
    ("n", "exp_windows"),
    [(500, 1), (1023, 1), (1024, 1), (1025, 2), (1536, 2), (2048, 3), (3072, 5)],
)
def test_short_and_boundary_contigs(n: int, exp_windows: int) -> None:
    nc, bp, nw = sw.count_genome_windows(_fasta([n]))
    assert (nc, bp, nw) == (1, n, exp_windows)
    # a contig <= window emits its own (un-padded) sequence as the single window
    if n <= 1024:
        ((_name, seq),) = list(sw.iter_genome_windows("GCA_X.1", _fasta([n])))
        assert len(seq) == n


def test_empty_contig_skipped() -> None:
    txt = ">a keeps\n\n>b\n" + _contig(1200) + "\n"  # a is empty; b -> 2 windows
    nc, bp, nw = sw.count_genome_windows(txt)
    assert (nc, bp, nw) == (1, 1200, 2)


# ── clause (v): unique, whitespace-free names ────────────────────────────────
def test_window_names_unique_and_whitespace_free(tmp_path) -> None:
    from tbox_finder import infernal

    wins = dict(sw.iter_genome_windows("GCA_Y.2", _fasta([1536, 500, 2048])))
    names = list(wins)
    assert len(names) == len(set(names)) == 2 + 1 + 3
    assert all(not any(c.isspace() for c in nm) for nm in names)
    infernal.write_fasta(wins, tmp_path / "w.fna")  # raises on a whitespace name; all names pass


def test_write_fasta_would_reject_a_whitespace_name(tmp_path) -> None:
    # MUST-FIRE proof that the name-safety check the emitter relies on is real.
    from tbox_finder import infernal

    with pytest.raises(ValueError):
        infernal.write_fasta({"bad name": "ACGT"}, tmp_path / "bad.fna")


# ── the build_shard_segment invariant (Σ genome_windows == len(production)) ───
def test_shard_spec_satisfies_gate_invariant(tmp_path) -> None:
    gdir = tmp_path / "g"
    _write_genomes(gdir, {"GCA_A.1": [1025, 500], "GCA_B.1": [2048]})  # 3 + 3
    spec = sw.build_shard_spec(
        0,
        ["GCA_A.1", "GCA_B.1"],
        genome_dir=gdir,
        expected_by_accession={"GCA_A.1": 3, "GCA_B.1": 3},
    )
    assert set(spec) == {"shard", "expected_production_windows", "genome_windows", "production"}
    assert spec["expected_production_windows"] == 6 == len(spec["production"])
    assert sum(spec["genome_windows"].values()) == 6
    # the gate's own consumer accepts it
    p = tmp_path / "shard_0.json"
    p.write_text(json.dumps(spec))
    assert set(sp.read_shard_spec(p)) >= set(spec)
    seg = sp.build_shard_segment(
        0,
        arm_windows={
            "production": spec["production"],
            "spike": {f"s{i}": "ACGT" * 30 for i in range(sp.MIN_SPIKE_N)},
            "null": {f"n{i}": "TGCA" * 30 for i in range(sp.MIN_NULL_N)},
        },
        hits=[],
        tblout_text="",
        cm_sha256=sp.RF00230_CM_SHA256,
        expected_production_windows=spec["expected_production_windows"],
        score_threshold=30.0,
        shard_ok=True,
        genome_windows=spec["genome_windows"],
    )
    assert seg["expected_production_windows"] == 6


def test_gate_rejects_mismatched_genome_windows() -> None:
    # MUST-FIRE: a genome_windows total that disagrees with len(production) is refused by the gate,
    # so the emitter's construction (which guarantees agreement) is load-bearing, not cosmetic.
    with pytest.raises(sp.SubstratePrescanError):
        sp.build_shard_segment(
            0,
            arm_windows={
                "production": {"w0": "ACGT" * 20, "w1": "ACGT" * 20},
                "spike": {f"s{i}": "ACGT" * 30 for i in range(sp.MIN_SPIKE_N)},
                "null": {f"n{i}": "TGCA" * 30 for i in range(sp.MIN_NULL_N)},
            },
            hits=[],
            tblout_text="",
            cm_sha256=sp.RF00230_CM_SHA256,
            expected_production_windows=2,
            score_threshold=30.0,
            shard_ok=True,
            genome_windows={"GCA_A.1": 1},  # claims 1 window, production has 2
        )


def test_expected_mismatch_fails_loud(tmp_path) -> None:
    # MUST-FIRE: expected (frozen manifest) != live-tiled production -> the FASTA drifted; refuse.
    gdir = tmp_path / "g"
    _write_genomes(gdir, {"GCA_A.1": [2048]})  # 3 windows
    with pytest.raises(sw.SubstrateWindowsError):
        sw.build_shard_spec(0, ["GCA_A.1"], genome_dir=gdir, expected_by_accession={"GCA_A.1": 99})


def test_missing_genome_fails_loud(tmp_path) -> None:
    # MUST-FIRE: a selected genome with no FASTA is a clause-(viii) silent-shrink, not a skip.
    (tmp_path / "g").mkdir()
    with pytest.raises(sw.SubstrateWindowsError):
        sw.read_genome_fasta(tmp_path / "g", "GCA_MISSING.9")


# ── manifest: re-derivation + clause-(viii) baseline shape ───────────────────
def test_manifest_rows_rederived_not_readback(tmp_path) -> None:
    gdir = tmp_path / "g"
    genomes = {"GCA_A.1": [1025, 500], "GCA_B.1": [2048, 1536], "GCA_C.1": [700]}
    _write_genomes(gdir, genomes)
    rows = sw.build_window_rows(gdir, list(genomes))
    assert [r["assembly_accession"] for r in rows] == sorted(genomes)  # sorted
    by = {r["assembly_accession"]: r for r in rows}
    for acc, lengths in genomes.items():
        exp = sum(len(tile_windows(n, window=1024, stride=512)) for n in lengths)
        assert by[acc]["n_windows"] == exp
        assert by[acc]["total_bp"] == sum(lengths)
        assert by[acc]["n_contigs"] == len(lengths)


def test_report_is_valid_clause_viii_baseline(tmp_path) -> None:
    rows = [
        {"assembly_accession": "GCA_A.1", "n_contigs": 2, "total_bp": 1525, "n_windows": 3},
        {"assembly_accession": "GCA_B.1", "n_contigs": 1, "total_bp": 2048, "n_windows": 3},
    ]
    report = sw.build_window_report(rows)
    assert report["expected_production_windows"] == 6
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    base = sp.prior_genome_windows_from_fetch_report(p)  # the gate's own reader
    assert base == {"GCA_A.1": 3, "GCA_B.1": 3}


def test_report_without_n_windows_is_rejected_by_gate(tmp_path) -> None:
    # MUST-FIRE: n_windows is the load-bearing field; a report lacking it is not a valid baseline.
    bad = {"per_genome": [{"assembly_accession": "GCA_A.1", "total_bp": 1525}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(sp.SubstratePrescanError):
        sp.prior_genome_windows_from_fetch_report(p)


def test_build_window_report_refuses_empty() -> None:
    with pytest.raises(sw.SubstrateWindowsError):
        sw.build_window_report([])


# ── emit-specs: LPT partition completeness + determinism + gate round-trip ───
def _tiny_corpus(tmp_path):
    import pandas as pd

    gdir = tmp_path / "g"
    genomes = {"GCA_A.1": [2048, 500], "GCA_B.1": [3072], "GCA_C.1": [1025, 1536], "GCA_D.1": [700]}
    _write_genomes(gdir, genomes)
    rows = sw.build_window_rows(gdir, list(genomes))
    wman = tmp_path / "windows.parquet"
    pd.DataFrame(rows).to_parquet(wman, index=False)
    sel = tmp_path / "selection.parquet"
    pd.DataFrame({"assembly_accession": sorted(genomes)}).to_parquet(sel, index=False)
    return gdir, wman, sel, genomes


def test_emit_specs_partition_complete_and_roundtrips(tmp_path) -> None:
    gdir, wman, sel, genomes = _tiny_corpus(tmp_path)
    out = tmp_path / "specs"
    paths = sw.emit_shard_specs(
        genome_dir=gdir, window_manifest=wman, selection_manifest=sel, n_shards=3, out_dir=out
    )
    assert len(paths) == 3
    scanned = set()
    total = 0
    for p in paths:
        obj = sp.read_shard_spec(p)  # 4-key contract
        assert (
            obj["expected_production_windows"]
            == len(obj["production"])
            == sum(obj["genome_windows"].values())
        )
        for acc in obj["genome_windows"]:
            assert acc not in scanned  # each genome in exactly one shard
            scanned.add(acc)
        total += obj["expected_production_windows"]
    assert scanned == set(genomes)  # clause (viii) set_ok: union == selection
    # global window total is conserved across the partition
    all_windows = sum(
        len(tile_windows(n, window=1024, stride=512)) for lens in genomes.values() for n in lens
    )
    assert total == all_windows


def test_emit_specs_is_deterministic(tmp_path) -> None:
    gdir, wman, sel, _ = _tiny_corpus(tmp_path)
    a = tmp_path / "a"
    b = tmp_path / "b"
    pa = sw.emit_shard_specs(
        genome_dir=gdir, window_manifest=wman, selection_manifest=sel, n_shards=2, out_dir=a
    )
    pb = sw.emit_shard_specs(
        genome_dir=gdir, window_manifest=wman, selection_manifest=sel, n_shards=2, out_dir=b
    )
    assert [json.loads(p.read_text()) for p in pa] == [json.loads(p.read_text()) for p in pb]


def test_emit_specs_rejects_selection_absent_from_manifest(tmp_path) -> None:
    # MUST-FIRE: a selected genome absent from the window manifest -> refuse the emit.
    import pandas as pd

    gdir, wman, _sel, genomes = _tiny_corpus(tmp_path)
    sel2 = tmp_path / "sel2.parquet"
    pd.DataFrame({"assembly_accession": sorted(genomes) + ["GCA_UNSEEN.9"]}).to_parquet(
        sel2, index=False
    )
    with pytest.raises(sw.SubstrateWindowsError):
        sw.emit_shard_specs(
            genome_dir=gdir,
            window_manifest=wman,
            selection_manifest=sel2,
            n_shards=2,
            out_dir=tmp_path / "o",
        )
