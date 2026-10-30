"""P3-16″-hardneg — the ADR-0005 A17 scoring side: the hard-negative mint, the Pin 4 power
gate, and the schema-1.3 precision report.

The committed A17 artifacts (the refused census report, its τ annexes and the two pool reports)
are run through the validators in their own section. Every other case is built through the REAL
producers on a tmp tree: ``scripts/mint_p3_16_benchmark.py``'s ``--hardneg`` functions mint the
manifest from a pool report that passes ``hardneg_pool.validate_report`` (the pool's own
validator, not a stub), ``precision power`` counts P from the payload manifest, ``precision fold``
binds every input, and ``precision_gain`` grades it. Window DNA is cut from
``tests/fixtures/two_stage/contigs.json`` (20 real 1,536-nt ``context_v0`` slabs); the per-window
metadata and the scores are synthetic, because what is under test is the bookkeeping, not the
models. The GATE-2 temperature source is the committed report, so the Stage-2 identity binds to
real digests.

Each 1.3 clause is broken ALONE on a deep copy with every other field consistent
([[all-true-fixture-cannot-test-a-conjunction]]), and the committed v1 / v2 reports are run
through the validators to show the 1.1 / 1.2 clause sets are untouched.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

from tbox_finder.data import window_dataset as wd
from tbox_finder.integration import host_screen as hs
from tbox_finder.integration import precision as P
from tbox_finder.integration import two_stage as TS

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mint_p3_16_benchmark.py"
SLABS = REPO / "tests" / "fixtures" / "two_stage" / "contigs.json"
GATE2_REL = "reports/gate2_p3_ece.json"
P3_16 = "data/interim/p3_16"
POOL_REL = "reports/p3/hardneg_pool.json"
POOL_CENSUS_REL = "reports/p3/hardneg_pool_census.json"
V2_BENCH_REL = f"{P3_16}/benchmark_v2.json"
V2_ITEMS_REL = "reports/p3/two_stage_precision_items_v2.json"
ITEMS_REL = "reports/p3/two_stage_precision_items_hardneg.json"
OUT_REL = "reports/two_stage_precision_hardneg.json"
ROUND0_REL = "data/processed/mining/round0_fp_manifest.json"
SPARE_REL = "data/processed/mining/round0_fp_spare_rule_inputs_v0.json"
GENOMES_REL = "data/interim/production_genomes"
KNOBS = "threshold=0.5,min_span=50,gap_merge=10,min_distinct_elements=2,flank=50,min_order_margin=1"
N_BOOT = 40  # the smallest budget the interval clause accepts at a 95 % level
ASSEMBLIES = ("GCA_000000001.1", "GCA_000000002.1", "GCA_000000003.1")
ORDERS = {ASSEMBLIES[0]: "Bacillales", ASSEMBLIES[1]: "Lactobacillales", ASSEMBLIES[2]: None}
ARMS = ("production", "twin")


def _load_minter():
    spec = importlib.util.spec_from_file_location("mint_p3_16_hardneg_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MINT = _load_minter()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(root: Path, rel: str, payload) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _windows() -> list[str]:
    """100 real 1,024-nt windows: five phases of each of the 20 committed slabs."""
    slabs = json.loads(SLABS.read_text(encoding="utf-8"))["contigs"]
    return [
        slab["sequence"][offset : offset + 1024] for slab in slabs for offset in range(0, 513, 128)
    ]


WINDOWS = _windows()
POSITIVE_WINDOWS = WINDOWS[:24]  # slabs 0-4
NEGATIVE_WINDOWS = WINDOWS[25:]  # slabs 5-19: no slab is shared with a positive


def _row(contig_id: str, peak: float, posterior: float) -> dict:
    return {"contig_id": contig_id, "peak_p_elem": peak, "stage2_named_posterior": posterior}


def _native(k: int, *, role: str) -> dict:
    """The k-th native window, shaped as ``hardneg_pool``'s graded / beside item lists."""
    assembly = ASSEMBLIES[k % 3]
    seqid = "ctgA" if k % 2 == 0 else "ctgB"
    start = 10_000 + 2_000 * k
    strand = "+" if k % 2 == 0 else "-"
    feature_type = "CDS" if k % 5 else "tRNA"
    sequence = NEGATIVE_WINDOWS[k]
    return {
        "item_id": f"{assembly}:{seqid}:{start}-{start + 1024}:{strand}",
        "assembly": assembly,
        "seqid": seqid,
        "start": start,
        "end": start + 1024,
        "strand": strand,
        "feature_type": feature_type,
        "feature_id": f"{feature_type.lower()}-{k}",
        "delta": k % 257,
        "sub_class": MINT.HARDNEG_SUB_CLASS[feature_type],
        "d4_context": role == "beside",
        "sha256": _sha_bytes(sequence.encode()),
        "sequence": sequence,
    }


def _controls() -> list[dict]:
    names = [
        hs.CONTROL_HOST_CLEAN,
        hs.CONTROL_COORD_ONLY,
        hs.CONTROL_CONTENT,
        hs.CONTROL_CONTENT_HOST,
        *(f"pin1_kmer_{source}" for source in hs.SCREENED_SOURCES),
        "pin1_accession_versioned",
    ]
    return [hs.control_entry(name, "a must-fire mechanism", True, True) for name in names]


def _pool_report(mode: str, graded: list[dict], beside: list[dict], files: dict) -> dict:
    """A pool report the REAL ``hardneg_pool.validate_report`` accepts (counts are synthetic)."""
    n_eligible = 250_000
    n_selected = 100_000 if mode == "draw" else n_eligible - 100_000

    def by_sub(group):
        out: dict[str, int] = {}
        for item in group:
            out[item["sub_class"]] = out.get(item["sub_class"], 0) + 1
        return {**dict(sorted(out.items())), "total": len(group)}

    fold = n_selected - len(graded) - len(beside) - 30
    audit_ids = sorted(item["item_id"] for item in graded)
    residue = [{"item_id": audit_ids[0], "model": "RF00230", "bits": 11.0}]
    return {
        "schema_version": "1.0",
        "step": "P3-16″-hardneg",
        "kind": "hardneg_pool",
        "adr": "ADR-0005 A17",
        "prd": "§9.1 (class 4); §2.3",
        "config": {
            "seed": 42,
            "draw_size": 100_000,
            "pinned_draw": True,
            "mode": mode,
            "delta_max": 256,
            "window_nt": 1024,
            "feature_types": ["CDS", "tRNA"],
            "max_hosts": None,
        },
        "population": {
            "n_hosts": 3,
            "n_features": 300_000,
            "n_features_by_type": {"CDS": 290_000, "tRNA": 10_000},
            "n_eligible": n_eligible,
            "n_selected": n_selected,
        },
        "files": {
            name: {"path": f"{P3_16}/hardneg_pool/{name}", "sha256": char * 64}
            for name, char in (
                ("features.tsv", "a"),
                ("eligible.tsv", "b"),
                ("drawn.txt", "c"),
                ("items.json", "d"),
            )
        },
        "fasta": {
            name: {
                "path": f"{P3_16}/hardneg_pool/{name}",
                "sha256": "e" * 64,
                "threshold_bits": 10.0,
            }
            for name in ("survivors_00.fa", "controls.fa")
        },
        "partition_counts": {
            "refused_assembly_gap": {"cds_upstream": 10, "total": 10},
            "refused_fold": {"cds_upstream": fold, "total": fold},
            "refused_content": {"cds_upstream": 15, "trna_upstream": 5, "total": 20},
            "graded": by_sub(graded),
            "beside_d4_context": by_sub(beside),
        },
        "fold_refused_by_rule": {"kmer_and_coord": 3, "kmer_only": fold - 4, "coord_only": 1},
        "content_refused_by_rule": {"cm": 18, "motif": 2},
        "controls": _controls(),
        "controls_all_fired": True,
        "item_files": files,
        "ids_annex": {"path": "reports/p3/hardneg_pool_ids.tsv", "sha256": "f" * 64},
        "audit_ids": audit_ids,
        "audit": {
            "n_audited": len(audit_ids),
            "search_threshold_bits": 0.0,
            "threshold_bits": dict(hs.POSTMINT_THRESHOLD_BITS),
            "calibration": {"null_max_bits": {"RF00230": 16.8, "TBDB001": 11.9}},
            "thresholds_clear_floor": {"RF00230": True, "TBDB001": True},
            "rows_at_or_above_10_bits": residue,
            "noise_residue": residue,
            "n_violations": 0,
            "passes": True,
        },
        "selection": "ADR-0005 A17: native §9.1 class-4 leaders on the 339 annotated a2 hosts",
    }


def _write_pool(root: Path, mode: str, graded: list[dict], beside: list[dict], rel: str) -> dict:
    """Write one pool's item lists and report; returns what the minter is handed."""
    lists, files, digests = {}, {}, {}
    for role, group in (("graded", graded), ("beside", beside)):
        item_rel = f"{P3_16}/hardneg_pool_{mode}/{role}.json"
        path = _write(root, item_rel, {"items": group, "partition": role})
        raw = path.read_bytes()
        lists[role] = MINT.read_hardneg_items(raw, role=role, where=item_rel)
        digests[role] = (_sha_bytes(raw), len(group))
        files[role] = {"path": item_rel, "sha256": digests[role][0], "n": len(group)}
    report_path = _write(root, rel, _pool_report(mode, graded, beside, files))
    return {
        "mode": mode,
        "rel": rel,
        "lists": lists,
        "digests": digests,
        "report": json.loads(report_path.read_text()),
        "record": {
            "path": rel,
            "sha256": _sha_bytes(report_path.read_bytes()),
            "mode": mode,
            "item_files": files,
        },
    }


def _v2(root: Path) -> tuple[dict, dict]:
    """A v2-shaped benchmark (schema 1.1, gated subset) and its folded item table."""
    positives, contigs, folded = [], [], []
    for i, sequence in enumerate(POSITIVE_WINDOWS):
        cid = f"pos:{i:03d}"
        positives.append(
            {
                "contig_id": cid,
                "label": 1,
                "pool": "corpus",
                "block": f"cluster:{i // 3}",
                "seen_by": {"twin": False, "production": True},
                "gated": True,
                "source": "corpus",
            }
        )
        contigs.append(
            {"contig_id": cid, "sequence": sequence, "truth_start": 100, "truth_end": 400}
        )
        posterior = 0.90 + (i % 5) / 100
        folded.append(
            {
                "contig_id": cid,
                "label": 1,
                "arms": {
                    "twin": {
                        "n_rows": 1,
                        "stage1_only": 0.60 + i / 100,
                        # v2 scored a quarter of the positives slightly differently: Stage-2 logits
                        # depend on batch composition (A15 Pin 6), which the delta must surface.
                        "two_stage": posterior - (0.001 if i % 4 == 0 else 0.0),
                    },
                    "production": {
                        "n_rows": 1,
                        "stage1_only": 0.70 + i / 100,
                        "two_stage": posterior,
                    },
                },
            }
        )
    benchmark = {
        "schema_version": "1.1",
        "step": "P3-16",
        "scope": {
            "rung": "test",
            "geometry": "1024-nt scan windows (ADR-0005 D3 / PRD §6 pinned geometry)",
            "n_positives": len(positives),
            "n_items_gated": len(positives),
        },
        "items": positives,
        "contigs": contigs,
    }
    _write(root, V2_BENCH_REL, benchmark)
    v2_items = {
        "arms": sorted(ARMS),
        "items": folded,
        "source": {"rule_knobs": P.parse_rule_knobs(KNOBS), "stage1_threshold": 0.5},
    }
    _write(root, V2_ITEMS_REL, v2_items)
    return benchmark, v2_items


def _mint(root: Path, pools: list[dict], v2: dict, rel: str) -> Path:
    """The A17 manifest through the minter's own functions, with a measured fold overlap."""
    rng = random.Random(1)
    unrelated = "".join(rng.choice("ACGT") for _ in range(3000))
    folds = {"twin": {"ctx:twin"}, "production": {"ctx:prod"}}
    # Production's fold context carries the first negative's window verbatim, so the measured
    # overlap is > 0 there and 0 on the gated twin: the measurement is a measurement.
    context = {"ctx:twin": unrelated, "ctx:prod": unrelated + NEGATIVE_WINDOWS[0]}
    contigs, items, scope, positives_sha = MINT.build_hardneg(
        v2=v2,
        pools=[(pool["mode"], pool["lists"]) for pool in pools],
        resolved_order=ORDERS,
        folds=folds,
        context_by_id=context,
    )
    payload = MINT.hardneg_manifest(
        contigs=contigs,
        items=items,
        scope=scope,
        positives_sha256=positives_sha,
        pool_records=[pool["record"] for pool in pools],
        v2_path=V2_BENCH_REL,
        v2_sha256=_sha_bytes((root / V2_BENCH_REL).read_bytes()),
        inputs={"host_orders": "data/processed/mining/production_host_orders_v0.parquet"},
    )
    return _write(root, rel, payload)


def _round0(root: Path) -> None:
    candidates = [
        # overlaps negative k = 0 (ASSEMBLIES[0], ctgA = FASTA record 0, [10000, 11024))
        {
            "accession": f"{ASSEMBLIES[0]}:c0",
            "candidate_id": "r0:a",
            "locus_start": 10_100,
            "locus_end": 10_184,
        },
        # same assembly, ctgB = record 1, nowhere near a window
        {
            "accession": f"{ASSEMBLIES[0]}:c1",
            "candidate_id": "r0:b",
            "locus_start": 1,
            "locus_end": 50,
        },
        # a host the benchmark does not hold: never resolved, never counted
        {
            "accession": "GCA_999999999.1:c0",
            "candidate_id": "r0:c",
            "locus_start": 0,
            "locus_end": 90,
        },
    ]
    for row in candidates:
        row.update(pool="genomic_window", score=0.99)
    _write(
        root,
        ROUND0_REL,
        {"schema_version": "1.0", "step": "P2-10e", "n_candidates": 3, "candidates": candidates},
    )
    statuses = {
        "r0:a": ("passed", "failed"),
        "r0:b": ("failed", "failed"),
        "r0:c": ("unavailable", "passed"),
    }
    rows = [
        {
            "candidate_id": cid,
            "synteny_status": syn,
            "covariation_status": cov,
            "stage2_posterior": 0.9,
        }
        for cid, (syn, cov) in statuses.items()
    ]
    _write(root, SPARE_REL, {"schema_version": "1.0", "n_candidates": 3, "rows": rows})
    fasta = root / GENOMES_REL / f"{ASSEMBLIES[0]}.fna"
    fasta.parent.mkdir(parents=True, exist_ok=True)
    fasta.write_text(">ctgA a contig\nACGT\n>ctgB\nACGT\n", encoding="utf-8")


def _model_outputs(root: Path, prefix: str, items: list[dict], called: set[str]) -> dict[str, Path]:
    """Both arms' candidate tables, payload manifests, Stage-1 archives and Stage-2 files."""
    gate2 = json.loads((REPO / GATE2_REL).read_text())
    load = gate2["scoring"]["load"]
    identity = {
        "arm": gate2["scoring"]["arm"],
        "checkpoint_root": "data/processed/checkpoints/stage2_rinalmo",
        "checkpoint_dir": load["checkpoint_dir"],
        "backbone": "multimolecule/rinalmo-giga",
        "revision": load["revision"],
        "dtype": "bfloat16",
        "adapter_sha256": load["adapter_sha256"],
        "heads_sha256": load["heads_sha256"],
        "attn_implementation": "flash_attention_2",
        "batch_size": 4,
        "env_lock": "envs/ml-rna.conda-lock.yml",
    }
    paths: dict[str, Path] = {}
    for arm in ARMS:
        rows = []
        for item in items:
            cid = item["contig_id"]
            if item["label"] == 1:
                index = int(cid.split(":")[1])
                base = 0.60 if arm == "twin" else 0.70
                rows.append(_row(cid, base + index / 100, 0.90 + (index % 5) / 100))
            elif cid in called:
                rank = sorted(called).index(cid)
                posterior = 0.30 if item.get("beside") else 0.05
                rows.append(_row(cid, 0.70 + rank / 1000, posterior))
        paths[f"{arm}_table"] = _write(
            root, f"{P3_16}/{prefix}table_{arm}_t0.5.json", {"rows": rows}
        )
        payloads = [
            {"row_id": f"{r['contig_id']}#{s}", "contig_id": r["contig_id"]}
            for r in rows
            for s in "+-"
        ]
        paths[f"{arm}_payloads"] = _write(
            root, f"{P3_16}/{prefix}payloads_{arm}_t0.5.json", {"payloads": payloads}
        )
        union = [*payloads, {"row_id": "pos:000#t0.9", "contig_id": "pos:000"}]
        union_path = _write(root, f"{P3_16}/{prefix}payloads_{arm}_union.json", {"payloads": union})
        paths[f"{arm}_stage2"] = _write(
            root,
            f"{P3_16}/{prefix}stage2_{arm}.json",
            {
                **identity,
                "schema_version": "1.1",
                "payloads_sha256": _sha_bytes(union_path.read_bytes()),
                "shard": None,
                "n_payloads": len(union),
                "logits": {p["row_id"]: 0.0 for p in union},
            },
        )
        stage1 = root / f"{P3_16}/{prefix}stage1_{arm}.npz"
        np.savez(stage1, **{"pos:000/logits": np.zeros((1, 4, 2)), "pos:000/starts": [0]})
        paths[f"{arm}_stage1"] = stage1
    return paths


def _cli(root: Path, argv: list[str]) -> int:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(P, "repo_relative", lambda path: str(Path(path).resolve().relative_to(root)))
        return P.main(argv)


def _power(
    root: Path, bench: Path, payloads: Path, out: str, *, stage: str, draw: str | None = None
) -> int:
    argv = ["power", "--benchmark", str(bench), "--twin-payloads", str(payloads)]
    argv += ["--stage1-threshold", "0.5", "--stage", stage, "--out", str(root / out)]
    if draw:
        argv += ["--draw-power", str(root / draw)]
    return _cli(root, argv)


def _fold_argv(
    root: Path, bench: Path, paths: dict, *, pools: list[str], power: str, draw_power: str | None
) -> list[str]:
    argv = ["fold", "--benchmark", str(bench), "--out", str(root / ITEMS_REL)]
    argv += ["--stage1-threshold", "0.5", "--rule-knobs", KNOBS]
    argv += ["--temperature-source", str(root / GATE2_REL)]
    for arm in ARMS:
        argv += [f"--{arm}-checkpoint", f"data/processed/checkpoints/stage1_{arm}/stage1.pt"]
        for kind in ("table", "stage1", "stage2", "payloads"):
            argv += [f"--{arm}-{kind}", str(paths[f"{arm}_{kind}"])]
    argv += ["--hardneg-pool", str(root / pools[0])]
    if len(pools) > 1:
        argv += ["--hardneg-pool-census", str(root / pools[1])]
    argv += ["--hardneg-power", str(root / power)]
    if draw_power:
        argv += ["--hardneg-power-draw", str(root / draw_power)]
    argv += ["--v2-items", str(root / V2_ITEMS_REL)]
    argv += [
        "--round0-manifest",
        str(root / ROUND0_REL),
        "--round0-spare-inputs",
        str(root / SPARE_REL),
    ]
    argv += ["--genome-dir", str(root / GENOMES_REL)]
    return argv


def _grade(root: Path) -> dict:
    folded = json.loads((root / ITEMS_REL).read_text())
    report = P.precision_gain(
        folded,
        stage2_operating_point=0.5,
        decoy_prevalence=100,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=N_BOOT,
        seed=42,
        sources={**folded["source"], "items": ITEMS_REL},
        generated_at_utc="2026-10-28T00:00:00+00:00",
    )
    report["provenance"] = {"inputs": {ITEMS_REL: _sha_bytes((root / ITEMS_REL).read_bytes())}}
    return report


def _common(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / GATE2_REL, root / GATE2_REL)
    _round0(root)
    v2, v2_items = _v2(root)
    return {"v2": v2, "v2_items": v2_items}


def build_powered(root: Path) -> dict:
    """A draw with 24 graded + 4 beside windows; 21 graded and 2 beside draw a candidate.

    P = 21 >= N_min, so the draw grades. Stage-2 rejects the called graded negatives, so the
    gate passes; the beside windows carry a candidate and must not rescue or move anything.
    """
    root = root.resolve()
    common = _common(root)
    graded = [_native(k, role="graded") for k in range(24)]
    beside = [_native(k, role="beside") for k in range(24, 28)]
    pool = _write_pool(root, "draw", graded, beside, POOL_REL)
    bench = _mint(root, [pool], common["v2"], f"{P3_16}/hardneg_benchmark.json")
    manifest = json.loads(bench.read_text())
    called = {f"hn:{item['item_id']}" for item in graded[:21]} | {
        f"hn:{item['item_id']}" for item in beside[:2]
    }
    paths = _model_outputs(root, "hardneg_", manifest["items"], called)
    rc = _power(root, bench, paths["twin_payloads"], f"{P3_16}/hardneg_power.json", stage="draw")
    assert rc == 0
    argv = _fold_argv(
        root, bench, paths, pools=[POOL_REL], power=f"{P3_16}/hardneg_power.json", draw_power=None
    )
    assert _cli(root, argv) == 0
    return {
        "root": root,
        "bench": bench,
        "paths": paths,
        "argv": argv,
        "pool": pool,
        "report": _grade(root),
        **common,
    }


def build_census(root: Path) -> dict:
    """A draw that decides extend_census (P = 3), and the census beside it (P = 3 + 2 = 5).

    The census benchmark is the draw's windows plus the census's — the census selects only what
    the draw left, and no drawn item is dropped. P < N_min on the census: underpowered, with the
    verdict computed exactly as A13 does.
    """
    root = root.resolve()
    common = _common(root)
    draw_graded = [_native(k, role="graded") for k in range(10)]
    draw_beside = [_native(k, role="beside") for k in range(10, 12)]
    census_graded = [_native(k, role="graded") for k in range(12, 20)]
    census_beside = [_native(k, role="beside") for k in range(20, 22)]
    draw = _write_pool(root, "draw", draw_graded, draw_beside, POOL_REL)
    census = _write_pool(root, "census", census_graded, census_beside, POOL_CENSUS_REL)
    called_draw = {f"hn:{item['item_id']}" for item in draw_graded[:3]}
    draw_bench = _mint(root, [draw], common["v2"], f"{P3_16}/hardneg_benchmark.json")
    draw_paths = _model_outputs(
        root, "hardneg_", json.loads(draw_bench.read_text())["items"], called_draw
    )
    draw_power = f"{P3_16}/hardneg_power.json"
    assert (
        _power(root, draw_bench, draw_paths["twin_payloads"], draw_power, stage="draw")
        == P.EXIT_EXTEND_CENSUS
    )
    bench = _mint(root, [draw, census], common["v2"], f"{P3_16}/hardneg_census_benchmark.json")
    called = called_draw | {f"hn:{item['item_id']}" for item in census_graded[:2]}
    paths = _model_outputs(root, "hardneg_census_", json.loads(bench.read_text())["items"], called)
    power = f"{P3_16}/hardneg_census_power.json"
    assert _power(root, bench, paths["twin_payloads"], power, stage="census", draw=draw_power) == 0
    argv = _fold_argv(
        root, bench, paths, pools=[POOL_REL, POOL_CENSUS_REL], power=power, draw_power=draw_power
    )
    assert _cli(root, argv) == 0
    return {
        "root": root,
        "bench": bench,
        "paths": paths,
        "argv": argv,
        **common,
        "report": _grade(root),
    }


@pytest.fixture(scope="module")
def powered_case(tmp_path_factory) -> dict:
    return build_powered(tmp_path_factory.mktemp("hardneg_powered"))


@pytest.fixture(scope="module")
def census_case(tmp_path_factory) -> dict:
    return build_census(tmp_path_factory.mktemp("hardneg_census"))


@pytest.fixture
def hn(powered_case) -> dict:
    return copy.deepcopy(powered_case["report"])


@pytest.fixture
def hn_census(census_case) -> dict:
    return copy.deepcopy(census_case["report"])


def _fires(report: dict, needle: str) -> list[str]:
    problems = P.precision_problems(report)
    assert any(needle in problem for problem in problems), problems
    return problems


# ══════════════════════════════════════════════════════════════════════════════════════
# v1 and v2 keep validating byte-unchanged under the clause sets they were published under
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_committed_v1_and_v2_reports_still_validate_to_nothing():
    """A17 Pin 5: the 1.3 clauses are gated on schema >= 1.3. The committed v1 (1.1) and v2
    (1.2) reports, and their τ annexes, return [] from today's validators, and their bytes are
    read, never rewritten."""
    for rel, schema in (
        ("reports/two_stage_precision.json", "1.1"),
        ("reports/two_stage_precision_v2.json", "1.2"),
        ("reports/p3/two_stage_precision_tau07.json", "1.1"),
        ("reports/p3/two_stage_precision_tau09.json", "1.1"),
        ("reports/p3/two_stage_precision_v2_tau07.json", "1.2"),
        ("reports/p3/two_stage_precision_v2_tau09.json", "1.2"),
    ):
        raw = (REPO / rel).read_bytes()
        report = json.loads(raw)
        assert report["schema_version"] == schema, rel
        assert P.precision_problems(report) == [], rel
        assert (REPO / rel).read_bytes() == raw, rel
    v2 = json.loads((REPO / "reports/two_stage_precision_v2.json").read_text())
    # The committed reports/ inputs are present; the interim ones are excused where absent.
    assert P.precision_input_problems(v2, repo_root=REPO) == []
    assert not P.folded_is_hardneg(json.loads((REPO / V2_ITEMS_REL).read_text()))


def test_n_min_and_the_sentence_are_a17_pin_4s():
    assert P.N_MIN == 20 and not isinstance(P.N_MIN, bool)
    assert f"fewer than {P.N_MIN} graded negatives" in P.HARDNEG_UNDERPOWERED_STATEMENT
    assert P.HARDNEG_UNDERPOWERED_STATEMENT == (
        "fewer than 20 graded negatives reached Stage-2; the hard-negative benchmark lacks power "
        "to test Stage-2"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# The committed A17 result: the census report is REFUSED, and on one clause alone
# ══════════════════════════════════════════════════════════════════════════════════════
REFUSED_REL = "reports/two_stage_precision_hardneg.invalid.json"
ANNEX_RELS = (
    "reports/p3/two_stage_precision_hardneg_tau07.json",
    "reports/p3/two_stage_precision_hardneg_tau09.json",
)
TAU_VARIANT = (
    "the gate's verdict changes across the Stage-1 threshold sweep; ADR-0005 D3 leaves that "
    "value unpinned until P5-01, so the gate would rest on it"
)


def test_the_committed_a17_report_is_refused_on_tau_invariance_alone():
    """The census run's report was diverted to ``.invalid.json`` (exit 3), and the canonical
    path was never written. Today's validators re-derive exactly one problem from its bytes:
    τ 0.5 and 0.7 pass A13's strict ``>`` while τ 0.9 ties, so the verdict rests on a value D3
    leaves unfrozen. The verdict is re-derived here, not read from the report's own ``problems``
    field. Both are asserted present, never skipped: they are plain JSON under ``reports/``.

    The item table this report binds is DVC-tracked (1.05 GB), so its input binding
    (``precision_input_problems``) is checked where it is pulled, not here."""
    assert not (REPO / OUT_REL).exists(), "a refused run must leave the canonical path absent"
    report = json.loads((REPO / REFUSED_REL).read_bytes())
    assert report["schema_version"] == "1.3" and report["is_science"] is True
    assert P.precision_problems(report) == [TAU_VARIANT]
    sensitivity = report["stage1_threshold_sensitivity"]
    verdicts = {sensitivity["base"]["stage1_threshold"]: sensitivity["base"]["passes"]}
    verdicts |= {row["measured_stage1_threshold"]: row["passes"] for row in sensitivity["points"]}
    assert verdicts == {0.5: True, 0.7: True, 0.9: False}
    power = report["power"]
    assert power["P"] == 6 < P.N_MIN and power["decision"] == "underpowered"
    assert power["extension"] == {"P_draw": 0, "P_final": 6, "triggered": True}
    assert report["gate"]["verdict_class"] == "underpowered"
    assert (REPO / f"{ITEMS_REL}.dvc").is_file(), "the replayable item table must be DVC-tracked"


@pytest.mark.parametrize("rel", ANNEX_RELS)
def test_the_committed_a17_tau_annexes_validate_to_nothing(rel):
    """Each annex is a report in its own right: no sweep of its own, so it validates clean. Its
    verdict (τ 0.7 passes, τ 0.9 ties) is what makes the τ 0.5 report τ-variant."""
    report = json.loads((REPO / rel).read_bytes())
    assert report["schema_version"] == "1.3"
    assert P.precision_problems(report) == []
    assert report["gate"]["verdict_class"] == "underpowered"


@pytest.mark.parametrize("rel", [POOL_REL, POOL_CENSUS_REL])
def test_the_committed_pool_reports_validate_and_bind_their_ids_annex(rel):
    """The committed draw and census pool reports pass their own validator. The draw replay
    needs the interim eligible list, which CI does not have, so ``root`` is left unset. The
    committed ids annex is re-hashed against the digest the report records."""
    from tbox_finder.integration import hardneg_pool

    report = json.loads((REPO / rel).read_bytes())
    assert hardneg_pool.validate_report(report) == []
    annex = report["ids_annex"]
    assert hashlib.sha256((REPO / annex["path"]).read_bytes()).hexdigest() == annex["sha256"]


# ══════════════════════════════════════════════════════════════════════════════════════
# The mint
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_manifest_is_v2s_contig_and_item_schema_and_two_stage_reads_it(powered_case):
    manifest = json.loads(powered_case["bench"].read_text())
    v2 = powered_case["v2"]
    assert manifest["schema_version"] == "1.2" and manifest["kind"] == P.HARDNEG_MANIFEST_KIND
    assert P.manifest_is_hardneg(manifest)
    contigs = TS.read_contigs(powered_case["bench"])  # the two_stage legs' own reader
    assert len(contigs) == len(manifest["items"]) == 24 + 28
    v2_keys = set(v2["items"][0])
    assert all(v2_keys <= set(item) for item in manifest["items"])
    negatives = [item for item in manifest["items"] if item["label"] == 0]
    assert {item["pool"] for item in negatives} == set(P.HARDNEG_SUB_CLASSES)
    assert "leader_decoy" not in manifest["scope"]["negatives_by_pool"]  # A17 has no leader decoy
    by_id = {contig["contig_id"]: contig for contig in manifest["contigs"]}
    for item in negatives:
        meta = item["hardneg"]
        k = int(meta["feature_id"].split("-")[1])
        assert by_id[item["contig_id"]]["sequence"] == NEGATIVE_WINDOWS[k]  # native bytes
        assert (
            hashlib.sha256(by_id[item["contig_id"]]["sequence"].encode()).hexdigest()
            == meta["sha256"]
        )
        assert item["block"] == f"assembly:{meta['assembly']}"
        assert item["gated"] is (not meta["d4_context"]) and item["beside"] is meta["d4_context"]
        assert item["resolved_order"] == ORDERS[meta["assembly"]]


def test_the_positives_are_v2s_byte_for_byte_and_the_digest_says_so(powered_case):
    manifest = json.loads(powered_case["bench"].read_text())
    v2_contigs = {c["contig_id"]: c for c in powered_case["v2"]["contigs"]}
    positives = [item for item in manifest["items"] if item["label"] == 1]
    assert len(positives) == len(v2_contigs) == manifest["source"]["n_positives"]
    for contig in manifest["contigs"]:
        if contig["contig_id"].startswith("pos:"):
            assert contig == v2_contigs[contig["contig_id"]]
    lines = sorted(
        f"{cid}\t{hashlib.sha256(c['sequence'].encode()).hexdigest()}\n"
        for cid, c in v2_contigs.items()
    )
    assert (
        manifest["source"]["positives_sha256"]
        == hashlib.sha256("".join(lines).encode()).hexdigest()
    )
    assert all(item["gated"] is True and item["beside"] is False for item in positives)


def test_the_fold_overlap_is_measured_on_the_whole_window_both_strands(powered_case):
    overlap = json.loads(powered_case["bench"].read_text())["scope"]["host_pool"][
        "overlap_with_training_folds"
    ]
    # Production's fold carries window 0 verbatim; windows 0-4 are five phases of ONE slab, so
    # all five share DNA with it. The twin's fold shares none.
    assert overlap["by_arm"] == {"production": 5, "twin": 0}
    assert overlap["n_seen_by_any_arm"] == 5 and overlap["n_negatives"] == 28
    # Both strands: a fold carrying only the reverse complement of a window still finds it.
    contigs = [{"contig_id": "hn:x", "sequence": NEGATIVE_WINDOWS[3]}]
    items = [{"contig_id": "hn:x", "label": 0}]
    rc = wd.reverse_complement(NEGATIVE_WINDOWS[3])
    seen = MINT.native_fold_overlap(contigs, items, {"twin": {"r"}}, {"r": rc})
    assert seen == {"twin": {"hn:x"}}
    assert MINT.native_fold_overlap(contigs, items, {"twin": {"r"}}, {"r": "ACGT" * 10}) == {
        "twin": set()
    }


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (
            lambda i: i.update(
                sequence=i["sequence"][:-1] + ("A" if i["sequence"][-1] != "A" else "C")
            ),
            "do not hash",
        ),
        (
            lambda i: i.update(
                sequence=i["sequence"].lower(),
                sha256=hashlib.sha256(i["sequence"].lower().encode()).hexdigest(),
            ),
            "encoder reads",
        ),
        (
            lambda i: i.update(
                sequence="N" + i["sequence"][1:],
                sha256=hashlib.sha256(("N" + i["sequence"][1:]).encode()).hexdigest(),
            ),
            "encoder reads",
        ),
        (lambda i: i.update(d4_context=True), "d4_context=True in the graded list"),
        (lambda i: i.update(end=i["end"] + 1), "is not 1024 nt"),
        (lambda i: i.update(delta=257), "outside U{0..256}"),
        (lambda i: i.update(feature_type="rRNA"), "two leader sub-classes"),
        (lambda i: i.update(item_id="x"), "is not its window's key"),
        (lambda i: i.update(start=True), "integer coordinates"),
        (lambda i: i.pop("sha256"), "carries no ['sha256']"),
    ],
)
def test_a_pool_item_that_is_not_the_screened_window_is_refused(edit, needle):
    item = _native(0, role="graded")
    assert MINT.hardneg_item_problems(item, role="graded") == []
    edit(item)
    problems = MINT.hardneg_item_problems(item, role="graded")
    assert any(needle in problem for problem in problems), problems


def test_the_pool_must_name_the_item_files_the_mint_was_handed(powered_case):
    pool = powered_case["pool"]
    from tbox_finder.integration import hardneg_pool

    ok = MINT.hardneg_pool_refusals(
        pool["report"],
        mode="draw",
        item_digests=pool["digests"],
        validate=hardneg_pool.validate_report,
    )
    assert ok == []
    swapped = {**pool["digests"], "beside": ("0" * 64, pool["digests"]["beside"][1])}
    found = MINT.hardneg_pool_refusals(
        pool["report"], mode="draw", item_digests=swapped, validate=lambda r: []
    )
    assert any("beside list" in p for p in found), found
    found = MINT.hardneg_pool_refusals(
        pool["report"], mode="census", item_digests=pool["digests"], validate=lambda r: []
    )
    assert any("mode is 'draw', not 'census'" in p for p in found), found
    found = MINT.hardneg_pool_refusals(
        {"kind": "x"}, mode="draw", item_digests={}, validate=lambda r: []
    )
    assert found and "kind" in found[0]


def test_a_pool_that_fails_its_own_validator_is_refused_by_the_minter(powered_case):
    hardneg_pool = pytest.importorskip(
        "tbox_finder.integration.hardneg_pool",
        reason="hardneg_pool (the pool's own validator) is not importable here",
    )
    pool = copy.deepcopy(powered_case["pool"]["report"])
    pool["audit"]["n_violations"] = 1  # the audit verdict re-derives from its rows: refused
    assert hardneg_pool.validate_report(pool)
    found = MINT.hardneg_pool_refusals(
        pool,
        mode="draw",
        item_digests=powered_case["pool"]["digests"],
        validate=hardneg_pool.validate_report,
    )
    assert any("its own validator" in p for p in found), found


def test_a_window_minted_twice_is_refused():
    lists = {"graded": [_native(0, role="graded")], "beside": []}
    with pytest.raises(SystemExit, match="minted twice"):
        MINT.hardneg_negatives([("draw", lists), ("census", lists)], resolved_order={})


@pytest.mark.parametrize(
    "rel",
    [
        f"{P3_16}/benchmark_v1.json",
        f"{P3_16}/benchmark_v2.json",
        f"{P3_16}/v2_benchmark_hardneg.json",
        f"{P3_16}/benchmark_v3.json",
    ],
)
def test_the_mint_refuses_a_v1_v2_or_unnamed_output(tmp_path, rel):
    assert MINT.hardneg_out_violation(tmp_path / rel)
    assert MINT.hardneg_out_violation(tmp_path / f"{P3_16}/hardneg_benchmark.json") is None


# ══════════════════════════════════════════════════════════════════════════════════════
# The power gate (A17 Pin 4)
# ══════════════════════════════════════════════════════════════════════════════════════
def test_p_counts_graded_negatives_with_a_payload_and_never_the_beside(powered_case):
    power = json.loads((powered_case["root"] / f"{P3_16}/hardneg_power.json").read_text())
    assert power["P"] == 21 and power["decision"] == P.POWER_GRADE and power["stage"] == "draw"
    assert power["n_beside_with_payload"] == 2  # counted beside, never in P
    assert power["extension"] == {"triggered": False, "P_draw": 21, "P_final": 21}
    assert P.power_problems(power) == []


def test_the_draw_extends_and_the_census_is_underpowered(census_case):
    root = census_case["root"]
    draw = json.loads((root / f"{P3_16}/hardneg_power.json").read_text())
    census = json.loads((root / f"{P3_16}/hardneg_census_power.json").read_text())
    assert draw["decision"] == P.POWER_EXTEND and draw["P"] == 3
    assert draw["extension"] == {"triggered": True, "P_draw": 3, "P_final": None}
    assert census["decision"] == P.POWER_UNDERPOWERED and census["P"] == 5
    assert census["extension"] == {"triggered": True, "P_draw": 3, "P_final": 5}
    assert census["draw"]["sha256"] == _sha_bytes(
        (root / f"{P3_16}/hardneg_power.json").read_bytes()
    )


def test_the_power_gate_reads_payloads_alone_and_only_at_tau_one_half(powered_case):
    manifest = json.loads(powered_case["bench"].read_text())
    record = {"path": "x", "sha256": "0" * 64}
    graded = [i["contig_id"] for i in manifest["items"] if i["label"] == 0 and i["gated"]]
    bare = [{"contig_id": cid} for cid in graded[:20]]  # no score of any kind on a payload row
    power = P.hardneg_power(
        manifest,
        bare,
        stage="draw",
        stage1_threshold=0.5,
        benchmark_record=record,
        payloads_record=record,
    )
    assert power["P"] == 20 and power["decision"] == P.POWER_GRADE  # exactly N_min grades
    power = P.hardneg_power(
        manifest,
        bare[:19],
        stage="draw",
        stage1_threshold=0.5,
        benchmark_record=record,
        payloads_record=record,
    )
    assert power["decision"] == P.POWER_EXTEND
    with pytest.raises(P.PrecisionError, match="τ = 0.5"):
        P.hardneg_power(
            manifest,
            bare,
            stage="draw",
            stage1_threshold=0.7,
            benchmark_record=record,
            payloads_record=record,
        )
    with pytest.raises(P.PrecisionError, match="does not hold"):
        P.hardneg_power(
            manifest,
            [{"contig_id": "hn:nowhere"}],
            stage="draw",
            stage1_threshold=0.5,
            benchmark_record=record,
            payloads_record=record,
        )


def test_a_census_opens_only_on_a_draw_that_decided_extend(powered_case):
    manifest = json.loads(powered_case["bench"].read_text())
    record = {"path": "x", "sha256": "0" * 64}
    graded_draw = json.loads((powered_case["root"] / f"{P3_16}/hardneg_power.json").read_text())
    with pytest.raises(P.PrecisionError, match="needs --draw-power"):
        P.hardneg_power(
            manifest,
            [],
            stage="census",
            stage1_threshold=0.5,
            benchmark_record=record,
            payloads_record=record,
        )
    with pytest.raises(P.PrecisionError, match="no second extension"):
        P.hardneg_power(
            manifest,
            [],
            stage="census",
            stage1_threshold=0.5,
            benchmark_record=record,
            payloads_record=record,
            draw_power=graded_draw,
            draw_record=record,
        )


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (lambda p: p.update(decision=P.POWER_GRADE), "re-derives 'extend_census'"),
        (lambda p: p.update(P=4), "is not the number of distinct graded negatives"),
        (lambda p: p.update(n_min=10), "n_min is 10"),
        (lambda p: p.update(n_min=True), "n_min is True"),
        (lambda p: p["extension"].update(triggered=False), "draw-stage extension record"),
        (lambda p: p.update(stage1_threshold=0.9), "counted at τ = 0.9"),
        (lambda p: p.update(arm="production"), "counted on arm 'production'"),
        (lambda p: p["payloads"].pop("sha256"), "payloads is not bound"),
        (lambda p: p.update(stage="census"), "census extension record"),
    ],
)
def test_the_power_record_re_derives_its_own_decision(census_case, edit, needle):
    power = json.loads((census_case["root"] / f"{P3_16}/hardneg_power.json").read_text())
    assert P.power_problems(power) == []
    edit(power)
    found = P.power_problems(power)
    assert any(needle in p for p in found), found


def test_the_power_gate_stops_the_dag_at_extend_census(census_case, powered_case, tmp_path):
    root = census_case["root"]
    assert (
        P.main(["power-gate", "--power", str(root / f"{P3_16}/hardneg_power.json")])
        == P.EXIT_EXTEND_CENSUS
    )
    proot = powered_case["root"]
    sentinel = proot / f"{P3_16}/hardneg_power_gate.json"
    assert (
        _cli(
            proot,
            [
                "power-gate",
                "--power",
                str(proot / f"{P3_16}/hardneg_power.json"),
                "--out",
                str(sentinel),
            ],
        )
        == 0
    )
    assert json.loads(sentinel.read_text())["decision"] == P.POWER_GRADE
    broken = tmp_path / "hardneg_power.json"
    record = json.loads((proot / f"{P3_16}/hardneg_power.json").read_text())
    broken.write_text(json.dumps({**record, "P": 999}))
    assert P.main(["power-gate", "--power", str(broken)]) == 3


@pytest.mark.parametrize(
    "rel", [f"{P3_16}/v2_power.json", f"{P3_16}/v1_power.json", "reports/power.json"]
)
def test_the_power_leg_refuses_a_non_a17_output(powered_case, rel):
    root = powered_case["root"]
    with pytest.raises((P.PrecisionError, TS.TwoStageError)):
        _power(
            root, powered_case["bench"], powered_case["paths"]["twin_payloads"], rel, stage="draw"
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# The 1.3 report
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_powered_report_is_clean_and_bound_to_its_files(powered_case, hn):
    assert hn["schema_version"] == P.HARDNEG_SCHEMA_VERSION
    assert P.precision_problems(hn) == []
    assert (
        P.precision_input_problems(hn, repo_root=powered_case["root"], require_all_inputs=True)
        == []
    )
    assert hn["gate"]["verdict_class"] == "powered" and "underpowered_statement" not in hn["gate"]
    assert hn["gate"]["passes"] is True and hn["is_science"] is True
    assert "ADR-0005 A17" in hn["adr"] and P.HARDNEG_SELECTION_TAG in hn["selection"]


def test_the_census_report_is_underpowered_with_the_verdict_still_computed(census_case, hn_census):
    report = hn_census
    assert P.precision_problems(report) == []
    assert (
        P.precision_input_problems(report, repo_root=census_case["root"], require_all_inputs=True)
        == []
    )
    assert report["power"]["P"] == 5 and report["power"]["extension"]["triggered"] is True
    assert report["gate"]["verdict_class"] == "underpowered"
    assert report["gate"]["underpowered_statement"] == P.HARDNEG_UNDERPOWERED_STATEMENT
    point = report["arms"]["twin"]["prevalence"]["100:1"]["auprc"]
    assert report["gate"]["passes"] == (point["two_stage"] > point["stage1_only"])  # A13, unchanged
    assert report["population"]["stage"] == "census" and report["population"]["census"] is not None
    assert report["population"]["n_graded"] == 18  # 10 drawn + 8 census: no drawn item dropped


def test_the_graded_statistic_is_over_positives_and_graded_negatives_only(powered_case, hn):
    arm = hn["arms"]["twin"]
    assert arm["population"]["n_negatives"] == 24 and arm["population"]["n_positives"] == 24
    assert arm["prevalence"]["100:1"]["lambda"] == pytest.approx(100 * 24 / 24)
    # Moving every beside score cannot move the graded read: they are filtered at the call site.
    folded = json.loads((powered_case["root"] / ITEMS_REL).read_text())
    for item in folded["items"]:
        if item.get("beside"):
            for arm_name in item["arms"]:
                item["arms"][arm_name].update(n_rows=1, stage1_only=0.999, two_stage=0.999)
    moved = P.precision_gain(
        folded,
        stage2_operating_point=0.5,
        decoy_prevalence=100,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=N_BOOT,
        seed=42,
        sources=hn["sources"],
        generated_at_utc=hn["generated_at_utc"],
    )
    assert moved["gate"] == hn["gate"]
    assert moved["beside_gate"] != hn["beside_gate"]


def test_the_beside_block_reads_the_d4_context_windows_and_is_never_graded(hn):
    beside = hn["beside_gate"]
    twin = beside["by_arm"]["twin"]
    assert beside["graded"] is False and beside["n_beside"] == 4
    assert twin["n_with_stage1_candidate"] == 2 and len(twin["called"]) == 2
    denom = twin["with_beside_in_denominator"]
    assert denom["graded"] is False and denom["n_negatives"] == 24 + 4
    assert denom["lambda"] == pytest.approx(100 * 24 / 28)


def test_the_round0_crosstab_counts_overlaps_and_drops_nothing(hn):
    cross = hn["round0_crosstab"]
    assert cross["n_overlapping"] == 1 and cross["n_overlapping_with_twin_candidate"] == 1
    assert cross["n_round0_candidates_overlapped"] == 1
    assert cross["orthogonal_status"] == {
        "orthogonal_pass": 1,
        "no_orthogonal_pass": 0,
        "undetermined": 0,
    }
    assert (
        cross["per_criterion_over_overlapped_candidates"]["relaxed_architecture"]["not_recorded"]
        == 1
    )
    assert hn["arms"]["twin"]["population"]["n_negatives"] == cross["n_graded_negatives"]


def test_an_unrecorded_criterion_is_undetermined_never_failed():
    assert (
        P._orthogonal_status(
            [
                {
                    "downstream_aaRS_synteny": "failed",
                    "any_helix_rscape": "failed",
                    "relaxed_architecture": None,
                }
            ]
        )
        == "undetermined"
    )
    assert (
        P._orthogonal_status(
            [
                {
                    "downstream_aaRS_synteny": "failed",
                    "any_helix_rscape": "failed",
                    "relaxed_architecture": "failed",
                }
            ]
        )
        == "no_orthogonal_pass"
    )
    assert (
        P._orthogonal_status(
            [
                {
                    "downstream_aaRS_synteny": "unavailable",
                    "any_helix_rscape": "passed",
                    "relaxed_architecture": None,
                }
            ]
        )
        == "orthogonal_pass"
    )


def test_the_per_positive_delta_from_v2_is_published(hn):
    twin = hn["v2_positive_delta"]["by_arm"]["twin"]["two_stage"]
    assert twin["n_compared"] == 24 and twin["n_changed"] == 6  # i % 4 == 0
    assert twin["max_abs_delta"] == pytest.approx(0.001)
    assert hn["v2_positive_delta"]["by_arm"]["production"]["two_stage"]["n_changed"] == 0


def test_the_disclosures_carry_a17_pin_5_a_to_f_exactly_once(hn, hn_census):
    for report in (hn, hn_census):
        for tag in P.HARDNEG_DISCLOSURE_TAGS:
            assert sum(line.startswith(tag) for line in report["disclosures"]) == 1, tag
    assert any(P.HARDNEG_UNDERPOWERED_STATEMENT in line for line in hn_census["disclosures"])
    assert not any(P.HARDNEG_UNDERPOWERED_STATEMENT in line for line in hn["disclosures"])


def _gated_count_edit(r):
    r["benchmark"]["scope"]["n_negatives_gated"] += 1


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        # (1) the graded / beside partition
        (
            lambda r: r["benchmark"]["scope"]["negatives_by_pool"].update(leader_decoy=1),
            "native leaders only",
        ),
        (
            lambda r: r["benchmark"]["scope"]["beside_by_pool"].update(cds_upstream=99),
            "graded pools plus its beside pools",
        ),
        (
            lambda r: r["benchmark"]["scope"].update(n_beside=5),
            "D4-context negatives (every positive is graded)",
        ),
        # (2) the beside block
        (lambda r: r["beside_gate"].update(graded=True), "never graded"),
        (lambda r: r["beside_gate"]["by_arm"]["twin"]["called"].pop(), "called rows for"),
        (
            lambda r: r["beside_gate"]["by_arm"]["twin"].update(n_with_stage2_payload=9),
            "reached a Stage-2 payload",
        ),
        (
            lambda r: r["beside_gate"]["by_arm"]["twin"]["called"][0].update(two_stage=1.5),
            "does not carry both scores",
        ),
        (
            lambda r: r["beside_gate"]["by_arm"]["twin"]["distribution"]["two_stage"].update(n=3),
            "not over the called windows",
        ),
        (
            lambda r: r["beside_gate"]["by_arm"]["twin"]["with_beside_in_denominator"].update(
                **{"lambda": 1.0}
            ),
            "with-beside AUPRC read",
        ),
        (
            lambda r: r["beside_gate"]["by_arm"]["twin"]["with_beside_in_denominator"].update(
                gain_pp=99.0
            ),
            "with-beside AUPRCs",
        ),
        # (3) power
        (lambda r: r["power"].update(n_min=10), "power.n_min is 10"),
        (lambda r: r["power"].update(P=20), "not the graded fold's own count"),
        (
            lambda r: r["power"].update(this_threshold_payload_count=3),
            "this_threshold_payload_count",
        ),
        (lambda r: r["power"]["extension"].update(triggered=True), "extension record"),
        (lambda r: r["power"]["extension"].update(P_draw=7), "must be the final P"),
        (lambda r: r["power"].update(decision="extend_census"), "past an extend_census"),
        (lambda r: r["power"].update(verdict_class="underpowered"), "re-derives 'powered'"),
        (lambda r: r["gate"].update(verdict_class="underpowered"), "does not mirror"),
        (
            lambda r: r["gate"].update(underpowered_statement=P.HARDNEG_UNDERPOWERED_STATEMENT),
            "powered gate carries",
        ),
        (
            lambda r: r["power"]["by_arm"]["twin"].update(n_gated_negatives=7),
            "power block counts 7",
        ),
        # (4) Stage-2 identity, shared with A15
        (
            lambda r: r["stage2_identity"]["twin"].update(dtype="float16"),
            "different Stage-2 identities",
        ),
        # (5) file bindings
        (
            lambda r: r["sources"]["interim_sha256"].pop(r["sources"]["v2_items"]),
            "sources.v2_items",
        ),
        (lambda r: r["power"]["source"].update(sha256="0" * 64), "power.source is not bound"),
        (lambda r: r["population"]["draw"].update(sha256="0" * 64), "population.draw is not bound"),
        (
            lambda r: r["round0_crosstab"]["source"]["manifest"].update(path="x.json"),
            "round0_crosstab.source.manifest",
        ),
        (
            lambda r: r["v2_positive_delta"]["source"].update(sha256="0" * 64),
            "v2_positive_delta.source",
        ),
        # (6) the population
        (
            lambda r: r["benchmark"]["source"]["hardneg_pools"][0].update(sha256="1" * 64),
            "minted from",
        ),
        (lambda r: r["population"].update(stage="census"), "population stage"),
        (lambda r: r["population"].update(n_graded=25), "the pool(s) record"),
        # (7) each pool's screens
        (
            lambda r: r["benchmark"]["scope"]["host_pool"]["overlap_with_training_folds"][
                "by_arm"
            ].update(twin=1),
            "conjunct 1/4",
        ),
        (lambda r: r["population"]["draw"].update(n_refused_kmer=0), "conjunct 2/4"),
        (lambda r: r["population"]["draw"].update(n_controls_fired=10), "conjunct 3/4"),
        (lambda r: r.update(selection="ADR-0005 A15"), "conjunct 4/4"),
        (lambda r: r["population"]["draw"].update(content_threshold_bits=20.0), "not A15 Pin 2's"),
        (lambda r: r["population"]["draw"]["audit"].update(n_violations=1), "§7 stop"),
        (lambda r: r["population"]["draw"]["audit"].update(passes=False), "§7 stop"),
        (lambda r: r["population"]["draw"].update(n_problems=1), "carrying 1 problem"),
        # (8) familiarity
        (lambda r: r.update(familiarity_status="supplied"), "familiarity baseline covers"),
        (lambda r: r["disclosures"].append(P.CONFOUNDED_STATEMENT), "familiarity_status"),
        # (10) disclosures
        (
            lambda r: r["disclosures"].__setitem__(
                slice(None),
                [s for s in r["disclosures"] if not s.startswith(P.HARDNEG_DISCLOSURE_TAGS[1])],
            ),
            "Pin 5 (b) exactly once",
        ),
        (
            lambda r: r["disclosures"].__setitem__(
                slice(None), [s for s in r["disclosures"] if "never replacing it" not in s]
            ),
            "never replacing it",
        ),
        (
            lambda r: r["disclosures"].append(P.HARDNEG_UNDERPOWERED_STATEMENT),
            "underpowered sentence does not match",
        ),
        # (11) round-0 cross-tab
        (
            lambda r: r["round0_crosstab"].update(n_overlapping_with_twin_candidate=2),
            "not nested in the graded",
        ),
        (
            lambda r: r["round0_crosstab"]["orthogonal_status"].update(undetermined=1),
            "does not partition",
        ),
        (
            lambda r: r["round0_crosstab"]["per_criterion_over_overlapped_candidates"][
                "any_helix_rscape"
            ].update(failed=0),
            "per-criterion status counts",
        ),
        # (12) v2 delta
        (
            lambda r: r["v2_positive_delta"]["by_arm"]["twin"]["stage1_only"].update(n_compared=23),
            "compare every graded positive",
        ),
        # (13) composition
        (
            lambda r: r["composition"]["graded"]["by_sub_class"].update(cds_upstream=0),
            "do not partition the graded",
        ),
        (lambda r: r["composition"].update(d4_share=0.5), "d4_share"),
        # (14b) A17 Pin 3 pins
        (lambda r: r["rule"].update(seed=7), "A17 Pin 3 pins seed 42"),
        (lambda r: r["sources"]["rule_knobs"].update(flank=20), "are not v2's recorded"),
        (
            lambda r: r["v2_positive_delta"]["source"]["rule_knobs"].update(min_span=40),
            "are not v2's recorded",
        ),
        # (14) blocks
        (lambda r: r["blocks"].update(negative_blocks_are_assemblies=False), "resampling blocks"),
        (lambda r: r["blocks"].update(n_shared=1), "resampling blocks"),
    ],
)
def test_each_a17_clause_is_required_alone(hn, edit, needle):
    assert P.precision_problems(hn) == []
    broken = copy.deepcopy(hn)
    edit(broken)
    _fires(broken, needle)


def test_an_underpowered_gate_must_carry_the_a17_sentence_verbatim(hn_census):
    broken = copy.deepcopy(hn_census)
    broken["gate"]["underpowered_statement"] = P.UNDERPOWERED_STATEMENT  # A15's, not A17's
    _fires(broken, "A17 Pin 4's sentence verbatim")
    broken = copy.deepcopy(hn_census)
    broken["power"]["extension"].update(P_draw=25)
    _fires(broken, "extends only when the draw's P < 20")
    broken = copy.deepcopy(hn_census)
    broken["sources"]["interim_sha256"].pop(broken["sources"]["hardneg_power_draw"])
    _fires(broken, "sources.hardneg_power_draw")
    broken = copy.deepcopy(hn_census)
    broken["population"]["census"].update(stage="draw")
    _fires(broken, "of mode 'census'")


def test_the_v2_clause_set_does_not_grade_a_hardneg_report(hn):
    """A 1.3 report relabelled 1.2 is graded under A15's clause set, which refuses it: the leader
    block, the screen and the post-mint record are absent. A version label cannot move a report
    to a clause set that would pass it."""
    relabelled = copy.deepcopy(hn)
    relabelled["schema_version"] = P.SCHEMA_VERSION
    assert P.precision_problems(relabelled)


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (
            lambda r: r["beside_gate"]["by_arm"]["twin"]["called"][0].update(two_stage=0.31),
            "beside_gate does not re-derive",
        ),
        (
            lambda r: r["round0_crosstab"].update(n_beside_overlapping=3),
            "round0_crosstab does not re-derive",
        ),
        (
            lambda r: r["v2_positive_delta"]["by_arm"]["twin"]["two_stage"].update(mean_delta=0.5),
            "v2_positive_delta does not re-derive",
        ),
        (
            lambda r: r["composition"]["graded"]["by_clade"].update(Bacillales=0),
            "composition does not re-derive",
        ),
        (
            lambda r: r["power"]["by_arm"]["twin"].update(n_gated_negatives_with_stage2_payload=4),
            "power.by_arm does not re-derive",
        ),
        (lambda r: r["power"].update(P=22), "not the P the power record"),
    ],
)
def test_the_a17_blocks_re_derive_from_the_item_table_and_files(powered_case, edit, needle):
    broken = copy.deepcopy(powered_case["report"])
    edit(broken)
    found = P.precision_input_problems(broken, repo_root=powered_case["root"])
    assert any(needle in p for p in found), found


def test_a_pool_report_failing_its_own_validator_refuses_the_report(tmp_path):
    hardneg_pool = pytest.importorskip(
        "tbox_finder.integration.hardneg_pool",
        reason="hardneg_pool (the pool's own validator) is not importable here",
    )
    case = build_powered(tmp_path / "case")
    root, report = case["root"], copy.deepcopy(case["report"])
    pool = json.loads((root / POOL_REL).read_text())
    pool["audit"]["passes"] = False  # a verdict that disagrees with its own rows
    assert hardneg_pool.validate_report(pool)
    path = _write(root, POOL_REL, pool)
    # Re-bind the digest so ONLY the validator can refuse it, not the hash check.
    report["sources"]["interim_sha256"][POOL_REL] = _sha_bytes(path.read_bytes())
    found = P.precision_input_problems(report, repo_root=root)
    assert any("fails its own validator" in p for p in found), found
    assert not any("no longer hashes" in p for p in found), found


# ══════════════════════════════════════════════════════════════════════════════════════
# The fold's refusals
# ══════════════════════════════════════════════════════════════════════════════════════
def _drop(argv: list[str], flag: str) -> list[str]:
    index = argv.index(flag)
    return argv[:index] + argv[index + 2 :]


@pytest.mark.parametrize(
    "flag",
    [
        "--hardneg-pool",
        "--hardneg-power",
        "--v2-items",
        "--round0-manifest",
        "--genome-dir",
        "--twin-stage2",
    ],
)
def test_an_a17_fold_missing_an_input_is_refused(powered_case, flag):
    with pytest.raises(P.PrecisionError, match="a 1.3 report binds every one"):
        _cli(powered_case["root"], _drop(powered_case["argv"], flag))


def test_an_a17_fold_refuses_the_a15_screen_and_postmint(powered_case):
    argv = [*powered_case["argv"], "--postmint", str(powered_case["root"] / GATE2_REL)]
    with pytest.raises(P.PrecisionError, match="no post-mint assertion"):
        _cli(powered_case["root"], argv)


def test_an_a17_fold_refuses_an_extend_census_power_record(census_case):
    """A draw-stage record that decided extend_census, counted on THIS benchmark: the fold
    refuses it, because A17 Pin 4 computes no Stage-2 score and no AUPRC before the census."""
    root = census_case["root"]
    rel = f"{P3_16}/hardneg_extend_power.json"
    rc = _power(
        root, census_case["bench"], census_case["paths"]["twin_payloads"], rel, stage="draw"
    )
    assert rc == P.EXIT_EXTEND_CENSUS
    argv = _drop(list(census_case["argv"]), "--hardneg-power-draw")
    argv[argv.index("--hardneg-power") + 1] = str(root / rel)
    with pytest.raises(P.PrecisionError, match="before the census"):
        _cli(root, argv)


def test_an_a17_fold_refuses_a_census_record_without_its_draw(census_case):
    with pytest.raises(P.PrecisionError, match="needs --hardneg-power-draw"):
        _cli(census_case["root"], _drop(census_case["argv"], "--hardneg-power-draw"))


def test_an_a17_fold_refuses_a_pool_the_benchmark_was_not_minted_from(powered_case, tmp_path):
    root = powered_case["root"]
    other = root / "reports/p3/hardneg_pool_other.json"
    other.write_text((root / POOL_REL).read_text() + " ")
    argv = list(powered_case["argv"])
    argv[argv.index("--hardneg-pool") + 1] = str(other)
    with pytest.raises(P.PrecisionError, match="was minted from"):
        _cli(root, argv)


def test_an_a17_fold_refuses_a_power_record_counted_on_another_work_list(powered_case):
    root = powered_case["root"]
    record = json.loads((root / f"{P3_16}/hardneg_power.json").read_text())
    record["payloads"]["sha256"] = "9" * 64
    forged = _write(root, f"{P3_16}/hardneg_power_forged.json", record)
    argv = list(powered_case["argv"])
    argv[argv.index("--hardneg-power") + 1] = str(forged)
    with pytest.raises(P.PrecisionError, match="different payload manifest"):
        _cli(root, argv)


def test_an_a17_fold_refuses_locus_knobs_that_are_not_v2s(powered_case):
    argv = list(powered_case["argv"])
    argv[argv.index("--rule-knobs") + 1] = KNOBS.replace("flank=50", "flank=20")
    with pytest.raises(P.PrecisionError, match="are not v2's recorded"):
        _cli(powered_case["root"], argv)


def test_the_hardneg_mint_cli_refuses_before_reading_anything(tmp_path, monkeypatch):
    """The output guard and the flag checks run before any input is read: every input path
    below is absent, so a refusal naming the guard proves the guard ran first."""
    import sys

    monkeypatch.chdir(tmp_path)
    base = ["mint", "--hardneg", "reports/p3/absent.json", "--data-root", str(tmp_path)]
    cases = [
        (["--hardneg-items", "a.json", "b.json", "--out", f"{P3_16}/benchmark_v2.json"], "v2"),
        (["--hardneg-items", "a.json", "b.json", "--out", f"{P3_16}/benchmark_v3.json"], "hardneg"),
        (["--out", f"{P3_16}/hardneg_benchmark.json"], "--hardneg-items GRADED BESIDE"),
        (
            ["--hardneg-items", "a.json", "b.json", "--hardneg-census-items", "c.json", "d.json"]
            + ["--out", f"{P3_16}/hardneg_benchmark.json"],
            "without --hardneg-census",
        ),
    ]
    for extra, needle in cases:
        monkeypatch.setattr(sys, "argv", [*base, *extra])
        with pytest.raises(SystemExit, match=needle):
            MINT.main()
    monkeypatch.setattr(sys, "argv", ["mint", "--out", f"{P3_16}/benchmark_v9.json"])
    with pytest.raises(SystemExit):
        MINT.main()  # the v1/v2 mint still requires --dataset and --decoys


def test_a_v2_fold_refuses_an_a17_input(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "precision_gain_v2_fixture", REPO / "tests" / "unit" / "test_precision_gain.py"
    )
    V2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V2)  # the v2 fixture, built by its own module
    case = V2._build_v2(tmp_path / "v2")
    del case  # the build itself proves the v2 fold path still runs; now hand it an A17 input
    root = (tmp_path / "v2").resolve()
    folded = json.loads((root / V2.V2_ITEMS_REL).read_text())
    assert folded["schema_version"] == P.SCHEMA_VERSION and not P.folded_is_hardneg(folded)
    bench = root / "data/interim/p3_16/benchmark_v2.json"
    argv = ["fold", "--benchmark", str(bench), "--out", str(root / "reports/p3/x_v2.json")]
    argv += ["--twin-table", str(root / "data/interim/p3_16/v2_table_twin_t0.5.json")]
    argv += ["--production-table", str(root / "data/interim/p3_16/v2_table_production_t0.5.json")]
    argv += ["--twin-checkpoint", "a", "--production-checkpoint", "b", "--stage1-threshold", "0.5"]
    argv += ["--hardneg-pool", str(bench)]
    with pytest.raises(P.PrecisionError, match="not ADR-0005 A17's hard-negative benchmark"):
        _cli(root, argv)


def test_a_half_declared_hardneg_manifest_is_refused():
    with pytest.raises(P.PrecisionError, match="carries schema >= 1.2 AND kind"):
        P.manifest_is_hardneg({"schema_version": "1.2"})
    with pytest.raises(P.PrecisionError, match="carries schema >= 1.2 AND kind"):
        P.manifest_is_hardneg({"schema_version": "1.1", "kind": P.HARDNEG_MANIFEST_KIND})
    assert not P.manifest_is_hardneg({"schema_version": "1.1"})


@pytest.mark.parametrize(
    "rel",
    [
        "reports/two_stage_precision_v2.json",
        "reports/p3/two_stage_precision_items_v2.json",
        "data/interim/p3_16/v2_items_t0.7.json",
        "reports/two_stage_precision.json",
        "reports/two_stage_precision_other.json",
    ],
)
def test_an_a17_output_is_never_a_v1_v2_or_unnamed_path(rel):
    with pytest.raises((P.PrecisionError, TS.TwoStageError)):
        P.refuse_hardneg_output(rel)
    P.refuse_hardneg_output("reports/two_stage_precision_hardneg.json")


# ══════════════════════════════════════════════════════════════════════════════════════
# The CLI end to end, totality, and the figure
# ══════════════════════════════════════════════════════════════════════════════════════
def _run_report(case: dict, monkeypatch) -> int:
    root = case["root"]
    (root / "envs").mkdir(exist_ok=True)
    shutil.copyfile(REPO / "envs" / "data.conda-lock.yml", root / "envs" / "data.conda-lock.yml")
    monkeypatch.chdir(root)
    monkeypatch.setattr(P, "REPO_ROOT", root)
    monkeypatch.setattr(
        P, "repo_relative", lambda path: str(Path(path).resolve().relative_to(root))
    )
    argv = ["report", "--items", ITEMS_REL, "--out", OUT_REL, "--stage2-operating-point", "0.5"]
    return P.main([*argv, "--n-boot", str(N_BOOT)])


def test_the_report_leg_writes_the_a17_report_canonically(tmp_path, monkeypatch):
    case = build_census(tmp_path / "census")
    rc = _run_report(case, monkeypatch)
    written = json.loads((case["root"] / OUT_REL).read_text())
    assert written["problems"] == [] and rc == (0 if written["gate"]["passes"] else 4)
    assert written["gate"]["verdict_class"] == "underpowered"
    assert "precision_comparison_hardneg" in written["provenance"]["rule"]


def test_the_report_leg_refuses_a_v2_output_for_an_a17_table(powered_case, monkeypatch):
    root = powered_case["root"]
    monkeypatch.chdir(root)
    argv = [
        "report",
        "--items",
        str(root / ITEMS_REL),
        "--out",
        str(root / "reports/two_stage_precision_v2.json"),
    ]
    with pytest.raises(P.PrecisionError, match="v2 artifact name"):
        P.main([*argv, "--stage2-operating-point", "0.5", "--n-boot", str(N_BOOT)])


def test_no_retyped_a17_node_can_make_the_checker_raise(hn_census):
    """The 1.3 clause set returns problems on a payload it did not write; it never raises."""
    bad_values = [None, "x", -1, 1.5, [], {}, True, 10**400]
    paths: list[tuple] = []

    def walk(node, path, depth):
        if depth > 4:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                paths.append((*path, key))
                walk(value, (*path, key), depth + 1)

    for key in (
        "power",
        "population",
        "beside_gate",
        "round0_crosstab",
        "v2_positive_delta",
        "composition",
        "blocks",
        "sources",
    ):
        walk(hn_census[key], (key,), 1)
        paths.append((key,))
    checked = 0
    for path in paths:
        for bad in bad_values:
            broken = copy.deepcopy(hn_census)
            node = broken
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = bad
            P.precision_problems(broken)
            checked += 1
    assert checked > 1_000, checked


def test_the_figure_reads_the_a17_fields(hn, hn_census):
    spec = importlib.util.spec_from_file_location(
        "plot_hardneg", REPO / "scripts" / "plot_two_stage_precision.py"
    )
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    assert plot.underpowered_note(hn_census) == P.HARDNEG_UNDERPOWERED_STATEMENT
    assert plot.underpowered_note(hn) is None
    assert plot.familiarity_point(hn) is None  # not supplied for this run
    assert plot.beside_point(hn) == (
        100.0,
        hn["beside_gate"]["by_arm"]["twin"]["with_beside_in_denominator"]["auprc"],
    )
    assert (
        plot.beside_point(json.loads((REPO / "reports/two_stage_precision_v2.json").read_text()))
        is None
    )
    assert plot.power_note(hn_census).startswith("P = 5")
    assert plot.power_note(hn).startswith("P = 21")
