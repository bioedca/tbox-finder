"""P3-17a — the CM + learned-calibration Stage-2 confirmer (PRD §6 ablation (ii)).

Three things the step names are pinned here: the ``cmsearch`` tblout parse (on real Infernal
1.1.5 output), the learned calibration being **monotone** in bit score, and the confirmer
scoring **RNA sequence only**. The committed report is then validated and tampered with.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from tbox_finder import infernal, metrics
from tbox_finder.integration import precision as P
from tbox_finder.stage2 import cm_confirmer as C

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO / "tests/fixtures/cm_confirmer"
_SEARCH = _FIXTURE / "search"
_REPORT = _REPO / "reports/cm_confirmer_ablation.json"
_SIDECAR = _REPO / "reports/p3/cm_confirmer_scores.json"
_SIDECAR_LOO = _REPO / "reports/p3/cm_confirmer_scores_loo.json"
_RIN_SIDECAR = _REPO / "reports/p3/stage2_scores.json"
_RIN_SIDECAR_LOO = _REPO / "reports/p3/stage2_scores_loo.json"


def _spec() -> dict:
    return json.loads((_FIXTURE / "queries.json").read_text(encoding="utf-8"))


# ── RNA sequence only ────────────────────────────────────────────────────────────────────
def test_normalise_rna_upper_cases_and_keeps_ambiguity_codes():
    assert C.normalise_rna(" acgun ") == "ACGUN"


@pytest.mark.parametrize("bad", ["ACGT", "AC-GU", "AC GU", "ACGU*", ""])
def test_normalise_rna_refuses_anything_that_is_not_rna(bad):
    with pytest.raises(C.ConfirmerError):
        C.normalise_rna(bad)


def test_query_id_is_the_content_address_of_the_normalised_rna():
    assert C.query_id("acgu") == C.query_id("ACGU") == hashlib.sha256(b"ACGU").hexdigest()


def test_build_queries_reads_the_rna_and_nothing_else(tmp_path):
    """Relabelling every row must leave the query shards byte-identical."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    recs = _spec()["records"]
    payloads = tmp_path / "payloads.json"
    # One payload duplicates a dataset row: the confirmer must score it once.
    payloads.write_text(
        json.dumps({"payloads": [{"row_id": "p0", "rna_sequence": recs[0]["rna_sequence"]}]}),
        encoding="utf-8",
    )
    shards = []
    for flip in (False, True):
        frame = pd.DataFrame(
            {
                "row_id": [r["row_id"] for r in recs],
                "rna_sequence": [r["rna_sequence"] for r in recs],
                "is_tbox": [bool(r["is_tbox"]) != flip for r in recs],
                "pool": [r["pool"] if not flip else "x" for r in recs],
            }
        )
        dataset = tmp_path / f"d{int(flip)}.parquet"
        frame.to_parquet(dataset, index=False)
        out = tmp_path / f"q{int(flip)}"
        manifest = C.build_queries(dataset=dataset, payloads=payloads, out_dir=out, n_shards=2)
        assert manifest["n_queries"] == len({r["rna_sequence"] for r in recs})
        shards.append([(out / s).read_bytes() for s in manifest["shards"]])
    assert shards[0] == shards[1]


def test_write_query_shards_refuses_a_query_that_is_not_its_own_address(tmp_path):
    with pytest.raises(C.ConfirmerError, match="content address"):
        C.write_query_shards({"0" * 64: "ACGU"}, tmp_path, n_shards=1, sources={})
    # positive control: the honest key is accepted
    C.write_query_shards({C.query_id("ACGU"): "ACGU"}, tmp_path, n_shards=1, sources={})


def test_search_flags_score_every_query_on_the_given_strand():
    flags = C.search_flags()
    assert "--toponly" in flags and "--max" in flags
    assert flags[flags.index("-T") + 1] == repr(C.SEARCH_THRESHOLD_BITS)
    assert "--cut_ga" not in flags


def test_run_cmsearch_refuses_two_reporting_thresholds(tmp_path):
    """``-T`` and ``--cut_ga`` both set the reporting threshold; the wrapper refuses both."""
    with pytest.raises(ValueError, match="reporting threshold"):
        infernal.run_cmsearch(
            "x.cm", "x.fa", tmp_path / "x.tbl", cut_ga=True, score_threshold=C.SEARCH_THRESHOLD_BITS
        )


# ── the cmsearch parse, on real Infernal 1.1.5 output ────────────────────────────────────
def _direct_best(cm: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for tbl in sorted((_SEARCH / "tblout" / cm).glob("*.tblout")):
        for hit in infernal.parse_tblout(tbl.read_text(encoding="utf-8")):
            out.setdefault(hit.target, []).append(hit.score)
    return out


def test_read_search_keeps_each_querys_best_hit_per_model():
    searched = C.read_search(query_dir=_SEARCH / "queries", tblout_dir=_SEARCH / "tblout")
    assert len(searched["queries"]) == len(_spec()["records"])
    for cm, _ in C.CONFIRMER_CMS:
        direct = _direct_best(cm)
        # The fixture must make "max" matter: a query with several distinct hit scores.
        assert any(len(set(v)) > 1 for v in direct.values())
        assert searched["bits"][cm] == {q: max(v) for q, v in direct.items()}
    table = C.score_table(searched)
    for qid, row in table.items():
        assert row["best"] == max(searched["bits"][cm][qid] for cm, _ in C.CONFIRMER_CMS)


def test_real_scores_rank_every_fixture_t_box_above_every_decoy():
    """PRD §5's circularity, measured: CM-derived positives versus non-T-box decoys."""
    table = C.score_table(
        C.read_search(query_dir=_SEARCH / "queries", tblout_dir=_SEARCH / "tblout")
    )
    best = {r["is_tbox"]: [] for r in _spec()["records"]}
    for r in _spec()["records"]:
        best[r["is_tbox"]].append(table[C.query_id(r["rna_sequence"])]["best"])
    assert min(best[True]) > max(best[False])


def _copy_search(tmp_path: Path) -> Path:
    dst = tmp_path / "search"
    shutil.copytree(_SEARCH, dst)
    return dst


def test_read_search_refuses_a_hit_for_a_query_it_did_not_search(tmp_path):
    dst = _copy_search(tmp_path)
    tbl = dst / "tblout/RF00230/shard_000.tblout"
    rows = [ln for ln in tbl.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    forged = "f" * 64 + rows[0][rows[0].index(" ") :]
    tbl.write_text(tbl.read_text(encoding="utf-8") + forged + "\n", encoding="utf-8")
    # A CONSISTENT forge: DONE.json re-stamped, so the digest and count checks pass and the
    # per-shard query binding is what has to refuse.
    done_path = dst / "tblout/DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    rec = done["tblouts"]["RF00230"]["shard_000.fa"]
    rec["n_hits"] += 1
    rec["data_sha256"] = C.tblout_digest(tbl.read_text(encoding="utf-8"))
    done_path.write_text(json.dumps(done), encoding="utf-8")
    with pytest.raises(C.ConfirmerError, match="not a query"):
        C.read_search(query_dir=dst / "queries", tblout_dir=dst / "tblout")


def test_read_search_refuses_tblouts_from_other_flags_or_another_manifest(tmp_path):
    dst = _copy_search(tmp_path)
    done_path = dst / "tblout/DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    C.read_search(query_dir=dst / "queries", tblout_dir=dst / "tblout")  # positive control
    bad = {**done, "flags": [f for f in done["flags"] if f != "--max"]}
    done_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(C.ConfirmerError, match="ran with"):
        C.read_search(query_dir=dst / "queries", tblout_dir=dst / "tblout")
    done_path.write_text(json.dumps({**done, "query_manifest_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(C.ConfirmerError, match="different query manifest"):
        C.read_search(query_dir=dst / "queries", tblout_dir=dst / "tblout")
    done_path.unlink()
    with pytest.raises(C.ConfirmerError, match="did not finish"):
        C.read_search(query_dir=dst / "queries", tblout_dir=dst / "tblout")


def test_calibration_fit_ids_are_exactly_the_calib_rung():
    sidecar = {"row_ids": ["a", "b", "c", "d"], "rungs": ["calib", "test", "calib", "val"]}
    assert C.calibration_fit_ids(sidecar) == ["a", "c"]


@pytest.mark.parametrize("edit", ["drop", "rescore"])
def test_read_search_refuses_a_dropped_or_edited_hit_row(tmp_path, edit):
    """A dropped row can turn a T-box's best hit from 100 bits into a decoy-like score."""
    dst = _copy_search(tmp_path)
    tbl = dst / "tblout/RF00230/shard_000.tblout"
    lines = tbl.read_text(encoding="utf-8").splitlines()
    first = next(i for i, ln in enumerate(lines) if ln and not ln.startswith("#"))
    if edit == "drop":
        del lines[first]
    else:
        fields = lines[first].split()
        lines[first] = lines[first].replace(f" {fields[14]} ", " 999.9 ", 1)
        assert lines[first].split()[14] == "999.9"
    tbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(C.ConfirmerError, match="hits parsed|data rows differ"):
        C.read_search(query_dir=dst / "queries", tblout_dir=dst / "tblout")


def test_read_search_refuses_tblouts_from_another_covariance_model(tmp_path):
    """DONE.json carries each model's content digest; a model changed since the search refuses."""
    dst = _copy_search(tmp_path)
    done_path = dst / "tblout/DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    assert set(done["cms"]) == {name for name, _ in C.CONFIRMER_CMS}
    C.read_search(query_dir=dst / "queries", tblout_dir=dst / "tblout")  # positive control
    done["cms"]["TBDB001"]["sha256"] = "0" * 64
    done_path.write_text(json.dumps(done), encoding="utf-8")
    with pytest.raises(C.ConfirmerError, match="different covariance models"):
        C.read_search(query_dir=dst / "queries", tblout_dir=dst / "tblout")


@pytest.mark.skipif(not infernal.cmsearch_available(), reason="cmsearch (infernal env) not on PATH")
def test_search_reproduces_the_fixture_and_is_shard_invariant(tmp_path):
    """Re-run the pinned search: same best scores, whether the queries sit in 1 or 3 shards."""
    queries = C.collect_queries([r["rna_sequence"] for r in _spec()["records"]])
    committed = C.score_table(
        C.read_search(query_dir=_SEARCH / "queries", tblout_dir=_SEARCH / "tblout")
    )
    cms = tuple((name, _REPO / cm) for name, cm in C.CONFIRMER_CMS)
    for n_shards in (1, 3):
        qdir, tdir = tmp_path / f"q{n_shards}", tmp_path / f"t{n_shards}"
        C.write_query_shards(queries, qdir, n_shards=n_shards, sources={})
        C.search(query_dir=qdir, out_dir=tdir, jobs=2, cms=cms)
        again = C.score_table(C.read_search(query_dir=qdir, tblout_dir=tdir, cms=cms))
        assert again == committed


# ── the learned calibration ──────────────────────────────────────────────────────────────
def test_platt_targets_are_platts_smoothed_targets():
    t, n_pos, n_neg, t_pos, t_neg = C.platt_targets([1, 1, 1, 0])
    assert (n_pos, n_neg) == (3, 1)
    assert t_pos == 4 / 5 and t_neg == 1 / 3
    assert list(t) == [t_pos, t_pos, t_pos, t_neg]


def _overlapping(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = rng.normal(60.0, 30.0, 400)
    neg = rng.normal(0.0, 6.0, 300)
    return np.concatenate([pos, neg]), np.concatenate([np.ones(400, int), np.zeros(300, int)])


def _gradient(s: np.ndarray, y: np.ndarray, fit: C.PlattFit) -> np.ndarray:
    t, *_ = C.platt_targets(y)
    p = 1.0 / (1.0 + np.exp(-(fit.slope * s + fit.intercept)))
    return np.array([np.dot(p - t, s), np.sum(p - t)])


def test_fit_platt_lands_on_the_stationary_point_of_platts_objective():
    s, y = _overlapping()
    fit = C.fit_platt(s, y)
    assert fit.converged and fit.slope > 0
    g = _gradient(s, y, fit)
    assert np.all(np.abs(g) <= 1e-6 * np.array([np.abs(s).sum(), s.size]))


def test_fit_platt_stays_finite_under_perfect_separation():
    """The smoothed targets are why a CM that separates its own positives still calibrates."""
    s = np.array([-3.0, -2.0, -1.5, 0.5, 90.0, 100.0, 110.0, 120.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    fit = C.fit_platt(s, y)
    assert fit.converged and math.isfinite(fit.slope) and math.isfinite(fit.intercept)
    assert np.all(np.abs(_gradient(s, y, fit)) <= 1e-6 * np.array([np.abs(s).sum(), s.size]))
    p = 1.0 / (1.0 + np.exp(-np.array(C.platt_logit(s, fit))))
    # The fit honours Platt's soft targets — written out here, not read back from the
    # module, so a module that reverted to hard 0/1 targets cannot also move the expectation.
    assert p[y == 1].mean() == pytest.approx(5 / 6, abs=0.02)
    assert p[y == 0].mean() == pytest.approx(1 / 6, abs=0.02)


def test_learned_calibration_is_monotone_in_bit_score():
    s, y = _overlapping(1)
    fit = C.fit_platt(s, y)
    grid = np.linspace(-1000.0, 200.0, 2001)
    z = np.array(C.platt_logit(grid, fit))
    assert np.all(np.diff(z) > 0)


def test_fit_platt_refuses_a_score_that_ranks_backwards():
    s, y = _overlapping(2)
    C.fit_platt(s, y)  # positive control: the honest orientation fits
    with pytest.raises(C.ConfirmerError, match="monotone"):
        C.fit_platt(-s, y)


@pytest.mark.parametrize(
    ("scores", "labels", "match"),
    [
        ([1.0, 2.0], [1, 1], "both classes"),
        ([5.0, 5.0], [0, 1], "identical"),
        ([1.0, math.nan], [0, 1], "non-finite"),
        ([1.0, 2.0], [0, 2], "0/1"),
        ([1.0], [0, 1], "equal-length"),
    ],
)
def test_fit_platt_refuses_degenerate_input(scores, labels, match):
    with pytest.raises(C.ConfirmerError, match=match):
        C.fit_platt(scores, labels)


# ── the A14 grade ─────────────────────────────────────────────────────────────────────────
def test_platt_posterior_is_the_logistic_of_the_logit():
    z = [-30.0, -1.0, 0.0, 2.5, 40.0]
    assert C.platt_posterior(z) == pytest.approx([1 / (1 + math.exp(-v)) for v in z], abs=1e-15)


def _a14_scores(seed: int = 3) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    n = 400
    labels = (rng.random(n) < 0.7).astype(int)
    logits = np.where(labels == 1, rng.normal(3.0, 2.0, n), rng.normal(-3.0, 2.0, n))
    rungs = ["calib" if i % 4 == 0 else "test" for i in range(n)]
    ids = [f"r{i:04d}" for i in range(n)]
    scores = {"row_ids": ids, "labels": labels.tolist(), "logits": logits.tolist(), "rungs": rungs}
    return scores, {r: f"cluster:{i // 3}" for i, r in enumerate(ids)}


def test_a14_grade_is_d11s_estimator_on_the_test_rung_of_the_platt_posterior():
    scores, blocks = _a14_scores()
    out = C.grade_in_distribution_a14(scores=scores, blocks_by_row=blocks, n_boot=50, seed=1)
    test = [i for i, g in enumerate(scores["rungs"]) if g == "test"]
    y = [scores["labels"][i] for i in test]
    p = C.platt_posterior([scores["logits"][i] for i in test])
    assert out["ece"] == metrics.binned_ece(y, p, 15, debias=True)
    assert out["n"] == len(test)
    assert out["graded_posterior_key"] == C.POSTERIOR_KEY
    assert out["temperature_stage"] == "not_applied" and out["gated"] is False
    assert "passes" not in out and "calibration" not in out
    # The calib rows are the calibrator's fit set; they must not move the graded number.
    moved = copy.deepcopy(scores)
    for i, g in enumerate(moved["rungs"]):
        if g == "calib":
            moved["logits"][i] = -moved["logits"][i]
    again = C.grade_in_distribution_a14(scores=moved, blocks_by_row=blocks, n_boot=50, seed=1)
    assert again["ece"] == out["ece"]


# ── the benchmark comparison ─────────────────────────────────────────────────────────────
def _items(n: int = 60) -> list[dict]:
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        label = int(i % 3 != 0)
        out.append(
            {
                "contig_id": f"c{i:03d}",
                "label": label,
                "pool": "corpus" if label else "structured_rna",
                "block": f"cluster:{i // 2}",
                "seen_by": {"twin": False, "production": False},
                "n_rows": 1,
                "rinalmo": float(np.clip(0.8 * label + rng.normal(0.1, 0.15), 0, 1)),
                C.ARM: float(np.clip(0.4 * label + rng.normal(0.3, 0.25), 0, 1)),
            }
        )
    return out


def _walk_keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_keys(v)


def test_benchmark_metrics_gives_each_system_its_own_scores():
    """Identity, not counts: a swapped slot would pass any symmetric check."""
    items = _items()
    m = C.benchmark_metrics(items, operating_point=0.5, n_boot=50, seed=42)
    y = [it["label"] for it in items]
    rin = metrics.average_precision(y, [it["rinalmo"] for it in items])
    cm = metrics.average_precision(y, [it[C.ARM] for it in items])
    assert rin != cm
    assert m["auprc"] == {"rinalmo": rin, C.ARM: cm}
    assert m["auprc_gain_pp"] == pytest.approx(
        m["prevalence"]["100:1"]["auprc"]["rinalmo"] * 100
        - m["prevalence"]["100:1"]["auprc"][C.ARM] * 100
    )
    keys = set(_walk_keys(m))
    assert not any("two_stage" in k or "stage1_only" in k for k in keys)
    assert "passes" not in m and "gated" not in m


def _rows_and_committed() -> tuple[list[dict], dict]:
    rows = [
        {"contig_id": "a", "payload_key": "r1", "peak_p_elem": 0.9, "stage2_named_posterior": 0.8},
        {"contig_id": "a", "payload_key": "r2", "peak_p_elem": 0.7, "stage2_named_posterior": 0.95},
        {"contig_id": "b", "payload_key": "r3", "peak_p_elem": 0.6, "stage2_named_posterior": 0.1},
    ]
    base = {"label": 1, "pool": "corpus", "block": "cluster:1", "seen_by": {"twin": False}}
    committed = {
        "arms": ["twin"],
        "items": [
            {
                "contig_id": "a",
                **base,
                "arms": {"twin": {"n_rows": 2, "stage1_only": 0.9, "two_stage": 0.95}},
            },
            {
                "contig_id": "b",
                **base,
                "label": 0,
                "arms": {"twin": {"n_rows": 1, "stage1_only": 0.6, "two_stage": 0.1}},
            },
            {
                "contig_id": "c",
                **base,
                "arms": {
                    "twin": {
                        "n_rows": 0,
                        "stage1_only": P.UNCALLED_SCORE,
                        "two_stage": P.UNCALLED_SCORE,
                    }
                },
            },
        ],
    }
    return rows, committed


def test_benchmark_items_swaps_only_the_re_ranker():
    rows, committed = _rows_and_committed()
    cm = {"r1": 0.2, "r2": 0.6, "r3": 0.05}
    items, binding = C.benchmark_items(
        committed=committed, rows=rows, cm_posterior_by_row=cm, arm="twin"
    )
    by = {it["contig_id"]: it for it in items}
    assert by["a"]["rinalmo"] == 0.95 and by["a"][C.ARM] == 0.6  # max over the item's rows
    assert by["b"][C.ARM] == 0.05
    assert by["c"]["rinalmo"] == by["c"][C.ARM] == P.UNCALLED_SCORE
    assert binding["n_items_mismatched_vs_committed"] == 0


def test_benchmark_items_refuses_a_replay_that_does_not_reproduce_p3_16():
    rows, committed = _rows_and_committed()
    bad = copy.deepcopy(committed)
    bad["items"][1]["arms"]["twin"]["two_stage"] = 0.11
    with pytest.raises(C.ConfirmerError, match="does not reproduce"):
        C.benchmark_items(
            committed=bad, rows=rows, cm_posterior_by_row={"r1": 0, "r2": 0, "r3": 0}, arm="twin"
        )
    with pytest.raises(C.ConfirmerError, match="no CM-confirmer posterior"):
        C.benchmark_items(committed=committed, rows=rows, cm_posterior_by_row={"r1": 0}, arm="twin")


def test_tree_close_tolerates_float_noise_and_nothing_else():
    base = {"a": [1.0, 2, "x", True], "b": {"c": 0.1}}
    assert C._tree_close(base, {"a": [1.0 + 1e-13, 2, "x", True], "b": {"c": 0.1}})
    for bad in (
        {"a": [1.001, 2, "x", True], "b": {"c": 0.1}},  # beyond the tolerance
        {"a": [1.0, 3, "x", True], "b": {"c": 0.1}},  # an int is exact
        {"a": [1.0, 2, "y", True], "b": {"c": 0.1}},  # a string is exact
        {"a": [1.0, 2, "x", False], "b": {"c": 0.1}},  # a bool is exact
        {"a": [1.0, 2, "x", True], "b": {"c": 0.1, "d": 1}},  # keys are exact
        {"a": [1.0, 2, "x"], "b": {"c": 0.1}},  # lengths are exact
        {"a": [1.0, 2.0000000001, "x", True], "b": {"c": 0.1}},  # a count stays an int
        {"a": [1, 2, "x", True], "b": {"c": 0.1}},  # a float stays a float
    ):
        assert not C._tree_close(base, bad), bad


def test_relabel_refuses_a_key_collision():
    """Two source keys that rename to one target must not silently overwrite each other."""
    C._relabel({"fp_two_stage": 1, "fp_x": 2})  # positive control: distinct targets
    with pytest.raises(C.ConfirmerError, match="collides"):
        C._relabel({"fp_two_stage": 1, "fp_rinalmo": 2})


def test_relabel_names_the_cm_confirmer_in_the_stage1_slot():
    assert C._relabel({"fp_two_stage": 1, "x": [{"stage1_only": 2}]}) == {
        "fp_rinalmo": 1,
        "x": [{C.ARM: 2}],
    }


# ── the committed artifacts ──────────────────────────────────────────────────────────────
_committed = pytest.mark.skipif(not _REPORT.exists(), reason="no committed confirmer report")


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(_REPORT.read_text(encoding="utf-8"))


@_committed
def test_committed_report_validates(report):
    assert C.validate_report(report, repo_root=_REPO, require_all_inputs=False) == []
    assert report["is_science"] is True
    assert report["gated"] is False and report["role"] == "ablation"
    assert set(report["clauses"]) == set(C.CLAUSES)


@_committed
def test_committed_calibrator_is_fitted_on_calib_and_grades_the_a14_posterior(report):
    """ADR-0005 A14: the fit set is the whole calib rung and nothing it fitted is graded."""
    conf = report["confirmer"]
    rin = json.loads(_RIN_SIDECAR.read_text(encoding="utf-8"))
    n_calib = sum(1 for g in rin["rungs"] if g == "calib")
    assert conf["calibrator"]["fitted_on"] == "calib"
    assert conf["n_fit_rows"] == conf["n_calib_rows_in_sidecar"] == n_calib
    assert conf["n_fit_rows_also_graded"] == 0
    assert conf["calibrator"]["slope"] > 0 and conf["calibrator"]["converged"] is True
    graded = report["calibration"]["in_distribution"][C.ARM]
    assert graded["graded_posterior_key"] == C.POSTERIOR_KEY
    assert "temperature" not in graded and "calibration" not in graded


@_committed
def test_committed_sidecars_are_rinalmos_population_and_the_calibrators_logits(report):
    fit = report["confirmer"]["calibrator"]
    for ours_path, theirs_path, fields in (
        (_SIDECAR, _RIN_SIDECAR, ("row_ids", "labels", "rungs", "dataset_sha256")),
        (
            _SIDECAR_LOO,
            _RIN_SIDECAR_LOO,
            ("row_ids", "labels", "units", "blocks", "dataset_sha256"),
        ),
    ):
        ours = json.loads(ours_path.read_text(encoding="utf-8"))
        theirs = json.loads(theirs_path.read_text(encoding="utf-8"))
        for field in fields:
            assert ours[field] == theirs[field], field
        arm = ours["arms"][C.ARM]
        want = np.asarray(arm["bit_scores"]) * fit["slope"] + fit["intercept"]
        assert np.array_equal(np.asarray(arm["logits"]), want)


@_committed
@pytest.mark.parametrize(
    "clause",
    [
        "circularity_disclosed",
        "calibrator_monotone",
        "replay_reproduces_p3_16_items",
        "graded_object_is_the_a14_posterior",
        "calibrator_fit_on_the_whole_calib_rung",
    ],
)
def test_a_flipped_clause_is_caught(report, clause):
    bad = copy.deepcopy(report)
    bad["clauses"][clause] = not bad["clauses"][clause]
    assert C.validate_report(
        bad, repo_root=_REPO, require_all_inputs=False, rederive_precision=False
    )


@_committed
def test_dropping_the_circularity_disclosure_is_caught(report):
    bad = copy.deepcopy(report)
    bad["disclosures"] = [d for d in bad["disclosures"] if d["id"] != "cm_derived_positives"]
    problems = C.validate_report(
        bad, repo_root=_REPO, require_all_inputs=False, rederive_precision=False
    )
    assert any("circularity_disclosed" in p for p in problems)


@_committed
def test_a_negated_slope_is_caught(report):
    bad = copy.deepcopy(report)
    bad["confirmer"]["calibrator"]["slope"] = -abs(bad["confirmer"]["calibrator"]["slope"])
    problems = C.validate_report(
        bad, repo_root=_REPO, require_all_inputs=False, rederive_precision=False
    )
    assert any("calibrator_monotone" in p for p in problems)


@_committed
def test_an_edited_precision_number_does_not_re_derive(report):
    bad = copy.deepcopy(report)
    bad["precision"]["metrics"]["auprc_gain_pp"] += 1.0
    problems = C.validate_report(bad, repo_root=_REPO, require_all_inputs=False)
    assert any("re-derive" in p for p in problems)


@_committed
def test_an_edited_sidecar_breaks_the_hash_binding(report, tmp_path):
    root = _repo_copy(tmp_path)
    assert (
        C.validate_report(
            report, repo_root=root, require_all_inputs=False, rederive_precision=False
        )
        == []
    )
    side = root / "reports/p3/cm_confirmer_scores.json"
    payload = json.loads(side.read_text(encoding="utf-8"))
    payload["arms"][C.ARM]["logits"][0] += 1.0
    side.write_text(json.dumps(payload), encoding="utf-8")
    problems = C.validate_report(
        report, repo_root=root, require_all_inputs=False, rederive_precision=False
    )
    assert any("cm_confirmer_scores.json" in p for p in problems)


@_committed
def test_an_lfs_pointer_is_checked_through_its_oid(report, tmp_path):
    """CI has no git-LFS: a committed CM is a pointer there, and its oid IS the content hash."""
    root = tmp_path / "repo"
    inputs = report["provenance"]["inputs"]
    for rel in inputs:
        if rel.startswith("reports/") and (_REPO / rel).exists():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_REPO / rel, root / rel)
    cms = [rel for rel in inputs if rel.endswith(".cm")]
    assert len(cms) == 2

    def pointer(oid: str) -> str:
        return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 1\n"

    for rel in cms:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(pointer(inputs[rel]), encoding="utf-8")
    kw = {"repo_root": root, "require_all_inputs": False, "rederive_precision": False}
    assert C.validate_report(report, **kw) == []
    (root / cms[0]).write_text(pointer("0" * 64), encoding="utf-8")
    assert any(cms[0] in p for p in C.validate_report(report, **kw))


def _repo_copy(tmp_path: Path) -> Path:
    """The committed upstream files a CI checkout carries, copied under a scratch root."""
    root = tmp_path / "repo"
    for rel in C.CANONICAL_SOURCES.values():
        if rel.startswith(C.COMMITTED_PREFIXES) and (_REPO / rel).exists():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_REPO / rel, root / rel)
    return root


_CI = {"require_all_inputs": False, "rederive_precision": False}


@_committed
@pytest.mark.parametrize(
    ("where", "clause"),
    [
        (("calibration", "in_distribution", C.ARM, "ece"), "cm_in_distribution_ece_rederives"),
        (
            ("calibration", "in_distribution", "rinalmo", "ece"),
            "rinalmo_values_are_its_committed_report",
        ),
        (
            ("calibration", "leave_clade_out", "paired", "ci", "point"),
            "cm_macro_and_paired_difference_rederive",
        ),
        (
            ("calibration", "leave_clade_out", C.ARM, "macro_average", "point"),
            "cm_macro_and_paired_difference_rederive",
        ),
        (("confirmer", "calibrator", "intercept"), "calibrator_refits_from_the_committed_sidecar"),
        (
            ("precision", "p3_16_reference", "target_recall"),
            "rinalmo_benchmark_reproduces_p3_16_report",
        ),
    ],
)
def test_a_forged_number_with_its_clauses_left_true_is_caught(report, tmp_path, where, clause):
    """The reviewer's forge: edit a published number, keep every recorded clause TRUE."""
    root = _repo_copy(tmp_path)
    assert C.validate_report(report, repo_root=root, **_CI) == []  # positive control
    bad = copy.deepcopy(report)
    node = bad
    for key in where[:-1]:
        node = node[key]
    node[where[-1]] = float(node[where[-1]]) * 0.5 + 1e-3
    problems = C.validate_report(bad, repo_root=root, **_CI)
    assert any(clause in p for p in problems), problems


@_committed
def test_a_forged_rinalmo_item_score_is_caught_even_with_its_hash_restamped(report, tmp_path):
    from tbox_finder import refs

    root = _repo_copy(tmp_path)
    items_path = root / C.CANONICAL_SOURCES["out_items"]
    payload = json.loads(items_path.read_text(encoding="utf-8"))
    for item in payload["items"]:
        if item["label"] == 0 and item["rinalmo"] > 0.9:
            item["rinalmo"] = 0.0
    items_path.write_text(json.dumps(payload), encoding="utf-8")
    bad = copy.deepcopy(report)
    bad["provenance"]["inputs"][C.CANONICAL_SOURCES["out_items"]] = refs.content_sha256(items_path)
    problems = C.validate_report(bad, repo_root=root, **_CI)
    assert any("replay_reproduces_p3_16_items" in p for p in problems), problems


@_committed
def test_a_rinalmo_sidecar_other_than_the_graded_one_is_caught(report, tmp_path):
    from tbox_finder import refs

    root = _repo_copy(tmp_path)
    side_path = root / C.CANONICAL_SOURCES["rinalmo_scores"]
    side = json.loads(side_path.read_text(encoding="utf-8"))
    side["generated_by"] = "a different producer"
    side_path.write_text(json.dumps(side), encoding="utf-8")
    bad = copy.deepcopy(report)
    key = C.CANONICAL_SOURCES["rinalmo_scores"]
    bad["provenance"]["inputs"][key] = refs.content_sha256(side_path)
    problems = C.validate_report(bad, repo_root=root, **_CI)
    assert any("rinalmo_sidecars_are_the_graded_ones" in p for p in problems), problems


@_committed
def test_a_calibrator_fitted_on_the_test_rung_is_caught(report, tmp_path):
    """The mutant the reviewer ran: Platt fitted on the graded rows instead of calib."""
    from tbox_finder import refs

    root = _repo_copy(tmp_path)
    side_path = root / C.CANONICAL_SOURCES["out_scores"]
    side = json.loads(side_path.read_text(encoding="utf-8"))
    arm = side["arms"][C.ARM]
    test = [i for i, g in enumerate(side["rungs"]) if g == "test"]
    wrong = C.fit_platt([arm["bit_scores"][i] for i in test], [side["labels"][i] for i in test])
    arm["logits"] = C.platt_logit(arm["bit_scores"], wrong)
    side_path.write_text(json.dumps(side), encoding="utf-8")
    # Forge the LOO sidecar consistently too, so `a*bits + b` holds everywhere and ONLY the
    # re-fit on the calib rows can tell ([[all-true-fixture-cannot-test-a-conjunction]]).
    loo_path = root / C.CANONICAL_SOURCES["out_loo_scores"]
    loo = json.loads(loo_path.read_text(encoding="utf-8"))
    loo["arms"][C.ARM]["logits"] = C.platt_logit(loo["arms"][C.ARM]["bit_scores"], wrong)
    loo_path.write_text(json.dumps(loo), encoding="utf-8")
    bad = copy.deepcopy(report)
    bad["confirmer"]["calibrator"].update(slope=wrong.slope, intercept=wrong.intercept)
    for key, path in (("out_scores", side_path), ("out_loo_scores", loo_path)):
        bad["provenance"]["inputs"][C.CANONICAL_SOURCES[key]] = refs.content_sha256(path)
    problems = C.validate_report(bad, repo_root=root, **_CI)
    assert any("calibrator_refits_from_the_committed_sidecar" in p for p in problems), problems


@_committed
def test_a_committed_input_may_not_go_missing_even_in_ci_mode(report, tmp_path):
    root = _repo_copy(tmp_path)
    (root / C.CANONICAL_SOURCES["out_scores"]).unlink()
    assert any("is absent" in p for p in C.validate_report(report, repo_root=root, **_CI))
    bad = copy.deepcopy(report)
    bad["provenance"]["inputs"] = {"nowhere/at_all.json": "0" * 64}
    assert any("canonical sources" in p for p in C.validate_report(bad, repo_root=root, **_CI))
    bad = copy.deepcopy(report)
    bad["sources"] = {**bad["sources"], "rinalmo_report": "reports/elsewhere.json"}
    assert any("CANONICAL_SOURCES" in p for p in C.validate_report(bad, repo_root=root, **_CI))


@_committed
def test_a_report_cannot_steer_its_rederivation_to_another_file(report, tmp_path):
    """precision.items and confirmer.artifact_dir are pinned, not read off the report."""
    root = _repo_copy(tmp_path)
    forged_items = root / "reports/p3/forged_items.json"
    shutil.copy2(root / C.CANONICAL_SOURCES["out_items"], forged_items)
    kw = {"repo_root": root, "require_all_inputs": False}
    assert C.validate_report(report, **kw) == []  # positive control, precision re-derived
    bad = copy.deepcopy(report)
    bad["precision"]["items"] = "reports/p3/forged_items.json"
    assert any("precision.items" in p for p in C.validate_report(bad, **kw))
    bad = copy.deepcopy(report)
    bad["confirmer"]["artifact_dir"] = "data/processed/elsewhere"
    assert any("artifact_dir" in p for p in C.validate_report(bad, **kw))
