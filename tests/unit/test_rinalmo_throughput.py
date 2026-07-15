"""Unit tests for the P1-14 RiNALMo forward-throughput probe (advisory).

Two tiers:

- **stdlib** (always runs, incl. bare CI): the pinned constants; the synthetic
  RNA-window generators (deterministic, A/C/G/U, correct lengths); the latency /
  percentile / candidates-per-sec math; the illustrative-extrapolation arithmetic
  + its guards; ``_sanitize``; and the report validator proven to **bite** on every
  schema/honesty invariant (§10.3) — advisory-only, non-binding, non-science, bf16,
  the pinned mirror checkpoint, GPU identity, and a finite positive headline rate.
- **torch** (``importorskip torch``; CPU-only — no model download / GPU): ``build_report``
  on a stub measured dict produces a **validator-clean** report with the 9-row
  illustrative extrapolation. The GPU forward itself is measured on the cluster A4000
  (artifact-based, §9.3) — not in CI (the P1-13 torch-tier precedent).

The measured ``reports/p1/rinalmo_throughput.json`` is produced by the SLURM run;
this authoring commit ships the harness + its logic tests.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from tbox_finder.eval import rinalmo_parity as R
from tbox_finder.eval import rinalmo_throughput as T


# --------------------------------------------------------------------------- #
# A hand-built schema-valid report (torch-free — mirrors build_report's shape).
# --------------------------------------------------------------------------- #
def _summary(cps: float, bs: int = 1) -> dict:
    return {
        "batch_size": bs,
        "iters": 30,
        "latency_ms_mean": round(1000.0 * bs / cps, 4),
        "latency_ms_median": round(1000.0 * bs / cps, 4),
        "latency_ms_p95": round(1000.0 * bs / cps, 4),
        "latency_ms_min": round(1000.0 * bs / cps, 4),
        "candidates_per_sec_per_gpu": cps,
    }


def _valid_report() -> dict:
    return {
        "schema_version": T.SCHEMA_VERSION,
        "step": T.STEP,
        "advisory_only": True,
        "binding": False,
        "is_science_result": False,
        "measured": True,
        "precision": "bfloat16",
        "sequence_source": "synthetic length-stratified RNA (A/C/G/U), seed 42 — latency probe.",
        "checkpoint": {"repo_id": R.REPO_ID, "revision": R.REVISION, "hub_url": R.HUB_URL},
        "hardware": {"gpu_name": "NVIDIA RTX A4000", "n_gpus_measured": 1, "device": "cuda"},
        "headline_candidates_per_sec_per_gpu": 42.0,
        "encoder_forward": {
            "batch1_per_length": {"150": _summary(60.0), "300": _summary(42.0)},
            "batch1_representative": _summary(42.0),
        },
    }


# --------------------------------------------------------------------------- #
# stdlib tier — constants
# --------------------------------------------------------------------------- #
def test_pinned_constants():
    assert T.SCHEMA_VERSION == 1
    assert T.STEP == "P1-14"
    assert T.SEED == 42
    assert T.RNA_ALPHABET == "ACGU"
    assert T.REPRESENTATIVE_LEN_NT in T.LENGTH_SWEEP_NT
    assert all(150 <= n <= 350 for n in T.LENGTH_SWEEP_NT)  # T-box leader length band
    assert T.BATCH_SWEEP[0] == 1  # batch-1 (per-candidate) is always measured


# --------------------------------------------------------------------------- #
# stdlib tier — synthetic windows
# --------------------------------------------------------------------------- #
def test_synthetic_window_length_alphabet_and_determinism():
    w1 = T.synthetic_window(200, random.Random(7))
    w2 = T.synthetic_window(200, random.Random(7))
    assert w1 == w2  # same rng seed -> identical
    assert len(w1) == 200
    assert set(w1) <= set(T.RNA_ALPHABET)
    assert T.synthetic_window(200, random.Random(8)) != w1  # different seed -> different


def test_synthetic_window_rejects_nonpositive_length():
    with pytest.raises(ValueError):
        T.synthetic_window(0, random.Random(1))


def test_synthetic_batch_shape_determinism_and_distinctness():
    b = T.synthetic_batch(300, 8, seed=42)
    assert len(b) == 8
    assert all(len(s) == 300 and set(s) <= set(T.RNA_ALPHABET) for s in b)
    assert b == T.synthetic_batch(300, 8, seed=42)  # reproducible
    assert b != T.synthetic_batch(300, 8, seed=43)  # seed-sensitive
    assert T.synthetic_batch(150, 4, seed=42) != T.synthetic_batch(300, 4, seed=42)[:4]


def test_synthetic_batch_rejects_bad_batch_size():
    with pytest.raises(ValueError):
        T.synthetic_batch(300, 0, seed=42)


# --------------------------------------------------------------------------- #
# stdlib tier — latency / percentile / candidates-per-sec math
# --------------------------------------------------------------------------- #
def test_percentile_linear_interpolation():
    xs = [10.0, 20.0, 30.0, 40.0]
    assert T._percentile(xs, 0) == 10.0
    assert T._percentile(xs, 100) == 40.0
    assert T._percentile(xs, 50) == pytest.approx(25.0)
    assert T._percentile([], 50) is None
    assert T._percentile([5.0], 95) == 5.0


def test_summarize_batch_latencies_math():
    # 10 ms mean at batch 1 -> 100 candidates/sec.
    s = T.summarize_batch_latencies([10.0, 10.0, 10.0], batch_size=1)
    assert s["candidates_per_sec_per_gpu"] == pytest.approx(100.0)
    assert s["latency_ms_mean"] == pytest.approx(10.0)
    assert s["batch_size"] == 1 and s["iters"] == 3
    # batch 8 at 20 ms mean -> 400 candidates/sec.
    s8 = T.summarize_batch_latencies([20.0, 20.0], batch_size=8)
    assert s8["candidates_per_sec_per_gpu"] == pytest.approx(400.0)


def test_summarize_batch_latencies_guards():
    with pytest.raises(ValueError):
        T.summarize_batch_latencies([], 1)
    with pytest.raises(ValueError):
        T.summarize_batch_latencies([10.0], 0)
    with pytest.raises(ValueError):
        T.summarize_batch_latencies([0.0], 1)  # non-positive latency
    with pytest.raises(ValueError):
        T.summarize_batch_latencies([float("inf")], 1)  # non-finite


# --------------------------------------------------------------------------- #
# stdlib tier — illustrative extrapolation
# --------------------------------------------------------------------------- #
def test_extrapolate_wall_hours_arithmetic():
    # 100 cand/s/GPU, 3.6e6 candidates, 1 GPU -> 36000 s = 10 h.
    assert T.extrapolate_wall_hours(100.0, 3_600_000, 1) == pytest.approx(10.0)
    # 8 GPUs -> 1/8 the wall.
    assert T.extrapolate_wall_hours(100.0, 3_600_000, 8) == pytest.approx(1.25)
    assert T.extrapolate_wall_hours(100.0, 0, 1) == 0.0


def test_extrapolate_wall_hours_guards():
    with pytest.raises(ValueError):
        T.extrapolate_wall_hours(0.0, 1000, 1)  # non-positive rate
    with pytest.raises(ValueError):
        T.extrapolate_wall_hours(100.0, 1000, 0)  # < 1 GPU
    with pytest.raises(ValueError):
        T.extrapolate_wall_hours(100.0, -1, 1)  # negative N


# --------------------------------------------------------------------------- #
# stdlib tier — _sanitize + JSON strictness
# --------------------------------------------------------------------------- #
def test_sanitize_maps_nonfinite_to_none():
    out = T._sanitize({"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": -math.inf}})
    assert out == {"a": None, "b": [None, 1.0], "c": {"d": None}}


def test_valid_report_json_round_trips():
    rep = _valid_report()
    assert json.loads(json.dumps(rep)) == rep


# --------------------------------------------------------------------------- #
# stdlib tier — the validator BITES (fail-closed on each invariant)
# --------------------------------------------------------------------------- #
def test_validate_report_accepts_valid():
    assert T.validate_report(_valid_report()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema_version=2),
        lambda r: r.update(step="P1-13"),
        lambda r: r.update(advisory_only=False),
        lambda r: r.update(binding=True),
        lambda r: r.update(is_science_result=True),
        lambda r: r.update(measured=False),
        lambda r: r.update(measured=1),  # int 1 is not bool True
        lambda r: r.update(precision="float16"),
        lambda r: r.update(sequence_source=""),
        lambda r: r.__setitem__("checkpoint", {"repo_id": "other", "revision": R.REVISION}),
        lambda r: r["checkpoint"].__setitem__("revision", "deadbeef"),
        lambda r: r.pop("checkpoint"),
        lambda r: r.pop("hardware"),
        lambda r: r["hardware"].__setitem__("gpu_name", ""),
        lambda r: r.update(headline_candidates_per_sec_per_gpu=0.0),
        lambda r: r.update(headline_candidates_per_sec_per_gpu=-1.0),
        lambda r: r.update(headline_candidates_per_sec_per_gpu=float("nan")),
        lambda r: r.pop("encoder_forward"),
        lambda r: r["encoder_forward"].__setitem__("batch1_representative", {"x": 1}),
        lambda r: r["encoder_forward"].__setitem__("batch1_per_length", {}),
        lambda r: r["encoder_forward"]["batch1_per_length"].__setitem__("150", {"bad": 1}),
    ],
)
def test_validate_report_bites(mutate):
    rep = _valid_report()
    mutate(rep)
    assert T.validate_report(rep), "validator must reject the mutated report"


def test_validate_report_headline_nan_sanitized_still_bites():
    # After _sanitize a NaN headline becomes None -> still not a finite positive number.
    bad = {**_valid_report(), "headline_candidates_per_sec_per_gpu": float("nan")}
    rep = T._sanitize(bad)
    assert rep["headline_candidates_per_sec_per_gpu"] is None
    assert T.validate_report(rep)


# --------------------------------------------------------------------------- #
# torch tier (CPU-only; no model download) — build_report end-to-end shape
# --------------------------------------------------------------------------- #
def test_build_report_is_validator_clean_and_extrapolates():
    pytest.importorskip("torch")
    measured = {
        "device": "cpu",
        "encoder_forward": {
            "description": "enc",
            "batch1_per_length": {"150": _summary(60.0), "300": _summary(42.0)},
            "batch1_representative": _summary(42.0),
            "batched_representative": {"1": _summary(42.0), "8": _summary(120.0, 8)},
            "best_batched": _summary(120.0, 8),
            "peak_vram_gib_best_batch": 6.5,
            "oom_at_batch": 16,
        },
        "ss_head_forward": {
            "description": "ss",
            "batch1_per_length": {"300": _summary(9.0)},
            "batch1_representative": _summary(9.0),
        },
        "headline_candidates_per_sec_per_gpu": 42.0,
    }
    report = T.build_report(measured)
    assert T.validate_report(report) == []
    ex = report["illustrative_extrapolation"]
    assert ex["rate_encoder_batch1_per_sec_per_gpu"] == 42.0
    assert len(ex["scenarios"]) == 9  # 3 candidate counts x 3 GPU counts
    # best-batched wall < batch-1 wall for the same scenario (batching is faster).
    s = ex["scenarios"][0]
    assert s["est_wall_hours_encoder_best_batched"] < s["est_wall_hours_encoder_batch1"]
    # strict JSON (no NaN/Inf survived _sanitize).
    json.dumps(report)
