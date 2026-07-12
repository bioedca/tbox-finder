"""CPU-only schema/consistency validation for the P1-01 kernel-smoke report.

No torch / numpy / GPU: this test only parses and validates the JSON schema of
``reports/p1/kernel_smoke.json`` and the committed schema skeleton. It runs in CI.

Two layers, mirroring the P0 golden pattern:
  * structural schema (keys + types) — always enforced;
  * numeric consistency (parity vs atol, sm_86 vs capability, gate AND-of-subgates,
    throughput ratio) — enforced only when ``measured is True``.
Every consistency rule has a clean-pass case AND a mutated fail case, so the guard is
proven to bite (§8.7 / §10.3). If the real cluster report exists it is validated and the
gate is asserted to pass; otherwise that tier skips (the report is produced on an A4000).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tbox_finder.kernels import kernel_smoke as ks

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "tests/fixtures/kernel_smoke_sample/sample_report.json"
REAL = REPO / "reports/p1/kernel_smoke.json"

IMPORT_KEYS = [
    "selective_scan_cuda",
    "causal_conv1d_cuda",
    "mamba_ssm",
    "causal_conv1d",
    "flash_attn",
]
SS_KEYS = [
    "dtype",
    "shapes",
    "atol",
    "cuda_forward_ok",
    "ref_forward_ok",
    "max_abs_diff",
    "parity_pass",
    "cuda_tokens_per_s",
    "ref_tokens_per_s",
    "throughput_ratio_cuda_over_ref",
    "iters",
    "ref_iters",
    "warmup",
    "error",
]
CC_KEYS = [
    "dtype",
    "width",
    "activation",
    "atol",
    "cuda_forward_ok",
    "ref_forward_ok",
    "max_abs_diff",
    "parity_pass",
    "cuda_tokens_per_s",
    "ref_tokens_per_s",
    "throughput_ratio_cuda_over_ref",
    "iters",
    "warmup",
    "error",
]
GATE_KEYS = ["import_ok", "forward_ok", "parity_ok", "overall_pass"]
NUM_FIELDS = [
    "atol",
    "max_abs_diff",
    "cuda_tokens_per_s",
    "ref_tokens_per_s",
    "throughput_ratio_cuda_over_ref",
]
BOOL_FIELDS = ["cuda_forward_ok", "ref_forward_ok", "parity_pass"]


def _num_or_none(v):
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


def _bool_or_none(v):
    return v is None or isinstance(v, bool)


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_block(block, keys, name, req):
    if block is None:  # kernel_smoke sets the block to None when the kernel didn't run
        return
    req(isinstance(block, dict), f"{name} not dict/null")
    if not isinstance(block, dict):
        return
    for k in keys:
        req(k in block, f"{name} missing {k}")
    for k in NUM_FIELDS:
        if k in block:
            req(_num_or_none(block[k]), f"{name}.{k} not number/null")
    for k in BOOL_FIELDS:
        if k in block:
            req(_bool_or_none(block[k]), f"{name}.{k} not bool/null")


def _consistency(d, req):
    env = d["env"]
    if env.get("is_sm86") is True:
        req(env.get("device_capability") == [8, 6], "is_sm86 True but capability != [8,6]")
    for name in ("selective_scan", "causal_conv1d"):
        b = d[name]
        if not isinstance(b, dict):
            continue
        diff, atol, pp = b.get("max_abs_diff"), b.get("atol"), b.get("parity_pass")
        if pp is True:
            req(
                _isnum(diff) and _isnum(atol) and diff <= atol,
                f"{name}.parity_pass True but max_abs_diff > atol",
            )
        ctps, rtps = b.get("cuda_tokens_per_s"), b.get("ref_tokens_per_s")
        ratio = b.get("throughput_ratio_cuda_over_ref")
        if ratio is not None:
            req(_isnum(ratio) and ratio > 0, f"{name}.throughput_ratio not > 0")
            if _isnum(ratio) and _isnum(ctps) and _isnum(rtps) and rtps != 0:
                req(
                    abs(ratio - ctps / rtps) <= 1e-6 * max(1.0, abs(ratio)),
                    f"{name}.ratio != cuda/ref",
                )
    g = d["gate"]
    subs = [g.get("import_ok"), g.get("forward_ok"), g.get("parity_ok")]
    if all(isinstance(x, bool) for x in subs) and isinstance(g.get("overall_pass"), bool):
        req(g["overall_pass"] == all(subs), "overall_pass != AND(import,forward,parity)")
    ok = d["imports"]["selective_scan_cuda"].get("ok")
    req(isinstance(ok, bool), "measured report: selective_scan_cuda.ok must be bool (A2 C2)")


def validate_report(d):
    """Return a list of schema/consistency error strings (empty == valid)."""
    errs: list[str] = []

    def req(cond, msg):
        if not cond:
            errs.append(msg)

    top = [
        "schema_version",
        "step",
        "measured",
        "generated_by",
        "env",
        "imports",
        "fixture",
        "selective_scan",
        "causal_conv1d",
        "gate",
        "provenance",
    ]
    for k in top:
        req(k in d, f"missing top key {k}")
    if errs:
        return errs
    req(d["schema_version"] == "1", "schema_version != '1'")
    req(d["step"] == "P1-01", "step != P1-01")
    req(isinstance(d["measured"], bool), "measured not bool")

    env = d["env"]
    for k in [
        "python",
        "torch",
        "cuda_available",
        "torch_cuda_version",
        "device_name",
        "device_capability",
        "is_sm86",
        "hostname",
    ]:
        req(k in env, f"env missing {k}")
    req(_bool_or_none(env.get("cuda_available")), "env.cuda_available not bool/null")
    req(_bool_or_none(env.get("is_sm86")), "env.is_sm86 not bool/null")
    cap = env.get("device_capability")
    req(
        cap is None
        or (
            isinstance(cap, list)
            and len(cap) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in cap)
        ),
        "device_capability not [int,int]/null",
    )

    imp = d["imports"]
    for k in IMPORT_KEYS:
        req(k in imp, f"imports missing {k}")
        sub = imp.get(k)
        req(
            isinstance(sub, dict) and {"ok", "version", "error"} <= set(sub),
            f"imports.{k} malformed",
        )
        if isinstance(sub, dict):
            req(_bool_or_none(sub.get("ok")), f"imports.{k}.ok not bool/null")

    for k in ["path", "shape", "dtype", "seed", "sha256"]:
        req(k in d["fixture"], f"fixture missing {k}")

    _check_block(d["selective_scan"], SS_KEYS, "selective_scan", req)
    _check_block(d["causal_conv1d"], CC_KEYS, "causal_conv1d", req)

    for k in GATE_KEYS:
        req(k in d["gate"], f"gate missing {k}")
        req(_bool_or_none(d["gate"].get(k)), f"gate.{k} not bool/null")

    for k in ["git_sha", "env_lock", "slurm_job_id", "timestamp_utc"]:
        req(k in d["provenance"], f"provenance missing {k}")

    if d.get("measured") is True:
        _consistency(d, req)
    return errs


def good_measured_report():
    """A fully-populated, schema-valid *measured* report (in-test scaffolding only)."""
    return {
        "schema_version": "1",
        "step": "P1-01",
        "measured": True,
        "generated_by": "src/tbox_finder/kernels/kernel_smoke.py",
        "env": {
            "python": "3.12.11",
            "torch": "2.7.1+cu128",
            "cuda_available": True,
            "torch_cuda_version": "12.8",
            "device_name": "NVIDIA RTX A4000",
            "device_capability": [8, 6],
            "is_sm86": True,
            "hostname": "two",
        },
        "imports": {k: {"ok": True, "version": "x", "error": None} for k in IMPORT_KEYS},
        "fixture": {
            "path": "f",
            "shape": [2, 512],
            "dtype": "int8",
            "seed": 42,
            "sha256": "a" * 64,
        },
        "selective_scan": {
            "dtype": "float32",
            "shapes": {"u": [2, 512, 512]},
            "atol": 1e-2,
            "cuda_forward_ok": True,
            "ref_forward_ok": True,
            "max_abs_diff": 1e-4,
            "parity_pass": True,
            "cuda_tokens_per_s": 2.0e7,
            "ref_tokens_per_s": 5.0e5,
            "throughput_ratio_cuda_over_ref": 40.0,
            "iters": 50,
            "ref_iters": 10,
            "warmup": 10,
            "error": None,
        },
        "causal_conv1d": {
            "dtype": "float32",
            "width": 4,
            "activation": "silu",
            "atol": 1e-2,
            "cuda_forward_ok": True,
            "ref_forward_ok": True,
            "max_abs_diff": 1e-5,
            "parity_pass": True,
            "cuda_tokens_per_s": 3.0e7,
            "ref_tokens_per_s": 6.0e6,
            "throughput_ratio_cuda_over_ref": 5.0,
            "iters": 50,
            "warmup": 10,
            "error": None,
        },
        "gate": {"import_ok": True, "forward_ok": True, "parity_ok": True, "overall_pass": True},
        "provenance": {
            "git_sha": "a" * 40,
            "env_lock": "envs/ml-dna.conda-lock.yml",
            "seed": 42,
            "slurm_job_id": "123",
            "timestamp_utc": "2026-07-12T00:00:00Z",
        },
    }


# ---- module ↔ schema link ------------------------------------------------------------
def test_module_schema_constants():
    assert ks.SCHEMA_VERSION == "1"
    assert ks.DEFAULT_FIXTURE == "tests/fixtures/kernel_smoke_dna/dna_tokens.npy"
    assert (REPO / ks.DEFAULT_FIXTURE).is_file()


# ---- clean-pass cases ----------------------------------------------------------------
def test_sample_is_schema_valid():
    d = json.loads(SAMPLE.read_text())
    assert validate_report(d) == []


def test_sample_has_no_fabricated_metrics():
    """The committed skeleton must carry NO metric numbers (§10.3)."""
    d = json.loads(SAMPLE.read_text())
    assert d["measured"] is False
    for name in ("selective_scan", "causal_conv1d"):
        for k in NUM_FIELDS:
            assert d[name][k] is None, f"{name}.{k} must be null in the skeleton"


def test_good_measured_report_valid():
    assert validate_report(good_measured_report()) == []


def test_fixture_error_report_is_schema_valid():
    """run_smoke's fault-tolerant path (fixture load failed) must stay schema-valid:
    the fixture dict keeps its required keys (values null) plus an `error`, the kernel
    blocks are None, and the gate fails cleanly."""
    d = good_measured_report()
    d["fixture"] = {
        "path": "missing.npy",
        "shape": None,
        "dtype": None,
        "seed": None,
        "sha256": None,
        "error": "FileNotFoundError: missing.npy",
    }
    d["selective_scan"] = None
    d["causal_conv1d"] = None
    d["gate"] = {
        "import_ok": True,
        "forward_ok": False,
        "parity_ok": False,
        "overall_pass": False,
    }
    d["setup_error"] = "RuntimeError: build failed"
    assert validate_report(d) == []


# ---- guard bites: each mutation must produce >=1 error --------------------------------
def _drop_top(d):
    del d["gate"]
    return d


def _bad_schema_version(d):
    d["schema_version"] = "2"
    return d


def _sm86_capability_mismatch(d):
    d["env"]["device_capability"] = [7, 5]  # is_sm86 stays True
    return d


def _parity_pass_but_diff_gt_atol(d):
    d["selective_scan"]["parity_pass"] = True
    d["selective_scan"]["max_abs_diff"] = 5.0
    d["selective_scan"]["atol"] = 1e-2
    return d


def _overall_pass_but_subgate_false(d):
    d["gate"]["parity_ok"] = False
    d["gate"]["overall_pass"] = True
    return d


def _negative_ratio(d):
    d["selective_scan"]["throughput_ratio_cuda_over_ref"] = -3.0
    return d


def _ratio_not_cuda_over_ref(d):
    d["selective_scan"]["throughput_ratio_cuda_over_ref"] = 999.0  # != 2e7/5e5
    return d


def _import_ok_not_bool(d):
    d["imports"]["selective_scan_cuda"]["ok"] = None  # measured report needs a real bool
    return d


def _num_field_is_string(d):
    d["selective_scan"]["max_abs_diff"] = "1e-4"
    return d


@pytest.mark.parametrize(
    "mutate",
    [
        _drop_top,
        _bad_schema_version,
        _sm86_capability_mismatch,
        _parity_pass_but_diff_gt_atol,
        _overall_pass_but_subgate_false,
        _negative_ratio,
        _ratio_not_cuda_over_ref,
        _import_ok_not_bool,
        _num_field_is_string,
    ],
)
def test_guard_bites(mutate):
    bad = mutate(copy.deepcopy(good_measured_report()))
    assert len(validate_report(bad)) >= 1


# ---- real cluster report (present only after the A4000 run) ---------------------------
@pytest.mark.skipif(not REAL.exists(), reason="cluster kernel_smoke.json not present")
def test_real_report_valid_and_gate_passes():
    d = json.loads(REAL.read_text())
    assert validate_report(d) == []
    assert d["measured"] is True
    assert d["gate"]["overall_pass"] is True, "P1-01 kernel smoke gate must pass"
    assert d["env"]["is_sm86"] is True, "must run on an Ampere sm_86 A4000"
    assert d["imports"]["selective_scan_cuda"]["ok"] is True, "A2 C2 load-bearing import"
