"""P1-01 — Stage-1 CUDA-kernel import/forward smoke gate + pure-PyTorch fallback cost.

Runs on a cluster RTX A4000 (Ampere ``sm_86``) inside the pinned ``tbox-ml-dna`` env
(ADR-0002 D2/A2; PRD §10.3, §15). It:

  1. probes the runtime (torch build, CUDA device, compute capability == (8, 6));
  2. asserts the load-bearing ``import selective_scan_cuda`` succeeds — the packaged
     ``Mamba`` block hard-imports it, so a missing kernel is a *capability* loss, not a
     speed cost (ADR-0002 A2 correction C2). ``causal_conv1d`` is the only genuinely
     optional accelerator (it degrades to ``nn.Conv1d``);
  3. runs a forward on a short random-DNA-derived tensor through both the CUDA kernel
     (``selective_scan_fn`` / ``causal_conv1d_fn``) and the pure-PyTorch reference
     (``selective_scan_ref`` / ``causal_conv1d_ref``, hand-wired per A2 C2 — the
     packaged block never reaches ``selective_scan_ref``), records the max-abs output
     difference (parity), and times both paths (tokens/sec);
  4. writes ``reports/p1/kernel_smoke.json``. The ``selective_scan`` CUDA-over-ref
     throughput ratio is the *measured genome-scan cost of the pure-PyTorch fallback*
     that D3/D9 pre-registered for ADR-0002 (folded back as Amendment A5; no fabricated
     value — this smoke produces it on a real A4000).

The exit code is 0 iff the import + parity gate passes, so the sbatch wrapper can gate
its ``DONE`` marker on it (§9.3 artifact-based verification).

No numerical ``atol`` is pinned in ADR-0002 (its only ±tolerance is the D5 RiNALMo
mirror-parity margin, unrelated to kernels). The parity ``atol`` here is therefore an
implementer choice, documented and configurable (``--atol``); the raw max-abs-diff is
always recorded so the gate cannot hide a mismatch (§10.3).

All ``torch`` / ``mamba_ssm`` / ``causal_conv1d`` imports are lazy (inside functions):
the module imports cleanly with no torch present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"
DEFAULT_FIXTURE = "tests/fixtures/kernel_smoke_dna/dna_tokens.npy"

# Model shape defaults — a small Caduceus/Mamba-2-shaped block.
DEFAULT_D_MODEL = 512  # SSM channel dim D
DEFAULT_D_STATE = 16  # SSM state dim N (Mamba-2 default)
DEFAULT_D_CONV = 4  # causal-conv width (Mamba d_conv default)

# Parity atol per dtype (implementer choice — NOT ADR-pinned; see module docstring).
# fp32 selective-scan vs its reference accumulates over the sequence; a correct kernel
# agrees to well under 1e-2. A broken kernel differs by O(1), so this still bites.
DEFAULT_ATOL = {"float32": 1e-2, "bfloat16": 3e-1, "float16": 2e-1}


# --------------------------------------------------------------------------------------
# runtime / import probes
# --------------------------------------------------------------------------------------
def probe_env() -> dict[str, Any]:
    """Torch build + CUDA device facts (no kernel imports)."""
    import torch  # lazy

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "device_name": None,
        "device_capability": None,
        "is_sm86": None,
        "hostname": platform.node(),
    }
    if info["cuda_available"]:
        cap = torch.cuda.get_device_capability(0)
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = [int(cap[0]), int(cap[1])]
        info["is_sm86"] = list(info["device_capability"]) == [8, 6]
    return info


def _try_import(modname: str) -> dict[str, Any]:
    try:
        mod = __import__(modname)
        for part in modname.split(".")[1:]:
            mod = getattr(mod, part)
        ver = getattr(mod, "__version__", None)
        return {"ok": True, "version": ver, "error": None}
    except Exception as exc:  # noqa: BLE001 - we record every failure verbatim
        return {"ok": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def probe_imports() -> dict[str, Any]:
    """Record kernel-import outcomes. ``selective_scan_cuda`` is load-bearing (A2 C2)."""
    return {
        # The A2-C2 gate: the compiled selective-scan extension must import.
        "selective_scan_cuda": _try_import("selective_scan_cuda"),
        "causal_conv1d_cuda": _try_import("causal_conv1d.cpp_functions"),
        "mamba_ssm": _try_import("mamba_ssm"),
        "causal_conv1d": _try_import("causal_conv1d"),
        # Recorded for completeness; flash-attn is not gating here (SDPA fallback holds).
        "flash_attn": _try_import("flash_attn"),
    }


# --------------------------------------------------------------------------------------
# fixture → SSM inputs
# --------------------------------------------------------------------------------------
def load_dna_tokens(fixture_path: str) -> tuple[Any, dict[str, Any]]:
    """Load the committed random-DNA token fixture (``(B, L)`` int, values 0..3)."""
    import numpy as np  # lazy

    p = Path(fixture_path)
    arr = np.load(p)
    manifest_path = p.with_name("manifest.json")
    seed = 42
    if manifest_path.exists():
        seed = int(json.loads(manifest_path.read_text()).get("seed", 42))
    meta = {
        "path": str(p),
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "seed": seed,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    return arr, meta


def build_ssm_inputs(tokens, d_model, d_state, dtype, device, seed):
    """Deterministically derive selective-scan inputs from the DNA token fixture.

    ``u`` is a seeded embedding of the real DNA tokens (ties the smoke to the fixture);
    the remaining SSM tensors are seeded random with the standard Mamba conventions
    (``A`` negative real, ``delta`` softplus-positive via ``delta_softplus=True``).
    """
    import torch  # lazy

    gen = torch.Generator().manual_seed(seed)
    tok = torch.as_tensor(tokens, dtype=torch.long)  # (B, L)
    batch, seqlen = tok.shape
    emb_w = torch.randn(4, d_model, generator=gen)  # (vocab=4, D)
    u = torch.nn.functional.embedding(tok, emb_w)  # (B, L, D)
    u = u.transpose(1, 2).contiguous()  # (B, D, L)

    delta = torch.rand(batch, d_model, seqlen, generator=gen)  # positive-ish; softplus'd
    A = -torch.exp(torch.rand(d_model, d_state, generator=gen))  # (D, N) negative real
    B = torch.randn(batch, d_state, seqlen, generator=gen)  # (B, N, L), ngroups=1
    C = torch.randn(batch, d_state, seqlen, generator=gen)  # (B, N, L)
    D = torch.randn(d_model, generator=gen)  # (D,)
    delta_bias = torch.randn(d_model, generator=gen)  # (D,), fp32

    tdt = getattr(torch, dtype)
    out = {
        "u": u.to(device=device, dtype=tdt),
        "delta": delta.to(device=device, dtype=tdt),
        # A and delta_bias stay fp32 (kernel + ref both expect fp32 for these).
        "A": A.to(device=device, dtype=torch.float32),
        "B": B.to(device=device, dtype=tdt),
        "C": C.to(device=device, dtype=tdt),
        "D": D.to(device=device, dtype=torch.float32),
        "delta_bias": delta_bias.to(device=device, dtype=torch.float32),
        "batch": int(batch),
        "seqlen": int(seqlen),
    }
    return out


# --------------------------------------------------------------------------------------
# timing helper
# --------------------------------------------------------------------------------------
def _timed(fn, iters, warmup):
    """Return (elapsed_seconds, iters) for ``iters`` calls after ``warmup``, sync'd."""
    import torch  # lazy

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0, iters


# --------------------------------------------------------------------------------------
# selective-scan: CUDA kernel vs pure-PyTorch reference
# --------------------------------------------------------------------------------------
def run_selective_scan(inp, atol, iters, ref_iters, warmup) -> dict[str, Any]:
    res: dict[str, Any] = {
        "dtype": str(inp["u"].dtype).replace("torch.", ""),
        "shapes": {
            "u": list(inp["u"].shape),
            "A": list(inp["A"].shape),
            "B": list(inp["B"].shape),
        },
        "atol": atol,
        "cuda_forward_ok": False,
        "ref_forward_ok": False,
        "max_abs_diff": None,
        "parity_pass": None,
        "cuda_tokens_per_s": None,
        "ref_tokens_per_s": None,
        "throughput_ratio_cuda_over_ref": None,
        "iters": iters,
        "ref_iters": ref_iters,
        "warmup": warmup,
        "error": None,
    }
    tokens = inp["batch"] * inp["seqlen"]

    # The import is inside the try so a missing/broken kernel is RECORDED in the report
    # (res["error"]) rather than crashing the smoke with no report written.
    try:
        from mamba_ssm.ops.selective_scan_interface import (  # lazy
            selective_scan_fn,
            selective_scan_ref,
        )

        def call_cuda():
            return selective_scan_fn(
                inp["u"],
                inp["delta"],
                inp["A"],
                inp["B"],
                inp["C"],
                D=inp["D"],
                z=None,
                delta_bias=inp["delta_bias"],
                delta_softplus=True,
            )

        def call_ref():
            return selective_scan_ref(
                inp["u"],
                inp["delta"],
                inp["A"],
                inp["B"],
                inp["C"],
                D=inp["D"],
                z=None,
                delta_bias=inp["delta_bias"],
                delta_softplus=True,
            )

        out_cuda = call_cuda()
        res["cuda_forward_ok"] = True
        out_ref = call_ref()
        res["ref_forward_ok"] = True
        diff = (out_cuda.float() - out_ref.float()).abs().max().item()
        res["max_abs_diff"] = diff
        res["parity_pass"] = bool(diff <= atol)

        cuda_s, _ = _timed(call_cuda, iters, warmup)
        ref_s, _ = _timed(call_ref, ref_iters, warmup)
        res["cuda_tokens_per_s"] = tokens * iters / cuda_s
        res["ref_tokens_per_s"] = tokens * ref_iters / ref_s
        res["throughput_ratio_cuda_over_ref"] = res["cuda_tokens_per_s"] / res["ref_tokens_per_s"]
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


# --------------------------------------------------------------------------------------
# causal-conv1d: CUDA kernel vs pure-PyTorch reference
# --------------------------------------------------------------------------------------
def run_causal_conv1d(inp, d_conv, atol, iters, warmup) -> dict[str, Any]:
    x = inp["u"]  # (B, D, L)
    device, tdt = x.device, x.dtype
    d_model = x.shape[1]

    res: dict[str, Any] = {
        "dtype": str(tdt).replace("torch.", ""),
        "width": d_conv,
        "activation": "silu",
        "atol": atol,
        "cuda_forward_ok": False,
        "ref_forward_ok": False,
        "max_abs_diff": None,
        "parity_pass": None,
        "cuda_tokens_per_s": None,
        "ref_tokens_per_s": None,
        "throughput_ratio_cuda_over_ref": None,
        "iters": iters,
        "warmup": warmup,
        "error": None,
    }
    tokens = inp["batch"] * inp["seqlen"]

    # Imports inside the try so a missing/broken kernel is RECORDED, not raised
    # (causal-conv1d is the optional accelerator, ADR-0002 D3 — its absence must still
    # leave a report behind, and the gate then fails cleanly with a diagnostic).
    try:
        import torch  # lazy
        from causal_conv1d import causal_conv1d_fn  # lazy
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref  # lazy

        gen = torch.Generator().manual_seed(1234)
        weight = torch.randn(d_model, d_conv, generator=gen).to(device=device, dtype=tdt)
        bias = torch.randn(d_model, generator=gen).to(device=device, dtype=tdt)

        def call_cuda():
            return causal_conv1d_fn(x, weight, bias, activation="silu")

        def call_ref():
            return causal_conv1d_ref(x, weight, bias, activation="silu")

        out_cuda = call_cuda()
        res["cuda_forward_ok"] = True
        out_ref = call_ref()
        res["ref_forward_ok"] = True
        diff = (out_cuda.float() - out_ref.float()).abs().max().item()
        res["max_abs_diff"] = diff
        res["parity_pass"] = bool(diff <= atol)

        cuda_s, _ = _timed(call_cuda, iters, warmup)
        ref_s, _ = _timed(call_ref, iters, warmup)
        res["cuda_tokens_per_s"] = tokens * iters / cuda_s
        res["ref_tokens_per_s"] = tokens * iters / ref_s
        res["throughput_ratio_cuda_over_ref"] = res["cuda_tokens_per_s"] / res["ref_tokens_per_s"]
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------
def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def run_smoke(args) -> dict[str, Any]:
    import torch  # lazy

    # Deterministic-ish; TF32 off so fp32 is true fp32 for a clean parity read.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    env = probe_env()
    imports = probe_imports()
    device = "cuda" if env["cuda_available"] else "cpu"
    atol = args.atol if args.atol is not None else DEFAULT_ATOL.get(args.dtype, 1e-2)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": "P1-01",
        "measured": True,
        "generated_by": "src/tbox_finder/kernels/kernel_smoke.py",
        "env": env,
        "imports": imports,
        "fixture": None,
        "selective_scan": None,
        "causal_conv1d": None,
        "gate": None,
        "provenance": {
            "git_sha": _git_sha(),
            "env_lock": "envs/ml-dna.conda-lock.yml",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    # Fixture load is fault-tolerant too: a missing/corrupt fixture must still leave a
    # schema-valid report (all required fixture keys present, plus an `error`), never crash.
    try:
        tokens, fmeta = load_dna_tokens(args.fixture)
        report["fixture"] = fmeta
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        report["fixture"] = {
            "path": args.fixture,
            "shape": None,
            "dtype": None,
            "seed": None,
            "sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        tokens = None

    # The load-bearing import must succeed to run any kernel forward (A2 C2). SSM-input
    # construction is wrapped so a build error is recorded (report still written), not raised.
    ss_import_ok = imports["selective_scan_cuda"]["ok"]
    if tokens is not None and ss_import_ok and env["cuda_available"]:
        try:
            inp = build_ssm_inputs(
                tokens, args.d_model, args.d_state, args.dtype, device, args.seed
            )
            report["selective_scan"] = run_selective_scan(
                inp, atol, args.iters, args.ref_iters, args.warmup
            )
            report["causal_conv1d"] = run_causal_conv1d(
                inp, args.d_conv, atol, args.iters, args.warmup
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            report["setup_error"] = f"{type(exc).__name__}: {exc}"

    ss = report["selective_scan"] or {}
    cc = report["causal_conv1d"] or {}
    import_ok = bool(
        ss_import_ok and imports["causal_conv1d_cuda"]["ok"] and env.get("is_sm86") is True
    )
    parity_ok = bool(ss.get("parity_pass")) and bool(cc.get("parity_pass"))
    forward_ok = bool(ss.get("cuda_forward_ok")) and bool(cc.get("cuda_forward_ok"))
    report["gate"] = {
        "import_ok": import_ok,
        "forward_ok": forward_ok,
        "parity_ok": parity_ok,
        "overall_pass": bool(import_ok and forward_ok and parity_ok),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P1-01 Stage-1 CUDA-kernel smoke gate")
    p.add_argument("--fixture", default=DEFAULT_FIXTURE)
    p.add_argument("--out", default="reports/p1/kernel_smoke.json")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL, dest="d_model")
    p.add_argument("--d-state", type=int, default=DEFAULT_D_STATE, dest="d_state")
    p.add_argument("--d-conv", type=int, default=DEFAULT_D_CONV, dest="d_conv")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--ref-iters", type=int, default=10, dest="ref_iters")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = run_smoke(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    g = report["gate"]
    print(json.dumps({"gate": g, "out": str(out)}, indent=2))
    return 0 if g and g["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
