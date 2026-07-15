"""RiNALMo forward-throughput probe — candidates/sec/GPU on one A4000 (P1-14).

**Advisory only.** This step surfaces the dominant genome-scale latency risk
(PRD §10.2 condition (b)) *before* the P3 Stage-2 LoRA fine-tune, but the
**binding** latency decision is frozen later at the **P5 sizing gate** (which
needs the Stage-1 candidate density that only P2 produces). There is therefore
**no threshold** here (imp.md P1-14): the harness *reports* a measured
candidates/sec/GPU number and an illustrative (non-binding) extrapolation. A
"clearly-hopeless" reading is a **CLAUDE.md §7 stop-and-ask** — a human call made
from the reported number, **never** an automatic switch to RNA-FM (ADR-0002 D5).
Because that judgement is qualitative and no budget is pinned at P1, it is
deliberately **not** encoded as a numeric gate in :func:`validate_report`
(pinning an unpinned threshold would violate §10.3).

What is measured
----------------
The load-bearing genome-scale cost is the **backbone encoder forward**
(``RiNALMoModel`` → ``last_hidden_state``) — the state any Stage-2 T-box head
(a light LoRA-tuned classifier/segmenter) consumes. That is the **headline**
number. A second, heavier bracket is also reported: the encoder **plus** the
mirror's O(L²) secondary-structure contact head (via
:func:`tbox_finder.eval.rinalmo_parity.contact_logits`) — an upper bound, since
the eventual T-box head is lighter than the SS head. Both run **bf16**
(autocast) under ``torch.inference_mode`` on one A4000 (Ampere sm_86), timed with
CUDA events + ``synchronize`` after warmup (the canonical PyTorch idiom).

The probe inputs are **synthetic** length-stratified RNA windows (A/C/G/U,
deterministic seed): RiNALMo forward throughput is **length-bound, not
nucleotide-content-bound**, so a length sweep of random RNA is a faithful
*hardware latency* measurement (NOT a T-box science result — §10.3; the report
carries ``is_science_result=False``).

Env: ``tbox-ml-rna`` (transformers 5.13.0 + multimolecule 0.1.0; ADR-0002 A8;
``import multimolecule`` needs ``CUDA_HOME`` — the sbatch exports it). Every
torch / multimolecule import below is **lazy**, so importing this module in bare
CI does not pull torch (CLAUDE.md §8.4); the pure helpers + :func:`validate_report`
are stdlib-only and bare-testable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder import provenance
from tbox_finder.eval import rinalmo_parity as R

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = 1
STEP = "P1-14"
SEED = 42

#: T-box leader RNAs are ~150–350 nt; the probe sweeps this range (PRD §10.2).
LENGTH_SWEEP_NT: tuple[int, ...] = (150, 200, 250, 300, 350)
REPRESENTATIVE_LEN_NT = 300
#: Batch sizes tried at the representative length (stop at the first CUDA OOM).
BATCH_SWEEP: tuple[int, ...] = (1, 4, 8, 16, 32)
WARMUP_ITERS = 5
MEASURE_ITERS = 30
RNA_ALPHABET = "ACGU"

DEFAULT_OUT = Path("reports/p1/rinalmo_throughput.json")


# --------------------------------------------------------------------------- #
# Pure stdlib helpers (bare-CI testable — no numpy / torch)
# --------------------------------------------------------------------------- #
def synthetic_window(length: int, rng: random.Random) -> str:
    """A single random RNA window of ``length`` nt drawn from ``rng`` (A/C/G/U)."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    return "".join(rng.choice(RNA_ALPHABET) for _ in range(length))


def synthetic_batch(length: int, batch_size: int, *, seed: int) -> list[str]:
    """``batch_size`` deterministic distinct random RNA windows, all ``length`` nt.

    Seeded by ``(seed, length, batch_size)`` so a given (length, batch) is exactly
    reproducible across runs (§8.3) yet distinct across lengths/batches.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    # A *string* seed (not a tuple — unsupported) so a given (seed, length, batch) is
    # reproducible across runs (sha512-hashed, PYTHONHASHSEED-independent; §8.3).
    rng = random.Random(f"{seed}:{length}:{batch_size}")
    return [synthetic_window(length, rng) for _ in range(batch_size)]


def _percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy 'linear'/type-7). ``None`` if empty."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_batch_latencies(latencies_ms: Sequence[float], batch_size: int) -> dict[str, Any]:
    """Per-batch wall-latencies → a latency + candidates/sec/GPU summary.

    ``candidates_per_sec_per_gpu = 1000 * batch_size / mean_batch_latency_ms``
    (for ``batch_size == 1`` this is the per-candidate rate ``1000 / mean_ms``).
    """
    lat = [float(x) for x in latencies_ms]
    if not lat:
        raise ValueError("latencies_ms is empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if any((not math.isfinite(x)) or x <= 0.0 for x in lat):
        raise ValueError("every batch latency must be a finite positive number of ms")
    mean_ms = sum(lat) / len(lat)
    return {
        "batch_size": int(batch_size),
        "iters": len(lat),
        "latency_ms_mean": round(mean_ms, 4),
        "latency_ms_median": round(_percentile(lat, 50), 4),
        "latency_ms_p95": round(_percentile(lat, 95), 4),
        "latency_ms_min": round(min(lat), 4),
        "candidates_per_sec_per_gpu": round(1000.0 * batch_size / mean_ms, 3),
    }


def extrapolate_wall_hours(rate_per_sec_per_gpu: float, n_candidates: float, n_gpus: int) -> float:
    """Illustrative (non-binding) wall-hours = ``N / (rate * GPUs * 3600)``."""
    if not (_finite_number(rate_per_sec_per_gpu) and rate_per_sec_per_gpu > 0.0):
        raise ValueError("rate_per_sec_per_gpu must be a finite positive number")
    if n_gpus < 1:
        raise ValueError(f"n_gpus must be >= 1, got {n_gpus}")
    if n_candidates < 0:
        raise ValueError(f"n_candidates must be >= 0, got {n_candidates}")
    return n_candidates / (rate_per_sec_per_gpu * n_gpus * 3600.0)


def _finite_number(v: Any) -> bool:
    """True iff ``v`` is a real, finite number (bools rejected)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _finite_positive(v: Any) -> bool:
    """True iff ``v`` is a finite number strictly > 0."""
    return _finite_number(v) and float(v) > 0.0


def _bad_bool(value: Any, expected: bool) -> bool:
    """True iff ``value`` is not the exact boolean ``expected`` (int 0/1 rejected)."""
    return not (isinstance(value, bool) and value is expected)


def _sanitize(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` so the JSON is strict (no NaN/Inf)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _valid_summary(block: Any) -> bool:
    """True iff ``block`` is a latency summary with a finite positive rate."""
    return isinstance(block, dict) and _finite_positive(block.get("candidates_per_sec_per_gpu"))


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of schema/honesty problems (empty ⇒ valid). Never raises.

    Fails **closed** on the anti-fabrication invariants (§10.3): a report that is
    not marked advisory/non-binding/non-science, a precision other than bf16, a
    checkpoint that is not the pinned mirror revision, a missing GPU identity on a
    measured report, or a non-finite / non-positive candidates/sec headline. There
    is **no** throughput *threshold* check — P1-14 pins no binding budget
    (imp.md; the §7 "clearly-hopeless" call is a human judgement, not a gate).
    """
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if report.get("step") != STEP:
        errors.append(f"step must be {STEP!r}")
    if _bad_bool(report.get("advisory_only"), True):
        errors.append("advisory_only must be boolean True (PRD §10.2 — no binding gate at P1)")
    if _bad_bool(report.get("binding"), False):
        errors.append("binding must be boolean False (binding latency frozen to the P5 gate)")
    if _bad_bool(report.get("is_science_result"), False):
        errors.append("is_science_result must be boolean False (synthetic latency probe, §10.3)")
    if _bad_bool(report.get("measured"), True):
        errors.append("measured must be boolean True")
    if report.get("precision") != "bfloat16":
        errors.append("precision must be 'bfloat16' (imp.md P1-14 output spec)")
    if not (isinstance(report.get("sequence_source"), str) and report["sequence_source"].strip()):
        errors.append("sequence_source must be a non-empty string (synthetic-input disclosure)")

    ckpt = report.get("checkpoint")
    if not isinstance(ckpt, dict):
        errors.append("checkpoint block missing")
    else:
        if ckpt.get("repo_id") != R.REPO_ID:
            errors.append(f"checkpoint.repo_id must be {R.REPO_ID!r}")
        if ckpt.get("revision") != R.REVISION:
            errors.append("checkpoint.revision must be the code-pinned mirror REVISION")

    hw = report.get("hardware")
    if not isinstance(hw, dict):
        errors.append("hardware block missing")
    elif not (isinstance(hw.get("gpu_name"), str) and hw["gpu_name"].strip()):
        errors.append("hardware.gpu_name must be a non-empty string on a measured report")

    if not _finite_positive(report.get("headline_candidates_per_sec_per_gpu")):
        errors.append("headline_candidates_per_sec_per_gpu must be a finite positive number")

    enc = report.get("encoder_forward")
    if not isinstance(enc, dict):
        errors.append("encoder_forward block missing")
    else:
        if not _valid_summary(enc.get("batch1_representative")):
            errors.append("encoder_forward.batch1_representative must be a valid latency summary")
        per_len = enc.get("batch1_per_length")
        if not isinstance(per_len, dict) or not per_len:
            errors.append("encoder_forward.batch1_per_length must be a non-empty dict")
        else:
            for k, v in per_len.items():
                if not _valid_summary(v):
                    errors.append(f"encoder_forward.batch1_per_length[{k}] invalid summary")
    return errors


# --------------------------------------------------------------------------- #
# Heavy path (lazy torch / multimolecule; GPU)
# --------------------------------------------------------------------------- #
def _is_oom(exc: BaseException) -> bool:
    """True iff ``exc`` looks like a CUDA out-of-memory (vs a real code bug)."""
    return "out of memory" in str(exc).lower()


def _bench(run: Callable[[], Any], *, warmup: int, iters: int, device: str) -> list[float]:
    """Warmup then time ``iters`` calls of ``run`` — per-call ms.

    CUDA path uses ``torch.cuda.Event(enable_timing=True)`` + ``synchronize`` (the
    canonical PyTorch idiom — async kernels must be synchronized before reading the
    elapsed time); the CPU path uses ``perf_counter``. ``run`` is expected to return
    nothing reusable (throughput only); its output is discarded.
    """
    import time

    import torch  # lazy

    is_cuda = device.startswith("cuda")
    for _ in range(max(0, warmup)):
        run()
    if is_cuda:
        torch.cuda.synchronize()
    lat: list[float] = []
    for _ in range(iters):
        if is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            lat.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            run()
            t1 = time.perf_counter()
            lat.append((t1 - t0) * 1000.0)
    return lat


def measure_throughput(
    *,
    revision: str = R.REVISION,
    device: str | None = None,
    seed: int = SEED,
    warmup: int = WARMUP_ITERS,
    iters: int = MEASURE_ITERS,
    lengths: Sequence[int] = LENGTH_SWEEP_NT,
    representative_len: int = REPRESENTATIVE_LEN_NT,
    batch_sweep: Sequence[int] = BATCH_SWEEP,
    require_cuda: bool = False,
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, Any]:
    """Measure RiNALMo-giga bf16 forward throughput; return the measured sub-report.

    Loads the pinned mirror (reusing :func:`rinalmo_parity.load_rinalmo_ss`), then,
    under ``inference_mode`` + bf16 autocast, times (1) the encoder forward at
    batch 1 across the length sweep, (2) the encoder at the representative length
    over a batch sweep (until the first CUDA OOM), and (3) the encoder + SS contact
    head (batch 1) as a heavier upper bracket.
    """
    R._require_pinned_revision(revision)
    import torch  # lazy

    model, tokenizer, device = R.load_rinalmo_ss(revision=revision, device=device, seed=seed)
    if require_cuda and not device.startswith("cuda"):
        raise RuntimeError("require_cuda set but no CUDA device is available (--require-cuda)")
    is_cuda = device.startswith("cuda")
    model.eval()
    encoder = model.model  # the RiNALMoModel backbone (see rinalmo_parity._encoder_and_head)

    def _ids(seq: str) -> Any:
        return tokenizer(R.normalize_rna(seq), return_tensors="pt")["input_ids"].to(device)

    def _batch_inputs(seqs: Sequence[str]) -> tuple[Any, Any]:
        enc = tokenizer([R.normalize_rna(s) for s in seqs], return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device) if "attention_mask" in enc else None
        return ids, mask

    autocast = torch.autocast(
        device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=is_cuda
    )
    with torch.inference_mode(), autocast:
        # (1) encoder, batch 1, length sweep. The forward thunk binds ``ids`` as a default
        # (not a loop-closure) so each iteration times its own input (ruff B023).
        per_length: dict[str, Any] = {}
        for length in lengths:
            ids = _ids(synthetic_batch(length, 1, seed=seed)[0])
            lat = _bench(
                lambda ids=ids: encoder(ids).last_hidden_state,
                warmup=warmup,
                iters=iters,
                device=device,
            )
            summ = summarize_batch_latencies(lat, 1)
            per_length[str(length)] = summ
            log(f"encoder batch1 len={length}: {summ['candidates_per_sec_per_gpu']} c/s")
        rep_key = str(representative_len)
        if rep_key not in per_length:  # ensure the representative length is measured
            ids = _ids(synthetic_batch(representative_len, 1, seed=seed)[0])
            lat = _bench(
                lambda ids=ids: encoder(ids).last_hidden_state,
                warmup=warmup,
                iters=iters,
                device=device,
            )
            per_length[rep_key] = summarize_batch_latencies(lat, 1)
        batch1_representative = per_length[rep_key]

        # (2) encoder, representative length, batch sweep (stop at first CUDA OOM).
        batched: dict[str, Any] = {}
        peak_vram_gib: float | None = None
        oom_at_batch: int | None = None
        best_batched = batch1_representative
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        for bs in batch_sweep:
            try:
                ids, mask = _batch_inputs(synthetic_batch(representative_len, bs, seed=seed))
                lat = _bench(
                    lambda ids=ids, mask=mask: encoder(
                        input_ids=ids, attention_mask=mask
                    ).last_hidden_state,
                    warmup=warmup,
                    iters=iters,
                    device=device,
                )
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise  # a real code/shape bug — never masked as an OOM
                oom_at_batch = bs
                log(f"encoder batched bs={bs}: CUDA OOM — stopping the batch sweep")
                if is_cuda:
                    torch.cuda.empty_cache()
                break
            summ = summarize_batch_latencies(lat, bs)
            batched[str(bs)] = summ
            if summ["candidates_per_sec_per_gpu"] > best_batched["candidates_per_sec_per_gpu"]:
                best_batched = summ
            if is_cuda:
                peak_vram_gib = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
                torch.cuda.empty_cache()
            log(f"encoder batched bs={bs}: {summ['candidates_per_sec_per_gpu']} c/s")

        # (3) encoder + SS contact head (batch 1), an upper bracket (O(L^2) head).
        ss_per_length: dict[str, Any] = {}
        for length in lengths:
            ids = _ids(synthetic_batch(length, 1, seed=seed)[0])
            lat = _bench(
                lambda ids=ids: R.contact_logits(model, ids),
                warmup=warmup,
                iters=iters,
                device=device,
            )
            summ = summarize_batch_latencies(lat, 1)
            ss_per_length[str(length)] = summ
            log(f"ss-head batch1 len={length}: {summ['candidates_per_sec_per_gpu']} c/s")

    return {
        "device": device,
        "encoder_forward": {
            "description": (
                "RiNALMoModel encoder forward -> last_hidden_state (the backbone cost a "
                "Stage-2 T-box head consumes; the headline metric)"
            ),
            "batch1_per_length": per_length,
            "batch1_representative": batch1_representative,
            "batched_representative": batched,
            "best_batched": best_batched,
            "peak_vram_gib_best_batch": peak_vram_gib,
            "oom_at_batch": oom_at_batch,
        },
        "ss_head_forward": {
            "description": (
                "encoder + the mirror's SS contact head (O(L^2)) — an UPPER bracket; the "
                "eventual T-box Stage-2 head is lighter than the SS head"
            ),
            "batch1_per_length": ss_per_length,
            "batch1_representative": ss_per_length.get(rep_key),
        },
        "headline_candidates_per_sec_per_gpu": batch1_representative["candidates_per_sec_per_gpu"],
    }


def _pkg_version(dist: str) -> str:
    """Resolve an installed distribution version via the package metadata.

    ``importlib.metadata.version`` is authoritative even for packages that expose no
    top-level ``__version__`` (e.g. ``multimolecule`` 0.1.0), which a ``getattr`` probe
    would miss and record as ``"unknown"``.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def _software_versions() -> dict[str, str]:
    import platform

    import torch  # lazy

    versions = {"torch": torch.__version__, "python": platform.python_version()}
    versions["transformers"] = _pkg_version("transformers")
    versions["multimolecule"] = _pkg_version("multimolecule")
    return versions


def _hardware_block(device: str) -> dict[str, Any]:
    import torch  # lazy

    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu_name": props.name,
            "gpu_total_memory_gib": round(props.total_memory / (1024**3), 3),
            "cuda_capability": f"{props.major}.{props.minor}",
            "n_gpus_measured": 1,
            "device": device,
        }
    return {"gpu_name": "cpu", "gpu_total_memory_gib": None, "n_gpus_measured": 0, "device": device}


def build_report(
    measured: Mapping[str, Any],
    *,
    seed: int = SEED,
    lengths: Sequence[int] = LENGTH_SWEEP_NT,
    representative_len: int = REPRESENTATIVE_LEN_NT,
    warmup: int = WARMUP_ITERS,
    iters: int = MEASURE_ITERS,
) -> dict[str, Any]:
    """Assemble the full P1-14 report from a :func:`measure_throughput` result."""
    device = measured["device"]
    headline = measured["headline_candidates_per_sec_per_gpu"]
    best = measured["encoder_forward"]["best_batched"]["candidates_per_sec_per_gpu"]

    scenarios = []
    for n in (1_000_000, 10_000_000, 100_000_000):
        for g in (1, 8, 16):
            scenarios.append(
                {
                    "n_candidates": n,
                    "n_gpus": g,
                    "est_wall_hours_encoder_batch1": round(
                        extrapolate_wall_hours(headline, n, g), 3
                    ),
                    "est_wall_hours_encoder_best_batched": round(
                        extrapolate_wall_hours(best, n, g), 3
                    ),
                }
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "prd": "§10.2",
        "adr": "ADR-0002",
        "advisory_only": True,
        "binding": False,
        "is_science_result": False,
        "binding_gate": "P5 sizing gate (needs Stage-1 candidate density; PRD §10.2)",
        "measured": True,
        "sequence_source": (
            "synthetic length-stratified RNA (A/C/G/U), deterministic seed "
            f"{seed} — a hardware forward-latency probe; RiNALMo throughput is "
            "length-bound, not nucleotide-content-bound. NOT a T-box science result."
        ),
        "precision": "bfloat16",
        "precision_mode": "autocast_mixed",
        "seed": seed,
        "checkpoint": {
            "repo_id": R.REPO_ID,
            "revision": R.REVISION,
            "hub_url": R.HUB_URL,
        },
        "hardware": _hardware_block(device),
        "software": _software_versions(),
        "length_sweep_nt": list(lengths),
        "representative_len_nt": representative_len,
        "warmup_iters": warmup,
        "measure_iters": iters,
        "encoder_forward": measured["encoder_forward"],
        "ss_head_forward": measured["ss_head_forward"],
        "headline_candidates_per_sec_per_gpu": headline,
        "illustrative_extrapolation": {
            "note": (
                "NON-BINDING & ILLUSTRATIVE. The genome-scale candidate count N is "
                "UNKNOWN until Stage-1 (P2) sets candidate density; the binding latency "
                "decision is frozen to the P5 sizing gate (PRD §10.2). Rows below are pure "
                "arithmetic N/(rate*GPUs*3600) at the measured encoder rates, for "
                "illustrative N only."
            ),
            "rate_encoder_batch1_per_sec_per_gpu": headline,
            "rate_encoder_best_batched_per_sec_per_gpu": best,
            "scenarios": scenarios,
        },
        "git_sha": provenance.git_sha(),
    }
    return _sanitize(report)


def run_probe(
    *,
    out_path: str | Path = DEFAULT_OUT,
    seed: int = SEED,
    warmup: int = WARMUP_ITERS,
    iters: int = MEASURE_ITERS,
    device: str | None = None,
    require_cuda: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure → build → validate (fail closed) → write the P1-14 report."""
    measured = measure_throughput(
        seed=seed, warmup=warmup, iters=iters, device=device, require_cuda=require_cuda, log=log
    )
    report = build_report(measured, seed=seed, warmup=warmup, iters=iters)
    problems = validate_report(report)
    if problems:
        raise ValueError("throughput report failed validation: " + "; ".join(problems))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    report["_path"] = str(out)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RiNALMo forward-throughput probe (advisory, P1-14)."
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=MEASURE_ITERS)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail loud if no CUDA device (the deliverable is A4000-measured)",
    )
    args = parser.parse_args(argv)
    report = run_probe(
        out_path=args.out,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
        require_cuda=args.require_cuda,
    )
    print(
        f"headline encoder batch-1 @ {report['representative_len_nt']} nt = "
        f"{report['headline_candidates_per_sec_per_gpu']} candidates/sec/GPU "
        f"({report['hardware']['gpu_name']}) -> {report['_path']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
