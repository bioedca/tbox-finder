"""P2-12 — forward-only scan-throughput probe across the three pinned Caduceus-PS checkpoints.

**Why this exists.** PRD §10.1 pre-registers the final Stage-1 checkpoint size as a Phase-2
ablation and words it as a **throughput-driven downward search**: 7.73M is the Caduceus-PS
family maximum, long context is unnecessary for ~150–350 nt T-box loci, and a bigger backbone
costs scan throughput genome-wide (PRD §6). A validation-F1 column alone therefore does **not**
deliver that axis — it answers "does the small model segment as well?", never "what does the
big one cost to run?". This probe supplies the second half: windows/second/GPU at the frozen
ADR-0005 D3+A3 1024/512 scan geometry, measured on each allow-listed checkpoint under the real
Stage-1 stack (backbone → RC-combine → seg head), forward-only.

**Scope, stated rather than implied (§10.3).**

* It is a **shape** measurement, not a corpus scan. Inputs are seeded random ACGT windows put
  through the checkpoint's own tokenizer. Caduceus is a Mamba stack of fixed-shape dense ops
  with no attention sparsity, no KV cache and no early exit, so the cost of a ``(B, L)``
  forward does not depend on which nucleotides are in it — but that is an argument, not a
  measurement, so it is recorded in the report's ``disclosures`` block and the report never
  claims to be a genome-scan rate. The end-to-end mining rate (which includes I/O, tiling,
  reconciliation and candidate calling) is what ``infer/rho_pilot.py`` and
  ``mining/mine_round.py`` measure; the numbers here are strictly the backbone+head forward.
* It is **single-GPU and forward-only** — no DDP, no backward, no optimiser. That is the right
  unit for a scan (inference is what runs genome-wide) and it is why the probe is a separate
  1-GPU job rather than a fifth task of the DDP×8 training array.
* ``use_crf`` is irrelevant here and is left at the linear default: the CRF is a decode-time
  cost under the frozen D3 reconcile→argmax operator, not a backbone cost.

**Budget.** ADR-0003 A3 charges this probe ``<= 1.0 GPU-h`` across all three checkpoints, inside
the ``<= 18 GPU-h`` P2-12 aggregate bound.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder.models.backbone_registry import (
    CHECKPOINT_KEYS,
    PRODUCTION_CHECKPOINT,
    checkpoint_summary,
    resolve_checkpoint,
)

SCHEMA_VERSION = "1"
STEP = "P2-12"
GENERATED_BY = "src/tbox_finder/eval/backbone_throughput.py"
ADR = "ADR-0002 A14; ADR-0003 A3; ADR-0005 D3+A3"
ENV_LOCK = "envs/ml-dna.conda-lock.yml"

#: The frozen scan geometry (ADR-0005 D3 + A3). Sweeping it would unpin the reconciliation
#: geometry GATE-4's boundary-IoU defence rests on, so the probe holds it exactly.
WINDOW_NT = 1024
STRIDE_NT = 512

DEFAULT_BATCH_SIZE = 8
#: Forwards discarded before timing starts. The first forwards pay CUDA context creation,
#: cuBLAS/selective-scan autotuning and lazy allocator growth; timing them would report the
#: warm-up as throughput and would do so *differently* per checkpoint (the 16-layer model
#: amortises a fixed cost over more work), i.e. it would bias the very comparison this probe
#: exists to make.
DEFAULT_WARMUP_BATCHES = 5
DEFAULT_TIMED_BATCHES = 30
DEFAULT_SEED = 42
DEFAULT_OUT = "reports/p2/backbone_throughput.json"

_ACGT = "ACGT"


def _sanitize(obj: Any) -> Any:
    """JSON-safety pass (the train_stage1/caduceus_backbone idiom)."""
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    return str(obj)


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def random_windows(n: int, *, window_nt: int, seed: int) -> list[str]:
    """``n`` seeded random ACGT strings of length ``window_nt``.

    Seeded so the probe is reproducible (§8.3) and so **every checkpoint sees the identical
    input**, which is what makes the per-checkpoint rates comparable rather than three
    measurements of three different workloads.
    """
    import random

    rng = random.Random(seed)
    return ["".join(rng.choice(_ACGT) for _ in range(window_nt)) for _ in range(n)]


def measure_throughput(
    checkpoint: str,
    *,
    window_nt: int = WINDOW_NT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    timed_batches: int = DEFAULT_TIMED_BATCHES,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> dict[str, Any]:
    """Measure forward-only windows/second for one allow-listed checkpoint.

    Timing uses CUDA events around each timed batch with a ``torch.cuda.synchronize()`` before
    reading them: a host-side wall clock around an async launch measures the *launch*, not the
    kernel. Per-batch times are all recorded, so the report carries the spread rather than a
    single mean a reader has to take on faith.
    """
    # Argument validation FIRST, before `import torch` — so the refusals below are testable in
    # the bare CI tier (no torch) and so a bad call fails in milliseconds on a login node
    # rather than after a CUDA context and a 2.5 GB checkpoint download.
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if timed_batches < 1:
        raise ValueError(f"timed_batches must be >= 1; got {timed_batches}")
    if warmup_batches < 0:
        raise ValueError(f"warmup_batches must be >= 0; got {warmup_batches}")
    spec = resolve_checkpoint(checkpoint)
    if device is not None and not str(device).startswith("cuda"):
        # The packaged Mamba hard-requires `selective_scan_cuda` (ADR-0002 A2 C2), so a CPU
        # "throughput" number would either fail or measure a ~871x slower fallback and be
        # reported as if it were the scan rate. Refuse rather than emit it (§10.3).
        raise RuntimeError(
            f"backbone throughput must be measured on CUDA, got device={device!r}: the "
            "packaged Mamba selective_scan is a CUDA kernel (ADR-0002 A2 C2), and a CPU "
            "number is not the quantity PRD §10.1's throughput-driven search asks for."
        )

    import torch

    from tbox_finder.models.caduceus_backbone import (
        count_parameters,
        load_caduceus_ps,
        load_tokenizer,
    )
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "backbone throughput must be measured on CUDA and no CUDA device is visible "
                "(ADR-0002 A2 C2); refusing to emit a CPU number as a scan rate (§10.3)."
            )
        device = "cuda"

    tokenizer = load_tokenizer(checkpoint=spec.key)
    backbone = load_caduceus_ps(checkpoint=spec.key, device=device)
    measured_d_model = int(backbone.config.d_model)
    segmenter = Stage1Segmenter(backbone=backbone, d_model=measured_d_model).to(device).eval()

    windows = random_windows(batch_size, window_nt=window_nt, seed=seed)
    encoded = tokenizer(windows, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    if int(input_ids.shape[-1]) != window_nt:
        raise ValueError(
            f"tokenizer produced {int(input_ids.shape[-1])} ids for a {window_nt}-nt window; "
            "the probe measures the frozen ADR-0005 D3 scan geometry and cannot silently "
            "measure a different one."
        )

    torch.cuda.reset_peak_memory_stats(device)
    batch_seconds: list[float] = []
    with torch.inference_mode():  # NOTE: does not imply eval() — .eval() is set above
        for _ in range(warmup_batches):
            segmenter(input_ids=input_ids)
        torch.cuda.synchronize(device)
        for _ in range(timed_batches):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            segmenter(input_ids=input_ids)
            end.record()
            torch.cuda.synchronize(device)
            batch_seconds.append(start.elapsed_time(end) / 1000.0)

    peak_vram_gib = float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
    total_seconds = float(sum(batch_seconds))
    n_windows = int(batch_size * timed_batches)
    return {
        **checkpoint_summary(spec),
        "measured_param_count": count_parameters(backbone),
        "measured_d_model": measured_d_model,
        "window_nt": int(window_nt),
        "stride_nt": int(STRIDE_NT),
        "batch_size": int(batch_size),
        "warmup_batches": int(warmup_batches),
        "timed_batches": int(timed_batches),
        "n_windows": n_windows,
        "seed": int(seed),
        "batch_seconds": [float(x) for x in batch_seconds],
        "total_seconds": total_seconds,
        "windows_per_s_per_gpu": (n_windows / total_seconds) if total_seconds > 0 else None,
        "seconds_per_window": (total_seconds / n_windows) if n_windows else None,
        "peak_vram_gib": peak_vram_gib,
        "device_name": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
    }


def build_report(
    *,
    checkpoints: Sequence[str] = CHECKPOINT_KEYS,
    window_nt: int = WINDOW_NT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    timed_batches: int = DEFAULT_TIMED_BATCHES,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> dict[str, Any]:
    """Probe every requested checkpoint and assemble the P2-12 throughput report."""
    measurements = [
        measure_throughput(
            key,
            window_nt=window_nt,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            timed_batches=timed_batches,
            seed=seed,
            device=device,
        )
        for key in checkpoints
    ]
    by_key = {m["key"]: m for m in measurements}
    baseline = by_key.get(PRODUCTION_CHECKPOINT)
    baseline_rate = baseline.get("windows_per_s_per_gpu") if baseline else None
    for m in measurements:
        rate = m.get("windows_per_s_per_gpu")
        m["speedup_vs_production"] = (
            (rate / baseline_rate)
            if _finite_positive(rate) and _finite_positive(baseline_rate)
            else None
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "env_lock": ENV_LOCK,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "is_science": False,
        "measurements": _sanitize(measurements),
        "disclosures": [
            "Forward-only, single-GPU, no DDP and no backward: this is the inference cost "
            "that runs genome-wide, not a training rate.",
            "Inputs are seeded random ACGT windows through each checkpoint's own tokenizer, "
            "identical across checkpoints. Caduceus is a Mamba stack of fixed-shape dense "
            "ops (no attention sparsity, no KV cache, no early exit), so a (B, L) forward's "
            "cost is content-independent — an argument, not a measurement, so this is NOT a "
            "genome-scan rate. End-to-end scan rates are measured by infer/rho_pilot.py and "
            "mining/mine_round.py, which include I/O, tiling and candidate calling.",
            "Window/stride are held at the ADR-0005 D3+A3 1024/512 geometry for every "
            "checkpoint, so the comparison is size/pretraining-context only.",
            "Both non-production checkpoints are seqlen-1k pretrained, so any rate delta is "
            "attributable to a joint size x pretraining-context change (ADR-0002 A14).",
        ],
    }


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of problems; empty ⇒ valid. Total — never raises, reports."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return [f"report must be a mapping, got {type(report).__name__}"]
    for key in ("schema_version", "step", "generated_by", "adr", "env_lock"):
        if not isinstance(report.get(key), str) or not report.get(key):
            problems.append(f"{key}: missing or not a non-empty string")
    if report.get("is_science") is not False:
        problems.append("is_science: a throughput probe is not a scientific claim (§10.3)")
    disclosures = report.get("disclosures")
    if not isinstance(disclosures, list) or not disclosures:
        problems.append("disclosures: missing — the synthetic-input scope must travel with it")
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        return [*problems, "measurements: missing or empty"]
    for i, m in enumerate(measurements):
        if not isinstance(m, Mapping):
            problems.append(f"measurements[{i}]: not a mapping")
            continue
        key = m.get("key")
        try:
            spec = resolve_checkpoint(key)
        except ValueError as exc:
            problems.append(f"measurements[{i}].key: {exc}")
            continue
        if m.get("measured_param_count") != spec.expected_param_count:
            problems.append(
                f"measurements[{i}]: measured_param_count "
                f"{m.get('measured_param_count')!r} != {spec.expected_param_count} for {key!r}"
            )
        if m.get("measured_d_model") != spec.d_model:
            problems.append(
                f"measurements[{i}]: measured_d_model {m.get('measured_d_model')!r} "
                f"!= {spec.d_model} for {key!r}"
            )
        if m.get("window_nt") != WINDOW_NT:
            problems.append(
                f"measurements[{i}]: window_nt {m.get('window_nt')!r} != the frozen "
                f"ADR-0005 D3 geometry {WINDOW_NT}"
            )
        if not _finite_positive(m.get("windows_per_s_per_gpu")):
            problems.append(f"measurements[{i}]: windows_per_s_per_gpu must be finite and > 0")
        # The rate must FOLLOW from the recorded evidence rather than sit beside it: a
        # throughput echoed independently of its own timings is unfalsifiable.
        seconds = m.get("batch_seconds")
        n_windows = m.get("n_windows")
        if not isinstance(seconds, list) or not seconds:
            problems.append(f"measurements[{i}]: batch_seconds missing — the rate is unbacked")
        elif m.get("timed_batches") != len(seconds):
            problems.append(
                f"measurements[{i}]: timed_batches {m.get('timed_batches')!r} != "
                f"len(batch_seconds) {len(seconds)}"
            )
        elif _finite_positive(n_windows) and _finite_positive(m.get("windows_per_s_per_gpu")):
            total = float(sum(seconds))
            # A degenerate total would make the re-derivation raise ZeroDivisionError inside
            # the validator — a validator that crashes reports nothing, which is strictly
            # worse than one that reports a problem (CodeRabbit, P2-12 RESULT).
            if not (math.isfinite(total) and total > 0.0):
                problems.append(
                    f"measurements[{i}]: sum(batch_seconds) is {total!r}; a rate cannot be "
                    "re-derived from a non-positive total, so the recorded "
                    "windows_per_s_per_gpu is unverifiable"
                )
                continue
            want = float(n_windows) / total
            got = float(m["windows_per_s_per_gpu"])
            if not math.isclose(want, got, rel_tol=1e-9):
                problems.append(
                    f"measurements[{i}]: windows_per_s_per_gpu {got!r} does not re-derive "
                    f"from n_windows/sum(batch_seconds) = {want!r}"
                )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--checkpoint",
        action="append",
        choices=list(CHECKPOINT_KEYS),
        help="allow-list key; repeatable. Default: all three.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    parser.add_argument("--timed-batches", type=int, default=DEFAULT_TIMED_BATCHES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        checkpoints=tuple(args.checkpoint) if args.checkpoint else CHECKPOINT_KEYS,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        seed=args.seed,
        device=args.device,
    )
    problems = validate_report(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Same rule as train/ablation_table.main: an invalid report must NOT occupy the canonical
    # path. Returning 2 after already writing it is fail-open to anything that opens the file
    # rather than reading the exit code — and this artifact is git-committed evidence, which
    # the sbatch copies out of scratch on the success path (CodeRabbit, P2-12 RESULT).
    if problems:
        out = out.with_suffix(".invalid.json")
        out.write_text(payload)
        print(f"wrote {out} — NOT PUBLISHED: the report failed its own validator")
    else:
        out.write_text(payload)
        print(f"wrote {out}")

    # Render nullable fields defensively. Diverting an invalid report to `.invalid.json`
    # made this loop reachable for reports whose rates are None, where a numeric format spec
    # raises TypeError *before* the REPORT INVALID diagnostic below is ever printed — the
    # failure would be a traceback about formatting rather than the validator's actual
    # complaint (CodeRabbit, P2-12 RESULT).
    def _fmt(value: Any, spec: str) -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "n/a"

    for m in report["measurements"]:
        print(
            f"  {str(m.get('key')):<28} {_fmt(m.get('measured_param_count'), '>10,')} params  "
            f"{_fmt(m.get('windows_per_s_per_gpu'), '.2f')} win/s/GPU  "
            f"peak {_fmt(m.get('peak_vram_gib'), '.3f')} GiB"
        )
    if problems:
        print("REPORT INVALID:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
