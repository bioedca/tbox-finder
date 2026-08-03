"""Backfill the P3-06 checkpoint provenance sidecars job 1064 could not write.

**Why this exists, and what it is honest about.** Every point of job 1064 trained, passed its
gate, and copied its checkpoint to the repo path — then crashed writing ``provenance.json``,
because the sbatch declared PEFT's ``lora_adapter`` **directory** as a provenance output and
:func:`tbox_finder.provenance.sha256_file` opens each declared output as a file. The
checkpoints are intact; only the sidecars are missing.

This reconstructs them **from each run's own report**, which is the artifact the producing
node wrote and which already carries ``git_sha``, ``env_lock_sha256``, ``seed``, the config,
the population census and the device record. Nothing here is invented: every field is either
copied from that report or computed by hashing the artifact on disk.

**It is nevertheless weaker than a sidecar written by the producing node**, and says so in
its own output: ``reconstructed: true`` plus the reason and the report it was derived from.
A reader must be able to tell a backfilled record from a native one without reading this
file. The one thing it genuinely cannot attest is that the node's working tree was unchanged
between the run and the backfill — so it copies the run's recorded ``git_sha`` rather than
re-reading git here, which would silently substitute the backfill machine's state for the
run's.

Run from the repo root, where the checkpoints and reports are::

    PYTHONPATH=src python scripts/backfill_stage2_provenance.py --check
    PYTHONPATH=src python scripts/backfill_stage2_provenance.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbox_finder.provenance import write_provenance  # noqa: E402
from tbox_finder.stage2 import train as T  # noqa: E402

SWEEP_DIR = Path("reports/p3/sweep")
CKPT_ROOT = Path("data/processed/checkpoints/stage2_rinalmo")
SIDECAR = "provenance.json"


def _points() -> list[tuple[str, Path, Path]]:
    """``(key, checkpoint_dir, report_path)`` for every point with a checkpoint on disk."""
    out: list[tuple[str, Path, Path]] = []
    for ckpt in sorted(CKPT_ROOT.glob("aux*")):
        if not ckpt.is_dir():
            continue
        report = SWEEP_DIR / f"{ckpt.name}.json"
        out.append((ckpt.name, ckpt, report))
    return out


def _sidecar(key: str, ckpt: Path, report: dict[str, Any]) -> dict[str, Any]:
    prov = report.get("provenance") or {}
    config = report.get("config") or {}
    loss = config.get("loss") or {}
    return {
        "out_path": str(ckpt / SIDECAR),
        "rule": f"slurm/p3/stage2_lora_finetune.sbatch :: {T.ENTRYPOINT} (P3-06 point {key})",
        "script": "src/tbox_finder/stage2/train.py",
        "seed": prov.get("seed"),
        "inputs": [
            report.get("data", {}).get("dataset_parquet"),
            "src/tbox_finder/stage2/head_vocab.json",
        ],
        # The defect that caused this backfill, fixed via the shared enumerator.
        "outputs": T.checkpoint_output_files(ckpt),
        "env_lock": prov.get("env_lock"),
        "adr": "ADR-0002; ADR-0004 D5; ADR-0005 D16",
        "extra": {
            "step": T.STEP,
            "point": key,
            "report": str(SWEEP_DIR / f"{key}.json"),
            # Copied from the RUN's report, never re-read here: re-reading git would record
            # the backfill machine's state as though it were the run's.
            "git_sha": prov.get("git_sha"),
            "git_branch": prov.get("git_branch"),
            "env_lock_sha256": prov.get("env_lock_sha256"),
            "world_size": (report.get("steps") or {}).get("world_size"),
            "device": report.get("device"),
            "loss_aux_weight": loss.get("aux_weight"),
            "optim_lr": config.get("lr"),
            "gate_overall_pass": (report.get("gate") or {}).get("overall_pass"),
            "best_val_total": (report.get("val") or {}).get("best_total"),
            "best_val_epoch": (report.get("val") or {}).get("best_epoch"),
            "population": (report.get("data") or {}).get("train_census"),
            # ── The honesty block. A reader must not have to infer this. ──
            "reconstructed": True,
            "reconstructed_reason": (
                "job 1064 crashed writing this sidecar with IsADirectoryError: the sbatch "
                "declared PEFT's lora_adapter DIRECTORY as a provenance output and "
                "provenance.sha256_file opens each output as a file. Training, the gate and "
                "the checkpoint copy all succeeded; only the sidecar was lost. Rebuilt from "
                "the run's own report by scripts/backfill_stage2_provenance.py."
            ),
            "reconstructed_from": str(SWEEP_DIR / f"{key}.json"),
            "reconstruction_limitation": (
                "written after the fact, not by the producing node, so it cannot attest that "
                "the node's working tree was unchanged between the run and this write. "
                "git_sha is COPIED from the run's report rather than re-read here. Output "
                "hashes ARE computed from the artifacts on disk, so they verify the retrieved "
                "copy."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="report what would be written")
    group.add_argument("--write", action="store_true", help="write the sidecars")
    args = parser.parse_args(argv)

    points = _points()
    if not points:
        print(f"FATAL: no checkpoint directories under {CKPT_ROOT}", file=sys.stderr)
        return 2

    problems: list[str] = []
    planned: list[dict[str, Any]] = []
    for key, ckpt, report_path in points:
        if not report_path.is_file():
            problems.append(f"{key}: no report at {report_path}")
            continue
        report = json.loads(report_path.read_text())
        # A sidecar must never describe a run that failed its own gate.
        if not (report.get("gate") or {}).get("overall_pass"):
            problems.append(f"{key}: report gate did not pass; refusing to write provenance")
            continue
        try:
            payload = _sidecar(key, ckpt, report)
        except (FileNotFoundError, NotADirectoryError) as exc:
            problems.append(f"{key}: {exc}")
            continue
        planned.append(payload)
        n_out = len(payload["outputs"])
        print(
            f"{key:>16}  outputs={n_out}  git_sha={payload['extra']['git_sha']}  "
            f"driver={(payload['extra'].get('device') or {}).get('driver_version')}  "
            f"gate_pass={payload['extra']['gate_overall_pass']}"
        )

    if problems:
        print("\nREFUSED:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)

    if args.check:
        print(f"\nwould write {len(planned)} sidecar(s); {len(problems)} refused")
        return 1 if problems else 0

    n_written = 0
    for payload in planned:
        out_path = payload.pop("out_path")
        run_sha = payload["extra"]["git_sha"]
        written = write_provenance(out_path, **payload)
        # ⚠ `build_provenance` derives the TOP-LEVEL git_sha from the working tree it runs in
        # and takes no parameter to override it — so a backfill from a checkout that has moved
        # on would silently attribute this run to the wrong commit, in the very field
        # CLAUDE.md §11 names. Verified rather than patched: if the tree is not sitting on the
        # run's own commit, the sidecar is deleted and the point refused. That makes "run the
        # backfill before re-syncing the checkout" an enforced precondition instead of
        # something I have to remember.
        record = json.loads(Path(written).read_text())
        if record.get("git_sha") != run_sha:
            Path(written).unlink()
            problems.append(
                f"{payload['extra']['point']}: this checkout is at "
                f"{record.get('git_sha')!r} but the run was {run_sha!r} — refusing to write a "
                "sidecar that would attribute the run to the wrong commit. Check out the "
                "run's commit and retry."
            )
            print(f"REFUSED {out_path} (checkout moved)", file=sys.stderr)
            continue
        n_written += 1
        print(f"wrote {out_path}")
    # Counted, not assumed: a refusal at write time must not be reported as a write.
    print(f"\nwrote {n_written} sidecar(s); {len(problems)} refused")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
