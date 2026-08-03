#!/bin/bash
# P3-06 SIZING SMOKE — measure the real Stage-2 step on ONE GPU before re-sizing the sweep.
# (PRD §10.3 the 16 GB A4000 budget; ADR-0002 A4 env split; CLAUDE.md §9.3.)
#
# ── WHY THIS EXISTS ─────────────────────────────────────────────────────────
# imp.md's P3-06 Compute line says "re-confirm est via a 100-record smoke before submit".
# That was skipped: the sweep was sized from reports/p1/lora_vram_smoke.json (P1-16, batch 8
# -> 2.014 GiB). P1-16's own config block says what it measured — `loss_is_placeholder: true`,
# `data_is_synthetic: true`, `is_science: false`, `seq_len_nt: 350`, 5 steps — i.e. a bare
# LoRA-wrapped BACKBONE under a placeholder loss, no heads, no objective. The real seven-head
# step on real sequences allocated 13.97 GiB and OOM'd all six points of job 1044. P1-16's
# number was not wrong; it answered a different question. This run answers the right one.
#
# It measures THE SHIPPED STEP: stage2/sizing.py calls train.forward_backward — the same
# function the trainer calls — over the same dataset class and the same MultitaskLoss, on
# REAL rows from the REAL admitted population. A harness that reimplemented the step would be
# free to drift from it in exactly the way P1-16 did.
#
# ── WHAT IT MEASURES ────────────────────────────────────────────────────────
#  * a DESCENDING batch sweep (8, 4, 2, 1), OOM caught per point so one failure does not end it
#  * both a WORST-CASE batch (the longest rows — the collator pads everything to the batch
#    max, and this corpus runs 59-552 tokens) and a TYPICAL one; sizing on the median is how
#    job 1044 happened
#  * SIX optimiser steps per point, because AdamW allocates its state lazily on the first
#    .step() and because a footprint that GROWS across steps — the open question job 1044
#    left — is only visible as a per-step series
#  * gradient-checkpointing effectiveness as an explicit on/off comparison, not an assumption
#
# ── COST ────────────────────────────────────────────────────────────────────
# ONE GPU, no DDP. ~9 measurement points x (model build + 6 steps) plus one checkpointing
# comparison. --time=00:45:00 is generous; the run is expected in ~10-20 min, dominated by
# rebuilding the 650M backbone per point (deliberate: a shared model would let one point's
# allocator state set the next point's peak).
#
# WHERE:   cluster `gpu` partition, ONE RTX A4000 (sm_86). env tbox-ml-rna (ADR-0002 A4).
#          No node carve-out: `two` was rebooted 2026-08-03 into kernel 6.8.0-134-generic with
#          a resident nvidia 595.84 matching userspace, verified. ⚠ The nodes now run DIFFERENT
#          drivers (`one` 590.48.01, `two` 595.84), so a re-measurement may land on either —
#          the report's `device` block records which, and a sizing number should be read
#          against the driver that produced it rather than assumed portable.
# INPUT:   data/processed/stage2_dataset.parquet, staged by rsync and verified below.
# SUBMIT:  ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p3/stage2_sizing_smoke.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p3/stage2_sizing_smoke.sbatch"'
# VERIFY (artifact-based; sacct off): (a) squeue empty, (b) reports/p3/stage2_sizing.DONE,
#          (c) reports/p3/stage2_sizing.json validates + passes its own gate. ⚠ the gate grades
#          the MEASUREMENT, not the answer: "batch 8 does not fit" is a SUCCESSFUL run, and
#          the job exits 0 for it. ⚠ .err is EXPECTED NON-EMPTY (HF chatter, OOM tracebacks
#          that were caught and recorded) — scan it, do not gate on zero bytes.
# RETRIEVE: rsync -avzP two:~/tbox-finder/reports/p3/stage2_sizing.json <local>/reports/p3/

#SBATCH --job-name=tbox-p3-06-sizing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=reports/p3/stage2_sizing_%j.out
#SBATCH --error=reports/p3/stage2_sizing_%j.err
# No --account / --qos (accounting disabled), no --nodelist.

set -eo pipefail   # NOT -u: the conda activate hooks read unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

OUT_DIR="reports/p3"
REPORT="$OUT_DIR/stage2_sizing.json"
DONE="$OUT_DIR/stage2_sizing.DONE"
mkdir -p "$OUT_DIR"

exec 9>"$OUT_DIR/.stage2_sizing.lock"
if ! flock -n 9; then
  echo "SIZING_SKIP: another job holds the lock — refusing to share outputs." >&2
  exit 1
fi

# ── Staged-input assertion ──────────────────────────────────────────────────
assert_dvc_artifact() {
  local data_path="$1" dvc_path="$2" want got
  if [ ! -s "$data_path" ]; then
    echo "FATAL: $data_path missing. rsync it from the laptop." >&2; exit 2
  fi
  if [ "$(head -c 4 "$data_path")" != "PAR1" ]; then
    echo "FATAL: $data_path is not a parquet (no PAR1 magic)." >&2; exit 2
  fi
  want="$(awk '/^- md5:/ {print $3; exit}' "$dvc_path")"
  got="$(md5sum "$data_path" | awk '{print $1}')"
  if [ -z "$want" ]; then echo "FATAL: no md5 in $dvc_path" >&2; exit 2; fi
  if [ "$want" != "$got" ]; then
    echo "FATAL: $data_path md5 $got != committed pointer $want." >&2; exit 2
  fi
  echo "staged-ok $data_path md5=$got"
}
assert_dvc_artifact data/processed/stage2_dataset.parquet \
                    data/processed/stage2_dataset.parquet.dvc

# Untracked outputs only — nothing here can dirty a git-tracked path.
rm -f "$DONE"   # untracked marker only — see the fingerprint below

# ── Freshness fingerprint, taken INSTEAD of deleting the report ──────────────
# "$REPORT" is now a GIT-TRACKED path (the P3-06 RESULT commit landed all six sweep
# reports), so `rm`-ing it would dirty the tree outside `_DATA_STAGING_PREFIXES` — the
# P2-09/P2-10d'-c ruling, and what tests/unit/test_sbatch_rm_targets.py enforces. Note the
# test's verdict CHANGED when those reports were committed: the same `rm` was safe while
# they were untracked. So freshness is proved by fingerprint, not by absence.
#
# Absent file -> empty string, which no md5 can equal, so "was missing" and "was replaced"
# both read as fresh and only "unchanged bytes" reads as stale.
REPORT_BEFORE="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-rna

export CUDA_HOME="$CONDA_PREFIX"
export WANDB_MODE=offline
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ── Node GPU health check — fail fast, before the 2.5 GB download ────────────
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  echo "STAGE2_NODE_UNHEALTHY $(hostname): nvidia-smi failed — GPU driver stack unusable." >&2
  exit 2
fi
if ! PYTHONPATH=src python -c '
import sys, torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
torch.zeros(1, device="cuda:0")
print(f"cuda ok: {torch.cuda.device_count()} device(s), capability {torch.cuda.get_device_capability(0)}")
'; then
  echo "STAGE2_NODE_UNHEALTHY $(hostname): CUDA will not initialise under torch." >&2
  exit 2
fi

JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}-sizing"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT
export HF_HOME="$JOB_SCRATCH/hf"
export WANDB_DIR="$JOB_SCRATCH"

# Warm the HF cache once (single process here, but keeps the measured region free of the
# download) and record the backend the real run selects.
PYTHONPATH=src python -c '
import torch
from tbox_finder.eval.rinalmo_parity import REPO_ID, REVISION
from tbox_finder.train import lora_harness as LH
model = LH.load_rinalmo_backbone()
warm_attn = getattr(model.config, "_attn_implementation", None)
evidence = LH.read_sm86_flash_attn_evidence()
selected, reason = LH.select_attention_backend(
    flash_attn_importable=evidence["flash_attn_importable"],
    sm86_confirmed=evidence["is_sm86"],
    model_supports_flash_attn=LH.model_supports_flash_attn(),
    dtype=LH.TRAIN_DTYPE,
)
print(f"hf cache warm {REPO_ID} {REVISION} torch={torch.__version__}")
print(f"warm-load attn (NOT the run backend)={warm_attn}")
print(f"run will select attn={selected} :: {reason}")
'

# ── The sweep ────────────────────────────────────────────────────────────────
# Single process, ONE GPU: DDP is deliberately out of scope here. This measures the step's
# own footprint; DDP adds gradient buckets on top, and conflating the two is what makes a
# sizing number un-actionable.
echo "── sizing sweep (1 GPU, real rows, real objective, real step) ──"
set +e
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.stage2.sizing \
  --dataset data/processed/stage2_dataset.parquet \
  --batch-sweep 8,4,2,1 \
  --steps 6 \
  --max-records 200 \
  --out "$REPORT"
rc=$?
set -e

if [ "$rc" -ne 0 ] || [ ! -s "$REPORT" ]; then
  echo "SIZING_FAIL rc=$rc (report not written, or it failed its own measurement gate)" >&2
  [ "$rc" -ne 0 ] || rc=1
  exit "$rc"
fi


# The report is no longer cleared before the run, so "a report exists" no longer implies
# "THIS run wrote it" — a stale committed one would satisfy `[ -s ]`. Assert it CHANGED.
# Both failure directions are explicit: an UNCOMPUTABLE fingerprint must not pass by being
# unequal to the old one, and an UNCHANGED one must not pass at all.
REPORT_AFTER="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"
if [ -z "$REPORT_AFTER" ]; then
  echo "SIZING_FAIL: could not fingerprint $REPORT after the run (missing or unreadable)." >&2
  exit 1
fi
if [ "$REPORT_BEFORE" = "$REPORT_AFTER" ]; then
  echo "SIZING_FAIL: $REPORT is byte-identical to the one that was there before this run" >&2
  echo "  (md5 $REPORT_AFTER) — this run did not write it, so it describes some OTHER run." >&2
  exit 1
fi

touch "$DONE"
echo "SIZING_DONE job=${SLURM_JOB_ID:-none} report=$REPORT"
