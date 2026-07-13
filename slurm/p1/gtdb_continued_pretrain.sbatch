#!/bin/bash
# P1-11 — Probe-triggered GTDB-prokaryotic continued-pretraining (MLM) pass
# (ADR-0002 D7 fallback #2; ADR-0003 D7 budget; PRD §10.1/§15; CLAUDE.md §7/§9.3).
#
# PROBE-TRIGGERED — NOT AUTO-RUN. The measured P1-07 verdict was GO, so this pass is
# SKIPPED. This sbatch is the AUTHORED, executable fallback the PRD §18.1 exit gate requires.
# It is submitted ONLY after a compute-budget §7 sign-off + a §9.3 SLURM submit-ack — never
# autonomously (CLAUDE.md §13). est-wall is UNKNOWN → a trigger-time sizing smoke fills it +
# the max_steps budget before the submit-ack; --time below is a conservative PROVISIONAL
# bound (ADR-0003 D7: the training/sweep GPU-hour budget's numeric value is frozen at P1
# first measurement — NOT a fabricated GPU-hour figure).
#
# WHERE:   cluster `gpu` partition, 4× RTX A4000 (Ampere sm_86), one node. NOT the laptop/CI.
#          The packaged Mamba forward hard-requires selective_scan_cuda (ADR-0002 A2 C2) →
#          GPU only. Data-parallel via `accelerate launch` (7.73 M params → plain DDP).
# INPUT:   data/interim/gtdb_cpt/shard.fasta[.gz] — the GTDB-prokaryotic DNA pretraining
#          shard, a REQUIRED external input sourced at trigger-time (a checksummed download,
#          CLAUDE.md §10.2). It does NOT ship in the repo and is NOT fabricated. The job
#          ASSERTS its presence and fails loud otherwise.
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout <branch> && git reset --hard origin/<branch> && \
#              sbatch slurm/p1/gtdb_continued_pretrain.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p1/gtdb_continued_pretrain.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): squeue empty + reports/p1/gtdb_cpt.DONE present +
#          a self-validated reports/p1/gtdb_cpt.json + a non-empty checkpoint. The %j.err file
#          is EXPECTED to be NON-empty on success (torch + transformers + trust_remote_code +
#          accelerate write benign warnings — the warn_only determinism notice, wandb-offline
#          setup, remote-code load notes); DONE + a valid report are the authoritative signal.
#          Scan .err for real tracebacks/CUDA/NCCL errors; do not gate on it being zero-byte.
# RETRIEVE: report (git-tracked): rsync -avzP two:~/tbox-finder/reports/p1/gtdb_cpt.json <local>/reports/p1/
#           checkpoint (DVC-tracked): rsync -avzP two:~/tbox-finder/checkpoints/p1/caduceus_gtdb_cpt/ <local>/checkpoints/p1/caduceus_gtdb_cpt/
#             then `dvc add checkpoints/p1/caduceus_gtdb_cpt` locally (push deferred to phase-exit, §5.2).
#
#SBATCH --job-name=tbox-p1-11-gtdb-cpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=reports/p1/gtdb_cpt_%j.out
#SBATCH --error=reports/p1/gtdb_cpt_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

# Repo/output paths, an exclusive singleton lock, and stale-artifact cleanup happen FIRST —
# before the failure-prone conda activation / input preflight — so (a) a concurrent gtdb_cpt
# job can never share the report/DONE/checkpoint paths, and (b) an activation or preflight
# failure can never leave a stale gtdb_cpt.DONE / gtdb_cpt.json behind (artifact-based
# verification depends on their absence-on-failure, §9.3).
REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"
DEST="$REPO/reports/p1"
mkdir -p "$DEST"

exec 9>"$DEST/.gtdb_cpt.lock"
if ! flock -n 9; then
  echo "CPT_SKIP: another gtdb_cpt job holds $DEST/.gtdb_cpt.lock — refusing to share outputs." >&2
  exit 1
fi

# Node-local build dir (§9.2: build in node-local /tmp, move only the final checkpoint to
# HOME). All 4 processes are on this one node, so a node-local path is shared among them.
BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
# Clean up node-local scratch on EVERY exit path (early exit 2 on a missing shard, a set -e
# abort during the HOME copy, or normal completion) — §9.2 "clean up /tmp on exit". Installed
# after mkdir so it never fires before $BUILD exists; touches only $BUILD, not the fd-9 lock.
trap 'rm -rf "$BUILD"' EXIT
FINAL_CKPT_DIR="$REPO/checkpoints/p1/caduceus_gtdb_cpt"
CKPT="$FINAL_CKPT_DIR/caduceus_gtdb_cpt.pt"
PROVENANCE="$FINAL_CKPT_DIR/provenance.json"
REPORT="$DEST/gtdb_cpt.json"
# Clear every stale output so a failed run cannot leave a stale success marker or let the
# postcondition pass on a prior run's blobs.
rm -f "$REPORT" "$DEST/gtdb_cpt.DONE" "$CKPT" "$PROVENANCE"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"      # deepspeed/nvcc probe CUDA_HOME at import.
export WANDB_MODE=offline             # compute nodes: no live streaming (§9.2); sync later.
export PYTHONHASHSEED=0               # determinism (§8.3; matches conf/train/gtdb_cpt.yaml).
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src                 # tbox_finder lives under src/ and is NOT pip-installed in
                                     # tbox-ml-dna; accelerate launch's workers inherit this
                                     # (the kernel_smoke/seg_smoke sbatch `PYTHONPATH=src` pattern).

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Assert the GTDB shard is staged (the job does not source it — see INPUT above). Accept the
# plain or gzipped form.
SHARD="data/interim/gtdb_cpt/shard.fasta"
if [ ! -s "$REPO/$SHARD" ] && [ ! -s "$REPO/$SHARD.gz" ]; then
  echo "FATAL: GTDB shard $SHARD[.gz] missing/empty — source it (checksummed download, §10.2) before submit." >&2
  exit 2
fi
[ -s "$REPO/$SHARD.gz" ] && SHARD="$SHARD.gz"

# Multi-GPU continued-pretraining via accelerate (plain DDP, mixed_precision off for
# determinism). Hydra entry (no `--`-prefixed args → Hydra path, not the argparse CLI); the
# checkpoint is built node-local and moved to HOME afterward; the report is written straight
# to the git-tracked repo path.
set +e
# --num_processes tracks the actual GPU allocation (SLURM_GPUS_ON_NODE) so an edit to
# --gres=gpu:a4000:<N> above stays in lockstep without a second hand-edit (defaults to 4).
accelerate launch --multi_gpu --num_processes "${SLURM_GPUS_ON_NODE:-4}" --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m tbox_finder.train.continued_pretrain \
  data.shard_path="$SHARD" \
  checkpoint_dir="$BUILD/caduceus_gtdb_cpt" \
  report_path="$REPORT"
RC=$?

# Move the final checkpoint + provenance from node-local /tmp to HOME (§9.2). Kept under
# `set +e` (non-fatal): a failing cp — e.g. a tight HOME quota — must NOT abort the script
# before the DONE-gate + CPT_FAIL breadcrumb below; a partial copy leaves $CKPT empty, so the
# DONE gate correctly fails and logs CPT_FAIL (artifact-based verification stays sound).
if [ "$RC" -eq 0 ] && [ -s "$BUILD/caduceus_gtdb_cpt/caduceus_gtdb_cpt.pt" ]; then
  mkdir -p "$FINAL_CKPT_DIR"
  cp -f "$BUILD/caduceus_gtdb_cpt/caduceus_gtdb_cpt.pt" "$CKPT"
  cp -f "$BUILD/caduceus_gtdb_cpt/provenance.json" "$PROVENANCE"
fi
set -e

# Publish DONE only when the run succeeded AND all required outputs (report + checkpoint +
# provenance) are non-empty; a missing artifact forces a nonzero exit even if accelerate
# returned 0.
if [ "$RC" -eq 0 ] \
  && [ -s "$REPORT" ] \
  && [ -s "$CKPT" ] \
  && [ -s "$PROVENANCE" ]; then
  touch "$DEST/gtdb_cpt.DONE"
  echo "CPT_DONE job=${SLURM_JOB_ID:-none}"
else
  echo "CPT_FAIL rc=$RC job=${SLURM_JOB_ID:-none}"
  [ "$RC" -ne 0 ] || RC=1
fi

# Node-local scratch is cleaned by the EXIT trap (installed after mkdir "$BUILD" above).
exit "$RC"
