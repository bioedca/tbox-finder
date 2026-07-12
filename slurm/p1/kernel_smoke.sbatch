#!/bin/bash
# P1-01 — Stage-1 CUDA-kernel import/forward smoke gate (mamba-ssm + causal-conv1d, sm_86).
# ADR-0002 D2/A2 (+ pre-registered A5 fallback-cost measurement); PRD §10.3, §15; CLAUDE.md §9.3.
#
# WHERE:   cluster `gpu` partition, one RTX A4000 (Ampere sm_86). NOT the laptop, NOT CI.
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout <branch> && git reset --hard origin/<branch> && \
#              sbatch slurm/p1/kernel_smoke.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p1/kernel_smoke.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): squeue empty + reports/p1/kernel_smoke.DONE present
#          + the %j.err file zero bytes.
# RETRIEVE: rsync -avzP two:~/tbox-finder/reports/p1/kernel_smoke.json <local>/reports/p1/
#          (small git-tracked JSON report — not DVC; committed with the P1-01 atomic step.)
#
#SBATCH --job-name=tbox-p1-01-kernel-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=reports/p1/kernel_smoke_%j.out
#SBATCH --error=reports/p1/kernel_smoke_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"   # deepspeed/nvcc probe CUDA_HOME at import; harmless here.

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Build in fast node-local scratch; move only the final artifact to the shared checkout.
WORK="/tmp/${USER}-${SLURM_JOB_ID:-manual}"
mkdir -p "$WORK"
OUT="$WORK/kernel_smoke.json"
DEST="$REPO/reports/p1"
mkdir -p "$DEST"

# Clear any stale artifacts from a prior run FIRST, so a failed run can never leave a
# stale success marker or a stale report behind (artifact-based verification, §9.3).
rm -f "$DEST/kernel_smoke.json" "$DEST/kernel_smoke.json.tmp" "$DEST/kernel_smoke.DONE"

set +e
PYTHONPATH=src python -m tbox_finder.kernels.kernel_smoke \
  --fixture tests/fixtures/kernel_smoke_dna/dna_tokens.npy \
  --out "$OUT" \
  --dtype float32 --iters 50 --ref-iters 10 --warmup 10
RC=$?
set -e

# Always publish the report (even on gate failure, to see why), atomically: write to a
# temp then mv (same-FS rename is atomic) so a reader never sees a half-copied file.
# DONE is created only after a successful report is in place.
if [ -f "$OUT" ]; then
  cp "$OUT" "$DEST/kernel_smoke.json.tmp"
  mv -f "$DEST/kernel_smoke.json.tmp" "$DEST/kernel_smoke.json"
fi
if [ "$RC" -eq 0 ]; then
  touch "$DEST/kernel_smoke.DONE"
  echo "SMOKE_DONE job=${SLURM_JOB_ID:-none}"
else
  echo "SMOKE_FAIL rc=$RC job=${SLURM_JOB_ID:-none}"
fi

rm -rf "$WORK"
exit "$RC"
