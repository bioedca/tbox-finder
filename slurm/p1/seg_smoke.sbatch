#!/bin/bash
# P1-07 — Stage-1 per-nucleotide 8-class segmentation smoke fine-tune + BINDING transfer
# go/no-go (ADR-0002 D7; ADR-0004 D6; ADR-0005; PRD §10.1/§11; CLAUDE.md §9.3).
#
# WHERE:   cluster `gpu` partition, one RTX A4000 (Ampere sm_86). NOT the laptop, NOT CI.
#          The packaged Mamba forward hard-requires selective_scan_cuda (ADR-0002 A2 C2) →
#          GPU only. est-wall ~20-25 min (300 windows × 20 epochs; laptop sm_89 sizing:
#          2 epochs = 2.5 min incl. backbone load); --time 1h is headroom.
# INPUT:   data/interim/p1_seg_smoke/{windows.parquet,labels.npy} — DVC-tracked (P1-06),
#          STAGED INTO THE CHECKOUT BEFORE SUBMIT (targeted `dvc push` from the laptop →
#          `dvc pull` on the login node, or rsync from the laptop). The job ASSERTS their
#          presence and fails loud otherwise (it does not run dvc; the ml-dna env has none).
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout <branch> && git reset --hard origin/<branch> && \
#              sbatch slurm/p1/seg_smoke.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p1/seg_smoke.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): squeue empty + reports/p1/seg_smoke.DONE present
#          + a self-validated reports/p1/seg_smoke_gonogo.json. NOTE: unlike the P1-01 kernel
#          smoke, the %j.err file is EXPECTED to be NON-empty on success — a torch +
#          transformers + trust_remote_code GPU run reliably writes benign warnings to stderr
#          (the warn_only determinism notice for the Mamba selective-scan kernel, wandb-offline
#          setup, remote-code load notes). DONE + a valid report are the authoritative success
#          signal; scan .err for real tracebacks/CUDA errors, do not gate on it being zero-byte.
# RETRIEVE: report (git-tracked): rsync -avzP two:~/tbox-finder/reports/p1/seg_smoke_gonogo.json <local>/reports/p1/
#           checkpoint (DVC-tracked): rsync -avzP two:~/tbox-finder/checkpoints/p1/seg_smoke/ <local>/checkpoints/p1/seg_smoke/
#             then `dvc add checkpoints/p1/seg_smoke` locally (push deferred to phase-exit, §5.2).
#
#SBATCH --job-name=tbox-p1-07-seg-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=reports/p1/seg_smoke_%j.out
#SBATCH --error=reports/p1/seg_smoke_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"      # deepspeed/nvcc probe CUDA_HOME at import.
export WANDB_MODE=offline             # compute nodes: no live streaming (§9.2); sync later.
export PYTHONHASHSEED=0               # determinism (§8.3; matches conf/train/stage1_smoke.yaml).

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Assert the DVC-tracked smoke set is staged (the job does not run dvc — see INPUT above).
WIN="data/interim/p1_seg_smoke/windows.parquet"
LAB="data/interim/p1_seg_smoke/labels.npy"
for f in "$WIN" "$LAB"; do
  if [ ! -s "$REPO/$f" ]; then
    echo "FATAL: input $f is missing/empty — stage the P1-06 smoke set (dvc pull / rsync) before submit." >&2
    exit 2
  fi
done

DEST="$REPO/reports/p1"
mkdir -p "$DEST"
# Clear any stale artifacts FIRST so a failed run cannot leave a stale success marker.
rm -f "$DEST/seg_smoke_gonogo.json" "$DEST/seg_smoke.DONE"

# Hydra entry (no `--` args → the Hydra path, not the argparse CLI). Writes the report to
# reports/p1/seg_smoke_gonogo.json and the checkpoint to checkpoints/p1/seg_smoke/ (config
# defaults). run_smoke self-validates the report and raises on any inconsistency.
set +e
PYTHONPATH=src python -m tbox_finder.train.stage1_smoke
RC=$?
set -e

if [ "$RC" -eq 0 ] && [ -f "$DEST/seg_smoke_gonogo.json" ]; then
  touch "$DEST/seg_smoke.DONE"
  echo "SMOKE_DONE job=${SLURM_JOB_ID:-none}"
else
  echo "SMOKE_FAIL rc=$RC job=${SLURM_JOB_ID:-none}"
fi
exit "$RC"
