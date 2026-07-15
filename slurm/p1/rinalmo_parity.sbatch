#!/bin/bash
# P1-13 — RiNALMo mirror parity: per-fold ArchiveII nine-family LOFO SS fine-tune +
# base-pair-F1 score (PRD §10.2; ADR-0002 D5/A8; CLAUDE.md §9.3). ARRAY job: one
# task per leave-one-family-out fold (0..8 → FAMILY_ORDER). Each task fine-tunes the
# multimolecule/rinalmo-giga mirror's fresh SS head on the eight-family complement
# and scores the held-out family; the pure-stdlib `--mode aggregate` step (a
# follow-up, NOT in this array) decides parity from the nine per-fold JSONs.
#
# WHERE:   cluster `gpu` partition, one RTX A4000 (Ampere sm_86) per array task. NOT
#          the laptop, NOT CI. env tbox-ml-rna (transformers 5.13.0 + multimolecule
#          0.1.0; ADR-0002 A8). est-wall ~2.5-4 h/fold (laptop 4060 sizing smoke:
#          80 ms/step head-only → 139 ms/step at 6 unfrozen layers; ~15×~3100 steps
#          + eval); --time 6h is headroom. 9 tasks × ~3.5 h ≈ ~32 GPU-h; wall ≈ the
#          slowest fold if the scheduler runs the array in parallel (8×A4000/node).
# INPUT:   the dl-rna ct/fam-fold splits (ct-splits.tar.gz, pinned sha256 below) —
#          NOT in the checkout (data/external .ct is gitignored, §5.2). This job
#          DOWNLOADS + checksum-verifies + extracts them to node-local /tmp; the
#          rinalmo-giga weights (2.5 GB) download to a node-local HF cache (off the
#          tight HOME quota). Fail-loud on any checksum/extract drift.
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout <branch> && git reset --hard origin/<branch> && \
#              sbatch slurm/p1/rinalmo_parity.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p1/rinalmo_parity.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): squeue empty + all nine
#          reports/p1/rinalmo_parity_folds/SHARD_OK_{0..8} present + a self-consistent
#          <family>.json per fold (measured=true, non_weighted_mean_f1 ∈ [0,1]). The
#          %A_%a.err is EXPECTED non-empty on success (multimolecule/transformers load
#          notes + fresh-head init warnings); scan for real tracebacks, do not gate on
#          zero bytes. A missing SHARD_OK ⇒ STOP and ask (§9.3; no auto-resubmit).
# RETRIEVE + AGGREGATE (a deliberate follow-up step — carries the ADR-0002 parity
#          sign-off gate): rsync the fold JSONs back, then decide parity locally:
#            rsync -avzP two:~/tbox-finder/reports/p1/rinalmo_parity_folds/ <local>/reports/p1/rinalmo_parity_folds/
#            PYTHONPATH=src python -m tbox_finder.eval.archiveii_lofo --mode aggregate
#          → writes reports/p1/rinalmo_parity.json (measured verdict PASS→commit / FAIL→Zenodo fallback).
#
#SBATCH --job-name=tbox-p1-13-rinalmo-parity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --output=reports/p1/rinalmo_parity_%A_%a.out
#SBATCH --error=reports/p1/rinalmo_parity_%A_%a.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

# Pinned dl-rna splits (must match src/tbox_finder/eval/archiveii_lofo.py SPLITS_*).
SPLITS_URL="https://github.com/marcellszi/dl-rna/releases/download/Data/ct-splits.tar.gz"
SPLITS_SHA256="0ff7209016ba1775288572794e850d5857937c391d67c9ccacfe9ae77c0188e5"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID is required — submit as an array job (--array=0-8), not plain sbatch" >&2
  exit 2
fi
TASK="$SLURM_ARRAY_TASK_ID"
if [ "$TASK" -lt 0 ] || [ "$TASK" -gt 8 ]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID=$TASK out of range [0,8]" >&2
  exit 2
fi

# Repo/output paths + a PER-TASK singleton lock + per-task stale cleanup FIRST — before
# the failure-prone download/conda so a failed run cannot leave a stale SHARD_OK/JSON,
# and concurrent array tasks (different folds) never collide on each other's outputs.
REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"
FOLD_DIR="$REPO/reports/p1/rinalmo_parity_folds"
mkdir -p "$FOLD_DIR"

exec 9>"$FOLD_DIR/.parity_${TASK}.lock"
if ! flock -n 9; then
  echo "PARITY_SKIP: another job holds the fold-$TASK lock — refusing to share outputs." >&2
  exit 1
fi
# Clear THIS task's stale success marker FIRST — before the failure-prone download/conda —
# so a run that dies during activation/import can never leave a prior run's SHARD_OK behind
# (artifact-based verification keys on its absence-on-failure, §9.3). Needs only $TASK; the
# per-fold JSON is cleared after FAM is resolved (it needs the family name).
rm -f "$FOLD_DIR/SHARD_OK_${TASK}"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-rna
export CUDA_HOME="$CONDA_PREFIX"          # multimolecule→danling→deepspeed probe CUDA_HOME at import (ADR-0002 A4).
export WANDB_MODE=offline                 # defensive: no live streaming on compute nodes (§9.2).
export PYTHONHASHSEED=0                    # determinism (§8.3).
export TOKENIZERS_PARALLELISM=false

# Resolve this task's family from the code (single source of truth), then clear ONLY
# this fold's stale outputs (never the sibling folds').
FAM=$(PYTHONPATH=src python -c "from tbox_finder.eval.archiveii_lofo import FAMILY_ORDER as F; print(F[$TASK])")
rm -f "$FOLD_DIR/$FAM.json"

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-none} task=$TASK family=$FAM sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Node-local scratch: the tarball, its extraction, and the HF weight cache (off HOME).
BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.
export HF_HOME="$BUILD/hf"

# Download + checksum-verify + extract the dl-rna splits (fail-loud on drift, §10.2/§10.3).
TARBALL="$BUILD/ct-splits.tar.gz"
curl -fsSL --retry 3 -o "$TARBALL" "$SPLITS_URL"
echo "${SPLITS_SHA256}  ${TARBALL}" | sha256sum -c -
mkdir -p "$BUILD/ext"
tar -xzf "$TARBALL" -C "$BUILD/ext"
ROOT="$BUILD/ext/ct/fam-fold"
[ -d "$ROOT" ] || { echo "FATAL: fam-fold root missing after extract: $ROOT" >&2; exit 2; }

# Fine-tune + score this fold (heavy torch/multimolecule; the fold JSON self-validates
# F1 ∈ [0,1] and measured=true, else run_single_fold raises).
set +e
PYTHONPATH=src python -m tbox_finder.eval.archiveii_lofo \
  --mode fold --fold-index "$TASK" --fam-fold-root "$ROOT" --fold-dir "$FOLD_DIR"
RC=$?
set -e

# Publish the per-shard marker ONLY on success AND a non-empty measured fold JSON.
if [ "$RC" -eq 0 ] && [ -s "$FOLD_DIR/$FAM.json" ]; then
  touch "$FOLD_DIR/SHARD_OK_${TASK}"
  echo "PARITY_FOLD_DONE task=$TASK family=$FAM job=${SLURM_ARRAY_JOB_ID:-none}"
else
  echo "PARITY_FOLD_FAIL rc=$RC task=$TASK family=$FAM job=${SLURM_ARRAY_JOB_ID:-none}" >&2
  [ "$RC" -ne 0 ] || RC=1
fi
exit "$RC"
