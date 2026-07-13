#!/bin/bash
# P1-08 — Stage-1 segmentation-smoke REPRODUCIBILITY re-run (ADR-0002 A7; PRD §8.3/§11/§18.1;
# CLAUDE.md §9.3). Re-runs the P1-07 BINDING transfer go/no-go (ADR-0002 A6) with the SAME
# committed config (conf/train/stage1_smoke.yaml: seed 42, PYTHONHASHSEED=0, TF32/cudnn off)
# and the UNCHANGED harness (src/tbox_finder/train/stage1_smoke.py::main) — ONLY the output
# paths are overridden (report_path / checkpoint_dir → *_repro), so the re-run cannot clobber
# the committed P1-07 report or its DVC-tracked checkpoint. The P1-08 gate then checks the
# re-run reproduces the P1-07 go/no-go within the ADR-0002 A7 tolerance (verdict + max|Δ|≤1e-3).
#
# WHERE:   cluster `gpu` partition, one RTX A4000 (Ampere sm_86). NOT the laptop, NOT CI.
#          The packaged Mamba forward hard-requires selective_scan_cuda (ADR-0002 A2 C2) →
#          GPU only. est-wall ~20-25 min (300 windows × 20 epochs; same as P1-07 job 464);
#          --time 1h is headroom. NOTE (ADR-0002 A7): the Mamba selective_scan kernel has no
#          deterministic variant registered (use_deterministic_algorithms warn_only=True), so
#          the re-run is within-tolerance, NOT bit-exact — the pinned tolerance is load-bearing.
# INPUT:   data/interim/p1_seg_smoke/{windows.parquet,labels.npy} — DVC-tracked (P1-06),
#          STAGED INTO THE CHECKOUT BEFORE SUBMIT (already present from the P1-07 run, or
#          `dvc pull` / rsync). The job ASSERTS their presence and fails loud otherwise (it
#          does not run dvc; the ml-dna env has none).
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout <branch> && git reset --hard origin/<branch> && \
#              sbatch slurm/p1/seg_smoke_repro.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p1/seg_smoke_repro.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): squeue empty + reports/p1/seg_smoke_repro.DONE present
#          + a self-validated reports/p1/seg_smoke_repro.json. As with P1-07, the %j.err file is
#          EXPECTED to be NON-empty on success — a torch + transformers + trust_remote_code GPU
#          run reliably writes benign warnings to stderr (the warn_only determinism notice for
#          the Mamba selective-scan kernel, wandb-offline setup, remote-code load notes). DONE +
#          a valid report are the authoritative success signal; scan .err for real
#          tracebacks/CUDA errors, do not gate on it being zero-byte.
# RETRIEVE: report only (git-tracked): rsync -avzP two:~/tbox-finder/reports/p1/seg_smoke_repro.json <local>/reports/p1/
#           The repro checkpoint (checkpoints/p1/seg_smoke_repro/) is a throwaway success-signal
#           for the DONE gate — NOT a deliverable and NOT DVC-tracked; leave it on the cluster.
#
#SBATCH --job-name=tbox-p1-08-seg-smoke-repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=reports/p1/seg_smoke_repro_%j.out
#SBATCH --error=reports/p1/seg_smoke_repro_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

# Repo/output paths, an exclusive singleton lock, and stale-artifact cleanup happen FIRST —
# before the failure-prone conda activation / input preflight — so (a) a concurrent
# seg_smoke_repro job can never share the report/DONE/checkpoint paths, and (b) an activation
# or preflight failure can never leave a stale seg_smoke_repro.DONE / seg_smoke_repro.json
# behind (artifact-based verification depends on their absence-on-failure, §9.3). A distinct
# lock/report/checkpoint namespace from P1-07 means a stray concurrent P1-07 job is orthogonal.
REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"
DEST="$REPO/reports/p1"
mkdir -p "$DEST"

exec 9>"$DEST/.seg_smoke_repro.lock"
if ! flock -n 9; then
  echo "REPRO_SKIP: another seg_smoke_repro job holds $DEST/.seg_smoke_repro.lock — refusing to share outputs." >&2
  exit 1
fi
CHECKPOINT="$REPO/checkpoints/p1/seg_smoke_repro/seg_smoke.pt"
PROVENANCE="$REPO/checkpoints/p1/seg_smoke_repro/provenance.json"
# Clear every stale output (report, DONE marker, checkpoint, provenance) so a failed run
# cannot leave a stale success marker or let the postcondition pass on a prior run's blobs.
rm -f "$DEST/seg_smoke_repro.json" "$DEST/seg_smoke_repro.DONE" "$CHECKPOINT" "$PROVENANCE"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"      # deepspeed/nvcc probe CUDA_HOME at import.
export WANDB_MODE=offline             # compute nodes: no live streaming (§9.2); sync later.
export PYTHONHASHSEED=0               # determinism (§8.3; matches conf/train/stage1_smoke.yaml).

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

# Hydra entry (no `--` args → the Hydra path, not the argparse CLI). IDENTICAL committed config;
# ONLY the two output paths are overridden so the re-run writes to the *_repro namespace and
# never touches the committed P1-07 report / DVC checkpoint. run_smoke self-validates the report.
set +e
PYTHONPATH=src python -m tbox_finder.train.stage1_smoke \
  report_path=reports/p1/seg_smoke_repro.json \
  checkpoint_dir=checkpoints/p1/seg_smoke_repro
RC=$?
set -e

# Publish DONE only when the run succeeded AND all required outputs (report + checkpoint +
# provenance) are non-empty; a missing artifact forces a nonzero exit even if python
# returned 0.
if [ "$RC" -eq 0 ] \
  && [ -s "$DEST/seg_smoke_repro.json" ] \
  && [ -s "$CHECKPOINT" ] \
  && [ -s "$PROVENANCE" ]; then
  touch "$DEST/seg_smoke_repro.DONE"
  echo "REPRO_DONE job=${SLURM_JOB_ID:-none}"
else
  echo "REPRO_FAIL rc=$RC job=${SLURM_JOB_ID:-none}"
  [ "$RC" -ne 0 ] || RC=1
fi
exit "$RC"
