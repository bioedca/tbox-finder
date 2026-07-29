#!/bin/bash
# P2-12 — forward-only scan-throughput probe across the three pinned Caduceus-PS checkpoints
# (PRD §6 :134, §10.1 :210; ADR-0002 A14; ADR-0003 A3; ADR-0005 D3+A3; CLAUDE.md §9.3).
#
# The SECOND half of PRD §10.1's size axis. That axis is worded as a **throughput-driven
# downward search** — 7.73M is the family maximum, long context is unnecessary for ~150-350 nt
# loci, and a bigger backbone costs scan rate genome-wide — so a validation-F1 column alone does
# not deliver it. This job measures windows/second/GPU at the frozen ADR-0005 D3+A3 1024/512
# geometry for all three allow-listed checkpoints under the real Stage-1 stack (backbone ->
# RC-combine -> seg head), forward-only.
#
# ── WHY ONE GPU AND A SEPARATE JOB ───────────────────────────────────────────────────────
# Scanning is inference: the quantity that matters genome-wide is one GPU's forward rate, not a
# DDP x8 training rate. Folding it into ablate_stage1.sbatch as a fifth array task would reserve
# a whole 8-GPU node for a ~10-minute single-GPU measurement AND make the sharing leg's own
# timings noisier. It is therefore its own 1-GPU job — the slurm/p1/rinalmo_throughput.sbatch and
# slurm/p2/mine_round_probe.sbatch precedent — which also schedules far more easily against the
# cryosparc fragmentation that starves whole-node requests.
#
# ── SCOPE, DISCLOSED IN THE REPORT ITSELF (§10.3) ────────────────────────────────────────
# Inputs are seeded random ACGT windows through each checkpoint's own tokenizer, identical
# across checkpoints. Caduceus is a Mamba stack of fixed-shape dense ops (no attention sparsity,
# no KV cache, no early exit), so a (B, L) forward's cost is content-independent — that is an
# ARGUMENT, not a measurement, so the report says so and does NOT claim to be a genome-scan
# rate. End-to-end scan rates (I/O, tiling, candidate calling) are what infer/rho_pilot.py and
# mining/mine_round.py measure. Both smaller checkpoints are seqlen-1k pretrained, so any rate
# delta is a joint size x pretraining-context delta (ADR-0002 A14).
#
# WHERE:   cluster `gpu` partition, ONE RTX A4000 (sm_86), env tbox-ml-dna (ADR-0002 A4). fp32.
# COST:    ADR-0003 A3 charges <= 1.0 GPU-h for this probe across all three checkpoints, inside
#          the <= 18 GPU-h P2-12 aggregate bound. 3 checkpoints x (5 warmup + 30 timed) batches
#          of 8 x 1024 nt is minutes; --time=00:45:00 is generous headroom for the cold HF
#          downloads of the two ablation checkpoints (~8 MB and ~2 MB) plus torch/CUDA init.
# INPUT:   NONE. The probe synthesises its own seeded windows, so no corpus parquet is needed
#          and no staging step precedes it — deliberately, so throughput is measured without a
#          data-loader in the path.
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-12-ablations && git reset --hard origin/p2-12-ablations && \
#              sbatch slurm/p2/backbone_throughput.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/backbone_throughput.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b)
#          reports/p2/status/backbone_throughput.OK present, (c) the report validates (the module
#          re-derives every rate from its own recorded per-batch timings and exits 2 otherwise).
#          ⚠️ the `.err` may be NON-EMPTY on success (transformers trust_remote_code notes) —
#          SCAN for `Traceback|Error|FAILED|assert|OOM|Killed`, do NOT gate on zero bytes.
# RETRIEVE: rsync -avzP two:~/tbox-finder/reports/p2/backbone_throughput.json <local>/reports/p2/

#SBATCH --job-name=tbox-p2-12-throughput
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=reports/p2/status/backbone_throughput_%j.out
#SBATCH --error=reports/p2/status/backbone_throughput_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

STATUS_DIR="reports/p2/status"
mkdir -p "$STATUS_DIR" reports/p2

OK="$STATUS_DIR/backbone_throughput.OK"
# Clear only the stale success marker, before the failure-prone conda activate / HF download, so
# a crash during import cannot leave a previous run's OK behind (§9.3 keys on absence-on-failure).
# $STATUS_DIR is gitignored and this literal path matches no tracked file.
rm -f "$OK"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna

export CUDA_HOME="$CONDA_PREFIX"          # deepspeed/nvcc probe CUDA_HOME at import (ADR-0002 A4).
export PYTHONHASHSEED=0                    # determinism (§8.3).
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/tbox-finder-hf-cache"
mkdir -p "$HF_HOME"

# Node-local scratch. NAMED JOB_SCRATCH, never BUILD (the conda-toolchain clobber, job 789).
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT   # touches only node-local scratch, never the checkout.

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# ⚠️ `import torch` BEFORE selective_scan_cuda (the job-567 loader-path trap); the kernel is a
# capability (ADR-0002 A2 C2) whose absence would make this a ~871x slower fallback reported as
# a throughput. Assert it, don't hope.
python -c 'import torch; import selective_scan_cuda; print(f"selective_scan_cuda OK (torch {torch.__version__})")'

# The report is written to node-local scratch first and copied into the checkout afterwards, for
# the same reason ablate_stage1.sbatch does it: reports/p2/backbone_throughput.json becomes
# git-tracked at the RESULT commit, and a job that writes a tracked path mid-run dirties the tree.
SCRATCH_REPORT="$JOB_SCRATCH/backbone_throughput.json"

echo "── probing all three pinned checkpoints ──"
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.eval.backbone_throughput \
  --out "$SCRATCH_REPORT"

cp "$SCRATCH_REPORT" reports/p2/backbone_throughput.json
touch "$OK"
echo "THROUGHPUT_OK report=reports/p2/backbone_throughput.json"
