#!/bin/bash
# P1-14 — RiNALMo forward-throughput probe (ADVISORY; PRD §10.2; ADR-0002; CLAUDE.md §9.3).
# Measures candidates/sec/GPU for the multimolecule/rinalmo-giga backbone forward on ONE
# RTX A4000 (bf16, autocast) over synthetic T-box-length RNA windows (~150-350 nt). This
# is a hardware LATENCY probe — RiNALMo throughput is length-bound, not content-bound — so
# random RNA of realistic lengths is a faithful measurement (NOT a T-box science result,
# §10.3). Advisory only: there is NO binding threshold at P1 (the binding latency decision
# is frozen to the P5 sizing gate, which needs Stage-1 candidate density). A "clearly-
# hopeless" reading is a §7 human stop-and-ask, NEVER an auto-switch to RNA-FM.
#
# WHERE:   cluster `gpu` partition, one RTX A4000 (Ampere sm_86). NOT the laptop, NOT CI.
#          env tbox-ml-rna (transformers 5.13.0 + multimolecule 0.1.0; ADR-0002 A8).
#          est-wall ~5-10 min (weights download ~2.5 GB to node-local HF cache + a short
#          length/batch sweep of forward passes); --time 30 min is headroom.
# INPUT:   the rinalmo-giga weights only (downloaded to a node-local HF cache off the tight
#          HOME quota); NO dataset (windows are synthetic + seeded). Fail-loud on any load
#          / arch drift (load_rinalmo_ss asserts hidden_size/num_hidden_layers).
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout <branch> && git reset --hard origin/<branch> && \
#              sbatch slurm/p1/rinalmo_throughput.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p1/rinalmo_throughput.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): squeue empty + reports/p1/rinalmo_throughput.DONE
#          present + a self-validated reports/p1/rinalmo_throughput.json (measured=true,
#          advisory_only=true, precision=bfloat16, headline candidates/sec finite+positive —
#          the module's run_probe raises on any validate_report failure, so RC!=0 leaves NO
#          DONE marker). The .err is EXPECTED non-empty on success (transformers/multimolecule
#          load notes + fresh-head init warnings); scan for real tracebacks, do not gate on
#          zero bytes. A missing DONE ⇒ STOP and ask (§9.3; no auto-resubmit).
# RETRIEVE: rsync the report back, then it is git-committed (reports/ is git-tracked; the
#          2.5 GB weights are re-fetchable by pinned revision, not committed):
#            rsync -avzP two:~/tbox-finder/reports/p1/rinalmo_throughput.json <local>/reports/p1/
#
#SBATCH --job-name=tbox-p1-14-rinalmo-throughput
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=reports/p1/rinalmo_throughput_%j.out
#SBATCH --error=reports/p1/rinalmo_throughput_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"
OUT_DIR="$REPO/reports/p1"
mkdir -p "$OUT_DIR"
DONE="$OUT_DIR/rinalmo_throughput.DONE"
REPORT="$OUT_DIR/rinalmo_throughput.json"

# Singleton lock + clear this run's stale success marker/report FIRST — before the
# failure-prone download/conda — so a run that dies during activation/import cannot leave a
# prior run's DONE/JSON behind (artifact-based verification keys on absence-on-failure, §9.3).
exec 9>"$OUT_DIR/.throughput.lock"
if ! flock -n 9; then
  echo "THROUGHPUT_SKIP: another job holds the lock — refusing to share outputs." >&2
  exit 1
fi
rm -f "$DONE" "$REPORT"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-rna
export CUDA_HOME="$CONDA_PREFIX"          # multimolecule→danling→deepspeed probe CUDA_HOME at import (ADR-0002 A4).
export WANDB_MODE=offline                 # defensive: no live streaming on compute nodes (§9.2).
export PYTHONHASHSEED=0                    # determinism (§8.3).
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Node-local scratch for the HF weight cache (off the tight HOME quota).
BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.
export HF_HOME="$BUILD/hf"

# Measure (heavy torch/multimolecule; --require-cuda fails loud if the node has no GPU; the
# report self-validates advisory_only/bf16/finite-positive-headline, else run_probe raises).
set +e
PYTHONPATH=src python -m tbox_finder.eval.rinalmo_throughput \
  --require-cuda --out "$REPORT"
RC=$?
set -e

# Publish the DONE marker ONLY on success AND a non-empty measured report.
if [ "$RC" -eq 0 ] && [ -s "$REPORT" ]; then
  touch "$DONE"
  echo "THROUGHPUT_DONE job=${SLURM_JOB_ID:-none}"
else
  echo "THROUGHPUT_FAIL rc=$RC job=${SLURM_JOB_ID:-none}" >&2
  [ "$RC" -ne 0 ] || RC=1
fi
exit "$RC"
