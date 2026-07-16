#!/bin/bash
# P1-16 — Stage-2 LoRA smoke fine-tune + VRAM smoke (PRD §10.2 condition (b) VRAM half;
# §10.3; §11 · ADR-0002 D3/D5/D6(b)/A5/A9 · CLAUDE.md §9.3).
#
# Runs ONE real LoRA fine-tune step end-to-end on ONE RTX A4000 (Ampere sm_86) against the
# parity-confirmed multimolecule/rinalmo-giga @2a71f6f9 (ADR-0002 A9), wrapped in the exact
# PRD §10.3 LoraConfig(r=16, α=32, dropout=0.05, all-linear) + gradient checkpointing + bf16,
# and measures PEAK VRAM + steps/sec. The batch is SEEDED RANDOM RNA and the loss is a
# PLACEHOLDER (mean-squared activation) — this is a mechanics/VRAM measurement, NOT a T-box
# science result, NOT a convergence claim (§10.3; the report's validator refuses any artifact
# claiming otherwise).
#
# GATE (imp.md P1-16 + the carried P1-15 handoff): peak VRAM ≤ 16 GB (the ONLY number the
# PRD/ADRs pin here) AND the step trains the adapters (finite grads at every LoRA site) AND
# it ran on real sm_86 AND the P1-15-SELECTED attention backend's forward actually ran.
# On a VRAM miss → CLAUDE.md §7 stop-and-ask (escalate to FSDP FULL_SHARD or the §10.2
# RNA-FM swap consideration) — the budget is NOT weakenable (the A7/A9 precedent).
# On an FA-2 forward failure → the run falls back to SDPA so the VRAM number still lands,
# but gate.attention_forward_verified=false ⇒ overall_pass=false ⇒ §7 stop-and-ask (flipping
# the recorded P1-15 selection touches a committed artifact + conf/train/lora_stage2.yaml).
#
# WHERE:   cluster `gpu` partition, ONE RTX A4000 (sm_86). NOT the laptop (sm_89 — it cannot
#          answer an sm_86 question, and RiNALMo-giga OOMs on the 8 GB 4060 under WSL2), NOT
#          CI (no torch). env tbox-ml-rna (transformers 5.13.0 + multimolecule 0.1.0 + peft;
#          ADR-0002 A8). est-wall ~10-15 min (~2.5 GB weight download to a node-local HF
#          cache + a load + a short batch sweep of train steps); --time 30 min is headroom.
# INPUT:   the rinalmo-giga weights only (node-local HF cache, off the tight HOME quota) +
#          the committed reports/p1/kernel_smoke.json (the ADR-0002 A5 sm_86 evidence the
#          backend decision reads). NO dataset — the batch is synthetic + seeded (§8.3).
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout <branch> && git reset --hard origin/<branch> && \
#              sbatch slurm/p1/lora_smoke.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p1/lora_smoke.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): squeue empty + reports/p1/lora_vram_smoke.DONE
#          present + a self-validated reports/p1/lora_vram_smoke.json (measured=true,
#          peak_vram_gib finite+positive, config.is_science=false — run_vram_smoke RAISES on
#          any validate_smoke_report failure, so RC!=0 leaves NO DONE marker). The .err is
#          EXPECTED non-empty on success (transformers/multimolecule load notes); scan for
#          real tracebacks, do not gate on zero bytes. A missing DONE ⇒ STOP and ask (§9.3;
#          no auto-resubmit).
# RETRIEVE: rsync the report back, then git-commit it (reports/ is git-tracked; the 2.5 GB
#          weights are re-fetchable by pinned revision, and the LoRA adapter is a THROWAWAY
#          mechanics artifact left on the cluster — the P1-08 `*_repro` checkpoint precedent):
#            rsync -avzP two:~/tbox-finder/reports/p1/lora_vram_smoke.json <local>/reports/p1/
#
#SBATCH --job-name=tbox-p1-16-lora-vram-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=reports/p1/lora_vram_smoke_%j.out
#SBATCH --error=reports/p1/lora_vram_smoke_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"
OUT_DIR="$REPO/reports/p1"
mkdir -p "$OUT_DIR"
DONE="$OUT_DIR/lora_vram_smoke.DONE"
REPORT="$OUT_DIR/lora_vram_smoke.json"

# Singleton lock + clear this run's stale success marker/report FIRST — before the
# failure-prone download/conda — so a run that dies during activation/import cannot leave a
# prior run's DONE/JSON behind (artifact-based verification keys on absence-on-failure, §9.3).
exec 9>"$OUT_DIR/.lora_smoke.lock"
if ! flock -n 9; then
  echo "LORA_SMOKE_SKIP: another job holds the lock — refusing to share outputs." >&2
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

# Node-local scratch for the HF weight cache + the throwaway adapter (off the tight HOME
# quota — a 650M-param download plus an adapter has no business in a 737 GB-free shared HOME).
BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.
export HF_HOME="$BUILD/hf"

# Measure. --require-cuda fails loud if the node has no GPU (a CPU peak-VRAM number would be
# meaningless for the §10.2 A4000 question). The adapter is written to node-local scratch and
# dies with the trap: imp.md calls it throwaway, and committing a mechanics-smoke adapter
# trained on random RNA would be shipping a meaningless checkpoint.
set +e
PYTHONPATH=src python -m tbox_finder.train.lora_harness \
  --smoke --require-cuda --out "$REPORT" --adapter-dir "$BUILD/lora_adapter"
RC=$?
set -e

# Publish the DONE marker ONLY on success AND a non-empty measured report.
if [ "$RC" -eq 0 ] && [ -s "$REPORT" ]; then
  touch "$DONE"
  echo "LORA_SMOKE_DONE job=${SLURM_JOB_ID:-none}"
else
  echo "LORA_SMOKE_FAIL rc=$RC job=${SLURM_JOB_ID:-none}" >&2
  [ "$RC" -ne 0 ] || RC=1
fi
exit "$RC"
