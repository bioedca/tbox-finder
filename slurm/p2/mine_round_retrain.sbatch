#!/bin/bash
# P2-10e-msa-producer — LEG (d): merge the produced covariation status → spare rule → DDP retrain.
#
# Runs afterok-dependent on the whole leg-(c) producer array (mine_round.sbatch submits it). It (1)
# merges the per-shard status tables into the round's covariation-status table, (2) applies the spare
# rule to the round's false positives under this round's backend availability (covariation MSA-supply
# now available), producing the mined/spared outcome, and (3) DDP-retrains from round k−1 on the mined
# hard negatives + the §9.1 10:1 seed mix, then runs the Tier-2N gate.
#
# ── SHAPE: one whole-8-GPU-node job (the DDP retrain wants the node); node `two`. ──────────────────
# §9.2 forbids `compute`(=zero)/pinning a node. The mined-window pool assembly (mined coords →
# 1024-nt training windows, the mining/pool.py genomic_window schema) is the A10 Phase-2 remaining
# wire; this leg fails loud rather than retrain on a fabricated/empty pool (§10.3).

#SBATCH --job-name=tbox-p2-10e-retrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=reports/p2/mine_round_retrain_%j.out
#SBATCH --error=reports/p2/mine_round_retrain_%j.err
# No --account/--qos, no --nodelist.

set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

ROUND="${ROUND:-0}"
ROUND_DIR="${ROUND_DIR:?set ROUND_DIR via --export}"
FP_MANIFEST="${FP_MANIFEST:?set FP_MANIFEST via --export}"
CKPT="${CKPT:-data/processed/checkpoints/stage1_production/stage1.pt}"   # round k−1 parent
STATUS_DIR="$ROUND_DIR/status"
MERGED_STATUS="$ROUND_DIR/covariation_status.json"
ROUND_REPORT="$ROUND_DIR/mine_round_report.json"
MINED_POOL="$ROUND_DIR/mined_pool.parquet"
DONE="$ROUND_DIR/mine_round_retrain.DONE"
rm -f "$DONE"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"; export PYTHONHASHSEED=0; export TOKENIZERS_PARALLELISM=false
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} round=$ROUND sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ── (1) verify every producer shard landed, then merge → the round covariation-status table ───────
n_shards="$(find "$ROUND_DIR/shards" -maxdepth 1 -name 'shard_*.json' 2>/dev/null | wc -l)"
n_ok="$(find "$STATUS_DIR" -maxdepth 1 -name 'shard_*.SHARD_OK' 2>/dev/null | wc -l)"
[ "$n_shards" -ge 1 ] && [ "$n_ok" -eq "$n_shards" ] || {
  echo "FATAL: producer array incomplete — $n_ok/$n_shards SHARD_OK markers (§9.3 gap; no auto-resubmit)" >&2
  exit 3; }
PYTHONPATH=src python -m tbox_finder.mining.covariation_producer merge \
  --tables "$STATUS_DIR"/status_*.json --out "$MERGED_STATUS"

# ── (2) apply the spare rule: FP manifest + produced status → mined/spared outcome ────────────────
PYTHONPATH=src python -m tbox_finder.mining.mine_round apply-spare-rule \
  --manifest "$FP_MANIFEST" --status-table "$MERGED_STATUS" --msa-supply-available \
  --out "$ROUND_REPORT"
N_MINED="$(PYTHONPATH=src python -c "import json,sys;print(json.load(open('$ROUND_REPORT'))['n_mined'])")"
echo "spare rule applied → $ROUND_REPORT ; n_mined=$N_MINED"

# ── (3) DDP retrain on the mined hard negatives (skip an honest no-op if nothing was mined) ───────
if [ "$N_MINED" -eq 0 ]; then
  echo "n_mined=0 — no new hard negatives this round; a retrain would be a no-op. Recording the" >&2
  echo "round as complete WITHOUT a retrain (honest §10.3: do not emit a checkpoint that trained on" >&2
  echo "nothing new). At P2 with only the covariation backend available and relaxed-architecture +" >&2
  echo "synteny structurally unavailable, the spare rule spares every candidate — a per-round finding" >&2
  echo "the A10 measurement + Phase-2 review weigh before proceeding, not a silent success." >&2
  touch "$DONE"; exit 0
fi

[ -s "$MINED_POOL" ] || {
  echo "FATAL: n_mined=$N_MINED but the mined-window pool $MINED_POOL is not built — assembling the" >&2
  echo "mined coords into the mining/pool.py genomic_window training pool is the A10 Phase-2 wire." >&2
  echo "Refusing to retrain on a missing/fabricated pool (§10.3)." >&2
  exit 2; }

torchrun --standalone --nproc_per_node=8 -m tbox_finder.train.train_stage1 \
  init_from_checkpoint="$CKPT" \
  negative_pool_parquet="$MINED_POOL" \
  negative_fraction=0.9090909090909091   # §9.1 10:1 (A7 negative_fraction = 10/11)
# Then the Tier-2N gate (scan_probe_variants → evaluate_probe_round: round_decision + degenerate guard)
# grades the retrained checkpoint against the best round so far (a §7 halt/rollback stop, ADR-0005 D14).

touch "$DONE"
echo "round $ROUND retrain complete — DONE marker written; run the LOCAL Tier-2N gate + result commit."
