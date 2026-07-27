#!/bin/bash
# P2-10e-msa-producer — LEG (c): the DECOUPLED per-candidate covariation PRODUCER array (ADR-0005 A10).
#
# One array task per false-positive shard. A10 Pin 2 pins this as its OWN SLURM array, between the
# round's scan leg (a) and its retrain leg (d), CPU-only, so the 3-env producer (nhmmer → mlocarna →
# R-scape) does not hold the 8-GPU node and is not capped by the round sbatch's 5 h wall. It SCORES
# EVERY false positive (no cap, no subsample); a candidate that is not producible (< min_sequences
# homologs, no SS_cons, R-scape error) resolves to `unavailable` ⇒ spared (never silently mined).
#
# ── SHAPE: a CPU-only --array on the `gpu` partition, NO --gres (CPU tasks on node `two`'s 96 cores) ─
#   task t reads $SHARD_DIR/shard_<t>.json (from mine_round.sbatch make-shards) and runs the three
#   env stages over its candidates → $ROUND_DIR/status/status_<t>.json + a per-shard SHARD_OK marker.
# §9.2 forbids `compute`(=zero) and pinning a node. The array WIDTH + --time are the A10 Phase-2 pin
# (mine_round.sbatch submits with `--array=0-(W-1)`; W chosen so a shard's wall fits --time).
#
# ── SUBMITTED BY mine_round.sbatch (leg c) with --export ROUND/SHARD_DIR/ROUND_DIR and --array=0-(W-1).
# Standalone (debug) submit: `sbatch --array=0-0 --export=ALL,SHARD_DIR=…,ROUND_DIR=… mine_round_producer.sbatch`
# VERIFY (artifact-based, §9.3): every task's $ROUND_DIR/status/shard_<t>.SHARD_OK present + its .err
#   zero bytes; the retrain leg (d) merges the per-shard status tables and fails loud on a gap.

#SBATCH --job-name=tbox-p2-10e-producer
#SBATCH --partition=gpu
#SBATCH --array=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=reports/p2/mine_round_producer_%A_%a.out
#SBATCH --error=reports/p2/mine_round_producer_%A_%a.err
# No --account/--qos, NO --gres (CPU-only), no --nodelist. #SBATCH --array is overridden at submit.

set -eo pipefail   # NOT -u: conda activate hooks read unset vars.

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_DIR="${SHARD_DIR:?set SHARD_DIR (the make-shards output dir) via --export}"
ROUND_DIR="${ROUND_DIR:?set ROUND_DIR via --export}"
SHARD="$(printf "%s/shard_%03d.json" "$SHARD_DIR" "$TASK")"
STATUS_DIR="$ROUND_DIR/status"
STATUS_OUT="$(printf "%s/status_%03d.json" "$STATUS_DIR" "$TASK")"
OK_MARKER="$(printf "%s/shard_%03d.SHARD_OK" "$STATUS_DIR" "$TASK")"
mkdir -p "$STATUS_DIR"
rm -f "$OK_MARKER"   # clear FIRST so a stale marker cannot green a shard that did not complete.

DB_DIR="data/interim/homolog_db"
BLAST_PREFIX="$DB_DIR/target"
FM_INDEX="$DB_DIR/target.fm"
GENOME_DIR="data/interim/production_genomes"
# CONTROL cutoffs — the certified nhmmer config (job 766); D17 P6-frozen, value at the phase gate.
CTRL_ENGINE="nhmmer"; CTRL_EVALUE="100"; CTRL_MAX_TARGET_SEQS="500"; CTRL_MIN_PIDENT="40"; CTRL_MIN_COV="0.5"

[ -s "$SHARD" ]    || { echo "FATAL: shard $SHARD missing (make-shards / array width mismatch)" >&2; exit 2; }
[ -s "$FM_INDEX" ] || { echo "FATAL: $FM_INDEX missing — stage the homolog DB" >&2; exit 2; }

BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}-${TASK}"
mkdir -p "$BUILD/work"
trap 'rm -rf "$BUILD"' EXIT

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array_task=$TASK shard=$SHARD"

# ── STAGE 1: SEARCH (tbox-homology) ──────────────────────────────────────────────────────────────
conda activate tbox-homology
for t in blastn nhmmer; do command -v "$t" >/dev/null || { echo "FATAL: $t not on PATH" >&2; exit 2; }; done
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer search-shard \
  --shard "$SHARD" --workroot "$BUILD/work" --engine "$CTRL_ENGINE" \
  --evalue "$CTRL_EVALUE" --max-target-seqs "$CTRL_MAX_TARGET_SEQS" \
  --min-pident "$CTRL_MIN_PIDENT" --min-cov "$CTRL_MIN_COV" \
  --db-prefix "$BLAST_PREFIX" --fm-index "$FM_INDEX" --genome-dir "$GENOME_DIR"

# ── STAGE 2: ALIGN (tbox-locarna) ────────────────────────────────────────────────────────────────
conda activate tbox-locarna
command -v mlocarna >/dev/null || { echo "FATAL: mlocarna not on PATH — tbox-locarna not created?" >&2; exit 2; }
mlocarna --version 2>&1 | grep -q "LocARNA 2.0.1" || { echo "FATAL: mlocarna != LocARNA 2.0.1 (ADR-0002 A13)" >&2; exit 2; }
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer align-shard \
  --shard "$SHARD" --workroot "$BUILD/work" --cpu 2

# ── STAGE 3: SCORE (tbox-rscape) → the shard status table ────────────────────────────────────────
conda activate tbox-rscape
command -v R-scape >/dev/null || { echo "FATAL: R-scape not on PATH — tbox-rscape not created?" >&2; exit 2; }
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer score-shard \
  --shard "$SHARD" --workroot "$BUILD/work" --out-table "$STATUS_OUT"

touch "$OK_MARKER"
echo "shard $TASK scored → $STATUS_OUT ; SHARD_OK written"
