#!/bin/bash
# P3-16″-hardneg — ADR-0005 A17 Pin 4's census extension, job 2 of 3: Stage-1 on every contig.
# (ADR-0005 A17 Pins 3-4, A3 the reconciliation operator; CLAUDE.md §9.3.)
#
# Both arms (twin = graded, production = reported; A17 Pin 3) score the WHOLE census benchmark —
# the draw's windows, the census's, and v2's 1,201 positives — on one device class, so no arm
# mixes laptop and A4000 logits. Read hardneg_census_pool.sbatch first: it explains $WORK, the
# chain, and the input manifest.
#
# SHAPE: 8 GPUs; on GPU i, the twin's shard i/8 and the production arm's shard i/8 run side by
# side (the leg forwards one-window contigs one at a time, so one process does not fill a card).
# `two_stage merge-stage1` then rebuilds one archive per arm and REFUSES unless shard i holds
# exactly `shard_contigs(contigs, i, 8)` — a shard cut from another contig file or at another
# count cannot merge.
#
# SIZING — partly UNKNOWN, stated as such: the leg ran at 10.7 contigs/s on the laptop RTX 4060
# over the draw's 99,826 contigs (156 min; it forwards every window twice, the second time as its
# own scan_encoded_windows cross-check). The A4000 rate for THIS leg is unmeasured. At the
# laptop's rate, ~820k contigs x 2 arms / 16 processes ≈ 2.7 h if two processes share a card
# without slowing, ≈ 5.3 h if they serialise. --time=12:00:00 covers the slow case with margin;
# a timeout loses only unfinished shards (each shard's marker is written on its own success).
# Shard-vs-whole equivalence, measured on tests/fixtures/two_stage (20 contigs, laptop 4060):
# max |Δ logit| 8.1e-6 sharded vs whole, against 9.5e-6 between two whole runs — sharding adds
# nothing beyond the GPU's own run-to-run nondeterminism.
#
# MEMORY: each process parses the ~2 GB benchmark before keeping its slice (16 at once at the
# start); each merge holds one arm's ~27 GB of float32 logits. --mem=200G.
#
# SUBMIT / PREFLIGHT / VERIFY: as hardneg_census_pool.sbatch. VERIFY here: 16
# "$MARKERS/stage1_<arm>_<i>of8.OK", both "$MARKERS/merge_<arm>.OK", every per-shard log ends in
# "prediction mismatches = 0", and "$MARKERS/stage1.DONE".

#SBATCH --job-name=tbox-p3-16pp-census-stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=reports/p3/hardneg_census_stage1_%j.out
#SBATCH --error=reports/p3/hardneg_census_stage1_%j.err
# No --account/--qos, no --nodelist.

set -eo pipefail   # NOT -u: the conda activate hooks read unset variables.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

P=data/interim/p3_16
MARKERS="$P/hardneg_census_markers"
BENCHMARK="$P/hardneg_census_benchmark.json"
SHARD_DIR="$P/hardneg_census_stage1_shards"
N_SHARDS=8
ARMS=(twin production)
declare -A CKPT=(
  [twin]=data/processed/checkpoints/stage1_gate4_twin/stage1.pt
  [production]=data/processed/checkpoints/stage1_production/stage1.pt
)
mkdir -p "$SHARD_DIR" logs

[ -e "$MARKERS/pool.DONE" ] || { echo "FATAL: job 1 did not finish (no pool.DONE)." >&2; exit 2; }
[ -s "$BENCHMARK" ] || { echo "FATAL: $BENCHMARK missing or empty." >&2; exit 2; }
for arm in "${ARMS[@]}"; do
  [ -s "${CKPT[$arm]}" ] || { echo "FATAL: ${CKPT[$arm]} missing." >&2; exit 2; }
done

source /exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"   # multimolecule/mamba imports need it
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=2
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Stage-1 shards: GPU i runs twin i/8 and production i/8 concurrently ─────────────────────────
pids=()
names=()
for i in $(seq 0 $((N_SHARDS - 1))); do
  for arm in "${ARMS[@]}"; do
    name="stage1_${arm}_${i}of${N_SHARDS}"
    if [ -e "$MARKERS/$name.OK" ]; then
      echo "== $name: marker present from an earlier attempt, skipping"
      continue
    fi
    out="$SHARD_DIR/hardneg_census_stage1_${arm}.shard${i}of${N_SHARDS}.npz"
    (
      PYTHONPATH=src python -m tbox_finder.integration.two_stage stage1 \
        --contigs "$BENCHMARK" --checkpoint "${CKPT[$arm]}" --out "$out" \
        --dtype float32 --shard "$i/$N_SHARDS" --device "cuda:$i" \
        > "logs/hardneg_census_$name.log" 2>&1 \
        && touch "$MARKERS/$name.OK"
    ) &
    pids+=("$!")
    names+=("$name")
  done
done
failed=0
for k in "${!pids[@]}"; do
  if ! wait "${pids[$k]}"; then
    echo "FAILED: ${names[$k]} — see logs/hardneg_census_${names[$k]}.log" >&2
    failed=1
  fi
done
[ "$failed" -eq 0 ] || exit 1
for arm in "${ARMS[@]}"; do
  for i in $(seq 0 $((N_SHARDS - 1))); do
    [ -e "$MARKERS/stage1_${arm}_${i}of${N_SHARDS}.OK" ] || {
      echo "FATAL: shard $arm $i/$N_SHARDS has no marker." >&2; exit 1; }
  done
done

# ── One archive per arm; merge-stage1 refuses any shard that is not its assigned slice ──────────
pids=()
names=()
for arm in "${ARMS[@]}"; do
  name="merge_${arm}"
  if [ -e "$MARKERS/$name.OK" ]; then
    echo "== $name: marker present from an earlier attempt, skipping"
    continue
  fi
  shards=()
  for i in $(seq 0 $((N_SHARDS - 1))); do
    shards+=("$SHARD_DIR/hardneg_census_stage1_${arm}.shard${i}of${N_SHARDS}.npz")
  done
  (
    PYTHONPATH=src python -m tbox_finder.integration.two_stage merge-stage1 \
      --contigs "$BENCHMARK" --shards "${shards[@]}" \
      --out "$P/hardneg_census_stage1_${arm}.npz" \
      > "logs/hardneg_census_$name.log" 2>&1 \
      && touch "$MARKERS/$name.OK"
  ) &
  pids+=("$!")
  names+=("$name")
done
failed=0
for k in "${!pids[@]}"; do
  if ! wait "${pids[$k]}"; then
    echo "FAILED: ${names[$k]} — see logs/hardneg_census_${names[$k]}.log" >&2
    failed=1
  fi
done
[ "$failed" -eq 0 ] || exit 1

touch "$MARKERS/stage1.DONE"
echo "Stage-1 complete for both arms — job 3 (payloads, power, Stage-2, tables) may start."
