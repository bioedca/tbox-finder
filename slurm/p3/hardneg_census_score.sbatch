#!/bin/bash
# P3-16″-hardneg — ADR-0005 A17 Pin 4's census extension, job 3 of 3: payloads -> census power
# -> Stage-2 -> candidate tables. (ADR-0005 A17 Pins 3-4, D3 the locus knobs; CLAUDE.md §9.3.)
#
# Every leg that reads a ~27 GB Stage-1 archive runs here, because the laptop has 13 GB of RAM.
# Read hardneg_census_pool.sbatch first: it explains $WORK, the chain, and the input manifest.
#
# A17 Pin 4's ORDER, kept by this file's own sequence: the payloads read Stage-1 only; P is
# recounted on the census from the twin's τ 0.5 work list alone (`precision power --stage census`,
# bound to the draw record that opened it); `power-gate` must exit 0 (grade or underpowered — on
# the census both compute the A13 verdict); only then does Stage-2 score anything. There is no
# second extension, so no exit-5 branch exists here: any non-zero exit stops the job.
#
# The rule knobs are the v2 / draw ones (workflow/rules/integration.smk `two_stage_payloads_
# hardneg` and `two_stage_table_hardneg` defaults): global scope, min-span 50, gap-merge 10,
# >= 2 distinct elements, flank 50, order margin 1; τ ∈ {0.5 graded, 0.7, 0.9 reported}; the
# GATE-2 temperature from reports/gate2_p3_ece.json; Stage-2 operating point 0.5; arm
# aux1.0_lr1e-4, batch 4.
#
# SIZING (draw, laptop): payloads 3 min 55 s per 99,826 contigs => ~33 min each at ~820k, three
# τ at once per arm (each holds its arm's ~27 GB archive + the benchmark => ~35 GB; three ≈ 105 GB).
# Stage-2: 0.0206 s/row at batch 4 (P3-15'-b, measured) over ~5-7k union rows per arm + the
# 2.5 GB backbone download into node-local HF_HOME. The tables (`two_stage run`) are UNMEASURED
# at this scale; budgeted at 2x payloads. Est. ~4 h; --time=10:00:00.
#
# VERIFY: "$MARKERS/score.DONE", every leg marker, power-gate.json present, no traceback in .err
# (the .err is EXPECTED non-empty: HF download chatter and the flash-attn bf16 notice).
# RETRIEVE (to the laptop's main checkout, then `dvc add` the interim files there):
#   reports/p3/hardneg_pool_census.json  reports/p3/hardneg_pool_census_ids.tsv
#   data/interim/p3_16/hardneg_pool_census/        (the census pool's interim, for its validator)
#   data/interim/p3_16/hardneg_census_{benchmark.json,power.json,power_gate.json}
#   data/interim/p3_16/hardneg_census_stage1_{twin,production}.npz
#   data/interim/p3_16/hardneg_census_payloads_{twin,production}_{t0.5,t0.7,t0.9,union}.json
#   data/interim/p3_16/hardneg_census_stage2_{twin,production}.json
#   data/interim/p3_16/hardneg_census_{table,strand_robustness}_{twin,production}_t{0.5,0.7,0.9}.json

#SBATCH --job-name=tbox-p3-16pp-census-score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=reports/p3/hardneg_census_score_%j.out
#SBATCH --error=reports/p3/hardneg_census_score_%j.err
# No --account/--qos, no --nodelist.

set -eo pipefail   # NOT -u: the conda activate hooks read unset variables.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

P=data/interim/p3_16
MARKERS="$P/hardneg_census_markers"
BENCHMARK="$P/hardneg_census_benchmark.json"
DRAW_POWER="$P/hardneg_power.json"
ARMS=(twin production)
TAUS=(0.5 0.7 0.9)
RULE=(--threshold-scope global --min-span 50 --gap-merge 10 --min-distinct-elements 2
  --flank 50 --min-order-margin 1)
mkdir -p logs

[ -e "$MARKERS/stage1.DONE" ] || { echo "FATAL: job 2 did not finish (no stage1.DONE)." >&2; exit 2; }
[ -s "$DRAW_POWER" ] || { echo "FATAL: the draw's power record $DRAW_POWER is missing." >&2; exit 2; }

# JOB_SCRATCH, never BUILD: `conda activate` exports BUILD and clobbers it (job 789).
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

source /exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

run_bg() {  # $1 = marker name; the rest = the command (run with PYTHONPATH=src, logged)
  local name="$1"
  shift
  if [ -e "$MARKERS/$name.OK" ]; then
    echo "== $name: marker present from an earlier attempt, skipping"
    return 0
  fi
  ( PYTHONPATH=src "$@" > "logs/hardneg_census_$name.log" 2>&1 && touch "$MARKERS/$name.OK" ) &
  PIDS+=("$!")
  NAMES+=("$name")
}
wait_all() {
  local k failed=0
  for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then
      echo "FAILED: ${NAMES[$k]} — see logs/hardneg_census_${NAMES[$k]}.log" >&2
      failed=1
    fi
  done
  PIDS=()
  NAMES=()
  [ "$failed" -eq 0 ] || exit 1
}
PIDS=()
NAMES=()

conda activate tbox-data

# ── payloads: Stage-1 only; one arm at a time, its three τ side by side ─────────────────────────
for arm in "${ARMS[@]}"; do
  for tau in "${TAUS[@]}"; do
    run_bg "payloads_${arm}_t${tau}" python -m tbox_finder.integration.two_stage payloads \
      --contigs "$BENCHMARK" --stage1 "$P/hardneg_census_stage1_${arm}.npz" \
      --out "$P/hardneg_census_payloads_${arm}_t${tau}.json" --threshold "$tau" "${RULE[@]}"
  done
  wait_all
done
for arm in "${ARMS[@]}"; do
  run_bg "union_${arm}" python -m tbox_finder.integration.two_stage payloads-union \
    --manifests "$P/hardneg_census_payloads_${arm}_t0.5.json" \
    "$P/hardneg_census_payloads_${arm}_t0.7.json" "$P/hardneg_census_payloads_${arm}_t0.9.json" \
    --out "$P/hardneg_census_payloads_${arm}_union.json"
done
wait_all

# ── A17 Pin 4: P recounted on the census, from the twin's τ 0.5 work list alone ─────────────────
run_bg power python -m tbox_finder.integration.precision power \
  --benchmark "$BENCHMARK" --twin-payloads "$P/hardneg_census_payloads_twin_t0.5.json" \
  --stage1-threshold 0.5 --stage census --draw-power "$DRAW_POWER" \
  --out "$P/hardneg_census_power.json"
wait_all
run_bg power_gate python -m tbox_finder.integration.precision power-gate \
  --power "$P/hardneg_census_power.json" --out "$P/hardneg_census_power_gate.json"
wait_all
cat "logs/hardneg_census_power.log"

# ── Stage-2: the union work list per arm, after the census decision and never before ────────────
conda deactivate
conda activate tbox-ml-rna
export CUDA_HOME="$CONDA_PREFIX"     # multimolecule will not import without it
export HF_HOME="$JOB_SCRATCH/hf"     # 2.5 GB of backbone weights stay off HOME
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
for arm in "${ARMS[@]}"; do
  run_bg "stage2_${arm}" python -m tbox_finder.integration.two_stage stage2 \
    --payloads "$P/hardneg_census_payloads_${arm}_union.json" \
    --checkpoint-root data/processed/checkpoints/stage2_rinalmo --sweep-dir reports/p3/sweep \
    --arm aux1.0_lr1e-4 --batch-size 4 --device cuda \
    --out "$P/hardneg_census_stage2_${arm}.json"
  wait_all   # one card: the arms run one after the other
done

# ── candidate tables: one arm at a time, its three τ side by side ───────────────────────────────
conda deactivate
conda activate tbox-data
for arm in "${ARMS[@]}"; do
  for tau in "${TAUS[@]}"; do
    run_bg "table_${arm}_t${tau}" python -m tbox_finder.integration.two_stage run \
      --contigs "$BENCHMARK" --stage1 "$P/hardneg_census_stage1_${arm}.npz" \
      --stage2 "$P/hardneg_census_stage2_${arm}.json" \
      --report "$P/hardneg_census_strand_robustness_${arm}_t${tau}.json" \
      --table "$P/hardneg_census_table_${arm}_t${tau}.json" --threshold "$tau" "${RULE[@]}" \
      --temperature-from reports/gate2_p3_ece.json --stage2-operating-point 0.5
  done
  wait_all
done

touch "$MARKERS/score.DONE"
echo "census scoring complete — retrieve (header list), dvc add, then fold + report LOCALLY."
