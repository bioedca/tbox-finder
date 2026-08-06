#!/bin/bash
# P3-15'-b — the per-candidate Stage-2 posterior producer.
# (PRD §6/§9.1/§11; ADR-0005 D14 the P3 spare-rule disjunct, D11/A11 the named posterior;
#  ADR-0002 A4 the ml-rna env; CLAUDE.md §9.3.)
#
# Turns the round-0 FP manifest's COORDINATES into the `candidate_id -> posterior` table
# `remine.load_stage2_posteriors` consumes and `slurm/p3/stage1_remine.sbatch` reads at
# "$ROUND_DIR/stage2_posteriors.json". Composition only — see the module docstring:
#   resolve_candidate_sequence -> transcribe_to_rna -> score_rows -> calibrated_posterior
#
# ── SIZING — MEASURED end-to-end, not extrapolated (laptop RTX 4060, 2026-08-05) ──
# All 941 candidates, BOTH strands (1,882 rows), one GPU:
#   resolve+transcribe   2.0 s      (0.0021 s/candidate, CPU)
#   checkpoint load     18.9 s      (warm HF cache; add ~2-4 min for a cold download)
#   score_rows          38.7 s      (0.0206 s/row at batch 4)
#   peak VRAM         1.33 GiB      of 15.6 GiB on an A4000 — ~12x headroom
# => the WORK is ~1 minute. Wall is dominated by the 2.5 GB backbone download into
# node-local HF_HOME, which is why --time is 01:00:00 and not 00:10:00.
#
# ── WHY AN ARRAY OF 2 FOR A ONE-MINUTE JOB ──────────────────────────────────
# Not for throughput. `merge`'s coverage refusal and its duplicate-candidate refusal are
# the two guards that stop a truncated run from certifying, and a single-shard array
# would never exercise either on the cluster. Two tasks is the smallest width that
# actually runs the merge path it ships. Widen with `--export=...,N_SHARDS=<n>` plus a
# matching `--array=0-<n-1>`; the two MUST agree (asserted below, exit 2 if not).
#
# ── WHY EACH TASK SHARDS THE MANIFEST ITSELF ────────────────────────────────
# `write_shards` is a deterministic partition of a git-tracked manifest, so every task
# computes the identical split and reads only its own slice. Sharding into node-local
# scratch instead of a shared directory removes the array's only write race outright,
# rather than guarding it with a lock.
#
# ── THE MERGE LEG, AND WHY IT IS IN THIS FILE ───────────────────────────────
# The last task to finish (whichever it is) finds every SHARD_OK present, takes the
# flock, and merges. Deterministic without a second submit and without a dependency job:
# a task that does not see a full marker set simply exits 0 having done its own share.
# ⚠ `merge` REFUSES below --min-coverage (default 1.0) with exit 3 and writes nothing:
# an unscored candidate resolves to `unavailable` => SPARED, so a partial table is a
# round that decided almost nothing while reporting cleanly.
#
# ── NO STALE-MARKER DELETION, DELIBERATELY ──────────────────────────────────
# Markers live under a per-ARRAY-job directory ($SLURM_ARRAY_JOB_ID), so a fresh submit
# cannot inherit a previous one's markers and there is nothing to `rm`. That also keeps
# this file clear of the tests/unit/test_sbatch_rm_targets.py trap, where an
# unresolvable `rm` target (a `${VAR:?}` or a `$(...)`) widens to a bare `*` and matches
# every tracked path.
#
# WHERE:   cluster `gpu` partition, ONE RTX A4000 per array task. env tbox-ml-rna
#          (ADR-0002 A4 — multimolecule needs transformers 5.x). No --nodelist.
# INPUT:   all git-tracked or already staged on `two` (verified 2026-08-05):
#            data/processed/mining/round0_fp_manifest.json   (git-tracked, 941 candidates)
#            reports/gate2_p3_ece.json                       (git-tracked, carries T)
#            reports/p3/sweep/*.json                          (git-tracked, all six arms)
#            data/processed/checkpoints/stage2_rinalmo/       (staged; six arms present)
#            data/interim/production_genomes/*.fna            (staged; 2,500 present)
#          Nothing is fetched by this job except the pinned HF backbone revision.
# SUBMIT:  from the repo root so the relative --output/--error resolve. SYNC, THEN SUBMIT.
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p3-15b-stage2-producer && git reset --hard origin/p3-15b-stage2-producer"'
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch --export=ALL,ROUND_DIR=<dir> slurm/p3/stage2_producer.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p3/stage2_producer.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b) every
#          "$ROUND_DIR/markers_<arrayjobid>/shard_NNN.SHARD_OK" present, (c)
#          "$ROUND_DIR/stage2_posteriors.json" exists and reports coverage 1.0 over 941,
#          (d) `python -c "from tbox_finder.mining.remine import load_stage2_posteriors as L;
#          print(len(L('$ROUND_DIR/stage2_posteriors.json')))"` prints 941.
#          ⚠️ the `.err` is EXPECTED NON-EMPTY on success (HF download chatter, the
#          transformers LOAD REPORT, the flash-attn bf16 cast notice) — scan for real
#          tracebacks, DO NOT gate on zero bytes. A missing marker is a §9.3 stop:
#          surface it, no auto-resubmit.
# RETRIEVE: plain scratch, NOT dvc (the ml env has no dvc):
#            rsync -avzP two:<ROUND_DIR>/stage2_posteriors.json <local-round-dir>/
#            rsync -avzP two:<ROUND_DIR>/provenance.json        <local-round-dir>/

#SBATCH --job-name=tbox-p3-15b-stage2-producer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=reports/p3/stage2_producer_%A_%a.out
#SBATCH --error=reports/p3/stage2_producer_%A_%a.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often
# down and cryosparc keeps both nodes fragmented). One GPU per task, so both tasks can
# co-schedule beside cryosparc rather than waiting for a whole free node.

set -eo pipefail   # NOT -u: the conda activate hooks read unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# Every default path in the producer is repo-relative, so the `cd` above is load-bearing:
# run from anywhere else and GENOME_DIR/DEFAULT_CKPT_ROOT resolve to nothing and every
# candidate raises.

TASK="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-2}"
STRAND_POLICY="${STRAND_POLICY:-max_over_strands}"
MIN_COVERAGE="${MIN_COVERAGE:-1.0}"
FP_MANIFEST="${FP_MANIFEST:-data/processed/mining/round0_fp_manifest.json}"
GATE2_REPORT="${GATE2_REPORT:-reports/gate2_p3_ece.json}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
printf -v TASK3 "%03d" "$TASK"

# ROUND_DIR is SCRATCH and must never be a git-tracked path: the round's tables are run
# artifacts, and an operator pointing this at data/processed/mining/ would leave untracked
# JSON inside the repo that every `git status`-reading provenance gate then sees.
ROUND_DIR="${ROUND_DIR:-$HOME/tbox-scratch/round_p3_15b}"
if [ -z "$ROUND_DIR" ]; then
  echo "FATAL: ROUND_DIR resolved empty; refusing to write anywhere." >&2
  exit 2
fi
case "$ROUND_DIR" in
  "$REPO"/*|./*|data/*|reports/*|src/*)
    echo "FATAL: ROUND_DIR=$ROUND_DIR is inside the repo. It must be scratch (§9.2)." >&2
    exit 2 ;;
esac

if [ "$TASK" -ge "$N_SHARDS" ]; then
  echo "FATAL: array index $TASK exceeds N_SHARDS=$N_SHARDS — the --array width and" >&2
  echo "       N_SHARDS must agree, or a slice of the manifest is never scored and the" >&2
  echo "       merge would refuse on coverage after paying for every other task." >&2
  exit 2
fi

# ── staged-input assertions (fail loud, before any GPU time) ─────────────────
ARM_DIR="data/processed/checkpoints/stage2_rinalmo/aux1.0_lr1e-4"
for required in \
  "$FP_MANIFEST" \
  "$GATE2_REPORT" \
  "reports/p3/sweep/aux1.0_lr1e-4.json" \
  "$ARM_DIR/stage2_heads.pt" \
  "$ARM_DIR/lora_adapter/adapter_model.safetensors" ; do
  if [ ! -s "$required" ]; then
    echo "FATAL: $required is missing or empty on this checkout." >&2
    echo "       Stage it from the laptop (see INPUT in this header) before resubmitting." >&2
    exit 2
  fi
done
# The arm name above is asserted, not chosen: the producer resolves the production arm
# through conf/ + each arm's own run report and raises if they disagree. This check only
# fails EARLY on a checkout where the staged bytes are absent.
if [ "$(ls data/interim/production_genomes/*.fna 2>/dev/null | wc -l)" -lt 1 ]; then
  echo "FATAL: no data/interim/production_genomes/*.fna — the candidate spans cannot" >&2
  echo "       be resolved to nucleotides. Stage the DVC artifact first." >&2
  exit 2
fi

# ── node-local scratch ──────────────────────────────────────────────────────
# JOB_SCRATCH, never BUILD: `conda activate` exports BUILD=x86_64-conda-linux-gnu and
# silently clobbers it (that killed job 789 six seconds in).
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

source /exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
conda activate tbox-ml-rna
export CUDA_HOME="$CONDA_PREFIX"        # multimolecule will not import without it
export HF_HOME="$JOB_SCRATCH/hf"        # 2.5 GB of weights stay OFF the 88%-full HOME
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=2
export WANDB_MODE=offline

MARKER_DIR="$ROUND_DIR/markers_$ARRAY_ID"
POST_DIR="$ROUND_DIR/posteriors_$ARRAY_ID"
mkdir -p "$MARKER_DIR" "$POST_DIR"
SHARD_DIR="$JOB_SCRATCH/shards"
POST_OUT="$POST_DIR/posteriors_$TASK3.json"

echo "== task $TASK/$N_SHARDS  strand_policy=$STRAND_POLICY  round_dir=$ROUND_DIR"

# ── leg 1: the deterministic partition, into node-local scratch ─────────────
python -m tbox_finder.mining.stage2_producer make-shards \
  --manifest "$FP_MANIFEST" \
  --n-shards "$N_SHARDS" \
  --out-dir "$SHARD_DIR"

# ── leg 2: resolve -> transcribe -> score -> calibrate this shard ───────────
python -m tbox_finder.mining.stage2_producer score-shard \
  --shard "$SHARD_DIR/shard_$TASK3.json" \
  --out "$POST_OUT" \
  --gate2-report "$GATE2_REPORT" \
  --strand-policy "$STRAND_POLICY" \
  --device cuda

# The marker is written LAST, after the table exists, and only on the success path:
# `set -e` means any failure above exits before this line.
touch "$MARKER_DIR/shard_$TASK3.SHARD_OK"
echo "== shard $TASK3 done -> $POST_OUT"

# ── leg 3: the merge, run by whichever task finishes last ───────────────────
# The flock serialises the check-then-merge so two tasks finishing together cannot both
# pass the completeness test and race on the same output.
exec 9>"$MARKER_DIR/.merge.lock"
if ! flock -n 9; then
  echo "== another task holds the merge lock; this task's share is complete."
  exit 0
fi
N_DONE="$(ls "$MARKER_DIR"/shard_*.SHARD_OK 2>/dev/null | wc -l)"
if [ "$N_DONE" -lt "$N_SHARDS" ]; then
  echo "== $N_DONE/$N_SHARDS shards done; a later task will merge."
  exit 0
fi

echo "== merging $N_DONE shard tables"
python -m tbox_finder.mining.stage2_producer merge \
  --tables "$POST_DIR"/posteriors_*.json \
  --manifest "$FP_MANIFEST" \
  --min-coverage "$MIN_COVERAGE" \
  --out "$ROUND_DIR/stage2_posteriors.json"

# Read the artifact back through the CONSUMER, not through the writer: the table is only
# useful if `remine` accepts it, and that is a different validator from the one that wrote
# it. A shape that merges cleanly and then refuses at the round would otherwise surface
# only after the GPU legs of the re-mine.
python - "$ROUND_DIR/stage2_posteriors.json" <<'PY'
import sys

from tbox_finder.mining.remine import load_stage2_posteriors

posteriors = load_stage2_posteriors(sys.argv[1])
if not posteriors:
    raise SystemExit(f"FATAL: {sys.argv[1]} round-trips to an EMPTY map")
lo, hi = min(posteriors.values()), max(posteriors.values())
print(f"consumer accepted {len(posteriors)} posteriors, range [{lo:.6g}, {hi:.6g}]")
PY

touch "$MARKER_DIR/MERGE_OK"
echo "== round table written: $ROUND_DIR/stage2_posteriors.json"
