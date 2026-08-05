#!/bin/bash
# P3-15 — the P3 hard-negative RE-MINING round: the high-Stage-2-posterior spare-rule disjunct.
#
# Structure mirrors slurm/p2/mine_round_retrain.sbatch (the leg this supersedes at P3), with one
# addition that comes FIRST and can end the job before a single GPU-second is spent:
#
#   LEG (0)  remine plan  — the structural preflight. Sparing is a disjunction; mining is its
#            negation, so it is a CONJUNCTION over every disjunct. One unavailable backend caps the
#            round's yield at 0 for EVERY candidate, whatever the others score — decidable from the
#            backend set alone, before any scan. ADR-0005 A10's Phase-2 §7 deferral measured exactly
#            that state at P2 (`mining_round_readiness` ready=True, 941/941 spared, 0 mined) and the
#            round still reported success. This leg turns that into an explicit non-zero exit.
#   LEG (1)  apply the spare rule to the round's FPs under the four-disjunct evidence.
#   LEG (2)  DDP retrain from the P2 production checkpoint on the mined hard negatives.
#   LEG (3)  the Tier-2N gate (LOCAL, after this job) — a per-round recall drop halts/rolls back.
#
# ── SHAPE: one whole-8-GPU-node job on node `two` (the DDP retrain wants the node). ────────────────
# §9.2 forbids `compute`(=zero) and forbids pinning a node. No --account / --qos.
#
# ⚠ NOT SUBMITTABLE AS SHIPPED. `remine.STAGE2_SUPPLY_AVAILABLE` is False (no per-candidate Stage-2
# posterior producer yet) and neither the relaxed-architecture nor the downstream-aaRS-synteny
# backend exists, so leg (0) refuses with exit 1. That refusal is the point: it is the honest,
# computed form of the A10 §7 finding, not a bug to work around by passing the flags anyway.
#
# Retrieval: the round report + checkpoint come back via `dvc pull` (the checkpoint is DVC-tracked);
# the plan/report JSONs via rsync from $ROUND_DIR.

#SBATCH --job-name=tbox-p3-15-remine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=reports/p3/stage1_remine_%j.out
#SBATCH --error=reports/p3/stage1_remine_%j.err
# No --account/--qos, no --nodelist.

set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

ROUND_DIR="${ROUND_DIR:?set ROUND_DIR via --export}"
FP_MANIFEST="${FP_MANIFEST:-data/processed/mining/round0_fp_manifest.json}"
# The P2 production scanner this round re-mines FROM; P3-15's output supersedes it (§10.1).
CKPT="${CKPT:-data/processed/checkpoints/stage1_production/stage1.pt}"
# No default: ADR-0005 D14 pins the RULE; the §13.1 phase gate pins the Stage-2 operating point.
# An unset value must stop the job, never silently become one.
STAGE2_THRESHOLD="${STAGE2_THRESHOLD:?set STAGE2_THRESHOLD via --export (no default; §13.1 owns it)}"

PLAN_REPORT="$ROUND_DIR/remine_plan.json"
STATUS_TABLE="$ROUND_DIR/covariation_status.json"
POSTERIORS="$ROUND_DIR/stage2_posteriors.json"
ROUND_REPORT="$ROUND_DIR/remine_round_report.json"
MINED_POOL="$ROUND_DIR/mined_pool.parquet"
DONE="$ROUND_DIR/stage1_remine.DONE"
mkdir -p "$ROUND_DIR"
# Only the round's own markers are removed, and $ROUND_DIR is scratch — never a git-tracked path
# (an rm of a tracked output dirties the tree and fails this job's own provenance gate).
rm -f "$DONE"

[ -s "$CKPT" ] || { echo "FATAL: parent checkpoint $CKPT missing — stage it (rsync/dvc pull)." >&2; exit 2; }

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"; export PYTHONHASHSEED=0; export TOKENIZERS_PARALLELISM=false
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ── LEG (0) the structural preflight — refuse a round that provably cannot mine ────────────────────
# The *_AVAILABLE flags are passed only when the corresponding backend genuinely exists. Passing one
# that does not is how a round goes green while mining nothing; the per-candidate evidence would then
# contradict the declared availability and hard_negative.mine_round raises on that contradiction.
PLAN_FLAGS=""
[ "${RSCAPE_INSTALLED:-0}" = "1" ]        && PLAN_FLAGS="$PLAN_FLAGS --rscape-installed"
[ "${MSA_SUPPLY_AVAILABLE:-0}" = "1" ]    && PLAN_FLAGS="$PLAN_FLAGS --msa-supply-available"
[ "${STAGE2_SUPPLY_AVAILABLE:-0}" = "1" ] && PLAN_FLAGS="$PLAN_FLAGS --stage2-supply-available"
[ "${RELAXED_ARCH_AVAILABLE:-0}" = "1" ]  && PLAN_FLAGS="$PLAN_FLAGS --relaxed-arch-available"
[ "${SYNTENY_AVAILABLE:-0}" = "1" ]       && PLAN_FLAGS="$PLAN_FLAGS --synteny-available"

set +e
PYTHONPATH=src python -m tbox_finder.mining.remine plan \
  --stage2-threshold "$STAGE2_THRESHOLD" $PLAN_FLAGS --out "$PLAN_REPORT"
PLAN_RC=$?
set -e
if [ "$PLAN_RC" -ne 0 ]; then
  echo "P3 re-mining round REFUSED at the preflight (rc=$PLAN_RC) — see $PLAN_REPORT." >&2
  echo "This is not a failure to work around: the spare rule mines only a candidate for which EVERY" >&2
  echo "disjunct ran and failed, so a missing backend makes the round a verified no-op. Building the" >&2
  echo "missing backend(s) is a CLAUDE.md §7 decision, not a flag to pass." >&2
  exit "$PLAN_RC"
fi

# ── LEG (1) apply the four-disjunct spare rule to the round's false positives ──────────────────────
[ -s "$STATUS_TABLE" ] || { echo "FATAL: covariation status table $STATUS_TABLE missing." >&2; exit 3; }
[ -s "$POSTERIORS" ]   || { echo "FATAL: Stage-2 posterior table $POSTERIORS missing." >&2; exit 3; }
PYTHONPATH=src python -m tbox_finder.mining.remine apply-spare-rule \
  --manifest "$FP_MANIFEST" --status-table "$STATUS_TABLE" --posteriors "$POSTERIORS" \
  --stage2-threshold "$STAGE2_THRESHOLD" $PLAN_FLAGS --out "$ROUND_REPORT"
N_MINED="$(PYTHONPATH=src python -c "import json;print(json.load(open('$ROUND_REPORT'))['round']['n_mined'])")"
echo "spare rule applied → $ROUND_REPORT ; n_mined=$N_MINED"

# ── LEG (2) DDP retrain on the mined hard negatives (an honest skip on a 0-yield round) ────────────
if [ "$N_MINED" -eq 0 ]; then
  echo "n_mined=0 — no new hard negatives; a retrain would emit a checkpoint trained on nothing new." >&2
  echo "Recording the round as complete WITHOUT a retrain (§10.3), for the Tier-2N gate to weigh." >&2
  touch "$DONE"; exit 0
fi

[ -s "$MINED_POOL" ] || {
  echo "FATAL: n_mined=$N_MINED but the mined-window pool $MINED_POOL is not built." >&2
  echo "Refusing to retrain on a missing/fabricated pool (§10.3)." >&2
  exit 2; }

torchrun --standalone --nproc_per_node=8 -m tbox_finder.train.train_stage1 \
  init_from_checkpoint="$CKPT" \
  negative_pool_parquet="$MINED_POOL" \
  negative_fraction=0.9090909090909091   # §9.1 10:1 (ADR-0005 A7 negative_fraction = 10/11)

# ── LEG (3) LOCAL, after this job: the Tier-2N gate + GATE-4 re-grade ──────────────────────────────
# scan_probe_variants → evaluate_probe_round (round_decision + degenerate guard) grades the re-mined
# checkpoint against the best round so far; a recall drop is a §7 halt/rollback stop (ADR-0005 D14).
# GATE-4 is RE-CHECKED on the re-mined scanner (§8.5 after a training-config change) — the P2 grade
# is not inherited. Only then does this checkpoint become checkpoints/stage1_production.ckpt.

touch "$DONE"
echo "P3 re-mining round complete — DONE marker written; run the LOCAL Tier-2N gate + GATE-4 re-grade."
