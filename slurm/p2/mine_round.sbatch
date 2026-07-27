#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# P2-10e — one §9.1 hard-negative-mining round (scan → FP-collect → spare → retrain).
#
# SHAPE.  One round off the current Stage-1 checkpoint, in four legs:
#   (a) SCAN the a2 host-order-admissible negative substrate (660 genomes /
#       3,630,683 windows, 1024/512) with the current checkpoint →
#       tbox_finder.mining.mine_round.scan_substrate_windows (reuses infer.scan +
#       infer.call under the PROVISIONAL cross-round-constant criterion, ADR-0005 A9);
#   (b) COLLECT FPs — window-relative called loci → genome-coordinate MiningCandidates
#       (window_candidates_to_mining); every call on the negative substrate is an FP;
#   (c) SPARE RULE — hard_negative.mine_round(candidates, union-prior mask, availability):
#       union-prior masking + the three-valued spare rule; a candidate is mined only if
#       every disjunct ran and failed, else spared (fail-closed);
#   (d) RETRAIN from round k−1: torchrun --standalone --nproc_per_node=8 -m
#       tbox_finder.train.train_stage1 init_from_checkpoint=<round k−1 stage1.pt>
#       negative_pool_parquet=<mined a2 FP pool, mining/pool.py genomic_window schema>
#       negative_fraction=0.9090909090909091 (§9.1 10:1) — then the Tier-2N gate:
#       scan_probe_variants → evaluate_probe_round (round_decision + degenerate guard).
#
# ⛔ RUN BLOCKED AT P2 (recorded honestly — this sbatch is correct fail-closed machinery).
#   The round-level readiness gate (spare_rule.mining_round_readiness) refuses a round with
#   no PROTECTIVE spare-rule backend available (covariation-(a) or synteny-(c); relaxed-
#   architecture-(b) alone is False on every Tier-2N locus, ADR-0006 D9 row 5). At P2:
#     • covariation-(a) needs the per-candidate homolog-search + CM-free de-novo MSA supply
#       (ADR-0006 D7/A1). The homolog-search TARGET DB exists (P2-10c′-homologdb, PR #82) but
#       the per-candidate search+alignment does NOT (mine_round.MSA_SUPPLY_AVAILABLE=False);
#     • synteny-(c) has no backend.
#   So `mine_round plan` REFUSES below and this job exits BEFORE any GPU time. The legs (a)–(d)
#   above are wired at the MSA-supply unblock step (a new imp.md step). Do NOT submit this
#   sbatch until then — at P2 it only proves the readiness refusal, which is cheaper to check
#   locally. (This is the ADR-0006 A2 scope-guard correction: the covariation pins do not
#   block a *submit*, but the round-level readiness gate does block the *round* until a
#   protective backend is MSA-producible — not merely R-scape-installed.)
#
# WHERE.   partition=gpu, 8× A4000 (the DDP retrain leg wants the whole node); node `two`.
# INPUT (ready-path; staged by rsync — the ml env has no dvc, a checkout carries no DVC data):
#   rsync -avzP data/processed/checkpoints/<round k−1 ckpt>/ two:~/tbox-finder/<...>/
#   rsync -avzP data/interim/production_genomes/ two:~/tbox-finder/data/interim/production_genomes/
#   The a2 substrate table (data/processed/mining/production_host_orders_v0.parquet) and the
#   window manifest are git-committed — they ride the checkout, no staging.
#   The round-0 parent = the A7 cold-start stage1_production (stage1.pt sha256 09931a22…940).
# SUBMIT (per round — each is one CLAUDE.md §9.3 submit-ack; SYNC FIRST, STAGE SECOND, SUBMIT THIRD):
#   ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && git checkout p2-10e && git reset --hard origin/p2-10e"'
#   # …stage the checkpoint + genome dirs (see INPUT)…
#   ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/mine_round.sbatch"'
# PREFLIGHT:  ssh two 'bash -lc "sbatch --test-only slurm/p2/mine_round.sbatch"'
# VERIFY (artifact-based; sacct off): the torch legs write stderr on success, so the per-round
#   .err is NOT gated on zero bytes — gate on the DONE marker + the self-validating round report
#   (readiness ready==true, round_decision.decision, Tier-2N recall non-decreasing vs best).
# RETRIEVE (LOCAL follow-up): rsync the round checkpoint + report back; dvc add on the laptop.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=tbox-p2-10e-mine-round
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=reports/p2/mine_round_%j.out
#SBATCH --error=reports/p2/mine_round_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
DONE="$REPO/reports/p2/mine_round.DONE"
PLAN="$REPO/reports/p2/mine_round_plan.json"
# Clear the stale completion marker FIRST — before `cd`/`mkdir` or anything else that can exit
# under `set -e` — so a DONE surviving a `git reset --hard` cannot green a round that never
# ran. Absolute paths (from REPO) so the clear does not depend on the working directory.
rm -f "$DONE"
cd "$REPO"
mkdir -p reports/p2

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna

export CUDA_HOME="$CONDA_PREFIX"          # nvcc/deepspeed probe CUDA_HOME at import (ADR-0002 A4).
export PYTHONHASHSEED=0                    # determinism (§8.3); inference is deterministic anyway.
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ── LEG 0: readiness preflight (cheap, no GPU) — the honest gate ──────────────
# `mine_round plan` probes covariation.backend_available() for R-scape presence and gates
# the covariation leg on the per-candidate MSA supply (MSA_SUPPLY_AVAILABLE, False at P2).
# It exits 3 when the round is refused (no protective backend) — the expected P2 outcome.
set +e
PYTHONPATH=src python -m tbox_finder.mining.mine_round plan --out "$PLAN"
plan_rc=$?
set -e

if [ "$plan_rc" -eq 3 ]; then
  echo "MINE_ROUND_REFUSED job=${SLURM_JOB_ID:-none} plan=$PLAN" >&2
  echo "The P2-10e mining round is blocked at the readiness gate: no protective spare-rule" >&2
  echo "backend (covariation-(a) MSA supply / synteny-(c)) is available. This is the honest," >&2
  echo "expected P2 outcome (ADR-0006 A2 scope-guard correction). The RUN unblocks when the" >&2
  echo "ADR-0006 D7/A1 per-candidate homolog-search + de-novo MSA supply lands. No round was" >&2
  echo "run and no checkpoint was produced — by design. Exiting 0 (readiness check succeeded)." >&2
  # Exit 0: the job correctly determined the round cannot run. It did NOT complete a round —
  # the ABSENCE of $DONE is the signal to the §9.3 artifact-verify that no round landed.
  exit 0
elif [ "$plan_rc" -ne 0 ]; then
  echo "FATAL: mine_round plan failed rc=$plan_rc (not a clean refusal)." >&2
  exit "$plan_rc"
fi

# ── LEGS (a)–(b): the ready-path — reached only once a protective backend is MSA-producible ──
# Unreachable at P2 (the plan refuses above; MSA_SUPPLY_AVAILABLE=False). This job runs the GPU
# legs (a)+(b) inline, then SUBMITS the decoupled CPU producer array (leg c) and the GPU retrain
# (leg d, dependent on the array) and EXITS — releasing its 8-GPU allocation so the CPU producer
# does not hold 8 idle GPUs (A10 Pin 2 decoupling). The array WIDTH is the A10 Phase-2 pin
# (measured by mine_round_measure.sbatch); ARRAY_WIDTH is passed in, no silent default guess.
ROUND="${ROUND:-0}"
ARRAY_WIDTH="${ARRAY_WIDTH:?set ARRAY_WIDTH (the A10 Phase-2-pinned producer array width) via --export}"
ROUND_DIR="data/processed/mining/round_${ROUND}"
FP_MANIFEST="$ROUND_DIR/fp_manifest.json"
SHARD_DIR="$ROUND_DIR/shards"
mkdir -p "$ROUND_DIR"

echo "READY — round $ROUND, producer array width $ARRAY_WIDTH (A10 Phase-2 pin)."

# LEG a: 8-way GPU-sharded scan over the 660 a2-admissible genomes → per-GPU FP manifests.
export CUDA_HOME="$CONDA_PREFIX"
export TOKENIZERS_PARALLELISM=false
CKPT="${CKPT:-data/processed/checkpoints/stage1_production/stage1.pt}"   # round k−1 parent
BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
PYTHONPATH=src python -m tbox_finder.mining.mine_round admissible-accessions --out "$BUILD/accessions.txt"
split -n l/8 --numeric-suffixes=0 --additional-suffix=.txt "$BUILD/accessions.txt" "$BUILD/slice_"
pids=()
for i in $(seq 0 7); do
  slice="$(printf "%s/slice_%02d.txt" "$BUILD" "$i")"
  [ -s "$slice" ] || continue
  CUDA_VISIBLE_DEVICES="$i" PYTHONHASHSEED=0 PYTHONPATH=src \
    python -m tbox_finder.mining.mine_round scan-shard --checkpoint "$CKPT" \
      --accessions "$slice" --out-manifest "$BUILD/fp_$i.json" > "$BUILD/scan_$i.log" 2>&1 &
  pids+=("$!")
done
scan_fail=0; for p in "${pids[@]}"; do wait "$p" || scan_fail=1; done
[ "$scan_fail" -eq 0 ] || { echo "FATAL: a scan shard failed" >&2; cat "$BUILD"/scan_*.log >&2; exit 3; }

# LEG b: collect per-GPU FP manifests → the round FP manifest → array shards.
PYTHONPATH=src python -m tbox_finder.mining.mine_round collect --manifests "$BUILD"/fp_*.json --out "$FP_MANIFEST"
PYTHONPATH=src python -m tbox_finder.mining.covariation_producer make-shards \
  --manifest "$FP_MANIFEST" --n-shards "$ARRAY_WIDTH" --out-dir "$SHARD_DIR"
rm -rf "$BUILD"

# LEG c: submit the decoupled producer array (CPU-only, partition=gpu no --gres); LEG d: the retrain,
# afterok-dependent on the whole array. Each is its own §9.3 artifact-verified job.
PRODUCER_JOB="$(sbatch --parsable --array="0-$((ARRAY_WIDTH - 1))" \
  --export=ALL,ROUND="$ROUND",SHARD_DIR="$SHARD_DIR",ROUND_DIR="$ROUND_DIR" \
  slurm/p2/mine_round_producer.sbatch)"
echo "LEG c: submitted producer array job $PRODUCER_JOB (width $ARRAY_WIDTH)"
RETRAIN_JOB="$(sbatch --parsable --dependency="afterok:$PRODUCER_JOB" \
  --export=ALL,ROUND="$ROUND",ROUND_DIR="$ROUND_DIR",FP_MANIFEST="$FP_MANIFEST",CKPT="$CKPT" \
  slurm/p2/mine_round_retrain.sbatch)"
echo "LEG d: submitted retrain job $RETRAIN_JOB (afterok:$PRODUCER_JOB)"
echo "mine_round orchestrator done — legs (c)+(d) run downstream; verify their DONE markers (§9.3)."
