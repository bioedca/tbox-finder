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
# ⛔ STILL DO NOT SUBMIT — BUT THE REASON CHANGED AT P3-15′-a (2026-08-05). READ THIS.
#   Until P3-15′-a the readiness gate (spare_rule.mining_round_readiness) refused this round
#   outright — no PROTECTIVE backend was available (covariation-(a) or synteny-(c); relaxed-
#   architecture-(b) alone is False on every Tier-2N locus, ADR-0006 D9 row 5) — so leg 0
#   below exited BEFORE any GPU time and the job was self-limiting.
#   THAT IS NO LONGER TRUE. The covariation-(a) supply landed (target DB job 741 + certified
#   producer job 766; mine_round.MSA_SUPPLY_AVAILABLE=True), so readiness now PASSES and leg 0
#   returns 0 — this job WILL fall through to the 8-GPU scan legs if submitted.
#   And it would mine NOTHING. Sparing is a disjunction, so mining is its negation, a
#   CONJUNCTION: a candidate is mined only when EVERY disjunct ran and failed. Relaxed-
#   architecture-(b) and synteny-(c) still have no backend, so every candidate is spared
#   fail-closed and the round's maximum yield is exactly 0 — decidable before a single window
#   is scanned. The authority is tbox_finder.mining.remine.max_possible_yield (P3-15), which
#   slurm/p3/stage1_remine.sbatch enforces at ITS leg 0; this P2 driver predates that gate and
#   requires the STRUCTURAL_YIELD_ACK below instead, so the no-op cannot be submitted by
#   accident. It unblocks when P3-15′-c (synteny) and P3-15′-d (relaxed architecture) land.
#   (ADR-0006 A2's scope-guard correction still holds: covariation-(a) availability is gated on
#   MSA-PRODUCIBILITY, not on R-scape being installed. What changed is that it is now producible.)
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
# 12 h wall: at the MEASURED 23.87 win/s/GPU (job 806) the heaviest `split -n l/8` shard
# (524,959 windows) ≈ 6.1 h scan; the old 5 h wall timed leg (a) out before the producer
# array (job 793). The gpu partition TIMELIMIT is infinite, so a longer soft ceiling is free.
#SBATCH --time=12:00:00
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
# `mine_round plan` probes covariation.backend_available() for R-scape presence and gates the
# covariation leg on the per-candidate MSA supply (MSA_SUPPLY_AVAILABLE, True since P3-15′-a).
# Exit codes: 3 = the round is REFUSED at readiness (no protective backend); 4 = this checkout
# DECLARES the MSA supply but cannot evidence it (derive_msa_supply_available) — a staging
# fault, not a clean refusal, so it must NOT be swallowed as "no round today".
set +e
PYTHONPATH=src python -m tbox_finder.mining.mine_round plan --out "$PLAN"
plan_rc=$?
set -e

if [ "$plan_rc" -eq 3 ]; then
  echo "MINE_ROUND_REFUSED job=${SLURM_JOB_ID:-none} plan=$PLAN" >&2
  echo "The mining round is blocked at the readiness gate: no protective spare-rule backend" >&2
  echo "(covariation-(a) MSA supply / synteny-(c)) is available on this run. Since P3-15′-a the" >&2
  echo "(a) supply exists, so reaching this branch means it was declared unavailable here" >&2
  echo "(--no-msa-supply-available, or the DB was not staged). No round was run and no" >&2
  echo "checkpoint was produced — by design. Exiting 0 (readiness check succeeded)." >&2
  # Exit 0: the job correctly determined the round cannot run. It did NOT complete a round —
  # the ABSENCE of $DONE is the signal to the §9.3 artifact-verify that no round landed.
  exit 0
elif [ "$plan_rc" -ne 0 ]; then
  echo "FATAL: mine_round plan failed rc=$plan_rc (not a clean refusal; 4 = declared-but-unevidenced MSA supply)." >&2
  exit "$plan_rc"
fi

# ── LEGS (a)–(b): the ready-path — reached once a protective backend is MSA-producible ──
# REACHABLE since P3-15′-a (readiness passes on covariation-(a)). This job runs the GPU
# legs (a)+(b) inline, then SUBMITS the decoupled CPU producer array (leg c) and the GPU retrain
# (leg d, dependent on the array) and EXITS — releasing its 8-GPU allocation so the CPU producer
# does not hold 8 idle GPUs (A10 Pin 2 decoupling). The array WIDTH is the A10 Phase-2 pin
# (measured by mine_round_measure.sbatch); ARRAY_WIDTH is passed in, no silent default guess.
ROUND="${ROUND:-0}"
ARRAY_WIDTH="${ARRAY_WIDTH:?set ARRAY_WIDTH (the A10 Phase-2-pinned producer array width) via --export}"
# The producer array's per-candidate mlocarna bound. Required here too, and forwarded to leg (c)
# below: mine_round_producer.sbatch declares it `:?`, so a round that omitted it would submit an
# array whose every task aborts on the first line of its own config — a 48-task no-op that reports
# nothing. Exceeding the bound ⇒ no MSA ⇒ `unavailable` ⇒ SPARED (D14), never a mined negative.
ALIGN_TIMEOUT_S="${ALIGN_TIMEOUT_S:?set ALIGN_TIMEOUT_S (per-candidate mlocarna bound, s) via --export}"
# The structural-yield ack (header ⛔). Readiness passing is NOT permission to mine: with (b)
# and (c) unbacked, mining's conjunction cannot be satisfied by any candidate and this round's
# yield is provably 0 — ~8 GPU × 6 h to change nothing. Required, no default, so the no-op
# cannot be submitted by accident; drop it once P3-15′-c/-d land.
STRUCTURAL_YIELD_ACK="${STRUCTURAL_YIELD_ACK:?refusing: relaxed_architecture and downstream_aaRS_synteny have no backend, so remine.max_possible_yield bounds this round at 0 mined. Set STRUCTURAL_YIELD_ACK=1 via --export only if you intend to spend the GPU legs on a knowingly 0-yield round}"
[ "$STRUCTURAL_YIELD_ACK" = "1" ] || { echo "FATAL: STRUCTURAL_YIELD_ACK must be exactly 1 (got '$STRUCTURAL_YIELD_ACK')" >&2; exit 2; }
ROUND_DIR="data/processed/mining/round_${ROUND}"
FP_MANIFEST="$ROUND_DIR/fp_manifest.json"
SHARD_DIR="$ROUND_DIR/shards"
mkdir -p "$ROUND_DIR"

echo "READY — round $ROUND, producer array width $ARRAY_WIDTH (A10 Phase-2 pin)."

# LEG a: 8-way GPU-sharded scan over the 660 a2-admissible genomes → per-GPU FP manifests.
export CUDA_HOME="$CONDA_PREFIX"
export TOKENIZERS_PARALLELISM=false
CKPT="${CKPT:-data/processed/checkpoints/stage1_production/stage1.pt}"   # round k−1 parent
[ -s "$CKPT" ] || { echo "FATAL: parent checkpoint $CKPT missing — stage the round k−1 ckpt (rsync)." >&2; exit 2; }
# NOTE: name this JOB_SCRATCH, never BUILD/HOST — `conda activate tbox-ml-dna` (the compiler
# toolchain) exports BUILD=HOST=x86_64-conda-linux-gnu and would clobber a $BUILD scratch path the
# instant the env activates (job 789 died on exactly this). The EXIT trap cleans it on EVERY exit
# path (incl. the failing-shard `exit 3` below) — a bare `rm` only ran on the success path.
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT
PYTHONPATH=src python -m tbox_finder.mining.mine_round admissible-accessions --out "$JOB_SCRATCH/accessions.txt"
# Fail loud on an empty admissible set — the cluster has no git-lfs, so a `git reset --hard` can
# leave split_assignments.parquet a 132-byte LFS pointer that reads 0 rows ([[git-lfs-pointers-in-ci]]);
# without this the 8-way split yields all-empty slices and `collect` dies later on an unexpanded glob.
n_acc="$(wc -l < "$JOB_SCRATCH/accessions.txt")"
[ "$n_acc" -ge 8 ] || { echo "FATAL: only $n_acc admissible accessions (<8) — is split_assignments.parquet an LFS pointer?" >&2; exit 2; }
split -n l/8 --numeric-suffixes=0 --additional-suffix=.txt "$JOB_SCRATCH/accessions.txt" "$JOB_SCRATCH/slice_"
pids=()
for i in $(seq 0 7); do
  slice="$(printf "%s/slice_%02d.txt" "$JOB_SCRATCH" "$i")"
  [ -s "$slice" ] || continue
  # Persistent per-shard throughput progress in reports/ (NOT $JOB_SCRATCH, which the EXIT trap
  # wipes) so a wall-clock timeout still leaves win/s/GPU + FP-rate + in_progress_accession per shard
  # — the job-793 diagnostic gap. --progress-out installs the SIGTERM→finally-flush handler.
  shard_progress="$(printf "reports/p2/mine_round_r%s_shard%d.json" "$ROUND" "$i")"
  CUDA_VISIBLE_DEVICES="$i" PYTHONHASHSEED=0 PYTHONPATH=src \
    python -m tbox_finder.mining.mine_round scan-shard --checkpoint "$CKPT" \
      --accessions "$slice" --out-manifest "$JOB_SCRATCH/fp_$i.json" \
      --progress-out "$shard_progress" --flush-every-windows 500 \
      > "$JOB_SCRATCH/scan_$i.log" 2>&1 &
  pids+=("$!")
done
scan_fail=0; for p in "${pids[@]}"; do wait "$p" || scan_fail=1; done
[ "$scan_fail" -eq 0 ] || { echo "FATAL: a scan shard failed" >&2; cat "$JOB_SCRATCH"/scan_*.log >&2; exit 3; }

# LEG b: collect per-GPU FP manifests → the round FP manifest → array shards.
PYTHONPATH=src python -m tbox_finder.mining.mine_round collect --manifests "$JOB_SCRATCH"/fp_*.json --out "$FP_MANIFEST"
PYTHONPATH=src python -m tbox_finder.mining.covariation_producer make-shards \
  --manifest "$FP_MANIFEST" --n-shards "$ARRAY_WIDTH" --out-dir "$SHARD_DIR"

# LEG c: submit the decoupled producer array (CPU-only, partition=gpu no --gres); LEG d: the retrain,
# afterok-dependent on the whole array. Each is its own §9.3 artifact-verified job.
PRODUCER_JOB="$(sbatch --parsable --array="0-$((ARRAY_WIDTH - 1))" \
  --export=ALL,ROUND="$ROUND",SHARD_DIR="$SHARD_DIR",ROUND_DIR="$ROUND_DIR",ALIGN_TIMEOUT_S="$ALIGN_TIMEOUT_S" \
  slurm/p2/mine_round_producer.sbatch)"
echo "LEG c: submitted producer array job $PRODUCER_JOB (width $ARRAY_WIDTH)"
RETRAIN_JOB="$(sbatch --parsable --dependency="afterok:$PRODUCER_JOB" \
  --export=ALL,ROUND="$ROUND",ROUND_DIR="$ROUND_DIR",FP_MANIFEST="$FP_MANIFEST",CKPT="$CKPT" \
  slurm/p2/mine_round_retrain.sbatch)"
echo "LEG d: submitted retrain job $RETRAIN_JOB (afterok:$PRODUCER_JOB)"
echo "mine_round orchestrator done — legs (c)+(d) run downstream; verify their DONE markers (§9.3)."
