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
# ⚠ THE STANDING REFUSAL IS GONE — CORRECTED AT P3-15′-h. This header used to read *"NOT SUBMITTABLE
# AS SHIPPED … `remine.STAGE2_SUPPLY_AVAILABLE` is False … neither the relaxed-architecture nor the
# downstream-aaRS-synteny backend exists, so leg (0) refuses with exit 1"*. All four constants are
# now True (P3-15′-a/-b/-c-ii/-d) and all four producers have run, so leg (0) no longer refuses on a
# missing backend and this file is the thing standing between the produced evidence and the round.
# What still gates a submit is the CLAUDE.md §9.3 ack plus the four staged inputs below — never this
# comment. A stale ⛔ header that shipped operationally false is exactly what P3-15′-a had to correct
# in four places; this one is corrected here rather than left to read as a live refusal.
#
# ⚠ EVERY SUPPLY DECLARATION THIS JOB MAKES IS AN OBLIGATION TO HAND THE ROUND THE PRODUCED TABLE.
# `remine.apply_remine_spare_rule` refuses BOTH directions (`remine.py`, the two paired guards):
# declared-available with no table would make the disjunct read `unavailable` for every candidate and
# spare them all — a clean-looking zero yield indistinguishable from an honest one — and a table for
# an undeclared backend is produced evidence the round never said it had. So the declarations and the
# `--*-status` arguments are composed together, below, and neither can drift from the other.
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
# The P2 production scanner this round re-mines FROM. ⚠ It is ALSO the canonical shipped
# checkpoint: ADR-0005 A12 ratified the decision-only round, so nothing has superseded it and
# nothing will unless a retrain is separately commissioned. PRD §10.1's map trains both Stage-1
# checkpoints in P2; the "supersedes the P2 checkpoint" wording this comment used to carry came
# from imp.md, not from the PRD (A12 Pin 3).
CKPT="${CKPT:-data/processed/checkpoints/stage1_production/stage1.pt}"
# No default: ADR-0005 D14 pins the RULE; the §13.1 phase gate pins the Stage-2 operating point.
# An unset value must stop the job, never silently become one.
STAGE2_THRESHOLD="${STAGE2_THRESHOLD:?set STAGE2_THRESHOLD via --export (no default; §13.1 owns it)}"

PLAN_REPORT="$ROUND_DIR/remine_plan.json"
STATUS_TABLE="$ROUND_DIR/covariation_status.json"
POSTERIORS="$ROUND_DIR/stage2_posteriors.json"
# The (b) and (c) producer merge outputs, named by their producers' own DEFAULT_STATUS_TABLE
# constants (`architecture_producer.py`, `synteny_producer.py`) and written to $ROUND_DIR by
# `slurm/p3/architecture_producer.sbatch` and the LOCAL synteny merge. Overridable via --export for
# a round whose supplies were staged elsewhere; both are cluster-scratch artifacts, never tracked.
RELAXED_ARCH_STATUS="${RELAXED_ARCH_STATUS:-$ROUND_DIR/architecture_status.json}"
SYNTENY_STATUS="${SYNTENY_STATUS:-$ROUND_DIR/synteny_status.json}"
# The round's Tier-2N probe ids ({"natural": [...], "synthetic": [...]}). No default: the committed
# reports/p2/tier2n_probe.json carries COUNTS, not member ids, so pointing at it would build an
# empty probe set and the exclusion would report 0 while excluding nothing.
PROBE_SET="${PROBE_SET:?set PROBE_SET via --export (Tier-2N probe ids; counts-only artifacts are refused)}"
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

# ── The four supply declarations, in BOTH directions ───────────────────────────────────────────────
# ⚠ THIS BLOCK USED TO BE FAIL-OPEN, AND ITS OWN COMMENT SAID THE OPPOSITE. It appended only the
# POSITIVE flag — `[ "${X:-0}" = "1" ] && PLAN_FLAGS="$PLAN_FLAGS --x"` — under a comment claiming
# *"the *_AVAILABLE flags are passed only when the corresponding backend genuinely exists"*. Each of
# the four is a two-way flag pair on the python side whose default is the MODULE constant, and since
# P3-15′-a/-b/-c-ii/-d every one of those constants is True. So an UNSET env var did not mean
# "unavailable" — argparse fell back to True and the round declared the supply available anyway,
# with `X=0` in the environment a silent no-op. Emitting the `--no-` half makes this job's own
# environment decide, in both directions, and the operator's 0 mean what it says.
declare_supply() {  # $1 = flag stem (the two-way pair's name)   $2 = env value
  if [ "${2:-0}" = "1" ]; then PLAN_FLAGS+=("--$1"); else PLAN_FLAGS+=("--no-$1"); fi
}
PLAN_FLAGS=()
# One-way on purpose: `--rscape-installed` is a bare store_true with no `--no-` half (it is a
# tool-presence fact, not a supply), so an absent flag is already the conservative direction.
if [ "${RSCAPE_INSTALLED:-0}" = "1" ]; then PLAN_FLAGS+=(--rscape-installed); fi
declare_supply msa-supply-available    "${MSA_SUPPLY_AVAILABLE:-0}"
declare_supply stage2-supply-available "${STAGE2_SUPPLY_AVAILABLE:-0}"
declare_supply relaxed-arch-available  "${RELAXED_ARCH_AVAILABLE:-0}"
declare_supply synteny-available       "${SYNTENY_AVAILABLE:-0}"

# ── ...and the produced tables those declarations oblige the round to be handed ────────────────────
# ⚠ THE DEFECT THIS FIXES: the declarations above were built into $PLAN_FLAGS and reused verbatim by
# leg (1), which passed NEITHER `--relaxed-arch-status` NOR `--synteny-status` — the two arguments
# `remine.apply_remine_spare_rule` REQUIRES whenever the matching backend is declared. A round
# declaring either backend died there with a RemineError, so criterion (b)'s producer output (SLURM
# job 1288, 941 rows) and criterion (c)'s had no path into the round at all. Checked here rather than
# at leg (1) so a mis-staged supply costs nothing: this runs before the preflight, let alone the GPUs.
STATUS_FLAGS=()
require_status() {  # $1 = env value   $2 = --flag   $3 = path   $4 = which producer writes it
  if [ "${1:-0}" != "1" ]; then
    # NOT a refusal — an undeclared backend legitimately passes no table, and declaring one
    # unavailable on purpose is a supported round. But a table sitting here is produced evidence
    # this round will never read: the disjunct resolves 'unavailable' for every candidate, all of
    # them are spared, and the round publishes a zero yield that looks exactly like an honest one.
    # Said out loud, because the only other trace of it is a 0 nobody can interpret.
    [ -s "$3" ] && echo "WARNING: $3 exists but its backend is declared UNAVAILABLE ($2 will not" \
      "be passed); this round spares every candidate on that disjunct and reads no produced" \
      "evidence. Export the matching *_AVAILABLE=1 if that is not what you meant." >&2
    return 0
  fi
  # A cheap early check only — it catches the absent/truncated file before the preflight. It is
  # NOT the authoritative one: a well-formed table that decides ZERO of this round's candidates
  # is non-empty and would pass here. `refuse_status_table_that_decides_nothing` in the round
  # itself is what closes that, because the sbatch cannot know the manifest's ids.
  [ -s "$3" ] || {
    echo "FATAL: this round declares $2 available but its status table $3 is missing/empty." >&2
    echo "Stage it from $4, or declare the backend unavailable — a declared backend with no table" >&2
    echo "makes every candidate's disjunct read 'unavailable' and spares them all (§10.3)." >&2
    exit 3; }
  STATUS_FLAGS+=("$2" "$3")
}
require_status "${RELAXED_ARCH_AVAILABLE:-0}" --relaxed-arch-status "$RELAXED_ARCH_STATUS" \
  "slurm/p3/architecture_producer.sbatch (its MERGE_OK leg)"
require_status "${SYNTENY_AVAILABLE:-0}" --synteny-status "$SYNTENY_STATUS" \
  "the LOCAL synteny_producer merge"

# ── LEG (0) the structural preflight — refuse a round that provably cannot mine ────────────────────
set +e
PYTHONPATH=src python -m tbox_finder.mining.remine plan \
  --stage2-threshold "$STAGE2_THRESHOLD" "${PLAN_FLAGS[@]}" --out "$PLAN_REPORT"
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
[ -s "$PROBE_SET" ]    || { echo "FATAL: Tier-2N probe id list $PROBE_SET missing." >&2; exit 3; }
PYTHONPATH=src python -m tbox_finder.mining.remine apply-spare-rule \
  --manifest "$FP_MANIFEST" --status-table "$STATUS_TABLE" --posteriors "$POSTERIORS" \
  --probe-set "$PROBE_SET" \
  --stage2-threshold "$STAGE2_THRESHOLD" "${PLAN_FLAGS[@]}" "${STATUS_FLAGS[@]}" \
  --out "$ROUND_REPORT"
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
# is not inherited. Only then would a re-mined checkpoint become the canonical shipped one.
# ⚠ NOT DONE, AND NOT OWED. The P3 round ran decision-only (ADR-0005 A12, signed 2026-08-13): no
# retrain, so §8.5's re-grade trigger never fired and the P2 GATE-4 grade stands. This leg is
# reachable only if a retrain is separately commissioned — it is not the P3 arc's exit condition.

touch "$DONE"
echo "P3 re-mining round complete — DONE marker written; run the LOCAL Tier-2N gate + GATE-4 re-grade."
