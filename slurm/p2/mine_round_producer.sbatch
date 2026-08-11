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
# §9.2 forbids `compute`(=zero) and pinning a node. The array WIDTH + --cpus-per-task + --time are
# the A10 Phase-2 pin (mine_round.sbatch submits with `--array=0-(W-1)`; W chosen so a shard's wall
# fits --time).
#
# ⚠ --cpus-per-task IS PART OF THE PIN, AND SHIPPED WRONG UNTIL P3-15'-e. ADR-0005 A10 Phase-2 pins
# "Array width = 48, cpus-per-task = 2 … All 48 tasks run concurrently (48 × 2 = 96 cores, no
# queueing)". This header read 4, and mine_round.sbatch's leg (c) submits WITHOUT a --cpus-per-task
# override, so the header won: 48 × 4 = 192 cores requested on a 96-core node, i.e. the pinned
# no-queueing property could not hold. 2 is also what the code actually consumes — `align-shard`
# is called with `--cpu 2` (STAGE 2 below; named rather than numbered, because a line reference in
# a comment goes stale the next time the comment above it grows) and `homolog_msa.nhmmer_argv`
# passes no `--cpu` at all, so
# nhmmer runs at HMMER's 2-thread default. tests/unit/test_sbatch_producer_envelope.py re-derives
# the pinned numbers from the ADR's own text and compares them to these bytes.
#
# ⚠ MEMORY, NOT CPU, BINDS CONCURRENCY — and A10 did not price it. At --mem=16G a 256 GB node holds
# ~16 co-resident tasks, so a 48-wide array runs in ~3 waves and the wall is ~15-25 min, NOT A10's
# "~7 min". The PINNED COST is unaffected (≈4.42 CPU-core-h sum-faithful, 0 GPU-h) because total
# core-hours do not depend on how the waves pack. --mem is left at 16G: no per-task RSS has ever
# been measured (the K=50 Phase-1 job ran ONE task at --mem=128G), and lowering it on a guess would
# risk an OOM mid-array. Quote the wall from the run, never from A10's CPU-only arithmetic.
#
# ⚠ ONE CANDIDATE CAN CONSUME THE WHOLE SHARD'S WALL — HENCE ALIGN_TIMEOUT_S (P3-15'-e-ii).
# Job 1205 (the full-corpus run) lost shard 016 to the 12 h wall having completed 1 of its 8
# alignable candidates: a single mlocarna ran 6h11m without finishing. Across the 47 shards that
# DID finish, 271 of 272 successful alignments took ≤ 353 s and one took 8500 s at depth 844 — that
# one candidate is 69 % of all align wall-clock in the corpus. LocARNA's cost climbs steeply in MSA
# depth (A10 Pin 2's own "scaling caveat"), so the distribution has no useful tail bound and a
# longer --time is not a fix. ALIGN_TIMEOUT_S bounds each alignment; exceeding it leaves no MSA ⇒
# `unavailable` ⇒ SPARED (ADR-0005 D14), which costs yield and never correctness.
#
# ⚠ OPTIONAL SEQUENCE ROUTE — `QUERY_FASTA` (P3-15'-g-iii). Unset or empty (every run so far) ⇒
# nothing changes: each query is re-resolved from $GENOME_DIR at its manifest coordinates, A10
# Pin 2's route. Set ⇒ the shard's queries come from that FASTA keyed by candidate_id and ONLY the
# query changes — homolog extraction, cutoffs, align, promotion, score and merge are the same bytes,
# which is what lets the matched positive control claim it ran this instrument rather than one like
# it. It exists because a curated record's context is not one of the 2,500 searched genomes, so the
# coordinate route raises HomologDbError on it and that exception aborts the whole array task.
#
# ── SUBMITTED BY mine_round.sbatch (leg c) with --export ROUND/SHARD_DIR/ROUND_DIR/ALIGN_TIMEOUT_S
# and --array=0-(W-1).
# Standalone (debug) submit:
#   `sbatch --array=0-0 --export=ALL,SHARD_DIR=…,ROUND_DIR=…,ALIGN_TIMEOUT_S=600 mine_round_producer.sbatch`
# Curated-control submit (P3-15'-g-iii; the token list is DERIVED, not typed — see
#   `reports/p3/curated_control_run_plan.json`, which re-derives it from the committed manifest on
#   every run, and `python -m tbox_finder.mining.curated_control_run plan` to regenerate it):
#   make-shards first, into the control's OWN scratch ROUND_DIR (never the FP supply's — the
#   control's consensuses must not be mixed into the supply the P3-15'-f sweep reads), then submit.
# VERIFY (artifact-based, §9.3): every task's $ROUND_DIR/status/shard_<t>.SHARD_OK present + its .err
#   zero bytes; the retrain leg (d) merges the per-shard status tables and fails loud on a gap.
#   ⚠ A standalone run has NO leg (d): count the SHARD_OK markers against the shard files yourself.

#SBATCH --job-name=tbox-p2-10e-producer
#SBATCH --partition=gpu
#SBATCH --array=0
#SBATCH --cpus-per-task=2
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
# Per-candidate mlocarna wall-clock bound; exceeding it ⇒ no MSA ⇒ `unavailable` ⇒ SPARED (D14).
# Required, no default: it decides which candidates yield a consensus, so the round names it and
# `--test-only` cannot catch its absence (headers only) — the `:?` is what actually catches it.
ALIGN_TIMEOUT_S="${ALIGN_TIMEOUT_S:?set ALIGN_TIMEOUT_S (per-candidate mlocarna bound, s) via --export}"
# ⚠ `:?` REJECTS ONLY AN ABSENT OR EMPTY VALUE, NOT A MALFORMED ONE. Without this, `60o` or `abc`
# travels all the way into the producer task and is caught by argparse only AFTER that task has
# spent its ~37 min search stage — and, from the orchestrator, after the 8-GPU scan legs. Shape
# first (digits with at most one dot), then reject a numeric zero: a zero bound times out EVERY
# alignment, and because a timeout SPARES rather than fails that would read as a clean,
# producible-nothing round. Stripping `0` and `.` leaves nothing iff the value is zero.
case "$ALIGN_TIMEOUT_S" in
  ''|*[!0-9.]*|.|*.*.*) echo "FATAL: ALIGN_TIMEOUT_S must be a positive number of seconds (got '$ALIGN_TIMEOUT_S')" >&2; exit 2 ;;
esac
if [ -z "${ALIGN_TIMEOUT_S//[0.]/}" ]; then
  echo "FATAL: ALIGN_TIMEOUT_S must be > 0 (got '$ALIGN_TIMEOUT_S'); a zero bound spares every candidate" >&2; exit 2
fi
# ⚠ OPTIONAL, AND ITS DEFAULT IS THE WHOLE POINT: empty ⇒ the coordinate route, byte-for-byte what
# every previous run did. A non-empty value is checked HERE, beside the other cheap guards and
# before the ~37 min search stage, because `--query-fasta` on a missing path would otherwise be
# argparse-valid and die inside the producer only after that stage was spent — and because a shard
# that silently searched nothing would resolve every candidate `unavailable`, which reads exactly
# like an honest not-producible round. Validated as a NON-EMPTY FILE, not merely as present.
QUERY_FASTA="${QUERY_FASTA:-}"
SEARCH_QUERY_ARGS=()
if [ -n "$QUERY_FASTA" ]; then
  [ -s "$QUERY_FASTA" ] || { echo "FATAL: QUERY_FASTA='$QUERY_FASTA' is missing or empty (sequence route requested)" >&2; exit 2; }
  # ⚠ `*.fasta` IS GIT-LFS-TRACKED (.gitattributes), and this checkout has no smudge filter:
  # after the §9.3 step-3 `git reset --hard` the path holds a ~130-byte POINTER, which is a
  # perfectly good non-empty file and sails past `-s`. Caught here by name so the operator is
  # told `git lfs pull` rather than reading "no FASTA records" on all 16 tasks after the queue.
  case "$(head -n 1 "$QUERY_FASTA")" in
    version\ https://git-lfs.github.com/spec/v1*) echo "FATAL: QUERY_FASTA='$QUERY_FASTA' is a git-LFS POINTER, not sequences — run \`git lfs pull --include=$QUERY_FASTA\` on this checkout (\`git reset --hard\` re-reverts it every sync)" >&2; exit 2 ;;
  esac
  SEARCH_QUERY_ARGS=(--query-fasta "$QUERY_FASTA")
fi
# ⚠ THESE PATHS ARRIVE VIA `--export`, WHOSE VALUES ARE EXPANDED BY THE SUBMITTING SHELL, NOT
# BY THIS ONE. A surviving literal `$` means the submit line reached sbatch without shell
# expansion (an argv list handed to a non-shell exec, or a quoted `--export` token), so every
# path below would resolve against the CWD instead of $HOME. It does fail closed — the next
# guard reports `shard … missing` — but that message names the wrong problem, and on the
# ROUND_DIR path it would silently create a scratch tree inside the repo checkout.
for _p in "$SHARD_DIR" "$ROUND_DIR" "$QUERY_FASTA"; do
  case "$_p" in
    *'$'*) echo "FATAL: exported path '$_p' still contains a literal '\$' — run the submit line in a shell so --export values are expanded before sbatch sees them" >&2; exit 2 ;;
  esac
done
# Zero-pad the task id to match make-shards' `shard_{i:03d}.json` (covariation_producer.py:199).
# Build these paths with plain interpolation, NOT `$(printf …)`: the tracked-path gate
# (tests/unit/test_sbatch_rm_targets.py) cannot resolve a command substitution, so a `rm` on a
# `$(printf …)`-derived path resolves to the widest `*` and matches every tracked file. The plain
# form below resolves to `*/status/shard_*.SHARD_OK` — a suffix that matches no tracked path —
# exactly like the sibling `mine_round_retrain.sbatch:39-40` clears its own `.DONE` marker.
printf -v TASK3 "%03d" "$TASK"
SHARD="$SHARD_DIR/shard_$TASK3.json"
STATUS_DIR="$ROUND_DIR/status"
STATUS_OUT="$STATUS_DIR/status_$TASK3.json"
OK_MARKER="$STATUS_DIR/shard_$TASK3.SHARD_OK"
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
  --db-prefix "$BLAST_PREFIX" --fm-index "$FM_INDEX" --genome-dir "$GENOME_DIR" \
  "${SEARCH_QUERY_ARGS[@]}"

# ── STAGE 2: ALIGN (tbox-locarna) ────────────────────────────────────────────────────────────────
conda activate tbox-locarna
command -v mlocarna >/dev/null || { echo "FATAL: mlocarna not on PATH — tbox-locarna not created?" >&2; exit 2; }
mlocarna --version 2>&1 | grep -q "LocARNA 2.0.1" || { echo "FATAL: mlocarna != LocARNA 2.0.1 (ADR-0002 A13)" >&2; exit 2; }
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer align-shard \
  --shard "$SHARD" --workroot "$BUILD/work" --cpu 2 --align-timeout-s "$ALIGN_TIMEOUT_S"

# ── PROMOTE the per-candidate #=GC SS_cons consensus OUT of node-local scratch ────────────────────
# ADR-0006 A4 makes this artifact load-bearing for criterion (b) as well as (a): (b)'s
# named_elements_present / ncca_bulge_detected are localized from THIS file. Until P3-15'-d it was
# written under $BUILD/work and destroyed by the EXIT trap below, so the only CM-free structure the
# repo produces did not survive the job that produced it. Copy runs BEFORE stage 3 so a scoring
# failure cannot take the consensus down with it. Per-candidate subdirectory names are the (a)
# producer's own slugs, which is exactly what architecture_producer.candidate_msa_path() rebuilds.
#
# ⚠ EACH CANDIDATE'S PROMOTED DIRECTORY IS CLEARED FIRST, whether or not this run produced
# an msa.sto for it. Copying only on success would leave a PREVIOUS run's alignment in place
# for a candidate that is not producible this time, and the (b) producer would then score a
# stale consensus instead of resolving `unavailable` ⇒ spared. The loop iterates this
# shard's own workroot, so the cleared set is exactly the candidates this task owns — no
# other task's output is touched, and there is no cross-shard race.
MSA_OUT="${ROUND_DIR:?}/msa"
mkdir -p "$MSA_OUT"
n_promoted=0
n_cleared=0
for d in "$BUILD"/work/*/; do
  slug=$(basename "$d")
  [ -n "$slug" ] && [ "$slug" != "*" ] || continue     # the glob matched nothing
  if [ -e "$MSA_OUT/$slug/msa.sto" ]; then
    rm -f "$MSA_OUT/$slug/msa.sto"                     # a stale promotion from an earlier run
    n_cleared=$((n_cleared + 1))
  fi
  [ -s "$d/msa.sto" ] || continue                      # not producible (< min_sequences, align failed)
  mkdir -p "$MSA_OUT/$slug"
  # ⚠ ATOMIC: copy to a temporary name in the DESTINATION directory, then rename. `mv`
  # within one filesystem is atomic, so a task killed mid-copy (wall limit, preemption,
  # node failure) cannot leave a partial consensus at the path the (b) producer reads.
  # A truncation is NOT reliably self-announcing: one that drops trailing sequence rows but
  # keeps `#=GC SS_cons` parses as a perfectly valid alignment of REDUCED DEPTH, and (b)
  # would score it as real evidence at the wrong depth against the A2 floor. The
  # stale-promotion guard above does not cover this — that one is about an OLD file, this
  # is about a HALF-WRITTEN one.
  cp -p "$d/msa.sto" "$MSA_OUT/$slug/.msa.sto.$$.tmp"
  mv -f "$MSA_OUT/$slug/.msa.sto.$$.tmp" "$MSA_OUT/$slug/msa.sto"
  n_promoted=$((n_promoted + 1))
done
echo "promoted $n_promoted per-candidate msa.sto → $MSA_OUT (cleared $n_cleared stale)"

# ── STAGE 3: SCORE (tbox-rscape) → the shard status table ────────────────────────────────────────
conda activate tbox-rscape
command -v R-scape >/dev/null || { echo "FATAL: R-scape not on PATH — tbox-rscape not created?" >&2; exit 2; }
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer score-shard \
  --shard "$SHARD" --workroot "$BUILD/work" --out-table "$STATUS_OUT"

touch "$OK_MARKER"
echo "shard $TASK scored → $STATUS_OUT ; SHARD_OK written"
