#!/bin/bash
# P3-15'-d — the per-candidate criterion-(b) relaxed-architecture producer.
# (PRD §13.2/§13.3(b); ADR-0006 D3 the predicate, D6 the short-Stem-I threshold, D9 row 5,
#  A3/A4 the de-novo comparative-consensus instrument; ADR-0005 D14 the spare rule; CLAUDE.md §9.3.)
#
# Turns each candidate's PROMOTED `#=GC SS_cons` consensus ("$ROUND_DIR/msa/<slug>/msa.sto",
# written by slurm/p2/mine_round_producer.sbatch since P3-15'-d) into the
# `candidate_id -> status` table `mine_round.apply_spare_rule --relaxed-arch-status` reads.
#
# ── CPU-ONLY, AND GENUINELY SMALL ────────────────────────────────────────────────────
# The whole job is stdlib Stockholm parsing + a pair-table walk per candidate: no aligner,
# no HMM, no GPU, no network. Measured locally on the certified 24-row / 289-column
# consensus: parse + pair_table + decompose + NCCA scan is ~1 ms per candidate, so 941
# candidates is ~1 s of actual work. --time is 00:20:00 because the wall is filesystem
# latency over $ROUND_DIR, not compute. NO --gres: a GPU allocation here would idle a
# scarce A4000 (cryosparc already keeps both nodes fragmented).
#
# ── WHY AN ARRAY OF 2 FOR A ONE-SECOND JOB ───────────────────────────────────────────
# Same reason as slurm/p3/stage2_producer.sbatch: `merge`'s duplicate-candidate refusal
# and its config-disagreement refusal are the guards that stop a truncated or
# mixed-config run from certifying, and a single-shard array would never exercise either
# on the cluster. Two tasks is the smallest width that actually runs the merge path it
# ships. Widen with `--export=...,N_SHARDS=<n>` plus a matching `--array=0-<n-1>`; the two
# MUST agree (asserted below, exit 2 if not).
#
# ── EVERY RULE PARAMETER IS SUPPLIED HERE, NONE IS DEFAULTED ─────────────────────────
# ADR-0006 A4: D6's short-Stem-I nt threshold ships keyword-required-with-no-default,
# supplied per round and recorded in provenance; D17's P6 freeze is NOT consumed. The same
# holds for every other number the predicate uses. The values below are this ROUND's
# choices and are exported so a re-run is reproducible from the submit line alone — they
# are NOT pinned anywhere in the code, and reports/p3/architecture_freeze.json is the
# held-out-canonical measurement that informs them (it pins nothing either).
#   STEM_I_NT_THRESHOLD  D6's carve-out gate.
#   MIN_NAMED_HELICES    how many distinct helices count as "the expected helices".
#   MIN_HELIX_PAIRS      minimum stacked pairs before a helix counts at all.
#   BULGE_MIN_NT/MAX_NT  admissible bulge length IN THE CANDIDATE'S OWN RESIDUES.
#                        ⚠ NOT alignment columns: on the certified consensus a 16-column
#                        run is 8 residues and a 37-column run is 11, so a column-width
#                        filter rejects real bulges. Measured, not assumed.
#   NCCA_PAIRING_NT      how many tRNA-acceptor-end contacts must be evidenced.
#   ALLOW_WOBBLE         0 or 1 — whether a G·U wobble counts at the acceptor interface.
#
# ── NO STALE-MARKER DELETION, DELIBERATELY ───────────────────────────────────────────
# Markers live under a per-ARRAY-job directory ($SLURM_ARRAY_JOB_ID), so a fresh submit
# cannot inherit a previous one's markers and there is nothing to `rm`. That also keeps
# this file clear of the tests/unit/test_sbatch_rm_targets.py trap, where an unresolvable
# `rm` target widens to a bare `*` and matches every tracked path.
#
# WHERE:   cluster `gpu` partition, CPU-only (no --gres). env tbox-data (stdlib + pandas;
#          the producer's run-shard leg needs neither). No --nodelist.
# INPUT:   data/processed/mining/round0_fp_manifest.json  (git-tracked, 941 candidates)
#          "$ROUND_DIR/msa/<slug>/msa.sto"                (promoted by the (a) producer array)
#          ⚠ A candidate with no msa.sto resolves to `unavailable` => SPARED, never mined
#          (ADR-0005 D14). Running this BEFORE the (a) array therefore produces a table of
#          941 `unavailable` — honest, and useless. Run it AFTER.
# SUBMIT:  from the repo root so the relative --output/--error resolve. SYNC, THEN SUBMIT.
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p3-15-prime-d && git reset --hard origin/p3-15-prime-d"'
#          ⚠ ALL SEVEN RULE PARAMETERS MUST BE EXPORTED TOO — each is `${VAR:?}` below, so a
#          submit that omits any of them exits before producing a single shard. There is
#          deliberately no default: ADR-0006 A4 ships them keyword-required so the round
#          chooses them and provenance records the choice. Consult
#          reports/p3/architecture_freeze.json before choosing; the placeholders below are
#          NOT a pin.
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch --export=ALL,ROUND_DIR=<dir>,\
#          STEM_I_NT_THRESHOLD=<nt>,MIN_NAMED_HELICES=<n>,MIN_HELIX_PAIRS=<n>,\
#          BULGE_MIN_NT=<nt>,BULGE_MAX_NT=<nt>,NCCA_PAIRING_NT=<n>,ALLOW_WOBBLE=<0|1> \
#          slurm/p3/architecture_producer.sbatch"'
# PREFLIGHT: `sbatch --test-only` validates the SLURM HEADERS ONLY — it does not run the
#          script body, so it will NOT catch a missing rule parameter. Run both:
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p3/architecture_producer.sbatch"'
#            ssh two 'bash -lc "cd ~/tbox-finder && bash -n slurm/p3/architecture_producer.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b) every
#          "$ROUND_DIR/markers_<arrayjobid>/shard_NNN.SHARD_OK" present, (c) every `.err`
#          zero bytes (this job has no chatty downloads, unlike the Stage-2 producer), (d)
#          "$ROUND_DIR/architecture_status.json" exists and its n_candidates == 941.
#          A missing marker is a §9.3 stop: surface it, no auto-resubmit.
# RETRIEVE: plain scratch, NOT dvc:
#            rsync -avzP two:<ROUND_DIR>/architecture_status.json <local-round-dir>/

#SBATCH --job-name=tbox-p3-15d-arch-producer
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-1
#SBATCH --output=reports/p3/architecture_producer_%A_%a.out
#SBATCH --error=reports/p3/architecture_producer_%A_%a.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often
# down). No --gres: this leg is CPU-only and must not hold a GPU.

set -eo pipefail   # NOT -u: the conda activate hooks read unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# The cluster env carries no editable install of this package; every sbatch here prefixes
# PYTHONPATH=src for that reason. `src` FIRST so an inherited PYTHONPATH pointing at
# another checkout cannot take precedence and score with a different revision.
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-2}"
FP_MANIFEST="${FP_MANIFEST:-data/processed/mining/round0_fp_manifest.json}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
printf -v TASK3 "%03d" "$TASK"

# ── THIS ROUND'S RULE PARAMETERS (see the header) ────────────────────────────────────
STEM_I_NT_THRESHOLD="${STEM_I_NT_THRESHOLD:?set the D6 short-Stem-I nt threshold for this round}"
MIN_NAMED_HELICES="${MIN_NAMED_HELICES:?set the expected-helix count for this round}"
MIN_HELIX_PAIRS="${MIN_HELIX_PAIRS:?set the minimum stacked pairs per helix for this round}"
BULGE_MIN_NT="${BULGE_MIN_NT:?set the minimum bulge length (residues) for this round}"
BULGE_MAX_NT="${BULGE_MAX_NT:?set the maximum bulge length (residues) for this round}"
NCCA_PAIRING_NT="${NCCA_PAIRING_NT:?set how many acceptor-end contacts must be evidenced}"
# ⚠ ALSO REQUIRED, and it was missed: without this the round's G·U policy is decided by
# ArchitectureRunConfig.allow_wobble's code default (False) — exactly the outcome the
# producer docstring says must not happen. `--min-sequences` is deliberately NOT here: it
# is the ADR-0006 A2 Pin 2 floor, a pin rather than a round choice.
ALLOW_WOBBLE="${ALLOW_WOBBLE:?set 0 or 1 — whether G-U counts at the acceptor interface}"
case "$ALLOW_WOBBLE" in
  0) WOBBLE_FLAG=() ;;
  1) WOBBLE_FLAG=(--allow-wobble) ;;
  *) echo "FATAL: ALLOW_WOBBLE must be 0 or 1, got '$ALLOW_WOBBLE'" >&2; exit 2 ;;
esac

# ROUND_DIR is SCRATCH and must never be a git-tracked path: the round's tables are run
# artifacts, and an operator pointing this at data/processed/mining/ would leave untracked
# JSON inside the repo that every `git status`-reading provenance gate then sees.
ROUND_DIR="${ROUND_DIR:-$HOME/tbox-scratch/round_p3_15d}"
if [ -z "$ROUND_DIR" ]; then
  echo "FATAL: ROUND_DIR resolved empty; refusing to write anywhere." >&2
  exit 2
fi
# CANONICALISED, not pattern-matched: a lexical `case` accepts a bare relative value,
# `$REPO` itself, and anything reachable via `..` or a symlink — all of which land inside
# the repo once the script has cd'd there. `realpath -m` resolves a not-yet-existing path.
ROUND_DIR="$(realpath -m -- "$ROUND_DIR")"
REPO_REAL="$(realpath -m -- "$REPO")"
case "$ROUND_DIR" in
  /*) : ;;
  *)  echo "FATAL: ROUND_DIR must be absolute, got $ROUND_DIR" >&2; exit 2 ;;
esac
if [ "$ROUND_DIR" = "$REPO_REAL" ] || case "$ROUND_DIR" in "$REPO_REAL"/*) true ;; *) false ;; esac; then
  echo "FATAL: ROUND_DIR=$ROUND_DIR resolves inside the repo ($REPO_REAL)." >&2
  echo "       The round's tables are run artifacts and must live on scratch (§9.2)." >&2
  exit 2
fi

# The COMPLETE array shape, not just this task's index: N_SHARDS=3 with --array=0-1 would
# silently drop a third of the candidates into a table that still merges cleanly.
ARRAY_COUNT="${SLURM_ARRAY_TASK_COUNT:-$N_SHARDS}"
if [ "$ARRAY_COUNT" -ne "$N_SHARDS" ]; then
  echo "FATAL: --array width $ARRAY_COUNT != N_SHARDS $N_SHARDS; a shard would be dropped" >&2
  exit 2
fi

MSA_DIR="$ROUND_DIR/msa"
SHARD_DIR="$ROUND_DIR/arch_shards"
MARKER_DIR="$ROUND_DIR/markers_$ARRAY_ID"
STATUS_OUT="$SHARD_DIR/status_$TASK3.json"
MERGED_OUT="$ROUND_DIR/architecture_status.json"
mkdir -p "$SHARD_DIR" "$MARKER_DIR"

[ -s "$FP_MANIFEST" ] || { echo "FATAL: $FP_MANIFEST missing" >&2; exit 2; }
if [ ! -d "$MSA_DIR" ]; then
  echo "FATAL: $MSA_DIR missing — the (a) producer array has not promoted its consensuses." >&2
  echo "       Every candidate would read 'unavailable' and be spared; refusing." >&2
  exit 2
fi

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-data
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array_task=$TASK/$N_SHARDS round_dir=$ROUND_DIR"

PYTHONHASHSEED=0 python -m tbox_finder.mining.architecture_producer run-shard \
  --manifest "$FP_MANIFEST" \
  --msa-dir "$MSA_DIR" \
  --shard-index "$TASK" --n-shards "$N_SHARDS" \
  --stem-i-nt-threshold "$STEM_I_NT_THRESHOLD" \
  --min-named-helices "$MIN_NAMED_HELICES" \
  --min-helix-pairs "$MIN_HELIX_PAIRS" \
  --bulge-min-nt "$BULGE_MIN_NT" \
  --bulge-max-nt "$BULGE_MAX_NT" \
  --ncca-pairing-nt "$NCCA_PAIRING_NT" \
  "${WOBBLE_FLAG[@]}" \
  --out "$STATUS_OUT"

touch "$MARKER_DIR/shard_$TASK3.SHARD_OK"
echo "shard $TASK scored -> $STATUS_OUT ; SHARD_OK written"

# ── MERGE LEG: whichever task finishes last sees a full marker set and merges ────────
# Deterministic without a second submit and without a dependency job.
#
# The flock serialises the check-then-merge so two tasks finishing together cannot both
# pass the completeness test and race on the same output — `_write` uses `Path.write_text`,
# which is not an atomic replace, so a concurrent second writer can interleave and leave a
# truncated table that still parses. Same idiom as slurm/p3/stage2_producer.sbatch.
#
# BLOCKING, not `flock -n`. With a non-blocking lock this interleaving leaves NOBODY
# merging: task A takes the lock and counts 1 marker (B has not written its yet); B writes
# its marker, fails the lock, and exits; A then finds its own count short and exits too.
# Both markers exist and no merge ever ran. Waiting instead means the second task counts
# the complete set. MERGE_OK is re-checked after acquisition so the wait cannot become a
# double merge.
exec 9>"$MARKER_DIR/.merge.lock"
flock 9
if [ -f "$MARKER_DIR/MERGE_OK" ]; then
  echo "another task already merged this round."
  exit 0
fi
n_ok=$(find "$MARKER_DIR" -name 'shard_*.SHARD_OK' -type f | wc -l)
if [ "$n_ok" -ne "$N_SHARDS" ]; then
  echo "shard $TASK done; $n_ok/$N_SHARDS markers present, not the merger"
  exit 0
fi

TABLE_ARGS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
  printf -v i3 "%03d" "$i"
  [ -s "$SHARD_DIR/status_$i3.json" ] || {
    echo "FATAL: marker set complete but $SHARD_DIR/status_$i3.json is missing/empty" >&2
    exit 2
  }
  TABLE_ARGS+=(--table "$SHARD_DIR/status_$i3.json")
done

PYTHONHASHSEED=0 python -m tbox_finder.mining.architecture_producer merge \
  "${TABLE_ARGS[@]}" --out "$MERGED_OUT"

# Read the artifact back through the CONSUMER, not the writer: the table is only useful if
# the round accepts it, and `load_status_map` re-derives `status` from `rows` — a different
# validator from the one that wrote it.
PYTHONHASHSEED=0 python -c "
import sys
from tbox_finder.mining.architecture_producer import load_status_map
print('merged table accepted by the round consumer:', len(load_status_map(sys.argv[1])), 'candidates')
" "$MERGED_OUT"

touch "$MARKER_DIR/MERGE_OK"
echo "merged -> $MERGED_OUT"
