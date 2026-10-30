#!/bin/bash
# P3-16″-hardneg — ADR-0005 A17 Pin 4's ONE census extension, job 1 of 3: the census pool + mint.
# (PRD §9.1 class 4; ADR-0005 A17 Pins 1-4, A15 Pins 1-2; CLAUDE.md §9.3.)
#
# WHY THIS RUNS: the draw's power gate decided `extend_census` — P = 0 of 90,243 graded negatives
# reached a twin Stage-2 payload at τ 0.5 (N_min 20; data/interim/p3_16/hardneg_power.json). A17
# Pin 4 then pre-registers exactly one extension: every eligible window the draw did NOT take,
# through the same Pins 1-3, decided on P alone, with no Stage-2 score or AUPRC before it.
#
# THE THREE JOBS (one §9.3 ack covers the chain; each is `afterok` on the one before):
#   1. THIS FILE   CPU only.  census prep (gap + fold screen) -> cmsearch content screen ->
#                  report (D4 partition) -> the 1,000-window --max audit -> the committed census
#                  pool report -> the census benchmark (draw + census windows + v2's positives).
#   2. hardneg_census_stage1.sbatch  8 GPUs: Stage-1, both arms, all contigs, 8 shards + merge.
#   3. hardneg_census_score.sbatch   1 GPU: payloads -> census power -> Stage-2 -> tables.
# The fold + report run LOCALLY afterwards (they hash the two Stage-1 archives, which come back
# by rsync for `dvc add`).
#
# WHERE IT RUNS — NOT ~/tbox-finder. HOME is quota-tight (a 7.5 GB conda env once tripped it) and
# job 2 writes ~46 GB of float32 window logits (3.3 GB uncompressed per 100k contigs, measured on
# the draw). The chain runs from a sacrificial checkout on the beegfs scratch that already holds
# the DVC remote, and every path below stays repo-relative to it:
#     WORK=/onedata/mondragonlab/ecc1695/tbox-finder/census
# Staging (before the ack; no compute): `git clone` the branch into $WORK, copy the gitignored
# inputs from ~/tbox-finder, rsync the laptop-only ones (production_annotations, the draw's pool
# interim, benchmark_v2.json, hardneg_power.json), then LEG 0 below checks every one of them
# against slurm/p3/hardneg_census_inputs.sha256 — the laptop bytes the draw itself ran on.
#
# SIZING (measured on the draw, laptop, 24 threads): prep 301 s per 100k drawn; content screen
# 2,004 s wall at 8 x 3 cpus = 13.4 CPU-h per 98,915 survivors => ~98 CPU-h for the census's
# ~724k (A17 Pin 4 says ~100); the --max audit 1,965 s for 1,000 windows. At 63 cpus: ~0.5 h
# prep + ~1.6 h screen + ~0.6 h audit + ~0.3 h mint. --time leaves 2x headroom.
#
# RESUMABLE: each leg touches "$MARKERS/<leg>.OK" only on success, and a re-submit skips a leg
# whose marker exists. Nothing here deletes anything (tests/unit/test_sbatch_rm_targets.py):
# re-running a leg on purpose means removing its marker BY HAND.
#
# SUBMIT (from $WORK, so --output/--error and SLURM_SUBMIT_DIR are the checkout):
#   ssh two 'bash -ls' <<'EOF'
#   cd /onedata/mondragonlab/ecc1695/tbox-finder/census
#   J1=$(sbatch --parsable slurm/p3/hardneg_census_pool.sbatch)
#   J2=$(sbatch --parsable --dependency=afterok:$J1 slurm/p3/hardneg_census_stage1.sbatch)
#   J3=$(sbatch --parsable --dependency=afterok:$J2 slurm/p3/hardneg_census_score.sbatch)
#   echo "$J1 $J2 $J3"
#   EOF
# PREFLIGHT: sbatch --test-only on each of the three files, from $WORK.
# VERIFY (artifact-based; sacct is off): squeue empty, "$MARKERS/pool.DONE" present, the .err
#   holds no traceback, and reports/p3/hardneg_pool_census.json exists (audit-report writes it
#   ONLY when hardneg_pool.validate_report returns [], else it writes .invalid.json and exits 3).
# RETRIEVE: rsync -avP two:$WORK/{reports/p3/hardneg_pool_census*,data/interim/p3_16/...} — the
#   whole list is in hardneg_census_score.sbatch's header, after job 3.

#SBATCH --job-name=tbox-p3-16pp-census-pool
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=reports/p3/hardneg_census_pool_%j.out
#SBATCH --error=reports/p3/hardneg_census_pool_%j.err
# No --gres (CPU legs only), no --account/--qos, no --nodelist.

set -eo pipefail   # NOT -u: the conda activate hooks read unset variables.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

P=data/interim/p3_16
DRAW_DIR="$P/hardneg_pool"
CENSUS_DIR="$P/hardneg_pool_census"
MARKERS="$P/hardneg_census_markers"
BENCHMARK="$P/hardneg_census_benchmark.json"
DRAW_REPORT=reports/p3/hardneg_pool.json
CENSUS_REPORT=reports/p3/hardneg_pool_census.json
CENSUS_IDS=reports/p3/hardneg_pool_census_ids.tsv
INPUTS=slurm/p3/hardneg_census_inputs.sha256
mkdir -p "$MARKERS" logs

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
export PYTHONHASHSEED=0
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# leg NAME ENV CMD... — run CMD in conda env ENV unless NAME's marker exists; mark on success.
leg() {
  local name="$1" env="$2"
  shift 2
  if [ -e "$MARKERS/$name.OK" ]; then
    echo "== $name: marker present from an earlier attempt, skipping"
    return 0
  fi
  echo "== $name ($env): start $(date -u +%FT%TZ)"
  conda activate "$env"
  PYTHONPATH=src "$@"
  conda deactivate
  touch "$MARKERS/$name.OK"
  echo "== $name: done $(date -u +%FT%TZ)"
}

# ── LEG 0: every staged input is the laptop's byte-for-byte ─────────────────────────────────────
# A git-LFS pointer or a truncated copy is non-empty and passes `[ -s ]`; a sha256 does not.
[ -s "$INPUTS" ] || { echo "FATAL: $INPUTS missing — wrong checkout?" >&2; exit 2; }
sha256sum --check --quiet --strict "$INPUTS" || {
  echo "FATAL: a staged input differs from the laptop bytes the draw ran on (see above)." >&2
  echo "Re-stage it; the census must go through Pins 1-3 on the draw's own inputs." >&2
  exit 2; }
echo "== inputs: $(wc -l < "$INPUTS") files match $INPUTS"

# ── LEGS 1-5: the census pool, through A17 Pins 1-2 unchanged ───────────────────────────────────
leg prep tbox-data python -m tbox_finder.integration.hardneg_pool prep \
  --mode census --out-dir "$CENSUS_DIR" --shards 32
leg search tbox-infernal python -m tbox_finder.integration.hardneg_pool search \
  --out-dir "$CENSUS_DIR" --jobs 21 --cpu 3
leg report tbox-data python -m tbox_finder.integration.hardneg_pool report \
  --out-dir "$CENSUS_DIR"
leg audit-search tbox-infernal python -m tbox_finder.integration.hardneg_pool audit-search \
  --out-dir "$CENSUS_DIR" --jobs 2 --cpu 31
# Exit 3 (and a .invalid.json, never the committed path) when validate_report is not [].
leg audit-report tbox-data python -m tbox_finder.integration.hardneg_pool audit-report \
  --out-dir "$CENSUS_DIR" --out "$CENSUS_REPORT" --ids-out "$CENSUS_IDS"

# ── LEG 6: the census benchmark = the draw's windows + the census's + v2's positives ────────────
# Both pool reports are re-validated against this checkout (RNG replay + annex re-read) by the
# mint itself before a contig is written.
leg mint tbox-data python scripts/mint_p3_16_benchmark.py \
  --hardneg "$DRAW_REPORT" --hardneg-items "$DRAW_DIR/graded.json" "$DRAW_DIR/beside.json" \
  --hardneg-census "$CENSUS_REPORT" \
  --hardneg-census-items "$CENSUS_DIR/graded.json" "$CENSUS_DIR/beside.json" \
  --hardneg-positives "$P/benchmark_v2.json" \
  --split-table data/processed/splits/split_assignments.parquet \
  --context data/interim/flank_context/context_v0.parquet \
  --labels data/processed/labels/labels_v0.parquet \
  --host-orders data/processed/mining/production_host_orders_v0.parquet \
  --out "$BENCHMARK"

touch "$MARKERS/pool.DONE"
echo "census pool + benchmark complete — job 2 (Stage-1) may start."
