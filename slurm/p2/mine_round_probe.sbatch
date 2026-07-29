#!/bin/bash
# P2-10e-msa-producer — A10 Phase-1 scan-throughput PROBE (§7 DECIDED 2026-07-28: "throughput probe first").
#
# WHY (job 793): the whole-8-GPU-node Phase-1 measurement (mine_round_measure.sbatch) TIMED OUT at its
# fixed 6 h wall — the 8-way GPU scan never finished, so it produced NO N₀ and NO throughput number, and
# its EXIT trap auto-cleaned the /tmp evidence on SIGTERM. The implied rate was ≤ ~24 win/s/GPU, ~3× below
# the ADR-0005 A9 estimate (67.68 win/s/GPU, job 687) — but "real scan is ~3× slower" and "a shard stalled"
# read identically with no persistent per-shard progress. This PROBE MEASURES the scan throughput first, on
# ONE GPU, under a hard 45-min wall, over a small deterministic genome slice, writing win/s/GPU + the partial
# N₀ rate (FPs/window) to a PERSISTENT reports/ path that survives a wall-clock kill. Its number sizes the
# hardened full Phase-1 measurement (wall, decoupling, per-shard progress). It measures ONLY — nothing here
# runs the producer (no nhmmer/mlocarna/R-scape, no locarna/rscape envs), mines, retrains, or flips a flag.
#
# ── SHAPE: LEG (a) scan only, ONE A4000 ─────────────────────────────────────────────────────────────
#   SCAN (tbox-ml-dna): scan-shard over the first --limit accession-sorted a2-admissible genomes, stopping
#   cleanly after --max-windows windows (a self-terminating throughput sample that finishes well inside the
#   45-min wall even at the pessimistic ~24 win/s). The throughput snapshot is flushed to a persistent path
#   at start + per genome + every --flush-every-windows windows + once on exit — so even a stall→SIGTERM at
#   the wall leaves the last measurement (with the in-progress genome pinned) on disk.
# §9.2 forbids `compute`(=zero) and pinning a node (`one` is often down); no --nodelist.
#
# ── SUBMIT (SYNC FIRST, STAGE SECOND — incl. the LFS split table — SUBMIT THIRD, §9.3) ──────────────
#   ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && git checkout p2-10e-msa-producer && \
#     git reset --hard origin/p2-10e-msa-producer"'
#   # ⚠ the cluster has NO git-lfs: `reset --hard` leaves split_assignments.parquet a 132-byte pointer
#   #   ([ -s ] passes; pd.read_parquet then reads 0 rows → admissible-accessions returns 0). Stage the
#   #   REAL split table by rsync from the laptop AFTER the reset (which reverts it to the pointer):
#   #     rsync -avzP data/processed/splits/split_assignments.parquet two:~/tbox-finder/data/processed/splits/
#   ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/mine_round_probe.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/mine_round_probe.sbatch"'
# INPUT (staged by rsync — the ml/cluster checkout carries no DVC data; the probe needs NO homolog DB):
#   rsync -avzP data/processed/checkpoints/stage1_production/ two:~/tbox-finder/data/processed/checkpoints/stage1_production/
#   rsync -avzP data/interim/production_genomes/ two:~/tbox-finder/data/interim/production_genomes/
#   (the a2 host-order table is plain-git — rides the checkout; the split table is LFS → rsync it, above.)
# VERIFY (artifact-based; sacct off, §9.3 — a PROBE, so a wall-clock timeout that captured throughput is a
#   SUCCESS, not a failure): (a) `squeue -j <jobid>` empty; (b) reports/p2/mine_round_probe.json present with
#   windows_scanned > 0 AND windows_per_s != null (the measurement). The .err is EITHER 0-byte (finished the
#   window cap → mine_round_probe.DONE present, "complete": true) OR contains ONLY the SLURM time-limit notice
#   (a stall → "complete": false + in_progress_accession pins the offending genome). Any OTHER .err content, or
#   windows_per_s == null → §9.3 stop (surface it; NO auto-resubmit).
# RETRIEVE (LOCAL follow-up — the full-measurement sizing session):
#   rsync -avzP two:~/tbox-finder/reports/p2/mine_round_probe.json <local>/reports/p2/
#   then size + harden mine_round_measure.sbatch (wall, per-shard progress, preserve-partials) from win/s/GPU.

#SBATCH --job-name=tbox-p2-10e-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=reports/p2/mine_round_probe_%j.out
#SBATCH --error=reports/p2/mine_round_probe_%j.err
# No --account/--qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc + conda activate hooks read unset vars.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
PROGRESS_OUT="$REPO/reports/p2/mine_round_probe.json"
DONE="$REPO/reports/p2/mine_round_probe.DONE"
# Clear the stale completion marker FIRST (before cd/mkdir or anything that can exit under set -e) so a
# DONE surviving a `git reset --hard` cannot green a probe that never ran.
rm -f "$DONE"
cd "$REPO"
mkdir -p reports/p2

CKPT="data/processed/checkpoints/stage1_production/stage1.pt"   # the A7 cold-start round-0 parent
GENOME_DIR="data/interim/production_genomes"

N_PROBE="20"          # first-N accession-sorted a2 genomes (deterministic slice)
MAX_WINDOWS="30000"   # self-terminating window cap: ~21 min @ 24 win/s, ~7.5 min @ 67 win/s — clears 45 min
FLUSH="200"           # flush the persistent throughput snapshot every 200 windows (~8 s @ 24 win/s)

# NOTE: name this JOB_SCRATCH, never BUILD/HOST — `conda activate tbox-ml-dna` (the compiler toolchain)
# exports BUILD=HOST=x86_64-conda-linux-gnu, which would clobber a $BUILD scratch path the instant the env
# activates (job 789 died on exactly this). $PROGRESS_OUT lives in persistent reports/, NOT here.
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT   # cleans node-local scratch only; the persistent report is untouched.

# ── Staged-input assertions (fail loud, before any GPU time) ──────────────────────────────────────
[ -s "$CKPT" ] || { echo "FATAL: $CKPT missing — stage the round-0 checkpoint (rsync)." >&2; exit 2; }
N_GENOMES="$(find "$GENOME_DIR" -maxdepth 1 -name '*.fna' 2>/dev/null | wc -l)"
[ "$N_GENOMES" -ge 1 ] || { echo "FATAL: $GENOME_DIR has no .fna — stage the fetched genomes." >&2; exit 2; }

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
command -v python >/dev/null || { echo "FATAL: tbox-ml-dna not active?" >&2; exit 2; }

# ── The a2 admissible substrate (same rule the §8.2 CI reads); first --limit are the deterministic slice ──
PYTHONPATH=src python -m tbox_finder.mining.mine_round admissible-accessions --out "$JOB_SCRATCH/accessions.txt"
# Fail loud on an empty admissible set: the cluster has no git-lfs, so a `git reset --hard` can leave
# split_assignments.parquet a 132-byte LFS pointer that reads 0 rows ([[git-lfs-pointers-in-ci]]);
# without this the scan finds 0 windows and the probe greens (DONE, exit 0) having measured nothing.
n_acc="$(wc -l < "$JOB_SCRATCH/accessions.txt")"
[ "$n_acc" -ge 1 ] || { echo "FATAL: 0 admissible accessions — is split_assignments.parquet an LFS pointer?" >&2; exit 2; }

# ── LEG (a) scan → persistent throughput snapshot (the measurement) + an FP manifest (byproduct) ──────
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.mine_round scan-shard \
  --checkpoint "$CKPT" --accessions "$JOB_SCRATCH/accessions.txt" --limit "$N_PROBE" \
  --genome-dir "$GENOME_DIR" --out-manifest "$JOB_SCRATCH/probe_fp.json" \
  --progress-out "$PROGRESS_OUT" --flush-every-windows "$FLUSH" --max-windows "$MAX_WINDOWS"

# Only reached on a CLEAN finish (the window cap, inside the wall). A timeout exits above under set -e with
# the persistent snapshot already on disk ("complete": false) — the intended fallback, not an error.
PYTHONPATH=src python - "$PROGRESS_OUT" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(
    f"PROBE complete={r['complete']} note={r['note']} windows={r['windows_scanned']} "
    f"win/s/GPU={r['windows_per_s']} FPs/window={r['candidates_per_window']} "
    f"genomes_completed={r['genomes_completed']} wall_s={r['wall_s']:.1f}"
)
PYEOF

touch "$DONE"
echo "A10 Phase-1 scan-throughput probe complete → $PROGRESS_OUT (DONE marker written)."
