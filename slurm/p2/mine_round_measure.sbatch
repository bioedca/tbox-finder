#!/bin/bash
# P2-10e-msa-producer — A10 Phase-1 MEASUREMENT (round-0 scan → real N₀ + a K-sample producer run).
#
# A10 Pin 1 is measure-first: before the per-round producer budget is pinned (Phase 2) and before
# mine_round.MSA_SUPPLY_AVAILABLE is flipped, MEASURE the two unknowns A9 under-counted —
#   (i)  N₀, the real Stage-1 false-positive count over the a2 negative substrate; and
#   (ii) the per-candidate producer wall (nhmmer / mlocarna / R-scape) + homolog depth,
# on a bounded, deterministic K≥50 sample. The report is the input to the Phase-2 array-width pin
# (A10 Pin 2), not the pin itself. Nothing here mines, retrains, or flips a flag (§10.3).
#
# ── SHAPE: ONE whole-8-GPU-node job; the scan is GPU (8-way sharded), the producer is CPU (3 envs) ──
#   LEG a  SCAN     (tbox-ml-dna) : 8 background scan-shard procs, one A4000 each (CUDA_VISIBLE_DEVICES),
#                                   over the 660 a2-admissible genomes → per-GPU FP manifests → collect → N₀.
#   SAMPLE          (tbox-ml-dna) : select-sample the first K≥50 accession-sorted FPs (deterministic).
#   PRODUCE Stage 1 (tbox-homology): resolve+search each sample candidate → candidate+homolog FASTA.
#   PRODUCE Stage 2 (tbox-locarna) : mlocarna align → Pfam Stockholm (+ #=GC SS_cons).
#   PRODUCE Stage 3 (tbox-rscape)  : R-scape covariation status → per-candidate wall + depth table.
#   REPORT          (any env)      : N₀ + wall/depth distributions + extrapolated envelope.
# The whole node is held for the CPU producer phase (a one-off measurement, ~2 h); the RUN decouples
# that into its own CPU array (mine_round_producer.sbatch, A10 Pin 2). §9.2 forbids `compute`(=zero)
# and pinning a node (`one` is often down); no --nodelist.
#
# ── SIZED + HARDENED from the job-806 probe (23.87 win/s/GPU; reports/p2/mine_round_probe.json) ─────
#   WALL: the probe measured 23.87 win/s/GPU (robust across genomes), so the heaviest a2 shard
#   (524,959 windows under `split -n l/8`) needs 524,959/23.87 ≈ 6.1 h of scan, and the CPU producer
#   phase adds ~2 h → ~8.1 h total. `--time=12:00:00` clears that with margin (the gpu partition
#   TIMELIMIT is *infinite*, so a longer wall is a free soft ceiling; job 793 TIMED OUT at the old
#   fixed 6 h, exactly the 6.1 h heaviest-shard scan).
#   PER-SHARD PROGRESS: each of the 8 scan-shard legs writes win/s/GPU + the partial FP-rate to a
#   PERSISTENT reports/p2/mine_round_measure_shard<i>.json (NOT /tmp) via --progress-out, flushed
#   every 500 windows + on the wall-clock SIGTERM (preserve-partials) — so a second timeout would
#   still pin which shard/genome was mid-scan, the exact diagnostic job 793 lacked.
#   ⚠ SCAN THROUGHPUT NOT OPTIMISED HERE (deliberate): the ~2.8× gap vs the A9 67.68 win/s/GPU
#   estimate is the mining scan forwarding one 1024-nt window per model call (batch=1;
#   mine_round._scan_one_window → scan.scan_sequence, one window → one forward). Batching across
#   substrate windows is result-preserving in principle but is a numerics-sensitive scan-hot-path
#   change with NO validation path this session (CI installs no torch; no local torch+pytest env;
#   WSL2 CUDA is unreliable for RNA inference) — so it is DEFERRED to its own validated step rather
#   than bundled into the run that produces the headline N₀ (§10.3). This measurement runs the exact
#   scan the probe (job 806) validated; the 12 h wall accommodates it at the measured 23.87 win/s.
#
# ── ⛔ CLUSTER-ENV PREREQUISITE (generalises [[cluster-has-no-tbox-data-env]]) ─────────────────────
# tbox-ml-dna + tbox-homology exist; tbox-locarna + tbox-rscape may not. Create both ONCE from the
# pinned locks (CPU-only, ~1-3 GB each; watch the tight HOME quota) — part of the §9.3 submit-ack, on
# `submit`, NOT before it:
#   ssh two 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate condalock && \
#     conda-lock install --name tbox-locarna envs/locarna.conda-lock.yml"'
#   ssh two 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate condalock && \
#     conda-lock install --name tbox-rscape  envs/rscape.conda-lock.yml"'
#
# ── SUBMIT (SYNC FIRST, STAGE SECOND — incl. the LFS split table — SUBMIT THIRD, §9.3) ──────────────
#   ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && git checkout p2-10e-msa-producer && \
#     git reset --hard origin/p2-10e-msa-producer"'
#   # ⚠ the cluster has NO git-lfs: `reset --hard` leaves split_assignments.parquet a 132-byte pointer
#   #   ([ -s ] passes; pd.read_parquet then reads 0 rows → the a2 blacklist guard refuses). Stage the
#   #   REAL split table by rsync from the laptop AFTER the reset (which reverts it to the pointer):
#   #     rsync -avzP data/processed/splits/split_assignments.parquet two:~/tbox-finder/data/processed/splits/
#   # …create the two envs (above); stage the checkpoint + genome dirs + homolog DB (see INPUT)…
#   ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/mine_round_measure.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/mine_round_measure.sbatch"'
# INPUT (staged by rsync — the ml/cluster checkout carries no DVC data):
#   rsync -avzP data/processed/checkpoints/stage1_production/ two:~/tbox-finder/data/processed/checkpoints/stage1_production/
#   rsync -avzP data/interim/production_genomes/ two:~/tbox-finder/data/interim/production_genomes/
#   rsync -avzP data/interim/homolog_db/         two:~/tbox-finder/data/interim/homolog_db/
#   (the a2 host-order table is plain-git — rides the checkout; the split table is LFS → rsync it, above.)
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty; (b) DONE marker
#   reports/p2/mine_round_measure.DONE present; (c) reports/p2/mine_round_measure.json present with
#   n0_false_positives > 0 and k_sample >= 50. Any missing → §9.3 stop (surface it; NO auto-resubmit).
#   ON A TIMEOUT/STALL (no DONE): the persistent reports/p2/mine_round_measure_shard<i>.json carry
#   each shard's last win/s/GPU + FP-rate + in_progress_accession (`complete:false`) — read them to
#   tell a slow scan (all shards progressing, near 23.87 win/s) from a stalled genome (one shard's
#   in_progress_accession frozen) before deciding whether to raise the wall or fix throughput.
# RETRIEVE (LOCAL follow-up — the Phase-2 pin session; the cluster checkout has no dvc):
#   rsync -avzP two:~/tbox-finder/reports/p2/mine_round_measure.json <local>/reports/p2/
#   rsync -avzP two:~/tbox-finder/data/processed/mining/round0_fp_manifest.json <local>/data/processed/mining/
#   then pin the A10 Phase-2 addendum (array width + budget) → flip MSA_SUPPLY_AVAILABLE → the N=4 RUN.

#SBATCH --job-name=tbox-p2-10e-measure
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=reports/p2/mine_round_measure_%j.out
#SBATCH --error=reports/p2/mine_round_measure_%j.err
# No --account/--qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc + conda activate hooks read unset vars.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
DONE="$REPO/reports/p2/mine_round_measure.DONE"
# Clear the stale completion marker FIRST (before cd/mkdir or anything that can exit under set -e) so a
# DONE surviving a `git reset --hard` cannot green a measurement that never ran.
rm -f "$DONE"
cd "$REPO"
mkdir -p reports/p2 data/processed/mining

CKPT="data/processed/checkpoints/stage1_production/stage1.pt"   # the A7 cold-start round-0 parent
DB_DIR="data/interim/homolog_db"
BLAST_PREFIX="$DB_DIR/target"
FM_INDEX="$DB_DIR/target.fm"
GENOME_DIR="data/interim/production_genomes"
MEASURE_REPORT="reports/p2/mine_round_measure.json"
FP_MANIFEST_OUT="data/processed/mining/round0_fp_manifest.json"

# ── CONTROL cutoffs — the certified nhmmer config (P2-10e-msa job 766); D17 thresholds P6-frozen, ──
# value at the phase gate — these provisional values carry no methodological weight (homolog_msa D17).
CTRL_ENGINE="nhmmer"
CTRL_EVALUE="100"
CTRL_MAX_TARGET_SEQS="500"
CTRL_MIN_PIDENT="40"   # ignored on the nhmmer path; kept for the CLI-required arg
CTRL_MIN_COV="0.5"
K_SAMPLE="50"          # A10 Pin 1 bounded sample (K >= 50); accession-sorted, deterministic

# NOTE: name this JOB_SCRATCH, never BUILD/HOST — `conda activate tbox-ml-dna` (the compiler
# toolchain) exports BUILD=HOST=x86_64-conda-linux-gnu, which would clobber a $BUILD scratch path
# the instant LEG a's env activates (job 789 died here: $BUILD/accessions.txt → a bad relative path).
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH" "$JOB_SCRATCH/work"
trap 'rm -rf "$JOB_SCRATCH"' EXIT   # clean node-local scratch on every exit path; touches only $JOB_SCRATCH.

# ── Staged-input assertions (fail loud, before any GPU/tool time) ─────────────────────────────────
[ -s "$CKPT" ]      || { echo "FATAL: $CKPT missing — stage the round-0 checkpoint (rsync)." >&2; exit 2; }
[ -s "$FM_INDEX" ]  || { echo "FATAL: $FM_INDEX missing — stage the P2-10c′ homolog DB (rsync/dvc)." >&2; exit 2; }
ls "$BLAST_PREFIX".n* >/dev/null 2>&1 || { echo "FATAL: BLAST DB $BLAST_PREFIX.n* missing." >&2; exit 2; }
N_GENOMES="$(find "$GENOME_DIR" -maxdepth 1 -name '*.fna' 2>/dev/null | wc -l)"
[ "$N_GENOMES" -ge 1 ] || { echo "FATAL: $GENOME_DIR has no .fna — stage the 2,500 fetched genomes." >&2; exit 2; }

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ── LEG a: 8-way GPU-sharded scan → per-GPU FP manifests → collect → N₀ ───────────────────────────
conda activate tbox-ml-dna
export CUDA_HOME="$CONDA_PREFIX"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
command -v python >/dev/null || { echo "FATAL: tbox-ml-dna not active?" >&2; exit 2; }

PYTHONPATH=src python -m tbox_finder.mining.mine_round admissible-accessions --out "$JOB_SCRATCH/accessions.txt"
# Split the 660 admissible accessions into 8 line-balanced slices (slice_00.txt … slice_07.txt).
split -n l/8 --numeric-suffixes=0 --additional-suffix=.txt "$JOB_SCRATCH/accessions.txt" "$JOB_SCRATCH/slice_"

# Persistent per-shard throughput progress lives in reports/ (NOT $JOB_SCRATCH) so it survives the
# wall-clock SIGTERM + the EXIT trap: --progress-out makes scan-shard flush win/s/GPU + the partial
# FP-rate at start + per genome + every --flush-every-windows windows, and installs a SIGTERM→
# SystemExit handler so the wall-kill runs run_measured_scan's finally-flush (preserve-partials —
# job 793's gap: it cleaned its /tmp evidence on timeout with no persistent per-shard progress).
# The FP manifest ($JOB_SCRATCH/fp_$i.json) is node-local; leg (b) `collect` persists the merged N₀.
pids=()
for i in $(seq 0 7); do
  slice="$(printf "%s/slice_%02d.txt" "$JOB_SCRATCH" "$i")"
  [ -s "$slice" ] || continue   # fewer than 8 non-empty slices if the admissible set ever shrinks
  shard_progress="$(printf "reports/p2/mine_round_measure_shard%d.json" "$i")"
  CUDA_VISIBLE_DEVICES="$i" PYTHONHASHSEED=0 PYTHONPATH=src \
    python -m tbox_finder.mining.mine_round scan-shard \
      --checkpoint "$CKPT" --accessions "$slice" --genome-dir "$GENOME_DIR" \
      --out-manifest "$JOB_SCRATCH/fp_$i.json" \
      --progress-out "$shard_progress" --flush-every-windows 500 \
      > "$JOB_SCRATCH/scan_$i.log" 2>&1 &
  pids+=("$!")
done
scan_fail=0
for p in "${pids[@]}"; do wait "$p" || scan_fail=1; done
if [ "$scan_fail" -ne 0 ]; then
  echo "FATAL: a scan shard failed — logs follow" >&2; cat "$JOB_SCRATCH"/scan_*.log >&2; exit 3
fi

PYTHONPATH=src python -m tbox_finder.mining.mine_round collect \
  --manifests "$JOB_SCRATCH"/fp_*.json --out "$FP_MANIFEST_OUT"
echo "LEG a: collected round-0 FP manifest → $FP_MANIFEST_OUT"

# ── SAMPLE: deterministic first-K accession-sorted FPs ────────────────────────────────────────────
PYTHONPATH=src python -m tbox_finder.mining.covariation_producer select-sample \
  --manifest "$FP_MANIFEST_OUT" --k "$K_SAMPLE" --out "$JOB_SCRATCH/sample.json"

# ── PRODUCE Stage 1: SEARCH (tbox-homology) ───────────────────────────────────────────────────────
conda activate tbox-homology
for t in blastn nhmmer; do command -v "$t" >/dev/null || { echo "FATAL: $t not on PATH — tbox-homology?" >&2; exit 2; }; done
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer search-shard \
  --shard "$JOB_SCRATCH/sample.json" --workroot "$JOB_SCRATCH/work" --engine "$CTRL_ENGINE" \
  --evalue "$CTRL_EVALUE" --max-target-seqs "$CTRL_MAX_TARGET_SEQS" \
  --min-pident "$CTRL_MIN_PIDENT" --min-cov "$CTRL_MIN_COV" \
  --db-prefix "$BLAST_PREFIX" --fm-index "$FM_INDEX" --genome-dir "$GENOME_DIR"

# ── PRODUCE Stage 2: ALIGN (tbox-locarna) ─────────────────────────────────────────────────────────
conda activate tbox-locarna
command -v mlocarna >/dev/null || { echo "FATAL: mlocarna not on PATH — tbox-locarna not created?" >&2; exit 2; }
MLOCARNA_VERSION="$(mlocarna --version 2>&1 || true)"
echo "$MLOCARNA_VERSION" | grep -q "LocARNA 2.0.1" || { echo "FATAL: mlocarna != LocARNA 2.0.1 (ADR-0002 A13): $MLOCARNA_VERSION" >&2; exit 2; }
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer align-shard \
  --shard "$JOB_SCRATCH/sample.json" --workroot "$JOB_SCRATCH/work" --cpu 2

# ── PRODUCE Stage 3: SCORE (tbox-rscape) ──────────────────────────────────────────────────────────
conda activate tbox-rscape
command -v R-scape >/dev/null || { echo "FATAL: R-scape not on PATH — tbox-rscape not created?" >&2; exit 2; }
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer score-shard \
  --shard "$JOB_SCRATCH/sample.json" --workroot "$JOB_SCRATCH/work" --out-table "$JOB_SCRATCH/sample_status.json"

# ── REPORT: N₀ + per-stage wall + depth + extrapolated envelope (A10 Pin 1) ───────────────────────
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.covariation_producer measure-report \
  --manifest "$FP_MANIFEST_OUT" --status-table "$JOB_SCRATCH/sample_status.json" --out "$MEASURE_REPORT"

echo "A10 Phase-1 measurement complete → $MEASURE_REPORT (N₀ manifest → $FP_MANIFEST_OUT)"
touch "$DONE"
echo "DONE marker written — run the LOCAL retrieval + the A10 Phase-2 pin (see header)."
