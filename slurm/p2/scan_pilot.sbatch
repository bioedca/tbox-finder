#!/bin/bash
# P2-10c′-c-ii — the ρ-pilot scan: forward the 100 divergent-clade pilot genomes with the
# job-680 production Stage-1 checkpoint and count Stage-1 candidates/Mbp = ρ (ADR-0003 D6).
# (PRD §5–§7 window/throughput/scan, §9.1; ADR-0003 D6; ADR-0005 D3/A3; ADR-0002 A2/A4
#  backbone+kernels; CLAUDE.md §9.3.)
#
# Measures the ρ(τ, min_span, gap_merge) SURFACE (user decision 2026-07-22: "sweep ρ(τ), pin
# nothing") over the src/tbox_finder/infer/call.py provisional grid, plus the Caduceus
# throughput w = windows/sec/GPU (ADR-0003 D6). ρ is a MEASURED OPS NUMBER that sizes the
# 0.7–68 GB homolog-DB + negative-window fetch (P2-10c′); it pins NO ADR value and asserts NO
# scientific claim (ADR-0005 D3 freezes production locus values at the phase gate, not at a
# sizing pilot). The recall-favouring caller makes ρ a conservative UPPER bound on the fetch.
#
# ── SHAPE: embarrassingly parallel, NO DDP ──────────────────────────────────
# Genome scanning needs no gradient and no cross-rank sync, so this is NOT torchrun. The job
# takes K A4000s and launches ONE single-GPU `scan-shard` worker per GPU (worker i → cuda:i),
# each scanning a deterministic LPT-by-size slice of the 100 genomes and writing a partial to
# NODE-LOCAL /tmp; a final `reduce` globs the partials → the one git-tracked report. K is read
# from the allocation (`nvidia-smi -L`), so changing --gres is the only edit needed to rescale.
#
# WHERE:   cluster `gpu` partition, K=4 RTX A4000 (sm_86) on ONE node, env tbox-ml-dna
#          (ADR-0002 A4). No --nodelist (never pin; `one` is often down, cryosparc fragments
#          both nodes). 4 GPUs (not a whole node) packs into partial free capacity, so it
#          places far more reliably than an 8-GPU whole-node request — but still budget an
#          open-ended queue wait.
# INPUT:   staged by rsync (the ml env has no dvc; a checkout does not carry DVC data):
#            rsync -avzP data/processed/checkpoints/stage1_production/ two:~/tbox-finder/data/processed/checkpoints/stage1_production/
#            rsync -avzP data/interim/pilot_genomes/ two:~/tbox-finder/data/interim/pilot_genomes/
#          The manifest fetch report (data/processed/audits/pilot_fetch_report.json) is
#          GIT-TRACKED — it rides the checkout, no staging. This body verifies the checkpoint
#          against the sha256 in provenance.json (a `.dir` DVC hash cannot md5 a single file)
#          and the genome set against its count + FASTA magic, exit 2 on mismatch.
# SUBMIT:  from the repo root so relative --output/--error resolve. SYNC FIRST, STAGE SECOND,
#          SUBMIT THIRD — never in one `&&` chain:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-scan-pilot && git reset --hard origin/p2-scan-pilot"'
#            # …stage the checkpoint + genome dirs (see INPUT above)…
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/scan_pilot.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/scan_pilot.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b)
#          reports/p2/scan_pilot.DONE present, (c) data/processed/pilot/rho_pilot_report.json
#          present AND self-validating (reduce writes it ONLY if validate_report is clean — it
#          fails the process otherwise). ⚠️ the `.err` is EXPECTED NON-EMPTY on success
#          (trust_remote_code / selective_scan_cuda / HF chatter) — scan for real tracebacks,
#          DO NOT gate on zero bytes. A missing DONE is a §9.3 stop (surface it; no auto-resubmit).
# RETRIEVE: the git-tracked report + its provenance sidecar (plain scratch; `dvc add` is not
#          needed — the report is committed to git, not DVC):
#            rsync -avzP two:~/tbox-finder/data/processed/pilot/rho_pilot_report.json <local>/data/processed/pilot/
#            rsync -avzP two:~/tbox-finder/data/processed/pilot/rho_pilot_report.provenance.json <local>/data/processed/pilot/

#SBATCH --job-name=tbox-p2-10cpc-rho-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=reports/p2/scan_pilot_%j.out
#SBATCH --error=reports/p2/scan_pilot_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

GENOME_DIR="data/interim/pilot_genomes"
CKPT_DIR="data/processed/checkpoints/stage1_production"
CKPT="$CKPT_DIR/stage1.pt"
MANIFEST="data/processed/audits/pilot_fetch_report.json"
PILOT_DIR="data/processed/pilot"
REPORT="$PILOT_DIR/rho_pilot_report.json"
PROV="$PILOT_DIR/rho_pilot_report.provenance.json"
DONE="reports/p2/scan_pilot.DONE"
mkdir -p "$PILOT_DIR" reports/p2

# The sha256 of the job-680 production checkpoint (stage1.pt), from its provenance.json
# `outputs` — the per-FILE integrity anchor a `.dir` DVC hash cannot provide (see INPUT).
CKPT_SHA256="09931a223c3e670731a39b8ef0b0bb4bcb36a8ff9dcbee889f1715362b380940"
N_GENOMES_EXPECTED=100

# ── Singleton lock ───────────────────────────────────────────────────────────
exec 9>"reports/p2/.scan_pilot.lock"
if ! flock -n 9; then
  echo "SCAN_SKIP: another ρ-pilot job holds the lock — refusing to share outputs." >&2
  exit 1
fi

# ── Invalidate any stale DONE — RIGHT AFTER the lock, before anything that can exit ──
# DONE is the load-bearing §9.3 success signal (the .err is expected non-empty, so it is NOT
# gated on), and it is gitignored → a prior successful run's DONE (job 687 wrote one) SURVIVES
# the mandated `git reset --hard` cluster-sync into this run's tree. If this run then aborts at
# a staged-input assertion below (e.g. a NEW checkpoint whose sha256 != the pinned CKPT_SHA256
# → exit 2, or a partially-evicted genome dir under the tight HOME quota), the stale DONE + the
# hard-reset-restored committed report would present a clean green for a scan that never ran —
# ρ from the OLD checkpoint. Clearing here (only the lock-holding run reaches this; a lock-loser
# exits above WITHOUT clearing the winner's DONE) restores the invariant: DONE present ⇔ THIS
# run succeeded. Carries no jobid, so its mere presence must mean freshness.
rm -f "$DONE"

# ── Staged-input assertions (fail loud, before any GPU time) ──────────────────
# The checkpoint is content-verified against its committed sha256 (a wrong checkpoint = a
# wrong ρ, silently). The genome set is verified by count + FASTA magic. The manifest report
# is git-tracked, so its presence is a checkout invariant, but assert it anyway.
if [ ! -s "$CKPT" ]; then
  echo "FATAL: $CKPT missing. Stage the checkpoint dir from the laptop (see INPUT):" >&2
  echo "  rsync -avzP $CKPT_DIR/ two:~/tbox-finder/$CKPT_DIR/" >&2
  exit 2
fi
GOT_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"
if [ "$GOT_SHA" != "$CKPT_SHA256" ]; then
  echo "FATAL: $CKPT sha256 $GOT_SHA != committed $CKPT_SHA256 — wrong/stale checkpoint." >&2
  exit 2
fi
echo "staged-ok $CKPT sha256=$GOT_SHA"

N_FNA="$(find "$GENOME_DIR" -maxdepth 1 -name '*.fna' | wc -l)"
if [ "$N_FNA" -ne "$N_GENOMES_EXPECTED" ]; then
  echo "FATAL: found $N_FNA .fna under $GENOME_DIR, expected $N_GENOMES_EXPECTED. Stage the genomes:" >&2
  echo "  rsync -avzP $GENOME_DIR/ two:~/tbox-finder/$GENOME_DIR/" >&2
  exit 2
fi
# Every genome non-empty and a real FASTA (first byte '>') — catches a truncated/partial rsync.
while IFS= read -r f; do
  if [ ! -s "$f" ] || [ "$(head -c 1 "$f")" != ">" ]; then
    echo "FATAL: $f is empty or not FASTA (no '>' header) — re-stage $GENOME_DIR." >&2
    exit 2
  fi
done < <(find "$GENOME_DIR" -maxdepth 1 -name '*.fna')
echo "staged-ok $GENOME_DIR ($N_FNA genomes)"

if [ ! -s "$MANIFEST" ]; then
  echo "FATAL: $MANIFEST (git-tracked pilot-fetch report) missing from the checkout." >&2
  exit 2
fi

# ── RESTORE the tracked report to HEAD — idempotent re-runs (P2-10d′-c lesson) ─
# $REPORT becomes git-tracked once this step merges; a successful run overwrites it in place,
# so a re-run starts from a dirty tree there. RESTORE to the committed bytes (never `rm` — a
# deletion dirties the tree the same way). On the FIRST run it is untracked, so the check is a
# no-op. This touches ONE working-tree file, no history.
if git ls-files --error-unmatch "$REPORT" >/dev/null 2>&1; then
  git checkout HEAD -- "$REPORT"
fi
# (DONE was already cleared right after the lock — see above — so a failure between there and
# here cannot leave a stale success signal.)

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna

export CUDA_HOME="$CONDA_PREFIX"          # nvcc/deepspeed probe CUDA_HOME at import (ADR-0002 A4).
export PYTHONHASHSEED=0                    # determinism (§8.3); inference is deterministic anyway.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2                   # 16 cpus / 4 workers.

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD/shards"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.
export HF_HOME="$BUILD/hf"      # Caduceus weights stay OFF the 87%-full HOME quota (§9.2).

# ⚠️ `import torch` BEFORE selective_scan_cuda: the kernel is a C extension linked against
# torch's libc10.so; a bare `import selective_scan_cuda` dies "libc10.so: cannot open shared
# object file" even when the kernel is fine (this killed training job 567). The Mamba forward
# is a hard capability (ADR-0002 A2 C2) — assert it, don't hope.
python -c 'import torch; import selective_scan_cuda; print(f"selective_scan_cuda OK (torch {torch.__version__})")'

# ── Pre-warm the HF cache ONCE ───────────────────────────────────────────────
# The K workers each call from_pretrained; without a warm cache they race on K concurrent
# downloads of the same pinned revision into one HF_HOME.
PYTHONPATH=src python -c '
from tbox_finder.models.caduceus_backbone import REPO_ID, REVISION
from transformers import AutoModel
AutoModel.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)
print("hf cache warm", REPO_ID, REVISION)
'

# ── Scan: one single-GPU worker per allocated A4000 ──────────────────────────
# K = the number of CUDA-VISIBLE devices — what torch can address as cuda:0..cuda:K-1 — and
# worker i pins cuda:i. ⚠️ NOT `nvidia-smi -L | wc -l`: nvidia-smi lists ALL physical GPUs on
# the node (8 on `one`) regardless of the --gres allocation, while torch honours the
# SLURM-set CUDA_VISIBLE_DEVICES (the 4 this job was given). Job 685 died "invalid device
# ordinal" because nvidia-smi reported 8 → workers 4–7 pinned cuda:4..7 that torch could not
# see. torch.cuda.device_count() is the ground truth: cuda:0..K-1 are exactly the allocated
# GPUs. Deterministic LPT sharding means every worker computes the same partition.
K="$(PYTHONPATH=src python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$K" -lt 1 ]; then
  echo "FATAL: torch sees no CUDA devices (torch.cuda.device_count()==0)." >&2
  exit 2
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} → torch device_count K=$K"
echo "── scanning $N_FNA genomes across K=$K A4000 (one worker per GPU) ──"

pids=()
for ((i = 0; i < K; i++)); do
  (
    PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.infer.rho_pilot scan-shard \
      --genome-dir "$GENOME_DIR" --shard "$i" --n-shards "$K" --gpu-index "$i" \
      --checkpoint "$CKPT" --out "$BUILD/shards/shard_$i.json" \
      && touch "$BUILD/shards/shard_$i.OK"
  ) >"$BUILD/shards/shard_$i.log" 2>"$BUILD/shards/shard_$i.err" &
  pids+=($!)
done

# Wait for ALL workers (collect every exit, don't abort on the first failure).
set +e
for pid in "${pids[@]}"; do wait "$pid"; done
set -e

# Success is per-shard SHARD_OK presence, NOT .err emptiness (torch/HF chatter). A missing OK
# is a §9.3 stop — surface the shard's .err tail and exit without a DONE (no auto-resubmit).
missing=0
for ((i = 0; i < K; i++)); do
  if [ ! -f "$BUILD/shards/shard_$i.OK" ]; then
    echo "SCAN_FAIL: shard $i has no SHARD_OK. Tail of its .err:" >&2
    tail -n 20 "$BUILD/shards/shard_$i.err" >&2 || true
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "SCAN_FAIL: at least one shard did not complete — no report written (§9.3 stop)." >&2
  exit 1
fi

# ── Reduce: sum the partials → the certified ρ surface + w ────────────────────
# reduce_partials re-derives every headline from the per-genome evidence and refuses to write
# on any validation problem (coverage != manifest, ρ mis-derived, dead forward, empty
# permissive corner) — a broken scan leaves NO green artifact behind.
shopt -s nullglob
PARTIALS=("$BUILD"/shards/shard_*.json)
shopt -u nullglob
if [ "${#PARTIALS[@]}" -ne "$K" ]; then
  echo "SCAN_FAIL: expected $K partials, found ${#PARTIALS[@]}." >&2
  exit 1
fi
PYTHONPATH=src python -m tbox_finder.infer.rho_pilot reduce \
  --partials "${PARTIALS[@]}" \
  --manifest-report "$MANIFEST" \
  --checkpoint "$CKPT" \
  --report "$REPORT" --provenance "$PROV" \
  --accessed "$(date -u +%Y-%m-%d)" \
  --env-lock envs/ml-dna.conda-lock.yml

if [ ! -s "$REPORT" ] || [ ! -s "$PROV" ]; then
  echo "SCAN_FAIL: reduce did not write the report/provenance (it fails closed on a bad gate)." >&2
  exit 1
fi

touch "$DONE"
echo "RHO_PILOT_DONE job=${SLURM_JOB_ID:-none} report=$REPORT provenance=$PROV"
