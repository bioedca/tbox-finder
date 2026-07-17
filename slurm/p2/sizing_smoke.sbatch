#!/bin/bash
# P2-05 — Stage-1 training sizing smoke: fix the footprint before the full run
# (PRD §10.3/§11/§15; ADR-0002 D2/A2-C2/A4/A7/D8; ADR-0003 D1/D2/D7; CLAUDE.md §9.3).
#
# Sweeps the real P2-04 entrypoint over a configuration matrix on real A4000s and
# aggregates the per-point reports into reports/p2/sizing_smoke.json: per-GPU VRAM margin
# vs the PRD §10.3 16 GB budget, windows/sec/GPU, DDP scaling efficiency, and the
# gradient-checkpointing throughput cost P2-04 measured the VRAM half of but deliberately
# left untimed. This is a FOOTPRINT measurement, NOT a T-box science result, NOT a
# convergence claim, and NOT a GATE-4 grade (§10.3): every point trains a seeded slice for
# a handful of steps. `is_science=false`, `gate4_graded=false` in the report.
#
# ⚠️ IT DOES NOT FREEZE A BUDGET. ADR-0003 D7 makes the training/sweep GPU-hour number an
#    ADR-sign-off change (§7 item 2). The extrapolation lands under
#    `illustrative_extrapolation` with advisory_only=true / binding=false / binding_gate
#    naming D7 (the P1-14 precedent), and sizing.py's validator REJECTS a report claiming
#    otherwise. The freeze is a separate step with user sign-off.
#
# GATE (imp.md): VRAM < 16 GB with the chosen config + artifact-based exit. Reported as the
#    MARGIN, not the boolean: on a 15.6 GiB card a >16 GiB peak is unreachable (the
#    allocator OOMs first), so the boolean is near-unfailable and means little — P1-16's
#    honest-reading precedent. A gate miss is a §7 stop-and-ask, never a loosened threshold.
#
# WHERE:   cluster `gpu` partition, EIGHT RTX A4000 (sm_86) on one node. NOT the laptop
#          (an 8 GB RTX 4060 under WSL2 is not the 16 GB A4000 the §10.3 claim is about,
#          and spills to host memory instead of OOMing), NOT CI (no torch).
#          env tbox-ml-dna (ADR-0002 A4). est-wall ~0.35 h (16 points, 35 steps each);
#          --time=01:00:00 is ~3x headroom.
#          ⚠️ 8 GPUs, not imp.md's 2: imp.md's own Outputs ask for "DDP×8 scaling
#          efficiency" and PRD §10.3 pins "DDP×8 for throughput" — unmeasurable on 2.
#          ADR-0003 D1 parameterises `--gres=gpu:a4000:<N>` with "N ≤ 8", so 8 is inside it.
# INPUT:   ALL THREE corpus parquets are STAGED by rsync, NOT dvc-pulled and NOT carried by
#          the checkout. The ml envs have no dvc (the cluster's only dvc lives in an
#          unrelated tboxevo-infernal env; using it would be an ADR-0002 D8 ad-hoc-env
#          violation), and context_v0.parquet has never been `dvc push`ed (§5.2 defers push
#          to the phase-exit gate) so it is absent from the cluster remote anyway.
#          ⚠️ split_assignments.parquet is git-LFS (.gitattributes) and THIS CLUSTER HAS NO
#          git-lfs — a fresh checkout leaves a 132-byte POINTER, which is non-empty and so
#          passes a naive existence check. It must be rsynced too. Stage before submitting:
#            rsync -avzP data/interim/flank_context/context_v0.parquet \
#              two:~/tbox-finder/data/interim/flank_context/
#            rsync -avzP data/processed/labels/labels_v0.parquet \
#              two:~/tbox-finder/data/processed/labels/
#            rsync -avzP data/processed/splits/split_assignments.parquet \
#              two:~/tbox-finder/data/processed/splits/
#          The DVC pair is .gitignore'd so `git reset --hard` on re-sync keeps them; the LFS
#          file IS tracked, so a re-checkout can revert it to a pointer — hence the check.
#          This body verifies each against its committed pointer (md5 for .dvc, sha256 for
#          LFS) plus PAR1 magic, and exits 2 on any mismatch: a silently different or
#          pointer-shaped corpus would size the wrong run.
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-05-sizing-smoke && git reset --hard origin/p2-05-sizing-smoke && \
#              sbatch slurm/p2/sizing_smoke.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/sizing_smoke.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b)
#          reports/p2/sizing_smoke.DONE present, (c) reports/p2/sizing_smoke.json validates.
#          ⚠️ The `.err` is EXPECTED NON-EMPTY on success (transformers/Caduceus
#          trust_remote_code load notes, torchrun rendezvous chatter) — scan for real
#          tracebacks, DO NOT gate on zero bytes. imp.md's gate text says zero-byte; that
#          clause is the P1-01 pre-torch convention and every torch job here overrides it
#          (slurm/p1/seg_smoke.sbatch:19-24).
# RETRIEVE: small git-tracked JSON reports — not DVC:
#            rsync -avzP two:~/tbox-finder/reports/p2/sizing_smoke.json <local>/reports/p2/
#            rsync -avzP two:~/tbox-finder/reports/p2/sizing/ <local>/reports/p2/sizing/
#          (the per-point dir is rsynced, so .gitignore carries rules for its run signals.)

#SBATCH --job-name=tbox-p2-05-sizing-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=reports/p2/sizing_smoke_%j.out
#SBATCH --error=reports/p2/sizing_smoke_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

OUT_DIR="reports/p2"
POINT_DIR="$OUT_DIR/sizing"
REPORT="$OUT_DIR/sizing_smoke.json"
DONE="$OUT_DIR/sizing_smoke.DONE"
mkdir -p "$POINT_DIR"

# Singleton + stale-output clearing BEFORE the failure-prone conda activate / HF download,
# so a crash during import cannot leave a previous run's DONE behind — artifact-based
# verification keys on absence-on-failure (§9.3).
exec 9>"$OUT_DIR/.sizing_smoke.lock"
if ! flock -n 9; then
  echo "SIZING_SKIP: another job holds the lock — refusing to share outputs." >&2
  exit 1
fi
rm -f "$DONE" "$REPORT" "$POINT_DIR"/*.json

# ── Staged-input assertions (fail loud, before any GPU time) ─────────────────
# Existence is NOT enough, twice over:
#   (1) a stale/truncated parquet would size a different corpus and the numbers would look
#       perfectly reasonable, so the DVC-tracked files are md5'd against their COMMITTED
#       .dvc pointer;
#   (2) split_assignments.parquet is git-LFS (.gitattributes) and THERE IS NO git-lfs ON
#       THIS CLUSTER, so a fresh checkout leaves a 132-byte POINTER TEXT in its place —
#       non-empty, so `[ -s ]` passes it happily, and the run then dies reading a pointer
#       as parquet (or worse, wouldn't). Every parquet is checked for the `PAR1` magic,
#       which catches the pointer AND truncation in one cheap test.
stage_hint() {
  echo "  rsync -avzP $1 two:~/tbox-finder/$(dirname "$1")/" >&2
}

assert_parquet_magic() {
  local data_path="$1"
  if [ ! -s "$data_path" ]; then
    echo "FATAL: $data_path missing. Stage it from the laptop (see INPUT in this file's header):" >&2
    stage_hint "$data_path"
    exit 2
  fi
  if [ "$(head -c 4 "$data_path")" != "PAR1" ]; then
    echo "FATAL: $data_path is not a parquet (no PAR1 magic; $(stat -c %s "$data_path") bytes)." >&2
    if head -c 40 "$data_path" | grep -q 'git-lfs.github.com'; then
      echo "  It is an UNSMUDGED git-LFS POINTER — this cluster has no git-lfs on PATH." >&2
      echo "  Stage the real file from the laptop:" >&2
      stage_hint "$data_path"
    fi
    exit 2
  fi
}

assert_dvc_artifact() {
  local data_path="$1" dvc_path="$2" want got
  assert_parquet_magic "$data_path"
  want="$(awk '/^- md5:/ {print $3; exit}' "$dvc_path")"
  got="$(md5sum "$data_path" | awk '{print $1}')"
  if [ -z "$want" ]; then
    echo "FATAL: could not read an md5 out of $dvc_path" >&2
    exit 2
  fi
  if [ "$want" != "$got" ]; then
    echo "FATAL: $data_path md5 $got != committed .dvc pointer $want — wrong/stale corpus." >&2
    exit 2
  fi
  echo "staged-ok $data_path md5=$got"
}

# The LFS pointer is itself the provenance record: with no git-lfs installed, `git show`
# hands back the POINTER TEXT rather than smudging it, so its `oid sha256` is a committed
# checksum for the real file — the exact analogue of the .dvc md5 above. Read it and verify.
assert_lfs_artifact() {
  local data_path="$1" ptr want_oid want_size got_oid got_size
  assert_parquet_magic "$data_path"
  ptr="$(git show "HEAD:$data_path" 2>/dev/null || true)"
  want_oid="$(printf '%s\n' "$ptr" | awk -F'sha256:' '/^oid sha256:/ {print $2; exit}')"
  want_size="$(printf '%s\n' "$ptr" | awk '/^size / {print $2; exit}')"
  if [ -z "$want_oid" ]; then
    # git-lfs present and smudging (or the file left LFS): magic check already passed.
    echo "staged-ok $data_path (no readable LFS pointer to verify against; PAR1 magic ok)"
    return 0
  fi
  got_oid="$(sha256sum "$data_path" | awk '{print $1}')"
  got_size="$(stat -c %s "$data_path")"
  if [ "$want_oid" != "$got_oid" ] || [ "$want_size" != "$got_size" ]; then
    echo "FATAL: $data_path does not match its committed LFS pointer." >&2
    echo "  want oid=$want_oid size=$want_size" >&2
    echo "  got  oid=$got_oid size=$got_size" >&2
    exit 2
  fi
  echo "staged-ok $data_path sha256=$got_oid"
}

assert_dvc_artifact data/interim/flank_context/context_v0.parquet \
                    data/interim/flank_context/context_v0.parquet.dvc
assert_dvc_artifact data/processed/labels/labels_v0.parquet \
                    data/processed/labels/labels_v0.parquet.dvc
assert_lfs_artifact data/processed/splits/split_assignments.parquet

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna

export CUDA_HOME="$CONDA_PREFIX"          # deepspeed/nvcc probe CUDA_HOME at import (ADR-0002 A4).
export WANDB_MODE=offline                 # no live streaming from compute nodes (§9.2).
export PYTHONHASHSEED=0                    # determinism (§8.3; matches conf/train/stage1.yaml).
                                           # train_stage1.check_pythonhashseed RAISES if unset —
                                           # it VERIFIES, never sets (CPython fixes hash
                                           # randomisation at interpreter startup, so an
                                           # in-process os.environ write is a no-op).
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2                   # 16 cpus / 8 ranks; torchrun warns and defaults to 1.

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.
export HF_HOME="$BUILD/hf"      # 2.5 GB of weights stay OFF the 87%-full HOME quota (§9.2).

# ── ADR-0002 A2 C2: the kernel is a capability, not a speed knob ─────────────
# `import Mamba` FAILS without the compiled selective_scan_cuda (the "pure-PyTorch
# fallback" D3 promised does not exist), and A5 measured the fallback at ~871x slower — so a
# silent fallback would make every throughput number here meaningless. Assert, don't hope.
#
# ⚠️ `import torch` FIRST is load-bearing, not tidiness: selective_scan_cuda is a C
# extension linked against torch's libc10.so, and nothing puts torch/lib on the loader path
# until `import torch` dlopens it — so a bare `import selective_scan_cuda` dies with
# "ImportError: libc10.so: cannot open shared object file" EVEN WHEN THE KERNEL IS PRESENT
# AND FINE. That is not hypothetical: it killed job 567 at this line. mamba_ssm imports
# torch before the kernel, so this order is also the one the real training path takes —
# assert the import as it actually happens, not as a bare name.
python -c 'import torch; import selective_scan_cuda; print(f"selective_scan_cuda OK (torch {torch.__version__})")'

# ── Pre-warm the HF cache ONCE ───────────────────────────────────────────────
# torchrun's 8 ranks each call from_pretrained with no cache_dir; without a warm cache they
# would race on 8 concurrent downloads of the same pinned revision into one HF_HOME.
PYTHONPATH=src python -c '
from tbox_finder.models.caduceus_backbone import REPO_ID, REVISION
from transformers import AutoModel
AutoModel.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)
print("hf cache warm", REPO_ID, REVISION)
'

# ── The configuration matrix ─────────────────────────────────────────────────
# One process per point: a CUDA OOM poisons the context, so an in-process batch sweep
# cannot tell "batch 32 OOMs" from "batch 32 OOMs because batch 16 leaked" (P2-04).
# A point that OOMs writes NO report and is skipped — absence is the evidence, and the
# aggregator's largest_fitting_batch reads it that way. `|| true` keeps the sweep going.
#
# max_records slices the fold; throughput is a per-step property so the slice does not bias
# it, and the EXTRAPOLATION denominator is the FULL-fold epoch size (8303 windows), which
# sizing.py recomputes from the datamodule rather than from this slice.
#
# ⚠️ 4096, NOT imp.md's "few-hundred windows" — that hint was written against imp.md's
# 2-GPU plan and does not survive DDP×8. len(dataset) == n_records (one window per record
# per epoch) and ShardedSampler gives each rank len//world_size draws, so at ws=8/batch=8 a
# 512-record slice yields 512//8//8 = 8 steps/rank — 3 measured after the 5-step warmup
# discard, i.e. the DDP×8 point that PRD §10.3's whole "DDP×8 for throughput" claim rests on
# would be warmup noise (the exact reason P2-04 refused to quote step times). 4096 gives
# every point in the matrix the full 35 steps. Measured cost of the bigger slice: 1.8 s of
# CPU preamble per point (build_stream 1.3 s + class-count scan 0.5 s) — free.
MAX_RECORDS=4096
STEPS_PER_EPOCH=35   # 5 warmup discarded + 30 measured (the P1-14 shape).
EPOCHS_FULL_RUN=10   # the epochs the illustrative extrapolation assumes; NOT a pinned value.

run_point() {
  local ws="$1" bs="$2" ckpt="$3"
  local key="ws${ws}_bs${bs}_ckpt${ckpt}"
  local out="$POINT_DIR/${key}.json"
  local ckpt_flag="false"
  [ "$ckpt" -eq 1 ] && ckpt_flag="true"

  echo "── point $key ──"
  set +e
  if [ "$ws" -eq 1 ]; then
    PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.train.train_stage1 \
      batch_size="$bs" gradient_checkpointing="$ckpt_flag" epochs=1 \
      steps_per_epoch="$STEPS_PER_EPOCH" max_records="$MAX_RECORDS" \
      save_checkpoint=false report_path="$out" tracking.mode=offline
  else
    # No --nproc_per_node > available GPUs; torchrun exports RANK/WORLD_SIZE/LOCAL_RANK +
    # MASTER_ADDR/PORT, which train_stage1 reads (it never calls torchrun itself) and
    # ShardedSampler takes its rank/world_size from. Only rank 0 writes the report.
    PYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone --nproc_per_node="$ws" \
      -m tbox_finder.train.train_stage1 \
      batch_size="$bs" gradient_checkpointing="$ckpt_flag" epochs=1 \
      steps_per_epoch="$STEPS_PER_EPOCH" max_records="$MAX_RECORDS" \
      save_checkpoint=false report_path="$out" tracking.mode=offline
  fi
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ] || [ ! -s "$out" ]; then
    echo "SIZING_POINT_SKIP $key rc=$rc (OOM or failure — no report written)" >&2
    rm -f "$out"
  else
    echo "SIZING_POINT_OK $key"
  fi
}

# (a) Batch/VRAM sweep on ONE GPU, checkpointing on and off — the §10.3 "full fine-tune
#     fits one GPU" clause plus the checkpointing VRAM/throughput trade at matched batch.
for bs in 1 2 4 8 16 32; do
  for ckpt in 1 0; do
    run_point 1 "$bs" "$ckpt"
  done
done

# (b) DDP scaling at a fixed batch, checkpointing ON (the PRD §10.3 default) — the
#     "DDP×8 for throughput" claim. ws=1 is already covered by (a) at bs=8/ckpt=1 and is
#     the baseline the efficiencies divide by.
for ws in 2 4 8; do
  run_point "$ws" 8 1
done

# (c) The §10.3 target point without checkpointing, for the DDP×8 checkpointing trade.
run_point 8 8 0

# ── Aggregate ────────────────────────────────────────────────────────────────
shopt -s nullglob
POINTS=("$POINT_DIR"/*.json)
shopt -u nullglob
if [ "${#POINTS[@]}" -eq 0 ]; then
  echo "SIZING_FAIL: every point failed — nothing to aggregate." >&2
  exit 1
fi

set +e
PYTHONPATH=src python -m tbox_finder.train.sizing \
  --points "${POINTS[@]}" --out "$REPORT" --epochs "$EPOCHS_FULL_RUN"
RC=$?
set -e

# DONE on the success path only, gated on RC *and* a non-empty report — if python exited 0
# but wrote nothing, force RC!=0 rather than certify a run that produced no evidence.
if [ "$RC" -eq 0 ] && [ -s "$REPORT" ]; then
  touch "$DONE"
  echo "SIZING_DONE job=${SLURM_JOB_ID:-none} points=${#POINTS[@]}"
else
  echo "SIZING_FAIL rc=$RC job=${SLURM_JOB_ID:-none}" >&2
  [ "$RC" -ne 0 ] || RC=1
fi
exit "$RC"
