#!/bin/bash
# P2-06 — Stage-1 hyperparameter sweep: γ × learning-rate × class_weight_alpha → best config
# (PRD §11/§12; ADR-0005 D3/A3/D15; ADR-0004 GATE-4; ADR-0003 D7 + Amendment A2; CLAUDE.md §9.3).
#
# Trains the real P2-04 entrypoint once per grid point over the full inner_train fold, evaluates
# each point on the P2-06a selection-val rung, and writes one train_stage1 report per point into
# reports/p2/sweep/<KEY>.json. The winner is picked LATER (a separate, post-sweep step) by
# train/select_best.py, which ranks on eval_metrics.gate4_core_min_f1.min_f1 over the 3 GATE-4
# core elements and REJECTS any point not scored on the full selection_val fold. This sbatch does
# NOT select, does NOT grade GATE-4 (that is the P2-14 outer rung on the leave-one-order-out
# holdout), and does NOT freeze a budget by itself — the freeze is ADR-0003 Amendment A2.
#
# THE GRID (user decision 2026-07-17): γ∈{0.5,1,2,3} × optim.lr∈{3e-5,1e-4,3e-4} ×
#   class_weight_alpha∈{0,0.25,0.5} = 36 points. Window/stride is HELD at the ADR-0005 D3+A3
#   1024/512 geometry (frozen in window_dataset.py; sweeping it would unpin the reconciliation
#   geometry GATE-4's boundary-IoU defence rests on); RC-combination + CRF are DEFERRED to P2-12
#   per train_stage1.py's own code-emitted `swept_by` map (they are NOT swept here). LoRA-rank is
#   moot (ADR-0002:129 makes full fine-tuning the Stage-1 default) and the Stage-2 aux-loss weight
#   is a P3 axis — the two dead-on-arrival members of the stale PRD §11 "six-axis" enumeration.
#
# ── WHY --array, NOT literal Hydra --multirun ────────────────────────────────
# Each point is a DDP×8 run (PRD §10.3 "DDP×8 for throughput") and therefore wants the WHOLE
# 8-GPU node. Hydra --multirun's basic launcher iterates points IN ONE PROCESS, and P2-04/P2-05
# established that one process per point is load-bearing: a CUDA OOM or a NaN-grad poisons the
# process for every later point (you cannot tell "point k OOMs" from "point k OOMs because point
# k-1 leaked"). So the sweep is a SLURM --array: one array task = one node = one point = one
# torchrun DDP×8 = one process, fully isolated. Each task drives a SINGLE Hydra run with the
# point's overrides — "Hydra-driven", just not "--multirun". A failed task writes no <KEY>.OK
# marker and is caught by the §9.3 artifact-based verification (no auto-resubmit; ADR-0003 D8
# governs a bounded ≤2% batch re-submit under a SEPARATE ack).
#
# ── OVERRIDE PRECEDENCE (or the sweep is a silent no-op) ─────────────────────
# train_stage1.py::_cfg_from_mapping (line ~1622) sets top-level keys FIRST, then runs the group
# loop SECOND — so a flat `lr=` / `++lr=` is OVERWRITTEN by the default optim.lr and every point
# would train at 1e-4 while self-consistently reporting the swept value. lr therefore MUST be set
# as `optim.lr=…`. gamma and class_weight_alpha ARE top-level dataclass fields (@package _global_)
# and are set as bare `gamma=…` / `class_weight_alpha=…`.
#
# ── HANDOFF DECISIONS P2-06 OWNS (recorded here + in code + in the dev-log) ──
#  (b) class_counts scope: RETAINED at record-scope (compute_class_counts scans each record once,
#      NOT the oversampled draw stream; class_counts_scope.weighted_draw_stream=false). At α>0 the
#      inverse-frequency weights therefore describe the RAW inner_train fold, a fixed and
#      interpretable semantics held constant across all 36 points — so α is a well-posed axis
#      ("how much raw-fold inverse-freq loss-weighting ON TOP of the P2-01 curriculum sampler"),
#      and select_best ranks on val min-F1, not on loss_mass_share. The draw-stream alternative is
#      declined (it is epoch-dependent — the draws reshuffle — for no selection benefit).
#  (c) per-step grad-finiteness scan (inside step_seconds): RETAINED. γ=0.5 is the exact fractional
#      regime P2-02 found a silent NaN gradient in, so per-step NaN detection is worth its cost; and
#      the ADR-0003 A2 budget (≈58 GPU-h estimate under a ≤90 GPU-h aggregate bound) is conservative
#      BECAUSE the scan is included — the right direction for a budget. (A clip_grad_norm_-derived
#      cheap replacement is noted for P2-09.)
#  report_path / checkpoint_dir: SCOPED PER POINT below (else the single fixed path is clobbered by
#      whichever task finishes last — silently). save_checkpoint=false: the sweep ranks on the val
#      metric in the report; P2-09 retrains the WINNER on the full D5 fold (exclude_selection_val
#      =false), so keeping 36 inner_train checkpoints buys nothing and costs disk.
#
# WHERE:   cluster `gpu` partition, EIGHT RTX A4000 (sm_86) per array task. env tbox-ml-dna
#          (ADR-0002 A4). fp32 (no precision knob; A7 disables TF32) — same honesty scope as P2-05.
# INPUT:   the THREE corpus parquets are STAGED by rsync, NOT dvc-pulled (the ml envs have no dvc)
#          and NOT carried by the checkout. Stage them ONCE before submitting (they are shared by
#          all 36 tasks):
#            rsync -avzP data/interim/flank_context/context_v0.parquet two:~/tbox-finder/data/interim/flank_context/
#            rsync -avzP data/processed/labels/labels_v0.parquet        two:~/tbox-finder/data/processed/labels/
#            rsync -avzP data/processed/splits/split_assignments.parquet two:~/tbox-finder/data/processed/splits/
#          ⚠️ split_assignments.parquet is git-LFS and THIS CLUSTER HAS NO git-lfs — a fresh
#          checkout / `git reset --hard` leaves a 132-byte POINTER (non-empty, so `[ -s ]` passes
#          it) that would size/train the wrong corpus. This body verifies every parquet against its
#          committed pointer (md5 for .dvc, sha256 for LFS) plus PAR1 magic and exits 2 on mismatch.
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-06-sweep && git reset --hard origin/p2-06-sweep && \
#              sbatch slurm/p2/sweep_stage1.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/sweep_stage1.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty (all array tasks gone),
#          (b) all 36 reports/p2/sweep/<KEY>.OK markers present, (c) each reports/p2/sweep/<KEY>.json
#          validates (train_stage1 writes only a self-validated report). ⚠️ the `.err` is EXPECTED
#          NON-EMPTY on success (transformers/Caduceus trust_remote_code notes, torchrun rendezvous
#          chatter) — scan for real tracebacks, DO NOT gate on zero bytes (P2-05 / slurm/p1
#          precedent). A missing <KEY>.OK is a §9.3 stop (surface it; no auto-resubmit).
# RETRIEVE: small git-tracked JSON reports + run signals — not DVC:
#            rsync -avzP two:~/tbox-finder/reports/p2/sweep/ <local>/reports/p2/sweep/

#SBATCH --job-name=tbox-p2-06-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-35
#SBATCH --output=reports/p2/sweep/slurm_%A_%a.out
#SBATCH --error=reports/p2/sweep/slurm_%A_%a.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).
# Each task requests all 8 GPUs of a node, so the scheduler runs at most one task per node
# (≈2 concurrent when both `one` and `two` are up, ≈1 when only `two` is up) and queues the rest.

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# ── ⛔ REFUSES TO RUN — known-destructive, disabled at P2-10d′-c ──────────────
# The per-point clear below removes `$SWEEP_DIR/${KEY}.json` / `.OK`, and all 36 of those
# `reports/p2/sweep/*.json` point reports are now GIT-TRACKED (committed at P2-06 as the
# sweep's evidence). Deleting a tracked path leaves the tree dirty outside
# `train_stage1._DATA_STAGING_PREFIXES = ("data/",)`, and this job runs `train_stage1` per
# point — so every point would fail its own `provenance_complete` clause AFTER training,
# the same way `train_production.sbatch` would have (measured on the cluster:
# dirty=[staged parquet] -> True; +[a deleted tracked report] -> FALSE).
#
# The P2-06 sweep is COMPLETE and its winner (`g0.5_lr3e-4_a0.5`) is already promoted, so
# nothing is waiting on this job. Re-enabling it needs the point reports written to
# node-local scratch and copied back after the array finishes (§9.2) — which cannot be
# executed or verified from the laptop, so the job is stopped before it can destroy the
# committed evidence rather than left runnable behind a test exemption (CodeRabbit r1).
#
# To re-enable: move the point paths under $BUILD, copy them back at the end, delete this
# block, and drop `sweep_stage1.sbatch` from `_KNOWN_DEFECTIVE`/`_MUST_BE_DISABLED` in
# `tests/unit/test_sbatch_rm_targets.py` (the xfail is strict, so the suite fails until
# that entry goes).
echo "SWEEP_DISABLED: slurm/p2/sweep_stage1.sbatch is disabled at P2-10d'-c — its" >&2
echo "  per-point clear deletes 36 git-TRACKED sweep reports, which fails every point's" >&2
echo "  provenance_complete clause after training. The P2-06 sweep is already complete." >&2
echo "  Nothing has been deleted." >&2
exit 2

SWEEP_DIR="reports/p2/sweep"
mkdir -p "$SWEEP_DIR"

# ── Decode this array task's id → exactly one (γ, lr, α) grid point ───────────
# tid = gi*(|LRS|*|ALPHAS|) + li*|ALPHAS| + ai, so tid 0..35 enumerates the full 4×3×3 product
# once. GAMMAS is the slowest axis; ALPHAS the fastest. Enforced against |grid| to fail loud if the
# --array range and this table ever drift apart.
GAMMAS=(0.5 1.0 2.0 3.0)
LRS=(3e-5 1e-4 3e-4)
ALPHAS=(0.0 0.25 0.5)
NG=${#GAMMAS[@]}; NL=${#LRS[@]}; NA=${#ALPHAS[@]}
NGRID=$(( NG * NL * NA ))
tid="${SLURM_ARRAY_TASK_ID:?this sbatch must be launched as an --array job (no SLURM_ARRAY_TASK_ID)}"
if [ "$tid" -lt 0 ] || [ "$tid" -ge "$NGRID" ]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID=$tid out of range [0,$NGRID); --array range disagrees with the grid table." >&2
  exit 2
fi
gi=$(( tid / (NL * NA) ))
rem=$(( tid % (NL * NA) ))
li=$(( rem / NA ))
ai=$(( rem % NA ))
GAMMA="${GAMMAS[$gi]}"
LR="${LRS[$li]}"
ALPHA="${ALPHAS[$ai]}"
KEY="g${GAMMA}_lr${LR}_a${ALPHA}"

REPORT="$SWEEP_DIR/${KEY}.json"
OK="$SWEEP_DIR/${KEY}.OK"

# epochs/batch are implementer defaults, NOT pins. epochs=10 is the count ADR-0003 A2's 1.613
# GPU-h/run basis assumes (GPU-hours scale linearly in it; rescale rather than re-measure);
# batch=8 is P2-05's measured basis point. Both are held constant across all 36 points, so the
# sweep compares configs fairly. P2-09 (production retrain of the winner) owns the epoch count /
# early-stopping.
EPOCHS=10
BATCH=8

# Clear only THIS point's stale success markers, before the failure-prone conda activate / HF
# download, so a crash during import cannot leave a previous run's OK behind (§9.3 keys on
# absence-on-failure). Distinct KEY per task ⇒ no cross-task interference.
rm -f "$OK" "$REPORT"

# ── Staged-input assertions (fail loud, before any GPU time) ─────────────────
# Existence is not enough: (1) a stale/truncated parquet sizes/trains a different corpus and the
# numbers look reasonable, so DVC-tracked files are md5'd against their committed .dvc pointer;
# (2) split_assignments.parquet is git-LFS with NO git-lfs on this cluster, so a checkout leaves a
# 132-byte POINTER TEXT — non-empty, `[ -s ]` passes it. Every parquet is checked for `PAR1` magic
# (catches the pointer AND truncation) and against its committed checksum.
stage_hint() { echo "  rsync -avzP $1 two:~/tbox-finder/$(dirname "$1")/" >&2; }

assert_parquet_magic() {
  local data_path="$1"
  if [ ! -s "$data_path" ]; then
    echo "FATAL: $data_path missing. Stage it from the laptop (see INPUT in this file's header):" >&2
    stage_hint "$data_path"; exit 2
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
  if [ -z "$want" ]; then echo "FATAL: could not read an md5 out of $dvc_path" >&2; exit 2; fi
  if [ "$want" != "$got" ]; then
    echo "FATAL: $data_path md5 $got != committed .dvc pointer $want — wrong/stale corpus." >&2
    exit 2
  fi
  echo "staged-ok $data_path md5=$got"
}

# With no git-lfs installed, `git show HEAD:<path>` hands back the POINTER TEXT (rather than
# smudging), so its `oid sha256` is a committed checksum for the real file — the LFS analogue of
# the .dvc md5. Read it and verify.
assert_lfs_artifact() {
  local data_path="$1" ptr want_oid want_size got_oid got_size
  assert_parquet_magic "$data_path"
  # ⚠ FAIL CLOSED on an unreadable / non-pointer object. An earlier version returned "staged-ok"
  # here whenever the pointer could not be read, leaving PAR1 magic as the only check — and
  # PAR1 magic passes ANY parquet, including a stale or wrong-fold one. A staged-input guard
  # that degrades to "it's a parquet" is precisely the shape that burns a submit-ack: the job
  # starts, trains on the wrong table, and reports a self-consistent result.
  if ! ptr="$(git show "HEAD:$data_path" 2>/dev/null)"; then
    echo "FATAL: could not read the committed object for $data_path (git show failed)." >&2
    exit 2
  fi
  want_oid="$(printf '%s\n' "$ptr" | awk -F'sha256:' '/^oid sha256:/ {print $2; exit}')"
  want_size="$(printf '%s\n' "$ptr" | awk '/^size / {print $2; exit}')"
  if [ -z "$want_oid" ] || [ -z "$want_size" ]; then
    echo "FATAL: the committed object for $data_path is not a readable git-LFS pointer," >&2
    echo "  so there is no committed checksum to verify the staged file against." >&2
    exit 2
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
                                           # train_stage1.check_pythonhashseed RAISES if unset — it
                                           # VERIFIES, never sets (CPython fixes hash randomisation
                                           # at interpreter startup, so an in-process write is inert).
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2                   # 16 cpus / 8 ranks; torchrun warns and defaults to 1.

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}:${SLURM_ARRAY_TASK_ID:-none}"
echo "point KEY=$KEY  gamma=$GAMMA  optim.lr=$LR  class_weight_alpha=$ALPHA  epochs=$EPOCHS  batch=$BATCH"
echo "sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.
export WANDB_DIR="$BUILD"        # offline W&B curves stay node-local (the JSON report carries the
                                 # selection metric; W&B here is secondary telemetry, discarded on
                                 # exit to keep 36 concurrent tasks off the NFS write path).

# Shared HF cache on HOME (NFS, visible to every array task and both GPU nodes): the pinned
# backbone revision downloads ONCE and every later task cache-hits, instead of 36× re-downloads
# into 36 node-local /tmp dirs. ~2.5 GB on HOME (737 GB free, §9.2); HF file locks make the
# concurrent first-tasks safe. Pre-warm ONCE per task (idempotent) before torchrun, so the 8 ranks
# never race on a cold cache (the P2-05 pattern).
export HF_HOME="$HOME/tbox-finder-hf-cache"
mkdir -p "$HF_HOME"

# ⚠️ `import torch` BEFORE selective_scan_cuda: the kernel is a C extension linked against torch's
# libc10.so, and nothing puts torch/lib on the loader path until `import torch` dlopens it — so a
# bare `import selective_scan_cuda` dies with "libc10.so: cannot open shared object file" EVEN WHEN
# THE KERNEL IS FINE. That killed job 567. mamba_ssm imports torch first, so this is also the order
# the real training path takes. The kernel is a capability (ADR-0002 A2 C2): its absence would make
# every run a ~871× slower silent fallback, so assert it, don't hope.
python -c 'import torch; import selective_scan_cuda; print(f"selective_scan_cuda OK (torch {torch.__version__})")'

PYTHONPATH=src python -c '
from tbox_finder.models.caduceus_backbone import REPO_ID, REVISION
from transformers import AutoModel
AutoModel.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)
print("hf cache warm", REPO_ID, REVISION)
'

# ── Train this point (DDP×8) + evaluate on the P2-06a selection_val rung ──────
# steps_per_epoch/max_records are left null (config default) ⇒ the FULL inner_train fold (7,472
# records, D5 fold minus the P2-06a selection-val carve) — a real training run, not a smoke.
# eval_val=true / eval_max_records=null ⇒ the full 830-record selection_val fold (full_fold=true),
# so select_best does not reject the point. Only rank 0 writes $REPORT.
echo "── training point $KEY ──"
set +e
PYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \
  -m tbox_finder.train.train_stage1 \
  gamma="$GAMMA" optim.lr="$LR" class_weight_alpha="$ALPHA" \
  epochs="$EPOCHS" batch_size="$BATCH" gradient_checkpointing=true \
  eval_val=true eval_max_records=null save_checkpoint=false \
  report_path="$REPORT" checkpoint_dir="checkpoints/p2/sweep/${KEY}" \
  tracking.mode=offline
rc=$?
set -e

if [ "$rc" -ne 0 ] || [ ! -s "$REPORT" ]; then
  echo "SWEEP_POINT_FAIL $KEY rc=$rc (report not written — this point is NOT counted; §9.3 sees the missing .OK)" >&2
  [ "$rc" -ne 0 ] || rc=1
  exit "$rc"
fi

touch "$OK"
echo "SWEEP_POINT_OK $KEY report=$REPORT"
