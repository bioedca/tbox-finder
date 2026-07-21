#!/bin/bash
# P2-09 — round-0 PRODUCTION Stage-1 scanner: retrain the P2-06 winner on the full D5 fold.
# (PRD §10.1/§10.3/§11; ADR-0002 backbone+kernels+A7 precision; ADR-0004 D5 fold;
#  ADR-0005 D3/A3 geometry+reconciliation + curriculum; ADR-0003 D7/A2 budget; CLAUDE.md §9.3.)
#
# Trains the real P2-04 entrypoint ONCE, DDP×8, over the FULL ADR-0004 D5 nested training
# fold (8,303 records — `exclude_selection_val=false`), with the P2-06-promoted config
# (`g0.5_lr3e-4_a0.5`: γ=0.5, optim.lr=3e-4, class_weight_alpha=0.5), and SAVES the
# checkpoint. This is the shipped scanner lineage (§10.1); the FINAL production checkpoint
# emerges only after the P2-10 mining rounds. Round-0 is NOT itself gate-graded — GATE-4 is
# P2-14 on the leave-one-order-out holdout. Its gate here is operational only.
#
# ── WHY THE FULL FOLD, AND WHY NO VAL METRIC (user decision 2026-07-19) ──────
# A sweep selects a config on the P2-06a `selection_val` rung, so that rung must stay OUT of
# the points' gradients (`exclude_selection_val=true`, the default). Once the config is
# fixed, that reason is spent: withholding 10.01% of the corpus (8,303 → 7,472) from the
# SHIPPED checkpoint buys nothing, so P2-09 trains the full D5 fold. The cost is that there
# is then no free held-out rung to report a val number on — the LOO holdout is P2-14's
# grading set (using it here contaminates the §12 headline), and `selection_val` is now
# inside the training set. So this run reports NO val metric (`eval_val=false`); an in-sample
# number labelled "held-out" is the exact failure the P2-06a rung exists to prevent. The
# entrypoint REFUSES `eval_val=true` alongside `exclude_selection_val=false` at construction
# (Stage1TrainConfig.__post_init__) — the disjointness evidence would read zero even then,
# because it is measured against the inner_train fold DEFINITION, not the trained records.
#
# ── PRECISION: fp32, TF32 OFF — as ADR-0002 A7 pins, NOT bf16 ────────────────
# imp.md's Outputs line says "bf16/TF32", but no PRD line or ADR pins bf16 for Stage-1:
# PRD §10.3's "bf16/TF32 native" describes the A4000 CARD, its Caduceus bullet names no
# precision, and A7 disables TF32/cudnn autotune for determinism (a TF32 re-enable moves
# metrics ≥1e-2). `train_stage1` exposes no precision knob. So this run is fp32/TF32-off,
# the same honesty scope as P2-05/P2-06 (user decision 2026-07-19). imp.md's wording is
# descriptive drift, flagged in the dev-log, not a pin.
#
# WHERE:   cluster `gpu` partition, EIGHT RTX A4000 (sm_86) on ONE node. env tbox-ml-dna
#          (ADR-0002 A4). No --nodelist (never pin; `one` is often down, and cryosparc keeps
#          both nodes fragmented — budget an open-ended queue wait).
# INPUT:   FIVE parquets, STAGED by rsync (the ml envs have no dvc; a checkout does not
#          carry them). P2-10d′-b added the last two — the §9.1 seed-negative substrate —
#          and they are assert_dvc_artifact-guarded below, so an unstaged one exits 2
#          AFTER the queue wait. Stage them ALL once before submitting:
#            rsync -avzP data/interim/flank_context/context_v0.parquet two:~/tbox-finder/data/interim/flank_context/
#            rsync -avzP data/processed/labels/labels_v0.parquet        two:~/tbox-finder/data/processed/labels/
#            rsync -avzP data/processed/splits/split_assignments.parquet two:~/tbox-finder/data/processed/splits/
#            rsync -avzP data/processed/negatives/mining_pool_v0.parquet two:~/tbox-finder/data/processed/negatives/
#            rsync -avzP data/processed/negatives/decoys_v0.parquet      two:~/tbox-finder/data/processed/negatives/
#          ⚠️ split_assignments.parquet is git-LFS and THIS CLUSTER HAS NO git-lfs — a fresh
#          checkout / `git reset --hard` leaves a 132-byte POINTER (non-empty, so `[ -s ]`
#          passes it) that would train the wrong corpus. This body verifies every parquet
#          against its committed pointer (md5 for .dvc, sha256 for LFS) + PAR1 magic, exit 2
#          on mismatch.
# SUBMIT:  from the repo root so relative --output/--error resolve. SYNC FIRST, SUBMIT
#          SECOND — never in one `&&` chain: `git reset --hard` rewrites the LFS-pointered
#          split table back to its 132-byte pointer, undoing the staging the job then
#          verifies, so the parquets MUST be re-rsync'd after the reset and before sbatch.
#          (The old recipe here chained them and named `p2-10dp-b`, a branch since deleted
#          from origin — squash-merged as 748c4f2.)
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout main && git reset --hard origin/main"'
#            # …re-stage ALL FIVE parquets (see INPUT above)…
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/train_production.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/train_production.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b)
#          reports/p2/train_stage1_production.DONE present, (c) the checkpoint +
#          provenance.json present, (d) reports/p2/train_stage1_production.json validates
#          (train_stage1 writes only a self-validated report AND fails the process on a
#          failed gate). ⚠️ the `.err` is EXPECTED NON-EMPTY on success (transformers/Caduceus
#          trust_remote_code notes, torchrun rendezvous + W&B-offline chatter) — scan for real
#          tracebacks, DO NOT gate on zero bytes (P2-05/P2-06 precedent). A missing DONE is a
#          §9.3 stop (surface it; no auto-resubmit).
# RETRIEVE: the checkpoint + its sidecar (plain scratch, NOT dvc — the ml env has no dvc; the
#          `dvc add` runs LOCALLY on the laptop after retrieval), plus the git-tracked report:
#            rsync -avzP two:~/tbox-finder/data/processed/checkpoints/stage1_production/ <local>/data/processed/checkpoints/stage1_production/
#            rsync -avzP two:~/tbox-finder/reports/p2/train_stage1_production.json <local>/reports/p2/

#SBATCH --job-name=tbox-p2-10dpb-production
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=reports/p2/train_production_%j.out
#SBATCH --error=reports/p2/train_production_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

OUT_DIR="reports/p2"
REPORT="$OUT_DIR/train_stage1_production.json"
DONE="$OUT_DIR/train_stage1_production.DONE"
CKPT_DIR="data/processed/checkpoints/stage1_production"
mkdir -p "$OUT_DIR" "$CKPT_DIR"

# ── Singleton lock ───────────────────────────────────────────────────────────
# Held before anything else so two jobs cannot share the output paths. The stale-output
# CLEAR is deliberately deferred until AFTER the staged-input checks below (not here): a
# re-submit whose staging fails — e.g. `git reset --hard` left split_assignments.parquet an
# unsmudged 132-byte LFS pointer, the exact trap this file's header warns about — must NOT
# destroy the last-good production checkpoint before it has a verified input to replace it
# with. The clear still lands before the failure-prone conda activate / HF download, so a
# crash mid-import cannot leave a previous run's DONE behind (§9.3 keys on absence-on-failure).
exec 9>"$OUT_DIR/.train_production.lock"
if ! flock -n 9; then
  echo "PRODUCTION_SKIP: another job holds the lock — refusing to share outputs." >&2
  exit 1
fi

# ── Staged-input assertions (fail loud, before any GPU time OR any destructive clear) ────
# Existence is not enough: a stale/truncated parquet trains a different corpus and the loss
# looks reasonable. DVC-tracked files are md5'd against their committed .dvc pointer;
# split_assignments.parquet is git-LFS with NO git-lfs here, so a checkout leaves a 132-byte
# POINTER TEXT — non-empty, `[ -s ]` passes it. Every parquet is checked for `PAR1` magic
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
# smudging), so its `oid sha256` is a committed checksum for the real file — the LFS analogue
# of the .dvc md5. ⚠ FAIL CLOSED on an unreadable / non-pointer object (P2-06 fixed a version
# that returned "staged-ok" whenever the pointer could not be read, leaving PAR1 magic — which
# passes ANY parquet — as the only check).
assert_lfs_artifact() {
  local data_path="$1" ptr want_oid want_size got_oid got_size
  assert_parquet_magic "$data_path"
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
# P2-10d′-b: the §9.1 seed-negative substrate. Both are DVC-tracked and must be rsync'd like
# the corpus parquets. Asserted here so a missing one fails at staging rather than deep in
# the run: a missing mining pool surfaces as "0 injectable negatives" and a missing decoy
# pool as EmbeddingError("decoy pool not found") — different messages, both only AFTER the
# queue wait.
assert_dvc_artifact data/processed/negatives/mining_pool_v0.parquet \
                    data/processed/negatives/mining_pool_v0.parquet.dvc
assert_dvc_artifact data/processed/negatives/decoys_v0.parquet \
                    data/processed/negatives/decoys_v0.parquet.dvc

# ── Stale-output clear — ONLY now that the staged inputs are verified ─────────
# Deferred to here (not at lock acquisition) so a failed staging check above cannot destroy
# a previously successful production checkpoint/report/provenance (CodeRabbit P2-09 r1). Still
# ahead of conda/HF, so a crash in that failure-prone path leaves no stale DONE.
#
# ⚠️ "$REPORT" IS DELIBERATELY *NOT* CLEARED HERE — it is a GIT-TRACKED path (committed at
# 082f8c7), and deleting it makes the working tree dirty at a path outside
# `_DATA_STAGING_PREFIXES = ("data/",)`. `build_report` snapshots `git status` ONCE while
# that deletion is live, so `_provenance_complete` re-derives FALSE -> `overall_pass` FALSE
# -> `torch.save` is SKIPPED -> the report is written over the committed P2-09 baseline
# carrying `overall_pass: false` -> RuntimeError -> rc!=0 -> the EXIT trap wipes $BUILD with
# the checkpoint inside it. That is the ENTIRE run — ~20 GPU-h — destroyed at the finish
# line, and it had never fired before because job 671's report predates the clause
# (schema_version 1, no `git_dirty_paths` key). Rank 0 overwrites the report on success
# anyway, and freshness is still signalled by DONE + checkpoint + provenance, all untracked.
# Measured: dirty=[staged parquet] -> provenance_complete True; +[the report] -> FALSE.
# Do NOT "fix" this by widening `_DATA_STAGING_PREFIXES` — that blinds the clause to real
# `reports/` dirt, which is the one thing it exists to catch (P2-10d′-c).
rm -f "$DONE" "$CKPT_DIR/stage1.pt" "$CKPT_DIR/provenance.json"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna

export CUDA_HOME="$CONDA_PREFIX"          # deepspeed/nvcc probe CUDA_HOME at import (ADR-0002 A4).
export WANDB_MODE=offline                 # no live streaming from compute nodes (§9.2).
export PYTHONHASHSEED=0                    # determinism (§8.3; matches conf/train/stage1.yaml).
                                           # train_stage1.check_pythonhashseed RAISES if unset —
                                           # it VERIFIES, never sets (CPython fixes hash
                                           # randomisation at interpreter startup).
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2                   # 16 cpus / 8 ranks; torchrun warns and defaults to 1.

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.
export HF_HOME="$BUILD/hf"      # 2.5 GB of weights stay OFF the 87%-full HOME quota (§9.2).
export WANDB_DIR="$BUILD"       # offline W&B curves stay node-local (the JSON report carries the
                                # gate; W&B here is secondary telemetry, discarded on exit).

# ⚠️ `import torch` BEFORE selective_scan_cuda: the kernel is a C extension linked against
# torch's libc10.so, and nothing puts torch/lib on the loader path until `import torch`
# dlopens it — so a bare `import selective_scan_cuda` dies with "libc10.so: cannot open shared
# object file" EVEN WHEN THE KERNEL IS FINE. That killed job 567. The kernel is a capability
# (ADR-0002 A2 C2): its absence would make training a ~871× slower silent fallback, so assert
# it, don't hope.
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

# ── Train the shipped scanner (DDP×8) ────────────────────────────────────────
# The P2-06 winner config, held on the two values the sweep never varied (epochs=10,
# batch_size=8 — ADR-0003 A2: "epochs=10 is an assumption, not a pin"; no val fold means no
# early-stopping signal, so a fixed count is the only option). optim.lr MUST be set through
# the optim group — a top-level `lr=` is overwritten by _cfg_from_mapping's group loop.
# exclude_selection_val=false → full 8,303-record D5 fold; eval_val=false → no val metric
# (see the header). save_checkpoint=true → rank 0 writes $CKPT/stage1.pt, but ONLY if the
# report's own gate passes (a failed run raises → rc≠0 → no DONE). Checkpoint is written to
# node-local scratch first, then moved to the repo path below (§9.2: build in /tmp, move only
# the final artifact).
BUILD_CKPT="$BUILD/checkpoint"
# Freshness fingerprint, taken INSTEAD of deleting the tracked report (see the clear above).
# Absent file -> empty string, which no md5 can equal, so "was missing" and "was replaced"
# both read as fresh and only "unchanged bytes" reads as stale.
REPORT_BEFORE="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"
echo "── training P2-09 round-0 production (full D5 fold, DDP×8, fp32) ──"
set +e
PYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \
  -m tbox_finder.train.train_stage1 \
  gamma=0.5 optim.lr=3e-4 class_weight_alpha=0.5 \
  epochs=10 batch_size=8 gradient_checkpointing=true \
  exclude_selection_val=false eval_val=false save_checkpoint=true \
  negative_pool_parquet=data/processed/negatives/mining_pool_v0.parquet \
  negative_decoy_parquet=data/processed/negatives/decoys_v0.parquet \
  negative_fraction=0.9090909090909091 \
  init_from_checkpoint=null \
  report_path="$REPORT" checkpoint_dir="$BUILD_CKPT" \
  tracking.mode=offline
rc=$?
set -e

if [ "$rc" -ne 0 ] || [ ! -s "$REPORT" ]; then
  echo "PRODUCTION_FAIL rc=$rc (report not written or process failed its own gate — §9.3 sees the missing DONE)" >&2
  [ "$rc" -ne 0 ] || rc=1
  exit "$rc"
fi
# The report is no longer cleared before the run, so "a report exists" no longer implies
# "THIS run wrote it" — a stale committed one would satisfy `[ -s ]`. Assert it changed.
# Both failure directions are explicit: an UNCOMPUTABLE fingerprint must not pass by being
# unequal to the old one (`md5sum` errors are swallowed by the pipe, so an unreadable report
# yields "" — which differs from every real md5 and would have sailed through), and an
# UNCHANGED one must not pass at all (CodeRabbit, P2-10d′-c r2).
REPORT_AFTER="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"
if [ -z "$REPORT_AFTER" ]; then
  echo "PRODUCTION_FAIL: could not fingerprint $REPORT after the run (missing or unreadable)." >&2
  exit 1
fi
if [ "$REPORT_BEFORE" = "$REPORT_AFTER" ]; then
  echo "PRODUCTION_FAIL: $REPORT is byte-identical to the one that was there before this run" >&2
  echo "  (md5 $REPORT_AFTER) — rank 0 did not write it, so it describes some OTHER run." >&2
  exit 1
fi
if [ ! -s "$BUILD_CKPT/stage1.pt" ]; then
  echo "PRODUCTION_FAIL: report passed its gate but no checkpoint was written at $BUILD_CKPT/stage1.pt." >&2
  exit 1
fi

# ── Move the checkpoint to the repo path + write its provenance sidecar ───────
# Provenance is written HERE, on the node that produced the checkpoint, so git_sha (the
# branch HEAD of this checkout), env-lock hash, seed, and the input/output content hashes are
# all the run's own. The `dvc add` of the checkpoint happens LOCALLY after retrieval (no dvc
# in the ml env); provenance.json is a small git-tracked sidecar, NOT dvc-tracked.
cp "$BUILD_CKPT/stage1.pt" "$CKPT_DIR/stage1.pt"
PYTHONPATH=src python - <<'PY'
from tbox_finder.provenance import write_provenance

write_provenance(
    "data/processed/checkpoints/stage1_production/provenance.json",
    rule=(
        "slurm/p2/train_production.sbatch :: train_stage1 "
        "(P2-10d'-b round-0 production Stage-1 scanner, §9.1 seed-negative mix)"
    ),
    script="src/tbox_finder/train/train_stage1.py",
    seed=42,
    inputs=[
        "data/interim/flank_context/context_v0.parquet",
        "data/processed/labels/labels_v0.parquet",
        "data/processed/splits/split_assignments.parquet",
        "data/processed/negatives/mining_pool_v0.parquet",
        "data/processed/negatives/decoys_v0.parquet",
    ],
    outputs=["data/processed/checkpoints/stage1_production/stage1.pt"],
    env_lock="envs/ml-dna.conda-lock.yml",
    adr="ADR-0002",
    extra={
        "step": "P2-10d'-b",
        "report": "reports/p2/train_stage1_production.json",
        "config": "P2-06 winner g0.5_lr3e-4_a0.5 (gamma=0.5, optim.lr=3e-4, class_weight_alpha=0.5)",
        "fold": "full ADR-0004 D5 nested_train (exclude_selection_val=false, ~8,303 records)",
        "negative_mix": (
            "PRD §9.1 seed ratio as a STREAM share: negative_fraction=10/11, so "
            "negative_draw_count emits exactly 10x the positive draws per epoch "
            "(ADR-0005 A7 pin 1). Pool = the mined genomic windows + the embedded "
            "dinuc_shuffled/structured_rna decoys, drawn uniformly, so per-class shares "
            "are pool-proportional (A7 pin 3). gc_background and leader_decoy take "
            "share 0 (A7 pins 4-5); a decoy whose OWN parent is outside nested_train is "
            "refused (A7 pin 6a). Realized counts are in the run's own report, not "
            "restated here."
        ),
        "round": "0 — COLD start (init_from_checkpoint=null; A7 pin 8). P2-09 is the "
                 "pre-mix baseline, NOT this run's parent.",
        "eval_val": False,
        "epochs": 10,
        "batch_size": 8,
        "world_size": 8,
        "precision": "fp32 (ADR-0002 A7: TF32/cudnn autotune off; NO bf16)",
    },
)
print("wrote data/processed/checkpoints/stage1_production/provenance.json")
PY

if [ ! -s "$CKPT_DIR/provenance.json" ]; then
  echo "PRODUCTION_FAIL: checkpoint copied but provenance.json was not written." >&2
  exit 1
fi

touch "$DONE"
echo "PRODUCTION_DONE job=${SLURM_JOB_ID:-none} checkpoint=$CKPT_DIR/stage1.pt report=$REPORT"
