#!/bin/bash
# P2-11 — class-II-CM-naive Stage-1 ABLATION: retrain the P2-06 winner on the full D5 fold
# with class-II CM structure WITHHELD from the labels.
# (PRD §5 mech 3 / §8 / §10.1 / §2.3; ADR-0004 D1/D5 label-source flag; ADR-0005 D9
#  anti-mimicry Stage-1-only; ADR-0002 backbone+kernels+A7 precision; CLAUDE.md §9.3.)
#
# THE ABLATION. This is a byte-for-byte clone of slurm/p2/train_production.sbatch — SAME
# fold (full ADR-0004 D5 nested_train, exclude_selection_val=false), SAME P2-06 winner config
# (g0.5_lr3e-4_a0.5), SAME §9.1 seed-negative curriculum (mining_pool_v0 + decoys_v0,
# negative_fraction=10/11), SAME cold start (init_from_checkpoint=null) — with exactly ONE
# difference: `label_source=naive`. That flag makes the datamodule load `naive_label_string`
# (labels.py:416) instead of `label_string`, so every class-II (TBDB001.cm-derived) record's
# per-nt target is all-`background`: its structure is withheld, class-I records unchanged
# (ADR-0004 D5 — "the two checkpoints share one leakage-controlled partition, differing only
# in label source"). Any class-II the naive checkpoint later recovers is GENERALIZATION, not
# RF00230 mimicry (PRD §5 mech 3). It is RETAINED-NOT-SHIPPED (PRD §10.1) and graded ONLY by
# the GATE-1 anti-mimicry sub-arm at P4, Stage-1-only — NEVER routed through the class-II-
# trained production Stage-2, which would rescue class-II hits and confound the test
# (ADR-0005 D9).
#
# ── WHY A RUNTIME label-source ASSERTION (the §10.3 footgun this guards) ──────
# `label_source` is a REAL Stage1TrainConfig field wired to the loader (P2-11), so
# _cfg_from_mapping threads `label_source=naive`. Were that wiring ever lost, the unknown
# key would be SILENTLY DROPPED (train_stage1._cfg_from_mapping ignores unknown keys) and
# this run would train on the production, TBDB001.cm-derived labels while looking naive — a
# checkpoint that PASSES its own gate and silently mimics RF00230, the exact confound this
# ablation exists to rule out. So after the run, before publishing the checkpoint, this body
# re-derives the label source from the run's OWN report — both the requested config
# (diagnostics.config.label_source) AND the resolved fold (class_counts_scope.label_source /
# .label_column) — and REFUSES to move the checkpoint if either is not naive.
#
# ── WHY THE FULL FOLD, NO VAL METRIC, fp32/TF32-off ──────────────────────────
# Identical rationale to train_production.sbatch (see its header): the config is already
# selected, so P2-11 trains the full D5 fold and reports NO val metric (eval_val=false;
# __post_init__ refuses eval_val=true + exclude_selection_val=false). fp32, TF32/cudnn
# autotune OFF (ADR-0002 A7); train_stage1 exposes no precision knob.
#
# WHERE:   cluster `gpu` partition, EIGHT RTX A4000 (sm_86) on ONE node. env tbox-ml-dna
#          (ADR-0002 A4). No --nodelist (never pin; `one` is often down, cryosparc fragments
#          both nodes — budget an open-ended queue wait). MEASURED footprint (the P2-10d'-b
#          twin, job 680): ~2.35 h train wall / ~18.8 GPU-h, peak 1.09 GiB/rank.
# INPUT:   the SAME FIVE parquets as train_production.sbatch — labels_v0.parquet ALREADY
#          carries the `naive_label_string` column (P0-20), so there is no new input. STAGED
#          by rsync (the ml envs have no dvc; a checkout does not carry them):
#            rsync -avzP data/interim/flank_context/context_v0.parquet two:~/tbox-finder/data/interim/flank_context/
#            rsync -avzP data/processed/labels/labels_v0.parquet        two:~/tbox-finder/data/processed/labels/
#            rsync -avzP data/processed/splits/split_assignments.parquet two:~/tbox-finder/data/processed/splits/
#            rsync -avzP data/processed/negatives/mining_pool_v0.parquet two:~/tbox-finder/data/processed/negatives/
#            rsync -avzP data/processed/negatives/decoys_v0.parquet      two:~/tbox-finder/data/processed/negatives/
#          ⚠️ split_assignments.parquet is git-LFS and THIS CLUSTER HAS NO git-lfs — a fresh
#          checkout / `git reset --hard` leaves a 132-byte POINTER ([ -s ] passes it). This
#          body verifies every parquet against its committed pointer (md5 for .dvc, sha256
#          for LFS) + PAR1 magic, exit 2 on mismatch.
# SUBMIT:  from the repo root so relative --output/--error resolve. SYNC FIRST, SUBMIT SECOND
#          (never one `&&` chain: `git reset --hard` re-breaks the LFS split table):
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-11-classII-naive && git reset --hard origin/p2-11-classII-naive"'
#            # …re-stage ALL FIVE parquets (see INPUT above)…
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/train_classII_naive.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/train_classII_naive.sbatch"'
#          AND compose this sbatch's OWN Hydra tokens in tbox-ml-dna (the job-669 lesson):
#            conda activate tbox-ml-dna && PYTHONPATH=src python -c "from hydra import ...; compose('train/stage1', overrides=['label_source=naive', ...])"
#          (tests/unit/test_sbatch_overrides.py composes it in CI too.)
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b)
#          reports/p2/train_stage1_classII_naive.DONE present, (c) the checkpoint +
#          provenance.json present, (d) the report validates AND its label_source is naive
#          (this body asserts it before publishing — a mis-wired naive run produces NO DONE).
#          ⚠️ the `.err` is EXPECTED NON-EMPTY on success (trust_remote_code notes, torchrun
#          rendezvous + W&B-offline chatter) — scan for real tracebacks, DO NOT gate on zero
#          bytes. A missing DONE is a §9.3 stop (surface it; no auto-resubmit).
# RETRIEVE: the checkpoint + its sidecar (plain scratch, NOT dvc — the `dvc add` runs LOCALLY
#          on the laptop after retrieval), plus the git-tracked report:
#            rsync -avzP two:~/tbox-finder/data/processed/checkpoints/stage1_classII_naive/ <local>/data/processed/checkpoints/stage1_classII_naive/
#            rsync -avzP two:~/tbox-finder/reports/p2/train_stage1_classII_naive.json <local>/reports/p2/

#SBATCH --job-name=tbox-p2-11-classII-naive
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=reports/p2/train_classII_naive_%j.out
#SBATCH --error=reports/p2/train_classII_naive_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

OUT_DIR="reports/p2"
REPORT="$OUT_DIR/train_stage1_classII_naive.json"
DONE="$OUT_DIR/train_stage1_classII_naive.DONE"
CKPT_DIR="data/processed/checkpoints/stage1_classII_naive"
mkdir -p "$OUT_DIR" "$CKPT_DIR"

# ── Singleton lock ───────────────────────────────────────────────────────────
# Held before anything else so two jobs cannot share the output paths. The stale-output
# CLEAR is deferred until AFTER the staged-input checks below (not here): a re-submit whose
# staging fails must NOT destroy the last-good naive checkpoint before it has a verified
# input to replace it with (CodeRabbit P2-09 r1).
exec 9>"$OUT_DIR/.train_classII_naive.lock"
if ! flock -n 9; then
  echo "NAIVE_SKIP: another job holds the lock — refusing to share outputs." >&2
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
# of the .dvc md5. ⚠ FAIL CLOSED on an unreadable / non-pointer object.
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
# labels_v0.parquet carries BOTH label_string (production) and naive_label_string (this run
# reads the latter). One committed artifact, one md5 — the naive selection is a column choice
# INSIDE the trainer (label_source=naive), not a different file.
assert_dvc_artifact data/processed/labels/labels_v0.parquet \
                    data/processed/labels/labels_v0.parquet.dvc
assert_lfs_artifact data/processed/splits/split_assignments.parquet
assert_dvc_artifact data/processed/negatives/mining_pool_v0.parquet \
                    data/processed/negatives/mining_pool_v0.parquet.dvc
assert_dvc_artifact data/processed/negatives/decoys_v0.parquet \
                    data/processed/negatives/decoys_v0.parquet.dvc

# ── Stale-output clear — ONLY now that the staged inputs are verified ─────────
# ⚠️ "$REPORT" IS DELIBERATELY *NOT* CLEARED HERE — it is a GIT-TRACKED path; deleting it
# dirties the tree outside `_DATA_STAGING_PREFIXES = ("data/",)` → build_report's git snapshot
# re-derives `_provenance_complete` FALSE → torch.save SKIPPED → the whole run (~19 GPU-h) is
# lost at the finish line. It is RESTORED to HEAD below instead. Do NOT widen
# `_DATA_STAGING_PREFIXES` (P2-10d′-c).
rm -f "$DONE" "$CKPT_DIR/stage1.pt" "$CKPT_DIR/provenance.json"

# ── RESTORE the tracked report to HEAD — idempotent re-runs (P2-10d′-e) ───────
# A SUCCESSFUL run overwrites $REPORT in place, so run #2's git snapshot would read the
# leftover dirt as modified code outside `_DATA_STAGING_PREFIXES` and lose ~19 GPU-h
# deterministically. Restore to committed bytes (never `rm` — deleting dirties just the same).
# Restore only if tracked (a checkout predating the first commit leaves it untracked).
if git ls-files --error-unmatch "$REPORT" >/dev/null 2>&1; then
  git checkout HEAD -- "$REPORT"
fi

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna

export CUDA_HOME="$CONDA_PREFIX"          # deepspeed/nvcc probe CUDA_HOME at import (ADR-0002 A4).
export WANDB_MODE=offline                 # no live streaming from compute nodes (§9.2).
export PYTHONHASHSEED=0                    # determinism (§8.3; matches conf/train/stage1.yaml).
                                           # train_stage1.check_pythonhashseed RAISES if unset —
                                           # it VERIFIES, never sets.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2                   # 16 cpus / 8 ranks; torchrun warns and defaults to 1.

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# ⚠️ Node-local scratch named JOB_SCRATCH, NOT BUILD: `conda activate tbox-ml-dna` (above)
# exports BUILD=HOST=x86_64-conda-linux-gnu from its compiler-toolchain hook, which would
# clobber a scratch var named $BUILD (that is what killed job 789 — conda-toolchain-clobbers-
# build-host). JOB_SCRATCH is verified unset across the ml env.
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT   # clean node-local scratch on every exit path; touches only $JOB_SCRATCH.
export HF_HOME="$JOB_SCRATCH/hf"    # 2.5 GB of weights stay OFF the 87%-full HOME quota (§9.2).
export WANDB_DIR="$JOB_SCRATCH"     # offline W&B curves stay node-local (the JSON report carries the gate).

# ⚠️ `import torch` BEFORE selective_scan_cuda: the kernel is a C extension linked against
# torch's libc10.so; a bare `import selective_scan_cuda` dies with "libc10.so: cannot open
# shared object file" EVEN WHEN THE KERNEL IS FINE (killed job 567).
python -c 'import torch; import selective_scan_cuda; print(f"selective_scan_cuda OK (torch {torch.__version__})")'

# ── Pre-warm the HF cache ONCE (8 ranks would else race 8 downloads into one HF_HOME) ──
PYTHONPATH=src python -c '
from tbox_finder.models.caduceus_backbone import REPO_ID, REVISION
from transformers import AutoModel
AutoModel.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)
print("hf cache warm", REPO_ID, REVISION)
'

# ── Train the naive ablation (DDP×8) ─────────────────────────────────────────
# IDENTICAL to train_production.sbatch's torchrun EXCEPT `label_source=naive` (the last
# override). optim.lr MUST go through the optim group (a top-level lr= is overwritten by
# _cfg_from_mapping's group loop). exclude_selection_val=false → full 8,303-record D5 fold;
# eval_val=false → no val metric. save_checkpoint=true → rank 0 writes the checkpoint iff the
# report's own gate passes.
JOB_CKPT="$JOB_SCRATCH/checkpoint"
REPORT_BEFORE="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"
echo "── training P2-11 class-II-CM-naive ablation (full D5 fold, DDP×8, fp32, label_source=naive) ──"
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
  label_source=naive \
  report_path="$REPORT" checkpoint_dir="$JOB_CKPT" \
  tracking.mode=offline
rc=$?
set -e

if [ "$rc" -ne 0 ] || [ ! -s "$REPORT" ]; then
  echo "NAIVE_FAIL rc=$rc (report not written or process failed its own gate — §9.3 sees the missing DONE)" >&2
  [ "$rc" -ne 0 ] || rc=1
  exit "$rc"
fi
# The report is not cleared before the run, so "a report exists" ≠ "THIS run wrote it".
# Assert it changed (an uncomputable fingerprint yields "" and must not pass by inequality;
# an unchanged one must not pass at all).
REPORT_AFTER="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"
if [ -z "$REPORT_AFTER" ]; then
  echo "NAIVE_FAIL: could not fingerprint $REPORT after the run (missing or unreadable)." >&2
  exit 1
fi
if [ "$REPORT_BEFORE" = "$REPORT_AFTER" ]; then
  echo "NAIVE_FAIL: $REPORT is byte-identical to the one that was there before this run" >&2
  echo "  (md5 $REPORT_AFTER) — rank 0 did not write it, so it describes some OTHER run." >&2
  exit 1
fi
if [ ! -s "$JOB_CKPT/stage1.pt" ]; then
  echo "NAIVE_FAIL: report passed its gate but no checkpoint was written at $JOB_CKPT/stage1.pt." >&2
  exit 1
fi

# ── Assert the RESOLVED label source is naive — the anti-mimicry guard (§10.3) ─
# The whole point of this ablation is that class-II CM structure was WITHHELD. Re-derive the
# label source from the run's OWN report — both the requested config AND the resolved fold —
# and REFUSE to publish the checkpoint if either is not naive. A silently-dropped override
# (were the P2-11 wiring lost) would have trained on the production, TBDB001.cm-derived labels
# while its own gate passed; this is the guard that catches that before any artifact ships.
python - "$REPORT" <<'PY'
import json, sys

rep = json.load(open(sys.argv[1]))
requested = (rep.get("diagnostics") or {}).get("config", {}).get("label_source")
resolved = (rep.get("class_counts_scope") or {}).get("label_source")
column = (rep.get("class_counts_scope") or {}).get("label_column")
if requested != "naive" or resolved != "naive" or column != "naive_label_string":
    sys.exit(
        f"NAIVE_FAIL: label source is not naive — requested={requested!r} "
        f"resolved={resolved!r} column={column!r}. The naive override did not reach the "
        f"loaded labels; refusing to publish a checkpoint that would silently carry "
        f"TBDB001.cm structure (the RF00230-mimicry confound this ablation rules out)."
    )
print(f"label-source OK: requested={requested} resolved={resolved} column={column}")
PY

# ── Move the checkpoint to the repo path + write its provenance sidecar ───────
# Provenance written HERE, on the node that produced the checkpoint. The `dvc add` runs
# LOCALLY after retrieval (no dvc in the ml env); provenance.json is a git-tracked sidecar.
cp "$JOB_CKPT/stage1.pt" "$CKPT_DIR/stage1.pt"
PYTHONPATH=src python - <<'PY'
from tbox_finder.provenance import write_provenance

write_provenance(
    "data/processed/checkpoints/stage1_classII_naive/provenance.json",
    rule=(
        "slurm/p2/train_classII_naive.sbatch :: train_stage1 "
        "(P2-11 class-II-CM-naive Stage-1 ablation, label_source=naive)"
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
    outputs=["data/processed/checkpoints/stage1_classII_naive/stage1.pt"],
    env_lock="envs/ml-dna.conda-lock.yml",
    adr="ADR-0004",
    extra={
        "step": "P2-11",
        "report": "reports/p2/train_stage1_classII_naive.json",
        "ablation": (
            "class-II-CM-naive: IDENTICAL to the production scanner "
            "(data/processed/checkpoints/stage1_production, P2-10d'-b) EXCEPT label_source=naive "
            "— every class-II (TBDB001.cm-derived) record's per-nt target is all-background, "
            "class-I records unchanged (ADR-0004 D5; the two checkpoints share one "
            "leakage-controlled partition). Withheld class-II records: 1200 "
            "(labels provenance class_ii_cm_naive_withheld_records)."
        ),
        "label_source": "naive",
        "scored": (
            "GATE-1 anti-mimicry sub-arm at P4, Stage-1-only — NEVER routed through the "
            "class-II-trained production Stage-2 (ADR-0005 D9). NOT GATE-4-graded (P2-14 "
            "grades only the production scanner)."
        ),
        "shipped": "no — retained for reproducibility (PRD §10.1 checkpoint map).",
        "config": "P2-06 winner g0.5_lr3e-4_a0.5 (gamma=0.5, optim.lr=3e-4, class_weight_alpha=0.5)",
        "fold": "full ADR-0004 D5 nested_train (exclude_selection_val=false, ~8,303 records)",
        "negative_mix": (
            "SAME §9.1 seed curriculum as the production twin: negative_fraction=10/11 stream "
            "share over the mined pool + embedded dinuc_shuffled/structured_rna decoys "
            "(ADR-0005 A7 pins 1/3/4-5/6a). Shared for a clean ablation. Realized counts are "
            "in the run's own report."
        ),
        "round": "0 — COLD start (init_from_checkpoint=null).",
        "eval_val": False,
        "epochs": 10,
        "batch_size": 8,
        "world_size": 8,
        "precision": "fp32 (ADR-0002 A7: TF32/cudnn autotune off; NO bf16)",
    },
)
print("wrote data/processed/checkpoints/stage1_classII_naive/provenance.json")
PY

if [ ! -s "$CKPT_DIR/provenance.json" ]; then
  echo "NAIVE_FAIL: checkpoint copied but provenance.json was not written." >&2
  exit 1
fi

touch "$DONE"
echo "NAIVE_DONE job=${SLURM_JOB_ID:-none} checkpoint=$CKPT_DIR/stage1.pt report=$REPORT"
