#!/bin/bash
# P2-12 — Stage-1 architecture ablations: RC-combination, head (linear vs CRF), size x context
# (PRD §6 :133/:134, §10.1 :210/:211/:212, §18.1 :358; ADR-0003 A3; ADR-0002 A14;
#  ADR-0005 D15 + D3/A3; CLAUDE.md §9.3).
#
# FOUR new legs, one array task each, on the *executed* P2-06 protocol so the 36 committed
# sweep points stay comparable and the promoted point is the baseline row at 0 new GPU-h:
#   tid 0  rc_combine=gate                       (axis i  — RC hidden-state combination)
#   tid 1  use_crf=true                          (axis iii — head, linear vs +CRF)
#   tid 2  backbone=caduceus-ps-1k-d256-l4       (axis ii  — ~1.93M, seqlen-1k)
#   tid 3  backbone=caduceus-ps-1k-d118-l4       (axis ii  — ~471k,  seqlen-1k)
# The BASELINE (production backbone, rc_combine=concat, use_crf=false) is NOT retrained: it is
# reports/p2/sweep/g0.5_lr3e-4_a0.5.json, the promoted P2-06 point — same fold, same protocol,
# same seed. Retraining it would spend ~1.9 GPU-h reproducing a committed measurement.
#
# ⛔ THE AVERAGED RC ARM IS NOT RUN, AND IS NOT MISSING. imp.md once described an averaged-RC
# "documented negative control"; PRD:133 + PRD:212 + ADR-0005 D15 forbid the order-destroying
# average OUTRIGHT (the §6 strand-resolver reads orientation off predicted element order), and
# models/rc_combine.py::FORBIDDEN_RC_COMBINE raises on it. Document-precedence conflict flagged
# and resolved 2026-07-29: PRD/ADR win. Both arms of axis (i) are directionality-preserving.
#
# WITHHELD, disclosed: (iv) the frozen-embedding probe. PRD:212 words it as a FALLBACK "if
# end-to-end fine-tuning underperforms" — untriggered (ADR-0002 A6 GO, min-core-F1 0.99990) —
# and PRD:358's own P2 enumeration is "ablations (RC, context, head)". probes/frozen_seg_probe.py
# also trains AND grades on the same 300 P4 fine_tune_set loci with no selection_val path and no
# cluster blocking, so it cannot emit a comparable val CI without new code.
#
# ── PROTOCOL: every leg reuses conf/train/stage1_best (INHERITED, not restated) ───────────
# --config-name train/stage1_best is the promoted P2-06 overlay: gamma 0.5 / optim.lr 3e-4 /
# class_weight_alpha 0.5 / epochs 10 / batch_size 8, exclude_selection_val=true (inherited),
# label_source=production, positives-only (negative_fraction 0.0), the ADR-0005 D3+A3 1024/512
# geometry, seed 42. Restating those axes here would fork the promoted config and let the two
# copies drift; the ONLY overrides below are the three ablation axes plus per-leg output paths.
#
# ⚠ `backbone` is a TOP-LEVEL key (conf/train/stage1.yaml), deliberately not a `model.*` one:
# train_stage1._cfg_from_mapping runs its config-group loop AFTER the top-level pass, so a key
# living in both places is silently won by the group — the `lr` trap conf/optim/stage1_best.yaml
# documents. `rc_combine` is the opposite case: it is single-sourced to the /model group, so it
# MUST be overridden as `model.rc_combine.mode=`. Both forms are composed pre-ack in tbox-ml-dna
# and are gated by tests/unit/test_sbatch_overrides.py (the job-669 struct-mode lesson).
#
# ── OUTPUTS GO TO NODE-LOCAL SCRATCH FIRST (the P2-10d'-c lesson) ────────────────────────
# The per-leg JSON reports become GIT-TRACKED at the P2-12 RESULT commit. A job that `rm`s or
# writes a tracked path dirties the tree outside train_stage1._DATA_STAGING_PREFIXES=("data/",),
# so every rank re-derives `provenance_complete` FALSE and the run fails its own gate AFTER
# training (measured: sweep_stage1.sbatch and sizing_smoke.sbatch are both disabled for exactly
# this). So: train with report_path under $JOB_SCRATCH, copy the finished report back only after
# the run completes, and clear only markers under reports/p2/ablation/status/ — a gitignored
# directory whose glob matches no tracked path (tests/unit/test_sbatch_rm_targets.py enforces).
#
# ⚠ THE SCRATCH VARIABLE IS `JOB_SCRATCH`, NEVER `BUILD`. `conda activate tbox-ml-dna` exports
# BUILD=HOST=x86_64-conda-linux-gnu from the conda toolchain's activate hook, silently clobbering
# an sbatch's own $BUILD — that killed job 789 six seconds in, after the ack.
#
# WHERE:   cluster `gpu` partition, EIGHT RTX A4000 (sm_86) per array task, env tbox-ml-dna
#          (ADR-0002 A4). fp32, TF32 disabled (ADR-0002 A7) — no precision knob exists.
# COST:    ADR-0003 A3 — per-leg basis 1.6201 GPU-h train (the heaviest of the 36 uniform
#          1,160-step x ws8 sweep points) + <= 0.065 val-eval; 1.889 GPU-h/leg node-reserved;
#          4 legs => ~9.5 GPU-h node-reserved / ~8.6 active, pre-registered bound <= 18 GPU-h.
#          Measured wall is 0.2025 h/leg at worst, so --time=02:00:00 is ~10x headroom (the CRF
#          and gate per-step costs are UNMEASURED; A3 carries a +25% allowance for them).
#          A2's <= 90 GPU-h *sweep* bound is untouched and is NOT drawn against.
# INPUT:   the THREE corpus parquets are STAGED by rsync, NOT dvc-pulled (the ml envs have no
#          dvc) and NOT carried by the checkout. Stage them ONCE before submitting:
#            rsync -avzP data/interim/flank_context/context_v0.parquet two:~/tbox-finder/data/interim/flank_context/
#            rsync -avzP data/processed/labels/labels_v0.parquet        two:~/tbox-finder/data/processed/labels/
#            rsync -avzP data/processed/splits/split_assignments.parquet two:~/tbox-finder/data/processed/splits/
#          ⚠️ split_assignments.parquet is git-LFS and THIS CLUSTER HAS NO git-lfs — a fresh
#          checkout / `git reset --hard` leaves a 132-byte POINTER (non-empty, so `[ -s ]` passes
#          it) that would train the wrong corpus. Verified below against the committed pointer.
# SUBMIT:  from the repo root so relative --output/--error resolve:
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-12-ablations && git reset --hard origin/p2-12-ablations && \
#              sbatch slurm/p2/ablate_stage1.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/ablate_stage1.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty (all 4 array tasks
#          gone), (b) all 4 reports/p2/ablation/status/leg_<tid>.OK markers present, (c) each
#          reports/p2/ablation/<KEY>.json validates (train_stage1 writes only a self-validated
#          report). ⚠️ the `.err` is EXPECTED NON-EMPTY on success — every measured DDP+W&B run
#          emits 897-3,249 B of offline chatter — so SCAN it for
#          `Traceback|Error|FAILED|assert|OOM|Killed`, DO NOT gate on zero bytes. A missing
#          leg_<tid>.OK is a §9.3 stop (surface it; no auto-resubmit — ADR-0003 D8 governs a
#          bounded re-submit under a SEPARATE ack, and A3's headroom is exactly one).
# RETRIEVE: small git-tracked JSON reports — not DVC:
#            rsync -avzP two:~/tbox-finder/reports/p2/ablation/ <local>/reports/p2/ablation/
# THEN:    PYTHONPATH=src python -m tbox_finder.train.ablation_table --expect-rows 5

#SBATCH --job-name=tbox-p2-12-ablate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=reports/p2/ablation/status/slurm_%A_%a.out
#SBATCH --error=reports/p2/ablation/status/slurm_%A_%a.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).
# Each task requests all 8 GPUs of a node, so the scheduler runs at most one task per node and
# queues the rest. Budget an OPEN-ENDED queue wait: cryosparc fragments both gpu nodes.

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

ABLATE_DIR="reports/p2/ablation"
STATUS_DIR="$ABLATE_DIR/status"
mkdir -p "$ABLATE_DIR" "$STATUS_DIR"

# ── Decode this array task's id → exactly one ablation leg ───────────────────────────────
# Parallel arrays, one entry per leg, indexed by SLURM_ARRAY_TASK_ID. Enforced against the
# table length so the --array range and this table cannot drift apart silently.
PRODUCTION_BACKBONE=caduceus-ps-131k-d256-l16
KEYS=(rc_gate head_crf size_d256_l4 size_d118_l4)
BACKBONES=("$PRODUCTION_BACKBONE" "$PRODUCTION_BACKBONE" caduceus-ps-1k-d256-l4 caduceus-ps-1k-d118-l4)
RC_MODES=(gate concat concat concat)
USE_CRFS=(false true false false)
NLEGS=${#KEYS[@]}
tid="${SLURM_ARRAY_TASK_ID:?this sbatch must be launched as an --array job (no SLURM_ARRAY_TASK_ID)}"
if [ "$tid" -lt 0 ] || [ "$tid" -ge "$NLEGS" ]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID=$tid out of range [0,$NLEGS); --array range disagrees with the leg table." >&2
  exit 2
fi
if [ "${#BACKBONES[@]}" -ne "$NLEGS" ] || [ "${#RC_MODES[@]}" -ne "$NLEGS" ] || [ "${#USE_CRFS[@]}" -ne "$NLEGS" ]; then
  echo "FATAL: the leg table's parallel arrays have different lengths — a leg would silently take another leg's axis." >&2
  exit 2
fi
KEY="${KEYS[$tid]}"
BACKBONE="${BACKBONES[$tid]}"
RC_MODE="${RC_MODES[$tid]}"
USE_CRF="${USE_CRFS[$tid]}"

# Exactly ONE leg may differ from the baseline arm on more than nothing, and none may differ on
# two axes at once — an ablation table cannot attribute a delta to an axis a leg shares with a
# second change. Re-derived here rather than trusted to the table above.
NDIFF=0
[ "$BACKBONE" = "$PRODUCTION_BACKBONE" ] || NDIFF=$((NDIFF + 1))
[ "$RC_MODE" = "concat" ] || NDIFF=$((NDIFF + 1))
[ "$USE_CRF" = "false" ] || NDIFF=$((NDIFF + 1))
if [ "$NDIFF" -ne 1 ]; then
  echo "FATAL: leg $KEY differs from the baseline arm on $NDIFF axes (must be exactly 1);" >&2
  echo "  backbone=$BACKBONE rc_combine=$RC_MODE use_crf=$USE_CRF" >&2
  exit 2
fi

OK="$STATUS_DIR/leg_${tid}.OK"
# Clear only THIS leg's stale success marker, before the failure-prone conda activate / HF
# download, so a crash during import cannot leave a previous run's OK behind (§9.3 keys on
# absence-on-failure). $STATUS_DIR is gitignored and matches no tracked path.
rm -f "$OK"

# ── Staged-input assertions (fail loud, before any GPU time) ─────────────────────────────
# Existence is not enough: (1) a stale/truncated parquet trains a different corpus and the
# numbers look reasonable, so DVC-tracked files are md5'd against their committed .dvc pointer;
# (2) split_assignments.parquet is git-LFS with NO git-lfs on this cluster, so a checkout leaves
# a 132-byte POINTER TEXT — non-empty, `[ -s ]` passes it. Every parquet is checked for `PAR1`
# magic (catches the pointer AND truncation) and against its committed checksum.
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
# smudging), so its `oid sha256` is a committed checksum for the real file. FAIL CLOSED on an
# unreadable / non-pointer object: PAR1 magic alone passes ANY parquet, including a wrong-fold one.
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

# Node-local scratch. NAMED JOB_SCRATCH, never BUILD (the conda-toolchain clobber, job 789).
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT   # touches only node-local scratch, never the checkout.
export WANDB_DIR="$JOB_SCRATCH"     # offline W&B curves stay node-local; the JSON report carries
                                    # the selection metric.

# The report is written HERE, inside the run, and copied into the checkout only afterwards — so
# train_stage1's per-rank `provenance_complete` re-derivation sees a clean tree (see the header).
SCRATCH_REPORT="$JOB_SCRATCH/${KEY}.json"
REPORT="$ABLATE_DIR/${KEY}.json"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}:${SLURM_ARRAY_TASK_ID:-none}"
echo "leg tid=$tid KEY=$KEY backbone=$BACKBONE rc_combine=$RC_MODE use_crf=$USE_CRF"
echo "sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Shared HF cache on HOME (NFS, visible to every array task and both GPU nodes): each pinned
# revision downloads ONCE and every later task cache-hits. HF file locks make the concurrent
# first-tasks safe. Pre-warm ONCE per task (idempotent) before torchrun so the 8 ranks never
# race on a cold cache (the P2-05 pattern).
export HF_HOME="$HOME/tbox-finder-hf-cache"
mkdir -p "$HF_HOME"

# ⚠️ `import torch` BEFORE selective_scan_cuda: the kernel is a C extension linked against torch's
# libc10.so, and nothing puts torch/lib on the loader path until `import torch` dlopens it — so a
# bare `import selective_scan_cuda` dies with "libc10.so: cannot open shared object file" EVEN WHEN
# THE KERNEL IS FINE. That killed job 567. The kernel is a capability (ADR-0002 A2 C2): its absence
# would make every run a ~871x slower silent fallback, so assert it, don't hope.
python -c 'import torch; import selective_scan_cuda; print(f"selective_scan_cuda OK (torch {torch.__version__})")'

# Warm THIS leg's checkpoint through the allow-list resolver, so a leg whose backbone key is not
# pinned fails here (seconds, on the resolver) rather than after 8 ranks have spun up. The
# parameter count and d_model are checked against ADR-0002 A14 inside load_caduceus_ps.
PYTHONPATH=src BACKBONE="$BACKBONE" python -c '
import os
from tbox_finder.models.backbone_registry import resolve_checkpoint
from transformers import AutoModel
spec = resolve_checkpoint(os.environ["BACKBONE"])
m = AutoModel.from_pretrained(spec.repo_id, revision=spec.revision, trust_remote_code=True)
n = int(sum(p.numel() for p in m.parameters()))
assert n == spec.expected_param_count, f"{spec.key}: measured {n} != pinned {spec.expected_param_count}"
assert int(m.config.d_model) == spec.d_model, f"{spec.key}: d_model {m.config.d_model} != {spec.d_model}"
print(f"hf cache warm {spec.key} {spec.repo_id}@{spec.revision} params={n} d_model={spec.d_model}")
'

# ── Train this leg (DDP x8) + evaluate on the P2-06a selection_val rung ──────────────────
# --config-name train/stage1_best inherits the promoted P2-06 axes (gamma 0.5 / optim.lr 3e-4 /
# class_weight_alpha 0.5 / epochs 10 / batch 8 / exclude_selection_val=true / positives-only).
# steps_per_epoch/max_records stay null ⇒ the FULL 7,472-record inner_train fold. eval_val=true /
# eval_max_records=null ⇒ the full 830-record selection_val fold (full_fold=true), so
# ablation_table does not reject the leg. save_checkpoint=false: this table ranks on the val
# metric in the report, and P2-14 owns the shipped-config choice. Only rank 0 writes the report.
echo "── training leg $KEY ──"
set +e
PYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \
  -m tbox_finder.train.train_stage1 --config-name train/stage1_best \
  backbone="$BACKBONE" model.rc_combine.mode="$RC_MODE" use_crf="$USE_CRF" \
  eval_val=true eval_max_records=null save_checkpoint=false \
  report_path="$SCRATCH_REPORT" checkpoint_dir="$JOB_SCRATCH/ckpt" \
  tracking.mode=offline
rc=$?
set -e

if [ "$rc" -ne 0 ] || [ ! -s "$SCRATCH_REPORT" ]; then
  echo "ABLATION_LEG_FAIL $KEY rc=$rc (report not written — this leg is NOT counted; §9.3 sees the missing .OK)" >&2
  [ "$rc" -ne 0 ] || rc=1
  exit "$rc"
fi

# Verify the report describes the leg that was REQUESTED before publishing it. A silently-dropped
# override is the P2-11 label_source footgun class: the run would train the production backbone
# with a concat head and self-consistently report a green gate under the name "size_d118_l4".
PYTHONPATH=src SCRATCH_REPORT="$SCRATCH_REPORT" WANT_BACKBONE="$BACKBONE" \
  WANT_RC="$RC_MODE" WANT_CRF="$USE_CRF" python -c '
import json, os
r = json.load(open(os.environ["SCRATCH_REPORT"]))
cfg = r["diagnostics"]["config"]
bb = r["backbone"]
want_crf = os.environ["WANT_CRF"] == "true"
assert cfg["backbone"] == os.environ["WANT_BACKBONE"], (cfg["backbone"], os.environ["WANT_BACKBONE"])
assert bb["key"] == os.environ["WANT_BACKBONE"], (bb["key"], os.environ["WANT_BACKBONE"])
assert cfg["rc_combine"] == os.environ["WANT_RC"], (cfg["rc_combine"], os.environ["WANT_RC"])
assert cfg["use_crf"] is want_crf, (cfg["use_crf"], want_crf)
assert bb["measured_param_count"] == bb["expected_param_count"], bb
assert r["gate"]["overall_pass"] is True, r["gate"]
key, params = bb["key"], bb["measured_param_count"]
rc, crf = cfg["rc_combine"], cfg["use_crf"]
print("leg-axes OK backbone={} params={} rc={} crf={}".format(key, params, rc, crf))
'

cp "$SCRATCH_REPORT" "$REPORT"
touch "$OK"
echo "ABLATION_LEG_OK $KEY report=$REPORT"
