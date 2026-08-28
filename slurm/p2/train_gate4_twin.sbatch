#!/bin/bash
# P2-14 — the GATE-4 EVAL TWIN: the production protocol with the graded fold withheld.
# (PRD §2.3/§10.1/§12; ADR-0004 D5 fold + D6 gate + A6 population; ADR-0005 D3/A3 geometry
#  + A7 negative mix; ADR-0003 D7/A2 budget; CLAUDE.md §9.3.)
#
# A byte-for-byte clone of `slurm/p2/train_production.sbatch` with **ONE changed override**
# — `exclude_gate4_eval=true` — plus re-pointed outputs and a publish guard. Everything
# else (config, epochs, batch size, negative mix, precision, seed, world size) is held, so
# the twin differs from the shipped checkpoint in exactly one thing: it did not train on
# the 1,204 records GATE-4 grades (8,303 → 7,099; whole clusters, 1,031 of them).
#
# ── WHY A TWIN EXISTS (ADR-0004 A6; user sign-off 2026-07-30) ────────────────
# ADR-0004 D6 gates the min per-element per-nt F1 on "the in-distribution homology-clustered
# (genus-stratified) split". The shipped P2-10d′-b production checkpoint trained on the FULL
# D5 fold (`exclude_selection_val=false`, 8,303 records) — so it has **no** in-distribution
# population held out from it at all:
#   * scheme-A `test` is 51.2% inside the training fold (1,204 / 2,353 corpus rows);
#   * of the 1,149 corpus rows that are not, 796 are LOO / Actinobacteria / anchor holdouts —
#     the generalization arm, which D6 does not gate and P4 owns;
#   * the remaining 353 are clade-crossing debris (271) + the no-clade D4 residue (82),
#     which `splits.py` marks never-scored;
#   * `selection_val` is both inside the training stream AND the fold the P2-06 sweep
#     selected the config on.
# So GATE-4 grades this twin, the SHIPPED checkpoint stays the full-fold one (it saw
# strictly more data, making the twin's grade a CONSERVATIVE proxy), and the difference is
# the two-run split PRD §10.1 requires the model card to disclose.
#
# ── NO VAL METRIC, SAME AS PRODUCTION ────────────────────────────────────────
# `exclude_selection_val=false` is HELD from the production protocol (changing it would make
# the twin differ in two things and its grade non-comparable), so `eval_val=false` is
# required alongside it — `Stage1TrainConfig.__post_init__` refuses the pair. The twin's
# held-out number is GATE-4 itself, computed afterwards by `eval/gate4.py`, not here.
#
# WHERE:   cluster `gpu` partition, EIGHT RTX A4000 (sm_86) on ONE node. env tbox-ml-dna
#          (ADR-0002 A4). No --nodelist (never pin; `one` is often down, and cryosparc keeps
#          both nodes fragmented — budget an open-ended queue wait).
# COST:    MEASURED, from the shipped production run's own report
#          (`reports/p2/train_stage1_production.json`: 14,270 steps, Σstep 8,467.2 s +
#          Σwait 36.4 s = 2.36 h wall = 18.9 GPU-h on 8,303 positives). The twin's step
#          count scales with the positive pool: 14,270 × 7,099/8,303 ≈ 12,201 steps ×
#          0.5946 s/step ≈ 7,255 s ≈ **2.02 h wall ≈ 16.1 GPU-h**. `--time=05:00:00` keeps
#          the production headroom (gpu TIMELIMIT is infinite, so a longer wall is free).
# INPUT:   the SAME FIVE parquets production stages by rsync (the ml envs have no dvc; a
#          checkout does not carry them). Stage them ALL once before submitting:
#            rsync -avzP data/interim/flank_context/context_v0.parquet two:~/tbox-finder/data/interim/flank_context/
#            rsync -avzP data/processed/labels/labels_v0.parquet        two:~/tbox-finder/data/processed/labels/
#            rsync -avzP data/processed/splits/split_assignments.parquet two:~/tbox-finder/data/processed/splits/
#            rsync -avzP data/processed/negatives/mining_pool_v0.parquet two:~/tbox-finder/data/processed/negatives/
#            rsync -avzP data/processed/negatives/decoys_v0.parquet      two:~/tbox-finder/data/processed/negatives/
#          ⚠️ split_assignments.parquet is git-LFS and THIS CLUSTER HAS NO git-lfs — a fresh
#          checkout / `git reset --hard` leaves a 132-byte POINTER (non-empty, so `[ -s ]`
#          passes it) that would train the wrong corpus AND define a different graded fold.
#          Every parquet is verified against its committed pointer (md5 for .dvc, sha256 for
#          LFS) + PAR1 magic below, exit 2 on mismatch.
# SUBMIT:  from the repo root so relative --output/--error resolve. SYNC FIRST, SUBMIT
#          SECOND — never in one `&&` chain: `git reset --hard` rewrites the LFS-pointered
#          split table back to its 132-byte pointer, undoing the staging the job verifies.
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-14-gate4 && git reset --hard origin/p2-14-gate4"'
#            # …re-stage ALL FIVE parquets (see INPUT above)…
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/train_gate4_twin.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/train_gate4_twin.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b)
#          reports/p2/train_stage1_gate4_twin.DONE present, (c) the checkpoint +
#          provenance.json present, (d) the report validates AND its own fold scope reads
#          `gate4_twin_train` with `n_gate4_eval_excluded > 0` (the publish guard below
#          already refuses otherwise). ⚠️ the `.err` is EXPECTED NON-EMPTY on success
#          (transformers/Caduceus trust_remote_code notes, torchrun rendezvous + W&B-offline
#          chatter) — scan for real tracebacks, DO NOT gate on zero bytes. A missing DONE is
#          a §9.3 stop (surface it; no auto-resubmit).
# RETRIEVE: the checkpoint + its sidecar (plain scratch, NOT dvc — the ml env has no dvc;
#          the `dvc add` runs LOCALLY on the laptop after retrieval), plus the report:
#            rsync -avzP two:~/tbox-finder/data/processed/checkpoints/stage1_gate4_twin/ <local>/data/processed/checkpoints/stage1_gate4_twin/
#            rsync -avzP two:~/tbox-finder/reports/p2/train_stage1_gate4_twin.json <local>/reports/p2/

#SBATCH --job-name=tbox-p2-14-gate4-twin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=reports/p2/train_gate4_twin_%j.out
#SBATCH --error=reports/p2/train_gate4_twin_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).

set -eo pipefail   # NOT -u: the cuda-nvcc conda activate hook reads unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

OUT_DIR="reports/p2"
REPORT="$OUT_DIR/train_stage1_gate4_twin.json"
DONE="$OUT_DIR/train_stage1_gate4_twin.DONE"
CKPT_DIR="data/processed/checkpoints/stage1_gate4_twin"
mkdir -p "$OUT_DIR" "$CKPT_DIR"

# ── Singleton lock ───────────────────────────────────────────────────────────
exec 9>"$OUT_DIR/.train_gate4_twin.lock"
if ! flock -n 9; then
  echo "GATE4_TWIN_SKIP: another job holds the lock — refusing to share outputs." >&2
  exit 1
fi

# ── Staged-input assertions (fail loud, before any GPU time OR any destructive clear) ────
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

# FAIL CLOSED on an unreadable / non-pointer object (the P2-06 fix: a version that returned
# "staged-ok" whenever the pointer could not be read left PAR1 magic — which passes ANY
# parquet — as the only check).
assert_lfs_artifact() {
  local data_path="$1" ptr want_oid want_size got_oid got_size want_blob got_blob
  assert_parquet_magic "$data_path"
  # ⚠ FAIL CLOSED on an unverifiable object. An earlier version returned "staged-ok" whenever
  # the pointer could not be read, leaving PAR1 magic as the only check — and PAR1 magic passes
  # ANY parquet, including a stale or wrong-fold one. A staged-input guard that degrades to
  # "it's a parquet" is precisely the shape that burns a submit-ack: the job starts, trains on
  # the wrong table, and reports a self-consistent result.
  #
  # Which representation is committed? Ask git's ATTRIBUTES rather than reading the object.
  # Since PR #135 this table is an ordinary git blob, so `git show` on it would pour 5 MB of
  # binary through a command substitution — which bash truncates at the first null byte, with
  # a warning in every job log.
  if git check-attr filter -- "$data_path" 2>/dev/null | grep -q 'filter: lfs$'; then
    # git-LFS-tracked: the committed object is a ~130-byte pointer whose `oid sha256` IS the
    # content hash (git-lfs spec), so it is a committed checksum for the real file.
    if ! ptr="$(git show "HEAD:$data_path" 2>/dev/null)"; then
      echo "FATAL: could not read the committed LFS pointer for $data_path (git show failed)." >&2
      exit 2
    fi
    want_oid="$(printf '%s\n' "$ptr" | awk -F'sha256:' '/^oid sha256:/ {print $2; exit}')"
    want_size="$(printf '%s\n' "$ptr" | awk '/^size / {print $2; exit}')"
    if [ -z "$want_oid" ] || [ -z "$want_size" ]; then
      echo "FATAL: $data_path is git-LFS-tracked but its committed object is not a readable" >&2
      echo "  pointer, so there is no committed checksum to verify the staged file against." >&2
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
    return 0
  fi
  # An ordinary git blob — the NORMAL case for this table since PR #135, which moved it out of
  # git-LFS so CI stops paying metered LFS bandwidth. That also means this cluster (which has
  # no git-lfs binary) now receives the real bytes from a plain checkout instead of a 132-byte
  # pointer. Verify content-addressed against the commit: a git blob id hashes the exact bytes,
  # the same strength the pointer's `oid sha256` gave. Until this branch existed the clause
  # above exited 2 here, so every one of these jobs died on submit once the table left LFS.
  # --verify --quiet: a bare `git rev-parse HEAD:<path>` ECHOES the unresolvable argument to
  # stdout instead of failing, which would set want_blob to the literal "HEAD:<path>" and
  # report a bogus mismatch rather than the honest "nothing to verify against" below.
  want_blob="$(git rev-parse --verify --quiet "HEAD:$data_path" 2>/dev/null || true)"
  got_blob="$(git hash-object -- "$data_path" 2>/dev/null || true)"
  if [ -z "$want_blob" ] || [ -z "$got_blob" ]; then
    echo "FATAL: $data_path is neither git-LFS-tracked nor a resolvable git blob, so there is" >&2
    echo "  no committed checksum to verify the staged file against." >&2
    exit 2
  fi
  if [ "$want_blob" != "$got_blob" ]; then
    echo "FATAL: $data_path does not match its committed git blob." >&2
    echo "  want blob=$want_blob" >&2
    echo "  got  blob=$got_blob" >&2
    exit 2
  fi
  echo "staged-ok $data_path blob=$got_blob"
}

assert_dvc_artifact data/interim/flank_context/context_v0.parquet \
                    data/interim/flank_context/context_v0.parquet.dvc
assert_dvc_artifact data/processed/labels/labels_v0.parquet \
                    data/processed/labels/labels_v0.parquet.dvc
# ⚠ THE load-bearing one for THIS job: the split table does not merely select the training
# corpus, it DEFINES the graded fold (`fold_random == "test"` ∧ `nested_train`). A stale or
# pointer-stubbed table would withhold a different population than `eval/gate4.py` later
# grades, and both halves would look internally consistent.
assert_lfs_artifact data/processed/splits/split_assignments.parquet
assert_dvc_artifact data/processed/negatives/mining_pool_v0.parquet \
                    data/processed/negatives/mining_pool_v0.parquet.dvc
assert_dvc_artifact data/processed/negatives/decoys_v0.parquet \
                    data/processed/negatives/decoys_v0.parquet.dvc

# ── Stale-output clear — ONLY now that the staged inputs are verified ─────────
# Deferred to here (not at lock acquisition) so a failed staging check cannot destroy a
# previously successful twin. ⚠️ "$REPORT" IS DELIBERATELY *NOT* CLEARED: once committed it
# is a git-tracked path, and deleting it dirties the tree outside `_DATA_STAGING_PREFIXES`,
# which makes `_provenance_complete` re-derive FALSE → `overall_pass` FALSE → the checkpoint
# is never saved → the EXIT trap wipes the scratch dir with it. That is the entire run
# destroyed at the finish line (P2-10d′-c/e). Freshness is proven by the md5 fingerprint
# below instead. Do NOT "fix" this by widening `_DATA_STAGING_PREFIXES`.
rm -f "$DONE" "$CKPT_DIR/stage1.pt" "$CKPT_DIR/provenance.json"

# ── RESTORE the tracked report to HEAD — idempotent re-runs (P2-10d′-e) ───────
# A successful run overwrites $REPORT in place, so after run #1 a committed checkout is
# permanently dirty there and run #2 would re-derive `provenance_complete` FALSE on every
# rank. Restore to the committed bytes (never `rm` — deleting dirties the tree just the
# same). No-op while the report is still untracked (before the RESULT commit lands it).
if git ls-files --error-unmatch "$REPORT" >/dev/null 2>&1; then
  git checkout HEAD -- "$REPORT"
fi

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-dna

export CUDA_HOME="$CONDA_PREFIX"          # deepspeed/nvcc probe CUDA_HOME at import (ADR-0002 A4).
export WANDB_MODE=offline                 # no live streaming from compute nodes (§9.2).
export PYTHONHASHSEED=0                    # determinism (§8.3); train_stage1 VERIFIES, never sets.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2                   # 16 cpus / 8 ranks; torchrun warns and defaults to 1.

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# ⚠ NOT named BUILD: `conda activate tbox-ml-dna` exports BUILD=x86_64-conda-linux-gnu from
# the compiler-toolchain hook, which clobbered the scratch var and killed job 789 six
# seconds in ([[conda-toolchain-clobbers-build-host]]).
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT   # clean node-local scratch; touches only $JOB_SCRATCH.
export HF_HOME="$JOB_SCRATCH/hf"    # 2.5 GB of weights stay OFF the 87%-full HOME quota (§9.2).
export WANDB_DIR="$JOB_SCRATCH"

# ⚠️ `import torch` BEFORE selective_scan_cuda: the kernel is a C extension linked against
# torch's libc10.so, and nothing puts torch/lib on the loader path until `import torch`
# dlopens it. That killed job 567. Its absence would make training ~871× slower in silence,
# so assert it, don't hope.
python -c 'import torch; import selective_scan_cuda; print(f"selective_scan_cuda OK (torch {torch.__version__})")'

# ── Pre-warm the HF cache ONCE (8 ranks would otherwise race on one download) ──
PYTHONPATH=src python -c '
from tbox_finder.models.caduceus_backbone import REPO_ID, REVISION
from transformers import AutoModel
AutoModel.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)
print("hf cache warm", REPO_ID, REVISION)
'

# ── Train the GATE-4 eval twin (DDP×8) ───────────────────────────────────────
# Identical to slurm/p2/train_production.sbatch except for `exclude_gate4_eval=true`.
JOB_CKPT="$JOB_SCRATCH/checkpoint"
REPORT_BEFORE="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"
echo "── training the P2-14 GATE-4 eval twin (D5 fold MINUS the graded clusters, DDP×8, fp32) ──"
set +e
PYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \
  -m tbox_finder.train.train_stage1 \
  gamma=0.5 optim.lr=3e-4 class_weight_alpha=0.5 \
  epochs=10 batch_size=8 gradient_checkpointing=true \
  exclude_selection_val=false eval_val=false save_checkpoint=true \
  exclude_gate4_eval=true \
  negative_pool_parquet=data/processed/negatives/mining_pool_v0.parquet \
  negative_decoy_parquet=data/processed/negatives/decoys_v0.parquet \
  negative_fraction=0.9090909090909091 \
  init_from_checkpoint=null \
  report_path="$REPORT" checkpoint_dir="$JOB_CKPT" \
  tracking.mode=offline
rc=$?
set -e

if [ "$rc" -ne 0 ] || [ ! -s "$REPORT" ]; then
  echo "GATE4_TWIN_FAIL rc=$rc (report not written or process failed its own gate — §9.3 sees the missing DONE)" >&2
  [ "$rc" -ne 0 ] || rc=1
  exit "$rc"
fi
REPORT_AFTER="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"
if [ -z "$REPORT_AFTER" ]; then
  echo "GATE4_TWIN_FAIL: could not fingerprint $REPORT after the run (missing or unreadable)." >&2
  exit 1
fi
if [ "$REPORT_BEFORE" = "$REPORT_AFTER" ]; then
  echo "GATE4_TWIN_FAIL: $REPORT is byte-identical to the one that was there before this run" >&2
  echo "  (md5 $REPORT_AFTER) — rank 0 did not write it, so it describes some OTHER run." >&2
  exit 1
fi
if [ ! -s "$JOB_CKPT/stage1.pt" ]; then
  echo "GATE4_TWIN_FAIL: report passed its gate but no checkpoint was written at $JOB_CKPT/stage1.pt." >&2
  exit 1
fi

# ── PUBLISH GUARD: re-derive the twin property from the report, off the artifact ──
# The P2-11 idiom. `exclude_gate4_eval=true` above is a REQUEST; this checks the run's own
# report for the MEASURED result, from two independently-written fields, and refuses to
# publish a checkpoint that is not a twin. Without it, a silently-dropped override (the
# job-669 / P2-11 `label_source` footgun) would publish a full-fold checkpoint under the
# twin's name, and `eval/gate4.py` would then grade a model on its own training data and
# emit a number indistinguishable from a real one (CLAUDE.md §10.3).
PYTHONPATH=src python - "$REPORT" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
scope = report.get("class_counts_scope") or {}
cfg = (report.get("diagnostics") or {}).get("config") or {}
problems = []
if scope.get("fold_scope") != "gate4_twin_train":
    problems.append(f"class_counts_scope.fold_scope = {scope.get('fold_scope')!r}, want 'gate4_twin_train'")
if scope.get("exclude_gate4_eval") is not True:
    problems.append(f"class_counts_scope.exclude_gate4_eval = {scope.get('exclude_gate4_eval')!r}, want True")
if cfg.get("exclude_gate4_eval") is not True:
    problems.append(f"diagnostics.config.exclude_gate4_eval = {cfg.get('exclude_gate4_eval')!r}, want True")
n_ex = scope.get("n_gate4_eval_excluded")
if not isinstance(n_ex, int) or isinstance(n_ex, bool) or n_ex <= 0:
    problems.append(f"class_counts_scope.n_gate4_eval_excluded = {n_ex!r}, want a positive int")
# ...and a positive count cannot tell 1,204 from 900. ADR-0004 A6 pins the carve by
# measurement (8,303 -> 7,099, withholding 1,204 records in 1,031 clusters); a drifted split
# table would train a DIFFERENT twin that still satisfies every rule-shaped clause above, so
# the checkpoint is not published unless its own report re-derives the signed numbers. The
# same pins gate the grade in eval.gate4.gate4_problems; this one just refuses earlier.
# IMPORTED, never retyped: two copies of a pinned number is how one gets amended and the
# other ships the old value ([[promote-dont-duplicate-is-a-correctness-rule]]).
from tbox_finder.data.window_dataset import (
    A6_N_TWIN_EXCLUDED_CLUSTERS,
    A6_N_TWIN_EXCLUDED_RECORDS,
    A6_N_TWIN_TRAIN_RECORDS,
)

for key, pinned in (
    ("n_gate4_eval_excluded", A6_N_TWIN_EXCLUDED_RECORDS),
    ("n_gate4_eval_clusters", A6_N_TWIN_EXCLUDED_CLUSTERS),
    ("n_records", A6_N_TWIN_TRAIN_RECORDS),
):
    if scope.get(key) != pinned:
        problems.append(f"class_counts_scope.{key} = {scope.get(key)!r}, want {pinned} (ADR-0004 A6)")
if scope.get("label_source") != "production":
    problems.append(f"class_counts_scope.label_source = {scope.get('label_source')!r}, want 'production'")
if problems:
    print("GATE4_TWIN_FAIL: the run's own report does not describe a GATE-4 eval twin:", file=sys.stderr)
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    raise SystemExit(1)
print(f"twin-ok fold_scope=gate4_twin_train n_gate4_eval_excluded={n_ex} n_records={scope.get('n_records')}")
PY

# ── Move the checkpoint to the repo path + write its provenance sidecar ───────
cp "$JOB_CKPT/stage1.pt" "$CKPT_DIR/stage1.pt"
PYTHONPATH=src python - <<'PY'
from tbox_finder.provenance import write_provenance

write_provenance(
    "data/processed/checkpoints/stage1_gate4_twin/provenance.json",
    rule="slurm/p2/train_gate4_twin.sbatch :: train_stage1 (P2-14 GATE-4 eval twin)",
    script="src/tbox_finder/train/train_stage1.py",
    seed=42,
    inputs=[
        "data/interim/flank_context/context_v0.parquet",
        "data/processed/labels/labels_v0.parquet",
        "data/processed/splits/split_assignments.parquet",
        "data/processed/negatives/mining_pool_v0.parquet",
        "data/processed/negatives/decoys_v0.parquet",
    ],
    outputs=["data/processed/checkpoints/stage1_gate4_twin/stage1.pt"],
    env_lock="envs/ml-dna.conda-lock.yml",
    adr="ADR-0004 A6 (graded population + eval twin); ADR-0002 (backbone + precision)",
    extra={
        "step": "P2-14",
        "report": "reports/p2/train_stage1_gate4_twin.json",
        "config": "P2-06 winner g0.5_lr3e-4_a0.5 (gamma=0.5, optim.lr=3e-4, class_weight_alpha=0.5)",
        "fold": (
            "gate4_twin_train — the full ADR-0004 D5 nested_train fold MINUS the whole "
            "clusters of the GATE-4 graded population (scheme-A test inside nested_train): "
            "8,303 -> 7,099 records, 1,031 clusters withheld"
        ),
        "graded_by": (
            "GATE-4 (ADR-0004 D6) via src/tbox_finder/eval/gate4.py on "
            "window_dataset.load_gate4_eval_records — 1,201 records / 1,029 clusters"
        ),
        "shipped": (
            "NO. This checkpoint exists to be GRADED, not shipped: the shipped production "
            "scanner stays the full-fold P2-10d'-b artifact (user decision 2026-07-30), "
            "which trained on 1,204 more records and is therefore NOT gradeable on this "
            "population. PRD §10.1's two-run split disclosure."
        ),
        "differs_from_production_by": "exclude_gate4_eval=true — one override, nothing else",
        "eval_val": False,
        "epochs": 10,
        "batch_size": 8,
        "world_size": 8,
        "precision": "fp32 (ADR-0002 A7: TF32/cudnn autotune off; NO bf16)",
    },
)
print("wrote data/processed/checkpoints/stage1_gate4_twin/provenance.json")
PY

if [ ! -s "$CKPT_DIR/provenance.json" ]; then
  echo "GATE4_TWIN_FAIL: checkpoint copied but provenance.json was not written." >&2
  exit 1
fi

touch "$DONE"
echo "GATE4_TWIN_DONE job=${SLURM_JOB_ID:-none} checkpoint=$CKPT_DIR/stage1.pt report=$REPORT"
