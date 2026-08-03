#!/bin/bash
# P3-06 — Stage-2 RiNALMo-giga LoRA fine-tune: the production arm + the PRD §11 sweep.
# (PRD §10.2/§10.3/§11; ADR-0002 D2/D3/D5/A2/A4/A8 stack+kernels+env split; ADR-0004 D5 fold;
#  ADR-0005 D16 objective; CLAUDE.md §9.3.)
#
# Six points = loss.aux_weight {0.0, 0.5, 1.0} × optim.lr {1e-4, 3e-4}. Each point is ONE
# array task = ONE node = DDP×8 = one torchrun process, and each saves its own LoRA adapter +
# head state. The `aux_weight=0.0` points are the no-aux arm P3-08 consumes; the point
# `aux1.0_lr1e-4` is the shipped-config PRODUCTION arm (it is exactly conf/train/stage2.yaml's
# defaults). Promotion between points is the validation ladder's job, NOT this file's — this
# run produces arms, it does not pick one.
#
# ── WHY --array, NOT literal Hydra --multirun ────────────────────────────────
# imp.md's Outputs line says "--multirun sweep arms". Each point wants the WHOLE 8-GPU node,
# and Hydra --multirun's basic launcher iterates points IN ONE PROCESS — so a CUDA OOM or a
# NaN-grad in point k poisons every later point and you cannot tell "point k+1 OOMs" from
# "point k+1 OOMs because point k leaked". P2-06 settled this for Stage-1 and the reasoning
# is identical here, so the sweep is a SLURM --array: one task, one point, one process, fully
# isolated. Each task drives a SINGLE Hydra run with the point's overrides — "Hydra-driven",
# just not "--multirun". A failed task writes no <KEY>.OK and is caught by the §9.3
# artifact-based verification (no auto-resubmit).
#
# ── WHAT IS SWEPT, AND WHAT IS NOT ──────────────────────────────────────────
# PRD §11 names γ, LoRA rank, learning rate, window/stride, RC-combination and the Stage-2
# aux-loss weight. Swept here: `loss.aux_weight` and `optim.lr`. NOT swept, each for a stated
# reason: the focal γs are exposed as `loss.*_gamma` but held at their P3-04 defaults;
# window/stride and RC-combination are Stage-1 axes with no Stage-2 analogue (a Stage-2 input
# is a whole locus, not a tiled window); and **LoRA rank is frozen** — PRD §10.3 pins
# LoraConfig(r=16, α=32, dropout=0.05, target_modules="all-linear") and lora_harness holds
# those four as module constants with a drift-guard test asserting config == code, so varying
# r means editing that frozen contract, which is an ADR-0002 question and not a sweep value.
#
# ── SIZING (derived from MEASURED numbers, not guessed) ─────────────────────
# P1-16 (reports/p1/lora_vram_smoke.json) measured this exact stack on a real A4000
# (sm_86, bf16, gradient checkpointing, seq 350 nt): batch 8 → 985.8 ms/step, 2.014 GiB peak;
# batch 16 → 1910.6 ms/step, 2.647 GiB; device total 15.602 GiB; largest fitting batch 16.
# Training population (see the eligibility note below): 9,059 rows. At batch_size=8 over
# DDP×8 that is 142 steps/epoch/rank. Scaling the measured step time by the corpus's real
# token length (289 mean / 475 p95 / 552 max vs the smoke's 350) gives ≈1.35 s/step ⇒
# ≈3.2 min/epoch + ≈0.3 min for the 2,070-row val rung ⇒ ≈35 min for 10 epochs. --time=02:00:00
# is ≈3.4× that. Estimated cost: 6 points × 8 GPUs × ~0.6 h ≈ 29 GPU-h.
# ⚠ NO FRESH GPU SMOKE WAS RUN before the ack: a GPU smoke on this cluster is itself a §9.3
# submit, and the login node `two` is also a gpu compute node (running training off-scheduler
# there is not a thing this project does). The estimate above is an extrapolation from P1-16's
# measurement on the same stack and the same card — stated as such, not as a measurement.
#
# ── THE TRAINING POPULATION (user decision, AskUserQuestion 2026-08-03) ─────
#   admit iff  fold_random == train  AND  NOT calib  AND
#              ( nested_train is True  OR  fold_basis == decoy_pool_random )
# = the ADR-0004 D5 nested fold ∩ the scheme-A train rung, minus P3-02's calib split, plus the
# parentless decoys (structured_rna / gc_background / leader_decoy — built, not carved, so no
# order to hold out). 5,199 corpus + 3,860 decoy = 9,059 rows (4,795 pos / 4,264 neg).
# D5 because PRD §12 grades leave-one-order-out on the TWO-STAGE system; the scheme-A
# restriction so P3-07 fits its temperature on `calib` and P3-10 grades GATE-2's
# in-distribution ECE on THIS checkpoint rather than on an ADR-0004 A6-style eval twin.
# The rule is enforced in stage2/train.py::row_eligibility and two of its refusal counters are
# MUST-FIRE gate clauses — a filter that refuses nothing is one that is not running.
#
# WHERE:   cluster `gpu` partition, EIGHT RTX A4000 (sm_86) per array task. env tbox-ml-rna
#          (ADR-0002 A4 — NOT ml-dna; multimolecule needs transformers 5.x). No --nodelist
#          (never pin; `one` is often down, and cryosparc keeps both nodes fragmented).
# INPUT:   ONE parquet, STAGED by rsync (the ml envs have no dvc, and a checkout does not
#          carry it). It is NOT git-LFS, so the P2 LFS-pointer trap does not apply here — but
#          it IS DVC-tracked, and this body verifies it against the committed .dvc md5 plus
#          PAR1 magic, exiting 2 before any GPU time on a mismatch:
#            rsync -avzP data/processed/stage2_dataset.parquet two:~/tbox-finder/data/processed/
#          src/tbox_finder/stage2/head_vocab.json is git-tracked and rides the checkout.
# SUBMIT:  from the repo root so relative --output/--error resolve. SYNC FIRST, SUBMIT SECOND.
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p3-06-stage2-lora && git reset --hard origin/p3-06-stage2-lora"'
#            # …re-stage the parquet (the reset does not touch it, but a fresh clone has none)…
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p3/stage2_lora_finetune.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p3/stage2_lora_finetune.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b) all SIX
#          reports/p3/sweep/<KEY>.OK markers present, (c) each reports/p3/sweep/<KEY>.json
#          validates (train.py writes only a self-validated report AND raises on a failed
#          gate, so rc≠0 ⇒ no .OK), (d) each checkpoint dir carries lora_adapter/ +
#          stage2_heads.pt + provenance.json. ⚠️ the `.err` is EXPECTED NON-EMPTY on success
#          (torchrun rendezvous, HF download chatter, W&B-offline notices) — scan for real
#          tracebacks, DO NOT gate on zero bytes (the P2-05/P2-06/P2-09 precedent). A missing
#          .OK is a §9.3 stop: surface it, no auto-resubmit.
# RETRIEVE: plain scratch, NOT dvc (the ml env has no dvc; the `dvc add` runs LOCALLY after):
#            rsync -avzP two:~/tbox-finder/data/processed/checkpoints/stage2_rinalmo/ <local>/data/processed/checkpoints/stage2_rinalmo/
#            rsync -avzP two:~/tbox-finder/reports/p3/sweep/ <local>/reports/p3/sweep/

#SBATCH --job-name=tbox-p3-06-stage2-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=reports/p3/sweep/slurm_%A_%a.out
#SBATCH --error=reports/p3/sweep/slurm_%A_%a.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down).
# Each task requests all 8 GPUs of a node, so the scheduler runs at most one task per node
# (≈2 concurrent when both `one` and `two` are up) and queues the rest.

set -eo pipefail   # NOT -u: the conda activate hooks read unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# ── The sweep grid ───────────────────────────────────────────────────────────
# Written as two parallel arrays indexed by SLURM_ARRAY_TASK_ID rather than computed, so the
# point a task ran is readable in this file and in the KEY it writes.
AUX=(0.0  0.0  0.5  0.5  1.0  1.0)
LRS=(1e-4 3e-4 1e-4 3e-4 1e-4 3e-4)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#AUX[@]}" ]; then
  echo "FATAL: array index $IDX exceeds the ${#AUX[@]}-point grid." >&2
  exit 2
fi
AUX_WEIGHT="${AUX[$IDX]}"
LR="${LRS[$IDX]}"
KEY="aux${AUX_WEIGHT}_lr${LR}"

OUT_DIR="reports/p3/sweep"
REPORT="$OUT_DIR/${KEY}.json"
OK="$OUT_DIR/${KEY}.OK"
CKPT_DIR="data/processed/checkpoints/stage2_rinalmo/${KEY}"
mkdir -p "$OUT_DIR"
# Guard the interpolation explicitly rather than with `${CKPT_DIR:?}`: the shipped
# tests/unit/test_sbatch_rm_targets.py gate cannot resolve the `:?` form, so it scores the
# deletion target as a bare wildcard and fails CLOSED against every tracked path — which is
# how a `$(printf …)`-built rm target once left CI red for sessions. An explicit emptiness
# check is the same protection in a form the gate can read.
if [ -z "$KEY" ]; then
  echo "FATAL: the sweep KEY resolved empty; refusing to touch $CKPT_DIR." >&2
  exit 2
fi

# ── Singleton lock, per POINT ────────────────────────────────────────────────
# Per-KEY rather than global: the six points have disjoint outputs and are meant to run
# concurrently. What must not happen is two tasks sharing one point's paths (a re-submitted
# array overlapping a running one), which is exactly what this excludes.
exec 9>"$OUT_DIR/.${KEY}.lock"
if ! flock -n 9; then
  echo "STAGE2_SKIP $KEY: another job holds this point's lock — refusing to share outputs." >&2
  exit 1
fi

# ── Staged-input assertion (fail loud, before any GPU time) ──────────────────
# Existence is not enough: a stale or truncated parquet trains a different population and the
# loss looks perfectly reasonable. Checked for PAR1 magic (catches truncation) AND against the
# committed .dvc md5 (catches staleness).
assert_dvc_artifact() {
  local data_path="$1" dvc_path="$2" want got
  if [ ! -s "$data_path" ]; then
    echo "FATAL: $data_path missing. Stage it from the laptop (see INPUT in this header):" >&2
    echo "  rsync -avzP $data_path two:~/tbox-finder/$(dirname "$data_path")/" >&2
    exit 2
  fi
  if [ "$(head -c 4 "$data_path")" != "PAR1" ]; then
    echo "FATAL: $data_path is not a parquet (no PAR1 magic; $(stat -c %s "$data_path") bytes)." >&2
    exit 2
  fi
  want="$(awk '/^- md5:/ {print $3; exit}' "$dvc_path")"
  got="$(md5sum "$data_path" | awk '{print $1}')"
  if [ -z "$want" ]; then echo "FATAL: could not read an md5 out of $dvc_path" >&2; exit 2; fi
  if [ "$want" != "$got" ]; then
    echo "FATAL: $data_path md5 $got != committed .dvc pointer $want — wrong/stale dataset." >&2
    exit 2
  fi
  echo "staged-ok $data_path md5=$got"
}
assert_dvc_artifact data/processed/stage2_dataset.parquet \
                    data/processed/stage2_dataset.parquet.dvc

# The head vocabulary is git-tracked (P3-03) and rides the checkout, but it is what every
# target index means — a checkout that somehow lacks it would train head widths against
# another head's axis. Assert presence rather than assume the checkout.
if [ ! -s src/tbox_finder/stage2/head_vocab.json ]; then
  echo "FATAL: src/tbox_finder/stage2/head_vocab.json missing from the checkout." >&2
  exit 2
fi

# ── Stale-output clear for THIS point only ───────────────────────────────────
# Deferred to here so a failed staging check cannot destroy a previous good arm. Every path
# below is untracked (reports/p3/sweep/ and the checkpoint dir are both DVC/scratch, not git),
# so no deletion here can dirty a tracked path — which is the condition that turns a run's own
# provenance clause against it.
rm -f "$OK" "$REPORT"
rm -rf "$CKPT_DIR"
mkdir -p "$CKPT_DIR"

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-rna

export CUDA_HOME="$CONDA_PREFIX"          # multimolecule -> danling -> deepspeed probes it at import (ADR-0002 A4/A5).
export WANDB_MODE=offline                 # no live streaming from compute nodes (§9.2).
export PYTHONHASHSEED=0                    # determinism (§8.3); train.py VERIFIES, never sets.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2                   # 16 cpus / 8 ranks; torchrun warns and defaults to 1.

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=$IDX key=$KEY sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# ⚠ NOT named BUILD: `conda activate` exports BUILD=x86_64-conda-linux-gnu from the compiler
# toolchain hook, which clobbered the scratch var and killed job 789 six seconds in.
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}-${IDX}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT   # clean node-local scratch; touches only $JOB_SCRATCH.
export HF_HOME="$JOB_SCRATCH/hf"    # 2.5 GB of weights stay OFF the 87%-full HOME quota (§9.2).
export WANDB_DIR="$JOB_SCRATCH"     # offline W&B curves stay node-local; the JSON report carries the gate.

# ── Import + backbone probe ──────────────────────────────────────────────────
# `import torch` FIRST: multimolecule pulls danling/deepspeed, which probe CUDA_HOME and link
# against torch's libs. This also pre-warms the HF cache ONCE — torchrun's 8 ranks each call
# from_pretrained with no cache_dir, and without a warm cache they race on 8 concurrent
# downloads of the same 2.5 GB pinned revision into one HF_HOME.
# ⚠ NO BACKSLASH-ESCAPED QUOTES INSIDE THIS BLOCK. The body is a single-quoted shell string,
# so a `\"` survives into the Python source and an f-string expression containing one is a
# SyntaxError — the job dies seconds in, after the queue wait, exactly like job 789. The
# first draft of this probe had that defect and it was caught only by executing these bytes
# on the cluster before the ack. Bind to a name first; do not inline the getattr.
#
# ⚠ The warm-load's OWN attention backend is NOT the run's. `load_rinalmo_backbone()` called
# bare passes attn_implementation=None straight to from_pretrained, so transformers picks its
# default (eager here) — whereas the training run goes through `build_peft_model`, which
# SELECTS the backend from the measured sm_86 evidence in reports/p1/kernel_smoke.json. The
# first draft printed the warm-load's value and would have written "attn=eager" into the job
# log for a run that uses FA-2. Both are printed, labelled, so the log cannot mislead.
PYTHONPATH=src python -c '
import torch
from tbox_finder.eval.rinalmo_parity import REPO_ID, REVISION
from tbox_finder.train import lora_harness as LH
model = LH.load_rinalmo_backbone()
warm_attn = getattr(model.config, "_attn_implementation", None)
evidence = LH.read_sm86_flash_attn_evidence()
selected, reason = LH.select_attention_backend(
    flash_attn_importable=evidence["flash_attn_importable"],
    sm86_confirmed=evidence["is_sm86"],
    model_supports_flash_attn=LH.model_supports_flash_attn(),
    dtype=LH.TRAIN_DTYPE,
)
print(f"hf cache warm {REPO_ID} {REVISION} torch={torch.__version__}")
print(f"warm-load attn (NOT the run backend)={warm_attn}")
print(f"run will select attn={selected} :: {reason}")
'

# ── Train this point (DDP×8) ─────────────────────────────────────────────────
# Every token after `-m` is a literal key in conf/train/stage2.yaml (or its /loss and /optim
# groups) — tests/unit/test_sbatch_overrides.py composes them for real, because job 669 died
# after a 14 h queue wait on an override the config lacked. `optim.lr=` NOT `lr=`: the group
# pass runs after the top-level pass in _cfg_from_mapping, so a flat lr would be silently
# overwritten by the group default and every point would train at 1e-4 while faithfully
# reporting its swept value.
# The checkpoint is built in node-local scratch and moved below (§9.2: build in /tmp, move
# only the final artifact).
SCRATCH_CKPT="$JOB_SCRATCH/checkpoint"
echo "── training point $KEY (aux_weight=$AUX_WEIGHT lr=$LR, DDP×8, bf16 LoRA) ──"
set +e
PYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \
  -m tbox_finder.stage2.train \
  loss.aux_weight="$AUX_WEIGHT" optim.lr="$LR" \
  epochs=10 batch_size=8 gradient_checkpointing=true \
  eval_val=true save_checkpoint=true \
  report_path="$REPORT" checkpoint_dir="$SCRATCH_CKPT" \
  tracking.mode=offline
rc=$?
set -e

if [ "$rc" -ne 0 ] || [ ! -s "$REPORT" ]; then
  echo "STAGE2_POINT_FAIL $KEY rc=$rc (report not written or the run failed its own gate — §9.3 sees the missing .OK)" >&2
  [ "$rc" -ne 0 ] || rc=1
  exit "$rc"
fi
if [ ! -d "$SCRATCH_CKPT/lora_adapter" ] || [ ! -s "$SCRATCH_CKPT/stage2_heads.pt" ]; then
  echo "STAGE2_POINT_FAIL $KEY: report passed its gate but no checkpoint was written at $SCRATCH_CKPT." >&2
  exit 1
fi

# ── Move the checkpoint to the repo path + write its provenance sidecar ──────
# Provenance is written HERE, on the node that produced it, so git_sha, env-lock hash, seed
# and the input/output paths are all this run's own. The `dvc add` happens LOCALLY after
# retrieval (no dvc in the ml envs); provenance.json is a small sidecar, not dvc-tracked.
cp -r "$SCRATCH_CKPT/lora_adapter" "$CKPT_DIR/lora_adapter"
cp "$SCRATCH_CKPT/stage2_heads.pt" "$CKPT_DIR/stage2_heads.pt"

PYTHONPATH=src KEY="$KEY" AUX_WEIGHT="$AUX_WEIGHT" LR="$LR" CKPT_DIR="$CKPT_DIR" REPORT="$REPORT" \
python - <<'PY'
import os

from tbox_finder.provenance import write_provenance
from tbox_finder.stage2 import train as T

key = os.environ["KEY"]
ckpt_dir = os.environ["CKPT_DIR"]
write_provenance(
    f"{ckpt_dir}/provenance.json",
    rule=f"slurm/p3/stage2_lora_finetune.sbatch :: {T.ENTRYPOINT} (P3-06 point {key})",
    script="src/tbox_finder/stage2/train.py",
    seed=42,
    inputs=[
        "data/processed/stage2_dataset.parquet",
        "src/tbox_finder/stage2/head_vocab.json",
    ],
    outputs=[
        f"{ckpt_dir}/lora_adapter",
        f"{ckpt_dir}/stage2_heads.pt",
    ],
    env_lock=T.ENV_LOCK,
    adr="ADR-0002; ADR-0004 D5; ADR-0005 D16",
    extra={
        "step": T.STEP,
        "point": key,
        "loss.aux_weight": os.environ["AUX_WEIGHT"],
        "optim.lr": os.environ["LR"],
        "report": os.environ["REPORT"],
        "world_size": 8,
        "precision": "bfloat16 LoRA (PRD §10.3; r=16, alpha=32, dropout=0.05, all-linear)",
        "population": (
            "ADR-0004 D5 nested_train fold INTERSECTED with the scheme-A train rung, minus "
            "the P3-02 calib split, plus the parentless (fold_basis=decoy_pool_random) "
            "decoys — the P3-06 eligibility decision (user sign-off 2026-08-03). The realised "
            "counts are in this point's own report, not restated here."
        ),
    },
)
print(f"wrote {ckpt_dir}/provenance.json")
PY

if [ ! -s "$CKPT_DIR/provenance.json" ]; then
  echo "STAGE2_POINT_FAIL $KEY: checkpoint copied but provenance.json was not written." >&2
  exit 1
fi

touch "$OK"
echo "STAGE2_POINT_OK $KEY report=$REPORT checkpoint=$CKPT_DIR"
