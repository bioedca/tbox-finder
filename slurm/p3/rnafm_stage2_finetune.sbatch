#!/bin/bash
# P3-17 — the RNA-FM Stage-2 COMPARATOR: fine-tune + score, one arm, one node.
# (PRD §6 ablation / §10.2 condition (c) / §10.3 / §12; ADR-0002 D6 + **A15**; ADR-0005 D17;
#  CLAUDE.md §9.3.)
#
# ── WHAT THIS IS, AND WHAT IT IS NOT ────────────────────────────────────────
# This builds the arm PRD §10.2's condition (c) has to be *measured against*: "post-calibration
# leave-clade-out ECE is worse than RNA-FM's by a pre-registered numeric margin" (ADR-0005 D17:
# > 0.02 absolute ECE, sustained across the held-out-order distribution). Until now there was no
# RNA-FM number, so condition (c) was unreadable — not passing, unreadable.
#
# It does NOT fire the swap. P3-18 reads the margin; firing it substitutes the shipped Stage-2
# backbone and drags the whole absolute GATE-1/GATE-2 re-clearance and the P4→P5 go/no-go with
# it (PRD §2.3/§10.2, ADR-0002 D6), which is a CLAUDE.md §7 decision on top of the number.
#
# ── TWO POINTS, AND THE SECOND ONE IS NOT OPTIONAL ──────────────────────────
# imp.md asks for "the identical heads/losses/calibration stack so the ECE comparison is
# apples-to-apples". So this trains EXACTLY the production arm's configuration and changes
# exactly one thing: the backbone. `aux_weight=1.0, lr=1e-4` are conf/{loss,optim}/stage2.yaml's
# shipped defaults — the `aux1.0_lr1e-4` point — and `batch_size=4, gradient_accumulation_steps=2`
# are carried over unchanged rather than re-tuned. ⚠ THAT IS DELIBERATE EVEN THOUGH RNA-FM WOULD
# FIT A LARGER BATCH (measured: 2.02 of 15.6 GiB at batch 4, vs RiNALMo's 8.18): batch size
# changes the optimisation, and an arm that differed from the production one on batch AND
# backbone could not attribute an ECE difference to either. The sizing leg below therefore
# VERIFIES batch 4 fits; it does not go shopping for a bigger one.
#
# ⚠ AND IT ALSO TRAINS THE aux_weight=0 CONTROL, BECAUSE THE SCORERS REQUIRE IT. Job 1372
# trained the production point cleanly and then BOTH scoring legs refused:
#
#   ValueError: expected exactly one aux_weight=0 arm at lr=0.0001, found [];
#   the ablation needs a learning-rate-matched control
#
# `stage2.eval` is the P3-08 aux-ablation driver — it resolves `select_arm_pair` before it
# scores anything and builds an ablation report afterwards — and `calib.gate2 score-loo` calls
# the SAME selector. A single-arm checkpoint root cannot be scored by either shipped entry
# point. Training the lr-matched control is therefore the enabling condition for the posteriors
# this step owes, not a side quest: it lets both legs run COMPLETELY UNCHANGED, so "reuses the
# identical harness" is literally true rather than nearly true, and it touches nothing in the
# P3-08 machinery that produces committed artifacts. It costs one more ~13-minute run, and it
# answers ADR-0005 D16's aux-loss question for BOTH backbones instead of one.
#
# The two points run SEQUENTIALLY in one job rather than as a `--array`: they share this node,
# the scoring legs need both to exist, and one job means one provenance chain and one ack. The
# P3-06 array exists because six whole-node points cannot share a node; two 13-minute points on
# one node can.
#
# The arm is laid out to be structurally identical to a production sweep point —
#   checkpoint  data/processed/checkpoints/stage2_rnafm/<KEY>/
#   report      reports/p3/sweep_rnafm/<KEY>.json
# — because `stage2.eval.discover_arms` joins `<checkpoint_root>/<ARM>/` to
# `<sweep_dir>/<ARM>.json` and reads `config.lr` / `loss.aux_weight` out of the arm's own report.
# Mirroring the layout means the whole downstream scoring + calibration stack runs UNCHANGED
# against `--checkpoint-root`/`--sweep-dir`, which is what makes "identical harness" a fact
# rather than a claim.
#
# ── ⚠ THIS ARM DOES NOT RUN IN `tbox-ml-rna`, AND THAT IS THE WHOLE ADR-0002 A15 STORY ─────
# `multimolecule` 0.1.0 — the A8 pin `ml-rna` carries, and the pin every shipped Stage-2 artifact
# was produced under — CANNOT LOAD `multimolecule/rnafm` at all. Its
# `models/rnafm/configuration_rnafm.py:136` computes `expected_vocab_size = 131 if codon else 26`
# and raises, while the published checkpoint declares `vocab_size: 28` with a 28-line vocab.txt:
#     ValueError: vocab_size (28) must be 26 when codon=False
# and there is no older checkpoint revision to pin around it (the repo has three commits, all
# 2026-05-31). A15 added `envs/ml-rnafm.yml` (`multimolecule` 0.2.0, every other pin identical)
# rather than bumping `ml-rna` and firing CLAUDE.md §8.5 across every published RiNALMo number.
# So this file activates **tbox-ml-rnafm**, and asserts the version before spending GPU time —
# activating the wrong env is a one-word mistake with a very confusing failure mode.
# ⚠ The disclosure that buys: the two arms are compared across a one-minor-version library
# difference. P3-18's report must say so (A15); tokenisation is provably identical (below), so
# the difference is not in what the models read.
#
# ── WHAT MAKES THE COMPARISON CLEAN, MEASURED RATHER THAN ASSUMED ───────────
# RNA-FM's vocab.txt at the pinned revision is BYTE-IDENTICAL to rinalmo-giga's and to
# `src/tbox_finder/stage2/tokenizer.py::VOCAB` — same 28 tokens, same order, same six specials,
# same <cls>…<eos> 2-flanking layout that `strip_special_tokens` assumes. Verified 2026-08-21 by
# `T.VOCAB == RnaTokenizer.from_pretrained(...).get_vocab()` and by an id-level encode
# comparison on a real-length locus. So the shared tokenizer mirror needs no second table and
# the ECE comparison is not confounded by tokenisation. RNA-FM: hidden 640, 12 layers, 20 heads,
# absolute positions, max_position_embeddings 1026 (≥ the mirror's pinned 1024, so the 1022-nt
# ceiling stays conservative), 99,111,680 backbone parameters.
#
# ── SIZING — MEASURED IN-JOB, NOT EXTRAPOLATED ──────────────────────────────
# P3-06 sized its first two submits from a report that had measured a different question
# (`loss_is_placeholder: true`) and lost both to OOM. So leg (1) RUNS
# `tbox_finder.stage2.sizing --backbone rnafm` on this node, on real rows, through the
# trainer's own `forward_backward`, and the job REFUSES to train if the configured batch's
# worst case does not fit. It costs ~2 minutes and it cannot be wrong about the model it
# measured — `--backbone` is threaded all the way to `Stage2TrainConfig`.
#
# ⚠ AND IT ALREADY EARNED ITS KEEP. The first submit (job 1370) died in this very leg:
# `run_sizing` hardcoded gradient checkpointing ON, and this port adds its ABSOLUTE position
# embeddings in place, so checkpointing makes the embedding output a leaf requiring grad and
# the forward raises. RiNALMo is rotary and never reaches that branch. Fixed at the harness
# (the sweep flag now comes from the port's recorded usability, and the config refuses the
# combination at compose time).
#
# MEASURED afterwards on a cluster A4000 (rnafm, real rows, real objective, gc off). Every
# figure below is READ OUT of the committed `reports/p3/stage2_rnafm_sizing.json`, not retyped
# from an earlier sweep — an earlier revision of this header quoted a superseded run
# (3.3382 / 2.7522 / 1.9849 / 1.5902 GiB) and disagreed with the artifact beside it:
#     batch 8 worst_case  3.3450 GiB    batch 8 typical  2.8133 GiB / 177.0 ms
#     batch 4 worst_case  2.0206 GiB    batch 4 typical  1.6263 GiB / 122.6 ms
# (`step_ms` is the mean of steps 1..5; step 0 carries warm-up — 2162 ms at batch 8 worst case.)
# So the pinned batch 4 uses ~13 % of the 15.6 GiB card — RiNALMo needed 8.18 GiB for the same
# batch. That headroom is NOT spent: the batch stays 4 for comparability (see above).
# 9,059 rows / (8 x 4) = 283 steps/epoch/rank at ~123 ms => ~6 min for 10 epochs, plus the
# 2,070-row val rung and the two scoring legs. ⚠ That estimate is the FLOOR and the run
# overran it: `steps.elapsed_seconds` came in at 752.1 s (aux=1.0) and 731.4 s (aux=0.0), i.e.
# ~12.5 min per arm, because the estimate counts neither the val rung nor DDP synchronisation.
# `--time=04:00:00` is far above both.
# ⚠ `--time=04:00:00`, not P3-06's 02:00:00, and the difference is not padding. P3-06's two
# hours covered TRAINING ONLY — its scoring ran as separate steps. This job also runs legs (3)
# and (4), the in-distribution and leave-clade-out scoring passes, and those are SINGLE-PROCESS
# inference over the corpus while all 8 GPUs sit allocated. Disclosed rather than hidden: the
# arm's artifacts are kept together in one job deliberately (one submit-ack, one provenance
# chain, and the scores are produced by the same process state that trained the weights), and
# the cost is idle GPUs during the scoring tail. Sizing the wall clock to the training half
# alone is how a job dies at the limit with a trained checkpoint it never got to write out.
# imp.md's own Compute line says est-wall ~4h; this agrees with it. The real number is whatever
# the sizing report's step times imply, and it is measured in-job before training starts.
#
# WHERE:   cluster `gpu` partition, EIGHT RTX A4000 (sm_86). env **tbox-ml-rnafm** (ADR-0002
#          A15 — NOT ml-rna, which cannot load this checkpoint). No --nodelist (never pin).
# INPUT:   ONE parquet, STAGED by rsync (the ml envs have no dvc, and a checkout does not carry
#          it). Verified against the committed .dvc md5 + PAR1 magic before any GPU time:
#            rsync -avzP data/processed/stage2_dataset.parquet two:~/tbox-finder/data/processed/
#          src/tbox_finder/stage2/head_vocab.json is git-tracked and rides the checkout.
# SUBMIT:  from the repo root so relative --output/--error resolve. SYNC FIRST, SUBMIT SECOND.
#          ⚠ Slurm opens --output/--error BEFORE this script runs, so `reports/p3/sweep_rnafm/`
#          must already exist on the node — the `mkdir -p "$OUT_DIR"` below is too late and a
#          fresh checkout would simply lose the job's own logs. The directory is therefore
#          COMMITTED (its .gitkeep says so); the mkdir stays as a belt-and-braces no-op.
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p3-17-rnafm-comparator && git reset --hard origin/p3-17-rnafm-comparator"'
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p3/rnafm_stage2_finetune.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p3/rnafm_stage2_finetune.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty, (b)
#          reports/p3/sweep_rnafm/rnafm_stage2.DONE present, (c) the run report validates and
#          passes its own gate (train.py raises on a failed gate, so rc≠0 ⇒ no DONE), (d) the
#          BOTH point dirs (aux1.0_lr1e-4 and aux0.0_lr1e-4) carry lora_adapter/ +
#          stage2_heads.pt + provenance.json, (e) both score files AND the eval report exist.
#          ⚠️ the `.err` is EXPECTED NON-EMPTY on success (torchrun rendezvous, HF download
#          chatter, W&B-offline notices) — scan for real tracebacks, DO NOT gate on zero bytes.
#          imp.md's Validation line says "zero-byte .err"; that wording predates the
#          P2-05/P2-06/P2-09/P3-06 precedent and is wrong for a torchrun job. Recorded here
#          rather than silently ignored.
# RETRIEVE: plain scratch, NOT dvc (the ml envs have no dvc; `dvc add` runs LOCALLY after):
#            rsync -avzP two:~/tbox-finder/data/processed/checkpoints/stage2_rnafm/ <local>/data/processed/checkpoints/stage2_rnafm/
#            rsync -avzP two:~/tbox-finder/reports/p3/sweep_rnafm/ <local>/reports/p3/sweep_rnafm/
#            rsync -avzP two:~/tbox-finder/reports/p3/stage2_rnafm_sizing.json <local>/reports/p3/
#            rsync -avzP two:~/tbox-finder/reports/p3/stage2_scores_rnafm.json two:~/tbox-finder/reports/p3/stage2_scores_loo_rnafm.json <local>/reports/p3/

#SBATCH --job-name=tbox-p3-17-rnafm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=reports/p3/sweep_rnafm/slurm_%j.out
#SBATCH --error=reports/p3/sweep_rnafm/slurm_%j.err
# No --account / --qos (accounting disabled), no --nodelist (never pin; `one` is often down and
# cryosparc keeps both nodes fragmented, so a whole-node request may queue).

set -eo pipefail   # NOT -u: the conda activate hooks read unset NVCC_PREPEND_FLAGS.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# ── The two points, written out rather than computed ─────────────────────────
BACKBONE="rnafm"
LR="1e-4"
# The production point FIRST, then its lr-matched aux=0 control. Written out rather than
# computed so the two arms this file trains are readable in this file; a drift from the shipped
# defaults shows up as a test failure (tests/ml/test_stage2_train_smoke.py reads them out of
# conf/) rather than as a comparator that quietly stopped matching production.
AUX_WEIGHTS=(1.0 0.0)
BATCH_SIZE=4
GRAD_ACCUM=2
EPOCHS=10

OUT_DIR="reports/p3/sweep_rnafm"
SIZING_REPORT="reports/p3/stage2_rnafm_sizing.json"
SCORES="reports/p3/stage2_scores_rnafm.json"
SCORES_LOO="reports/p3/stage2_scores_loo_rnafm.json"
EVAL_REPORT="reports/p3/stage2_rnafm_eval.json"
CKPT_ROOT="data/processed/checkpoints/stage2_rnafm"
DONE="$OUT_DIR/rnafm_stage2.DONE"
mkdir -p "$OUT_DIR"

# Guard every interpolation used as a deletion target explicitly rather than with `${VAR:?}`:
# tests/unit/test_sbatch_rm_targets.py cannot resolve the `:?` form, scores the target as a bare
# wildcard and fails CLOSED against every tracked path. An explicit emptiness check is the same
# protection in a form the gate can read (the P3-06 ruling).
if [ -z "$BACKBONE" ] || [ -z "$LR" ] || [ "${#AUX_WEIGHTS[@]}" -ne 2 ]; then
  echo "FATAL: the point grid resolved empty or is not the two-point pair the scorers need;" >&2
  echo "  refusing to touch anything under $CKPT_ROOT." >&2
  exit 2
fi

# ── Singleton lock, for the whole ARM ────────────────────────────────────────
# One lock rather than P3-06's per-point locks: its six points were concurrent array tasks with
# disjoint outputs, whereas these two run sequentially in ONE job and the scoring legs read both
# of them. What must not happen is two jobs sharing this arm's root.
exec 9>"$OUT_DIR/.rnafm.lock"
if ! flock -n 9; then
  echo "RNAFM_SKIP: another job holds this arm's lock — refusing to share outputs." >&2
  exit 1
fi

# ── Staged-input assertion (fail loud, before any GPU time) ──────────────────
# Existence is not enough: a stale or truncated parquet trains a different population and the
# loss looks perfectly reasonable. Checked for PAR1 magic (catches truncation) AND against the
# committed .dvc md5 (catches staleness) — and it must be the SAME parquet the production arm
# trained on, or the two arms are not comparable at all.
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

if [ ! -s src/tbox_finder/stage2/head_vocab.json ]; then
  echo "FATAL: src/tbox_finder/stage2/head_vocab.json missing from the checkout." >&2
  exit 2
fi

# ── Freshness fingerprint, taken INSTEAD of deleting the report ──────────────
# "$REPORT" is untracked today and becomes GIT-TRACKED the moment the P3-17 RESULT commit lands,
# at which point an `rm` of it would dirty the tree outside `_DATA_STAGING_PREFIXES` and turn
# this run's own provenance against it (the P2-09/P3-06 ruling). Freshness is therefore proved
# by fingerprint, not by absence. Absent file -> empty string, which no md5 can equal, so "was
# missing" and "was replaced" both read as fresh and only "unchanged bytes" reads as stale.
# (taken per point, inside the training loop below — each point has its own report.)

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-ml-rnafm

export CUDA_HOME="$CONDA_PREFIX"          # multimolecule -> danling -> deepspeed probes it at import (ADR-0002 A4/A5).
export WANDB_MODE=offline                 # no live streaming from compute nodes (§9.2).
export PYTHONHASHSEED=0                    # determinism (§8.3); train.py VERIFIES, never sets.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4                   # 32 cpus / 8 ranks; torchrun warns and defaults to 1.

# `arms=`, not `key=`: KEY is a per-point name assigned inside the training loop 130 lines
# below, so this banner printed `key=` empty on every run (`set -u` is deliberately off, so it
# expanded to nothing rather than failing). The grid is what exists at this point in the job.
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} arms=${AUX_WEIGHTS[*]} lr=$LR backbone=$BACKBONE sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ── LEG (0) the ENV assertion — the ADR-0002 A15 trap, caught in seconds ─────
# Activating `tbox-ml-rna` here instead is a one-word mistake, and its failure mode is a
# ValueError about vocab sizes thrown from inside a config constructor after the queue wait.
# Assert the version and the class up front so the message names the actual problem.
if ! PYTHONPATH=src python -c '
import importlib.metadata as md
import sys
v = md.version("multimolecule")
print(f"multimolecule={v}")
if tuple(int(p) for p in v.split(".")[:2]) < (0, 2):
    sys.exit(
        f"multimolecule {v} is the ml-rna pin and CANNOT load multimolecule/rnafm "
        "(configuration_rnafm demands vocab_size==26; the checkpoint declares 28). "
        "Activate tbox-ml-rnafm (ADR-0002 A15)."
    )
import multimolecule
if not hasattr(multimolecule, "RnaFmModel"):
    sys.exit("multimolecule exports no RnaFmModel — wrong env (ADR-0002 A15).")
print("env ok: RnaFmModel available")
'; then
  echo "RNAFM_ENV_WRONG: see above — this arm needs envs/ml-rnafm (ADR-0002 A15)." >&2
  exit 2
fi

# ── Node GPU health check — FAIL FAST, before any download ───────────────────
# Job 1036 lost three points on a node whose userspace NVML had moved past its loaded kernel
# module; SLURM still called the node idle and the tasks died ~2 min in inside
# init_process_group with an opaque ncclSystemError. Node-agnostic on purpose (§13: never pin).
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  echo "RNAFM_NODE_UNHEALTHY $(hostname): nvidia-smi failed — the GPU driver stack on this" >&2
  echo "  node is unusable. Exiting before the download; §9.3 sees the missing DONE." >&2
  exit 2
fi
if ! PYTHONPATH=src python -c '
import sys, torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
n = torch.cuda.device_count()
torch.zeros(1, device="cuda:0")
print(f"cuda ok: {n} device(s), capability {torch.cuda.get_device_capability(0)}")
'; then
  echo "RNAFM_NODE_UNHEALTHY $(hostname): CUDA will not initialise under torch." >&2
  exit 2
fi

# ⚠ NOT named BUILD: `conda activate` exports BUILD=x86_64-conda-linux-gnu from the compiler
# toolchain hook, which clobbered the scratch var and killed job 789 six seconds in.
JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
trap 'rm -rf "$JOB_SCRATCH"' EXIT   # clean node-local scratch; touches only $JOB_SCRATCH.
export HF_HOME="$JOB_SCRATCH/hf"    # the weights stay OFF the 87%-full HOME quota (§9.2).
export WANDB_DIR="$JOB_SCRATCH"     # offline W&B curves stay node-local.

# ── Warm the HF cache ONCE, and print what was actually loaded ───────────────
# torchrun's 8 ranks each call from_pretrained with no cache_dir; without a warm cache they race
# on 8 concurrent downloads of the same pinned revision into one HF_HOME. The loader also
# re-measures the parameter count against the ADR-0002 A15 registry, so a hub re-upload under
# the pinned revision is caught here rather than after ten epochs.
# ⚠ NO BACKSLASH-ESCAPED QUOTES INSIDE THIS BLOCK: the body is a single-quoted shell string, so
# a `\"` survives into the Python source and an f-string containing one is a SyntaxError that
# kills the job seconds in, after the queue wait (P3-06's job-789-shaped near miss). Bind to a
# name first; do not inline the getattr.
PYTHONPATH=src BACKBONE="$BACKBONE" python -c '
import os
import torch
from tbox_finder.models.rna_backbone_registry import resolve_backbone
from tbox_finder.train import lora_harness as LH
key = os.environ["BACKBONE"]
spec = resolve_backbone(key)
model = LH.load_rna_backbone(backbone=key)
warm_attn = getattr(model.config, "_attn_implementation", None)
n_params = sum(p.numel() for p in model.parameters())
evidence = LH.read_sm86_flash_attn_evidence()
selected, reason = LH.select_attention_backend(
    flash_attn_importable=evidence["flash_attn_importable"],
    sm86_confirmed=evidence["is_sm86"],
    model_supports_flash_attn=LH.model_supports_flash_attn(key),
    dtype=LH.TRAIN_DTYPE,
    backbone=key,
)
print(f"hf cache warm {spec.repo_id} {spec.revision} torch={torch.__version__}")
print(f"params={n_params} expected={spec.expected_param_count} hidden={spec.hidden_size}")
print(f"warm-load attn (NOT the run backend)={warm_attn}")
print(f"run will select attn={selected} :: {reason}")
'

# ── LEG (1) SIZING — measured on THIS card, for THIS backbone ────────────────
# Writes its own self-validated report and raises if the MEASUREMENT is unsound (its gate grades
# the measurement, not the answer). The refusal below is separate and grades the answer.
echo "── sizing $BACKBONE on this node ──"
PYTHONPATH=src python -m tbox_finder.stage2.sizing \
  --backbone "$BACKBONE" --batch-sweep "8,4,2,1" --out "$SIZING_REPORT"

# The batch this arm is PINNED to (for comparability with the production arm) must fit the
# WORST case — the longest rows, which the collator pads everything up to. Sizing on the typical
# case is the trap that cost job 1044 sixteen minutes and an OOM.
PYTHONPATH=src SIZING_REPORT="$SIZING_REPORT" BATCH_SIZE="$BATCH_SIZE" BACKBONE="$BACKBONE" python -c '
import json, os, sys
report = json.load(open(os.environ["SIZING_REPORT"], encoding="utf-8"))
want = int(os.environ["BATCH_SIZE"])
key = os.environ["BACKBONE"]
measured_on = {m.get("backbone") for m in report["measurements"]}
if measured_on != {key}:
    sys.exit(
        f"the sizing report measured {sorted(measured_on)}, not {key!r} alone — it cannot "
        "answer whether this arm fits."
    )
worst = [
    m for m in report["measurements"]
    if m["regime"] == "worst_case" and m["batch_size"] == want
]
if not worst:
    sys.exit(f"the sizing sweep never measured batch {want} worst-case; cannot clear the run.")
point = worst[0]
if point["oom"]:
    sys.exit(
        f"batch {want} worst-case OOMs for {key}. The batch is PINNED to the production arms "
        "value for comparability, so lowering it silently would make the ECE comparison "
        "attributable to batch size as well as backbone - that is a CLAUDE.md 7 stop, not a "
        "flag to change here."
    )
peak = point["peak_vram_gib"]
step_ms = point["step_ms"]
print(f"SIZING_OK backbone={key} batch={want} worst_case_peak_gib={peak} step_ms={step_ms}")
'

# ── LEG (2) train BOTH points (DDP×8, sequentially on this node) ────────────
# The production point first, then its lr-matched aux=0 control. A point that fails ends the
# job: `set -e` plus the explicit exits below, so a half-trained pair can never reach the
# scoring legs and be graded as if it were whole.
for AUX_WEIGHT in "${AUX_WEIGHTS[@]}"; do
  KEY="aux${AUX_WEIGHT}_lr${LR}"
  REPORT="$OUT_DIR/${KEY}.json"
  CKPT_DIR="$CKPT_ROOT/${KEY}"
  SCRATCH_CKPT="$JOB_SCRATCH/checkpoint-${KEY}"

  # Guard the interpolation used as a deletion target explicitly rather than with `${VAR:?}`:
  # tests/unit/test_sbatch_rm_targets.py cannot resolve the `:?` form, scores the target as a
  # bare wildcard and fails CLOSED against every tracked path (the P3-06 ruling).
  if [ -z "$KEY" ]; then
    echo "FATAL: the point KEY resolved empty; refusing to touch $CKPT_ROOT." >&2
    exit 2
  fi

  # Freshness fingerprint, taken PER POINT and BEFORE its run — see the note above on why this
  # is a fingerprint rather than an `rm`. Absent file -> empty string, which no md5 can equal,
  # so "was missing" and "was replaced" both read as fresh and only "unchanged" reads as stale.
  REPORT_BEFORE="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"

    # Every token after `-m` is a literal key in conf/train/stage2.yaml (or its /loss and /optim
  # groups) or a config-GROUP selection — tests/unit/test_sbatch_overrides.py composes them for
  # real, because job 669 died after a 14 h queue wait on an override the config lacked.
  # `backbone=rnafm` AND `model=rnafm_stage2` are BOTH required and neither is sufficient:
  # `_cfg_from_mapping` refuses each half alone, so the run cannot record one backbone's identity
  # while loading the other's weights.
  echo "── training $KEY on $BACKBONE (aux_weight=$AUX_WEIGHT lr=$LR, DDP×8, bf16 LoRA) ──"
  set +e
  PYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \
    -m tbox_finder.stage2.train \
    backbone="$BACKBONE" model=rnafm_stage2 \
    loss.aux_weight="$AUX_WEIGHT" optim.lr="$LR" \
    epochs="$EPOCHS" batch_size="$BATCH_SIZE" gradient_accumulation_steps="$GRAD_ACCUM" \
    gradient_checkpointing=false \
    eval_val=true save_checkpoint=true \
    report_path="$REPORT" checkpoint_dir="$SCRATCH_CKPT" \
    tracking.mode=offline
  rc=$?
  set -e

  if [ "$rc" -ne 0 ] || [ ! -s "$REPORT" ]; then
    echo "RNAFM_TRAIN_FAIL rc=$rc (report not written or the run failed its own gate — §9.3 sees the missing DONE)" >&2
    [ "$rc" -ne 0 ] || rc=1
    exit "$rc"
  fi

  # "A report exists" does not imply "THIS run wrote it" — a stale committed one satisfies `[ -s ]`.
  # Assert it CHANGED. Both failure directions are explicit: an UNCOMPUTABLE fingerprint must not
  # pass by being unequal to the old one, and an UNCHANGED one must not pass at all.
  REPORT_AFTER="$(md5sum "$REPORT" 2>/dev/null | awk '{print $1}' || true)"
  if [ -z "$REPORT_AFTER" ]; then
    echo "RNAFM_TRAIN_FAIL: could not fingerprint $REPORT after the run (missing or unreadable)." >&2
    exit 1
  fi
  if [ "$REPORT_BEFORE" = "$REPORT_AFTER" ]; then
    echo "RNAFM_TRAIN_FAIL: $REPORT is byte-identical to the one that was there before this run" >&2
    echo "  (md5 $REPORT_AFTER) — this run did not write it, so it describes some OTHER run." >&2
    exit 1
  fi

  if [ ! -d "$SCRATCH_CKPT/lora_adapter" ] || [ ! -s "$SCRATCH_CKPT/stage2_heads.pt" ]; then
    echo "RNAFM_TRAIN_FAIL $KEY: report passed its gate but no checkpoint was written at $SCRATCH_CKPT." >&2
    exit 1
  fi

  # ── Move the checkpoint to the repo path + write its provenance sidecar ──────
  # Cleared immediately before the copy that REPLACES it, never earlier: an OOM, a NaN gate or a
  # failed freshness check must not end the run with no checkpoint AND the previous good one
  # already deleted (the P2-09 ruling). Every path here is untracked (DVC/scratch), so no deletion
  # can dirty a git-tracked path.
  rm -rf "$CKPT_DIR"
  mkdir -p "$CKPT_DIR"
  cp -r "$SCRATCH_CKPT/lora_adapter" "$CKPT_DIR/lora_adapter"
  cp "$SCRATCH_CKPT/stage2_heads.pt" "$CKPT_DIR/stage2_heads.pt"

  PYTHONPATH=src KEY="$KEY" BACKBONE="$BACKBONE" AUX_WEIGHT="$AUX_WEIGHT" LR="$LR" \
  CKPT_DIR="$CKPT_DIR" REPORT="$REPORT" python - <<'PY'
import os

from tbox_finder.models.rna_backbone_registry import backbone_summary, resolve_backbone
from tbox_finder.provenance import write_provenance
from tbox_finder.stage2 import train as T

key = os.environ["KEY"]
backbone = os.environ["BACKBONE"]
ckpt_dir = os.environ["CKPT_DIR"]
spec = resolve_backbone(backbone)
write_provenance(
    f"{ckpt_dir}/provenance.json",
    rule=f"slurm/p3/rnafm_stage2_finetune.sbatch :: {T.ENTRYPOINT} (P3-17 comparator {key})",
    script="src/tbox_finder/stage2/train.py",
    seed=42,
    inputs=[
        "data/processed/stage2_dataset.parquet",
        "src/tbox_finder/stage2/head_vocab.json",
    ],
    # FILES, not the adapter DIRECTORY — sha256_file raises on a dir, which cost job 1064's
    # first two points their sidecar. Shared with the production arm so the two cannot
    # disagree about what a checkpoint's outputs are.
    outputs=T.checkpoint_output_files(ckpt_dir),
    # The COMPARATOR's lock, resolved from the backbone — not the module constant, which is
    # the production arm's (ADR-0002 A15).
    env_lock=T.env_lock_for(backbone),
    adr="ADR-0002 D6 + A15; ADR-0004 D5; ADR-0005 D16/D17",
    extra={
        "step": "P3-17",
        "role": "comparator",
        "point": key,
        "backbone": backbone_summary(spec),
        "loss.aux_weight": os.environ["AUX_WEIGHT"],
        "optim.lr": os.environ["LR"],
        "report": os.environ["REPORT"],
        "world_size": 8,
        "precision": "bfloat16 LoRA (PRD §10.3; r=16, alpha=32, dropout=0.05, all-linear)",
        "population": (
            "IDENTICAL to the production arm: the ADR-0004 D5 nested_train fold INTERSECTED "
            "with the scheme-A train rung, minus the P3-02 calib split, plus the parentless "
            "decoys. The realised counts are in this run's own report, not restated here."
        ),
        "comparability_note": (
            "This arm differs from the shipped rinalmo-giga arm in the BACKBONE ONLY: same "
            "corpus, same eligibility rule, same heads/losses, same batch/accumulation/epochs, "
            "same calibration stack. It runs under multimolecule 0.2.0 rather than 0.1.0 "
            "because 0.1.0 cannot load this checkpoint at all (ADR-0002 A15) — a one-minor-"
            "version library difference P3-18 must disclose beside the D17 margin."
        ),
    },
)
print(f"wrote {ckpt_dir}/provenance.json")
PY

  if [ ! -s "$CKPT_DIR/provenance.json" ]; then
    echo "RNAFM_TRAIN_FAIL $KEY: checkpoint copied but provenance.json was not written." >&2
    exit 1
  fi
done

# ── The pair is complete BEFORE anything grades it ──────────────────────────
# `set -e` plus the per-point exits already end the job on a failed point, so this cannot fire
# in practice — which is exactly why it is here. The scorers resolve an ablation PAIR, and a
# root holding one arm is the state that produced job 1372's refusal AFTER a full training run.
# Asserting the pair costs nothing and names the condition the next reader will care about.
for AUX_WEIGHT in "${AUX_WEIGHTS[@]}"; do
  KEY="aux${AUX_WEIGHT}_lr${LR}"
  [ -s "$OUT_DIR/${KEY}.json" ] || {
    echo "RNAFM_PAIR_INCOMPLETE: $OUT_DIR/${KEY}.json missing — refusing to score a partial arm." >&2
    exit 1; }
  [ -s "$CKPT_ROOT/${KEY}/stage2_heads.pt" ] || {
    echo "RNAFM_PAIR_INCOMPLETE: $CKPT_ROOT/${KEY} has no heads — refusing to score." >&2
    exit 1; }
done

# ── LEG (3) in-distribution posteriors ──────────────────────────────────────
# The same `stage2.eval` the production arm is scored by, pointed at this arm's root. It
# re-derives the backbone from the adapter's own `base_model_name_or_path`, so it loads RNA-FM
# because the checkpoint says so — not because this line said so.
echo "── scoring (in-distribution) ──"
PYTHONPATH=src python -m tbox_finder.stage2.eval \
  --checkpoint-root "$CKPT_ROOT" --sweep-dir "$OUT_DIR" \
  --scores "$SCORES" --report "$EVAL_REPORT"

# ── LEG (4) leave-clade-out posteriors — the number P3-18's condition (c) reads ──
echo "── scoring (leave-clade-out) ──"
PYTHONPATH=src python -m tbox_finder.calib.gate2 score-loo \
  --checkpoint-root "$CKPT_ROOT" --sweep-dir "$OUT_DIR" --out "$SCORES_LOO"

# Every artifact the §9.3 VERIFY block above promises — including leg (3)'s own report, which
# an exit-0 run could otherwise fail to write while the marker still went green.
for f in "$SCORES" "$SCORES_LOO" "$EVAL_REPORT"; do
  [ -s "$f" ] || { echo "RNAFM_SCORE_FAIL: $f was not written." >&2; exit 1; }
done

touch "$DONE"
# Report the ARM, not whichever point the loop variable happens to hold last.
echo "RNAFM_OK backbone=$BACKBONE arms=${AUX_WEIGHTS[*]} (lr=$LR) root=$CKPT_ROOT reports=$OUT_DIR"
echo "RNAFM_OK scores=$SCORES loo=$SCORES_LOO eval=$EVAL_REPORT sizing=$SIZING_REPORT"
echo "NEXT (LOCAL): grade the leave-clade-out ECE and hand it to P3-18 — the swap decision is NOT this job's."
