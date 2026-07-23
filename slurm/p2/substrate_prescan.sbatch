#!/bin/bash
# P2-10c′-d — the off-catalogue cmsearch substrate PRE-SCAN (ADR-0005 A8): gate the fetched
# GTDB/RefSeq negative substrate against RF00230.cm so no window overlapping a *detectable*
# T-box enters the mining/negative pool as a mislabeled background (the fail-open-off-catalogue
# harm the union-prior mask cannot prevent). PRD §9.1/§9.2; ADR-0005 A8 (amends D14); ADR-0006
# D7/D2; ADR-0002 A11; CLAUDE.md §9.3.
#
# ── SHAPE: SLURM --array, CPU-only, one batched cmsearch per shard ────────────
# There is NO per-window fast path and, at ρ-scale, no single global cmsearch (the ρ-pilot
# already sharded ×4; the production fetch shards far wider). Each array task runs ONE
# `cmsearch --tblout` over its shard's production windows PLUS that shard's own folded-in
# matched spike/null control arms, and writes a per-shard segment + SHARD_OK. A LOCAL `reduce`
# (torch-free stdlib) aggregates the segments → the certified reports/p2/substrate_prescan.json
# whose 8 same-artifact must-fire clauses make overall_pass FALSE by construction on a dead
# scan / dead shard / degraded CM / shrunken fetch.
#
# NOT the production Stage-1 scanner (masking negatives with the model's own detector is
# circular — it removes exactly its would-be false-negatives and inflates GATE-1). No --cut_ga:
# the gate is recall-favouring (errs toward REFUSING windows); the numeric SCORE_THRESHOLD is a
# phase-gate-frozen value (ADR-0005 D3 pattern) supplied at submit, pinned by NO default (§10.3).
#
# WHERE:   cluster `gpu` partition (nodes one/two; §9.2 forbids the `compute` partition = zero),
#          env tbox-infernal (envs/infernal.yml, infernal=1.1.5). CPU-only: NO --gres, NO
#          --nodelist (never pin; `one` is often down). cmsearch is quiet on success, so the
#          per-shard .err IS gated on zero bytes (unlike the torch scan jobs).
#
# ── PARAMETERS resolved by P2-10c′ (the ρ-sized fetch) — set at submit, no default ──
#   SUBSTRATE_DIR   dir of per-shard specs shard_<tid>.json (production windows + genome_windows
#                   + expected_production_windows), emitted by the fetch/tiling step.
#   SCORE_THRESHOLD recall-favouring cmsearch score cutoff, frozen at the P2/P6 phase gate.
#   NSHARDS         shard count from the fetch manifest → `sbatch --array=0-$((NSHARDS-1))`.
#
# SUBMIT:  SYNC FIRST, STAGE SECOND, SUBMIT THIRD — never one `&&` chain (§9.3):
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout <branch> && git reset --hard origin/<branch>"'
#            # …rsync the fetched substrate spec dir → two:~/tbox-finder/$SUBSTRATE_DIR …
#            ssh two 'bash -lc "cd ~/tbox-finder && SUBSTRATE_DIR=... SCORE_THRESHOLD=... \
#              sbatch --array=0-$((NSHARDS-1)) slurm/p2/substrate_prescan.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && SUBSTRATE_DIR=... SCORE_THRESHOLD=... \
#              sbatch --test-only --array=0-$((NSHARDS-1)) slurm/p2/substrate_prescan.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty; (b) every shard's
#          $SEG_DIR/shard_<tid>.OK present AND $SEG_DIR/shard_<tid>.json present; (c) every
#          per-shard .err zero-byte. Any missing → §9.3 stop (surface it; no auto-resubmit).
# RETRIEVE + REDUCE (LOCAL — reduce is torch-free stdlib, CLAUDE.md §9.1):
#            rsync -avzP two:~/tbox-finder/$SEG_DIR/ <local>/$SEG_DIR/
#            python -m tbox_finder.mining.substrate_prescan reduce \
#              $(for f in <local>/$SEG_DIR/shard_*.json; do echo --segment "$f"; done) \
#              --n-shards-expected <NSHARDS> \
#              --selection-manifest <production selection manifest .parquet> \
#              [--prior-fetch-report <prior fetch report; OMIT on the maiden fetch>] \
#              --score-threshold <SCORE_THRESHOLD> --accessed <YYYY-MM-DD> \
#              --out-report reports/p2/substrate_prescan.json
#          reduce writes the report ONLY if all 8 clauses certify (it raises otherwise), so a
#          failing gate leaves no admitted substrate — the §9.3 stop.

#SBATCH --job-name=tbox-p2-10cpd-substrate-prescan
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=reports/p2/substrate_prescan_%A_%a.out
#SBATCH --error=reports/p2/substrate_prescan_%A_%a.err
# --array=0-3 is a PLACEHOLDER matching the ρ-pilot's 4 shards; P2-10c′ overrides it with
# `sbatch --array=0-$((NSHARDS-1))` from the fetch manifest. No --account/--qos (accounting
# disabled), no --gres (CPU-only), no --nodelist (never pin).

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# ── Parameters (no default — a missing one is a fail-loud, not a silent guess; §10.3) ─────────
SUBSTRATE_DIR="${SUBSTRATE_DIR:?set SUBSTRATE_DIR to the fetch's per-shard spec dir}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:?set SCORE_THRESHOLD (recall-favouring cutoff, frozen at the phase gate)}"
CM="data/external/refs/RF00230.cm"
CM_SHA256_PIN="082616efe0616479dc9b9f565b5bcab7648d7ea1d629430940319f247af52cea"
SEG_DIR="${SEG_DIR:-reports/p2/substrate_prescan_segments}"
mkdir -p "$SEG_DIR" reports/p2

tid="${SLURM_ARRAY_TASK_ID:?this sbatch must be launched as an --array job (no SLURM_ARRAY_TASK_ID)}"
SPEC="$SUBSTRATE_DIR/shard_${tid}.json"
CONTROL="$SEG_DIR/control_${tid}.json"
SEGMENT="$SEG_DIR/shard_${tid}.json"
OK="$SEG_DIR/shard_${tid}.OK"
BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}-${tid}"

# ── Clear THIS shard's stale markers FIRST, before anything that can exit (P2-10c′-c-ii fix) ──
# The .OK/.json/.control are the §9.3 success signals; if they survived a prior run's tree and
# this task then aborted at an assertion below, a stale .OK would present a green for a scan that
# never ran. Per-shard clearing (each array task owns only its own tid files) restores
# ".OK present ⇔ THIS task succeeded".
rm -f "$OK" "$SEGMENT" "$CONTROL"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

# ── Staged-input assertions (fail loud, before any cmsearch time) ─────────────────────────────
[ -s "$CM" ] || { echo "FATAL: $CM missing (git-LFS covariance model — check checkout)" >&2; exit 2; }
GOT_SHA="$(sha256sum "$CM" | awk '{print $1}')"
[ "$GOT_SHA" = "$CM_SHA256_PIN" ] || {
  echo "FATAL: $CM sha256 $GOT_SHA != pinned $CM_SHA256_PIN — wrong/truncated CM (git-lfs pointer?)" >&2
  exit 2
}
[ -s "$SPEC" ] || { echo "FATAL: shard spec $SPEC missing — produced by the P2-10c′ fetch/tiling step" >&2; exit 2; }

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-infernal
command -v cmsearch >/dev/null 2>&1 || { echo "FATAL: cmsearch not on PATH after activating tbox-infernal" >&2; exit 2; }
cmsearch -h 2>&1 | grep -q "INFERNAL 1.1.5" || echo "WARN: cmsearch is not the pinned INFERNAL 1.1.5" >&2

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} tid=$tid sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ── Emit THIS shard's matched spike/null control (folded into the one cmsearch invocation) ────
PYTHONPATH=src python scripts/make_substrate_prescan_control.py emit-control \
  --shard "$tid" --out "$CONTROL"

# ── One batched cmsearch over production + spike + null → the shard segment + SHARD_OK ─────────
# scan-shard writes the SHARD_OK marker ONLY on the success path, after the segment lands.
PYTHONPATH=src python -m tbox_finder.mining.substrate_prescan scan-shard \
  --shard-spec "$SPEC" \
  --control "$CONTROL" \
  --score-threshold "$SCORE_THRESHOLD" \
  --cm "$CM" \
  --workdir "$BUILD" \
  --out-segment "$SEGMENT" \
  --ok-marker "$OK"

echo "shard $tid OK — segment $SEGMENT; run the LOCAL reduce once all shards' .OK are present (see header)"
