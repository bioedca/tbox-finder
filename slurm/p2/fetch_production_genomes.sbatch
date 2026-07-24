#!/bin/bash
# P2-10c′-fetch — the ρ-sized PRODUCTION whole-genome fetch (ADR-0006 A1): pull the 2,500 GTDB
# R232 species-representative genomes named in data/processed/mining/production_genomes_v0.parquet
# from NCBI, one normalized FASTA per genome. This substrate is the model-independent
# homolog-search target DB (ADR-0006 D7) AND the ADR-0005 D14 1,024-nt negative-window source
# (P2-10e) — one acquisition, two blockers. PRD §9.1/§13.2; ADR-0006 A1/D7; ADR-0003 A1
# (R232 source); CLAUDE.md §9.3.
#
# ── SHAPE: a SINGLE CPU-only job on the `gpu` partition (NOT an --array) ───────────────────────
# The NCBI E-utilities rate ceiling is GLOBAL per API key (10 req/s with a key, 3 without), so
# array tasks would each spin their own RateLimiter and collectively blow the ceiling → 429 bans.
# One job with the fetcher's internal ThreadPoolExecutor (workers = the applicable rate) + the one
# global RateLimiter is the correct shape. cmsearch-style sharding does not apply to a
# rate-limited network fetch. It measures genome SIZE (bp) only — no ρ, no candidate count, no
# homolog DB, no negatives, no cmsearch, no threshold (§10.3); the fetched substrate is gated by
# the ADR-0005 A8 cmsearch pre-scan BEFORE it enters the mining/negative pool (a downstream step).
#
# WHERE:   cluster `gpu` partition (nodes one/two; §9.2 forbids the `compute` partition = zero),
#          env tbox-data (envs/data.yml: biopython + pandas + pyarrow). CPU-only: NO --gres, NO
#          --nodelist (never pin; `one` is often down). The gpu compute nodes have outbound
#          internet — committed precedent: slurm/p1/rinalmo_parity.sbatch curls over --partition=gpu;
#          the P6-02 large-download job is likewise a gpu-partition CPU-only fetch (imp.md).
#
# NCBI CREDENTIALS (no repo secret, ever — CLAUDE.md §4). The fetcher reads NCBI_EMAIL (MANDATORY;
# E-utilities etiquette — the module hard-raises without it) and NCBI_API_KEY (optional; 3→10 req/s)
# from the environment. Two supported handoffs (public repo → never committed):
#   (a) a non-committed creds file `~/.ncbi_creds` on the cluster that `export`s the two vars, or
#   (b) exporting them in the submit shell (SLURM propagates the submit env by default):
#       ssh two 'bash -lc "export NCBI_EMAIL=you@example.org NCBI_API_KEY=... && \
#         cd ~/tbox-finder && sbatch slurm/p2/fetch_production_genomes.sbatch"'
#
# BUILD + STAGE: the ~2,500 FASTAs (~7 GB projected) are fetched into node-local /tmp
# ($BUILD, ~600 GB free, fast), then the genome dir + fetch log + report + provenance are moved to
# the shared checkout on success; the EXIT trap wipes /tmp. HOME quota is tight (a ~7.5 GB env has
# tripped it) — if the final move fails on quota, the run fails loud (no DONE marker, non-zero
# exit; §9.3 verify catches it) rather than certifying a partial fetch.
#
# SUBMIT:  SYNC FIRST, STAGE SECOND, SUBMIT THIRD — never one `&&` chain (§9.3):
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-10cprime-fetch && git reset --hard origin/p2-10cprime-fetch"'
#            # stage the DVC-tracked manifest into the (sacrificial) cluster checkout:
#            rsync -avzP data/processed/mining/production_genomes_v0.parquet \
#              two:~/tbox-finder/data/processed/mining/
#            ssh two 'bash -lc "export NCBI_EMAIL=… NCBI_API_KEY=… && cd ~/tbox-finder && \
#              sbatch slurm/p2/fetch_production_genomes.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/fetch_production_genomes.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty; (b) the DONE marker
#          reports/p2/fetch_production_genomes.DONE present; (c) the report
#          data/processed/audits/production_fetch_report.json present (written only on the success
#          gate: ≥MIN_SUCCESS_RATE, ≥MIN_PHYLA_OK, digest re-derived). Any missing → §9.3 stop
#          (surface it; NO auto-resubmit) — the resumable log means a fixed re-run re-costs no quota
#          for genomes already fetched into that run's /tmp, but a fresh job re-fetches from zero.
# RETRIEVE (LOCAL follow-up — its own later session; the ml/data envs on the cluster have no dvc):
#            rsync -avzP two:~/tbox-finder/data/interim/production_genomes/ <local>/data/interim/production_genomes/
#            rsync -avzP two:~/tbox-finder/data/interim/production_genomes_fetch_log.jsonl <local>/data/interim/
#            rsync -avzP two:~/tbox-finder/data/interim/production_genomes.provenance.json <local>/data/interim/
#            rsync -avzP two:~/tbox-finder/data/processed/audits/production_fetch_report.json <local>/data/processed/audits/
#          then `dvc add data/interim/production_genomes` + commit the report/provenance (next session).

#SBATCH --job-name=tbox-p2-10cpf-fetch-production-genomes
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=reports/p2/fetch_production_genomes_%j.out
#SBATCH --error=reports/p2/fetch_production_genomes_%j.err
# No --account/--qos (accounting disabled), no --gres (CPU-only network fetch), no --nodelist
# (never pin), no --array (single job — the NCBI rate ceiling is global per key).

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# ── DONE run-signal: clear FIRST, before anything that can exit, so a stale DONE from a prior
# run can never present a §9.3 green for a fetch that did not complete (the scan_pilot fix /
# "gate reads state the run mutates" class). It is git-ignored (a run signal, not an artifact);
# the committed deliverable is the report, written only on the success gate.
DONE="reports/p2/fetch_production_genomes.DONE"
mkdir -p reports/p2 data/interim data/processed/audits data/processed/mining
rm -f "$DONE"

MANIFEST="data/processed/mining/production_genomes_v0.parquet"
GENOME_DIR="data/interim/production_genomes"
FETCH_LOG="data/interim/production_genomes_fetch_log.jsonl"
REPORT="data/processed/audits/production_fetch_report.json"
PROVENANCE="data/interim/production_genomes.provenance.json"

BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
B_GENOME_DIR="$BUILD/production_genomes"
B_FETCH_LOG="$BUILD/production_genomes_fetch_log.jsonl"
B_REPORT="$BUILD/production_fetch_report.json"
B_PROVENANCE="$BUILD/production_genomes.provenance.json"
mkdir -p "$B_GENOME_DIR"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.

# ── NCBI credentials (never committed; §4). A non-committed ~/.ncbi_creds may export them. ─────
[ -f "$HOME/.ncbi_creds" ] && source "$HOME/.ncbi_creds"
: "${NCBI_EMAIL:?set NCBI_EMAIL (NCBI E-utilities etiquette — the fetcher hard-raises without it; export it in the submit shell or via ~/.ncbi_creds, never in the repo)}"

# ── Staged-input assertion (fail loud, before any NCBI time) ───────────────────────────────────
[ -s "$MANIFEST" ] || { echo "FATAL: $MANIFEST missing/empty — stage the DVC-tracked manifest: rsync -avzP data/processed/mining/production_genomes_v0.parquet two:~/tbox-finder/data/processed/mining/" >&2; exit 2; }
# Parquet magic (PAR1) — a DVC-unpulled placeholder / truncated rsync must not read as a manifest.
head -c 4 "$MANIFEST" | grep -q "PAR1" || { echo "FATAL: $MANIFEST is not a parquet file (no PAR1 magic — unpulled DVC pointer or truncated rsync)" >&2; exit 2; }

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-data
python -c "import Bio, pandas, pyarrow" 2>/dev/null || { echo "FATAL: tbox-data env missing biopython/pandas/pyarrow" >&2; exit 2; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA) key=$([ -n "${NCBI_API_KEY:-}" ] && echo set || echo unset)"

# ── Fetch into node-local /tmp (fast, no HOME-quota pressure during the run) ───────────────────
# The module reads NCBI_EMAIL / NCBI_API_KEY from the environment; it writes a per-genome .fna
# only on the STATUS_OK path and writes the report + provenance ONLY on its success gate
# (≥MIN_SUCCESS_RATE, ≥MIN_PHYLA_OK, every headline re-derived from per-genome evidence), so a
# failed run leaves no certified artifact.
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.production_fetch \
  --manifest "$MANIFEST" \
  --genome-dir "$B_GENOME_DIR" \
  --log "$B_FETCH_LOG" \
  --report "$B_REPORT" \
  --provenance "$B_PROVENANCE" \
  --env-lock envs/data.conda-lock.yml

# ── Move the certified outputs to the shared checkout (small deliverables always to $REPO; the
# large genome dir + log land where the retrieval rsync expects them). A quota failure here aborts
# under `set -e` BEFORE the DONE marker, so §9.3 verify sees no green (fail-closed).
rsync -a --delete "$B_GENOME_DIR/" "$REPO/$GENOME_DIR/"
cp "$B_FETCH_LOG" "$REPO/$FETCH_LOG"
cp "$B_REPORT" "$REPO/$REPORT"
cp "$B_PROVENANCE" "$REPO/$PROVENANCE"

N_FNA="$(find "$REPO/$GENOME_DIR" -maxdepth 1 -name '*.fna' | wc -l)"
echo "staged $N_FNA genome FASTAs → $GENOME_DIR; report → $REPORT"

touch "$DONE"
echo "fetch complete — DONE marker written; run the LOCAL retrieval (see header) then dvc add + commit"
