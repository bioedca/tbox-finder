#!/bin/bash
# P2-10c′-homologdb — build the model-independent homolog-search TARGET DB (ADR-0006 D7 + A1) over
# the 2,500 GTDB R232 species-rep genomes fetched at data/interim/production_genomes/ (P2-10c′-fetch).
# The candidate is the query; this set is the target sequence DB. From one concatenated target FASTA
# (contig headers rewritten to <accession>:c<contig_index>) it builds:
#   • a makeblastdb nucleotide DB  (blastn searches it),
#   • a makehmmerdb FM-index       (nhmmer searches it, tractably over ~7 Gbp),
# and keeps the FASTA as the cmsearch-from-candidate target — the three D7-named routes. It builds an
# index and PROVES it searchable (a first+last-genome round-trip self-recovery control); it runs no
# genome scan, no cmsearch, counts no candidates, and pins no D7 inclusion threshold (P6-frozen, D17).
# PRD §2; ADR-0006 D7/A1; ADR-0002 A12 (envs/homology.yml); CLAUDE.md §9.3.
#
# ── SHAPE: a SINGLE CPU-only job on the `gpu` partition (NOT an --array) ───────────────────────────
# makeblastdb + makehmmerdb are single-jobs over the whole target; there is no per-shard work and no
# GPU. §9.2 forbids the `compute` partition (= zero) and pinning a node (`one` is often down); the gpu
# nodes carry the tools + the genomes. NO --gres, NO --nodelist, NO --array, NO --account/--qos.
#
# WHERE:   cluster `gpu` partition (nodes one/two), env tbox-homology (envs/homology.yml: hmmer 3.4 +
#          blast 2.17.0 + biopython; NO pandas — the driver reads the frozen accession set + the
#          per-genome integrity baseline from the stdlib-JSON production_windows_report.json). The env
#          is created once from the pinned lock (the [[cluster-has-no-tbox-data-env]] gap):
#            conda activate condalock && conda-lock install --name tbox-homology envs/homology.conda-lock.yml
#
# BUILD + STAGE: the target FASTA + BLAST DB + FM-index (~15-20 GB) are built in node-local /tmp
# ($BUILD, ~600 GB free, fast); the certified homolog_db/ dir + report + provenance are moved to the
# shared checkout on success; the EXIT trap wipes /tmp. A quota failure on the final move aborts under
# `set -e` BEFORE the DONE marker (fail-closed; §9.3 verify sees no green).
#
# SUBMIT:  SYNC FIRST, SUBMIT SECOND — never one `&&` chain (§9.3). The genomes + baseline report are
#          already on the cluster from job 739 (P2-10c′-fetch); no data staging is needed, only a
#          branch sync (git reset --hard leaves the gitignored genome dir untouched):
#            ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && \
#              git checkout p2-10cprime-homologdb && git reset --hard origin/p2-10cprime-homologdb"'
#            ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/build_homolog_db.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/build_homolog_db.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty; (b) the DONE marker
#          reports/p2/build_homolog_db.DONE present; (c) the report
#          data/processed/audits/homolog_db_report.json present with round_trip.all_recovered==true
#          (written only on the success gate: integrity + blastdbcmd cross-check + both-engine
#          self-recovery of the first+last genome). Any missing → §9.3 stop (surface it; NO
#          auto-resubmit).
# RETRIEVE (LOCAL follow-up — its own later session; the cluster env has no dvc):
#            rsync -avzP two:~/tbox-finder/data/interim/homolog_db/ <local>/data/interim/homolog_db/
#            rsync -avzP two:~/tbox-finder/data/interim/homolog_db.provenance.json <local>/data/interim/
#            rsync -avzP two:~/tbox-finder/data/processed/audits/homolog_db_report.json <local>/data/processed/audits/
#          then `dvc add data/interim/homolog_db` + commit the report/provenance (next session).

#SBATCH --job-name=tbox-p2-10cph-build-homolog-db
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=reports/p2/build_homolog_db_%j.out
#SBATCH --error=reports/p2/build_homolog_db_%j.err
# No --account/--qos (accounting disabled), no --gres (CPU-only index build), no --nodelist (never
# pin), no --array (single job over the whole target).

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# ── DONE run-signal: clear FIRST, before anything that can exit, so a stale DONE from a prior run
# can never present a §9.3 green for a build that did not complete (the "gate reads state the run
# mutates" class). It is git-ignored; the committed deliverable is the report, written only on the
# driver's success gate.
DONE="reports/p2/build_homolog_db.DONE"
mkdir -p reports/p2 data/interim data/processed/audits
rm -f "$DONE"

GENOME_DIR="data/interim/production_genomes"
BASELINE="data/processed/audits/production_windows_report.json"
DB_DIR="data/interim/homolog_db"
REPORT="data/processed/audits/homolog_db_report.json"
PROVENANCE="data/interim/homolog_db.provenance.json"

BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
B_DB_DIR="$BUILD/homolog_db"
B_REPORT="$BUILD/homolog_db_report.json"
B_PROVENANCE="$BUILD/homolog_db.provenance.json"
mkdir -p "$B_DB_DIR"
trap 'rm -rf "$BUILD"' EXIT     # clean node-local scratch on every exit path; touches only $BUILD.

# ── Staged-input assertions (fail loud, before any index time) ─────────────────────────────────
N_FNA="$(find "$GENOME_DIR" -maxdepth 1 -name '*.fna' 2>/dev/null | wc -l)"
[ "$N_FNA" -eq 2500 ] || { echo "FATAL: $GENOME_DIR has $N_FNA .fna, expected 2500 — the fetched set (job 739) is not staged here; rsync it or re-run P2-10c′-fetch." >&2; exit 2; }
[ -s "$BASELINE" ] || { echo "FATAL: $BASELINE missing/empty (git-tracked; sync the branch)" >&2; exit 2; }
head -c 1 "$BASELINE" | grep -q "{" || { echo "FATAL: $BASELINE is not JSON" >&2; exit 2; }

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate tbox-homology
python -c "import Bio" 2>/dev/null || { echo "FATAL: tbox-homology env missing biopython" >&2; exit 2; }
for t in makeblastdb blastdbcmd makehmmerdb blastn nhmmer; do
  command -v "$t" >/dev/null || { echo "FATAL: $t not on PATH — tbox-homology env not active?" >&2; exit 2; }
done

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA) n_fna=$N_FNA"

# ── Build into node-local /tmp (target FASTA + BLAST DB + FM-index), prove it searchable ─────────
# The driver writes the report + provenance ONLY after every gate passes (integrity re-measure ==
# frozen baseline; blastdbcmd -info == the concatenated totals; first+last genome self-recover by
# blastn AND nhmmer), so a failed build leaves no certified artifact.
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.homolog_db build \
  --genome-dir "$GENOME_DIR" \
  --baseline-report "$BASELINE" \
  --out-dir "$B_DB_DIR" \
  --report "$B_REPORT" \
  --provenance "$B_PROVENANCE" \
  --env-lock envs/homology.conda-lock.yml

# ── Move the certified outputs to the shared checkout (fail-closed on quota under `set -e`) ──────
rsync -a --delete "$B_DB_DIR/" "$REPO/$DB_DIR/"
cp "$B_REPORT" "$REPO/$REPORT"
cp "$B_PROVENANCE" "$REPO/$PROVENANCE"

echo "staged homolog DB → $DB_DIR ($(du -sh "$REPO/$DB_DIR" | cut -f1)); report → $REPORT"

touch "$DONE"
echo "homolog-DB build complete — DONE marker written; run the LOCAL retrieval (see header) then dvc add + commit"
