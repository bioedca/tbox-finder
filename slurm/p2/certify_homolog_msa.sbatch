#!/bin/bash
# P2-10e-msa — CERTIFY the per-candidate homolog-search + CM-free de-novo MSA supply (ADR-0006 D7/A1).
#
# This proves the ADR-0006 A2 claim that the covariation-(a) backend is MSA-*producible* (not merely
# "R-scape installed") BEFORE mine_round.MSA_SUPPLY_AVAILABLE is flipped to True. It is the analogue
# of build_homolog_db.sbatch's must-fire round-trip control, one layer up: a real class-II T-box run
# through the whole producer must yield a >=20-sequence covariation-PASS MSA, and that MSA's matched
# column-shuffled twin (same depth/width/SS_cons/composition, only covariation destroyed) must NOT
# pass. Positive fires + matched negative does not ⇒ the PASS reflects real inter-column covariation
# in the producer's own output, not the pipeline. (matched-control-before-certifying;
# control-matchedness-must-be-asserted.)
#
# ── SHAPE: a SINGLE CPU-only job on the `gpu` partition (NOT an --array), THREE conda envs ──────────
#   Stage 1 SEARCH  (tbox-homology): blastn the seed against the A1 target DB → candidate+homolog FASTA
#   Stage 2 ALIGN   (tbox-locarna) : mlocarna(candidate+homologs) → Pfam Stockholm (+ #=GC SS_cons alifold)
#   Stage 3 CERTIFY (tbox-rscape)  : R-scape score positive (PASS) + column-shuffled twin (not PASS)
# No GPU (search/align/score are CPU). §9.2 forbids the `compute` partition (=zero) and pinning a node
# (`one` is often down); the gpu nodes carry the tools + the 15 GB DB. NO --gres/--nodelist/--array.
#
# ── ⛔ CLUSTER-ENV PREREQUISITE (generalises [[cluster-has-no-tbox-data-env]]) ─────────────────────
# The cluster has tbox-homology but NOT tbox-locarna / tbox-rscape (tbox-infernal is NO LONGER used by
# this sbatch — the ALIGN stage is mlocarna now, ADR-0006 A3 / ADR-0002 A13). Create both ONCE from the
# pinned locks before submitting (CPU-only, ~1-3 GB each; watch the tight HOME quota). This is part of
# the §9.3 submit-ack, done on `submit`, NOT before it:
#   ssh two 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate condalock && \
#     conda-lock install --name tbox-locarna envs/locarna.conda-lock.yml"'
#   ssh two 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate condalock && \
#     conda-lock install --name tbox-rscape  envs/rscape.conda-lock.yml"'
#
# ── SUBMIT (SYNC FIRST, SUBMIT SECOND — never one `&&` chain, §9.3) ─────────────────────────────────
#   ssh two 'bash -lc "cd ~/tbox-finder && git fetch origin && git checkout p2-10e-msa && \
#     git reset --hard origin/p2-10e-msa"'
#   # …then create the two envs (above)…
#   ssh two 'bash -lc "cd ~/tbox-finder && sbatch slurm/p2/certify_homolog_msa.sbatch"'
# PREFLIGHT: ssh two 'bash -lc "cd ~/tbox-finder && sbatch --test-only slurm/p2/certify_homolog_msa.sbatch"'
# VERIFY (artifact-based; sacct off, §9.3): (a) `squeue -j <jobid>` empty; (b) DONE marker
#   reports/p2/certify_homolog_msa.DONE present; (c) data/processed/audits/homolog_msa_certify_report.json
#   present with certified==true (written only on the must-fire success gate). Any missing → §9.3 stop
#   (surface it; NO auto-resubmit). A non-firing positive (found <20 homologs, or positive != passed) is
#   a control-cutoff iteration, NOT a resubmit — adjust the CONTROL cutoffs below and re-ack.
# RETRIEVE (LOCAL follow-up — the result commit; the cluster envs have no dvc):
#   rsync -avzP two:~/tbox-finder/data/processed/audits/homolog_msa_certify_report.json <local>/...
#   rsync -avzP two:~/tbox-finder/reports/p2/homolog_msa_search_report.json            <local>/...
#   rsync -avzP two:~/tbox-finder/data/interim/homolog_msa/                            <local>/data/interim/homolog_msa/
#   then flip mine_round.MSA_SUPPLY_AVAILABLE→True + wire mine_round.sbatch legs (a)-(d), dvc add the MSA.

#SBATCH --job-name=tbox-p2-10e-certify-msa
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=reports/p2/certify_homolog_msa_%j.out
#SBATCH --error=reports/p2/certify_homolog_msa_%j.err
# No --account/--qos (accounting disabled), no --gres (CPU-only), no --nodelist (never pin), no --array.

set -eo pipefail   # NOT -u: conda activate hooks read unset vars.

REPO="${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}"
cd "$REPO"

# ── DONE run-signal: clear FIRST (before anything that can exit under set -e) so a stale DONE from a
# prior run can never green a certification that did not complete. Gitignored; the committed
# deliverable is the certify report, written only on the must-fire success gate. ─────────────────────
DONE="reports/p2/certify_homolog_msa.DONE"
mkdir -p reports/p2 data/interim/homolog_msa data/processed/audits
rm -f "$DONE"

DB_DIR="data/interim/homolog_db"
BLAST_PREFIX="$DB_DIR/target"
FM_INDEX="$DB_DIR/target.fm"
GENOME_DIR="data/interim/production_genomes"   # homolog subranges are carved from the source genomes
SEED_STO="tests/fixtures/rscape/classII_sub.sto"   # committed, real class-II corpus rows (non-heldout)
SEARCH_REPORT="reports/p2/homolog_msa_search_report.json"
CERTIFY_REPORT="data/processed/audits/homolog_msa_certify_report.json"
CERTIFIED_MSA="data/interim/homolog_msa/certified_positive.sto"

# ── CONTROL cutoffs — the certification's control parameters, NOT a pin (D17: the D7 inclusion ─────
# thresholds are P6-frozen; nhmmer is one of D7's three sanctioned engines, none singly pinned).
# Recall-favouring, chosen so a real class-II seed recovers a DIVERGENT FULL-LENGTH >=20-homolog set
# with covariation power. Engine switched blastn→nhmmer at P2-10e-msa-locarna (job 755 diagnostic):
# `-task blastn` found 69 hits but ALL short (~20-33 nt) high-identity anchors on a conserved core →
# only 4 cleared cov>=0.2 (blastn cannot extend across the divergent flanks of a structured RNA).
# nhmmer against the FM-index recovers FULL-LENGTH divergent homologs: at E<=100/cov>=0.5, 24 distinct
# targets each covering >=50% of the 221-nt element (18 at E<=10). Switching to nhmmer + demanding
# cov>=0.5 stays within D7 (engine is a menu choice; thresholds "frozen at P6, value at the phase
# gate" — no number pinned to contradict; the P2 values here are provisional, frozen at P6). If <20
# recovered or the positive does not pass, adjust these (raise E / lower cov) and re-ack — they carry
# no methodological weight; min_sequences=20 stays the pinned floor (ADR-0006 A2). ─────────────────
CTRL_ENGINE="nhmmer"
CTRL_EVALUE="100"
CTRL_MAX_TARGET_SEQS="500"
CTRL_MIN_PIDENT="40"   # ignored on the nhmmer path (no pident filter); kept for the CLI-required arg
CTRL_MIN_COV="0.5"
SHUFFLE_SEED="20260726"

BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT   # clean node-local scratch on every exit path; touches only $BUILD.
SEED_FA="$BUILD/seed.fa"
HOMOLOGS_FA="$BUILD/homologs.fa"
POS_STO="$BUILD/positive.sto"
B_SEARCH_REPORT="$BUILD/search_report.json"
B_CERTIFY_REPORT="$BUILD/certify_report.json"

# ── Staged-input assertions (fail loud, before any tool time) ────────────────────────────────────
[ -s "$SEED_STO" ] || { echo "FATAL: $SEED_STO missing (git-tracked; sync the branch)" >&2; exit 2; }
[ -s "$FM_INDEX" ] || { echo "FATAL: $FM_INDEX missing — the P2-10c′ homolog DB (job 741) is not staged; rsync data/interim/homolog_db/ or dvc pull." >&2; exit 2; }
ls "$BLAST_PREFIX".n* >/dev/null 2>&1 || { echo "FATAL: BLAST DB $BLAST_PREFIX.n* missing under $DB_DIR" >&2; exit 2; }
N_GENOMES="$(find "$GENOME_DIR" -maxdepth 1 -name '*.fna' 2>/dev/null | wc -l)"
[ "$N_GENOMES" -ge 1 ] || { echo "FATAL: $GENOME_DIR has no .fna — the 2,500 fetched genomes (job 739) are not staged; homolog extraction carves subranges from them." >&2; exit 2; }

CONDA_SH=/exports/people/mondragonlab/ecc1695/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sha=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ── STAGE 1: SEARCH (tbox-homology) ──────────────────────────────────────────────────────────────
conda activate tbox-homology
for t in blastn nhmmer; do
  command -v "$t" >/dev/null || { echo "FATAL: $t not on PATH — tbox-homology not active?" >&2; exit 2; }
done
# Seed = a real degapped class-II corpus row (stdlib-only; runs in any env).
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.homolog_msa seed \
  --from-stockholm "$SEED_STO" --index 0 --name classII_seed --out "$SEED_FA"

set +e
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.homolog_msa search \
  --candidate-fasta "$SEED_FA" --out-fasta "$HOMOLOGS_FA" \
  --engine "$CTRL_ENGINE" --evalue "$CTRL_EVALUE" --max-target-seqs "$CTRL_MAX_TARGET_SEQS" \
  --min-pident "$CTRL_MIN_PIDENT" --min-cov "$CTRL_MIN_COV" \
  --db-prefix "$BLAST_PREFIX" --fm-index "$FM_INDEX" --genome-dir "$GENOME_DIR" \
  --tblout "$BUILD/nhmmer.tblout" --report "$B_SEARCH_REPORT"
search_rc=$?
set -e
if [ "$search_rc" -eq 4 ]; then
  echo "CERTIFY_FAIL: the positive seed recovered < min_sequences homologs (not producible). Loosen the" >&2
  echo "CONTROL cutoffs (CTRL_EVALUE/CTRL_MIN_PIDENT/CTRL_MIN_COV) and re-ack — no DONE written." >&2
  cp "$B_SEARCH_REPORT" "$SEARCH_REPORT" 2>/dev/null || true
  exit 4
elif [ "$search_rc" -ne 0 ]; then
  echo "FATAL: search failed rc=$search_rc" >&2; exit "$search_rc"
fi

# ── STAGE 2: ALIGN (tbox-locarna) ────────────────────────────────────────────────────────────────
conda activate tbox-locarna
command -v mlocarna >/dev/null || { echo "FATAL: mlocarna not on PATH — tbox-locarna not active/created?" >&2; exit 2; }
# Assert the pinned LocARNA (ADR-0002 A13); decouple the version grep from mlocarna's exit code.
MLOCARNA_VERSION="$(mlocarna --version 2>&1 || true)"
echo "$MLOCARNA_VERSION" | grep -q "LocARNA 2.0.1" || { echo "FATAL: mlocarna != LocARNA 2.0.1 (envs/locarna.yml, ADR-0002 A13): $MLOCARNA_VERSION" >&2; exit 2; }
# mlocarna aligns candidate+homologs from ONE multi-FASTA — the search stage already wrote the seed as
# record 0 of $HOMOLOGS_FA; --cpu is passed to mlocarna --threads, kept small (LocARNA scaling caveat).
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.homolog_msa align \
  --candidate-fasta "$SEED_FA" --homologs-fasta "$HOMOLOGS_FA" \
  --out-sto "$POS_STO" --work-dir "$BUILD/align" --cpu 2

# ── STAGE 3: CERTIFY (tbox-rscape) ───────────────────────────────────────────────────────────────
conda activate tbox-rscape
command -v R-scape >/dev/null || { echo "FATAL: R-scape not on PATH — tbox-rscape not active/created?" >&2; exit 2; }
# covariation.covariation_spare_status asserts R-scape == 2.0.4.a where it runs (ADR-0002 A11).
PYTHONHASHSEED=0 PYTHONPATH=src python -m tbox_finder.mining.homolog_msa certify \
  --msa "$POS_STO" --report "$B_CERTIFY_REPORT" --shuffle-seed "$SHUFFLE_SEED"

# ── Promote the small certified artifacts to the shared checkout, DONE LAST ──────────────────────
cp "$B_SEARCH_REPORT" "$SEARCH_REPORT"
cp "$B_CERTIFY_REPORT" "$CERTIFY_REPORT"
cp "$POS_STO" "$CERTIFIED_MSA"
echo "certified homolog-MSA supply → $CERTIFY_REPORT ; MSA → $CERTIFIED_MSA"

touch "$DONE"
echo "certification complete — DONE marker written; run the LOCAL retrieval + result commit (see header)"
