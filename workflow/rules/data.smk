"""data.smk — data ingest / staging rules (P0 data-layer).

`stage_reference_assets` stages the PRD §7.1 immutable reference assets (the two
class covariance models, the tboxevo structural sub-element CMs, the TBDB master
reference FASTAs, and the class-II blind set) into the canonical committed
location `data/external/refs/`, each verified against a pinned SHA-256 and
recorded in a `provenance.json` manifest (CLAUDE.md §5.2/§10.2/§11).

It is a **one-time LOCAL** rule: the immutable sources live in sibling laptop
checkouts (`tboxdb-master/`, `tbox-scan-master/`, `tboxevo/` under
`refs_sources_root`, default `..`) that are absent on a fresh clone or the cluster
checkout, so the *staged* copies (git-LFS: `*.cm`/`*.fa`/`*.fasta`) are what travel
with the repo. Like `setup.smk::wandb_sync`, it is deliberately kept out of
`rule all` (staging is not a DAG product) and declares no `input:` — the sources
are external to the Snakemake DAG. Invoke from the main checkout:

    snakemake --cores 1 --use-conda stage_reference_assets
"""

import os

_REFS_DIR = "data/external/refs"

# The §7.1 assets staged by src/tbox_finder/refs.py::stage_refs (kept in step with
# that module's MANIFEST). Hardcoded here (not imported) so `snakemake --lint` /
# `-n` parse without tbox_finder installed in the DAG env.
_REF_ASSETS = [
    "RF00230.cm",
    "TBDB001.cm",
    "idtm_seq_aware.cm",
    "idtm_structure_only.cm",
    "idtm_structure_only.amendment2.cm",
    "stem2_structure_only.cm",
    "RF00230_master.fa",
    "RFGC3V_master.fa",
    "Vitreschak_master.fa",
    "gecont3_master.fa",
    "heldout_blind_set.fasta",
]


rule stage_reference_assets:
    """Stage + checksum-verify the §7.1 reference assets into data/external/refs/."""
    output:
        [f"{_REFS_DIR}/{name}" for name in _REF_ASSETS],
        manifest=f"{_REFS_DIR}/provenance.json",
    params:
        sources_root=config.get("refs_sources_root", ".."),
        # Derived from the output (not a hardcoded prefix) so `snakemake --lint`
        # stays clean — dirname of the manifest is the staging dir _REFS_DIR.
        dest_dir=lambda wildcards, output: os.path.dirname(output.manifest),
    log:
        "logs/stage_reference_assets.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.refs --sources-root {params.sources_root:q} "
        "--dest-dir {params.dest_dir:q} >{log} 2>&1"


rule ingest_master:
    """Ingest Master_tboxes.csv → count/hash parse-correctness gate (P0-12; PRD §7.1).

    Reproduces the tboxevo canonical cleaner so a fresh ingest of the immutable raw
    TBDB export proves — at 100% per-record identity — that it reconstructs the
    canonical cleaned training corpus. Emits the interim ingest artifact (+ a
    per-record hash column), the processed training corpus P1–P4 consume, a
    count-parse report, and a provenance.json (CLAUDE.md §11).

    Like ``stage_reference_assets``, this is a **one-time LOCAL** rule kept out of
    ``rule all`` with no ``input:`` — the raw CSV and the tboxevo canonical parquet
    live in sibling laptop checkouts (``master_csv`` / ``canonical_clean_parquet``,
    default under ``..``) that are absent on a fresh clone or the cluster checkout;
    the DVC-tracked parquet outputs are what travel downstream (dvc pull). Invoke:

        snakemake --cores 1 --use-conda ingest_master
    """
    output:
        interim="data/interim/master_tboxes_ingested.parquet",
        processed="data/processed/master_clean_v0.parquet",
        report="data/processed/audits/count_parse_report.json",
        provenance="data/interim/master_tboxes_ingested.provenance.json",
    params:
        raw_csv=config.get("master_csv", "../tboxdb-master/Master_tboxes.csv"),
        canonical_parquet=config.get(
            "canonical_clean_parquet", "../tboxevo/data/interim/master_clean_v0.parquet"
        ),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/ingest_master.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.ingest "
        "--raw-csv {params.raw_csv:q} "
        "--canonical-parquet {params.canonical_parquet:q} "
        "--out-interim {output.interim:q} "
        "--out-processed {output.processed:q} "
        "--out-report {output.report:q} "
        "--out-provenance {output.provenance:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


_GTDB_DIR = "data/external/gtdb"


rule pin_gtdb_release:
    """Pin the governing GTDB release (R232) + stage the species-rep crosswalk (P0-13; PRD §7.2/§13.2).

    Fetches + MD5-verifies the R232 species-representative crosswalk
    (``sp_clusters_r232.tsv``), runs the species-rep **count gate** (counted reps == the
    release's published value: bac120 189,801 + ar53 10,122 = 199,923), and writes a
    ``provenance.json`` pinning the release, the available GTDB-Tk reference package
    (contingency that makes R232 — not the r220 fallback — the governing pin), the GTDB
    data license, and the on-demand-fetch taxonomy/metadata targets (P0-14/15/22, P6).

    Like the other ``data.smk`` rules it is a **one-time LOCAL** rule kept out of ``rule
    all`` with no ``input:`` — GTDB is an external fetch target, not a DAG product. The
    staged crosswalk (~49 MB) is gitignored + re-fetched (CLAUDE.md §5.2); only the small
    ``provenance.json`` travels with the repo. Invoke:

        snakemake --cores 1 --use-conda pin_gtdb_release
    """
    output:
        crosswalk=f"{_GTDB_DIR}/sp_clusters_r232.tsv",
        provenance=f"{_GTDB_DIR}/provenance.json",
    params:
        # dest_dir derived from the output (not a hardcoded prefix) so `snakemake --lint`
        # stays clean — dirname of the provenance manifest is the staging dir _GTDB_DIR.
        dest_dir=lambda wildcards, output: os.path.dirname(output.provenance),
    log:
        "logs/pin_gtdb_release.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.taxonomy --dest-dir {params.dest_dir:q} >{log} 2>&1"
