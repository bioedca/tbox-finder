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


_PRIORS_DIR = "data/processed/priors"
_AUDIT_DIR = "data/processed/audits"


rule reconcile_union_prior:
    """Reconcile the union novelty prior + NCBI→GTDB projection + unprojectable audit (P0-14; PRD §7.2/§4/§13.3).

    Builds ``union_prior.parquet`` — the union of TBDB (``master_clean_v0.parquet``,
    P0-12), the RF00230-only masking loci (``RF00230_master.fa``, P0-11a), and the curated
    literature-occurrence-by-clade artifact (Vitreschak 2008 + ≥2-source corroboration;
    §10.1) — with every record projected into the governing GTDB release (R232, P0-13) at
    finest-available (phylum) resolution, NCBI names demoted to display labels so a
    renaming/splitting artifact cannot mis-score a known lineage novel. Emits the
    unprojectable audit + the re-derived no-prior-record phylum list
    (``union_prior_report.json``) and a provenance.json. Fetches the R232 taxonomy TSVs on
    demand (MD5-verified via tbox_finder.taxonomy).

    Like the other ``data.smk`` rules it is a **one-time LOCAL** rule kept out of ``rule
    all`` with no ``input:`` — its inputs are a DVC artifact (the corpus), a git-LFS asset
    (the RF00230 FASTA) and an on-demand GTDB fetch, whose lineage DVC + provenance.json
    track (not the Snakemake DAG). The parquet is DVC-tracked (dvc pull downstream). Invoke:

        snakemake --cores 1 --use-conda reconcile_union_prior
    """
    output:
        union_prior=f"{_PRIORS_DIR}/union_prior.parquet",
        provenance=f"{_PRIORS_DIR}/union_prior.provenance.json",
        report=f"{_AUDIT_DIR}/union_prior_report.json",
    params:
        # dirs derived from the outputs (not hardcoded prefixes) so `snakemake --lint`
        # stays clean; the module writes the parquet + provenance under --priors-dir.
        priors_dir=lambda wildcards, output: os.path.dirname(output.union_prior),
        audit_dir=lambda wildcards, output: os.path.dirname(output.report),
        gtdb_dir=_GTDB_DIR,
    log:
        "logs/reconcile_union_prior.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.priors "
        "--priors-dir {params.priors_dir:q} "
        "--audit-dir {params.audit_dir:q} "
        "--gtdb-dir {params.gtdb_dir:q} >{log} 2>&1"


_INTERIM_DIR = "data/interim"
_NCBI_TAX_DIR = "data/external/ncbi_taxonomy"
_GATE1_DIR = "data/external/gate1_anchor"


rule replace_taxid_lineage:
    """Re-derive lineage-by-rank for the taxonomy-incomplete positives from TaxId (P0-15; PRD §9.2/§12).

    ~4% of the 23,535-record corpus lacks a clade label (453 no phylum, 841 no class, 928
    no order), which the leave-one-order-out headline split + the no-leakage test cannot
    silently absorb. Every such record carries an NCBI ``TaxId``, so this re-derives its
    lineage from a **frozen, MD5-pinned NCBI taxdump snapshot** and fills only the missing
    ranks (fill-only — the curated 96% is never overwritten), reconciling recovered labels
    to the corpus's pre-2021 vintage via the taxdump's own synonym records. Emits
    ``lineage_replaced.parquet`` (keyed by the row-aligned ``record_sha256``), the per-rank
    recovery audit (``lineage_replacement_report.json``), an artifact provenance, and the
    taxdump pin manifest. Residue with no formal NCBI rank is flagged
    ``dropped_from_clade_holdout`` so a no-clade record can never enter a clade fold.

    Like the other ``data.smk`` rules it is a **one-time LOCAL** rule kept out of ``rule
    all`` with no ``input:`` — its inputs are two DVC artifacts (the corpus + the ingested
    parquet, tracked by DVC + provenance) and an on-demand NCBI-taxdump fetch (~75 MB,
    gitignored + re-fetched; only its ``provenance.json`` travels, CLAUDE.md §5.2). The
    parquet is DVC-tracked (dvc pull downstream). Invoke:

        snakemake --cores 1 --use-conda replace_taxid_lineage
    """
    output:
        lineage_replaced=f"{_INTERIM_DIR}/lineage_replaced.parquet",
        provenance=f"{_INTERIM_DIR}/lineage_replaced.provenance.json",
        report=f"{_AUDIT_DIR}/lineage_replacement_report.json",
        taxdump_provenance=f"{_NCBI_TAX_DIR}/provenance.json",
    params:
        # dirs derived from the outputs (not hardcoded prefixes) so `snakemake --lint`
        # stays clean; the module writes the parquet + provenance under --interim-dir.
        interim_dir=lambda wildcards, output: os.path.dirname(output.lineage_replaced),
        audit_dir=lambda wildcards, output: os.path.dirname(output.report),
        taxdump_dir=lambda wildcards, output: os.path.dirname(output.taxdump_provenance),
    log:
        "logs/replace_taxid_lineage.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.taxonomy replace-lineage "
        "--interim-dir {params.interim_dir:q} "
        "--audit-dir {params.audit_dir:q} "
        "--taxdump-dir {params.taxdump_dir:q} >{log} 2>&1"


rule source_gate1_anchor:
    """Source the independent non-Firmicutes GATE-1 anchor (arm c) — P0-16 (PRD §7.1/§9.2(c)/§2.3).

    The headline generalization claim needs a *model-independent, beyond-Firmicutes*
    positive anchor whose selection is independent of the RF00230 CM (the GATE-1
    ``cmsearch`` baseline) and whose sequences are independent of the TBDB training
    corpus. This re-derives each locus from its **primary NCBI genome**, using Vitreschak
    et al. 2008's supplementary alignment (Fig S1) as the CM-free ground truth: parse the
    gap-elided alignment rows into contiguous genomic segments, localize them in the
    host's primary genome (the literature sequence IS the query — no CM), verify the
    inter-segment gaps match the elided lengths, and extract the full contiguous leader.
    Emits the re-derived non-Firmicutes leader FASTA, an artifact provenance, and an audit
    report (raw counts per clade/host, GTDB placement, independence statement, and a
    preliminary corpus-overlap leakage report for P0-24 to hold out). Dictyoglomi is
    WITHHELD (single-source; CLAUDE.md §10.1).

    Like the other ``data.smk`` rules it is a **one-time LOCAL** rule kept out of ``rule
    all`` with no ``input:`` — its inputs are the DVC corpus (read directly) plus on-demand
    checksummed fetches (the Vitreschak .doc supplement + primary genomes via NCBI
    E-utilities), which live in a gitignored ``.cache/`` and are re-fetched; only the
    re-derived FASTA + provenance + report travel with the repo (CLAUDE.md §5.2). Invoke:

        snakemake --cores 1 --use-conda source_gate1_anchor
    """
    output:
        anchor=f"{_GATE1_DIR}/gate1_anchor.fasta",
        provenance=f"{_GATE1_DIR}/provenance.json",
        report=f"{_GATE1_DIR}/gate1_anchor_report.json",
    params:
        # dir derived from the output (not a hardcoded prefix) so `snakemake --lint`
        # stays clean; the module writes the FASTA + provenance + report under --anchor-dir.
        anchor_dir=lambda wildcards, output: os.path.dirname(output.anchor),
    log:
        "logs/source_gate1_anchor.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.anchors source-anchor "
        "--anchor-dir {params.anchor_dir:q} >{log} 2>&1"
