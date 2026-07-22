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
_CLASSII_DIR = "data/external/classII_positives"


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


rule source_classII_positives:
    """Source additional independent non-Actinobacteria class-II positives — P0-17
    (PRD §7.1, §5 mechanism 3, §2.3 anti-mimicry sub-arm, §8).

    The class-II anti-mimicry pillar (PRD §5 mechanism 3) needs held-out class-II
    (translational) T-boxes BEYOND the single-phylum 18-record Actinobacteria/ILE set.
    A ≥2-source literature survey (7 angles + adversarial verification, 2026-07-09;
    user sign-off 2026-07-10) established the VERIFIED NEGATIVE: the peer-reviewed
    literature documents class-II T-boxes only in Actinobacteria (the reduced ileS
    system) — no phylogenetically-independent non-Actinobacteria class-II positive
    exists. Per CLAUDE.md §10.2/§10.3 (source-or-withhold; never fabricate) this rule
    publishes an HONEST EMPTY positive set (raw count 0) + the ≥2-source evidence chain,
    and catalogues the corpus's CM/DB-derived non-Actinobacteria ``type=Translational``
    records BY REFERENCE as P2 de-novo discovery LEADS (explicitly NOT positives —
    ingesting them would be the CM/DB circularity P0-16/P0-17 exist to break). The
    min-N-reachability verdict is deferred to P0-26 (ADR-0005).

    Like the other ``data.smk`` rules it is a **one-time LOCAL** rule kept out of ``rule
    all`` with no ``input:`` — the DVC corpus is read directly (only to enumerate the
    leads); the empty FASTA + provenance + report travel with the repo (git-LFS /
    ``.gitignore`` carve-out; CLAUDE.md §5.2). Invoke:

        snakemake --cores 1 --use-conda source_classII_positives
    """
    output:
        fasta=f"{_CLASSII_DIR}/classII_positives.fasta",
        provenance=f"{_CLASSII_DIR}/provenance.json",
        report=f"{_CLASSII_DIR}/classII_report.json",
    params:
        # dir derived from the output (not a hardcoded prefix) so `snakemake --lint`
        # stays clean; the module writes the FASTA + provenance + report under --out-dir.
        out_dir=lambda wildcards, output: os.path.dirname(output.fasta),
    log:
        "logs/source_classII_positives.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.anchors source-classII "
        "--out-dir {params.out_dir:q} >{log} 2>&1"


_LABELS_DIR = "data/processed/labels"


rule derive_labels:
    """Derive the 8-class per-nucleotide segmentation labels + class-II-CM-naive flag (P0-20; PRD §8/§11).

    Maps each corpus record's element annotations onto its per-record local window
    (``[1, tbox_length]``, 1-based inclusive) to produce a **single-label** per-nt vector
    over the 8 classes (``background``/``Stem_I``/``Specifier``/``Stem_II``/``Stem_III``/
    ``Antiterminator_Tbox_seq``/``Terminator``/``Discriminator``), resolving every overlap
    by the total precedence order pinned in **ADR-0004 D1** (Discriminator ▸ Specifier ▸
    Antiterminator ▸ Terminator ▸ Stem_II ▸ Stem_III ▸ Stem_I ▸ background). Terminator is
    painted only for class-I records (class II has no terminator; PMID:25583497). Also emits
    a per-record ``label_source`` / ``class_ii_cm_naive`` flag (Translational =
    ``TBDB001.cm``-derived) and a naive label vector that withholds all ``TBDB001.cm``-derived
    structure, plus aux labels (codon, cognate aa, tRNA family, regulatory mode) and
    element-coverage completeness flags (below-threshold flagged, never dropped). Writes
    ``labels_v0.parquet`` (DVC), the audit report, and a provenance.json.

    Like the other ``data.smk`` rules it is a **one-time LOCAL** rule kept out of ``rule
    all`` with no ``input:`` — its input is the DVC-tracked corpus (``master_clean_v0.parquet``,
    P0-12; tracked by DVC + provenance, not the Snakemake DAG). The parquet is DVC-tracked
    (dvc pull downstream). Invoke:

        snakemake --cores 1 --use-conda derive_labels
    """
    output:
        labels=f"{_LABELS_DIR}/labels_v0.parquet",
        provenance=f"{_LABELS_DIR}/labels_v0.provenance.json",
        report=f"{_AUDIT_DIR}/labels_report.json",
    params:
        # corpus + dirs derived from the outputs (not hardcoded prefixes) so `snakemake
        # --lint` stays clean; the module writes the parquet + provenance under --labels-dir.
        corpus=config.get("labels_corpus", "data/processed/master_clean_v0.parquet"),
        labels_dir=lambda wildcards, output: os.path.dirname(output.labels),
        audit_dir=lambda wildcards, output: os.path.dirname(output.report),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/derive_labels.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.labels "
        "--corpus {params.corpus:q} "
        "--labels-dir {params.labels_dir:q} "
        "--audit-dir {params.audit_dir:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


_SPLITS_INPUTS_DIR = "data/interim/splits/inputs"
_SPLITS_ALIGNED_DIR = "data/interim/splits/aligned"
_SPLITS_DIR = "data/interim/splits"
_PROCESSED_SPLITS_DIR = "data/processed/splits"
_FIGURES_DIR = "figures"


rule extract_split_sequences:
    """Extract per-class window FASTAs + a clustering manifest (P0-22 stage 1; PRD §9.2).

    Reads the DVC corpus (``master_clean_v0.parquet``, P0-12) + re-placed lineages
    (``lineage_replaced.parquet``, P0-15) and the independent literature-anchor (P0-16)
    / blind (18 Actinobacteria class-II) / P0-17 positives, and writes a per-class FASTA
    of the full per-record T-box window (``FASTA_sequence``) plus ``manifest.parquet``
    (identity + lineage + source tag per aligned sequence). One-time LOCAL, no ``input:``
    (its inputs are DVC/committed, external to the DAG), kept out of ``rule all``.
    """
    output:
        class_i=f"{_SPLITS_INPUTS_DIR}/class_I.fa",
        class_ii=f"{_SPLITS_INPUTS_DIR}/class_II.fa",
        manifest=f"{_SPLITS_INPUTS_DIR}/manifest.parquet",
    params:
        inputs_dir=lambda wildcards, output: os.path.dirname(output.manifest),
    log:
        "logs/extract_split_sequences.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.splits extract-sequences "
        "--inputs-dir {params.inputs_dir:q} >{log} 2>&1"


rule align_split_positives:
    """cmalign each per-class window FASTA to its class CM (P0-22 stage 2; ADR-0004 D2).

    Aligns to ``RF00230.cm`` (class I) / ``TBDB001.cm`` (class II) so the downstream
    distance is over consensus (match-state) columns, not raw identity. A pure ``cmalign``
    shell op in the infernal env (the flags mirror ``splits.run_cmalign``: ``--notrunc``
    complete loci, ``--noprob`` no posterior line); an empty per-class FASTA yields an
    empty afa. LOCAL, out of ``rule all``.
    """
    input:
        class_i=f"{_SPLITS_INPUTS_DIR}/class_I.fa",
        class_ii=f"{_SPLITS_INPUTS_DIR}/class_II.fa",
        # CMs declared as inputs so Snakemake tracks them (stage_reference_assets
        # outputs; committed git-LFS assets) and rebuilds if a CM changes.
        cm_i=f"{_REFS_DIR}/RF00230.cm",
        cm_ii=f"{_REFS_DIR}/TBDB001.cm",
    output:
        class_i=f"{_SPLITS_ALIGNED_DIR}/class_I.sto",
        class_ii=f"{_SPLITS_ALIGNED_DIR}/class_II.sto",
    log:
        "logs/align_split_positives.log",
    conda:
        "../../envs/infernal.yml"
    shell:
        "( for pair in 'I {input.class_i} {input.cm_i} {output.class_i}' "
        "'II {input.class_ii} {input.cm_ii} {output.class_ii}'; do "
        "set -- $pair; "
        "if [ -s \"$2\" ]; then "
        "cmalign --cpu 8 --notrunc --noprob --outformat pfam -o \"$4\" \"$3\" \"$2\"; "
        "else : > \"$4\"; fi; done ) >{log} 2>&1"


rule cluster_and_split:
    """Structure-aware clustering + nested split ladder + clade-crossing rule (P0-22; ADR-0004 D2/D3/D5).

    Single-linkage clusters positives on consensus-column identity at the pinned D2 cut
    (id ≥ 0.70 AND coverage ≥ 0.70), assigns whole clusters to a single fold, builds the
    nested split ladder (random genus-stratified / leave-one-order-out + class/phylum
    stress / independent anchor), applies the cluster–clade-crossing forced rule + the
    per-scheme phylogenetic-independence diagnostic (D3) and the D5 nested most-restrictive
    training fold, and emits the D2 adequacy net (train↔test distance histogram +
    tighter-cutoff re-cluster sensitivity sweep). Writes the DVC interim split-assignment
    table (P0-23 commits the compact git/LFS copy), the audit report, and the figures.
    LOCAL, out of ``rule all``.
    """
    input:
        manifest=f"{_SPLITS_INPUTS_DIR}/manifest.parquet",
        class_i=f"{_SPLITS_ALIGNED_DIR}/class_I.sto",
        class_ii=f"{_SPLITS_ALIGNED_DIR}/class_II.sto",
    output:
        table=f"{_SPLITS_DIR}/split_assignments.parquet",
        provenance=f"{_SPLITS_DIR}/split_assignments.provenance.json",
        report=f"{_AUDIT_DIR}/split_construction_report.json",
        figure_data=f"{_SPLITS_DIR}/figure_data.json",
    params:
        inputs_dir=lambda wildcards, input: os.path.dirname(input.manifest),
        aligned_dir=lambda wildcards, input: os.path.dirname(input.class_i),
        splits_dir=lambda wildcards, output: os.path.dirname(output.table),
        audit_dir=lambda wildcards, output: os.path.dirname(output.report),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/cluster_and_split.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONHASHSEED=0 python -m tbox_finder.splits cluster-split "
        "--inputs-dir {params.inputs_dir:q} "
        "--aligned-dir {params.aligned_dir:q} "
        "--splits-dir {params.splits_dir:q} "
        "--audit-dir {params.audit_dir:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


rule plot_split_figures:
    """Render the D2 adequacy figures from figure_data.json (P0-22; viz env).

    Split from ``cluster_and_split`` so the heavy clustering stays in the data env
    (pyarrow, no matplotlib) and only this numeric→PNG render needs the viz env
    (matplotlib, no pyarrow) — CLAUDE.md §3.2 rule = environment. Emits the
    train↔test structure-distance histogram + the re-cluster sensitivity sweep
    (git-LFS: ``figures/**``). LOCAL, out of ``rule all``.
    """
    input:
        figure_data=f"{_SPLITS_DIR}/figure_data.json",
    output:
        histogram=f"{_FIGURES_DIR}/split_train_test_distance_histogram.png",
        sweep=f"{_FIGURES_DIR}/split_sensitivity_sweep.png",
    params:
        figures_dir=lambda wildcards, output: os.path.dirname(output.histogram),
    log:
        "logs/plot_split_figures.log",
    conda:
        "../../envs/viz.yml"
    shell:
        "python -m tbox_finder.splits plot-figures "
        "--figure-data {input.figure_data:q} "
        "--figures-dir {params.figures_dir:q} >{log} 2>&1"


rule classII_recovery_set:
    """Build the construction-powered synthetic-class-II recovery set (P2-08; ADR-0005 D9 + A5).

    Evaluation-only presentation variants (window-phase offset, reverse complement) of
    real **held-out** class-II corpus parents, for Stage-1-only anti-mimicry grading at
    P4 on the class-II-CM-naive checkpoint (P2-11) — never through Stage-2. Parents are
    held-out rather than training-fold per ADR-0004 A3; the 18-record blind set is
    excluded as a parent so the natural arm stays independent of the synthetic one.
    Emits the sequence-free recovery table that ``split_assignment_table`` appends.
    The CLI exit code is the gate (block-level min-N + a measurable within-phylum
    control). LOCAL, out of ``rule all``.
    """
    input:
        interim=f"{_SPLITS_DIR}/split_assignments.parquet",
        aligned_i=f"{_SPLITS_ALIGNED_DIR}/class_I.sto",
        aligned_ii=f"{_SPLITS_ALIGNED_DIR}/class_II.sto",
    output:
        variants="data/processed/synth/classII_recovery.parquet",
        report="reports/p2/classII_recovery.json",
    params:
        # Derived from a DECLARED input, so the alignment is a real DAG dependency:
        # the D9 control silently reports "unmeasured" if the .sto files are absent,
        # which an undeclared read would have let happen without rebuilding them.
        aligned_dir=lambda wildcards, input: os.path.dirname(input.aligned_ii),
        seed=20260719,
    log:
        "logs/classII_recovery_set.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.synth.classII "
        "--interim-table {input.interim:q} "
        "--aligned-dir {params.aligned_dir:q} "
        "--table {output.variants:q} "
        "--report {output.report:q} "
        "--seed {params.seed} >{log} 2>&1"


rule split_assignment_table:
    """Commit the compact, sequence-free split-assignment table to git/LFS (P0-23; ADR-0004, PRD §9.2/§16).

    Promotes the DVC-interim split table (``cluster_and_split``, P0-22) to the git/LFS
    carve-out copy that the no-leakage CI reads on every PR (§8.2) — a trivial table
    scan over the *real* ~23.5k-record partition, no DVC pull needed. Projects onto the
    canonical ``record_id``/``parent_record_id`` schema (the variant→parent fold-inheritance
    column is present now; **P2-08 appends the synthetic-class-II recovery variants**,
    each inheriting its parent's fold — ADR-0004 D7 + A3), hash-links
    each corpus row to ``master_clean_v0.parquet`` (per-record ``corpus_record_sha256`` +
    the whole-file hash in the provenance ``inputs``), re-asserts the no-cluster-split
    leakage invariant, and records the DOME redundancy + partition-strategy fields (§16).
    LOCAL, out of ``rule all``.
    """
    input:
        interim=f"{_SPLITS_DIR}/split_assignments.parquet",
        corpus="data/processed/master_clean_v0.parquet",
        report=f"{_AUDIT_DIR}/split_construction_report.json",
        variants="data/processed/synth/classII_recovery.parquet",
    output:
        table=f"{_PROCESSED_SPLITS_DIR}/split_assignments.parquet",
        provenance=f"{_PROCESSED_SPLITS_DIR}/split_assignments.provenance.json",
    params:
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/split_assignment_table.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.splits write-table "
        "--interim {input.interim:q} "
        "--corpus {input.corpus:q} "
        "--audit-report {input.report:q} "
        "--out {output.table:q} "
        "--variants {input.variants:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


rule power_budget_audit:
    """GATE-1 power-budget audit — pins the min-real-homolog N (P0-26; ADR-0005 D18 amend).

    Audits, per pre-registered %-identity bin (ADR-0005 D1) and per headline-certifying
    arm (D6), how many real held-out positives the leave-clade-out partition (P0-22)
    actually supplies — per leave-one-order-out round AND pooled — plus the count of real
    low-identity homologs, so which GATE-1 arms are powered vs reported-not-gated is known
    ex ante (D8). Consumes the P0-16 anchor + P0-17 class-II raw counts to render the
    min-N-reachability verdict + the arm-(c) sourcing-fallback trigger (D6/D7). Reuses the
    ADR-0004 D2 consensus-identity metric (no divergent second definition). Never fabricates
    counts (§10.3). LOCAL, out of ``rule all``.
    """
    input:
        table=f"{_SPLITS_DIR}/split_assignments.parquet",
        class_i=f"{_SPLITS_ALIGNED_DIR}/class_I.sto",
        class_ii=f"{_SPLITS_ALIGNED_DIR}/class_II.sto",
        anchor=f"{_GATE1_DIR}/gate1_anchor_report.json",
        classii=f"{_CLASSII_DIR}/classII_report.json",
    output:
        report=f"{_AUDIT_DIR}/power_budget_report.json",
        provenance=f"{_AUDIT_DIR}/power_budget_report.provenance.json",
        figure_data=f"{_AUDIT_DIR}/power_figure_data.json",
    params:
        aligned_dir=lambda wildcards, input: os.path.dirname(input.class_i),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/power_budget_audit.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONHASHSEED=0 python -m tbox_finder.power audit "
        "--table {input.table:q} "
        "--aligned-dir {params.aligned_dir:q} "
        "--anchor-report {input.anchor:q} "
        "--classII-report {input.classii:q} "
        "--out-report {output.report:q} "
        "--figure-data {output.figure_data:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


rule plot_power_figures:
    """Render the power-budget figures from power_figure_data.json (P0-26; viz env).

    Split from ``power_budget_audit`` so the parquet/GEMM audit stays in the data env
    (pyarrow, no matplotlib) and only this numeric→PNG render needs the viz env
    (matplotlib, no pyarrow) — CLAUDE.md §3.2 rule = environment. Emits the per-identity-bin
    held-out N, the per-order N, and the headline-arm reachability figures
    (git-LFS: ``figures/**``). LOCAL, out of ``rule all``.
    """
    input:
        figure_data=f"{_AUDIT_DIR}/power_figure_data.json",
    output:
        bins=f"{_FIGURES_DIR}/power/identity_bin_counts.png",
        per_order=f"{_FIGURES_DIR}/power/per_order_counts.png",
        arms=f"{_FIGURES_DIR}/power/arm_reachability.png",
    params:
        out_dir=lambda wildcards, output: os.path.dirname(output.bins),
    log:
        "logs/plot_power_figures.log",
    conda:
        "../../envs/viz.yml"
    shell:
        "python -m tbox_finder.power plot-figures "
        "--figure-data {input.figure_data:q} "
        "--out-dir {params.out_dir:q} >{log} 2>&1"


rule ood_ece_coverage_sim:
    """OOD-ECE min-N coverage simulation — pins the ADR-0005 D13 admissibility floor (P0-27).

    Counts, over the leave-clade-out partition (P0-22), how many held-out orders clear the
    OOD-ECE min-N admissibility floor, so the adjudicable fraction of the §13 scan is known
    ex ante (PRD §12) and "discovery-predominantly-inconclusive" is a *pre-registered* modal
    outcome (PRD §2.3), not a surprise. Cross-checks per-order N against the signed P0-26
    power-budget report (non-fabrication guard, §10.3). Seeded block bootstrap for the
    adjudicable-fraction CI. LOCAL, out of ``rule all``.
    """
    input:
        table=f"{_SPLITS_DIR}/split_assignments.parquet",
        power_report=f"{_AUDIT_DIR}/power_budget_report.json",
        config="conf/data/coverage.yaml",
    output:
        report=f"{_AUDIT_DIR}/ood_ece_coverage_report.json",
        provenance=f"{_AUDIT_DIR}/ood_ece_coverage_report.provenance.json",
        figure_data=f"{_AUDIT_DIR}/ood_ece_coverage_figure_data.json",
    params:
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/ood_ece_coverage_sim.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONHASHSEED=0 python -m tbox_finder.coverage simulate "
        "--table {input.table:q} "
        "--power-report {input.power_report:q} "
        "--config {input.config:q} "
        "--out-report {output.report:q} "
        "--figure-data {output.figure_data:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


rule plot_coverage_figures:
    """Render the OOD-ECE coverage figures from the figure-data JSON (P0-27; viz env).

    Split from ``ood_ece_coverage_sim`` so the parquet audit stays in the data env and only
    this numeric→PNG render needs the viz env (CLAUDE.md §3.2 rule = environment). Emits the
    per-order coverage, the floor-sweep sensitivity, and the verdict-vector-shape figures
    (git-LFS: ``figures/**``). LOCAL, out of ``rule all``.
    """
    input:
        figure_data=f"{_AUDIT_DIR}/ood_ece_coverage_figure_data.json",
    output:
        per_order=f"{_FIGURES_DIR}/coverage/per_order_coverage.png",
        sweep=f"{_FIGURES_DIR}/coverage/floor_sweep_coverage.png",
        shape=f"{_FIGURES_DIR}/coverage/verdict_vector_shape.png",
    params:
        out_dir=lambda wildcards, output: os.path.dirname(output.per_order),
    log:
        "logs/plot_coverage_figures.log",
    conda:
        "../../envs/viz.yml"
    shell:
        "python -m tbox_finder.coverage plot-figures "
        "--figure-data {input.figure_data:q} "
        "--out-dir {params.out_dir:q} >{log} 2>&1"


_NEG_DIR = "data/processed/negatives"


rule build_decoys:
    """Build the four §9.1 static decoy/negative pools + union-prior loci-masking (P0-30).

    Assembles the four §9.1 training/benchmark negative classes — (1) a seeded
    GC+length-matched 0th-order composition background, (2) other structured RNAs
    (a checksummed Rfam subsample staged under ``data/external/refs/decoys/`` by
    ``python -m tbox_finder.decoys fetch-refs``), (3) dinucleotide-shuffled positives
    (Altschul-Erikson), and (4) the tboxevo 5'UTR/tRNA-adjacent leader anchor — then
    masks the **full union prior + own training positives + a flank** from every pool
    (``masking.py``; ADR-0006 D11 / ADR-0005 D14) and reports the residual contamination
    against the union denominator. The ADR-0006 D11 spare-rule masking key lives in
    ``masking.spare_rule_excludes_from_mining`` (encoded + unit-tested; the P2 mining
    guard). Writes ``decoys_v0.parquet`` (DVC), the audit report, and a provenance.json.

    A **one-time LOCAL** rule kept out of ``rule all`` with no ``input:`` — its inputs
    are the DVC-tracked corpus (P0-12) + union prior (P0-14) and the committed/staged
    decoy refs (tracked by DVC/git-LFS + provenance, not the Snakemake DAG). Seeded
    (``PYTHONHASHSEED=0`` + ``conf/data/decoys.yaml``). Invoke:

        python -m tbox_finder.decoys fetch-refs \
            --tboxevo-negatives <path>/idtm_validation_negatives.fasta   # one-time network stage
        snakemake --cores 1 --use-conda build_decoys
    """
    output:
        decoys=f"{_NEG_DIR}/decoys_v0.parquet",
        provenance=f"{_NEG_DIR}/decoys_v0.provenance.json",
        report=f"{_AUDIT_DIR}/decoys_report.json",
    params:
        corpus=config.get("decoys_corpus", "data/processed/master_clean_v0.parquet"),
        union_prior=config.get("decoys_union_prior", "data/processed/priors/union_prior.parquet"),
        structured_refs=config.get(
            "decoys_structured_refs", "data/external/refs/decoys/structured_rna_refs.fa"
        ),
        leader_refs=config.get(
            "decoys_leader_refs", "data/external/refs/decoys/leader_decoys.fa"
        ),
        conf="conf/data/decoys.yaml",
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/build_decoys.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONHASHSEED=0 python -m tbox_finder.decoys build "
        "--corpus {params.corpus:q} "
        "--union-prior {params.union_prior:q} "
        "--structured-refs {params.structured_refs:q} "
        "--leader-refs {params.leader_refs:q} "
        "--out {output.decoys:q} "
        "--provenance {output.provenance:q} "
        "--report {output.report:q} "
        "--config {params.conf:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


rule build_mining_pool:
    """Carve the P2-10b coordinate-bearing mining substrate (ADR-0005 D14 / A6).

    The ADR-0005 D14 mining loop's first guard is the union-prior locus mask, and
    ``classify_candidate`` refuses any candidate it cannot evaluate. At P0 that was
    **100%** of every mineable pool: ``gc_background`` is emitted i.i.d. and exists
    in no genome, ``dinuc_shuffled``'s only coordinates are its source positives'
    (so carrying them would mask the pool against its own parents — ADR-0005 A6
    withdraws its mineability), ``leader_decoy``'s ids are opaque surrogates with no
    recoverable accession, and ``structured_rna`` — whose coordinates P2-10b does
    recover from the Rfam headers — contains **zero** true overlaps with the union
    prior, so it cannot demonstrate the mask fires.

    This rule supplies what none of them can: real genomic windows carved from the
    P2-00 ``flank_context`` regions, each with an exact
    ``(accession, locus_start, locus_end, strand)`` on a replicon that hosts a known
    T-box. It also emits **locus-centred designed controls**, which overlap a known
    locus by construction and must therefore mask at **100%** — the gate that proves
    the coordinate frame is right rather than merely plausible (the minus-strand
    frame is reverse-complemented server-side and ``region_stop`` is the *requested*
    stop, so the naive arithmetic scores 98.35%).

    NOT a fifth PRD §9.1 negative class and NOT annotation-verified 5'UTRs — it is
    written to its own artifact so the §9.1 pool sizes, the ~10:1 seed ratio, and the
    committed decoy golden digest are untouched. The true 5'UTR/tRNA-adjacent leader
    pool needs CDS/tRNA annotation the repo does not have (deferred: P2-10b').

    A **one-time LOCAL** rule kept out of ``rule all`` with no ``input:`` — its inputs
    are DVC-tracked (P2-00 context, P0-14 union prior, P0-12 corpus). Seeded from
    ``conf/data/decoys.yaml``. Invoke:

        snakemake --cores 1 --use-conda build_mining_pool
    """
    output:
        pool=f"{_NEG_DIR}/mining_pool_v0.parquet",
        provenance=f"{_NEG_DIR}/mining_pool_v0.provenance.json",
        report=f"{_AUDIT_DIR}/mining_pool_report.json",
    params:
        context=config.get(
            "mining_pool_context", "data/interim/flank_context/context_v0.parquet"
        ),
        union_prior=config.get("decoys_union_prior", "data/processed/priors/union_prior.parquet"),
        corpus=config.get("decoys_corpus", "data/processed/master_clean_v0.parquet"),
        # P2-10d′-a: the parent-fold source. Every carved window is stamped with
        # its parent corpus record's `nested_train`, so admissibility as a §9.1
        # negative is data on the artifact rather than a promise by its loader.
        split_table=config.get(
            "mining_pool_split_table", "data/processed/splits/split_assignments.parquet"
        ),
        conf="conf/data/decoys.yaml",
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/build_mining_pool.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONHASHSEED=0 python -m tbox_finder.mining.pool "
        "--context {params.context:q} "
        "--union-prior {params.union_prior:q} "
        "--corpus {params.corpus:q} "
        "--split-table {params.split_table:q} "
        "--out {output.pool:q} "
        "--provenance {output.provenance:q} "
        "--report {output.report:q} "
        "--config {params.conf:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


_PILOT_DIR = "data/processed/pilot"


rule select_pilot_genomes:
    """Select the ρ-pilot genome manifest (P2-10c′-a; ADR-0003 D6).

    ADR-0003 D6 pins a scan-sizing pilot on "a ~100-genome sample spanning divergent
    clades" to measure ρ (Stage-1 candidates/Mbp) — the pivot of the P2-10e / P2-10c′
    homolog-DB + negative-window fetch, priced at 0.7-68 GB pivoting on ρ. This rule
    builds the SELECTION half: a reproducible, phylum-stratified, divergence-spanning
    ~100-genome manifest of GTDB R232 species representatives (accession + lineage),
    drawn round-robin across phyla from the pinned ``sp_clusters_r232.tsv`` crosswalk.

    It fetches no genomes (the LOCAL fetch sub-step does that), scans nothing, and
    chooses no Stage-1 detection threshold (the SLURM scan sub-step measures ρ) — so it
    pins no ADR value and carries no scientific claim (CLAUDE.md §10.3). Selection knobs
    (n_target ~100, seed, per-phylum cap, non-vacuity floors) are pinned in the module.

    A one-time LOCAL rule kept out of ``rule all`` with no ``input:`` — the crosswalk is
    an external checksummed fetch staged by ``pin_gtdb_release`` (ADR-0003 A1). Invoke:

        snakemake --cores 1 --use-conda select_pilot_genomes
    """
    output:
        manifest=f"{_PILOT_DIR}/pilot_genomes_v0.parquet",
        provenance=f"{_PILOT_DIR}/pilot_genomes_v0.provenance.json",
        report=f"{_AUDIT_DIR}/pilot_genomes_report.json",
    params:
        crosswalk=config.get("pilot_crosswalk", "data/external/gtdb/sp_clusters_r232.tsv"),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/select_pilot_genomes.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONHASHSEED=0 python -m tbox_finder.mining.pilot_genomes "
        "--crosswalk {params.crosswalk:q} "
        "--out {output.manifest:q} "
        "--provenance {output.provenance:q} "
        "--report {output.report:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"


rule p2_flank_context:
    """Source real flanking genomic context for the training corpus (P2-00).

    PRD §6 pins Stage-1 training positives as offset-augmented random window-phase
    placements in a **1024-nt window**, but the corpus stores each locus as only
    104-550 nt of dense DNA (median 281) with negligible native flank — so the
    window cannot be filled from `master_clean_v0.parquet` alone. Rather than
    synthesise the remainder (a 0th-order background splice would hand the model a
    trivially separable boundary cue and inflate GATE-4/GATE-1 — PRD §5/§10.3),
    this rule **sources the real flank** from NCBI `nuccore` (PRD §10.2).

    The stored `Name` coordinates are **not trusted**: their span reconciles with
    `len(FASTA_sequence)` on only 61.7% of records. A padded region is fetched and
    the stored sequence is located inside it by **exact string match**; a record is
    anchored iff it occurs exactly once. Fails closed — zero/multiple hits are
    recorded with a reason, never guessed. The anchor rate is measured + reported.

    A **one-time LOCAL network** rule kept out of `rule all` with no `input:` — its
    input is the DVC-tracked corpus (P0-12) + the git-LFS committed split table,
    and its true source is an external API, not a DAG product. Latency-bound: the
    fetch is rate-limited to NCBI's courtesy ceiling (3 req/s anonymous, 10 req/s
    when `NCBI_API_KEY` is set) and is **resumable** via `fetch_cache.jsonl`.
    Requires `NCBI_EMAIL` (E-utilities etiquette). Invoke:

        NCBI_EMAIL=you@example.org \
            snakemake --cores 1 --use-conda p2_flank_context
    """
    output:
        context="data/interim/flank_context/context_v0.parquet",
        provenance="data/interim/flank_context/context_v0.provenance.json",
        source_provenance="data/external/ncbi_flank/provenance.json",
        report="reports/p2/flank_context.json",
    params:
        master=config.get("flank_master", "data/processed/master_clean_v0.parquet"),
        split_table=config.get(
            "flank_split_table", "data/processed/splits/split_assignments.parquet"
        ),
        # Derived from `output` (never hardcoded): a literal path param that
        # prefixes an output trips `snakemake --lint`, which is CI-blocking.
        out_dir=lambda wildcards, output: os.path.dirname(output.context),
        external_dir=lambda wildcards, output: os.path.dirname(output.source_provenance),
        # Derived from conf/data/decoys.yaml (mining_window_nt + mining_margin_nt),
        # never hardcoded — see common.smk::flank_pad_nt. The CLI flag has existed
        # since P2-00; this rule simply never passed it, which is why the shipped
        # context was padded to 1024 and yielded zero carvable 1024-nt windows (P2-10d).
        pad_nt=flank_pad_nt(),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/p2_flank_context.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONHASHSEED=0 python -m tbox_finder.data.flank_context "
        "--master {params.master:q} "
        "--split-table {params.split_table:q} "
        "--out-dir {params.out_dir:q} "
        "--external-dir {params.external_dir:q} "
        "--report {output.report:q} "
        "--pad-nt {params.pad_nt:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"
