# Phase-1 backbone / segmentation data rules (Caduceus Stage-1).
#
# NOTE ON INCLUDE ORDER: ``workflow/Snakefile`` includes ``rules/*.smk`` in sorted
# order, so ``backbones.smk`` is processed *before* ``data.smk`` — it must import
# ``os`` and declare its own directory constants rather than borrowing data.smk's.

import os

# Node-local module constants (self-contained; see the include-order note above).
_SEG_SMOKE_DIR = "data/interim/p1_seg_smoke"
_SEG_SMOKE_AUDIT = "data/processed/audits/seg_smoke_report.json"
_ARCHIVEII_LOFO_PROV = "data/external/archiveii_lofo/provenance.json"


rule archiveii_lofo_prep:
    """Stage the ArchiveII nine-family inter-family LOFO benchmark (P1-12; PRD §10.2).

    Downloads RiNALMo's exact ``ARCHIVEII_SPLITS`` asset (Szikszai et al. dl-rna
    ``ct-splits.tar.gz``), verifies the pinned SHA-256 **fail-loud**, extracts the
    ``ct/fam-fold/<family>/{train,valid,test}`` tree, parses every .ct into
    sequence + reference base pairs, and writes a checksummed ``provenance.json``
    (+ a gitignored ``folds.json``) into the immutable ``data/external/`` staging
    area (CLAUDE.md §5.2). This is the parity benchmark the **P1-13** RiNALMo-mirror
    gate is evaluated on; input is sequence-only (no structure channel), matching
    RiNALMo. ADR-0002 D5 pins the parity numerics; ``reports/p1/rinalmo_published_target.json``
    records the published target.

    A **one-time LOCAL** fetch rule (network — not run in CI), kept out of
    ``rule all`` with no ``input:``. Invoke from the main checkout:

        snakemake --cores 1 --use-conda archiveii_lofo_prep
    """
    output:
        provenance=_ARCHIVEII_LOFO_PROV,
    params:
        # dest-dir derived from the output (not a hardcoded prefix) so
        # `snakemake --lint` stays clean (the P1-06 param-prefix precedent).
        dest_dir=lambda wildcards, output: os.path.dirname(output.provenance),
    log:
        "logs/archiveii_lofo_prep.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.eval.archiveii_lofo "
        "--dest-dir {params.dest_dir:q} >{log} 2>&1"


rule p1_seg_smoke_set:
    """Build the held-out prokaryotic per-nt 8-class segmentation smoke set (P1-06; PRD §8/§9.2).

    Selects a small (~a few hundred loci) subset of **held-out (leave-clade-out)**
    prokaryotic T-box loci from the committed split table — the P1-05 predicate
    ``nested_role == "heldout" & source == "corpus"`` (ADR-0004 D2; §8.2 leakage
    unit) — reuses the P0 per-nucleotide 8-class ``label_string`` (``labels_v0.parquet``,
    re-derived via ``labels.derive_label_codes`` as a consistency guard, ADR-0004 D1),
    and emits ``windows.parquet`` + a padded ``labels.npy`` (pad = -100, the ADR-0005
    seg-head ignore-index) for the P1-07 fine-tune / P1-09 continued-pretraining. It is
    a **smoke** gate, not training data. No biology is re-derived here.

    Like the ``data.smk`` rules it is a **one-time LOCAL** rule kept out of ``rule all``
    with no ``input:`` — its inputs are the git-tracked split table plus the DVC-tracked
    corpus + labels (tracked by DVC + provenance, not the Snakemake DAG; ``dvc pull``
    downstream). ``PYTHONHASHSEED=0`` for a hash-seed-independent selection. Invoke:

        snakemake --cores 1 --use-conda p1_seg_smoke_set
    """
    output:
        windows=f"{_SEG_SMOKE_DIR}/windows.parquet",
        labels=f"{_SEG_SMOKE_DIR}/labels.npy",
        provenance=f"{_SEG_SMOKE_DIR}/windows.provenance.json",
        report=_SEG_SMOKE_AUDIT,
    params:
        # inputs + out-dir passed as params (out-dir derived from an output, not a
        # hardcoded prefix, so `snakemake --lint` stays clean).
        split_table=config.get(
            "seg_smoke_split_table", "data/processed/splits/split_assignments.parquet"
        ),
        master=config.get("seg_smoke_master", "data/processed/master_clean_v0.parquet"),
        labels=config.get("seg_smoke_labels", "data/processed/labels/labels_v0.parquet"),
        out_dir=lambda wildcards, output: os.path.dirname(output.windows),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/p1_seg_smoke_set.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONHASHSEED=0 python -m tbox_finder.data.seg_smoke "
        "--split-table {params.split_table:q} "
        "--master {params.master:q} "
        "--labels {params.labels:q} "
        "--out-dir {params.out_dir:q} "
        "--report {output.report:q} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"
