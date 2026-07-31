"""stage2.smk — Stage-2 (RiNALMo RNA re-ranker) data-layer rules (P3).

`stage2_dataset` assembles the P3-01 supervised table: the 23,535 T-box positives plus
the §9.1 static decoy pools, as **RNA** (T→U transcribed) with their binary label,
per-nucleotide boundary target, §8 aux targets, dot-bracket pairing target and ADR-0004
fold/parentage columns. Details of the three load-bearing choices (sequence-only, flank
0 nt for both classes, folds inherited never invented) are in the module docstring of
`src/tbox_finder/stage2/dataset.py`.

The rule runs in the **`data`** env, not `ml-rna`: the RiNALMo alphabet is pinned in pure
Python by `src/tbox_finder/stage2/tokenizer.py` (parity-tested against the live
`RnaTokenizer` at the ADR-0002 A9 revision), so no `multimolecule`/`deepspeed`/GPU stack
is needed to build a CPU table — and the golden test stays runnable in CI.

Like the `data.smk` rules it is a **one-time LOCAL** rule kept out of `rule all`, with no
`input:` — its inputs are DVC-tracked (`master_clean_v0`, `labels_v0`, `decoys_v0`) or
git-LFS-tracked (`split_assignments`), not Snakemake DAG products. Invoke:

    snakemake --cores 1 --use-conda stage2_dataset
"""

import os

_STAGE2_PROCESSED_DIR = "data/processed"
_STAGE2_AUDIT_DIR = "data/processed/audits"


rule stage2_dataset:
    """Assemble `data/processed/stage2_dataset.parquet` (P3-01; PRD §6/§8/§10.2/§11)."""
    output:
        dataset=f"{_STAGE2_PROCESSED_DIR}/stage2_dataset.parquet",
        provenance=f"{_STAGE2_PROCESSED_DIR}/stage2_dataset.provenance.json",
        report=f"{_STAGE2_AUDIT_DIR}/stage2_dataset_report.json",
    params:
        # inputs + dirs derived from the outputs (not hardcoded prefixes) so
        # `snakemake --lint` stays clean; the module writes both sidecars itself.
        corpus=config.get("stage2_corpus", "data/processed/master_clean_v0.parquet"),
        labels=config.get("stage2_labels", "data/processed/labels/labels_v0.parquet"),
        split_table=config.get(
            "stage2_split_table", "data/processed/splits/split_assignments.parquet"
        ),
        decoys=config.get("stage2_decoys", "data/processed/negatives/decoys_v0.parquet"),
        out_dir=lambda wildcards, output: os.path.dirname(output.dataset),
        audit_dir=lambda wildcards, output: os.path.dirname(output.report),
        flank_nt=config.get("stage2_flank_nt", 0),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/stage2_dataset.log",
    conda:
        "../../envs/data.yml"
    shell:
        "python -m tbox_finder.stage2.dataset "
        "--corpus {params.corpus:q} "
        "--labels {params.labels:q} "
        "--split-table {params.split_table:q} "
        "--decoys {params.decoys:q} "
        "--out-dir {params.out_dir:q} "
        "--audit-dir {params.audit_dir:q} "
        "--flank-nt {params.flank_nt} "
        "--env-lock {params.env_lock:q} >{log} 2>&1"
