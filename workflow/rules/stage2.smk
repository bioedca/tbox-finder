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


# ── P3-17a: the CM + learned-calibration confirmer ablation (PRD §6 ablation (ii)) ──────
# Three rules because no single env holds both cmsearch and pandas: the query shards and the
# report are `data`-env work, the search is `infernal`-env work. Like `stage2_dataset` they
# are one-time LOCAL rules kept out of `rule all`; their inputs are DVC-tracked (the dataset),
# committed (the RiNALMo sidecars + reports) or local-only P3-16 replay files. Invoke:
#
#     snakemake --cores 22 --use-conda cm_confirmer_ablation
_CM_CONFIRMER_INTERIM = "data/interim/p3_17a"
_P3_16_INTERIM = "data/interim/p3_16"


rule cm_confirmer_queries:
    """Every Stage-2 input RNA + every P3-16 twin payload, deduplicated, as query shards."""
    output:
        manifest=f"{_CM_CONFIRMER_INTERIM}/queries/manifest.json",
    params:
        dataset=config.get("stage2_dataset", "data/processed/stage2_dataset.parquet"),
        payloads=config.get(
            "cm_confirmer_payloads", f"{_P3_16_INTERIM}/v1_payloads_twin_t0.5.json"
        ),
        out_dir=lambda wildcards, output: os.path.dirname(output.manifest),
        n_shards=config.get("cm_confirmer_n_shards", 22),
    log:
        "logs/cm_confirmer_queries.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.stage2.cm_confirmer queries "
        "--dataset {params.dataset:q} "
        "--payloads {params.payloads:q} "
        "--out-dir {params.out_dir:q} "
        "--n-shards {params.n_shards} >{log} 2>&1"


rule cm_confirmer_search:
    """cmsearch --toponly --max -T -1000 of every shard against RF00230 + TBDB001."""
    input:
        manifest=rules.cm_confirmer_queries.output.manifest,
    output:
        done=f"{_CM_CONFIRMER_INTERIM}/tblout/DONE.json",
    params:
        query_dir=lambda wildcards, input: os.path.dirname(input.manifest),
        out_dir=lambda wildcards, output: os.path.dirname(output.done),
    threads: 22
    log:
        "logs/cm_confirmer_search.log",
    conda:
        "../../envs/infernal.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.stage2.cm_confirmer search "
        "--query-dir {params.query_dir:q} "
        "--out-dir {params.out_dir:q} "
        "--jobs {threads} >{log} 2>&1"


rule cm_confirmer_ablation:
    """Learn the calibration, grade it beside RiNALMo, write the report (P3-17a)."""
    input:
        manifest=rules.cm_confirmer_queries.output.manifest,
        done=rules.cm_confirmer_search.output.done,
    output:
        report="reports/cm_confirmer_ablation.json",
        scores="reports/p3/cm_confirmer_scores.json",
        loo_scores="reports/p3/cm_confirmer_scores_loo.json",
        bench_items="reports/p3/cm_confirmer_items.json",
        confirmer=directory("data/processed/cm_confirmer"),
    params:
        dataset=config.get("stage2_dataset", "data/processed/stage2_dataset.parquet"),
        query_dir=lambda wildcards, input: os.path.dirname(input.manifest),
        tblout_dir=lambda wildcards, input: os.path.dirname(input.done),
        rinalmo_scores="reports/p3/stage2_scores.json",
        rinalmo_loo_scores="reports/p3/stage2_scores_loo.json",
        rinalmo_report="reports/gate2_p3_ece.json",
        benchmark=f"{_P3_16_INTERIM}/benchmark_v1.json",
        stage1=f"{_P3_16_INTERIM}/v1_stage1_twin.npz",
        stage2=f"{_P3_16_INTERIM}/v1_stage2_twin.json",
        payloads=config.get(
            "cm_confirmer_payloads", f"{_P3_16_INTERIM}/v1_payloads_twin_t0.5.json"
        ),
        precision_items="reports/p3/two_stage_precision_items.json",
        precision_report="reports/two_stage_precision.json",
        # P3-16's replay knobs. Not trusted: the module refuses unless they reproduce the
        # committed P3-16 items exactly, and the threshold must equal the one P3-16 recorded.
        threshold_scope=config.get("cm_confirmer_replay_threshold_scope", "global"),
        threshold=config.get("cm_confirmer_replay_threshold", 0.5),
        min_span=config.get("cm_confirmer_replay_min_span", 50),
        gap_merge=config.get("cm_confirmer_replay_gap_merge", 10),
        min_distinct_elements=config.get("cm_confirmer_replay_min_distinct_elements", 2),
        flank=config.get("cm_confirmer_replay_flank", 50),
        min_order_margin=config.get("cm_confirmer_replay_min_order_margin", 1),
        env_lock="envs/data.conda-lock.yml",
        infernal_env_lock="envs/infernal.conda-lock.yml",
    log:
        "logs/cm_confirmer_ablation.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.stage2.cm_confirmer ablation "
        "--dataset {params.dataset:q} "
        "--query-dir {params.query_dir:q} "
        "--tblout-dir {params.tblout_dir:q} "
        "--rinalmo-scores {params.rinalmo_scores:q} "
        "--rinalmo-loo-scores {params.rinalmo_loo_scores:q} "
        "--rinalmo-report {params.rinalmo_report:q} "
        "--benchmark {params.benchmark:q} "
        "--stage1 {params.stage1:q} "
        "--stage2 {params.stage2:q} "
        "--payloads {params.payloads:q} "
        "--precision-items {params.precision_items:q} "
        "--precision-report {params.precision_report:q} "
        "--out-report {output.report:q} "
        "--out-scores {output.scores:q} "
        "--out-loo-scores {output.loo_scores:q} "
        "--out-items {output.bench_items:q} "
        "--out-confirmer-dir {output.confirmer:q} "
        "--env-lock {params.env_lock:q} "
        "--infernal-env-lock {params.infernal_env_lock:q} "
        "--threshold-scope {params.threshold_scope:q} "
        "--threshold {params.threshold} "
        "--min-span {params.min_span} "
        "--gap-merge {params.gap_merge} "
        "--min-distinct-elements {params.min_distinct_elements} "
        "--flank {params.flank} "
        "--min-order-margin {params.min_order_margin} >{log} 2>&1"
