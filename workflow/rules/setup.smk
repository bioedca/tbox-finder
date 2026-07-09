"""setup.smk — one-off setup / maintenance rules (NOT part of the CPU DAG).

`wandb_sync` runs on the cluster LOGIN NODE (`two`) to upload W&B runs that were
logged OFFLINE on the `gpu` compute nodes (PRD §16; CLAUDE.md §9.2/§11) — compute
nodes have no reliable outbound for live streaming. Invoke it by name, on the login
node, after a training/sweep job finishes:

    snakemake --cores 1 --use-conda wandb_sync

It is deliberately kept out of `rule all` (a manual upload step, not a DAG product)
and needs `WANDB_API_KEY` in the environment (login node only; never committed —
CLAUDE.md §4). `wandb sync --sync-all` uploads every unsynced run under the offline
dir; `--clean` (a separate manual step) prunes already-synced local copies.
"""


rule wandb_sync:
    """Login node: upload all offline W&B runs under `wandb_offline_dir` to the cloud."""
    params:
        offline_dir=config.get("wandb_offline_dir", "wandb"),
    log:
        "logs/wandb_sync.log",
    conda:
        "../../envs/ml-dna.yml"
    shell:
        "wandb sync --sync-all {params.offline_dir:q} >{log} 2>&1"
