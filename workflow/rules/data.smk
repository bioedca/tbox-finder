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
