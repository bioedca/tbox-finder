# Evaluation rules — the phase gates that are computed, not trained (P2-14 onward).
#
# NOTE ON INCLUDE ORDER: ``workflow/Snakefile`` globs ``rules/*.smk`` in sorted order, so
# this module is processed after ``data.smk`` — it declares its own constants rather than
# borrowing another module's, exactly as ``backbones.smk`` does. Helper *functions* belong
# in ``common.smk``: ``snakemake --lint`` refuses "Mixed rules and functions in same
# snakefile" and a red lint is CI-blocking.

_GATE4_REPORT = "reports/p2/gate4.json"
_GATE4_TWIN_CKPT = "data/processed/checkpoints/stage1_gate4_twin/stage1.pt"
_GATE4_TWIN_REPORT = "reports/p2/train_stage1_gate4_twin.json"
_GATE4_TWIN_PROVENANCE = "data/processed/checkpoints/stage1_gate4_twin/provenance.json"
_PDB_EXTENTS = "tests/fixtures/pdb_element_extents/pdb_element_extents.json"


rule gate4_stage1:
    """GATE-4 — Stage-1 segmentation quality (P2-14; PRD §2.3/§12, ADR-0004 D6 + A6).

    Grades the **minimum per-element per-nucleotide-class F1 over {Stem I, Specifier,
    Antiterminator}** — a min, never a mean — against the ADR-0004 D6 floor (>= 0.80,
    single-sourced from ``power.GATE4_F1_FLOOR``), on the A6 in-distribution population
    (scheme-A ``test`` inside ``nested_train``: 1,201 records / 1,029 clusters), with
    predictions reconciled by the frozen ADR-0005 D3+A3 operator so boundary IoU is not a
    512-grid artifact. Reports, non-gated: leave-one-order-out per-element F1 macro across
    held-out orders, Specifier exact-3-nt-codon detection, the 9-PDB label-noise ceiling,
    boundary IoU per element, and all four non-core classes.

    ⚠ The graded checkpoint is the **GATE-4 eval twin** (``exclude_gate4_eval=true``), not
    the shipped production checkpoint — which trained on this very population and therefore
    cannot be graded on it (ADR-0004 A6; PRD §10.1's two-run split). The twin is produced by
    ``slurm/p2/train_gate4_twin.sbatch`` under the CLAUDE.md §9.3 submit-ack protocol, NOT
    by this rule and NOT by the Snakemake SLURM executor (PRD §16: it submits
    programmatically and auto-retries, bypassing the --test-only preflight and the
    no-auto-resubmit rule). ``eval.gate4`` refuses to grade a checkpoint whose training
    report does not re-derive ``fold_scope == "gate4_twin_train"``.

    LOCAL (CLAUDE.md §9.1): a forward pass over ~3.0M nucleotides plus, non-gated, ~21.0M
    for the LOO read; the ``ml-dna`` env, one GPU. Kept out of ``rule all`` (its inputs are
    produced by a §9.3 SLURM job, so an unattended DAG must not schedule it).

        snakemake --cores 1 --use-conda gate4_stage1
    """
    input:
        checkpoint=_GATE4_TWIN_CKPT,
        train_report=_GATE4_TWIN_REPORT,
        provenance=_GATE4_TWIN_PROVENANCE,
        pdb_extents=_PDB_EXTENTS,
        # Read by the command, just not through it: `eval.gate4` calls
        # `window_dataset.load_gate4_eval_records`, whose `context_parquet` / `labels_parquet`
        # / `split_table` default to exactly these three paths (DEFAULT_CONTEXT,
        # DEFAULT_LABELS, DEFAULT_SPLIT_TABLE). Declaring them is what makes the DAG rebuild
        # the gate when the corpus, the labels, or the split assignment changes — and the
        # split table above is the file that DEFINES the graded population (ADR-0004 A6), so
        # dropping it because no `--flag` names it would leave GATE-4 silently stale against
        # a re-cut fold. Do not "clean up" these three. CodeRabbit, r1.
        context="data/interim/flank_context/context_v0.parquet",
        labels="data/processed/labels/labels_v0.parquet",
        splits="data/processed/splits/split_assignments.parquet",
    output:
        report=_GATE4_REPORT,
    log:
        "logs/gate4_stage1.log",
    conda:
        "../../envs/ml-dna.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.eval.gate4 run "
        "--checkpoint {input.checkpoint:q} "
        "--train-report {input.train_report:q} "
        "--checkpoint-provenance {input.provenance:q} "
        "--pdb-fixture {input.pdb_extents:q} "
        "--out {output.report:q} >{log} 2>&1"
