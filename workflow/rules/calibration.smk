# Calibration rules — GATE-2's P3 half (P3-10 onward).
#
# NOTE ON INCLUDE ORDER: ``workflow/Snakefile`` globs ``rules/*.smk`` in sorted order, so
# this module is processed after ``backbones.smk`` and before ``common.smk`` — it declares
# its own constants rather than borrowing another module's, exactly as ``eval.smk`` does.
# Helper *functions* belong in ``common.smk``: ``snakemake --lint`` refuses "Mixed rules and
# functions in same snakefile" and a red lint is CI-blocking.

_GATE2_REPORT = "reports/gate2_p3_ece.json"
_GATE2_FIGURE_DATA = "reports/p3/gate2_figure_data.json"
_STAGE2_SCORES = "reports/p3/stage2_scores.json"
_STAGE2_LOO_SCORES = "reports/p3/stage2_scores_loo.json"
_STAGE2_DATASET = "data/processed/stage2_dataset.parquet"
# Which model produced `_STAGE2_SCORES` / `_STAGE2_LOO_SCORES`. Those sidecars were written
# BEFORE ADR-0002 A15 — when this repo pinned exactly one Stage-2 backbone — so they record no
# `load.backbone`, and `grade` refuses to guess one rather than defaulting to production. The
# value is retyped here because importing `tbox_finder` at Snakefile-parse time would make
# `snakemake --lint` depend on `src` being importable; `tests/unit/test_gate2.py` pins it
# against `rna_backbone_registry.PRODUCTION_BACKBONE` so the copy cannot drift.
_SCORED_BACKBONE = "rinalmo-giga"
_GATE2_FIGURES = [
    "figures/calib/gate2_reliability.png",
    "figures/calib/gate2_ood_by_order.png",
]


rule stage2_scores_loo:
    """Stage-2 production-arm logits over the ADR-0004 D5 leave-one-order-out holdout (P3-10).

    P3-08 scored the random split's calib/val/test rungs only, so the leave-clade-out
    population the D13 drift read needs — the 9,345 rows the nested fold designates as
    held out, spanning 30 orders — had no logits at all. This rule produces them with the
    **production** arm (``aux1.0_lr1e-4``), reusing ``stage2.eval``'s checkpoint loader and
    scorer rather than a second implementation, so the adapter-really-loaded verification
    and the ``(n_tokens, row_id)`` batch ordering are the same code that produced the
    in-distribution scores.

    The producer REFUSES to run if the holdout is not disjoint from Stage-2 training —
    either by row or, the failure a row check cannot see, by taxonomic **order**. A leak of
    the second kind yields an in-distribution number labelled OOD.

    LOCAL (CLAUDE.md §9.1): ~9.3k forward passes at batch 4 on one GPU, measured at
    0.174 s/row (~28 min wall). Kept out of ``rule all`` — it needs the P3-06 checkpoints,
    which are DVC-tracked and produced by a §9.3 SLURM job.

        snakemake --cores 1 --use-conda stage2_scores_loo
    """
    input:
        dataset=_STAGE2_DATASET,
    output:
        scores=_STAGE2_LOO_SCORES,
    log:
        "logs/stage2_scores_loo.log",
    conda:
        "../../envs/ml-rna.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.calib.gate2 score-loo "
        "--dataset {input.dataset:q} "
        "--out {output.scores:q} >{log} 2>&1"


rule gate2_ece:
    """GATE-2, P3 half — in-distribution ECE on the named posterior (P3-10).

    Gated: the ADR-0005 **D11** binned ECE (15 equal-mass bins, debiased) of the
    **temperature-scaled, PRE-prior-shift** posterior on the ``test`` rung, at that rung's
    own prevalence, against the blinded-frozen ``<= 0.05`` default. ``T`` is fitted on the
    disjoint ``calib`` split and nowhere else.

    Reported, never gated: the deployment-prevalence ECE swept across the PRD's prose
    ``~10^3-10^4 : 1`` band (no single deployment prior is pinned — that would be a new
    blinded-frozen default, CLAUDE.md §7 item 2), and the per-held-out-order **D13** OOD ECE
    with block-resampled CIs and the Amendment-A2 ``OOD_ECE_MIN_N = 20`` admissibility flag.
    D13's drift bound is unpinned repo-wide, so condition (i) — and therefore any
    calibrated-negative PASS — is left unadjudicated rather than derived from two of three
    conditions.

    LOCAL (CLAUDE.md §9.1): numpy only. The leave-clade-out CIs are a pure-Python
    leave-one-out kernel bootstrap dominated by the largest held-out order (Lactobacillales,
    5,763 rows) — ~40 min single-core at the default ``n_boot``.

        snakemake --cores 1 --use-conda gate2_ece
    """
    input:
        dataset=_STAGE2_DATASET,
        scores=_STAGE2_SCORES,
        loo_scores=_STAGE2_LOO_SCORES,
    output:
        report=_GATE2_REPORT,
        figure_data=_GATE2_FIGURE_DATA,
    params:
        scored_backbone=_SCORED_BACKBONE,
    log:
        "logs/gate2_ece.log",
    conda:
        "../../envs/ml-rna.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.calib.gate2 grade "
        "--dataset {input.dataset:q} "
        "--scores {input.scores:q} "
        "--loo-scores {input.loo_scores:q} "
        "--report {output.report:q} "
        "--figure-data {output.figure_data:q} "
        "--scored-backbone {params.scored_backbone:q} >{log} 2>&1"


rule plot_gate2_figures:
    """Render the GATE-2 reliability diagram + the per-held-out-order OOD panel (P3-10).

    Split from ``gate2_ece`` so the grading stays in the Stage-2 env and only the drawing
    needs ``viz`` — the same split ``plot_coverage_figures`` uses. Figures read the
    figure-data JSON and recompute nothing.

        snakemake --cores 1 --use-conda plot_gate2_figures
    """
    input:
        figure_data=_GATE2_FIGURE_DATA,
    output:
        figures=_GATE2_FIGURES,
    log:
        "logs/plot_gate2_figures.log",
    conda:
        "../../envs/viz.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.calib.gate2 plot-figures "
        "--figure-data {input.figure_data:q} "
        "--figures-dir figures/calib >{log} 2>&1"
