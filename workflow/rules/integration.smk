# Cross-stage integration rules (P3-14 onward) — the PRD §6 two-stage path, assembled.
#
# NOTE ON INCLUDE ORDER: ``workflow/Snakefile`` globs ``rules/*.smk`` in sorted order, so
# this module is processed after ``common``/``data``/``eval`` and before ``setup``/``stage2``.
# It declares its own constants rather than borrowing another module's, exactly as
# ``eval.smk`` does. Helper *functions* belong in ``common.smk``: ``snakemake --lint``
# refuses "Mixed rules and functions in same snakefile" and a red lint is CI-blocking.
#
# ENV CHOICE: ``data``. The rule below is the harness's **replay** leg and is torch-free by
# construction — the two model legs (``two_stage stage1`` in ``ml-dna``,
# ``two_stage stage2`` in ``ml-rna``) are hand-run because they need GPUs and two mutually
# incompatible envs, the same reason every GPU step in this repo is a hand-authored sbatch
# rather than a Snakemake rule. Their outputs are committed fixtures, so the replay is
# fully reproducible from the repo alone.
#
# NOT IN ``rule all``, and deliberately: the GATE-2 report is a declared *input*, so asking
# Snakemake for this target on a checkout whose DVC data is absent makes it plan to rebuild
# ``gate2_ece`` (and the Stage-2 scoring beneath it) first. That dependency is real — a
# re-fitted temperature makes this artifact stale — so it is declared rather than hidden in
# ``params``, and the cost is that this is a regression target you ask for, not part of the
# default build. The equivalent check with no rebuild chain is
# ``pytest tests/golden/test_two_stage_golden.py``, which replays from committed files only.
#
#   snakemake -s workflow/Snakefile --use-conda two_stage_eval

_TWO_STAGE_FIXTURE = "tests/fixtures/two_stage"
_TWO_STAGE_REPORT = "reports/strand_robustness.json"
_TWO_STAGE_TABLE = "reports/p3/two_stage_candidates.json"
_GATE2_REPORT = "reports/gate2_p3_ece.json"
_PRECISION_ITEMS = "reports/p3/two_stage_precision_items.json"
_PRECISION_REPORT = "reports/two_stage_precision.json"
_PRECISION_TAU = ("0.7", "0.9")
_PRECISION_SENSITIVITY = {
    tau: f"reports/p3/two_stage_precision_tau{tau.replace('.', '')}.json"
    for tau in _PRECISION_TAU
}


rule two_stage_eval:
    """P3-14 — replay the committed two-stage fixture into a candidate table + the D15 diagnostic.

    Runs the whole PRD §6 path over real Stage-1 and Stage-2 output: reconcile (ADR-0005 A3)
    → construct_loci (D3) → resolve_strand (D15) → transcribe (T→U, sequence-only) →
    calibrated Stage-2 re-rank → the §13.1 candidate table, then measures the
    strand-robustness diagnostic by re-scoring every locus on the strand the resolver did
    *not* choose.

    On the rule values, precisely — because an earlier draft of this docstring claimed a
    discipline the ``params`` block does not apply, and CodeRabbit r1 was right to catch it.
    The **module** defaults nothing: every knob on ``run_two_stage`` is keyword-only with no
    default and the CLI marks each ``required=True``, so no caller can take a value by omission.
    This rule then supplies, via overridable ``config.get`` defaults, **the values the committed
    fixture was minted under** — because its job is to reproduce that artifact, and requiring a
    config file to do so would make the regression unrunnable on a fresh clone. That is safe
    only because ``--expect-digest`` is passed: a changed value that changes the outcome fails
    the rule rather than quietly writing a different table. None of these is an ADR freeze;
    ADR-0005 D3/D15 freeze the production values at the §13.1 phase gate (P5-01) and the written
    report carries ``rule.pinned: false``. The temperature is **derived** from the GATE-2 report
    rather than restated, so a re-fit moves both together.

    ``--expect-digest`` makes the rule fail on a drift the golden test would also catch, so a
    pipeline run and CI cannot disagree about whether the harness still reproduces its fixture.
    """
    input:
        contigs=f"{_TWO_STAGE_FIXTURE}/contigs.json",
        stage1=f"{_TWO_STAGE_FIXTURE}/stage1_window_logits.npz",
        stage2=f"{_TWO_STAGE_FIXTURE}/stage2_logits.json",
        expected=f"{_TWO_STAGE_FIXTURE}/expected.sha256",
        gate2=_GATE2_REPORT,
    output:
        report=_TWO_STAGE_REPORT,
        table=_TWO_STAGE_TABLE,
    params:
        # The rule this committed fixture was built under. Provisional, never a freeze —
        # `rule.pinned` is false in the written report and a test asserts it.
        threshold_scope=config.get("two_stage_threshold_scope", "global"),
        threshold=config.get("two_stage_threshold", 0.9),
        min_span=config.get("two_stage_min_span", 50),
        gap_merge=config.get("two_stage_gap_merge", 10),
        min_distinct_elements=config.get("two_stage_min_distinct_elements", 2),
        flank=config.get("two_stage_flank", 50),
        min_order_margin=config.get("two_stage_min_order_margin", 1),
        operating_point=config.get("two_stage_operating_point", 0.5),
        expect_digest=lambda wildcards, input: open(input.expected).read().strip(),
        env_lock="envs/data.conda-lock.yml",
    log:
        "logs/two_stage_eval.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.two_stage run "
        "--contigs {input.contigs:q} "
        "--stage1 {input.stage1:q} "
        "--stage2 {input.stage2:q} "
        "--report {output.report:q} "
        "--table {output.table:q} "
        "--threshold-scope {params.threshold_scope:q} "
        "--threshold {params.threshold} "
        "--min-span {params.min_span} "
        "--gap-merge {params.gap_merge} "
        "--min-distinct-elements {params.min_distinct_elements} "
        "--flank {params.flank} "
        "--min-order-margin {params.min_order_margin} "
        "--temperature-from {input.gate2:q} "
        "--stage2-operating-point {params.operating_point} "
        "--expect-digest {params.expect_digest:q} >{log} 2>&1"


rule precision_comparison:
    """P3-16 — the P3 exit gate: two-stage vs Stage-1-only precision at matched recall.

    Torch-free and CI-runnable by construction, the same split ``two_stage_eval`` uses. The
    GPU work — two Stage-1 passes (the GATE-4 eval twin, which is the **gated** arm, and the
    shipped scanner, reported beside it) and their Stage-2 re-scoring — is hand-run in
    ``ml-dna`` / ``ml-rna`` and folded down to the committed per-item score table this rule
    reads. That table is one row per benchmark item per arm; everything the grade depends on
    is arithmetic over it, so a reviewer can re-derive the published verdict from the repo
    alone without a GPU.

    ``--stage2-operating-point`` is passed explicitly and has **no module default**:
    ADR-0005 D3 freezes it at the §13.1 phase gate (P5-01), and the written report carries
    ``rule.pinned: false``. The value here is the one the harness fixture was minted under,
    supplied via an overridable ``config.get`` so a fresh clone can reproduce the artifact
    without a config file — the same reasoning ``two_stage_eval`` documents above.

    The rule **fails** when the gate fails (exit 4) or when the report's own clause set
    refuses it (exit 3), rather than writing a report that says so quietly.
    """
    input:
        # NOT `items=`: Snakemake reserves that name on the io namespace and rejects the
        # rule at lint time ("items is reserved for internal use").
        score_table=_PRECISION_ITEMS,
        # The Stage-1-threshold sensitivity annex. Declared as INPUTS, not left to a
        # hand-run invocation: `_cmd_report` adds the block only when `--sensitivity` is
        # passed, and `precision_problems` treats it as optional — so a rule that omitted it
        # would silently overwrite the committed artifact with a report missing the annex
        # and still pass (CodeRabbit, PR #133).
        sensitivity=[_PRECISION_SENSITIVITY[tau] for tau in _PRECISION_TAU],
    output:
        report=_PRECISION_REPORT,
    params:
        sensitivity=" ".join(
            f"--sensitivity {tau}={_PRECISION_SENSITIVITY[tau]}" for tau in _PRECISION_TAU
        ),
        operating_point=config.get("two_stage_operating_point", 0.5),
        decoy_prevalence=config.get("benchmark_decoy_prevalence", 100),
        n_boot=config.get("precision_n_boot", 2000),
        seed=config.get("precision_seed", 42),
    log:
        "logs/precision_comparison.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.precision report "
        "--items {input.score_table:q} "
        "--out {output.report:q} "
        "--stage2-operating-point {params.operating_point} "
        "--decoy-prevalence {params.decoy_prevalence} "
        "--n-boot {params.n_boot} "
        "--seed {params.seed} "
        "{params.sensitivity} >{log} 2>&1"
