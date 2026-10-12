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

import os

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

    The rule **fails** when the gate fails (exit 4), when the report's own clause set refuses
    it (exit 3), or when the run is incomplete — any completeness clause false, i.e.
    ``is_science: false`` (exit 3, diverted to ``.invalid.json``) — rather than writing a
    report that says so quietly.
    """
    input:
        # NOT `items=`: Snakemake reserves that name on the io namespace and rejects the
        # rule at lint time ("items is reserved for internal use").
        score_table=_PRECISION_ITEMS,
        # The Stage-1-threshold sensitivity annex. Declared as INPUTS, not left to a
        # hand-run invocation: `_cmd_report` adds the block only when `--sensitivity` is
        # passed, and `precision_problems` treats it as optional — so a rule that omitted it
        # would silently overwrite the committed artifact with a report missing the annex
        # and still pass (CodeRabbit, PR #133). ⚠ These two annex reports are HAND-RUN,
        # COMMITTED artifacts with no producing rule in this workflow: their own item tables
        # are folded at tau = 0.7 / 0.9 and live under `data/interim/p3_16/`, which is not a
        # committed input. Snakemake therefore accepts whatever bytes are on disk, and if an
        # annex is regenerated out of band the parent report's `report_sha256` goes stale
        # without the DAG noticing — regenerate the parent whenever an annex changes.
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


# ── P3-16′-remint: the screened v2 benchmark (ADR-0005 A15) ──────────────────────────────────
#
# v1 stays byte-unchanged (A15 Pin 5); every path below is a v2 path, and each leg refuses a v1
# output path itself. The torch-free legs are rules. The two GPU legs stay hand-run, as at v1:
# `two_stage stage1` (env `ml-dna`, `--dtype float32`) writes `v2_stage1_{arm}.npz`, and
# `two_stage stage2 --shard I/N` + `merge-stage2` (env `ml-rna`) write `v2_stage2_{arm}.json`
# from the union work list. None of these targets is in `rule all`, for the reason
# `two_stage_eval` gives above: each declares a chain of DVC-tracked inputs.
#
#   snakemake -s workflow/Snakefile --use-conda precision_comparison_v2

_P3_16 = "data/interim/p3_16"
_HOST_SCREEN_DIR = f"{_P3_16}/host_screen"
_HOST_SCREEN_FASTAS = ("survivors.fa", "controls.fa", "v1_hosts.fa", "panel.fa")
_HOST_SCREEN_REPORT = "reports/p3/host_screen.json"
_HOST_SCREEN_ADMITTED = "reports/p3/host_screen_admitted.txt"
_HOST_SCREEN_REFUSED = "reports/p3/host_screen_refused.json"
_BENCHMARK_V2 = f"{_P3_16}/benchmark_v2.json"
# The post-mint FASTAs are written beside the screen's own (one prep dir, one idmap namespace),
# but into a SEPARATE tblout dir: the default (`<out-dir>/tblout`) is the screen's, and a
# post-mint tblout landing there would be read by the screen's report as a pool search.
_POSTMINT_DIR = _HOST_SCREEN_DIR
_POSTMINT_TBLOUT = f"{_HOST_SCREEN_DIR}/tblout_postmint"
_POSTMINT_REPORT = "reports/p3/benchmark_v2_postmint_screen.json"

# A15 Pin 3 Correction 2 — the matched-null floor the per-model assertion thresholds are signed
# against. A separate directory from the screen's: a null search writing its manifest into
# `_HOST_SCREEN_DIR/tblout` would overwrite the pool search's own search.json.
_NOISE_FLOOR_DIR = f"{_P3_16}/noise_floor"
_NOISE_FLOOR_TBLOUT = f"{_NOISE_FLOOR_DIR}/tblout"
_NOISE_FLOOR_REPORT = "reports/p3/postmint_noise_floor.json"
_FAMILIARITY_V2 = f"{_P3_16}/v2_familiarity.json"
_STAGE1_CHECKPOINTS = {
    "twin": "data/processed/checkpoints/stage1_gate4_twin/stage1.pt",
    "production": "data/processed/checkpoints/stage1_production/stage1.pt",
}
# The two hand-run GPU legs' outputs, declared so the DAG states what the grade rests on.
# These are the ONLY names those legs may write; `refuse_v1_output` enforces the v1 side.
_STAGE1_V2 = {arm: f"{_P3_16}/v2_stage1_{arm}.npz" for arm in _STAGE1_CHECKPOINTS}
_STAGE2_V2 = {arm: f"{_P3_16}/v2_stage2_{arm}.json" for arm in _STAGE1_CHECKPOINTS}
#: The D3 locus knobs every v2 table was built under, recorded in the folded table (A15 Pin 6).
_V2_RULE_KNOBS = (
    "threshold=0.5,min_span=50,gap_merge=10,min_distinct_elements=2,flank=50,min_order_margin=1"
)
_PRECISION_FIGURE_V2 = "figures/integration/two_stage_precision_v2.png"
_PRECISION_ITEMS_V2 = "reports/p3/two_stage_precision_items_v2.json"
_PRECISION_REPORT_V2 = "reports/two_stage_precision_v2.json"
_PRECISION_TAU_V2 = ("0.5", "0.7", "0.9")
_GRADED_TAU_V2 = "0.5"
#: The annex τ, keyed by the tag that appears in a file name: a Snakemake output cannot be a
#: function of its wildcards, so the τ a rule grades is derived from the tag it matched.
_ANNEX_TAU_V2 = {tau.replace(".", ""): tau for tau in _PRECISION_TAU_V2 if tau != _GRADED_TAU_V2}
_PRECISION_SENSITIVITY_V2 = {
    tau: f"reports/p3/two_stage_precision_v2_tau{tag}.json" for tag, tau in _ANNEX_TAU_V2.items()
}
_ANNEX_ITEMS_V2 = f"{_P3_16}/v2_items_t{{tau}}.json"


rule host_screen_prep:
    """P3-16′-remint — A15 Pin 1 over the out-of-fold mining pool, plus the Pin 2 search FASTAs.

    The fold screen (canonical N-free 32-mers on either strand, or a same-accession interval
    overlap, against every Stage-1/Stage-2 training sequence and every decoys_v0 insert) runs
    here; the content screen's FASTAs are written for `host_screen_search`. Deterministic: a
    re-run writes byte-identical FASTAs, which the report leg checks by digest.
    """
    input:
        mining_pool="data/processed/negatives/mining_pool_v0.parquet",
        decoys="data/processed/negatives/decoys_v0.parquet",
        dataset="data/processed/stage2_dataset.parquet",
        context="data/interim/flank_context/context_v0.parquet",
        labels="data/processed/labels/labels_v0.parquet",
        split_table="data/processed/splits/split_assignments.parquet",
        benchmark_v1=f"{_P3_16}/benchmark_v1.json",
        rf00230_master="data/external/refs/RF00230_master.fa",
        stage2_provenance="data/processed/checkpoints/stage2_rinalmo/aux1.0_lr1e-4/provenance.json",
    output:
        fastas=[f"{_HOST_SCREEN_DIR}/{name}" for name in _HOST_SCREEN_FASTAS],
        idmap=f"{_HOST_SCREEN_DIR}/idmap.json",
        prep=f"{_HOST_SCREEN_DIR}/prep.json",
    params:
        out_dir=lambda wildcards, output: os.path.dirname(output.prep),
    log:
        "logs/host_screen_prep.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.host_screen prep "
        "--mining-pool {input.mining_pool:q} --decoys {input.decoys:q} "
        "--dataset {input.dataset:q} --context {input.context:q} --labels {input.labels:q} "
        "--split-table {input.split_table:q} --benchmark-v1 {input.benchmark_v1:q} "
        "--rf00230-master {input.rf00230_master:q} "
        "--stage2-provenance {input.stage2_provenance:q} "
        "--out-dir {params.out_dir:q} >{log} 2>&1"


rule host_screen_search:
    """P3-16′-remint — A15 Pin 2's cmsearch: every screen FASTA × {RF00230, TBDB001}.

    `-T` per FASTA (10 bits for the screen, 0 for the v1 audit and the sensitivity panel) and
    `-Z 1`, default pipeline, both strands; the verdict reads the bit-score column, never `inc`.
    The panel × RF00230 search at -T 0 is ~95 % of the compute (~4.4 CPU-h).
    """
    input:
        fastas=[f"{_HOST_SCREEN_DIR}/{name}" for name in _HOST_SCREEN_FASTAS],
        prep=f"{_HOST_SCREEN_DIR}/prep.json",
    output:
        search=f"{_HOST_SCREEN_DIR}/tblout/search.json",
    threads: 24
    params:
        out_dir=lambda wildcards, input: os.path.dirname(input.prep),
        jobs=3,
    log:
        "logs/host_screen_search.log",
    conda:
        "../../envs/infernal.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.host_screen search "
        "--out-dir {params.out_dir:q} --jobs {params.jobs} "
        "--cpu $(( {threads} / {params.jobs} > 0 ? {threads} / {params.jobs} : 1 )) >{log} 2>&1"


rule host_screen_report:
    """P3-16′-remint — the A15 screen report, its admitted list and its refused-window annex.

    Exit 3 (and a `.invalid.json`) when the report's own clause set refuses it: every must-fire
    control re-derived, every tblout bound to the FASTA bytes it was searched from.
    """
    input:
        prep=f"{_HOST_SCREEN_DIR}/prep.json",
        idmap=f"{_HOST_SCREEN_DIR}/idmap.json",
        search=f"{_HOST_SCREEN_DIR}/tblout/search.json",
    output:
        report=_HOST_SCREEN_REPORT,
        admitted=_HOST_SCREEN_ADMITTED,
        refused=_HOST_SCREEN_REFUSED,
    params:
        out_dir=lambda wildcards, input: os.path.dirname(input.prep),
    log:
        "logs/host_screen_report.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.host_screen report "
        "--out-dir {params.out_dir:q} --out {output.report:q} "
        "--admitted {output.admitted:q} --refused {output.refused:q} >{log} 2>&1"


rule benchmark_v2_mint:
    """P3-16′-remint — the screened v2 benchmark: the v1 construction, drawn from admitted hosts.

    Same seed (42), same positives, same decoys, the leader kept in the draw (A15 Pin 3); only
    the host pool changes. The minter refuses a screen report that does not bind to this mining
    pool, and refuses a v1 output path.
    """
    input:
        dataset="data/processed/stage2_dataset.parquet",
        decoys="data/processed/negatives/decoys_v0.parquet",
        split_table="data/processed/splits/split_assignments.parquet",
        context="data/interim/flank_context/context_v0.parquet",
        labels="data/processed/labels/labels_v0.parquet",
        mining_pool="data/processed/negatives/mining_pool_v0.parquet",
        screen=_HOST_SCREEN_REPORT,
        admitted=_HOST_SCREEN_ADMITTED,
    output:
        benchmark=_BENCHMARK_V2,
    params:
        seed=config.get("benchmark_seed", 42),
    log:
        "logs/benchmark_v2_mint.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python scripts/mint_p3_16_benchmark.py "
        "--dataset {input.dataset:q} --decoys {input.decoys:q} "
        "--split-table {input.split_table:q} --context {input.context:q} "
        "--labels {input.labels:q} --mining-pool {input.mining_pool:q} "
        "--host-screen {input.screen:q} --seed {params.seed} "
        "--out {output.benchmark:q} >{log} 2>&1"


rule benchmark_v2_postmint_prep:
    """P3-16′-remint — the spliced v2 contigs as FASTA, for A15 Pin 3's post-mint assertion."""
    input:
        benchmark=_BENCHMARK_V2,
    output:
        all_contigs=f"{_POSTMINT_DIR}/postmint_all.fa",
        label0=f"{_POSTMINT_DIR}/postmint_label0.fa",
        idmap=f"{_POSTMINT_DIR}/postmint_idmap.json",
    params:
        out_dir=lambda wildcards, output: os.path.dirname(output.idmap),
    log:
        "logs/benchmark_v2_postmint_prep.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.host_screen postmint-prep "
        "--benchmark {input.benchmark:q} --out-dir {params.out_dir:q} >{log} 2>&1"


rule benchmark_v2_postmint_search:
    """P3-16′-remint — the Pin 2 configuration over the spliced contigs, plus the `--max` arm.

    The filter-off arm is scoped to the label-0 contigs, the only ones the assertion reads.
    """
    input:
        all_contigs=f"{_POSTMINT_DIR}/postmint_all.fa",
        label0=f"{_POSTMINT_DIR}/postmint_label0.fa",
        idmap=f"{_POSTMINT_DIR}/postmint_idmap.json",
    output:
        search=f"{_POSTMINT_TBLOUT}/search.json",
    threads: 24
    params:
        out_dir=lambda wildcards, input: os.path.dirname(input.idmap),
        tblout_dir=lambda wildcards, output: os.path.dirname(output.search),
        jobs=3,
    log:
        "logs/benchmark_v2_postmint_search.log",
    conda:
        "../../envs/infernal.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.host_screen search "
        "--out-dir {params.out_dir:q} --tblout-dir {params.tblout_dir:q} "
        "--fasta postmint_all.fa --fasta postmint_label0.fa "
        "--max-fasta postmint_label0.fa --jobs {params.jobs} "
        "--cpu $(( {threads} / {params.jobs} > 0 ? {threads} / {params.jobs} : 1 )) >{log} 2>&1"


rule postmint_noise_floor_prep:
    """P3-16′-remint — the three matched nulls A15 Pin 3's thresholds are calibrated against.

    Deterministic at seed 42. Refuses (exit 1) unless every matchedness control fires on the
    bytes it wrote: a control silently replaced by a copy of its positive reads as "no power"
    rather than as a broken control.
    """
    input:
        benchmark=_BENCHMARK_V2,
        idmap=f"{_POSTMINT_DIR}/postmint_idmap.json",
        label0=f"{_POSTMINT_DIR}/postmint_label0.fa",
        admitted=_HOST_SCREEN_ADMITTED,
        pool="data/processed/negatives/mining_pool_v0.parquet",
    output:
        prep=f"{_NOISE_FLOOR_DIR}/prep.json",
        n1=f"{_NOISE_FLOOR_DIR}/n1_inserts.fa",
        n2a=f"{_NOISE_FLOOR_DIR}/n2_scramble1.fa",
        n2b=f"{_NOISE_FLOOR_DIR}/n2_scramble2.fa",
        n3=f"{_NOISE_FLOOR_DIR}/n3_pool.fa",
    params:
        out_dir=lambda wildcards, output: os.path.dirname(output.prep),
        screen_dir=lambda wildcards, input: os.path.dirname(input.idmap),
    log:
        "logs/postmint_noise_floor_prep.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.noise_floor prep "
        "--benchmark {input.benchmark:q} --screen-dir {params.screen_dir:q} "
        "--admitted {input.admitted:q} --mining-pool {input.pool:q} "
        "--out-dir {params.out_dir:q} >{log} 2>&1"


rule postmint_noise_floor_search:
    """P3-16′-remint — every null arm x model x pipeline, at `-T 0`.

    `-T 0` is load-bearing and not a default: a null searched at the threshold it is meant to
    calibrate reports nothing below that threshold, so the measured maximum would be an artefact
    of the flag and the measurement would appear to support any threshold at all.
    """
    input:
        prep=f"{_NOISE_FLOOR_DIR}/prep.json",
        n1=f"{_NOISE_FLOOR_DIR}/n1_inserts.fa",
        n2a=f"{_NOISE_FLOOR_DIR}/n2_scramble1.fa",
        n2b=f"{_NOISE_FLOOR_DIR}/n2_scramble2.fa",
        n3=f"{_NOISE_FLOOR_DIR}/n3_pool.fa",
    output:
        search=f"{_NOISE_FLOOR_TBLOUT}/search.json",
    threads: 24
    params:
        out_dir=lambda wildcards, input: os.path.dirname(input.prep),
        tblout_dir=lambda wildcards, output: os.path.dirname(output.search),
        jobs=4,
    log:
        "logs/postmint_noise_floor_search.log",
    conda:
        "../../envs/infernal.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.noise_floor search "
        "--out-dir {params.out_dir:q} --tblout-dir {params.tblout_dir:q} "
        "--jobs {params.jobs} "
        "--cpu $(( {threads} / {params.jobs} > 0 ? {threads} / {params.jobs} : 1 )) >{log} 2>&1"


rule postmint_noise_floor_report:
    """P3-16′-remint — the measured per-model floor, and the matchedness re-derivation.

    Exit 1 on any problem. `host_screen postmint-report` refuses to consume an artifact whose
    own `problems` list is non-empty, so a refused measurement cannot calibrate a threshold.
    """
    input:
        prep=f"{_NOISE_FLOOR_DIR}/prep.json",
        search=f"{_NOISE_FLOOR_TBLOUT}/search.json",
        benchmark=_BENCHMARK_V2,
        idmap=f"{_POSTMINT_DIR}/postmint_idmap.json",
    output:
        report=_NOISE_FLOOR_REPORT,
    params:
        out_dir=lambda wildcards, input: os.path.dirname(input.prep),
        tblout_dir=lambda wildcards, input: os.path.dirname(input.search),
        screen_dir=lambda wildcards, input: os.path.dirname(input.idmap),
    log:
        "logs/postmint_noise_floor_report.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.noise_floor report "
        "--out-dir {params.out_dir:q} --tblout-dir {params.tblout_dir:q} "
        "--benchmark {input.benchmark:q} --screen-dir {params.screen_dir:q} "
        "--out {output.report:q} >{log} 2>&1"


rule benchmark_v2_postmint_report:
    """P3-16′-remint — A15 Pin 3: no gated contig carries a ≥ 10-bit hit or the motif.

    Exit 4 when the assertion fails. The only remedy A15 allows is a re-signed screen and a full
    re-mint; nothing downstream may run on a benchmark that failed it.
    """
    input:
        idmap=f"{_POSTMINT_DIR}/postmint_idmap.json",
        search=f"{_POSTMINT_TBLOUT}/search.json",
    output:
        report=_POSTMINT_REPORT,
    params:
        out_dir=lambda wildcards, input: os.path.dirname(input.idmap),
        tblout_dir=lambda wildcards, input: os.path.dirname(input.search),
    log:
        "logs/benchmark_v2_postmint_report.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.host_screen postmint-report "
        "--out-dir {params.out_dir:q} --tblout-dir {params.tblout_dir:q} "
        "--out {output.report:q} >{log} 2>&1"


rule benchmark_v2_familiarity:
    """P3-16′-remint — A15 Pin 7(b)'s flank-familiarity score, per item per Stage-1 arm."""
    input:
        benchmark=_BENCHMARK_V2,
        mining_pool="data/processed/negatives/mining_pool_v0.parquet",
        decoys="data/processed/negatives/decoys_v0.parquet",
        context="data/interim/flank_context/context_v0.parquet",
        labels="data/processed/labels/labels_v0.parquet",
        split_table="data/processed/splits/split_assignments.parquet",
    output:
        familiarity=_FAMILIARITY_V2,
    log:
        "logs/benchmark_v2_familiarity.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.host_screen familiarity "
        "--benchmark {input.benchmark:q} --out {output.familiarity:q} "
        "--mining-pool {input.mining_pool:q} --decoys {input.decoys:q} "
        "--context {input.context:q} --labels {input.labels:q} "
        "--split-table {input.split_table:q} >{log} 2>&1"


rule two_stage_payloads_v2:
    """P3-16′-remint — the Stage-2 work list for one arm at one Stage-1 threshold (torch-free).

    One rule per (arm, τ) via wildcards, over the hand-run `v2_stage1_{arm}.npz`. The knobs are
    the same overridable `config.get` defaults the v1 legs used, and the τ sweep's three work
    lists are unioned by `two_stage_payloads_union_v2` so Stage-2 scores each row once.
    """
    input:
        benchmark=_BENCHMARK_V2,
        stage1=lambda wildcards: _STAGE1_V2[wildcards.arm],
    output:
        payloads=f"{_P3_16}/v2_payloads_{{arm}}_t{{tau}}.json",
    wildcard_constraints:
        arm="|".join(_STAGE1_V2),
        tau="|".join(_PRECISION_TAU_V2),
    params:
        threshold_scope=config.get("two_stage_threshold_scope", "global"),
        min_span=config.get("two_stage_min_span", 50),
        gap_merge=config.get("two_stage_gap_merge", 10),
        min_distinct_elements=config.get("two_stage_min_distinct_elements", 2),
        flank=config.get("two_stage_flank", 50),
        min_order_margin=config.get("two_stage_min_order_margin", 1),
    log:
        "logs/two_stage_payloads_v2_{arm}_t{tau}.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.two_stage payloads "
        "--contigs {input.benchmark:q} --stage1 {input.stage1:q} "
        "--out {output.payloads:q} "
        "--threshold-scope {params.threshold_scope:q} --threshold {wildcards.tau} "
        "--min-span {params.min_span} --gap-merge {params.gap_merge} "
        "--min-distinct-elements {params.min_distinct_elements} "
        "--flank {params.flank} --min-order-margin {params.min_order_margin} >{log} 2>&1"


rule two_stage_payloads_union_v2:
    """P3-16′-remint — the τ sweep's three work lists → the one list Stage-2 scores.

    A row's id is its content address, so the union gives every τ's replay each logit it needs
    from a single GPU pass. v1's union was assembled by hand; this is that step as a leg.
    """
    input:
        payloads=lambda wildcards: [
            f"{_P3_16}/v2_payloads_{wildcards.arm}_t{tau}.json" for tau in _PRECISION_TAU_V2
        ],
    output:
        union=f"{_P3_16}/v2_payloads_{{arm}}_union.json",
    wildcard_constraints:
        arm="|".join(_STAGE1_V2),
    log:
        "logs/two_stage_payloads_union_v2_{arm}.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.two_stage payloads-union "
        "--manifests {input.payloads:q} --out {output.union:q} >{log} 2>&1"


rule two_stage_table_v2:
    """P3-16′-remint — replay one arm at one τ into its §13.1 candidate table (torch-free).

    Reads the hand-run Stage-1 logits and the merged Stage-2 logits; `T` is derived from the
    GATE-2 report rather than restated, so a re-fit moves both together. No `--expect-digest`:
    v2 has no committed fixture to reproduce — it is the artifact being produced.
    """
    input:
        benchmark=_BENCHMARK_V2,
        stage1=lambda wildcards: _STAGE1_V2[wildcards.arm],
        stage2=lambda wildcards: _STAGE2_V2[wildcards.arm],
        gate2=_GATE2_REPORT,
    output:
        table=f"{_P3_16}/v2_table_{{arm}}_t{{tau}}.json",
        report=f"{_P3_16}/v2_strand_robustness_{{arm}}_t{{tau}}.json",
    wildcard_constraints:
        arm="|".join(_STAGE1_V2),
        tau="|".join(_PRECISION_TAU_V2),
    params:
        threshold_scope=config.get("two_stage_threshold_scope", "global"),
        min_span=config.get("two_stage_min_span", 50),
        gap_merge=config.get("two_stage_gap_merge", 10),
        min_distinct_elements=config.get("two_stage_min_distinct_elements", 2),
        flank=config.get("two_stage_flank", 50),
        min_order_margin=config.get("two_stage_min_order_margin", 1),
        operating_point=config.get("two_stage_operating_point", 0.5),
    log:
        "logs/two_stage_table_v2_{arm}_t{tau}.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.two_stage run "
        "--contigs {input.benchmark:q} --stage1 {input.stage1:q} --stage2 {input.stage2:q} "
        "--report {output.report:q} --table {output.table:q} "
        "--threshold-scope {params.threshold_scope:q} --threshold {wildcards.tau} "
        "--min-span {params.min_span} --gap-merge {params.gap_merge} "
        "--min-distinct-elements {params.min_distinct_elements} "
        "--flank {params.flank} --min-order-margin {params.min_order_margin} "
        "--temperature-from {input.gate2:q} "
        "--stage2-operating-point {params.operating_point} >{log} 2>&1"


rule precision_fold_v2:
    """P3-16′-remint — the two v2 candidate tables → the replayable per-item score table (1.2).

    Every v2 input is hash-recorded here (A15 Pin 6): both Stage-1 archives, both Stage-2
    artifacts with their identity and shard layout, both τ-0.5 work lists, the GATE-2 report
    `T` was read from, the screen report, the post-mint assertion and the familiarity scores.
    The fold refuses unless all of them are given for a schema-1.1 manifest.
    """
    input:
        benchmark=_BENCHMARK_V2,
        twin_table=f"{_P3_16}/v2_table_twin_t0.5.json",
        production_table=f"{_P3_16}/v2_table_production_t0.5.json",
        twin_stage1=_STAGE1_V2["twin"],
        production_stage1=_STAGE1_V2["production"],
        twin_stage2=_STAGE2_V2["twin"],
        production_stage2=_STAGE2_V2["production"],
        twin_payloads=f"{_P3_16}/v2_payloads_twin_t0.5.json",
        production_payloads=f"{_P3_16}/v2_payloads_production_t0.5.json",
        twin_checkpoint=_STAGE1_CHECKPOINTS["twin"],
        production_checkpoint=_STAGE1_CHECKPOINTS["production"],
        gate2=_GATE2_REPORT,
        screen=_HOST_SCREEN_REPORT,
        postmint=_POSTMINT_REPORT,
        familiarity=_FAMILIARITY_V2,
    output:
        score_table=_PRECISION_ITEMS_V2,
    params:
        stage1_threshold=_GRADED_TAU_V2,
        rule_knobs=_V2_RULE_KNOBS,
    log:
        "logs/precision_fold_v2.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.precision fold "
        "--benchmark {input.benchmark:q} "
        "--twin-table {input.twin_table:q} --production-table {input.production_table:q} "
        "--twin-checkpoint {input.twin_checkpoint:q} "
        "--production-checkpoint {input.production_checkpoint:q} "
        "--stage1-threshold {params.stage1_threshold} "
        "--twin-stage1 {input.twin_stage1:q} --production-stage1 {input.production_stage1:q} "
        "--twin-stage2 {input.twin_stage2:q} --production-stage2 {input.production_stage2:q} "
        "--twin-payloads {input.twin_payloads:q} "
        "--production-payloads {input.production_payloads:q} "
        "--temperature-source {input.gate2:q} --host-screen {input.screen:q} "
        "--postmint {input.postmint:q} --familiarity {input.familiarity:q} "
        "--rule-knobs {params.rule_knobs:q} --out {output.score_table:q} >{log} 2>&1"


rule precision_fold_v2_sensitivity:
    """P3-16′-remint — the annex τ's own folded item table (A15's sweep, re-graded on v2).

    The same fold as `precision_fold_v2`, at the annex Stage-1 threshold: each annex is graded
    on its OWN item table, never on the τ = 0.5 one, so its matched-recall read is its own.
    """
    input:
        benchmark=_BENCHMARK_V2,
        twin_table=f"{_P3_16}/v2_table_twin_t{{tau}}.json",
        production_table=f"{_P3_16}/v2_table_production_t{{tau}}.json",
        twin_stage1=_STAGE1_V2["twin"],
        production_stage1=_STAGE1_V2["production"],
        twin_stage2=_STAGE2_V2["twin"],
        production_stage2=_STAGE2_V2["production"],
        twin_payloads=f"{_P3_16}/v2_payloads_twin_t{{tau}}.json",
        production_payloads=f"{_P3_16}/v2_payloads_production_t{{tau}}.json",
        twin_checkpoint=_STAGE1_CHECKPOINTS["twin"],
        production_checkpoint=_STAGE1_CHECKPOINTS["production"],
        gate2=_GATE2_REPORT,
        screen=_HOST_SCREEN_REPORT,
        postmint=_POSTMINT_REPORT,
        familiarity=_FAMILIARITY_V2,
    output:
        score_table=_ANNEX_ITEMS_V2,
    wildcard_constraints:
        tau="|".join(_ANNEX_TAU_V2.values()),
    params:
        rule_knobs=lambda wildcards: _V2_RULE_KNOBS.replace(
            f"threshold={_GRADED_TAU_V2}", f"threshold={wildcards.tau}"
        ),
    log:
        "logs/precision_fold_v2_sensitivity_t{tau}.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.precision fold "
        "--benchmark {input.benchmark:q} "
        "--twin-table {input.twin_table:q} --production-table {input.production_table:q} "
        "--twin-checkpoint {input.twin_checkpoint:q} "
        "--production-checkpoint {input.production_checkpoint:q} "
        "--stage1-threshold {wildcards.tau} "
        "--twin-stage1 {input.twin_stage1:q} --production-stage1 {input.production_stage1:q} "
        "--twin-stage2 {input.twin_stage2:q} --production-stage2 {input.production_stage2:q} "
        "--twin-payloads {input.twin_payloads:q} "
        "--production-payloads {input.production_payloads:q} "
        "--temperature-source {input.gate2:q} --host-screen {input.screen:q} "
        "--postmint {input.postmint:q} --familiarity {input.familiarity:q} "
        "--rule-knobs {params.rule_knobs:q} --out {output.score_table:q} >{log} 2>&1"


rule precision_sensitivity_v2:
    """P3-16′-remint — the Stage-1-threshold annex: the same grade at τ = 0.7 / 0.9.

    Graded by the same leg without `--sensitivity`, so each annex is itself a schema-1.2 report
    bound to v2's benchmark digest and to its own τ's reads.
    """
    input:
        score_table=lambda wildcards: _ANNEX_ITEMS_V2.format(
            tau=_ANNEX_TAU_V2[wildcards.tau_tag]
        ),
    output:
        report="reports/p3/two_stage_precision_v2_tau{tau_tag}.json",
    wildcard_constraints:
        tau_tag="|".join(_ANNEX_TAU_V2),
    params:
        operating_point=config.get("two_stage_operating_point", 0.5),
        decoy_prevalence=config.get("benchmark_decoy_prevalence", 100),
        n_boot=config.get("precision_n_boot", 2000),
        seed=config.get("precision_seed", 42),
    log:
        "logs/precision_sensitivity_v2_tau{tau_tag}.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.precision report "
        "--items {input.score_table:q} --out {output.report:q} "
        "--stage2-operating-point {params.operating_point} "
        "--decoy-prevalence {params.decoy_prevalence} "
        "--n-boot {params.n_boot} --seed {params.seed} >{log} 2>&1"


rule precision_comparison_v2:
    """P3-16′-remint — the A13 Pin 1 gate, re-graded on the screened benchmark (ADR-0005 A15).

    Fails when the gate fails (exit 4), when the report's own clause set refuses it (exit 3),
    or when the run is incomplete. A15 Pin 5's power clause means a gated arm with no
    candidate-bearing negative is graded `underpowered`: the verdict A13 computes is unchanged,
    and the class is carried beside it rather than substituted for it.
    """
    input:
        score_table=_PRECISION_ITEMS_V2,
        sensitivity=[_PRECISION_SENSITIVITY_V2[tau] for tau in _PRECISION_SENSITIVITY_V2],
    output:
        report=_PRECISION_REPORT_V2,
    params:
        sensitivity=" ".join(
            f"--sensitivity {tau}={path}" for tau, path in sorted(_PRECISION_SENSITIVITY_V2.items())
        ),
        operating_point=config.get("two_stage_operating_point", 0.5),
        decoy_prevalence=config.get("benchmark_decoy_prevalence", 100),
        n_boot=config.get("precision_n_boot", 2000),
        seed=config.get("precision_seed", 42),
    log:
        "logs/precision_comparison_v2.log",
    conda:
        "../../envs/data.yml"
    shell:
        "PYTHONPATH=src python -m tbox_finder.integration.precision report "
        "--items {input.score_table:q} --out {output.report:q} "
        "--stage2-operating-point {params.operating_point} "
        "--decoy-prevalence {params.decoy_prevalence} "
        "--n-boot {params.n_boot} --seed {params.seed} "
        "{params.sensitivity} >{log} 2>&1"


rule precision_figure_v2:
    """P3-16′-remint — the v2 PR-curve figure, with the familiarity baseline and power note."""
    input:
        report=_PRECISION_REPORT_V2,
    output:
        figure=_PRECISION_FIGURE_V2,
    log:
        "logs/precision_figure_v2.log",
    conda:
        "../../envs/viz.yml"
    shell:
        "PYTHONPATH=src python scripts/plot_two_stage_precision.py "
        "--report {input.report:q} --out {output.figure:q} >{log} 2>&1"
