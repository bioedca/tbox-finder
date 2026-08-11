"""Unit tests for the P3-15'-g-iii curated-control producer run plan.

Every number the §9.3 ack quotes is derived here from a committed artifact, so every
derivation is exercised on inputs whose right answer is known by hand — the committed
report cannot test the code that wrote it ([[artifact-pinning-test-cannot-see-the-code]]).
Fixtures use non-round values (5 candidates at shard size 2, 7.5 s/candidate) so a
hardcoded constant cannot pass ([[pinned-constant-that-nothing-reads]]).

The clause set is exercised by breaking **one clause at a time** against an
otherwise-valid fixture: an all-TRUE fixture cannot tell ``all(clauses)`` from ``True``
([[all-true-fixture-cannot-test-a-conjunction]]).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder.mining import curated_control_run as ccr
from tbox_finder.mining.covariation_producer import (
    CandidateSpec,
    read_candidate_manifest,
    resolve_shard_queries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_PLAN = REPO_ROOT / ccr.DEFAULT_OUT
COMMITTED_MANIFEST = REPO_ROOT / ccr.DEFAULT_MANIFEST
COMMITTED_FASTA = REPO_ROOT / ccr.DEFAULT_QUERY_FASTA
COMMITTED_SBATCH = REPO_ROOT / ccr.DEFAULT_SBATCH


# --------------------------------------------------------------------------- #
# Fixtures — a small, hand-checkable supply
# --------------------------------------------------------------------------- #
def _specs(n: int = 5, length: int = 7) -> list[CandidateSpec]:
    """``n`` candidates, each spanning exactly ``length`` nt at a distinct offset."""
    return [
        CandidateSpec(
            candidate_id=f"acc{i}:c0:{10 * i}-{10 * i + length}",
            accession=f"acc{i}:c0",
            locus_start=10 * i,
            locus_end=10 * i + length,
        )
        for i in range(n)
    ]


def _fasta(
    tmp_path: Path,
    specs,
    *,
    drop: str | None = None,
    truncate: str | None = None,
    name: str = "queries.fa",
) -> Path:
    # ⚠ `name` is not decoration: a variant written to the same path as the valid fixture is
    # silently overwritten by whichever call runs last, and the mutation under test disappears.
    path = tmp_path / name
    lines = []
    for spec in specs:
        if spec.candidate_id == drop:
            continue
        span = spec.locus_end - spec.locus_start
        seq = "ACGT" * span
        seq = seq[: span - 1] if spec.candidate_id == truncate else seq[:span]
        lines += [f">{spec.candidate_id}", seq]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sizing() -> dict:
    return {
        "worst_case_per_candidate_s": 7.5,
        "expected_per_candidate_s": 1.25,
        "align_timeout_s": 600.0,
        "cpus_per_task": 2,
        "expected_wall_caveat": "the expected column extrapolates a different population",
        "measured_from": "a fixture",
    }


def _sbatch_facts() -> dict:
    return {
        "path": "slurm/p2/mine_round_producer.sbatch",
        "wall_limit_h": 12.0,
        "cutoffs": {
            "engine": "nhmmer",
            "evalue": 100.0,
            "max_target_seqs": 500,
            "min_pident": 40.0,
            "min_cov": 0.5,
        },
        "has_query_fasta_guard": True,
        "passes_query_fasta_argument": True,
    }


def _certified() -> dict:
    return dict(_sbatch_facts()["cutoffs"])


def _good_clause_kwargs(tmp_path: Path, round_dir: str = ccr.DEFAULT_ROUND_DIR):
    specs = _specs()
    fasta = _fasta(tmp_path, specs)
    layout = ccr.shard_layout(specs, shard_size=2)
    sizing = _sizing()
    submit = ccr.submit_plan(
        layout,
        sizing,
        manifest="data/m.json",
        query_fasta="data/q.fasta",
        query_fasta_sha256="0" * 64,
        sbatch="slurm/p2/mine_round_producer.sbatch",
        round_dir=round_dir,
    )
    return specs, {
        "query_fasta": fasta,
        "layout": layout,
        "sizing": sizing,
        "sbatch": _sbatch_facts(),
        "certified": _certified(),
        "env": ccr.envelope(layout, sizing),
        "submit": submit,
    }


# --------------------------------------------------------------------------- #
# shard_layout — the array width, from the SHIPPED partitioner
# --------------------------------------------------------------------------- #
def test_shard_layout_ceils_and_reports_the_real_shard_sizes():
    layout = ccr.shard_layout(_specs(5), shard_size=2)
    # 5 candidates at 2/shard = 3 shards; the shipped partitioner balances them 2/2/1, so the
    # LARGEST real shard is 2 — the number the worst case must be computed on.
    assert layout["n_shards"] == 3
    assert layout["array_spec"] == "0-2"
    assert layout["shard_sizes"] == {"min": 1, "max": 2, "total": 5}
    assert [len(s) for s in layout["shards"]] == [2, 2, 1]


def test_shard_layout_partitions_without_loss_or_duplication():
    specs = _specs(7)
    layout = ccr.shard_layout(specs, shard_size=3)
    flat = [s.candidate_id for shard in layout["shards"] for s in shard]
    assert sorted(flat) == sorted(s.candidate_id for s in specs)
    assert len(set(flat)) == len(specs)


def test_shard_layout_max_can_be_below_the_requested_shard_size():
    """4 candidates at 3/shard is 2 shards of 2 — the requested size is never reached.

    The distinction is load-bearing: the worst-case wall is ``max_shard × per-candidate``,
    so sizing on the *requested* 3 would over-quote by 50 % here and, on a manifest that
    packs the other way, under-quote.
    """
    layout = ccr.shard_layout(_specs(4), shard_size=3)
    assert layout["requested_shard_size"] == 3
    assert layout["shard_sizes"]["max"] == 2
    assert layout["n_shards"] == 2


def test_shard_layout_refuses_an_empty_manifest_and_a_zero_shard_size():
    with pytest.raises(ccr.RunPlanError):
        ccr.shard_layout([], shard_size=20)
    with pytest.raises(ccr.RunPlanError):
        ccr.shard_layout(_specs(2), shard_size=0)


# --------------------------------------------------------------------------- #
# envelope — bounded on the LARGEST real shard, not on the requested size
# --------------------------------------------------------------------------- #
def test_envelope_bounds_the_task_on_the_largest_real_shard():
    layout = ccr.shard_layout(_specs(5), shard_size=2)
    env = ccr.envelope(layout, _sizing())
    # max_shard 2 x 7.5 s = 15 s = 0.0042 h/task; 3 tasks x 2 cpus.
    assert env["worst_case_wall_h_per_task"] == round(2 * 7.5 / 3600.0, 4)
    assert env["worst_case_core_h"] == round(2 * 7.5 / 3600.0 * 3 * 2, 2)
    assert env["expected_wall_h_per_task"] == round(2 * 1.25 / 3600.0, 4)
    assert env["gpu_h"] == 0.0


def test_envelope_uses_the_real_max_shard_not_the_requested_size():
    """Same fixture as above: 4 at 3/shard packs 2+2, so the bound is 2 candidates, not 3."""
    env = ccr.envelope(ccr.shard_layout(_specs(4), shard_size=3), _sizing())
    assert env["worst_case_wall_h_per_task"] == round(2 * 7.5 / 3600.0, 4)
    assert env["worst_case_wall_h_per_task"] != round(3 * 7.5 / 3600.0, 4)


def test_envelope_carries_the_sizing_reports_own_caveat_verbatim():
    env = ccr.envelope(ccr.shard_layout(_specs(4), shard_size=2), _sizing())
    assert env["expected_wall_caveat"] == _sizing()["expected_wall_caveat"]


def test_envelope_scales_with_the_number_of_tasks():
    # 1800 s/candidate x 2 per shard = exactly 1.0 h/task, so the scaling is checked on
    # whole numbers rather than on values the 2-dp rounding would blur.
    sizing = {**_sizing(), "worst_case_per_candidate_s": 1800.0}
    small = ccr.envelope(ccr.shard_layout(_specs(4), shard_size=2), sizing)
    big = ccr.envelope(ccr.shard_layout(_specs(8), shard_size=2), sizing)
    assert (small["worst_case_wall_h_per_task"], small["worst_case_core_h"]) == (1.0, 4.0)
    assert (big["worst_case_wall_h_per_task"], big["worst_case_core_h"]) == (1.0, 8.0)


# --------------------------------------------------------------------------- #
# min_cov_reach — the same fraction is a different absolute demand per population
# --------------------------------------------------------------------------- #
def test_min_cov_reach_reports_nucleotides_at_each_median():
    reach = ccr.min_cov_reach(
        {"query_length_nt": {"p50": 104.0}, "fp_length_nt": {"p50": 110.0}}, 0.5
    )
    assert reach["required_covered_nt_at_query_median"] == 52.0
    assert reach["required_covered_nt_at_fp_median"] == 55.0


def test_min_cov_reach_tracks_the_cutoff_not_a_constant():
    lengths = {"query_length_nt": {"p50": 104.0}, "fp_length_nt": {"p50": 110.0}}
    assert ccr.min_cov_reach(lengths, 0.25)["required_covered_nt_at_query_median"] == 26.0


# --------------------------------------------------------------------------- #
# submit_plan — the token list the ack authorizes
# --------------------------------------------------------------------------- #
def _submit(tmp_path: Path, round_dir="$HOME/tbox-scratch/round_p3_15g_control"):
    specs = _specs(5)
    layout = ccr.shard_layout(specs, shard_size=2)
    return ccr.submit_plan(
        layout,
        _sizing(),
        manifest="data/m.json",
        query_fasta="data/q.fasta",
        query_fasta_sha256="0" * 64,
        sbatch="slurm/p2/mine_round_producer.sbatch",
        round_dir=round_dir,
    )


def test_submit_plan_array_spec_and_shard_count_agree_with_the_layout(tmp_path: Path):
    plan = _submit(tmp_path)
    assert "--array=0-2" in plan["sbatch_argv"]
    assert "--n-shards" in plan["make_shards_argv"]
    assert plan["make_shards_argv"][plan["make_shards_argv"].index("--n-shards") + 1] == "3"
    assert plan["verification"]["expect_shard_ok_markers"] == 3


def test_submit_plan_exports_every_variable_the_sbatch_requires(tmp_path: Path):
    export = next(t for t in _submit(tmp_path)["sbatch_argv"] if t.startswith("--export="))
    for key in ("SHARD_DIR=", "ROUND_DIR=", "ALIGN_TIMEOUT_S=", "QUERY_FASTA="):
        assert key in export, f"{key} missing — the sbatch's `:?` guard would kill the task"
    assert export.startswith("--export=ALL,")
    # `600.0` would reach the sbatch's shape guard as a valid number, but the round names the
    # bound as an integer everywhere else; `:g` keeps one spelling.
    assert "ALIGN_TIMEOUT_S=600," in export or export.endswith("ALIGN_TIMEOUT_S=600")


def test_submit_plan_stages_the_query_fasta_out_of_the_checkout(tmp_path: Path):
    """MEASURED at the §9.3 preflight: the cluster checkout holds a POINTER, not sequences.

    `*.fasta` is git-LFS-tracked and the cluster has **no git-lfs binary**, so the file the
    array would search is ~130 bytes of metadata that passes every non-empty check there is.
    The sequences are staged outside the checkout — where the next `git reset --hard` cannot
    revert them — and the copy is bound by digest, which is the one failure the sbatch's own
    guards cannot see.
    """
    plan = _submit(tmp_path)
    staging = plan["staging"]
    assert staging["staged_query_fasta"] == plan["round_dir"] + "/q.fasta"
    assert staging["query_fasta_sha256"] == "0" * 64
    assert "sha256sum" in staging["verify_command"]
    # ⚠ MEASURED 2026-08-11: an rsync remote path is not run through a remote shell, so a
    # `two:'$HOME/…'` destination sends the four literal characters `$HOME` and fails with
    # "change_dir … No such file or directory". rsync resolves a RELATIVE remote path against
    # the remote home, so that is the form the copy command must carry — while the export and
    # verify spellings, which do pass through a shell, keep `$HOME`.
    assert "two:tbox-scratch/" in staging["copy_command"]
    assert "$HOME" not in staging["copy_command"].split("two:", 1)[1]
    assert staging["staged_query_fasta"].startswith("$HOME/")
    assert "$HOME" in staging["verify_command"]
    assert ccr.exported_value(plan, "QUERY_FASTA") == staging["staged_query_fasta"]


@pytest.mark.parametrize(
    "given,expected",
    [
        ("$HOME/tbox-scratch/x", "tbox-scratch/x"),
        ("/abs/tbox-scratch/x", "/abs/tbox-scratch/x"),  # absolute: the caller's to get right
        ("rel/$HOME/x", "rel/$HOME/x"),  # only a LEADING $HOME/ is a home reference
    ],
)
def test_home_relative_only_strips_a_leading_home(given: str, expected: str):
    assert ccr._home_relative(given) == expected


def test_submit_plan_leaves_home_unexpanded(tmp_path: Path):
    plan = _submit(tmp_path)
    assert plan["round_dir"].startswith("$HOME/")
    assert str(Path.home()) not in json.dumps(plan), "a local home directory leaked into the plan"


def test_submit_plan_shard_dir_is_under_the_round_dir(tmp_path: Path):
    plan = _submit(tmp_path)
    assert plan["shard_dir"] == plan["round_dir"] + "/shards"


# --------------------------------------------------------------------------- #
# read_sbatch — parsed from the shipped bytes, not restated
# --------------------------------------------------------------------------- #
def test_read_sbatch_parses_the_real_producer():
    facts = ccr.read_sbatch(COMMITTED_SBATCH)
    assert facts["wall_limit_h"] == 12.0
    assert facts["cutoffs"]["engine"] == "nhmmer"
    assert facts["cutoffs"]["min_cov"] == 0.5
    assert facts["has_query_fasta_guard"] and facts["passes_query_fasta_argument"]


def test_read_sbatch_refuses_a_file_with_no_time_header(tmp_path: Path):
    path = tmp_path / "no_time.sbatch"
    path.write_text('#SBATCH --partition=gpu\nCTRL_ENGINE="nhmmer"\n', encoding="utf-8")
    with pytest.raises(ccr.RunPlanError):
        ccr.read_sbatch(path)


def test_read_sbatch_refuses_a_file_whose_cutoffs_moved(tmp_path: Path):
    path = tmp_path / "no_cutoffs.sbatch"
    path.write_text("#SBATCH --time=12:00:00\n", encoding="utf-8")
    with pytest.raises(ccr.RunPlanError):
        ccr.read_sbatch(path)


@pytest.mark.parametrize("dropped", ccr.SBATCH_CUTOFF_KEYS)
def test_read_sbatch_refuses_when_any_single_cutoff_moves(tmp_path: Path, dropped: str):
    """Each of the five, alone — a guard naming a subset lets the rest escape as a KeyError.

    Parametrized over the shipped tuple rather than a retyped list, so a sixth cutoff is
    covered the moment it is required.
    """
    kept = [f'{k}="1"' for k in ccr.SBATCH_CUTOFF_KEYS if k != dropped]
    path = tmp_path / "partial.sbatch"
    path.write_text("#SBATCH --time=12:00:00\n" + "; ".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(ccr.RunPlanError, match=dropped):
        ccr.read_sbatch(path)


def test_read_sbatch_sees_a_missing_sequence_route(tmp_path: Path):
    """An sbatch that predates the seam must report both halves absent, not be assumed ready."""
    path = tmp_path / "old.sbatch"
    path.write_text(
        "#SBATCH --time=12:00:00\n"
        'CTRL_ENGINE="nhmmer"; CTRL_EVALUE="100"; CTRL_MAX_TARGET_SEQS="500"; '
        'CTRL_MIN_PIDENT="40"; CTRL_MIN_COV="0.5"\n'
        "python -m tbox_finder.mining.covariation_producer search-shard --shard x\n",
        encoding="utf-8",
    )
    facts = ccr.read_sbatch(path)
    assert facts["has_query_fasta_guard"] is False
    assert facts["passes_query_fasta_argument"] is False


def test_read_sbatch_reads_the_wall_limit_it_is_given(tmp_path: Path):
    path = tmp_path / "short.sbatch"
    path.write_text(
        "#SBATCH --time=01:30:00\n"
        'CTRL_ENGINE="nhmmer"; CTRL_EVALUE="100"; CTRL_MAX_TARGET_SEQS="500"; '
        'CTRL_MIN_PIDENT="40"; CTRL_MIN_COV="0.5"\n',
        encoding="utf-8",
    )
    assert ccr.read_sbatch(path)["wall_limit_h"] == 1.5


# --------------------------------------------------------------------------- #
# read_sizing / read_query_lengths — refuse a reshaped input rather than KeyError
# --------------------------------------------------------------------------- #
def test_read_sizing_refuses_a_report_missing_the_envelope(tmp_path: Path):
    path = tmp_path / "sizing.json"
    path.write_text(json.dumps({"envelope": {"align_timeout_s": 600}}), encoding="utf-8")
    with pytest.raises(ccr.RunPlanError):
        ccr.read_sizing(path)


def test_read_sizing_reads_the_committed_report():
    sizing = ccr.read_sizing(REPO_ROOT / ccr.DEFAULT_SIZING_REPORT)
    assert sizing["align_timeout_s"] == 600.0
    assert sizing["cpus_per_task"] == 2
    assert sizing["worst_case_per_candidate_s"] > 0


def test_read_query_lengths_reads_both_populations():
    lengths = ccr.read_query_lengths(REPO_ROOT / ccr.DEFAULT_DETECT_REPORT)
    assert lengths["query_length_nt"]["p50"] > 0 and lengths["fp_length_nt"]["p50"] > 0
    assert 0 < lengths["n_records_with_a_query"] <= lengths["n_records_scanned"]


# --------------------------------------------------------------------------- #
# plan_clauses — one broken clause at a time
# --------------------------------------------------------------------------- #
def _submit_with_export(submit: dict, name: str, value: str) -> dict:
    """The same submit plan with ONE export token rewritten — the run-shape mutation."""
    argv = []
    for token in submit["sbatch_argv"]:
        if token.startswith("--export="):
            items = [
                f"{name}={value}" if item.split("=")[0] == name else item
                for item in token[len("--export=") :].split(",")
            ]
            token = "--export=" + ",".join(items)
        argv.append(token)
    return {**submit, "sbatch_argv": argv}


def test_plan_clauses_pass_on_a_valid_supply(tmp_path: Path):
    specs, kwargs = _good_clause_kwargs(tmp_path)
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["all_pass"] is True
    assert set(result["clauses"]) == set(ccr.REQUIRED_PLAN_CLAUSES)
    assert result["measured"]["n_with_a_query"] == len(specs)


def test_a_missing_query_is_caught_by_the_supply_clause_and_by_the_validator(tmp_path: Path):
    specs = _specs()
    fasta = _fasta(tmp_path, specs, drop=specs[2].candidate_id, name="dropped.fa")
    _, kwargs = _good_clause_kwargs(tmp_path)
    kwargs["query_fasta"] = fasta
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["clauses"]["every_manifest_candidate_has_a_query"] is False
    assert result["clauses"]["shipped_validator_accepts_every_shard"] is False
    assert result["all_pass"] is False
    assert result["measured"]["n_with_a_query"] == len(specs) - 1


def test_a_length_disagreement_leaves_the_presence_clause_true(tmp_path: Path):
    specs = _specs()
    fasta = _fasta(tmp_path, specs, truncate=specs[1].candidate_id, name="truncated.fa")
    _, kwargs = _good_clause_kwargs(tmp_path)
    kwargs["query_fasta"] = fasta
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["clauses"]["every_manifest_candidate_has_a_query"] is True
    assert result["clauses"]["every_query_length_matches_its_span"] is False
    assert result["clauses"]["shipped_validator_accepts_every_shard"] is False
    assert "different locus" in (result["measured"]["validator_error"] or "")


@pytest.mark.parametrize(
    "round_dir",
    [
        ccr.FP_SUPPLY_ROUND_DIR,
        ccr.FP_SUPPLY_ROUND_DIR + "/msa",
        "$HOME/tbox-scratch",  # a PARENT of the FP supply promotes into it too
    ],
)
def test_a_round_dir_that_touches_the_fp_supply_is_refused(tmp_path: Path, round_dir: str):
    specs, kwargs = _good_clause_kwargs(tmp_path, round_dir=round_dir)
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["clauses"]["round_dir_is_not_the_fp_supply"] is False
    assert result["all_pass"] is False


def test_a_sibling_round_dir_is_allowed(tmp_path: Path):
    """The refusal must be about CONTAINMENT, not about sharing a prefix string."""
    specs, kwargs = _good_clause_kwargs(tmp_path, round_dir=ccr.FP_SUPPLY_ROUND_DIR + "_control")
    assert ccr.plan_clauses(specs, **kwargs)["clauses"]["round_dir_is_not_the_fp_supply"] is True


def test_an_sbatch_without_the_sequence_route_is_refused(tmp_path: Path):
    specs, kwargs = _good_clause_kwargs(tmp_path)
    kwargs["sbatch"] = {**_sbatch_facts(), "passes_query_fasta_argument": False}
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["clauses"]["sbatch_carries_the_sequence_route"] is False
    kwargs["sbatch"] = {**_sbatch_facts(), "has_query_fasta_guard": False}
    assert (
        ccr.plan_clauses(specs, **kwargs)["clauses"]["sbatch_carries_the_sequence_route"] is False
    )


def test_drifted_cutoffs_are_refused_against_the_certified_config(tmp_path: Path):
    specs, kwargs = _good_clause_kwargs(tmp_path)
    kwargs["certified"] = {**_certified(), "min_cov": 0.75}
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["clauses"]["sbatch_cutoffs_match_the_certified_config"] is False


def test_a_task_that_cannot_finish_inside_the_wall_is_refused(tmp_path: Path):
    specs, kwargs = _good_clause_kwargs(tmp_path)
    kwargs["sbatch"] = {**_sbatch_facts(), "wall_limit_h": 0.001}
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["clauses"]["worst_case_task_fits_the_wall_limit"] is False


@pytest.mark.parametrize("exported", ["300", "600s", ""])
def test_an_exported_align_timeout_that_is_not_the_sized_one_is_refused(
    tmp_path: Path, exported: str
):
    """The clause reads the `--export` token, not the argument that built it.

    Comparing `sizing["align_timeout_s"]` against itself would be true by construction on
    every real run — vacuous exactly where it is meant to bite (CodeRabbit CLI r2).
    """
    specs, kwargs = _good_clause_kwargs(tmp_path)
    kwargs["submit"] = _submit_with_export(kwargs["submit"], "ALIGN_TIMEOUT_S", exported)
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["clauses"]["align_timeout_matches_the_sized_bound"] is False
    assert result["all_pass"] is False


def test_exporting_the_repo_query_fasta_instead_of_the_staged_copy_is_refused(tmp_path: Path):
    """On the cluster the repo path is a git-LFS pointer; only the staged copy has sequences."""
    specs, kwargs = _good_clause_kwargs(tmp_path)
    kwargs["submit"] = _submit_with_export(
        kwargs["submit"], "QUERY_FASTA", "data/processed/mining/curated_control_queries_v0.fasta"
    )
    result = ccr.plan_clauses(specs, **kwargs)
    assert result["clauses"]["query_fasta_export_is_the_staged_copy"] is False
    assert result["all_pass"] is False


def test_exported_value_reads_the_token_the_run_receives(tmp_path: Path):
    submit = _submit(tmp_path)
    assert ccr.exported_value(submit, "ALIGN_TIMEOUT_S") == "600"
    assert ccr.exported_value(submit, "ROUND_DIR") == submit["round_dir"]
    assert ccr.exported_value(submit, "NOT_EXPORTED") is None


def test_an_empty_supply_cannot_pass_vacuously(tmp_path: Path):
    """Every "for all" clause is trivially true over nothing — the guards are what stop it."""
    specs, kwargs = _good_clause_kwargs(tmp_path)
    empty = tmp_path / "empty.fa"
    empty.write_text("", encoding="utf-8")
    kwargs["query_fasta"] = empty
    with pytest.raises(ValueError):  # ProducerError: an empty query supply searches nothing
        ccr.plan_clauses(specs, **kwargs)
    # And with no candidates at all the layout itself refuses, before any clause is computed.
    with pytest.raises(ccr.RunPlanError):
        ccr.shard_layout([], shard_size=2)


def test_clauses_do_not_pass_vacuously_over_an_empty_spec_list(tmp_path: Path):
    """Every "for all" clause is TRUE over zero candidates; only the guards stop it.

    ``shard_layout`` refuses an empty manifest before this can happen in the CLI, so the
    guards are reachable only from a direct call — which is exactly why they are tested from
    one ([[clauses-must-guard-emptiness]]).
    """
    _, kwargs = _good_clause_kwargs(tmp_path)
    kwargs["layout"] = {"n_shards": 1, "array_spec": "0-0", "shards": [[]]}
    result = ccr.plan_clauses([], **kwargs)
    assert result["clauses"]["manifest_nonempty"] is False
    assert result["clauses"]["every_manifest_candidate_has_a_query"] is False
    assert result["clauses"]["every_query_length_matches_its_span"] is False
    assert result["all_pass"] is False


def test_the_clause_set_is_enumerated_not_whatever_was_computed(tmp_path: Path):
    """A clause added to the required tuple but never computed must fail loudly.

    Without this, `all(...)` over a dict comprehension would simply skip it — the failure
    mode [[new-gate-clause-invalidates-old-reports]] exists to prevent.
    """
    specs, kwargs = _good_clause_kwargs(tmp_path)
    original = ccr.REQUIRED_PLAN_CLAUSES
    try:
        ccr.REQUIRED_PLAN_CLAUSES = original + ("a_clause_nobody_computes",)
        with pytest.raises(ccr.RunPlanError):
            ccr.plan_clauses(specs, **kwargs)
    finally:
        ccr.REQUIRED_PLAN_CLAUSES = original


# --------------------------------------------------------------------------- #
# plan() end to end + the committed report
# --------------------------------------------------------------------------- #
@pytest.fixture
def in_repo_root(monkeypatch: pytest.MonkeyPatch):
    """`portable_path` publishes paths relative to the CWD, so pin it to the repo root.

    Without this the same call emits repo-relative paths under one working directory and
    bare basenames under another, and the comparison against the committed report would be
    a comparison against wherever pytest happened to be invoked from.
    """
    monkeypatch.chdir(REPO_ROOT)


def _plan_from_repo(**overrides):
    kwargs = {
        "manifest": COMMITTED_MANIFEST,
        "query_fasta": COMMITTED_FASTA,
        "sizing_report": REPO_ROOT / ccr.DEFAULT_SIZING_REPORT,
        "detect_report": REPO_ROOT / ccr.DEFAULT_DETECT_REPORT,
        "sbatch": COMMITTED_SBATCH,
        "certified_report": REPO_ROOT / ccr.CERTIFIED_SEARCH_REPORT,
    }
    kwargs.update(overrides)
    return ccr.plan(**kwargs)


def test_plan_over_the_real_supply_reaches_the_committed_numbers(in_repo_root):
    body = _plan_from_repo()
    committed = json.loads(COMMITTED_PLAN.read_text(encoding="utf-8"))
    assert body["array"] == committed["array"]
    assert body["envelope"] == committed["envelope"]
    assert body["submit"] == committed["submit"]
    assert body["preflight"]["all_pass"] is True


def test_plan_refuses_when_a_clause_fails(in_repo_root):
    with pytest.raises(ccr.RunPlanError, match="round_dir_is_not_the_fp_supply"):
        _plan_from_repo(round_dir=ccr.FP_SUPPLY_ROUND_DIR)


def test_the_staging_digest_agrees_with_the_provenance_entry(in_repo_root):
    """One file, two independent hashes — the operator verifies the cluster copy against one
    of them and the provenance records the other, so they must not be free to disagree."""
    body = _plan_from_repo()
    recorded = body["provenance"]["inputs"][ccr.DEFAULT_QUERY_FASTA]
    assert body["submit"]["staging"]["query_fasta_sha256"] == recorded


def test_a_staging_digest_that_disagrees_with_provenance_is_refused(in_repo_root, monkeypatch):
    """The guard, exercised: without it the plan would name bytes nobody hashed."""
    monkeypatch.setattr(ccr, "_sha256_of", lambda _path: "f" * 64)
    with pytest.raises(ccr.RunPlanError, match="disagree"):
        _plan_from_repo()


def test_plan_publishes_no_absolute_path(in_repo_root):
    """The repo is public; a plan is a committed artifact ([[verify-the-line-you-ship]])."""
    payload = json.dumps(_plan_from_repo())
    assert str(Path.home()) not in payload
    for value in json.loads(payload).get("sized_from", {}).values():
        assert not str(value).startswith("/")


def test_the_committed_plan_carries_provenance_and_pins_nothing():
    committed = json.loads(COMMITTED_PLAN.read_text(encoding="utf-8"))
    assert committed["pins_nothing"] is True
    assert committed["step"] == ccr.STEP
    prov = committed["provenance"]
    assert prov["git_sha"] and prov["env_lock_hash"]
    assert prov["inputs"], "the plan must record what it was derived from"


# --------------------------------------------------------------------------- #
# The sbatch's own QUERY_FASTA guard, executed as its own bytes
# --------------------------------------------------------------------------- #
def _extract_query_fasta_guard(sbatch: Path = COMMITTED_SBATCH) -> str:
    """The `QUERY_FASTA` declaration + its guard, lifted verbatim from the sbatch.

    Run as the shipped bytes rather than retyped: a retyped equivalent proves only that
    this test agrees with itself ([[verify-the-line-you-ship]]).
    """
    lines = sbatch.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("QUERY_FASTA="))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "fi")
    return "\n".join(lines[start : end + 1])


def _run_guard(env_value: str | None):
    import os
    import subprocess

    script = _extract_query_fasta_guard() + '\nprintf "%s\\n" "${SEARCH_QUERY_ARGS[@]}"\n'
    env = {"PATH": os.environ["PATH"]}
    if env_value is not None:
        env["QUERY_FASTA"] = env_value
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )


def test_the_sbatch_guard_is_inert_when_query_fasta_is_unset():
    """The default is the whole point: the FP coordinate route must be untouched."""
    for value in (None, ""):
        proc = _run_guard(value)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "", "an empty QUERY_FASTA must add no argument"


def test_the_sbatch_guard_passes_the_argument_when_the_file_is_real(tmp_path: Path):
    fasta = tmp_path / "q.fa"
    fasta.write_text(">a\nACGT\n", encoding="utf-8")
    proc = _run_guard(str(fasta))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["--query-fasta", str(fasta)]


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_the_sbatch_guard_refuses_a_query_fasta_that_cannot_be_searched(tmp_path: Path, kind: str):
    """`-s`, not `-e`: an empty file is argparse-valid and would search nothing, and a shard
    that searched nothing resolves every candidate `unavailable` — which reads exactly like an
    honest not-producible round."""
    path = tmp_path / "q.fa"
    if kind == "empty":
        path.write_text("", encoding="utf-8")
    proc = _run_guard(str(path))
    assert proc.returncode == 2
    assert "QUERY_FASTA" in proc.stderr


def test_the_sbatch_guard_refuses_a_git_lfs_pointer(tmp_path: Path):
    """The failure the §9.3 preflight actually found, refused by name.

    A pointer is ~130 non-empty bytes: `-s` passes it, the array activates three conda envs
    and reaches the producer, and 16 tasks then die on a message about FASTA records.
    """
    pointer = tmp_path / "q.fa"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:08a3d160e3\nsize 61373\n",
        encoding="utf-8",
    )
    proc = _run_guard(str(pointer))
    assert proc.returncode == 2
    assert "git lfs pull" in proc.stderr and "POINTER" in proc.stderr


# --------------------------------------------------------------------------- #
# The real supply satisfies the seam the run depends on
# --------------------------------------------------------------------------- #
def test_every_committed_query_resolves_through_the_shipped_validator():
    specs = read_candidate_manifest(COMMITTED_MANIFEST)
    resolved = resolve_shard_queries(specs, COMMITTED_FASTA)
    assert len(resolved) == len(specs) == 317
    assert all(len(resolved[s.candidate_id]) == s.locus_end - s.locus_start for s in specs)
