"""P2-12 — the ablation reducer, the report's backbone block, and the leg table's shape.

Bare-CI tier (torch-free). Legs are built by **mutating the real committed P2-06 point**
``reports/p2/sweep/g0.5_lr3e-4_a0.5.json`` rather than by hand-rolling a report dict: that
point is the actual baseline row, it already satisfies every ``select_best.point_problems``
floor, and a hand-built stand-in would silently drift from the schema the reducer must consume
(§8.7 — real fixtures, never a mocked pipeline).

The reducer's job is to *not* over-read a 4-leg table. So the tests here lock, in order: that
it refuses a leg it cannot place on an axis; that it ranks on the interval; that
``separated_from_baseline`` is TRUE only on a **measured** non-overlap and ``None`` when it
could not be measured; and that a table of zero rows cannot pass for a complete one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tbox_finder.eval import backbone_throughput as T
from tbox_finder.models import backbone_registry as R
from tbox_finder.train import ablation_table as A
from tbox_finder.train.train_stage1 import (
    BLOCK_FIELD_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Stage1TrainConfig,
    _backbone_block_problems,
    _schema_precedes,
    diagnostics,
)

REPO = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO / "reports" / "p2" / "sweep" / "g0.5_lr3e-4_a0.5.json"
SBATCH = REPO / "slurm" / "p2" / "ablate_stage1.sbatch"


def _baseline() -> dict:
    report = json.loads(BASELINE_PATH.read_text())
    report["report_path"] = str(BASELINE_PATH)
    return report


def _leg(*, key: str, backbone: str, rc: str, crf: bool, lower: float, upper: float) -> dict:
    """A synthetic P2-12 leg: the real baseline report with the axes and CI moved."""
    leg = copy.deepcopy(_baseline())
    leg["report_path"] = f"reports/p2/ablation/{key}.json"
    leg["schema_version"] = SCHEMA_VERSION
    leg["diagnostics"]["config"]["backbone"] = backbone
    leg["diagnostics"]["config"]["rc_combine"] = rc
    leg["diagnostics"]["config"]["use_crf"] = crf
    spec = R.resolve_checkpoint(backbone)
    leg["backbone"] = {
        **R.checkpoint_summary(spec),
        "measured_param_count": spec.expected_param_count,
        "measured_d_model": spec.d_model,
    }
    point = (lower + upper) / 2.0
    leg["eval_metrics"]["gate4_core_min_f1"]["min_f1"] = point
    leg["eval_metrics"]["block_bootstrap_ci"] = {
        **leg["eval_metrics"]["block_bootstrap_ci"],
        "lower": lower,
        "point": point,
        "upper": upper,
    }
    return leg


PRODUCTION = R.PRODUCTION_CHECKPOINT


def _four_legs() -> list[dict]:
    """Four legs whose CIs all OVERLAP the real committed baseline's [0.9125, 0.9550].

    Chosen to be realistic, not convenient: with 4 legs graded on an 830-record / 469-block
    fold whose baseline CI is already ~0.043 wide, mutually overlapping intervals are the
    expected outcome. A fixture with cleanly separated arms would test a table shape this
    step has no reason to expect and would hide the "no separation" path entirely.
    """
    return [
        _leg(key="rc_gate", backbone=PRODUCTION, rc="gate", crf=False, lower=0.90, upper=0.95),
        _leg(key="head_crf", backbone=PRODUCTION, rc="concat", crf=True, lower=0.89, upper=0.94),
        _leg(
            key="size_d256_l4",
            backbone="caduceus-ps-1k-d256-l4",
            rc="concat",
            crf=False,
            lower=0.88,
            upper=0.93,
        ),
        _leg(
            key="size_d118_l4",
            backbone="caduceus-ps-1k-d118-l4",
            rc="concat",
            crf=False,
            lower=0.87,
            upper=0.92,
        ),
    ]


# ────────────────────────────────────────────────────────────────────────────────────────
# The committed baseline row
# ────────────────────────────────────────────────────────────────────────────────────────
def test_the_committed_baseline_point_is_the_baseline_arm():
    """The row that costs 0 new GPU-h must actually BE the (i)/(iii) baseline arm.

    If the promoted P2-06 point had been trained with ``rc_combine=gate``, reusing it as the
    ``concat`` baseline would silently compare gate against gate. Re-derived from the point's
    own recorded config, not assumed from its filename.
    """
    cfg = _baseline()["diagnostics"]["config"]
    assert cfg["rc_combine"] == "concat"
    assert cfg["use_crf"] is False
    assert (cfg["gamma"], cfg["lr"], cfg["class_weight_alpha"]) == (0.5, 0.0003, 0.5)
    assert cfg["epochs"] == 10 and cfg["batch_size"] == 8
    assert not A.row_problems(_baseline()), A.row_problems(_baseline())


def test_the_pre_p2_12_baseline_backbone_axis_is_re_derived_not_assumed():
    """The committed point predates the ``backbone`` config field, so its axis must come from
    its own recorded ``backbone.repo_id`` + ``revision`` — and only when BOTH match a pin."""
    baseline = _baseline()
    assert "backbone" not in baseline["diagnostics"]["config"]
    axes = A.axes_of(baseline)
    assert axes["backbone"] == PRODUCTION
    assert "re-derived" in axes["backbone_source"]

    # A repo id alone is a repository, not a checkpoint: a mismatched revision must NOT
    # resolve, and the row must then be refused rather than silently placed on production.
    drifted = copy.deepcopy(baseline)
    drifted["backbone"]["revision"] = "0" * 40
    assert A.axes_of(drifted)["backbone"] is None
    assert any("backbone" in p for p in A.row_problems(drifted))


# ────────────────────────────────────────────────────────────────────────────────────────
# Ranking, separation, and refusal
# ────────────────────────────────────────────────────────────────────────────────────────
def test_the_table_ranks_on_the_lower_ci_bound_not_the_point():
    """P2-06's own lesson: its winner's CI contained all four runners-up and the margin over
    second was a fortieth of the CI width. A table sorted on the point invites reading that
    as a finding, so this one sorts on the conservative bound."""
    table = A.build_table([_baseline(), *_four_legs()], baseline_leg=str(BASELINE_PATH))
    assert table["n_rejected"] == 0, table["rejected"]
    assert table["n_rows"] == 5
    lowers = [row["ci"]["lower"] for row in table["rows"]]
    assert lowers == sorted(lowers, reverse=True), lowers
    assert "block_bootstrap_ci.lower" in table["ranked_on"]
    # No winner is named — the shipped-config choice is P2-14's.
    assert "winner" not in table


def test_separation_is_true_only_on_a_measured_non_overlap():
    """Three outcomes, never two. An unmeasurable interval must not render as "separated",
    and it must not render as "overlapping" either ([[control-matchedness-must-be-asserted]]:
    "no power" reading identically to "no signal" is the failure mode)."""
    baseline = _baseline()
    base_ci = baseline["eval_metrics"]["block_bootstrap_ci"]
    overlapping = _leg(
        key="rc_gate",
        backbone=PRODUCTION,
        rc="gate",
        crf=False,
        lower=base_ci["lower"] + 0.001,
        upper=base_ci["upper"] + 0.001,
    )
    separated = _leg(
        key="head_crf",
        backbone=PRODUCTION,
        rc="concat",
        crf=True,
        lower=base_ci["upper"] + 0.005,
        upper=base_ci["upper"] + 0.02,
    )
    table = A.build_table([baseline, overlapping, separated], baseline_leg=str(BASELINE_PATH))
    by_leg = {r["leg"]: r for r in table["rows"]}
    assert by_leg[overlapping["report_path"]]["separated_from_baseline"] is False
    assert by_leg[separated["report_path"]]["separated_from_baseline"] is True
    # The baseline's own row is neither separated from nor overlapping itself.
    assert by_leg[str(BASELINE_PATH)]["separated_from_baseline"] is None
    assert table["n_separated_from_baseline"] == 1
    assert table["any_axis_separated_from_baseline"] is True

    assert A._overlaps(None, base_ci) is None
    assert A._overlaps({"lower": None, "upper": 1.0}, base_ci) is None


def test_the_expected_outcome_of_four_legs_is_no_separation_and_it_is_said_plainly():
    """With 4 legs on an 830-record fold the honest answer is "the axes do not separate", and
    the artifact must state it as a re-derived fact rather than leave it to a reader."""
    table = A.build_table([_baseline(), *_four_legs()], baseline_leg=str(BASELINE_PATH))
    assert table["n_separated_from_baseline"] == 0
    assert table["any_axis_separated_from_baseline"] is False


def test_a_leg_without_a_placeable_axis_is_refused():
    """A row that cannot be placed on the axis it is supposed to occupy is not a row."""
    orphan = _leg(key="rc_gate", backbone=PRODUCTION, rc="gate", crf=False, lower=0.9, upper=0.95)
    del orphan["diagnostics"]["config"]["rc_combine"]
    table = A.build_table([_baseline(), orphan], baseline_leg=str(BASELINE_PATH))
    assert table["n_rejected"] == 1
    assert any("rc_combine" in p for p in table["rejected"][0]["problems"])


def test_a_leg_without_a_ci_is_refused_because_the_table_ranks_on_the_interval():
    legless = _leg(
        key="head_crf", backbone=PRODUCTION, rc="concat", crf=True, lower=0.9, upper=0.95
    )
    legless["eval_metrics"]["block_bootstrap_ci"] = {"n_blocks": 469}
    problems = A.row_problems(legless)
    assert any("block_bootstrap_ci" in p for p in problems), problems


def test_the_reused_baseline_reports_its_provenance_spread_rather_than_hiding_it():
    """The baseline was trained at an EARLIER commit. That is a real difference between the
    rows; the artifact re-derives and reports it (disclosed, not gated) instead of asserting
    the legs are identical apart from the axis."""
    legs = _four_legs()
    for leg in legs:
        leg["provenance"] = {**leg["provenance"], "git_sha": "f" * 40}
    table = A.build_table([_baseline(), *legs], baseline_leg=str(BASELINE_PATH))
    consistency = table["cross_point_consistency"]
    assert consistency["per_field"]["git_sha"]["agrees"] is False
    assert len(consistency["per_field"]["git_sha"]["distinct"]) == 2
    # Everything that MUST agree still does — same env, same seed, same fold.
    for field in ("env_lock", "seed", "n_eval_records", "n_eval_blocks"):
        assert consistency["per_field"][field]["agrees"] is True, field


# ────────────────────────────────────────────────────────────────────────────────────────
# The artifact validator
# ────────────────────────────────────────────────────────────────────────────────────────
def test_a_valid_table_validates_and_carries_every_disclosure():
    table = A.build_table([_baseline(), *_four_legs()], baseline_leg=str(BASELINE_PATH))
    assert A.artifact_problems(table, expect_rows=5) == []
    for disclosure in A.DISCLOSURES:
        assert disclosure in table["disclosures"]
    # The four §10.3 disclosures imp.md pins by name must each have a referent.
    joined = " ".join(table["disclosures"]).lower()
    for phrase in ("positives-only", "viterbi", "size x context", "firmicutes", "micro"):
        assert phrase in joined, phrase


def test_an_empty_table_cannot_pass_for_a_complete_one():
    """[[clauses-must-guard-emptiness]] — a table of zero rows is structurally valid and reads
    exactly like a table of clean rows in any all()/any() summary."""
    empty = A.build_table([], baseline_leg=None)
    assert empty["n_rows"] == 0
    problems = A.artifact_problems(empty, expect_rows=5)
    assert any("rows: 0 != expected 5" in p for p in problems), problems
    assert any("baseline" in p for p in problems), problems


def test_a_partial_table_is_not_read_as_a_complete_one():
    orphan = _leg(key="rc_gate", backbone=PRODUCTION, rc="gate", crf=False, lower=0.9, upper=0.95)
    del orphan["diagnostics"]["config"]["use_crf"]
    table = A.build_table([_baseline(), orphan], baseline_leg=str(BASELINE_PATH))
    assert any("not comparable" in p for p in A.artifact_problems(table, expect_rows=2))


def test_a_forged_separation_flag_does_not_survive_re_derivation():
    """[[gate-clauses-need-re-derivation]] — a flag echoed by the builder is not evidence.
    ``all(clauses)`` catches a clause flipped FALSE but never one fabricated TRUE, so the
    validator recomputes the verdict from the recorded intervals."""
    table = A.build_table([_baseline(), *_four_legs()], baseline_leg=str(BASELINE_PATH))
    forged = copy.deepcopy(table)
    row = next(r for r in forged["rows"] if not r["is_baseline"])
    row["separated_from_baseline"] = True
    forged["n_separated_from_baseline"] = 1
    problems = A.artifact_problems(forged, expect_rows=5)
    assert any("does not re-derive" in p for p in problems), problems


def test_two_runs_of_one_leg_are_caught():
    """A duplicated axis tuple means the same leg entered the table twice — a re-submit whose
    older report was not cleaned. Two rows for one arm would double its weight in any read."""
    legs = _four_legs()
    dup = copy.deepcopy(legs[0])
    dup["report_path"] = "reports/p2/ablation/rc_gate_rerun.json"
    table = A.build_table([_baseline(), *legs, dup], baseline_leg=str(BASELINE_PATH))
    assert any("duplicate axis tuple" in p for p in A.artifact_problems(table, expect_rows=6))


# ────────────────────────────────────────────────────────────────────────────────────────
# The report's backbone block (train_stage1)
# ────────────────────────────────────────────────────────────────────────────────────────
def test_a_current_schema_report_must_carry_the_resolved_and_measured_backbone():
    leg = _leg(key="rc_gate", backbone=PRODUCTION, rc="gate", crf=False, lower=0.9, upper=0.95)
    assert _backbone_block_problems(leg) == []
    for field in ("key", "measured_param_count"):
        stripped = copy.deepcopy(leg)
        del stripped["backbone"][field]
        assert any(field in p for p in _backbone_block_problems(stripped)), field


def test_an_older_schema_report_is_excused_but_not_exempted():
    """[[new-gate-clause-invalidates-old-reports]] — the 36 committed schema-2 sweep points
    measured no parameter count, and back-filling one would forge a measurement (§10.3). They
    are excused the NEW fields and still checked on the ones they do carry."""
    baseline = _baseline()
    assert baseline["schema_version"] != SCHEMA_VERSION
    assert _backbone_block_problems(baseline) == []
    # The invariant is that the baseline predates EVERY block field it is excused — not that
    # those fields were introduced at whatever the current schema happens to be. The former
    # coupling held only while P2-12's ENCODE bump and these fields landed together; the
    # P2-12 RESULT bumped the schema again (for the rc_combine census) without adding a block
    # field, which would have falsified it. Assert the excuse itself.
    assert all(
        _schema_precedes(baseline["schema_version"], introduced)
        for introduced in BLOCK_FIELD_SCHEMA_VERSION.values()
    )

    drifted = copy.deepcopy(baseline)
    drifted["backbone"]["revision"] = "0" * 40
    # An old report cannot hide a failure either: its recorded pin is still unresolvable.
    assert A.axes_of(drifted)["backbone"] is None


def test_a_block_naming_one_checkpoint_and_counting_another_is_caught():
    """The exact silent-ablation shape this step exists to make impossible."""
    leg = _leg(
        key="size_d118_l4",
        backbone="caduceus-ps-1k-d118-l4",
        rc="concat",
        crf=False,
        lower=0.8,
        upper=0.87,
    )
    forged = copy.deepcopy(leg)
    forged["backbone"]["measured_param_count"] = 7_725_312  # the production count
    assert any("measured_param_count" in p for p in _backbone_block_problems(forged))

    swapped = copy.deepcopy(leg)
    swapped["backbone"]["repo_id"] = R.CHECKPOINTS[PRODUCTION].repo_id
    assert any("repo_id" in p for p in _backbone_block_problems(swapped))

    unknown = copy.deepcopy(leg)
    unknown["backbone"]["key"] = "caduceus-ps-1k-d118-l5"
    assert any("unknown backbone" in p for p in _backbone_block_problems(unknown))


def test_the_config_rejects_an_unpinned_backbone_at_construction():
    """Compose-time, on the login node — not after a backbone load on a GPU node.

    ``Stage1TrainConfig`` is where a typo has to die. The alternative is what P2-11 showed:
    an override that composes, is dropped, and lets the run train the pinned production model
    under the ablation's name with a green gate (§10.3).
    """
    assert Stage1TrainConfig().backbone == R.PRODUCTION_CHECKPOINT
    for key in R.CHECKPOINT_KEYS:
        assert Stage1TrainConfig(backbone=key).backbone == key
    for bad in ("caduceus-ps-1k-d118-l5", "production", "", None, 3):
        with pytest.raises(ValueError):
            Stage1TrainConfig(backbone=bad)


def test_the_config_rejects_the_forbidden_rc_combination_at_construction():
    """ADR-0005 D15's order-destroying average composed happily until P2-12 and died only
    after the backbone load on a GPU node — ~20 GPU-h of exposure per submit (A9/A10 class).
    Canonical form is required too: ``diagnostics.config.rc_combine`` is echoed verbatim into
    the report and keys the ablation table, so "CONCAT" would split one leg across two rows.
    """
    from tbox_finder.models.rc_combine import ALLOWED_RC_COMBINE, FORBIDDEN_RC_COMBINE

    for mode in ALLOWED_RC_COMBINE:
        assert Stage1TrainConfig(rc_combine=mode).rc_combine == mode
    for mode in FORBIDDEN_RC_COMBINE:
        with pytest.raises(ValueError, match="order-destroying"):
            Stage1TrainConfig(rc_combine=mode)
    with pytest.raises(ValueError, match="canonical form"):
        Stage1TrainConfig(rc_combine="CONCAT")
    with pytest.raises(ValueError, match="unknown rc_combine"):
        Stage1TrainConfig(rc_combine="bogus")


def test_build_report_refuses_to_emit_a_report_it_would_itself_reject():
    """``backbone_info`` is required, and the refusal is the point.

    A default would have to invent a ``measured_param_count`` — forging a measurement (§10.3)
    — or omit it, and a schema-3 report that omits it fails this module's own
    ``validate_report``. The convenient default therefore writes an artifact the writer would
    reject, which is worse than either: the run "succeeds" and the evidence is invalid. So it
    raises, and a caller with no model must state what it is claiming.
    """
    import tbox_finder.train.train_stage1 as ts

    kwargs = dict(
        cfg=Stage1TrainConfig(),
        class_counts=[10, 2, 1, 1, 1, 1, 1, 1],
        counts_scope={"n_records": 1, "n_training_fold_records": 1, "full_stream": True},
        n_blocks=16,
        n_blocks_wrapped=16,
        hf_flag_supported=False,
        losses=[1.6, 1.5],
        grads_finite=True,
        world_size=1,
        wandb_run_id=None,
    )
    with pytest.raises(ValueError, match="requires backbone_info"):
        ts.build_report(**kwargs)

    spec = R.resolve_checkpoint("caduceus-ps-1k-d118-l4")
    report = ts.build_report(
        **kwargs,
        backbone_info={
            **R.checkpoint_summary(spec),
            "measured_param_count": spec.expected_param_count,
            "measured_d_model": spec.d_model,
        },
    )
    # With the evidence supplied the report validates AND names the leg's own checkpoint —
    # not production, which is what the old module-constant echo would have written.
    assert ts.validate_report(report) == []
    assert report["backbone"]["key"] == "caduceus-ps-1k-d118-l4"
    assert report["backbone"]["measured_param_count"] == 470_702
    assert report["schema_version"] == SCHEMA_VERSION


def test_backbone_is_declared_swept_by_p2_12():
    """``diagnostics.swept_by`` is what tells a later reader which step owns an axis; the
    36 committed sweep points already carry rc_combine/use_crf -> P2-12."""
    swept = diagnostics(Stage1TrainConfig())["swept_by"]
    assert swept["backbone"] == "P2-12"
    assert swept["rc_combine"] == "P2-12" and swept["use_crf"] == "P2-12"


# ────────────────────────────────────────────────────────────────────────────────────────
# The sbatch leg table
# ────────────────────────────────────────────────────────────────────────────────────────
def _sbatch_array(name: str) -> list[str]:
    """The literal bash array ``name=(...)`` as written in the shipped sbatch."""
    text = SBATCH.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name}=("):
            return stripped[len(name) + 2 : stripped.rindex(")")].replace('"', "").split()
    raise AssertionError(f"{name}=(...) not found in {SBATCH.name}")


def test_the_shipped_leg_table_covers_all_three_axes_one_at_a_time():
    """Read the SHIPPED bytes, not a retyped equivalent ([[verify-the-line-you-ship]]).

    Each leg must differ from the baseline arm on **exactly one** axis — a leg that moved two
    would make the table unable to attribute its delta — and the four legs together must cover
    all three axes, or an axis PRD §18.1 enumerates went unmeasured while the table looked
    complete.
    """
    keys = _sbatch_array("KEYS")
    backbones = _sbatch_array("BACKBONES")
    rc_modes = _sbatch_array("RC_MODES")
    use_crfs = _sbatch_array("USE_CRFS")
    assert len(keys) == 4, keys
    assert len(backbones) == len(rc_modes) == len(use_crfs) == 4

    moved: list[str] = []
    for key, backbone, rc, crf in zip(keys, backbones, rc_modes, use_crfs, strict=True):
        backbone = PRODUCTION if backbone == "$PRODUCTION_BACKBONE" else backbone
        R.resolve_checkpoint(backbone)  # every leg names an allow-listed checkpoint
        axes = [
            ("backbone", backbone != PRODUCTION),
            ("rc_combine", rc != "concat"),
            ("use_crf", crf != "false"),
        ]
        differing = [name for name, changed in axes if changed]
        assert len(differing) == 1, f"leg {key} moves {differing}, must move exactly one"
        moved.append(differing[0])
    assert sorted(set(moved)) == ["backbone", "rc_combine", "use_crf"], moved
    assert moved.count("backbone") == 2, "the size axis needs both downward-search points"
    # And the two size legs must be the two DISTINCT ablation checkpoints, not one twice.
    size_backbones = [b for b in backbones if b != "$PRODUCTION_BACKBONE"]
    assert len(set(size_backbones)) == 2, size_backbones


def test_the_averaged_rc_arm_is_absent_from_the_leg_table():
    """ADR-0005 D15 + PRD:133/:212 forbid the order-destroying average outright; it is not a
    negative control and must not appear as a leg (the resolved precedence conflict)."""
    from tbox_finder.models.rc_combine import FORBIDDEN_RC_COMBINE

    assert set(_sbatch_array("RC_MODES")) <= {"concat", "gate"}
    assert not set(_sbatch_array("RC_MODES")) & set(FORBIDDEN_RC_COMBINE)


@pytest.mark.parametrize("name", ["ablate_stage1.sbatch", "backbone_throughput.sbatch"])
def test_the_p2_12_sbatch_headers_and_scratch_variable(name: str):
    """§9.2/§9.3 headers, and the job-789 scratch-variable lesson.

    ``conda activate tbox-ml-dna`` exports ``BUILD=HOST=x86_64-conda-linux-gnu`` from the conda
    toolchain's activate hook, silently clobbering an sbatch's own ``$BUILD``. Name the scratch
    dir anything else.
    """
    body = (REPO / "slurm" / "p2" / name).read_text()
    directives = "\n".join(ln for ln in body.splitlines() if ln.startswith("#SBATCH"))
    assert "--partition=gpu" in directives
    assert "--nodelist" not in directives, "never pin a node (§9.2 — `one` is often down)"
    assert "--account" not in directives and "--qos" not in directives
    assert 'export CUDA_HOME="$CONDA_PREFIX"' in body
    assert "conda activate tbox-ml-dna" in body
    assert 'JOB_SCRATCH="/tmp/${USER}-${SLURM_JOB_ID:-local}"' in body
    assert 'BUILD="/tmp/' not in body, "job 789: conda's activate hook clobbers $BUILD"
    assert "trap 'rm -rf \"$JOB_SCRATCH\"' EXIT" in body


def test_the_array_walltime_covers_the_measured_per_leg_wall():
    """MEASURED: the heaviest of the 36 uniform sweep points ran 0.2025 wall-h. The header must
    clear that with real headroom — the CRF and gate per-step costs are UNMEASURED, which is
    why ADR-0003 A3 carries a +25% allowance for them."""
    import re

    body = SBATCH.read_text()
    match = re.search(r"^#SBATCH --time=(\d+):(\d+):", body, re.M)
    assert match is not None, "the sbatch has no --time header"
    hours = int(match.group(1)) + int(match.group(2)) / 60.0
    assert hours >= 1.0, f"--time={match.group(0)} is thin over a measured 0.2025 wall-h/leg"
    assert re.search(r"^#SBATCH --array=0-3$", body, re.M), "four legs, task ids 0-3"


# ────────────────────────────────────────────────────────────────────────────────────────
# The throughput probe's validator (the report is produced on a GPU; the rules are not)
# ────────────────────────────────────────────────────────────────────────────────────────
def _measurement(key: str, *, seconds: list[float]) -> dict:
    spec = R.resolve_checkpoint(key)
    n_windows = 8 * len(seconds)
    return {
        **R.checkpoint_summary(spec),
        "measured_param_count": spec.expected_param_count,
        "measured_d_model": spec.d_model,
        "window_nt": T.WINDOW_NT,
        "batch_size": 8,
        "timed_batches": len(seconds),
        "n_windows": n_windows,
        "batch_seconds": seconds,
        "windows_per_s_per_gpu": n_windows / sum(seconds),
    }


def _throughput_report() -> dict:
    return {
        "schema_version": T.SCHEMA_VERSION,
        "step": T.STEP,
        "generated_by": T.GENERATED_BY,
        "adr": T.ADR,
        "env_lock": T.ENV_LOCK,
        "is_science": False,
        "disclosures": ["synthetic seeded windows; not a genome-scan rate"],
        "measurements": [_measurement(key, seconds=[0.1] * 30) for key in R.CHECKPOINT_KEYS],
    }


def test_a_well_formed_throughput_report_validates():
    assert T.validate_report(_throughput_report()) == []


def test_a_throughput_rate_that_does_not_re_derive_is_rejected():
    """A rate echoed independently of its own timings is unfalsifiable — the whole point of
    recording per-batch seconds is that the headline number must follow from them."""
    report = _throughput_report()
    report["measurements"][0]["windows_per_s_per_gpu"] *= 2.0
    assert any("does not re-derive" in p for p in T.validate_report(report))


def test_a_throughput_report_without_timings_or_disclosures_is_rejected():
    stripped = _throughput_report()
    del stripped["measurements"][1]["batch_seconds"]
    assert any("unbacked" in p for p in T.validate_report(stripped))

    undisclosed = _throughput_report()
    undisclosed["disclosures"] = []
    assert any("disclosures" in p for p in T.validate_report(undisclosed))

    empty = _throughput_report()
    empty["measurements"] = []
    assert any("measurements" in p for p in T.validate_report(empty))


def test_a_throughput_measurement_must_match_its_pinned_checkpoint():
    report = _throughput_report()
    report["measurements"][0]["measured_param_count"] = 1
    assert any("measured_param_count" in p for p in T.validate_report(report))

    geometry = _throughput_report()
    geometry["measurements"][0]["window_nt"] = 2048
    assert any("frozen" in p for p in T.validate_report(geometry))


def test_a_cpu_throughput_number_is_refused_rather_than_emitted():
    """§10.3 — the packaged Mamba selective_scan is a CUDA kernel (ADR-0002 A2 C2). A CPU
    number would be a ~871x slower fallback reported as if it were the scan rate."""
    with pytest.raises(RuntimeError, match="must be measured on CUDA"):
        T.measure_throughput(PRODUCTION, device="cpu")


def test_an_invalid_table_is_not_published_to_the_canonical_path(tmp_path) -> None:
    """Returning 2 while having already written the canonical file is fail-open in practice.

    ``reports/p2/ablation_table.json`` is git-committed evidence; anything that opens the
    file rather than reading the exit code would take a rejected build for the deliverable.
    So a failed validation diverts to a clearly-marked sibling (CodeRabbit, P2-12 RESULT).
    """
    out = tmp_path / "ablation_table.json"
    # --expect-rows 99 cannot be satisfied by the committed baseline alone, so the artifact
    # is built but fails its own validator.
    rc = A.main(
        [
            "--baseline",
            str(REPO / A.DEFAULT_BASELINE),
            "--ablation-dir",
            str(tmp_path / "no-legs-here"),
            "--out",
            str(out),
            "--expect-rows",
            "99",
        ]
    )
    assert rc == 2
    assert not out.exists(), "a table that failed its validator must not occupy the real path"
    assert out.with_suffix(".invalid.json").exists(), "the rejected build stays inspectable"


def test_a_valid_table_is_published_to_the_canonical_path(tmp_path) -> None:
    out = tmp_path / "ablation_table.json"
    rc = A.main(
        [
            "--baseline",
            str(REPO / A.DEFAULT_BASELINE),
            "--ablation-dir",
            str(tmp_path / "no-legs-here"),
            "--out",
            str(out),
            "--expect-rows",
            "1",
        ]
    )
    assert rc == 0
    assert out.exists()
    assert not out.with_suffix(".invalid.json").exists()
