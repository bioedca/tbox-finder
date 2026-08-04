"""P3-08 — the with/without-aux ablation gate, graded without torch.

Everything the gate decides lives in the numpy tier of ``stage2/eval.py``, which is
why this file can exercise it end to end: calibration, ECE, AUPRC, the clause set and
the validator all run here on synthetic logits. The torch tier (checkpoint loading,
scoring) is graded in ``tests/ml/test_stage2_eval_smoke.py``.

Fixture discipline used throughout:

* **Asymmetric rung sizes.** ``calib``/``val``/``test`` are 24/13/31, and both the
  calib and the test arm are independently fittable, so the "T came from calib" test
  can swap the two senses and assert the resulting ``T`` **is exactly the test arm's
  own** — an identity, not a difference. Equal-sized arms plus a count assertion would
  survive a sense inversion untouched ([[symmetric-count-fixture-blind-to-inversion]]).
* **Every refusal is paired with a positive control** — the same input, one thing
  changed, succeeding. ``pytest.raises`` alone is equally satisfied by a guard that
  refuses everything ([[raises-test-needs-a-positive-control]]).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tbox_finder import metrics as M
from tbox_finder.calib import temperature as TEMP
from tbox_finder.stage2 import eval as E

REPO_ROOT = Path(__file__).resolve().parents[2]

N_CALIB, N_VAL, N_TEST = 24, 13, 31


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _arm(
    *,
    name: str = "arm",
    calib_scale: float = 3.0,
    test_scale: float = 3.0,
    val_scale: float = 3.0,
    seed: int = 7,
    swap_calib_and_test: bool = False,
) -> E.ArmScores:
    """Synthetic logits whose *true* well-calibrated scale is 1.0 per rung.

    Labels are drawn from ``sigma(z)``, then the recorded logit is ``scale * z`` — so
    an arm built with ``calib_scale=3`` is overconfident by exactly 3× on calib and a
    correct fit must return ``T ~= 3``. Rung-specific scales let a test tell "fitted on
    calib" apart from "fitted on whatever came first".
    """
    rng = np.random.default_rng(seed)
    rungs: list[str] = ["calib"] * N_CALIB + ["val"] * N_VAL + ["test"] * N_TEST
    scale_of = {"calib": calib_scale, "val": val_scale, "test": test_scale}
    z_true = rng.normal(0.0, 1.2, len(rungs))
    labels = (rng.random(len(rungs)) < 1.0 / (1.0 + np.exp(-z_true))).astype(np.int64)
    logits = np.asarray([z * scale_of[r] for z, r in zip(z_true, rungs, strict=True)])
    if swap_calib_and_test:
        rungs = ["test" if r == "calib" else "calib" if r == "test" else r for r in rungs]
    return E.ArmScores(
        arm=name,
        row_ids=tuple(f"row{i:03d}" for i in range(len(rungs))),
        logits=logits,
        labels=labels,
        rungs=tuple(rungs),
        # 4 rows per block, so the bootstrap has real blocks rather than N singletons.
        blocks=tuple(f"cluster:{i // 4}" for i in range(len(rungs))),
    )


def _graded(**kwargs: Any) -> dict[str, Any]:
    return E.grade_arm(_arm(**kwargs), n_boot=20)


def _report(
    *,
    with_aux_ece_scale: float = 3.0,
    no_aux_ece_scale: float = 3.0,
) -> dict[str, Any]:
    """A complete, passing report — the object the clause tests sabotage."""
    with_aux = _arm(name="aux1.0_lr1e-4", calib_scale=with_aux_ece_scale, seed=11)
    no_aux = _arm(name="aux0.0_lr1e-4", calib_scale=no_aux_ece_scale, seed=12)
    configs = {
        "aux1.0_lr1e-4": {"aux_weight": 1.0, "lr": 1e-4, "checkpoint_dir": "ckpt/a"},
        "aux0.0_lr1e-4": {"aux_weight": 0.0, "lr": 1e-4, "checkpoint_dir": "ckpt/b"},
    }
    loads = {
        name: {
            "n_adapter_tensors_in_file": 462,
            "n_adapter_tensors_matched": 462,
            "n_adapter_tensors_mismatched": 0,
            "n_module_adapter_tensors": 462,
            "n_module_adapter_tensors_absent_from_file": 0,
            "n_lora_b_tensors": 231,
            "n_lora_b_nonzero": 231,
            "n_head_tensors_in_file": 12,
            "n_head_tensors_matched": 12,
            "attn_implementation": "flash_attention_2",
            "attn_implementation_trained_under": "flash_attention_2",
            "attn_implementation_matches_training": True,
        }
        for name in configs
    }
    return E.aux_ablation_check(
        {"aux1.0_lr1e-4": with_aux, "aux0.0_lr1e-4": no_aux},
        arm_configs=configs,
        with_aux_arm="aux1.0_lr1e-4",
        no_aux_arm="aux0.0_lr1e-4",
        dataset={
            "path": "data/processed/stage2_dataset.parquet",
            "rung_census": {"calib": N_CALIB, "val": N_VAL, "test": N_TEST},
        },
        load_records=loads,
        n_boot=20,
    )


# --------------------------------------------------------------------------- #
# ArmScores — the shape guards
# --------------------------------------------------------------------------- #
def test_arm_scores_accepts_a_well_formed_arm() -> None:
    """Positive control for every refusal below."""
    scores = _arm()
    assert scores.census() == {"calib": N_CALIB, "train": 0, "val": N_VAL, "test": N_TEST}
    assert len(scores.index_of("test")) == N_TEST


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda kw: kw.update(labels=kw["labels"][:-1]), "labels has"),
        (lambda kw: kw.update(rungs=kw["rungs"][:-1]), "rungs has"),
        (lambda kw: kw.update(blocks=kw["blocks"][:-1]), "blocks has"),
        (
            lambda kw: kw.update(row_ids=("dup",) * len(kw["row_ids"])),
            "row_ids are not unique",
        ),
        (
            lambda kw: kw.update(rungs=("bogus",) + kw["rungs"][1:]),
            "unknown rung token",
        ),
    ],
)
def test_arm_scores_refuses_malformed_input(mutate: Any, message: str) -> None:
    base = _arm()
    kwargs: dict[str, Any] = {
        "arm": base.arm,
        "row_ids": base.row_ids,
        "logits": base.logits,
        "labels": base.labels,
        "rungs": base.rungs,
        "blocks": base.blocks,
    }
    mutate(kwargs)
    with pytest.raises(ValueError, match=message):
        E.ArmScores(**kwargs)


def test_arm_scores_refuses_an_empty_arm() -> None:
    with pytest.raises(ValueError, match="no rows were scored"):
        E.ArmScores(
            arm="empty",
            row_ids=(),
            logits=np.asarray([]),
            labels=np.asarray([]),
            rungs=(),
            blocks=(),
        )


# --------------------------------------------------------------------------- #
# block_keys — NaN must not become one giant block
# --------------------------------------------------------------------------- #
def test_cluster_less_rows_become_singleton_blocks_not_one_block() -> None:
    """A NaN ``cluster_id`` is "no cluster", not "the NaN cluster".

    Every decoy carries NaN here. Folding them together would leave the bootstrap
    resampling 3 blocks instead of 22 and silently collapse its variance
    ([[nulls-inflate-block-counts]]).
    """
    rows = [{"row_id": f"c{i}", "cluster_id": float(i % 2)} for i in range(4)]
    rows += [{"row_id": f"d{i}", "cluster_id": float("nan")} for i in range(20)]
    keys, census = E.block_keys(rows)
    assert census["n_rows_with_cluster"] == 4
    assert census["n_rows_without_cluster"] == 20
    assert census["n_blocks_from_clusters"] == 2
    assert census["n_blocks"] == 22, "the 20 cluster-less rows collapsed into one block"
    assert len(set(keys[4:])) == 20


def test_block_keys_refuses_a_cluster_less_row_with_no_row_id() -> None:
    with pytest.raises(ValueError, match="no block"):
        E.block_keys([{"cluster_id": float("nan")}])
    # Positive control: the identical row, with an id, is fine.
    keys, _ = E.block_keys([{"row_id": "x", "cluster_id": float("nan")}])
    assert keys == ["row:x"]


def test_none_cluster_is_treated_the_same_as_nan() -> None:
    keys, census = E.block_keys(
        [{"row_id": "a", "cluster_id": None}, {"row_id": "b", "cluster_id": None}]
    )
    assert census["n_blocks"] == 2
    assert keys == ["row:a", "row:b"]


# --------------------------------------------------------------------------- #
# grade_arm — the calibration contract
# --------------------------------------------------------------------------- #
def test_temperature_recovers_a_manufactured_miscalibration() -> None:
    """Overconfidence of 3× must come back as ``T ~= 3`` — the fit's whole job."""
    graded = _graded(calib_scale=3.0, seed=3)
    assert graded["calibration"]["temperature"] == pytest.approx(3.0, rel=0.35)
    assert graded["calibration"]["fitted_on"] == "calib"
    assert graded["calibration"]["n_fitted"] == N_CALIB


def test_temperature_is_fitted_on_calib_and_the_swap_yields_the_test_arms_own_T() -> None:
    """Identity, not difference: swapping the rung senses must return the *other* fit.

    The arms are given different miscalibration scales and different sizes, and the
    swapped run's ``T`` is compared against a direct fit on the test arm alone. A test
    that only asserted "the two differ" would pass for a fit that read some third
    thing ([[symmetric-count-fixture-blind-to-inversion]]).
    """
    normal = _graded(calib_scale=3.0, test_scale=1.6, seed=5)
    swapped = _graded(calib_scale=3.0, test_scale=1.6, seed=5, swap_calib_and_test=True)

    reference = _arm(calib_scale=3.0, test_scale=1.6, seed=5)
    test_idx = reference.index_of("test")
    direct = TEMP.fit_temperature(
        np.stack([np.zeros(test_idx.size), np.asarray(reference.logits)[test_idx]], axis=1),
        np.asarray(reference.labels)[test_idx],
    )

    assert normal["calibration"]["n_fitted"] == N_CALIB
    assert swapped["calibration"]["n_fitted"] == N_TEST
    assert swapped["calibration"]["temperature"] == pytest.approx(
        direct.temperature, rel=1e-9
    ), "the swapped fit is not the test arm's own T, so the rung is not what selects"
    assert normal["calibration"]["temperature"] != pytest.approx(
        swapped["calibration"]["temperature"], rel=1e-6
    )


def _separated_arm(*, flip_one_calib_row: bool) -> E.ArmScores:
    n = 30
    rungs = ["calib"] * 12 + ["val"] * 6 + ["test"] * 12
    labels = np.asarray([i % 2 for i in range(n)], dtype=np.int64)
    logits = np.where(labels == 1, 8.0, -8.0).astype(np.float64)
    if flip_one_calib_row:
        logits[0] = -logits[0]
    return E.ArmScores(
        arm="separated",
        row_ids=tuple(f"r{i}" for i in range(n)),
        logits=logits,
        labels=labels,
        rungs=tuple(rungs),
        blocks=tuple(f"cluster:{i // 3}" for i in range(n)),
    )


def test_a_perfectly_separated_calib_split_yields_no_ece_rather_than_a_default_T() -> None:
    """The risk that actually fired on the real checkpoints, in miniature.

    When every calib row is already on the right side of zero, the exact minimiser of
    the NLL is ``beta -> inf`` (``T -> 0``) and P3-07's fitter refuses it. The contract
    asserted here is what happens **next**: the refusal is recorded as evidence, the
    gated ``ece`` comes back ``None``, and no temperature is invented. Substituting
    ``T = 1`` would put a real-looking number under the gated key while measuring
    something GATE-2 never asked for (§10.3).
    """
    graded = E.grade_arm(_separated_arm(flip_one_calib_row=False), n_boot=10)
    calibration = graded["calibration"]
    assert calibration["fitted"] is False
    assert calibration["temperature"] is None
    assert calibration["refusal"]["classification"] == "perfect_separation_beta_to_infinity"
    assert calibration["calib_separation"]["is_perfectly_separated"] is True
    assert calibration["calib_separation"]["n_misclassified_at_zero"] == 0
    assert graded["stack"]["named_posterior_exists"] is False

    block = graded["grades"]["test"]
    assert block["ece"] is None and block["ece_gate_pass"] is None
    assert "does not exist" in block["ece_unavailable_reason"]
    # Ranking survives: average precision needs no temperature and a monotone
    # rescaling could not have moved it.
    assert block["auprc"] == pytest.approx(1.0)
    # The T=1 diagnostic is present but is NOT under the gated key.
    assert block["uncalibrated_diagnostic"]["ece_at_T1"] is not None


def test_the_positive_control_one_wrong_calib_row_makes_the_fit_admissible() -> None:
    """Same arm, one calib row flipped: a temperature exists and the ECE is a number.

    Without this the refusal test above is equally satisfied by a ``grade_arm`` that
    never fits anything at all ([[raises-test-needs-a-positive-control]]).
    """
    graded = E.grade_arm(_separated_arm(flip_one_calib_row=True), n_boot=10)
    assert graded["calibration"]["fitted"] is True
    assert graded["calibration"]["temperature"] > 0.0
    assert graded["calibration"]["calib_separation"]["is_perfectly_separated"] is False
    assert graded["stack"]["named_posterior_exists"] is True
    assert isinstance(graded["grades"]["test"]["ece"], float)


def test_an_unfittable_arm_makes_the_ece_axis_undecidable_not_equal() -> None:
    """A missing ECE must not be differenced into a comforting ``delta_ece = 0``."""
    unfittable = E.grade_arm(_separated_arm(flip_one_calib_row=False), n_boot=10)
    fittable = E.grade_arm(_separated_arm(flip_one_calib_row=True), n_boot=10)
    unfittable["arm"], fittable["arm"] = "aux1.0_lr1e-4", "aux0.0_lr1e-4"
    out = E.compare_arms(unfittable, fittable)
    assert out["ece_comparison_available"] is False
    assert out["delta_ece"] is None
    assert out["reading_absolute"]["passes"] is None
    assert out["reading_delta"]["ece_axis_decidable"] is False
    assert out["divergence"]["readings_disagree_for_ece_tolerance_below"] is None
    assert "unanswerable" in out["divergence"]["note"]
    # The ranking axis stays decidable — that is the whole point of separating them.
    assert out["delta_auprc"] is not None


def test_a_graded_rung_with_no_rows_refuses_instead_of_scoring_nothing() -> None:
    """An empty grade reads exactly like a passing one, so it must not be produced."""
    scores = _arm()
    with pytest.raises(ValueError, match="no rows on the 'train' rung"):
        E.grade_arm(scores, graded_rungs=("train",), n_boot=10)
    # Positive control: the identical arm graded on a rung it *has*.
    assert E.grade_arm(scores, graded_rungs=("test",), n_boot=10)["grades"]["test"]["n"] == N_TEST


def test_the_graded_object_is_the_pre_prior_shift_named_posterior() -> None:
    graded = _graded()
    assert graded["stack"]["gated_posterior_key"] == "named_posterior"
    assert graded["stack"]["prior_shift_applied"] is False
    assert graded["stack"]["stack_applied"] == ["train", "temperature_scale"]
    assert graded["stack"]["stack_order"] == ["train", "temperature_scale", "prior_shift"]


def test_ece_uses_the_adr_pinned_estimator() -> None:
    graded = _graded()
    for rung in ("val", "test"):
        block = graded["grades"][rung]
        assert block["ece_n_bins"] == 15 == M.ECE_N_BINS
        assert block["ece_binning"] == "equal_mass"
        assert block["ece_debiased"] is True
        assert block["ece_gate"] == 0.05
        # Debiasing subtracts a noise floor, so it can only lower the estimate.
        assert block["ece"] <= block["ece_plugin"] + 1e-12


def test_auprc_is_computed_on_logits_and_is_invariant_to_the_temperature() -> None:
    """``z -> z/T`` is strictly monotone, so average precision cannot move."""
    graded = _graded()
    block = graded["grades"]["test"]
    assert block["auprc"] == block["auprc_scaled_logits"]
    assert block["auprc_rank_invariant"] is True


def test_posterior_saturation_is_counted_because_it_destroys_ranking() -> None:
    """A saturated posterior manufactures ties the logits never had.

    This is the reason AUPRC is graded on logits. With a small ``T`` and large logits
    ``sigma(z/T)`` hits exactly 1.0, every saturated row ties, and AP computed on the
    posterior stops agreeing with AP on the logits. The report must record how many
    rows are in that state rather than quietly grading through it.
    """
    # calib carries MODEST margins with two mistakes, so the fit is admissible and
    # lands near T = 1. (A mistaken calib row pushes T *up*; saturation on test comes
    # from test margins being far larger than calib's, not from a small T.)
    calib_labels = [1] * 8 + [0] * 8
    calib_logits = [2.0] * 8 + [-2.0] * 8
    calib_logits[0], calib_logits[8] = -2.0, 2.0  # two errors

    # test carries margins so large that sigma() reaches exactly 1.0 / 0.0 in float64,
    # and the top-ranked row is a NEGATIVE — so the ordering the logits carry is real
    # information that saturation destroys by tying it away.
    test_labels = [0] + [1] * 6 + [1] + [0] * 6
    test_logits = [910.0] + [900.0] * 6 + [890.0] + [-900.0] * 6

    labels = np.asarray(calib_labels + test_labels, dtype=np.int64)
    logits = np.asarray(calib_logits + test_logits, dtype=np.float64)
    rungs = ["calib"] * len(calib_labels) + ["test"] * len(test_labels)
    scores = E.ArmScores(
        arm="saturating",
        row_ids=tuple(f"r{i}" for i in range(len(labels))),
        logits=logits,
        labels=labels,
        rungs=tuple(rungs),
        blocks=tuple(f"cluster:{i // 4}" for i in range(len(labels))),
    )
    block = E.grade_arm(scores, graded_rungs=("test",), n_boot=10)["grades"]["test"]

    saturated = block["posterior_saturation"]
    assert saturated["n_at_one"] == 8, "the fixture did not actually saturate"
    assert saturated["n_at_zero"] == 6
    # The invariant survives on logits precisely because it was not read off the
    # posterior — which here has tied 7 positives to a negative and lost the ordering.
    assert block["auprc_rank_invariant"] is True
    assert block["auprc_on_posterior"] != pytest.approx(block["auprc"], abs=1e-9), (
        "grading AUPRC on the posterior would have returned a different number here, "
        "which is exactly why it is graded on the logits"
    )


def test_grades_carry_prevalence_and_a_block_resampled_ci() -> None:
    block = _graded()["grades"]["test"]
    assert block["n"] == N_TEST
    assert block["n_positive"] + block["n_negative"] == N_TEST
    assert block["prevalence"] == pytest.approx(block["n_positive"] / N_TEST)
    assert block["auprc_baseline_prevalence"] == pytest.approx(block["prevalence"])
    assert block["ece_ci"]["n_blocks"] == block["block_census"]["n_blocks"]
    assert block["ece_ci"]["lower"] <= block["ece_ci"]["point"] <= block["ece_ci"]["upper"]


# --------------------------------------------------------------------------- #
# compare_arms — both readings of D16, neither resolved
# --------------------------------------------------------------------------- #
def _fake_grade(*, arm: str, ece: float, auprc: float, passes: bool) -> dict[str, Any]:
    return {
        "arm": arm,
        "grades": {
            "test": {
                "ece": ece,
                "auprc": auprc,
                "ece_gate_pass": passes,
                "ece_ci": {"point": ece, "lower": ece - 0.01, "upper": ece + 0.01},
            }
        },
    }


def test_the_delta_reading_emits_no_verdict_because_no_tolerance_exists() -> None:
    out = E.compare_arms(
        _fake_grade(arm="w", ece=0.03, auprc=0.90, passes=True),
        _fake_grade(arm="n", ece=0.02, auprc=0.92, passes=True),
    )
    assert out["reading_delta"]["tolerance"] is None
    assert out["reading_delta"]["verdict"] == "unpinned"
    assert out["verdict"] == "requires_signoff"


def test_the_divergence_window_is_the_observed_degradation() -> None:
    """The number a §7 sign-off needs: every tau below it separates the two readings."""
    out = E.compare_arms(
        _fake_grade(arm="w", ece=0.030, auprc=0.900, passes=True),
        _fake_grade(arm="n", ece=0.020, auprc=0.925, passes=True),
    )
    assert out["reading_absolute"]["passes"] is True
    assert out["delta_ece"] == pytest.approx(0.010)
    assert out["delta_auprc"] == pytest.approx(-0.025)
    assert out["divergence"]["readings_disagree_for_ece_tolerance_below"] == pytest.approx(0.010)
    assert out["divergence"]["readings_disagree_for_auprc_tolerance_below"] == pytest.approx(0.025)


def test_a_with_aux_arm_that_is_better_leaves_an_empty_divergence_window() -> None:
    out = E.compare_arms(
        _fake_grade(arm="w", ece=0.010, auprc=0.950, passes=True),
        _fake_grade(arm="n", ece=0.030, auprc=0.900, passes=True),
    )
    assert out["delta_ece"] < 0
    assert out["reading_delta"]["observed_ece_degradation"] == 0.0
    assert out["divergence"]["readings_disagree_for_ece_tolerance_below"] == 0.0
    assert out["divergence"]["readings_disagree_for_auprc_tolerance_below"] == 0.0


def test_the_absolute_reading_fails_when_the_with_aux_arm_misses_the_d11_gate() -> None:
    out = E.compare_arms(
        _fake_grade(arm="w", ece=0.20, auprc=0.90, passes=False),
        _fake_grade(arm="n", ece=0.01, auprc=0.90, passes=True),
    )
    assert out["reading_absolute"]["passes"] is False
    assert "moot" in out["divergence"]["note"]
    # Positive control: identical shape, gate held -> the reading passes.
    assert (
        E.compare_arms(
            _fake_grade(arm="w", ece=0.02, auprc=0.90, passes=True),
            _fake_grade(arm="n", ece=0.01, auprc=0.90, passes=True),
        )["reading_absolute"]["passes"]
        is True
    )


def test_the_d17_reference_margins_are_labelled_non_governing() -> None:
    out = E.compare_arms(
        _fake_grade(arm="w", ece=0.03, auprc=0.90, passes=True),
        _fake_grade(arm="n", ece=0.02, auprc=0.92, passes=True),
    )
    reference = out["reference_margins_not_governing"]
    assert "NOT the D16 aux-ablation bar" in reference["source"]
    assert reference["would_pass_ece"] is True  # 0.010 <= 0.02


def test_the_d17_reference_margins_still_match_the_adr() -> None:
    """Drift guard: this module carries a copy, so the copy is checked against source.

    ADR-0005 D17 is prose, not a constant, so the two numbers had to be written down
    here. Re-reading them out of the decision means an amendment that moves either one
    turns this test red instead of leaving a stale quotation in a shipped report
    ([[pinned-constant-that-nothing-reads]]).
    """
    adr = (REPO_ROOT / "docs/decisions/ADR-0005-non-circular-eval-design.md").read_text(
        encoding="utf-8"
    )
    d17 = adr.split("### D17.", 1)[1].split("### D18.", 1)[0]
    assert re.search(r"0\.02\b", d17), "ADR-0005 D17 no longer states a 0.02 ECE margin"
    assert re.search(r"0\.03\b", d17), "ADR-0005 D17 no longer states a 0.03 AUPRC margin"
    assert E.D17_REFERENCE_MARGINS == {"ece": 0.02, "auprc": 0.03}


# --------------------------------------------------------------------------- #
# Arm discovery + the production arm
# --------------------------------------------------------------------------- #
def _write_arm(root: Path, sweep: Path, name: str, *, aux_weight: float, lr: float) -> None:
    (root / name / E.ADAPTER_SUBDIR).mkdir(parents=True)
    (root / name / E.HEADS_STATE_NAME).write_bytes(b"")
    (sweep / f"{name}.json").write_text(
        json.dumps(
            {
                "config": {"lr": lr, "loss": {"aux_weight": aux_weight}},
                "legacy": {"saved_val_total": 0.01, "saved_from_epoch": 9},
            }
        ),
        encoding="utf-8",
    )


def _sweep(tmp_path: Path) -> tuple[Path, Path]:
    root, sweep = tmp_path / "ckpt", tmp_path / "sweep"
    root.mkdir()
    sweep.mkdir()
    for aux in (0.0, 0.5, 1.0):
        for lr in (1e-4, 3e-4):
            _write_arm(root, sweep, f"aux{aux}_lr{lr:.0e}", aux_weight=aux, lr=lr)
    return root, sweep


def test_discover_arms_keeps_an_openable_path_beside_the_recordable_one(
    tmp_path: Path,
) -> None:
    """CodeRabbit r1 follow-on: relativising the path the LOADER opens broke the run.

    `checkpoint_dir` is written into a committed public artifact and must be
    repo-relative; `checkpoint_path` is what `load_stage2_checkpoint` opens and must
    resolve from wherever the process runs. Collapsing them into one field made the
    report clean and the scoring step raise FileNotFoundError.
    """
    root, sweep = _sweep(tmp_path)
    arms = E.discover_arms(root, sweep_dir=sweep)
    for arm in arms.values():
        assert Path(arm["checkpoint_path"]).is_dir(), "the openable path does not resolve"
        assert Path(arm["checkpoint_path"]).is_absolute()


def test_discover_arms_reads_each_arms_own_run_report(tmp_path: Path) -> None:
    root, sweep = _sweep(tmp_path)
    arms = E.discover_arms(root, sweep_dir=sweep)
    assert len(arms) == 6
    assert {a["aux_weight"] for a in arms.values()} == {0.0, 0.5, 1.0}
    assert all(a["saved_from_epoch"] == 9 for a in arms.values())


def test_discover_arms_refuses_a_checkpoint_with_no_run_report(tmp_path: Path) -> None:
    root, sweep = _sweep(tmp_path)
    (root / "aux9.9_lr9e-9" / E.ADAPTER_SUBDIR).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no run report"):
        E.discover_arms(root, sweep_dir=sweep)
    # Positive control: give it a report and the same tree is accepted.
    (sweep / "aux9.9_lr9e-9.json").write_text(
        json.dumps({"config": {"lr": 9e-9, "loss": {"aux_weight": 9.9}}, "legacy": {}}),
        encoding="utf-8",
    )
    assert len(E.discover_arms(root, sweep_dir=sweep)) == 7


def test_select_arm_pair_picks_the_production_point_and_its_lr_matched_control(
    tmp_path: Path,
) -> None:
    root, sweep = _sweep(tmp_path)
    arms = E.discover_arms(root, sweep_dir=sweep)
    with_aux, no_aux = E.select_arm_pair(arms, production={"aux_weight": 1.0, "lr": 1e-4})
    assert arms[with_aux]["aux_weight"] == 1.0
    assert arms[no_aux]["aux_weight"] == 0.0
    assert arms[with_aux]["lr"] == arms[no_aux]["lr"] == 1e-4


def test_select_arm_pair_refuses_when_conf_matches_no_trained_arm(tmp_path: Path) -> None:
    """The drift this derivation exists to catch: conf/ moved, the sweep did not."""
    root, sweep = _sweep(tmp_path)
    arms = E.discover_arms(root, sweep_dir=sweep)
    with pytest.raises(ValueError, match="matches 0 trained arms"):
        E.select_arm_pair(arms, production={"aux_weight": 0.7, "lr": 1e-4})
    # Positive control: the identical arm set, at a config that *was* trained.
    assert E.select_arm_pair(arms, production={"aux_weight": 0.5, "lr": 3e-4})


def test_select_arm_pair_refuses_a_production_config_that_is_itself_no_aux(
    tmp_path: Path,
) -> None:
    root, sweep = _sweep(tmp_path)
    arms = E.discover_arms(root, sweep_dir=sweep)
    with pytest.raises(ValueError, match="itself the no-aux arm"):
        E.select_arm_pair(arms, production={"aux_weight": 0.0, "lr": 1e-4})


def test_select_arm_pair_refuses_when_no_lr_matched_control_exists(tmp_path: Path) -> None:
    """Unmatched arms would confound the aux effect with a learning-rate effect."""
    root, sweep = tmp_path / "c2", tmp_path / "s2"
    root.mkdir()
    sweep.mkdir()
    _write_arm(root, sweep, "aux1.0_lr1e-4", aux_weight=1.0, lr=1e-4)
    _write_arm(root, sweep, "aux0.0_lr3e-4", aux_weight=0.0, lr=3e-4)
    arms = E.discover_arms(root, sweep_dir=sweep)
    with pytest.raises(ValueError, match="learning-rate-matched control"):
        E.select_arm_pair(arms, production={"aux_weight": 1.0, "lr": 1e-4})
    # Positive control: add the matched control and the same call succeeds.
    _write_arm(root, sweep, "aux0.0_lr1e-4", aux_weight=0.0, lr=1e-4)
    assert E.select_arm_pair(
        E.discover_arms(root, sweep_dir=sweep), production={"aux_weight": 1.0, "lr": 1e-4}
    ) == ("aux1.0_lr1e-4", "aux0.0_lr1e-4")


def test_production_arm_config_coerces_a_string_typed_learning_rate(tmp_path: Path) -> None:
    """``lr: 1e-4`` (no dot) comes back a *string* from PyYAML; it must still match."""
    pytest.importorskip("yaml")
    (tmp_path / "loss.yaml").write_text("aux_weight: 1.0\n", encoding="utf-8")
    (tmp_path / "optim.yaml").write_text("lr: 1e-4\n", encoding="utf-8")
    resolved = E.production_arm_config(
        loss_conf=tmp_path / "loss.yaml", optim_conf=tmp_path / "optim.yaml"
    )
    assert resolved == {"aux_weight": 1.0, "lr": 1e-4}


def test_production_arm_config_refuses_a_config_missing_the_key(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    (tmp_path / "loss.yaml").write_text("something_else: 1.0\n", encoding="utf-8")
    (tmp_path / "optim.yaml").write_text("lr: 1e-4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no top-level 'aux_weight'"):
        E.production_arm_config(
            loss_conf=tmp_path / "loss.yaml", optim_conf=tmp_path / "optim.yaml"
        )


def test_the_shipped_conf_still_names_a_trained_arm() -> None:
    """The live drift guard: ``conf/`` and the P3-06 sweep must still agree.

    This reads the repo's real configs and its real sweep reports. Editing
    ``conf/loss/stage2.yaml``'s ``aux_weight`` without retraining turns this red,
    which is the whole reason the production arm is derived instead of written down.
    """
    pytest.importorskip("yaml")
    sweep_dir = REPO_ROOT / E.DEFAULT_SWEEP_DIR
    if not sweep_dir.is_dir():
        pytest.skip("P3-06 sweep reports are not present in this checkout")
    production = E.production_arm_config(
        loss_conf=REPO_ROOT / E.LOSS_CONF, optim_conf=REPO_ROOT / E.OPTIM_CONF
    )
    arms: dict[str, dict[str, Any]] = {}
    for report_path in sorted(sweep_dir.glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report.get("config") or {}
        arms[report_path.stem] = {
            "aux_weight": float(config["loss"]["aux_weight"]),
            "lr": float(config["lr"]),
        }
    with_aux, no_aux = E.select_arm_pair(arms, production=production)
    assert arms[with_aux]["aux_weight"] > 0.0
    assert arms[no_aux]["aux_weight"] == 0.0
    assert arms[with_aux]["lr"] == arms[no_aux]["lr"]


# --------------------------------------------------------------------------- #
# The report: clauses, validator, and what they refuse to certify
# --------------------------------------------------------------------------- #
def test_a_complete_run_passes_its_own_validator() -> None:
    report = _report()
    assert E.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True
    assert set(report["gate"]["clauses"]) == set(E.derive_clauses(report))


def test_the_verdict_is_not_a_gate_clause() -> None:
    """The machinery is graded; the ablation verdict is not — its threshold is missing."""
    report = _report()
    assert all("verdict" not in name for name in report["gate"]["clauses"])
    assert report["ablation"]["verdict"] == "requires_signoff"
    assert report["ablation"]["reading_delta"]["tolerance"] is None


def test_the_validator_refuses_a_report_that_pinned_a_tolerance() -> None:
    """Setting τ is an ADR-0005 D16 amendment, not something a run may do to itself."""
    report = _report()
    report["ablation"]["reading_delta"]["tolerance"] = 0.02
    problems = E.validate_report(report)
    assert any("no tolerance is pre-registered" in p for p in problems)


def test_the_validator_refuses_a_self_certified_verdict() -> None:
    report = _report()
    report["ablation"]["verdict"] = "pass"
    assert any("may not self-certify" in p for p in E.validate_report(report))


def test_the_validator_catches_a_fabricated_clause() -> None:
    """``all(clauses)`` never catches a clause flipped TRUE — re-derivation does."""
    report = _report()
    report["arms"]["aux1.0_lr1e-4"]["load"]["n_lora_b_nonzero"] = 0
    report["gate"]["clauses"]["adapter_is_live_not_identity"] = True  # the lie
    problems = E.validate_report(report)
    assert any("adapter_is_live_not_identity" in p for p in problems)


CLAUSE_SABOTAGE: list[tuple[str, Any]] = [
    (
        "adapter_weights_verified_against_file",
        lambda r: r["arms"]["aux0.0_lr1e-4"]["load"].update(n_adapter_tensors_mismatched=1),
    ),
    (
        "adapter_is_live_not_identity",
        lambda r: r["arms"]["aux1.0_lr1e-4"]["load"].update(n_lora_b_nonzero=0),
    ),
    (
        "head_weights_verified_against_file",
        lambda r: r["arms"]["aux1.0_lr1e-4"]["load"].update(n_head_tensors_matched=11),
    ),
    (
        "scored_under_the_training_attention_backend",
        lambda r: r["arms"]["aux0.0_lr1e-4"]["load"].update(
            attn_implementation_matches_training=False
        ),
    ),
    (
        "temperature_fitted_on_calib_only",
        lambda r: r["arms"]["aux1.0_lr1e-4"]["calibration"].update(fitted_on="test"),
    ),
    (
        "temperature_positive_and_converged",
        lambda r: r["arms"]["aux0.0_lr1e-4"]["calibration"].update(converged=False),
    ),
    (
        "in_distribution_ece_is_computable",
        lambda r: r["arms"]["aux1.0_lr1e-4"]["grades"]["test"].update(ece=None),
    ),
    (
        "graded_object_is_pre_prior_shift",
        lambda r: r["arms"]["aux1.0_lr1e-4"]["stack"].update(prior_shift_applied=True),
    ),
    (
        "ece_estimator_matches_adr_d11",
        lambda r: r["arms"]["aux1.0_lr1e-4"]["grades"]["test"].update(ece_n_bins=10),
    ),
    (
        "auprc_is_rank_invariant_under_scaling",
        lambda r: r["arms"]["aux0.0_lr1e-4"]["grades"]["test"].update(auprc_rank_invariant=False),
    ),
    (
        "graded_rung_is_the_gate2_split",
        lambda r: r["ablation"].update(graded_on_rung="val"),
    ),
    (
        "scored_every_row_of_every_scored_rung",
        lambda r: r["arms"]["aux1.0_lr1e-4"]["scored_census"].update(test=N_TEST - 1),
    ),
    (
        "both_ablation_arms_present",
        lambda r: r["ablation"].update(no_aux_arm="aux1.0_lr1e-4"),
    ),
    ("ablation_arms_are_lr_matched", lambda r: r["ablation"].update(matched_lr=False)),
    (
        "ablation_contrast_is_aux_weight",
        lambda r: r["arms"]["aux0.0_lr1e-4"].update(aux_weight=0.5),
    ),
]


@pytest.mark.parametrize(
    ("clause", "sabotage"), CLAUSE_SABOTAGE, ids=[c for c, _ in CLAUSE_SABOTAGE]
)
def test_every_clause_bites(clause: str, sabotage: Any) -> None:
    """Each clause is flipped false by a targeted, *plausible* corruption of the evidence.

    Named per clause rather than asserting "the gate went red": a red gate proves some
    clause bit, not the one meant ([[sabotage-attribution-names-the-test]]).
    """
    report = _report()
    assert E.derive_clauses(report)[clause] is True, "the clause was not TRUE to begin with"
    sabotage(report)
    clauses = E.derive_clauses(report)
    assert clauses[clause] is False, f"{clause} survived its sabotage"
    assert not all(clauses.values())


def test_the_completeness_clause_catches_a_truncated_run() -> None:
    """Every rule-shaped clause survives ``--max-rows-per-rung``; this one must not.

    A smoke run at 8 rows a rung fits a temperature, grades an ECE and satisfies every
    correctness clause in the set. Only a clause comparing what was scored against the
    dataset's own census can tell that run apart from the real one
    ([[cost-knobs-can-certify]]).
    """
    report = _report()
    truncated = {"calib": 8, "val": 8, "test": 8}
    for arm in report["arms"].values():
        arm["scored_census"] = {**arm["scored_census"], **truncated}
    clauses = E.derive_clauses(report)
    assert clauses["scored_every_row_of_every_scored_rung"] is False
    others = {k: v for k, v in clauses.items() if k != "scored_every_row_of_every_scored_rung"}
    assert all(others.values()), (
        "a truncated run should trip ONLY the completeness clause — if a correctness "
        "clause also fired, this fixture is not testing what it claims"
    )


def test_clauses_are_false_not_missing_when_the_report_is_empty() -> None:
    """A clause read off an absent block must be FALSE, never vacuously true."""
    clauses = E.derive_clauses({})
    assert clauses
    assert not any(clauses.values())


# --------------------------------------------------------------------------- #
# The adapter-key normalisation the torch tier depends on
# --------------------------------------------------------------------------- #
def test_adapter_key_normalisation_strips_only_the_adapter_name() -> None:
    live = "base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight"
    on_disk = "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight"
    assert E._normalise_adapter_key(live) == on_disk
    # Idempotent on a key that never carried the segment.
    assert E._normalise_adapter_key(on_disk) == on_disk


def test_normalisation_is_injective_over_the_real_key_shape() -> None:
    keys = [
        f"base_model.model.encoder.layer.{i}.attention.self.{proj}.lora_{ab}.default.weight"
        for i in range(3)
        for proj in ("query", "key", "value")
        for ab in ("A", "B")
    ]
    assert len({E._normalise_adapter_key(k) for k in keys}) == len(keys)


def test_scored_rungs_exclude_train() -> None:
    """The arms were fitted on ``train``; anything computed there is in-sample."""
    assert "train" not in E.SCORED_RUNGS
    assert set(E.SCORED_RUNGS) == {"calib", "val", "test"}
    assert E.GRADE_RUNG == "test" and E.SELECT_RUNG == "val"


def test_module_constants_are_borrowed_not_retyped() -> None:
    """Path and threshold constants must be the producer's, not a second copy."""
    from tbox_finder.stage2 import train as T

    assert E.ADAPTER_SUBDIR is T.ADAPTER_SUBDIR
    assert E.HEADS_STATE_NAME is T.HEADS_STATE_NAME
    assert E.ECE_N_BINS == M.ECE_N_BINS
    assert math.isclose(E.ECE_GATE, 0.05)


def test_a_float32_nan_cluster_id_is_still_no_cluster() -> None:
    """CodeRabbit r1: the guard was `isinstance(raw, float)`, which np.float32 fails.

    `np.float64` is a subclass of Python `float` and today's column happens to be
    float64, so the old reading worked *by accident of dtype*. Under float32 every
    cluster-less row would key `cluster:nan` and collapse into one block — the exact
    failure `block_keys` is written to prevent, reintroduced by a type check.
    """
    rows = [{"row_id": f"d{i}", "cluster_id": np.float32("nan")} for i in range(6)]
    rows.append({"row_id": "c0", "cluster_id": np.float32(7.0)})
    keys, census = E.block_keys(rows)
    assert census["n_rows_without_cluster"] == 6
    assert census["n_blocks"] == 7, "the float32 NaNs collapsed into one block"
    assert keys[-1] == "cluster:7"


def test_a_non_numeric_cluster_id_is_refused_by_name() -> None:
    """A garbage id is a schema fault, not a silent promotion to 'no cluster'."""
    with pytest.raises(ValueError, match="neither a number nor absent"):
        E.block_keys([{"row_id": "x", "cluster_id": "not-a-cluster"}])
    # Positive control: the identical row with a numeric id is accepted.
    assert E.block_keys([{"row_id": "x", "cluster_id": 3}])[0] == ["cluster:3"]


def test_scaling_that_creates_a_tie_is_detected_not_tolerated() -> None:
    """CodeRabbit r1: exact AP equality could flip the clause on float rounding.

    The invariant is stated directly now — scaling created no ties — with the tie count
    and the AP delta recorded beside it. This drives a case where ``z/T`` genuinely
    collapses two distinct logits onto one float64 value and asserts the report *says
    so* rather than either silently tolerating it or going red without a reason.
    """
    graded = E.grade_arm(_arm(seed=17), n_boot=10)
    block = graded["grades"]["test"]
    assert block["n_ties_created_by_scaling"] == 0
    assert block["auprc_scaling_abs_delta"] == 0.0
    assert block["auprc_rank_invariant"] is True

    # The hazard is real, not hypothetical, and this pair was FOUND rather than assumed:
    # two ADJACENT float64 logits that `z / T` maps onto a single value. It needs T < 1,
    # so the quotient crosses into a coarser binade (here the one above 2.0) — and T < 1
    # is an ordinary fit result: P2-13's Stage-1 temperature was 0.9896.
    z = np.array([1.9999999000000002, 1.9999999000000004], dtype=np.float64)
    assert len(set(z.tolist())) == 2, "the two logits are not distinct to begin with"
    assert len({float(v) for v in z / 0.6}) == 1, (
        "this pair no longer demonstrates a scaling-induced tie, so the clause's "
        "exact-equality hazard would be undocumented"
    )


def test_recorded_paths_are_repo_relative_not_developer_absolute() -> None:
    """CodeRabbit r1: the committed artifact leaked the OS user name and local layout.

    This is a **public** repo and reports are committed, so an absolute path is
    permanent history that resolves on exactly one machine. The sha256 beside each path
    is the identity evidence; the string is only a locator
    ([[committing-real-tool-output-fixtures]]).
    """
    inside = REPO_ROOT / "src" / "tbox_finder" / "stage2" / "eval.py"
    assert E.repo_relative(inside) == "src/tbox_finder/stage2/eval.py"
    assert E.repo_relative(str(inside)) == "src/tbox_finder/stage2/eval.py"


def test_paths_in_the_main_checkout_relativise_from_a_linked_worktree() -> None:
    """The first fix missed this and left every INPUT path absolute.

    Development happens in a worktree under `.claude/worktrees/`, but the DVC-materialised
    inputs live in the main checkout — a different root. Relativising against only this
    file's own root silently passed them through unchanged, which is precisely the leak
    the fix was for.
    """
    roots = E._candidate_roots()
    assert REPO_ROOT.resolve() in roots
    for root in roots:
        probe = root / "data" / "processed" / "stage2_dataset.parquet"
        assert E.repo_relative(probe) == "data/processed/stage2_dataset.parquet"


def test_a_path_outside_every_root_is_passed_through_not_mangled() -> None:
    """Better an honest absolute path than a `../../..` that resolves nowhere."""
    assert E.repo_relative("/definitely/not/in/this/repo.txt") == "/definitely/not/in/this/repo.txt"


def _absolute_path_strings(node: Any, trail: str = "") -> list[tuple[str, str]]:
    """Every string in a JSON tree that is shaped like an absolute filesystem path.

    Structural rather than a list of substrings to grep for (r3): `/home/` is Linux
    only, so a contributor regenerating the artifact on macOS would leak `/Users/...`
    and on Windows `C:\\Users\\...`, and both would sail past a `/home/` check. What
    makes a path a leak is that it is *absolute*, not which OS produced it.
    """
    hits: list[tuple[str, str]] = []
    if isinstance(node, str):
        looks_absolute = node.startswith(("/", "\\\\")) or bool(
            re.match(r"^[A-Za-z]:[\\\\/]", node)
        )
        if looks_absolute:
            hits.append((trail, node))
    elif isinstance(node, dict):
        for key, value in node.items():
            hits.extend(_absolute_path_strings(value, f"{trail}.{key}"))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits.extend(_absolute_path_strings(value, f"{trail}[{i}]"))
    return hits


def test_the_committed_artifacts_carry_no_absolute_paths_at_all() -> None:
    """The end-to-end assertion: walk the shipped artifacts, not just the helper.

    Checks every string value for absoluteness rather than grepping for one OS's home
    prefix, so the guard does not silently weaken the first time somebody regenerates
    these files on a different platform.
    """
    for name in (E.DEFAULT_REPORT, E.DEFAULT_SCORES):
        path = REPO_ROOT / name
        if not path.is_file():
            pytest.skip(f"{name} has not been produced yet")
        payload = json.loads(path.read_text(encoding="utf-8"))
        hits = _absolute_path_strings(payload)
        assert hits == [], f"{name} embeds absolute path(s): {hits[:5]}"


def test_the_absolute_path_detector_is_not_vacuous() -> None:
    """A detector that finds nothing is indistinguishable from one that looks nowhere."""
    planted = {"a": {"b": "/home/someone/x"}, "c": ["ok", "/Users/someone/y", "C:\\Users\\z"]}
    trails = {t for t, _ in _absolute_path_strings(planted)}
    assert trails == {".a.b", ".c[1]", ".c[2]"}
    assert _absolute_path_strings({"rel": "data/processed/x.parquet", "n": 3}) == []


def test_ranking_preserved_catches_a_tie_that_leaves_average_precision_unmoved() -> None:
    """The case `grade_arm` alone cannot reach — and the one an AP-only check misses.

    Sabotaging the clause back to `auprc == auprc_scaled` left the whole suite green,
    which meant the tie half of the fix was untested. It matters because two logits can
    collide under `z / T` *without* moving AP at all (here the colliding rows share a
    label), and an AP-only check calls that "ranking preserved" when ranking information
    was in fact destroyed.
    """
    # A found collision: these two adjacent float64 values divide onto one under T=0.6.
    a, b = 1.9999999000000002, 1.9999999000000004
    z_raw = np.array([a, b, -1.0], dtype=np.float64)
    z_scaled = z_raw / 0.6
    assert len(set(z_raw.tolist())) == 3
    assert len(set(z_scaled.tolist())) == 2, "the fixture no longer collides"

    # Same AP either way (the collided pair shares a label), so AP alone sees nothing.
    ap = M.average_precision([1, 1, 0], [float(v) for v in z_raw])
    ap_scaled = M.average_precision([1, 1, 0], [float(v) for v in z_scaled])
    assert ap == ap_scaled

    preserved, n_ties, ap_delta = E.ranking_preserved(z_raw, z_scaled, ap, ap_scaled)
    assert n_ties == 1
    assert ap_delta == 0.0
    assert preserved is False, "an AP-only check would have called this preserved"

    # Positive control: no collision -> preserved.
    clean = np.array([3.0, 1.0, -1.0], dtype=np.float64)
    ap_c = M.average_precision([1, 1, 0], [float(v) for v in clean])
    ap_cs = M.average_precision([1, 1, 0], [float(v) for v in clean / 1.14])
    assert E.ranking_preserved(clean, clean / 1.14, ap_c, ap_cs) == (True, 0, 0.0)


def test_the_absolute_reading_answers_when_only_the_CONTROL_lacks_an_ece() -> None:
    """The shipped run's exact configuration — and the r1 finding that fixed it.

    D16's absolute reading asks only whether the with-aux arm still holds the D11
    grade, so it is answerable whenever *that* arm has an ECE. Gating it on the
    comparison's availability returned `None` for a run that has a definite verdict.
    Sabotaging the fix left the suite green until this test existed, because every
    other unfittable-arm case put the missing ECE on the with-aux side.
    """
    with_aux = E.grade_arm(_separated_arm(flip_one_calib_row=True), n_boot=10)
    no_aux = E.grade_arm(_separated_arm(flip_one_calib_row=False), n_boot=10)
    with_aux["arm"], no_aux["arm"] = "aux1.0_lr1e-4", "aux0.0_lr1e-4"
    out = E.compare_arms(with_aux, no_aux)

    assert out["ece_comparison_available"] is False, "the control must be the unfittable one"
    assert out["delta_ece"] is None
    # ...and yet the absolute reading HAS an answer, because it never needed the control.
    assert out["reading_absolute"]["passes"] is True
    assert out["reading_absolute"]["observed_ece"] == with_aux["grades"]["test"]["ece"]
    assert out["reading_absolute"]["unavailable_reason"] is None
    assert out["reading_absolute"]["depends_only_on_the_with_aux_arm"] is True


# --------------------------------------------------------------------------- #
# reconcile_cached_scores — the shortcut that must not silently re-grade
# --------------------------------------------------------------------------- #
def _cache_and_rows() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows = [
        {"row_id": f"r{i}", "is_tbox": i % 2 == 0, "_rung": "test", "_block": f"cluster:{i // 2}"}
        for i in range(6)
    ]
    cache = {
        "dataset_sha256": "abc123",
        "row_ids": [r["row_id"] for r in rows],
        "device": "cuda",
        "batch_size": 4,
        "arms": {"a": {"logits": [1.0, -1.0, 2.0, -2.0, 0.5, -0.5], "load": {"n": 1}}},
    }
    return cache, rows, {"sha256": "abc123"}


def test_a_matching_cache_reconciles() -> None:
    """Positive control for every refusal below."""
    cache, rows, meta = _cache_and_rows()
    scores, loads, scoring = E.reconcile_cached_scores(
        cache, rows=rows, dataset_meta=meta, wanted=["a"]
    )
    assert scores["a"].row_ids == tuple(r["row_id"] for r in rows)
    assert loads["a"] == {"n": 1}
    assert scoring["batch_size"] == 4 and scoring["regraded_from_cached_scores"] is True


def test_the_recorded_batch_size_is_the_cached_run_s_not_this_invocation_s() -> None:
    """CodeRabbit r2: batch composition perturbs the bf16 reductions.

    Recording the flag this process was invoked with, rather than the batching that
    actually produced the logits, documents a run that never happened. The failure is
    invisible in the numbers and permanent in the artifact.
    """
    cache, rows, meta = _cache_and_rows()
    cache["batch_size"] = 16
    _, _, scoring = E.reconcile_cached_scores(cache, rows=rows, dataset_meta=meta, wanted=["a"])
    assert scoring["batch_size"] == 16


def test_a_cache_from_a_different_dataset_is_refused() -> None:
    cache, rows, meta = _cache_and_rows()
    cache["dataset_sha256"] = "a-different-digest"
    with pytest.raises(ValueError, match="dataset sha256"):
        E.reconcile_cached_scores(cache, rows=rows, dataset_meta=meta, wanted=["a"])


def test_a_cache_over_a_different_row_set_is_refused() -> None:
    """Same count, different rows: the shortcut's most dangerous near-miss."""
    cache, rows, meta = _cache_and_rows()
    cache["row_ids"] = list(reversed(cache["row_ids"]))
    with pytest.raises(ValueError, match="not the same rows in the same order"):
        E.reconcile_cached_scores(cache, rows=rows, dataset_meta=meta, wanted=["a"])


def test_a_cache_missing_a_requested_arm_is_refused() -> None:
    cache, rows, meta = _cache_and_rows()
    with pytest.raises(KeyError, match="nothing for arm"):
        E.reconcile_cached_scores(cache, rows=rows, dataset_meta=meta, wanted=["a", "b"])


def test_validate_report_is_self_consistency_not_a_pass() -> None:
    """CodeRabbit r2: the docstring claimed "and passing", which the shipped run refutes.

    A report whose gate is honestly false is fully valid; conflating validity with
    passing would make a truthful failing artifact look malformed.
    """
    report = _report()
    report["arms"]["aux1.0_lr1e-4"]["load"]["n_lora_b_nonzero"] = 0
    report["gate"]["clauses"] = E.derive_clauses(report)
    report["gate"]["overall_pass"] = all(report["gate"]["clauses"].values())
    assert report["gate"]["overall_pass"] is False
    assert E.validate_report(report) == [], "a truthfully-failing report must still validate"
    assert "self-consistent" in (E.validate_report.__doc__ or "")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "every one of the 1089 positions is already its own arg-max",
            "perfect_separation_beta_to_infinity",
        ),
        ("the 'calib' rung is single-class (8 positive of 8)", "single_class_calib_rung"),
        ("no rows on the 'calib' rung: census ...", "empty_calib_rung"),
        ("zero rows carry the 'calib' flag", "empty_calib_rung"),
        # r4: `and` binds tighter than `or`, so the unbracketed form labelled ANY
        # "no rows" refusal an empty CALIB rung — including this one, about a graded rung.
        ("no rows on the 'test' rung", "other"),
        ("logits carry a non-finite value", "other"),
    ],
)
def test_refusals_are_classified_by_what_they_actually_say(message: str, expected: str) -> None:
    """The classification names the cause, so a message about another rung must not borrow it."""
    assert E._classify_refusal(ValueError(message)) == expected
