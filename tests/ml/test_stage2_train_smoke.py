"""P3-06 — the Stage-2 LoRA fine-tune entrypoint, in four fail-closed tiers.

1. **pure** — eligibility, config refusal, the LR schedule, DDP sharding, and the report
   clauses + validator. No torch, no hydra, no pandas: this tier runs and **blocks** in bare
   CI, which is where the eligibility rule this step's §7 stop settled has to be guarded.
2. **hydra** (``TBOX_REQUIRE_STAGE2_HYDRA``) — real composition of ``conf/train/stage2.yaml``
   and the exact override tokens the sbatch writes. Job 669 died after a 14 h queue wait on
   an override the config lacked; this tier is what keeps that from recurring here.
3. **torch** (``TBOX_REQUIRE_STAGE2_TORCH``) — the dataset/collator contracts and one real
   end-to-end optimiser step through a tiny same-architecture RiNALMo.
4. **committed report** (``TBOX_REQUIRE_STAGE2_SMOKE``) — validates the run's own report once
   one exists. It is deliberately armed by its own variable so a green bare-CI run never
   implies the cluster run happened.

Each tier gets its **own** environment variable. Folding the torch tier into the pure tier's
variable was the P1-16 landmine: a tier that cannot run must skip loudly, and a tier that is
*required* must fail rather than skip.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pytest

from tbox_finder.models import rna_backbone_registry as BR
from tbox_finder.stage2 import losses as L
from tbox_finder.stage2 import train as T

_REPO = Path(__file__).resolve().parents[2]
_CONF = _REPO / "conf"
_TRAIN_CONF = _CONF / "train" / "stage2.yaml"
_OPTIM_CONF = _CONF / "optim" / "stage2.yaml"
_SBATCH = _REPO / "slurm" / "p3" / "stage2_lora_finetune.sbatch"
#: The P3-17 comparator arm's launcher (ADR-0002 A15 / D6).
_SBATCH_RNAFM = _REPO / "slurm" / "p3" / "rnafm_stage2_finetune.sbatch"
_DATASET = _REPO / T.DEFAULT_DATASET


def _fail_or_skip(var: str, reason: str) -> None:
    if os.environ.get(var) == "1":
        pytest.fail(f"{var}=1 but the tier is unrunnable: {reason}")
    pytest.skip(reason)


def _require_hydra():
    try:
        from hydra import compose, initialize_config_dir  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env-dependent
        _fail_or_skip("TBOX_REQUIRE_STAGE2_HYDRA", f"hydra not importable: {exc}")
    from hydra import compose, initialize_config_dir

    return compose, initialize_config_dir


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env-dependent
        _fail_or_skip("TBOX_REQUIRE_STAGE2_TORCH", f"torch not importable: {exc}")
    import torch

    return torch


def _require_stack():
    """The pinned ml-rna stack (torch + peft + multimolecule), or skip loudly."""
    if os.environ.get("CUDA_HOME") is None:
        _fail_or_skip("TBOX_REQUIRE_STAGE2_TORCH", "CUDA_HOME unset — multimolecule won't import")
    try:
        import multimolecule  # noqa: F401
        import peft  # noqa: F401
        import torch
    except Exception as exc:  # pragma: no cover - env-dependent
        _fail_or_skip("TBOX_REQUIRE_STAGE2_TORCH", f"pinned ml-rna stack unavailable ({exc})")
    return torch


# --------------------------------------------------------------------------------------
# Row fixtures — plain dicts, the exact shape `load_rows` produces
# --------------------------------------------------------------------------------------
def _row(**over: Any) -> dict[str, Any]:
    base = {
        "row_id": "r0",
        "source": "corpus",
        "pool": "corpus",
        "fold_basis": "corpus_record",
        "fold_random": "train",
        "nested_train": True,
        "nested_role": "train",
        "calib": False,
        "is_tbox": True,
        "rna_sequence": "GGCUUAUCAAGAGAGG",
        "label_string": "." * 16,  # `.` is the background class code (labels.CLASS_ORDER)
        "regulatory_mode": "Terminator",
        "specifier_codon": "AUG",
        "cognate_aa": "Met",
        "trna_family": "Met (CAU)",
        "pairing_dotbracket": None,
    }
    base.update(over)
    return base


def _decoy(**over: Any) -> dict[str, Any]:
    """A parentless decoy: no corpus parent, hence no order and a NULL `nested_train`."""
    fields: dict[str, Any] = {
        "row_id": "d0",
        "source": "decoy",
        "pool": "gc_background",
        "fold_basis": "decoy_pool_random",
        "nested_train": None,
        "nested_role": None,
        "is_tbox": False,
        "regulatory_mode": None,
        "specifier_codon": None,
        "cognate_aa": None,
        "trna_family": None,
    }
    # `label_string` is None for ALL 7,007 real decoys — a decoy carries no T-box element
    # annotation. The first version of this fixture inherited a 16-char label_string from
    # `_row()`, which no real decoy has, and that unfaithfulness is exactly what let job
    # 1036 reach the cluster with an unconditional alignment guard.
    fields["label_string"] = None
    fields.update(over)
    return _row(**fields)


# --------------------------------------------------------------------------------------
# TIER 1 — eligibility
# --------------------------------------------------------------------------------------
def test_the_signed_off_rule_admits_exactly_its_two_routes() -> None:
    """D5 ∩ scheme-A-train, minus calib, plus the parentless decoys — and nothing else."""
    assert T.row_eligibility(_row(), rung="train", admit_parentless_decoys=True) == T.ADMITTED
    assert T.row_eligibility(_decoy(), rung="train", admit_parentless_decoys=True) == T.ADMITTED


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (_row(calib=True), T.REFUSE_CALIB),
        (_row(fold_random="test"), T.REFUSE_OTHER_RUNG),
        (_row(fold_random="val"), T.REFUSE_OTHER_RUNG),
        (_row(nested_train=False, nested_role="heldout"), T.REFUSE_LOO_HOLDOUT),
        (_row(nested_train=False, nested_role="excluded_clade_crossing"), T.REFUSE_LOO_HOLDOUT),
        (_row(nested_train=False, nested_role="dropped"), T.REFUSE_LOO_HOLDOUT),
        (_row(nested_train=None, fold_basis="parent_record"), T.REFUSE_UNPLACEABLE),
        (_row(nested_train=None, fold_basis="corpus_record"), T.REFUSE_UNPLACEABLE),
    ],
)
def test_every_other_row_is_refused_by_name(row: dict[str, Any], reason: str) -> None:
    """Fail-closed, and the reason is the refusal — not a silent drop into a bucket."""
    assert T.row_eligibility(row, rung="train", admit_parentless_decoys=True) == reason


def test_a_calib_row_is_refused_even_when_every_other_condition_admits_it() -> None:
    """`calib` is checked FIRST, so no later branch can re-admit a calibration row.

    P3-07 fits its temperature on `calib` and P3-10 grades GATE-2 on the `test` rung; a
    calib row that trained would make the temperature an in-sample fit and the whole
    calibration stack self-referential.
    """
    row = _row(calib=True, nested_train=True, fold_random="train")
    assert T.row_eligibility(row, rung="train", admit_parentless_decoys=True) == T.REFUSE_CALIB


def test_the_parentless_branch_cannot_swallow_a_placed_row() -> None:
    """A row claiming the decoy fold basis but carrying a fold assignment RAISES.

    This is the one way the admit branch could leak a held-out clade: a corpus record whose
    order IS known, admitted through the door built for rows that have no order at all.
    """
    with pytest.raises(ValueError, match="parentless-decoy admit branch"):
        T.row_eligibility(
            _row(fold_basis="decoy_pool_random", nested_train=False),
            rung="train",
            admit_parentless_decoys=True,
        )
    with pytest.raises(ValueError, match="parentless-decoy admit branch"):
        T.row_eligibility(
            _row(fold_basis="decoy_pool_random", nested_train=None, source="corpus"),
            rung="train",
            admit_parentless_decoys=True,
        )
    # Positive control: the identical row WITHOUT the corruption is admitted, so the guard
    # is not simply refusing everything it is handed.
    assert T.row_eligibility(_decoy(), rung="train", admit_parentless_decoys=True) == T.ADMITTED


def test_turning_the_parentless_clause_off_refuses_them_rather_than_reclassifying() -> None:
    assert (
        T.row_eligibility(_decoy(), rung="train", admit_parentless_decoys=False)
        == T.REFUSE_UNPLACEABLE
    )


@pytest.mark.parametrize("spelling", [True, "True", "true", 1])
def test_nested_train_truthiness_survives_the_parquet_round_trip(spelling: Any) -> None:
    """`numpy.bool_`, `"True"` and `1` are all how a parquet hands this back."""
    assert (
        T.row_eligibility(_row(nested_train=spelling), rung="train", admit_parentless_decoys=True)
        == T.ADMITTED
    )


@pytest.mark.parametrize("spelling", [float("nan"), None, "nan", "NaN", ""])
def test_a_missing_nested_train_is_missing_in_every_spelling(spelling: Any) -> None:
    """`str(x or "")` reads None as "" but NaN as "nan" under the ml-rna env's pandas.

    That exact difference once deleted 60% of a training mix with every clause green, so
    missingness goes through `masking.is_missing` and the tri-state is asserted here on all
    the spellings a parquet actually produces.
    """
    row = _row(nested_train=spelling, fold_basis="corpus_record")
    assert (
        T.row_eligibility(row, rung="train", admit_parentless_decoys=True) == T.REFUSE_UNPLACEABLE
    )


def test_the_census_partitions_the_input_and_names_both_admit_routes() -> None:
    rows = [
        _row(row_id="a"),
        _row(row_id="b", calib=True),
        _row(row_id="c", nested_train=False, nested_role="heldout"),
        _row(row_id="d", fold_random="test"),
        _decoy(row_id="e"),
    ]
    admitted, census = T.select_rows(rows, rung="train")
    assert admitted == [0, 4]
    assert census["n_admitted"] + sum(census["refused"].values()) == census["n_rows_scanned"]
    assert census["admitted_by_route"] == {T.ADMIT_NESTED_TRAIN: 1, T.ADMIT_PARENTLESS_DECOY: 1}
    assert census["d5_refusal_roles"] == {"heldout": 1}
    assert (census["n_admitted_positive"], census["n_admitted_negative"]) == (1, 1)


def test_the_val_rung_selects_a_disjoint_population() -> None:
    rows = [
        _row(row_id="a"),
        _row(row_id="b", fold_random="val"),
        _decoy(row_id="c", fold_random="val"),
    ]
    train_idx, _ = T.select_rows(rows, rung="train")
    val_idx, val_census = T.select_rows(rows, rung="val")
    assert set(train_idx) & set(val_idx) == set()
    assert val_census["n_admitted"] == 2


# --------------------------------------------------------------------------------------
# TIER 1 — config refusal
# --------------------------------------------------------------------------------------
def test_the_boundary_crf_is_refused_at_construction() -> None:
    """The P3-03/P3-04 hand-off, decided here — and refused before any GPU time.

    `multitask_loss` already raises on a CRF, but only once a batch reaches it. Refusing in
    `__post_init__` means refusing on the login node rather than after the queue wait.
    """
    with pytest.raises(ValueError, match="boundary_use_crf=True is refused"):
        T.Stage2TrainConfig(boundary_use_crf=True)
    T.Stage2TrainConfig(boundary_use_crf=False)  # positive control


def test_the_structure_head_and_its_term_must_agree() -> None:
    with pytest.raises(ValueError, match="structure_head"):
        T.Stage2TrainConfig(structure_head=True)
    with pytest.raises(ValueError, match="structure_head"):
        T.Stage2TrainConfig(loss=L.Stage2LossConfig(structure_enabled=True))
    T.Stage2TrainConfig(
        structure_head=True, loss=L.Stage2LossConfig(structure_enabled=True)
    )  # positive control


def test_the_two_rungs_cannot_be_the_same() -> None:
    with pytest.raises(ValueError, match="train_rung and val_rung"):
        T.Stage2TrainConfig(train_rung="train", val_rung="train")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"epochs": True},
        {"batch_size": 0},
        {"lr": 0.0},
        {"lr": float("nan")},
        {"lr": -1.0},
        {"dropout": 1.0},
        {"warmup_ratio": 1.0},
        {"max_records": 0},
        {"log_every": 0},
        {"gradient_accumulation_steps": 0},
    ],
)
def test_impossible_config_values_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        T.Stage2TrainConfig(**kwargs)


def test_a_bool_is_not_accepted_as_a_count() -> None:
    """`True` is an int in Python; `epochs=True` would silently train for one epoch."""
    with pytest.raises(ValueError):
        T.Stage2TrainConfig(batch_size=True)


# --------------------------------------------------------------------------------------
# TIER 1 — LR schedule + sharding
# --------------------------------------------------------------------------------------
def test_the_schedule_warms_up_then_decays_to_zero() -> None:
    warmup, total = 10, 100
    values = [T.lr_scale(s, warmup_steps=warmup, total_steps=total) for s in range(total)]
    assert values[0] == pytest.approx(0.1)  # not 0 — a zero first step is a wasted step
    assert values[warmup - 1] == pytest.approx(1.0)
    assert values[:warmup] == sorted(values[:warmup])  # strictly ramping
    assert values[warmup:] == sorted(values[warmup:], reverse=True)  # then decaying
    assert values[-1] < 1e-2
    assert T.lr_scale(total, warmup_steps=warmup, total_steps=total) == pytest.approx(0.0)


def test_the_schedule_is_not_secretly_a_constant() -> None:
    """A scheduler that returns 1.0 everywhere is invisible in a loss curve for a long time."""
    values = {T.lr_scale(s, warmup_steps=5, total_steps=50) for s in range(50)}
    assert len(values) > 40, values


def test_the_schedule_refuses_an_impossible_horizon() -> None:
    for kwargs in ({"warmup_steps": 5, "total_steps": 0}, {"warmup_steps": 51, "total_steps": 50}):
        with pytest.raises(ValueError):
            T.lr_scale(0, **kwargs)
    with pytest.raises(ValueError):
        T.lr_scale(-1, warmup_steps=5, total_steps=50)


def test_the_sampler_is_a_seeded_permutation_that_moves_with_the_epoch() -> None:
    a = T.EpochShuffleSampler(64, seed=42)
    b = T.EpochShuffleSampler(64, seed=42)
    assert list(iter(a)) == list(iter(b))  # reproducible
    assert sorted(iter(a)) == list(range(64))  # a permutation, not a sample
    a.set_epoch(1)
    assert list(iter(a)) != list(iter(b))  # and the epoch really reaches the order
    assert list(iter(a)) == list(iter(a))  # …deterministically within an epoch
    c = T.EpochShuffleSampler(64, seed=43)
    assert list(iter(c)) != list(iter(b))


def test_ddp_shards_are_equal_length_or_the_job_deadlocks() -> None:
    """A ragged shard hangs the job rather than failing it — the worst way to go wrong."""
    from tbox_finder.train.ddp import ShardedSampler

    for world in (1, 2, 3, 4, 8):
        lengths = {
            len(
                list(
                    iter(
                        ShardedSampler(T.EpochShuffleSampler(23, seed=7), rank=r, world_size=world)
                    )
                )
            )
            for r in range(world)
        }
        assert len(lengths) == 1, (world, lengths)


def test_ddp_shards_are_disjoint_and_a_subset_of_the_stream() -> None:
    from tbox_finder.train.ddp import ShardedSampler

    inner = T.EpochShuffleSampler(23, seed=7)
    union: list[int] = []
    for r in range(4):
        union.extend(iter(ShardedSampler(T.EpochShuffleSampler(23, seed=7), rank=r, world_size=4)))
    assert len(union) == len(set(union))  # no draw twice in an epoch
    assert set(union) <= set(iter(inner))  # a SUBSET — asserting equality would require
    assert len(union) == 20  # the ragged shards that deadlock


# --------------------------------------------------------------------------------------
# TIER 1 — the report gate
# --------------------------------------------------------------------------------------
def _passing_report() -> dict[str, Any]:
    cfg = T.Stage2TrainConfig()
    weights = cfg.loss.effective_weights()
    return T.build_report(
        cfg,
        data={
            "dataset_parquet": T.DEFAULT_DATASET,
            "n_dataset_rows": 30542,
            "n_train_rows": 9059,
            "n_val_rows": 2070,
            "train_census": {
                "rung": "train",
                "n_rows_scanned": 30542,
                "n_admitted": 9059,
                "admitted_by_route": {T.ADMIT_NESTED_TRAIN: 5199, T.ADMIT_PARENTLESS_DECOY: 3860},
                "refused": {
                    T.REFUSE_CALIB: 1089,
                    T.REFUSE_OTHER_RUNG: 6112,
                    T.REFUSE_LOO_HOLDOUT: 14282,
                    T.REFUSE_UNPLACEABLE: 0,
                },
                "d5_refusal_roles": {"heldout": 7718},
                "n_admitted_positive": 4795,
                "n_admitted_negative": 4264,
            },
            "val_census": {"n_admitted": 2070},
            "n_train_val_row_id_overlap": 0,
        },
        steps={
            # Mirrors the shipped shape exactly (train_stage2's `steps` block). A fixture that
            # drifts from it lets a clause pass here and fail on a real report — which is how
            # the scheduler-domain regression stayed invisible to the gate.
            "n_steps": 2830,
            "expected_n_steps": 2830,
            "n_optimizer_steps": 1415,
            "expected_n_optimizer_steps": 1415,
            "batches_per_epoch_per_rank": 283,
            "optimizer_steps_per_epoch_per_rank": 142,
            "gradient_accumulation_steps": 2,
            "world_size": 8,
            "warmup_steps": 85,
            "total_scheduled_optimizer_steps": 1415,
            "elapsed_seconds": 2100.0,
        },
        wrap={
            # Built from the registry rather than hand-typed, so the fixture cannot drift from
            # the pins the clause re-derives against — a hand-copied revision that went stale
            # would make `backbone_pinned` fail here for a reason that is not a defect, and
            # tempt the next reader to weaken the clause instead of the fixture.
            "backbone": {
                **BR.backbone_summary(BR.resolve_backbone(BR.PRODUCTION_BACKBONE)),
                "loaded_from_registry": True,
            },
            "base_frozen": True,
            "n_base_trainable_params": 0,
            "n_adapter_sites": 231,
            "gradient_checkpointing": True,
            "n_modules_with_checkpointing": 33,
            "checkpoint_use_reentrant": False,
            "applied_lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "all_linear_fully_covered": True,
            },
            "stage2_heads": {
                # MEASURED off the live module by `build_stage2_model` — the independent
                # evidence `backbone_pinned` cross-checks the recorded identity against.
                "d_model": BR.resolve_backbone(BR.PRODUCTION_BACKBONE).hidden_size,
                "heads_outside_peft_wrapper": True,
                "all_heads_trainable": True,
                "n_head_parameters": 199836,
            },
        },
        objective={
            "active_terms": list(cfg.loss.active_terms()),
            "terms_seen": list(cfg.loss.active_terms()),
            "dominance": cfg.loss.dominance(),
            "weighting": "fixed",
            "effective_weights": weights,
            "max_aux_weight": cfg.loss.max_aux_weight,
        },
        losses={
            "final_train_total": 0.31,
            "per_term_final": {"binary": 0.1, "boundary": 0.2},
            "n_nonfinite_steps": 0,
            "n_nonfinite_grad_steps": 0,
        },
        val={"history": [], "best_total": 0.4, "best_epoch": 7},
        checkpoint={
            "adapter_dir": "data/processed/checkpoints/stage2_rinalmo/lora_adapter",
            "adapter_bytes": 51_000_000,
            "heads_bytes": 800_000,
            "n_saved_parameters": 13_040_268,
            "saved_from_epoch": 9,
            "saved_val_total": 0.0088,
            "best_val_total_observed_during_training": 0.00038,
            "best_val_epoch_observed_during_training": 6,
        },
        device={
            "is_cuda": True,
            "name": "NVIDIA RTX A4000",
            "driver_version": "590.48.01",
            "capability": [8, 6],
            "n_visible_devices": 8,
        },
        git_snapshot={"git_sha": "deadbeef", "git_dirty": False, "git_dirty_paths": []},
    )


def test_a_valid_report_passes_and_every_clause_holds() -> None:
    report = _passing_report()
    assert T.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True
    assert all(report["gate"]["clauses"].values()), report["gate"]["failed"]


def test_the_validator_catches_a_clause_fabricated_true() -> None:
    """`all(clauses)` catches a clause flipped FALSE but never one fabricated TRUE."""
    report = _passing_report()
    report["data"]["train_census"]["refused"][T.REFUSE_LOO_HOLDOUT] = 0
    report["gate"]["clauses"]["no_d5_holdout_trained"] = True  # the lie
    problems = T.validate_report(report)
    assert any("no_d5_holdout_trained" in p for p in problems), problems


@pytest.mark.parametrize(
    ("mutate", "clause"),
    [
        (
            lambda r: r["data"]["train_census"]["refused"].__setitem__(T.REFUSE_LOO_HOLDOUT, 0),
            "no_d5_holdout_trained",
        ),
        (
            lambda r: r["data"]["train_census"]["refused"].__setitem__(T.REFUSE_CALIB, 0),
            "calib_refused",
        ),
        (
            lambda r: r["data"]["train_census"]["admitted_by_route"].__setitem__(
                T.ADMIT_PARENTLESS_DECOY, 0
            ),
            "both_admit_routes_fired",
        ),
        (lambda r: r["data"].__setitem__("n_train_rows", 8), "full_population"),
        (
            lambda r: r["data"].__setitem__("n_train_val_row_id_overlap", 3),
            "val_disjoint_from_train",
        ),
        (lambda r: r["steps"].__setitem__("n_steps", 12), "steps_ran"),
        (lambda r: r["losses"].__setitem__("n_nonfinite_steps", 1), "losses_finite"),
        (lambda r: r["losses"].__setitem__("final_train_total", None), "losses_finite"),
        (lambda r: r["wrap"]["applied_lora"].__setitem__("r", 8), "lora_contract_held"),
        (lambda r: r["wrap"].__setitem__("base_frozen", False), "lora_contract_held"),
        (
            lambda r: r["wrap"]["stage2_heads"].__setitem__("heads_outside_peft_wrapper", False),
            "heads_outside_wrapper",
        ),
        (
            lambda r: r["wrap"].__setitem__("n_modules_with_checkpointing", 0),
            "gradient_checkpointing_flag_consistent",
        ),
        (
            lambda r: r["wrap"].__setitem__("checkpoint_use_reentrant", True),
            "gradient_checkpointing_flag_consistent",
        ),
        (lambda r: r["objective"].__setitem__("terms_seen", ["binary"]), "objective_terms_match"),
        (
            lambda r: r["objective"]["effective_weights"].__setitem__("boundary", 9.0),
            "objective_terms_match",
        ),
        (lambda r: r["checkpoint"].__setitem__("adapter_bytes", 0), "checkpoint_written"),
        (lambda r: r["provenance"].__setitem__("git_sha", None), "provenance_recorded"),
        (lambda r: r["provenance"].__setitem__("env_lock_sha256", None), "provenance_recorded"),
    ],
)
def test_each_clause_bites_on_its_own_evidence(mutate: Any, clause: str) -> None:
    """Every clause is re-derived from recorded evidence — so corrupt the evidence, not the flag."""
    report = _passing_report()
    mutate(report)
    clauses = T.derive_clauses(report)
    assert clauses[clause] is False, (clause, clauses)


def test_the_no_aux_arm_passes_the_gate_rather_than_failing_it_by_construction() -> None:
    """`aux_weight=0` is a sweep POINT, not a broken run — two of six arms depend on it.

    A clause that demanded `terms_seen == active_terms` would fail the arm P3-08 exists to
    consume, on every one of its steps, for the whole run.
    """
    report = _passing_report()
    cfg = T.Stage2TrainConfig(loss=L.Stage2LossConfig(aux_weight=0.0))
    report["objective"]["effective_weights"] = cfg.loss.effective_weights()
    report["objective"]["terms_seen"] = [L.BINARY_TERM]
    assert T.derive_clauses(report)["objective_terms_match"] is True
    # …and it still bites if a term that SHOULD have contributed never did.
    report["objective"]["terms_seen"] = []
    assert T.derive_clauses(report)["objective_terms_match"] is False


def test_the_truncation_clause_is_the_one_that_catches_a_cost_knob() -> None:
    """`--max-records 8` leaves every rule-shaped clause green over a handful of rows.

    So completeness is asserted beside correctness: the population clauses still pass on a
    truncated run, and only `full_population` refuses it.
    """
    report = _passing_report()
    report["data"]["n_train_rows"] = 8
    clauses = T.derive_clauses(report)
    assert clauses["no_d5_holdout_trained"] is True
    assert clauses["calib_refused"] is True
    assert clauses["full_population"] is False


def test_a_must_fire_clause_refuses_a_filter_that_refused_nothing() -> None:
    """Zeroed refusal counters read exactly like a filter whose join matched nothing."""
    report = _passing_report()
    census = report["data"]["train_census"]
    census["refused"] = dict.fromkeys(T.REFUSAL_REASONS, 0)
    census["n_rows_scanned"] = census["n_admitted"]
    clauses = T.derive_clauses(report)
    assert clauses["no_d5_holdout_trained"] is False
    assert clauses["calib_refused"] is False


def test_the_validator_rejects_a_wrong_schema_or_step() -> None:
    report = _passing_report()
    report["schema_version"] = "0"
    report["step"] = "P9-99"
    problems = T.validate_report(report)
    assert any("schema_version" in p for p in problems)
    assert any("step" in p for p in problems)


def test_the_gate_is_not_recorded_from_a_requested_setting() -> None:
    """`derive_clauses` must not read the config block — only measured evidence.

    A clause sourced from `report["config"]` restates the request and can never fail.
    """
    report = _passing_report()
    report["config"] = {}
    assert T.derive_clauses(report) == _passing_report()["gate"]["clauses"]


# --------------------------------------------------------------------------------------
# TIER 1 — the shipped sbatch, read as text
# --------------------------------------------------------------------------------------
def test_the_sbatch_requests_a_whole_gpu_node_and_pins_nothing() -> None:
    text = _SBATCH.read_text()
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:a4000:8" in text
    # Unserialised and unrestricted: both gpu nodes are healthy again, so the `%1` +
    # `--exclude=two` pair that guarded against the broken one is spent. The caveat that
    # replaced it — points may land on either driver — is recorded in the run's own report,
    # which is asserted separately.
    assert "#SBATCH --array=0-5\n" in text
    # The health check must run BEFORE the 2.5 GB HF download, or a bad node costs
    # minutes instead of seconds (job 1036 lost three points that way).
    assert text.index("STAGE2_NODE_UNHEALTHY") < text.index("hf cache warm")
    # Scan the DIRECTIVES, not the prose: the header explains at length why `--nodelist` is
    # never used, and a whole-file substring search would read that explanation as the flag.
    directives = [ln for ln in text.splitlines() if ln.startswith("#SBATCH ")]
    assert directives
    for forbidden in ("--nodelist", "--account", "--qos", "--partition=compute"):
        assert not any(forbidden in ln for ln in directives), forbidden
    # `--exclude=two` is a DATED, temporary carve-out for a measured node fault (job 1036).
    # If it is present it must carry its removal condition, so it cannot quietly outlive the
    # reboot that fixes the node and silently halve the cluster.
    # Any node carve-out must carry BOTH a date and a stated removal condition, so it cannot
    # outlive its reason — the first one cited a node fault that has since been fixed, and the
    # file would have gone on asserting it. None is present now; this guards a re-addition.
    if any("--exclude" in ln for ln in directives):
        assert "REMOVE THIS" in text
        assert "2026-" in text
    # The driver heterogeneity accepted in exchange must stay written down.
    assert "ACCEPTED CAVEAT" in text
    assert "590.48.01" in text and "595.84" in text


def test_the_sbatch_activates_the_rna_env_not_the_dna_one() -> None:
    """ADR-0002 A4: multimolecule needs transformers 5.x; Caduceus caps it at 4.57.5."""
    text = _SBATCH.read_text()
    assert "conda activate tbox-ml-rna" in text
    assert "tbox-ml-dna" not in text
    assert 'export CUDA_HOME="$CONDA_PREFIX"' in text
    assert "export PYTHONHASHSEED=0" in text
    assert "export WANDB_MODE=offline" in text


def test_the_sbatch_scratch_var_is_not_named_BUILD() -> None:
    """`conda activate` exports BUILD=x86_64-conda-linux-gnu and killed job 789 with it."""
    text = _SBATCH.read_text()
    assert "JOB_SCRATCH=" in text
    assert "\nBUILD=" not in text


def test_every_inline_python_block_in_every_sbatch_actually_compiles() -> None:
    """`python -c '...'` bodies are Python, and nothing was checking that they parse.

    Found the hard way at P3-06: the first draft of this step's HF-warm probe wrote
    ``f"attn={getattr(cfg, \\"_attn\\", None)}"`` inside a single-quoted shell string, so the
    backslash survived into the Python source and the block was a SyntaxError. It would have
    died seconds into the job — after the queue wait — which is job 789's failure shape
    exactly. Repo-wide rather than scoped to this file's sbatch, because the defect is a
    property of the quoting pattern and not of this step.
    """
    import re

    # ⚠ WIDENED at P3-17. The original pattern anchored on `^PYTHONPATH=… python -c '` at line
    # start and required a closing line of exactly `'`, so it silently skipped every block
    # written as `if ! PYTHONPATH=src python -c '…'; then` — the guard-style invocation. Both
    # of P3-17's env/CUDA assertions use that form, and so does P3-06's own node health check,
    # which means this "repo-wide" gate had been covering less than it claimed since it was
    # written. Its emptiness guard could not notice: other blocks matched, so it stayed green.
    # Any leading command prefix is allowed now, and the terminator is a line that STARTS with
    # the closing quote.
    patterns = (
        r"^[^\n]*?\bpython -c '\n(.*?)\n'(?:;|\s|$)",
        # The ONE-LINER form, `python -c 'import x; print(y)'` — a third shape, found by the
        # coverage guard below the moment it was added (slurm/p2/backbone_throughput.sbatch's
        # selective_scan_cuda probe). Its body has no newline, so neither multi-line pattern
        # reaches it, and it had never been parsed by this gate either.
        r"python -c '([^'\n]+)'",
        # Heredoc form too — `python - <<'PY' … PY`. The provenance writer lives in one
        # of these, and an unparsed heredoc is exactly as dead as an unparsed -c block.
        r"python - <<'PY'\n(.*?)\nPY$",
    )
    sbatches = sorted((_REPO / "slurm").rglob("*.sbatch"))
    blocks = [
        (path, match.group(1))
        for path in sbatches
        for pattern in patterns
        for match in re.finditer(pattern, path.read_text(), re.S | re.M)
    ]
    # Emptiness guard: a regex that matched nothing would make this vacuously green.
    assert blocks, "no inline `python -c` blocks discovered in any sbatch at all"
    # COVERAGE guard, which the emptiness guard alone does not give: every sbatch that
    # contains an inline python invocation must yield at least one block. This is what would
    # have caught the narrow pattern above, and it is the same shape as
    # `test_no_sbatch_launch_is_silently_dropped` in the override gate.
    covered = {path for path, _ in blocks}
    expected = {
        p for p in sbatches if "python -c '" in p.read_text() or "python - <<'PY'" in p.read_text()
    }
    uncovered = {p.relative_to(_REPO) for p in expected - covered}
    assert (
        not uncovered
    ), f"sbatch files carry inline python that this gate never parses: {uncovered}"
    broken = []
    for path, body in blocks:
        try:
            compile(body, str(path), "exec")
        except SyntaxError as exc:
            broken.append(f"{path.relative_to(_REPO)}: line {exc.lineno}: {exc.msg}")
    assert not broken, broken


def test_the_sweep_grid_is_the_six_points_the_header_claims() -> None:
    text = _SBATCH.read_text()
    assert "AUX=(0.0  0.0  0.5  0.5  1.0  1.0)" in text
    assert "LRS=(1e-4 3e-4 1e-4 3e-4 1e-4 3e-4)" in text
    # The no-aux arm P3-08 consumes must exist, and the LR must go through the group.
    assert 'loss.aux_weight="$AUX_WEIGHT"' in text
    assert 'optim.lr="$LR"' in text
    # The batch size is the one reports/p3/stage2_sizing.json measured as the largest
    # WORST-CASE fit; batch 8 OOM'd there and in job 1044. If this ever reads 8 again it
    # must be because a fresh measurement said so.
    assert "batch_size=4" in text
    # Claiming checkpointing on this backbone would be claiming a no-op (saving ratio 0.9986).
    assert "gradient_checkpointing=false" in text
    assert " lr=" not in text.split("-m tbox_finder.stage2.train")[1].split("\n")[1]


# --------------------------------------------------------------------------------------
# TIER 2 — Hydra composition
# --------------------------------------------------------------------------------------
def test_the_primary_config_exists_and_declares_its_groups() -> None:
    text = _TRAIN_CONF.read_text()
    assert text.startswith("# @package _global_")
    for group in ("/model: rinalmo_stage2", "/loss: stage2", "/optim: stage2", "/tracking: wandb"):
        assert group in text, group
    assert _OPTIM_CONF.read_text().startswith("# @package optim")


def test_hydra_composes_and_reaches_every_group() -> None:
    compose, initialize_config_dir = _require_hydra()
    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(config_name="train/stage2")
    assert cfg.seed == 42
    assert cfg.optim.lr == pytest.approx(1e-4)
    assert cfg.loss.aux_weight == pytest.approx(1.0)
    assert cfg.model.repo_id == "multimolecule/rinalmo-giga"
    assert cfg.tracking.mode == "offline"


def test_every_override_the_sbatch_writes_actually_composes() -> None:
    """Struct mode: an override against a key the YAML lacks is a hard failure.

    That is how job 669 was lost after a 14 h queue wait, so the exact tokens this step's
    sbatch writes are composed here rather than trusted.
    """
    compose, initialize_config_dir = _require_hydra()
    overrides = [
        "loss.aux_weight=0.0",
        "optim.lr=3e-4",
        "epochs=10",
        "batch_size=8",
        "gradient_checkpointing=true",
        "eval_val=true",
        "save_checkpoint=true",
        "report_path=reports/p3/sweep/x.json",
        "checkpoint_dir=/tmp/x",
        "tracking.mode=offline",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(config_name="train/stage2", overrides=overrides)
    assert cfg.loss.aux_weight == pytest.approx(0.0)
    assert cfg.optim.lr == pytest.approx(3e-4)


def test_the_overrides_reach_the_dataclass_and_the_group_wins_lr() -> None:
    compose, initialize_config_dir = _require_hydra()
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(
            config_name="train/stage2",
            overrides=["loss.aux_weight=0.0", "optim.lr=3e-4", "epochs=2"],
        )
    built = T._cfg_from_mapping(OmegaConf.to_container(cfg, resolve=True))
    assert built.lr == pytest.approx(3e-4)  # NOT the top-level default
    assert built.loss.aux_weight == pytest.approx(0.0)
    assert built.epochs == 2
    # The no-aux arm P3-08 consumes. `active_terms` still lists all six — a zero-weighted
    # term is present and supervised, merely not counted — so the arm is identified by its
    # EFFECTIVE weights, which is also how the gate's `objective_terms_match` re-derives it.
    weights = built.loss.effective_weights()
    assert weights[L.BINARY_TERM] > 0
    assert all(weights[term] == 0.0 for term in L.AUX_TERMS)


def test_a_mistyped_loss_key_is_refused_rather_than_ignored() -> None:
    """A stray `loss.aux_wieght` would train the default and report the value it was handed."""
    with pytest.raises(ValueError, match="Stage2LossConfig does not define"):
        T._cfg_from_mapping({"loss": {"aux_wieght": 0.0}})


def test_a_model_group_that_drifts_from_the_pinned_checkpoint_is_refused() -> None:
    """`model.revision=<x>` is an override that changes nothing — so it must not be silent."""
    with pytest.raises(ValueError, match="frozen checkpoint identity"):
        T._cfg_from_mapping({"model": {"revision": "main"}})
    T._cfg_from_mapping({"model": {}})  # positive control


# --------------------------------------------------------------------------------------
# TIER 3 — torch: dataset, collator, and one real step
# --------------------------------------------------------------------------------------
def test_a_dataset_item_carries_one_target_per_active_term() -> None:
    _require_torch()
    from tbox_finder.stage2 import heads as H

    spec = H.load_head_spec()
    cfg = L.Stage2LossConfig()
    ds = T.Stage2SequenceDataset([_row()], spec, loss_config=cfg)
    item = ds[0]
    assert set(cfg.active_terms()) <= set(item)
    assert len(item["input_ids"]) == 16 + 2  # cls + 16 nt + eos
    assert len(item["boundary"]) == 16


def test_a_label_string_of_the_wrong_length_raises_rather_than_misaligning() -> None:
    _require_torch()
    from tbox_finder.stage2 import heads as H

    ds = T.Stage2SequenceDataset(
        [_row(label_string="." * 15)], H.load_head_spec(), loss_config=L.Stage2LossConfig()
    )
    with pytest.raises(ValueError, match="nucleotides"):
        _ = ds[0]


def test_the_collator_pads_targets_to_the_nucleotide_axis_the_model_produces() -> None:
    """`(B, T-2)`, because `strip_special_tokens` drops cls and each row's own eos."""
    torch = _require_torch()
    from tbox_finder.stage2 import heads as H

    rows = [_row(row_id="a"), _row(row_id="b", rna_sequence="GGCUUAUC", label_string="." * 8)]
    ds = T.Stage2SequenceDataset(rows, H.load_head_spec(), loss_config=L.Stage2LossConfig())
    batch = T.collate_stage2([ds[0], ds[1]])
    width = batch["input_ids"].shape[1]
    assert width == 18
    assert batch["targets"]["boundary"].shape == (2, width - 2)
    # The short row's padding is the IGNORE sentinel, not class 0 (which is a real class).
    assert int(batch["targets"]["boundary"][1, -1]) == H.IGNORE_INDEX
    assert int(batch["attention_mask"][1].sum()) == 10
    from tbox_finder.stage2 import tokenizer as tok

    assert int(batch["input_ids"][1, -1]) == tok.PAD_ID
    assert torch.is_tensor(batch["input_ids"])


def test_the_collator_refuses_an_empty_batch() -> None:
    _require_torch()
    with pytest.raises(ValueError, match="empty batch"):
        T.collate_stage2([])


def test_a_decoy_supervises_the_binary_term_and_ignores_the_record_level_ones() -> None:
    """7,007 decoy rows carry the sentinel everywhere but the binary and boundary terms."""
    _require_torch()
    from tbox_finder.stage2 import heads as H

    ds = T.Stage2SequenceDataset([_decoy()], H.load_head_spec(), loss_config=L.Stage2LossConfig())
    item = ds[0]
    assert item[L.BINARY_TERM] == 0
    for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family"):
        assert item[term] == H.IGNORE_INDEX, term


def _tiny_backbone():
    from multimolecule import RiNALMoConfig, RiNALMoModel

    cfg = RiNALMoConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128
    )
    cfg._attn_implementation = "sdpa"  # CPU tier; FA-2 needs a GPU
    return RiNALMoModel(cfg, add_pooling_layer=False)


def test_one_step_runs_end_to_end_and_moves_only_the_adapters_and_heads() -> None:
    """The whole composition — model, objective, backward — on a tiny same-architecture model."""
    torch = _require_stack()
    cfg = T.Stage2TrainConfig(batch_size=2, epochs=1, gradient_checkpointing=False)
    model, wrap = T.build_model(cfg, base_model=_tiny_backbone())
    assert wrap["base_frozen"] is True
    assert wrap["stage2_heads"]["heads_outside_peft_wrapper"] is True

    from tbox_finder.stage2 import heads as H

    ds = T.Stage2SequenceDataset(
        [_row(row_id="a"), _decoy(row_id="b")], H.load_head_spec(), loss_config=cfg.loss
    )
    batch = T.collate_stage2([ds[0], ds[1]])
    loss_fn = L.MultitaskLoss(cfg.loss)
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    loss, components = loss_fn(outputs, batch["targets"])
    assert torch.isfinite(loss)
    loss.backward()

    moved = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    assert moved, "no trainable parameter received a gradient"
    frozen_with_grad = [
        n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None
    ]
    assert not frozen_with_grad, frozen_with_grad
    assert L.BINARY_TERM in components["included"]


def test_the_no_aux_arm_leaves_the_aux_heads_without_a_gradient() -> None:
    """Which is exactly why DDP is constructed with `find_unused_parameters=True`.

    If this ever stopped being true the flag would be dead weight; if the flag were dropped
    while it stayed true, the no-aux arm would abort in the backward pass on every rank.
    """
    _require_stack()
    cfg = T.Stage2TrainConfig(gradient_checkpointing=False, loss=L.Stage2LossConfig(aux_weight=0.0))
    model, _ = T.build_model(cfg, base_model=_tiny_backbone())
    from tbox_finder.stage2 import heads as H

    ds = T.Stage2SequenceDataset([_row()], H.load_head_spec(), loss_config=cfg.loss)
    batch = T.collate_stage2([ds[0]])
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    loss, _ = L.MultitaskLoss(cfg.loss)(outputs, batch["targets"])
    loss.backward()
    assert all(p.grad is None for p in model.regulatory_mode_head.parameters())
    assert any(p.grad is not None for p in model.tbox_head.parameters())


def test_gradient_checkpointing_is_verified_not_assumed() -> None:
    """A flag that no-ops looks exactly like one that works — so the count is measured."""
    _require_stack()
    cfg = T.Stage2TrainConfig(gradient_checkpointing=True)
    _, wrap = T.build_model(cfg, base_model=_tiny_backbone())
    assert wrap["n_modules_with_checkpointing"] > 0
    assert wrap["checkpoint_use_reentrant"] is False


# --------------------------------------------------------------------------------------
# TIER 3b — the real committed dataset (local / cluster only)
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_STAGE2_DATA") != "1",
    reason="reads the DVC-tracked Stage-2 dataset; local/cluster only",
)
def test_the_real_corpus_yields_a_population_with_both_routes_and_both_classes() -> None:
    rows = T.load_rows(_DATASET)
    _, census = T.select_rows(rows, rung="train")
    assert census["n_admitted"] > 0
    assert census["admitted_by_route"][T.ADMIT_NESTED_TRAIN] > 0
    assert census["admitted_by_route"][T.ADMIT_PARENTLESS_DECOY] > 0
    assert census["n_admitted_positive"] > 0 and census["n_admitted_negative"] > 0
    assert census["refused"][T.REFUSE_LOO_HOLDOUT] > 0
    assert census["refused"][T.REFUSE_CALIB] > 0
    _, val_census = T.select_rows(rows, rung="val")
    assert val_census["n_admitted"] > 0


@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_STAGE2_DATA") != "1",
    reason="reads the DVC-tracked Stage-2 dataset; local/cluster only",
)
def test_a_real_row_of_every_pool_builds_an_item_with_an_aligned_boundary_target() -> None:
    """The gap that let job 1036 die on the cluster: counting rows is not building items.

    The old real-data tier only ran `select_rows` and checked counts, so nothing ever
    constructed a dataset item from a real row — and the hand-written decoy fixture carried
    a `label_string` no real decoy has. One real row per pool, put through `__getitem__`,
    would have caught it locally in seconds. Parametrised over the pools rather than a
    sample, so a new pool with a different labelling convention cannot slip in unexercised.
    """
    _require_torch()
    from tbox_finder.stage2 import heads as H

    rows = T.load_rows(_DATASET)
    spec = H.load_head_spec()
    background = spec.encode_boundary_string(
        __import__("tbox_finder.labels", fromlist=["CLASS_CODE"]).CLASS_CODE["background"]
    )[0]

    by_pool: dict[str, Any] = {}
    for row in rows:
        by_pool.setdefault(str(row.get("pool")), row)
    assert set(by_pool) == {
        "corpus",
        "dinuc_shuffled",
        "gc_background",
        "leader_decoy",
        "structured_rna",
    }, sorted(by_pool)

    for pool, row in sorted(by_pool.items()):
        ds = T.Stage2SequenceDataset([row], spec, loss_config=L.Stage2LossConfig())
        item = ds[0]
        n_nt = len(item["input_ids"]) - 2
        assert len(item["boundary"]) == n_nt, pool
        assert n_nt == int(row["seq_length"]), pool
        if pool == "corpus":
            # A positive keeps its real annotation, and it is not all-background.
            assert item["boundary"] == spec.encode_boundary_string(row["label_string"]), pool
            assert set(item["boundary"]) != {background}, pool
            assert item[L.BINARY_TERM] == 1, pool
        else:
            # A decoy is supervised as all-background — NOT ignored (the Stage-1
            # convention in data/negatives.py), and not left empty.
            assert item["boundary"] == [background] * n_nt, pool
            assert H.IGNORE_INDEX not in item["boundary"], pool
            assert item[L.BINARY_TERM] == 0, pool


@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_STAGE2_DATA") != "1",
    reason="reads the DVC-tracked Stage-2 dataset; local/cluster only",
)
def test_every_admitted_row_builds_an_item_without_raising() -> None:
    """The whole admitted population, not a sample — this is what the run actually iterates.

    Job 1036 reached the cluster because nothing had ever walked the training set. It costs
    a few seconds and it is the difference between a guard that is right about the corpus
    and a guard that is right about the fixture.
    """
    _require_torch()
    from tbox_finder.stage2 import heads as H

    rows = T.load_rows(_DATASET)
    admitted, _ = T.select_rows(rows, rung="train")
    ds = T.Stage2SequenceDataset(
        [rows[i] for i in admitted], H.load_head_spec(), loss_config=L.Stage2LossConfig()
    )
    for index in range(len(ds)):
        item = ds[index]
        assert len(item["boundary"]) == len(item["input_ids"]) - 2


def test_an_unlabelled_POSITIVE_still_hits_the_alignment_guard() -> None:
    """The all-background branch is for decoys only, and it is fail-closed.

    A corpus row with an empty `label_string` is a data defect, not a negative; handing it a
    fabricated all-background target would train the segmenter on 141 nucleotides of
    invented annotation while every loss stayed finite.
    """
    _require_torch()
    from tbox_finder.stage2 import heads as H

    spec = H.load_head_spec()
    for bad in (
        _row(label_string=None),  # corpus, no label
        _row(label_string=None, source="decoy"),  # decoy-sourced but marked positive
        _decoy(is_tbox=True),  # decoy pool, marked positive
    ):
        ds = T.Stage2SequenceDataset([bad], spec, loss_config=L.Stage2LossConfig())
        with pytest.raises(ValueError, match="label_string is 0 long"):
            _ = ds[0]
    # Positive control: the genuine decoy, unmodified, builds cleanly.
    ok = T.Stage2SequenceDataset([_decoy()], spec, loss_config=L.Stage2LossConfig())[0]
    assert len(ok["boundary"]) == 16


# --------------------------------------------------------------------------------------
# TIER 4 — the committed report, once a run has produced one
# --------------------------------------------------------------------------------------
def test_the_committed_report_validates_and_passes_its_gate() -> None:
    report_path = _REPO / T.DEFAULT_REPORT
    if not report_path.is_file():
        _fail_or_skip(
            "TBOX_REQUIRE_STAGE2_SMOKE",
            f"{report_path.relative_to(_REPO)} not present (the P3-06 run has not landed)",
        )
    report = json.loads(report_path.read_text())
    assert T.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True, report["gate"]["failed"]


def test_the_gate_is_scoped_to_the_rank_that_holds_its_evidence() -> None:
    """Job 1053's finish-line failure: rank-0-only evidence graded on every rank.

    `checkpoint_written` and `provenance_recorded` read artifacts only the primary rank
    produces. On ranks 1..N-1 those clauses are correctly False and were being raised on,
    which killed a run whose rank-0 report said `overall_pass: true` — and cost the
    checkpoint, because the sbatch's `rc` check gates the copy out of node-local scratch.

    Asserted on the CLAUSES, not by reading the source: a non-primary rank's report — the
    one with no checkpoint and no git snapshot — must derive exactly those two as False,
    which is why the verdict cannot be taken on it. The positive control is the same report
    WITH the evidence, where both hold.
    """
    cfg = T.Stage2TrainConfig()
    base = _passing_report()

    non_primary = json.loads(json.dumps(base))
    non_primary["checkpoint"] = {
        "adapter_dir": None,
        "adapter_bytes": 0,
        "heads_bytes": 0,
        "n_saved_parameters": 0,
    }
    non_primary["provenance"] = {
        k: v for k, v in base["provenance"].items() if k not in {"git_sha", "git_branch"}
    }
    clauses = T.derive_clauses(non_primary)
    assert clauses["checkpoint_written"] is False
    assert clauses["provenance_recorded"] is False
    # …and every OTHER clause still holds, which is what makes these two rank-scoped rather
    # than simply broken: the run itself was fine.
    others = {
        k: v for k, v in clauses.items() if k not in {"checkpoint_written", "provenance_recorded"}
    }
    assert all(others.values()), sorted(k for k, v in others.items() if not v)

    # Positive control: with rank 0's evidence present, both hold.
    assert T.derive_clauses(base)["checkpoint_written"] is True
    assert T.derive_clauses(base)["provenance_recorded"] is True
    assert cfg.save_checkpoint is True


def test_train_stage2_returns_without_judging_on_a_non_primary_rank() -> None:
    """The scoping is in the shipped control flow, not just in a comment.

    Located by `ast` so a refactor that re-broadens the gate is caught: the early return for
    a non-primary rank must sit BEFORE `validate_report` is called.
    """
    import ast
    import inspect

    source = inspect.getsource(T.train_stage2)
    assert "validate_report" in ast.dump(ast.parse(source.lstrip()))
    body = source[source.index("if not primary:") :]
    assert body.index("return report") < body.index("validate_report"), (
        "the non-primary early return must precede validate_report, or ranks without the "
        "evidence will judge rank 0's artifacts again"
    )


def test_the_report_records_the_driver_the_run_actually_used() -> None:
    """The two gpu nodes are on different drivers and points may land on either.

    Not a gate clause — a run on either driver is legitimate — but it must be WRITTEN DOWN,
    or a sweep whose arms are compared against each other is uniform on that axis only by
    assumption. The driver comes from nvidia-smi rather than torch because `torch.version.cuda`
    is the CUDA *runtime*, and the runtime is not what diverged between the nodes.
    """
    report = _passing_report()
    device = report["device"]
    assert device["driver_version"] == "590.48.01"
    assert device["name"] and device["capability"] == [8, 6]
    # It is provenance, not a verdict: a run must not fail because of which driver it got.
    clauses = T.derive_clauses(report)
    stripped = json.loads(json.dumps(report))
    stripped["device"] = {"is_cuda": True, "name": None, "driver_version": None}
    assert (
        T.derive_clauses(stripped) == clauses
    ), "the device block leaked into the gate — a legitimate run on the other node would fail"


def test_device_record_survives_a_missing_nvidia_smi() -> None:
    """Provenance must not be able to kill a finished run.

    Losing ~45 GPU-minutes because a version string was unreadable would be the tail wagging
    the dog — the failure this whole addendum is about, in miniature.
    """
    torch = _require_torch()
    record = T.device_record()
    assert set(record) >= {"is_cuda", "torch_version", "driver_version", "n_visible_devices"}
    assert record["torch_version"] == torch.__version__
    # On a CPU-only box driver_version is None and that is a recorded fact, not an error.
    assert record["driver_version"] is None or isinstance(record["driver_version"], str)


def test_checkpoint_outputs_are_files_never_the_adapter_DIRECTORY(tmp_path=None) -> None:
    """The job-1064 defect: PEFT's `lora_adapter` is a directory, and provenance hashes files.

    `provenance.sha256_file` opens each declared output and raises on a directory — correct
    fail-loud behaviour for a shared helper, deliberately not weakened. So the enumeration
    must yield only files, recursively, and must refuse the two ways it could silently record
    nothing: a path that is not a directory, and a directory with no files in it.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        ckpt = Path(root) / "aux1.0_lr1e-4"
        adapter = ckpt / "lora_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"\x00")
        (ckpt / "stage2_heads.pt").write_bytes(b"\x00")

        outputs = T.checkpoint_output_files(ckpt)
        assert len(outputs) == 3
        assert all(Path(o).is_file() for o in outputs), outputs
        assert not any(Path(o).is_dir() for o in outputs)
        # Sorted, so the sidecar is stable across runs rather than filesystem-order dependent.
        assert outputs == sorted(outputs)
        # It is the ADAPTER DIR that must never appear — the exact value that raised.
        assert str(adapter) not in outputs

        # Every declared output must actually be hashable by the real helper.
        from tbox_finder.provenance import sha256_file

        for o in outputs:
            assert len(sha256_file(o)) == 64

        # And the directory itself still raises, so the contract is asserted, not assumed.
        with pytest.raises(IsADirectoryError):
            sha256_file(adapter)

    with tempfile.TemporaryDirectory() as empty, pytest.raises(FileNotFoundError):
        T.checkpoint_output_files(empty)
    with pytest.raises(NotADirectoryError):
        T.checkpoint_output_files(Path(__file__))


# --------------------------------------------------------------------------------------
# CodeRabbit r1 — the scheduler domain, the accumulation flush, the optim-key guard
# --------------------------------------------------------------------------------------
def test_the_lr_schedule_is_sized_in_OPTIMISER_steps_not_micro_batches() -> None:
    """Job 1064's cosine traversed exactly 50% of its domain, ending at 0.5501x base.

    `scheduler.step()` fires once per optimiser step, so sizing the schedule in micro-batches
    makes it cover `1 / gradient_accumulation_steps` of its range. The run reported
    total_scheduled_steps 2830 against n_optimizer_steps 1415 — the two domains conflated in
    one field. This asserts the arithmetic the fix restores.
    """
    micro_per_epoch, epochs, accum = 283, 10, 2
    opt_per_epoch = (micro_per_epoch + accum - 1) // accum
    total_opt = opt_per_epoch * epochs
    warmup = int(round(0.06 * total_opt))

    # The LR must actually reach ~0 by the LAST optimiser step, which is the whole point.
    assert T.lr_scale(total_opt, warmup_steps=warmup, total_steps=total_opt) == pytest.approx(0.0)
    # …and under the OLD (micro-batch) domain it would still be more than half-way up.
    stale_total = micro_per_epoch * epochs
    stale_warmup = int(round(0.06 * stale_total))
    assert T.lr_scale(total_opt, warmup_steps=stale_warmup, total_steps=stale_total) > 0.5

    # ⚠ The arithmetic above is necessary and NOT sufficient: a first version of this test
    # stopped there, and reverting the trainer to the micro-batch domain left it green — a
    # test named for a bug that could not see it. So assert the SHIPPED WIRING too: the
    # scheduler must be constructed over the optimiser-step domain, and the report must carry
    # both counts under names that cannot be confused.
    import inspect

    source = inspect.getsource(T.train_stage2)
    assert "total_opt_steps = opt_per_epoch * cfg.epochs" in source
    assert "warmup_steps = int(round(cfg.warmup_ratio * total_opt_steps))" in source
    assert "total_steps=total_opt_steps" in source, (
        "the LambdaLR is sized in micro-batches again; the cosine will traverse only "
        "1/gradient_accumulation_steps of its domain, as it did in job 1064"
    )
    assert '"total_scheduled_optimizer_steps": total_opt_steps' in source
    assert '"expected_n_optimizer_steps": total_opt_steps' in source


def test_gradient_accumulation_flushes_a_trailing_partial_group_each_epoch() -> None:
    """283 micro-batches against accumulation 2 ends every epoch mid-group.

    With a cumulative `n_steps % accum` the phase also shifts between epochs, so which batches
    get dropped changes epoch to epoch — and the next `zero_grad` discards them. Asserted on
    the shipped source: the loop must count per-epoch and flush after it.
    """
    import inspect

    source = inspect.getsource(T.train_stage2)
    assert "micro_in_group = 0" in source
    assert "if micro_in_group == cfg.gradient_accumulation_steps:" in source
    assert (
        "n_steps % cfg.gradient_accumulation_steps" not in source
    ), "the cumulative-phase form is back; a trailing partial group is silently discarded"
    # The flush must precede the val pass, or the epoch's last gradients never land.
    tail = source[source.index("if micro_in_group:") :]
    assert tail.index("optimizer.step()") < tail.index("if cfg.eval_val")


def test_an_unknown_optim_key_is_refused_exactly_as_an_unknown_loss_key_is() -> None:
    """A misspelled `optim.lr` would otherwise train the default and report the swept value.

    This is the same failure the /loss guard exists to stop; /optim had no equivalent.
    """
    with pytest.raises(ValueError, match="optim"):
        T._cfg_from_mapping({"optim": {"learning_rate": 3e-4}})
    with pytest.raises(ValueError, match="optim"):
        T._cfg_from_mapping({"optim": {"lr": 3e-4, "wieght_decay": 0.01}})
    # Positive control: every key the trainer really consumes composes.
    built = T._cfg_from_mapping(
        {
            "optim": {
                "name": "adamw",
                "lr": 3e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.06,
                "grad_clip": 1.0,
            }
        }
    )
    assert built.lr == pytest.approx(3e-4)


def test_world_size_zero_raises_rather_than_becoming_one() -> None:
    """Unset means "not under torchrun"; zero means a launcher computed it and got it wrong."""
    import os as _os

    from tbox_finder.train import ddp

    old = _os.environ.get("WORLD_SIZE")
    try:
        _os.environ["WORLD_SIZE"] = "0"
        with pytest.raises(ValueError, match="WORLD_SIZE=0"):
            ddp.ddp_world_size()
        _os.environ["WORLD_SIZE"] = "8"
        assert ddp.ddp_world_size() == 8
        _os.environ.pop("WORLD_SIZE")
        assert ddp.ddp_world_size() == 1  # unset is still the local-smoke default
    finally:
        if old is None:
            _os.environ.pop("WORLD_SIZE", None)
        else:
            _os.environ["WORLD_SIZE"] = old


def test_the_sbatch_clears_the_checkpoint_only_after_the_node_is_known_good() -> None:
    """The clear must sit next to the COPY that replaces it — not merely after the health check.

    Two versions of this were too weak. Clearing before the health check let a bad node destroy
    a good artifact on its way out; clearing after the health check but BEFORE training let an
    OOM, a NaN gate, or a failed freshness check end the point with no checkpoint and the
    previous one already gone. The property that actually holds is: nothing is deleted until
    training has produced a replacement.
    """
    text = _SBATCH.read_text()
    clear_at = text.index('rm -rf "$CKPT_DIR"')
    assert text.index("STAGE2_NODE_UNHEALTHY") < clear_at, "clear precedes the health check"
    assert text.index("torchrun --standalone") < clear_at, (
        "the checkpoint clear precedes the training launch — a failed run destroys the last "
        "good artifact and produces nothing to replace it"
    )
    # …and the copy that supersedes it follows immediately.
    assert clear_at < text.index('cp -r "$SCRATCH_CKPT/lora_adapter"')


@pytest.mark.parametrize(
    ("per_epoch", "accum"),
    [(283, 2), (284, 2), (283, 1), (283, 3), (283, 4), (1, 2), (2, 2), (9, 4)],
)
def test_the_predicted_optimiser_step_count_equals_what_the_loop_actually_does(
    per_epoch: int, accum: int
) -> None:
    """`total_opt_steps` must equal the number of `scheduler.step()` calls, exactly.

    The fix for the micro-batch domain introduced a second way to get this wrong: the flush of
    a trailing partial group adds a `scheduler.step()` that the ceiling-division prediction has
    to have anticipated. One too many and the cosine overruns its domain; one too few and it
    stops short — the same defect as job 1064's, in either direction.

    So the loop's stepping logic is replayed here rather than reasoned about, over the real
    shape (283 micro-batches per rank) plus the even, unit, and remainder-heavy cases.
    """
    epochs = 10
    opt_per_epoch = T._n_batches(per_epoch, accum)
    total_opt = opt_per_epoch * epochs

    calls = 0
    for _ in range(epochs):
        micro = 0
        for _ in range(per_epoch):
            micro += 1
            if micro == accum:
                calls += 1
                micro = 0
        if micro:  # the trailing-group flush the shipped loop performs
            calls += 1

    assert calls == total_opt, (
        f"predicted {total_opt} scheduler steps but the loop makes {calls} at "
        f"per_epoch={per_epoch}, accum={accum}"
    )
    # …and the schedule therefore lands exactly at zero, not short of it and not past it.
    warmup = int(round(0.06 * total_opt))
    assert T.lr_scale(calls, warmup_steps=warmup, total_steps=total_opt) == pytest.approx(0.0)


def test_the_gate_grades_BOTH_step_domains() -> None:
    """Job 1064's every recorded number was self-consistent while the optimiser did half.

    A clause that checks only micro-batches cannot see the scheduler-domain regression
    return, which is exactly how it shipped.
    """
    report = _passing_report()
    assert T.derive_clauses(report)["steps_ran"] is True
    stale = json.loads(json.dumps(report))
    stale["steps"]["n_optimizer_steps"] = 707  # half, as job 1064 did
    assert T.derive_clauses(stale)["steps_ran"] is False
    drifted = json.loads(json.dumps(report))
    drifted["steps"]["total_scheduled_optimizer_steps"] = 2830  # the micro-batch domain
    assert T.derive_clauses(drifted)["steps_ran"] is False
    # A case ONLY the expected-vs-measured comparison catches: the scheduler's domain still
    # agrees with what was taken, but the loop took a different number than predicted. Without
    # this the two new comparisons cover for each other and neither is individually tested.
    under = json.loads(json.dumps(report))
    under["steps"]["expected_n_optimizer_steps"] = 1500
    assert under["steps"]["total_scheduled_optimizer_steps"] == under["steps"]["n_optimizer_steps"]
    assert T.derive_clauses(under)["steps_ran"] is False


def test_a_nan_GRADIENT_fails_the_gate_even_when_the_rank0_loss_is_finite() -> None:
    """Greptile P2: `n_nonfinite_steps` counts the rank-0 forward loss and nothing else.

    Under DDP a NaN gradient produced on any other rank all-reduces into every rank's gradient
    tensor and poisons the weights, while rank 0's loss value stays finite — so the loss
    counter is structurally blind to it. The post-all-reduce `clip_grad_norm_` total norm is
    the quantity that sees it, and the gate now reads both.
    """
    report = _passing_report()
    assert T.derive_clauses(report)["losses_finite"] is True
    poisoned = json.loads(json.dumps(report))
    poisoned["losses"]["n_nonfinite_grad_steps"] = 1
    assert poisoned["losses"]["final_train_total"] == report["losses"]["final_train_total"]
    assert poisoned["losses"]["n_nonfinite_steps"] == 0, "the rank-0 loss is still finite"
    assert (
        T.derive_clauses(poisoned)["losses_finite"] is False
    ), "a NaN gradient from a non-primary rank passed the gate"


def test_the_tri_state_helper_names_the_column_it_actually_read() -> None:
    """Greptile P2: it was written for `nested_train` and is also called for `calib`/`is_tbox`.

    A schema migration putting "yes" into `calib` would otherwise raise an error blaming
    `nested_train`, sending the reader to a column that is perfectly fine.
    """
    for field in ("nested_train", "calib", "is_tbox"):
        with pytest.raises(ValueError, match=f"{field}='yes'"):
            T._bool_or_none("yes", field=field)
    # And the call sites pass their own column name rather than defaulting.
    import inspect

    source = inspect.getsource(T.row_eligibility)
    assert 'field="calib"' in source and 'field="nested_train"' in source


# --------------------------------------------------------------------------------------
# The job-1064 reports vs the schema-2 gate — the defect PINNED, not excused
# --------------------------------------------------------------------------------------
_SWEEP_REPORTS = sorted((_REPO / "reports" / "p3" / "sweep").glob("aux*.json"))


def test_the_committed_sweep_reports_are_all_present_and_schema_1() -> None:
    """Six arms, all written by job 1064 before the clause set changed.

    An emptiness guard first: a glob that matched nothing would make every assertion below
    vacuously true, which is exactly how a "the artifacts are fine" claim gets made about no
    artifacts at all.
    """
    assert len(_SWEEP_REPORTS) == 6, [p.name for p in _SWEEP_REPORTS]
    for path in _SWEEP_REPORTS:
        report = json.loads(path.read_text())
        assert report["schema_version"] == "1", (
            f"{path.name} is no longer schema 1 — if it was regenerated, the pinned "
            "scheduler-defect expectation below is stale and must be re-derived"
        )
        assert report["step"] == T.STEP


def test_the_committed_reports_FAIL_the_corrected_gate_on_the_scheduler_defect() -> None:
    """This is the point of the whole exercise: the defect is recorded, not excused.

    Job 1064's arms trained with the LR schedule sized in micro-batches, so the optimiser
    advanced 1,415 of 2,830 scheduled steps and the cosine stopped at 0.5501x base. Schema 2's
    `steps_ran` compares the two domains and therefore refuses those runs — correctly.

    The temptation on a version bump is to excuse a clause the older schema lacks. That is
    right when the old report would re-derive TRUE, and WRONG here: this clause re-derives
    FALSE for a real reason, and excusing it would launder a known defect through a
    compatibility shim. So the failure is asserted instead. If these reports ever start
    passing, someone regenerated or edited them and this test says so.
    """
    for path in _SWEEP_REPORTS:
        report = json.loads(path.read_text())
        clauses = T.derive_clauses(report)
        failing = sorted(k for k, v in clauses.items() if not v)

        # EXACTLY three, for reasons in TWO different categories that must not be conflated:
        #
        #  • `steps_ran` — a REAL defect. Those runs sized the cosine in micro-batches, so the
        #    optimiser advanced half the scheduled steps. Schema 2 refuses them correctly, and
        #    that refusal is the thing being pinned.
        #
        #  • `losses_finite` — NOT a defect: schema 1 predates `n_nonfinite_grad_steps`, so the
        #    post-all-reduce gradient check simply was not measured. "Unmeasured" is not
        #    "passed" and it is not "failed" either; the clause cannot be satisfied by a report
        #    that lacks its evidence, and excusing it into TRUE would assert those runs had
        #    finite gradients when nobody looked. It stays False, and it stays documented here.
        #
        #  • `backbone_pinned` — added at P3-17 (schema 3), and in the SAME category as
        #    `losses_finite`. Schema 1 predates the `wrap.backbone` identity block that
        #    ADR-0002 A15's allow-list writes, so there is no evidence here to re-derive from.
        #    These runs did adapt the pinned rinalmo-giga — they never recorded it, which is
        #    the gap the clause exists to close. Excused into TRUE it would assert an identity
        #    nobody wrote down, so it stays False and stays documented.
        assert failing == ["backbone_pinned", "losses_finite", "steps_ran"], (
            f"{path.name} fails {failing}; expected exactly the scheduler defect plus the two "
            "unmeasured-evidence clauses. A new entry means a second defect surfaced; a "
            "missing one means a clause stopped biting or the report was regenerated"
        )
        # And the distinction is checked, not just asserted: one has its evidence and fails on
        # it, the other two have no evidence at all.
        assert report["steps"]["n_optimizer_steps"] != report["steps"]["n_steps"]
        assert "n_nonfinite_grad_steps" not in report["losses"]
        assert "backbone" not in report.get("wrap", {})
        assert report["losses"]["n_nonfinite_steps"] == 0  # what schema 1 DID measure: clean


def test_the_recorded_step_counts_show_exactly_the_defect_that_was_diagnosed() -> None:
    """Not "it failed" but "it failed by half" — the number that made this diagnosable."""
    for path in _SWEEP_REPORTS:
        steps = json.loads(path.read_text())["steps"]
        # Schema 1 wrote the scheduler's domain under the micro-batch name.
        assert steps["total_scheduled_steps"] == steps["n_steps"]
        assert steps["n_optimizer_steps"] * 2 == steps["n_steps"], (
            "gradient_accumulation_steps was 2, so the optimiser took exactly half the "
            "micro-batch count — the ratio that made the cosine stop at 50%"
        )


# --------------------------------------------------------------------------------------
# PRODUCER tests — these run train_stage2. The predicate tests above do not.
# --------------------------------------------------------------------------------------
def _tiny_run(tmp: Path, **over: Any):
    """Run `train_stage2` on CPU through a tiny backbone; return (report, raised).

    The gate fails on **two** clauses by construction, and both are the point:
    `full_population` because the run is truncated, and — since ADR-0002 A15 —
    `backbone_pinned`, because the weights under the adapters are this fixture rather than a
    pinned checkpoint (`loaded_from_registry` is False). A tiny smoke must never be able to
    certify a production artifact. The caller gets the written report rather than an exception
    it has to unpick; `test_the_tiny_run_is_refused_for_BOTH_reasons` asserts the pair.
    """
    from multimolecule import RiNALMoConfig, RiNALMoModel

    cfg_kwargs: dict[str, Any] = dict(
        epochs=2,
        batch_size=1,
        eval_batch_size=2,
        max_records=9,  # ODD, so accumulation leaves a trailing partial group each epoch
        eval_max_records=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        device="cpu",
        log_every=10_000,
        report_path=str(tmp / "r.json"),
        checkpoint_dir=str(tmp / "ckpt"),
        wandb_dir=str(tmp / "wb"),
    )
    cfg_kwargs.update(over)
    cfg = T.Stage2TrainConfig(**cfg_kwargs)
    c = RiNALMoConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
    )
    c._attn_implementation = "sdpa"
    raised = None
    try:
        T.train_stage2(
            cfg, base_model=RiNALMoModel(c, add_pooling_layer=False), log=lambda *a: None
        )
    except RuntimeError as exc:  # the truncated-run gate failure
        raised = exc
    return json.loads(Path(cfg.report_path).read_text()), raised


@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_STAGE2_DATA") != "1",
    reason="runs the real trainer over the DVC-tracked dataset; local/cluster only",
)
def test_the_TRAINER_takes_exactly_the_optimiser_steps_the_schedule_was_sized_for() -> None:
    """CodeRabbit r3: the earlier version reimplemented the loop and never ran it.

    Removing `scheduler.step()` from the trailing flush left that test green, because it was
    asserting arithmetic it had written itself. This one spies on the real `optimizer.step`
    and `scheduler.step` through an actual `train_stage2` call, with an ODD micro-batch count
    and accumulation 2 so the trailing partial group is exercised on every epoch.
    """
    _require_stack()
    import tempfile

    import torch

    opt_calls, sched_calls = [], []
    real_opt, real_sched = torch.optim.AdamW.step, torch.optim.lr_scheduler.LambdaLR.step
    torch.optim.AdamW.step = lambda self, *a, **k: (opt_calls.append(1), real_opt(self, *a, **k))[1]
    torch.optim.lr_scheduler.LambdaLR.step = lambda self, *a, **k: (
        sched_calls.append(1),
        real_sched(self, *a, **k),
    )[1]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = _tiny_run(Path(tmp))
    finally:
        torch.optim.AdamW.step = real_opt
        torch.optim.lr_scheduler.LambdaLR.step = real_sched

    steps = report["steps"]
    expected = steps["expected_n_optimizer_steps"]
    assert steps["n_optimizer_steps"] == expected
    assert (
        len(opt_calls) == expected
    ), f"optimizer.step called {len(opt_calls)}, expected {expected}"
    # The one the old test could not see: LambdaLR is constructed with an initial step, so the
    # loop's own calls are what must match the schedule's domain.
    assert (
        len(sched_calls) - 1 <= expected <= len(sched_calls)
    ), f"scheduler.step called {len(sched_calls)} times against a domain of {expected}"
    assert steps["total_scheduled_optimizer_steps"] == expected


@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_STAGE2_DATA") != "1",
    reason="runs the real trainer over the DVC-tracked dataset; local/cluster only",
)
def test_the_TRAINER_records_a_non_finite_gradient_it_actually_encountered() -> None:
    """CodeRabbit r3: the earlier version edited a report fixture, not the producer.

    Removing the increment from `train_stage2` left it green. Here a real `inf` is injected
    into a live gradient after the real backward, so the real `gradient_total_norm` observes
    it, the real counter increments, and the real report carries it — and the gate refuses.
    """
    _require_stack()
    import tempfile

    real_fb = T.forward_backward

    def poisoning(model, loss_fn, batch, *, scale=1.0):
        out = real_fb(model, loss_fn, batch, scale=scale)
        for prm in model.parameters():
            if prm.requires_grad and prm.grad is not None:
                prm.grad[0] = float("inf")
                break
        return out

    T.forward_backward = poisoning
    try:
        with tempfile.TemporaryDirectory() as tmp:
            report, raised = _tiny_run(Path(tmp))
    finally:
        T.forward_backward = real_fb

    assert (
        report["losses"]["n_nonfinite_grad_steps"] > 0
    ), "the trainer did not record the non-finite gradient it was handed"
    assert T.derive_clauses(report)["losses_finite"] is False
    assert raised is not None and "losses_finite" in str(raised)
    # ⚠ NOT asserting the forward loss stayed finite. A first draft did, and it failed —
    # correctly. Injecting `inf` into a live gradient and then stepping poisons the WEIGHTS,
    # so later forward passes produce non-finite losses too. What this clause exists for is
    # that the gradient signal fires on the step the gradient goes bad, before the loss
    # counter could have seen anything — so it necessarily fires at least as often. The
    # "finite loss, poisoned gradient" case is exercised directly on the predicate by
    # test_a_nan_GRADIENT_fails_the_gate_even_when_the_rank0_loss_is_finite.
    # ⚠ And NOT comparing the two counters. They have different denominators: the loss counter
    # increments per MICRO-BATCH, the gradient counter only per OPTIMISER step (one per
    # `gradient_accumulation_steps`). A first draft asserted grad >= loss, which is arithmetic
    # nonsense once accumulation > 1 — the third assertion in this test that sounded right and
    # was not derived from the mechanism. What is actually true, and enough:
    assert report["losses"]["n_nonfinite_grad_steps"] <= report["steps"]["n_optimizer_steps"]


def test_grad_clip_zero_reads_the_norm_without_touching_the_gradients() -> None:
    """CodeRabbit r3: `clip_grad_norm_(..., inf)` is NOT a no-op — it writes `nan`.

    PyTorch scales by `max_norm / (total_norm + 1e-6)`; with both infinite that is `nan`, and
    the gradients are multiplied by it. Verified on the pinned torch: `[inf, 1, 2]` came back
    `[nan, nan, nan]` while the returned norm still read `inf`, so the detection signal was
    fine and the optimiser's input was not.
    """
    torch = _require_torch()

    for grads, clip, expect_finite in (
        ([float("inf"), 1.0, 2.0], 0.0, False),
        ([float("nan"), 1.0], 0.0, False),
        ([3.0, 4.0], 0.0, True),
    ):
        prm = torch.nn.Parameter(torch.zeros(len(grads)))
        prm.grad = torch.tensor(grads)
        norm = T.gradient_total_norm([prm], grad_clip=clip)
        assert math.isfinite(norm) is expect_finite
        # …and the gradients are untouched, which is the whole point. Compared elementwise
        # with NaN treated as equal to itself, since `nan == nan` is False.
        after = prm.grad.tolist()
        assert len(after) == len(grads)
        assert all(
            (a != a and b != b) or a == b for a, b in zip(after, grads, strict=True)
        ), f"gradients were mutated: {grads} -> {after}"
    # A finite norm is also correct, not merely finite.
    prm = torch.nn.Parameter(torch.zeros(2))
    prm.grad = torch.tensor([3.0, 4.0])
    assert T.gradient_total_norm([prm], grad_clip=0.0) == pytest.approx(5.0)


def test_every_committed_report_carries_its_legacy_annotation() -> None:
    """CodeRabbit r3 (High): the producer schema was fixed; the ARTIFACTS stayed misleading.

    A reader opening `aux0.0_lr1e-4.json` saw `best_val_total: 0.00037977` beside
    `saved_from_epoch: 9` and had to infer that the two describe different weights. The
    annotation states it, and derives `saved_val_total` from the file's own `val.history`
    rather than from anywhere else.

    The reports are ANNOTATED, not rewritten — they are job 1064's own record, and the
    decision was to keep the checkpoints and record the defect. This asserts both halves: the
    annotation is present and correct, and the original measurements are still there.
    """
    assert len(_SWEEP_REPORTS) == 6
    for path in _SWEEP_REPORTS:
        report = json.loads(path.read_text())
        legacy = report.get("legacy")
        assert legacy, f"{path.name} lost its legacy annotation"

        # Derived from this file's own history, not restated from elsewhere.
        history = {h["epoch"]: h["total"] for h in report["val"]["history"]}
        saved_epoch = report["checkpoint"]["saved_from_epoch"]
        assert legacy["saved_from_epoch"] == saved_epoch
        assert legacy["saved_val_total"] == pytest.approx(history[saved_epoch])

        # The original measurements survive untouched — annotation, not rewrite.
        assert report["checkpoint"]["best_val_total"] == pytest.approx(
            legacy["best_val_total_observed_during_training"]
        )
        assert report["schema_version"] == "1" == legacy["schema_version_of_this_file"]
        assert legacy["fails_current_gate_on"] == [
            "backbone_pinned",
            "losses_finite",
            "steps_ran",
        ]
        # The flag is DERIVED from the epochs, not asserted. A first version hardcoded it True
        # on all six while two arms have best == saved — the annotation contradicting the very
        # test below that counts four differing arms.
        differs = legacy["best_val_epoch_observed_during_training"] != legacy["saved_from_epoch"]
        assert legacy["best_val_total_is_NOT_the_saved_weights"] is differs
        # …and the note matches the case it describes, rather than one blanket wording.
        assert ("discarded" in legacy["note"]) is differs
        assert ("agree and neither is misleading" in legacy["note"]) is not differs
        # …and that list must still be the truth, not a stale copy.
        assert (
            sorted(k for k, v in T.derive_clauses(report).items() if not v)
            == legacy["fails_current_gate_on"]
        )


def test_the_annotation_names_a_gap_that_is_real_in_four_of_six_arms() -> None:
    """Two arms had best == final, so an annotation asserting a gap everywhere would be wrong.

    The point is not that every checkpoint underperforms its best — it is that the report must
    not *claim* a score its weights do not achieve. Where best == epoch 9 the two agree, and
    the annotation still correctly reports them as equal.
    """
    gaps = 0
    for path in _SWEEP_REPORTS:
        report = json.loads(path.read_text())
        legacy = report["legacy"]
        best = legacy["best_val_total_observed_during_training"]
        saved = legacy["saved_val_total"]
        if legacy["best_val_epoch_observed_during_training"] == legacy["saved_from_epoch"]:
            assert saved == pytest.approx(best), path.name
        else:
            assert saved > best, f"{path.name}: best epoch should beat the saved epoch"
            gaps += 1
    assert gaps == 4, f"expected 4 arms whose best epoch precedes epoch 9, found {gaps}"


# ======================================================================================
# P3-17 — the RNA-FM COMPARATOR arm (ADR-0002 D6 + A15)
# ======================================================================================
# The comparator's value is entirely in *differing from the production arm in one place*.
# So these tests are mostly about sameness: same corpus, same objective, same batch geometry,
# same harness — and one backbone. A comparator that drifted on any other axis would still
# train, still score, still produce an ECE, and P3-18 would attribute the difference to the
# backbone anyway. That is the failure this section exists to make loud.


def _sbatch_code(path: Path) -> str:
    """An sbatch's executable lines, with comments and prose stripped.

    ⚠ EVERY assertion about what an sbatch *does* must read this, not the raw text. These
    headers are long and explain, at length, the flags they deliberately do not use and the
    tokens they deliberately do pass — so a whole-file substring search reads the explanation
    as the thing. It bit twice while this file was being written: once on `--nodelist` (caught
    by the production sbatch's own warning) and once on `model=rnafm_stage2`, where deleting
    the token from the launch line left the guard GREEN because the header still mentioned it.
    Fixing only the second instance would have left the class.
    """
    out = []
    for line in path.read_text().splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and not stripped.startswith("#SBATCH"):
            continue
        out.append(line)
    return "\n".join(out)


def _launch_tokens(path: Path, module: str) -> list[str]:
    """The tokens of the logical `-m <module>` launch line — the command, not the commentary.

    Backslash continuations are joined, exactly as `tests/unit/test_sbatch_overrides.py` does,
    so a multi-line torchrun invocation is read as one command.
    """
    lines = path.read_text().splitlines()
    joined, i = [], 0
    while i < len(lines):
        chunk = []
        while i < len(lines):
            chunk.append(lines[i].rstrip())
            if not lines[i].rstrip().endswith("\\"):
                break
            i += 1
        text = " ".join(c.rstrip("\\") for c in chunk if not c.lstrip().startswith("#"))
        if text.strip():
            joined.append(text)
        i += 1
    marker = f"-m {module}"
    hits = [t for t in joined if marker in t]
    assert len(hits) == 1, f"expected exactly one `{marker}` launch in {path.name}, got {len(hits)}"
    return hits[0].split()


def _require_rnafm_stack():
    """The `ml-rnafm` stack (multimolecule >= 0.2.0 + torch + peft), or skip loudly.

    Separate from `_require_stack` on purpose: `ml-rna`'s multimolecule 0.1.0 satisfies that
    one and **cannot load RNA-FM at all**, so reusing it would let this tier skip for the wrong
    reason in one env and fail confusingly in the other.
    """
    if os.environ.get("CUDA_HOME") is None:
        _fail_or_skip("TBOX_REQUIRE_RNAFM", "CUDA_HOME unset — multimolecule won't import")
    try:
        import importlib.metadata as md

        import peft  # noqa: F401
        import torch

        version = md.version("multimolecule")
    except Exception as exc:  # pragma: no cover - env-dependent
        _fail_or_skip("TBOX_REQUIRE_RNAFM", f"ml-rnafm stack unavailable ({exc})")
    if tuple(int(p) for p in version.split(".")[:2]) < (0, 2):
        _fail_or_skip(
            "TBOX_REQUIRE_RNAFM",
            f"multimolecule {version} is the ml-rna pin and cannot load multimolecule/rnafm "
            "(ADR-0002 A15) — activate tbox-ml-rnafm",
        )
    return torch


# -- the launcher ---------------------------------------------------------------------
def test_the_comparator_sbatch_exists_and_asks_for_a_whole_node_without_pinning_one() -> None:
    text = _SBATCH_RNAFM.read_text()
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:a4000:8" in text
    # Scan the DIRECTIVES, not the prose — the header explains at length why `--nodelist` is
    # never used, and a whole-file substring search reads that explanation as the flag. (It
    # did, on the first cut of this test; the production sbatch's equivalent already carried
    # the warning.)
    directives = [ln for ln in text.splitlines() if ln.startswith("#SBATCH ")]
    assert directives
    for forbidden in ("--nodelist", "--account", "--qos", "--partition=compute"):
        assert not any(forbidden in ln for ln in directives), forbidden
    # One arm, so no --array: this is the production POINT, not a sweep.
    assert not any("--array" in ln for ln in directives)
    # The health check must run BEFORE the download, or a bad node costs minutes instead of
    # seconds (job 1036 lost three points that way).
    assert text.index("RNAFM_NODE_UNHEALTHY") < text.index("hf cache warm")


def test_the_comparator_sbatch_activates_the_RNAFM_env_not_ml_rna() -> None:
    """The ADR-0002 A15 trap: `conda activate tbox-ml-rna` here is a one-word mistake whose
    failure mode is a ValueError about vocab sizes, thrown after the queue wait."""
    code = _sbatch_code(_SBATCH_RNAFM)
    assert "conda activate tbox-ml-rnafm" in code
    assert "conda activate tbox-ml-rna\n" not in code
    assert "conda activate tbox-ml-dna" not in code
    # ...and it says so before spending GPU time, rather than discovering it in the loader.
    assert 'md.version("multimolecule")' in code, "no in-job env assertion"


def test_the_comparator_sbatch_scratch_var_is_not_named_BUILD() -> None:
    """`conda activate` exports BUILD=x86_64-conda-linux-gnu and clobbers it (job 789)."""
    code = _sbatch_code(_SBATCH_RNAFM)
    assert "JOB_SCRATCH=" in code
    assert not re.search(r"^BUILD=", code, re.MULTILINE)


def test_the_comparator_sbatch_selects_BOTH_halves_of_the_backbone_switch() -> None:
    """Neither half is sufficient, and the launcher must carry both.

    `backbone=` reaches the loader; `model=` selects the /model group that RECORDS the
    checkpoint identity. `_cfg_from_mapping` refuses either alone (asserted below), so a
    launcher carrying only one would die at compose time after the queue wait.
    """
    tokens = _launch_tokens(_SBATCH_RNAFM, "tbox_finder.stage2.train")
    assert 'backbone="$BACKBONE"' in tokens, tokens
    assert "model=rnafm_stage2" in tokens, tokens
    # ...and $BACKBONE resolves to the comparator, not to whatever was last assigned.
    code = _sbatch_code(_SBATCH_RNAFM)
    assert f'BACKBONE="{BR.COMPARATOR_BACKBONE}"' in code


def test_the_comparator_trains_the_PRODUCTION_configuration() -> None:
    """Same point, same batch geometry — the arm differs in the backbone and nothing else.

    Read out of the shipped configs rather than hard-coded here, so a drift in the production
    defaults surfaces as a failure instead of silently making the comparator non-comparable.
    """
    code = _sbatch_code(_SBATCH_RNAFM)
    loss_yaml = (_CONF / "loss" / "stage2.yaml").read_text()
    optim_yaml = _OPTIM_CONF.read_text()
    aux = re.search(r"^aux_weight:\s*(\S+)", loss_yaml, re.MULTILINE).group(1)
    lr = re.search(r"^lr:\s*(\S+)", optim_yaml, re.MULTILINE).group(1)
    assert float(aux) == 1.0 and float(lr) == pytest.approx(1e-4)
    # The comparator trains a PAIR: the production point plus its lr-matched aux=0 control,
    # which is what `select_arm_pair` requires and what job 1372 discovered was missing after a
    # full training run. Both must be present, and the production one must match conf/.
    assert (
        f"AUX_WEIGHTS=({float(aux):.1f} 0.0)" in code
    ), "the comparator must train the production aux_weight AND an lr-matched aux=0 control"
    assert 'LR="1e-4"' in code and float("1e-4") == pytest.approx(float(lr))
    # The production arm's measured batch geometry, carried over unchanged. Changing it would
    # make an ECE difference attributable to batch size as well as backbone.
    train_yaml = _TRAIN_CONF.read_text()
    assert re.search(r"^batch_size:\s*4\s*$", train_yaml, re.MULTILINE)
    assert re.search(r"^gradient_accumulation_steps:\s*2\s*$", train_yaml, re.MULTILINE)
    assert "BATCH_SIZE=4" in code and "GRAD_ACCUM=2" in code


def test_the_comparator_sbatch_sizes_the_backbone_it_trains() -> None:
    """A sizing leg that measured the *other* model would be worse than none: it would license
    a batch size on evidence from a 6.5x larger network and read as diligence.
    """
    tokens = _launch_tokens(_SBATCH_RNAFM, "tbox_finder.stage2.sizing")
    assert "--backbone" in tokens, tokens
    assert tokens[tokens.index("--backbone") + 1] == '"$BACKBONE"', tokens
    # ...and the report is checked to be ABOUT this backbone before it licenses anything.
    assert 'm.get("backbone")' in _sbatch_code(_SBATCH_RNAFM)


def test_the_comparator_sbatch_never_writes_the_production_destinations() -> None:
    prod_ckpt = T.DEFAULT_CKPT_DIR
    code = _sbatch_code(_SBATCH_RNAFM)
    assert prod_ckpt not in code, f"comparator sbatch names the production checkpoint {prod_ckpt}"
    assert "reports/p3/sweep/" not in code, "comparator sbatch writes into the production sweep dir"
    assert T.default_checkpoint_dir(BR.COMPARATOR_BACKBONE) in code
    assert T.DEFAULT_REPORT not in code


# -- the config wiring -----------------------------------------------------------------
def test_the_comparator_model_group_exists_and_records_the_pinned_checkpoint() -> None:
    text = (_CONF / "model" / "rnafm_stage2.yaml").read_text()
    spec = BR.resolve_backbone(BR.COMPARATOR_BACKBONE)
    assert f"repo_id: {spec.repo_id}" in text
    assert f"revision: {spec.revision}" in text
    assert f"hidden_dim: {spec.hidden_size}" in text
    assert f"expected_param_count: {spec.expected_param_count}" in text
    assert f"env_lock: {spec.env_lock}" in text


def test_the_primary_config_carries_the_backbone_key_as_a_literal() -> None:
    """Struct mode: a dataclass field with no YAML key is the job-669 class."""
    text = _TRAIN_CONF.read_text()
    assert re.search(rf"^backbone:\s*{re.escape(BR.PRODUCTION_BACKBONE)}\s*$", text, re.MULTILINE)
    assert "backbone" in T.Stage2TrainConfig.__dataclass_fields__


def test_hydra_composes_the_comparator_arm() -> None:
    compose, initialize_config_dir = _require_hydra()
    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(
            config_name="train/stage2",
            overrides=["backbone=rnafm", "model=rnafm_stage2"],
        )
    assert cfg.backbone == "rnafm"
    assert cfg.model.repo_id == "multimolecule/rnafm"
    # Everything else is the production arm's, untouched.
    assert cfg.loss.aux_weight == pytest.approx(1.0)
    assert cfg.optim.lr == pytest.approx(1e-4)
    assert cfg.batch_size == 4
    assert cfg.gradient_accumulation_steps == 2


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        (["backbone=rnafm"], "backbone switched, /model group left on RiNALMo"),
        (["model=rnafm_stage2"], "/model group switched, backbone left on production"),
    ],
)
def test_half_a_backbone_switch_is_refused(overrides, why) -> None:
    """Both directions. Refusing only one would leave the other as a silent way to record one
    checkpoint's identity while loading the other's weights."""
    compose, initialize_config_dir = _require_hydra()
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(config_name="train/stage2", overrides=overrides)
    with pytest.raises(ValueError, match="checkpoint identity"):
        T._cfg_from_mapping(OmegaConf.to_container(cfg, resolve=True))


def test_a_comparator_config_aimed_at_the_production_destinations_is_refused() -> None:
    """The shipped checkpoint dir is DVC-tracked and holds the six P3-06 arms; `train_stage2`
    clears and rewrites `checkpoint_dir` ([[two-outputs-one-path-destroys-the-first]])."""
    # ⚠ `gradient_checkpointing=False` is passed so this test isolates the guard it NAMES.
    # `Stage2TrainConfig` defaults checkpointing ON, and since the job-1370 fix that is
    # refused first for this backbone — so without it this test would pass on the wrong
    # refusal and stop covering destinations at all
    # ([[sabotage-attribution-names-the-test]]).
    isolate = dict(backbone=BR.COMPARATOR_BACKBONE, gradient_checkpointing=False)
    with pytest.raises(ValueError, match="PRODUCTION arm's destination"):
        T.Stage2TrainConfig(**isolate)
    with pytest.raises(ValueError, match="PRODUCTION arm's destination"):
        T.Stage2TrainConfig(
            **isolate,
            checkpoint_dir=T.default_checkpoint_dir(BR.COMPARATOR_BACKBONE),
            report_path=T.DEFAULT_REPORT,
        )
    # The per-arm pair composes cleanly — the guard refuses the collision, not the arm.
    cfg = T.Stage2TrainConfig(
        **isolate,
        checkpoint_dir=T.default_checkpoint_dir(BR.COMPARATOR_BACKBONE),
        report_path=T.default_report_path(BR.COMPARATOR_BACKBONE),
    )
    assert cfg.backbone == BR.COMPARATOR_BACKBONE

    # ...and the two guards are independent: the checkpointing one fires even when the
    # destinations are correct, so neither is standing in for the other.
    with pytest.raises(ValueError, match="not usable on backbone"):
        T.Stage2TrainConfig(
            backbone=BR.COMPARATOR_BACKBONE,
            gradient_checkpointing=True,
            checkpoint_dir=T.default_checkpoint_dir(BR.COMPARATOR_BACKBONE),
            report_path=T.default_report_path(BR.COMPARATOR_BACKBONE),
        )


def test_the_comparator_records_its_OWN_env_lock_not_the_production_one() -> None:
    """ADR-0002 A15: the arms run under different locks, so a run that stamped the module
    constant would name an environment it did not use and hash the wrong file."""
    assert T.env_lock_for(BR.COMPARATOR_BACKBONE) != T.ENV_LOCK
    assert T.env_lock_for(BR.PRODUCTION_BACKBONE) == T.ENV_LOCK


# -- the gate clause -------------------------------------------------------------------
def _rnafm_report() -> dict[str, Any]:
    """A passing report for the comparator arm — the production fixture with the backbone
    swapped, which is exactly what the run itself is."""
    spec = BR.resolve_backbone(BR.COMPARATOR_BACKBONE)
    report = _passing_report()
    report["config"]["backbone"] = spec.key
    report["wrap"]["backbone"] = {
        **BR.backbone_summary(spec),
        "loaded_from_registry": True,
    }
    report["wrap"]["stage2_heads"]["d_model"] = spec.hidden_size
    return report


def test_the_comparator_report_passes_the_same_gate_the_production_arm_does() -> None:
    report = _rnafm_report()
    report["gate"]["clauses"] = T.derive_clauses(report)
    report["gate"]["overall_pass"] = all(report["gate"]["clauses"].values())
    report["gate"]["failed"] = [k for k, v in report["gate"]["clauses"].items() if not v]
    assert T.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True


def test_the_backbone_clause_catches_an_identity_that_contradicts_the_measured_width() -> None:
    """THE failure this clause exists for: a report claiming one backbone while the heads were
    built on the other's hidden state. 640 vs 1280 is the discriminator, and it is *measured*
    off the live module rather than restated from the config.
    """
    report = _rnafm_report()
    # Claim RNA-FM, but the heads were sized for RiNALMo — i.e. RiNALMo was what loaded.
    report["wrap"]["stage2_heads"]["d_model"] = BR.resolve_backbone(
        BR.PRODUCTION_BACKBONE
    ).hidden_size
    assert T.derive_clauses(report)["backbone_pinned"] is False


@pytest.mark.parametrize(
    ("mutate", "why"),
    [
        (
            lambda bb: bb.__setitem__("repo_id", "multimolecule/rnafm-ss"),
            "repo outside the allow-list",
        ),
        (lambda bb: bb.__setitem__("revision", "main"), "a branch, not the pinned commit"),
        (lambda bb: bb.__setitem__("key", "rinalmo-giga"), "key disagrees with its repo_id"),
        (
            lambda bb: bb.__setitem__("loaded_from_registry", False),
            "a caller's fixture, not the pin",
        ),
        (lambda bb: bb.pop("hidden_size"), "identity block missing its width"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_each_way_the_backbone_clause_bites(mutate, why) -> None:
    report = _rnafm_report()
    assert T.derive_clauses(report)["backbone_pinned"] is True, "control must start TRUE"
    mutate(report["wrap"]["backbone"])
    assert T.derive_clauses(report)["backbone_pinned"] is False, why


def test_the_backbone_clause_reads_no_requested_setting() -> None:
    """A clause sourced from `report["config"]` restates the request and can never fail. The
    first draft of `backbone_pinned` did exactly that and this invariant caught it."""
    report = _rnafm_report()
    before = T.derive_clauses(report)
    report["config"] = {}
    assert T.derive_clauses(report) == before


# -- the torch tier: a real RNA-FM wrap ------------------------------------------------
def test_a_tiny_rnafm_wraps_and_takes_one_step(tmp_path) -> None:
    """The composition end-to-end on a tiny same-architecture RNA-FM — no multi-GB download.

    Mirrors `_tiny_backbone`'s RiNALMo equivalent so the two arms are exercised the same way.
    Skips loudly unless the `ml-rnafm` stack is present, because `ml-rna`'s 0.1.0 cannot build
    an `RnaFmConfig` at all.
    """
    _require_rnafm_stack()
    from multimolecule import RnaFmConfig, RnaFmModel

    spec = BR.resolve_backbone(BR.COMPARATOR_BACKBONE)
    cfg = RnaFmConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=spec.vocab_size,
    )
    cfg._attn_implementation = "sdpa"  # CPU tier; FA-2 needs a GPU
    base = RnaFmModel(cfg, add_pooling_layer=False)

    train_cfg = T.Stage2TrainConfig(
        backbone=BR.COMPARATOR_BACKBONE,
        checkpoint_dir=str(tmp_path / "ckpt"),
        report_path=str(tmp_path / "report.json"),
        batch_size=2,
        gradient_checkpointing=False,
    )
    model, wrap = T.build_model(train_cfg, base_model=base)
    # The identity travelled with the model, and the heads were sized off the LIVE width.
    assert wrap["backbone"]["key"] == BR.COMPARATOR_BACKBONE
    assert wrap["backbone"]["loaded_from_registry"] is False, "a fixture must say it is one"
    assert wrap["stage2_heads"]["d_model"] == 64, "heads must follow the tiny fixture's width"
    assert model.d_model == 64


def test_build_model_refuses_a_wrap_whose_identity_is_not_the_one_requested(monkeypatch) -> None:
    """The refactor-drops-the-kwarg failure, made loud.

    `derive_clauses` may not read `report["config"]`, so the agreement between the REQUESTED
    backbone and the one actually built cannot be a gate clause. It is enforced at the call
    instead — and this is what pins that. The stub stands in for a `build_stage2_model` that
    stopped forwarding `backbone=`: the parameter silently reverts to the production default
    and every downstream artifact then carries the wrong identity while training the wrong
    model. Nothing else in the suite would notice, because every individual piece is
    self-consistent.
    """
    torch = _require_torch()

    class _FakeBackbone(torch.nn.Module):
        def gradient_checkpointing_enable(self, **_kwargs):  # pragma: no cover - not reached
            raise AssertionError("gradient checkpointing is off in this fixture")

    class _FakeModel:
        backbone = _FakeBackbone()

    def _fake_build(spec, **kwargs):
        # Reports the PRODUCTION key regardless of what was asked for.
        return _FakeModel(), {
            "backbone": {
                **BR.backbone_summary(BR.resolve_backbone(BR.PRODUCTION_BACKBONE)),
                "loaded_from_registry": True,
            }
        }

    import tbox_finder.stage2.model as M

    monkeypatch.setattr(M, "build_stage2_model", _fake_build)

    cfg = T.Stage2TrainConfig(
        backbone=BR.COMPARATOR_BACKBONE,
        checkpoint_dir=T.default_checkpoint_dir(BR.COMPARATOR_BACKBONE),
        report_path=T.default_report_path(BR.COMPARATOR_BACKBONE),
        gradient_checkpointing=False,
    )
    with pytest.raises(RuntimeError, match="not the one requested"):
        T.build_model(cfg)

    # Positive control: the same stub reporting the REQUESTED key must NOT raise, so the test
    # cannot pass by way of a refusal that fires on everything
    # ([[raises-test-needs-a-positive-control]]).
    def _honest_build(spec, **kwargs):
        return _FakeModel(), {
            "backbone": {
                **BR.backbone_summary(BR.resolve_backbone(BR.COMPARATOR_BACKBONE)),
                "loaded_from_registry": True,
            }
        }

    monkeypatch.setattr(M, "build_stage2_model", _honest_build)
    model, info = T.build_model(cfg)
    assert info["backbone"]["key"] == BR.COMPARATOR_BACKBONE


@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_STAGE2_DATA") != "1",
    reason="runs the real trainer over the DVC-tracked dataset; local/cluster only",
)
def test_the_tiny_run_is_refused_for_BOTH_reasons_and_the_refusal_names_them(tmp_path) -> None:
    """A fixture-backed smoke must fail `backbone_pinned` as well as `full_population`.

    The other `_tiny_run` callers discard `raised` or check a single clause, so without this
    the new clause's behaviour on the full loop would be unasserted — and an unasserted clause
    is one that can silently stop biting. Both reasons are real and neither substitutes for
    the other: `full_population` says the run was truncated, `backbone_pinned` says the weights
    were a fixture. A smoke that lost the second could hand a toy model's numbers to a gate.
    """
    report, raised = _tiny_run(tmp_path)
    clauses = report["gate"]["clauses"]
    assert clauses["full_population"] is False
    assert clauses["backbone_pinned"] is False
    assert report["wrap"]["backbone"]["loaded_from_registry"] is False
    # ...and the run refused rather than exiting 0 with a failed gate.
    assert raised is not None
    assert "backbone_pinned" in str(raised) and "full_population" in str(raised)
    # Positive control on the OTHER half: the identity itself is well-formed, so the clause is
    # failing on the fixture provenance rather than on a malformed block.
    assert report["wrap"]["backbone"]["key"] == BR.PRODUCTION_BACKBONE
    assert (
        report["wrap"]["backbone"]["revision"]
        == BR.resolve_backbone(BR.PRODUCTION_BACKBONE).revision
    )


# ======================================================================================
# Review round 1 (CodeRabbit, PR #134) — every fix gets the guard it was missing
# ======================================================================================
def test_the_slurm_log_directory_is_committed_because_slurm_opens_it_first() -> None:
    """Slurm opens `--output`/`--error` BEFORE the script runs, so `mkdir -p` in the body is
    too late and a fresh cluster checkout loses the job's own logs — the hardest failure to
    diagnose, because the evidence is the thing that is missing. Verified absent on the cluster
    before the fix; the directory is committed now, so this asserts it stays that way.
    """
    text = _SBATCH_RNAFM.read_text()
    out_dirs = set(re.findall(r"^#SBATCH --(?:output|error)=(.+)/[^/]+$", text, re.MULTILINE))
    assert out_dirs, "no #SBATCH --output/--error directives found"
    for d in out_dirs:
        assert (_REPO / d).is_dir(), f"{d} must exist in a fresh checkout"
        keep = _REPO / d / ".gitkeep"
        assert keep.is_file(), f"{d} needs a committed .gitkeep or it will not survive a clone"


def test_the_env_guard_compares_versions_as_numbers_not_strings() -> None:
    """`["0", "10"] < ["0", "2"]` is True, so a string compare rejects multimolecule 0.10.0 with
    a message naming the wrong cause. The guard must compare integers — and this asserts the
    property, not the spelling, by exercising the predicate the sbatch actually contains.
    """
    code = _sbatch_code(_SBATCH_RNAFM)
    predicate = re.search(r"^if (tuple\(int\(p\).+?) < \(0, 2\):$", code, re.MULTILINE)
    assert predicate, "the env guard's version predicate is not the integer-tuple form"
    # ⚠ Evaluate the SBATCH's OWN captured expression. The first cut re-implemented the
    # predicate here and never used `predicate.group(1)`, so the loop asserted that this test
    # file's copy behaves as written — true regardless of what the launcher contains, and the
    # exact opposite of what the docstring claims. The only real coupling was the regex
    # spelling, which is a weaker check than the one being advertised
    # ([[artifact-pinning-test-cannot-see-the-code]]).
    expr = f"({predicate.group(1)}) < (0, 2)"
    for version, want_reject in (
        ("0.1.0", True),
        ("0.0.9", True),
        ("0.2.0", False),
        ("0.10.0", False),
        ("1.0.0", False),
    ):
        # eval of a string this repo's own sbatch supplied, captured two lines above.
        assert eval(expr, {"v": version}) is want_reject, version  # noqa: S307


def test_the_DONE_marker_checks_every_artifact_the_verify_block_promises() -> None:
    """A leg that exits 0 without writing its report must not publish a green marker."""
    code = _sbatch_code(_SBATCH_RNAFM)
    loop = re.search(r"^for f in (.+); do$", code, re.MULTILINE)
    assert loop, "no artifact-check loop before the DONE marker"
    checked = set(loop.group(1).split())
    assert {'"$SCORES"', '"$SCORES_LOO"', '"$EVAL_REPORT"'} <= checked, checked
    # ...and the loop runs BEFORE the marker, or it grades nothing.
    assert code.index("for f in ") < code.index('touch "$DONE"')


@pytest.mark.parametrize(
    ("recorded", "requested", "expect"),
    [
        ("multimolecule/rinalmo-giga", None, "rinalmo-giga"),
        ("multimolecule/rnafm", None, "rnafm"),
        ("multimolecule/rnafm", "rnafm", "rnafm"),
        (None, "rnafm", "rnafm"),  # the escape hatch for a checkpoint predating the field
    ],
)
def test_a_checkpoints_backbone_is_re_derived_from_its_own_evidence(recorded, requested, expect):
    from tbox_finder.stage2.eval import resolve_checkpoint_backbone

    assert resolve_checkpoint_backbone(recorded, requested=requested) == expect


@pytest.mark.parametrize(
    ("recorded", "requested", "why"),
    [
        (None, None, "nothing recorded and nothing requested must RAISE, never default"),
        ("", None, "an empty recorded base is not evidence either"),
        ("multimolecule/rnafm-ss", None, "a base outside the closed allow-list"),
        ("multimolecule/rnafm", "rinalmo-giga", "asked for a base the checkpoint contradicts"),
        ("multimolecule/rinalmo-giga", "rnafm", "the same, the other way round"),
    ],
)
def test_a_checkpoint_whose_backbone_cannot_be_re_derived_is_refused(recorded, requested, why):
    """⚠ The first cut DEFAULTED the last case to production, contradicting its own comment.
    A guessed base is applied to weights it never saw and then recorded in exactly the shape a
    measured value has, so nothing downstream can tell the two apart."""
    from tbox_finder.stage2.eval import resolve_checkpoint_backbone

    with pytest.raises(ValueError):
        resolve_checkpoint_backbone(recorded, requested=requested)


def test_a_P1_15_report_cannot_certify_rinalmo_for_a_wrap_that_adapted_another_backbone():
    """The report's `backbone` block is hardcoded from the production pins, `parity_confirmed`
    included. Before this guard, `build_peft_model(backbone="rnafm")` + `build_report` produced
    a report claiming the parity-confirmed checkpoint while `wrap.backbone.key` said `rnafm` —
    and `validate_report` returned no errors. The contradiction was recorded and never graded.
    """
    from tbox_finder.train import lora_harness as LH

    def wrap(key: str) -> dict[str, Any]:
        return {
            "backbone": {
                **BR.backbone_summary(BR.resolve_backbone(key)),
                "loaded_from_registry": True,
            },
            "n_adapter_sites": 4,
            "applied_lora": {
                "r": LH.LORA_R,
                "lora_alpha": LH.LORA_ALPHA,
                "lora_dropout": LH.LORA_DROPOUT,
                "target_modules": LH.LORA_TARGET_MODULES,
            },
            "gradient_checkpointing": True,
        }

    kwargs = dict(
        attn_backend=LH.ATTN_SDPA,
        attn_reason="fixture",
        evidence={"flash_attn_importable": False, "is_sm86": False},
        supports_fa=False,
        fallback_validated=True,
    )
    # The producer refuses the contradiction outright...
    with pytest.raises(ValueError, match="statement about RiNALMo"):
        LH.build_report(wrap_info=wrap(BR.COMPARATOR_BACKBONE), **kwargs)
    # ...and the PRODUCTION wrap is accepted, so the refusal is not firing on everything
    # ([[raises-test-needs-a-positive-control]]).
    report = LH.build_report(wrap_info=wrap(BR.PRODUCTION_BACKBONE), **kwargs)
    assert report["wrap"]["backbone"]["key"] == BR.PRODUCTION_BACKBONE

    # ...and the VALIDATOR catches the same contradiction in a report read off disk, which is
    # the half a producer-side refusal cannot cover.
    tampered = json.loads(json.dumps(report))
    tampered["wrap"]["backbone"]["repo_id"] = BR.resolve_backbone(BR.COMPARATOR_BACKBONE).repo_id
    problems = LH.validate_report(tampered)
    assert any("wrap.backbone.repo_id" in p for p in problems), problems
    # A report with no wrap.backbone at all is NOT contradicted — the committed P1-15 artifact
    # predates the block, and "unrecorded" must not be graded as "wrong".
    legacy = json.loads(json.dumps(report))
    legacy["wrap"].pop("backbone")
    assert not any("wrap.backbone.repo_id" in p for p in LH.validate_report(legacy))


def test_the_P1_16_SMOKE_report_carries_the_same_binding() -> None:
    """The smoke record hardcodes the identical production backbone block, so it had the
    identical hole. Guarding only the P1-15 pair would have been fixing one of two identical
    things ([[fixed-one-of-two-identical-things]]) — and the smoke report is the one that
    carries the §10.2 condition-(b) VRAM verdict, so a wrong backbone there is a wrong budget.
    """
    from tbox_finder.train import lora_harness as LH

    def wrap(key: str) -> dict[str, Any]:
        return {
            "backbone": {
                **BR.backbone_summary(BR.resolve_backbone(key)),
                "loaded_from_registry": True,
            },
            "n_adapter_sites": 4,
            "applied_lora": {
                "r": LH.LORA_R,
                "lora_alpha": LH.LORA_ALPHA,
                "lora_dropout": LH.LORA_DROPOUT,
                "target_modules": LH.LORA_TARGET_MODULES,
            },
            "gradient_checkpointing": True,
            "n_trainable_params": 10,
            "n_total_params": 100,
        }

    kwargs = dict(
        measured_smoke={"peak_vram_gib": 2.0, "steps": 5, "oom": False},
        attn_selected=LH.ATTN_SDPA,
        attn_used=LH.ATTN_SDPA,
        attn_reason="fixture",
        forward_verified=True,
        forward_error=None,
        evidence={"flash_attn_importable": False, "is_sm86": True},
        supports_fa=False,
        hardware={"name": "NVIDIA RTX A4000", "capability": [8, 6]},
    )
    with pytest.raises(ValueError, match="P1-16 smoke record"):
        LH.build_smoke_report(wrap_info=wrap(BR.COMPARATOR_BACKBONE), **kwargs)
    # Positive control: the production wrap is accepted.
    report = LH.build_smoke_report(wrap_info=wrap(BR.PRODUCTION_BACKBONE), **kwargs)
    assert report["wrap"]["backbone"]["key"] == BR.PRODUCTION_BACKBONE

    # ...and the smoke VALIDATOR catches the contradiction in a report read off disk.
    tampered = json.loads(json.dumps(report))
    tampered["wrap"]["backbone"]["repo_id"] = BR.resolve_backbone(BR.COMPARATOR_BACKBONE).repo_id
    assert any("wrap.backbone.repo_id" in p for p in LH.validate_smoke_report(tampered))
    legacy = json.loads(json.dumps(report))
    legacy["wrap"].pop("backbone")
    assert not any("wrap.backbone.repo_id" in p for p in LH.validate_smoke_report(legacy))


def test_the_attention_reason_names_the_backbone_it_is_about() -> None:
    """A correct decision explained by a sentence about a different model is still a wrong
    record. The pre-ack run of this step's sbatch on the cluster printed *"the pinned RiNALMo
    classes advertise flash-attn"* for an RNA-FM job — the decision was right (it comes from
    `model_supports_flash_attn(key)`, measured per class) and the prose was not. That prose is
    written verbatim into a job log and into a report's `attention.reason`, and prose in a
    provenance record is a claim.
    """
    from tbox_finder.train import lora_harness as LH

    common = dict(flash_attn_importable=True, sm86_confirmed=True, dtype=LH.TRAIN_DTYPE)
    for key in BR.BACKBONE_KEYS:
        backend, reason = LH.select_attention_backend(
            model_supports_flash_attn=True, backbone=key, **common
        )
        assert backend == LH.ATTN_FLASH2
        assert key in reason, reason
        other = next(k for k in BR.BACKBONE_KEYS if k != key)
        assert other not in reason, f"the {key} reason names {other}: {reason}"
        # ⚠ And the CLAIM, not just the name. A sabotage that made every arm assert A10's
        # sm_86 forward verification stayed GREEN against the name check alone, because the
        # verified sentence names no backbone at all. Inheriting a verification the arm does
        # not have is the whole defect — ADR-0002 A10 verified the FA-2 forward for the
        # production backbone and for nothing else.
        claims_verified = "VERIFIED on sm_86" in reason
        assert claims_verified is (key == BR.PRODUCTION_BACKBONE), (
            f"{key} claims_verified={claims_verified}; only {BR.PRODUCTION_BACKBONE} may "
            f"claim A10's sm_86 forward verification. reason: {reason}"
        )
        if key != BR.PRODUCTION_BACKBONE:
            assert "NOT separately forward-verified" in reason, reason

        # ⚠ EVERY branch, not the one that happened to be exercised. The first cut named the
        # arm in two of five reason strings and tested only one of them; the three fallback
        # branches — the ones an operator reads when it went WRONG — recorded no arm identity
        # at all ([[fixed-one-of-two-identical-things]]).
        # Each branch is paired with the fragment its OWN condition must produce. Distinctness
        # alone catches a COLLAPSE but not a SWAP: four messages can stay unique and correctly
        # named while pointing an operator at the wrong cause — "flash-attn does not import"
        # when the real problem is unconfirmed sm_86.
        # ([[symmetric-count-fixture-blind-to-inversion]])
        branches = (
            (
                dict(
                    sm86_confirmed=False,
                    flash_attn_importable=True,
                    model_supports_flash_attn=True,
                    dtype=LH.TRAIN_DTYPE,
                ),
                "sm_86 not confirmed",
            ),
            (
                dict(
                    sm86_confirmed=True,
                    flash_attn_importable=False,
                    model_supports_flash_attn=True,
                    dtype=LH.TRAIN_DTYPE,
                ),
                "flash-attn does not import",
            ),
            (
                dict(
                    sm86_confirmed=True,
                    flash_attn_importable=True,
                    model_supports_flash_attn=False,
                    dtype=LH.TRAIN_DTYPE,
                ),
                "do not advertise flash-attn support",
            ),
            (
                dict(
                    sm86_confirmed=True,
                    flash_attn_importable=True,
                    model_supports_flash_attn=True,
                    dtype="float32",
                ),
                "is not half-precision",
            ),
        )
        seen = set()
        for kw, expected_fragment in branches:
            backend_b, reason_b = LH.select_attention_backend(backbone=key, **kw)
            assert backend_b == LH.ATTN_SDPA, kw
            assert key in reason_b, f"fallback reason omits the arm: {reason_b}"
            assert other not in reason_b, f"fallback reason names {other}: {reason_b}"
            assert expected_fragment in reason_b, (
                f"branch {kw} produced a reason that does not name its own cause "
                f"({expected_fragment!r} absent): {reason_b}"
            )
            seen.add(reason_b)
        assert len(seen) == len(branches), "two fallback branches share a reason string"
        # ...and the fallback branch too, which is the one an operator reads when it went wrong.
        _, fallback_reason = LH.select_attention_backend(
            model_supports_flash_attn=False, backbone=key, **common
        )
        assert key in fallback_reason and other not in fallback_reason, fallback_reason

    # The default stays the production arm, so every pre-A15 caller's recorded prose is
    # unchanged — this must not silently rewrite the committed P1-15/P1-16 reason strings.
    _, default_reason = LH.select_attention_backend(model_supports_flash_attn=True, **common)
    assert BR.PRODUCTION_BACKBONE in default_reason


def test_the_comparator_trains_the_lr_MATCHED_AUX_ZERO_CONTROL_the_scorers_require() -> None:
    """Job 1372 trained the production point cleanly and then BOTH scoring legs refused:

        ValueError: expected exactly one aux_weight=0 arm at lr=0.0001, found [];
        the ablation needs a learning-rate-matched control

    `stage2.eval` resolves `select_arm_pair` before it scores anything, and `gate2 score-loo`
    calls the same selector — so a single-arm checkpoint root cannot be scored by either shipped
    entry point. The control is the enabling condition for the posteriors this step owes, not a
    nice-to-have, and nothing else in the suite would notice its removal until a 4-hour job had
    already trained.
    """
    code = _sbatch_code(_SBATCH_RNAFM)
    grid = re.search(r"^AUX_WEIGHTS=\((.+?)\)$", code, re.MULTILINE)
    assert grid, "the comparator sbatch declares no AUX_WEIGHTS grid"
    weights = [float(w) for w in grid.group(1).split()]
    assert len(weights) == 2, weights
    assert 0.0 in weights, "no aux=0 control — both scoring legs will refuse this root"
    # The production point must be in the pair too, and it is the one read out of conf/.
    aux = float(
        re.search(
            r"^aux_weight:\s*(\S+)", (_CONF / "loss" / "stage2.yaml").read_text(), re.MULTILINE
        ).group(1)
    )
    assert aux in weights, f"the shipped aux_weight {aux} is not one of the trained points"
    # Both points share ONE lr, or they are not a matched pair.
    assert re.search(r'^LR="1e-4"$', code, re.MULTILINE), "the pair must share a single lr"
    # ...and the job refuses to score a half-trained pair rather than grading one arm.
    # ⚠ BOTH clauses, asserted separately. The guard checks the run report AND the checkpoint
    # for each point, and a first cut of this assertion only required the marker to appear
    # somewhere — so renaming one of the two occurrences left it GREEN, satisfied by the
    # survivor ([[sabotage-attribution-names-the-test]] / compound disjuncts bite separately).
    assert code.count("RNAFM_PAIR_INCOMPLETE") >= 2, "the pair guard lost one of its clauses"
    guard = code[code.index("RNAFM_PAIR_INCOMPLETE") : code.index("-m tbox_finder.stage2.eval")]
    assert "${KEY}.json" in guard, "the pair guard no longer checks each point's run report"
    assert "stage2_heads.pt" in guard, "the pair guard no longer checks each point's checkpoint"
    assert code.index("RNAFM_PAIR_INCOMPLETE") < code.index("-m tbox_finder.stage2.eval")


# --------------------------------------------------------------------------------------
# P3-17 review round 5 — the pair the scorers require, pinned ON THE LAUNCH LINE
# --------------------------------------------------------------------------------------
def test_the_launch_line_takes_aux_weight_from_the_LOOP_not_a_literal() -> None:
    """Job 1372 died because a single-arm root cannot be scored: `stage2.eval` and
    `gate2 score-loo` both call `select_arm_pair`, which needs an aux=0 control at the same lr.

    The fix trains both points in one job — but nothing pinned the launch line to the loop
    variable. Hardcoding `loss.aux_weight=1.0` there trains two identical arms, leaves the
    whole comparator suite GREEN, and burns ~25 GPU-minutes before the scorers refuse exactly
    as they did on 1372.
    """
    tokens = _launch_tokens(_SBATCH_RNAFM, "tbox_finder.stage2.train")
    assert 'loss.aux_weight="$AUX_WEIGHT"' in tokens, tokens
    assert 'optim.lr="$LR"' in tokens, tokens
    # No literal aux weight may appear on the launch line at all — a second, hardcoded
    # override would win or shadow depending on order.
    literals = [t for t in tokens if t.startswith("loss.aux_weight=") and "$AUX_WEIGHT" not in t]
    assert literals == [], literals


def test_the_grid_the_loop_iterates_actually_contains_the_pair_the_scorers_need() -> None:
    """The launch line reading `$AUX_WEIGHT` is worth nothing if the grid holds one value.

    `AUX_WEIGHTS=(1.0)` would satisfy every launch-line assertion and still produce the
    single-arm root that killed 1372.
    """
    code = _sbatch_code(_SBATCH_RNAFM)
    grid = re.search(r"^AUX_WEIGHTS=\(([^)]*)\)", code, flags=re.MULTILINE)
    assert grid is not None, "the comparator sbatch must declare an AUX_WEIGHTS grid"
    values = {float(v) for v in grid.group(1).split()}
    assert values == {1.0, 0.0}, values
    # ...and EVERY loop over the points must iterate THAT array, binding the name the launch
    # line reads. There are two — the training loop and the freshness/verification loop — and
    # an `in` check is satisfied by either one alone: a sabotage that rewrote the training
    # loop's header to a literal list stayed green because the second loop still matched
    # ([[fixed-one-of-two-identical-things]]).
    loops = code.count('for AUX_WEIGHT in "${AUX_WEIGHTS[@]}"; do')
    assert loops == 2, (
        f"expected both point loops to iterate the declared grid, found {loops}; a loop over a "
        "literal list makes the grid a declaration nothing binds to"
    )
    # Both points must land in DISTINCT arm directories, or the second overwrites the first
    # and the pair collapses to one ([[symmetric-count-fixture-blind-to-inversion]]).
    assert 'KEY="aux${AUX_WEIGHT}_lr${LR}"' in code
    assert 'CKPT_DIR="$CKPT_ROOT/${KEY}"' in code
    assert 'REPORT="$OUT_DIR/${KEY}.json"' in code


def test_the_launchers_MEASURED_figures_are_the_committed_sizing_reports() -> None:
    """The header presents four VRAM figures and two step-times as "MEASURED ... on a cluster
    A4000". Prose in a provenance record is a claim, and this file argues that elsewhere.

    This test exists because the fix for that claim was written, verified, and then **lost
    before it reached a commit** — the reviewer's thread stayed open against a defect I had
    reported as fixed, and only re-reading the pushed bytes found it
    ([[verify-the-line-you-ship]]). A number checked against its artifact cannot go stale
    quietly, and cannot silently vanish either.
    """
    report = json.loads(
        (_REPO / "reports" / "p3" / "stage2_rnafm_sizing.json").read_text(encoding="utf-8")
    )
    measured = {
        (m["batch_size"], m["regime"]): m
        for m in report["measurements"]
        if not m.get("oom") and m.get("peak_vram_gib")
    }
    assert measured, "the committed sizing report carries no usable measurement"

    header = _SBATCH_RNAFM.read_text()
    block = re.search(
        r"^# MEASURED afterwards.*?(?=^# So the pinned)", header, re.MULTILINE | re.DOTALL
    )
    assert block is not None, "the launcher's MEASURED block has moved or been removed"
    quoted = {
        (int(batch), regime): (float(gib), float(ms) if ms else None)
        for batch, regime, gib, ms in re.findall(
            r"batch (\d+) (worst_case|typical)\s+([\d.]+) GiB(?:\s*/\s*([\d.]+) ms)?",
            block.group(0),
        )
    }
    assert len(quoted) == 4, f"expected four quoted figures, parsed {sorted(quoted)}"

    for key, (gib, ms) in sorted(quoted.items()):
        assert key in measured, f"the header quotes {key} and the report does not measure it"
        assert gib == pytest.approx(measured[key]["peak_vram_gib"], abs=5e-5), (
            f"header says {key} peaked at {gib} GiB; the committed report says "
            f"{measured[key]['peak_vram_gib']}"
        )
        if ms is not None:
            steps = measured[key]["step_ms"][1:]  # step 0 is warm-up, as the header says
            assert ms == pytest.approx(sum(steps) / len(steps), abs=0.05), (
                f"header says {key} ran at {ms} ms; the report's steady-state mean is "
                f"{sum(steps) / len(steps):.1f}"
            )

    # ...and the headroom sentence quotes the same batch-4 worst case to 2 decimals.
    headroom = re.search(r"measured: ([\d.]+) of 15\.6 GiB at batch 4", header)
    assert headroom is not None, "the batch-4 headroom claim has moved"
    assert float(headroom.group(1)) == pytest.approx(
        measured[(4, "worst_case")]["peak_vram_gib"], abs=5e-3
    )
