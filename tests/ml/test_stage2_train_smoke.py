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
import os
from pathlib import Path
from typing import Any

import pytest

from tbox_finder.stage2 import losses as L
from tbox_finder.stage2 import train as T

_REPO = Path(__file__).resolve().parents[2]
_CONF = _REPO / "conf"
_TRAIN_CONF = _CONF / "train" / "stage2.yaml"
_OPTIM_CONF = _CONF / "optim" / "stage2.yaml"
_SBATCH = _REPO / "slurm" / "p3" / "stage2_lora_finetune.sbatch"
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
            "n_steps": 1420,
            "expected_n_steps": 1420,
            "n_optimizer_steps": 1420,
            "batches_per_epoch_per_rank": 142,
            "world_size": 8,
            "warmup_steps": 85,
            "total_scheduled_steps": 1420,
            "elapsed_seconds": 2100.0,
        },
        wrap={
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
        },
        val={"history": [], "best_total": 0.4, "best_epoch": 7},
        checkpoint={
            "adapter_dir": "data/processed/checkpoints/stage2_rinalmo/lora_adapter",
            "adapter_bytes": 51_000_000,
            "heads_bytes": 800_000,
            "n_saved_parameters": 13_040_268,
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
            "gradient_checkpointing_verified",
        ),
        (
            lambda r: r["wrap"].__setitem__("checkpoint_use_reentrant", True),
            "gradient_checkpointing_verified",
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
    assert "#SBATCH --array=0-5%1" in text  # serialised while one gpu node is unhealthy
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
    # `--exclude` is a dated carve-out. It must carry BOTH a stated removal condition and a
    # rationale, so it cannot outlive its reason — the first version cited a node fault that
    # has since been fixed, and the file would have gone on asserting it.
    if any("--exclude" in ln for ln in directives):
        assert "EXCLUDE RATIONALE" in text
        assert "REMOVE THIS" in text
        assert "2026-" in text  # dated, so a reader can tell how stale the reason is


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

    blocks = [
        (path, match.group(1))
        for path in sorted((_REPO / "slurm").rglob("*.sbatch"))
        for match in re.finditer(
            r"^PYTHONPATH=\S* python -c '\n(.*?)\n'$", path.read_text(), re.S | re.M
        )
    ]
    # Emptiness guard: a regex that matched nothing would make this vacuously green.
    assert blocks, "no inline `python -c` blocks discovered in any sbatch at all"
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
    tree = ast.parse(inspect.cleandoc(source.split("\n", 1)[1]) if False else source.lstrip())
    text = ast.dump(tree)
    assert "validate_report" in text
    body = source[source.index("if not primary:") :]
    assert body.index("return report") < body.index("validate_report"), (
        "the non-primary early return must precede validate_report, or ranks without the "
        "evidence will judge rank 0's artifacts again"
    )
