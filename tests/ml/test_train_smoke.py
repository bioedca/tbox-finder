"""P2-04 — the Stage-1 training-entrypoint smoke (imp.md P2-04 validation gate).

Three tiers, fail-closed, mirroring `test_eval_gate.py` / `test_lora_smoke.py`:

1. **Pure tier** (no torch / hydra / yaml — runs and BLOCKS in bare CI): the entrypoint's
   decidable logic. Config validation, the DDP shard, the class-count arithmetic against
   hand-computed values, the `conf/train/stage1.yaml` wiring, and the report validator —
   each guard proven to bite in **both** directions (a clean pass *and* a violating fail),
   per §8.7.
2. **Hydra tier** (`TBOX_REQUIRE_TRAIN_HYDRA`): the config really composes — the four groups
   resolve and `_cfg_from_mapping` flattens them into the dataclass. CI installs no hydra,
   so this var is deliberately **not** armed there.
3. **Torch tier** (`TBOX_REQUIRE_TRAIN_TORCH`): the gradient-checkpointing wiring and, on
   CUDA, one real train step end to end. CI installs no torch — **its own var**, never
   folded into the pure tier's. That exact folding was the P1-16 landmine: one var guarding
   both a pure-JSON tier and a torch tier means arming it in CI fails every run.

None of this grades the model. `is_science=false`: the smoke asserts the entrypoint
*composes and runs*, never that it *learns* (§10.3). GATE-4 is P2-14, on the real split.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pytest

from tbox_finder.labels import CLASS_ORDER
from tbox_finder.train import train_stage1 as T

_REPO = Path(__file__).resolve().parents[2]
_TRAIN_CONF = _REPO / "conf" / "train" / "stage1.yaml"
_OPTIM_CONF = _REPO / "conf" / "optim" / "stage1.yaml"
_REPORT = _REPO / T.DEFAULT_REPORT


def _fail_or_skip(var: str, reason: str) -> None:
    if os.environ.get(var) == "1":
        pytest.fail(f"{var}=1 but the tier is unrunnable: {reason}")
    pytest.skip(reason)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — pure logic (bare CI)
# ═══════════════════════════════════════════════════════════════════════════
def test_config_rejects_impossible_values() -> None:
    """__post_init__ must bite on each guarded field, and accept a sane config."""
    T.Stage1TrainConfig()  # clean pass — the guard is not a blanket reject
    for kwargs in (
        {"epochs": 0},
        {"batch_size": 0},
        {"lr": -1e-4},
        {"gamma": -1.0},
        {"weight_decay": -0.01},
        {"wandb_mode": "sync"},
    ):
        with pytest.raises(ValueError):
            T.Stage1TrainConfig(**kwargs)


def test_config_rejects_bool_as_a_number() -> None:
    """`isinstance(True, int)` is True — a bool lr would otherwise sail through (P1-16)."""
    with pytest.raises(ValueError):
        T.Stage1TrainConfig(lr=True)


def _fake_dataset(label_rows: list[list[int]]) -> object:
    """A stand-in exposing only what compute_class_counts uses: len + window_at().labels.

    Not a mocked *pipeline* (§8.7) — this pins the counting **arithmetic** against a
    hand-computed expectation, the `metrics.py` hand-computed-kernel idiom. The real
    dataset is exercised in the torch/data tier below.
    """
    import numpy as np

    class _W:
        def __init__(self, labels: list[int]) -> None:
            self.labels = np.asarray(labels, dtype=np.int16)

    class _DS:
        def __len__(self) -> int:
            return len(label_rows)

        def window_at(self, index: int, occurrence: int = 0) -> _W:
            assert (
                occurrence == 0
            ), "counts are pinned at occurrence 0 (not the same as a deterministic lead)"
            return _W(label_rows[index])

    return _DS()


def test_class_counts_are_hand_computed_and_exclude_ignore_index() -> None:
    """Counts over the stream, IGNORE_INDEX excluded, in CLASS_ORDER order."""
    ds = _fake_dataset(
        [
            [0, 0, 1, T.IGNORE_INDEX, T.IGNORE_INDEX],
            [2, 2, 2, 0, T.IGNORE_INDEX],
            [7, 5, 5, 0, 0],
        ]
    )
    counts = T.compute_class_counts(ds)
    # background 0: 2 + 1 + 2 = 5 | Stem_I 1: 1 | Specifier 2: 3 | Antiterm 5: 2 | Discrim 7: 1
    assert counts == (5, 1, 3, 0, 0, 2, 0, 1)
    assert sum(counts) == 12, "the 3 IGNORE_INDEX positions must not be counted"
    assert len(counts) == len(CLASS_ORDER)


def test_class_counts_respect_max_records() -> None:
    ds = _fake_dataset([[0, 0], [1, 1], [2, 2]])
    assert T.compute_class_counts(ds, max_records=2) == (2, 2, 0, 0, 0, 0, 0, 0)


def test_class_counts_fail_closed_on_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="empty stream"):
        T.compute_class_counts(_fake_dataset([]))
    with pytest.raises(ValueError, match="all zero"):
        T.compute_class_counts(_fake_dataset([[T.IGNORE_INDEX, T.IGNORE_INDEX]]))
    with pytest.raises(ValueError, match="out of range"):
        T.compute_class_counts(_fake_dataset([[0, 8]]))  # 8 == NUM_CLASSES


class _FakeSampler:
    """Mimics WeightedIndexSampler's contract.

    Two properties are load-bearing and an earlier draft got both wrong:

    - **The stream must change with the epoch.** The real sampler reshuffles on `set_epoch`;
      a fake whose `__iter__` ignores its epoch is *structurally incapable* of catching a
      `set_epoch` that does nothing, which is exactly the regression that matters (every
      epoch replaying one draw stream = the P2-01 memorisation failure).
    - **occurrence is the global draw ordinal** (unique, monotone over the epoch), while
      *indices* repeat — the real sampler draws with replacement. The earlier fake inverted
      this (`occurrence = i % 3`, unique indices), which let a shard test pass against an
      operator that invented `occurrence` from the rank.
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        # Index repeats (draw-with-replacement, epoch-dependent); occurrence is the unique
        # global ordinal — the real shape, so a rank/occurrence alias cannot survive.
        return iter([((i * 7 + self.epoch * 3) % 5, i) for i in range(self._n)])


def test_ddp_shards_are_equal_length_or_ddp_deadlocks() -> None:
    """Every rank must yield the SAME number of draws.

    23 draws over 4 ranks is the case that bites: a `rank::world_size` stride alone shards
    6/6/6/5, the short rank runs one fewer backward, drops out of the gradient all-reduce,
    and the other three block forever — the job hangs rather than failing. So the stream is
    truncated to a multiple of world_size before striding.
    """
    world = 4
    shards = [
        list(iter(T.ShardedSampler(_FakeSampler(23), rank=r, world_size=world)))
        for r in range(world)
    ]
    assert (
        len({len(s) for s in shards}) == 1
    ), f"ragged shards deadlock DDP: {[len(s) for s in shards]}"
    assert all(len(s) == 5 for s in shards), "23 // 4 == 5 draws per rank; the 3 remainder drop"
    # __len__ must agree with what __iter__ actually yields, or a DataLoader mis-sizes the epoch.
    for r, shard in enumerate(shards):
        assert len(shard) == len(T.ShardedSampler(_FakeSampler(23), rank=r, world_size=world))


def test_ddp_shards_are_disjoint_and_a_subset_of_the_stream() -> None:
    """No draw may land on two ranks, and none may be invented."""
    world = 4
    full = set(iter(_FakeSampler(23)))
    union = [
        d
        for r in range(world)
        for d in iter(T.ShardedSampler(_FakeSampler(23), rank=r, world_size=world))
    ]
    assert len(union) == len(set(union)), "a draw appeared on two ranks"
    assert set(union) <= full, "a shard yielded a draw that is not in the stream"
    assert len(union) == 20, "exactly the truncated stream is covered (23 -> 20)"


def test_ddp_shards_are_exact_when_the_stream_divides_evenly() -> None:
    """No draw is dropped when len % world_size == 0 — truncation is not over-eager."""
    world = 4
    full = sorted(iter(_FakeSampler(24)))
    union = sorted(
        d
        for r in range(world)
        for d in iter(T.ShardedSampler(_FakeSampler(24), rank=r, world_size=world))
    )
    assert union == full


def test_ddp_shard_set_epoch_reaches_the_underlying_sampler() -> None:
    """`set_epoch` must advance the real stream — `def set_epoch(self, e): pass` is a regression.

    Nothing tested this before, so gutting the body to `pass` was green. It matters: the
    module's own docstring claims "which draws are dropped changes every epoch, because
    set_epoch reshuffles the underlying stream", and `train_stage1` calls it per epoch with
    the comment "both must advance, or augmentation/draws freeze (P2-01)". With a stub, every
    epoch replays one draw stream, the same draws are dropped forever, and the 9× oversampled
    class-II records repeat identically — the exact memorisation P2-01 was designed against.
    """
    inner = _FakeSampler(12)
    shard = T.ShardedSampler(inner, rank=0, world_size=2)

    shard.set_epoch(0)
    epoch0 = list(iter(shard))
    shard.set_epoch(1)
    epoch1 = list(iter(shard))

    assert inner.epoch == 1, "set_epoch must reach the underlying sampler, not be swallowed"
    assert epoch0 != epoch1, "the draw stream must change between epochs"
    assert len(epoch0) == len(epoch1), "but the per-rank count must stay fixed (DDP)"


def test_ddp_shard_preserves_the_index_occurrence_tuples() -> None:
    """The occurrence ordinal is the dataset's per-draw RNG key — dropping it collapses an
    oversampled record to identical copies (the P2-01 memorisation failure).

    The occurrences in a shard must be **distinct**, which is what makes this bite. An earlier
    draft used `world_size=3` against a fixture whose occurrence was `i % 3` — so every
    occurrence in the shard equalled the rank, and an operator that simply *invented*
    `occurrence = rank` passed. The fixture now carries the real shape (unique global
    ordinals) and the stride is coprime to nothing in it.
    """
    inner = _FakeSampler(12)
    shard = list(iter(T.ShardedSampler(inner, rank=1, world_size=4)))
    expected = [d for i, d in enumerate(iter(_FakeSampler(12))) if i % 4 == 1]
    assert shard == expected
    occurrences = [occ for _, occ in shard]
    assert occurrences == [1, 5, 9], "the global draw ordinal must survive the shard verbatim"
    assert len(set(occurrences)) == len(occurrences), "occurrences must stay distinct"
    assert all(isinstance(d, tuple) and len(d) == 2 for d in shard)


def test_ddp_shard_single_process_is_the_identity() -> None:
    full = list(iter(_FakeSampler(7)))
    assert list(iter(T.ShardedSampler(_FakeSampler(7), rank=0, world_size=1))) == full


def test_ddp_shard_rejects_bad_rank() -> None:
    for rank, world in ((4, 4), (-1, 4), (0, 0)):
        with pytest.raises(ValueError):
            T.ShardedSampler(_FakeSampler(3), rank=rank, world_size=world)


def test_ddp_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (T._RANK_ENV, T._WORLD_SIZE_ENV, T._LOCAL_RANK_ENV):
        monkeypatch.delenv(var, raising=False)
    assert (T.ddp_rank(), T.ddp_world_size(), T.ddp_local_rank()) == (0, 1, 0)
    assert T.is_primary()
    monkeypatch.setenv(T._RANK_ENV, "3")
    monkeypatch.setenv(T._WORLD_SIZE_ENV, "8")
    monkeypatch.setenv(T._LOCAL_RANK_ENV, "3")
    assert (T.ddp_rank(), T.ddp_world_size(), T.ddp_local_rank()) == (3, 8, 3)
    assert not T.is_primary()
    monkeypatch.setenv(T._WORLD_SIZE_ENV, "not-an-int")
    with pytest.raises(ValueError):
        T.ddp_world_size()


def test_pythonhashseed_is_verified_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """It must be inherited from the launcher — setting it in-process is a no-op (§8.3).

    CPython fixes hash randomisation at interpreter startup, so an in-process assignment
    changes nothing and merely *looks* compliant. The entrypoint therefore verifies.
    """
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        T.check_pythonhashseed(0)
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    with pytest.raises(RuntimeError, match="pins"):
        T.check_pythonhashseed(0)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    T.check_pythonhashseed(0)  # clean pass — the guard is not a blanket reject


def test_class_counts_scope_clause_requires_a_self_consistent_scope() -> None:
    """§10.2: `full_stream` must FOLLOW from the arithmetic, not be asserted beside it."""
    report = _valid_report()
    assert T.derive_clauses(report)["class_counts_from_stream"] is True
    # A slice claiming to be the whole fold.
    report["class_counts_scope"] = {
        "n_records": 64,
        "n_training_fold_records": 8303,
        "full_stream": True,
        "training_fold_only": True,
        "occurrence": 0,
    }
    assert T.derive_clauses(report)["class_counts_from_stream"] is False
    # A missing scope must not pass.
    del report["class_counts_scope"]
    assert T.derive_clauses(report)["class_counts_from_stream"] is False
    # More records than the fold contains is incoherent.
    report["class_counts_scope"] = {
        "n_records": 9000,
        "n_training_fold_records": 8303,
        "full_stream": False,
        "training_fold_only": True,
        "occurrence": 0,
    }
    assert T.derive_clauses(report)["class_counts_from_stream"] is False


def test_backbone_blocks_fails_loud_rather_than_reporting_a_no_op() -> None:
    """An empty/absent stack must raise, not return 0 blocks and look successful."""

    class _NoLayers:
        pass

    class _Empty:
        class backbone:  # noqa: N801
            layers: list = []

    with pytest.raises(AttributeError, match="backbone.layers"):
        T.backbone_blocks(_NoLayers())
    with pytest.raises(ValueError, match="empty"):
        T.backbone_blocks(_Empty())


# ── the report contract: every clause re-derived, never echoed ──────────────────────────
def _valid_report() -> dict:
    return {
        "class_counts_scope": {
            "n_records": 64,
            "n_training_fold_records": 8303,
            "full_stream": False,
            "training_fold_only": True,
            "occurrence": 0,
            "weighted_draw_stream": False,
            # This report claims a scored selection-val fold, so the run that produced it
            # must have withheld that fold from training — the P2-09 cross-check the eval
            # clause reads (see tests/unit/test_eval_val_clause.py::
            # test_gate_is_false_when_the_run_trained_on_the_eval_fold).
            "selection_val_excluded": True,
        },
        "schema_version": "1",
        "step": "P2-04",
        "generated_by": T.GENERATED_BY,
        "adr": "ADR-0002",
        "env_lock": T.ENV_LOCK,
        "is_science": False,
        "gate4_graded": False,
        "backbone": {"repo_id": "x", "revision": "y"},
        "gradient_checkpointing": {
            "requested": True,
            "n_blocks": 16,
            "n_blocks_wrapped": 16,
            "hf_flag_supported": False,
        },
        "steps": {"n_steps": 2, "losses": [1.6, 1.5], "world_size": 1},
        "class_counts": [10, 2, 1, 1, 1, 1, 1, 1],
        "grads_finite": True,
        # P2-06a: the validation ladder. A report claiming an eval must carry the
        # disjointness evidence for it — see tests/unit/test_eval_val_clause.py for the
        # clause's own tests (incl. the absence branch this fixture's sibling covers).
        "eval_requested": True,
        "eval_scope": {
            "fold_scope": "selection_val",
            "n_records_scored": 830,
            "n_blocks": 469,
            "full_fold": True,
            "eval_max_records": None,
            "leakage": {
                "n_designated_loo_holdout": 0,
                "n_not_nested_train": 0,
                "shared_record_ids_with_inner_train": 0,
                "shared_cluster_ids_with_inner_train": 0,
                "n_inner_train_records": 7472,
            },
        },
        "eval_metrics": {
            "eval_split": "selection_val",
            "n_positions": 2_100_000,
            "gate4_core_min_f1": {"min_f1": 0.31, "per_element_f1": {}},
        },
        "provenance": {"git_sha": "abc", "env_lock_sha256": "def", "seed": 42},
        "diagnostics": {"pinned": False, "config": {"seed": 42}},
        "gate": {},
    }


def _valid_report_without_eval() -> dict:
    """The `eval_val=False` shape: no eval requested, and none claimed.

    P2-06a's clause is total — this is its other branch, and it must PASS. A run that
    deliberately skips the val fold (a pure footprint/timing run, as P2-05's sizing points
    are) is legitimate; what is illegitimate is *claiming* an eval without the evidence.
    """
    rep = _valid_report()
    rep["eval_requested"] = False
    rep["eval_scope"] = None
    rep["eval_metrics"] = None
    return rep


def _sealed(report: dict) -> dict:
    clauses = T.derive_clauses(report)
    report["gate"] = {**clauses, "overall_pass": all(clauses.values())}
    return report


def test_a_valid_report_passes_and_all_clauses_hold() -> None:
    report = _sealed(_valid_report())
    assert T.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True


def test_a_valid_report_without_an_eval_also_passes() -> None:
    """P2-06a's absence branch, through the real gate — not the predicate."""
    report = _sealed(_valid_report_without_eval())
    assert T.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True
    assert report["gate"]["eval_val_scored_on_disjoint_fold"] is True


def test_a_report_claiming_an_eval_without_evidence_fails_the_gate() -> None:
    """The P2-05 shape, guarded at the level that ships: an eval that silently no-ops
    leaves no scope, and the gate must refuse it rather than certify the absence."""
    rep = _valid_report()
    rep["eval_scope"] = None
    sealed = _sealed(rep)
    assert sealed["gate"]["eval_val_scored_on_disjoint_fold"] is False
    assert sealed["gate"]["overall_pass"] is False


def test_a_report_whose_val_set_touched_training_fails_the_gate() -> None:
    """The defect the whole rung exists to prevent, asserted end-to-end."""
    rep = _valid_report()
    rep["eval_scope"]["leakage"]["shared_cluster_ids_with_inner_train"] = 7
    sealed = _sealed(rep)
    assert sealed["gate"]["eval_val_scored_on_disjoint_fold"] is False
    assert sealed["gate"]["overall_pass"] is False


def test_a_report_whose_val_set_was_the_loo_holdout_fails_the_gate() -> None:
    """P2-06a's actual defect, asserted end-to-end: a fold disjoint from train, fully
    scored, real metrics — and 778/880 of it the leave-one-order-out headline."""
    rep = _valid_report()
    rep["eval_scope"]["leakage"]["n_designated_loo_holdout"] = 778
    sealed = _sealed(rep)
    assert sealed["gate"]["eval_val_scored_on_disjoint_fold"] is False
    assert sealed["gate"]["overall_pass"] is False


def test_validator_catches_a_clause_fabricated_true() -> None:
    """The P1-15/P1-16 lesson: `all(clauses)` cannot catch a clause asserted TRUE.

    Here the evidence says 0 of 16 blocks were wrapped — a no-op wrap — while the stored
    clause claims the checkpointing was applied. The validator must re-derive and object.
    """
    report = _sealed(_valid_report())
    report["gradient_checkpointing"]["n_blocks_wrapped"] = 0
    report["gate"]["gradient_checkpointing_applied"] = True  # the lie
    report["gate"]["overall_pass"] = True
    problems = T.validate_report(report)
    assert any("gradient_checkpointing_applied" in p for p in problems), problems


def test_p2_04_report_without_timing_stays_valid() -> None:
    """The P2-05 timing keys are OPTIONAL.

    P2-04's committed artifact predates them and records what P2-04 measured; adding a key
    to the schema must not retroactively invalidate it — the only alternative would be to
    regenerate it, which would forge a measurement.
    """
    report = _sealed(_valid_report())
    assert "step_seconds" not in report["steps"]
    assert T.validate_report(report) == []


def test_timing_length_must_match_n_steps() -> None:
    """These lists are the denominator of every extrapolated GPU-hour.

    A length that disagrees with `n_steps` would mis-scale the budget while the report
    still looked internally consistent.
    """
    report = _sealed(_valid_report())
    report["steps"]["step_seconds"] = [0.1]  # n_steps is 2
    problems = T.validate_report(report)
    assert any("step_seconds" in p and "n_steps" in p for p in problems), problems


def test_timing_rejects_bools() -> None:
    """`isinstance(True, int)` is True and `True + 0.0 == 1.0` — a bool would read as a
    1-second step and sail through a naive numeric check (the P1-15/P1-16 lesson)."""
    report = _sealed(_valid_report())
    report["steps"]["step_seconds"] = [0.1, True]
    problems = T.validate_report(report)
    assert any("step_seconds" in p and "non-bool" in p for p in problems), problems


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5])
def test_timing_rejects_non_finite_or_negative(bad: float) -> None:
    report = _sealed(_valid_report())
    report["steps"]["step_seconds"] = [0.1, bad]
    problems = T.validate_report(report)
    assert any("step_seconds" in p for p in problems), problems


def test_timing_rejects_a_non_list() -> None:
    report = _sealed(_valid_report())
    report["steps"]["batch_wait_seconds"] = 0.5
    assert any("batch_wait_seconds" in p for p in T.validate_report(report))


def test_well_formed_timing_passes() -> None:
    report = _sealed(_valid_report())
    report["steps"]["step_seconds"] = [0.21, 0.19]
    report["steps"]["batch_wait_seconds"] = [0.01, 0.02]
    assert T.validate_report(report) == []


def test_build_report_plumbs_the_timing_kwargs_through() -> None:
    """`build_report(step_seconds=...)` must reach `steps.step_seconds`.

    Torch-free on purpose: the end-to-end emit check needs CUDA, so without this the whole
    plumbing would be untested wherever CUDA is absent — including CI.
    """
    report = T.build_report(
        cfg=T.Stage1TrainConfig(),
        class_counts=[10, 2, 1, 1, 1, 1, 1, 1],
        counts_scope={"n_records": 1, "n_training_fold_records": 1, "full_stream": True},
        n_blocks=16,
        n_blocks_wrapped=16,
        hf_flag_supported=False,
        losses=[1.6, 1.5],
        grads_finite=True,
        world_size=1,
        wandb_run_id=None,
        step_seconds=[0.21, 0.19],
        batch_wait_seconds=[0.01, 0.02],
    )
    assert report["steps"]["step_seconds"] == [0.21, 0.19]
    assert report["steps"]["batch_wait_seconds"] == [0.01, 0.02]
    assert T.validate_report(report) == []


def test_build_report_omits_timing_keys_when_not_instrumented() -> None:
    """Omitted, not `[]` — "not measured" must stay distinguishable from "zero steps"."""
    report = T.build_report(
        cfg=T.Stage1TrainConfig(),
        class_counts=[10, 2, 1, 1, 1, 1, 1, 1],
        counts_scope={"n_records": 1, "n_training_fold_records": 1, "full_stream": True},
        n_blocks=16,
        n_blocks_wrapped=16,
        hf_flag_supported=False,
        losses=[1.6],
        grads_finite=True,
        world_size=1,
        wandb_run_id=None,
    )
    assert "step_seconds" not in report["steps"]
    assert "batch_wait_seconds" not in report["steps"]


def test_a_no_op_wrap_does_not_satisfy_the_checkpointing_clause() -> None:
    """0 blocks wrapped while checkpointing was requested is exactly the §10.3 stub."""
    report = _valid_report()
    report["gradient_checkpointing"]["n_blocks_wrapped"] = 0
    assert T.derive_clauses(report)["gradient_checkpointing_applied"] is False
    report["gradient_checkpointing"]["n_blocks_wrapped"] = 8  # partial wrap
    assert T.derive_clauses(report)["gradient_checkpointing_applied"] is False


def test_not_requesting_checkpointing_satisfies_the_clause_vacuously() -> None:
    report = _valid_report()
    report["gradient_checkpointing"] = {"requested": False, "n_blocks": 16, "n_blocks_wrapped": 0}
    assert T.derive_clauses(report)["gradient_checkpointing_applied"] is True


def test_a_missing_or_non_bool_request_fails_closed() -> None:
    report = _valid_report()
    report["gradient_checkpointing"] = {"n_blocks": 16, "n_blocks_wrapped": 16}
    assert T.derive_clauses(report)["gradient_checkpointing_applied"] is False
    report["gradient_checkpointing"]["requested"] = "yes"
    assert T.derive_clauses(report)["gradient_checkpointing_applied"] is False


def test_provenance_clause_needs_every_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLAUDE.md §11. seed==0 alone must NOT satisfy it — the `a and b or c` precedence trap."""
    report = _valid_report()
    report["provenance"] = {"git_sha": None, "env_lock_sha256": "d", "seed": 0}
    assert T.derive_clauses(report)["provenance_complete"] is False
    report["provenance"] = {"git_sha": "a", "env_lock_sha256": None, "seed": 42}
    assert T.derive_clauses(report)["provenance_complete"] is False
    report["provenance"] = {"git_sha": "a", "env_lock_sha256": "d", "seed": 0}
    report["diagnostics"] = {"pinned": False, "config": {"seed": 0}}  # both copies agree
    assert T.derive_clauses(report)["provenance_complete"] is True, "seed 0 is a legal seed"


def test_provenance_clause_rejects_an_empty_sha() -> None:
    """`isinstance("", str)` is True — a blank SHA is a missing SHA wearing the right type."""
    report = _valid_report()
    for blank in ("", "   "):
        report["provenance"] = {"git_sha": blank, "env_lock_sha256": "d", "seed": 42}
        assert T.derive_clauses(report)["provenance_complete"] is False
        report["provenance"] = {"git_sha": "a", "env_lock_sha256": blank, "seed": 42}
        assert T.derive_clauses(report)["provenance_complete"] is False


def test_provenance_clause_cross_checks_the_duplicated_seed() -> None:
    """The seed is recorded twice; disagreeing copies must not certify (P1-15/P1-16)."""
    report = _valid_report()
    report["provenance"] = {"git_sha": "a", "env_lock_sha256": "d", "seed": 42}
    report["diagnostics"] = {"pinned": False, "config": {"seed": 7}}  # disagrees
    assert T.derive_clauses(report)["provenance_complete"] is False
    report["diagnostics"] = {"pinned": False, "config": {"seed": 42}}
    assert T.derive_clauses(report)["provenance_complete"] is True
    report["diagnostics"] = {"pinned": False, "config": {}}  # absent ⇒ fail closed
    assert T.derive_clauses(report)["provenance_complete"] is False


def test_provenance_clause_separates_staged_data_dirt_from_dirty_code() -> None:
    """P2-09: `git_dirty` was ignored outright, so job 671 certified complete provenance
    beside `git_dirty: true`. True in fact — only the re-staged corpus parquet differed —
    but the gate did not establish it; a run with modified TRAINING CODE certified
    identically, with a `git_sha` naming a commit that does not contain what ran.

    Requiring a clean tree instead would fail every cluster run: the cluster has no
    git-lfs, so each run re-stages `split_assignments.parquet` over its committed pointer
    and the tree is dirty by construction. Hence classification, not prohibition.
    """
    report = _valid_report()
    base = {"git_sha": "a", "env_lock_sha256": "d", "seed": 42}

    # The real job-671 shape: dirty, but only the staged corpus → certifies.
    report["provenance"] = {
        **base,
        "git_dirty": True,
        "git_dirty_paths": ["data/processed/splits/split_assignments.parquet"],
    }
    assert T.derive_clauses(report)["provenance_complete"] is True

    # One dirty CODE path anywhere in the list voids it, even beside legitimate data dirt.
    for code_path in (
        "src/tbox_finder/train/train_stage1.py",
        "conf/train/stage1.yaml",
        "slurm/p2/train_production.sbatch",
    ):
        report["provenance"] = {
            **base,
            "git_dirty": True,
            "git_dirty_paths": ["data/processed/splits/split_assignments.parquet", code_path],
        }
        assert T.derive_clauses(report)["provenance_complete"] is False, code_path

    # Dirty with NO path list is unverifiable, not innocent — fail closed. This is the
    # exact pre-P2-09 shape, and it must no longer certify.
    report["provenance"] = {**base, "git_dirty": True}
    assert T.derive_clauses(report)["provenance_complete"] is False
    report["provenance"] = {**base, "git_dirty": True, "git_dirty_paths": []}
    assert T.derive_clauses(report)["provenance_complete"] is False
    report["provenance"] = {**base, "git_dirty": True, "git_dirty_paths": "data/x"}
    assert T.derive_clauses(report)["provenance_complete"] is False

    # A clean tree still certifies, and the pre-existing fixtures (no git_dirty key) are
    # deliberately unaffected — see the BOUNDARY note in _provenance_complete.
    report["provenance"] = {**base, "git_dirty": False}
    assert T.derive_clauses(report)["provenance_complete"] is True


def test_git_dirty_paths_parses_nul_delimited_porcelain() -> None:
    """`--porcelain=v1 -z`: NUL-separated `XY <path>`, renames carry an extra origin field.

    The line-based first cut split on a literal `" -> "`, which a FILENAME may legally
    contain — it would have reported a phantom path and, worse, could have turned a dirty
    source file into a `data/`-prefixed one and certified provenance (CodeRabbit, P2-09).
    """
    seen = {}

    def fake_git(*args: str):
        seen["args"] = args
        return (
            " M data/processed/splits/split_assignments.parquet\0"
            "M  src/tbox_finder/train/train_stage1.py\0"
            # rename: destination first, then the origin field that must be consumed
            "R  conf/train/stage1.yaml\0conf/old.yaml\0"
            # a filename containing the literal " -> " must survive intact
            "M  data/raw/weird -> name.txt\0"
        )

    orig = T._git
    T._git = fake_git
    try:
        paths = T._git_dirty_paths()
        dirty = T._git_dirty()
    finally:
        T._git = orig

    assert paths == [
        "conf/train/stage1.yaml",
        "data/processed/splits/split_assignments.parquet",
        "data/raw/weird -> name.txt",
        "src/tbox_finder/train/train_stage1.py",
    ]
    assert "conf/old.yaml" not in paths, "the rename ORIGIN is not a path on disk"
    assert dirty is True
    assert "-z" in seen["args"] and "--untracked-files=no" in seen["args"]

    # One snapshot, two derivations: they must never disagree.
    T._git = lambda *a: ""
    try:
        assert T._git_dirty() is False
        assert T._git_dirty_paths() == []
    finally:
        T._git = orig

    T._git = lambda *a: None
    try:
        assert T._git_dirty() is None
        assert T._git_dirty_paths() is None
    finally:
        T._git = orig


def test_build_report_reads_the_working_tree_only_once() -> None:
    """git_dirty and git_dirty_paths must come from ONE snapshot, not two reads.

    Two `git status` calls can straddle a change to the tree, producing a boolean and a
    path list that never co-existed — and `_provenance_complete` would then classify
    evidence from two different worlds.

    This drives the REAL `build_report`. An earlier cut of this test called
    `_git_status_snapshot()` directly and asserted one call, which is a tautology: it
    proved that calling a function once calls it once, and would have stayed green with
    `build_report` reading the tree twice — the exact defect it is named for.
    """
    calls: list[tuple[str, ...]] = []
    orig = T._git

    def counting_git(*args: str):
        calls.append(args)
        if args and args[0] == "status":
            return " M data/processed/splits/split_assignments.parquet\0"
        if args and args[0] == "rev-parse":
            return "a" * 40
        return ""

    T._git = counting_git
    try:
        report = T.build_report(
            cfg=T.Stage1TrainConfig(),
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
    finally:
        T._git = orig

    status_calls = [c for c in calls if c and c[0] == "status"]
    assert len(status_calls) == 1, f"build_report read the working tree {len(status_calls)}x"

    prov = report["provenance"]
    # Both fields derive from that single read, and they agree.
    assert prov["git_dirty"] is True
    assert prov["git_dirty_paths"] == ["data/processed/splits/split_assignments.parquet"]
    # Data-only dirt ⇒ the SHA still describes the code, so the clause certifies.
    assert report["gate"]["provenance_complete"] is True


def test_cublas_workspace_config_pins_torch_s_own_literals() -> None:
    """Only the values torch itself accepts for deterministic cuBLAS (§8.3; ADR-0002 A6).

    Pinned to the EXTERNAL published literals read from the installed torch 2.7.1
    `use_deterministic_algorithms` docstring — not to our own default, which would be a
    tautology. `:4096:2` is torch's *cuDNN-RNN* value and is NOT accepted here.
    """
    from tbox_finder.train import repro

    assert frozenset({":4096:8", ":16:8"}) == repro.CUBLAS_DETERMINISTIC_CONFIGS
    assert repro.DEFAULT_CUBLAS_WORKSPACE_CONFIG in repro.CUBLAS_DETERMINISTIC_CONFIGS
    assert (
        ":0:0" not in repro.CUBLAS_DETERMINISTIC_CONFIGS
    ), "workspace-disabling is not deterministic"
    assert ":4096:2" not in repro.CUBLAS_DETERMINISTIC_CONFIGS, "that is the cuDNN-RNN value"


def test_non_finite_loss_fails_the_clause() -> None:
    """A NaN loss must not certify a run — the P2-02 focal-γ NaN lesson."""
    report = _valid_report()
    report["steps"]["losses"] = [1.6, float("nan")]
    assert T.derive_clauses(report)["loss_finite"] is False
    report["steps"]["losses"] = []
    assert T.derive_clauses(report)["train_step_ran"] is False


def test_class_counts_clause_rejects_bools_and_wrong_arity() -> None:
    report = _valid_report()
    report["class_counts"] = [True] * 8  # isinstance(True, int) is True
    assert T.derive_clauses(report)["class_counts_from_stream"] is False
    report["class_counts"] = [1, 2, 3]
    assert T.derive_clauses(report)["class_counts_from_stream"] is False
    report["class_counts"] = [0] * 8
    assert T.derive_clauses(report)["class_counts_from_stream"] is False


def test_validator_enforces_the_honesty_invariants() -> None:
    """A composition smoke may never claim to be science or to have graded GATE-4 (§10.3)."""
    for key in ("is_science", "gate4_graded"):
        report = _sealed(_valid_report())
        report[key] = True
        assert any(key in p for p in T.validate_report(report))


def test_a_valid_report_can_still_record_a_failed_run() -> None:
    """`validate_report` checks consistency, NOT success — so it cannot be the exit gate.

    This is why `train_stage1` raises separately on `overall_pass is False`: a run that
    trained zero steps (e.g. batch_size > the per-rank draw count) produces a perfectly
    *valid* report saying it failed, and would otherwise exit 0 — leaving the sbatch looking
    successful and §9.3's artifact-based verification passing it.
    """
    report = _valid_report()
    report["steps"] = {"n_steps": 0, "losses": [], "world_size": 1}
    report = _sealed(report)
    assert T.validate_report(report) == [], "a report recording a failure is still VALID"
    assert report["gate"]["overall_pass"] is False, "...but the gate must not pass"
    assert report["gate"]["train_step_ran"] is False


def test_validator_is_total_and_never_raises() -> None:
    """A malformed report is reported, not crashed on (the P1-16 `list(86)` lesson)."""
    for junk in (None, 42, "a string", [], {}, {"step": "P2-04"}):
        problems = T.validate_report(junk)  # type: ignore[arg-type]
        assert isinstance(problems, list) and problems


# ── the conf/ wiring (dependency-free scalar read — bare CI has no yaml) ────────────────
def _lines(path: Path) -> list[str]:
    """Stripped lines, inline comments removed. Full-line comments are kept intact, because
    Hydra's `# @package` directive IS a comment and must still be matchable."""
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("#"):
            line = line.split("#", 1)[0].strip()
        out.append(line)
    return out


def test_train_config_is_a_hydra_primary_that_reaches_its_groups() -> None:
    """A conf/<group>/ primary needs BOTH `@package _global_` and leading-slash defaults.

    Without them Hydra silently composes nothing and the run trains against defaults
    instead of the real stream — it succeeds, on the wrong data.
    """
    assert _TRAIN_CONF.is_file(), f"missing {_TRAIN_CONF}"
    lines = _lines(_TRAIN_CONF)
    assert lines[0] == "# @package _global_", "primary must carry the package directive first"
    assert "defaults:" in lines
    for group in (
        "- /data: stage1",
        "- /model: caduceus_stage1",
        "- /tracking: wandb",
        "- /optim: stage1",
    ):
        assert group in lines, f"missing leading-slash default: {group}"


def test_optim_stage1_is_a_group_option_not_a_primary() -> None:
    assert _OPTIM_CONF.is_file(), f"missing {_OPTIM_CONF}"
    for line in _lines(_OPTIM_CONF):
        assert not line.startswith(
            "# @package"
        ), f"group option carries a package directive: {line}"
        assert not line.startswith("defaults:"), "group option carries a defaults list"


def test_gradient_checkpointing_defaults_on_per_prd_10_3() -> None:
    """PRD §10.3 pins it for the Stage-1 full fine-tune (user sign-off 2026-07-16)."""
    assert "gradient_checkpointing: true" in _lines(_TRAIN_CONF)


def test_wandb_sync_rule_is_not_duplicated() -> None:
    """imp.md names `train.smk::wandb_sync`, but setup.smk::wandb_sync already exists (P0).

    Two rules of one name break the Snakefile's `rules/*.smk` auto-include, and the Snakefile
    states GPU stages are intentionally not in the DAG (PRD §16) — so there is no train.smk.

    Counts *declarations*, not files: a single .smk declaring the rule twice would still
    yield one filename and pass, while breaking the include just the same.
    """
    rules = _REPO / "workflow" / "rules"
    declaring = [
        p.name
        for p in sorted(rules.glob("*.smk"))
        for line in p.read_text().splitlines()
        if line.strip() == "rule wandb_sync:"
    ]
    assert declaring == ["setup.smk"], f"wandb_sync must be declared exactly once; got {declaring}"


def test_the_backbone_pin_has_exactly_one_home() -> None:
    """imp.md names conf/model/caduceus_ps.yaml; caduceus_stage1.yaml already owns the pin.

    A second file carrying repo_id/revision is a drift surface for a load-bearing pin.
    """
    models = _REPO / "conf" / "model"
    carrying = sorted(
        p.name for p in models.glob("*.yaml") if "caduceus-ps_seqlen-131k" in p.read_text()
    )
    assert carrying == ["caduceus_stage1.yaml"], f"backbone pin must have one home; got {carrying}"


# ── the committed report, if a real run produced one ────────────────────────────────────
def test_committed_report_was_produced_from_a_clean_tree() -> None:
    """A committed report whose SHA does not describe its own code is false provenance (§11).

    `git_dirty=true` would mean the recorded commit does not contain the code that produced
    the run — so the SHA names the wrong tree. The committed artifact must come from the
    authoring commit, not from a work-in-progress checkout.
    """
    if not _REPORT.is_file():
        _fail_or_skip("TBOX_REQUIRE_TRAIN_SMOKE", f"no committed report at {_REPORT}")
    prov = json.loads(_REPORT.read_text())["provenance"]
    assert prov["git_dirty"] is False, (
        "the committed smoke report was generated from a dirty tree — its git_sha names a "
        "commit that does not contain the code that produced it; re-run after committing"
    )
    assert isinstance(prov["git_sha"], str) and re.fullmatch(
        r"[0-9a-f]{40}", prov["git_sha"]
    ), f"git_sha must be 40 lowercase hex chars, got {prov['git_sha']!r}"


_PRODUCTION_REPORT = _REPO / "reports" / "p2" / "train_stage1_production.json"


def test_committed_production_report_records_its_audited_dirty_tree() -> None:
    """P2-09 job 671 — the ONE committed report whose tree was legitimately dirty.

    Unlike the smoke report above, a cluster run cannot come from a clean tree: the cluster
    has no git-lfs, so every run re-stages `split_assignments.parquet` over its committed
    132-byte pointer and `git status` is dirty by construction. The P2-06 precedent
    (`test_sweep_selection.py`, 36 points) handled this by auditing at retrieval and locking
    the flag with a test; this is that lock for the production checkpoint.

    AUDITED at retrieval, 2026-07-20, on the cluster checkout at 0f8d527:
    `git status --porcelain` showed exactly ` M data/processed/splits/split_assignments.parquet`
    plus untracked run outputs, and `git diff --stat` over tracked non-`data` paths was
    EMPTY — zero tracked code differed, so `git_sha` does describe what ran.

    The report PREDATES the `git_dirty_paths` field added in the same step, so the
    strengthened `_provenance_complete` cannot re-confirm that audit from the recorded
    fields — it re-derives False against a stored True. That divergence is deliberate and
    is pinned here rather than left to be rediscovered: the checkpoint was NOT re-run,
    because ADR-0002 A7 gives metric-level (not bitwise) determinism, so regenerating the
    report purely to satisfy a gate would ship different weights for no scientific gain
    (user decision, 2026-07-20). Every run from P2-10 on records the paths and is verified
    automatically; when this file is regenerated, this test must be updated consciously.
    """
    if not _PRODUCTION_REPORT.is_file():
        _fail_or_skip("TBOX_REQUIRE_TRAIN_SMOKE", f"no committed report at {_PRODUCTION_REPORT}")
    report = json.loads(_PRODUCTION_REPORT.read_text())
    prov = report["provenance"]
    assert isinstance(prov["git_sha"], str) and re.fullmatch(
        r"[0-9a-f]{40}", prov["git_sha"]
    ), f"git_sha must be 40 lowercase hex chars, got {prov['git_sha']!r}"
    assert prov["git_dirty"] is True, (
        "the production report's git_dirty changed — re-audit the working tree at "
        "retrieval and update this test's recorded audit"
    )
    assert "git_dirty_paths" not in prov, (
        "this report now carries git_dirty_paths — it was regenerated by post-P2-09 code, "
        "so the manual-audit lock above is obsolete: assert the paths are data-only instead"
    )
    # The stored clause reflects the pre-fix contract; the re-derivation reflects the new
    # one. Pinning BOTH is what makes the divergence impossible to drift through silently.
    assert report["gate"]["provenance_complete"] is True
    assert T.derive_clauses(report)["provenance_complete"] is False


def test_committed_report_validates_and_its_clauses_re_derive() -> None:
    if not _REPORT.is_file():
        _fail_or_skip("TBOX_REQUIRE_TRAIN_SMOKE", f"no committed report at {_REPORT}")
    report = json.loads(_REPORT.read_text())
    assert T.validate_report(report) == []
    assert report["is_science"] is False
    assert report["gate"]["overall_pass"] is True
    derived = T.derive_clauses(report)
    for name, value in derived.items():
        if name in report["gate"]:
            assert report["gate"][name] is value, f"{name} was asserted, not derived"
            continue
        # A clause added after this artifact was written. Regenerating the report to carry
        # it would forge a measurement the run never made (CLAUDE.md §10.3), so
        # `validate_report` excuses it — but ONLY for an older schema and ONLY when it
        # re-derives TRUE. Both halves are asserted here, so the excuse cannot silently
        # widen into "a missing clause is fine".
        introduced = T.CLAUSE_SCHEMA_VERSION.get(name)
        assert introduced is not None, f"{name} is missing and was not introduced later"
        assert int(report["schema_version"]) < int(introduced), name
        assert value is True, f"{name} is missing AND false — the excuse must not hide it"


def test_committed_report_scopes_its_class_counts_honestly() -> None:
    """§10.2: a slice must be labelled a slice.

    P2-03 shipped 56-row sample properties worded as corpus facts and no test read the field
    that said so. `full_stream` must therefore agree with the arithmetic, not be asserted:
    a smoke over 64 of 8,303 records may not present itself as the fold.
    """
    if not _REPORT.is_file():
        _fail_or_skip("TBOX_REQUIRE_TRAIN_SMOKE", f"no committed report at {_REPORT}")
    report = json.loads(_REPORT.read_text())
    scope = report["class_counts_scope"]
    assert scope["full_stream"] is (scope["n_records"] == scope["n_training_fold_records"])
    assert scope["occurrence"] == 0, "the scan must be pinned at a single occurrence"
    assert scope["counted_at_epoch"] == 0, "counts depend on the epoch — record which"
    assert sum(report["class_counts"]) > 0


def test_committed_report_records_the_hand_wired_checkpointing() -> None:
    """The PRD §10.3 deviation-that-wasn't: the evidence for hand-wiring must not rot."""
    if not _REPORT.is_file():
        _fail_or_skip("TBOX_REQUIRE_TRAIN_SMOKE", f"no committed report at {_REPORT}")
    ckpt = json.loads(_REPORT.read_text())["gradient_checkpointing"]
    assert ckpt["hf_flag_supported"] is False, (
        "the pinned Caduceus revision does not support HF gradient checkpointing — if this "
        "ever becomes True, the hand-wiring should be revisited"
    )
    assert ckpt["requested"] is True and ckpt["n_blocks_wrapped"] == ckpt["n_blocks"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Hydra composition (TBOX_REQUIRE_TRAIN_HYDRA; CI installs no hydra)
# ═══════════════════════════════════════════════════════════════════════════
def _require_hydra():
    try:
        from hydra import compose, initialize_config_dir  # noqa: F401
    except ImportError as exc:
        _fail_or_skip("TBOX_REQUIRE_TRAIN_HYDRA", f"hydra not importable: {exc}")
    from hydra import compose, initialize_config_dir

    return compose, initialize_config_dir


def test_hydra_config_really_composes_and_reaches_every_group() -> None:
    compose, initialize_config_dir = _require_hydra()
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(_REPO / "conf")):
        cfg = OmegaConf.to_container(compose(config_name="train/stage1"), resolve=True)
    # @package _global_ lifted the primary's own keys to the root...
    assert cfg["seed"] == 42 and cfg["gradient_checkpointing"] is True
    # ...and the leading-slash defaults actually pulled the sibling groups in.
    assert cfg["data"]["window_nt"] == 1024 and cfg["data"]["stride_nt"] == 512
    assert cfg["model"]["revision"] == "d89eeb853136ea64da7feb3d0c8e909771b17ae6"
    assert cfg["tracking"]["mode"] == "offline"
    assert cfg["optim"]["lr"] == 1.0e-4


def test_cfg_from_mapping_flattens_the_groups_into_the_dataclass() -> None:
    compose, initialize_config_dir = _require_hydra()
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(_REPO / "conf")):
        raw = OmegaConf.to_container(compose(config_name="train/stage1"), resolve=True)
    cfg = T._cfg_from_mapping(raw)
    assert isinstance(cfg, T.Stage1TrainConfig)
    assert cfg.lr == 1.0e-4, "optim.lr must reach the dataclass"
    assert cfg.wandb_mode == "offline", "tracking.mode must reach the dataclass"
    assert cfg.rc_combine == "concat", "model.rc_combine.mode must reach the dataclass"
    assert cfg.seed == 42 and cfg.gradient_checkpointing is True


def test_cli_override_reaches_the_dataclass() -> None:
    """PRD §11 sweeps via `--multirun`; an override that silently didn't apply would make
    every sweep point train the same config while reporting different ones."""
    compose, initialize_config_dir = _require_hydra()
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(_REPO / "conf")):
        raw = OmegaConf.to_container(
            compose(config_name="train/stage1", overrides=["optim.lr=3e-4", "gamma=0.5"]),
            resolve=True,
        )
    cfg = T._cfg_from_mapping(raw)
    assert cfg.lr == 3e-4 and cfg.gamma == 0.5


def test_the_data_group_reaches_the_datamodule_not_just_the_config() -> None:
    """PRD §11 sweeps window/stride (P2-06) — the values must reach Stage1DataConfig.

    Asserting `cfg["data"]["window_nt"] == 1024` on the *composed dict* proves only that
    Hydra read the file; it says nothing about whether the datamodule ever sees it. An
    earlier draft built `Stage1DataConfig(seed=...)` alone, so the data group's values were
    silently dropped and the defaults merely happened to agree — a window/stride sweep would
    have run every point on the same 1024/512 stream while reporting the swept value. So the
    override is followed all the way to the dataclass the dataset is constructed from.
    """
    compose, initialize_config_dir = _require_hydra()
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(_REPO / "conf")):
        raw = OmegaConf.to_container(
            compose(
                config_name="train/stage1",
                overrides=["data.window_nt=512", "data.stride_nt=256", "data.klass_alpha=0.5"],
            ),
            resolve=True,
        )
    cfg = T._cfg_from_mapping(raw)
    assert (cfg.window_nt, cfg.stride_nt, cfg.klass_alpha) == (512, 256, 0.5)


def test_data_group_defaults_mirror_the_authoritative_module() -> None:
    """The dataclass defaults must not drift from window_dataset.py, which is authoritative."""
    from tbox_finder.data import window_dataset as wd

    cfg = T.Stage1TrainConfig()
    assert (cfg.window_nt, cfg.stride_nt) == (wd.WINDOW_NT, wd.STRIDE_NT)
    assert (cfg.phylum_alpha, cfg.klass_alpha, cfg.aa_alpha) == (
        wd.DEFAULT_PHYLUM_ALPHA,
        wd.DEFAULT_KLASS_ALPHA,
        wd.DEFAULT_AA_ALPHA,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 — torch (TBOX_REQUIRE_TRAIN_TORCH; CI installs no torch — its OWN var)
# ═══════════════════════════════════════════════════════════════════════════
def _require_torch():
    try:
        import torch
    except ImportError as exc:
        _fail_or_skip("TBOX_REQUIRE_TRAIN_TORCH", f"torch not importable: {exc}")
    import torch

    return torch


def test_gradient_checkpointing_wraps_every_block_and_is_idempotent() -> None:
    """The wrap must be measured, not asserted: count the marked blocks off the tree."""
    torch = _require_torch()

    class _Block(torch.nn.Module):
        def forward(self, x, residual=None, inference_params=None):  # noqa: D102
            return x, residual

    class _Fake(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.layers = torch.nn.ModuleList([_Block() for _ in range(16)])

    model = _Fake()
    assert not T.is_gradient_checkpointing_enabled(model)
    assert T.enable_gradient_checkpointing(model) == 16
    assert T.is_gradient_checkpointing_enabled(model)
    # Idempotent: a second call must wrap nothing (double-wrapping recomputes twice).
    assert T.enable_gradient_checkpointing(model) == 0
    assert T.is_gradient_checkpointing_enabled(model)


def test_set_determinism_replaces_a_nondeterministic_cublas_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launcher's bad CUBLAS_WORKSPACE_CONFIG must be REPLACED, not preserved.

    The r2 fix (setdefault → validate) had no test at all, so reverting it was green — and
    the failure it prevents is silent: `:0:0` disables the cuBLAS workspace, `warn_only=True`
    downgrades torch's complaint to a warning, and the run carries on quietly non-reproducible.
    Runs on CPU; the env manipulation is the whole point.
    """
    _require_torch()
    from tbox_finder.train.repro import set_determinism

    # A value torch rejects must be overwritten...
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    set_determinism(42)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # ...as must a missing one...
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_determinism(42)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # ...but a legitimate deterministic choice by the launcher is HONOURED, not clobbered.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    set_determinism(42)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_set_determinism_actually_seeds_the_rngs() -> None:
    """`def set_determinism(seed): pass` must not be green — it seeds, or it is a lie."""
    torch = _require_torch()
    import random as _random

    set_det = __import__("tbox_finder.train.repro", fromlist=["set_determinism"]).set_determinism
    set_det(1234)
    a_torch, a_py = torch.randn(4), _random.random()
    set_det(1234)
    b_torch, b_py = torch.randn(4), _random.random()
    torch.testing.assert_close(a_torch, b_torch, rtol=0, atol=0)
    assert a_py == b_py
    # A *different* seed must give a different stream, or "seeding" is a no-op that
    # reproduces because nothing is random.
    set_det(4321)
    assert not torch.equal(a_torch, torch.randn(4))


def test_checkpointed_block_is_numerically_transparent_and_recomputes() -> None:
    """Checkpointing must change the memory schedule, not the maths."""
    torch = _require_torch()

    calls = {"n": 0}

    class _Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x, residual=None, inference_params=None):  # noqa: D102
            calls["n"] += 1
            return self.lin(x).tanh(), residual

    class _Fake(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.layers = torch.nn.ModuleList([_Block()])

    torch.manual_seed(0)
    model = _Fake()
    x = torch.randn(2, 4, requires_grad=True)

    plain, _ = model.backbone.layers[0](x, None)
    plain.sum().backward()
    plain_grad = x.grad.clone()
    n_plain = calls["n"]

    x.grad = None
    calls["n"] = 0
    T.enable_gradient_checkpointing(model)
    ckpt_out, _ = model.backbone.layers[0](x, None)

    # Sample the counter BETWEEN forward and backward — this is the assertion that has
    # content. Checking only the total after backward (`calls == 2`) cannot tell real
    # recomputation from a wrap that simply calls the block twice eagerly and returns the
    # plain result: that stub has identical output, identical grads, ZERO memory saving and
    # 2x compute — the exact inverse of checkpointing — and it passed the earlier form.
    assert calls["n"] == 1, "the forward must run exactly once before backward"
    ckpt_out.sum().backward()
    assert calls["n"] == 2, "backward must RECOMPUTE the forward exactly once (that is the wrap)"

    torch.testing.assert_close(ckpt_out, plain, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, plain_grad, rtol=0, atol=1e-7)
    assert n_plain == 1, "the unwrapped baseline runs the forward once and never recomputes"


@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_TRAIN_CUDA") != "1",
    reason="loads the pinned Caduceus-PS backbone + the DVC corpus; local/cluster only",
)
def test_one_train_step_composes_end_to_end() -> None:
    """imp.md's gate: one train step composes on the real stream, and writes a valid report.

    Fails closed on its own var: `_require_torch` would otherwise *skip* on a missing torch
    via TBOX_REQUIRE_TRAIN_TORCH, so someone arming the CUDA gate alone would get a green
    skip — the tier reporting success by not running.

    PYTHONHASHSEED is **verified, not monkeypatched**. An earlier draft set it here so the
    guard would pass, which asserted a lie: the env var would read "0" while the interpreter's
    actual hash seed stayed whatever it was launched with, since CPython fixes it at startup.
    That is the same in-process no-op the guard exists to catch — reproduced inside the test
    for it. So this tier must be *launched* correctly: `PYTHONHASHSEED=0 pytest ...`.
    """
    try:
        import torch
    except ImportError as exc:
        pytest.fail(f"TBOX_REQUIRE_TRAIN_CUDA=1 but torch is not importable: {exc}")
    if not torch.cuda.is_available():
        pytest.fail("TBOX_REQUIRE_TRAIN_CUDA=1 but no CUDA device (Caduceus has no CPU path)")
    T.check_pythonhashseed(0)
    cfg = T.Stage1TrainConfig(
        epochs=1,
        batch_size=2,
        max_records=8,
        steps_per_epoch=1,
        wandb_mode="disabled",
        save_checkpoint=False,
        report_path=str(_REPO / "reports" / "p2" / "_smoke_test.json"),
    )
    report = T.train_stage1(cfg, log=lambda *_: None)
    assert T.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True
    assert report["gradient_checkpointing"]["n_blocks_wrapped"] == 16
    assert report["steps"]["n_steps"] == 1

    # The P2-05 instrumentation must actually be EMITTED by the real loop. Nothing else
    # checks this: `validate_report` skips the timing keys when absent (by design — P2-04's
    # committed artifact predates them), no test called `build_report` with a non-None
    # timing kwarg, and this test previously asserted nothing about it. So deleting
    # `step_seconds=step_seconds` from the `build_report(...)` call left every test green
    # while the entire sizing pipeline lost its input — a guard that could not fire, which
    # is the exact defect class P2-04 shipped and this step inherited.
    steps = report["steps"]
    for key in ("step_seconds", "batch_wait_seconds"):
        assert key in steps, f"the instrumented loop must emit steps.{key}"
        assert len(steps[key]) == steps["n_steps"], f"{key}: one entry per step"
        assert all(isinstance(x, float) and math.isfinite(x) and x >= 0.0 for x in steps[key])
    # A real CUDA step is not instantaneous; a zero here would mean the clock is measuring
    # kernel LAUNCH rather than execution (i.e. the synchronize was dropped).
    assert steps["step_seconds"][0] > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# P2-06a — the validation ladder's ALIGNMENT, proven with an oracle
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_TRAIN_CUDA") != "1",
    reason="reads the DVC corpus; local/cluster only",
)
def test_the_val_eval_scores_a_perfect_oracle_at_f1_one() -> None:
    """THE test the P2-06a eval rests on: prove the label alignment, independent of the model.

    The measured smoke reports ``min_f1 = 0.0`` — correctly, because a 4-step model with a
    randomly-initialised head has learned nothing. But **a broken eval reports 0.0 too**: an
    off-by-one between the reconciled prediction and ``context_labels``, a wrong tiling
    origin, or a transposed window would all collapse the score, and an untrained model
    gives us no way to tell those apart. That is the P2-03 lesson exactly — an operator
    misaligning every left-overhanging window passed all 48 of its tests, because nothing
    numeric ever read the slice.

    So: replace the model with an **oracle** that emits one-hot logits of the TRUE labels
    for each window's own span. If every index in the chain agrees, the reconciled argmax
    must reproduce ``context_labels`` EXACTLY and every per-element F1 must be 1.0. A
    one-nucleotide shift anywhere destroys it — which the sibling test below proves.
    """
    torch = _require_torch()
    import numpy as np

    from tbox_finder.data.window_dataset import (
        context_labels,
        load_selection_val_records,
        tile_windows,
    )

    records, _ = load_selection_val_records()
    rec = records[0]
    truth = context_labels(rec)
    starts = tile_windows(len(rec.context_seq), window=1024, stride=512)

    class _Oracle:
        """Emits one-hot logits of the true labels for each window, in tiling order."""

        training = False

        def __init__(self, shift: int = 0) -> None:
            self.i = 0
            self.shift = shift

        def eval(self):
            return self

        def train(self):
            return self

        def __call__(self, *, input_ids):
            n = int(input_ids.shape[0])
            out = torch.full((n, 1024, T.NUM_CLASSES), -10.0)
            for k in range(n):
                s = starts[self.i + k] + self.shift
                span = np.asarray(
                    [truth[p] if 0 <= p < len(truth) else 0 for p in range(s, s + 1024)]
                )
                out[k, np.arange(1024), span] = 10.0
            self.i += n
            return out

    cfg = T.Stage1TrainConfig(eval_val=True, eval_max_records=1, eval_n_boot=25)
    metrics_block, scope = T.evaluate_selection_val(_Oracle(), torch.device("cpu"), cfg=cfg)

    assert scope["n_records_scored"] == 1
    assert metrics_block["n_positions"] == len(truth)
    for elem, f1 in metrics_block["gate4_core_min_f1"]["per_element_f1"].items():
        assert f1 == pytest.approx(1.0), f"{elem} scored {f1} on a PERFECT oracle — misaligned"
    assert metrics_block["gate4_core_min_f1"]["min_f1"] == pytest.approx(1.0)
    assert metrics_block["gate4_core_min_f1"]["passes"] is True
    assert metrics_block["micro_f1"] == pytest.approx(1.0)


@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_TRAIN_CUDA") != "1",
    reason="reads the DVC corpus; local/cluster only",
)
def test_a_one_nucleotide_shift_destroys_the_oracle_score() -> None:
    """Sabotage: prove the oracle test above BITES rather than passing by construction.

    Without this, ``f1 == 1.0`` might hold for a reason unrelated to alignment. Shifting the
    oracle's span by a single nucleotide must break it — if it does not, the test above is
    measuring nothing.
    """
    torch = _require_torch()
    import numpy as np

    from tbox_finder.data.window_dataset import (
        context_labels,
        load_selection_val_records,
        tile_windows,
    )

    records, _ = load_selection_val_records()
    rec = records[0]
    truth = context_labels(rec)
    starts = tile_windows(len(rec.context_seq), window=1024, stride=512)

    class _ShiftedOracle:
        training = False

        def __init__(self) -> None:
            self.i = 0

        def eval(self):
            return self

        def train(self):
            return self

        def __call__(self, *, input_ids):
            n = int(input_ids.shape[0])
            out = torch.full((n, 1024, T.NUM_CLASSES), -10.0)
            for k in range(n):
                s = starts[self.i + k] + 1  # <-- the sabotage: one nucleotide off
                span = np.asarray(
                    [truth[p] if 0 <= p < len(truth) else 0 for p in range(s, s + 1024)]
                )
                out[k, np.arange(1024), span] = 10.0
            self.i += n
            return out

    cfg = T.Stage1TrainConfig(eval_val=True, eval_max_records=1, eval_n_boot=25)
    metrics_block, _ = T.evaluate_selection_val(_ShiftedOracle(), torch.device("cpu"), cfg=cfg)
    assert metrics_block["gate4_core_min_f1"]["min_f1"] < 1.0, (
        "a 1-nt shift still scored a perfect 1.0 — the alignment test cannot bite, so it "
        "proves nothing about the eval"
    )
