"""P3-06 sizing smoke — the harness that exists because a smoke measured the wrong thing.

The gate under test here grades the **measurement**, not the answer: "batch 8 does not fit"
is a successful sizing run. What must never pass is a measurement that quietly describes some
*other* computation — which is exactly what P1-16's `loss_is_placeholder: true` backbone smoke
did for P3-06, at a cost of two failed production submits.

Tiers, each with its own env var (folding them was the P1-16 landmine):
* **pure** — the report clauses, the validator, the row pickers, the growth ratio. Bare CI.
* **torch** (``TBOX_REQUIRE_STAGE2_TORCH``) — a real measurement through a tiny backbone.
* **data** (``TBOX_REQUIRE_STAGE2_DATA``) — the real corpus's length distribution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tbox_finder.models import rna_backbone_registry as BR
from tbox_finder.stage2 import losses as L
from tbox_finder.stage2 import sizing as S
from tbox_finder.stage2 import train as T

_REPO = Path(__file__).resolve().parents[2]
_SBATCH = _REPO / "slurm" / "p3" / "stage2_sizing_smoke.sbatch"
_DATASET = _REPO / T.DEFAULT_DATASET


def _fail_or_skip(var: str, reason: str) -> None:
    if os.environ.get(var) == "1":
        pytest.fail(f"{var}=1 but the tier is unrunnable: {reason}")
    pytest.skip(reason)


def _require_stack():
    if os.environ.get("CUDA_HOME") is None:
        _fail_or_skip("TBOX_REQUIRE_STAGE2_TORCH", "CUDA_HOME unset — multimolecule won't import")
    try:
        import multimolecule  # noqa: F401
        import peft  # noqa: F401
        import torch
    except Exception as exc:  # pragma: no cover - env-dependent
        _fail_or_skip("TBOX_REQUIRE_STAGE2_TORCH", f"pinned ml-rna stack unavailable ({exc})")
    return torch


def _row(n_tokens: int, row_id: str) -> dict[str, Any]:
    nt = n_tokens - 2
    return {
        "row_id": row_id,
        "source": "corpus",
        "pool": "corpus",
        "fold_basis": "corpus_record",
        "fold_random": "train",
        "nested_train": True,
        "nested_role": "train",
        "calib": False,
        "is_tbox": True,
        "n_tokens": n_tokens,
        "seq_length": nt,
        "rna_sequence": "GCAU" * (nt // 4) + "GCAU"[: nt % 4],
        "label_string": "." * nt,
        "regulatory_mode": "Terminator",
        "specifier_codon": "AUG",
        "cognate_aa": "Met",
        "trna_family": "Met (CAU)",
        "pairing_dotbracket": None,
    }


def _measurements(**over: Any) -> list[dict[str, Any]]:
    base = [
        {
            "batch_size": b,
            # A fitting point must have measured the batch it claims — otherwise "batch 8
            # fits" can mean "3 rows fit", the request standing in for the measurement.
            "measured_batch_size": b,
            "regime": r,
            "oom": False,
            "n_steps": 6,
            "peak_vram_gib": 3.0,
            "peak_vram_gib_per_step": [3.0, 3.0, 3.0],
            "step_ms": [100.0],
            "padded_tokens": 552,
            # The flag each point actually ran under. `checkpointing_skip_is_earned`
            # cross-checks it against the port's recorded usability, so a sweep cannot
            # measure one configuration while the report claims another.
            "gradient_checkpointing": True,
        }
        for b in (4, 2)
        for r in ("worst_case", "typical")
    ]
    for m in base:
        m.update(over)
    return base


def _report(**over: Any) -> dict[str, Any]:
    payload = {
        "measurements": _measurements(),
        "population": {"n_rows": 200, "n_tokens_max": 552},
        "device": {"is_cuda": True, "name": "NVIDIA RTX A4000", "total_memory_gib": 15.6},
        "checkpointing": {
            "on_peak_gib": 3.0,
            "off_peak_gib": 9.0,
            "saving_ratio": 3.0,
            # The production port CAN run checkpointing (it is merely a no-op there), so the
            # fixture's sweep points carry the flag ON and the comparison is measurable.
            "usable_on_this_backbone": True,
        },
        "config": {"batch_sweep": [4, 2], "steps": 6},
        "backbone": {
            **BR.backbone_summary(BR.resolve_backbone(BR.PRODUCTION_BACKBONE)),
            "requested_key": BR.PRODUCTION_BACKBONE,
        },
    }
    payload.update(over)
    return S.build_report(**payload)


# --------------------------------------------------------------------------------------
# TIER 1 — the gate grades the MEASUREMENT
# --------------------------------------------------------------------------------------
def test_a_sound_measurement_passes() -> None:
    report = _report()
    assert S.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True


def test_an_all_oom_sweep_is_still_a_SUCCESSFUL_measurement() -> None:
    """Learning that nothing fits is the run doing its job, not the run failing.

    A gate that demanded a fitting batch size would pressure the harness toward a convenient
    answer — which is the failure mode this whole step exists to correct.
    """
    report = _report(measurements=_measurements(oom=True, peak_vram_gib=None, n_steps=0))
    assert report["gate"]["overall_pass"] is True, report["gate"]["failed"]
    assert report["largest_fitting_batch_worst_case"] is None
    assert report["recommendation"] is None


@pytest.mark.parametrize(
    ("mutate", "clause"),
    [
        (lambda p: p["device"].__setitem__("is_cuda", False), "measured_on_cuda"),
        (lambda p: p["population"].__setitem__("n_rows", 0), "real_objective"),
        (lambda p: p.__setitem__("measurements", []), "swept"),
        (
            lambda p: p.__setitem__("measurements", _measurements(n_steps=1)),
            "enough_steps_for_optimizer_state",
        ),
        (
            lambda p: p.__setitem__(
                "measurements", [dict(m, regime="typical") for m in _measurements()]
            ),
            "worst_case_measured",
        ),
        (
            lambda p: p.__setitem__("checkpointing", {"on_peak_gib": None, "off_peak_gib": None}),
            "checkpointing_effect_measured",
        ),
    ],
)
def test_each_clause_bites_on_its_own_evidence(mutate: Any, clause: str) -> None:
    payload = {
        "measurements": _measurements(),
        "population": {"n_rows": 200},
        "device": {"is_cuda": True, "name": "A4000"},
        "checkpointing": {
            "on_peak_gib": 3.0,
            "off_peak_gib": 9.0,
            "usable_on_this_backbone": True,
        },
        "config": {},
        "backbone": {
            **BR.backbone_summary(BR.resolve_backbone(BR.PRODUCTION_BACKBONE)),
            "requested_key": BR.PRODUCTION_BACKBONE,
        },
    }
    mutate(payload)
    assert S.derive_clauses(S.build_report(**payload))[clause] is False


def test_a_placeholder_loss_measurement_could_not_pass_this_gate() -> None:
    """The specific regression: P1-16's shape must be inexpressible here.

    `real_objective` reads the three flags whose values P1-16 recorded as
    synthetic/placeholder. A report claiming those would fail, which is the guard that stops
    this harness from ever becoming the thing it replaced.
    """
    report = _report()
    for key in ("data_is_synthetic", "loss_is_placeholder"):
        broken = dict(report)
        broken[key] = True
        assert S.derive_clauses(broken)["real_objective"] is False
    broken = dict(report)
    broken["step_function"] = "some.other.reimplementation"
    assert S.derive_clauses(broken)["real_objective"] is False


def test_the_validator_catches_a_clause_fabricated_true() -> None:
    report = _report()
    report["device"]["is_cuda"] = False
    report["gate"]["clauses"]["measured_on_cuda"] = True
    assert any("measured_on_cuda" in p for p in S.validate_report(report))


# --------------------------------------------------------------------------------------
# TIER 1 — the pickers and the growth ratio
# --------------------------------------------------------------------------------------
def test_the_worst_case_picker_takes_the_LONGEST_rows() -> None:
    """Sizing on the median is how job 1044 OOM'd; the collator pads to the batch maximum."""
    rows = [_row(n, f"r{n}") for n in (60, 200, 552, 400, 100)]
    chosen = S._worst_case_rows(rows, batch_size=2)
    assert [r["n_tokens"] for r in chosen] == [552, 400]
    typical = S._typical_rows(rows, batch_size=2)
    assert max(r["n_tokens"] for r in typical) < 552


def test_the_growth_ratio_detects_a_climbing_footprint() -> None:
    """Job 1044 left open whether 13.97 GiB was per-step or accumulating. This answers it."""
    assert S._growth_ratio([2.0, 2.0, 2.0]) == pytest.approx(1.0)
    assert S._growth_ratio([2.0, 3.0, 4.0]) == pytest.approx(2.0)
    assert S._growth_ratio([]) is None
    assert S._growth_ratio([2.0]) is None


# --------------------------------------------------------------------------------------
# TIER 1 — the shipped sbatch
# --------------------------------------------------------------------------------------
def test_the_sizing_sbatch_asks_for_one_gpu_and_fails_fast_on_a_bad_node() -> None:
    text = _SBATCH.read_text()
    directives = [ln for ln in text.splitlines() if ln.startswith("#SBATCH ")]
    assert "#SBATCH --gres=gpu:a4000:1" in text  # ONE gpu: this measures a step, not DDP
    assert not any("--nodelist" in ln for ln in directives)
    assert "conda activate tbox-ml-rna" in text
    assert text.index("STAGE2_NODE_UNHEALTHY") < text.index("hf cache warm")
    if any("--exclude" in ln for ln in directives):
        assert "REMOVE once node `two` has been rebooted" in text


def test_the_sizing_sbatch_measures_the_worst_case_batch_sizes_that_matter() -> None:
    text = _SBATCH.read_text()
    assert "--batch-sweep 8,4,2,1" in text  # 8 is what job 1044 tried and lost
    assert "--steps 6" in text  # AdamW state is allocated on the FIRST .step()


# --------------------------------------------------------------------------------------
# TIER 2 — a real measurement through a tiny backbone
# --------------------------------------------------------------------------------------
def _tiny_backbone():
    from multimolecule import RiNALMoConfig, RiNALMoModel

    cfg = RiNALMoConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128
    )
    cfg._attn_implementation = "sdpa"
    return RiNALMoModel(cfg, add_pooling_layer=False)


def test_measure_batch_runs_the_real_step_and_records_a_series() -> None:
    """It must call the trainer's own `forward_backward`, and record EVERY step's peak."""
    _require_stack()
    rows = [_row(64, f"r{i}") for i in range(4)]
    record = S.measure_batch(
        rows,
        batch_size=2,
        steps=4,
        gradient_checkpointing=False,
        loss_config=L.Stage2LossConfig(),
        base_model=_tiny_backbone(),
        device="cpu",
    )
    assert record["oom"] is False and record["error"] is None
    assert record["n_steps"] == 4
    assert len(record["step_ms"]) == 4
    assert record["padded_tokens"] == 64


def test_the_harness_calls_the_trainer_s_step_not_a_copy_of_it(monkeypatch) -> None:
    """If this ever stops being true the harness can drift from the trainer — the P1-16 defect.

    Asserted by interception rather than by reading the source, so a future refactor that
    inlines the step is caught rather than merely discouraged.
    """
    _require_stack()
    calls: list[int] = []
    real = T.forward_backward

    def counting(*args: Any, **kwargs: Any):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(T, "forward_backward", counting)
    S.measure_batch(
        [_row(64, f"r{i}") for i in range(2)],
        batch_size=2,
        steps=3,
        gradient_checkpointing=False,
        loss_config=L.Stage2LossConfig(),
        base_model=_tiny_backbone(),
        device="cpu",
    )
    assert len(calls) == 3, calls


# --------------------------------------------------------------------------------------
# TIER 3 — the real corpus
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_STAGE2_DATA") != "1",
    reason="reads the DVC-tracked Stage-2 dataset; local/cluster only",
)
def test_the_real_corpus_has_the_length_spread_that_makes_worst_case_matter() -> None:
    """If every row were the same length the worst-case/typical split would be theatre."""
    rows, census = S._select_rows(str(_DATASET), max_records=200)
    assert census["n_admitted"] > 0
    lengths = [int(r["n_tokens"]) for r in rows]
    assert max(lengths) > min(lengths), "no length spread — the two regimes would be identical"
    worst = S._worst_case_rows(rows, batch_size=8)
    typical = S._typical_rows(rows, batch_size=8)
    assert min(r["n_tokens"] for r in worst) >= max(r["n_tokens"] for r in typical)


def test_the_pinned_rinalmo_still_does_not_implement_gradient_checkpointing() -> None:
    """A drift guard on a no-op we now depend on knowing about.

    multimolecule 0.0.9's RiNALMo advertises ``supports_gradient_checkpointing = True`` and
    stores ``self.gradient_checkpointing``, but its encoder loop never calls
    ``_gradient_checkpointing_func`` — so enabling it sets an attribute nothing reads.
    Measured at job 1051: 3.1640 GiB on vs 3.1595 GiB off, a saving ratio of 0.9986 across
    36 modules carrying the flag. `conf/train/stage2.yaml` therefore ships it FALSE rather
    than claiming a setting with no effect.

    This asserts the *source*, so that if a future multimolecule implements the hook the test
    fails and the config is revisited — rather than the project quietly running without a
    memory optimisation it could have had. It is the counterpart to the lesson that a flag
    which no-ops looks exactly like one that works: here the no-op is known, and the guard is
    against it silently ceasing to be one.
    """
    _require_stack()
    import inspect

    from multimolecule.models.rinalmo import modeling_rinalmo

    source = inspect.getsource(modeling_rinalmo)
    assert "supports_gradient_checkpointing = True" in source, (
        "the port no longer advertises checkpointing support — re-read it before trusting "
        "either setting"
    )
    assert "_gradient_checkpointing_func" not in source, (
        "multimolecule's RiNALMo now CALLS the checkpointing hook. Enabling "
        "gradient_checkpointing would finally save memory and admit a larger batch — "
        "re-run slurm/p3/stage2_sizing_smoke.sbatch and re-size conf/train/stage2.yaml."
    )


def test_a_point_that_measured_fewer_rows_than_it_requested_is_refused() -> None:
    """`batch 8 fits` must not be sayable when only 3 rows were put through the step.

    `measure_batch` truncates the batch to `len(ds)`, so a short row pool silently shrinks
    the thing being measured while the record still names the requested size — the same
    request-standing-in-for-measurement defect this whole harness exists to correct.
    """
    payload = {
        "measurements": [dict(m, measured_batch_size=3) for m in _measurements()],
        "population": {"n_rows": 200},
        "device": {"is_cuda": True, "name": "A4000", "total_memory_gib": 15.6},
        "checkpointing": {
            "on_peak_gib": 3.0,
            "off_peak_gib": 9.0,
            "usable_on_this_backbone": True,
        },
        "config": {},
        "backbone": {
            **BR.backbone_summary(BR.resolve_backbone(BR.PRODUCTION_BACKBONE)),
            "requested_key": BR.PRODUCTION_BACKBONE,
        },
    }
    assert S.derive_clauses(S.build_report(**payload))["measured_the_requested_batch"] is False
    # Positive control: the honest fixture passes.
    assert S.derive_clauses(_report())["measured_the_requested_batch"] is True


def test_a_stale_recommendation_is_caught_by_the_validator() -> None:
    """The one field a reader acts on — it chose batch_size=4 for the production sweep.

    Nothing re-derived it, so a hand-edited or older-shape report could ship a recommendation
    that does not follow from its own measurements, past a green gate.
    """
    report = _report()
    assert S.validate_report(report) == []
    report["recommendation"] = {"batch_size": 8, "basis": "fits with headroom"}
    problems = S.validate_report(report)
    assert any("recommendation" in p for p in problems), problems


# ======================================================================================
# P3-17 — checkpointing is a property of the PORT, not a preference (ADR-0002 A15)
# ======================================================================================
def test_gradient_checkpointing_usability_is_recorded_per_backbone() -> None:
    """The two arms genuinely differ, and neither answer is "it helps".

    On the rotary production backbone checkpointing RUNS and is a measured no-op; on the
    RNA-FM port it RAISES, because that port adds its absolute position embeddings in place
    while checkpointing makes the embedding output a leaf requiring grad. SLURM job 1370 died
    on exactly that, in the sizing leg, before any training.
    """
    prod = BR.resolve_backbone(BR.PRODUCTION_BACKBONE)
    comp = BR.resolve_backbone(BR.COMPARATOR_BACKBONE)
    assert prod.gradient_checkpointing_usable is True
    assert comp.gradient_checkpointing_usable is False
    # The note must say WHY, or an absent measurement downstream is unexplainable.
    assert "in place" in comp.gradient_checkpointing_note.lower()
    assert "no-op" in prod.gradient_checkpointing_note.lower()


def test_a_config_that_enables_checkpointing_on_a_port_that_cannot_run_it_is_refused() -> None:
    """Refused at COMPOSE time on the login node — job 1370 discovered it after the queue."""
    kw = dict(
        checkpoint_dir=T.default_checkpoint_dir(BR.COMPARATOR_BACKBONE),
        report_path=T.default_report_path(BR.COMPARATOR_BACKBONE),
    )
    with pytest.raises(ValueError, match="not usable on backbone"):
        T.Stage2TrainConfig(backbone=BR.COMPARATOR_BACKBONE, gradient_checkpointing=True, **kw)
    # Positive controls, both directions: the comparator runs with it off, and the production
    # arm is unaffected — so the refusal is not firing on everything
    # ([[raises-test-needs-a-positive-control]]).
    assert (
        T.Stage2TrainConfig(
            backbone=BR.COMPARATOR_BACKBONE, gradient_checkpointing=False, **kw
        ).gradient_checkpointing
        is False
    )
    assert T.Stage2TrainConfig(gradient_checkpointing=True).gradient_checkpointing is True


def test_an_unmeasurable_checkpointing_comparison_is_STATED_not_omitted() -> None:
    """An absent measurement must be a recorded fact, not a missing key.

    The two arms' sizing reports are asymmetric — the comparator's carries one measurement
    fewer — and that asymmetry has to be visible to a reader rather than inferred.
    """
    comp = BR.resolve_backbone(BR.COMPARATOR_BACKBONE)
    payload = {
        "measurements": _measurements(gradient_checkpointing=False),
        "population": {"n_rows": 200, "n_tokens_max": 552},
        "device": {"is_cuda": True, "name": "NVIDIA RTX A4000", "total_memory_gib": 15.6},
        "checkpointing": {
            "batch_size": 2,
            "on_peak_gib": None,
            "off_peak_gib": None,
            "usable_on_this_backbone": False,
            "usability_note": comp.gradient_checkpointing_note,
            "comparison_skipped_reason": (
                f"gradient checkpointing is not usable on backbone {comp.key!r}, so there is "
                f"no 'on' arm to compare against: {comp.gradient_checkpointing_note}"
            ),
        },
        "config": {"batch_sweep": [4, 2], "steps": 6, "gradient_checkpointing": False},
        "backbone": {**BR.backbone_summary(comp), "requested_key": comp.key},
    }
    report = S.build_report(**payload)
    assert report["gate"]["overall_pass"] is True, report["gate"]["failed"]
    assert S.validate_report(report) == []
    ckpt = report["gradient_checkpointing"]
    assert ckpt["usable_on_this_backbone"] is False
    assert ckpt["comparison_skipped_reason"]
    assert report["backbone"]["requested_key"] == comp.key


def test_a_skip_reason_cannot_be_asserted_on_a_port_that_CAN_run_it() -> None:
    """`checkpointing_effect_measured` accepts any non-null reason, so on its own it lets a run
    opt out of the comparison by writing a sentence. `checkpointing_skip_is_earned` binds the
    skip to the report's own recorded evidence."""
    payload = {
        "measurements": _measurements(),
        "population": {"n_rows": 200, "n_tokens_max": 552},
        "device": {"is_cuda": True, "name": "A4000", "total_memory_gib": 15.6},
        "checkpointing": {
            "on_peak_gib": None,
            "off_peak_gib": None,
            "usable_on_this_backbone": True,  # it CAN run it...
            "comparison_skipped_reason": "we did not feel like measuring it",  # ...but didn't
        },
        "config": {},
        "backbone": {
            **BR.backbone_summary(BR.resolve_backbone(BR.PRODUCTION_BACKBONE)),
            "requested_key": BR.PRODUCTION_BACKBONE,
        },
    }
    clauses = S.derive_clauses(S.build_report(**payload))
    assert clauses["checkpointing_effect_measured"] is True, "the weaker clause is satisfied"
    assert clauses["checkpointing_skip_is_earned"] is False, "the binding clause must refuse"


def test_the_sweep_flag_must_match_the_recorded_usability() -> None:
    """A report claiming one configuration while its points measured another is refused —
    sizing a setting the arm cannot run measures a configuration that never trains."""
    comp = BR.resolve_backbone(BR.COMPARATOR_BACKBONE)
    payload = {
        # Points say they ran WITH checkpointing...
        "measurements": _measurements(gradient_checkpointing=True),
        "population": {"n_rows": 200, "n_tokens_max": 552},
        "device": {"is_cuda": True, "name": "A4000", "total_memory_gib": 15.6},
        # ...while the report says the port cannot run it.
        "checkpointing": {
            "on_peak_gib": None,
            "off_peak_gib": None,
            "usable_on_this_backbone": False,
            "comparison_skipped_reason": "not usable",
        },
        "config": {},
        "backbone": {**BR.backbone_summary(comp), "requested_key": comp.key},
    }
    assert S.derive_clauses(S.build_report(**payload))["checkpointing_skip_is_earned"] is False
