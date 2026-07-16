"""P1-16 — Stage-2 LoRA smoke fine-tune + VRAM gate (PRD §10.2 condition (b), VRAM half).

Three tiers, the tests/ml convention:

* **Tier 1 — pure stdlib**, runs + BLOCKS in bare CI. The P1-16 report validator and the
  gate adjudication, each proven to *bite* (a clean pass AND a violating fail, §8.7). This
  is the tier that keeps the 16 GB budget un-loosenable: if someone edits the gate so a
  4 GB-over report passes, these fail.
* **Tier 2 — torch/peft**, local or cluster only. One REAL LoRA fine-tune step end-to-end on
  a tiny same-architecture RiNALMo (CPU-sized), including the grad-flow check that the
  frozen-base + gradient-checkpointing combination does not silently train nothing.
* **Tier 3 — committed-report gate**, pure JSON, runs in CI once the SLURM run lands the
  artifact. Gated by ``TBOX_REQUIRE_LORA_SMOKE=1`` (wired into CI in the result commit, once
  the report exists — the P1-08 ``TBOX_REQUIRE_REPRO_GATE`` precedent).

**On a gate failure, do not touch the threshold.** ``VRAM_BUDGET_GIB`` is PRD §10.2's number;
a miss is a CLAUDE.md §7 stop-and-ask (escalate to FSDP ``FULL_SHARD`` or the §10.2 RNA-FM
swap), never a loosened bound (the ADR-0002 A7/A9 precedent).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tbox_finder.train import lora_harness as L

_REPO = Path(__file__).resolve().parents[2]
_SMOKE_REPORT = _REPO / "reports" / "p1" / "lora_vram_smoke.json"
_SBATCH = _REPO / "slurm" / "p1" / "lora_smoke.sbatch"

#: A measured-A4000 hardware block (what :func:`_hardware_block` returns on the cluster) —
#: passed to ``build_smoke_report`` so the bare tier can exercise the gate without torch.
_A4000 = {
    "device": "cuda:0",
    "gpu_name": "NVIDIA RTX A4000",
    "gpu_total_memory_gib": 15.6,
    "cuda_capability": [8, 6],
    "is_sm86": True,
    "n_gpus_measured": 1,
}
#: A CPU block — no VRAM number, and not the architecture §10.2 asks about.
_CPU = {
    "device": "cpu",
    "gpu_name": "cpu",
    "gpu_total_memory_gib": None,
    "cuda_capability": None,
    "is_sm86": False,
    "n_gpus_measured": 0,
}


def _fail_or_skip(reason: str) -> None:
    """Fail closed under ``TBOX_REQUIRE_LORA_SMOKE=1`` (CI once the report lands); skip
    gracefully otherwise (the SLURM run has not produced it yet).

    Guards the **pure-JSON Tier 3 only**. The torch tier has its own var
    (:func:`_torch_fail_or_skip`) — sharing one would make the var un-settable in CI: bare CI
    has no torch, so a single var that also armed the torch tier would turn the intended
    "the report must exist" contract into a guaranteed failure on every CI run.
    """
    if os.environ.get("TBOX_REQUIRE_LORA_SMOKE") == "1":
        pytest.fail(f"TBOX_REQUIRE_LORA_SMOKE=1 but the P1-16 VRAM gate is unrunnable: {reason}")
    pytest.skip(reason)


def _torch_fail_or_skip(reason: str) -> None:
    """Fail closed under ``TBOX_REQUIRE_LORA_TORCH=1`` (the local/cluster ml-rna env); skip
    gracefully otherwise (bare CI has no torch — it must not fail there)."""
    if os.environ.get("TBOX_REQUIRE_LORA_TORCH") == "1":
        pytest.fail(f"TBOX_REQUIRE_LORA_TORCH=1 but the P1-16 torch tier is unrunnable: {reason}")
    pytest.skip(reason)


def _stack_or_skip():
    """The pinned ml-rna stack (torch + peft + multimolecule), or skip."""
    if os.environ.get("CUDA_HOME") is None:
        # `import multimolecule` probes CUDA_HOME at import (deepspeed transitive) —
        # ADR-0002 A5 operational note.
        _torch_fail_or_skip("CUDA_HOME unset — multimolecule is not importable")
    try:
        import multimolecule  # noqa: F401
        import peft  # noqa: F401
        import torch
    except Exception as exc:  # noqa: BLE001
        _torch_fail_or_skip(f"pinned ml-rna stack unavailable ({exc})")
    return torch


def _good_smoke_report() -> dict:
    """A schema-valid, gate-passing P1-16 report — the mutation base for the bite tests."""
    return {
        "schema_version": L.SMOKE_SCHEMA_VERSION,
        "step": L.SMOKE_STEP,
        "measured": True,
        "backbone": {
            "repo_id": L.REPO_ID,
            "revision": L.REVISION,
            "hidden_dim": L.D_MODEL,
            "n_layer": L.N_LAYERS,
            "parity_confirmed": True,
        },
        "lora": {
            **L.lora_config_kwargs(),
            "dtype": L.TRAIN_DTYPE,
            "gradient_checkpointing": True,
        },
        "wrap": {
            "dtype": L.TRAIN_DTYPE,
            "gradient_checkpointing": True,
            "n_adapter_sites": 231,
            "applied_lora": {
                "r": L.LORA_R,
                "lora_alpha": L.LORA_ALPHA,
                "lora_dropout": L.LORA_DROPOUT,
                "n_modules_adapted": 231,
                "n_eligible_linear_modules": 231,
                "all_linear_fully_covered": True,
            },
        },
        "attention": {
            "selected": L.ATTN_FLASH2,
            "used": L.ATTN_FLASH2,
            "fallback": L.ATTN_SDPA,
            "reason": "FA-2 selected on measured sm_86 evidence",
            "model_supports_flash_attn": True,
            "sm86_evidence": {"source": "reports/p1/kernel_smoke.json", "is_sm86": True},
            "forward_verified_on_sm86": True,
            "forward_error": None,
        },
        "config": L.smoke_config(),
        "measured_smoke": {
            "gate_run": {
                "batch_size": 1,
                "n_timed_steps": 5,
                "steps_per_sec": 2.5,
                "peak_vram_gib": 4.2,
                "grad_flow": {
                    "n_lora_params": 462,
                    "n_lora_params_with_grad": 462,
                    "all_lora_grads_finite": True,
                    "all_lora_params_received_grad": True,
                },
            },
            "batch_sweep": {"1": {"peak_vram_gib": 4.2}},
            "device_total_memory_gib": 15.6,
        },
        "vram_budget": {
            "budget_gib": L.VRAM_BUDGET_GIB,
            "source": "PRD §10.2 condition (b)",
            "peak_vram_gib": 4.2,
            "device_total_memory_gib": 15.6,
            "within_budget": True,
            "on_miss": "§7 stop-and-ask",
        },
        "hardware": {
            "device": "cuda:0",
            "gpu_name": "NVIDIA RTX A4000",
            "gpu_total_memory_gib": 15.6,
            "cuda_capability": [8, 6],
            "is_sm86": True,
            "n_gpus_measured": 1,
        },
        "env": {"python": "3.12.0", "platform": "linux"},
        "gate": {
            "step_runs_end_to_end": True,
            "lora_grads_flow": True,
            "lora_config_exact": True,
            "peak_vram_within_budget": True,
            "fits_on_device": True,
            "measured_on_sm86": True,
            "bf16_and_gradient_checkpointing": True,
            "attention_forward_verified": True,
            "steps_per_sec_reported": True,
            "overall_pass": True,
        },
        "git_sha": "deadbeef",
    }


# ======================================================================================
# Tier 1 — pure stdlib (runs + blocks in bare CI)
# ======================================================================================
def test_vram_budget_is_the_prd_pin():
    """The ONLY number PRD §10.2/§10.3/§15 + ADR-0002 D6(b) pin for this step."""
    assert L.VRAM_BUDGET_GIB == 16.0, (
        "VRAM_BUDGET_GIB is PRD §10.2 condition (b)'s A4000 16 GB budget. A miss is a "
        "CLAUDE.md §7 stop-and-ask (escalate to FSDP FULL_SHARD or the §10.2 RNA-FM swap) — "
        "never a loosened threshold (the ADR-0002 A7/A9 precedent)."
    )


def test_smoke_shape_is_recorded_as_an_implementer_choice_not_a_pin():
    """Nothing pins batch/length/steps — the report must not imply otherwise (§10.3)."""
    cfg = L.smoke_config()
    assert cfg["pinned"] is False
    assert cfg["is_science"] is False
    assert cfg["data_is_synthetic"] is True
    assert cfg["loss_is_placeholder"] is True
    # The chosen length must sit in PRD §10.2's stated T-box locus range (~150-350 nt).
    assert 150 <= cfg["seq_len_nt"] <= 350
    assert cfg["gate_batch_size"] >= 1
    assert cfg["dtype"] == L.TRAIN_DTYPE == "bfloat16"
    assert cfg["gradient_checkpointing"] is True


def test_good_smoke_report_validates():
    assert L.validate_smoke_report(_good_smoke_report()) == []


@pytest.mark.parametrize(
    "garbage", [None, 3, "x", [], [1, 2], True], ids=["none", "int", "str", "list", "list2", "bool"]
)
def test_smoke_validator_never_raises_on_garbage(garbage):
    """A truncated artifact must produce an error list, not an opaque crash."""
    errs = L.validate_smoke_report(garbage)
    assert errs and "not a mapping" in errs[0]


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        # --- schema ---
        (lambda r: r.pop("gate"), "missing block: gate"),
        (lambda r: r.pop("measured_smoke"), "missing block: measured_smoke"),
        (lambda r: r.pop("config"), "missing block: config"),
        (lambda r: r.__setitem__("gate", "nope"), "block is not a mapping: gate"),
        (lambda r: r.__setitem__("schema_version", "99"), "schema_version"),
        (lambda r: r.__setitem__("step", "P1-15"), "step !="),
        # --- the backbone the claim is about ---
        (lambda r: r["backbone"].__setitem__("revision", "abc123"), "parity-confirmed"),
        (lambda r: r["backbone"].__setitem__("repo_id", "other/model"), "repo_id"),
        # --- the §10.3 contract the budget is claimed under ---
        (lambda r: r["lora"].__setitem__("r", 8), "lora.r != PRD §10.3"),
        (lambda r: r["lora"].__setitem__("lora_alpha", 16), "lora.lora_alpha"),
        (lambda r: r["lora"].__setitem__("lora_dropout", 0.1), "lora.lora_dropout"),
        (lambda r: r["lora"].__setitem__("target_modules", ["q"]), "lora.target_modules"),
        (lambda r: r["lora"].__setitem__("dtype", "float32"), "pins bf16"),
        (
            lambda r: r["lora"].__setitem__("gradient_checkpointing", False),
            "gradient_checkpointing",
        ),
        # --- §10.3 honesty: the smoke may not promote itself to science ---
        (lambda r: r["config"].__setitem__("is_science", True), "config.is_science"),
        (lambda r: r["config"].__setitem__("pinned", True), "config.pinned"),
        (lambda r: r["config"].__setitem__("loss_is_placeholder", False), "loss_is_placeholder"),
        (lambda r: r["config"].__setitem__("data_is_synthetic", False), "data_is_synthetic"),
        # --- the VRAM claim ---
        # A report that quietly raised its own budget must be rejected — this is the guard
        # that makes "16 GB is not weakenable" a property of the artifact, not a promise.
        (lambda r: r["vram_budget"].__setitem__("budget_gib", 24.0), "not weakenable"),
        (
            lambda r: r["vram_budget"].__setitem__("peak_vram_gib", None),
            "must be a positive number",
        ),
        (lambda r: r["vram_budget"].__setitem__("peak_vram_gib", 0), "must be a positive number"),
        (lambda r: r["vram_budget"].__setitem__("within_budget", "yes"), "must be a bool"),
        # the verdict must FOLLOW from the number
        (lambda r: r["vram_budget"].__setitem__("peak_vram_gib", 20.5), "does not follow"),
        # --- the attention handoff ---
        (lambda r: r["attention"].__setitem__("selected", "xformers"), "not a known backend"),
        (lambda r: r["attention"].__setitem__("used", "xformers"), "not a known backend"),
        (lambda r: r["attention"].__setitem__("fallback", "eager"), "attention.fallback"),
        (
            lambda r: r["attention"].__setitem__("forward_verified_on_sm86", "true"),
            "must be a bool",
        ),
        # a fallback run cannot have verified the selection
        (lambda r: r["attention"].__setitem__("used", L.ATTN_SDPA), "cannot have verified"),
        # --- gate consistency ---
        (lambda r: r["gate"].__setitem__("measured_on_sm86", False), "AND of the individual"),
        (lambda r: r["gate"].__setitem__("attention_forward_verified", False), "AND of the"),
        (lambda r: r["gate"].__setitem__("peak_vram_within_budget", False), "!= vram_budget"),
        (lambda r: r["gate"].pop("lora_grads_flow"), "gate missing key: lora_grads_flow"),
        (lambda r: r["gate"].pop("fits_on_device"), "gate missing key: fits_on_device"),
        # --- the mechanics evidence ---
        (
            lambda r: r["measured_smoke"]["gate_run"].pop("grad_flow"),
            "unevidenced",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"]["grad_flow"].__setitem__(
                "n_lora_params_with_grad", 0
            ),
            "did not train the adapters",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"]["grad_flow"].__setitem__("n_lora_params", 0),
            "nothing was adapted",
        ),
        (lambda r: r["measured_smoke"].pop("gate_run"), "gate_run must be a mapping"),
        # --- the evidence blocks the gate is re-derived from must be REQUIRED ---
        (lambda r: r.pop("wrap"), "missing block: wrap"),
        (lambda r: r.pop("hardware"), "missing block: hardware"),
        # --- FABRICATED-TRUE clauses: the direction `all(gate)` structurally cannot catch.
        # Each mutates the EVIDENCE and leaves the gate all-True, so only re-derivation bites.
        (
            lambda r: r["measured_smoke"].__setitem__("device_total_memory_gib", 3.0),
            "gate.fits_on_device does not follow",
        ),
        (
            lambda r: r["hardware"].__setitem__("cuda_capability", [8, 9]),
            "gate.measured_on_sm86 does not follow",
        ),
        (
            lambda r: r["hardware"].__setitem__("is_sm86", False),
            "gate.measured_on_sm86 does not follow",
        ),
        (
            lambda r: r["wrap"].__setitem__("dtype", "float32"),
            "gate.bf16_and_gradient_checkpointing does not follow",
        ),
        (
            lambda r: r["wrap"].__setitem__("gradient_checkpointing", False),
            "gate.bf16_and_gradient_checkpointing does not follow",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"].pop("steps_per_sec"),
            "gate.steps_per_sec_reported does not follow",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"].__setitem__("steps_per_sec", "fast"),
            "gate.steps_per_sec_reported does not follow",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"].__setitem__("n_timed_steps", 0),
            "gate.step_runs_end_to_end does not follow",
        ),
        # --- the P1-15 anti-tautology lesson, applied to the P1-16 gate ---
        (
            lambda r: r["wrap"]["applied_lora"].__setitem__("r", 8),
            "wrap.applied_lora.r != PRD §10.3",
        ),
        (
            lambda r: r["wrap"]["applied_lora"].__setitem__("lora_alpha", 64),
            "wrap.applied_lora.lora_alpha != PRD §10.3",
        ),
        (
            lambda r: r["wrap"]["applied_lora"].__setitem__("all_linear_fully_covered", False),
            "gate.lora_config_exact does not follow",
        ),
        (
            lambda r: r["wrap"]["applied_lora"].__setitem__("n_modules_adapted", 0),
            "gate.lora_config_exact does not follow",
        ),
        (lambda r: r["wrap"].pop("applied_lora"), "wrap.applied_lora is missing"),
        (lambda r: r["gate"].pop("lora_config_exact"), "gate missing key: lora_config_exact"),
        # --- CodeRabbit r1: the duplicated peak must agree with the budget's ---
        (
            lambda r: r["measured_smoke"]["gate_run"].__setitem__("peak_vram_gib", 9.9),
            "!= vram_budget.peak_vram_gib",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"].pop("peak_vram_gib"),
            "!= vram_budget.peak_vram_gib",
        ),
        # --- CodeRabbit r1: a bool is not a count / not a step tally ---
        (
            lambda r: r["measured_smoke"]["gate_run"].__setitem__("n_timed_steps", True),
            "gate.step_runs_end_to_end does not follow",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"]["grad_flow"].__setitem__(
                "n_lora_params", True
            ),
            "must be a positive int",
        ),
        # --- CodeRabbit r1: the handoff + mechanics clauses must follow from evidence ---
        # A fabricated-TRUE handoff clause over a recorded forward FAILURE. The mutation also
        # sets forward_error, so it clears the earlier "must say why" guard and actually
        # reaches the re-derivation this case is here to prove.
        (
            lambda r: (
                r["attention"].__setitem__("forward_verified_on_sm86", False),
                r["attention"].__setitem__("forward_error", "ImportError: flash_attn"),
            ),
            "gate.attention_forward_verified does not follow",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"]["grad_flow"].__setitem__(
                "all_lora_grads_finite", False
            ),
            "gate.lora_grads_flow does not follow",
        ),
        (
            lambda r: r["measured_smoke"]["gate_run"]["grad_flow"].__setitem__(
                "all_lora_params_received_grad", False
            ),
            "gate.lora_grads_flow does not follow",
        ),
        # --- CodeRabbit r1: the §10.2 question is about ONE A4000, not any sm_86 card ---
        (
            lambda r: r["hardware"].__setitem__("gpu_name", "NVIDIA A10"),
            "gate.measured_on_sm86 does not follow",
        ),
        (
            lambda r: r["hardware"].__setitem__("n_gpus_measured", 2),
            "gate.measured_on_sm86 does not follow",
        ),
        # A scalar capability must not crash list() — the validator's totality contract.
        (
            lambda r: r["hardware"].__setitem__("cuda_capability", 86),
            "gate.measured_on_sm86 does not follow",
        ),
    ],
)
def test_smoke_validator_bites(mutate, needle):
    """Every honesty/consistency guard must actually fail on its violation (§8.7)."""
    report = _good_smoke_report()
    mutate(report)
    errs = L.validate_smoke_report(report)
    assert any(needle in e for e in errs), f"expected {needle!r} in {errs}"


def test_a_fabricated_true_gate_clause_cannot_survive_its_evidence():
    """The load-bearing property: `all(gate)` catches a clause flipped FALSE, but only
    re-derivation catches one fabricated TRUE. A report whose gate is entirely True over
    evidence that contradicts it must be rejected — otherwise the committed artifact is
    an assertion rather than a measurement (§10.3)."""
    report = _good_smoke_report()
    # Every clause still True; only the underlying evidence is tampered with.
    report["hardware"]["cuda_capability"] = [8, 9]  # an sm_89 laptop, not the A4000
    report["wrap"]["dtype"] = "float32"  # not bf16
    report["measured_smoke"]["device_total_memory_gib"] = 3.0  # peak 4.2 cannot fit
    assert report["gate"]["overall_pass"] is True
    errs = L.validate_smoke_report(report)
    assert any("measured_on_sm86 does not follow" in e for e in errs)
    assert any("bf16_and_gradient_checkpointing does not follow" in e for e in errs)
    assert any("fits_on_device does not follow" in e for e in errs)


def test_gate_clauses_are_shared_derivations_not_parallel_copies():
    """build and validate must compute each clause from ONE definition, or they can drift."""
    assert L._is_sm86(_A4000) is True
    assert L._is_sm86({**_A4000, "cuda_capability": [8, 9]}) is False
    assert L._is_sm86({**_A4000, "is_sm86": False}) is False
    # PRD §10.2 (b) asks about ONE A4000 — not any sm_86 card, not a multi-GPU measurement.
    assert L._is_sm86({**_A4000, "gpu_name": "NVIDIA A10"}) is False
    assert L._is_sm86({**_A4000, "n_gpus_measured": 2}) is False
    assert L._is_sm86({**_A4000, "n_gpus_measured": 0}) is False
    assert L._is_sm86({}) is False
    assert L._is_sm86(_CPU) is False
    assert L._bf16_and_ckpt({"dtype": "bfloat16", "gradient_checkpointing": True}) is True
    assert L._bf16_and_ckpt({"dtype": "float32", "gradient_checkpointing": True}) is False
    assert L._bf16_and_ckpt({"dtype": "bfloat16", "gradient_checkpointing": False}) is False
    # A bool must never be read as a number (isinstance(True, int) is True in Python).
    assert L._num(True) is None
    assert L._num(False) is None
    assert L._num("4.2") is None
    assert L._num(4.2) == 4.2
    assert L._num(4) == 4.0
    assert L._pos_int(True) is None
    assert L._pos_int(0) is None
    assert L._pos_int(-1) is None
    assert L._pos_int(3) == 3
    # The mechanics derivation reads the COUNTS, not the summary booleans.
    good = {
        "n_lora_params": 4,
        "n_lora_params_with_grad": 4,
        "all_lora_grads_finite": True,
        "all_lora_params_received_grad": True,
    }
    assert L._grads_flow(good) is True
    assert L._grads_flow({**good, "n_lora_params_with_grad": 3}) is False
    assert L._grads_flow({**good, "all_lora_grads_finite": False}) is False
    assert L._grads_flow({**good, "n_lora_params": True}) is False
    assert L._grads_flow({}) is False


@pytest.mark.parametrize(
    "cap",
    [86, "8.6", None, {"major": 8}, [8], [8, 6, 1]],
    ids=["int", "str", "none", "dict", "short", "long"],
)
def test_a_malformed_capability_cannot_crash_the_validator(cap):
    """`list(86)` raises TypeError — the validator must RETURN errors, never crash (§8.7).

    Its callers (CI, `run_vram_smoke`'s fail-closed publish) rely on an error list; a crash
    turns a clean gate failure into an opaque traceback.
    """
    report = _good_smoke_report()
    report["hardware"]["cuda_capability"] = cap
    errs = L.validate_smoke_report(report)  # must not raise
    assert any("measured_on_sm86 does not follow" in e for e in errs)


def test_unmeasured_smoke_report_cannot_claim_a_pass():
    report = _good_smoke_report()
    report["measured"] = False
    errs = L.validate_smoke_report(report)
    assert any("measured is not set" in e for e in errs)


def test_measured_must_be_a_strict_bool():
    """A truthy string must not walk into the measured path (§8.7)."""
    report = _good_smoke_report()
    report["measured"] = "false"  # truthy!
    errs = L.validate_smoke_report(report)
    assert any("must be a bool" in e for e in errs)


def test_an_over_budget_report_cannot_pass_the_gate():
    """The load-bearing one: a peak over 16 GB must fail, not be waved through.

    Note the card size here is 24 GiB, not the A4000's 15.6. That is deliberate and is what
    makes the scenario coherent: it isolates the **budget** clause (peak > the PRD's 16 GB)
    from the **device** clause (peak still fits the card). On a real 15.6 GiB A4000 a peak
    above 16 GiB is unreachable — the allocator OOMs first — so the honest §10.2 (b) failure
    signature on that hardware is an OOM in the gate run, which propagates out of
    `run_vram_smoke`, leaves no DONE marker, and becomes a §9.3 stop-and-ask. This test
    covers the adjudication logic, not the physics.
    """
    report = _good_smoke_report()
    over = L.VRAM_BUDGET_GIB + 0.1
    report["vram_budget"]["peak_vram_gib"] = over
    report["vram_budget"]["within_budget"] = False
    report["vram_budget"]["device_total_memory_gib"] = 24.0
    report["measured_smoke"]["gate_run"]["peak_vram_gib"] = over
    report["measured_smoke"]["device_total_memory_gib"] = 24.0
    report["hardware"]["gpu_total_memory_gib"] = 24.0
    report["gate"]["peak_vram_within_budget"] = False
    report["gate"]["overall_pass"] = False
    # It must VALIDATE (an honest failing report is well-formed) but not claim a pass.
    assert L.validate_smoke_report(report) == []
    assert report["gate"]["overall_pass"] is False

    # ...and a report that claims the pass anyway must be rejected.
    report["gate"]["overall_pass"] = True
    errs = L.validate_smoke_report(report)
    assert any("AND of the individual" in e for e in errs)


def test_build_smoke_report_verdict_follows_from_the_measurement():
    """within_budget/overall_pass are COMPUTED from the peak, never asserted."""
    wrap = {
        "dtype": L.TRAIN_DTYPE,
        "gradient_checkpointing": True,
        "n_adapter_sites": 4,
        "applied_lora": {
            "r": L.LORA_R,
            "lora_alpha": L.LORA_ALPHA,
            "lora_dropout": L.LORA_DROPOUT,
            "n_modules_adapted": 4,
            "all_linear_fully_covered": True,
        },
    }
    grad_flow = {
        "n_lora_params": 8,
        "n_lora_params_with_grad": 8,
        "all_lora_grads_finite": True,
        "all_lora_params_received_grad": True,
    }

    def _measured(peak):
        return {
            "gate_run": {
                "batch_size": 1,
                "n_timed_steps": 5,
                "steps_per_sec": 2.0,
                "peak_vram_gib": peak,
                "grad_flow": grad_flow,
            },
            "batch_sweep": {},
            "device_total_memory_gib": 15.6,
        }

    common = dict(
        wrap_info=wrap,
        attn_selected=L.ATTN_FLASH2,
        attn_used=L.ATTN_FLASH2,
        attn_reason="r",
        forward_verified=True,
        forward_error=None,
        evidence={"source": "s", "is_sm86": True},
        supports_fa=True,
        hardware=_A4000,
    )
    under = L.build_smoke_report(measured_smoke=_measured(4.0), **common)
    assert under["vram_budget"]["within_budget"] is True
    assert under["gate"]["peak_vram_within_budget"] is True

    over = L.build_smoke_report(measured_smoke=_measured(17.9), **common)
    assert over["vram_budget"]["within_budget"] is False
    assert over["gate"]["peak_vram_within_budget"] is False
    assert over["gate"]["overall_pass"] is False


def test_a_cpu_run_cannot_certify_the_a4000_budget():
    """No VRAM number (device=cpu) must never read as a pass — fail closed (§10.3)."""
    report = L.build_smoke_report(
        wrap_info={"dtype": L.TRAIN_DTYPE, "gradient_checkpointing": True},
        measured_smoke={
            "gate_run": {
                "n_timed_steps": 5,
                "steps_per_sec": 1.0,
                "peak_vram_gib": None,  # a CPU run measures no VRAM
                "grad_flow": {
                    "n_lora_params": 8,
                    "n_lora_params_with_grad": 8,
                    "all_lora_grads_finite": True,
                    "all_lora_params_received_grad": True,
                },
            },
            "batch_sweep": {},
            "device_total_memory_gib": None,
        },
        attn_selected=L.ATTN_SDPA,
        attn_used=L.ATTN_SDPA,
        attn_reason="r",
        forward_verified=True,
        forward_error=None,
        evidence={"source": "s", "is_sm86": True},
        supports_fa=True,
        hardware=_CPU,
    )
    assert report["gate"]["peak_vram_within_budget"] is False
    assert report["gate"]["measured_on_sm86"] is False
    assert report["gate"]["overall_pass"] is False


def test_forward_failure_records_the_error_and_fails_the_gate():
    """An FA-2 forward failure must be recorded + fail closed, never silently absorbed."""
    report = L.build_smoke_report(
        wrap_info={"dtype": L.TRAIN_DTYPE, "gradient_checkpointing": True},
        measured_smoke={
            "gate_run": {
                "n_timed_steps": 5,
                "steps_per_sec": 1.0,
                "peak_vram_gib": 4.0,
                "grad_flow": {
                    "n_lora_params": 8,
                    "n_lora_params_with_grad": 8,
                    "all_lora_grads_finite": True,
                    "all_lora_params_received_grad": True,
                },
            },
            "batch_sweep": {},
            "device_total_memory_gib": 15.6,
        },
        attn_selected=L.ATTN_FLASH2,
        attn_used=L.ATTN_SDPA,  # fell back
        attn_reason="r",
        forward_verified=False,
        forward_error="ImportError: flash_attn",
        evidence={"source": "s", "is_sm86": True},
        supports_fa=True,
        hardware=_A4000,
    )
    assert report["attention"]["forward_error"] == "ImportError: flash_attn"
    assert report["gate"]["attention_forward_verified"] is False
    assert report["gate"]["overall_pass"] is False
    # And the artifact is still well-formed — an honest failure validates.
    assert L.validate_smoke_report(report) == []


def test_unverified_forward_without_a_reason_is_rejected():
    report = _good_smoke_report()
    report["attention"]["forward_verified_on_sm86"] = False
    report["attention"]["forward_error"] = None
    report["gate"]["attention_forward_verified"] = False
    report["gate"]["overall_pass"] = False
    errs = L.validate_smoke_report(report)
    assert any("must say why" in e for e in errs)


def test_sbatch_exists_and_requests_exactly_one_a4000():
    """The §10.2 question is about ONE A4000; the sbatch must ask for exactly that (§9.3)."""
    assert _SBATCH.is_file(), f"{_SBATCH} missing"
    body = _SBATCH.read_text()
    # Read the DIRECTIVES, not the prose: the header comments legitimately mention
    # `--nodelist`/`--account` to say why they are absent, and a substring scan over the whole
    # file would fail on the explanation rather than on a real pin.
    directives = [ln.strip() for ln in body.splitlines() if ln.startswith("#SBATCH")]
    joined = "\n".join(directives)
    assert "--partition=gpu" in joined
    assert "--gres=gpu:a4000:1" in joined, "the §10.2 question is about ONE A4000"
    assert "--nodelist" not in joined, "never pin a node (CLAUDE.md §9.2 — `one` is often down)"
    assert "--account" not in joined and "--qos" not in joined, "accounting is disabled (§9.2)"
    assert "--require-cuda" in body, "a CPU run cannot answer the §10.2 A4000 question"
    assert "conda activate tbox-ml-rna" in body
    assert 'export CUDA_HOME="$CONDA_PREFIX"' in body, "multimolecule import probes CUDA_HOME (A4)"


# ======================================================================================
# Tier 2 — torch/peft: one REAL LoRA fine-tune step end-to-end (local/cluster only)
# ======================================================================================
def test_one_lora_step_runs_end_to_end_and_trains_the_adapters():
    """The mechanics claim, on a real (tiny) RiNALMo: a step runs AND the adapters move.

    This is the CPU-sized proof of the same code path the A4000 runs. It bites on the
    frozen-base + gradient-checkpointing failure mode: if the checkpointed segments never
    receive grad-requiring inputs, the step still "succeeds" while every LoRA gradient stays
    None — a smoke that measures the VRAM of nothing.
    """
    torch = _stack_or_skip()
    from multimolecule import RiNALMoConfig, RiNALMoModel

    torch.manual_seed(L.SMOKE_SEED)
    cfg = RiNALMoConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128
    )
    cfg._attn_implementation = L.ATTN_SDPA  # CPU tier — FA-2 needs a GPU
    base = RiNALMoModel(cfg, add_pooling_layer=False)

    peft_model, info = L.build_peft_model(
        base_model=base, dtype="float32", attn_implementation=None, gradient_checkpointing=True
    )
    assert info["base_frozen"] is True
    peft_model.train()

    before = {
        n: p.detach().clone()
        for n, p in peft_model.named_parameters()
        if "lora_B" in n and p.requires_grad
    }
    assert before, "no trainable lora_B params — the wrap adapted nothing"

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    ids = torch.randint(4, 9, (2, 24))
    out = peft_model(input_ids=ids)
    loss = L.placeholder_loss(out.last_hidden_state)
    loss.backward()
    flow = L.measure_lora_grad_flow(peft_model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    assert flow["n_lora_params"] > 0
    assert flow["all_lora_params_received_grad"] is True, (
        f"only {flow['n_lora_params_with_grad']}/{flow['n_lora_params']} LoRA params got a "
        "gradient — the step ran but trained nothing (frozen base + gradient checkpointing "
        "needs enable_input_require_grads)"
    )
    assert flow["all_lora_grads_finite"] is True

    # lora_B initialises to zeros, so a real step MUST move it — proof the optimizer bit.
    moved = [
        n
        for n, old in before.items()
        if not torch.equal(old, dict(peft_model.named_parameters())[n])
    ]
    assert moved, "no lora_B parameter changed after optimizer.step() — nothing was trained"


def test_smoke_step_returns_a_finite_loss_and_clears_grads():
    """:func:`smoke_step` is the imp.md entry — it must complete a whole step, not half of one."""
    torch = _stack_or_skip()
    from multimolecule import RiNALMoConfig, RiNALMoModel

    torch.manual_seed(L.SMOKE_SEED)
    cfg = RiNALMoConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
    )
    cfg._attn_implementation = L.ATTN_SDPA
    base = RiNALMoModel(cfg, add_pooling_layer=False)
    peft_model, _ = L.build_peft_model(
        base_model=base, dtype="float32", attn_implementation=None, gradient_checkpointing=True
    )
    peft_model.train()
    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-3)
    loss = L.smoke_step(peft_model, optimizer, torch.randint(4, 9, (1, 16)))
    assert isinstance(loss, float)
    assert loss == loss and abs(loss) != float("inf"), f"non-finite loss {loss}"
    # zero_grad(set_to_none=True) ran → the grads are released between steps, which is what
    # makes the *peak* (not the post-step reading) the honest VRAM number.
    assert all(
        p.grad is None for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad
    )


def test_placeholder_loss_is_scalar_and_differentiable():
    torch = _stack_or_skip()
    h = torch.randn(2, 8, 16, requires_grad=True)
    loss = L.placeholder_loss(h)
    assert loss.ndim == 0
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


# ======================================================================================
# Tier 3 — committed-report gate (pure JSON; runs in CI once the SLURM report lands)
# ======================================================================================
def test_committed_smoke_report_passes_the_vram_gate():
    """The PRD §10.2 condition-(b) VRAM gate, read off the measured artifact.

    On a FAIL: CLAUDE.md §7 stop-and-ask — escalate to the FSDP FULL_SHARD fallback or the
    §10.2 RNA-FM swap consideration. Do NOT raise VRAM_BUDGET_GIB, and do not relax this test.
    """
    if not _SMOKE_REPORT.is_file():
        _fail_or_skip(
            "P1-16 report reports/p1/lora_vram_smoke.json absent (SLURM run pending — §9.3)"
        )
    report = json.loads(_SMOKE_REPORT.read_text())

    errs = L.validate_smoke_report(report)
    assert errs == [], f"the committed P1-16 report fails its own validator: {errs}"

    assert report["measured"] is True
    peak = report["vram_budget"]["peak_vram_gib"]
    assert peak <= L.VRAM_BUDGET_GIB, (
        f"PRD §10.2 condition (b) VRAM gate FAILED: peak {peak} GiB > the pinned "
        f"{L.VRAM_BUDGET_GIB} GiB A4000 budget. This is a CLAUDE.md §7 stop-and-ask — "
        "escalate to FSDP FULL_SHARD sharding or the §10.2 RNA-FM swap. Never loosen the "
        "budget (the ADR-0002 A7/A9 precedent)."
    )
    assert report["gate"]["overall_pass"] is True, f"P1-16 gate did not pass: {report['gate']}"
    # It must have been measured on the architecture the claim is about.
    assert report["hardware"]["is_sm86"] is True
    assert report["hardware"]["cuda_capability"] == [8, 6]
    # ...under the config §10.3 pins.
    assert report["lora"]["r"] == L.LORA_R
    assert report["lora"]["dtype"] == L.TRAIN_DTYPE
    assert report["lora"]["gradient_checkpointing"] is True
    # ...and it must not have quietly become a science claim.
    assert report["config"]["is_science"] is False


def test_committed_smoke_report_closes_the_p1_15_forward_handoff():
    """P1-15 selected a backend on an IMPORT confirmation and deferred the forward here."""
    if not _SMOKE_REPORT.is_file():
        _fail_or_skip("P1-16 report absent (SLURM run pending)")
    report = json.loads(_SMOKE_REPORT.read_text())
    attn = report["attention"]
    assert attn["forward_verified_on_sm86"] is True, (
        f"the P1-15-selected backend {attn['selected']!r} did not survive a forward through "
        f"RiNALMo on sm_86 (used {attn['used']!r}; error: {attn.get('forward_error')!r}). "
        "This is a §7 stop-and-ask: the recorded selection in "
        "reports/p1/attention_backend.json + conf/train/lora_stage2.yaml must flip to SDPA."
    )
    assert attn["used"] == attn["selected"]
