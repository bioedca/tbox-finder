"""P1-15 — Stage-2 LoRA/FSDP harness + FA-2/SDPA selection.

Tiers (the P1-10/P1-14 precedent):
  * **Bare (always)** — pure decision logic, the config drift-guard, the committed-report
    validity check, and validator bite-tests. Stdlib only; runs in CI with no torch.
  * **peft/torch tier** — gated on the pinned ml-rna stack being importable; wraps a *tiny*
    same-architecture RiNALMo so the tier needs no 2.5 GB download.
  * **giga tier** — the real parity-confirmed backbone; opt-in via ``TBOX_REQUIRE_RINALMO=1``
    (2.5 GB + CPU RAM), so CI never pulls it.

The bare tier is where the §10.3 gate is actually locked: every honesty invariant in
``validate_report`` gets a mutator proving it bites, because a validator that cannot fail is
not a gate (§8.7).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tbox_finder.train import lora_harness as L

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_REPORT = REPO_ROOT / "reports" / "p1" / "attention_backend.json"
KERNEL_SMOKE = REPO_ROOT / "reports" / "p1" / "kernel_smoke.json"
CONF = REPO_ROOT / "conf" / "train" / "lora_stage2.yaml"


# ====================================================================================== #
# The PRD §10.3 pinned LoRA contract.
# ====================================================================================== #
def test_lora_constants_are_the_prd_pins():
    """PRD §10.3 pins these four literally — a silent edit here is a spec violation."""
    assert L.LORA_R == 16
    assert L.LORA_ALPHA == 32
    assert L.LORA_DROPOUT == 0.05
    assert L.LORA_TARGET_MODULES == "all-linear"
    assert L.TRAIN_DTYPE == "bfloat16"


def test_lora_config_kwargs_exact():
    kw = L.lora_config_kwargs()
    assert kw["r"] == 16
    assert kw["lora_alpha"] == 32
    assert kw["lora_dropout"] == 0.05
    assert kw["target_modules"] == "all-linear"


def test_conf_yaml_echoes_the_code_pins_exactly():
    """Drift guard: conf/train/lora_stage2.yaml may echo the pins, never redefine them.

    The config is documentation; the code is the contract. If they disagree, a reader of the
    config would be misled about what actually runs (the rinalmo_stage2.yaml precedent).
    """
    text = CONF.read_text()
    # Parsed without yaml (bare CI has no pyyaml guarantee), but ANCHORED to a real mapping
    # line: a bare `"r: 16" in text` would happily match a comment (`# r: 16`) while a live
    # `r: 8` sat below it, and would also match `num_heads_per_layer: 16`.
    for key, want in (
        ("r", L.LORA_R),
        ("lora_alpha", L.LORA_ALPHA),
        ("lora_dropout", L.LORA_DROPOUT),
        ("target_modules", L.LORA_TARGET_MODULES),
        ("dtype", L.TRAIN_DTYPE),
        ("gradient_checkpointing", "true"),
        ("zero_stage", L.ZERO_STAGE),
        ("fsdp_sharding_strategy", L.FSDP_SHARDING_STRATEGY),
        ("fsdp_version", L.FSDP_VERSION),
        ("n_gpus", L.N_GPUS),
    ):
        pattern = rf"(?m)^\s+{re.escape(key)}:\s+{re.escape(str(want))}\s*(?:#.*)?$"
        assert re.search(
            pattern, text
        ), f"conf/train/lora_stage2.yaml lost/drifted the pin echo: {key}: {want}"
    assert re.search(r"(?m)^qlora:\n\s+enabled: false\s*$", text)


# ====================================================================================== #
# select_attention_backend — the PRD §10.3 decision, pure.
# ====================================================================================== #
def _fa_inputs(**over):
    base = {
        "flash_attn_importable": True,
        "sm86_confirmed": True,
        "model_supports_flash_attn": True,
        "dtype": "bfloat16",
    }
    base.update(over)
    return base


def test_fa2_selected_when_every_input_confirms():
    backend, reason = L.select_attention_backend(**_fa_inputs())
    assert backend == L.ATTN_FLASH2
    assert "MEASURED sm_86" in reason
    # The scope caveat must ride along — FA-2 is selected on an import confirmation, and the
    # forward is P1-16's (§10.3 no-overclaim).
    assert "P1-16" in reason


@pytest.mark.parametrize(
    "override,needle",
    [
        ({"sm86_confirmed": False}, "sm_86 not confirmed"),
        ({"flash_attn_importable": False}, "does not import"),
        ({"model_supports_flash_attn": False}, "do not advertise"),
        ({"dtype": "float32"}, "not half-precision"),
    ],
)
def test_sdpa_fallback_when_any_input_fails(override, needle):
    """Each blocking input must independently force the fallback — no input is decorative."""
    backend, reason = L.select_attention_backend(**_fa_inputs(**override))
    assert backend == L.ATTN_SDPA
    assert needle in reason


def test_fallback_is_never_eager_on_any_blocking_input():
    """ADR-0002 D3 names SDPA (an exact-softmax swap), not eager, as the RiNALMo fallback.

    Asserting `backend == ATTN_SDPA != ATTN_EAGER` would be tautological — the second clause
    compares two module constants and is always True regardless of `backend`. Sweep every
    blocking input instead and assert eager is never returned.
    """
    for override in (
        {"sm86_confirmed": False},
        {"flash_attn_importable": False},
        {"model_supports_flash_attn": False},
        {"dtype": "float32"},
    ):
        backend, _ = L.select_attention_backend(**_fa_inputs(**override))
        assert backend != L.ATTN_EAGER, f"eager must never be the fallback (input {override})"
        assert backend == L.ATTN_SDPA


def test_train_dtype_is_fa2_eligible():
    """A module invariant the validator relies on (it omits an unreachable dtype re-check):
    the PRD-pinned training dtype must be half-precision, or FA-2 could never be selected."""
    assert L.TRAIN_DTYPE in L.FA2_DTYPES


def test_select_attention_backend_rejects_non_str_dtype():
    with pytest.raises(TypeError):
        L.select_attention_backend(**_fa_inputs(dtype=16))


@pytest.mark.parametrize(
    "override",
    [
        {"flash_attn_importable": "false"},  # truthy string!
        {"sm86_confirmed": "no"},
        {"model_supports_flash_attn": 0.1},
        {"sm86_confirmed": 1},
    ],
)
def test_select_attention_backend_rejects_non_bool_flags(override):
    """§8.7: a truthy non-bool must never slip past a gate input. Without this,
    `sm86_confirmed="no"` (truthy) would SELECT FA-2 on evidence that says no."""
    with pytest.raises(TypeError):
        L.select_attention_backend(**_fa_inputs(**override))


# ====================================================================================== #
# read_sm86_flash_attn_evidence — fails closed (absence of evidence != confirmation).
# ====================================================================================== #
def test_reads_the_committed_a5_kernel_smoke():
    """The real, measured artifact: ADR-0002 A5 / SLURM job 462 on a cluster RTX A4000."""
    ev = L.read_sm86_flash_attn_evidence(KERNEL_SMOKE)
    assert ev["is_sm86"] is True
    assert ev["flash_attn_importable"] is True
    assert ev["device_capability"] == [8, 6]
    assert "A4000" in ev["device_name"]


def test_require_pinned_revision_rejects_unpinned():
    """The ADR-0002 D5/A9 parity-checkpoint pin, bare tier.

    `_require_pinned_revision` is pure stdlib, but its only other bite-test routes through
    `load_rinalmo_backbone`, whose `import torch` precedes the guard — so without this the
    pin would be entirely unguarded in bare CI.
    """
    for bad in ("main", "b" * 40, "", L.REVISION[:-1]):
        with pytest.raises(ValueError, match="parity-confirmed"):
            L._require_pinned_revision(bad)
    L._require_pinned_revision(L.REVISION)  # the pinned one must pass


def test_missing_evidence_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        L.read_sm86_flash_attn_evidence(tmp_path / "nope.json")


@pytest.mark.parametrize("payload", ["null", "123", '"a string"', "[1, 2, 3]"])
def test_valid_json_but_not_an_object_raises(tmp_path, payload):
    p = tmp_path / "x.json"
    p.write_text(payload)
    with pytest.raises(ValueError):
        L.read_sm86_flash_attn_evidence(p)


def test_self_inconsistent_arch_cannot_confirm_fa2(tmp_path):
    """The one input class that could FALSELY confirm: an artifact flagged is_sm86=true but
    whose recorded capability is some other card (e.g. this sm_89 laptop). Confirming FA-2
    for an architecture the measurement never touched is exactly the §10.3 hazard."""
    report = json.loads(KERNEL_SMOKE.read_text())
    report["env"]["device_capability"] = [8, 9]
    report["env"]["device_name"] = "NVIDIA GeForce RTX 4060 Laptop GPU"
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="self-inconsistent"):
        L.read_sm86_flash_attn_evidence(p)


def test_measured_on_target_arch_is_derived_not_hardcoded():
    ev = L.read_sm86_flash_attn_evidence(KERNEL_SMOKE)
    assert ev["measured_on_target_arch"] is True
    assert list(ev["device_capability"]) == list(L.SM86_CAPABILITY)


def test_unparseable_evidence_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ValueError):
        L.read_sm86_flash_attn_evidence(p)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.pop("env"), id="no-env-block"),
        pytest.param(lambda r: r.pop("imports"), id="no-imports-block"),
        pytest.param(lambda r: r["imports"].pop("flash_attn"), id="no-flash_attn-block"),
        pytest.param(lambda r: r["env"].__setitem__("is_sm86", "yes"), id="is_sm86-not-bool"),
        pytest.param(
            lambda r: r["imports"]["flash_attn"].__setitem__("ok", 1), id="fa-ok-not-bool"
        ),
        pytest.param(
            lambda r: r["gate"].__setitem__("overall_pass", False), id="smoke-gate-failed"
        ),
    ],
)
def test_evidence_reader_fails_closed(tmp_path, mutate):
    """A tampered/failed/incomplete smoke must never read as an FA-2 confirmation (§10.3)."""
    report = json.loads(KERNEL_SMOKE.read_text())
    mutate(report)
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        L.read_sm86_flash_attn_evidence(p)


# ====================================================================================== #
# Full-FT fallback configs (PRD §10.3).
# ====================================================================================== #
def test_zero3_config_shape():
    cfg = L.zero3_config()
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["bf16"]["enabled"] is True
    # No offload: §10.3 sizes the fallback to fit on the 8 GPUs.
    assert cfg["zero_optimization"]["offload_optimizer"]["device"] == "none"
    assert cfg["zero_optimization"]["offload_param"]["device"] == "none"
    # Batch fields stay "auto" — inventing a number here would be a fabricated compute
    # figure (§10.3); the real one is a P3 decision.
    assert cfg["train_batch_size"] == "auto"


@pytest.mark.parametrize("bad", [0, -1])
def test_zero3_config_rejects_bad_accumulation(bad):
    with pytest.raises(ValueError):
        L.zero3_config(gradient_accumulation_steps=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_zero3_config_rejects_bad_grad_clip(bad):
    with pytest.raises(ValueError):
        L.zero3_config(grad_clip=bad)


def test_fsdp_kwargs_are_the_prd_full_shard():
    kw = L.fsdp_plugin_kwargs()
    assert kw["sharding_strategy"] == "FULL_SHARD"
    assert kw["fsdp_version"] == 1
    assert kw["transformer_cls_names_to_wrap"] == ["RiNALMoLayer"]
    assert kw["activation_checkpointing"] is True


# ====================================================================================== #
# QLoRA — flagged off, and it cannot be turned on silently.
# ====================================================================================== #
def test_qlora_disabled_records_why():
    q = L.qlora_config()
    assert q["enabled"] is False
    assert q["dependency"] == "bitsandbytes"
    assert q["dependency_pinned_in_env"] is False


def test_qlora_enable_raises_pointing_at_the_env_gate():
    """PRD §10.3 admits QLoRA 'only if needed'; its dep is not in the pinned env, so the
    flag must fail loud rather than stub or ad-hoc-install (§3.1/§10.3)."""
    with pytest.raises(NotImplementedError, match="bitsandbytes"):
        L.qlora_config(enabled=True)


def test_bitsandbytes_really_is_absent_from_the_pinned_env():
    """The claim `dependency_pinned_in_env: False` must be TRUE of the actual env files —
    a claim about the env that the env could contradict is a §10.3 hazard."""
    for f in ("envs/ml-rna.yml", "envs/ml-rna.conda-lock.yml"):
        text = (REPO_ROOT / f).read_text().lower()
        assert "bitsandbytes" not in text, f"{f} now pins bitsandbytes — qlora_config() lies"


# ====================================================================================== #
# validate_report — every invariant must bite.
# ====================================================================================== #
def _good_report() -> dict:
    """A minimal schema-valid measured report (mirrors what build_report emits)."""
    return {
        "schema_version": L.SCHEMA_VERSION,
        "step": L.STEP,
        "measured": True,
        "backbone": {"repo_id": L.REPO_ID, "revision": L.REVISION},
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "bias": "none",
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
        "wrap": {
            "trainable_params": 100,
            "total_params": 10_000,
            "base_frozen": True,
            "tiny_fixture": False,
            "n_adapter_sites": 231,
            # The APPLIED config read back off the model — what the §10.3 gate must check.
            "applied_lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules_selector": "all-linear",
                "n_modules_adapted": 231,
            },
        },
        "attention": {
            "selected": L.ATTN_FLASH2,
            "fallback": L.ATTN_SDPA,
            "reason": "because the evidence says so",
            "model_supports_flash_attn": True,
            "forward_verified_on_sm86": False,
            "sm86_evidence": {
                "source": "reports/p1/kernel_smoke.json",
                "is_sm86": True,
                "flash_attn_importable": True,
                "device_name": "NVIDIA RTX A4000",
            },
        },
        "full_ft_fallback": {
            "zero_stage": 3,
            "fsdp_sharding_strategy": "FULL_SHARD",
            "validated": True,
        },
        "qlora": {"enabled": False, "dependency_pinned_in_env": False},
        "env": {"python": "3.12.0"},
        "gate": {
            "lora_config_exact": True,
            "wraps_rinalmo": True,
            "base_frozen": True,
            "fallback_config_validates": True,
            "attention_backend_recorded": True,
            "overall_pass": True,
        },
    }


def test_good_report_validates():
    assert L.validate_report(_good_report()) == []


def test_validator_never_raises_on_garbage():
    """Must not raise AND must not validate. Asserting only `isinstance(..., list)` would be
    vacuous — the return annotation already promises that, and a mutant returning [] for
    everything would pass."""
    for junk in (
        {},
        {"step": None},
        {"lora": "not-a-mapping"},
        {"gate": []},
        None,  # a truncated artifact is valid JSON `null` — must not TypeError
        123,
        "a string",
        [1, 2, 3],
    ):
        errs = L.validate_report(junk)
        assert isinstance(errs, list)
        assert errs, f"garbage must never validate clean: {junk!r}"


@pytest.mark.parametrize(
    "mutate,needle",
    [
        # --- the PRD §10.3 LoRA contract ---
        pytest.param(lambda r: r["lora"].__setitem__("r", 8), "lora.r", id="lora-rank-drift"),
        pytest.param(
            lambda r: r["lora"].__setitem__("lora_alpha", 16), "lora.lora_alpha", id="alpha-drift"
        ),
        pytest.param(
            lambda r: r["lora"].__setitem__("lora_dropout", 0.1),
            "lora.lora_dropout",
            id="dropout-drift",
        ),
        pytest.param(
            lambda r: r["lora"].__setitem__("target_modules", ["query", "value"]),
            "lora.target_modules",
            id="targets-drift",
        ),
        pytest.param(
            lambda r: r["lora"].__setitem__("dtype", "float32"), "lora.dtype", id="dtype-drift"
        ),
        pytest.param(
            lambda r: r["lora"].__setitem__("gradient_checkpointing", False),
            "gradient_checkpointing",
            id="no-grad-ckpt",
        ),
        # --- backbone identity (the parity-confirmed checkpoint, ADR-0002 A9) ---
        pytest.param(
            lambda r: r["backbone"].__setitem__("revision", "a" * 40),
            "backbone.revision",
            id="unpinned-revision",
        ),
        pytest.param(
            lambda r: r["backbone"].__setitem__("repo_id", "multimolecule/rnafm"),
            "backbone.repo_id",
            id="wrong-backbone",
        ),
        # --- the FA-2 honesty invariants (§10.3): FA-2 needs ALL its evidence ---
        pytest.param(
            lambda r: r["attention"]["sm86_evidence"].__setitem__("is_sm86", False),
            "is_sm86 is not True",
            id="fa2-without-sm86",
        ),
        pytest.param(
            lambda r: r["attention"]["sm86_evidence"].__setitem__("flash_attn_importable", False),
            "flash_attn_importable",
            id="fa2-without-wheel",
        ),
        pytest.param(
            lambda r: r["attention"].__setitem__("model_supports_flash_attn", False),
            "model_supports_flash_attn is not True",
            id="fa2-without-model-support",
        ),
        pytest.param(
            lambda r: r["attention"].__setitem__("forward_verified_on_sm86", True),
            "deferred to the P1-16",
            id="fa2-overclaims-forward",
        ),
        pytest.param(
            lambda r: r["attention"].__setitem__("reason", ""),
            "reason must be recorded",
            id="unrecorded-reason",
        ),
        pytest.param(
            lambda r: r["attention"].__setitem__("selected", "made_up_backend"),
            "not a known backend",
            id="unknown-backend",
        ),
        pytest.param(
            lambda r: r["attention"].__setitem__("fallback", "eager"),
            "attention.fallback must be",
            id="wrong-fallback",
        ),
        pytest.param(
            lambda r: r["attention"].pop("sm86_evidence"),
            "sm86_evidence must be a mapping",
            id="no-evidence-block",
        ),
        # --- fallback + qlora ---
        pytest.param(
            lambda r: r["full_ft_fallback"].__setitem__("zero_stage", 2),
            "zero_stage",
            id="wrong-zero-stage",
        ),
        pytest.param(
            lambda r: r["full_ft_fallback"].__setitem__("fsdp_sharding_strategy", "SHARD_GRAD_OP"),
            "fsdp_sharding_strategy",
            id="wrong-shard-strategy",
        ),
        pytest.param(
            lambda r: r["full_ft_fallback"].__setitem__("validated", False),
            "validated must be True",
            id="fallback-not-validated",
        ),
        pytest.param(
            lambda r: r["qlora"].__setitem__("enabled", True), "qlora.enabled", id="qlora-on"
        ),
        # --- LoRA-ness itself ---
        pytest.param(
            lambda r: r["wrap"].__setitem__("base_frozen", False),
            "not LoRA",
            id="base-not-frozen",
        ),
        pytest.param(
            lambda r: r["wrap"].__setitem__("trainable_params", 10_000),
            "0 < trainable < total",
            id="all-params-trainable",
        ),
        # --- schema ---
        pytest.param(
            lambda r: r.__setitem__("schema_version", "99"), "schema_version", id="schema-drift"
        ),
        pytest.param(lambda r: r.__setitem__("step", "P1-99"), "step !=", id="wrong-step"),
        pytest.param(lambda r: r.pop("attention"), "missing block", id="missing-block"),
        pytest.param(
            lambda r: r.__setitem__("lora", "not-a-mapping"),
            "block is not a mapping",
            id="block-not-mapping",
        ),
        # --- the APPLIED LoRA contract (the gate must read the MODEL, not the constants) ---
        pytest.param(
            lambda r: r["wrap"]["applied_lora"].__setitem__("r", 8),
            "wrap.applied_lora.r",
            id="applied-rank-drift",
        ),
        pytest.param(
            lambda r: r["wrap"]["applied_lora"].__setitem__("lora_alpha", 64),
            "wrap.applied_lora.lora_alpha",
            id="applied-alpha-drift",
        ),
        pytest.param(
            lambda r: r["wrap"]["applied_lora"].__setitem__("lora_dropout", 0.2),
            "wrap.applied_lora.lora_dropout",
            id="applied-dropout-drift",
        ),
        pytest.param(
            lambda r: r["wrap"]["applied_lora"].__setitem__("n_modules_adapted", 0),
            "must actually reach modules",
            id="all-linear-resolved-to-nothing",
        ),
        pytest.param(
            lambda r: r["wrap"].pop("applied_lora"),
            "wrap.applied_lora missing",
            id="no-applied-lora",
        ),
        # --- previously uncovered error-appends ---
        pytest.param(
            lambda r: r["qlora"].__setitem__("dependency_pinned_in_env", True),
            "dependency_pinned_in_env",
            id="qlora-dep-claimed-pinned",
        ),
        pytest.param(
            lambda r: r["gate"].pop("base_frozen"), "gate missing key", id="gate-missing-key"
        ),
        pytest.param(
            lambda r: r["attention"]["sm86_evidence"].pop("device_name"),
            "sm86_evidence missing key",
            id="evidence-missing-key",
        ),
        pytest.param(lambda r: r.pop("wrap"), "wrap.applied_lora missing", id="no-wrap-block"),
    ],
)
def test_validator_bites(mutate, needle):
    report = _good_report()
    mutate(report)
    errs = L.validate_report(report)
    assert any(needle in e for e in errs), f"expected an error containing {needle!r}, got {errs}"


def test_truthy_string_cannot_fake_a_bool():
    """§8.7: a tampered JSON must not slip a truthy non-bool past a gate check."""
    report = _good_report()
    report["qlora"]["enabled"] = "false"  # truthy string!
    assert any("qlora.enabled" in e for e in L.validate_report(report))


def test_unmeasured_report_cannot_claim_a_pass():
    """Fail-closed: only a real measurement certifies a gate (§10.3)."""
    report = _good_report()
    report["measured"] = False
    assert any("measured" in e for e in L.validate_report(report))


def test_overall_pass_must_be_the_and_of_its_parts():
    report = _good_report()
    report["gate"]["base_frozen"] = False
    report["gate"]["overall_pass"] = True  # a lie
    assert any("AND of the individual gate checks" in e for e in L.validate_report(report))


def test_lora_config_exact_gate_reads_the_model_not_the_constants():
    """The anti-tautology test.

    `gate.lora_config_exact` must be computed from `wrap_info["applied_lora"]` — what PEFT
    put on the model. If it were computed from `lora_config_kwargs()` (as it originally was)
    it would compare the module constants to themselves and could never be False: a wrap
    that applied r=8 would still certify the PRD §10.3 pin. Feed build_report a wrap whose
    applied config diverges and require the gate to notice.
    """
    diverged = {
        "gradient_checkpointing": True,
        "n_adapter_sites": 231,
        "base_frozen": True,
        "trainable_params": 100,
        "total_params": 10_000,
        "applied_lora": {  # the model got r=8, NOT the pinned 16
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "n_modules_adapted": 231,
        },
    }
    report = L.build_report(
        wrap_info=diverged,
        attn_backend=L.ATTN_SDPA,
        attn_reason="test",
        evidence={
            "source": "x",
            "is_sm86": True,
            "flash_attn_importable": True,
            "device_name": "NVIDIA RTX A4000",
        },
        supports_fa=True,
        fallback_validated=True,
    )
    assert (
        report["gate"]["lora_config_exact"] is False
    ), "the gate certified the §10.3 pin while the model carried r=8 — it is a tautology"
    assert report["gate"]["overall_pass"] is False
    assert any("wrap.applied_lora.r" in e for e in L.validate_report(report))


def test_build_report_without_applied_lora_fails_closed():
    """No applied config ⇒ no measurement ⇒ the gate clause must be False, never vacuously
    True."""
    report = L.build_report(
        wrap_info={"gradient_checkpointing": True, "n_adapter_sites": 1, "base_frozen": True},
        attn_backend=L.ATTN_SDPA,
        attn_reason="test",
        evidence={
            "source": "x",
            "is_sm86": True,
            "flash_attn_importable": True,
            "device_name": "NVIDIA RTX A4000",
        },
        supports_fa=True,
        fallback_validated=True,
    )
    assert report["gate"]["lora_config_exact"] is False
    assert report["gate"]["overall_pass"] is False


# ====================================================================================== #
# The committed artifact — the recorded decision, re-validated on every push.
# ====================================================================================== #
def test_committed_report_is_valid_and_records_the_decision():
    """The step's deliverable. UNCONDITIONAL — a skipif here would let the artifact vanish
    from a fresh checkout with CI still green (the nt_backbone `assert _PROVENANCE.exists()`
    precedent; the load-bearing gate must not rot)."""
    assert COMMITTED_REPORT.is_file(), (
        "the committed P1-15 artifact reports/p1/attention_backend.json is REQUIRED — it is "
        "the recorded §10.3 attention decision this step exists to produce"
    )
    report = json.loads(COMMITTED_REPORT.read_text())
    assert L.validate_report(report) == []
    assert report["gate"]["overall_pass"] is True
    # The PRD §10.3 gate: a backend was chosen AND its reasoning recorded.
    assert report["attention"]["selected"] in (L.ATTN_FLASH2, L.ATTN_SDPA)
    assert report["attention"]["reason"]
    # The measured wrap really is LoRA over the real giga backbone.
    assert report["wrap"]["base_frozen"] is True
    assert report["wrap"]["trainable_params"] < report["wrap"]["total_params"] * 0.05
    # The committed artifact must be the REAL backbone, not the tiny CPU fixture.
    assert report["wrap"]["tiny_fixture"] is False


def test_committed_report_evidence_matches_the_real_kernel_smoke():
    """The chain that the whole FA-2 verdict hangs on: the recorded sm86_evidence must BE
    what reports/p1/kernel_smoke.json actually says — re-derived here, not trusted.

    validate_report only checks the evidence block for *internal* consistency, so without
    this the artifact could drift from (or contradict) the real A5 measurement with CI green.
    This is what makes "sourced, not assumed" enforceable rather than merely asserted.
    """
    report = json.loads(COMMITTED_REPORT.read_text())
    recorded = dict(report["attention"]["sm86_evidence"])
    fresh = dict(L.read_sm86_flash_attn_evidence(KERNEL_SMOKE))
    # `source` is the path the reader was handed (the report records the repo-relative
    # default; this test passes an absolute path) — compare it separately by basename.
    assert Path(recorded.pop("source")).name == Path(fresh.pop("source")).name
    assert recorded == fresh, (
        "the committed report's sm86_evidence has drifted from what reports/p1/"
        "kernel_smoke.json actually measures — the FA-2 selection's evidence chain is broken"
    )


def test_conf_attention_mirrors_the_committed_decision():
    """conf/train/lora_stage2.yaml must not advertise a backend the evidence dropped."""
    report = json.loads(COMMITTED_REPORT.read_text())
    text = CONF.read_text()
    assert re.search(
        rf"(?m)^\s+selected:\s+{re.escape(report['attention']['selected'])}\b", text
    ), "conf/train/lora_stage2.yaml `attention.selected` drifted from the committed report"


# ====================================================================================== #
# peft/torch tier — needs the pinned ml-rna stack (skipped in bare CI).
# ====================================================================================== #
def _stack_available() -> bool:
    if os.environ.get("CUDA_HOME") is None:
        # `import multimolecule` probes CUDA_HOME at import (deepspeed transitive) —
        # ADR-0002 A5 operational note.
        return False
    try:
        import multimolecule  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


requires_stack = pytest.mark.skipif(
    not _stack_available(), reason="pinned ml-rna stack (torch+peft+multimolecule) unavailable"
)


@requires_stack
def test_model_supports_flash_attn_on_the_pinned_stack():
    """ADR-0002 A2 K1 established this on multimolecule 0.0.9; A8 moved us to 0.1.0 +
    transformers 5.13.0 — so it is re-measured, not inherited."""
    assert L.model_supports_flash_attn() is True


@requires_stack
def test_lora_wraps_a_tiny_rinalmo_and_freezes_the_base():
    """The §10.3 gate in miniature: the exact LoraConfig constructs and wraps RiNALMo's
    architecture, adapters attach, and the base comes out frozen."""
    import torch
    from multimolecule import RiNALMoConfig, RiNALMoModel

    torch.manual_seed(42)
    cfg = RiNALMoConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128
    )
    cfg._attn_implementation = "sdpa"
    base = RiNALMoModel(cfg, add_pooling_layer=False)

    peft_model, info = L.build_peft_model(base_model=base, dtype="float32")

    assert info["n_adapter_sites"] > 0
    assert info["base_frozen"] is True
    assert info["n_base_trainable_params"] == 0
    assert 0 < info["trainable_params"] < info["total_params"]
    # "all-linear" must reach the attention projections and the SwiGLU linears.
    for leaf in ("query", "key", "value"):
        assert leaf in info["targeted_module_leaf_names"]
    # The adapters carry the pinned rank. Collect first, then assert — a `break` inside the
    # loop would pass vacuously if no module ever matched.
    ranks = {
        r
        for _, mod in peft_model.named_modules()
        if isinstance(getattr(mod, "r", None), dict)
        for r in mod.r.values()
    }
    assert ranks == {L.LORA_R}, f"adapters did not all carry r={L.LORA_R}: {ranks}"


@requires_stack
def test_qlora_flag_raises_before_any_download():
    """The flag must fail loud BEFORE the 2.5 GB load path.

    Passing `base_model=object()` would skip the load regardless of where the guard sits, so
    it could not detect the guard being moved after the fetch. `base_model=None` is the real
    test: `qlora_config` must raise before `read_sm86_flash_attn_evidence` /
    `load_rinalmo_backbone` run.
    """
    with pytest.raises(NotImplementedError, match="bitsandbytes"):
        L.build_peft_model(qlora=True)


@requires_stack
def test_non_pinned_revision_is_rejected():
    """ADR-0002 D5/A9: only the parity-confirmed checkpoint; never a runtime re-pin."""
    with pytest.raises(ValueError, match="parity-confirmed"):
        L.load_rinalmo_backbone(revision="main")
    with pytest.raises(ValueError):
        L.load_rinalmo_backbone(revision="b" * 40)


@requires_stack
def test_full_ft_fallback_actually_constructs():
    """The gate's 'the fallback config validates' clause — a kwarg accelerate 1.12.0 would
    reject must fail HERE, not at a P3 submit."""
    fsdp_plugin, ds_config = L.build_full_ft_fallback()
    assert ds_config["zero_optimization"]["stage"] == 3
    assert fsdp_plugin is not None


# ====================================================================================== #
# giga tier — the real 2.5 GB parity-confirmed backbone. Opt-in only.
# ====================================================================================== #
@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_RINALMO") != "1",
    reason="set TBOX_REQUIRE_RINALMO=1 to wrap the real 2.5 GB rinalmo-giga",
)
def test_lora_wraps_the_real_giga_backbone():
    _, info = L.build_peft_model(attn_implementation=L.ATTN_SDPA, dtype="float32")
    assert info["base_frozen"] is True
    assert info["trainable_pct"] < 5.0
    assert info["n_adapter_sites"] > 200  # 33 encoder layers × ~7 linears
