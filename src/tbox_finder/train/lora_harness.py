"""P1-15 — Stage-2 RiNALMo LoRA/FSDP fine-tuning harness + attention-backend selection.

Builds the **PEFT harness** the Stage-2 T-box fine-tune (P3) will run on, against the
parity-confirmed RiNALMo-giga backbone (ADR-0002 D5/A9, PRD §10.2). **Harness only — there
is no science result here**: no T-box data is touched, nothing is trained, and the VRAM /
throughput questions are P1-16's GPU smoke. What lands is the wiring plus a *recorded,
evidence-backed* attention-backend decision.

The three things PRD §10.3 pins, and where each lives
-----------------------------------------------------
1. **LoRA default (bf16).** ``LoraConfig(r=16, α=32, dropout=0.05,
   target_modules="all-linear")`` + ``gradient_checkpointing``. These four scalars are the
   PRD-pinned contract, so they are **frozen as module constants** (:data:`LORA_R`,
   :data:`LORA_ALPHA`, :data:`LORA_DROPOUT`, :data:`LORA_TARGET_MODULES`) and *not*
   config-overridable — ``conf/train/lora_stage2.yaml`` echoes them for the record and a
   drift-guard test asserts config == code (the ``rinalmo_stage2.yaml`` precedent). A sweep
   over LoRA rank (PRD §11) is a P3 concern that must re-pin, not a runtime argument here.
2. **Full fine-tune fallback.** DeepSpeed ZeRO-3 (:func:`zero3_config`) / FSDP
   ``FULL_SHARD`` (:func:`fsdp_plugin_kwargs`) across 8 GPUs. Config builders only — the
   fallback is *built*, not selected (LoRA is the default; §10.3).
3. **FlashAttention-2 sm_86 confirm, else SDPA.** See below.

The attention-backend decision (the "no silent assumption" gate)
----------------------------------------------------------------
PRD §10.3: *"FlashAttention-2 sm_86 wheel must be confirmed; SDPA fallback otherwise."*
:func:`select_attention_backend` is a **pure** decision over four inputs, every one of
which is *measured* rather than assumed:

  - **sm_86 + flash-attn import — MEASURED on a real cluster A4000**, not on this laptop.
    Sourced from the committed ``reports/p1/kernel_smoke.json`` (ADR-0002 A5, SLURM job
    462: ``env.is_sm86=true``, ``imports.flash_attn.ok=true`` @ 2.8.3.post1) via
    :func:`read_sm86_flash_attn_evidence`, which **fails closed** if the artifact is
    missing or self-inconsistent. The laptop is sm_89 (RTX 4060) and *cannot* confirm
    sm_86 — reading the A4000 artifact is the only honest local path (§10.3).
  - **Model-side support** — ``RiNALMoModel._supports_flash_attn`` on the *pinned* stack.
    ADR-0002 A2 K1 established this for ``multimolecule`` 0.0.9; A8 moved us to 0.1.0 +
    transformers 5.13.0, so it is **re-introspected** here rather than inherited.
  - **dtype** — FA-2 kernels are half-precision only, so fp32 ⇒ SDPA regardless of the
    wheel. PRD §10.3 pins bf16, which is FA-2-eligible.

**Selected: FlashAttention-2, with SDPA as the recorded fallback.** Scope, stated plainly
(§10.3): what is confirmed is that the FA-2 wheel *imports on sm_86* and that RiNALMo
*advertises* flash-attn support (``_supports_flash_attn``) on the pinned stack. A
FA-2 **forward** through RiNALMo on an A4000 is **not** verified here — this step is LOCAL
and the laptop is the wrong architecture — and is deferred to **P1-16**'s GPU smoke. The
fallback is therefore live wiring, not decoration: :func:`select_attention_backend` returns
SDPA the moment any input goes false, and ADR-0002 A2 K1's caveat (*passing
``"flash_attention_2"`` without flash-attn raises ImportError*) is exactly what the guard
prevents. **No ADR pin changes** — A2 K1 already permits an SDPA/eager pin and names FA-2 a
selectable accelerator; this step records a selection A2 K1 anticipates.

QLoRA (flagged off, and it cannot be silently turned on)
--------------------------------------------------------
PRD §10.3 admits QLoRA **"only if needed"** (NF4 quant-error risk on a structure-sensitive
model). It is **not needed**: PRD §10.3's *estimated* LoRA footprint (frozen bf16 base
≈1.3 GB — 650 M × 2 bytes, an arithmetic estimate, **not** a measurement) sits well inside a
16 GB A4000; the measured footprint is P1-16's. ``bitsandbytes`` is **absent from**
``envs/ml-rna.yml`` + its lock + the installed env — verified, not assumed — so the QLoRA
path :func:`qlora_config` **fails loud** pointing at the ADR-0002 D8 env-amendment + §7
sign-off it would require. That is the honest shape of a flag whose dependency the pinned
env does not carry: a stub that silently no-ops, or an unpinned ``pip install``, would
violate §3.1/§10.3.

**Compute.** LOCAL. All heavy imports (torch / peft / accelerate / multimolecule) are lazy
(inside functions) so the module + its pure helpers + the validator import cleanly in a
bare CI env with no torch — the ``nt_backbone`` / ``rinalmo_throughput`` precedent. Note
``import multimolecule`` needs ``CUDA_HOME`` set (its deepspeed transitive import probes it
at import time — ADR-0002 A5 operational note).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Backbone identity is SINGLE-SOURCED from the parity module (ADR-0002 D5/A9): the
# checkpoint P1-13 proved faithful is the checkpoint P3 fine-tunes. Re-declaring the
# repo/revision here would let the two drift silently. rinalmo_parity is bare-importable
# (its multimolecule import is lazy), so this costs the bare-CI tier nothing.
from tbox_finder import provenance
from tbox_finder.eval.rinalmo_parity import D_MODEL, N_LAYERS, REPO_ID, REVISION

SCHEMA_VERSION = "1"
STEP = "P1-15"

# ====================================================================================== #
# PRD §10.3 pinned LoRA contract — FROZEN IN CODE, not config-overridable.
# ====================================================================================== #
#: LoRA rank. PRD §10.3 pins 16. (PRD §11 sweeps rank at P3 — a re-pin, not an override.)
LORA_R = 16
#: LoRA scaling α. PRD §10.3 pins 32.
LORA_ALPHA = 32
#: LoRA dropout. PRD §10.3 pins 0.05.
LORA_DROPOUT = 0.05
#: PEFT target-module selector. PRD §10.3 pins the literal string ``"all-linear"`` — PEFT
#: resolves it to every ``nn.Linear`` in the model (minus the output layer), so the target
#: set follows the architecture rather than a hand-listed set that could rot.
LORA_TARGET_MODULES = "all-linear"
#: LoRA bias handling. PEFT's default; PRD pins nothing (implementer choice, recorded).
LORA_BIAS = "none"
#: Training dtype. PRD §10.3 pins bf16 (A4000 = Ampere → bf16 native, ADR-0002 D1).
TRAIN_DTYPE = "bfloat16"

# ====================================================================================== #
# Attention backends (PRD §10.3; ADR-0002 D3 + A2 K1).
# ====================================================================================== #
ATTN_FLASH2 = "flash_attention_2"
ATTN_SDPA = "sdpa"
ATTN_EAGER = "eager"
#: The dtypes FA-2's kernels accept — they are half-precision only, so fp32 ⇒ SDPA.
FA2_DTYPES = ("bfloat16", "float16")
#: The measured sm_86 evidence artifact (ADR-0002 A5, SLURM job 462, cluster RTX A4000).
#: This step is LOCAL on an sm_89 laptop, so sm_86 is READ from here, never re-measured.
KERNEL_SMOKE_REPORT = "reports/p1/kernel_smoke.json"
#: CUDA compute capability of the A4000 target — the definition of "sm_86" (ADR-0002 D1).
#: Used to cross-check that an artifact claiming ``is_sm86`` really ran on that arch.
SM86_CAPABILITY = (8, 6)
#: The flash-attn version ``envs/ml-rna.yml`` pins by release-asset URL (ADR-0002 D3/A2).
FLASH_ATTN_PINNED = "2.8.3.post1"

# ====================================================================================== #
# Full-FT fallback (PRD §10.3): DeepSpeed ZeRO-3 / FSDP FULL_SHARD across 8 GPUs.
# ====================================================================================== #
#: The cluster's per-node GPU count (PRD §15 — 8× RTX A4000 on the `gpu` partition).
N_GPUS = 8
#: FSDP sharding strategy PRD §10.3 names verbatim.
FSDP_SHARDING_STRATEGY = "FULL_SHARD"
#: FSDP major version. accelerate 1.12.0 exposes both; ``sharding_strategy="FULL_SHARD"``
#: is the **FSDP1** knob PRD §10.3 names (FSDP2's equivalent is ``reshard_after_forward``).
#: Pinned to 1 so the config means literally what the PRD pins.
FSDP_VERSION = 1
#: RiNALMo's transformer block class — the FSDP auto-wrap unit (shard per encoder layer).
FSDP_TRANSFORMER_CLS = "RiNALMoLayer"
#: DeepSpeed ZeRO stage for the full-FT fallback (ZeRO-3 ≡ FSDP FULL_SHARD; both shard
#: params + grads + optimizer states → ~1.3 GB/GPU at 650 M × 8, PRD §10.3).
ZERO_STAGE = 3

#: ``bitsandbytes`` — the QLoRA/NF4 dependency. Verified ABSENT from envs/ml-rna.yml, its
#: conda-lock, and the installed env (2026-07-15). Enabling QLoRA is an env change.
QLORA_DEPENDENCY = "bitsandbytes"

DEFAULT_OUT = "reports/p1/attention_backend.json"

# The blocks a schema-valid P1-15 report must carry.
_REPORT_BLOCKS = ("backbone", "lora", "attention", "full_ft_fallback", "qlora", "env", "gate")


# ====================================================================================== #
# Pure helpers — stdlib only (import in any env; unit-tested to bite).
# ====================================================================================== #
def _sanitize(obj: Any) -> Any:
    """Map non-finite floats (NaN/Inf) → None so the report is always strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _bad_bool(value: Any, expected: bool) -> bool:
    """True (a violation) iff ``value`` is not a real ``bool`` equal to ``expected`` — so a
    tampered JSON cannot slip a truthy string/number past a gate check (§8.7/§10.3)."""
    return not isinstance(value, bool) or value != expected


def lora_config_kwargs() -> dict[str, Any]:
    """The exact PRD §10.3 ``LoraConfig`` kwargs, as a plain dict (torch-free).

    The single source of truth for the pinned contract: :func:`build_peft_model` passes
    these to the real ``LoraConfig``, the report records them, and the config drift-guard
    test compares ``conf/train/lora_stage2.yaml`` against them. One dict ⇒ they cannot
    disagree.
    """
    return {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "bias": LORA_BIAS,
    }


def select_attention_backend(
    *,
    flash_attn_importable: bool,
    sm86_confirmed: bool,
    model_supports_flash_attn: bool,
    dtype: str,
) -> tuple[str, str]:
    """Choose the attention implementation, PRD §10.3: FA-2 iff confirmed, else SDPA.

    Pure — every input is a *measured* fact supplied by the caller (the sm_86 + wheel facts
    come from the committed A5 artifact via :func:`read_sm86_flash_attn_evidence`; the
    model-support fact from introspecting the pinned classes). Returns
    ``(backend, reason)``; the reason is recorded verbatim in the report so the decision is
    auditable without re-running anything.

    SDPA — not eager — is the fallback: ADR-0002 D3 names SDPA the RiNALMo fallback path,
    and it is an *exact-softmax* kernel swap (the same swap P1-13 proved parity-faithful
    under, A9), so falling back costs speed and not numerics.
    """
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a str, got {type(dtype).__name__}")
    # Every flag must be a real bool. The module refuses truthy strings/numbers at the
    # report layer (`_bad_bool`); the same rigour belongs here, or a caller passing the
    # string "false" (truthy!) would silently select FA-2 on failed evidence (§8.7/§10.3).
    for name, flag in (
        ("flash_attn_importable", flash_attn_importable),
        ("sm86_confirmed", sm86_confirmed),
        ("model_supports_flash_attn", model_supports_flash_attn),
    ):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be a bool, got {type(flag).__name__}: {flag!r}")
    # Order matters: report the *first* blocking reason, most-fundamental first, so the
    # recorded reason names the thing an operator would have to fix.
    if not sm86_confirmed:
        return ATTN_SDPA, (
            "sm_86 not confirmed in the measured kernel-smoke artifact → SDPA fallback "
            "(PRD §10.3)"
        )
    if not flash_attn_importable:
        return ATTN_SDPA, (
            "flash-attn does not import on the sm_86 target → SDPA fallback; pinning "
            "flash_attention_2 without the wheel raises ImportError (ADR-0002 A2 K1)"
        )
    if not model_supports_flash_attn:
        return ATTN_SDPA, (
            "the pinned RiNALMo classes do not advertise flash-attn support → SDPA "
            "fallback (ADR-0002 A2 K1)"
        )
    if dtype not in FA2_DTYPES:
        return ATTN_SDPA, (
            f"dtype {dtype!r} is not half-precision; FA-2 kernels accept only "
            f"{list(FA2_DTYPES)} → SDPA fallback"
        )
    return ATTN_FLASH2, (
        "FA-2 selected: the flash-attn wheel imports on a MEASURED sm_86 A4000 "
        "(ADR-0002 A5), the pinned RiNALMo classes advertise flash-attn, and the pinned "
        f"dtype {dtype} is FA-2-eligible. The FA-2 forward through RiNALMo is NOT verified "
        "here (LOCAL step, sm_89 laptop) — deferred to the P1-16 GPU smoke."
    )


def read_sm86_flash_attn_evidence(path: str | Path = KERNEL_SMOKE_REPORT) -> dict[str, Any]:
    """Read the MEASURED sm_86 + flash-attn facts out of the committed A5 kernel smoke.

    The PRD §10.3 "FA-2 sm_86 wheel must be confirmed" evidence. This step runs LOCAL on an
    sm_89 laptop, so the confirmation is *sourced* from the real-A4000 artifact rather than
    re-measured (§10.3 — a laptop cannot honestly confirm sm_86).

    **Fails closed** (raises): a missing file, unparseable JSON, a missing block, a
    non-``bool`` flag, or a report whose own gate did not pass is an *absence of evidence*
    and must never read as a confirmation.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"sm_86 evidence artifact not found: {p} — the FA-2 confirmation is sourced "
            "from the measured P1-01 kernel smoke (ADR-0002 A5); it is never assumed."
        )
    try:
        report = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(report, Mapping):
        raise ValueError(f"{p} is not a JSON object")

    env = report.get("env")
    imports = report.get("imports")
    gate = report.get("gate")
    for name, blk in (("env", env), ("imports", imports), ("gate", gate)):
        if not isinstance(blk, Mapping):
            raise ValueError(f"{p} missing/invalid block: {name}")

    fa = imports.get("flash_attn")
    if not isinstance(fa, Mapping):
        raise ValueError(f"{p} missing imports.flash_attn block")

    is_sm86 = env.get("is_sm86")
    fa_ok = fa.get("ok")
    if not isinstance(is_sm86, bool):
        raise ValueError(f"{p} env.is_sm86 is not a bool (got {is_sm86!r})")
    if not isinstance(fa_ok, bool):
        raise ValueError(f"{p} imports.flash_attn.ok is not a bool (got {fa_ok!r})")
    # A smoke whose own gate failed cannot certify anything downstream.
    if not isinstance(gate.get("overall_pass"), bool) or not gate["overall_pass"]:
        raise ValueError(
            f"{p} gate.overall_pass is not True — a failed kernel smoke cannot confirm FA-2"
        )

    # Self-consistency: `is_sm86` must agree with the recorded compute capability. Without
    # this, an artifact measured on some *other* card (e.g. this sm_89 laptop) but flagged
    # is_sm86=true would silently confirm FA-2 for an architecture it never touched — the
    # one input class that could produce a false confirmation (§10.3).
    capability = env.get("device_capability")
    if is_sm86 and list(capability or []) != list(SM86_CAPABILITY):
        raise ValueError(
            f"{p} claims env.is_sm86=True but env.device_capability={capability!r} is not "
            f"{list(SM86_CAPABILITY)} — a self-inconsistent artifact cannot confirm FA-2 "
            "on sm_86"
        )

    # The evidence must be about the wheel we actually pin. An artifact measured against a
    # different flash-attn build confirms *that* build, not ours (ADR-0002 D3/A2 pin
    # 2.8.3.post1 by release-asset URL) — so a version mismatch is an absence of evidence
    # for the pinned wheel, not a confirmation of it.
    fa_version = fa.get("version")
    if fa_ok and fa_version != FLASH_ATTN_PINNED:
        raise ValueError(
            f"{p} recorded flash_attn version {fa_version!r}, but envs/ml-rna.yml pins "
            f"{FLASH_ATTN_PINNED!r} — evidence from a different wheel cannot confirm the "
            "pinned FA-2 build"
        )

    return {
        "source": str(path),
        "source_step": report.get("step"),
        "is_sm86": is_sm86,
        "device_name": env.get("device_name"),
        "device_capability": capability,
        "flash_attn_importable": fa_ok,
        "flash_attn_version": fa.get("version"),
        # DERIVED from the capability agreeing with sm_86 — never hardcoded True.
        "measured_on_target_arch": bool(is_sm86),
    }


def zero3_config(
    *,
    gradient_accumulation_steps: int = 1,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """The DeepSpeed **ZeRO-3** full-FT fallback config (PRD §10.3), as a plain dict.

    Fallback only — LoRA is the §10.3 default. ZeRO-3 shards params + grads + optimizer
    states across the 8 A4000s (≈1.3 GB/GPU at 650 M). ``"auto"`` batch fields let the
    accelerate/DeepSpeed integration fill them from the runtime plugin rather than freezing
    a number here (a real batch size is a P3 decision, and inventing one now would be a
    fabricated compute figure — §10.3).

    No offload: PRD §10.3 sizes the fallback to fit *on* the 8 GPUs, and CPU/NVMe offload
    would silently trade the VRAM question for a bandwidth one. Turning it on is a P3
    decision with its own measurement.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}"
        )
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError(f"grad_clip must be finite and > 0, got {grad_clip}")
    return {
        "bf16": {"enabled": True},  # PRD §10.3 pins bf16 (Ampere-native)
        "zero_optimization": {
            "stage": ZERO_STAGE,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": grad_clip,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 2_000_000,
    }


def fsdp_plugin_kwargs() -> dict[str, Any]:
    """The FSDP ``FULL_SHARD`` full-FT fallback kwargs (PRD §10.3), as a plain dict.

    The FSDP-flavoured twin of :func:`zero3_config` — accelerate documents
    ``FULL_SHARD`` ≡ ZeRO-3, so the two fallbacks are interchangeable and we build both
    (PRD §10.3 names them together). Consumed as
    ``FullyShardedDataParallelPlugin(**fsdp_plugin_kwargs())``.

    ``fsdp_version=1`` is deliberate: ``sharding_strategy="FULL_SHARD"`` is the FSDP1 knob
    PRD §10.3 pins by name (accelerate 1.12.0 also exposes FSDP2, whose equivalent is
    ``reshard_after_forward=True``). Wrapping per ``RiNALMoLayer`` shards at the encoder
    block, the granularity that makes the ~1.3 GB/GPU arithmetic hold.
    """
    return {
        "fsdp_version": FSDP_VERSION,
        "sharding_strategy": FSDP_SHARDING_STRATEGY,
        "auto_wrap_policy": "transformer_based_wrap",
        "transformer_cls_names_to_wrap": [FSDP_TRANSFORMER_CLS],
        "activation_checkpointing": True,  # §10.3 gradient checkpointing, FSDP-side
        "cpu_offload": False,  # see zero3_config — fits on-GPU by design
    }


def qlora_config(*, enabled: bool = False) -> dict[str, Any]:
    """The QLoRA/NF4 path — **behind a flag, and disabled** (PRD §10.3 "only if needed").

    Returns the recorded (disabled) state when ``enabled=False``. When ``enabled=True`` it
    **raises**: ``bitsandbytes`` is not in ``envs/ml-rna.yml`` or its lock, so there is no
    honest way to turn QLoRA on from here — it needs an env amendment + lockfile
    regeneration + ADR-0002 D8 sign-off (CLAUDE.md §3.1/§7). Failing loud is the point: a
    stub that silently no-opped, or an ad-hoc ``pip install``, would let a *quantized* run
    masquerade as the pinned one (§10.3).
    """
    if enabled:
        raise NotImplementedError(
            "QLoRA is flagged OFF and cannot be enabled from code: its dependency "
            f"{QLORA_DEPENDENCY!r} is absent from envs/ml-rna.yml + envs/ml-rna.conda-lock.yml. "
            "PRD §10.3 admits QLoRA 'only if needed' (NF4 quant-error risk on a "
            "structure-sensitive model) and it is not needed — bf16 LoRA is the default. "
            "Enabling it requires an env amendment + re-lock + ADR-0002 D8 sign-off "
            "(CLAUDE.md §3.1/§7), never a runtime flag."
        )
    return {
        "enabled": False,
        "reason": (
            "PRD §10.3 admits QLoRA only if needed; bf16 LoRA is the default and the NF4 "
            "quant-error risk on a structure-sensitive RNA model is unquantified"
        ),
        "dependency": QLORA_DEPENDENCY,
        "dependency_pinned_in_env": False,
        "enabling_requires": "envs/ml-rna.yml amendment + re-lock + ADR-0002 D8 sign-off",
    }


# ====================================================================================== #
# Torch tier — lazy imports (bare CI never reaches these).
# ====================================================================================== #
def _require_pinned_revision(revision: str) -> None:
    """Reject any non-pinned revision (ADR-0002 D5/A9 — the parity-confirmed checkpoint).

    P1-13 proved *this* revision faithful (A9); a different one inherits no such proof, so
    re-pinning is a code change + re-sign-off, never a runtime argument.
    """
    if revision != REVISION:
        raise ValueError(
            f"revision {revision!r} != the pinned parity-confirmed {REVISION!r} "
            "(ADR-0002 D5/A9). Re-pinning needs a code change + ADR sign-off."
        )


def model_supports_flash_attn() -> bool:
    """Introspect the **pinned** RiNALMo classes for flash-attn support (measured).

    ADR-0002 A2 K1 established this on ``multimolecule`` 0.0.9; A8 moved the env to 0.1.0 +
    transformers 5.13.0, so it is re-checked against what is actually installed rather than
    inherited from the amendment. transformers 5 renamed the attribute
    ``_supports_flash_attn_2`` → ``_supports_flash_attn``; both are probed so the check does
    not silently read False on a rename.
    """
    from multimolecule import RiNALMoModel  # lazy

    return bool(
        getattr(RiNALMoModel, "_supports_flash_attn", False)
        or getattr(RiNALMoModel, "_supports_flash_attn_2", False)
    )


def flash_attn_importable() -> bool:
    """Whether ``flash_attn`` imports **in this process** (informational only).

    **NOT the gate input**, and deliberately so: this runs on the sm_89 laptop, and PRD
    §10.3 asks about **sm_86**. The gate reads :func:`read_sm86_flash_attn_evidence` (a real
    A4000). Recorded under the report's unambiguously local-only
    ``attention.local_probe_non_authoritative`` key so a reader can see both facts and never
    mistake this one for the sm_86 confirmation.
    """
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False


def load_rinalmo_backbone(
    *,
    revision: str = REVISION,
    dtype: str = TRAIN_DTYPE,
    attn_implementation: str | None = None,
    device: str | None = None,
):
    """Load the parity-confirmed RiNALMo-giga encoder, ready for LoRA wrapping.

    ``add_pooling_layer=False``: the checkpoint carries no pooler, so the default would
    freshly-initialise one (a MISSING-weights warning) and hand LoRA a randomly-initialised
    module to adapt. Stage-2 consumes per-position encoder states (its heads are P3), so the
    pooler is unwanted weight — dropping it makes the load report clean, which is what lets a
    real MISSING warning mean something later.

    ``attn_implementation=None`` leaves the backend to the caller (:func:`build_peft_model`
    resolves it via :func:`select_attention_backend`) — ADR-0002 A2 K1: pin ``sdpa``/
    ``eager``, or ``flash_attention_2`` only when the wheel is present.
    """
    import torch  # lazy
    from multimolecule import RiNALMoModel  # lazy

    _require_pinned_revision(revision)
    torch_dtype = getattr(torch, dtype)
    kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": torch_dtype,
        "add_pooling_layer": False,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = RiNALMoModel.from_pretrained(REPO_ID, **kwargs)
    if device is not None:
        model = model.to(device)
    return model


def build_peft_model(
    base_model=None,
    *,
    revision: str = REVISION,
    dtype: str = TRAIN_DTYPE,
    attn_implementation: str | None = None,
    gradient_checkpointing: bool = True,
    qlora: bool = False,
    device: str | None = None,
):
    """**The P1-15 entry (imp.md ``Rule/Entry``).** Wrap RiNALMo in the §10.3 LoRA config.

    Returns ``(peft_model, info)`` where ``info`` records the *measured* wrap facts
    (trainable/total params, adapter-site count, resolved attention backend) that
    :func:`build_report` publishes.

    ``base_model=None`` loads the pinned backbone (:func:`load_rinalmo_backbone`); passing
    one in lets a test wrap a tiny same-architecture model without the 2.5 GB download.

    The LoRA scalars are **not** parameters — they are the PRD §10.3 pinned contract
    (:func:`lora_config_kwargs`). ``gradient_checkpointing`` defaults on per §10.3.
    """
    from peft import LoraConfig, get_peft_model  # lazy

    qlora_config(enabled=qlora)  # raises when qlora=True — the flag cannot silently pass

    if attn_implementation is None:
        evidence = read_sm86_flash_attn_evidence()
        attn_implementation, _reason = select_attention_backend(
            flash_attn_importable=evidence["flash_attn_importable"],
            sm86_confirmed=evidence["is_sm86"],
            model_supports_flash_attn=model_supports_flash_attn(),
            dtype=dtype,
        )

    if base_model is None:
        base_model = load_rinalmo_backbone(
            revision=revision,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device=device,
        )

    # Record the ELIGIBLE nn.Linear set BEFORE wrapping. "all-linear" is only meaningfully
    # "all" if the adapted set equals the eligible set — n_modules_adapted > 0 would pass
    # even if PEFT reached exactly one Linear. This is the pre/post comparison that makes
    # the §10.3 selector's coverage a measurement rather than a hope.
    import torch  # lazy

    eligible = {n for n, m in base_model.named_modules() if isinstance(m, torch.nn.Linear)}

    peft_model = get_peft_model(base_model, LoraConfig(**lora_config_kwargs()))

    if gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()

    suffix = ".lora_A.default"
    targets = [n[: -len(suffix)] for n, _ in peft_model.named_modules() if n.endswith(suffix)]
    # PEFT rewrites module paths under `base_model.model.` when it wraps; compare on the
    # trailing path so the two sets are commensurable.
    adapted_paths = {t.split("base_model.model.", 1)[-1] for t in targets}
    uncovered = sorted(eligible - adapted_paths)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    # The base MUST be frozen — that is what makes this LoRA rather than a full FT (§10.3).
    # Measured, not asserted: any non-adapter trainable param is a real defect.
    base_trainable = [
        n for n, p in peft_model.named_parameters() if p.requires_grad and "lora_" not in n
    ]

    # READ BACK what PEFT actually applied, rather than re-stating what we asked for. The
    # §10.3 gate must compare the *model* to the pins; a gate that compares
    # lora_config_kwargs() to the constants that produced it can never fail (it would assert
    # the pin while the model carried something else). PEFT resolves the "all-linear"
    # selector into a concrete module set, so the resolution is recorded too.
    applied = peft_model.peft_config["default"]
    resolved = applied.target_modules
    # CAREFUL: this is PEFT's post-injection target_modules SELECTOR, not a list of adapted
    # modules. PEFT rewrites "all-linear" to the matched module names, then — once ≥
    # MIN_TARGET_MODULES_FOR_OPTIMIZATION (20) match — *minimises* that set to the shortest
    # unambiguous suffixes. So rinalmo-giga's 231 adapted Linears collapse to 6 suffixes
    # ({query, key, value, dense, linear, linear_gate}), while a small fixture keeps full
    # paths. Recording this count as "modules adapted" would understate by ~40×; the
    # authoritative count is `n_adapter_sites`, counted off the real module tree below.
    resolved = sorted(resolved) if not isinstance(resolved, str) else None

    # `attn_implementation` is only APPLIED on the load path; when a caller supplies its own
    # base model we must record what that model actually carries, not what we were passed —
    # otherwise the report names a backend the model never used.
    applied_attn = getattr(getattr(base_model, "config", None), "_attn_implementation", None)

    info = {
        "attn_implementation": applied_attn or attn_implementation,
        "attn_implementation_requested": attn_implementation,
        "dtype": dtype,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "n_adapter_sites": len(targets),
        "targeted_module_leaf_names": sorted({t.split(".")[-1] for t in targets}),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": round(100.0 * trainable / total, 4) if total else None,
        "base_frozen": not base_trainable,
        "n_base_trainable_params": len(base_trainable),
        # The measured LoRA contract — what the wrapped model carries (§10.3).
        "applied_lora": {
            "r": applied.r,
            "lora_alpha": applied.lora_alpha,
            "lora_dropout": applied.lora_dropout,
            "target_modules_selector": LORA_TARGET_MODULES,
            # PEFT's post-injection selector, possibly suffix-minimised (see above) — NOT a
            # count of adapted modules. `n_adapter_sites` is that count.
            "target_modules_after_injection": resolved,
            "target_modules_suffix_minimised": (
                None if resolved is None else len(resolved) < len(targets)
            ),
            "n_modules_adapted": len(targets),
            # "all-linear" coverage, MEASURED as a pre/post set comparison.
            "n_eligible_linear_modules": len(eligible),
            "all_linear_fully_covered": not uncovered,
            "uncovered_linear_modules": uncovered,
        },
    }
    return peft_model, info


def build_full_ft_fallback(*, gradient_accumulation_steps: int = 1):
    """Instantiate the accelerate FSDP ``FULL_SHARD`` plugin + the ZeRO-3 dict (§10.3).

    Returns ``(fsdp_plugin, ds_config)``. Constructing the plugin is what *validates* the
    fallback: a kwarg accelerate 1.12.0 rejects raises here rather than at P3 submit time.
    """
    from accelerate import FullyShardedDataParallelPlugin  # lazy

    fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_plugin_kwargs())
    ds_config = zero3_config(gradient_accumulation_steps=gradient_accumulation_steps)
    return fsdp_plugin, ds_config


# ====================================================================================== #
# Report — build + fail-closed validator.
# ====================================================================================== #
def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a (possibly empty) list of schema/consistency errors for a P1-15 report.

    Runs in a bare CI env (stdlib only) over the committed artifact, so the recorded
    decision is re-checked on every push regardless of whether torch is present. Robust to
    a malformed/tampered report — never raises, always returns an error list. Empty ⇒ valid.
    """
    # `report` may be ANY parsed JSON (a truncated artifact can be `null`, a bare int, a
    # list). `x in None` raises TypeError, which would turn a gate failure into an opaque
    # crash — so the top-level type is checked before anything else.
    if not isinstance(report, Mapping):
        return [f"report is not a mapping (got {type(report).__name__})"]

    errs: list[str] = []
    for blk in _REPORT_BLOCKS:
        if blk not in report:
            errs.append(f"missing block: {blk}")
        elif not isinstance(report[blk], Mapping):
            errs.append(f"block is not a mapping: {blk}")
    if errs:  # a non-mapping block would crash the .get() checks below — stop here
        return errs

    if report.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version != {SCHEMA_VERSION!r}")
    if report.get("step") != STEP:
        errs.append(f"step != {STEP!r}")

    bb = report["backbone"]
    if bb.get("repo_id") != REPO_ID:
        errs.append("backbone.repo_id != pinned REPO_ID")
    if bb.get("revision") != REVISION:
        errs.append("backbone.revision != pinned REVISION (the parity-confirmed checkpoint)")

    # --- the PRD §10.3 LoRA contract: the report may not record anything but the pins ---
    lora = report["lora"]
    for key, want in (
        ("r", LORA_R),
        ("lora_alpha", LORA_ALPHA),
        ("lora_dropout", LORA_DROPOUT),
        ("target_modules", LORA_TARGET_MODULES),
    ):
        if lora.get(key) != want:
            errs.append(f"lora.{key} != PRD §10.3 pinned {want!r}")
    if lora.get("dtype") != TRAIN_DTYPE:
        errs.append(f"lora.dtype != {TRAIN_DTYPE!r} (PRD §10.3 pins bf16)")
    if _bad_bool(lora.get("gradient_checkpointing"), True):
        errs.append("lora.gradient_checkpointing must be True (PRD §10.3)")

    # The load-bearing half: the block above only proves the report ECHOES the pins. This
    # proves the MODEL carries them — the applied config PEFT read back off the wrapped
    # model. Without it a report could assert §10.3 while the model ran something else.
    wrap_blk = report.get("wrap")
    applied = wrap_blk.get("applied_lora") if isinstance(wrap_blk, Mapping) else None
    if report.get("measured") is True and not isinstance(applied, Mapping):
        errs.append(
            "wrap.applied_lora missing — a measured report must record the APPLIED "
            "LoRA config read back off the model, not just the pinned echo"
        )
    elif isinstance(applied, Mapping):
        for key, want in (
            ("r", LORA_R),
            ("lora_alpha", LORA_ALPHA),
            ("lora_dropout", LORA_DROPOUT),
        ):
            if applied.get(key) != want:
                errs.append(
                    f"wrap.applied_lora.{key} != PRD §10.3 pinned {want!r} — the model does "
                    "not carry the pinned LoRA contract"
                )
        n_adapted = applied.get("n_modules_adapted")
        if not isinstance(n_adapted, int) or n_adapted < 1:
            errs.append(
                "wrap.applied_lora.n_modules_adapted must be a positive int — 'all-linear' "
                "must actually reach modules"
            )
        # Consistency: the adapted count is the same tree walk `n_adapter_sites` reports.
        elif isinstance(wrap_blk, Mapping) and wrap_blk.get("n_adapter_sites") != n_adapted:
            errs.append(
                "wrap.applied_lora.n_modules_adapted != wrap.n_adapter_sites — the two "
                "counts of the same wrap disagree"
            )
        # "all-linear" means ALL of them: a run that adapted a proper subset of the eligible
        # Linears is not the §10.3 config, however many sites it reports.
        if _bad_bool(applied.get("all_linear_fully_covered"), True):
            errs.append(
                "wrap.applied_lora.all_linear_fully_covered must be True — 'all-linear' did "
                f"not cover every eligible nn.Linear (uncovered: "
                f"{applied.get('uncovered_linear_modules')!r})"
            )

    # --- the attention decision must be internally consistent with its own evidence ---
    attn = report["attention"]
    backend = attn.get("selected")
    if backend not in (ATTN_FLASH2, ATTN_SDPA, ATTN_EAGER):
        errs.append(f"attention.selected is not a known backend: {backend!r}")
    if not attn.get("reason"):
        errs.append("attention.reason must be recorded (no silent assumption, PRD §10.3)")
    if attn.get("fallback") != ATTN_SDPA:
        errs.append(f"attention.fallback must be {ATTN_SDPA!r} (PRD §10.3)")
    ev = attn.get("sm86_evidence")
    if not isinstance(ev, Mapping):
        errs.append("attention.sm86_evidence must be a mapping (the measured A5 artifact)")
        return errs  # the consistency checks below need it
    for k in ("source", "is_sm86", "flash_attn_importable", "device_name"):
        if k not in ev:
            errs.append(f"attention.sm86_evidence missing key: {k}")
    # Every FA-2 decision input must be a REAL bool before it is trusted or re-derived —
    # a truthy "false" must not be able to steer the decision (§8.7).
    for label, value in (
        ("sm86_evidence.is_sm86", ev.get("is_sm86")),
        ("sm86_evidence.flash_attn_importable", ev.get("flash_attn_importable")),
        ("model_supports_flash_attn", attn.get("model_supports_flash_attn")),
        ("forward_verified_on_sm86", attn.get("forward_verified_on_sm86")),
    ):
        if not isinstance(value, bool):
            errs.append(f"attention.{label} must be a bool, got {type(value).__name__} ({value!r})")
    if errs:
        return errs

    # RE-DERIVE the decision from the recorded evidence and require the report to match.
    # Checking only "FA-2 implies its evidence is True" would leave the converse open: a
    # report could record `sdpa` while every input says FA-2, or vice versa. The pure
    # decision function is the single source of truth, so the artifact must be exactly what
    # it returns (the P1-05 "verdict follows from classify_separability" precedent).
    expected_backend, _expected_reason = select_attention_backend(
        flash_attn_importable=ev["flash_attn_importable"],
        sm86_confirmed=ev["is_sm86"],
        model_supports_flash_attn=attn["model_supports_flash_attn"],
        dtype=lora.get("dtype") if isinstance(lora.get("dtype"), str) else "",
    )
    if backend != expected_backend:
        errs.append(
            f"attention.selected={backend!r} does not follow from the recorded evidence — "
            f"select_attention_backend returns {expected_backend!r} for it"
        )

    # Fail-closed (§10.3): FA-2 may ONLY be selected when every measured input backs it.
    # (Kept explicit rather than folded into the re-derivation above: these name the exact
    # input at fault, which is what an operator reading a failure needs.)
    if backend == ATTN_FLASH2:
        if _bad_bool(ev.get("is_sm86"), True):
            errs.append(
                "attention.selected=flash_attention_2 but sm86_evidence.is_sm86 is not True"
            )
        if _bad_bool(ev.get("flash_attn_importable"), True):
            errs.append(
                "attention.selected=flash_attention_2 but sm86_evidence.flash_attn_importable "
                "is not True"
            )
        if _bad_bool(attn.get("model_supports_flash_attn"), True):
            errs.append(
                "attention.selected=flash_attention_2 but model_supports_flash_attn is not True"
            )
        # (No dtype check here: `lora.dtype != TRAIN_DTYPE` above already errors and returns
        # before this block, and `TRAIN_DTYPE in FA2_DTYPES` is a module invariant locked by
        # its own test — a check here would be unreachable, i.e. false coverage.)
        # The honesty invariant: FA-2 is selected on an *import* confirmation, and the
        # forward is P1-16's. A report that claimed the forward was verified here would be
        # claiming a measurement this LOCAL sm_89 step cannot make (§10.3).
        if _bad_bool(attn.get("forward_verified_on_sm86"), False):
            errs.append(
                "attention.forward_verified_on_sm86 must be False — the FA-2 forward through "
                "RiNALMo is deferred to the P1-16 GPU smoke, not measured at P1-15"
            )

    # --- full-FT fallback + QLoRA flag ---
    ft = report["full_ft_fallback"]
    if ft.get("zero_stage") != ZERO_STAGE:
        errs.append(f"full_ft_fallback.zero_stage != {ZERO_STAGE}")
    if ft.get("fsdp_sharding_strategy") != FSDP_SHARDING_STRATEGY:
        errs.append(f"full_ft_fallback.fsdp_sharding_strategy != {FSDP_SHARDING_STRATEGY!r}")
    if _bad_bool(ft.get("validated"), True):
        errs.append("full_ft_fallback.validated must be True (the config must parse — gate)")

    q = report["qlora"]
    if _bad_bool(q.get("enabled"), False):
        errs.append("qlora.enabled must be False (PRD §10.3 — only if needed; not needed)")
    if _bad_bool(q.get("dependency_pinned_in_env"), False):
        errs.append("qlora.dependency_pinned_in_env must be False (bitsandbytes is not pinned)")

    gate = report["gate"]
    for k in (
        "lora_config_exact",
        "wraps_rinalmo",
        "base_frozen",
        "fallback_config_validates",
        "attention_backend_recorded",
        "overall_pass",
    ):
        if k not in gate:
            errs.append(f"gate missing key: {k}")
    if errs:
        return errs

    # `measured` must be a REAL bool. A truthy string like "false" would otherwise walk
    # straight into the measured path and let overall_pass=True stand (§8.7/§10.3).
    if not isinstance(report.get("measured"), bool):
        errs.append(
            f"report.measured must be a bool, got {type(report.get('measured')).__name__} "
            f"({report.get('measured')!r})"
        )
        return errs

    # Fail-closed: an unmeasured report must not claim any gate pass.
    if not report["measured"]:
        if gate.get("overall_pass") is True:
            errs.append("gate.overall_pass is True but report.measured is not set")
        return errs

    expected_pass = all(
        gate.get(k) is True
        for k in (
            "lora_config_exact",
            "wraps_rinalmo",
            "base_frozen",
            "fallback_config_validates",
            "attention_backend_recorded",
        )
    )
    if _bad_bool(gate.get("overall_pass"), expected_pass):
        errs.append("gate.overall_pass is not the AND of the individual gate checks")

    wrap = report.get("wrap")
    if isinstance(wrap, Mapping):
        if _bad_bool(wrap.get("base_frozen"), True):
            errs.append("wrap.base_frozen must be True — a non-frozen base is not LoRA (§10.3)")
        n_tr = wrap.get("trainable_params")
        n_tot = wrap.get("total_params")
        if not isinstance(n_tr, int) or not isinstance(n_tot, int) or not 0 < n_tr < n_tot:
            errs.append("wrap trainable/total params must be ints with 0 < trainable < total")
    else:
        errs.append("measured report must carry a `wrap` block")

    return errs


def build_report(
    *,
    wrap_info: Mapping[str, Any],
    attn_backend: str,
    attn_reason: str,
    evidence: Mapping[str, Any],
    supports_fa: bool,
    fallback_validated: bool,
    measured: bool = True,
) -> dict[str, Any]:
    """Assemble the P1-15 record from *measured* inputs (no defaults that invent facts).

    ``lora`` is the **pinned spec** (an echo of PRD §10.3, flagged as such); the gate is
    computed against ``wrap_info["applied_lora"]`` — what PEFT actually put on the model —
    so ``lora_config_exact`` compares the *model* to the pins rather than the constants to
    themselves.
    """
    lora = dict(lora_config_kwargs())
    # Flagged as an echo, not a measurement: `wrap.dtype` carries the dtype actually used
    # (they differ on the tiny CPU fixture, which wraps at float32).
    lora["is_pinned_spec_echo"] = True
    lora["dtype"] = TRAIN_DTYPE
    lora["gradient_checkpointing"] = bool(wrap_info.get("gradient_checkpointing"))

    # The gate's LoRA clause reads the APPLIED config. Absent applied_lora there is no
    # measurement to certify → the clause is False (fail closed), never vacuously True.
    applied = wrap_info.get("applied_lora")
    if isinstance(applied, Mapping):
        lora_exact = (
            applied.get("r") == LORA_R
            and applied.get("lora_alpha") == LORA_ALPHA
            and applied.get("lora_dropout") == LORA_DROPOUT
            # "all-linear" must have actually reached modules — and reached ALL the eligible
            # ones. Checked off the module tree (PEFT's own target_modules is
            # suffix-minimised and would understate the count).
            and bool(applied.get("n_modules_adapted"))
            and applied.get("all_linear_fully_covered") is True
        )
    else:
        lora_exact = False

    gate = {
        "lora_config_exact": lora_exact,
        "wraps_rinalmo": int(wrap_info.get("n_adapter_sites", 0)) > 0,
        "base_frozen": bool(wrap_info.get("base_frozen")),
        "fallback_config_validates": bool(fallback_validated),
        "attention_backend_recorded": bool(attn_backend) and bool(attn_reason),
    }
    gate["overall_pass"] = all(gate.values())

    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "measured": bool(measured),
        "backbone": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "hidden_dim": D_MODEL,
            "n_layer": N_LAYERS,
            "parity_confirmed": True,  # ADR-0002 A9 (P1-13)
            "pooling_layer": False,
        },
        "lora": lora,
        "wrap": dict(wrap_info),
        "attention": {
            "selected": attn_backend,
            "fallback": ATTN_SDPA,
            "reason": attn_reason,
            "model_supports_flash_attn": bool(supports_fa),
            "flash_attn_pinned_version": FLASH_ATTN_PINNED,
            "sm86_evidence": dict(evidence),
            # Informational ONLY — this machine is not the sm_86 target, so this must never
            # be read as the §10.3 confirmation (that is `sm86_evidence`). Named to make
            # the distinction unmissable.
            "local_probe_non_authoritative": {
                "flash_attn_imports_here": flash_attn_importable(),
                "note": (
                    "the local machine is NOT the sm_86 target; this is not the PRD §10.3 "
                    "confirmation — see sm86_evidence"
                ),
            },
            # The honesty invariant this step must not overstate (§10.3): what is confirmed
            # is the sm_86 *import* + model support, NOT a RiNALMo FA-2 forward.
            "forward_verified_on_sm86": False,
            "forward_verification_deferred_to": "P1-16 (GPU VRAM/mechanics smoke)",
        },
        "full_ft_fallback": {
            "role": "fallback only — LoRA is the PRD §10.3 default",
            "zero_stage": ZERO_STAGE,
            "fsdp_version": FSDP_VERSION,
            "fsdp_sharding_strategy": FSDP_SHARDING_STRATEGY,
            "n_gpus": N_GPUS,
            "validated": bool(fallback_validated),
            "deepspeed_config": zero3_config(),
            "fsdp_plugin_kwargs": fsdp_plugin_kwargs(),
        },
        "qlora": qlora_config(enabled=False),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "gate": gate,
        # Inline provenance (the P1-14 precedent): the report is git-tracked and
        # self-contained, so it carries its own git SHA rather than a sidecar file.
        "git_sha": provenance.git_sha(),
    }
    return _sanitize(report)


def _versions() -> dict[str, str]:
    """Record the installed versions of the load-bearing libs (importlib.metadata, not
    ``__version__`` — the P1-14 CodeRabbit correction)."""
    import importlib.metadata as md

    out: dict[str, str] = {}
    for pkg in ("torch", "transformers", "peft", "accelerate", "deepspeed", "multimolecule"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = "<not installed>"
    return out


def run_harness_dryrun(*, out_path: str | Path = DEFAULT_OUT, tiny: bool = False) -> dict[str, Any]:
    """The LOCAL dry-run: wrap the real backbone, validate the fallback, record the decision.

    ``tiny=True`` wraps a small same-architecture RiNALMo instead of downloading the 2.5 GB
    giga checkpoint — for CI/smoke use. The committed artifact is produced with
    ``tiny=False`` (the real parity-confirmed backbone), and the report records which.

    A ``tiny`` run **may not write** :data:`DEFAULT_OUT`: that path is the committed
    evidence artifact, and a fixture-derived report silently overwriting the real-backbone
    one is exactly how a toy number ends up quoted as a result (§10.3). Tiny runs must name
    their own output path.
    """
    if tiny and str(out_path) == str(DEFAULT_OUT):
        raise ValueError(
            f"refusing to write the canonical artifact {DEFAULT_OUT} from a tiny fixture "
            "run — it would overwrite the real-backbone record with a toy one. Pass an "
            "explicit --out for tiny runs."
        )
    evidence = read_sm86_flash_attn_evidence()
    supports_fa = model_supports_flash_attn()
    backend, reason = select_attention_backend(
        flash_attn_importable=evidence["flash_attn_importable"],
        sm86_confirmed=evidence["is_sm86"],
        model_supports_flash_attn=supports_fa,
        dtype=TRAIN_DTYPE,
    )

    base = None
    if tiny:
        import torch  # lazy
        from multimolecule import RiNALMoConfig, RiNALMoModel  # lazy

        torch.manual_seed(42)
        cfg = RiNALMoConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128
        )
        cfg._attn_implementation = ATTN_SDPA  # tiny path is CPU — FA-2 needs a GPU
        base = RiNALMoModel(cfg, add_pooling_layer=False)

    # The wrap runs on CPU (LOCAL step). FA-2 is *selected + recorded*, but a CPU load must
    # not instantiate FA-2 kernels — so the wrap uses SDPA and the selection stands on the
    # measured A5 evidence. P1-16 exercises the selected backend on a real A4000.
    peft_model, info = build_peft_model(
        base_model=base,
        attn_implementation=ATTN_SDPA if base is None else None,
        dtype="float32" if tiny else TRAIN_DTYPE,
    )
    info["wrap_attn_implementation_note"] = (
        "wrapped under SDPA on CPU (this step is LOCAL); the SELECTED backend is "
        f"{backend!r}, which is NOT exercised here — it is TO BE exercised on an A4000 at "
        "P1-16 (no forward through the selected backend has been run)"
    )
    info["tiny_fixture"] = bool(tiny)

    # The full-FT fallback must construct — that IS the gate's "fallback config validates"
    # clause. Let the real exception propagate (chained): a failure here means accelerate
    # rejected a kwarg, and the operator needs THAT message. Recording it and continuing
    # would fail the validator two frames later with the true cause discarded.
    try:
        build_full_ft_fallback()
    except Exception as exc:
        raise RuntimeError(
            f"P1-15 full-FT fallback config failed to construct ({type(exc).__name__}: {exc}) "
            "— PRD §10.3 requires the ZeRO-3/FSDP fallback to validate"
        ) from exc
    fallback_validated = True

    report = build_report(
        wrap_info=info,
        attn_backend=backend,
        attn_reason=reason,
        evidence=evidence,
        supports_fa=supports_fa,
        fallback_validated=fallback_validated,
    )
    report["env"]["versions"] = _versions()

    errs = validate_report(report)
    if errs:  # fail closed — never publish a report that fails its own validator
        raise RuntimeError(f"P1-15 report failed its own validator: {errs}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    del peft_model
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m tbox_finder.train.lora_harness --dry-run``."""
    ap = argparse.ArgumentParser(description="P1-15 Stage-2 LoRA/FSDP harness dry-run")
    ap.add_argument("--dry-run", action="store_true", help="build + wrap + record (LOCAL)")
    ap.add_argument("--tiny", action="store_true", help="wrap a tiny fixture, not the 2.5 GB giga")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"report path (default {DEFAULT_OUT})")
    args = ap.parse_args(argv)

    if not args.dry_run:
        ap.error("nothing to do: pass --dry-run")

    report = run_harness_dryrun(out_path=args.out, tiny=args.tiny)
    print(json.dumps(report["attention"], indent=2, sort_keys=True))
    print(json.dumps(report["gate"], indent=2, sort_keys=True))
    return 0 if report["gate"]["overall_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
