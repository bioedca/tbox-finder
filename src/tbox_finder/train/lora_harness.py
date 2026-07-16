"""P1-15/P1-16 — Stage-2 RiNALMo LoRA/FSDP harness, backend selection, and the GPU smoke.

Builds the **PEFT harness** the Stage-2 T-box fine-tune (P3) will run on, against the
parity-confirmed RiNALMo-giga backbone (ADR-0002 D5/A9, PRD §10.2). **Harness only — there
is no science result here**: no T-box data is touched, and nothing that trains here trains on
anything real. What lands is the wiring, a *recorded, evidence-backed* attention-backend
decision (P1-15), and a *measured* VRAM/mechanics smoke on one A4000 (P1-16).

Two steps live in this module because imp.md pins both entries here
(``build_peft_model`` for P1-15, ``smoke_step`` for P1-16), and P1-16 is the GPU half of the
harness P1-15 built. They keep **separate report schemas** — ``build_report`` /
``validate_report`` publish the P1-15 attention decision, ``build_smoke_report`` /
``validate_smoke_report`` the P1-16 measurement — because they certify different claims and a
shared validator would let one step's evidence vouch for the other's.

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

The P1-16 GPU smoke (PRD §10.2 condition (b), VRAM half)
--------------------------------------------------------
PRD §10.2 makes the RNA-FM swap fire if *"the giga model cannot meet the A4000 16 GB-VRAM +
genome-scale latency budget after PEFT/bf16"*, and phase-assigns the two halves separately:
**VRAM at P1 (here); genome-scale latency frozen at the P5 sizing gate.** So this step
answers exactly one question — *does a real LoRA fine-tune step fit on one A4000?* — and
:data:`VRAM_BUDGET_GIB` (16 GB) is the **only** number the PRD/ADRs pin for it. A miss is a
CLAUDE.md §7 stop-and-ask (escalate to the FSDP ``FULL_SHARD`` fallback or the §10.2 RNA-FM
consideration), never a loosened threshold.

Batch size, sequence length, and step count are **not pinned anywhere** (verified across PRD
§10.2/§10.3/§11/§15 and every ADR-0002 decision/amendment) — they are implementer choices,
recorded with ``pinned: false`` so they cannot be misread as pre-registered.

**The batch is seeded random RNA, and the loss is a placeholder** (:func:`placeholder_loss`)
— stated, not glossed (§10.3). That is honest *for this question*: VRAM and step latency are
set by the encoder's activation/gradient/optimizer footprint at a given shape, which is
sequence-length- and batch-bound rather than content-bound (the P1-14 precedent), and the P3
heads that do not exist yet are a ``Linear(1280 → k)`` whose footprint is negligible beside a
650 M-param encoder. It is **not** a T-box result, a convergence claim, or a learning-rate
finding, and the report's validator refuses any artifact that says otherwise.

**The FA-2 forward — the carried P1-15 handoff.** P1-15 selected ``flash_attention_2`` on an
*import*-level sm_86 confirmation (ADR-0002 A5) plus model advertisement; **no FA-2 forward
through RiNALMo had ever run** (A5's only forward-parity tests are the Stage-1
selective-scan / causal-conv1d kernels). imp.md's P1-16 gate is written around VRAM and
steps/sec and does not mention attention, but this is the last step before an FA-2-selecting
config reaches P3 — and ADR-0002 A2 K1 warns the config path is exactly where it breaks.
:func:`run_vram_smoke` therefore loads under the **selected** backend and
``gate.attention_forward_verified`` **fails closed** if that forward does not run. A failure
is a §7 stop-and-ask rather than an auto-flip: the recorded selection lives in a committed
artifact (``reports/p1/attention_backend.json``) and is mirrored in
``conf/train/lora_stage2.yaml``, so changing it is a decision, not a side effect.

**Compute.** P1-15 is LOCAL; **P1-16 is SLURM** (``gpu``, ``gres=gpu:a4000:1``) — the laptop
is sm_89 and cannot answer an sm_86 question, and heavy RiNALMo work belongs on the cluster
A4000 regardless. All heavy imports (torch / peft / accelerate / multimolecule) are lazy
(inside functions) so the module + its pure helpers + both validators import cleanly in a
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
# P1-16 — the GPU VRAM / mechanics smoke (PRD §10.2 condition (b), the VRAM half).
# ====================================================================================== #
SMOKE_STEP = "P1-16"
SMOKE_SCHEMA_VERSION = "1"
DEFAULT_SMOKE_OUT = "reports/p1/lora_vram_smoke.json"

#: **The only pinned number in this step.** PRD §10.2 condition (b) / §10.3 / §15 + ADR-0002
#: D6(b): Stage-2 must fit the A4000's 16 GB after PEFT/bf16, or the RNA-FM swap is
#: considered. Read as GiB — the conventional reading of a "16 GB" card (ADR-0002 Context
#: line: "RTX A4000 = Ampere, sm_86, 16 GB VRAM"). NOT weakenable: a miss is a CLAUDE.md §7
#: stop-and-ask (escalate to FSDP FULL_SHARD or the §10.2 RNA-FM swap), never a loosened
#: threshold (the A7 / A9 precedent). `fits_on_device` cross-checks against the card's real
#: usable `total_memory`, which torch reports as ~15.6 GiB — so the gate is the *stricter* of
#: the PRD's number and the physical card.
VRAM_BUDGET_GIB = 16.0

#: Seed for the synthetic batch + torch RNG (PRD §11 "explicit seeds everywhere", §8.3).
SMOKE_SEED = 42

# ----- IMPLEMENTER CHOICES, recorded as such (§10.3) ---------------------------------- #
# NOTHING in the PRD, ADR-0002, or imp.md pins a batch size, sequence length, or step count
# for this smoke — verified across PRD §10.2/§10.3/§11/§15 and every ADR-0002 D/A. imp.md
# says only "a tiny placeholder RNA batch" and "one LoRA fine-tune step runs end-to-end".
# The numbers below are therefore CHOSEN, and the report flags them `pinned: false` so no
# reader mistakes them for pre-registered values (the ADR-0002 A5 `atol` precedent: "no atol
# is ADR-pinned; it is an implementer choice and the raw diff is always recorded").
#
#: Sequence length: the UPPER end of PRD §10.2's "T-box loci (~150-350 nt)" — the
#: conservative representative of what Stage-2 actually sees (well under RiNALMo's 1022-token
#: ceiling, which is an architectural max, not a workload).
SMOKE_SEQ_LEN_NT = 350
#: The GATE batch size. 1 is deliberate and is the *feasibility* question §10.2 (b) asks: if a
#: single-sequence LoRA step will not fit one A4000, Stage-2 cannot train on this cluster at
#: all. Larger batches are a throughput preference P3 can always trade for gradient
#: accumulation, so gating on them would gate on a choice rather than on feasibility.
SMOKE_BATCH_SIZE = 1
#: Headroom sweep (stop at the first CUDA OOM) — reports what P3 can actually afford. This
#: is REPORTED, never gated: an OOM at batch 16 is a budgeting fact, not a §10.2 failure.
SMOKE_BATCH_SWEEP: tuple[int, ...] = (1, 2, 4, 8, 16)
#: Optimizer steps timed per config, after `SMOKE_WARMUP_STEPS` untimed ones (the first step
#: pays lazy CUDA-context + autotune costs and would poison steps/sec).
SMOKE_STEPS = 5
SMOKE_WARMUP_STEPS = 2
#: LR for the throwaway optimizer. Irrelevant to VRAM/mechanics (AdamW's state size is
#: LR-independent) and this trains on random RNA against a placeholder loss — it is NOT a P3
#: hyperparameter and must never be read as one (PRD §11 sweeps LR at P3).
SMOKE_LR = 1e-4

#: The blocks a schema-valid P1-16 report must carry. ``wrap`` and ``hardware`` are required
#: because every gate clause is re-derived from them: a block the validator does not require
#: is a block a report can simply omit, taking the corresponding clause on faith.
_SMOKE_BLOCKS = (
    "backbone",
    "lora",
    "attention",
    "config",
    "measured_smoke",
    "env",
    "gate",
    "wrap",
    "hardware",
)


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


# ====================================================================================== #
# P1-16 — the GPU VRAM / mechanics smoke.
# ====================================================================================== #
def _gib(n_bytes: float) -> float:
    """Bytes → GiB, rounded to 3 dp (the P1-14 reporting convention)."""
    return round(float(n_bytes) / (1024**3), 3)


# -------------------------------------------------------------------------------------- #
# Gate-clause derivations. Each is defined ONCE and used by BOTH :func:`build_smoke_report`
# (to compute the clause) and :func:`validate_smoke_report` (to re-derive it from the
# recorded evidence and require the artifact to match).
#
# Why re-derivation, and not just the AND: `overall_pass = all(clauses)` catches a clause
# flipped FALSE (it then contradicts a True overall_pass), but it structurally CANNOT catch a
# clause fabricated TRUE — an all-True gate is self-consistent no matter what the evidence
# says. So a hand-edited report claiming `measured_on_sm86: true` beside an sm_89 capability,
# or `bf16_and_gradient_checkpointing: true` beside a fp32 wrap, would validate clean. Sharing
# the derivation is what makes each clause a *measurement* rather than an assertion (§10.3),
# and is the same lesson P1-15 learned when its `lora_config_exact` turned out tautological.
# -------------------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    """``value`` as a float iff it is a real number — ``bool`` is NOT a number here.

    ``isinstance(True, int)`` is True in Python, so without the bool rejection a report
    carrying ``peak_vram_gib: true`` would sail into the numeric comparisons (§8.7).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _pos_int(value: Any) -> int | None:
    """``value`` as an int iff it is a real, positive, non-``bool`` integer.

    Same ``isinstance(True, int)`` trap as :func:`_num`: without this, a report carrying
    ``n_timed_steps: true`` or ``n_lora_params: true`` would certify that a step ran / that
    adapters were adapted, on a value that counts nothing (§8.7).
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


#: The GPU PRD §10.2 (b) / §15 name. `measured_on_sm86` is not merely "some sm_86 card": the
#: budget is the A4000's, and another Ampere part (an A10 has 24 GB) would answer a question
#: the PRD never asked while looking identical in the capability tuple.
A4000_NAME_FRAGMENT = "A4000"


def _is_sm86(hardware: Mapping[str, Any]) -> bool:
    """True iff the hardware block records exactly ONE real sm_86 A4000.

    Derived from the measured compute capability, never from the ``is_sm86`` flag alone —
    the same self-consistency discipline :func:`read_sm86_flash_attn_evidence` applies to the
    A5 artifact. An artifact flagged ``is_sm86: true`` beside an sm_89 capability describes an
    architecture it never touched.

    Total, never raising: ``cuda_capability`` may be any parsed JSON in a tampered artifact,
    and ``list(86)`` would raise ``TypeError`` — which would turn :func:`validate_smoke_report`
    from "returns an error list" into "crashes", breaking the contract its callers rely on.
    """
    if not isinstance(hardware, Mapping):
        return False
    cap = hardware.get("cuda_capability")
    if not isinstance(cap, (list, tuple)):  # a scalar/None/str must NOT reach list()
        return False
    name = hardware.get("gpu_name")
    return (
        list(cap) == list(SM86_CAPABILITY)
        and hardware.get("is_sm86") is True
        # PRD §10.2 (b) asks about ONE A4000; a multi-GPU or other-card measurement is a
        # different question wearing the same capability tuple.
        and _pos_int(hardware.get("n_gpus_measured")) == 1
        and isinstance(name, str)
        and A4000_NAME_FRAGMENT in name
    )


def _bf16_and_ckpt(wrap_info: Mapping[str, Any]) -> bool:
    """True iff the wrap that ran really carried PRD §10.3's bf16 + gradient checkpointing."""
    if not isinstance(wrap_info, Mapping):
        return False
    return wrap_info.get("dtype") == TRAIN_DTYPE and wrap_info.get("gradient_checkpointing") is True


def _grads_flow(grad_flow: Mapping[str, Any]) -> bool:
    """True iff EVERY LoRA param measurably received a finite gradient.

    The mechanics claim, derived from counts rather than from the summary booleans: a report
    could carry ``all_lora_params_received_grad: true`` beside ``0 of 462``, and it is the
    counts that were actually measured.
    """
    if not isinstance(grad_flow, Mapping):
        return False
    total = _pos_int(grad_flow.get("n_lora_params"))
    with_grad = _pos_int(grad_flow.get("n_lora_params_with_grad"))
    return (
        total is not None
        and with_grad == total
        and grad_flow.get("all_lora_grads_finite") is True
        and grad_flow.get("all_lora_params_received_grad") is True
    )


def smoke_config() -> dict[str, Any]:
    """The chosen (NOT pinned) smoke shape, recorded so a reader can see it is a choice."""
    return {
        "seq_len_nt": SMOKE_SEQ_LEN_NT,
        "gate_batch_size": SMOKE_BATCH_SIZE,
        "batch_sweep": list(SMOKE_BATCH_SWEEP),
        "timed_steps_per_config": SMOKE_STEPS,
        "warmup_steps_per_config": SMOKE_WARMUP_STEPS,
        "lr": SMOKE_LR,
        "optimizer": "AdamW",
        "seed": SMOKE_SEED,
        "dtype": TRAIN_DTYPE,
        "gradient_checkpointing": True,
        # The honesty flags the validator enforces.
        "pinned": False,
        "pinned_note": (
            "PRD §10.2/§10.3/§11/§15 and ADR-0002 pin NO batch size, sequence length, step "
            "count, or LR for this smoke — only the 16 GB VRAM budget. These are implementer "
            "choices, recorded (the ADR-0002 A5 `atol` precedent). seq_len_nt is the upper "
            "end of §10.2's ~150-350 nt T-box locus range; gate_batch_size=1 is the "
            "feasibility question (larger batches trade against gradient accumulation at P3)."
        ),
        "data_is_synthetic": True,
        "loss_is_placeholder": True,
        "is_science": False,
    }


def placeholder_loss(hidden_states):
    """A **placeholder** scalar loss over the encoder's per-position states (§10.3).

    Mean-squared activation. It exists to make a real backward pass happen through every LoRA
    site, and it is **not** an objective: there is no T-box data at P1 and Stage-2's real
    heads are P3's. Computed in fp32 (``.float()``) so the scalar reduction over a bf16
    tensor does not accumulate in bf16.

    Why this does not compromise the measurement: the question is peak VRAM and step latency,
    which are set by the 650 M-param encoder's activation/gradient/optimizer footprint at a
    given (batch, length) — a P3 head is a ``Linear(1280 → k)`` and is negligible beside it.
    Using a *fabricated* label set and calling the resulting number a training loss, on the
    other hand, would be a fabricated science result — hence a loss that is transparently not
    one.
    """
    return hidden_states.float().pow(2).mean()


def smoke_step(peft_model, optimizer, input_ids, attention_mask=None) -> float:
    """**The P1-16 entry (imp.md ``Rule/Entry``).** Run ONE LoRA fine-tune step end-to-end.

    forward → :func:`placeholder_loss` → ``backward`` → ``optimizer.step`` →
    ``zero_grad``. Returns the loss as a float. This is the unit imp.md's gate is written
    about ("one LoRA fine-tune step runs end-to-end on a single A4000"); the surrounding
    measurement (VRAM, steps/sec, the sweep) is :func:`run_vram_smoke`'s.

    ``set_to_none=True`` on ``zero_grad`` is the torch default and frees the grad buffers
    between steps — which is why the *peak* is read across a whole timed run rather than
    after a single step.
    """
    out = peft_model(input_ids=input_ids, attention_mask=attention_mask)
    loss = placeholder_loss(out.last_hidden_state)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().float().item())


def measure_lora_grad_flow(peft_model) -> dict[str, Any]:
    """Measure whether gradients actually reached the LoRA sites (fail-closed evidence).

    The load-bearing mechanics check, and the reason it is a *measurement* rather than an
    assertion in a comment: with a **frozen** base + gradient checkpointing, the checkpointed
    segments receive inputs with ``requires_grad=False``, so autograd has nothing to
    recompute and the LoRA params silently receive **no gradient** — a step that "runs" and
    trains nothing. transformers guards this by calling ``enable_input_require_grads()``
    inside ``gradient_checkpointing_enable()`` when ``main_input_name == "input_ids"`` (which
    RiNALMo's is), but that is a fact about the installed stack, not a guarantee — so it is
    checked against the real model after a real backward.

    Call **after** ``loss.backward()`` and **before** ``optimizer.zero_grad()``.
    """
    import torch  # lazy

    total = 0
    with_grad = 0
    finite = True
    for name, p in peft_model.named_parameters():
        if "lora_" not in name or not p.requires_grad:
            continue
        total += 1
        if p.grad is not None:
            with_grad += 1
            if not bool(torch.isfinite(p.grad).all().item()):
                finite = False
    return {
        "n_lora_params": total,
        "n_lora_params_with_grad": with_grad,
        "all_lora_grads_finite": finite,
        "all_lora_params_received_grad": total > 0 and with_grad == total,
    }


def _adapter_dtypes(peft_model) -> list[str]:
    """The distinct dtypes PEFT actually gave the adapters (recorded, not assumed).

    PEFT's ``autocast_adapter_dtype=True`` default upcasts adapters to fp32 over a bf16 base;
    the report records what the model carries rather than what the default is documented to
    do.
    """
    return sorted(
        {str(p.dtype) for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad}
    )


def _time_steps(
    peft_model,
    optimizer,
    input_ids,
    attention_mask,
    *,
    warmup: int,
    steps: int,
    device: str,
) -> dict[str, Any]:
    """Run warmup + timed :func:`smoke_step` calls; return latency + the grad-flow evidence.

    CUDA-event timed with an explicit ``synchronize`` (the P1-14 / context7 torch idiom) —
    wall-clock around an async CUDA queue would time the launch, not the work.
    """
    import time

    import torch  # lazy

    is_cuda = device.startswith("cuda")
    losses: list[float] = []
    for _ in range(max(0, warmup)):
        losses.append(smoke_step(peft_model, optimizer, input_ids, attention_mask))

    # Grad-flow evidence from a step whose grads are still live (zero_grad has not run yet).
    out = peft_model(input_ids=input_ids, attention_mask=attention_mask)
    loss = placeholder_loss(out.last_hidden_state)
    loss.backward()
    grad_flow = measure_lora_grad_flow(peft_model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if is_cuda:
        torch.cuda.synchronize(device)
    step_ms: list[float] = []
    for _ in range(steps):
        if is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            losses.append(smoke_step(peft_model, optimizer, input_ids, attention_mask))
            end.record()
            torch.cuda.synchronize(device)
            step_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            losses.append(smoke_step(peft_model, optimizer, input_ids, attention_mask))
            step_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(step_ms) / len(step_ms) if step_ms else None
    return {
        "n_timed_steps": len(step_ms),
        "step_ms_mean": round(mean_ms, 3) if mean_ms else None,
        "step_ms_min": round(min(step_ms), 3) if step_ms else None,
        "step_ms_max": round(max(step_ms), 3) if step_ms else None,
        "steps_per_sec": round(1000.0 / mean_ms, 4) if mean_ms else None,
        "final_loss": losses[-1] if losses else None,
        "grad_flow": grad_flow,
    }


def _is_oom(exc: BaseException) -> bool:
    """True iff ``exc`` looks like a CUDA OOM (a budgeting fact) vs a real code bug."""
    return "out of memory" in str(exc).lower()


def run_vram_smoke(
    *,
    out_path: str | Path = DEFAULT_SMOKE_OUT,
    revision: str = REVISION,
    device: str | None = None,
    seed: int = SMOKE_SEED,
    require_cuda: bool = False,
    adapter_dir: str | Path | None = None,
    log: Any = lambda _m: None,
) -> dict[str, Any]:
    """Load → wrap → train one step → measure peak VRAM + steps/sec → record (P1-16).

    Loads the parity-confirmed backbone under the **P1-15-selected** attention backend (so
    the FA-2 forward is finally exercised on sm_86), wraps it in the §10.3 LoRA config with
    gradient checkpointing, and runs :func:`smoke_step` at the gate shape plus a headroom
    batch sweep. Fails closed: the report must pass :func:`validate_smoke_report` or this
    raises rather than publishing.
    """
    import torch  # lazy
    from multimolecule import RnaTokenizer  # lazy

    from tbox_finder.eval.rinalmo_throughput import synthetic_batch  # bare-importable

    _require_pinned_revision(revision)

    # --- the attention decision, re-derived through the SAME P1-15 path (not copied) ----- #
    evidence = read_sm86_flash_attn_evidence()
    supports_fa = model_supports_flash_attn()
    backend, reason = select_attention_backend(
        flash_attn_importable=evidence["flash_attn_importable"],
        sm86_confirmed=evidence["is_sm86"],
        model_supports_flash_attn=supports_fa,
        dtype=TRAIN_DTYPE,
    )
    log(f"P1-15 selected backend = {backend!r}; exercising its forward on this device")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if require_cuda and not device.startswith("cuda"):
        raise RuntimeError(
            "--require-cuda set but no CUDA device is available. P1-16 answers an sm_86 "
            "A4000 question; a CPU run cannot (PRD §10.2 condition (b))."
        )
    is_cuda = device.startswith("cuda")

    torch.manual_seed(seed)
    tokenizer = RnaTokenizer.from_pretrained(REPO_ID, revision=revision)

    # --- exercise the SELECTED backend's forward; record, never silently paper over ------ #
    forward_verified = False
    forward_error: str | None = None
    attn_used = backend
    try:
        peft_model, wrap_info = build_peft_model(
            revision=revision,
            dtype=TRAIN_DTYPE,
            attn_implementation=backend,
            gradient_checkpointing=True,
            device=device,
        )
        probe = tokenizer(["ACGU" * 8], return_tensors="pt")["input_ids"].to(device)
        peft_model(input_ids=probe)  # the forward P1-15 could not run
        forward_verified = True
        log(f"{backend!r} forward through RiNALMo: OK on {device}")
    except Exception as exc:  # noqa: BLE001 — any failure of the selected backend
        forward_error = f"{type(exc).__name__}: {exc}"
        log(f"{backend!r} forward FAILED ({forward_error}) — retrying under {ATTN_SDPA!r}")
        if backend == ATTN_SDPA:
            raise  # the fallback itself failing is a real bug, not a backend question
        # Re-load under the recorded fallback so the VRAM number is still produced; the
        # BACKEND question then becomes a §7 stop-and-ask (the gate below fails closed).
        attn_used = ATTN_SDPA
        peft_model, wrap_info = build_peft_model(
            revision=revision,
            dtype=TRAIN_DTYPE,
            attn_implementation=ATTN_SDPA,
            gradient_checkpointing=True,
            device=device,
        )

    peft_model.train()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=SMOKE_LR)

    def _batch(bs: int):
        seqs = synthetic_batch(SMOKE_SEQ_LEN_NT, bs, seed=seed)
        enc = tokenizer(seqs, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device) if "attention_mask" in enc else None
        return ids, mask

    # --- the GATE config: peak VRAM is read across the whole timed run ------------------- #
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    ids, mask = _batch(SMOKE_BATCH_SIZE)
    gate_run = _time_steps(
        peft_model,
        optimizer,
        ids,
        mask,
        warmup=SMOKE_WARMUP_STEPS,
        steps=SMOKE_STEPS,
        device=device,
    )
    gate_peak_gib = _gib(torch.cuda.max_memory_allocated(device)) if is_cuda else None
    gate_run["peak_vram_gib"] = gate_peak_gib
    gate_run["batch_size"] = SMOKE_BATCH_SIZE
    log(f"gate batch={SMOKE_BATCH_SIZE} len={SMOKE_SEQ_LEN_NT}: peak {gate_peak_gib} GiB")

    # --- headroom sweep (REPORTED, never gated): stop at the first OOM ------------------- #
    sweep: dict[str, Any] = {}
    oom_at_batch: int | None = None
    largest_fitting_batch: int | None = None
    for bs in SMOKE_BATCH_SWEEP:
        try:
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            ids_b, mask_b = _batch(bs)
            run = _time_steps(
                peft_model,
                optimizer,
                ids_b,
                mask_b,
                warmup=1,
                steps=2,
                device=device,
            )
            run["peak_vram_gib"] = (
                _gib(torch.cuda.max_memory_allocated(device)) if is_cuda else None
            )
            run.pop("grad_flow", None)  # the gate run carries the authoritative evidence
            sweep[str(bs)] = run
            largest_fitting_batch = bs
            log(f"sweep batch={bs}: peak {run['peak_vram_gib']} GiB")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if not _is_oom(exc):
                raise
            oom_at_batch = bs
            log(f"sweep batch={bs}: CUDA OOM — stopping sweep (a budgeting fact, not a gate)")
            if is_cuda:
                torch.cuda.empty_cache()
            break

    # --- the throwaway adapter checkpoint (imp.md Outputs; cluster-side, not committed) -- #
    adapter_saved = None
    if adapter_dir is not None:
        d = Path(adapter_dir)
        d.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(d))
        adapter_saved = str(d)
        log(f"throwaway LoRA adapter saved to {d}")

    measured_smoke = {
        "gate_run": gate_run,
        "batch_sweep": sweep,
        "largest_fitting_batch": largest_fitting_batch,
        "oom_at_batch": oom_at_batch,
        "adapter_checkpoint_dir": adapter_saved,
        "adapter_param_dtypes": _adapter_dtypes(peft_model),
        "device_total_memory_gib": (
            _gib(torch.cuda.get_device_properties(_device_index(device)).total_memory)
            if is_cuda
            else None
        ),
    }

    report = build_smoke_report(
        wrap_info=wrap_info,
        measured_smoke=measured_smoke,
        attn_selected=backend,
        attn_used=attn_used,
        attn_reason=reason,
        forward_verified=forward_verified,
        forward_error=forward_error,
        evidence=evidence,
        supports_fa=supports_fa,
        hardware=_hardware_block(device),
    )
    report["env"]["versions"] = _versions()

    errs = validate_smoke_report(report)
    if errs:  # fail closed — never publish a report that fails its own validator
        raise RuntimeError(f"P1-16 report failed its own validator: {errs}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def _device_index(device: str) -> int:
    """The integer index of a ``cuda``/``cuda:N`` selector (the P1-14 correction)."""
    return int(device.split(":", 1)[1]) if ":" in device else 0


def _hardware_block(device: str) -> dict[str, Any]:
    """The measured GPU identity — recorded so a reader can check it really was an A4000."""
    import torch  # lazy

    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(_device_index(device))
        return {
            "device": device,
            "gpu_name": props.name,
            "gpu_total_memory_gib": _gib(props.total_memory),
            "cuda_capability": [props.major, props.minor],
            "is_sm86": [props.major, props.minor] == list(SM86_CAPABILITY),
            "n_gpus_measured": 1,
        }
    return {
        "device": device,
        "gpu_name": "cpu",
        "gpu_total_memory_gib": None,
        "cuda_capability": None,
        "is_sm86": False,
        "n_gpus_measured": 0,
    }


def build_smoke_report(
    *,
    wrap_info: Mapping[str, Any],
    measured_smoke: Mapping[str, Any],
    attn_selected: str,
    attn_used: str,
    attn_reason: str,
    forward_verified: bool,
    forward_error: str | None,
    evidence: Mapping[str, Any],
    supports_fa: bool,
    hardware: Mapping[str, Any],
    measured: bool = True,
) -> dict[str, Any]:
    """Assemble the P1-16 record from *measured* inputs (no default invents a fact).

    ``hardware`` is passed in (:func:`_hardware_block`'s output) rather than probed here, so
    this stays **stdlib-pure** and the bare-CI tier can prove the gate arithmetic — including
    that an over-budget peak cannot produce a pass — without torch installed.
    """
    gate_run = measured_smoke.get("gate_run") or {}
    peak = gate_run.get("peak_vram_gib")
    grad_flow = gate_run.get("grad_flow") or {}
    total_mem = measured_smoke.get("device_total_memory_gib")

    # The §10.2 condition-(b) clause. `is not None` matters: a CPU run has no VRAM number,
    # and `None <= 16.0` must never read as a pass.
    within_budget = _num(peak) is not None and float(peak) <= VRAM_BUDGET_GIB
    fits_on_device = (
        _num(peak) is not None and _num(total_mem) is not None and float(peak) <= float(total_mem)
    )

    # The §10.3 contract clause, read off the APPLIED config PEFT put on the model — the
    # P1-15 discipline (`build_report`'s `lora_config_exact`). A clause computed from the
    # module constants instead would compare the pins to themselves and could never fail, so
    # a run that wrapped `r=8` would still certify §10.3 while reporting its VRAM.
    applied = wrap_info.get("applied_lora")
    if isinstance(applied, Mapping):
        lora_exact = (
            applied.get("r") == LORA_R
            and applied.get("lora_alpha") == LORA_ALPHA
            and applied.get("lora_dropout") == LORA_DROPOUT
            and bool(applied.get("n_modules_adapted"))
            and applied.get("all_linear_fully_covered") is True
        )
    else:
        lora_exact = False  # no measurement to certify → never vacuously True

    gate = {
        # imp.md: "one LoRA fine-tune step runs end-to-end on a single A4000"
        "step_runs_end_to_end": _pos_int(gate_run.get("n_timed_steps")) is not None,
        # ...and it must actually have trained the adapters, not just executed.
        "lora_grads_flow": _grads_flow(grad_flow),
        # The VRAM number is only about §10.3's config if §10.3's config is what ran.
        "lora_config_exact": lora_exact,
        # imp.md: "peak VRAM measured and reported <= 16 GB (bf16 + gradient checkpointing)"
        "peak_vram_within_budget": within_budget,
        "fits_on_device": fits_on_device,
        # The §10.2 question is about an A4000 (sm_86); a pass measured elsewhere is not one.
        "measured_on_sm86": _is_sm86(hardware),
        # The §10.3 config the budget is claimed under must be the one that ran.
        "bf16_and_gradient_checkpointing": _bf16_and_ckpt(wrap_info),
        # The carried P1-15 handoff (not in imp.md's gate; see the module docstring).
        "attention_forward_verified": bool(forward_verified),
        # imp.md: "steps/sec reported" — a real, positive rate, not merely a present key.
        "steps_per_sec_reported": _num(gate_run.get("steps_per_sec")) is not None
        and gate_run["steps_per_sec"] > 0,
    }
    gate["overall_pass"] = all(gate.values())

    report = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "step": SMOKE_STEP,
        "measured": bool(measured),
        "backbone": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "hidden_dim": D_MODEL,
            "n_layer": N_LAYERS,
            "parity_confirmed": True,  # ADR-0002 A9 (P1-13)
        },
        # The PINNED SPEC, flagged as an echo (the P1-15 convention) — NOT a measurement.
        # `gradient_checkpointing` is sourced from the wrap so this block cannot advertise a
        # setting the run did not use; what the model actually carried is in `wrap`, and the
        # gate's clauses are re-derived from there.
        "lora": {
            **lora_config_kwargs(),
            "is_pinned_spec_echo": True,
            "dtype": TRAIN_DTYPE,
            "gradient_checkpointing": bool(wrap_info.get("gradient_checkpointing")),
        },
        "wrap": dict(wrap_info),
        "attention": {
            # What P1-15 recorded, re-derived here through select_attention_backend.
            "selected": attn_selected,
            # What this run's model actually carried (they differ iff the FA-2 forward failed).
            "used": attn_used,
            "fallback": ATTN_SDPA,
            "reason": attn_reason,
            "model_supports_flash_attn": bool(supports_fa),
            "sm86_evidence": dict(evidence),
            # THE P1-15 HANDOFF, now measured rather than deferred.
            "forward_verified_on_sm86": bool(forward_verified),
            "forward_error": forward_error,
            "forward_verification_note": (
                "P1-15 selected this backend on an IMPORT-level sm_86 confirmation (ADR-0002 "
                "A5 carries no FA-2 forward) and deferred the forward here; this run "
                "exercises it on the measured device."
            ),
        },
        "config": smoke_config(),
        "measured_smoke": dict(measured_smoke),
        "vram_budget": {
            "budget_gib": VRAM_BUDGET_GIB,
            "source": "PRD §10.2 condition (b) / §10.3 / §15; ADR-0002 D6(b) — the A4000's 16 GB",
            "peak_vram_gib": peak,
            "device_total_memory_gib": total_mem,
            "within_budget": within_budget,
            "on_miss": (
                "CLAUDE.md §7 stop-and-ask — escalate to the FSDP FULL_SHARD fallback or the "
                "§10.2 RNA-FM swap consideration. The budget is NOT weakenable (A7/A9 precedent)."
            ),
        },
        "hardware": hardware,
        "env": {"python": platform.python_version(), "platform": platform.platform()},
        "gate": gate,
        "git_sha": provenance.git_sha(),
    }
    return _sanitize(report)


def validate_smoke_report(report: Mapping[str, Any]) -> list[str]:
    """Return a (possibly empty) list of schema/consistency errors for a P1-16 report.

    Stdlib-only and total (never raises), so the committed artifact is re-checked on every
    push in a bare CI env with no torch. Empty ⇒ valid.
    """
    if not isinstance(report, Mapping):
        return [f"report is not a mapping (got {type(report).__name__})"]

    errs: list[str] = []
    for blk in _SMOKE_BLOCKS:
        if blk not in report:
            errs.append(f"missing block: {blk}")
        elif not isinstance(report[blk], Mapping):
            errs.append(f"block is not a mapping: {blk}")
    if errs:
        return errs

    if report.get("schema_version") != SMOKE_SCHEMA_VERSION:
        errs.append(f"schema_version != {SMOKE_SCHEMA_VERSION!r}")
    if report.get("step") != SMOKE_STEP:
        errs.append(f"step != {SMOKE_STEP!r}")

    bb = report["backbone"]
    if bb.get("repo_id") != REPO_ID:
        errs.append("backbone.repo_id != pinned REPO_ID")
    if bb.get("revision") != REVISION:
        errs.append("backbone.revision != pinned REVISION (the parity-confirmed checkpoint)")

    # --- the §10.3 contract the VRAM claim is made under -------------------------------- #
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

    # --- §10.3 honesty: this step may not claim to be science --------------------------- #
    cfg = report["config"]
    for key, want in (
        ("pinned", False),
        ("data_is_synthetic", True),
        ("loss_is_placeholder", True),
        ("is_science", False),
    ):
        if _bad_bool(cfg.get(key), want):
            errs.append(
                f"config.{key} must be {want} — P1-16 is a mechanics/VRAM smoke on seeded "
                "random RNA against a placeholder loss, and nothing pins its shape (§10.3)"
            )

    # --- the attention decision + the P1-15 handoff -------------------------------------- #
    attn = report["attention"]
    for key in ("selected", "used"):
        if attn.get(key) not in (ATTN_FLASH2, ATTN_SDPA, ATTN_EAGER):
            errs.append(f"attention.{key} is not a known backend: {attn.get(key)!r}")
    if attn.get("fallback") != ATTN_SDPA:
        errs.append(f"attention.fallback must be {ATTN_SDPA!r} (PRD §10.3)")
    fv = attn.get("forward_verified_on_sm86")
    if not isinstance(fv, bool):
        errs.append(
            f"attention.forward_verified_on_sm86 must be a bool, got {type(fv).__name__} ({fv!r})"
        )
    # A verified forward means the SELECTED backend ran. If `used` fell back to SDPA, the
    # selected backend demonstrably did NOT run — the report may not claim otherwise.
    elif fv and attn.get("used") != attn.get("selected"):
        errs.append(
            "attention.forward_verified_on_sm86 is True but attention.used != "
            "attention.selected — a run that fell back cannot have verified the selection"
        )
    elif not fv and not attn.get("forward_error"):
        errs.append(
            "attention.forward_verified_on_sm86 is False but no forward_error is recorded — "
            "an unverified forward must say why (§10.3, no silent assumption)"
        )

    # --- the VRAM claim ------------------------------------------------------------------ #
    budget = report.get("vram_budget")
    if not isinstance(budget, Mapping):
        errs.append("missing block: vram_budget")
        return errs
    if budget.get("budget_gib") != VRAM_BUDGET_GIB:
        errs.append(
            f"vram_budget.budget_gib != {VRAM_BUDGET_GIB} — the PRD §10.2 budget is not "
            "weakenable (CLAUDE.md §7/§10.3)"
        )
    peak = budget.get("peak_vram_gib")
    within = budget.get("within_budget")
    if not isinstance(within, bool):
        errs.append(f"vram_budget.within_budget must be a bool, got {within!r}")
    elif report.get("measured") is True:
        if not isinstance(peak, (int, float)) or isinstance(peak, bool) or not peak > 0:
            errs.append(
                "vram_budget.peak_vram_gib must be a positive number in a measured report — "
                "a CPU/absent measurement cannot answer the §10.2 A4000 question"
            )
        # The verdict must FOLLOW from the number, not be asserted alongside it.
        elif within != (float(peak) <= VRAM_BUDGET_GIB):
            errs.append(
                f"vram_budget.within_budget={within} does not follow from peak_vram_gib="
                f"{peak} vs budget {VRAM_BUDGET_GIB}"
            )

    gate = report["gate"]
    gate_keys = (
        "step_runs_end_to_end",
        "lora_grads_flow",
        "lora_config_exact",
        "peak_vram_within_budget",
        "fits_on_device",
        "measured_on_sm86",
        "bf16_and_gradient_checkpointing",
        "attention_forward_verified",
        "steps_per_sec_reported",
    )
    for k in (*gate_keys, "overall_pass"):
        if k not in gate:
            errs.append(f"gate missing key: {k}")
    if errs:
        return errs

    if not isinstance(report.get("measured"), bool):
        errs.append(f"report.measured must be a bool, got {report.get('measured')!r}")
        return errs
    if not report["measured"]:
        if gate.get("overall_pass") is True:
            errs.append("gate.overall_pass is True but report.measured is not set")
        return errs

    # The gate's VRAM clause must agree with the budget block's verdict — two records of the
    # same fact that disagree mean one of them is wrong, and we cannot know which.
    if isinstance(within, bool) and _bad_bool(gate.get("peak_vram_within_budget"), within):
        errs.append("gate.peak_vram_within_budget != vram_budget.within_budget")

    # ---- RE-DERIVE every clause from its recorded evidence -------------------------------
    # `expected_pass` below only catches a clause flipped FALSE. These catch the dangerous
    # direction — a clause fabricated TRUE over evidence that says otherwise (§10.3).
    wrap = report["wrap"]
    hardware = report["hardware"]
    ms_block = report["measured_smoke"]
    run_blk = ms_block.get("gate_run")
    if not isinstance(run_blk, Mapping):
        # Name the missing evidence explicitly rather than letting every clause below fail
        # with its own downstream symptom — the operator needs the cause.
        errs.append("measured_smoke.gate_run must be a mapping in a measured report")
        return errs

    # gate.fits_on_device must FOLLOW from the recorded peak vs the recorded card size.
    total_mem = ms_block.get("device_total_memory_gib")
    if _num(peak) is not None and _num(total_mem) is not None:
        if _bad_bool(gate.get("fits_on_device"), float(peak) <= float(total_mem)):
            errs.append(
                f"gate.fits_on_device does not follow from peak_vram_gib={peak} vs "
                f"measured_smoke.device_total_memory_gib={total_mem}"
            )
    elif _bad_bool(gate.get("fits_on_device"), False):
        errs.append(
            "gate.fits_on_device is True but measured_smoke.device_total_memory_gib is not a "
            "number — the clause is unevidenced"
        )

    # gate.measured_on_sm86 must FOLLOW from the measured compute capability.
    if _bad_bool(gate.get("measured_on_sm86"), _is_sm86(hardware)):
        errs.append(
            f"gate.measured_on_sm86 does not follow from hardware.cuda_capability="
            f"{hardware.get('cuda_capability')!r} / hardware.is_sm86="
            f"{hardware.get('is_sm86')!r} (sm_86 == {list(SM86_CAPABILITY)}) — PRD §10.2 (b) "
            "asks about an A4000"
        )

    # gate.bf16_and_gradient_checkpointing must FOLLOW from the wrap that actually ran.
    if _bad_bool(gate.get("bf16_and_gradient_checkpointing"), _bf16_and_ckpt(wrap)):
        errs.append(
            f"gate.bf16_and_gradient_checkpointing does not follow from wrap.dtype="
            f"{wrap.get('dtype')!r} / wrap.gradient_checkpointing="
            f"{wrap.get('gradient_checkpointing')!r} (PRD §10.3 pins bf16 + checkpointing)"
        )

    # The peak is recorded in TWO places (vram_budget + the gate run). Two records of the
    # same measurement that disagree mean one is wrong and we cannot tell which — so the
    # budget's number, which the whole §10.2 (b) verdict rests on, must be the run's.
    run_peak = run_blk.get("peak_vram_gib")
    if _num(run_peak) is None or _num(peak) is None or float(run_peak) != float(peak):
        errs.append(
            f"measured_smoke.gate_run.peak_vram_gib={run_peak!r} != vram_budget.peak_vram_gib="
            f"{peak!r} — the §10.2 (b) verdict must rest on the peak the gate run measured"
        )

    # The imp.md-named reporting clauses must FOLLOW from the measured run.
    if _bad_bool(
        gate.get("step_runs_end_to_end"), _pos_int(run_blk.get("n_timed_steps")) is not None
    ):
        errs.append(
            f"gate.step_runs_end_to_end does not follow from measured_smoke.gate_run."
            f"n_timed_steps={run_blk.get('n_timed_steps')!r} (must be a positive int)"
        )

    # The carried P1-15 handoff clause must FOLLOW from the recorded forward verification —
    # in BOTH directions, so neither a fabricated pass nor a fabricated failure can stand.
    if _bad_bool(
        gate.get("attention_forward_verified"), attn.get("forward_verified_on_sm86") is True
    ):
        errs.append(
            "gate.attention_forward_verified does not follow from "
            f"attention.forward_verified_on_sm86={attn.get('forward_verified_on_sm86')!r}"
        )

    # The mechanics clause must FOLLOW from the measured grad counts, not from its own
    # summary booleans: a step that ran while training nothing measures the VRAM of nothing.
    if _bad_bool(gate.get("lora_grads_flow"), _grads_flow(run_blk.get("grad_flow"))):
        errs.append(
            f"gate.lora_grads_flow does not follow from measured_smoke.gate_run.grad_flow="
            f"{run_blk.get('grad_flow')!r}"
        )
    sps = run_blk.get("steps_per_sec")
    if _bad_bool(gate.get("steps_per_sec_reported"), _num(sps) is not None and sps > 0):
        errs.append(
            f"gate.steps_per_sec_reported does not follow from measured_smoke.gate_run."
            f"steps_per_sec={sps!r} — imp.md requires steps/sec REPORTED (a real positive rate)"
        )

    # gate.lora_config_exact must FOLLOW from the APPLIED LoRA config read off the model —
    # the P1-15 anti-tautology lesson. Without this the VRAM number could be measured under
    # any config while the report certifies the §10.3 pins.
    applied = wrap.get("applied_lora")
    if not isinstance(applied, Mapping):
        if _bad_bool(gate.get("lora_config_exact"), False):
            errs.append(
                "gate.lora_config_exact is True but wrap.applied_lora is missing — a measured "
                "report must certify the pins against the config PEFT put on the model, not "
                "against the constants that produced it"
            )
    else:
        for key, want in (
            ("r", LORA_R),
            ("lora_alpha", LORA_ALPHA),
            ("lora_dropout", LORA_DROPOUT),
        ):
            if applied.get(key) != want:
                errs.append(
                    f"wrap.applied_lora.{key} != PRD §10.3 pinned {want!r} — the VRAM number "
                    "was measured under a config that is not the pinned one"
                )
        expected_exact = (
            applied.get("r") == LORA_R
            and applied.get("lora_alpha") == LORA_ALPHA
            and applied.get("lora_dropout") == LORA_DROPOUT
            and bool(applied.get("n_modules_adapted"))
            and applied.get("all_linear_fully_covered") is True
        )
        if _bad_bool(gate.get("lora_config_exact"), expected_exact):
            errs.append(
                "gate.lora_config_exact does not follow from wrap.applied_lora "
                f"(r={applied.get('r')!r}, alpha={applied.get('lora_alpha')!r}, "
                f"dropout={applied.get('lora_dropout')!r}, "
                f"n_modules_adapted={applied.get('n_modules_adapted')!r}, "
                f"all_linear_fully_covered={applied.get('all_linear_fully_covered')!r})"
            )

    expected_pass = all(gate.get(k) is True for k in gate_keys)
    if _bad_bool(gate.get("overall_pass"), expected_pass):
        errs.append("gate.overall_pass is not the AND of the individual gate checks")

    # A measured pass must carry the grad-flow evidence: a step that ran without training the
    # adapters would satisfy "runs end-to-end" while measuring the VRAM of nothing (§10.3).
    # (The gate CLAUSE is re-derived above via `_grads_flow`; these name the exact field at
    # fault, which is what an operator reading a failure needs.)
    gf = run_blk.get("grad_flow")
    if not isinstance(gf, Mapping):
        errs.append(
            "measured_smoke.gate_run.grad_flow missing — the mechanics claim is unevidenced"
        )
    else:
        n_lora = _pos_int(gf.get("n_lora_params"))
        n_grad = _pos_int(gf.get("n_lora_params_with_grad"))
        if n_lora is None:
            errs.append(
                f"grad_flow.n_lora_params={gf.get('n_lora_params')!r} must be a positive int — "
                "nothing was adapted (a bool is not a count)"
            )
        elif n_grad != n_lora:
            errs.append(
                f"grad_flow: only {gf.get('n_lora_params_with_grad')!r} of {n_lora} LoRA params "
                "received a gradient — the step ran but did not train the adapters (frozen-base "
                "+ gradient-checkpointing requires enable_input_require_grads)"
            )

    return errs


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--dry-run`` (P1-15, LOCAL) or ``--smoke`` (P1-16, SLURM one-A4000)."""
    ap = argparse.ArgumentParser(description="P1-15/P1-16 Stage-2 LoRA/FSDP harness")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="P1-15: build + wrap + record (LOCAL)")
    mode.add_argument(
        "--smoke", action="store_true", help="P1-16: one LoRA step + peak VRAM on one A4000"
    )
    ap.add_argument("--tiny", action="store_true", help="wrap a tiny fixture, not the 2.5 GB giga")
    ap.add_argument("--out", default=None, help="report path (defaults per mode)")
    ap.add_argument(
        "--require-cuda",
        action="store_true",
        help="P1-16: fail loud if no CUDA device (a CPU run cannot answer the §10.2 question)",
    )
    ap.add_argument(
        "--adapter-dir", default=None, help="P1-16: where to write the throwaway LoRA adapter"
    )
    args = ap.parse_args(argv)

    if args.dry_run:
        report = run_harness_dryrun(out_path=args.out or DEFAULT_OUT, tiny=args.tiny)
        print(json.dumps(report["attention"], indent=2, sort_keys=True))
        print(json.dumps(report["gate"], indent=2, sort_keys=True))
        return 0 if report["gate"]["overall_pass"] else 1

    if args.tiny:
        ap.error("--tiny is a P1-15 dry-run fixture; P1-16 measures the real backbone on a GPU")
    report = run_vram_smoke(
        out_path=args.out or DEFAULT_SMOKE_OUT,
        require_cuda=args.require_cuda,
        adapter_dir=args.adapter_dir,
        log=lambda m: print(f"[P1-16] {m}", flush=True),
    )
    print(json.dumps(report["vram_budget"], indent=2, sort_keys=True))
    print(json.dumps(report["gate"], indent=2, sort_keys=True))
    return 0 if report["gate"]["overall_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
