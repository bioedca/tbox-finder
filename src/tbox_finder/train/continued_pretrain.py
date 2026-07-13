"""P1-11 — Probe-triggered GTDB-prokaryotic **continued-pretraining** (MLM) pass.

Domain-adaptive continued pretraining of the human-reference-pretrained Caduceus-PS
backbone on a **GTDB-prokaryotic DNA shard**, using the checkpoint's *native* masked-
language-modeling head (``CaduceusForMaskedLM``, a weight-tied RCPS LM head). This is
**fallback #2** of the ADR-0002 D7 / PRD §10.1 transfer-fallback ladder — run only to
rescue a **weak-transfer / NO-GO** P1-07 verdict before escalating to the NT-multispecies
backbone (P1-10):

    full fine-tune (default)  →  **GTDB continued-pretraining (this step)**  →  NT-multispecies

**PROBE-TRIGGERED — NOT AUTO-RUN.** The measured P1-07 verdict was **GO**, so this pass is
**skipped entirely**; this module is the *authored, executable* fallback the PRD §18.1 P1
**fallback-triggered** gate needs — operationalized in the imp.md P1 exit-gate checklist as
"GTDB continued-pretraining pass authored + probe-triggered only". It is never part of
``rule all`` and is submitted only after a **compute-budget §7 sign-off + SLURM submit-ack**
(CLAUDE.md §7/§9.3). A passing P1-07 makes every one of those Stops moot.

**Objective (implementer choice — NOT pinned by PRD/ADR-0002/ADR-0003/imp.md).** Standard
BERT-style masked language modeling: ~15% of maskable nucleotides are selected; of those
80% → ``[MASK]``, 10% → a random base, 10% → left unchanged; the loss is cross-entropy over
the LM logits at the selected positions only. Caduceus was itself MLM-pretrained, so
continuing that objective on prokaryotic DNA is domain adaptation, not a new task. The
masking scheme is recorded in ``conf/train/gtdb_cpt.yaml`` and the run report; it can be
re-tuned at trigger-time. **Note:** ``CaduceusForMaskedLM``'s built-in loss ignores the
``[PAD]`` id (4), *not* the HF-standard ``-100`` — so this harness builds ``-100`` labels
and computes its own :func:`mlm_cross_entropy` over ``output.logits`` (context7-verified
``MaskedLMOutput.logits`` shape ``(B, L, vocab_size)``; transformers 4.57.5).

**Budget (ADR-0003 D7).** The pass is an **additive term** in the *separate* aggregate
training/sweep GPU-hour budget whose numeric value is **frozen at P1 first measurement** —
so the config ``max_steps`` and the sbatch ``--time`` are **provisional bounds** set at
authoring, finalized by a trigger-time sizing smoke. No GPU-hour figure is fabricated here
(§10.3).

**Re-clear gate (moot unless triggered).** After the pass, the P1-07 go/no-go is re-run on
the continued-pretrained checkpoint (:mod:`tbox_finder.train.stage1_smoke`); the pass is
kept iff its min-core-F1 **improves vs P1-07** and clears the ADR-0002 A6 go threshold
(``DEFAULT_GO_THRESHOLD``, single-sourced from the P1-07 grader), else escalate to the
NT-multispecies fallback (P1-10). That re-run is a trigger-time follow-up, not wired here.

**Compute.** SLURM ``gpu`` A4000 (the packaged Mamba forward hard-requires
``selective_scan_cuda``, ADR-0002 A2 C2 → GPU only; no CPU forward). Multi-GPU data
parallel via :mod:`accelerate` (the 7.73 M-param model is tiny → plain DDP, no FSDP/ZeRO).
Rule/Entry: :func:`main` (Hydra) → :func:`run_cpt`.

**Two-tier importability.** All heavy imports (``torch``, ``transformers``, ``accelerate``,
``numpy``, ``wandb``) are lazy inside functions, so the masking/windowing/report machinery
imports and unit-tests in a bare env (the CI Tier-1 path).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import provenance
from tbox_finder.models.caduceus_backbone import (
    PRETRAINING_DOMAIN,
    REPO_ID,
    REVISION,
)

# The ADR-0002 A6 go/no-go thresholds are single-sourced from the P1-07 harness (the
# re-clear gate reuses that grader) — never re-declared here, so a threshold change lands
# in exactly one place.
from tbox_finder.train.stage1_smoke import (
    DEFAULT_GO_THRESHOLD,
    DEFAULT_NOGO_THRESHOLD,
)

# --------------------------------------------------------------------------------------
# Constants (single-sourced from code; a config override can never weaken an identity).
# --------------------------------------------------------------------------------------
SCHEMA_VERSION = "1"
STEP = "P1-11"
GENERATED_BY = "src/tbox_finder/train/continued_pretrain.py"
ADR = "ADR-0002"
ENV_LOCK = "envs/ml-dna.conda-lock.yml"

OBJECTIVE = "masked_language_modeling"
MASK_TOKEN = "[MASK]"  # the Caduceus char-tokenizer mask token (id 3)
BASE_TOKENS = "ACGT"  # the four DNA bases used for the 10%-random-replacement pool

#: MLM ignore label — the HF-standard cross-entropy ignore index (NOT the checkpoint's
#: ``pad_token_id``; the built-in Caduceus MLM loss uses pad, so we compute our own, §10.3).
IGNORE_INDEX = -100

#: Default BERT-style masking scheme (implementer choice; ADR-0002/PRD/imp pin none).
DEFAULT_MLM_PROBABILITY = 0.15
DEFAULT_MASK_REPLACE_PROB = 0.80  # of selected → [MASK]
DEFAULT_RANDOM_REPLACE_PROB = 0.10  # of selected → random base (remainder → unchanged)

# Provisional bounds (ADR-0003 D7: numeric budget frozen at P1 first measurement).
DEFAULT_MAX_STEPS = 5_000
DEFAULT_WINDOW_NT = 1_024  # matches conf/model/caduceus_stage1.yaml window_nt
DEFAULT_STRIDE_NT = 512
DEFAULT_MIN_WINDOW_NT = 128

# Repo-relative defaults (overridable via Hydra / CLI).
DEFAULT_SHARD = "data/interim/gtdb_cpt/shard.fasta"  # expected input; sourced at trigger-time
DEFAULT_CHECKPOINT_DIR = "checkpoints/p1/caduceus_gtdb_cpt"
DEFAULT_REPORT = "reports/p1/gtdb_cpt.json"


# --------------------------------------------------------------------------------------
# Run configuration (a plain dataclass so ``run_cpt`` is callable without Hydra).
# --------------------------------------------------------------------------------------
@dataclass
class CPTConfig:
    """Fully-resolved knobs for :func:`run_cpt`. Seeds + determinism are §8.3."""

    seed: int = 42
    max_steps: int = DEFAULT_MAX_STEPS
    lr: float = 1.0e-4  # continued-pretraining LR < the 3e-4 fine-tune smoke LR
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    log_every: int = 50
    mlm_probability: float = DEFAULT_MLM_PROBABILITY
    mask_replace_prob: float = DEFAULT_MASK_REPLACE_PROB
    random_replace_prob: float = DEFAULT_RANDOM_REPLACE_PROB
    window_nt: int = DEFAULT_WINDOW_NT
    stride_nt: int = DEFAULT_STRIDE_NT
    min_window_nt: int = DEFAULT_MIN_WINDOW_NT
    device: str | None = None  # None ⇒ accelerator device (cuda on the cluster)
    shard_path: str = DEFAULT_SHARD
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    report_path: str = DEFAULT_REPORT
    save_checkpoint: bool = True
    wandb_mode: str = "offline"
    wandb_project: str = "tbox-finder"
    wandb_entity: str | None = None
    wandb_dir: str = "wandb"

    def sanitized_knobs(self) -> dict[str, Any]:
        """The subset of knobs recorded in the report/provenance (no paths)."""
        return {
            "seed": self.seed,
            "max_steps": self.max_steps,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "warmup_steps": self.warmup_steps,
            "mlm_probability": self.mlm_probability,
            "mask_replace_prob": self.mask_replace_prob,
            "random_replace_prob": self.random_replace_prob,
            "window_nt": self.window_nt,
            "stride_nt": self.stride_nt,
            "min_window_nt": self.min_window_nt,
        }

    @property
    def keep_prob(self) -> float:
        """The unchanged-token fraction of selected positions (BERT's remaining 10%)."""
        return 1.0 - self.mask_replace_prob - self.random_replace_prob


# --------------------------------------------------------------------------------------
# Pure stdlib helpers (bare-testable — no numpy / torch).
# --------------------------------------------------------------------------------------
def _sanitize(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` so the JSON is strict (no NaN/Inf)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _finite_number(v: Any) -> bool:
    """True iff ``v`` is a real, finite number (bools rejected)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _nonneg_number(v: Any) -> bool:
    """True iff ``v`` is a finite, non-negative real number (bools rejected)."""
    return _finite_number(v) and float(v) >= 0.0


def _prob01(v: Any) -> bool:
    """True iff ``v`` is a finite number in the closed unit interval."""
    return _finite_number(v) and 0.0 <= float(v) <= 1.0


def _bad_bool(value: Any, expected: bool) -> bool:
    """True iff ``value`` is not the exact boolean ``expected`` (int 0/1 rejected)."""
    return not (isinstance(value, bool) and value is expected)


def validate_masking(
    mlm_probability: float, mask_replace_prob: float, random_replace_prob: float
) -> None:
    """Fail loud on a malformed masking scheme so a stray override cannot silently no-op.

    ``mlm_probability`` must be in ``(0, 1]`` (a rate of 0 masks nothing → no MLM signal),
    and the mask/random replacement probabilities must each be in ``[0, 1]`` and sum to at
    most 1 (the remainder is the keep-unchanged fraction).
    """
    if not (_finite_number(mlm_probability) and 0.0 < float(mlm_probability) <= 1.0):
        raise ValueError(f"mlm_probability must be in (0, 1], got {mlm_probability!r}")
    if not (_prob01(mask_replace_prob) and _prob01(random_replace_prob)):
        raise ValueError("mask_replace_prob and random_replace_prob must each be in [0, 1]")
    if float(mask_replace_prob) + float(random_replace_prob) > 1.0 + 1e-9:
        raise ValueError(
            f"mask_replace_prob ({mask_replace_prob}) + random_replace_prob "
            f"({random_replace_prob}) must be <= 1 (the remainder is keep-unchanged)"
        )


def read_fasta_sequences(path: str | Path) -> list[str]:
    """Minimal stdlib FASTA reader → the list of upper-cased sequences (``.gz`` supported).

    Concatenates the wrapped sequence lines of each ``>`` record; ignores blank lines. Kept
    pure-stdlib (no Biopython) so the windowing path unit-tests in a bare env. Raises
    :class:`FileNotFoundError` on a missing path and :class:`ValueError` on an empty file
    (a fail-loud input contract — the GTDB shard is a required, externally-sourced input).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GTDB continued-pretraining shard not found: {p}")

    opener = _open_maybe_gzip(p)
    seqs: list[str] = []
    cur: list[str] = []
    with opener as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur).upper())
    if not seqs:
        raise ValueError(f"GTDB continued-pretraining shard is empty (no FASTA records): {p}")
    return seqs


def _open_maybe_gzip(path: Path):
    """Open ``path`` as text, transparently gunzipping ``.gz`` (stdlib only)."""
    if str(path).endswith(".gz"):
        import gzip  # lazy stdlib

        return gzip.open(path, "rt")
    return open(path, encoding="utf-8")


def window_sequences(
    sequences: Iterable[str],
    *,
    window_nt: int,
    stride_nt: int,
    min_window_nt: int,
) -> list[str]:
    """Chunk each contig into fixed-stride windows (deterministic order).

    A window is emitted every ``stride_nt`` nucleotides; the trailing window of a contig is
    kept only if it is at least ``min_window_nt`` long (shorter tails are dropped rather than
    padded, so every training window is dense DNA). Raises :class:`ValueError` on a
    non-positive ``window_nt``/``stride_nt`` or ``min_window_nt > window_nt``.
    """
    if window_nt <= 0 or stride_nt <= 0:
        raise ValueError("window_nt and stride_nt must be positive")
    if min_window_nt <= 0 or min_window_nt > window_nt:
        raise ValueError("min_window_nt must be in (0, window_nt]")
    windows: list[str] = []
    for seq in sequences:
        n = len(seq)
        if n < min_window_nt:
            continue
        start = 0
        while start < n:
            chunk = seq[start : start + window_nt]
            if len(chunk) >= min_window_nt:
                windows.append(chunk)
            if start + window_nt >= n:
                break
            start += stride_nt
    return windows


def build_mlm_example(
    token_ids: Sequence[int],
    *,
    special_ids: Sequence[int],
    mask_id: int,
    base_ids: Sequence[int],
    mlm_probability: float,
    mask_replace_prob: float,
    random_replace_prob: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Build one BERT-style masked-LM example from a token-id window (pure/deterministic).

    Selects each **maskable** position (a token id not in ``special_ids``) independently with
    probability ``mlm_probability``. For every selected position the label is the *original*
    id; the input is replaced by ``mask_id`` with probability ``mask_replace_prob``, by a
    random base (from ``base_ids``) with probability ``random_replace_prob``, and otherwise
    left unchanged. Non-selected positions get label :data:`IGNORE_INDEX` and are unchanged.

    Determinism (§8.3): all randomness is drawn from the passed :class:`random.Random`, so a
    seeded run is reproducible and this is unit-testable with no torch present.

    Returns ``(masked_ids, labels)`` — equal-length lists.
    """
    specials = set(int(s) for s in special_ids)
    bases = [int(b) for b in base_ids]
    masked: list[int] = []
    labels: list[int] = []
    rand_cut = float(mask_replace_prob) + float(random_replace_prob)
    for tid in token_ids:
        tid = int(tid)
        if tid in specials or rng.random() >= float(mlm_probability):
            masked.append(tid)
            labels.append(IGNORE_INDEX)
            continue
        labels.append(tid)  # the target is always the original token
        action = rng.random()
        if action < float(mask_replace_prob):
            masked.append(int(mask_id))
        elif action < rand_cut:
            masked.append(rng.choice(bases))
        else:
            masked.append(tid)  # keep unchanged (still a supervised target)
    return masked, labels


def _epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """A deterministic, ``PYTHONHASHSEED``-independent per-epoch shuffle of ``range(n)``."""
    rng = random.Random((int(seed) * 1_000_003) ^ (int(epoch) + 1))
    order = list(range(n))
    rng.shuffle(order)
    return order


def window_index_stream(n: int, num_processes: int, process_index: int, seed: int):
    """Yield an endless, per-process **shuffled** window-index schedule (DDP-disjoint).

    Each epoch is a fresh :func:`_epoch_order` permutation of ``range(n)``; process ``p`` of
    ``num_processes`` takes the strided slice ``order[p::num_processes]`` — so across a full
    epoch every window is visited exactly once and the per-process slices are **disjoint**
    (correct data-parallel coverage, unlike an unshuffled modulo cycle). When ``n <
    num_processes`` a process whose slice is empty falls back to a single index so the loop
    can never spin without yielding. Determinism (§8.3): the schedule is a pure function of
    ``(seed, epoch, process_index)`` and is ``PYTHONHASHSEED``-independent.
    """
    epoch = 0
    while True:
        order = _epoch_order(n, seed, epoch)
        shard = order[process_index::num_processes] or [order[process_index % n]]
        yield from shard
        epoch += 1


# --------------------------------------------------------------------------------------
# Report — assembly + fail-closed self-validation (stdlib; validates a measured run).
# --------------------------------------------------------------------------------------
def build_report(
    *,
    cfg: CPTConfig,
    n_records: int,
    n_windows: int,
    n_train_tokens: int,
    steps: int,
    final_loss: float | None,
    mean_last_loss: float | None,
    masked_token_count: int,
    num_processes: int,
    wandb_run_id: str | None,
    measured: bool = True,
) -> dict[str, Any]:
    """Assemble the ``gtdb_cpt.json`` continued-pretraining report."""
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "measured": bool(measured),
        "probe_triggered": True,  # fallback #2; run only on a NO-GO/weak-transfer P1-07
        "objective": OBJECTIVE,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "backbone": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "pretraining_domain": PRETRAINING_DOMAIN,
            "mlm_head": "CaduceusForMaskedLM",
        },
        "masking": {
            "mlm_probability": float(cfg.mlm_probability),
            "mask_replace_prob": float(cfg.mask_replace_prob),
            "random_replace_prob": float(cfg.random_replace_prob),
            "keep_prob": float(cfg.keep_prob),
            "mask_token": MASK_TOKEN,
            "ignore_index": IGNORE_INDEX,
        },
        "data": {
            "shard_path": str(cfg.shard_path),
            "domain": "GTDB-prokaryotic DNA",
            "n_records": int(n_records),
            "n_windows": int(n_windows),
            "window_nt": int(cfg.window_nt),
            "stride_nt": int(cfg.stride_nt),
            "n_train_tokens": int(n_train_tokens),
        },
        "optim": {
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "grad_clip": float(cfg.grad_clip),
            "warmup_steps": int(cfg.warmup_steps),
            "lr_schedule": LR_SCHEDULE,
        },
        "budget": {
            # ADR-0003 D7: the training/sweep GPU-hour budget's numeric value is frozen at
            # P1 FIRST MEASUREMENT — max_steps is a PROVISIONAL bound set at authoring, not a
            # pinned GPU-hour figure (§10.3: no fabricated compute number).
            "kind": "provisional",
            "max_steps": int(cfg.max_steps),
            "note": (
                "additive term in the separate aggregate training/sweep GPU-hour budget "
                "(ADR-0003 D7); numeric frozen at P1 first measurement — provisional here"
            ),
        },
        "train": {
            "steps": int(steps),
            "final_loss": final_loss,
            "mean_last_loss": mean_last_loss,
            "masked_token_count": int(masked_token_count),
            "num_processes": int(num_processes),
        },
        "gonogo_reference": {
            # The re-clear gate reuses the P1-07 grader; thresholds single-sourced (ADR-0002 A6).
            "go_threshold": float(DEFAULT_GO_THRESHOLD),
            "nogo_threshold": float(DEFAULT_NOGO_THRESHOLD),
            "source": "ADR-0002 A6 via tbox_finder.train.stage1_smoke",
        },
        "next_step_on_trigger": (
            "re-run the P1-07 go/no-go (tbox_finder.train.stage1_smoke) on this "
            "continued-pretrained checkpoint; KEEP iff min-core-F1 improves vs P1-07 AND "
            f"clears ADR-0002 A6 (>= {DEFAULT_GO_THRESHOLD}), else escalate to "
            "NT-multispecies (P1-10)"
        ),
        "wandb_run_id": wandb_run_id,
    }
    return _sanitize(report)


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of schema/consistency problems (empty ⇒ valid). Never raises.

    Fails **closed**: identity fields must match the code pins, the probe-triggered/objective
    honesty flags must hold, and for a *measured* run the training metrics must be finite and
    the masking scheme well-formed. A hand-edited report that contradicts the pins is rejected
    (§8.7/§10.3).
    """
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if report.get("step") != STEP:
        errors.append(f"step must be {STEP!r}")
    if report.get("objective") != OBJECTIVE:
        errors.append(f"objective must be {OBJECTIVE!r}")
    if _bad_bool(report.get("probe_triggered"), True):
        errors.append("probe_triggered must be boolean True (ADR-0002 D7 fallback #2)")

    backbone = report.get("backbone")
    if not isinstance(backbone, dict):
        errors.append("backbone block missing")
    else:
        if backbone.get("revision") != REVISION:
            errors.append("backbone.revision must be the pinned Caduceus-PS commit")
        if backbone.get("repo_id") != REPO_ID:
            errors.append("backbone.repo_id must be the pinned Caduceus-PS repo")

    masking = report.get("masking")
    if not isinstance(masking, dict):
        errors.append("masking block missing")
    else:
        p = masking.get("mlm_probability")
        if not (_finite_number(p) and 0.0 < float(p) <= 1.0):
            errors.append("masking.mlm_probability must be in (0, 1]")
        for key in ("mask_replace_prob", "random_replace_prob", "keep_prob"):
            if not _prob01(masking.get(key)):
                errors.append(f"masking.{key} must be a finite number in [0, 1]")
        parts = [masking.get(k) for k in ("mask_replace_prob", "random_replace_prob", "keep_prob")]
        if all(_prob01(v) for v in parts) and abs(sum(float(v) for v in parts) - 1.0) > 1e-6:
            errors.append("masking mask/random/keep probabilities must sum to 1")
        if masking.get("ignore_index") != IGNORE_INDEX:
            errors.append(f"masking.ignore_index must be {IGNORE_INDEX}")

    gono = report.get("gonogo_reference")
    if not isinstance(gono, dict):
        errors.append("gonogo_reference block missing")
    else:
        if gono.get("go_threshold") != DEFAULT_GO_THRESHOLD:
            errors.append("gonogo_reference.go_threshold must match ADR-0002 A6 (stage1_smoke)")
        if gono.get("nogo_threshold") != DEFAULT_NOGO_THRESHOLD:
            errors.append("gonogo_reference.nogo_threshold must match ADR-0002 A6 (stage1_smoke)")

    budget = report.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget block missing")
    elif budget.get("kind") != "provisional":
        # ADR-0003 D7: the numeric budget is frozen at first measurement — a report may not
        # assert a "pinned"/"final" GPU-hour figure at authoring/first-run time (§10.3).
        errors.append("budget.kind must be 'provisional' (ADR-0003 D7 numeric frozen at P1)")

    if report.get("measured"):
        train = report.get("train")
        if not isinstance(train, dict):
            errors.append("train block missing for a measured report")
        else:
            if not _nonneg_number(train.get("final_loss")):
                errors.append("train.final_loss must be a finite, non-negative number (measured)")
            if not (isinstance(train.get("steps"), int) and train["steps"] >= 0):
                errors.append("train.steps must be a non-negative int (measured)")
            mtc = train.get("masked_token_count")
            if not (isinstance(mtc, int) and not isinstance(mtc, bool) and mtc >= 0):
                errors.append("train.masked_token_count must be a non-negative int (measured)")
            elif mtc == 0:
                errors.append(
                    "train.masked_token_count must be > 0 (a measured MLM run masked something)"
                )
    return errors


# --------------------------------------------------------------------------------------
# Torch tier — determinism, MLM loss, model/tokenizer, accelerate loop (lazy imports).
# --------------------------------------------------------------------------------------
def set_determinism(seed: int) -> None:
    """Seed every RNG + disable nondeterministic fast paths (§8.3)."""
    import numpy as np  # lazy
    import torch  # lazy

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def mlm_cross_entropy(logits, labels, *, ignore_index: int = IGNORE_INDEX):
    """Masked-LM cross-entropy over the LM logits, averaged over non-ignored positions.

    ``logits`` is ``(B, L, V)`` (context7-verified ``MaskedLMOutput.logits``), ``labels`` is
    ``(B, L)`` with :data:`IGNORE_INDEX` at non-masked positions. We compute this ourselves
    (rather than passing ``labels`` to the model) because ``CaduceusForMaskedLM``'s built-in
    loss ignores ``config.pad_token_id`` (4), not ``-100`` — see the module docstring. With
    ``reduction="none"`` the ignored positions contribute exactly 0, so the sum is over the
    masked positions only; the denominator is clamped to ≥1 so an all-ignored batch yields a
    finite 0 rather than NaN.
    """
    import torch.nn.functional as F  # lazy

    logits_t = logits.transpose(1, 2)  # (B, V, L) for F.cross_entropy
    ce = F.cross_entropy(logits_t, labels, ignore_index=ignore_index, reduction="none")  # (B, L)
    valid = labels != ignore_index
    denom = valid.sum().clamp_min(1)
    return ce.sum() / denom


def _make_ids_of(tokenizer):
    """Build ``(ids_of, special_ids, mask_id, base_ids)`` from the Caduceus char tokenizer."""
    base_ids = []
    for base in BASE_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(base)
        if tid is not None:
            base_ids.append(int(tid))
    mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))
    special_ids = [int(s) for s in tokenizer.all_special_ids]
    unk = tokenizer.unk_token_id
    if unk is None:
        unk = base_ids[0]
    id_of_base = {b: tokenizer.convert_tokens_to_ids(b) for b in BASE_TOKENS}

    def ids_of(seq: str) -> list[int]:
        return [int(id_of_base.get(c, unk)) for c in seq.upper()]

    return ids_of, special_ids, mask_id, base_ids


def load_shard_windows(cfg: CPTConfig) -> tuple[list[str], int]:
    """Assert the GTDB shard is present, read it, and window it → ``(windows, n_records)``.

    Fails loud (the sbatch also pre-asserts the shard) because the GTDB-prokaryotic DNA shard
    is a **required external input sourced at trigger-time** — this harness never fabricates a
    corpus (§10.2/§10.3).
    """
    records = read_fasta_sequences(cfg.shard_path)
    windows = window_sequences(
        records,
        window_nt=cfg.window_nt,
        stride_nt=cfg.stride_nt,
        min_window_nt=cfg.min_window_nt,
    )
    if not windows:
        raise ValueError(
            f"no training windows from {cfg.shard_path} "
            f"(all {len(records)} records shorter than min_window_nt={cfg.min_window_nt})"
        )
    return windows, len(records)


class _SingleProcessAccelerator:
    """A no-op stand-in for :class:`accelerate.Accelerator` (single device, no DDP).

    Lets :func:`run_cpt` share one code path whether launched under ``accelerate launch``
    (multi-GPU on the cluster) or bare (a single-GPU local sizing smoke). It mirrors only the
    subset of the Accelerator surface the loop uses.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0

    def prepare(self, *objects):
        return objects if len(objects) != 1 else objects[0]

    def backward(self, loss) -> None:
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        import torch  # lazy

        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self) -> None:
        pass

    def gather(self, tensor):
        return tensor

    def print(self, *args, **kwargs) -> None:
        print(*args, **kwargs)


def _build_accelerator(cfg: CPTConfig):
    """Return a real :class:`accelerate.Accelerator` if available, else the single-process shim."""
    if cfg.device is not None:
        return _SingleProcessAccelerator(cfg.device)
    try:
        from accelerate import Accelerator  # lazy
    except ImportError:
        import torch  # lazy

        return _SingleProcessAccelerator("cuda" if torch.cuda.is_available() else "cpu")
    return Accelerator()


def _build_lr_scheduler(opt, cfg: CPTConfig):
    """Linear warmup (``warmup_steps``) then linear decay to 0 over ``max_steps`` (pure torch).

    A ``LambdaLR`` (no ``transformers`` dependency) so the loop stays deterministic and
    bare-testable; matches the ``transformers.get_scheduler("linear", …)`` shape. Honors the
    ``warmup_steps`` knob so the report's ``optim.warmup_steps`` reflects a real schedule.
    """
    import torch  # lazy

    warmup = max(0, int(cfg.warmup_steps))
    total = max(1, int(cfg.max_steps))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


#: The LR-schedule shape recorded in the report (honest provenance for the warmup knob).
LR_SCHEDULE = "linear_warmup_then_linear_decay"


def train_cpt(model, accelerator, windows, ids_of, special_ids, mask_id, base_ids, cfg, *, log):
    """Continued MLM pretraining loop (accelerate DDP; deterministic shuffled schedule).

    Runs exactly ``cfg.max_steps`` optimizer steps on **every** process (each draws from a
    per-process **disjoint, per-epoch-shuffled** window stream, :func:`window_index_stream`)
    so DDP never desyncs on unequal shard sizes. Applies a linear warmup+decay LR schedule
    (:func:`_build_lr_scheduler`). Returns ``(final_loss, mean_last_loss, masked_token_count,
    steps)``.
    """
    import torch  # lazy

    n = len(windows)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_lr_scheduler(opt, cfg)
    model, opt, scheduler = accelerator.prepare(model, opt, scheduler)
    model.train()

    # Per-process masking RNG (offset by process_index so processes mask differently).
    rng = random.Random((int(cfg.seed) * 2_654_435_761) ^ (int(accelerator.process_index) + 1))
    stream = window_index_stream(n, accelerator.num_processes, accelerator.process_index, cfg.seed)
    losses: list[float] = []
    masked_total = 0
    tail = max(1, int(cfg.log_every))
    for step in range(int(cfg.max_steps)):
        seq = windows[next(stream)]
        token_ids = ids_of(seq)
        masked_ids, labels = build_mlm_example(
            token_ids,
            special_ids=special_ids,
            mask_id=mask_id,
            base_ids=base_ids,
            mlm_probability=cfg.mlm_probability,
            mask_replace_prob=cfg.mask_replace_prob,
            random_replace_prob=cfg.random_replace_prob,
            rng=rng,
        )
        masked_total += sum(1 for v in labels if v != IGNORE_INDEX)
        x = torch.tensor([masked_ids], device=accelerator.device)
        y = torch.tensor([labels], device=accelerator.device)
        out = model(input_ids=x)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = mlm_cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        if cfg.grad_clip and cfg.grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % tail == 0:
            log(f"  step {step + 1}/{cfg.max_steps} mlm_loss={losses[-1]:.4f}")

    final_loss = losses[-1] if losses else None
    mean_last = (sum(losses[-tail:]) / len(losses[-tail:])) if losses else None
    return final_loss, mean_last, masked_total, len(losses)


def _init_wandb(cfg: CPTConfig, *, enabled: bool):
    """Best-effort offline W&B init (main process only); returns ``(run, run_id)``."""
    if not enabled or cfg.wandb_mode == "disabled":
        return None, None
    try:
        import wandb  # lazy
    except ImportError:
        return None, None
    try:
        run = wandb.init(
            mode=cfg.wandb_mode,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            dir=cfg.wandb_dir,
            job_type="p1-11-gtdb-cpt",
            config=cfg.sanitized_knobs(),
            reinit=True,
        )
        return run, run.id
    except Exception:  # noqa: BLE001 — W&B must never break a training run
        return None, None


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_cpt(cfg: CPTConfig, *, log=print) -> dict[str, Any]:
    """Run the continued-pretraining pass: seed → shard → MLM train → checkpoint + report.

    Returns the report dict. Never loosens a threshold or fabricates a metric (§10.3); the
    budget stays provisional (ADR-0003 D7). Writes the checkpoint (+ provenance) and the
    ``gtdb_cpt.json`` report on the main process only.
    """
    validate_masking(cfg.mlm_probability, cfg.mask_replace_prob, cfg.random_replace_prob)
    set_determinism(cfg.seed)

    accelerator = _build_accelerator(cfg)
    log(f"[{STEP}] loading GTDB shard {cfg.shard_path}")
    windows, n_records = load_shard_windows(cfg)
    n_tokens = sum(len(w) for w in windows)
    log(f"[{STEP}] {n_records} records → {len(windows)} windows, {n_tokens} nt")

    from tbox_finder.models.caduceus_backbone import (  # lazy heavy
        load_caduceus_ps_for_masked_lm,
        load_tokenizer,
    )

    model = load_caduceus_ps_for_masked_lm(device=accelerator.device)
    tokenizer = load_tokenizer()
    ids_of, special_ids, mask_id, base_ids = _make_ids_of(tokenizer)
    log(
        f"[{STEP}] backbone {REPO_ID}@{REVISION[:8]} on {accelerator.device}; "
        f"MLM p={cfg.mlm_probability} for {cfg.max_steps} steps × {accelerator.num_processes} proc"
    )

    run, wandb_run_id = _init_wandb(cfg, enabled=accelerator.is_main_process)
    final_loss, mean_last, masked_total, steps = train_cpt(
        model, accelerator, windows, ids_of, special_ids, mask_id, base_ids, cfg, log=log
    )

    report = build_report(
        cfg=cfg,
        n_records=n_records,
        n_windows=len(windows),
        n_train_tokens=n_tokens,
        steps=steps,
        final_loss=final_loss,
        mean_last_loss=mean_last,
        masked_token_count=masked_total,
        num_processes=accelerator.num_processes,
        wandb_run_id=wandb_run_id,
    )
    problems = validate_report(report)
    if problems:
        raise RuntimeError("gtdb_cpt report failed self-validation: " + "; ".join(problems))

    if run is not None:
        try:
            run.log({"mlm_loss/final": final_loss, "mlm_loss/mean_last": mean_last})
            run.finish()
        except Exception:  # noqa: BLE001
            pass

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _write_outputs(cfg, accelerator.unwrap_model(model), report, log=log)
    log(f"[{STEP}] steps={steps} final_mlm_loss={final_loss} masked_tokens={masked_total}")
    return report


def _write_outputs(cfg: CPTConfig, model, report: Mapping[str, Any], *, log=print) -> None:
    """Persist the report (git-tracked) + the continued-pretrained checkpoint & provenance."""
    import torch  # lazy

    rp = Path(cfg.report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"[{STEP}] wrote report → {rp}")

    if not cfg.save_checkpoint:
        return
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "caduceus_gtdb_cpt.pt"
    torch.save(model.state_dict(), ckpt)
    provenance.write_provenance(
        ckpt_dir / "provenance.json",
        rule="src/tbox_finder/train/continued_pretrain.py::run_cpt",
        script=GENERATED_BY,
        seed=cfg.seed,
        inputs=[cfg.shard_path],
        outputs=[ckpt],
        env_lock=ENV_LOCK if Path(ENV_LOCK).is_file() else None,
        adr=ADR,
        extra={
            "step": STEP,
            "backbone_revision": REVISION,
            "objective": OBJECTIVE,
            "final_mlm_loss": report["train"]["final_loss"],
        },
    )
    log(f"[{STEP}] wrote checkpoint → {ckpt}")


# --------------------------------------------------------------------------------------
# Entry points — Hydra (`main`) for the sbatch, argparse (`_run`) for local sizing.
# --------------------------------------------------------------------------------------
def _cfg_from_mapping(cfg: Mapping[str, Any]) -> CPTConfig:
    """Build a :class:`CPTConfig` from a (possibly nested) composed config mapping."""
    optim = dict(cfg.get("optim", {}) or {})
    tracking = dict(cfg.get("tracking", {}) or {})
    masking = dict(cfg.get("masking", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    return CPTConfig(
        seed=int(cfg.get("seed", 42)),
        max_steps=int(cfg.get("max_steps", DEFAULT_MAX_STEPS)),
        lr=float(optim.get("lr", cfg.get("lr", 1.0e-4))),
        weight_decay=float(optim.get("weight_decay", cfg.get("weight_decay", 0.01))),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        warmup_steps=int(cfg.get("warmup_steps", 100)),
        log_every=int(cfg.get("log_every", 50)),
        mlm_probability=float(masking.get("mlm_probability", DEFAULT_MLM_PROBABILITY)),
        mask_replace_prob=float(masking.get("mask_replace_prob", DEFAULT_MASK_REPLACE_PROB)),
        random_replace_prob=float(masking.get("random_replace_prob", DEFAULT_RANDOM_REPLACE_PROB)),
        window_nt=int(data.get("window_nt", cfg.get("window_nt", DEFAULT_WINDOW_NT))),
        stride_nt=int(data.get("stride_nt", cfg.get("stride_nt", DEFAULT_STRIDE_NT))),
        min_window_nt=int(
            data.get("min_window_nt", cfg.get("min_window_nt", DEFAULT_MIN_WINDOW_NT))
        ),
        device=cfg.get("device"),
        shard_path=str(data.get("shard_path", cfg.get("shard_path", DEFAULT_SHARD))),
        checkpoint_dir=str(cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)),
        report_path=str(cfg.get("report_path", DEFAULT_REPORT)),
        save_checkpoint=bool(cfg.get("save_checkpoint", True)),
        wandb_mode=str(tracking.get("mode", cfg.get("wandb_mode", "offline"))),
        wandb_project=str(tracking.get("project", cfg.get("wandb_project", "tbox-finder"))),
        wandb_entity=tracking.get("entity", cfg.get("wandb_entity")),
        wandb_dir=str(tracking.get("dir", cfg.get("wandb_dir", "wandb"))),
    )


def main() -> None:
    """Hydra entry (the sbatch target). Composes ``conf/train/gtdb_cpt.yaml``."""
    import hydra  # lazy
    from omegaconf import OmegaConf  # lazy

    @hydra.main(version_base=None, config_path="../../../conf", config_name="train/gtdb_cpt")
    def _entry(cfg) -> None:
        resolved = OmegaConf.to_container(cfg, resolve=True)
        run_cpt(_cfg_from_mapping(resolved))

    _entry()


def _run(argv: list[str] | None = None) -> int:
    """Minimal argparse CLI — used for a local sizing smoke (no Hydra composition)."""
    parser = argparse.ArgumentParser(description="P1-11 GTDB continued-pretraining (MLM)")
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, dest="max_steps")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--mlm-probability", type=float, default=DEFAULT_MLM_PROBABILITY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = CPTConfig(
        seed=args.seed,
        max_steps=args.max_steps,
        lr=args.lr,
        mlm_probability=args.mlm_probability,
        device=args.device,
        shard_path=args.shard,
        report_path=args.report,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoint=not args.no_checkpoint,
        wandb_mode=args.wandb_mode,
    )
    run_cpt(cfg)
    return 0


if __name__ == "__main__":
    if any(a.startswith("--") for a in sys.argv[1:]):
        sys.exit(_run())
    main()
