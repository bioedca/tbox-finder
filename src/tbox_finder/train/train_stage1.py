"""P2-04 — the Stage-1 training entrypoint: Hydra + DDP + W&B-offline.

Composes the three P2 halves that already exist into one runnable full fine-tune:

- **data** — P2-01's :mod:`tbox_finder.data.window_dataset` (1024-nt windows over the real
  P2-00 flank context, curriculum/oversampling, offset + strand augmentation),
- **model** — P1-04's :class:`~tbox_finder.models.stage1_segmenter.Stage1Segmenter`
  (Caduceus-PS backbone → non-averaged RC-combine → 8-class per-position head),
- **objective** — P2-02's :class:`~tbox_finder.train.objective.Stage1Loss` (focal CE +
  optional inverse-frequency weighting + optional CRF).

This module *wires*; it does not re-implement any of them. It owns exactly four things the
three halves cannot own individually: the Hydra config surface, gradient checkpointing, the
DDP shard, and the run's provenance record.

**What this step does NOT do.** It trains nothing for real: P2-05 fixes the SLURM footprint,
P2-06 sweeps γ/α/LR/window/RC-combine, P2-09 trains the production scanner and P2-11 the
class-II-CM-naive ablation (ADR-0005 D9). The smoke here asserts *composition* — that one
train+eval step runs end to end and the config resolves — and makes **no** performance claim
(``is_science=false``; CLAUDE.md §10.3). Every number this module reports is a mechanics
measurement, never a GATE-4 result.

Gradient checkpointing (PRD §10.3, hand-wired — see :func:`enable_gradient_checkpointing`)
--------------------------------------------------------------------------------------
PRD §10.3 pins *"full fine-tune fits one GPU with gradient checkpointing; DDP×8 for
throughput"* for Stage-1. The pinned checkpoint **cannot** honour that through the standard
HuggingFace path — see the function docstring for the measured evidence and the hand-wiring
this module does instead (user sign-off 2026-07-16).

Determinism (§8.3; ADR-0002 A6/A7)
----------------------------------
Seeding is delegated to :func:`tbox_finder.train.repro.set_determinism` — one implementation,
shared with the P1-07 smoke. Runs are reproducible at **metric level, not bitwise**: the
Caduceus Mamba ``selective_scan_cuda`` kernel registers no deterministic algorithm (ADR-0002
A2 C2), so ``use_deterministic_algorithms(True, warn_only=True)`` is the pinned recipe.

Compute (PRD §16)
-----------------
GPU runs are launched by a hand-authored ``sbatch`` under ``slurm/`` through the CLAUDE.md
§9.3 submit-ack protocol — **never** the Snakemake SLURM executor, which would bypass the
``--test-only`` preflight and the no-auto-resubmit rule. Hydra drives in-job config; W&B
logs offline on the compute node and is uploaded afterwards by ``setup.smk::wandb_sync`` on
the login node. Any process using this env must export ``CUDA_HOME=$CONDA_PREFIX``
(ADR-0002 A4) or deepspeed raises ``MissingCUDAException`` at import.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from tbox_finder.labels import CLASS_ORDER

# --------------------------------------------------------------------------------------
# Provenance constants (CLAUDE.md §11). Single-sourced here; the conf/ files echo them.
# --------------------------------------------------------------------------------------
SCHEMA_VERSION = "1"
STEP = "P2-04"
GENERATED_BY = "src/tbox_finder/train/train_stage1.py"
ADR = "ADR-0002"
ENV_LOCK = "envs/ml-dna.conda-lock.yml"
CONFIG_PATH = "conf/train/stage1.yaml"

DEFAULT_REPORT = "reports/p2/train_stage1_smoke.json"
DEFAULT_CHECKPOINT_DIR = "checkpoints/p2/stage1"

NUM_CLASSES: int = len(CLASS_ORDER)
IGNORE_INDEX: int = -100

#: Marker set on a block whose ``forward`` we have wrapped, so re-enabling is idempotent
#: (a second wrap would checkpoint the checkpoint — silent double recompute).
_CKPT_MARKER = "_tbox_gradient_checkpointing"

#: Env vars torchrun sets. Absent ⇒ single-process (the local smoke), world_size 1.
_RANK_ENV = "RANK"
_WORLD_SIZE_ENV = "WORLD_SIZE"
_LOCAL_RANK_ENV = "LOCAL_RANK"


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage1TrainConfig:
    """The Stage-1 training knobs.

    **Nothing here is ADR-pinned.** Neither ADR-0002 nor ADR-0005 fixes a Stage-1 learning
    rate, batch size, epoch count, optimiser, scheduler, precision or DDP world size (the
    only ``×8`` in ADR-0002 is DeepSpeed VRAM arithmetic for the *Stage-2* fallback, not a
    Stage-1 DDP pin). Every field below is therefore an **implementer default on a swept
    axis** (P2-06 sweeps γ/LR/window/RC-combine; P2-12 the CRF), recorded as
    ``pinned=False`` in :func:`diagnostics` — the P1-15/P1-16/P2-02 precedent.

    The three values that ARE pinned elsewhere and are *not* configurable here: the backbone
    repo/revision (code-frozen in ``models/caduceus_backbone.py``), the 1024/512 window
    geometry (ADR-0005 D3, code-frozen in ``data/window_dataset.py``), and the non-averaged
    RC-combine constraint (ADR-0005 D15, code-frozen in ``models/rc_combine.py`` — ``mean``
    is rejected at construction, so ``rc_combine`` below cannot weaken it).
    """

    # Determinism (§8.3; ADR-0002 A6).
    seed: int = 42
    pythonhashseed: int = 0

    # Optimisation (implementer defaults; P2-06 sweeps lr).
    epochs: int = 1
    batch_size: int = 4
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Footprint (PRD §10.3). See enable_gradient_checkpointing for why this is hand-wired.
    gradient_checkpointing: bool = True

    # Objective (P2-02 Stage1LossConfig; nothing ADR-pinned — P2-06 γ/α, P2-12 CRF).
    gamma: float = 2.0
    class_weight_alpha: float = 0.0
    use_crf: bool = False
    crf_weight: float = 1.0

    # Model (P1-04; rc_combine constrained non-averaged by ADR-0005 D15, enforced in code).
    rc_combine: str = "concat"
    dropout: float = 0.0

    # Datamodule (the `/data: stage1` group). These MUST be threaded through to
    # Stage1DataConfig rather than left to its defaults: PRD §11 makes window/stride a
    # `--multirun` sweep axis (P2-06), and a sweep whose values never reach the dataset would
    # run every point on the same 1024/512 stream while faithfully reporting different ones.
    # Defaults mirror window_dataset.py, which stays authoritative (ADR-0005 D3 pins 1024/512;
    # the drift guard in test_window_dataset.py asserts config == code).
    window_nt: int = 1024
    stride_nt: int = 512
    offset_augmentation: bool = True
    both_strands: bool = True
    phylum_alpha: float = 0.25
    klass_alpha: float = 0.25
    aa_alpha: float = 0.25

    # Stream shaping. `steps_per_epoch`/`max_records` exist for the smoke: a full epoch is
    # 8,303 records and the smoke must compose in seconds, not hours. None ⇒ the full stream.
    steps_per_epoch: int | None = None
    max_records: int | None = None
    #: ⚠ ACCEPTED BUT NOT APPLIED. `_batches` iterates the sampler directly rather than
    #: through a `DataLoader` (see its docstring), so there is nowhere for worker processes to
    #: go. It is kept because P2-05 may add the loader once throughput is the question — but
    #: until then a `num_workers=8` sweep point would record itself as set while changing
    #: nothing, so it is named here rather than left to be discovered. Flagged at P2-04 review.
    num_workers: int = 0

    # W&B (PRD §16: offline on the node; setup.smk::wandb_sync uploads from the login node).
    wandb_mode: str = "offline"
    wandb_project: str = "tbox-finder"
    wandb_entity: str | None = None
    wandb_dir: str = "wandb"

    # Outputs.
    report_path: str = DEFAULT_REPORT
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    save_checkpoint: bool = False
    device: str | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1; got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}")
        for name in ("lr", "weight_decay", "grad_clip", "gamma", "class_weight_alpha"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a real number; got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0; got {value}")
        if self.wandb_mode not in ("offline", "online", "disabled"):
            raise ValueError(f"wandb_mode must be offline/online/disabled; got {self.wandb_mode!r}")


# --------------------------------------------------------------------------------------
# Gradient checkpointing — hand-wired (PRD §10.3; user sign-off 2026-07-16)
# --------------------------------------------------------------------------------------
def backbone_blocks(backbone: Any) -> Sequence[Any]:
    """The Caduceus-PS Mamba block stack (``backbone.backbone.layers``), or raise.

    Fails loud rather than returning an empty list: a silently-empty stack would make
    :func:`enable_gradient_checkpointing` report ``0 blocks wrapped`` and still look like a
    success, which is the §10.3 failure mode (a stub that certifies itself).
    """
    inner = getattr(backbone, "backbone", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            "expected a Caduceus-PS model exposing `.backbone.layers` (the RCPSMambaBlock "
            f"stack); got {type(backbone).__name__} with no such attribute. The gradient-"
            "checkpointing wiring is bound to that structure — see PRD §10.3."
        )
    if len(layers) == 0:
        raise ValueError("Caduceus-PS block stack is empty; refusing to report a no-op wrap")
    return layers


def is_gradient_checkpointing_enabled(backbone: Any) -> bool:
    """True iff every block in the stack carries our wrap marker."""
    blocks = backbone_blocks(backbone)
    return all(getattr(b, _CKPT_MARKER, False) for b in blocks)


def hf_gradient_checkpointing_supported(backbone: Any) -> bool:
    """Probe whether the backbone supports the **standard HuggingFace** checkpointing path.

    Measured off the model, never asserted. `build_report` previously wrote
    ``hf_flag_supported: False`` as a hard-coded literal and a test asserted it was False —
    a clause compared to the constant that produced it, which is this repo's documented
    tautology class and could never become True however the backbone changed.

    This is the fact that justifies the hand-wiring (PRD §10.3): if a future revision *does*
    implement it, this flips to True and the committed-report test fails, prompting a
    revisit — which is exactly the alarm the hard-coded literal could never raise.
    """
    return bool(getattr(backbone, "supports_gradient_checkpointing", False))


def enable_gradient_checkpointing(backbone: Any) -> int:
    """Hand-wire per-block gradient checkpointing. Returns the number of blocks wrapped.

    **Why hand-wired.** PRD §10.3 pins gradient checkpointing for the Stage-1 Caduceus full
    fine-tune, but the pinned checkpoint cannot honour it through the standard HuggingFace
    path. Measured at P2-04 against revision ``d89eeb85``:

    - ``Caduceus.supports_gradient_checkpointing`` is ``False``;
    - ``backbone.gradient_checkpointing_enable()`` raises
      ``ValueError: Caduceus does not support gradient checkpointing``;
    - the string ``gradient_checkpointing`` appears nowhere in the remote-code class, and
      ``modeling_caduceus.py`` carries a literal ``# TODO: Add support for gradient
      checkpointing`` directly above its layer loop.

    So HF's ``_set_gradient_checkpointing`` machinery has nothing to hook: there is no flag
    to flip. This is the same shape as ADR-0002 A2 C2's finding about the pure-PyTorch
    selective-scan fallback — *"a code change, not a flag"* — and the same resolution: wire
    it by hand. A silent no-op flag would violate §10.3 (it would look exactly like a
    working one), which is why :func:`diagnostics` reports the wrapped-block count rather
    than echoing the request.

    **What it does.** Wraps each ``RCPSMambaBlock.forward`` in
    ``torch.utils.checkpoint(..., use_reentrant=False)``, so the block's activations are
    dropped after the forward and recomputed during the backward. Upstream calls the block
    as ``layer(hidden_states, residual, inference_params=None)`` and it returns
    ``(hidden_states, residual)``; both pass through ``checkpoint`` unchanged. Idempotent —
    re-enabling is a no-op, because double-wrapping would recompute twice for no gain.

    **Measured** (P2-04 sizing smoke; laptop RTX 4060 8 GiB sm_89, 1024 nt, full fine-tune
    + AdamW step): batch 8 peak VRAM **4.424 → 0.961 GiB** (4.6×), gradients finite, loss
    ``1.5939 → 1.5940`` — i.e. the same computation, not a broken wrap. The **throughput
    cost is deliberately unmeasured here**: single-step timings on that box are
    warmup-dominated noise, and P2-05 owns the footprint on the real A4000 (§10.3 — no
    fabricated throughput claim).

    **Caveat, disclosed** (ADR-0002 A7): checkpointing recomputes the forward during the
    backward, and the Mamba ``selective_scan_cuda`` kernel registers no deterministic
    algorithm. Recomputed activations may therefore differ from the originals in the last
    bits, so gradients are taken w.r.t. marginally different activations than the forward
    saw. A7's reproducibility contract is already **metric-level, not bitwise**, so this is
    within the pinned tolerance rather than a new violation — but it is a real interaction
    and P2-05/P2-06 should watch it.

    **Requires trainable inputs.** ``use_reentrant=False`` silently skips checkpointing when
    no input requires grad (e.g. a fully frozen backbone). Stage-1 is a *full* fine-tune
    (ADR-0002 D7 (1) + A6), so the embedding output always requires grad; the frozen-backbone
    probe path (P1-05/P1-09) does not use this function.

    **Two consequences of rebinding ``forward``, latent today but disclosed** (they bite the
    moment someone adds an obvious feature, which P2-05/P2-09 might):

    - ``copy.deepcopy`` treats a plain function as atomic, so a deep-copied block shares this
      closure — whose ``_orig`` is still bound to the **original** block. ``deepcopy(model)``
      would therefore forward through the *original* module's parameters, silently, with
      correct-looking output, while the copy's own parameters never move. Any EMA/SWA/
      best-model snapshot hits this.
    - the wrapped ``forward`` is a local function and therefore **unpicklable**, so a
      whole-module ``torch.save(model)`` (as opposed to ``state_dict()``) or a spawn-based
      worker shipping the module raises ``PicklingError``.

    Neither is live here: the checkpoint path saves ``state_dict()`` only, and DDP does not
    pickle the module. If either becomes needed, wrap via a module subclass rather than by
    rebinding the attribute.
    """
    from torch.utils.checkpoint import checkpoint  # lazy — keeps this module bare-importable

    blocks = backbone_blocks(backbone)
    wrapped = 0
    for block in blocks:
        if getattr(block, _CKPT_MARKER, False):
            continue
        original = block.forward

        def _checkpointed(*args: Any, _orig: Any = original, **kwargs: Any) -> Any:
            return checkpoint(_orig, *args, use_reentrant=False, **kwargs)

        block.forward = _checkpointed
        setattr(block, _CKPT_MARKER, True)
        wrapped += 1
    return wrapped


# --------------------------------------------------------------------------------------
# Determinism preconditions (§8.3; PRD §11)
# --------------------------------------------------------------------------------------
def check_pythonhashseed(expected: int) -> None:
    """Verify ``PYTHONHASHSEED`` was set **by the launcher**; raise if it was not.

    PRD §11 pins *"Explicit seeds everywhere (Hydra config), ``PYTHONHASHSEED``, deterministic
    flags where feasible"*. This function deliberately **verifies rather than sets**, because
    setting it from inside the process is a **no-op that looks like it works**: CPython fixes
    string-hash randomisation while initialising the interpreter, long before this module is
    imported, so ``os.environ["PYTHONHASHSEED"] = "0"`` here changes an env var that nothing
    will ever read again — the run stays as non-deterministic as it was, and the line sits in
    the source as evidence that the requirement was handled. That is the same failure shape as
    a gradient-checkpointing flag that silently no-ops (§10.3): the artifact of compliance
    without the substance.

    So the launcher owns it — the §9.3 sbatch body and any local invocation must export
    ``PYTHONHASHSEED`` **before** python starts. This raises rather than warns because a
    determinism precondition that is merely logged is one nobody reads until a run fails to
    reproduce and the reason is a year old.
    """
    raw = os.environ.get("PYTHONHASHSEED")
    if raw is None:
        raise RuntimeError(
            f"PYTHONHASHSEED is not set. It must be exported BEFORE python starts — CPython "
            f"fixes hash randomisation at interpreter startup, so this process cannot set it "
            f"for itself (§8.3; PRD §11). Re-run as: PYTHONHASHSEED={expected} python -m "
            f"tbox_finder.train.train_stage1 ..."
        )
    if raw != str(expected):
        raise RuntimeError(
            f"PYTHONHASHSEED={raw!r} but the config pins {expected!r}. The inherited value is "
            f"the one in force (it cannot be changed in-process), so the run would not match "
            f"its own recorded config (§8.3)."
        )


# --------------------------------------------------------------------------------------
# DDP (PRD §10.3 "DDP×8 for throughput")
# --------------------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc
    if value < 0:
        raise ValueError(f"{name}={value} must be >= 0")
    return value


def ddp_world_size() -> int:
    """Number of DDP ranks (torchrun's ``WORLD_SIZE``); 1 when unset (the local smoke)."""
    return max(1, _env_int(_WORLD_SIZE_ENV, 1))


def ddp_rank() -> int:
    """This process's global rank; 0 when unset."""
    return _env_int(_RANK_ENV, 0)


def ddp_local_rank() -> int:
    """This process's node-local rank (selects the CUDA device); 0 when unset."""
    return _env_int(_LOCAL_RANK_ENV, 0)


def is_primary() -> bool:
    """True on the one rank that writes artifacts / logs to W&B."""
    return ddp_rank() == 0


class ShardedSampler:
    """A rank-disjoint, **equal-length** view of a :class:`WeightedIndexSampler` stream (DDP).

    Every rank builds the *same* seeded draw stream and takes the ``rank::world_size`` slice,
    so the curriculum weighting (P2-01) is preserved rather than re-derived per rank, and no
    draw is seen twice in an epoch.

    **Every rank must yield exactly the same number of draws, or DDP deadlocks.** The stream
    is therefore truncated to ``(len // world_size) * world_size`` draws *before* striding.
    Without that, a 23-draw stream over 4 ranks shards 6/6/6/5: the short rank runs one fewer
    backward pass, stops joining the gradient all-reduce, and the other three block on a
    collective that can never complete — the job hangs rather than fails, which is the worst
    way for it to go wrong. The cost is dropping at most ``world_size - 1`` draws per epoch
    (standard ``drop_last`` behaviour); which draws are dropped changes every epoch, because
    ``set_epoch`` reshuffles the underlying stream.

    Note the union over ranks is therefore a *subset* of the single-process stream, not equal
    to it. An earlier draft asserted equality — which silently **required** the ragged shards
    that deadlock, i.e. the test encoded the bug as the contract.

    **The tuples are load-bearing.** ``WeightedIndexSampler`` yields ``(index, occurrence)``,
    not bare ints, and the occurrence ordinal is part of the dataset's per-draw RNG key: it
    is what makes a 9× oversampled class-II record emit nine *different* window phases /
    strands instead of nine identical copies. This wrapper passes the tuples through
    untouched. Swapping in ``torch.utils.data.DistributedSampler`` would drop them and
    silently re-create the memorisation P2-01 measured and designed against.
    """

    def __init__(self, sampler: Any, *, rank: int, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1; got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}); got {rank}")
        self._sampler = sampler
        self._rank = rank
        self._world_size = world_size

    def set_epoch(self, epoch: int) -> None:
        """Advance the underlying draw stream (must be called every epoch)."""
        self._sampler.set_epoch(epoch)

    def _usable(self) -> int:
        """Draws kept before striding: the largest multiple of ``world_size`` that fits."""
        return (len(self._sampler) // self._world_size) * self._world_size

    def __len__(self) -> int:
        return self._usable() // self._world_size

    def __iter__(self) -> Iterator[Any]:
        # Truncate globally FIRST, then stride — so every rank gets exactly _usable() //
        # world_size draws. Striding first and truncating after would reintroduce the skew.
        return islice(
            islice(iter(self._sampler), self._usable()), self._rank, None, self._world_size
        )


# --------------------------------------------------------------------------------------
# Class counts — computed from the configured stream, never read from labels_report.json
# --------------------------------------------------------------------------------------
def compute_class_counts(dataset: Any, *, max_records: int | None = None) -> tuple[int, ...]:
    """Per-class **valid-position** counts over the configured window stream (CLASS_ORDER order).

    P2-02 ships no counts file on purpose: the inverse-frequency weights depend on the fold
    and the config, so they must be derived from the stream this run will actually train on.
    Reading ``labels_report.json`` instead would misweight **every** class — its totals are
    locus-only (background 31.5%) whereas the real 1024-nt windowed stream is **78.68%**
    background (P2-02's measurement over the 8,303 ``nested_train`` records).

    Counted at ``occurrence=0`` — **which is not the same as "the deterministic lead"**, and
    an earlier draft of this docstring said it was. With ``offset_augmentation=True`` (the
    default, and what ``build_stream`` configures) ``Stage1WindowDataset.window_at`` takes the
    *sampled* branch: the lead and the strand are drawn from the augmentation RNG keyed on
    ``(seed, epoch, index, occurrence)``. ``deterministic_lead`` is a different function,
    reached only when augmentation is off. So what is actually true is narrower: the counts
    are **reproducible** (the RNG is seeded, and the scan is pinned at ``occurrence=0`` on the
    dataset's *current* epoch) — not phase-independent. Two consequences worth naming rather
    than discovering: the counts depend on ``dataset._epoch``, so they are correct today only
    because :func:`train_stage1` calls this **before** the ``set_epoch`` loop; and they shift
    with ``offset_augmentation`` / ``both_strands``. Class *frequencies* move very little with
    phase (a window's label histogram is nearly phase-invariant), so this remains a faithful
    estimate — the estimate is just not the *deterministic* one the field name suggests.

    **Scope, stated (it is not the weighted draw stream).** This scans each record **once**.
    It is *not* a tally over the draws :class:`WeightedIndexSampler` actually emits, which
    oversample rare strata (class II ≈ 9× at α=0.25). So when ``class_weight_alpha > 0``, the
    inverse-frequency weights would be derived from the **unweighted** record distribution
    while the model trains on the **oversampled** one — the two do not describe the same
    population. With ``class_weight_alpha = 0`` (the default, and what P2-04 ships) nothing
    consumes these counts and the discrepancy is inert; it becomes live at **P2-06**, which
    owns the α sweep and must decide whether the weights should describe the draw stream
    (and, if so, which epoch's draws — they reshuffle). Recorded in the report as
    ``class_counts_scope.weighted_draw_stream = false`` rather than left for someone to
    discover from the arithmetic. Flagged by CodeRabbit at P2-04 review.

    ``IGNORE_INDEX`` positions (pad-only, carrying no DNA) are excluded — they take no loss.
    The result feeds ``Stage1Loss(class_counts=…)`` directly.
    """
    import numpy as np  # lazy

    total = len(dataset)
    n = total if max_records is None else min(total, max_records)
    if n < 1:
        raise ValueError("cannot compute class counts over an empty stream")

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for index in range(n):
        labels = np.asarray(dataset.window_at(index, 0).labels)
        valid = labels[labels != IGNORE_INDEX]
        if valid.size == 0:
            continue
        lo, hi = int(valid.min()), int(valid.max())
        if lo < 0 or hi >= NUM_CLASSES:
            raise ValueError(
                f"record {index}: label index out of range [0, {NUM_CLASSES}); got [{lo}, {hi}]"
            )
        counts += np.bincount(valid, minlength=NUM_CLASSES)
    if not counts.any():
        raise ValueError("class counts are all zero over the configured stream")
    return tuple(int(c) for c in counts)


# --------------------------------------------------------------------------------------
# Provenance (CLAUDE.md §11)
# --------------------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _git_sha() -> str | None:
    sha = (_git("rev-parse", "HEAD") or "").strip()
    return sha or None


def _git_dirty() -> bool | None:
    """True iff tracked files differ from HEAD (``None`` if git is unavailable).

    Recorded because a bare SHA can be actively misleading: a report generated from an
    uncommitted tree names a commit that does **not** contain the code that produced it.
    ``git_dirty=true`` says so out loud instead of letting the SHA imply otherwise (§11).
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status.strip())


def _env_lock_sha256(path: str | Path = ENV_LOCK) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sanitize(obj: Any) -> Any:
    """JSON-safe echo of a config/mapping (dataclasses, Paths, numpy scalars)."""
    import math

    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    return str(obj)


def diagnostics(cfg: Stage1TrainConfig) -> dict[str, Any]:
    """What this run pins vs chooses — the honesty block the report carries.

    ``pinned=False`` because no ADR fixes any Stage-1 training hyperparameter (verified
    across ADR-0002 D1-D9/A1-A10 and ADR-0005 D1-D18/A1-A3 at P2-04). The P1-15/P1-16/P2-02
    precedent: record the absence rather than implying authority the ADRs do not grant.
    """
    return {
        "step": STEP,
        "config": _sanitize(asdict(cfg)),
        "pinned": False,
        "swept_by": {
            "gamma": "P2-06",
            "lr": "P2-06",
            "class_weight_alpha": "P2-06",
            "rc_combine": "P2-12",
            "use_crf": "P2-12",
        },
        "class_order": list(CLASS_ORDER),
    }


# --------------------------------------------------------------------------------------
# Report + fail-closed validator (the P1-15/P1-16 "re-derive, never echo" contract)
# --------------------------------------------------------------------------------------
def _finite_number(v: Any) -> bool:
    import math

    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _pos_int(v: Any) -> bool:
    """True iff ``v`` is a positive int. Rejects bools — ``isinstance(True, int)`` is True."""
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _non_neg_int(v: Any) -> bool:
    """True iff ``v`` is a non-negative int (seed 0 is legal). Rejects bools."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """Re-derive every gate clause **from the report's recorded evidence**.

    Shared by :func:`build_report` and :func:`validate_report` so the two cannot disagree.
    This is the P1-15/P1-16 lesson, which bit twice: ``overall_pass = all(clauses)`` catches
    a clause flipped FALSE but **structurally cannot** catch one fabricated TRUE — an
    all-true gate is self-consistent whatever the evidence says. So no clause is ever echoed
    from the builder; each is recomputed here from the evidence blocks, and
    :func:`validate_report` requires the stored clauses to equal these.
    """
    ckpt = report.get("gradient_checkpointing")
    ckpt = ckpt if isinstance(ckpt, Mapping) else {}
    steps = report.get("steps")
    steps = steps if isinstance(steps, Mapping) else {}
    counts = report.get("class_counts")
    prov = report.get("provenance")
    prov = prov if isinstance(prov, Mapping) else {}

    requested = ckpt.get("requested")
    n_blocks = ckpt.get("n_blocks")
    n_wrapped = ckpt.get("n_blocks_wrapped")
    # The checkpointing clause is the whole point of the hand-wiring: it holds only if EVERY
    # block in the stack was actually wrapped when checkpointing was requested. A no-op wrap
    # (0 blocks) must NOT satisfy it — that is exactly the stub §10.3 forbids. A missing or
    # non-bool `requested` fails closed rather than defaulting either way.
    if requested is False:
        ckpt_ok = True  # not requested ⇒ vacuously satisfied
    elif requested is True:
        ckpt_ok = _pos_int(n_blocks) and n_wrapped == n_blocks
    else:
        ckpt_ok = False

    losses = steps.get("losses")
    losses_ok = (
        isinstance(losses, list) and len(losses) > 0 and all(_finite_number(x) for x in losses)
    )

    # The counts must be well-shaped AND carry a self-consistent provenance scope. The scope
    # is the §10.2 honesty field — it is what tells a reader a 64-record slice from the 8,303
    # -record fold — so it is re-derived like any other clause rather than trusted: `full_stream`
    # must FOLLOW from the record arithmetic, not be asserted alongside it.
    scope = report.get("class_counts_scope")
    scope = scope if isinstance(scope, Mapping) else {}
    n_records = scope.get("n_records")
    n_fold = scope.get("n_training_fold_records")
    scope_ok = (
        _pos_int(n_records)
        and _pos_int(n_fold)
        and n_records <= n_fold
        and scope.get("full_stream") is (n_records == n_fold)
        and scope.get("training_fold_only") is True
        and _non_neg_int(scope.get("occurrence"))
    )
    counts_ok = (
        isinstance(counts, list)
        and len(counts) == NUM_CLASSES
        and all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in counts)
        and any(c > 0 for c in counts)
        and scope_ok
    )

    return {
        "gradient_checkpointing_applied": bool(ckpt_ok),
        "train_step_ran": bool(_pos_int(steps.get("n_steps")) and losses_ok),
        "loss_finite": bool(losses_ok),
        "class_counts_from_stream": bool(counts_ok),
        "grads_finite": report.get("grads_finite") is True,
        "provenance_complete": _provenance_complete(report),
    }


def _provenance_complete(report: Mapping[str, Any]) -> bool:
    """CLAUDE.md §11: a run without its git SHA + env-lock hash + seed is not reproducible.

    Three traps this closes, each of which passed an earlier draft:

    - ``a and b and c or d`` binds as ``(a and b and c) or d`` — the first form certified
      provenance on ``seed == 0`` alone, whatever the SHA said.
    - ``isinstance("", str)`` is **True**, so an empty ``git_sha`` satisfied a bare type
      check. A blank SHA is a *missing* SHA wearing the right type.
    - the seed was recorded twice (``provenance.seed`` and ``diagnostics.config.seed``) and
      never cross-checked, so the two could disagree and the report would still certify —
      the P1-15/P1-16 duplicated-``peak_vram_gib`` lesson.
    """
    prov = report.get("provenance")
    prov = prov if isinstance(prov, Mapping) else {}
    diag = report.get("diagnostics")
    diag = diag if isinstance(diag, Mapping) else {}
    config = diag.get("config")
    config = config if isinstance(config, Mapping) else {}

    def _nonempty_str(v: Any) -> bool:
        return isinstance(v, str) and bool(v.strip())

    seed = prov.get("seed")
    # The seed must be present in BOTH places and agree; a missing config seed voids it
    # (fail-closed) rather than vacuously matching.
    seed_ok = _non_neg_int(seed) and "seed" in config and config.get("seed") == seed
    return bool(
        _nonempty_str(prov.get("git_sha"))
        and _nonempty_str(prov.get("env_lock_sha256"))
        and seed_ok
    )


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of problems with a P2-04 smoke report; empty ⇒ valid. Fails closed.

    Every structural floor is required *and* every clause re-derived: a report that omits an
    evidence block, or whose stored clause disagrees with the re-derivation, is invalid.
    Total by construction — it never raises on a malformed report, it reports.
    """
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return [f"report must be a mapping, got {type(report).__name__}"]

    for key in ("schema_version", "step", "generated_by", "adr", "env_lock"):
        if not isinstance(report.get(key), str) or not report.get(key):
            problems.append(f"{key}: missing or not a non-empty string")
    if report.get("step") != STEP:
        problems.append(f"step: expected {STEP!r}, got {report.get('step')!r}")

    # The honesty invariants (§10.3): a composition smoke is never a science result.
    if report.get("is_science") is not False:
        problems.append("is_science: must be exactly False (a composition smoke, not a result)")
    if report.get("gate4_graded") is not False:
        problems.append("gate4_graded: must be exactly False (GATE-4 is P2-14, on the real split)")

    for block in ("gradient_checkpointing", "steps", "provenance", "diagnostics", "backbone"):
        if not isinstance(report.get(block), Mapping):
            problems.append(f"{block}: missing or not a mapping")

    counts = report.get("class_counts")
    if not isinstance(counts, list) or len(counts) != NUM_CLASSES:
        problems.append(f"class_counts: must be a list of {NUM_CLASSES} ints")

    derived = derive_clauses(report)
    stored = report.get("gate")
    if not isinstance(stored, Mapping):
        problems.append("gate: missing or not a mapping")
    else:
        for name, value in derived.items():
            if name not in stored:
                problems.append(f"gate.{name}: missing")
            elif stored[name] is not value:
                problems.append(
                    f"gate.{name}: stored {stored[name]!r} != re-derived {value!r} "
                    "(a clause must follow from the recorded evidence, never be asserted)"
                )
        expected_overall = all(derived.values())
        if stored.get("overall_pass") is not expected_overall:
            problems.append(
                f"gate.overall_pass: stored {stored.get('overall_pass')!r} != "
                f"re-derived {expected_overall!r}"
            )
    return problems


def build_report(
    *,
    cfg: Stage1TrainConfig,
    class_counts: Sequence[int],
    counts_scope: Mapping[str, Any],
    hardware: Mapping[str, Any] | None = None,
    n_blocks: int,
    n_blocks_wrapped: int,
    hf_flag_supported: bool,
    losses: Sequence[float],
    grads_finite: bool,
    world_size: int,
    wandb_run_id: str | None,
    peak_vram_gib: float | None = None,
    eval_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the P2-04 smoke report. Clauses are **re-derived**, never asserted."""
    from tbox_finder.models.caduceus_backbone import REPO_ID, REVISION

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "env_lock": ENV_LOCK,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        # §10.3: this is a composition smoke on a seeded stream. It establishes that the
        # entrypoint RUNS, not that the model LEARNS. GATE-4 is P2-14, on the real split.
        "is_science": False,
        "gate4_graded": False,
        "backbone": {"repo_id": REPO_ID, "revision": REVISION},
        "gradient_checkpointing": {
            "requested": bool(cfg.gradient_checkpointing),
            "n_blocks": int(n_blocks),
            "n_blocks_wrapped": int(n_blocks_wrapped),
            # MEASURED off the model (hf_gradient_checkpointing_supported), not asserted:
            # this is the fact that justifies the hand-wiring, so a hard-coded False would be
            # a clause compared to the constant that produced it.
            "hf_flag_supported": bool(hf_flag_supported),
            "mechanism": "manual torch.utils.checkpoint per RCPSMambaBlock (use_reentrant=False)",
        },
        "steps": {
            "n_steps": len(losses),
            "losses": [float(x) for x in losses],
            "world_size": int(world_size),
        },
        "class_counts": [int(c) for c in class_counts],
        # §10.2 claim accuracy: a smoke run slices the fold (`max_records`), so these counts
        # describe THAT SLICE, not the corpus. Stating the scope is not pedantry — P2-03
        # shipped sample properties worded as corpus facts and no test caught it. A consumer
        # must be able to tell a 64-record slice from the 8,303-record `nested_train` fold.
        "class_counts_scope": _sanitize(counts_scope),
        "grads_finite": bool(grads_finite),
        "peak_vram_gib": float(peak_vram_gib) if peak_vram_gib is not None else None,
        # The card `peak_vram_gib` was measured on. Without it the number is unattributable —
        # and P2-04's laptop RTX 4060 is not the A4000 the PRD §10.3 budget is about.
        "hardware": _sanitize(hardware) if hardware else None,
        "eval_metrics": _sanitize(eval_metrics) if eval_metrics else None,
        "provenance": {
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "env_lock_sha256": _env_lock_sha256(),
            "seed": int(cfg.seed),
            "config_path": CONFIG_PATH,
            "wandb_run_id": wandb_run_id,
        },
        "diagnostics": diagnostics(cfg),
    }
    clauses = derive_clauses(report)
    report["gate"] = {**clauses, "overall_pass": all(clauses.values())}
    return report


# --------------------------------------------------------------------------------------
# Build + run
# --------------------------------------------------------------------------------------
def build_model(cfg: Stage1TrainConfig, *, device: str) -> tuple[Any, int, int, bool]:
    """Load the pinned backbone + P1-04 segmenter → (segmenter, blocks, wrapped, hf_supported).

    ``load_caduceus_ps`` returns the backbone in ``.eval()``; a full fine-tune needs
    ``.train()``, so the segmenter is switched explicitly — otherwise dropout and any other
    train-mode module keep their inference behaviour, a silent no-training trap.
    """
    from tbox_finder.models.caduceus_backbone import load_caduceus_ps
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter

    backbone = load_caduceus_ps(device=device)  # pinned revision; rejects any other
    segmenter = Stage1Segmenter(
        backbone=backbone,
        rc_combine=cfg.rc_combine,  # ADR-0005 D15: "mean" is rejected in rc_combine.py
        use_crf=cfg.use_crf,
        dropout=cfg.dropout,
    ).to(device)
    segmenter.train()

    n_blocks = len(backbone_blocks(backbone))
    # Probe BEFORE wrapping — the wrap does not touch the HF flag, but reading it first keeps
    # the measurement about the pristine backbone rather than our mutation of it.
    hf_supported = hf_gradient_checkpointing_supported(backbone)
    n_wrapped = enable_gradient_checkpointing(backbone) if cfg.gradient_checkpointing else 0
    return segmenter, n_blocks, n_wrapped, hf_supported


def _init_wandb(cfg: Stage1TrainConfig) -> Any:
    """Init W&B on the primary rank only. Offline by default (PRD §16). Never fails the run."""
    if not is_primary() or cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb  # lazy
    except ImportError:
        return None
    # Belt-and-braces: the sbatch also exports this. Compute nodes have no reliable outbound,
    # so an accidental online init would block the run on a network call.
    os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)
    Path(cfg.wandb_dir).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        mode=cfg.wandb_mode,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        dir=cfg.wandb_dir,
        job_type="train",
        group=STEP,
        config=_sanitize(asdict(cfg)),
    )


def build_stream(cfg: Stage1TrainConfig) -> tuple[Any, Any, dict[str, Any]]:
    """Build the P2-01 dataset + curriculum sampler. Returns (dataset, sampler, counts_scope).

    The dataset config is single-sourced from :class:`Stage1DataConfig`'s own defaults for the
    pinned geometry (1024/512, ADR-0005 D3) — this entrypoint does not restate them, so it
    cannot drift from ``window_dataset.py``, which is authoritative.
    """
    from tbox_finder.data.window_dataset import (
        Stage1DataConfig,
        Stage1WindowDataset,
        load_corpus_records,
    )

    records, _ = load_corpus_records(training_fold_only=True, window=cfg.window_nt)
    n_fold = len(records)
    if cfg.max_records is not None:
        records = records[: cfg.max_records]
    data_config = Stage1DataConfig(
        window_nt=cfg.window_nt,
        stride_nt=cfg.stride_nt,
        seed=cfg.seed,
        offset_augmentation=cfg.offset_augmentation,
        both_strands=cfg.both_strands,
        phylum_alpha=cfg.phylum_alpha,
        klass_alpha=cfg.klass_alpha,
        aa_alpha=cfg.aa_alpha,
    )
    dataset = Stage1WindowDataset(records, config=data_config)
    sampler = ShardedSampler(dataset.sampler(), rank=ddp_rank(), world_size=ddp_world_size())
    scope = {
        "n_records": len(records),
        "n_training_fold_records": n_fold,
        "full_stream": len(records) == n_fold,
        "training_fold_only": True,
        # The occurrence the scan is pinned at. NOT "the deterministic lead": with
        # offset_augmentation on, window_at draws the phase/strand from the seeded
        # augmentation RNG, so this is reproducible-at-epoch-0, not phase-independent.
        "occurrence": 0,
        "counted_at_epoch": 0,
        "offset_augmentation": cfg.offset_augmentation,
        # The counts are a per-record scan at the deterministic lead, NOT a tally over the
        # weighted draw stream the sampler actually emits. With class_weight_alpha == 0 (the
        # default) nothing consumes them, so this is inert; but P2-06 sweeps α, and inverse-
        # frequency weights derived from the *unweighted* record scan would not describe the
        # *oversampled* distribution the model sees. Recorded rather than silently assumed —
        # P2-06 owns the resolution. See compute_class_counts.
        "weighted_draw_stream": False,
        "data_config": asdict(data_config),
    }
    return dataset, sampler, scope


def _batches(dataset: Any, sampler: Any, cfg: Stage1TrainConfig) -> Iterator[dict[str, Any]]:
    """Yield collated batches from the sampler's draw stream.

    Deliberately not a ``torch.utils.data.DataLoader``: P2-01's sampler yields
    ``(index, occurrence)`` tuples and its ``collate_windows`` is already the collate fn, so
    a DataLoader would add process machinery for no benefit at ``num_workers=0`` while making
    the tuple contract easier to break. P2-05 revisits this if the loader becomes the
    bottleneck on the A4000 (it is a throughput question, and P2-05 owns throughput).
    """
    from tbox_finder.data.window_dataset import collate_windows

    keys = list(iter(sampler))
    n_steps = len(keys) // cfg.batch_size
    if cfg.steps_per_epoch is not None:
        n_steps = min(n_steps, cfg.steps_per_epoch)
    for step in range(n_steps):
        chunk = keys[step * cfg.batch_size : (step + 1) * cfg.batch_size]
        yield collate_windows([dataset[k] for k in chunk])


def train_stage1(cfg: Stage1TrainConfig, *, log: Any = print) -> dict[str, Any]:
    """Run the Stage-1 full fine-tune. Returns the validated report.

    The loop itself is deliberately plain (AdamW, focal CE, grad-clip, DDP): every
    interesting decision already lives in the three composed modules, and P2-05/P2-06 tune
    the knobs. Its job here is to prove the composition runs and to record provenance.

    This wrapper owns only the determinism preconditions and the DDP process-group lifecycle;
    the body is :func:`_train_stage1_inner`, split out so teardown can live in a ``finally``.
    """
    import torch

    from tbox_finder.train.repro import set_determinism

    check_pythonhashseed(cfg.pythonhashseed)  # §8.3 — verified, NOT set (see the function)
    set_determinism(cfg.seed)

    world_size = ddp_world_size()
    ddp_active = world_size > 1
    if ddp_active:
        import torch.distributed as dist

        # NCCL is the A4000 path; the §9.3 sbatch launches this under torchrun, which sets
        # RANK/WORLD_SIZE/LOCAL_RANK and the rendezvous env this reads.
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(ddp_local_rank())
    try:
        return _train_stage1_inner(cfg, log=log, world_size=world_size, ddp_active=ddp_active)
    finally:
        # Reachable on ANY exit path, including a raise. Without this, a rank that dies (an
        # OOM, a collate error, a failed gate) unwinds without destroying the process group,
        # and the surviving ranks block in the next NCCL collective until the watchdog fires
        # — burning the SLURM wall-clock instead of failing fast. Same shape as the ragged-
        # shard deadlock, different trigger.
        if ddp_active:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


def _train_stage1_inner(
    cfg: Stage1TrainConfig, *, log: Any, world_size: int, ddp_active: bool
) -> dict[str, Any]:
    """The body of :func:`train_stage1`, split out so DDP teardown can be a ``finally``."""
    import torch

    from tbox_finder.train.objective import Stage1Loss, Stage1LossConfig

    device = cfg.device or (f"cuda:{ddp_local_rank()}" if torch.cuda.is_available() else "cpu")
    dataset, sampler, counts_scope = build_stream(cfg)

    # The loss weights must come from the stream this run trains on (P2-02 ships no counts
    # file precisely so this cannot be read from a stale/locus-only table). Computed once,
    # before the model, so a counts failure costs no GPU time. The dataset is already sliced
    # to `max_records` by build_stream, so the counts cover exactly the configured stream.
    class_counts = compute_class_counts(dataset)
    by_class = dict(zip(CLASS_ORDER, class_counts, strict=True))
    log(f"class counts over {len(dataset)} records: {by_class}")

    segmenter, n_blocks, n_wrapped, hf_supported = build_model(cfg, device=device)
    log(f"gradient checkpointing: wrapped {n_wrapped}/{n_blocks} RCPSMambaBlocks")

    model = segmenter
    if ddp_active:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(segmenter, device_ids=[ddp_local_rank()])

    loss_fn = Stage1Loss(
        Stage1LossConfig(
            gamma=cfg.gamma,
            class_weight_alpha=cfg.class_weight_alpha,
            use_crf=cfg.use_crf,
            crf_weight=cfg.crf_weight,
        ),
        class_counts=class_counts if cfg.class_weight_alpha > 0 else None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    run = _init_wandb(cfg)

    # Device-scoped, not global: `reset_peak_memory_stats()`/`max_memory_allocated()` default
    # to torch.cuda.current_device(), but `set_device` is only called on the DDP path — so a
    # single-process run with `device=cuda:1` (an exposed knob; the obvious way to pick a free
    # GPU) would train on cuda:1 and report cuda:0's peak, i.e. ~0.0 GiB, in the very field
    # the PRD §10.3 footprint claim rests on. Self-consistently, so the validator would pass
    # it. `measuring_vram` is False for a CPU run even on a CUDA box, so a CPU run records
    # None rather than a fabricated 0.0. (lora_harness.py + eval/rinalmo_throughput.py already
    # pass the device explicitly; this module was the outlier.)
    measuring_vram = torch.cuda.is_available() and str(device).startswith("cuda")
    if measuring_vram:
        torch.cuda.reset_peak_memory_stats(device)

    losses: list[float] = []
    grads_finite = True
    for epoch in range(cfg.epochs):
        dataset.set_epoch(epoch)  # both must advance, or augmentation/draws freeze (P2-01)
        sampler.set_epoch(epoch)
        for batch in _batches(dataset, sampler, cfg):
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            real_mask = batch["real_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids)  # (B, L, 8)
            loss = loss_fn(
                logits,
                targets,
                crf=segmenter.head.crf if cfg.use_crf else None,
                real_mask=real_mask if cfg.use_crf else None,
            )
            loss.backward()
            # Measured, not assumed: a frozen-base + checkpointing combination silently
            # trains nothing, and a NaN grad floods every parameter while the forward stays
            # finite (the P2-02 focal-γ lesson). Cheap enough at smoke scale to always check.
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            losses.append(float(loss.detach().item()))
            if run is not None:
                run.log({"train/loss": losses[-1], "epoch": epoch})

    peak = torch.cuda.max_memory_allocated(device) / 2**30 if measuring_vram else None
    # A VRAM number is meaningless without the card it was measured on — the sibling P1
    # reports (lora_vram_smoke, kernel_smoke, attention_backend) all carry device_name, and
    # this one did not, leaving `peak_vram_gib` unattributable. The laptop and the A4000 are
    # different cards with different capacities; a reader comparing two runs must be able to
    # see whether they are comparable at all.
    hardware = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if measuring_vram else None,
        "capability": list(torch.cuda.get_device_capability(device)) if measuring_vram else None,
        "total_vram_gib": (
            torch.cuda.get_device_properties(device).total_memory / 2**30
            if measuring_vram
            else None
        ),
        "torch": torch.__version__,
    }
    report = build_report(
        hardware=hardware,
        cfg=cfg,
        class_counts=class_counts,
        counts_scope=counts_scope,
        n_blocks=n_blocks,
        n_blocks_wrapped=n_wrapped,
        hf_flag_supported=hf_supported,
        losses=losses,
        grads_finite=grads_finite,
        world_size=world_size,
        wandb_run_id=getattr(run, "id", None),
        peak_vram_gib=peak,
    )
    problems = validate_report(report)
    if problems:
        raise ValueError("P2-04 smoke report failed its own validator:\n  " + "\n  ".join(problems))
    if is_primary():
        out = Path(cfg.report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        log(f"wrote {out}")
        if cfg.save_checkpoint and report["gate"]["overall_pass"]:
            ckpt_dir = Path(cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(segmenter.state_dict(), ckpt_dir / "stage1.pt")
    if run is not None:
        run.finish()

    # A *valid* report can still record a *failed* run: the validator only checks that the
    # clauses follow from the evidence, so a run that trained zero steps yields a perfectly
    # consistent report saying `overall_pass: false`. Without this, that run would exit 0 —
    # the sbatch would look successful and §9.3's artifact-based verification (a DONE marker
    # plus a zero-byte `.err`) would pass it, because the failure lives only in a JSON field
    # nobody re-reads. `batch_size` exceeding the per-rank draw count reaches exactly that in
    # one step. The report is written first (it is the evidence), then we fail loud: the
    # process exit code must mean what the gate means (§10.3).
    if not report["gate"]["overall_pass"]:
        failed = [k for k, v in report["gate"].items() if k != "overall_pass" and not v]
        raise RuntimeError(
            f"P2-04 smoke gate FAILED (clauses: {', '.join(failed) or 'unknown'}). "
            f"n_steps={report['steps']['n_steps']}, world_size={world_size}, "
            f"batch_size={cfg.batch_size}. The report was written to {cfg.report_path} for "
            f"inspection, but the run did not pass its own gate — a failed run must not exit 0."
        )
    return report


def _cfg_from_mapping(cfg: Mapping[str, Any]) -> Stage1TrainConfig:
    """Build a :class:`Stage1TrainConfig` from a resolved Hydra config.

    Hydra groups land as nested blocks (``optim.lr``, ``tracking.mode``, ``model.rc_combine``),
    so they are flattened here into the flat dataclass. Unknown keys are ignored rather than
    raising: the composed config legitimately carries group metadata (``data.*``, ``model.*``)
    this trainer reads through the authoritative modules, not through the config.
    """
    optim = cfg.get("optim") or {}
    tracking = cfg.get("tracking") or {}
    data = cfg.get("data") or {}
    model = cfg.get("model") or {}
    rc = (model.get("rc_combine") or {}) if isinstance(model, Mapping) else {}

    fields: dict[str, Any] = {}
    for key in Stage1TrainConfig.__dataclass_fields__:
        if key in cfg:
            fields[key] = cfg[key]
    for src, key, dst in (
        (optim, "lr", "lr"),
        (optim, "weight_decay", "weight_decay"),
        (tracking, "mode", "wandb_mode"),
        (tracking, "project", "wandb_project"),
        (tracking, "entity", "wandb_entity"),
        (tracking, "dir", "wandb_dir"),
        (rc, "mode", "rc_combine"),
        # The /data group must reach the datamodule, not just the composed config: PRD §11
        # sweeps window/stride, and a sweep whose values stop at the config object trains
        # every point on the same stream while reporting different ones.
        (data, "window_nt", "window_nt"),
        (data, "stride_nt", "stride_nt"),
        (data, "offset_augmentation", "offset_augmentation"),
        (data, "both_strands", "both_strands"),
        (data, "phylum_alpha", "phylum_alpha"),
        (data, "klass_alpha", "klass_alpha"),
        (data, "aa_alpha", "aa_alpha"),
    ):
        if isinstance(src, Mapping) and key in src:
            fields[dst] = src[key]
    return Stage1TrainConfig(**fields)


def main() -> None:
    """Hydra entry: ``python -m tbox_finder.train.train_stage1``.

    The primary config lives at ``conf/train/stage1.yaml`` and is selected by its *slashed*
    group path — a ``conf/<group>/`` primary needs ``@package _global_`` plus leading-slash
    ``/group`` defaults to reach its sibling groups, or composition silently no-ops (the
    P1-07 pattern this mirrors).
    """
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../../../conf", config_name="train/stage1")
    def _entry(cfg: Any) -> None:
        train_stage1(_cfg_from_mapping(OmegaConf.to_container(cfg, resolve=True)))

    _entry()


if __name__ == "__main__":  # pragma: no cover
    main()
