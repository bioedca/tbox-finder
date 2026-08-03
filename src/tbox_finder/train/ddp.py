"""Launcher-owned run environment: torchrun rank discovery + the ``PYTHONHASHSEED`` check.

Promoted out of :mod:`tbox_finder.train.train_stage1` at P3-06 so the Stage-2 trainer can
use the *same* helpers rather than a second copy. A forked copy is a correctness problem,
not a style one: the two trainers run in **different conda envs** (``ml-dna`` for Stage-1,
``ml-rna`` for Stage-2 — ADR-0002 A4), so a bug fixed in one copy would keep shipping in
the other with nothing failing. ``train_stage1`` now imports these names instead of
defining them, so "the two agree" is a fact about one implementation.

Nothing here imports ``torch``. That is what makes the module importable from *both* env
closures — and from the bare CI env, where the P3-06 unit tier exercises it. Rank
discovery is env-var reading; ``torch.distributed`` initialisation is the caller's job.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from itertools import islice
from typing import Any

__all__ = [
    "LOCAL_RANK_ENV",
    "PYTHONHASHSEED_ENV",
    "RANK_ENV",
    "WORLD_SIZE_ENV",
    "ShardedSampler",
    "check_pythonhashseed",
    "ddp_local_rank",
    "ddp_rank",
    "ddp_world_size",
    "is_primary",
]

#: torchrun's per-process environment (PRD §10.3 "DDP×8 for throughput").
RANK_ENV = "RANK"
WORLD_SIZE_ENV = "WORLD_SIZE"
LOCAL_RANK_ENV = "LOCAL_RANK"

#: The determinism variable CPython freezes at interpreter startup (CLAUDE.md §8.3).
PYTHONHASHSEED_ENV = "PYTHONHASHSEED"


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
    """Number of DDP ranks (torchrun's ``WORLD_SIZE``); 1 when unset (the local smoke).

    An explicit ``WORLD_SIZE=0`` RAISES rather than being promoted to 1: unset means "not
    under torchrun", but zero means a launcher computed a world size and got it wrong, and
    silently training single-process on 1/N of the data would look exactly like success.
    """
    value = _env_int(WORLD_SIZE_ENV, 1)
    if value == 0:
        raise ValueError(
            f"{WORLD_SIZE_ENV}=0 is invalid; world size must be >= 1. Unset it for a "
            "single-process run — 0 means a launcher miscomputed it."
        )
    return value


def ddp_rank() -> int:
    """This process's global rank; 0 when unset."""
    return _env_int(RANK_ENV, 0)


def ddp_local_rank() -> int:
    """This process's node-local rank (selects the CUDA device); 0 when unset."""
    return _env_int(LOCAL_RANK_ENV, 0)


def is_primary() -> bool:
    """True on the one rank that writes artifacts / logs to W&B."""
    return ddp_rank() == 0


def check_pythonhashseed(expected: int, *, entrypoint: str) -> None:
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

    Args:
        expected: the value the run's config pins.
        entrypoint: the ``python -m`` target quoted back in the failure message, so the fix
            line is the caller's own and not another stage's.
    """
    raw = os.environ.get(PYTHONHASHSEED_ENV)
    if raw is None:
        raise RuntimeError(
            f"PYTHONHASHSEED is not set. It must be exported BEFORE python starts — CPython "
            f"fixes hash randomisation at interpreter startup, so this process cannot set it "
            f"for itself (§8.3; PRD §11). Re-run as: PYTHONHASHSEED={expected} python -m "
            f"{entrypoint} ..."
        )
    if raw != str(expected):
        raise RuntimeError(
            f"PYTHONHASHSEED={raw!r} but the config pins {expected!r}. The inherited value is "
            f"the one in force (it cannot be changed in-process), so the run would not match "
            f"its own recorded config (§8.3)."
        )


class ShardedSampler:
    """A rank-disjoint, **equal-length** view of a draw stream (DDP).

    Every rank builds the *same* seeded draw stream and takes the ``rank::world_size`` slice,
    so any curriculum weighting the inner sampler applies (Stage-1's ``WeightedIndexSampler``,
    P2-01) is preserved rather than re-derived per rank, and no draw is seen twice in an epoch.

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

    **Whatever the inner sampler yields is passed through untouched.** Stage-1's
    ``WeightedIndexSampler`` yields ``(index, occurrence)`` tuples, not bare ints, and the
    occurrence ordinal is part of that dataset's per-draw RNG key: it is what makes a 9×
    oversampled class-II record emit nine *different* window phases / strands instead of nine
    identical copies. Swapping in ``torch.utils.data.DistributedSampler`` would drop the tuple
    and silently re-create the memorisation P2-01 measured and designed against. Stage-2's
    stream yields bare ints; this wrapper does not care which, and that is the point — it
    shards, it does not interpret.
    """

    def __init__(self, sampler: Any, *, rank: int, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1; got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}); got {rank}")
        self._sampler = sampler
        self._rank = rank
        self._world_size = world_size

    @property
    def inner(self) -> Any:
        """The wrapped sampler — the object that knows the draw stream's composition.

        Exposed so the P2-10d negative-mix measurement reads it off the sampler that
        *built* the stream instead of re-deriving the positive/negative boundary from a
        second source that could drift.
        """
        return self._sampler

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
