"""P3-09 — the **shared block resampler** (PRD §12; ADR-0005 D5).

Every confidence interval this project reports is resampled at the **block** level —
a homology cluster or a held-out taxonomic order — and **never** at the record level.
Records inside one block are not exchangeable: they are homologs, so a record-level
bootstrap would treat correlated copies as independent draws and return an interval far
narrower than the data support. The headline generalization claim is an interval claim,
so that distinction is load-bearing (PRD §2.3, §12).

**This module is the one implementation.** imp.md's roadmap names
``eval/resample.py::block_bootstrap`` as the shared resampler "built once (first gated
CI), imported by P4 GATE-1, P5 FDR CI, P6". The arithmetic shipped at P0-31 as
``metrics.block_bootstrap_ci``; P3-09 **moves** it here — its roadmap home — and leaves
``metrics.block_bootstrap_ci`` as a one-line delegation so the ~10 existing call sites
keep working against a single implementation
([[promote-dont-duplicate-is-a-correctness-rule]]: a forked resampler means fixing the
CI in one place and shipping the old one everywhere else). Because that delegation makes
any "metrics agrees with resample" test a tautology, the guards here are **hand-computed**
expectations and sabotage-checked refusals, not cross-module agreement.

Two things live here:

``block_bootstrap``
    The seeded percentile CI. Draws ``len(blocks)`` whole blocks with replacement.
``blocks_by_key``
    Grouping records into blocks *by a named split-table column*, with the record-level
    columns refused by name. Before P3-09 every caller rolled its own ``{cluster: [...]}``
    dict, so "resamples at block granularity" was a property of each call site rather
    than of the resampler — nothing in the repo could refuse a record-keyed grouping.
    Here the key name is checked against a closed allowlist and the record-identifier
    columns raise, so a record-level bootstrap cannot be assembled by accident.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from tbox_finder import masking

__all__ = [
    "BLOCK_GRANULARITY_COLUMNS",
    "DEFAULT_N_BOOT",
    "RECORD_LEVEL_COLUMNS",
    "block_bootstrap",
    "blocks_by_key",
    "percentile",
]

#: Bootstrap replicates for a block-level CI. Matches the P0-31 value the ~10 existing
#: call sites were written against, so moving the implementation changes no committed
#: number.
DEFAULT_N_BOOT = 2000

#: Columns of the committed split table that name a **block** — a phylogenetic /
#: homology exchangeability unit (PRD §12; ADR-0005 D5). ``cluster_id`` is the homology
#: block; the ``*_unit`` columns are the §9.2 held-out clade units. Kept as a literal
#: rather than derived from ``splits.FOLD_SCHEME_COLUMNS`` because this module is
#: deliberately stdlib-only (``metrics`` imports it, and ``metrics`` must import in the
#: bare CI env) while ``splits`` is a heavy CLI module; ``tests/unit/test_block_bootstrap.py``
#: carries the drift guard that re-derives this tuple from ``splits`` and fails if a new
#: ``*_unit`` column appears upstream.
BLOCK_GRANULARITY_COLUMNS: tuple[str, ...] = (
    "cluster_id",
    "loo_order_unit",
    "class_holdout_unit",
    "phylum_holdout_unit",
)

#: Columns that identify a **record**. Grouping by one of these yields one block per
#: record — i.e. a record-level bootstrap wearing a block-level API — so they are
#: refused by name rather than merely discouraged in a docstring.
RECORD_LEVEL_COLUMNS: tuple[str, ...] = (
    "record_id",
    "parent_record_id",
    "corpus_record_sha256",
)


def blocks_by_key(
    items: Sequence[Any],
    labels: Sequence[Any],
    *,
    key_name: str,
) -> list[list[Any]]:
    """Group ``items`` into resampling blocks by their ``labels``, taken from the split
    column ``key_name``.

    Returns one list per distinct label, in a deterministic label order (natural sort
    where the labels are mutually comparable, else by ``str``) so the block *sequence* —
    and therefore every seeded draw made from it — is reproducible.

    Refuses, rather than returns a degenerate grouping:

    - a ``key_name`` in :data:`RECORD_LEVEL_COLUMNS` (that grouping *is* a record-level
      bootstrap; ADR-0005 D5 forbids it);
    - a ``key_name`` outside :data:`BLOCK_GRANULARITY_COLUMNS` (fail closed on an
      unrecognised column rather than silently blocking on whatever it holds);
    - a missing / null block label — including the *stringified* nulls ``"None"``,
      ``"nan"``, ``"NA"`` that survive an ``is None`` check and would otherwise collapse
      every unlabelled record into one giant pseudo-block
      ([[stringified-null-survives-missing-checks]]);
    - a length mismatch between ``items`` and ``labels``.
    """
    if len(items) != len(labels):
        raise ValueError(f"items and labels must be the same length: {len(items)} != {len(labels)}")
    if key_name in RECORD_LEVEL_COLUMNS:
        raise ValueError(
            f"key_name={key_name!r} identifies a record, not a block: grouping by it "
            "yields one block per record, which is the record-level bootstrap ADR-0005 "
            f"D5 forbids. Block-granularity columns: {list(BLOCK_GRANULARITY_COLUMNS)}"
        )
    if key_name not in BLOCK_GRANULARITY_COLUMNS:
        raise ValueError(
            f"key_name={key_name!r} is not a known block-granularity column "
            f"{list(BLOCK_GRANULARITY_COLUMNS)} — refusing to resample on an "
            "unrecognised key (PRD §12)"
        )

    grouped: dict[Any, list[Any]] = {}
    for position, (item, label) in enumerate(zip(items, labels, strict=True)):
        if masking.is_missing(label) or masking.is_null_token(label):
            raise ValueError(
                f"{key_name}[{position}] is missing ({label!r}) — a record with no block "
                "label cannot be assigned to an exchangeability unit (PRD §12)"
            )
        grouped.setdefault(label, []).append(item)

    try:
        keys = sorted(grouped)
    except TypeError:  # mixed label types — still deterministic, just not natural order
        keys = sorted(grouped, key=str)
    return [grouped[k] for k in keys]


def block_bootstrap(
    blocks: Sequence,
    statistic: Callable[[list], float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile CI for ``statistic`` resampled at the **block** (homology-cluster /
    held-out-order) level, never per-record (ADR-0005 D5) — the phylogenetic
    exchangeability unit. Each ``blocks[i]`` is one block's data; a bootstrap replicate
    draws ``len(blocks)`` blocks with replacement (seeded, so reproducible —
    CLAUDE.md §8.3), concatenates them, and applies ``statistic``. Returns
    ``{point, lower, upper, ci_level, n_boot, n_blocks}`` over the finite replicates;
    fewer than 2 blocks → not block-resamplable (NaN CI, ADR-0005 Amendment A1).

    ``ci_level`` and ``n_boot`` are validated before any sampling (CodeRabbit CLI r2/r3).
    ``ci_level`` must be a fraction in the **open** interval ``(0, 1)``: outside it ``alpha``
    goes negative and ``lower > upper`` — an interval-shaped object that is not an interval —
    and a caller passing a percentage (``ci_level=95``) is a real way to land there. The
    endpoints are refused too, because a 0 % or 100 % "confidence interval" is a degenerate
    request rather than a measurement. ``n_boot`` must be ``>= 1`` so that a reported
    ``n_boot: 0`` has exactly **one** meaning — every replicate was undefined — instead of
    also covering "the caller asked for none", which is indistinguishable in the payload.

    ``n_boot`` in the return value is the number of replicates that actually produced a
    **finite** statistic, which is ``<=`` the requested count when the statistic is
    undefined on some draws. ``point`` is deliberately *not* filtered: it is whatever the
    statistic returns on the full sample, so an undefined point estimate stays visible
    rather than being replaced by a percentile of the replicates.

    Non-finite replicates are dropped, not only NaN ones. The P0-31 version this moved from
    tested ``isnan`` alone, so a statistic that could return ``±inf`` on a degenerate draw
    put an infinity into the replicate list and carried it straight through to ``lower`` or
    ``upper`` while ``n_boot`` still counted it (CodeRabbit CLI r1, reproduced by
    execution). No committed CI moves: every statistic resampled in this repo today —
    ``binned_ece``, ``ood_ece``, AUPRC, per-element F1 — is bounded, so the two filters
    agree on all of them.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot={n_boot!r} must be >= 1")
    if not math.isfinite(ci_level) or not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level={ci_level!r} must be a fraction in (0, 1)")
    blocks = list(blocks)
    n_blocks = len(blocks)
    point = statistic([x for blk in blocks for x in blk]) if n_blocks else float("nan")
    if n_blocks < 2:
        return {
            "point": point,
            "lower": float("nan"),
            "upper": float("nan"),
            "ci_level": ci_level,
            "n_boot": 0,
            "n_blocks": n_blocks,
        }
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        drawn = [blocks[rng.randrange(n_blocks)] for _ in range(n_blocks)]
        val = statistic([x for blk in drawn for x in blk])
        if math.isfinite(val):
            reps.append(val)
    reps.sort()
    alpha = (1.0 - ci_level) / 2.0
    lower = percentile(reps, alpha)
    upper = percentile(reps, 1.0 - alpha)
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "ci_level": ci_level,
        "n_boot": len(reps),
        "n_blocks": n_blocks,
    }


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy default ``'linear'``) over a pre-sorted
    list. NaN on empty; the single value when there is one.

    ``q`` outside ``[0, 1]`` is a caller error, not a clamp: it indexes off the ends of the
    list and returns an extrapolated number that looks like a quantile (CodeRabbit CLI r2).
    """
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError(f"q={q!r} must be finite and within [0, 1]")
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
