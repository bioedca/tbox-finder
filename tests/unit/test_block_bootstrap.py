"""P3-09 — the shared block resampler (:mod:`tbox_finder.eval.resample`).

Two things are under test and they need different kinds of evidence.

``block_bootstrap`` was **moved** here from ``metrics.block_bootstrap_ci``, which now
delegates. That makes "the two agree" a tautology
([[promote-dont-duplicate-is-a-correctness-rule]]),
so every check below is a **hand-computed** value or a structural property of the draw —
never a cross-module comparison. The one comparison that *is* worth making is the opposite
one: that the delegation is real (same object identity), because a delegation that quietly
forked would pass an agreement test on the day it was written and drift afterwards.

``blocks_by_key`` is new. Its whole job is refusing groupings that would silently become a
record-level bootstrap, so each refusal is paired with the corresponding clean input
succeeding ([[raises-test-needs-a-positive-control]]): a guard that raised on *everything*
would satisfy a bare ``pytest.raises`` just as well.

Pure stdlib — runs in the bare CI env.
"""

from __future__ import annotations

import math

import pytest

from tbox_finder import metrics as M
from tbox_finder.eval import resample


# ========================================================================== #
# blocks_by_key — grouping, and the refusals that keep it block-level
# ========================================================================== #
def test_blocks_by_key_groups_and_orders_deterministically() -> None:
    items = ["a", "b", "c", "d", "e"]
    labels = [7, 3, 7, 3, 11]
    blocks = resample.blocks_by_key(items, labels, key_name="cluster_id")
    # Sorted by label: 3 -> [b, d], 7 -> [a, c], 11 -> [e].
    assert blocks == [["b", "d"], ["a", "c"], ["e"]]
    # Every item lands in exactly one block; nothing is dropped or duplicated.
    assert sorted(x for blk in blocks for x in blk) == sorted(items)


def test_blocks_by_key_orders_mixed_type_labels_without_raising() -> None:
    # Mixed int/str labels are not mutually comparable, so the natural sort raises and the
    # function must fall back to a str sort rather than propagate a TypeError.
    blocks = resample.blocks_by_key([1, 2, 3], ["b", 10, "a"], key_name="loo_order_unit")
    assert blocks == [[2], [3], [1]]  # "10" < "a" < "b"


def test_blocks_by_key_refuses_record_level_keys_but_accepts_block_keys() -> None:
    items, labels = [1, 2], ["x", "y"]
    # Positive control: the same call with a block-granularity key succeeds.
    assert resample.blocks_by_key(items, labels, key_name="cluster_id") == [[1], [2]]
    for record_key in resample.RECORD_LEVEL_COLUMNS:
        with pytest.raises(ValueError, match="identifies a record, not a block"):
            resample.blocks_by_key(items, labels, key_name=record_key)


def test_blocks_by_key_fails_closed_on_an_unknown_key() -> None:
    with pytest.raises(ValueError, match="not a known block-granularity column"):
        resample.blocks_by_key([1, 2], ["x", "y"], key_name="resolved_genus")


@pytest.mark.parametrize("bad", [None, float("nan"), "None", "nan", " NA ", "<NA>"])
def test_blocks_by_key_refuses_missing_and_stringified_null_labels(bad: object) -> None:
    # The stringified spellings are the dangerous half: they survive an `is None` test and
    # would collapse every unlabelled record into one giant pseudo-block.
    with pytest.raises(ValueError, match="is missing"):
        resample.blocks_by_key([1, 2], ["c1", bad], key_name="cluster_id")
    # Positive control: a real label in that slot is accepted.
    assert resample.blocks_by_key([1, 2], ["c1", "c2"], key_name="cluster_id") == [[1], [2]]


def test_blocks_by_key_refuses_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        resample.blocks_by_key([1, 2, 3], ["c1", "c2"], key_name="cluster_id")


def test_block_granularity_columns_track_the_split_schema() -> None:
    """Drift guard for the allowlist, which is a literal here because this module must stay
    stdlib-only (``metrics`` imports it) while ``splits`` is a heavy CLI module.

    If §9.2 gains another held-out-unit column, the allowlist must gain it too — otherwise
    the new scheme's CIs quietly fail closed at the first ``blocks_by_key`` call.
    """
    from tbox_finder import splits

    upstream_units = {c for c in splits.FOLD_SCHEME_COLUMNS if c.endswith("_unit")}
    assert upstream_units, "no *_unit columns found — has FOLD_SCHEME_COLUMNS been renamed?"
    assert upstream_units <= set(resample.BLOCK_GRANULARITY_COLUMNS)
    # cluster_id is the homology block and is not a fold column, so it is pinned separately.
    assert "cluster_id" in resample.BLOCK_GRANULARITY_COLUMNS
    assert "cluster_id" in splits.COMMITTED_TABLE_COLUMNS
    # The refused names must be real record identifiers in the committed table, not strawmen.
    for col in resample.RECORD_LEVEL_COLUMNS:
        assert col in splits.COMMITTED_TABLE_COLUMNS
    assert not set(resample.RECORD_LEVEL_COLUMNS) & set(resample.BLOCK_GRANULARITY_COLUMNS)


# ========================================================================== #
# block_bootstrap — hand-computed point, block-granularity draws, guards
# ========================================================================== #
def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def test_point_estimate_is_the_statistic_over_all_records() -> None:
    blocks = [[1.0, 2.0], [6.0], [3.0, 4.0, 5.0]]
    out = resample.block_bootstrap(blocks, _mean, n_boot=10)
    # Hand-computed: (1+2+6+3+4+5)/6 = 3.5 — pooled over records, NOT a mean of block means
    # (which would be (1.5 + 6 + 4)/3 = 3.833...).
    assert out["point"] == pytest.approx(3.5)
    assert out["n_blocks"] == 3


#: Blocks of **unequal** size, each record tagged with its block. Unequal sizes are the
#: point: with equal sizes a record-level draw and a block-level draw produce replicates of
#: the same length, and the two are indistinguishable from the outside
#: ([[symmetric-count-fixture-blind-to-inversion]]).
_TAGGED_BLOCKS = [[("a", 0)], [("b", 0), ("b", 1)], [("c", i) for i in range(5)]]


#: Block sizes, keyed by tag — the shape `_all_blocks_whole` checks against.
_BLOCK_SIZES = {"a": 1, "b": 2, "c": 5}


def _all_blocks_whole(rows: list) -> float:
    """1.0 iff every original row appears in **complete block copies**.

    Checked as equal multiplicity across each block's own rows, not as a tag count
    divisible by the block size (CodeRabbit CLI r2, major). The count form accepted
    ``[("b", 0), ("b", 0)]`` — two copies of one row, with ``("b", 1)`` missing — because
    the tag occurs twice and 2 % 2 == 0. That is a draw the predicate exists to reject, so
    the count form made the control weaker than its own docstring claimed.
    """
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
    return float(
        all(
            len({counts.get((tag, index), 0) for index in range(size)}) == 1
            for tag, size in _BLOCK_SIZES.items()
        )
    )


def test_the_whole_block_predicate_rejects_a_partial_block_copy() -> None:
    """The predicate's own unit test, because the two tests below both rest on it.

    Each case is a draw the count-based form scored as whole: the multiplicities within a
    block are unequal, so a row is missing or over-represented relative to its blockmates.
    """
    assert _all_blocks_whole([("b", 0), ("b", 1)]) == 1.0  # one whole copy of block b
    assert _all_blocks_whole([("b", 0), ("b", 1), ("b", 0), ("b", 1)]) == 1.0  # two copies
    assert _all_blocks_whole([]) == 1.0  # no blocks drawn is vacuously whole
    assert _all_blocks_whole([("b", 0), ("b", 0)]) == 0.0  # 2x row 0, row 1 absent
    assert _all_blocks_whole([("c", 0)] * 5) == 0.0  # 5 copies of one row, not one block
    assert _all_blocks_whole([("a", 0), ("b", 0)]) == 0.0  # half of block b


def test_replicates_draw_whole_blocks_never_records() -> None:
    """The load-bearing property (ADR-0005 D5): a replicate is a multiset of *whole* blocks.

    Checked two ways, because "the CI looked plausible" is not evidence of block granularity:
    every replicate must consist of complete block copies, and the replicate *lengths* must
    fall in the enumerable set of three-block sums — which excludes the constant length a
    record-level bootstrap would always produce.
    """
    out = resample.block_bootstrap(_TAGGED_BLOCKS, _all_blocks_whole, n_boot=300, seed=1)
    assert (out["point"], out["lower"], out["upper"]) == (1.0, 1.0, 1.0)

    lengths = resample.block_bootstrap(
        _TAGGED_BLOCKS, lambda rows: float(len(rows)), n_boot=300, seed=1
    )
    attainable = {
        float(x + y + z) for x in (1, 2, 5) for y in (1, 2, 5) for z in (1, 2, 5)
    }  # {3,4,5,6,7,8,9,11,12,15} — note 10, 13, 14 are unreachable
    assert lengths["lower"] in attainable and lengths["upper"] in attainable
    assert (
        lengths["lower"] < 8.0 < lengths["upper"]
    ), "replicate size never varies — that is what a record-level bootstrap looks like"


def test_the_whole_block_control_bites_on_a_record_level_draw() -> None:
    """Matched control for the test above ([[matched-control-before-certifying]]).

    The same statistic applied to a hand-built **record-level** resample must fail it —
    otherwise `_all_blocks_whole` is measuring nothing and the test above is vacuous.
    """
    import random

    rows = [r for blk in _TAGGED_BLOCKS for r in blk]
    rng = random.Random(0)
    violations = sum(
        _all_blocks_whole([rows[rng.randrange(len(rows))] for _ in range(len(rows))]) == 0.0
        for _ in range(200)
    )
    # Measured **199/200** at this seed with the complete-block-copy predicate (it was
    # 174/200 with the weaker tag-count form r2 replaced — the sharpening is itself evidence
    # the finding was material). Block-level draws violate 0/300 in the test above, so the
    # two regimes are separated by essentially the whole range; the bar sits below the
    # measured rate so it pins the *contrast*, not this seed's exact count.
    assert violations >= 180, f"record-level draws must almost always violate; got {violations}/200"


def test_seeded_reproducibility_and_seed_sensitivity() -> None:
    blocks = [[float(i), float(i) + 0.5] for i in range(8)]
    a = resample.block_bootstrap(blocks, _mean, n_boot=200, seed=7)
    b = resample.block_bootstrap(blocks, _mean, n_boot=200, seed=7)
    assert a == b  # CLAUDE.md §8.3 — same seed, identical interval
    c = resample.block_bootstrap(blocks, _mean, n_boot=200, seed=8)
    assert (c["lower"], c["upper"]) != (a["lower"], a["upper"])  # the seed is actually used


def test_fewer_than_two_blocks_is_not_resamplable() -> None:
    one = resample.block_bootstrap([[1.0, 3.0]], _mean, n_boot=100)
    assert one["point"] == pytest.approx(2.0)  # the point is still defined
    assert math.isnan(one["lower"]) and math.isnan(one["upper"])
    assert one["n_boot"] == 0 and one["n_blocks"] == 1
    empty = resample.block_bootstrap([], _mean, n_boot=100)
    assert math.isnan(empty["point"]) and empty["n_blocks"] == 0
    # Positive control: two blocks *are* resamplable, so the guard is about block count.
    two = resample.block_bootstrap([[1.0], [3.0]], _mean, n_boot=100)
    assert two["n_boot"] == 100 and not math.isnan(two["lower"])


def test_nan_replicates_are_dropped_and_reported_in_n_boot() -> None:
    """``n_boot`` is the number of replicates that produced a finite statistic — not the
    number requested. A statistic that is undefined on some draws must shrink the count
    rather than poison the percentiles."""

    def only_if_mixed(xs: list) -> float:
        return _mean(xs) if len(set(xs)) > 1 else float("nan")

    out = resample.block_bootstrap([[1.0], [2.0]], only_if_mixed, n_boot=100, seed=3)
    # All-same draws (both blocks identical) are NaN and dropped; mixed draws survive.
    assert 0 < out["n_boot"] < 100
    assert out["lower"] == pytest.approx(1.5) and out["upper"] == pytest.approx(1.5)


def test_non_finite_replicates_are_dropped_not_only_nan_ones() -> None:
    """The filter is ``isfinite``, not ``not isnan`` (CodeRabbit CLI r1, reproduced by
    execution before the fix was accepted).

    ``math.isnan(inf)`` is False, so the P0-31 filter this moved from let an ``inf``
    replicate into the list, where a single one drags ``upper`` to infinity *and* is still
    counted in ``n_boot`` — which the docstring says counts finite replicates. Both signs
    are checked, and the NaN behaviour is asserted unchanged alongside so the fix is a
    widening, not a swap.
    """

    def inf_on_degenerate(xs: list) -> float:
        return float("inf") if len(set(xs)) == 1 else _mean(xs)

    def neg_inf_on_degenerate(xs: list) -> float:
        return float("-inf") if len(set(xs)) == 1 else _mean(xs)

    blocks = [[1.0], [2.0], [3.0]]
    up = resample.block_bootstrap(blocks, inf_on_degenerate, n_boot=50, seed=1)
    assert math.isfinite(up["upper"]) and up["n_boot"] < 50
    down = resample.block_bootstrap(blocks, neg_inf_on_degenerate, n_boot=50, seed=1)
    assert math.isfinite(down["lower"]) and down["n_boot"] < 50
    # Positive control: a bounded statistic keeps every replicate, so the filter is not
    # discarding good draws.
    plain = resample.block_bootstrap(blocks, _mean, n_boot=50, seed=1)
    assert plain["n_boot"] == 50 and math.isfinite(plain["lower"])
    # `point` is NOT filtered — an undefined point estimate must stay visible.
    degenerate = resample.block_bootstrap([[1.0], [1.0]], inf_on_degenerate, n_boot=10, seed=1)
    assert math.isinf(degenerate["point"]) and degenerate["n_boot"] == 0


def test_ci_level_widens_the_interval() -> None:
    blocks = [[float(i)] for i in range(20)]
    narrow = resample.block_bootstrap(blocks, _mean, n_boot=300, seed=5, ci_level=0.50)
    wide = resample.block_bootstrap(blocks, _mean, n_boot=300, seed=5, ci_level=0.99)
    assert wide["lower"] < narrow["lower"] <= narrow["upper"] < wide["upper"]
    assert narrow["ci_level"] == 0.50 and wide["ci_level"] == 0.99


def test_public_arguments_are_validated_before_sampling() -> None:
    """``ci_level`` and ``n_boot`` are checked up front (CodeRabbit CLI r2).

    A negative ``ci_level`` inverts ``alpha`` and yields ``lower > upper`` — an
    interval-shaped object that is not an interval — and a negative ``n_boot`` reports zero
    replicates exactly as an always-undefined statistic would. Each refusal is paired with
    the same call at a valid value succeeding, so the guards are shown to be selective and
    not simply raising on everything ([[raises-test-needs-a-positive-control]]).
    """
    blocks = [[1.0], [2.0], [3.0]]
    # 95 is the percentage form — a real way for a caller to land outside (0, 1).
    for bad_level in (-0.95, 1.5, 95, 0.0, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ci_level"):
            resample.block_bootstrap(blocks, _mean, n_boot=10, ci_level=bad_level)
    for bad_boot in (-1, 0):
        with pytest.raises(ValueError, match="n_boot"):
            resample.block_bootstrap(blocks, _mean, n_boot=bad_boot)
    ok = resample.block_bootstrap(blocks, _mean, n_boot=10, ci_level=0.9)
    assert ok["ci_level"] == 0.9 and ok["lower"] <= ok["upper"]
    # n_boot=0 is refused so that a REPORTED n_boot of 0 has exactly one meaning: every
    # replicate was undefined. The <2-blocks path still reports 0, distinguishable by
    # n_blocks.
    assert resample.block_bootstrap([[1.0]], _mean, n_boot=10)["n_boot"] == 0

    for bad_q in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="q="):
            resample.percentile([0.0, 1.0], bad_q)
    assert resample.percentile([0.0, 1.0], 0.5) == pytest.approx(0.5)


def test_percentile_is_linear_interpolation() -> None:
    # Hand-computed against numpy's default 'linear' method on [0, 1, 2, 3]:
    # q=0.5 -> pos 1.5 -> 1.5 ; q=0.25 -> pos 0.75 -> 0.75 ; q=0 -> 0 ; q=1 -> 3.
    vals = [0.0, 1.0, 2.0, 3.0]
    assert resample.percentile(vals, 0.5) == pytest.approx(1.5)
    assert resample.percentile(vals, 0.25) == pytest.approx(0.75)
    assert resample.percentile(vals, 0.0) == pytest.approx(0.0)
    assert resample.percentile(vals, 1.0) == pytest.approx(3.0)
    assert math.isnan(resample.percentile([], 0.5))
    assert resample.percentile([2.5], 0.9) == pytest.approx(2.5)


# ========================================================================== #
# The delegation is real — not a fork that happens to agree today
# ========================================================================== #
def test_metrics_entry_point_delegates_to_this_module() -> None:
    blocks = [[1.0, 2.0], [3.0], [4.0, 5.0]]
    assert M.block_bootstrap_ci(blocks, _mean, n_boot=50, seed=4) == resample.block_bootstrap(
        blocks, _mean, n_boot=50, seed=4
    )
    # The above is a tautology *if* the delegation holds, so pin the delegation itself:
    # a re-implementation under the old name would keep the agreement test green.
    assert M.block_bootstrap is resample.block_bootstrap  # identity, not just provenance
    assert M.block_bootstrap.__module__ == "tbox_finder.eval.resample"
    assert M.DEFAULT_N_BOOT == resample.DEFAULT_N_BOOT == 2000
    src = __import__("inspect").getsource(M.block_bootstrap_ci)
    assert "return block_bootstrap(" in src, "metrics must forward, not re-implement"
