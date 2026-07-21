"""The junction-SYMMETRY construction and its gate (P2-10d′-b, ADR-0005 A7 pin 7 amended).

The original pin asked whether the R2 splice junction was invisible to a k-mer probe.
Measured on the real substrate it is not — 0.5217 / 0.5264 / 0.5281 / 0.5328 at k=6 across
four independent draws, against nulls of 0.4947–0.5042. A chimera of two genomes is
readable. Because only negatives were spliced, that made "chimeric" a class-correlated
cue.

The amended construction splices positives too, at the negatives' own chimeric rate, so
the cue carries no class information. What this file defends is that the splice **cannot
move a label** (it would otherwise corrupt the very targets GATE-4 grades), that the rate
is **derived** rather than configured, and that the gate's clauses bite.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tbox_finder.data.embedding import (
    apply_symmetric_splice,
    symmetric_splice_interval,
)
from tbox_finder.data.window_dataset import BACKGROUND_INDEX, IGNORE_INDEX
from tbox_finder.eval.junction_probe import junction_symmetry_clauses

W = 128


def _labels(n_fore: int = 20, at: int = 50) -> np.ndarray:
    lab = np.full(W, BACKGROUND_INDEX, dtype=np.int16)
    lab[at : at + n_fore] = 3  # some non-background element class
    return lab


def _ids() -> np.ndarray:
    raw = hashlib.shake_256(b"host").digest(W)
    return np.array([b & 3 for b in raw], dtype=np.int16)


def _donor() -> np.ndarray:
    raw = hashlib.shake_256(b"donor").digest(W)
    return np.array([(b >> 2) & 3 for b in raw], dtype=np.int16)


# ── the interval selector ────────────────────────────────────────────────────────────
def test_interval_never_overlaps_a_labelled_element() -> None:
    lab = _labels()
    real = np.ones(W, dtype=bool)
    rng = np.random.default_rng(0)
    for _ in range(200):
        span = symmetric_splice_interval(lab, real, 12, rng, background_index=BACKGROUND_INDEX)
        assert span is not None
        lo, hi = span
        assert (lab[lo:hi] == BACKGROUND_INDEX).all()


def test_interval_never_covers_a_pad_position() -> None:
    """A zero-flanked position carries no DNA; writing donor bases there would invent
    sequence at a contig end that does not exist."""
    lab = _labels()
    real = np.ones(W, dtype=bool)
    real[:30] = False
    lab[:30] = IGNORE_INDEX
    rng = np.random.default_rng(1)
    for _ in range(200):
        span = symmetric_splice_interval(lab, real, 10, rng, background_index=BACKGROUND_INDEX)
        assert span is not None
        lo, hi = span
        assert real[lo:hi].all()
        assert lo >= 30


def test_interval_is_none_when_no_room_exists() -> None:
    """A locus that fills its window has no spare flank. The caller must count that, not
    force a placement over labelled elements."""
    lab = np.full(W, 3, dtype=np.int16)
    real = np.ones(W, dtype=bool)
    rng = np.random.default_rng(2)
    assert symmetric_splice_interval(lab, real, 4, rng, background_index=BACKGROUND_INDEX) is None


@pytest.mark.parametrize("bad", [0, -1])
def test_interval_refuses_a_non_positive_length(bad: int) -> None:
    rng = np.random.default_rng(3)
    assert (
        symmetric_splice_interval(
            _labels(), np.ones(W, dtype=bool), bad, rng, background_index=BACKGROUND_INDEX
        )
        is None
    )


def test_interval_spans_more_than_one_placement() -> None:
    """A selector that always returned the same offset would put every positive's patch at
    one coordinate — itself a cue, and a degenerate one."""
    lab = _labels()
    real = np.ones(W, dtype=bool)
    seen = {
        symmetric_splice_interval(
            lab, real, 8, np.random.default_rng(s), background_index=BACKGROUND_INDEX
        )
        for s in range(60)
    }
    assert len(seen) > 5, seen


# ── the splice itself ────────────────────────────────────────────────────────────────
def test_splice_preserves_length_and_changes_only_the_interval() -> None:
    ids, lab, real = _ids(), _labels(), np.ones(W, dtype=bool)
    out, span = apply_symmetric_splice(
        ids, lab, real, _donor(), 12, np.random.default_rng(4), background_index=BACKGROUND_INDEX
    )
    assert span is not None
    lo, hi = span
    assert len(out) == len(ids)
    assert np.array_equal(out[:lo], ids[:lo])
    assert np.array_equal(out[hi:], ids[hi:])


def test_splice_actually_changes_the_bases_it_covers() -> None:
    """A no-op splice would leave 'chimeric' absent from the positive arm while the rate
    said otherwise — the augmentation would look configured and do nothing."""
    ids, lab, real = _ids(), _labels(), np.ones(W, dtype=bool)
    out, span = apply_symmetric_splice(
        ids, lab, real, _donor(), 24, np.random.default_rng(5), background_index=BACKGROUND_INDEX
    )
    assert span is not None
    lo, hi = span
    assert not np.array_equal(out[lo:hi], ids[lo:hi])


def test_splice_does_not_mutate_its_input() -> None:
    ids = _ids()
    before = ids.copy()
    apply_symmetric_splice(
        ids,
        _labels(),
        np.ones(W, dtype=bool),
        _donor(),
        12,
        np.random.default_rng(6),
        background_index=BACKGROUND_INDEX,
    )
    assert np.array_equal(ids, before)


def test_splice_is_a_no_op_when_no_interval_exists() -> None:
    ids = _ids()
    out, span = apply_symmetric_splice(
        ids,
        np.full(W, 3, dtype=np.int16),
        np.ones(W, dtype=bool),
        _donor(),
        8,
        np.random.default_rng(7),
        background_index=BACKGROUND_INDEX,
    )
    assert span is None and np.array_equal(out, ids)


# ── the gate clauses ─────────────────────────────────────────────────────────────────
def _summary(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "rate": 3701 / 19409,
        "n_chimeric_negatives": 3701,
        "n_negative_records": 19409,
        "n_labels_changed": 0,
        "n_windows_label_checked": 500,
    }
    base.update(over)
    return base


def _realized(pos_rate: float, neg_rate: float, n: int = 20000) -> dict[str, object]:
    return {
        "n_positive_draws": n,
        "n_negative_draws": n,
        "n_positive_chimeric": int(round(pos_rate * n)),
        "n_negative_chimeric": int(round(neg_rate * n)),
    }


def test_all_clauses_pass_on_a_symmetric_construction() -> None:
    """MUST FIRE, or every failure case below is vacuous."""
    realized = {
        "augmented": _realized(0.1907, 0.1907),
        "control_disabled": _realized(0.0, 0.1907),
    }
    assert all(junction_symmetry_clauses(_summary(), realized).values())


@pytest.mark.parametrize("bad", [None, {}, "x", 5])
def test_a_malformed_summary_fails_closed(bad: object) -> None:
    clauses = junction_symmetry_clauses(bad)  # type: ignore[arg-type]
    assert set(clauses) and not any(clauses.values())


def test_a_configured_rate_that_is_not_the_derived_one_fails() -> None:
    """The rate must be n_chimeric/n_negatives, not a number someone typed."""
    assert not junction_symmetry_clauses(_summary(rate=0.5))["rate_is_derived"]


def test_a_moved_label_fails() -> None:
    assert not junction_symmetry_clauses(_summary(n_labels_changed=1))["labels_unchanged"]


def test_label_invariance_is_not_vacuous_when_nothing_was_checked() -> None:
    """`0 labels changed` over `0 windows checked` is the classic empty-evidence pass."""
    assert not junction_symmetry_clauses(_summary(n_windows_label_checked=0))["labels_unchanged"]


def test_divergent_realized_rates_fail() -> None:
    realized = {
        "augmented": _realized(0.05, 0.1907),
        "control_disabled": _realized(0.0, 0.1907),
    }
    assert not junction_symmetry_clauses(_summary(), realized)["realized_rates_agree"]


def test_an_augmentation_that_never_fires_is_caught() -> None:
    """Configured but inert: the positive rate stays 0 while negatives are 19 % chimeric."""
    realized = {
        "augmented": _realized(0.0, 0.1907),
        "control_disabled": _realized(0.0, 0.1907),
    }
    assert not junction_symmetry_clauses(_summary(), realized)["realized_rates_agree"]


def test_the_disabled_control_must_be_separable() -> None:
    """If the disabled arm ALSO agrees, the measurement cannot tell a working augmentation
    from a broken measurement that reports zeros everywhere."""
    realized = {
        "augmented": _realized(0.1907, 0.1907),
        "control_disabled": _realized(0.1907, 0.1907),
    }
    clauses = junction_symmetry_clauses(_summary(), realized)
    assert clauses["realized_rates_agree"] is True
    assert clauses["control_fires_when_disabled"] is False
    assert not all(clauses.values())


def test_a_missing_control_arm_fails_rather_than_passing() -> None:
    realized = {"augmented": _realized(0.1907, 0.1907)}
    assert not junction_symmetry_clauses(_summary(), realized)["control_fires_when_disabled"]


def test_realized_clauses_are_false_without_a_realized_block() -> None:
    clauses = junction_symmetry_clauses(_summary())
    assert clauses["rate_is_derived"] and clauses["labels_unchanged"]
    assert not clauses["realized_rates_agree"]
    assert not clauses["control_fires_when_disabled"]


@pytest.mark.parametrize("key", ["n_positive_draws", "n_negative_chimeric"])
def test_an_incomplete_realized_arm_fails_closed(key: str) -> None:
    arm = _realized(0.1907, 0.1907)
    arm.pop(key)
    realized = {"augmented": arm, "control_disabled": _realized(0.0, 0.1907)}
    assert not junction_symmetry_clauses(_summary(), realized)["realized_rates_agree"]


def test_bools_are_not_accepted_as_counts() -> None:
    assert not junction_symmetry_clauses(_summary(n_chimeric_negatives=True))["rate_is_derived"]
