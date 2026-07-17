"""P2-06a — the validation ladder: the leak-free selection fold + the selection rule.

Bare-CI tier: no torch, no model. The loader tests touch the committed parquet trio and
skip when it is absent (the DVC inputs are gitignored); the selection-rule and confusion
tests are pure and always run.

What these tests are FOR, stated plainly, because this repo's recurring defect is a green
suite that proves nothing:

* The selection fold's whole reason to exist is that it is **disjoint from the training
  fold**. So the tests that matter are the ones that would fail if it were not — and each
  is proven to bite by sabotage, not by reading.
* The gate clause's failure mode is going **vacuously TRUE when the evidence is absent**
  (the P2-05 lesson). So the absence branch's *gate* is asserted, not just its fields.
"""

from __future__ import annotations

import math

import pytest

from tbox_finder.train import select_best as SB

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

DATA_DEPS = (
    "data/interim/flank_context/context_v0.parquet",
    "data/processed/labels/labels_v0.parquet",
    "data/processed/splits/split_assignments.parquet",
)


def _have_data() -> bool:
    from pathlib import Path

    return all(Path(p).exists() for p in DATA_DEPS)


def _pandas_or_skip():
    return pytest.importorskip("pandas")


requires_corpus = pytest.mark.skipif(
    not _have_data(), reason="DVC/LFS corpus inputs not materialised locally"
)


# ══════════════════════════════════════════════════════════════════════════════
# The selection fold — measured against the committed table
# ══════════════════════════════════════════════════════════════════════════════
@requires_corpus
def test_selection_val_is_disjoint_from_the_training_fold():
    """The load-bearing invariant: zero shared records AND zero shared clusters.

    Sharing a *cluster* is the subtle half — scheme A assigns whole clusters, but
    ``nested_train`` is an independent scheme, so a cluster can straddle the boundary. On
    the committed table 7 clusters do.
    """
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import load_corpus_records, load_selection_val_records

    val, report = load_selection_val_records()
    train, _ = load_corpus_records(training_fold_only=True)

    val_ids = {r.record_id for r in val}
    train_ids = {r.record_id for r in train}
    val_clusters = {r.cluster_id for r in val}
    train_clusters = {r.cluster_id for r in train}

    assert val_ids & train_ids == set(), "a selection record was trained on"
    assert (
        val_clusters & train_clusters == set()
    ), "a selection cluster was trained on (homology leak)"
    # The report must AGREE with the independent recomputation above, not merely assert 0.
    assert report["disjointness"]["shared_record_ids_with_train"] == 0
    assert report["disjointness"]["shared_cluster_ids_with_train"] == 0


@requires_corpus
def test_the_naive_val_filter_would_have_leaked_and_this_is_why_the_test_exists():
    """Anti-tautology: prove the two extra filters are doing real work.

    ``fold_random == "val"`` alone is 61.4% training records on the committed table. If a
    future refactor drops filter 2 or 3, the disjointness test above bites — this test
    documents the magnitude so nobody 'simplifies' the filter back.
    """
    pd = _pandas_or_skip()
    from tbox_finder.data.window_dataset import (
        CLUSTER_COL,
        DEFAULT_SPLIT_TABLE,
        FOLD_RANDOM_COL,
        FOLD_RANDOM_VAL,
        NESTED_TRAIN_COL,
        POSITIVE_SOURCE,
        SOURCE_COL,
    )

    spl = pd.read_parquet(DEFAULT_SPLIT_TABLE)
    c = spl[spl[SOURCE_COL] == POSITIVE_SOURCE]
    naive = c[c[FOLD_RANDOM_COL] == FOLD_RANDOM_VAL]
    n_trained_on = int(naive[NESTED_TRAIN_COL].astype(bool).sum())

    assert n_trained_on > 0, (
        "the naive fold_random=='val' filter no longer overlaps the training fold — either "
        "the split table changed or this guard is now vacuous; re-derive before deleting"
    )
    # And the cluster-straddle that survives filter 2 is real too.
    train_clusters = set(c[c[NESTED_TRAIN_COL].astype(bool)][CLUSTER_COL])
    after_filter2 = naive[~naive[NESTED_TRAIN_COL].astype(bool)]
    straddling = int(after_filter2[CLUSTER_COL].isin(train_clusters).sum())
    assert straddling > 0, (
        "no cluster straddles the val/train boundary any more — filter 3 may be vacuous; "
        "re-derive rather than assume"
    )


@requires_corpus
def test_selection_val_is_powered_and_shaped_as_measured():
    """Pin the measured shape against the committed table (2026-07-17).

    These are counts re-derived from the real artifact, not constants chosen to make the
    code pass: 880 records / 726 blocks / 35 class-II, after all four filters. A change in
    either the derivation or the table moves them, which is the point.
    """
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import load_selection_val_records, selection_val_problems

    records, report = load_selection_val_records()
    assert selection_val_problems(report) == []
    assert report["n_records"] == 880 == len(records)
    assert report["n_blocks"] == 726
    assert report["class_ii_records"] == 35
    assert report["class_ii_records"] >= report["min_heldout_positives"]
    assert report["fold_scope"] == "selection_val"
    assert report["excluded_by_reason"]["in_training_fold"] == 1445
    assert report["excluded_by_reason"]["cluster_shared_with_training_fold"] == 12


@requires_corpus
def test_every_selection_record_admits_an_unpadded_full_context_tiling():
    """Filter 4's consequence: the eval never zero-flanks an invented boundary."""
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import (
        encode_eval_window,
        load_selection_val_records,
        tile_windows,
    )

    records, _ = load_selection_val_records()
    for rec in records[:50]:
        seq_len = len(rec.context_seq)
        assert seq_len >= 1024
        for s in tile_windows(seq_len, window=1024, stride=512):
            assert s >= 0 and s + 1024 <= seq_len
            assert encode_eval_window(rec, s, window=1024).shape == (1024,)


@requires_corpus
def test_encode_eval_window_refuses_to_invent_a_boundary():
    """A window off the end must RAISE, not silently pad — padding asserts a contig end."""
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import encode_eval_window, load_selection_val_records

    rec = load_selection_val_records()[0][0]
    with pytest.raises(ValueError, match="invent a boundary"):
        encode_eval_window(rec, len(rec.context_seq) - 10, window=1024)
    with pytest.raises(ValueError, match="invent a boundary"):
        encode_eval_window(rec, -1, window=1024)


@requires_corpus
def test_context_labels_paint_the_locus_on_a_background_field():
    _pandas_or_skip()
    import numpy as np

    from tbox_finder.data.window_dataset import (
        BACKGROUND_INDEX,
        context_labels,
        label_indices,
        load_selection_val_records,
    )

    for rec in load_selection_val_records()[0][:20]:
        lab = context_labels(rec)
        assert lab.shape == (len(rec.context_seq),)
        lo, hi = rec.locus_offset, rec.locus_offset + rec.locus_length
        np.testing.assert_array_equal(lab[lo:hi], label_indices(rec.label_string))
        assert np.all(lab[:lo] == BACKGROUND_INDEX)
        assert np.all(lab[hi:] == BACKGROUND_INDEX)
        assert not np.any(lab == -100), "eval truth must have nothing to ignore"


# ══════════════════════════════════════════════════════════════════════════════
# selection_val_problems — total, and fails closed
# ══════════════════════════════════════════════════════════════════════════════
def _good_scope() -> dict:
    return {
        "n_records": 880,
        "n_blocks": 726,
        "class_ii_records": 35,
        "min_heldout_positives": 20,
        "disjointness": {
            "shared_record_ids_with_train": 0,
            "shared_cluster_ids_with_train": 0,
        },
    }


def test_selection_val_problems_accepts_a_clean_scope():
    from tbox_finder.data.window_dataset import selection_val_problems

    assert selection_val_problems(_good_scope()) == []


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda s: s.pop("disjointness"), "disjointness"),
        (lambda s: s.__setitem__("disjointness", {}), "disjointness"),
        (
            lambda s: s["disjointness"].__setitem__("shared_record_ids_with_train", 1),
            "train-on-train",
        ),
        (
            lambda s: s["disjointness"].__setitem__("shared_cluster_ids_with_train", 7),
            "train-on-train",
        ),
        (lambda s: s.__setitem__("n_blocks", 1), "block-resamplable"),
        (lambda s: s.__setitem__("class_ii_records", 3), "min_heldout_positives"),
        (lambda s: s.__setitem__("n_records", 0), "n_records"),
    ],
)
def test_selection_val_problems_bites_on_each_violation(mutate, expect):
    """Every clause must FAIL on its own violation — proven one at a time, by sabotage."""
    from tbox_finder.data.window_dataset import selection_val_problems

    scope = _good_scope()
    mutate(scope)
    problems = selection_val_problems(scope)
    assert problems, f"sabotage {expect!r} was not caught — the clause is vacuous"
    assert any(expect in p for p in problems), f"{expect!r} not in {problems}"


def test_selection_val_problems_rejects_bool_as_a_count():
    """``isinstance(True, int)`` is True — the P1-15/P1-16 bool-as-count trap."""
    from tbox_finder.data.window_dataset import selection_val_problems

    scope = _good_scope()
    scope["disjointness"]["shared_record_ids_with_train"] = False
    assert any("must be an int" in p for p in selection_val_problems(scope))


# ══════════════════════════════════════════════════════════════════════════════
# The confusion route vs the PINNED metrics kernel (an independent cross-check)
# ══════════════════════════════════════════════════════════════════════════════
def test_confusions_agree_with_metrics_kernel():
    """The bootstrap's sufficient statistic must reproduce ``metrics.gate4_core_min_f1``.

    This is the check that earns the second implementation its place: the fast per-block
    confusion route is graded against the pinned pure-stdlib kernel on pooled data, never
    against itself. Deterministic pseudo-random labels — an independent draw per position,
    NOT a cheap hash whose values collapse ([[degenerate-fixture-generators]]).
    """
    np = pytest.importorskip("numpy")
    from tbox_finder import metrics as M
    from tbox_finder.train.train_stage1 import class_confusions, min_core_f1_from_confusions

    rng = np.random.default_rng(20260717)
    for _ in range(12):
        n = int(rng.integers(200, 900))
        y_true = rng.integers(0, 8, size=n)
        # Correlate the prediction with truth or every core F1 is ~0 and the comparison is
        # between two near-zeros, which would pass whatever the arithmetic did.
        y_pred = np.where(rng.random(n) < 0.6, y_true, rng.integers(0, 8, size=n))

        want = M.gate4_core_min_f1(y_true.tolist(), y_pred.tolist())["min_f1"]
        got = min_core_f1_from_confusions([class_confusions(y_true, y_pred)])
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want, rel=0, abs=1e-12)


def test_the_confusion_fixture_is_expressive_not_degenerate():
    """Guard the cross-check above from becoming a comparison of two NaNs / two zeros."""
    np = pytest.importorskip("numpy")
    from tbox_finder import metrics as M
    from tbox_finder.train.train_stage1 import class_confusions, min_core_f1_from_confusions

    rng = np.random.default_rng(20260717)
    n = 900
    y_true = rng.integers(0, 8, size=n)
    y_pred = np.where(rng.random(n) < 0.6, y_true, rng.integers(0, 8, size=n))
    val = min_core_f1_from_confusions([class_confusions(y_true, y_pred)])
    assert not math.isnan(val), "fixture yields NaN — the kernel comparison would be vacuous"
    assert 0.0 < val < 1.0, f"fixture yields a degenerate {val} — it cannot discriminate"
    assert M.gate4_core_min_f1(y_true.tolist(), y_pred.tolist())["min_f1"] > 0.0


def test_confusions_are_additive_across_blocks():
    """The identity the block bootstrap rests on: summing per-block confusions == pooling."""
    np = pytest.importorskip("numpy")
    from tbox_finder.train.train_stage1 import class_confusions, min_core_f1_from_confusions

    rng = np.random.default_rng(7)
    parts = []
    all_t, all_p = [], []
    for _ in range(5):
        n = int(rng.integers(100, 300))
        t = rng.integers(0, 8, size=n)
        p = np.where(rng.random(n) < 0.6, t, rng.integers(0, 8, size=n))
        parts.append(class_confusions(t, p))
        all_t.append(t)
        all_p.append(p)
    pooled = class_confusions(np.concatenate(all_t), np.concatenate(all_p))
    np.testing.assert_array_equal(np.sum(np.stack(parts), axis=0), pooled)
    assert min_core_f1_from_confusions(parts) == min_core_f1_from_confusions([pooled])


def test_min_core_f1_is_a_min_not_a_mean():
    """ADR-0004 D6 — proven by a case where min and mean differ."""
    np = pytest.importorskip("numpy")
    from tbox_finder.labels import CLASS_INDEX, CORE_ELEMENTS
    from tbox_finder.train.train_stage1 import _f1_from_confusion, min_core_f1_from_confusions

    conf = np.zeros((8, 3), dtype=np.int64)
    # Three deliberately unequal core F1s.
    unequal = [(90, 10, 10), (50, 50, 50), (10, 90, 90)]
    for elem, (tp, fp, fn) in zip(CORE_ELEMENTS, unequal, strict=True):
        conf[CLASS_INDEX[elem]] = (tp, fp, fn)
    got = min_core_f1_from_confusions([conf])
    f1s = [_f1_from_confusion(*(int(x) for x in conf[CLASS_INDEX[e]])) for e in CORE_ELEMENTS]
    assert got == pytest.approx(min(f1s))
    assert got != pytest.approx(sum(f1s) / len(f1s)), "a mean would have passed here — it must not"


def test_min_core_f1_is_nan_when_a_core_element_is_unmeasurable():
    """An absent core element must not silently certify (metrics.gate4_core_min_f1's rule)."""
    np = pytest.importorskip("numpy")
    from tbox_finder.train.train_stage1 import min_core_f1_from_confusions

    assert math.isnan(min_core_f1_from_confusions([np.zeros((8, 3), dtype=np.int64)]))
    assert math.isnan(min_core_f1_from_confusions([]))


def test_is_real_matches_sizing_is_real():
    """Drift guard: the duplicated predicate must agree with sizing's on the tricky values."""
    from tbox_finder.train.sizing import _is_real as sizing_is_real
    from tbox_finder.train.train_stage1 import _is_real as train_is_real

    tricky = [
        True,
        False,
        0,
        1,
        -1,
        0.0,
        1.5,
        -2.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        "1",
        None,
        [],
        {},
        10**20,
    ]
    for v in tricky:
        assert train_is_real(v) is sizing_is_real(v), f"predicates disagree on {v!r}"


# ══════════════════════════════════════════════════════════════════════════════
# select_best — the promotion rule
# ══════════════════════════════════════════════════════════════════════════════
def _point(score, *, lr=1e-4, gamma=2.0, alpha=0.0, label="p", ok=True, scope=True):
    rep = {
        "report_path": label,
        "gate": {"overall_pass": ok, "loss_finite": True},
        "diagnostics": {"config": {"gamma": gamma, "lr": lr, "class_weight_alpha": alpha}},
        "eval_metrics": {
            "eval_split": "selection_val",
            "gate4_core_min_f1": {"min_f1": score, "per_element_f1": {}},
            "block_bootstrap_ci": {"point": score, "lower": score - 0.05, "upper": score + 0.05},
        },
    }
    if scope:
        rep["eval_scope"] = {
            "fold_scope": "selection_val",
            "disjointness": {
                "shared_record_ids_with_train": 0,
                "shared_cluster_ids_with_train": 0,
            },
        }
    return rep


def test_select_best_promotes_the_highest_core_min_f1():
    out = SB.select_best(
        [_point(0.41, label="a"), _point(0.77, label="b"), _point(0.63, label="c")]
    )
    assert out["winner"]["point"] == "b"
    assert out["winner"]["score"] == pytest.approx(0.77)
    assert out["n_promotable"] == 3 and out["n_rejected"] == 0
    assert [p["point"] for p in out["ranking"]] == ["b", "c", "a"]


def test_select_best_returns_no_winner_on_an_empty_ladder():
    """Withhold rather than promote an arbitrary point (§10.3)."""
    out = SB.select_best([])
    assert out["winner"] is None and out["n_promotable"] == 0


def test_select_best_rejects_a_point_scored_on_the_wrong_fold():
    """The defect that must never pass: a config promoted on a non-selection fold."""
    bad = _point(0.99, label="leaky")
    bad["eval_scope"]["fold_scope"] = "loo_order_unit"
    out = SB.select_best([bad, _point(0.42, label="honest")])
    assert out["winner"]["point"] == "honest", "a 0.99 on the WRONG fold won the sweep"
    assert out["n_rejected"] == 1
    assert any("fold_scope" in p for p in out["rejected"][0]["problems"])


def test_select_best_rejects_a_point_whose_val_set_overlapped_training():
    bad = _point(0.99, label="leaky")
    bad["eval_scope"]["disjointness"]["shared_cluster_ids_with_train"] = 3
    out = SB.select_best([bad, _point(0.42, label="honest")])
    assert out["winner"]["point"] == "honest"
    assert any("shared_cluster_ids_with_train" in p for p in out["rejected"][0]["problems"])


def test_select_best_rejects_a_failed_gate_and_a_missing_scope():
    out = SB.select_best(
        [
            _point(0.99, label="ungated", ok=False),
            _point(0.98, label="noscope", scope=False),
            _point(0.10, label="honest"),
        ]
    )
    assert out["winner"]["point"] == "honest"
    assert out["n_rejected"] == 2


def test_a_nan_score_never_wins():
    """NaN is an ABSENT score, not a low one — it must not sort to the top."""
    out = SB.select_best([_point(float("nan"), label="nan"), _point(0.3, label="real")])
    assert out["winner"]["point"] == "real"
    assert out["n_rejected"] == 1


def test_ties_break_deterministically_and_are_reported():
    a = _point(0.5, lr=3e-4, gamma=2.0, label="hi-lr")
    b = _point(0.5, lr=3e-5, gamma=2.0, label="lo-lr")
    out = SB.select_best([a, b])
    assert out["winner"]["point"] == "lo-lr", "tie-break must prefer the lower lr"
    assert out["n_tied_at_winning_score"] == 2
    # Deterministic under input permutation.
    assert SB.select_best([b, a])["winner"]["point"] == "lo-lr"


def test_score_of_raises_rather_than_returning_a_sentinel():
    with pytest.raises(ValueError, match="not a promotable sweep point"):
        SB.score_of(_point(0.5, scope=False))
    assert SB.score_of(_point(0.5)) == pytest.approx(0.5)


def test_load_reports_surfaces_an_unreadable_point_by_name(tmp_path):
    good = tmp_path / "0.json"
    good.write_text(__import__("json").dumps(_point(0.6, label="x")))
    bad = tmp_path / "1.json"
    bad.write_text("{truncated")
    missing = tmp_path / "2.json"

    reports = SB.load_reports([good, bad, missing])
    assert len(reports) == 3
    out = SB.select_best(reports)
    assert out["n_promotable"] == 1 and out["n_rejected"] == 2
    assert {r["point"] for r in out["rejected"]} == {str(bad), str(missing)}
