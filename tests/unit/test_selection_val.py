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
def test_selection_val_holds_no_designated_loo_holdout_record():
    """**THE** invariant, and the one P2-06a's first definition failed at 778 / 880.

    Selecting a config on designated-LOO-holdout records optimises the exact population
    P2-14 reports as the PRD §12:241 leave-one-order-out headline, turning a generalization
    number into a selection-optimised one. Measured off the emitted records via splits.py's
    dedicated indicator — not off ``nested_role`` (whose ``"heldout"`` value conflates the
    LOO arm with the Actinobacteria phylum arm and external anchors), and above all not off
    the negation of ``nested_train`` (which is what shipped the defect).
    """
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import load_selection_val_records

    val, report = load_selection_val_records()

    leaked = [r.record_id for r in val if r.is_designated_loo_holdout]
    assert leaked == [], (
        f"{len(leaked)} selection records are in the designated LOO holdout — a config "
        "chosen here contaminates the PRD §12:241 headline"
    )
    # The report must AGREE with the independent recomputation above, not merely assert 0.
    assert report["leakage"]["n_designated_loo_holdout"] == 0
    assert report["leakage"]["n_not_nested_train"] == 0


@requires_corpus
def test_selection_val_is_disjoint_from_what_the_model_actually_trains_on():
    """Zero shared records AND zero shared clusters against ``inner_train``.

    Against ``inner_train`` — the fold ``load_corpus_records`` returns by default — not
    against the full D5 fold. The old code proved disjointness from the full fold and was
    still wrong; the population that matters is the one training actually sees.
    """
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import load_corpus_records, load_selection_val_records

    val, report = load_selection_val_records()
    train, train_report = load_corpus_records(training_fold_only=True)
    assert train_report["fold_scope"] == "inner_train"

    val_ids = {r.record_id for r in val}
    train_ids = {r.record_id for r in train}
    val_clusters = {r.cluster_id for r in val}
    train_clusters = {r.cluster_id for r in train}

    assert val_ids & train_ids == set(), "a selection record was trained on"
    assert (
        val_clusters & train_clusters == set()
    ), "a selection cluster was trained on (homology leak)"
    assert report["leakage"]["shared_record_ids_with_inner_train"] == 0
    assert report["leakage"]["shared_cluster_ids_with_inner_train"] == 0
    # The two loaders must agree on the size of the population they partition — they run
    # the carve independently, so a divergence here means the shared rule is not shared.
    assert report["leakage"]["n_inner_train_records"] == len(train)


@requires_corpus
def test_the_ORIGINAL_selection_filter_would_have_sat_inside_the_loo_holdout():
    """The regression guard for P2-06a's actual defect — pinned by measurement.

    ``nested_train = is_corpus & has_order & ~linked`` (splits.py:706), so its **negation
    means "in the LOO holdout"**, not "held out from training".

    ⚠ Two populations, one numerator — do not conflate them (a maintainer trap). This test
    reconstructs only the FIRST TWO filters of the original definition
    (``fold_random == "val" & not nested_train``); against the committed table that is
    **778 / 909 = 85.6% designated LOO holdout**. The original loader then applied two more
    filters (cluster-straddle + ``len(context_seq) >= 1024``), which removed 29 *non-LOO*
    records and so left the fully-filtered **emitted** fold at **778 / 880 = 88.4%** — the
    figure quoted in the dev-log, ``select_best.py`` and MEMORY.md. Same 778 LOO records in
    both; different denominators. Whichever slice you take, the top order (Lactobacillales,
    510) is itself a designated holdout order, and the fold's own disjointness block read a
    truthful and useless ``0 shared with train``.

    This test reconstructs the 2-filter version (the full one needs the context join) and
    asserts it is still a disaster, so nobody can 'simplify' the carve back to it and find
    the suite green. The pins below are the exact measured values; ``778`` is the invariant
    (the LOO records themselves), the ratio guard is deliberately loose so an ordinary table
    refresh does not trip it. If either stops holding, that is a finding to re-derive — not
    a guard to delete.
    """
    pd = _pandas_or_skip()
    from tbox_finder.data.window_dataset import (
        DEFAULT_SPLIT_TABLE,
        FOLD_RANDOM_COL,
        FOLD_RANDOM_VAL,
        LOO_HOLDOUT_COL,
        NESTED_TRAIN_COL,
        POSITIVE_SOURCE,
        SOURCE_COL,
    )

    spl = pd.read_parquet(DEFAULT_SPLIT_TABLE)
    c = spl[spl[SOURCE_COL] == POSITIVE_SOURCE]
    original = c[(c[FOLD_RANDOM_COL] == FOLD_RANDOM_VAL) & (~c[NESTED_TRAIN_COL].astype(bool))]
    n_loo = int(original[LOO_HOLDOUT_COL].astype(bool).sum())

    assert n_loo > 0, (
        "the original 'fold_random==val & not nested_train' filter no longer reaches the "
        "LOO holdout — the table changed; re-derive the whole rung before trusting this"
    )
    # The exact measured values of THIS 2-filter reconstruction (not the 880 emitted fold).
    assert (n_loo, len(original)) == (778, 909)
    # Not a marginal contamination: the substantial majority of that fold was the holdout.
    assert n_loo / len(original) > 0.5


@requires_corpus
def test_selection_val_is_shaped_as_measured():
    """Pin the measured shape against the committed table (2026-07-17, re-taken decision).

    Counts re-derived from the real artifact, not constants chosen to make the code pass:
    830 records / 469 blocks carved from the 8,303-record / 4,775-cluster D5 fold at
    fraction 0.10 / seed 20260717. A change in either the derivation or the table moves
    them, which is the point.
    """
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import load_selection_val_records, selection_val_problems

    records, report = load_selection_val_records()
    assert selection_val_problems(report) == []
    assert report["n_records"] == 830 == len(records)
    assert report["n_blocks"] == 469
    assert report["fold_scope"] == "selection_val"
    assert report["carve"]["n_fold_records"] == 8303
    assert report["carve"]["n_fold_clusters"] == 4775
    assert report["carve"]["n_val_clusters"] == 469
    assert report["leakage"]["n_inner_train_records"] == 7472
    # ~10% of the fold's records, by construction. Pinned as a consequence, not an intent.
    assert 0.09 < report["n_records"] / report["carve"]["n_fold_records"] < 0.11


@requires_corpus
def test_class_ii_is_reported_and_explicitly_NOT_gated():
    """The disclosed cost of the re-taken decision (user, 2026-07-17).

    The inner rung holds 5 class-II of the D5 fold's 22 — far under
    ``min_heldout_positives``. That is REPORTED, and deliberately does not fail the fold:
    the selection statistic is the GATE-4 core min-F1 over {Stem I, Specifier,
    Antiterminator}, structural elements to which class-I and class-II records both
    contribute. Gating a *selection* rung on a *generalization* min-N floor was a category
    error. This test exists so the exemption stays visible and deliberate rather than
    quietly eroding into 'we never checked'.
    """
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import load_selection_val_records, selection_val_problems

    _, report = load_selection_val_records()
    assert report["class_ii_records"] == 5
    assert report["class_ii_below_min_heldout_positives"] is True
    assert report["class_ii_records"] < report["min_heldout_positives"]
    # Reported, under the floor, and STILL a usable fold — that is the decision.
    assert selection_val_problems(report) == []


@requires_corpus
def test_the_carve_costs_exactly_what_was_signed_off():
    """The training fold loses 10.01% (8,303 -> 7,472) — the price the user accepted."""
    _pandas_or_skip()
    from tbox_finder.data.window_dataset import load_corpus_records

    inner, inner_rep = load_corpus_records(training_fold_only=True)
    full, full_rep = load_corpus_records(training_fold_only=True, exclude_selection_val=False)

    assert full_rep["fold_scope"] == "train"
    assert inner_rep["fold_scope"] == "inner_train"
    assert len(full) == 8303
    assert len(inner) == 7472
    lost = (len(full) - len(inner)) / len(full)
    assert 0.09 < lost < 0.11, f"the carve now costs {lost:.1%} of training data"


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
        "n_records": 830,
        "n_blocks": 469,
        "class_ii_records": 5,
        "min_heldout_positives": 20,
        "carve": {
            "fraction": 0.10,
            "seed": 20260717,
            "n_fold_records": 8303,
            "n_fold_clusters": 4775,
            "n_val_clusters": 469,
        },
        "leakage": {
            "n_designated_loo_holdout": 0,
            "n_not_nested_train": 0,
            "shared_record_ids_with_inner_train": 0,
            "shared_cluster_ids_with_inner_train": 0,
            "n_inner_train_records": 7472,
        },
    }


def test_selection_val_problems_accepts_a_clean_scope():
    from tbox_finder.data.window_dataset import selection_val_problems

    assert selection_val_problems(_good_scope()) == []


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda s: s.pop("leakage"), "leakage"),
        (lambda s: s.__setitem__("leakage", {}), "leakage"),
        # THE clause: the exact value P2-06a's first definition would have reported.
        (
            lambda s: s["leakage"].__setitem__("n_designated_loo_holdout", 778),
            "leave-one-order-out",
        ),
        (
            lambda s: s["leakage"].__setitem__("n_not_nested_train", 880),
            "outside the ADR-0004 D5 training fold",
        ),
        (
            lambda s: s["leakage"].__setitem__("shared_record_ids_with_inner_train", 1),
            "train-on-train",
        ),
        (
            lambda s: s["leakage"].__setitem__("shared_cluster_ids_with_inner_train", 7),
            "homology-leaky",
        ),
        # A 0-overlap claim against an EMPTY training fold is vacuously true.
        (
            lambda s: s["leakage"].__setitem__("n_inner_train_records", 0),
            "vacuously true",
        ),
        (lambda s: s.pop("carve"), "carve"),
        (lambda s: s["carve"].__setitem__("fraction", 1.0), "carve.fraction"),
        (lambda s: s["carve"].__setitem__("seed", "42"), "carve.seed"),
        (lambda s: s["carve"].__setitem__("n_val_clusters", 4775), "took the whole fold"),
        (lambda s: s.__setitem__("n_blocks", 1), "block-resamplable"),
        (lambda s: s.__setitem__("n_records", 0), "n_records"),
        (lambda s: s.__setitem__("class_ii_records", -1), "class_ii_records"),
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
    scope["leakage"]["n_designated_loo_holdout"] = False
    assert any("must be an int" in p for p in selection_val_problems(scope))


def test_the_class_ii_floor_is_NOT_a_clause():
    """The exemption, asserted rather than assumed.

    5 class-II against a floor of 20 must NOT make the fold unusable — that is the signed-off
    decision (2026-07-17). Written as a test because "we removed a check" and "the check was
    never right for this fold" are indistinguishable from the diff alone, and only one of
    them is allowed (§10.3 forbids weakening a test to hide a gap; this records why it is
    not one).
    """
    from tbox_finder.data.window_dataset import selection_val_problems

    scope = _good_scope()
    scope["class_ii_records"] = 0
    assert selection_val_problems(scope) == []


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
def _point(score, *, lr=1e-4, gamma=2.0, alpha=0.0, label="p", ok=True, scope=True, n_blocks=469):
    rep = {
        "report_path": label,
        "gate": {"overall_pass": ok, "loss_finite": True},
        "diagnostics": {"config": {"gamma": gamma, "lr": lr, "class_weight_alpha": alpha}},
        "eval_metrics": {
            "eval_split": "selection_val",
            "gate4_core_min_f1": {"min_f1": score, "per_element_f1": {}},
            "block_bootstrap_ci": {
                "point": score,
                "lower": score - 0.05,
                "upper": score + 0.05,
                "n_blocks": n_blocks,
            },
        },
    }
    if scope:
        rep["eval_scope"] = {
            "fold_scope": "selection_val",
            "full_fold": True,
            "eval_max_records": None,
            "n_blocks": n_blocks,
            "leakage": {
                "n_designated_loo_holdout": 0,
                "n_not_nested_train": 0,
                "shared_record_ids_with_inner_train": 0,
                "shared_cluster_ids_with_inner_train": 0,
                "n_inner_train_records": 7472,
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
    bad["eval_scope"]["leakage"]["shared_cluster_ids_with_inner_train"] = 3
    out = SB.select_best([bad, _point(0.42, label="honest")])
    assert out["winner"]["point"] == "honest"
    assert any("shared_cluster_ids_with_inner_train" in p for p in out["rejected"][0]["problems"])


def test_select_best_rejects_a_point_scored_on_the_loo_holdout():
    """P2-06a's own defect, at the promotion rung: a point whose val fold WAS the headline.

    The first version could not have caught this — its only leak check was disjointness
    from train, which such a point passes truthfully.
    """
    bad = _point(0.99, label="contaminated")
    bad["eval_scope"]["leakage"]["n_designated_loo_holdout"] = 778
    out = SB.select_best([bad, _point(0.42, label="honest")])
    assert out["winner"]["point"] == "honest", "a 0.99 scored on the LOO holdout won"
    assert any("n_designated_loo_holdout" in p for p in out["rejected"][0]["problems"])


def test_select_best_rejects_a_capped_point_however_high_it_scores():
    """The `full_fold` finding: the field was written from day one and read by NOBODY.

    A capped eval scores a slice. An 8-record slice can trivially post a 0.91 min-F1 that a
    full-fold 0.55 cannot touch — and before this clause the capped point simply won.
    """
    capped = _point(0.91, label="capped")
    capped["eval_scope"]["full_fold"] = False
    capped["eval_scope"]["eval_max_records"] = 8
    out = SB.select_best([capped, _point(0.55, label="full")])
    assert out["winner"]["point"] == "full", "a capped 8-record slice out-ranked a full fold"
    assert any("full_fold" in p for p in out["rejected"][0]["problems"])


def test_select_best_rejects_a_point_whose_ci_resampled_a_different_fold():
    """The builder raises on this — but a hand-assembled report never runs the builder."""
    bad = _point(0.99, label="mismatched")
    bad["eval_metrics"]["block_bootstrap_ci"]["n_blocks"] = 12
    out = SB.select_best([bad, _point(0.42, label="honest")])
    assert out["winner"]["point"] == "honest"
    assert any("did not resample" in p for p in out["rejected"][0]["problems"])


def test_selection_hygiene_is_re_derived_and_reports_absence_as_absence():
    """The `never_selected_on` finding: a hardcoded literal replaced by a measurement.

    The old field asserted ``["test", "loo_order_unit"]`` on every sweep it reduced, was
    re-derived by nothing, and was demonstrably FALSE for the fold that shipped. The
    replacement must (a) reflect what the points measured, and (b) distinguish "all clean"
    from "nothing carried the measurement" — which a bare max/all would render identically.
    """
    out = SB.select_best([_point(0.5, label="a"), _point(0.6, label="b")])
    hyg = out["selection_hygiene"]
    assert hyg["n_points_measured"] == 2
    assert hyg["max_designated_loo_holdout_over_points"] == 0
    assert hyg["all_points_zero_loo_holdout"] is True
    assert hyg["fold_scopes_seen"] == ["selection_val"]

    # No promotable points ⇒ nothing was measured. Absence must read as absence, never as
    # a clean bill of health ([[clauses-must-guard-emptiness]]).
    empty = SB.select_best([])["selection_hygiene"]
    assert empty["n_points_measured"] == 0
    assert empty["max_designated_loo_holdout_over_points"] is None
    assert empty["all_points_zero_loo_holdout"] is False


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
