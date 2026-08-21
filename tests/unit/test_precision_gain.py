"""P3-16 — unit tests for the two-stage vs Stage-1-only precision comparison.

The committed report is a *pinning* artifact, so most of what matters here is tested against
constructed payloads rather than by reading it: a test that only reads
``reports/two_stage_precision.json`` cannot see the code that produced it
([[artifact-pinning-test-cannot-see-the-code]]). Where a test does read the committed report
it says so and pairs with a constructed twin that exercises the same function.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import pytest

from tbox_finder.integration import precision as P
from tbox_finder.metrics import precision_at_matched_recall, precision_recall_at_threshold

REPO = Path(__file__).resolve().parents[2]
ITEMS_PATH = REPO / "reports" / "p3" / "two_stage_precision_items.json"
REPORT_PATH = REPO / "reports" / "two_stage_precision.json"


# ──────────────────────────────────────────────────────────────────────────────
# The metric kernel
# ──────────────────────────────────────────────────────────────────────────────
def test_precision_at_matched_recall_hand_computed():
    y = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.4, 0.7, 0.3, 0.2]
    # recall >= 0 admits every threshold; the best precision is the 0.8 cut (2 TP, 0 FP).
    assert precision_at_matched_recall(y, s, 0.0) == {
        "threshold": 0.8,
        "precision": 1.0,
        "recall": pytest.approx(2 / 3),
        "matched": True,
    }
    # A target above 2/3 forces the 0.4 cut: 3 TP, 1 FP.
    got = precision_at_matched_recall(y, s, 0.7)
    assert got["threshold"] == 0.4
    assert got["precision"] == pytest.approx(0.75)
    assert got["recall"] == pytest.approx(1.0)


def test_precision_at_matched_recall_fails_closed_when_unreachable():
    """An arm that cannot reach the target recall reports precision 0.0, never NaN and never
    a number — NaN would make ``a > b`` False by accident rather than by design, and a
    number would let an unmeasurable arm win the gate."""
    got = precision_at_matched_recall([1, 0], [0.5, 0.4], 1.5)
    assert got["matched"] is False
    assert got["precision"] == 0.0
    assert math.isnan(got["recall"])
    # The failing direction: it must not beat a real arm.
    assert not got["precision"] > 0.5


def test_candidates_restriction_keeps_an_uncalled_item_uncallable():
    """The sentinel score of an item the candidate stage never emitted must not become a
    threshold. Without ``candidates`` the sweep would offer -1.0 and call it."""
    y = [1, 1, 0]
    s = [0.9, P.UNCALLED_SCORE, 0.8]
    restricted = precision_at_matched_recall(y, s, 0.5, candidates=[0.9, 0.8])
    assert restricted["threshold"] == 0.9
    assert restricted["recall"] == pytest.approx(0.5)
    unrestricted = precision_at_matched_recall(y, s, 1.0)
    # The positive control for the clause above: unrestricted, the sentinel IS reachable and
    # recall 1.0 becomes attainable. If this ever stops being true the restriction is moot.
    assert unrestricted["matched"] is True
    assert unrestricted["threshold"] == P.UNCALLED_SCORE
    # …and the half that makes ``candidates`` load-bearing. The sentinel is the LOWEST score,
    # so it can never win on precision — the restriction only bites where the target recall is
    # unreachable without it. Deleting the parameter turns this False into True.
    restricted_ceiling = precision_at_matched_recall(y, s, 1.0, candidates=[0.9, 0.8])
    assert restricted_ceiling["matched"] is False


def test_the_kernel_is_the_mirror_of_recall_at_matched_precision():
    """Both kernels must agree about the same operating point from opposite directions."""
    y = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.4, 0.7, 0.3, 0.2]
    at_recall = precision_at_matched_recall(y, s, 0.9)
    p, r = precision_recall_at_threshold(y, s, at_recall["threshold"])
    assert (p, r) == (at_recall["precision"], at_recall["recall"])


# ──────────────────────────────────────────────────────────────────────────────
# Prevalence reweighting, and the invariance the gate rests on
# ──────────────────────────────────────────────────────────────────────────────
def test_reweighted_precision_hand_computed():
    # 100:1 over a benchmark that sampled 1 decoy per positive => lambda = 100.
    assert P.prevalence_lambda(10, 10, 100) == pytest.approx(100.0)
    # Asymmetric on purpose: with n_pos == n_neg the numerator and denominator can be
    # swapped and every assertion in this file still passes, while every published AUPRC
    # moves. These are the committed benchmark's own counts.
    assert P.prevalence_lambda(1201, 692, 100) == pytest.approx(100.0 * 1201 / 692)
    assert P.prevalence_lambda(692, 1201, 100) == pytest.approx(100.0 * 692 / 1201)
    assert P.reweighted_precision(tp=8, fp=1, lam=100.0) == pytest.approx(8 / 108)
    # A threshold that calls nothing has undefined precision, not 0.
    assert math.isnan(P.reweighted_precision(tp=0, fp=0, lam=3.0))


def test_reweighting_refuses_a_degenerate_benchmark():
    for kwargs in (
        {"n_positives": 0, "n_negatives": 5, "decoy_ratio": 10},
        {"n_positives": 5, "n_negatives": 0, "decoy_ratio": 10},
        {"n_positives": 5, "n_negatives": 5, "decoy_ratio": 0},
    ):
        with pytest.raises(P.PrecisionError):
            P.prevalence_lambda(**kwargs)
    # Positive control: the same call with legal arguments does NOT raise, so the three
    # above are refusals of their own condition and not of every input.
    assert P.prevalence_lambda(5, 5, 10) == pytest.approx(10.0)


def test_the_verdict_is_invariant_under_prevalence_but_the_magnitude_is_not():
    """The property the whole D7 tension rests on, exercised rather than argued.

    Under ``tp/(tp + lambda*fp)`` the *sign* of the difference between two arms is
    lambda-independent while the magnitude is not; if this ever fails, the report's
    ``verdict_invariant_across_prevalence`` clause is not a tautology and the gate would
    genuinely depend on an unpinned-for-P3 number."""
    two = {"tp": 90, "fp": 2}
    one = {"tp": 90, "fp": 9}
    gaps = []
    for ratio in (1, 10, 100, 1_000, 10_000):
        lam = P.prevalence_lambda(100, 100, ratio)
        p_two = P.reweighted_precision(two["tp"], two["fp"], lam)
        p_one = P.reweighted_precision(one["tp"], one["fp"], lam)
        assert p_two > p_one
        gaps.append(p_two - p_one)
    assert len(set(round(gap, 9) for gap in gaps)) == len(gaps), "magnitudes must move"


# ──────────────────────────────────────────────────────────────────────────────
# Folding candidate-table rows onto benchmark items
# ──────────────────────────────────────────────────────────────────────────────
def _item(contig_id: str, label: int, pool: str = "corpus", block: str = "b1") -> dict:
    return {
        "contig_id": contig_id,
        "label": label,
        "pool": pool,
        "block": block,
        "seen_by": {"twin": False, "production": True},
    }


def _row(contig_id: str, peak: float, posterior: float) -> dict:
    return {
        "contig_id": contig_id,
        "peak_p_elem": peak,
        "stage2_named_posterior": posterior,
    }


def test_item_score_is_the_max_over_the_items_loci_not_the_first_or_the_mean():
    scored = P.item_scores(
        [_item("a", 1)],
        [_row("a", 0.2, 0.1), _row("a", 0.9, 0.8), _row("a", 0.4, 0.3)],
    )
    assert scored[0]["n_rows"] == 3
    assert scored[0]["stage1_only"] == pytest.approx(0.9)
    assert scored[0]["two_stage"] == pytest.approx(0.8)


def test_an_item_with_no_locus_is_uncalled_not_zero_scored():
    scored = P.item_scores([_item("a", 1), _item("b", 0)], [_row("a", 0.9, 0.8)])
    by_id = {item["contig_id"]: item for item in scored}
    assert by_id["b"]["n_rows"] == 0
    assert by_id["b"]["stage1_only"] == P.UNCALLED_SCORE
    assert P.reachable_scores(scored, "stage1_only") == [0.9]


def test_a_table_row_for_an_unknown_contig_is_refused():
    with pytest.raises(P.PrecisionError, match="not a benchmark item"):
        P.item_scores([_item("a", 1)], [_row("ghost", 0.9, 0.8)])


def test_a_duplicate_item_is_refused_rather_than_merged():
    with pytest.raises(P.PrecisionError, match="duplicate benchmark item"):
        P.item_scores([_item("a", 1), _item("a", 0)], [])


def test_a_missing_score_is_refused_rather_than_ranked_last():
    with pytest.raises(P.PrecisionError, match="would silently rank"):
        P.item_scores(
            [_item("a", 1)],
            [{"contig_id": "a", "peak_p_elem": None, "stage2_named_posterior": 0.5}],
        )
    with pytest.raises(P.PrecisionError, match="would silently rank"):
        P.item_scores([_item("a", 1)], [_row("a", 0.5, float("nan"))])


def test_a_non_binary_label_is_refused():
    with pytest.raises(P.PrecisionError, match="not 0 or 1"):
        P.item_scores([_item("a", 2)], [])


def test_arm_items_refuses_an_arm_the_table_does_not_carry():
    folded = {"arms": ["twin"], "items": []}
    with pytest.raises(P.PrecisionError, match="no arm 'production'"):
        P.arm_items(folded, "production")


# ──────────────────────────────────────────────────────────────────────────────
# The vectorised bootstrap twin must not be a second definition
# ──────────────────────────────────────────────────────────────────────────────
def test_fast_twin_matches_the_stdlib_kernel_on_the_real_benchmark():
    """The bootstrap's numpy inner loop and the published kernel must agree everywhere.

    Two implementations of one quantity is the shape that lets a CI be fixed while a
    published number keeps the bug ([[promote-dont-duplicate-is-a-correctness-rule]]). The
    twin exists only for speed, so it is checked against the kernel on the real data at every
    grid point and every sweep ratio rather than on a toy."""
    import numpy as np

    assert ITEMS_PATH.exists(), f"{ITEMS_PATH} is committed by this step and must exist"
    folded = json.loads(ITEMS_PATH.read_text())
    for arm in folded["arms"]:
        scored = P.arm_items(folded, arm)
        y_true, _ = P.arrays(scored, "two_stage")
        n_pos, n_neg = sum(y_true), len(y_true) - sum(y_true)
        labels = np.asarray(y_true, dtype=np.int64)
        for system in P.SYSTEMS:
            scores = np.asarray(P.arrays(scored, system)[1], dtype=np.float64)
            candidates = P.reachable_scores(scored, system)
            for target in (0.1, 0.5, 0.9, 0.99):
                for ratio in (1, 10, 100, 10_000):
                    lam = P.prevalence_lambda(n_pos, n_neg, ratio)
                    fast = P._fast_precision_at_matched_recall(labels, scores, target, lam=lam)
                    match = precision_at_matched_recall(
                        y_true, list(scores), target, candidates=candidates
                    )
                    if not match["matched"]:
                        assert math.isnan(fast)
                        continue
                    hit = P.confusion_at(y_true, list(scores), match["threshold"])
                    slow = P.reweighted_precision(hit["tp"], hit["fp"], lam)
                    assert fast == pytest.approx(slow, abs=1e-12), (arm, system, target, ratio)


def test_fast_auprc_twin_matches_the_stdlib_kernel_on_the_real_benchmark():
    """The GATED statistic's bootstrap twin, checked against its stdlib kernel.

    Held to the same standard as the matched-recall twin: the published AUPRC comes from
    ``metrics.average_precision_reweighted`` and the numpy version exists only so a
    2,000-replicate block bootstrap finishes. Checked at every sweep ratio on the real data,
    with ties present (the uncalled-item sentinel repeats), so agreement is not only on
    distinct inputs — a twin that mishandled tie grouping would still pass on distinct ones.
    """
    import numpy as np

    from tbox_finder.metrics import average_precision_reweighted

    assert ITEMS_PATH.exists(), f"{ITEMS_PATH} is committed by this step and must exist"
    folded = json.loads(ITEMS_PATH.read_text())
    for arm in folded["arms"]:
        scored = P.arm_items(folded, arm)
        y_true, _ = P.arrays(scored, "two_stage")
        n_pos, n_neg = sum(y_true), len(y_true) - sum(y_true)
        labels = np.asarray(y_true, dtype=np.int64)
        for system in P.SYSTEMS:
            values = P.arrays(scored, system)[1]
            scores = np.asarray(values, dtype=np.float64)
            assert len(set(values)) < len(values), "expected ties in the real scores"
            for ratio in (1, 10, 100, 1_000, 10_000):
                lam = P.prevalence_lambda(n_pos, n_neg, ratio)
                fast = P._fast_average_precision(labels, scores, lam=lam)
                slow = average_precision_reweighted(y_true, values, decoy_weight=lam)
                assert fast == pytest.approx(slow, abs=1e-12), (arm, system, ratio)


def test_reweighted_ap_at_unit_weight_is_exactly_the_shipped_average_precision():
    """``average_precision`` delegates, so this is the guard that the delegation did not
    change a committed number: unit weight must be bit-identical, not merely close."""
    from tbox_finder.metrics import average_precision, average_precision_reweighted

    y = [1, 0, 1, 1, 0, 0, 1]
    s = [0.9, 0.9, 0.8, 0.4, 0.4, 0.2, 0.1]
    assert average_precision_reweighted(y, s, decoy_weight=1.0) == average_precision(y, s)
    # …and a weight above 1 must actually move it, or the parameter is decorative
    assert average_precision_reweighted(y, s, decoy_weight=100.0) < average_precision(y, s)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            average_precision_reweighted(y, s, decoy_weight=bad)


# ──────────────────────────────────────────────────────────────────────────────
# The clause set — each clause broken ALONE, so `all(...)` cannot hide it
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def report() -> dict:
    """The committed report — **asserted present, never skipped**.

    Both artifacts land in the same commit as this file and are plain JSON under
    ``reports/`` (no LFS, no DVC), so "absent" can only mean the commit is broken. A
    ``pytest.skip`` here would take the whole clause suite below with it and read green
    ([[vacuous-test-perturbations]]).
    """
    assert REPORT_PATH.exists(), f"{REPORT_PATH} is committed by this step and must exist"
    return json.loads(REPORT_PATH.read_text())


def test_the_committed_report_is_clean(report):
    assert P.precision_problems(report) == []


def test_a_gated_arm_that_trained_on_the_positives_is_refused(report):
    """The clause the whole §7 decision turns on. Note it is broken on the *exposure count*,
    not on the arm's name: renaming the arm must not be able to move the grade."""
    broken = copy.deepcopy(report)
    broken["arms"][P.GATED_ARM]["exposure"]["n_positives_seen_by_arm"] = 1
    problems = P.precision_problems(broken)
    assert any("not a measurement" in problem for problem in problems)


def test_grading_the_in_sample_arm_is_refused(report):
    """Swapping the gate onto the shipped scanner — the arm that trained on every positive —
    must be refused by re-derivation, not permitted because the label says so."""
    broken = copy.deepcopy(report)
    broken["gated_arm"] = "production"
    broken["arms"]["production"]["gated"] = True
    broken["arms"][P.GATED_ARM]["gated"] = False
    problems = P.precision_problems(broken)
    assert any("trained on" in problem for problem in problems)


def test_a_truncated_run_cannot_certify(report):
    """[[cost-knobs-can-certify]] — a benchmark scored on fewer items than it declares."""
    broken = copy.deepcopy(report)
    broken["arms"][P.GATED_ARM]["population"]["n_items"] -= 1
    assert any("truncated run" in problem for problem in P.precision_problems(broken))


def test_a_verdict_that_contradicts_its_own_auprcs_is_refused(report):
    """The verdict is re-derived from the GATED statistic (ADR-0005 A13: AUPRC), not from
    the matched-recall precisions, which are reported."""
    broken = copy.deepcopy(report)
    broken["arms"][P.GATED_ARM]["passes"] = not broken["arms"][P.GATED_ARM]["passes"]
    assert any("re-derive" in problem for problem in P.precision_problems(broken))


def test_moving_the_matched_recall_precisions_does_not_move_the_verdict(report):
    """The failing direction of A13: on this benchmark the matched-recall read is NEGATIVE
    (−0.039 pp) while the gate passes on AUPRC. Editing the reported precisions must not be
    able to flip `passes`, or the gate would still be the threshold-dependent one."""
    edited = copy.deepcopy(report)
    key = edited["arms"][P.GATED_ARM]["gated_prevalence_key"]
    edited["arms"][P.GATED_ARM]["prevalence"][key]["precision"] = {
        "stage1_only": 0.0,
        "two_stage": 1.0,
    }
    edited["arms"][P.GATED_ARM]["prevalence"][key]["two_stage_beats_stage1_only"] = True
    # `passes` is untouched and the clause set still re-derives it from the AUPRCs, so the
    # verdict clause must NOT fire. Were the gate still threshold-dependent, this edit would
    # make `passes` inconsistent with the published precisions and that clause would fire.
    problems = P.precision_problems(edited)
    assert not any("re-derive" in problem for problem in problems), problems
    assert edited["arms"][P.GATED_ARM]["passes"] == report["arms"][P.GATED_ARM]["passes"]


def test_the_gate_must_name_the_a13_statistic(report):
    broken = copy.deepcopy(report)
    broken["gate"]["statistic"] = "precision_at_matched_recall"
    assert any("A13 statistic" in problem for problem in P.precision_problems(broken))


def test_dropping_the_reported_matched_recall_read_is_refused(report):
    """Retained deliberately: it is the read that FAILS, and hiding it would make the
    published verdict look unqualified."""
    broken = copy.deepcopy(report)
    broken["gate"]["reported_not_gated"] = {}
    assert any("reported, non-gated" in problem for problem in P.precision_problems(broken))


def test_a_gated_prevalence_point_that_does_not_exist_returns_a_problem_not_a_traceback(report):
    """A checker that raises on a payload it did not write is indistinguishable from a
    crashed run (the PR #131 finding)."""
    broken = copy.deepcopy(report)
    broken["arms"][P.GATED_ARM]["gated_prevalence_key"] = "7:1"
    problems = P.precision_problems(broken)
    assert any("nowhere to be read from" in problem for problem in problems)


def test_gate_passes_must_agree_with_the_arm(report):
    broken = copy.deepcopy(report)
    broken["gate"]["passes"] = not broken["gate"]["passes"]
    assert any("gate.passes disagrees" in problem for problem in P.precision_problems(broken))


def test_a_non_invariant_prevalence_sweep_is_refused(report):
    """If the sweep and the selection ever disagree the invariance argument is broken, and
    the gate would silently depend on a number ADR-0005 scopes to P4."""
    broken = copy.deepcopy(report)
    key = next(iter(broken["arms"][P.GATED_ARM]["prevalence"]))
    point = broken["arms"][P.GATED_ARM]["prevalence"][key]
    point["two_stage_beats_stage1_only"] = not point["two_stage_beats_stage1_only"]
    assert any("not invariant" in problem for problem in P.precision_problems(broken))


def test_a_one_point_sweep_cannot_satisfy_the_invariance_clause(report):
    """An all-TRUE single-element set satisfies ``len(set(...)) == 1`` vacuously
    ([[all-true-fixture-cannot-test-a-conjunction]]), so the clause needs its own floor.
    The kept point is the GATED one, so the failure is the sweep's length and not a missing
    gated statistic — which a different clause already refuses."""
    broken = copy.deepcopy(report)
    prevalence = broken["arms"][P.GATED_ARM]["prevalence"]
    key = broken["arms"][P.GATED_ARM]["gated_prevalence_key"]
    broken["arms"][P.GATED_ARM]["prevalence"] = {key: prevalence[key]}
    assert any("shorter than" in problem for problem in P.precision_problems(broken))


def test_is_science_must_be_the_conjunction_not_an_assertion(report):
    broken = copy.deepcopy(report)
    broken["completeness"]["all_benchmark_items_scored"] = False
    assert any("conjunction" in problem for problem in P.precision_problems(broken))


def test_a_per_pool_breakdown_that_does_not_account_for_the_false_positives_is_refused(report):
    """Without this the memorised-``structured_rna`` disclosure would be decorative."""
    broken = copy.deepcopy(report)
    pools = broken["arms"][P.GATED_ARM]["per_pool"]
    pool = next(iter(pools))
    pools[pool]["fp_two_stage"] += 1
    assert any("per-pool" in problem for problem in P.precision_problems(broken))


def test_a_single_block_cannot_carry_a_block_level_ci(report):
    broken = copy.deepcopy(report)
    broken["arms"][P.GATED_ARM]["population"]["n_blocks"] = 1
    assert any("at least two" in problem for problem in P.precision_problems(broken))


def test_an_absolute_path_anywhere_is_refused(report):
    """Not an allowlist of two prefixes — that shape was a real finding on PR #131."""
    for leaked in ("/home/someone/checkout/x.json", "/exports/people/x/y.json", "/srv/z/a.json"):
        broken = copy.deepcopy(report)
        broken["sources"]["items"] = leaked
        assert any("absolute path" in problem for problem in P.precision_problems(broken))


def test_the_clause_set_is_not_a_universal_refuser(report):
    """The positive control: ``precision_problems`` must return [] on the real report (above)
    AND must not fire on an unrelated cosmetic edit, or every ``assert any(...)`` here would
    pass against a function that refuses everything ([[raises-test-needs-a-positive-control]]).
    """
    cosmetic = copy.deepcopy(report)
    cosmetic["disclosures"] = [*cosmetic["disclosures"], "an extra, harmless sentence"]
    cosmetic["generated_at_utc"] = "2000-01-01T00:00:00+00:00"
    assert P.precision_problems(cosmetic) == []


def test_a_missing_gated_arm_is_refused_at_the_producer():
    with pytest.raises(P.PrecisionError, match="no arm 'twin'"):
        P.precision_gain(
            {"arms": ["production"], "items": [], "benchmark": {"scope": {}}},
            stage2_operating_point=0.5,
            decoy_prevalence=100,
            prevalence_sweep=P.PREVALENCE_SWEEP,
            recall_grid=P.RECALL_GRID,
            n_boot=10,
            seed=42,
        )


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end on a constructed payload — the producer, not the artifact
# ──────────────────────────────────────────────────────────────────────────────
def _folded(n_pos: int = 40, n_neg: int = 40) -> dict:
    """A benchmark where Stage-2 is strictly the better ranker: both systems see the same
    candidates, but Stage-1's score is noisy on the decoys."""
    items = []
    for index in range(n_pos):
        items.append(
            {
                "contig_id": f"pos:{index:03d}",
                "label": 1,
                "pool": "corpus",
                "block": f"cluster:{index // 4}",
                "seen_by": {"twin": False, "production": True},
                # The two arms are deliberately NOT given the same scores: with identical
                # columns an arm mix-up inside ``arm_items`` is unobservable and every test
                # here stays green under one. ``production`` is the in-sample arm, so it
                # ranks the corpus higher, exactly as the real tables do.
                "arms": {
                    "production": {
                        "n_rows": 1,
                        "stage1_only": 0.70 + index / 500,
                        "two_stage": 0.95,
                    },
                    "twin": {"n_rows": 1, "stage1_only": 0.60 + index / 500, "two_stage": 0.95},
                },
            }
        )
    for index in range(n_neg):
        items.append(
            {
                "contig_id": f"dec:{index:03d}",
                "label": 0,
                "pool": "gc_background" if index % 2 else "structured_rna",
                "block": f"singleton:dec{index}",
                "seen_by": {"twin": index % 2 == 0, "production": True},
                "arms": {
                    "production": {
                        "n_rows": 1,
                        "stage1_only": 0.59 + index / 500,
                        "two_stage": 0.05,
                    },
                    "twin": {"n_rows": 1, "stage1_only": 0.61 + index / 500, "two_stage": 0.05},
                },
            }
        )
    return {
        "schema_version": P.SCHEMA_VERSION,
        "step": P.STEP,
        "arms": ["production", "twin"],
        "benchmark": {
            "scope": {
                "n_items": n_pos + n_neg,
                "n_positives": n_pos,
                "n_negatives": n_neg,
                # What the minter writes; the geometry disclosure is read from it rather
                # than spelled out in the sentence (CodeRabbit, PR #133 round 2), and the
                # block count is re-derived against the arms' own resampling unit.
                "geometry": "1024-nt scan windows",
                "n_blocks": len({item["block"] for item in items}),
                "negatives_by_pool": {"gc_background": n_neg // 2, "structured_rna": n_neg // 2},
                "seen_by_counts": {
                    "twin": {"positives": 0, "negatives": n_neg // 2},
                    "production": {"positives": n_pos, "negatives": n_neg},
                },
            },
            "source": {},
            "embedded_by_pool": {},
        },
        "items": items,
    }


def test_end_to_end_on_a_constructed_benchmark_passes_and_self_validates():
    report = P.precision_gain(
        _folded(),
        stage2_operating_point=0.5,
        decoy_prevalence=100,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=50,
        seed=42,
        generated_at_utc="2026-08-13T00:00:00+00:00",
    )
    assert P.precision_problems(report) == []
    assert report["gate"]["passes"] is True
    assert report["is_science"] is True
    assert report["gated_arm"] == "twin"
    # Stage-2 separates perfectly here, Stage-1 does not: the AUPRC gain must be positive.
    assert report["gate"]["statistic"] == "auprc_at_pinned_prevalence"
    assert report["gate"]["gain_pp"] > 0
    assert report["gate"]["auprc"]["two_stage"] > report["gate"]["auprc"]["stage1_only"]


def test_a_stage2_that_adds_nothing_fails_the_gate_rather_than_erroring():
    """The failing direction of the gate, which no committed artifact can exercise if the
    real one passes ([[gate-predicates-both-directions]] in spirit)."""
    folded = _folded()
    for item in folded["items"]:
        for arm in item["arms"]:
            # Stage-2 becomes a monotone copy of Stage-1 => identical ranking, no gain.
            item["arms"][arm]["two_stage"] = item["arms"][arm]["stage1_only"]
    report = P.precision_gain(
        folded,
        stage2_operating_point=0.5,
        decoy_prevalence=100,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=40,
        seed=42,
        generated_at_utc="2026-08-13T00:00:00+00:00",
    )
    assert report["gate"]["passes"] is False
    assert report["gate"]["gain_pp"] == pytest.approx(0.0)
    # A failing gate is still a *valid* report — it must not be refused as malformed, or a
    # fail would be indistinguishable from a broken run.
    assert P.precision_problems(report) == []


# ──────────────────────────────────────────────────────────────────────────────
# The block scheme (ADR-0005 D5) — tested at the PRODUCER, not only on the artifact
# ──────────────────────────────────────────────────────────────────────────────
def _minter():
    """Load the mint script by path — `scripts/` is not a package."""
    import importlib.util

    path = REPO / "scripts" / "mint_p3_16_benchmark.py"
    spec = importlib.util.spec_from_file_location("mint_p3_16_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_parented_decoy_blocks_on_its_parents_cluster_not_its_host():
    """The regression CodeRabbit caught on PR #133: the documented rule existed as a
    function that **nothing called**, so every negative keyed on its unique host and a
    dinuc shuffle was treated as independent of the locus it is a permutation of."""
    mint = _minter()
    parented = {mint.negatives_mod.SOURCE_RECORD_ID_COL: "rec-a"}
    assert mint.decoy_block_id(parented, "host-1", {"rec-a": 77}) == "cluster:77"
    # a parent the cluster map does not know falls back to the host — never to a shared
    # `None` bucket ([[nulls-inflate-block-counts]])
    assert mint.decoy_block_id(parented, "host-1", {}) == "host:host-1"
    for empty in (None, "", "   "):
        row = {mint.negatives_mod.SOURCE_RECORD_ID_COL: empty}
        assert mint.decoy_block_id(row, "host-2", {"rec-a": 77}) == "host:host-2"


def test_the_committed_item_table_realises_that_block_scheme():
    """The artifact half of the pair above: every dinuc decoy must share a cluster block,
    and the parentless pools must not."""
    folded = json.loads(ITEMS_PATH.read_text())
    by_pool: dict[str, set[str]] = {}
    for item in folded["items"]:
        by_pool.setdefault(item["pool"], set()).add(item["block"].split(":")[0])
    assert by_pool["dinuc_shuffled"] == {"cluster"}
    for pool in ("gc_background", "structured_rna", "leader_decoy"):
        assert by_pool[pool] == {"host"}, pool
    # and a dinuc decoy must actually LAND in a block a positive also occupies, or the
    # inheritance is nominal
    positive_blocks = {i["block"] for i in folded["items"] if i["label"] == 1}
    dinuc_blocks = {i["block"] for i in folded["items"] if i["pool"] == "dinuc_shuffled"}
    assert dinuc_blocks & positive_blocks


# ──────────────────────────────────────────────────────────────────────────────
# The Stage-1-threshold sensitivity annex, and the disclosures it sits beside
# (CodeRabbit, PR #133 round 2)
# ──────────────────────────────────────────────────────────────────────────────
def _swept_point(threshold: float, *, passes: bool = True) -> dict:
    """One swept point shaped like the sub-report ``--sensitivity`` reads off disk."""
    return {
        "declared": threshold,
        "sha256": f"{int(threshold * 100):064d}",
        "report": {
            "sources": {"stage1_threshold": threshold},
            "gate": {
                "auprc": {"stage1_only": 0.78, "two_stage": 0.95},
                "gain_pp": 16.5,
                "passes": passes,
                "reported_not_gated": {"matched_recall": {"gain_pp": 6.3}},
            },
        },
    }


def test_the_annex_carries_the_threshold_this_report_was_measured_at():
    block = P.stage1_threshold_sensitivity(
        [_swept_point(0.7), _swept_point(0.9)], base_threshold=0.5, base_passes=True
    )
    assert block["base"]["stage1_threshold"] == 0.5
    assert block["base"]["source"] == "this report"


def test_the_invariance_read_spans_the_base_verdict_not_only_the_swept_points():
    """The bite. Swept points that agree with each other but not with the report they annex
    must not read as invariant: that shape published ``verdict_invariant: true`` over a set
    excluding the only threshold the verdict is actually graded at."""
    agreeing = [_swept_point(0.7), _swept_point(0.9)]
    assert P.stage1_threshold_sensitivity(agreeing, base_threshold=0.5, base_passes=True)[
        "verdict_invariant"
    ]
    assert not P.stage1_threshold_sensitivity(agreeing, base_threshold=0.5, base_passes=False)[
        "verdict_invariant"
    ]


def test_brackets_base_says_whether_the_sweep_encloses_the_base_threshold():
    """Stated rather than implied: this run's sweep sits entirely ABOVE its base."""
    one_side = P.stage1_threshold_sensitivity(
        [_swept_point(0.7), _swept_point(0.9)], base_threshold=0.5, base_passes=True
    )
    assert one_side["brackets_base"] is False
    enclosing = P.stage1_threshold_sensitivity(
        [_swept_point(0.3), _swept_point(0.9)], base_threshold=0.5, base_passes=True
    )
    assert enclosing["brackets_base"] is True


def test_an_annex_whose_base_is_not_this_reports_threshold_is_refused(report):
    broken = copy.deepcopy(report)
    broken["stage1_threshold_sensitivity"]["base"]["stage1_threshold"] = 0.9
    assert any("measured at" in problem for problem in P.precision_problems(broken))


def test_an_annex_whose_base_verdict_contradicts_the_gate_is_refused(report):
    broken = copy.deepcopy(report)
    base = broken["stage1_threshold_sensitivity"]["base"]
    base["passes"] = not broken["gate"]["passes"]
    assert any("base verdict disagrees" in problem for problem in P.precision_problems(broken))


def _disclosure_inputs(*, admitted: int, retained: int, geometry: str) -> tuple[dict, dict]:
    """The smallest ``benchmark``/``arms`` pair ``disclosures`` reads."""
    benchmark = {
        "scope": {
            "geometry": geometry,
            "n_positives": 100,
            "n_negatives": 60,
            "negatives_by_pool": {"leader_decoy": 1},
            "seen_by_counts": {
                P.GATED_ARM: {"positives": 0, "negatives": 20},
                "production": {"positives": 100, "negatives": 60},
            },
        }
    }
    arm = {
        "population": {
            "observed_decoy_ratio": 0.6,
            "n_candidate_items": 100 + admitted,
            "n_positive_candidates": 100,
        },
        "matched_recall": {"two_stage": {"fp": retained}},
    }
    return benchmark, {P.GATED_ARM: arm, "production": copy.deepcopy(arm)}


def test_the_geometry_disclosure_is_read_from_the_benchmark_not_spelled_out():
    benchmark, arms = _disclosure_inputs(
        admitted=6, retained=4, geometry="4096-nt scan windows (a different pin)"
    )
    line = next(text for text in P.disclosures(benchmark, arms) if "Benchmark items are" in text)
    assert "4096-nt scan windows (a different pin)" in line
    assert "1,024-nt" not in line and "1024-nt" not in line


def test_the_disclosed_admitted_and_retained_counts_track_the_measured_arms():
    """A fixed numeral would publish the same sentence for both of these payloads — which is
    exactly what a regeneration on different inputs would do."""
    benchmark, arms = _disclosure_inputs(admitted=6, retained=4, geometry="1024-nt windows")
    line = next(text for text in P.disclosures(benchmark, arms) if "Benchmark items are" in text)
    assert "admits 6 of 60" in line and "retains 4 at matched recall" in line

    other_benchmark, other_arms = _disclosure_inputs(
        admitted=11, retained=9, geometry="1024-nt windows"
    )
    other = next(
        text for text in P.disclosures(other_benchmark, other_arms) if "Benchmark items are" in text
    )
    assert "admits 11 of 60" in other and "retains 9 at matched recall" in other
    assert other != line


# ──────────────────────────────────────────────────────────────────────────────
# Every published MAGNITUDE re-derived — not only the boolean verdict
# ──────────────────────────────────────────────────────────────────────────────
def test_a_rewritten_headline_gain_is_refused(report):
    """The gain is what a reader quotes. Before this clause it could be rewritten to any
    value at all and the clause set returned []."""
    broken = copy.deepcopy(report)
    broken["gate"]["gain_pp"] = 99.0
    assert any("is not the difference of the AUPRCs" in p for p in P.precision_problems(broken))


def test_the_published_interval_must_be_the_gated_arms_own(report):
    broken = copy.deepcopy(report)
    broken["gate"]["gain_ci"] = {**broken["gate"]["gain_ci"], "lower": 9.9, "upper": 99.0}
    problems = P.precision_problems(broken)
    assert any("not the gated arm's own AUPRC-gain interval" in p for p in problems)


def test_an_interval_resampled_over_a_different_block_count_is_refused(report):
    """The round-1 major moved this number (1,721 → 1,555); nothing re-derived it."""
    broken = copy.deepcopy(report)
    broken["gate"]["gain_ci"]["n_blocks"] = 3
    assert any("blocks but the gated arm scored" in p for p in P.precision_problems(broken))


def test_a_sweep_point_whose_gain_contradicts_its_own_auprcs_is_refused(report):
    """The prevalence curve is the result this step reports; only its PRESENCE was checked."""
    broken = copy.deepcopy(report)
    broken["arms"][P.GATED_ARM]["prevalence"]["10000:1"]["auprc_gain_pp"] = 99.0
    assert any("is not the difference of its own AUPRCs" in p for p in P.precision_problems(broken))


def test_the_gated_arms_headline_gain_must_be_read_at_the_prevalence_it_is_graded_on(report):
    broken = copy.deepcopy(report)
    broken["arms"][P.GATED_ARM]["auprc_gain_pp"] = 99.0
    assert any("is not the gain at the prevalence point" in p for p in P.precision_problems(broken))


def test_a_gate_naming_one_prevalence_and_reading_another_is_refused(report):
    broken = copy.deepcopy(report)
    broken["gate"]["decoy_prevalence"] = 10
    assert any("reads its statistic from" in p for p in P.precision_problems(broken))


def test_the_reported_matched_recall_read_must_be_the_gated_arms_own(report):
    """A13 keeps the failing −0.039 pp published; a second, drifting copy of it would let the
    published read disagree with the arm it was measured on."""
    broken = copy.deepcopy(report)
    broken["gate"]["reported_not_gated"]["matched_recall"]["gain_pp"] = 9.0
    assert any("not the gated arm's own" in p for p in P.precision_problems(broken))


@pytest.mark.parametrize(
    ("path", "value", "needle"),
    [
        (("n_positives",), 60, "but the benchmark manifest declares"),
        (("n_negatives",), 60, "but the benchmark manifest declares"),
        (("n_blocks",), 7, "blocks but the gated arm resampled"),
    ],
)
def test_a_manifest_that_does_not_describe_what_was_graded_is_refused(report, path, value, needle):
    """``n_items`` alone bound the graded population to the manifest, so a relabelled corpus
    could certify the gate while the report published the manifest's composition."""
    broken = copy.deepcopy(report)
    broken["benchmark"]["scope"][path[0]] = value
    assert any(needle in p for p in P.precision_problems(broken))


def test_a_manifest_pool_count_that_no_arm_scored_is_refused(report):
    broken = copy.deepcopy(report)
    pool = next(iter(broken["benchmark"]["scope"]["negatives_by_pool"]))
    broken["benchmark"]["scope"]["negatives_by_pool"][pool] += 3
    problems = P.precision_problems(broken)
    assert any("per-pool negatives sum to" in p for p in problems)
    assert any("the manifest declares" in p for p in problems)


# ──────────────────────────────────────────────────────────────────────────────
# The annex's summary fields, and the refusal an incomplete run now gets
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("n_points", 5, "points and carries"),
        ("all_labels_match", False, "not the conjunction of its own rows"),
        ("verdict_invariant", False, "not re-derivable from its rows and its base"),
        ("brackets_base", True, "brackets_base is not the comparison it names"),
    ],
)
def test_the_annex_summary_fields_are_re_derived_from_its_rows(report, field, value, needle):
    """They were read as written: the same call that built the table also wrote the verdicts
    about it, so a wrong summary was indistinguishable from a right one."""
    broken = copy.deepcopy(report)
    broken["stage1_threshold_sensitivity"][field] = value
    assert any(needle in p for p in P.precision_problems(broken))


def test_a_sensitivity_rows_label_verdict_must_be_the_comparison_it_names(report):
    broken = copy.deepcopy(report)
    broken["stage1_threshold_sensitivity"]["points"][0]["label_matches_report"] = False
    assert any("is not the comparison it names" in p for p in P.precision_problems(broken))


def test_an_incomplete_run_does_not_land_on_the_canonical_path(monkeypatch):
    """``is_science`` was derived and then consumed by nothing: a run with a false
    completeness clause wrote the graded artifact to the phase-exit path and exited 0
    ([[cost-knobs-can-certify]]). Asserts the SIDE EFFECT — which path was written — not just
    the return code ([[guard-runs-after-what-it-guards]])."""
    written: dict[str, dict] = {}
    monkeypatch.setattr(
        P, "write_json", lambda path, payload: written.__setitem__(str(path), payload)
    )
    canonical = "reports/p3/_incomplete_run_never_written.json"
    args = argparse.Namespace(
        # Repo-relative, exactly as the Snakemake rule invokes it: an absolute path would be
        # refused by a different clause and the test would pass for the wrong reason.
        items=str(ITEMS_PATH.relative_to(REPO)),
        out=canonical,
        stage2_operating_point=1.5,  # unreachable ⇒ target_recall_is_positive is False
        decoy_prevalence=100,
        n_boot=40,
        seed=42,
        sensitivity=None,
    )
    assert P._cmd_report(args) == 3
    assert canonical not in written, "an is_science=False report reached the canonical path"
    assert list(written) == ["reports/p3/_incomplete_run_never_written.invalid.json"]
    diverted = written["reports/p3/_incomplete_run_never_written.invalid.json"]
    assert diverted["is_science"] is False
    assert diverted["problems"] == [], "the clause set is clean — it is is_science that refuses"
    assert not Path(canonical).exists()


# ──────────────────────────────────────────────────────────────────────────────
# The interval's own budget, its reweighting, and the unit it resamples
# ──────────────────────────────────────────────────────────────────────────────
def test_an_interval_that_excludes_its_own_point_estimate_is_refused(report):
    broken = copy.deepcopy(report)
    broken["gate"]["gain_ci"] = {**broken["gate"]["gain_ci"], "lower": 20.0, "upper": 30.0}
    assert any("does not contain its" in p for p in P.precision_problems(broken))


@pytest.mark.parametrize("where", ["gate", "arm"])
def test_a_resample_budget_too_small_for_the_requested_tails_is_refused(report, where):
    """``--n-boot`` is a cost knob and nothing read it: at B = 1 the "95 % interval" is one
    resample wide and need not contain its own estimate ([[cost-knobs-can-certify]]). The
    floor is re-derived from the requested ci_level, not a magic number."""
    broken = copy.deepcopy(report)
    node = broken["gate"]["gain_ci"] if where == "gate" else broken["arms"][P.GATED_ARM]["gain_ci"]
    node["n_boot"] = 5
    assert any("cannot resolve a" in p for p in P.precision_problems(broken))


def test_the_intervals_point_must_be_the_gain_it_is_published_beside(report):
    """Otherwise the CI can be resampled at one prevalence and the point estimate read at
    another, with both in the same object."""
    broken = copy.deepcopy(report)
    broken["gate"]["gain_ci"]["point"] = 1.0
    assert any("is not the gain" in p for p in P.precision_problems(broken))


def test_the_gated_points_reweighting_is_re_derived_from_the_composition(report):
    broken = copy.deepcopy(report)
    key = broken["arms"][P.GATED_ARM]["gated_prevalence_key"]
    broken["arms"][P.GATED_ARM]["prevalence"][key]["lambda"] *= 10
    assert any("decoys per positive over this benchmark" in p for p in P.precision_problems(broken))


def test_the_bootstrap_resamples_blocks_not_records():
    """ADR-0005 D5's resampling unit. Keyed on the item id instead, the reported ``n_blocks``
    silently becomes the record count and the interval narrows — a change no artifact-reading
    test can see."""
    folded = _folded(n_pos=16, n_neg=16)
    distinct = len({item["block"] for item in folded["items"]})
    assert distinct < len(folded["items"]), "the fixture must SHARE blocks or this is vacuous"
    report = P.precision_gain(
        folded,
        stage2_operating_point=0.5,
        decoy_prevalence=100,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=40,
        seed=42,
    )
    for name, arm in report["arms"].items():
        assert arm["population"]["n_blocks"] == distinct, name


def test_the_two_arms_are_graded_on_their_own_columns():
    """With byte-identical arm columns in the fixture, swapping the two inside ``arm_items``
    changes nothing observable and the whole suite stays green under the mix-up."""
    report = P.precision_gain(
        _folded(),
        stage2_operating_point=0.5,
        decoy_prevalence=100,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=40,
        seed=42,
    )
    production = report["arms"]["production"]["auprc"]["stage1_only"]
    twin = report["arms"]["twin"]["auprc"]["stage1_only"]
    # The fixture gives `production` the in-sample separation: it ranks corpus above decoys
    # by a wider margin, so its Stage-1-only AUPRC is the higher of the two. A swapped lookup
    # inverts this comparison.
    assert production > twin
