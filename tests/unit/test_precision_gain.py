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
    ``matched_recall_verdict_invariant_across_prevalence`` clause is not a tautology and the
    gate would
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
                # Measured by the minter, never asserted: a selection rule on the host's
                # PARENT does not make the drawn window's DNA unseen (PR #133 round 2).
                "host_pool": {
                    "overlap_with_training_folds": {
                        "k": 32,
                        "method": "verbatim k-mer of the host portion",
                        "by_arm": {"production": 3, "twin": 2},
                        "n_seen_by_any_arm": 3,
                        "n_negatives": n_neg,
                    }
                },
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
            "host_pool": {
                "overlap_with_training_folds": {
                    "k": 32,
                    "by_arm": {"production": 9, P.GATED_ARM: 7},
                    "n_seen_by_any_arm": 9,
                    "n_negatives": 60,
                }
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
    """The fixture gives the two arms DIFFERENT ``stage1_only`` columns on purpose: with
    byte-identical columns, swapping the two inside ``arm_items`` changes nothing observable
    and the whole suite stays green under the mix-up."""
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


# ──────────────────────────────────────────────────────────────────────────────
# The host pool's disclosure must be a measurement, not a selection rule
# ──────────────────────────────────────────────────────────────────────────────
def test_a_host_pool_with_no_measured_overlap_is_refused(report):
    """The published string was "parent_nested_train == False (clean for BOTH Stage-1
    checkpoints)" for a benchmark in which 115 of 692 negatives carry host DNA the production
    arm trained on. A rule about the host's PARENT is not a measurement of the window."""
    broken = copy.deepcopy(report)
    broken["benchmark"]["scope"]["host_pool"].pop("overlap_with_training_folds")
    assert any("no measured overlap" in p for p in P.precision_problems(broken))


def test_a_host_overlap_that_misses_an_arm_is_refused(report):
    broken = copy.deepcopy(report)
    overlap = broken["benchmark"]["scope"]["host_pool"]["overlap_with_training_folds"]
    overlap["by_arm"].pop(P.GATED_ARM)
    assert any("not the arms this" in p for p in P.precision_problems(broken))


@pytest.mark.parametrize("union", [0, 10_000])
def test_a_host_overlap_union_outside_its_own_arms_is_refused(report, union):
    """Below the largest arm it cannot be a union; above their sum it cannot be one either."""
    broken = copy.deepcopy(report)
    broken["benchmark"]["scope"]["host_pool"]["overlap_with_training_folds"][
        "n_seen_by_any_arm"
    ] = union
    assert P.precision_problems(broken), f"union {union!r} was accepted"


@pytest.mark.parametrize(
    "by_arm",
    [
        {"production": 999, "twin": 999},  # more negatives than the benchmark has
        {"production": -5, "twin": -5},  # a count cannot be negative
        {"production": "lots", "twin": 97},  # not a number at all
        {"production": True, "twin": 97},  # a bool is not a count
        {},  # no arm measured
    ],
)
def test_a_host_overlap_count_that_is_not_a_count_of_this_pool_is_refused(report, by_arm):
    """The overlap is a published claim, so it is bounded like one: each count is a number of
    NEGATIVES and cannot exceed the pool it counts. 999 of 692 passed until round 7, and a
    filter that silently dropped a malformed value let one through unexamined."""
    broken = copy.deepcopy(report)
    overlap = broken["benchmark"]["scope"]["host_pool"]["overlap_with_training_folds"]
    overlap["by_arm"] = by_arm
    assert any("negatives" in p for p in P.precision_problems(broken))


def test_a_fractional_negative_count_cannot_serve_as_the_host_overlap_limit(report):
    """Not exploitable on its own — four other clauses already refuse a fractional
    ``n_negatives`` — but the bound a count is checked against must itself be a count, so the
    clause stands on its own rather than on its neighbours."""
    broken = copy.deepcopy(report)
    broken["benchmark"]["scope"]["n_negatives"] = 692.5
    problems = P.precision_problems(broken)
    assert any("is not a count of this benchmark" in p for p in problems), problems


def test_a_host_overlap_measured_over_a_different_corpus_is_refused(report):
    broken = copy.deepcopy(report)
    broken["benchmark"]["scope"]["host_pool"]["overlap_with_training_folds"]["n_negatives"] = 7
    assert any("negatives, not this benchmark's" in p for p in P.precision_problems(broken))


def test_the_committed_report_discloses_the_measured_host_overlap(report):
    """The artifact half: the number a reader is owed must be IN the disclosure, and it must
    be the gated arm's own, not the larger production one."""
    overlap = report["benchmark"]["scope"]["host_pool"]["overlap_with_training_folds"]
    gated = overlap["by_arm"][P.GATED_ARM]
    assert gated > 0, "if this ever reaches zero the disclosure below stops being a caveat"
    line = next(text for text in report["disclosures"] if "HOST windows" in text)
    assert f"{gated} of {report['benchmark']['scope']['n_negatives']} negatives" in line
    assert "clean for BOTH" not in json.dumps(report)


def test_the_checker_returns_a_problem_on_a_truncated_payload_and_never_raises(report):
    """The contract the module states in its own docstring: "a traceback from the validator
    is indistinguishable from a crashed run". Every block a clause reads is emptied, nulled,
    replaced by a list and by a scalar, at the top level and one level down."""
    paths = [
        ("gate",),
        ("arms",),
        ("benchmark",),
        ("completeness",),
        ("sources",),
        ("gated_arm",),
        ("is_science",),
        ("disclosures",),
        # `stage1_threshold_sensitivity` itself is legitimately optional (a report produced
        # without --sensitivity carries none), so only its INNER blocks are swept.
        ("gate", "gain_ci"),
        ("gate", "auprc"),
        ("gate", "reported_not_gated"),
        ("benchmark", "scope"),
        ("stage1_threshold_sensitivity", "points"),
        ("stage1_threshold_sensitivity", "base"),
        *(
            ("arms", arm, block)
            for arm in report["arms"]
            for block in ("population", "prevalence", "per_pool", "matched_recall", "exposure")
        ),
    ]
    checked = 0
    for path in paths:
        for value in (None, {}, [], "x", 0):
            broken = copy.deepcopy(report)
            node = broken
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            problems = P.precision_problems(broken)  # must not raise
            assert problems, f"{'/'.join(path)} = {value!r} was accepted silently"
            checked += 1
    assert checked == len(paths) * 5


def test_no_retyped_node_anywhere_can_make_the_checker_raise(report):
    """The no-raise contract, swept at full width: EVERY node, not only every leaf.

    This test was narrowed three times by three review rounds, each finding the gap the
    previous width left ([[fixed-one-of-two-identical-things]]):

    * round 3 replaced whole top-level BLOCKS — missed a well-formed ``base`` block carrying
      a *string* threshold;
    * round 4 retyped every LEAF — but its list branch walked only ``node[:2]``, and an
      ``int`` with no ``float`` (``10**400``) still passed the ``isinstance`` check;
    * round 5 widened both — and missed an intermediate ``auprc`` block replaced by a scalar,
      because a mapping swapped for a string is neither a leaf nor a top-level block.

    So the traversal now yields every path in the payload, interior nodes included, and each
    is replaced by a scalar, a bool, an unrepresentable int, an empty mapping and an empty
    list. Whether a given replacement is also *caught* is a separate question — much of the
    report is metadata no clause reads — but none of them may take the validator down.
    """

    def paths(node, path=()):
        yield path
        if isinstance(node, dict):
            for key, value in node.items():
                yield from paths(value, (*path, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from paths(value, (*path, index))

    checked = 0
    for path in paths(report):
        if not path or path[:1] in (("provenance",), ("problems",), ("generated_at_utc",)):
            continue
        for bad in ("x", None, True, -1, 10**400, {}, []):
            broken = copy.deepcopy(report)
            node = broken
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = bad
            P.precision_problems(broken)  # must not raise
            checked += 1
    assert checked > 4_000, f"only {checked} replacements — the sweep stopped finding nodes"


def test_the_host_overlap_excludes_the_spliced_decoy():
    """A negative contig carries NO ``truth_start``/``truth_end`` — the insert's location is
    on the ITEM (``splice_phase`` + ``locus_length``). The first implementation fell back to
    the whole window for exactly that case, so it counted the decoy's own k-mers as host DNA
    while the published method string said *"spliced insert excluded"*. Corrected 115/97 to
    112/96 (CodeRabbit, PR #133 round 4 — the thread stayed unresolved until round 8)."""
    mint = _minter()
    insert = "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"  # 36 nt, unique to the decoy
    flank = "ACGT" * 40
    contigs = [
        {
            "contig_id": "dec:000",
            "sequence": flank + insert + flank,
            "truth_start": None,
            "truth_end": None,
        }
    ]
    items = [
        {
            "contig_id": "dec:000",
            "label": 0,
            "splice_phase": len(flank),
            "locus_length": len(insert),
        }
    ]
    # A fold made ONLY of the decoy: with the insert excluded there is no host overlap.
    seen = mint.host_fold_overlap(contigs, items, {"twin": {"r1"}}, {"r1": insert}, k=32)
    assert seen["twin"] == set(), "the spliced decoy was counted as host DNA"
    # …and a fold made of the flank IS host overlap, so the test is not vacuous.
    seen = mint.host_fold_overlap(contigs, items, {"twin": {"r1"}}, {"r1": flank}, k=32)
    assert seen["twin"] == {"dec:000"}


def test_a_negative_with_no_splice_bounds_anywhere_is_refused_not_measured_whole():
    """Refusing beats falling back: a silent fall-back is what made the published claim
    false in the first place."""
    mint = _minter()
    contigs = [
        {"contig_id": "dec:000", "sequence": "ACGT" * 40, "truth_start": None, "truth_end": None}
    ]
    items = [{"contig_id": "dec:000", "label": 0}]
    with pytest.raises(SystemExit, match="no splice bounds"):
        mint.host_fold_overlap(contigs, items, {"twin": {"r1"}}, {"r1": "ACGT" * 40}, k=32)


@pytest.mark.parametrize(
    ("phase", "insert"),
    [
        (-5, 36),
        (900, 400),
        (100, 0),
        (100, -10),
        ("x", 36),
        (None, 36),
        # int() TRUNCATES: 100.5 would slide past the interval check as 100
        (100.5, 36),
        (100, 36.5),
        (True, 36),  # a bool is not a coordinate
    ],
)
def test_splice_bounds_outside_the_contig_are_refused_not_clamped(phase, insert):
    """Python slicing clamps, so a malformed bound would quietly excise the wrong interval
    and publish an overlap count for a host sequence that never existed."""
    mint = _minter()
    contigs = [
        {"contig_id": "dec:000", "sequence": "ACGT" * 100, "truth_start": None, "truth_end": None}
    ]
    items = [{"contig_id": "dec:000", "label": 0, "splice_phase": phase, "locus_length": insert}]
    with pytest.raises(SystemExit):
        mint.host_fold_overlap(contigs, items, {"twin": {"r1"}}, {"r1": "ACGT" * 100}, k=32)


def test_valid_splice_bounds_are_still_accepted():
    """The positive control for the refusals above: a well-formed pair does NOT raise."""
    mint = _minter()
    flank, insert = "ACGT" * 40, "GGGG" * 9
    contigs = [
        {
            "contig_id": "dec:000",
            "sequence": flank + insert + flank,
            "truth_start": None,
            "truth_end": None,
        }
    ]
    items = [
        {
            "contig_id": "dec:000",
            "label": 0,
            "splice_phase": len(flank),
            "locus_length": len(insert),
        }
    ]
    assert mint.host_fold_overlap(contigs, items, {"twin": {"r1"}}, {"r1": flank}, k=32) == {
        "twin": {"dec:000"}
    }


# ══════════════════════════════════════════════════════════════════════════════
# P3-16′-remint — the schema-1.2 re-grade (ADR-0005 A15)
#
# The v2 artifacts do not exist until the orchestrated real-data run lands them, so these
# tests build a small v2 case through the REAL producers: a synthetic manifest, candidate
# tables, payload manifests and familiarity scores are written to a tmp tree, folded by
# `precision fold` (the CLI), and graded by `precision_gain`. The temperature source is the
# real committed GATE-2 report, so the Stage-2 identity binds to real digests. The only
# shim is path normalisation: `repo_relative` resolves against the tmp tree instead of this
# checkout. Each clause is then broken ALONE on a deep copy
# ([[all-true-fixture-cannot-test-a-conjunction]]).
# ══════════════════════════════════════════════════════════════════════════════
V2_ITEMS_REL = "reports/p3/two_stage_precision_items_v2.json"
V2_OUT_REL = "reports/two_stage_precision_v2.json"
GATE2_REL = "reports/gate2_p3_ece.json"
V2_KNOBS = (
    "threshold=0.5,min_span=50,gap_merge=10,min_distinct_elements=2,flank=50,min_order_margin=1"
)
V2_N_BOOT = 40  # the smallest budget the interval clause accepts at a 95 % level


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(root: Path, rel: str, payload) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _v2_items(*, powered: bool) -> tuple[list[dict], dict[str, list[dict]]]:
    """24 positives, 24 gated negatives over three §9.1 pools, one leader_decoy.

    ``powered``: three gated decoys draw a Stage-1 candidate on both arms, interleaved among
    the positives, and Stage-2 rejects them. Otherwise no gated decoy draws one on the gated
    arm, while production alone admits one, because power is counted per arm and the class
    is read on the gated arm only. The leader always draws a candidate. It is reported
    beside the gate, so it must not rescue the power count.
    """
    items: list[dict] = []
    rows: dict[str, list[dict]] = {"twin": [], "production": []}
    for i in range(24):
        cid = f"pos:{i:03d}"
        items.append(
            {
                "contig_id": cid,
                "label": 1,
                "pool": "corpus",
                "block": f"cluster:{i // 3}",
                "seen_by": {"twin": False, "production": True},
                "gated": True,
            }
        )
        posterior = 0.90 + (i % 5) / 100
        rows["twin"].append(_row(cid, 0.60 + i / 100, posterior))
        rows["production"].append(_row(cid, 0.70 + i / 100, posterior))
    pools = ("structured_rna", "gc_background", "dinuc_shuffled")
    for j in range(24):
        pool = pools[j % 3]
        cid = f"dec:{j:03d}"
        items.append(
            {
                "contig_id": cid,
                "label": 0,
                "pool": pool,
                "block": f"host:h{j:02d}",
                "seen_by": {
                    "twin": pool == "structured_rna",
                    "production": pool != "gc_background",
                },
                "gated": True,
            }
        )
        if powered and j < 3:
            for arm in rows:
                rows[arm].append(_row(cid, 0.70 + j / 100, 0.05))
        elif not powered and j == 0:
            rows["production"].append(_row(cid, 0.75, 0.05))
    items.append(
        {
            "contig_id": "dec:leader",
            "label": 0,
            "pool": P.NON_GATED_POOL,
            "block": "host:hL",
            "seen_by": {"twin": False, "production": False},
            "gated": False,
        }
    )
    for arm in rows:
        rows[arm].append(_row("dec:leader", 0.88, 0.30))
    return items, rows


def _v2_scope(items: list[dict], *, screen_sha: str) -> dict:
    def by_pool(group):
        return {
            p: sum(1 for i in group if i["pool"] == p) for p in sorted({i["pool"] for i in group})
        }

    negatives = [i for i in items if i["label"] == 0]
    gated = [i for i in items if i["gated"]]
    gated_negatives = [i for i in gated if i["label"] == 0]
    return {
        "geometry": "1024-nt scan windows (ADR-0005 D3 / PRD §6 pinned geometry)",
        "n_items": len(items),
        "n_positives": sum(1 for i in items if i["label"] == 1),
        "n_negatives": len(negatives),
        "n_blocks": len({i["block"] for i in items}),
        "n_positive_blocks": len({i["block"] for i in items if i["label"] == 1}),
        "negatives_by_pool": by_pool(negatives),
        "n_items_gated": len(gated),
        "n_negatives_gated": len(gated_negatives),
        "negatives_by_pool_gated": by_pool(gated_negatives),
        "n_blocks_gated": len({i["block"] for i in gated}),
        "gated_rule": "ADR-0005 A15 Pin 4 — leader_decoy reported beside the gate",
        "seen_by_counts": {
            arm: {
                "positives": sum(1 for i in items if i["label"] == 1 and i["seen_by"][arm]),
                "negatives": sum(1 for i in items if i["label"] == 0 and i["seen_by"][arm]),
            }
            for arm in ("production", "twin")
        },
        "host_pool": {
            "source": "data/processed/negatives/mining_pool_v0.parquet",
            "selection": "ADR-0005 A15 screen (whole-window N-free 32-mer + coordinate; content)",
            "screen": {
                "report": "reports/p3/host_screen.json",
                "sha256": screen_sha,
                "admitted_list": "reports/p3/host_screen_admitted.txt",
                "n_admitted": 90,
                "threshold_bits": 10.0,
            },
            "composition": {
                "n_hosts": len(negatives),
                "side": {"lead": 17, "trail": len(negatives) - 17},
                "parent_nested_role": {"test": 20, "val": len(negatives) - 20},
            },
            "overlap_with_training_folds": {
                "k": 32,
                "method": "N-free verbatim 32-mer of the host portion",
                "by_arm": {"production": 2, "twin": 0},
                "n_seen_by_any_arm": 2,
                "n_negatives": len(negatives),
            },
        },
    }


def _v2_screen_report() -> dict:
    """Shaped as package A's ``reports/p3/host_screen.json`` (the fields this module reads)."""
    return {
        "schema_version": "1.0",
        "step": "P3-16′-remint",
        "pool": {"sha256": "ab" * 32, "n_candidates": 100},
        "screened_dna": {"k": 32},
        "fold_screen": {
            "n_refused_kmer": 7,
            "n_refused_coord": 2,
            "n_refused": 8,
            "n_survivors": 92,
        },
        "content_screen": {
            "threshold_bits": 10.0,
            "motif": "GGTGGNACCGCG",
            "n_refused_cm": {"RF00230": 2, "TBDB001": 1, "either": 2},
            "n_refused_motif": 1,
            "n_refused": 2,
            "n_admitted": 90,
        },
        "admitted": {"n": 90, "sha256": "cd" * 32},
        "controls": [{"name": f"control_{n}", "fired": True} for n in range(6)],
        "controls_all_fired": True,
        "v1_audit": {"composition": {"side": {"lead": 400, "trail": 292}}},
        "problems": [],
    }


def _build_v2(root: Path, *, powered: bool = True, familiar: bool = False) -> dict:
    """Write a v2 case under ``root``, fold it with the CLI, grade it with the producer."""
    import numpy as np

    root = root.resolve()
    items, rows = _v2_items(powered=powered)
    screen_path = _write(root, "reports/p3/host_screen.json", _v2_screen_report())
    bench_rel = "data/interim/p3_16/benchmark_v2.json"
    bench_path = _write(
        root,
        bench_rel,
        {
            "schema_version": "1.1",
            "step": "P3-16′-remint",
            "items": items,
            "contigs": [],
            "scope": _v2_scope(items, screen_sha=_sha(screen_path)),
            "source": {"dataset": "data/processed/stage2_dataset.parquet", "embedded_by_pool": {}},
        },
    )
    bench_sha = _sha(bench_path)
    shutil_copy = __import__("shutil").copyfile
    (root / "reports").mkdir(parents=True, exist_ok=True)
    shutil_copy(REPO / GATE2_REL, root / GATE2_REL)
    gate2 = json.loads((REPO / GATE2_REL).read_text())
    load = gate2["scoring"]["load"]
    identity = {
        "arm": gate2["scoring"]["arm"],
        "checkpoint_root": "data/processed/checkpoints/stage2_rinalmo",
        "checkpoint_dir": load["checkpoint_dir"],
        "backbone": "multimolecule/rinalmo-giga",
        "revision": load["revision"],
        "dtype": "bfloat16",
        "adapter_sha256": load["adapter_sha256"],
        "heads_sha256": load["heads_sha256"],
        "attn_implementation": "flash_attention_2",
        "batch_size": 32,
        "env_lock": "envs/ml-rna.conda-lock.yml",
    }
    argv = []
    for arm in ("twin", "production"):
        table = _write(root, f"data/interim/p3_16/v2_table_{arm}_t0.5.json", {"rows": rows[arm]})
        payloads = [
            {"row_id": f"{row['contig_id']}#{strand}", "contig_id": row["contig_id"]}
            for row in rows[arm]
            for strand in "+-"
        ]
        payload_path = _write(
            root, f"data/interim/p3_16/v2_payloads_{arm}_t0.5.json", {"payloads": payloads}
        )
        # As P3-16 did: Stage-2 scores the UNION manifest once (here the τ = 0.5 work list
        # plus loci only a stricter τ emits), so its payloads_sha256 is NOT the work list's.
        union = [*payloads, {"row_id": "pos:000#t0.9", "contig_id": "pos:000"}]
        union_path = _write(
            root, f"data/interim/p3_16/v2_payloads_{arm}_union.json", {"payloads": union}
        )
        stage2_path = _write(
            root,
            f"data/interim/p3_16/v2_stage2_{arm}.json",
            {
                **identity,
                "schema_version": "1.1",
                "payloads_sha256": _sha(union_path),
                "shard": None,
                "n_payloads": len(union),
                "logits": {p["row_id"]: 0.0 for p in union},
            },
        )
        stage1_path = root / f"data/interim/p3_16/v2_stage1_{arm}.npz"
        np.savez(stage1_path, **{"pos:000/logits": np.zeros((1, 4, 2)), "pos:000/starts": [0]})
        argv += [
            f"--{arm}-table",
            str(table),
            f"--{arm}-checkpoint",
            f"data/processed/checkpoints/stage1_{arm}/stage1.pt",
            f"--{arm}-stage1",
            str(stage1_path),
            f"--{arm}-stage2",
            str(stage2_path),
            f"--{arm}-payloads",
            str(payload_path),
        ]
    postmint_path = _write(
        root,
        "reports/p3/benchmark_v2_postmint_screen.json",
        {
            # Schema 1.1 and the per-model thresholds of A15 Pin 3 Correction 2. The thresholds
            # and the calibration floor are both carried because the gate now checks WHICH
            # threshold produced `n_violations == 0`, not only that it is zero.
            "schema_version": "1.1",
            "step": "P3-16′-remint",
            "n_violations": 0,
            "n_exempt": 1,
            "threshold_bits": {"RF00230": 20.0, "TBDB001": 10.0},
            "search_threshold_bits": 10.0,
            "n_noise_residue": 5,
            "calibration": {
                "path": "reports/p3/postmint_noise_floor.json",
                "sha256": "1" * 64,
                "null_max_bits": {"RF00230": 16.8, "TBDB001": 9.9},
                "n_null_sequences": {"RF00230": 3073, "TBDB001": 3073},
            },
            "problems": [],
            "benchmark": {"sha256": bench_sha},
        },
    )
    familiarity_path = _write(
        root,
        "data/interim/p3_16/v2_familiarity.json",
        {
            "schema_version": "1.0",
            "step": "P3-16′-remint",
            "benchmark": {"path": bench_rel, "sha256": bench_sha},
            "k": 32,
            "arms": {arm: {"n_kmers": 1000} for arm in ("twin", "production")},
            "items": {
                item["contig_id"]: {
                    arm: (
                        0.40 + index / 1000 if item["label"] == 1 else (0.01 if familiar else 0.90)
                    )
                    for arm in ("twin", "production")
                }
                for index, item in enumerate(items)
            },
        },
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(P, "repo_relative", lambda path: str(Path(path).resolve().relative_to(root)))
        rc = P.main(
            [
                "fold",
                "--benchmark",
                str(bench_path),
                "--out",
                str(root / V2_ITEMS_REL),
                "--stage1-threshold",
                "0.5",
                "--temperature-source",
                str(root / GATE2_REL),
                "--host-screen",
                str(screen_path),
                "--postmint",
                str(postmint_path),
                "--familiarity",
                str(familiarity_path),
                "--rule-knobs",
                V2_KNOBS,
                *argv,
            ]
        )
    assert rc == 0
    folded = json.loads((root / V2_ITEMS_REL).read_text())
    report = P.precision_gain(
        folded,
        stage2_operating_point=0.5,
        decoy_prevalence=100,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=V2_N_BOOT,
        seed=42,
        sources={**folded["source"], "items": V2_ITEMS_REL},
        generated_at_utc="2026-10-11T00:00:00+00:00",
    )
    report["provenance"] = {"inputs": {V2_ITEMS_REL: _sha(root / V2_ITEMS_REL)}}
    return {"root": root, "folded": folded, "report": report}


@pytest.fixture(scope="module")
def v2_case(tmp_path_factory) -> dict:
    """Powered (three gated decoys reach Stage-2), familiarity NOT rivalling Stage-1."""
    return _build_v2(tmp_path_factory.mktemp("v2_powered"), powered=True, familiar=False)


@pytest.fixture(scope="module")
def v2_under_case(tmp_path_factory) -> dict:
    """Underpowered (no gated decoy reaches Stage-2 on the gated arm), familiarity rivalling."""
    return _build_v2(tmp_path_factory.mktemp("v2_under"), powered=False, familiar=True)


@pytest.fixture
def v2(v2_case) -> dict:
    return copy.deepcopy(v2_case["report"])


@pytest.fixture
def v2_under(v2_under_case) -> dict:
    return copy.deepcopy(v2_under_case["report"])


def _fires(report: dict, needle: str) -> list[str]:
    problems = P.precision_problems(report)
    assert any(needle in problem for problem in problems), problems
    return problems


# ── the v1 regression: the 1.1 path is untouched, bit for bit ─────────────────
def test_the_v1_path_reproduces_the_committed_v1_report(report):
    """A15 Pin 5 keeps v1 byte-unchanged as the record of the withdrawn grade, and Pin 6 keeps
    the 1.1 clause set passing on it. The strongest form of both: today's producer, given the
    committed v1 item table and the committed rule, regenerates the committed report
    EXACTLY, block for block. The only fields it does not produce are the three `_cmd_report`
    adds after it (problems, provenance, the τ annex)."""
    folded = json.loads(ITEMS_PATH.read_text())
    assert not P.folded_declares_gated_subset(folded)
    rule = report["rule"]
    regenerated = P.precision_gain(
        folded,
        stage2_operating_point=rule["stage2_operating_point"],
        decoy_prevalence=rule["decoy_prevalence"],
        prevalence_sweep=rule["prevalence_sweep"],
        recall_grid=rule["recall_grid"],
        n_boot=rule["n_boot"],
        seed=rule["seed"],
        sources=dict(report["sources"]),
        generated_at_utc=report["generated_at_utc"],
    )
    assert regenerated["schema_version"] == P.V1_SCHEMA_VERSION == report["schema_version"]
    for key, value in regenerated.items():
        assert value == report[key], key
    assert set(report) - set(regenerated) == {
        "problems",
        "provenance",
        "stage1_threshold_sensitivity",
    }


def test_the_committed_v1_artifacts_validate_under_every_validator(report):
    assert P.precision_problems(report) == []
    # 1.1 records no interim digests; its bindings are P3-17a's fixed-path sources.
    assert P.precision_input_problems(report) == []
    for tau in ("07", "09"):
        annex = json.loads(
            (REPO / "reports" / "p3" / f"two_stage_precision_tau{tau}.json").read_text()
        )
        assert annex["schema_version"] == P.V1_SCHEMA_VERSION
        assert P.precision_problems(annex) == [], tau


def test_a_v1_refold_is_identical_to_the_committed_item_table():
    """The legacy fold must not grow a single key: a manifest that predates A15 folds to 1.1,
    with no `gated` flag and no manifest version, exactly as the committed items were. The
    tables are rebuilt from the committed items themselves (each item's n_rows rows at its
    max score), so this runs on the real benchmark composition in CI."""
    committed = json.loads(ITEMS_PATH.read_text())
    manifest = {
        "schema_version": "1.0",
        "items": [
            {k: item[k] for k in ("contig_id", "label", "pool", "block", "seen_by")}
            for item in committed["items"]
        ],
        "scope": committed["benchmark"]["scope"],
        "source": {
            **committed["benchmark"]["source"],
            "embedded_by_pool": committed["benchmark"]["embedded_by_pool"],
        },
    }
    tables = {
        arm: [
            _row(
                item["contig_id"], item["arms"][arm]["stage1_only"], item["arms"][arm]["two_stage"]
            )
            for item in committed["items"]
            for _ in range(item["arms"][arm]["n_rows"])
        ]
        for arm in committed["arms"]
    }
    refolded = P.fold_arms(manifest, tables)
    assert refolded == {key: value for key, value in committed.items() if key != "source"}


# ── the fold: gated flags, refusals, and the v2 inputs ───────────────────────
def test_a_gated_manifest_folds_to_schema_1_2_with_its_flags(v2_case):
    folded = v2_case["folded"]
    assert folded["schema_version"] == P.SCHEMA_VERSION == "1.2"
    assert folded["benchmark"]["manifest_schema_version"] == "1.1"
    flags = {item["contig_id"]: item["gated"] for item in folded["items"]}
    assert flags["dec:leader"] is False
    assert sum(1 for flag in flags.values() if flag is False) == 1
    # every v2 input recorded repo-relative with its digest
    interim = folded["source"]["interim_sha256"]
    for key in P.V2_SOURCE_FILES:
        assert folded["source"][key] in interim, key
    assert interim[GATE2_REL] == _sha(REPO / GATE2_REL)
    assert folded["source"]["rule_knobs"]["min_span"] == 50
    assert folded["source"]["benchmark_sha256"] == interim[folded["source"]["benchmark"]]


def test_a_gated_manifest_item_without_a_strict_bool_flag_is_refused():
    items, rows = _v2_items(powered=True)
    for bad in (None, 1, "true"):
        broken = copy.deepcopy(items)
        broken[0]["gated"] = bad
        if bad is None:
            del broken[0]["gated"]
        manifest = {"schema_version": "1.1", "items": broken, "scope": {}, "source": {}}
        with pytest.raises(P.PrecisionError, match="strict bool"):
            P.fold_arms(manifest, rows)
    # positive control: the same manifest with its flags intact folds
    assert P.fold_arms({"schema_version": "1.1", "items": items, "scope": {}}, rows)["items"]


def test_an_old_manifest_carrying_gated_flags_is_refused():
    items, rows = _v2_items(powered=True)
    with pytest.raises(P.PrecisionError, match="predates"):
        P.fold_arms({"schema_version": "1.0", "items": items, "scope": {}}, rows)


def test_a_stripped_version_cannot_downgrade_a_gated_table_to_1_1(v2_case):
    """At the PRODUCER: gated content without the manifest version is refused, not graded
    under the clause set that would put the leader back in the denominator."""
    folded = copy.deepcopy(v2_case["folded"])
    del folded["benchmark"]["manifest_schema_version"]
    with pytest.raises(P.PrecisionError, match="does not declare"):
        P.precision_gain(
            folded,
            stage2_operating_point=0.5,
            decoy_prevalence=100,
            prevalence_sweep=P.PREVALENCE_SWEEP,
            recall_grid=P.RECALL_GRID,
            n_boot=V2_N_BOOT,
            seed=42,
        )


def test_a_downgraded_report_is_refused_by_the_validator(v2):
    """…and at the VALIDATOR: a v2 manifest graded under a pre-1.2 schema label."""
    broken = copy.deepcopy(v2)
    broken["schema_version"] = "1.1"
    _fires(broken, "graded under the 1.2 clause set or not at all")


@pytest.mark.parametrize("dropped", ["--postmint", "--familiarity", "--twin-stage2"])
def test_a_v2_fold_missing_an_input_is_refused(v2_case, tmp_path, dropped):
    """Every v2 input is required together: a 1.2 report binds each of them."""
    root = v2_case["root"]
    args = [
        "fold",
        "--benchmark",
        str(root / "data/interim/p3_16/benchmark_v2.json"),
        "--out",
        str(tmp_path / "items.json"),
        "--stage1-threshold",
        "0.5",
        "--twin-table",
        str(root / "data/interim/p3_16/v2_table_twin_t0.5.json"),
        "--production-table",
        str(root / "data/interim/p3_16/v2_table_production_t0.5.json"),
        "--twin-checkpoint",
        "a",
        "--production-checkpoint",
        "b",
        "--postmint",
        str(root / "reports/p3/benchmark_v2_postmint_screen.json"),
        "--familiarity",
        str(root / "data/interim/p3_16/v2_familiarity.json"),
        "--twin-stage2",
        str(root / "data/interim/p3_16/v2_stage2_twin.json"),
    ]
    index = args.index(dropped)
    del args[index : index + 2]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(P, "repo_relative", lambda path: str(Path(path).resolve().relative_to(root)))
        with pytest.raises(P.PrecisionError, match=dropped):
            P.main(args)
    assert not (tmp_path / "items.json").exists()


def test_a_v1_fold_given_a_v2_input_is_refused(tmp_path):
    items, rows = _v2_items(powered=True)
    for item in items:
        del item["gated"]
    bench = _write(tmp_path, "bench.json", {"schema_version": "1.0", "items": items, "scope": {}})
    tables = {arm: _write(tmp_path, f"{arm}.json", {"rows": rows[arm]}) for arm in rows}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            P, "repo_relative", lambda path: str(Path(path).resolve().relative_to(tmp_path))
        )
        args = [
            "fold",
            "--benchmark",
            str(bench),
            "--twin-table",
            str(tables["twin"]),
            "--production-table",
            str(tables["production"]),
            "--twin-checkpoint",
            "a",
            "--production-checkpoint",
            "b",
            "--stage1-threshold",
            "0.5",
            "--out",
            str(tmp_path / "out.json"),
        ]
        with pytest.raises(P.PrecisionError, match="declares no gated subset"):
            P.main([*args, "--rule-knobs", V2_KNOBS])
        # positive control: the same v1 fold without it succeeds and stays 1.1
        assert P.main(args) == 0
    assert json.loads((tmp_path / "out.json").read_text())["schema_version"] == "1.1"


def test_payload_counts_refuse_a_contig_the_benchmark_does_not_hold(v2_case):
    folded = copy.deepcopy(v2_case["folded"])
    with pytest.raises(P.PrecisionError, match="does not\\s+hold"):
        P.attach_payload_counts(folded, "twin", [{"contig_id": "ghost", "row_id": "x"}])
    P.attach_payload_counts(folded, "twin", [{"contig_id": "pos:000", "row_id": "x"}])
    assert {i["contig_id"]: i["arms"]["twin"]["n_payloads"] for i in folded["items"]}[
        "pos:000"
    ] == 1


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (lambda f: f["benchmark"].update(sha256="0" * 64), "scores benchmark"),
        (lambda f: f["items"].pop("pos:000"), "of the benchmark's"),
        (lambda f: f["items"]["pos:000"].update(twin=1.5), "not a fraction"),
        (lambda f: f["items"]["pos:000"].pop("production"), "not a fraction"),
    ],
)
def test_familiarity_scores_for_a_different_item_set_are_refused(v2_case, edit, needle):
    folded = copy.deepcopy(v2_case["folded"])
    familiarity = json.loads(
        (v2_case["root"] / "data/interim/p3_16/v2_familiarity.json").read_text()
    )
    bench_sha = folded["source"]["benchmark_sha256"]
    P.attach_familiarity(copy.deepcopy(folded), familiarity, benchmark_sha256=bench_sha)
    edit(familiarity)
    with pytest.raises(P.PrecisionError, match=needle):
        P.attach_familiarity(folded, familiarity, benchmark_sha256=bench_sha)


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        (V2_KNOBS + ",threshold=0.7", "twice"),
        (V2_KNOBS.replace(",flank=50", ""), "missing"),
        (V2_KNOBS + ",window=512", "unknown"),
        (V2_KNOBS.replace("min_span=50", "min_span=abc"), "not a number"),
        (V2_KNOBS.replace("flank=50", "flank=nan"), "not finite"),
        (V2_KNOBS.replace("flank=50", "flank"), "not '<name>=<value>'"),
    ],
)
def test_rule_knobs_are_exactly_the_d3_set(text, needle):
    with pytest.raises(P.PrecisionError, match=needle):
        P.parse_rule_knobs(text)


def test_rule_knobs_parse_to_typed_values():
    knobs = P.parse_rule_knobs(V2_KNOBS)
    assert list(knobs) == list(P.RULE_KNOB_NAMES)
    assert knobs["threshold"] == 0.5 and isinstance(knobs["min_span"], int)


# ── the producer: the gated subset, both verdict classes ─────────────────────
def test_a_powered_v2_report_is_clean_and_the_leader_is_beside_the_gate(v2, v2_case):
    assert P.precision_problems(v2) == []
    assert v2["schema_version"] == "1.2"
    assert v2["gate"]["verdict_class"] == "powered"
    assert "underpowered_statement" not in v2["gate"]
    twin = v2["arms"][P.GATED_ARM]
    # The leader is minted and scored but NOT graded: the gated population is the manifest's
    # gated subset, and its scores are published beside the gate on both arms.
    assert twin["population"]["n_items"] == v2["benchmark"]["scope"]["n_items_gated"]
    assert P.NON_GATED_POOL not in twin["per_pool"]
    leader = v2["reported_not_gated"]["non_gated_items"][P.GATED_ARM]
    assert [row["contig_id"] for row in leader] == ["dec:leader"]
    assert leader[0]["two_stage"] == pytest.approx(0.30)
    assert v2["power"]["by_arm"][P.GATED_ARM]["n_gated_negatives_with_stage1_candidate"] == 3
    # each candidate decoy has two payloads (both strands); no other gated decoy has one
    assert v2["power"]["by_arm"][P.GATED_ARM]["n_gated_negatives_with_stage2_payload"] == 3
    assert v2["gate"]["passes"] is True
    assert v2["familiarity_baseline"]["gated_arm_rivals_stage1"] is False


def test_an_underpowered_report_is_valid_ties_and_fails(v2_under):
    """A15 Pin 5's pre-registered expectation, end to end on the producer: no gated negative
    reaches Stage-2 on the gated arm, so both systems produce one PR curve. The gain is
    EXACTLY zero, A13's strict `>` fails it, and the report is valid and says why."""
    assert P.precision_problems(v2_under) == []
    gate = v2_under["gate"]
    assert gate["gain_pp"] == 0.0
    assert gate["auprc"]["two_stage"] == gate["auprc"]["stage1_only"]
    assert gate["passes"] is False
    assert gate["verdict_class"] == "underpowered"
    assert gate["underpowered_statement"] == P.UNDERPOWERED_STATEMENT
    power = v2_under["power"]["by_arm"]
    assert power[P.GATED_ARM]["n_gated_negatives_with_stage1_candidate"] == 0
    assert power[P.GATED_ARM]["n_gated_negatives_with_stage2_payload"] == 0
    # production admitted one decoy: power is per arm, the class is the gated arm's
    assert power["production"]["n_gated_negatives_with_stage1_candidate"] == 1
    # the leader drew a candidate on the gated arm and must not rescue the count
    assert v2_under["reported_not_gated"]["non_gated_items"][P.GATED_ARM][0]["n_rows"] == 1
    assert v2_under["is_science"] is True


def test_the_leader_is_filtered_at_the_call_site_not_inside_arm_metrics(v2_case):
    """`arm_metrics` is unchanged (P3-17a re-derives through it): it still grades whatever it
    is given. Handed the FULL item set, it counts the leader; the report's population does
    not, so the exclusion is the call site's."""
    folded = v2_case["folded"]
    full = P.arm_metrics(
        P.arm_items(folded, P.GATED_ARM),
        P.GATED_ARM,
        stage2_operating_point=0.5,
        decoy_prevalence=100,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=V2_N_BOOT,
        seed=42,
    )
    graded = v2_case["report"]["arms"][P.GATED_ARM]
    assert full["population"]["n_items"] == graded["population"]["n_items"] + 1
    assert P.NON_GATED_POOL in full["per_pool"]


def test_verdict_class_never_overrides_the_a13_verdict(v2, v2_under):
    """`gate.passes` is A13's strict `>`, re-derived from the AUPRCs; the class is recorded
    beside it and moves nothing. Forcing an underpowered tie to pass is refused by the A13
    re-derivation, and a powered pass stays a pass whatever the class field says."""
    forced = copy.deepcopy(v2_under)
    forced["gate"]["passes"] = forced["arms"][P.GATED_ARM]["passes"] = True
    _fires(forced, "re-derive to False")
    relabelled = copy.deepcopy(v2)
    relabelled["power"]["verdict_class"] = relabelled["gate"]["verdict_class"] = "underpowered"
    problems = P.precision_problems(relabelled)
    assert not any("re-derive to" in p for p in problems)  # passes is untouched
    assert any("power.verdict_class" in p for p in problems)  # the class itself is refused


def test_the_v2_clause_set_is_not_a_universal_refuser(v2, v2_under):
    for clean in (v2, v2_under):
        cosmetic = copy.deepcopy(clean)
        cosmetic["disclosures"] = [*cosmetic["disclosures"], "an extra, harmless sentence"]
        cosmetic["generated_at_utc"] = "2000-01-01T00:00:00+00:00"
        assert P.precision_problems(cosmetic) == []


# ── each 1.2 clause broken ALONE ─────────────────────────────────────────────
def test_a_gated_report_without_the_declared_subset_is_refused(v2):
    broken = copy.deepcopy(v2)
    del broken["benchmark"]["scope"]["n_items_gated"]
    _fires(broken, "none is declared")


def test_completeness_compares_against_the_gated_counts(v2):
    """The four completeness clauses re-expressed against the `*_gated` fields: a gated count
    that disagrees with the graded population is refused by the SAME clause that refused a
    truncated run under 1.1."""
    broken = copy.deepcopy(v2)
    broken["benchmark"]["scope"]["n_items_gated"] += 1
    _fires(broken, "a truncated run cannot certify")


def test_the_gated_pools_must_be_the_benchmark_pools_minus_the_leader(v2):
    broken = copy.deepcopy(v2)
    broken["benchmark"]["scope"]["negatives_by_pool"]["structured_rna"] += 1
    broken["benchmark"]["scope"]["n_negatives"] += 1
    _fires(broken, "are not its pools minus")


def test_the_gated_counts_must_be_the_full_counts_less_the_leader(v2):
    broken = copy.deepcopy(v2)
    broken["benchmark"]["scope"]["n_items"] += 1
    _fires(broken, "less the")


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (lambda r: r["reported_not_gated"]["non_gated_items"].pop("production"), "every arm"),
        (
            lambda r: r["reported_not_gated"]["non_gated_items"]["twin"][0].update(
                pool="structured_rna"
            ),
            "is reported beside the gate",
        ),
        (lambda r: r["reported_not_gated"]["non_gated_items"].update(twin=[]), "0 non-gated"),
        (
            lambda r: r["reported_not_gated"]["non_gated_items"]["twin"][0].update(two_stage=None),
            "both systems' scores",
        ),
        (
            lambda r: r["reported_not_gated"]["non_gated_items"]["twin"][0].update(n_rows=0),
            "both systems' scores",
        ),
        (
            lambda r: r["reported_not_gated"]["non_gated_items"]["twin"][0].update(
                contig_id="dec:other"
            ),
            "different non-gated items",
        ),
    ],
)
def test_the_non_gated_set_is_exactly_the_leader_with_both_scores(v2, edit, needle):
    broken = copy.deepcopy(v2)
    edit(broken)
    _fires(broken, needle)


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("n_gated_negatives", 25, "but the arm graded"),
        ("n_gated_negatives_with_stage1_candidate", 2, "re-derives"),
        ("n_gated_negatives_with_stage2_payload", None, "is not between"),
        ("n_gated_negatives_with_stage2_payload", 2, "is not between"),
        ("n_gated_negatives_with_stage2_payload", True, "is not between"),
    ],
)
def test_the_power_counts_are_re_derived_from_the_arm(v2, field, value, needle):
    broken = copy.deepcopy(v2)
    broken["power"]["by_arm"][P.GATED_ARM][field] = value
    _fires(broken, needle)


def test_the_verdict_class_is_re_derived_from_the_gated_arms_count(v2, v2_under):
    for clean, wrong in ((v2, "underpowered"), (v2_under, "powered")):
        broken = copy.deepcopy(clean)
        broken["power"]["verdict_class"] = wrong
        _fires(broken, "power.verdict_class")


def test_the_power_block_names_the_gated_arm(v2):
    broken = copy.deepcopy(v2)
    broken["power"]["gated_arm"] = "production"
    _fires(broken, "the power block names gated arm")


def test_the_gate_mirrors_the_power_class_and_carries_the_sentence(v2, v2_under):
    broken = copy.deepcopy(v2_under)
    broken["gate"]["verdict_class"] = "powered"
    _fires(broken, "does not mirror")
    broken = copy.deepcopy(v2_under)
    del broken["gate"]["underpowered_statement"]
    _fires(broken, "verbatim")
    broken = copy.deepcopy(v2)
    broken["gate"]["underpowered_statement"] = P.UNDERPOWERED_STATEMENT
    _fires(broken, "a powered gate carries")


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (lambda r: r["stage2_identity"].pop("production"), "does not cover every graded arm"),
        (lambda r: r["stage2_identity"]["twin"].update(revision=None), "records no ['revision']"),
        (lambda r: r["stage2_identity"]["production"].update(batch_size=8), "different Stage-2"),
        (
            lambda r: r["sources"]["temperature_source"].update(heads_sha256="0" * 64),
            "not the arm the temperature source graded",
        ),
        (
            lambda r: r["sources"]["temperature_source"].update(temperature=0.0),
            "positive finite temperature",
        ),
        (
            lambda r: r["stage2_identity"]["twin"].update(shard={"index": 0, "count": 2}),
            "one slice",
        ),
        (
            lambda r: r["stage2_identity"]["twin"]["work_list_coverage"].update(n_scored=0),
            "payloads in the graded work list",
        ),
        (
            lambda r: r["stage2_identity"]["twin"].update(work_list_coverage=None),
            "payloads in the graded work list",
        ),
        (
            lambda r: r["stage2_identity"]["twin"].update(payloads_sha256=None),
            "records no payloads_sha256",
        ),
    ],
)
def test_the_stage2_identity_is_bound(v2, edit, needle):
    broken = copy.deepcopy(v2)
    edit(broken)
    _fires(broken, needle)


def test_every_v2_input_is_hash_recorded_and_every_block_bound(v2):
    broken = copy.deepcopy(v2)
    del broken["sources"]["interim_sha256"][broken["sources"]["twin_stage1"]]
    _fires(broken, "sources.twin_stage1")
    broken = copy.deepcopy(v2)
    broken["postmint"]["sha256"] = "0" * 64
    _fires(broken, "postmint is not bound")
    broken = copy.deepcopy(v2)
    broken["sources"]["benchmark_sha256"] = "0" * 64
    _fires(broken, "benchmark_sha256 is not the recorded digest")


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (
            lambda r: r["benchmark"]["scope"]["host_pool"]["overlap_with_training_folds"][
                "by_arm"
            ].update(twin=1),
            "conjunct 1/4",
        ),
        (lambda r: r["screen"].update(n_refused_kmer=0), "conjunct 2/4"),
        (lambda r: r["screen"].update(controls_all_fired=False), "conjunct 3/4"),
        (lambda r: r["screen"].update(n_controls_fired=5), "conjunct 3/4"),
        (
            lambda r: r["benchmark"]["scope"]["host_pool"].update(
                selection="parent_nested_train == False"
            ),
            "conjunct 4/4",
        ),
    ],
)
def test_the_host_overlap_invariant_is_a_conjunction_each_conjunct_required(v2, edit, needle):
    """A bare `by_arm[gated] == 0` is NOT the test (A15 Pin 6): a zero from a screen that
    refused nothing, whose controls did not fire, or a selection that does not name the
    screen, must each be refused on its own, with every other conjunct still true."""
    broken = copy.deepcopy(v2)
    edit(broken)
    problems = _fires(broken, needle)
    assert sum("conjunct" in p for p in problems) == 1, problems


def test_the_screen_read_is_the_one_the_benchmark_was_minted_under(v2):
    broken = copy.deepcopy(v2)
    broken["benchmark"]["scope"]["host_pool"]["screen"]["sha256"] = "0" * 64
    _fires(broken, "was minted under")
    broken = copy.deepcopy(v2)
    broken["screen"]["n_problems"] = 1
    _fires(broken, "the screen report carries 1 problem")


@pytest.mark.parametrize("bits", [20.0, 9.99, True, "10.0", None])
def test_the_screen_threshold_is_the_signed_ten_bits(v2, bits):
    broken = copy.deepcopy(v2)
    broken["screen"]["threshold_bits"] = bits
    _fires(broken, "not ADR-0005 A15 Pin 2's")


def test_a_table_ranked_by_another_stage2_run_is_refused():
    """Row-ID coverage and input hashing both PASS when the table and the Stage-2 file are
    different runs, so the report would publish one run's scoring identity beside another
    run's scores ([[gate-must-bind-to-upstream-evidence]])."""
    rows = [
        {"payload_key": "aaa", "stage2_logit": 1.25},
        {"payload_key": "bbb", "stage2_logit": -3.5},
    ]
    stage2 = {"logits": {"aaa": 1.25, "bbb": -3.5}}
    # Positive control first: the matching pair must be accepted, or a guard that refused
    # everything would satisfy both cases below ([[raises-test-needs-a-positive-control]]).
    P.refuse_table_scored_by_another_run(rows, stage2, arm="twin", path="s2.json")
    with pytest.raises(P.PrecisionError, match="different Stage-2 run"):
        P.refuse_table_scored_by_another_run(
            rows, {"logits": {"aaa": 1.25, "bbb": -3.4999}}, arm="twin", path="s2.json"
        )
    with pytest.raises(P.PrecisionError, match="does not"):
        P.refuse_table_scored_by_another_run(
            rows, {"logits": {"aaa": 1.25}}, arm="twin", path="s2.json"
        )
    # An absent logits block must refuse too, not read as "nothing to check".
    with pytest.raises(P.PrecisionError):
        P.refuse_table_scored_by_another_run(rows, {}, arm="twin", path="s2.json")


@pytest.mark.parametrize(
    "relative",
    [
        "reports/two_stage_precision.json",
        "reports/p3/two_stage_precision_items.json",
        "data/interim/p3_16/v1_stage2_twin.json",
    ],
)
@pytest.mark.parametrize("leg", ["fold", "report"])
def test_a_v2_leg_refuses_to_write_over_a_v1_artifact(tmp_path, leg, relative):
    """A15 Pin 5: "no v2 artifact is written to a v1 path".

    The guard already protected five two_stage entrypoints and not these two, so
    ``precision report --out reports/two_stage_precision.json`` would have overwritten the
    withdrawn v1 grade that Pin 5 keeps byte-unchanged as the record of it
    ([[fixed-one-of-two-identical-things]]). It fires at entry, before the fold reads or the
    bootstrap runs ([[guard-runs-after-what-it-guards]]).
    """
    from types import SimpleNamespace

    from tbox_finder.integration.two_stage import TwoStageError

    command = P._cmd_fold if leg == "fold" else P._cmd_report
    with pytest.raises(TwoStageError, match="refusing to write"):
        command(SimpleNamespace(out=str(tmp_path / relative)))
    # Positive control: a v2 destination must get PAST the guard (and then fail on the
    # arguments this stub does not carry), or a guard that refused everything would pass.
    with pytest.raises(AttributeError):
        command(SimpleNamespace(out=str(tmp_path / "reports/two_stage_precision_v2.json")))


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("n_violations", 1, "violation(s)"),
        ("n_violations", True, "violation(s)"),
        ("n_violations", None, "violation(s)"),
        ("n_problems", 2, "post-mint screen report carries 2"),
        # A15 Pin 6's conjunction: the count alone certifies nothing about the threshold.
        ("threshold_bits", None, "no per-model threshold_bits"),
        ("threshold_bits", {}, "no per-model threshold_bits"),
        ("threshold_bits", 10.0, "no per-model threshold_bits"),
        ("calibration_sha256", None, "names no calibration artifact"),
        ("calibration_sha256", "", "names no calibration artifact"),
        ("threshold_bits", {"RF00230": 16.8, "TBDB001": 10.0}, "inside the noise floor"),
        ("threshold_bits", {"RF00230": 20.0, "TBDB001": 9.9}, "inside the noise floor"),
        ("threshold_bits", {"RF00230": 20.0, "TBDB001": 5.0}, "stronger than its own screen"),
        ("null_max_bits", {"RF00230": 25.0, "TBDB001": 9.9}, "inside the noise floor"),
        ("null_max_bits", None, "inside the noise floor"),
        ("null_max_bits", {"RF00230": 16.8}, "inside the noise floor"),
    ],
)
def test_the_postmint_assertion_must_have_held(v2, field, value, needle):
    broken = copy.deepcopy(v2)
    broken["postmint"][field] = value
    _fires(broken, needle)


def test_a_zero_violation_count_alone_does_not_satisfy_the_postmint_gate(v2):
    """The blindness Correction 2 closed: ``n_violations == 0`` reads identically at the signed
    thresholds and at a forgotten 10.0, so the count cannot be the whole gate.

    The positive control is the unmodified fixture, asserted clean beside the two failures —
    without it a clause that refused everything would satisfy both cases
    ([[raises-test-needs-a-positive-control]]).
    """
    assert not [p for p in P.precision_problems(copy.deepcopy(v2)) if "noise floor" in p]
    for pinned in ({"RF00230": 10.0, "TBDB001": 10.0}, {"RF00230": 10.0, "TBDB001": 9.95}):
        broken = copy.deepcopy(v2)
        broken["postmint"]["threshold_bits"] = pinned
        assert broken["postmint"]["n_violations"] == 0, "the count is unchanged by construction"
        _fires(broken, "inside the noise floor")


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (lambda b: b["by_arm"].pop("production"), "the familiarity baseline covers"),
        (lambda b: b["by_arm"]["twin"].update(auprc=1.5), "is not in [0, 1]"),
        (lambda b: b["by_arm"]["twin"].update(stage1_only_auprc=0.123), "not the arm's own"),
        (lambda b: b["by_arm"]["twin"].update(two_stage_auprc=0.123), "not the arm's own"),
        (lambda b: b["by_arm"]["twin"].update(**{"lambda": 1.0}), "pinned reweighting"),
        (lambda b: b["by_arm"]["twin"].update(n_items=3), "pinned reweighting"),
        (lambda b: b["by_arm"]["twin"].update(minus_stage1_pp=9.0), "minus_stage1_pp"),
        (lambda b: b["by_arm"]["production"].update(rivals_stage1=True), "comparison it names"),
        (lambda b: b.update(gated_arm_rivals_stage1=None), "gated arm's own comparison"),
    ],
)
def test_the_familiarity_baseline_is_re_derived_against_each_arm(v2, edit, needle):
    broken = copy.deepcopy(v2)
    edit(broken["familiarity_baseline"])
    _fires(broken, needle)


def test_a_rivalling_baseline_must_be_disclosed_and_only_then(v2, v2_under):
    """A15 Pin 7(b): if the familiarity baseline rivals Stage-1, the report states the gate is
    confounded by construction. It says so only when it is true, in both directions."""
    assert v2_under["familiarity_baseline"]["gated_arm_rivals_stage1"] is True
    assert any(P.CONFOUNDED_STATEMENT in line for line in v2_under["disclosures"])
    silent = copy.deepcopy(v2_under)
    silent["disclosures"] = [
        line for line in silent["disclosures"] if P.CONFOUNDED_STATEMENT not in line
    ]
    _fires(silent, "no disclosure says")
    false_claim = copy.deepcopy(v2)
    false_claim["disclosures"] = [*false_claim["disclosures"], P.CONFOUNDED_STATEMENT]
    _fires(false_claim, "but the baseline does not rival")


def _annexed(report: dict) -> dict:
    """The report with a τ annex of v2's OWN reads (the report re-labelled at 0.7 / 0.9)."""
    points = []
    for tau in (0.7, 0.9):
        swept = copy.deepcopy(report)
        swept["sources"]["stage1_threshold"] = tau
        points.append({"declared": tau, "sha256": f"{int(tau * 100):064d}", "report": swept})
    out = copy.deepcopy(report)
    out["stage1_threshold_sensitivity"] = P.json_safe(
        P.stage1_threshold_sensitivity(
            points,
            base_threshold=report["sources"]["stage1_threshold"],
            base_passes=report["gate"]["passes"],
            bind_benchmark={
                "benchmark_sha256": report["sources"]["benchmark_sha256"],
                "matched_recall": report["gate"]["reported_not_gated"]["matched_recall"],
            },
        )
    )
    return out


def test_the_tau_annex_rebinds_to_v2s_own_reads(v2, v2_under):
    """A13 Pin 2's failing-read clause rebinds to v2's own τ reads, whatever their sign (A15
    Pin 5): clean on both cases, including the tie, whose matched-recall read is 0."""
    for clean in (v2, v2_under):
        assert P.precision_problems(_annexed(clean)) == []
    broken = _annexed(v2)
    broken["stage1_threshold_sensitivity"]["points"][0]["benchmark_sha256"] = "0" * 64
    _fires(broken, "graded on a different benchmark")
    broken = _annexed(v2)
    broken["stage1_threshold_sensitivity"]["base"]["matched_recall"] = {"gain_pp": 1.0}
    _fires(broken, "own matched-recall read")


def test_the_committed_v1_tau_reports_cannot_stand_in_a_v2_sweep(v2):
    """The real v1 τ annexes, graded on benchmark_v1, bound into a v2 annex: refused."""
    points = []
    for tau, name in ((0.7, "07"), (0.9, "09")):
        path = REPO / "reports" / "p3" / f"two_stage_precision_tau{name}.json"
        points.append(
            {"declared": tau, "sha256": _sha(path), "report": json.loads(path.read_text())}
        )
    broken = copy.deepcopy(v2)
    broken["stage1_threshold_sensitivity"] = P.json_safe(
        P.stage1_threshold_sensitivity(
            points,
            base_threshold=0.5,
            base_passes=v2["gate"]["passes"],
            bind_benchmark={
                "benchmark_sha256": v2["sources"]["benchmark_sha256"],
                "matched_recall": v2["gate"]["reported_not_gated"]["matched_recall"],
            },
        )
    )
    _fires(broken, "graded on a different benchmark")


def test_the_composition_is_the_minted_one_beside_the_v1_audit(v2):
    broken = copy.deepcopy(v2)
    broken["composition"]["v2_drawn_hosts"]["side"]["lead"] += 1
    _fires(broken, "minted scope.host_pool.composition")
    for v1_hosts in ({}, {"side": {"lead": 401, "trail": 291}}):
        broken = copy.deepcopy(v2)
        broken["composition"]["v1_hosts"] = v1_hosts
        _fires(broken, "v1_audit composition")


def test_the_v2_disclosures_are_read_from_the_payload(v2, v2_under):
    """No fixed numeral: the counts each Pin 7 sentence states move with the payload."""
    text = "\n".join(v2["disclosures"])
    assert "Pin 7(a)" in text and "Pin 7(c)" in text and "Pin 7(d)" in text
    assert "Pin 7(e)" in text and "lead 17" in text and "lead 400" in text
    assert "3 of 24 gated negatives carry a Stage-1 candidate" in text
    under = "\n".join(v2_under["disclosures"])
    assert P.UNDERPOWERED_STATEMENT in under and "NOT 'two-stage did not beat" in under
    assert "dec:leader two-stage 0.300000" in text
    assert "clean for BOTH" not in json.dumps(v2)


@pytest.mark.parametrize(
    "block",
    [
        ("power",),
        ("familiarity_baseline",),
        ("stage2_identity",),
        ("screen",),
        ("postmint",),
        ("composition",),
        ("reported_not_gated",),
        ("sources",),
        ("sources", "interim_sha256"),
        ("sources", "temperature_source"),
        ("benchmark", "scope", "host_pool", "screen"),
        ("benchmark", "scope", "negatives_by_pool_gated"),
    ],
)
def test_an_emptied_v2_block_is_refused_never_read_as_absent(v2, block):
    """[[clauses-must-guard-emptiness]]: each v2 block emptied, nulled or retyped must come
    back as a problem, never as a vacuously true clause, and never as a traceback."""
    for value in (None, {}, [], "x", 0):
        broken = copy.deepcopy(v2)
        node = broken
        for key in block[:-1]:
            node = node[key]
        node[block[-1]] = value
        assert P.precision_problems(broken), f"{'/'.join(block)} = {value!r} was accepted"


def test_no_retyped_v2_node_can_make_the_checker_raise(v2_under):
    """The no-raise contract on the 1.2 clause set, swept over every node of the v2-only
    blocks (the 1.1 nodes are swept on the committed report above)."""

    def paths(node, path=()):
        yield path
        if isinstance(node, dict):
            for key, value in node.items():
                yield from paths(value, (*path, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from paths(value, (*path, index))

    roots = (
        "power",
        "familiarity_baseline",
        "stage2_identity",
        "screen",
        "postmint",
        "composition",
        "reported_not_gated",
        "sources",
        "gate",
    )
    report = _annexed(v2_under)
    checked = 0
    for root in (*roots, "stage1_threshold_sensitivity"):
        for path in paths(report[root], (root,)):
            for bad in ("x", None, True, -1, 10**400, {}, []):
                broken = copy.deepcopy(report)
                node = broken
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = bad
                P.precision_problems(broken)  # must not raise
                checked += 1
    scope_paths = list(paths(report["benchmark"]["scope"], ("benchmark", "scope")))[1:]
    for path in scope_paths:
        for bad in ("x", None, True, 10**400, {}, []):
            broken = copy.deepcopy(report)
            node = broken
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = bad
            P.precision_problems(broken)
            checked += 1
    assert checked > 1_000, checked


# ── the file binding (precision_input_problems) ──────────────────────────────
def test_the_file_binding_is_clean_on_the_files_the_report_names(v2_case):
    assert (
        P.precision_input_problems(
            v2_case["report"], repo_root=v2_case["root"], require_all_inputs=True
        )
        == []
    )


def test_a_rewritten_committed_input_is_refused(tmp_path):
    case = _build_v2(tmp_path)
    screen = case["root"] / "reports/p3/host_screen.json"
    screen.write_text(screen.read_text().replace('"n_refused_kmer": 7', '"n_refused_kmer": 8'))
    problems = P.precision_input_problems(case["report"], repo_root=case["root"])
    assert any("host_screen.json no longer hashes" in p for p in problems), problems


def test_a_missing_committed_input_is_refused_but_a_missing_interim_one_is_excused(tmp_path):
    case = _build_v2(tmp_path)
    root, report = case["root"], case["report"]
    (root / "data/interim/p3_16/v2_stage1_twin.npz").unlink()
    assert P.precision_input_problems(report, repo_root=root) == []
    strict = P.precision_input_problems(report, repo_root=root, require_all_inputs=True)
    assert any("v2_stage1_twin.npz is absent" in p for p in strict), strict
    (root / "reports/p3/benchmark_v2_postmint_screen.json").unlink()
    problems = P.precision_input_problems(report, repo_root=root)
    assert any("benchmark_v2_postmint_screen.json is absent" in p for p in problems), problems


def test_the_item_table_must_be_the_one_provenance_hashed(tmp_path):
    case = _build_v2(tmp_path)
    items = case["root"] / V2_ITEMS_REL
    items.write_text(items.read_text() + "\n")
    problems = P.precision_input_problems(case["report"], repo_root=case["root"])
    assert any("item table" in p and "no longer hashes" in p for p in problems), problems
    unbound = copy.deepcopy(case["report"])
    unbound["provenance"] = {"inputs": {}}
    assert any(
        "carries no provenance digest" in p
        for p in P.precision_input_problems(unbound, repo_root=case["root"])
    )


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (
            lambda r: r["power"]["by_arm"]["twin"].update(n_gated_negatives_with_stage2_payload=4),
            "power counts do not re-derive",
        ),
        (
            lambda r: r["reported_not_gated"]["non_gated_items"]["twin"][0].update(two_stage=0.31),
            "non-gated scores do not re-derive",
        ),
        (
            lambda r: r["familiarity_baseline"]["by_arm"]["twin"].update(auprc=0.5),
            "familiarity AUPRC",
        ),
    ],
)
def test_the_item_level_blocks_re_derive_from_the_item_table(v2_case, edit, needle):
    """What the report cannot carry — the per-item payload counts, the leader's raw scores,
    the familiarity ranking — is re-derived from the committed item table itself."""
    broken = copy.deepcopy(v2_case["report"])
    edit(broken)
    problems = P.precision_input_problems(broken, repo_root=v2_case["root"])
    assert any(needle in p for p in problems), problems


def test_work_list_coverage_counts_only_the_payloads_that_were_scored():
    payload = {"logits": {"c:1#+": 0.1, "c:1#-": -0.2, "c:9#+": 0.0}}
    work_list = [{"row_id": "c:1#+"}, {"row_id": "c:1#-"}, {"row_id": "c:2#+"}]
    assert P.work_list_coverage(payload, work_list) == {"n_rows": 3, "n_scored": 2}
    assert P.work_list_coverage({"logits": None}, work_list) == {"n_rows": 3, "n_scored": 0}


def test_the_stage2_work_list_coverage_re_derives_from_its_files(tmp_path):
    """The fold's coverage count is re-derived from the Stage-2 logits and the work list
    whenever both are on disk. A self-consistent forged count passes the report-only
    clauses, but it is refused here."""
    case = _build_v2(tmp_path)
    report, root = copy.deepcopy(case["report"]), case["root"]
    assert report["stage2_identity"]["twin"]["work_list_coverage"]["n_rows"] > 0
    assert report["stage2_identity"]["twin"]["payloads_sha256"] != (
        report["sources"]["interim_sha256"][report["sources"]["twin_payloads"]]
    ), "the fixture must score a union manifest, or coverage is indistinguishable from identity"
    report["stage2_identity"]["twin"]["work_list_coverage"] = {"n_rows": 999, "n_scored": 999}
    assert P.precision_problems(report) == []
    problems = P.precision_input_problems(report, repo_root=root)
    assert any("work-list coverage does not re-derive" in p for p in problems), problems
    (root / "data/interim/p3_16/v2_stage2_twin.json").unlink()
    assert P.precision_input_problems(report, repo_root=root) == []  # interim: excused


def test_an_absolute_or_escaping_input_path_is_refused(v2_case):
    for bad in ("/etc/passwd", "../outside.json"):
        broken = copy.deepcopy(v2_case["report"])
        broken["sources"]["interim_sha256"][bad] = "0" * 64
        problems = P.precision_input_problems(broken, repo_root=v2_case["root"])
        assert any("is not repo-relative" in p for p in problems), (bad, problems)


# ── the CLI: unchanged exit codes, and an underpowered tie lands canonically ──
def _run_report(case: dict, monkeypatch) -> int:
    root = case["root"]
    (root / "envs").mkdir(exist_ok=True)
    __import__("shutil").copyfile(
        REPO / "envs" / "data.conda-lock.yml", root / "envs" / "data.conda-lock.yml"
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(P, "REPO_ROOT", root)
    monkeypatch.setattr(
        P, "repo_relative", lambda path: str(Path(path).resolve().relative_to(root))
    )
    return P.main(
        [
            "report",
            "--items",
            V2_ITEMS_REL,
            "--out",
            V2_OUT_REL,
            "--stage2-operating-point",
            "0.5",
            "--n-boot",
            str(V2_N_BOOT),
        ]
    )


def test_an_underpowered_tie_is_written_canonically_and_fails(tmp_path, monkeypatch):
    """A15 Pin 5: "no power" is a measured outcome, not a broken run. The CLI writes it to the
    canonical path (not .invalid.json) and returns 4, the gate-fails code."""
    case = _build_v2(tmp_path, powered=False, familiar=True)
    assert _run_report(case, monkeypatch) == 4
    written = json.loads((case["root"] / V2_OUT_REL).read_text())
    assert written["problems"] == []
    assert written["gate"]["verdict_class"] == "underpowered"
    assert written["gate"]["passes"] is False
    assert not (case["root"] / V2_OUT_REL).with_suffix(".invalid.json").exists()


def test_a_powered_pass_exits_zero(tmp_path, monkeypatch):
    case = _build_v2(tmp_path, powered=True)
    assert _run_report(case, monkeypatch) == 0
    assert json.loads((case["root"] / V2_OUT_REL).read_text())["gate"]["passes"] is True


def test_a_report_whose_bound_file_drifted_is_diverted(tmp_path, monkeypatch):
    """The CLI runs the file binding on 1.2 while the interim files are still on disk: a
    payload manifest rewritten after the fold diverts the report (exit 3)."""
    case = _build_v2(tmp_path, powered=True)
    payloads = case["root"] / "data/interim/p3_16/v2_payloads_twin_t0.5.json"
    payloads.write_text(payloads.read_text() + " ")
    assert _run_report(case, monkeypatch) == 3
    invalid = json.loads((case["root"] / V2_OUT_REL).with_suffix(".invalid.json").read_text())
    assert any("v2_payloads_twin_t0.5.json no longer hashes" in p for p in invalid["problems"])
    assert not (case["root"] / V2_OUT_REL).exists()


def test_route_report_diverts_a_refused_v2_report(v2, monkeypatch):
    written: dict[str, dict] = {}
    monkeypatch.setattr(
        P, "write_json", lambda path, payload: written.__setitem__(str(path), payload)
    )
    broken = copy.deepcopy(v2)
    broken["postmint"]["n_violations"] = 1
    assert P.route_report(broken, "reports/_never_written_v2.json") == 3
    assert list(written) == ["reports/_never_written_v2.invalid.json"]
    assert P.route_report(copy.deepcopy(v2), "reports/_never_written_v2.json") == 0


# ── the figure script reads the v2 fields and leaves v1 alone ─────────────────
def _plotter():
    """Load the figure script by path. It imports matplotlib only inside `main`, so its
    report-reading helpers run here without matplotlib (absent from CI and labvault)."""
    import importlib.util

    path = REPO / "scripts" / "plot_two_stage_precision.py"
    spec = importlib.util.spec_from_file_location("plot_two_stage_precision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_figure_annotates_an_underpowered_gate_and_only_then(report, v2, v2_under):
    plot = _plotter()
    assert plot.underpowered_note(v2_under) == P.UNDERPOWERED_STATEMENT
    assert plot.underpowered_note(v2) is None
    assert plot.underpowered_note(report) is None  # v1 draws exactly as before
    assert plot.familiarity_point(report) is None
    point = plot.familiarity_point(v2)
    assert point == (100.0, v2["familiarity_baseline"]["by_arm"][P.GATED_ARM]["auprc"])
