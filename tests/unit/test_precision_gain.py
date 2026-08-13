"""P3-16 — unit tests for the two-stage vs Stage-1-only precision comparison.

The committed report is a *pinning* artifact, so most of what matters here is tested against
constructed payloads rather than by reading it: a test that only reads
``reports/two_stage_precision.json`` cannot see the code that produced it
([[artifact-pinning-test-cannot-see-the-code]]). Where a test does read the committed report
it says so and pairs with a constructed twin that exercises the same function.
"""

from __future__ import annotations

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
                "arms": {
                    arm: {"n_rows": 1, "stage1_only": 0.60 + index / 500, "two_stage": 0.95}
                    for arm in ("production", "twin")
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
                    arm: {"n_rows": 1, "stage1_only": 0.61 + index / 500, "two_stage": 0.05}
                    for arm in ("production", "twin")
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
        n_boot=20,
        seed=42,
        generated_at_utc="2026-08-13T00:00:00+00:00",
    )
    assert report["gate"]["passes"] is False
    assert report["gate"]["gain_pp"] == pytest.approx(0.0)
    # A failing gate is still a *valid* report — it must not be refused as malformed, or a
    # fail would be indistinguishable from a broken run.
    assert P.precision_problems(report) == []
