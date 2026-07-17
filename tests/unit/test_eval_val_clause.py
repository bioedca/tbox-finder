"""P2-06a — the ``eval_val_scored_on_disjoint_fold`` gate clause.

Bare-CI tier: pure JSON, no torch.

This file exists because of one specific, twice-paid lesson. ``overall_pass = all(clauses)``
catches a clause flipped **FALSE** but structurally cannot catch one fabricated **TRUE**
(P1-15/P1-16), and a clause that reads its *requested config* rather than its *found
evidence* goes vacuously TRUE exactly when the evidence is missing (P2-05 — a green sizing
gate resting on zero DDP×8 points). The clause under test here is the highest-risk instance
of that shape in the repo: an eval that silently no-ops leaves ``eval_scope`` absent, and a
"no violation found" phrasing would certify that as clean.

So the tests below assert the **gate**, not the fields, on the absence branch.
"""

from __future__ import annotations

import pytest

from tbox_finder.train.train_stage1 import _eval_val_ok, derive_clauses


def _scope() -> dict:
    return {
        "fold_scope": "selection_val",
        "n_records_scored": 880,
        "n_blocks": 726,
        "disjointness": {
            "shared_record_ids_with_train": 0,
            "shared_cluster_ids_with_train": 0,
        },
    }


def _metrics() -> dict:
    return {
        "eval_split": "selection_val",
        "n_positions": 2_100_000,
        "gate4_core_min_f1": {"min_f1": 0.42, "per_element_f1": {}},
    }


def _report(**over) -> dict:
    rep = {"eval_requested": True, "eval_scope": _scope(), "eval_metrics": _metrics()}
    rep.update(over)
    return rep


def test_clause_passes_on_a_real_scored_disjoint_fold():
    assert _eval_val_ok(_report()) is True


# ── The absence branch: the P2-05 shape ──────────────────────────────────────
def test_gate_is_false_when_eval_requested_but_scope_missing():
    """THE test this file exists for.

    Requested an eval, produced no evidence ⇒ the GATE must be False. A clause phrased as
    "no disjointness violation was found" would return True here, because no violation can
    be found in an absent block. That is exactly how P2-05 shipped a green gate with zero
    measurements.
    """
    assert _eval_val_ok(_report(eval_scope=None)) is False
    assert _eval_val_ok(_report(eval_scope={})) is False
    assert _eval_val_ok(_report(eval_metrics=None)) is False
    assert _eval_val_ok(_report(eval_metrics={})) is False
    # ...and the same through the real gate, not just the predicate.
    clauses = derive_clauses(_report(eval_scope=None))
    assert clauses["eval_val_scored_on_disjoint_fold"] is False


def test_gate_is_false_when_the_disjointness_block_is_absent():
    rep = _report()
    rep["eval_scope"].pop("disjointness")
    assert _eval_val_ok(rep) is False
    rep = _report()
    rep["eval_scope"]["disjointness"] = {}
    assert _eval_val_ok(rep) is False


def test_not_requested_branch_requires_evidence_to_be_explicitly_absent():
    """``eval_requested: False`` passes only if nothing is ALSO claimed — an incoherent
    report (no eval requested, yet metrics present) is not a pass."""
    assert _eval_val_ok({"eval_requested": False, "eval_scope": None, "eval_metrics": None}) is True
    assert (
        _eval_val_ok({"eval_requested": False, "eval_scope": _scope(), "eval_metrics": None})
        is False
    )
    assert (
        _eval_val_ok({"eval_requested": False, "eval_scope": None, "eval_metrics": _metrics()})
        is False
    )


def test_a_missing_eval_requested_key_is_not_a_pass():
    """Absent ⇒ neither branch ⇒ False. The clause is total; it never falls through."""
    assert _eval_val_ok({}) is False
    assert _eval_val_ok({"eval_requested": None}) is False
    assert _eval_val_ok({"eval_requested": "yes"}) is False
    assert _eval_val_ok({"eval_requested": 1}) is False  # 1 is not True here


# ── Each leak condition bites ────────────────────────────────────────────────
@pytest.mark.parametrize("key", ["shared_record_ids_with_train", "shared_cluster_ids_with_train"])
@pytest.mark.parametrize("bad", [1, 7, -1, True, False, "0", None, 0.0])
def test_clause_is_false_on_any_non_zero_or_ill_typed_overlap(key, bad):
    """Includes the bool trap: ``False == 0`` is True in Python, so a bare ``!= 0`` check
    would accept ``False`` as evidence of zero overlap. It is not evidence; it is a type
    error wearing the right value."""
    rep = _report()
    rep["eval_scope"]["disjointness"][key] = bad
    assert _eval_val_ok(rep) is False


def test_clause_is_false_on_the_wrong_fold_scope():
    for scope in ["train", "loo_order_unit", "test", "", None]:
        rep = _report()
        rep["eval_scope"]["fold_scope"] = scope
        assert _eval_val_ok(rep) is False, f"{scope!r} was accepted as a selection fold"


def test_clause_is_false_when_the_fold_is_not_block_resamplable():
    for n in [1, 0, -3, True, None, "726"]:
        rep = _report()
        rep["eval_scope"]["n_blocks"] = n
        assert _eval_val_ok(rep) is False, f"n_blocks={n!r} was accepted"


def test_clause_is_false_on_a_nan_or_absent_gated_statistic():
    for bad in [float("nan"), float("inf"), None, "0.42", True]:
        rep = _report()
        rep["eval_metrics"]["gate4_core_min_f1"]["min_f1"] = bad
        assert _eval_val_ok(rep) is False, f"min_f1={bad!r} was accepted"
    rep = _report()
    rep["eval_metrics"].pop("gate4_core_min_f1")
    assert _eval_val_ok(rep) is False


def test_clause_is_false_when_nothing_was_actually_scored():
    for n in [0, -1, None, True]:
        rep = _report()
        rep["eval_scope"]["n_records_scored"] = n
        assert _eval_val_ok(rep) is False
    rep = _report()
    rep["eval_metrics"]["n_positions"] = 0
    assert _eval_val_ok(rep) is False


def test_clause_is_false_when_metrics_claim_a_different_split():
    rep = _report()
    rep["eval_metrics"]["eval_split"] = "fine_tune_set"
    assert _eval_val_ok(rep) is False, "a train-on-train split label was accepted"


def test_clause_is_in_the_gate_and_not_merely_defined():
    """A clause nothing calls is dead code — the P2-04 `TBOX_REQUIRE_TRAIN_SMOKE` lesson."""
    assert "eval_val_scored_on_disjoint_fold" in derive_clauses(_report())


def test_n_blocks_must_describe_the_scored_set_not_the_loaded_fold():
    """Regression: found by running the code, not by reading it (P2-06a probe, 2026-07-17).

    The first draft copied the loader's fold-level ``n_blocks`` (726) into the scope of a
    run that scored **4** records. The gate's ">= 2 blocks" clause then passed on blocks the
    bootstrap never saw — a field describing the *requested* population rather than the
    *measured* one. The scope's block count must be a set over the records actually scored,
    so a 1-block capped eval FAILS rather than borrowing the fold's 726.
    """
    rep = _report()
    rep["eval_scope"]["n_records_scored"] = 1
    rep["eval_scope"]["n_blocks"] = 1
    rep["eval_scope"]["n_blocks_fold"] = 726  # the fold is big; THIS eval was not
    assert _eval_val_ok(rep) is False, "the fold's block count certified a 1-block eval"
