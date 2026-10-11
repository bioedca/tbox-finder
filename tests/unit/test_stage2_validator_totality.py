"""P3-17′-validators — ``stage2.{eval,sizing}.validate_report`` return problems, never raise.

Booked at P3-17 review round 12 and fixed here. A mutation census on ``main`` @ ``9507912`` —
every JSON path of the four committed reports, replaced by each of 19 wrong-typed values or
deleted — found the two validators raising on **11** (sizing) and **21** (eval) distinct lines,
each counted with the top-level non-mapping guard. A raise is not a verdict, so each one now
returns problems.

Two things make "no longer raises" safe rather than a quieter failure mode:

* **No clause moves.** ``derive_clauses`` reads a malformed value as absent
  (:func:`~tbox_finder.report_schema.as_mapping`; a non-comparable count compares ``False``),
  which agrees with the ``x or {}`` idiom on every value the idiom did not raise on. Across
  the census, each clause verdict ``main`` returned is identical here.
* **No fail-open.** Absent is what a clause *fails* on, but a report whose recorded clause is
  already ``False`` would agree with it. The shipped P3-08 gate is honestly ``false``.
  ``_SHAPE_RULES`` names the malformed field itself, so every formerly-raising input still
  returns at least one problem. :data:`FORMERLY_RAISING` holds one census representative per
  site per report.

The schema-1 production sizing report (job 1051) also failed its own validator. Its
``recommendation`` was graded against the computed-headroom shape that schema 2 introduced; it
is now re-derived in its own schema's shape, with the batch size still bound to the
measurements.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tbox_finder import report_schema as RSCH
from tbox_finder.stage2 import eval as E
from tbox_finder.stage2 import sizing as S

_REPO = Path(__file__).resolve().parents[2]

SIZING_PROD = "reports/p3/stage2_sizing.json"  # schema 1, job 1051
SIZING_RNAFM = "reports/p3/stage2_rnafm_sizing.json"  # schema 4, job 1374
EVAL_PROD = "reports/stage2_aux_ablation.json"  # schema 1, gate honestly false
EVAL_RNAFM = "reports/p3/stage2_rnafm_eval.json"  # schema 2

VALIDATORS: dict[str, Callable[[Any], list[str]]] = {
    "sizing": S.validate_report,
    "eval": E.validate_report,
}
REPORTS = [
    ("sizing", SIZING_PROD),
    ("sizing", SIZING_RNAFM),
    ("eval", EVAL_PROD),
    ("eval", EVAL_RNAFM),
]

_DELETE = object()

#: The wrong-typed replacements the totality sweep tries at every path. A subset of the census's
#: 19 that still reaches every crash class: absence-like, a string, a negative and a fractional
#: number, a bool, both empty containers, an unhashable list, and an int no float can hold.
SWEEP_VALUES: tuple[Any, ...] = (None, "x", -1, 1.5, True, [], {}, ["x"], 10**400, _DELETE)

#: One census representative per raising line per report: on ``main`` @ ``9507912`` each of
#: these RAISED at the line named. Each must now return at least one problem.
FORMERLY_RAISING: tuple[tuple[str, str, str, Any], ...] = (
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4/aux_weight", 10**400),  # eval.py:1045
    ("eval", EVAL_PROD, "arms/aux0.0_lr1e-4/aux_weight", 10**400),  # eval.py:1045
    ("eval", EVAL_RNAFM, "arms", "x"),  # eval.py:1068
    ("eval", EVAL_PROD, "arms", "x"),  # eval.py:1068
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4/load", "x"),  # eval.py:1073
    ("eval", EVAL_PROD, "arms/aux0.0_lr1e-4/load", "x"),  # eval.py:1073
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4/grades", "x"),  # eval.py:1103
    ("eval", EVAL_PROD, "arms/aux0.0_lr1e-4/grades", "x"),  # eval.py:1103
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4/calibration", "x"),  # eval.py:1105
    ("eval", EVAL_PROD, "arms/aux0.0_lr1e-4/calibration", "x"),  # eval.py:1105
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4/calibration/n_by_rung", "x"),  # eval.py:1107
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4/grades/test", "x"),  # eval.py:1121
    ("eval", EVAL_PROD, "arms/aux0.0_lr1e-4/grades/test", "x"),  # eval.py:1121
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4/stack", "x"),  # eval.py:1124
    ("eval", EVAL_PROD, "arms/aux0.0_lr1e-4/stack", "x"),  # eval.py:1124
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4/stack/stack_applied", -1),  # eval.py:1127
    ("eval", EVAL_PROD, "arms/aux1.0_lr1e-4/grades/test", "x"),  # eval.py:1134
    ("eval", EVAL_RNAFM, "ablation", "x"),  # eval.py:1143
    ("eval", EVAL_PROD, "ablation", "x"),  # eval.py:1143
    ("eval", EVAL_RNAFM, "dataset", "x"),  # eval.py:1149
    ("eval", EVAL_PROD, "dataset", "x"),  # eval.py:1149
    ("eval", EVAL_RNAFM, "dataset/rung_census", "x"),  # eval.py:1155
    ("eval", EVAL_PROD, "dataset/rung_census", "x"),  # eval.py:1155
    ("eval", EVAL_RNAFM, "ablation/with_aux_arm", []),  # eval.py:1162
    ("eval", EVAL_PROD, "ablation/with_aux_arm", []),  # eval.py:1162
    ("eval", EVAL_RNAFM, "ablation/no_aux_arm", []),  # eval.py:1163
    ("eval", EVAL_PROD, "ablation/no_aux_arm", []),  # eval.py:1163
    ("eval", EVAL_RNAFM, "provenance", "x"),  # eval.py:1196
    ("eval", EVAL_PROD, "provenance", "x"),  # eval.py:1196
    ("eval", EVAL_RNAFM, "arms/aux1.0_lr1e-4", "x"),  # eval.py:1202
    ("eval", EVAL_PROD, "arms/aux1.0_lr1e-4", "x"),  # eval.py:1202
    ("eval", EVAL_RNAFM, "arms/aux0.0_lr1e-4", "x"),  # eval.py:1203
    ("eval", EVAL_PROD, "arms/aux0.0_lr1e-4", "x"),  # eval.py:1203
    ("eval", EVAL_RNAFM, "ablation/reading_delta", "x"),  # eval.py:1247
    ("eval", EVAL_PROD, "ablation/reading_delta", "x"),  # eval.py:1247
    ("eval", EVAL_RNAFM, "ablation/reading_absolute", "x"),  # eval.py:1253
    ("eval", EVAL_PROD, "ablation/reading_absolute", "x"),  # eval.py:1253
    ("sizing", SIZING_RNAFM, "measurements", "x"),  # sizing.py:342
    ("sizing", SIZING_PROD, "measurements", "x"),  # sizing.py:342
    ("sizing", SIZING_RNAFM, "device", "x"),  # sizing.py:345
    ("sizing", SIZING_PROD, "device", "x"),  # sizing.py:345
    ("sizing", SIZING_RNAFM, "population", "x"),  # sizing.py:350
    ("sizing", SIZING_PROD, "population", "x"),  # sizing.py:350
    ("sizing", SIZING_RNAFM, "measurements/0/n_steps", None),  # sizing.py:360
    ("sizing", SIZING_PROD, "measurements/0/n_steps", None),  # sizing.py:360 (an OOM point)
    ("sizing", SIZING_RNAFM, "gradient_checkpointing", "x"),  # sizing.py:372
    ("sizing", SIZING_PROD, "gradient_checkpointing", "x"),  # sizing.py:372
    ("sizing", SIZING_RNAFM, "backbone", "x"),  # sizing.py:425
    ("sizing", SIZING_RNAFM, "schema_version", []),  # sizing.py:439
    ("sizing", SIZING_PROD, "schema_version", []),  # sizing.py:439
    ("sizing", SIZING_RNAFM, "provenance", "x"),  # sizing.py:475
    ("sizing", SIZING_RNAFM, "measurements/0/batch_size", None),  # sizing.py:520
    ("sizing", SIZING_PROD, "measurements/2/batch_size", None),  # sizing.py:520
    ("sizing", SIZING_RNAFM, "device/total_memory_gib", 10**400),  # sizing.py:546
    ("sizing", SIZING_PROD, "device/total_memory_gib", 10**400),  # sizing.py:546
)


def _load(rel: str) -> dict[str, Any]:
    return json.loads((_REPO / rel).read_text(encoding="utf-8"))


def _steps(path: str) -> list[str | int]:
    return [int(p) if p.isdigit() else p for p in path.split("/")]


def _mutated(report: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(report)
    *parents, leaf = _steps(path)
    node: Any = out
    for step in parents:
        node = node[step]
    if value is _DELETE:
        del node[leaf]
    else:
        node[leaf] = value
    return out


def _paths(node: Any, prefix: tuple[str, ...] = ()) -> Iterator[str]:
    if prefix:
        yield "/".join(prefix)
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _paths(child, (*prefix, str(key)))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from _paths(child, (*prefix, str(index)))


# --------------------------------------------------------------------------------------------- #
# The committed reports
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(("suite", "rel"), REPORTS)
def test_every_committed_report_validates_clean(suite: str, rel: str) -> None:
    """Including the schema-1 production sizing report, which failed before this step."""
    assert VALIDATORS[suite](_load(rel)) == []


@pytest.mark.parametrize(("suite", "rel"), REPORTS)
def test_no_committed_report_trips_a_shape_rule(suite: str, rel: str) -> None:
    """The rules describe what the producers actually write: an honest report is not malformed."""
    rules = S._SHAPE_RULES if suite == "sizing" else E._SHAPE_RULES
    assert RSCH.shape_problems(_load(rel), rules) == []


# --------------------------------------------------------------------------------------------- #
# Totality
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(("suite", "rel"), REPORTS)
def test_validate_report_never_raises_on_any_single_field_mutation(suite: str, rel: str) -> None:
    base = _load(rel)
    validate = VALIDATORS[suite]
    n_cases = 0
    for path in _paths(base):
        for value in SWEEP_VALUES:
            problems = validate(_mutated(base, path, value))
            assert isinstance(problems, list), (path, value)
            n_cases += 1
    assert n_cases > 1000  # the sweep reached the report, not an empty walk


@pytest.mark.parametrize("suite", sorted(VALIDATORS))
@pytest.mark.parametrize("report", [None, "x", 1, [], [{}]])
def test_a_report_that_is_not_a_mapping_is_a_problem_not_a_raise(suite: str, report: Any) -> None:
    problems = VALIDATORS[suite](report)
    assert problems and problems[0].startswith(RSCH.MALFORMED)


@pytest.mark.parametrize(("suite", "rel", "path", "value"), FORMERLY_RAISING)
def test_every_formerly_raising_input_is_reported_never_passed(
    suite: str, rel: str, path: str, value: Any
) -> None:
    """Fail-open guard: each census representative raised on ``main``; it must not now pass."""
    assert VALIDATORS[suite](_mutated(_load(rel), path, value)) != []


@pytest.mark.parametrize(
    ("suite", "rel", "path", "value"),
    [
        # A recorded clause that is already FALSE agrees with "absent", so only the shape rule
        # can object. These are the exact cases where dropping it would fail open.
        ("sizing", SIZING_PROD, "measurements/0/n_steps", None),
        ("eval", EVAL_PROD, "ablation/reading_delta", "x"),
        ("eval", EVAL_PROD, "arms/aux0.0_lr1e-4/stack/stack_applied", -1),
        ("eval", EVAL_PROD, "ablation/with_aux_arm", ["x"]),
    ],
)
def test_the_malformed_field_itself_is_named(suite: str, rel: str, path: str, value: Any) -> None:
    problems = VALIDATORS[suite](_mutated(_load(rel), path, value))
    assert any(p.startswith(RSCH.MALFORMED) and path in p for p in problems), problems


def test_the_sweep_covers_every_formerly_raising_value_kind() -> None:
    """Guard the sweep itself: a representative whose value the sweep never tries would leave
    its crash class covered only by the one hand-picked case."""
    kinds = {type(v).__name__ if v is not None else "None" for v in SWEEP_VALUES}
    for _, _, _, value in FORMERLY_RAISING:
        assert (type(value).__name__ if value is not None else "None") in kinds, value
    assert 10**400 in SWEEP_VALUES  # the OverflowError class needs this exact magnitude


# --------------------------------------------------------------------------------------------- #
# No clause moves: a malformed block reads as an absent one
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("module", "rel", "path"),
    [
        (S, SIZING_RNAFM, "device"),
        (S, SIZING_RNAFM, "gradient_checkpointing"),
        (S, SIZING_RNAFM, "backbone"),
        (S, SIZING_RNAFM, "measurements"),
        (E, EVAL_RNAFM, "arms"),
        (E, EVAL_RNAFM, "arms/aux0.0_lr1e-4/load"),
        (E, EVAL_RNAFM, "arms/aux0.0_lr1e-4/grades/test"),
        (E, EVAL_RNAFM, "dataset/rung_census"),
        (E, EVAL_RNAFM, "provenance"),
    ],
)
@pytest.mark.parametrize("junk", ["x", 7, ["x"]])
def test_a_malformed_block_derives_the_clauses_of_an_absent_one(
    module: Any, rel: str, path: str, junk: Any
) -> None:
    base = _load(rel)
    assert module.derive_clauses(_mutated(base, path, junk)) == module.derive_clauses(
        _mutated(base, path, _DELETE)
    )


def test_a_count_that_does_not_compare_fails_its_clause_on_a_point_that_ran() -> None:
    """``_exceeds`` is a verdict: a point with a junk ``n_steps`` that did NOT OOM neither ran
    steps nor failed, so ``swept`` is FALSE — not TRUE, and not a raise."""
    base = _load(SIZING_RNAFM)
    assert base["measurements"][0]["oom"] is False
    assert S.derive_clauses(base)["swept"] is True
    for junk in (None, "6", [6]):
        clauses = S.derive_clauses(_mutated(base, "measurements/0/n_steps", junk))
        assert clauses["swept"] is False, junk
        assert clauses["enough_steps_for_optimizer_state"] is False, junk


def test_a_non_mapping_point_is_an_empty_record_not_a_dropped_one() -> None:
    """Dropping it would leave seven honest points, and ``swept`` would stay TRUE over evidence
    that silently shrank; an empty record neither ran steps nor OOM'd, so ``swept`` fails."""
    base = _load(SIZING_RNAFM)
    for junk in ("x", 7, ["x"]):
        assert S.derive_clauses(_mutated(base, "measurements/0", junk))["swept"] is False, junk


def test_an_unhashable_arm_name_names_no_arm() -> None:
    base = _load(EVAL_RNAFM)
    assert E.derive_clauses(base)["both_ablation_arms_present"] is True
    clauses = E.derive_clauses(_mutated(base, "ablation/with_aux_arm", ["aux1.0_lr1e-4"]))
    assert clauses["both_ablation_arms_present"] is False


def test_an_honest_nan_temperature_is_a_failed_clause_not_a_malformed_report() -> None:
    """``is_real`` refuses only what a float cannot hold. A diverged fit writing NaN is an
    honest failure the clause grades; calling it malformed would invite someone to "fix" it."""
    mutated = _mutated(_load(EVAL_RNAFM), "arms/aux0.0_lr1e-4/calibration/temperature", math.nan)
    problems = E.validate_report(mutated)
    assert problems  # the recorded clause no longer re-derives
    assert not any(p.startswith(RSCH.MALFORMED) for p in problems), problems


# --------------------------------------------------------------------------------------------- #
# The schema-1 sizing recommendation is graded in its own schema's shape
# --------------------------------------------------------------------------------------------- #
def test_the_production_report_is_schema_1_with_the_legacy_recommendation() -> None:
    report = _load(SIZING_PROD)
    assert report["schema_version"] == "1"
    assert report["schema_version"] in S.LEGACY_SCHEMAS_WITH_ASSERTED_HEADROOM
    assert report["recommendation"] == {
        "batch_size": 4,
        "basis": S.LEGACY_RECOMMENDATION_BASIS,
    }


def test_a_schema_1_batch_size_is_still_re_derived_from_the_measurements() -> None:
    """The exemption is for the basis STRING's age, never for the number a reader acts on."""
    report = _mutated(_load(SIZING_PROD), "recommendation/batch_size", 8)
    assert any("recommendation" in p for p in S.validate_report(report))


def test_a_schema_1_report_whose_measurements_move_fails_its_recommendation() -> None:
    """Make batch 8 the largest worst-case point that ran; the recorded 4 no longer follows."""
    report = _mutated(_load(SIZING_PROD), "measurements/0/oom", False)
    report["measurements"][0]["peak_vram_gib"] = 15.0
    assert any("recommendation" in p for p in S.validate_report(report))


@pytest.mark.parametrize("basis", ["", "fits", S.LEGACY_RECOMMENDATION_BASIS + "."])
def test_a_schema_1_basis_is_matched_verbatim(basis: str) -> None:
    report = _mutated(_load(SIZING_PROD), "recommendation/basis", basis)
    assert any("recommendation" in p for p in S.validate_report(report))


def test_a_current_shape_recommendation_on_a_schema_1_report_is_refused() -> None:
    """Schema 1's producer could not have written it: a hand-edit, not an age difference."""
    report = _load(SIZING_PROD)
    fitting = [m for m in report["measurements"] if not m["oom"] and m["regime"] == "worst_case"]
    report["recommendation"] = S._recommend(fitting, 4, report["device"])
    assert any("recommendation" in p for p in S.validate_report(report))


@pytest.mark.parametrize("schema", ["4", "2", "9", 1, ["1"]])
def test_the_legacy_shape_is_excused_only_at_schema_1(schema: Any) -> None:
    """Relabel the production report: at any other schema — current, intermediate, unknown, a
    JSON number, an unhashable list — the legacy recommendation is graded as current."""
    problems = S.validate_report(_mutated(_load(SIZING_PROD), "schema_version", schema))
    assert any(p.startswith("recommendation ") for p in problems), problems


def test_a_current_schema_report_is_still_graded_against_the_computed_headroom_shape() -> None:
    report = _load(SIZING_RNAFM)
    report["recommendation"] = {"batch_size": 8, "basis": S.LEGACY_RECOMMENDATION_BASIS}
    assert any(p.startswith("recommendation ") for p in S.validate_report(report))


# --------------------------------------------------------------------------------------------- #
# The shared shape helpers
# --------------------------------------------------------------------------------------------- #
def test_as_mapping_agrees_with_the_or_idiom_wherever_the_idiom_did_not_raise() -> None:
    for value in ({"a": 1}, {}, None, 0, "", [], False):
        assert RSCH.as_mapping(value) == (value or {})
    for value in ("x", 1, [1], True, 1.5):
        assert RSCH.as_mapping(value) == {}


def test_is_count_refuses_bools_floats_and_negatives() -> None:
    assert RSCH.is_count(0) and RSCH.is_count(6)
    for value in (True, False, 1.0, -1, "1", None):
        assert not RSCH.is_count(value), value


def test_is_real_refuses_only_what_a_float_cannot_hold() -> None:
    for value in (0, -3, 1.5, math.nan, math.inf, 10**300):
        assert RSCH.is_real(value), value
    for value in (10**400, True, "1", None, [1]):
        assert not RSCH.is_real(value), value


def test_shape_problems_absence_none_nullable_and_each() -> None:
    rules = (
        (("a",), RSCH.is_mapping, "a mapping"),
        (("b",), RSCH.nullable(RSCH.is_mapping), "a mapping"),
        (("xs", RSCH.EACH, "n"), RSCH.is_count, "a count"),
        (("m", RSCH.EACH), RSCH.is_mapping, "a mapping"),
    )
    assert RSCH.shape_problems({}, rules) == []  # absence is never a shape problem
    assert RSCH.shape_problems({"b": None}, rules) == []  # nullable
    assert RSCH.shape_problems({"a": None}, rules) == [
        f"{RSCH.MALFORMED} a is NoneType, not a mapping"
    ]
    assert RSCH.shape_problems({"xs": [{"n": 1}, {"n": "2"}, {}]}, rules) == [
        f"{RSCH.MALFORMED} xs/1/n is str, not a count"
    ]
    assert RSCH.shape_problems({"m": {"k": 3}}, rules) == [
        f"{RSCH.MALFORMED} m/k is int, not a mapping"
    ]
    # A malformed parent is not descended into: it is reported at its own rule, once.
    assert RSCH.shape_problems({"xs": "abc"}, rules) == []
