"""P2-14 — GATE-4: the gated statistic, the graded population, and the non-gated companions.

Torch-free by construction (the CI unit tier has numpy but no torch): every test here works
on synthetic label vectors, synthetic confusion vectors, and synthetic *reports*. The
forward pass is exercised by ``tests/ml/`` and by the run itself.

The organising discipline is the repo's: a clause is only worth having if it **bites**, so
every clause in ``gate4_problems`` / ``gate4_eval_problems`` is sabotaged one at a time and
must name its own violation ([[sabotage-attribution-must-name-the-test]]).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PDB_FIXTURE = REPO / "tests" / "fixtures" / "pdb_element_extents" / "pdb_element_extents.json"


# ══════════════════════════════════════════════════════════════════════════════
# Run extraction + Specifier exact-3-nt-codon detection (reported, non-gated)
# ══════════════════════════════════════════════════════════════════════════════
def test_class_runs_are_maximal_half_open_and_include_a_trailing_run():
    from tbox_finder.metrics import class_runs

    assert class_runs([0, 2, 2, 2, 0, 2], 2) == [(1, 4), (5, 6)]
    assert class_runs([2, 2], 2) == [(0, 2)]  # a run that reaches the end is closed
    assert class_runs([0, 0], 2) == []
    assert class_runs([], 2) == []


def _codon_case(true_at: int | None, pred: list[tuple[int, int]], n: int = 12):
    """Build one ``(y_true, y_pred)`` record: a 3-nt Specifier at ``true_at`` (or none),
    and predicted Specifier runs at the given half-open spans."""
    from tbox_finder.metrics import CLASS_INDEX

    s = CLASS_INDEX["Specifier"]
    y_true = [0] * n
    if true_at is not None:
        y_true[true_at : true_at + 3] = [s] * 3
    y_pred = [0] * n
    for a, b in pred:
        y_pred[a:b] = [s] * (b - a)
    return y_true, y_pred


def test_specifier_exact_codon_separates_exact_from_shifted_from_fragmented():
    """Three nested strictnesses, so a shortfall is attributable rather than just low.

    An IoU-style measure cannot make these distinctions — it is set-wise, so one 3-nt call
    and three scattered 1-nt calls with the same overlap score identically. That is why D6
    asks for *exact-codon detection* as its own sanity check.
    """
    from tbox_finder.metrics import specifier_exact_codon_detection

    got = specifier_exact_codon_detection(
        [
            _codon_case(4, [(4, 7)]),  # exact
            _codon_case(4, [(5, 8)]),  # right length, shifted 1 nt, overlapping
            _codon_case(4, [(4, 5), (6, 7)]),  # fragmented, overlapping
            _codon_case(4, []),  # missed entirely
            _codon_case(4, [(9, 12)]),  # right length, disjoint — NOT overlapping
        ]
    )
    assert got["n_scorable"] == 5
    assert got["n_exact"] == 1
    assert got["n_right_length"] == 3  # exact + shifted + disjoint
    assert got["n_overlapping"] == 3  # exact + shifted + fragmented
    assert got["exact_rate"] == pytest.approx(1 / 5)
    assert got["gated"] is False


def test_specifier_exact_codon_excludes_label_side_artifacts_instead_of_scoring_them_as_misses():
    """A record whose *truth* is not one clean 3-nt codon is a label defect, not a model
    error — 393 corpus records carry no specifier and ~155 a clamped extent. Counting them
    as misses would report annotation coverage as segmentation quality (§10.3)."""
    from tbox_finder.metrics import CLASS_INDEX, specifier_exact_codon_detection

    s = CLASS_INDEX["Specifier"]
    no_specifier = ([0] * 8, [0] * 8)
    two_nt = ([0, 0, s, s, 0, 0, 0, 0], [0, 0, s, s, 0, 0, 0, 0])
    fragmented_truth = ([0, s, 0, s, 0, s, 0, 0], [0, s, 0, s, 0, s, 0, 0])
    got = specifier_exact_codon_detection([no_specifier, two_nt, fragmented_truth])
    assert got["n_records"] == 3
    assert got["n_scorable"] == 0
    assert got["excluded_by_reason"] == {
        "truth_has_no_specifier": 1,
        "truth_specifier_fragmented": 1,
        "truth_specifier_len_2": 1,
    }
    # ⚠ A rate over an empty denominator is UNDEFINED. A silent 0.0 here reads as a
    # measured total failure, which is the opposite of "nothing was measurable".
    assert got["exact_rate"] is None
    assert got["overlapping_rate"] is None


def test_specifier_exact_codon_refuses_mismatched_lengths():
    from tbox_finder.metrics import specifier_exact_codon_detection

    with pytest.raises(ValueError):
        specifier_exact_codon_detection([([0, 0, 0], [0, 0])])


# ══════════════════════════════════════════════════════════════════════════════
# The confusion route == the pinned kernels (no second arithmetic)
# ══════════════════════════════════════════════════════════════════════════════
def _expressive_pair():
    """A truth/prediction pair with every class present and errors in both directions."""
    from tbox_finder.metrics import CLASS_ORDER

    n_classes = len(CLASS_ORDER)
    y_true, y_pred = [], []
    for c in range(n_classes):
        y_true.extend([c] * 9)
        # 6 correct, 2 confused with the next class, 1 dropped to background
        y_pred.extend([c] * 6 + [(c + 1) % n_classes] * 2 + [0])
    return y_true, y_pred


def test_confusion_route_reproduces_the_pinned_f1_and_iou_kernels():
    """``f1_by_class_from_confusion`` / ``iou_by_class_from_confusion`` must be the SAME
    numbers as ``per_nt_f1_by_class`` / ``boundary_iou_by_element`` — the confusion route is
    a re-association of identical counts (it exists so a 21M-nt read is affordable), never a
    second, drift-prone formula ([[promote-dont-duplicate-is-a-correctness-rule]])."""
    np = pytest.importorskip("numpy")
    from tbox_finder import metrics as M
    from tbox_finder.eval.gate4 import f1_by_class_from_confusion, iou_by_class_from_confusion

    y_true, y_pred = _expressive_pair()
    total = np.zeros((len(M.CLASS_ORDER), 3), dtype=np.int64)
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    for c in range(len(M.CLASS_ORDER)):
        total[c] = [
            int(np.sum((t == c) & (p == c))),
            int(np.sum((t != c) & (p == c))),
            int(np.sum((t == c) & (p != c))),
        ]

    assert f1_by_class_from_confusion(total) == pytest.approx(M.per_nt_f1_by_class(y_true, y_pred))
    assert iou_by_class_from_confusion(total) == pytest.approx(
        M.boundary_iou_by_element(y_true, y_pred)
    )
    # And the fixture is expressive, not degenerate: nothing is perfect, nothing is zero.
    f1s = list(f1_by_class_from_confusion(total).values())
    assert all(0.0 < v < 1.0 for v in f1s), f1s


def test_iou_from_confusion_is_nan_on_an_empty_union_not_zero():
    from tbox_finder.metrics import element_extent_iou, iou_from_confusion

    assert math.isnan(iou_from_confusion(0, 0, 0))
    assert math.isnan(element_extent_iou([0, 0], [0, 0], 3))


def test_summed_confusions_returns_none_on_empty_rather_than_a_zero_matrix():
    """A zero matrix would score every class as an honest NaN and hide that NOTHING was
    measured — the [[clauses-must-guard-emptiness]] shape."""
    from tbox_finder.eval.gate4 import summed_confusions

    assert summed_confusions([]) is None


# ══════════════════════════════════════════════════════════════════════════════
# The cross-route consistency check (CodeRabbit r1): one statistic, two exact routes
# ══════════════════════════════════════════════════════════════════════════════
def test_same_number_treats_two_nans_as_agreement_not_disagreement():
    """NaN is the honest score for an element no position exercises, and ``nan == nan`` is
    False — so a naive equality check would report two AGREEING routes as disagreeing on
    exactly the undefined elements, i.e. the check would fire on every clean run and be
    turned off. ``None`` matches only ``None``."""
    from tbox_finder.eval.gate4 import same_number

    nan = float("nan")
    assert same_number(nan, nan)
    assert not same_number(nan, 0.0)
    assert not same_number(0.0, nan)
    assert same_number(None, None)
    assert not same_number(None, 0.0)
    assert same_number(0.5, 0.5)
    assert not same_number(0.5, 0.5 + 1e-6)
    # float noise from two orders of summation is agreement, not a defect
    assert same_number(0.1 + 0.2, 0.3)


def _agreeing_routes():
    """Both routes on one expressive fixture — the clean-run state the check must not fire on."""
    np = pytest.importorskip("numpy")
    from tbox_finder import metrics as M
    from tbox_finder.eval.gate4 import f1_by_class_from_confusion, iou_by_class_from_confusion

    y_true, y_pred = _expressive_pair()
    total = np.zeros((len(M.CLASS_ORDER), 3), dtype=np.int64)
    t, p = np.asarray(y_true), np.asarray(y_pred)
    for c in range(len(M.CLASS_ORDER)):
        total[c] = [
            int(np.sum((t == c) & (p == c))),
            int(np.sum((t != c) & (p == c))),
            int(np.sum((t == c) & (p != c))),
        ]
    gate = M.gate4_core_min_f1(y_true, y_pred)
    return {
        "pooled_f1": M.per_nt_f1_by_class(y_true, y_pred),
        "pooled_iou": M.boundary_iou_by_element(y_true, y_pred),
        "confusion_f1": f1_by_class_from_confusion(total),
        "confusion_iou": iou_by_class_from_confusion(total),
        "pooled_min_f1": gate["min_f1"],
        "confusion_min_f1": min(gate["per_element_f1"].values()),
    }


def test_cross_route_check_is_silent_when_the_two_routes_agree():
    from tbox_finder.eval.gate4 import cross_route_disagreements

    assert cross_route_disagreements(**_agreeing_routes()) == []


@pytest.mark.parametrize(
    "field,key,expect",
    [
        ("confusion_f1", "Stem_I", "F1[Stem_I]"),
        ("confusion_f1", "Terminator", "F1[Terminator]"),
        ("confusion_iou", "Specifier", "IoU[Specifier]"),
        ("confusion_min_f1", None, "gate4.min_f1"),
    ],
)
def test_cross_route_check_bites_on_each_perturbed_score(field, key, expect):
    """One route perturbed at one score ⇒ that score is NAMED. Parametrised per quantity
    because a check that only notices `min_f1` would let a wrong per-class F1 or IoU into the
    non-gated half of the report unremarked ([[sabotage-attribution-names-the-test]])."""
    from tbox_finder.eval.gate4 import cross_route_disagreements

    routes = _agreeing_routes()
    if key is None:
        routes[field] = float(routes[field]) - 0.01
    else:
        routes[field] = {**routes[field], key: float(routes[field][key]) - 0.01}
    problems = cross_route_disagreements(**routes)
    assert len(problems) == 1, problems
    assert problems[0].startswith(expect), problems


def test_cross_route_check_bites_when_one_route_alone_is_nan():
    """The asymmetric case: a route that silently drops an element to NaN while the other
    scores it is exactly what two implementations drifting looks like, and it must NOT be
    absorbed by the NaN-tolerance that the both-NaN case needs."""
    from tbox_finder.eval.gate4 import cross_route_disagreements

    routes = _agreeing_routes()
    routes["confusion_iou"] = {**routes["confusion_iou"], "Antiterminator_Tbox_seq": float("nan")}
    problems = cross_route_disagreements(**routes)
    assert [p.split(":")[0] for p in problems] == ["IoU[Antiterminator_Tbox_seq]"], problems


# ══════════════════════════════════════════════════════════════════════════════
# Macro-over-blocks (ADR-0005 D5) — undefined blocks are excluded, never 0-filled
# ══════════════════════════════════════════════════════════════════════════════
def test_macro_over_blocks_excludes_undefined_blocks_and_counts_them():
    from tbox_finder.eval.gate4 import macro_over_blocks

    per_block = {
        "OrderA": {"Terminator": 0.90},
        "OrderB": {"Terminator": 0.70},
        # An order with no Terminator at all: F1 undefined, NOT a zero.
        "OrderC": {"Terminator": float("nan")},
    }
    got = macro_over_blocks(per_block, "Terminator")
    assert got["macro"] == pytest.approx(0.80)  # not 0.5333 (the 0-filled average)
    assert got["n_blocks_defined"] == 2
    assert got["n_blocks_undefined"] == 1
    assert got["min"] == pytest.approx(0.70)
    assert got["max"] == pytest.approx(0.90)


def test_macro_over_blocks_is_none_when_no_block_defines_the_element():
    from tbox_finder.eval.gate4 import macro_over_blocks

    got = macro_over_blocks({"OrderA": {"Terminator": float("nan")}}, "Terminator")
    assert got["macro"] is None
    assert got["n_blocks_defined"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# The 9-PDB cross-source label-noise ceiling (D6 δ-governance)
# ══════════════════════════════════════════════════════════════════════════════
def test_pdb_ceiling_re_derives_non_estimability_from_the_committed_fixture():
    """Re-derived from the fixture's own resolved-class map, not copied from its prose: 0 of
    the 9 depositions resolve Specifier or Antiterminator to a residue extent, so no
    cross-source per-nt F1 over the gated elements exists and D6's δ-recalibration path
    stays CLOSED — the floor is 0.80 whatever GATE-4 measures."""
    from tbox_finder.eval.gate4 import pdb_label_noise_ceiling

    got = pdb_label_noise_ceiling(json.loads(PDB_FIXTURE.read_text()))
    assert got["estimable"] is False
    assert got["ceiling_c"] is None
    assert got["unresolved_core_elements"] == ["Antiterminator_Tbox_seq", "Specifier"]
    assert got["resolved_depositions_by_class"]["Stem_I"] == 5
    assert got["floor_unchanged"] == 0.80
    assert got["gated"] is False


def test_pdb_ceiling_flips_to_estimable_when_every_core_element_resolves():
    """The clause is LIVE, not a constant: a future deposition resolving the missing core
    elements re-opens D6's δ question on its own, and this proves the code would say so."""
    from tbox_finder.eval.gate4 import pdb_label_noise_ceiling

    fixture = {
        "entries": {"X": {}, "Y": {}},
        "label_noise_ceiling": {
            "cross_source_resolved_classes": {
                "Stem_I": ["X"],
                "Specifier": ["X", "Y"],
                "Antiterminator_Tbox_seq": ["Y"],
            }
        },
    }
    got = pdb_label_noise_ceiling(fixture)
    assert got["estimable"] is True
    assert got["unresolved_core_elements"] == []
    assert "ADR-0004 re-sign-off" in got["delta_governance"]


def test_pdb_ceiling_prose_reports_the_derived_count_and_floor_not_baked_in_literals():
    """The δ-governance sentence is a REPORTED number, so it must track the evidence.

    The block re-derives `estimable` from the fixture precisely so a tenth deposition flips it
    on its own — but the sentence beside it used to say "0 of the 9" and "the floor stays 0.80"
    as literals, which would keep asserting nine depositions and a 0.80 floor after the fixture
    grew or ADR-0004 re-signed δ. A wrong number in a shipped report reads exactly like a right
    one (CLAUDE.md §10.3). CodeRabbit, r1.
    """
    from tbox_finder.eval.gate4 import pdb_label_noise_ceiling
    from tbox_finder.power import GATE4_F1_FLOOR

    fixture = {
        "entries": {f"E{i}": {} for i in range(12)},  # NOT 9
        "label_noise_ceiling": {"cross_source_resolved_classes": {"Stem_I": ["E0", "E1"]}},
    }
    msg = pdb_label_noise_ceiling(fixture)["delta_governance"]
    assert "0 of the 12" in msg, msg
    assert "N<=12" in msg, msg
    assert "of the 9" not in msg and "N<=9" not in msg, msg
    assert f"floor stays {GATE4_F1_FLOOR}" in msg, msg

    # …and on the committed 9-deposition fixture it still says 9, so the derivation is not
    # just "any number": the two assertions together pin it to the fixture, not to a constant.
    committed = pdb_label_noise_ceiling(json.loads(PDB_FIXTURE.read_text()))
    assert "0 of the 9" in committed["delta_governance"]
    assert committed["n_depositions"] == 9


def test_pdb_ceiling_prose_degrades_honestly_when_the_deposition_count_is_unknown():
    """`n_entries` is None when the fixture carries no `entries` block — the sentence must not
    invent a count (an f-string over None would render the literal "0 of the None")."""
    from tbox_finder.eval.gate4 import pdb_label_noise_ceiling

    got = pdb_label_noise_ceiling(
        {"label_noise_ceiling": {"cross_source_resolved_classes": {"Stem_I": ["E0"]}}}
    )
    assert got["n_depositions"] is None
    assert "None" not in got["delta_governance"], got["delta_governance"]
    assert "the depositions supplied" in got["delta_governance"]


def test_pdb_ceiling_refuses_to_assert_non_estimability_without_the_evidence():
    """Asserting "not estimable" with no evidence block is the same fabrication as asserting
    a number (§10.3)."""
    from tbox_finder.eval.gate4 import pdb_label_noise_ceiling

    with pytest.raises(ValueError, match="label_noise_ceiling"):
        pdb_label_noise_ceiling({"entries": {}})
    with pytest.raises(ValueError, match="cross_source_resolved_classes"):
        pdb_label_noise_ceiling({"label_noise_ceiling": {"cross_source_resolved_classes": {}}})


# ══════════════════════════════════════════════════════════════════════════════
# The graded population's own clauses (window_dataset.gate4_eval_problems)
# ══════════════════════════════════════════════════════════════════════════════
def _good_gate4_scope() -> dict:
    """The shape measured on the committed table (1,201 / 1,029 / 7,099)."""
    return {
        "fold_scope": "gate4_eval",
        "n_records": 1201,
        "n_blocks": 1029,
        "carve": {"fold_random_value": "test", "n_gate4_clusters": 1031},
        "leakage": {
            "n_designated_loo_holdout": 0,
            "n_not_nested_train": 0,
            "shared_record_ids_with_twin_train": 0,
            "shared_cluster_ids_with_twin_train": 0,
            "n_twin_train_records": 7099,
            "n_twin_train_blocks": 3744,
        },
    }


def test_gate4_eval_problems_accepts_the_measured_scope():
    from tbox_finder.data.window_dataset import gate4_eval_problems

    assert gate4_eval_problems(_good_gate4_scope()) == []


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda s: s.pop("leakage"), "leakage"),
        (lambda s: s.__setitem__("leakage", {}), "leakage"),
        # The population must be in-distribution: 786 is the measured count of designated
        # LOO records in the scheme-A test fold, i.e. the wrong-population failure.
        (
            lambda s: s["leakage"].__setitem__("n_designated_loo_holdout", 786),
            "leave-one-order-out",
        ),
        (lambda s: s["leakage"].__setitem__("n_not_nested_train", 1149), "outside the D5"),
        (
            lambda s: s["leakage"].__setitem__("shared_record_ids_with_twin_train", 1),
            "train-on-train",
        ),
        (
            lambda s: s["leakage"].__setitem__("shared_cluster_ids_with_twin_train", 12),
            "homology-leaky",
        ),
        # 0 overlap against an EMPTY training fold is vacuously true.
        (lambda s: s["leakage"].__setitem__("n_twin_train_records", 0), "vacuously true"),
        (lambda s: s.pop("carve"), "carve"),
        (lambda s: s["carve"].__setitem__("fold_random_value", "val"), "fold_random_value"),
        (lambda s: s["carve"].__setitem__("n_gate4_clusters", 0), "n_gate4_clusters"),
        (lambda s: s.__setitem__("fold_scope", "selection_val"), "fold_scope"),
        (lambda s: s.__setitem__("n_records", 0), "n_records"),
        (lambda s: s.__setitem__("n_blocks", 1), "block-resamplable"),
        # isinstance(True, int) is True — the bool-as-count trap.
        (lambda s: s["leakage"].__setitem__("n_twin_train_records", True), "positive int"),
    ],
)
def test_gate4_eval_problems_bites_on_each_violation(mutate, expect):
    from tbox_finder.data.window_dataset import gate4_eval_problems

    scope = _good_gate4_scope()
    mutate(scope)
    problems = gate4_eval_problems(scope)
    assert problems, f"sabotage {expect!r} was not caught — the clause is vacuous"
    assert any(expect in p for p in problems), f"{expect!r} not in {problems}"


# ══════════════════════════════════════════════════════════════════════════════
# The GATE-4 report's own clauses (eval.gate4.gate4_problems)
# ══════════════════════════════════════════════════════════════════════════════
def _good_report() -> dict:
    from tbox_finder.infer.reconcile import OPERATOR

    return {
        "schema_version": "1",
        "step": "P2-14",
        "posterior": "uncalibrated",
        "operator": OPERATOR,
        "twin": {
            "fold_scope": "gate4_twin_train",
            "exclude_gate4_eval_config": True,
            "exclude_gate4_eval_measured": True,
            "n_gate4_eval_excluded": 1204,
            "n_gate4_eval_clusters": 1031,
            "n_training_records": 7099,
            "label_source": "production",
            "checkpoint_identity_verified": True,
        },
        # `full_fold` / `max_records` are added by `compute_gate4`, not by the loader, so
        # they are absent from `_good_gate4_scope()` and present here — the real shape.
        "scope": {**_good_gate4_scope(), "full_fold": True, "max_records": None},
        "gate": {
            "per_element_f1": {
                "Stem_I": 0.93,
                "Specifier": 0.88,
                "Antiterminator_Tbox_seq": 0.91,
            },
            "min_f1": 0.88,
            "floor": 0.80,
            "passes": True,
            "block_bootstrap_ci": {"n_blocks": 1029, "lo": 0.86, "hi": 0.90},
            "n_records": 1201,
            "n_positions": 2976259,
        },
        "reported_non_gated": {
            "per_nt_f1_by_class": {"background": 0.99},
            "boundary_iou_by_element": {"Stem_I": 0.9},
            "non_core_classes": {"Stem_II": {"per_nt_f1": 0.7, "boundary_iou": 0.6}},
            "specifier_exact_codon": {"n_scorable": 1190, "exact_rate": 0.8},
            "pdb_label_noise_ceiling": {"estimable": False, "ceiling_c": None},
        },
        "loo_holdout": {
            "gated": False,
            "macro_per_element_f1": {},
            "scope": {"full_fold": True},
        },
        "geometry": {"window_nt": 1024, "stride_nt": 512, "n_boot": 2000},
    }


def test_gate4_problems_accepts_a_clean_report():
    from tbox_finder.eval.gate4 import gate4_problems

    assert gate4_problems(_good_report()) == []


def test_the_committed_gate4_report_still_re_derives_clean():
    """The shipped P2-14 verdict must survive every clause added after it was measured.

    Adding a clause is how a previously-valid committed artifact silently becomes
    uncertifiable ([[new-gate-clause-invalidates-old-reports]]). This runs the CURRENT
    validator over the REAL committed report, so a future clause that the measured run
    cannot satisfy goes red here rather than being discovered at the phase-exit gate —
    and it cannot be fixed by re-writing the report, because re-running is the only
    honest way to change a measurement.
    """
    from tbox_finder.eval.gate4 import gate4_problems

    path = Path(__file__).resolve().parents[2] / "reports" / "p2" / "gate4.json"
    assert path.exists(), f"{path} is the P2-14 deliverable and must stay committed"
    report = json.loads(path.read_text())
    assert gate4_problems(report) == []
    # The verdict itself, pinned: a clause change must not quietly flip the gate.
    assert report["gate"]["passes"] is True
    assert report["scope"]["full_fold"] is True and report["scope"]["max_records"] is None


@pytest.mark.parametrize(
    "mutate, expect",
    [
        # ── THE clause of this step: the shipped checkpoint is not gradeable here ──
        (lambda r: r["twin"].__setitem__("fold_scope", "train"), "gate4_twin_train"),
        (lambda r: r.pop("twin"), "twin"),
        (lambda r: r["twin"].__setitem__("exclude_gate4_eval_measured", False), "withholding"),
        # A flag that was SET but excluded nothing — how this clause fabricates TRUE.
        (lambda r: r["twin"].__setitem__("n_gate4_eval_excluded", 0), "excluded NOTHING"),
        (
            lambda r: r["twin"].__setitem__("checkpoint_identity_verified", False),
            "DIFFERENT checkpoint",
        ),
        (lambda r: r["twin"].__setitem__("label_source", "naive"), "PRODUCTION scanner"),
        # ── the gated statistic, re-derived ──
        # 0.9067 is the MEAN of the three; D6 gates the MIN.
        (lambda r: r["gate"].__setitem__("min_f1", 0.9066666666666666), "MIN, not a mean"),
        (lambda r: r["gate"].__setitem__("passes", False), "re-derives"),
        (lambda r: (r["gate"].__setitem__("min_f1", 0.5)), "re-derives"),
        (lambda r: r["gate"].__setitem__("floor", 0.5), "blinded-frozen"),
        (lambda r: r["gate"]["per_element_f1"].pop("Specifier"), "missing core element"),
        (lambda r: r["gate"].pop("per_element_f1"), "per_element_f1"),
        (lambda r: r["gate"].__setitem__("n_positions", 0), "n_positions"),
        (
            lambda r: r["gate"]["block_bootstrap_ci"].__setitem__("n_blocks", 469),
            "does not describe",
        ),
        (lambda r: r["gate"].pop("block_bootstrap_ci"), "block_bootstrap_ci"),
        (lambda r: r.pop("gate"), "gate"),
        # ── posterior + operator ──
        (lambda r: r.__setitem__("posterior", "temperature_scaled"), "A11 Pin 4"),
        (lambda r: r.__setitem__("operator", "argmax_of_last_window"), "512-grid artifact"),
        # ── the population's own clauses, surfaced through the report ──
        (
            lambda r: r["scope"]["leakage"].__setitem__("n_designated_loo_holdout", 786),
            "scope.leakage",
        ),
        (lambda r: r.pop("scope"), "scope"),
        # ── the non-gated companions D6 requires ALONGSIDE the gate ──
        (lambda r: r["reported_non_gated"].pop("specifier_exact_codon"), "specifier_exact_codon"),
        (
            lambda r: r["reported_non_gated"].pop("pdb_label_noise_ceiling"),
            "pdb_label_noise_ceiling",
        ),
        (lambda r: r["reported_non_gated"].pop("boundary_iou_by_element"), "boundary_iou"),
        (lambda r: r.pop("reported_non_gated"), "reported_non_gated"),
        (
            lambda r: r["reported_non_gated"]["pdb_label_noise_ceiling"].__setitem__(
                "estimable", True
            ),
            "re-sign-off",
        ),
        (
            lambda r: r["reported_non_gated"]["non_core_classes"].__setitem__("Specifier", {}),
            "excluded from the gate",
        ),
        # ── the LOO read must stay non-gated at P2 ──
        (lambda r: r["loo_holdout"].__setitem__("gated", True), "REPORTED at"),
        # ── completeness: a cost knob must never produce a certified grade ──
        # Every other clause survives truncation unharmed (a prefix of a disjoint
        # population is still disjoint), so absent these the cheap smoke writes the
        # phase-exit deliverable and it reads exactly like a full grade. CodeRabbit, r5.
        (lambda r: r["scope"].__setitem__("full_fold", False), "scope.full_fold"),
        (lambda r: r["scope"].pop("full_fold"), "scope.full_fold"),
        (lambda r: r["scope"].__setitem__("max_records", 8), "max_records"),
        # `--no-loo` sets the block to None: the clause must fire on ABSENCE, which is
        # exactly where a `isinstance(x, Mapping) and x and ...` guard goes vacuous.
        (lambda r: r.__setitem__("loo_holdout", None), "loo_holdout: block missing"),
        (lambda r: r.pop("loo_holdout"), "loo_holdout: block missing"),
        # `--loo-max-records` truncates the ORDER set the D5 macro is taken over.
        (lambda r: r["loo_holdout"]["scope"].__setitem__("full_fold", False), "truncated order"),
        (lambda r: r["loo_holdout"].pop("scope"), "truncated order"),
        # ── the A6 identity pins: a positive count cannot tell 1,204 from 900 ──
        # Every clause above is rule-shaped and a DRIFTED split table satisfies all of them:
        # a re-clustered or re-drawn fold is still gate4_eval, still disjoint, still
        # resamplable. Counts alone are also blind to a swap
        # ([[symmetric-count-fixture-blind-to-inversion]]), which is why the pins are on the
        # population's size AND its nucleotide extent. CodeRabbit, r6.
        (lambda r: r["twin"].__setitem__("n_gate4_eval_excluded", 900), "pins it at 1204"),
        (lambda r: r["twin"].__setitem__("n_gate4_eval_clusters", 1030), "pins it at 1031"),
        (lambda r: r["twin"].__setitem__("n_training_records", 7100), "pins it at 7099"),
        (lambda r: r["twin"].pop("n_gate4_eval_clusters"), "pins it at 1031"),
        (lambda r: r["gate"].__setitem__("n_positions", 2_976_258), "different nucleotide"),
        (lambda r: r["scope"].__setitem__("n_records", 1200), "population drift"),
        (lambda r: r["scope"].__setitem__("n_blocks", 1028), "population drift"),
        # ── the last cost knob: a cheap CI reads exactly like the shipped one ──
        (lambda r: r["geometry"].__setitem__("n_boot", 50), "not the ADR-0005 D5 estimator"),
        (lambda r: r["geometry"].__setitem__("stride_nt", 1024), "frozen scan geometry"),
        (lambda r: r["geometry"].__setitem__("window_nt", 2048), "frozen scan geometry"),
        (lambda r: r.pop("geometry"), "frozen scan geometry"),
    ],
)
def test_gate4_problems_bites_on_each_violation(mutate, expect):
    """Every clause must FAIL on its own violation — one at a time, by sabotage. A red list
    proves *some* clause bit; asserting the message proves it was **this** one
    ([[sabotage-attribution-must-name-the-test]])."""
    from tbox_finder.eval.gate4 import gate4_problems

    report = _good_report()
    mutate(report)
    problems = gate4_problems(report)
    assert problems, f"sabotage {expect!r} was not caught — the clause is vacuous"
    assert any(expect in p for p in problems), f"{expect!r} not in {problems}"


def test_gate4_min_f1_clause_accepts_a_genuine_nan_but_never_a_pass():
    """An unmeasurable core element makes the gated statistic NaN, and NaN must never
    certify: ``gate4_pass(nan)`` is False and the clause must agree with it."""
    from tbox_finder.eval.gate4 import gate4_problems

    report = _good_report()
    report["gate"]["per_element_f1"]["Specifier"] = float("nan")
    report["gate"]["min_f1"] = float("nan")
    report["gate"]["passes"] = False
    assert gate4_problems(report) == []

    report["gate"]["passes"] = True
    problems = gate4_problems(report)
    assert any("re-derives" in p for p in problems), problems


def test_the_gate_floor_is_single_sourced_from_power():
    """No local 0.80. The floor is blinded-frozen at P0 and lives in one place (ADR-0004
    D6/A2); a second copy is how a threshold drifts without an ADR."""
    from tbox_finder.eval import gate4 as G
    from tbox_finder.power import GATE4_F1_FLOOR

    assert G.GATE4_F1_FLOOR is GATE4_F1_FLOOR
    src = (REPO / "src" / "tbox_finder" / "eval" / "gate4.py").read_text()
    # The number itself may appear in prose; it must never appear as an assignment.
    assert "floor = 0.8" not in src
    assert "FLOOR = 0.8" not in src


def test_non_core_classes_are_exactly_the_four_d6_excludes():
    from tbox_finder.eval.gate4 import NON_CORE_CLASSES

    assert set(NON_CORE_CLASSES) == {"Stem_II", "Stem_III", "Terminator", "Discriminator"}


def test_report_disclosures_name_the_two_run_split_and_the_conservative_direction():
    """The disclosures are load-bearing, not decoration: a reader of ``gate4.json`` must be
    able to see that the graded artifact is not the shipped one, and in which direction the
    difference points (PRD §10.1; ADR-0004 A6)."""
    from tbox_finder.eval.gate4 import build_report

    report = build_report(
        checkpoint="data/processed/checkpoints/stage1_gate4_twin/stage1.pt",
        twin=_good_report()["twin"],
        scope=_good_gate4_scope(),
        gated=_good_report()["gate"],
        non_gated=_good_report()["reported_non_gated"],
        loo=None,
        geometry={"window_nt": 1024, "stride_nt": 512, "n_boot": 2000},
    )
    blob = " ".join(report["disclosures"])
    assert "two-run split" in blob
    assert "CONSERVATIVE" in blob
    assert "NEVER GATED" in blob  # the LOO read
    assert report["gated"] is True
    assert report["posterior"] == "uncalibrated"


def test_disclosures_report_the_evidence_they_were_handed_not_baked_in_literals():
    """The disclosures ship inside `gate4.json` and are the only place a reader learns the
    graded artifact is not the shipped one — so their numbers must track the run that made
    them. Literals here would keep asserting one run's twin size, phylum share, deposition
    count and floor under another's (§10.3). CodeRabbit, r2."""
    from tbox_finder.eval.gate4 import disclosures
    from tbox_finder.power import GATE4_F1_FLOOR

    blob = " ".join(
        disclosures(
            twin={"n_training_records": 4242},
            scope={"n_records": 400, "phylum_counts": {"Actinomycetota": 300, "Bacillota": 100}},
            non_gated={
                "pdb_label_noise_ceiling": {
                    "estimable": False,
                    "n_depositions": 12,
                    "unresolved_core_elements": ["Specifier"],
                }
            },
        )
    )
    assert "4242 records" in blob, blob
    assert "75.0% Actinomycetota (300/400 records)" in blob, blob
    assert "0 of 12 depositions" in blob, blob
    assert f"the {GATE4_F1_FLOOR} floor stands" in blob, blob
    # …and none of the previously-baked-in values survive when the evidence says otherwise.
    for stale in ("7,099 records", "98.7%", "1,188/1,204", "0 of 9 ", "Firmicutes"):
        assert stale not in blob, f"{stale!r} is a literal from another run: {blob}"


@pytest.mark.parametrize(
    "twin,scope,non_gated,expect_absent,expect_refusal",
    [
        (
            {},
            {"n_records": 4, "phylum_counts": {"B": 4}},
            {},
            "records — strictly fewer",
            "NOT re-derivable from this report",
        ),
        (
            {"n_training_records": 7},
            {"n_records": 0, "phylum_counts": {}},
            {},
            "% ",
            "NOT re-derivable from this report",
        ),
        (
            {"n_training_records": 7},
            {"n_records": 4, "phylum_counts": {"B": 4}},
            {},
            "0 of",
            "NOT reported here",
        ),
    ],
    ids=["twin-size-missing", "phylum-counts-missing", "ceiling-missing"],
)
def test_disclosures_refuse_by_name_when_the_evidence_is_absent(
    twin, scope, non_gated, expect_absent, expect_refusal
):
    """Absent evidence must produce a NAMED refusal, never an invented number and never a
    silently-dropped sentence — a disclosure that quietly vanishes is how a reader concludes
    the caveat did not apply ([[clauses-must-guard-emptiness]])."""
    from tbox_finder.eval.gate4 import disclosures

    got = disclosures(twin=twin, scope=scope, non_gated=non_gated)
    assert len(got) == 5, got  # the sentence is still there, saying it cannot say
    blob = " ".join(got)
    assert expect_refusal in blob, blob
    assert expect_absent not in blob, blob
    assert "None" not in blob, blob


def test_disclosures_say_the_delta_path_re_opened_when_the_ceiling_becomes_estimable():
    """The estimable branch is not decoration: it is the sentence that tells a reader D6's
    recalibration question is live again — and it must still refuse to move the floor."""
    from tbox_finder.eval.gate4 import disclosures
    from tbox_finder.power import GATE4_F1_FLOOR

    blob = " ".join(
        disclosures(
            twin={"n_training_records": 7},
            scope={"n_records": 4, "phylum_counts": {"B": 4}},
            non_gated={
                "pdb_label_noise_ceiling": {
                    "estimable": True,
                    "n_depositions": 11,
                    "unresolved_core_elements": [],
                }
            },
        )
    )
    assert "ESTIMABLE from 11 depositions" in blob, blob
    assert "ADR-0004 re-sign-off" in blob, blob
    assert f"still {GATE4_F1_FLOOR}" in blob, blob


@pytest.mark.parametrize("bad", [None, "0.9", {"f1": 0.9}, [0.9], True])
def test_gate4_problems_reports_a_non_numeric_per_element_f1_instead_of_raising(bad):
    """``gate4_problems`` is TOTAL — problems out, never an exception. `float(...)` over the
    per-element F1s raised on exactly the corrupt artifacts the clause exists to catch, and a
    validator that dies reports nothing at all. CodeRabbit, r2."""
    from tbox_finder.eval.gate4 import gate4_problems

    report = _good_report()
    report["gate"]["per_element_f1"] = {**report["gate"]["per_element_f1"], "Specifier": bad}
    problems = gate4_problems(report)  # must not raise
    assert any("non-numeric core element" in p for p in problems), problems
    # …and it must NOT also claim the min disagrees — that re-derivation was skipped, not run
    # on coerced garbage.
    assert not any("is a MIN, not a mean" in p for p in problems), problems


def test_a_report_that_fails_its_own_clauses_is_still_buildable_but_never_clean():
    """``build_report`` is pure assembly — it must NOT silently repair a bad input, or the
    clause set would be checking the builder instead of the run."""
    from tbox_finder.eval.gate4 import build_report, gate4_problems

    bad_twin = copy.deepcopy(_good_report()["twin"])
    bad_twin["fold_scope"] = "train"
    report = build_report(
        checkpoint="x.pt",
        twin=bad_twin,
        scope=_good_gate4_scope(),
        gated=_good_report()["gate"],
        non_gated=_good_report()["reported_non_gated"],
        loo=None,
        geometry={"window_nt": 1024, "stride_nt": 512, "n_boot": 2000},
    )
    assert report["twin"]["fold_scope"] == "train"
    assert any("gate4_twin_train" in p for p in gate4_problems(report))


# ══════════════════════════════════════════════════════════════════════════════
# Serialisation + provenance binding (CodeRabbit r3)
# ══════════════════════════════════════════════════════════════════════════════
def test_the_written_report_is_valid_json_even_though_gate4_produces_nans():
    """`json.dumps` writes bare `NaN` tokens, which Python reads back and nothing else does.

    GATE-4 emits NaN by design — an unmeasurable element must not certify the gate, IoU is
    NaN on an empty union, the CI is NaN below two blocks — so a real run very likely writes
    one, and the phase-exit deliverable would be unparseable by jq / JS / R / Go while looking
    fine from Python. Asserting on the TEXT, not on a Python round-trip, is the whole point:
    `json.loads` accepts `NaN` and would pass either way.
    """
    from tbox_finder.eval.gate4 import json_safe

    report = {
        "gate": {"min_f1": float("nan"), "floor": 0.8},
        "reported_non_gated": {"boundary_iou_by_element": {"Stem_III": float("nan")}},
        "ci": {"lower": float("-inf"), "upper": float("inf"), "point": 0.9},
        "counts": [1, 2, 3],
    }
    text = json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text, text
    back = json.loads(text)
    assert back["gate"]["min_f1"] is None
    assert back["reported_non_gated"]["boundary_iou_by_element"]["Stem_III"] is None
    assert back["ci"] == {"lower": None, "upper": None, "point": 0.9}
    assert back["counts"] == [1, 2, 3]  # finite values pass through untouched


def test_json_safe_converts_numpy_scalars_and_leaves_unknown_types_to_fail_loudly():
    """np.int64 does not subclass int, so an unconverted one raises inside `json.dumps` — and
    an unknown type must keep raising rather than be stringified into a plausible value."""
    np = pytest.importorskip("numpy")
    from tbox_finder.eval.gate4 import json_safe

    got = json_safe({"i": np.int64(7), "f": np.float64("nan"), "b": np.bool_(True)})
    assert got == {"i": 7, "f": None, "b": True}
    assert isinstance(got["i"], int) and isinstance(got["b"], bool)
    with pytest.raises(TypeError):
        json.dumps(json_safe({"bad": {1, 2}}), allow_nan=False)


_TWIN_CKPT = "data/processed/checkpoints/stage1_gate4_twin/stage1.pt"


@pytest.mark.parametrize(
    "outputs,path,expect",
    [
        ({_TWIN_CKPT: "aa"}, _TWIN_CKPT, "aa"),
        # the graded checkpoint is NOT first in insertion order
        (
            {"data/processed/checkpoints/stage1_production/stage1.pt": "pp", _TWIN_CKPT: "tt"},
            _TWIN_CKPT,
            "tt",
        ),
        # same basename under two directories — ambiguous, refuse rather than pick
        ({"a/stage1.pt": "aa", "b/stage1.pt": "bb"}, "c/stage1.pt", None),
        # a lone entry spelled differently still resolves (it must still match by sha256)
        ({"./checkpoints/stage1.pt": "zz"}, _TWIN_CKPT, "zz"),
        # several entries and no path to disambiguate on
        ({"a/x.pt": "aa", "b/y.pt": "bb"}, None, None),
        ({}, _TWIN_CKPT, None),
        (None, _TWIN_CKPT, None),
    ],
    ids=[
        "single-match",
        "not-first",
        "ambiguous-basename",
        "lone-other-spelling",
        "multi-no-path",
        "empty",
        "no-sidecar",
    ],
)
def test_provenance_lookup_binds_to_the_graded_checkpoint_not_the_first_entry(
    outputs, path, expect
):
    """`checkpoint_identity_verified` is what binds the report to the checkpoint BYTES.

    Iterating `outputs` and taking the first value binds it to whichever artifact happens to
    come first — evidence about "some output", not about the thing the clause names."""
    from tbox_finder.eval.gate4 import provenance_output_sha256

    prov = None if outputs is None else {"outputs": outputs}
    assert provenance_output_sha256(prov, path) == expect


def test_twin_evidence_verifies_identity_against_the_matching_sidecar_entry():
    """End-to-end through `twin_evidence`: a multi-output sidecar must still verify the twin."""
    from tbox_finder.eval.gate4 import twin_evidence

    prov = {
        "outputs": {
            "data/processed/checkpoints/stage1_production/stage1.pt": "production-sha",
            _TWIN_CKPT: "twin-sha",
        }
    }
    got = twin_evidence(
        {"class_counts_scope": {"fold_scope": "gate4_twin_train"}},
        checkpoint_sha256="twin-sha",
        checkpoint_provenance=prov,
        checkpoint_path=_TWIN_CKPT,
    )
    assert got["provenance_output_sha256"] == "twin-sha"
    assert got["checkpoint_identity_verified"] is True
    # …and grading the OTHER checkpoint against the same sidecar must not verify.
    other = twin_evidence(
        {"class_counts_scope": {"fold_scope": "gate4_twin_train"}},
        checkpoint_sha256="twin-sha",
        checkpoint_provenance=prov,
        checkpoint_path="data/processed/checkpoints/stage1_production/stage1.pt",
    )
    assert other["checkpoint_identity_verified"] is False


def test_the_selection_carve_does_not_move_when_the_gate4_carve_is_also_applied():
    """CodeRabbit r3 asked whether `exclude_selection_val` + `exclude_gate4_eval` together
    make the selection bucket drift. They cannot: both cluster sets are computed from the
    SAME unfiltered rows in `load_corpus_records` and then applied as independent membership
    filters, so the seeded draw never sees a shrunken pool. That is the property that keeps a
    twin's inner rung identical to the one the P2-06 sweep selected on — pin it, because a
    future "derive the selection bucket from the non-GATE-4 remainder" refactor would silently
    re-cut the fold the sweep's winner was chosen on."""
    from tbox_finder.data.window_dataset import (
        _training_fold_cluster_sizes,
        selection_val_cluster_ids,
    )

    rows = [{"cluster_id": i // 3, "nested_train": (i % 7) != 0} for i in range(300)]
    full = selection_val_cluster_ids(_training_fold_cluster_sizes(rows))
    # The gate-4 carve removes whole clusters; the selection draw must be computed BEFORE any
    # such removal, i.e. from the same full row set.
    gate4_like = {c for c in {int(r["cluster_id"]) for r in rows} if c % 5 == 0}
    remainder = [r for r in rows if int(r["cluster_id"]) not in gate4_like]
    drifted = selection_val_cluster_ids(_training_fold_cluster_sizes(remainder))

    assert full, "the fixture must actually carve something, or this test is vacuous"
    assert full != drifted, (
        "the fixture is degenerate: drawing from the remainder gave the SAME bucket, so this "
        "test could not tell the two derivations apart"
    )
    # The shipped loader uses `full`. Prove it, from the source, rather than from memory.
    src = (REPO / "src" / "tbox_finder" / "data" / "window_dataset.py").read_text()
    draw = src.index("val_clusters = selection_val_cluster_ids(")
    carve = src.index("gate4_clusters = gate4_eval_cluster_ids(")
    assert draw < carve, "the selection draw must be computed before the gate-4 cluster set"
    assert "_training_fold_cluster_sizes(rows)" in src[draw : draw + 200], (
        "the selection draw must read the unfiltered `rows`; deriving it from a gate-4-carved "
        "remainder would re-cut the fold the P2-06 sweep selected its winner on"
    )
