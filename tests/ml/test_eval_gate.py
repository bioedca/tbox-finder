"""P0-31 — the eval-gate regression harness (CLAUDE.md §8.4; ADR-0004 D6 / ADR-0005 D4/D7/D11).

This gate verifies the **eval harness itself, not model quality** (§8.4): that P2/P3/P4
will compute the PRD §2 gate metrics — per-nt/per-element F1 (GATE-4), boundary IoU,
AUPRC + recall@matched-precision (GATE-1), and the ADR-0005 D11 binned-ECE (GATE-2) —
**correctly**. A wrong metric implementation looks exactly like a real one to a reviewer
(§10.3), so the harness is pinned two ways:

- **Pure-stdlib tiers** (run + BLOCK in the bare CI ``test`` env, no numpy/pandas/sklearn):
  every kernel in :mod:`tbox_finder.metrics` is checked against a **hand-computed** value,
  and each guard is proven to bite — a clean-pass **and** a violating-fail (§8.7). These
  need no scientific stack, so the harness is always exercised.
- **Heavy tier** (``importorskip`` numpy / scikit-learn / pandas — the pinned ``data``
  env, and the CI ``test`` env once P0-31 adds those pins): builds a **real** smoke fixture
  — the 100 real T-box records from ``tests/fixtures/ingest_sample/`` with their derived
  per-nt labels, plus real decoys from ``decoys.build_corpus_pools`` (never mocked, §8.7)
  — runs a **deterministic synthetic smoke "model"** (a seeded stdlib perturbation /
  score generator; it makes **no** performance claim, §10.3), computes every gate metric,
  and asserts each **within tolerance** against the committed
  ``tests/fixtures/eval_gate_sample/expected.json``. It additionally **cross-checks** the
  stdlib ``average_precision`` / ECE binning against ``sklearn.metrics.average_precision_score``
  / a numpy recomputation — proving the stdlib kernels equal the canonical library.

Fail-closed (mirrors ``test_no_leakage.py``): under ``TBOX_REQUIRE_EVAL_GATE=1`` (set in
CI) a missing stack / fixture **fails** the gate instead of silently skipping, so it
cannot rot; locally (env unset) the heavy tier skips gracefully when the stack is absent —
run it under the ``data`` env for the §8.5 manual gate.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from tbox_finder import metrics as M

_REPO = Path(__file__).resolve().parents[2]
_INGEST_CSV = _REPO / "tests" / "fixtures" / "ingest_sample" / "Master_tboxes_sample.csv"
_EXPECTED_JSON = _REPO / "tests" / "fixtures" / "eval_gate_sample" / "expected.json"

#: Smoke-fixture determinism knobs (stdlib RNG → identical across envs; CLAUDE.md §8.3).
_SEED = 42
#: Golden decoy params — the P0-30 golden slice sizes (50 GC-background + 80 dinuc = 130).
_DECOY = dict(seed=42, n_gc=50, n_dinuc_sources=40, dinuc_per_source=2)
#: Absolute tolerance for the committed-expectation check. The kernels are pure-stdlib and
#: the smoke model is a seeded stdlib RNG, so values are reproducible to ~machine epsilon;
#: 1e-9 is generous headroom, not a slack that could hide a wrong metric.
_TOL = 1e-9


# ========================================================================== #
# Pure-stdlib tiers — hand-computed metric checks (run + BLOCK in bare CI).
# Every guard has a clean-pass AND a violating-fail so it is proven to bite.
# ========================================================================== #
def test_average_precision_hand_computed() -> None:
    # y=[1,0,1], desc scores -> AP = 0.5*1 + 0*0.5 + 0.5*(2/3) = 0.8333...
    assert M.average_precision([1, 0, 1], [0.9, 0.8, 0.7]) == pytest.approx(0.83333333, abs=1e-7)
    # perfect ranking -> AP == 1.0
    assert M.average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    # all negatives -> undefined
    assert M.average_precision([0, 0, 0], [0.9, 0.5, 0.1]) != M.average_precision(
        [0, 0, 0], [0.9, 0.5, 0.1]
    )  # NaN


def test_per_nt_class_f1_hand_computed() -> None:
    # class 1: true at {0,1}, pred at {0} -> tp=1, fp=0, fn=1 -> F1 = 2/3
    assert M.per_nt_class_f1([1, 1, 0, 0], [1, 0, 0, 0], 1) == pytest.approx(2 / 3)
    # perfect -> 1.0; absent-from-both -> NaN
    assert M.per_nt_class_f1([1, 0], [1, 0], 1) == pytest.approx(1.0)
    val = M.per_nt_class_f1([0, 0], [0, 0], 1)
    assert val != val  # NaN (class 1 absent from truth and prediction)


def test_gate4_uses_min_over_core_not_mean() -> None:
    # Stem_I(1) & Specifier(2) perfect (F1=1.0); Antiterminator(5) tp=3,fn=3,fp=0 -> F1=6/9=0.667.
    y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] + [5, 5, 5, 5, 5, 5] + [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] + [5, 5, 5, 0, 0, 0] + [0, 0, 0, 0]
    res = M.gate4_core_min_f1(y_true, y_pred)
    f1s = res["per_element_f1"]
    assert f1s["Stem_I"] == pytest.approx(1.0)
    assert f1s["Specifier"] == pytest.approx(1.0)
    assert f1s["Antiterminator_Tbox_seq"] == pytest.approx(6 / 9)
    # The gate is the MIN (ADR-0004 D6), which fails the 0.80 floor ...
    assert res["min_f1"] == pytest.approx(6 / 9)
    assert res["passes"] is False
    # ... while a (forbidden) cross-unit MEAN would be 0.889 and wrongly pass — the guard
    # proves the min/mean distinction is load-bearing.
    mean_f1 = sum(f1s.values()) / 3
    assert mean_f1 >= M.GATE4_F1_FLOOR
    # an undefined core element cannot silently certify the gate
    res_nan = M.gate4_core_min_f1([1, 1], [1, 1])  # Specifier/Antiterm absent -> NaN min
    assert res_nan["passes"] is False


def test_boundary_iou_hand_computed() -> None:
    # class 1: true {0,1}, pred {0} -> inter 1, union 2 -> 0.5
    assert M.element_extent_iou([1, 1, 0, 0], [1, 0, 0, 0], 1) == pytest.approx(0.5)
    # perfect overlap -> 1.0; absent from both -> NaN
    assert M.element_extent_iou([1, 0], [1, 0], 1) == pytest.approx(1.0)
    v = M.element_extent_iou([0, 0], [0, 0], 1)
    assert v != v  # NaN


def test_recall_at_matched_precision() -> None:
    y = [1, 1, 0, 0]
    s = [0.9, 0.8, 0.4, 0.3]
    # at precision 1.0 the model calls both positives before any decoy -> recall 1.0
    r = M.recall_at_matched_precision(y, s, 1.0)
    assert r["matched"] is True and r["recall"] == pytest.approx(1.0)
    # a target precision the model can never reach on this pool -> unmatched, recall 0
    y2 = [1, 0, 0, 0]
    s2 = [0.5, 0.9, 0.8, 0.7]  # every high score is a decoy -> max precision here is low
    r2 = M.recall_at_matched_precision(y2, s2, 1.0)
    assert r2["matched"] is False and r2["recall"] == 0.0


def test_binned_ece_plugin_and_debias() -> None:
    # perfectly-calibrated 2-bin case -> plug-in ECE 0
    assert M.binned_ece(
        [0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], n_bins=2, debias=False
    ) == pytest.approx(0.0)
    # a miscalibrated case: one bin conf 0.9 but acc 0.5 -> plug-in gap 0.4 (weight 1)
    plug = M.binned_ece([1, 0], [0.9, 0.9], n_bins=1, debias=False)
    assert plug == pytest.approx(0.4)
    # debiasing never increases ECE and never drives it below 0 (a monotone correction)
    deb = M.binned_ece([1, 0], [0.9, 0.9], n_bins=1, debias=True)
    assert 0.0 <= deb <= plug


def test_gate_predicates_both_directions() -> None:
    # GATE-4
    assert M.gate4_pass(0.81) is True
    assert M.gate4_pass(0.79) is False
    assert M.gate4_pass(float("nan")) is False
    # GATE-2 ECE
    assert M.gate2_ece_pass(0.049) is True
    assert M.gate2_ece_pass(0.051) is False
    assert M.gate2_ece_pass(float("nan")) is False
    # GATE-1 two-part bar (strong) + min-N fallback (weak)
    assert M.gate1_recall_bar(12.0, 6.0) is True  # point>=10 AND ci>5
    assert M.gate1_recall_bar(9.0, 6.0) is False  # point<10
    assert M.gate1_recall_bar(12.0, 4.0) is False  # ci<=5 (strong)
    assert M.gate1_recall_bar(12.0, 4.0, min_n_fallback=True) is True  # weak: ci>0
    assert M.gate1_recall_bar(12.0, 0.0, min_n_fallback=True) is False  # weak needs ci>0
    # fails CLOSED on an undefined estimate: a NaN point with a valid CI must NOT pass
    # (NaN < 10 is False, so an unguarded bar would fall through to the CI clause) — an
    # unmeasurable arm can never certify GATE-1.
    assert M.gate1_recall_bar(float("nan"), 6.0) is False
    assert M.gate1_recall_bar(12.0, float("nan")) is False
    assert M.gate1_recall_bar(float("nan"), float("nan")) is False


def test_pins_are_single_sourced_no_drift() -> None:
    # metrics.py must never re-declare a gate number; it imports the power.py pins.
    from tbox_finder import power

    assert M.GATE4_F1_FLOOR == power.GATE4_F1_FLOOR == 0.80  # ADR-0004 D6
    assert M.ECE_GATE == power.ECE_GATE == 0.05  # ADR-0005 D11
    assert M.RECALL_POINT_BAR_PP == power.RECALL_POINT_BAR_PP == 10  # ADR-0005 D4
    assert M.RECALL_CI_FLOOR_PP == power.RECALL_CI_FLOOR_PP == 5  # ADR-0005 D4
    assert M.ECE_N_BINS == 15  # ADR-0005 D11 (15 equal-mass bins)
    from tbox_finder import labels

    assert tuple(labels.CLASS_INDEX[e] for e in labels.CORE_ELEMENTS) == M.CORE_ELEMENT_INDICES


def test_macro_average_and_block_bootstrap() -> None:
    # macro-average is equal-weight over orders and drops undefined (NaN) orders
    assert M.macro_average([0.6, 0.8, 1.0]) == pytest.approx(0.8)
    assert M.macro_average([0.6, float("nan"), 1.0]) == pytest.approx(0.8)
    # block bootstrap is seeded → reproducible, and needs ≥2 blocks (ADR-0005 D5 / A1)
    blocks = [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    stat = lambda xs: sum(xs) / len(xs)  # noqa: E731
    a = M.block_bootstrap_ci(blocks, stat, n_boot=500, seed=7)
    b = M.block_bootstrap_ci(blocks, stat, n_boot=500, seed=7)
    assert a == b  # determinism
    assert a["lower"] <= a["point"] <= a["upper"]
    one = M.block_bootstrap_ci([[1, 0]], stat)  # <2 blocks -> not resamplable
    assert one["n_boot"] == 0 and one["lower"] != one["lower"]  # NaN CI


# -------------------------------------------------------------------------- #
# P3-09 — the second, OOD calibration estimator (ADR-0005 D13 + Amendment A2).
#
# The harness now computes TWO calibration errors and they are graded differently:
# D11's binned ECE is the GATE-2 quantity, D13's OOD ECE is reported and never gated.
# Wiring them the wrong way round — grading the OOD number, or reporting an
# inadmissible one — is exactly the harness failure this file exists to catch (§8.4),
# so the distinction is pinned here and not only in tests/unit/test_ece.py.
#
# Deliberately NOT added to the heavy tier's committed expectation: the smoke fixture
# is 100 real positives plus corpus-derived decoys and carries **no homology-cluster
# structure** (clustering happens downstream, in the splits pipeline). `ood_ece`
# requires block labels, and inventing a blocking for the fixture would fabricate the
# very structure the estimator exists to respect (§10.2/§10.3). Its committed golden
# belongs to the step that has real clusters to give it (P3-10).
# -------------------------------------------------------------------------- #
def test_ood_ece_is_a_second_estimator_wired_to_the_ungated_path() -> None:
    from tbox_finder import coverage
    from tbox_finder.calib import ece

    # Hand-computed, exactly as for every other kernel in this file. With a constant
    # posterior all kernel weights are equal, so g_hat_{-i} = (S - y_i)/(n-1) and the mean
    # gap collapses to S/n - p = 0.8 - 0.5 = 0.30, for ANY bandwidth.
    y, p = [1] * 80 + [0] * 20, [0.5] * 100
    blocks = [f"c{i % 5}" for i in range(100)]
    out = ece.ood_ece(y, p, blocks, n_boot=25)
    assert out["ood_ece"] == pytest.approx(0.30, abs=1e-12)

    # The two estimators are distinct functions and disagree on this input, so a harness
    # that silently swapped one for the other would be caught.
    assert out["ood_ece"] != pytest.approx(M.binned_ece(y, p), abs=1e-6)
    assert ece.binned_ece is M.binned_ece  # D11's estimator is re-exported, not re-written

    # The OOD number is reported, never gated: it carries no pass/fail and the GATE-2
    # predicate is not applied to it anywhere.
    assert out["gated"] is False
    assert "gate_pass" not in out and "ece_gate" not in out

    # The A2 admissibility floor is enforced on the way out, on POSITIVES not rows.
    assert out["min_n"] == coverage.OOD_ECE_MIN_N == 20
    thin = ece.ood_ece([1] * 19 + [0] * 200, [0.5] * 219, [f"c{i % 5}" for i in range(219)])
    assert thin["admissible"] is False and thin["ood_ece"] is None and thin["ci"] is None


def test_gate2_predicate_still_grades_only_the_binned_in_distribution_ece() -> None:
    """GATE-2's threshold belongs to D11's estimator. Guard against the harness ever
    routing the (larger, small-N-biased) OOD number through the 0.05 bar."""
    from tbox_finder.calib import ece

    assert not hasattr(ece, "gate2_ood_pass")
    assert "ECE_GATE" not in ece.__all__
    # The gate predicate is unchanged and still keyed to the D11 pin.
    assert M.gate2_ece_pass(M.ECE_GATE - 1e-9) is True
    assert M.gate2_ece_pass(M.ECE_GATE + 1e-9) is False


# ========================================================================== #
# Heavy tier — real smoke fixture + committed expectation + library cross-check.
# ========================================================================== #
def _fail_or_skip(reason: str) -> None:
    """Under ``TBOX_REQUIRE_EVAL_GATE=1`` (CI) an unrunnable heavy tier FAILS the gate so
    it cannot rot; locally it skips gracefully."""
    if os.environ.get("TBOX_REQUIRE_EVAL_GATE") == "1":
        pytest.fail(
            f"TBOX_REQUIRE_EVAL_GATE=1 but the eval-gate heavy tier is unrunnable: {reason}"
        )
    pytest.skip(reason)


def _require_stack():
    """Import the pinned scientific stack or fail-closed/skip. Returns the modules."""
    try:
        import numpy as np  # noqa: F401
        import pandas  # noqa: F401
        from sklearn.metrics import average_precision_score  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when the stack is absent
        _fail_or_skip(f"scientific stack not importable ({exc})")
    if not _INGEST_CSV.is_file():
        _fail_or_skip("ingest_sample fixture CSV absent")


def _build_smoke():
    """Build the REAL smoke fixture + the DETERMINISTIC synthetic smoke model.

    Real inputs (never mocked, §8.7): the 100-record ingest slice → derived 8-class per-nt
    labels (segmentation truth), and ``decoys.build_corpus_pools`` decoys (detection
    negatives). The "model" is a seeded **stdlib** perturbation / score generator — it
    makes no performance claim; it only exercises the metric math (§8.4/§10.3).
    """
    from tbox_finder import decoys, ingest, labels

    raw = ingest.read_raw(_INGEST_CSV)
    ingest.assert_parse_gate(raw, expected_records=100, expected_raw_cols=107)
    clean = ingest.clean(raw, expect_records=100, expect_named_cols=None)
    _, _, prod_codes, _ = labels.derive_labels(clean)

    # --- segmentation task: concat real per-nt labels; perturb deterministically ---
    # `seg_lengths` keeps the per-RECORD boundaries the pooled vectors erase. GATE-4's
    # exact-3-nt-codon check (ADR-0004 D6) is a per-locus question — "did the model call the
    # right three nucleotides in THIS record" — and pooling would let one record's Specifier
    # hit pay for another's miss (P2-14).
    seg_true: list[int] = []
    seg_lengths: list[int] = []
    for code in prod_codes:
        idx = labels.label_string_to_indices(code)
        seg_true.extend(idx)
        seg_lengths.append(len(idx))
    rng = random.Random(_SEED)
    seg_pred: list[int] = []
    for t in seg_true:
        u = rng.random()
        if u < 0.08:
            seg_pred.append(0)  # drop to background -> per-element FN
        elif u < 0.13:
            seg_pred.append((t % 7) + 1)  # shift to a different non-background class -> FP
        else:
            seg_pred.append(t)

    # --- detection task: 100 positives vs real decoys; seeded overlapping scores ---
    # `record_ids` links each dinucleotide decoy to the positive it was shuffled from. It
    # does NOT enter the golden digest (which covers `(pool, decoy_id, sequence)`) and is
    # read, never sampled, so supplying it cannot perturb any committed value above — it is
    # what lets the P3-10 OOD block below be keyed on a REAL taxon instead of a row index.
    names = clean["Name"].astype(str).tolist()
    decoy_records = decoys.build_corpus_pools(
        clean[decoys.GC_COL].astype(float).tolist(),
        clean[decoys.LEN_COL].astype(int).tolist(),
        clean[decoys.SEQ_COL].astype(str).tolist(),
        record_ids=names,
        **_DECOY,
    )
    n_pos = len(prod_codes)
    n_dec = len(decoy_records)
    det_labels = [1] * n_pos + [0] * n_dec
    srng = random.Random(_SEED + 1)
    det_scores: list[float] = []
    for y in det_labels:
        mu = 0.75 if y == 1 else 0.25
        det_scores.append(min(0.999, max(0.001, srng.gauss(mu, 0.15))))
    # synthetic "cmsearch" baseline: independent seeded hit mask (a fixed operating point)
    brng = random.Random(_SEED + 2)
    baseline_hit = [1 if (brng.random() < (0.80 if y == 1 else 0.05)) else 0 for y in det_labels]
    det_units = _detection_block_units(clean, names, decoy_records)
    return (
        seg_true,
        seg_pred,
        seg_lengths,
        det_labels,
        det_scores,
        baseline_hit,
        n_pos,
        n_dec,
        det_units,
    )


def _detection_block_units(clean, names, decoy_records):
    """One block label per detection row: its REAL taxonomic order, or ``None``.

    P3-09 left the OOD-ECE golden to P3-10 because the smoke fixture "carries no homology-
    cluster structure", and inventing one would fabricate the very structure the estimator
    exists to respect (§10.2/§10.3). It does, however, carry each record's real ``order``
    — which is exactly the granularity of ADR-0004 D5's leave-one-order-out unit, and one
    of ``resample.BLOCK_GRANULARITY_COLUMNS``. So:

    * a **positive** blocks on its own order;
    * a **dinucleotide-shuffled decoy** blocks on the order of the record it was shuffled
      from — real provenance, carried by ``source_record_id``, and the same construction the
      real leave-clade-out units have (their negatives are dinuc decoys carved from the same
      held-out records);
    * a **GC-background decoy** has no source record at all, so it gets ``None`` and is
      excluded downstream. Assigning it an order would be the fabrication this avoids.

    Records whose ``order`` is null in the source catalogue (3 of 100) are ``None`` too.
    """
    order_by_name: dict[str, str] = {}
    for name, order in zip(names, clean["order"].tolist(), strict=True):
        if order is None or order != order or str(order).strip() == "":  # NaN-safe
            continue
        previous = order_by_name.setdefault(str(name), str(order))
        assert previous == str(order), (
            f"record name {name!r} maps to two orders ({previous!r}, {order!r}); the "
            "name→order map would not be single-valued and the block key would be ambiguous"
        )
    units: list[str | None] = [order_by_name.get(str(n)) for n in names]
    for record in decoy_records:
        parent = record.get("source_record_id")
        units.append(order_by_name.get(str(parent)) if parent is not None else None)
    return units


def _per_record_pairs(seg_true, seg_pred, seg_lengths):
    """Slice the pooled segmentation vectors back into per-record ``(true, pred)`` pairs."""
    pairs, start = [], 0
    for n in seg_lengths:
        pairs.append((seg_true[start : start + n], seg_pred[start : start + n]))
        start += n
    assert start == len(seg_true), "record lengths do not tile the pooled vector"
    return pairs


def _compute_smoke_metrics() -> dict:
    """Compute every PRD §2 gate metric on the smoke fixture (the committed-expectation
    payload). Deterministic — the source of ``expected.json``."""
    (
        seg_true,
        seg_pred,
        seg_lengths,
        det_labels,
        det_scores,
        baseline_hit,
        n_pos,
        n_dec,
        det_units,
    ) = _build_smoke()
    gate4 = M.gate4_core_min_f1(seg_true, seg_pred)
    base_p, base_r = M.baseline_operating_point(det_labels, baseline_hit)
    matched = M.recall_at_matched_precision(det_labels, det_scores, base_p)
    codon = M.specifier_exact_codon_detection(_per_record_pairs(seg_true, seg_pred, seg_lengths))
    return {
        # GATE-4's reported (non-gated) exact-codon sanity check, on the same fixture and
        # the same synthetic perturbation as the gated statistic above (P2-14).
        "specifier_exact_codon": {
            "n_scorable": codon["n_scorable"],
            "n_exact": codon["n_exact"],
            "n_right_length": codon["n_right_length"],
            "n_overlapping": codon["n_overlapping"],
            "exact_rate": codon["exact_rate"],
        },
        "n_positives": n_pos,
        "n_decoys": n_dec,
        "n_seg_positions": len(seg_true),
        "gate4_core_min_f1": gate4["min_f1"],
        "gate4_per_element_f1": gate4["per_element_f1"],
        "boundary_iou_core": {
            e: M.element_extent_iou(seg_true, seg_pred, M.CLASS_INDEX[e])
            for e in ("Stem_I", "Specifier", "Antiterminator_Tbox_seq")
        },
        "auprc": M.average_precision(det_labels, det_scores),
        "ece_plugin": M.binned_ece(det_labels, det_scores, debias=False),
        "ece_debiased": M.binned_ece(det_labels, det_scores, debias=True),
        "baseline_precision": base_p,
        "baseline_recall": base_r,
        "recall_at_matched_precision": matched["recall"],
        "recall_gap_pp": M.recall_gap_pp(matched["recall"], base_r),
        # GATE-2's OTHER estimator (ADR-0005 D13 / Amendment A2) — reported, never gated.
        # P3-09 built it and deferred this golden to P3-10 (imp.md:1477).
        "ood_ece": _smoke_ood_ece(det_labels, det_scores, det_units),
    }


#: The OOD golden's bootstrap size. Small on purpose: the leave-one-out beta kernel is
#: O(n^2) per replicate in pure Python, and this tier has a CI wall-clock budget (§8.6).
#: It is a *golden*, so the number only has to be reproducible, not tight.
_OOD_N_BOOT = 60
_OOD_SEED = 42


def _smoke_ood_ece(det_labels, det_scores, det_units):
    """The D13 estimator over the fixture rows that have a real taxonomic block.

    Excludes the 50 GC-background decoys and the 3 records with no ``order`` — they have no
    block, and ``resample.blocks_by_key`` refuses a missing label rather than collapsing
    them into one giant pseudo-block. The exclusion count is part of the golden, so silently
    dropping more rows changes a committed number instead of passing quietly.
    """
    from tbox_finder.calib import ece as ECE

    # The three vectors are indexed by ONE shared index below, and they are built from
    # different sources — `det_labels` is sized from `len(prod_codes)`, `det_units` from
    # `len(names)` plus the decoys. If those ever diverged, every unit after the gap would
    # attach to the wrong label and `ood_ece`'s own length check could not see it (`y` and
    # `block_labels` are both derived from `keep`, so they always agree with each other).
    if not (len(det_labels) == len(det_scores) == len(det_units)):
        raise AssertionError(
            "detection vectors are misaligned: labels="
            f"{len(det_labels)}, scores={len(det_scores)}, units={len(det_units)}"
        )
    keep = [i for i, unit in enumerate(det_units) if unit is not None]
    out = ECE.ood_ece(
        [det_labels[i] for i in keep],
        [det_scores[i] for i in keep],
        [det_units[i] for i in keep],
        block_key="loo_order_unit",
        n_boot=_OOD_N_BOOT,
        seed=_OOD_SEED,
    )
    return {
        "n_records": out["n_records"],
        "n_positives": out["n_positives"],
        "n_blocks": out["n_blocks"],
        "n_excluded_no_block": len(det_units) - len(keep),
        "admissible": out["admissible"],
        "min_n": out["min_n"],
        "ood_ece": out["ood_ece"],
        "bandwidth": out["bandwidth"],
        "bandwidth_loo_log_likelihood": out["bandwidth_loo_log_likelihood"],
        "brier": out["brier_decomposition"]["brier"],
        "brier_calibration": out["brier_decomposition"]["calibration"],
        "brier_refinement": out["brier_decomposition"]["refinement"],
        "ci_lower": out["ci"]["lower"],
        "ci_upper": out["ci"]["upper"],
        "gated": out["gated"],
    }


def test_smoke_fixture_is_real_not_mocked() -> None:
    _require_stack()
    m = _compute_smoke_metrics()
    # real 100-record slice + real decoys (§8.7): sizes are the committed shapes
    assert m["n_positives"] == 100
    assert m["n_decoys"] == _DECOY["n_gc"] + _DECOY["n_dinuc_sources"] * _DECOY["dinuc_per_source"]
    assert m["n_seg_positions"] > 100  # real per-nt vectors, not a stub


def test_eval_gate_matches_committed_expectation() -> None:
    _require_stack()
    if not _EXPECTED_JSON.is_file():
        _fail_or_skip("committed expected.json absent")
    expected = json.loads(_EXPECTED_JSON.read_text())
    got = _compute_smoke_metrics()

    def _close(a, b):
        assert a == pytest.approx(b, abs=_TOL), f"{a} != {b}"

    # Full key-set parity first — otherwise a truncated expected.json would silently pass
    # (a per-key loop only checks the keys that remain). Both directions, so neither a
    # missing nor an extra key slips through (§8.7 — never let a weakened fixture pass).
    assert set(got) == set(expected)
    assert set(got["gate4_per_element_f1"]) == set(expected["gate4_per_element_f1"])
    assert set(got["boundary_iou_core"]) == set(expected["boundary_iou_core"])

    for key in (
        "gate4_core_min_f1",
        "auprc",
        "ece_plugin",
        "ece_debiased",
        "baseline_precision",
        "baseline_recall",
        "recall_at_matched_precision",
        "recall_gap_pp",
    ):
        _close(got[key], expected[key])
    for e, v in expected["gate4_per_element_f1"].items():
        _close(got["gate4_per_element_f1"][e], v)
    for e, v in expected["boundary_iou_core"].items():
        _close(got["boundary_iou_core"][e], v)
    assert got["n_positives"] == expected["n_positives"]
    assert got["n_decoys"] == expected["n_decoys"]
    assert got["n_seg_positions"] == expected["n_seg_positions"]

    # GATE-2's OOD half (ADR-0005 D13 / A2) — the golden P3-09 deferred to P3-10.
    ood_got, ood_exp = got["ood_ece"], expected["ood_ece"]
    assert set(ood_got) == set(ood_exp)
    for key in ("n_records", "n_positives", "n_blocks", "n_excluded_no_block", "min_n"):
        assert ood_got[key] == ood_exp[key], key
    assert ood_got["admissible"] is ood_exp["admissible"]
    assert ood_got["gated"] is False is ood_exp["gated"]
    for key in (
        "ood_ece",
        "bandwidth",
        "bandwidth_loo_log_likelihood",
        "brier",
        "brier_calibration",
        "brier_refinement",
        "ci_lower",
        "ci_upper",
    ):
        _close(ood_got[key], ood_exp[key])

    # GATE-4's non-gated exact-codon check (P2-14). Counts are exact; the rate is float.
    codon_got, codon_exp = got["specifier_exact_codon"], expected["specifier_exact_codon"]
    assert set(codon_got) == set(codon_exp)
    for key in ("n_scorable", "n_exact", "n_right_length", "n_overlapping"):
        assert codon_got[key] == codon_exp[key], key
    _close(codon_got["exact_rate"], codon_exp["exact_rate"])
    # The check must DISCRIMINATE on this fixture, not merely run: the synthetic model
    # scatters Specifier calls across each record, so it overlaps the true codon almost
    # always and reproduces it exactly almost never. A harness where those two numbers
    # coincided would be measuring nothing (§8.4).
    assert codon_got["n_overlapping"] > codon_got["n_exact"]
    assert 0 < codon_got["n_scorable"] <= expected["n_positives"]


def test_the_ood_golden_is_blocked_by_a_real_taxon_and_not_by_record() -> None:
    """The OOD golden must DISCRIMINATE, not merely run (P3-10).

    Four things it would still produce a number without, and therefore has to be pinned on:
    the blocking is a taxon and not a row; the block key is one ``resample`` accepts; the
    unblockable rows were excluded rather than folded into one pseudo-block; and the D11
    gate predicate is not applied to it.
    """
    _require_stack()
    from tbox_finder.calib import ece as ECE
    from tbox_finder.eval import resample as RS

    got = _compute_smoke_metrics()["ood_ece"]

    # 1. Block granularity is real: 13 orders standing for 177 rows. A record-level key
    #    would make these equal, which is exactly the bootstrap PRD §12 forbids.
    assert got["n_blocks"] < got["n_records"] // 5, (
        f"{got['n_blocks']} blocks over {got['n_records']} rows is not a taxon-level "
        "grouping — a record-keyed bootstrap gives an interval far narrower than the data"
    )
    assert "loo_order_unit" in RS.BLOCK_GRANULARITY_COLUMNS
    for column in RS.RECORD_LEVEL_COLUMNS:
        with pytest.raises(ValueError, match="identifies a record, not a block"):
            RS.blocks_by_key([1, 2], ["a", "b"], key_name=column)

    # 2. The unblockable rows were dropped, not defaulted. 50 GC-background decoys have no
    #    source record at all; 3 of the 100 catalogue records carry no order.
    #    ⚠ 53 = 50 + 3 holds only while none of the 40 sampled dinucleotide source records
    #    is one of those 3 null-order records — a property of the decoy SEED, not of the
    #    construction. The decomposition is asserted explicitly so a seed change fails here
    #    with a readable cause instead of silently shifting a committed golden.
    assert got["n_excluded_no_block"] == 53
    assert got["n_positives"] == 100 - 3, "3 catalogue records carry no order"
    n_dinuc = _DECOY["n_dinuc_sources"] * _DECOY["dinuc_per_source"]
    assert got["n_records"] - got["n_positives"] == n_dinuc, (
        "every dinucleotide decoy inherited an order; if a sampled source record had a null "
        "order this would drop below 80 and the 53 above would no longer decompose as 50 + 3"
    )
    assert got["n_records"] + got["n_excluded_no_block"] == 100 + _DECOY["n_gc"] + (
        _DECOY["n_dinuc_sources"] * _DECOY["dinuc_per_source"]
    )
    assert got["n_positives"] == 97

    # 3. It clears the A2 floor, so the number is admissible rather than withheld — and the
    #    floor it cleared is the pinned one.
    assert got["min_n"] == 20 and got["admissible"] is True

    # 4. GATE-2's predicate belongs to the D11 in-distribution estimator ONLY. This OOD
    #    value would FAIL it, and nothing may apply it — D13 reports, never gates.
    assert got["gated"] is False
    assert M.gate2_ece_pass(got["ood_ece"]) is False, (
        "the OOD value happens to sit above 0.05; if a future fixture moved it below, this "
        "assertion would stop demonstrating that the two estimators are not interchangeable"
    )
    assert not hasattr(ECE, "gate2_ood_pass")


def test_stdlib_kernels_match_sklearn_and_numpy() -> None:
    """Prove the stdlib AUPRC / ECE binning equal the canonical scikit-learn / numpy
    reference on the real smoke detection set — the anti-fabrication guard for the two
    kernels that would otherwise be hard to hand-verify at scale (§10.3)."""
    _require_stack()
    import numpy as np
    from sklearn.metrics import average_precision_score

    _, _, _, det_labels, det_scores, *_ = _build_smoke()

    mine_ap = M.average_precision(det_labels, det_scores)
    ref_ap = float(average_precision_score(det_labels, det_scores))
    assert mine_ap == pytest.approx(ref_ap, abs=1e-12)

    # plug-in ECE must equal a numpy equal-mass (array_split) recomputation
    y = np.asarray(det_labels)
    p = np.asarray(det_scores)
    order = np.argsort(p, kind="stable")
    ref_ece = 0.0
    n = len(p)
    for b in np.array_split(order, M.ECE_N_BINS):
        if len(b) == 0:
            continue
        ref_ece += (len(b) / n) * abs(y[b].mean() - p[b].mean())
    assert M.binned_ece(det_labels, det_scores, debias=False) == pytest.approx(ref_ece, abs=1e-12)


def test_fixture_present() -> None:
    # stdlib-only guard so the committed fixture/expectation cannot silently disappear
    assert _INGEST_CSV.is_file()
    assert _EXPECTED_JSON.is_file()
