"""P1-08 — ML-gate regression for the Stage-1 smoke **reproducibility** gate (ADR-0002 A7).

The Phase-1 exit condition *"smoke run reproducible"* (PRD §18.1): a seeded re-run of the
P1-07 binding transfer go/no-go (ADR-0002 A6) reproduces its go/no-go **verdict** and its
per-nt F1 metrics **within the ADR-0002 A7 tolerance** (max|Δ| ≤ 1e-3). The re-run is not
bit-exact (the Mamba ``selective_scan_cuda`` kernel has no deterministic variant, A2 C2), so
the gate is metric-level; τ is pinned a priori in ADR-0002 A7 and **not weakenable on a
fail** (§8.5/§10.3).

Two tiers, mirroring ``tests/ml/test_eval_gate.py`` / ``test_transfer_gonogo.py``:

- **Tier 1 (pure stdlib, runs + BLOCKS in bare CI):** the reproducibility **adjudication**
  (:mod:`tbox_finder.train.repro`) — verdict comparison, per-metric abs-diff, config/backbone
  matching, the τ predicate, and the pinned-constant drift guard. Every honesty- and
  gate-critical guard has a clean-pass **and** a violating-fail so it is proven to bite (§8.7).
  These import only stdlib + the bare ``train.repro`` module.
- **Tier 2 (committed-report gate; pure JSON, runs in CI once BOTH reports are committed):**
  the actual P1-08 exit-gate assertion over the committed P1-07 report
  (``reports/p1/seg_smoke_gonogo.json``) vs the P1-08 re-run report
  (``reports/p1/seg_smoke_repro.json``). Fail-closed under ``TBOX_REQUIRE_REPRO_GATE=1`` (set
  in CI once the re-run report lands); skips gracefully while the re-run report is absent (the
  SLURM re-run has not produced it yet). No torch/GPU — a plain JSON comparison.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from tbox_finder.train import repro as R

_REPO = Path(__file__).resolve().parents[2]
_REF_REPORT = _REPO / "reports" / "p1" / "seg_smoke_gonogo.json"  # P1-07 reference (A6)
_RERUN_REPORT = _REPO / "reports" / "p1" / "seg_smoke_repro.json"  # P1-08 re-run

_REVISION = "d89eeb853136ea64da7feb3d0c8e909771b17ae6"  # pinned Caduceus-PS commit (backbone)


def _base_report() -> dict:
    """A fresh synthetic go/no-go report shaped like ``seg_smoke_gonogo.json`` (verdict GO)."""
    return copy.deepcopy(
        {
            "backbone": {"revision": _REVISION},
            "config": {
                "seed": 42,
                "epochs": 20,
                "lr": 3.0e-4,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
                "gamma": 2.0,
                "rc_combine": "concat",
                "use_crf": False,
                "dropout": 0.0,
                "batch_size": 1,
            },
            "metrics": {
                "macro_f1": 0.9995,
                "micro_f1": 0.9995,
                "min_core_f1": 0.9998,
                "per_element_f1": {
                    "Stem_I": 0.9998,
                    "Specifier": 1.0,
                    "Antiterminator_Tbox_seq": 1.0,
                },
                "per_class_f1": {
                    "background": 0.9995,
                    "Stem_I": 0.9998,
                    "Stem_II": 0.9987,
                    "Stem_III": 0.9982,
                    "Specifier": 1.0,
                    "Antiterminator_Tbox_seq": 1.0,
                    "Discriminator": 1.0,
                    "Terminator": 1.0,
                },
            },
            "gate": {"verdict": "GO"},
        }
    )


# ======================================================================================
# Tier 1 — pure stdlib (runs + blocks in bare CI)
# ======================================================================================
def test_tolerance_constant_is_pinned() -> None:
    # Drift guard: the ADR-0002 A7 tolerance is single-sourced here — a silent change breaks
    # this (the value is not weakenable on a fail, §8.5/§10.3).
    assert R.REPRO_TOLERANCE == 1e-3


def test_flatten_metrics_expands_nested_and_scalars() -> None:
    flat = R.flatten_metrics(_base_report())
    # 3 scalars + 3 per_element + 8 per_class = 14 dotted keys.
    assert flat["min_core_f1"] == 0.9998
    assert flat["macro_f1"] == 0.9995
    assert flat["per_element_f1.Stem_I"] == 0.9998
    assert flat["per_class_f1.Discriminator"] == 1.0
    assert len(flat) == 14
    # A missing/invalid metrics block flattens to {} (→ downstream fails closed).
    assert R.flatten_metrics({}) == {}
    assert R.flatten_metrics({"metrics": None}) == {}


def test_verdict_of_extracts_or_none() -> None:
    assert R.verdict_of(_base_report()) == "GO"
    assert R.verdict_of({}) is None
    assert R.verdict_of({"gate": None}) is None
    assert R.verdict_of({"gate": {}}) is None


def test_metric_abs_diffs_identical_is_zero() -> None:
    diffs = R.metric_abs_diffs(_base_report(), _base_report())
    assert len(diffs) == 14
    assert all(d == 0.0 for d in diffs.values())


def test_metric_abs_diffs_missing_key_and_nonfinite_are_none() -> None:
    ref = _base_report()
    rer = _base_report()
    # A truncated re-run (a class dropped) → that key is uncomparable (None), not ignored.
    del rer["metrics"]["per_class_f1"]["Terminator"]
    diffs = R.metric_abs_diffs(ref, rer)
    assert diffs["per_class_f1.Terminator"] is None
    # A non-finite (None from a sanitised NaN) value → uncomparable.
    rer2 = _base_report()
    rer2["metrics"]["min_core_f1"] = None
    assert R.metric_abs_diffs(ref, rer2)["min_core_f1"] is None


def test_metric_abs_diffs_extra_rerun_key_bites() -> None:
    # The `| set(frer)` union half: a metric present ONLY in the re-run (an extra/renamed key)
    # is surfaced as uncomparable, not silently dropped — so structural drift fails closed.
    ref = _base_report()
    rer = _base_report()
    rer["metrics"]["per_class_f1"]["Extra"] = 0.5  # a key the reference lacks
    assert R.metric_abs_diffs(ref, rer)["per_class_f1.Extra"] is None
    assert R.check_reproducibility(ref, rer)["all_metrics_comparable"] is False
    assert R.check_reproducibility(ref, rer)["reproducible"] is False


def test_config_mismatches_clean_and_biting() -> None:
    assert R.config_mismatches(_base_report(), _base_report()) == []
    # A differing seed voids the reproducibility test.
    rer = _base_report()
    rer["config"]["seed"] = 7
    assert "config.seed" in R.config_mismatches(_base_report(), rer)
    # A missing config key is a mismatch (fail-closed), not a pass.
    rer2 = _base_report()
    del rer2["config"]["gamma"]
    assert "config.gamma" in R.config_mismatches(_base_report(), rer2)
    # A differing backbone revision voids it.
    rer3 = _base_report()
    rer3["backbone"]["revision"] = "deadbeef"
    assert "backbone.revision" in R.config_mismatches(_base_report(), rer3)
    # An ABSENT backbone block on BOTH sides must NOT read as "identical" (None == None would
    # fail open) — a re-run with no backbone provenance is not a reproducibility test.
    ref_nb, rer_nb = _base_report(), _base_report()
    del ref_nb["backbone"]
    del rer_nb["backbone"]
    assert "backbone.revision" in R.config_mismatches(ref_nb, rer_nb)


def test_check_identical_is_reproducible() -> None:
    res = R.check_reproducibility(_base_report(), _base_report())
    assert res["reproducible"] is True
    assert res["verdict_reproduces"] is True
    assert res["config_ok"] is True
    assert res["all_metrics_comparable"] is True
    assert res["max_abs_diff"] == 0.0
    assert res["within_tolerance"] is True


def test_check_within_tolerance_passes() -> None:
    rer = _base_report()
    # A sub-τ argmax-flip-sized perturbation on the smallest element still reproduces.
    rer["metrics"]["min_core_f1"] = 0.9998 - 5.0e-4
    rer["metrics"]["per_element_f1"]["Stem_I"] = 0.9998 - 5.0e-4
    res = R.check_reproducibility(rer, _base_report())
    assert res["reproducible"] is True
    assert res["within_tolerance"] is True
    assert res["max_abs_diff"] == pytest.approx(5.0e-4)


def test_check_tolerance_boundary_inclusive() -> None:
    # |Δ| exactly == τ passes (inclusive); just above τ fails. Constructed as |τ − 0.0| and
    # |(τ+ε) − 0.0| so the diff is exactly representable (a `0.9995 − τ` style diff rounds to
    # slightly above τ in float64 and would spuriously trip the strict `<= τ`).
    at, at_ref = _base_report(), _base_report()
    at["metrics"]["macro_f1"] = R.REPRO_TOLERANCE
    at_ref["metrics"]["macro_f1"] = 0.0
    res_at = R.check_reproducibility(at, at_ref)
    assert res_at["max_abs_diff"] == R.REPRO_TOLERANCE
    assert res_at["within_tolerance"] is True
    over, over_ref = _base_report(), _base_report()
    over["metrics"]["macro_f1"] = R.REPRO_TOLERANCE + 1.0e-5
    over_ref["metrics"]["macro_f1"] = 0.0
    assert R.check_reproducibility(over, over_ref)["within_tolerance"] is False


def test_check_over_tolerance_bites() -> None:
    # A supra-τ metric drift → within_tolerance False → not reproducible, even with GO==GO.
    rer = _base_report()
    rer["metrics"]["min_core_f1"] = 0.9998 - 2.0e-3
    res = R.check_reproducibility(rer, _base_report())
    assert res["verdict_reproduces"] is True  # verdict unchanged...
    assert res["within_tolerance"] is False  # ...but the determinism check bites
    assert res["reproducible"] is False


def test_check_verdict_flip_bites() -> None:
    # A verdict change (the primary gate) → not reproducible, even if metrics are within τ.
    rer = _base_report()
    rer["gate"]["verdict"] = "borderline"
    res = R.check_reproducibility(rer, _base_report())
    assert res["verdict_reproduces"] is False
    assert res["reproducible"] is False


def test_check_config_mismatch_bites() -> None:
    # Different config (a lower epoch count) → not a reproducibility test → fail-closed.
    rer = _base_report()
    rer["config"]["epochs"] = 5
    res = R.check_reproducibility(rer, _base_report())
    assert res["config_ok"] is False
    assert res["reproducible"] is False


def test_check_uncomparable_metric_bites() -> None:
    # A non-finite / missing metric can never certify reproducibility (§10.3).
    rer = _base_report()
    rer["metrics"]["min_core_f1"] = None
    res = R.check_reproducibility(rer, _base_report())
    assert res["all_metrics_comparable"] is False
    assert res["within_tolerance"] is False
    assert res["reproducible"] is False
    # A missing verdict is also fail-closed (both-None must NOT read as "reproduced").
    rer2 = _base_report()
    rer2["gate"]["verdict"] = None
    ref2 = _base_report()
    ref2["gate"]["verdict"] = None
    assert R.check_reproducibility(ref2, rer2)["verdict_reproduces"] is False


def test_check_symmetric_metric_reduction_bites() -> None:
    # A SYMMETRIC reduction (both reports drop the same graded metric) must NOT pass on the
    # remaining subset — the expected 14-metric floor (ADR-0002 A7) is enforced, so the
    # reference report alone cannot shrink the graded set (§10.3).
    ref = _base_report()
    rer = _base_report()
    del ref["metrics"]["per_class_f1"]["Terminator"]
    del rer["metrics"]["per_class_f1"]["Terminator"]
    res = R.check_reproducibility(ref, rer)
    # Every shared metric is finite + identical, so the naive checks would pass...
    assert res["all_metrics_comparable"] is True
    assert res["max_abs_diff"] == 0.0
    # ...but the expected-metric floor bites: a graded metric is missing → not reproducible.
    assert res["expected_metrics_present"] is False
    assert res["within_tolerance"] is False
    assert res["reproducible"] is False


def test_expected_metric_keys_are_the_pinned_fourteen() -> None:
    # Drift guard: the ADR-0002 A7 "14 scalars" floor, single-sourced from the label
    # vocabulary (a class rename can't silently shrink it).
    assert len(R.EXPECTED_METRIC_KEYS) == 14
    assert set(R.EXPECTED_METRIC_KEYS) == set(R.flatten_metrics(_base_report()))


# ======================================================================================
# Tier 2 — committed-report gate (pure JSON; runs in CI once both reports are committed)
# ======================================================================================
def _fail_or_skip(reason: str) -> None:
    """Fail closed under ``TBOX_REQUIRE_REPRO_GATE=1`` (CI once the re-run report lands);
    skip gracefully otherwise (the SLURM re-run has not produced it yet)."""
    if os.environ.get("TBOX_REQUIRE_REPRO_GATE") == "1":
        pytest.fail(
            f"TBOX_REQUIRE_REPRO_GATE=1 but the reproducibility gate is unrunnable: {reason}"
        )
    pytest.skip(reason)


def test_committed_reference_report_present_and_go() -> None:
    # The P1-07 reference (A6) is committed on main, so this always runs and locks that the
    # reproducibility gate compares against a real GO report.
    assert _REF_REPORT.is_file(), "P1-07 reference report reports/p1/seg_smoke_gonogo.json absent"
    ref = R.load_report(_REF_REPORT)
    assert R.verdict_of(ref) == "GO"


def test_seeded_rerun_reproduces_within_tolerance() -> None:
    """THE P1-08 EXIT-GATE ASSERTION: the seeded re-run reproduces the P1-07 go/no-go
    (verdict GO + max|Δ| ≤ ADR-0002 A7 τ). Skips until the re-run report is committed."""
    if not _RERUN_REPORT.is_file():
        _fail_or_skip(
            "P1-08 re-run report reports/p1/seg_smoke_repro.json absent (SLURM run pending)"
        )
    ref = R.load_report(_REF_REPORT)
    rerun = R.load_report(_RERUN_REPORT)
    res = R.check_reproducibility(ref, rerun)
    assert res[
        "config_ok"
    ], f"config/backbone differs, not a reproducibility test: {res['config_mismatches']}"
    assert (
        res["ref_verdict"] == "GO" and res["rerun_verdict"] == "GO"
    ), f"verdict did not reproduce: {res['ref_verdict']} vs {res['rerun_verdict']}"
    assert res["all_metrics_comparable"], f"uncomparable metrics: {res['per_metric_abs_diff']}"
    assert res["within_tolerance"], (
        f"max|Δ|={res['max_abs_diff']} exceeds τ={R.REPRO_TOLERANCE} — determinism failure "
        "(ADR-0002 A7: §7 stop-and-ask, do NOT loosen τ)"
    )
    assert res["reproducible"] is True
