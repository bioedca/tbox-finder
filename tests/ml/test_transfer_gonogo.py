"""P1-07 — ML-gate regression for the Stage-1 transfer go/no-go harness (ADR-0002 D7).

Two tiers, mirroring ``tests/ml/test_eval_gate.py``:

- **Tier 1 (pure stdlib, runs + BLOCKS in bare CI):** the go/no-go **adjudication**, the
  per-nt **F1-over-the-3-core-elements** harness (single-sourced from
  :mod:`tbox_finder.metrics` — this is the "harness computes the gate metric within
  tolerance" check that *extends* the eval-gate coverage to the transfer context), the
  pad-stripping, and the **report validator** — every honesty- and gate-critical guard has
  a clean-pass **and** a violating-fail so it is proven to bite (§8.7). These import only
  stdlib + the bare ``metrics``/``train.stage1_smoke`` modules.
- **Tier 2 (torch; local/cluster only):** the **focal loss** and the segmenter head path on
  synthetic hidden states. Gated by ``TBOX_REQUIRE_TRANSFER_GATE`` — fails closed under the
  env var (so a missing stack can't silently rot the gate), skips gracefully otherwise
  (CI has no GPU/torch, so this tier skips there; it is exercised under ``tbox-ml-dna``
  locally and on the cluster). The full fine-tune path is validated by the SLURM go/no-go
  run itself (``slurm/p1/seg_smoke.sbatch``).
"""

from __future__ import annotations

import os

import pytest

from tbox_finder import metrics as M
from tbox_finder.train import stage1_smoke as T

_TOL = 1e-9


def _fail_or_skip(reason: str) -> None:
    """Fail closed under ``TBOX_REQUIRE_TRANSFER_GATE=1``; skip gracefully otherwise."""
    if os.environ.get("TBOX_REQUIRE_TRANSFER_GATE") == "1":
        pytest.fail(
            f"TBOX_REQUIRE_TRANSFER_GATE=1 but the transfer-gate torch tier is unrunnable: {reason}"
        )
    pytest.skip(reason)


# ======================================================================================
# Tier 1 — pure stdlib (runs + blocks in bare CI)
# ======================================================================================
def test_verdict_bites_both_directions() -> None:
    # GO: min-core-F1 clearly above the go threshold (and the 0.0 baseline).
    assert T.classify_gonogo(0.60) == "GO"
    assert T.classify_gonogo(T.DEFAULT_GO_THRESHOLD) == "GO"  # boundary inclusive
    # borderline: between the two thresholds → §7 stop-and-ask.
    assert T.classify_gonogo(0.20) == "borderline"
    # NO-GO: at/near background.
    assert T.classify_gonogo(T.DEFAULT_NOGO_THRESHOLD) == "NO-GO"  # boundary inclusive
    assert T.classify_gonogo(0.05) == "NO-GO"


def test_verdict_fail_closed_on_unmeasurable_and_baseline() -> None:
    # NaN (a core element absent from truth+pred) → fail-closed NO-GO (§10.3).
    assert T.classify_gonogo(float("nan")) == "NO-GO"
    # A non-number is fail-closed too.
    assert T.classify_gonogo(None) == "NO-GO"
    # The explicit `<= baseline` clause (ADR-0002 D7 "clearly above the background-only
    # baseline") must BITE independently of the nogo threshold. With the default
    # baseline=0.0 it is subsumed by nogo (0.0 <= 0.10), so exercise it with a raised
    # baseline: min-core-F1 above nogo but at/below the baseline is still NO-GO.
    assert T.classify_gonogo(0.15, nogo_threshold=0.10, baseline=0.20) == "NO-GO"
    # and just above a lower baseline → borderline (the clause does not over-fire).
    assert T.classify_gonogo(0.15, nogo_threshold=0.10, baseline=0.05) == "borderline"


def test_verdict_threshold_overrides() -> None:
    # The directional thresholds are provisional/overridable (ADR-0002 D7 "set with fixture").
    assert T.classify_gonogo(0.45, go_threshold=0.50, nogo_threshold=0.10) == "borderline"
    assert T.classify_gonogo(0.55, go_threshold=0.50, nogo_threshold=0.10) == "GO"


def test_strip_pad_removes_ignore_positions() -> None:
    kt, kp = T.strip_pad([1, 2, -100, 5, -100], [1, 0, 2, 5, 3])
    assert kt == [1, 2, 5]
    assert kp == [1, 0, 5]  # pred at pad positions is dropped, not counted as an FP


def test_strip_pad_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        T.strip_pad([1, 2, 3], [1, 2])


def test_flatten_valid_across_windows() -> None:
    yt, yp = T.flatten_valid([[1, -100], [2, 5]], [[1, 0], [2, 5]])
    assert yt == [1, 2, 5]
    assert yp == [1, 2, 5]


def test_flatten_valid_row_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        T.flatten_valid([[1], [2]], [[1]])


def test_background_only_baseline_is_zero_when_core_present() -> None:
    # A background(class-0)-only predictor scores 0 on each core element ⇒ min = 0.0.
    assert T.background_only_min_f1([0, 1, 2, 5, 0, 1]) == 0.0
    assert T.classify_gonogo(T.background_only_min_f1([0, 1, 2, 5])) == "NO-GO"


def test_core_f1_single_sourced_from_gate4_metric() -> None:
    # The transfer harness and GATE-4 grade the SAME per-nt-class unit (no re-derivation):
    # extends the eval-gate metric coverage to the transfer context.
    y_true = [1, 1, 2, 5, 5, 0, 0]
    y_pred = [1, 0, 2, 5, 0, 0, 0]
    harness = T.core_element_f1(y_true, y_pred)
    gate = M.gate4_core_min_f1(y_true, y_pred)["per_element_f1"]
    assert harness == gate
    assert T.CORE_ELEMENT_INDICES == M.CORE_ELEMENT_INDICES == (1, 2, 5)


def test_compute_metrics_harness_within_tolerance() -> None:
    # Hand-computed core F1s (imp.md: "harness computes F1-over-3-core-elements within tol").
    y_true = [1, 1, 2, 5, 5, 0, 0]
    y_pred = [1, 0, 2, 5, 0, 0, 0]
    #   Stem_I (1):  tp=1 fp=0 fn=1 → F1 = 2/3
    #   Specifier(2): tp=1 fp=0 fn=0 → F1 = 1.0
    #   Antiterm (5): tp=1 fp=0 fn=1 → F1 = 2/3
    m = T.compute_metrics(y_true, y_pred)
    assert abs(m["per_element_f1"]["Stem_I"] - 2 / 3) < _TOL
    assert abs(m["per_element_f1"]["Specifier"] - 1.0) < _TOL
    assert abs(m["per_element_f1"]["Antiterminator_Tbox_seq"] - 2 / 3) < _TOL
    assert abs(m["min_core_f1"] - 2 / 3) < _TOL  # min, not mean (ADR-0004 D6)
    assert set(m["per_class_f1"]) == set(M.CLASS_ORDER)


def test_perfect_prediction_is_go() -> None:
    y = [0, 1, 2, 5, 0, 1, 2, 5]
    m = T.compute_metrics(y, y)
    assert m["min_core_f1"] == 1.0
    assert T.classify_gonogo(m["min_core_f1"]) == "GO"


def test_run_smoke_rejects_unwired_and_malformed_knobs() -> None:
    # The run_smoke input guards fire BEFORE set_determinism / any heavy import (§10.3:
    # a stray override must not silently no-op or produce a nonsensical verdict), so they
    # are bare-testable without torch/numpy.
    with pytest.raises(ValueError, match="use_crf"):
        T.run_smoke(T.SmokeConfig(use_crf=True))
    with pytest.raises(ValueError, match="batch_size"):
        T.run_smoke(T.SmokeConfig(batch_size=4))
    with pytest.raises(ValueError, match="go_threshold"):
        T.run_smoke(T.SmokeConfig(go_threshold=0.1, nogo_threshold=0.5))  # go < nogo
    with pytest.raises(ValueError, match="go_threshold"):
        T.run_smoke(T.SmokeConfig(go_threshold=1.5))  # out of [0, 1]


def _valid_report(min_core_f1: float = 0.55) -> dict:
    return T.build_report(
        metrics_block={
            "per_element_f1": {
                "Stem_I": 0.9,
                "Specifier": min_core_f1,
                "Antiterminator_Tbox_seq": 0.8,
            },
            "min_core_f1": min_core_f1,
            "macro_f1": 0.7,
            "micro_f1": 0.75,
            "per_class_f1": {},
        },
        baseline_min_f1=T.BACKGROUND_ONLY_MIN_F1,
        config={"seed": 42, "epochs": 20},
        n_windows=300,
        n_valid_positions=93764,
        go_threshold=T.DEFAULT_GO_THRESHOLD,
        nogo_threshold=T.DEFAULT_NOGO_THRESHOLD,
        windows_parquet="data/interim/p1_seg_smoke/windows.parquet",
        labels_npy="data/interim/p1_seg_smoke/labels.npy",
        prefilter={
            "path": "reports/p1/prefilter_separability.json",
            "verdict": "PASS",
            "binding": False,
        },
        wandb_run_id=None,
    )


def test_build_report_validates_clean() -> None:
    report = _valid_report(0.55)
    assert T.validate_report(report) == []
    assert report["gate"]["verdict"] == "GO"
    assert report["binding"] is True
    assert report["data"]["eval_split"] == "fine_tune_set"


def test_validate_flags_inconsistent_verdict() -> None:
    report = _valid_report(0.55)
    report["gate"]["verdict"] = "NO-GO"  # contradicts min_core_f1=0.55
    problems = T.validate_report(report)
    assert any("verdict inconsistent" in p for p in problems)


def test_validate_flags_bad_baseline() -> None:
    report = _valid_report(0.55)
    report["baseline"]["min_core_f1"] = 0.2  # a background-only predictor must be 0.0
    assert any("baseline.min_core_f1 must be 0.0" in p for p in T.validate_report(report))


def test_validate_flags_missing_eval_split() -> None:
    report = _valid_report(0.55)
    del report["data"]["eval_split"]  # §10.3 honest disclosure required
    assert any("eval_split" in p for p in T.validate_report(report))


def test_validate_flags_non_binding() -> None:
    report = _valid_report(0.55)
    report["binding"] = False  # the P1-07 go/no-go is the BINDING half
    assert any("binding must be boolean True" in p for p in T.validate_report(report))


def test_validate_flags_wrong_core_keys() -> None:
    report = _valid_report(0.55)
    report["metrics"]["per_element_f1"] = {"Stem_I": 0.9, "Specifier": 0.55}  # missing Antiterm
    assert any(
        "per_element_f1 must have exactly the core elements" in p for p in T.validate_report(report)
    )


def test_validate_flags_wrong_revision() -> None:
    report = _valid_report(0.55)
    report["backbone"]["revision"] = "deadbeef"  # not the pinned Caduceus-PS commit
    assert any("revision" in p for p in T.validate_report(report))


def test_validate_flags_measured_false() -> None:
    # `measured` is a §10.3 honesty flag — a report that isn't measured must not validate.
    report = _valid_report(0.55)
    report["measured"] = False
    assert any("measured must be boolean True" in p for p in T.validate_report(report))


def test_validate_flags_not_held_out_only() -> None:
    # held_out_only is the §9.2/§10.3 leave-clade-out honesty flag (highest science stakes).
    report = _valid_report(0.55)
    report["data"]["held_out_only"] = False
    assert any("held_out_only must be boolean True" in p for p in T.validate_report(report))


def test_validate_flags_gate_binding_false() -> None:
    # gate.binding is a distinct guard from the top-level binding flag.
    report = _valid_report(0.55)
    report["gate"]["binding"] = False
    assert any("gate.binding must be boolean True" in p for p in T.validate_report(report))


def test_validate_flags_out_of_range_element_f1() -> None:
    # A fabricated / corrupt F1 outside [0, 1] must be rejected (§10.3 anti-fabrication).
    report = _valid_report(0.55)
    report["metrics"]["per_element_f1"]["Stem_I"] = 1.5
    assert any("per_element_f1" in p and "[0, 1]" in p for p in T.validate_report(report))


def test_validate_flags_out_of_range_min_core_f1() -> None:
    report = _valid_report(0.55)
    report["metrics"]["min_core_f1"] = 1.7  # impossible F1
    assert any("min_core_f1 must be null or in [0, 1]" in p for p in T.validate_report(report))


def test_validate_flags_wrong_schema_version_and_step() -> None:
    report = _valid_report(0.55)
    report["schema_version"] = "9"
    report["step"] = "P9-99"
    problems = T.validate_report(report)
    assert any("schema_version" in p for p in problems)
    assert any("step must be" in p for p in problems)


def test_ignore_index_drift_guard() -> None:
    # -100 is the labels.npy pad / seg-head ignore value; kept as a local literal (bare
    # import) but must not drift from the P1-06 source of truth.
    assert T.IGNORE_INDEX == -100
    try:
        from tbox_finder.data.seg_smoke import IGNORE_INDEX as SEG_IGNORE
    except ImportError:
        pytest.skip("seg_smoke unimportable in a bare env (numpy absent)")
    assert T.IGNORE_INDEX == SEG_IGNORE


def test_report_json_is_strict_no_nan() -> None:
    # A NaN core F1 (unmeasurable) sanitizes to null and yields a fail-closed NO-GO.
    report = T.build_report(
        metrics_block={
            "per_element_f1": {
                "Stem_I": 0.9,
                "Specifier": float("nan"),
                "Antiterminator_Tbox_seq": 0.8,
            },
            "min_core_f1": float("nan"),
            "macro_f1": 0.7,
            "micro_f1": 0.75,
            "per_class_f1": {},
        },
        baseline_min_f1=0.0,
        config={"seed": 42},
        n_windows=300,
        n_valid_positions=1,
        go_threshold=T.DEFAULT_GO_THRESHOLD,
        nogo_threshold=T.DEFAULT_NOGO_THRESHOLD,
        windows_parquet="w",
        labels_npy="l",
        prefilter=None,
        wandb_run_id=None,
    )
    assert report["gate"]["verdict"] == "NO-GO"
    assert report["gate"]["min_core_f1"] is None  # NaN → None (strict JSON)
    assert T.validate_report(report) == []


# ======================================================================================
# Tier 2 — torch (focal loss + segmenter head path); local/cluster only
# ======================================================================================
def _torch_or_skip():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env-dependent
        _fail_or_skip(f"torch not importable ({exc})")
    import torch

    return torch


def test_focal_recovers_cross_entropy_at_gamma_zero() -> None:
    torch = _torch_or_skip()
    import torch.nn.functional as F

    torch.manual_seed(0)
    logits = torch.randn(2, 6, 8)  # (B, L, C)
    targets = torch.randint(0, 8, (2, 6))
    focal = T.focal_cross_entropy(logits, targets, gamma=0.0)
    ce = F.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=T.IGNORE_INDEX, reduction="mean"
    )
    assert torch.allclose(focal, ce, atol=1e-6)


def test_focal_ignores_pad_positions() -> None:
    torch = _torch_or_skip()

    torch.manual_seed(1)
    logits = torch.randn(1, 5, 8)
    targets = torch.tensor([[1, 2, 5, T.IGNORE_INDEX, T.IGNORE_INDEX]])
    # Loss over the full (padded) window == loss over only the 3 valid tokens: the two pad
    # positions contribute nothing, and the mean normalizes by the valid-token count.
    la = T.focal_cross_entropy(logits[:, :3], targets[:, :3], gamma=2.0)
    lb = T.focal_cross_entropy(logits, targets, gamma=2.0)
    assert torch.allclose(la, lb, atol=1e-6)
    assert not torch.isnan(lb)


def test_focal_downweights_easy_more_than_hard() -> None:
    torch = _torch_or_skip()
    import torch.nn.functional as F

    # The defining property of focal loss: an EASY (confident-correct) example is
    # downweighted MORE than a HARD (near-uniform) one — i.e. the focal/CE ratio is smaller
    # for the easy case. (weight <= 1 alone is trivially true for any example.)
    def focal_ce_ratio(peak: float) -> float:
        logits = torch.zeros(1, 1, 8)
        logits[0, 0, 3] = peak  # larger peak ⇒ more confident ⇒ easier
        targets = torch.tensor([[3]])
        focal = float(T.focal_cross_entropy(logits, targets, gamma=2.0))
        ce = float(F.cross_entropy(logits.transpose(1, 2), targets, reduction="mean"))
        return focal / ce

    easy = focal_ce_ratio(10.0)  # confident-correct → p_t ≈ 1 → tiny (1-p_t)^2 weight
    hard = focal_ce_ratio(0.0)  # uniform logits → p_t = 1/8 → weight ≈ (7/8)^2
    assert easy < hard  # easy is selectively downweighted far more than hard
    assert easy < 1.0 and hard < 1.0  # both are <= plain CE


def test_segmenter_head_path_logits_shape() -> None:
    torch = _torch_or_skip()
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter

    seg = Stage1Segmenter(backbone=None, rc_combine="concat")  # backbone-free head path
    hidden = torch.randn(2, 7, 512)  # (B, L, 2*d_model)
    logits = seg.logits_from_hidden(hidden)
    assert logits.shape == (2, 7, 8)  # 8-class per-position
