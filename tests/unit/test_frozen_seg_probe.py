"""Unit tests for the P1-09 frozen-embedding + per-position linear/CRF seg-head probe.

Three tiers (mirroring the P1-05 pre-filter tests):

- **stdlib** (always runs, incl. bare CI): pinned constants, the reused-``IGNORE_INDEX``
  drift guard, the report validator (proven to *bite* on every honesty/anti-fabrication
  invariant), ``_sanitize``, and validation of the committed measured report.
- **torch** (``importorskip torch``; **CPU-only** — no backbone/GPU needed): the head
  trains on synthetic frozen per-position embeddings and decodes a **valid length-L
  8-class segmentation** (the step's validation gate); a separable set is *learned* and a
  pure-noise set *cannot be faked* (the metric bites); the CRF variant decodes validly;
  and only the seg head trains (backbone frozen). Runs under ``tbox-ml-dna``; skips in
  bare CI.
- **torch + GPU** (``importorskip transformers`` + CUDA-or-skip): frozen embedding
  extraction shape on a tiny real-sequence fixture (ADR-0002 A2 C2). Skips in CI.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tbox_finder.labels import CLASS_ORDER, CORE_ELEMENTS
from tbox_finder.models.caduceus_backbone import D_MODEL, REVISION
from tbox_finder.probes import frozen_seg_probe as fsp
from tbox_finder.train import stage1_smoke as smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_REPORT = REPO_ROOT / "reports" / "p1" / "frozen_seg_probe.json"

# Soft imports so each tier gates independently.
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

_HAS_NUMPY = np is not None
requires_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not importable")
requires_torch = pytest.mark.skipif(
    not (_HAS_NUMPY and _HAS_TORCH), reason="numpy+torch not importable"
)


# ======================================================================================
# stdlib tier — always runs.
# ======================================================================================
def test_pinned_constants():
    assert fsp.STEP == "P1-09"
    assert fsp.EMB_DIM == 2 * D_MODEL == 512
    assert fsp.NUM_CLASSES == len(CLASS_ORDER) == 8
    assert fsp.DEFAULT_RC_COMBINE == "concat"
    assert fsp.PRD_SECTIONS == ("§10.1", "§18.2")


def test_ignore_index_matches_smoke():
    # The probe reuses the P1-06/P1-07 pad convention (drift guard, §8.3).
    assert fsp.IGNORE_INDEX == smoke.IGNORE_INDEX == -100


def test_reused_smoke_helpers_present():
    # The probe reuses these private stage1_smoke helpers by attribute access (runtime),
    # so a rename upstream would only break at call time — and _make_ids_of's sole caller
    # (extract) is CUDA-gated, so a rename could otherwise ship green. This bare-CI drift
    # guard fails at import-check time instead (mirrors test_ignore_index_matches_smoke).
    for name in (
        "_make_ids_of",
        "_epoch_order",
        "load_smoke_windows",
        "set_determinism",
        "focal_cross_entropy",
        "compute_metrics",
        "flatten_valid",
    ):
        assert hasattr(smoke, name), f"stage1_smoke.{name} missing — reuse contract drifted"


def _good_report() -> dict:
    return {
        "schema_version": fsp.SCHEMA_VERSION,
        "step": "P1-09",
        "measured": True,
        "binding": False,
        "frozen_backbone": True,
        "backbone": {"revision": REVISION},
        "metrics": {
            "per_class_f1": {name: 0.9 for name in CLASS_ORDER},
            "per_element_f1": {name: 0.9 for name in CORE_ELEMENTS},
            "min_core_f1": 0.85,
            "macro_f1": 0.88,
            "micro_f1": 0.92,
        },
        "segmentation_validity": {
            "all_windows_valid_8class": True,
            "n_windows_decoded": 300,
            "min_pred_class": 0,
            "max_pred_class": 7,
        },
        "data": {"held_out_only": True, "eval_split": "fine_tune_set"},
        "gate": {"binding": False, "gated": False, "kind": "fallback_baseline"},
    }


def test_validate_report_good():
    assert fsp.validate_report(_good_report()) == []


def test_validate_report_allows_null_absent_class_f1():
    # A class absent from truth+pred yields NaN → sanitized None; the validator allows it.
    r = _good_report()
    r["metrics"]["per_class_f1"]["Terminator"] = None
    r["metrics"]["min_core_f1"] = None
    assert fsp.validate_report(r) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(step="P1-99"),
        lambda r: r.update(schema_version="0"),
        lambda r: r.update(measured=False),
        lambda r: r.update(measured=1),  # int 1 is not bool True
        lambda r: r.update(binding=True),  # a binding gate is invalid (non-gated fallback)
        lambda r: r.update(frozen_backbone=False),  # backbone must be frozen
        lambda r: r["backbone"].update(revision="deadbeef"),  # stale/unpinned revision
        lambda r: r["metrics"]["per_class_f1"].pop("Discriminator"),  # missing a class
        lambda r: r["metrics"]["per_class_f1"].update(EXTRA_CLASS=0.5),  # extra 9th class
        lambda r: r["metrics"]["per_class_f1"].update(background=1.5),  # out of [0,1]
        lambda r: r["metrics"]["per_class_f1"].update(background="x"),  # non-numeric
        lambda r: r["metrics"].__setitem__("per_element_f1", {"Stem_I": 0.9}),  # wrong core set
        lambda r: r["metrics"]["per_element_f1"].update(Specifier=1.5),  # core F1 out of [0,1]
        lambda r: r["metrics"]["per_element_f1"].update(Specifier="x"),  # core F1 non-numeric
        lambda r: r["metrics"].update(min_core_f1=1.2),  # out of range
        lambda r: r["metrics"].update(macro_f1=float("nan")),  # NaN not allowed (must be None)
        lambda r: r.__setitem__("metrics", {}),  # missing metrics
        lambda r: r["segmentation_validity"].update(all_windows_valid_8class=False),
        lambda r: r["segmentation_validity"].update(max_pred_class=8),  # >= NUM_CLASSES
        lambda r: r["segmentation_validity"].update(min_pred_class=-1),  # < 0
        lambda r: r["segmentation_validity"].update(max_pred_class=True),  # bool not int
        lambda r: r.pop("segmentation_validity"),  # missing block
        lambda r: r["data"].update(held_out_only=False),  # §9.2 leakage guard
        lambda r: r["data"].update(eval_split="test"),  # §10.3 honest disclosure
        lambda r: r["gate"].update(binding=True),
        lambda r: r["gate"].update(gated=True),  # a gated fallback is invalid
    ],
)
def test_validate_report_bites(mutate):
    report = _good_report()
    mutate(report)
    assert fsp.validate_report(report), "validator should reject the mutated report"


def test_sanitize_nan_inf():
    out = fsp._sanitize({"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": -math.inf}})
    assert out == {"a": None, "b": [None, 1.0], "c": {"d": None}}


@pytest.mark.skipif(not COMMITTED_REPORT.exists(), reason="measured report not present")
def test_committed_report_valid():
    report = json.loads(COMMITTED_REPORT.read_text())
    assert fsp.validate_report(report) == []
    assert report["measured"] is True
    assert report["binding"] is False  # non-gated fallback baseline (PRD §10.1)
    assert report["frozen_backbone"] is True
    assert report["gate"]["gated"] is False
    # The deliverable: every window decoded to a valid length-L 8-class segmentation.
    assert report["segmentation_validity"]["all_windows_valid_8class"] is True
    # Held-out-only + honest eval-split disclosure (§9.2 / §10.3).
    assert report["data"]["held_out_only"] is True
    assert report["data"]["eval_split"] == "fine_tune_set"


# ======================================================================================
# torch tier — CPU-only (no backbone / GPU); runs under tbox-ml-dna, skips in bare CI.
# ======================================================================================
def _synthetic_seg(*, n_windows=10, dim=None, seed=0, separable=True):
    """Synthetic frozen per-position embeddings: ``[(hidden (L,dim), labels (L,)), ...]``.

    Each class has a prototype vector; a position's hidden state is its class prototype +
    noise. ``separable=True`` ⇒ well-separated prototypes (a linear head can learn it);
    ``separable=False`` ⇒ zero prototypes ⇒ pure noise ⇒ labels unlearnable.
    """
    dim = dim or fsp.EMB_DIM
    rng = np.random.default_rng(seed)
    scale = 3.0 if separable else 0.0
    protos = rng.normal(0.0, 1.0, size=(fsp.NUM_CLASSES, dim)) * scale
    windows = []
    for _ in range(n_windows):
        length = int(rng.integers(20, 40))
        labels = rng.integers(0, fsp.NUM_CLASSES, size=length)
        hidden = protos[labels] + rng.normal(0.0, 0.3, size=(length, dim))
        windows.append((hidden.astype(np.float32), labels.astype(np.int16)))
    return windows


@requires_torch
def test_unpack_windows_roundtrip():
    windows = _synthetic_seg(n_windows=4, dim=fsp.EMB_DIM, seed=5)
    hidden_flat = np.concatenate([w[0] for w in windows], axis=0)
    labels_flat = np.concatenate([w[1] for w in windows], axis=0)
    lengths = np.array([w[0].shape[0] for w in windows], dtype=np.int32)
    unpacked = fsp._unpack_windows(hidden_flat, labels_flat, lengths)
    assert len(unpacked) == 4
    for (h, lab), (h0, lab0) in zip(unpacked, windows, strict=True):
        assert h.shape == h0.shape
        assert np.array_equal(lab, lab0)


@requires_torch
def test_frozen_probe_only_head_trains():
    # The defining property: with backbone=None + concat (parameter-free), ONLY the P1-03
    # seg head trains — the frozen-embedding fallback (nothing in the backbone updates).
    seg = fsp.build_frozen_probe(fsp.ProbeConfig())
    trainable = [name for name, p in seg.named_parameters() if p.requires_grad]
    assert trainable, "the seg head must be trainable"
    assert all(name.startswith("head.") for name in trainable)
    assert any("classifier" in name for name in trainable)


@requires_torch
def test_frozen_probe_valid_segmentation_and_learns():
    windows = _synthetic_seg(separable=True, seed=0)
    cfg = fsp.ProbeConfig(epochs=60, lr=1.0e-3, device="cpu")
    smoke.set_determinism(cfg.seed)
    seg = fsp.build_frozen_probe(cfg)
    fsp.train_frozen_head(seg, windows, cfg, log=lambda _m: None)
    y_true, y_pred, validity = fsp.evaluate_frozen_head(seg, windows)

    # The step's validation gate: a valid length-L 8-class segmentation for every window.
    assert validity["all_windows_valid_8class"] is True
    assert validity["n_windows_decoded"] == len(windows)
    assert 0 <= validity["min_pred_class"] <= validity["max_pred_class"] < fsp.NUM_CLASSES
    assert len(y_true) == len(y_pred) == sum(len(w[1]) for w in windows)

    metrics = smoke.compute_metrics(y_true, y_pred)
    # A linear head over separable frozen features learns the per-position classes well.
    assert metrics["macro_f1"] > 0.8


@requires_torch
def test_frozen_probe_cannot_fake_signal_from_noise():
    # Pure noise (zero prototypes) ⇒ labels are unlearnable. Evaluated on a HELD-OUT set
    # of noise windows the head cannot generalize ⇒ low F1 (the metric bites), yet the
    # decode is still a VALID 8-class segmentation (the deliverable holds regardless).
    #
    # NB: a 512-dim head over few same-set positions can *memorize* noise (features >
    # samples), so this honesty check must eval on disjoint windows. Production is immune
    # — the real P1-06 set has ~94k positions ≫ 512 features, so no such memorization.
    windows = _synthetic_seg(separable=False, seed=1, n_windows=24)
    train_w, eval_w = windows[:16], windows[16:]
    cfg = fsp.ProbeConfig(epochs=40, lr=1.0e-3, device="cpu")
    smoke.set_determinism(cfg.seed)
    seg = fsp.build_frozen_probe(cfg)
    fsp.train_frozen_head(seg, train_w, cfg, log=lambda _m: None)
    y_true, y_pred, validity = fsp.evaluate_frozen_head(seg, eval_w)
    assert validity["all_windows_valid_8class"] is True
    metrics = smoke.compute_metrics(y_true, y_pred)
    assert metrics["macro_f1"] < 0.5


@requires_torch
def test_frozen_probe_generalizes_real_signal_held_out():
    # The dual of the noise check: separable frozen features DO generalize to held-out
    # windows (real per-position structure, not memorization) ⇒ high held-out F1.
    windows = _synthetic_seg(separable=True, seed=3, n_windows=24)
    train_w, eval_w = windows[:16], windows[16:]
    cfg = fsp.ProbeConfig(epochs=60, lr=1.0e-3, device="cpu")
    smoke.set_determinism(cfg.seed)
    seg = fsp.build_frozen_probe(cfg)
    fsp.train_frozen_head(seg, train_w, cfg, log=lambda _m: None)
    y_true, y_pred, validity = fsp.evaluate_frozen_head(seg, eval_w)
    assert validity["all_windows_valid_8class"] is True
    metrics = smoke.compute_metrics(y_true, y_pred)
    assert metrics["macro_f1"] > 0.8


@requires_torch
def test_evaluate_flags_invalid_decode():
    # The validity flag must genuinely COMPUTE at the decode layer, not just be checked at
    # the report layer: a decoder that emits an out-of-range class (length-preserving, so
    # the strict-zip flatten still runs) flips all_windows_valid_8class to False.
    class _BadDecoder:
        def parameters(self):
            yield torch.zeros(1)  # so next(...).device resolves to cpu

        def eval(self):
            return self

        def decode_from_hidden(self, hidden, mask=None):
            length = hidden.shape[1]
            return [[fsp.NUM_CLASSES + 3] * length]  # correct length, out of [0, NUM_CLASSES)

    windows = [(np.zeros((5, fsp.EMB_DIM), np.float32), np.zeros(5, np.int16))]
    _, _, validity = fsp.evaluate_frozen_head(_BadDecoder(), windows)
    assert validity["all_windows_valid_8class"] is False
    assert validity["max_pred_class"] == fsp.NUM_CLASSES + 3


@requires_torch
def test_frozen_probe_crf_valid_segmentation():
    # The CRF variant (Linear + in-repo linear-chain CRF) decodes a valid length-L path.
    windows = _synthetic_seg(separable=True, seed=2, n_windows=6)
    cfg = fsp.ProbeConfig(epochs=10, lr=1.0e-3, use_crf=True, device="cpu")
    smoke.set_determinism(cfg.seed)
    seg = fsp.build_frozen_probe(cfg)
    fsp.train_frozen_head(seg, windows, cfg, log=lambda _m: None)
    y_true, y_pred, validity = fsp.evaluate_frozen_head(seg, windows)
    assert validity["all_windows_valid_8class"] is True
    assert 0 <= validity["min_pred_class"] <= validity["max_pred_class"] < fsp.NUM_CLASSES
    assert len(y_true) == len(y_pred) == sum(len(w[1]) for w in windows)


# ======================================================================================
# torch + GPU tier — local only (skips in CI; runs under tbox-ml-dna on a GPU).
# ======================================================================================
def _fail_or_skip(reason: str):
    if os.environ.get("TBOX_REQUIRE_CADUCEUS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@requires_numpy
def test_extract_seg_embeddings_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        _fail_or_skip("Caduceus forward requires CUDA (ADR-0002 A2 C2)")
    from tbox_finder.models.caduceus_backbone import load_tokenizer

    ids_of = smoke._make_ids_of(load_tokenizer())
    windows = [
        ("ACGTACGTACGT", [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], 12),
        ("GGGCCCAAA", [0, 0, 0, 1, 1, 1, 2, 2, 2], 9),
    ]
    hidden_flat, labels_flat, lengths = fsp.extract_seg_embeddings(windows, ids_of, seed=0)
    assert hidden_flat.shape == (21, fsp.EMB_DIM)
    assert labels_flat.shape == (21,)
    assert lengths.tolist() == [12, 9]
    assert bool(np.isfinite(hidden_flat).all())
