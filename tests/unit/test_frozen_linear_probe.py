"""Unit tests for the P1-05 frozen-embedding linear-separability pre-filter.

Three tiers:

- **stdlib** (always runs, incl. bare CI): the verdict logic, the report validator
  (proven to *bite*), ``_sanitize``, pinned constants, and validation of the committed
  measured report.
- **sklearn** (``importorskip`` numpy+sklearn; runs in CI — the pytest job installs the
  data stack — and locally under ``tbox-data``): the probe fit/report on synthetic
  embeddings, incl. the leave-clade-out split partitioning by order and the verdict
  biting on a non-separable fixture.
- **torch** (``importorskip`` torch + GPU-or-skip): frozen embedding extraction shape
  on a tiny real-sequence fixture (skips in CI; runs locally under ``tbox-ml-dna``).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tbox_finder.models.caduceus_backbone import D_MODEL
from tbox_finder.probes import frozen_linear_probe as flp

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_REPORT = REPO_ROOT / "reports" / "p1" / "prefilter_separability.json"

# Soft imports so each tier gates independently (a module-level importorskip would skip
# every test *after* it — e.g. the torch tier would wrongly skip under a no-sklearn env).
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import sklearn  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

_HAS_NUMPY = np is not None
requires_sklearn = pytest.mark.skipif(
    not (_HAS_NUMPY and _HAS_SKLEARN), reason="numpy+scikit-learn not importable"
)
requires_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not importable")


# ======================================================================================
# stdlib tier — always runs.
# ======================================================================================
def test_pinned_constants():
    assert flp.STEP == "P1-05"
    assert flp.CHANCE_LEVEL == 0.5
    assert flp.EMB_DIM == 2 * D_MODEL == 512
    assert flp.VERDICTS == ("PASS", "borderline", "FAIL")
    assert flp.BACKGROUND_POOL == "gc_background"
    assert flp.HELDOUT_ROLE == "heldout"


@pytest.mark.parametrize(
    "auroc, ci_lower, expected",
    [
        (0.90, 0.60, "PASS"),  # CI clears chance -> PASS
        (0.55, 0.45, "borderline"),  # point above chance, CI includes it
        (0.70, float("nan"), "borderline"),  # point above chance, CI unmeasurable
        (0.50, 0.40, "FAIL"),  # exactly chance -> FAIL
        (0.45, 0.40, "FAIL"),  # below chance -> FAIL
        (float("nan"), 0.60, "FAIL"),  # unmeasurable point -> fail-closed
        (0.501, 0.5001, "PASS"),  # strictly-above CI boundary
    ],
)
def test_classify_separability(auroc, ci_lower, expected):
    assert flp.classify_separability(auroc, ci_lower) == expected


def _good_report() -> dict:
    return {
        "schema_version": flp.SCHEMA_VERSION,
        "step": "P1-05",
        "measured": True,
        "floor": {"chance_level": 0.5, "non_binding": True},
        "metrics": {
            "balanced_accuracy": 0.83,
            "auroc": 0.91,
            "auroc_ci_lower": 0.88,
            "auroc_ci_upper": 0.94,
        },
        "leakage": {
            "order_spans_boundary": False,
            "cluster_spans_boundary": False,
            "positives_heldout_only": True,
        },
        # verdict/above_chance follow from metrics: auroc 0.91, CI-lower 0.88 > 0.5 -> PASS.
        "gate": {
            "verdict": "PASS",
            "binding": False,
            "chance_level": 0.5,
            "above_chance": True,
        },
    }


def test_validate_report_good():
    assert flp.validate_report(_good_report()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(step="P1-99"),
        lambda r: r.update(schema_version="0"),
        lambda r: r.update(measured=False),
        lambda r: r.update(measured=1),  # int 1 is not bool True
        lambda r: r["floor"].update(chance_level=0.6),
        lambda r: r["floor"].update(non_binding=False),
        lambda r: r["metrics"].update(auroc=1.5),  # out of [0,1]
        lambda r: r["metrics"].update(auroc=float("nan")),
        lambda r: r["metrics"].update(balanced_accuracy=None),
        lambda r: r["metrics"].update(auroc_ci_lower=1.2),  # present but out of range
        lambda r: r["metrics"].update(auroc_ci_lower=0.95, auroc_ci_upper=0.90),  # lo > hi
        lambda r: r.__setitem__("metrics", {}),  # missing metrics
        lambda r: r["gate"].update(verdict="MAYBE"),
        lambda r: r["gate"].update(verdict="FAIL"),  # inconsistent with PASS metrics
        lambda r: r["gate"].update(above_chance=False),  # inconsistent with the verdict
        lambda r: r["gate"].update(chance_level=0.6),  # must be 0.5
        lambda r: r["gate"].update(binding=True),  # a binding pre-filter is invalid
        lambda r: r["leakage"].update(cluster_spans_boundary=True),  # §8.2 fail-closed
        lambda r: r["leakage"].update(order_spans_boundary="yes"),  # must be a bool
        lambda r: r["leakage"].update(positives_heldout_only=False),
    ],
)
def test_validate_report_bites(mutate):
    report = _good_report()
    mutate(report)
    assert flp.validate_report(report), "validator should reject the mutated report"


def test_sanitize_nan_inf():
    out = flp._sanitize({"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": -math.inf}})
    assert out == {"a": None, "b": [None, 1.0], "c": {"d": None}}


@pytest.mark.skipif(not COMMITTED_REPORT.exists(), reason="measured report not present")
def test_committed_report_valid():
    report = json.loads(COMMITTED_REPORT.read_text())
    assert flp.validate_report(report) == []
    assert report["measured"] is True
    assert report["gate"]["verdict"] in flp.VERDICTS
    assert report["gate"]["binding"] is False
    # The measured report must not certify a homology-leaky probe split (§8.2).
    assert report["leakage"]["cluster_spans_boundary"] is False


# ======================================================================================
# sklearn tier — runs in CI (data stack) and locally under tbox-data.
# ======================================================================================
def _synthetic(delta: float, seed: int = 0, dim: int = 16, per_order: int = 20, n_orders: int = 6):
    """Positives across ``n_orders`` clades vs a same-size background.

    ``delta`` shifts the positive class mean; ``delta == 0`` ⇒ non-separable.
    """
    rng = np.random.default_rng(seed)
    n_pos = per_order * n_orders
    x_pos = rng.normal(delta, 1.0, size=(n_pos, dim))
    x_neg = rng.normal(-delta, 1.0, size=(n_pos, dim))
    pos_order = np.array([f"Order_{i // per_order}" for i in range(n_pos)], dtype=object)
    # 2 windows per cluster (order-pure, since per_order is even) so the cluster-grouped
    # split has non-trivial groups to keep together.
    pos_cluster = np.arange(n_pos) // 2
    return x_pos, x_neg, pos_order, pos_cluster


@requires_sklearn
def test_probe_separable_passes():
    x_pos, x_neg, order, cluster = _synthetic(delta=1.5)
    rep = flp.build_probe_report(x_pos, x_neg, order, cluster, seed=42, n_bootstrap=300)
    assert rep["metrics"]["auroc"] > 0.9
    assert rep["gate"]["verdict"] == "PASS"
    assert rep["leakage"]["cluster_spans_boundary"] is False  # §8.2 homology guard
    assert flp.validate_report(rep) == []


@requires_sklearn
def test_probe_non_separable_not_pass():
    # Identical distributions -> at chance -> must NOT falsely certify PASS.
    x_pos, x_neg, order, cluster = _synthetic(delta=0.0, seed=1)
    rep = flp.build_probe_report(x_pos, x_neg, order, cluster, seed=42, n_bootstrap=300)
    assert rep["gate"]["verdict"] in ("borderline", "FAIL")
    assert rep["metrics"]["auroc"] < 0.75


@requires_sklearn
def test_homology_safe_split_by_cluster():
    # GroupShuffleSplit by cluster_id must place every cluster wholly in train OR test
    # (§8.2 homology guard), so no cluster spans the boundary.
    x_pos, x_neg, order, cluster = _synthetic(delta=1.0)
    rep = flp.build_probe_report(x_pos, x_neg, order, cluster, seed=7, n_bootstrap=100)
    n_train = rep["probe"]["n_clusters_train"]
    n_test = rep["probe"]["n_clusters_test"]
    total_clusters = len(set(cluster.tolist()))
    assert n_train + n_test == total_clusters  # disjoint cluster partition
    assert rep["leakage"]["cluster_spans_boundary"] is False
    assert rep["leakage"]["n_shared_clusters"] == 0
    assert n_test >= 1 and n_train >= 1


@requires_sklearn
def test_probe_report_json_serializable():
    x_pos, x_neg, order, cluster = _synthetic(delta=1.2)
    rep = flp.build_probe_report(x_pos, x_neg, order, cluster, seed=3, n_bootstrap=100)
    # Must round-trip through strict JSON after sanitizing (no NaN/Inf leaks).
    json.dumps(flp._sanitize(rep), allow_nan=False)


# ======================================================================================
# torch tier — local only (skips in CI; runs under tbox-ml-dna on a GPU).
# ======================================================================================
def _fail_or_skip(reason: str):
    if os.environ.get("TBOX_REQUIRE_CADUCEUS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@requires_numpy
def test_extract_embeddings_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        _fail_or_skip("Caduceus forward requires CUDA (ADR-0002 A2 C2)")
    seqs = ["ACGTACGTACGTACGT", "GGGCCCAAATTT", "ACGTNNNNACGT"]
    emb = flp.extract_embeddings(seqs, seed=0)
    assert emb.shape == (3, flp.EMB_DIM)
    assert bool(np.isfinite(emb).all())
