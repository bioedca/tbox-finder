#!/usr/bin/env python3
"""Measure what ADR-0005 A18 Pin 2's plug-in conjunct does to GATE-2's error rates.

A18 Pin 2 (P3-exit-b, 2026-11-04) grades GATE-2 on the D11 debiased binned ECE **and**
the plug-in ECE: both must be ``<= 0.05``. This script is the reproducible record of the
trade-off that decision was signed on, so the ADR quotes measured numbers rather than
asserted ones (CLAUDE.md §10.3).

**The simulation.** For each cell ``(n, scale)`` and each of ``REPS`` replicates:

* draw ``z ~ Uniform[-3.2, 3.2]`` and ``y ~ Bernoulli(sigma(z))``, so ``sigma(z)`` is the
  true conditional mean;
* report ``p = sigma(scale * z)``, which is calibrated at ``scale = 1`` and over-confident
  above it;
* grade ``p`` with the SHIPPED estimator (``metrics.binned_ece``, 15 equal-mass bins,
  debiased and plug-in) and the SHIPPED predicate (``metrics.gate2_ece_pass``).

The true calibration error of ``p`` is ``E|sigma(scale*z) - sigma(z)|`` (``p`` is monotone
in ``z``, so ``E[y | p] = sigma(z)``), computed by midpoint quadrature over ``z``.

**Scope.** The logit range is deliberately spread, which is the hardest shape for binning
noise. The production posterior is near-separated, which is far easier (P3-10: 12 of 15
bins at accuracy exactly 0 or 1), so these rates are a conservative picture of the noise,
not a model of the shipped arm. ``n = 3045`` is GATE-2's own test-rung size; ``1089`` is
the calib carve's; ``420`` is the pre-A18 unit-test fixture's.

Each cell has its own seed (``[BASE_SEED, n, round(100 * scale)]``), so any one cell
re-derives without re-running the others.

Usage::

    PYTHONPATH=src python scripts/measure_gate2_ece_conjunct.py \\
        --out reports/p3/gate2_ece_conjunct_sim.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np

from tbox_finder import metrics as M
from tbox_finder.provenance import build_provenance

BASE_SEED = 20261104
REPS = 200
N_BINS = M.ECE_N_BINS
Z_RANGE = (-3.2, 3.2)
QUADRATURE_POINTS = 1_000_000
ROW_COUNTS: tuple[int, ...] = (420, 1089, 3045)
SCALES: tuple[float, ...] = (1.0, 1.4, 1.6, 1.9)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def true_ece(scale: float) -> float:
    """``E|sigma(scale*z) - sigma(z)|`` over ``z ~ Uniform(Z_RANGE)``, by midpoint rule."""
    lo, hi = Z_RANGE
    z = lo + (hi - lo) * (np.arange(QUADRATURE_POINTS) + 0.5) / QUADRATURE_POINTS
    return float(np.mean(np.abs(_sigmoid(scale * z) - _sigmoid(z))))


def cell(n: int, scale: float, reps: int = REPS) -> dict:
    """One ``(n, scale)`` cell, seeded on its own coordinates."""
    rng = np.random.default_rng([BASE_SEED, n, round(100 * scale)])
    debiased, plugin, conjunct = [], [], []
    for _ in range(reps):
        z = rng.uniform(*Z_RANGE, n)
        y = (rng.random(n) < _sigmoid(z)).astype(int).tolist()
        p = _sigmoid(scale * z).tolist()
        d = M.binned_ece(y, p, N_BINS, debias=True)
        g = M.binned_ece(y, p, N_BINS, debias=False)
        debiased.append(d)
        plugin.append(g)
        conjunct.append(M.gate2_ece_pass(d, ece_plugin=g))
    gate = M.ECE_GATE
    return {
        "n": n,
        "scale": scale,
        "reps": reps,
        "true_ece": true_ece(scale),
        "mean_debiased_ece": math.fsum(debiased) / reps,
        "mean_plugin_ece": math.fsum(plugin) / reps,
        "p_pass_debiased_only": sum(v <= gate for v in debiased) / reps,
        "p_pass_plugin_only": sum(v <= gate for v in plugin) / reps,
        "p_pass_a18_conjunct": sum(conjunct) / reps,
    }


def measure() -> dict:
    cells = [cell(n, s) for n in ROW_COUNTS for s in SCALES]
    return {
        "schema_version": "1",
        "step": "P3-exit-b",
        "adr": "ADR-0005 A18 Pin 2",
        "generated_by": "scripts/measure_gate2_ece_conjunct.py",
        "design": {
            "z_distribution": f"Uniform{list(Z_RANGE)}",
            "labels": "Bernoulli(sigma(z))",
            "reported_posterior": "sigma(scale * z)",
            "estimator": "metrics.binned_ece, 15 equal-mass bins (debiased and plug-in)",
            "predicate": "metrics.gate2_ece_pass (debiased AND plug-in <= 0.05)",
            "gate": M.ECE_GATE,
            "n_bins": N_BINS,
            "reps_per_cell": REPS,
            "base_seed": BASE_SEED,
            "per_cell_seed": "[base_seed, n, round(100 * scale)]",
            "true_ece_quadrature_points": QUADRATURE_POINTS,
        },
        "scope_note": (
            "A deliberately SPREAD logit range, the hardest shape for binning noise. The "
            "production posterior is near-separated, so these rates are conservative about "
            "noise and are not a model of the shipped arm. n = 3045 is GATE-2's test rung."
        ),
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = measure()
    report["provenance"] = build_provenance(
        rule="scripts/measure_gate2_ece_conjunct.py",
        script="scripts/measure_gate2_ece_conjunct.py",
        seed=BASE_SEED,
        adr="ADR-0005",
        extra={"numpy": np.__version__, "python": platform.python_version()},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for c in report["cells"]:
        print(
            f"n={c['n']:5d} scale={c['scale']:.1f} true={c['true_ece']:.4f} "
            f"deb={c['mean_debiased_ece']:.4f} plug={c['mean_plugin_ece']:.4f} "
            f"P(deb)={c['p_pass_debiased_only']:.3f} P(A18)={c['p_pass_a18_conjunct']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
