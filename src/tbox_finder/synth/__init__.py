"""Synthetic-construct generators (P2-07+).

Three **distinct** generators live here, built at different steps and feeding
different gates — they share no perturbation code and must not be conflated:

``tier2n``
    The synthetic **non-canonical-architecture** (Tier-2N) generator (P2-07).
    Ablates the architecture of *real* corpus parents along literature-grounded
    departures and keeps only the pairs where the ablation demonstrably breaks
    covariance-model detection. Backs the Tier-2N probe set (ADR-0005 D14) and,
    downstream, the P5 synthetic-Tier-2N spike-in recovery (§12 detection-power
    floor) and P6.
``classII``
    The construction-powered **class-II recovery set** (P2-08) — built from real
    class-II parents for Stage-1-only anti-mimicry grading at P4 (ADR-0005 D9).
``divergence``
    The synthetic-**divergence** generator (P4) for the GATE-1 divergence arm.

The subpackage top-level stays import-light (pure stdlib); heavy backends are
imported lazily.
"""

from __future__ import annotations
