"""Post-hoc recalibration machinery for tbox-finder (ADR-0005 D11 + Amendment A11).

The subpackage top-level stays import-light (pure stdlib) so the unit tier runs in
bare CI, which installs **no torch**. ``numpy`` is imported at module scope inside
``temperature`` (CI installs it); ``torch``/``pandas`` are lazy, inside the functions
that forward a checkpoint.

Modules
-------
``temperature``
    Temperature scaling: the exact convex 1-D fit of one shared scalar ``T`` on the
    Stage-1 per-nucleotide 8-class logits, ``T`` applied **before** the frozen ADR-0005
    D3+A3 reconciliation operator, and the **non-gated** P2-13 reliability read
    (per-class one-vs-rest binned ECE through the frozen ``metrics.binned_ece``, with
    cluster-blocked CIs). GATE-2's gated ECE is P3-exit business on the P3-02 ``calib``
    carve — nothing here is a calibration claim, and no ``T`` fitted here is shipped.
``recalibrate``
    The P3-07 stack for the **Stage-2 binary head**: ``T`` fitted on the P3-02 ``calib``
    rung (and structurally on nothing else), then the Saerens/Elkan log-odds prior-shift,
    then a producer exposing **both** the D11 named posterior (temperature-scaled,
    pre-prior-shift — the GATE-2-gated object) and the prior-shifted one (reported,
    non-gated). It **reuses** ``temperature.fit_temperature`` rather than carrying a second
    temperature fit: Stage 2's one logit stacked as ``[0, z]`` makes that multi-class fit
    identically the binary one. No deployment prior is pinned here — the PRD gives it only
    as prose, so both priors are required arguments with no defaults.
"""

from __future__ import annotations

__all__ = ["recalibrate", "temperature"]
