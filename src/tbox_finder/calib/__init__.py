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
"""

from __future__ import annotations

__all__ = ["temperature"]
