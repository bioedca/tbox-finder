"""Evaluation harnesses for tbox-finder (Phase-1+).

The subpackage top-level stays import-light (pure stdlib) so the golden/unit
tiers run in bare CI without pandas/numpy/torch. Heavy backends, if any, are
imported lazily inside the functions that need them.

Modules
-------
``archiveii_lofo``
    The ArchiveII nine-family leave-one-family-out (inter-family) secondary-
    structure benchmark + base-pair-F1 metric used for the RiNALMo mirror
    **parity gate** (P1-12 builds the benchmark + records the published target;
    P1-13 runs the mirror against it). PRD §10.2; ADR-0002 D5.
"""

from __future__ import annotations

__all__ = ["archiveii_lofo"]
