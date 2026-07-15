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
``rinalmo_parity``
    The torch + ``multimolecule`` per-fold SS fine-tune + decode driven by the
    parity gate above (P1-13). Heavy; every torch import is lazy.
``rinalmo_throughput``
    The advisory RiNALMo-giga forward-throughput probe (candidates/sec/GPU on one
    A4000, bf16) — surfaces the §10.2 genome-scale latency risk early; the binding
    latency decision is frozen to the P5 sizing gate (P1-14). PRD §10.2; ADR-0002.
"""

from __future__ import annotations

__all__ = ["archiveii_lofo", "rinalmo_parity", "rinalmo_throughput"]
