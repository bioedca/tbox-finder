"""Inference-side operators: window reconciliation, locus construction, scan (P2 onward).

This package holds the *post-model* machinery that turns per-window model output into
per-position predictions and, later, into along-sequence loci (PRD §6, §13.1).

Modules
-------
``reconcile``
    P2-03 — the frozen overlapping-window logit-reconciliation operator: per-position
    log-sum-exp average across all covering windows, then arg-max, applied *before*
    along-sequence element merging (ADR-0005 D3).

``scan``
    P2-10a — the transport around that operator: rebuild a Stage-1 segmenter from a saved
    ``state_dict``, tile an arbitrary sequence at the pinned geometry (padding real contig
    ends), forward every window, and reconcile. Holds the single implementation of the
    tile→forward→reconcile loop, which ``train.train_stage1`` delegates to.

``call``
    P2-10c′-c — the along-sequence candidate-caller ``scan`` names as "a later step":
    threshold ``1 − P(background)``, gap-merge, minimum span → called Stage-1 candidate loci.
    The minimal, recall-favouring form the ρ-pilot (ADR-0003 D6) needs; pins no ADR value
    (ADR-0005 D3 freezes production locus values at the phase gate).

Heavy dependencies (``torch``, ``transformers``) are imported **lazily inside functions**
so this package imports in a bare environment (the CI Tier-1 path); the operators
themselves are ``numpy``-only and accept any array-like (a CPU ``torch.Tensor``
converts through ``numpy.asarray``).
"""

from __future__ import annotations

__all__ = ["call", "reconcile", "scan"]
