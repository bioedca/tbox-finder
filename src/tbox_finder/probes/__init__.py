"""Frozen-embedding probes over the Stage-1 Caduceus backbone.

This package holds cheap diagnostics over the **frozen** Caduceus-PS hidden states —
the backbone is never fine-tuned (a lightweight probe head may be trained on top):

- :mod:`tbox_finder.probes.frozen_linear_probe` — P1-05 transfer go/no-go *part (i)*:
  a scikit-learn logistic linear-separability pre-filter over frozen embeddings of
  held-out bacterial T-box windows vs GC-matched prokaryotic background (advisory,
  **non-binding**; the binding gate is the P1-07 per-nucleotide segmentation smoke).
- :mod:`tbox_finder.probes.frozen_seg_probe` — P1-09 segmentation-head fallback: freezes
  the Caduceus-PS backbone and trains **only** the P1-03 per-position 8-class seg head
  (Linear, optional CRF) on the cached frozen embeddings of the P1-06 smoke set. The
  per-nucleotide counterpart of the P1-05 window-level probe; a **non-gated** fallback
  baseline (PRD §10.1/§18.2) ready for P2 if end-to-end fine-tuning underperforms.

Heavy libraries (``torch``/``transformers`` for extraction, and for the P1-05 probe
``scikit-learn``) are imported lazily inside functions so the package imports in a bare env.
"""
