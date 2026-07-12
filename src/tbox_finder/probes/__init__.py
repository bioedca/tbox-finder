"""Frozen-embedding probes over the Stage-1 Caduceus backbone.

This package holds cheap, *non-training* diagnostics that read the frozen
Caduceus-PS hidden states without fine-tuning the backbone:

- :mod:`tbox_finder.probes.frozen_linear_probe` — P1-05 transfer go/no-go *part (i)*:
  a scikit-learn logistic linear-separability pre-filter over frozen embeddings of
  held-out bacterial T-box windows vs GC-matched prokaryotic background (advisory,
  **non-binding**; the binding gate is the P1-07 per-nucleotide segmentation smoke).

Heavy libraries (``torch``/``transformers`` for extraction, ``scikit-learn`` for the
probe) are imported lazily inside functions so the package imports in a bare env.
"""
