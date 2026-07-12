"""Data-artifact builders that are *not* part of the P0 corpus-cleaning pipeline.

Currently holds the Phase-1 Stage-1 segmentation **smoke set** builder
(:mod:`tbox_finder.data.seg_smoke`, P1-06): a small held-out-fold, per-nucleotide
8-class labelled slice used to exercise the segmenter end-to-end before any
training run. Heavy deps (pandas / numpy) are imported lazily inside the IO
functions so the pure selection/label/digest logic imports bare.
"""
