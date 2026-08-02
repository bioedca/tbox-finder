"""Stage-2 (RiNALMo RNA re-ranker) dataset, tokenizer and model package (PRD §6, §10.2).

Stage 2 ingests **sequence only** (PRD §6): predicted/annotated structure is used
solely as the §8/§11 auxiliary *target*, never as a model input channel. Nothing in
this package may put a structure string on the input side of the model.

Modules are import-light on purpose — :mod:`tbox_finder.stage2.tokenizer` pins the
RiNALMo vocabulary in pure Python and :mod:`tbox_finder.stage2.heads` derives the head
vocabularies with stdlib only, so the dataset build runs in the ``data`` env
(``envs/data.yml``) alongside every other ``data.smk``-class rule, while ``multimolecule``
lives only in ``envs/ml-rna.yml`` (ADR-0002 A4/A8 env split). In
:mod:`~tbox_finder.stage2.dataset`, :mod:`~tbox_finder.stage2.tokenizer` and
:mod:`~tbox_finder.stage2.heads`, any ``multimolecule`` / ``torch`` import must be
**lazy** (inside a function), matching the precedent in
:mod:`tbox_finder.eval.rinalmo_parity`.

The one exception is :mod:`tbox_finder.stage2.model`, which *is* a ``torch`` model and
imports ``torch`` at module top (the :mod:`tbox_finder.models.stage1_segmenter`
precedent). This ``__init__`` deliberately does not import it, so the package itself
still imports bare and the ``data``-env rules are unaffected.
:mod:`tbox_finder.stage2.losses` follows the lazy-import rule, not the exception: its
weighting scheme, dominance rule and config validation are all pure, so they run — and are
tested — in the bare CI env where torch is absent.
"""

from __future__ import annotations

__all__ = ["dataset", "heads", "losses", "model", "tokenizer"]
