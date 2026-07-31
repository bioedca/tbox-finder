"""Stage-2 (RiNALMo RNA re-ranker) dataset + tokenizer package (PRD §6, §10.2).

Stage 2 ingests **sequence only** (PRD §6): predicted/annotated structure is used
solely as the §8/§11 auxiliary *target*, never as a model input channel. Nothing in
this package may put a structure string on the input side of the model.

Modules are import-light on purpose — :mod:`tbox_finder.stage2.tokenizer` pins the
RiNALMo vocabulary in pure Python so the dataset build runs in the ``data`` env
(``envs/data.yml``) alongside every other ``data.smk``-class rule, while ``multimolecule``
lives only in ``envs/ml-rna.yml`` (ADR-0002 A4/A8 env split). Any ``multimolecule`` /
``torch`` import in this package must be **lazy** (inside a function), matching the
precedent in :mod:`tbox_finder.eval.rinalmo_parity`.
"""

from __future__ import annotations

__all__ = ["dataset", "tokenizer"]
