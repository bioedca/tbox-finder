"""P2-09 — the production-fold selector (``exclude_selection_val``) and its config guard.

Bare-CI tier: pure dataclass + mapping logic, no torch (``train_stage1`` imports torch only
lazily inside functions, so :class:`Stage1TrainConfig` and :func:`_cfg_from_mapping` load in
the bare env).

P2-09 retrains the P2-06 winner on the **full ADR-0004 D5 fold** (8,303 records) rather than
the ``inner_train`` fold (7,472) a sweep selects on: the config is already chosen, so
withholding 10.01% of the corpus from the shipped checkpoint buys nothing. That is
``exclude_selection_val=False``.

The trap this file guards is the one described in the field's own docstring and in
``test_eval_val_clause.py::test_gate_is_false_when_the_run_trained_on_the_eval_fold``: a run
that trains on the full fold AND scores the selection-val fold puts the eval set inside the
training stream, and the disjointness evidence would **still read zero** — because
``load_selection_val_records`` measures overlap against the inner_train fold *definition*, not
against the records the optimiser saw. The combination cannot produce an honest val number,
so ``__post_init__`` refuses it at construction, making it unreachable from Hydra rather than
merely detected afterwards.
"""

from __future__ import annotations

import pytest

from tbox_finder.train.train_stage1 import Stage1TrainConfig, _cfg_from_mapping


# ── The config guard: the trap is refused at construction ─────────────────────
def test_full_fold_with_eval_val_is_refused():
    """``eval_val=True`` + ``exclude_selection_val=False`` = train on the val fold, then
    score it. Refused at construction, not after the model loads."""
    with pytest.raises(ValueError, match="in-sample"):
        Stage1TrainConfig(eval_val=True, exclude_selection_val=False)


def test_the_guard_names_why_the_disjointness_evidence_does_not_catch_it():
    """The message must explain the non-obvious part — that the leakage block reads zero
    for this combination — or a reader will 'fix' it by trusting the green evidence."""
    with pytest.raises(ValueError) as exc:
        Stage1TrainConfig(eval_val=True, exclude_selection_val=False)
    msg = str(exc.value)
    assert "inner_train fold" in msg.lower() or "definition" in msg.lower()


def test_production_config_is_allowed():
    """The P2-09 config: full fold, no val metric. Must construct cleanly."""
    cfg = Stage1TrainConfig(eval_val=False, exclude_selection_val=False)
    assert cfg.exclude_selection_val is False
    assert cfg.eval_val is False


@pytest.mark.parametrize(
    ("eval_val", "exclude"),
    [
        (True, True),  # inner_train + score val — the P2-06 sweep shape
        (False, True),  # inner_train, no eval — a plain inner-fold run
        (False, False),  # full D5 fold, no eval — the P2-09 production shape
    ],
)
def test_coherent_combinations_are_allowed(eval_val, exclude):
    cfg = Stage1TrainConfig(eval_val=eval_val, exclude_selection_val=exclude)
    assert cfg.eval_val is eval_val
    assert cfg.exclude_selection_val is exclude


def test_default_config_holds_out_the_selection_fold():
    """The default is the SAFE direction: exclude the val fold, so a run that forgets to
    choose does not silently train-on-train. Losing 10% is loud; train-on-train is silent,
    and the two failure directions are not symmetric (load_corpus_records docstring)."""
    cfg = Stage1TrainConfig()
    assert cfg.exclude_selection_val is True
    assert cfg.eval_val is True  # and the default pair is coherent


# ── Threading from Hydra: a bare top-level override reaches the dataclass ──────
def test_exclude_selection_val_threads_from_a_top_level_key():
    """It is a top-level dataclass field, so ``_cfg_from_mapping`` picks it up from a bare
    ``exclude_selection_val=…`` override — the same path the sbatch uses. A field that
    composed into the config but never reached the loader is the P2-04 `/data`-group defect;
    this asserts it does reach the dataclass."""
    cfg = _cfg_from_mapping({"exclude_selection_val": False, "eval_val": False})
    assert cfg.exclude_selection_val is False
    # And omitting it keeps the safe default.
    cfg2 = _cfg_from_mapping({})
    assert cfg2.exclude_selection_val is True


def test_threading_the_trap_combination_still_fails_closed():
    """Passing only ``exclude_selection_val=False`` (forgetting ``eval_val=False``) must NOT
    silently train-on-train: the default ``eval_val=True`` then trips the __post_init__
    guard. The sbatch must pass BOTH — this proves a half-specified override fails loud."""
    with pytest.raises(ValueError, match="in-sample"):
        _cfg_from_mapping({"exclude_selection_val": False})
