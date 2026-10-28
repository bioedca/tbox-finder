"""Which corpus records each Stage-1 arm trained on — one definition, read by every consumer.

Promoted out of ``scripts/mint_p3_16_benchmark.py`` at P3-16′-remint (ADR-0005 A15 Pin 1):
the benchmark minter and the decoy-host screen (``integration/host_screen.py``) must read
the *same* fold, not two re-implementations that agree by coincidence.

The two arms differ by exactly one loader override (``checkpoints/stage1_gate4_twin/
provenance.json``: "exclude_gate4_eval=true — one override"): the shipped production scanner
trained on the full ADR-0004 D5 ``nested_train`` fold, and the GATE-4 eval twin on that fold
minus the GATE-4 eval records. So production's fold contains the twin's.
"""

from __future__ import annotations

from tbox_finder.data import window_dataset as wd

#: ``exclude_gate4_eval`` per arm — the *one* override that separates the twin from the
#: shipped scanner.
ARM_EXCLUDE_GATE4_EVAL: dict[str, bool] = {"production": False, "twin": True}


def training_fold_ids(*, arm: str, split_table: str, context: str, labels: str) -> set[str]:
    """The corpus record ids one Stage-1 arm trained on, from the shipped loader.

    Refuses an arm it does not know rather than defaulting to either fold: a typo must not
    silently select production's (larger) fold.
    """
    if arm not in ARM_EXCLUDE_GATE4_EVAL:
        raise ValueError(f"unknown Stage-1 arm {arm!r}; known: {sorted(ARM_EXCLUDE_GATE4_EVAL)}")
    records, _ = wd.load_corpus_records(
        context_parquet=context,
        labels_parquet=labels,
        split_table=split_table,
        exclude_selection_val=False,
        exclude_gate4_eval=ARM_EXCLUDE_GATE4_EVAL[arm],
    )
    return {record.record_id for record in records}
