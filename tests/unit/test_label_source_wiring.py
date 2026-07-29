"""P2-11 — the label-source WIRING: does ``label_source=naive`` reach the loaded labels?

``tests/unit/test_label_derivation.py`` proves ``labels.py`` DERIVES the withheld naive
column (``naive_label_string``, class-II CM structure removed). This file proves the
**training path actually CONSUMES it**. The gap is load-bearing and was real: P0 shipped
``naive_label_string`` into ``labels_v0.parquet`` but the datamodule hard-coded the
production ``label_string`` column, and ``train_stage1._cfg_from_mapping`` silently drops
unknown Hydra keys — so before P2-11 a ``label_source=naive`` override would have been
ignored and a "naive" run would have trained on the TBDB001.cm-derived labels it exists to
withhold, its own gate passing (CLAUDE.md §10.3; the anti-mimicry test would then measure
RF00230 mimicry, the exact confound PRD §5 mech 3 rules out).

Three guards: the selector (:func:`resolve_label_column`), the merge column-selection
(:func:`_merge_corpus_tables`), and the :func:`load_corpus_records` report keys the sbatch
asserts on before publishing the checkpoint.

Torch-free: ``window_dataset`` imports torch lazily, so the whole chain runs in the bare/data
env. The merge/loader tiers are ``pandas``+``pyarrow``-gated (``importorskip``), matching the
repo's 2-tier convention; the pure selector tier is stdlib-only.
"""

from __future__ import annotations

import pytest

from tbox_finder.data import window_dataset as wd


# --------------------------------------------------------------------------- #
# Selector (stdlib only — no pandas)
# --------------------------------------------------------------------------- #
def test_resolve_label_column_maps_each_source():
    assert wd.resolve_label_column("production") == wd.LABEL_STRING_COL
    assert wd.resolve_label_column("naive") == wd.NAIVE_LABEL_STRING_COL
    # The two MUST be different columns — a mapping that collapsed them would make the
    # ablation byte-identical to production while its report said "naive".
    assert wd.LABEL_STRING_COL != wd.NAIVE_LABEL_STRING_COL


def test_resolve_label_column_rejects_unknown():
    # A typo must fail LOUD, never fall back to the production column (that fallback is the
    # §10.3 footgun: a "naive" run silently carrying TBDB001.cm structure).
    with pytest.raises(ValueError, match="unknown label_source"):
        wd.resolve_label_column("prod")
    with pytest.raises(ValueError, match="unknown label_source"):
        wd.resolve_label_column("")


def test_config_and_selector_agree_on_the_valid_set():
    """``Stage1TrainConfig.__post_init__`` rejects the same sources the selector accepts.

    The config validates ``label_source in ("production", "naive")`` by a hardcoded tuple
    (it must not import torch-heavy ``window_dataset`` at construction), while
    ``LABEL_SOURCE_COLUMNS`` is authoritative for the column names. Assert by AST (no import
    — ``train_stage1`` pulls torch) that the two lists cannot silently drift apart.
    """
    import ast
    from pathlib import Path

    src = Path(wd.__file__).resolve().parents[1] / "train" / "train_stage1.py"
    tree = ast.parse(src.read_text())
    literals = {
        elt.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "label_source"
        for cmp in node.comparators
        if isinstance(cmp, ast.Tuple)
        for elt in cmp.elts
        if isinstance(elt, ast.Constant)
    }
    assert literals, "no `self.label_source in (...)` membership check found in __post_init__"
    assert literals == set(wd.LABEL_SOURCE_COLUMNS), (literals, set(wd.LABEL_SOURCE_COLUMNS))


# --------------------------------------------------------------------------- #
# Merge column-selection (pandas + pyarrow)
# --------------------------------------------------------------------------- #
def _write_fixture(tmp_path):
    """A minimal, valid context/labels/split trio.

    ``r1`` mimics a class-II record: production paints structure (``"SISI"``), the naive
    derivation withholds it (all-background ``"...."``). ``r2`` mimics a class-I record:
    production == naive. So a correct selector produces DIFFERENT labels for ``r1`` and
    IDENTICAL labels for ``r2``.
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    ctx = pd.DataFrame({wd.RECORD_ID_COL: ["r1", "r2"]})
    lab = pd.DataFrame(
        {
            wd.LABELS_ID_COL: ["r1", "r2"],
            wd.LABEL_STRING_COL: ["SISI", "PPPP"],
            wd.NAIVE_LABEL_STRING_COL: ["....", "PPPP"],
            wd.COGNATE_AA_COL: ["Ile", "Ala"],
        }
    )
    spl = pd.DataFrame(
        {
            wd.RECORD_ID_COL: ["r1", "r2"],
            wd.SOURCE_COL: [wd.POSITIVE_SOURCE, wd.POSITIVE_SOURCE],
            wd.PARENT_ID_COL: ["r1", "r2"],
            wd.KLASS_COL: ["II", "I"],
            wd.PHYLUM_COL: ["Actinobacteria", "Firmicutes"],
            wd.CLUSTER_COL: [1, 2],
            wd.LOO_HOLDOUT_COL: [False, False],
            "fold_random": ["train", "train"],
            "loo_order_unit": ["u1", "u2"],
            "class_holdout_unit": ["c1", "c2"],
            "phylum_holdout_unit": ["p1", "p2"],
            "nested_train": [True, True],
            "nested_role": ["train", "train"],
        }
    )
    # Guard: the split fixture must carry exactly the fold columns the merge keeps.
    assert set(wd.FOLD_SCHEME_COLUMNS) <= set(spl.columns), wd.FOLD_SCHEME_COLUMNS
    ctx_p, lab_p, spl_p = (tmp_path / n for n in ("ctx.parquet", "lab.parquet", "spl.parquet"))
    ctx.to_parquet(ctx_p)
    lab.to_parquet(lab_p)
    spl.to_parquet(spl_p)
    return ctx_p, lab_p, spl_p


def test_merge_selects_the_requested_label_column(tmp_path):
    ctx_p, lab_p, spl_p = _write_fixture(tmp_path)

    def merged_labels(label_column):
        m = wd._merge_corpus_tables(
            context_parquet=ctx_p,
            labels_parquet=lab_p,
            split_table=spl_p,
            label_column=label_column,
        )
        # Whichever source was chosen, it is normalised to LABEL_STRING_COL so _corpus_record
        # reads one column name — assert the VALUES under it are the selected source's.
        return dict(zip(m[wd.RECORD_ID_COL], m[wd.LABEL_STRING_COL], strict=True))

    prod = merged_labels(wd.LABEL_STRING_COL)
    naive = merged_labels(wd.NAIVE_LABEL_STRING_COL)
    assert prod == {"r1": "SISI", "r2": "PPPP"}
    assert naive == {"r1": "....", "r2": "PPPP"}
    # Identity, not just "something loaded": on the class-II-like record the two DIFFER, so a
    # selector that ignored label_column (a copy-of-production bug) fails here rather than
    # passing on a count ([[symmetric-count-fixture-blind-to-inversion]]).
    assert naive["r1"] != prod["r1"]
    # The default (no label_column) is production — the shipped scanner's behaviour, unchanged.
    m_default = wd._merge_corpus_tables(
        context_parquet=ctx_p, labels_parquet=lab_p, split_table=spl_p
    )
    assert (
        dict(zip(m_default[wd.RECORD_ID_COL], m_default[wd.LABEL_STRING_COL], strict=True)) == prod
    )


def test_merge_fails_closed_on_a_labels_table_missing_the_naive_column(tmp_path):
    """A stale production-only labels_v0.parquet must NOT silently un-withhold."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    ctx = pd.DataFrame({wd.RECORD_ID_COL: ["r1"]})
    lab = pd.DataFrame(
        {wd.LABELS_ID_COL: ["r1"], wd.LABEL_STRING_COL: ["SISI"], wd.COGNATE_AA_COL: ["Ile"]}
    )  # NO naive_label_string column
    spl = pd.DataFrame(
        {
            wd.RECORD_ID_COL: ["r1"],
            wd.SOURCE_COL: [wd.POSITIVE_SOURCE],
            wd.PARENT_ID_COL: ["r1"],
            wd.KLASS_COL: ["II"],
            wd.PHYLUM_COL: ["Actinobacteria"],
            wd.CLUSTER_COL: [1],
            wd.LOO_HOLDOUT_COL: [False],
            "fold_random": ["train"],
            "loo_order_unit": ["u1"],
            "class_holdout_unit": ["c1"],
            "phylum_holdout_unit": ["p1"],
            "nested_train": [True],
            "nested_role": ["train"],
        }
    )
    ctx_p, lab_p, spl_p = (tmp_path / n for n in ("ctx.parquet", "lab.parquet", "spl.parquet"))
    ctx.to_parquet(ctx_p)
    lab.to_parquet(lab_p)
    spl.to_parquet(spl_p)
    with pytest.raises(ValueError, match="naive_label_string"):
        wd._merge_corpus_tables(
            context_parquet=ctx_p,
            labels_parquet=lab_p,
            split_table=spl_p,
            label_column=wd.NAIVE_LABEL_STRING_COL,
        )


# --------------------------------------------------------------------------- #
# load_corpus_records threads + records the source (the sbatch asserts on the report keys)
# --------------------------------------------------------------------------- #
def test_load_corpus_records_threads_and_records_the_source(monkeypatch):
    """``label_source`` reaches the merge as the right column AND is recorded in the report.

    Stubs the merge (an EMPTY frame ⇒ no per-row geometry fixture needed) to isolate the
    trainer-facing wiring: resolve → pass ``label_column`` to the merge → record the resolved
    source in the report the sbatch's runtime guard reads.
    """
    pd = pytest.importorskip("pandas")
    captured: dict = {}

    def fake_merge(*, context_parquet, labels_parquet, split_table, label_column):
        captured["label_column"] = label_column
        return pd.DataFrame({wd.RECORD_ID_COL: []})

    monkeypatch.setattr(wd, "_merge_corpus_tables", fake_merge)

    records, report = wd.load_corpus_records(
        label_source="naive",
        training_fold_only=False,
        exclude_selection_val=False,
        context_parquet="unused",
        labels_parquet="unused",
        split_table="unused",
    )
    assert records == []
    assert captured["label_column"] == wd.NAIVE_LABEL_STRING_COL
    assert report["label_source"] == "naive"
    assert report["label_column"] == wd.NAIVE_LABEL_STRING_COL

    # The default is production — the shipped scanner's column, unchanged.
    _, report_prod = wd.load_corpus_records(
        training_fold_only=False,
        exclude_selection_val=False,
        context_parquet="unused",
        labels_parquet="unused",
        split_table="unused",
    )
    assert captured["label_column"] == wd.LABEL_STRING_COL
    assert report_prod["label_source"] == "production"
    assert report_prod["label_column"] == wd.LABEL_STRING_COL


def test_load_corpus_records_rejects_unknown_source():
    """An unknown source raises BEFORE any parquet is read (resolve_label_column is first)."""
    with pytest.raises(ValueError, match="unknown label_source"):
        wd.load_corpus_records(
            label_source="bogus",
            training_fold_only=False,
            exclude_selection_val=False,
        )
