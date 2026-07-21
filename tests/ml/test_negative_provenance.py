"""P2-10d′-a — the §9.2 leakage guard for **runtime-injected** negatives.

`tests/ml/test_no_leakage.py` is the CLAUDE.md §8.2 gate, and it is structurally blind to
this class of leak. That file reads the **committed per-record split-assignment table** and
checks that no cluster or taxon spans a fold boundary in it. A mined negative is injected
into the training stream at run time and appears in **no row of that table**, so every
predicate there stays green whatever the injector does. This is the same hole as
`ci-leakage-gate-blind-to-runtime-augmentation`, one substrate over.

What makes it a real leak rather than a bookkeeping nit: `mining/pool.py`'s
``genomic_window`` substrate is carved from the flank of **every anchored corpus record**,
held-out ones included, and `negatives.background_record` stamps every resulting record
``nested_train=True`` / ``is_designated_loo_holdout=False``. Those are *assertions about
the negative*, not checks on where its DNA came from. Measured on the shipped P2-10b pool:
of 46,006 natural windows, **17,074 (37.1 %)** were carved beside a designated
leave-one-order-out holdout locus — the immediate genomic neighbourhood of the loci
**GATE-4** grades, entering training stamped as training data.

The admission rule (user decision 2026-07-20) is **symmetry with the positives**: a window
is admissible iff its parent corpus record is in ``nested_train``, exactly as
``load_corpus_records(training_fold_only=True)`` requires of a positive. Nothing weaker is
defensible — ``excluded_clade_crossing`` and ``dropped`` parents are withheld *because*
their clade membership is unsafe, so readmitting their DNA under a negative label readmits
the taxon. Note the rule is strictly stronger than "exclude the LOO holdout": on the
committed table ``nested_train`` and ``is_designated_loo_holdout`` never co-occur (0 rows),
so filtering to ``nested_train`` subsumes the LOO exclusion and additionally drops the
clade-crossing and no-order records. That is the conservative direction, and it is stated
here so nobody has to re-derive it to know the filter is not merely a LOO filter.

Three tiers, mirroring `test_no_leakage.py`'s shape:

* **A** — a pure predicate over plain dicts, bare-CI green, with a clean twin and a
  deliberately-leaky twin so the predicate is proven to bite.
* **B** — the real artifacts: every window in the DVC-tracked mining pool, joined to the
  committed split table.
* **C** — the runtime hook: the records `build_stream` actually put in the stream, checked
  through `CorpusRecord.source_record_id` rather than by parsing an id string.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPLIT_TABLE = _REPO / "data/processed/splits/split_assignments.parquet"
_MINING_POOL = _REPO / "data/processed/negatives/mining_pool_v0.parquet"
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"
_WINDOW = 1024

#: The split table's own composition, so a green run cannot be a green run over nothing.
#: Sourced from `tests/ml/test_no_leakage.py`'s pins and re-verified 2026-07-20.
_EXPECTED_NESTED_TRAIN_RECORDS = 8_303
_EXPECTED_CORPUS_RECORDS = 23_535


# ═════════════════════════════════════════════════════════════════════════════════════
# Tier A — the pure predicate (stdlib only; runs everywhere, including bare CI)
# ═════════════════════════════════════════════════════════════════════════════════════
def negatives_parented_outside_training(
    parent_ids: list[str], nested_train_by_id: dict[str, bool]
) -> list[str]:
    """Parent ids that must not have supplied a training negative.

    Fail-closed on an unknown parent: ``.get(pid)`` returning ``None`` is a **violation**,
    not a pass. An id the split table cannot resolve is one whose fold nobody has checked,
    and the version of this predicate that treats it as absent is the version that reports
    a clean sweep when the join has silently stopped matching
    ([[namespace-mismatch-invisible-noop]]).
    """
    return sorted({pid for pid in parent_ids if nested_train_by_id.get(pid) is not True})


def test_the_predicate_passes_a_clean_set() -> None:
    fold = {"a": True, "b": True, "c": False}
    assert negatives_parented_outside_training(["a", "b", "a"], fold) == []


def test_the_predicate_catches_a_held_out_parent() -> None:
    """The leaky twin — without this the clean-pass test proves only that it returns [].

    `c` is a real corpus record in a held-out clade. A negative carved from its flank is
    held-out genomic context wearing a training label.
    """
    fold = {"a": True, "b": True, "c": False}
    assert negatives_parented_outside_training(["a", "c"], fold) == ["c"]


def test_the_predicate_treats_an_unresolvable_parent_as_a_violation() -> None:
    """The branch that decides whether a broken join reads as a clean corpus.

    Asserted by name because it is the *only* difference between this predicate and the
    plausible one-liner `if not fold.get(pid, True)`, and the two agree on every input
    except this one.
    """
    fold = {"a": True}
    assert negatives_parented_outside_training(["a", "ghost"], fold) == ["ghost"]


def test_the_predicate_is_not_satisfied_by_a_truthy_non_true() -> None:
    """`is not True` rather than `not ...`: a stray 1/"yes"/np.True_ must not pass as a fold.

    A pandas read can hand back `numpy.bool_`, and a hand-built dict can hand back anything.
    Only a real Python `True` is evidence; everything else is unverified.
    """
    assert negatives_parented_outside_training(["x"], {"x": 1}) == ["x"]
    assert negatives_parented_outside_training(["x"], {"x": "True"}) == ["x"]


# ═════════════════════════════════════════════════════════════════════════════════════
# Tier B/C plumbing
# ═════════════════════════════════════════════════════════════════════════════════════
def _fail_or_skip(reason: str) -> None:
    """``TBOX_REQUIRE_NEGATIVE_PROVENANCE=1`` makes an unusable input a FAILURE, not a skip.

    Same contract as `test_no_leakage.py::_fail_or_skip`. Both inputs here are DVC-tracked,
    so the default in CI is a skip; the variable is what a §8.5 manual gate sets to make the
    tier bite. A tier whose inputs are absent must never report green as though it ran.
    """
    if os.environ.get("TBOX_REQUIRE_NEGATIVE_PROVENANCE") == "1":
        pytest.fail(f"TBOX_REQUIRE_NEGATIVE_PROVENANCE=1 but an input is unusable: {reason}")
    pytest.skip(reason)


def _readable(path: Path, what: str) -> None:
    if not path.is_file():
        _fail_or_skip(f"{what} absent: {path.relative_to(_REPO)} (DVC-tracked; run `dvc pull`)")
    with path.open("rb") as handle:
        if handle.read(len(_LFS_MAGIC)) == _LFS_MAGIC:
            _fail_or_skip(
                f"{what} is an unsmudged Git-LFS pointer, not a parquet "
                f"({path.relative_to(_REPO)}) — non-empty, so a size check would pass it"
            )


@pytest.fixture(scope="module")
def nested_train_by_id() -> dict[str, bool]:
    """``{corpus record_id -> nested_train}`` read straight off the committed split table.

    Deliberately re-derived here rather than imported from `mining.pool.load_parent_folds`:
    the pool's stamped column is produced by that function, so testing the column against
    the same function would be a tautology ([[promote-dont-duplicate-is-a-correctness-rule]]).
    This is the independent second reader.
    """
    _readable(_SPLIT_TABLE, "committed split table")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(_SPLIT_TABLE, columns=["record_id", "nested_train", "source"]).to_pydict()
    fold = {
        str(rid): bool(flag)
        for rid, flag, src in zip(
            table["record_id"], table["nested_train"], table["source"], strict=True
        )
        if src == "corpus"
    }
    assert len(fold) == _EXPECTED_CORPUS_RECORDS, (
        f"split table has {len(fold)} corpus records, expected {_EXPECTED_CORPUS_RECORDS} — "
        "re-pin this number deliberately, do not relax the assertion"
    )
    assert sum(fold.values()) == _EXPECTED_NESTED_TRAIN_RECORDS
    return fold


# ═════════════════════════════════════════════════════════════════════════════════════
# Tier B — the artifact
# ═════════════════════════════════════════════════════════════════════════════════════
def test_the_pool_stamps_a_parent_fold_that_matches_the_split_table(nested_train_by_id) -> None:
    """The stamped column is *data*, and it agrees with an independent read of the table.

    The whole point of `parent_nested_train` is that admissibility does not rest on a
    promise by whoever loads the pool. That is only true if the column is right, so it is
    checked row-by-row against a second, independent read — not against the function that
    wrote it.
    """
    _readable(_MINING_POOL, "mining pool")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    pool = pd.read_parquet(
        _MINING_POOL, columns=["source_record_id", "parent_nested_train", "is_designed_control"]
    )
    assert len(pool) > 0
    parents = [str(p) for p in pool["source_record_id"]]
    unresolved = [p for p in set(parents) if p not in nested_train_by_id]
    assert unresolved == [], (
        f"{len(unresolved)} of {len(set(parents))} distinct parents do not resolve against the "
        "split table — the join is broken, not the corpus"
    )
    expected = [nested_train_by_id[p] for p in parents]
    stamped = [bool(v) for v in pool["parent_nested_train"]]
    assert stamped == expected

    # Non-vacuity, both directions: the stamp must discriminate. All-True would mean the
    # join matched nothing and defaulted; all-False would mean it matched the wrong
    # namespace. Neither can be distinguished from a working filter by the equality above
    # alone, because a broken second reader would agree with a broken column.
    assert 0 < sum(stamped) < len(stamped)
    assert len(set(parents)) > 5_000


def test_no_admissible_window_was_carved_beside_a_held_out_locus(nested_train_by_id) -> None:
    """Tier B's headline: the admissible slice of the artifact is training-fold DNA.

    Run over the pool's *natural* windows only — designed controls are locus-centred, exist
    to prove the coordinate frame, and are refused at load by `REASON_DESIGNED_CONTROL`
    before the fold rule is ever consulted.
    """
    _readable(_MINING_POOL, "mining pool")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    pool = pd.read_parquet(
        _MINING_POOL,
        columns=["source_record_id", "parent_nested_train", "is_designed_control", "masked"],
    )
    natural = pool[~pool["is_designed_control"].astype(bool)]
    admissible = natural[
        natural["parent_nested_train"].astype(bool) & ~natural["masked"].astype(bool)
    ]
    assert len(admissible) > 0, "no admissible windows at all — the substrate, not the filter"
    assert (
        negatives_parented_outside_training(
            [str(p) for p in admissible["source_record_id"]], nested_train_by_id
        )
        == []
    )

    # And the filter is not a no-op: the pool genuinely contains windows it must refuse.
    # Gated on a number whose legitimate value can never be zero — the corpus holds ~15.2k
    # anchored non-`nested_train` records and windows are carved from every anchored record,
    # so zero refusals means the join stopped working, not that the corpus is clean.
    refused = natural[~natural["parent_nested_train"].astype(bool)]
    assert len(refused) > 1_000, (
        f"only {len(refused)} out-of-fold natural windows in a pool of {len(natural)} — "
        "a filter that removes nothing is broken, not clean"
    )


# ═════════════════════════════════════════════════════════════════════════════════════
# Tier C — the runtime stream (the tier no committed-table gate can reach)
# ═════════════════════════════════════════════════════════════════════════════════════
def test_every_negative_build_stream_injects_is_training_fold_dna(nested_train_by_id) -> None:
    """The invariant, checked on the records that actually reach the model.

    Tier B grades the artifact; this grades the *stream*, which is what training consumes
    and what no committed-table predicate can see. The parent is read from
    `CorpusRecord.source_record_id` — a required field carried through from the pool row —
    rather than parsed out of `record_id`. That matters: `record_id` is
    ``f"neg:{candidate_id}"`` and `candidate_id` is ``f"{parent}:{side}"``, so a string
    parse works today and mis-attributes silently the first time either format moves.

    `is_negative_record` keys on ``cluster_id < 0`` (the enforced private namespace), not on
    the cosmetic ``neg:`` prefix, so a caller-chosen id cannot smuggle a record past this.
    """
    _readable(_MINING_POOL, "mining pool")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from tbox_finder.data.negatives import is_negative_record
    from tbox_finder.train.train_stage1 import Stage1TrainConfig, build_stream

    dataset, _sampler, scope = build_stream(
        Stage1TrainConfig(
            max_records=100,
            eval_val=False,
            exclude_selection_val=True,
            negative_fraction=10 / 11,
            negative_max_records=200,
            negative_pool_parquet=str(_MINING_POOL),
        )
    )
    injected = [r for r in dataset.records if is_negative_record(r)]
    assert len(injected) == scope["n_negative_records"] == 200

    offenders = negatives_parented_outside_training(
        [r.source_record_id for r in injected], nested_train_by_id
    )
    assert offenders == [], (
        f"{len(offenders)} injected negatives were carved beside non-nested_train loci; "
        f"first few: {offenders[:5]}"
    )

    # The field is real provenance, not a restatement of the record's own id — the exact
    # thing a well-meaning "fix" (defaulting source_record_id to record_id) would break,
    # and it would break it while leaving the assertion above green.
    assert all(r.source_record_id != r.record_id for r in injected)
    # …and it is the parent the pool named, not a re-derivation: every injected record's id
    # is `neg:<parent>:<side>`, so the field and the id must agree on the parent. If they
    # ever disagree, one of the two is being synthesised.
    assert all(
        r.record_id.split(":", 1)[1].rsplit(":", 1)[0] == r.source_record_id for r in injected
    )


def test_a_positive_is_its_own_provenance() -> None:
    """The other half of the field's contract, so it cannot rot into a negatives-only knob.

    A positive's window is carved from the context fetched for that very locus, so the fold
    the record carries and the fold of its DNA are the same fold — and stating that as data
    is what lets a single predicate grade positives and negatives alike.
    """
    _readable(_SPLIT_TABLE, "committed split table")
    _readable(_REPO / "data/interim/flank_context/context_v0.parquet", "flank context")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from tbox_finder.data.window_dataset import load_corpus_records

    records, _report = load_corpus_records(training_fold_only=True)
    assert len(records) > 1_000
    assert all(r.source_record_id == r.record_id for r in records)
    assert all(r.nested_train for r in records)
