"""The P2-10d mined-negative injection hook: geometry, namespace, and a MEASURED mix.

Four properties this file exists to pin, each one a defect that has already shipped in
this repo in some other costume:

* **Geometry is a refusal, not a repair.** A negative that is not exactly one window of
  real DNA must raise, because the only ways to make it fit invent a contig boundary or
  emit an unnamed sub-span. Tested by construction *and* by sabotage.
* **The private stratum namespace preserves positive weights exactly.** Landing negatives
  in a shared bucket would move real positives' curriculum weights, silently.
* **The mix ratio is counted off the emitted stream.** A ratio read from the pool size or
  echoed from the config is the recurring `basis_point` defect; here the gate re-derives an
  exact integer identity from the counted keys, and the tests below check both that it
  holds when it should and that it FAILS when the sampler and the request disagree.
* **A negative's DNA must come from the §9.2 training fold (P2-10d′-a).**
  ``background_record`` *asserts* ``nested_train=True`` about the record it builds; the
  only thing that can check that against where the DNA actually came from is the parent
  locus the window was carved beside, so ``source_record_id`` is required and
  ``parent_nested_train`` gates admission. The CI §8.2 no-leakage gate is structurally
  blind here — it reads the committed per-record split table, in which a runtime-injected
  negative has no row ([[ci-leakage-gate-blind-to-runtime-augmentation]]) — so the refusal
  ladder below is the only place the rule is enforced, and the tests keep
  "parent out of fold" and "parent unresolved" apart so a join that resolves *nothing*
  cannot read as a filter that found nothing to refuse
  ([[namespace-mismatch-invisible-noop]]).

Bare-CI: numpy only (the module's pandas paths are exercised in
``tests/ml/test_negative_injection.py``, which needs the real parquet).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from tbox_finder.data import negatives as neg
from tbox_finder.data.window_dataset import (
    BACKGROUND_INDEX,
    IGNORE_INDEX,
    PAD_TOKEN_ID,
    CorpusRecord,
    Stage1DataConfig,
    Stage1WindowDataset,
    curriculum_weights,
)

WINDOW = 64


def _seq(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _positive(
    i: int, *, phylum: str = "Firmicutes", klass: str = "I", aa: str = "ILE"
) -> CorpusRecord:
    """A hand-checkable positive: 30-nt locus centred in 3x the window of real context."""
    return CorpusRecord(
        record_id=f"pos{i}",
        context_seq=_seq(3 * WINDOW, i),
        locus_offset=WINDOW,
        locus_length=30,
        label_string="1" * 30,
        clipped_start=False,
        clipped_end=False,
        klass=klass,
        phylum=phylum,
        cognate_aa=aa,
        cluster_id=i,
        nested_train=True,
        is_designated_loo_holdout=False,
        folds=("train", None, None, None, True, "train"),
        # A positive's DNA is its own genomic context, so the parent is itself. That
        # identity is exactly what makes it the WRONG default for a negative — see
        # `test_an_emitted_negative_names_its_parent_not_itself`.
        source_record_id=f"pos{i}",
    )


def _negative(j: int) -> CorpusRecord:
    """A negative carved beside positive ``pos{j}`` — a parent that is not its own id."""
    return neg.background_record(
        record_id=f"neg{j}",
        sequence=_seq(WINDOW, 10_000 + j),
        cluster_id=-(j + 1),
        source_record_id=f"pos{j}",
        window=WINDOW,
    )


def _row(candidate_id: str, sequence: str, **overrides: Any) -> dict[str, Any]:
    """One mined-pool row that clears **every** guard, so a test can break exactly one.

    ``source_record_id`` + ``parent_nested_train`` are P2-10d′-a's admission columns. A
    row that omits them is refused as ``parent_fold_unknown`` *before* any width or mask
    guard is reached, which would silently turn every refusal test in this file into a
    test of the fold rule — the fixtures would still be green, and would no longer test
    what their names say.
    """
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "sequence": sequence,
        "masked": False,
        "is_designed_control": False,
        "source_record_id": f"parent:{candidate_id}",
        "parent_nested_train": True,
    }
    row.update(overrides)
    return row


def _dataset(n_pos: int, n_neg: int) -> Stage1WindowDataset:
    return Stage1WindowDataset(
        [*(_positive(i) for i in range(n_pos)), *(_negative(j) for j in range(n_neg))],
        config=Stage1DataConfig(window_nt=WINDOW, stride_nt=WINDOW // 2),
    )


# ── Geometry ─────────────────────────────────────────────────────────────────────────
def test_background_record_window_is_all_real_all_background() -> None:
    """The whole point: every position real DNA, every label background, nothing ignored.

    Checked on the emitted **window**, not on the record's fields — a record whose
    `label_string` is all-`.` but whose carve padded would still fail here.
    """
    ds = _dataset(1, 1)
    w = ds.window_at(1)
    assert w.labels.shape == (WINDOW,)
    assert set(w.labels.tolist()) == {BACKGROUND_INDEX}
    assert IGNORE_INDEX not in w.labels.tolist()
    assert bool(w.real_mask.all())
    assert PAD_TOKEN_ID not in w.input_ids.tolist()
    assert (w.pad_left, w.pad_right) == (0, 0)
    assert not w.zero_flanked


def test_background_record_refuses_a_short_sequence() -> None:
    """Padding a short negative would assert a contig boundary that does not exist."""
    with pytest.raises(neg.NegativeInjectionError, match="exactly one window of real DNA"):
        neg.background_record(
            record_id="short",
            sequence=_seq(WINDOW - 1, 1),
            cluster_id=-1,
            source_record_id="pos0",
            window=WINDOW,
        )


def test_background_record_refuses_a_long_sequence() -> None:
    """A longer sequence would emit an unnamed sub-span under the record's whole-region id."""
    with pytest.raises(neg.NegativeInjectionError, match="exactly one window of real DNA"):
        neg.background_record(
            record_id="long",
            sequence=_seq(WINDOW + 1, 1),
            cluster_id=-1,
            source_record_id="pos0",
            window=WINDOW,
        )


def test_background_record_refuses_non_acgtn() -> None:
    """`encode_bases` maps anything unknown to the N id **silently** — refuse instead.

    IUPAC ambiguity codes and alignment gaps are the realistic sources: a window of `R`s
    would encode as a run of Ns and train as one, without a word.
    """
    for bad in ("R", "-", "."):
        with pytest.raises(neg.NegativeInjectionError, match="non-ACGTN"):
            neg.background_record(
                record_id="amb",
                sequence=_seq(WINDOW - 1, 1) + bad,
                cluster_id=-1,
                source_record_id="pos0",
                window=WINDOW,
            )


def test_soft_masked_sequence_is_upper_cased_not_refused() -> None:
    """The one deliberate normalisation: soft-masking is an annotation, not a base.

    Refusing it would reject legitimate genomic FASTA; encoding it unchanged would train
    1,024 Ns. Upper-casing is lossless, and this test pins that the emitted window is the
    same one the upper-case sequence gives — so the normalisation cannot quietly become
    something else.
    """
    seq = _seq(WINDOW, 11)
    soft = neg.background_record(
        record_id="soft",
        sequence=seq.lower(),
        cluster_id=-1,
        source_record_id="pos0",
        window=WINDOW,
    )
    hard = neg.background_record(
        record_id="soft", sequence=seq, cluster_id=-1, source_record_id="pos0", window=WINDOW
    )
    assert soft.context_seq == hard.context_seq == seq


def test_background_record_refuses_a_positive_cluster_namespace() -> None:
    """A non-negative cluster id can be swept into the P2-06a selection-val carve."""
    for bad in (0, 1, 4775):
        with pytest.raises(neg.NegativeInjectionError, match="private namespace"):
            neg.background_record(
                record_id="n",
                sequence=_seq(WINDOW, 1),
                cluster_id=bad,
                source_record_id="pos0",
                window=WINDOW,
            )


def test_is_negative_record_keys_on_the_cluster_not_the_id() -> None:
    """The id prefix is cosmetic; the cluster namespace is the enforced property."""
    n = neg.background_record(
        record_id="not_prefixed_at_all",
        sequence=_seq(WINDOW, 3),
        cluster_id=-9,
        source_record_id="pos0",
        window=WINDOW,
    )
    assert neg.is_negative_record(n)
    assert not neg.is_negative_record(_positive(0))


# ── The private namespace preserves positive weights exactly ─────────────────────────
def test_injecting_negatives_leaves_positive_curriculum_weights_unchanged() -> None:
    """Relative positive weights must be unchanged with and without the negatives.

    The three inverse-frequency terms are `Counter`-based over one global pool, so a
    shared stratum key would change the counts; with a fresh key every normalisation is a
    common scalar and the *ratios* survive. `rtol=1e-12` is floating-point round-off from a
    different multiplication order, not a slack allowance — the shared-bucket control below
    moves the same ratios by ~1e-2, ten orders of magnitude larger, which is what makes
    this bound meaningful rather than merely satisfiable.
    """
    positives = [
        _positive(
            i,
            phylum=("Firmicutes" if i % 4 else "Actinobacteriota"),
            klass=("I" if i % 7 else "II"),
        )
        for i in range(40)
    ]
    negs = [_negative(j) for j in range(25)]
    kw = {"phylum_alpha": 0.25, "klass_alpha": 0.25, "aa_alpha": 0.25}

    def weights(records):
        return curriculum_weights(
            phyla=[r.phylum for r in records],
            klasses=[r.klass for r in records],
            amino_acids=[r.cognate_aa for r in records],
            **kw,
        )

    w_pos = weights(positives)
    w_all = weights([*positives, *negs])
    np.testing.assert_allclose(
        w_all[: len(positives)] / w_all[0], w_pos / w_pos[0], rtol=1e-12, atol=0
    )


def test_the_unknown_stratum_would_have_perturbed_them_the_control() -> None:
    """The designed control for the test above: the shared-bucket variant DOES perturb.

    Without this, "unchanged" could be a property of `curriculum_weights` rather than of
    the private namespace — a passing test that measures nothing. `UNKNOWN` is the exact
    bucket a `None`/blank stratum would land in, and it is already populated by real
    positives whose lineage could not be resolved.
    """
    positives = [_positive(i, phylum=("UNKNOWN" if i < 5 else "Firmicutes")) for i in range(40)]
    kw = {"phylum_alpha": 0.25, "klass_alpha": 0.25, "aa_alpha": 0.25}

    def weights(records):
        return curriculum_weights(
            phyla=[r.phylum for r in records],
            klasses=[r.klass for r in records],
            amino_acids=[r.cognate_aa for r in records],
            **kw,
        )

    w_pos = weights(positives)
    # Negatives that (wrongly) share the UNKNOWN phylum bucket.
    shared = [
        CorpusRecord(
            **{
                **_negative(j).__dict__,
                "phylum": "UNKNOWN",
                "klass": "I",
                "cognate_aa": "ILE",
            }
        )
        for j in range(25)
    ]
    w_all = weights([*positives, *shared])
    deviation = np.max(np.abs(w_all[: len(positives)] / w_all[0] - w_pos / w_pos[0]))
    # Not just "different": bigger than the 1e-12 bound the real test asserts, by orders
    # of magnitude. A control that only failed by 1e-11 would make that bound arbitrary.
    assert deviation > 1e-3, deviation


# ── The mix ──────────────────────────────────────────────────────────────────────────
def test_negative_draw_count_solves_the_share_not_a_pool_ratio() -> None:
    """`f` is the share of the emitted stream: PRD §9.1's ~10:1 is f = 10/11."""
    assert neg.negative_draw_count(100, 0.0) == 0
    assert neg.negative_draw_count(100, 0.5) == 100
    assert neg.negative_draw_count(100, 10 / 11) == 1000
    with pytest.raises(neg.NegativeInjectionError):
        neg.negative_draw_count(100, 1.0)
    with pytest.raises(neg.NegativeInjectionError):
        neg.negative_draw_count(100, -0.1)
    # Bools are not fractions — isinstance(True, int) is True.
    with pytest.raises(neg.NegativeInjectionError):
        neg.negative_draw_count(100, True)


@pytest.mark.parametrize("fraction", [0.1, 0.5, 10 / 11, 0.95])
def test_realized_mix_equals_the_request_measured_off_the_emitted_keys(fraction: float) -> None:
    """The gate's identity, on the counted stream — no tolerance, exact integers."""
    ds = _dataset(40, 12)
    mixer = neg.MixedIndexSampler(
        neg.positive_only_sampler(ds, n_positive=40),
        n_positive_records=40,
        n_negative_records=12,
        negative_fraction=fraction,
        seed=7,
    )
    keys = list(mixer)
    summary = mixer.mix_summary(keys)
    assert summary["n_total"] == len(keys)
    assert summary["n_negative"] == neg.negative_draw_count(40, fraction)
    assert all(neg.mix_clauses(summary, negative_fraction=fraction).values())
    # And the classification is real: every key counted negative indexes a negative record.
    negative_keys = [k for k in keys if k[0] >= 40]
    assert len(negative_keys) == summary["n_negative"]
    assert all(neg.is_negative_record(ds.records[k[0]]) for k in negative_keys)


def test_a_mix_that_does_not_realize_the_request_fails_the_clause() -> None:
    """The sabotage: a summary whose counts disagree with the request must NOT pass.

    Named explicitly because a green suite under sabotage proves *some* test bit, not this
    one — this asserts on `negative_mix_matches_request` by name.
    """
    honest = {
        "n_total": 110,
        "n_negative": 10,
        "n_positive": 100,
        "negative_pool_size": 5,
        "realized_negative_fraction": 10 / 110,
        "requested_negative_fraction": 0.5,
    }
    assert neg.mix_clauses(honest, negative_fraction=10 / 110)["negative_mix_matches_request"]
    # Same evidence, a different request: the identity must break.
    assert not neg.mix_clauses(honest, negative_fraction=0.5)["negative_mix_matches_request"]


def test_mix_clauses_are_false_on_missing_or_malformed_evidence() -> None:
    """A clause derived from a *missing* measurement must be FALSE, never vacuously TRUE."""
    for bad in (
        None,
        {},
        {"n_total": 10},
        {"n_total": 10, "n_negative": 3, "n_positive": 8, "negative_pool_size": 1},
    ):
        clauses = neg.mix_clauses(bad, negative_fraction=0.5)
        assert not all(clauses.values()), bad
    # Bools masquerading as counts.
    assert not all(
        neg.mix_clauses(
            {"n_total": True, "n_negative": True, "n_positive": False, "negative_pool_size": 0},
            negative_fraction=0.0,
        ).values()
    )


def test_requested_zero_asserts_an_empty_pool_rather_than_skipping() -> None:
    """f = 0 is an assertion: no pool AND no negative draws, both recorded."""
    zero = {"n_total": 40, "n_negative": 0, "n_positive": 40, "negative_pool_size": 0}
    assert all(neg.mix_clauses(zero, negative_fraction=0.0).values())
    loaded_but_unsampled = {**zero, "negative_pool_size": 12}
    assert not neg.mix_clauses(loaded_but_unsampled, negative_fraction=0.0)[
        "negative_mix_pool_consistent"
    ]
    sampled_from_nothing = {
        "n_total": 44,
        "n_negative": 4,
        "n_positive": 40,
        "negative_pool_size": 0,
    }
    assert not neg.mix_clauses(sampled_from_nothing, negative_fraction=0.1)[
        "negative_mix_pool_consistent"
    ]


def test_zero_fraction_stream_is_bit_identical_to_the_bare_sampler() -> None:
    """Routing every run through the mixer must not perturb the positives-only stream.

    P2-04's smoke report, P2-06's sweep ranking and P2-09's production checkpoint were all
    produced on the bare `WeightedIndexSampler`. If the mixer changed the draw order at
    f = 0, every one of those becomes irreproducible for a reason no config records.
    """
    ds = _dataset(40, 0)
    bare = neg.positive_only_sampler(ds, n_positive=40)
    mixer = neg.MixedIndexSampler(
        neg.positive_only_sampler(ds, n_positive=40),
        n_positive_records=40,
        n_negative_records=0,
        negative_fraction=0.0,
        seed=7,
    )
    for epoch in (0, 1, 5):
        bare.set_epoch(epoch)
        mixer.set_epoch(epoch)
        assert list(mixer) == list(bare)


def test_mix_is_reproducible_and_advances_with_the_epoch() -> None:
    ds = _dataset(40, 12)

    def build():
        return neg.MixedIndexSampler(
            neg.positive_only_sampler(ds, n_positive=40),
            n_positive_records=40,
            n_negative_records=12,
            negative_fraction=0.5,
            seed=7,
        )

    a, b = build(), build()
    assert list(a) == list(b)
    a.set_epoch(1)
    assert list(a) != list(b)
    # ...but still the same composition: the identity is epoch-invariant.
    assert a.mix_summary()["n_negative"] == b.mix_summary()["n_negative"]


def test_negatives_are_spread_across_the_epoch_not_appended_as_a_tail() -> None:
    """A tail block would hand whole DDP ranks a single-class stream.

    Measured as: the first and second halves of the emitted stream each carry a
    non-trivial share of negatives. A concatenation-without-shuffle would put 0 in the
    first half.
    """
    ds = _dataset(40, 12)
    mixer = neg.MixedIndexSampler(
        neg.positive_only_sampler(ds, n_positive=40),
        n_positive_records=40,
        n_negative_records=12,
        negative_fraction=0.5,
        seed=7,
    )
    keys = list(mixer)
    half = len(keys) // 2
    first = sum(1 for k in keys[:half] if k[0] >= 40)
    second = sum(1 for k in keys[half:] if k[0] >= 40)
    assert first > 0 and second > 0
    assert abs(first - second) < half // 2


def test_repeated_negative_draws_get_independent_strand_draws() -> None:
    """`occurrence` must vary per draw, or an oversampled negative arrives as copies.

    Same defect P2-01 measured for oversampled class-II positives; a 40:12 mix draws each
    negative several times per epoch.
    """
    ds = _dataset(4, 2)
    mixer = neg.MixedIndexSampler(
        neg.positive_only_sampler(ds, n_positive=4),
        n_positive_records=4,
        n_negative_records=2,
        negative_fraction=0.9,
        seed=3,
    )
    negative_keys = [k for k in mixer if k[0] >= 4]
    assert len(negative_keys) > 4
    for index in {k[0] for k in negative_keys}:
        occurrences = [occ for i, occ in negative_keys if i == index]
        assert len(set(occurrences)) == len(occurrences)
    emitted = {(k, ds[k]["record_id"]) for k in negative_keys}
    assert len({rid for _, rid in emitted}) > 1  # not all the same variant


def test_an_empty_pool_with_a_positive_fraction_is_refused() -> None:
    """Degrading to positives-only while reporting a mix is the failure mode to avoid."""
    ds = _dataset(10, 0)
    with pytest.raises(neg.NegativeInjectionError, match="pool is EMPTY"):
        neg.MixedIndexSampler(
            neg.positive_only_sampler(ds, n_positive=10),
            n_positive_records=10,
            n_negative_records=0,
            negative_fraction=0.5,
            seed=1,
        )


# ── The positive/negative boundary is verified, not assumed ──────────────────────────
def test_positive_only_sampler_rejects_a_misplaced_boundary() -> None:
    """An off-by-one would weight a negative by a phylum it does not have, silently."""
    ds = _dataset(10, 3)
    with pytest.raises(neg.NegativeInjectionError, match="boundary is not at n_positive"):
        neg.positive_only_sampler(ds, n_positive=9)  # one negative below the line
    with pytest.raises(neg.NegativeInjectionError, match="boundary is not at n_positive"):
        neg.positive_only_sampler(ds, n_positive=11)  # one positive above it


def test_positive_only_sampler_never_draws_a_negative_index() -> None:
    """The structural guarantee: negatives are outside the curriculum's index space."""
    ds = _dataset(20, 8)
    sampler = neg.positive_only_sampler(ds, n_positive=20)
    assert max(sampler.indices()) < 20


# ── Row ingestion counts every refusal ───────────────────────────────────────────────
def test_rows_report_counts_every_refusal_by_reason() -> None:
    rows = [
        _row("ok1", _seq(WINDOW, 1)),
        _row("short", _seq(WINDOW - 5, 2)),
        _row("long", _seq(WINDOW + 5, 3)),
        _row("masked", _seq(WINDOW, 4), masked=True),
        _row("ctl", _seq(WINDOW, 5), is_designed_control=True),
        _row("", _seq(WINDOW, 6)),
        _row("iupac", "R" * WINDOW),
    ]
    records, report = neg.negative_records_from_rows(rows, window=WINDOW)
    assert [r.record_id for r in records] == ["neg:ok1"]
    assert report["excluded_by_reason"] == {
        neg.REASON_DESIGNED_CONTROL: 1,
        neg.REASON_EMPTY_ID: 1,
        neg.REASON_MASKED: 1,
        neg.REASON_NON_ACGTN: 1,
        neg.REASON_TOO_LONG: 1,
        neg.REASON_TOO_SHORT: 1,
    }
    assert report["n_rows_read"] == len(rows)
    assert report["n_records"] + report["n_excluded"] == report["n_rows_read"]
    assert report["n_rows_at_window_length"] == 5


def test_designed_controls_are_dropped_by_default() -> None:
    """P2-10b's locus controls overlap a known T-box by construction.

    Training on one is direct label poisoning: an all-background label over a real T-box.
    """
    rows = [_row(f"c{i}", _seq(WINDOW, i), is_designed_control=True) for i in range(5)]
    records, report = neg.negative_records_from_rows(rows, window=WINDOW)
    assert records == []
    assert report["excluded_by_reason"][neg.REASON_DESIGNED_CONTROL] == 5


def test_duplicate_candidate_ids_are_refused() -> None:
    """A duplicated hard negative is silently oversampled — the mix would misdescribe it."""
    rows = [_row("dup", _seq(WINDOW, 1)), _row("dup", _seq(WINDOW, 2))]
    with pytest.raises(neg.NegativeInjectionError, match="duplicate candidate_id"):
        neg.negative_records_from_rows(rows, window=WINDOW)


def test_negative_cluster_ids_are_unique_and_disjoint_from_positives() -> None:
    rows = [_row(f"c{i}", _seq(WINDOW, i)) for i in range(6)]
    records, _ = neg.negative_records_from_rows(rows, window=WINDOW)
    ids = [r.cluster_id for r in records]
    assert len(set(ids)) == len(ids)
    assert all(i < 0 for i in ids)


def test_negative_folds_match_the_split_scheme_arity() -> None:
    """`provenance_rows` zips folds with FOLD_SCHEME_COLUMNS under `strict=True`."""
    from tbox_finder.data.window_dataset import FOLD_SCHEME_COLUMNS

    assert len(neg.NEGATIVE_FOLDS) == len(FOLD_SCHEME_COLUMNS)
    ds = _dataset(2, 2)
    rows = ds.provenance_rows()  # raises on an arity mismatch
    assert len(rows) == 4


# ── P2-10d′-a: the parent-fold admission rule ────────────────────────────────────────
def test_a_row_whose_parent_is_out_of_fold_is_refused_and_never_emitted() -> None:
    """Held-out genomic context must not enter the training stream under a negative label.

    `genomic_window` carves flank from *every* anchored corpus record, held-out ones
    included, and `background_record` stamps `nested_train=True` on whatever it is handed.
    Measured on the P2-10b pool, 37.1 % of natural windows sit beside a designated
    leave-one-order-out locus — i.e. the immediate neighbourhood of the loci GATE-4
    grades. The refused row must be **absent from `records`**, not merely counted: a
    report that names the refusal while the record still ships is the worst of both.
    """
    rows = [
        _row("in_fold", _seq(WINDOW, 1)),
        _row("held_out", _seq(WINDOW, 2), parent_nested_train=False),
    ]
    records, report = neg.negative_records_from_rows(rows, window=WINDOW)
    assert [r.record_id for r in records] == ["neg:in_fold"]
    assert "neg:held_out" not in {r.record_id for r in records}
    assert report["excluded_by_reason"][neg.REASON_PARENT_OUT_OF_FOLD] == 1
    assert report["n_refused_parent_out_of_fold"] == 1
    # The in-fold row is the companion control: it proves the rule refuses *this* row for
    # its fold, not every row for some unrelated reason.
    assert report["n_records"] == 1
    assert report["n_refused_parent_unresolved"] == 0


def test_an_unresolved_parent_fold_is_a_distinct_reason_from_out_of_fold() -> None:
    """A join that resolves NOTHING must not read as a filter that found nothing to refuse.

    Both a null `parent_nested_train` cell and a wholly absent column mean "the parent's
    fold is unknown", which is a broken join — the P2-10b namespace mismatch in its
    training-side costume ([[namespace-mismatch-invisible-noop]]). Folding it into
    `parent_not_nested_train` would make a 0-key stamp indistinguishable from a pool
    every one of whose parents is genuinely held out, and only the second is a data fact.
    """
    null_cell = _row("null_fold", _seq(WINDOW, 1), parent_nested_train=None)
    absent_column = _row("no_fold_col", _seq(WINDOW, 2))
    absent_column.pop("parent_nested_train")
    # …and the pandas-3 spelling of the same null. `fold is None` does NOT catch NaN, and
    # `bool(NaN)` is **True**, so the guard used to read a null fold as "in fold" — a
    # fail-open on the §9.2 rule, latent only because the shipped pool has no nulls in this
    # column. Added at P2-10d′-c with the `is_missing` fix ([[row_text]]).
    nan_cell = _row("nan_fold", _seq(WINDOW, 3), parent_nested_train=float("nan"))
    for row in (null_cell, absent_column, nan_cell):
        records, report = neg.negative_records_from_rows([row], window=WINDOW)
        assert records == [], row["candidate_id"]
        assert report["excluded_by_reason"] == {neg.REASON_PARENT_UNRESOLVED: 1}
        assert report["n_refused_parent_unresolved"] == 1
        # The distinction under test, asserted in both directions.
        assert report["n_refused_parent_out_of_fold"] == 0
        assert neg.REASON_PARENT_OUT_OF_FOLD not in report["excluded_by_reason"]
    # The companion control that must fire by construction: an explicitly False fold on an
    # otherwise identical row lands under the OTHER reason. Without it, "unresolved"
    # passing could just mean the ladder refuses everything.
    _records, out = neg.negative_records_from_rows(
        [_row("null_fold", _seq(WINDOW, 1), parent_nested_train=False)], window=WINDOW
    )
    assert out["excluded_by_reason"] == {neg.REASON_PARENT_OUT_OF_FOLD: 1}


def test_a_nan_source_record_id_is_unresolved_not_a_parent_named_nan() -> None:
    """A NaN parent id must read as ABSENT, never as the literal id "nan".

    Same defect as the decoy side (P2-10d′-c), in the direction that matters here: under
    `str(x or "")` a pandas-3 null yields the truthy string "nan", which passes the
    `if not parent` guard, so a row that cannot name its parent would be admitted with an
    unfalsifiable §9.2 provenance instead of refused.
    """
    row = _row("nan_parent", _seq(WINDOW, 4), parent_nested_train=True)
    row["source_record_id"] = float("nan")
    records, report = neg.negative_records_from_rows([row], window=WINDOW)
    assert records == []
    assert report["n_refused_parent_unresolved"] == 1
    assert report["excluded_by_reason"] == {neg.REASON_PARENT_UNRESOLVED: 1}


def test_a_blank_source_record_id_is_unresolved_even_with_the_rule_disarmed() -> None:
    """A record that cannot name its origin is never constructible, flag or no flag.

    `require_parent_nested_train=False` relaxes *which* parents are admissible; it cannot
    relax the requirement that a parent be named, because the name is the only thing that
    makes the §9.2 provenance claim falsifiable at all. A blank that slipped through with
    the rule disarmed would produce a negative whose origin no later audit can recover.
    """
    for blank in ("", "   ", None):
        for require in (True, False):
            row = _row("nameless", _seq(WINDOW, 1), source_record_id=blank)
            records, report = neg.negative_records_from_rows(
                [row], window=WINDOW, require_parent_nested_train=require
            )
            assert records == [], (blank, require)
            assert report["excluded_by_reason"] == {neg.REASON_PARENT_UNRESOLVED: 1}, (
                blank,
                require,
            )
            assert report["n_refused_parent_unresolved"] == 1
    # Control: the same row WITH a parent is emitted under both settings, so the refusals
    # above are about the blank id and not about the flag or the fixture.
    for require in (True, False):
        records, _report = neg.negative_records_from_rows(
            [_row("nameless", _seq(WINDOW, 1))],
            window=WINDOW,
            require_parent_nested_train=require,
        )
        assert [r.record_id for r in records] == ["neg:nameless"], require


def test_disarming_the_fold_rule_admits_the_row_and_says_so_in_the_report() -> None:
    """The escape hatch must be visible in the artifact, never silent.

    A reader of a report holding `require_parent_nested_train: false` must be able to see
    that the §9.2 admission rule was off; inferring it from the *absence* of a
    `parent_not_nested_train` key is exactly the ambiguity between "refused nothing" and
    "never armed" that [[clauses-must-guard-emptiness]] names.
    """
    row = _row("held_out", _seq(WINDOW, 1), parent_nested_train=False)
    records, report = neg.negative_records_from_rows(
        [row], window=WINDOW, require_parent_nested_train=False
    )
    assert [r.record_id for r in records] == ["neg:held_out"]
    assert report["require_parent_nested_train"] is False
    assert report["n_refused_parent_out_of_fold"] == 0
    # The armed control on the identical row: refused, and the report says the rule was on.
    armed_records, armed = neg.negative_records_from_rows([row], window=WINDOW)
    assert armed_records == []
    assert armed["require_parent_nested_train"] is True
    assert armed["n_refused_parent_out_of_fold"] == 1


def test_background_record_refuses_a_blank_source_record_id() -> None:
    """Blank would make the §9.2 provenance check unfalsifiable — refuse at construction.

    Whitespace is included on purpose: a stripped-to-empty parent id is the shape a
    parquet round-trip or a hand-edited fixture produces, and `str.strip()` is the only
    thing between it and a record that claims a parent it does not have.
    """
    blank_message = "source_record_id must be a non-empty"
    for blank in ("", " ", "\t\n  "):
        with pytest.raises(neg.NegativeInjectionError, match=blank_message):
            neg.background_record(
                record_id="n",
                sequence=_seq(WINDOW, 1),
                cluster_id=-1,
                source_record_id=blank,
                window=WINDOW,
            )
    # Control: the same call with a real parent builds, so the refusals above are about
    # the blank and not about some other argument the constructor also rejects.
    built = neg.background_record(
        record_id="n",
        sequence=_seq(WINDOW, 1),
        cluster_id=-1,
        source_record_id="pos0",
        window=WINDOW,
    )
    assert built.source_record_id == "pos0"


def test_an_emitted_negative_names_its_parent_not_itself() -> None:
    """The field must carry the PARENT's id, not a default copied from `record_id`.

    `record_id` is `f"neg:{candidate_id}"` and the parent is a corpus record id — two
    different namespaces. Defaulting `source_record_id` to `record_id` is the plausible
    "fix" for a constructor that suddenly requires an argument, and it is right for
    positives (whose DNA is their own context) and silently wrong for exactly the records
    the rule exists for: the fold lookup would then join on a record that is in no split
    table row, and every window would resolve to unknown or, worse, be stamped from a
    negative's own fabricated fold.
    """
    rows = [_row("c0", _seq(WINDOW, 1), source_record_id="RF00230_master.fa:0042")]
    records, _report = neg.negative_records_from_rows(rows, window=WINDOW)
    (record,) = records
    assert record.record_id == "neg:c0"
    assert record.source_record_id == "RF00230_master.fa:0042"
    assert record.source_record_id != record.record_id
    assert not record.source_record_id.startswith(neg.NEGATIVE_ID_PREFIX + ":")


def test_parent_refusal_counts_are_present_at_zero_and_agree_with_the_reasons() -> None:
    """The two counts are the armed-ness evidence, so they must be stated, not inferred.

    `excluded_by_reason` omits a reason that never fired, so the report needs both numbers
    unconditionally — and they must equal the reason counts whenever those DO fire, or the
    number a gate reads and the number a human reads could drift apart.
    """
    clean = [_row(f"c{i}", _seq(WINDOW, i)) for i in range(3)]
    records, report = neg.negative_records_from_rows(clean, window=WINDOW)
    assert len(records) == 3
    # Present, not absent — a `.get(..., 0)` reader cannot tell those apart.
    assert "n_refused_parent_out_of_fold" in report
    assert "n_refused_parent_unresolved" in report
    assert report["n_refused_parent_out_of_fold"] == 0
    assert report["n_refused_parent_unresolved"] == 0
    # The companion control that must fire by construction: the same assertions over a
    # pool that DOES carry both refusals, so the zeros above are a measurement rather than
    # a field that is always zero.
    mixed = [
        *clean,
        _row("o1", _seq(WINDOW, 11), parent_nested_train=False),
        _row("o2", _seq(WINDOW, 12), parent_nested_train=False),
        _row("u1", _seq(WINDOW, 13), parent_nested_train=None),
    ]
    records, report = neg.negative_records_from_rows(mixed, window=WINDOW)
    assert len(records) == 3
    assert report["n_refused_parent_out_of_fold"] == 2
    assert report["n_refused_parent_unresolved"] == 1
    assert (
        report["n_refused_parent_out_of_fold"]
        == report["excluded_by_reason"][neg.REASON_PARENT_OUT_OF_FOLD]
    )
    assert (
        report["n_refused_parent_unresolved"]
        == report["excluded_by_reason"][neg.REASON_PARENT_UNRESOLVED]
    )
    assert report["n_records"] + report["n_excluded"] == report["n_rows_read"]


def test_the_refusal_ladder_order_is_pinned_and_each_row_counted_once() -> None:
    """A multiply-failing row is counted ONCE, under the first matching reason.

    Order is not cosmetic: it decides which cause a zero-record pool reports. A masked,
    designed-control, out-of-fold, wrong-width row counted under `shorter_than_window`
    would send a reader to re-carve a pool whose real problem is that the mask fired.
    Pinned by peeling one rung at a time — each patch clears exactly one failure and must
    expose exactly the next reason, which also proves no rung is dead (a rung that could
    never be reached would never appear).
    """
    row = _row(
        "",
        _seq(WINDOW - 7, 1),
        masked=True,
        is_designed_control=True,
        source_record_id="",
        parent_nested_train=False,
    )
    ladder = [
        ({}, neg.REASON_EMPTY_ID),
        ({"candidate_id": "ladder"}, neg.REASON_MASKED),
        ({"masked": False}, neg.REASON_DESIGNED_CONTROL),
        ({"is_designed_control": False}, neg.REASON_PARENT_UNRESOLVED),
        ({"source_record_id": "parent:ladder"}, neg.REASON_PARENT_OUT_OF_FOLD),
        ({"parent_nested_train": True}, neg.REASON_TOO_SHORT),
        ({"sequence": _seq(WINDOW + 7, 2)}, neg.REASON_TOO_LONG),
        ({"sequence": "R" * WINDOW}, neg.REASON_NON_ACGTN),
    ]
    for patch, expected in ladder:
        row = {**row, **patch}
        records, report = neg.negative_records_from_rows([row], window=WINDOW)
        assert records == [], expected
        # Exactly once, under exactly this reason — equality, not membership.
        assert report["excluded_by_reason"] == {expected: 1}, expected
        assert report["n_excluded"] == 1, expected
    # The terminal control: with the last failure cleared the SAME row is emitted, so the
    # ladder above ordered refusals rather than refusing a row that was never buildable.
    row = {**row, "sequence": _seq(WINDOW, 3)}
    records, report = neg.negative_records_from_rows([row], window=WINDOW)
    assert [r.record_id for r in records] == ["neg:ladder"]
    assert report["excluded_by_reason"] == {}
    assert records[0].source_record_id == "parent:ladder"


# ── CodeRabbit r1: caps and rounded-to-zero mixes ────────────────────────────────────
def test_max_records_zero_yields_zero_records() -> None:
    """A cap is a ceiling on what is built, not on what is built next.

    The post-append break emitted one record for a cap of 0, so the report's
    `max_records: 0` sat beside a pool of size 1 — a count contradicting its own scope.
    """
    rows = [_row(f"c{i}", _seq(WINDOW, i)) for i in range(5)]
    for cap in (0, 1, 3, 5, 9):
        records, report = neg.negative_records_from_rows(rows, window=WINDOW, max_records=cap)
        assert len(records) == min(cap, len(rows)), cap
        assert report["n_records"] == len(records)
        # P2-10d′-a briefly dropped this key: the new `require_parent_nested_train`
        # entry replaced its line in the report dict. Caught by this assertion, which was
        # left standing rather than deleted — without the key a pool that is *short* and
        # a pool that was *truncated by a cap* read identically in the artifact, the same
        # ambiguity the two parent-refusal counts exist to close.
        assert report["max_records"] == cap


def test_a_fraction_that_rounds_to_zero_draws_is_refused() -> None:
    """A mix that samples none of its pool is not a mix — refuse rather than round down.

    Refused at construction so `mix_clauses` can keep requiring a non-empty draw whenever
    a fraction was requested; relaxing that clause instead would also admit a run that
    trained on no negatives at all while its report described a negative curriculum.
    """
    ds = _dataset(10, 4)
    assert neg.negative_draw_count(10, 0.001) == 0
    with pytest.raises(neg.NegativeInjectionError, match="rounds to 0 negative draws"):
        neg.MixedIndexSampler(
            neg.positive_only_sampler(ds, n_positive=10),
            n_positive_records=10,
            n_negative_records=4,
            negative_fraction=0.001,
            seed=1,
        )
    # ...and the smallest fraction that DOES round to a draw is accepted, so the guard is
    # a boundary rather than a blanket refusal of small mixes.
    assert neg.negative_draw_count(10, 0.05) == 1
    ok = neg.MixedIndexSampler(
        neg.positive_only_sampler(ds, n_positive=10),
        n_positive_records=10,
        n_negative_records=4,
        negative_fraction=0.05,
        seed=1,
    )
    assert ok.mix_summary()["n_negative"] == 1
