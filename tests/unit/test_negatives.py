"""The P2-10d mined-negative injection hook: geometry, namespace, and a MEASURED mix.

Three properties this file exists to pin, each one a defect that has already shipped in
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

Bare-CI: numpy only (the module's pandas paths are exercised in
``tests/ml/test_negative_injection.py``, which needs the real parquet).
"""

from __future__ import annotations

import random

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
    )


def _negative(j: int) -> CorpusRecord:
    return neg.background_record(
        record_id=f"neg{j}", sequence=_seq(WINDOW, 10_000 + j), cluster_id=-(j + 1), window=WINDOW
    )


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
            record_id="short", sequence=_seq(WINDOW - 1, 1), cluster_id=-1, window=WINDOW
        )


def test_background_record_refuses_a_long_sequence() -> None:
    """A longer sequence would emit an unnamed sub-span under the record's whole-region id."""
    with pytest.raises(neg.NegativeInjectionError, match="exactly one window of real DNA"):
        neg.background_record(
            record_id="long", sequence=_seq(WINDOW + 1, 1), cluster_id=-1, window=WINDOW
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
        record_id="soft", sequence=seq.lower(), cluster_id=-1, window=WINDOW
    )
    hard = neg.background_record(record_id="soft", sequence=seq, cluster_id=-1, window=WINDOW)
    assert soft.context_seq == hard.context_seq == seq


def test_background_record_refuses_a_positive_cluster_namespace() -> None:
    """A non-negative cluster id can be swept into the P2-06a selection-val carve."""
    for bad in (0, 1, 4775):
        with pytest.raises(neg.NegativeInjectionError, match="private namespace"):
            neg.background_record(
                record_id="n", sequence=_seq(WINDOW, 1), cluster_id=bad, window=WINDOW
            )


def test_is_negative_record_keys_on_the_cluster_not_the_id() -> None:
    """The id prefix is cosmetic; the cluster namespace is the enforced property."""
    n = neg.background_record(
        record_id="not_prefixed_at_all", sequence=_seq(WINDOW, 3), cluster_id=-9, window=WINDOW
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
        {
            "candidate_id": "ok1",
            "sequence": _seq(WINDOW, 1),
            "masked": False,
            "is_designed_control": False,
        },
        {
            "candidate_id": "short",
            "sequence": _seq(WINDOW - 5, 2),
            "masked": False,
            "is_designed_control": False,
        },
        {
            "candidate_id": "long",
            "sequence": _seq(WINDOW + 5, 3),
            "masked": False,
            "is_designed_control": False,
        },
        {
            "candidate_id": "masked",
            "sequence": _seq(WINDOW, 4),
            "masked": True,
            "is_designed_control": False,
        },
        {
            "candidate_id": "ctl",
            "sequence": _seq(WINDOW, 5),
            "masked": False,
            "is_designed_control": True,
        },
        {
            "candidate_id": "",
            "sequence": _seq(WINDOW, 6),
            "masked": False,
            "is_designed_control": False,
        },
        {
            "candidate_id": "iupac",
            "sequence": "R" * WINDOW,
            "masked": False,
            "is_designed_control": False,
        },
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
    rows = [
        {
            "candidate_id": f"c{i}",
            "sequence": _seq(WINDOW, i),
            "masked": False,
            "is_designed_control": True,
        }
        for i in range(5)
    ]
    records, report = neg.negative_records_from_rows(rows, window=WINDOW)
    assert records == []
    assert report["excluded_by_reason"][neg.REASON_DESIGNED_CONTROL] == 5


def test_duplicate_candidate_ids_are_refused() -> None:
    """A duplicated hard negative is silently oversampled — the mix would misdescribe it."""
    rows = [
        {
            "candidate_id": "dup",
            "sequence": _seq(WINDOW, 1),
            "masked": False,
            "is_designed_control": False,
        },
        {
            "candidate_id": "dup",
            "sequence": _seq(WINDOW, 2),
            "masked": False,
            "is_designed_control": False,
        },
    ]
    with pytest.raises(neg.NegativeInjectionError, match="duplicate candidate_id"):
        neg.negative_records_from_rows(rows, window=WINDOW)


def test_negative_cluster_ids_are_unique_and_disjoint_from_positives() -> None:
    rows = [
        {
            "candidate_id": f"c{i}",
            "sequence": _seq(WINDOW, i),
            "masked": False,
            "is_designed_control": False,
        }
        for i in range(6)
    ]
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
