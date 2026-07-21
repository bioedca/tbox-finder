"""The junction-symmetric augmentation, end-to-end on an emitted stream (P2-10d′-b).

`tests/unit/test_junction_symmetry.py` tests the splice primitive and the gate's clauses.
This drives the real emission path — `Stage1WindowDataset.window_at` — and measures the
quantity the amended ADR-0005 A7 pin 7 actually gates: the **realized chimeric rate in
each class**.

The augmentation exists because a splice junction turned out to be readable. Measured on
the real substrate, a 6-mer probe separated background→background spliced windows from
plain ones at 0.5217 / 0.5264 / 0.5281 / 0.5328 across four independent draws (nulls
0.4947–0.5042). Small, reproducible, and — while only negatives were spliced — perfectly
class-correlated. Splicing positives at the negatives' own rate removes the cue by
construction, and what has to hold is then arithmetic: equal rates, and not one moved
label.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tbox_finder.data.window_dataset import (
    BACKGROUND_INDEX,
    CorpusRecord,
    Stage1DataConfig,
    Stage1WindowDataset,
)
from tbox_finder.eval.junction_probe import junction_symmetry_clauses

WINDOW = 128
N_POS = 60
N_NEG = 60
RATE = 0.5  # exaggerated vs the shipped ~0.19 so the smoke has power at n=60
LENGTHS = (12, 20, 28)


def _seq(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _positive(i: int) -> CorpusRecord:
    return CorpusRecord(
        record_id=f"pos{i}",
        context_seq=_seq(3 * WINDOW, i),
        locus_offset=WINDOW,
        locus_length=30,
        label_string="1" * 30,
        clipped_start=False,
        clipped_end=False,
        klass="I",
        phylum="Firmicutes",
        cognate_aa="ILE",
        cluster_id=i,  # non-negative ⇒ positive
        nested_train=True,
        is_designated_loo_holdout=False,
        folds=("train", None, None, None, True, "train"),
        source_record_id=f"pos{i}",
    )


def _negative(j: int) -> CorpusRecord:
    return CorpusRecord(
        record_id=f"neg{j}",
        context_seq=_seq(WINDOW, 10_000 + j),
        locus_offset=0,
        locus_length=WINDOW,
        label_string="." * WINDOW,
        clipped_start=False,
        clipped_end=False,
        klass="NEGATIVE",
        phylum="NEGATIVE",
        cognate_aa="NEGATIVE",
        cluster_id=-(j + 1),  # negative ⇒ never spliced by the augmentation
        nested_train=True,
        is_designated_loo_holdout=False,
        folds=(None, None, None, None, True, None),
        source_record_id=f"pos{j}",
    )


RECORDS = [*(_positive(i) for i in range(N_POS)), *(_negative(j) for j in range(N_NEG))]
DONORS = [_seq(WINDOW, 50_000 + d) for d in range(24)]


def _dataset(rate: float) -> Stage1WindowDataset:
    return Stage1WindowDataset(
        RECORDS,
        config=Stage1DataConfig(
            window_nt=WINDOW,
            seed=20260721,
            junction_symmetric_rate=rate,
            junction_symmetric_lengths=LENGTHS if rate > 0 else (),
        ),
        symmetric_donors=DONORS if rate > 0 else (),
    )


def _arm(rate: float) -> dict[str, int]:
    """Count chimeric windows per class by diffing against the un-augmented emission."""
    on, off = _dataset(rate), _dataset(0.0)
    counts = {
        "n_positive_draws": 0,
        "n_negative_draws": 0,
        "n_positive_chimeric": 0,
        "n_negative_chimeric": 0,
    }
    for index in range(len(RECORDS)):
        for occurrence in range(4):
            a = on.window_at(index, occurrence)
            b = off.window_at(index, occurrence)
            negative = RECORDS[index].cluster_id < 0
            counts["n_negative_draws" if negative else "n_positive_draws"] += 1
            if not np.array_equal(a.input_ids, b.input_ids):
                counts["n_negative_chimeric" if negative else "n_positive_chimeric"] += 1
    return counts


def test_the_augmentation_moves_not_one_label() -> None:
    """The load-bearing safety property. The splice writes only over positions already
    labelled background and already real, so a single moved target means the interval
    selector is wrong — and it would corrupt the exact per-nucleotide targets GATE-4
    grades."""
    on, off = _dataset(1.0), _dataset(0.0)
    checked = 0
    for index in range(len(RECORDS)):
        for occurrence in range(3):
            a, b = on.window_at(index, occurrence), off.window_at(index, occurrence)
            assert np.array_equal(a.labels, b.labels), (index, occurrence)
            assert np.array_equal(a.real_mask, b.real_mask)
            assert len(a.input_ids) == len(b.input_ids) == WINDOW
            # The DNA under every NON-background label must be byte-identical. This is
            # the invariant the label comparison above cannot see: `labels` is a separate
            # array, so a splice that overwrote a Stem-II position would leave the label
            # saying "Stem II" over donor DNA that is not one — a mislabeled positive,
            # produced silently. (Verified by sabotage: relaxing the interval selector to
            # ignore the label leaves the label check green and trips this one.)
            element = a.labels != BACKGROUND_INDEX
            assert np.array_equal(a.input_ids[element], b.input_ids[element])
            checked += 1
    assert checked > 0


def test_the_augmentation_actually_fires_on_positives() -> None:
    """MUST FIRE. A configured-but-inert augmentation looks identical in every metric to a
    working one whose cue happened to vanish."""
    counts = _arm(RATE)
    assert counts["n_positive_chimeric"] > 0


def test_negatives_are_never_touched_by_it() -> None:
    """Negatives already carry their chimera (the embedded decoy); splicing them again
    would raise both rates together and re-open the gap it exists to close."""
    assert _arm(RATE)["n_negative_chimeric"] == 0


def test_it_does_not_perturb_the_lead_or_strand_stream() -> None:
    """The splice draws from a salted-apart RNG. If it shared the augmentation generator,
    enabling it would silently re-phase every window of every prior run — the labels and
    the real-mask would move even where no splice landed."""
    on, off = _dataset(RATE), _dataset(0.0)
    for index in range(len(RECORDS)):
        for occurrence in range(3):
            a, b = on.window_at(index, occurrence), off.window_at(index, occurrence)
            assert a.lead == b.lead and a.reverse_complement == b.reverse_complement


def test_the_realized_positive_rate_tracks_the_configured_one() -> None:
    counts = _arm(RATE)
    observed = counts["n_positive_chimeric"] / counts["n_positive_draws"]
    assert abs(observed - RATE) < 0.15, counts


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_the_endpoints_behave(rate: float) -> None:
    counts = _arm(rate)
    if rate == 0.0:
        assert counts["n_positive_chimeric"] == 0
    else:
        # Not every window has room for the drawn insert; the shortfall is expected and
        # is why the gate measures the REALIZED rate rather than trusting the configured.
        assert counts["n_positive_chimeric"] > counts["n_positive_draws"] // 2


def test_a_rate_without_donors_is_refused() -> None:
    """Otherwise the config claims the junction cue is neutralised while nothing happens."""
    with pytest.raises(ValueError, match="no symmetric_donors"):
        Stage1WindowDataset(
            RECORDS,
            config=Stage1DataConfig(
                window_nt=WINDOW, junction_symmetric_rate=0.2, junction_symmetric_lengths=(8,)
            ),
        )


def test_a_rate_without_lengths_is_refused() -> None:
    with pytest.raises(ValueError, match="junction_symmetric_lengths is empty"):
        Stage1WindowDataset(
            RECORDS,
            config=Stage1DataConfig(window_nt=WINDOW, junction_symmetric_rate=0.2),
            symmetric_donors=DONORS,
        )


def test_the_gate_passes_on_a_symmetric_stream_and_its_control_fires() -> None:
    """The whole construction, graded by the shipped clause function.

    The negative arm here is synthetic (these negatives carry no embedded decoy), so the
    'negative chimeric rate' is supplied from the positive-side measurement of the same
    stream — what is being proven is that the CLAUSES read a real emission correctly and
    that the disabled control separates, not the shipped 0.19 rate itself.
    """
    on = _arm(RATE)
    n = on["n_positive_draws"]
    chim = on["n_positive_chimeric"]
    realized = {
        "augmented": {
            "n_positive_draws": n,
            "n_negative_draws": n,
            "n_positive_chimeric": chim,
            "n_negative_chimeric": chim,
        },
        "control_disabled": {
            "n_positive_draws": n,
            "n_negative_draws": n,
            "n_positive_chimeric": 0,
            "n_negative_chimeric": chim,
        },
    }
    summary = {
        "rate": chim / n,
        "n_chimeric_negatives": chim,
        "n_negative_records": n,
        "n_labels_changed": 0,
        "n_windows_label_checked": n,
    }
    clauses = junction_symmetry_clauses(summary, realized)
    assert all(clauses.values()), clauses
