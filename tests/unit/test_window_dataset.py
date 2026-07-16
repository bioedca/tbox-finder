"""Unit tests for the P2-01 Stage-1 window datamodule + curriculum sampling.

Covers the three things imp.md P2-01 names in its `Artifacts:` line — sampler
class-balance within tolerance, offset-augmentation determinism under a fixed
seed, and **variant→parent→fold provenance** (every oversampled/augmented
class-II variant inherits its parent's fold, ADR-0004 D7) — plus the window
geometry, the label-alignment rule, and the §10.3 honesty guards.

Tiering: the bulk is stdlib+numpy (bare CI). ``pandas`` (parquet join) and
``torch`` (collate / DataLoader) tiers are gated with an in-body
``importorskip`` so bare CI skips them green, per the repo convention.

Anti-tautology discipline ([[review-agents-mutate-code-under-review]]): the
tokenizer ids and the ignore index are asserted against their **external
published literals** (the Caduceus vocab / the ADR-0005 convention), never
against the module constant that produces them — a test comparing a function to
the constant behind it passes at any value.
"""

from __future__ import annotations

import numpy as np
import pytest

from tbox_finder import labels as labels_mod
from tbox_finder.data import window_dataset as wd

# ── Fixtures: synthetic records with fully controlled geometry ───────────────
_ACGT = "ACGT"


def _seq(n: int, *, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "".join(_ACGT[i] for i in rng.integers(0, 4, size=n))


def _record(
    *,
    rid: str = "r0",
    lead: int = 1024,
    locus: int = 300,
    trail: int = 1024,
    clipped_start: bool = False,
    clipped_end: bool = False,
    klass: str = "I",
    phylum: str = "Firmicutes",
    aa: str = "ILE",
    nested_train: bool = True,
    folds: tuple = ("train", None, None, None, True, "train"),
    label_string: str | None = None,
    seed: int = 0,
) -> wd.CorpusRecord:
    """A synthetic corpus record with exact, hand-checkable geometry."""
    return wd.CorpusRecord(
        record_id=rid,
        context_seq=_seq(lead + locus + trail, seed=seed),
        locus_offset=lead,
        locus_length=locus,
        label_string=label_string if label_string is not None else "1" * locus,
        clipped_start=clipped_start,
        clipped_end=clipped_end,
        klass=klass,
        phylum=phylum,
        cognate_aa=aa,
        cluster_id=0,
        nested_train=nested_train,
        folds=folds,
    )


# ═══════════════════════════════════════════════════════════════════════════
# External-literal pins (NOT compared to the constant that produces them)
# ═══════════════════════════════════════════════════════════════════════════
def test_tokenizer_vocab_matches_the_published_caduceus_literals() -> None:
    """Pin the Caduceus-PS vocab to its published values, not to our constants.

    Source: ``tokenization_caduceus.py`` @ revision d89eeb85 — specials occupy
    0-6 with ``[PAD] == 4``, then ``A,C,G,T,N`` at ``i + 7``. If our ids drift
    from these, every window we emit is silently mistokenised.
    """
    assert wd.PAD_TOKEN_ID == 4
    assert wd.BASE_TO_ID == {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
    assert wd.UNKNOWN_BASE_ID == 11


def test_ignore_index_is_the_adr0005_literal_and_does_not_drift() -> None:
    """-100 is the seg-head CE ignore value; also drift-guard vs seg_smoke."""
    assert wd.IGNORE_INDEX == -100
    from tbox_finder.data import seg_smoke

    assert wd.IGNORE_INDEX == seg_smoke.IGNORE_INDEX


def test_class_vocabulary_pins() -> None:
    """background is index 0 and there are 8 classes (PRD §8 / ADR-0004 D1)."""
    assert wd.BACKGROUND_INDEX == 0
    assert wd.NUM_CLASSES == 8
    assert labels_mod.CLASS_ORDER[0] == "background"


def test_window_and_stride_match_the_prd_pins() -> None:
    """PRD §6 pins W=1024 / stride=512 — these are NOT implementer choices."""
    assert wd.WINDOW_NT == 1024
    assert wd.STRIDE_NT == 512


def test_sampling_alphas_are_declared_unpinned() -> None:
    """PRD §11 pins that the samplers exist, not their strength (§10.3 honesty)."""
    assert wd.WEIGHTS_PINNED is False


# ═══════════════════════════════════════════════════════════════════════════
# Encoding / RC
# ═══════════════════════════════════════════════════════════════════════════
def test_encode_bases_maps_every_base() -> None:
    assert list(wd.encode_bases("ACGTN")) == [7, 8, 9, 10, 11]
    assert wd.encode_bases("").shape == (0,)


def test_encode_bases_sends_unknown_characters_to_N() -> None:
    assert list(wd.encode_bases("ACXG")) == [7, 8, 11, 9]


def test_reverse_complement_is_an_involution() -> None:
    s = _seq(101, seed=3)
    assert wd.reverse_complement(wd.reverse_complement(s)) == s


def test_reverse_complement_ids_matches_string_reverse_complement() -> None:
    """The id-space RC must agree with the string-space RC (independent paths)."""
    s = _seq(64, seed=5)
    assert np.array_equal(
        wd.reverse_complement_ids(wd.encode_bases(s)),
        wd.encode_bases(wd.reverse_complement(s)),
    )


def test_reverse_complement_ids_preserves_pad_as_pad() -> None:
    """A zero-flank must survive the RC as a zero-flank, not become a base."""
    ids = np.asarray([wd.PAD_TOKEN_ID, 7, 8, wd.PAD_TOKEN_ID], dtype=np.int16)
    rc = wd.reverse_complement_ids(ids)
    assert rc[0] == wd.PAD_TOKEN_ID and rc[-1] == wd.PAD_TOKEN_ID
    assert list(rc) == [4, 9, 10, 4]  # revcomp of [PAD],A,C,[PAD] -> [PAD],G,T,[PAD]


def test_reverse_complement_ids_does_not_mutate_its_input() -> None:
    ids = wd.encode_bases("ACGT")
    before = ids.copy()
    wd.reverse_complement_ids(ids)
    assert np.array_equal(ids, before)


# ═══════════════════════════════════════════════════════════════════════════
# tile_windows (the PRD §6 scan tiling P2-03 consumes)
# ═══════════════════════════════════════════════════════════════════════════
def test_tile_windows_covers_every_position_and_anchors_the_tail() -> None:
    starts = wd.tile_windows(2500, window=1024, stride=512)
    assert starts[0] == 0
    assert starts[-1] == 2500 - 1024  # tail anchored, no uncovered 3' remainder
    covered = set()
    for s in starts:
        covered.update(range(s, s + 1024))
    assert covered == set(range(2500))


def test_tile_windows_gives_every_interior_position_at_least_two_windows() -> None:
    """Stride 512 / window 1024 is the >=2-coverage PRD §6 relies on for P2-03."""
    starts = wd.tile_windows(3000, window=1024, stride=512)
    counts = np.zeros(3000, dtype=int)
    for s in starts:
        counts[s : s + 1024] += 1
    assert counts[1024:-1024].min() >= 2


def test_tile_windows_short_sequence_is_a_single_window() -> None:
    assert wd.tile_windows(500, window=1024, stride=512) == [0]
    assert wd.tile_windows(1024, window=1024, stride=512) == [0]


@pytest.mark.parametrize("bad", [(0, 1024, 512), (100, 0, 512), (100, 1024, 0)])
def test_tile_windows_rejects_non_positive_arguments(bad: tuple) -> None:
    seq_len, window, stride = bad
    with pytest.raises(ValueError):
        wd.tile_windows(seq_len, window=window, stride=stride)


# ═══════════════════════════════════════════════════════════════════════════
# window_lead_range — the §10.3 honesty rule
# ═══════════════════════════════════════════════════════════════════════════
def test_lead_range_is_the_full_phase_range_when_flank_is_ample() -> None:
    """1024 nt of real flank each side => every window phase is honest."""
    r = wd.window_lead_range(
        locus_offset=1024,
        locus_length=300,
        context_length=1024 + 300 + 1024,
        clipped_start=False,
        clipped_end=False,
    )
    assert r == (0, 1024 - 300)


def test_lead_range_refuses_to_step_off_unclipped_context_start() -> None:
    """Not clipped => context_seq[0] is where our FETCH began, not a contig start.

    Stepping left of it would invent sequence, so the lead is capped at the real
    lead flank (CLAUDE.md §10.3).
    """
    r = wd.window_lead_range(
        locus_offset=100,
        locus_length=300,
        context_length=100 + 300 + 1024,
        clipped_start=False,
        clipped_end=False,
    )
    assert r is not None and r[1] == 100


def test_lead_range_allows_stepping_off_a_clipped_contig_start() -> None:
    """clipped_start => that boundary is real, so zero-flanking it is honest."""
    r = wd.window_lead_range(
        locus_offset=100,
        locus_length=300,
        context_length=100 + 300 + 1024,
        clipped_start=True,
        clipped_end=False,
    )
    assert r == (0, 1024 - 300)


def test_lead_range_refuses_to_step_off_unclipped_context_end() -> None:
    r = wd.window_lead_range(
        locus_offset=1024,
        locus_length=300,
        context_length=1024 + 300 + 50,
        clipped_start=False,
        clipped_end=False,
    )
    # need window - lead - locus <= trail(50)  =>  lead >= 1024-300-50 = 674
    assert r is not None and r[0] == 674


def test_lead_range_allows_stepping_off_a_clipped_contig_end() -> None:
    r = wd.window_lead_range(
        locus_offset=1024,
        locus_length=300,
        context_length=1024 + 300 + 50,
        clipped_start=False,
        clipped_end=True,
    )
    assert r == (0, 1024 - 300)


def test_lead_range_returns_none_when_no_honest_window_exists() -> None:
    """Context shorter than the window with no real contig end to pad against."""
    assert (
        wd.window_lead_range(
            locus_offset=100,
            locus_length=300,
            context_length=500,
            clipped_start=False,
            clipped_end=False,
        )
        is None
    )


def test_lead_range_returns_none_when_the_locus_exceeds_the_window() -> None:
    assert (
        wd.window_lead_range(
            locus_offset=10,
            locus_length=2000,
            context_length=4000,
            clipped_start=True,
            clipped_end=True,
        )
        is None
    )


def test_lead_range_is_full_when_both_ends_are_real_contig_ends() -> None:
    r = wd.window_lead_range(
        locus_offset=10,
        locus_length=300,
        context_length=400,
        clipped_start=True,
        clipped_end=True,
    )
    assert r == (0, 1024 - 300)


def test_lead_range_rejects_a_locus_running_past_the_context() -> None:
    with pytest.raises(ValueError, match="runs past the end"):
        wd.window_lead_range(
            locus_offset=900,
            locus_length=300,
            context_length=1000,
            clipped_start=False,
            clipped_end=False,
        )


# ═══════════════════════════════════════════════════════════════════════════
# carve_window — label alignment + zero-flank
# ═══════════════════════════════════════════════════════════════════════════
def test_carve_window_places_labels_at_the_lead_offset() -> None:
    r = _record(locus=300, label_string="S" * 300)
    w = wd.carve_window(
        context_seq=r.context_seq,
        locus_offset=r.locus_offset,
        locus_length=r.locus_length,
        label_string=r.label_string,
        lead=137,
        record_id=r.record_id,
        clipped_start=False,
        clipped_end=False,
    )
    specifier = labels_mod.CLASS_INDEX["Specifier"]
    assert np.all(w.labels[137:437] == specifier)
    assert np.all(w.labels[:137] == wd.BACKGROUND_INDEX)
    assert np.all(w.labels[437:] == wd.BACKGROUND_INDEX)


def test_carve_window_real_flank_is_background_never_ignored() -> None:
    """imp.md P2-01: the flank is real DNA; real background is the training signal."""
    r = _record()
    w = wd.carve_window(
        context_seq=r.context_seq,
        locus_offset=r.locus_offset,
        locus_length=r.locus_length,
        label_string=r.label_string,
        lead=200,
        record_id=r.record_id,
        clipped_start=False,
        clipped_end=False,
    )
    assert not w.zero_flanked
    assert not np.any(w.labels == wd.IGNORE_INDEX)
    assert np.all(w.real_mask)


def test_carve_window_zero_flank_is_ignored_and_padded() -> None:
    """A real contig start => [PAD] tokens + IGNORE_INDEX labels, and a flag."""
    r = _record(lead=10, locus=300, trail=1024, clipped_start=True)
    w = wd.carve_window(
        context_seq=r.context_seq,
        locus_offset=r.locus_offset,
        locus_length=r.locus_length,
        label_string=r.label_string,
        lead=500,  # asks for 500 nt of lead, only 10 exist -> 490 zero-flank
        record_id=r.record_id,
        clipped_start=True,
        clipped_end=False,
    )
    assert w.pad_left == 490
    assert w.zero_flanked
    assert np.all(w.input_ids[:490] == wd.PAD_TOKEN_ID)
    assert np.all(w.labels[:490] == wd.IGNORE_INDEX)
    assert not np.any(w.labels[:490] == wd.BACKGROUND_INDEX)
    assert not w.real_mask[:490].any()
    assert w.real_mask[490:].all()


def test_carve_window_rejects_a_lead_outside_the_honest_range() -> None:
    r = _record(lead=100, locus=300, trail=1024)
    with pytest.raises(ValueError, match="outside the honest range"):
        wd.carve_window(
            context_seq=r.context_seq,
            locus_offset=r.locus_offset,
            locus_length=r.locus_length,
            label_string=r.label_string,
            lead=500,  # > lead_flank(100) and not clipped_start -> would invent DNA
            record_id=r.record_id,
            clipped_start=False,
            clipped_end=False,
        )


def test_carve_window_rejects_a_label_string_of_the_wrong_length() -> None:
    r = _record(locus=300)
    with pytest.raises(ValueError, match="label_string length"):
        wd.carve_window(
            context_seq=r.context_seq,
            locus_offset=r.locus_offset,
            locus_length=300,
            label_string="1" * 299,
            lead=0,
            record_id=r.record_id,
            clipped_start=False,
            clipped_end=False,
        )


def test_carve_window_sequence_matches_the_context_slice() -> None:
    r = _record()
    w = wd.carve_window(
        context_seq=r.context_seq,
        locus_offset=r.locus_offset,
        locus_length=r.locus_length,
        label_string=r.label_string,
        lead=300,
        record_id=r.record_id,
        clipped_start=False,
        clipped_end=False,
    )
    start = r.locus_offset - 300
    assert np.array_equal(w.input_ids, wd.encode_bases(r.context_seq[start : start + 1024]))


def test_carve_window_rc_reverses_sequence_and_labels_together() -> None:
    """RC must move sequence and per-nt targets in lockstep, or labels desync."""
    r = _record(label_string="S" * 300)
    kw = dict(
        context_seq=r.context_seq,
        locus_offset=r.locus_offset,
        locus_length=r.locus_length,
        label_string=r.label_string,
        lead=137,
        record_id=r.record_id,
        clipped_start=False,
        clipped_end=False,
    )
    fwd = wd.carve_window(**kw, rc=False)
    rev = wd.carve_window(**kw, rc=True)
    assert np.array_equal(rev.input_ids, wd.reverse_complement_ids(fwd.input_ids))
    assert np.array_equal(rev.labels, fwd.labels[::-1])
    assert np.array_equal(rev.real_mask, fwd.real_mask[::-1])
    # the locus still carries exactly the same class multiset after the flip
    assert (rev.labels == labels_mod.CLASS_INDEX["Specifier"]).sum() == 300


def test_carve_window_rc_lead_describes_the_emitted_window() -> None:
    """Regression: `lead` must locate the locus in the RC window, not the forward one.

    Reversing the array moves the locus to `window - lead - locus_length`. Reporting
    the forward lead makes `labels[lead : lead + locus_length]` read the wrong span
    for every reverse-strand sample — silently mistraining half the both-strand data.
    """
    r = _record(label_string="S" * 300)
    kw = dict(
        context_seq=r.context_seq,
        locus_offset=r.locus_offset,
        locus_length=r.locus_length,
        label_string=r.label_string,
        lead=137,
        record_id=r.record_id,
        clipped_start=False,
        clipped_end=False,
    )
    fwd = wd.carve_window(**kw, rc=False)
    rev = wd.carve_window(**kw, rc=True)
    specifier = labels_mod.CLASS_INDEX["Specifier"]

    assert fwd.lead == 137
    assert rev.lead == wd.WINDOW_NT - 137 - 300 == 587
    # The reported lead must actually locate the locus in each emitted window.
    for w in (fwd, rev):
        assert np.all(w.labels[w.lead : w.lead + 300] == specifier)
        assert int(np.flatnonzero(w.labels == specifier)[0]) == w.lead
    # ...and the variant id must follow the emitted geometry, not the forward one.
    assert rev.record_id == wd.variant_id(r.record_id, lead=587, rc=True)


def test_carve_window_rc_swaps_the_pad_sides() -> None:
    r = _record(lead=10, locus=300, trail=1024, clipped_start=True)
    kw = dict(
        context_seq=r.context_seq,
        locus_offset=r.locus_offset,
        locus_length=r.locus_length,
        label_string=r.label_string,
        lead=500,
        record_id=r.record_id,
        clipped_start=True,
        clipped_end=False,
    )
    fwd = wd.carve_window(**kw, rc=False)
    rev = wd.carve_window(**kw, rc=True)
    assert (fwd.pad_left, fwd.pad_right) == (490, 0)
    assert (rev.pad_left, rev.pad_right) == (0, 490)


def test_variant_id_is_deterministic_and_distinct_from_the_parent() -> None:
    """A variant must not collide with its parent id (ADR-0004 D7 predicate)."""
    vid = wd.variant_id("abc", lead=7, rc=True)
    assert vid == wd.variant_id("abc", lead=7, rc=True)
    assert vid != "abc"
    assert wd.variant_id("abc", lead=7, rc=False) != vid


# ═══════════════════════════════════════════════════════════════════════════
# Curriculum weights (PRD §11)
# ═══════════════════════════════════════════════════════════════════════════
def test_inverse_frequency_alpha_zero_is_the_natural_distribution() -> None:
    w = wd.inverse_frequency_weights(["a"] * 9 + ["b"], alpha=0.0)
    assert np.allclose(w, 1.0)


def test_inverse_frequency_alpha_one_fully_balances_strata() -> None:
    """alpha=1 => every stratum receives equal total mass, regardless of size."""
    keys = ["a"] * 9 + ["b"]
    w = wd.inverse_frequency_weights(keys, alpha=1.0)
    mass_a = w[:9].sum()
    mass_b = w[9:].sum()
    assert np.isclose(mass_a, mass_b)


def test_inverse_frequency_alpha_half_is_between_natural_and_balanced() -> None:
    keys = ["a"] * 100 + ["b"] * 4
    share = lambda a: (lambda w: w[100:].sum() / w.sum())(  # noqa: E731
        wd.inverse_frequency_weights(keys, alpha=a)
    )
    natural, half, balanced = share(0.0), share(0.5), share(1.0)
    assert np.isclose(natural, 4 / 104)
    assert np.isclose(balanced, 0.5)
    assert natural < half < balanced


def test_inverse_frequency_weights_are_mean_normalised() -> None:
    w = wd.inverse_frequency_weights(["a"] * 5 + ["b"] * 3 + ["c"], alpha=0.5)
    assert np.isclose(w.mean(), 1.0)


def test_missing_and_blank_strata_form_their_own_bucket() -> None:
    """A blank cognate_aa must not silently join the majority stratum."""
    w = wd.inverse_frequency_weights(["ILE", "ILE", "", None, float("nan")], alpha=1.0)
    # 2 ILE + 3 UNKNOWN -> the UNKNOWN members share equal total mass with ILE
    assert np.isclose(w[:2].sum(), w[2:].sum())


@pytest.mark.parametrize("alpha", [-0.01, 1.01, 2.0])
def test_inverse_frequency_rejects_alpha_outside_the_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        wd.inverse_frequency_weights(["a", "b"], alpha=alpha)


def test_inverse_frequency_on_empty_input_is_empty() -> None:
    assert wd.inverse_frequency_weights([], alpha=0.5).shape == (0,)


def test_curriculum_weights_lift_the_rare_class_and_the_rare_phylum() -> None:
    """A record rare on all three axes is oversampled, but not to full balance."""
    n_major, n_minor = 200, 4
    phyla = ["Firmicutes"] * n_major + ["Chloroflexi"] * n_minor
    klasses = ["I"] * n_major + ["II"] * n_minor
    aas = ["ILE"] * n_major + ["LYS"] * n_minor
    w = wd.curriculum_weights(phyla=phyla, klasses=klasses, amino_acids=aas)
    minor_share = w[n_major:].sum() / w.sum()
    natural_share = n_minor / (n_major + n_minor)
    assert minor_share > natural_share  # oversampled...
    assert minor_share < 0.5  # ...but the majority still carries the stream


def test_curriculum_alpha_terms_compound_across_axes() -> None:
    """The documented gotcha: three correlated axes multiply, they do not average.

    This is why the defaults are 0.25 and not 0.5 — pin the behaviour so the
    docstring's sweep table cannot silently stop describing the code.
    """
    n_major, n_minor = 200, 4
    phyla = ["Firmicutes"] * n_major + ["Chloroflexi"] * n_minor
    klasses = ["I"] * n_major + ["II"] * n_minor
    aas = ["ILE"] * n_major + ["LYS"] * n_minor
    share = lambda **kw: (lambda w: w[n_major:].sum() / w.sum())(  # noqa: E731
        wd.curriculum_weights(phyla=phyla, klasses=klasses, amino_acids=aas, **kw)
    )

    one_axis = share(phylum_alpha=0.3, klass_alpha=0.0, aa_alpha=0.0)
    three_axes = share(phylum_alpha=0.3, klass_alpha=0.3, aa_alpha=0.3)
    joint_equivalent = share(phylum_alpha=0.9, klass_alpha=0.0, aa_alpha=0.0)
    assert three_axes > one_axis
    assert np.isclose(three_axes, joint_equivalent), "terms must compound multiplicatively"


def test_curriculum_weights_reject_ragged_inputs() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        wd.curriculum_weights(phyla=["a", "b"], klasses=["I"], amino_acids=["ILE", "LEU"])


# ═══════════════════════════════════════════════════════════════════════════
# WeightedIndexSampler — determinism + class balance within tolerance
# ═══════════════════════════════════════════════════════════════════════════
def test_sampler_is_deterministic_under_a_fixed_seed() -> None:
    w = np.asarray([1.0, 2.0, 3.0, 4.0])
    a = list(wd.WeightedIndexSampler(w, num_samples=50, seed=42))
    b = list(wd.WeightedIndexSampler(w, num_samples=50, seed=42))
    assert a == b


def test_sampler_yields_index_occurrence_keys_with_unique_occurrences() -> None:
    """The key carries the draw ordinal so repeated draws can be re-augmented."""
    keys = list(wd.WeightedIndexSampler(np.ones(3), num_samples=10, seed=42))
    assert all(isinstance(k, tuple) and len(k) == 2 for k in keys)
    assert [occ for _, occ in keys] == list(range(10))  # unique, ordered
    assert all(0 <= i < 3 for i, _ in keys)
    assert wd.WeightedIndexSampler(np.ones(3), num_samples=10, seed=42).indices() == [
        i for i, _ in keys
    ]


def test_sampler_differs_across_seeds_and_epochs() -> None:
    w = np.asarray([1.0] * 20)
    a = wd.WeightedIndexSampler(w, num_samples=50, seed=42).indices()
    b = wd.WeightedIndexSampler(w, num_samples=50, seed=43).indices()
    assert a != b
    s = wd.WeightedIndexSampler(w, num_samples=50, seed=42)
    e0 = s.indices()
    s.set_epoch(1)
    e1 = s.indices()
    assert e0 != e1
    s.set_epoch(0)
    assert s.indices() == e0  # epochs are reproducible, not just different


def test_sampler_class_balance_is_within_tolerance_of_the_analytic_share() -> None:
    """imp.md P2-01: 'sampler class-balance within tolerance'.

    The empirical class-II draw share must match the analytic weight share the
    curriculum asks for — this is what proves the oversampler actually reshapes
    the stream rather than merely computing a weight vector nobody honours.
    """
    n_major, n_minor = 500, 10
    phyla = ["Firmicutes"] * n_major + ["Chloroflexi"] * n_minor
    klasses = ["I"] * n_major + ["II"] * n_minor
    aas = ["ILE"] * n_major + ["LYS"] * n_minor
    w = wd.curriculum_weights(phyla=phyla, klasses=klasses, amino_acids=aas)
    expected = w[n_major:].sum() / w.sum()

    draws = np.asarray(wd.WeightedIndexSampler(w, num_samples=200_000, seed=42).indices())
    observed = float((draws >= n_major).mean())
    # 200k draws: the sampling s.e. of a ~p share is well under 0.005
    assert abs(observed - expected) < 0.01, f"observed {observed:.4f} vs expected {expected:.4f}"
    assert observed > n_minor / (n_major + n_minor)  # genuinely oversampled


def test_sampler_never_draws_a_zero_weight_record() -> None:
    """A zero weight must mean *excluded*, not merely unlikely."""
    w = np.asarray([0.0, 0.0, 1.0, 0.0])
    assert set(wd.WeightedIndexSampler(w, num_samples=500, seed=1).indices()) == {2}


def test_sampler_length_is_the_requested_epoch_size() -> None:
    s = wd.WeightedIndexSampler(np.ones(7), num_samples=31, seed=1)
    assert len(s) == 31
    assert len(list(s)) == 31
    assert len(wd.WeightedIndexSampler(np.ones(7), seed=1)) == 7  # default = one pass


@pytest.mark.parametrize(
    "weights, match",
    [
        (np.zeros(3), "positive value"),
        (np.asarray([-1.0, 1.0]), "non-negative"),
        (np.asarray([np.nan, 1.0]), "finite"),
        (np.asarray([np.inf, 1.0]), "finite"),
        (np.zeros(0), "non-empty"),
        (np.ones((2, 2)), "1-D"),
    ],
)
def test_sampler_rejects_degenerate_weights(weights: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        wd.WeightedIndexSampler(weights, seed=1)


def test_sampler_rejects_non_positive_num_samples() -> None:
    with pytest.raises(ValueError, match="num_samples must be positive"):
        wd.WeightedIndexSampler(np.ones(3), num_samples=0, seed=1)


# ═══════════════════════════════════════════════════════════════════════════
# Stage1WindowDataset — offset augmentation determinism
# ═══════════════════════════════════════════════════════════════════════════
def _dataset(n: int = 6, **kw) -> wd.Stage1WindowDataset:
    records = [_record(rid=f"r{i}", seed=i) for i in range(n)]
    return wd.Stage1WindowDataset(records, **kw)


def test_offset_augmentation_is_deterministic_under_a_fixed_seed() -> None:
    """imp.md P2-01: 'offset-augmentation determinism under fixed seed' (§8.3)."""
    a = _dataset()
    b = _dataset()
    for i in range(len(a)):
        wa, wb = a.window_at(i), b.window_at(i)
        assert wa.lead == wb.lead
        assert wa.reverse_complement == wb.reverse_complement
        assert np.array_equal(wa.input_ids, wb.input_ids)
        assert np.array_equal(wa.labels, wb.labels)


def test_offset_augmentation_varies_across_epochs_and_reproduces_on_return() -> None:
    ds = _dataset(n=24)
    e0 = [ds.window_at(i).lead for i in range(len(ds))]
    ds.set_epoch(1)
    e1 = [ds.window_at(i).lead for i in range(len(ds))]
    assert e0 != e1, "augmentation must actually re-phase across epochs"
    ds.set_epoch(0)
    assert [ds.window_at(i).lead for i in range(len(ds))] == e0


def test_repeated_draws_of_one_record_get_independent_augmentation() -> None:
    """Regression — the bug that made oversampling pointless.

    With replacement sampling, a 9x-oversampled class-II record is drawn 9 times
    in one epoch. When the augmentation RNG was keyed only on (seed, epoch, index),
    every one of those draws returned the SAME window — 9 identical copies, so the
    curriculum's whole reason for pairing oversampling with offset augmentation
    (PRD §11) was defeated and the model would simply memorise the locus. The draw
    `occurrence` is now part of the key, so each draw is independently re-phased.
    """
    ds = _dataset(n=1)
    leads = [ds.window_at(0, occurrence=o).lead for o in range(16)]
    assert len(set(leads)) > 1, "repeated draws of one record returned identical windows"
    # ...and it is still exactly reproducible.
    assert leads == [ds.window_at(0, occurrence=o).lead for o in range(16)]


def test_occurrence_keys_reach_the_dataset_through_getitem() -> None:
    """`dataset[(index, occurrence)]` is how the sampler's key arrives via DataLoader."""
    ds = _dataset(n=1)
    a = ds[(0, 0)]
    b = ds[(0, 7)]
    assert ds[0]["record_id"] == a["record_id"]  # bare int == occurrence 0
    assert a["parent_record_id"] == b["parent_record_id"] == "r0"
    assert not np.array_equal(a["input_ids"], b["input_ids"]) or a["record_id"] != b["record_id"]


def test_offset_augmentation_varies_across_records_within_an_epoch() -> None:
    ds = _dataset(n=24)
    leads = {ds.window_at(i).lead for i in range(len(ds))}
    assert len(leads) > 1, "every record drew the same phase — the RNG is not per-index"


def test_offset_augmentation_covers_the_phase_range_over_many_epochs() -> None:
    """Phase robustness (PRD §6) needs real coverage of [0, W-m], not a jitter."""
    ds = _dataset(n=1)
    leads = []
    for epoch in range(400):
        ds.set_epoch(epoch)
        leads.append(ds.window_at(0).lead)
    lo, hi = 0, wd.WINDOW_NT - 300
    assert min(leads) < lo + 0.1 * (hi - lo)
    assert max(leads) > lo + 0.9 * (hi - lo)


def test_eval_mode_is_unaugmented_centred_and_forward_strand() -> None:
    ds = _dataset(augment=False)
    for i in range(len(ds)):
        w = ds.window_at(i)
        assert w.lead == (wd.WINDOW_NT - 300) // 2
        assert w.reverse_complement is False


def test_eval_mode_lead_is_clamped_into_the_honest_range() -> None:
    """A short-flank record must not centre itself off the end of real context."""
    r = _record(lead=50, locus=300, trail=1024)
    ds = wd.Stage1WindowDataset([r], augment=False)
    w = ds.window_at(0)
    assert w.lead == 50  # centre (362) clamped down to the real lead flank
    assert not w.zero_flanked


def test_dataset_rejects_a_record_with_no_honest_window() -> None:
    """The dataset must refuse what load_corpus_records should have excluded."""
    bad = _record(lead=100, locus=300, trail=50)  # 450 nt context, no clip flags
    with pytest.raises(ValueError, match="admits no honest window"):
        wd.Stage1WindowDataset([bad])


def test_dataset_rejects_an_empty_record_set() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        wd.Stage1WindowDataset([])


def test_getitem_shapes_and_keys() -> None:
    item = _dataset()[0]
    assert item["input_ids"].shape == (1024,)
    assert item["labels"].shape == (1024,)
    assert item["real_mask"].shape == (1024,)
    assert item["parent_record_id"] == "r0"
    assert item["record_id"].startswith("r0#w")


def test_labels_are_within_the_class_range_or_ignored() -> None:
    """No emitted target may fall outside {0..7} ∪ {IGNORE_INDEX}."""
    ds = _dataset(n=8)
    for i in range(len(ds)):
        lab = ds.window_at(i).labels
        valid = (lab == wd.IGNORE_INDEX) | ((lab >= 0) & (lab < wd.NUM_CLASSES))
        assert bool(valid.all())


# ═══════════════════════════════════════════════════════════════════════════
# variant → parent → fold provenance (ADR-0004 D7)
# ═══════════════════════════════════════════════════════════════════════════
def _variant_parent_fold_mismatches(rows: list[dict]) -> list[tuple[str, int]]:
    """The ADR-0004 D7 predicate, mirroring tests/ml/test_no_leakage.py.

    Re-stated here (rather than imported) so this unit test pins the invariant
    independently of the CI gate's own implementation.
    """
    index = {r[wd.RECORD_ID_COL]: i for i, r in enumerate(rows)}
    viol: list[tuple[str, int]] = []
    for i, r in enumerate(rows):
        pid = r[wd.PARENT_ID_COL]
        if pid is None or pid == r[wd.RECORD_ID_COL]:
            continue
        j = index.get(pid)
        if j is None:
            viol.append(("orphan_parent", i))
            continue
        for col in wd.FOLD_SCHEME_COLUMNS:
            if rows[i][col] != rows[j][col]:
                viol.append((col, i))
                break
    return viol


def test_every_emitted_variant_inherits_its_parents_fold() -> None:
    """imp.md P2-01: every oversampled/augmented class-II variant inherits its
    parent's fold — the ADR-0004 D7 invariant, on the dataset's own output."""
    train_folds = ("train", None, None, None, True, "train")
    records = [
        _record(rid=f"r{i}", seed=i, klass="II" if i % 3 == 0 else "I", folds=train_folds)
        for i in range(9)
    ]
    ds = wd.Stage1WindowDataset(records)
    rows = ds.provenance_rows()
    assert len(rows) == len(records)
    # Every row IS a variant (parent_record_id != record_id), so the predicate is
    # not vacuous the way the committed base table's self-parented rows are.
    assert all(r[wd.PARENT_ID_COL] != r[wd.RECORD_ID_COL] for r in rows)

    # Append the parents themselves so the predicate can resolve them.
    parent_rows = [
        {
            wd.RECORD_ID_COL: r.record_id,
            wd.PARENT_ID_COL: r.record_id,
            wd.CLUSTER_COL: r.cluster_id,
            **dict(zip(wd.FOLD_SCHEME_COLUMNS, r.folds, strict=True)),
        }
        for r in records
    ]
    assert _variant_parent_fold_mismatches(rows + parent_rows) == []


def test_the_provenance_predicate_bites_on_a_corrupted_variant_fold() -> None:
    """Anti-tautology: a variant carrying the WRONG fold must be caught.

    Without this, `test_every_emitted_variant_inherits_its_parents_fold` would
    pass even if the predicate could never fail.
    """
    records = [_record(rid="r0", folds=("train", None, None, None, True, "train"))]
    ds = wd.Stage1WindowDataset(records)
    rows = ds.provenance_rows()
    parent = {
        wd.RECORD_ID_COL: "r0",
        wd.PARENT_ID_COL: "r0",
        wd.CLUSTER_COL: 0,
        **dict(zip(wd.FOLD_SCHEME_COLUMNS, records[0].folds, strict=True)),
    }
    assert _variant_parent_fold_mismatches(rows + [parent]) == []
    rows[0]["nested_role"] = "heldout"  # smuggle a held-out fold onto the variant
    assert _variant_parent_fold_mismatches(rows + [parent]) != []


def test_the_provenance_predicate_bites_on_an_orphan_parent() -> None:
    records = [_record(rid="r0")]
    rows = wd.Stage1WindowDataset(records).provenance_rows()
    assert _variant_parent_fold_mismatches(rows) == [("orphan_parent", 0)]


def test_variants_carry_the_parents_cluster_id() -> None:
    """Cluster identity must ride along, or cluster non-splitting is uncheckable."""
    records = [_record(rid="r0")]
    rows = wd.Stage1WindowDataset(records).provenance_rows()
    assert rows[0][wd.CLUSTER_COL] == records[0].cluster_id


# ═══════════════════════════════════════════════════════════════════════════
# pandas tier — the real parquet join + training-fold constraint
# ═══════════════════════════════════════════════════════════════════════════
def _require_real_artifacts() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    for p in (wd.DEFAULT_CONTEXT, wd.DEFAULT_LABELS, wd.DEFAULT_SPLIT_TABLE):
        if not p.exists():
            pytest.skip(f"DVC/LFS artifact not materialised: {p}")


def test_load_corpus_records_returns_only_training_fold_parents() -> None:
    """PRD §11 / ADR-0004 D5+D7: augmentation may only see training-fold parents.

    This is the structural guarantee behind 'class-II augmentation constrained to
    training-fold parents' — a held-out parent is never loaded, so no sampler can
    reach it.
    """
    _require_real_artifacts()
    records, report = wd.load_corpus_records(training_fold_only=True)
    assert records, "training fold is empty"
    assert all(r.nested_train for r in records)
    assert report["n_records"] == len(records)
    assert report["synthetic_flank"] is False


def test_load_corpus_records_excludes_the_unanchored_records_and_reports_them() -> None:
    """The 3 non-`ok` P2-00 records are dropped by reason, never padded (§10.3)."""
    _require_real_artifacts()
    _, report = wd.load_corpus_records(training_fold_only=False)
    assert report["n_excluded"] == 3
    assert report["excluded_by_reason"] == {
        "unanchored:multi_hit": 2,
        "unanchored:unavailable": 1,
    }
    assert report["n_records"] == report["n_context_records"] - 3


def test_no_training_fold_record_is_excluded_for_lack_of_context() -> None:
    """Measured at P2-01: all 8,303 training-fold records admit an honest window."""
    _require_real_artifacts()
    _, report = wd.load_corpus_records(training_fold_only=True)
    assert report["n_excluded"] == 0
    assert report["excluded_by_reason"] == {}


def test_real_training_fold_is_class_ii_scarce_and_firmicutes_dominated() -> None:
    """The two facts that make both samplers load-bearing (PRD §8/§12).

    If this ever stops holding, the sampler defaults need revisiting — so pin the
    shape rather than let it drift silently.
    """
    _require_real_artifacts()
    _, report = wd.load_corpus_records(training_fold_only=True)
    klass = report["klass_counts"]
    phyla = report["phylum_counts"]
    n = report["n_records"]
    assert klass.get("II", 0) / n < 0.01, "class II is no longer scarce"
    assert phyla.get("Firmicutes", 0) / n > 0.90, "the Firmicutes skew has changed"


def test_the_real_curriculum_lifts_class_ii_well_above_its_natural_rate() -> None:
    """End-to-end: the sampler must materially enrich the 22 class-II loci."""
    _require_real_artifacts()
    records, _ = wd.load_corpus_records(training_fold_only=True)
    ds = wd.Stage1WindowDataset(records)
    w = ds.weights()
    is_ii = np.asarray([r.klass == "II" for r in records])
    natural = is_ii.mean()
    weighted = w[is_ii].sum() / w.sum()
    assert weighted > 5 * natural, f"class II enrichment {weighted / natural:.1f}x is too weak"


def test_the_real_curriculum_softens_the_firmicutes_skew() -> None:
    """PRD §12: balanced-phylum sampling exists to bound the Firmicutes overfit."""
    _require_real_artifacts()
    records, _ = wd.load_corpus_records(training_fold_only=True)
    w = wd.Stage1WindowDataset(records).weights()
    is_fw = np.asarray([r.phylum == "Firmicutes" for r in records])
    natural = is_fw.mean()
    weighted = w[is_fw].sum() / w.sum()
    assert weighted < natural, "balanced-phylum sampling did not reduce the Firmicutes share"


def test_curriculum_bounds_the_mass_on_any_single_locus() -> None:
    """The load-bearing guard on the sampling defaults (measured at P2-01).

    The three alpha terms compound across correlated axes, so an innocent-looking
    bump (0.25 -> 0.5 on each) sends 22.5% of all draws to TEN loci out of 8,303
    — the model would memorise them, and a class-II recall built on that is an
    artefact. This test bounds the concentration rather than the alpha, so it
    catches any parameterisation that re-creates the pathology, not just the one
    that did.
    """
    _require_real_artifacts()
    records, _ = wd.load_corpus_records(training_fold_only=True)
    w = wd.Stage1WindowDataset(records).weights()
    p = np.sort(w / w.sum())[::-1]
    top10 = float(p[:10].sum())
    assert top10 < 0.05, (
        f"the 10 highest-weighted loci absorb {top10:.1%} of the sampling mass "
        "— the curriculum has collapsed onto a handful of records"
    )
    assert float(p[0]) < 0.01, f"a single locus takes {p[0]:.2%} of the stream"


# ═══════════════════════════════════════════════════════════════════════════
# conf/data/stage1.yaml drift guard (ADR-0002 A5 precedent: code is authoritative)
# ═══════════════════════════════════════════════════════════════════════════
def _read_stage1_yaml() -> dict[str, str]:
    """Minimal dependency-free scalar reader (the coverage.py/decoys.py idiom).

    Avoids a yaml dependency so this guard runs in bare CI.
    """
    values: dict[str, str] = {}
    for line in wd.CONFIG_PATH.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        values[key.strip()] = val.strip()
    return values


def test_stage1_config_exists_and_is_not_a_hydra_primary() -> None:
    """A group option (`- /data: stage1`) must carry no package/defaults directive.

    Hydra only honours a `# @package ...` directive as a standalone comment line,
    so the check is line-anchored — prose mentioning the directive is fine.
    """
    assert wd.CONFIG_PATH.is_file(), f"missing {wd.CONFIG_PATH}"
    for raw in wd.CONFIG_PATH.read_text().splitlines():
        line = raw.strip()
        assert not line.startswith("# @package"), f"carries a package directive: {line}"
        assert not line.startswith("defaults:"), "group option carries a defaults list"


def test_stage1_config_echoes_the_code_constants_exactly() -> None:
    """A config edit alone must not be able to move the datamodule."""
    cfg = _read_stage1_yaml()
    default = wd.Stage1DataConfig()
    assert int(cfg["window_nt"]) == wd.WINDOW_NT == default.window_nt
    assert int(cfg["stride_nt"]) == wd.STRIDE_NT == default.stride_nt
    assert int(cfg["seed"]) == wd.DEFAULT_SEED == default.seed
    assert float(cfg["phylum_alpha"]) == wd.DEFAULT_PHYLUM_ALPHA == default.phylum_alpha
    assert float(cfg["klass_alpha"]) == wd.DEFAULT_KLASS_ALPHA == default.klass_alpha
    assert float(cfg["aa_alpha"]) == wd.DEFAULT_AA_ALPHA == default.aa_alpha
    assert cfg["offset_augmentation"] == "true" and default.offset_augmentation is True
    assert cfg["both_strands"] == "true" and default.both_strands is True
    assert int(cfg["pythonhashseed"]) == 0


def test_stage1_config_is_not_wired_as_a_default_anywhere_yet() -> None:
    """P2-04 wires it; until then nothing may silently depend on it."""
    from pathlib import Path

    for cfg in Path("conf/train").glob("*.yaml"):
        assert "/data: stage1" not in cfg.read_text(), f"{cfg} wires stage1 before P2-04"


# ═══════════════════════════════════════════════════════════════════════════
# torch tier — collate + DataLoader duck-typing
# ═══════════════════════════════════════════════════════════════════════════
def test_collate_windows_builds_embedding_ready_tensors() -> None:
    torch = pytest.importorskip("torch")
    ds = _dataset(n=4)
    batch = wd.collate_windows([ds[i] for i in range(4)])
    assert batch["input_ids"].shape == (4, 1024)
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].shape == (4, 1024)
    assert batch["labels"].dtype == torch.long
    assert batch["real_mask"].dtype == torch.bool
    assert len(batch["parent_record_id"]) == 4


def test_collate_windows_preserves_the_ignore_index() -> None:
    """The seg head consumes -100 directly; collate must not clobber it."""
    pytest.importorskip("torch")
    r = _record(lead=10, locus=300, trail=1024, clipped_start=True)
    ds = wd.Stage1WindowDataset([r], augment=False)
    w = ds.window_at(0)
    # Deterministic by construction: centred lead 362 > lead_flank 10, and the
    # record is clipped_start, so 352 nt are zero-flanked. Assert rather than
    # skip — a conditional skip here could only ever hide a broken fixture.
    assert w.zero_flanked, "fixture must exercise a zero-flanked window"
    batch = wd.collate_windows([ds[0]])
    assert (batch["labels"] == wd.IGNORE_INDEX).any()


def test_collate_windows_rejects_an_empty_batch() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="non-empty batch"):
        wd.collate_windows([])


def test_dataset_drives_a_real_dataloader_with_the_curriculum_sampler() -> None:
    """The dataset is map-style duck-typed: DataLoader needs only __len__/__getitem__.

    Proves the torch-free design actually composes with the real DataLoader +
    a weighted sampler, which is the whole point of not subclassing Dataset.
    """
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    ds = _dataset(n=10)
    loader = DataLoader(
        ds,
        batch_size=4,
        sampler=ds.sampler(num_samples=8),
        collate_fn=wd.collate_windows,
    )
    batches = list(loader)
    assert sum(b["input_ids"].shape[0] for b in batches) == 8
    assert batches[0]["input_ids"].shape[1] == 1024


def test_dataloader_oversampling_yields_distinct_variants_and_is_reproducible() -> None:
    """End-to-end proof of the occurrence fix, through the real DataLoader.

    A single-record dataset drawn 12 times is the oversampling limit case: the
    variants must differ (not 12 identical copies) while the whole sampled
    sequence stays reproducible under the same seed (CLAUDE.md §8.3).
    """
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    def sample() -> list[str]:
        ds = _dataset(n=1)
        loader = DataLoader(
            ds, batch_size=3, sampler=ds.sampler(num_samples=12), collate_fn=wd.collate_windows
        )
        return [rid for b in loader for rid in b["record_id"]]

    ids = sample()
    assert len(ids) == 12
    assert all(i.startswith("r0#w") for i in ids)
    assert len(set(ids)) > 1, "12 oversampled draws collapsed to one variant"
    assert sample() == ids, "the sampled sequence is not reproducible under a fixed seed"
