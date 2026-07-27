"""Unit tests for the promoted column-shuffle matched-control helpers (P2-10e-msa).

The three helpers were lifted verbatim from ``scripts/make_rscape_fixture.py`` so the runtime
homolog-MSA certification and the fixture minter share one implementation
([[promote-dont-duplicate-is-a-correctness-rule]]). These tests lock that the promotion is
*behaviour-identical* — the committed P2-10b fixtures must regenerate byte-for-byte — and that the
shuffle is a genuinely matched control (composition/width/depth/SS_cons preserved, correlation not).
Stdlib-only, so it runs in the bare-CI unit tier.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from tbox_finder.mining.msa_shuffle import (
    read_pfam_alignment,
    shuffle_alignment_columns,
    write_pfam_alignment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FX = REPO_ROOT / "tests" / "fixtures" / "rscape"
POSITIVE = FX / "classII_sub.sto"
SHUFFLED = FX / "classII_sub.shuffled.sto"
# Real mlocarna 2.0.1 output (P2-10e-msa job 760): 24 rows wrapped over 3 column-blocks.
INTERLEAVED = FX / "mlocarna_interleaved.sto"
FIXTURE_SHUFFLE_SEED = 20260721  # SEED + 1 in make_rscape_fixture.py (SEED = 20260720)


def test_read_write_round_trip_is_byte_identical() -> None:
    seqs, gc = read_pfam_alignment(POSITIVE)
    assert write_pfam_alignment(seqs, gc) == POSITIVE.read_text(encoding="utf-8")
    assert {"SS_cons", "RF"} <= set(gc)


def test_seeded_shuffle_reproduces_the_committed_twin_byte_for_byte() -> None:
    seqs, gc = read_pfam_alignment(POSITIVE)
    shuffled = shuffle_alignment_columns(seqs, seed=FIXTURE_SHUFFLE_SEED)
    assert write_pfam_alignment(shuffled, gc) == SHUFFLED.read_text(encoding="utf-8")


def test_shuffle_preserves_the_matched_invariants() -> None:
    seqs, _gc = read_pfam_alignment(POSITIVE)
    shuffled = shuffle_alignment_columns(seqs, seed=FIXTURE_SHUFFLE_SEED)
    # depth + names + width preserved
    assert [n for n, _ in shuffled] == [n for n, _ in seqs]
    assert {len(s) for _, s in shuffled} == {len(s) for _, s in seqs}
    # every column is a permutation of the positive's same column (composition identical)
    pos_cols = [Counter(c) for c in zip(*[s for _, s in seqs], strict=True)]
    shf_cols = [Counter(c) for c in zip(*[s for _, s in shuffled], strict=True)]
    assert pos_cols == shf_cols


def test_shuffle_actually_changes_the_alignment() -> None:
    # A matched control that equals the positive proves nothing; assert it differs somewhere.
    seqs, _gc = read_pfam_alignment(POSITIVE)
    shuffled = shuffle_alignment_columns(seqs, seed=FIXTURE_SHUFFLE_SEED)
    assert [s for _, s in shuffled] != [s for _, s in seqs]


def test_shuffle_is_deterministic_in_the_seed() -> None:
    seqs, _gc = read_pfam_alignment(POSITIVE)
    a = shuffle_alignment_columns(seqs, seed=123)
    b = shuffle_alignment_columns(seqs, seed=123)
    c = shuffle_alignment_columns(seqs, seed=124)
    assert a == b
    assert a != c  # a different seed gives a different permutation


def test_read_accepts_text_and_path() -> None:
    text = POSITIVE.read_text(encoding="utf-8")
    from_text = read_pfam_alignment(text)
    from_path = read_pfam_alignment(POSITIVE)
    assert from_text == from_path


def test_read_merges_interleaved_mlocarna_blocks() -> None:
    # Regression (P2-10e-msa job 760): mlocarna emits INTERLEAVED Stockholm — 24 rows wrapped over
    # 3 column-blocks (widths 120/120/49 = 289). The pre-fix "one line == one row" read produced
    # 24*3 = 72 ragged rows (48 wide + 24 short) and crashed certify_msa's shuffle with
    # `ValueError: zip() argument 49 is shorter than arguments 1-48`. The reader must merge by name.
    seqs, gc = read_pfam_alignment(INTERLEAVED)
    assert len(seqs) == 24  # merged rows, not 72
    assert len({n for n, _ in seqs}) == 24  # keyed on name (distinct)
    assert len({len(s) for _, s in seqs}) == 1  # every row full-width...
    assert len(seqs[0][1]) == 289  # ...= 120 + 120 + 49
    # #=GC SS_cons concatenated across all three blocks, not overwritten to the last fragment only.
    assert len(gc["SS_cons"]) == 289


def test_shuffle_runs_on_interleaved_mlocarna_output() -> None:
    # The exact end-to-end failure: certify_msa read the interleaved file then shuffled it. Assert
    # no crash and that the shuffle is still a matched control on real wrapped input.
    seqs, _gc = read_pfam_alignment(INTERLEAVED)
    shuffled = shuffle_alignment_columns(seqs, seed=20260726)
    assert [n for n, _ in shuffled] == [n for n, _ in seqs]
    assert {len(s) for _, s in shuffled} == {len(s) for _, s in seqs} == {289}
    pos_cols = [Counter(c) for c in zip(*[s for _, s in seqs], strict=True)]
    shf_cols = [Counter(c) for c in zip(*[s for _, s in shuffled], strict=True)]
    assert pos_cols == shf_cols  # matched control: per-column composition preserved
