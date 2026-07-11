"""Unit tests for the §9.1 decoy-pool generators (P0-30).

Stdlib-only (no pandas/biopython) so they run in the bare CI ``test`` env; the
pandas ``build`` path is covered by ``tests/golden/test_decoys_golden.py``. The
load-bearing generator invariant is
``test_dinucleotide_shuffle_preserves_composition`` (the pool is only a valid
dinucleotide null if the shuffle is exact).
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from tbox_finder import decoys


# --------------------------------------------------------------------------- #
# GC + FASTA primitives
# --------------------------------------------------------------------------- #
def test_gc_fraction():
    assert decoys.gc_fraction("GGCC") == 1.0
    assert decoys.gc_fraction("AATT") == 0.0
    assert math.isclose(decoys.gc_fraction("ATGC"), 0.5)
    assert decoys.gc_fraction("") == 0.0
    assert math.isclose(decoys.gc_fraction("augc"), 0.5)  # RNA + lowercase


def test_parse_write_fasta_roundtrip():
    text = ">seq1 desc\nACGT\nACGT\n>seq2\nTTTT\n"
    recs = decoys.parse_fasta(text)
    assert recs == [("seq1 desc", "ACGTACGT"), ("seq2", "TTTT")]
    # write→parse round-trips the (header, sequence) content
    assert decoys.parse_fasta(decoys.write_fasta(recs)) == recs


# --------------------------------------------------------------------------- #
# GC + length-matched generator
# --------------------------------------------------------------------------- #
def test_gc_matched_sequence_length_and_gc():
    rng = random.Random(0)
    seq = decoys.gc_matched_sequence(5000, 0.7, rng)
    assert len(seq) == 5000
    assert set(seq) <= set("ACGT")
    assert abs(decoys.gc_fraction(seq) - 0.7) < 0.03  # large-N law of large numbers


def test_gc_matched_sequence_is_deterministic():
    a = decoys.gc_matched_sequence(200, 0.5, random.Random(123))
    b = decoys.gc_matched_sequence(200, 0.5, random.Random(123))
    assert a == b


def test_gc_matched_sequence_validation():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        decoys.gc_matched_sequence(-1, 0.5, rng)
    with pytest.raises(ValueError):
        decoys.gc_matched_sequence(10, 1.5, rng)


# --------------------------------------------------------------------------- #
# Dinucleotide shuffle — the load-bearing null invariant
# --------------------------------------------------------------------------- #
_SEQS = [
    "AUGCAUGCAUGCGGCCAUAUAUAUCGCGCGCGAUAUGCGCUAGCUAGCUAGC",
    "GGGGGGCCCCCCAAAAAATTTTTT",
    "ACGTACGTACGTACGTTGCATGCATGCA",
    "AAAAAAAAAA",
    "AT",
    "GATTACA" * 6,
]


@pytest.mark.parametrize("seq", _SEQS)
def test_dinucleotide_shuffle_preserves_composition(seq):
    rng = random.Random(7)
    for _ in range(25):
        shuf = decoys.dinucleotide_shuffle(seq, rng)
        # exact dinucleotide composition preserved (the whole point of the pool)
        assert decoys.dinucleotide_counts(shuf) == decoys.dinucleotide_counts(seq)
        # ⇒ mononucleotide composition + length + endpoints preserved
        assert Counter(shuf) == Counter(seq)
        assert len(shuf) == len(seq)
        assert shuf[0] == seq[0] and shuf[-1] == seq[-1]


def test_dinucleotide_shuffle_is_deterministic():
    seq = _SEQS[0]
    a = decoys.dinucleotide_shuffle(seq, random.Random(42))
    b = decoys.dinucleotide_shuffle(seq, random.Random(42))
    assert a == b


def test_dinucleotide_shuffle_actually_permutes():
    # a structured sequence should usually move under shuffling (not a no-op identity)
    seq = "ACGTACGTACGTACGTACGTACGTTGCATGCATGCATGCA"
    rng = random.Random(1)
    outs = {decoys.dinucleotide_shuffle(seq, rng) for _ in range(20)}
    assert any(o != seq for o in outs)


def test_dinucleotide_shuffle_short_edges():
    assert decoys.dinucleotide_shuffle("", random.Random(0)) == ""
    assert decoys.dinucleotide_shuffle("A", random.Random(0)) == "A"


# --------------------------------------------------------------------------- #
# Corpus-derived pools + digest
# --------------------------------------------------------------------------- #
def _corpus():
    gc = [0.4, 0.5, 0.6, 0.55, 0.45]
    lengths = [120, 200, 180, 160, 140]
    seqs = [
        "ACGTACGTACGTGGCCAUAUACGCGCGATATATCGCG".replace("U", "T"),
        "GGCCGGCCAATTAATTCGCGCGCGATATATAT",
        "ATGCATGCGGGGCCCCATATATATCGCGCGCG",
        "TTTTAAAAGGGGCCCCACGTACGTTGCATGCA",
        "ACACACACGTGTGTGTATATATATCGCGCGCG",
    ]
    return gc, lengths, seqs


def test_build_corpus_pools_shape_and_determinism():
    gc, lengths, seqs = _corpus()
    kw = dict(seed=42, n_gc=10, n_dinuc_sources=5, dinuc_per_source=2)
    recs = decoys.build_corpus_pools(gc, lengths, seqs, **kw)
    pools = Counter(r["pool"] for r in recs)
    assert pools[decoys.POOL_GC] == 10
    assert pools[decoys.POOL_DINUC] == 10  # 5 sources × 2 shuffles
    # every dinuc record preserves its source's dinucleotide composition
    src_dinuc = [decoys.dinucleotide_counts(s) for s in seqs]
    for r in recs:
        if r["pool"] == decoys.POOL_DINUC:
            assert decoys.dinucleotide_counts(r["sequence"]) in src_dinuc
        assert r["accession"] is None  # synthetic pools carry no genomic coords
        assert r["length"] == len(r["sequence"])
    # deterministic construction → identical digest
    recs2 = decoys.build_corpus_pools(gc, lengths, seqs, **kw)
    assert decoys.decoys_digest(recs) == decoys.decoys_digest(recs2)
    assert len(decoys.decoys_digest(recs)) == 64


def test_build_corpus_pools_seed_changes_digest():
    gc, lengths, seqs = _corpus()
    kw = dict(n_gc=10, n_dinuc_sources=5, dinuc_per_source=2)
    d1 = decoys.decoys_digest(decoys.build_corpus_pools(gc, lengths, seqs, seed=1, **kw))
    d2 = decoys.decoys_digest(decoys.build_corpus_pools(gc, lengths, seqs, seed=2, **kw))
    assert d1 != d2


def test_build_corpus_pools_validation():
    with pytest.raises(ValueError):
        decoys.build_corpus_pools(
            [0.5], [100, 200], ["ACGT"], seed=1, n_gc=1, n_dinuc_sources=1, dinuc_per_source=1
        )
    with pytest.raises(ValueError):
        decoys.build_corpus_pools([], [], [], seed=1, n_gc=1, n_dinuc_sources=1, dinuc_per_source=1)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_read_config_parses_scalars(tmp_path):
    p = tmp_path / "decoys.yaml"
    p.write_text("seed: 7\nflank_nt: 25  # comment\ngc_background_n: 5\n")
    cfg = decoys.read_config(p)
    assert cfg.seed == 7
    assert cfg.flank_nt == 25
    assert cfg.gc_background_n == 5
    # unset keys fall back to the pins
    assert cfg.dinuc_per_source == decoys.DEFAULT_DINUC_PER_SOURCE


def test_read_config_missing_file_returns_defaults():
    cfg = decoys.read_config("/nonexistent/decoys.yaml")
    assert cfg.seed == decoys.provenance.DEFAULT_SEED
