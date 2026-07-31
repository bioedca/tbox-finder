"""Unit tests for the pinned RiNALMo tokenizer adapter (P3-01; PRD §6, §10.2).

Two guards, deliberately failing in **different** environments so neither is vacuous:

* everything below the parity test is stdlib-only and runs in bare CI — it locks the
  vocabulary digest, the id assignments, the round-trip and the 1022-nucleotide context
  bound against a hand-authored alphabet fixture;
* :func:`test_pinned_vocab_matches_live_tokenizer` loads the real ``multimolecule``
  ``RnaTokenizer`` at the ADR-0002 A9 pinned revision and asserts the pin **is** that
  tokenizer. It skips where ``multimolecule`` is absent (bare CI, the ``data`` env) and
  runs in ``tbox-ml-rna``; the P3-01 dev-log stanza records the run.

A hand-edit of :data:`tokenizer.VOCAB` therefore fails the digest test in CI even when
the parity test skips, and a *silent upstream change* to the mirror fails parity in
``tbox-ml-rna`` even though the digest still matches its own table.

The imp.md P3-01 gate names "the 9-PDB + hand-checked fixture" for the round-trip. Those
P0-21 fixtures record **element extents and window lengths, not sequences** (the
depositions were transcribed for boundary coordinates), so they cannot supply RNA to
round-trip. They are used here for what they can assert — that the token axis stays
index-aligned with the per-nucleotide label axis at exactly those window lengths, which
is what a per-nucleotide target depends on — and the round-trip itself runs over the
committed 100-record ingest slice of **real T-box loci** plus the alphabet fixture below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder.stage2 import tokenizer as tok

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_PDB = _FIXTURES / "pdb_element_extents" / "pdb_element_extents.json"
_HANDCHECKED = _FIXTURES / "pdb_element_extents" / "synthetic_handchecked.json"
_INGEST_CSV = _FIXTURES / "ingest_sample" / "Master_tboxes_sample.csv"

#: Hand-authored alphabet cases: every glyph the adapter claims to map, plus the
#: normalisations (case, T→U) and the out-of-alphabet fallback.
_ALPHABET_CASES: tuple[tuple[str, list[int]], ...] = (
    ("ACGU", [6, 7, 8, 9]),
    ("acgu", [6, 7, 8, 9]),
    ("ACGT", [6, 7, 8, 9]),  # DNA in → U ids out
    ("acgt", [6, 7, 8, 9]),
    ("N", [10]),
    ("RYSWKM", [11, 12, 13, 14, 15, 16]),
    ("BDHV", [17, 18, 19, 20]),
    ("IX", [21, 22]),
    ("|.*-?", [23, 24, 25, 26, 27]),
    ("Z@1", [3, 3, 3]),  # out of alphabet → <unk>, never dropped
)


def test_vocab_digest_is_pinned() -> None:
    """The committed digest must equal the digest of the committed table."""
    assert tok.vocab_digest() == tok.VOCAB_DIGEST
    assert len(tok.VOCAB) == 28


def test_vocab_ids_are_contiguous_and_unique() -> None:
    ids = sorted(tok.VOCAB.values())
    assert ids == list(range(len(tok.VOCAB)))
    assert len(set(tok.VOCAB)) == len(tok.VOCAB)
    assert list(tok.VOCAB.values()) == list(range(len(tok.VOCAB))), "insertion order is id order"
    assert (tok.PAD_ID, tok.CLS_ID, tok.EOS_ID, tok.UNK_ID) == (0, 1, 2, 3)


def test_context_bound_is_derived_not_asserted() -> None:
    """1022 must be the config's 1024 minus the two flanking specials, not a magic number."""
    assert tok.MAX_POSITION_EMBEDDINGS == 1024
    assert tok.N_FLANKING_SPECIAL_TOKENS == 2
    assert tok.MAX_NUCLEOTIDE_TOKENS == 1022
    assert tok.token_length("A" * 1022) == 1024
    assert tok.within_context("A" * 1022)
    assert not tok.within_context("A" * 1023)


def test_assert_within_context_refuses_an_over_long_locus() -> None:
    tok.assert_within_context("A" * 1022, row_id="ok")
    with pytest.raises(ValueError, match="exceeds RiNALMo"):
        tok.assert_within_context("A" * 1023, row_id="too-long")


def test_transcribe_is_upper_and_t_to_u() -> None:
    assert tok.transcribe("acgtn") == "ACGUN"
    assert tok.transcribe("ACGTN") == "ACGUN"
    assert "T" not in tok.transcribe("TTTT")


@pytest.mark.parametrize(("text", "expected"), _ALPHABET_CASES)
def test_alphabet_fixture_encodes_exactly(text: str, expected: list[int]) -> None:
    assert tok.encode(text, add_special_tokens=False) == expected
    assert tok.encode(text) == [tok.CLS_ID, *expected, tok.EOS_ID]


def test_encode_never_drops_a_character() -> None:
    """Token count must equal character count, or a per-nucleotide target de-synchronises."""
    for text, _ in _ALPHABET_CASES:
        assert len(tok.encode(text, add_special_tokens=False)) == len(text)
        assert tok.token_length(text, add_special_tokens=False) == len(text)


def test_round_trip_on_the_alphabet_fixture() -> None:
    for text, _ in _ALPHABET_CASES:
        rna = tok.transcribe(text)
        if "Z" in text or "@" in text or "1" in text:
            continue  # <unk> is lossy by construction; covered by the next test
        assert tok.decode(tok.encode(rna)) == rna


def test_unknown_characters_decode_as_unk_not_silently() -> None:
    ids = tok.encode("Z", add_special_tokens=False)
    assert ids == [tok.UNK_ID]
    # skip_special_tokens drops <unk>; keeping specials makes the loss visible.
    assert tok.decode(ids, skip_special_tokens=True) == ""
    assert tok.decode(ids, skip_special_tokens=False) == "<unk>"


def test_decode_refuses_an_out_of_vocabulary_id() -> None:
    with pytest.raises(ValueError, match="outside the RiNALMo vocabulary"):
        tok.decode([len(tok.VOCAB)])
    with pytest.raises(ValueError, match="outside the RiNALMo vocabulary"):
        tok.id_to_token(-1)


def test_decode_refuses_a_non_integer_id() -> None:
    """``int(6.9) == 6`` — a float id must not silently decode as the token at 6.

    (CodeRabbit r2, PR #93.) NumPy integers, which is what a parquet round-trip yields,
    must keep working.
    """
    assert int(6.9) == 6, "the premise: int() truncates rather than refusing"
    for bad in (6.9, "6", None):
        with pytest.raises(ValueError, match="is not an integer"):
            tok.decode([bad])
        with pytest.raises(ValueError, match="is not an integer"):
            tok.id_to_token(bad)
    np = pytest.importorskip("numpy")
    assert tok.decode([np.int64(6), np.int32(7)]) == "AC"


def test_token_axis_stays_aligned_at_the_pdb_window_lengths() -> None:
    """The 9 PDB + 2 hand-checked cases: one token per labelled nucleotide, exactly.

    A codon/nmer tokenizer, or one that dropped a glyph, would break the index identity
    the per-nucleotide boundary target depends on — these are the same window lengths
    ``tests/unit/test_label_derivation.py`` pins the label vectors at.
    """
    entries = json.loads(_PDB.read_text())["entries"]
    entries += json.loads(_HANDCHECKED.read_text())["entries"]
    assert len(entries) == 11, "9 crystal depositions + 2 hand-checked cases"
    for entry in entries:
        window = int(entry["window_length"])
        n_labelled = sum(int(n) for _, n in entry["expected_runs"])
        assert n_labelled == window, f"{entry['name']}: label vector does not span its window"
        probe = "ACGU" * window
        ids = tok.encode(probe[:window], add_special_tokens=False)
        assert len(ids) == window == n_labelled


def test_round_trip_on_real_tbox_loci() -> None:
    """Real T-box RNA from the committed 100-record ingest slice round-trips exactly."""
    pytest.importorskip("pandas")
    from tbox_finder import ingest

    raw = ingest.read_raw(_INGEST_CSV)
    clean = ingest.clean(raw, expect_records=100, expect_named_cols=None)
    sequences = [str(s) for s in clean["FASTA_sequence"]]
    assert len(sequences) == 100
    for dna in sequences:
        rna = tok.transcribe(dna)
        ids = tok.encode(rna)
        assert ids[0] == tok.CLS_ID and ids[-1] == tok.EOS_ID
        assert len(ids) == len(rna) + 2
        assert tok.decode(ids) == rna
        assert tok.within_context(rna)


def test_pinned_vocab_matches_live_tokenizer() -> None:
    """Parity: the pinned table IS ``RnaTokenizer``'s at the ADR-0002 A9 revision.

    Runs in ``tbox-ml-rna`` (needs ``CUDA_HOME``; ``multimolecule`` pulls ``deepspeed``).
    """
    pytest.importorskip("multimolecule")
    live = tok.load_reference_tokenizer()
    assert tok.reference_vocab(live) == tok.VOCAB

    probes = ["ACGUNACGU", "acgtn", "GGGAAACCC", "AUGCAUGCAUGC", "NNNN", "RYKM"]
    for probe in probes:
        rna = tok.transcribe(probe)
        assert tok.encode(rna) == list(live(rna)["input_ids"])
        assert tok.encode(rna, add_special_tokens=False) == list(
            live(rna, add_special_tokens=False)["input_ids"]
        )
        # live decode space-joins its tokens; the adapter returns the bare sequence.
        assert tok.decode(tok.encode(rna)) == live.decode(
            live(rna)["input_ids"], skip_special_tokens=True
        ).replace(" ", "")


def test_reference_tokenizer_refuses_an_unpinned_revision() -> None:
    pytest.importorskip("multimolecule")
    with pytest.raises(ValueError, match="code-pinned REVISION"):
        tok.load_reference_tokenizer(revision="main")
