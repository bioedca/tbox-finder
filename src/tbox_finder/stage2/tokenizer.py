"""RiNALMo RNA tokenizer adapter — a pinned, dependency-free mirror of ``RnaTokenizer``.

**Why a pure-Python adapter rather than calling ``multimolecule`` directly.** The
Stage-2 supervised dataset (P3-01) is a data-pipeline artifact: it is assembled by a
``data.smk``-class LOCAL rule in ``envs/data.yml`` like every other processed table.
``multimolecule`` exists **only** in ``envs/ml-rna.yml`` (the transformers-5 half of the
ADR-0002 A4 env split) and drags in ``deepspeed`` at import (needing ``CUDA_HOME``), so
importing it in the dataset builder would force a GPU-stack env on a CPU table build and
make the golden test unrunnable in CI. RiNALMo's alphabet is single-nucleotide and tiny
(28 tokens), so the faithful thing to do is **pin the vocabulary** here and *prove* the
pin by parity against the live tokenizer.

**The pin is not trusted on faith.** :func:`load_reference_tokenizer` loads the real
``RnaTokenizer`` at the ADR-0002 A9 pinned revision, and
``tests/unit/test_stage2_tokenizer.py::test_pinned_vocab_matches_live_tokenizer``
asserts (a) :data:`VOCAB` equals ``tokenizer.get_vocab()`` exactly and (b)
:func:`encode`/:func:`decode` agree with it token-for-token on real T-box RNA. That test
runs in ``tbox-ml-rna``; in CI (no ``multimolecule``) it skips, but :data:`VOCAB_DIGEST`
is re-derived and asserted there, so a hand-edit of the table cannot pass unnoticed —
the two guards fail in different environments, and neither is vacuous on its own.

**Alphabet semantics** (measured against ``multimolecule`` 0.1.0's ``RnaTokenizer`` at
``multimolecule/rinalmo-giga @2a71f6f9…`` — see the P3-01 dev-log stanza):

* input is upper-cased, and ``T``/``t`` map to the ``U`` id (DNA is accepted, but PRD §6
  requires an explicit T→U transcription at the Stage-1→Stage-2 handoff, so callers go
  through :func:`transcribe` and the dataset stores RNA);
* the 15 IUPAC nucleotide codes, the gap/pad glyphs ``- . * | ? X I`` and ``N`` are all
  in-vocabulary;
* anything else maps to ``<unk>`` (id 3) — never dropped, so token length always equals
  sequence length and a per-nucleotide target stays index-aligned.

**Context length.** ``RiNALMoConfig.max_position_embeddings`` is 1024 at the pinned
revision, and ``RnaTokenizer`` wraps the sequence in ``<cls>``/``<eos>``, so at most
:data:`MAX_NUCLEOTIDE_TOKENS` = 1022 nucleotides fit — the PRD §10.2 figure, re-derived
here from the config rather than copied from prose.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

from tbox_finder.eval.rinalmo_parity import REPO_ID, REVISION

__all__ = [
    "CLS_ID",
    "EOS_ID",
    "MASK_ID",
    "MAX_NUCLEOTIDE_TOKENS",
    "MAX_POSITION_EMBEDDINGS",
    "NULL_ID",
    "N_FLANKING_SPECIAL_TOKENS",
    "PAD_ID",
    "REPO_ID",
    "REVISION",
    "UNK_ID",
    "VOCAB",
    "VOCAB_DIGEST",
    "assert_within_context",
    "decode",
    "encode",
    "id_to_token",
    "load_reference_tokenizer",
    "token_length",
    "transcribe",
    "vocab_digest",
    "within_context",
]


#: RiNALMo's standard RNA alphabet, verbatim from ``RnaTokenizer.get_vocab()`` at
#: ``multimolecule/rinalmo-giga`` revision :data:`REVISION` (``multimolecule`` 0.1.0).
#: Insertion order **is** id order; the parity test asserts both.
VOCAB: dict[str, int] = {
    "<pad>": 0,
    "<cls>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "<null>": 5,
    "A": 6,
    "C": 7,
    "G": 8,
    "U": 9,
    "N": 10,
    "R": 11,
    "Y": 12,
    "S": 13,
    "W": 14,
    "K": 15,
    "M": 16,
    "B": 17,
    "D": 18,
    "H": 19,
    "V": 20,
    "I": 21,
    "X": 22,
    "|": 23,
    ".": 24,
    "*": 25,
    "-": 26,
    "?": 27,
}

PAD_ID = VOCAB["<pad>"]
CLS_ID = VOCAB["<cls>"]
EOS_ID = VOCAB["<eos>"]
UNK_ID = VOCAB["<unk>"]
MASK_ID = VOCAB["<mask>"]
NULL_ID = VOCAB["<null>"]

#: The special tokens ``RnaTokenizer`` never emits from sequence content. ``decode``
#: strips exactly these under ``skip_special_tokens``.
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>")

_ID_TO_TOKEN: dict[int, str] = {i: t for t, i in VOCAB.items()}

#: ``RiNALMoConfig.max_position_embeddings`` at :data:`REVISION`.
MAX_POSITION_EMBEDDINGS = 1024
#: ``<cls>`` + ``<eos>``.
N_FLANKING_SPECIAL_TOKENS = 2
#: The longest nucleotide run RiNALMo can ingest in one forward (PRD §10.2's "1022").
MAX_NUCLEOTIDE_TOKENS = MAX_POSITION_EMBEDDINGS - N_FLANKING_SPECIAL_TOKENS

#: sha256 over the canonical serialisation of :data:`VOCAB`. Locked by a stdlib-only
#: test so the table cannot be edited in an env where the parity test skips.
VOCAB_DIGEST = "ac3cff22ff7eee31923e5b921470be247314794f3d283b3bc01f26049d3902b4"

# Upper-casing + T→U is exactly what the live tokenizer's normaliser does; doing it in
# one table keeps encode() a single dict lookup per character.
_NORMALISE: dict[int, str] = {ord(c.lower()): c for c in VOCAB if len(c) == 1}
_NORMALISE[ord("t")] = "U"
_NORMALISE[ord("T")] = "U"


def vocab_digest(vocab: dict[str, int] | None = None) -> str:
    """sha256 of the canonical ``token -> id`` serialisation (sorted, no whitespace)."""
    table = VOCAB if vocab is None else vocab
    payload = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def transcribe(sequence: str) -> str:
    """DNA → RNA: upper-case and T→U (PRD §6 alphabet handoff).

    Applied explicitly at assembly time rather than relying on the tokenizer's own
    normalisation, because the stored artifact is the RNA the model sees and a reviewer
    must be able to read it as such.
    """
    return str(sequence).upper().replace("T", "U")


def encode(sequence: str, *, add_special_tokens: bool = True) -> list[int]:
    """Token ids for ``sequence``, one id per character (plus ``<cls>``/``<eos>``).

    Out-of-alphabet characters become ``<unk>`` rather than being dropped, so
    ``len(encode(s, add_special_tokens=False)) == len(s)`` always holds and a
    per-nucleotide target stays aligned to the token axis.
    """
    body = [VOCAB.get(_NORMALISE.get(ord(ch), ch.upper()), UNK_ID) for ch in str(sequence)]
    if add_special_tokens:
        return [CLS_ID, *body, EOS_ID]
    return body


def decode(ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
    """Ids → the bare sequence string (no separators).

    ``RnaTokenizer.decode`` space-joins its tokens; this returns the contiguous
    sequence, which is what a round-trip check and a FASTA writer both want. An id
    outside the vocabulary is an error, not a silent ``<unk>`` — the only way to
    produce one is a corrupted artifact.
    """
    out: list[str] = []
    for i in ids:
        key = int(i)
        try:
            token = _ID_TO_TOKEN[key]
        except KeyError:
            raise ValueError(
                f"id {key} is outside the RiNALMo vocabulary (0-{len(VOCAB) - 1})"
            ) from None
        if skip_special_tokens and token in SPECIAL_TOKENS:
            continue
        out.append(token)
    return "".join(out)


def id_to_token(token_id: int) -> str:
    """The vocabulary token for ``token_id`` (raises on an out-of-vocabulary id)."""
    try:
        return _ID_TO_TOKEN[int(token_id)]
    except KeyError:
        raise ValueError(
            f"id {token_id} is outside the RiNALMo vocabulary (0-{len(VOCAB) - 1})"
        ) from None


def token_length(sequence: str, *, add_special_tokens: bool = True) -> int:
    """Token count without materialising the ids."""
    n = len(str(sequence))
    return n + N_FLANKING_SPECIAL_TOKENS if add_special_tokens else n


def within_context(sequence: str) -> bool:
    """Does ``sequence`` (plus ``<cls>``/``<eos>``) fit RiNALMo's 1024 positions?"""
    return len(str(sequence)) <= MAX_NUCLEOTIDE_TOKENS


def assert_within_context(sequence: str, *, row_id: str) -> None:
    """Fail loud when a locus exceeds the context window.

    Truncating instead would silently drop the 3′ elements (terminator / discriminator)
    that the boundary and regulatory-mode heads are trained on, and the resulting metric
    would look like a modelling result rather than a clipped input (CLAUDE.md §10.3).
    """
    n = len(str(sequence))
    if n > MAX_NUCLEOTIDE_TOKENS:
        raise ValueError(
            f"{row_id}: {n} nt exceeds RiNALMo's {MAX_NUCLEOTIDE_TOKENS}-nucleotide context "
            f"(max_position_embeddings {MAX_POSITION_EMBEDDINGS} minus "
            f"{N_FLANKING_SPECIAL_TOKENS} special tokens)"
        )


def load_reference_tokenizer(*, revision: str = REVISION) -> Any:
    """The live ``multimolecule`` ``RnaTokenizer`` — the parity reference for :data:`VOCAB`.

    Lazy-imports ``multimolecule`` (env-split precedent, :mod:`tbox_finder.eval.rinalmo_parity`)
    and refuses any revision but the ADR-0002 A9 pin, so a parity "pass" can never have
    been measured against a different checkpoint.
    """
    from tbox_finder.eval.rinalmo_parity import _require_pinned_revision

    _require_pinned_revision(revision)
    from multimolecule import RnaTokenizer  # noqa: PLC0415  (lazy: ml-rna-only dependency)

    return RnaTokenizer.from_pretrained(REPO_ID, revision=revision)


def reference_vocab(tokenizer: Any) -> dict[str, int]:
    """``tokenizer.get_vocab()`` as a plain ``dict[str, int]`` for exact comparison."""
    return {str(k): int(v) for k, v in tokenizer.get_vocab().items()}


def encode_batch(sequences: Sequence[str], *, add_special_tokens: bool = True) -> list[list[int]]:
    """:func:`encode` over a sequence of strings (no padding — that is a collator's job)."""
    return [encode(s, add_special_tokens=add_special_tokens) for s in sequences]
