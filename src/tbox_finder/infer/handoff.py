"""P3-13 — the PRD §6 alphabet handoff: a resolved DNA locus ± flank becomes Stage-2 RNA.

Stage 1 is DNA, Stage 2 is RNA. Once :mod:`tbox_finder.infer.strand` has oriented a locus, this
module carves its flanked span out of the scanned sequence, reverse-complements it if the locus
reads along the minus strand, and transcribes it T→U into the RNA string Stage 2 ingests
**sequence-only** — predicted or annotated structure is a §8/§11 auxiliary *target*, never a
Stage-2 input (PRD §6). An ambiguous locus produces one payload per carried strand, so it
survives on whichever strand is real.

Composition, not a fork
-----------------------
T→U is **not** re-derived. :func:`tbox_finder.stage2.tokenizer.transcribe` is already the
shipped §6 alphabet handoff — the same function
``tbox_finder.stage2.dataset`` transcribes the training corpus with — so a second
``.replace("T", "U")`` here would mean the loci Stage 2 scores at inference and the records it
was trained on could drift apart one edit at a time. That module is stdlib-only, so importing
it costs this module nothing on the bare CI path.

The reverse complement, however, IS written here, and deliberately
-------------------------------------------------------------------
The repo carries five reverse-complement implementations (``anchors.py``,
``data/flank_context.py``, ``data/window_dataset.py``, ``mining/homolog_msa.py``, plus an
id-space one in ``models/caduceus_backbone.py``) and **not one of them is safe for this
input**. Three are ``str.maketrans`` tables over ``ACGTN`` that leave anything else *unchanged*,
so an IUPAC ambiguity code (``R``, ``Y``, ``S``, ``W``, ``K``, ``M``, ``B``, ``D``, ``H``, ``V``)
survives reverse-complementation **uncomplemented** — a silently wrong base on the strand handed
to Stage 2. The fourth maps every unrecognised character to ``N``, destroying the code instead.
Biopython is not an option either: it is absent from ``envs/ml-dna.yml``, ``envs/ml-rna.yml``
and the CI install (so importing it would break both the P5 scan env and the bare-CI tier this
module runs in), and ``Bio.Seq.complement`` has the same silent-mangling behaviour by design —
its own tutorial shows a protein sequence ``"EVRNAK"`` complementing to ``"EBYNTM"`` without
complaint.

:func:`reverse_complement` therefore covers the **full IUPAC nucleotide alphabet** (NC-IUB 1985)
in both cases, and **fails closed** on anything else. Genome assemblies and MAG contigs really
do carry ambiguity codes; a scanner that mis-reads them on the minus strand produces a wrong
RNA that looks entirely ordinary. Promoting the other five to this one is a separate concern
(they have their own callers and tests) and is not attempted here — but this is the only one on
the Stage-1 → Stage-2 path.

Soft-masking is counted, not silently swallowed
------------------------------------------------
Lower-case input (soft-masked repeats, near-universal in genomic FASTA) complements correctly
and keeps its case through :func:`reverse_complement`; ``transcribe`` then upper-cases, because
the stored artifact should read as the RNA the model sees. The count of soft-masked positions
travels with the payload as ``n_soft_masked`` rather than being lost at the upper-case, and
non-ACGTU characters travel as ``n_ambiguous`` — the same recall-favouring "keep and flag"
discipline :mod:`tbox_finder.infer.locus` applies to zero-flanked and singly-covered positions.

``str``-only and torch-free; imports on the bare CI Tier-1 path.
PRD §6, §13.1; ADR-0005 D15.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

from tbox_finder.infer.locus import Locus
from tbox_finder.infer.strand import STRAND_MINUS, STRAND_PLUS, StrandCall
from tbox_finder.stage2 import tokenizer as rna_tokenizer

SCHEMA_VERSION = "1.0"
STEP = "P3-13"

#: The IUPAC **DNA** alphabet (NC-IUB 1985) → its complement, upper case.
#:
#: ``U`` is deliberately **absent**, and that is a fail-closed choice rather than an omission.
#: The input on this path is the scanned genomic DNA, and admitting ``U`` would make the table
#: non-involutive — ``T`` and ``U`` would both complement to ``A``, so ``A`` could not
#: complement back to both and a round trip would silently rewrite the sequence. An RNA string
#: arriving here means a caller has already transcribed and is about to transcribe again, which
#: is worth an exception rather than a quiet answer. Every remaining pair *is* its own inverse,
#: which ``test_complement_table_is_an_involution`` checks rather than trusting the
#: transcription.
IUPAC_COMPLEMENT_UPPER: dict[str, str] = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "R": "Y",  # A/G   -> T/C
    "Y": "R",  # C/T   -> G/A
    "S": "S",  # C/G   self-complementary
    "W": "W",  # A/T   self-complementary
    "K": "M",  # G/T   -> C/A
    "M": "K",  # A/C   -> T/G
    "B": "V",  # C/G/T -> G/C/A
    "V": "B",  # A/C/G -> T/G/C
    "D": "H",  # A/G/T -> T/C/A
    "H": "D",  # A/C/T -> T/G/A
    "N": "N",  # any
    "-": "-",  # alignment gap, kept in place rather than refused
}

#: Case-preserving complement table: a soft-masked base complements to a soft-masked base, so
#: the repeat annotation survives the strand flip and ``n_soft_masked`` stays meaningful.
IUPAC_COMPLEMENT: dict[str, str] = {
    **IUPAC_COMPLEMENT_UPPER,
    **{k.lower(): v.lower() for k, v in IUPAC_COMPLEMENT_UPPER.items()},
}

_COMPLEMENT_TABLE = str.maketrans(IUPAC_COMPLEMENT)

#: Characters that are unambiguously one base. Anything else in a payload is counted as
#: ambiguous, which is a fact about the contig, not an error.
_UNAMBIGUOUS = frozenset("ACGTUacgtu")


class HandoffError(ValueError):
    """Raised on a sequence this module refuses to transcribe, or a malformed span."""


def reverse_complement(sequence: str) -> str:
    """Reverse complement over the full IUPAC alphabet, case-preserving, **fail-closed**.

    Raises :class:`HandoffError` listing the offending characters rather than passing them
    through or folding them to ``N``. Both silent behaviours exist elsewhere in this repo and
    both produce a wrong minus-strand sequence that looks completely ordinary downstream — see
    the module docstring.
    """
    text = str(sequence)
    bad = sorted({ch for ch in text if ch not in IUPAC_COMPLEMENT})
    if bad:
        raise HandoffError(
            f"sequence carries {len(bad)} character(s) outside the IUPAC nucleotide alphabet: "
            f"{bad}. Refused rather than complemented to itself or to N — either would put a "
            "silently wrong base on the minus strand handed to Stage-2"
        )
    return text.translate(_COMPLEMENT_TABLE)[::-1]


def transcribe_to_rna(sequence: str, *, strand: str) -> str:
    """The §6 alphabet handoff for one span: orient, then DNA → RNA.

    ``strand="+"`` transcribes the span as given; ``strand="-"`` reverse-complements it first,
    so the returned RNA always reads 5′→3′ in the locus's own orientation. T→U and upper-casing
    are delegated to :func:`tbox_finder.stage2.tokenizer.transcribe`, the same function the
    Stage-2 training table was built with.
    """
    if strand == STRAND_MINUS:
        sequence = reverse_complement(sequence)
    elif strand != STRAND_PLUS:
        raise HandoffError(
            f"strand must be {STRAND_PLUS!r} or {STRAND_MINUS!r}, got {strand!r}; "
            "an ambiguous locus carries both, one payload each — it has no third value"
        )
    # Validate the alphabet on the plus strand too: a caller must not be able to get an
    # unchecked sequence into a Stage-2 payload just by resolving to '+'.
    bad = sorted({ch for ch in sequence if ch not in IUPAC_COMPLEMENT})
    if bad:
        raise HandoffError(
            f"sequence carries {len(bad)} character(s) outside the IUPAC nucleotide alphabet: "
            f"{bad}"
        )
    # Attribute call, not a ``from`` import: the delegation is then observable — a test can
    # substitute ``rna_tokenizer.transcribe`` and assert this path used it, so "does not
    # fork T→U" is checked rather than asserted in prose.
    return rna_tokenizer.transcribe(sequence)


@dataclass(frozen=True)
class Handoff:
    """One RNA payload handed to Stage-2 — a locus on one carried strand.

    **Sequence-only by construction** (PRD §6): the record has an ``rna`` string and no
    structure, dot-bracket or pairing field, and ``test_handoff_carries_no_structure`` asserts
    that over the live field set rather than by reading.

    Attributes
    ----------
    locus_index:
        Position of the source locus in the list handed to :func:`handoff_loci`, so a payload
        can be joined back to its :class:`~tbox_finder.infer.locus.Locus` and
        :class:`~tbox_finder.infer.strand.StrandCall`.
    strand:
        The strand **this payload** is on. For an ambiguous locus two payloads exist, one per
        strand, and both carry ``low_order_confidence=True``.
    low_order_confidence:
        The §13.1 flag, copied from the strand call so the payload is self-describing.
    start, end, length:
        The flanked half-open span in the frame of the **scanned** sequence — unchanged by the
        strand, so coordinates stay comparable between the two payloads of an ambiguous locus.
        The RNA of a ``"-"`` payload reads from ``end`` back to ``start``.
    rna:
        The transcribed sequence. ``len(rna) == length`` always: transcription is
        position-preserving, which ``test_transcription_is_length_preserving`` pins.
    n_soft_masked:
        Lower-case positions in the source span (soft-masked repeats). Counted before
        upper-casing, so the information survives into the payload.
    n_ambiguous:
        Positions that are not unambiguously one base (``N`` and the IUPAC ambiguity codes).
        A large count on a novel locus is a reason to distrust it, not a reason to drop it.
    """

    locus_index: int
    strand: str
    low_order_confidence: bool
    start: int
    end: int
    length: int
    rna: str
    n_soft_masked: int
    n_ambiguous: int


def handoff_loci(
    sequence: str,
    loci: Sequence[Locus],
    calls: Sequence[StrandCall],
) -> list[Handoff]:
    """Every locus × every strand it carries → the Stage-2 RNA payloads.

    A resolved locus yields one payload; an ambiguous one yields two (``+`` then ``-``), which
    is D15's both-strand carry-through: the locus is never dropped for being under-determined,
    and a mis-resolution degrades to a bounded false negative rather than a false novelty claim.

    ``sequence`` is the DNA that was scanned, in the same frame the loci's coordinates are in.

    Raises
    ------
    HandoffError
        On a ``loci``/``calls`` length mismatch (which would silently pair a locus with another
        locus's strand call), a span running past the end of ``sequence``, or a sequence
        carrying characters outside the IUPAC alphabet.
    """
    if len(loci) != len(calls):
        raise HandoffError(
            f"got {len(loci)} loci and {len(calls)} strand calls; they are paired by position "
            "and a mismatch would hand Stage-2 a locus oriented by another locus's evidence"
        )
    text = str(sequence)
    payloads: list[Handoff] = []
    for index, (locus, call) in enumerate(zip(loci, calls, strict=True)):
        if locus.end > len(text):
            raise HandoffError(
                f"locus {index} spans [{locus.start}, {locus.end}) but the sequence is "
                f"{len(text)} nt; this is not the sequence the locus was called from"
            )
        span = text[locus.start : locus.end]
        n_soft_masked = sum(1 for ch in span if ch.islower())
        n_ambiguous = sum(1 for ch in span if ch not in _UNAMBIGUOUS)
        for strand in call.strands:
            payloads.append(
                Handoff(
                    locus_index=index,
                    strand=strand,
                    low_order_confidence=call.low_order_confidence,
                    start=locus.start,
                    end=locus.end,
                    length=locus.length,
                    rna=transcribe_to_rna(span, strand=strand),
                    n_soft_masked=n_soft_masked,
                    n_ambiguous=n_ambiguous,
                )
            )
    return payloads


def handoff_is_sequence_only(record: type = Handoff) -> bool:
    """True iff the Stage-2 payload carries no structure field (PRD §6).

    RiNALMo ingests **sequence only**; predicted or annotated structure is the §8/§11 auxiliary
    *target*, and feeding it in as a Stage-2 input would make the structure-consistency
    objective circular. Re-derived from the live field set so adding a ``dot_bracket`` /
    ``pairing`` / ``structure`` field to :class:`Handoff` makes this return False, rather than
    the contract living only in prose.
    """
    banned = ("structure", "dot_bracket", "dotbracket", "pairing", "ss_cons", "secondary")
    names = {f.name.lower() for f in fields(record)}
    return not any(any(token in name for token in banned) for name in names)


__all__ = [
    "IUPAC_COMPLEMENT",
    "IUPAC_COMPLEMENT_UPPER",
    "SCHEMA_VERSION",
    "STEP",
    "Handoff",
    "HandoffError",
    "handoff_is_sequence_only",
    "handoff_loci",
    "reverse_complement",
    "transcribe_to_rna",
]
