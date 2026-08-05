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

Validation is per **emitted span**; the rest of the contig is counted
---------------------------------------------------------------------
:func:`handoff_loci` refuses an emitted span that carries a character outside the IUPAC
alphabet — every byte that reaches Stage 2 has been through :func:`require_iupac`. It does
**not** refuse the scanned sequence as a whole, and that asymmetry is deliberate: a contig is
megabases long and a locus is ~200 nt, so validating the whole contig would let one stray
character anywhere veto every locus called on it. That is anti-recall on exactly the MAG
contigs PRD §6 targets, and unrecoverable at this layer — the caller has no way to say "keep
the clean loci". So the out-of-alphabet characters outside every emitted span are **counted**
and reported on :class:`HandoffResult` rather than raised on: a filthy contig becomes visible
without becoming fatal. ``n_outside_alphabet_in_spans`` is the same count restricted to the
union of the emitted spans and is therefore ``0`` on every successful return — it is
*measured*, not asserted, so it is the tripwire that goes non-zero the moment the per-span
refusal stops firing.

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


def count_outside_alphabet(sequence: str) -> int:
    """Characters in ``sequence`` outside the IUPAC nucleotide alphabet — counted, not refused.

    The measurement half of :func:`require_iupac`, sharing its alphabet (``IUPAC_COMPLEMENT``)
    so the two can never disagree about what "outside" means. Used on the parts of a scanned
    contig that no payload carries, where refusing would cost every locus on the contig.

    Counted through the distinct out-of-alphabet characters rather than per position: this runs
    over whole contigs at P5 scan scale, and ``set`` + ``str.count`` stay in C where a
    per-character Python loop would not.
    """
    text = str(sequence)
    unknown = set(text) - set(IUPAC_COMPLEMENT)
    return sum(text.count(ch) for ch in unknown)


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans, sorted and unioned.

    Loci are disjoint as :func:`tbox_finder.infer.locus.construct_loci` emits them, but a
    non-zero ``flank`` can push two neighbours into overlap. Counting per span would then
    double-count a shared position and ``n_outside_alphabet_in_spans <= n_outside_alphabet``
    — the invariant that makes the pair a partition — would stop holding.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def require_iupac(sequence: str) -> str:
    """Return ``sequence`` as ``str``, refusing any character outside the IUPAC DNA alphabet.

    The single allowlist check for this module, so the two call sites cannot drift into
    different messages — or, worse, into different alphabets. Refuses rather than passing the
    character through or folding it to ``N``: both silent behaviours exist elsewhere in this
    repo and both produce a wrong strand that looks completely ordinary downstream (see the
    module docstring).
    """
    text = str(sequence)
    bad = sorted({ch for ch in text if ch not in IUPAC_COMPLEMENT})
    if bad:
        raise HandoffError(
            f"sequence carries {len(bad)} character(s) outside the IUPAC nucleotide alphabet: "
            f"{bad}. Refused rather than complemented to itself or to N — either would put a "
            "silently wrong base on the strand handed to Stage-2"
        )
    return text


def reverse_complement(sequence: str) -> str:
    """Reverse complement over the full IUPAC alphabet, case-preserving, **fail-closed**.

    Validates through :func:`require_iupac`, so a direct caller gets the same refusal the
    handoff does.
    """
    return require_iupac(sequence).translate(_COMPLEMENT_TABLE)[::-1]


def transcribe_to_rna(sequence: str, *, strand: str) -> str:
    """The §6 alphabet handoff for one span: orient, then DNA → RNA.

    ``strand="+"`` transcribes the span as given; ``strand="-"`` reverse-complements it first,
    so the returned RNA always reads 5′→3′ in the locus's own orientation. T→U and upper-casing
    are delegated to :func:`tbox_finder.stage2.tokenizer.transcribe`, the same function the
    Stage-2 training table was built with.
    """
    if strand not in (STRAND_PLUS, STRAND_MINUS):
        raise HandoffError(
            f"strand must be {STRAND_PLUS!r} or {STRAND_MINUS!r}, got {strand!r}; "
            "an ambiguous locus carries both, one payload each — it has no third value"
        )
    # Checked on BOTH strands: a caller must not be able to get an unchecked sequence into a
    # Stage-2 payload just by resolving to '+'. One helper, so the plus and minus paths cannot
    # drift into different alphabets or different messages.
    sequence = require_iupac(sequence)
    if strand == STRAND_MINUS:
        sequence = reverse_complement(sequence)
    # Attribute call, not a ``from`` import: the delegation is then observable — a test can
    # substitute ``rna_tokenizer.transcribe`` and assert this path used it, so "does not
    # fork T→U" is checked rather than asserted in prose.
    return rna_tokenizer.transcribe(sequence)


@dataclass(frozen=True)
class Handoff:
    """One RNA payload handed to Stage-2 — a locus on one carried strand.

    **Sequence-only by construction** (PRD §6): the record has an ``rna`` string and no
    structure, dot-bracket or pairing field, and ``test_handoff_is_sequence_only`` asserts
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


@dataclass(frozen=True)
class HandoffResult:
    """The payloads of one :func:`handoff_loci` call, plus what the rest of the contig was.

    Attributes
    ----------
    payloads:
        One :class:`Handoff` per (locus, carried strand), in locus order then ``+`` before
        ``-``. The only thing that reaches Stage 2.
    n_outside_alphabet:
        Characters in the **whole scanned sequence** outside the IUPAC nucleotide alphabet.
        A fact about the contig, reported rather than raised on (see the module docstring):
        one stray byte must not veto the clean loci around it. A large count is a reason to
        distrust what was called on this contig.
    n_outside_alphabet_in_spans:
        The same count restricted to the **union** of the emitted spans, so
        ``n_outside_alphabet - n_outside_alphabet_in_spans`` is the part no payload carries.
        ``0`` on every successful return, because a span carrying one raises instead — and it
        is *measured* rather than hardcoded, so it is the tripwire that goes non-zero if the
        per-span refusal ever stops firing.
    """

    payloads: tuple[Handoff, ...]
    n_outside_alphabet: int
    n_outside_alphabet_in_spans: int


def handoff_loci(
    sequence: str,
    loci: Sequence[Locus],
    calls: Sequence[StrandCall],
) -> HandoffResult:
    """Every locus × every strand it carries → the Stage-2 RNA payloads.

    A resolved locus yields one payload; an ambiguous one yields two (``+`` then ``-``), which
    is D15's both-strand carry-through: the locus is never dropped for being under-determined,
    and a mis-resolution degrades to a bounded false negative rather than a false novelty claim.

    ``sequence`` is the DNA that was scanned, in the same frame the loci's coordinates are in.

    Returns a :class:`HandoffResult`, **not** a bare list: the out-of-alphabet content of the
    parts of ``sequence`` that no payload carries is a property of this call and has nowhere
    else to live.

    Raises
    ------
    HandoffError
        On a ``loci``/``calls`` length mismatch (which would silently pair a locus with another
        locus's strand call), a span running past the end of ``sequence``, or an **emitted
        span** carrying characters outside the IUPAC alphabet. Out-of-alphabet characters
        outside every emitted span are **counted**, not refused — refusing a whole contig for
        one stray byte would drop every locus on it, which is anti-recall on exactly the MAG
        contigs this path targets. See ``n_outside_alphabet`` on the result.
    """
    if len(loci) != len(calls):
        raise HandoffError(
            f"got {len(loci)} loci and {len(calls)} strand calls; they are paired by position "
            "and a mismatch would hand Stage-2 a locus oriented by another locus's evidence"
        )
    text = str(sequence)
    payloads: list[Handoff] = []
    emitted_spans: list[tuple[int, int]] = []
    for index, (locus, call) in enumerate(zip(loci, calls, strict=True)):
        if not (0 <= locus.start <= locus.end <= len(text)):
            raise HandoffError(
                f"locus {index} spans [{locus.start}, {locus.end}) but the sequence is "
                f"{len(text)} nt; this is not the sequence the locus was called from. "
                "Refused rather than sliced: a negative start silently wraps from the far end "
                "of the string and a reversed span silently yields nothing, so either would "
                "hand Stage-2 a sequence that is not the one the coordinates name"
            )
        span = text[locus.start : locus.end]
        if len(span) != locus.length:
            raise HandoffError(
                f"locus {index} declares length {locus.length} but spans "
                f"{len(span)} nt over [{locus.start}, {locus.end}); the payload's coordinates "
                "and its sequence would disagree, which downstream reads as a real candidate"
            )
        n_soft_masked = sum(1 for ch in span if ch.islower())
        n_ambiguous = sum(1 for ch in span if ch not in _UNAMBIGUOUS)
        emitted_spans.append((locus.start, locus.end))
        for strand in call.strands:
            payloads.append(
                Handoff(
                    locus_index=index,
                    strand=strand,
                    low_order_confidence=call.low_order_confidence,
                    start=locus.start,
                    end=locus.end,
                    length=len(span),
                    rna=transcribe_to_rna(span, strand=strand),
                    n_soft_masked=n_soft_masked,
                    n_ambiguous=n_ambiguous,
                )
            )
    # Reached only once every emitted span has been through ``require_iupac``, so the in-span
    # count below is 0 — measured that way rather than written that way.
    return HandoffResult(
        payloads=tuple(payloads),
        n_outside_alphabet=count_outside_alphabet(text),
        n_outside_alphabet_in_spans=sum(
            count_outside_alphabet(text[start:end]) for start, end in _merge_spans(emitted_spans)
        ),
    )


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
    "HandoffResult",
    "count_outside_alphabet",
    "handoff_is_sequence_only",
    "handoff_loci",
    "require_iupac",
    "reverse_complement",
    "transcribe_to_rna",
]
