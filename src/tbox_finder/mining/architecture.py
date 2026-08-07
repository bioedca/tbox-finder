"""ADR-0006 **D3** — criterion (b), relaxed architecture, off the de-novo consensus.

Amendment **A4** (signed 2026-08-07) pins the *instrument*: (b)'s
``named_elements_present`` / ``ncca_bulge_detected`` inputs are localized from the
**A3 per-candidate ``mlocarna`` ``#=GC SS_cons`` comparative consensus** of the
candidate's own homolog set, together with the candidate's own primary sequence.
Never RF00230, never an Rfam/class-I template, never a ``cmalign`` bit-score, never
the Stage-1/Stage-2 model's own output.  A4 pins the instrument and its
anti-circularity constraints **only, not an algorithm** — the algorithm is this
module, and every number it could have baked in is instead a required keyword.

What D3 pins, verbatim
----------------------
::

    criterion_b(named_elements_present, ncca_bulge_detected,
                *, ultrashort_relax, class_ii_relax) -> bool
        # base: named_elements_present AND ncca_bulge_detected.
        # ultrashort_relax: drop the expected-helix-presence requirement.
        # class_ii_relax: NCCA bulge required only where detectable (confidence-relaxed).
        # never a function of a cmalign-vs-RF00230 bit-score.

:func:`criterion_b` is that predicate and nothing else, so the P6 suite that freezes
D3 tests the ADR's own signature.  Everything three-valued lives in
:func:`architecture_status`, exactly as :func:`~tbox_finder.mining.synteny.criterion_c`
and :func:`~tbox_finder.mining.synteny.synteny_status` are split at (c).

The two relaxations are **not** symmetric, and reading them as symmetric is vacuous
--------------------------------------------------------------------------------
``ultrashort_relax`` drops a requirement outright.  ``class_ii_relax`` does **not**:
D3's wording is *"the bulge remains the target **where detectable** rather than being
treated as absent"* — it relaxes the **detection confidence**, not the requirement.
That distinction is load-bearing.  Read as a plain requirement-drop, the two
relaxations fire **together** on every translational locus (D6 passes on
``regulatory_mode == "translational"``, and a translational locus *is* class II), so
``criterion_b`` would return ``True`` unconditionally on the whole class-II corpus —
(b) would never be ``False``, and by **D9 row 5** (``(c)✓ ∧ (a)✓ ∧ (b)✗``) **Tier-2N
would become unreachable**.  The flagship class would be defined out of existence by
its own predicate.

So the "where detectable" arm is resolved **upstream**, in the localizer, which
reports the bulge three-valued (:data:`BULGE_DETECTED` / :data:`BULGE_ABSENT` /
:data:`BULGE_UNDETECTABLE`):

* ``detected``      → the requirement is met;
* ``absent``        → the architecture *was* resolvable and the bulge is not there;
  this is a real failure and ``class_ii_relax`` does **not** excuse it;
* ``undetectable``  → nothing was resolvable to judge; only here does
  ``class_ii_relax`` drop the requirement.

:func:`architecture_status` additionally refuses to report ``passed`` when **both**
relaxations are active and **neither** element was actually observed — a pass with no
evidence under it.  That routes to ``unavailable`` ⇒ **spared** under ADR-0005 D14
(fail-closed) rather than to a fabricated ``passed``.

Every value the round must choose, and why none of them is defaulted
--------------------------------------------------------------------
D3 requires the predicate be *"frozen on the held-out canonical set and unit-tested"*,
and A4 ships D6's short-Stem-I threshold **keyword-required with no default** rather
than consuming D17's P6 freeze.  The same discipline covers every number here:
:func:`named_elements_present`'s ``min_named_helices``, :func:`ncca_bulge_status`'s
``ncca_pairing_nt`` and ``bulge_size_range``, and
:func:`~tbox_finder.mining.architecture_producer` 's ``stem_i_nt_threshold`` are all
required keywords.  ``architecture_producer freeze`` **measures** their behaviour on
the held-out canonical carve of the ADR-0004 split and writes the distribution; it
pins nothing (§10.3).  The relaxations only ever make (b) *more* likely to pass ⇒
more sparing ⇒ the fail-closed direction, so an imperfect value errs safe.

The NCCA motif is **derived, not transcribed**
----------------------------------------------
The one sequence fact the bulge detector needs is what the bulge must look like to
pair the tRNA acceptor end.  It is not copied from a paper and not hardcoded: it is
*derived* in :func:`acceptor_pairing_motif` from (i) the universal tRNA 3′ terminus
``N73-C74-C75-A76`` and (ii) antiparallel Watson–Crick complementarity.  Reading the
bulge 5′→3′ against the acceptor end 3′→5′ gives ``U·A76 G·C75 G·C74 N·N73`` — i.e.
``UGGN``.

The **cited** facts behind that derivation (CLAUDE.md §10.1; high-stakes — this
predicate decides which candidates are mined; four agreeing peer-reviewed sources,
accessed 2026-08-07):

* the antiterminator is *"two short helices with an intervening 7nt bulge"*, whose 3′
  end carries *"the highly conserved ACC residues"* — Gerdeman et al. 2003, *J Mol
  Biol* [PMID:12547201; DOI:10.1016/s0022-2836(02)01327-01];
* *"the tRNA acceptor end base pairs with the **first four nucleotides** in the seven
  nucleotide bulge of the highly conserved antiterminator element"* — Fauzi et al.
  2008, *BBA* [DOI:10.1016/j.bbagrm.2008.09.001];
* *"base pairing of the acceptor end of cognate tRNA with **four bases in a 7 nt
  bulge** of the antiterminator RNA"* — Fauzi et al. 2005, *NAR* [PMID:15879350;
  DOI:10.1093/nar/gki573];
* *"the non-aminoacylated tRNA acceptor end base pairing with the first four
  nucleotides at the 5′-end of a bulge in a highly conserved antiterminator
  element"* — Hines et al. 2010, *Biophys J* [DOI:10.1016/j.bpj.2009.12.2831].

⚠ Fauzi 2008 also reports that *in vitro* the pairing register is *"not constrained to
the first four bases of the bulge"*.  That is a selection result about what an
isolated bulge can accept, not the phylogenetic consensus — so :func:`ncca_bulge_status`
searches the **whole** bulge for the pairing motif rather than only its 5′ edge.  The
looser reading is also the sparing one.

**Confirmed in-repo before it was encoded**, on the project's own curated corpus
(``data/processed/master_clean_v0.parquet``, 23,208 records carrying a curated
antiterminator ``discriminator``): the discriminator is 4 nt in **all** of them and
matches the derived ``UGGN`` motif in **99.79 %**; it occurs as a literal substring of
the record's own element sequence in **99.99 %**.  On the curated antiterminator
structure the discriminator is the 5′ half of a 7-nt unpaired run flanked by two
helices, with ``ACC`` behind it — the architecture the citations describe.  ⚠ The
corpus is CM-derived (§5), so this is a *consistency* check on the element vocabulary,
**not** independent evidence; the production path never reads a CM.

Coordinates and alphabet are **not** trusted from TBDB
------------------------------------------------------
``master_clean_v0``'s element coordinates index a different frame than its own
``Sequence`` column (measured: the ``discrim_start``/``discrim_end`` span reproduces
the ``discriminator`` on 5 of 2,000 rows).  Every anchor here is therefore an **exact
subsequence match**, never a coordinate.

Pure stdlib: no I/O, no third-party imports.  PRD §13.2/§13.3(b); ADR-0006 D3/D6/D9/A3/A4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tbox_finder.labels import CLASS_ORDER
from tbox_finder.mining.spare_rule import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
)

ADR = "ADR-0006"
STEP = "P3-15'-d"

__all__ = [
    "ArchitectureError",
    "BULGE_ABSENT",
    "BULGE_DETECTED",
    "BULGE_STATES",
    "BULGE_UNDETECTABLE",
    "Bulge",
    "ConsensusStructure",
    "EXPECTED_HELIX_ELEMENTS",
    "Helix",
    "Localization",
    "MAX_NAMED_HELICES",
    "PAIR_OPEN_TO_CLOSE",
    "SS_CONS_TAG",
    "TRNA_ACCEPTOR_3PRIME",
    "WOBBLE_PAIRS",
    "acceptor_pairing_motif",
    "architecture_status",
    "criterion_b",
    "degapped_span",
    "find_bulges",
    "find_helices",
    "localize",
    "named_elements_status",
    "ncca_bulge_status",
    "pair_table",
    "parse_stockholm",
    "short_stem_i_or_class_ii",
]


class ArchitectureError(ValueError):
    """Raised on malformed consensus input (unbalanced SS_cons, ragged alignment)."""


# ═════════════════════════════════════════════════════════════════════════════
# The tRNA acceptor end, and the bulge motif derived from it
# ═════════════════════════════════════════════════════════════════════════════
#: The tRNA 3′ terminus the antiterminator bulge reads, 5′→3′: the variable
#: discriminator base ``N73`` followed by the universal ``C74-C75-A76`` (``CCA``).
#: ``N`` is the wildcard — it is a real position with a real pairing partner, it is
#: simply not constrained, and writing it out keeps the motif's *length* honest.
TRNA_ACCEPTOR_3PRIME = "NCCA"

#: The wildcard symbol, in both the acceptor string and the derived motif.
ANY_BASE = "N"

#: Watson–Crick complements, RNA alphabet.  ``T`` is folded to ``U`` on input rather
#: than given an entry here, so a DNA-alphabet sequence cannot quietly take a
#: different code path from an RNA one.
WATSON_CRICK: Mapping[str, str] = {"A": "U", "U": "A", "G": "C", "C": "G"}

#: G·U wobble, accepted **only** where a caller opts in.  A wobble at the acceptor
#: interface is a real pairing but a weaker claim, so it is never the default.
WOBBLE_PAIRS: frozenset[tuple[str, str]] = frozenset({("G", "U"), ("U", "G")})


def acceptor_pairing_motif(acceptor_3prime: str = TRNA_ACCEPTOR_3PRIME) -> str:
    """The bulge motif (5′→3′) that pairs ``acceptor_3prime`` (5′→3′) antiparallel.

    Derived, never transcribed: reverse the acceptor end and complement each base, so
    the default ``NCCA`` yields ``UGGN``.  Wildcards pass through as wildcards — a
    variable base in the acceptor end is a variable base in the motif, not a gap in
    it.

    >>> acceptor_pairing_motif()
    'UGGN'
    >>> acceptor_pairing_motif("CCA")
    'UGG'
    """
    text = str(acceptor_3prime).upper().replace("T", "U")
    if not text:
        raise ArchitectureError("acceptor_3prime is empty")
    unknown = sorted(set(text) - set(WATSON_CRICK) - {ANY_BASE})
    if unknown:
        raise ArchitectureError(f"acceptor_3prime carries non-RNA symbols {unknown}")
    return "".join(ANY_BASE if b == ANY_BASE else WATSON_CRICK[b] for b in reversed(text))


def _pairs_with(base: str, motif_base: str, *, allow_wobble: bool) -> bool:
    """Does ``base`` satisfy ``motif_base`` of an :func:`acceptor_pairing_motif` motif?"""
    if motif_base == ANY_BASE:
        return base in WATSON_CRICK
    if base == motif_base:
        return True
    return allow_wobble and (base, motif_base) in WOBBLE_PAIRS


# ═════════════════════════════════════════════════════════════════════════════
# Consensus secondary structure: parse, pair, and decompose
# ═════════════════════════════════════════════════════════════════════════════
#: Every bracket family a consensus line may use.  ``mlocarna --consensus-structure
#: alifold`` emits plain ``()``; the curated TBDB structures the freeze reads are WUSS
#: (``<>``); Rfam-style seeds add ``[]``/``{}``.  All are accepted so the *same*
#: parser serves the production consensus and the freeze corpus — one implementation,
#: so a bug cannot be fixed in one and shipped in the other.
PAIR_OPEN_TO_CLOSE: Mapping[str, str] = {"(": ")", "<": ">", "[": "]", "{": "}"}

#: Unpaired symbols.  WUSS spells single-strandedness five different ways
#: (``.,_-:~``) and treating any of them as "unknown" rather than "unpaired" would
#: silently shrink every bulge this module measures.
UNPAIRED_SYMBOLS: frozenset[str] = frozenset(".,_-:~")


#: The ``#=GC`` tag carrying the RNAalifold comparative consensus ``mlocarna
#: --consensus-structure alifold`` writes and R-scape scores (ADR-0006 A3).
SS_CONS_TAG = "SS_cons"


def parse_stockholm(source: str | Path) -> ConsensusStructure:
    """A Pfam/Stockholm alignment → its sequences and its ``#=GC SS_cons``.

    **Delegates** the parse to
    :func:`tbox_finder.mining.msa_shuffle.read_pfam_alignment`, which already handles
    the interleaved (wrapped) layout ``mlocarna`` actually writes and already returns
    the ``#=GC`` block separately.  Forking a second Stockholm reader would mean a
    wrap-handling bug could be fixed in one and shipped in the other — and this module
    and the covariation producer read the *same file*, so they must agree byte for
    byte about what is in it.

    What this adds is the three refusals (b) needs and a shuffle does not:

    * no ``#=GC SS_cons`` → there is no consensus to localize from;
    * no sequence rows → nothing to read the bulge out of;
    * a row whose width disagrees with the consensus → the alignment cannot be
      indexed column-wise, so any decomposition of it would be fiction.

    ``source`` may be a path or the alignment text itself (inherited from the
    delegate).
    """
    from tbox_finder.mining.msa_shuffle import read_pfam_alignment

    try:
        seqs, gc = read_pfam_alignment(source)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArchitectureError(f"cannot read Stockholm alignment: {exc}") from exc

    ss_cons = gc.get(SS_CONS_TAG)
    if not ss_cons:
        raise ArchitectureError(f"alignment carries no '#=GC {SS_CONS_TAG}' line")
    if not seqs:
        raise ArchitectureError("alignment carries no sequence rows")
    ragged = {name: len(row) for name, row in seqs if len(row) != len(ss_cons)}
    if ragged:
        raise ArchitectureError(
            f"alignment is ragged: {SS_CONS_TAG} is {len(ss_cons)} columns but "
            f"{sorted(ragged)[:3]} are {sorted(set(ragged.values()))[:3]}"
        )
    return ConsensusStructure(
        names=tuple(name for name, _ in seqs),
        sequences={name: row for name, row in seqs},
        ss_cons=ss_cons,
    )


@dataclass(frozen=True)
class ConsensusStructure:
    """One parsed alignment: row order, aligned rows, and the consensus structure."""

    names: tuple[str, ...]
    sequences: Mapping[str, str]
    ss_cons: str

    @property
    def n_sequences(self) -> int:
        return len(self.names)

    @property
    def width(self) -> int:
        return len(self.ss_cons)

    def row(self, name_or_index: str | int) -> str:
        """One aligned row, by name or by 0-based position in file order."""
        if isinstance(name_or_index, int):
            try:
                name = self.names[name_or_index]
            except IndexError as exc:
                raise ArchitectureError(
                    f"row {name_or_index} out of range ({self.n_sequences} rows)"
                ) from exc
        else:
            name = name_or_index
            if name not in self.sequences:
                raise ArchitectureError(f"no row named {name!r}")
        return self.sequences[name]


def pair_table(ss_cons: str) -> list[int]:
    """Consensus structure → ``pairs[i]`` = partner column of ``i``, or ``-1``.

    Every bracket family in :data:`PAIR_OPEN_TO_CLOSE` gets its own stack, so
    pseudoknotted notation (``<<..AA..>>..aa``) does not cross-pair with the primary
    family.  Symbols outside both tables (upper/lower-case pseudoknot letters, gap
    characters) are treated as unpaired rather than rejected: an unrecognised
    annotation must shrink what is claimed, never crash the producer mid-array.

    Raises :class:`ArchitectureError` only on an **unbalanced** bracket, which means
    the consensus itself is corrupt and any decomposition of it would be fiction.
    """
    text = str(ss_cons)
    pairs = [-1] * len(text)
    stacks: dict[str, list[int]] = {opener: [] for opener in PAIR_OPEN_TO_CLOSE}
    close_to_open = {close: opener for opener, close in PAIR_OPEN_TO_CLOSE.items()}
    for i, ch in enumerate(text):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in close_to_open:
            opener = close_to_open[ch]
            if not stacks[opener]:
                raise ArchitectureError(f"unbalanced {ch!r} at column {i} in SS_cons")
            j = stacks[opener].pop()
            pairs[i], pairs[j] = j, i
    unclosed = {opener: len(stack) for opener, stack in stacks.items() if stack}
    if unclosed:
        raise ArchitectureError(f"unbalanced SS_cons: unclosed {unclosed}")
    return pairs


@dataclass(frozen=True)
class Helix:
    """A maximal stack of consecutive base pairs.

    ``(left_start..left_end)`` paired with ``(right_start..right_end)``.
    """

    left_start: int
    left_end: int
    right_start: int
    right_end: int

    @property
    def n_pairs(self) -> int:
        return self.left_end - self.left_start + 1


def find_helices(pairs: Sequence[int], *, min_pairs: int) -> list[Helix]:
    """Decompose a pair table into maximal helices of at least ``min_pairs`` pairs.

    A helix is a run of pairs ``(i, j), (i+1, j-1), …`` with no interruption.  A
    single isolated pair is not a helix in any useful sense, so ``min_pairs`` has
    **no default**: how much stacking counts as "an expected helix is present" is a
    judgement the round makes and records, not one this module makes silently.
    """
    if int(min_pairs) < 1:
        raise ArchitectureError(f"min_pairs must be >= 1, got {min_pairs}")
    helices: list[Helix] = []
    i = 0
    n = len(pairs)
    while i < n:
        j = pairs[i]
        if j <= i:
            i += 1
            continue
        left_start, right_end = i, j
        # Extend while the next column pairs with the previous partner's neighbour.
        while i + 1 < n and j - 1 > i + 1 and pairs[i + 1] == j - 1:
            i, j = i + 1, j - 1
        helix = Helix(left_start=left_start, left_end=i, right_start=j, right_end=right_end)
        if helix.n_pairs >= int(min_pairs):
            helices.append(helix)
        i += 1
    return helices


@dataclass(frozen=True)
class Bulge:
    """An unpaired run flanked by helix columns on **both** sides.

    ``start``/``end`` are inclusive alignment-column bounds.  ``flank_left`` and
    ``flank_right`` are the paired columns immediately outside it — their presence is
    what makes this a bulge rather than a terminal tail or a hairpin loop.
    """

    start: int
    end: int
    flank_left: int
    flank_right: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def find_bulges(pairs: Sequence[int]) -> list[Bulge]:
    """Every unpaired run flanked by a pair on **both** sides, in column order.

    ⚠ **No size filter here, and that is the point.** ``start``/``end`` are alignment
    *columns*, and a bulge's column width is not its length: on the certified
    24-sequence ``mlocarna`` consensus a 16-column run is **8 residues** in the
    candidate's own row and a 37-column run is 11.  Filtering on column width would
    have rejected both — measured, not hypothesised — so the size test belongs where
    the sequence is known, in :func:`ncca_bulge_status`, and is applied to the
    **degapped residues of the candidate's own row**.

    A **hairpin loop** is excluded: its two flanks are partners of each other, so it
    is enclosed by a single helix rather than sitting between two.  Admitting it
    would let every stem-loop in the consensus offer its loop as a candidate bulge.
    """
    bulges: list[Bulge] = []
    n = len(pairs)
    i = 0
    while i < n:
        if pairs[i] != -1:
            i += 1
            continue
        start = i
        while i < n and pairs[i] == -1:
            i += 1
        end = i - 1
        left, right = start - 1, end + 1
        if left < 0 or right >= n:
            continue
        if pairs[left] == -1 or pairs[right] == -1:
            continue
        if pairs[left] == right:
            # Hairpin loop: the flanks close each other.
            continue
        bulges.append(Bulge(start=start, end=end, flank_left=left, flank_right=right))
    return bulges


# ═════════════════════════════════════════════════════════════════════════════
# The two D3 inputs, localized
# ═════════════════════════════════════════════════════════════════════════════
#: The bulge was found and pairs the acceptor end.
BULGE_DETECTED = "detected"
#: The architecture was resolvable and no acceptor-pairing bulge is in it.  A real
#: negative — ``class_ii_relax`` does **not** excuse this one.
BULGE_ABSENT = "absent"
#: Nothing was resolvable to judge (no flanked bulge of admissible size anywhere).
#: The **only** state ``class_ii_relax`` converts into "requirement dropped".
BULGE_UNDETECTABLE = "undetectable"

BULGE_STATES: tuple[str, ...] = (BULGE_DETECTED, BULGE_ABSENT, BULGE_UNDETECTABLE)


def degapped_span(row: str, start: int, end: int) -> str:
    """Residues of ``row`` in inclusive columns ``[start, end]``, gaps removed, RNA alphabet.

    Public because the freeze reads spans the same way the producer does; two copies of
    this three-line rule would be two chances to disagree about what a gap is.
    """
    chunk = row[start : end + 1].upper().replace("T", "U")
    return "".join(c for c in chunk if c in WATSON_CRICK)


def ncca_bulge_status(
    row: str,
    pairs: Sequence[int],
    *,
    bulge_size_range: tuple[int, int],
    ncca_pairing_nt: int,
    allow_wobble: bool = False,
    acceptor_3prime: str = TRNA_ACCEPTOR_3PRIME,
) -> tuple[str, dict[str, Any]]:
    """Is an NCCA-pairing bulge present in this row's consensus architecture?

    Returns ``(state, detail)`` where ``state`` is one of :data:`BULGE_STATES`.

    The test is the derived one (:func:`acceptor_pairing_motif`): somewhere inside a
    flanked, admissibly-sized unpaired run, ``ncca_pairing_nt`` consecutive residues
    of **this candidate's own sequence** must satisfy the motif that pairs the tRNA 3′
    end antiparallel.  The whole run is searched, not only its 5′ edge — the register
    is not phylogenetically fixed (Fauzi 2008), and the looser reading is also the
    sparing one.

    ``ncca_pairing_nt`` has **no default**: it is how many of the four acceptor
    contacts must be evidenced, and that decides which candidates are mined.

    ``undetectable`` (no flanked bulge of admissible size at all) is kept distinct
    from ``absent`` (there were bulges; none pairs the acceptor end).  Collapsing the
    two is precisely what would make ``class_ii_relax`` vacuous — see the module
    docstring.
    """
    want = int(ncca_pairing_nt)
    motif = acceptor_pairing_motif(acceptor_3prime)
    if want < 1 or want > len(motif):
        raise ArchitectureError(
            f"ncca_pairing_nt must be in [1, {len(motif)}] for acceptor {acceptor_3prime!r}, "
            f"got {ncca_pairing_nt}"
        )
    low, high = int(bulge_size_range[0]), int(bulge_size_range[1])
    if low < 1 or high < low:
        raise ArchitectureError(
            f"bulge_size_range must be 1 <= min <= max, got {bulge_size_range!r}"
        )
    if high < want:
        raise ArchitectureError(
            f"bulge_size_range max {high} is shorter than ncca_pairing_nt {want}: no bulge "
            "could ever hold the motif, so every candidate would read 'undetectable'"
        )
    # Size in the candidate's OWN residues, not alignment columns — see `find_bulges`.
    sized = [
        (b, res)
        for b, res in ((b, degapped_span(row, b.start, b.end)) for b in find_bulges(pairs))
        if low <= len(res) <= high
    ]
    detail: dict[str, Any] = {
        "motif": motif,
        "ncca_pairing_nt": want,
        "n_bulges_considered": len(sized),
        "bulge_size_range": [low, high],
        "allow_wobble": bool(allow_wobble),
    }
    if not sized:
        return BULGE_UNDETECTABLE, detail

    # Any contiguous `want`-mer of the motif is an acceptable register: requiring the
    # motif's own 5' prefix would re-impose the register Fauzi 2008 measured as free.
    registers = [motif[s : s + want] for s in range(len(motif) - want + 1)]
    for bulge, residues in sized:
        for offset in range(0, max(0, len(residues) - want + 1)):
            window = residues[offset : offset + want]
            for register in registers:
                if all(
                    _pairs_with(b, m, allow_wobble=allow_wobble)
                    for b, m in zip(window, register, strict=True)
                ):
                    detail.update(
                        {
                            "bulge_start": bulge.start,
                            "bulge_end": bulge.end,
                            "bulge_size": bulge.size,
                            "bulge_residues": residues,
                            "matched": window,
                            "matched_register": register,
                            "matched_offset": offset,
                        }
                    )
                    return BULGE_DETECTED, detail
    detail["bulge_residues_seen"] = [res for _, res in sized[:8]]
    return BULGE_ABSENT, detail


#: The **helix-shaped** members of the ADR-0004 D1 element vocabulary — derived from
#: :data:`tbox_finder.labels.CLASS_ORDER` rather than re-typed, so adding or renaming a
#: D1 element moves this set with it instead of leaving a stale copy behind.  D1's
#: other members are not helices: ``background`` is the absence of an element, and
#: ``Specifier`` / ``Discriminator`` are single-stranded motifs *inside* elements
#: (the specifier loop; the antiterminator bulge this module localizes).
#: ``Terminator`` is excluded deliberately — it is a helix, but ``labels`` never
#: paints it on class-II records, and D3's expected-helix requirement must mean the
#: same thing on both classes or the ``ultrashort_relax`` carve-out would be doing
#: two jobs at once.
EXPECTED_HELIX_ELEMENTS: tuple[str, ...] = tuple(
    name
    for name in CLASS_ORDER
    if name not in ("background", "Specifier", "Discriminator", "Terminator")
)

#: The most helices a consensus can be asked to evidence before the requirement stops
#: being a statement about T-box architecture and becomes an arbitrary count.
MAX_NAMED_HELICES = len(EXPECTED_HELIX_ELEMENTS)


def named_elements_status(
    pairs: Sequence[int],
    *,
    min_named_helices: int,
    min_helix_pairs: int,
) -> tuple[bool, dict[str, Any]]:
    """Are the expected helices present in this consensus?

    D3 reuses the **ADR-0004 D1 element vocabulary** for what is expected
    (:data:`EXPECTED_HELIX_ELEMENTS`, derived from
    :data:`~tbox_finder.labels.CLASS_ORDER`); what a de-novo comparative consensus can
    actually evidence is *how many distinct helices of real stacking depth* it
    resolves.  Both counts are required keywords — the canonical class-I core is
    Stem I + Stem III + antiterminator (with Stem II in many lineages), but writing a
    number here would freeze at P3 a value D3 assigns to the held-out canonical set.
    ``architecture_producer freeze`` measures the distribution; the round supplies the
    value.

    Asking for more helices than D1 names is refused rather than silently made
    unsatisfiable: it would make (b) ``False`` everywhere, which by **D9 row 5** would
    route the *entire* corpus to Tier-2N.
    """
    if int(min_named_helices) < 1:
        raise ArchitectureError(f"min_named_helices must be >= 1, got {min_named_helices}")
    if int(min_named_helices) > MAX_NAMED_HELICES:
        raise ArchitectureError(
            f"min_named_helices={min_named_helices} exceeds the {MAX_NAMED_HELICES} "
            f"helix-shaped ADR-0004 D1 elements {EXPECTED_HELIX_ELEMENTS}; (b) would be "
            "False on every candidate and D9 row 5 would route the whole corpus to Tier-2N"
        )
    helices = find_helices(pairs, min_pairs=min_helix_pairs)
    detail = {
        "n_helices": len(helices),
        "min_named_helices": int(min_named_helices),
        "min_helix_pairs": int(min_helix_pairs),
        "helix_pair_counts": [h.n_pairs for h in helices],
        "expected_helix_elements": list(EXPECTED_HELIX_ELEMENTS),
    }
    return len(helices) >= int(min_named_helices), detail


# ═════════════════════════════════════════════════════════════════════════════
# D6 — the model-independent short-Stem-I / class-II predicate
# ═════════════════════════════════════════════════════════════════════════════
#: D6's translational arm.  ``regulatory_mode`` is read from the §6 resolver / the
#: corpus, never from the Stage-2 model.
TRANSLATIONAL_MODE = "translational"


def short_stem_i_or_class_ii(
    stem_i_extent_nt: int | None,
    regulatory_mode: str | None,
    *,
    stem_i_nt_threshold: int,
) -> bool:
    """ADR-0006 **D6**, exactly as pinned — gates *both* the (a) and (b) carve-outs.

    ``True`` iff ``stem_i_extent_nt < stem_i_nt_threshold`` **or**
    ``regulatory_mode == "translational"``.

    ``stem_i_nt_threshold`` is **keyword-required with no default**, per ADR-0006 A4:
    the round supplies it and it is recorded in provenance.  This does **not** consume
    D17's P6 delegation — the P6 freeze on the held-out canonical Stem-I extent
    distribution stands unchanged.

    An unknown extent (``None``) does not satisfy the threshold arm: absence of a
    measurement is not evidence of shortness.  The mode arm can still fire.
    """
    if regulatory_mode is not None and str(regulatory_mode).strip().lower() == TRANSLATIONAL_MODE:
        return True
    if stem_i_extent_nt is None:
        return False
    return int(stem_i_extent_nt) < int(stem_i_nt_threshold)


# ═════════════════════════════════════════════════════════════════════════════
# D3 — the frozen predicate, and its three-valued wrapper
# ═════════════════════════════════════════════════════════════════════════════
def criterion_b(
    named_elements_present: bool,
    ncca_bulge_detected: bool,
    *,
    ultrashort_relax: bool,
    class_ii_relax: bool,
) -> bool:
    """ADR-0006 D3's frozen predicate — signature and semantics exactly as pinned.

    Base: ``named_elements_present AND ncca_bulge_detected``.  ``ultrashort_relax``
    drops the expected-helix-presence requirement; ``class_ii_relax`` drops the NCCA
    requirement (it is reached only where the bulge is *undetectable* — see
    :func:`architecture_status`, which is what decides that).

    **Never** a function of a ``cmalign``-vs-RF00230 bit-score.  A4 strengthens this:
    no ``cmalign`` is on (b)'s path at all, so there is no bit-score to key it to.

    ⚠ With both relaxations true this is unconditionally ``True``.  That is the
    ADR's literal text and it is preserved here so the P6 freeze tests D3 itself;
    the guard against a *vacuous* pass lives one layer up, in
    :func:`architecture_status`, which refuses to publish ``passed`` on no evidence.
    """
    # The base rule is a conjunction of two requirements; each relaxation waives exactly
    # one of them. Written as `observed OR waived` per conjunct so the correspondence to
    # D3's two sentences is one-to-one and a future edit cannot silently move a
    # relaxation onto the wrong conjunct.
    named_ok = bool(named_elements_present) or bool(ultrashort_relax)
    bulge_ok = bool(ncca_bulge_detected) or bool(class_ii_relax)
    return named_ok and bulge_ok


@dataclass(frozen=True)
class Localization:
    """Everything the consensus said about one candidate, before the verdict."""

    candidate_id: str
    named_elements_present: bool
    bulge_state: str
    ultrashort_relax: bool
    class_ii_relax: bool
    n_sequences: int
    detail: dict[str, Any] = field(default_factory=dict)


def localize(
    candidate_id: str,
    consensus: ConsensusStructure,
    *,
    row: str | int = 0,
    stem_i_extent_nt: int | None,
    regulatory_mode: str | None,
    stem_i_nt_threshold: int,
    class_ii: bool,
    min_named_helices: int,
    min_helix_pairs: int,
    bulge_size_range: tuple[int, int],
    ncca_pairing_nt: int,
    allow_wobble: bool = False,
) -> Localization:
    """Localize D3's two inputs from the A3/A4 de-novo comparative consensus.

    ``row`` selects the candidate's own row of the alignment (its name, or its 0-based
    position — ``covariation_producer`` writes the candidate first).  The consensus
    structure is the alignment's; the *sequence* read through it is the candidate's
    own, which is what makes the bulge test a statement about this locus.

    No parameter that decides an outcome carries a default.
    """
    pairs = pair_table(consensus.ss_cons)
    named, named_detail = named_elements_status(
        pairs, min_named_helices=min_named_helices, min_helix_pairs=min_helix_pairs
    )
    state, bulge_detail = ncca_bulge_status(
        consensus.row(row),
        pairs,
        bulge_size_range=bulge_size_range,
        ncca_pairing_nt=ncca_pairing_nt,
        allow_wobble=allow_wobble,
    )
    ultrashort = short_stem_i_or_class_ii(
        stem_i_extent_nt, regulatory_mode, stem_i_nt_threshold=stem_i_nt_threshold
    )
    return Localization(
        candidate_id=str(candidate_id),
        named_elements_present=named,
        bulge_state=state,
        ultrashort_relax=ultrashort,
        class_ii_relax=bool(class_ii),
        n_sequences=consensus.n_sequences,
        detail={"named_elements": named_detail, "ncca_bulge": bulge_detail},
    )


def architecture_status(
    localization: Localization | None,
    *,
    min_sequences: int,
) -> tuple[str, dict[str, Any]]:
    """``passed`` / ``failed`` / ``unavailable`` for the ``relaxed_architecture`` disjunct.

    ``unavailable`` **spares** the candidate under ADR-0005 D14, and three things
    produce it — each one a genuine "could not be evaluated", never a disguised
    failure:

    1. ``localization is None`` — no per-candidate consensus was produced at all (no
       homolog set, no ``SS_cons``, a failed align).  A dropped shard cannot fail open.
    2. the alignment is **below the ADR-0006 A2 Pin 2 depth floor** (``min_sequences``,
       ``MIN_REAL_HOMOLOG_N``).  A4 disclosed this cost explicitly: (b) now reads the
       same MSA (a) does, so it inherits (a)'s power floor and the two disjuncts are
       no longer independent.
    3. the **vacuous-pass guard** — both relaxations active *and* neither element
       observed.  D3's literal text returns ``True`` there (see :func:`criterion_b`);
       publishing that as ``passed`` would be a pass with nothing under it, and by D9
       row 5 a (b) that is never ``False`` makes **Tier-2N unreachable**.  Routed to
       ``unavailable`` ⇒ spared: the fail-closed direction, and the routing survives.

    ``class_ii_relax`` reaches :func:`criterion_b` **only** when the bulge is
    :data:`BULGE_UNDETECTABLE`.  A bulge that is :data:`BULGE_ABSENT` was detectable
    and is not there — a real failure D3's confidence-relaxation does not excuse.
    """
    if localization is None:
        return STATUS_UNAVAILABLE, {"reason": "no per-candidate consensus was produced"}
    if int(localization.n_sequences) < int(min_sequences):
        return STATUS_UNAVAILABLE, {
            "reason": (
                f"alignment depth {localization.n_sequences} is below the ADR-0006 A2 Pin 2 "
                f"floor of {int(min_sequences)}"
            ),
            "n_sequences": localization.n_sequences,
        }

    detected = localization.bulge_state == BULGE_DETECTED
    undetectable = localization.bulge_state == BULGE_UNDETECTABLE
    # D3's "required only where detectable": the relaxation is reachable only on the
    # undetectable arm. An `absent` bulge is a decided negative on any class.
    effective_class_ii_relax = bool(localization.class_ii_relax) and undetectable

    if (
        localization.ultrashort_relax
        and effective_class_ii_relax
        and not localization.named_elements_present
        and not detected
    ):
        return STATUS_UNAVAILABLE, {
            "reason": (
                "both D3 relaxations are active and neither element was observed — a pass "
                "here would rest on no evidence and would make Tier-2N (D9 row 5) "
                "unreachable; spared instead"
            ),
            "bulge_state": localization.bulge_state,
        }

    passed = criterion_b(
        localization.named_elements_present,
        detected,
        ultrashort_relax=localization.ultrashort_relax,
        class_ii_relax=effective_class_ii_relax,
    )
    detail = {
        "named_elements_present": localization.named_elements_present,
        "bulge_state": localization.bulge_state,
        "ultrashort_relax": localization.ultrashort_relax,
        "class_ii_relax_declared": localization.class_ii_relax,
        "class_ii_relax_effective": effective_class_ii_relax,
        "n_sequences": localization.n_sequences,
    }
    return (STATUS_PASSED if passed else STATUS_FAILED), detail
