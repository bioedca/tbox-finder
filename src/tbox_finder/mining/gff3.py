"""P3-15′-c-i — a GFF3 reader, and the coordinate conventions ADR-0006 D4 is evaluated in.

`imp.md` P3-15′-c records *"zero coordinate-reading code"* as one of the three gaps blocking
the (c) synteny backend. This module closes that one. It is deliberately **only** a reader:
it turns GFF3 bytes into CDS features with faithful coordinates, a strand, and the attribute
dictionary a gene-identity test can be built on. It decides **nothing** about T-boxes.

**Why this is not a thin wrapper over a library.** The whole of `envs/*.yml` that ships a GFF
parser also ships biopython — and CI installs no biopython (`.github/workflows/ci.yml` pins
pytest + pyarrow/numpy/pandas/scikit-learn/hydra and nothing else). A parser behind an
``importorskip`` is a parser CI does not run, and this one is load-bearing for D4. Pure stdlib
keeps the whole thing in the bare tier, where the sabotage campaign can actually reach it.

Coordinate conventions, stated once so no downstream step has to guess
----------------------------------------------------------------------
GFF3 columns 4/5 are **1-based and inclusive on both ends** (the Sequence Ontology GFF3 spec,
https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md, accessed
2026-08-06). :class:`CdsFeature` carries them through **unchanged** — no half-open rewrite, no
0-based rewrite — because a silent convention change is the one parser bug that produces
plausible coordinates ([[int-truncation-changes-a-pinned-value]] in kind: the value survives
the type system and means something else). A consumer that wants Python slice semantics
converts at its own boundary and says so.

ADR-0006 D4 measures to *"the first downstream same-strand CDS **start**"*. On a ``-`` strand
CDS that is the feature's **high** coordinate, not its low one — the single most inviting
off-by-a-whole-gene error in this file. :func:`cds_start_position` is the only sanctioned way
to ask for it, and it refuses a feature whose strand is unresolved rather than defaulting to
``+`` (an unresolved strand cannot answer a same-strand question at all).

What the real data actually contains (measured on the fetched corpus, not assumed)
---------------------------------------------------------------------------------
* **Attribute values are URL-escaped.** ``country=USA: Crystal Geyser near Green River%2C Utah``
  is one real value from ``GCA_002790315.1``. Multi-valued attributes are comma-separated, so
  the split on ``,`` must happen **before** unescaping — a product name containing a literal
  comma arrives as ``%2C`` precisely so that ordering is well defined, and unescaping first
  would shatter it into two bogus values.
* **A CDS can span several lines.** ``GCF_002895085.1`` carries ``cds-WP_011096872.1`` on two
  rows (a programmed ribosomal frameshift). Grouping by ``ID`` is therefore not an
  optimisation — a per-row reading reports two genes where there is one, and puts a spurious
  "CDS start" in the middle of it.
* **Pseudogenes are already here.** 38 of 455 CDS rows in ``GCA_002790315.1`` carry
  ``pseudo=true``. D4 routes exactly those to its Pfam/KO fallback, so the flag is surfaced
  (:func:`is_pseudo`) rather than left for a caller to re-derive from a raw attribute.
* **``##FASTA`` must terminate the feature section.** NCBI does not emit one, but the
  annotator arm commissioned at P3-15′-c (bakta/prokka) does, and its FASTA payload contains
  lines that a 9-column split would happily mangle into features.

Failures are refusals, never degraded rows: a 7-column line, a non-integer coordinate, a
``start > end``, or one CDS ``ID`` appearing on two different contigs raises :class:`Gff3Error`.
A parser that silently drops a malformed line reports a smaller, cleaner annotation than the
file contains, and (c) then reads ``unavailable``/``failed`` for reasons no artifact records.
"""

from __future__ import annotations

import gzip
import re
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CDS_TYPE",
    "FASTA_DIRECTIVE",
    "GFF_VERSION_DIRECTIVE",
    "STRAND_MINUS",
    "STRAND_PLUS",
    "STRANDS_RESOLVED",
    "CdsFeature",
    "Gff3Error",
    "GffFeature",
    "attribute_first",
    "cds_start_position",
    "gene_identity_text",
    "group_cds",
    "is_pseudo",
    "iter_gff3_features",
    "parse_attributes",
    "parse_gff3_cds",
    "parse_gff3_line",
    "read_gff3_text",
    "unescape",
]


class Gff3Error(ValueError):
    """A refusal: this input is not GFF3 we are willing to read."""


#: the feature type D4's rule is about. Everything else (``gene``, ``exon``, ``tRNA``,
#: ``pseudogene``, ``region``) is parsed but not grouped.
CDS_TYPE = "CDS"

#: the directive that ends the feature section. Everything after it is sequence.
FASTA_DIRECTIVE = "##FASTA"

#: the mandatory first line of a GFF3 file.
GFF_VERSION_DIRECTIVE = "##gff-version"

STRAND_PLUS = "+"
STRAND_MINUS = "-"

#: the two strands a same-strand question can be answered on. ``.`` (not stranded) and ``?``
#: (unknown) are parsed faithfully and then refused by :func:`cds_start_position`.
STRANDS_RESOLVED: frozenset[str] = frozenset({STRAND_PLUS, STRAND_MINUS})

_VALID_STRANDS: frozenset[str] = frozenset({STRAND_PLUS, STRAND_MINUS, ".", "?"})

_N_COLUMNS = 9

#: attribute keys whose values name what a gene *does*, in the order D4's "product name"
#: route should consult them. ``product`` is NCBI's primary product name; ``gene`` and
#: ``Name`` carry the gene symbol; ``Note`` and the GO term names carry the free text a
#: hypothetical-but-characterised ORF is described in. This module only *collects* them —
#: deciding whether the text means "aaRS" is D4's job, at P3-15′-c-ii.
GENE_IDENTITY_KEYS: tuple[str, ...] = (
    "product",
    "gene",
    "Name",
    "gene_synonym",
    "Note",
    "go_function",
    "go_process",
    "Ontology_term",
    "Dbxref",
)

_TRUEISH: frozenset[str] = frozenset({"true", "yes", "1"})


@dataclass(frozen=True)
class GffFeature:
    """One 9-column GFF3 feature line, unescaped, with 1-based inclusive coordinates."""

    seqid: str
    source: str
    type: str
    start: int
    end: int
    score: float | None
    strand: str
    phase: int | None
    attributes: Mapping[str, tuple[str, ...]]

    @property
    def feature_id(self) -> str:
        """The ``ID`` attribute, or a positional synthetic key when the line carries none.

        NCBI always writes ``ID``; a minimal third-party GFF3 need not. The synthetic key is
        unique per line, so an ID-less file degrades to one-feature-per-line rather than
        collapsing every unnamed CDS into a single 5-Mb pseudo-gene
        ([[duplicate-key-merges-instead-of-colliding]] — the merge is the silent failure).
        """
        got = attribute_first(self.attributes, "ID")
        if got:
            return got
        return f"{self.seqid}:{self.start}-{self.end}:{self.strand}:{self.type}"


@dataclass(frozen=True)
class CdsFeature:
    """One coding sequence — all rows sharing a CDS ``ID``, merged.

    ``start``/``end`` are the **span** (min/max over segments), 1-based inclusive, as they
    appear in the file. ``segments`` keeps the individual rows in file order so a caller can
    tell a two-segment frameshifted CDS from a one-segment one; ``len(segments) > 1`` is a
    real, measured shape in this corpus, not a theoretical one.
    """

    seqid: str
    feature_id: str
    start: int
    end: int
    strand: str
    segments: tuple[tuple[int, int], ...]
    attributes: Mapping[str, tuple[str, ...]]

    @property
    def length_bp(self) -> int:
        """Span length in bp, inclusive of both endpoints (so a 1-bp feature is 1, not 0)."""
        return self.end - self.start + 1


def unescape(value: str) -> str:
    """Decode GFF3 column-9 ``%XX`` escaping.

    ``urllib.parse.unquote``, **not** ``unquote_plus``: GFF3 does not use ``+`` for space, and
    a product name like ``NAD(P)+ transhydrogenase`` must keep its plus sign.
    """
    return urllib.parse.unquote(str(value))


def parse_attributes(column9: str) -> dict[str, tuple[str, ...]]:
    """GFF3 column 9 → ``{key: (value, ...)}``, unescaped.

    Split order is load-bearing: ``;`` then ``=`` then ``,`` then unescape. A value carrying a
    literal comma arrives escaped as ``%2C`` exactly so that ``,`` can be the value separator;
    unescaping earlier would turn one product name into two.

    A repeated key merges its values in file order rather than overwriting — the overwrite is
    the shape that silently loses evidence ([[duplicate-key-merges-instead-of-colliding]]).
    An empty column (``.`` or ``""``) is an attribute-less feature, not an error; a fragment
    with no ``=`` is malformed and refused.
    """
    text = str(column9).strip()
    out: dict[str, tuple[str, ...]] = {}
    if text in ("", "."):
        return out
    for chunk in text.split(";"):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise Gff3Error(f"attribute fragment has no '=': {chunk[:120]!r}")
        raw_key, raw_val = chunk.split("=", 1)
        key = unescape(raw_key.strip())
        if not key:
            raise Gff3Error(f"attribute fragment has an empty key: {chunk[:120]!r}")
        values = tuple(unescape(v) for v in raw_val.split(","))
        out[key] = out.get(key, ()) + values
    return out


def attribute_first(attributes: Mapping[str, Sequence[str]], key: str) -> str | None:
    """The first value of ``key``, or ``None`` when absent or present-but-empty.

    ``None`` and ``""`` are collapsed on purpose: a caller asking for a product name wants
    "there isn't one" in a single check, and an empty ``product=`` is exactly as uninformative
    as an absent one.
    """
    values = attributes.get(key)
    if not values:
        return None
    first = str(values[0]).strip()
    return first or None


def _parse_coordinate(raw: str, *, field: str, line_no: int) -> int:
    text = str(raw).strip()
    if not re.fullmatch(r"\d+", text):
        raise Gff3Error(f"line {line_no}: {field} is not a positive integer: {raw!r}")
    value = int(text)
    if value < 1:
        raise Gff3Error(f"line {line_no}: {field} is {value}; GFF3 coordinates are 1-based")
    return value


def parse_gff3_line(line: str, *, line_no: int = 0) -> GffFeature:
    """One tab-separated GFF3 feature line → :class:`GffFeature`.

    Refuses (never degrades) on: a column count other than 9, a non-integer or sub-1
    coordinate, ``start > end``, an unknown strand token, or a non-``.``/non-integer phase.
    """
    parts = str(line).rstrip("\r\n").split("\t")
    if len(parts) != _N_COLUMNS:
        raise Gff3Error(
            f"line {line_no}: expected {_N_COLUMNS} tab-separated columns, got {len(parts)}"
        )
    seqid, source, ftype, raw_start, raw_end, raw_score, strand, raw_phase, col9 = parts

    start = _parse_coordinate(raw_start, field="start", line_no=line_no)
    end = _parse_coordinate(raw_end, field="end", line_no=line_no)
    if start > end:
        raise Gff3Error(f"line {line_no}: start {start} > end {end}")

    if strand not in _VALID_STRANDS:
        raise Gff3Error(f"line {line_no}: unknown strand {strand!r}")

    score: float | None = None
    if raw_score.strip() != ".":
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise Gff3Error(
                f"line {line_no}: score is neither '.' nor a float: {raw_score!r}"
            ) from exc

    phase: int | None = None
    if raw_phase.strip() != ".":
        if raw_phase.strip() not in ("0", "1", "2"):
            raise Gff3Error(f"line {line_no}: phase is neither '.' nor 0/1/2: {raw_phase!r}")
        phase = int(raw_phase)

    return GffFeature(
        seqid=unescape(seqid),
        source=unescape(source),
        type=ftype,
        start=start,
        end=end,
        score=score,
        strand=strand,
        phase=phase,
        attributes=parse_attributes(col9),
    )


def iter_gff3_features(
    lines: Iterable[str], *, types: Iterable[str] | None = None
) -> Iterator[GffFeature]:
    """Yield features from GFF3 lines, stopping dead at ``##FASTA``.

    ``types`` filters by column 3 (``{"CDS"}`` for D4); ``None`` yields every feature. Comment
    and directive lines are skipped, blank lines ignored. Everything at or after ``##FASTA``
    is sequence and is **not** parsed — an annotator that appends its contigs (prokka does)
    would otherwise contribute nucleotide lines to the feature census.
    """
    wanted = frozenset(types) if types is not None else None
    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip("\r\n")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper() == FASTA_DIRECTIVE:
            return
        if line.startswith("#"):
            continue
        feature = parse_gff3_line(line, line_no=line_no)
        if wanted is None or feature.type in wanted:
            yield feature


def group_cds(features: Iterable[GffFeature]) -> list[CdsFeature]:
    """Merge CDS rows sharing an ``ID`` into one :class:`CdsFeature`, in first-seen order.

    Segments of one CDS must agree on contig and strand; a disagreement is a refusal, because
    the alternative — picking one — silently invents a gene at coordinates that exist on
    neither contig. Order is first-appearance, so the output tracks the file rather than a
    dict's iteration order.
    """
    order: list[str] = []
    rows: dict[str, list[GffFeature]] = {}
    for index, feature in enumerate(features):
        # Group by a **declared** ``ID`` only. The coordinate-derived fallback is a display
        # name, not an identity: two ID-less CDS rows sharing contig, coordinates, strand and
        # type collapse into one, and the later row's attributes are lost — a merge, which is
        # the silent shape ([[duplicate-key-merges-instead-of-colliding]]), not a crash. The
        # row index makes an undeclared key unique without changing anything for declared ones.
        declared = attribute_first(feature.attributes, "ID")
        key = declared if declared else f"{feature.feature_id}#{index}"
        if key not in rows:
            rows[key] = []
            order.append(key)
        rows[key].append(feature)

    out: list[CdsFeature] = []
    for key in order:
        group = rows[key]
        head = group[0]
        for other in group[1:]:
            if other.seqid != head.seqid:
                raise Gff3Error(
                    f"CDS {key!r} spans two contigs: {head.seqid!r} and {other.seqid!r}"
                )
            if other.strand != head.strand:
                raise Gff3Error(
                    f"CDS {key!r} carries two strands: {head.strand!r} and {other.strand!r}"
                )
        segments = tuple((f.start, f.end) for f in group)
        merged: dict[str, tuple[str, ...]] = {}
        for feature in group:
            for attr_key, values in feature.attributes.items():
                if attr_key not in merged:
                    merged[attr_key] = values
        out.append(
            CdsFeature(
                seqid=head.seqid,
                feature_id=key,
                start=min(s for s, _ in segments),
                end=max(e for _, e in segments),
                strand=head.strand,
                segments=segments,
                attributes=merged,
            )
        )
    return out


def parse_gff3_cds(lines: Iterable[str]) -> list[CdsFeature]:
    """GFF3 lines → the file's CDS features, multi-row IDs merged. The module's entry point."""
    return group_cds(iter_gff3_features(lines, types={CDS_TYPE}))


def read_gff3_text(path: str | Path) -> str:
    """Read a ``.gff`` or ``.gff.gz`` as text.

    gzip is detected by the two magic bytes, not by the file name, so a ``.gz`` suffix on
    plain text (or the reverse) reads correctly instead of raising something that looks like a
    corrupt download. Decoded as UTF-8: NCBI product names carry non-ASCII (Greek letters in
    enzyme names), and ``latin-1`` would mangle them into plausible mojibake rather than fail.
    """
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def cds_start_position(cds: CdsFeature) -> int:
    """The 1-based coordinate of the CDS **start codon** — D4's "downstream CDS start".

    ``+`` strand → the low coordinate; ``-`` strand → the **high** one. Refuses ``.``/``?``:
    a feature with no resolved strand cannot answer a same-strand question, and returning its
    low coordinate would silently treat it as ``+`` on exactly the divergent loci where the
    §6 resolver is least sure.
    """
    if cds.strand not in STRANDS_RESOLVED:
        raise Gff3Error(
            f"CDS {cds.feature_id!r} has unresolved strand {cds.strand!r}; no start codon position"
        )
    return cds.start if cds.strand == STRAND_PLUS else cds.end


def is_pseudo(cds: CdsFeature) -> bool:
    """True iff the CDS is flagged pseudogenised — D4's Pfam/KO fallback trigger.

    NCBI writes ``pseudo=true`` (38 of 455 CDS rows in the committed fixture). A written-but-
    **empty** ``pseudo=`` also counts: the tag was emitted at all, and True is the conservative
    reading — a real gene sent to D4's HMM fallback costs an HMM search, whereas a pseudogene
    read as a normal gene has its frameshifted product name *trusted*. An explicit
    ``pseudo=false`` does not count. A bare ``pseudo`` with no ``=`` is not GFF3 and is refused
    upstream by :func:`parse_attributes`, not silently read as set.
    """
    values = cds.attributes.get("pseudo")
    if values is None:
        return False
    if not values:
        return True
    # EVERY value, not just the first: ``parse_attributes`` preserves repeated keys, so
    # ``pseudo=false;pseudo=true`` arrives as a two-tuple and reading ``values[0]`` alone
    # would call it a normal gene — and undercount ``n_cds_pseudo`` in the census.
    return any(not str(v).strip() or str(v).strip().lower() in _TRUEISH for v in values)


def gene_identity_text(cds: CdsFeature) -> tuple[str, ...]:
    """Every attribute value describing what this CDS *does*, in :data:`GENE_IDENTITY_KEYS` order.

    This is the raw material for D4's "product name" route and nothing more — the module makes
    no aaRS/biosynthesis/transport/transamidation judgement, which ADR-0006 leaves to
    ``criterion_c``'s already-resolved ``downstream_gene_fn`` input. Collecting it here keeps
    the *set of consulted keys* in one reviewable place instead of scattered across the
    predicate, so widening it later is a visible diff.
    """
    out: list[str] = []
    for key in GENE_IDENTITY_KEYS:
        for value in cds.attributes.get(key, ()):
            text = str(value).strip()
            if text:
                out.append(text)
    return tuple(out)
