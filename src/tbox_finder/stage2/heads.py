"""Stage-2 head label vocabularies (P3-03) — the torch-free half of the model.

:class:`Stage2Model` needs one number per head: how wide is its logit axis. PRD §10.2
names the four heads — "(a) binary T-box (calibrated); (b) boundary refinement; (c)
regulatory-mode classifier; (d) auxiliary specifier/amino-acid/tRNA-family head
(multi-task)" — and PRD §8 names the label *kinds*, but **neither the PRD nor any ADR
states a cardinality for any of them**. That gap is the whole reason this module exists,
and it is filled two different ways depending on whether a canonical source exists:

* **Derived from an in-repo canonical constant** — never re-typed here, so a change
  upstream propagates instead of going stale (the ``[[pinned-constant-that-nothing-reads]]``
  failure mode):

  - head **(b)** boundary → :data:`tbox_finder.labels.CLASS_ORDER` (the ADR-0004 D1
    8-class per-nt vocabulary, already the Stage-1 segmentation head's output space);
  - head **(c)** regulatory mode → the keys of :data:`tbox_finder.ingest.TYPE_TO_CLASS`,
    i.e. exactly the two modes PRD §8 names ("transcriptional/translational");
  - head **(d)** specifier codon → the 64 RNA triplets, enumerated from the alphabet
    rather than listed.

* **Derived from the committed Stage-2 dataset** — the cognate-amino-acid and
  tRNA-family vocabularies are TBDB spellings (``"ILE"``, ``"ILE (GAU)"``), not a
  biological constant, so there is nothing canonical to import. :func:`derive_head_spec`
  reads them off the corpus and :data:`HEAD_VOCAB_PATH` carries the result of running it
  over the **whole** committed dataset, with provenance. Deriving them per-caller would
  make a head's width depend on which split the caller happened to hold.

**Everything unmappable is masked, never guessed.** A value that is missing, blank, or
outside its vocabulary encodes to :data:`IGNORE_INDEX` (torch's cross-entropy default),
so the P3-04 loss drops that term rather than training head (d) on TBDB's 38 gap-bearing
"codons" (``"CC-"``, ``"---"``, ``"7]*"``) as if they were a 65th class. Because silent
masking and a correct model look identical from the outside, :func:`coverage_report`
counts what got masked so the number is reportable instead of invisible.

Missing values are tested with :func:`tbox_finder.masking.is_missing`, not truthiness:
the training env runs pandas 3, where a null string cell arrives as a **truthy** ``NaN``
and ``str(x or "")`` yields the string ``"nan"`` — a present-looking value in no
vocabulary (P2-10d′-c). :func:`is_admissible_label` extends that to nulls something
upstream already rendered into text, which a missing check can no longer see: nine rows
carry ``trna_family == "None (UCA)"``, composed as ``f"{amino_acid} ({anticodon})"`` from
a blank amino acid, and without the guard head (d3) would get a logit for a tRNA family
that does not exist.

Torch-free on purpose (stdlib + two in-repo imports), so the bare CI tier locks it while
:mod:`tbox_finder.stage2.model` stays behind a ``torch`` skip.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

from tbox_finder import ingest, labels, masking

__all__ = [
    "AMINO_ACID_FIELD",
    "BOUNDARY_CLASSES",
    "BOUNDARY_FIELD",
    "CODON_FIELD",
    "DERIVED_VOCAB_FIELDS",
    "HEAD_VOCAB_PATH",
    "HEAD_VOCAB_SCHEMA_VERSION",
    "IGNORE_INDEX",
    "NULL_TOKENS",
    "REGULATORY_MODE_CLASSES",
    "REGULATORY_MODE_FIELD",
    "RNA_BASES",
    "SPECIFIER_CODON_CLASSES",
    "TRNA_FAMILY_FIELD",
    "VOCAB_FIELDS",
    "Stage2HeadSpec",
    "coverage_report",
    "derive_head_spec",
    "head_vocab_document",
    "is_admissible_label",
    "load_head_spec",
    "read_head_vocab",
]

#: Cross-entropy's masked-target sentinel (``torch.nn.functional.cross_entropy``'s own
#: default ``ignore_index``). Every unmappable label encodes to this.
IGNORE_INDEX = -100

# --------------------------------------------------------------------------- #
# Field names — the Stage-2 dataset columns each head is trained against.
# They are the ``OUTPUT_COLUMNS`` spellings, so a rename upstream surfaces here.
# --------------------------------------------------------------------------- #
BOUNDARY_FIELD = "label_string"
REGULATORY_MODE_FIELD = "regulatory_mode"
CODON_FIELD = "specifier_codon"
AMINO_ACID_FIELD = "cognate_aa"
TRNA_FAMILY_FIELD = "trna_family"

#: The four categorical heads' fields, in head order (b, c, d1, d2, d3).
VOCAB_FIELDS: tuple[str, ...] = (
    BOUNDARY_FIELD,
    REGULATORY_MODE_FIELD,
    CODON_FIELD,
    AMINO_ACID_FIELD,
    TRNA_FAMILY_FIELD,
)

#: The fields with no canonical in-repo source, derived from the corpus instead.
DERIVED_VOCAB_FIELDS: tuple[str, ...] = (AMINO_ACID_FIELD, TRNA_FAMILY_FIELD)

# --------------------------------------------------------------------------- #
# Canonical vocabularies — imported/enumerated, never re-typed.
# --------------------------------------------------------------------------- #
#: Head (b). The ADR-0004 D1 per-nucleotide class order; index i == softmax index i.
BOUNDARY_CLASSES: tuple[str, ...] = labels.CLASS_ORDER

#: Head (c). PRD §8's "transcriptional/translational" — the two modes the corpus's
#: type column maps to a T-box class. ``ingest.REGULATION_ALLOWED`` additionally admits
#: ``"Unknown"``, which is a *record* state, not a mode to predict: it encodes to
#: :data:`IGNORE_INDEX` like any other unmappable value.
REGULATORY_MODE_CLASSES: tuple[str, ...] = tuple(ingest.TYPE_TO_CLASS)

#: The RNA alphabet the P3-01 adapter transcribes into (T→U).
RNA_BASES = "ACGU"

#: Head (d1). All 64 RNA triplets, enumerated. Every one of them is attested in the
#: committed corpus, so this is the genetic code rather than an over-wide guess; TBDB's
#: gap-bearing and corrupt entries fall outside it and are masked.
SPECIFIER_CODON_CLASSES: tuple[str, ...] = tuple(
    "".join(triplet) for triplet in product(RNA_BASES, repeat=3)
)

#: Schema version of the committed vocabulary document.
HEAD_VOCAB_SCHEMA_VERSION = "1.0"

#: The derived vocabularies, computed over the whole committed Stage-2 dataset.
HEAD_VOCAB_PATH = Path(__file__).with_name("head_vocab.json")


#: Spellings a *stringified* null arrives as. :func:`tbox_finder.masking.is_missing`
#: catches the real sentinels (``None`` / NaN / pandas-NA); these are what is left after
#: something upstream has already rendered one into text.
#:
#: **Re-exported, not redefined** (P3-07): the vocabulary lives in
#: :data:`tbox_finder.masking.NULL_TOKENS`, beside the missing test it completes, so the
#: label readers here and the boolean readers in ``masking.bool_or_none`` cannot drift apart.
NULL_TOKENS: frozenset[str] = masking.NULL_TOKENS


def _clean(value: Any) -> str:
    """A cell's label text, with every missing sentinel (``None``/NaN/NA) collapsed to ``""``."""
    return masking.row_text(value).strip()


def _is_null_token(text: str) -> bool:
    return text.strip().lower() in NULL_TOKENS


def is_admissible_label(field: str, text: str) -> bool:
    """Is this cleaned cell a real class, or a null that has been rendered into text?

    A stringified null is worse than a missing value: it survives every missing check,
    becomes a vocabulary entry, and gives a head a logit for a category that does not
    exist. It has to be refused where the vocabulary is built *and* where a value is
    encoded, or the two disagree about what the corpus contains.

    Two forms are refused:

    * the whole cell is a null token (``"None"``, ``"nan"``, ``"NA"``, …);
    * for ``trna_family``, the **amino-acid component** is. TBDB composes the family as
      ``f"{amino_acid} ({anticodon})"``, so a record with no amino acid renders the
      literal ``"None (UCA)"``. Nine rows of the committed corpus carry exactly that —
      every one a ``UGA`` (opal) specifier whose ``cognate_aa`` is blank, i.e. the same
      absence twice, once masked correctly and once disguised as a 62nd tRNA family.
    """
    if not text or _is_null_token(text):
        return False
    if field == TRNA_FAMILY_FIELD:
        # TBDB composes the family as f"{amino_acid} ({anticodon})", so the amino-acid
        # component is where a missing record leaks in as the literal "None".
        return not _is_null_token(text.split(" ", 1)[0])
    return True


def _label_text(field: str, value: Any) -> str:
    """``value`` as an admissible class label, or ``""`` if it is absent or a rendered null."""
    text = _clean(value)
    return text if is_admissible_label(field, text) else ""


def _validate_vocabulary(field: str, classes: tuple[str, ...]) -> None:
    """A vocabulary must be non-empty, all-non-blank, and duplicate-free."""
    if not classes:
        raise ValueError(
            f"{field}: vocabulary is empty — a head cannot have a zero-width logit axis"
        )
    blank = [i for i, name in enumerate(classes) if not str(name).strip()]
    if blank:
        raise ValueError(f"{field}: blank class label(s) at index {blank}")
    if len(set(classes)) != len(classes):
        seen: set[str] = set()
        dupes = sorted({name for name in classes if name in seen or seen.add(name)})  # type: ignore[func-returns-value]
        raise ValueError(f"{field}: duplicate class label(s) {dupes} — indices would be ambiguous")


@dataclass(frozen=True)
class Stage2HeadSpec:
    """The four heads' label vocabularies.

    :class:`~tbox_finder.stage2.model.Stage2Model` sizes every logit axis off this.

    The two corpus-derived vocabularies are **required** — there is no defensible default
    for a TBDB spelling set, and inventing one here would be a fabricated constant
    (CLAUDE.md §10.3). Build one with :func:`derive_head_spec` or :func:`load_head_spec`.

    Args:
        cognate_aa_classes: head (d2) vocabulary, corpus-derived.
        trna_family_classes: head (d3) vocabulary, corpus-derived.
        boundary_classes: head (b); defaults to the ADR-0004 D1 class order.
        regulatory_mode_classes: head (c); defaults to the two PRD §8 modes.
        specifier_codon_classes: head (d1); defaults to the 64 RNA triplets.
    """

    cognate_aa_classes: tuple[str, ...]
    trna_family_classes: tuple[str, ...]
    boundary_classes: tuple[str, ...] = BOUNDARY_CLASSES
    regulatory_mode_classes: tuple[str, ...] = REGULATORY_MODE_CLASSES
    specifier_codon_classes: tuple[str, ...] = SPECIFIER_CODON_CLASSES

    def __post_init__(self) -> None:
        # Coerce first (a caller passing a list would otherwise make the frozen dataclass
        # mutable through the back door), then validate every axis on the same rules.
        for field_name in VOCAB_FIELDS:
            attr = _FIELD_TO_ATTR[field_name]
            coerced = tuple(str(name) for name in getattr(self, attr))
            object.__setattr__(self, attr, coerced)
            _validate_vocabulary(field_name, coerced)

    # -- vocabulary access ---------------------------------------------------- #
    def classes(self, field: str) -> tuple[str, ...]:
        """``field``'s vocabulary (one of :data:`VOCAB_FIELDS`), index-aligned to its logits."""
        try:
            return getattr(self, _FIELD_TO_ATTR[field])  # type: ignore[no-any-return]
        except KeyError:
            raise KeyError(
                f"unknown head field {field!r}; expected one of {VOCAB_FIELDS}"
            ) from None

    def size(self, field: str) -> int:
        """Width of ``field``'s logit axis."""
        return len(self.classes(field))

    def index_map(self, field: str) -> dict[str, int]:
        """``class label -> logit index`` for ``field``."""
        return {name: i for i, name in enumerate(self.classes(field))}

    @property
    def head_sizes(self) -> dict[str, int]:
        """Every categorical head's logit width, keyed by dataset column."""
        return {field: self.size(field) for field in VOCAB_FIELDS}

    # -- encoding ------------------------------------------------------------- #
    def encode(self, field: str, value: Any) -> int:
        """``value`` → logit index, or :data:`IGNORE_INDEX` when missing/blank/out-of-vocabulary.

        Fail-*masked* rather than fail-loud on purpose: a decoy row legitimately carries
        ``None`` in every head-(c)/(d) column, so an unmappable value is the normal case
        and must not raise. :func:`coverage_report` is what keeps the masking visible.
        """
        text = _label_text(field, value)
        if not text:
            return IGNORE_INDEX
        return self.index_map(field).get(text, IGNORE_INDEX)

    def encode_boundary_string(self, value: Any) -> list[int]:
        """A ``label_string`` → per-nucleotide head-(b) targets; ``[]`` when the row has none.

        Maps through :data:`tbox_finder.labels.CODE_TO_CLASS` and then this spec's own
        boundary vocabulary, so the two cannot silently disagree. An unrecognised code
        character **raises** — unlike a missing whole-row label, a stray glyph inside a
        present label string means the upstream derivation changed, and masking it would
        hide that.
        """
        text = _clean(value)
        if not text:
            return []
        index_map = self.index_map(BOUNDARY_FIELD)
        out: list[int] = []
        for position, char in enumerate(text):
            class_name = labels.CODE_TO_CLASS.get(char)
            if class_name is None:
                raise ValueError(
                    f"label_string position {position}: unknown class code {char!r} "
                    f"(known: {sorted(labels.CODE_TO_CLASS)})"
                )
            index = index_map.get(class_name)
            if index is None:
                raise ValueError(
                    f"label_string position {position}: class {class_name!r} is absent from this "
                    f"spec's boundary vocabulary {self.boundary_classes}"
                )
            out.append(index)
        return out

    # -- serialisation -------------------------------------------------------- #
    def to_dict(self) -> dict[str, list[str]]:
        """The five vocabularies as plain lists, keyed by dataset column."""
        return {field: list(self.classes(field)) for field in VOCAB_FIELDS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Stage2HeadSpec:
        """Inverse of :meth:`to_dict`.

        The three **canonical** axes fall back to their in-repo defaults when absent. The
        two **corpus-derived** axes are required and have no defensible default, so a
        payload missing one is refused by name rather than raising a bare ``KeyError``
        that says only which dict key was looked up.
        """
        missing = [field for field in DERIVED_VOCAB_FIELDS if field not in payload]
        if missing:
            raise KeyError(
                f"head vocabulary document is missing the required corpus-derived "
                f"{missing} — regenerate it with derive_head_spec over the full dataset"
            )
        spec = cls(
            cognate_aa_classes=tuple(payload[AMINO_ACID_FIELD]),
            trna_family_classes=tuple(payload[TRNA_FAMILY_FIELD]),
        )
        overrides = {
            _FIELD_TO_ATTR[field]: tuple(payload[field])
            for field in (BOUNDARY_FIELD, REGULATORY_MODE_FIELD, CODON_FIELD)
            if field in payload
        }
        return replace(spec, **overrides) if overrides else spec


_FIELD_TO_ATTR: dict[str, str] = {
    BOUNDARY_FIELD: "boundary_classes",
    REGULATORY_MODE_FIELD: "regulatory_mode_classes",
    CODON_FIELD: "specifier_codon_classes",
    AMINO_ACID_FIELD: "cognate_aa_classes",
    TRNA_FAMILY_FIELD: "trna_family_classes",
}


def derive_head_spec(rows: Iterable[Mapping[str, Any]]) -> Stage2HeadSpec:
    """Build a spec by reading the corpus-derived vocabularies off ``rows``.

    Only :data:`DERIVED_VOCAB_FIELDS` are read; the other three axes stay canonical.
    Values are sorted so the same corpus always yields the same logit indices regardless
    of row order, and blank/missing cells (every decoy row) contribute nothing.

    Run this over the **whole** dataset, not a split: a vocabulary derived from the train
    fold alone would give the test fold classes the head has no logit for.
    """
    observed: dict[str, set[str]] = {field: set() for field in DERIVED_VOCAB_FIELDS}
    for row in rows:
        for field in DERIVED_VOCAB_FIELDS:
            text = _label_text(field, row.get(field))
            if text:
                observed[field].add(text)
    return Stage2HeadSpec(
        cognate_aa_classes=tuple(sorted(observed[AMINO_ACID_FIELD])),
        trna_family_classes=tuple(sorted(observed[TRNA_FAMILY_FIELD])),
    )


def coverage_report(spec: Stage2HeadSpec, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count, per head field, what ``spec`` maps and what it masks.

    Masking is invisible by construction — a masked target and a correctly-encoded one
    produce the same shaped tensor — so the only way a reviewer can tell head (d1) is
    trained on 23,143 codons rather than 23,535 is if something counts. This does.

    Returns ``{"n_rows": int, "fields": {field: {"present", "encoded", "masked",
    "out_of_vocabulary", "stringified_null", "examples"}}}``. The three masked reasons are
    kept apart on purpose — ``absent`` (``masked - out_of_vocabulary - stringified_null``)
    is expected and benign for every decoy row, ``out_of_vocabulary`` is a real value the
    vocabulary does not cover, and ``stringified_null`` is
    :func:`is_admissible_label`'s catch. Collapsing them would make a data defect read
    exactly like a decoy. ``examples`` lists up to 10 of the refused values.
    """
    fields = [f for f in VOCAB_FIELDS if f != BOUNDARY_FIELD]
    stats: dict[str, dict[str, Any]] = {
        field: {
            "present": 0,
            "encoded": 0,
            "masked": 0,
            "out_of_vocabulary": 0,
            "stringified_null": 0,
            "_seen": set(),
        }
        for field in fields
    }
    boundary = {"rows_with_label": 0, "rows_masked": 0, "n_positions": 0}
    n_rows = 0
    for row in rows:
        n_rows += 1
        for field in fields:
            bucket = stats[field]
            raw = _clean(row.get(field))
            if not raw:
                bucket["masked"] += 1
                continue
            bucket["present"] += 1
            if not is_admissible_label(field, raw):
                bucket["masked"] += 1
                bucket["stringified_null"] += 1
                if len(bucket["_seen"]) < 10:
                    bucket["_seen"].add(raw)
            elif spec.encode(field, raw) == IGNORE_INDEX:
                bucket["masked"] += 1
                bucket["out_of_vocabulary"] += 1
                if len(bucket["_seen"]) < 10:
                    bucket["_seen"].add(raw)
            else:
                bucket["encoded"] += 1
        label_text = _clean(row.get(BOUNDARY_FIELD))
        if label_text:
            boundary["rows_with_label"] += 1
            boundary["n_positions"] += len(label_text)
        else:
            boundary["rows_masked"] += 1
    for bucket in stats.values():
        bucket["examples"] = sorted(bucket.pop("_seen"))
    return {"n_rows": n_rows, "fields": stats, BOUNDARY_FIELD: boundary}


def head_vocab_document(
    spec: Stage2HeadSpec, *, provenance: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The committed-artifact form of ``spec``: vocabularies + sizes + provenance.

    ``sizes`` is redundant with ``len(vocabularies[field])`` **on purpose** — it is the
    number a reader compares against a model's logit shape, and :func:`load_head_spec`
    refuses a document where the two disagree, so the redundancy is a checksum rather
    than a second source of truth.
    """
    return {
        "schema_version": HEAD_VOCAB_SCHEMA_VERSION,
        "step": "P3-03",
        "prd": "PRD §8, §10.2",
        "adr": "ADR-0004 (D1 class order); ADR-0002 (RiNALMo Stage-2)",
        "canonical_sources": {
            BOUNDARY_FIELD: "tbox_finder.labels.CLASS_ORDER",
            REGULATORY_MODE_FIELD: "tbox_finder.ingest.TYPE_TO_CLASS (keys)",
            CODON_FIELD: f"itertools.product({RNA_BASES!r}, repeat=3)",
            AMINO_ACID_FIELD: "derived from the committed Stage-2 dataset",
            TRNA_FAMILY_FIELD: "derived from the committed Stage-2 dataset",
        },
        "sizes": spec.head_sizes,
        "vocabularies": spec.to_dict(),
        "provenance": dict(provenance or {}),
    }


def read_head_vocab(path: str | Path = HEAD_VOCAB_PATH) -> dict[str, Any]:
    """The raw committed vocabulary document (vocabularies + sizes + provenance)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_head_spec(path: str | Path = HEAD_VOCAB_PATH) -> Stage2HeadSpec:
    """Load :data:`HEAD_VOCAB_PATH` into a spec, checking the document against itself.

    Refuses a document written under another ``schema_version``; one whose ``sizes`` and
    ``vocabularies`` do not describe the same axes, or disagree on a width; or whose
    canonical axes have drifted from the in-repo constants — the last means the file was
    written against a different class order, and loading it would silently re-index every
    target.

    The ``sizes`` block is a checksum, so both directions of the key-set have to match: a
    ``sizes`` key with no vocabulary is an inconsistency (not a bare ``KeyError`` on a
    lookup), and a vocabulary with no ``sizes`` entry is silently *unchecked*, which
    removes the redundancy the block exists to provide.
    """
    document = read_head_vocab(path)
    version = document.get("schema_version")
    if version != HEAD_VOCAB_SCHEMA_VERSION:
        # A version field nothing reads records intent and enforces nothing.
        raise ValueError(
            f"{path}: schema_version {version!r} is not the supported "
            f"{HEAD_VOCAB_SCHEMA_VERSION!r} — regenerate the document"
        )
    vocabularies = document["vocabularies"]
    sizes = document.get("sizes", {})
    if set(sizes) != set(vocabularies):
        raise ValueError(
            f"{path}: sizes describes {sorted(sizes)} but vocabularies has "
            f"{sorted(vocabularies)} — the document is internally inconsistent"
        )
    for field, expected in sizes.items():
        actual = len(vocabularies[field])
        if int(expected) != actual:
            raise ValueError(
                f"{path}: sizes[{field!r}] = {expected} but vocabularies[{field!r}] has {actual} "
                f"entries — the document is internally inconsistent"
            )
    for field, canonical in (
        (BOUNDARY_FIELD, BOUNDARY_CLASSES),
        (REGULATORY_MODE_FIELD, REGULATORY_MODE_CLASSES),
        (CODON_FIELD, SPECIFIER_CODON_CLASSES),
    ):
        stored = tuple(vocabularies.get(field, canonical))
        if stored != canonical:
            raise ValueError(
                f"{path}: {field!r} vocabulary has drifted from its canonical in-repo source "
                f"({len(stored)} entries vs {len(canonical)}); regenerate the document"
            )
    return Stage2HeadSpec.from_dict(vocabularies)
