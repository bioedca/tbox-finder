"""P3-01 — assemble the Stage-2 supervised dataset (PRD §6, §8, §10.2, §11).

Emits ``data/processed/stage2_dataset.parquet``: one row per RNA sequence Stage 2 will
ever be asked to score in training, calibration, or evaluation — the 23,535 T-box
positives plus the §9.1 static decoy pools — each carrying its binary label, its
per-nucleotide boundary target, the §8 aux targets (regulatory mode, specifier codon,
cognate amino acid, tRNA family), a dot-bracket pairing target for the P3-05
structure-consistency loss, and the ADR-0004 fold/parentage columns.

Three decisions in this module are load-bearing and are stated here because a reader
must not have to reverse-engineer them from the code.

**1. Sequence-only, RNA, one locus per row.** ``rna_sequence`` is the T→U transcription
of the corpus record's ``FASTA_sequence`` (or the decoy's sequence). Structure appears
**only** as the ``pairing_dotbracket`` *target* column — PRD §6 forbids a structure input
channel for Stage 2, so nothing here may be fed to the model as input.

**2. Flank is 0 nt in v0, deliberately.** PRD §6 hands Stage 2 a "locus ± flank". The
named P3-01 inputs supply real genomic flank for positives (P2-00's ``context_v0``) but
**none** for any of the four decoy pools, which are generated, Rfam-sourced, or shuffled.
A positives-only flank would make "is embedded in real genomic context" a perfectly
separable shortcut for the binary head — precisely the §5 circularity failure the project
exists to avoid. ``flank_nt`` is therefore a pinned parameter, recorded per row and in
provenance, and v0 pins it to 0 for **both** classes. Raising it is a dataset change that
must raise it symmetrically.

**3. Folds are inherited, never invented.** A row's fold comes from the committed
ADR-0004 D7 split table by one of exactly three routes, recorded in ``fold_basis``:

* ``corpus_record`` — a positive: every scheme column is copied verbatim from its own
  split-table row.
* ``parent_record`` — a decoy derived from a corpus positive (the dinucleotide-shuffled
  pool): it **inherits its parent's fold across every scheme**, which is ADR-0004 D7's
  variant→parent→fold rule applied to the negative side. Fail-closed: a derived decoy
  whose parent is not in the split table is refused, never floated into training.
* ``decoy_pool_random`` — a decoy with **no corpus parent** (generated GC-matched
  background, Rfam structured RNAs, tboxevo leader decoys). These have no cluster, no
  clade and no genomic neighbourhood in the corpus, so no leave-clade-out holdout unit
  can contain them and inventing one would be fabrication. They are handled the way
  ADR-0004 D4 handles taxonomy-incomplete positives — **kept only in the random split** —
  with every clade-scheme column left null and ``clade_holdout_eligible`` False. Their
  ``fold_random`` is a deterministic keyed-hash assignment (:data:`DECOY_FOLD_SEED`), so
  it is reproducible without an RNG walk and does not depend on row order.

``nested_train`` stays **null** for that third class rather than being set True: whether a
parentless decoy may enter the nested training fold is a P3-03 sampling policy, and a
``True`` here would be a policy decision disguised as a data field.

**Pairing target — measured, fail-closed.** TBDB's ``Structure`` (WUSS) is aligned to
``Sequence``, the gapped cmalign row, **not** to the genomic ``FASTA_sequence`` locus
(median 238 vs 281 nt). The target is therefore *projected*: validate the alignment row,
strip gaps, and require the gap-stripped sequence to occur **exactly once** in the locus,
then map each ``<``/``>`` pair through that offset. Measured on the full corpus, every
record whose ``Sequence`` uses a clean nucleotide alphabet anchors uniquely — 18,269 /
23,535 (77.6 %), with zero no-hit and zero multi-hit rows. The remaining 5,266 render
unaligned inserts as ``~`` / ``[n]`` / ``*`` with the nucleotides **elided**, so their
alignment row is a lossy view that cannot be re-anchored; they are recorded
``pairing_status = "unanchorable_alignment"`` with a **null** target. No approximate or
re-folded structure is ever substituted (CLAUDE.md §10.3).

``Structure`` is the **model/consensus** structure line aligned to this member, so a
minority of its pairs fall on columns this sequence has deleted: measured, 20,854 of
850,029 pairs (2.5 %), touching 8,634 of the 18,269 anchorable rows, per-row median
fraction 0.000 and max 0.426. A pair with a deleted partner is **not formed in this
sequence**, so it is dropped from that row's target and counted in
``n_pairs_dropped_gap`` — never invented as a pair to the neighbouring base, and never
grounds for discarding the other 97.5 % of the row's real pairs. Dropping pairs from a
nested set leaves it nested, so the emitted dot-bracket stays well-formed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder import ingest, masking, provenance
from tbox_finder.stage2 import tokenizer as rna_tokenizer

SCHEMA_VERSION = "1.0"
STEP = "P3-01"

DEFAULT_CORPUS = Path("data/processed/master_clean_v0.parquet")
DEFAULT_LABELS = Path("data/processed/labels/labels_v0.parquet")
DEFAULT_SPLIT_TABLE = Path("data/processed/splits/split_assignments.parquet")
DEFAULT_DECOYS = Path("data/processed/negatives/decoys_v0.parquet")

OUT_PARQUET_NAME = "stage2_dataset.parquet"
OUT_PROVENANCE_NAME = "stage2_dataset.provenance.json"
REPORT_NAME = "stage2_dataset_report.json"
DEFAULT_OUT_DIR = Path("data/processed")
DEFAULT_AUDIT_DIR = Path("data/processed/audits")

#: v0 pins the supervised pool to the bare locus for both classes (module docstring, §2).
DEFAULT_FLANK_NT = 0

#: Keyed-hash seed for the parentless-decoy random-fold assignment. Pinned so the
#: assignment is reproducible from the row id alone, with no RNG-order dependence.
DECOY_FOLD_SEED = 20260731
#: train / val / test mass for that assignment.
DECOY_FOLD_FRACTIONS: tuple[float, float, float] = (0.80, 0.10, 0.10)

SOURCE_CORPUS = "corpus"
SOURCE_DECOY = "decoy"
POOL_CORPUS = "corpus"

FOLD_BASIS_CORPUS = "corpus_record"
FOLD_BASIS_PARENT = "parent_record"
FOLD_BASIS_DECOY_RANDOM = "decoy_pool_random"
FOLD_BASES: tuple[str, ...] = (FOLD_BASIS_CORPUS, FOLD_BASIS_PARENT, FOLD_BASIS_DECOY_RANDOM)

PAIRING_ANCHORED = "anchored"
PAIRING_UNANCHORABLE = "unanchorable_alignment"
PAIRING_NOT_APPLICABLE = "not_applicable"
PAIRING_STATUSES: tuple[str, ...] = (
    PAIRING_ANCHORED,
    PAIRING_UNANCHORABLE,
    PAIRING_NOT_APPLICABLE,
)

FOLD_RANDOM_VALUES: tuple[str, str, str] = ("train", "val", "test")

# --- input column names -------------------------------------------------------------
CORPUS_SEQ_COL = "FASTA_sequence"
CORPUS_ALIGNED_SEQ_COL = "Sequence"
CORPUS_STRUCTURE_COL = "Structure"
LABELS_ID_COL = ingest.RECORD_HASH_COL  # "record_sha256"
LABEL_STRING_COL = "label_string"
SPLIT_ID_COL = "record_id"
SPLIT_PARENT_COL = "parent_record_id"
SPLIT_SOURCE_COL = "source"

#: Copied verbatim onto a row whose fold basis is a corpus record (its own or its
#: parent's). Order is the artifact's column order for these fields.
SPLIT_CARRIED_COLUMNS: tuple[str, ...] = (
    "klass",
    "cluster_id",
    "resolved_phylum",
    "resolved_class",
    "resolved_order",
    "resolved_genus",
    "fold_random",
    "loo_order_unit",
    "class_holdout_unit",
    "phylum_holdout_unit",
    "nested_train",
    "nested_role",
    "is_designated_loo_holdout",
    "dropped_from_clade_holdout",
)

#: Exact output schema (order included). Asserted by the unit tests, so adding a column
#: is a deliberate, reviewed act rather than a silent widening.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "row_id",
    "source",
    "pool",
    "parent_record_id",
    "rna_sequence",
    "seq_length",
    "n_tokens",
    "flank_nt",
    "is_tbox",
    "label_string",
    "regulatory_mode",
    "specifier_codon",
    "cognate_aa",
    "trna_family",
    "pairing_dotbracket",
    "pairing_status",
    "n_base_pairs",
    "n_pairs_dropped_gap",
    *SPLIT_CARRIED_COLUMNS,
    "fold_basis",
    "clade_holdout_eligible",
)

# --- WUSS / dot-bracket -------------------------------------------------------------
#: The only nucleotide glyphs a projectable alignment row may use. Anything else (``*``,
#: ``[``, ``]``, digits, spaces) marks TBDB's lossy unaligned-insert rendering, where
#: nucleotides are elided and the row cannot be re-anchored (module docstring).
ALIGNMENT_SEQ_ALPHABET = frozenset("ACGUNacgun-")
GAP_CHAR = "-"
WUSS_OPEN = "<"
WUSS_CLOSE = ">"
#: Unpaired WUSS glyphs seen in the corpus ``Structure`` column.
WUSS_UNPAIRED = frozenset("-_,.:~")
WUSS_ALPHABET = WUSS_UNPAIRED | {WUSS_OPEN, WUSS_CLOSE}

DOT = "."
OPEN = "("
CLOSE = ")"
UNPAIRED_PARTNER = -1

#: Why a record produced no pairing target. Reported per reason so a drop in coverage
#: names its own cause instead of showing up as an unexplained count.
REJECT_NULL_FIELD = "null_alignment_field"
REJECT_LENGTH_MISMATCH = "seq_structure_length_mismatch"
REJECT_SEQ_ALPHABET = "lossy_insert_rendering"
REJECT_STRUCT_ALPHABET = "unexpected_wuss_glyph"
REJECT_UNBALANCED = "unbalanced_wuss"
REJECT_NO_ANCHOR = "no_exact_locus_match"
REJECT_MULTI_ANCHOR = "ambiguous_locus_match"
REJECT_REASONS: tuple[str, ...] = (
    REJECT_NULL_FIELD,
    REJECT_LENGTH_MISMATCH,
    REJECT_SEQ_ALPHABET,
    REJECT_STRUCT_ALPHABET,
    REJECT_UNBALANCED,
    REJECT_NO_ANCHOR,
    REJECT_MULTI_ANCHOR,
)

__all__ = [
    "DECOY_FOLD_SEED",
    "DEFAULT_FLANK_NT",
    "FOLD_BASES",
    "OUTPUT_COLUMNS",
    "PAIRING_STATUSES",
    "REJECT_REASONS",
    "build_dataset",
    "dataset_digest",
    "decoy_fold",
    "dot_bracket_to_partners",
    "project_structure_to_locus",
    "run_stage2_dataset",
    "wuss_pairs",
]


# ------------------------------------------------------------------ structure helpers


def wuss_pairs(structure: str) -> list[tuple[int, int]] | None:
    """Nested ``<``/``>`` pairs of a WUSS string as 0-based ``(i, j)``, ``i < j``.

    Returns ``None`` when the string uses an unexpected glyph or its brackets do not
    balance — a rejection, not a repair. The corpus ``Structure`` column carries no
    pseudoknot brackets (measured: the glyph set is exactly ``<>-_,.:~``), so a nested
    stack parse is complete for it; an unexpected glyph therefore means the input is not
    the annotation this function was written for and must not be guessed at.
    """
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for i, ch in enumerate(structure):
        if ch == WUSS_OPEN:
            stack.append(i)
        elif ch == WUSS_CLOSE:
            if not stack:
                return None
            pairs.append((stack.pop(), i))
        elif ch not in WUSS_UNPAIRED:
            return None
    if stack:
        return None
    pairs.sort()
    return pairs


def dot_bracket_to_partners(dot_bracket: str) -> list[int]:
    """``()``-dot-bracket → per-base 0-based partner index, :data:`UNPAIRED_PARTNER` if unpaired.

    The P3-05 structure-consistency loss reads this; the artifact stores the compact
    string (as ``labels_v0`` stores ``label_string``) and decodes here.
    """
    partners = [UNPAIRED_PARTNER] * len(dot_bracket)
    stack: list[int] = []
    for i, ch in enumerate(dot_bracket):
        if ch == OPEN:
            stack.append(i)
        elif ch == CLOSE:
            if not stack:
                raise ValueError(f"unbalanced dot-bracket at position {i}")
            j = stack.pop()
            partners[i] = j
            partners[j] = i
        elif ch != DOT:
            raise ValueError(f"unexpected dot-bracket glyph {ch!r} at position {i}")
    if stack:
        raise ValueError(f"unbalanced dot-bracket: {len(stack)} unclosed pair(s)")
    return partners


def project_structure_to_locus(
    *,
    aligned_sequence: Any,
    aligned_structure: Any,
    locus_dna: str,
) -> tuple[str | None, str, int]:
    """Project a gapped WUSS alignment row onto the genomic locus.

    Returns ``(dot_bracket_or_None, status_or_reason, n_pairs_dropped_gap)``: on success
    the dot-bracket is ``len(locus_dna)`` long and the second element is
    :data:`PAIRING_ANCHORED`; on failure the dot-bracket is ``None`` and the second
    element is one of :data:`REJECT_REASONS`.

    The anchor is an **exact, unique** substring match of the gap-stripped alignment row
    (in DNA form) inside the locus. Uniqueness is required, not just existence: a
    repeated match would place the pairs at an arbitrary one of several offsets, and a
    silently mis-placed structure target is worse than an absent one.

    ``Structure`` is a consensus line, so a pair may have a partner in a column this
    sequence deleted. Such a pair is not formed here: it is **dropped and counted**
    (third return value), not re-based onto the neighbouring nucleotide and not treated
    as grounds for rejecting the row (module docstring).
    """
    if masking.is_missing(aligned_sequence) or masking.is_missing(aligned_structure):
        return None, REJECT_NULL_FIELD, 0
    seq = str(aligned_sequence)
    struct = str(aligned_structure)
    if len(seq) != len(struct):
        return None, REJECT_LENGTH_MISMATCH, 0
    if not set(seq) <= ALIGNMENT_SEQ_ALPHABET:
        return None, REJECT_SEQ_ALPHABET, 0
    if not set(struct) <= WUSS_ALPHABET:
        return None, REJECT_STRUCT_ALPHABET, 0

    pairs = wuss_pairs(struct)
    if pairs is None:
        return None, REJECT_UNBALANCED, 0

    # aligned column -> ungapped index (None for a gap column)
    ungapped_index: list[int | None] = []
    ungapped_chars: list[str] = []
    for ch in seq:
        if ch == GAP_CHAR:
            ungapped_index.append(None)
        else:
            ungapped_index.append(len(ungapped_chars))
            ungapped_chars.append(ch.upper())

    probe = "".join(ungapped_chars).replace("U", "T")
    locus = str(locus_dna).upper()
    # `str.count` counts NON-overlapping occurrences ("AAA".count("AA") == 1), so it can
    # report a genuinely ambiguous probe as unique and this function would then place the
    # structure at the first offset — the exact silent mis-placement the uniqueness rule
    # exists to prevent. Scan every start position instead. (Measured: 0 of 23,535 corpus
    # records differ between the two counts, so this changes no current number; it closes
    # the hole rather than papering over a symptom.)
    offset = locus.find(probe)
    if offset < 0:
        return None, REJECT_NO_ANCHOR, 0
    if locus.find(probe, offset + 1) >= 0:
        return None, REJECT_MULTI_ANCHOR, 0

    chars = [DOT] * len(locus)
    dropped = 0
    for i, j in pairs:
        ui = ungapped_index[i]
        uj = ungapped_index[j]
        if ui is None or uj is None:
            dropped += 1
            continue
        chars[offset + ui] = OPEN
        chars[offset + uj] = CLOSE
    return "".join(chars), PAIRING_ANCHORED, dropped


# ------------------------------------------------------------------------ fold routing


def decoy_fold(row_id: str, *, seed: int = DECOY_FOLD_SEED) -> str:
    """Deterministic ``train``/``val``/``test`` for a decoy with no corpus parent.

    Keyed on the decoy id, not on an RNG walk, so the assignment does not depend on row
    order, pool size, or how many decoys were built before it — re-running the build on a
    superset of the pool leaves every existing assignment unchanged.
    """
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    # 53 bits keeps the ratio exactly representable in a float without bias worth naming.
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    train, val, _test = DECOY_FOLD_FRACTIONS
    if unit < train:
        return FOLD_RANDOM_VALUES[0]
    if unit < train + val:
        return FOLD_RANDOM_VALUES[1]
    return FOLD_RANDOM_VALUES[2]


def _carried(split_row: Mapping[str, Any]) -> dict[str, Any]:
    return {c: split_row[c] for c in SPLIT_CARRIED_COLUMNS}


def _null_carried() -> dict[str, Any]:
    return dict.fromkeys(SPLIT_CARRIED_COLUMNS)


# ------------------------------------------------------------------------- the builder


def _text(value: Any) -> str | None:
    """A nullable string cell as ``str`` or ``None`` (pandas-2/3 safe)."""
    return None if masking.is_missing(value) else str(value)


def build_dataset(
    *,
    corpus: Any,
    labels: Any,
    splits: Any,
    decoys: Any,
    flank_nt: int = DEFAULT_FLANK_NT,
    decoy_fold_seed: int = DECOY_FOLD_SEED,
    require_full_join: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Assemble the Stage-2 supervised table.

    Args:
        corpus: the cleaned corpus frame (``master_clean_v0`` schema, **no** id column —
            record ids are recomputed here with :func:`ingest.compute_record_hashes`, the
            project's single identity anchor).
        labels: the P0-20 ``labels_v0`` frame, keyed by ``record_sha256``.
        splits: the committed ADR-0004 D7 split-assignment frame.
        decoys: the §9.1 ``decoys_v0`` frame (``masked`` rows are dropped here).
        flank_nt: nucleotides of genomic flank added to each locus. v0 pins 0 and any
            other value is refused, because the decoy pools cannot supply a matching
            flank and an asymmetric one is a shortcut feature (module docstring, §2).
        decoy_fold_seed: key for :func:`decoy_fold`.
        require_full_join: when True (the default) every corpus record must join to both
            the labels and the split table, and every derived decoy must find its parent
            — anything else raises. The golden fixture sets it False (its 100-record
            slice predates the corpus de-duplication) and asserts the refused count, so
            the relaxation is never silent.

    Returns:
        ``(dataframe, report)``.
    """
    import pandas as pd

    if int(flank_nt) != 0:
        raise ValueError(
            f"flank_nt={flank_nt}: v0 pins the supervised pool to the bare locus for both "
            "classes; a positives-only flank is a separable shortcut (PRD §5). Raising it "
            "requires supplying a matching flank for every decoy pool."
        )

    corpus = corpus.reset_index(drop=True)
    record_ids = ingest.compute_record_hashes(corpus)

    label_by_id = {
        str(r[LABELS_ID_COL]): r
        for r in labels.to_dict("records")
        if not masking.is_missing(r[LABELS_ID_COL])
    }
    split_by_id = {
        str(r[SPLIT_ID_COL]): r
        for r in splits.to_dict("records")
        if not masking.is_missing(r[SPLIT_ID_COL])
    }

    rows: list[dict[str, Any]] = []
    pairing_counts: dict[str, int] = dict.fromkeys(PAIRING_STATUSES, 0)
    reject_counts: dict[str, int] = dict.fromkeys(REJECT_REASONS, 0)
    refused_positives: list[str] = []
    n_pairs_dropped_gap = 0
    n_rows_with_dropped_pairs = 0

    for record_id, record in zip(record_ids, corpus.to_dict("records"), strict=True):
        label_row = label_by_id.get(record_id)
        split_row = split_by_id.get(record_id)
        if label_row is None or split_row is None:
            refused_positives.append(record_id)
            continue

        locus_dna = str(record[CORPUS_SEQ_COL]).upper()
        rna = rna_tokenizer.transcribe(locus_dna)
        rna_tokenizer.assert_within_context(rna, row_id=record_id)

        dot_bracket, status_or_reason, dropped_gap = project_structure_to_locus(
            aligned_sequence=record.get(CORPUS_ALIGNED_SEQ_COL),
            aligned_structure=record.get(CORPUS_STRUCTURE_COL),
            locus_dna=locus_dna,
        )
        if dot_bracket is None:
            pairing_status = PAIRING_UNANCHORABLE
            reject_counts[status_or_reason] += 1
            n_pairs = 0
        else:
            pairing_status = PAIRING_ANCHORED
            n_pairs = dot_bracket.count(OPEN)
        pairing_counts[pairing_status] += 1
        n_pairs_dropped_gap += dropped_gap
        if dropped_gap:
            n_rows_with_dropped_pairs += 1

        label_string = _text(label_row[LABEL_STRING_COL])
        n_labelled = 0 if label_string is None else len(label_string)
        if n_labelled != len(rna):
            raise ValueError(
                f"{record_id}: label_string length {n_labelled} != locus length {len(rna)}; "
                "the per-nucleotide target would be misaligned"
            )

        rows.append(
            {
                "row_id": record_id,
                "source": SOURCE_CORPUS,
                "pool": POOL_CORPUS,
                # `_text`, not `str()`: under pandas 3 a null string cell arrives as NaN
                # and `str(NaN)` is the *present-looking* value "nan", which would sail
                # past the null gate in `_assert_dataset_invariants` as a fabricated
                # parent link ([[pandas-3-nan-truthy-in-training-env]]).
                "parent_record_id": _text(split_row[SPLIT_PARENT_COL]),
                "rna_sequence": rna,
                "seq_length": len(rna),
                "n_tokens": rna_tokenizer.token_length(rna),
                "flank_nt": int(flank_nt),
                "is_tbox": True,
                "label_string": label_string,
                "regulatory_mode": _text(label_row["regulatory_mode"]),
                "specifier_codon": _text(label_row["specifier_codon"]),
                "cognate_aa": _text(label_row["cognate_aa"]),
                "trna_family": _text(label_row["trna_family"]),
                "pairing_dotbracket": dot_bracket,
                "pairing_status": pairing_status,
                "n_base_pairs": n_pairs,
                "n_pairs_dropped_gap": dropped_gap,
                **_carried(split_row),
                "fold_basis": FOLD_BASIS_CORPUS,
                "clade_holdout_eligible": not bool(split_row["dropped_from_clade_holdout"]),
            }
        )

    if refused_positives and require_full_join:
        raise ValueError(
            f"{len(refused_positives)} corpus record(s) have no labels/split row "
            f"(first: {refused_positives[0]}); the fold and the per-nt target cannot be "
            "supplied and a Stage-2 row without them is unusable"
        )

    # ---- negatives -----------------------------------------------------------------
    n_decoys_masked = 0
    refused_decoys: list[str] = []
    for decoy in decoys.to_dict("records"):
        if bool(decoy["masked"]):
            # Union-prior / own-positive overlap: a known T-box locus is not a negative.
            n_decoys_masked += 1
            continue
        decoy_id = str(decoy["decoy_id"])
        sequence = str(decoy["sequence"]).upper()
        rna = rna_tokenizer.transcribe(sequence)
        rna_tokenizer.assert_within_context(rna, row_id=decoy_id)

        parent_id = _text(decoy.get("source_record_id"))
        if parent_id is None:
            carried = _null_carried()
            carried["fold_random"] = decoy_fold(decoy_id, seed=decoy_fold_seed)
            fold_basis = FOLD_BASIS_DECOY_RANDOM
            parent = decoy_id
            clade_eligible = False
        else:
            split_row = split_by_id.get(parent_id)
            if split_row is None:
                refused_decoys.append(decoy_id)
                continue
            carried = _carried(split_row)
            fold_basis = FOLD_BASIS_PARENT
            parent = parent_id
            clade_eligible = not bool(split_row["dropped_from_clade_holdout"])

        pairing_counts[PAIRING_NOT_APPLICABLE] += 1
        rows.append(
            {
                "row_id": decoy_id,
                "source": SOURCE_DECOY,
                "pool": str(decoy["pool"]),
                "parent_record_id": parent,
                "rna_sequence": rna,
                "seq_length": len(rna),
                "n_tokens": rna_tokenizer.token_length(rna),
                "flank_nt": int(flank_nt),
                "is_tbox": False,
                "label_string": None,
                "regulatory_mode": None,
                "specifier_codon": None,
                "cognate_aa": None,
                "trna_family": None,
                "pairing_dotbracket": None,
                "pairing_status": PAIRING_NOT_APPLICABLE,
                "n_base_pairs": 0,
                "n_pairs_dropped_gap": 0,
                **carried,
                "fold_basis": fold_basis,
                "clade_holdout_eligible": clade_eligible,
            }
        )

    if refused_decoys and require_full_join:
        raise ValueError(
            f"{len(refused_decoys)} derived decoy(s) name a parent absent from the split "
            f"table (first: {refused_decoys[0]}); ADR-0004 D7 requires a derived record to "
            "inherit its parent's fold, so it cannot be admitted"
        )

    frame = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    _assert_dataset_invariants(frame)

    positives = frame[frame["is_tbox"]]
    negatives = frame[~frame["is_tbox"]]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "n_rows": int(len(frame)),
        "n_positives": int(len(positives)),
        "n_negatives": int(len(negatives)),
        "flank_nt": int(flank_nt),
        "decoy_fold_seed": int(decoy_fold_seed),
        "tokenizer": {
            "repo_id": rna_tokenizer.REPO_ID,
            "revision": rna_tokenizer.REVISION,
            "vocab_size": len(rna_tokenizer.VOCAB),
            "vocab_digest": rna_tokenizer.vocab_digest(),
            "max_nucleotide_tokens": rna_tokenizer.MAX_NUCLEOTIDE_TOKENS,
        },
        "max_seq_length": int(frame["seq_length"].max()) if len(frame) else 0,
        "max_n_tokens": int(frame["n_tokens"].max()) if len(frame) else 0,
        "pool_counts": {str(k): int(v) for k, v in frame["pool"].value_counts().items()},
        "fold_basis_counts": {
            str(k): int(v) for k, v in frame["fold_basis"].value_counts().items()
        },
        "fold_random_counts": {
            str(k): int(v) for k, v in frame["fold_random"].value_counts().items()
        },
        "clade_holdout_eligible": int(frame["clade_holdout_eligible"].sum()),
        "pairing_status_counts": pairing_counts,
        "pairing_reject_reasons": reject_counts,
        "pairing_coverage_over_positives": (
            float(pairing_counts[PAIRING_ANCHORED] / len(positives)) if len(positives) else 0.0
        ),
        "n_base_pairs_total": int(frame["n_base_pairs"].sum()),
        "n_pairs_dropped_gap_total": int(n_pairs_dropped_gap),
        "n_rows_with_pairs_dropped_gap": int(n_rows_with_dropped_pairs),
        "n_anchored_rows_with_zero_pairs": int(
            ((frame["pairing_status"] == PAIRING_ANCHORED) & (frame["n_base_pairs"] == 0)).sum()
        ),
        "n_decoys_masked_out": int(n_decoys_masked),
        "n_corpus_refused_unjoined": int(len(refused_positives)),
        "n_decoys_refused_no_parent": int(len(refused_decoys)),
        "require_full_join": bool(require_full_join),
        "digest": dataset_digest(frame),
    }
    return frame, report


def _assert_dataset_invariants(frame: Any) -> None:
    """The P3-01 validation gate, enforced at build time as well as in the tests."""
    if list(frame.columns) != list(OUTPUT_COLUMNS):
        raise ValueError(f"unexpected schema: {list(frame.columns)}")
    if not len(frame):
        raise ValueError("empty dataset: the build produced no rows")
    if frame["row_id"].duplicated().any():
        dup = frame.loc[frame["row_id"].duplicated(), "row_id"].iloc[0]
        raise ValueError(f"duplicate row_id {dup!r}")
    for column in ("row_id", "parent_record_id", "rna_sequence", "fold_random", "fold_basis"):
        if frame[column].map(masking.is_missing).any():
            raise ValueError(f"{column} is null on at least one row (fold/provenance gate)")
    bad_basis = set(frame["fold_basis"]) - set(FOLD_BASES)
    if bad_basis:
        raise ValueError(f"unknown fold_basis value(s): {sorted(bad_basis)}")
    bad_fold = set(frame["fold_random"]) - set(FOLD_RANDOM_VALUES)
    if bad_fold:
        raise ValueError(f"unknown fold_random value(s): {sorted(bad_fold)}")
    bad_status = set(frame["pairing_status"]) - set(PAIRING_STATUSES)
    if bad_status:
        raise ValueError(f"unknown pairing_status value(s): {sorted(bad_status)}")
    over = frame[frame["n_tokens"] > rna_tokenizer.MAX_POSITION_EMBEDDINGS]
    if len(over):
        raise ValueError(
            f"{len(over)} row(s) exceed RiNALMo's {rna_tokenizer.MAX_POSITION_EMBEDDINGS}-token "
            f"context (first: {over['row_id'].iloc[0]})"
        )
    if not (
        frame["n_tokens"] == frame["seq_length"] + rna_tokenizer.N_FLANKING_SPECIAL_TOKENS
    ).all():
        raise ValueError("n_tokens does not equal seq_length + 2 on every row")
    if frame["rna_sequence"].str.contains("T").any():
        raise ValueError("rna_sequence still contains T: the PRD §6 T→U transcription did not run")
    lengths = frame["rna_sequence"].str.len()
    if not (lengths == frame["seq_length"]).all():
        raise ValueError("seq_length disagrees with rna_sequence length")
    anchored = frame[frame["pairing_status"] == PAIRING_ANCHORED]
    if (
        len(anchored)
        and not (anchored["pairing_dotbracket"].str.len() == anchored["seq_length"]).all()
    ):
        raise ValueError("an anchored pairing target is not co-extensive with its sequence")
    unanchored = frame[frame["pairing_status"] != PAIRING_ANCHORED]
    if len(unanchored) and not unanchored["pairing_dotbracket"].map(masking.is_missing).all():
        raise ValueError("a non-anchored row carries a pairing target")


def dataset_digest(frame: Any) -> str:
    """Whole-artifact digest over the row content, independent of row order and parquet bytes.

    Mirrors :func:`ingest.records_digest`: per-row :func:`ingest.record_hash` over the
    emitted columns, sorted by ``row_id``, then hashed. This is the committed golden
    value, so it changes if and only if the assembly contract changes.
    """
    per = [
        ingest.record_hash([row[c] for c in OUTPUT_COLUMNS])
        for row in frame.sort_values("row_id").to_dict("records")
    ]
    return ingest.records_digest(per)


# ------------------------------------------------------------------------------- CLI


def run_stage2_dataset(
    *,
    corpus_parquet: str | Path = DEFAULT_CORPUS,
    labels_parquet: str | Path = DEFAULT_LABELS,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    decoys_parquet: str | Path = DEFAULT_DECOYS,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
    env_lock: str | Path | None = None,
    flank_nt: int = DEFAULT_FLANK_NT,
    seed: int = provenance.DEFAULT_SEED,
) -> int:
    """Build the artifact, its provenance sidecar and its audit report. Returns row count."""
    import pandas as pd

    corpus = pd.read_parquet(corpus_parquet)
    labels = pd.read_parquet(labels_parquet)
    splits = pd.read_parquet(split_table)
    decoys = pd.read_parquet(decoys_parquet)

    frame, report = build_dataset(
        corpus=corpus,
        labels=labels,
        splits=splits,
        decoys=decoys,
        flank_nt=flank_nt,
    )

    out_parquet = Path(out_dir) / OUT_PARQUET_NAME
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_parquet, index=False)

    report_path = Path(audit_dir) / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    provenance.write_provenance(
        Path(out_dir) / OUT_PROVENANCE_NAME,
        rule="workflow/rules/stage2.smk :: stage2_dataset",
        script="src/tbox_finder/stage2/dataset.py",
        seed=seed,
        inputs=[corpus_parquet, labels_parquet, split_table, decoys_parquet],
        outputs=[out_parquet, report_path],
        env_lock=env_lock,
        adr="ADR-0004",
        extra={
            "step": STEP,
            "schema_version": SCHEMA_VERSION,
            "flank_nt": int(flank_nt),
            "decoy_fold_seed": DECOY_FOLD_SEED,
            "tokenizer": report["tokenizer"],
            "digest": report["digest"],
            "n_rows": report["n_rows"],
        },
    )
    return int(len(frame))


def _run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="P3-01 Stage-2 supervised dataset")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--split-table", default=str(DEFAULT_SPLIT_TABLE))
    parser.add_argument("--decoys", default=str(DEFAULT_DECOYS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--env-lock", default=None)
    parser.add_argument("--flank-nt", type=int, default=DEFAULT_FLANK_NT)
    parser.add_argument("--seed", type=int, default=provenance.DEFAULT_SEED)
    args = parser.parse_args(list(argv))

    n = run_stage2_dataset(
        corpus_parquet=args.corpus,
        labels_parquet=args.labels,
        split_table=args.split_table,
        decoys_parquet=args.decoys,
        out_dir=args.out_dir,
        audit_dir=args.audit_dir,
        env_lock=args.env_lock,
        flank_nt=args.flank_nt,
        seed=args.seed,
    )
    print(f"stage2_dataset: {n} rows")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return _run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
