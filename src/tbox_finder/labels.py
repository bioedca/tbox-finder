"""labels.py — 8-class per-nucleotide segmentation-label derivation (P0-20).

Maps each TBDB record's element annotations onto its per-record local coordinate
window to produce a **single-label** per-nucleotide vector over the 8 classes
(PRD §8; the softmax target for the Stage-1 segmenter, PRD §11). Where two element
extents overlap, the base is resolved by the **total precedence order pinned in
ADR-0004 D1** (most-specific / on-conformation wins):

    Discriminator ▸ Specifier ▸ Antiterminator_Tbox_seq ▸ Terminator ▸
    Stem_II ▸ Stem_III ▸ Stem_I ▸ background

A base takes the class of the highest-precedence element whose extent covers it;
``background`` only where no element covers it. The three material overlaps this
resolves (ADR-0004 D1, measured on the 23,535-record corpus) rest on established
T-box biology, verified against ≥2 independent peer-reviewed sources (CLAUDE.md
§10.1):

  1. **Specifier carves out of Stem_I** — the specifier codon sits in the Stem I
     / Specifier-Loop domain and base-pairs with the tRNA anticodon
     [Grigg & Ke 2013, DOI:10.4161/rna.26996; Wang & Nikonowicz 2011,
     DOI:10.1016/j.jmb.2011.02.014; TBDB, PMID:32882008 / DOI:10.1093/nar/gkaa721].
  2. **Antiterminator∩Terminator → Antiterminator_Tbox_seq** — the antiterminator
     and terminator are *mutually exclusive conformations* of the same RNA (the
     dot-bracket encodes only one), so the overlap is assigned the T-box-defining
     read-through conformation [Grigg & Ke 2013, DOI:10.4161/rna.26996;
     Sherwood et al. 2015, PMID:25583497 / DOI:10.1073/pnas.1424175112].
  3. **Discriminator carves out of Antiterminator_Tbox_seq** — the discriminator
     base(s) read the tRNA NCCA acceptor end inside the antiterminator bulge
     [Grigg & Ke 2013, DOI:10.4161/rna.26996; PMID:25583497; PMID:32882008].

**Terminator is class-I only** — class II (Translational) has no terminator hairpin
[Sherwood et al. 2015, PMID:25583497]; the derivation never paints a Terminator on a
Translational record, even where the corpus carries a spurious ``term_*`` annotation
(PRD §8: "the label scheme and model must not require a terminator").

**Pseudoknot (IIA/B)** is folded into ``Stem_II`` by class definition, not by
precedence — no standalone pseudoknot class is trained (its crossing pairs cannot be
encoded in the nested dot-bracket that sources the labels); it is retained only as a
PDB-fixture structural diagnostic (PRD §8; ADR-0004 D1).

**Class-II-CM-naive label source (PRD §8; ADR-0002; §5 mechanism 3).** The GATE-1
anti-mimicry ablation is a separate Stage-1 checkpoint whose labels are *never*
derived from the class-II CM (``TBDB001.cm``). This module exposes a per-record
``label_source`` flag (``class_I_cm`` / ``class_II_cm``) keyed on the class field —
Translational records are ``TBDB001.cm``-derived (ADR-0004 D1). In the **naive**
derivation those records withhold *all* ``TBDB001.cm``-derived structure (their label
vector is all ``background``), so no class-II-CM structure enters the naive target and
any class-II detection by the naive checkpoint is generalization, not CM mimicry. The
corpus records no per-element CM provenance, so withholding the whole class-II record
is the conservative, non-fabricating realization (CLAUDE.md §10.3); P2 may re-introduce
class-I-CM (RF00230)-derived shared-element labels once a class-I re-annotation exists.
Class-I records are identical between the production and naive derivations.

**Coordinate frame.** Element coordinates are **1-based, inclusive-nt** in a
per-record local frame whose length is ``tbox_length`` (verified: ``len(FASTA_sequence)
== tbox_length`` and every element coord ``≤ tbox_length`` on the corpus). An element
is **absent** when either endpoint is missing — NaN, or ``< 1`` (the ``0`` / ``-1`` /
``-2`` sentinels the ingest contract leaves; P0-12): both endpoints must be present to
paint. Minus-strand records with ``start > end`` are normalized to ``(min, max)``,
matching ``scripts/measure_overlap_prevalence.py`` (ADR-0004).

Stdlib-only at import time (pandas is imported lazily inside the heavy functions), so
the pure-logic derivation primitives run in the bare CI test env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tbox_finder import ingest, provenance
from tbox_finder.provenance import DEFAULT_SEED

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# --------------------------------------------------------------------------- #
# The 8-class label vocabulary (PRD §8; ADR-0004 D1)
# --------------------------------------------------------------------------- #
#: Softmax class order — the integer index of each class is its position here.
CLASS_ORDER: tuple[str, ...] = (
    "background",
    "Stem_I",
    "Specifier",
    "Stem_II",
    "Stem_III",
    "Antiterminator_Tbox_seq",
    "Terminator",
    "Discriminator",
)
#: class name → softmax index (0..7).
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_ORDER)}

#: Compact single-char code per class, for the human-inspectable / hashable
#: ``label_string`` serialization of the per-nt vector.
CLASS_CODE: dict[str, str] = {
    "background": ".",
    "Stem_I": "1",
    "Specifier": "S",
    "Stem_II": "2",
    "Stem_III": "3",
    "Antiterminator_Tbox_seq": "A",
    "Terminator": "T",
    "Discriminator": "D",
}
#: inverse of :data:`CLASS_CODE`.
CODE_TO_CLASS: dict[str, str] = {v: k for k, v in CLASS_CODE.items()}

#: Total precedence order, **highest → lowest** (ADR-0004 D1). ``background`` is the
#: implicit fallback and is not painted.
PRECEDENCE_HIGH_TO_LOW: tuple[str, ...] = (
    "Discriminator",
    "Specifier",
    "Antiterminator_Tbox_seq",
    "Terminator",
    "Stem_II",
    "Stem_III",
    "Stem_I",
)
#: Paint order, **lowest → highest**: later paints overwrite earlier ones, so the
#: highest-precedence element that covers a base wins.
PAINT_ORDER_LOW_TO_HIGH: tuple[str, ...] = tuple(reversed(PRECEDENCE_HIGH_TO_LOW))

#: element → (start-col, end-col) in ``master_clean_v0.parquet`` (canonical map,
#: matching ``scripts/measure_overlap_prevalence.py``). The ``Specifier`` extent is
#: the annotated codon (``codon_*``); P0-21 may widen it to the specifier loop against
#: the crystal fixtures without changing the more-specific-wins rule (ADR-0004 D1).
ELEMENT_COORDS: dict[str, tuple[str, str]] = {
    "Stem_I": ("s1_start", "s1_end"),
    "Specifier": ("codon_start", "codon_end"),
    "Stem_II": ("stem2_region_start", "stem2_region_end"),
    "Stem_III": ("stem3_start", "stem3_end"),
    "Antiterminator_Tbox_seq": ("antiterm_start", "antiterm_end"),
    "Terminator": ("term_start", "term_end"),
    "Discriminator": ("discrim_start", "discrim_end"),
}

#: The three core elements GATE-4 scores (ADR-0004 D6) — the element-coverage
#: completeness flag is keyed on their joint presence.
CORE_ELEMENTS: tuple[str, ...] = ("Stem_I", "Specifier", "Antiterminator_Tbox_seq")

#: Column holding the per-record window length (== len(FASTA_sequence), verified).
WINDOW_LEN_COL = "tbox_length"
#: Full-window (ungapped) sequence column; its length must equal the window length.
WINDOW_SEQ_COL = "FASTA_sequence"

#: Class-II records (Translational) are ``TBDB001.cm``-derived; class-I are RF00230.
CLASS_II_TYPE = "Translational"
LABEL_SOURCE_CLASS_I = "class_I_cm"
LABEL_SOURCE_CLASS_II = "class_II_cm"

#: Aux-label passthrough columns (PRD §8): specifier codon, cognate aa, tRNA family.
CODON_COL = "codon"
COGNATE_AA_COL = "amino_acid_top"
TRNA_FAMILY_COL = "trna_family_top"
COMPLETENESS_COL = "Completeness"
NAME_COL = "Name"

#: Default paths (mirroring priors.py / ingest.py module constants).
CORPUS_PARQUET = Path("data/processed/master_clean_v0.parquet")
LABELS_DIR = Path("data/processed/labels")
AUDIT_DIR = Path("data/processed/audits")
OUT_PARQUET_NAME = "labels_v0.parquet"
REPORT_NAME = "labels_report.json"


# --------------------------------------------------------------------------- #
# Pure-logic derivation primitives (stdlib only — bare-importable, unit-tested)
# --------------------------------------------------------------------------- #
def _is_absent(value: Any) -> bool:
    """True when a coordinate endpoint means "element not annotated".

    Absent iff ``None``, NaN, non-numeric, or ``< 1`` — 1-based coords start at 1, so
    the ingest sentinels (``0`` → NaN for most cols; raw ``-1`` for ``codon_*`` /
    ``s1_loop_*``; ``-2`` → NaN for ``discrim_end``) all read as absent. Mirrors the
    ``isnan | (a < 0)`` gate in ``scripts/measure_overlap_prevalence.py`` (widened to
    ``< 1`` so a surviving ``0`` sentinel is also caught).
    """
    if value is None:
        return True
    try:
        f = float(value)
    except (TypeError, ValueError):
        return True
    if f != f:  # NaN
        return True
    return f < 1.0


def element_extent(row: dict[str, Any], element: str) -> tuple[int, int] | None:
    """Return the 1-based inclusive ``(start, end)`` of ``element`` or ``None`` if absent.

    Both endpoints must be present. ``start > end`` (minus-strand) is normalized to
    ``(min, max)`` (ADR-0004; overlap script).
    """
    start_col, end_col = ELEMENT_COORDS[element]
    start, end = row.get(start_col), row.get(end_col)
    if _is_absent(start) or _is_absent(end):
        return None
    s_i = int(round(float(start)))
    e_i = int(round(float(end)))
    if s_i > e_i:
        s_i, e_i = e_i, s_i
    return s_i, e_i


def is_class_ii(row: dict[str, Any]) -> bool:
    """True iff the record is a Translational (class-II, ``TBDB001.cm``-derived) T-box."""
    return row.get(ingest.CLASS_TYPE_COL) == CLASS_II_TYPE


def label_source(row: dict[str, Any]) -> str:
    """Which CM the record's structure derives from — the class-II-CM-naive flag key.

    ``class_II_cm`` iff Translational (``TBDB001.cm``-derived, ADR-0004 D1); else
    ``class_I_cm`` (RF00230). A missing/unknown ``type`` defaults to ``class_I_cm``,
    the conservative majority (RF00230-derived) assignment.
    """
    return LABEL_SOURCE_CLASS_II if is_class_ii(row) else LABEL_SOURCE_CLASS_I


def class_of(row: dict[str, Any]) -> str:
    """T-box class ``"I"`` / ``"II"`` from the ``type`` field (``"?"`` if unknown)."""
    return ingest.TYPE_TO_CLASS.get(row.get(ingest.CLASS_TYPE_COL), "?")


def derive_label_codes(row: dict[str, Any], *, window_length: int, naive: bool) -> str:
    """Derive the single-label per-nt code string for one record.

    Returns a string of length ``window_length`` over :data:`CLASS_CODE`. Elements are
    painted **lowest → highest precedence** (ADR-0004 D1) so the highest-precedence
    element covering a base wins; ``background`` (``"."``) fills the rest.

    - **Terminator** is painted only for class-I records (class II has no terminator;
      PMID:25583497).
    - **naive=True**: a ``class_II_cm`` (Translational) record withholds *all*
      ``TBDB001.cm``-derived structure → all ``background``. Class-I records are
      unchanged (identical to the production derivation).
    """
    codes = [CLASS_CODE["background"]] * window_length
    if naive and label_source(row) == LABEL_SOURCE_CLASS_II:
        return "".join(codes)  # class-II CM structure withheld from the naive target
    class_ii = is_class_ii(row)
    for element in PAINT_ORDER_LOW_TO_HIGH:
        if element == "Terminator" and class_ii:
            continue  # class-I-only element (PMID:25583497)
        extent = element_extent(row, element)
        if extent is None:
            continue
        start, end = extent
        code = CLASS_CODE[element]
        lo = max(1, start)
        hi = min(window_length, end)
        for pos in range(lo, hi + 1):
            codes[pos - 1] = code
    return "".join(codes)


def label_string_to_indices(label_string: str) -> list[int]:
    """Expand a ``label_string`` to the list of softmax class indices (0..7)."""
    return [CLASS_INDEX[CODE_TO_CLASS[ch]] for ch in label_string]


def labels_digest(prod_codes: list[str], naive_codes: list[str]) -> str:
    """Whole-artifact golden digest over the per-record (production, naive) label pair.

    Reuses the ingest hashing contract (kind-tagged, length-framed, row-order-
    dependent) so the digest depends only on the derivation logic and the cleaned
    inputs — not on parquet/pyarrow bytes — and is reproducible in any env.
    """
    per_record = [
        ingest.record_hash([prod, naive])
        for prod, naive in zip(prod_codes, naive_codes, strict=True)
    ]
    return ingest.records_digest(per_record)


def _window_length(row: dict[str, Any], index: int) -> int:
    """The per-record window length from ``tbox_length`` (fail-loud on absent/≤0)."""
    raw = row.get(WINDOW_LEN_COL)
    if _is_absent(raw):
        raise ValueError(
            f"record {index} ({row.get(NAME_COL)!r}): {WINDOW_LEN_COL} is absent — cannot "
            f"derive a label window (CLAUDE.md §10.3)"
        )
    return int(round(float(raw)))


def _as_str(value: Any) -> str:
    """Coerce an aux-label cell to a plain string ("" for None / NaN)."""
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value)


# --------------------------------------------------------------------------- #
# Orchestration (lazy pandas — heavy path)
# --------------------------------------------------------------------------- #
def derive_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str]]:
    """Derive per-record labels over the cleaned corpus frame.

    Returns ``(labels_df, report, prod_codes, naive_codes)``. Every per-nt vector spans
    the full window and is single-label by construction; low-completeness records are
    flagged (never dropped). Fails loud (CLAUDE.md §10.3) if any element extent exceeds
    its window or any ``len(FASTA_sequence) != tbox_length``.
    """
    import pandas as pd

    n = len(df)
    needed: list[str] = [
        WINDOW_LEN_COL,
        WINDOW_SEQ_COL,
        ingest.CLASS_TYPE_COL,
        COMPLETENESS_COL,
        CODON_COL,
        COGNATE_AA_COL,
        TRNA_FAMILY_COL,
        NAME_COL,
    ]
    for start_col, end_col in ELEMENT_COORDS.values():
        needed.extend((start_col, end_col))
    # Column-wise extraction (avoids the itertuples `class`-keyword gotcha, P0-14).
    colvals: dict[str, list[Any]] = {
        col: (df[col].tolist() if col in df.columns else [None] * n)
        for col in dict.fromkeys(needed)
    }
    record_hashes = ingest.compute_record_hashes(df)

    rows_out: list[dict[str, Any]] = []
    prod_codes: list[str] = []
    naive_codes: list[str] = []
    seq_mismatch: list[str] = []
    start_beyond_window: list[str] = []
    clamped_examples: list[dict[str, Any]] = []
    n_clamped_records = 0
    class_nt_prod = {name: 0 for name in CLASS_ORDER}
    class_nt_naive = {name: 0 for name in CLASS_ORDER}
    presence_counts = {element: 0 for element in ELEMENT_COORDS}

    for i in range(n):
        row = {col: colvals[col][i] for col in colvals}
        window_length = _window_length(row, i)
        seq = _as_str(row.get(WINDOW_SEQ_COL))
        if len(seq) != window_length:
            seq_mismatch.append(f"{row.get(NAME_COL)!r}(seq={len(seq)},win={window_length})")
        presence = {element: element_extent(row, element) is not None for element in ELEMENT_COORDS}
        record_class_ii = is_class_ii(row)
        clamped_this_record = False
        for element in ELEMENT_COORDS:
            extent = element_extent(row, element)
            if extent is None:
                continue
            presence_counts[element] += 1
            if element == "Terminator" and record_class_ii:
                continue  # not painted for class II → its extent cannot affect the labels
            start, end = extent
            if start > window_length:
                # the whole element is beyond the window — a real annotation anomaly
                start_beyond_window.append(
                    f"{row.get(NAME_COL)!r}({element}:{extent},win={window_length})"
                )
            elif end > window_length:
                # only the 3' tail runs past the window (no window sequence there to
                # label) → the painter clamps to [start, window]; report, don't drop.
                clamped_this_record = True
                if len(clamped_examples) < 20:
                    clamped_examples.append(
                        {
                            "name": _as_str(row.get(NAME_COL)),
                            "element": element,
                            "extent": [start, end],
                            "window_length": window_length,
                        }
                    )
        if clamped_this_record:
            n_clamped_records += 1

        prod = derive_label_codes(row, window_length=window_length, naive=False)
        naive = derive_label_codes(row, window_length=window_length, naive=True)
        if (
            len(prod) != window_length or len(naive) != window_length
        ):  # pragma: no cover - invariant
            raise ValueError(
                f"record {i} ({row.get(NAME_COL)!r}): label vector does not span the window "
                f"(prod={len(prod)}, naive={len(naive)}, window={window_length})"
            )
        for ch in prod:
            class_nt_prod[CODE_TO_CLASS[ch]] += 1
        for ch in naive:
            class_nt_naive[CODE_TO_CLASS[ch]] += 1

        source = label_source(row)
        n_core = sum(1 for e in CORE_ELEMENTS if presence[e])
        low_completeness = n_core < len(CORE_ELEMENTS)
        prod_labeled = window_length - prod.count(CLASS_CODE["background"])
        rows_out.append(
            {
                "record_index": i,
                "record_sha256": record_hashes[i],
                "name": _as_str(row.get(NAME_COL)),
                "window_length": window_length,
                "class_type": class_of(row),
                "label_source": source,
                "class_ii_cm_naive": source == LABEL_SOURCE_CLASS_II,
                "label_string": prod,
                "naive_label_string": naive,
                "specifier_codon": _as_str(row.get(CODON_COL)),
                "cognate_aa": _as_str(row.get(COGNATE_AA_COL)),
                "trna_family": _as_str(row.get(TRNA_FAMILY_COL)),
                "regulatory_mode": _as_str(row.get(ingest.CLASS_TYPE_COL)),
                "has_stem_i": presence["Stem_I"],
                "has_specifier": presence["Specifier"],
                "has_stem_ii": presence["Stem_II"],
                "has_stem_iii": presence["Stem_III"],
                "has_antiterminator": presence["Antiterminator_Tbox_seq"],
                "has_terminator": presence["Terminator"],
                "has_discriminator": presence["Discriminator"],
                "n_core_present": n_core,
                "low_completeness": low_completeness,
                "tbdb_completeness": _as_str(row.get(COMPLETENESS_COL)),
                "n_labeled_nt": prod_labeled,
                "background_nt": prod.count(CLASS_CODE["background"]),
                "coord_clamped": clamped_this_record,
            }
        )
        prod_codes.append(prod)
        naive_codes.append(naive)

    if seq_mismatch:
        raise ValueError(
            f"{len(seq_mismatch)} record(s) with len(FASTA_sequence) != {WINDOW_LEN_COL} "
            f"(label vector would not align to the sequence; CLAUDE.md §10.3): "
            f"{seq_mismatch[:5]}"
        )
    if start_beyond_window:
        raise ValueError(
            f"{len(start_beyond_window)} element(s) start beyond the record window — the "
            f"whole element is out of frame, a real annotation anomaly (CLAUDE.md §10.3): "
            f"{start_beyond_window[:5]}"
        )

    labels_df = pd.DataFrame(rows_out)
    n_class_ii = int(sum(1 for r in rows_out if r["label_source"] == LABEL_SOURCE_CLASS_II))
    n_low = int(sum(1 for r in rows_out if r["low_completeness"]))
    report: dict[str, Any] = {
        "n_records": n,
        "adr": "ADR-0004",
        "prd": "§8, §11",
        "class_order": list(CLASS_ORDER),
        "class_index": CLASS_INDEX,
        "class_code": CLASS_CODE,
        "precedence_high_to_low": list(PRECEDENCE_HIGH_TO_LOW),
        "precedence_biology_pmids": {
            "specifier_in_stem_i": ["24356646", "21333656", "32882008"],
            "antiterminator_terminator_exclusive": ["24356646", "25583497"],
            "discriminator_reads_ncca": ["24356646", "25583497", "32882008"],
            "class_ii_no_terminator": ["25583497"],
        },
        "class_counts": {
            "I": int(sum(1 for r in rows_out if r["class_type"] == "I")),
            "II": int(sum(1 for r in rows_out if r["class_type"] == "II")),
            "unknown": int(sum(1 for r in rows_out if r["class_type"] == "?")),
        },
        "label_source_counts": {
            LABEL_SOURCE_CLASS_I: n - n_class_ii,
            LABEL_SOURCE_CLASS_II: n_class_ii,
        },
        "class_ii_cm_naive_withheld_records": n_class_ii,
        "low_completeness_records": n_low,
        "low_completeness_fraction": round(n_low / n, 6) if n else 0.0,
        "records_dropped": 0,
        "records_coord_clamped_to_window": n_clamped_records,
        "coord_clamped_note": (
            "class-I intrinsic-terminator 3' arms extending past the extracted leader "
            "window (term_start == tbox_length); the out-of-window tail has no window "
            "sequence and is left unlabeled (clamped to [start, tbox_length]). Labels "
            "still span exactly the window; no record is dropped."
        ),
        "coord_clamped_examples": clamped_examples,
        "element_presence_counts": presence_counts,
        "class_nt_totals_production": class_nt_prod,
        "class_nt_totals_naive": class_nt_naive,
    }
    return labels_df, report, prod_codes, naive_codes


def run_labels(
    *,
    corpus_parquet: str | Path = CORPUS_PARQUET,
    labels_dir: str | Path = LABELS_DIR,
    audit_dir: str | Path = AUDIT_DIR,
    env_lock: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    out_parquet_name: str = OUT_PARQUET_NAME,
) -> int:
    """Read the corpus, derive labels, and write the parquet + audit + provenance."""
    import pandas as pd

    corpus_parquet = Path(corpus_parquet)
    labels_dir = Path(labels_dir)
    audit_dir = Path(audit_dir)
    df = pd.read_parquet(corpus_parquet)
    labels_df, report, prod_codes, naive_codes = derive_labels(df)

    labels_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = labels_dir / out_parquet_name
    labels_df.to_parquet(out_parquet, index=False)

    report["labels_digest"] = labels_digest(prod_codes, naive_codes)
    report["corpus_sha256"] = provenance.sha256_file(corpus_parquet)
    report_path = audit_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    stem = (
        out_parquet_name[: -len(".parquet")]
        if out_parquet_name.endswith(".parquet")
        else out_parquet_name
    )
    prov_path = labels_dir / f"{stem}.provenance.json"
    provenance.write_provenance(
        prov_path,
        rule="workflow/rules/data.smk :: derive_labels",
        script="src/tbox_finder/labels.py",
        seed=seed,
        inputs=[corpus_parquet],
        outputs=[out_parquet, report_path],
        env_lock=env_lock,
        adr="ADR-0004",
        extra={
            "class_order": list(CLASS_ORDER),
            "precedence_high_to_low": list(PRECEDENCE_HIGH_TO_LOW),
            "labels_digest": report["labels_digest"],
            "n_records": report["n_records"],
            "class_ii_cm_naive_withheld_records": report["class_ii_cm_naive_withheld_records"],
            "low_completeness_records": report["low_completeness_records"],
        },
    )

    print(
        f"[labels] {len(labels_df)} records → {out_parquet} | "
        f"class I/II={report['class_counts']['I']}/{report['class_counts']['II']} "
        f"low_completeness={report['low_completeness_records']} "
        f"class-II-CM-naive-withheld={report['class_ii_cm_naive_withheld_records']} "
        f"digest={report['labels_digest'][:12]}…"
    )
    return 0


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="tbox_finder.labels",
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_PARQUET)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--env-lock", type=Path, default=Path("envs/data.conda-lock.yml"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-name", default=OUT_PARQUET_NAME)
    args = parser.parse_args(argv)
    return run_labels(
        corpus_parquet=args.corpus,
        labels_dir=args.labels_dir,
        audit_dir=args.audit_dir,
        env_lock=args.env_lock,
        seed=args.seed,
        out_parquet_name=args.out_name,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry (``python -m tbox_finder.labels``)."""
    return _run(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
