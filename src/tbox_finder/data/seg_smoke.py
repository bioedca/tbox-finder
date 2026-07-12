"""P1-06 — held-out prokaryotic per-nucleotide 8-class segmentation smoke set.

Builds a small (~a few hundred loci) per-nucleotide 8-class labelled slice on
**held-out (leave-clade-out) prokaryotic** T-box loci — the GATE-4 task in
miniature, used to (a) exercise the Stage-1 segmenter end-to-end before any
training run (P1-07 fine-tune, P1-09 continued-pretraining) and (b) provide a
golden-hashed regression fixture. It is a *smoke* gate, not a training set.

Design (why it is not a re-derivation of biology):

* **Selection** reuses the P1-05 held-out-positive predicate verbatim —
  ``(nested_role == "heldout") & (source == "corpus")`` on the committed split
  table (PRD §9.2; ADR-0004 D2). Every emitted locus is therefore 100%
  held-out-fold (the §8.2 leakage unit). "prokaryotic" needs no extra predicate:
  the corpus is all-bacterial and no domain column exists (P1-05 precedent).
* **Labels** are the P0 per-nucleotide labels — the committed ``label_string``
  from ``labels_v0.parquet`` — with the P0 derivation function
  (:func:`tbox_finder.labels.derive_label_codes`) re-run on the same coordinate
  columns *as a consistency guard* (fail-loud on any drift, CLAUDE.md §10.3).
  No new precedence/label logic is introduced here.
* **8-class vocabulary + overlap precedence** are ADR-0004 D1, single-sourced
  from :mod:`tbox_finder.labels` (``CLASS_ORDER``/``CLASS_CODE``/``ELEMENT_COORDS``).
* **Padding / ignore-index** is the ADR-0005 segmentation convention
  (``-100``, matching ``seg_head`` cross-entropy masking).

Outputs (imp.md P1-06):
    data/interim/p1_seg_smoke/windows.parquet   one row per locus (sorted by record_id)
    data/interim/p1_seg_smoke/labels.npy        (n_loci, max_len) int16, pad = -100
    data/processed/audits/seg_smoke_report.json coverage + leakage audit (git-tracked)
    data/interim/p1_seg_smoke/windows.provenance.json
    tests/fixtures/p1_seg_smoke/heldout_slice.csv  (optional --emit-fixture) the
        golden CI slice — a real held-out slice with coords, CI-reproducible
        without the DVC-tracked master/labels tables.

The heavy imports (pandas/numpy) are lazy; the pure selection/label/digest/
coverage helpers import bare so the leakage + golden tests run in bare CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tbox_finder import ingest, labels, provenance
from tbox_finder.labels import (
    CLASS_CODE,
    CLASS_ORDER,
    CODE_TO_CLASS,
    ELEMENT_COORDS,
    NAME_COL,
    WINDOW_LEN_COL,
    WINDOW_SEQ_COL,
    derive_label_codes,
    element_extent,
    label_string_to_indices,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

# --------------------------------------------------------------------------- #
# Selection contract (reused verbatim from P1-05 frozen_linear_probe.py)
# --------------------------------------------------------------------------- #
#: split-table columns + the held-out-positive predicate values (ADR-0004 D2).
RECORD_ID_COL = "record_id"
SOURCE_COL = "source"
NESTED_ROLE_COL = "nested_role"
KLASS_COL = "klass"
CLUSTER_COL = "cluster_id"
ORDER_COL = "resolved_order"
HELDOUT_ROLE = "heldout"
POSITIVE_SOURCE = "corpus"

#: labels_v0.parquet join key (== split ``record_id``) + reused columns.
LABELS_ID_COL = "record_sha256"
LABEL_STRING_COL = "label_string"
RECORD_INDEX_COL = "record_index"
#: labels_v0's ``name`` (== master ``Name`` at ``record_index``) — used only to
#: verify the positional row-order contract, then dropped before output.
LABELS_NAME_COL = "name"
_LABELS_NAME_TMP = "_labels_name"

#: master taxonomy passthrough (informational; no domain column exists).
PHYLUM_COL = "phylum"
MASTER_ORDER_COL = "order"
CLASS_TYPE_COL = ingest.CLASS_TYPE_COL  # "type" — class-II detection for labels

#: per-element coordinate columns, flattened (Stem_I..Discriminator); carried so
#: the golden fixture can re-derive labels without the DVC-tracked labels table.
COORD_COLS: tuple[str, ...] = tuple(col for pair in ELEMENT_COORDS.values() for col in pair)

#: ADR-0005 segmentation masking / ignore-index for padded positions.
IGNORE_INDEX = -100

#: default artifact locations.
DEFAULT_SPLIT_TABLE = Path("data/processed/splits/split_assignments.parquet")
DEFAULT_MASTER = Path("data/processed/master_clean_v0.parquet")
DEFAULT_LABELS = Path("data/processed/labels/labels_v0.parquet")
DEFAULT_OUT_DIR = Path("data/interim/p1_seg_smoke")
DEFAULT_AUDIT_DIR = Path("data/processed/audits")
DEFAULT_REPORT = DEFAULT_AUDIT_DIR / "seg_smoke_report.json"
WINDOWS_NAME = "windows.parquet"
LABELS_NPY_NAME = "labels.npy"
PROVENANCE_NAME = "windows.provenance.json"

#: sizing — small by design (a smoke gate, not training).
DEFAULT_MAX_LOCI = 300
DEFAULT_MIN_PER_CLASS = 8
DEFAULT_MIN_CLASS_II = 20
DEFAULT_SEED = 42

#: columns written to windows.parquet (lean — coords live in master).
WINDOW_COLS: tuple[str, ...] = (
    RECORD_ID_COL,
    NAME_COL,
    KLASS_COL,
    SOURCE_COL,
    NESTED_ROLE_COL,
    CLUSTER_COL,
    ORDER_COL,
    PHYLUM_COL,
    MASTER_ORDER_COL,
    CLASS_TYPE_COL,
    WINDOW_LEN_COL,
    WINDOW_SEQ_COL,
    LABEL_STRING_COL,
)
#: columns written to the golden fixture CSV (adds coords so labels re-derive).
FIXTURE_COLS: tuple[str, ...] = WINDOW_COLS + COORD_COLS

#: fixture columns forced to ``str`` on reload — the golden digest hashes these
#: (kind-tagged), so pandas must not coerce a numeric-looking id/label to int and
#: change its token. Coords + counts stay numeric (NaN ⇒ "element absent").
FIXTURE_STR_COLS: tuple[str, ...] = (
    RECORD_ID_COL,
    KLASS_COL,
    SOURCE_COL,
    NESTED_ROLE_COL,
    CLASS_TYPE_COL,
    WINDOW_SEQ_COL,
    LABEL_STRING_COL,
)


# --------------------------------------------------------------------------- #
# Pure logic (stdlib only — bare-importable, unit-/golden-tested in bare CI)
# --------------------------------------------------------------------------- #
def is_heldout_corpus(source: Any, nested_role: Any) -> bool:
    """The P1-05 held-out-positive predicate (ADR-0004 D2; §8.2 leakage unit)."""
    return nested_role == HELDOUT_ROLE and source == POSITIVE_SOURCE


def present_classes(label_string: str) -> set[str]:
    """The set of 8-class identifiers that appear in a ``label_string``.

    Unknown codes are skipped (not raised) so this and :func:`coverage_report`
    never crash on a malformed label — :func:`validate_smoke` reports the invalid
    code as a problem instead (fail-closed, not fail-hard).
    """
    return {CODE_TO_CLASS[ch] for ch in set(label_string) if ch in CODE_TO_CLASS}


def _has_element(row: Mapping[str, Any], element: str) -> bool:
    """True iff ``element`` has a valid annotated extent in ``row`` (coord-driven)."""
    return element_extent(row, element) is not None


def select_smoke_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    max_loci: int = DEFAULT_MAX_LOCI,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    min_class_ii: int = DEFAULT_MIN_CLASS_II,
) -> list[dict[str, Any]]:
    """Deterministically pick a small, class-covering subset of held-out loci.

    Order is fully determined by ``record_id`` (a sha256 hex → PYTHONHASHSEED- and
    row-order-independent, the P0-30 reproducibility contract). Passes, in order:

    1. **coverage (rarest element class first)** — add record_id-sorted records
       carrying each element class until ``min_per_class`` records carry it (rarest
       classes get their slots before the ``max_loci`` budget is spent);
    2. **class-II coverage** — ensure ≥ ``min_class_ii`` Translational (class-II)
       records so the "Terminator is class-I-only" invariant is non-vacuously tested;
    3. **fill** — add the record_id-sorted remainder up to ``max_loci``.

    Returns the chosen records sorted by ``record_id``.
    """
    ordered = sorted(records, key=lambda r: r[RECORD_ID_COL])
    chosen: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {c: 0 for c in CLASS_ORDER[1:]}

    def _room() -> bool:
        return max_loci is None or len(chosen) < max_loci

    def _add(r: Mapping[str, Any]) -> None:
        rid = r[RECORD_ID_COL]
        if rid in chosen or not _room():
            return
        rec = dict(r)
        chosen[rid] = rec
        present = present_classes(rec[LABEL_STRING_COL])
        for c in CLASS_ORDER[1:]:
            if c in present:
                counts[c] += 1

    # frequency of each element class across the pool → rarest-first coverage.
    freq = {
        c: sum(1 for r in ordered if CLASS_CODE[c] in r[LABEL_STRING_COL]) for c in CLASS_ORDER[1:]
    }
    for element in sorted(CLASS_ORDER[1:], key=lambda c: (freq[c], c)):
        code = CLASS_CODE[element]
        for r in ordered:
            if counts[element] >= min_per_class or not _room():
                break
            if code in r[LABEL_STRING_COL]:
                _add(r)

    n_ii = sum(1 for r in chosen.values() if r.get(CLASS_TYPE_COL) == labels.CLASS_II_TYPE)
    for r in ordered:
        if n_ii >= min_class_ii or not _room():
            break
        if r.get(CLASS_TYPE_COL) == labels.CLASS_II_TYPE and r[RECORD_ID_COL] not in chosen:
            _add(r)
            n_ii += 1

    for r in ordered:
        if not _room():
            break
        _add(r)

    return [chosen[rid] for rid in sorted(chosen)]


def coverage_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-nt / per-record 8-class coverage + ADR-0004 overlap-convention audit.

    Reports (and, via :func:`validate_smoke`, gates) the ADR-0004 invariants:

    * every one of the 8 classes appears (``all_eight_classes_present``);
    * **Terminator is class-I only** — 0 Translational records carry ``T``;
    * **Specifier carves out of Stem_I** — every record annotating *both* Stem_I and
      Specifier shows an ``S`` (Specifier survived precedence over Stem_I);
    * **antiterminator ∩ terminator → Antiterminator** — no position inside an
      annotated antiterminator extent is labelled ``T`` (the on-conformation wins).
    """
    per_class_nt = {c: 0 for c in CLASS_ORDER}
    per_class_records = {c: 0 for c in CLASS_ORDER}
    terminator_on_class_ii = 0
    specifier_carveout_violations = 0
    antiterm_term_violations = 0
    term_code = CLASS_CODE["Terminator"]
    spec_code = CLASS_CODE["Specifier"]

    for r in records:
        ls = r[LABEL_STRING_COL]
        for c in present_classes(ls):
            per_class_records[c] += 1
        for ch in ls:
            cls = CODE_TO_CLASS.get(ch)
            if cls is not None:
                per_class_nt[cls] += 1

        if r.get(CLASS_TYPE_COL) == labels.CLASS_II_TYPE and term_code in ls:
            terminator_on_class_ii += 1

        if _has_element(r, "Stem_I") and _has_element(r, "Specifier") and spec_code not in ls:
            specifier_carveout_violations += 1

        at = element_extent(r, "Antiterminator_Tbox_seq")
        if at is not None:
            lo, hi = at
            hi = min(hi, len(ls))
            if any(ls[pos - 1] == term_code for pos in range(max(1, lo), hi + 1)):
                antiterm_term_violations += 1

    all_eight = all(per_class_records[c] > 0 for c in CLASS_ORDER)
    return {
        "n_records": len(records),
        "total_nt": sum(per_class_nt.values()),
        "per_class_nt": per_class_nt,
        "per_class_records": per_class_records,
        "n_class_ii": sum(1 for r in records if r.get(CLASS_TYPE_COL) == labels.CLASS_II_TYPE),
        "all_eight_classes_present": all_eight,
        "terminator_on_class_ii": terminator_on_class_ii,
        "specifier_carveout_violations": specifier_carveout_violations,
        "antiterm_term_violations": antiterm_term_violations,
    }


def validate_smoke(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Fail-closed validator — returns a list of problems (empty ⇒ valid).

    Enforces the P1-06 validation gate: 100% held-out-fold (leakage guard),
    label/sequence/window length agreement, a valid 8-class code alphabet, full
    8-class coverage, and the ADR-0004 overlap conventions.
    """
    problems: list[str] = []
    if not records:
        return ["empty smoke set"]

    for r in records:
        rid = r.get(RECORD_ID_COL)
        if not is_heldout_corpus(r.get(SOURCE_COL), r.get(NESTED_ROLE_COL)):
            problems.append(
                f"{rid}: not held-out corpus "
                f"(source={r.get(SOURCE_COL)!r}, nested_role={r.get(NESTED_ROLE_COL)!r})"
            )
        ls = r.get(LABEL_STRING_COL)
        seq = r.get(WINDOW_SEQ_COL)
        if not isinstance(ls, str) or not isinstance(seq, str):
            problems.append(f"{rid}: missing label_string/sequence")
            continue
        if len(ls) != len(seq):
            problems.append(f"{rid}: label/seq length {len(ls)} != {len(seq)}")
        wl = r.get(WINDOW_LEN_COL)
        if wl is not None and not _is_absent(wl) and len(ls) != int(round(float(wl))):
            problems.append(f"{rid}: label length {len(ls)} != {WINDOW_LEN_COL} {wl}")
        bad = {ch for ch in set(ls) if ch not in CODE_TO_CLASS}
        if bad:
            problems.append(f"{rid}: invalid label code(s) {sorted(bad)}")

    rep = coverage_report(records)
    if not rep["all_eight_classes_present"]:
        missing = [c for c in CLASS_ORDER if rep["per_class_records"][c] == 0]
        problems.append(f"missing 8-class coverage: {missing}")
    if rep["terminator_on_class_ii"]:
        problems.append(
            f"Terminator painted on {rep['terminator_on_class_ii']} class-II "
            "record(s) (ADR-0004 D1: class-I only)"
        )
    if rep["specifier_carveout_violations"]:
        problems.append(
            f"{rep['specifier_carveout_violations']} record(s) annotate Stem_I+Specifier "
            "but show no Specifier (ADR-0004 D1 carve-out violated)"
        )
    if rep["antiterm_term_violations"]:
        problems.append(
            f"{rep['antiterm_term_violations']} record(s) label a Terminator inside the "
            "antiterminator extent (ADR-0004 D1 on-conformation convention violated)"
        )
    return problems


def _is_absent(value: Any) -> bool:
    """Absent iff None/NaN/non-numeric (reuses the labels.py sentinel semantics)."""
    return labels._is_absent(value)  # noqa: SLF001 - single-sourced sentinel


def assert_labels_consistent(records: Sequence[Mapping[str, Any]]) -> None:
    """Fail loud if a reused ``label_string`` disagrees with the P0 derivation.

    The committed labels *are* the output of :func:`labels.derive_label_codes`; re-running
    that function on the same coordinate columns must reproduce the reused string.
    A mismatch means the corpus/labels drifted — never silently emit it (§10.3).
    """
    for r in records:
        wl_raw = r.get(WINDOW_LEN_COL)
        if _is_absent(wl_raw):
            raise ValueError(
                f"{r.get(RECORD_ID_COL)}: missing/invalid {WINDOW_LEN_COL} — cannot verify "
                f"label derivation (CLAUDE.md §10.3)"
            )
        wl = int(round(float(wl_raw)))
        rederived = derive_label_codes(r, window_length=wl, naive=False)
        if rederived != r[LABEL_STRING_COL]:
            raise ValueError(
                f"{r.get(RECORD_ID_COL)}: reused label_string disagrees with the P0 "
                f"derive_label_codes output — corpus/labels drift (CLAUDE.md §10.3)"
            )


def smoke_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Row-order-independent golden digest over ``(record_id, sequence, label_string)``.

    Reuses the ingest hashing contract (kind-tagged, length-framed) and sorts by
    ``record_id`` first, so the digest depends only on the selected loci and their
    P0 labels — not on parquet/pyarrow bytes or row order — and reproduces in any env.
    """
    ordered = sorted(records, key=lambda r: r[RECORD_ID_COL])
    per_record = [
        ingest.record_hash([r[RECORD_ID_COL], r[WINDOW_SEQ_COL], r[LABEL_STRING_COL]])
        for r in ordered
    ]
    return ingest.records_digest(per_record)


def leakage_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The §8.2-style leakage audit dict for the smoke set (100% held-out check)."""
    n = len(records)
    n_heldout = sum(
        1 for r in records if is_heldout_corpus(r.get(SOURCE_COL), r.get(NESTED_ROLE_COL))
    )
    clusters = {r.get(CLUSTER_COL) for r in records}
    orders = {r.get(ORDER_COL) for r in records}
    return {
        "n_records": n,
        "n_heldout_corpus": n_heldout,
        "positives_heldout_only": n > 0 and n_heldout == n,
        "n_clusters": len(clusters),
        "n_resolved_orders": len(orders),
    }


# --------------------------------------------------------------------------- #
# IO / orchestration (lazy pandas + numpy — heavy path, run LOCAL)
# --------------------------------------------------------------------------- #
def labels_to_matrix(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Pad per-nt label vectors into an ``(n_loci, max_len)`` int16 matrix.

    Rows are ordered by ``record_id`` (== windows.parquet order); padded positions
    carry :data:`IGNORE_INDEX` (the ADR-0005 / seg-head cross-entropy ignore value).
    Written with ``allow_pickle=False`` — a plain integer matrix, no pickle surface.
    """
    import numpy as np

    ordered = sorted(records, key=lambda r: r[RECORD_ID_COL])
    max_len = max((len(r[LABEL_STRING_COL]) for r in ordered), default=0)
    mat = np.full((len(ordered), max_len), IGNORE_INDEX, dtype=np.int16)
    for i, r in enumerate(ordered):
        idx = label_string_to_indices(r[LABEL_STRING_COL])
        mat[i, : len(idx)] = np.asarray(idx, dtype=np.int16)
    return mat


def build_smoke_records(
    split_df: pd.DataFrame,
    master_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    max_loci: int = DEFAULT_MAX_LOCI,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    min_class_ii: int = DEFAULT_MIN_CLASS_II,
) -> list[dict[str, Any]]:
    """Join split + labels + master and select the held-out smoke loci.

    * held-out positives from the split table (the P1-05 predicate),
    * reused P0 ``label_string`` (+ ``record_index``) from labels_v0,
    * sequence + coords + taxonomy from master (indexed by ``record_index``).
    """
    import pandas as pd

    mask = (split_df[NESTED_ROLE_COL] == HELDOUT_ROLE) & (split_df[SOURCE_COL] == POSITIVE_SOURCE)
    sel = split_df.loc[
        mask, [RECORD_ID_COL, SOURCE_COL, NESTED_ROLE_COL, KLASS_COL, CLUSTER_COL, ORDER_COL]
    ].copy()

    lab = labels_df[[LABELS_ID_COL, RECORD_INDEX_COL, LABELS_NAME_COL, LABEL_STRING_COL]].rename(
        columns={LABELS_ID_COL: RECORD_ID_COL, LABELS_NAME_COL: _LABELS_NAME_TMP}
    )
    sel = sel.merge(lab, on=RECORD_ID_COL, how="inner").reset_index(drop=True)

    master_cols = [
        NAME_COL,
        WINDOW_SEQ_COL,
        WINDOW_LEN_COL,
        CLASS_TYPE_COL,
        PHYLUM_COL,
        MASTER_ORDER_COL,
        *COORD_COLS,
    ]
    # ``record_index`` is a *positional* index into master (labels_v0 was built row-for-row
    # from master, so labels_v0.name == master.Name[record_index]). Validate that contract
    # explicitly before trusting ``iloc`` — a re-sorted/re-filtered master would silently
    # mis-join otherwise (CLAUDE.md §10.3, fail loud).
    idx = sel[RECORD_INDEX_COL].to_numpy()
    n_master = len(master_df)
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_master):
        raise ValueError(
            f"record_index out of range for master ([{int(idx.min())}, {int(idx.max())}] "
            f"vs {n_master} rows) — corpus/labels row-order drift (CLAUDE.md §10.3)"
        )
    msub = master_df.iloc[idx][master_cols].reset_index(drop=True)
    name_mismatch = msub[NAME_COL].to_numpy() != sel[_LABELS_NAME_TMP].to_numpy()
    if bool(name_mismatch.any()):
        bad = sel.loc[name_mismatch, RECORD_ID_COL].tolist()[:5]
        raise ValueError(
            f"record_index → master row-order contract violated for "
            f"{int(name_mismatch.sum())} record(s) (e.g. {bad}) — corpus/labels drift "
            f"(CLAUDE.md §10.3)"
        )
    joined = pd.concat([sel.drop(columns=[_LABELS_NAME_TMP]), msub], axis=1)

    records = joined.to_dict(orient="records")
    subset = select_smoke_subset(
        records,
        max_loci=max_loci,
        min_per_class=min_per_class,
        min_class_ii=min_class_ii,
    )
    assert_labels_consistent(subset)
    return subset


def emit_fixture_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the golden CI slice (real held-out records + coords) to CSV.

    CSV (not parquet) so the golden test reads it with pandas alone; coords are
    carried so the label re-derivation runs without the DVC-tracked labels table.
    """
    import pandas as pd

    ordered = sorted(records, key=lambda r: r[RECORD_ID_COL])
    df = pd.DataFrame(ordered)[list(FIXTURE_COLS)]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def load_fixture_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load the golden fixture slice as record dicts (the golden-test entry point).

    Forces the digest-/derivation-critical columns to ``str`` so pandas dtype
    inference cannot alter a hashed token; coordinate columns stay numeric (an
    empty cell ⇒ NaN ⇒ "element absent", per :func:`labels.element_extent`).
    """
    import pandas as pd

    df = pd.read_csv(path, dtype={c: str for c in FIXTURE_STR_COLS})
    return df.to_dict(orient="records")


def build_smoke_set(
    *,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    master: str | Path = DEFAULT_MASTER,
    labels_path: str | Path = DEFAULT_LABELS,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    report_path: str | Path = DEFAULT_REPORT,
    max_loci: int = DEFAULT_MAX_LOCI,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    min_class_ii: int = DEFAULT_MIN_CLASS_II,
    seed: int = DEFAULT_SEED,
    env_lock: str | Path | None = None,
    emit_fixture: str | Path | None = None,
) -> dict[str, Any]:
    """Build the P1-06 smoke set: windows.parquet + labels.npy + audit + provenance.

    Fails loud (never writes) if the validation gate does not pass (§10.3).
    Returns a summary dict (digest, counts, output paths).
    """
    import numpy as np
    import pandas as pd

    split_df = pd.read_parquet(split_table)
    master_df = pd.read_parquet(master)
    labels_df = pd.read_parquet(labels_path)

    records = build_smoke_records(
        split_df,
        master_df,
        labels_df,
        max_loci=max_loci,
        min_per_class=min_per_class,
        min_class_ii=min_class_ii,
    )
    problems = validate_smoke(records)
    if problems:
        raise ValueError(
            "P1-06 smoke-set validation failed (nothing written):\n  - " + "\n  - ".join(problems)
        )

    ordered = sorted(records, key=lambda r: r[RECORD_ID_COL])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_path = out_dir / WINDOWS_NAME
    labels_npy_path = out_dir / LABELS_NPY_NAME
    prov_path = out_dir / PROVENANCE_NAME

    windows_df = pd.DataFrame(ordered)[list(WINDOW_COLS)]
    windows_df.to_parquet(windows_path, index=False)

    mat = labels_to_matrix(ordered)
    np.save(labels_npy_path, mat, allow_pickle=False)

    digest = smoke_digest(ordered)
    cov = coverage_report(ordered)
    leak = leakage_report(ordered)
    report = {
        "n_loci": len(ordered),
        "digest": digest,
        "labels_shape": list(mat.shape),
        "ignore_index": IGNORE_INDEX,
        "coverage": cov,
        "leakage": leak,
        "selection": {
            "max_loci": max_loci,
            "min_per_class": min_per_class,
            "min_class_ii": min_class_ii,
            "seed": seed,
        },
        "adr": "ADR-0004/ADR-0005",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    fixture_out: Path | None = None
    if emit_fixture is not None:
        fixture_out = emit_fixture_csv(ordered, emit_fixture)

    provenance.write_provenance(
        prov_path,
        rule="workflow/rules/backbones.smk :: p1_seg_smoke_set",
        script="src/tbox_finder/data/seg_smoke.py",
        seed=seed,
        inputs=[split_table, master, labels_path],
        outputs=[windows_path, labels_npy_path],
        env_lock=env_lock,
        adr="ADR-0004",
        extra={"digest": digest, "n_loci": len(ordered), "ignore_index": IGNORE_INDEX},
    )

    return {
        "digest": digest,
        "n_loci": len(ordered),
        "windows": str(windows_path),
        "labels_npy": str(labels_npy_path),
        "report": str(report_path),
        "provenance": str(prov_path),
        "fixture": str(fixture_out) if fixture_out is not None else None,
        "coverage": cov,
        "leakage": leak,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-table", default=str(DEFAULT_SPLIT_TABLE))
    p.add_argument("--master", default=str(DEFAULT_MASTER))
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument("--max-loci", type=int, default=DEFAULT_MAX_LOCI)
    p.add_argument("--min-per-class", type=int, default=DEFAULT_MIN_PER_CLASS)
    p.add_argument("--min-class-ii", type=int, default=DEFAULT_MIN_CLASS_II)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--env-lock", default=None)
    p.add_argument(
        "--emit-fixture",
        default=None,
        help="also write the golden CI slice (CSV) to this path",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    summary = build_smoke_set(
        split_table=args.split_table,
        master=args.master,
        labels_path=args.labels,
        out_dir=args.out_dir,
        report_path=args.report,
        max_loci=args.max_loci,
        min_per_class=args.min_per_class,
        min_class_ii=args.min_class_ii,
        seed=args.seed,
        env_lock=args.env_lock if args.env_lock else None,
        emit_fixture=args.emit_fixture,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
