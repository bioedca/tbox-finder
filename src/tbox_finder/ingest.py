"""ingest.py — Master_tboxes.csv ingest + count/hash parse-correctness gate.

P0-12; PRD §7.1 (count parse-gate), §16. Reproduces the tboxevo canonical
cleaner (``ingest_master_clean.py``) so a *fresh* ingest of the immutable raw
TBDB export proves — at 100 % per-record identity — that it reconstructs the
canonical cleaned training corpus.

The gate exists because a naive line count double-reads the corpus: the raw
``Master_tboxes.csv`` has embedded newlines inside quoted tRNA 2°-structure
fields, so ``wc -l`` reports 39 459 *lines* for 23 535 *records*. A proper
RFC-4180 parse recovers **23 535 records × 107 raw columns** (the leading,
unnamed pandas-export index column brings the raw count to 107; the cleaned
corpus has 106 named columns).

Reproduction contract — identical to tboxevo's ``ingest_master_clean.py`` so the
per-record hash-identity gate holds (**every change here breaks the golden
hash**):

1. **Drop the leading unnamed pandas-export index column** (107 → 106 named
   columns). The raw header begins ``,Name,…`` — the first cell is the
   ``0..23534`` row index written by an upstream ``DataFrame.to_csv``, not a
   TBDB field.
2. **Coordinate sentinel normalisation.** Literal ``0`` is a "no value found"
   sentinel in the INFERNAL coordinate columns listed in
   :data:`COORD_COLS_ZERO_SENTINEL` (not a valid 1-indexed position) → ``NaN``;
   ``discrim_end`` additionally carries ``-2`` as a sentinel → ``NaN``.
   Window-relative negatives in other columns are legitimate and preserved.
3. **``Regulation`` value-set assertion** (``Unknown`` preserved, never
   reclassified here — that is a later step).

Outputs (P0-12):

- ``data/interim/master_tboxes_ingested.parquet`` — the 106 canonical columns
  plus a deterministic per-record :data:`RECORD_HASH_COL` provenance column
  (DVC-tracked).
- ``data/processed/master_clean_v0.parquet`` — the 106-column cleaned training
  corpus (aliased from the interim artifact; the hash column dropped) that
  P1/P2/P3/P4 consume at that exact path (DVC-tracked).
- a **count-parse report** JSON: record/column counts, per-record
  hash-identity vs the tboxevo canonical parquet, the recomputed per-phylum
  counts + ratio, and the class I:II split (the §20 open item — reported).
- a ``provenance.json`` sidecar (CLAUDE.md §11) for the interim artifact.

The per-record hash is a **controlled SHA-256** over a canonical row
serialisation (not ``pandas.util.hash_pandas_object``, whose algorithm is
pandas-version-dependent), so the hash column and the golden digest are stable
and human-auditable across environments.

``pandas`` / ``pyarrow`` are imported **lazily** so this module imports in the
bare CI test env (``pyproject`` ``dependencies = []``); the heavy path runs in
the pinned ``data`` conda env (``pyarrow`` added for the parquet engine, P0-12).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime in a bare env
    import pandas as pd

#: Canonical record / column counts (PRD §4, §7.1).
EXPECTED_RECORDS = 23535
EXPECTED_RAW_COLS = 107  # includes the leading unnamed pandas-export index column
EXPECTED_NAMED_COLS = 106  # after dropping that index column

#: The deterministic per-record hash column appended to the interim artifact.
RECORD_HASH_COL = "record_sha256"

#: Columns where a literal ``0`` is a TBDB "no value found" sentinel (not a valid
#: 1-indexed coordinate). Kept identical to tboxevo's ``ingest_master_clean.py``.
COORD_COLS_ZERO_SENTINEL: tuple[str, ...] = (
    "Tbox_start",
    "Tbox_end",
    "s1_start",
    "s1_loop_start",
    "s1_loop_end",
    "s1_end",
    "antiterm_start",
    "antiterm_end",
    "term_start",
    "term_end",
    "discrim_start",
    "stem3_start",
    "stem3_end",
    "stem2_region_start",
    "stem2_region_end",
)

#: ``discrim_end`` carries the literal ``-2`` as its "no value" sentinel and is
#: **deliberately NOT in** :data:`COORD_COLS_ZERO_SENTINEL` — the pinned tboxevo
#: cleaner (``ingest_master_clean.py``) sentinels it on ``-2`` only, and the
#: canonical parquet has **0** ``discrim_end == 0`` records, so 0-sentinelling it
#: is a no-op that would encode an unvalidated semantics change vs the reproduced
#: contract (P0-12 gate is per-record identity with that canonical). Kept separate.
DISCRIM_END_NEG2_SENTINEL_COL = "discrim_end"

#: Allowed ``Regulation`` labels — asserted as an early-warning if TBDB changes
#: the upstream vocabulary. ``Unknown`` is preserved (PRD §7.1; CLAUDE.md §10).
REGULATION_ALLOWED: frozenset[str] = frozenset({"Transcriptional", "Translational", "Unknown"})

#: Column encoding the T-box structural/regulatory type. Its ``Transcriptional``
#: / ``Translational`` values map to class I / class II respectively
#: (PRD §3: "Transcriptional (class I) … Translational (class II)" [PMID:25583497;
#: DOI:10.1073/pnas.1424175112]). This is the most-complete class field (vs the
#: annotated ``Regulation`` column, which carries ~2 179 ``Unknown``).
CLASS_TYPE_COL = "type"
PHYLUM_COL = "phylum"

#: type → T-box class. Reported (non-gated) side output for the §20 open item.
TYPE_TO_CLASS: dict[str, str] = {"Transcriptional": "I", "Translational": "II"}

#: Single-char kind tags prefixed to each cell token so a value can never collide
#: with the missing sentinel across types (a real string ``"N"`` is ``"SN"``, the
#: missing token is ``"N"``); floats/ints/bools are typed so ``"1.5"`` the string
#: (``"S1.5"``) differs from ``1.5`` the float (``"F1.5"``).
_KIND_NA = "N"
_KIND_FLOAT = "F"
_KIND_INT = "I"
_KIND_BOOL = "B"
_KIND_STR = "S"


# --------------------------------------------------------------------------- #
# Controlled per-record hashing (stdlib only — runs in the bare CI test env)
# --------------------------------------------------------------------------- #
def _row_token(value: Any) -> str:
    """Canonicalise one cell to a **kind-tagged** stable string token.

    - ``None`` / NaN (numpy) / ``pd.NA`` (pandas nullable) → :data:`_KIND_NA`.
      ``pd.NA`` is detected via the ``TypeError`` its ambiguous ``__bool__`` raises
      on the self-inequality test, so a nullable cell never falls through to
      ``str(pd.NA) == "<NA>"``.
    - numpy scalars are demoted to their Python scalar (``.item()``) so the token
      does not depend on the numpy repr (``np.float64(1.5)`` reprs differently
      across numpy versions; the demoted ``float`` does not).
    - ``float`` → ``F`` + ``repr`` (shortest round-trip form, stable since Python
      3.1); ``bool`` → ``B0``/``B1``; ``int`` → ``I`` + decimal; ``str`` → ``S`` +
      the string. Any *other* type is tagged ``S<type-name>:<str>`` so two exotic
      values sharing a ``str()`` cannot collide — this branch is not reached on the
      TBDB corpus (cells are str/float/int/bool), so committed hashes are unaffected.

    Combined with the length framing in :func:`record_hash`, the encoding is
    injective: no cell value can be confused with the framing or another cell.
    """
    if value is None:
        return _KIND_NA
    # NaN is the only value not equal to itself; pd.NA's `!=` yields NA whose
    # bool() raises TypeError — both mean "missing".
    try:
        if bool(value != value):  # noqa: PLR0124 - deliberate NaN test
            return _KIND_NA
    except (TypeError, ValueError):
        return _KIND_NA  # pandas.NA (ambiguous truth value) → missing
    item = getattr(value, "item", None)
    if callable(item):
        with contextlib.suppress(ValueError, TypeError):  # pragma: no cover - defensive
            value = item()
    if isinstance(value, bool):
        return _KIND_BOOL + ("1" if value else "0")
    if isinstance(value, float):
        return _KIND_FLOAT + repr(value)
    if isinstance(value, int):
        return _KIND_INT + str(value)
    if isinstance(value, str):
        return _KIND_STR + value
    return _KIND_STR + type(value).__name__ + ":" + str(value)


def record_hash(values: Iterable[Any]) -> str:
    """SHA-256 hexdigest of one record's cell values, unambiguously framed.

    Each kind-tagged token is length-prefixed (``<byte-len>:<token>``) before it
    is fed to the digest, so a cell value that contains a separator, a delimiter,
    or the missing sentinel cannot collide with the framing or with another cell.
    """
    h = hashlib.sha256()
    for value in values:
        tok = _row_token(value).encode("utf-8")
        h.update(b"%d:" % len(tok))
        h.update(tok)
    return h.hexdigest()


def records_digest(hashes: Sequence[str]) -> str:
    """SHA-256 over the ordered per-record hashes — the whole-artifact digest.

    This is the value committed as a golden ``expected.sha256``: it depends only
    on the cleaning contract and the row order, not on parquet/pyarrow bytes, so
    it is reproducible in any env that has the cleaning inputs.
    """
    return hashlib.sha256(("\n".join(hashes) + "\n").encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Cleaning contract (lazy pandas — heavy path)
# --------------------------------------------------------------------------- #
def read_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw TBDB CSV with a proper RFC-4180 parse (no ``index_col``).

    Reading *without* ``index_col`` keeps the leading unnamed index column so the
    caller can assert the 107-raw-column parse gate; :func:`clean` then drops it,
    which is value-identical to tboxevo's ``read_csv(index_col=0)`` (verified).
    """
    import pandas as pd

    return pd.read_csv(path, low_memory=False)


def assert_parse_gate(
    df: pd.DataFrame,
    *,
    expected_records: int = EXPECTED_RECORDS,
    expected_raw_cols: int = EXPECTED_RAW_COLS,
) -> None:
    """Assert the raw parse recovered exactly ``expected_records × expected_raw_cols``.

    A mismatch is a **parse defect** (e.g. embedded-newline records folded in) —
    CLAUDE.md §7 requires a stop-and-ask, never silently absorbing "extra" rows.
    """
    n_rows, n_cols = df.shape
    if n_rows != expected_records or n_cols != expected_raw_cols:
        raise ValueError(
            "parse-gate FAIL: got "
            f"{n_rows} records × {n_cols} columns, expected "
            f"{expected_records} × {expected_raw_cols}. A raw line count is "
            "newline-inflated (embedded newlines in tRNA 2°-structure fields); "
            "a mismatch here is a parse defect, not extra records (PRD §7.1)."
        )


def _drop_unnamed_index(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the leading unnamed pandas-export index column.

    Handles both the ``Unnamed: 0`` name pandas assigns an empty-header first
    column and, defensively, a leading column whose name is empty.
    """
    if "Unnamed: 0" in df.columns:
        return df.drop(columns=["Unnamed: 0"])
    first = df.columns[0]
    if first == "" or str(first).startswith("Unnamed:"):
        return df.drop(columns=[first])
    return df


def _normalise_coord(series: pd.Series, sentinel: int, col: str) -> pd.Series:
    """Coerce a coordinate column to numeric, ``sentinel`` → NaN, failing loud on junk.

    Only the known sentinel (``0`` for the zero-sentinel columns, ``-2`` for
    ``discrim_end``) is treated as missing. Any *other* non-numeric, non-empty
    value — which ``to_numeric(errors="coerce")`` would silently turn into NaN and
    hide — raises instead (CLAUDE.md §10.3 fail-loud). On the real corpus no such
    value exists, so this never changes the cleaned output.
    """
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() & series.notna() & series.astype("string").str.strip().ne("")
    if bool(invalid.any()):
        bad = list(series[invalid].unique()[:5])
        raise ValueError(
            f"{col}: non-numeric coordinate value(s) {bad} would be silently coerced to "
            f"NaN (CLAUDE.md §10.3); expected a number or the {sentinel} sentinel"
        )
    return numeric.mask(numeric == sentinel)


def clean(
    df: pd.DataFrame,
    *,
    expect_records: int | None = EXPECTED_RECORDS,
    expect_named_cols: int | None = EXPECTED_NAMED_COLS,
) -> pd.DataFrame:
    """Apply the reproduction contract → the 106-column cleaned frame.

    ``expect_records=None`` / ``expect_named_cols=None`` skip the count
    assertions (small fixtures). Row order is preserved; the index is reset so
    the frame is positionally comparable to the canonical parquet.
    """
    out = _drop_unnamed_index(df).reset_index(drop=True)

    if expect_records is not None and len(out) != expect_records:
        raise ValueError(
            f"record count mismatch after clean: got {len(out)}, expected {expect_records} (PRD §4)"
        )
    if expect_named_cols is not None and out.shape[1] != expect_named_cols:
        raise ValueError(
            f"named-column count mismatch: got {out.shape[1]}, "
            f"expected {expect_named_cols} (PRD §4; cheatsheet §0)"
        )

    if "Regulation" in out.columns:
        seen = set(out["Regulation"].dropna().unique())
        bogus = seen - REGULATION_ALLOWED
        if bogus:
            raise ValueError(
                f"Regulation has unexpected labels: {sorted(bogus)} "
                f"(allowed: {sorted(REGULATION_ALLOWED)})"
            )

    for col in COORD_COLS_ZERO_SENTINEL:
        if col in out.columns:
            out[col] = _normalise_coord(out[col], 0, col)
    if DISCRIM_END_NEG2_SENTINEL_COL in out.columns:
        out[DISCRIM_END_NEG2_SENTINEL_COL] = _normalise_coord(
            out[DISCRIM_END_NEG2_SENTINEL_COL], -2, DISCRIM_END_NEG2_SENTINEL_COL
        )

    return out


def compute_record_hashes(df: pd.DataFrame) -> list[str]:
    """Per-record :func:`record_hash` over the frame's columns, in row order.

    Any :data:`RECORD_HASH_COL` already present is excluded so a frame's hashes
    do not depend on whether it has been hashed before.
    """
    cols = [c for c in df.columns if c != RECORD_HASH_COL]
    return [record_hash(row) for row in df[cols].itertuples(index=False, name=None)]


def add_record_hashes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the :data:`RECORD_HASH_COL` column appended."""
    out = df.copy()
    out[RECORD_HASH_COL] = compute_record_hashes(df)
    return out


def hash_identity(
    hashes: Sequence[str],
    canonical: pd.DataFrame,
    our_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare our per-record hashes against the canonical parquet's, positionally.

    When ``our_columns`` is given, the **column schema** (names + order, excluding
    the hash column) is compared too, so value-identical records under a renamed or
    reordered column cannot pass as identical. The gate requires ``pct == 100.0``
    *and* a matching schema.
    """
    canon_cols = [c for c in canonical.columns if c != RECORD_HASH_COL]
    canon_hashes = compute_record_hashes(canonical)
    n = len(hashes)
    n_canon = len(canon_hashes)
    # strict=False is deliberate: a length mismatch is reported via `identical`
    # (n == n_canon) rather than raising; positional comparison stops at the shorter.
    n_match = sum(1 for a, b in zip(hashes, canon_hashes, strict=False) if a == b)
    # denominator is max(n, n_canon) so BOTH a missing prefix (n < n_canon) and
    # extra records (n > n_canon) drop pct below 100 — not just the shorter side.
    denom = max(n, n_canon)
    pct = (100.0 * n_match / denom) if denom else 100.0
    columns_match = None if our_columns is None else (list(our_columns) == canon_cols)
    return {
        "n_records": n,
        "n_canonical_records": n_canon,
        "n_matching": n_match,
        "pct_identity": round(pct, 6),
        "n_canonical_columns": len(canon_cols),
        "columns_match": columns_match,
        "identical": (n == n_canon and n_match == n and columns_match is not False),
        "our_digest": records_digest(list(hashes)),
        "canonical_digest": records_digest(canon_hashes),
    }


# --------------------------------------------------------------------------- #
# §20 open-item reporting: per-phylum counts + class I:II split
# --------------------------------------------------------------------------- #
def phylum_report(df: pd.DataFrame) -> dict[str, Any]:
    """Per-phylum record counts + fractions (NaN phylum reported explicitly)."""
    import pandas as pd

    if PHYLUM_COL not in df.columns:
        return {"available": False}
    vc = df[PHYLUM_COL].value_counts(dropna=False)
    n = int(len(df))
    counts: dict[str, int] = {}
    fracs: dict[str, float] = {}
    for key, cnt in vc.items():
        label = "NaN" if pd.isna(key) else str(key)
        counts[label] = int(cnt)
        fracs[label] = round(100.0 * int(cnt) / n, 4) if n else 0.0
    top = max(counts, key=counts.get) if counts else None
    return {
        "available": True,
        "n_phyla": int(df[PHYLUM_COL].nunique(dropna=True)),
        "counts": counts,
        "pct": fracs,
        "dominant_phylum": top,
        "dominant_pct": fracs.get(top) if top else None,
    }


def class_split_report(df: pd.DataFrame) -> dict[str, Any]:
    """Class I:II split from the ``type`` column (transcriptional:translational)."""
    import pandas as pd

    if CLASS_TYPE_COL not in df.columns:
        return {"available": False}
    vc = df[CLASS_TYPE_COL].value_counts(dropna=False)
    by_type: dict[str, int] = {}
    by_class: dict[str, int] = {"I": 0, "II": 0, "unassigned": 0}
    for key, cnt in vc.items():
        label = "NaN" if pd.isna(key) else str(key)
        by_type[label] = int(cnt)
        cls = TYPE_TO_CLASS.get(label)
        if cls is None:
            by_class["unassigned"] += int(cnt)
        else:
            by_class[cls] += int(cnt)
    ratio = (by_class["I"] / by_class["II"]) if by_class["II"] else None
    return {
        "available": True,
        "class_col": CLASS_TYPE_COL,
        "class_mapping": TYPE_TO_CLASS,
        "by_type": by_type,
        "class_I": by_class["I"],
        "class_II": by_class["II"],
        "unassigned": by_class["unassigned"],
        "ratio_I_to_II": round(ratio, 4) if ratio is not None else None,
        "citation": "PRD §3; PMID:25583497; DOI:10.1073/pnas.1424175112",
    }


def build_report(
    *,
    raw_shape: tuple[int, int],
    cleaned_shape: tuple[int, int],
    identity: dict[str, Any],
    df_clean: pd.DataFrame,
    expected_records: int = EXPECTED_RECORDS,
    expected_raw_cols: int = EXPECTED_RAW_COLS,
    expected_named_cols: int = EXPECTED_NAMED_COLS,
) -> dict[str, Any]:
    """Assemble the count-parse report (PRD §7.1).

    The ``expected_*`` counts default to the production constants but are threaded
    from :func:`run_ingest` so a fixture-sized run (e.g. 100 records) reports and
    grades against its own expected counts, not the 23,535 production values.
    """
    return {
        "schema_version": "1.0",
        "step": "P0-12",
        "prd_section": "7.1",
        "parse_gate": {
            "raw_records": raw_shape[0],
            "raw_columns": raw_shape[1],
            "expected_raw_records": expected_records,
            "expected_raw_columns": expected_raw_cols,
            "cleaned_records": cleaned_shape[0],
            "cleaned_columns": cleaned_shape[1],
            "expected_cleaned_columns": expected_named_cols,
            "raw_line_count_note": (
                "39459 raw lines is a newline-inflated count (embedded newlines "
                "in tRNA 2°-structure fields); the proper parse recovers "
                f"{EXPECTED_RECORDS} records."
            ),
            "passed": (
                raw_shape == (expected_records, expected_raw_cols)
                and cleaned_shape == (expected_records, expected_named_cols)
            ),
        },
        "hash_identity": identity,
        "phylum": phylum_report(df_clean),
        "class_split": class_split_report(df_clean),
    }


def _write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", compression="snappy", index=False)


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def run_ingest(
    *,
    raw_csv: str | Path,
    canonical_parquet: str | Path,
    out_interim: str | Path,
    out_processed: str | Path,
    out_report: str | Path,
    out_provenance: str | Path | None,
    env_lock: str | Path | None,
    expect_records: int = EXPECTED_RECORDS,
    expect_raw_cols: int = EXPECTED_RAW_COLS,
    require_identity: bool = True,
) -> dict[str, Any]:
    """Full P0-12 ingest: parse gate → clean → hash-identity → write artifacts.

    Returns the count-parse report dict. Raises on a parse-gate failure or, when
    ``require_identity`` is set, on any per-record hash-identity below 100 %
    (CLAUDE.md §8.5/§10.3 — never emit a corpus that fails the reproduction gate).
    """
    import pandas as pd

    from tbox_finder import provenance

    # Cleaned named-column count is an invariant: exactly one column (the unnamed
    # index) is dropped, so it is always raw_cols - 1 — asserted for fixture runs
    # too, not inferred from the observed width (no self-fulfilling schema check).
    expect_named = expect_raw_cols - 1

    raw = read_raw(raw_csv)
    assert_parse_gate(raw, expected_records=expect_records, expected_raw_cols=expect_raw_cols)
    df_clean = clean(raw, expect_records=expect_records, expect_named_cols=expect_named)

    hashes = compute_record_hashes(df_clean)
    canonical = pd.read_parquet(canonical_parquet)
    identity = hash_identity(hashes, canonical, our_columns=list(df_clean.columns))

    if require_identity and not identity["identical"]:
        raise ValueError(
            "per-record hash-identity gate FAIL: "
            f"{identity['n_matching']}/{identity['n_records']} match "
            f"({identity['pct_identity']}%), columns_match={identity['columns_match']}; "
            f"the ingest does not reproduce the canonical cleaned corpus "
            f"{canonical_parquet} (PRD §7.1; CLAUDE.md §10.3)"
        )

    df_interim = df_clean.copy()
    df_interim[RECORD_HASH_COL] = hashes
    _write_parquet(df_interim, out_interim)
    _write_parquet(df_clean, out_processed)  # processed alias: 106 canonical cols

    report = build_report(
        raw_shape=(int(raw.shape[0]), int(raw.shape[1])),
        cleaned_shape=(int(df_clean.shape[0]), int(df_clean.shape[1])),
        identity=identity,
        df_clean=df_clean,
        expected_records=expect_records,
        expected_raw_cols=expect_raw_cols,
        expected_named_cols=expect_named,
    )
    _write_json(report, out_report)

    if out_provenance is not None:
        provenance.write_provenance(
            out_provenance,
            rule="workflow/rules/data.smk :: ingest_master",
            script="src/tbox_finder/ingest.py",
            inputs=[raw_csv, canonical_parquet],
            outputs=[out_interim, out_processed],
            env_lock=env_lock,
            adr=None,
            extra={
                "record_count": int(df_clean.shape[0]),
                "named_columns": int(df_clean.shape[1]),
                "record_hash_column": RECORD_HASH_COL,
                "pct_identity_vs_canonical": identity["pct_identity"],
            },
        )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-csv", required=True, type=Path)
    parser.add_argument("--canonical-parquet", required=True, type=Path)
    parser.add_argument("--out-interim", required=True, type=Path)
    parser.add_argument("--out-processed", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--out-provenance", type=Path, default=None)
    parser.add_argument(
        "--env-lock",
        type=Path,
        default=Path("envs/data.conda-lock.yml"),
        help="conda-lock lockfile hashed into provenance.json (CLAUDE.md §11).",
    )
    parser.add_argument("--expect-records", type=int, default=EXPECTED_RECORDS)
    parser.add_argument(
        "--allow-identity-shortfall",
        action="store_true",
        help="Do NOT use in the gate — for diagnostics only; skips the 100%% "
        "per-record identity assertion.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_ingest(
        raw_csv=args.raw_csv,
        canonical_parquet=args.canonical_parquet,
        out_interim=args.out_interim,
        out_processed=args.out_processed,
        out_report=args.out_report,
        out_provenance=args.out_provenance,
        env_lock=args.env_lock if args.env_lock and str(args.env_lock) else None,
        expect_records=args.expect_records,
        require_identity=not args.allow_identity_shortfall,
    )
    pg = report["parse_gate"]
    hi = report["hash_identity"]
    print(
        f"[ingest] parse-gate {'PASS' if pg['passed'] else 'FAIL'}: "
        f"{pg['raw_records']}×{pg['raw_columns']} raw → "
        f"{pg['cleaned_records']}×{pg['cleaned_columns']} clean; "
        f"identity {hi['pct_identity']}% ({hi['n_matching']}/{hi['n_records']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
