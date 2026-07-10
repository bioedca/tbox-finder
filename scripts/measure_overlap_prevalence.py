#!/usr/bin/env python3
"""Measure the all-pairs element-overlap prevalence on the curated T-box corpus.

This is the reproducible provenance for the **precedence table pinned in
ADR-0004** (P0-19). It maps each of the seven annotated T-box structural
elements to its ``(start, end)`` coordinate columns in
``data/processed/master_clean_v0.parquet`` (the P0-12 ingest), then, for every
unordered pair of elements, reports over the records where *both* elements are
annotated:

  * ``n_both``      — records with valid coordinates for both elements;
  * ``n_overlap``   — records whose extents overlap (inclusive-nt);
  * ``prev_pct``    — 100 * n_overlap / n_both (the overlap prevalence);
  * ``median_bp`` / ``mean_bp`` — overlap length among overlapping records;
  * ``A_in_B`` / ``B_in_A`` (+ pct) — directional containment counts.

The label-derivation single-label precedence rule (PRD §8; ADR-0004 D1) is
justified only where an overlap is *material*; every other pair must be shown to
be effectively disjoint. This script produces the numbers that back both claims,
so ADR-0004 pins measured values, not asserted ones (CLAUDE.md §10.3).

Coordinates are per-record local (each T-box locus carries its own origin), so
overlaps are computed within a record between that record's own elements and
then aggregated across records — no cross-record coordinate comparison is made.
Intervals are normalised to ``(min, max)`` (minus-strand robustness); any
coordinate that is null or negative (the corpus uses ``-1`` / NaN sentinels for
an unannotated element) marks that element absent for the record.

Pure-stdlib import surface (argparse/json/hashlib/pathlib); pandas/pyarrow are
imported lazily inside :func:`measure` so the module stays bare-importable in
the CI test env (matching ``src/tbox_finder/priors.py`` / ``anchors.py``).

Usage::

    python scripts/measure_overlap_prevalence.py \
        --corpus data/processed/master_clean_v0.parquet \
        --out data/processed/audits/overlap_prevalence_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

# Element name -> (start_column, end_column) in master_clean_v0.parquet.
# `Specifier` is the annotated specifier codon (codon_start/codon_end); P0-20 /
# P0-21 may widen it to the specifier loop against the crystal fixtures, which
# does not change the more-specific-wins precedence rule.
ELEMENT_COLUMNS: dict[str, tuple[str, str]] = {
    "Stem_I": ("s1_start", "s1_end"),
    "Specifier": ("codon_start", "codon_end"),
    "Stem_II": ("stem2_region_start", "stem2_region_end"),
    "Stem_III": ("stem3_start", "stem3_end"),
    "Antiterminator": ("antiterm_start", "antiterm_end"),
    "Terminator": ("term_start", "term_end"),
    "Discriminator": ("discrim_start", "discrim_end"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_overlaps(df) -> dict:
    """Pure core: per-element presence + all-pairs overlap/containment stats.

    Takes a DataFrame carrying the :data:`ELEMENT_COLUMNS` coordinate columns and
    returns ``{"n_records", "presence", "pairs"}``. Kept separate from I/O so it
    is unit-testable on a synthetic fixture (``tests/unit/test_overlap_prevalence``)
    without the full gitignored corpus.
    """
    import numpy as np  # noqa: PLC0415 (lazy — keep module bare-importable)

    n_records = int(len(df))

    intervals: dict[str, tuple] = {}
    present: dict[str, object] = {}
    for name, (s_col, e_col) in ELEMENT_COLUMNS.items():
        if s_col not in df.columns or e_col not in df.columns:
            raise KeyError(f"{name}: missing coordinate column {s_col!r}/{e_col!r}")
        a = df[s_col].to_numpy(dtype=float)
        b = df[e_col].to_numpy(dtype=float)
        bad = np.isnan(a) | np.isnan(b) | (a < 0) | (b < 0)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        lo[bad] = np.nan
        hi[bad] = np.nan
        intervals[name] = (lo, hi)
        present[name] = ~bad

    presence = {
        name: {
            "n": int(mask.sum()),
            "pct": (round(100.0 * int(mask.sum()) / n_records, 2) if n_records else 0.0),
        }
        for name, mask in present.items()
    }

    def overlap_len(lo1, hi1, lo2, hi2):
        # inclusive-nt overlap length: min(hi) - max(lo) + 1, clipped at 0.
        return np.maximum(0.0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2) + 1.0)

    pairs = []
    for a_name, b_name in itertools.combinations(ELEMENT_COLUMNS, 2):
        lo_a, hi_a = intervals[a_name]
        lo_b, hi_b = intervals[b_name]
        both = present[a_name] & present[b_name]
        n_both = int(both.sum())
        record = {
            "A": a_name,
            "B": b_name,
            "n_both": n_both,
            "n_overlap": 0,
            "prev_pct": 0.0,
            "median_bp": None,
            "mean_bp": None,
            "A_in_B": 0,
            "B_in_A": 0,
            "A_in_B_pct": 0.0,
            "B_in_A_pct": 0.0,
        }
        if n_both:
            ol = overlap_len(lo_a, hi_a, lo_b, hi_b)
            ov = both & (ol > 0)
            n_ov = int(ov.sum())
            a_in_b = both & (lo_a >= lo_b) & (hi_a <= hi_b)
            b_in_a = both & (lo_b >= lo_a) & (hi_b <= hi_a)
            record.update(
                n_overlap=n_ov,
                prev_pct=round(100.0 * n_ov / n_both, 2),
                median_bp=(float(np.median(ol[ov])) if n_ov else None),
                mean_bp=(round(float(np.mean(ol[ov])), 2) if n_ov else None),
                A_in_B=int(a_in_b.sum()),
                B_in_A=int(b_in_a.sum()),
                A_in_B_pct=round(100.0 * int(a_in_b.sum()) / n_both, 2),
                B_in_A_pct=round(100.0 * int(b_in_a.sum()) / n_both, 2),
            )
        pairs.append(record)

    return {"n_records": n_records, "presence": presence, "pairs": pairs}


def measure(corpus: Path) -> dict:
    """Read the corpus parquet and compute the overlap report with provenance."""
    import pandas as pd  # noqa: PLC0415 (lazy — keep module bare-importable)

    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus} — `dvc pull` the P0-12 artifact first")
    df = pd.read_parquet(corpus)  # compute_overlaps validates columns fail-loud
    return {
        "corpus": str(corpus),
        "corpus_sha256": _sha256(corpus),
        "element_columns": {k: list(v) for k, v in ELEMENT_COLUMNS.items()},
        **compute_overlaps(df),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/master_clean_v0.parquet"),
        help="curated corpus parquet (P0-12)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/audits/overlap_prevalence_report.json"),
        help="destination audit JSON",
    )
    args = p.parse_args(argv)

    # Never let --out alias --corpus: writing the JSON report would overwrite
    # (and destroy) the input parquet. Guard resolved-path and link identity.
    if args.out.resolve() == args.corpus.resolve() or (
        args.out.exists() and args.corpus.exists() and args.out.samefile(args.corpus)
    ):
        p.error("--out must not alias --corpus (would overwrite the input artifact)")

    report = measure(args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")

    # Human-readable summary of the material overlaps.
    material = [r for r in report["pairs"] if r["prev_pct"] >= 1.0]
    print(f"corpus: {report['corpus']}  n_records={report['n_records']}")
    print(f"corpus_sha256={report['corpus_sha256']}")
    print("material overlaps (prev >= 1%):")
    for r in material:
        print(
            f"  {r['A']:15}∩{r['B']:15} "
            f"prev={r['prev_pct']:6.2f}%  median={r['median_bp']}bp  "
            f"({r['A']}⊂{r['B']}={r['A_in_B_pct']}%, "
            f"{r['B']}⊂{r['A']}={r['B_in_A_pct']}%)"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
