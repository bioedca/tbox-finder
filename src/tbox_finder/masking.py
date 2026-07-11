"""Union-prior loci-masking + the ADR-0006 D11 / ADR-0005 D14 mining spare-rule (P0-30).

Two model-independent guards protect the §9.1 negative/decoy pools from being
contaminated by, or from mining away, real T-box loci:

1. **Loci masking (this module's ``LocusIndex``).** All known T-box loci — the
   **full union prior** (``data/processed/priors/union_prior.parquet``; masking
   key = ``accession`` + ``locus_start``/``locus_end`` where present) **+ the
   run's own training positives + a flank** — are masked from every negative
   pool (PRD §9.1; ADR-0006 D11; matching the §13.3 clade-level null's masking
   reference). The **residual contamination rate against that union denominator**
   is reported: the fraction of the union prior that carries *no* coordinates
   (the literature-clade records) and therefore cannot be coordinate-masked.

2. **The spare rule (``spare_rule_excludes_from_mining``).** Because a CM-missed
   **non-canonical (Tier-2N)** T-box is unknown to masking *and* fails any
   canonical-architecture predicate, the mining-exclusion rule is **not keyed to
   canonical architecture** but to a spare rule: a candidate is excluded from the
   hard-negative-mining pool if it passes **relaxed-architecture detection OR
   any-helix R-scape covariation OR downstream-aaRS synteny** (the three
   *model-independent* disjuncts, sufficient for the P2 Stage-1 mining loop where
   no Stage-2 exists yet) **OR**, at the P3 re-mining round once Stage-2 exists,
   a **high Stage-2 posterior**. This is the anti-mimicry guard: aggressive
   mining cannot directionally train the production scanner to reject the
   flagship Tier-2N class (worst case = directionally-bounded discovery
   sensitivity, not an invalid generalization claim; ADR-0005 D14).

Everything above the pandas boundary is stdlib-only so ``tests/unit/test_masking.py``
exercises it without pandas/numpy; the parquet loaders (lazy pandas) are below.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Union-prior columns (P0-14 schema; see data/processed/priors/union_prior.parquet).
UNION_ACCESSION_COL = "accession"
UNION_START_COL = "locus_start"
UNION_END_COL = "locus_end"
# Own-positives corpus columns (P0-12 master_clean_v0 schema).
CORPUS_ACCESSION_COL = "accession_name"
CORPUS_START_COL = "locus_start"
CORPUS_END_COL = "locus_end"

# Default flank (nt) added on each side of every known locus before masking
# (PRD §9.1 "plus a flank"). Config-overridable via conf/data/decoys.yaml.
DEFAULT_FLANK_NT = 50


# --------------------------------------------------------------------------- #
# Pure-stdlib locus geometry (unit-testable without pandas)
# --------------------------------------------------------------------------- #
def normalize_locus(start: int, end: int) -> tuple[int, int]:
    """Return ``(lo, hi)`` inclusive bounds, folding reverse-strand ``start > end``.

    The union prior and the corpus both encode strand implicitly as
    ``locus_start > locus_end`` (P0-14 / P0-12); masking is strand-agnostic, so a
    locus is the closed genomic interval ``[min, max]``.
    """
    s, e = int(start), int(end)
    return (s, e) if s <= e else (e, s)


def intervals_overlap(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> bool:
    """True iff closed intervals ``[a_lo, a_hi]`` and ``[b_lo, b_hi]`` intersect."""
    return a_lo <= b_hi and b_lo <= a_hi


class LocusIndex:
    """Accession → sorted, merged known-T-box intervals, with a flanked overlap test.

    Built from the union prior + the run's own training positives (both projected
    to ``(accession, lo, hi)`` closed intervals). ``is_masked`` expands the query
    locus by ``flank`` on both sides before testing overlap, realising the PRD §9.1
    "known loci + flank are masked from every pool" rule.
    """

    def __init__(self, by_accession: Mapping[str, Sequence[tuple[int, int]]]):
        # Store merged, start-sorted intervals per accession + a parallel list of
        # interval starts for a bisect-narrowed overlap scan.
        self._starts: dict[str, list[int]] = {}
        self._intervals: dict[str, list[tuple[int, int]]] = {}
        for acc, ivals in by_accession.items():
            merged = _merge_intervals(ivals)
            self._intervals[str(acc)] = merged
            self._starts[str(acc)] = [lo for lo, _ in merged]

    @classmethod
    def from_records(cls, records: Iterable[tuple[str, int, int]]) -> LocusIndex:
        """Build from ``(accession, start, end)`` triples (strand folded, coords normalized)."""
        by_acc: dict[str, list[tuple[int, int]]] = {}
        for acc, start, end in records:
            if acc is None:
                continue
            by_acc.setdefault(str(acc), []).append(normalize_locus(start, end))
        return cls(by_acc)

    @property
    def n_accessions(self) -> int:
        return len(self._intervals)

    @property
    def n_intervals(self) -> int:
        return sum(len(v) for v in self._intervals.values())

    def is_masked(self, accession: Any, start: int, end: int, flank: int = 0) -> bool:
        """True iff ``[start, end]`` (± ``flank``) overlaps a known locus on ``accession``.

        A record with no accession (synthetic decoy) carries no genomic
        coordinates and can never overlap a known locus → never masked.
        """
        if accession is None or _is_missing(accession):
            return False
        if flank < 0:
            raise ValueError(f"flank must be >= 0, got {flank}")
        acc = str(accession)
        intervals = self._intervals.get(acc)
        if not intervals:
            return False
        q_lo, q_hi = normalize_locus(start, end)
        q_lo -= flank
        q_hi += flank
        # Intervals are disjoint and start-sorted: the only one that can overlap
        # [q_lo, q_hi] is the last whose start <= q_hi. If that one's hi < q_lo then
        # (disjointness) every earlier interval ends even further left, so no overlap.
        starts = self._starts[acc]
        cut = bisect_right(starts, q_hi)
        if cut == 0:
            return False
        lo, hi = intervals[cut - 1]
        return intervals_overlap(q_lo, q_hi, lo, hi)


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge overlapping/adjacent closed intervals into disjoint spans."""
    items = sorted((int(lo), int(hi)) for lo, hi in intervals)
    if not items:
        return []
    merged: list[tuple[int, int]] = [items[0]]
    for lo, hi in items[1:]:
        m_lo, m_hi = merged[-1]
        if lo <= m_hi + 1:  # overlapping or directly adjacent
            merged[-1] = (m_lo, max(m_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _is_missing(value: Any) -> bool:
    """True for ``None`` / NaN / pandas-NA (mirrors ingest._row_token missing test)."""
    if value is None:
        return True
    try:
        return bool(value != value)  # noqa: PLR0124 - NaN is the only x != x
    except (TypeError, ValueError):
        return True


# --------------------------------------------------------------------------- #
# The ADR-0006 D11 / ADR-0005 D14 mining spare-rule (pure predicate)
# --------------------------------------------------------------------------- #
def spare_rule_excludes_from_mining(
    *,
    relaxed_architecture: bool = False,
    any_helix_rscape: bool = False,
    downstream_aaRS_synteny: bool = False,
    stage2_posterior: float | None = None,
    stage2_threshold: float | None = None,
) -> bool:
    """Exclude a candidate from the hard-negative-mining pool per the spare rule.

    Returns ``True`` (⇒ **keep out of the mining pool**) iff the candidate passes
    any of the three **model-independent** disjuncts — relaxed-architecture
    detection, any-helix R-scape covariation, downstream-aaRS synteny — which are
    sufficient for the **P2 Stage-1 mining loop** where no Stage-2 exists yet;
    **or**, at the **P3 re-mining round once Stage-2 exists**, a **high Stage-2
    posterior** (``stage2_posterior >= stage2_threshold``). The rule is deliberately
    *not keyed to canonical architecture*, so a CM-invisible Tier-2N locus (which
    fails every canonical predicate) is still protected from being mined away
    (ADR-0006 D11; ADR-0005 D14; PRD §9.1).

    ``stage2_posterior`` alone is inert (the P2 round has no Stage-2); it excludes
    only when a ``stage2_threshold`` is also supplied (the P3 round).
    """
    for name, val in (
        ("relaxed_architecture", relaxed_architecture),
        ("any_helix_rscape", any_helix_rscape),
        ("downstream_aaRS_synteny", downstream_aaRS_synteny),
    ):
        if not isinstance(val, bool):
            raise TypeError(f"{name} must be bool, got {type(val).__name__}")
    if relaxed_architecture or any_helix_rscape or downstream_aaRS_synteny:
        return True
    if stage2_posterior is None or stage2_threshold is None:
        return False
    for name, val in (
        ("stage2_posterior", stage2_posterior),
        ("stage2_threshold", stage2_threshold),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a real number, got {type(val).__name__}")
        if not 0.0 <= float(val) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {val}")
    return float(stage2_posterior) >= float(stage2_threshold)


# --------------------------------------------------------------------------- #
# Residual-contamination report (pure)
# --------------------------------------------------------------------------- #
def residual_contamination_report(
    *, n_union_total: int, n_union_maskable: int, pool_mask_counts: Mapping[str, int]
) -> dict[str, Any]:
    """Residual contamination of the negative pools against the union denominator.

    ``n_union_total`` = every known-T-box record in the union prior;
    ``n_union_maskable`` = those carrying ``(accession, locus_start, locus_end)``.
    The **residual** is the un-maskable remainder (literature-clade records with no
    coordinates, ADR-0006 D11): they cannot be coordinate-masked from any pool, so
    they are the reported residual-contamination risk against the union denominator.
    ``pool_mask_counts`` maps each pool → the number of its records masked out.
    """
    if n_union_total < 0 or n_union_maskable < 0:
        raise ValueError("union counts must be non-negative")
    if n_union_maskable > n_union_total:
        raise ValueError(f"n_union_maskable ({n_union_maskable}) > n_union_total ({n_union_total})")
    n_unmaskable = n_union_total - n_union_maskable
    residual_fraction = (n_unmaskable / n_union_total) if n_union_total else 0.0
    return {
        "union_denominator": int(n_union_total),
        "union_maskable_with_coords": int(n_union_maskable),
        "union_residual_no_coords": int(n_unmaskable),
        "residual_contamination_fraction": residual_fraction,
        "pool_records_masked": {str(k): int(v) for k, v in pool_mask_counts.items()},
        "total_pool_records_masked": int(sum(pool_mask_counts.values())),
    }


# --------------------------------------------------------------------------- #
# Lazy-pandas loaders (heavy path)
# --------------------------------------------------------------------------- #
def load_union_loci(union_parquet: str | Path) -> tuple[list[tuple[str, int, int]], int, int]:
    """Load ``(accession, start, end)`` triples from the union prior.

    Returns ``(loci, n_total, n_maskable)`` where ``n_total`` is every union-prior
    record and ``n_maskable`` is the subset carrying all of accession/start/end.
    """
    import pandas as pd

    df = pd.read_parquet(
        union_parquet, columns=[UNION_ACCESSION_COL, UNION_START_COL, UNION_END_COL]
    )
    n_total = len(df)
    has_coords = (
        df[UNION_ACCESSION_COL].notna() & df[UNION_START_COL].notna() & df[UNION_END_COL].notna()
    )
    kept = df[has_coords]
    loci = [
        (str(a), int(s), int(e))
        for a, s, e in zip(
            kept[UNION_ACCESSION_COL], kept[UNION_START_COL], kept[UNION_END_COL], strict=True
        )
    ]
    return loci, int(n_total), int(len(kept))


def load_own_positive_loci(corpus_parquet: str | Path) -> list[tuple[str, int, int]]:
    """Load the run's own training-positive loci from the P0-12 corpus.

    Uses ``(accession_name, locus_start, locus_end)`` — the P0-20-labelled corpus
    is the run's own positive set (PRD §9.1 "the run's own training positives").
    """
    import pandas as pd

    df = pd.read_parquet(
        corpus_parquet, columns=[CORPUS_ACCESSION_COL, CORPUS_START_COL, CORPUS_END_COL]
    )
    has_coords = (
        df[CORPUS_ACCESSION_COL].notna() & df[CORPUS_START_COL].notna() & df[CORPUS_END_COL].notna()
    )
    kept = df[has_coords]
    return [
        (str(a), int(s), int(e))
        for a, s, e in zip(
            kept[CORPUS_ACCESSION_COL], kept[CORPUS_START_COL], kept[CORPUS_END_COL], strict=True
        )
    ]
