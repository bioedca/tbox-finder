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

import re
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

#: Trailing INSDC sequence-version suffix (``ABFD02000009.1`` → ``ABFD02000009``).
#: Stripped on **both** sides of every masking comparison — see
#: :func:`normalize_accession` for why this is a correctness requirement, not a
#: convenience.
_ACCESSION_VERSION_RE = re.compile(r"\.\d+$")


# --------------------------------------------------------------------------- #
# Pure-stdlib locus geometry (unit-testable without pandas)
# --------------------------------------------------------------------------- #
def normalize_accession(accession: Any) -> str:
    """Strip the trailing INSDC ``.version`` so both sides of a mask share a namespace.

    The union prior and the P0-12 corpus store accessions **unversioned**
    (measured P2-10b: 0 of 12,041 union-prior accessions carry a ``.N`` suffix),
    while every external coordinate source the negative pools draw on stores them
    **versioned** (2,999 of 3,000 Rfam decoy headers; 23,535 of 23,535
    ``flank_context`` rows). Comparing the two namespaces raw makes the mask a
    silent no-op: measured intersection of the Rfam decoy accessions with the
    ``LocusIndex`` keys is **0 of 2,750 distinct** as-is versus **269** after
    stripping.

    That failure mode is worse than the P0 one it replaces. At P0 the pools
    carried no coordinates at all, so ``total_pool_records_masked = 0`` was
    *visibly* vacuous. With versioned coordinates the columns are non-null, the
    pipeline runs clean, and the count is still 0 — with nothing in the artifact
    saying why. Normalising here, at the shared boundary rather than in each
    producer, is what stops the next pool re-introducing it (ADR-0006 D11
    impact list; PRD §9.1).

    A version bump can in principle move coordinates, so this trades a small
    false-*positive* mask risk (a stale version's interval masking a record it
    no longer covers) for the certainty of masking nothing at all. Masking is a
    protective over-approximation — PRD §9.1 already widens every locus by a
    flank — so erring toward masking is the correct direction.
    """
    return _ACCESSION_VERSION_RE.sub("", str(accession))


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

    Keys are stored :func:`normalize_accession`-normalised and queries are
    normalised the same way, so a versioned query accession cannot silently miss
    an unversioned index key (P2-10b). The **raw** keys are retained in
    :attr:`raw_accession_keys` purely so a report can measure the as-is
    intersection and show that the normalisation is doing work.
    """

    def __init__(self, by_accession: Mapping[str, Sequence[tuple[int, int]]]):
        # Store merged, start-sorted intervals per accession + a parallel list of
        # interval starts for a bisect-narrowed overlap scan. Two source accessions
        # differing only by version collapse to one key, so pool their intervals
        # before merging rather than letting the later one overwrite the earlier.
        self._starts: dict[str, list[int]] = {}
        self._intervals: dict[str, list[tuple[int, int]]] = {}
        self._raw_keys: frozenset[str] = frozenset(str(a) for a in by_accession)
        pooled: dict[str, list[tuple[int, int]]] = {}
        for acc, ivals in by_accession.items():
            pooled.setdefault(normalize_accession(acc), []).extend(ivals)
        # Pre-merge intervals, kept for exact-identity queries only. Merging is
        # what makes the overlap scan fast and correct, but it is lossy: two
        # overlapping or adjacent loci collapse into one span, so an individual
        # locus's own interval may no longer be present. Measured P2-10b: 769 of
        # 23,532 locus-centred windows fail an exact match against the merged
        # spans and all 23,532 match against these.
        self._source_intervals: dict[str, frozenset[tuple[int, int]]] = {}
        for acc, ivals in pooled.items():
            merged = _merge_intervals(ivals)
            self._intervals[acc] = merged
            self._starts[acc] = [lo for lo, _ in merged]
            self._source_intervals[acc] = frozenset(normalize_locus(lo, hi) for lo, hi in ivals)

    @classmethod
    def from_records(cls, records: Iterable[tuple[str, int, int]]) -> LocusIndex:
        """Build from ``(accession, start, end)`` triples (strand folded, coords normalized)."""
        by_acc: dict[str, list[tuple[int, int]]] = {}
        for acc, start, end in records:
            if is_missing(acc):
                continue
            by_acc.setdefault(str(acc), []).append(normalize_locus(start, end))
        return cls(by_acc)

    @property
    def n_accessions(self) -> int:
        return len(self._intervals)

    @property
    def n_intervals(self) -> int:
        return sum(len(v) for v in self._intervals.values())

    @property
    def accession_keys(self) -> frozenset[str]:
        """The normalised accession keys actually used for lookup."""
        return frozenset(self._intervals)

    @property
    def raw_accession_keys(self) -> frozenset[str]:
        """The accession strings as supplied, before version normalisation."""
        return self._raw_keys

    def matches_interval_exactly(self, accession: Any, start: int, end: int) -> bool:
        """True iff ``[start, end]`` reproduces a known interval **exactly**.

        Overlap is the right test for masking — a decoy anywhere near a known
        locus must go — but it is far too weak to *validate a coordinate frame*.
        A window carved to be the locus itself still overlaps it after a shift of
        up to the locus length (median 281 nt), so a 100 %-overlap control
        tolerates a ±165 nt frame error (measured P2-10b: uniform shifts through
        ±160 keep 500/500 controls overlapping). Exact reproduction has zero
        tolerance, and it holds on the real data — **23,532/23,532** locus-centred
        windows reproduce their own union-prior/corpus interval — so the stricter
        test is available for free and every control becomes discriminating
        rather than only the ~1.6 % that a shift happens to push off the end.

        Matched against the **pre-merge source** intervals, not the merged spans
        :meth:`is_masked` scans: merging is lossy, and 769 of those 23,532
        windows sit on accessions whose loci merged, so the merged form would
        report 22,763/23,532 and no honest threshold could be set on it.
        """
        if is_missing(accession):
            return False
        intervals = self._source_intervals.get(normalize_accession(accession))
        if not intervals:
            return False
        return normalize_locus(start, end) in intervals

    def is_masked(self, accession: Any, start: int, end: int, flank: int = 0) -> bool:
        """True iff ``[start, end]`` (± ``flank``) overlaps a known locus on ``accession``.

        A record with no accession (synthetic decoy) carries no genomic
        coordinates and can never overlap a known locus → never masked.
        """
        if accession is None or is_missing(accession):
            return False
        if flank < 0:
            raise ValueError(f"flank must be >= 0, got {flank}")
        acc = normalize_accession(accession)
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


def is_missing(value: Any) -> bool:
    """True for ``None`` / NaN / pandas-NA (mirrors ingest._row_token missing test).

    Public because every guard that decides "does this candidate have usable
    coordinates?" must apply the *same* missing test as the mask itself. A guard
    that tests only ``is not None`` accepts the ``NaN`` a pandas round-trip
    through a nullable column produces, and the candidate then reaches the mask,
    fails to match anything, and is classified **minable** rather than refused —
    a fail-open direction (P2-10b; :mod:`tbox_finder.mining.hard_negative`).
    """
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


def accession_namespace_report(
    candidate_accessions: Iterable[Any], index: LocusIndex
) -> dict[str, Any]:
    """Measure whether a pool's accessions can address the mask at all.

    Returns the intersection of the pool's distinct accessions with the index
    keys **both as-is and version-normalised**. This is the cheap measurement
    that distinguishes the two indistinguishable readings of a zero mask count —
    "this pool genuinely overlaps no known locus" from "this pool's accessions
    are in a different namespace and could never have matched" (P2-10b measured
    the latter: 0 as-is versus 269 normalised, on the Rfam pool).

    ``namespace_compatible`` is the gateable clause: it is ``False`` when the
    pool has accessions but **none** of them addresses an index key even after
    normalisation, i.e. the mask is structurally incapable of firing.
    """
    raw = [a for a in candidate_accessions if not is_missing(a)]
    distinct_raw = {str(a) for a in raw}
    distinct_norm = {normalize_accession(a) for a in raw}
    hit_raw = distinct_raw & index.raw_accession_keys
    hit_norm = distinct_norm & index.accession_keys
    return {
        "n_with_accession": len(raw),
        "n_distinct_accessions": len(distinct_raw),
        # Published beside the raw count because the intersection below is
        # computed on normalised keys: comparing 11,447 hits against 11,453 raw
        # accessions reads as a 6-accession shortfall when it is really 6
        # version-collapse groups and coverage is 100 %.
        "n_distinct_normalized": len(distinct_norm),
        "n_unaddressable_normalized": len(distinct_norm - index.accession_keys),
        "n_index_accessions": index.n_accessions,
        "n_intersect_as_is": len(hit_raw),
        "n_intersect_normalized": len(hit_norm),
        "normalization_recovered": len(hit_norm) - len(hit_raw),
        "namespace_compatible": bool(distinct_raw) and bool(hit_norm),
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
