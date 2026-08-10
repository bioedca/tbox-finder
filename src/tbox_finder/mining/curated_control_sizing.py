"""Size the matched de-novo positive control criterion (b)'s parameters need (P3-15'-g).

``reports/p3/architecture_parameter_measurement.json`` (P3-15'-f) swept criterion
(b)'s seven rule parameters over 278 **de-novo** consensuses — every one of them a
round-0 false-positive-manifest candidate of *unknown* status.  It therefore has no
positives in it, and the §7 decision of 2026-08-09 declined to pin (b)'s thresholds
on the repo's only de-novo positive control (``certified_positive.sto``, **n = 1**)
plus criterion (a) as a proxy.  The powered instrument is the same
``nhmmer`` + ``mlocarna`` pipeline run on a sample of **curated positives**.

This module does **not** run that pipeline.  It answers the three questions
``imp.md`` marks ``UNKNOWN`` before any sample may be drawn or any job submitted:

1. **Is the supply there at all** — can a curated held-out record be turned into a
   query the shipped searcher accepts, and by which route?
2. **Would such a control be *matched*** — the FP candidates the sweep read were
   Stage-1 spans resolved from genome coordinates; a curated record is a whole
   curated element.  If the two query populations differ, the control measures a
   different instrument and the comparison is confounded.  That is measured here,
   not assumed either way.
3. **What would it cost, and what would it resolve** — the SLURM envelope, and the
   confidence interval a pass *share* would carry at each sample size.

Disciplines carried over from P3-15'-f:

* **It calls the shipped gate, never a copy.**  "Is this sequence searchable" is
  :func:`tbox_finder.mining.homolog_msa.is_clean_nucleotide` — the same predicate
  ``search_homologs`` and ``resolve_candidate_sequence`` refuse on — reached through
  ``degap_to_dna``.  A second spelling of the alphabet rule would let this report
  and the producer disagree about which records are even runnable.
* **It pins nothing** (``pins_nothing: true``).  Sample design is a CLAUDE.md §7
  question; this module supplies its inputs and takes no decision.
* **Every projection names the corpus it was measured on.**  The only per-candidate
  wall measurements this repo owns come from FP candidates (job 816).  Using them
  to price curated queries is an extrapolation across exactly the population
  difference question 2 measures, so each projected number carries that caveat and
  a *bound* — the ``--align-timeout-s`` worst case — that does not depend on it.

Run::

    PYTHONPATH=src python -m tbox_finder.mining.curated_control_sizing size \\
        --k 50 --k 100 --k 200 --k 400 \\
        --shard-size 20 --align-timeout-s 600 \\
        --out reports/p3/curated_control_sizing.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder.mining.architecture_param_measure import (
    _sha256_of,
    is_inside_repo,
    portable_path,
)
from tbox_finder.mining.homolog_msa import (
    HomologMsaError,
    degap_to_dna,
    is_clean_nucleotide,
    read_stockholm_sequence,
)
from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.provenance import build_provenance

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-g"
ADR = (
    "ADR-0006 A2 (min_sequences floor) / A3 (mlocarna) / A4 (criterion (b)'s instrument), "
    "D3/D6/D7; ADR-0005 A10 (the covariation-producer envelope), D14 (unavailable spares)"
)

#: Percentiles reported for every length distribution.
PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)

#: Default inputs — all committed or DVC-tracked in this repo.
DEFAULT_CORPUS = "data/processed/master_clean_v0.parquet"
DEFAULT_SPLIT_TABLE = "data/processed/splits/split_assignments.parquet"
DEFAULT_CONTEXT = "data/interim/flank_context/context_v0.parquet"
DEFAULT_FP_MANIFEST = "data/processed/mining/round0_fp_manifest.json"
DEFAULT_HOST_MANIFEST = "data/processed/mining/production_host_orders_v0.parquet"
DEFAULT_MEASURE_REPORT = "reports/p2/mine_round_measure.json"
#: The producing env: this module reads parquet, so it runs under `envs/data.yml`.
ENV_LOCK = "envs/data.conda-lock.yml"
#: Row 0 of this fixture is what SLURM job 766 seeded `certified_positive.sto` from.
DEFAULT_CONTROL_SEED = "tests/fixtures/rscape/classII_sub.sto"

#: `slurm/p2/mine_round_producer.sbatch` pins `--cpus-per-task=2` (ADR-0005 A10 Pin 2).
PRODUCER_CPUS_PER_TASK = 2

#: The producibility anchors a K must survive.  Both are MEASURED, both are on the FP
#: corpus, and they disagree by 22 points — which is the whole reason they are carried
#: as a pair rather than averaged into one number.
PRODUCIBILITY_ANCHORS: tuple[tuple[str, float], ...] = (
    ("job816_k50_fp_sample", 26 / 50),
    ("job1205_1254_full_fp_corpus", 278 / 941),
    ("optimistic_all_producible", 1.0),
)


class SizingError(ValueError):
    """The sizing could not be measured from the inputs as given."""


# ═════════════════════════════════════════════════════════════════════════════
# Small statistics, spelled out (no scipy in the data env's hot path)
# ═════════════════════════════════════════════════════════════════════════════
def percentiles(values: Sequence[float], pcts: Sequence[int] = PERCENTILES) -> dict[str, float]:
    """Nearest-rank percentiles plus n/min/max/mean, or an explicit empty summary.

    Nearest-rank (not interpolated) so every reported value is a value the corpus
    actually contains — a length of 279.5 nt would be a number no record has.
    """
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    n = len(ordered)
    out: dict[str, float] = {
        "n": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
    }
    for pct in pcts:
        # Nearest-rank: ceil(p/100 * n), 1-indexed, clamped into range.
        rank = max(1, math.ceil(pct / 100 * n))
        out[f"p{pct:02d}"] = ordered[min(rank, n) - 1]
    return out


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov–Smirnov D — the largest gap between the two ECDFs.

    Reported as a single scale-free number for "do these two query populations
    overlap"; D = 1 means the supports are disjoint, D = 0 means the ECDFs coincide.
    No p-value is computed: with n in the thousands any real difference is
    "significant", and the decision needs the *size* of the difference.
    """
    if not a or not b:
        raise SizingError("KS needs two non-empty samples")
    sa, sb = sorted(a), sorted(b)
    na, nb = len(sa), len(sb)
    d = 0.0
    ia = ib = 0
    while ia < na and ib < nb:
        value = min(sa[ia], sb[ib])
        while ia < na and sa[ia] <= value:
            ia += 1
        while ib < nb and sb[ib] <= value:
            ib += 1
        d = max(d, abs(ia / na - ib / nb))
    return round(d, 4)


def wilson_interval(successes: float, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (the CI a pass *share* carries).

    Wilson rather than Wald: at the small n this control can afford, and at the
    extreme shares (b) plausibly produces, the Wald interval runs past 0/1 and
    understates the width — the two failure modes that would make a sample look
    more decisive than it is.
    """
    if n <= 0:
        raise SizingError("a confidence interval needs n > 0")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_width(n: int, p: float = 0.5, *, z: float = 1.96) -> float:
    """Width of the 95 % Wilson interval at share ``p`` and sample size ``n``."""
    lo, hi = wilson_interval(p * n, n, z=z)
    return round(hi - lo, 4)


def expected_distinct_strata(sizes: Sequence[int], k: int) -> float:
    """E[# strata hit] by a UNIFORM draw of ``k`` records without replacement.

    Written as a running product rather than binomial coefficients: the exact
    ``comb`` form on N ≈ 8,700 produces 800-digit integers, and the product form is
    both cheaper and free of the intermediate overflow a float ``comb`` ratio would
    hit.  This is the number that makes the skew concrete — a uniform draw on a
    corpus that is 81 % one phylum reaches few strata, however large K is
    ([[uniform-cluster-draw-collapses-on-skew]]).
    """
    total = sum(sizes)
    if k < 0 or k > total:
        raise SizingError(f"cannot draw k={k} from {total} records")
    expected = 0.0
    for size in sizes:
        # P(stratum missed) = prod_{i<k} (total - size - i) / (total - i); 0 once the
        # complement is exhausted, which the loop reaches naturally via a 0 factor.
        p_missed = 1.0
        for i in range(k):
            numerator = total - size - i
            if numerator <= 0:
                p_missed = 0.0
                break
            p_missed *= numerator / (total - i)
        expected += 1 - p_missed
    return round(expected, 2)


# ═════════════════════════════════════════════════════════════════════════════
# The frame and its query supply
# ═════════════════════════════════════════════════════════════════════════════
#: The keys :func:`load_frame` puts on every record.  Named once so the loader and
#: the pure functions below cannot drift.
RECORD_KEYS: tuple[str, ...] = (
    "record_sha256",
    "context_status",
    "locus_length",
    "locus_seq",
    "tbox_type",
    "phylum",
    "order",
    "genus",
    "cluster_id",
    "host_taxid",
)


def load_frame(
    *,
    corpus: str | Path = DEFAULT_CORPUS,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    context: str | Path = DEFAULT_CONTEXT,
) -> list[dict[str, Any]]:
    """The held-out curated carve, joined to its genomic context, as plain records.

    The carve itself is :func:`architecture_producer.heldout_canonical` — the same
    ``source == 'corpus' ∧ nested_role == 'heldout'`` content-hash join
    ``reports/p3/architecture_freeze.json`` was measured on, reused rather than
    re-derived so the control's frame and the curated half of the P3-15'-f
    comparison are the same 8,715 records by construction.

    The locus nucleotides come from ``flank_context/context_v0.parquet``
    (``context_seq[locus_offset : locus_offset + locus_length]``), which is the
    NCBI-anchored genomic context P2-00 fetched — **not** the TBDB ``Name``
    coordinates, which agree with the record's own length on only 61.7 % of rows.

    Returns plain dicts, not a frame: every function below this one is pandas-free
    so the unit tier can exercise it without the DVC artifacts.
    """
    import pandas as pd  # lazy: CI's unit tier has pandas, but nothing else here needs it

    from tbox_finder.mining.architecture_producer import heldout_canonical

    carved = heldout_canonical(corpus=corpus, split_table=split_table)
    ctx = pd.read_parquet(context)
    splits = pd.read_parquet(
        split_table,
        columns=[
            "corpus_record_sha256",
            "cluster_id",
            "resolved_phylum",
            "resolved_order",
            "resolved_genus",
        ],
    ).drop_duplicates("corpus_record_sha256")

    joined = carved.merge(
        ctx, left_on="_record_sha256", right_on="record_id", how="left", suffixes=("", "_ctx")
    ).merge(splits, left_on="_record_sha256", right_on="corpus_record_sha256", how="left")
    if len(joined) != len(carved):
        raise SizingError(
            f"the context join changed the row count ({len(carved)} -> {len(joined)}); "
            "flank_context.record_id is not unique per curated record"
        )

    return records_from_joined(joined.to_dict("records"))


#: The columns :func:`records_from_joined` indexes.  A join that silently loses one
#: of these would otherwise surface as a bare ``KeyError`` traceback (exit 1) instead
#: of this module's exit-3 refusal — and a renamed upstream column is exactly the
#: kind of drift a sizing report must refuse rather than fill with ``None``.
JOINED_REQUIRED_COLUMNS: tuple[str, ...] = (
    "_record_sha256",
    "status",
    "context_seq",
    "locus_offset",
    "locus_length",
    "type",
    "TaxId",
    "cluster_id",
    "resolved_phylum",
    "resolved_order",
    "resolved_genus",
)


def records_from_joined(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Shape joined rows into the plain records the pure functions consume.

    Split out of :func:`load_frame` so the shaping — including its refusal on a
    missing column — is testable without the DVC artifacts.  The signature takes any
    sequence of mappings, so the column check runs per row rather than on row 0: for
    a frame every row carries the same keys and the check is a schema check, but for
    the hand-built rows the tests (and any future caller) pass, row 0 says nothing
    about row 5.

    Missing *values* are normalised to ``None`` here, at the pandas boundary: a
    float ``nan`` is not ``None``, so an un-normalised absent genus would survive
    every ``is not None`` filter downstream and be counted as a stratum named
    ``nan`` ([[nulls-inflate-block-counts]]).
    """
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        missing = [column for column in JOINED_REQUIRED_COLUMNS if column not in row]
        if missing:
            raise SizingError(
                f"joined row {i} is missing required column(s) {', '.join(missing)}; "
                "the corpus / context / split tables are not the expected schema"
            )
        seq = row["context_seq"]
        offset, length = row["locus_offset"], row["locus_length"]
        locus_seq: str | None = None
        # ⚠ The bounds are checked BEFORE the slice, not inferred from its length
        # afterwards. Python reads a negative start as an offset from the end, so a
        # negative `locus_offset` with enough tail left yields a slice of exactly
        # `locus_length` characters — the wrong locus, at the right length, passing
        # `query_supply`'s truncation check and going on to be searched as though it
        # were this record. `>= 0` is the guard that check cannot supply.
        if (
            isinstance(seq, str)
            and _finite(offset)
            and _finite(length)
            and int(offset) >= 0
            and int(length) > 0
            and int(offset) + int(length) <= len(seq)
        ):
            locus_seq = seq[int(offset) : int(offset) + int(length)]
        records.append(
            {
                "record_sha256": row["_record_sha256"],
                "context_status": _or_none(row["status"]),
                "locus_length": int(length) if _finite(length) else None,
                "locus_seq": locus_seq,
                "tbox_type": _or_none(row["type"]),
                "phylum": _or_none(row["resolved_phylum"]),
                "order": _or_none(row["resolved_order"]),
                "genus": _or_none(row["resolved_genus"]),
                "cluster_id": int(row["cluster_id"]) if _finite(row["cluster_id"]) else None,
                "host_taxid": int(row["TaxId"]) if _finite(row["TaxId"]) else None,
            }
        )
    return records


def _or_none(value: Any) -> Any:
    """``None`` for a missing value — including pandas' float ``nan``, which is not ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _finite(value: Any) -> bool:
    """Is ``value`` a usable number?  ``None``/NaN are both 'no' — pandas emits both."""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def query_supply(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which curated records could be handed to the searcher, and why the rest could not.

    The verdict is the shipped one: ``degap_to_dna`` then
    :func:`~tbox_finder.mining.homolog_msa.is_clean_nucleotide`, which is exactly
    what ``search_homologs`` raises ``HomologMsaError`` on.  A record refused here
    would abort its own shard task, so the refusals are part of the supply, not a
    footnote.
    """
    reasons: Counter[str] = Counter()
    usable_lengths: list[int] = []
    usable: list[dict[str, Any]] = []
    for row in records:
        status = row.get("context_status")
        seq = row.get("locus_seq")
        declared = row.get("locus_length")
        if status != "ok":
            reasons[f"context_status:{status}"] += 1
            continue
        if not isinstance(seq, str) or not seq:
            reasons["no_locus_sequence"] += 1
            continue
        if declared is None or len(seq) != int(declared):
            # The context window is clipped at a replicon end for some records, so the
            # slice can be short: a truncated T-box is a different query, not a query.
            reasons["locus_truncated_by_context_clip"] += 1
            continue
        dna = degap_to_dna(seq)
        if not is_clean_nucleotide(dna):
            reasons["ambiguous_alphabet"] += 1
            continue
        usable_lengths.append(len(dna))
        usable.append(dict(row))
    return {
        "gate": (
            "homolog_msa.degap_to_dna + homolog_msa.is_clean_nucleotide — the same "
            "refusal search_homologs raises on a query it cannot search"
        ),
        "n_records": len(records),
        "n_usable": len(usable),
        "n_refused": len(records) - len(usable),
        "refusal_reasons": dict(sorted(reasons.items())),
        "usable_length_nt": percentiles(usable_lengths),
        "_usable": usable,
    }


def span_matchedness(
    curated_lengths: Sequence[int],
    fp_lengths: Sequence[int],
) -> dict[str, Any]:
    """Is a curated query the same *kind of object* as the FP queries the sweep read?

    The sweep's 278 consensuses came from Stage-1 detected spans resolved out of
    genome coordinates; a curated record is a whole curated element.  Query length
    drives ``nhmmer`` sensitivity, the ``min_cov = 0.5`` inclusion filter and hence
    homolog depth, which is what ``mlocarna`` and the localizer see — so if these
    two distributions differ, a control built from raw curated elements measures the
    instrument on an easier (or harder) input than the corpus it is meant to
    calibrate, and the difference is a confound rather than a result.
    """
    if not curated_lengths or not fp_lengths:
        raise SizingError("matchedness needs both populations")
    # ⚠ Guarded here as well as in `fp_span_lengths`: this function is public and takes
    # two plain sequences, so it cannot assume its caller came through that reader. A
    # zero median would otherwise raise ZeroDivisionError, which `main` does not catch
    # — a traceback where every other malformed input exits 3 with a refusal.
    fp_median = statistics.median(fp_lengths)
    curated_median = statistics.median(curated_lengths)
    if fp_median <= 0 or curated_median <= 0:
        raise SizingError(
            f"a median query length is not positive (curated {curated_median}, "
            f"FP {fp_median}); these spans cannot be compared"
        )
    lo, hi = min(fp_lengths), max(fp_lengths)
    inside = sum(1 for value in curated_lengths if lo <= value <= hi)
    return {
        "curated_length_nt": percentiles(curated_lengths),
        "fp_length_nt": percentiles(fp_lengths),
        "fp_span_range_nt": [lo, hi],
        "n_curated_inside_fp_range": inside,
        "share_curated_inside_fp_range": round(inside / len(curated_lengths), 4),
        "ks_d": ks_statistic(curated_lengths, fp_lengths),
        "median_ratio_curated_over_fp": round(curated_median / fp_median, 3),
    }


def substrate_overlap(
    records: Sequence[Mapping[str, Any]],
    rep_taxids: Sequence[int],
    *,
    fp_assemblies: Sequence[str],
    rep_assemblies: Sequence[str],
) -> dict[str, Any]:
    """Could the control instead reuse ``covariation_producer search-shard`` verbatim?

    That route resolves a query from ``data/interim/production_genomes/<assembly>.fna``
    at contig coordinates, so it can only be handed a record whose host genome is one
    of the ADR-0006 A1 2,500 GTDB R232 species representatives.  The repo holds no
    curated-record → assembly join at all, so this is the *upper bound* on one: a
    record whose host **species** (NCBI taxid) is not represented cannot become a
    coordinate row however the join is built.  A taxid that *is* represented still
    needs the join, so this number is a ceiling, never a yield — and relaxing the
    match below species does not raise it, because a *different* strain's assembly
    does not contain the curated record's own locus to resolve coordinates into.

    The same overlap measures a second, independent asymmetry: the FP candidates the
    sweep read were carved out of genomes that **are** in the searched database, so
    every FP query self-hits; a curated query mostly does not.  That is a one-sequence
    difference in homolog depth, which matters only against ADR-0006 A2's
    ``min_sequences`` floor — recorded so the comparison carries it rather than
    discovers it.
    """
    reps = {int(t) for t in rep_taxids}
    curated_taxids = [row["host_taxid"] for row in records if row.get("host_taxid") is not None]
    hit = [t for t in curated_taxids if t in reps]
    fp_set, rep_set = set(fp_assemblies), set(rep_assemblies)
    return {
        # ⚠ Derived from the SAME set the self-hit intersection uses, not from a
        # separately-passed count: a manifest with two rows for one representative
        # would otherwise publish a genome count larger than the population the
        # intersection below was measured against.
        "n_production_genomes": len(rep_set),
        "n_rep_taxids": len(reps),
        "n_records_with_taxid": len(curated_taxids),
        "n_distinct_curated_taxids": len(set(curated_taxids)),
        "n_records_whose_host_taxid_is_a_rep": len(hit),
        "n_distinct_curated_taxids_in_reps": len(set(hit)),
        "share_records_coordinate_routable_ceiling": (
            round(len(hit) / len(records), 5) if records else 0.0
        ),
        "fp_self_hit": {
            "n_fp_assemblies": len(fp_set),
            "n_fp_assemblies_in_searched_db": len(fp_set & rep_set),
            "note": (
                "an FP query is carved from a genome that is itself in the searched "
                "database, so it self-hits; a curated query self-hits only for the "
                "records counted above"
            ),
        },
    }


def stratum_census(
    records: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int],
) -> dict[str, Any]:
    """What a draw can reach: the strata, the clusters, and the cost of drawing uniformly.

    Two orthogonal design constraints live here.  **Representativeness**: the frame
    is 81 % one phylum, so a uniform draw concentrates there and the pass share it
    measures is a Firmicutes pass share ([[uniform-cluster-draw-collapses-on-skew]]).
    **Pseudo-replication**: ``cluster_id`` is the ADR-0004 structure-aware
    single-linkage cluster, so two records in one cluster are near-identical
    sequences — they would produce near-identical homolog sets and near-identical
    consensuses, inflating n without adding information.
    """
    by: dict[str, Counter[Any]] = {}
    for key in ("phylum", "order", "genus", "tbox_type"):
        by[key] = Counter(row.get(key) for row in records if row.get(key) is not None)
    clusters = Counter(row["cluster_id"] for row in records if row.get("cluster_id") is not None)
    order_clusters: dict[Any, set[Any]] = {}
    for row in records:
        if row.get("order") is not None and row.get("cluster_id") is not None:
            order_clusters.setdefault(row["order"], set()).add(row["cluster_id"])

    order_sizes = sorted(by["order"].values(), reverse=True)
    phylum_sizes = sorted(by["phylum"].values(), reverse=True)
    draw = []
    for k in ks:
        per_order_quota = k // len(order_sizes) if order_sizes else 0
        draw.append(
            {
                "k": k,
                "expected_orders_hit_uniform": expected_distinct_strata(order_sizes, k),
                "orders_available": len(order_sizes),
                "expected_phyla_hit_uniform": expected_distinct_strata(phylum_sizes, k),
                "phyla_available": len(phylum_sizes),
                "equal_allocation_per_order": per_order_quota,
                "orders_with_enough_clusters_for_quota": sum(
                    1 for ids in order_clusters.values() if len(ids) >= max(1, per_order_quota)
                ),
                "one_per_cluster_feasible": k <= len(clusters),
            }
        )
    return {
        "n_records": len(records),
        "phylum": dict(by["phylum"].most_common()),
        "order": dict(by["order"].most_common()),
        "tbox_type": dict(by["tbox_type"].most_common()),
        "genus": {
            "n_distinct": len(by["genus"]),
            "n_records_without_genus": sum(1 for row in records if row.get("genus") is None),
            "n_singleton_genera": sum(1 for count in by["genus"].values() if count == 1),
            "top10": dict(by["genus"].most_common(10)),
        },
        "clusters": {
            "n_clusters": len(clusters),
            "n_records_without_cluster": sum(1 for row in records if row.get("cluster_id") is None),
            "n_singleton_clusters": sum(1 for count in clusters.values() if count == 1),
            "largest_cluster_size": max(clusters.values()) if clusters else 0,
            "n_orders_with_at_least_1_cluster": len(order_clusters),
            "n_orders_with_at_least_5_clusters": sum(
                1 for ids in order_clusters.values() if len(ids) >= 5
            ),
        },
        "draw": draw,
    }


def power_table(
    *,
    ks: Sequence[int],
    anchors: Sequence[tuple[str, float]] = PRODUCIBILITY_ANCHORS,
) -> list[dict[str, Any]]:
    """The CI a pass *share* would carry — after the producible fraction takes its cut.

    A record whose homolog set falls below ADR-0006 A2's ``min_sequences`` floor, or
    whose alignment hits the ``--align-timeout-s`` bound, resolves ``unavailable`` and
    leaves the **denominator** — so the sample that decides (b)'s parameters is not K,
    it is K × (producible share).  Both anchors are measured on the FP corpus; the
    curated share is unmeasured, and no value is asserted for it here.
    """
    rows = []
    for k in ks:
        for label, share in anchors:
            effective = int(round(k * share))
            rows.append(
                {
                    "k": k,
                    "producibility_anchor": label,
                    "producible_share": round(share, 4),
                    "effective_n": effective,
                    "wilson_width_95_at_p50": (wilson_width(effective) if effective > 0 else None),
                    "wilson_width_95_at_p90": (
                        wilson_width(effective, 0.9) if effective > 0 else None
                    ),
                }
            )
    return rows


def compute_envelope(
    *,
    measure_report: Mapping[str, Any],
    ks: Sequence[int],
    shard_size: int,
    align_timeout_s: float,
) -> dict[str, Any]:
    """The SLURM envelope, expected and bounded.

    ``expected`` extrapolates job 816's measured per-candidate wall — which was
    measured on **FP** candidates, i.e. on exactly the query population
    :func:`span_matchedness` compares against, so it is an extrapolation across the
    difference this report exists to measure.  ``worst_case`` does not depend on it:
    ``--align-timeout-s`` bounds every alignment by construction, so
    ``search_max + align_timeout_s + score_max`` bounds a candidate whatever its
    homolog depth turns out to be.  A submit should be sized on the bound; the
    expected column is only there to say what is likely.
    """
    if shard_size <= 0:
        raise SizingError(f"shard_size must be positive, got {shard_size}")
    if not math.isfinite(align_timeout_s) or align_timeout_s <= 0:
        # ⚠ isfinite as well as > 0: `float('nan')` compares False to every bound, so a
        # naive `<= 0` guard admits it and the "bound" is recorded and inert.
        raise SizingError(
            f"align_timeout_s must be a positive, finite number of seconds, got {align_timeout_s!r}"
        )
    wall = measure_report.get("wall_seconds")
    if not isinstance(wall, Mapping):
        raise SizingError("measure report carries no wall_seconds block")
    try:
        per_candidate_mean = float(wall["total_per_candidate"]["mean"])
        search_max = float(wall["search"]["max"])
        score_max = float(wall["score"]["max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SizingError(
            f"measure report's wall_seconds block is not the expected shape: {exc}"
        ) from exc

    worst_per_candidate = search_max + align_timeout_s + score_max
    per_k = []
    for k in ks:
        array_width = math.ceil(k / shard_size)
        # The last shard may be short; a task's wall is set by a FULL shard.
        expected_task_h = shard_size * per_candidate_mean / 3600
        worst_task_h = shard_size * worst_per_candidate / 3600
        per_k.append(
            {
                "k": k,
                "shard_size": shard_size,
                "array_width": array_width,
                "expected_wall_h_per_task": round(expected_task_h, 3),
                "worst_case_wall_h_per_task": round(worst_task_h, 3),
                "expected_core_h": round(k * per_candidate_mean / 3600 * PRODUCER_CPUS_PER_TASK, 3),
                "worst_case_core_h": round(array_width * worst_task_h * PRODUCER_CPUS_PER_TASK, 2),
            }
        )
    return {
        "measured_from": {
            "k_sample": measure_report.get("k_sample"),
            "corpus": "round-0 FP candidates (job 816)",
            "per_candidate_wall_s": dict(wall["total_per_candidate"]),
            "homolog_depth": dict(measure_report.get("homolog_depth", {})),
            "status_counts": dict(measure_report.get("status_counts", {})),
        },
        "align_timeout_s": align_timeout_s,
        "cpus_per_task": PRODUCER_CPUS_PER_TASK,
        "worst_case_per_candidate_s": round(worst_per_candidate, 2),
        "caveat": (
            "the expected_* columns extrapolate FP-candidate wall onto curated queries, "
            "which this report measures to be a different query population, so they "
            "inherit that difference; the worst_case_* columns do not — they are "
            "search_max + align_timeout_s + score_max, which bounds a candidate at any "
            "homolog depth, so a submit should be sized on those"
        ),
        "per_k": per_k,
    }


# ═════════════════════════════════════════════════════════════════════════════
# The report
# ═════════════════════════════════════════════════════════════════════════════
def _fp_rows(manifest: Mapping[str, Any] | Sequence[Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("candidates") if isinstance(manifest, Mapping) else manifest
    if not isinstance(rows, list) or not rows:
        raise SizingError("FP manifest carries no candidate rows")
    for row in rows:
        if not isinstance(row, Mapping):
            raise SizingError("an FP manifest row is not an object")
    return rows


def fp_span_lengths(manifest: Mapping[str, Any] | Sequence[Any]) -> list[int]:
    """The Stage-1 span lengths of the FP manifest the P3-15'-f sweep was read on."""
    lengths = []
    for row in _fp_rows(manifest):
        try:
            span = int(row["locus_end"]) - int(row["locus_start"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SizingError(f"an FP manifest row has no usable locus span: {exc}") from exc
        # A candidate with end <= start is not a short query, it is a broken row: it
        # would drag the minimum (and possibly the median) of the population this
        # report compares against, and every matchedness number reads off that.
        if span <= 0:
            raise SizingError(
                f"an FP manifest row has a non-positive locus span ({span} nt): "
                f"{row.get('candidate_id') or row.get('accession')}"
            )
        lengths.append(span)
    return lengths


def fp_assemblies(manifest: Mapping[str, Any] | Sequence[Any]) -> list[str]:
    """The assembly accessions the FP candidates were carved from.

    ``accession`` is contig-scoped (``<assembly>:c<contig_index>``,
    ``homolog_db.target_seq_name``), so the assembly is everything before the final
    ``:c`` — split from the right, since an assembly accession carries no ``:c``.
    """
    out = []
    for row in _fp_rows(manifest):
        accession = row.get("accession")
        if not isinstance(accession, str) or ":c" not in accession:
            raise SizingError(f"an FP manifest row has no contig-scoped accession: {accession!r}")
        out.append(accession.rsplit(":c", 1)[0])
    return out


def existing_control_placement(
    records: Sequence[Mapping[str, Any]],
    seed_record_id: str | None,
) -> dict[str, Any]:
    """Where the repo's only de-novo positive control sits relative to *this* frame.

    ``certified_positive.sto`` (SLURM job 766) was aligned from row 0 of
    ``tests/fixtures/rscape/classII_sub.sto``.  Whether that record is inside the
    held-out carve decides whether a control drawn here would re-measure it or
    extend it — and a control that silently re-used its own precedent would report
    n = K while carrying K − 1 independent observations plus one already spent.
    """
    if seed_record_id is None:
        return {
            "seed_record_id": None,
            "in_this_frame": None,
            "note": "no seed alignment was supplied, so the placement is unmeasured",
        }
    in_frame = seed_record_id in {row.get("record_sha256") for row in records}
    return {
        "artifact": "data/interim/homolog_msa/certified_positive.sto (SLURM job 766, n = 1)",
        "seed_record_id": seed_record_id,
        "in_this_frame": in_frame,
        "note": (
            "the existing control's seed IS inside the held-out carve, so a draw from "
            "this frame can re-select it"
            if in_frame
            else "the existing control's seed is NOT in the held-out carve — it comes "
            "from the non-heldout side of the ADR-0004 split, so a draw from this "
            "frame adds to it rather than re-measuring it"
        ),
    }


def size_report(
    *,
    records: Sequence[Mapping[str, Any]],
    fp_manifest: Mapping[str, Any] | Sequence[Any],
    rep_taxids: Sequence[int],
    rep_assemblies: Sequence[str],
    measure_report: Mapping[str, Any],
    ks: Sequence[int],
    shard_size: int,
    align_timeout_s: float,
    seed_record_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the whole sizing report body (no provenance — ``main`` adds it)."""
    if not records:
        raise SizingError("the held-out curated frame is empty; there is nothing to size")
    fp_lengths = fp_span_lengths(fp_manifest)
    supply = query_supply(records)
    usable = supply.pop("_usable")
    if not usable:
        raise SizingError(
            "no curated record survived the query gate; a control cannot be drawn from this frame"
        )
    usable_lengths = [len(degap_to_dna(row["locus_seq"])) for row in usable]
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "pins_nothing": True,
        "frame": {
            "definition": "source == 'corpus' AND nested_role == 'heldout'",
            "join": (
                "ingest.record_hash(master_clean row) == split.corpus_record_sha256 "
                "== flank_context.record_id"
            ),
            "carve": "mining.architecture_producer.heldout_canonical (reused, not re-derived)",
            "n_records": len(records),
            "context_status": dict(
                sorted(Counter(str(row.get("context_status")) for row in records).items())
            ),
            "locus_source": (
                "flank_context context_seq[locus_offset : locus_offset + locus_length] — the "
                "NCBI-anchored genomic context, never the TBDB Name coordinates"
            ),
        },
        "query_supply": supply,
        "existing_positive_control": existing_control_placement(records, seed_record_id),
        "matchedness": span_matchedness(usable_lengths, fp_lengths),
        "substrate": {
            **substrate_overlap(
                records,
                rep_taxids,
                fp_assemblies=fp_assemblies(fp_manifest),
                rep_assemblies=rep_assemblies,
            ),
            "coordinate_route": (
                "covariation_producer search-shard resolves a query from "
                "data/interim/production_genomes/<assembly>.fna at contig coordinates, so it "
                "needs a curated-record -> GTDB-rep-assembly join the repo does not hold"
            ),
            "sequence_route": (
                "homolog_msa seed/search/align takes a sequence directly — the route that "
                "produced the repo's only de-novo positive control (certified_positive.sto, "
                "SLURM job 766) — and needs no genome join"
            ),
        },
        "strata": stratum_census(records, ks=ks),
        "power": {
            "min_sequences_floor": MIN_REAL_HOMOLOG_N,
            "note": (
                "a record below the ADR-0006 A2 min_sequences floor, or whose alignment hits "
                "--align-timeout-s, resolves unavailable and leaves the DENOMINATOR; the "
                "producible share of curated queries is UNMEASURED and no value is asserted"
            ),
            "rows": power_table(ks=ks),
        },
        "envelope": compute_envelope(
            measure_report=measure_report,
            ks=ks,
            shard_size=shard_size,
            align_timeout_s=align_timeout_s,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curated_control_sizing",
        description=(
            "Size the matched de-novo positive control (P3-15'-g): query supply, "
            "matchedness against the FP query population, stratification frame, "
            "statistical power, and the SLURM envelope."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("size", help="write the sizing report")
    s.add_argument(
        "--k",
        type=int,
        action="append",
        required=True,
        help="a candidate sample size (repeatable); the report tabulates one row per K",
    )
    s.add_argument(
        "--shard-size",
        type=int,
        required=True,
        help="candidates per SLURM array task (the envelope's per-task unit)",
    )
    s.add_argument(
        "--align-timeout-s",
        type=float,
        required=True,
        help="the per-alignment mlocarna bound the run would carry (P3-15'-e-ii measured 600)",
    )
    s.add_argument("--corpus", default=DEFAULT_CORPUS)
    s.add_argument("--split-table", default=DEFAULT_SPLIT_TABLE)
    s.add_argument("--context", default=DEFAULT_CONTEXT)
    s.add_argument("--fp-manifest", default=DEFAULT_FP_MANIFEST)
    s.add_argument("--host-manifest", default=DEFAULT_HOST_MANIFEST)
    s.add_argument("--measure-report", default=DEFAULT_MEASURE_REPORT)
    s.add_argument(
        "--control-seed-sto",
        default=DEFAULT_CONTROL_SEED,
        help=(
            "the Stockholm alignment row 0 of which seeded certified_positive.sto "
            "(job 766); used only to place the existing n=1 control against this frame"
        ),
    )
    s.add_argument("--out", default="reports/p3/curated_control_sizing.json")
    return parser


def _load_rep_columns(path: str | Path) -> tuple[list[int], list[str]]:
    """The taxids + assembly accessions of the ADR-0006 A1 2,500 GTDB R232 reps."""
    import pandas as pd

    frame = pd.read_parquet(path, columns=["ncbi_taxid", "assembly_accession"])
    taxids = pd.to_numeric(frame["ncbi_taxid"], errors="coerce").dropna().astype(int)
    # ⚠ Deduplicated here, once: `n_production_genomes` is len(assemblies) while the
    # self-hit count intersects a SET of them, so a manifest with two rows for one
    # representative would publish a genome count larger than the population the
    # other number was measured on.
    assemblies = frame["assembly_accession"].dropna().astype(str).drop_duplicates()
    return [int(v) for v in taxids], [str(v) for v in assemblies]


def partition_inputs(
    labelled: Sequence[tuple[str, str | Path | None]],
) -> tuple[list[str], dict[str, Any]]:
    """Split the inputs into repo-relative paths and hashed external entries.

    ⚠ This repo is **public**: an input staged outside the checkout is recorded by
    basename + sha256, never by its absolute path ([[verify-the-line-you-ship]]).
    The hash is the load-bearing half — a basename alone is a provenance entry a
    reader cannot check anything against.

    Extracted from :func:`main` because the external branch is unreachable from the
    committed report (every default input is inside the repo), so a test that reads
    that report cannot exercise it — it would pass with the hashing deleted
    ([[artifact-pinning-test-cannot-see-the-code]]).
    """
    repo_inputs: list[str] = []
    external: dict[str, Any] = {}
    for label, candidate in labelled:
        if not candidate or not Path(candidate).is_file():
            continue
        if is_inside_repo(candidate):
            repo_inputs.append(portable_path(candidate))
        else:
            external[label] = {"name": Path(candidate).name, "sha256": _sha256_of(candidate)}
    return repo_inputs, external


def _seed_record_id(path: str | Path | None) -> str | None:
    """Row 0's record name from the seed alignment, or ``None`` if it is not there."""
    if not path or not Path(path).is_file():
        return None
    name, _ = read_stockholm_sequence(path, index=0)
    return name


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "size":  # pragma: no cover - argparse enforces
        raise SizingError(f"unknown command {args.command!r}")
    try:
        ks = sorted({int(k) for k in args.k})
        if any(k <= 0 for k in ks):
            raise SizingError(f"every --k must be positive, got {sorted(args.k)}")
        rep_taxids, rep_assemblies = _load_rep_columns(args.host_manifest)
        body = size_report(
            records=load_frame(
                corpus=args.corpus, split_table=args.split_table, context=args.context
            ),
            fp_manifest=json.loads(Path(args.fp_manifest).read_text(encoding="utf-8")),
            rep_taxids=rep_taxids,
            rep_assemblies=rep_assemblies,
            measure_report=json.loads(Path(args.measure_report).read_text(encoding="utf-8")),
            ks=ks,
            shard_size=args.shard_size,
            align_timeout_s=args.align_timeout_s,
            seed_record_id=_seed_record_id(args.control_seed_sto),
        )
    # OSError/JSONDecodeError too: every input path is operator-supplied, so a missing
    # or malformed file exits 3 with a refusal rather than 1 with a traceback.
    # HomologMsaError too: it subclasses RuntimeError, so an empty or out-of-range
    # seed alignment would exit 1 with a traceback from `read_stockholm_sequence`.
    # KeyError is the backstop for a table whose schema drifted past
    # JOINED_REQUIRED_COLUMNS — pandas raises it from `read_parquet(columns=...)` and
    # from a merge key that is no longer there, both before this module can name it.
    # (`pyarrow.lib.ArrowInvalid` needs no separate arm: it subclasses ValueError.)
    except (SizingError, ValueError, OSError, KeyError, HomologMsaError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3

    repo_inputs, external = partition_inputs(
        (
            ("corpus", args.corpus),
            ("split_table", args.split_table),
            ("context", args.context),
            ("fp_manifest", args.fp_manifest),
            ("host_manifest", args.host_manifest),
            ("measure_report", args.measure_report),
            # The seed alignment decides `existing_positive_control.seed_record_id`, so
            # it is an input like any other; leaving it out let a reported record id
            # trace back to nothing.
            ("control_seed_sto", args.control_seed_sto),
        )
    )
    body["provenance"] = build_provenance(
        rule="P3-15'-g curated-control sizing",
        script=portable_path(__file__),
        inputs=repo_inputs,
        outputs=[],
        env_lock=ENV_LOCK,
        adr=ADR,
        extra={"schema_version": SCHEMA_VERSION, "external_inputs": external},
    )
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"refused: cannot write report to {portable_path(out)}: {exc}", file=sys.stderr)
        return 3
    print(
        f"sized {body['query_supply']['n_usable']} usable curated queries of "
        f"{body['frame']['n_records']} held-out records -> {out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
