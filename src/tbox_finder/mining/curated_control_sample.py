"""P3-15'-g-ii — the matched de-novo positive control's QUERY-CONSTRUCTION leg.

``reports/p3/architecture_parameter_measurement.json`` (P3-15'-f) measured what criterion
(b) resolves on 278 de-novo consensuses built from **941 round-0 false-positive candidates
of unknown status**.  It has no positives in it, so (b)'s seven rule parameters cannot be
chosen from it: the only de-novo positive control the repo owns is ``n = 1``.  P3-15'-g
sized the matched control that would fix this and measured the trap — a curated element is
median **279** nt against the FP candidates' **110** (KS *D* = 0.965), and query length
drives ``nhmmer`` sensitivity, the ``min_cov = 0.5`` filter, homolog depth, and therefore
what ``mlocarna`` and the localizer ever see.  Handing the pipeline raw curated elements
would measure the instrument on a **different input**: a confound, not a result.

The §7 sample-design decision (user, 2026-08-10) resolved that in three parts, and this
module executes all three:

1. **Query construction = Stage-1 re-detect.**  Each drawn record's own 1,024-nt genomic
   context window is scanned by the *same* checkpoint that produced the 941 FP candidates,
   at the *same* ADR-0005 A9 Pin-3 detection triple (τ = 0.9 / ``min_span`` = 50 /
   ``gap_merge`` = 10), and the predicted spans **are** the queries.  The control's query
   then comes from the identical instrument chain as the population it calibrates.
   Records where Stage-1 does not fire **drop out, and that dropout is part of the
   result** — it is reported, never silently excluded (a control that quietly loses its
   hardest records overstates the instrument).
2. **Sampling = one record per ADR-0004 cluster, order-stratified.**  The largest cluster
   holds 812 near-identical records, so one-per-cluster is what stops *n* from being
   pseudo-replicated.  ⚠ The decision's *"reaches 43/43 orders"* is **not achievable**, and
   this module measures why rather than repeating it: one record per cluster means one
   *order* per cluster, and that 812-record cluster spans **11 orders**, three of which
   occur nowhere else.  :func:`maximum_order_coverage` computes the exact ceiling — **41**
   of the 43 orders present — and equal allocation reaches all 41, against a uniform
   draw's expected 22.8 ([[uniform-cluster-draw-collapses-on-skew]]).
3. **K = 200.**

**The leakage question this leg had to answer, answered by measurement.**  The frame is
``source == 'corpus' ∧ nested_role == 'heldout'``; the scanning checkpoint
(``data/processed/checkpoints/stage1_production/stage1.pt``) records its own fold as the
full ADR-0004 D5 ``nested_train`` carve.  :func:`leakage_clauses` does not take that on
trust from either provenance file — it **re-derives** the disjointness from the committed
split table on every run, over record ids, sequence digests **and** ADR-0004 cluster ids
(D3's rule: a shared cluster is leakage even when no record is shared), and refuses the
draw if any clause fails.  Every clause guards its own emptiness, because a clause
evaluated over an empty ``nested_train`` set is vacuously true exactly when the evidence
is missing ([[clauses-must-guard-emptiness]]).

**Coordinate frame.**  A drawn record's context is *not* one of the 2,500 searched
genomes, so the emitted candidates are **not** in the FP manifest's assembly namespace.
They are in this record's own context frame: ``accession`` is ``<record_sha256>:c0`` and
``locus_start`` / ``locus_end`` are offsets into ``context_seq``.  The frame is named
rather than disguised — minting a plausible-looking NCBI coordinate here would be a
fabricated one on the minus strand, where ``context_seq`` is already reverse-complemented
and ``window_start + span`` runs the wrong way.  The true replicon accession, strand and
forward bounds ride along per record so any query is locatable in NCBI, and the query
nucleotides are emitted as a FASTA so the producer leg can take the **sequence route**
(``homolog_msa`` seed/search/align, the job-766 path) rather than the coordinate route,
whose ceiling on this frame P3-15'-g measured at 20 of 8,715 records.

This module **pins nothing** (``pins_nothing: true``).  It draws, scans, and reports; the
producer run and criterion (b)'s parameter choice are downstream and still open.

Run (both legs are LOCAL; the scan is ~200 windows and needs the ``tbox-ml-dna`` env)::

    PYTHONPATH=src python -m tbox_finder.mining.curated_control_sample draw --k 200
    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=src \\
      python -m tbox_finder.mining.curated_control_sample detect
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder.data.window_dataset import WINDOW_NT, deterministic_lead, window_lead_range
from tbox_finder.mining.architecture_param_measure import portable_path, sha256_of
from tbox_finder.mining.curated_control_sizing import (
    DEFAULT_CONTEXT,
    DEFAULT_CORPUS,
    DEFAULT_FP_MANIFEST,
    DEFAULT_SPLIT_TABLE,
    ENV_LOCK,
    SizingError,
    fp_span_lengths,
    load_joined_frame,
    partition_inputs,
    percentiles,
    query_supply,
    records_from_joined,
    span_matchedness,
)
from tbox_finder.mining.homolog_msa import HomologMsaError, degap_to_dna, is_clean_nucleotide
from tbox_finder.provenance import build_provenance

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-g-ii"
ADR = (
    "ADR-0004 D3 (cluster-crossing leakage) / D5 (the nested_train fold the scanning "
    "checkpoint was trained on); ADR-0005 A9 Pin 3 (the round-0 detection triple) / A7 "
    "(the cold-start round-0 checkpoint); ADR-0006 A2 (min_sequences floor) / A4"
)

#: The §7-decided sample size (user, 2026-08-10).  Not a pin — a CLI default that the
#: sizing report's power table justifies; ``--k`` overrides it.
DEFAULT_K = 200

#: The checkpoint that produced the 941 FP candidates (ADR-0005 A9 Pin 1's cold-start
#: round-0 production scanner).  Using *this* one is what makes the control matched, and
#: its ADR-0004 D5 ``nested_train`` fold is what makes scanning the held-out frame legal —
#: :func:`leakage_clauses` re-derives that second fact rather than trusting this comment.
DEFAULT_CHECKPOINT = "data/processed/checkpoints/stage1_production/stage1.pt"

#: Emitted beside ``round0_fp_manifest.json``: the control's manifest is the same kind of
#: object in the same shape, and lives in the same git-tracked directory.
DEFAULT_WINDOWS = "data/processed/mining/curated_control_windows_v0.jsonl"
DEFAULT_MANIFEST = "data/processed/mining/curated_control_manifest_v0.json"
DEFAULT_QUERY_FASTA = "data/processed/mining/curated_control_queries_v0.fasta"
DEFAULT_DRAW_REPORT = "reports/p3/curated_control_draw.json"
DEFAULT_DETECT_REPORT = "reports/p3/curated_control_detect.json"

#: P3-15'-g's committed sizing report — the raw-curated-element baseline this leg is
#: measured against, read at runtime rather than re-typed as a literal.
SIZING_REPORT = "reports/p3/curated_control_sizing.json"

#: The single "contig" of a curated record's context frame.  A record's context is one
#: fetched region of one replicon; the ``:c<int>:`` shape is reused so
#: ``mine_round.parse_window_name`` and ``covariation_producer.candidate_slug`` accept it
#: unchanged, and the index is always 0 because there is only ever one.
CONTROL_CONTIG_INDEX = 0

#: Columns :func:`emit_window` indexes on the wide joined row, beyond the ones
#: ``records_from_joined`` already refuses on.  A renamed upstream column must surface as
#: this module's exit-3 refusal, not a bare ``KeyError``.
WINDOW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "_record_sha256",
    "context_seq",
    "locus_offset",
    "locus_length",
    "clipped_start",
    "clipped_end",
    "accession",
    "strand",
    "region_start",
)

#: The leakage gate is a **versioned, enumerated** clause set; ``all_pass`` is over
#: exactly these keys, never over whatever happens to be present.  A missing key is a hard
#: failure, and adding a clause bumps the version so an older report re-validates FALSE
#: rather than silently skipping the new check ([[new-gate-clause-invalidates-old-reports]]).
CLAUSE_SCHEMA_VERSION = "1.0"
REQUIRED_LEAKAGE_CLAUSES: tuple[str, ...] = (
    "drawn_nonempty",
    "nested_train_nonempty",
    "every_drawn_record_resolved",
    "every_drawn_record_is_heldout",
    "no_shared_record_id",
    "no_shared_record_digest",
    "no_shared_cluster_id",
    "nested_train_flag_agrees_with_role",
)


class ControlSampleError(ValueError):
    """The control sample could not be drawn or detected from the inputs as given."""


# ═════════════════════════════════════════════════════════════════════════════
# Eligibility — the searcher's own gate, not a second spelling of it
# ═════════════════════════════════════════════════════════════════════════════
def eligible_records(
    joined: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Mapping[str, Any]]]:
    """``(usable records, supply summary, record_sha256 → wide row)``.

    The usable set is whatever :func:`curated_control_sizing.query_supply` accepts — the
    same call, on the same frame, that measured 8,709 of 8,715.  Re-implementing the
    eligibility test here would let the draw and the sizing disagree about which records
    exist while both reports still read clean.
    """
    narrow = records_from_joined(joined)
    supply = query_supply(narrow)
    usable = supply.pop("_usable")

    wide: dict[str, Mapping[str, Any]] = {}
    for i, row in enumerate(joined):
        missing = [column for column in WINDOW_REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ControlSampleError(
                f"joined row {i} is missing required column(s) {', '.join(missing)}; "
                "the corpus / context / split tables are not the expected schema"
            )
        key = str(row["_record_sha256"])
        if key in wide:
            raise ControlSampleError(
                f"record {key} appears twice in the joined frame — two rows sharing an id "
                "would silently overwrite each other and every summed count would still "
                "reconcile"
            )
        wide[key] = row
    return usable, supply, wide


# ═════════════════════════════════════════════════════════════════════════════
# The draw — one record per cluster, equal allocation across orders
# ═════════════════════════════════════════════════════════════════════════════
def cluster_order_options(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[Any, dict[str | None, dict[str, Any]]], dict[str, int]]:
    """``cluster_id → {order → that order's smallest-``record_sha256`` member}``.

    A record with no ``cluster_id`` is excluded and counted rather than treated as its own
    singleton cluster, which would admit an unresolved record under a rule that exists to
    bound near-identical duplicates.  ``None`` is a legitimate key: a cluster whose members
    all lack a resolved order still yields a representative, it just cannot fill an order
    quota.
    """
    options: dict[Any, dict[str | None, dict[str, Any]]] = {}
    exclusions: Counter[str] = Counter()
    for record in records:
        cluster = record.get("cluster_id")
        if cluster is None:
            exclusions["no_cluster_id"] += 1
            continue
        rid = str(record.get("record_sha256") or "")
        if not rid:
            exclusions["no_record_id"] += 1
            continue
        order = record.get("order")
        key = str(order) if order is not None else None
        bucket = options.setdefault(cluster, {})
        held = bucket.get(key)
        if held is None or rid < str(held["record_sha256"]):
            bucket[key] = dict(record)
    return options, dict(exclusions)


def maximum_order_coverage(
    options: Mapping[Any, Mapping[str | None, Mapping[str, Any]]],
) -> dict[str, Any]:
    """An **exact** maximum matching of orders to clusters (Kuhn's augmenting paths).

    One record per cluster means one *order* per cluster, so an order that lives only
    inside clusters other orders also claim can be squeezed out entirely — and which
    orders survive is otherwise decided by an arbitrary id tiebreak.  This computes the
    provable ceiling instead: the largest set of orders that can each be given a distinct
    cluster.  Iteration is over sorted names and sorted cluster ids, so the matching is
    deterministic (§8.3), and the size of the matching is the number this frame can
    actually stratify over — reported beside the number of orders merely *present*, since
    an order present only inside a shared cluster is not an independent stratum.
    """
    clusters_by_order: dict[str, list[Any]] = {}
    for cluster in sorted(options, key=str):
        for order in options[cluster]:
            if order is None:
                continue
            clusters_by_order.setdefault(order, []).append(cluster)

    matched_cluster_to_order: dict[Any, str] = {}

    def augment(order: str, seen: set[Any]) -> bool:
        for cluster in clusters_by_order[order]:
            if cluster in seen:
                continue
            seen.add(cluster)
            holder = matched_cluster_to_order.get(cluster)
            if holder is None or augment(holder, seen):
                matched_cluster_to_order[cluster] = order
                return True
        return False

    for order in sorted(clusters_by_order):
        augment(order, set())

    order_to_cluster = {order: cluster for cluster, order in matched_cluster_to_order.items()}
    unreachable = sorted(set(clusters_by_order) - set(order_to_cluster))
    return {
        "order_to_cluster": order_to_cluster,
        "n_orders_present": len(clusters_by_order),
        "n_orders_coverable": len(order_to_cluster),
        "unreachable_orders": {
            order: {
                "n_clusters": len(clusters_by_order[order]),
                "clusters": sorted(clusters_by_order[order], key=str)[:8],
                "why": (
                    "every cluster this order appears in is claimed by another order — its "
                    "records are in the same ADR-0004 near-identity cluster as records of "
                    "other orders, so counting it as an independent stratum would be the "
                    "pseudo-replication one-per-cluster exists to prevent"
                ),
            }
            for order in unreachable
        },
    }


def cluster_representatives(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict], dict[str, int], dict[str, Any]]:
    """One record per ADR-0004 cluster, chosen to maximise the orders represented.

    Deterministic and seedless: within an order a cluster's representative is its
    lexicographically smallest ``record_sha256``, the matching is computed over sorted
    names, and unmatched clusters fall back to their smallest order-bearing member (or
    their smallest member outright when no member carries an order).  A re-run reproduces
    the draw exactly from the id set alone (§8.3) — the convention
    ``covariation_producer.select_sample`` already uses.
    """
    options, exclusions = cluster_order_options(records)
    coverage = maximum_order_coverage(options)
    assigned: dict[Any, str] = {
        cluster: order for order, cluster in coverage["order_to_cluster"].items()
    }

    reps: list[dict[str, Any]] = []
    for cluster, bucket in options.items():
        order = assigned.get(cluster)
        if order is not None:
            reps.append(bucket[order])
            continue
        ordered_keys = sorted(k for k in bucket if k is not None)
        if ordered_keys:
            best = min((bucket[k] for k in ordered_keys), key=lambda r: str(r["record_sha256"]))
        else:
            best = bucket[None]
        reps.append(best)
    reps.sort(key=lambda r: str(r["record_sha256"]))
    return reps, dict(exclusions), coverage


def order_stratified_draw(records: Sequence[Mapping[str, Any]], *, k: int) -> dict[str, Any]:
    """Draw ``k`` cluster representatives with **equal allocation across orders**.

    Round-robin over the orders in name order: every order contributes its first
    representative before any order contributes a second.  That is what makes the draw
    reach every **coverable** order at K = 200 — 41 of the 43 present, the ceiling
    :func:`maximum_order_coverage` proves — where a uniform draw reaches an expected 22.8.
    The frame is 81 % Firmicutes and its largest cluster holds 812 records, so uniformity
    buys pseudo-replication, not representativeness.

    Deterministic and seedless: representatives are fixed by
    :func:`cluster_representatives`, orders are visited in sorted name order, and within
    an order representatives are taken in ``record_sha256`` order.
    """
    if k < 1:
        raise ControlSampleError(f"k must be >= 1, got {k}")
    reps, rep_exclusions, coverage = cluster_representatives(records)

    pools: dict[str, list[dict[str, Any]]] = {}
    exclusions = Counter(rep_exclusions)
    for rep in reps:
        order = rep.get("order")
        if order is None:
            exclusions["no_resolved_order"] += 1
            continue
        pools.setdefault(str(order), []).append(rep)
    for pool in pools.values():
        pool.sort(key=lambda r: str(r["record_sha256"]))

    available = sum(len(pool) for pool in pools.values())
    if available < k:
        raise ControlSampleError(
            f"cannot draw k={k}: only {available} cluster representatives carry a resolved "
            f"order (from {len(records)} usable records)"
        )

    order_names = sorted(pools)
    drawn: list[dict[str, Any]] = []
    depth = 0
    deepest = max(len(pool) for pool in pools.values())
    while len(drawn) < k and depth < deepest:
        for order in order_names:
            if len(drawn) >= k:
                break
            pool = pools[order]
            if depth < len(pool):
                drawn.append(pool[depth])
        depth += 1
    if len(drawn) != k:
        raise ControlSampleError(
            f"round-robin allocation produced {len(drawn)} records, expected {k}"
        )

    allocation = Counter(str(r["order"]) for r in drawn)
    return {
        "k": k,
        "drawn": drawn,
        "n_usable_records": len(records),
        "n_cluster_representatives": len(reps),
        "n_representatives_with_order": available,
        "n_orders_available": len(pools),
        "n_orders_reached": len(allocation),
        # The ceiling and the gap, kept apart from "how many did we hit": an order present
        # in the frame is NOT necessarily an order this frame can stratify over.
        "n_orders_present_in_frame": coverage["n_orders_present"],
        "n_orders_coverable_one_per_cluster": coverage["n_orders_coverable"],
        "orders_unreachable_one_per_cluster": coverage["unreachable_orders"],
        "allocation_per_order": dict(sorted(allocation.items())),
        "min_records_per_order": min(allocation.values()),
        "max_records_per_order": max(allocation.values()),
        "phyla_reached": dict(sorted(Counter(str(r.get("phylum")) for r in drawn).items())),
        "tbox_type": dict(sorted(Counter(str(r.get("tbox_type")) for r in drawn).items())),
        "n_distinct_clusters": len({r["cluster_id"] for r in drawn}),
        "exclusions": dict(sorted(exclusions.items())),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The leakage gate — re-derived from the split table, never read back from config
# ═════════════════════════════════════════════════════════════════════════════
def refuse_duplicate_digests(digests: Iterable[str]) -> None:
    """Refuse a repeated ``corpus_record_sha256`` rather than letting the last row win.

    Split out of :func:`load_split_index` so the rule is exercised without a parquet
    engine (the local test env has pandas but no pyarrow; CI has both), and so a test can
    show the *rule* biting rather than only that pandas can read a file.
    """
    counts = Counter(digests)
    repeated = sorted(digest for digest, n in counts.items() if n > 1)
    if repeated:
        raise ControlSampleError(
            f"the split table carries {len(repeated)} duplicated corpus_record_sha256 "
            f"value(s) (e.g. {repeated[0]}); a duplicate would overwrite its own role and "
            "the leakage clauses would read the surviving row as the whole truth"
        )


def load_split_index(split_table: str | Path = DEFAULT_SPLIT_TABLE) -> dict[str, Any]:
    """The ``nested_train`` / ``heldout`` sets the leakage clauses are derived from.

    Separated from :func:`leakage_clauses` so the clause logic is pandas-free and the unit
    tier can break one clause at a time without a parquet.
    """
    import pandas as pd

    frame = pd.read_parquet(
        split_table,
        columns=[
            "record_id",
            "corpus_record_sha256",
            "cluster_id",
            "nested_role",
            "nested_train",
        ],
    )
    trained = frame[frame["nested_role"] == "train"]
    flagged = frame[frame["nested_train"].astype(bool)]
    corpus_rows = frame.dropna(subset=["corpus_record_sha256"])
    # Refused, not de-duplicated.  The digest→role and digest→record_id maps below are
    # dict comprehensions, so a repeated digest silently keeps its LAST row: a digest
    # carrying both a `train` row and a `heldout` row would resolve to whichever came
    # last, and `every_drawn_record_is_heldout` would pass on a record that IS in the
    # training fold.  `every_drawn_record_resolved` cannot see it — that clause detects
    # two digests collapsing onto one record_id, not one digest collapsing two split rows.
    # `load_joined_frame` calls `.drop_duplicates("corpus_record_sha256")` on this very
    # column, so the join path already treats duplicates as possible; here the leakage
    # gate reads them, and a silent overwrite leaves every summed count reconciling
    # ([[duplicate-key-merges-instead-of-colliding]]).  Measured 2026-08-10: 23,535 rows,
    # 23,535 distinct digests — this refusal does not fire on the committed table.
    refuse_duplicate_digests(str(v) for v in corpus_rows["corpus_record_sha256"])
    return {
        "n_rows": int(len(frame)),
        "trained_record_ids": {str(v) for v in trained["record_id"]},
        "trained_digests": {str(v) for v in trained["corpus_record_sha256"].dropna()},
        "trained_clusters": {int(v) for v in trained["cluster_id"].dropna()},
        "n_trained": int(len(trained)),
        "trained_role_ids": {str(v) for v in trained["record_id"]},
        "flagged_record_ids": {str(v) for v in flagged["record_id"]},
        # Digest → the OTHER two identities the clauses need.  Built here, at the pandas
        # boundary, because a drawn record is keyed by its content digest while the split
        # table's own `record_id` is a different string: comparing a digest set to a
        # record_id set would intersect two namespaces and be vacuously empty — a clause
        # that passes because it can never fire ([[namespace-mismatch-invisible-noop]]).
        "record_id_by_digest": {
            str(d): str(r)
            for d, r in zip(
                corpus_rows["corpus_record_sha256"], corpus_rows["record_id"], strict=True
            )
        },
        "role_by_digest": {
            str(d): str(r)
            for d, r in zip(
                corpus_rows["corpus_record_sha256"], corpus_rows["nested_role"], strict=True
            )
        },
    }


def leakage_clauses(
    drawn: Sequence[Mapping[str, Any]], *, index: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-derive the ADR-0004 D3/D5 disjointness clauses for this draw.

    The scanning checkpoint saw the ``nested_train`` fold; the control frame is its
    complement.  That is a *claim about two artifacts*, and this function checks it on the
    committed table rather than believing either one's provenance — including the ADR-0004
    **D3** clause, ``no_shared_cluster_id``, which is the one a record-id comparison
    cannot see: two structurally near-identical records in one cluster are leakage even
    when no id or digest is shared.

    Every clause carries an emptiness guard, because a set-intersection clause over an
    empty ``nested_train`` set is TRUE for the wrong reason — the exact shape in which a
    silently-missing input reads as a passing gate.
    """
    drawn_digests = {str(r.get("record_sha256") or "") for r in drawn}
    drawn_digests.discard("")
    drawn_clusters = {int(r["cluster_id"]) for r in drawn if r.get("cluster_id") is not None}

    role_by_digest: Mapping[str, str] = index["role_by_digest"]
    record_id_by_digest: Mapping[str, str] = index["record_id_by_digest"]
    resolved = {d for d in drawn_digests if d in role_by_digest}
    roles = Counter(role_by_digest[d] for d in resolved)
    # Translated into the split table's OWN id namespace before intersecting: the drawn
    # set is keyed by content digest, `trained_record_ids` by `record_id`.
    drawn_record_ids = {record_id_by_digest[d] for d in drawn_digests if d in record_id_by_digest}

    clauses = {
        "drawn_nonempty": bool(drawn_digests),
        "nested_train_nonempty": int(index["n_trained"]) > 0,
        "every_drawn_record_resolved": bool(drawn_digests)
        and len(resolved) == len(drawn_digests)
        and len(drawn_record_ids) == len(drawn_digests),
        "every_drawn_record_is_heldout": bool(resolved) and set(roles) == {"heldout"},
        "no_shared_record_id": bool(drawn_record_ids)
        and not (drawn_record_ids & set(index["trained_record_ids"])),
        "no_shared_record_digest": bool(drawn_digests)
        and not (drawn_digests & set(index["trained_digests"])),
        "no_shared_cluster_id": bool(drawn_clusters)
        and not (drawn_clusters & set(index["trained_clusters"])),
        # Set equality, not a count match: two 8,303-row sets of different membership
        # would agree on every count while disagreeing about which records were trained.
        "nested_train_flag_agrees_with_role": int(index["n_trained"]) > 0
        and set(index["trained_role_ids"]) == set(index["flagged_record_ids"]),
    }

    missing = [name for name in REQUIRED_LEAKAGE_CLAUSES if name not in clauses]
    if missing:
        raise ControlSampleError(f"leakage clause set is incomplete — missing {', '.join(missing)}")
    non_bool = [name for name in REQUIRED_LEAKAGE_CLAUSES if not isinstance(clauses[name], bool)]
    if non_bool:
        raise ControlSampleError(
            f"leakage clause(s) {', '.join(non_bool)} are not boolean — a truthy non-bool "
            "would pass `all()` without ever having been evaluated"
        )

    return {
        "clause_schema_version": CLAUSE_SCHEMA_VERSION,
        "clauses": dict(sorted(clauses.items())),
        "all_pass": all(clauses[name] for name in REQUIRED_LEAKAGE_CLAUSES),
        "n_drawn": len(drawn_digests),
        "n_drawn_clusters": len(drawn_clusters),
        "n_drawn_resolved_in_split_table": len(resolved),
        "n_nested_train_records": int(index["n_trained"]),
        "n_nested_train_clusters": len(set(index["trained_clusters"])),
        "drawn_roles": dict(sorted(roles.items())),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Window emission — one unpadded 1,024-nt scan window per drawn record
# ═════════════════════════════════════════════════════════════════════════════
def control_window_name(record_sha256: str, window_start: int) -> str:
    """``<record_sha256>:c0:<window_start>`` — the round-0 window-id shape, this frame."""
    return f"{record_sha256}:c{CONTROL_CONTIG_INDEX}:{int(window_start)}"


def _finite_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if as_float != as_float:  # NaN
        return None
    return int(as_float)


def emit_window(row: Mapping[str, Any], *, window: int = WINDOW_NT) -> dict[str, Any]:
    """The record's single **unpadded** 1,024-nt scan window, or a named refusal.

    The lead is chosen by the shipped eval-mode rule (:func:`window_lead_range` +
    :func:`deterministic_lead`, the locus centred and clamped into the honest range), with
    ``clipped_start`` / ``clipped_end`` forced ``False``.  That is deliberately **stricter**
    than the training-time honesty rule: there, a window may run past ``context_seq`` at a
    real contig boundary and the model reads those positions as zero-flanked padding.
    Here the detected span becomes a **search query**, so a padded position would be a
    fabricated nucleotide handed to ``nhmmer`` (§10.3).  Records with no fully-interior
    window are refused as ``no_unpadded_window`` and counted.

    The round-trip is asserted, not assumed: the locus re-sliced out of the emitted window
    must equal ``context_seq[locus_offset : locus_offset + locus_length]`` byte-for-byte.
    A window that is the right *length* but the wrong *locus* is precisely the defect the
    sizing step's bounds check was added for.
    """
    rid = str(row["_record_sha256"])
    seq = row["context_seq"]
    offset = _finite_int(row["locus_offset"])
    length = _finite_int(row["locus_length"])
    if not isinstance(seq, str) or not seq:
        return {"ok": False, "record_sha256": rid, "reason": "no_context_sequence"}
    if offset is None or length is None or offset < 0 or length < 1:
        return {"ok": False, "record_sha256": rid, "reason": "locus_coordinates_unusable"}
    if offset + length > len(seq):
        return {"ok": False, "record_sha256": rid, "reason": "locus_past_context_end"}
    if length > window:
        return {"ok": False, "record_sha256": rid, "reason": "locus_longer_than_window"}

    lead_range = window_lead_range(
        locus_offset=offset,
        locus_length=length,
        context_length=len(seq),
        clipped_start=False,
        clipped_end=False,
        window=window,
    )
    if lead_range is None:
        return {"ok": False, "record_sha256": rid, "reason": "no_unpadded_window"}
    lead = deterministic_lead(lead_range, window=window, locus_length=length)

    start = offset - lead
    stop = start + window
    if start < 0 or stop > len(seq):
        raise ControlSampleError(
            f"record {rid}: unpadded lead {lead} still leaves window [{start}, {stop}) "
            f"outside a {len(seq)} nt context — window_lead_range's contract is broken"
        )
    window_seq = seq[start:stop]
    if len(window_seq) != window:
        raise ControlSampleError(
            f"record {rid}: emitted window is {len(window_seq)} nt, expected {window}"
        )
    if window_seq[lead : lead + length] != seq[offset : offset + length]:
        raise ControlSampleError(
            f"record {rid}: the locus re-sliced from the emitted window is not the "
            "record's locus — the window is at the wrong offset"
        )

    return {
        "ok": True,
        "reason": None,
        "record_sha256": rid,
        "window_name": control_window_name(rid, start),
        "window_seq": window_seq,
        "window_start": start,
        "window_nt": window,
        "lead": lead,
        "locus_offset": offset,
        "locus_length": length,
        "context_length": len(seq),
        # Carried, not consulted: the no-pad rule forces both to False when choosing the
        # lead, and recording the record's real flags is what lets the report say how many
        # drawn records the stricter rule actually cost.
        "clipped_start": bool(row["clipped_start"]),
        "clipped_end": bool(row["clipped_end"]),
        "replicon": _replicon_provenance(row, start, window),
    }


def _replicon_provenance(row: Mapping[str, Any], start: int, window: int) -> dict[str, Any]:
    """The window's NCBI coordinates — provenance only, never the manifest's frame.

    Recorded so a reviewer can locate any control query in NCBI, and kept *out* of the
    candidate coordinates because ``context_seq`` is already reverse-complemented on the
    minus strand: there, forward coordinates run backwards relative to the context index,
    so the ``window_start + span`` arithmetic every downstream consumer performs would
    mint a coordinate that points at the wrong place while looking entirely plausible.
    """
    from tbox_finder.data.flank_context import forward_bounds

    accession = row.get("accession")
    # Normalised at the pandas boundary: a missing accession arrives as float ``nan``,
    # and ``str(nan)`` is the perfectly present-looking string ``"nan"`` — an accession
    # that addresses no replicon, published with ``reason: None`` as though it were real
    # ([[stringified-null-survives-missing-checks]]).  ``strand`` and ``region_start`` are
    # already covered by ``_finite_int``; this column was the one unguarded seam.
    if not isinstance(accession, str) or not accession.strip():
        accession = None
    strand = _finite_int(row.get("strand"))
    region_start = _finite_int(row.get("region_start"))
    region_len = len(row["context_seq"])
    out: dict[str, Any] = {
        "accession": str(accession) if accession is not None else None,
        "strand": strand,
        "forward_start": None,
        "forward_end": None,
        "reason": None,
    }
    if out["accession"] is None or strand is None or region_start is None:
        out["reason"] = "replicon_geometry_missing"
        return out
    try:
        lo, hi = forward_bounds(
            strand=strand,
            region_start=region_start,
            region_len=region_len,
            offset=start,
            length=window,
        )
    except ValueError as exc:
        out["reason"] = f"forward_bounds_refused:{exc}"
        return out
    out["forward_start"], out["forward_end"] = int(lo), int(hi)
    return out


def emit_windows(
    drawn: Sequence[Mapping[str, Any]],
    wide: Mapping[str, Mapping[str, Any]],
    *,
    window: int = WINDOW_NT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Emit one scan window per drawn record → ``(windows, refusal summary)``."""
    windows: list[dict[str, Any]] = []
    refused: Counter[str] = Counter()
    refused_ids: list[str] = []
    for record in drawn:
        rid = str(record["record_sha256"])
        row = wide.get(rid)
        if row is None:
            refused["record_not_in_frame"] += 1
            refused_ids.append(rid)
            continue
        emitted = emit_window(row, window=window)
        if not emitted["ok"]:
            refused[str(emitted["reason"])] += 1
            refused_ids.append(rid)
            continue
        emitted["order"] = record.get("order")
        emitted["phylum"] = record.get("phylum")
        emitted["genus"] = record.get("genus")
        emitted["cluster_id"] = record.get("cluster_id")
        emitted["tbox_type"] = record.get("tbox_type")
        windows.append(emitted)
    return windows, {
        "n_windows": len(windows),
        "n_refused": len(refused_ids),
        "refusal_reasons": dict(sorted(refused.items())),
        "refused_record_ids": sorted(refused_ids),
    }


def write_windows(windows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One JSON object per line, ``record_sha256``-sorted — the detect leg's input."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(windows, key=lambda w: str(w["record_sha256"]))
    body = "".join(json.dumps(w, sort_keys=True) + "\n" for w in ordered)
    out.write_text(body, encoding="utf-8")
    return out


def read_windows(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ControlSampleError(f"{portable_path(path)} holds no windows")
    widths = {int(row["window_nt"]) for row in rows}
    if len(widths) != 1:
        raise ControlSampleError(
            f"{portable_path(path)} mixes window widths {sorted(widths)}; the control's "
            "queries would then come from differently-sized scans"
        )
    lengths = {len(str(row["window_seq"])) for row in rows}
    if lengths != widths:
        raise ControlSampleError(
            f"{portable_path(path)} declares window_nt {sorted(widths)} but carries "
            f"sequences of length {sorted(lengths)}"
        )
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Stage-1 re-detect → the queries
# ═════════════════════════════════════════════════════════════════════════════
def _default_scan(
    checkpoint: str | Path, pairs: Sequence[tuple[str, str]], *, device: Any = None
) -> list[Any]:
    """The shipped round-0 scan primitive — same checkpoint, same A9 Pin-3 triple.

    ``mining.mine_round.scan_substrate_windows`` is the *only* call site reused here: it
    loads the checkpoint once, runs ``infer.scan.scan_sequence``, applies
    ``call_candidates`` at ``PROVISIONAL_THRESHOLD`` / ``PROVISIONAL_MIN_SPAN`` /
    ``PROVISIONAL_GAP_MERGE``, and maps window-relative spans to frame coordinates through
    ``window_candidates_to_mining``.  Re-spelling any of that here would make "the control
    ran the same detector" a claim rather than a fact.  Torch is imported lazily by it.
    """
    from tbox_finder.mining.mine_round import scan_substrate_windows

    return scan_substrate_windows(checkpoint, list(pairs), device=device)


def detect_candidates(
    windows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: Any = None,
    scan: Callable[..., Sequence[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Scan every emitted window → the round-0-shaped candidate rows.

    ``scan`` is injectable so the unit tier exercises the whole pipeline without torch;
    the default is the shipped scanner and nothing else in this module knows how a span
    is called.
    """
    runner = scan if scan is not None else _default_scan
    pairs = [(str(w["window_name"]), str(w["window_seq"])) for w in windows]
    candidates = runner(checkpoint, pairs, device=device)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_id": str(candidate.candidate_id),
                "accession": str(candidate.accession),
                "locus_start": int(candidate.locus_start),
                "locus_end": int(candidate.locus_end),
                "score": float(candidate.score),
                "pool": str(candidate.pool),
            }
        )
    return rows


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> tuple[int, float]:
    """``(intersection nt, IoU)`` for two half-open spans."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter, (round(inter / union, 4) if union > 0 else 0.0)


def build_queries(
    windows: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Slice each detected span out of its window → the queries + the dropout report.

    Every called span becomes a query, exactly as round 0 turned every called span into a
    candidate — no "best span per record" selection rule is invented here, because that
    rule would be a new decision and this leg pins nothing.  Each query carries whether it
    overlaps the record's **true** curated locus and by how much, which is the free
    by-product of scanning known positives with a checkpoint proved disjoint from them:
    Stage-1's per-locus recall on held-out curated records.
    """
    by_name = {str(w["window_name"]): w for w in windows}
    if len(by_name) != len(windows):
        raise ControlSampleError(
            "two windows share a window_name — the per-record slices would overwrite "
            "each other and every summed count would still reconcile"
        )

    queries: list[dict[str, Any]] = []
    refused: Counter[str] = Counter()
    per_window: Counter[str] = Counter()

    for candidate in sorted(candidates, key=lambda c: str(c["candidate_id"])):
        name_parts = str(candidate["candidate_id"]).rsplit(":", 1)
        window_name = name_parts[0]
        window = by_name.get(window_name)
        if window is None:
            refused["candidate_window_unknown"] += 1
            continue
        start = int(window["window_start"])
        rel_start = int(candidate["locus_start"]) - start
        rel_end = int(candidate["locus_end"]) - start
        span_nt = int(window["window_nt"])
        if not (0 <= rel_start < rel_end <= span_nt):
            refused["span_outside_window"] += 1
            continue
        raw = str(window["window_seq"])[rel_start:rel_end]
        query = degap_to_dna(raw)
        if not is_clean_nucleotide(query):
            refused["ambiguous_alphabet"] += 1
            continue
        # Counted only once the span has survived every refusal, so `spans_per_record`
        # and `n_queries` describe the same population.  Counted earlier, a record whose
        # only span is refused would contribute 1 span, 0 queries, and still appear in
        # `dropped_record_ids` — three statements about one record that cannot all be read
        # together.
        per_window[window_name] += 1
        lead = int(window["lead"])
        locus_length = int(window["locus_length"])
        inter, iou = _overlap(rel_start, rel_end, lead, lead + locus_length)
        queries.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "accession": str(candidate["accession"]),
                "locus_start": int(candidate["locus_start"]),
                "locus_end": int(candidate["locus_end"]),
                "score": float(candidate["score"]),
                "pool": str(candidate["pool"]),
                "record_sha256": str(window["record_sha256"]),
                "window_name": window_name,
                "query_nt": len(query),
                "query_seq": query,
                "overlap_nt": inter,
                "overlap_iou": iou,
                "overlaps_true_locus": inter > 0,
                "order": window.get("order"),
                "phylum": window.get("phylum"),
                "tbox_type": window.get("tbox_type"),
            }
        )

    fired = {q["window_name"] for q in queries}
    silent = sorted(str(w["record_sha256"]) for w in windows if str(w["window_name"]) not in fired)
    recovered = sorted({q["record_sha256"] for q in queries if q["overlaps_true_locus"]})
    best_iou = {}
    for q in queries:
        rid = str(q["record_sha256"])
        best_iou[rid] = max(best_iou.get(rid, 0.0), float(q["overlap_iou"]))

    return {
        "queries": queries,
        "dropout": {
            "n_windows_scanned": len(windows),
            "n_records_with_a_query": len(fired),
            "n_records_dropped": len(silent),
            "dropped_record_ids": silent,
            "dropout_share": round(len(silent) / len(windows), 4) if windows else None,
            "refusal_reasons": dict(sorted(refused.items())),
        },
        "stage1_locus_recall": {
            "note": (
                "Stage-1's per-locus recall on HELD-OUT curated records — legitimate only "
                "because the scanning checkpoint's nested_train fold is re-derived disjoint "
                "from this frame in the draw report's leakage clauses. A record counts as "
                "recovered when at least one called span overlaps its curated locus by >= 1 nt."
            ),
            "n_records": len(windows),
            "n_recovered": len(recovered),
            "recall": round(len(recovered) / len(windows), 4) if windows else None,
            # A record only has a best IoU if it produced a span at all, so this summary's
            # denominator is n_records_with_a_query, NOT n_records.  The key says so:
            # sitting beside `n_records`, a bare `best_iou_per_record` invites reading the
            # mean as an average over every drawn record, silently excluding the dropout
            # from a statistic the dropout is the most interesting part of.
            "n_records_with_a_best_iou": len(best_iou),
            "best_iou_per_record_with_a_query": percentiles(sorted(best_iou.values())),
        },
        "spans_per_record": percentiles(
            sorted(float(per_window.get(str(w["window_name"]), 0)) for w in windows)
        ),
    }


def write_query_fasta(queries: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """``>candidate_id`` → query nucleotides; the producer leg's **sequence-route** input."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for query in sorted(queries, key=lambda q: str(q["candidate_id"])):
        seq = str(query["query_seq"])
        lines.append(f">{query['candidate_id']}")
        lines.extend(seq[i : i + 80] for i in range(0, len(seq), 80))
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def write_manifest(queries: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """The queries in ``covariation_producer``'s manifest shape (its own writer)."""
    from tbox_finder.mining.covariation_producer import CandidateSpec, write_candidate_manifest

    specs = [
        CandidateSpec(
            candidate_id=str(q["candidate_id"]),
            accession=str(q["accession"]),
            locus_start=int(q["locus_start"]),
            locus_end=int(q["locus_end"]),
        )
        for q in sorted(queries, key=lambda q: str(q["candidate_id"]))
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return write_candidate_manifest(specs, path)


# ═════════════════════════════════════════════════════════════════════════════
# Reports
# ═════════════════════════════════════════════════════════════════════════════
def _header() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "pins_nothing": True,
    }


def draw_report(
    *,
    draw: Mapping[str, Any],
    supply: Mapping[str, Any],
    leakage: Mapping[str, Any],
    windows_summary: Mapping[str, Any],
    window_nt: int,
) -> dict[str, Any]:
    body = _header()
    body["design"] = {
        "note": (
            "The §7 sample-design decision of 2026-08-10, executed: query = Stage-1 "
            "re-detect at the ADR-0005 A9 Pin-3 triple, sample = one record per ADR-0004 "
            "cluster with equal allocation across orders, K = 200."
        ),
        "window_nt": window_nt,
        "detection_triple": _detection_triple(),
    }
    body["query_supply"] = dict(supply)
    body["draw"] = {k: v for k, v in draw.items() if k != "drawn"}
    body["drawn_record_ids"] = sorted(str(r["record_sha256"]) for r in draw["drawn"])
    body["leakage"] = dict(leakage)
    body["windows"] = dict(windows_summary)
    return body


def detect_report(
    *,
    built: Mapping[str, Any],
    fp_lengths: Sequence[int],
    checkpoint_sha256: str,
    window_nt: int,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    queries = built["queries"]
    lengths = [int(q["query_nt"]) for q in queries]
    body = _header()
    body["design"] = {"window_nt": window_nt, "detection_triple": _detection_triple()}
    body["checkpoint"] = {
        # The path that was actually scanned, not the module default: with `--checkpoint`
        # pointed elsewhere the report would otherwise name one file beside the sha256 of
        # another, while `provenance.inputs` recorded the real one — one report making two
        # disagreeing statements about which checkpoint produced the queries.
        "path": portable_path(checkpoint_path),
        "sha256": checkpoint_sha256,
        "fold": (
            "ADR-0004 D5 nested_train — the fold the draw report's leakage clauses "
            "re-derive as disjoint from this frame"
        ),
        "why_this_one": (
            "It is the ADR-0005 A9 round-0 cold-start scanner that produced the 941 FP "
            "candidates this control is read against; a different checkpoint would make "
            "the query a different instrument's output."
        ),
    }
    body["dropout"] = dict(built["dropout"])
    body["stage1_locus_recall"] = dict(built["stage1_locus_recall"])
    body["spans_per_record"] = dict(built["spans_per_record"])
    body["queries"] = {
        "n_queries": len(queries),
        "n_overlapping_true_locus": sum(1 for q in queries if q["overlaps_true_locus"]),
        "query_length_nt": percentiles(sorted(float(v) for v in lengths)),
        "score": _score_summary([float(q["score"]) for q in queries]),
        "by_order": dict(sorted(Counter(str(q["order"]) for q in queries).items())),
        "by_tbox_type": dict(sorted(Counter(str(q["tbox_type"]) for q in queries).items())),
    }
    body["matchedness_vs_fp_candidates"] = _matchedness(lengths, fp_lengths)
    return body


#: Decimal places the reported posterior summary is rounded to.  A CUDA forward pass is
#: not bit-reproducible across runs — kernel reduction order moves ``peak_p_elem`` at
#: ~1e-8 — so an unrounded summary would make a byte-identical re-derivation impossible
#: for a *committed* artifact while nothing about the result had changed.  6 dp is two
#: orders of magnitude above the observed noise and far below any meaningful posterior
#: difference; the calling *decisions* are unaffected, which
#: :func:`_score_summary` states as a measured margin rather than an assurance.
SCORE_DECIMALS = 6


def _score_summary(scores: Sequence[float]) -> dict[str, Any]:
    """The posterior summary, rounded, with the margin that makes the rounding safe.

    ``call_candidates`` keeps a span only where ``p_elem >= threshold``, so the smallest
    ``peak_p_elem`` in the set is the closest any call came to the boundary.  Reporting
    that margin turns "the float noise cannot flip a call" from a claim into an arithmetic
    check a reader can do: the margin is ~1e-2 and the noise ~1e-8.
    """
    if not scores:
        return {"n": 0}
    out = percentiles(sorted(scores))
    rounded = {
        key: (round(value, SCORE_DECIMALS) if isinstance(value, float) else value)
        for key, value in out.items()
    }
    threshold = float(_detection_triple()["threshold"])
    rounded["rounded_to_decimals"] = SCORE_DECIMALS
    rounded["min_margin_over_threshold"] = round(min(scores) - threshold, SCORE_DECIMALS)
    rounded["margin_note"] = (
        "min(peak_p_elem) - threshold. A CUDA forward is not bit-reproducible (~1e-8 "
        "across runs); this margin is the distance of the closest call to the calling "
        "boundary, so no span's inclusion turns on that noise."
    )
    return rounded


def _detection_triple() -> dict[str, Any]:
    from tbox_finder.eval import mining_criterion

    return {
        "threshold": mining_criterion.PROVISIONAL_THRESHOLD,
        "min_span": mining_criterion.PROVISIONAL_MIN_SPAN,
        "gap_merge": mining_criterion.PROVISIONAL_GAP_MERGE,
        "source": "ADR-0005 A9 Pin 3, read from eval.mining_criterion (not re-spelled)",
    }


def _matchedness(
    lengths: Sequence[int],
    fp_lengths: Sequence[int],
    *,
    sizing_report: str | Path = SIZING_REPORT,
) -> dict[str, Any]:
    """Did the re-detect actually make the query matched?  The whole point, measured.

    P3-15'-g measured raw curated elements against these same FP candidates.  The same
    statistic on the **re-detected** spans is the test of the §7 decision, and it is
    reported whatever it says — including when it says the re-detect did not close the gap.

    The raw-curated baseline is **read out of the committed sizing report**, never
    re-typed here: a hardcoded 0.9647 would keep reading 0.9647 after the report that
    justifies it changed, which is how a comparison becomes a fabricated one (§10.3).
    """
    if not lengths:
        return {"measured": False, "reason": "no queries were built"}
    out = span_matchedness([int(v) for v in lengths], [int(v) for v in fp_lengths])
    out["measured"] = True
    baseline: float | None = None
    reason: str | None = None
    try:
        sizing = json.loads(Path(sizing_report).read_text(encoding="utf-8"))
        baseline = float(sizing["matchedness"]["ks_d"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        reason = f"unavailable:{exc}"
    out["baseline_raw_curated_ks_d"] = baseline
    out["baseline_source"] = (
        f"{portable_path(sizing_report)} matchedness.ks_d — the raw-curated-element "
        "arm this leg exists to replace."
    )
    if reason is not None:
        out["baseline_reason"] = reason
    return out


#: The shared chunked implementation, bound as a module attribute so a test can
#: monkeypatch it and so a future rename in its home module breaks loudly here.
_sha256_of = sha256_of


def _write_report(body: dict[str, Any], out: str | Path) -> Path:
    path = Path(out)
    payload = json.dumps(body, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _attach_provenance(
    body: dict[str, Any], *, rule: str, labelled: Sequence[tuple[str, str | Path | None]]
) -> None:
    repo_inputs, external = partition_inputs(list(labelled))
    body["provenance"] = build_provenance(
        rule=rule,
        script=portable_path(__file__),
        inputs=repo_inputs,
        outputs=[],
        env_lock=ENV_LOCK,
        adr=ADR,
        extra={"schema_version": SCHEMA_VERSION, "external_inputs": external},
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curated_control_sample",
        description="P3-15'-g-ii: draw the matched control's curated records and "
        "re-detect their queries with the round-0 Stage-1 scanner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_draw = sub.add_parser("draw", help="the leakage-clean order-stratified draw + windows")
    p_draw.add_argument("--k", type=int, default=DEFAULT_K)
    p_draw.add_argument("--corpus", default=DEFAULT_CORPUS)
    p_draw.add_argument("--split-table", default=DEFAULT_SPLIT_TABLE)
    p_draw.add_argument("--context", default=DEFAULT_CONTEXT)
    p_draw.add_argument("--window-nt", type=int, default=WINDOW_NT)
    p_draw.add_argument("--out-windows", default=DEFAULT_WINDOWS)
    p_draw.add_argument("--out", default=DEFAULT_DRAW_REPORT)

    p_detect = sub.add_parser("detect", help="Stage-1 re-detect → queries + manifest + FASTA")
    p_detect.add_argument("--windows", default=DEFAULT_WINDOWS)
    p_detect.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p_detect.add_argument("--fp-manifest", default=DEFAULT_FP_MANIFEST)
    p_detect.add_argument("--device", default=None)
    p_detect.add_argument("--out-manifest", default=DEFAULT_MANIFEST)
    p_detect.add_argument("--out-fasta", default=DEFAULT_QUERY_FASTA)
    p_detect.add_argument("--out", default=DEFAULT_DETECT_REPORT)
    return parser


def _run_draw(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.k < 1:
        raise ControlSampleError(f"--k must be >= 1, got {args.k}")
    joined = load_joined_frame(
        corpus=args.corpus, split_table=args.split_table, context=args.context
    )
    usable, supply, wide = eligible_records(joined)
    draw = order_stratified_draw(usable, k=args.k)
    leakage = leakage_clauses(draw["drawn"], index=load_split_index(args.split_table))
    if not leakage["all_pass"]:
        failed = [name for name, ok in leakage["clauses"].items() if not ok]
        raise ControlSampleError(
            "the draw is not leakage-clean — failing clause(s): " + ", ".join(sorted(failed))
        )
    windows, windows_summary = emit_windows(draw["drawn"], wide, window=args.window_nt)
    if not windows:
        raise ControlSampleError("no drawn record yielded an unpadded scan window")
    write_windows(windows, args.out_windows)
    body = draw_report(
        draw=draw,
        supply=supply,
        leakage=leakage,
        windows_summary=windows_summary,
        window_nt=args.window_nt,
    )
    _attach_provenance(
        body,
        rule="P3-15'-g-ii curated-control draw",
        labelled=[
            ("corpus", args.corpus),
            ("split_table", args.split_table),
            ("context", args.context),
        ],
    )
    return body, (
        f"drew {draw['k']} records over {draw['n_orders_reached']}/"
        f"{draw['n_orders_available']} orders; {windows_summary['n_windows']} scan windows"
    )


def _run_detect(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    windows = read_windows(args.windows)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise ControlSampleError(
            f"checkpoint {portable_path(checkpoint)} is not on disk — `dvc pull` it; the "
            "control must be re-detected by the round-0 scanner, not a substitute"
        )
    candidates = detect_candidates(windows, checkpoint=checkpoint, device=args.device)
    built = build_queries(windows, candidates)
    queries = built["queries"]
    if not queries:
        raise ControlSampleError(
            "Stage-1 called no usable span on any of the "
            f"{len(windows)} windows — there is no control to build"
        )
    write_manifest(queries, args.out_manifest)
    write_query_fasta(queries, args.out_fasta)
    fp_manifest = json.loads(Path(args.fp_manifest).read_text(encoding="utf-8"))
    body = detect_report(
        built=built,
        fp_lengths=fp_span_lengths(fp_manifest),
        checkpoint_sha256=_sha256_of(checkpoint),
        window_nt=int(windows[0]["window_nt"]),
        checkpoint_path=checkpoint,
    )
    _attach_provenance(
        body,
        rule="P3-15'-g-ii curated-control Stage-1 re-detect",
        labelled=[
            ("windows", args.windows),
            ("checkpoint", args.checkpoint),
            ("fp_manifest", args.fp_manifest),
        ],
    )
    return body, (
        f"detected {len(queries)} queries on {built['dropout']['n_records_with_a_query']}"
        f"/{built['dropout']['n_windows_scanned']} records "
        f"({built['dropout']['n_records_dropped']} dropped out)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = _run_draw if args.command == "draw" else _run_detect
    try:
        body, message = runner(args)
    except (
        ControlSampleError,
        SizingError,
        ValueError,
        OSError,
        KeyError,
        HomologMsaError,
    ) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    try:
        out = _write_report(body, args.out)
    except TypeError as exc:
        print(f"refused: the report body is not JSON-serializable: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(
            f"refused: cannot write report to {portable_path(args.out)}: {exc}",
            file=sys.stderr,
        )
        return 3
    print(f"{message} -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
