"""host_order.py — the whole-genome host-order negative-admission path (ADR-0004 A5; P2-10c′-a2).

**The gap A5 closes.** ADR-0004 **A4** shipped the flank-carved negative rule (**a1**),
keyed on each window's *parent corpus record* (``mining/pool.py::load_parent_folds`` /
``stamp_parent_folds`` → keyed on ``source_record_id``, ``source == "corpus"`` only), and
*explicitly deferred* the whole-genome case to "the point a whole-genome background pool
is introduced". P2-10c′ introduced it: the fetched substrate (ADR-0006 A1 — 2,500 GTDB
R232 species reps, tiled 1024/512 into 13,481,953 windows) carries a **GCA_/GCF_ assembly
accession** as its host, *not* a corpus ``record_id`` — so the a1 path has no key for it
and refuses every fetched window ``parent_fold_unknown``. Fail-closed, but the pool is
unusable (P2-10e has no negative substrate).

**A5 — the rule (option a2, user sign-off 2026-07-26).** A fetched window is admissible as
a training negative **iff its host genome resolves to a corpus-namespace ``resolved_order``
(and phylum/class) that is not a designated D5 holdout unit**:

1. **Bridge (new provenance).** Persist an NCBI taxid per production genome from the
   URL+MD5-pinned GTDB metadata (``bac120_metadata_r232.tsv.gz`` / ``ar53_metadata_r232.tsv.gz``,
   ``taxonomy.FILES``) joined on ``gtdb_accession`` → its ``ncbi_taxid``; resolve that taxid
   to an NCBI order/phylum/class via the **already-pinned** taxdump and the **identical**
   ``taxonomy.read_taxdump`` + ``resolve_row`` + vintage ``reconcile_name`` path that produced
   the corpus ``resolved_order`` (same pinned R232 release, same corpus vintage vocab) — so
   the host order lands in the **same namespace** as the corpus holdout set. No hand-built
   GTDB↔NCBI rename map (D4's anti-pattern); reconcile via the taxdump's own synonym records.
2. **Exclusion (blacklist held-out — the mirror of a1).** Refuse the window **iff** its host's
   resolved lineage intersects **any designated D5 holdout unit** — one of the 30
   leave-one-order-out held-out orders or the Actinobacteria phylum-holdout, exactly the
   ``member_heldout`` set the nested most-restrictive training fold excludes
   (``splits.py``: ``(order ∈ heldout_orders) or (phylum == HOLDOUT_PHYLUM)``). **Admit** iff
   the host resolves entirely to **training** taxa **or** to a taxon with **no corpus
   positives at all** (independent-negative territory the discovery set exists to supply;
   such a genome hosts no held-out *catalogued* positive and cannot leak one).
3. **Fail-closed (the mirror of D4).** Refuse any host whose order is **unresolvable** — a
   MAG/environmental taxid with no formal order rank, an unresolved name, or a missing taxid;
   and refuse any window whose host is **absent from the host-order table** (a broken join
   turns the gate RED, never green-with-nothing-refused — [[namespace-mismatch-invisible-noop]]).

The class-holdout the A5 prose names is **subsumed by the Actinobacteria phylum-holdout**
(holding out the whole phylum removes all its classes); :data:`HeldoutUnits.classes` carries
the phylum's classes as an explicit, redundant belt-and-suspenders check that can never
under-refuse. **No hardcoded held-out set** — the blacklist is derived from the committed
``split_assignments.parquet`` the §8.2 no-leakage CI already reads (single source of truth).

This module is **torch/pandas-import-light at module scope** (pandas is lazily imported only
inside the IO functions) so the pure logic — :func:`host_accession`, :func:`classify_host`,
:func:`stamp_host_folds` — is unit-testable without a heavy env, matching ``mining/pool.py``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tbox_finder import masking, provenance, splits, taxonomy

# --------------------------------------------------------------------------- #
# Canonical artifact paths (one path per artifact).
# --------------------------------------------------------------------------- #
MINING_DIR = "data/processed/mining"
#: The committed, sequence-free host-order table — keyed on ``assembly_accession``. Committed
#: to git (NOT DVC) for the same reason ``split_assignments.parquet`` is (``splits.py`` §"P0-23"):
#: the §8.2 no-leakage CI must read it on every PR without a DVC pull.
HOST_ORDER_TABLE = f"{MINING_DIR}/production_host_orders_v0.parquet"
HOST_ORDER_PROVENANCE = f"{MINING_DIR}/production_host_orders_v0.provenance.json"
HOST_ORDER_REPORT = "data/processed/audits/production_host_orders_report.json"

PRODUCTION_GENOMES = f"{MINING_DIR}/production_genomes_v0.parquet"
PRODUCTION_WINDOWS = f"{MINING_DIR}/production_windows_v0.parquet"

HOST_ORDER_RULE = "workflow/rules/data.smk :: build_production_host_orders"
HOST_ORDER_SCRIPT = "src/tbox_finder/mining/host_order.py"
HOST_ORDER_ADR = "ADR-0004"

# --------------------------------------------------------------------------- #
# Column / value vocabulary (closed; mirrors mining/pool.py's discipline).
# --------------------------------------------------------------------------- #
#: The GTDB metadata files carrying ``accession → ncbi_taxid`` (``taxonomy.FILES``, staged=False).
METADATA_FILE_NAMES = ("bac120_metadata_r232.tsv.gz", "ar53_metadata_r232.tsv.gz")
#: The two metadata TSV columns the join reads (GTDB R232 header: accession, ncbi_taxid).
META_ACCESSION_COL = "accession"
META_TAXID_COL = "ncbi_taxid"

#: Host-order-table columns. ``resolved_{phylum,class,order}`` are in the SAME corpus vintage
#: namespace as ``split_assignments.parquet``'s columns of the same name (that is the whole point).
ACCESSION_COL = "assembly_accession"
GTDB_ACCESSION_COL = "gtdb_accession"
TAXID_COL = "ncbi_taxid"
#: The D5 holdout ranks (``taxonomy.HOLDOUT_RANKS`` == ("phylum", "class", "order")); the a2 rule
#: needs no finer rank than order, so the bridge resolves exactly these (fewer taxdump walks).
RANKS: tuple[str, ...] = taxonomy.HOLDOUT_RANKS

#: Per-window admissibility column + refusal-reason column (analog of ``pool.PARENT_FOLD_COL``).
HOST_FOLD_COL = "host_order_admissible"
HOST_REASON_COL = "host_order_reason"

#: Closed refusal-reason vocabulary. ``admitted`` is the only admit verdict; the rest refuse.
ADMITTED = "admitted"
REFUSED_HELDOUT_ORDER = "refused_heldout_order"
REFUSED_HELDOUT_PHYLUM = "refused_heldout_phylum"
REFUSED_HELDOUT_CLASS = "refused_heldout_class"
REFUSED_UNRESOLVABLE = "refused_unresolvable_host"
REFUSED_HOST_UNKNOWN = "refused_host_unknown"
REFUSAL_REASONS = frozenset(
    {
        REFUSED_HELDOUT_ORDER,
        REFUSED_HELDOUT_PHYLUM,
        REFUSED_HELDOUT_CLASS,
        REFUSED_UNRESOLVABLE,
        REFUSED_HOST_UNKNOWN,
    }
)

#: Committed split table columns this module reads (source of the blacklist).
SPLIT_SOURCE_COL = "source"
SPLIT_CORPUS_SOURCE = "corpus"

#: First line of a Git LFS v1 pointer (mirrors ``pool._LFS_POINTER_MAGIC``): an unsmudged
#: pointer parses as *no rows*, so a blacklist read off it would silently be empty
#: ([[git-lfs-pointers-in-ci]]).
_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


class HostOrderError(ValueError):
    """Raised when the host-order bridge or admit path cannot be built or resolved."""


# ======================================================================================
# Pure logic — window-id parsing, missing-value cleaning, the a2 classification rule
# ======================================================================================


def host_accession(window_name: str) -> str:
    """Extract a fetched window's host ``assembly_accession`` from its window id.

    The id is ``substrate_windows.window_name(accession, contig_index, start)`` ==
    ``f"{accession}:c{contig_index}:{start}"``. The accession (``GC[AF]_<digits>.<digits>``)
    carries no ``:``; ``:c<int>:<int>`` is the geometry suffix. Fails loud on any id that is
    not that shape rather than laundering a malformed host into an accession that addresses
    no genome (the fail-open direction — mirrors ``pool.carve_window``'s missing-accession guard).
    """
    head, sep, tail = str(window_name).rpartition(":c")
    ci, dot, start = tail.partition(":")
    if not sep or not dot or not ci.isdigit() or not start.isdigit() or not head:
        raise HostOrderError(
            f"window id {window_name!r} is not "
            "'<accession>:c<contig_index>:<start>' — cannot recover a host accession"
        )
    return head


def _clean(value: Any) -> str | None:
    """A resolved lineage name, or ``None`` for any missing sentinel.

    Routes through ``masking.is_missing`` so a ``NaN`` read back from parquet — **truthy**
    under the training env's pandas 3 ([[pandas-3-nan-truthy-in-training-env]]) — becomes
    ``None`` rather than the literal string ``"nan"`` (a present-looking name in no vocab).
    """
    if masking.is_missing(value):
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class HeldoutUnits:
    """The designated D5 holdout units, derived from the committed split table.

    ``orders`` — the leave-one-order-out held-out orders (``is_designated_loo_holdout``).
    ``phyla`` — the well-powered held-out phylum (``splits.HOLDOUT_PHYLUM``), cross-checked
    against the table so a drift between the constant and the committed partition turns RED.
    ``classes`` — the classes *within* the held-out phyla; redundant with the phylum rule
    (holding out the phylum removes all its classes) but carried explicitly per ADR-0004 A5.
    """

    orders: frozenset[str]
    phyla: frozenset[str]
    classes: frozenset[str]


def classify_host(
    resolved_order: str | None,
    resolved_phylum: str | None,
    resolved_class: str | None,
    heldout: HeldoutUnits,
) -> tuple[bool, str]:
    """The a2 rule for one host lineage → ``(admissible, reason)``.

    Order of tests is fail-closed first: an unresolvable order refuses before any admit path
    is reached; a held-out order/phylum/class refuses; only a fully-resolved, non-held-out
    host is admitted. The class test is subsumed by the phylum test and can never be the sole
    refusal on the real partition, but it is applied unconditionally so a future partition that
    designates a class outside the held-out phylum would still refuse (leakage-safe direction).
    """
    order = _clean(resolved_order)
    phylum = _clean(resolved_phylum)
    klass = _clean(resolved_class)
    if order is None:
        return False, REFUSED_UNRESOLVABLE
    if order in heldout.orders:
        return False, REFUSED_HELDOUT_ORDER
    if phylum is not None and phylum in heldout.phyla:
        return False, REFUSED_HELDOUT_PHYLUM
    if klass is not None and klass in heldout.classes:
        return False, REFUSED_HELDOUT_CLASS
    return True, ADMITTED


# ======================================================================================
# Blacklist derivation — from the committed split table (single source of truth)
# ======================================================================================


def _assert_smudged_parquet(path: Path) -> None:
    """Fail loud on a Git-LFS pointer masquerading as a parquet (mirrors ``pool``'s guard).

    An unsmudged pointer parses as no rows — a silently empty read the a2 gate must refuse.
    """
    if not path.is_file():
        raise HostOrderError(f"table not found: {path}")
    with path.open("rb") as handle:
        if handle.read(len(_LFS_POINTER_MAGIC)) == _LFS_POINTER_MAGIC:
            raise HostOrderError(
                f"table {path} is an unsmudged Git-LFS pointer, not a parquet — it would "
                "resolve an empty blacklist while looking like a healthy read "
                "(git lfs pull, or re-stage it after `git reset --hard`)"
            )


def derive_heldout_units(split_table: str | Path = splits.COMMITTED_TABLE) -> HeldoutUnits:
    """Derive the designated D5 holdout units from the committed ``split_assignments`` table.

    ``orders`` == ``{resolved_order : is_designated_loo_holdout and source == "corpus"}`` — by
    construction (``splits.py``: ``is_designated_loo_holdout = (order in heldout_set) and corp``)
    exactly the leave-one-order-out held-out orders, read off the same table the §8.2 CI reads.

    ``phyla`` == ``{splits.HOLDOUT_PHYLUM}``, single-sourced from the module that built the
    partition, then **cross-checked**: no ``nested_train`` corpus record may carry a held-out
    phylum (else the constant has drifted from the committed partition — fail loud, RED).

    An **empty** held-out order set is refused: it means the join found nothing (a broken /
    unsmudged / wrong table), which would make every host admissible — the exact silent no-op
    [[namespace-mismatch-invisible-noop]] warns against.
    """
    import pandas as pd

    path = Path(split_table)
    _assert_smudged_parquet(path)
    frame = pd.read_parquet(
        path,
        columns=[
            SPLIT_SOURCE_COL,
            "resolved_order",
            "resolved_phylum",
            "resolved_class",
            "nested_train",
            "is_designated_loo_holdout",
        ],
    )
    corpus = frame[frame[SPLIT_SOURCE_COL] == SPLIT_CORPUS_SOURCE]
    # Store the CLEANED name (not the raw cell): classify_host looks up ``_clean(order)``, so a
    # whitespace-padded / non-str cell in the split table would otherwise store one form and be
    # queried by another — a silent miss that ADMITS a held-out host (CodeRabbit).
    orders = frozenset(
        cleaned
        for o, d in zip(corpus["resolved_order"], corpus["is_designated_loo_holdout"], strict=True)
        if bool(d) and (cleaned := _clean(o)) is not None
    )
    if not orders:
        raise HostOrderError(
            f"split table {path} yielded ZERO designated held-out orders "
            "(is_designated_loo_holdout all False / broken join / wrong table) — refusing to "
            "build a blacklist that would admit every host"
        )
    phyla = frozenset({splits.HOLDOUT_PHYLUM})
    leaked = [
        _clean(p)
        for p, t in zip(corpus["resolved_phylum"], corpus["nested_train"], strict=True)
        if bool(t) and _clean(p) in phyla
    ]
    if leaked:
        raise HostOrderError(
            f"HOLDOUT_PHYLUM {sorted(phyla)} has {len(leaked)} nested_train corpus records in "
            f"{path} — splits.HOLDOUT_PHYLUM has drifted from the committed partition"
        )
    classes = frozenset(
        cleaned
        for c, p in zip(corpus["resolved_class"], corpus["resolved_phylum"], strict=True)
        if _clean(p) in phyla and (cleaned := _clean(c)) is not None
    )
    return HeldoutUnits(orders=orders, phyla=phyla, classes=classes)


# ======================================================================================
# The bridge — GTDB accession → NCBI taxid → corpus-namespace order/phylum/class
# ======================================================================================


def _metadata_files() -> list[taxonomy.GtdbFile]:
    """The ``taxonomy.FILES`` entries for the two GTDB metadata TSVs (fail loud if absent)."""
    files = [f for f in taxonomy.FILES if f.name in METADATA_FILE_NAMES]
    missing = set(METADATA_FILE_NAMES) - {f.name for f in files}
    if missing:
        raise HostOrderError(f"taxonomy.FILES is missing metadata pin(s): {sorted(missing)}")
    return files


def load_host_taxids(
    metadata_paths: Iterable[str | Path], wanted_gtdb_accessions: Iterable[str]
) -> dict[str, int | None]:
    """Stream the gz metadata TSVs → ``{gtdb_accession: ncbi_taxid|None}`` for the wanted set.

    Read with the stdlib (``gzip`` + ``csv``), never pandas: the ~2 GB, ~700-column bacterial
    metadata TSV need not be materialised to recover two columns for 2,500 accessions, and
    streaming sidesteps the pandas missing-value dtype trap entirely. A non-integer / blank
    ``ncbi_taxid`` maps to ``None`` (→ unresolvable → fail-closed downstream), never coerced.
    """
    wanted = {str(a) for a in wanted_gtdb_accessions}
    taxids: dict[str, int | None] = {}
    for path in metadata_paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header is None:
                raise HostOrderError(f"metadata {path} is empty (no header row)")
            try:
                acc_i = header.index(META_ACCESSION_COL)
                tax_i = header.index(META_TAXID_COL)
            except ValueError as exc:
                raise HostOrderError(
                    f"metadata {path} header missing "
                    f"{META_ACCESSION_COL!r}/{META_TAXID_COL!r}: {exc}"
                ) from exc
            need = max(acc_i, tax_i)
            for row in reader:
                if len(row) <= need:
                    continue
                acc = row[acc_i]
                if acc not in wanted:
                    continue
                raw = row[tax_i].strip()
                taxids[acc] = int(raw) if raw.isdigit() else None
    return taxids


def build_host_order_table(
    production_genomes_path: str | Path = PRODUCTION_GENOMES,
    *,
    gtdb_dir: str | Path = taxonomy.GTDB_DIR,
    taxdump_dir: str | Path = taxonomy.NCBI_TAXONOMY_DIR,
    corpus_path: str | Path = taxonomy.CORPUS_PARQUET,
    download: bool = True,
    ranks: Sequence[str] = RANKS,
) -> tuple[Any, dict[str, str]]:
    """Resolve every production genome's host order/phylum/class into the corpus namespace.

    Returns ``(host_order_dataframe, metadata_sha256s)``. The dataframe carries one row per
    production genome: ``assembly_accession``, ``gtdb_accession``, ``ncbi_taxid`` (nullable),
    ``resolved_{phylum,class,order}`` + ``{rank}_source`` — resolved by the **identical**
    ``taxonomy.resolve_row`` path (taxid lineage + corpus vintage vocab) that produced the
    corpus ``resolved_order``, so string equality against the D5 holdout set is well-defined.
    """
    import pandas as pd

    genomes = pd.read_parquet(production_genomes_path, columns=[ACCESSION_COL, GTDB_ACCESSION_COL])
    gtdb_dir = Path(gtdb_dir)

    # 1. Fetch (MD5-verified) + stream the metadata → gtdb_accession -> ncbi_taxid.
    meta_shas: dict[str, str] = {}
    meta_paths: list[Path] = []
    for gfile in _metadata_files():
        if download:
            sha, _ = taxonomy.ensure_file(gfile, gtdb_dir)
            meta_shas[gfile.name] = sha
        meta_paths.append(gtdb_dir / gfile.name)
    wanted_gtdb = [str(a) for a in genomes[GTDB_ACCESSION_COL]]
    taxid_by_gtdb = load_host_taxids(meta_paths, wanted_gtdb)

    # 2. Resolve the host taxids through the pinned taxdump (order/phylum/class lineage + names).
    host_taxids = {t for t in taxid_by_gtdb.values() if t is not None}
    zip_path = taxonomy.fetch_taxdump(taxdump_dir, download=download)
    lineages, names_by_taxid = taxonomy.read_taxdump(zip_path, host_taxids, ranks)

    # 3. Corpus vintage vocab (the reconciliation target) — the SAME source replace_lineage uses.
    corpus = pd.read_parquet(corpus_path, columns=list(ranks))
    vocab = {r: frozenset(corpus[r].dropna().astype(str)) for r in ranks}

    empty_existing: dict[str, str | None] = {r: None for r in ranks}
    rows: list[dict[str, Any]] = []
    for gtdb_acc, asm_acc in zip(
        genomes[GTDB_ACCESSION_COL].astype(str),
        genomes[ACCESSION_COL].astype(str),
        strict=True,
    ):
        taxid = taxid_by_gtdb.get(gtdb_acc)
        if taxid is None:
            resolved: dict[str, Any] = {}
            for r in ranks:
                resolved[f"resolved_{r}"] = None
                resolved[f"{r}_source"] = taxonomy.SOURCE_UNRESOLVED
        else:
            resolved = taxonomy.resolve_row(
                empty_existing, lineages.get(taxid, {}), names_by_taxid, vocab, ranks
            )
        rows.append(
            {
                ACCESSION_COL: asm_acc,
                GTDB_ACCESSION_COL: gtdb_acc,
                TAXID_COL: None if taxid is None else int(taxid),
                **resolved,
            }
        )
    out = pd.DataFrame.from_records(rows)
    out[TAXID_COL] = out[TAXID_COL].astype("Int64")
    return out, meta_shas


# ======================================================================================
# Admit path — the genome-accession-keyed analog of pool.load_parent_folds/stamp_parent_folds
# ======================================================================================


def load_host_folds(
    host_order_table: str | Path = HOST_ORDER_TABLE,
    split_table: str | Path = splits.COMMITTED_TABLE,
) -> tuple[dict[str, tuple[bool, str]], HeldoutUnits]:
    """``{assembly_accession -> (admissible, reason)}`` + the derived holdout units.

    The analog of ``pool.load_parent_folds``: it reads the committed host-order table (the
    bridge's output) and the committed split table (the blacklist source), then applies the
    a2 rule per host. Fail-closed everywhere — an LFS pointer, an empty held-out set, or a
    missing host all raise / refuse rather than silently admitting.
    """
    import pandas as pd

    heldout = derive_heldout_units(split_table)
    path = Path(host_order_table)
    _assert_smudged_parquet(path)
    frame = pd.read_parquet(
        path, columns=[ACCESSION_COL, "resolved_order", "resolved_phylum", "resolved_class"]
    )
    verdicts: dict[str, tuple[bool, str]] = {}
    for acc, order, phylum, klass in zip(
        frame[ACCESSION_COL].astype(str),
        frame["resolved_order"],
        frame["resolved_phylum"],
        frame["resolved_class"],
        strict=True,
    ):
        verdicts[str(acc)] = classify_host(order, phylum, klass, heldout)
    return verdicts, heldout


def stamp_host_folds(
    records: Sequence[dict[str, Any]], admissible_by_accession: Mapping[str, tuple[bool, str]]
) -> dict[str, Any]:
    """Stamp :data:`HOST_FOLD_COL`/:data:`HOST_REASON_COL` on each fetched window + return evidence.

    The analog of ``pool.stamp_parent_folds``: admissibility is **data on the window**, not a
    promise by whoever loads it. Each record's host accession is read from an
    ``assembly_accession`` field if present, else recovered from its ``window_name`` id. A host
    **absent** from the admission map is counted separately and refused ``refused_host_unknown``
    — a broken join, not a silent admit ([[namespace-mismatch-invisible-noop]]).
    """
    reasons: Counter[str] = Counter()
    hosts: set[str] = set()
    n_host_unknown = 0
    n_admissible = 0
    for record in records:
        raw = record.get(ACCESSION_COL)
        acc = str(raw) if raw not in (None, "") else host_accession(str(record["window_name"]))
        hosts.add(acc)
        verdict = admissible_by_accession.get(acc)
        if verdict is None:
            n_host_unknown += 1
            admissible, reason = False, REFUSED_HOST_UNKNOWN
        else:
            admissible, reason = verdict
        record[HOST_FOLD_COL] = bool(admissible)
        record[HOST_REASON_COL] = reason
        reasons[reason] += 1
        n_admissible += bool(admissible)
    return {
        "n_records": len(records),
        "n_admissible": n_admissible,
        "n_refused": len(records) - n_admissible,
        "n_host_unknown": n_host_unknown,
        "n_distinct_hosts": len(hosts),
        "refusal_reasons": dict(sorted(reasons.items())),
    }


# ======================================================================================
# CLI — build the committed host-order table + provenance + audit report
# ======================================================================================


def _audit_report(
    table, heldout: HeldoutUnits, windows_by_accession: Mapping[str, int]
) -> dict[str, Any]:
    """Genome- and window-weighted admission accounting from the resolved table (fail-loud)."""
    genome_reasons: Counter[str] = Counter()
    window_reasons: Counter[str] = Counter()
    n_windows_total = 0
    heldout_orders_seen: set[str] = set()
    for acc, order, phylum, klass in zip(
        table[ACCESSION_COL].astype(str),
        table["resolved_order"],
        table["resolved_phylum"],
        table["resolved_class"],
        strict=True,
    ):
        admissible, reason = classify_host(order, phylum, klass, heldout)
        n_win = int(windows_by_accession.get(str(acc), 0))
        genome_reasons[reason] += 1
        window_reasons[reason] += n_win
        n_windows_total += n_win
        cleaned = _clean(order)
        if reason == REFUSED_HELDOUT_ORDER and cleaned is not None:
            heldout_orders_seen.add(cleaned)
    n_genomes = int(len(table))
    n_admissible_genomes = int(genome_reasons.get(ADMITTED, 0))
    n_admissible_windows = int(window_reasons.get(ADMITTED, 0))
    if sum(genome_reasons.values()) != n_genomes:
        raise HostOrderError("genome accounting mismatch (§10.3) in host-order audit")
    return {
        "n_genomes": n_genomes,
        "n_admissible_genomes": n_admissible_genomes,
        "n_refused_genomes": n_genomes - n_admissible_genomes,
        "genome_reason_counts": dict(sorted(genome_reasons.items())),
        "n_windows_total": n_windows_total,
        "n_admissible_windows": n_admissible_windows,
        "n_refused_windows": n_windows_total - n_admissible_windows,
        "window_reason_counts": dict(sorted(window_reasons.items())),
        "n_designated_heldout_orders": len(heldout.orders),
        "n_heldout_orders_present_among_hosts": len(heldout_orders_seen),
        "heldout_phyla": sorted(heldout.phyla),
        "admissible_window_fraction": (
            round(n_admissible_windows / n_windows_total, 6) if n_windows_total else None
        ),
    }


def build(
    *,
    production_genomes_path: str | Path = PRODUCTION_GENOMES,
    windows_manifest: str | Path = PRODUCTION_WINDOWS,
    split_table: str | Path = splits.COMMITTED_TABLE,
    out_table: str | Path = HOST_ORDER_TABLE,
    provenance_path: str | Path = HOST_ORDER_PROVENANCE,
    report_path: str | Path = HOST_ORDER_REPORT,
    gtdb_dir: str | Path = taxonomy.GTDB_DIR,
    taxdump_dir: str | Path = taxonomy.NCBI_TAXONOMY_DIR,
    corpus_path: str | Path = taxonomy.CORPUS_PARQUET,
    env_lock: str | None = None,
    download: bool = True,
) -> int:
    """Resolve, write the committed host-order table, its provenance, and the audit report."""
    import pandas as pd

    table, meta_shas = build_host_order_table(
        production_genomes_path,
        gtdb_dir=gtdb_dir,
        taxdump_dir=taxdump_dir,
        corpus_path=corpus_path,
        download=download,
    )
    heldout = derive_heldout_units(split_table)

    win_df = pd.read_parquet(windows_manifest, columns=[ACCESSION_COL, "n_windows"])
    windows_by_accession = {
        str(a): int(n) for a, n in zip(win_df[ACCESSION_COL], win_df["n_windows"], strict=True)
    }
    report = _audit_report(table, heldout, windows_by_accession)

    out_table = Path(out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_table, index=False)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )

    taxdump_zip = Path(taxdump_dir) / taxonomy.TAXDUMP_SNAPSHOT
    provenance.write_provenance(
        provenance_path,
        rule=HOST_ORDER_RULE,
        script=HOST_ORDER_SCRIPT,
        adr=HOST_ORDER_ADR,
        env_lock=env_lock,
        inputs=[production_genomes_path, windows_manifest, split_table, corpus_path, taxdump_zip],
        outputs=[out_table, report_path],
        extra={
            "metadata_md5_pinned": {f.name: f.md5 for f in _metadata_files()},
            "metadata_sha256_fetched": meta_shas,
            "taxdump_snapshot": taxonomy.TAXDUMP_SNAPSHOT,
            "n_genomes": report["n_genomes"],
            "n_admissible_genomes": report["n_admissible_genomes"],
            "n_admissible_windows": report["n_admissible_windows"],
            "admissible_window_fraction": report["admissible_window_fraction"],
        },
    )
    print(
        f"host-order table: {report['n_admissible_genomes']}/{report['n_genomes']} genomes "
        f"admissible ({report['n_admissible_windows']}/{report['n_windows_total']} windows); "
        f"reasons={report['genome_reason_counts']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbox_finder.mining.host_order")
    parser.add_argument("--production-genomes", default=PRODUCTION_GENOMES)
    parser.add_argument("--windows-manifest", default=PRODUCTION_WINDOWS)
    parser.add_argument("--split-table", default=splits.COMMITTED_TABLE)
    parser.add_argument("--out", default=HOST_ORDER_TABLE)
    parser.add_argument("--provenance", default=HOST_ORDER_PROVENANCE)
    parser.add_argument("--report", default=HOST_ORDER_REPORT)
    parser.add_argument("--gtdb-dir", default=str(taxonomy.GTDB_DIR))
    parser.add_argument("--taxdump-dir", default=str(taxonomy.NCBI_TAXONOMY_DIR))
    parser.add_argument("--corpus", default=str(taxonomy.CORPUS_PARQUET))
    parser.add_argument("--env-lock", default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="use already-staged metadata/taxdump; do not fetch (offline re-derive)",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    return build(
        production_genomes_path=args.production_genomes,
        windows_manifest=args.windows_manifest,
        split_table=args.split_table,
        out_table=args.out,
        provenance_path=args.provenance,
        report_path=args.report,
        gtdb_dir=args.gtdb_dir,
        taxdump_dir=args.taxdump_dir,
        corpus_path=args.corpus,
        env_lock=args.env_lock,
        download=not args.no_download,
    )


if __name__ == "__main__":
    raise SystemExit(main())
