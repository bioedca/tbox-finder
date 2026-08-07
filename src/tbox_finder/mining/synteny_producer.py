"""The per-candidate criterion-(c) producer + ADR-0006 D4's two **symmetric** diagnostics.

Three artifacts, and D4 asks for all three together — the status table on its own would be a
number with no error bars in either direction:

``$ROUND_DIR/synteny_status.json``
    ``candidate_id → passed | failed | unavailable`` for the ``downstream_aaRS_synteny``
    disjunct, in the same shape
    :func:`tbox_finder.mining.covariation_producer.merge_status_tables` writes for (a), so
    :func:`tbox_finder.mining.mine_round.candidate_evidence` reads it with the identical
    fail-closed absent-id rule.

``reports/p3/synteny_false_pass.json``
    D4: *"the (c) — and joint a ∧ b ∧ c — false-pass rate is estimated on clade-matched
    random leaders + the §9.1 decoys"*.

``reports/p3/synteny_exclusion_diagnostic.json``
    D4: *"a false-FAIL rate + per-clade (c)-exclusion + pseudogene diagnostic is reported
    symmetric with the false-pass rate, so the annotation-driven recall cap in incompletely
    annotated CPR/DPANN/MAG lineages is measured, not silent"*.

Three measured gaps this step reports rather than papers over
-------------------------------------------------------------
1. **The shipped §9.1 decoy rows cannot carry the (c) arm.** Criterion (c) is a statement
   about genomic *context*, so a decoy needs coordinates in a host whose annotation is on
   disk.  ``decoys_v0.parquet``'s ``gc_background``/``dinuc_shuffled``/``leader_decoy`` pools
   carry no accession at all and its ``structured_rna`` pool's 2,999 accessions are Rfam
   source contigs; ``mining_pool_v0.parquet``'s 45,988 leader/trail windows sit on the TBDB
   source genomes.  **Zero** of either land on one of the 339 annotated production
   assemblies.  So the shipped-rows arm is reported ``unavailable`` **with its counts**, and
   the §9.1 leader class is instead *re-instantiated inside the annotated corpus* — 5′UTR
   windows immediately upstream of an annotated CDS and windows upstream of annotated tRNA
   genes, which is the class §9.1 names and the only place (c) is even defined.
2. **The false-FAIL arm has almost no positive denominator here, by construction.** The
   production substrate is a *negative* substrate — host-order-admissible windows masked of
   known T-boxes — so only **2 TBDB records on 1 assembly** intersect the 339 annotated
   hosts.  That spot-check is reported with its n; the *rate* D4 wants is carried by the
   positive-context arm (windows upstream of a CDS that is already in a D4 class), which
   isolates exactly the annotation-driven failure mode D4 commissions the diagnostic for.
3. **The joint a ∧ b ∧ c arm is structurally unavailable.** (b) has no backend until
   P3-15′-d and (a)'s producer was not run over the control corpus in this step.  Recorded
   as ``available: false`` with the reason, never as a computed zero.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tbox_finder.mining import annotation_fetch, gff3, synteny
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED, STATUS_UNAVAILABLE

__all__ = [
    "ADR",
    "CONTROL_MIN_MARGIN",
    "DEFAULT_FP_MANIFEST",
    "DEFAULT_GENOME_DIR",
    "DEFAULT_STATUS_TABLE",
    "EXCLUSION_REPORT",
    "FALSE_PASS_REPORT",
    "SCHEMA_VERSION",
    "STEP",
    "ProducerError",
    "SyntenyRunConfig",
    "build_status_table",
    "derive_synteny_supply_available",
    "evaluate_candidate",
    "false_pass_report",
    "exclusion_report",
    "load_clades",
    "load_status_map",
    "main",
    "merge_status_tables",
]

STEP = "P3-15'-c-ii"
ADR = "ADR-0006 D4 (criterion (c)); ADR-0005 D14 (spare rule), D15 (strand)"
SCHEMA_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_FP_MANIFEST = "data/processed/mining/round0_fp_manifest.json"
DEFAULT_GENOME_DIR = "data/interim/production_genomes"
DEFAULT_ANNOTATION_DIR = annotation_fetch.DEFAULT_ANNOTATION_DIR
DEFAULT_CLADE_TABLE = "data/processed/mining/production_genomes_v0.parquet"
DEFAULT_STATUS_TABLE = "synteny_status.json"

FALSE_PASS_REPORT = "reports/p3/synteny_false_pass.json"
EXCLUSION_REPORT = "reports/p3/synteny_exclusion_diagnostic.json"

#: The false-pass diagnostic is only informative if a *positive-context* arm separates from
#: the random-leader arm.  Equal rates mean the vocabulary decides nothing and the whole
#: report is noise — so the report grades **itself** unpowered rather than publishing a
#: false-pass rate nobody can interpret.
CONTROL_MIN_MARGIN = 0.20

#: Reasons a candidate's (c) disjunct reads ``unavailable``.  A closed vocabulary, because the
#: per-clade exclusion diagnostic is a breakdown *by* this field and a free-text reason would
#: make the breakdown uncountable.
REASON_HOST_UNANNOTATED = "host_unannotated"
REASON_CONTIG_ABSENT = "contig_absent_from_gff"
REASON_PSEUDOGENE = "first_downstream_orf_pseudogenized"
REASON_HYPOTHETICAL = "first_downstream_orf_unjudgeable"
REASON_GENOME_ABSENT = "genome_fasta_absent"
EXCLUSION_REASONS: tuple[str, ...] = (
    REASON_HOST_UNANNOTATED,
    REASON_CONTIG_ABSENT,
    REASON_PSEUDOGENE,
    REASON_HYPOTHETICAL,
    REASON_GENOME_ABSENT,
)


class ProducerError(RuntimeError):
    """The producer could not carry out the run as configured."""


@dataclass(frozen=True)
class SyntenyRunConfig:
    """Every value that changes a verdict, carried as data so it lands in the report.

    ``max_intervening_orfs`` and ``sub_threshold_orf_nt`` have **no defaults anywhere** —
    ADR-0006 D4 commissions the tandem carve-out and pins no number for it, so a default
    would decide which candidates are mined without anyone choosing it (the same discipline
    ``--strand-policy`` carries for the Stage-2 producer).
    """

    strand_policy: str
    max_intervening_orfs: int
    sub_threshold_orf_nt: int
    window_bp: int = synteny.DEFAULT_WINDOW_BP

    def as_dict(self) -> dict[str, Any]:
        return {
            "strand_policy": self.strand_policy,
            "max_intervening_orfs": self.max_intervening_orfs,
            "sub_threshold_orf_nt": self.sub_threshold_orf_nt,
            "window_bp": self.window_bp,
            "hmm_fallback_available": synteny.HMM_FALLBACK_AVAILABLE,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Corpus access — one parse per host, cached across the candidates that share it
# ═════════════════════════════════════════════════════════════════════════════
class HostAnnotation:
    """One host's CDS features + its FASTA-record ids, parsed once."""

    def __init__(self, assembly: str, *, annotation_dir: str, genome_dir: str) -> None:
        self.assembly = assembly
        gff_path = Path(annotation_dir) / annotation_fetch.destination_name(assembly)
        self.available = gff_path.exists()
        self.cds: list[gff3.CdsFeature] = []
        self.by_seqid: dict[str, list[gff3.CdsFeature]] = defaultdict(list)
        self.contig_ids: list[str] = []
        self.reason = "" if self.available else REASON_HOST_UNANNOTATED
        if self.available:
            self.cds = gff3.parse_gff3_document(gff3.read_gff3_text(gff_path))
            for cds in self.cds:
                self.by_seqid[cds.seqid].append(cds)
        fasta = Path(genome_dir) / f"{assembly}.fna"
        if fasta.exists():
            self.contig_ids = synteny.load_contig_ids(str(fasta))
        elif self.available:
            # An annotated host whose genome FASTA is missing cannot have its ``:c<ci>``
            # resolved at all; that is an unavailability, not a failure.
            self.available = False
            self.reason = REASON_GENOME_ABSENT

    def features_for(self, seqid: str) -> list[gff3.CdsFeature]:
        return self.by_seqid.get(seqid, [])


def _host_cache(annotation_dir: str, genome_dir: str) -> Callable[[str], HostAnnotation]:
    """A memoised ``assembly → HostAnnotation``: 941 candidates share 76 hosts.

    Parsing per candidate instead of per host would re-read a 900 KB GFF up to 60 times for a
    single assembly; the cache is what keeps the whole round at seconds rather than minutes.
    """
    cache: dict[str, HostAnnotation] = {}

    def get(assembly: str) -> HostAnnotation:
        if assembly not in cache:
            cache[assembly] = HostAnnotation(
                assembly, annotation_dir=annotation_dir, genome_dir=genome_dir
            )
        return cache[assembly]

    return get


# ═════════════════════════════════════════════════════════════════════════════
# Per-candidate evaluation
# ═════════════════════════════════════════════════════════════════════════════
def _strand_detail(
    host: HostAnnotation,
    *,
    seqid: str,
    strand: str,
    locus_start: int,
    locus_end: int,
    config: SyntenyRunConfig,
) -> dict[str, Any]:
    three_prime = synteny.locus_three_prime_1based(locus_start, locus_end, strand)
    resolved = synteny.resolve_downstream_gene(
        host.features_for(seqid),
        seqid=seqid,
        strand=strand,
        three_prime=three_prime,
        element_span_nt=locus_end - locus_start,
        window_bp=config.window_bp,
        max_intervening_orfs=config.max_intervening_orfs,
        sub_threshold_orf_nt=config.sub_threshold_orf_nt,
    )
    status = synteny.synteny_status(resolved, window_bp=config.window_bp)
    return {
        "status": status,
        "function_class": resolved.function_class,
        "distance_bp": resolved.distance_bp,
        "feature_id": resolved.feature_id,
        "is_pseudo": resolved.is_pseudo,
        "n_pseudo_seen": resolved.n_pseudo_seen,
        "n_unjudgeable_seen": resolved.n_unjudgeable_seen,
        "n_intervening": resolved.n_intervening,
        "carve_out_applied": resolved.carve_out_applied,
        "note": resolved.note,
    }


def evaluate_locus(
    host: HostAnnotation,
    *,
    contig_index: int,
    locus_start: int,
    locus_end: int,
    config: SyntenyRunConfig,
) -> dict[str, Any]:
    """Evaluate criterion (c) at one contig span, on both strands, then fold by policy.

    Both strands are always computed and both ride in the row: the manifest records no strand
    and ADR-0005 D15 carries orientation-ambiguous loci through on both, so a report that kept
    only the folded answer would make the alternative policy un-rederivable without a re-run.
    """
    if not host.available:
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": host.reason or REASON_HOST_UNANNOTATED,
            "seqid": None,
            "per_strand": {},
        }
    try:
        seqid = synteny.contig_seqid(host.contig_ids, contig_index)
    except synteny.SyntenyError as exc:
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": REASON_CONTIG_ABSENT,
            "seqid": None,
            "per_strand": {},
            "note": str(exc),
        }
    if seqid not in host.by_seqid:
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": REASON_CONTIG_ABSENT,
            "seqid": seqid,
            "per_strand": {},
            "note": f"{seqid} carries no CDS in {host.assembly}'s GFF",
        }

    per_strand = {
        strand: _strand_detail(
            host,
            seqid=seqid,
            strand=strand,
            locus_start=locus_start,
            locus_end=locus_end,
            config=config,
        )
        for strand in (gff3.STRAND_PLUS, gff3.STRAND_MINUS)
    }
    status = synteny.combine_strand_statuses(
        {s: d["status"] for s, d in per_strand.items()}, policy=config.strand_policy
    )
    reason = ""
    if status == STATUS_UNAVAILABLE:
        pseudo = any(d["is_pseudo"] for d in per_strand.values())
        reason = REASON_PSEUDOGENE if pseudo else REASON_HYPOTHETICAL
    return {"status": status, "reason": reason, "seqid": seqid, "per_strand": per_strand}


def evaluate_candidate(
    candidate: Mapping[str, Any], host: HostAnnotation, *, config: SyntenyRunConfig
) -> dict[str, Any]:
    """One manifest row → one status-table row."""
    accession = str(candidate["accession"])
    assembly, _, contig_token = accession.partition(":c")
    if not contig_token.isdigit():
        raise ProducerError(f"{accession}: expected '<assembly>:c<contig_index>'")
    verdict = evaluate_locus(
        host,
        contig_index=int(contig_token),
        locus_start=int(candidate["locus_start"]),
        locus_end=int(candidate["locus_end"]),
        config=config,
    )
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "accession": accession,
        "assembly": assembly,
        "contig_index": int(contig_token),
        "seqid": verdict["seqid"],
        "locus_start": int(candidate["locus_start"]),
        "locus_end": int(candidate["locus_end"]),
        "status": verdict["status"],
        "reason": verdict.get("reason", ""),
        "per_strand": verdict["per_strand"],
        "note": verdict.get("note", ""),
    }


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise ProducerError(f"{path}: expected a non-empty 'candidates' list")
    return [dict(row) for row in rows]


def run_shard(
    candidates: Sequence[Mapping[str, Any]],
    *,
    annotation_dir: str,
    genome_dir: str,
    config: SyntenyRunConfig,
) -> list[dict[str, Any]]:
    """Evaluate a shard's candidates, parsing each host's GFF exactly once."""
    get_host = _host_cache(annotation_dir, genome_dir)
    return [
        evaluate_candidate(row, get_host(str(row["accession"]).partition(":c")[0]), config=config)
        for row in candidates
    ]


def build_status_table(rows: Sequence[Mapping[str, Any]], *, config: SyntenyRunConfig) -> dict:
    """The ``candidate_id → status`` table, in the (a)-producer's shape.

    ``status`` and ``rows`` are written from the same list, and :func:`load_status_map`
    re-derives one from the other rather than trusting either — the check that caught the
    (c)-i acquisition gate comparing a report with itself.
    """
    status = {str(r["candidate_id"]): str(r["status"]) for r in rows}
    if len(status) != len(rows):
        duplicates = [
            cid for cid, n in Counter(str(r["candidate_id"]) for r in rows).items() if n > 1
        ]
        raise ProducerError(f"duplicate candidate_id in shard output: {sorted(duplicates)[:5]}")
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "config": config.as_dict(),
        "n_candidates": len(rows),
        "status_counts": dict(sorted(Counter(status.values()).items())),
        "status": status,
        "rows": list(rows),
    }


def merge_status_tables(
    table_paths: Sequence[str | Path], *, out_path: str | Path | None = None
) -> dict:
    """Concatenate shard tables; a ``candidate_id`` in two shards is an error, not a merge.

    ([[duplicate-key-merges-instead-of-colliding]] — a silent overwrite keeps every *summed*
    invariant satisfied while losing a verdict.)
    """
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for path in table_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        configs.append(payload.get("config", {}))
        rows.extend(payload.get("rows", []))
    if not configs:
        raise ProducerError("no shard tables to merge")
    if any(c != configs[0] for c in configs[1:]):
        raise ProducerError(f"shards disagree on the run config: {configs}")
    config = SyntenyRunConfig(
        strand_policy=configs[0]["strand_policy"],
        max_intervening_orfs=configs[0]["max_intervening_orfs"],
        sub_threshold_orf_nt=configs[0]["sub_threshold_orf_nt"],
        window_bp=configs[0]["window_bp"],
    )
    table = build_status_table(rows, config=config)
    if out_path is not None:
        _write(out_path, table)
    return table


def load_status_map(table_path: str | Path) -> dict[str, str]:
    """Read ``candidate_id → status``, re-deriving it from ``rows`` and refusing a mismatch.

    The stored ``status`` map is **not** trusted on its own: a table whose map and rows
    disagree is a table that can say two different things about the same candidate, and the
    disjunct that decides whether a locus is mined is not a place to let that pass.
    """
    payload = json.loads(Path(table_path).read_text(encoding="utf-8"))
    stored = payload.get("status")
    rows = payload.get("rows")
    if not isinstance(stored, Mapping) or not isinstance(rows, list):
        raise ProducerError(f"{table_path}: expected 'status' mapping and 'rows' list")
    derived = {str(r["candidate_id"]): str(r["status"]) for r in rows}
    if derived != {str(k): str(v) for k, v in stored.items()}:
        only_stored = sorted(set(stored) - set(derived))[:5]
        only_rows = sorted(set(derived) - set(stored))[:5]
        raise ProducerError(
            f"{table_path}: 'status' disagrees with 'rows' "
            f"(stored-only {only_stored}, rows-only {only_rows})"
        )
    bad = sorted({v for v in derived.values()} - {STATUS_PASSED, STATUS_FAILED, STATUS_UNAVAILABLE})
    if bad:
        raise ProducerError(f"{table_path}: unknown status value(s) {bad}")
    return derived


# ═════════════════════════════════════════════════════════════════════════════
# Clade join (for both diagnostics)
# ═════════════════════════════════════════════════════════════════════════════
def load_clades(table: str | Path = DEFAULT_CLADE_TABLE) -> dict[str, str]:
    """``assembly_accession → GTDB phylum`` for the production substrate.

    Imported lazily: the clade join is the one thing in this module that needs pandas/pyarrow,
    and the predicate itself must stay importable in the stdlib-only environments the (c)-i
    reader was built for.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ProducerError(f"the clade join needs pandas: {exc}") from exc
    frame = pd.read_parquet(table, columns=["assembly_accession", "phylum"])
    pairs = zip(frame["assembly_accession"], frame["phylum"], strict=True)
    return {str(a): str(p) for a, p in pairs}


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when the denominator is empty — never a 0.0 standing in for 'no data'."""
    return None if denominator <= 0 else numerator / denominator


# ═════════════════════════════════════════════════════════════════════════════
# D4 diagnostic 1 — the false-pass rate
# ═════════════════════════════════════════════════════════════════════════════
def _window_lengths(candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(int(c["locus_end"]) - int(c["locus_start"]) for c in candidates)


def _draw_random_leaders(
    host: HostAnnotation,
    *,
    lengths: Sequence[int],
    n: int,
    rng: random.Random,
    forbidden: Sequence[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    """Random contig windows in this host, length-matched to the candidates, mask-respecting.

    ``forbidden`` are the host's own mined candidate spans; a control leader that overlaps one
    would be measuring the candidates again rather than the background.  The union-prior mask
    is **fail-open on this substrate** (ADR-0005 A8: every accession is off-catalogue), which
    is why this is a candidate-overlap exclusion and is reported as such, not as a clean-pool
    claim.
    """
    contigs = [s for s in host.by_seqid if host.by_seqid[s]]
    if not contigs:
        return []
    extents = {s: max(max(c.end for c in host.by_seqid[s]), 1) for s in contigs}
    blocked = defaultdict(list)
    for seqid, start, end in forbidden:
        blocked[seqid].append((start, end))
    out: list[tuple[str, int, int]] = []
    for _ in range(n * 4):
        if len(out) >= n:
            break
        seqid = rng.choice(contigs)
        span = rng.choice(lengths)
        extent = extents[seqid]
        if extent <= span + 2:
            continue
        start = rng.randrange(0, extent - span)
        end = start + span
        if any(start < b_end and b_start < end for b_start, b_end in blocked.get(seqid, ())):
            continue
        out.append((seqid, start, end))
    return out


def _upstream_window(
    feature: gff3.GffFeature | gff3.CdsFeature, *, span: int
) -> tuple[str, int, int, str] | None:
    """The window immediately 5′ of a feature, as a 0-based half-open contig span.

    This is the §9.1 *5′UTR / leader* decoy class re-instantiated in the annotated corpus:
    by construction its downstream same-strand CDS start sits at distance ≈ 0, so the pass
    rate it measures is **entirely** the gene-identity requirement's doing — which is exactly
    what D4 says the criterion's specificity rests on.
    """
    if feature.strand == gff3.STRAND_PLUS:
        end0 = feature.start - 1  # 1-based inclusive start → 0-based half-open end
        start0 = end0 - span
        if start0 < 0:
            return None
        return (feature.seqid, start0, end0, gff3.STRAND_PLUS)
    if feature.strand == gff3.STRAND_MINUS:
        start0 = feature.end  # 0-based half-open start just past the 1-based inclusive end
        return (feature.seqid, start0, start0 + span, gff3.STRAND_MINUS)
    return None


def _evaluate_windows(
    host: HostAnnotation,
    windows: Iterable[tuple[str, int, int]],
    *,
    config: SyntenyRunConfig,
) -> Counter:
    counts: Counter = Counter()
    for seqid, start, end in windows:
        per_strand = {
            strand: _strand_detail(
                host,
                seqid=seqid,
                strand=strand,
                locus_start=start,
                locus_end=end,
                config=config,
            )
            for strand in (gff3.STRAND_PLUS, gff3.STRAND_MINUS)
        }
        counts[
            synteny.combine_strand_statuses(
                {s: d["status"] for s, d in per_strand.items()}, policy=config.strand_policy
            )
        ] += 1
    return counts


def _evaluate_oriented(
    host: HostAnnotation,
    windows: Iterable[tuple[str, int, int, str]],
    *,
    config: SyntenyRunConfig,
) -> Counter:
    """Evaluate windows whose orientation IS known, on that strand only.

    The 5′UTR and tRNA-adjacent arms know their strand (it is the feature's), so folding them
    over both strands would let the *other* strand's genes decide a decoy's verdict and quietly
    inflate the false-pass rate this arm exists to measure.
    """
    counts: Counter = Counter()
    for seqid, start, end, strand in windows:
        detail = _strand_detail(
            host, seqid=seqid, strand=strand, locus_start=start, locus_end=end, config=config
        )
        counts[detail["status"]] += 1
    return counts


def _arm(counts: Counter) -> dict[str, Any]:
    total = sum(counts.values())
    return {
        "n": total,
        "status_counts": dict(sorted(counts.items())),
        "false_pass_rate": _rate(counts.get(STATUS_PASSED, 0), total),
    }


def shipped_decoy_availability(
    decoys_parquet: str | Path, mining_pool_parquet: str | Path, annotated: set[str]
) -> dict[str, Any]:
    """How many shipped §9.1 rows could carry the (c) arm — measured, not assumed.

    ([[clauses-must-guard-emptiness]]: an arm that silently vanishes reads as an arm that
    found nothing.  It is reported ``available: false`` **with** the numbers that make it so.)
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return {"available": False, "reason": f"pandas unavailable: {exc}"}
    detail: dict[str, Any] = {}
    usable = 0
    for label, path in (("decoys_v0", decoys_parquet), ("mining_pool_v0", mining_pool_parquet)):
        if not Path(path).exists():
            detail[label] = {"present": False}
            continue
        frame = pd.read_parquet(path, columns=["accession", "locus_start"])
        with_coords = int(frame["accession"].notna().sum())
        hosts = frame["accession"].dropna().astype(str).str.split(":").str[0]
        on_annotated = int(hosts.isin(annotated).sum())
        usable += on_annotated
        detail[label] = {
            "present": True,
            "n_rows": int(len(frame)),
            "n_with_accession": with_coords,
            "n_on_annotated_production_host": on_annotated,
        }
    return {
        "available": usable > 0,
        "n_usable_rows": usable,
        "reason": (
            ""
            if usable > 0
            else "no shipped §9.1 decoy row carries coordinates on one of the 339 annotated "
            "production assemblies, so criterion (c) has no genomic context to evaluate"
        ),
        "detail": detail,
    }


def false_pass_report(
    candidates: Sequence[Mapping[str, Any]],
    *,
    annotation_dir: str,
    genome_dir: str,
    config: SyntenyRunConfig,
    clades: Mapping[str, str],
    n_per_host: int,
    seed: int,
    decoys_parquet: str | Path,
    mining_pool_parquet: str | Path,
) -> dict[str, Any]:
    """D4's false-pass arm: clade-matched random leaders + the §9.1 leader class.

    The report grades **itself**: a positive-context arm (windows upstream of a CDS already in
    one of D4's classes) must separate from the random-leader arm by
    :data:`CONTROL_MIN_MARGIN`, or ``powered`` is ``False`` and the false-pass rates are
    published with that verdict attached.  A control that cannot fail is not a control
    ([[control-matchedness-must-be-asserted]]).
    """
    rng = random.Random(seed)
    lengths = _window_lengths(candidates)
    get_host = _host_cache(annotation_dir, genome_dir)

    spans_by_host: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in candidates:
        assembly, _, ci = str(row["accession"]).partition(":c")
        spans_by_host[assembly].append((int(ci), int(row["locus_start"]), int(row["locus_end"])))

    random_leaders: Counter = Counter()
    utr_decoys: Counter = Counter()
    trna_decoys: Counter = Counter()
    positive_context: Counter = Counter()
    per_clade: dict[str, Counter] = defaultdict(Counter)
    hosts_used = 0

    for assembly, spans in sorted(spans_by_host.items()):
        host = get_host(assembly)
        if not host.available:
            continue
        hosts_used += 1
        forbidden = []
        for ci, start, end in spans:
            try:
                forbidden.append((synteny.contig_seqid(host.contig_ids, ci), start, end))
            except synteny.SyntenyError:
                continue
        windows = _draw_random_leaders(
            host, lengths=lengths, n=n_per_host, rng=rng, forbidden=forbidden
        )
        counts = _evaluate_windows(host, windows, config=config)
        random_leaders += counts
        per_clade[clades.get(assembly, "unassigned")] += counts

        # §9.1 leader class, re-instantiated: 5′UTR windows upstream of annotated CDS.
        sample = host.cds if len(host.cds) <= n_per_host else rng.sample(host.cds, n_per_host)
        utr_windows = [
            w for w in (_upstream_window(c, span=rng.choice(lengths)) for c in sample) if w
        ]
        utr_decoys += _evaluate_oriented(host, utr_windows, config=config)

        # …and windows upstream of annotated tRNA genes — §9.1's tRNA-adjacent sub-class.
        trna_features = list(
            gff3.iter_gff3_features(
                gff3.read_gff3_text(
                    Path(annotation_dir) / annotation_fetch.destination_name(assembly)
                ).splitlines(),
                types={"tRNA"},
            )
        )
        if trna_features:
            picked = (
                trna_features
                if len(trna_features) <= n_per_host
                else rng.sample(trna_features, n_per_host)
            )
            trna_windows = [
                w for w in (_upstream_window(t, span=rng.choice(lengths)) for t in picked) if w
            ]
            trna_decoys += _evaluate_oriented(host, trna_windows, config=config)

        # The power arm: windows upstream of a CDS that IS in one of D4's four classes.
        positives = [
            c
            for c in host.cds
            if synteny.classify_gene_identity(
                gff3.gene_identity_text(c), gene_symbols=synteny._gene_symbols(c)
            )
            in synteny.PASSING_CLASSES
        ]
        if positives:
            picked_pos = (
                positives if len(positives) <= n_per_host else rng.sample(positives, n_per_host)
            )
            pos_windows = [
                w for w in (_upstream_window(c, span=rng.choice(lengths)) for c in picked_pos) if w
            ]
            positive_context += _evaluate_oriented(host, pos_windows, config=config)

    background = _arm(random_leaders)
    positive = _arm(positive_context)
    margin = (
        None
        if background["false_pass_rate"] is None or positive["false_pass_rate"] is None
        else positive["false_pass_rate"] - background["false_pass_rate"]
    )
    powered = margin is not None and margin >= CONTROL_MIN_MARGIN

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "config": config.as_dict(),
        "seed": seed,
        "n_per_host": n_per_host,
        "n_hosts": hosts_used,
        "arms": {
            "clade_matched_random_leaders": background,
            "nine_one_five_prime_utr_decoys": _arm(utr_decoys),
            "nine_one_trna_adjacent_decoys": _arm(trna_decoys),
            "shipped_nine_one_decoy_rows": shipped_decoy_availability(
                decoys_parquet, mining_pool_parquet, set(clades) & _annotated_set(annotation_dir)
            ),
        },
        "positive_context_control": positive,
        "control": {
            "powered": powered,
            "margin": margin,
            "min_margin": CONTROL_MIN_MARGIN,
            "verdict": (
                "positive-context and random-leader arms separate; the false-pass rates below "
                "are interpretable"
                if powered
                else "the arms do NOT separate by the required margin — treat the false-pass "
                "rates as uninterpretable, not as evidence of a clean rule"
            ),
        },
        "per_clade_random_leader": {
            clade: _arm(counts) for clade, counts in sorted(per_clade.items())
        },
        "joint_abc": {
            "available": False,
            "reason": (
                "criterion (b) has no backend until P3-15′-d, and criterion (a)'s producer was "
                "not run over this control corpus in this step; the joint a∧b∧c false-pass "
                "rate D4 also asks for is therefore withheld rather than computed from two "
                "unavailable disjuncts"
            ),
        },
    }


def _annotated_set(annotation_dir: str) -> set[str]:
    directory = Path(annotation_dir)
    if not directory.is_dir():
        return set()
    return {p.name.replace(".gff.gz", "") for p in directory.glob("*.gff.gz")}


# ═════════════════════════════════════════════════════════════════════════════
# D4 diagnostic 2 — the symmetric false-FAIL / per-clade exclusion / pseudogene arm
# ═════════════════════════════════════════════════════════════════════════════
def exclusion_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    clades: Mapping[str, str],
    config: SyntenyRunConfig,
    annotation_dir: str,
    genome_dir: str,
    false_fail_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """D4's symmetric arm: what (c) *cannot decide*, and where that falls by clade.

    The headline is the **per-clade exclusion rate** — the fraction of candidates whose (c)
    disjunct is ``unavailable``.  It is the annotation-driven recall cap D4 commissions this
    for: an unavailable disjunct spares its candidate, so a clade with a high exclusion rate
    contributes structurally fewer hard negatives, and the cap is a property of the clade's
    annotation completeness rather than of its biology.
    """
    by_clade: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    pseudo_seen = 0
    unjudgeable_seen = 0
    carve_out_used = 0
    distances: list[int] = []
    class_counts: Counter = Counter()

    for row in rows:
        clade = clades.get(str(row.get("assembly", "")), "unassigned")
        by_clade[clade][str(row["status"])] += 1
        if row["status"] == STATUS_UNAVAILABLE:
            reasons[clade][str(row.get("reason") or REASON_HOST_UNANNOTATED)] += 1
        for detail in (row.get("per_strand") or {}).values():
            pseudo_seen += int(detail.get("n_pseudo_seen") or 0)
            unjudgeable_seen += int(detail.get("n_unjudgeable_seen") or 0)
            if detail.get("carve_out_applied"):
                carve_out_used += 1
            if detail.get("function_class") in synteny.PASSING_CLASSES:
                class_counts[detail["function_class"]] += 1
                if detail.get("distance_bp") is not None:
                    distances.append(int(detail["distance_bp"]))

    total = len(rows)
    unavailable = sum(1 for r in rows if r["status"] == STATUS_UNAVAILABLE)
    distances.sort()

    def _pct(p: float) -> int | None:
        if not distances:
            return None
        return distances[min(len(distances) - 1, int(round(p * (len(distances) - 1))))]

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "config": config.as_dict(),
        "n_candidates": total,
        "n_unavailable": unavailable,
        "exclusion_rate": _rate(unavailable, total),
        "exclusion_reason_totals": dict(
            sorted(
                Counter(
                    str(r.get("reason") or "") for r in rows if r["status"] == STATUS_UNAVAILABLE
                ).items()
            )
        ),
        "per_clade": {
            clade: {
                "n": sum(counts.values()),
                "status_counts": dict(sorted(counts.items())),
                "exclusion_rate": _rate(counts.get(STATUS_UNAVAILABLE, 0), sum(counts.values())),
                "reasons": dict(sorted(reasons[clade].items())),
            }
            for clade, counts in sorted(by_clade.items())
        },
        "pseudogene_diagnostic": {
            "n_pseudogenized_orfs_encountered": pseudo_seen,
            "n_unjudgeable_orfs_encountered": unjudgeable_seen,
            "hmm_fallback_available": synteny.HMM_FALLBACK_AVAILABLE,
            "consequence": (
                "D4 routes a pseudogenized or hypothetical first-downstream ORF to targeted "
                "Pfam/KO profiles.  That profile database is an unmet §10.2 acquisition, so "
                "these loci resolve 'unavailable' and ADR-0005 D14 SPARES them — the "
                "fail-closed direction.  The count above is the size of the recall the "
                "missing fallback costs."
            ),
        },
        "tandem_carve_out": {
            "n_strand_evaluations_using_carve_out": carve_out_used,
            "max_intervening_orfs": config.max_intervening_orfs,
            "sub_threshold_orf_nt": config.sub_threshold_orf_nt,
        },
        "passing_distance_sensitivity": {
            "n": len(distances),
            "p50_bp": _pct(0.50),
            "p95_bp": _pct(0.95),
            "p99_bp": _pct(0.99),
            "max_bp": distances[-1] if distances else None,
            "window_bp": config.window_bp,
            "note": (
                "D4 asks for the empirical p95/p99 as a sensitivity check on the 500 bp pad; "
                "these are measured on the passing candidates of THIS corpus, not on "
                "catalogued Firmicutes T-boxes"
            ),
        },
        "passing_class_counts": dict(sorted(class_counts.items())),
        "false_fail_probe": false_fail_probe
        or {
            "available": False,
            "reason": (
                "the production substrate is a NEGATIVE substrate (host-order-admissible "
                "windows masked of known T-boxes), so it carries essentially no catalogued "
                "T-box loci to measure a false-FAIL RATE on; the annotation-driven failure "
                "mode is carried by the positive-context arm of the false-pass report"
            ),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Supply derivation — the constant has to prove itself (P3-15′-a/-b's discipline)
# ═════════════════════════════════════════════════════════════════════════════
#: Every clause reads **git-tracked, non-DVC** evidence except the annotation corpus itself,
#: which is reported separately so CI (where DVC data is absent) and the laptop disagree
#: loudly rather than silently.
SUPPLY_CLAUSES: tuple[tuple[str, str], ...] = (
    ("predicate_module_present", "src/tbox_finder/mining/synteny.py"),
    ("producer_module_present", "src/tbox_finder/mining/synteny_producer.py"),
    ("gff3_reader_present", "src/tbox_finder/mining/gff3.py"),
    ("supply_report_present", annotation_fetch.DEFAULT_SUPPLY_REPORT),
    ("fetch_report_present", annotation_fetch.DEFAULT_FETCH_REPORT),
    ("false_pass_report_present", FALSE_PASS_REPORT),
    ("exclusion_report_present", EXCLUSION_REPORT),
)


def _json_or_none(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def derive_synteny_supply_available(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Can *this checkout* evidence a criterion-(c) supply?  Fail-closed, clause by clause.

    Mirrors :func:`tbox_finder.mining.mine_round.derive_msa_supply_available` and
    :func:`tbox_finder.mining.stage2_producer.derive_stage2_supply_available`.  A missing or
    false clause makes the whole thing ``False``; every clause is re-derived from disk on each
    call so the constant can never drift from what is actually here.

    ⚠ **Every clause reads git-tracked evidence, and that constraint is load-bearing.** The
    obvious clause — *"the 339 GFFs are in ``data/interim/production_annotations``"* — is
    DVC-tracked, so it is ``False`` in CI and in any fresh clone, which would make the
    derivation disagree with the constant in exactly the environments that must agree.  The
    corpus is therefore evidenced through the **committed acquisition report** (which records
    339/339 md5-verified) and reported separately as an observation.  Same trap, same fix, as
    ``derive_stage2_supply_available``'s absent checkpoint clause.

    The two committed diagnostic reports are clauses in their own right, and the false-pass
    report's own ``powered`` verdict is a third: D4 asks for the symmetric diagnostics *as
    part of* the criterion, so a checkout carrying the predicate but no measurement — or a
    measurement that graded **itself** uninterpretable — has not evidenced the supply D4
    specifies.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    clauses = {name: (root / rel).exists() for name, rel in SUPPLY_CLAUSES}

    fetch = _json_or_none(root / annotation_fetch.DEFAULT_FETCH_REPORT)
    counts = (fetch or {}).get("status_counts") or {}
    n_ok = int(counts.get(annotation_fetch.STATUS_OK, 0)) if isinstance(counts, Mapping) else 0
    clauses["acquisition_report_records_a_corpus"] = n_ok > 0

    false_pass = _json_or_none(root / FALSE_PASS_REPORT)
    clauses["false_pass_control_powered"] = bool(
        ((false_pass or {}).get("control") or {}).get("powered") is True
    )

    annotated = _annotated_set(str(root / DEFAULT_ANNOTATION_DIR))
    reasons = [f"{name} is missing or false" for name, ok in sorted(clauses.items()) if not ok]
    return {
        "available": all(clauses.values()),
        "clauses": dict(sorted(clauses.items())),
        "n_annotated_hosts_in_acquisition_report": n_ok,
        "n_annotated_hosts_on_disk": len(annotated),
        "reasons": reasons,
        "repo_root": str(root),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def _provenance(entry: str, inputs: Sequence[str | Path], extra: Mapping[str, Any]) -> dict:
    """CLAUDE.md §11 provenance for a committed report.

    ``inputs`` only — never the report being written.  ``build_provenance`` hashes whatever it
    is given, and an output that does not exist yet raises at the END of the run that would
    have produced it ([[build-provenance-hashes-its-outputs]]).
    """
    from tbox_finder.provenance import build_provenance

    return build_provenance(
        rule=f"synteny_producer::{entry}",
        script="src/tbox_finder/mining/synteny_producer.py",
        inputs=[str(i) for i in inputs if Path(i).exists()],
        adr=ADR,
        extra=dict(extra),
    )


def _write(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ProducerError(f"{path}: cannot write ({exc})") from exc
    return target


def _config_from(args: argparse.Namespace) -> SyntenyRunConfig:
    return SyntenyRunConfig(
        strand_policy=args.strand_policy,
        max_intervening_orfs=args.max_intervening_orfs,
        sub_threshold_orf_nt=args.sub_threshold_orf_nt,
        window_bp=args.window_bp,
    )


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strand-policy",
        choices=synteny.STRAND_POLICIES,
        required=True,
        help=(
            "REQUIRED, no default: the manifest records no strand, so 'the' downstream gene "
            "is undefined until the round says which orientation it means"
        ),
    )
    parser.add_argument(
        "--max-intervening-orfs",
        type=int,
        required=True,
        help="REQUIRED, no default: D4's tandem carve-out hop limit (ADR pins no number)",
    )
    parser.add_argument(
        "--sub-threshold-orf-nt",
        type=int,
        required=True,
        help="REQUIRED, no default: below this CDS length an ORF counts as intervening",
    )
    parser.add_argument("--window-bp", type=int, default=synteny.DEFAULT_WINDOW_BP)
    parser.add_argument("--manifest", default=DEFAULT_FP_MANIFEST)
    parser.add_argument("--annotation-dir", default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--genome-dir", default=DEFAULT_GENOME_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P3-15′-c-ii — criterion (c) synteny producer + D4's symmetric diagnostics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-shard", help="evaluate a shard (or the whole manifest)")
    _add_config_flags(run)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--n-shards", type=int, default=1)
    run.add_argument("--out", required=True)

    merge = subparsers.add_parser("merge", help="concatenate shard tables into the status table")
    merge.add_argument("--table", action="append", required=True)
    merge.add_argument("--out", required=True)

    diag = subparsers.add_parser("diagnostics", help="D4's two symmetric diagnostic reports")
    _add_config_flags(diag)
    diag.add_argument("--status-table", required=True)
    diag.add_argument("--clade-table", default=DEFAULT_CLADE_TABLE)
    diag.add_argument("--n-per-host", type=int, default=200)
    diag.add_argument("--seed", type=int, default=42)
    diag.add_argument("--decoys", default="data/processed/negatives/decoys_v0.parquet")
    diag.add_argument("--mining-pool", default="data/processed/negatives/mining_pool_v0.parquet")
    diag.add_argument("--false-pass-out", default=FALSE_PASS_REPORT)
    diag.add_argument("--exclusion-out", default=EXCLUSION_REPORT)

    subparsers.add_parser("derive-supply", help="print the supply derivation for this checkout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run-shard":
            candidates = read_manifest(args.manifest)
            if args.n_shards < 1 or not 0 <= args.shard_index < args.n_shards:
                raise ProducerError(f"shard {args.shard_index} of {args.n_shards} is out of range")
            shard = candidates[args.shard_index :: args.n_shards]
            rows = run_shard(
                shard,
                annotation_dir=args.annotation_dir,
                genome_dir=args.genome_dir,
                config=_config_from(args),
            )
            table = build_status_table(rows, config=_config_from(args))
            _write(args.out, table)
            print(json.dumps({"out": args.out, "status_counts": table["status_counts"]}))
            return 0

        if args.command == "merge":
            table = merge_status_tables(args.table, out_path=args.out)
            print(json.dumps({"out": args.out, "status_counts": table["status_counts"]}))
            return 0

        if args.command == "diagnostics":
            config = _config_from(args)
            rows = json.loads(Path(args.status_table).read_text(encoding="utf-8"))["rows"]
            load_status_map(args.status_table)  # refuse a table that disagrees with itself
            clades = load_clades(args.clade_table)
            candidates = read_manifest(args.manifest)
            fp = false_pass_report(
                candidates,
                annotation_dir=args.annotation_dir,
                genome_dir=args.genome_dir,
                config=config,
                clades=clades,
                n_per_host=args.n_per_host,
                seed=args.seed,
                decoys_parquet=args.decoys,
                mining_pool_parquet=args.mining_pool,
            )
            fp["provenance"] = _provenance(
                "diagnostics.false_pass",
                [args.manifest, args.clade_table, args.decoys, args.mining_pool],
                {
                    "step": STEP,
                    "seed": args.seed,
                    "n_per_host": args.n_per_host,
                    **config.as_dict(),
                },
            )
            _write(args.false_pass_out, fp)
            ex = exclusion_report(
                rows,
                clades=clades,
                config=config,
                annotation_dir=args.annotation_dir,
                genome_dir=args.genome_dir,
            )
            ex["provenance"] = _provenance(
                "diagnostics.exclusion",
                [args.status_table, args.clade_table],
                {"step": STEP, **config.as_dict()},
            )
            _write(args.exclusion_out, ex)
            print(
                json.dumps(
                    {
                        "false_pass_out": args.false_pass_out,
                        "exclusion_out": args.exclusion_out,
                        "powered": fp["control"]["powered"],
                        "exclusion_rate": ex["exclusion_rate"],
                    }
                )
            )
            return 0 if fp["control"]["powered"] else 3

        if args.command == "derive-supply":
            print(json.dumps(derive_synteny_supply_available(), indent=2, sort_keys=True))
            return 0
    except (ProducerError, synteny.SyntenyError, gff3.Gff3Error) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
