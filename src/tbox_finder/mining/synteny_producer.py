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
    "DEFAULT_ANNOTATION_DIR",
    "DEFAULT_GENOME_DIR",
    "DEFAULT_STATUS_TABLE",
    "EXCLUSION_REPORT",
    "FALSE_PASS_REPORT",
    "SCHEMA_VERSION",
    "STEP",
    "ProducerError",
    "SyntenyRunConfig",
    "as_control_arm",
    "build_status_table",
    "derive_synteny_supply_available",
    "evaluate_candidate",
    "false_pass_report",
    "exclusion_report",
    "load_clades",
    "load_status_map",
    "validate_status_payload",
    "strict_subsample",
    "utr_arm_windows",
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
#: A host whose GFF is present but truncated / undeclared / unreadable.  Distinct from
#: ``host_unannotated`` on purpose: a corrupt corpus must not hide inside a missing one.
REASON_ANNOTATION_UNREADABLE = "host_annotation_unreadable"
#: A host whose genome FASTA is present but empty / unreadable / has a header with no id.
#: Distinct from ``genome_fasta_absent`` for the same reason the annotation pair is split:
#: a corrupt file and a missing one must not be indistinguishable in the diagnostic.
REASON_GENOME_UNREADABLE = "genome_fasta_unreadable"
EXCLUSION_REASONS: tuple[str, ...] = (
    REASON_HOST_UNANNOTATED,
    REASON_CONTIG_ABSENT,
    REASON_PSEUDOGENE,
    REASON_HYPOTHETICAL,
    REASON_GENOME_ABSENT,
    REASON_ANNOTATION_UNREADABLE,
    REASON_GENOME_UNREADABLE,
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
        self.trna: list[gff3.GffFeature] = []
        self.by_seqid: dict[str, list[gff3.CdsFeature]] = defaultdict(list)
        self.contig_ids: list[str] = []
        self.reason = "" if self.available else REASON_HOST_UNANNOTATED
        self.note = ""
        if self.available:
            # ⚠ One truncated, undeclared or unreadable GFF among the 339 would otherwise
            # abort all 941 verdicts.  It is a per-HOST unavailability — which ADR-0005 D14
            # spares — not a run-level fault, and the exclusion diagnostic counts it under its
            # own reason so a corrupt corpus cannot hide inside `host_unannotated`.
            # (`-c-i` recorded the same defect: a census read sitting above the `except
            # OSError` written for it aborted all 339 on one bad file.)
            try:
                text = gff3.read_gff3_text(gff_path)
                self.cds = gff3.parse_gff3_document(text)
                # One parse per host covers BOTH arms.  Re-reading the file for the tRNA
                # windows defeated the cache that is the whole reason the round runs in
                # seconds — and read a *second* snapshot of a file the CDS arm had already
                # committed to.
                self.trna = list(gff3.iter_gff3_features(text.splitlines(), types={"tRNA"}))
            except (gff3.Gff3Error, OSError, ValueError) as exc:
                self.available = False
                self.reason = REASON_ANNOTATION_UNREADABLE
                self.note = f"{gff_path.name}: {exc}"
                return
            for cds in self.cds:
                self.by_seqid[cds.seqid].append(cds)
        fasta = Path(genome_dir) / f"{assembly}.fna"
        if fasta.exists():
            # ⚠ Guarded exactly as the GFF read above is.  ``load_contig_ids`` raises
            # ``SyntenyError`` for an empty FASTA or a header with no record id, and ``open``
            # raises ``OSError`` for an unreadable file — so one bad ``.fna`` among the 660
            # would take the whole round down instead of costing its own host.  Same defect,
            # sibling read; round 12 fixed only the annotation half.
            try:
                self.contig_ids = synteny.load_contig_ids(str(fasta))
            except (synteny.SyntenyError, OSError, UnicodeDecodeError) as exc:
                self.available = False
                self.reason = REASON_GENOME_UNREADABLE
                self.note = f"{fasta.name}: {exc}"
                return
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
        "decision_distance_bp": resolved.decision_distance_bp,
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
            "note": host.note,
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
        # Only the strand(s) the policy actually folded may explain the verdict.  Scanning
        # both under ``plus``/``minus`` lets the strand that decided nothing supply the
        # reason, and the per-clade exclusion diagnostic is a breakdown *by* that field.
        if config.strand_policy == "both":
            # Only the strands whose OWN status is unavailable caused this verdict; a strand
            # that passed or failed cannot explain why the fold came out unavailable.
            deciding = [
                d for d in per_strand.values() if d["status"] == STATUS_UNAVAILABLE
            ] or list(per_strand.values())
        else:
            selected = gff3.STRAND_PLUS if config.strand_policy == "plus" else gff3.STRAND_MINUS
            deciding = [per_strand[selected]]
        pseudo = any(d["is_pseudo"] for d in deciding)
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
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProducerError(f"{path}: candidate {index} is {type(row).__name__}, not an object")
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
    required = {
        "strand_policy",
        "max_intervening_orfs",
        "sub_threshold_orf_nt",
        "window_bp",
        "hmm_fallback_available",
    }
    for path in table_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ProducerError(f"{path}: expected a shard table object")
        config_block = payload.get("config")
        shard_rows = payload.get("rows")
        if not isinstance(config_block, Mapping):
            raise ProducerError(f"{path}: shard table has no 'config' block")
        missing = sorted(required - set(config_block))
        if missing:
            raise ProducerError(f"{path}: shard config is missing {missing}")
        if not isinstance(shard_rows, list) or not shard_rows:
            # A shard that contributes nothing is the silent version of a dropped shard: the
            # merge would succeed, the coverage would be short, and every absent candidate
            # would read ``unavailable`` — spared rather than mined, but for the wrong reason.
            raise ProducerError(f"{path}: shard table contributes no rows")
        configs.append(dict(config_block))
        rows.extend(shard_rows)
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
    # ⚠ ``config.as_dict()`` re-reads the LIVE module constant, so a merge run on a checkout
    # where ``HMM_FALLBACK_AVAILABLE`` had flipped would silently rewrite what the shards
    # actually ran under.  The shard-recorded value wins, and a shard disagreeing with its
    # siblings has already been refused above.
    recorded = configs[0].get("hmm_fallback_available")
    if recorded is not None:
        table["config"]["hmm_fallback_available"] = recorded
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
    return validate_status_payload(payload, label=str(table_path))


def validate_status_payload(payload: Any, *, label: str) -> dict[str, str]:
    """Validate an already-parsed status table and return its ``candidate_id → status`` map.

    Split out from :func:`load_status_map` so a caller that has the payload in hand does not
    have to re-read the file: validating one snapshot and then building a report from a second
    read leaves a window for the two to differ.
    """
    stored = payload.get("status") if isinstance(payload, Mapping) else None
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(stored, Mapping) or not isinstance(rows, list):
        raise ProducerError(f"{label}: expected 'status' mapping and 'rows' list")
    for index, r in enumerate(rows):
        if not isinstance(r, Mapping) or "candidate_id" not in r or "status" not in r:
            # A bare KeyError here is not this function's contract; every other malformed
            # table shape refuses as a ProducerError naming the file.
            raise ProducerError(f"{label}: row {index} lacks 'candidate_id' or 'status'")
    # ⚠ A dict comprehension COLLAPSES a repeated candidate_id to its last occurrence, so two
    # rows saying ``passed`` and ``failed`` for one candidate would silently pick one and the
    # status↔rows cross-check below would still agree with itself.  ``build_status_table``
    # refuses duplicates at write time; the reader has to as well, because a table can reach
    # this function without having been written by it
    # ([[duplicate-key-merges-instead-of-colliding]], on the read path this time).
    ids = [str(r["candidate_id"]) for r in rows]
    repeated = sorted({cid for cid, n in Counter(ids).items() if n > 1})
    if repeated:
        raise ProducerError(f"{label}: duplicate candidate_id in 'rows': {repeated[:5]}")
    derived = {cid: str(r["status"]) for cid, r in zip(ids, rows, strict=True)}
    if derived != {str(k): str(v) for k, v in stored.items()}:
        only_stored = sorted(set(stored) - set(derived))[:5]
        only_rows = sorted(set(derived) - set(stored))[:5]
        raise ProducerError(
            f"{label}: 'status' disagrees with 'rows' "
            f"(stored-only {only_stored}, rows-only {only_rows})"
        )
    bad = sorted({v for v in derived.values()} - {STATUS_PASSED, STATUS_FAILED, STATUS_UNAVAILABLE})
    if bad:
        raise ProducerError(f"{label}: unknown status value(s) {bad}")
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
    pairs = list(zip(frame["assembly_accession"], frame["phylum"], strict=True))
    # ⚠ Same duplicate-key rule this module applies to the status table.  A repeated
    # ``assembly_accession`` collapses to its last row, and a repeat carrying a *different*
    # phylum silently reassigns every candidate on that host — and the per-clade exclusion
    # rate is the headline of the whole exclusion diagnostic.
    repeated = sorted({str(a) for a, n in Counter(str(a) for a, _p in pairs).items() if n > 1})
    if repeated:
        raise ProducerError(f"{table}: duplicate assembly_accession: {repeated[:5]}")
    return {str(a): str(p) for a, p in pairs}


def _exclusion_reason(row: Mapping[str, Any]) -> str:
    """The one derivation both exclusion breakdowns read.

    ⚠ They used to disagree: the totals fell back to ``""`` and the per-clade breakdown to
    ``host_unannotated`` for the *same* row, so a row with an empty reason was counted under
    two different keys and the closed :data:`EXCLUSION_REASONS` vocabulary — which the
    per-clade assertion checks against — was broken in the totals only.
    """
    reason = str(row.get("reason") or "") or REASON_HOST_UNANNOTATED
    if reason not in EXCLUSION_REASONS:
        raise ProducerError(f"unknown exclusion reason {reason!r}; expected {EXCLUSION_REASONS}")
    return reason


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
    feature: gff3.GffFeature | gff3.CdsFeature,
    *,
    span: int,
    contig_extent: int | None = None,
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
        if contig_extent is not None and start0 + span > contig_extent:
            return None
        return (feature.seqid, start0, start0 + span, gff3.STRAND_MINUS)
    return None


def strict_subsample(sample: Sequence[gff3.CdsFeature]) -> list[gff3.CdsFeature]:
    """The §9.1 5′UTR arm's strict reading: ``sample`` minus the CDS naming a D4 class.

    ⚠ A **filter of the given sample**, never a second draw from the population.  Drawing the
    two arms independently — each capped at ``n_per_host`` — made the "subset" larger than its
    source (9,087 rows against 9,065), and a subset that outnumbers its population is not a
    second reading of one arm, it is a second arm.  Kept as a named function so the subset
    property is testable without regenerating a report.
    """
    return [
        cds
        for cds in sample
        if synteny.classify_gene_identity(
            gff3.gene_identity_text(cds), gene_symbols=synteny.gene_symbols(cds)
        )
        not in synteny.PASSING_CLASSES
    ]


def utr_arm_windows(
    sample: Sequence[gff3.CdsFeature],
    *,
    spans: Sequence[int],
    extents: Mapping[str, int],
) -> tuple[list[tuple[str, int, int, str]], list[tuple[str, int, int, str]]]:
    """The §9.1 5′UTR arm's wide and strict window lists, from **one** draw.

    Returns ``(wide, strict)`` where ``strict`` is a sublist of ``wide`` — the same window
    objects, filtered to those whose own CDS names none of D4's classes.  Extracted as a named
    function because the property that matters is a relation *between* the two lists, and the
    committed-report assertion that was supposed to guard it stayed green under sabotage: a
    test that reads a committed artifact validates the artifact, never the code that made it.
    """
    if len(spans) != len(sample):
        raise ProducerError(f"need one span per CDS, got {len(spans)} for {len(sample)}")
    pairs = [
        (cds, window)
        for cds, window in (
            (cds, _upstream_window(cds, span=span, contig_extent=extents.get(cds.seqid)))
            for cds, span in zip(sample, spans, strict=True)
        )
        if window
    ]
    keep = {id(cds) for cds in strict_subsample(sample)}
    return [w for _c, w in pairs], [w for c, w in pairs if id(c) in keep]


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


def as_control_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Re-key a decoy-shaped arm as the **positive control**: a pass rate, not a false-pass one.

    Windows drawn upstream of a CDS already in a D4 class are *supposed* to pass, so this arm's
    ~99.5 % is the control working — but published under ``false_pass_rate`` it tells any
    consumer that reads every such key in the file that criterion (c) false-passes almost
    always.  A named function because the property is *"the key is renamed, not copied"*, and
    an assertion that reads the committed report cannot see a producer that copies it.
    """
    out = dict(arm)
    out["pass_rate"] = out.pop("false_pass_rate", None)
    out.pop("false_pass_rate_denominator", None)
    out["pass_rate_denominator"] = "decided_windows"
    return out


def _arm(counts: Counter) -> dict[str, Any]:
    """One control arm's counts and its false-pass rate.

    ⚠ The rate divides by the **decided** windows, not by every drawn one.  A window whose (c)
    disjunct is ``unavailable`` is not a window where the rule declined to false-pass — it is
    one where the rule was never asked, and folding it into the denominator dilutes the rate
    by exactly the annotation gaps the *other* diagnostic exists to report.  Both denominators
    ride in the arm so the dilution is visible rather than assumed away.
    """
    total = sum(counts.values())
    decided = counts.get(STATUS_PASSED, 0) + counts.get(STATUS_FAILED, 0)
    return {
        "n": total,
        "n_decided": decided,
        "n_unavailable": counts.get(STATUS_UNAVAILABLE, 0),
        "status_counts": dict(sorted(counts.items())),
        "false_pass_rate": _rate(counts.get(STATUS_PASSED, 0), decided),
        "false_pass_rate_denominator": "decided_windows",
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
        # The same keys every other return path carries: a reader (or a shape assertion)
        # must not have to special-case the environment this ran in.
        return {
            "available": False,
            "n_usable_rows": 0,
            "reason": f"pandas unavailable, so the shipped decoy rows were not read: {exc}",
            "detail": {},
        }
    detail: dict[str, Any] = {}
    usable = 0
    for label, path in (("decoys_v0", decoys_parquet), ("mining_pool_v0", mining_pool_parquet)):
        if not Path(path).exists():
            detail[label] = {"present": False}
            continue
        try:
            frame = pd.read_parquet(path, columns=["accession"])
        except (ValueError, KeyError, OSError) as exc:
            # ``pyarrow.lib.ArrowIOError`` subclasses ``OSError``: a present-but-unreadable
            # file would otherwise abort the whole sweep before the report is written.
            # The abort would happen AFTER the whole false-pass sweep, losing every arm to a
            # schema change in a file this arm only reports the *absence* of.
            detail[label] = {"present": True, "readable": False, "reason": str(exc)}
            continue
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
    if not candidates:
        raise ProducerError("no candidates: the control arms have no length distribution to match")
    lengths = _window_lengths(candidates)
    get_host = _host_cache(annotation_dir, genome_dir)

    spans_by_host: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in candidates:
        assembly, _, ci = str(row["accession"]).partition(":c")
        spans_by_host[assembly].append((int(ci), int(row["locus_start"]), int(row["locus_end"])))

    candidate_clades = {
        clades.get(str(r["accession"]).partition(":c")[0], "unassigned") for r in candidates
    }
    random_leaders: Counter = Counter()
    utr_decoys: Counter = Counter()
    strict_utr_decoys: Counter = Counter()
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
        # ⚠ Two readings of the §9.1 5′UTR class, and reporting only one would be a choice
        # disguised as a measurement.  Drawn over ALL CDS, the arm's pass rate is the
        # per-clade density of D4's gene classes — but a 5′UTR sitting upstream of an aaRS in
        # a real genome may well BE a T-box leader, so counting it as a *false* pass inflates
        # the rate.  Drawn over the CDS that name no D4 class, every pass is unambiguously
        # false, but the arm no longer represents a random leader.  Both are reported; the
        # gap between them is exactly the ambiguity, and it is not this step's to resolve.
        sample = host.cds if len(host.cds) <= n_per_host else rng.sample(host.cds, n_per_host)
        # ⚠ The strict arm is a FILTER OF THIS SAMPLE, not a second independent draw.  Drawing
        # separately made the "subset" larger than its source (9,087 vs 9,065 rows), because
        # two draws each capped at ``n_per_host`` are not nested — and a filtered subset that
        # outnumbers its population is not a second reading of one arm, it is a second arm.

        extents = {sid: max(c.end for c in feats) for sid, feats in host.by_seqid.items()}
        # ⚠ Build the (CDS, window) pairs ONCE.  The strict arm reads these very window
        # objects — re-drawing spans with a fresh ``rng.choice(lengths)`` would make it a
        # second draw over a nested CDS population, and the report said "the same windows,
        # filtered" while doing exactly that.
        wide_windows, strict_windows = utr_arm_windows(
            sample, spans=[rng.choice(lengths) for _ in sample], extents=extents
        )
        utr_decoys += _evaluate_oriented(host, wide_windows, config=config)
        strict_utr_decoys += _evaluate_oriented(host, strict_windows, config=config)

        # …and windows upstream of annotated tRNA genes — §9.1's tRNA-adjacent sub-class.
        trna_features = host.trna
        # ⚠ ``extents`` is built from ``by_seqid``, which holds **CDS only**.  A tRNA on a
        # contig carrying no CDS has no entry, so the minus-strand upper bound was skipped and
        # the window could run past the end of the contig.  Extend the map with the tRNA spans
        # before the windows are cut.
        for feature in trna_features:
            extents[feature.seqid] = max(extents.get(feature.seqid, 0), feature.end)
        if trna_features:
            picked = (
                trna_features
                if len(trna_features) <= n_per_host
                else rng.sample(trna_features, n_per_host)
            )
            trna_windows = [
                w
                for w in (
                    _upstream_window(
                        t, span=rng.choice(lengths), contig_extent=extents.get(t.seqid)
                    )
                    for t in picked
                )
                if w
            ]
            trna_decoys += _evaluate_oriented(host, trna_windows, config=config)

        # The power arm: windows upstream of a CDS that IS in one of D4's four classes.
        positives = [
            c
            for c in host.cds
            if synteny.classify_gene_identity(
                gff3.gene_identity_text(c), gene_symbols=synteny.gene_symbols(c)
            )
            in synteny.PASSING_CLASSES
        ]
        if positives:
            picked_pos = (
                positives if len(positives) <= n_per_host else rng.sample(positives, n_per_host)
            )
            pos_windows = [
                w
                for w in (
                    _upstream_window(
                        c, span=rng.choice(lengths), contig_extent=extents.get(c.seqid)
                    )
                    for c in picked_pos
                )
                if w
            ]
            positive_context += _evaluate_oriented(host, pos_windows, config=config)

    background = _arm(random_leaders)
    positive = _arm(positive_context)
    # ⚠ The control arm's rate is a **pass** rate — windows drawn upstream of a CDS that is
    # already in a D4 class are *supposed* to pass.  Publishing 99.5 % under the key
    # ``false_pass_rate`` would report a 99.5 % false-pass rate to any consumer that reads
    # every ``false_pass_rate`` in this file.  The key is renamed here and only here.
    positive = as_control_arm(positive)
    # ``_arm`` reports ``None`` (not 0.0) when an arm decided nothing, so both operands are
    # checked before the subtraction — otherwise an all-unavailable arm raises TypeError
    # instead of leaving the control honestly ungraded.
    margin = (
        None
        if background["false_pass_rate"] is None or positive["pass_rate"] is None
        else positive["pass_rate"] - background["false_pass_rate"]
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
            "nine_one_five_prime_utr_decoys_excluding_d4_classes": dict(
                _arm(strict_utr_decoys),
                note=(
                    "the SAME window objects as the arm above — one draw, then filtered to "
                    "the windows whose own CDS names none of D4's four classes, so this arm "
                    "is a strict subset of it rather than a second draw.  Every pass here is "
                    "therefore unambiguously false, whereas the arm above includes 5′UTRs of "
                    "aaRS/biosynthesis genes, which in a real genome may themselves be T-box "
                    "leaders."
                ),
            ),
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
        # [[clauses-must-guard-emptiness]] — a clade with no drawn windows is absent from the
        # breakdown above and reads as "not measured" only if it is named somewhere.
        "clades_with_no_background_windows": sorted(
            clade for clade, counts in per_clade.items() if sum(counts.values()) == 0
        ),
        "clades_of_candidate_hosts_not_sampled": sorted(candidate_clades - set(per_clade)),
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
    """Accessions whose GFF is materialized, derived through the acquisition module's own
    filename contract rather than a second copy of the suffix.

    ``destination_name`` is the single definition of ``accession -> filename``; matching on a
    hardcoded ``*.gff.gz`` here would silently undercount every host the day that contract
    changes, and the undercount would look like a smaller corpus rather than a bug.
    """
    directory = Path(annotation_dir)
    if not directory.is_dir():
        return set()
    by_filename = {}
    for path in directory.iterdir():
        accession = path.name.split(".gff")[0] if ".gff" in path.name else None
        if accession and path.name == annotation_fetch.destination_name(accession):
            by_filename[path.name] = accession
    return set(by_filename.values())


# ═════════════════════════════════════════════════════════════════════════════
# D4 diagnostic 2 — the symmetric false-FAIL / per-clade exclusion / pseudogene arm
# ═════════════════════════════════════════════════════════════════════════════
def exclusion_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    clades: Mapping[str, str],
    config: SyntenyRunConfig,
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
    decision_distances: list[int] = []
    passed_via_carve_out = 0
    class_counts: Counter = Counter()

    for row in rows:
        clade = clades.get(str(row.get("assembly", "")), "unassigned")
        by_clade[clade][str(row["status"])] += 1
        if row["status"] == STATUS_UNAVAILABLE:
            reasons[clade][_exclusion_reason(row)] += 1
        for detail in (row.get("per_strand") or {}).values():
            pseudo_seen += int(detail.get("n_pseudo_seen") or 0)
            unjudgeable_seen += int(detail.get("n_unjudgeable_seen") or 0)
            if detail.get("carve_out_applied"):
                carve_out_used += 1
            # ⚠ Filter on the strand detail's OWN status, not on its function class.  A
            # carve-out target whose element-relative distance exceeds the window is a real
            # pass, but an out-of-window hit that FAILED is not — and collecting by class
            # alone put 554 entries (max 1,652 bp) into a statistic documented as "measured
            # on the passing candidates" against 541 passed candidates.
            if detail.get("status") != STATUS_PASSED:
                continue
            if detail.get("function_class") in synteny.PASSING_CLASSES:
                class_counts[detail["function_class"]] += 1
                if detail.get("distance_bp") is not None:
                    distances.append(int(detail["distance_bp"]))
                if detail.get("decision_distance_bp") is not None:
                    decision_distances.append(int(detail["decision_distance_bp"]))
                if detail.get("carve_out_applied"):
                    passed_via_carve_out += 1

    total = len(rows)
    unavailable = sum(1 for r in rows if r["status"] == STATUS_UNAVAILABLE)
    passing_candidates = sum(1 for r in rows if r["status"] == STATUS_PASSED)
    both_strand_passes = sum(
        1
        for r in rows
        if r["status"] == STATUS_PASSED
        and sum(1 for d in (r.get("per_strand") or {}).values() if d.get("status") == STATUS_PASSED)
        == 2
    )
    distances.sort()
    decision_distances.sort()

    def _pct(values: list[int], p: float) -> int | None:
        if not values:
            return None
        return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]

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
                    _exclusion_reason(r) for r in rows if r["status"] == STATUS_UNAVAILABLE
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
            # ⚠ Two different units live in this report and the difference is not a
            # discrepancy: the per-clade block counts CANDIDATES (546 + 96 + 299 = 941),
            # while this block counts STRAND EVALUATIONS, and a candidate that passes on
            # both strands contributes two.  Both are named so a reader cannot subtract one
            # from the other and conclude something is missing.
            "unit": "strand_evaluations",
            "n": len(distances),
            "n_candidates_passing": passing_candidates,
            "n_candidates_passing_on_both_strands": both_strand_passes,
            "p50_bp": _pct(distances, 0.50),
            "p95_bp": _pct(distances, 0.95),
            "p99_bp": _pct(distances, 0.99),
            "max_bp": distances[-1] if distances else None,
            "window_bp": config.window_bp,
            "n_passed_via_tandem_carve_out": passed_via_carve_out,
            "decision_distance": {
                "n": len(decision_distances),
                "p50_bp": _pct(decision_distances, 0.50),
                "p95_bp": _pct(decision_distances, 0.95),
                "p99_bp": _pct(decision_distances, 0.99),
                "max_bp": decision_distances[-1] if decision_distances else None,
            },
            "note": (
                "D4 asks for the empirical p95/p99 as a sensitivity check on the 500 bp pad; "
                "measured over the strand evaluations that PASSED in THIS corpus, not on "
                "catalogued Firmicutes T-boxes.  Two series: the element-relative distance "
                "(the reportable quantity) and the re-anchored decision distance criterion_c "
                "actually judged.  They differ only where D4's tandem carve-out fired, which "
                "is why the element-relative max may exceed window_bp while every decision "
                "distance is inside it."
            ),
        },
        "passing_class_counts": dict(sorted(class_counts.items())),
        "passing_class_counts_unit": "strand_evaluations",
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
    counts = fetch.get("status_counts") if isinstance(fetch, Mapping) else None
    counts = counts if isinstance(counts, Mapping) else {}
    raw_ok = counts.get(annotation_fetch.STATUS_OK, 0)
    n_ok = raw_ok if isinstance(raw_ok, int) and not isinstance(raw_ok, bool) else 0
    clauses["acquisition_report_records_a_corpus"] = n_ok > 0

    false_pass = _json_or_none(root / FALSE_PASS_REPORT)
    # ``_json_or_none`` can return a list or a scalar for a malformed report, and a
    # fail-closed derivation that raises is not fail-closed — it takes the whole preflight
    # down instead of answering False.
    control = false_pass.get("control") if isinstance(false_pass, Mapping) else None
    clauses["false_pass_control_powered"] = bool(
        isinstance(control, Mapping) and control.get("powered") is True
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
def _split_inputs(inputs: Sequence[str | Path]) -> tuple[list[str], list[dict[str, str]]]:
    """In-repo inputs (hashed by ``build_provenance``) vs external ones (hashed here).

    ⚠ A committed provenance record must never carry an absolute local path: it leaks a
    username and names a machine-specific location no reader can resolve.  ``$ROUND_DIR``
    genuinely lives outside the repo, so its status table cannot become a repo-relative
    path — but it also must not simply vanish from the record.  It is therefore hashed
    here and recorded as ``{name, sha256}``, which is *more* traceable than the path was:
    a reader can verify the bytes without knowing where they sat.
    """
    from tbox_finder.provenance import sha256_file

    inside: list[str] = []
    external: list[dict[str, str]] = []
    for item in inputs:
        resolved = Path(item).resolve()
        try:
            inside.append(str(resolved.relative_to(REPO_ROOT)))
        except ValueError:
            external.append({"name": resolved.name, "sha256": sha256_file(resolved)})
    return inside, external


def _provenance(entry: str, inputs: Sequence[str | Path], extra: Mapping[str, Any]) -> dict:
    """CLAUDE.md §11 provenance for a committed report.

    ``inputs`` only — never the report being written.  ``build_provenance`` hashes whatever it
    is given, and an output that does not exist yet raises at the END of the run that would
    have produced it ([[build-provenance-hashes-its-outputs]]).
    """
    from tbox_finder.provenance import build_provenance

    inside, external = _split_inputs([i for i in inputs if Path(i).exists()])
    payload = dict(extra)
    if external:
        payload["external_inputs"] = external
    return build_provenance(
        rule=f"synteny_producer::{entry}",
        script="src/tbox_finder/mining/synteny_producer.py",
        inputs=inside,
        adr=ADR,
        extra=payload,
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
            # Validate first: reading ``rows`` out of a table that contradicts its own status
            # map would build both diagnostics on evidence the loader is about to reject.
            # Parsed ONCE: reading the file twice validates one snapshot and then builds the
            # diagnostics from another, so a concurrent write lands between the two.
            payload = json.loads(Path(args.status_table).read_text(encoding="utf-8"))
            validate_status_payload(payload, label=str(args.status_table))
            rows = payload["rows"]
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
            # Built before either is written: a failure between the two would otherwise
            # leave a committed false-pass report beside a stale exclusion report, and the
            # pair is asserted to describe the same run.
            ex = exclusion_report(rows, clades=clades, config=config)
            ex["provenance"] = _provenance(
                "diagnostics.exclusion",
                [args.status_table, args.clade_table],
                {"step": STEP, **config.as_dict()},
            )
            _write(args.false_pass_out, fp)
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
    except (
        ProducerError,
        synteny.SyntenyError,
        gff3.Gff3Error,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
