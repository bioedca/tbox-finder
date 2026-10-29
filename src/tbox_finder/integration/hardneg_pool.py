"""P3-16″-hardneg — the §9.1 class-4 leader pool behind the A13 Pin 1 hard-negative amendment.

Why this module exists
----------------------
v2's screened benchmark has no power: 0 of 691 graded negatives drew a Stage-1 candidate on
either arm (ADR-0005 A15 Pin 5). Its negatives are flank windows with a decoy spliced in — the
background the scanner never confuses — and PRD §9.1's class 4 (5′UTR / tRNA-adjacent leaders,
"the hardest false-positive context for a tRNA-sensing finder") has no genuine record in it.
ADR-0005 **A17** (signed 2026-10-28, before any of its scores existed) pins a hard-negative
amendment built from that class, and this module is its population, draw and screens:

* **Pin 1, the population.** Every GFF3 ``CDS`` and ``tRNA`` on the 339 NCBI-annotated admissible
  a2 hosts (``reports/p3/annotation_supply.json``). A host is in because NCBI publishes an
  annotation for it, a property of the assembly and of no score. Each feature yields one
  **native** 1,024-nt window on its own strand, ending at the feature's first base + δ, with
  δ ~ U{0..256} at seed 42. **100,000** eligible windows are drawn uniformly at seed 42.
* **Pin 2, the screens, on the drawn windows.** An assembly-gap refusal, then A15 Pin 1's fold
  screen and A15 Pin 2's content screen **unchanged** (this module calls ``host_screen``'s own
  functions — it re-implements neither), then the **D4-context partition**: a survivor whose
  ±500-bp neighbourhood holds a D4-class CDS (aaRS / amino-acid biosynthesis / transport /
  transamidation) is reported *beside* the graded denominator, never inside it.
* **Pin 2's audit.** A native window's screened bytes are its scored bytes, so A15 Pin 3's
  junction-and-insert assertion has no object here; the filter-off sensitivity gap D2 names is
  covered instead by a score-blind seeded audit of 1,000 graded windows at ``--max -T 0``, read
  against ``host_screen.POSTMINT_THRESHOLD_BITS`` and the N3 matched-null floor.

What this module does **not** do: it grades nothing, reads no Stage-1 or Stage-2 output, and
computes no AUPRC. The power clause (A17 Pin 4) and the statistic live in ``precision.py``. No
selection step here reads a score — that is the whole of A17 Pin 1's direction-of-bias claim, so
the round-0 candidate manifest, whose production Stage-1 and Stage-2 scores already exist on
these hosts, is not an input to any leg.

Six legs, two environments (the ``infernal`` env has ``cmsearch`` and no pandas)::

    prep          (data)      features → δ → eligible → draw → gap + fold screen → FASTAs
    search        (infernal)  survivor shards + controls × {RF00230, TBDB001} at -T 10
    report        (data)      content screen + controls + D4 partition + audit sample
    audit-search  (infernal)  the audit FASTA × models at --max -T 0
    audit-report  (data)      the audit verdict → reports/p3/hardneg_pool.json (validated)

``--census`` (A17 Pin 4's one pre-registered extension) selects every eligible window **not** in
the draw instead of the draw. It is triggered only by ``precision power`` and needs a §9.3 ack.

Every recorded path is repo-relative ([[committing-real-tool-output-fixtures]]); ``--data-root``
points at the checkout that holds the DVC-tracked bytes, as in ``host_screen``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tbox_finder import infernal, provenance
from tbox_finder.data import embedding as emb
from tbox_finder.data import window_dataset as wd
from tbox_finder.integration import host_screen as hs
from tbox_finder.integration.two_stage import write_json

SCHEMA_VERSION = "1.0"
STEP = "P3-16″-hardneg"
GENERATED_BY = "src/tbox_finder/integration/hardneg_pool.py"
RULE = "workflow/rules/integration.smk :: hardneg_pool"
ADR = "ADR-0005 A17"
PRD = "§9.1 (class 4); §2.3"

KIND_POOL = "hardneg_pool"
KIND_PREP = "hardneg_pool_prep"
KIND_INTERIM = "hardneg_pool_interim"

#: A17 Pin 1's three new construction values, pinned before any A17 score existed.
SEED = 42
DRAW_SIZE = 100_000
DELTA_MAX = 256
#: The scanner's window, the same 1,024 nt every v2 item is.
WINDOW_NT = 1024
#: A17 Pin 2 step 4: ADR-0006 D4's 500-bp window, reused as the D4-context neighbourhood.
D4_CONTEXT_PAD_BP = 500
#: A17 Pin 2's score-blind audit size.
AUDIT_SIZE = 1_000
#: The audit's search floor: report from 0 bits, so the sub-threshold band is published.
AUDIT_SEARCH_BITS = 0.0

FEATURE_TYPES: tuple[str, ...] = ("CDS", "tRNA")
SUB_CLASS = {"CDS": "cds_upstream", "tRNA": "trna_upstream"}
MODE_DRAW = "draw"
MODE_CENSUS = "census"

#: Partition labels, in the order Pin 2 applies them. A window gets exactly one.
P_GAP = "refused_assembly_gap"
P_FOLD = "refused_fold"
P_CONTENT = "refused_content"
P_GRADED = "graded"
P_BESIDE = "beside_d4_context"
PARTITIONS: tuple[str, ...] = (P_GAP, P_FOLD, P_CONTENT, P_GRADED, P_BESIDE)

#: The A15 Pin 1 coordinate-only v1 host, resolved by prefix from the mining pool as host_screen
#: resolves it. It is a control of the *screen*, not a member of this population.
COORDINATE_ONLY_PREFIX = "4ef2d573"

FILE_FEATURES = "features.tsv"
FILE_ELIGIBLE = "eligible.tsv"
FILE_DRAWN = "drawn.txt"
FILE_ITEMS = "items.json"
FASTA_CONTROLS = "controls.fa"
FASTA_AUDIT = "audit.fa"
DEFAULT_SHARDS = 8

DEFAULT_SUPPLY = "reports/p3/annotation_supply.json"
DEFAULT_ANNOTATIONS = "data/interim/production_annotations"
DEFAULT_GENOMES = "data/interim/production_genomes"
DEFAULT_CLADES = "data/processed/mining/production_genomes_v0.parquet"
DEFAULT_NOISE_FLOOR = "reports/p3/postmint_noise_floor.json"


class HardnegPoolError(RuntimeError):
    """A refusal: the pool, a screen or a binding would certify something it does not show."""


# ══════════════════════════════════════════════════════════════════════════════════════
# Pin 1 — features, δ, windows (stdlib only)
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Feature:
    """One item-bearing feature: its 5′-most base (1-based) on its own strand."""

    assembly: str
    seqid: str
    first1: int
    strand: str
    ftype: str
    feature_id: str

    @property
    def sort_key(self) -> tuple[str, str, int, str, str, str]:
        return (self.assembly, self.seqid, self.first1, self.strand, self.ftype, self.feature_id)


def window_span(first1: int, strand: str, delta: int) -> tuple[int, int]:
    """A17 Pin 1's native window, as a 0-based half-open ``[start, end)`` on the ``+`` strand.

    ``+``: ``[s₀ − 1024 + δ, s₀ + δ)`` with ``s₀ = first1 − 1``. ``−``: the mirror image — the
    feature's first base is its **highest** coordinate, so the window holds that base and the
    δ − 1 below it plus the 1,024 − δ above it, i.e. ``[E − δ, E + 1024 − δ)`` with ``E = first1``.
    At δ = 0 both strands give the 1,024 nt immediately upstream, excluding the first base.
    """
    if not 0 <= int(delta) <= DELTA_MAX:
        raise HardnegPoolError(f"delta {delta!r} is outside A17 Pin 1's U{{0..{DELTA_MAX}}}")
    if strand == "+":
        s0 = int(first1) - 1
        return s0 - WINDOW_NT + int(delta), s0 + int(delta)
    if strand == "-":
        e = int(first1)
        return e - int(delta), e + WINDOW_NT - int(delta)
    raise HardnegPoolError(f"strand {strand!r} is unresolved; a window needs a feature sense")


def item_id(assembly: str, seqid: str, start: int, end: int, strand: str) -> str:
    """The de-duplication key, rendered — unique by construction after :func:`eligible_windows`."""
    return f"{assembly}:{seqid}:{int(start)}-{int(end)}:{strand}"


def assign_deltas(features: Sequence[Feature], rng: random.Random) -> list[int]:
    """δ for every feature, in sorted order, **before** eligibility — A17 Pin 1's RNG order.

    Drawn before eligibility so a later change to the contig-edge rule cannot shift every
    downstream δ; the draw then consumes the same generator, so replaying both needs only the
    seed and the feature count.
    """
    return [rng.randint(0, DELTA_MAX) for _ in features]


def eligible_windows(
    features: Sequence[Feature],
    deltas: Sequence[int],
    contig_lengths: Mapping[tuple[str, str], int],
) -> tuple[list[dict[str, Any]], Counter]:
    """Feature + δ → in-contig windows, de-duplicated on (assembly, seqid, start, end, strand).

    ``features`` must already be sorted; the first feature producing a window keeps it. A window
    off either contig end is ineligible, not clipped: a clipped window is shorter than every
    positive and would be zero-padded into a different object.
    """
    if len(features) != len(deltas):
        raise HardnegPoolError(f"{len(deltas)} deltas for {len(features)} features")
    status: Counter = Counter()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for feature, delta in zip(features, deltas, strict=True):
        start, end = window_span(feature.first1, feature.strand, delta)
        length = contig_lengths.get((feature.assembly, feature.seqid))
        if length is None:
            raise HardnegPoolError(
                f"{feature.assembly}: seqid {feature.seqid!r} is in the GFF but not the genome"
            )
        if start < 0 or end > length:
            status["off_contig"] += 1
            continue
        key = item_id(feature.assembly, feature.seqid, start, end, feature.strand)
        if key in seen:
            status["duplicate_window"] += 1
            continue
        seen.add(key)
        status["eligible"] += 1
        out.append(
            {
                "item_id": key,
                "assembly": feature.assembly,
                "seqid": feature.seqid,
                "start": start,
                "end": end,
                "strand": feature.strand,
                "feature_type": feature.ftype,
                "feature_id": feature.feature_id,
                "delta": int(delta),
                "sub_class": SUB_CLASS[feature.ftype],
            }
        )
    return out, status


def eligible_line(window: Mapping[str, Any]) -> str:
    """One canonical line of the eligible list — what its sha256 is taken over."""
    return "\t".join(str(window[key]) for key in ("item_id", "feature_type", "feature_id", "delta"))


def eligible_digest(windows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for window in windows:
        digest.update((eligible_line(window) + "\n").encode("utf-8"))
    return digest.hexdigest()


def select(
    eligible: Sequence[Mapping[str, Any]], rng: random.Random, *, mode: str, size: int = DRAW_SIZE
) -> list[Mapping[str, Any]]:
    """The draw (``rng.sample``, A17 Pin 1) or the census (every eligible window not drawn).

    The census still consumes the draw from the same generator, so "not in the draw" names the
    very windows the draw left — it is the draw's complement, never a second draw.
    """
    if size > len(eligible):
        raise HardnegPoolError(f"draw of {size} from only {len(eligible)} eligible windows")
    drawn = rng.sample(list(eligible), size)
    if mode == MODE_DRAW:
        return drawn
    if mode == MODE_CENSUS:
        taken = {str(window["item_id"]) for window in drawn}
        return [window for window in eligible if str(window["item_id"]) not in taken]
    raise HardnegPoolError(f"mode {mode!r} is neither {MODE_DRAW!r} nor {MODE_CENSUS!r}")


def replay_selection(
    n_features: int, eligible: Sequence[Mapping[str, Any]], *, mode: str, size: int
) -> tuple[list[int], list[Mapping[str, Any]]]:
    """Re-run A17 Pin 1's RNG from the seed: the δ sequence and the selection it implies."""
    rng = random.Random(SEED)
    deltas = [rng.randint(0, DELTA_MAX) for _ in range(int(n_features))]
    return deltas, select(eligible, rng, mode=mode, size=size)


def window_sequence(contig: str, window: Mapping[str, Any]) -> str:
    """The window's bytes in the feature's sense (reverse-complemented on ``−``)."""
    raw = contig[int(window["start"]) : int(window["end"])]
    return raw if window["strand"] == "+" else wd.reverse_complement(raw)


def has_assembly_gap(sequence: str) -> bool:
    """A17 Pin 2 step 1: any non-ACGT character after A15's alphabet rule."""
    return any(base not in "ACGT" for base in hs.normalise_dna(sequence))


# ══════════════════════════════════════════════════════════════════════════════════════
# Pin 2 step 4 — the D4-context partition (stdlib only)
# ══════════════════════════════════════════════════════════════════════════════════════
def d4_intervals(cds_features: Iterable[Any]) -> dict[str, list[tuple[int, int]]]:
    """Per seqid, the 0-based half-open spans of every CDS naming a D4 class, either strand.

    "Any CDS" is read literally: a pseudogenised CDS still names its function, and counting it
    widens the beside set, which is the conservative direction for a label rule.
    """
    from tbox_finder.mining import gff3, synteny  # lazy: keeps the search leg's imports minimal

    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for cds in cds_features:
        klass = synteny.classify_gene_identity(
            gff3.gene_identity_text(cds), gene_symbols=synteny.gene_symbols(cds)
        )
        if klass in synteny.PASSING_CLASSES:
            out[str(cds.seqid)].append((int(cds.start) - 1, int(cds.end)))
    for spans in out.values():
        spans.sort()
    return dict(out)


def is_d4_context(window: Mapping[str, Any], intervals: Mapping[str, Sequence[tuple[int, int]]]):
    """True iff a D4-class CDS overlaps ``[start − 500, end + 500)`` on the window's seqid."""
    lo = int(window["start"]) - D4_CONTEXT_PAD_BP
    hi = int(window["end"]) + D4_CONTEXT_PAD_BP
    return any(start < hi and lo < end for start, end in intervals.get(str(window["seqid"]), ()))


# ══════════════════════════════════════════════════════════════════════════════════════
# Host loading (data env)
# ══════════════════════════════════════════════════════════════════════════════════════
def annotated_hosts(root: Path, supply: str) -> list[dict[str, Any]]:
    """The 339 ``annotated`` rows, refused unless the count is the report's own."""
    from tbox_finder.mining import annotation_fetch

    payload = json.loads((root / supply).read_text(encoding="utf-8"))
    targets = annotation_fetch.annotation_targets(payload)
    expected = annotation_fetch.strict_count(payload["admissible_status_counts"], "annotated")
    if len(targets) != expected:
        raise HardnegPoolError(f"{len(targets)} annotated targets, the report says {expected}")
    return sorted(targets, key=lambda row: str(row["accession"]))


def load_host(
    root: Path, target: Mapping[str, Any], *, annotations: str, genomes: str
) -> tuple[list[Feature], list[Any], dict[str, str], Counter]:
    """One host's item features, every CDS (for D4), and its contigs; md5-verified."""
    from tbox_finder.infer import rho_pilot
    from tbox_finder.mining import annotation_fetch, gff3

    assembly = str(target["accession"])
    gff_path = root / annotations / annotation_fetch.destination_name(assembly)
    raw = gff_path.read_bytes()
    if not annotation_fetch.digest_matches(raw, str(target["gff_md5"])):
        raise HardnegPoolError(f"{gff_path.name}: md5 differs from the supply report's gff_md5")
    text = gff3.decode_gff3_bytes(raw)
    cds = gff3.parse_gff3_document(text)
    census: Counter = Counter()
    features: list[Feature] = []
    for one in cds:
        if gff3.is_pseudo(one):
            census["cds_pseudo_excluded"] += 1
            continue
        if one.strand not in ("+", "-"):
            census["cds_strand_unresolved"] += 1
            continue
        features.append(
            Feature(
                assembly, one.seqid, gff3.cds_start_position(one), one.strand, "CDS", one.feature_id
            )
        )
    by_id: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in gff3.iter_gff3_features(text.splitlines(), types={"tRNA"}):
        by_id[(row.seqid, row.feature_id)].append(row)
    for (seqid, fid), rows in sorted(by_id.items()):
        strands = {row.strand for row in rows}
        if len(strands) != 1 or next(iter(strands)) not in ("+", "-"):
            census["trna_strand_unresolved"] += 1
            continue
        strand = next(iter(strands))
        if any(gff3.attribute_first(row.attributes, "pseudo") for row in rows):
            census["trna_pseudo_excluded"] += 1
            continue
        first1 = min(r.start for r in rows) if strand == "+" else max(r.end for r in rows)
        features.append(Feature(assembly, seqid, first1, strand, "tRNA", fid))
    contigs = dict(rho_pilot.read_fasta_file(root / genomes / f"{assembly}.fna"))
    return features, cds, contigs, census


# ══════════════════════════════════════════════════════════════════════════════════════
# Small I/O helpers
# ══════════════════════════════════════════════════════════════════════════════════════
def _write_lines(path: Path, lines: Iterable[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    return provenance.sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root(args: argparse.Namespace) -> Path:
    return Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]


def _provenance(leg: str, inputs: Sequence[str | Path], root: Path, *, env_lock: str) -> dict:
    return hs.provenance_record(
        leg=leg,
        inputs=inputs,
        root=root,
        env_lock=root / env_lock,
        rule=RULE,
        script=GENERATED_BY,
        adr=ADR,
    )


def shard_names(n_shards: int) -> list[str]:
    return [f"survivors_{index:02d}.fa" for index in range(int(n_shards))]


# ══════════════════════════════════════════════════════════════════════════════════════
# prep (data env)
# ══════════════════════════════════════════════════════════════════════════════════════
def _coordinate_only_verdicts(root: Path, mining_pool: str, dna: hs.ScreenedDna) -> dict:
    """The A15 coordinate-only control's subject, screened through the same invocation."""
    candidates, _report = hs.load_candidates(root, mining_pool)
    verdicts: dict[str, dict[str, Any]] = {}
    for row in candidates:
        cid = str(row["candidate_id"])
        if not cid.startswith(COORDINATE_ONLY_PREFIX):
            continue
        verdict = hs.screen_window(
            hs.normalise_dna(row["sequence"]),
            row["accession"],
            row["locus_start"],
            row["locus_end"],
            dna,
        )
        verdicts[cid] = {
            **verdict.as_json(),
            "accession": str(row["accession"]),
            "locus_start": int(row["locus_start"]),
            "locus_end": int(row["locus_end"]),
        }
    return verdicts


def _cmd_prep(args: argparse.Namespace) -> int:
    """LEG 1: Pin 1's population and draw, then Pin 2 steps 1–2 on the drawn windows."""
    root = _root(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    size = int(args.draw_size)

    # ── Pin 1: every host's features, sorted, then δ, eligibility, the draw ────────────────
    targets = annotated_hosts(root, args.supply)
    if args.max_hosts is not None:
        targets = targets[: int(args.max_hosts)]
    features: list[Feature] = []
    contig_lengths: dict[tuple[str, str], int] = {}
    census: Counter = Counter()
    for target in targets:
        host_features, _cds, contigs, host_census = load_host(
            root, target, annotations=args.annotations, genomes=args.genomes
        )
        features.extend(host_features)
        census.update(host_census)
        for seqid, sequence in contigs.items():
            contig_lengths[(str(target["accession"]), seqid)] = len(sequence)
    features.sort(key=lambda feature: feature.sort_key)
    rng = random.Random(SEED)
    deltas = assign_deltas(features, rng)
    eligible, status = eligible_windows(features, deltas, contig_lengths)
    selected = select(eligible, rng, mode=args.mode, size=size)

    features_sha = _write_lines(
        out_dir / FILE_FEATURES,
        (
            "\t".join([f.assembly, f.seqid, str(f.first1), f.strand, f.ftype, f.feature_id, str(d)])
            for f, d in zip(features, deltas, strict=True)
        ),
    )
    eligible_sha = _write_lines(out_dir / FILE_ELIGIBLE, (eligible_line(w) for w in eligible))
    drawn_sha = _write_lines(out_dir / FILE_DRAWN, sorted(str(w["item_id"]) for w in selected))

    # ── Pin 2 steps 1–2 on the selected windows, host by host (one genome in memory) ──────
    dna, dna_blocks = hs.build_screened_dna(
        root=root,
        mining_pool=args.mining_pool,
        decoys=args.decoys,
        dataset=args.dataset,
        context=args.context,
        labels=args.labels,
        split_table=args.split_table,
        stage2_provenance=args.stage2_provenance,
    )
    by_host: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for window in selected:
        by_host[str(window["assembly"])].append(window)
    items: dict[str, dict[str, Any]] = {}
    per_source: dict[str, Counter] = {name: Counter() for name in hs.SCREENED_SOURCES}
    for target in targets:
        assembly = str(target["accession"])
        if assembly not in by_host:
            continue
        _features, _cds, contigs, _census = load_host(
            root, target, annotations=args.annotations, genomes=args.genomes
        )
        for window in by_host[assembly]:
            sequence = window_sequence(contigs[str(window["seqid"])], window)
            record = {**window, "sha256": hashlib.sha256(sequence.encode()).hexdigest()}
            if has_assembly_gap(sequence):
                record["partition"] = P_GAP
            else:
                # A15 Pin 1's coordinate rule reads closed intervals, so the half-open window
                # goes in as 1-based inclusive [start + 1, end] on the contig accession.
                verdict = hs.screen_window(
                    sequence, window["seqid"], int(window["start"]) + 1, int(window["end"]), dna
                )
                record["fold"] = verdict.as_json()
                for name, count in verdict.kmer_sources.items():
                    per_source[name]["kmer"] += bool(count)
                for name in verdict.coord_sources:
                    per_source[name]["coord"] += 1
                record["motif"] = hs.motif_present(sequence)
                record["partition"] = P_FOLD if verdict.refused else None
                record["sequence"] = sequence
            items[str(window["item_id"])] = record
    survivors = sorted(cid for cid, rec in items.items() if rec["partition"] is None)
    if not survivors:
        raise HardnegPoolError("the fold screen refused every drawn window")

    # ── the content-screen FASTAs: survivors in shards, plus the controls ──────────────────
    n_shards = max(1, min(int(args.shards), len(survivors)))
    shards = shard_names(n_shards)
    shard_records: list[dict[str, str]] = [{} for _ in shards]
    for index, cid in enumerate(survivors, start=1):
        name = hs.surrogate("w", index)
        items[cid]["surrogate"] = name
        items[cid]["shard"] = shards[(index - 1) % n_shards]
        shard_records[(index - 1) % n_shards][name] = items[cid]["sequence"]
    for basename, records in zip(shards, shard_records, strict=True):
        infernal.write_fasta(records, out_dir / basename)

    # The control host is the first motif-free survivor in id order — chosen before any search,
    # as host_screen chooses its own, so no control is selected on the strength of its result.
    host_id = next((cid for cid in survivors if not items[cid]["motif"]), None)
    if host_id is None:
        raise HardnegPoolError("every fold survivor carries the motif; no clean control host")
    host = items[host_id]
    host_row = {
        "candidate_id": host_id,
        "sequence": host["sequence"],
        "accession": host["seqid"],
        "locus_start": int(host["start"]) + 1,
        "locus_end": int(host["end"]),
    }
    controls, control_sequences = hs.pin1_controls(
        dna=dna,
        host=host_row,
        coordinate_only=hs.coordinate_only_candidate(
            _coordinate_only_verdicts(root, args.mining_pool, dna), COORDINATE_ONLY_PREFIX
        ),
    )
    tbox_name, tbox_seq = hs.first_fasta_record(root / args.rf00230_master)
    tbox_dna = hs.normalise_dna(tbox_seq)
    control_sequences[hs.CONTROL_CONTENT] = emb.splice(
        hs.normalise_dna(host["sequence"]), tbox_dna, hs.CONTROL_PHASE
    )
    control_sequences[hs.CONTROL_CONTENT_HOST] = hs.normalise_dna(host["sequence"])
    control_meta = {
        meaning: {
            "surrogate": hs.surrogate("c", index),
            "length": len(sequence),
            "motif": hs.motif_present(sequence),
        }
        for index, (meaning, sequence) in enumerate(sorted(control_sequences.items()), start=1)
    }
    infernal.write_fasta(
        {meta["surrogate"]: control_sequences[m] for m, meta in control_meta.items()},
        out_dir / FASTA_CONTROLS,
    )

    write_json(out_dir / FILE_ITEMS, {"items": items})
    fastas = [*shards, FASTA_CONTROLS]
    prep = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": KIND_PREP,
        "adr": ADR,
        "config": {
            "seed": SEED,
            "draw_size": size,
            "pinned_draw": size == DRAW_SIZE and args.max_hosts is None,
            "mode": args.mode,
            "delta_max": DELTA_MAX,
            "window_nt": WINDOW_NT,
            "feature_types": list(FEATURE_TYPES),
            "max_hosts": args.max_hosts,
            "sort_key": "assembly, seqid, first1, strand, feature_type, feature_id",
            "rng_order": "random.Random(42): randint(0, 256) per sorted feature, then sample",
        },
        "population": {
            "n_hosts": len(targets),
            "n_features": len(features),
            "n_features_by_type": dict(Counter(f.ftype for f in features)),
            "feature_census": dict(census),
            "eligibility": dict(status),
            "n_eligible": len(eligible),
            "n_selected": len(selected),
        },
        "files": {
            FILE_FEATURES: {
                "path": hs.recorded_path(out_dir / FILE_FEATURES, root),
                "sha256": features_sha,
            },
            FILE_ELIGIBLE: {
                "path": hs.recorded_path(out_dir / FILE_ELIGIBLE, root),
                "sha256": eligible_sha,
            },
            FILE_DRAWN: {"path": hs.recorded_path(out_dir / FILE_DRAWN, root), "sha256": drawn_sha},
            FILE_ITEMS: {
                "path": hs.recorded_path(out_dir / FILE_ITEMS, root),
                "sha256": provenance.sha256_file(out_dir / FILE_ITEMS),
            },
        },
        "fasta": {
            name: {
                "path": hs.recorded_path(out_dir / name, root),
                "sha256": provenance.sha256_file(out_dir / name),
                "threshold_bits": hs.CONTENT_THRESHOLD_BITS,
            }
            for name in fastas
        },
        "screened_dna": dna_blocks,
        "fold_per_source": {name: dict(counts) for name, counts in per_source.items()},
        "controls": controls,
        "control_meta": control_meta,
        "content_control": {
            "name": hs.CONTROL_CONTENT,
            "insert": tbox_name,
            "insert_length": len(tbox_dna),
            "insert_source": args.rf00230_master,
            "host_item_id": host_id,
            "phase": hs.CONTROL_PHASE,
        },
        "provenance": _provenance(
            "prep",
            [
                root / args.supply,
                root / args.mining_pool,
                root / args.decoys,
                root / args.dataset,
                root / args.context,
                root / args.labels,
                root / args.split_table,
                root / args.stage2_provenance,
                root / args.rf00230_master,
            ],
            root,
            env_lock=hs.ENV_LOCK_DATA,
        ),
        "wall_s": round(time.monotonic() - started, 2),
    }
    write_json(out_dir / "prep.json", prep)
    print(
        f"prep: {len(features)} features → {len(eligible)} eligible → {len(selected)} "
        f"{args.mode}; {len(survivors)} fold survivors in {n_shards} shards"
    )
    return 0


# ══════════════════════════════════════════════════════════════════════════════════════
# search + audit-search (infernal env)
# ══════════════════════════════════════════════════════════════════════════════════════
def _search(args: argparse.Namespace, *, basenames: Sequence[str], audit: bool) -> int:
    root = _root(args)
    out_dir = Path(args.out_dir)
    tblout_root = out_dir / ("audit_tblout" if audit else "tblout")
    tblout_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "root": root,
            "out_dir": out_dir,
            "tblout_root": tblout_root,
            "arm_dir": "filter_off/" if audit else "",
            "basename": basename,
            "model": model,
            "cm": cm,
            "threshold_bits": AUDIT_SEARCH_BITS if audit else hs.CONTENT_THRESHOLD_BITS,
            "cpu": args.cpu,
            "no_filters": audit,
            "timeout_s": args.timeout_s,
        }
        for basename in basenames
        for model, cm in hs.SCREEN_CMS
    ]
    for task in tasks:
        if not (out_dir / task["basename"]).exists():
            raise HardnegPoolError(f"{out_dir / task['basename']} does not exist; run prep first")
    started = time.monotonic()
    searches = hs.run_search_tasks(tasks, jobs=int(args.jobs))
    write_json(
        tblout_root / "search.json",
        {
            "schema_version": SCHEMA_VERSION,
            "step": STEP,
            "kind": "hardneg_pool_audit_search" if audit else "hardneg_pool_search",
            "adr": ADR,
            "infernal_version": hs.infernal_version_line(),
            "env_lock": hs.ENV_LOCK,
            "env_lock_sha256": provenance.sha256_file(root / hs.ENV_LOCK),
            "cpu": int(args.cpu),
            "jobs": int(args.jobs),
            "searches": searches,
            "wall_s": round(time.monotonic() - started, 2),
        },
    )
    print(f"search: {len(searches)} searches in {round(time.monotonic() - started, 1)}s")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """LEG 2: A15 Pin 2's invocation, unchanged, over every survivor shard and the controls."""
    prep = _read_json(Path(args.out_dir) / "prep.json")
    return _search(args, basenames=sorted(prep["fasta"]), audit=False)


def _cmd_audit_search(args: argparse.Namespace) -> int:
    """LEG 4: the score-blind audit FASTA, filter-off (``--max``), reporting from 0 bits."""
    return _search(args, basenames=[FASTA_AUDIT], audit=True)


# ══════════════════════════════════════════════════════════════════════════════════════
# report (data env)
# ══════════════════════════════════════════════════════════════════════════════════════
def audit_sample(graded_ids: Iterable[str]) -> list[str]:
    """A17 Pin 2's audit: 1,000 graded ids, ``random.Random(42).sample`` over the sorted ids."""
    pool = sorted(str(cid) for cid in graded_ids)
    if len(pool) <= AUDIT_SIZE:
        return pool
    return sorted(random.Random(SEED).sample(pool, AUDIT_SIZE))


def partition_counts(items: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """``{partition: {sub_class: n, "total": n}}`` — the conservation identity's terms."""
    out: dict[str, Counter] = {name: Counter() for name in PARTITIONS}
    for record in items.values():
        out[str(record["partition"])][str(record["sub_class"])] += 1
    return {
        name: {**dict(sorted(counts.items())), "total": sum(counts.values())}
        for name, counts in out.items()
    }


def _clade_of(root: Path, table: str) -> dict[str, str]:
    from tbox_finder.mining import synteny_producer

    path = root / table
    return synteny_producer.load_clades(path) if path.exists() else {}


def composition(items: Mapping[str, Mapping[str, Any]], clades: Mapping[str, str]) -> dict:
    """A17 Pin 5 (d)'s tables: per phylum, per sub-class, per partition — beside v2's own."""
    out: dict[str, dict[str, Counter]] = {
        "phylum": defaultdict(Counter),
        "sub_class": defaultdict(Counter),
        "host": defaultdict(Counter),
    }
    for record in items.values():
        partition = str(record["partition"])
        out["phylum"][clades.get(str(record["assembly"]), "unassigned")][partition] += 1
        out["sub_class"][str(record["sub_class"])][partition] += 1
        out["host"][str(record["assembly"])][partition] += 1
    return {
        axis: {key: dict(sorted(counts.items())) for key, counts in sorted(table.items())}
        for axis, table in out.items()
    }


def _cmd_report(args: argparse.Namespace) -> int:
    """LEG 3: Pin 2 step 3 (content), the content controls, step 4 (D4), the audit sample."""
    root = _root(args)
    out_dir = Path(args.out_dir)
    prep = _read_json(out_dir / "prep.json")
    items = _read_json(out_dir / FILE_ITEMS)["items"]
    if provenance.sha256_file(out_dir / FILE_ITEMS) != prep["files"][FILE_ITEMS]["sha256"]:
        raise HardnegPoolError(f"{FILE_ITEMS} differs from what prep wrote")
    search = _read_json(out_dir / "tblout" / "search.json")
    digests = {name: entry["sha256"] for name, entry in prep["fasta"].items()}
    grouped = hs.read_searches(out_dir / "tblout", search, fasta_digests=digests)
    models = [model for model, _cm in hs.SCREEN_CMS]
    threshold = hs.CONTENT_THRESHOLD_BITS

    by_shard: dict[str, list[str]] = defaultdict(list)
    for record in items.values():
        if record["partition"] is None:
            by_shard[record["shard"]].append(record["surrogate"])
    bits: dict[str, dict[str, float | None]] = {}
    for shard, names in by_shard.items():
        bits.update(hs._bits_per_record(names, grouped, shard, models))
    for record in items.values():
        if record["partition"] is not None:
            continue
        record["bits"] = bits[record["surrogate"]]
        if hs.content_refused(record["bits"], record["motif"], threshold):
            record["partition"] = P_CONTENT

    controls = list(prep["controls"])
    surrogates = [meta["surrogate"] for meta in prep["control_meta"].values()]
    control_bits = hs._bits_per_record(surrogates, grouped, FASTA_CONTROLS, models)
    for meaning, expected in ((hs.CONTROL_CONTENT, True), (hs.CONTROL_CONTENT_HOST, False)):
        meta = prep["control_meta"][meaning]
        observed = hs.content_refused(control_bits[meta["surrogate"]], meta["motif"], threshold)
        controls.append(
            hs.control_entry(
                meaning,
                (
                    "content screen (A15 Pin 2): a catalogued RF00230 T-box spliced into the "
                    "clean control host"
                    if expected
                    else "the same host, unspliced, must be admitted (the matched control)"
                ),
                expected,
                observed,
                note=f"best bits {control_bits[meta['surrogate']]!r}, motif={bool(meta['motif'])}",
            )
        )

    # ── Pin 2 step 4: the D4-context partition, host by host ──────────────────────────────
    targets = {str(t["accession"]): t for t in annotated_hosts(root, args.supply)}
    survivors_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in items.values():
        if record["partition"] is None:
            survivors_by_host[str(record["assembly"])].append(record)
    for assembly, records in sorted(survivors_by_host.items()):
        _features, cds, _contigs, _census = load_host(
            root, targets[assembly], annotations=args.annotations, genomes=args.genomes
        )
        intervals = d4_intervals(cds)
        for record in records:
            record["d4_context"] = is_d4_context(record, intervals)
            record["partition"] = P_BESIDE if record["d4_context"] else P_GRADED

    graded = sorted(cid for cid, rec in items.items() if rec["partition"] == P_GRADED)
    beside = sorted(cid for cid, rec in items.items() if rec["partition"] == P_BESIDE)
    audited = audit_sample(graded)
    infernal.write_fasta(
        {hs.surrogate("a", i): items[cid]["sequence"] for i, cid in enumerate(audited, start=1)},
        out_dir / FASTA_AUDIT,
    )
    keep = ("item_id", "assembly", "seqid", "start", "end", "strand", "feature_type")
    keep += ("feature_id", "delta", "sub_class", "d4_context", "sha256", "sequence")
    for name, ids in (("graded", graded), ("beside", beside)):
        write_json(
            out_dir / f"{name}.json",
            {"partition": name, "items": [{k: items[cid][k] for k in keep} for cid in ids]},
        )
    interim = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": KIND_INTERIM,
        "adr": ADR,
        "prep": {
            "path": hs.recorded_path(out_dir / "prep.json", root),
            "sha256": provenance.sha256_file(out_dir / "prep.json"),
        },
        "search": {
            "path": hs.recorded_path(out_dir / "tblout" / "search.json", root),
            "sha256": provenance.sha256_file(out_dir / "tblout" / "search.json"),
            "infernal_version": search.get("infernal_version"),
            "wall_s": search.get("wall_s"),
        },
        "partition_counts": partition_counts(items),
        "fold_refused_by_rule": dict(
            Counter(
                (
                    "kmer_and_coord"
                    if r["fold"]["rule_kmer"] and r["fold"]["rule_coord"]
                    else "kmer_only" if r["fold"]["rule_kmer"] else "coord_only"
                )
                for r in items.values()
                if r["partition"] == P_FOLD
            )
        ),
        "content_refused_by_rule": dict(
            Counter(
                "motif" if r["motif"] else "cm"
                for r in items.values()
                if r["partition"] == P_CONTENT
            )
        ),
        "composition": composition(items, _clade_of(root, args.clades)),
        "controls": controls,
        "controls_all_fired": all(entry["fired"] for entry in controls),
        "audit_ids": audited,
        "item_files": {
            name: {
                "path": hs.recorded_path(out_dir / f"{name}.json", root),
                "sha256": provenance.sha256_file(out_dir / f"{name}.json"),
                "n": len(ids),
            }
            for name, ids in (("graded", graded), ("beside", beside))
        },
        "audit_fasta": {
            "path": hs.recorded_path(out_dir / FASTA_AUDIT, root),
            "sha256": provenance.sha256_file(out_dir / FASTA_AUDIT),
        },
        "item_partitions": {cid: rec["partition"] for cid, rec in sorted(items.items())},
    }
    write_json(out_dir / "pool.json", interim)
    counts = interim["partition_counts"]
    print("report: " + ", ".join(f"{k} {v['total']}" for k, v in counts.items()))
    return 0


# ══════════════════════════════════════════════════════════════════════════════════════
# audit-report (data env) → the committed report
# ══════════════════════════════════════════════════════════════════════════════════════
def audit_rows(
    grouped: Mapping[tuple[str, str], Mapping[str, Sequence[hs.Hit]]],
    names: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Every audit hit at or above the content screen's 10 bits, per model, as rows.

    The verdict is **re-derived** from these rows by :func:`validate_report`; a headline count
    alone would certify any threshold ([[gate-clauses-need-re-derivation]]).
    """
    rows: list[dict[str, Any]] = []
    for model, _cm in hs.SCREEN_CMS:
        hits = grouped.get((FASTA_AUDIT, model))
        if hits is None:
            raise HardnegPoolError(f"{FASTA_AUDIT} was not searched with {model}")
        for surrogate_name, target_hits in sorted(hits.items()):
            best = hs.best_bits({model: target_hits})[model]
            if best is not None and best >= hs.CONTENT_THRESHOLD_BITS:
                rows.append({"item_id": names[surrogate_name], "model": model, "bits": float(best)})
    return rows


def audit_block(rows: Sequence[Mapping[str, Any]], n_audited: int, calibration: Mapping) -> dict:
    thresholds = dict(hs.POSTMINT_THRESHOLD_BITS)
    violations = [r for r in rows if float(r["bits"]) >= thresholds[str(r["model"])]]
    floors = calibration.get("null_max_bits") or {}
    clears = {
        model: isinstance(floors.get(model), (int, float)) and thresholds[model] > floors[model]
        for model in thresholds
    }
    return {
        "n_audited": int(n_audited),
        "search_threshold_bits": AUDIT_SEARCH_BITS,
        "threshold_bits": thresholds,
        "calibration": dict(calibration),
        "thresholds_clear_floor": clears,
        "rows_at_or_above_10_bits": list(rows),
        "noise_residue": [r for r in rows if float(r["bits"]) < thresholds[str(r["model"])]],
        "n_violations": len(violations),
        "passes": not violations and all(clears.values()),
    }


def _cmd_audit_report(args: argparse.Namespace) -> int:
    """LEG 5: the audit verdict, then the committed report — written only when it validates."""
    root = _root(args)
    out_dir = Path(args.out_dir)
    prep = _read_json(out_dir / "prep.json")
    interim = _read_json(out_dir / "pool.json")
    search = _read_json(out_dir / "audit_tblout" / "search.json")
    audit_sha = provenance.sha256_file(out_dir / FASTA_AUDIT)
    if audit_sha != interim["audit_fasta"]["sha256"]:
        raise HardnegPoolError(f"{FASTA_AUDIT} differs from what the report leg wrote")
    grouped = hs.read_searches(
        out_dir / "audit_tblout", search, arm="filter_off", fasta_digests={FASTA_AUDIT: audit_sha}
    )
    names = {hs.surrogate("a", i): cid for i, cid in enumerate(interim["audit_ids"], start=1)}
    calibration = hs._calibration_record(root / args.noise_floor, root)
    audit = audit_block(audit_rows(grouped, names), len(names), calibration)
    audit["search"] = {
        "path": hs.recorded_path(out_dir / "audit_tblout" / "search.json", root),
        "sha256": provenance.sha256_file(out_dir / "audit_tblout" / "search.json"),
        "argv": [entry["argv"] for entry in search["searches"]],
        "wall_s": search.get("wall_s"),
    }
    partitions = interim["item_partitions"]
    annex = Path(args.ids_out)
    annex_sha = _write_lines(
        annex,
        (
            f"{cid}\t{part}"
            for cid, part in sorted(partitions.items())
            if part in (P_GRADED, P_BESIDE)
        ),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": KIND_POOL,
        "adr": ADR,
        "prd": PRD,
        "config": prep["config"],
        "population": prep["population"],
        "files": prep["files"],
        "fasta": prep["fasta"],
        "screened_dna": prep["screened_dna"],
        "fold_per_source": prep["fold_per_source"],
        "content_control": prep["content_control"],
        "search": interim["search"],
        "partition_counts": interim["partition_counts"],
        "fold_refused_by_rule": interim["fold_refused_by_rule"],
        "content_refused_by_rule": interim["content_refused_by_rule"],
        "composition": interim["composition"],
        "controls": interim["controls"],
        "controls_all_fired": interim["controls_all_fired"],
        "item_files": interim["item_files"],
        "ids_annex": {"path": hs.recorded_path(annex, root), "sha256": annex_sha},
        "audit_ids": interim["audit_ids"],
        "audit": audit,
        "selection": (
            "ADR-0005 A17: native §9.1 class-4 leaders (CDS-upstream + tRNA-upstream 1,024-nt "
            "windows, δ ~ U{0..256}) on the 339 annotated admissible a2 hosts; seed-42 uniform "
            "draw; A15 Pin 1 + Pin 2 screens unchanged; D4-context windows beside the graded set"
        ),
        "provenance": _provenance(
            "audit-report",
            [out_dir / "prep.json", out_dir / "pool.json", root / args.noise_floor],
            root,
            env_lock=hs.ENV_LOCK_DATA,
        ),
    }
    problems = validate_report(report, root=root)
    if problems:
        invalid = Path(args.out).with_suffix(".invalid.json")
        write_json(invalid, {**report, "problems": problems})
        for problem in problems:
            print(f"REFUSED: {problem}")
        return 3
    write_json(Path(args.out), report)
    print(f"audit-report: {audit['n_audited']} audited, {audit['n_violations']} violations")
    return 0


# ══════════════════════════════════════════════════════════════════════════════════════
# validate_report — every clause re-derived from rows; malformed never reads as absent
# ══════════════════════════════════════════════════════════════════════════════════════
def _int(node: Any) -> int | None:
    return node if isinstance(node, int) and not isinstance(node, bool) else None


def validate_report(report: Any, *, root: str | Path | None = None) -> list[str]:
    """The pool report's clause set. Returns problems; never raises on malformed input."""
    if not isinstance(report, Mapping):
        return ["report is not an object"]
    problems: list[str] = []
    if report.get("kind") != KIND_POOL:
        problems.append(f"kind is {report.get('kind')!r}, not {KIND_POOL!r}")
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version is {report.get('schema_version')!r}")
    config = report.get("config") if isinstance(report.get("config"), Mapping) else {}
    population = report.get("population") if isinstance(report.get("population"), Mapping) else {}
    counts = report.get("partition_counts")
    problems += _config_problems(config, population)
    problems += _conservation_problems(counts, population)
    problems += hs._control_problems(report)
    problems += _audit_problems(report, counts)
    problems += _path_problems(report)
    if root is not None:
        problems += _rederive_problems(report, Path(root))
    return problems


def _config_problems(config: Mapping[str, Any], population: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    for key, pinned in (("seed", SEED), ("delta_max", DELTA_MAX), ("window_nt", WINDOW_NT)):
        if config.get(key) != pinned:
            problems.append(f"config.{key} is {config.get(key)!r}, A17 pins {pinned}")
    # A cost knob must never certify ([[cost-knobs-can-certify]]): a smoke-sized draw or a host
    # cap writes `pinned_draw: false`, and that alone refuses the report.
    if config.get("pinned_draw") is not True or config.get("max_hosts") is not None:
        problems.append("the draw is not A17's pinned draw (draw_size / max_hosts override)")
    if config.get("draw_size") != DRAW_SIZE:
        problems.append(f"config.draw_size is {config.get('draw_size')!r}, A17 pins {DRAW_SIZE}")
    mode = config.get("mode")
    n_eligible, n_selected = _int(population.get("n_eligible")), _int(population.get("n_selected"))
    if n_eligible is None or n_selected is None:
        problems.append("population.n_eligible / n_selected are missing or not integers")
    elif mode == MODE_DRAW and n_selected != DRAW_SIZE:
        problems.append(f"draw selected {n_selected}, not {DRAW_SIZE}")
    elif mode == MODE_CENSUS and n_selected != n_eligible - DRAW_SIZE:
        problems.append(
            f"census selected {n_selected}, not eligible − draw = {n_eligible - DRAW_SIZE}"
        )
    elif mode not in (MODE_DRAW, MODE_CENSUS):
        problems.append(f"config.mode is {mode!r}")
    return problems


def _conservation_problems(counts: Any, population: Mapping[str, Any]) -> list[str]:
    if not isinstance(counts, Mapping) or set(counts) != set(PARTITIONS):
        return [f"partition_counts must carry exactly {list(PARTITIONS)}"]
    problems: list[str] = []
    totals: dict[str, int] = {}
    for name in PARTITIONS:
        block = counts[name]
        if not isinstance(block, Mapping) or _int(block.get("total")) is None:
            problems.append(f"partition_counts.{name}.total is missing or not an integer")
            continue
        parts = {k: v for k, v in block.items() if k != "total"}
        if any(_int(v) is None or v < 0 for v in parts.values()):
            problems.append(f"partition_counts.{name} has a non-count sub-class entry")
        elif sum(parts.values()) != block["total"]:
            problems.append(f"partition_counts.{name}: sub-classes sum to {sum(parts.values())}")
        totals[name] = int(block["total"])
    selected = _int(population.get("n_selected"))
    if len(totals) == len(PARTITIONS) and selected is not None and sum(totals.values()) != selected:
        problems.append(
            f"conservation: {sum(totals.values())} partitioned windows, {selected} selected"
        )
    if totals.get(P_GRADED, 0) <= 0:
        problems.append("no graded negative: the amendment would have no denominator")
    return problems


def _audit_problems(report: Mapping[str, Any], counts: Any) -> list[str]:
    audit = report.get("audit")
    if not isinstance(audit, Mapping):
        return ["audit block is missing"]
    problems: list[str] = []
    ids = report.get("audit_ids")
    graded_block = counts.get(P_GRADED) if isinstance(counts, Mapping) else None
    n_graded = graded_block.get("total") if isinstance(graded_block, Mapping) else None
    if not isinstance(ids, list) or len(set(map(str, ids))) != len(ids):
        problems.append("audit_ids is not a list of distinct ids")
    elif isinstance(n_graded, int) and len(ids) != min(AUDIT_SIZE, n_graded):
        problems.append(f"audit has {len(ids)} ids, not min({AUDIT_SIZE}, n_graded={n_graded})")
    if audit.get("n_audited") != (len(ids) if isinstance(ids, list) else None):
        problems.append("audit.n_audited disagrees with audit_ids")
    if audit.get("threshold_bits") != dict(hs.POSTMINT_THRESHOLD_BITS):
        problems.append("audit.threshold_bits is not A15 Pin 3's per-model table")
    if audit.get("search_threshold_bits") != AUDIT_SEARCH_BITS:
        problems.append("audit was not searched from A17's 0-bit floor")
    rows = audit.get("rows_at_or_above_10_bits")
    if not isinstance(rows, list):
        return [*problems, "audit rows are missing"]
    thresholds = dict(hs.POSTMINT_THRESHOLD_BITS)
    try:
        violations = sum(float(r["bits"]) >= thresholds[str(r["model"])] for r in rows)
    except (KeyError, TypeError, ValueError):
        return [*problems, "an audit row is malformed"]
    if audit.get("n_violations") != violations:
        problems.append(f"audit.n_violations disagrees with its rows ({violations})")
    floors = (audit.get("calibration") or {}).get("null_max_bits") or {}
    clears = all(
        isinstance(floors.get(m), (int, float)) and thresholds[m] > floors[m] for m in thresholds
    )
    derived = violations == 0 and clears
    if audit.get("passes") is not derived:
        problems.append("audit.passes disagrees with its rows and calibration")
    if not derived:
        problems.append(
            f"the score-blind audit FAILS ({violations} hit(s) at or above A15 Pin 3's thresholds"
            f"{'' if clears else '; a threshold does not clear its measured floor'}): "
            "A15 Pin 3's §7 stop is the only remedy"
        )
    return problems


def _path_problems(report: Mapping[str, Any]) -> list[str]:
    text = json.dumps(report.get("files", {})) + json.dumps(report.get("item_files", {}))
    text += json.dumps(report.get("fasta", {})) + json.dumps(report.get("ids_annex", {}))
    return ["a recorded path is absolute (public repo)"] if '"path": "/' in text else []


def _located(recorded: str, root: Path) -> Path:
    """A recorded repo-relative path: a ``data/`` file under the data root ONLY; a committed
    file under the data root, else the code root.

    :func:`host_screen.recorded_path` records a committed annex relative to the code root and an
    interim file relative to the data root; on ``main`` the two are the same checkout. A
    ``data/`` file has no code-root fallback: a worktree can hold its own staged copy at the same
    relative path, and reading that in place of a file missing from the data root would validate
    the report against a different population.
    """
    candidate = Path(root) / recorded
    if candidate.exists() or PurePosixPath(recorded).parts[:1] == ("data",):
        return candidate
    fallback = Path(hs.CODE_ROOT) / recorded
    return fallback if fallback.exists() else candidate


def _rederive_problems(report: Mapping[str, Any], root: Path) -> list[str]:
    """With the data root: replay the RNG and re-read the annex. Missing ⇒ problem, not pass."""
    problems: list[str] = []
    files = report.get("files") if isinstance(report.get("files"), Mapping) else {}
    config = report.get("config") if isinstance(report.get("config"), Mapping) else {}
    try:
        feature_rows = _located(files[FILE_FEATURES]["path"], root).read_text().splitlines()
        eligible_text = _located(files[FILE_ELIGIBLE]["path"], root).read_text()
        drawn = _located(files[FILE_DRAWN]["path"], root).read_text().splitlines()
        deltas = [int(row.rsplit("\t", 1)[1]) for row in feature_rows]
        eligible = [
            dict(
                zip(
                    ("item_id", "feature_type", "feature_id", "delta"),
                    line.split("\t"),
                    strict=True,
                )
            )
            for line in eligible_text.splitlines()
        ]
        replayed_deltas, selected = replay_selection(
            len(feature_rows),
            eligible,
            mode=str(config.get("mode")),
            size=int(config.get("draw_size")),
        )
    except (KeyError, TypeError, ValueError, IndexError, OSError, HardnegPoolError) as exc:
        return [f"cannot re-derive the draw: {exc!r}"]
    if hashlib.sha256(eligible_text.encode()).hexdigest() != files[FILE_ELIGIBLE].get("sha256"):
        problems.append("eligible list differs from its recorded sha256")
    if replayed_deltas != deltas:
        problems.append("the δ column does not replay from seed 42")
    if sorted(str(w["item_id"]) for w in selected) != drawn:
        problems.append("the drawn list does not replay from seed 42")
    annex = report.get("ids_annex") if isinstance(report.get("ids_annex"), Mapping) else {}
    try:
        annex_path = _located(annex["path"], root)
        rows = [line.split("\t") for line in annex_path.read_text().splitlines()]
        graded = {cid for cid, part in rows if part == P_GRADED}
        beside = {cid for cid, part in rows if part == P_BESIDE}
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return [*problems, f"cannot read the ids annex: {exc!r}"]
    if provenance.sha256_file(annex_path) != annex.get("sha256"):
        problems.append("the ids annex differs from its recorded sha256")
    counts = report.get("partition_counts") or {}
    if len(graded) != (counts.get(P_GRADED) or {}).get("total"):
        problems.append("graded ids in the annex disagree with partition_counts")
    if len(beside) != (counts.get(P_BESIDE) or {}).get("total"):
        problems.append("beside ids in the annex disagree with partition_counts")
    if not set(map(str, report.get("audit_ids") or [])) <= graded:
        problems.append("an audited id is not a graded id")
    if not (graded | beside) <= set(drawn):
        problems.append("a graded or beside id is not a selected window")
    return problems


# ══════════════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tbox_finder.integration.hardneg_pool")
    sub = parser.add_subparsers(dest="leg", required=True)

    prep = sub.add_parser("prep", help="LEG 1 (data env): population, draw, gap + fold screen")
    hs._data_root_arg(prep)
    prep.add_argument("--out-dir", required=True)
    prep.add_argument("--supply", default=DEFAULT_SUPPLY)
    prep.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    prep.add_argument("--genomes", default=DEFAULT_GENOMES)
    for flag, default in (
        ("--mining-pool", "data/processed/negatives/mining_pool_v0.parquet"),
        ("--decoys", "data/processed/negatives/decoys_v0.parquet"),
        ("--dataset", "data/processed/stage2_dataset.parquet"),
        ("--context", "data/interim/flank_context/context_v0.parquet"),
        ("--labels", "data/processed/labels/labels_v0.parquet"),
        ("--split-table", "data/processed/splits/split_assignments.parquet"),
        (
            "--stage2-provenance",
            "data/processed/checkpoints/stage2_rinalmo/aux1.0_lr1e-4/provenance.json",
        ),
        ("--rf00230-master", "data/external/refs/RF00230_master.fa"),
    ):
        prep.add_argument(flag, default=default)
    prep.add_argument("--mode", choices=(MODE_DRAW, MODE_CENSUS), default=MODE_DRAW)
    prep.add_argument(
        "--draw-size",
        type=int,
        default=DRAW_SIZE,
        help="smoke override only: any value but 100,000 writes pinned_draw=false, which refuses",
    )
    prep.add_argument("--max-hosts", type=int, default=None, help="smoke override; refuses too")
    prep.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    prep.set_defaults(func=_cmd_prep)

    for name, func, help_text in (
        ("search", _cmd_search, "LEG 2 (infernal env): A15 Pin 2's cmsearch over the survivors"),
        ("audit-search", _cmd_audit_search, "LEG 4 (infernal env): the audit at --max -T 0"),
    ):
        leg = sub.add_parser(name, help=help_text)
        hs._data_root_arg(leg)
        leg.add_argument("--out-dir", required=True)
        leg.add_argument("--jobs", type=int, default=8)
        leg.add_argument("--cpu", type=int, default=3)
        leg.add_argument("--timeout-s", type=float, default=6 * 3600.0)
        leg.set_defaults(func=func)

    report = sub.add_parser("report", help="LEG 3 (data env): content screen + D4 partition")
    hs._data_root_arg(report)
    report.add_argument("--out-dir", required=True)
    report.add_argument("--supply", default=DEFAULT_SUPPLY)
    report.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    report.add_argument("--genomes", default=DEFAULT_GENOMES)
    report.add_argument("--clades", default=DEFAULT_CLADES)
    report.set_defaults(func=_cmd_report)

    audit = sub.add_parser("audit-report", help="LEG 5 (data env): the committed report")
    hs._data_root_arg(audit)
    audit.add_argument("--out-dir", required=True)
    audit.add_argument("--noise-floor", default=DEFAULT_NOISE_FLOOR)
    audit.add_argument("--out", required=True)
    audit.add_argument("--ids-out", required=True)
    audit.set_defaults(func=_cmd_audit_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


__all__ = [
    "AUDIT_SIZE",
    "D4_CONTEXT_PAD_BP",
    "DELTA_MAX",
    "DRAW_SIZE",
    "Feature",
    "HardnegPoolError",
    "KIND_POOL",
    "PARTITIONS",
    "SEED",
    "assign_deltas",
    "audit_block",
    "audit_sample",
    "eligible_digest",
    "eligible_windows",
    "is_d4_context",
    "partition_counts",
    "replay_selection",
    "select",
    "validate_report",
    "window_span",
]

if __name__ == "__main__":
    raise SystemExit(main())
