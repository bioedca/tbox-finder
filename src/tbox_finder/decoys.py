"""§9.1 static decoy / negative-pool construction (P0-30).

Builds the **four §9.1 training/benchmark negative classes** (PRD §9.1) and — via
:mod:`tbox_finder.masking` — masks the full union prior + own positives + a flank
from every pool and reports the residual contamination against the union
denominator:

1. **GC + length-matched genomic background** — the dominant real-world negative.
   *P0 realisation:* no genome panel is staged at P0 (genomes are the P5/SLURM
   scan corpora, §7.2), so this pool is a **seeded 0th-order composition null**
   whose ``(length, GC)`` pairs are bootstrapped from the P0-12 T-box GC/length
   joint distribution and whose bases are emitted i.i.d. at the matched GC — the
   faithful realisation of PRD pool-1 ("random windows matched to T-box GC +
   length") given the available inputs. The genome-window realisation and the
   three genome-scale FDR nulls are a distinct, later set (§12/P5; these four are
   training/benchmark negatives, not the FDR nulls).
2. **Other structured RNAs (Rfam)** — SAM/TPP/tRNA/other riboswitches, a seeded
   subsample of a checksummed Rfam FASTA fetch (EBI FTP ``fasta_files``). Forces
   T-box-specific structure, not merely "structured RNA".
3. **Dinucleotide-shuffled decoys** — a seeded **Altschul & Erikson (1985)**
   random-Eulerian-walk shuffle of the T-box positives (preserves exact
   dinucleotide composition; algorithmically identical to ``esl-shuffle -d``).
   Implemented in pure Python so the pool + its golden fixture are deterministic
   and reproducible in any env (no cross-env Easel dependency). *Caveat (PRD §9.1):*
   dinucleotide shuffling may not preserve structured-RNA background — the null
   must reproduce ``cmsearch`` decoy behaviour before it is trusted for FDR
   (validated at §12/P5, a distinct estimator set). Ref: Altschul & Erikson,
   Mol. Biol. Evol. 2(6):526-538, 1985 [DOI:10.1093/oxfordjournals.molbev.a040370].
4. **5'UTR / tRNA-adjacent leader decoys** — the hardest false-positive context
   for a tRNA-sensing finder; the fixed external anchor is the tboxevo
   ``idtm_validation_negatives.fasta`` (PRD §9.1 "reuse the existing tboxevo null
   DBs + validation negatives as a fixed external negative anchor").

Pure-stdlib generators/parsers on top are unit-testable without pandas/biopython;
the fetch + full build (lazy pandas) are below. CLI: ``fetch-refs`` (one-time
network stage into ``data/external/refs/decoys/``) and ``build`` (network-free,
seeded, data env). context7 §3.3: Biopython ``gc_fraction``/``SeqIO`` consulted —
the trivial GC + FASTA primitives are reimplemented in stdlib for bare-env
testability, matching ``gc_fraction``'s (G+C)/len definition.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import urllib.request
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import ingest, masking, provenance

# --------------------------------------------------------------------------- #
# Paths / pins
# --------------------------------------------------------------------------- #
_NEG_DIR = "data/processed/negatives"
_AUDIT_DIR = "data/processed/audits"
_REFS_DIR = "data/external/refs/decoys"
DECOYS_PARQUET = f"{_NEG_DIR}/decoys_v0.parquet"
DECOYS_PROVENANCE = f"{_NEG_DIR}/decoys_v0.provenance.json"
DECOYS_REPORT = f"{_AUDIT_DIR}/decoys_report.json"
STRUCTURED_REFS_FA = f"{_REFS_DIR}/structured_rna_refs.fa"
LEADER_REFS_FA = f"{_REFS_DIR}/leader_decoys.fa"
REFS_MANIFEST = f"{_REFS_DIR}/decoy_refs.manifest.json"

# Corpus columns (P0-12 master_clean_v0).
GC_COL = "GC"
LEN_COL = "tbox_length"
SEQ_COL = "FASTA_sequence"

# The four §9.1 pool names.
POOL_GC = "gc_background"
POOL_STRUCTURED = "structured_rna"
POOL_DINUC = "dinuc_shuffled"
POOL_LEADER = "leader_decoy"
POOL_NAMES = (POOL_GC, POOL_STRUCTURED, POOL_DINUC, POOL_LEADER)

# Rfam families for the structured-RNA pool (SAM/TPP/tRNA + other riboswitches;
# PRD §9.1). name → RFAM accession; sequences fetched from EBI FTP fasta_files.
RFAM_FAMILIES: dict[str, str] = {
    "SAM": "RF00162",
    "TPP": "RF00059",
    "tRNA": "RF00005",
    "purine": "RF00167",
    "FMN": "RF00050",
    "glycine": "RF00504",
}
RFAM_FASTA_URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/{acc}.fa.gz"
# Fixed external leader/tRNA-adjacent anchor (tboxevo validation negatives).
TBOXEVO_NEGATIVES = (
    "/home/bioedca/tbox-phylogeny/tboxevo/data/processed/cms/idtm_validation_negatives.fasta"
)

# Seeded-config defaults (conf/data/decoys.yaml overrides).
DEFAULT_GC_BACKGROUND_N = 2000
DEFAULT_DINUC_N_SOURCES = 1000
DEFAULT_DINUC_PER_SOURCE = 2
DEFAULT_RFAM_PER_FAMILY_CAP = 500
# Cap on records scanned per family before subsampling — bounds download + memory
# for huge families (e.g. tRNA RF00005 full is ~950k seqs / 171 MB gz). The gz is
# streamed lazily, so an early stop downloads only the scanned prefix.
DEFAULT_RFAM_MAX_SCAN = 20000


# --------------------------------------------------------------------------- #
# Pure-stdlib sequence primitives (unit-testable without pandas/biopython)
# --------------------------------------------------------------------------- #
_ALPHABET = ("A", "C", "G", "T")


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Minimal FASTA parser → ``[(header, uppercase_sequence), ...]`` (stdlib)."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks).upper()))
            header = line[1:].strip()
            chunks = []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    return records


def write_fasta(records: Sequence[tuple[str, str]]) -> str:
    """Serialize ``[(header, sequence), ...]`` to FASTA text (one line per sequence)."""
    return "".join(f">{h}\n{s}\n" for h, s in records)


def gc_fraction(seq: str) -> float:
    """(G + C + S) / length — matches Bio.SeqUtils.gc_fraction for unambiguous seqs."""
    if not seq:
        return 0.0
    up = seq.upper()
    gc = sum(up.count(b) for b in ("G", "C", "S"))
    return gc / len(up)


def dinucleotide_counts(seq: str) -> Counter[str]:
    """Counter of the ordered adjacent dinucleotides in ``seq`` (uppercased)."""
    up = seq.upper()
    return Counter(up[i : i + 2] for i in range(len(up) - 1))


def gc_matched_sequence(
    length: int, gc: float, rng: random.Random, alphabet: Sequence[str] = _ALPHABET
) -> str:
    """Emit a length-``length`` i.i.d. sequence at target GC fraction ``gc`` (0th order).

    P(G) = P(C) = gc/2 and P(A) = P(T) = (1 - gc)/2 over ``alphabet`` = (A,C,G,T).
    Deterministic given ``rng``.
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    if not 0.0 <= gc <= 1.0:
        raise ValueError(f"gc must be in [0, 1], got {gc}")
    a, c, g, t = alphabet
    weights = [(1.0 - gc) / 2.0, gc / 2.0, gc / 2.0, (1.0 - gc) / 2.0]  # A,C,G,T
    letters = (a, c, g, t)
    return "".join(rng.choices(letters, weights=weights, k=length))


def dinucleotide_shuffle(seq: str, rng: random.Random) -> str:
    """Altschul & Erikson (1985) random-Eulerian-walk dinucleotide shuffle.

    Returns a permutation of ``seq`` preserving its **exact dinucleotide
    composition** and its first & last symbol. Deterministic given ``rng``.
    Algorithm: pick one random "last edge" out of every vertex except the terminal
    symbol; retry until those last edges form an arborescence rooted at the
    terminal symbol; randomly permute the remaining out-edges (last edge appended);
    then traverse the resulting Eulerian walk from the first symbol.
    """
    s = seq.upper()
    n = len(s)
    if n < 2:
        return s
    last = s[-1]
    # Deterministic vertex order (sorted, not set-iteration) so the sequence in which
    # the rng is consumed is independent of PYTHONHASHSEED → reproducible shuffle.
    vertices = sorted(set(s))
    out_edges: dict[str, list[str]] = {v: [] for v in vertices}
    for a, b in zip(s[:-1], s[1:], strict=True):
        out_edges[a].append(b)

    while True:
        last_edge: dict[str, str] = {}
        remaining: dict[str, list[str]] = {}
        for v in vertices:
            outs = out_edges[v]
            if v == last or not outs:
                remaining[v] = list(outs)
                continue
            idx = rng.randrange(len(outs))
            last_edge[v] = outs[idx]
            remaining[v] = outs[:idx] + outs[idx + 1 :]
        if _last_edges_form_arborescence(last_edge, vertices, last):
            break

    walk: dict[str, list[str]] = {}
    for v in vertices:
        rest = remaining[v]
        rng.shuffle(rest)
        if v in last_edge:
            rest.append(last_edge[v])
        walk[v] = rest

    result = [s[0]]
    ptr: dict[str, int] = {v: 0 for v in vertices}
    cur = s[0]
    for _ in range(n - 1):
        nxt = walk[cur][ptr[cur]]
        ptr[cur] += 1
        result.append(nxt)
        cur = nxt
    return "".join(result)


def _last_edges_form_arborescence(
    last_edge: dict[str, str], vertices: Sequence[str], last: str
) -> bool:
    """True iff following ``last_edge`` from every vertex reaches ``last`` acyclically."""
    for v in vertices:
        if v == last or v not in last_edge:
            continue
        seen = {v}
        cur = last_edge[v]
        while cur != last:
            if cur in seen or cur not in last_edge:
                return False
            seen.add(cur)
            cur = last_edge[cur]
    return True


# --------------------------------------------------------------------------- #
# Corpus-derived pools (deterministic, network-free — used by the golden fixture)
# --------------------------------------------------------------------------- #
def build_corpus_pools(
    gc_values: Sequence[float],
    lengths: Sequence[int],
    sequences: Sequence[str],
    *,
    seed: int,
    n_gc: int,
    n_dinuc_sources: int,
    dinuc_per_source: int,
) -> list[dict[str, Any]]:
    """Build the two corpus-derived pools (GC-matched background + dinuc-shuffle).

    Deterministic given ``seed`` and the input positives; needs no network/refs, so
    it is the reproducible basis of the golden fixture. ``gc_values``/``lengths``
    are the per-positive T-box GC/length (P0-12) bootstrapped for pool 1;
    ``sequences`` are the T-box positive sequences shuffled for pool 3.
    """
    if not (len(gc_values) == len(lengths) == len(sequences)):
        raise ValueError("gc_values, lengths, sequences must be the same length")
    if not sequences:
        raise ValueError("no positive sequences supplied")
    records: list[dict[str, Any]] = []

    # Pool 1 — GC + length-matched 0th-order background (bootstrap the joint dist).
    rng_gc = random.Random(f"{seed}:gc_background")
    joint = list(zip((float(x) for x in gc_values), (int(x) for x in lengths), strict=True))
    for i in range(int(n_gc)):
        gc, length = rng_gc.choice(joint)
        seq = gc_matched_sequence(length, gc, rng_gc)
        records.append(_record(POOL_GC, f"gcbg_{i:06d}", seq, "gc_length_matched_0th_order"))

    # Pool 3 — dinucleotide-shuffled positives (Altschul-Erikson).
    rng_dn = random.Random(f"{seed}:dinuc_shuffled")
    n_src = min(int(n_dinuc_sources), len(sequences))
    src_idx = sorted(rng_dn.sample(range(len(sequences)), n_src))
    for i in src_idx:
        src = str(sequences[i]).upper()
        for k in range(int(dinuc_per_source)):
            shuf = dinucleotide_shuffle(src, rng_dn)
            records.append(
                _record(POOL_DINUC, f"dinuc_{i:06d}_{k}", shuf, "dinuc_shuffle_altschul_erikson")
            )
    return records


def _record(pool: str, decoy_id: str, seq: str, source: str) -> dict[str, Any]:
    """One decoy pool record. Synthetic/external pools carry no genomic coordinates."""
    return {
        "pool": pool,
        "decoy_id": decoy_id,
        "sequence": seq,
        "length": len(seq),
        "gc": gc_fraction(seq),
        "source": source,
        "accession": None,
        "locus_start": None,
        "locus_end": None,
    }


def decoys_digest(records: Sequence[dict[str, Any]]) -> str:
    """Whole-artifact golden digest over ``(pool, decoy_id, sequence)`` per record.

    Reuses the ingest hashing contract (kind-tagged, length-framed, row-order-
    dependent), so the digest depends only on the construction logic + seed, not on
    parquet/pyarrow bytes, and is reproducible in any env with the inputs.
    """
    per_record = [ingest.record_hash([r["pool"], r["decoy_id"], r["sequence"]]) for r in records]
    return ingest.records_digest(per_record)


# --------------------------------------------------------------------------- #
# Seeded-config reader (dependency-free scalar reader; mirrors coverage._read_config)
# --------------------------------------------------------------------------- #
class _Config:
    __slots__ = (
        "seed",
        "flank_nt",
        "gc_background_n",
        "dinuc_n_sources",
        "dinuc_per_source",
        "rfam_per_family_cap",
        "rfam_max_scan",
    )

    def __init__(self) -> None:
        self.seed = provenance.DEFAULT_SEED
        self.flank_nt = masking.DEFAULT_FLANK_NT
        self.gc_background_n = DEFAULT_GC_BACKGROUND_N
        self.dinuc_n_sources = DEFAULT_DINUC_N_SOURCES
        self.dinuc_per_source = DEFAULT_DINUC_PER_SOURCE
        self.rfam_per_family_cap = DEFAULT_RFAM_PER_FAMILY_CAP
        self.rfam_max_scan = DEFAULT_RFAM_MAX_SCAN


def read_config(config: str | Path | None) -> _Config:
    """Read seed + pool knobs from the seeded config (falls back to the pins)."""
    cfg = _Config()
    if config is None:
        return cfg
    path = Path(config)
    if not path.exists():
        return cfg
    int_keys = {
        "seed": "seed",
        "flank_nt": "flank_nt",
        "gc_background_n": "gc_background_n",
        "dinuc_n_sources": "dinuc_n_sources",
        "dinuc_per_source": "dinuc_per_source",
        "rfam_per_family_cap": "rfam_per_family_cap",
        "rfam_max_scan": "rfam_max_scan",
    }
    for line in path.read_text().splitlines():
        raw = line.split("#", 1)[0].strip()
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        key, val = key.strip(), val.strip()
        if key in int_keys and val:
            setattr(cfg, int_keys[key], int(val))
    return cfg


# --------------------------------------------------------------------------- #
# fetch-refs (one-time network stage into data/external/refs/decoys/)
# --------------------------------------------------------------------------- #
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stream_fasta_gz(url: str, timeout: int = 180):
    """Yield ``(header, uppercase_seq)`` from a gzipped FASTA URL, decompressed lazily.

    ``gzip.GzipFile`` reads the response on demand, so a caller that stops early
    only downloads/decompresses the consumed prefix — bounding cost on huge Rfam
    families.
    """
    with (
        urllib.request.urlopen(url, timeout=timeout) as resp,  # noqa: S310 - pinned https EBI host
        gzip.GzipFile(fileobj=resp) as gz,
    ):
        header: str | None = None
        chunks: list[str] = []
        for raw in gz:
            line = raw.decode("ascii", errors="strict").rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
            elif header is not None:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks).upper()


def _reservoir_sample_stream(iterable, k: int, rng: random.Random, max_scan: int):
    """Reservoir-sample ``k`` items from the first ``max_scan`` of ``iterable``.

    Returns ``(sample, n_scanned)``. Bounds work/download when ``iterable`` is a
    lazily-decompressed stream that may be far larger than ``max_scan``.
    """
    reservoir: list = []
    n_scanned = 0
    for item in iterable:
        if n_scanned >= max_scan:
            break
        n_scanned += 1
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randrange(n_scanned)
            if j < k:
                reservoir[j] = item
    return reservoir, n_scanned


def fetch_refs(
    *,
    refs_dir: str | Path = _REFS_DIR,
    config: str | Path | None = None,
    tboxevo_negatives: str | Path = TBOXEVO_NEGATIVES,
) -> int:
    """Stage the structured-RNA (Rfam) + leader (tboxevo) decoy sources (checksummed).

    Downloads each :data:`RFAM_FAMILIES` FASTA (EBI FTP), seeded-subsamples to the
    per-family cap, writes a combined ``structured_rna_refs.fa``; copies the
    tboxevo validation negatives to ``leader_decoys.fa``; and writes a checksum
    manifest (per-file sha256, source URL/path, Rfam release, accessed date). The
    ``build`` step is then network-free.
    """
    cfg = read_config(config)
    out = Path(refs_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{cfg.seed}:rfam_subsample")
    accessed = datetime.now(UTC).isoformat()
    manifest: dict[str, Any] = {
        "accessed_utc": accessed,
        "seed": cfg.seed,
        "rfam_per_family_cap": cfg.rfam_per_family_cap,
        "rfam_max_scan": cfg.rfam_max_scan,
        "families": {},
    }

    structured: list[tuple[str, str]] = []
    for name, acc in RFAM_FAMILIES.items():
        url = RFAM_FASTA_URL.format(acc=acc)
        print(f"fetch {name} ({acc}) <- {url}", file=sys.stderr)
        picked, n_scanned = _reservoir_sample_stream(
            _stream_fasta_gz(url), cfg.rfam_per_family_cap, rng, cfg.rfam_max_scan
        )
        picked.sort()  # stable order for a reproducible written artifact
        for h, seq in picked:
            structured.append((f"{name}|{acc}|{h}", seq))
        manifest["families"][name] = {
            "rfam_accession": acc,
            "url": url,
            "n_sequences_scanned": n_scanned,
            "scan_capped": n_scanned >= cfg.rfam_max_scan,
            "n_sequences_subsampled": len(picked),
        }
    structured_text = write_fasta(structured)
    Path(STRUCTURED_REFS_FA).write_text(structured_text)
    manifest["structured_rna_refs"] = {
        "path": STRUCTURED_REFS_FA,
        "sha256": _sha256_bytes(structured_text.encode()),
        "n_sequences": len(structured),
    }

    neg_text = Path(tboxevo_negatives).read_text()
    leader_recs = parse_fasta(neg_text)
    leader_text = write_fasta(leader_recs)
    Path(LEADER_REFS_FA).write_text(leader_text)
    manifest["leader_decoys"] = {
        "path": LEADER_REFS_FA,
        "source_path": str(tboxevo_negatives),
        "source_sha256": _sha256_bytes(neg_text.encode()),
        "sha256": _sha256_bytes(leader_text.encode()),
        "n_sequences": len(leader_recs),
    }
    Path(REFS_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"staged {len(structured)} structured-RNA + {len(leader_recs)} leader refs "
        f"→ {refs_dir}",
        file=sys.stderr,
    )
    return 0


# --------------------------------------------------------------------------- #
# build (network-free, seeded, data env)
# --------------------------------------------------------------------------- #
def build(
    *,
    corpus_parquet: str | Path,
    union_parquet: str | Path,
    structured_refs: str | Path = STRUCTURED_REFS_FA,
    leader_refs: str | Path = LEADER_REFS_FA,
    out_parquet: str | Path = DECOYS_PARQUET,
    provenance_path: str | Path = DECOYS_PROVENANCE,
    report_path: str | Path = DECOYS_REPORT,
    config: str | Path | None = None,
    env_lock: str | Path | None = None,
) -> int:
    """Build the four §9.1 pools, mask against the union prior + own positives, report."""
    import pandas as pd

    cfg = read_config(config)

    corpus = pd.read_parquet(corpus_parquet, columns=[GC_COL, LEN_COL, SEQ_COL])
    gc_values = corpus[GC_COL].astype(float).tolist()
    lengths = corpus[LEN_COL].astype(int).tolist()
    sequences = corpus[SEQ_COL].astype(str).tolist()

    records = build_corpus_pools(
        gc_values,
        lengths,
        sequences,
        seed=cfg.seed,
        n_gc=cfg.gc_background_n,
        n_dinuc_sources=cfg.dinuc_n_sources,
        dinuc_per_source=cfg.dinuc_per_source,
    )
    # Pool 2 — structured RNAs (staged Rfam subsample).
    for h, seq in parse_fasta(Path(structured_refs).read_text()):
        records.append(_record(POOL_STRUCTURED, h, seq, "rfam_fasta_subsample"))
    # Pool 4 — leader/tRNA-adjacent decoys (staged tboxevo validation negatives).
    for h, seq in parse_fasta(Path(leader_refs).read_text()):
        records.append(_record(POOL_LEADER, h, seq, "tboxevo_validation_negatives"))

    # --- Loci masking: union prior + own positives + flank, from every pool. ---
    union_loci, n_union_total, n_union_maskable = masking.load_union_loci(union_parquet)
    own_loci = masking.load_own_positive_loci(corpus_parquet)
    index = masking.LocusIndex.from_records(union_loci + own_loci)
    pool_mask_counts = dict.fromkeys(POOL_NAMES, 0)
    for r in records:
        masked = (
            index.is_masked(r["accession"], r["locus_start"], r["locus_end"], flank=cfg.flank_nt)
            if r["accession"] is not None
            else False
        )
        r["masked"] = masked
        r["mask_reason"] = "union_prior_or_own_positive_flank" if masked else None
        if masked:
            pool_mask_counts[r["pool"]] += 1

    residual = masking.residual_contamination_report(
        n_union_total=n_union_total,
        n_union_maskable=n_union_maskable,
        pool_mask_counts=pool_mask_counts,
    )
    digest = decoys_digest(records)

    # --- Write the artifact (parquet), the audit report, and provenance. ---
    df = pd.DataFrame.from_records(records)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    pool_counts = {p: int((df["pool"] == p).sum()) for p in POOL_NAMES}
    pool_retained = {p: int(((df["pool"] == p) & (~df["masked"])).sum()) for p in POOL_NAMES}
    report = {
        "n_records": len(df),
        "pools_built": list(POOL_NAMES),
        "pool_counts": pool_counts,
        "pool_retained_after_masking": pool_retained,
        "flank_nt": cfg.flank_nt,
        "seed": cfg.seed,
        "residual_contamination": residual,
        "decoys_digest": digest,
        "corpus_sha256": provenance.sha256_file(corpus_parquet),
        "union_prior_sha256": provenance.sha256_file(union_parquet),
        "rfam_families": RFAM_FAMILIES,
        "notes": (
            "P0 realisation: gc_background is a seeded 0th-order composition null "
            "(no genome panel at P0); coordinate-masking is a no-op on these "
            "coordinate-free P0 pools and becomes load-bearing at P2 mined windows; "
            "residual = un-maskable union records with no coordinates."
        ),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    provenance.write_provenance(
        provenance_path,
        rule="workflow/rules/data.smk :: build_decoys",
        script="src/tbox_finder/decoys.py",
        seed=cfg.seed,
        inputs=[corpus_parquet, union_parquet, structured_refs, leader_refs],
        outputs=[out_parquet, report_path],
        env_lock=env_lock,
        adr="ADR-0005",
        extra={
            "decoys_digest": digest,
            "n_records": len(df),
            "pool_counts": pool_counts,
            "residual_contamination_fraction": residual["residual_contamination_fraction"],
        },
    )
    print(
        f"built {len(df)} decoys across {len(POOL_NAMES)} pools "
        f"(masked {residual['total_pool_records_masked']}; residual "
        f"{residual['residual_contamination_fraction']:.5f}); digest {digest}",
        file=sys.stderr,
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI (manual subcommand dispatch; mirrors splits.py)
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m tbox_finder.decoys {fetch-refs|build} ...", file=sys.stderr)
        return 2
    sub, rest = argv[0], argv[1:]

    if sub == "fetch-refs":
        p = argparse.ArgumentParser(prog="tbox_finder.decoys fetch-refs")
        p.add_argument("--refs-dir", default=_REFS_DIR)
        p.add_argument("--config", default="conf/data/decoys.yaml")
        p.add_argument("--tboxevo-negatives", default=TBOXEVO_NEGATIVES)
        a = p.parse_args(rest)
        return fetch_refs(
            refs_dir=a.refs_dir, config=a.config, tboxevo_negatives=a.tboxevo_negatives
        )

    if sub == "build":
        p = argparse.ArgumentParser(prog="tbox_finder.decoys build")
        p.add_argument("--corpus", default="data/processed/master_clean_v0.parquet")
        p.add_argument("--union-prior", default="data/processed/priors/union_prior.parquet")
        p.add_argument("--structured-refs", default=STRUCTURED_REFS_FA)
        p.add_argument("--leader-refs", default=LEADER_REFS_FA)
        p.add_argument("--out", default=DECOYS_PARQUET)
        p.add_argument("--provenance", default=DECOYS_PROVENANCE)
        p.add_argument("--report", default=DECOYS_REPORT)
        p.add_argument("--config", default="conf/data/decoys.yaml")
        p.add_argument("--env-lock", default=None)
        a = p.parse_args(rest)
        return build(
            corpus_parquet=a.corpus,
            union_parquet=a.union_prior,
            structured_refs=a.structured_refs,
            leader_refs=a.leader_refs,
            out_parquet=a.out,
            provenance_path=a.provenance,
            report_path=a.report,
            config=a.config,
            env_lock=a.env_lock,
        )

    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
