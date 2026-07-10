"""splits.py — P0-22 split construction (ADR-0004 D2/D3/D5; PRD §9.2/§12/§5).

Builds the leakage-controlled split ladder on which the project's headline
generalization claim (GATE-1) rests. Four Snakemake rules, each declaring its own
conda env (CLAUDE.md §3.2) — the ``plot-figures`` render is split off so the heavy
clustering runs in the data env (pyarrow, no matplotlib) and only the numeric→PNG
step needs the viz env (matplotlib, no pyarrow):

1. ``extract-sequences`` (data env) — read the corpus + re-placed lineages
   (P0-15) and the independent literature-anchor / blind positives (P0-16/17),
   split by class (Transcriptional→class I / Translational→class II), and write a
   per-class FASTA of the **full per-record T-box window** (``FASTA_sequence``)
   plus a manifest carrying every record's identity + lineage + source tag.
2. ``align`` (infernal env) — ``cmalign`` each per-class FASTA to its class CM
   (``RF00230.cm`` / ``TBDB001.cm``; the P0-11a staged models) so downstream
   distance is over **consensus (match-state) columns**, not raw identity
   (ADR-0004 D2; the premise is sequence-divergent-but-structurally-conserved
   homologs — DOI:10.1038/s41576-021-00434-9).
3. ``cluster-split`` (data env) — structure-aware single-linkage clustering at
   the pinned cut (identity ≥ 0.70 AND coverage ≥ 0.70 **of the RF00230 model
   consensus**, ``d = 1 − identity``; ADR-0004 D2 + Amendment A1 — so a sequence
   aligning to < 0.70 of the model is a forced singleton, never a hub bridge),
   whole clusters to a single fold (never split), the nested split
   ladder (random genus-stratified / leave-one-order-out + class/phylum stress /
   independent anchor), the cluster–clade-crossing forced rule + per-scheme
   phylogenetic-independence diagnostic (D3), the D5 nested most-restrictive
   training fold, and the D2 adequacy net numeric data (all-vs-all train↔test
   distance + tighter-cutoff re-cluster sensitivity sweep).
4. ``plot-figures`` (viz env) — render the D2 adequacy figures from that data.

Stdlib-only at import; numpy/pandas/matplotlib are imported lazily inside the
functions that need them so the pure-logic core (unit-tested in
``tests/unit/test_split_assignment.py``) imports in a bare env.

The single-linkage clustering is a **memory-bounded union-find over thresholded
CM-consensus-identity edges** — scipy.cluster.hierarchy would need the full
O(N²) condensed distance matrix (N ≈ 22k → infeasible), so we stream the
pairwise comparison in row-blocks (BLAS GEMM) and union on the fly, never
materialising the distance matrix or a global edge list.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tbox_finder import provenance

# --------------------------------------------------------------------------- #
# Pinned constants (ADR-0004 D2; PRD §9.2/§12). Any change to the D2 cut needs
# ADR-0004 re-sign-off (§2.3 carve-out). The rest live in conf/data/splits.yaml.
# --------------------------------------------------------------------------- #
CLASS_I = "I"
CLASS_II = "II"
CLASS_CM = {
    CLASS_I: "data/external/refs/RF00230.cm",
    CLASS_II: "data/external/refs/TBDB001.cm",
}
#: corpus ``type`` (P0-12) → class; Terminator present only in class I (PMID:25583497).
TYPE_TO_CLASS = {"Transcriptional": CLASS_I, "Translational": CLASS_II}

#: ADR-0004 D2 pinned structure-aware cut (blinded-frozen at P0).
IDENTITY_CUT = 0.70
#: Coverage floor measured against the RF00230 **model consensus** length (clen),
#: not the shorter member's span (ADR-0004 D2 Amendment A1, user sign-off
#: 2026-07-10): a sequence covering < 0.70 of the model forms no edge → singleton.
COVERAGE_CUT = 0.70
#: ADR-0004 D2 adequacy net (b): tighter-cutoff re-cluster sensitivity sweep.
SWEEP_IDENTITIES = (0.60, 0.70, 0.80, 0.90)

#: Scheme A (random, genus-stratified) fold record-count targets.
RANDOM_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
#: PRD §12 well-powered leave-one-order-out unit (≥ 20 held-out positives).
MIN_HELDOUT_POSITIVES = 20
#: Reserve the largest orders for the D5 nested training fold until their
#: cumulative coverage reaches this floor; the remaining ≥-min orders are the
#: designated leave-one-order-out holdouts (keeps the single checkpoint viable
#: despite the ~90%-Firmicutes / top-3-orders-81% corpus).
TRAIN_COVERAGE_FLOOR = 0.60
#: The one well-powered held-out phylum (ADR-0004 D5 (ii); PRD §9.2).
HOLDOUT_PHYLUM = "Actinobacteria"

DEFAULT_SEED = provenance.DEFAULT_SEED  # 42

#: Nucleotide → integer code for consensus-column comparison; gaps / degenerate
#: → -1 (missing), so an aligned position only counts when both members carry a
#: canonical base there.
_NT_CODE = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

# Default artefact locations (Snakemake overrides via CLI).
INPUTS_DIR = "data/interim/splits/inputs"
ALIGNED_DIR = "data/interim/splits/aligned"
SPLITS_DIR = "data/interim/splits"
AUDIT_DIR = "data/processed/audits"
FIGURES_DIR = "figures"
CORPUS_PARQUET = "data/processed/master_clean_v0.parquet"
LINEAGE_PARQUET = "data/interim/lineage_replaced.parquet"
ANCHOR_FASTA = "data/external/gate1_anchor/gate1_anchor.fasta"
BLIND_FASTA = "data/external/refs/heldout_blind_set.fasta"
CLASSII_FASTA = "data/external/classII_positives/classII_positives.fasta"


# ========================================================================== #
# Stdlib FASTA helpers
# ========================================================================== #
def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Return ``[(header, sequence), …]`` for ``path`` (empty file → ``[]``)."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            elif header is not None:
                chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def write_fasta(records: list[tuple[str, str]], path: str | Path) -> None:
    """Write ``[(name, seq), …]`` as an uppercase FASTA (one seq per 2 lines)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq.upper()}\n")


# ========================================================================== #
# Pure-logic split-ladder core (unit-tested without cmalign / numpy)
# ========================================================================== #
def single_linkage(n_nodes: int, edges) -> list[int]:
    """Connected components (single-linkage) over an undirected edge iterable.

    Reference implementation (path-compressed union-find) used for the unit
    tests and small inputs; the scale path uses the vectorised union-find in
    :func:`_cluster_consensus_matrix`, which this function pins as correct.
    Labels are normalised to ``0..K-1`` in order of first appearance.
    """
    parent = list(range(n_nodes))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:  # path compression
            parent[i], i = root, parent[i]
        return root

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[min(ri, rj)] = min(ri, rj)
            parent[max(ri, rj)] = min(ri, rj)
    return _normalise_labels([find(i) for i in range(n_nodes)])


def _normalise_labels(roots) -> list[int]:
    """Map arbitrary root ids to dense ``0..K-1`` in first-appearance order."""
    remap: dict[int, int] = {}
    out: list[int] = []
    for r in roots:
        r = int(r)
        if r not in remap:
            remap[r] = len(remap)
        out.append(remap[r])
    return out


def designate_heldout_orders(
    order_counts: dict[str, int],
    *,
    min_positives: int = MIN_HELDOUT_POSITIVES,
    train_coverage_floor: float = TRAIN_COVERAGE_FLOOR,
) -> tuple[list[str], list[str]]:
    """Split orders into (designated leave-one-order-out holdouts, train-reserved).

    ADR-0004 D5: the single production checkpoint cannot hold out *every*
    ≥-min order (the top-3 Firmicutes orders alone are ~81% of the corpus, so
    holding all out collapses training). We reserve the **largest** orders for
    training until their cumulative record coverage reaches ``train_coverage_floor``,
    and designate the remaining orders with ≥ ``min_positives`` positives as the
    leave-one-order-out holdouts (each is still well-powered for scoring). Orders
    with < ``min_positives`` are neither: too small to score, they stay available
    to training but are never a holdout unit. Deterministic (size, then name).
    """
    total = sum(order_counts.values())
    ordered = sorted(order_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    reserved: list[str] = []
    cumulative = 0
    for name, count in ordered:
        if total and cumulative / total >= train_coverage_floor:
            break
        reserved.append(name)
        cumulative += count
    reserved_set = set(reserved)
    heldout = sorted(
        name
        for name, count in order_counts.items()
        if count >= min_positives and name not in reserved_set
    )
    return heldout, sorted(reserved_set)


def assign_random_folds(
    cluster_records: dict[int, int],
    cluster_genus: dict[int, str],
    *,
    ratios: dict[str, float] = RANDOM_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[int, str]:
    """Assign whole clusters to train/val/test, genus-stratified (scheme A).

    ``cluster_records`` maps ``cluster_id → record count``; ``cluster_genus``
    maps ``cluster_id → majority genus`` (``""`` if none). Whole clusters go to a
    single fold (never split). Within each genus, clusters are shuffled with a
    genus-seeded RNG and greedily placed in the fold whose current record share
    is furthest below its target — so each genus is spread across folds and the
    global record ratios track ``ratios``. Deterministic in ``seed``.
    """
    import random

    fold_names = list(ratios)
    counts = {f: 0 for f in fold_names}
    by_genus: dict[str, list[int]] = defaultdict(list)
    for cid in sorted(cluster_records):
        by_genus[cluster_genus.get(cid, "")].append(cid)

    assignment: dict[int, str] = {}
    for genus in sorted(by_genus):
        clusters = by_genus[genus]
        rng = random.Random(f"{seed}:{genus}")
        rng.shuffle(clusters)
        for cid in clusters:
            total = sum(counts.values()) + cluster_records[cid]
            # deficit = target share − share this cluster would leave the fold at
            fold = max(
                fold_names,
                key=lambda f: ratios[f] - (counts[f] + cluster_records[cid]) / total,
            )
            assignment[cid] = fold
            counts[fold] += cluster_records[cid]
    return assignment


def cluster_taxon_spread(
    cluster_ids,
    taxa,
) -> dict[int, set[str]]:
    """Map ``cluster_id → set of distinct (non-empty) taxa`` in that cluster."""
    spread: dict[int, set[str]] = defaultdict(set)
    for cid, taxon in zip(cluster_ids, taxa, strict=True):
        if taxon:
            spread[int(cid)].add(taxon)
    return spread


# ========================================================================== #
# Stage 1 — extract-sequences (data env)
# ========================================================================== #
def _parse_external_lineage(source: str, header: str) -> dict[str, str]:
    """Parse phylum/order out of an anchor/blind FASTA header (best-effort)."""
    fields = [f.strip() for f in header.split("|")]
    out = {"phylum": "", "order": ""}
    if source == "anchor" and len(fields) >= 4:  # id|coord|organism|phylum|src
        out["phylum"] = fields[3]
    elif source == "blind" and len(fields) >= 3:  # acc|phylum|order|gene|hash
        out["phylum"] = fields[1]
        out["order"] = fields[2]
    return out


def extract_sequences(
    *,
    corpus_parquet: str | Path = CORPUS_PARQUET,
    lineage_parquet: str | Path = LINEAGE_PARQUET,
    anchor_fasta: str | Path = ANCHOR_FASTA,
    blind_fasta: str | Path = BLIND_FASTA,
    classii_fasta: str | Path = CLASSII_FASTA,
    inputs_dir: str | Path = INPUTS_DIR,
) -> dict:
    """Write per-class input FASTAs + a manifest for clustering (stage 1)."""
    import pandas as pd

    from tbox_finder import ingest

    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(corpus_parquet)
    lineage = pd.read_parquet(lineage_parquet)
    if len(corpus) != len(lineage):
        raise ValueError(f"corpus ({len(corpus)}) and lineage ({len(lineage)}) row counts differ")
    hashes = ingest.compute_record_hashes(corpus)
    if list(hashes) != list(lineage["record_sha256"]):
        raise ValueError("corpus record_sha256 does not row-align with lineage table")

    manifest_rows: list[dict] = []
    per_class: dict[str, list[tuple[str, str]]] = {CLASS_I: [], CLASS_II: []}
    type_series = corpus["type"].astype("object")
    seq_series = corpus["FASTA_sequence"].astype("object")
    for i, rid in enumerate(hashes):
        klass = TYPE_TO_CLASS.get(type_series.iloc[i], CLASS_I)  # 1 unknown → class I
        seq = seq_series.iloc[i]
        if not isinstance(seq, str) or not seq:
            raise ValueError(f"record {rid} has an empty FASTA_sequence")
        lin = lineage.iloc[i]
        per_class[klass].append((rid, seq))
        manifest_rows.append(
            {
                "seq_name": rid,
                "source": "corpus",
                "klass": klass,
                "record_sha256": rid,
                "resolved_phylum": lin["resolved_phylum"],
                "resolved_class": lin["resolved_class"],
                "resolved_order": lin["resolved_order"],
                "resolved_genus": lin["resolved_genus"],
                "dropped_from_clade_holdout": bool(lin["dropped_from_clade_holdout"]),
            }
        )

    # Independent positives (held out; clustered *with* the corpus so a held-out
    # anchor cannot sit next to a training twin — ADR-0004 D5/D7).
    for source, fasta, klass in (
        ("anchor", anchor_fasta, CLASS_I),  # Vitreschak-2008 non-Firmicutes leaders
        ("blind", blind_fasta, CLASS_II),  # 18 Actinobacteria/ILE class-II
        ("classII_extra", classii_fasta, CLASS_II),  # P0-17 (currently empty)
    ):
        if not Path(fasta).exists():
            continue
        for header, seq in read_fasta(fasta):
            if not seq:
                continue
            name = f"{source}:{header.split('|')[0].strip()}"
            lin = _parse_external_lineage(source, header)
            per_class[klass].append((name, seq))
            manifest_rows.append(
                {
                    "seq_name": name,
                    "source": source,
                    "klass": klass,
                    "record_sha256": "",
                    "resolved_phylum": lin["phylum"] or None,
                    "resolved_class": None,
                    "resolved_order": lin["order"] or None,
                    "resolved_genus": None,
                    "dropped_from_clade_holdout": False,
                }
            )

    for klass, recs in per_class.items():
        write_fasta(recs, inputs_dir / f"class_{klass}.fa")
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(inputs_dir / "manifest.parquet", index=False)
    summary = {
        "n_total": len(manifest),
        "n_class_I": int((manifest["klass"] == CLASS_I).sum()),
        "n_class_II": int((manifest["klass"] == CLASS_II).sum()),
        "n_by_source": manifest["source"].value_counts().to_dict(),
    }
    print(f"[splits.extract] {summary}")
    return summary


# ========================================================================== #
# Stage 2 — align (infernal env)
# ========================================================================== #
def run_cmalign(cm: str | Path, fasta: str | Path, out_sto: str | Path, *, cpu: int = 8) -> None:
    """``cmalign`` ``fasta`` → ``out_sto`` (Pfam) against ``cm`` (Infernal ≥ 1.1.4).

    ``--outformat pfam`` (single-block Stockholm with a ``#=GC RF`` consensus line;
    afa caps at 10,000 sequences, which class I exceeds), ``--notrunc`` (windows
    are complete T-box loci, not fragments) + ``--noprob`` (no posterior line
    needed for consensus-column comparison). Empty input → empty output.
    """
    out_sto = Path(out_sto)
    out_sto.parent.mkdir(parents=True, exist_ok=True)
    if Path(fasta).stat().st_size == 0:
        out_sto.write_text("")
        return
    cmd = [
        "cmalign",
        "--cpu",
        str(cpu),
        "--notrunc",
        "--noprob",
        "--outformat",
        "pfam",
        "-o",
        str(out_sto),
        str(cm),
        str(fasta),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def align(
    *, inputs_dir: str | Path = INPUTS_DIR, aligned_dir: str | Path = ALIGNED_DIR, cpu: int = 8
) -> dict:
    """cmalign each per-class FASTA to its class CM (stage 2)."""
    inputs_dir = Path(inputs_dir)
    aligned_dir = Path(aligned_dir)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for klass, cm in CLASS_CM.items():
        fasta = inputs_dir / f"class_{klass}.fa"
        sto = aligned_dir / f"class_{klass}.sto"
        run_cmalign(cm, fasta, sto, cpu=cpu)
        out[klass] = str(sto)
        print(f"[splits.align] class {klass}: {fasta} → {sto}")
    return out


# ========================================================================== #
# Stage 3 — cluster-split (data env)
# ========================================================================== #
def read_aligned(path: str | Path) -> list[tuple[str, str]]:
    """Read a cmalign alignment as ``[(name, aligned_seq), …]``.

    Handles both aligned FASTA (afa) and single-block Stockholm/Pfam
    (``--outformat pfam``; the size-uncapped format used for the >10k class I).
    ``#`` annotation lines (incl. ``#=GC RF``) and ``//`` are skipped; sequence
    lines are ``name<whitespace>aligned`` with the aligned string carrying no
    internal whitespace. Empty file → ``[]``.
    """
    text = Path(path).read_text()
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith(">"):
        return read_fasta(path)
    seqs: dict[str, list[str]] = {}
    order: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name, chunk = parts[0], parts[1].strip()
        if name not in seqs:
            seqs[name] = []
            order.append(name)
        seqs[name].append(chunk)
    return [(n, "".join(seqs[n])) for n in order]


def consensus_matrix(aligned_path: str | Path):
    """Parse a cmalign alignment into (names, consensus-column code matrix).

    Match (consensus) columns are those carrying no lowercase / '.' (Infernal
    marks insert states lowercase with '.' gaps, match states uppercase with '-'
    gaps). The returned int8 matrix is ``N × clen`` over {A:0,C:1,G:2,U/T:3},
    with -1 for a gap / degenerate base at a match column.
    """
    import numpy as np

    records = read_aligned(aligned_path)
    if not records:
        return [], np.zeros((0, 0), dtype=np.int8)
    names = [n for n, _ in records]
    width = len(records[0][1])
    if any(len(s) != width for _, s in records):
        raise ValueError(f"{aligned_path}: aligned sequences have unequal widths")
    raw = np.frombuffer("".join(s for _, s in records).encode("ascii"), dtype=np.uint8)
    raw = raw.reshape(len(records), width)
    lowercase = (raw >= ord("a")) & (raw <= ord("z"))
    dot = raw == ord(".")
    match_cols = ~((lowercase | dot).any(axis=0))
    sub = raw[:, match_cols]
    codes = np.full(sub.shape, -1, dtype=np.int8)
    for nt, val in _NT_CODE.items():
        codes[sub == ord(nt)] = val
    return names, codes


def _cluster_consensus_matrix(codes, sweep_identities, coverage_cut: float):
    """Single-linkage clusters at each sweep identity via blocked GEMM union-find.

    Returns ``{identity_cut: labels_array}``. Pairwise consensus-column identity
    and coverage are streamed in row-blocks (float32 GEMM) and unioned on the fly
    with a vectorised path-compressed union-find, so neither the O(N²) distance
    matrix nor a global edge list is ever materialised.
    """
    import numpy as np

    n, clen = codes.shape
    if n == 0:
        return {c: np.zeros(0, dtype=np.int64) for c in sweep_identities}
    present = (codes >= 0).astype(np.float32)  # n × clen
    onehot = [((codes == k).astype(np.float32)) for k in range(4)]
    parents = {c: np.arange(n, dtype=np.int64) for c in sweep_identities}

    def find(parent, idx):
        root = idx
        while True:
            nxt = parent[root]
            if np.array_equal(nxt, root):
                break
            root = nxt
        parent[idx] = root  # path compression
        return root

    def union(parent, a, b):
        while True:
            ra, rb = find(parent, a), find(parent, b)
            lo = np.minimum(ra, rb)
            hi = np.maximum(ra, rb)
            active = lo != hi
            if not active.any():
                break
            np.minimum.at(parent, hi[active], lo[active])

    block = 1024
    for start in range(0, n, block):
        stop = min(start + block, n)
        rows = slice(start, stop)
        co = present[rows] @ present.T  # co-occupied consensus columns
        match = np.zeros_like(co)
        for oh in onehot:
            match += oh[rows] @ oh.T
        with np.errstate(invalid="ignore", divide="ignore"):
            identity = np.where(co > 0, match / co, 0.0)
        # Coverage is the co-occupied fraction of the RF00230 **model** consensus
        # (clen), not of the shorter member's span — a sequence aligning to < the
        # cut of the model can form no edge (co ≤ its span) → forced singleton, the
        # ADR-0004 D2 "aligns-poorly-to-RF00230 → singleton" behaviour (Amendment A1;
        # the shorter-member denominator let 1-column hubs bridge the whole corpus).
        coverage = co / clen
        # upper triangle only (global index j > i)
        ii = np.arange(start, stop)[:, None]
        jj = np.arange(n)[None, :]
        upper = jj > ii
        cov_ok = (coverage >= coverage_cut) & upper
        for cut in sweep_identities:
            ri, ci = np.nonzero((identity >= cut) & cov_ok)
            if ri.size:
                union(parents[cut], ri + start, ci)
    labels = {}
    for cut, parent in parents.items():
        roots = find(parent, np.arange(n))
        labels[cut] = np.asarray(_normalise_labels(roots.tolist()), dtype=np.int64)
    return labels


def _nearest_cross_fold_identity(codes, is_train, is_test, coverage_cut: float):
    """Max consensus identity (coverage-adequate) from each test seq to any train seq.

    Used for the ADR-0004 D2 adequacy histogram: whole-cluster holdout implies no
    test↔train pair may be inside the cut (identity ≥ cut with coverage ≥ cut).
    """
    import numpy as np

    n, clen = codes.shape
    present = (codes >= 0).astype(np.float32)
    onehot = [((codes == k).astype(np.float32)) for k in range(4)]
    train_idx = np.nonzero(is_train)[0]
    best = np.full(n, -1.0, dtype=np.float32)
    if train_idx.size == 0:
        return best[is_test]
    p_train = present[train_idx]
    oh_train = [oh[train_idx] for oh in onehot]
    block = 1024
    test_idx = np.nonzero(is_test)[0]
    for start in range(0, test_idx.size, block):
        rows = test_idx[start : start + block]
        co = present[rows] @ p_train.T
        match = np.zeros_like(co)
        for oh, oht in zip(onehot, oh_train, strict=True):
            match += oh[rows] @ oht.T
        with np.errstate(invalid="ignore", divide="ignore"):
            identity = np.where(co > 0, match / co, 0.0)
        coverage = co / clen  # vs RF00230 model consensus (Amendment A1)
        identity = np.where(coverage >= coverage_cut, identity, -1.0)
        best[rows] = identity.max(axis=1)
    return best[is_test]


def _build_ladder(
    manifest, cluster_ids, *, seed: int, min_positives: int, train_coverage_floor: float
):
    """Assign the nested split ladder + clade-crossing rule (pure over a frame).

    Returns (assignments DataFrame, diagnostics dict). ``manifest`` carries one
    row per aligned sequence with source/lineage; ``cluster_ids`` is the
    production-cut (0.70) cluster label per row.
    """
    import numpy as np

    df = manifest.copy()
    df["cluster_id"] = np.asarray(cluster_ids, dtype=np.int64)
    is_corpus = (df["source"] == "corpus").to_numpy()
    is_external = ~is_corpus
    order = df["resolved_order"].fillna("").to_numpy()
    phylum = df["resolved_phylum"].fillna("").to_numpy()

    # Designate leave-one-order-out holdouts vs train-reserved (corpus only).
    order_counts = Counter(o for o, c in zip(order, is_corpus, strict=True) if c and o)
    heldout_orders, reserved_orders = designate_heldout_orders(
        dict(order_counts),
        min_positives=min_positives,
        train_coverage_floor=train_coverage_floor,
    )
    heldout_set = set(heldout_orders)

    # Per-cluster "held-out-linked" (D3 forced rule): any member in a designated
    # held-out order, the held-out phylum, or an external anchor/blind positive.
    member_heldout = np.array(
        [
            (o in heldout_set) or (p == HOLDOUT_PHYLUM) or ext
            for o, p, ext in zip(order, phylum, is_external, strict=True)
        ]
    )
    cluster_heldout_linked: dict[int, bool] = defaultdict(bool)
    for cid, flag in zip(df["cluster_id"].to_numpy(), member_heldout, strict=True):
        if flag:
            cluster_heldout_linked[int(cid)] = True
    linked = np.array([cluster_heldout_linked[int(c)] for c in df["cluster_id"]])

    # D5 nested most-restrictive training fold: corpus, has an order, cluster not
    # held-out-linked (so it crosses into no held-out order/phylum/anchor).
    has_order = np.array([bool(o) for o in order])
    nested_train = is_corpus & has_order & ~linked
    # Clade-crossing training-clade members: in a held-out-linked cluster but not
    # themselves held out — excluded from train AND not scored (D3).
    excluded_clade_crossing = is_corpus & has_order & linked & ~member_heldout
    df["nested_train"] = nested_train
    # "heldout" covers both corpus held-out positives (leave-one-order-out /
    # phylum) and the external anchor/blind positives (scheme C) so the D2(a)
    # adequacy histogram checks every scored held-out positive against the train
    # fold; "dropped" is the no-clade random-only residue (D4), never scored.
    df["nested_role"] = np.where(
        nested_train,
        "train",
        np.where(
            member_heldout,
            "heldout",
            np.where(excluded_clade_crossing, "excluded_clade_crossing", "dropped"),
        ),
    )

    # Scheme A — random genus-stratified over corpus clusters (whole clusters).
    corpus_df = df[is_corpus]
    cluster_records = Counter(int(c) for c in corpus_df["cluster_id"])
    cluster_genus: dict[int, str] = {}
    for cid, grp in corpus_df.groupby("cluster_id"):
        genera = [g for g in grp["resolved_genus"].fillna("") if g]
        cluster_genus[int(cid)] = Counter(genera).most_common(1)[0][0] if genera else ""
    fold_map = assign_random_folds(dict(cluster_records), cluster_genus, seed=seed)
    df["fold_random"] = [
        fold_map.get(int(c)) if corp else None
        for c, corp in zip(df["cluster_id"], is_corpus, strict=True)
    ]

    # Scheme B holdout units (the taxon at each holdout rank) + scheme C anchor.
    df["loo_order_unit"] = [
        o if (corp and o) else None for o, corp in zip(order, is_corpus, strict=True)
    ]
    df["class_holdout_unit"] = [
        c if corp else None
        for c, corp in zip(df["resolved_class"].fillna("").to_numpy(), is_corpus, strict=True)
    ]
    df["phylum_holdout_unit"] = [
        p if (corp and p) else None for p, corp in zip(phylum, is_corpus, strict=True)
    ]
    df["is_designated_loo_holdout"] = [o in heldout_set for o in order]
    # Scheme C held-out set is exactly the external anchor/blind positives; corpus
    # records that cluster with one are captured by clade_crossing_cluster (D3).
    df["is_anchor_heldout"] = is_external
    df["clade_crossing_cluster"] = linked & is_corpus
    df["parent_record_id"] = df["record_sha256"]  # base records; variants added P0-23

    diagnostics = _ladder_diagnostics(df, heldout_orders, reserved_orders, order, phylum, is_corpus)
    return df, diagnostics


def _ladder_diagnostics(df, heldout_orders, reserved_orders, order, phylum, is_corpus):
    """Per-scheme clade-crossing counts + spread + fold sizes (D3 diagnostic)."""
    import numpy as np

    order_spread = cluster_taxon_spread(df["cluster_id"], order)
    phylum_spread = cluster_taxon_spread(df["cluster_id"], phylum)
    crossing_order = {c: s for c, s in order_spread.items() if len(s) > 1}
    crossing_phylum = {c: s for c, s in phylum_spread.items() if len(s) > 1}
    n_records_in_crossing_order = int(
        np.isin(df["cluster_id"].to_numpy(), list(crossing_order)).sum()
    )
    role_counts = Counter(df["nested_role"])
    return {
        "n_sequences": int(len(df)),
        "n_corpus": int(is_corpus.sum()),
        "n_external": int((~is_corpus).sum()),
        "n_clusters": int(df["cluster_id"].nunique()),
        "largest_cluster": int(Counter(df["cluster_id"].tolist()).most_common(1)[0][1]),
        "n_designated_heldout_orders": len(heldout_orders),
        "designated_heldout_orders": heldout_orders,
        "n_train_reserved_orders": len(reserved_orders),
        "train_reserved_orders": reserved_orders,
        "nested_role_counts": dict(role_counts),
        "clade_crossing": {
            "order": {
                "n_crossing_clusters": len(crossing_order),
                "n_records_in_crossing_clusters": n_records_in_crossing_order,
                "max_orders_per_cluster": max((len(s) for s in order_spread.values()), default=0),
            },
            "phylum": {
                "n_crossing_clusters": len(crossing_phylum),
                "max_phyla_per_cluster": max((len(s) for s in phylum_spread.values()), default=0),
            },
        },
        "fold_random_counts": Counter(f for f in df["fold_random"].tolist() if f is not None),
    }


def _sweep_stability(manifest, labels_by_cut, *, min_positives, train_coverage_floor):
    """Per-cut split-structure stats for the ADR-0004 D2 sensitivity sweep.

    At P0 no trained model exists, so the sweep reports **split-structural**
    stability (cluster/holdout structure invariance to a tighter cut); the
    model leave-one-order-out headline-metric stability across the sweep is the
    P4 GATE-1 acceptance condition (ADR-0004 D2 adequacy net (b)). The held-out
    order designation reuses the same ``min_positives`` / ``train_coverage_floor``
    as the production build so ``n_nested_train`` is comparable across cuts.
    """
    import numpy as np

    is_corpus = (manifest["source"] == "corpus").to_numpy()
    order = manifest["resolved_order"].fillna("").to_numpy()
    # Held-out order set is cut-independent (order membership), so derive it once.
    heldout_orders, _ = designate_heldout_orders(
        dict(Counter(o for o, c in zip(order, is_corpus, strict=True) if c and o)),
        min_positives=min_positives,
        train_coverage_floor=train_coverage_floor,
    )
    heldout_set = set(heldout_orders)
    out = {}
    for cut, labels in sorted(labels_by_cut.items()):
        labels = np.asarray(labels)
        sizes = Counter(labels.tolist())
        order_spread = cluster_taxon_spread(labels, order)
        crossing = sum(1 for s in order_spread.values() if len(s) > 1)
        phy = manifest["resolved_phylum"].fillna("").to_numpy()
        member_heldout = np.array(
            [
                (o in heldout_set) or (p == HOLDOUT_PHYLUM) or (not c)
                for o, p, c in zip(order, phy, is_corpus, strict=True)
            ]
        )
        linked_clusters = {int(lab) for lab, m in zip(labels, member_heldout, strict=True) if m}
        linked = np.array([int(lab) in linked_clusters for lab in labels])
        has_order = np.array([bool(o) for o in order])
        n_train = int((is_corpus & has_order & ~linked).sum())
        out[f"{cut:.2f}"] = {
            "n_clusters": int(len(sizes)),
            "largest_cluster": int(max(sizes.values())),
            "n_clade_crossing_order_clusters": crossing,
            "n_nested_train": n_train,
        }
    return out


def _histogram_stats(nearest_identity, cut):
    """Histogram summary (numpy-only, no matplotlib) — the validation numbers.

    ``n_inside_cut`` must be 0: whole-cluster holdout means no held-out sequence
    may be coverage-adequately ≥ the cut from any nested-train sequence.
    """
    import numpy as np

    vals = np.asarray(nearest_identity, dtype=float)
    finite = vals[vals >= 0]
    return {
        "n_heldout": int(vals.size),
        "n_inside_cut": int((finite >= cut).sum()),
        "max_identity": float(finite.max()) if finite.size else None,
    }


def _plot_histogram(nearest_identity, cut, out_path):
    """All-vs-all train↔test consensus-identity histogram (ADR-0004 D2 (a))."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    vals = np.asarray(nearest_identity)
    finite = vals[vals >= 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    if finite.size:
        ax.hist(finite, bins=40, range=(0, 1), color="#3b6", edgecolor="white")
    ax.axvline(cut, color="#c33", ls="--", lw=1.5, label=f"cut = {cut:.2f}")
    n_inside = int((finite >= cut).sum())
    ax.set_xlabel("max coverage-adequate consensus identity to any nested-train seq")
    ax.set_ylabel("held-out sequences")
    ax.set_title(
        f"Train↔test structure-distance (held-out N={vals.size}; " f"inside cut={n_inside})"
    )
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {
        "n_heldout": int(vals.size),
        "n_inside_cut": n_inside,
        "max_identity": float(finite.max()) if finite.size else None,
    }


def _plot_sweep(sweep_stats, out_path):
    """Cluster/train structure vs identity cut (ADR-0004 D2 (b))."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cuts = sorted(sweep_stats, key=float)
    x = [float(c) for c in cuts]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(
        x, [sweep_stats[c]["n_clusters"] for c in cuts], "o-", color="#36b", label="n clusters"
    )
    ax1.plot(
        x,
        [sweep_stats[c]["n_clade_crossing_order_clusters"] for c in cuts],
        "s--",
        color="#b63",
        label="clade-crossing clusters",
    )
    ax1.set_xlabel("consensus-identity cut")
    ax1.set_ylabel("count")
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        [sweep_stats[c]["n_nested_train"] for c in cuts],
        "^-",
        color="#3a3",
        label="nested-train records",
    )
    ax2.set_ylabel("nested-train records")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8)
    ax1.set_title("Re-cluster sensitivity sweep (split-structure stability)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_figures(
    *,
    figure_data: str | Path = f"{SPLITS_DIR}/figure_data.json",
    figures_dir: str | Path = FIGURES_DIR,
) -> int:
    """Render the D2 adequacy figures from ``figure_data.json`` (viz env).

    Split out from ``cluster-split`` so the heavy clustering runs in the data env
    (pyarrow, no matplotlib) and only this thin numeric→PNG step needs the viz
    env (matplotlib, no pyarrow) — CLAUDE.md §3.2 rule = environment.
    """
    figures_dir = Path(figures_dir)
    data = json.loads(Path(figure_data).read_text())
    hist_path = figures_dir / "split_train_test_distance_histogram.png"
    sweep_path = figures_dir / "split_sensitivity_sweep.png"
    _plot_histogram(data["nearest_identity"], data["identity_cut"], hist_path)
    _plot_sweep(data["sweep_stats"], sweep_path)
    print(f"[splits.plot] {hist_path} + {sweep_path}")
    return 0


def cluster_split(
    *,
    inputs_dir: str | Path = INPUTS_DIR,
    aligned_dir: str | Path = ALIGNED_DIR,
    splits_dir: str | Path = SPLITS_DIR,
    audit_dir: str | Path = AUDIT_DIR,
    env_lock: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    min_positives: int = MIN_HELDOUT_POSITIVES,
    train_coverage_floor: float = TRAIN_COVERAGE_FLOOR,
) -> int:
    """Cluster, build the ladder, emit the split table + diagnostics + figure data."""
    import numpy as np
    import pandas as pd

    inputs_dir, aligned_dir = Path(inputs_dir), Path(aligned_dir)
    splits_dir, audit_dir = Path(splits_dir), Path(audit_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(inputs_dir / "manifest.parquet")

    # Cluster within each class (clusters never merge across classes), then
    # concatenate rows in manifest order with a global cluster_id offset.
    global_labels = np.full(len(manifest), -1, dtype=np.int64)
    sweep_labels_global = {c: np.full(len(manifest), -1, dtype=np.int64) for c in SWEEP_IDENTITIES}
    offset = 0
    name_to_row = {n: i for i, n in enumerate(manifest["seq_name"])}
    codes_by_class = {}
    rows_by_class = {}
    for klass in (CLASS_I, CLASS_II):
        names, codes = consensus_matrix(aligned_dir / f"class_{klass}.sto")
        if not names:
            continue
        rows = np.array([name_to_row[n] for n in names])
        codes_by_class[klass] = codes
        rows_by_class[klass] = rows
        labels_by_cut = _cluster_consensus_matrix(codes, SWEEP_IDENTITIES, COVERAGE_CUT)
        for cut, labels in labels_by_cut.items():
            sweep_labels_global[cut][rows] = labels + offset
        prod = labels_by_cut[IDENTITY_CUT]
        global_labels[rows] = prod + offset
        offset += int(prod.max()) + 1 if prod.size else 0

    if (global_labels < 0).any():
        raise ValueError("some manifest rows were never aligned/clustered")

    assignments, diagnostics = _build_ladder(
        manifest,
        global_labels,
        seed=seed,
        min_positives=min_positives,
        train_coverage_floor=train_coverage_floor,
    )

    # ADR-0004 D2 adequacy net (a): train↔test consensus-identity histogram,
    # computed per class then pooled (identity is undefined across CMs).
    role = assignments["nested_role"].to_numpy()
    nearest_all = []
    for klass, codes in codes_by_class.items():
        rows = rows_by_class[klass]
        is_train = role[rows] == "train"
        is_test = role[rows] == "heldout"
        if is_test.any():
            nearest_all.append(_nearest_cross_fold_identity(codes, is_train, is_test, COVERAGE_CUT))
    nearest = np.concatenate(nearest_all) if nearest_all else np.zeros(0)
    hist_stats = _histogram_stats(nearest, IDENTITY_CUT)

    # ADR-0004 D2 adequacy net (b): sensitivity sweep (split-structure stability).
    sweep_stats = _sweep_stability(
        manifest,
        sweep_labels_global,
        min_positives=min_positives,
        train_coverage_floor=train_coverage_floor,
    )

    # Numeric figure data → rendered to PNG by the viz-env ``plot-figures`` step
    # (data env has no matplotlib; the figures are not on the leakage critical path).
    figure_data_path = splits_dir / "figure_data.json"
    figure_data_path.write_text(
        json.dumps(
            {
                "nearest_identity": [float(x) for x in nearest],
                "identity_cut": IDENTITY_CUT,
                "sweep_stats": sweep_stats,
            }
        )
        + "\n"
    )

    # Persist the per-record split-assignment table (P0-23 commits the compact
    # git/LFS copy; here it is the DVC interim full table).
    out_cols = [
        "record_sha256",
        "seq_name",
        "source",
        "klass",
        "cluster_id",
        "resolved_phylum",
        "resolved_class",
        "resolved_order",
        "resolved_genus",
        "fold_random",
        "loo_order_unit",
        "class_holdout_unit",
        "phylum_holdout_unit",
        "nested_train",
        "nested_role",
        "is_designated_loo_holdout",
        "is_anchor_heldout",
        "clade_crossing_cluster",
        "dropped_from_clade_holdout",
        "parent_record_id",
    ]
    table = assignments[out_cols]
    out_parquet = splits_dir / "split_assignments.parquet"
    table.to_parquet(out_parquet, index=False)

    # Load-bearing invariant (§8.2): no cluster is split across nested roles that
    # would put a within-cluster pair on both sides of a holdout boundary.
    _assert_no_cluster_split(table)

    report = {
        "n_records": int(len(table)),
        "identity_cut": IDENTITY_CUT,
        "coverage_cut": COVERAGE_CUT,
        "sweep_identities": list(SWEEP_IDENTITIES),
        "seed": seed,
        "min_heldout_positives": min_positives,
        "train_coverage_floor": train_coverage_floor,
        "diagnostics": diagnostics,
        "sweep_stability": sweep_stats,
        "histogram": hist_stats,
        "no_cluster_split": True,
    }
    (audit_dir / "split_construction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    )

    prov_path = splits_dir / "split_assignments.provenance.json"
    provenance.write_provenance(
        prov_path,
        rule="workflow/rules/data.smk :: cluster_and_split",
        script="src/tbox_finder/splits.py",
        seed=seed,
        inputs=[
            inputs_dir / "manifest.parquet",
            aligned_dir / f"class_{CLASS_I}.sto",
            aligned_dir / f"class_{CLASS_II}.sto",
        ],
        outputs=[out_parquet, figure_data_path],
        env_lock=env_lock,
        adr="ADR-0004",
        extra={
            "identity_cut": IDENTITY_CUT,
            "coverage_cut": COVERAGE_CUT,
            "n_clusters": diagnostics["n_clusters"],
            "nested_role_counts": diagnostics["nested_role_counts"],
        },
    )
    print(
        f"[splits.cluster] {len(table)} records | clusters={diagnostics['n_clusters']} "
        f"| roles={diagnostics['nested_role_counts']} "
        f"| histogram inside-cut={hist_stats['n_inside_cut']}"
    )
    return 0


def _assert_no_cluster_split(table) -> None:
    """Fail loud if a cluster straddles the nested train/heldout boundary."""
    train = set(table.loc[table["nested_role"] == "train", "cluster_id"])
    heldout = set(table.loc[table["nested_role"] == "heldout", "cluster_id"])
    straddle = train & heldout
    if straddle:
        raise ValueError(
            f"{len(straddle)} clusters straddle nested train/heldout (leakage): "
            f"{sorted(straddle)[:5]}"
        )


def _json_default(obj):
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Counter):
        return dict(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


# ========================================================================== #
# CLI (manual subcommand dispatch, matching taxonomy.py / anchors.py)
# ========================================================================== #
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: tbox_finder.splits " "{extract-sequences|align|cluster-split|plot-figures} …",
            file=sys.stderr,
        )
        return 2
    sub, rest = argv[0], argv[1:]
    if sub == "extract-sequences":
        p = argparse.ArgumentParser(prog="tbox_finder.splits extract-sequences")
        p.add_argument("--corpus", default=CORPUS_PARQUET)
        p.add_argument("--lineage", default=LINEAGE_PARQUET)
        p.add_argument("--anchor", default=ANCHOR_FASTA)
        p.add_argument("--blind", default=BLIND_FASTA)
        p.add_argument("--classii", default=CLASSII_FASTA)
        p.add_argument("--inputs-dir", default=INPUTS_DIR)
        a = p.parse_args(rest)
        extract_sequences(
            corpus_parquet=a.corpus,
            lineage_parquet=a.lineage,
            anchor_fasta=a.anchor,
            blind_fasta=a.blind,
            classii_fasta=a.classii,
            inputs_dir=a.inputs_dir,
        )
        return 0
    if sub == "align":
        p = argparse.ArgumentParser(prog="tbox_finder.splits align")
        p.add_argument("--inputs-dir", default=INPUTS_DIR)
        p.add_argument("--aligned-dir", default=ALIGNED_DIR)
        p.add_argument("--cpu", type=int, default=8)
        a = p.parse_args(rest)
        align(inputs_dir=a.inputs_dir, aligned_dir=a.aligned_dir, cpu=a.cpu)
        return 0
    if sub == "cluster-split":
        p = argparse.ArgumentParser(prog="tbox_finder.splits cluster-split")
        p.add_argument("--inputs-dir", default=INPUTS_DIR)
        p.add_argument("--aligned-dir", default=ALIGNED_DIR)
        p.add_argument("--splits-dir", default=SPLITS_DIR)
        p.add_argument("--audit-dir", default=AUDIT_DIR)
        p.add_argument("--env-lock", default="envs/data.conda-lock.yml")
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--min-positives", type=int, default=MIN_HELDOUT_POSITIVES)
        p.add_argument("--train-coverage-floor", type=float, default=TRAIN_COVERAGE_FLOOR)
        a = p.parse_args(rest)
        return cluster_split(
            inputs_dir=a.inputs_dir,
            aligned_dir=a.aligned_dir,
            splits_dir=a.splits_dir,
            audit_dir=a.audit_dir,
            env_lock=a.env_lock,
            seed=a.seed,
            min_positives=a.min_positives,
            train_coverage_floor=a.train_coverage_floor,
        )
    if sub == "plot-figures":
        p = argparse.ArgumentParser(prog="tbox_finder.splits plot-figures")
        p.add_argument("--figure-data", default=f"{SPLITS_DIR}/figure_data.json")
        p.add_argument("--figures-dir", default=FIGURES_DIR)
        a = p.parse_args(rest)
        return plot_figures(figure_data=a.figure_data, figures_dir=a.figures_dir)
    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
