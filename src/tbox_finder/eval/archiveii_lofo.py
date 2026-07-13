"""ArchiveII nine-family LOFO benchmark + secondary-structure base-pair-F1 metric.

This module builds the **inter-family leave-one-family-out (LOFO / "fam-fold")**
ArchiveII benchmark and the base-pair-F1 metric used for the **RiNALMo mirror
parity gate** (PRD §10.2; ADR-0002 D5). P1-12 (this step) builds the benchmark,
the metric, and records the *published* RiNALMo parity target; **P1-13** runs the
``multimolecule/rinalmo-giga`` mirror against these exact splits and decides parity.

Provenance chain (verified 2026-07-13; CLAUDE.md §10.2)
------------------------------------------------------
* Raw ArchiveII was curated by the Mathews lab (U. Rochester); the standard
  citation is Sloma & Mathews 2016, *RNA* 22(12):1808-1818
  (DOI:10.1261/rna.053694.115; PMID:27852924).
* Szikszai et al. 2016->2022 built the **deduplicated inter-family CV splits**
  (*Bioinformatics* 38(16):3892-3899; DOI:10.1093/bioinformatics/btac415; repo
  ``github.com/marcellszi/dl-rna``), released as ``ct-splits.tar.gz``.
* **RiNALMo consumes those splits verbatim** — its ``remote_data.json``
  ``ARCHIVEII_SPLITS`` key points at exactly the URL pinned below. So evaluating
  the mirror on this archive is apples-to-apples with the published numbers.

The archive ships one **.ct (Connectivity Table)** file per RNA under
``ct/fam-fold/<family>/{train,valid,test}/``. For fold *F*, ``test/`` is family
*F* held out entirely and ``train/``+``valid/`` are the other eight families
(RNAs > 500 nt routed to ``valid``). RiNALMo ingests **sequence only** — one
token per nucleotide, no structure/covariance/MSA channel — so the harness reads
sequence + reference pairs and never feeds structure to a model.

Metric (matches RiNALMo ``rinalmo/utils/sec_struct.py``; adversarially verified)
--------------------------------------------------------------------------------
Base-pair precision / recall / F1 over predicted vs reference pairs, with a
**±1 nt slippage tolerance** (a reference pair ``(i, j)`` is also matched by
``(i±1, j)`` / ``(i, j±1)``; the prediction is relaxed before recall, the
reference before precision). The upstream default is **canonical-pairs-only**
(AU/UA/GC/CG/GU/UG) with a sharp-loop minimum pairing distance of 4, and it
removes pseudoknots from the *prediction* by a greedy non-crossing pass. Those
prediction-side choices belong to the model runner (P1-13); this module scores
whatever pair sets it is given and provides the slippage-tolerant metric plus
the canonical-pair and min-loop-distance helpers so the runner can reproduce the
exact protocol.

Everything here is **pure stdlib** so the golden + unit tiers run in bare CI.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tbox_finder import ingest, provenance

# --------------------------------------------------------------------------- #
# Pinned source (data/external/ is immutable + checksummed; CLAUDE.md §5.2)
# --------------------------------------------------------------------------- #
#: RiNALMo's exact ``ARCHIVEII_SPLITS`` asset (Szikszai et al. dl-rna release).
SPLITS_URL = "https://github.com/marcellszi/dl-rna/releases/download/Data/ct-splits.tar.gz"
#: SHA-256 of ``ct-splits.tar.gz`` (pinned; verified 2026-07-13). Fail-loud on drift.
SPLITS_SHA256 = "0ff7209016ba1775288572794e850d5857937c391d67c9ccacfe9ae77c0188e5"
#: Size in bytes of the pinned tarball (sanity cross-check).
SPLITS_BYTES = 33015660
#: Path inside the extracted tarball holding the inter-family (LOFO) splits.
FAM_FOLD_SUBPATH = "ct/fam-fold"

#: Directory name (in the archive) -> canonical family key (published naming).
FAMILY_DIRS: dict[str, str] = {
    "5s": "5S_rRNA",
    "srp": "SRP_RNA",
    "tRNA": "tRNA",
    "tmRNA": "tmRNA",
    "RNaseP": "RNaseP_RNA",
    "grp1": "group_I_intron",
    "16s": "16S_rRNA",
    "23s": "23S_rRNA",
    "telomerase": "telomerase_RNA",
}
#: Canonical family order (matches the published-target JSON + ADR-0002 D5).
FAMILY_ORDER: tuple[str, ...] = (
    "5S_rRNA",
    "SRP_RNA",
    "tRNA",
    "tmRNA",
    "RNaseP_RNA",
    "group_I_intron",
    "16S_rRNA",
    "23S_rRNA",
    "telomerase_RNA",
)
NUM_FAMILIES = len(FAMILY_ORDER)
#: The full (deduplicated) inter-family ArchiveII record count (Σ per-family
#: test sizes). Verified by extraction 2026-07-13; NOT quoted from the literature
#: (which disagrees on the small families) — see the dev-log.
EXPECTED_TOTAL_RECORDS = 3865
#: Canonical Watson-Crick + wobble pairs (the upstream ``allow_nc_pairs=False``).
CANONICAL_PAIRS = frozenset(
    {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
)
#: Minimum pairing distance |i-j| (upstream ``_SHARP_LOOP_DIST_THRESHOLD``).
SHARP_LOOP_MIN_DIST = 4

SCHEMA_VERSION = 1
ACCESSED_DATE = "2026-07-13"

_CITATIONS = {
    "archiveii": (
        "Sloma & Mathews 2016, RNA 22(12):1808-1818; " "DOI:10.1261/rna.053694.115; PMID:27852924"
    ),
    "interfamily_splits": (
        "Szikszai et al. 2022, Bioinformatics 38(16):3892-3899; "
        "DOI:10.1093/bioinformatics/btac415"
    ),
    "rinalmo": (
        "Penic et al. 2025, Nat Commun; "
        "DOI:10.1038/s41467-025-60872-5; PMID:40593636; PMC12219582"
    ),
}
_LICENSE_NOTE = (
    "No explicit data license on the Mathews-lab archiveII.tar.gz or the "
    "marcellszi/dl-rna release; publicly available for academic/research use. "
    "Cite Sloma & Mathews 2016 + Szikszai et al. 2022. (The HF "
    "multimolecule/archiveii mirror is a DIFFERENT 10-family AGPL curation and "
    "is NOT this 9-family benchmark.)"
)


# --------------------------------------------------------------------------- #
# .ct (Connectivity Table) parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CtRecord:
    """One parsed .ct record: sequence + reference base pairs (1-based, i<j).

    ``title`` is the raw .ct header title from :func:`parse_ct`; ``family`` is the
    canonical LOFO family key, filled by :func:`iter_test_records` (empty for a
    bare parse).
    """

    record_id: str
    title: str
    sequence: str
    pairs: tuple[tuple[int, int], ...]
    family: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)


def parse_ct(text: str, record_id: str = "") -> CtRecord:
    """Parse a single-structure .ct file. Fail-loud on any structural violation.

    Header: ``<length> <title...>``. Each residue line has >=6 whitespace fields
    ``i base (i-1) (i+1) j k``; a base pair is ``(min(i,j), max(i,j))`` when
    ``j > 0``. Asserts the residue count matches the header, indices are the
    contiguous ``1..L`` sequence, and pairings are symmetric.
    """
    raw = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not raw:
        raise ValueError(f"empty .ct file: {record_id!r}")
    header = raw[0].split(None, 1)
    try:
        declared_len = int(header[0])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{record_id!r}: bad .ct header {raw[0]!r}") from exc
    title = header[1].strip() if len(header) > 1 else ""

    body = raw[1:]
    if len(body) != declared_len:
        raise ValueError(
            f"{record_id!r}: header says {declared_len} residues but body has {len(body)}"
        )

    seq_chars: list[str] = []
    pair_map: dict[int, int] = {}
    for row, line in enumerate(body, start=1):
        cols = line.split()
        if len(cols) < 6:
            raise ValueError(f"{record_id!r}: malformed residue line {line!r}")
        i = int(cols[0])
        base = cols[1].upper()
        j = int(cols[4])
        if i != row:
            raise ValueError(f"{record_id!r}: non-contiguous index {i} at row {row}")
        seq_chars.append(base)
        if j > 0:
            pair_map[i] = j
    # symmetry: every i->j must have j->i (single-structure invariant)
    for i, j in pair_map.items():
        if pair_map.get(j) != i:
            raise ValueError(f"{record_id!r}: asymmetric pairing {i}->{j}")
    pairs = tuple(sorted((min(i, j), max(i, j)) for i, j in pair_map.items() if i < j))
    return CtRecord(
        record_id=record_id or title,
        title=title,
        sequence="".join(seq_chars),
        pairs=pairs,
    )


def parse_ct_file(path: str | Path) -> CtRecord:
    """Parse a .ct file on disk; ``record_id`` = the filename stem."""
    p = Path(path)
    return parse_ct(p.read_text(), record_id=p.stem)


def pairs_key(pairs: Iterable[tuple[int, int]]) -> str:
    """Canonical, lossless string for a pair set (pseudoknot-safe; sorted)."""
    return ";".join(f"{i}-{j}" for i, j in sorted(pairs))


# --------------------------------------------------------------------------- #
# Base-pair F1 metric (RiNALMo protocol: ±1 slippage)
# --------------------------------------------------------------------------- #
def _norm_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def relax_pairs(
    pairs: Iterable[tuple[int, int]], seq_len: int | None = None
) -> set[tuple[int, int]]:
    """±1-nt slippage neighborhood of a pair set (upstream ``_relax_ss``).

    Each pair ``(i, j)`` also admits ``(i±1, j)`` and ``(i, j±1)``. Indices are
    kept >= 1 (and <= ``seq_len`` when given); degenerate ``i == j`` shifts are
    dropped. Returns the canonicalized (i<j) neighborhood including the originals.
    """
    lo, hi = 1, seq_len if seq_len is not None else None
    out: set[tuple[int, int]] = set()
    for i, j in pairs:
        for di, dj in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if ni < lo or nj < lo or ni == nj:
                continue
            if hi is not None and (ni > hi or nj > hi):
                continue
            out.add(_norm_pair(ni, nj))
    return out


def base_pair_prf(
    pred_pairs: Iterable[tuple[int, int]],
    ref_pairs: Iterable[tuple[int, int]],
    *,
    slippage: bool = True,
    seq_len: int | None = None,
) -> tuple[float, float, float]:
    """Base-pair (precision, recall, F1) with optional ±1-nt slippage tolerance.

    Mirrors RiNALMo's ``sec_struct.py`` exactly, including its
    ``zero_division=0.0`` behavior: precision counts predicted pairs within the
    (relaxed) reference, recall counts reference pairs within the (relaxed)
    prediction, F1 = 2PR/(P+R). An **empty prediction or empty reference scores
    0.0** (never ``nan``) — RiNALMo uses sklearn ``precision_score``/
    ``recall_score`` with ``zero_division=0.0`` and returns F1 = 0.0 whenever
    ``P+R == 0``, so an empty-prediction record pulls the family mean toward 0.0
    rather than poisoning it. This equality with RiNALMo is load-bearing for the
    P1-13 parity gate (matching the published per-family means).
    """
    pred = {_norm_pair(i, j) for i, j in pred_pairs}
    ref = {_norm_pair(i, j) for i, j in ref_pairs}
    if slippage:
        ref_relaxed = relax_pairs(ref, seq_len)
        pred_relaxed = relax_pairs(pred, seq_len)
    else:
        ref_relaxed = ref
        pred_relaxed = pred
    tp_precision = sum(1 for p in pred if p in ref_relaxed)
    tp_recall = sum(1 for r in ref if r in pred_relaxed)
    # zero_division=0.0 (RiNALMo parity): a 0 denominator scores 0.0, not nan.
    precision = tp_precision / len(pred) if pred else 0.0
    recall = tp_recall / len(ref) if ref else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def canonical_pairs_only(
    sequence: str, pairs: Iterable[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Keep only canonical (AU/GC/GU) pairs at min-distance >= ``SHARP_LOOP_MIN_DIST``.

    Reproduces the upstream default (``allow_nc_pairs=False`` + sharp-loop cutoff)
    for a runner that needs to match the exact scored pair set. ``sequence`` is
    1-based via the pair indices.
    """
    seq = sequence.upper()
    kept: list[tuple[int, int]] = []
    for i, j in pairs:
        a, b = _norm_pair(i, j)
        if (b - a) < SHARP_LOOP_MIN_DIST:
            continue
        if (seq[a - 1], seq[b - 1]) in CANONICAL_PAIRS:
            kept.append((a, b))
    return tuple(sorted(kept))


# --------------------------------------------------------------------------- #
# LOFO benchmark construction
# --------------------------------------------------------------------------- #
@dataclass
class LofoManifest:
    """The parsed nine-family LOFO benchmark (digest + counts + provenance)."""

    lofo_digest: str
    n_records: int
    families: list[str]
    per_family_counts: dict[str, int]
    per_fold_counts: dict[str, dict[str, int]]
    records: list[CtRecord] = field(default_factory=list, repr=False)


def _family_of_dir(dirname: str) -> str:
    try:
        return FAMILY_DIRS[dirname]
    except KeyError as exc:
        raise ValueError(f"unknown ArchiveII family directory {dirname!r}") from exc


def iter_test_records(fam_fold_root: str | Path) -> list[CtRecord]:
    """Canonical records = each family's ``test/`` set (the held-out fold).

    In fam-fold, fold *F*'s ``test/`` is exactly family *F*, so the union of the
    nine ``test/`` dirs is the whole dataset with an unambiguous family label.
    Returns records sorted by ``(family_order, record_id)`` for a stable digest.
    """
    root = Path(fam_fold_root)
    records: list[CtRecord] = []
    for dirname, family in FAMILY_DIRS.items():
        test_dir = root / dirname / "test"
        if not test_dir.is_dir():
            continue
        for ct in sorted(test_dir.glob("*.ct")):
            rec = parse_ct_file(ct)
            records.append(CtRecord(rec.record_id, rec.title, rec.sequence, rec.pairs, family))
    family_rank = {f: n for n, f in enumerate(FAMILY_ORDER)}
    records.sort(key=lambda r: (family_rank.get(r.family, len(FAMILY_ORDER)), r.record_id))
    return records


def _count_ct(path: Path) -> int:
    return sum(1 for _ in path.glob("*.ct")) if path.is_dir() else 0


def lofo_fold_counts(fam_fold_root: str | Path) -> dict[str, dict[str, int]]:
    """Per-fold ``{train, valid, test}`` .ct counts (the exact split sizes)."""
    root = Path(fam_fold_root)
    out: dict[str, dict[str, int]] = {}
    for dirname, family in FAMILY_DIRS.items():
        fam_dir = root / dirname
        if not fam_dir.is_dir():
            continue
        out[family] = {role: _count_ct(fam_dir / role) for role in ("train", "valid", "test")}
    return out


def record_digest(records: Sequence[CtRecord]) -> str:
    """Golden digest over ``(record_id, family, sequence, pairs)`` per record.

    ``family`` is each record's LOFO test-fold, so the digest encodes both the
    split assignment and the parsed sequence + reference structure — the
    load-bearing benchmark content (pseudoknot-safe via :func:`pairs_key`).
    """
    per = [
        ingest.record_hash([r.record_id, r.family, r.sequence, pairs_key(r.pairs)]) for r in records
    ]
    return ingest.records_digest(per)


def build_lofo(fam_fold_root: str | Path) -> LofoManifest:
    """Parse the fam-fold tree into a validated :class:`LofoManifest`."""
    records = iter_test_records(fam_fold_root)
    per_family: dict[str, int] = {}
    for r in records:
        per_family[r.family] = per_family.get(r.family, 0) + 1
    manifest = LofoManifest(
        lofo_digest=record_digest(records),
        n_records=len(records),
        families=sorted(per_family, key=lambda f: FAMILY_ORDER.index(f)),
        per_family_counts=per_family,
        per_fold_counts=lofo_fold_counts(fam_fold_root),
        records=records,
    )
    return manifest


def validate_manifest(manifest: LofoManifest, *, require_full: bool = False) -> None:
    """Fail-closed structural checks (CLAUDE.md §10.3). ``require_full`` enforces
    the production dataset's nine-family / 3865-record shape."""
    if len(manifest.lofo_digest) != 64:
        raise ValueError("lofo_digest is not a 64-char sha256 hexdigest")
    if manifest.n_records != sum(manifest.per_family_counts.values()):
        raise ValueError("n_records disagrees with per-family counts")
    for fam in manifest.families:
        if fam not in FAMILY_ORDER:
            raise ValueError(f"unknown family {fam!r}")
    if require_full:
        if set(manifest.families) != set(FAMILY_ORDER):
            raise ValueError(f"expected {NUM_FAMILIES} families, got {sorted(manifest.families)}")
        if manifest.n_records != EXPECTED_TOTAL_RECORDS:
            raise ValueError(f"expected {EXPECTED_TOTAL_RECORDS} records, got {manifest.n_records}")
        # every fold: test == that family; train+valid == the rest
        for fam, counts in manifest.per_fold_counts.items():
            rest = EXPECTED_TOTAL_RECORDS - manifest.per_family_counts[fam]
            if counts["test"] != manifest.per_family_counts[fam]:
                raise ValueError(f"fold {fam}: test size != family size")
            if counts["train"] + counts["valid"] != rest:
                raise ValueError(f"fold {fam}: train+valid != held-out complement")


# --------------------------------------------------------------------------- #
# Staging into data/external/ (checksummed download; the rule entry)
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> None:  # pragma: no cover - network
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def stage_archiveii(
    dest_dir: str | Path,
    *,
    tarball: str | Path | None = None,
    write_folds: bool = True,
) -> Path:
    """Download (or reuse) + checksum-verify + extract + build the LOFO benchmark.

    Verifies the pinned SHA-256 fail-loud, extracts under a temp dir, parses the
    fam-fold tree, and writes ``provenance.json`` (+ optional ``folds.json`` with
    per-fold record-id lists) into ``dest_dir``. Returns the provenance path.
    ``data/external/`` is gitignored except ``provenance.json`` (CLAUDE.md §5.2).
    """
    dst = Path(dest_dir)
    dst.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tar_path = Path(tarball) if tarball else tmp_path / "ct-splits.tar.gz"
        if tarball is None:
            _download(SPLITS_URL, tar_path)
        size = tar_path.stat().st_size
        if size != SPLITS_BYTES:
            raise ValueError(f"ct-splits.tar.gz size {size} != pinned {SPLITS_BYTES}")
        got = provenance.sha256_file(tar_path)
        if got != SPLITS_SHA256:
            raise ValueError(f"ct-splits.tar.gz checksum mismatch: {got} != pinned {SPLITS_SHA256}")
        extract_dir = tmp_path / "extract"
        with tarfile.open(tar_path, "r:gz") as tf:
            _safe_extract(tf, extract_dir)
        fam_fold = extract_dir / FAM_FOLD_SUBPATH
        if not fam_fold.is_dir():
            raise FileNotFoundError(f"{FAM_FOLD_SUBPATH} missing in tarball")
        manifest = build_lofo(fam_fold)
        validate_manifest(manifest, require_full=True)
        if write_folds:
            folds = _fold_record_ids(fam_fold)
            (dst / "folds.json").write_text(
                json.dumps(folds, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            )
        prov_path = dst / "provenance.json"
        prov_path.write_text(
            json.dumps(
                _provenance(manifest, tar_path),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
    return prov_path


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract with a path-traversal guard (no member escapes ``dest``)."""
    dest = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise ValueError(f"unsafe tar member path: {member.name!r}")
    try:
        tf.extractall(dest, filter="data")  # py3.12+ safe extraction filter
    except TypeError:  # pragma: no cover - Python < 3.12 lacks filter=
        tf.extractall(dest)  # noqa: S202 - guarded above + member-path checked


def _fold_record_ids(fam_fold_root: str | Path) -> dict:
    root = Path(fam_fold_root)
    folds: dict[str, dict[str, list[str]]] = {}
    for dirname, family in FAMILY_DIRS.items():
        fam_dir = root / dirname
        if not fam_dir.is_dir():
            continue
        folds[family] = {
            role: sorted(p.stem for p in (fam_dir / role).glob("*.ct"))
            for role in ("train", "valid", "test")
        }
    return folds


def _provenance(manifest: LofoManifest, tarball: Path) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "P1-12: ArchiveII nine-family leave-one-family-out (inter-family) "
            "secondary-structure benchmark for the RiNALMo mirror parity gate "
            "(PRD §10.2; ADR-0002 D5). data/external/ is immutable + checksummed "
            "(CLAUDE.md §5.2); only this provenance.json is committed."
        ),
        "rule": "workflow/rules/backbones.smk :: archiveii_lofo_prep",
        "script": "src/tbox_finder/eval/archiveii_lofo.py",
        "prd": "§10.2",
        "adr": "ADR-0002 D5",
        "git_sha": provenance.git_sha(),
        "accessed_date": ACCESSED_DATE,
        "source": {
            "url": SPLITS_URL,
            "sha256": SPLITS_SHA256,
            "bytes": Path(tarball).stat().st_size,
            "citations": _CITATIONS,
            "license": _LICENSE_NOTE,
            "note": (
                "RiNALMo's remote_data.json ARCHIVEII_SPLITS points at this exact "
                "asset; the mirror is evaluated on these verbatim splits."
            ),
        },
        "lofo": {
            "kind": "inter-family (leave-one-family-out / fam-fold)",
            "input": "sequence-only (one token/nt; no structure/covariance/MSA channel)",
            "n_families": len(manifest.families),
            "n_records": manifest.n_records,
            "lofo_digest": manifest.lofo_digest,
            "per_family_counts": manifest.per_family_counts,
            "per_fold_counts": manifest.per_fold_counts,
            "counts_note": (
                "counts derived by extraction 2026-07-13, not quoted from the "
                "literature (which disagrees on the small families)."
            ),
        },
        "metric": {
            "name": "base-pair F1",
            "slippage_tolerance_nt": 1,
            "canonical_pairs_only_default": True,
            "sharp_loop_min_dist": SHARP_LOOP_MIN_DIST,
            "matches": "RiNALMo rinalmo/utils/sec_struct.py",
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage the ArchiveII nine-family LOFO benchmark (P1-12)."
    )
    parser.add_argument(
        "--dest-dir",
        default="data/external/archiveii_lofo",
        help="staging destination (data/external/, immutable/checksummed)",
    )
    parser.add_argument(
        "--tarball",
        default=None,
        help="optional pre-downloaded ct-splits.tar.gz (skips the network fetch)",
    )
    args = parser.parse_args(argv)
    out = stage_archiveii(args.dest_dir, tarball=args.tarball)
    print(f"staged ArchiveII LOFO benchmark -> {out.parent} (provenance: {out})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
