"""Source the independent non-Firmicutes GATE-1 anchor (arm c) — P0-16.

The headline generalization claim (GATE-1) needs a **model-independent** anchor of
*beyond-Firmicutes* T-box positives whose selection is independent of the RF00230
covariance model the model must beat, and whose sequences are independent of the
TBDB training corpus (PRD §7.1, §9.2(c), §5, §2.3 arm (c)). This module builds that
anchor by **re-deriving each locus from its primary genome**, using Vitreschak et
al. 2008's supplementary alignment (Fig S1) as the CM-free ground truth.

Provenance & independence (both load-bearing, both disclosed):

* **Detection independence.** Vitreschak 2008 (RNA 14(4):717-735, DOI:10.1261/
  rna.819308, PMID:18359782) detected T-boxes with a **hand-built sequence+structure
  descriptor + comparative-genomic synteny** — *not* the Rfam RF00230 CM at its GA
  cutoff (the GATE-1 ``cmsearch`` baseline). So its calls can reach loci below the
  RF00230 gathering cutoff (invisible to the GATE-1 baseline).
* **Ascertainment ceiling (disclosed).** Both the RF00230 CM *and* Vitreschak's
  descriptor are anchored on the same conserved T-box consensus, so arm (c) reaches
  the RF00230-cutoff-invisible-*but-consensus-detectable* region — **not** the
  all-consensus-invisible region (which only the synthetic-divergence arm (b) and the
  §13 discovery campaign reach). Same bound the PRD discloses for the CM-derived arms.
* **Corpus independence.** The anchor sequence is extracted from the **primary NCBI
  genome**, never from the TBDB/RF00230-derived training table. Because TBDB's own
  scan also covers these hosts, each locus is leakage-classified against the corpus
  so P0-24 can hold overlapping loci out of training (definitive cluster-level leakage
  is enforced there over the split table).

Re-derivation method (CM-free, per locus): parse the gap-elided Fig S1 alignment row
into contiguous genomic segments split by ``(N)`` omitted-length placeholders, localize
those segments in the host's primary genome (the literature sequence *is* the query —
no CM involved), verify the inter-segment gaps match the elided lengths, and extract the
full contiguous leader (filling the elided middle from the genome).

Confirmed non-Firmicutes host clades (each ≥2 independent peer-reviewed sources;
verified 2026-07-09): δ-proteobacteria → **Desulfobacterota** [DOI:10.1261/rna.819308
+ DOI:10.1128/MMBR.00026-08]; Deinococcus-Thermus → **Deinococcota** [same]; Chloroflexi
→ **Chloroflexota** [+ DOI:10.1016/j.febslet.2009.11.056]. **Dictyoglomi is
single-source (Vitreschak only) → WITHHELD** from certification (CLAUDE.md §10.1),
reproducing the signed-off P0-14 decision.

Stdlib-only at import (no pandas / no Biopython) so the module bare-imports in the CI
test env; pandas is imported lazily inside the orchestrator for the corpus/leakage read,
mirroring ``priors``/``taxonomy``. NCBI E-utilities are called with stdlib ``urllib``
(fail-loud, rate-limited, cached), matching ``taxonomy``'s stdlib fetch discipline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from tbox_finder.provenance import SCHEMA_VERSION, git_sha, write_provenance

# --------------------------------------------------------------------------- #
# Pins & constants
# --------------------------------------------------------------------------- #

#: Vitreschak et al. 2008 primary source (arm (c) provenance root).
SOURCE_DOI = "10.1261/rna.819308"
SOURCE_PMID = "18359782"
SOURCE_PMCID = "PMC2271356"
SOURCE_CITATION = (
    "Vitreschak AG, Mironov AA, Lyubetsky VA, Gelfand MS. Comparative genomic "
    "analysis of T-box regulatory systems in bacteria. RNA 2008;14(4):717-735."
)
#: Open-access supplement (rnajournal.cshlp.org). CSHL publishes no checksum companion,
#: so these SHA-256s are our own (computed on fetch, fail-loud re-verified) — same policy
#: as the NCBI taxdump pin (P0-15). Copyrighted .doc files are re-fetched, never committed.
SUPPLEMENT_BASE = "https://rnajournal.cshlp.org/content/suppl/2008/03/21/14.4.717.DC1/"
SUPPLEMENT_FILES: dict[str, dict[str, object]] = {
    "Supp_Table_1.doc": {
        "sha256": "d94fdb9c3ff3911ca2f2393aef921c1d69e9590e57bd5f596e8fc65499b39a7f",
        "bytes": 652288,
        "role": "Table S1 — T-box regulon table (host abbreviation → organism crosswalk)",
    },
    "Supp_Fig1.doc": {
        "sha256": "91d3bab6dbf65b66abd2b974eee114fbe19e6672894edb175e80984426a7bacd",
        "bytes": 1475072,
        "role": "Figure S1 — T-box leader alignment (the CM-free ground-truth sequences)",
    },
    "Supp_Legends.doc": {
        "sha256": "909b2a05d5f8cc1f478047bfe8989e644107ace8c566a7f0b47b982fec26410a",
        "bytes": 25600,
        "role": "Legends for Table S1 + Figure S1",
    },
}
SUPPLEMENT_LICENSE = (
    "© 2008 RNA Society / Cold Spring Harbor Laboratory Press (Vitreschak et al.). "
    "Supplement re-fetched for re-derivation; not redistributed in-repo. Re-derived "
    "leaders come from NCBI RefSeq (public domain)."
)


@dataclass(frozen=True)
class Host:
    """A confirmed non-Firmicutes anchor host (≥2-source verified)."""

    abbr: str  # Vitreschak Fig S1 / Table S1 genome abbreviation
    organism: str
    ncbi_phylum: str
    gtdb_phylum: str  # GTDB R232 phylum (name-projection; genome-resolution deferred to P6)
    assembly: str  # RefSeq assembly accession (GCF_…) — resolved + verified 2026-07-09
    genome_status: str
    sources: tuple[str, ...]


#: Confirmed non-Firmicutes hosts (Dictyoglomi WITHHELD — see ``WITHHELD``). Assemblies
#: resolved via NCBI Assembly 2026-07-09; type-strain (classic) GCF pinned for R1.
HOSTS: dict[str, Host] = {
    "GSU": Host(
        "GSU",
        "Geobacter sulfurreducens PCA",
        "Deltaproteobacteria",
        "Desulfobacterota",
        "GCF_000007985.2",
        "Complete Genome",
        ("DOI:10.1261/rna.819308", "DOI:10.1128/MMBR.00026-08"),
    ),
    "DAC": Host(
        "DAC",
        "Desulfuromonas acetoxidans DSM 684",
        "Deltaproteobacteria",
        "Desulfobacterota",
        "GCF_000167355.1",
        "Scaffold (draft)",
        ("DOI:10.1261/rna.819308", "DOI:10.1128/MMBR.00026-08"),
    ),
    "DR": Host(
        "DR",
        "Deinococcus radiodurans R1",
        "Deinococcus-Thermus",
        "Deinococcota",
        "GCF_000008565.1",
        "Complete Genome",
        ("DOI:10.1261/rna.819308", "DOI:10.1128/MMBR.00026-08"),
    ),
    "DG": Host(
        "DG",
        "Deinococcus geothermalis DSM 11300",
        "Deinococcus-Thermus",
        "Deinococcota",
        "GCF_000196275.1",
        "Complete Genome",
        ("DOI:10.1261/rna.819308", "DOI:10.1128/MMBR.00026-08"),
    ),
    "CAU": Host(
        "CAU",
        "Chloroflexus aurantiacus J-10-fl",
        "Chloroflexi",
        "Chloroflexota",
        "GCF_000018865.1",
        "Complete Genome",
        (
            "DOI:10.1261/rna.819308",
            "DOI:10.1128/MMBR.00026-08",
            "DOI:10.1016/j.febslet.2009.11.056",
        ),
    ),
    "DEH": Host(
        "DEH",
        "Dehalococcoides mccartyi 195",
        "Chloroflexi",
        "Chloroflexota",
        "GCF_000011905.1",
        "Complete Genome",
        (
            "DOI:10.1261/rna.819308",
            "DOI:10.1128/MMBR.00026-08",
            "DOI:10.1016/j.febslet.2009.11.056",
        ),
    ),
    "DE": Host(
        "DE",
        "Dehalococcoides mccartyi CBDB1",
        "Chloroflexi",
        "Chloroflexota",
        "GCF_000009025.1",
        "Complete Genome",
        (
            "DOI:10.1261/rna.819308",
            "DOI:10.1128/MMBR.00026-08",
            "DOI:10.1016/j.febslet.2009.11.056",
        ),
    ),
    "TR": Host(
        "TR",
        "Thermomicrobium roseum DSM 5159",
        "Chloroflexi",
        "Chloroflexota",
        "GCF_000021685.1",
        "Complete Genome",
        (
            "DOI:10.1261/rna.819308",
            "DOI:10.1128/MMBR.00026-08",
            "DOI:10.1016/j.febslet.2009.11.056",
        ),
    ),
}

#: Single-source clades excluded from certification (CLAUDE.md §10.1; reproduces the
#: signed-off P0-14 decision — Dictyoglomota stays has-prior via TBDB corpus loci but is
#: never a certifying anchor clade).
WITHHELD: dict[str, dict[str, str]] = {
    "DTH": {
        "organism": "Dictyoglomus thermophilum",
        "ncbi_phylum": "Dictyoglomi",
        "gtdb_phylum": "Dictyoglomota",
        "reason": "single-source (only Vitreschak 2008; MMBR-2009, FEBS-2010, TBDB-2021 all omit)",
    },
}

#: Hosts named in Table S1 but with no Fig S1 alignment row → no CM-free sequence to
#: re-derive → reported, not sourced (honest; §10.3). Pelobacter (δ-proteo, leu operon),
#: Thermus thermophilus (glyS2).
NO_FIGS1_ROW = ("PCA", "TTH", "THE")

# NCBI E-utilities (stdlib urllib, matching taxonomy.py's fetch discipline).
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NCBI_TOOL = "tbox-finder"
NCBI_EMAIL = "bioedca@gmail.com"
USER_AGENT = "tbox-finder/P0-16"
_MIN_INTERVAL = 0.4  # ≤3 req/s without an API key
_MAX_RETRY = 5
_CHUNK = 1 << 20

# Localization tolerances. The prototype localized GSU_LEUA with 0 mismatches and exact
# elision offsets; we allow a small slack for minor genome-version / alignment drift.
_SEG_MIN_LEN = 12  # a segment shorter than this is not used as a unique anchor
# Per-segment divergence cap between Vitreschak's genome vintage and current RefSeq. The
# most-divergent below-cutoff loci (Dehalococcoides ileS) reach ~12-15%; specificity is
# not carried by this cap but by the multi-segment offset agreement (gap == elided length).
_MAX_SEG_MISMATCH_FRAC = 0.15
_OFFSET_TOL = 5

CORPUS_PARQUET = Path("data/processed/master_clean_v0.parquet")
ANCHOR_DIR = Path("data/external/gate1_anchor")
ANCHOR_FASTA = "gate1_anchor.fasta"
ANCHOR_PROVENANCE = "provenance.json"
ANCHOR_REPORT = "gate1_anchor_report.json"
CACHE_SUBDIR = ".cache"

# The sequence field carries `(N)` omitted-length placeholders, so the alignment
# column must admit digits + parens alongside the nucleotide/gap alphabet.
_ROW_RE = re.compile(r"^([A-Z]{2,4})_([A-Z0-9]+)\s+([ACGTUacgtu0-9().\- ]{20,})$")

INDEPENDENCE_STATEMENT = (
    "Selection is CM-independent: loci were detected by Vitreschak 2008's hand-built "
    "sequence+structure descriptor + synteny, not the Rfam RF00230 CM at its GA cutoff "
    "(the GATE-1 cmsearch baseline), so the anchor can include loci below that cutoff. "
    "Sequence is corpus-independent: each leader is re-derived from its primary NCBI "
    "RefSeq genome, not the TBDB training table. Ascertainment ceiling (disclosed): both "
    "the CM and Vitreschak's descriptor rest on the same conserved T-box consensus, so "
    "arm (c) reaches the RF00230-cutoff-invisible-but-consensus-detectable region, not "
    "the all-consensus-invisible region (reached only by arm (b) synthetic divergence + "
    "the §13 discovery campaign)."
)


# --------------------------------------------------------------------------- #
# Pure logic (no network / no pandas) — unit-testable
# --------------------------------------------------------------------------- #


def revcomp(seq: str) -> str:
    """Reverse complement of an ACGT string (DNA; U already folded to T)."""
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def parse_figs1_row(row: str) -> tuple[list[str], list[int]]:
    """Parse a gap-elided Fig S1 alignment row into ``(segments, elisions)``.

    The row is contiguous genomic segments (aligned, gap ``-``) split by ``(N)``
    omitted-length placeholders. Returns the ungapped uppercase DNA segments (U→T)
    and the integer elided lengths between consecutive segments. A row with *k*
    elisions yields *k+1* segments (some possibly empty when two elisions abut).
    """
    parts = re.split(r"\((\d+)\)", row)
    segments: list[str] = []
    elisions: list[int] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            cleaned = re.sub(r"[^ACGT]", "", part.upper().replace("U", "T"))
            segments.append(cleaned)
        else:
            elisions.append(int(part))
    return segments, elisions


def _segment_offsets(segments: list[str], elisions: list[int]) -> list[int]:
    """Cumulative start offset of each segment within the full (un-elided) leader."""
    offsets = [0]
    for i in range(len(segments) - 1):
        offsets.append(offsets[-1] + len(segments[i]) + elisions[i])
    return offsets


_SEED = 14  # seed length for tolerant seed-and-extend anchoring


def _seed_positions(anchor: str, g: str) -> set[int]:
    """Candidate start positions of ``anchor`` in ``g`` via exact ``_SEED``-mers.

    Seeds are drawn along the anchor at stride ``_SEED`` (plus a trailing seed) so a
    slightly-divergent segment still localizes as long as one seed is intact — the
    full-segment hamming check (and the multi-segment offset agreement) enforce
    specificity, so a lone spurious seed cannot survive.
    """
    starts: set[int] = set()
    # dense seeds (half-seed stride) so a divergent segment still exposes ≥1 intact seed
    offs = list(range(0, max(1, len(anchor) - _SEED + 1), max(1, _SEED // 2)))
    if len(anchor) >= _SEED:
        offs.append(len(anchor) - _SEED)
    for off in sorted(set(offs)):
        kmer = anchor[off : off + _SEED]
        frm = 0
        while True:
            hit = g.find(kmer, frm)
            if hit < 0:
                break
            starts.add(hit - off)  # implied anchor start
            frm = hit + 1
    return starts


def _match_at(seg: str, g: str, pos: int) -> tuple[bool, int]:
    """(within-tolerance, mismatch count) for ``seg`` aligned to ``g`` at ``pos``."""
    if pos < 0 or pos + len(seg) > len(g):
        return False, len(seg)
    mm = sum(1 for x, y in zip(seg, g[pos : pos + len(seg)], strict=False) if x != y)
    return mm <= max(1, int(len(seg) * _MAX_SEG_MISMATCH_FRAC)), mm


def localize_leader(genome: str, segments: list[str], elisions: list[int]) -> dict | None:
    """Localize the Fig S1 segments in ``genome`` and extract the full leader.

    Seed-and-extend anchors on each usable segment (≥ ``_SEG_MIN_LEN``, longest first)
    via exact seeds, then requires **every** usable segment to match at its
    elision-predicted offset (within ``_MAX_SEG_MISMATCH_FRAC`` per segment, ``_OFFSET_TOL``
    positional slack) — the multi-segment offset agreement (gap == elided length) is the
    specificity guard that lets tolerant matching localize divergent below-cutoff loci
    without false positives. Searches both strands. Returns ``{strand, start, end, leader,
    n_segments_matched, exact_offsets, mismatches}`` (0-based half-open ``[start, end)``;
    for ``strand == '-'`` coordinates are on the reverse strand and ``leader`` is already
    the sense (reverse-complemented) sequence) — or ``None``.
    """
    usable = [(i, s) for i, s in enumerate(segments) if len(s) >= _SEG_MIN_LEN]
    if not usable:
        return None
    offsets = _segment_offsets(segments, elisions)
    best: dict | None = None
    seen: set[tuple[str, int]] = set()
    for strand, g in (("+", genome), ("-", revcomp(genome))):
        # try longer anchors first (more specific seeds)
        for anchor_i, anchor_seg in sorted(usable, key=lambda t: -len(t[1])):
            for leader_start in _seed_positions(anchor_seg, g):
                leader_start -= offsets[anchor_i]
                if leader_start < 0 or (strand, leader_start) in seen:
                    continue
                seen.add((strand, leader_start))
                matched = 0
                mismatches = 0
                exact = True
                ok = True
                for j, seg in usable:
                    predicted = leader_start + offsets[j]
                    hit, mm = _match_at(seg, g, predicted)
                    if hit:
                        matched += 1
                        mismatches += mm
                        if mm:
                            exact = False
                        continue
                    placed = False
                    for d in range(1, _OFFSET_TOL + 1):
                        for p in (predicted - d, predicted + d):
                            h2, mm2 = _match_at(seg, g, p)
                            if h2:
                                matched += 1
                                mismatches += mm2
                                exact = False
                                placed = True
                                break
                        if placed:
                            break
                    if not placed:
                        ok = False
                        break
                if not ok or matched < len(usable):
                    continue
                leader_end = leader_start + offsets[-1] + len(segments[-1])
                if leader_end > len(g):
                    continue
                cand = {
                    "strand": strand,
                    "start": leader_start,
                    "end": leader_end,
                    "leader": g[leader_start:leader_end],
                    "n_segments_matched": matched,
                    "exact_offsets": exact,
                    "mismatches": mismatches,
                }
                if best is None or mismatches < best["mismatches"]:
                    best = cand
    return best


def classify_leakage(
    accessions: set[str],
    start: int,
    end: int,
    corpus_rows: list[dict],
) -> dict:
    """Classify a re-derived locus against the corpus (preliminary; P0-24 is definitive).

    ``accessions`` = the versionless RefSeq + GenBank accessions of the anchor locus's
    replicon (so corpus GenBank accessions match RefSeq re-derivations). Reports
    same-replicon coordinate overlap and same-organism presence. Returns
    ``{coord_overlap, overlap_records, host_in_corpus}``.
    """
    lo, hi = min(start, end), max(start, end)
    overlaps = []
    for r in corpus_rows:
        if r.get("accession") in accessions:
            rs, re_ = r.get("start"), r.get("end")
            if rs is None or re_ is None:
                overlaps.append({**r, "coord": "no-coords"})
            else:
                # corpus minus-strand records carry start > end — normalize both intervals
                clo, chi = min(rs, re_), max(rs, re_)
                if lo < chi and clo < hi:  # half-open interval overlap
                    overlaps.append({**r, "coord": "overlap"})
    return {
        "coord_overlap": any(o["coord"] == "overlap" for o in overlaps),
        "overlap_records": overlaps,
        "host_in_corpus": bool(corpus_rows),
    }


# --------------------------------------------------------------------------- #
# NCBI E-utilities (stdlib urllib; fail-loud, rate-limited, cached)
# --------------------------------------------------------------------------- #

_last_call = [0.0]


def _throttle() -> None:
    dt = time.monotonic() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.monotonic()


def _eutils(util: str, **params: str) -> bytes:
    """Call an E-utility, retrying transient 429/5xx; fail-loud otherwise (§10.3)."""
    params.setdefault("tool", NCBI_TOOL)
    params.setdefault("email", NCBI_EMAIL)
    url = f"{EUTILS}{util}.fcgi?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(_MAX_RETRY):
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < _MAX_RETRY - 1:
                time.sleep(2.0 + 2.0 * attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < _MAX_RETRY - 1:
                time.sleep(2.0 + 2.0 * attempt)
                continue
            raise
    raise RuntimeError(f"NCBI {util} exhausted retries: {url}")


def _assembly_uid(gcf: str) -> str:
    data = json.loads(_eutils("esearch", db="assembly", term=gcf, retmode="json"))
    ids = data["esearchresult"]["idlist"]
    if not ids:
        raise ValueError(f"no NCBI Assembly UID for {gcf}")
    return ids[0]


def _elink_nuccore(uid: str, linkname: str) -> list[str]:
    data = json.loads(
        _eutils("elink", dbfrom="assembly", db="nuccore", id=uid, linkname=linkname, retmode="json")
    )
    out: list[str] = []
    for ls in data.get("linksets", []):
        for db in ls.get("linksetdbs", []):
            if db.get("linkname") == linkname:
                out.extend(db.get("links", []))
    return out


def _nuccore_accessions(uids: list[str]) -> list[str]:
    if not uids:
        return []
    data = json.loads(_eutils("esummary", db="nuccore", id=",".join(uids), retmode="json"))
    res = data["result"]
    return [res[u]["accessionversion"] for u in res.get("uids", [])]


def _efetch_fasta(accession: str) -> str:
    raw = _eutils("efetch", db="nuccore", id=accession, rettype="fasta", retmode="text")
    text = raw.decode()
    lines = text.splitlines()
    return "".join(lines[1:]).upper().replace("U", "T")


@dataclass
class GenomeCache:
    """Per-host resolved replicons + fetched sequences (RefSeq + GenBank accessions)."""

    refseq_acc: list[str] = field(default_factory=list)
    genbank_acc: list[str] = field(default_factory=list)
    seqs: dict[str, str] = field(default_factory=dict)  # refseq acc.version → sequence


def _versionless(acc: str) -> str:
    return acc.rsplit(".", 1)[0] if "." in acc else acc


def resolve_and_fetch_genome(host: Host, cache_dir: Path, download: bool) -> GenomeCache:
    """Resolve the host assembly's RefSeq+GenBank replicons and fetch RefSeq FASTA.

    Cached under ``cache_dir`` (``<assembly>.json`` accession map + ``<acc>.fasta``);
    ``download=False`` requires the cache (offline/tests), fail-loud if absent.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{host.assembly}.acc.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
    elif not download:
        raise FileNotFoundError(f"offline: missing genome cache {meta_path}")
    else:
        uid = _assembly_uid(host.assembly)
        refseq = _nuccore_accessions(_elink_nuccore(uid, "assembly_nuccore_refseq"))
        genbank = _nuccore_accessions(_elink_nuccore(uid, "assembly_nuccore_insdc"))
        meta = {"refseq": refseq, "genbank": genbank}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    gc = GenomeCache(refseq_acc=meta["refseq"], genbank_acc=meta["genbank"])
    for acc in gc.refseq_acc:
        fa = cache_dir / f"{acc}.fasta"
        if fa.is_file():
            gc.seqs[acc] = fa.read_text().strip().upper().replace("U", "T")
        elif not download:
            raise FileNotFoundError(f"offline: missing genome fasta {fa}")
        else:
            seq = _efetch_fasta(acc)
            fa.write_text(seq + "\n")
            gc.seqs[acc] = seq
    return gc


# --------------------------------------------------------------------------- #
# Supplement fetch + Fig S1 extraction
# --------------------------------------------------------------------------- #


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_supplement(cache_dir: Path, download: bool) -> dict[str, str]:
    """Fetch (or reuse) the 3 supplement .doc; verify pinned SHA-256; return sha map.

    Fail-loud on checksum mismatch (§10.3). ``download=False`` requires a present,
    checksum-matching cache. Copyrighted .doc files stay in the gitignored cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name, spec in SUPPLEMENT_FILES.items():
        pinned = str(spec["sha256"])
        dest = cache_dir / name
        if dest.is_file() and _sha256_file(dest) == pinned:
            out[name] = pinned
            continue
        if not download:
            raise FileNotFoundError(f"offline: missing/!=pinned supplement {dest}")
        req = urllib.request.Request(SUPPLEMENT_BASE + name, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        got = _sha256_bytes(data)
        if got != pinned:
            raise ValueError(
                f"supplement checksum mismatch for {name}: got {got} != pinned {pinned}"
            )
        dest.write_bytes(data)
        out[name] = got
    return out


def extract_doc_text(path: Path) -> str:
    """Extract text from a legacy OLE2 ``.doc`` in BOTH encodings (subprocess ``strings``).

    The δ-proteobacteria + Deinococcus Fig S1 rows are stored UTF-16LE-only, so a
    single-byte pass silently drops exactly the non-Firmicutes hosts — we scan both
    ``-e s`` (single-byte) and ``-e l`` (16-bit little-endian) and concatenate.
    """
    text = []
    for enc in ("s", "l"):
        res = subprocess.run(
            ["strings", "-e", enc, str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        text.append(res.stdout)
    return "\n".join(text)


def parse_figs1(text: str) -> dict[str, dict[str, object]]:
    """Parse Fig S1 text → ``{ABBR_GENE: {abbr, gene, row, segments, elisions}}``."""
    loci: dict[str, dict[str, object]] = {}
    for line in text.splitlines():
        m = _ROW_RE.match(line.strip())
        if not m:
            continue
        abbr, gene, seq = m.group(1), m.group(2), m.group(3)
        key = f"{abbr}_{gene}"
        if key in loci:
            continue  # first occurrence wins (both-encoding concat can duplicate)
        segments, elisions = parse_figs1_row(seq)
        if sum(len(s) for s in segments) < _SEG_MIN_LEN:
            continue
        loci[key] = {
            "abbr": abbr,
            "gene": gene,
            "row": seq.strip(),
            "segments": segments,
            "elisions": elisions,
        }
    return loci


# --------------------------------------------------------------------------- #
# GTDB placement + corpus rows
# --------------------------------------------------------------------------- #


def place_gtdb(host: Host, gtdb_lineages: dict[str, str] | None) -> dict[str, object]:
    """Place a host in GTDB R232. Genome-resolution lineage if the assembly is present
    in the (optional) staged GTDB taxonomy; else phylum-level name projection (verified
    ≥2 sources; genome-resolution GTDB-Tk deferred to P6, matching P0-14)."""
    key = host.assembly
    alt = key.replace("GCF_", "GCA_")
    if gtdb_lineages:
        for k in (key, alt, f"RS_{key}", f"GB_{alt}"):
            if k in gtdb_lineages:
                return {
                    "method": "gtdb_genome",
                    "lineage": gtdb_lineages[k],
                    "phylum": _phylum_of(gtdb_lineages[k]),
                }
    return {"method": "name_projection", "lineage": None, "phylum": host.gtdb_phylum}


def _phylum_of(lineage: str) -> str | None:
    for tok in lineage.split(";"):
        tok = tok.strip()
        if tok.startswith("p__"):
            return tok[3:]
    return None


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class AnchorResult:
    anchor_fasta: Path
    provenance: Path
    report: Path
    audit: dict


def _load_corpus_rows(corpus_parquet: str | Path) -> dict[str, list[dict]]:
    """Corpus rows grouped by lowercase host-genus token, for leakage classification."""
    import pandas as pd  # lazy — keeps module import stdlib-only for the unit tests

    df = pd.read_parquet(
        corpus_parquet,
        columns=["accession_name", "locus_start", "locus_end", "GBSeq_organism", "phylum", "genus"],
    )
    by_genus: dict[str, list[dict]] = {}
    for row in df.itertuples(index=False):
        genus = str(getattr(row, "genus", "") or "").strip().lower()
        if not genus:
            continue

        def _int(v: object) -> int | None:
            return int(v) if v is not None and not pd.isna(v) else None

        by_genus.setdefault(genus, []).append(
            {
                "accession": _versionless(str(row.accession_name)),
                "start": _int(row.locus_start),
                "end": _int(row.locus_end),
                "organism": str(row.GBSeq_organism),
                "phylum": str(row.phylum),
            }
        )
    return by_genus


def source_anchor(
    *,
    anchor_dir: str | Path = ANCHOR_DIR,
    corpus_parquet: str | Path = CORPUS_PARQUET,
    gtdb_lineages: dict[str, str] | None = None,
    download: bool = True,
    figs1_text: str | None = None,
) -> AnchorResult:
    """Source + re-derive the independent non-Firmicutes GATE-1 anchor (arm c).

    Args:
        anchor_dir: output dir (``data/external/gate1_anchor``). The FASTA + provenance +
            report are written here; the supplement + genomes cache under ``.cache/``.
        corpus_parquet: the training corpus for leakage classification.
        gtdb_lineages: optional ``{assembly_accession: lineage}`` for genome-resolution
            GTDB placement; ``None`` → phylum-level name projection.
        download: ``False`` uses only the cache (offline/tests), fail-loud if incomplete.
        figs1_text: pre-extracted Fig S1 text (tests inject a fixture to skip ``strings``).
    """
    anchor_dir = Path(anchor_dir)
    cache_dir = anchor_dir / CACHE_SUBDIR
    anchor_dir.mkdir(parents=True, exist_ok=True)

    # An injected ``figs1_text`` (tests) supplies the sequence directly, so the
    # copyrighted .doc need not be present; the real run fetches + SHA-verifies it.
    if figs1_text is None:
        supplement_sha = fetch_supplement(cache_dir, download)
        figs1_text = extract_doc_text(cache_dir / "Supp_Fig1.doc")
    else:
        supplement_sha = {name: "injected(test)" for name in SUPPLEMENT_FILES}
    figs1 = parse_figs1(figs1_text)

    corpus_rows = _load_corpus_rows(corpus_parquet)

    loci_out: list[dict] = []
    unresolved: list[dict] = []
    fasta_records: list[tuple[str, str]] = []

    for abbr, host in HOSTS.items():
        host_loci = {k: v for k, v in figs1.items() if v["abbr"] == abbr}
        if not host_loci:
            unresolved.append(
                {"abbr": abbr, "organism": host.organism, "reason": "no Fig S1 row parsed for host"}
            )
            continue
        gc = resolve_and_fetch_genome(host, cache_dir, download)
        replicon_accs = {_versionless(a) for a in gc.refseq_acc + gc.genbank_acc}
        placement = place_gtdb(host, gtdb_lineages)
        genus = host.organism.split()[0].lower()
        cRows = corpus_rows.get(genus, [])
        for key, locus in sorted(host_loci.items()):
            hit = None
            hit_acc = None
            for acc, seq in gc.seqs.items():
                found = localize_leader(seq, locus["segments"], locus["elisions"])
                if found and (
                    hit is None or found["n_segments_matched"] > hit["n_segments_matched"]
                ):
                    hit, hit_acc = found, acc
            if hit is None:
                unresolved.append(
                    {
                        "locus": key,
                        "organism": host.organism,
                        "reason": (
                            "not gapless-localizable within tolerance — the literature "
                            "segments carry intra-segment indels vs the current RefSeq "
                            "genome, so a rigorous gapless leader boundary cannot be set "
                            "(withheld rather than force-fit; CLAUDE.md §10.3)"
                        ),
                    }
                )
                continue
            leak = classify_leakage(replicon_accs, hit["start"], hit["end"], cRows)
            n_elis = len(locus["elisions"])
            rec = {
                "locus": key,
                "abbr": abbr,
                "gene": locus["gene"],
                "organism": host.organism,
                "ncbi_phylum": host.ncbi_phylum,
                "gtdb_phylum": placement["phylum"],
                "gtdb_method": placement["method"],
                "assembly": host.assembly,
                "refseq_accession": hit_acc,
                "strand": hit["strand"],
                "start_0based": hit["start"],
                "end_0based_excl": hit["end"],
                "leader_length": len(hit["leader"]),
                "n_segments": len(locus["segments"]),
                "n_segments_matched": hit["n_segments_matched"],
                "n_elisions": n_elis,
                "exact_offsets": hit["exact_offsets"],
                "segment_mismatches": hit["mismatches"],
                "match_quality": (
                    "exact" if hit["exact_offsets"] and hit["mismatches"] == 0 else "approximate"
                ),
                "corpus_coord_overlap": leak["coord_overlap"],
                "corpus_overlap_records": leak["overlap_records"],
                "host_in_corpus": leak["host_in_corpus"],
                "sources": list(host.sources),
            }
            loci_out.append(rec)
            header = (
                f"{key}|{hit_acc}:{hit['start']+1}-{hit['end']}({hit['strand']})"
                f"|{host.organism}|{placement['phylum']}|Vitreschak2008"
            )
            fasta_records.append((header, hit["leader"]))

    audit = _build_audit(loci_out, unresolved, supplement_sha, corpus_rows)

    # write FASTA
    fasta_path = anchor_dir / ANCHOR_FASTA
    with fasta_path.open("w") as fh:
        for header, seq in fasta_records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 70):
                fh.write(seq[i : i + 70] + "\n")

    report_path = anchor_dir / ANCHOR_REPORT
    report_path.write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    prov_path = write_provenance(
        anchor_dir / ANCHOR_PROVENANCE,
        rule="workflow/rules/data.smk :: source_gate1_anchor",
        script="src/tbox_finder/anchors.py",
        inputs=[str(corpus_parquet)],
        outputs=[str(fasta_path), str(report_path)],
        env_lock="envs/data.conda-lock.yml",
        adr="PRD §7.1/§9.2(c)/§2.3 (ADR-0005 pending, P0-19/25)",
        extra={
            "source": {
                "doi": SOURCE_DOI,
                "pmid": SOURCE_PMID,
                "pmcid": SOURCE_PMCID,
                "citation": SOURCE_CITATION,
                "supplement_base": SUPPLEMENT_BASE,
                "supplement_sha256": supplement_sha,
                "license": SUPPLEMENT_LICENSE,
            },
            "n_anchor_loci": len(loci_out),
            "independence": INDEPENDENCE_STATEMENT,
        },
    )
    return AnchorResult(fasta_path, prov_path, report_path, audit)


def _build_audit(
    loci_out: list[dict],
    unresolved: list[dict],
    supplement_sha: dict[str, str],
    corpus_rows: dict[str, list[dict]],
) -> dict:
    by_phylum: dict[str, int] = {}
    by_order_proxy: dict[str, int] = {}  # host organism as the order-proxy count key
    for r in loci_out:
        by_phylum[r["gtdb_phylum"]] = by_phylum.get(r["gtdb_phylum"], 0) + 1
        by_order_proxy[r["organism"]] = by_order_proxy.get(r["organism"], 0) + 1
    n_overlap = sum(1 for r in loci_out if r["corpus_coord_overlap"])
    return {
        "schema_version": SCHEMA_VERSION,
        "step": "P0-16",
        "git_sha": git_sha(),
        "prd": "§7.1 (arm c), §9.2(c), §5, §2.3 (GATE-1 arm (c))",
        "source": {
            "doi": SOURCE_DOI,
            "pmid": SOURCE_PMID,
            "pmcid": SOURCE_PMCID,
            "citation": SOURCE_CITATION,
            "supplement_sha256": supplement_sha,
        },
        "independence_statement": INDEPENDENCE_STATEMENT,
        "targeting": {
            "figs1_loci_targeted": len(loci_out) + len(unresolved),
            "re_derived": len(loci_out),
            "not_localized": len(unresolved),
            "note": (
                "targeted = confirmed non-Firmicutes Fig S1 rows for the included hosts; "
                "not_localized = indel-divergent loci withheld (see 'unresolved')."
            ),
        },
        "raw_record_count": len(loci_out),
        "counts_by_gtdb_phylum": by_phylum,
        "counts_by_host": by_order_proxy,
        "n_hosts": len({r["organism"] for r in loci_out}),
        "match_quality": {
            "exact": sum(1 for r in loci_out if r["match_quality"] == "exact"),
            "approximate": sum(1 for r in loci_out if r["match_quality"] == "approximate"),
        },
        "leakage_report": {
            "method": (
                "preliminary corpus-overlap footprint: per-locus same-replicon "
                "(RefSeq+GenBank accession) coordinate overlap + host-genus presence. "
                "Definitive cluster-level leakage (structure-aware, whole-cluster-to-fold) "
                "is enforced at P0-24 over the split table (PRD §9.2/§8.2)."
            ),
            "n_loci": len(loci_out),
            "n_corpus_coord_overlap": n_overlap,
            "n_novel_no_overlap": len(loci_out) - n_overlap,
            "note": (
                "TBDB's own scan covers these hosts (Vitreschak_master.fa fed TBDB), so "
                "overlap is expected; overlapping loci must be held out of training at "
                "P0-24 — their independent value is CM-free detection + primary-genome "
                "provenance, not novelty vs the corpus."
            ),
        },
        "withheld": {
            k: {**v, "policy": "excluded from GATE-1 certification (CLAUDE.md §10.1)"}
            for k, v in WITHHELD.items()
        },
        "not_re_derivable_no_figs1_row": list(NO_FIGS1_ROW),
        "unresolved": unresolved,
        "loci": loci_out,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m tbox_finder.anchors source-anchor --anchor-dir …``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "source-anchor":
        args = args[1:]
    parser = argparse.ArgumentParser(
        prog="tbox_finder.anchors source-anchor",
        description="Source the independent non-Firmicutes GATE-1 anchor (arm c) — P0-16.",
    )
    parser.add_argument("--anchor-dir", default=str(ANCHOR_DIR))
    parser.add_argument("--corpus", default=str(CORPUS_PARQUET))
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="use only the local cache (supplement + genomes); do not touch the network",
    )
    ns = parser.parse_args(args)
    out = source_anchor(
        anchor_dir=ns.anchor_dir,
        corpus_parquet=ns.corpus,
        download=not ns.no_download,
    )
    a = out.audit
    print(
        f"gate1 anchor: {a['raw_record_count']} non-Firmicutes loci "
        f"({a['n_hosts']} hosts) -> {out.anchor_fasta}\n"
        f"  by GTDB phylum: {a['counts_by_gtdb_phylum']}\n"
        f"  match: {a['match_quality']}; leakage coord-overlap "
        f"{a['leakage_report']['n_corpus_coord_overlap']}/{a['leakage_report']['n_loci']}\n"
        f"  report -> {out.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
