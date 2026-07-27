"""P2-10e-msa — the per-candidate homolog-search + CM-free de-novo MSA supply (ADR-0006 D7/A1).

This is the piece that turns one candidate sequence into the ``#=GC SS_cons`` Stockholm
alignment that :func:`tbox_finder.mining.covariation.covariation_spare_status` scores — the
**protective covariation-(a) backend** of the §9.1 mining spare rule. It is what the honest
round-readiness gate means by "MSA-*producible*" (ADR-0006 **A2**): ``backend_available`` answers
only "is R-scape installed"; the round-level gate additionally requires that a candidate
can actually be handed an MSA, which is exactly what this module builds.

ROUTE (ADR-0006 D7, anti-circularity §5 mechanism 5). Model-independent, never from RF00230 and
never from the model's own covariation:

  1. **search** the A1 target DB (2,500 GTDB R232 reps; ``data/interim/homolog_db/``) with the
     candidate as the query — ``blastn`` against the makeblastdb nucleotide DB, or ``nhmmer``
     against the makehmmerdb FM-index (D7's model-independent sequence-homology search).
  2. **assemble** the homolog set — the aligned subject subranges carved from the source genome
     FASTAs, one representative per subject contig, ``candidate`` + homologs; fewer than
     ``min_sequences`` homologs is **not producible** ⇒ ``None`` ⇒ ``covariation_spare_status``
     returns ``unavailable`` ⇒ the candidate is **spared, never silently mined** (fail-closed).
  3. **align** CM-free de-novo — a covariance model built *from the candidate*: ``RNAfold`` folds
     the candidate → ``#=GC SS_cons`` → ``cmbuild`` a CM from that single-sequence structure →
     ``cmalign --outformat Pfam`` aligns the homolog set to it. (D7's "a covariance model built
     from the candidates" alternative to LocARNA — chosen because LocARNA is in no pinned env,
     while ``cmbuild``/``cmalign``/``RNAfold`` are, so no env amendment is owed.)
  4. **score** — reuse :func:`covariation.covariation_spare_status` (pins
     ``TOTAL_ACROSS_HELICES`` + ``min_sequences=20``, ADR-0006 A2).

THRESHOLDS ARE **P6-FROZEN** (ADR-0006 **D17**). The homolog-search inclusion cutoffs
(``evalue`` / ``max_target_seqs`` / ``min_pident`` / ``min_cov``) are pinned *as a rule now, value
at the P6 phase gate*. So this module takes them as **keyword-required parameters with no default**
(the ADR-0005 A8 ``score_threshold`` precedent): the machinery is wired without any implementer
picking a number, and the certification supplies explicit *control* values that are demonstrably
not a pin. ``min_sequences`` is the one exception — it is *already pinned* (A2 Pin 2 =
``MIN_REAL_HOMOLOG_N`` = 20), so it defaults to that.

MULTI-ENV. The three stages run in three pinned envs (search: ``tbox-homology`` blast/hmmer;
align: ``tbox-infernal`` RNAfold/cmbuild/cmalign; score: ``tbox-rscape`` R-scape). A single Python
process cannot switch conda envs, so orchestration lives in the ``certify_homolog_msa.sbatch``,
and this module exposes one CLI subcommand per stage. Imports are stdlib + the pandas/torch-free
transport in :mod:`tbox_finder.mining.homolog_db` / :mod:`~.pilot_fetch`; ``covariation`` (and its
R-scape dependency) is imported **lazily** inside the score path only, so ``search``/``align`` never
drag it into an env that lacks R-scape.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

# Reuse the proven, pandas/torch-free transport (promote-don't-duplicate): the checked-subprocess
# runner + the PATH-or-raise tool resolver + the canonical <accession>:c<ci> subject scheme + the
# contig-aware genome-FASTA reader (homolog subranges are carved from the source genomes, not via
# blastdbcmd -entry — the A1 target DB was built without -parse_seqids, so -entry cannot resolve
# the string subject ids).
from tbox_finder.mining.homolog_db import (
    DB_DIR,
    GENOME_DIR,
    HomologDbError,
    _run,
    read_genome_fasta,
    tool_path,
)
from tbox_finder.mining.pilot_fetch import iter_fasta_records
from tbox_finder.power import MIN_REAL_HOMOLOG_N

# ── Schema / provenance ──────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"
STEP = "P2-10e-msa"
ADR = "ADR-0006"  # D7 (route) + A1 (target DB) + A2 (MSA-producibility gate); D17 (thresholds P6)

# ── Canonical artifact paths (one path per artifact) ─────────────────────────
BLAST_PREFIX = f"{DB_DIR}/target"  # makeblastdb -out prefix (target.n*)
FM_INDEX = f"{DB_DIR}/target.fm"  # makehmmerdb FM-index (nhmmer target)

# ── Pinned tool versions (assert where USED — the ADR-0002 A11 rule) ─────────
EXPECTED_INFERNAL_VERSION_PREFIX = "1.1.5"  # envs/infernal.yml / envs/rscape.yml co-pin
EXPECTED_VIENNARNA_VERSION_PREFIX = "2.7"  # envs/infernal.yml (RNAfold)

#: The blastn -outfmt 6 columns the producer requests + parses (request and parser cannot drift).
#: Adds sstart/send to the homolog_db self-recovery set — the aligned subject subrange whose
#: homolog subsequence is carved out of the source genome contig by coordinate.
BLAST_OUTFMT_COLUMNS = (
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "evalue",
    "bitscore",
    "sstart",
    "send",
)

#: The dot-bracket / WUSS alphabet a valid ``#=GC SS_cons`` line may contain.
_SS_PAIR_OPEN = "([{<"
_SS_PAIR_CLOSE = ")]}>"
_SS_UNPAIRED = ".,_-:~"
_SS_ALPHABET = frozenset(_SS_PAIR_OPEN + _SS_PAIR_CLOSE + _SS_UNPAIRED)

_ACGT = frozenset("ACGT")

#: A target subject id is ``<accession>:c<contig_index>`` (homolog_db.target_seq_name); accessions
#: (GCA_/GCF_...) carry no ``:c``, so the last ``:c<int>`` is unambiguously the contig separator.
_SUBJECT_RE = re.compile(r"^(?P<acc>.+):c(?P<ci>\d+)$")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class HomologMsaError(RuntimeError):
    """Raised when the per-candidate MSA cannot be *produced* for a reason that is a logic error
    rather than a fail-closed "not producible" (which returns ``None``): a malformed RNAfold
    structure, an unbalanced consensus structure, a certification whose must-fire control did not
    fire. Low-level tool/subprocess failures raise :class:`HomologDbError` (reused transport)."""


# ═════════════════════════════════════════════════════════════════════════════
# Sequence helpers (stdlib; safe in every env)
# ═════════════════════════════════════════════════════════════════════════════
def degap_to_dna(sequence: str) -> str:
    """Strip alignment gaps (``-`` and ``.``), uppercase, and map U→T (a DNA query for a DNA DB)."""
    return sequence.replace("-", "").replace(".", "").upper().replace("U", "T")


def read_single_sequence(fasta_source: str | Path) -> tuple[str, str]:
    """Return ``(header, sequence)`` of the first record of a FASTA file (path or text)."""
    text = (
        Path(fasta_source).read_text(encoding="utf-8")
        if _looks_like_path(fasta_source)
        else str(fasta_source)
    )
    records = iter_fasta_records(text)
    if not records:
        raise HomologMsaError(f"no FASTA record found in {fasta_source!r}")
    return records[0]


def read_stockholm_sequence(sto_source: str | Path, *, index: int = 0) -> tuple[str, str]:
    """Return ``(name, degapped-DNA-sequence)`` of the ``index``-th row of a Pfam/Stockholm MSA.

    Used to lift a real class-II T-box out of a committed corpus alignment as the certification
    seed. Reuses the promoted Pfam reader so the parse matches the fixture minter exactly.
    """
    from tbox_finder.mining.msa_shuffle import read_pfam_alignment

    rows, _gc = read_pfam_alignment(Path(sto_source))
    if not rows:
        raise HomologMsaError(f"no sequence rows in {sto_source!r}")
    if not (0 <= index < len(rows)):
        raise HomologMsaError(f"seed index {index} out of range (0..{len(rows) - 1})")
    name, aseq = rows[index]
    return name, degap_to_dna(aseq)


def write_fasta_record(name: str, sequence: str, out_path: str | Path, *, wrap: int = 80) -> Path:
    """Write a single-record FASTA (whitespace-free name, wrapped sequence)."""
    safe = _safe_seq_name(name)
    body = "\n".join(sequence[i : i + wrap] for i in range(0, len(sequence), wrap)) or sequence
    Path(out_path).write_text(f">{safe}\n{body}\n", encoding="utf-8")
    return Path(out_path)


def _safe_seq_name(name: str) -> str:
    """Whitespace-free sequence name (Infernal/blast subject ids must not contain spaces)."""
    token = name.split()[0] if name.split() else name
    return token.replace("|", "_")


def _looks_like_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    return "\n" not in source and not source.lstrip().startswith(">")


# ═════════════════════════════════════════════════════════════════════════════
# Search — argv builders (unit-testable) + parsers + runners
# ═════════════════════════════════════════════════════════════════════════════
def blastn_argv(
    query_fasta: str | Path,
    prefix: str | Path,
    *,
    evalue: float,
    max_target_seqs: int,
    columns: Sequence[str] = BLAST_OUTFMT_COLUMNS,
) -> list[str]:
    """argv for ``blastn`` against the makeblastdb nucleotide DB (cutoffs are the caller's)."""
    return [
        tool_path("blastn"),
        "-query",
        str(query_fasta),
        "-db",
        str(prefix),
        "-outfmt",
        "6 " + " ".join(columns),
        "-max_target_seqs",
        str(max_target_seqs),
        "-evalue",
        _fmt_evalue(evalue),
        "-dust",
        "no",
    ]


def nhmmer_argv(
    query_fasta: str | Path,
    fm_index: str | Path,
    tblout: str | Path,
    *,
    evalue: float,
) -> list[str]:
    """argv for ``nhmmer`` against the makehmmerdb FM-index (cutoff is the caller's)."""
    return [
        tool_path("nhmmer"),
        "--tblout",
        str(tblout),
        "--noali",
        "-E",
        _fmt_evalue(evalue),
        "-o",
        "/dev/null",
        str(query_fasta),
        str(fm_index),
    ]


def _fmt_evalue(evalue: float) -> str:
    if not isinstance(evalue, (int, float)) or isinstance(evalue, bool) or evalue <= 0:
        raise HomologMsaError(f"evalue must be a positive number, got {evalue!r}")
    return repr(float(evalue))


def parse_blast_hits(
    text: str, columns: Sequence[str] = BLAST_OUTFMT_COLUMNS
) -> list[dict[str, str]]:
    """Parse ``blastn -outfmt 6`` rows into dicts (comment/``#`` and blank lines skipped)."""
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < len(columns):
            raise HomologMsaError(
                f"blast row has {len(fields)} fields, need {len(columns)}: {line!r}"
            )
        rows.append(dict(zip(columns, fields, strict=False)))
    return rows


def parse_nhmmer_hits(text: str) -> list[dict[str, str]]:
    """Parse ``nhmmer --tblout`` into ``[{target, alifrom, alito, strand, evalue, score}]``.

    Fixed leading columns (0-indexed): target=0, alifrom=6, alito=7, strand=11, E-value=12,
    score=13. The trailing free-text description never shifts them, so positional indexing is safe.
    """
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 15:
            raise HomologMsaError(
                f"nhmmer tblout row has {len(fields)} fields (need >=15): {line!r}"
            )
        rows.append(
            {
                "target": fields[0],
                "alifrom": fields[6],
                "alito": fields[7],
                "strand": fields[11],
                "evalue": fields[12],
                "score": fields[13],
            }
        )
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Homolog-set assembly (pure logic — unit-testable)
# ═════════════════════════════════════════════════════════════════════════════
def blast_hit_to_range(hit: dict[str, str]) -> tuple[str, int, int, str]:
    """``(sseqid, lo, hi, strand)`` from a blastn hit; blastn reports send<sstart on minus."""
    sstart = int(hit["sstart"])
    send = int(hit["send"])
    strand = "plus" if sstart <= send else "minus"
    lo, hi = (sstart, send) if sstart <= send else (send, sstart)
    return hit["sseqid"], lo, hi, strand


def nhmmer_hit_to_range(hit: dict[str, str]) -> tuple[str, int, int, str]:
    """``(target, lo, hi, strand)`` from an nhmmer hit; nhmmer strand is ``+``/``-``."""
    a = int(hit["alifrom"])
    b = int(hit["alito"])
    lo, hi = (a, b) if a <= b else (b, a)
    strand = "plus" if hit["strand"] == "+" else "minus"
    return hit["target"], lo, hi, strand


def blast_coverage(hit: dict[str, str], query_len: int) -> float:
    """Aligned length / query length for a blastn hit (0 if query_len is 0)."""
    if query_len <= 0:
        return 0.0
    return int(hit["length"]) / query_len


def select_blast_homologs(
    hits: Sequence[dict[str, str]],
    *,
    query_len: int,
    min_pident: float,
    min_cov: float,
) -> list[tuple[str, int, int, str]]:
    """Filter blastn hits by pident/coverage, then keep the best (lowest-E) hit per subject.

    One representative per subject contig — distinct homologs, no paralog/near-duplicate
    inflation of the alignment depth. Deterministic order: subjects in first-seen order.
    """
    best: dict[str, tuple[float, tuple[str, int, int, str]]] = {}
    order: list[str] = []
    for hit in hits:
        if float(hit["pident"]) < min_pident:
            continue
        if blast_coverage(hit, query_len) < min_cov:
            continue
        sseqid, lo, hi, strand = blast_hit_to_range(hit)
        e = float(hit["evalue"])
        if sseqid not in best:
            order.append(sseqid)
            best[sseqid] = (e, (sseqid, lo, hi, strand))
        elif e < best[sseqid][0]:
            best[sseqid] = (e, (sseqid, lo, hi, strand))
    return [best[s][1] for s in order]


def select_nhmmer_homologs(
    hits: Sequence[dict[str, str]],
    *,
    query_len: int,
    min_cov: float,
) -> list[tuple[str, int, int, str]]:
    """Filter nhmmer hits by coverage, then keep the best (lowest-E) hit per subject."""
    best: dict[str, tuple[float, tuple[str, int, int, str]]] = {}
    order: list[str] = []
    for hit in hits:
        target, lo, hi, strand = nhmmer_hit_to_range(hit)
        cov = (hi - lo + 1) / query_len if query_len > 0 else 0.0
        if cov < min_cov:
            continue
        e = float(hit["evalue"])
        if target not in best:
            order.append(target)
            best[target] = (e, (target, lo, hi, strand))
        elif e < best[target][0]:
            best[target] = (e, (target, lo, hi, strand))
    return [best[s][1] for s in order]


def homolog_record_name(sseqid: str, lo: int, hi: int, strand: str) -> str:
    """A whitespace-free, unique homolog record name for the aligned FASTA."""
    return f"{_safe_seq_name(sseqid)}/{lo}-{hi}:{'p' if strand == 'plus' else 'm'}"


# ═════════════════════════════════════════════════════════════════════════════
# CM-free de-novo alignment (RNAfold → cmbuild → cmalign) — argv builders + parse
# ═════════════════════════════════════════════════════════════════════════════
def rnafold_argv() -> list[str]:
    """argv for ``RNAfold --noPS`` (candidate read from stdin; T→U auto-substituted)."""
    return [tool_path("RNAfold"), "--noPS"]


def cmbuild_argv(cm_out: str | Path, sto_in: str | Path, *, name: str) -> list[str]:
    """argv for ``cmbuild`` — a CM from the single-sequence candidate Stockholm (its SS_cons)."""
    return [tool_path("cmbuild"), "-F", "-n", _safe_seq_name(name), str(cm_out), str(sto_in)]


def cmalign_argv(
    cm: str | Path, seqfile: str | Path, out_sto: str | Path, *, cpu: int
) -> list[str]:
    """argv for ``cmalign`` — align the homolog set to the candidate CM, Pfam single-block out."""
    return [
        tool_path("cmalign"),
        "--outformat",
        "Pfam",
        "--noprob",
        "--cpu",
        str(cpu),
        "-o",
        str(out_sto),
        str(cm),
        str(seqfile),
    ]


def parse_rnafold_structure(text: str, *, expected_len: int) -> str:
    """Extract the dot-bracket structure line from ``RNAfold --noPS`` stdout.

    RNAfold prints ``>header`` / sequence / ``<structure> ( <energy>)``. The structure is the
    first whitespace-delimited token of the structure line — identified robustly as the first
    token that is all-in-:data:`_SS_ALPHABET` and exactly ``expected_len`` long (so the energy
    field, the echoed sequence, and any header cannot be mistaken for it).
    """
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        token = parts[0]
        if len(token) == expected_len and set(token) <= _SS_ALPHABET:
            return token
    raise HomologMsaError(
        f"could not parse a length-{expected_len} dot-bracket structure from RNAfold output"
    )


def assert_balanced_structure(structure: str) -> None:
    """Raise unless every pair symbol is nested-balanced (cmbuild needs a balanced SS_cons)."""
    close_to_open = dict(zip(_SS_PAIR_CLOSE, _SS_PAIR_OPEN, strict=True))
    pair_stack: list[str] = []
    for ch in structure:
        if ch in _SS_PAIR_OPEN:
            pair_stack.append(ch)
        elif ch in _SS_PAIR_CLOSE:
            want = close_to_open[ch]
            if not pair_stack or pair_stack[-1] != want:
                raise HomologMsaError(f"unbalanced SS_cons near {ch!r}: {structure!r}")
            pair_stack.pop()
        elif ch not in _SS_UNPAIRED:
            raise HomologMsaError(f"illegal SS_cons character {ch!r}: {structure!r}")
    if pair_stack:
        raise HomologMsaError(f"unclosed pairs in SS_cons: {structure!r}")


def build_candidate_stockholm(name: str, sequence: str, structure: str) -> str:
    """A single-sequence Pfam Stockholm (candidate row + ``#=GC SS_cons``) for ``cmbuild``."""
    if len(sequence) != len(structure):
        raise HomologMsaError(
            f"candidate seq ({len(sequence)}) and structure ({len(structure)}) length mismatch"
        )
    assert_balanced_structure(structure)
    tag = _safe_seq_name(name)
    width = max(len(tag), len("#=GC SS_cons"))
    return (
        "# STOCKHOLM 1.0\n\n"
        f"{tag.ljust(width)} {sequence}\n"
        f"{'#=GC SS_cons'.ljust(width)} {structure}\n"
        "//\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Stage runners (shell out; validated end-to-end by the SLURM certification)
# ═════════════════════════════════════════════════════════════════════════════
def run_blastn_search(
    query_fasta: str | Path,
    prefix: str | Path,
    *,
    evalue: float,
    max_target_seqs: int,
) -> list[dict[str, str]]:
    return parse_blast_hits(
        _run(
            blastn_argv(query_fasta, prefix, evalue=evalue, max_target_seqs=max_target_seqs)
        ).stdout
    )


def run_nhmmer_search(
    query_fasta: str | Path,
    fm_index: str | Path,
    tblout: str | Path,
    *,
    evalue: float,
) -> list[dict[str, str]]:
    _run(nhmmer_argv(query_fasta, fm_index, tblout, evalue=evalue))
    return parse_nhmmer_hits(Path(tblout).read_text(encoding="utf-8"))


def parse_subject_id(sseqid: str) -> tuple[str, int]:
    """``(accession, contig_index)`` from a ``<accession>:c<contig_index>`` target subject id."""
    m = _SUBJECT_RE.match(sseqid)
    if not m:
        raise HomologMsaError(f"subject id {sseqid!r} is not <accession>:c<contig_index>")
    return m.group("acc"), int(m.group("ci"))


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@lru_cache(maxsize=2048)
def _genome_records(genome_dir: str, accession: str) -> tuple[tuple[str, str], ...]:
    """The source genome's contigs, in the SAME order homolog_db used to name ``<acc>:c<ci>``."""
    return tuple(iter_fasta_records(read_genome_fasta(genome_dir, accession)))


def extract_homolog_sequence(
    genome_dir: str | Path, sseqid: str, lo: int, hi: int, strand: str
) -> str:
    """Carve a homolog subject subrange out of its source genome FASTA (uppercased, ungapped).

    ``<accession>:c<contig_index>`` maps to the genome's contig by the SAME ``iter_fasta_records``
    order homolog_db used to build the target names, so ``records[ci]`` is the hit's contig; ``lo``/
    ``hi`` are 1-based inclusive (blastn/nhmmer) and a minus-strand hit is reverse-complemented to
    the query orientation. Reading the genome (not ``blastdbcmd -entry``) sidesteps the
    ``-parse_seqids`` the A1 DB was not built with; the DB/FM-index names match the contig indices.
    """
    accession, ci = parse_subject_id(sseqid)
    records = _genome_records(str(genome_dir), accession)
    if not (0 <= ci < len(records)):
        raise HomologMsaError(
            f"contig index {ci} out of range for {accession} ({len(records)} records)"
        )
    seq = records[ci][1].upper()
    if not (1 <= lo <= hi <= len(seq)):
        raise HomologMsaError(f"range {lo}-{hi} out of bounds for {sseqid} (contig len {len(seq)})")
    sub = seq[lo - 1 : hi]
    return _revcomp(sub) if strand == "minus" else sub


def run_rnafold_structure(sequence: str) -> str:
    """Fold ``sequence`` with RNAfold and return its dot-bracket structure (same length)."""
    proc = subprocess.run(  # noqa: S603 (tool_path-resolved argv, fixed FASTA on stdin)
        rnafold_argv(),
        input=f">candidate\n{sequence}\n",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise HomologDbError(f"RNAfold failed ({proc.returncode}): {proc.stderr}")
    return parse_rnafold_structure(proc.stdout, expected_len=len(sequence))


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: search + assemble the homolog FASTA
# ═════════════════════════════════════════════════════════════════════════════
def search_homologs(
    *,
    candidate_fasta: str | Path,
    out_fasta: str | Path,
    engine: str,
    evalue: float,
    max_target_seqs: int,
    min_pident: float,
    min_cov: float,
    prefix: str | Path = BLAST_PREFIX,
    fm_index: str | Path = FM_INDEX,
    genome_dir: str | Path = GENOME_DIR,
    min_sequences: int = MIN_REAL_HOMOLOG_N,
    tblout: str | Path | None = None,
) -> dict[str, Any]:
    """Search the target DB with the candidate and write ``candidate`` + homologs to ``out_fasta``.

    Inclusion cutoffs (``evalue``/``max_target_seqs``/``min_pident``/``min_cov``) are the caller's
    (P6-frozen, D17 — no default). Returns a report dict; ``sufficient`` is ``len(records) >=
    min_sequences``. Writes the FASTA regardless so a marginal result can be inspected; the align
    stage only runs on a sufficient set (fail-closed at the round level = ``None`` ⇒ spared).
    """
    cand_name, cand_seq = read_single_sequence(candidate_fasta)
    cand_seq = degap_to_dna(cand_seq)
    if not cand_seq or set(cand_seq) - _ACGT - {"N"}:
        raise HomologMsaError(f"candidate {cand_name!r} is not a clean nucleotide sequence")
    qlen = len(cand_seq)

    if engine == "blastn":
        hits = run_blastn_search(
            candidate_fasta, prefix, evalue=evalue, max_target_seqs=max_target_seqs
        )
        selected = select_blast_homologs(
            hits, query_len=qlen, min_pident=min_pident, min_cov=min_cov
        )
    elif engine == "nhmmer":
        if tblout is None:
            raise HomologMsaError("engine='nhmmer' requires a --tblout path")
        hits = run_nhmmer_search(candidate_fasta, fm_index, tblout, evalue=evalue)
        selected = select_nhmmer_homologs(hits, query_len=qlen, min_cov=min_cov)
    else:
        raise HomologMsaError(f"engine must be 'blastn' or 'nhmmer', got {engine!r}")

    records: list[tuple[str, str]] = [("candidate", cand_seq)]
    for sseqid, lo, hi, strand in selected:
        # Extraction reads the source genome FASTA (works for both engines — the BLAST DB and the
        # nhmmer FM-index share the SAME <accession>:c<ci> names, which map to genome contigs by
        # index). This avoids blastdbcmd -entry, which the -parse_seqids-less A1 DB cannot resolve.
        seq = extract_homolog_sequence(genome_dir, sseqid, lo, hi, strand)
        if seq and set(seq) <= (_ACGT | {"N"}):
            records.append((homolog_record_name(sseqid, lo, hi, strand), seq))

    _write_multi_fasta(records, out_fasta)
    n_records = len(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "engine": engine,
        "candidate": _safe_seq_name(cand_name),
        "candidate_len": qlen,
        "n_hits": len(hits),
        "n_selected": len(selected),
        "n_records": n_records,
        "n_homologs": n_records - 1,
        "min_sequences": min_sequences,
        "sufficient": n_records >= min_sequences,
        "cutoffs": {
            "evalue": evalue,
            "max_target_seqs": max_target_seqs,
            "min_pident": min_pident,
            "min_cov": min_cov,
        },
        "out_fasta": str(out_fasta),
    }


def _write_multi_fasta(
    records: Sequence[tuple[str, str]], out_path: str | Path, *, wrap: int = 80
) -> None:
    lines: list[str] = []
    for name, seq in records:
        lines.append(f">{_safe_seq_name(name)}")
        lines += [seq[i : i + wrap] for i in range(0, len(seq), wrap)] or [seq]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: CM-free de-novo alignment
# ═════════════════════════════════════════════════════════════════════════════
def align_candidate(
    *,
    candidate_fasta: str | Path,
    homologs_fasta: str | Path,
    out_sto: str | Path,
    work_dir: str | Path,
    cpu: int = 4,
) -> Path:
    """RNAfold(candidate)→cmbuild→cmalign(homologs) → a Pfam Stockholm MSA with ``#=GC SS_cons``."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cand_name, cand_seq = read_single_sequence(candidate_fasta)
    cand_seq = degap_to_dna(cand_seq)

    structure = run_rnafold_structure(cand_seq)
    cand_sto = work / "candidate.sto"
    cand_sto.write_text(
        build_candidate_stockholm("candidate", cand_seq, structure), encoding="utf-8"
    )

    cm = work / "candidate.cm"
    _run(cmbuild_argv(cm, cand_sto, name="candidate"))
    _run(cmalign_argv(cm, homologs_fasta, out_sto, cpu=cpu))

    _assert_single_alignment_with_ss_cons(out_sto)
    return Path(out_sto)


def _assert_single_alignment_with_ss_cons(sto: str | Path) -> None:
    """Reuse covariation.stockholm_stats to confirm cmalign made one alignment; require SS_cons."""
    from tbox_finder.mining.covariation import stockholm_stats

    text = Path(sto).read_text(encoding="utf-8")
    n_alignments, n_sequences = stockholm_stats(text)
    if n_alignments != 1:
        raise HomologMsaError(
            f"cmalign output has {n_alignments} alignments, need exactly 1: {sto}"
        )
    if "#=GC SS_cons" not in text:
        raise HomologMsaError(f"cmalign output carries no #=GC SS_cons: {sto}")
    if n_sequences < 1:
        raise HomologMsaError(f"cmalign output has no sequence rows: {sto}")


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: score + the must-fire matched-control certification
# ═════════════════════════════════════════════════════════════════════════════
def score_msa(msa: str | Path | None, *, min_sequences: int = MIN_REAL_HOMOLOG_N) -> str:
    """Reuse the pinned per-candidate covariation status (A2: TOTAL_ACROSS_HELICES + min_seq=20)."""
    from tbox_finder.mining.covariation import covariation_spare_status

    return covariation_spare_status(None if msa is None else Path(msa), min_sequences=min_sequences)


def assert_matched_control(positive: str | Path, shuffled: str | Path) -> dict[str, Any]:
    """Assert the shuffled twin is a *matched* control of the positive MSA and return the summary.

    Matched = identical depth, identical width, identical ``#=GC SS_cons``, and identical
    per-column composition; only the inter-column correlation differs. A control that fails any of
    these is not matched — a green verdict on it would prove nothing (control-matchedness-must-be-
    asserted). Both files are the producer's own Pfam Stockholm output.
    """
    from tbox_finder.mining.msa_shuffle import read_pfam_alignment

    pos_rows, pos_gc = read_pfam_alignment(Path(positive))
    shf_rows, shf_gc = read_pfam_alignment(Path(shuffled))
    checks = {
        "depth_matched": len(pos_rows) == len(shf_rows),
        "width_matched": bool(pos_rows)
        and len({len(s) for _, s in pos_rows} | {len(s) for _, s in shf_rows}) == 1,
        "ss_cons_matched": pos_gc.get("SS_cons") == shf_gc.get("SS_cons")
        and pos_gc.get("SS_cons") is not None,
        "composition_matched": [Counter(c) for c in zip(*[s for _, s in pos_rows], strict=False)]
        == [Counter(c) for c in zip(*[s for _, s in shf_rows], strict=False)],
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise HomologMsaError(f"shuffled control is not matched to the positive: {failed}")
    return {"depth": len(pos_rows), **checks}


def certify_msa(
    *,
    positive_sto: str | Path,
    report_path: str | Path,
    shuffle_seed: int,
    min_sequences: int = MIN_REAL_HOMOLOG_N,
) -> dict[str, Any]:
    """The must-fire matched-control certification of the producer's output.

    Certified iff the producer's real MSA covaries (``positive`` == ``passed``) AND its
    column-shuffled twin — matched depth/width/SS_cons/composition, only covariation destroyed —
    does not (``shuffled`` != ``passed``, i.e. the signal came from real inter-column correlation,
    not the pipeline). Both are scored by the same pinned R-scape backend. Writes the report and
    returns it; raises :class:`HomologMsaError` if either leg does not fire.
    """
    from tbox_finder.mining.msa_shuffle import (
        read_pfam_alignment,
        shuffle_alignment_columns,
        write_pfam_alignment,
    )

    positive_status = score_msa(positive_sto, min_sequences=min_sequences)

    rows, gc = read_pfam_alignment(Path(positive_sto))
    shuffled_rows = shuffle_alignment_columns(rows, seed=shuffle_seed)
    with tempfile.TemporaryDirectory(prefix="tbox-msa-certify-") as tmp:
        shuffled_sto = Path(tmp) / "shuffled.sto"
        shuffled_sto.write_text(write_pfam_alignment(shuffled_rows, gc), encoding="utf-8")
        shuffled_status = score_msa(shuffled_sto, min_sequences=min_sequences)
        matched = assert_matched_control(positive_sto, shuffled_sto)

    from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED

    positive_fires = positive_status == STATUS_PASSED
    # The negative leg must be a *demonstrated* absence of covariation (STATUS_FAILED), not merely
    # "not passed": STATUS_UNAVAILABLE means R-scape could not score the (depth-matched) shuffle —
    # a timeout / error / unparseable output — i.e. "no power", which reads identically to "no
    # signal" and must not certify (control-matchedness-must-be-asserted). Fail-closed here.
    shuffled_negative = shuffled_status == STATUS_FAILED
    certified = positive_fires and shuffled_negative and matched["composition_matched"]

    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "certified": certified,
        "positive_status": positive_status,
        "shuffled_status": shuffled_status,
        "min_sequences": min_sequences,
        "shuffle_seed": shuffle_seed,
        "matched_control": matched,
        "expects": {"positive": STATUS_PASSED, "shuffled": STATUS_FAILED},
    }
    Path(report_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not certified:
        raise HomologMsaError(
            "homolog-MSA supply NOT certified: "
            f"positive={positive_status!r} (need {STATUS_PASSED!r}), "
            f"shuffled={shuffled_status!r} (need {STATUS_FAILED!r}); report={report_path}"
        )
    return report


# ═════════════════════════════════════════════════════════════════════════════
# CLI (one subcommand per stage; each runs in its own pinned env — see the sbatch)
# ═════════════════════════════════════════════════════════════════════════════
def _cmd_seed(args: argparse.Namespace) -> int:
    name, seq = read_stockholm_sequence(args.from_stockholm, index=args.index)
    write_fasta_record(args.name or name, seq, args.out)
    print(f"seed: wrote {args.out} ({len(seq)} nt) from {args.from_stockholm} row {args.index}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    report = search_homologs(
        candidate_fasta=args.candidate_fasta,
        out_fasta=args.out_fasta,
        engine=args.engine,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
        min_pident=args.min_pident,
        min_cov=args.min_cov,
        prefix=args.db_prefix,
        fm_index=args.fm_index,
        genome_dir=args.genome_dir,
        min_sequences=args.min_sequences,
        tblout=args.tblout,
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"search: {report['n_records']} records "
        f"({report['n_homologs']} homologs); sufficient={report['sufficient']}"
    )
    return 0 if report["sufficient"] else 4  # 4 = not producible (fail-closed ⇒ spared / cert-fail)


def _cmd_align(args: argparse.Namespace) -> int:
    out = align_candidate(
        candidate_fasta=args.candidate_fasta,
        homologs_fasta=args.homologs_fasta,
        out_sto=args.out_sto,
        work_dir=args.work_dir,
        cpu=args.cpu,
    )
    print(f"align: wrote {out}")
    return 0


def _cmd_certify(args: argparse.Namespace) -> int:
    report = certify_msa(
        positive_sto=args.msa,
        report_path=args.report,
        shuffle_seed=args.shuffle_seed,
        min_sequences=args.min_sequences,
    )
    print(
        f"certify: certified={report['certified']} "
        f"positive={report['positive_status']} shuffled={report['shuffled_status']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbox_finder.mining.homolog_msa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="extract a degapped DNA seed from a Stockholm MSA row")
    p_seed.add_argument("--from-stockholm", required=True)
    p_seed.add_argument("--index", type=int, default=0)
    p_seed.add_argument("--name", default=None)
    p_seed.add_argument("--out", required=True)
    p_seed.set_defaults(func=_cmd_seed)

    p_search = sub.add_parser("search", help="search the A1 target DB → candidate+homolog FASTA")
    p_search.add_argument("--candidate-fasta", required=True)
    p_search.add_argument("--out-fasta", required=True)
    p_search.add_argument("--engine", choices=("blastn", "nhmmer"), required=True)
    # P6-frozen inclusion cutoffs (ADR-0006 D17): keyword-required, NO default.
    p_search.add_argument("--evalue", type=float, required=True)
    p_search.add_argument("--max-target-seqs", type=int, required=True)
    p_search.add_argument("--min-pident", type=float, required=True)
    p_search.add_argument("--min-cov", type=float, required=True)
    p_search.add_argument("--db-prefix", default=BLAST_PREFIX)
    p_search.add_argument("--fm-index", default=FM_INDEX)
    p_search.add_argument("--genome-dir", default=GENOME_DIR)
    p_search.add_argument("--tblout", default=None)
    p_search.add_argument("--min-sequences", type=int, default=MIN_REAL_HOMOLOG_N)
    p_search.add_argument("--report", default=None)
    p_search.set_defaults(func=_cmd_search)

    p_align = sub.add_parser("align", help="CM-free de-novo alignment (RNAfold→cmbuild→cmalign)")
    p_align.add_argument("--candidate-fasta", required=True)
    p_align.add_argument("--homologs-fasta", required=True)
    p_align.add_argument("--out-sto", required=True)
    p_align.add_argument("--work-dir", required=True)
    p_align.add_argument("--cpu", type=int, default=4)
    p_align.set_defaults(func=_cmd_align)

    p_cert = sub.add_parser("certify", help="must-fire matched-control certification of an MSA")
    p_cert.add_argument("--msa", required=True)
    p_cert.add_argument("--report", required=True)
    p_cert.add_argument("--shuffle-seed", type=int, required=True)
    p_cert.add_argument("--min-sequences", type=int, default=MIN_REAL_HOMOLOG_N)
    p_cert.set_defaults(func=_cmd_certify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
