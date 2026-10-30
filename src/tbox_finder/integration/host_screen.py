"""P3-16′-remint — the decoy-host screen (ADR-0005 A15 Pins 1–3).

Why this module exists
----------------------
P3-16 spliced each of 692 §9.1 decoys into a 1,024-nt ``mining_pool_v0`` window and called the
result a labelled negative. Those windows are carved from the **flanks of catalogued T-boxes**
(``mining/pool.py::carve_window``), and their only T-box screen was a ±50-nt coordinate mask
against the union prior. A coordinate mask cannot see uncatalogued sequence, and a catalogued
coordinate can stop short of the element's own CM extent. Re-derived here as the ``v1_audit``
block: a handful of those 692 hosts carry an RF00230 hit well above the screen's threshold, and
every v1 decoy that drew a Stage-1 candidate is one of them. A gated negative containing T-box
sequence is a label error in the direction that *manufactures* a precision gain, which is how
the v1 pass was carried.

So A15 screens the host pool **before the seeded draw**, on two independent mechanisms, and
this module is that screen:

* **Pin 1, the fold screen** — a candidate window is refused if (1) its whole pre-splice
  1,024-nt window shares any N-free canonical 32-mer with the *screened DNA* (the union of both
  Stage-1 arms' training contexts, the admitted mined windows, every ``decoys_v0`` insert, and
  every ``stage2_dataset`` row the shipped Stage-2 trained on), or (2) it overlaps a
  same-accession training interval. Rule 1 reads **both strands** (k-mers are canonical) and the
  **whole** window, which is what the shipped ``host_fold_overlap`` did not.
* **Pin 2, the content screen** — every Pin-1 survivor is searched with the two ADR-0005 D2
  canonical models (``cmsearch -T 10 -Z 1``, default filtered pipeline, both strands) and
  against the CM-free T box core motif ``GGTGGNACCGCG``. A window is refused at **≥ 10.0 bits**
  on either model, or on a motif match.
* **Pin 3** — the survivors are the pool the v2 mint draws from, and the post-mint legs
  re-search the minted contigs, including the orthogonal filter-off (``--max``) arm. Its
  assertion verdict is **per model** (``POSTMINT_THRESHOLD_BITS``: RF00230 ≥ 20.0, TBDB001
  ≥ 12.0) while the searches still report from ``CONTENT_THRESHOLD_BITS``, so the band between
  the two is published as ``noise_residue`` rather than filtered away.

What a passing screen is **not** (A15 Pin 2, A8-style lower bound): the refusal count
*lower-bounds* T-box content in the pool. Admission means "below 10 bits on RF00230/TBDB001 and
motif-free", never "contains no T-box". The refused-window annex is committed so a later hit
there can be adjudicated rather than silently counted as a false positive, and A15 Pin 7(a)
carries the CM-circularity disclosure the screen itself creates.

Six legs, three environments
----------------------------
Nothing above the pandas boundary imports pandas, because the ``search`` leg runs in the
``infernal`` env, which has ``cmsearch`` and **no pandas**::

    prep            (data)      pool + screened DNA + Pin 1  -> FASTAs + prep.json
    search          (infernal)  FASTA x model                -> tblouts + search.json
    report          (data)      Pin 2 + controls + panel + v1 audit -> reports/p3/host_screen.json
    familiarity     (data)      A15 Pin 7(b) familiarity baseline, per benchmark item
    postmint-prep   (data)      minted v2 contigs            -> assertion FASTAs
    postmint-report (data)      A15 Pin 3's post-mint assertion

Every leg records **repo-relative** paths: an absolute path in a committed artifact publishes
this machine's layout into a public repo ([[committing-real-tool-output-fixtures]]). Data inputs
are refused unless they are repo-relative, exactly as ``scripts/mint_p3_16_benchmark.py`` does,
and ``--data-root`` points at the checkout that holds the DVC-tracked bytes.

This module grades nothing and pins no *gate* threshold: the A13 Pin 1 statistic and its strict
``>`` live in ``precision.py``. Pin 1's ``k = 32`` and Pin 2's 10.0 bits are A15 construction
values, fixed before any v2 score existed.

Pin 3's assertion thresholds (``POSTMINT_THRESHOLD_BITS``) were **re-signed after the mint**, on
2026-10-11, and that needs saying plainly rather than burying. They are nonetheless
**score-blind**: the re-sign was measured on three synthetic matched nulls and on the covariance
models' own bit scores, and no Stage-1 output, Stage-2 output, familiarity score or value of the
A13 statistic was computed, consulted or in existence for the re-signed configuration when it
was set. What changed is the *assertion's* calibration, not the draw — seed 42 stands, every
host is the one the screened draw took, and Pin 3's remedy clause (a §7 stop, then a re-mint)
governs a draw that admitted T-box content, which is not what the first run found. The
measurement that set the numbers is a committed artifact (``KIND_NOISE_FLOOR``) and the verdict
refuses to certify itself against a threshold that does not clear that artifact's measured floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder import infernal, masking, provenance, refs
from tbox_finder.data import embedding as emb
from tbox_finder.data import window_dataset as wd
from tbox_finder.integration import stage1_folds
from tbox_finder.integration.two_stage import write_json

SCHEMA_VERSION = "1.0"
STEP = "P3-16′-remint"
GENERATED_BY = "src/tbox_finder/integration/host_screen.py"
RULE = "workflow/rules/integration.smk :: p3_16_host_screen"
ADR = "ADR-0005 A15"
PRD = "§9.1; §2.3"

#: Report ``kind`` discriminators — :func:`validate_report` grades two different artifacts and
#: must never apply one clause set to the other: a clause that cannot find its subject is
#: vacuously satisfied ([[clauses-must-guard-emptiness]]).
KIND_SCREEN = "host_screen"
KIND_POSTMINT = "benchmark_v2_postmint_screen"

#: The calibration artifact ``POSTMINT_THRESHOLD_BITS`` is signed against. Produced by
#: ``src/tbox_finder/integration/noise_floor.py``, not by this module — this module only
#: *consumes* it, so a threshold cannot be re-signed without the measurement it claims.
KIND_NOISE_FLOOR = "postmint_noise_floor"

#: A15 Pin 1's k. Canonical (min of forward and reverse complement), so one array screens both
#: strands; 32 is also the widest k whose 2-bit code fits a ``uint64``.
KMER_K = 32

#: A15 Pin 1's alphabet rule, recorded in the report so no reader has to infer it.
ALPHABET_RULE = (
    "upper-case, U->T; any k-mer containing a non-ACGT character is skipped, on both sides"
)

#: A15 Pin 2's CM-free backstop: positions 3–14 of the T box sequence consensus
#: 5′-AGGGUGGNACCGCG-3′ [Grundy et al. 2002, PMID:11917026, DOI:10.1093/nar/30.7.1646; Zhang et
#: al. 2021, PMID:33809732, DOI:10.3390/ijms22063033; Campos-Chavez et al. 2024, PMID:39097611,
#: DOI:10.1038/s41467-024-50885-x (accessed 2026-10-11)]. ``N`` means **any of ACGT**, not the
#: literal character ``N``: an ambiguity code in a host window is not evidence of the consensus.
MOTIF = "GGTGGNACCGCG"
MOTIF_RE = re.compile("GGTGG[ACGT]ACCGCG")

#: A15 Pin 2's content threshold, in bits. The verdict reads the tblout **bit-score** column,
#: never the ``inc`` flag (an E-value-derived inclusion call).
CONTENT_THRESHOLD_BITS = 10.0

#: ``-T`` for the two audit FASTAs (the v1 hosts and the sensitivity panel). They exist to
#: record full best-bit distributions, so their reporting threshold is 0 bits, not the screen's.
AUDIT_THRESHOLD_BITS = 0.0

#: A15 Pin 3's **post-mint assertion** thresholds, per model, in bits — re-signed 2026-10-11
#: against a measured matched-null floor (``reports/p3/postmint_noise_floor.json``).
#:
#: Not ``CONTENT_THRESHOLD_BITS`` for both, which is what Pin 3 was drafted with. Three matched
#: nulls containing **zero** T-boxes by construction — the spliced inserts alone, the same hosts
#: re-spliced with the same inserts at a phase moved ≥ 50 nt, and 1,000 admitted-but-undrawn
#: windows, 3,073 sequences per model over both pipelines — measure the floor each model's
#: verdict has to clear:
#:
#: * **RF00230: 15.1 bits** (16 of 3,073 reach ≥ 10 under ``--max``), against a weakest genuine
#:   T-box of **31.9** over 1,201 positives, none below 20. A 10-bit RF00230 verdict therefore
#:   measures the noise floor of ``cmsearch`` on ~1 kb of DNA, not T-box homology. 20.0 clears
#:   it with 4.9 bits of margin.
#: * **TBDB001: 11.9 bits** (1 of 3,073, on an admitted-but-undrawn pool window under ``--max``),
#:   so **12.0**, signed 2026-10-11 as a second correction. An earlier figure of 9.9 came from a
#:   1,382-sequence subset — two arms of the four — and the fuller measurement withdrew it. The
#:   margin here is **0.1 bits**, and a null maximum is an extreme-value statistic that rises
#:   with the number of sequences searched, so unlike RF00230's this clause is *not* robust to a
#:   larger null; the artifact records ``n_null_sequences`` beside the maximum for that reason.
#:
#: TBDB001's floor now overlaps the bottom of its own genuine range — real T-boxes under it
#: start at 10.0 and 3.4 % score below 11.9 — so no TBDB001 threshold both excludes its noise
#: and admits its weakest real signal. 12.0 costs 3.8 % of real TBDB001 hits and, measured on
#: the v1 audit, costs **nothing**: it refuses the same 9 of the 11 v1 hosts that 10.0 did, and
#: the two TBDB001 hits it releases (11.4 and 10.1) carry RF00230 at 71.5 and 24.4.
#:
#: One threshold was pinned across two models whose score scales differ ~5×; that, not the
#: screen and not the draw, is what the first run's failure measured.
#:
#: Each value must be **≥** ``CONTENT_THRESHOLD_BITS``: an assertion configured more sensitively
#: than the screen that feeds it is unsatisfiable by construction, however sound both are
#: ([[assertion-stronger-than-its-screen-cannot-pass]]). ``_postmint_problems`` re-derives that,
#: and also that each value strictly clears its own model's measured floor.
POSTMINT_THRESHOLD_BITS: dict[str, float] = {"RF00230": 20.0, "TBDB001": 12.0}

#: The post-mint report's own schema, versioned separately from the screen report's
#: ``SCHEMA_VERSION``: the two kinds carry independent clause sets (``_postmint_problems`` vs
#: ``_screen_problems``), and per-model ``threshold_bits`` changes the shape of **this** report
#: only. Versioning the pair together would stale the committed screen bytes for no reason
#: ([[new-gate-clause-invalidates-old-reports]]).
POSTMINT_SCHEMA_VERSION = "1.1"

#: ``-Z`` in Mb. Fixed per invocation, so the filter regime and the E-value column do not depend
#: on how many windows were batched into one FASTA (A15 Pin 2).
SEARCH_SPACE_MB = 1.0

#: The two ADR-0005 D2 canonical models. RF00230 is the operative screen; TBDB001 is kept as
#: **declared redundancy** (A15 Pin 2 measured its marginal over RF00230 as nil), and the report
#: carries the per-model and marginal refusal counts that say so. Exactly two, because
#: "marginal over the other model" is defined for a pair.
SCREEN_CMS: tuple[tuple[str, str], ...] = (
    ("RF00230", "data/external/refs/RF00230.cm"),
    ("TBDB001", "data/external/refs/TBDB001.cm"),
)

#: The env locks the two tiers run under, recorded in their provenance.
ENV_LOCK = "envs/infernal.conda-lock.yml"
ENV_LOCK_DATA = "envs/data.conda-lock.yml"

#: The four screened-DNA sources of A15 Pin 1, in the order the amendment lists them.
SOURCE_FOLD_CONTEXTS = "fold_contexts"
SOURCE_MINED_WINDOWS = "mined_windows"
SOURCE_DECOY_INSERTS = "decoy_inserts"
SOURCE_STAGE2 = "stage2_population"
SCREENED_SOURCES: tuple[str, ...] = (
    SOURCE_FOLD_CONTEXTS,
    SOURCE_MINED_WINDOWS,
    SOURCE_DECOY_INSERTS,
    SOURCE_STAGE2,
)
#: Only two sources carry genomic coordinates, so only they can fire A15 Pin 1's rule 2.
COORDINATE_SOURCES: tuple[str, ...] = (SOURCE_FOLD_CONTEXTS, SOURCE_MINED_WINDOWS)

#: The shipped Stage-2 arm's admitted training population, from its checkpoint provenance
#: (``extra.population.n_admitted``). A15 Pin 1 (iv) requires the derived count to equal it, so
#: the screen refuses rather than screening against a population it merely believes in. Pinned
#: here *as well as* read from the sidecar: a provenance file that drifted would otherwise move
#: the expectation that is supposed to bind it.
STAGE2_EXPECTED_ADMITTED = 9059
#: The predicate A15 Pin 1 (iv) names, re-derived through ``stage2/train.py``'s own
#: :func:`~tbox_finder.stage2.train.select_rows` rather than re-implemented
#: ([[promote-dont-duplicate-is-a-correctness-rule]]). Measured 2026-10-11 on the committed
#: table: the shipped selector and the amendment's prose predicate agree on all 30,542 rows,
#: and both give 9,059.
STAGE2_PREDICATE = (
    'fold_random == "train" AND NOT calib AND (nested_train true OR nested_role null) — '
    "read through stage2.train.select_rows(rung='train', admit_parentless_decoys=True)"
)

#: What the two committed Stage-1 runs recorded embedding, per pool
#: (``scripts/mint_p3_16_benchmark.py::EXPECTED_EMBEDDED``). The familiarity leg refuses unless
#: its re-derivation reproduces them: a familiarity score built on the wrong insert set would
#: mis-state the A15 Pin 7(b) confound in the direction that makes the gate look clean.
EXPECTED_EMBEDDED: dict[str, dict[str, int]] = {
    "production": {"dinuc_shuffled": 702, "structured_rna": 2999},
    "twin": {"dinuc_shuffled": 602, "structured_rna": 2999},
}

#: The ``NAME`` each CM carries, which is what Infernal prints in the tblout's *query name*
#: column. Read off the pinned model files (``RF00230.cm``: ``NAME T-box``, ``ACC RF00230``;
#: ``TBDB001.cm``: ``NAME seed_ILE``, no ``ACC``), so neither equals this repo's model key. Every
#: tblout's rows are checked against the entry for the model the search says produced it: that is
#: what makes "this is RF00230's tblout" evidence rather than a filename convention.
CM_QUERY_NAMES: dict[str, str] = {"RF00230": "T-box", "TBDB001": "seed_ILE"}

#: Default input paths (repo-relative), mirroring the minter's defaults.
DEFAULT_MINING_POOL = "data/processed/negatives/mining_pool_v0.parquet"
DEFAULT_DECOYS = "data/processed/negatives/decoys_v0.parquet"
DEFAULT_DATASET = "data/processed/stage2_dataset.parquet"
DEFAULT_SPLIT_TABLE = "data/processed/splits/split_assignments.parquet"
DEFAULT_CONTEXT = "data/interim/flank_context/context_v0.parquet"
DEFAULT_LABELS = "data/processed/labels/labels_v0.parquet"
DEFAULT_BENCHMARK_V1 = "data/interim/p3_16/benchmark_v1.json"
DEFAULT_RF00230_MASTER = "data/external/refs/RF00230_master.fa"
DEFAULT_STAGE2_PROVENANCE = (
    "data/processed/checkpoints/stage2_rinalmo/aux1.0_lr1e-4/provenance.json"
)

DEFAULT_OUT_DIR = "data/interim/p3_16/host_screen"
DEFAULT_REPORT = "reports/p3/host_screen.json"
DEFAULT_ADMITTED_LIST = "reports/p3/host_screen_admitted.txt"
DEFAULT_REFUSED_ANNEX = "reports/p3/host_screen_refused.json"
DEFAULT_POSTMINT_REPORT = "reports/p3/benchmark_v2_postmint_screen.json"

#: Where ``noise_floor.py`` writes the calibration ``POSTMINT_THRESHOLD_BITS`` is signed against.
DEFAULT_NOISE_FLOOR_REPORT = "reports/p3/postmint_noise_floor.json"


#: FASTA basenames the three legs agree on, and the ``-T`` each is searched at.
FASTA_SURVIVORS = "survivors.fa"
FASTA_CONTROLS = "controls.fa"
FASTA_V1_HOSTS = "v1_hosts.fa"
FASTA_PANEL = "panel.fa"
SCREEN_FASTAS: tuple[str, ...] = (FASTA_SURVIVORS, FASTA_CONTROLS, FASTA_V1_HOSTS, FASTA_PANEL)
FASTA_THRESHOLDS: dict[str, float] = {
    FASTA_SURVIVORS: CONTENT_THRESHOLD_BITS,
    FASTA_CONTROLS: CONTENT_THRESHOLD_BITS,
    FASTA_V1_HOSTS: AUDIT_THRESHOLD_BITS,
    FASTA_PANEL: AUDIT_THRESHOLD_BITS,
}
FASTA_POSTMINT_ALL = "postmint_all.fa"
FASTA_POSTMINT_LABEL0 = "postmint_label0.fa"
#: The searches A15 Pin 3's assertion REQUIRES, per arm: every D2 model over every FASTA that arm
#: covers. Pinned as a set because the clause set otherwise validates only the searches that
#: happen to be present — a report with every TBDB001 search deleted, or with an empty
#: ``filter_off`` block, passed every arm clause and reported ``n_violations == 0``
#: ([[clauses-must-guard-emptiness]]). The ``--max`` arm is scoped to the label-0 contigs, the
#: only ones the assertion reads.
REQUIRED_POSTMINT_SEARCHES: dict[str, tuple[str, ...]] = {
    "screen": (FASTA_POSTMINT_ALL, FASTA_POSTMINT_LABEL0),
    "filter_off": (FASTA_POSTMINT_LABEL0,),
}

#: The fixed phase every control splice uses — a constant, so a control's refusal cannot be an
#: artefact of where its segment landed.
CONTROL_PHASE = 300
#: Length of a must-fire k-mer control segment (A15 Pin 1: "a ≥ 200-nt segment").
CONTROL_SEGMENT_NT = 200
#: How many sequences of a source to scan for a control segment. The controls need *one* segment
#: each; a bounded scan keeps the prep leg's cost in the pool, where it belongs.
CONTROL_SCAN_SEQUENCES = 400
#: The 5′ truncation fractions of A15 Pin 2's sensitivity panel.
PANEL_TRUNCATIONS: tuple[float, ...] = (0.25, 0.50)

#: Control names :func:`validate_report` requires to be present and fired.
CONTROL_HOST_CLEAN = "pin1_control_host_is_clean"
CONTROL_COORD_ONLY = "pin1_coordinate_only"
CONTROL_CONTENT = "content_rf00230_master"
CONTROL_CONTENT_HOST = "content_host_unspliced"


class HostScreenError(RuntimeError):
    """Raised when an input or an intermediate violates the screen's contract."""


# ══════════════════════════════════════════════════════════════════════════════════════
# Pure helpers (no pandas; unit-tested)
# ══════════════════════════════════════════════════════════════════════════════════════
def normalise_dna(seq: Any) -> str:
    """Upper-case one sequence and map ``U`` → ``T`` (A15 Pin 1's alphabet rule).

    Upper-cased **first**, so a lower-case ``u`` cannot survive as a ``u`` — the ordering
    ``embedding.normalise_insert`` and the minter's ``transcribe_to_dna`` both use. The screened
    DNA includes ``stage2_dataset.rna_sequence`` and the RNA-alphabet ``leader_decoy`` inserts,
    so a screen that did not transcribe would compare RNA against DNA and share nothing.
    """
    return str(seq).upper().replace("U", "T")


#: ``ACGT`` → 0–3; every other byte → 255, which :func:`canonical_kmers` reads as "not a base".
_BASE_CODE = np.full(256, 255, dtype=np.uint8)
for _index, _base in enumerate("ACGT"):
    _BASE_CODE[ord(_base)] = _index
_INVALID_BASE = np.uint8(255)


def _base_codes(text: str) -> np.ndarray:
    """2-bit codes for a normalised DNA string, 255 at every non-ACGT position.

    ``errors="replace"`` keeps the mapping length-preserving on a non-ASCII character (one
    ``?`` per character, which codes 255): a shifted code array would silently re-phase every
    k-mer downstream of it.
    """
    raw = np.frombuffer(text.encode("ascii", errors="replace"), dtype=np.uint8)
    return _BASE_CODE[raw]


def canonical_kmers(seq: Any, k: int = KMER_K) -> np.ndarray:
    """Canonical 2-bit ``k``-mers of ``seq`` as ``uint64``, in position order.

    Canonical = ``min(forward, reverse complement)``, which is what makes **one** sorted array a
    both-strand test: A15 Pin 1 refuses a shared k-mer "on either strand", and the shipped
    ``host_fold_overlap`` it replaces compared forward strands only.

    A k-mer containing any non-ACGT character is **skipped**, on both sides of every comparison.
    N-runs are not shared DNA and the mined pool admits ``ACGTN``, so counting them would let
    two windows "share" a run of ambiguity.

    :func:`normalise_dna` is applied here rather than demanded of the caller, because a caller
    that forgot it would get a silently empty intersection against the RNA-derived sources.
    """
    if not 1 <= int(k) <= 32:
        raise HostScreenError(f"k must be in 1..32 for a uint64 2-bit code; got {k!r}")
    k = int(k)
    codes = _base_codes(normalise_dna(seq))
    n_windows = codes.size - k + 1
    if n_windows <= 0:
        return np.empty(0, dtype=np.uint64)
    bad = codes == _INVALID_BASE
    # "No bad base inside the window", from one cumulative sum: O(L) rather than O(L·k).
    cumulative = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    valid = (cumulative[k:] - cumulative[:-k]) == 0
    safe = np.where(bad, 0, codes).astype(np.uint64)
    complement = np.uint64(3) - safe
    two = np.uint64(2)
    forward = np.zeros(n_windows, dtype=np.uint64)
    reverse = np.zeros(n_windows, dtype=np.uint64)
    for offset in range(k):
        # At k = 32 the left shift drops the top two bits, which IS the 64-bit mask.
        forward = (forward << two) | safe[offset : offset + n_windows]
        back = k - 1 - offset
        reverse = (reverse << two) | complement[back : back + n_windows]
    return np.minimum(forward, reverse)[valid]


def sorted_unique_kmers(
    sequences: Iterable[Any], k: int = KMER_K, *, compress_at: int = 1 << 23
) -> np.ndarray:
    """One sorted, unique ``uint64`` array over every sequence — a screened-DNA index.

    Deduplicated incrementally (``compress_at`` pending k-mers between compactions) so the whole
    corpus's k-mers never have to be materialised at once. The result is identical to a single
    ``np.unique`` over the concatenation, which the unit tier asserts.
    """
    parts: list[np.ndarray] = []
    pending = 0
    merged = np.empty(0, dtype=np.uint64)
    for sequence in sequences:
        chunk = canonical_kmers(sequence, k)
        if chunk.size:
            parts.append(chunk)
            pending += chunk.size
        if pending >= compress_at:
            merged = np.unique(np.concatenate([merged, *parts]))
            parts, pending = [], 0
    if parts:
        merged = np.unique(np.concatenate([merged, *parts]))
    return merged


def kmer_hits(window_kmers: np.ndarray, sorted_source_kmers: np.ndarray) -> int:
    """How many of ``window_kmers`` occur in a **sorted, unique** source array.

    A count rather than a bool: the per-source attribution A15 Pin 1 requires needs to say how
    much DNA a window shares, and ``> 0`` is the refusal. ``searchsorted`` rather than
    ``np.isin``, so the source array is bisected per query instead of re-sorted per call.
    """
    window_kmers = np.asarray(window_kmers, dtype=np.uint64)
    source = np.asarray(sorted_source_kmers, dtype=np.uint64)
    if window_kmers.size == 0 or source.size == 0:
        return 0
    position = np.searchsorted(source, window_kmers)
    position = np.minimum(position, source.size - 1)
    return int(np.count_nonzero(source[position] == window_kmers))


def intervals_overlap(
    accession_a: Any, a_start: Any, a_end: Any, accession_b: Any, b_start: Any, b_end: Any
) -> bool:
    """A15 Pin 1 rule 2 for one pair of genomic intervals, **closed** and strand-folded.

    Both sides are ``masking.normalize_accession``-normalised, because the coordinate sources in
    this repo disagree about the INSDC ``.N`` version suffix and comparing the namespaces raw
    makes the rule a silent no-op ([[namespace-mismatch-invisible-noop]]). Coordinates are
    ``masking.normalize_locus``-folded (the repo encodes strand as ``start > end``) and compared
    as 1-based **inclusive** bounds — what ``flank_context.forward_bounds`` emits, what
    ``mining_pool_v0.locus_start/locus_end`` hold, and what ``context_v0.region_start/
    region_stop`` are. A missing accession addresses no replicon and so never overlaps.
    """
    if masking.is_missing(accession_a) or masking.is_missing(accession_b):
        return False
    if masking.normalize_accession(accession_a) != masking.normalize_accession(accession_b):
        return False
    lo_a, hi_a = masking.normalize_locus(int(a_start), int(a_end))
    lo_b, hi_b = masking.normalize_locus(int(b_start), int(b_end))
    return masking.intervals_overlap(lo_a, hi_a, lo_b, hi_b)


@dataclass(frozen=True)
class Hit:
    """One ``cmsearch --tblout --fmt 1`` data row, keeping the fields a verdict needs.

    ``score`` is the **bit** column (field 15) and ``inc`` the inclusion flag (field 17). A15
    Pin 2 pins that the verdict reads ``score``: ``inc`` is an E-value call, and ``-Z 1`` makes
    the E-value column a function of a pinned search space rather than of the window's content.

    ⚠ ``model`` is the tblout's *query name* column, which is the **CM's own ``NAME`` field** —
    ``T-box`` for ``RF00230.cm`` and ``seed_ILE`` for ``TBDB001.cm`` — and **not** this repo's
    model key. A verdict therefore attributes a bit score by *which search wrote the tblout*,
    never by this string; keying a per-model dict on it would silently leave every model's score
    ``None`` while the report still showed a model breakdown. The string is kept as evidence and
    checked against :data:`CM_QUERY_NAMES`, which is what proves a tblout came from the CM the
    search says it did.
    """

    target: str
    model: str
    seq_from: int
    seq_to: int
    strand: str
    trunc: str
    score: float
    evalue: float
    inc: str


def parse_tblout_hits(text: str) -> list[Hit]:
    """Parse ``cmsearch --tblout`` (``--fmt 1``) text into :class:`Hit` rows.

    A short row raises rather than being dropped, exactly as
    :func:`tbox_finder.infernal.parse_tblout` does: a format change that silently shrank the hit
    set would read as "the CM found nothing", which is the fail-open direction for a screen.
    """
    hits: list[Hit] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=infernal.TBLOUT_FIXED_FIELDS)
        if len(fields) < infernal.TBLOUT_FIXED_FIELDS:
            raise HostScreenError(
                f"tblout line {lineno} has {len(fields)} fields, expected at least "
                f"{infernal.TBLOUT_FIXED_FIELDS}: {line!r}"
            )
        try:
            hits.append(
                Hit(
                    target=fields[infernal.TBLOUT_TARGET_COL],
                    model=fields[2],
                    seq_from=int(fields[7]),
                    seq_to=int(fields[8]),
                    strand=fields[9],
                    trunc=fields[10],
                    score=float(fields[infernal.TBLOUT_SCORE_COL]),
                    evalue=float(fields[infernal.TBLOUT_EVALUE_COL]),
                    inc=fields[16],
                )
            )
        except ValueError as error:
            raise HostScreenError(
                f"tblout line {lineno} is not a --fmt 1 data row: {line!r}"
            ) from error
    return hits


def motif_present(seq: Any) -> bool:
    """Whether ``GGTGGNACCGCG`` occurs on **either** strand of ``seq`` (A15 Pin 2).

    Both strands, because a host window carries no orientation: the mined pool is
    forward-genomic DNA, and a T-box in it may sit on the minus strand.
    """
    text = normalise_dna(seq)
    return bool(MOTIF_RE.search(text)) or bool(MOTIF_RE.search(wd.reverse_complement(text)))


def motif_spans(seq: Any) -> list[tuple[int, int]]:
    """Every ``GGTGGNACCGCG`` occurrence on either strand, as 0-based half-open spans.

    Positions are needed, not just presence: A15 Pin 3's leader exemption is an interval test,
    and a motif reported only as a whole-contig boolean could not be tested against the leader's
    insert interval at all. A minus-strand match is reported at its forward-strand coordinates,
    so both strands are comparable with a CM hit's span.
    """
    text = normalise_dna(seq)
    spans = [(m.start(), m.end()) for m in MOTIF_RE.finditer(text)]
    n = len(text)
    spans += [(n - m.end(), n - m.start()) for m in MOTIF_RE.finditer(wd.reverse_complement(text))]
    return sorted(set(spans))


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether two 0-based half-open intervals share at least one position."""
    return a[0] < b[1] and b[0] < a[1]


def _as_bits(value: Any) -> float | None:
    """A bit score, or ``None`` for "no hit". A ``bool`` is not a bit score.

    ``float(True)`` is ``1.0`` and would order against a threshold like any other number
    ([[bool-equals-one-passes-config-gates]]); a non-finite value is likewise refused rather
    than silently compared with 10.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def at_least(value: Any, threshold: float) -> bool:
    """Whether a recorded bit score is at or above ``threshold``; a missing score never is.

    Not ``(_as_bits(v) or -inf) >= threshold``: a **0.0-bit hit is falsy**, so that idiom reads a
    real score of zero as "no hit" — invisible at a 10-bit threshold and wrong the moment any
    caller asks a question at or below zero, which the audit searches (``-T 0``) do by design.
    """
    bits = _as_bits(value)
    return bits is not None and bits >= float(threshold)


def postmint_threshold(model: str, thresholds: Mapping[str, float] | None = None) -> float:
    """A15 Pin 3's assertion threshold for ``model``, in bits.

    Refuses an unpinned model rather than falling back to a default. A model the amendment never
    priced is an **unsigned configuration**, and the two plausible defaults are both wrong: the
    screen's 10.0 would re-impose the floor the 2026-10-11 re-sign measured away, and the
    strictest pinned value would silently grade a third model on a number measured for another.

    ``thresholds`` is read as an argument, not off the module global, so a caller that pins its
    own mapping is honoured and the global's value reaches every read path — a module global is
    not seen by a def-time default or a ``from``-import rebind
    ([[pinned-constant-has-two-read-paths]]).
    """
    table = POSTMINT_THRESHOLD_BITS if thresholds is None else thresholds
    try:
        value = table[model]
    except KeyError:
        raise HostScreenError(
            f"{model!r} has no A15 Pin 3 assertion threshold; pinned: "
            f"{sorted(table)} — add one to the amendment before asserting on it"
        ) from None
    bits = _as_bits(value)
    if bits is None:
        raise HostScreenError(f"the Pin 3 threshold for {model!r} is not a bit score: {value!r}")
    if bits < CONTENT_THRESHOLD_BITS:
        raise HostScreenError(
            f"the Pin 3 assertion threshold for {model!r} ({bits}) is below the Pin 2 screen's "
            f"({CONTENT_THRESHOLD_BITS}): an assertion stronger than its own screen cannot pass"
        )
    return bits


def content_refused(
    best_bits_by_model: Mapping[str, Any],
    motif: Any,
    threshold: float = CONTENT_THRESHOLD_BITS,
) -> bool:
    """A15 Pin 2's verdict: ``>= threshold`` bits on **either** model, or a motif match.

    ``>=`` and not ``>``: A15 pins "a hit of **≥ 10.0 bits**", so a window sitting exactly on
    the value is refused. A model with no hit contributes nothing rather than a floor.
    """
    if bool(motif):
        return True
    return any(at_least(value, threshold) for value in best_bits_by_model.values())


def truncate_window(sequence: str, truth_start: int, truth_end: int, fraction: float) -> str:
    """A15 Pin 2's 5′-truncated panel variant: the window from ``truth_start + ⌊f·L⌋`` on.

    ``L`` is the locus length, so ``f`` is a fraction **of the element**, not of the window. The
    result is *trail-window geometry*: the surviving 3′ part of the element sits flush at the 5′
    edge with real downstream context behind it, which is the shape a trail host window presents
    to the screen — and the shape of the v1 host that carried its parent's own element.
    """
    start = int(truth_start)
    length = int(truth_end) - start
    if length <= 0:
        raise HostScreenError(f"truth bounds [{truth_start}, {truth_end}) are not a locus")
    if not 0.0 <= float(fraction) < 1.0:
        raise HostScreenError(f"truncation fraction must be in [0, 1); got {fraction!r}")
    offset = start + int(math.floor(float(fraction) * length))
    if not 0 <= offset < len(sequence):
        raise HostScreenError(f"truncation offset {offset} is outside a {len(sequence)}-nt window")
    return sequence[offset:]


def surrogate(prefix: str, index: int) -> str:
    """A single-token FASTA name, ``w000001``-style.

    ``cmsearch``'s tblout ``target name`` column holds only the first whitespace-delimited token
    of a header, and a candidate id carries ``:`` while a decoy id carries ``|`` and ``/``. The
    id map is written beside the FASTA, so the surrogate is reversible.
    """
    return f"{prefix}{int(index):06d}"


def search_argv(
    *,
    cm: str,
    fasta: str,
    tblout: str,
    threshold_bits: float,
    cpu: int,
    no_filters: bool = False,
) -> list[str]:
    """The argv :func:`tbox_finder.infernal.run_cmsearch` builds for one screen search.

    A **mirror**, because ``run_cmsearch`` returns hits rather than its command line. The search
    leg refuses unless this mirror's flags equal the ones ``cmsearch`` itself printed in the
    tblout's ``# Option settings:`` line, and the unit tier pins the same equality against a
    committed real tblout — so a drift in either construction fails rather than publishing an
    argv nothing ran ([[verify-the-line-you-ship]]).
    """
    argv = ["cmsearch", "--noali", "--cpu", str(int(cpu)), "--tblout", str(tblout)]
    if no_filters:
        argv.append("--max")
    argv += ["-T", repr(float(threshold_bits)), "-Z", repr(float(SEARCH_SPACE_MB))]
    argv += [str(cm), str(fasta)]
    return argv


_OPTION_SETTINGS = "# Option settings:"
#: Flags whose following token is a value rather than a path. ``--tblout``'s value IS a path, so
#: it is deliberately absent.
_VALUED_FLAGS: frozenset[str] = frozenset({"--cpu", "-T", "-Z", "-E"})


def option_settings_line(text: str) -> str:
    """The ``# Option settings:`` line of a tblout, or ``""`` when the trailer is absent."""
    for line in text.splitlines():
        if line.startswith(_OPTION_SETTINGS):
            return line[len(_OPTION_SETTINGS) :].strip()
    return ""


def option_settings_options(line: str) -> dict[str, Any]:
    """An ``# Option settings:`` line (or an argv) as ``{flag: value}``, paths dropped.

    What survives is ``{"--noali": True, "--cpu": "8", "-T": "10.0", "-Z": "1.0"}``: the options
    that ran, free of the absolute paths Infernal prints for the binary, the CM, the FASTA and
    the tblout. Those paths are machine layout; the sha256 beside each model is what identifies
    it.

    A **mapping**, not a token list, for two reasons. Infernal prints its options in its own
    order (``-Z 1.0 --tblout … --noali -T 10.0 --cpu 2``), which is not the order
    :func:`search_argv` builds, so the comparison has to be order-insensitive — and a sorted
    *token list* would make ``-T 10.0 -Z 1.0`` compare equal to ``-T 1.0 -Z 10.0``, which is
    exactly the confusion the comparison exists to catch.
    """
    tokens = [token for token in line.split() if token]
    options: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            if token in _VALUED_FLAGS and index + 1 < len(tokens):
                options[token] = tokens[index + 1]
                index += 2
                continue
            options[token] = True
        index += 1
    return options


def infernal_version_line() -> str:
    """Infernal's own version line, from ``cmsearch -h``.

    ``cmsearch --version`` is **not** valid in this build (1.1.5 exits with a usage dump), so
    the version is read off the help banner.
    """
    import shutil
    import subprocess

    if shutil.which("cmsearch") is None:
        raise infernal.InfernalNotAvailableError(
            "cmsearch is not on PATH; run the search leg inside the pinned infernal env "
            "(envs/infernal.yml, infernal=1.1.5)"
        )
    proc = subprocess.run(["cmsearch", "-h"], capture_output=True, text=True, check=False)
    for line in proc.stdout.splitlines():
        if "INFERNAL" in line:
            return line.lstrip("# ").strip()
    raise HostScreenError("could not read a version line from `cmsearch -h`")


def tblout_digest(text: str) -> str:
    """sha256 of a tblout's **data** rows.

    The header comments carry absolute paths and a timestamp, which are sanitised out of a
    committed fixture; a dropped or edited hit row is not
    ([[committing-real-tool-output-fixtures]]).
    """
    rows = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def best_bits(hits_by_model: Mapping[str, Sequence[Hit]]) -> dict[str, float | None]:
    """Per **model key**, the best bit score among that model's hits (``None`` where it has none).

    The caller supplies the keying, because a tblout's own query-name column carries the CM's
    ``NAME`` rather than the repo's model key (see :class:`Hit`). Attribution by "which search
    produced these hits" is the only form that cannot silently collapse to all-``None``.
    """
    out: dict[str, float | None] = {}
    for model, hits in hits_by_model.items():
        scores = [hit.score for hit in hits]
        out[model] = max(scores) if scores else None
    return out


def check_query_names(hits: Sequence[Hit], model: str) -> None:
    """Refuse a tblout whose rows name a CM other than the one the search says it searched.

    The pinned ``NAME`` is in :data:`CM_QUERY_NAMES`. A tblout written by the wrong model, or a
    stale file left behind by an earlier run with a different pairing, would otherwise be read
    as that model's bit scores.
    """
    expected = CM_QUERY_NAMES.get(model)
    if expected is None:
        raise HostScreenError(f"no pinned CM query name for model {model!r}")
    unexpected = sorted({hit.model for hit in hits} - {expected})
    if unexpected:
        raise HostScreenError(
            f"a tblout attributed to {model} carries query name(s) {unexpected}, not {expected!r}"
        )


def hits_by_target(text: str, *, model: str | None = None) -> dict[str, list[Hit]]:
    """One tblout's hits, grouped by target name, with the CM's own name checked when given."""
    hits = parse_tblout_hits(text)
    if model is not None:
        check_query_names(hits, model)
    grouped: dict[str, list[Hit]] = {}
    for hit in hits:
        grouped.setdefault(hit.target, []).append(hit)
    return grouped


def familiarity_fraction(sequence: Any, sorted_source_kmers: np.ndarray) -> float | None:
    """A15 Pin 7(b): the fraction of a contig's canonical 32-mers present in training DNA.

    ``None`` when the contig admits no N-free 32-mer at all — a fraction over an empty
    denominator is not 0.0, and a leg that returned 0.0 would publish "unfamiliar" for a window
    it never measured.
    """
    kmers = canonical_kmers(sequence)
    if kmers.size == 0:
        return None
    return kmer_hits(kmers, sorted_source_kmers) / float(kmers.size)


# ══════════════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════════════
def require_repo_relative(value: Any, *, what: str) -> str:
    """Refuse an input path that is absolute or climbs out with ``..``.

    The minter's contract, for the minter's reason: these strings land in a provenance block in
    a public repo. ``--data-root`` is how a worktree reads the main checkout's DVC bytes while
    the recorded path stays repo-relative ([[repo-relative-resolves-to-the-modules-own-root]]).
    """
    text = str(value)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise HostScreenError(
            f"{what} must be repo-relative (got {text!r}); use --data-root to point at the "
            "checkout that holds the data"
        )
    return text


#: This module's own checkout. Outputs are recorded relative to it FIRST: the committed
#: artifacts (``reports/p3/*``) live beside the code, while ``--data-root`` may point at
#: another checkout's ``data/``. Relativising an output against the data root alone records
#: a worktree-nested path like ``.claude/worktrees/<branch>/reports/p3/...``, which does not
#: resolve from the repository root the committed report is read at.
CODE_ROOT = Path(__file__).resolve().parents[3]


def recorded_path(path: str | Path, root: str | Path) -> str:
    """How an **output** path is recorded: repo-relative, or its basename when outside.

    Relative to :data:`CODE_ROOT` when it is inside it, else to ``root`` (the data root),
    else its basename. A scratch run writes outside both — that is what a scratch run is
    for — and an absolute path must still not reach a report. Reduced to a basename rather
    than refused, so the screen can be run and inspected before anything is committed; the
    report says which of the two happened in ``out_dir_inside_data_root``.
    """
    resolved = Path(path).resolve()
    for candidate in (CODE_ROOT, Path(root).resolve()):
        try:
            return str(resolved.relative_to(candidate))
        except ValueError:
            continue
    return resolved.name


def inside(path: str | Path, root: str | Path) -> bool:
    """Whether ``path`` resolves inside ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def provenance_record(
    *,
    leg: str,
    inputs: Sequence[str | Path],
    root: Path,
    env_lock: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
    rule: str = RULE,
    script: str = GENERATED_BY,
    adr: str = ADR,
) -> dict[str, Any]:
    """A CLAUDE.md §11 provenance record whose input keys are **repo-relative**.

    ``provenance.sha256_paths`` keys by the path exactly as it was passed, and these legs are
    handed an absolute ``--data-root``, so passing the paths straight through would publish this
    machine's directory layout in a committed artifact (the PR #103 finding). The hashes are the
    identity evidence and the strings only locators, so the keys are re-written here — the same
    post-hoc re-key ``stage2/cm_confirmer.py`` applies to its own record.

    ``rule`` / ``script`` / ``adr`` default to this module's own identity. They exist so a
    sibling module that runs these screens on another population (``hardneg_pool.py``, ADR-0005
    A17) records *itself* as the producer through the same re-keying code, rather than carrying
    a copy of it.
    """
    record = provenance.build_provenance(
        rule=rule,
        script=script,
        seed=provenance.DEFAULT_SEED,
        inputs=list(inputs),
        env_lock=env_lock,
        adr=adr,
        repo_root=Path(__file__).resolve().parents[3],
        extra={"leg": leg, **dict(extra or {})},
    )
    record["inputs"] = {
        recorded_path(key, root): digest for key, digest in record["inputs"].items()
    }
    record["outputs"] = {
        recorded_path(key, root): digest for key, digest in record["outputs"].items()
    }
    return record


def _model_names() -> list[str]:
    """The screen's model names, refused unless there are exactly two.

    "Marginal over the other model" — the count A15 Pin 2's declared-redundancy claim rests on —
    is defined for a pair. A third model would need its own definition, so this raises instead
    of quietly reporting a marginal against an arbitrary one.
    """
    names = [model for model, _ in SCREEN_CMS]
    if len(names) != 2:
        raise HostScreenError(
            f"the screen is defined for the ADR-0005 D2 model PAIR; SCREEN_CMS holds {names!r}"
        )
    return names


def _other_model(model: str) -> str:
    names = _model_names()
    return names[1] if model == names[0] else names[0]


# ══════════════════════════════════════════════════════════════════════════════════════
# Screened DNA (A15 Pin 1) — data env
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ScreenedDna:
    """A15 Pin 1's four sources as k-mer indices, plus the two that carry coordinates.

    ``sequences`` keeps each source's own strings, because the must-fire controls must be built
    from the **same** bytes the index was built from: a control segment re-read from a second
    load could differ from the screened DNA and prove nothing about it.
    """

    kmers: dict[str, np.ndarray]
    loci: dict[str, masking.LocusIndex]
    blocks: dict[str, dict[str, Any]]
    sequences: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowVerdict:
    """A15 Pin 1's two-rule outcome for one candidate window."""

    n_kmers: int
    kmer_sources: dict[str, int]
    coord_sources: tuple[str, ...]

    @property
    def rule_kmer(self) -> bool:
        return any(count > 0 for count in self.kmer_sources.values())

    @property
    def rule_coord(self) -> bool:
        return bool(self.coord_sources)

    @property
    def refused(self) -> bool:
        return self.rule_kmer or self.rule_coord

    @property
    def attributed_to(self) -> list[str]:
        """The sources whose DNA this window shares, sorted — the controls' subject."""
        return sorted(name for name, count in self.kmer_sources.items() if count)

    def as_json(self) -> dict[str, Any]:
        return {
            "n_kmers": int(self.n_kmers),
            "kmer_sources": {
                name: int(count) for name, count in sorted(self.kmer_sources.items()) if count
            },
            "coord_sources": list(self.coord_sources),
            "rule_kmer": self.rule_kmer,
            "rule_coord": self.rule_coord,
            "refused": self.refused,
        }


def screen_window(
    sequence: Any, accession: Any, start: Any, end: Any, dna: ScreenedDna
) -> WindowVerdict:
    """Apply A15 Pin 1's two rules to one window and record **which source** fired.

    Attribution is the must-fire controls' subject: a screen that refused the right number of
    windows for the wrong reason would pass a bare count. Both rules are evaluated — no
    short-circuit — so a window refused by one still records the other, which is what makes
    ``n_refused_kmer_only`` / ``n_refused_coord_only`` measurable.
    """
    kmers = canonical_kmers(sequence)
    kmer_sources = {name: kmer_hits(kmers, index) for name, index in dna.kmers.items()}
    coord_sources = tuple(
        name
        for name in COORDINATE_SOURCES
        if name in dna.loci and dna.loci[name].is_masked(accession, int(start), int(end))
    )
    return WindowVerdict(
        n_kmers=int(kmers.size), kmer_sources=kmer_sources, coord_sources=coord_sources
    )


def _read_parquet(path: Path, columns: Sequence[str] | None = None) -> Any:
    import pandas as pd  # lazy: the search leg has no pandas

    return pd.read_parquet(path, columns=list(columns) if columns else None)


def _stage2_population(
    dataset_path: Path, provenance_path: Path, *, recorded: str
) -> tuple[list[str], dict[str, Any]]:
    """A15 Pin 1 (iv): the ``rna_sequence`` of every row the shipped Stage-2 arm trained on.

    Routed through ``stage2/train.py``'s own :func:`~tbox_finder.stage2.train.select_rows` — the
    function that selected the checkpoint's population — and **refused** unless the count equals
    the checkpoint provenance's ``extra.population.n_admitted`` (9,059). That provenance is
    itself a recorded reconstruction (``reconstructed: true``), so the pinned constant is
    checked too: a sidecar that drifted would otherwise move the expectation meant to bind it.
    """
    from tbox_finder.stage2 import train as stage2_train  # lazy: torch-free, but heavy

    frame = _read_parquet(dataset_path)
    rows = frame.to_dict("records")
    admitted_index, census = stage2_train.select_rows(rows, rung=stage2_train.TRAIN_RUNG)
    sidecar = json.loads(provenance_path.read_text(encoding="utf-8"))
    declared = ((sidecar.get("extra") or {}).get("population") or {}).get("n_admitted")
    n_admitted = len(admitted_index)
    if n_admitted != STAGE2_EXPECTED_ADMITTED or declared != STAGE2_EXPECTED_ADMITTED:
        raise HostScreenError(
            f"Stage-2 population mismatch: derived {n_admitted}, the checkpoint provenance says "
            f"{declared!r}, A15 Pin 1 (iv) pins {STAGE2_EXPECTED_ADMITTED}. Refusing to screen "
            "against a population that is not the one the shipped Stage-2 trained on."
        )
    sequences = [normalise_dna(rows[i]["rna_sequence"]) for i in admitted_index]
    block = {
        "n": n_admitted,
        "predicate": STAGE2_PREDICATE,
        "provenance": recorded,
        "provenance_n_admitted": int(declared),
        "census": {
            "n_rows_scanned": census.get("n_rows_scanned"),
            "admitted_by_route": census.get("admitted_by_route"),
            "refused": census.get("refused"),
        },
    }
    return sequences, block


def build_screened_dna(
    *,
    root: Path,
    mining_pool: str,
    decoys: str,
    dataset: str,
    context: str,
    labels: str,
    split_table: str,
    stage2_provenance: str,
) -> tuple[ScreenedDna, dict[str, Any]]:
    """Assemble A15 Pin 1's four screened-DNA sources and the two coordinate indices.

    The screened DNA is a deliberate **superset**: a candidate must be clean against the union,
    not only against the gated arm's DNA. Source (iii) (every ``decoys_v0`` insert)
    over-approximates on purpose — A7 measured ``gc_background`` and ``leader_decoy`` as
    unembedded, so their presence here is not a training-exposure claim.
    """
    from tbox_finder.data import negatives as negatives_mod  # lazy: pandas below

    # (i) both arms' training folds. Production's fold contains the twin's (one loader
    # override), but the union is taken from both rather than assumed, so a change to either
    # arm's definition cannot silently shrink the screened DNA.
    folds = {
        arm: stage1_folds.training_fold_ids(
            arm=arm,
            split_table=str(root / split_table),
            context=str(root / context),
            labels=str(root / labels),
        )
        for arm in sorted(stage1_folds.ARM_EXCLUDE_GATE4_EVAL)
    }
    fold_ids = set().union(*folds.values())
    context_frame = _read_parquet(
        root / context, ["record_id", "accession", "region_start", "region_stop", "context_seq"]
    )
    fold_rows = context_frame[context_frame["record_id"].astype(str).isin(fold_ids)]
    fold_sequences = [str(value) for value in fold_rows["context_seq"]]
    fold_intervals = [
        (str(accession), int(lo), int(hi))
        for accession, lo, hi in zip(
            fold_rows["accession"],
            fold_rows["region_start"],
            fold_rows["region_stop"],
            strict=True,
        )
    ]

    # (ii) the admitted mined windows both arms trained on — the loader's own default.
    mined_rows, mined_report = negatives_mod.load_admitted_pool_rows(str(root / mining_pool))
    mined_sequences = [str(row["sequence"]) for row in mined_rows]
    mined_intervals = [
        (str(row["accession"]), int(row["locus_start"]), int(row["locus_end"]))
        for row in mined_rows
    ]

    # (iii) every decoys_v0 insert.
    decoy_frame = _read_parquet(root / decoys, ["decoy_id", "sequence"])
    decoy_sequences = [str(value) for value in decoy_frame["sequence"]]

    # (iv) the shipped Stage-2 arm's admitted population.
    stage2_sequences, stage2_block = _stage2_population(
        root / dataset, root / stage2_provenance, recorded=stage2_provenance
    )

    sequences = {
        SOURCE_FOLD_CONTEXTS: fold_sequences,
        SOURCE_MINED_WINDOWS: mined_sequences,
        SOURCE_DECOY_INSERTS: decoy_sequences,
        SOURCE_STAGE2: stage2_sequences,
    }
    kmers = {name: sorted_unique_kmers(values) for name, values in sequences.items()}
    loci = {
        SOURCE_FOLD_CONTEXTS: masking.LocusIndex.from_records(fold_intervals),
        SOURCE_MINED_WINDOWS: masking.LocusIndex.from_records(mined_intervals),
    }
    blocks: dict[str, dict[str, Any]] = {
        SOURCE_FOLD_CONTEXTS: {
            "n_records": len(fold_sequences),
            "n_fold_ids": len(fold_ids),
            "n_fold_ids_by_arm": {arm: len(ids) for arm, ids in sorted(folds.items())},
            "n_kmers": int(kmers[SOURCE_FOLD_CONTEXTS].size),
            "n_intervals": loci[SOURCE_FOLD_CONTEXTS].n_intervals,
            "n_accessions": loci[SOURCE_FOLD_CONTEXTS].n_accessions,
            "interval_rule": "context_v0 [region_start, region_stop] joined on record_id",
            "source": context,
        },
        SOURCE_MINED_WINDOWS: {
            "n": len(mined_sequences),
            "n_kmers": int(kmers[SOURCE_MINED_WINDOWS].size),
            "n_intervals": loci[SOURCE_MINED_WINDOWS].n_intervals,
            "n_accessions": loci[SOURCE_MINED_WINDOWS].n_accessions,
            "interval_rule": (
                "mining_pool_v0 [locus_start, locus_end] — the WINDOW's own genomic interval, "
                "not its parent locus's (mining/pool.py::carve_window writes forward_bounds of "
                "the carved window)"
            ),
            "source": mining_pool,
            "admission": {
                "require_parent_nested_train": mined_report.get("require_parent_nested_train"),
                "max_records": mined_report.get("max_records"),
                "n_records": mined_report.get("n_records"),
            },
        },
        SOURCE_DECOY_INSERTS: {
            "n": len(decoy_sequences),
            "n_kmers": int(kmers[SOURCE_DECOY_INSERTS].size),
            "source": decoys,
            "sha256": provenance.sha256_file(root / decoys),
            "over_approximates": (
                "every decoys_v0 insert, including the gc_background and leader_decoy pools A7 "
                "kept out of both training mixes — not a training-exposure claim (A15 Pin 1 iii)"
            ),
        },
        SOURCE_STAGE2: {
            **stage2_block,
            "n_kmers": int(kmers[SOURCE_STAGE2].size),
            "source": dataset,
            "stage2_dataset_sha256": provenance.sha256_file(root / dataset),
        },
    }
    screened = ScreenedDna(kmers=kmers, loci=loci, blocks=blocks, sequences=sequences)
    report = {
        "k": KMER_K,
        "alphabet_rule": ALPHABET_RULE,
        "rule_1": (
            "the WHOLE pre-splice window's canonical (both-strand) 32-mers against the union of "
            "the four sources"
        ),
        "rule_2": (
            "same-accession overlap with a training interval, accession-version-normalised, "
            "strand-folded, closed bounds"
        ),
        "sources": blocks,
        "n_kmers_summed_over_sources": int(sum(array.size for array in kmers.values())),
    }
    return screened, report


def load_candidates(root: Path, mining_pool: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The candidate host pool, built **exactly** as the minter builds it.

    ``load_admitted_pool_rows(require_parent_nested_train=False)``, then "parent out of fold":
    the two steps ``scripts/mint_p3_16_benchmark.py`` takes, so the screen and the mint read one
    pool rather than two that agree by coincidence. Measured 2026-10-11 on the committed table:
    43,649 admitted, 27,941 out-of-fold candidates.

    There is deliberately **no** record cap: a screen measured on a truncated pool cannot
    certify the draw it feeds, and a cost knob that can write a passing artifact is the shape
    that certified a phase-exit path once already ([[cost-knobs-can-certify]]).
    """
    from tbox_finder.data import negatives as negatives_mod  # lazy: pandas below

    admitted, report = negatives_mod.load_admitted_pool_rows(
        str(root / mining_pool), require_parent_nested_train=False
    )
    if report.get("max_records") is not None:
        raise HostScreenError("the candidate pool was loaded with a max_records cap")
    candidates = [row for row in admitted if not row.get("parent_nested_train")]
    return candidates, report


# ══════════════════════════════════════════════════════════════════════════════════════
# The must-fire controls (A15 Pin 1)
# ══════════════════════════════════════════════════════════════════════════════════════
def control_entry(
    name: str, mechanism: str, expected: Any, observed: Any, *, note: str | None = None
) -> dict[str, Any]:
    """One must-fire control, with ``fired`` **derived** from expected vs observed.

    Never a flag a caller asserts: a control whose ``fired`` was written independently of its own
    measurement is a report field, not a control ([[gate-clauses-need-re-derivation]]).
    :func:`validate_report` re-derives this equality again on the written artifact.
    """
    entry: dict[str, Any] = {
        "name": name,
        "mechanism": mechanism,
        "expected": expected,
        "observed": observed,
        "fired": observed == expected,
    }
    if note:
        entry["note"] = note
    return entry


def control_segment(
    sequences: Sequence[str],
    source: str,
    dna: ScreenedDna,
    *,
    segment_nt: int = CONTROL_SEGMENT_NT,
    max_sequences: int = CONTROL_SCAN_SEQUENCES,
) -> tuple[str, bool] | None:
    """A ``segment_nt``-nt segment of ``source``, preferring one no other source carries.

    Returns ``(segment, exclusive)``. **Exclusivity is not always achievable, and that is a
    property of the sources rather than of the screen**: an admitted mined window is a slice of
    its in-fold parent's ``context_seq``, so every source-(ii) segment is also source-(i) DNA,
    and an embedded decoy's sequence is both a ``decoys_v0`` insert and a Stage-2 row. So the
    per-source must-fire control asserts that the attribution **contains** its source — which
    still fails if that source's index is dead, because the segment would then be attributed to
    the containing source alone — and the exclusivity variant is asserted only where a unique
    segment exists.
    """
    others = [name for name in dna.kmers if name != source]
    fallback: str | None = None
    for sequence in list(sequences)[:max_sequences]:
        text = normalise_dna(sequence)
        for begin in range(0, max(1, len(text) - segment_nt + 1), segment_nt):
            segment = text[begin : begin + segment_nt]
            if len(segment) < segment_nt:
                continue
            kmers = canonical_kmers(segment)
            if kmers.size == 0 or kmer_hits(kmers, dna.kmers[source]) == 0:
                # The segment must be IN its own source, or it tests nothing.
                continue
            if not any(kmer_hits(kmers, dna.kmers[other]) for other in others):
                return segment, True
            if fallback is None:
                fallback = segment
    return (fallback, False) if fallback is not None else None


def pin1_controls(
    *,
    dna: ScreenedDna,
    host: Mapping[str, Any],
    coordinate_only: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """A15 Pin 1's must-fire controls: one per source, plus coordinate and accession-suffix.

    Each k-mer control splices a source segment into a **Pin-1-clean survivor** window at a
    fixed phase, so the only DNA in the control that any source carries is the segment. The
    unspliced host is screened too and must come back clean: a control host that was itself
    refused would prove nothing about the segment
    ([[control-matchedness-must-be-asserted]]).
    """
    controls: list[dict[str, Any]] = []
    sequences: dict[str, str] = {}
    host_sequence = normalise_dna(host["sequence"])
    host_verdict = screen_window(
        host_sequence, host["accession"], host["locus_start"], host["locus_end"], dna
    )
    controls.append(
        control_entry(
            CONTROL_HOST_CLEAN,
            "matched control: the unspliced survivor host must pass both Pin 1 rules",
            {"rule_kmer": False, "rule_coord": False},
            {"rule_kmer": host_verdict.rule_kmer, "rule_coord": host_verdict.rule_coord},
            note=f"host candidate {host['candidate_id']}",
        )
    )
    for source in SCREENED_SOURCES:
        found = control_segment(dna.sequences.get(source, []), source, dna)
        if found is None:
            controls.append(
                control_entry(
                    f"pin1_kmer_{source}",
                    f"a {CONTROL_SEGMENT_NT}-nt segment of {source}, spliced into a clean host",
                    {"segment_found": True},
                    {"segment_found": False},
                    note="no scanned segment of this source is present in its own index",
                )
            )
            continue
        segment, exclusive = found
        spliced = emb.splice(host_sequence, segment, CONTROL_PHASE)
        verdict = screen_window(
            spliced, host["accession"], host["locus_start"], host["locus_end"], dna
        )
        name = f"pin1_kmer_{source}"
        sequences[name] = spliced
        controls.append(
            control_entry(
                name,
                (
                    f"a {len(segment)}-nt segment of {source}, spliced into a Pin-1-clean "
                    f"survivor at phase {CONTROL_PHASE}: rule 1 fires and the attribution "
                    f"contains {source}"
                ),
                {"rule_kmer": True, "contains_source": True, "rule_coord": False},
                {
                    "rule_kmer": verdict.rule_kmer,
                    "contains_source": source in verdict.attributed_to,
                    "rule_coord": verdict.rule_coord,
                },
                note=f"attributed_to={verdict.attributed_to}; segment_exclusive={exclusive}",
            )
        )
        if exclusive:
            controls.append(
                control_entry(
                    f"pin1_kmer_exclusive_{source}",
                    (
                        f"the same segment is carried by {source} alone, so the attribution is "
                        "exactly that source"
                    ),
                    [source],
                    verdict.attributed_to,
                )
            )
    if coordinate_only is None:
        controls.append(
            control_entry(
                CONTROL_COORD_ONLY,
                "the one v1 host refused by coordinate with no shared 32-mer (A15 Pin 1)",
                {"present": True},
                {"present": False},
                note="that candidate was not found in the pool under the configured id prefix",
            )
        )
    else:
        verdict = coordinate_only["verdict"]
        controls.append(
            control_entry(
                CONTROL_COORD_ONLY,
                "the coordinate-only v1 host (A15 Pin 1): rule 2 fires, rule 1 does not",
                {"rule_kmer": False, "rule_coord": True},
                {"rule_kmer": verdict["rule_kmer"], "rule_coord": verdict["rule_coord"]},
                note=f"candidate {coordinate_only['candidate_id']}",
            )
        )
        for label, accession in sorted(coordinate_only["accession_variants"].items()):
            fired = any(
                index.is_masked(
                    accession,
                    int(coordinate_only["locus_start"]),
                    int(coordinate_only["locus_end"]),
                )
                for index in dna.loci.values()
            )
            controls.append(
                control_entry(
                    f"pin1_accession_{label}",
                    f"the same candidate's coordinates under accession {accession!r}",
                    True,
                    fired,
                    note="masking.normalize_accession strips the INSDC .N version on both sides",
                )
            )
    return controls, sequences


def coordinate_only_candidate(
    verdicts: Mapping[str, Mapping[str, Any]], prefix: str
) -> dict[str, Any] | None:
    """The A15 Pin 1 coordinate-only v1 host, found by id prefix, with its ``.N`` variants.

    The amendment names it by an abbreviated id (``4ef2d573…:lead``), so the full id is resolved
    from the pool rather than transcribed. Both accession spellings are then tested: the rule
    must fire with and without the version suffix, which is what proves the normalisation is on
    both sides of the comparison.
    """
    matches = sorted(cid for cid in verdicts if cid.startswith(prefix))
    if not matches:
        return None
    candidate_id = matches[0]
    record = verdicts[candidate_id]
    accession = str(record["accession"])
    stripped = masking.normalize_accession(accession)
    variants = {"as_given": accession}
    if accession == stripped:
        variants["with_suffix"] = f"{accession}.1"
    else:
        variants["stripped"] = stripped
    return {
        "candidate_id": candidate_id,
        "n_matches": len(matches),
        "verdict": dict(record),
        "accession_variants": variants,
        "locus_start": record["locus_start"],
        "locus_end": record["locus_end"],
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# prep (data env)
# ══════════════════════════════════════════════════════════════════════════════════════
#: The parent attributes A15 Pin 7(e)'s composition tables carry, read off the split table:
#: the same three ``scripts/mint_p3_16_benchmark.py::COMPOSITION_PARENT_FIELDS`` tabulates for the
#: v2 draw, under the same labels, so the v1 audit's table and the v2 manifest's are read side by
#: side on the same axes (side, ``nested_role``, leave-clade-out designation, resolved order).
COMPOSITION_PARENT_FIELDS: tuple[str, ...] = (
    "nested_role",
    "is_designated_loo_holdout",
    "resolved_order",
)
COMPOSITION_AXES: tuple[str, ...] = (
    "side",
    *(f"parent_{field}" for field in COMPOSITION_PARENT_FIELDS),
)
#: A parent the split table does not carry: a labelled bucket, never a dropped window.
ABSENT_PARENT = "absent_from_split_table"


def _composition_label(value: Any) -> str:
    """A JSON-key label for one parent attribute; a null is ``"null"``, never a ``None`` bucket."""
    if masking.is_missing(value):
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return str(value)


def _parent_meta(root: Path, split_table: str) -> dict[str, dict[str, str]]:
    """``{corpus record_id -> {parent_<field>: label}}`` for the composition tables.

    A null attribute is recorded as the token ``"null"`` rather than dropped: a composition
    table whose buckets do not sum to its own ``n`` is unreadable, and a ``None`` bucket is the
    adjacent failure ([[nulls-inflate-block-counts]]).
    """
    frame = _read_parquet(root / split_table, ["record_id", "source", *COMPOSITION_PARENT_FIELDS])
    corpus = frame[frame["source"] == "corpus"]
    return {
        str(row["record_id"]): {
            f"parent_{field}": _composition_label(row[field]) for field in COMPOSITION_PARENT_FIELDS
        }
        for row in corpus.to_dict("records")
    }


def _parent_fields(meta: Mapping[str, Mapping[str, str]], parent: str) -> dict[str, str]:
    """One window's parent attributes, or :data:`ABSENT_PARENT` on every axis."""
    found = meta.get(parent)
    if found is None:
        return {f"parent_{field}": ABSENT_PARENT for field in COMPOSITION_PARENT_FIELDS}
    return dict(found)


def first_fasta_record(path: str | Path) -> tuple[str, str]:
    """The first ``(name, sequence)`` of a FASTA — the content control's real T-box."""
    name: str | None = None
    chunks: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    break
                name = line[1:].strip()
                continue
            if name is not None:
                chunks.append(line.strip())
    if name is None or not chunks:
        raise HostScreenError(f"{path} holds no FASTA record")
    return name, "".join(chunks)


def _cmd_prep(args: argparse.Namespace) -> int:
    """LEG 1 (data env): the candidate pool, A15 Pin 1, and every FASTA the search leg eats."""
    started = time.monotonic()
    for name in (
        "mining_pool",
        "decoys",
        "dataset",
        "context",
        "labels",
        "split_table",
        "benchmark_v1",
        "rf00230_master",
        "stage2_provenance",
    ):
        require_repo_relative(getattr(args, name), what=f"--{name.replace('_', '-')}")
    root = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, pool_report = load_candidates(root, args.mining_pool)
    dna, screened_report = build_screened_dna(
        root=root,
        mining_pool=args.mining_pool,
        decoys=args.decoys,
        dataset=args.dataset,
        context=args.context,
        labels=args.labels,
        split_table=args.split_table,
        stage2_provenance=args.stage2_provenance,
    )
    parents = _parent_meta(root, args.split_table)

    # ── A15 Pin 1 over the whole candidate pool ─────────────────────────────────────────
    verdicts: dict[str, dict[str, Any]] = {}
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    survivors: list[Mapping[str, Any]] = []
    per_source = {name: {"kmer": 0, "coord": 0} for name in SCREENED_SOURCES}
    n_kmer = n_coord = n_kmer_only = n_coord_only = 0
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        rows_by_id[candidate_id] = row
        verdict = screen_window(
            row["sequence"], row["accession"], row["locus_start"], row["locus_end"], dna
        )
        parent = masking.row_text(row.get("source_record_id"))
        record = {
            **verdict.as_json(),
            # Measured here, once, and carried: the report leg then reads the motif flag that
            # was computed on the same bytes the k-mer rule saw, rather than re-reading 26 MB
            # of parquet and risking a second, different measurement.
            "motif": motif_present(row["sequence"]),
            "side": masking.row_text(row.get("side")) or "null",
            "accession": str(row["accession"]),
            "locus_start": int(row["locus_start"]),
            "locus_end": int(row["locus_end"]),
            "source_record_id": parent,
            **_parent_fields(parents, parent),
        }
        verdicts[candidate_id] = record
        for name, count in verdict.kmer_sources.items():
            if count:
                per_source[name]["kmer"] += 1
        for name in verdict.coord_sources:
            per_source[name]["coord"] += 1
        n_kmer += verdict.rule_kmer
        n_coord += verdict.rule_coord
        n_kmer_only += verdict.rule_kmer and not verdict.rule_coord
        n_coord_only += verdict.rule_coord and not verdict.rule_kmer
        if not verdict.refused:
            survivors.append(row)
    if not survivors:
        raise HostScreenError("Pin 1 refused every candidate; there is no pool left to screen")
    survivors.sort(key=lambda row: str(row["candidate_id"]))

    survivor_names = {
        surrogate("w", index): str(row["candidate_id"])
        for index, row in enumerate(survivors, start=1)
    }
    infernal.write_fasta(
        {
            name: str(rows_by_id[candidate_id]["sequence"])
            for name, candidate_id in survivor_names.items()
        },
        out_dir / FASTA_SURVIVORS,
    )
    for name, candidate_id in survivor_names.items():
        verdicts[candidate_id]["surrogate"] = name

    # ── the must-fire controls ──────────────────────────────────────────────────────────
    # The control host is the first motif-free survivor in id order: deterministic, and chosen
    # before any search, so no control can be selected on the strength of its own result. A CM
    # hit on it is only knowable after the search, and then the matched
    # `content_host_unspliced` control fails loudly rather than quietly weakening the other one.
    host = next((row for row in survivors if not verdicts[str(row["candidate_id"])]["motif"]), None)
    if args.control_host:
        host = rows_by_id.get(args.control_host)
        if host is None or verdicts[args.control_host]["refused"]:
            raise HostScreenError(
                f"--control-host {args.control_host!r} is not a Pin-1 survivor of this pool"
            )
    if host is None:
        raise HostScreenError("every Pin-1 survivor carries the motif; no clean control host")
    controls, control_sequences = pin1_controls(
        dna=dna,
        host=host,
        coordinate_only=coordinate_only_candidate(verdicts, args.coordinate_only_prefix),
    )

    # ── the content must-fire control: a real catalogued T-box in a clean survivor ───────
    tbox_name, tbox_seq = first_fasta_record(root / args.rf00230_master)
    tbox_dna = normalise_dna(tbox_seq)
    host_sequence = normalise_dna(host["sequence"])
    control_sequences[CONTROL_CONTENT] = emb.splice(host_sequence, tbox_dna, CONTROL_PHASE)
    control_sequences[CONTROL_CONTENT_HOST] = host_sequence
    control_meta = {
        meaning: {
            "surrogate": surrogate("c", index),
            "length": len(sequence),
            "motif": motif_present(sequence),
        }
        for index, (meaning, sequence) in enumerate(sorted(control_sequences.items()), start=1)
    }
    infernal.write_fasta(
        {meta["surrogate"]: control_sequences[meaning] for meaning, meta in control_meta.items()},
        out_dir / FASTA_CONTROLS,
    )

    # ── the v1 hosts (A15's v1_audit) ───────────────────────────────────────────────────
    v1 = json.loads((root / args.benchmark_v1).read_text(encoding="utf-8"))
    v1_items = sorted(
        (item for item in v1["items"] if item["label"] == 0), key=lambda item: str(item["host_id"])
    )
    v1_hosts: dict[str, dict[str, Any]] = {}
    v1_names: dict[str, str] = {}
    for index, item in enumerate(v1_items, start=1):
        host_id = str(item["host_id"])
        record = verdicts.get(host_id)
        if record is None:
            raise HostScreenError(
                f"v1 host {host_id!r} is not in this candidate pool, so the v1 audit would not "
                "be measuring the windows the v1 draw used"
            )
        name = surrogate("h", index)
        v1_names[name] = host_id
        v1_hosts[host_id] = {
            "surrogate": name,
            "contig_id": item["contig_id"],
            "pool": item["pool"],
            "side": record["side"],
            **{axis: record[axis] for axis in COMPOSITION_AXES[1:]},
            "motif": record["motif"],
            "n_kmers": record["n_kmers"],
            "kmer_sources": record["kmer_sources"],
            "coord_sources": record["coord_sources"],
            "rule_kmer": record["rule_kmer"],
            "rule_coord": record["rule_coord"],
            "refused": record["refused"],
        }
    infernal.write_fasta(
        {name: str(rows_by_id[host_id]["sequence"]) for name, host_id in v1_names.items()},
        out_dir / FASTA_V1_HOSTS,
    )

    # ── the sensitivity panel: the 1,201 positive windows, full and 5′-truncated ─────────
    contigs = {str(contig["contig_id"]): contig for contig in v1["contigs"]}
    positives = sorted(
        (item for item in v1["items"] if item["label"] == 1),
        key=lambda item: str(item["contig_id"]),
    )
    panel_names: dict[str, dict[str, Any]] = {}
    panel_sequences: dict[str, str] = {}
    for index, item in enumerate(positives, start=1):
        contig = contigs[str(item["contig_id"])]
        sequence = normalise_dna(contig["sequence"])
        variants = {"full": sequence}
        for fraction in PANEL_TRUNCATIONS:
            variants[f"t{int(round(fraction * 100)):02d}"] = truncate_window(
                sequence, int(contig["truth_start"]), int(contig["truth_end"]), fraction
            )
        for label, text in variants.items():
            name = f"{surrogate('p', index)}_{label}"
            panel_names[name] = {
                "contig_id": contig["contig_id"],
                "variant": label,
                "length": len(text),
                "motif": motif_present(text),
            }
            panel_sequences[name] = text
    infernal.write_fasta(panel_sequences, out_dir / FASTA_PANEL)

    write_json(
        out_dir / "idmap.json",
        {
            "schema_version": SCHEMA_VERSION,
            "step": STEP,
            "kind": "host_screen_idmap",
            "survivors": survivor_names,
            "controls": {meta["surrogate"]: meaning for meaning, meta in control_meta.items()},
            "v1_hosts": v1_names,
            "panel": panel_names,
        },
    )

    n_refused = len(candidates) - len(survivors)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": "host_screen_prep",
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "pool": {
            "source": args.mining_pool,
            "sha256": provenance.sha256_file(root / args.mining_pool),
            "n_admitted": int(pool_report["n_records"]),
            "n_candidates": len(candidates),
            "selection": (
                "ADR-0005 A15 Pins 1-2 — load_admitted_pool_rows("
                "require_parent_nested_train=False) then parent out of fold, then the whole-"
                "window both-strand 32-mer fold screen and the RF00230/TBDB001 + motif content "
                "screen"
            ),
            "admission": {
                "max_records": pool_report.get("max_records"),
                "require_parent_nested_train": pool_report.get("require_parent_nested_train"),
                "n_rows_read": pool_report.get("n_rows_read"),
            },
        },
        "screened_dna": screened_report,
        "fold_screen": {
            "n_refused_kmer": int(n_kmer),
            "n_refused_coord": int(n_coord),
            "n_refused_kmer_only": int(n_kmer_only),
            "n_refused_coord_only": int(n_coord_only),
            "n_refused": int(n_refused),
            "per_source_windows": per_source,
            "n_survivors": len(survivors),
            # How often A15 Pin 1's alphabet rule actually bites on this pool. A window of
            # ACGT alone yields `window - k + 1` k-mers; an N-run costs up to k of them, so a
            # pool full of ambiguity would be screened on less DNA than it looks. Reported
            # rather than assumed: the rule is a pinned construction choice and this is its
            # measured cost.
            "alphabet": {
                "n_kmers_per_n_free_window": int(wd.WINDOW_NT - KMER_K + 1),
                "n_windows_with_skipped_kmers": sum(
                    1
                    for record in verdicts.values()
                    if record["n_kmers"] < wd.WINDOW_NT - KMER_K + 1
                ),
                "n_kmers_skipped": sum(
                    int(wd.WINDOW_NT - KMER_K + 1 - record["n_kmers"])
                    for record in verdicts.values()
                ),
            },
        },
        # Pin 1's own composition, beside the report's Pin-2 one: A15 Pin 7(e) is a claim about
        # the FOLD screen ("Pin 1 refuses trail windows about 3x as often as lead windows"), and
        # it is only readable where the content screen has not yet been applied.
        "composition": {
            "survivors": _composition([v for v in verdicts.values() if not v["refused"]]),
            "refused": _composition([v for v in verdicts.values() if v["refused"]]),
            "note": "A15 Pin 7(e), measured on the fold screen alone",
        },
        "candidates": verdicts,
        "v1_hosts": v1_hosts,
        "panel": panel_names,
        "controls": controls,
        "controls_all_fired": all(entry["fired"] for entry in controls),
        "control_meta": control_meta,
        "content_control": {
            "name": CONTROL_CONTENT,
            "insert": tbox_name,
            "insert_length": len(tbox_dna),
            "insert_source": args.rf00230_master,
            "host_candidate_id": str(host["candidate_id"]),
            "phase": CONTROL_PHASE,
            "motif_in_insert": motif_present(tbox_dna),
        },
        "fasta": {
            basename: {
                "path": recorded_path(out_dir / basename, root),
                "n_records": count,
                "sha256": provenance.sha256_file(out_dir / basename),
                "threshold_bits": FASTA_THRESHOLDS[basename],
            }
            for basename, count in (
                (FASTA_SURVIVORS, len(survivor_names)),
                (FASTA_CONTROLS, len(control_meta)),
                (FASTA_V1_HOSTS, len(v1_names)),
                (FASTA_PANEL, len(panel_names)),
            )
        },
        "out_dir_inside_data_root": inside(out_dir, root),
        "wall_s": round(time.monotonic() - started, 2),
        "provenance": provenance_record(
            leg="prep",
            inputs=[
                root / args.mining_pool,
                root / args.decoys,
                root / args.dataset,
                root / args.context,
                root / args.labels,
                root / args.split_table,
                root / args.benchmark_v1,
            ],
            root=root,
            env_lock=root / ENV_LOCK_DATA,
            extra={"out_dir": recorded_path(out_dir, root)},
        ),
    }
    write_json(out_dir / "prep.json", payload)
    print(
        f"prep: {len(candidates)} candidates -> {len(survivors)} survivors "
        f"({n_refused} refused: {n_kmer} k-mer, {n_coord} coordinate, {n_kmer_only} k-mer-only, "
        f"{n_coord_only} coordinate-only); {len(v1_names)} v1 hosts, {len(panel_names)} panel "
        f"records, {len(control_meta)} controls; controls_all_fired="
        f"{payload['controls_all_fired']}; {payload['wall_s']}s"
    )
    return 0


# ══════════════════════════════════════════════════════════════════════════════════════
# search (infernal env — no pandas)
# ══════════════════════════════════════════════════════════════════════════════════════
def _search_one(
    *,
    root: Path,
    out_dir: Path,
    tblout_root: Path,
    arm_dir: str,
    basename: str,
    model: str,
    cm: str,
    threshold_bits: float,
    cpu: int,
    no_filters: bool,
    timeout_s: float,
) -> dict[str, Any]:
    """One (FASTA, model) search, recording the argv, the executed option line, and the FASTA.

    Refuses when the argv and the option line disagree: the report would otherwise publish a
    command line that did not run ([[verify-the-line-you-ship]]).

    ``fasta_sha256`` is what binds the tblouts to the windows ``prep`` wrote. Without it, pairing
    a prep run's verdicts with tblouts searched from a *different* FASTA reads as "no hits" —
    the fail-open direction, which admits every window — and the surrogate names are positional,
    so they would collide rather than show up as unknown targets.
    """
    fasta = out_dir / basename
    relative = f"{arm_dir}{model}/{Path(basename).stem}.tblout"
    tblout = tblout_root / relative
    tblout.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hits = infernal.run_cmsearch(
        root / cm,
        fasta,
        tblout,
        cut_ga=False,
        cpu=cpu,
        timeout_s=timeout_s,
        toponly=False,
        no_filters=no_filters,
        score_threshold=float(threshold_bits),
        search_space_mb=SEARCH_SPACE_MB,
    )
    text = tblout.read_text(encoding="utf-8")
    argv = search_argv(
        cm=cm,
        fasta=basename,
        tblout=tblout.name,
        threshold_bits=threshold_bits,
        cpu=cpu,
        no_filters=no_filters,
    )
    mirrored = option_settings_options(" ".join(argv))
    executed = option_settings_options(option_settings_line(text))
    if executed != mirrored:
        raise HostScreenError(
            f"{tblout}: cmsearch recorded options {executed!r}, this leg's argv gives "
            f"{mirrored!r}; the report would publish a command line that did not run"
        )
    return {
        "fasta": basename,
        "fasta_sha256": provenance.sha256_file(fasta),
        "model": model,
        "cm": cm,
        "threshold_bits": float(threshold_bits),
        "no_filters": bool(no_filters),
        "arm": "filter_off" if no_filters else "screen",
        "tblout": relative,
        "argv": argv,
        # The options cmsearch itself printed, path-free: the evidence that the recorded argv is
        # the one that ran. Rendered back as a line so the report reads as a command line.
        "option_settings": " ".join(
            flag if value is True else f"{flag} {value}" for flag, value in sorted(executed.items())
        ),
        "n_hits": len(hits),
        "data_sha256": tblout_digest(text),
        "wall_s": round(time.monotonic() - started, 2),
    }


def search_line(entry: Mapping[str, Any]) -> str:
    """One search's progress line, as the search leg prints it."""
    suffix = " (--max)" if entry["no_filters"] else ""
    return (
        f"  {entry['fasta']} x {entry['model']}{suffix}: "
        f"{entry['n_hits']} hits, {entry['wall_s']}s"
    )


def run_search_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    jobs: int,
    line: Callable[[Mapping[str, Any]], str] = search_line,
) -> list[dict[str, Any]]:
    """Run :func:`_search_one` over ``tasks``, ``jobs`` at a time; results in **task** order.

    One process per (FASTA, model), ``jobs`` at a time — the ``stage2/cm_confirmer.py::search``
    shape. Results are collected in **task order**, so ``search.json`` does not depend on which
    search happened to finish first, while the progress lines print in COMPLETION order: a
    multi-hour run that said nothing until the end would be indistinguishable from a hung one.
    The whole run is abandoned on the first failure rather than letting queued searches start.

    Extracted from this module's own search leg, unchanged, so ``hardneg_pool.py`` (ADR-0005
    A17) runs A15 Pin 2's invocation through the very executor A15's screen used rather than a
    copy of it.
    """
    if jobs > 1:
        import concurrent.futures as futures

        def _progress(future: Any) -> None:
            if not future.cancelled() and future.exception() is None:
                print(line(future.result()), flush=True)

        with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            pending = [pool.submit(lambda task=task: _search_one(**task)) for task in tasks]
            for future in pending:
                future.add_done_callback(_progress)
            done, _ = futures.wait(pending, return_when=futures.FIRST_EXCEPTION)
            failed = next((f for f in pending if f in done and f.exception() is not None), None)
            if failed is not None:
                for future in pending:
                    future.cancel()
                raise failed.exception()
            return [future.result() for future in pending]
    searches: list[dict[str, Any]] = []
    for task in tasks:
        searches.append(_search_one(**task))
        print(line(searches[-1]), flush=True)
    return searches


def _cmd_search(args: argparse.Namespace) -> int:
    """LEG 2 (infernal env): every FASTA × every D2 model, at the threshold its purpose needs."""
    root = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir)
    tblout_root = Path(args.tblout_dir)
    tblout_root.mkdir(parents=True, exist_ok=True)
    requested = list(args.fasta) if args.fasta else list(SCREEN_FASTAS)
    max_arm = set(args.max_fasta or [])
    unknown = max_arm - set(requested)
    if unknown:
        raise HostScreenError(
            f"--max-fasta names FASTAs that are not being searched: {sorted(unknown)}"
        )
    tasks: list[dict[str, Any]] = []
    for basename in requested:
        if not (out_dir / basename).exists():
            raise HostScreenError(f"{out_dir / basename} does not exist; run the prep leg first")
        threshold = FASTA_THRESHOLDS.get(basename, CONTENT_THRESHOLD_BITS)
        for model, cm in SCREEN_CMS:
            for no_filters in (False, True) if basename in max_arm else (False,):
                tasks.append(
                    {
                        "root": root,
                        "out_dir": out_dir,
                        "tblout_root": tblout_root,
                        "arm_dir": "filter_off/" if no_filters else "",
                        "basename": basename,
                        "model": model,
                        "cm": cm,
                        # A15 Pin 3's orthogonal arm is the Pin 2 threshold with the filters off.
                        "threshold_bits": CONTENT_THRESHOLD_BITS if no_filters else threshold,
                        "cpu": args.cpu,
                        "no_filters": no_filters,
                        "timeout_s": args.timeout_s,
                    }
                )
    started = time.monotonic()
    searches = run_search_tasks(tasks, jobs=int(args.jobs))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": "host_screen_search",
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "infernal_version": infernal_version_line(),
        "env_lock": ENV_LOCK,
        "env_lock_sha256": provenance.sha256_file(root / ENV_LOCK),
        "cms": {
            model: {"path": cm, "sha256": refs.content_sha256(root / cm)}
            for model, cm in SCREEN_CMS
        },
        "cpu": int(args.cpu),
        "jobs": int(args.jobs),
        "searches": searches,
        "wall_s": round(time.monotonic() - started, 2),
    }
    write_json(tblout_root / "search.json", payload)
    print(f"search: {len(searches)} (fasta, model) searches in {payload['wall_s']}s")
    return 0


# ══════════════════════════════════════════════════════════════════════════════════════
# report (data env)
# ══════════════════════════════════════════════════════════════════════════════════════
def read_searches(
    tblout_root: Path,
    search: Mapping[str, Any],
    *,
    arm: str = "screen",
    fasta_digests: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], dict[str, list[Hit]]]:
    """``{(fasta, model) -> {target: hits}}``, each tblout bound to what the search recorded.

    Three bindings, each closing a way a hit set could be read as something it is not:

    * the tblout's **data rows** must still hash to ``data_sha256`` — the bit scores in the
      report have to be the ones the recorded argv produced;
    * the tblout's **query names** must be the CM's own ``NAME`` for the model the search claims
      (see :class:`Hit`);
    * when ``fasta_digests`` is given (the prep leg's own record), the **searched FASTA** must be
      the one that prep wrote. A tblout searched from a different FASTA reads as "no hits" for
      every window, which admits the whole pool; the surrogate names are positional, so they
      would collide rather than surface as unknown targets.
    """
    out: dict[tuple[str, str], dict[str, list[Hit]]] = {}
    for entry in search["searches"]:
        if entry.get("arm", "screen") != arm:
            continue
        basename = entry["fasta"]
        if fasta_digests is not None:
            expected = fasta_digests.get(basename)
            recorded = entry.get("fasta_sha256")
            if expected is None or recorded != expected:
                raise HostScreenError(
                    f"{basename}: the search ran on a FASTA with sha256 {recorded!r}, the prep "
                    f"leg wrote {expected!r}. Refusing to pair one run's windows with another "
                    "run's tblouts — unsearched windows would read as admitted."
                )
        path = tblout_root / entry["tblout"]
        text = path.read_text(encoding="utf-8")
        if tblout_digest(text) != entry["data_sha256"]:
            raise HostScreenError(f"{path}: data rows differ from what the search recorded")
        out[(basename, entry["model"])] = hits_by_target(text, model=entry["model"])
    return out


def _bits_per_record(
    names: Iterable[str],
    grouped: Mapping[tuple[str, str], dict[str, list[Hit]]],
    fasta: str,
    models: Sequence[str],
) -> dict[str, dict[str, float | None]]:
    """Per FASTA record: the best bit score **per model key**, over that FASTA's own tblouts.

    Keyed by the search that produced each tblout, never by the tblout's query-name column
    (see :class:`Hit`).
    """
    missing = [model for model in models if (fasta, model) not in grouped]
    if missing:
        raise HostScreenError(f"{fasta} was not searched with {missing}; run the search leg")
    return {
        name: best_bits({model: grouped[(fasta, model)].get(name, []) for model in models})
        for name in names
    }


def _cmd_report(args: argparse.Namespace) -> int:
    """LEG 3 (data env): Pin 2's verdicts, the controls, the panel, the v1 audit, the report."""
    started = time.monotonic()
    root = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir)
    tblout_root = Path(args.tblout_dir)
    prep = json.loads((out_dir / "prep.json").read_text(encoding="utf-8"))
    idmap = json.loads((out_dir / "idmap.json").read_text(encoding="utf-8"))
    search = json.loads((tblout_root / "search.json").read_text(encoding="utf-8"))
    grouped = read_searches(
        tblout_root,
        search,
        fasta_digests={
            basename: record["sha256"] for basename, record in (prep.get("fasta") or {}).items()
        },
    )
    models = _model_names()
    threshold = CONTENT_THRESHOLD_BITS

    # ── Pin 2 on the survivors ──────────────────────────────────────────────────────────
    survivors = idmap["survivors"]
    survivor_bits = _bits_per_record(survivors, grouped, FASTA_SURVIVORS, models)
    admitted: list[str] = []
    refused_annex: list[dict[str, Any]] = []
    n_refused_cm = {model: 0 for model in models}
    n_refused_either = n_refused_motif = n_refused_motif_only = 0
    for name, candidate_id in sorted(survivors.items()):
        bits = survivor_bits[name]
        record = prep["candidates"][candidate_id]
        motif = bool(record["motif"])
        per_model = {model: at_least(bits.get(model), threshold) for model in models}
        either = any(per_model.values())
        for model in models:
            n_refused_cm[model] += per_model[model]
        n_refused_either += either
        n_refused_motif += motif
        n_refused_motif_only += motif and not either
        if content_refused(bits, motif, threshold):
            refused_annex.append(
                {
                    "candidate_id": candidate_id,
                    "rules": ["content"],
                    "sources": [],
                    "bits": {model: bits.get(model) for model in models},
                    "motif": motif,
                    "side": record["side"],
                }
            )
        else:
            admitted.append(candidate_id)
    for candidate_id, record in sorted(prep["candidates"].items()):
        if not record["refused"]:
            continue
        refused_annex.append(
            {
                "candidate_id": candidate_id,
                "rules": [
                    label
                    for flag, label in (
                        (record["rule_kmer"], "kmer"),
                        (record["rule_coord"], "coord"),
                    )
                    if flag
                ],
                "sources": sorted({*record["kmer_sources"], *record["coord_sources"]}),
                "bits": None,
                "motif": record["motif"],
                "side": record["side"],
            }
        )
    refused_annex.sort(key=lambda entry: entry["candidate_id"])
    admitted.sort()

    # ── the controls: Pin 1's (measured at prep) plus Pin 2's two ───────────────────────
    controls = list(prep["controls"])
    control_bits = _bits_per_record(idmap["controls"], grouped, FASTA_CONTROLS, models)
    for meaning, expected in ((CONTROL_CONTENT, True), (CONTROL_CONTENT_HOST, False)):
        meta = (prep.get("control_meta") or {}).get(meaning)
        if meta is None:
            controls.append(
                control_entry(meaning, "content screen", {"present": True}, {"present": False})
            )
            continue
        bits = control_bits[meta["surrogate"]]
        observed = content_refused(bits, meta["motif"], threshold)
        controls.append(
            control_entry(
                meaning,
                (
                    "a real catalogued RF00230 T-box spliced into a Pin-1-clean survivor must be "
                    "refused by Pin 2"
                    if expected
                    else "the same host, unspliced, must be admitted — the matched control that "
                    "makes the spliced refusal attributable to the insert"
                ),
                expected,
                observed,
                note=f"best bits {bits!r}, motif={bool(meta['motif'])}",
            )
        )
    controls_all_fired = all(entry["fired"] for entry in controls)

    # ── the sensitivity panel and the v1 audit ──────────────────────────────────────────
    panel_bits = _bits_per_record(idmap["panel"], grouped, FASTA_PANEL, models)
    panel = _panel_block(idmap["panel"], panel_bits, models, threshold)
    v1_bits = _bits_per_record(idmap["v1_hosts"], grouped, FASTA_V1_HOSTS, models)
    v1_audit = _v1_audit_block(prep["v1_hosts"], v1_bits, models, threshold)
    panel["v1_host_hits"] = _v1_host_hit_panel(v1_audit, models, threshold)

    # ── composition (A15 Pin 7(e)) ──────────────────────────────────────────────────────
    admitted_set = set(admitted)
    composition = {
        "admitted": _composition(
            [record for cid, record in prep["candidates"].items() if cid in admitted_set]
        ),
        "refused": _composition(
            [record for cid, record in prep["candidates"].items() if cid not in admitted_set]
        ),
        "note": (
            "A15 Pin 7(e): Pin 1 refuses trail windows more often than lead windows, so the "
            "screened draw shifts toward lead windows and away from training-like genomes"
        ),
    }

    admitted_text = "\n".join(admitted) + "\n"
    admitted_path = Path(args.admitted)
    admitted_path.parent.mkdir(parents=True, exist_ok=True)
    admitted_path.write_text(admitted_text, encoding="utf-8")
    annex_path = Path(args.refused)
    write_json(
        annex_path,
        {
            "schema_version": SCHEMA_VERSION,
            "step": STEP,
            "kind": "host_screen_refused_annex",
            "adr": ADR,
            "threshold_bits": threshold,
            "motif": MOTIF,
            "n": len(refused_annex),
            "note": (
                "A15 Pin 2's lower-bound clause: a later hit on a refused window can be "
                "adjudicated from this annex rather than counted as a false positive"
            ),
            "refused": refused_annex,
        },
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": KIND_SCREEN,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "prd": PRD,
        "pool": prep["pool"],
        "screened_dna": prep["screened_dna"],
        "fold_screen": prep["fold_screen"],
        "content_screen": {
            "threshold_bits": threshold,
            "motif": MOTIF,
            "models": {
                model: {"cm": record["path"], "cm_sha256": record["sha256"]}
                for model, record in search["cms"].items()
            },
            "argv": {
                f"{entry['fasta']}:{entry['model']}": entry["argv"]
                for entry in search["searches"]
                if entry.get("arm", "screen") == "screen"
            },
            "option_settings": {
                f"{entry['fasta']}:{entry['model']}": entry["option_settings"]
                for entry in search["searches"]
                if entry.get("arm", "screen") == "screen"
            },
            "infernal_version": search["infernal_version"],
            "env_lock": search["env_lock"],
            "env_lock_sha256": search["env_lock_sha256"],
            "cpu": search.get("cpu"),
            "n_searched": len(survivors),
            "n_refused_cm": {
                **{model: int(n_refused_cm[model]) for model in models},
                "either": int(n_refused_either),
                **{
                    f"marginal_{model}": int(n_refused_either - n_refused_cm[_other_model(model)])
                    for model in models
                },
            },
            "n_refused_motif": int(n_refused_motif),
            "n_refused_motif_only": int(n_refused_motif_only),
            "n_refused": int(n_refused_either + n_refused_motif_only),
            "n_admitted": len(admitted),
            "lower_bound_clause": (
                "A15 Pin 2: the refusal count lower-bounds T-box content in the pool. Admission "
                "means below 10 bits on RF00230/TBDB001 and motif-free, never T-box-free."
            ),
        },
        "admitted": {
            "n": len(admitted),
            "list": recorded_path(admitted_path, root),
            "sha256": hashlib.sha256(admitted_text.encode("utf-8")).hexdigest(),
        },
        "refused_annex": {
            "path": recorded_path(annex_path, root),
            "sha256": provenance.sha256_file(annex_path),
            "n": len(refused_annex),
        },
        "controls": controls,
        "controls_all_fired": controls_all_fired,
        "content_control": prep["content_control"],
        "sensitivity_panel": panel,
        "v1_audit": v1_audit,
        "composition": composition,
        "disclosures": disclosures(),
        "out_dir_inside_data_root": prep.get("out_dir_inside_data_root"),
        "wall_s": round(time.monotonic() - started, 2),
    }
    report["provenance"] = provenance_record(
        leg="report",
        inputs=[out_dir / "prep.json", out_dir / "idmap.json", tblout_root / "search.json"],
        root=root,
        env_lock=root / ENV_LOCK_DATA,
        extra={"out_dir": recorded_path(out_dir, root)},
    )
    problems = validate_report(report)
    problems += _written_file_problems(report, admitted_path, annex_path)
    report["problems"] = problems
    out = Path(args.out)
    if problems:
        invalid = out.with_suffix(".invalid.json")
        write_json(invalid, report)
        out.unlink(missing_ok=True)
        print(f"REFUSED: {len(problems)} problem(s); wrote {invalid}")
        for problem in problems:
            print(f"  - {problem}")
        return 3
    write_json(out, report)
    print(
        f"wrote {out}: {report['pool']['n_candidates']} candidates -> "
        f"{report['fold_screen']['n_survivors']} Pin-1 survivors -> {len(admitted)} admitted "
        f"(Pin 2 refused {report['content_screen']['n_refused']}: {n_refused_either} CM, "
        f"{n_refused_motif_only} motif-only); controls_all_fired={controls_all_fired}; "
        f"{report['wall_s']}s"
    )
    return 0


def _written_file_problems(
    report: Mapping[str, Any], admitted_path: Path, annex_path: Path
) -> list[str]:
    """Bind the report to the two files the leg just wrote (sha256, count, order)."""
    problems: list[str] = []
    declared = _mapping(report, "admitted")
    try:
        text = admitted_path.read_text(encoding="utf-8")
    except OSError as error:
        return [f"admitted list unreadable: {error}"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != declared.get("sha256"):
        problems.append("admitted.sha256 does not match the written list")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != declared.get("n"):
        problems.append(
            f"the admitted list holds {len(lines)} ids, admitted.n says {declared.get('n')!r}"
        )
    if lines != sorted(lines):
        problems.append("the admitted list is not sorted")
    if len(set(lines)) != len(lines):
        problems.append("the admitted list repeats a candidate_id")
    if not text.endswith("\n"):
        problems.append("the admitted list has no trailing newline")
    try:
        annex_digest = provenance.sha256_file(annex_path)
    except OSError as error:
        return [*problems, f"refused annex unreadable: {error}"]
    if annex_digest != _mapping(report, "refused_annex").get("sha256"):
        problems.append("refused_annex.sha256 does not match the written annex")
    return problems


def _panel_block(
    names: Mapping[str, Mapping[str, Any]],
    bits: Mapping[str, Mapping[str, float | None]],
    models: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """A15 Pin 2's sensitivity panel: per variant, what each mechanism catches.

    Reported, **not** used to set the 10-bit value (A15 Pin 2 pins that explicitly). The
    per-mechanism rows are the informative ones: ``either`` on a positive window is the screen's
    recall, and the motif row is the measured sensitivity of the CM-free backstop, which A15
    calls a backstop rather than a sensitive test.
    """
    by_variant: dict[str, list[str]] = {}
    for name, meta in names.items():
        by_variant.setdefault(str(meta["variant"]), []).append(name)
    out: dict[str, Any] = {
        "threshold_bits": threshold,
        "note": (
            "A15 Pin 2: reported, not used to set the threshold. The screen's recall on real "
            "T-box windows is a sensitivity measurement, not a gate."
        ),
        "variants": {},
    }
    for variant, members in sorted(by_variant.items()):
        n = len(members)
        cells: dict[str, Any] = {}
        for model in models:
            hit = sum(1 for name in members if at_least(bits[name].get(model), threshold))
            cells[model] = {"n_hits": hit, "recall": (hit / n) if n else None}
        motif = sum(1 for name in members if bool(names[name]["motif"]))
        either = sum(
            1 for name in members if content_refused(bits[name], names[name]["motif"], threshold)
        )
        cells["motif"] = {"n_hits": motif, "recall": (motif / n) if n else None}
        cells["either"] = {"n_hits": either, "recall": (either / n) if n else None}
        out["variants"][variant] = {
            "n": n,
            "mean_length": (
                round(sum(int(names[name]["length"]) for name in members) / n, 1) if n else None
            ),
            **cells,
        }
    return out


def _v1_audit_block(
    hosts: Mapping[str, Mapping[str, Any]],
    bits_by_name: Mapping[str, Mapping[str, float | None]],
    models: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """A15's ``v1_audit``: what the v1 draw's 692 hosts carry, re-derived by this step.

    The measurement that triggered the amendment, committed beside the screen it motivated
    rather than left in the ADR's prose. The thresholds are RF00230 bit scores from the audit
    search (``-T 0 -Z 1``, default pipeline, both strands) — the same invocation the screen uses
    with a lower reporting threshold, so every host has a best score rather than a floor.
    """
    entries: dict[str, dict[str, Any]] = {}
    counts = {
        "n_hosts": len(hosts),
        "n_ge_93": 0,
        "n_ge_20": 0,
        "n_ge_10": 0,
        "n_motif": 0,
        "n_refused_fold": 0,
        "n_refused_content": 0,
        "n_refused": 0,
    }
    per_model_ge_threshold = {model: 0 for model in models}
    for host_id, record in sorted(hosts.items()):
        bits = bits_by_name[record["surrogate"]]
        values = [
            value for value in (_as_bits(bits.get(model)) for model in models) if value is not None
        ]
        rf = _as_bits(bits.get(models[0]))
        motif = bool(record["motif"])
        content_flag = content_refused(bits, motif, threshold)
        fold_flag = bool(record["refused"])
        entries[host_id] = {
            "host_id": host_id,
            "contig_id": record["contig_id"],
            "pool": record["pool"],
            "side": record["side"],
            **{axis: record[axis] for axis in COMPOSITION_AXES[1:]},
            "bits": {model: bits.get(model) for model in models},
            "best_bits": max(values) if values else None,
            "motif": motif,
            "rule_kmer": bool(record["rule_kmer"]),
            "rule_coord": bool(record["rule_coord"]),
            "kmer_sources": record["kmer_sources"],
            "refused_fold": fold_flag,
            "refused_content": content_flag,
            "refused": fold_flag or content_flag,
        }
        counts["n_ge_93"] += at_least(rf, 93.0)
        counts["n_ge_20"] += at_least(rf, 20.0)
        counts["n_ge_10"] += at_least(rf, 10.0)
        counts["n_motif"] += motif
        counts["n_refused_fold"] += fold_flag
        counts["n_refused_content"] += content_flag
        counts["n_refused"] += fold_flag or content_flag
        for model in models:
            per_model_ge_threshold[model] += at_least(bits.get(model), threshold)
    return {
        "note": (
            "re-derived by this step over the hosts the deterministic seed-42 v1 draw used; the "
            "bit thresholds are the operative model's (the first of SCREEN_CMS)"
        ),
        "threshold_bits": threshold,
        "bit_model": models[0],
        "counts": {**counts, "n_ge_threshold_by_model": per_model_ge_threshold},
        "composition": _composition(list(entries.values())),
        "hosts": entries,
    }


def _v1_host_hit_panel(
    v1_audit: Mapping[str, Any], models: Sequence[str], threshold: float
) -> dict[str, Any]:
    """The panel's third row: what each mechanism catches among the v1 hosts ≥ 10 bits.

    ``either`` is 1.0 **by construction** on this set (membership is an RF00230 hit at the same
    threshold), and saying so is the point: the informative numbers are the motif's and
    TBDB001's recall on windows the operative model already flags.
    """
    bit_model = models[0]
    members = [
        entry
        for entry in v1_audit["hosts"].values()
        if at_least(entry["bits"].get(bit_model), threshold)
    ]
    n = len(members)
    out: dict[str, Any] = {
        "n": n,
        "definition": f"v1 hosts with a {bit_model} hit >= {threshold} bits",
        "either_is_tautological": True,
    }
    for model in models:
        hit = sum(1 for entry in members if at_least(entry["bits"].get(model), threshold))
        out[model] = {"n_hits": hit, "recall": (hit / n) if n else None}
    motif = sum(1 for entry in members if bool(entry["motif"]))
    out["motif"] = {"n_hits": motif, "recall": (motif / n) if n else None}
    for cut in (20.0, 93.0):
        out[f"n_ge_{int(cut)}"] = sum(
            1 for entry in members if at_least(entry["bits"].get(bit_model), cut)
        )
    return out


def _composition(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts over a set of windows on every :data:`COMPOSITION_AXES` axis (A15 Pin 7(e)).

    ``side`` plus the parent's ``nested_role``, leave-clade-out designation and resolved
    order: the axes the v2 manifest's ``scope.host_pool.composition`` carries, so v1's table
    and v2's are comparable axis by axis. A record lacking an axis is ``"unresolved"``.
    """
    tables: dict[str, dict[str, int]] = {axis: {} for axis in COMPOSITION_AXES}
    for record in records:
        for axis in COMPOSITION_AXES:
            key = str(record.get(axis, "null" if axis == "side" else "unresolved"))
            tables[axis][key] = tables[axis].get(key, 0) + 1
    return {"n": len(records), **{axis: dict(sorted(t.items())) for axis, t in tables.items()}}


def disclosures() -> list[dict[str, str]]:
    """A15 Pin 2 / Pin 7(a)'s disclosures, carried by the artifact rather than by a reader."""
    return [
        {
            "id": "lower_bound",
            "text": (
                "A passing screen is not a claim that the pool is clean: the refusal count "
                "lower-bounds T-box content, and sub-threshold or CM-invisible content is "
                "invisible to both Pin 1 and Pin 2."
            ),
        },
        {
            "id": "cm_circularity",
            "text": (
                "After Pin 2 a gated negative means 'below 10 bits on RF00230/TBDB001 and "
                "motif-free', so a CM contestant is favoured by construction (A15 Pin 7(a)): a "
                "learned model that finds a real but CM-invisible T-box in a gated host is "
                "scored as a false positive."
            ),
        },
        {
            "id": "threshold_choice",
            "text": (
                "The 10-bit value was chosen after a v1 audit that also knew which hosts drew "
                "Stage-1 candidates. That knowledge is disclosed; the value is content-based, "
                "sitting below every v1 host hit traced to an element's own CM extent."
            ),
        },
        {
            "id": "motif_sensitivity",
            "text": (
                "The motif is a backstop, not a sensitive test: an exact GGTGGNACCGCG match "
                "misses real T-boxes, and the sensitivity panel reports how often."
            ),
        },
        {
            "id": "declared_redundancy",
            "text": (
                "TBDB001 carries no GA line and its bit scale is not RF00230's; it is kept as "
                "declared redundancy, and the report carries its measured marginal."
            ),
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════════════
# familiarity (data env) — A15 Pin 7(b)
# ══════════════════════════════════════════════════════════════════════════════════════
def decoy_embedded(row: Mapping[str, Any], fold: Iterable[str]) -> bool:
    """ADR-0005 A7's embedding rule for one decoy row against one arm's training fold.

    The minter's ``decoy_embedded``, re-expressed without pandas: ``masking.row_text`` handles
    both pandas dialects' spellings of missing, where ``str(x or "")`` does not
    ([[pandas-3-nan-truthy-in-training-env]]). The familiarity leg refuses unless the per-pool
    counts reproduce :data:`EXPECTED_EMBEDDED`, so the two implementations cannot drift apart
    silently.
    """
    if masking.row_text(row.get("pool")) not in emb.TRAINING_DECOY_POOLS:
        return False
    if bool(row.get("masked")):
        return False
    parent = masking.row_text(row.get("source_record_id")).strip()
    return parent == "" or parent in set(fold)


def _arm_training_dna(
    *,
    root: Path,
    arm: str,
    context: str,
    labels: str,
    split_table: str,
    mined_sequences: Sequence[str],
    decoy_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """One arm's Stage-1 training DNA: its fold contexts ∪ the mined windows ∪ its own inserts."""
    fold = stage1_folds.training_fold_ids(
        arm=arm,
        split_table=str(root / split_table),
        context=str(root / context),
        labels=str(root / labels),
    )
    context_frame = _read_parquet(root / context, ["record_id", "context_seq"])
    fold_sequences = [
        str(value)
        for record, value in zip(
            context_frame["record_id"], context_frame["context_seq"], strict=True
        )
        if str(record) in fold
    ]
    embedded = [row for row in decoy_rows if decoy_embedded(row, fold)]
    by_pool: dict[str, int] = {}
    for row in embedded:
        pool = masking.row_text(row.get("pool"))
        by_pool[pool] = by_pool.get(pool, 0) + 1
    if by_pool != EXPECTED_EMBEDDED[arm]:
        raise HostScreenError(
            f"the {arm} arm's embedding derivation gives {by_pool!r}, the committed run recorded "
            f"{EXPECTED_EMBEDDED[arm]!r}. Refusing to publish a familiarity score built on the "
            "wrong insert set."
        )
    kmers = sorted_unique_kmers(
        [*fold_sequences, *mined_sequences, *(str(row["sequence"]) for row in embedded)]
    )
    return kmers, {
        "n_fold_records": len(fold_sequences),
        "n_mined": len(mined_sequences),
        "n_inserts": len(embedded),
        "n_inserts_by_pool": dict(sorted(by_pool.items())),
        "n_kmers": int(kmers.size),
    }


def _cmd_familiarity(args: argparse.Namespace) -> int:
    """LEG 4 (data env): A15 Pin 7(b)'s familiarity baseline score, per item per arm.

    The score is deliberately **not** a model: it is the fraction of an item's 32-mers that the
    arm's Stage-1 training DNA already carries. A15 publishes it beside Stage-1-only and
    two-stage because every gated negative host is 32-mer-disjoint from training DNA **by
    construction** while the positives' flanks mostly are not — so if the baseline rivals
    Stage-1, the gate is confounded by construction and the report must say so.
    """
    started = time.monotonic()
    for name in ("mining_pool", "decoys", "context", "labels", "split_table"):
        require_repo_relative(getattr(args, name), what=f"--{name.replace('_', '-')}")
    root = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]
    from tbox_finder.data import negatives as negatives_mod

    benchmark_path = Path(args.benchmark)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    mined_rows, _ = negatives_mod.load_admitted_pool_rows(str(root / args.mining_pool))
    mined_sequences = [str(row["sequence"]) for row in mined_rows]
    decoy_rows = _read_parquet(
        root / args.decoys, ["decoy_id", "pool", "sequence", "masked", "source_record_id"]
    ).to_dict("records")
    arms: dict[str, Any] = {}
    kmers: dict[str, np.ndarray] = {}
    for arm in sorted(stage1_folds.ARM_EXCLUDE_GATE4_EVAL):
        kmers[arm], arms[arm] = _arm_training_dna(
            root=root,
            arm=arm,
            context=args.context,
            labels=args.labels,
            split_table=args.split_table,
            mined_sequences=mined_sequences,
            decoy_rows=decoy_rows,
        )
    items: dict[str, dict[str, float]] = {}
    for contig in benchmark["contigs"]:
        contig_id = str(contig["contig_id"])
        scores: dict[str, float] = {}
        for arm, index in kmers.items():
            value = familiarity_fraction(contig["sequence"], index)
            if value is None:
                raise HostScreenError(
                    f"contig {contig_id!r} admits no N-free {KMER_K}-mer, so its familiarity is "
                    "undefined; a 0.0 there would publish 'unfamiliar' for an unmeasured window"
                )
            scores[arm] = value
        items[contig_id] = scores
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": "familiarity_baseline",
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "benchmark": {
            "path": recorded_path(benchmark_path, root),
            "sha256": provenance.sha256_file(benchmark_path),
            "schema_version": benchmark.get("schema_version"),
            "n_contigs": len(benchmark["contigs"]),
        },
        "k": KMER_K,
        "score": (
            "the fraction of the contig's canonical N-free 32-mers present in that arm's "
            "Stage-1 training DNA: its own fold context_seq, the admitted mined windows, and "
            "the decoy inserts that arm embedded (ADR-0005 A7)"
        ),
        "arms": arms,
        "items": items,
        "wall_s": round(time.monotonic() - started, 2),
    }
    payload["provenance"] = provenance_record(
        leg="familiarity",
        inputs=[benchmark_path, root / args.mining_pool, root / args.decoys, root / args.context],
        root=root,
        env_lock=root / ENV_LOCK_DATA,
    )
    write_json(args.out, payload)
    summary = ", ".join(f"{arm}: {block['n_kmers']}" for arm, block in sorted(arms.items()))
    print(
        f"wrote {args.out}: {len(items)} items x {len(arms)} arms; n_kmers {{{summary}}}; "
        f"{payload['wall_s']}s"
    )
    return 0


# ══════════════════════════════════════════════════════════════════════════════════════
# the post-mint assertion (A15 Pin 3)
# ══════════════════════════════════════════════════════════════════════════════════════
def leader_exempt_interval(record: Mapping[str, Any]) -> tuple[int, int] | None:
    """``[splice_phase, splice_phase + locus_length)`` for a leader item, else ``None``.

    A15 Pin 3 exempts hits **overlapping** the leader item's own insert: all 8 ``leader_decoy``
    records are T-box-derived (A7 Pin 5), so a hit there is the insert being itself rather than
    the screen failing. As signed, the exemption read "wholly inside"; the 2026-10-11 correction
    widened it to overlap, because a CM aligning the leader's own element extends past the splice
    point into flanking DNA by construction. Nothing else is exempt, and a leader row that cannot
    state its own insert interval gets no exemption.
    """
    if str(record.get("pool")) != "leader_decoy":
        return None
    phase, length = record.get("splice_phase"), record.get("locus_length")
    if isinstance(phase, bool) or not isinstance(phase, int):
        return None
    if isinstance(length, bool) or not isinstance(length, int):
        return None
    return int(phase), int(phase) + int(length)


def _cmd_postmint_prep(args: argparse.Namespace) -> int:
    """A15 Pin 3: write the minted v2 contigs as surrogate FASTAs for the assertion search.

    Two FASTAs, because the two arms do not cost the same. The Pin 2 arm searches **every**
    contig (A15: "inserts are measured too"). The orthogonal filter-off arm (``--max``, which D2
    requires because the HMM pre-filter drops divergent hits) searches the **label-0** contigs,
    the only ones the assertion is about: running it over 1,201 positives that legitimately
    contain T-boxes would cost hours to measure nothing the assertion reads.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    items = {str(item["contig_id"]): item for item in benchmark["items"]}
    names: dict[str, dict[str, Any]] = {}
    all_sequences: dict[str, str] = {}
    label0_sequences: dict[str, str] = {}
    for index, contig in enumerate(
        sorted(benchmark["contigs"], key=lambda contig: str(contig["contig_id"])), start=1
    ):
        contig_id = str(contig["contig_id"])
        item = items.get(contig_id)
        if item is None:
            raise HostScreenError(f"contig {contig_id!r} has no item row in {args.benchmark}")
        name = surrogate("v", index)
        names[name] = {
            "contig_id": contig_id,
            "label": int(item["label"]),
            "pool": item.get("pool"),
            "gated": item.get("gated"),
            "splice_phase": item.get("splice_phase"),
            "locus_length": item.get("locus_length"),
            "motif": motif_present(contig["sequence"]),
            # Positions, not just presence: the leader exemption is an interval test, so a
            # motif match has to state where it is (A15 Pin 3, corrected 2026-10-11).
            "motif_spans": [list(span) for span in motif_spans(contig["sequence"])],
        }
        all_sequences[name] = normalise_dna(contig["sequence"])
        if int(item["label"]) == 0:
            label0_sequences[name] = all_sequences[name]
    if not label0_sequences:
        raise HostScreenError(
            f"{args.benchmark} holds no label-0 contig; there is nothing to assert"
        )
    infernal.write_fasta(all_sequences, out_dir / FASTA_POSTMINT_ALL)
    infernal.write_fasta(label0_sequences, out_dir / FASTA_POSTMINT_LABEL0)
    write_json(
        out_dir / "postmint_idmap.json",
        {
            "schema_version": SCHEMA_VERSION,
            "step": STEP,
            "kind": "postmint_idmap",
            "benchmark": {
                "path": str(args.benchmark),
                "sha256": provenance.sha256_file(args.benchmark),
                "schema_version": benchmark.get("schema_version"),
            },
            "contigs": names,
            "n_all": len(all_sequences),
            "n_label0": len(label0_sequences),
        },
    )
    print(
        f"postmint-prep: {len(all_sequences)} contigs ({len(label0_sequences)} label-0) -> "
        f"{out_dir / FASTA_POSTMINT_ALL}, {out_dir / FASTA_POSTMINT_LABEL0}"
    )
    return 0


def postmint_findings(
    contigs: Mapping[str, Mapping[str, Any]],
    searches: Sequence[Mapping[str, Any]],
    tblouts: Mapping[str, str],
    *,
    thresholds: Mapping[str, float] | None = None,
    fasta_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """A15 Pin 3's assertion, as a pure function over the parsed tblouts.

    Separated from the CLI leg so the unit tier can drive it on real tblout text: the exemption
    arithmetic (a hit **overlapping** the leader's insert) is the one place a half-open interval
    meets Infernal's 1-based inclusive coordinates, and it is where an off-by-one would exempt a
    hit that touches neither the insert nor its junctions.

    The verdict is **per model** (``POSTMINT_THRESHOLD_BITS``), while the searches still report
    from ``CONTENT_THRESHOLD_BITS``. The gap is deliberate: a label-0 hit above the search floor
    but below its model's assertion threshold is returned as ``noise_residue`` rather than
    dropped, so the artifact discloses what the re-signed threshold now tolerates instead of
    hiding it behind a ``-T`` the reader cannot see.
    """
    violations: list[dict[str, Any]] = []
    exempt: list[dict[str, Any]] = []
    positives: list[dict[str, Any]] = []
    residue: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    for entry in searches:
        arm = str(entry.get("arm", "screen"))
        text = tblouts[entry["tblout"]]
        if tblout_digest(text) != entry["data_sha256"]:
            raise HostScreenError(f"{entry['tblout']}: data rows differ from the search record")
        if fasta_digests is not None:
            # The sibling guard `read_searches` already applies on the screen path; without it
            # here, a prep that rewrote the contigs while old tblouts survived would pair one
            # run's windows with another run's hits. The surrogate ids are POSITIONAL, so they
            # collide silently rather than surfacing as unknown targets, and an unsearched
            # contig reads as "no hits" — the fail-open direction, which passes the assertion.
            expected = fasta_digests.get(entry["fasta"])
            if expected is None or entry.get("fasta_sha256") != expected:
                raise HostScreenError(
                    f"{entry['fasta']}: the search ran on a FASTA with sha256 "
                    f"{entry.get('fasta_sha256')!r}, the prep leg wrote {expected!r}. Refusing "
                    "to grade one run's contigs against another run's tblouts."
                )
        block = arms.setdefault(arm, {"argv": {}, "option_settings": {}, "n_hits": 0})
        key = f"{entry['fasta']}:{entry['model']}"
        block["argv"][key] = entry["argv"]
        block["option_settings"][key] = entry["option_settings"]
        hits = parse_tblout_hits(text)
        check_query_names(hits, entry["model"])
        for hit in hits:
            record = contigs.get(hit.target)
            if record is None:
                raise HostScreenError(
                    f"{entry['tblout']}: hit for {hit.target!r}, which is not a v2 contig"
                )
            block["n_hits"] += 1
            payload = {
                "arm": arm,
                # Which FASTA this hit came from: a contig that is in both the all-contig and
                # the label-0 FASTA is searched by the screen arm twice, so a reader counting
                # rows has to be able to see that the two are one finding.
                "fasta": entry["fasta"],
                "contig_id": record["contig_id"],
                "pool": record.get("pool"),
                "label": int(record["label"]),
                "model": entry["model"],
                "bits": hit.score,
                "span": [hit.seq_from, hit.seq_to],
                "strand": hit.strand,
            }
            if int(record["label"]) != 0:
                positives.append(payload)
                continue
            limit = postmint_threshold(entry["model"], thresholds)
            if hit.score < limit:
                # Sub-threshold is tested BEFORE the leader exemption so the two dispositions
                # stay independent: a hit below its model's threshold is not a violation for
                # anyone, exempt or not, and routing it through the exemption first would make
                # `n_exempt` depend on the threshold rather than on the interval arithmetic.
                payload["assertion_threshold_bits"] = limit
                residue.append(payload)
                continue
            interval = leader_exempt_interval(record)
            lo, hi = min(hit.seq_from, hit.seq_to), max(hit.seq_from, hit.seq_to)
            # tblout coordinates are 1-based inclusive; the insert interval is 0-based
            # half-open, so the hit occupies [lo-1, hi). A15 Pin 3 as signed exempted a hit
            # "wholly inside" that interval; the 2026-10-11 correction exempts a hit that
            # OVERLAPS it, because a CM aligning the leader's own element runs past the splice
            # point into flanking DNA by construction (measured: 162 nt upstream on RF00230).
            # Overlap, never pool membership: a leader hit disjoint from its own insert is
            # still a violation, and a test pins exactly that.
            if interval is not None and spans_overlap((lo - 1, hi), interval):
                payload["exempt_interval"] = list(interval)
                exempt.append(payload)
                continue
            violations.append(payload)
    for record in contigs.values():
        if int(record["label"]) != 0 or not bool(record["motif"]):
            continue
        interval = leader_exempt_interval(record)
        spans = record.get("motif_spans") or []
        if (
            interval is not None
            and spans
            and all(spans_overlap((int(lo), int(hi)), interval) for lo, hi in spans)
        ):
            exempt.append(
                {
                    "arm": "motif",
                    "fasta": None,
                    "contig_id": record["contig_id"],
                    "pool": record.get("pool"),
                    "label": int(record["label"]),
                    "model": "motif",
                    "bits": None,
                    "span": [int(spans[0][0]) + 1, int(spans[0][1])],
                    "strand": None,
                    "exempt_interval": list(interval),
                }
            )
            continue
        violations.append(
            {
                "arm": "motif",
                # The motif is read off the contig itself, not out of a tblout, so it has no
                # source FASTA — and it is recorded once per contig rather than once per
                # (FASTA, model) search, which is why the field is explicitly null.
                "fasta": None,
                "contig_id": record["contig_id"],
                "pool": record.get("pool"),
                "label": 0,
                "model": "motif",
                "bits": None,
                "span": [int(spans[0][0]) + 1, int(spans[0][1])] if spans else None,
                "strand": None,
            }
        )
    violations.sort(key=lambda entry: (str(entry["contig_id"]), str(entry["model"])))
    residue.sort(key=lambda entry: (str(entry["contig_id"]), str(entry["model"])))
    return {
        "arms": arms,
        "violations": violations,
        "exempt": exempt,
        "positive_hits": positives,
        "noise_residue": residue,
    }


def _calibration_record(path: Path, root: Path) -> dict[str, Any]:
    """The noise-floor artifact A15 Pin 3's thresholds are calibrated against.

    Recorded by path + digest + the measured per-model maxima, so the re-signed thresholds bind
    to the evidence that set them instead of floating free of it
    ([[gate-must-bind-to-upstream-evidence]]). ``_postmint_problems`` re-derives that every
    pinned threshold clears the floor this artifact measured, which is what stops a later edit
    to ``POSTMINT_THRESHOLD_BITS`` from quietly re-entering the noise.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    kind = report.get("kind")
    if kind != KIND_NOISE_FLOOR:
        raise HostScreenError(f"{path}: kind is {kind!r}, not {KIND_NOISE_FLOOR!r}")
    problems = report.get("problems")
    if problems:
        raise HostScreenError(
            f"{path}: the calibration artifact carries {len(problems)} problem(s); "
            "a threshold is not calibrated against evidence that refuses itself"
        )
    return {
        "path": recorded_path(path, root),
        "sha256": provenance.sha256_file(path),
        "schema_version": report.get("schema_version"),
        "null_max_bits": report.get("null_max_bits"),
        "n_null_sequences": report.get("n_null_sequences"),
    }


def _cmd_postmint_report(args: argparse.Namespace) -> int:
    """A15 Pin 3's assertion, per model: 0 label-0 hits at or above the model's re-signed
    threshold (``POSTMINT_THRESHOLD_BITS``) and 0 motif matches, bar the leader's own insert."""
    started = time.monotonic()
    root = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir)
    tblout_root = Path(args.tblout_dir)
    idmap = json.loads((out_dir / "postmint_idmap.json").read_text(encoding="utf-8"))
    search = json.loads((tblout_root / "search.json").read_text(encoding="utf-8"))
    tblouts = {
        entry["tblout"]: (tblout_root / entry["tblout"]).read_text(encoding="utf-8")
        for entry in search["searches"]
    }
    findings = postmint_findings(
        idmap["contigs"],
        search["searches"],
        tblouts,
        fasta_digests={
            name: provenance.sha256_file(out_dir / name)
            for name in sorted({str(entry["fasta"]) for entry in search["searches"]})
        },
    )
    report: dict[str, Any] = {
        "schema_version": POSTMINT_SCHEMA_VERSION,
        "step": STEP,
        "kind": KIND_POSTMINT,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "benchmark": idmap["benchmark"],
        # Per model since the 2026-10-11 re-sign. `search_threshold_bits` is what the tblouts
        # report from, and it is deliberately LOWER than the RF00230 verdict: the sub-threshold
        # rows stay visible in `noise_residue` rather than being filtered out at search time.
        "threshold_bits": dict(POSTMINT_THRESHOLD_BITS),
        "search_threshold_bits": CONTENT_THRESHOLD_BITS,
        "calibration": _calibration_record(Path(args.noise_floor), root),
        "motif": MOTIF,
        "models": {
            model: {"cm": record["path"], "cm_sha256": record["sha256"]}
            for model, record in search["cms"].items()
        },
        "infernal_version": search["infernal_version"],
        "env_lock": search["env_lock"],
        "env_lock_sha256": search["env_lock_sha256"],
        "arms": findings["arms"],
        "scope": {
            "n_contigs": idmap["n_all"],
            "n_label0": idmap["n_label0"],
            "asserted_over": "label-0 contigs only (A15 Pin 3; positives legitimately hit)",
            "n_positive_hits": len(findings["positive_hits"]),
        },
        "violations": findings["violations"],
        "n_violations": len(findings["violations"]),
        "exempt": findings["exempt"],
        "n_exempt": len(findings["exempt"]),
        "noise_residue": findings["noise_residue"],
        "n_noise_residue": len(findings["noise_residue"]),
        "positive_hits": findings["positive_hits"],
        "remedy": (
            "A15 Pin 3: the only legal remedy for a violation is a §7 stop — the pre-draw rule is "
            "strengthened and re-signed and the whole benchmark re-minted at seed 42. A seed "
            "change, a per-item redraw and an item drop are never allowed."
        ),
        "wall_s": round(time.monotonic() - started, 2),
    }
    report["provenance"] = provenance_record(
        leg="postmint-report",
        inputs=[out_dir / "postmint_idmap.json", tblout_root / "search.json"],
        root=root,
        env_lock=root / ENV_LOCK_DATA,
    )
    problems = validate_report(report)
    report["problems"] = problems
    out = Path(args.out)
    # A violation is a §7 stop, not an invalid report: the artifact is written to its canonical
    # path and the exit code says the assertion failed. A missing arm or a mis-scoped exemption
    # IS an invalid report, and `problems` carries both, so the two are separated by name.
    assertion_failed = bool(findings["violations"])
    structural = [
        problem for problem in problems if "the post-mint assertion failed" not in problem
    ]
    if structural:
        invalid = out.with_suffix(".invalid.json")
        write_json(invalid, report)
        out.unlink(missing_ok=True)
        print(f"REFUSED: {len(structural)} problem(s); wrote {invalid}")
        for problem in structural:
            print(f"  - {problem}")
        return 3
    write_json(out, report)
    print(
        f"wrote {out}: {report['n_violations']} violations, {report['n_exempt']} exempt (leader "
        f"insert), {len(findings['positive_hits'])} positive hits; {report['wall_s']}s"
    )
    return 4 if assertion_failed else 0


# ══════════════════════════════════════════════════════════════════════════════════════
# Validation (pure — never raises on a payload it did not write)
# ══════════════════════════════════════════════════════════════════════════════════════
def _mapping(node: Any, *path: str) -> dict[str, Any]:
    """The mapping at ``path``, or ``{}`` — never an attribute error on a payload we did not write.

    A validator that raised on a malformed report would be indistinguishable from a crashed run,
    and ``node or {}`` is not enough: a **string** in a field that should hold an object is
    truthy, so ``.get`` on it throws.
    """
    for key in path:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(key)
    return dict(node) if isinstance(node, Mapping) else {}


def _count(node: Any, *path: str) -> int | None:
    """An ``int`` at ``path``, or ``None``. A ``bool`` is not a count."""
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    if isinstance(node, bool) or not isinstance(node, int):
        return None
    return int(node)


def _number(node: Any, *path: str) -> float | None:
    """A finite real at ``path``, or ``None``. A ``bool`` is not a number here."""
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    try:
        value = float(node)
    except OverflowError:
        # A Python int has no width limit and float() on one past 1e308 raises, which would
        # take the validator down on a value that is simply not a count.
        return None
    return value if math.isfinite(value) else None


def validate_report(report: Mapping[str, Any], *, root: str | Path | None = None) -> list[str]:
    """Re-derive every load-bearing clause of a screen or post-mint report from its own content.

    Returns ``[]`` on a clean report and **never raises**: a traceback out of a validator is
    indistinguishable from a crashed run, and consumers (``scripts/mint_p3_16_benchmark.py``,
    ``integration/precision.py``) call this on a report they did not write. ``root``, when given,
    also binds the admitted-list file; without it the filesystem is untouched.

    It deliberately does not read ``problems`` (the field it produces) or trust
    ``controls_all_fired`` (re-derived from the controls themselves): a flag that can be written
    independently of its own evidence is a report field, not a gate
    ([[gate-clauses-need-re-derivation]]).
    """
    kind = report.get("kind")
    if kind == KIND_POSTMINT:
        return _postmint_problems(report)
    if kind != KIND_SCREEN:
        return [f"unknown report kind {kind!r}; expected {KIND_SCREEN!r} or {KIND_POSTMINT!r}"]
    return _screen_problems(report, root)


def _screen_problems(report: Mapping[str, Any], root: str | Path | None) -> list[str]:
    problems: list[str] = []

    def want(condition: Any, message: str) -> None:
        if not condition:
            problems.append(message)

    want(
        report.get("schema_version") == SCHEMA_VERSION, f"schema_version is not {SCHEMA_VERSION!r}"
    )
    want(report.get("step") == STEP, f"step is not {STEP!r}")
    want("ADR-0005 A15" in str(report.get("adr")), "adr does not name ADR-0005 A15")
    want(_count(report, "screened_dna", "k") == KMER_K, f"screened_dna.k is not {KMER_K}")
    want(
        "non-ACGT" in str(_mapping(report, "screened_dna").get("alphabet_rule")),
        "screened_dna.alphabet_rule does not state the non-ACGT skip",
    )
    want(
        _number(report, "content_screen", "threshold_bits") == CONTENT_THRESHOLD_BITS,
        f"content_screen.threshold_bits is not {CONTENT_THRESHOLD_BITS}",
    )
    want(
        _mapping(report, "content_screen").get("motif") == MOTIF,
        f"content_screen.motif is not {MOTIF!r}",
    )

    # A15 Pin 1 (iv): the screened Stage-2 population IS the shipped arm's.
    sources = _mapping(report, "screened_dna", "sources")
    for name in SCREENED_SOURCES:
        want(isinstance(sources.get(name), Mapping), f"screened_dna.sources.{name} is missing")
        want(
            (_count(sources, name, "n_kmers") or 0) > 0,
            f"screened_dna.sources.{name}.n_kmers is not positive — that source screened nothing",
        )
    stage2 = _mapping(sources, SOURCE_STAGE2)
    want(
        _count(stage2, "n") == STAGE2_EXPECTED_ADMITTED,
        f"screened_dna.sources.{SOURCE_STAGE2}.n is not {STAGE2_EXPECTED_ADMITTED}",
    )
    want(
        _count(stage2, "provenance_n_admitted") == STAGE2_EXPECTED_ADMITTED,
        f"the Stage-2 checkpoint provenance count is not {STAGE2_EXPECTED_ADMITTED}",
    )

    # Pin 1's arithmetic. `refused = kmer ∪ coord`, so BOTH identities below must hold: a single
    # total would pass with the per-rule attribution scrambled.
    fold = _mapping(report, "fold_screen")
    n_candidates = _count(report, "pool", "n_candidates")
    n_survivors = _count(fold, "n_survivors")
    n_refused = _count(fold, "n_refused")
    n_kmer = _count(fold, "n_refused_kmer")
    n_coord = _count(fold, "n_refused_coord")
    n_kmer_only = _count(fold, "n_refused_kmer_only")
    n_coord_only = _count(fold, "n_refused_coord_only")
    if None in (n_candidates, n_survivors, n_refused, n_kmer, n_coord, n_kmer_only, n_coord_only):
        problems.append("fold_screen or pool is missing one of its counts")
    else:
        want(
            n_survivors + n_refused == n_candidates,
            "n_survivors + n_refused != pool.n_candidates",
        )
        want(
            n_refused == n_kmer + n_coord_only,
            "n_refused != n_refused_kmer + n_refused_coord_only",
        )
        want(
            n_refused == n_coord + n_kmer_only,
            "n_refused != n_refused_coord + n_refused_kmer_only",
        )
        want(n_kmer >= n_kmer_only >= 0, "n_refused_kmer_only exceeds n_refused_kmer")
        want(n_coord >= n_coord_only >= 0, "n_refused_coord_only exceeds n_refused_coord")
        # A15 Pin 6's conjunction: a k-mer rule that refuses nothing is a rule that is not
        # running ([[namespace-mismatch-invisible-noop]]).
        want(n_kmer > 0, "n_refused_kmer is 0 — the k-mer rule refused nothing")
        want(n_survivors > 0, "n_survivors is 0 — there is no screened pool to draw from")
    per_source = _mapping(fold, "per_source_windows")
    for name in SCREENED_SOURCES:
        want(
            isinstance(per_source.get(name), Mapping),
            f"fold_screen.per_source_windows.{name} is missing",
        )

    # Pin 2's arithmetic.
    content = _mapping(report, "content_screen")
    n_searched = _count(content, "n_searched")
    n_admitted = _count(content, "n_admitted")
    n_content_refused = _count(content, "n_refused")
    n_either = _count(content, "n_refused_cm", "either")
    n_motif = _count(content, "n_refused_motif")
    n_motif_only = _count(content, "n_refused_motif_only")
    if None in (n_searched, n_admitted, n_content_refused, n_either, n_motif, n_motif_only):
        problems.append("content_screen is missing one of its counts")
    else:
        want(n_searched == n_survivors, "content_screen.n_searched != fold_screen.n_survivors")
        want(
            n_content_refused == n_either + n_motif_only,
            "content_screen.n_refused != n_refused_cm.either + n_refused_motif_only",
        )
        want(
            n_admitted + n_content_refused == n_searched,
            "content_screen.n_admitted + n_refused != n_searched",
        )
        want(n_motif >= n_motif_only >= 0, "n_refused_motif_only exceeds n_refused_motif")
        want(
            _count(report, "admitted", "n") == n_admitted,
            "admitted.n != content_screen.n_admitted",
        )
        try:
            models = _model_names()
        except HostScreenError as error:  # pragma: no cover - a two-model constant
            problems.append(str(error))
            models = []
        for model in models:
            per_model = _count(content, "n_refused_cm", model)
            marginal = _count(content, "n_refused_cm", f"marginal_{model}")
            other = _count(content, "n_refused_cm", _other_model(model))
            if None in (per_model, marginal, other):
                problems.append(f"content_screen.n_refused_cm is missing a count for {model}")
                continue
            want(per_model <= n_either, f"n_refused_cm.{model} exceeds n_refused_cm.either")
            want(
                marginal == n_either - other,
                f"n_refused_cm.marginal_{model} is not either - {_other_model(model)}",
            )

    # The argv A15 Pin 2 pins, checked on the EXECUTED option line as well as on the argv.
    problems += _argv_problems(content)

    want(
        "1.1.5" in str(content.get("infernal_version")),
        "content_screen.infernal_version is not the pinned Infernal 1.1.5",
    )
    want(content.get("env_lock") == ENV_LOCK, f"content_screen.env_lock is not {ENV_LOCK!r}")
    want(
        bool(re.fullmatch(r"[0-9a-f]{64}", str(content.get("env_lock_sha256")))),
        "content_screen.env_lock_sha256 is not a sha256",
    )
    for model, record in sorted(_mapping(content, "models").items()):
        want(
            bool(re.fullmatch(r"[0-9a-f]{64}", str((record or {}).get("cm_sha256")))),
            f"content_screen.models.{model}.cm_sha256 is not a sha256",
        )

    problems += _control_problems(report)

    # The annex accounts for every refused window, Pin 1's and Pin 2's.
    annex_n = _count(report, "refused_annex", "n")
    if None in (annex_n, n_candidates, n_admitted):
        problems.append("refused_annex.n or the admitted count is missing")
    else:
        want(
            annex_n == n_candidates - n_admitted,
            f"refused_annex.n ({annex_n}) != n_candidates - n_admitted "
            f"({n_candidates} - {n_admitted})",
        )

    problems += _v1_audit_problems(report)
    problems += _panel_problems(report)

    # Composition sums.
    composition = _mapping(report, "composition")
    for group, expected in (
        ("admitted", n_admitted),
        (
            "refused",
            None if n_candidates is None or n_admitted is None else n_candidates - n_admitted,
        ),
    ):
        block = _mapping(composition, group)
        n = _count(block, "n")
        if n is None or expected is None:
            problems.append(f"composition.{group}.n is missing")
            continue
        want(n == expected, f"composition.{group}.n ({n}) != {expected}")
        for axis in COMPOSITION_AXES:
            counts = _mapping(block, axis)
            total = sum(
                value
                for value in counts.values()
                if isinstance(value, int) and not isinstance(value, bool)
            )
            want(total == n, f"composition.{group}.{axis} sums to {total}, not {n}")

    if root is not None:
        problems += _admitted_list_problems(report, Path(root))
    return problems


def _argv_problems(content: Mapping[str, Any]) -> list[str]:
    """A15 Pin 2's invocation clauses, on both the recorded argv and the executed option line."""
    problems: list[str] = []
    argvs = _mapping(content, "argv")
    options = _mapping(content, "option_settings")
    if not argvs:
        return ["content_screen.argv is empty"]
    for fasta in SCREEN_FASTAS:
        for model, _cm in SCREEN_CMS:
            if f"{fasta}:{model}" not in argvs:
                problems.append(f"content_screen.argv has no entry for {fasta}:{model}")
    for key, argv in sorted(argvs.items()):
        tokens = (
            [str(token) for token in argv]
            if isinstance(argv, Sequence) and not isinstance(argv, str)
            else []
        )
        pairs = (
            ("argv", option_settings_options(" ".join(tokens))),
            ("option_settings", option_settings_options(str(options.get(key, "")))),
        )
        fasta = key.split(":", 1)[0]
        expected_t = repr(FASTA_THRESHOLDS.get(fasta, CONTENT_THRESHOLD_BITS))
        for label, flags in pairs:
            if not flags:
                problems.append(f"{label}[{key}] records no options")
                continue
            if "--toponly" in flags:
                problems.append(f"{label}[{key}] passes --toponly")
            if "--cut_ga" in flags:
                problems.append(f"{label}[{key}] passes --cut_ga")
            if "--max" in flags:
                problems.append(f"{label}[{key}] passes --max; Pin 2 is the filtered pipeline")
            if flags.get("-Z") != repr(SEARCH_SPACE_MB):
                problems.append(f"{label}[{key}] does not pass -Z {SEARCH_SPACE_MB!r}")
            if flags.get("-T") != expected_t:
                problems.append(f"{label}[{key}] does not pass -T {expected_t}")
    return problems


def _control_problems(report: Mapping[str, Any]) -> list[str]:
    """The must-fire controls: every mechanism present, every ``fired`` re-derived."""
    controls = report.get("controls")
    if not isinstance(controls, Sequence) or isinstance(controls, str) or not controls:
        return ["controls is empty — A15 Pin 1/2 require one control per mechanism"]
    problems: list[str] = []
    names = {str((entry or {}).get("name")) for entry in controls if isinstance(entry, Mapping)}
    required = {
        CONTROL_HOST_CLEAN,
        CONTROL_COORD_ONLY,
        CONTROL_CONTENT,
        CONTROL_CONTENT_HOST,
        *(f"pin1_kmer_{source}" for source in SCREENED_SOURCES),
    }
    missing = sorted(required - names)
    if missing:
        problems.append(f"controls are missing the must-fire mechanism(s) {missing}")
    if not any(name.startswith("pin1_accession_") for name in names):
        problems.append("controls carry no accession-normalisation control")
    for entry in controls:
        if not isinstance(entry, Mapping):
            problems.append("a control entry is not an object")
            continue
        fired = entry.get("fired")
        if not isinstance(fired, bool) or fired != (entry.get("observed") == entry.get("expected")):
            problems.append(f"control {entry.get('name')!r}: fired is not observed == expected")
    derived = all(bool((entry or {}).get("fired")) for entry in controls)
    if report.get("controls_all_fired") != derived:
        problems.append("controls_all_fired disagrees with the controls themselves")
    if not derived:
        failed = sorted(
            str((entry or {}).get("name"))
            for entry in controls
            if isinstance(entry, Mapping) and not entry.get("fired")
        )
        problems.append(f"must-fire control(s) did not fire: {failed}")
    return problems


def _admitted_list_problems(report: Mapping[str, Any], root: Path) -> list[str]:
    """Bind ``admitted.sha256`` to the file on disk, when it is there."""
    listed = _mapping(report, "admitted").get("list")
    if not isinstance(listed, str):
        return ["admitted.list is not a path"]
    # Resolved against the ONE root the caller gave, never a second candidate: trying two
    # roots lets a report bind to a same-named list in the other checkout and report a
    # mismatch against a file it is not about ([[gate-must-bind-to-upstream-evidence]]).
    path = root / listed
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:  # pragma: no cover - unreadable committed file
        return [f"admitted list unreadable: {error}"]
    problems: list[str] = []
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != _mapping(report, "admitted").get(
        "sha256"
    ):
        problems.append(f"{listed}: sha256 does not match admitted.sha256")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != _count(report, "admitted", "n"):
        problems.append(
            f"{listed}: holds {len(lines)} ids, admitted.n says {_count(report, 'admitted', 'n')!r}"
        )
    return problems


def _v1_audit_problems(report: Mapping[str, Any]) -> list[str]:
    """Re-derive the ``v1_audit`` counts from its own per-host entries."""
    audit = report.get("v1_audit")
    if not isinstance(audit, Mapping):
        return ["v1_audit is missing"]
    hosts = audit.get("hosts")
    counts = audit.get("counts")
    if not isinstance(hosts, Mapping) or not isinstance(counts, Mapping):
        return ["v1_audit.hosts or v1_audit.counts is missing"]
    problems: list[str] = []
    if _count(counts, "n_hosts") != len(hosts):
        problems.append(f"v1_audit.counts.n_hosts != len(hosts) ({len(hosts)})")
    bit_model = str(audit.get("bit_model"))
    derived = dict.fromkeys(
        (
            "n_ge_93",
            "n_ge_20",
            "n_ge_10",
            "n_motif",
            "n_refused",
            "n_refused_fold",
            "n_refused_content",
        ),
        0,
    )
    for entry in hosts.values():
        if not isinstance(entry, Mapping):
            problems.append("a v1_audit host entry is not an object")
            continue
        bits = _number(entry, "bits", bit_model)
        derived["n_ge_93"] += bool(bits is not None and bits >= 93.0)
        derived["n_ge_20"] += bool(bits is not None and bits >= 20.0)
        derived["n_ge_10"] += bool(bits is not None and bits >= 10.0)
        derived["n_motif"] += bool(entry.get("motif"))
        derived["n_refused"] += bool(entry.get("refused"))
        derived["n_refused_fold"] += bool(entry.get("refused_fold"))
        derived["n_refused_content"] += bool(entry.get("refused_content"))
    for key, value in derived.items():
        if _count(counts, key) != value:
            problems.append(f"v1_audit.counts.{key} is not the {value} its own hosts give")
    if _count(_mapping(audit, "composition"), "n") != len(hosts):
        problems.append("v1_audit.composition.n != len(hosts)")
    entries = [entry for entry in hosts.values() if isinstance(entry, Mapping)]
    if audit.get("composition") != _composition(entries):
        # A15 Pin 7(e)'s v1 column is read beside v2's by the precision report; it must be the
        # table its own per-host entries give, not one asserted alongside them.
        problems.append("v1_audit.composition is not the table its own hosts give")
    return problems


def _panel_problems(report: Mapping[str, Any]) -> list[str]:
    """Re-derive every panel recall from its own hit count and ``n``."""
    panel = report.get("sensitivity_panel")
    if not isinstance(panel, Mapping):
        return ["sensitivity_panel is missing"]
    problems: list[str] = []
    if _number(panel, "threshold_bits") != CONTENT_THRESHOLD_BITS:
        problems.append(f"sensitivity_panel.threshold_bits is not {CONTENT_THRESHOLD_BITS}")
    variants = panel.get("variants")
    if not isinstance(variants, Mapping) or not variants:
        return [*problems, "sensitivity_panel.variants is empty"]
    for label in ("full", "t25", "t50"):
        if label not in variants:
            problems.append(f"sensitivity_panel.variants has no {label!r} row")
    try:
        mechanisms = (*_model_names(), "motif", "either")
    except HostScreenError as error:  # pragma: no cover - a two-model constant
        return [*problems, str(error)]
    for label, block in sorted(variants.items()):
        if not isinstance(block, Mapping):
            problems.append(f"sensitivity_panel.variants.{label} is not an object")
            continue
        n = _count(block, "n")
        if not n:
            problems.append(f"sensitivity_panel.variants.{label}.n is not positive")
            continue
        for mechanism in mechanisms:
            cell = block.get(mechanism)
            if not isinstance(cell, Mapping):
                problems.append(f"sensitivity_panel.variants.{label}.{mechanism} is missing")
                continue
            hits = _count(cell, "n_hits")
            recall = _number(cell, "recall")
            if hits is None or recall is None:
                problems.append(f"sensitivity_panel.variants.{label}.{mechanism} is incomplete")
                continue
            if not 0 <= hits <= n:
                problems.append(
                    f"sensitivity_panel.variants.{label}.{mechanism}.n_hits is outside [0, n]"
                )
            elif abs(recall - hits / n) > 1e-12:
                problems.append(
                    f"sensitivity_panel.variants.{label}.{mechanism}.recall is not n_hits / n"
                )
    return problems


def _postmint_problems(report: Mapping[str, Any]) -> list[str]:
    """A15 Pin 3's clause set, re-derived from the post-mint report's own content."""
    problems: list[str] = []

    def want(condition: Any, message: str) -> None:
        if not condition:
            problems.append(message)

    want(
        report.get("schema_version") == POSTMINT_SCHEMA_VERSION,
        f"schema_version is not {POSTMINT_SCHEMA_VERSION!r}",
    )
    want(report.get("step") == STEP, f"step is not {STEP!r}")
    want("ADR-0005 A15" in str(report.get("adr")), "adr does not name ADR-0005 A15")
    want(report.get("motif") == MOTIF, f"motif is not {MOTIF!r}")

    # ── the re-signed per-model thresholds, and the evidence they are signed against ──────
    pinned = _mapping(report, "threshold_bits")
    thresholds: dict[str, float] = {}
    for model, value in sorted(pinned.items()):
        bits = _as_bits(value)
        if bits is None:
            problems.append(f"threshold_bits.{model} is not a bit score: {value!r}")
            continue
        thresholds[str(model)] = bits
    want(
        thresholds == {str(k): float(v) for k, v in POSTMINT_THRESHOLD_BITS.items()},
        f"threshold_bits is not A15 Pin 3's pinned table {dict(POSTMINT_THRESHOLD_BITS)}",
    )
    search_floor = _number(report, "search_threshold_bits")
    want(
        search_floor == CONTENT_THRESHOLD_BITS,
        f"search_threshold_bits is not {CONTENT_THRESHOLD_BITS}",
    )
    for model in sorted(_mapping(report, "models")):
        want(
            model in thresholds,
            f"{model} is searched but has no Pin 3 assertion threshold",
        )
    if search_floor is not None:
        for model, bits in sorted(thresholds.items()):
            # The structural rule, re-derived from the report's own two numbers: an assertion
            # configured more sensitively than the screen feeding it is unsatisfiable however
            # sound both are ([[assertion-stronger-than-its-screen-cannot-pass]]).
            want(
                bits >= search_floor,
                f"threshold_bits.{model} ({bits}) is below search_threshold_bits "
                f"({search_floor}): the assertion would be stronger than its own screen",
            )
    calibration = _mapping(report, "calibration")
    floor = _mapping(calibration, "null_max_bits")
    want(bool(calibration), "calibration is missing — the thresholds name no measured evidence")
    want(
        bool(calibration.get("sha256")),
        "calibration.sha256 is missing — the thresholds do not bind to a specific artifact",
    )
    for model, bits in sorted(thresholds.items()):
        measured = _as_bits(floor.get(model))
        if measured is None:
            problems.append(
                f"calibration.null_max_bits.{model} is missing — {model}'s threshold is signed "
                "against no measured null"
            )
            continue
        # Strict: a threshold EQUAL to the worst null sequence still fires on that sequence.
        want(
            bits > measured,
            f"threshold_bits.{model} ({bits}) does not clear the measured matched-null maximum "
            f"({measured}): it is inside the noise floor it was re-signed to escape",
        )
    violations = report.get("violations")
    if not isinstance(violations, Sequence) or isinstance(violations, str):
        problems.append("violations is missing")
    else:
        want(_count(report, "n_violations") == len(violations), "n_violations != len(violations)")
        want(
            not violations,
            f"the post-mint assertion failed: {len(violations)} violation(s) "
            f"on label-0 contigs",
        )
        for entry in violations:
            if not isinstance(entry, Mapping):
                problems.append("a violation entry is not an object")
                continue
            model, bits = str(entry.get("model")), _as_bits(entry.get("bits"))
            if model == "motif":
                want(
                    entry.get("bits") is None,
                    "a motif violation carries a bit score; the motif rule has no threshold",
                )
                continue
            limit = thresholds.get(model)
            if limit is None:
                problems.append(f"a violation names model {model!r}, which has no threshold")
            elif bits is None:
                problems.append(f"a {model} violation carries no bit score")
            else:
                # Re-derived, not trusted: a row listed as a violation must actually clear its
                # own model's threshold ([[gate-clauses-need-re-derivation]]).
                want(
                    bits >= limit,
                    f"violation on {entry.get('contig_id')!r} scores {bits} {model} bits, below "
                    f"its own threshold {limit} — it belongs in noise_residue",
                )
    residue = report.get("noise_residue")
    if not isinstance(residue, Sequence) or isinstance(residue, str):
        problems.append("noise_residue is missing")
    elif search_floor is not None:
        want(
            _count(report, "n_noise_residue") == len(residue),
            "n_noise_residue != len(noise_residue)",
        )
        for entry in residue:
            if not isinstance(entry, Mapping):
                problems.append("a noise_residue entry is not an object")
                continue
            want(int(entry.get("label", -1)) == 0, "a noise_residue entry is not label-0")
            model, bits = str(entry.get("model")), _as_bits(entry.get("bits"))
            limit = thresholds.get(model)
            if limit is None or bits is None:
                problems.append(f"a noise_residue entry for {model!r} carries no comparable score")
                continue
            # The residue is the half-open band between the search floor and the verdict. Both
            # ends are asserted: a row below the floor could not have been reported by the
            # search at all, and a row at or above the threshold is a violation being filed as
            # tolerated noise — the exact shape of a gate talked out of firing.
            want(
                search_floor <= bits < limit,
                f"noise_residue on {entry.get('contig_id')!r} scores {bits} {model} bits, "
                f"outside the disclosed band [{search_floor}, {limit})",
            )
            want(
                _as_bits(entry.get("assertion_threshold_bits")) == limit,
                f"noise_residue on {entry.get('contig_id')!r} records assertion threshold "
                f"{entry.get('assertion_threshold_bits')!r}, not {model}'s {limit}",
            )
    exempt = report.get("exempt")
    if not isinstance(exempt, Sequence) or isinstance(exempt, str):
        problems.append("exempt is missing")
    else:
        want(_count(report, "n_exempt") == len(exempt), "n_exempt != len(exempt)")
        for entry in exempt:
            if not isinstance(entry, Mapping):
                problems.append("an exempt entry is not an object")
                continue
            want(
                str(entry.get("pool")) == "leader_decoy",
                f"exempt entry for pool {entry.get('pool')!r}: only the leader insert is exempt",
            )
            interval, span = entry.get("exempt_interval"), entry.get("span")
            if (
                not isinstance(interval, Sequence)
                or not isinstance(span, Sequence)
                or len(interval) != 2
                or len(span) != 2
            ):
                problems.append("an exempt entry carries no interval/span pair")
                continue
            lo, hi = min(int(span[0]), int(span[1])), max(int(span[0]), int(span[1]))
            want(
                spans_overlap((lo - 1, hi), (int(interval[0]), int(interval[1]))),
                f"exempt hit {list(span)} does not overlap the leader insert {list(interval)}",
            )
    arms = _mapping(report, "arms")
    want("screen" in arms, "no screen arm in the post-mint report")
    want(
        "filter_off" in arms,
        "no filter-off (--max) arm in the post-mint report — A15 Pin 3 requires it",
    )
    # Coverage, before the per-search clauses. Those iterate the searches that are PRESENT, so
    # they cannot see one that is absent: a report with every TBDB001 search deleted, or with an
    # empty filter_off block, satisfied every one of them and still reported n_violations == 0
    # ([[clauses-must-guard-emptiness]]). Both D2 models over both arms is what Pin 3 requires,
    # and an assertion is only as strong as the searches it actually ran.
    for arm, fastas in sorted(REQUIRED_POSTMINT_SEARCHES.items()):
        block = _mapping(arms, arm)
        present = set(_mapping(block, "argv"))
        required = {f"{fasta}:{model}" for fasta in fastas for model, _cm in SCREEN_CMS}
        absent = sorted(required - present)
        want(
            not absent,
            f"arms.{arm} is missing {len(absent)} required search(es): {absent}; A15 Pin 3 "
            f"asserts over every D2 model in both arms",
        )
    for arm, block in sorted(arms.items()):
        for key, argv in sorted(_mapping(block, "argv").items()):
            flags = option_settings_options(" ".join(str(token) for token in argv))
            executed = option_settings_options(str(_mapping(block, "option_settings").get(key, "")))
            for label, tokens in (("argv", flags), ("option_settings", executed)):
                want("--toponly" not in tokens, f"arms.{arm}.{label}[{key}] passes --toponly")
                want("--cut_ga" not in tokens, f"arms.{arm}.{label}[{key}] passes --cut_ga")
                want(
                    tokens.get("-T") == repr(CONTENT_THRESHOLD_BITS),
                    f"arms.{arm}.{label}[{key}] is not -T {CONTENT_THRESHOLD_BITS!r}",
                )
                want(
                    tokens.get("-Z") == repr(SEARCH_SPACE_MB),
                    f"arms.{arm}.{label}[{key}] is not -Z {SEARCH_SPACE_MB!r}",
                )
                if arm == "filter_off":
                    want("--max" in tokens, f"arms.filter_off.{label}[{key}] does not pass --max")
                else:
                    want("--max" not in tokens, f"arms.screen.{label}[{key}] passes --max")
    want((_count(report, "scope", "n_label0") or 0) > 0, "scope.n_label0 is not positive")
    return problems


# ══════════════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════════════
def _data_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "checkout holding the DVC-tracked inputs; defaults to this module's own repo root. "
            "A worktree has no data/ of its own, so screening from one points this at the main "
            "checkout while every recorded path stays repo-relative"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prep", help="data env: pool + screened DNA + Pin 1 -> FASTAs")
    prep.add_argument("--mining-pool", default=DEFAULT_MINING_POOL)
    prep.add_argument("--decoys", default=DEFAULT_DECOYS)
    prep.add_argument("--dataset", default=DEFAULT_DATASET)
    prep.add_argument("--context", default=DEFAULT_CONTEXT)
    prep.add_argument("--labels", default=DEFAULT_LABELS)
    prep.add_argument("--split-table", default=DEFAULT_SPLIT_TABLE)
    prep.add_argument("--benchmark-v1", default=DEFAULT_BENCHMARK_V1)
    prep.add_argument("--rf00230-master", default=DEFAULT_RF00230_MASTER)
    prep.add_argument("--stage2-provenance", default=DEFAULT_STAGE2_PROVENANCE)
    prep.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    prep.add_argument(
        "--coordinate-only-prefix",
        default="4ef2d573",
        help="id prefix of the A15 Pin 1 coordinate-only v1 host must-fire control",
    )
    prep.add_argument(
        "--control-host",
        default=None,
        help=(
            "candidate_id of the Pin-1 survivor the controls splice into; the default is the "
            "first motif-free survivor in id order"
        ),
    )
    _data_root_arg(prep)
    prep.set_defaults(func=_cmd_prep)

    search = sub.add_parser("search", help="infernal env: FASTA x model -> tblouts + search.json")
    search.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    search.add_argument("--tblout-dir", default=None)
    search.add_argument(
        "--fasta", action="append", default=None, help="repeatable; default: the four screen FASTAs"
    )
    search.add_argument(
        "--max-fasta",
        action="append",
        default=None,
        help="repeatable: also search this FASTA with the orthogonal filter-off (--max) arm",
    )
    search.add_argument("--cpu", type=int, default=4)
    search.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="how many (FASTA, model) searches to run at once; each uses --cpu threads",
    )
    search.add_argument("--timeout-s", type=float, default=12 * 3600.0)
    _data_root_arg(search)
    search.set_defaults(func=_cmd_search)

    report = sub.add_parser(
        "report", help="data env: Pin 2 + controls + panel + v1 audit -> host_screen.json"
    )
    report.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    report.add_argument("--tblout-dir", default=None)
    report.add_argument("--out", default=DEFAULT_REPORT)
    report.add_argument("--admitted", default=DEFAULT_ADMITTED_LIST)
    report.add_argument("--refused", default=DEFAULT_REFUSED_ANNEX)
    _data_root_arg(report)
    report.set_defaults(func=_cmd_report)

    fam = sub.add_parser("familiarity", help="data env: A15 Pin 7(b) familiarity baseline score")
    fam.add_argument("--benchmark", required=True)
    fam.add_argument("--out", required=True)
    fam.add_argument("--mining-pool", default=DEFAULT_MINING_POOL)
    fam.add_argument("--decoys", default=DEFAULT_DECOYS)
    fam.add_argument("--context", default=DEFAULT_CONTEXT)
    fam.add_argument("--labels", default=DEFAULT_LABELS)
    fam.add_argument("--split-table", default=DEFAULT_SPLIT_TABLE)
    _data_root_arg(fam)
    fam.set_defaults(func=_cmd_familiarity)

    pm_prep = sub.add_parser(
        "postmint-prep", help="data env: minted v2 contigs -> the assertion FASTAs"
    )
    pm_prep.add_argument("--benchmark", required=True)
    pm_prep.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    _data_root_arg(pm_prep)
    pm_prep.set_defaults(func=_cmd_postmint_prep)

    pm_report = sub.add_parser("postmint-report", help="data env: A15 Pin 3's post-mint assertion")
    pm_report.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    pm_report.add_argument("--tblout-dir", default=None)
    pm_report.add_argument("--out", default=DEFAULT_POSTMINT_REPORT)
    pm_report.add_argument(
        "--noise-floor",
        default=DEFAULT_NOISE_FLOOR_REPORT,
        help=(
            "the matched-null calibration artifact A15 Pin 3's per-model thresholds are signed "
            "against; required, because a threshold with no measured floor beside it is a "
            "number rather than a calibration"
        ),
    )
    _data_root_arg(pm_report)
    pm_report.set_defaults(func=_cmd_postmint_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "tblout_dir", None) is None and hasattr(args, "out_dir"):
        args.tblout_dir = str(Path(args.out_dir) / "tblout")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())


__all__ = [
    "ABSENT_PARENT",
    "ADR",
    "ALPHABET_RULE",
    "COMPOSITION_AXES",
    "COMPOSITION_PARENT_FIELDS",
    "AUDIT_THRESHOLD_BITS",
    "CONTENT_THRESHOLD_BITS",
    "DEFAULT_NOISE_FLOOR_REPORT",
    "POSTMINT_SCHEMA_VERSION",
    "POSTMINT_THRESHOLD_BITS",
    "CONTROL_CONTENT",
    "CONTROL_CONTENT_HOST",
    "CONTROL_COORD_ONLY",
    "CM_QUERY_NAMES",
    "CONTROL_HOST_CLEAN",
    "EXPECTED_EMBEDDED",
    "GENERATED_BY",
    "KIND_NOISE_FLOOR",
    "KIND_POSTMINT",
    "KIND_SCREEN",
    "KMER_K",
    "MOTIF",
    "motif_spans",
    "spans_overlap",
    "SCHEMA_VERSION",
    "SCREENED_SOURCES",
    "SCREEN_CMS",
    "SCREEN_FASTAS",
    "STAGE2_EXPECTED_ADMITTED",
    "STEP",
    "Hit",
    "HostScreenError",
    "ScreenedDna",
    "WindowVerdict",
    "at_least",
    "best_bits",
    "check_query_names",
    "build_parser",
    "build_screened_dna",
    "canonical_kmers",
    "content_refused",
    "control_entry",
    "control_segment",
    "coordinate_only_candidate",
    "decoy_embedded",
    "disclosures",
    "familiarity_fraction",
    "first_fasta_record",
    "hits_by_target",
    "intervals_overlap",
    "kmer_hits",
    "leader_exempt_interval",
    "load_candidates",
    "main",
    "motif_present",
    "normalise_dna",
    "option_settings_options",
    "option_settings_line",
    "parse_tblout_hits",
    "provenance_record",
    "pin1_controls",
    "postmint_findings",
    "postmint_threshold",
    "read_searches",
    "recorded_path",
    "require_repo_relative",
    "run_search_tasks",
    "screen_window",
    "search_argv",
    "search_line",
    "sorted_unique_kmers",
    "surrogate",
    "truncate_window",
    "validate_report",
]
