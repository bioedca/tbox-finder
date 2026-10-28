"""P3-16′-remint — the matched-null noise floor A15 Pin 3's thresholds are signed against.

Why this module exists
----------------------
A15 Pin 3's post-mint assertion, as drafted, required **0** label-0 hits at ≥ 10.0 bits on either
D2 model. On the first real run it returned 7 rows over 5 gated contigs, all RF00230, after the
signed leader-overlap correction. Three independent channels produced them — ``--max``-only
filter asymmetry on the host, the ``decoys_v0`` inserts (never content-screened), and host+insert
junctions (objects that do not exist pre-draw) — and none of the three is a screen defect
([[assertion-stronger-than-its-screen-cannot-pass]]).

That diagnosis says where the hits came from. It does **not** say whether a ≥ 10-bit RF00230
alignment on ~1 kb of DNA is evidence of T-box homology at all. Only a matched control answers
that, and it has to be measured *before* a threshold is certified, not reasoned about
([[matched-control-before-certifying]]). This module is that measurement, as a reproducible
artifact rather than an argument: ``POSTMINT_THRESHOLD_BITS`` in ``host_screen.py`` is refused by
that module's own validator unless it clears the floor measured here.

The three nulls
---------------
Each contains **zero** T-boxes by construction, and each isolates one channel::

    n1_inserts    the 691 decoys_v0 inserts actually spliced into the gated negatives, alone
                  -> the insert channel
    n2_scramble1  the SAME 691 hosts re-spliced with the SAME 691 inserts at a phase moved
    n2_scramble2  >= MIN_PHASE_SHIFT nt; two independent redraws
                  -> the junction channel, and the only arm matched to the minted contigs
    n3_pool       N_POOL_WINDOWS screen-ADMITTED windows the seeded draw did not take, pre-splice
                  -> the host channel, i.e. the residual the Pin 2 screen leaves behind

``n2`` is the arm an aggregate comparison is read against, because it is the one that differs
from a minted contig in **exactly one** respect: where the junction falls. ``n3`` is conditioned
on having passed Pin 2, so it measures what admission leaves, and it cannot be used to estimate
Pin 2's own false-refusal rate — a limitation the report states rather than eliding.

Matchedness is asserted, never assumed
--------------------------------------
A control silently replaced by a copy of its positive reads as "no power" rather than as a broken
control ([[control-matchedness-must-be-assserted]] — recorded as
[[control-matchedness-must-be-asserted]]). So ``prep`` measures nine properties of the bytes it
wrote, records every observed value, and ``report`` re-derives them from the FASTAs that were
actually searched.

One of them has a trap worth naming. The mint is an equal-length **overwrite**, not an insertion:
a minted contig is the host window with ``[phase, phase + L)`` replaced by the insert. So the
host bases under the old window are destroyed, and "excise the insert from both and compare the
remainder" — the obvious check — compares ``host`` minus one window against ``host`` minus a
*different* window and fails on every item for a reason that has nothing to do with matchedness.
The property an overwrite actually implies is that a minted contig and its scramble agree
**outside both** insert windows, which is what ``_matchedness`` tests.

Three legs, two environments
----------------------------
``search`` runs in the ``infernal`` env (``cmsearch``, no pandas), so the pandas-dependent pool
load stays inside ``prep``::

    prep    (data)      benchmark + minted contigs + pool + admitted list -> 4 FASTAs + prep.json
    search  (infernal)  FASTA x model x pipeline                         -> tblouts + search.json
    report  (data)      per-cell distributions + null_max_bits           -> reports/p3/...json

Searches run at ``-T 0`` so the **whole** score distribution is recorded, not the tail above some
threshold: the quantity being measured is where the noise floor *is*, and a search that reports
only above 10 bits cannot say whether 10 bits is unusual.

What this artifact deliberately does not do
-------------------------------------------
It never reads the post-mint report. The comparison of observed violations against these nulls
belongs in that report and in the dev-log; putting it here would make two artifacts read each
other and leave neither able to validate the pair. This file measures the null, and nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder import provenance
from tbox_finder.data import window_dataset as wd
from tbox_finder.integration import host_screen as hs

SCHEMA_VERSION = "1.0"
STEP = "P3-16′-remint"
GENERATED_BY = "src/tbox_finder/integration/noise_floor.py"
ADR = "ADR-0005 A15"
KIND = hs.KIND_NOISE_FLOOR

#: The seed every draw here takes. 42, the same as the mint's, because this measurement is part
#: of the same pre-registered construction and a second seed would be a second degree of freedom.
SEED = 42

#: How far a scramble's splice phase must move. 50 nt is longer than the CM's own alignment slop
#: at these scores, so a re-spliced junction is a genuinely different junction rather than the
#: same one jittered. ``prep`` refuses an item for which no such phase exists.
MIN_PHASE_SHIFT = 50

#: Admitted-but-undrawn windows in the host-channel null. 1,000 resolves a rate near 0.3 % to
#: about a factor of two, which is the precision the comparison needs.
N_POOL_WINDOWS = 1000

#: ``-T``. Zero, so every cell records a full distribution rather than a thresholded tail.
REPORT_THRESHOLD_BITS = 0.0

#: The rate the report tabulates, for comparability with A15 Pin 3's drafted threshold. The
#: *verdict* threshold lives in ``host_screen.POSTMINT_THRESHOLD_BITS``; this is a reporting cut.
RATE_AT_BITS = 10.0

#: ``default`` is the Pin 2 pipeline; ``max`` is Pin 3's orthogonal filters-off arm. Both are
#: measured, because the drafted threshold applied one number to both and their floors differ by
#: more than 6 bits.
PIPELINES: tuple[tuple[str, bool], ...] = (("default", False), ("max", True))

ARMS: tuple[str, ...] = ("n1_inserts", "n2_scramble1", "n2_scramble2", "n3_pool")

DEFAULT_OUT_DIR = "data/interim/p3_16/noise_floor"
DEFAULT_REPORT = "reports/p3/postmint_noise_floor.json"
DEFAULT_BENCHMARK = "data/interim/p3_16/benchmark_v2.json"
DEFAULT_SCREEN_DIR = "data/interim/p3_16/host_screen"
DEFAULT_ADMITTED = "reports/p3/host_screen_admitted.txt"
DEFAULT_MINING_POOL = "data/processed/negatives/mining_pool_v0.parquet"

PREP_JSON = "prep.json"
SEARCH_JSON = "search.json"


class NoiseFloorError(RuntimeError):
    """A refusal that must stop the leg rather than be recorded as a measurement."""


# ══════════════════════════════════════════════════════════════════════════════════════
# FASTA
# ══════════════════════════════════════════════════════════════════════════════════════
def read_fasta(path: str | Path) -> dict[str, str]:
    """``{id: sequence}``, keyed on the header's first whitespace-delimited token.

    That token is what ``cmsearch`` writes into the tblout's target column, so keying on it is
    what lets a tblout row be traced back to the sequence searched. A duplicate id raises: two
    records sharing one id would silently merge and every *summed* invariant would still hold
    ([[duplicate-key-merges-instead-of-colliding]]).
    """
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name = line[1:].split()[0] if line[1:].split() else ""
            if name in records:
                raise NoiseFloorError(f"{path}: duplicate FASTA id {name!r}")
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        records[name] = "".join(chunks)
    if not records:
        raise NoiseFloorError(f"{path}: no FASTA records")
    return records


def write_fasta(path: Path, records: Mapping[str, str], *, width: int = 80) -> str:
    """Write ``records`` in id order and return the file's sha256."""
    lines: list[str] = []
    for name, sequence in records.items():
        lines.append(f">{name}")
        lines.extend(sequence[i : i + width] for i in range(0, len(sequence), width))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return provenance.sha256_file(path)


# ══════════════════════════════════════════════════════════════════════════════════════
# Building the nulls
# ══════════════════════════════════════════════════════════════════════════════════════
def gated_negatives(benchmark: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The gated label-0 items, in benchmark order.

    The ``leader_decoy`` is excluded: it is T-box-derived by construction (A15 Pin 4), so it is
    not a null and including it would put a real T-box into the control
    ([[control-matchedness-must-be-asserted]]).
    """
    items = [
        dict(item)
        for item in benchmark["items"]
        if int(item.get("label", 1)) == 0 and str(item.get("pool")) != "leader_decoy"
    ]
    if not items:
        raise NoiseFloorError("the benchmark carries no gated label-0 items")
    return items


def scramble_phase(rng: random.Random, *, window_nt: int, insert_nt: int, phase: int) -> int:
    """A splice phase at least ``MIN_PHASE_SHIFT`` from ``phase``, drawn from ``rng``.

    Refuses rather than relaxing the shift when the geometry leaves no such phase: a control
    quietly re-spliced 3 nt away is not a different junction, and the relaxation would be
    invisible in the artifact.
    """
    choices = [
        candidate
        for candidate in range(0, window_nt - insert_nt + 1)
        if abs(candidate - phase) >= MIN_PHASE_SHIFT
    ]
    if not choices:
        raise NoiseFloorError(
            f"no phase ≥ {MIN_PHASE_SHIFT} nt from {phase} fits a {insert_nt}-nt insert in a "
            f"{window_nt}-nt window"
        )
    return rng.choice(choices)


def splice(host: str, insert: str, phase: int) -> str:
    """The mint's operation: an equal-length **overwrite** of ``host[phase : phase + len)``.

    Not an insertion. The distinction is the whole reason ``_matchedness`` tests agreement
    *outside both* windows rather than excising and comparing remainders.
    """
    if phase < 0 or phase + len(insert) > len(host):
        raise NoiseFloorError(f"a {len(insert)}-nt insert at phase {phase} overruns the host")
    return host[:phase] + insert + host[phase + len(insert) :]


def build_nulls(
    items: Sequence[Mapping[str, Any]],
    minted: Mapping[str, str],
    hosts: Mapping[str, str],
    pool_windows: Sequence[tuple[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """The four null FASTAs plus the id map, all deterministic at :data:`SEED`.

    ``minted`` and ``hosts`` are both required: the insert is read out of the minted contig (it
    is the only place the spliced bytes exist as spliced) while the scramble is built from the
    **pool** host, because the overwrite destroyed the host bases under the minted window.
    """
    rng1 = random.Random(f"{SEED}:n2_scramble1")
    rng2 = random.Random(f"{SEED}:n2_scramble2")
    n1: dict[str, str] = {}
    s1: dict[str, str] = {}
    s2: dict[str, str] = {}
    idmap: dict[str, Any] = {arm: {} for arm in ARMS}
    shifts: list[int] = []
    for index, item in enumerate(items):
        contig_id, host_id = str(item["contig_id"]), str(item["host_id"])
        phase, insert_nt = int(item["splice_phase"]), int(item["locus_length"])
        contig = minted[contig_id]
        host = hosts[host_id]
        if len(host) != len(contig):
            raise NoiseFloorError(
                f"{contig_id}: pool host is {len(host)} nt, minted contig {len(contig)} nt"
            )
        insert = contig[phase : phase + insert_nt]
        if len(insert) != insert_nt:
            raise NoiseFloorError(f"{contig_id}: insert at phase {phase} is truncated")
        tag = f"{index:04d}"
        n1[f"i{tag}"] = insert
        idmap["n1_inserts"][f"i{tag}"] = {"contig_id": contig_id, "insert_nt": insert_nt}
        for arm, rng, store, prefix in (
            ("n2_scramble1", rng1, s1, "s1_"),
            ("n2_scramble2", rng2, s2, "s2_"),
        ):
            moved = scramble_phase(rng, window_nt=len(host), insert_nt=insert_nt, phase=phase)
            store[f"{prefix}{tag}"] = splice(host, insert, moved)
            shifts.append(abs(moved - phase))
            idmap[arm][f"{prefix}{tag}"] = {
                "contig_id": contig_id,
                "host_id": host_id,
                "minted_phase": phase,
                "scramble_phase": moved,
                "insert_nt": insert_nt,
            }
    p3 = {}
    for index, (host_id, sequence) in enumerate(pool_windows):
        name = f"p{index:04d}"
        p3[name] = sequence
        idmap["n3_pool"][name] = {"host_id": host_id}
    records = {"n1_inserts": n1, "n2_scramble1": s1, "n2_scramble2": s2, "n3_pool": p3}
    return records, {"idmap": idmap, "min_phase_shift_nt": min(shifts) if shifts else None}


def choose_pool_windows(
    admitted: Sequence[str],
    drawn: Iterable[str],
    rows: Mapping[str, str],
    *,
    n: int = N_POOL_WINDOWS,
) -> list[tuple[str, str]]:
    """``n`` admitted windows that the draw did **not** take, sampled at :data:`SEED`.

    Disjointness from the drawn hosts is the point of the arm: a null sharing windows with the
    thing it is a null for measures that thing, not its floor. Sampled from a **sorted** list so
    the draw does not inherit the admitted file's order.
    """
    taken = {str(value) for value in drawn}
    eligible = sorted({str(value) for value in admitted} - taken)
    missing = [value for value in eligible if value not in rows]
    if missing:
        raise NoiseFloorError(
            f"{len(missing)} admitted windows are absent from the candidate pool, e.g. "
            f"{missing[0]!r}"
        )
    if len(eligible) < n:
        raise NoiseFloorError(f"only {len(eligible)} admitted-and-undrawn windows; need {n}")
    chosen = random.Random(f"{SEED}:n3_pool").sample(eligible, n)
    return [(host_id, rows[host_id]) for host_id in sorted(chosen)]


#: The nine matchedness controls, by name and in order. The validator requires **exactly** this
#: set, because a clause set that only re-derives the rows it is *given* cannot see the rows it
#: was not: a post-hoc artifact that kept one trivial control and dropped the eight inconvenient
#: ones would validate clean ([[new-gate-clause-invalidates-old-reports]]).
MATCHEDNESS_CONTROLS: tuple[str, ...] = (
    "n1_covers_the_gated_negatives",
    "n2_length_matches_the_minted_contig",
    "n2_carries_its_insert_exactly_once",
    "n2_insert_is_the_minted_insert",
    "n2_agrees_outside_both_insert_windows",
    "n2_reproduces_no_minted_contig",
    "n2_phase_moved_far_enough",
    "n3_is_disjoint_from_the_drawn_hosts",
    "n3_is_the_requested_size",
)


def matchedness(
    items: Sequence[Mapping[str, Any]],
    minted: Mapping[str, str],
    records: Mapping[str, Mapping[str, str]],
    idmap: Mapping[str, Any],
    drawn: Iterable[str],
) -> list[dict[str, Any]]:
    """The nine properties that make these arms controls, each with its observed value.

    Expressed as ``hs.control_entry`` rows so ``fired`` is **derived** from expected vs observed
    rather than asserted, and so the report can re-derive the same equality on the written
    artifact ([[gate-clauses-need-re-derivation]]).

    Refuses an empty corpus rather than measuring one. Eight of the nine controls are counting
    equalities, and on zero items every one of them reads ``0 == 0`` — so an empty or
    namespace-mismatched null set would satisfy the whole block and certify a measurement that
    searched nothing ([[clauses-must-guard-emptiness]]). The denominators are therefore checked
    before the properties are, and a zero is a refusal, not a pass.
    """
    n = len(items)
    if n == 0:
        raise NoiseFloorError(
            "matchedness over 0 gated negatives: every counting control would read 0 == 0 and "
            "the block would certify a null set that was never built"
        )
    empty = sorted(arm for arm in ARMS if not records.get(arm))
    if empty:
        raise NoiseFloorError(f"these null arms carry no sequences: {empty}")
    minted_set = {minted[str(item["contig_id"])] for item in items}
    checks: list[dict[str, Any]] = []

    checks.append(
        hs.control_entry(
            "n1_covers_the_gated_negatives",
            "the insert arm is one record per gated negative, in benchmark order",
            [n, True],
            [
                len(records["n1_inserts"]),
                [entry["contig_id"] for entry in idmap["n1_inserts"].values()]
                == [str(item["contig_id"]) for item in items],
            ],
        )
    )

    lengths_ok = 0
    insert_once = insert_at_phase = outside_agrees = 0
    arms = ("n2_scramble1", "n2_scramble2")
    for arm in arms:
        for name, meta in idmap[arm].items():
            contig = minted[str(meta["contig_id"])]
            scramble = records[arm][name]
            insert_nt, moved = int(meta["insert_nt"]), int(meta["scramble_phase"])
            phase = int(meta["minted_phase"])
            lengths_ok += len(scramble) == len(contig)
            insert = scramble[moved : moved + insert_nt]
            insert_once += scramble.count(insert) == 1
            insert_at_phase += insert == contig[phase : phase + insert_nt]
            window = set(range(phase, phase + insert_nt)) | set(range(moved, moved + insert_nt))
            outside_agrees += all(
                contig[position] == scramble[position]
                for position in range(len(contig))
                if position not in window
            )
    total = n * len(arms)
    checks.append(
        hs.control_entry(
            "n2_length_matches_the_minted_contig",
            "a scramble is the same length as the contig it is a null for",
            total,
            lengths_ok,
        )
    )
    checks.append(
        hs.control_entry(
            "n2_carries_its_insert_exactly_once",
            "the same insert, one copy — a second copy would make the arm a positive",
            total,
            insert_once,
        )
    )
    checks.append(
        hs.control_entry(
            "n2_insert_is_the_minted_insert",
            "the scramble re-splices the SAME insert, not a different decoy",
            total,
            insert_at_phase,
        )
    )
    checks.append(
        hs.control_entry(
            "n2_agrees_outside_both_insert_windows",
            "same host: the only difference is where the junction falls",
            total,
            outside_agrees,
            note=(
                "the property an equal-length OVERWRITE implies; excising the insert from both "
                "and comparing remainders is the wrong test and fails on every item"
            ),
        )
    )
    checks.append(
        hs.control_entry(
            "n2_reproduces_no_minted_contig",
            "no scramble equals any minted contig",
            0,
            sum(1 for arm in arms for sequence in records[arm].values() if sequence in minted_set),
        )
    )
    shifts = [
        abs(int(meta["scramble_phase"]) - int(meta["minted_phase"]))
        for arm in arms
        for meta in idmap[arm].values()
    ]
    checks.append(
        hs.control_entry(
            "n2_phase_moved_far_enough",
            f"every junction moved ≥ {MIN_PHASE_SHIFT} nt",
            True,
            bool(shifts) and min(shifts) >= MIN_PHASE_SHIFT,
            note=f"min observed shift {min(shifts) if shifts else None} nt",
        )
    )
    taken = {str(value) for value in drawn}
    checks.append(
        hs.control_entry(
            "n3_is_disjoint_from_the_drawn_hosts",
            "the host arm shares no window with the benchmark it calibrates",
            0,
            len({str(meta["host_id"]) for meta in idmap["n3_pool"].values()} & taken),
        )
    )
    checks.append(
        hs.control_entry(
            "n3_is_the_requested_size",
            "the host arm was not silently truncated",
            N_POOL_WINDOWS,
            len(records["n3_pool"]),
        )
    )
    return checks


# ══════════════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════════════
def best_by_target(text: str) -> dict[str, float]:
    """Per-target best **bit** score, from the tblout's score column.

    Never the ``inc`` column (an E-value-derived inclusion call), which is A15 Pin 2's rule and
    is what makes this measurement commensurable with the assertion it calibrates.
    """
    best: dict[str, float] = {}
    for hit in hs.parse_tblout_hits(text):
        score = float(hit.score)
        current = best.get(hit.target)
        if current is None or score > current:
            best[hit.target] = score
    return best


def cell_stats(best: Mapping[str, float], n: int) -> dict[str, Any]:
    """One (arm, model, pipeline) cell.

    Quantiles are over the ``n`` **sequences searched**, not over the hits: a sequence with no
    reported hit is a measurement of "below the floor", and dropping it would compute the
    distribution of hits conditional on hitting — a much higher floor than the one being asked
    about. Unreported sequences are padded at 0.0, which is why ``-T 0`` matters.
    """
    if n <= 0:
        raise NoiseFloorError("a cell with no sequences cannot carry a distribution")
    if len(best) > n:
        raise NoiseFloorError(f"{len(best)} targets reported from {n} sequences")
    scores = sorted(best.values(), reverse=True)
    padded = scores + [0.0] * (n - len(scores))
    at_or_above = [value for value in scores if value >= RATE_AT_BITS]

    def quantile(fraction: float) -> float:
        return padded[min(int(fraction * n), n - 1)]

    return {
        "n": n,
        "n_with_hit": len(scores),
        f"n_ge_{RATE_AT_BITS:g}": len(at_or_above),
        f"rate_ge_{RATE_AT_BITS:g}": len(at_or_above) / n,
        "max_bits": scores[0] if scores else None,
        "p99_bits": quantile(0.01),
        "p95_bits": quantile(0.05),
        "median_bits": statistics.median(padded),
        f"scores_ge_{RATE_AT_BITS:g}": [round(value, 1) for value in at_or_above],
    }


#: A sequence carrying ``GGTGGNACCGCG`` on its **forward** strand, and its reverse complement,
#: which carries the motif only on the **reverse** strand. Both are positive controls for the
#: motif arm: ``0 of N`` is the answer a motif scan returns when the motif is genuinely absent
#: *and* when the scan never looked — an empty record set, a namespace that does not intersect
#: the sequences, a regex that cannot match ([[namespace-mismatch-invisible-noop]]) — so the
#: measured zero is only publishable beside a detection that must be non-zero.
MOTIF_CONTROL_FORWARD = "AAAA" + "GGTGGCACCGCG" + "AAAA"

#: The names the motif arm's positive controls carry, in order. The validator requires exactly
#: these: a report carrying two copies of the forward control reads to a human as independent
#: both-strand evidence and is not.
MOTIF_CONTROL_NAMES: tuple[str, ...] = (
    "motif_scan_detects_a_planted_motif_forward",
    "motif_scan_finds_a_reverse_only_motif_the_forward_scan_misses",
)


def motif_counts(records: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Per arm, the number of null sequences carrying ``GGTGGNACCGCG`` on either strand.

    Reported with its denominator, never as a bare count: a rate needs both, and a zero without
    a denominator cannot be told from a scan that ran over nothing.

    The second control is **strand-asymmetric on purpose.** ``hs.motif_present`` is itself a
    both-strand scan, so ``motif_present(x) == motif_present(revcomp(x))`` for every ``x`` — a
    "reverse" control that only reverse-complements its subject is the forward control twice,
    and fires identically with the reverse leg dead. The control therefore pairs the
    both-strand verdict with a **forward-only** scan of the same bytes and requires them to
    *disagree*: ``[True, False]``. Only a live reverse leg produces that.
    """
    per_arm = {
        arm: {
            "n": len(sequences),
            "n_motif": sum(1 for value in sequences.values() if hs.motif_present(value)),
        }
        for arm, sequences in records.items()
    }
    total_n = sum(block["n"] for block in per_arm.values())
    total_hit = sum(block["n_motif"] for block in per_arm.values())
    reverse_only = wd.reverse_complement(MOTIF_CONTROL_FORWARD)
    controls = [
        hs.control_entry(
            MOTIF_CONTROL_NAMES[0],
            "the motif scan is live on the forward strand",
            [True, True],
            [
                hs.motif_present(MOTIF_CONTROL_FORWARD),
                bool(hs.MOTIF_RE.search(hs.normalise_dna(MOTIF_CONTROL_FORWARD))),
            ],
            note="both-strand verdict, then a forward-only scan of the same bytes",
        ),
        hs.control_entry(
            MOTIF_CONTROL_NAMES[1],
            "the motif scan is live on the reverse strand, which a forward-only scan cannot be",
            [True, False],
            [
                hs.motif_present(reverse_only),
                bool(hs.MOTIF_RE.search(hs.normalise_dna(reverse_only))),
            ],
            note=(
                "the asymmetry IS the control: motif_present is both-strand, so a subject that "
                "merely reverse-complements the forward control fires identically with the "
                "reverse leg dead. Requiring the forward-only scan to MISS it does not."
            ),
        ),
    ]
    return {
        "motif": hs.MOTIF,
        "per_arm": per_arm,
        "n": total_n,
        "n_motif": total_hit,
        "rate": total_hit / total_n if total_n else None,
        "positive_controls": controls,
    }


def wilson_interval(k: int, n: int, *, z: float = 1.959963985) -> list[float]:
    """The 95 % Wilson interval for ``k / n``. Wilson, not Wald: at these rates ``k`` is small
    enough that a normal interval runs below zero and would read as a measurement."""
    if n <= 0:
        raise NoiseFloorError("an interval needs a denominator")
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - spread), min(1.0, centre + spread)]


def null_maxima(cells: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, float]:
    """Per model, the worst score any null sequence reached, over **every** arm and pipeline.

    The most conservative reading available, and the one ``host_screen`` binds its thresholds
    to: a threshold that clears the single worst sequence out of thousands containing no T-box
    is a threshold outside the noise, whichever channel produced that sequence.
    """
    maxima: dict[str, float] = {}
    for model, by_arm in cells.items():
        values = [
            float(cell["max_bits"])
            for by_pipeline in by_arm.values()
            for cell in by_pipeline.values()
            if cell.get("max_bits") is not None
        ]
        if not values:
            raise NoiseFloorError(f"{model}: every cell is empty, so there is no measured floor")
        maxima[model] = max(values)
    return maxima


# ══════════════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════════════
def _as_number(value: Any) -> float | None:
    """A finite number, or ``None``. A ``bool`` is not a number: ``float(True)`` is 1.0 and would
    order against a comparison like any other value ([[bool-equals-one-passes-config-gates]])."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """This artifact's clause set, re-derived from its own content.

    Returns problems rather than raising, which is the house convention: a producer that raises
    writes nothing, and an artifact that cannot be inspected cannot be adjudicated.
    """
    problems: list[str] = []

    def want(condition: Any, message: str) -> None:
        if not condition:
            problems.append(message)

    want(report.get("schema_version") == SCHEMA_VERSION, f"schema_version is not {SCHEMA_VERSION}")
    want(report.get("kind") == KIND, f"kind is not {KIND!r}")
    want(report.get("step") == STEP, f"step is not {STEP!r}")
    want("ADR-0005 A15" in str(report.get("adr")), "adr does not name ADR-0005 A15")
    want(
        report.get("search_threshold_bits") == REPORT_THRESHOLD_BITS,
        f"search_threshold_bits is not {REPORT_THRESHOLD_BITS} — a thresholded search cannot "
        "say where the floor is",
    )

    # The construction values, bound rather than assumed. Without these a report produced at a
    # different seed, a different phase-shift rule or a different reporting cut validates clean
    # while POSTMINT_THRESHOLD_BITS is signed against it ([[gate-must-bind-to-upstream-evidence]]).
    for field, pinned in (
        ("seed", SEED),
        ("rate_at_bits", RATE_AT_BITS),
        ("min_phase_shift_nt", MIN_PHASE_SHIFT),
        ("n_pool_windows", N_POOL_WINDOWS),
    ):
        if field in report:
            want(report.get(field) == pinned, f"{field} is {report.get(field)!r}, not {pinned!r}")

    controls = report.get("matchedness")
    if not isinstance(controls, Sequence) or isinstance(controls, str) or not controls:
        problems.append("matchedness is missing — the arms are not shown to be controls")
    else:
        # Exactly the nine, by name: re-deriving only the rows present cannot see a dropped one.
        named = [str(entry.get("name")) for entry in controls if isinstance(entry, Mapping)]
        missing = [name for name in MATCHEDNESS_CONTROLS if name not in named]
        want(
            not missing,
            f"matchedness is missing {len(missing)} of its {len(MATCHEDNESS_CONTROLS)} "
            f"controls: {missing}",
        )
        want(
            len(named) == len(set(named)),
            "matchedness carries a duplicated control name",
        )
        for entry in controls:
            if not isinstance(entry, Mapping):
                problems.append("a matchedness entry is not an object")
                continue
            # Re-derived, not read: `fired` written independently of its own measurement is a
            # report field rather than a control.
            want(
                entry.get("fired") == (entry.get("observed") == entry.get("expected")),
                f"matchedness {entry.get('name')!r}: fired is not observed == expected",
            )
            want(
                bool(entry.get("fired")),
                f"matchedness {entry.get('name')!r} failed: expected "
                f"{entry.get('expected')!r}, observed {entry.get('observed')!r}",
            )

    motif = report.get("motif")
    if not isinstance(motif, Mapping) or not motif:
        problems.append("motif is missing — the CM-free rule was not measured on the nulls")
    else:
        want(motif.get("motif") == hs.MOTIF, f"motif.motif is not {hs.MOTIF!r}")
        per_arm = motif.get("per_arm")
        if not isinstance(per_arm, Mapping) or set(per_arm) != set(ARMS):
            problems.append("motif.per_arm does not cover exactly the four null arms")
        else:
            want(
                motif.get("n") == sum(int(block["n"]) for block in per_arm.values()),
                "motif.n is not the sum of its arms",
            )
            want(
                motif.get("n_motif") == sum(int(block["n_motif"]) for block in per_arm.values()),
                "motif.n_motif is not the sum of its arms",
            )
            want(
                all(int(block["n"]) > 0 for block in per_arm.values()),
                "a motif arm has no sequences — 0 of 0 is not a measurement",
            )
        # Without these, `n_motif: 0` is equally the answer when the scan never ran.
        controls = motif.get("positive_controls")
        if not isinstance(controls, Sequence) or isinstance(controls, str) or len(controls) < 2:
            problems.append("motif.positive_controls is missing both strands")
        else:
            # By NAME: `motif_present` is a both-strand scan, so two copies of the forward
            # control are indistinguishable from a genuine pair and read to a human as
            # independent both-strand evidence. The reverse control is the one whose expected
            # value is asymmetric ([True, False]); requiring its name requires that asymmetry.
            got = [str(entry.get("name")) for entry in controls if isinstance(entry, Mapping)]
            absent = [name for name in MOTIF_CONTROL_NAMES if name not in got]
            want(
                not absent,
                f"motif.positive_controls does not carry {absent} — a forward control twice is "
                "not a both-strand control",
            )
            for entry in controls:
                want(
                    isinstance(entry, Mapping)
                    and entry.get("fired") == (entry.get("observed") == entry.get("expected"))
                    and bool(entry.get("fired")),
                    f"motif positive control {dict(entry).get('name')!r} did not fire: the "
                    "measured zero cannot be told from a scan that never looked",
                )

    cells = report.get("cells")
    if not isinstance(cells, Mapping) or not cells:
        problems.append("cells is missing")
        return problems

    for model, by_arm in sorted(cells.items()):
        if isinstance(by_arm, Mapping):
            # Extra arms are refused, not ignored: null_maxima maxes over whatever is here, so
            # an invented arm would feed the published floor.
            extra = sorted(set(by_arm) - set(ARMS))
            want(not extra, f"cells.{model} carries arms that are not null arms: {extra}")
        for arm in ARMS:
            if arm not in by_arm:
                problems.append(f"cells.{model}.{arm} is missing — an unsearched null arm")
                continue
            for pipeline, _ in PIPELINES:
                cell = by_arm[arm].get(pipeline) if isinstance(by_arm[arm], Mapping) else None
                if not isinstance(cell, Mapping):
                    problems.append(f"cells.{model}.{arm}.{pipeline} is missing")
                    continue
                n, hit = cell.get("n"), cell.get("n_with_hit")
                ge = cell.get(f"n_ge_{RATE_AT_BITS:g}")
                rate = cell.get(f"rate_ge_{RATE_AT_BITS:g}")
                if not isinstance(n, int) or n <= 0:
                    problems.append(f"cells.{model}.{arm}.{pipeline}.n is not positive")
                    continue
                want(
                    isinstance(hit, int) and 0 <= hit <= n,
                    f"cells.{model}.{arm}.{pipeline}.n_with_hit is outside [0, n]",
                )
                want(
                    isinstance(ge, int) and 0 <= ge <= (hit if isinstance(hit, int) else n),
                    f"cells.{model}.{arm}.{pipeline}.n_ge_{RATE_AT_BITS:g} exceeds n_with_hit",
                )
                if isinstance(ge, int) and isinstance(rate, (int, float)):
                    want(
                        abs(float(rate) - ge / n) < 1e-12,
                        f"cells.{model}.{arm}.{pipeline}.rate is not n_ge / n",
                    )
                top = _as_number(cell.get("max_bits"))
                for field in ("p99_bits", "p95_bits", "median_bits"):
                    value = _as_number(cell.get(field))
                    if value is None:
                        problems.append(f"cells.{model}.{arm}.{pipeline}.{field} is missing")
                    elif top is not None:
                        want(
                            value <= top,
                            f"cells.{model}.{arm}.{pipeline}.{field} ({value}) exceeds the "
                            f"cell's own max_bits ({top})",
                        )
                scores = cell.get(f"scores_ge_{RATE_AT_BITS:g}")
                if isinstance(scores, str):
                    problems.append(
                        f"cells.{model}.{arm}.{pipeline}.scores_ge_{RATE_AT_BITS:g} is a string; "
                        "a string is a Sequence and would be silently skipped"
                    )
                elif isinstance(scores, Sequence):
                    want(
                        len(scores) == ge,
                        f"cells.{model}.{arm}.{pipeline} lists {len(scores)} scores for "
                        f"{ge} hits at ≥ {RATE_AT_BITS:g} bits",
                    )
                    want(
                        all(float(value) >= RATE_AT_BITS for value in scores),
                        f"cells.{model}.{arm}.{pipeline} lists a score below "
                        f"{RATE_AT_BITS:g} bits among those at or above it",
                    )

    floor = report.get("null_max_bits")
    if not isinstance(floor, Mapping) or not floor:
        problems.append("null_max_bits is missing — nothing for a threshold to be signed against")
    else:
        want(set(floor) == set(cells), "null_max_bits does not cover exactly the searched models")
        for model, value in sorted(floor.items()):
            observed = [
                float(cell["max_bits"])
                for by_pipeline in cells.get(model, {}).values()
                if isinstance(by_pipeline, Mapping)
                for cell in by_pipeline.values()
                if isinstance(cell, Mapping) and cell.get("max_bits") is not None
            ]
            if not observed:
                problems.append(f"null_max_bits.{model} rests on no non-empty cell")
            else:
                want(
                    abs(float(value) - max(observed)) < 1e-12,
                    f"null_max_bits.{model} is {value}, not the measured maximum {max(observed)}",
                )
    counts = report.get("n_null_sequences")
    if not isinstance(counts, Mapping) or not counts:
        problems.append("n_null_sequences is missing")
    else:
        want(set(counts) == set(cells), "n_null_sequences does not cover exactly the models")
        for model, value in sorted(counts.items()):
            want(
                isinstance(value, int) and value > 0,
                f"n_null_sequences.{model} is not positive",
            )
            # Re-derived from the cells: the figure is quoted BESIDE an extreme-value maximum
            # precisely because the maximum rises with it, so a wrong denominator understates
            # how hard the floor was looked for. One `n` per arm, not one per search.
            by_arm = cells.get(model)
            if isinstance(by_arm, Mapping):
                derived = 0
                for arm_block in by_arm.values():
                    if not isinstance(arm_block, Mapping):
                        continue
                    sizes = {
                        _as_number(cell.get("n"))
                        for cell in arm_block.values()
                        if isinstance(cell, Mapping)
                    }
                    if len(sizes) > 1:
                        problems.append(
                            f"cells.{model} has one arm searched at two different n: {sizes}"
                        )
                    derived += int(next(iter(sizes))) if len(sizes) == 1 else 0
                want(
                    value == derived,
                    f"n_null_sequences.{model} is {value!r}, not the {derived} its own cells "
                    "report (one n per arm, not one per search)",
                )
    return problems


# ══════════════════════════════════════════════════════════════════════════════════════
# Legs
# ══════════════════════════════════════════════════════════════════════════════════════
def _cmd_prep(args: argparse.Namespace) -> int:
    """Build the four null FASTAs and assert every matchedness property on the bytes written."""
    started = time.monotonic()
    root = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]
    for value, what in (
        (args.benchmark, "--benchmark"),
        (args.admitted, "--admitted"),
        (args.mining_pool, "--mining-pool"),
    ):
        hs.require_repo_relative(value, what=what)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    items = gated_negatives(benchmark)
    screen_dir = Path(args.screen_dir)
    vmap = json.loads((screen_dir / "postmint_idmap.json").read_text(encoding="utf-8"))["contigs"]
    minted_fasta = read_fasta(screen_dir / hs.FASTA_POSTMINT_LABEL0)
    minted = {
        str(meta["contig_id"]): minted_fasta[name]
        for name, meta in vmap.items()
        if name in minted_fasta
    }

    candidates, _pool_report = hs.load_candidates(root, args.mining_pool)
    rows = {str(row["candidate_id"]): hs.normalise_dna(row["sequence"]) for row in candidates}
    drawn = {str(item["host_id"]) for item in benchmark["items"] if int(item.get("label", 1)) == 0}
    admitted = [
        line.strip()
        for line in Path(args.admitted).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool_windows = choose_pool_windows(admitted, drawn, rows)

    records, built = build_nulls(items, minted, rows, pool_windows)
    checks = matchedness(items, minted, records, built["idmap"], drawn)
    digests = {arm: write_fasta(out_dir / f"{arm}.fa", records[arm]) for arm in ARMS}
    prep = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": f"{KIND}_prep",
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "seed": SEED,
        "min_phase_shift_nt": MIN_PHASE_SHIFT,
        "n_pool_windows": N_POOL_WINDOWS,
        "benchmark": {
            "path": hs.recorded_path(Path(args.benchmark), root),
            "sha256": provenance.sha256_file(Path(args.benchmark)),
            "schema_version": benchmark.get("schema_version"),
        },
        "arms": {
            arm: {
                "fasta": f"{arm}.fa",
                "sha256": digests[arm],
                "n": len(records[arm]),
                "nt_min": min(len(value) for value in records[arm].values()),
                "nt_max": max(len(value) for value in records[arm].values()),
            }
            for arm in ARMS
        },
        "idmap": built["idmap"],
        "matchedness": checks,
        "wall_s": round(time.monotonic() - started, 2),
    }
    failed = [entry["name"] for entry in checks if not entry["fired"]]
    (out_dir / PREP_JSON).write_text(json.dumps(prep, indent=1, ensure_ascii=False), "utf-8")
    print(
        "noise-floor prep: "
        + ", ".join(f"{arm} n={len(records[arm])}" for arm in ARMS)
        + f" -> {out_dir}"
    )
    if failed:
        print(f"noise-floor prep: {len(failed)} matchedness control(s) failed: {failed}")
        return 1
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """FASTA × model × pipeline, at ``-T 0``. The infernal-env leg."""
    started = time.monotonic()
    root = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir)
    tblout_root = Path(args.tblout_dir)
    prep = json.loads((out_dir / PREP_JSON).read_text(encoding="utf-8"))
    tasks: list[tuple[dict[str, Any], str, str]] = []
    for arm in ARMS:
        expected = prep["arms"][arm]["sha256"]
        observed = provenance.sha256_file(out_dir / f"{arm}.fa")
        if observed != expected:
            raise NoiseFloorError(
                f"{arm}.fa has sha256 {observed}, prep recorded {expected} — the tblouts would "
                "not describe the FASTA prep wrote"
            )
        for model, cm in hs.SCREEN_CMS:
            for pipeline, no_filters in PIPELINES:
                tasks.append(
                    (
                        {
                            "root": root,
                            "out_dir": out_dir,
                            "tblout_root": tblout_root,
                            "arm_dir": f"{pipeline}/",
                            "basename": f"{arm}.fa",
                            "model": model,
                            "cm": cm,
                            # -T 0, always. A null searched at the threshold it is meant to
                            # calibrate reports nothing below that threshold, and the measured
                            # maximum becomes an artefact of the flag rather than a floor.
                            "threshold_bits": REPORT_THRESHOLD_BITS,
                            "cpu": int(args.cpu),
                            "no_filters": no_filters,
                            "timeout_s": float(args.timeout_s),
                        },
                        arm,
                        pipeline,
                    )
                )

    # ``jobs`` at a time, collected in TASK order so search.json does not depend on which search
    # finished first; progress prints in completion order, because a multi-hour run that says
    # nothing is indistinguishable from a hung one. The house shape, from host_screen's own
    # search leg. The whole run is abandoned on the first failure.
    searches: list[dict[str, Any]] = []
    if int(args.jobs) > 1:
        import concurrent.futures as futures

        with futures.ThreadPoolExecutor(max_workers=int(args.jobs)) as pool:
            pending = [pool.submit(lambda kw=kw: hs._search_one(**kw)) for kw, _, _ in tasks]
            for future in pending:
                future.add_done_callback(
                    lambda f: (
                        print(f"  searched {f.result()['tblout']}", flush=True)
                        if not f.cancelled() and f.exception() is None
                        else None
                    )
                )
            done, _ = futures.wait(pending, return_when=futures.FIRST_EXCEPTION)
            failed = next((f for f in pending if f in done and f.exception() is not None), None)
            if failed is not None:
                for future in pending:
                    future.cancel()
                raise failed.exception()
            results = [future.result() for future in pending]
    else:
        results = []
        for kw, _, _ in tasks:
            results.append(hs._search_one(**kw))
            print(f"  searched {results[-1]['tblout']}", flush=True)
    for record, (_, arm, pipeline) in zip(results, tasks, strict=True):
        searches.append(record | {"null_arm": arm, "pipeline": pipeline})
    record = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": f"{KIND}_search",
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "search_threshold_bits": REPORT_THRESHOLD_BITS,
        "search_space_mb": hs.SEARCH_SPACE_MB,
        "infernal_version": hs.infernal_version_line(),
        "env_lock": hs.ENV_LOCK,
        "env_lock_sha256": provenance.sha256_file(root / hs.ENV_LOCK),
        "cms": {
            model: {"path": cm, "sha256": provenance.sha256_file(root / cm)}
            for model, cm in hs.SCREEN_CMS
        },
        "searches": searches,
        "wall_s": round(time.monotonic() - started, 2),
    }
    tblout_root.mkdir(parents=True, exist_ok=True)
    (tblout_root / SEARCH_JSON).write_text(
        json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"noise-floor search: {len(searches)} searches -> {tblout_root}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Per-cell distributions, the per-model floor, and the matchedness re-derivation."""
    started = time.monotonic()
    root = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir)
    tblout_root = Path(args.tblout_dir)
    prep = json.loads((out_dir / PREP_JSON).read_text(encoding="utf-8"))
    search = json.loads((tblout_root / SEARCH_JSON).read_text(encoding="utf-8"))

    # Re-derived from the FASTAs that were SEARCHED, not from prep's own record of them: a
    # matchedness block copied forward from prep would certify bytes nobody re-read.
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    items = gated_negatives(benchmark)
    screen_dir = Path(args.screen_dir)
    vmap = json.loads((screen_dir / "postmint_idmap.json").read_text(encoding="utf-8"))["contigs"]
    minted_fasta = read_fasta(screen_dir / hs.FASTA_POSTMINT_LABEL0)
    minted = {
        str(meta["contig_id"]): minted_fasta[name]
        for name, meta in vmap.items()
        if name in minted_fasta
    }
    records = {arm: read_fasta(out_dir / f"{arm}.fa") for arm in ARMS}
    drawn = {str(item["host_id"]) for item in benchmark["items"] if int(item.get("label", 1)) == 0}
    checks = matchedness(items, minted, records, prep["idmap"], drawn)

    # The tblout digest binds a tblout to its own recorded rows; it says nothing about WHICH
    # FASTA produced them. `_cmd_search` checks the FASTA against prep before searching, but the
    # report is a separate run: a prep re-run between them would leave the old tblouts in place,
    # and the ids are positional, so the cells would be computed from one run's hits over another
    # run's sequences while the matchedness block below was re-derived on the NEW bytes.
    current = {arm: provenance.sha256_file(out_dir / f"{arm}.fa") for arm in ARMS}
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in search["searches"]:
        recorded, now = entry.get("fasta_sha256"), current.get(str(entry["null_arm"]))
        if recorded != now:
            raise NoiseFloorError(
                f"{entry['null_arm']}.fa has sha256 {now!r}; the search ran on {recorded!r}. "
                "Refusing to measure a floor from one run's tblouts over another run's nulls."
            )
        text = (tblout_root / entry["tblout"]).read_text(encoding="utf-8")
        if hs.tblout_digest(text) != entry["data_sha256"]:
            raise NoiseFloorError(f"{entry['tblout']}: data rows differ from the search record")
        model, arm, pipeline = entry["model"], entry["null_arm"], entry["pipeline"]
        n = len(records[arm])
        cells.setdefault(model, {}).setdefault(arm, {})[pipeline] = cell_stats(
            best_by_target(text), n
        )
    # Distinct null SEQUENCES per model, counted once per arm however many pipelines searched
    # it. Summing per search would report 6,146 where 3,073 were searched, and the figure is
    # quoted beside an extreme-value maximum, which rises with the number of sequences.
    counts = {model: sum(len(records[arm]) for arm in by_arm) for model, by_arm in cells.items()}

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "kind": KIND,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "purpose": (
            "the matched-null floor A15 Pin 3's per-model assertion thresholds are signed "
            "against; host_screen.POSTMINT_THRESHOLD_BITS must clear null_max_bits"
        ),
        "seed": SEED,
        "search_threshold_bits": REPORT_THRESHOLD_BITS,
        "rate_at_bits": RATE_AT_BITS,
        "search_space_mb": search["search_space_mb"],
        "infernal_version": search["infernal_version"],
        "env_lock": search["env_lock"],
        "env_lock_sha256": search["env_lock_sha256"],
        "models": search["cms"],
        "benchmark": prep["benchmark"],
        "arms": {
            arm: {
                **{k: v for k, v in prep["arms"][arm].items() if k != "fasta"},
                "construction": _ARM_PROSE[arm],
            }
            for arm in ARMS
        },
        "matchedness": checks,
        "motif": motif_counts(records),
        "cells": cells,
        "null_max_bits": null_maxima(cells),
        "n_null_sequences": counts,
        "pooled": _pooled(cells),
        "limitations": [
            "n3_pool is conditioned on having PASSED the Pin 2 screen, so it measures the "
            "residual admission leaves and cannot estimate Pin 2's own false-refusal rate.",
            "n2 carries the same host and the same insert as the contig it is a null for, so it "
            "cannot separate noise from genuine homology present in BOTH; n3, which carries no "
            "insert at all, is what bounds the host channel.",
            "A null maximum is an extreme-value statistic: it rises with the number of sequences "
            "searched, so it is reported beside n_null_sequences and never as a fixed property.",
        ],
        "wall_s": round(time.monotonic() - started, 2),
    }
    report["provenance"] = hs.provenance_record(
        leg="noise-floor-report",
        inputs=[out_dir / PREP_JSON, tblout_root / SEARCH_JSON],
        root=root,
        env_lock=root / hs.ENV_LOCK_DATA,
    )
    report["problems"] = validate_report(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    floor = ", ".join(
        f"{model} {value:.1f}" for model, value in sorted(report["null_max_bits"].items())
    )
    print(f"noise-floor report: null max bits {floor} -> {out}")
    if report["problems"]:
        print(f"noise-floor report: {len(report['problems'])} problem(s)")
        for problem in report["problems"]:
            print(f"  - {problem}")
        return 1
    return 0


_ARM_PROSE: dict[str, str] = {
    "n1_inserts": (
        "the decoys_v0 inserts actually spliced into the gated negatives, searched alone — the "
        "insert channel, which no per-host pre-draw screen covers"
    ),
    "n2_scramble1": (
        "the same hosts re-spliced with the same inserts at a phase moved ≥ "
        f"{MIN_PHASE_SHIFT} nt (draw 1) — the junction channel, and the arm matched to the "
        "minted contigs in every respect but where the junction falls"
    ),
    "n2_scramble2": "as n2_scramble1, an independent second phase draw",
    "n3_pool": (
        f"{N_POOL_WINDOWS} screen-admitted windows the seeded draw did not take, pre-splice — "
        "the host channel, i.e. what Pin 2 admission leaves behind"
    ),
}


def _pooled(cells: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Per model and pipeline, the rate pooled over all four arms, with a Wilson interval."""
    pooled: dict[str, Any] = {}
    key = f"n_ge_{RATE_AT_BITS:g}"
    for model, by_arm in cells.items():
        for pipeline, _ in PIPELINES:
            k = sum(int(by_arm[arm][pipeline][key]) for arm in by_arm if pipeline in by_arm[arm])
            n = sum(int(by_arm[arm][pipeline]["n"]) for arm in by_arm if pipeline in by_arm[arm])
            if n <= 0:
                continue
            pooled.setdefault(model, {})[pipeline] = {
                "n": n,
                key: k,
                f"rate_ge_{RATE_AT_BITS:g}": k / n,
                "wilson95": wilson_interval(k, n),
                "max_bits": max(
                    (
                        float(by_arm[arm][pipeline]["max_bits"])
                        for arm in by_arm
                        if pipeline in by_arm[arm]
                        and by_arm[arm][pipeline].get("max_bits") is not None
                    ),
                    default=None,
                ),
            }
    return pooled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="leg", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
        target.add_argument("--data-root", default=None)

    prep = sub.add_parser("prep", help="data env: build the four null FASTAs")
    common(prep)
    prep.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    prep.add_argument("--screen-dir", default=DEFAULT_SCREEN_DIR)
    prep.add_argument("--admitted", default=DEFAULT_ADMITTED)
    prep.add_argument("--mining-pool", default=DEFAULT_MINING_POOL)
    prep.set_defaults(func=_cmd_prep)

    search = sub.add_parser("search", help="infernal env: FASTA x model x pipeline at -T 0")
    common(search)
    search.add_argument("--tblout-dir", default=None)
    search.add_argument("--cpu", type=int, default=6)
    search.add_argument("--jobs", type=int, default=1)
    search.add_argument("--timeout-s", type=float, default=7200.0)
    search.set_defaults(func=_cmd_search)

    report = sub.add_parser("report", help="data env: the per-model measured floor")
    common(report)
    report.add_argument("--tblout-dir", default=None)
    report.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    report.add_argument("--screen-dir", default=DEFAULT_SCREEN_DIR)
    report.add_argument("--out", default=DEFAULT_REPORT)
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "tblout_dir", None) is None and hasattr(args, "out_dir"):
        args.tblout_dir = str(Path(args.out_dir) / "tblout")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ADR",
    "ARMS",
    "GENERATED_BY",
    "KIND",
    "MATCHEDNESS_CONTROLS",
    "MIN_PHASE_SHIFT",
    "MOTIF_CONTROL_FORWARD",
    "MOTIF_CONTROL_NAMES",
    "MATCHEDNESS_CONTROLS",
    "N_POOL_WINDOWS",
    "PIPELINES",
    "RATE_AT_BITS",
    "REPORT_THRESHOLD_BITS",
    "SCHEMA_VERSION",
    "SEED",
    "STEP",
    "NoiseFloorError",
    "best_by_target",
    "build_nulls",
    "build_parser",
    "cell_stats",
    "choose_pool_windows",
    "gated_negatives",
    "main",
    "matchedness",
    "motif_counts",
    "null_maxima",
    "read_fasta",
    "scramble_phase",
    "splice",
    "validate_report",
    "wilson_interval",
    "write_fasta",
]
