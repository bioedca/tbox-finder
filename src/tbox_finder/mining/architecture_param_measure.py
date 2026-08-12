"""Measure criterion (b)'s seven rule parameters on the **real de-novo consensuses**.

`P3-15′-f` must supply seven ADR-0006 A4 rule parameters to
``architecture_producer run-shard``.  Six of them
(``min_named_helices``, ``min_helix_pairs``, ``bulge_min_nt``, ``bulge_max_nt``,
``ncca_pairing_nt``, ``allow_wobble``) decide which candidates the round mines;
the seventh (``stem_i_nt_threshold``) is D6's, supplied per round under A4.
**No value may be chosen from the curated freeze.**
``reports/p3/architecture_freeze.json`` disclaims itself for exactly this purpose:

    "measured on CURATED structure, which is CM-derived: this is the localizer's
    sensitivity on known architecture, NOT the rate at which a de-novo consensus
    resolves one.  The second number needs the full-corpus (a)/(b) supply run."

That run has now happened (SLURM jobs 1205 + 1254), so this module reads its
output — the ``msa/<slug>/msa.sto`` A3/A4 comparative consensuses — and reports
what the **de-novo** instrument actually resolves, at every parameter value in an
explicit sweep.

Three disciplines this module holds to, each learned the hard way in this repo:

* **It calls the shipped localizer, never a copy.**  Every number here comes out
  of :mod:`tbox_finder.mining.architecture` — ``parse_stockholm``,
  ``find_helices``, ``find_bulges``, ``degapped_span``, ``named_elements_status``,
  ``ncca_bulge_status``, ``localize``, ``architecture_status``.  A second
  implementation of "what a helix is" would let the measurement and the producer
  disagree about the very thing the measurement exists to parameterise.
* **It pins nothing.**  ``pins_nothing: true``, as in the freeze.  It reports
  distributions; the §7 choice is the user's and lands in the round's provenance.
* **It reports what a parameter *cannot* decide.**  Two of the seven are inert
  here, and inertness that is not disclosed reads as a value that was chosen
  ([[pinned-constant-that-nothing-reads]]).  Both inertness claims below are
  proved from the shipped code's own semantics, not inferred from a sample.

Run::

    PYTHONPATH=src python -m tbox_finder.mining.architecture_param_measure measure \\
        --msa-root <dir of <slug>/msa.sto> \\
        --manifest data/processed/mining/round0_fp_manifest.json \\
        --covariation-status <covariation_status.json> \\
        --out reports/p3/architecture_parameter_measurement.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tbox_finder.mining import architecture as arch
from tbox_finder.mining.architecture import (
    ArchitectureError,
    ConsensusStructure,
    Localization,
    acceptor_pairing_motif,
    architecture_status,
    degapped_span,
    find_bulges,
    find_helices,
    named_elements_status,
    ncca_bulge_status,
    pair_table,
    parse_stockholm,
)
from tbox_finder.mining.covariation_producer import candidate_slug
from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.provenance import build_provenance

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-f"
ADR = (
    "ADR-0006 D3/D6 (criterion (b)), A4 (rule parameters supplied per round), "
    "D17 (P6 freeze map); ADR-0005 D14 (unavailable spares)"
)


# ═════════════════════════════════════════════════════════════════════════════
# Which supply is being measured
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SupplyArm:
    """The four self-descriptions a report makes about the corpus it measured.

    The sweep, the localizer calls and every count are **identical** across arms —
    that identity is the whole point of a matched control, so nothing here touches
    them.  What differs is what the report *says it measured*: `P3-15'-f` reads
    round-0 false-positive-manifest candidates of unknown status, `P3-15'-g-iv`
    reads Stage-1 spans on held-out **curated** records, which are believed
    positive.  Shipping the first arm's prose on the second arm's numbers would be
    a committed public report describing the wrong corpus — the failure mode
    [[gate-must-bind-to-upstream-evidence]] names, and one no internal check can
    see, because every count would still reconcile.

    ``ground_truth`` is the field the difference actually turns on, and it is the
    reason ``covariation_note`` cannot be shared: on the FP arm criterion (a) is
    the strongest control available *because* nothing in the supply is known
    positive; on the control arm the supply itself is the positive control and (a)
    becomes a second measurement rather than the anchor.
    """

    key: str
    step: str
    provenance_rule: str
    ground_truth: str
    canonical_out: str
    disclosure: str
    covariation_note: str
    no_positive_control_reason: str


#: ⚠ Keyed, frozen and exhaustive: an unknown ``--arm`` is refused rather than
#: defaulted, because the default's prose is *wrong* for every other supply and a
#: silent fallback would publish it anyway.
SUPPLY_ARMS: Mapping[str, SupplyArm] = {
    "round0_fp": SupplyArm(
        key="round0_fp",
        step=STEP,
        provenance_rule="P3-15'-f parameter measurement",
        ground_truth="unknown",
        canonical_out="reports/p3/architecture_parameter_measurement.json",
        disclosure=(
            "de-novo A3/A4 mlocarna consensuses of round-0 false-positive-manifest "
            "candidates. This is what the de-novo instrument RESOLVES, not what a "
            "curated CM-derived structure shows; reports/p3/architecture_freeze.json "
            "measures the latter and disclaims itself for this purpose."
        ),
        covariation_note=(
            "criterion (a) R-scape covariation on the same alignment is D3's own "
            "model-independent structural anchor; the share of (a)-passed candidates that "
            "(b) fails is the share it stops protecting despite independent covariation "
            "support. A control, not ground truth: (a) and (b) share their supply."
        ),
        no_positive_control_reason=(
            "no de-novo positive control was supplied or the path did not exist; the "
            "parameter choice below is bounded by the corpus and by criterion (a) alone"
        ),
    ),
    "curated_control": SupplyArm(
        key="curated_control",
        step="P3-15'-g-iv",
        provenance_rule="P3-15'-g-iv matched-control parameter measurement",
        ground_truth="believed_positive",
        canonical_out="reports/p3/architecture_parameter_measurement_control.json",
        disclosure=(
            "de-novo A3/A4 mlocarna consensuses of the MATCHED positive control: "
            "Stage-1 re-detected spans on held-out CURATED T-box records (P3-15'-g-ii, "
            "one record per ADR-0004 cluster, order-stratified), searched and aligned by "
            "the same instrument as the FP arm (P3-15'-g-iii, SLURM job 1264). The record "
            "is a known T-box; the query is the DETECTOR'S CALL on it (median 104 nt "
            "against a curated element median of 279, median best IoU 0.404), so every "
            "rate here is a rate on a partial element and not on the curated element. "
            "Read against reports/p3/architecture_parameter_measurement.json, whose grid "
            "is identical by construction (same module, same sweep)."
        ),
        covariation_note=(
            "criterion (a) R-scape covariation on the same alignment, measured here on a "
            "supply that is BELIEVED POSITIVE: on this arm the (a) split is (a)'s own "
            "sensitivity on known T-boxes, not a proxy for ground truth as it is on the "
            "FP arm. (a) and (b) still share their supply (ADR-0006 A4), so they are "
            "independent only in what they test on it."
        ),
        no_positive_control_reason=(
            "no de-novo positive control was supplied or the path did not exist; on this "
            "arm the corpus itself is the positive control, so the n=1 block is "
            "supplementary rather than load-bearing"
        ),
    ),
}

#: The arm whose prose the committed `P3-15'-f` report carries.  ``measure`` and the
#: CLI both default to it, so the FP report re-derives BYTE-IDENTICALLY across this
#: seam — the same discipline `covariation_producer.search_shard(query_fasta=None)`
#: holds to (P3-15'-g-iii).
DEFAULT_ARM_KEY = "round0_fp"
DEFAULT_ARM = DEFAULT_ARM_KEY

#: `--out`'s default: the committed FP report's path, i.e. the default arm's own
#: canonical output.  Every arm's canonical path is on its :class:`SupplyArm`, and the
#: CLI refuses to write one arm's numbers to ANOTHER arm's path — in both directions,
#: because `--arm round0_fp --out <the control report>` destroys a committed artifact
#: exactly as surely as the reverse, and that artifact is now an input to
#: `architecture_param_control_compare`.
DEFAULT_OUT = SUPPLY_ARMS[DEFAULT_ARM_KEY].canonical_out


def arm_for_step(step: str) -> SupplyArm:
    """The :class:`SupplyArm` whose ``step`` a measurement report carries.

    The inverse lookup, so a downstream reader can recover *which corpus* a report
    describes from the report itself instead of restating it.  Without this,
    ``ground_truth`` would be a constant nothing reads
    ([[pinned-constant-that-nothing-reads]]) while the comparison emitted its own
    independent spelling of the same fact — two statements that can disagree.
    """
    for candidate in SUPPLY_ARMS.values():
        if candidate.step == step:
            return candidate
    raise MeasureError(
        f"no supply arm declares step {step!r}; the report was not written by this "
        f"module's measure(). Known steps: {sorted(a.step for a in SUPPLY_ARMS.values())}"
    )


def resolve_arm(arm: str | SupplyArm | None) -> SupplyArm:
    """The :class:`SupplyArm` for ``arm``; ``None`` is :data:`DEFAULT_ARM`."""
    if isinstance(arm, SupplyArm):
        return arm
    key = DEFAULT_ARM if arm is None else str(arm)
    try:
        return SUPPLY_ARMS[key]
    except KeyError:
        raise MeasureError(
            f"unknown supply arm {key!r}; the report's disclosure, its (a) note and its "
            f"provenance rule all come from it, so it cannot be defaulted. "
            f"Known arms: {sorted(SUPPLY_ARMS)}"
        ) from None


#: The sweep axes.  These are **not** pins and **not** defaults — they are the
#: values the report tabulates so a reader can see the whole admissible surface.
#: ``min_named_helices`` is capped by the module's own
#: :data:`~tbox_finder.mining.architecture.MAX_NAMED_HELICES`, which refuses larger
#: values rather than making (b) unsatisfiable (D9 row 5 would route the entire
#: corpus to Tier-2N).
MIN_HELIX_PAIRS_SWEEP: tuple[int, ...] = (1, 2, 3, 4, 5)
MIN_NAMED_HELICES_SWEEP: tuple[int, ...] = tuple(range(1, arch.MAX_NAMED_HELICES + 1))
BULGE_MIN_NT_SWEEP: tuple[int, ...] = (1, 2, 3, 4, 5, 7)
BULGE_MAX_NT_SWEEP: tuple[int, ...] = (10, 20, 50, 10_000)
#: Bounded by the acceptor motif's own length — ``ncca_bulge_status`` raises above it.
NCCA_PAIRING_NT_SWEEP: tuple[int, ...] = tuple(range(1, len(acceptor_pairing_motif()) + 1))

#: Percentiles reported for every size distribution.  Written once so the bulge and
#: helix blocks cannot drift apart.
PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)


class MeasureError(ValueError):
    """A measurement could not be made from the supply as given."""


def is_inside_repo(path: str | Path) -> bool:
    """Is ``path`` under the working directory (i.e. inside the checkout)?

    Kept separate from :func:`portable_path` because "what do I publish" and "may
    this be hashed by path" are different questions: a *basename* that happens to
    exist in the checkout would look repo-relative and be recorded with the wrong
    file's hash.
    """
    try:
        Path(path).resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return False
    return True


def portable_path(path: str | Path) -> str:
    """A path safe to publish: repo-relative if inside the repo, else the basename.

    ⚠ **This repo is public.** A path recorded verbatim carries the developer's home
    directory and account name into every clone, and the supply for this measurement
    is staged outside the repo, so the naive form leaks four of them (the module's
    own ``__file__``, the staged ``msa_root``, the covariation table, and the
    provenance input list).  Content is bound by hash — ``supply_digest_sha256`` and
    ``provenance.inputs`` — so the absolute path adds nothing a reader can use and
    is dropped rather than sanitised in place.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.name


def is_local_path_shaped(value: str) -> bool:
    """Does ``value`` look like a LOCAL filesystem path rather than a host-qualified one?

    ``supply_origin`` is the one field recorded verbatim in a PUBLIC report, so it is
    the one place a home directory and an account name can still reach it. A leading
    ``/`` is only the most obvious spelling: ``~/work/round`` expands to the same
    home directory, and ``C:\\Users\\...`` is the Windows form. The legitimate value —
    ``two.amlab:$HOME/tbox-scratch/<round>/msa`` — is not absolute on POSIX, does not
    start with ``~``, and carries no drive-letter separator, so it passes.
    """
    text = str(value)
    return bool(
        Path(text).is_absolute()
        or text.startswith("~")
        # ⚠ Each Windows spelling needs saying separately, because none of them is
        # absolute under a POSIX `Path`: `C:\...` AND `C:/...` (a drive letter with
        # forward slashes), and the UNC `\\server\share` form. A `":\\" in text`
        # test caught only the first, and the test matrix hid the gap by using
        # `//server/share/...`, which POSIX already calls absolute.
        or re.match(r"^[A-Za-z]:[\\/]", text) is not None
        or text.startswith("\\\\")
    )


#: Read size for :func:`sha256_of`.  1 MiB, as the two chunked copies this function
#: replaced already used.
SHA256_CHUNK_BYTES = 1 << 20


def sha256_of(path: str | Path) -> str:
    """sha256 of one file, for an input recorded by name rather than by path.

    Public, and CHUNKED. Public because four modules now record external inputs this
    way and a private name imported across module boundaries is a rename away from
    breaking them silently. Chunked because two of those modules hash a **model
    checkpoint** — `read_bytes()` would load it whole, so consolidating on the
    one-shot form would have traded a duplication for a memory regression.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(SHA256_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Retained so existing importers keep working; new call sites use the public name.
_sha256_of = sha256_of


# ═════════════════════════════════════════════════════════════════════════════
# Reading the supply
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SupplyItem:
    """One de-novo consensus, parsed once and reused across the whole sweep.

    ``pairs`` and ``row`` are derived here so a 400-cell sweep re-parses nothing;
    both come from the shipped primitives, so the sweep sees exactly what
    ``localize`` would see.
    """

    slug: str
    path: Path
    consensus: ConsensusStructure
    pairs: tuple[int, ...]
    row: str

    @property
    def depth(self) -> int:
        return self.consensus.n_sequences


def read_supply(msa_root: str | Path) -> list[SupplyItem]:
    """Every ``<slug>/msa.sto`` under ``msa_root``, parsed, in slug order.

    A file that cannot be parsed is **not** skipped silently: the producer would
    score that candidate ``unavailable`` (⇒ spared), which is a different fact from
    "the measurement did not look at it", and a measurement that quietly drops rows
    understates its own denominator.
    """
    root = Path(msa_root)
    if not root.is_dir():
        raise MeasureError(f"msa root {root} is not a directory")
    items: list[SupplyItem] = []
    unreadable: list[tuple[str, str]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        sto = sub / "msa.sto"
        if not sto.is_file():
            continue
        try:
            consensus = parse_stockholm(sto)
            # ⚠ Inside the try with the parse. `pair_table` raises on unbalanced
            # brackets, which is the *same* class of "this file is not usable" as a
            # missing SS_cons — leaving it outside meant the first such file aborted
            # with a bare traceback instead of joining the refusal that names them all.
            pairs = tuple(pair_table(consensus.ss_cons))
        except (ArchitectureError, OSError, ValueError) as exc:
            unreadable.append((sub.name, str(exc)))
            continue
        items.append(
            SupplyItem(
                slug=sub.name,
                path=sto,
                consensus=consensus,
                pairs=pairs,
                # The covariation producer writes the candidate's own row first; the
                # bulge test is a statement about *this* locus, not about the alignment.
                row=consensus.row(0),
            )
        )
    if unreadable:
        raise MeasureError(
            f"{len(unreadable)} consensus file(s) could not be parsed, e.g. {unreadable[:2]}; "
            "the producer would score these 'unavailable' — measure them, do not drop them"
        )
    if not items:
        raise MeasureError(f"no <slug>/msa.sto found under {root}")
    return items


def supply_digest(items: Sequence[SupplyItem]) -> str:
    """A single sha256 over ``slug + sha256(bytes)`` of every consensus read.

    The 278 consensuses live on cluster scratch, not in git, so the report must
    carry something that says *which* supply it measured.  A count is not that: a
    re-run against a different round directory with the same cardinality would look
    identical ([[gate-must-bind-to-upstream-evidence]]).
    """
    outer = hashlib.sha256()
    for item in sorted(items, key=lambda i: i.slug):
        inner = hashlib.sha256(item.path.read_bytes()).hexdigest()
        outer.update(f"{item.slug}\t{inner}\n".encode())
    return outer.hexdigest()


def _distribution(values: Iterable[int]) -> dict[str, Any]:
    """Counts + min/max/percentiles for an integer distribution.

    Empty input is reported as ``n: 0`` with null statistics rather than raising —
    "there were none" is a measurement, and an empty helix or bulge set is a real
    outcome of a de-novo consensus.  ⚠ Both branches emit **the same keys**: a reader
    doing ``dist["median"]`` must not have to know which branch produced the block.

    ``median`` is coerced to ``float`` because ``statistics.median`` returns an
    ``int`` for an odd count and a ``float`` for an even one — the same JSON field
    would otherwise change type with the parity of ``n``.
    """
    ordered = sorted(values)
    if not ordered:
        return {
            "n": 0,
            "counts": {},
            "min": None,
            "max": None,
            # ⚠ The NESTED keys too, not just the top-level ones: a consumer reading
            # dist["percentiles"]["p50"] is exactly as broken by a missing inner key.
            "percentiles": {f"p{p}": None for p in PERCENTILES},
            "median": None,
        }
    return {
        "n": len(ordered),
        "counts": {str(k): v for k, v in sorted(Counter(ordered).items())},
        "min": ordered[0],
        "max": ordered[-1],
        "percentiles": {
            f"p{p}": ordered[min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))]
            for p in PERCENTILES
        },
        "median": float(statistics.median(ordered)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The helix arm — D3's `named_elements_present`
# ═════════════════════════════════════════════════════════════════════════════
def helix_marginals(
    items: Sequence[SupplyItem],
    *,
    min_helix_pairs_values: Sequence[int],
    min_named_helices_values: Sequence[int],
) -> dict[str, Any]:
    """What the de-novo instrument resolves on the helix arm, at every sweep value.

    ``helix_stack_depth`` is the distribution of ``Helix.n_pairs`` over every helix
    in every consensus at ``min_pairs=1`` — i.e. before any threshold is applied. It
    is what says whether ``min_helix_pairs`` has anything to bite on.
    """
    stack_depths: list[int] = []
    for item in items:
        stack_depths.extend(h.n_pairs for h in find_helices(item.pairs, min_pairs=1))

    n_helices_by_mhp: dict[str, Any] = {}
    for mhp in min_helix_pairs_values:
        counts = Counter(len(find_helices(item.pairs, min_pairs=mhp)) for item in items)
        n_helices_by_mhp[str(mhp)] = {str(k): v for k, v in sorted(counts.items())}

    grid: dict[str, Any] = {}
    for mhp in min_helix_pairs_values:
        row: dict[str, Any] = {}
        for mnh in min_named_helices_values:
            n_pass = sum(
                named_elements_status(item.pairs, min_named_helices=mnh, min_helix_pairs=mhp)[0]
                for item in items
            )
            row[str(mnh)] = {
                "n_pass": n_pass,
                "n_fail": len(items) - n_pass,
                "share_pass": round(n_pass / len(items), 6),
            }
        grid[str(mhp)] = row

    return {
        "n_consensuses": len(items),
        "helix_stack_depth": _distribution(stack_depths),
        "n_helices_by_min_helix_pairs": n_helices_by_mhp,
        "named_elements_present_by_min_helix_pairs_then_min_named_helices": grid,
        "max_named_helices": arch.MAX_NAMED_HELICES,
        "expected_helix_elements": list(arch.EXPECTED_HELIX_ELEMENTS),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The bulge arm — D3's `ncca_bulge_detected`
# ═════════════════════════════════════════════════════════════════════════════
def bulge_marginals(
    items: Sequence[SupplyItem],
    *,
    bulge_min_nt_values: Sequence[int],
    bulge_max_nt_values: Sequence[int],
    ncca_pairing_nt_values: Sequence[int],
    allow_wobble_values: Sequence[bool],
) -> dict[str, Any]:
    """The bulge-size distribution, and the three-valued bulge state over the sweep.

    ``bulge_residue_size`` is measured in the **candidate's own degapped residues**,
    not alignment columns — the same reading ``ncca_bulge_status`` applies, and the
    reason ``find_bulges`` carries no size filter of its own.

    Cells where ``bulge_min_nt < ncca_pairing_nt`` are **not** silently skipped:
    ``ncca_bulge_status`` refuses them (a bulge too short to hold the motif would
    read ``absent`` ⇒ minable rather than ``undetectable`` ⇒ spared), so the report
    records them as refused with the reason.
    """
    residue_sizes: list[int] = []
    n_bulges_per_consensus: list[int] = []
    for item in items:
        bulges = find_bulges(item.pairs)
        n_bulges_per_consensus.append(len(bulges))
        residue_sizes.extend(len(degapped_span(item.row, b.start, b.end)) for b in bulges)

    grid: dict[str, Any] = {}
    for ncca in ncca_pairing_nt_values:
        for low in bulge_min_nt_values:
            for high in bulge_max_nt_values:
                key = f"ncca={ncca};range={low}-{high}"
                if high < low:
                    grid[key] = {"refused": "bulge_max_nt < bulge_min_nt"}
                    continue
                if low < ncca:
                    grid[key] = {
                        "refused": (
                            f"bulge_min_nt {low} < ncca_pairing_nt {ncca}: a bulge of "
                            f"{low}..{ncca - 1} residues would read 'absent' (minable) "
                            "rather than 'undetectable' (spared)"
                        )
                    }
                    continue
                for wobble in allow_wobble_values:
                    counts: Counter[str] = Counter()
                    for item in items:
                        state, _ = ncca_bulge_status(
                            item.row,
                            item.pairs,
                            bulge_size_range=(low, high),
                            ncca_pairing_nt=ncca,
                            allow_wobble=wobble,
                        )
                        counts[state] += 1
                    grid.setdefault(key, {})[f"allow_wobble={str(bool(wobble)).lower()}"] = {
                        state: counts.get(state, 0) for state in arch.BULGE_STATES
                    }

    return {
        "n_consensuses": len(items),
        "acceptor_motif": acceptor_pairing_motif(),
        "n_flanked_bulges_per_consensus": _distribution(n_bulges_per_consensus),
        "bulge_residue_size": _distribution(residue_sizes),
        "bulge_state_grid": grid,
    }


# ═════════════════════════════════════════════════════════════════════════════
# The two inert parameters, proved rather than sampled
# ═════════════════════════════════════════════════════════════════════════════
def wobble_inertness(acceptor_3prime: str = arch.TRNA_ACCEPTOR_3PRIME) -> dict[str, Any]:
    """Can ``allow_wobble`` change **any** decision for this acceptor end?

    Proved from the module's own constants rather than measured on a sample, because
    the answer is a property of the motif, not of the corpus.  ``_pairs_with``'s
    wobble arm asks whether the bulge base wobble-pairs with the **acceptor** base
    (``WATSON_CRICK[motif_base]``).  So the flag can only ever matter at a motif
    position whose acceptor base participates in a wobble pair.  For the default
    ``NCCA`` the constrained motif positions are ``U`` and ``G``, whose acceptor
    bases are ``A`` and ``C`` — and ``WOBBLE_PAIRS`` is ``{(G,U), (U,G)}``, which
    contains neither.  ``allow_wobble`` is therefore **structurally inert** here:
    ``0`` and ``1`` cannot produce different output on any input whatsoever.

    A sample-based version of this claim would be weaker *and* misleading — it would
    say "no difference was observed", which is what a broken sweep also says
    ([[namespace-mismatch-invisible-noop]]).
    """
    motif = acceptor_pairing_motif(acceptor_3prime)
    wobble_partners = {partner for _, partner in arch.WOBBLE_PAIRS}
    live_positions = [
        {"index": i, "motif_base": base, "acceptor_base": arch.WATSON_CRICK[base]}
        for i, base in enumerate(motif)
        if base != arch.ANY_BASE and arch.WATSON_CRICK[base] in wobble_partners
    ]
    return {
        "acceptor_3prime": str(acceptor_3prime),
        "motif": motif,
        "wobble_pairs": sorted("".join(p) for p in arch.WOBBLE_PAIRS),
        "positions_where_wobble_can_fire": live_positions,
        "inert": not live_positions,
        "why": (
            "allow_wobble is compared against the ACCEPTOR base (WATSON_CRICK[motif_base]); "
            "no constrained position of this motif has an acceptor base in WOBBLE_PAIRS, so "
            "the flag cannot change any decision for this acceptor end"
        ),
    }


def stem_i_threshold_inertness(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Can ``stem_i_nt_threshold`` change **any** decision on this manifest?

    D6 is ``stem_i_extent_nt < stem_i_nt_threshold OR regulatory_mode ==
    "translational"``, and ``short_stem_i_or_class_ii`` returns ``False`` for a
    ``None`` extent ("absence of a measurement is not evidence of shortness").  So
    the threshold is inert on any manifest that supplies neither field — which is a
    property of the *manifest*, checked here against the real one rather than
    assumed, because it is the manifest that could change.

    Reported both ways: the field census (why it is inert) **and** an execution
    control (``short_stem_i_or_class_ii`` evaluated at both ends of the sweep on
    every row, asserted identical).  The census alone would be a clause read from
    the requested config; the control alone would not say *why*.
    """
    with_extent = [c for c in candidates if c.get("stem_i_extent_nt") is not None]
    with_mode = [c for c in candidates if c.get("regulatory_mode") is not None]
    low, high = 1, 10_000
    disagreements = [
        str(c.get("candidate_id", c.get("id")))
        for c in candidates
        if arch.short_stem_i_or_class_ii(
            c.get("stem_i_extent_nt"), c.get("regulatory_mode"), stem_i_nt_threshold=low
        )
        != arch.short_stem_i_or_class_ii(
            c.get("stem_i_extent_nt"), c.get("regulatory_mode"), stem_i_nt_threshold=high
        )
    ]
    n_relaxed = sum(
        arch.short_stem_i_or_class_ii(
            c.get("stem_i_extent_nt"), c.get("regulatory_mode"), stem_i_nt_threshold=high
        )
        for c in candidates
    )
    # ⚠ Every row, not row 0: heterogeneous rows would make the published census
    # under-report the fields present. (The counts above already scan every row, so the
    # inertness verdict was sound — only the field list was.)
    field_union: set[str] = set()
    for c in candidates:
        field_union.update(c)
    supplies_neither = not with_extent and not with_mode
    return {
        "n_candidates": len(candidates),
        "n_with_stem_i_extent_nt": len(with_extent),
        "n_with_regulatory_mode": len(with_mode),
        "manifest_row_fields": sorted(field_union),
        "thresholds_compared": [low, high],
        "n_rows_that_differ": len(disagreements),
        "n_ultrashort_relax_true_at_either_threshold": n_relaxed,
        "inert": not disagreements and n_relaxed == 0,
        # ⚠ CONDITIONAL on the counts beside it. As a fixed string it asserted "this
        # manifest supplies neither field" even on a manifest that supplies one — a
        # narrative contradicting its own adjacent numbers, which is exactly the shape
        # a reader trusts and cannot check ([[gate-clauses-need-re-derivation]]).
        "why": (
            "short_stem_i_or_class_ii returns False for a None extent and only the "
            "'translational' regulatory_mode fires the other arm; this manifest supplies "
            + (
                "neither field, so ultrashort_relax is False at every threshold. "
                if supplies_neither
                else f"{len(with_extent)} row(s) with stem_i_extent_nt and "
                f"{len(with_mode)} row(s) with regulatory_mode, so the threshold is not "
                "inert by absence of the fields. "
            )
            + "ADR-0006 A4's 'an imperfect value errs safe' argument rests on the "
            "relaxation firing, so where it does not fire that safety margin is "
            "structurally absent and (b) runs as its strict base predicate."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The joint (b) outcome — through the real decision path
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ParamTuple:
    """One full seven-parameter setting, named so the report is readable."""

    label: str
    stem_i_nt_threshold: int
    min_named_helices: int
    min_helix_pairs: int
    bulge_min_nt: int
    bulge_max_nt: int
    ncca_pairing_nt: int
    allow_wobble: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "stem_i_nt_threshold": self.stem_i_nt_threshold,
            "min_named_helices": self.min_named_helices,
            "min_helix_pairs": self.min_helix_pairs,
            "bulge_min_nt": self.bulge_min_nt,
            "bulge_max_nt": self.bulge_max_nt,
            "ncca_pairing_nt": self.ncca_pairing_nt,
            "allow_wobble": self.allow_wobble,
        }


def candidate_state(
    item: SupplyItem,
    params: ParamTuple,
    *,
    min_sequences: int = MIN_REAL_HOMOLOG_N,
) -> str:
    """One consensus's criterion-(b) state at ``params`` — ``passed`` or ``failed``.

    Extracted from :func:`evaluate_tuple`'s loop body so a caller that needs the
    **per-candidate** verdict (P3-15'-g-iv aggregates the matched control's queries
    up to their source records) gets it from the same code the counts come from.  A
    second spelling of "what (b) says about this consensus" is exactly the drift
    ADR-0006 A4 forbids between the measurement and the producer.

    ⚠ ``unavailable`` is *not* returned here and cannot be: a candidate with no
    ``msa.sto`` never becomes a :class:`SupplyItem`, and the depth floor is applied
    inside ``architecture_status``, which returns ``failed`` — not ``unavailable`` —
    for a shallow alignment it was handed.  The caller owns the missing-consensus
    arm (:func:`evaluate_tuple` adds ``n_without_consensus``).
    """
    localization: Localization = arch.localize(
        item.slug,
        item.consensus,
        row=0,
        # The manifest supplies neither field; passing them explicitly keeps the
        # measurement's inputs identical to the producer's on this manifest.
        stem_i_extent_nt=None,
        regulatory_mode=None,
        stem_i_nt_threshold=params.stem_i_nt_threshold,
        class_ii=False,
        min_named_helices=params.min_named_helices,
        min_helix_pairs=params.min_helix_pairs,
        bulge_size_range=(params.bulge_min_nt, params.bulge_max_nt),
        ncca_pairing_nt=params.ncca_pairing_nt,
        allow_wobble=params.allow_wobble,
    )
    state, _ = architecture_status(localization, min_sequences=min_sequences)
    return str(state)


def evaluate_tuple(
    items: Sequence[SupplyItem],
    params: ParamTuple,
    *,
    n_without_consensus: int,
    covariation_by_slug: Mapping[str, str] | None = None,
    min_sequences: int = MIN_REAL_HOMOLOG_N,
) -> dict[str, Any]:
    """``passed`` / ``failed`` / ``unavailable`` over the whole corpus at ``params``.

    Routed through the shipped ``localize`` → ``architecture_status`` path, not
    through a re-derived conjunction: the vacuous-pass guard, the A2 Pin 2 depth
    floor and the ``class_ii_relax``-only-where-undetectable rule all live in
    ``architecture_status``, and a measurement that reproduced only ``named AND
    detected`` would be measuring a predicate the producer does not run.

    ``n_without_consensus`` is added as ``unavailable`` — those candidates have no
    ``msa.sto`` on disk, which ``evaluate_candidate`` scores ``unavailable`` ⇒
    spared (ADR-0005 D14) before it ever reaches the localizer.

    ``covariation_by_slug`` stratifies the outcome by **criterion (a)**'s own
    verdict — see :func:`covariation_stratified_note` for why that is the strongest
    control this corpus can offer.
    """
    counts: Counter[str] = Counter()
    by_cov: dict[str, Counter[str]] = {}
    for item in items:
        state = candidate_state(item, params, min_sequences=min_sequences)
        counts[state] += 1
        if covariation_by_slug is not None:
            arm = covariation_by_slug.get(item.slug, "unknown")
            by_cov.setdefault(arm, Counter())[state] += 1
    counts[arch.STATUS_UNAVAILABLE] += int(n_without_consensus)
    total = sum(counts.values())
    out: dict[str, Any] = {
        "params": params.as_dict(),
        "counts": {k: counts.get(k, 0) for k in ("passed", "failed", "unavailable")},
        "n_candidates": total,
        # `failed` is the only minable state: `passed` and `unavailable` both spare
        # under ADR-0005 D14, so this is the share of the corpus (b) stops protecting.
        "share_failed": round(counts["failed"] / total, 6) if total else None,
        "share_failed_of_decided": (
            round(counts["failed"] / (counts["passed"] + counts["failed"]), 6)
            if (counts["passed"] + counts["failed"])
            else None
        ),
    }
    if covariation_by_slug is not None:
        out["by_covariation_status"] = {
            arm: {
                **{k: c.get(k, 0) for k in ("passed", "failed", "unavailable")},
                "n": sum(c.values()),
                "share_failed": (
                    round(c["failed"] / sum(c.values()), 6) if sum(c.values()) else None
                ),
            }
            for arm, c in sorted(by_cov.items())
        }
        a_passed = by_cov.get("passed", Counter())
        a_failed = by_cov.get("failed", Counter())
        n_a_passed = sum(a_passed.values())
        out["control_and_consequence"] = {
            # Sparing is a DISJUNCTION (ADR-0005 D14): a candidate criterion (a) spared
            # is spared whatever (b) says. So (b)'s verdict changes an outcome on exactly
            # one stratum — the candidates (a) decided against — and the (a)-passed
            # stratum is a free sensitivity control that costs no yield either way.
            "n_a_passed": n_a_passed,
            "n_b_agrees_with_a_passed": a_passed.get("passed", 0),
            "share_b_agrees_with_a_passed": (
                round(a_passed.get("passed", 0) / n_a_passed, 6) if n_a_passed else None
            ),
            "n_a_failed": sum(a_failed.values()),
            "n_failing_both_a_and_b": a_failed.get("failed", 0),
            "note": (
                "n_failing_both_a_and_b is the largest set (b) can hand to mining: every "
                "other candidate is already spared by (a) or has no consensus at all. "
                "share_b_agrees_with_a_passed is (b)'s sensitivity against D3's own "
                "model-independent anchor and changes no outcome."
            ),
        }
    return out


def covariation_stratified_note(arm: str | SupplyArm | None = None) -> str:
    """Why criterion (a)'s verdict is the strongest control this corpus offers.

    ⚠ On the ``curated_control`` arm it is **not** — the supply is believed
    positive, so (a) is a second measurement rather than the anchor.  The string
    comes from :class:`SupplyArm` so the report and this function cannot drift.

    Every consensus measured here belongs to a round-0 **false-positive-manifest**
    candidate, so none of them is a known positive and the corpus supplies no
    ground truth of its own.  What it does supply is a *second, independent*
    structural instrument: criterion (a), R-scape covariation on the very same
    alignment.  ADR-0006 D3 calls (a) "the model-independent structural anchor
    [that] remains … even under both (b) relaxations", so a candidate that (a)
    scored ``passed`` carries statistically-supported covariation — the best
    available evidence that a real structured RNA is there.

    The diagnostic is therefore: **at a given parameter setting, what share of the
    (a)-``passed`` candidates does (b) ``fail``?**  Those are the candidates (b)
    would stop protecting *despite* independent covariation support, and each one
    is a plausible real T-box entering the training set as a hard negative.

    ⚠ It is a control, not ground truth: (a) ``passed`` is evidence of structure,
    not of T-box identity, and the two criteria read the same MSA (ADR-0006 A4), so
    they are **not** independent in their supply — only in what they test on it.
    """
    return resolve_arm(arm).covariation_note


def positive_control(
    path: str | Path,
    *,
    min_named_helices_values: Sequence[int],
    min_helix_pairs_values: Sequence[int],
    ncca_pairing_nt_values: Sequence[int],
    bulge_max_nt: int,
) -> dict[str, Any]:
    """The one **de-novo** positive control the repo owns, swept the same way.

    ``data/interim/homolog_msa/certified_positive.sto`` is a real ``mlocarna``
    consensus of a **known** T-box's own homolog set (the A3/A4 instrument, job
    766) — the only place in this repo where the de-novo instrument has been run on
    something already believed positive.  Every other consensus measured here is a
    round-0 **false-positive-manifest** candidate of unknown status, so nothing else
    in the supply can say where the instrument's sensitivity limit is.

    ⚠ **n = 1.**  This is a single record and cannot support a rate.  It bounds the
    choice in one direction only: a parameter value that fails *this* consensus is
    a value that mines a known T-box.
    """
    # ⚠ Same refusal convention as `read_supply`: the caller checks only that the path
    # is a file, so a malformed one reaches these three calls unguarded and `main` (which
    # does not catch IndexError) would exit 1 with a traceback.
    try:
        consensus = parse_stockholm(path)
        pairs = pair_table(consensus.ss_cons)
        row = consensus.row(0)
    except (ArchitectureError, OSError, ValueError, IndexError) as exc:
        raise MeasureError(
            f"positive control {portable_path(path)} could not be parsed: {exc}"
        ) from exc
    named: dict[str, Any] = {}
    for mhp in min_helix_pairs_values:
        for mnh in min_named_helices_values:
            ok, _ = named_elements_status(pairs, min_named_helices=mnh, min_helix_pairs=mhp)
            named[f"min_helix_pairs={mhp};min_named_helices={mnh}"] = bool(ok)
    bulge: dict[str, Any] = {}
    for ncca in ncca_pairing_nt_values:
        state, _ = ncca_bulge_status(
            row,
            pairs,
            bulge_size_range=(ncca, bulge_max_nt),
            ncca_pairing_nt=ncca,
            allow_wobble=False,
        )
        # Keyed with the range actually used: the low bound TRACKS ncca_pairing_nt
        # here (the loosest bound `ncca_bulge_status` admits), so it is never a fixed
        # member of BULGE_MIN_NT_SWEEP and a reader could not otherwise say which
        # `bulge_state_grid` cell each entry corresponds to.
        bulge[f"ncca={ncca};range={ncca}-{int(bulge_max_nt)}"] = state
    return {
        "path": portable_path(path),
        "n": 1,
        "caveat": (
            "a single de-novo positive control; it bounds the choice in one direction "
            "(a value that fails it mines a known T-box) and supports no rate"
        ),
        "depth": consensus.n_sequences,
        "width": consensus.width,
        "helix_stack_depths": [h.n_pairs for h in find_helices(pairs, min_pairs=1)],
        "bulge_residue_sizes": sorted(
            len(degapped_span(row, b.start, b.end)) for b in find_bulges(pairs)
        ),
        "named_elements_present": named,
        "bulge_state": bulge,
        "bulge_min_nt_used": (
            "bulge_min_nt tracks ncca_pairing_nt at every entry — the loosest low bound "
            "ncca_bulge_status admits, and the most sparing one"
        ),
        "bulge_max_nt_used": int(bulge_max_nt),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The report
# ═════════════════════════════════════════════════════════════════════════════
def default_tuples() -> tuple[ParamTuple, ...]:
    """The named joint settings the report scores over the whole corpus.

    Spread deliberately across the admissible surface — the loosest non-vacuous
    corner, the strictest, and the biologically-argued middle — so the §7 reader can
    see the yield/sparing trade rather than one recommended point.  ``label`` is
    descriptive only; nothing here is a pin.
    """
    return (
        ParamTuple("loosest_nonvacuous", 1, 1, 2, 1, 10_000, 1, False),
        ParamTuple("sensitive_core", 1, 2, 2, 2, 50, 2, False),
        ParamTuple("canonical_core_3helix", 1, 3, 2, 4, 50, 2, False),
        ParamTuple("canonical_core_3helix_ncca3", 1, 3, 2, 4, 50, 3, False),
        ParamTuple("curated_freeze_transplant", 1, 4, 3, 4, 50, 4, False),
        ParamTuple("strictest", 1, 4, 5, 7, 20, 4, False),
    )


def measure(
    *,
    msa_root: str | Path,
    manifest_path: str | Path,
    covariation_status_path: str | Path | None,
    positive_control_path: str | Path | None,
    supply_origin: str | None = None,
    tuples: Sequence[ParamTuple] | None = None,
    arm: str | SupplyArm | None = None,
) -> dict[str, Any]:
    """The whole measurement, as the report body (no provenance — ``main`` adds it).

    ``arm`` selects only the report's **self-description** (:class:`SupplyArm`); the
    sweep and every count are arm-independent, which is what makes the control
    matched.  ``None`` is :data:`DEFAULT_ARM`, so the committed `P3-15'-f` report
    re-derives byte-identically across this seam.
    """
    supply_arm = resolve_arm(arm)
    # ⚠ `supply_origin` is the one path-shaped field recorded VERBATIM, because it is
    # operator-authored provenance naming the cluster (`two.amlab:$HOME/...`) rather
    # than a path this process discovered. That makes it the one place a local absolute
    # path could still reach a PUBLIC report, so it is refused rather than redacted —
    # redacting would destroy the cluster path that is the whole point of the field.
    if supply_origin is not None and is_local_path_shaped(supply_origin):
        raise MeasureError(
            f"supply_origin {supply_origin!r} is a local absolute path; it is recorded "
            "verbatim in a public report — name the host and use $HOME, e.g. "
            "'two.amlab:$HOME/tbox-scratch/<round>/msa'"
        )
    items = read_supply(msa_root)
    manifest = json.loads(Path(manifest_path).read_text())
    # ⚠ `.get`, not `[...]`: a JSON object without a `candidates` key would raise
    # KeyError, which `main` does not catch, so the CLI would exit 1 with a traceback
    # instead of 3 with a refusal — and `--manifest` accepts any path.
    candidates = manifest.get("candidates") if isinstance(manifest, Mapping) else manifest
    if not isinstance(candidates, list) or not candidates:
        raise MeasureError(f"manifest {portable_path(manifest_path)} carries no candidate rows")
    # ⚠ And the ROW shape before `row.get(...)`: a bare string entry raises
    # AttributeError, same escape. The shape of a file another process wrote is an
    # input, not an invariant.
    n_non_rows = sum(1 for row in candidates if not isinstance(row, Mapping))
    if n_non_rows:
        raise MeasureError(
            f"{n_non_rows} manifest row(s) are not objects; a candidate row must carry "
            "'candidate_id' or 'id'"
        )

    # ⚠ Built as a LIST first, and the set taken only after both defects below are
    # refused. A set comprehension deduplicates *before* anything can look, so a
    # repeated `candidate_id` — or a row carrying neither key, which stringifies to
    # "None" and collapses every such row onto one element — silently shrinks
    # `n_candidates_in_manifest`, which flows into `n_candidates_without_consensus`
    # and then into `unavailable` in every joint row. The arithmetic still reconciles,
    # which is exactly what makes it invisible ([[duplicate-key-merges-instead-of-colliding]]).
    raw_ids: list[str] = []
    n_rows_without_an_id = 0
    for row in candidates:
        cid = row.get("candidate_id")
        if cid is None:
            cid = row.get("id")
        if cid is None:
            n_rows_without_an_id += 1
            continue
        raw_ids.append(str(cid))
    if n_rows_without_an_id:
        raise MeasureError(
            f"{n_rows_without_an_id} manifest row(s) carry neither 'candidate_id' nor "
            "'id'; they would collapse onto one identifier and understate "
            "n_candidates_without_consensus in every joint row"
        )
    manifest_ids = set(raw_ids)
    if len(manifest_ids) != len(raw_ids):
        repeated = sorted({i for i in raw_ids if raw_ids.count(i) > 1})
        raise MeasureError(
            f"{len(raw_ids) - len(manifest_ids)} duplicate candidate id(s) in the "
            f"manifest, e.g. {repeated[:2]}; the deduplicated count would understate "
            "n_candidates_without_consensus in every joint row"
        )
    slugs_present = {item.slug for item in items}
    # ⚠ Every count below assumes candidate_id → slug is INJECTIVE on this manifest, and
    # the slug is only a 64-char sanitised prefix plus 12 hex of sha1. Two ids sharing a
    # slug would share one directory, so `n_candidates_without_consensus` would count the
    # collision as a missing consensus and inflate `unavailable` in every joint row —
    # silently, because the arithmetic still reconciles. Checked, not assumed.
    slug_owners: dict[str, list[str]] = {}
    for cid in sorted(manifest_ids):
        slug_owners.setdefault(candidate_slug(cid), []).append(cid)
    collisions = {slug: ids for slug, ids in slug_owners.items() if len(ids) > 1}
    if collisions:
        raise MeasureError(
            f"{len(collisions)} candidate slug collision(s) in the manifest, e.g. "
            f"{sorted(collisions.items())[:2]}; one directory would stand for two "
            "candidates and the missing-consensus count would be wrong"
        )
    slugs_expected = set(slug_owners)
    if not slugs_present <= slugs_expected:
        raise MeasureError(
            f"{len(slugs_present - slugs_expected)} consensus dir(s) do not correspond to "
            "any manifest candidate — the supply and the manifest are not the same corpus"
        )

    supply: dict[str, Any] = {
        # ⚠ `msa_root` is wherever the consensuses were staged when this ran, which on a
        # laptop is a scratch path that will not exist later, and this repo is PUBLIC so
        # the verbatim path would publish a home directory and an account name.
        # `supply_origin` names where they were PRODUCED and `supply_digest_sha256` says
        # which bytes were read; the path alone identifies nothing.
        "msa_root": portable_path(msa_root),
        "supply_origin": supply_origin,
        "n_candidates_in_manifest": len(manifest_ids),
        "n_consensuses_measured": len(items),
        "n_candidates_without_consensus": len(manifest_ids) - len(items),
        "supply_digest_sha256": supply_digest(items),
        "alignment_depth": _distribution([item.depth for item in items]),
        "min_sequences_floor": MIN_REAL_HOMOLOG_N,
        "n_below_depth_floor": sum(item.depth < MIN_REAL_HOMOLOG_N for item in items),
        "consensus_width": _distribution([item.consensus.width for item in items]),
    }

    covariation_by_slug: dict[str, str] | None = None
    if covariation_status_path is not None:
        cov = json.loads(Path(covariation_status_path).read_text())
        # ⚠ A missing or empty `status` map must REFUSE, not degrade. Defaulting it to
        # `{}` leaves a report that still looks complete — every joint row keeps its
        # `by_covariation_status` key, `control_and_consequence` reports `n_a_passed: 0`,
        # and the (a) control silently measures nothing. `read_supply` sets the
        # precedent: refuse an input you cannot read rather than measure a smaller
        # corpus ([[clauses-must-guard-emptiness]]).
        if not isinstance(cov, Mapping) or not isinstance(cov.get("status"), Mapping):
            raise MeasureError(
                f"covariation status {portable_path(covariation_status_path)} carries no "
                "'status' map; the (a) stratification would be silently empty"
            )
        cov_status = cov["status"]
        if not cov_status:
            raise MeasureError(
                f"covariation status {portable_path(covariation_status_path)} is empty"
            )
        # ⚠ The VALUES too, not just the map. An unhashable value (a list) makes
        # `Counter(cov_status.values())` raise TypeError, which `main` does not catch —
        # the same traceback-instead-of-refusal escape, through an operator-supplied path.
        non_strings = sorted(str(cid) for cid, s in cov_status.items() if not isinstance(s, str))
        if non_strings:
            raise MeasureError(
                f"{len(non_strings)} covariation status value(s) are not strings, e.g. "
                f"{non_strings[:2]}; the (a) stratification cannot read them"
            )
        decided = {cid for cid, s in cov_status.items() if s in ("passed", "failed")}
        # ⚠ The SIBLING of the manifest check above, and it was missing. The covariation
        # table is a separate operator-supplied input keyed by the same slug: two ids
        # collapsing onto one slug means last-write-wins, so a candidate is attributed to
        # the wrong criterion-(a) arm. That shifts `by_covariation_status`,
        # `control_and_consequence` and `helix_arm_on_covariation_passed` — and every
        # count still reconciles, so nothing in the report shows it
        # ([[fixed-one-of-two-identical-things]]).
        cov_slug_owners: dict[str, list[str]] = {}
        for cid in sorted(cov_status):
            cov_slug_owners.setdefault(candidate_slug(cid), []).append(str(cid))
        cov_collisions = {s: ids for s, ids in cov_slug_owners.items() if len(ids) > 1}
        if cov_collisions:
            raise MeasureError(
                f"{len(cov_collisions)} candidate slug collision(s) in the covariation "
                f"status, e.g. {sorted(cov_collisions.items())[:2]}; one arm would stand "
                "for two candidates and the (a) stratification would be misattributed"
            )
        covariation_by_slug = {candidate_slug(cid): str(state) for cid, state in cov_status.items()}
        supply["covariation"] = {
            "path": portable_path(covariation_status_path),
            "sha256": sha256_of(covariation_status_path),
            "counts": {k: v for k, v in sorted(Counter(cov_status.values()).items())},
            "n_decided": len(decided),
            # (b) reads the same MSA (a) does (ADR-0006 A4), so the set of candidates
            # with a consensus must be exactly (a)'s decided set. If it is not, the two
            # disjuncts are reading different corpora.
            "decided_set_equals_supply": {candidate_slug(c) for c in decided} == slugs_present,
            "control_note": covariation_stratified_note(supply_arm),
        }

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": supply_arm.step,
        "adr": ADR,
        "pins_nothing": True,
        "disclosure": supply_arm.disclosure,
        "supply": supply,
        "sweep": {
            "min_helix_pairs": list(MIN_HELIX_PAIRS_SWEEP),
            "min_named_helices": list(MIN_NAMED_HELICES_SWEEP),
            "bulge_min_nt": list(BULGE_MIN_NT_SWEEP),
            "bulge_max_nt": list(BULGE_MAX_NT_SWEEP),
            "ncca_pairing_nt": list(NCCA_PAIRING_NT_SWEEP),
            "allow_wobble": [False, True],
        },
        "helix_arm": helix_marginals(
            items,
            min_helix_pairs_values=MIN_HELIX_PAIRS_SWEEP,
            min_named_helices_values=MIN_NAMED_HELICES_SWEEP,
        ),
        "helix_arm_on_covariation_passed": (
            helix_marginals(
                [i for i in items if covariation_by_slug.get(i.slug) == "passed"],
                min_helix_pairs_values=MIN_HELIX_PAIRS_SWEEP,
                min_named_helices_values=MIN_NAMED_HELICES_SWEEP,
            )
            if covariation_by_slug
            and any(covariation_by_slug.get(i.slug) == "passed" for i in items)
            else None
        ),
        "bulge_arm": bulge_marginals(
            items,
            bulge_min_nt_values=BULGE_MIN_NT_SWEEP,
            bulge_max_nt_values=BULGE_MAX_NT_SWEEP,
            ncca_pairing_nt_values=NCCA_PAIRING_NT_SWEEP,
            allow_wobble_values=(False, True),
        ),
        "inert_parameters": {
            "allow_wobble": wobble_inertness(),
            "stem_i_nt_threshold": stem_i_threshold_inertness(candidates),
        },
        "joint": [
            {
                "label": params.label,
                **evaluate_tuple(
                    items,
                    params,
                    n_without_consensus=len(manifest_ids) - len(items),
                    covariation_by_slug=covariation_by_slug,
                ),
            }
            for params in (tuples if tuples is not None else default_tuples())
        ],
    }

    # ⚠ An ABSENT control is recorded, never omitted. The CLI default points into the
    # repo, so a missing file is far more likely a staging mistake than a deliberate
    # skip — and a report with no `positive_control` key reads as a report that was
    # never meant to have one.
    if positive_control_path is not None and Path(positive_control_path).is_file():
        body["positive_control"] = {
            "available": True,
            **positive_control(
                positive_control_path,
                min_named_helices_values=MIN_NAMED_HELICES_SWEEP,
                min_helix_pairs_values=MIN_HELIX_PAIRS_SWEEP,
                ncca_pairing_nt_values=NCCA_PAIRING_NT_SWEEP,
                bulge_max_nt=50,
            ),
        }
    else:
        body["positive_control"] = {
            "available": False,
            "reason": supply_arm.no_positive_control_reason,
            "path": portable_path(positive_control_path) if positive_control_path else None,
        }
    return body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="architecture_param_measure",
        description=(
            "Measure criterion (b)'s seven rule parameters on the real de-novo "
            "consensuses produced by the P3-15'-e/-e-ii supply run."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="write the parameter-measurement report")
    m.add_argument(
        "--msa-root",
        required=True,
        help="directory of <slug>/msa.sto (the supply run's $ROUND_DIR/msa)",
    )
    m.add_argument(
        "--manifest",
        default="data/processed/mining/round0_fp_manifest.json",
        help="the FP manifest whose candidates the supply corresponds to",
    )
    m.add_argument(
        "--covariation-status",
        default=None,
        help="optional covariation_status.json, cross-checked against the supply",
    )
    m.add_argument(
        "--positive-control",
        default="data/interim/homolog_msa/certified_positive.sto",
        help="the one de-novo positive control (n=1); pass an absent path to skip",
    )
    m.add_argument(
        "--supply-origin",
        default=None,
        help=(
            "free text naming where the consensuses were PRODUCED (cluster path + SLURM "
            "job ids); --msa-root is only where they were staged to read"
        ),
    )
    m.add_argument(
        "--arm",
        choices=sorted(SUPPLY_ARMS),
        default=DEFAULT_ARM,
        help=(
            "which supply this is: 'round0_fp' (P3-15'-f, FP-manifest candidates of "
            "unknown status) or 'curated_control' (P3-15'-g-iv, Stage-1 spans on "
            "held-out curated records). Selects the report's disclosure, its (a) note "
            "and its provenance rule — NOT the sweep, which is identical by construction"
        ),
    )
    m.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="report path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "measure":  # pragma: no cover - argparse enforces
        raise MeasureError(f"unknown command {args.command!r}")
    try:
        supply_arm = resolve_arm(args.arm)
        # ⚠ `--out` and `--manifest` default to the FP arm's paths and neither follows
        # `--arm`. `--arm curated_control` with a forgotten `--out` would overwrite the
        # committed P3-15'-f report with the control's counts AND the control's prose —
        # internally consistent, so nothing downstream could detect it. The defaults stay
        # as they are (that is what keeps the FP report byte-identically re-derivable);
        # the mismatched COMBINATION is refused instead.
        # ⚠ Resolved, not string-compared: `./reports/p3/...`, `reports/p3/../p3/...`
        # and an absolute path to the same file are all spellings a raw `==` waves
        # through, and each one writes the control's numbers over the committed report.
        # Residual gap, stated rather than papered over: `DEFAULT_OUT` resolves against
        # the CURRENT cwd, so an absolute path aimed at a checkout the process is not
        # standing in is still not caught.
        out_resolved = Path(args.out).resolve()
        for other in SUPPLY_ARMS.values():
            if other.key == supply_arm.key:
                continue
            if out_resolved == Path(other.canonical_out).resolve():
                raise MeasureError(
                    f"--arm {supply_arm.key!r} writing to {args.out!r} would overwrite "
                    f"the committed {other.step} report ({other.canonical_out}) with "
                    f"{supply_arm.step} numbers and {supply_arm.step} prose; the result "
                    "would be internally consistent and undetectable downstream. Pass "
                    f"--out {supply_arm.canonical_out} or a scratch path"
                )
        body = measure(
            msa_root=args.msa_root,
            manifest_path=args.manifest,
            covariation_status_path=args.covariation_status,
            positive_control_path=args.positive_control,
            supply_origin=args.supply_origin,
            arm=supply_arm,
        )
    # ⚠ OSError and JSONDecodeError too: `--manifest` and `--covariation-status` are
    # read directly, so a missing or malformed file otherwise exits 1 with a traceback
    # while `read_supply` refuses its own bad input with exit 3. One convention.
    except (MeasureError, ArchitectureError, OSError, json.JSONDecodeError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    out = Path(args.out)
    # ⚠ `provenance.inputs` records the path it hashed, so an input staged OUTSIDE the
    # repo would publish an absolute home path in a public repo. Inputs inside the repo
    # are hashed by `build_provenance`; anything outside is hashed here and recorded by
    # BASENAME + sha256, which is what a reader can actually check.
    repo_inputs: list[str] = []
    external: dict[str, Any] = {
        "msa_root": portable_path(args.msa_root),
        "supply_origin": args.supply_origin,
        "supply_digest_sha256": body["supply"]["supply_digest_sha256"],
        "n_consensuses": body["supply"]["n_consensuses_measured"],
    }
    for label, candidate in (
        ("manifest", args.manifest),
        ("covariation_status", args.covariation_status),
        ("positive_control", args.positive_control),
    ):
        if not candidate or not Path(candidate).is_file():
            continue
        if is_inside_repo(candidate):
            repo_inputs.append(portable_path(candidate))
        else:
            external[label] = {"name": Path(candidate).name, "sha256": sha256_of(candidate)}
    body["provenance"] = build_provenance(
        rule=supply_arm.provenance_rule,
        script=portable_path(__file__),
        inputs=repo_inputs,
        outputs=[],
        adr=ADR,
        extra={"schema_version": SCHEMA_VERSION, "external_inputs": external},
    )
    # ⚠ `--out` is operator-supplied and unvalidated too: a read-only directory or an
    # invalid path component otherwise exits 1 with a traceback while every *input* path
    # refuses with exit 3. One convention, both directions.
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"refused: cannot write report to {portable_path(out)}: {exc}", file=sys.stderr)
        return 3
    print(
        f"measured {body['supply']['n_consensuses_measured']} de-novo consensuses "
        f"over {body['supply']['n_candidates_in_manifest']} candidates -> {out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
