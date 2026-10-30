#!/usr/bin/env python
"""P3-16 — mint the in-distribution two-stage precision benchmark, in **scan geometry**.

What the benchmark is, and why it is built this way
---------------------------------------------------
PRD §2.3's closed-benchmark precision is defined over **items**: corpus T-box loci as
positives and the **four §9.1 decoy classes as labeled negatives**. Every item is one
**1,024-nt scan window** — the pinned Stage-1 geometry — because that is the only form in
which the two classes are commensurable *to the scanner*:

* a **positive** is the gate4_eval locus carved into a window at an honest phase
  (``window_dataset.window_lead_range`` + ``deterministic_lead``, the same geometry
  training used), so the locus sits in its real ±genomic context;
* a **negative** is a §9.1 decoy **spliced into a real mined genomic host window** by the
  shipped ``data/embedding.py::embed_decoy_rows`` — the exact operator ADR-0005 A7 uses to
  put decoys into the training stream.

⚠ **This replaces a first construction that had no power, and the failure is worth stating.**
The benchmark was first built item-as-contig — each decoy scanned as its own ~280-nt contig,
zero-flanked into one window. Measured on that build, Stage-1 emitted a candidate for
**1 of 692 decoys** (the twin arm; 6 of 692 for the shipped scanner): every
``structured_rna``, every ``gc_background`` and every ``dinuc_shuffled`` decoy produced no
candidate at all, so there was nothing for Stage-2 to reject and the whole measured
"precision gain" was a single ``leader_decoy`` item — the one §9.1 record that is itself
T-box-derived. A bare excised decoy is not the object the scanner meets; a decoy inside a
real host window is, which is precisely why ``embed_decoy_rows`` exists and why P2's mining
round found **941** genuine Stage-1 false positives on genomic substrate. A control that
cannot fire reads as a clean result ([[control-matchedness-must-be-asserted]]), so the
geometry is now the scanner's, not the item table's.

Both arms see the identical item set, so recall shares a denominator and "matched recall" is
well-posed.

The positive population is **GATE-4's graded population**, not a re-derivation
------------------------------------------------------------------------------
``window_dataset.load_gate4_eval_records()`` — ``nested_train ∧ fold_random == "test"``,
whole-cluster closed, 1,201 records / 1,029 clusters. That loader's own docstring works
through why the two obvious alternatives are wrong (raw ``fold_random == "test"`` is 51.2 %
inside the training fold; its complement is the leave-clade-out holdout, which is P4's
question). Using the same population makes the P2 segmentation gate and this P3 precision
gate commensurable, and it is the **only** in-distribution population any Stage-1
checkpoint in this repo withheld.

⚠ It is withheld by the **GATE-4 eval twin**, not by the shipped scanner: P2-10d′-b trained
on the full 8,303-record ``nested_train`` fold, so it has no in-distribution holdout at all
(`docs/model_card.md:51-52`, `reports/p2/gate4.json` disclosures). Per the §7 decision of
2026-08-13 the twin is the **gated** Stage-1 arm and the shipped scanner is scored on the
same items and **reported beside it**, disclosed as Stage-1-in-sample. This script mints one
item set; the two arms differ only in which checkpoint scores it.

The 1,201 is three fewer than the 1,204 rows ``fold_random == "test" ∧ nested_train`` selects
in the item table: the loader drops 3 records whose *genomic context* is shorter than one
window. That criterion is irrelevant to a locus-sized item, and the three are dropped anyway
— binding the population to the shipped loader is worth more than three records, and the
delta is recorded in the manifest rather than absorbed.

Train-exposure is **derived from the shipped predicates and cross-checked**
--------------------------------------------------------------------------
Every item records whether each Stage-1 checkpoint trained on it. For a corpus record that
is membership in ``load_corpus_records(...)``'s fold; for a decoy it is ADR-0005 A7's
embedding rule (``embedding.TRAINING_DECOY_POOLS``, minus masked rows, minus rows whose own
parent is outside the fold). The script **refuses to write** unless that derivation
reproduces the two committed training runs' ``n_embedded_by_pool`` exactly — 702 + 2,999 for
the production run and 602 + 2,999 for the twin. A derivation that merely looked plausible
would silently mislabel which negatives are memorised, which is the one thing the report's
disclosures rest on.

Measured on the committed tables: ``gc_background`` (206 items) and ``leader_decoy`` (1) were
kept out of **both** Stage-1 runs by A7 pins 4–5; all 2,999 ``structured_rna`` decoys were
embedded in **both**; ``dinuc_shuffled`` splits by parent fold. The per-item exposure flags are
what carries that into the report as a disclosure instead of a silent bias.

The gated subset (ADR-0005 A15 Pin 4, schema 1.1)
-------------------------------------------------
All four §9.1 pools are **minted and scored**, but ``leader_decoy`` is reported **beside** the
gate, not in it: all 8 of its records are T-box-derived, which A7 Pin 5 calls "a direct label
contradiction". This supersedes the 2026-08-13 §7 four-pool call. The manifest declares the
subset rather than leaving every consumer to re-derive it: a per-item ``gated`` flag (False iff
``pool == "leader_decoy"``) and ``scope.n_items_gated`` / ``n_negatives_gated`` /
``negatives_by_pool_gated`` / ``n_blocks_gated``. The leader stays in the decoy list, the draw
and the mint at its ``decoy_id``-sorted position: ``unique_hosts=True`` assigns hosts by
position in a seeded permutation, so removing it would re-index every later host.

The host screen (ADR-0005 A15 Pins 1–3, ``--host-screen``)
---------------------------------------------------------
v1's only host filter was ``parent_nested_train == False``, a rule about the mining-pool
record's *parent*. It says nothing about the DNA of the window itself, and v1's pass was
carried by hosts that held RF00230-detectable sequence (A15's trigger). ``--host-screen``
takes the committed ``reports/p3/host_screen.json`` (``integration/host_screen.py``) and
restricts the out-of-fold pool to the windows it admitted **before** the seeded permutation,
so the draw runs over the screened pool and all 692 hosts are redrawn. The minter refuses a
screen whose controls did not all fire, whose problems list is non-empty, whose threshold is
not 10.0 bits, or whose pool digest or admitted-list digest does not match the bytes on disk.
Without the flag the script reproduces v1's items and contigs (the new schema fields aside).

Blocks (ADR-0005 D5)
--------------------
CIs resample at the homology-cluster level. Positives block on ``cluster_id``;
``dinuc_shuffled`` decoys inherit their **parent locus's** ``cluster_id`` — they are
Altschul–Erikson permutations of a corpus positive, so a shuffle and the locus it came from
are not independent draws; the parentless pools (`gc_background`, `structured_rna`,
`leader_decoy`) block on their **unique host window**, one per negative. Sharing a block is
the conservative direction: it reduces the number of independent units and **widens** the
interval. A null is never a block id — a decoy whose parent is missing from the cluster map
falls back to its host rather than into a shared `None` bucket
([[nulls-inflate-block-counts]]).

Usage (from the checkout that holds the DVC data)::

    python scripts/mint_p3_16_benchmark.py \\
        --dataset data/processed/stage2_dataset.parquet \\
        --decoys data/processed/negatives/decoys_v0.parquet \\
        --host-screen reports/p3/host_screen.json \\
        --out data/interim/p3_16/benchmark_v2.json

``--out`` is refused if it names a v1 artifact (A15 Pin 5: v1 stays byte-unchanged).

The hard-negative benchmark (ADR-0005 A17, ``--hardneg``)
--------------------------------------------------------
``--hardneg`` mints ``hardneg_benchmark.json`` in v2's contig/item schema, so every
``integration.two_stage`` leg consumes it unchanged. The positives are v2's graded positives with
their contigs byte-identical (each copy re-hashed against v2's, and a digest over all of them
recorded). The negatives are ``integration/hardneg_pool``'s graded and D4-context ("beside")
native windows as one-window contigs: their bytes are asserted against the pool's per-item sha256
(A17 Pin 2: the screened bytes ARE the scored bytes), each is blocked on its ASSEMBLY (A17 Pin 3),
graded iff it is not D4-context, and tagged with its pool metadata. There is no leader decoy and
no §9.1 decoy pool. The pool report must pass its own validator and name the item files given; on
the one census extension the census pool is minted BESIDE the draw's (the census selects only what
the draw left). ``seen_by`` on a native window is the measured fold overlap, never a claim::

    python scripts/mint_p3_16_benchmark.py --hardneg reports/p3/hardneg_pool.json \\
        --hardneg-items data/interim/p3_16/hardneg_pool/graded.json \\
                        data/interim/p3_16/hardneg_pool/beside.json \\
        --out data/interim/p3_16/hardneg_benchmark.json

A v1 or v2 ``--out``, or one that does not name ``hardneg``, is refused (A17 Pin 5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tbox_finder.data import embedding as emb
from tbox_finder.data import negatives as negatives_mod
from tbox_finder.data import window_dataset as wd
from tbox_finder.integration.stage1_folds import ARM_EXCLUDE_GATE4_EVAL, training_fold_ids
from tbox_finder.masking import is_missing, row_text
from tbox_finder.stage2.cm_confirmer import CANONICAL_SOURCES

#: The rung the benchmark is drawn from. Stage-2 (P3-06) admits only ``fold_random ==
#: "train"``, so every ``test`` row is out of Stage-2's training set — measured 0 overlap.
RUNG = "test"

#: What the two committed Stage-1 runs recorded embedding, per pool. The derivation below
#: must reproduce these or the script refuses; see the module docstring.
EXPECTED_EMBEDDED: dict[str, dict[str, int]] = {
    "production": {"dinuc_shuffled": 702, "structured_rna": 2999},
    "twin": {"dinuc_shuffled": 602, "structured_rna": 2999},
}

# ``ARM_EXCLUDE_GATE4_EVAL`` and ``training_fold_ids`` are imported above from
# ``integration/stage1_folds.py`` (promoted at P3-16′-remint, ADR-0005 A15 Pin 1) so the minter
# and the host screen read ONE fold definition; they stay importable from this module.

#: All four §9.1 classes are MINTED and scored (PRD §2.3), unlike
#: `embedding.TRAINING_DECOY_POOLS`, which is A7's two-pool *training* mix. Which of them enter
#: the GATED denominator is :data:`NON_GATED_POOLS`' business, not this tuple's: dropping a pool
#: here would remove it from the draw and re-index every later host.
BENCHMARK_DECOY_POOLS: tuple[str, ...] = (
    "gc_background",
    "structured_rna",
    "dinuc_shuffled",
    "leader_decoy",
)

#: Pools minted and scored but reported BESIDE the gate (ADR-0005 A15 Pin 4): all 8
#: ``leader_decoy`` records are T-box-derived (A7 Pin 5: "a direct label contradiction").
NON_GATED_POOLS: tuple[str, ...] = ("leader_decoy",)
GATED_RULE = "ADR-0005 A15 Pin 4 — leader_decoy minted and scored, reported beside the gate"

#: 1.1 adds the per-item ``gated`` flag, the ``scope.*_gated`` counts, ``scope.host_pool``'s
#: ``screen`` / ``composition`` / ``n_available_to_draw``, the N-free host-overlap rule and
#: ``source.mining_pool_sha256`` / ``source.host_screen*`` and ``scope.insert_exposure``
#: (ADR-0005 A15).
SCHEMA_VERSION = "1.1"
STEP = "P3-16"

#: What a ``--host-screen`` report must carry for the minter to draw from its admitted list
#: (ADR-0005 A15 Pin 2's pinned value, never a default the report may restate differently).
HOST_SCREEN_THRESHOLD_BITS = 10.0
HOST_SCREEN_ADR = "ADR-0005 A15"

#: ``scope.host_pool.selection`` per mode. The unscreened string is v1's (post PR #133); the
#: screened one must name the screen, and neither may say the hosts are "clean" for any arm —
#: a selection RULE is not a measurement of what the drawn windows contain.
UNSCREENED_SELECTION = (
    "parent_nested_train == False — the mining-pool record's PARENT, not the DNA of "
    "the window drawn from it; see overlap_with_training_folds"
)
SCREENED_SELECTION = (
    "parent_nested_train == False, then admitted by the ADR-0005 A15 host screen: Pin 1 "
    "fold screen (whole pre-splice window, canonical N-free 32-mers on either strand against "
    "the union of every Stage-1/Stage-2 training sequence plus every decoys_v0 insert, or a "
    "same-accession training-interval overlap) and Pin 2 content screen (RF00230/TBDB001 "
    "cmsearch >= 10.0 bits, or the GGTGGNACCGCG motif), applied BEFORE the seeded draw; "
    "see screen and overlap_with_training_folds"
)

#: A15 Pin 5: v1 stays byte-unchanged, so no v2 artifact may be written to a v1 path.
V1_OUT_SUFFIX = "benchmark_v1.json"

#: The rows the shipped Stage-2 trained on: ``stage2/train.py::select_rows`` at its train rung,
#: equal to the checkpoint provenance's ``extra.population.n_admitted`` and to the host screen's
#: Pin 1 (iv) pin. A15 Pin 7(c)'s third count compares the test inserts against its decoys.
STAGE2_TRAINING_ROWS = 9059
INSERT_EXPOSURE_RULE = (
    "ADR-0005 A15 Pin 7(c) — a gated negative's decoys_v0 insert (U->T, upper-cased) equal in "
    "sequence to the rna_sequence of a decoy row in the shipped Stage-2 training population "
    "(stage2.train.select_rows at the train rung). By SEQUENCE, never by id: every test-rung "
    "decoy is out of that population by id, so an id comparison is 0 by construction."
)


def transcribe_to_dna(rna: str) -> str:
    """RNA → DNA. Stage 1 is a DNA model; the item table stores RNA (PRD §6 handoff, run
    backwards). Uppercased first so a lower-case ``u`` cannot survive as an ``u``."""
    return rna.upper().replace("U", "T")


def decoy_embedded(row: pd.Series, fold: set[str]) -> bool:
    """ADR-0005 A7's embedding rule for one decoy row, against one arm's training fold.

    Mirrors ``embedding.embed_decoy_rows``' admission: the pool must be one of the two A7
    embeds, the row must not be masked, and a decoy that *has* a parent must have that
    parent inside the fold (a permutation of held-out DNA is held-out DNA)."""
    if str(row["pool"]) not in emb.TRAINING_DECOY_POOLS:
        return False
    if bool(row["masked"]):
        return False
    parent = row["source_record_id"]
    parent = "" if parent is None or pd.isna(parent) else str(parent).strip()
    return parent == "" or parent in fold


def decoy_block_id(
    decoy_row: Mapping[str, Any], host_id: str, parent_cluster: Mapping[str, int]
) -> str:
    """The resampling block for one negative (ADR-0005 D5), conservatively.

    A ``dinuc_shuffled`` decoy is an Altschul–Erikson permutation **of a corpus locus**, so
    it is correlated with that locus and with every other item from the same homology
    cluster: it inherits its parent's ``cluster_id``, which puts it in the same bootstrap
    block as the positive it was made from. A parentless decoy (``gc_background``,
    ``structured_rna``, ``leader_decoy``) has no such relationship and blocks on its
    **unique** host window instead — one host per negative, so one block per negative.

    Sharing a block is the **conservative** direction: it reduces the number of independent
    units and widens the interval. An earlier build keyed every negative on its host, which
    silently treated a shuffle and its parent locus as independent draws and narrowed a
    published CI (CodeRabbit, PR #133).

    A null is never a block id: a decoy whose parent is absent from the cluster map falls
    back to its host rather than into a shared ``None`` bucket
    ([[nulls-inflate-block-counts]]).
    """
    parent = decoy_row.get(negatives_mod.SOURCE_RECORD_ID_COL)
    parent = "" if parent is None or pd.isna(parent) else str(parent).strip()
    cluster = parent_cluster.get(parent)
    if parent and cluster is not None:
        return f"cluster:{int(cluster)}"
    return f"host:{host_id}"


def build_positive_windows(
    records: Sequence[Any], window: int = wd.WINDOW_NT
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One honest ``window``-nt scan window per gate4_eval record, locus in real context.

    The phase comes from the shipped geometry — ``window_lead_range`` bounds the leads that
    keep the whole locus inside the window without inventing 5′/3′ sequence, and
    ``deterministic_lead`` picks the centred one — so a benchmark window is the same object
    a training window is, minus the augmentation draw. A record that admits no honest window
    is refused, not clipped: a truncated locus is a different item.
    """
    contigs: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for record in records:
        bounds = wd.window_lead_range(
            locus_offset=record.locus_offset,
            locus_length=record.locus_length,
            context_length=len(record.context_seq),
            clipped_start=record.clipped_start,
            clipped_end=record.clipped_end,
            window=window,
        )
        if bounds is None:
            raise SystemExit(
                f"record {record.record_id} admits no honest {window}-nt window; "
                "load_gate4_eval_records should have excluded it"
            )
        # `window_lead_range` deliberately admits windows that run past a *real contig end*
        # (there, zero-flanking is honest and `carve_window` pads with IGNORE_INDEX). A
        # benchmark contig is a DNA string with no pad alphabet, and a padded positive facing
        # unpadded negatives would be a length/composition confound, so the range is
        # intersected with the leads that keep the whole window inside the real context.
        # Every gate4_eval record carries >= 1,081 nt of context against a 1,024-nt window and
        # a <= 550-nt locus, so this intersection is never empty — asserted, not assumed.
        context_length = len(record.context_seq)
        lo = max(bounds[0], record.locus_offset - (context_length - window))
        hi = min(bounds[1], record.locus_offset)
        if lo > hi:
            raise SystemExit(
                f"record {record.record_id}: no in-context {window}-nt window contains the "
                f"locus (context {context_length} nt, locus at {record.locus_offset} "
                f"+{record.locus_length})"
            )
        lead = wd.deterministic_lead((lo, hi), window=window, locus_length=record.locus_length)
        start = record.locus_offset - lead
        sequence = record.context_seq[start : start + window].upper()
        if len(sequence) != window:
            raise SystemExit(
                f"record {record.record_id}: carved {len(sequence)} nt, expected {window}"
            )
        contig_id = f"pos:{record.record_id}"
        contigs.append(
            {
                "contig_id": contig_id,
                "sequence": sequence,
                "truth_start": lead,
                "truth_end": lead + record.locus_length,
                # `context_seq` is stored in the locus's own sense (flank_context
                # reverse-complements minus-strand rows on fetch), so every carved window
                # carries its locus on '+'.
                "truth_strand": "+",
                "truth_source_record_id": record.record_id,
                "truth_transform": "context_window",
            }
        )
        items.append(
            {
                "contig_id": contig_id,
                "label": 1,
                "source": "corpus",
                "pool": "corpus",
                "block": f"cluster:{int(record.cluster_id)}",
                "length": window,
                "locus_length": record.locus_length,
                "resolved_order": wd.record_order(record),
                "host_id": None,
                # A positive has no host window; the field is filled from the measurement
                # for negatives only (``host_fold_overlap``), never asserted.
                "host_seen_by_any_arm": None,
            }
        )
    return contigs, items


def build_negative_windows(
    decoy_rows: Sequence[Mapping[str, Any]],
    hosts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    all_corpus_ids: Collection[str],
    parent_cluster: Mapping[str, int],
    window: int = wd.WINDOW_NT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """One host window per §9.1 decoy, spliced by the **shipped** training operator.

    Three deliberate departures from the training call, each because this is a benchmark and
    not a training stream:

    1. **All four pools**, not ``TRAINING_DECOY_POOLS``. A7 pins 4–5 keep ``gc_background``
       and ``leader_decoy`` out of the *training* mix, and A7's own note says the
       ``gc_background`` pool is "retained unchanged as the ADR-0005 D7 GATE-1 **benchmark
       denominator**" — so excluding them here would drop the one negative class neither
       Stage-1 checkpoint has ever seen.
    2. **``unique_hosts=True``.** Host reuse is right for training (the decoy becomes the
       only difference between two windows) and wrong here: two benchmark negatives sharing
       a host are near-duplicates, which would make the block structure lie about how much
       independent evidence the CI has.
    3. **No training-fold restriction on the decoy's parent.** ``embed_decoy_rows`` refuses a
       parented decoy unless its parent is in the supplied fold, because embedding a
       permutation of held-out DNA into *training* is leakage. For an evaluation set the
       constraint is the opposite — held-out parentage is a virtue — so the full corpus id
       set is passed, which makes that admission vacuous **by construction and on purpose**.
       The count it would have refused under the training fold is recorded in the report.

    Hosts are supplied by the caller already filtered to ``parent_nested_train == False`` (and,
    under ``--host-screen``, to the ADR-0005 A15 screen's admitted windows — see
    :func:`mint_negative_windows`). The parent rule constrains the mining-pool record's PARENT,
    not the DNA of the window: v1 hosts were measured to share 32-mers with training DNA
    (:func:`host_fold_overlap`), which is why A15 screens the window itself.
    """
    embedded, report = emb.embed_decoy_rows(
        decoy_rows,
        hosts,
        seed=seed,
        window=window,
        pools=BENCHMARK_DECOY_POOLS,
        unique_hosts=True,
        training_fold_record_ids=all_corpus_ids,
    )
    by_id = {str(row[emb.DECOY_ID_COL]): row for row in decoy_rows}
    contigs: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for entry in embedded:
        contig_id = f"dec:{entry.insert_id}"
        contigs.append(
            {
                "contig_id": contig_id,
                "sequence": entry.sequence,
                "truth_start": None,
                "truth_end": None,
                "truth_strand": None,
                "truth_source_record_id": None,
                "truth_transform": "embedded_decoy_window",
            }
        )
        items.append(
            {
                "contig_id": contig_id,
                "label": 0,
                "source": "decoy",
                "pool": entry.insert_pool,
                "block": decoy_block_id(by_id[entry.insert_id], entry.host_id, parent_cluster),
                "length": len(entry.sequence),
                "locus_length": entry.insert_len,
                "resolved_order": None,
                "host_id": entry.host_id,
                # Placeholder: overwritten in main() from ``host_fold_overlap``, which needs
                # the finished contig set. A literal here was the defect (PR #133 round 2).
                "host_seen_by_any_arm": None,
                "splice_phase": entry.phase,
            }
        )
    return contigs, items, report


HOST_OVERLAP_K = 32

#: What ``scope.host_pool.overlap_with_training_folds`` measures, stated in the manifest.
HOST_OVERLAP_METHOD = (
    "verbatim forward-strand k-mers of the host portion (spliced insert excluded; the flanks "
    "either side of it are read separately, so no k-mer spans the excision) against the arm's "
    "own Stage-1 training-fold context_seq; any k-mer containing a non-ACGT character is "
    "skipped on both sides (ADR-0005 A15 Pin 1 alphabet rule: an N-run is not shared DNA)"
)

#: A maximal run of characters that cannot belong to a countable k-mer.
_NON_ACGT_RUN = re.compile(r"[^ACGT]+")


def acgt_kmers(sequence: str, k: int = HOST_OVERLAP_K) -> Iterator[str]:
    """Every ``k``-mer of ``sequence`` made only of A/C/G/T, in order.

    ADR-0005 A15 Pin 1's alphabet rule: a k-mer containing any non-ACGT character is skipped.
    Two windows that share a 32-nt run of ``N`` share an assembly gap, not DNA — counting it
    (as the shipped measure did) inflates the overlap with sequence nobody trained on. The
    caller must upper-case first: a lower-case base would be skipped as non-ACGT and the
    overlap under-counted, and both callers in this file upper-case before calling.
    """
    for run in _NON_ACGT_RUN.split(sequence):
        for offset in range(len(run) - k + 1):
            yield run[offset : offset + k]


def _coordinate(value: Any, name: str, contig_id: str) -> int:
    """One sequence coordinate, refused unless it *is* one.

    ``int()`` TRUNCATES — ``int(100.5)`` is ``100`` — so a fractional bound would slide past
    the interval check and excise the wrong window rather than be rejected
    ([[int-truncation-changes-a-pinned-value]], CodeRabbit PR #133). A coordinate is an
    integer; a bool is not a coordinate.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{contig_id}: {name} {value!r} is not a sequence coordinate")
    if isinstance(value, float) and not value.is_integer():
        raise SystemExit(f"{contig_id}: {name} {value!r} is fractional, not a coordinate")
    return int(value)


def host_fold_overlap(
    contigs: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    folds: Mapping[str, Collection[str]],
    context_by_id: Mapping[str, str],
    *,
    k: int = HOST_OVERLAP_K,
) -> dict[str, set[str]]:
    """Per arm, the negatives whose HOST DNA that arm's Stage-1 training fold already carries.

    Selecting hosts on ``parent_nested_train == False`` constrains the mining-pool record's
    **parent**; it does not make the DNA of the window drawn from it unseen. Measured here
    rather than asserted: the ``host_seen_by_any_arm: False`` this replaces was a hardcoded
    literal derived by nothing, and the disclosure beside it told the reader the negatives
    were "clean for BOTH Stage-1 checkpoints" (PR #133 round 2).

    The host portion **excludes the spliced insert**, so what is measured is the DNA the decoy
    was embedded into, not the decoy — decoy-level exposure is already ``seen_by_counts``. The
    two flanks are read separately: concatenating them (as the shipped measure did) manufactures
    junction k-mers that exist in neither the genome nor the contig.

    P3-16′-remint (ADR-0005 A15 Pin 1): k-mers containing a non-ACGT character are skipped
    (:func:`acgt_kmers`). Measured on the v1 reproduction this moves the shipped 112 / 96
    (production / twin) to 110 / 94; reading the flanks separately changes neither number. The
    semantics are otherwise the shipped ones (forward strand, host portion, each arm against its
    own fold) — this is the manifest's disclosure measure, not A15's both-strand whole-window
    screen, which lives in ``integration/host_screen.py``.
    """
    by_id = {contig["contig_id"]: contig for contig in contigs}
    owners: dict[str, set[str]] = {}
    for item in items:
        if item["label"] != 0:
            continue
        contig = by_id[item["contig_id"]]
        sequence = contig["sequence"].upper()
        start, end = contig.get("truth_start"), contig.get("truth_end")
        if start is None or end is None:
            # A negative contig carries NO truth bounds — the insert's location lives on the
            # ITEM, as `splice_phase` + `locus_length`. Falling back to the whole window (as
            # this did until round 8) counts the decoy's own k-mers as host DNA and makes the
            # published "spliced insert excluded" claim false. Refuse rather than fall back:
            # a measurement that silently changes what it measures is the defect, not the
            # missing field (CodeRabbit, PR #133 round 4, unresolved until round 8).
            phase, insert = item.get("splice_phase"), item.get("locus_length")
            if phase is None or insert is None:
                raise SystemExit(
                    f"{item['contig_id']}: no splice bounds on contig or item, so the host "
                    "portion cannot be isolated; measuring the whole window would count the "
                    "decoy as host DNA"
                )
            start = _coordinate(phase, "splice_phase", item["contig_id"])
            end = start + _coordinate(insert, "locus_length", item["contig_id"])
        # Python slicing CLAMPS: a negative or overlong bound would quietly excise the wrong
        # interval and publish an overlap count for a host sequence that does not exist. On
        # the one function in this file that has already published a wrong number, the bounds
        # are checked rather than trusted (CodeRabbit, PR #133).
        start = _coordinate(start, "start", item["contig_id"])
        end = _coordinate(end, "end", item["contig_id"])
        if not 0 <= start < end <= len(sequence):
            raise SystemExit(
                f"{item['contig_id']}: splice bounds [{start}, {end}) are outside the "
                f"{len(sequence)}-nt contig"
            )
        for flank in (sequence[:start], sequence[end:]):
            for kmer in acgt_kmers(flank, k):
                owners.setdefault(kmer, set()).add(item["contig_id"])
    seen: dict[str, set[str]] = {arm: set() for arm in folds}
    for arm, fold in folds.items():
        for record_id in fold:
            sequence = context_by_id.get(record_id)
            if sequence is None:
                continue
            # Skipped on this side too: an owner key never contains a non-ACGT character, so
            # this changes no count — it states the rule where a reader looks for it.
            for kmer in acgt_kmers(sequence.upper(), k):
                hit = owners.get(kmer)
                if hit:
                    seen[arm] |= hit
    return seen


def item_gated(item: Mapping[str, Any]) -> bool:
    """Whether one benchmark item enters the GATED denominator (ADR-0005 A15 Pin 4).

    False iff its pool is in :data:`NON_GATED_POOLS` (``leader_decoy``); True for every other
    item, positives included. A strict bool, never a truthy stand-in: a consumer filters on it.
    """
    return str(item["pool"]) not in NON_GATED_POOLS


def gated_scope(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The gated subset's counts, derived from the items' own ``gated`` flags.

    Declared in the manifest so the precision report's completeness clauses compare against a
    number the MINT wrote, not one each consumer re-derives (A15 Pin 4). Counted from the flags
    rather than from the pool names, so a flag that disagrees with this rule cannot hide behind
    a scope computed some other way.
    """
    gated = [item for item in items if item["gated"] is True]
    negatives = [item for item in gated if item["label"] == 0]
    return {
        "n_items_gated": len(gated),
        "n_negatives_gated": len(negatives),
        "negatives_by_pool_gated": {
            pool: sum(1 for item in negatives if item["pool"] == pool)
            for pool in sorted({item["pool"] for item in negatives})
        },
        "n_blocks_gated": len({item["block"] for item in gated}),
        "gated_rule": GATED_RULE,
    }


def build(
    positives: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    negatives: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    decoys: pd.DataFrame,
    folds: dict[str, set[str]],
    n_nested_train_rows: int,
    *,
    screened: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    decoy_meta = decoys.set_index("decoy_id")
    pos_contigs, pos_items = positives
    neg_contigs, neg_items, embed_report = negatives

    for item in pos_items:
        record_id = item["contig_id"].removeprefix("pos:")
        item["seen_by"] = {arm: record_id in fold for arm, fold in sorted(folds.items())}
    for item in neg_items:
        decoy_row = decoy_meta.loc[item["contig_id"].removeprefix("dec:")]
        # `seen_by` is about the INSERT — whether that decoy went into the arm's training
        # stream. Host-DNA exposure is a separate MEASUREMENT (`host_fold_overlap` ->
        # `host_seen_by_any_arm`), never a by-construction claim: `parent_nested_train ==
        # False` constrains the host's parent, not its DNA. Kept apart so the two exposures
        # are never conflated.
        item["seen_by"] = {
            arm: decoy_embedded(decoy_row, fold) for arm, fold in sorted(folds.items())
        }

    contigs = pos_contigs + neg_contigs
    items = pos_items + neg_items
    # Set AFTER the draw and the mint, on the finished items: gating is a property of the
    # denominator, never of the draw, so the leader keeps its host slot (A15 Pin 3/4).
    for item in items:
        item["gated"] = item_gated(item)
    construction = (
        "data/embedding.py::embed_decoy_rows — the shipped ADR-0005 A7 splice, all four "
        "§9.1 pools, unique hosts, hosts drawn parent_nested_train == False"
    )
    if screened:
        construction += " and admitted by the ADR-0005 A15 host screen before the draw"
    scope = {
        "rung": RUNG,
        "geometry": f"{wd.WINDOW_NT}-nt scan windows (ADR-0005 D3 / PRD §6 pinned geometry)",
        "positive_population": "gate4_eval (window_dataset.load_gate4_eval_records)",
        "negative_construction": construction,
        "n_positives": len(pos_items),
        "n_negatives": len(neg_items),
        "n_items": len(items),
        "n_blocks": len({item["block"] for item in items}),
        "n_positive_blocks": len({item["block"] for item in items if item["label"] == 1}),
        "negatives_by_pool": {
            pool: sum(1 for item in neg_items if item["pool"] == pool)
            for pool in sorted({item["pool"] for item in neg_items})
        },
        # The three records the shipped loader drops for short genomic context. Recorded, not
        # absorbed: a reader comparing 1,201 against the item table's 1,204 must not have to
        # rediscover why they differ.
        "n_nested_train_test_rows": n_nested_train_rows,
        "n_dropped_vs_item_table": n_nested_train_rows - len(pos_items),
        "dropped_reason": "context_shorter_than_window (load_gate4_eval_records filter 3)",
        "training_fold_sizes": {arm: len(fold) for arm, fold in sorted(folds.items())},
        "embedding_report": {
            key: value
            for key, value in embed_report.items()
            if key not in ("unique_hosts_available",)
        },
        "seen_by_counts": {
            arm: {
                "positives": sum(1 for i in items if i["label"] == 1 and i["seen_by"][arm]),
                "negatives": sum(1 for i in items if i["label"] == 0 and i["seen_by"][arm]),
            }
            for arm in sorted(folds)
        },
        **gated_scope(items),
    }
    return contigs, items, scope


def check_embedding_derivation(decoys: pd.DataFrame, folds: dict[str, set[str]]) -> dict[str, Any]:
    """Refuse unless :func:`decoy_embedded` reproduces both committed runs' pool counts."""
    measured: dict[str, dict[str, int]] = {}
    for arm, fold in sorted(folds.items()):
        embedded = decoys[decoys.apply(lambda row, fold=fold: decoy_embedded(row, fold), axis=1)]
        measured[arm] = {
            pool: int(count) for pool, count in embedded["pool"].value_counts().items()
        }
    for arm, expected in EXPECTED_EMBEDDED.items():
        if measured.get(arm) != expected:
            report_name = "production" if arm == "production" else "gate4_twin"
            raise SystemExit(
                f"embedding derivation does not reproduce the committed {arm} run: derived "
                f"{measured.get(arm)!r}, reports/p2/train_stage1_{report_name}.json records "
                f"{expected!r}. Refusing to mint a benchmark whose exposure flags are not "
                "bound to what was actually trained on."
            )
    return measured


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_decoy_rows(decoys: pd.DataFrame, wanted_ids: Collection[str]) -> list[dict[str, Any]]:
    """The test-rung decoy rows — ALL four pools — in ``decoy_id`` order.

    The order IS the draw: under ``unique_hosts=True`` the i-th admitted decoy takes host
    ``permutation[i]``. So the ``leader_decoy`` row stays here at its sorted position even
    though it is reported beside the gate (ADR-0005 A15 Pins 3–4); filtering it out would
    re-index every host after it, and v2 would silently stop being "v1's construction, screened".
    """
    wanted = {str(value) for value in wanted_ids}
    rows = [
        {key: row[key] for key in decoys.columns}
        for _, row in decoys[decoys["decoy_id"].astype(str).isin(wanted)]
        .sort_values("decoy_id", kind="mergesort")
        .iterrows()
    ]
    if len(rows) != len(wanted):
        raise SystemExit(f"{len(wanted)} test-rung decoys but {len(rows)} matched in decoys_v0")
    return rows


def _repo_relative_problem(name: str, value: Any) -> str | None:
    """Why ``value`` is not a repo-relative path, or ``None`` if it is."""
    if not isinstance(value, str) or not value.strip():
        return f"{name} {value!r} is not a path"
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return f"{name} must be repo-relative (got {value!r})"
    return None


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def host_screen_refusals(
    report: Any,
    *,
    mining_pool_sha256: str,
    n_candidates: int,
    admitted_list_sha256: str | None,
    screened_dna_sha256: Mapping[str, str] | None = None,
) -> list[str]:
    """Why a ``--host-screen`` report may NOT be drawn from; ``[]`` means it may.

    Pure and total: it returns problems and never raises on a payload this script did not
    write. A malformed field is a refusal, never an absence ([[total-validator-can-fail-open]]).
    Each clause binds the screen to THIS mint's subject rather than to itself
    ([[gate-must-bind-to-upstream-evidence]]):

    * ``problems == []`` and ``controls_all_fired is True`` — and the control list is
      re-derived: non-empty, with every entry ``fired is True``. A flag can be written TRUE by
      a report whose controls say otherwise ([[gate-clauses-need-re-derivation]]), and
      ``all()`` over no controls is vacuously true ([[clauses-must-guard-emptiness]]);
    * ``content_screen.threshold_bits == 10.0`` (A15 Pin 2), a number and not a bool
      (``True == 1.0`` passes an equality test, [[bool-equals-one-passes-config-gates]]);
    * ``pool.sha256`` is the digest of the ``--mining-pool`` bytes, and ``pool.n_candidates``
      is the out-of-fold pool this mint built, so the screen ran over the pool being drawn from;
    * ``admitted.sha256`` is the digest of the admitted-list bytes on disk;
    * every ``screened_dna.sources`` digest the screen recorded is the digest of the file THIS
      mint was given. The pool digest and the candidate count alone do not cover the screened
      DNA: a changed ``decoys_v0`` or ``stage2_dataset`` leaves both unchanged while altering
      which windows Pin 1 refuses, so the admitted list would be certified against a different
      screened-DNA union than the one it was built from.
    """
    if not isinstance(report, Mapping):
        return [f"the host-screen report is a {type(report).__name__}, not a JSON object"]
    problems: list[str] = []
    listed = report.get("problems")
    if not (isinstance(listed, list) and not listed):
        problems.append(f"the host-screen report's problems are {listed!r}, not []")
    controls = report.get("controls")
    if not isinstance(controls, list) or not controls:
        problems.append(
            "the host-screen report lists no must-fire controls; an empty control set "
            "certifies nothing"
        )
    else:
        unfired = [
            (control.get("name") if isinstance(control, Mapping) else repr(control))
            for control in controls
            if not (isinstance(control, Mapping) and control.get("fired") is True)
        ]
        if unfired:
            problems.append(f"must-fire control(s) did not fire: {unfired!r}")
    if report.get("controls_all_fired") is not True:
        problems.append(
            f"controls_all_fired is {report.get('controls_all_fired')!r}, not True "
            "(ADR-0005 A15 Pin 1)"
        )
    content = report.get("content_screen")
    threshold = content.get("threshold_bits") if isinstance(content, Mapping) else None
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or threshold != HOST_SCREEN_THRESHOLD_BITS
    ):
        problems.append(
            f"content_screen.threshold_bits is {threshold!r}, not the pinned "
            f"{HOST_SCREEN_THRESHOLD_BITS} bits (ADR-0005 A15 Pin 2)"
        )
    pool = report.get("pool") if isinstance(report.get("pool"), Mapping) else {}
    if pool.get("sha256") != mining_pool_sha256:
        problems.append(
            f"the screen ran over a pool with sha256 {pool.get('sha256')!r}, not this "
            f"--mining-pool ({mining_pool_sha256})"
        )
    if not _is_count(pool.get("n_candidates")) or pool.get("n_candidates") != n_candidates:
        problems.append(
            f"the screen's candidate pool holds {pool.get('n_candidates')!r} windows, not the "
            f"{n_candidates} out-of-fold windows this mint draws from"
        )
    admitted = report.get("admitted") if isinstance(report.get("admitted"), Mapping) else {}
    path_problem = _repo_relative_problem("admitted.list", admitted.get("list"))
    if path_problem:
        problems.append(path_problem)
    if admitted_list_sha256 is None:
        problems.append(f"the admitted list {admitted.get('list')!r} could not be read")
    elif admitted.get("sha256") != admitted_list_sha256:
        problems.append(
            f"the admitted list's bytes hash to {admitted_list_sha256}, but the screen "
            f"report records {admitted.get('sha256')!r}"
        )
    if not _is_count(admitted.get("n")):
        problems.append(f"admitted.n is {admitted.get('n')!r}, not a count")
    # Bind every screened-DNA digest the screen recorded to the file THIS mint was handed.
    # The pool digest and the candidate count do NOT cover the screened DNA: changing
    # decoys_v0 or stage2_dataset leaves both intact while altering which windows Pin 1
    # refuses, so the admitted list would be certified against a different union than the one
    # that produced it ([[gate-must-bind-to-upstream-evidence]]). A digest the screen recorded
    # but this mint cannot supply is a refusal, not a skip.
    sources = report.get("screened_dna")
    sources = sources.get("sources") if isinstance(sources, Mapping) else None
    sources = sources if isinstance(sources, Mapping) else {}
    for source, field, actual in sorted(screened_dna_sha256 or ()):
        block = sources.get(source)
        recorded = block.get(field) if isinstance(block, Mapping) else None
        if not isinstance(recorded, str):
            problems.append(
                f"the host-screen report records no screened_dna.sources.{source}.{field}, "
                "so the screened DNA it ran over cannot be bound to this mint's inputs"
            )
        elif recorded != actual:
            problems.append(
                f"screened_dna.sources.{source}.{field} is {recorded[:12]}…, but this mint's "
                f"file hashes {actual[:12]}… — the screen ran over different screened DNA"
            )

    return problems


def parse_admitted_list(text: str, *, expected_n: Any) -> tuple[list[str], list[str]]:
    """``(ids, problems)`` for the screen's admitted list: sorted ids, one per line, final LF.

    Strict on purpose. A duplicated, blank, padded or unsorted line means the file is not the
    one the screen wrote, and a count that disagrees with ``admitted.n`` means the report and
    the list describe different screens.
    """
    problems: list[str] = []
    if not text.endswith("\n"):
        problems.append("the admitted list does not end with a newline")
    ids = text.split("\n")
    if ids and ids[-1] == "":
        ids = ids[:-1]
    if any(not line.strip() for line in ids):
        problems.append("the admitted list has a blank line")
    if any(line != line.strip() for line in ids):
        problems.append("the admitted list has an id with surrounding whitespace")
    if len(set(ids)) != len(ids):
        problems.append(f"the admitted list repeats {len(ids) - len(set(ids))} id(s)")
    if ids != sorted(ids):
        problems.append("the admitted list is not sorted")
    if not _is_count(expected_n) or len(ids) != expected_n:
        problems.append(f"the admitted list holds {len(ids)} ids, the report says {expected_n!r}")
    return ids, problems


def read_host_screen(
    screen: str,
    *,
    reports_root: Path,
    mining_pool_sha256: str,
    n_candidates: int,
    screened_dna_sha256: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Load ``--host-screen`` and its admitted list; refuse unless every clause holds.

    Returns ``(admitted_ids, record)``; ``record`` becomes ``scope.host_pool.screen``. Both
    files are committed (``reports/``), so they resolve against ``reports_root`` — this
    checkout by default, not ``--data-root``, which holds the DVC data.
    """
    problem = _repo_relative_problem("--host-screen", screen)
    if problem:
        raise SystemExit(problem)
    try:
        raw = (reports_root / screen).read_bytes()
    except OSError as exc:
        raise SystemExit(f"--host-screen {screen!r} could not be read: {exc}") from exc
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--host-screen {screen!r} is not a JSON report: {exc}") from exc
    admitted = report.get("admitted") if isinstance(report, Mapping) else None
    admitted = admitted if isinstance(admitted, Mapping) else {}
    list_path = admitted.get("list")
    list_bytes: bytes | None = None
    if _repo_relative_problem("admitted.list", list_path) is None:
        try:
            list_bytes = (reports_root / list_path).read_bytes()
        except OSError:
            list_bytes = None
    list_sha = None if list_bytes is None else hashlib.sha256(list_bytes).hexdigest()
    problems = host_screen_refusals(
        report,
        mining_pool_sha256=mining_pool_sha256,
        n_candidates=n_candidates,
        admitted_list_sha256=list_sha,
        screened_dna_sha256=screened_dna_sha256,
    )
    ids: list[str] = []
    if list_bytes is not None:
        try:
            text = list_bytes.decode("utf-8")
        except UnicodeDecodeError:
            problems.append("the admitted list is not UTF-8 text")
        else:
            ids, list_problems = parse_admitted_list(text, expected_n=admitted.get("n"))
            problems.extend(list_problems)
    if problems:
        raise SystemExit(f"refusing --host-screen {screen}: " + "; ".join(problems))
    record = {
        "report": screen,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "admitted_list": list_path,
        "admitted_sha256": list_sha,
        "n_admitted": len(ids),
        "threshold_bits": float(report["content_screen"]["threshold_bits"]),
        "adr": HOST_SCREEN_ADR,
    }
    return ids, record


def apply_host_screen(
    hosts: Sequence[Mapping[str, Any]], admitted_ids: Collection[str]
) -> list[Mapping[str, Any]]:
    """Restrict the out-of-fold hosts to the screen's admitted windows, in POOL order.

    Pool (file) order, not the admitted list's sorted order: the seeded permutation indexes
    this list, so its order is part of the draw, and pool order is what v1 drew over. An
    admitted id the out-of-fold pool does not carry is refused, not skipped: it means the
    screen ran over a different pool than this mint draws from.
    """
    pool_ids = {str(row["candidate_id"]) for row in hosts}
    keep = {str(value) for value in admitted_ids}
    outside = sorted(keep - pool_ids)
    if outside:
        raise SystemExit(
            f"{len(outside)} admitted host id(s) are not in the out-of-fold candidate pool "
            f"(e.g. {outside[:3]!r}); the screen did not run over the pool this mint draws from"
        )
    return [row for row in hosts if str(row["candidate_id"]) in keep]


def mint_negative_windows(
    decoy_rows: Sequence[Mapping[str, Any]],
    hosts: Sequence[Mapping[str, Any]],
    *,
    admitted_ids: Collection[str] | None,
    seed: int,
    all_corpus_ids: Collection[str],
    parent_cluster: Mapping[str, int],
    window: int = wd.WINDOW_NT,
) -> tuple[
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    list[Mapping[str, Any]],
]:
    """The draw: the A15 screen filter FIRST, then the shipped splice over what survived.

    ``admitted_ids=None`` is v1's construction (the whole out-of-fold pool). Otherwise the pool
    is filtered BEFORE :func:`build_negative_windows`, so the seeded permutation runs over the
    screened pool. Filtering the finished items instead would drop negatives after the draw,
    which A15 Pin 3 forbids. A changed permutation length redraws all 692 hosts; that is
    expected, not a defect.

    Returns ``(negatives, drawable_hosts)``.
    """
    drawable = list(hosts) if admitted_ids is None else apply_host_screen(hosts, admitted_ids)
    if len(drawable) < len(decoy_rows):
        raise SystemExit(
            f"{len(drawable)} drawable host windows for {len(decoy_rows)} decoys; "
            "unique_hosts=True needs at least one each"
        )
    negatives = build_negative_windows(
        decoy_rows,
        drawable,
        seed=seed,
        all_corpus_ids=all_corpus_ids,
        parent_cluster=parent_cluster,
        window=window,
    )
    return negatives, drawable


#: The parent attributes ``host_composition`` tabulates (A15 Pin 7(e)).
COMPOSITION_PARENT_FIELDS: tuple[str, ...] = (
    "nested_role",
    "is_designated_loo_holdout",
    "resolved_order",
)
ABSENT_PARENT = "absent_from_split_table"


def _composition_label(value: Any) -> str:
    """A JSON-key label for one attribute value; a null is ``"null"``, never a dropped row."""
    if is_missing(value):
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def host_composition(
    items: Sequence[Mapping[str, Any]],
    host_by_id: Mapping[str, Mapping[str, Any]],
    parent_meta: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Composition of the DRAWN hosts — one per negative — for A15 Pin 7(e).

    ``side`` (lead / trail) from the host row; from the split table, via each host's
    ``source_record_id``: the parent's ``nested_role``, whether it is a designated
    leave-clade-out holdout, and its ``resolved_order``. A parent missing from the split table
    is counted as ``absent_from_split_table``, never dropped, so every block sums to the number
    of negatives ([[nulls-inflate-block-counts]] — a null is a labelled bucket, not a gap).
    """
    side: dict[str, int] = {}
    parents: dict[str, dict[str, int]] = {field: {} for field in COMPOSITION_PARENT_FIELDS}
    n_hosts = 0
    for item in items:
        if item["label"] != 0:
            continue
        host = host_by_id.get(str(item["host_id"]))
        if host is None:
            raise SystemExit(
                f"{item['contig_id']}: drawn host {item['host_id']!r} is not in the drawable pool"
            )
        n_hosts += 1
        label = _composition_label(host.get("side"))
        side[label] = side.get(label, 0) + 1
        meta = parent_meta.get(row_text(host.get("source_record_id")).strip())
        for field in COMPOSITION_PARENT_FIELDS:
            label = ABSENT_PARENT if meta is None else _composition_label(meta.get(field))
            parents[field][label] = parents[field].get(label, 0) + 1
    return {
        "n_hosts": n_hosts,
        "side": dict(sorted(side.items())),
        **{f"parent_{field}": dict(sorted(parents[field].items())) for field in parents},
    }


def host_pool_scope(
    *,
    mining_pool: str,
    n_admitted: int,
    n_out_of_fold: int,
    n_drawable: int,
    n_used: int,
    screen: Mapping[str, Any] | None,
    composition: Mapping[str, Any],
    host_seen: Mapping[str, Collection[str]],
    n_negatives: int,
) -> dict[str, Any]:
    """``scope.host_pool``: what the selection constrains, and what was measured.

    ``selection`` names the A15 screen only when one was applied; neither string calls the
    hosts "clean" for any arm, because a selection rule is not a measurement of what the drawn
    windows contain (PR #133 round 2). ``screen`` is ``None`` for an unscreened (v1) mint —
    stated, not omitted ([[clauses-must-guard-emptiness]]).
    """
    seen_any = set().union(*host_seen.values()) if host_seen else set()
    return {
        "source": mining_pool,
        "n_admitted": n_admitted,
        "n_out_of_fold_available": n_out_of_fold,
        # The permutation's length: the out-of-fold pool, or what the screen admitted from it.
        "n_available_to_draw": n_drawable,
        "n_used": n_used,
        "selection": UNSCREENED_SELECTION if screen is None else SCREENED_SELECTION,
        "screen": None if screen is None else dict(screen),
        "composition": dict(composition),
        "overlap_with_training_folds": {
            "k": HOST_OVERLAP_K,
            "method": HOST_OVERLAP_METHOD,
            "by_arm": {arm: len(ids) for arm, ids in sorted(host_seen.items())},
            "n_seen_by_any_arm": len(seen_any),
            "n_negatives": n_negatives,
        },
    }


def normalise_insert(sequence: Any) -> str:
    """An insert's sequence as DNA text — the one form both sides of the duplicate test share."""
    return str(sequence).strip().upper().replace("U", "T")


def insert_exposure(
    items: Sequence[Mapping[str, Any]],
    decoy_sequences: Mapping[str, Any],
    stage2_training_decoys: Collection[Any],
) -> dict[str, Any]:
    """A15 Pin 7(c)'s third count: test inserts that duplicate a Stage-2 training decoy.

    The other two Pin 7(c) counts are the per-arm ``seen_by`` flags already in ``scope``. This
    one is measured by sequence: a duplicate is a test decoy whose insert the shipped Stage-2
    saw verbatim under another id. The headline counts the GATED negatives; the non-gated leader
    is recorded beside it, as everything about the leader is (Pin 4).
    """
    training = {normalise_insert(sequence) for sequence in stage2_training_decoys}
    if not training:
        raise SystemExit(
            "no Stage-2 training decoys to compare against: the duplicate count would be a "
            "vacuous 0, not a measurement"
        )
    by_pool: dict[str, dict[str, int]] = {}
    non_gated: dict[str, bool] = {}
    n_gated = n_duplicating = 0
    for item in items:
        if item["label"] != 0:
            continue
        decoy_id = str(item["contig_id"]).removeprefix("dec:")
        if decoy_id not in decoy_sequences:
            raise SystemExit(f"{item['contig_id']}: its insert {decoy_id!r} is not in decoys_v0")
        duplicate = normalise_insert(decoy_sequences[decoy_id]) in training
        if not item_gated(item):
            non_gated[str(item["contig_id"])] = duplicate
            continue
        n_gated += 1
        n_duplicating += duplicate
        counts = by_pool.setdefault(str(item["pool"]), {"n": 0, "n_duplicating": 0})
        counts["n"] += 1
        counts["n_duplicating"] += duplicate
    return {
        "rule": INSERT_EXPOSURE_RULE,
        "n_stage2_training_decoy_sequences": len(training),
        "n_gated_negatives": n_gated,
        "n_test_inserts_duplicating_stage2_training_decoys": n_duplicating,
        "by_pool_gated": dict(sorted(by_pool.items())),
        "non_gated": dict(sorted(non_gated.items())),
    }


def stage2_training_decoys(dataset: pd.DataFrame) -> list[str]:
    """The decoy sequences of the shipped Stage-2 training population, through its own selector."""
    from tbox_finder.stage2 import train as stage2_train  # lazy: torch-free, but heavy

    rows = dataset.to_dict("records")
    admitted, _ = stage2_train.select_rows(rows, rung=stage2_train.TRAIN_RUNG)
    if len(admitted) != STAGE2_TRAINING_ROWS:
        raise SystemExit(
            f"Stage-2 training population is {len(admitted)} rows, not the {STAGE2_TRAINING_ROWS} "
            "the shipped checkpoint trained on; refusing to count insert duplicates against it"
        )
    return [rows[i]["rna_sequence"] for i in admitted if rows[i]["source"] == "decoy"]


def v1_out_violation(out: Path) -> str | None:
    """Why ``--out`` names a v1 artifact, or ``None`` (ADR-0005 A15 Pin 5).

    v1 stays byte-unchanged as the record of the withdrawn grade, and P3-17a hash-binds its
    files BY FIXED PATH (``cm_confirmer.CANONICAL_SOURCES``), so overwriting one would make a
    committed report silently describe v2 bytes. Matched on the resolved path's trailing
    components, so a canonical path is refused under any checkout root, and on the
    ``benchmark_v1.json`` suffix.
    """
    resolved = out.resolve()
    if str(resolved).endswith(V1_OUT_SUFFIX):
        return f"--out {out} ends with {V1_OUT_SUFFIX!r}; no v2 artifact is written to a v1 path"
    for key, relative in sorted(CANONICAL_SOURCES.items()):
        parts = Path(relative).parts
        if resolved.parts[-len(parts) :] == parts:
            return (
                f"--out {out} resolves to {relative}, a fixed-path input of the committed "
                f"P3-17a report (cm_confirmer.CANONICAL_SOURCES[{key!r}]); ADR-0005 A15 Pin 5 "
                "keeps it byte-unchanged"
            )
    return None


# ══════════════════════════════════════════════════════════════════════════════════════
# P3-16″-hardneg — ADR-0005 A17's hard-negative benchmark (``--hardneg``)
# ══════════════════════════════════════════════════════════════════════════════════════
#: The A17 manifest: schema 1.2 AND this kind. ``precision.manifest_is_hardneg`` reads both and
#: refuses either alone, so the 1.3 clause set is reached by this mint and by nothing else.
HARDNEG_SCHEMA_VERSION = "1.2"
HARDNEG_KIND = "hardneg_benchmark"
HARDNEG_STEP = "P3-16″-hardneg"
HARDNEG_POOL_KIND = "hardneg_pool"
#: ``hardneg_pool``'s two selection modes, and the item-file roles each pool writes.
HARDNEG_POOL_MODES: tuple[str, ...] = ("draw", "census")
HARDNEG_ROLES: tuple[str, ...] = ("graded", "beside")
#: Every field a pool item carries (``hardneg_pool._cmd_report``'s ``keep``).
HARDNEG_ITEM_KEYS: tuple[str, ...] = (
    "item_id",
    "assembly",
    "seqid",
    "start",
    "end",
    "strand",
    "feature_type",
    "feature_id",
    "delta",
    "sub_class",
    "d4_context",
    "sha256",
    "sequence",
)
HARDNEG_SUB_CLASS = {"CDS": "cds_upstream", "tRNA": "trna_upstream"}
HARDNEG_DELTA_MAX = 256
HARDNEG_SELECTION = (
    "ADR-0005 A17 — genuine §9.1 class-4 leaders: native 1,024-nt CDS- and tRNA-upstream "
    "windows on their feature's own strand (δ ~ U{0..256}, seed 42) on the 339 NCBI-annotated "
    "admissible a2 hosts, a 100,000-window seed-42 uniform draw (plus, on the one pre-registered "
    "extension, every remaining eligible window), screened by A15 Pins 1–2 unchanged; "
    "D4-context windows minted beside the graded denominator; no leader decoy, no §9.1 decoy "
    "pool; the selection reads no score"
)
HARDNEG_GATED_RULE = (
    "ADR-0005 A17 Pin 2 — D4-context windows minted and scored, reported beside the graded "
    "denominator; every positive and every non-D4-context window is graded"
)
HARDNEG_OVERLAP_METHOD = (
    "canonical N-free 32-mers on either strand of the WHOLE native window (no insert to "
    "exclude) against the arm's own Stage-1 training-fold context_seq "
    "(host_screen.canonical_kmers / sorted_unique_kmers / kmer_hits)"
)
HARDNEG_SEEN_BY_RULE = (
    "seen_by[arm] on a native window is the MEASURED fold overlap (any shared canonical 32-mer "
    "with the arm's training-fold context_seq), never a by-construction claim"
)


def hardneg_out_violation(out: Path) -> str | None:
    """Why ``--out`` may not hold the A17 manifest, or ``None`` (A17 Pin 5: no v1 or v2 path).

    v1 is refused by :func:`v1_out_violation`; v2's name (``benchmark_v2.json``, ``v2_*``) is
    refused here; and every A17 Pin 5 path names ``hardneg``, so one that does not is refused.
    """
    violation = v1_out_violation(out)
    if violation:
        return violation
    name = out.resolve().name
    if name.startswith("benchmark_v2") or name.startswith("v2_"):
        return f"--out {out} is a v2 artifact name; A17 writes beside v2, never over it"
    if "hardneg" not in name:
        return f"--out {out} does not name 'hardneg'; every A17 Pin 5 path does"
    return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hardneg_item_problems(item: Any, *, role: str) -> list[str]:
    """Why one pool item cannot be minted as written, or ``[]``. Total: never raises.

    The window's bytes must hash to the pool's recorded ``sha256`` — A17 Pin 2: "a native
    window's screened bytes ARE its scored bytes". They must also be upper-case ``ACGT`` only:
    the Stage-1 encoder maps any other character (a soft-masked ``a`` included) to ``N``, so a
    lower-case window would be scored as a different sequence from the one that was screened.
    """
    if not isinstance(item, Mapping):
        return [f"a {role} item is a {type(item).__name__}, not an object"]
    where = f"{role} item {item.get('item_id')!r}"
    problems: list[str] = []
    missing = [key for key in HARDNEG_ITEM_KEYS if key not in item]
    if missing:
        return [f"{where} carries no {missing}"]
    for key in ("item_id", "assembly", "seqid", "feature_id", "sha256", "sequence"):
        if not isinstance(item[key], str) or not item[key]:
            problems.append(f"{where}: {key} {item[key]!r} is not a non-empty string")
    coords = {key: item[key] for key in ("start", "end", "delta")}
    if not all(_is_count(value) for value in coords.values()):
        return [*problems, f"{where}: {coords!r} are not non-negative integer coordinates"]
    if item["end"] - item["start"] != wd.WINDOW_NT:
        problems.append(f"{where}: [{item['start']}, {item['end']}) is not {wd.WINDOW_NT} nt")
    if item["delta"] > HARDNEG_DELTA_MAX:
        problems.append(f"{where}: delta {item['delta']} is outside U{{0..{HARDNEG_DELTA_MAX}}}")
    if item["strand"] not in ("+", "-"):
        problems.append(f"{where}: strand {item['strand']!r} is not '+' or '-'")
    if HARDNEG_SUB_CLASS.get(item["feature_type"]) != item["sub_class"]:
        problems.append(
            f"{where}: feature_type {item['feature_type']!r} with sub_class "
            f"{item['sub_class']!r} is not one of A17 Pin 1's two leader sub-classes"
        )
    if item["d4_context"] is not (role == "beside"):
        problems.append(
            f"{where}: d4_context={item['d4_context']!r} in the {role} list; D4-context is "
            "exactly the beside partition (A17 Pin 2)"
        )
    if problems:
        return problems
    expected_id = (
        f"{item['assembly']}:{item['seqid']}:{item['start']}-{item['end']}:{item['strand']}"
    )
    if item["item_id"] != expected_id:
        problems.append(f"{where}: item_id is not its window's key {expected_id!r}")
    sequence = item["sequence"]
    if len(sequence) != wd.WINDOW_NT:
        problems.append(f"{where}: carries {len(sequence)} nt, not {wd.WINDOW_NT}")
    if set(sequence) - set("ACGT"):
        problems.append(
            f"{where}: carries {sorted(set(sequence) - set('ACGT'))!r}; the Stage-1 encoder reads "
            "any character but upper-case ACGT as N, so it would score a sequence that was "
            "never screened"
        )
    if _sha256_text(sequence) != item["sha256"]:
        problems.append(f"{where}: its bytes do not hash to the pool's sha256 {item['sha256']!r}")
    return problems


def read_hardneg_items(raw: bytes, *, role: str, where: str) -> list[dict[str, Any]]:
    """One pool item file (``{"partition": role, "items": [...]}``), refused unless every item
    passes :func:`hardneg_item_problems`."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{where} is not a JSON item list: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("partition") != role:
        raise SystemExit(f"{where} is not the pool's {role!r} item list")
    items = payload.get("items")
    if not isinstance(items, list):
        raise SystemExit(f"{where} carries no items list")
    problems = [problem for item in items for problem in hardneg_item_problems(item, role=role)]
    if problems:
        raise SystemExit(f"refusing {where}: " + "; ".join(problems[:10]))
    return [dict(item) for item in items]


def hardneg_pool_refusals(
    report: Any, *, mode: str, item_digests: Mapping[str, tuple[str, int]], validate: Any
) -> list[str]:
    """Why a pool report may NOT be minted from; ``[]`` means it may. Total: never raises.

    ``validate`` is the pool's OWN validator (``hardneg_pool.validate_report``): a pool it
    refuses is refused here. Beyond it, the report must be of ``mode`` and must name, by sha256
    and count, exactly the item files this mint was handed (``item_digests``: role ->
    ``(sha256, n)``), so the windows minted are the windows the pool partitioned.
    """
    if not isinstance(report, Mapping):
        return [f"the pool report is a {type(report).__name__}, not a JSON object"]
    if report.get("kind") != HARDNEG_POOL_KIND:
        return [f"the pool report's kind is {report.get('kind')!r}, not {HARDNEG_POOL_KIND!r}"]
    try:
        problems = [f"its own validator: {problem}" for problem in validate(report)]
    except Exception as exc:  # a validator that raises has validated nothing
        problems = [f"its own validator raised {exc!r}"]
    config = report.get("config") if isinstance(report.get("config"), Mapping) else {}
    if config.get("mode") != mode:
        problems.append(f"the pool's mode is {config.get('mode')!r}, not {mode!r}")
    files = report.get("item_files") if isinstance(report.get("item_files"), Mapping) else {}
    for role in HARDNEG_ROLES:
        entry = files.get(role) if isinstance(files.get(role), Mapping) else {}
        digest, count = item_digests.get(role, (None, None))
        if entry.get("sha256") != digest or entry.get("n") != count:
            problems.append(
                f"the pool records its {role} list as {entry.get('sha256')!r} ({entry.get('n')!r} "
                f"items); the file given hashes {digest!r} ({count!r} items)"
            )
    return problems


def native_fold_overlap(
    contigs: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    folds: Mapping[str, Collection[str]],
    context_by_id: Mapping[str, str],
) -> dict[str, set[str]]:
    """Per arm, the native negatives sharing a canonical 32-mer with that arm's training fold.

    The v2 measure (:func:`host_fold_overlap`) excises a spliced insert; a native window has
    none, so the WHOLE window is host DNA, read on both strands with ``host_screen``'s own
    vectorised k-mer index. The pool's fold screen refused every window sharing a 32-mer with the
    four-source training union, so this is expected to be 0 — measured, not assumed.
    """
    from tbox_finder.integration import host_screen as hs  # lazy: numpy only

    by_id = {str(contig["contig_id"]): str(contig["sequence"]) for contig in contigs}
    negatives = [str(item["contig_id"]) for item in items if item["label"] == 0]
    seen: dict[str, set[str]] = {}
    for arm, fold in sorted(folds.items()):
        index = hs.sorted_unique_kmers(
            [context_by_id[record] for record in sorted(fold) if record in context_by_id]
        )
        seen[arm] = {
            contig_id
            for contig_id in negatives
            if hs.kmer_hits(hs.canonical_kmers(by_id[contig_id]), index) > 0
        }
    return seen


def hardneg_negatives(
    pools: Sequence[tuple[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    *,
    resolved_order: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The pool items → one-window native contigs + benchmark items, in contig-id order.

    ``pools`` is ``[(mode, {"graded": [...], "beside": [...]})]`` — the draw, and on a census the
    census beside it (the census selects only what the draw left, so the census benchmark is
    both). Each negative blocks on its ASSEMBLY (A17 Pin 3: adjacent leaders share DNA), is
    graded iff it is not D4-context, and carries its pool metadata. A window minted twice — within
    a list, across the two lists, or across the draw and the census — is refused.
    """
    contigs: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mode, lists in pools:
        for role in HARDNEG_ROLES:
            for record in lists[role]:
                contig_id = f"hn:{record['item_id']}"
                if contig_id in seen:
                    raise SystemExit(f"{record['item_id']} is minted twice; a window is one item")
                seen.add(contig_id)
                beside = role == "beside"
                meta = {key: record[key] for key in HARDNEG_ITEM_KEYS if key != "sequence"}
                order = resolved_order.get(str(record["assembly"]))
                meta.update(
                    {"pool_mode": mode, "resolved_order": None if is_missing(order) else str(order)}
                )
                contigs.append(
                    {
                        "contig_id": contig_id,
                        "sequence": record["sequence"],
                        "truth_start": None,
                        "truth_end": None,
                        "truth_strand": None,
                        "truth_source_record_id": None,
                        "truth_transform": "native_leader_window",
                    }
                )
                items.append(
                    {
                        "contig_id": contig_id,
                        "label": 0,
                        "source": "native_window",
                        "pool": str(record["sub_class"]),
                        "block": f"assembly:{record['assembly']}",
                        "length": len(record["sequence"]),
                        "locus_length": None,
                        "resolved_order": meta["resolved_order"],
                        "host_id": None,
                        "host_seen_by_any_arm": None,  # the measurement, set by the caller
                        "gated": not beside,
                        "beside": beside,
                        "hardneg": meta,
                    }
                )
    order = sorted(range(len(items)), key=lambda index: items[index]["contig_id"])
    return [contigs[i] for i in order], [items[i] for i in order]


def hardneg_positives(
    v2: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """v2's graded positives, contigs byte-identical, and a digest over their sequences.

    A17 Pin 3's positives are v2's 1,201 graded positive windows. The contigs are copied, then
    each copy's sequence is hashed against the v2 contig's own, so the identity claim is checked
    on the bytes written rather than on the copy operation. Each positive item gains
    ``beside: False`` (every positive is graded) and nothing else.
    """
    if v2.get("schema_version") != SCHEMA_VERSION or "n_items_gated" not in v2.get("scope", {}):
        raise SystemExit(
            f"--hardneg-positives is schema {v2.get('schema_version')!r}; A17's positives are the "
            "graded positives of the v2 (schema 1.1, gated-subset) benchmark"
        )
    contig_by_id = {str(c["contig_id"]): c for c in v2["contigs"]}
    items = [dict(item) for item in v2["items"] if item["label"] == 1]
    if any(item.get("gated") is not True for item in items):
        raise SystemExit("a v2 positive is not in v2's gated subset")
    if len(items) != v2["scope"]["n_positives"]:
        raise SystemExit(f"{len(items)} positives, v2 declares {v2['scope']['n_positives']}")
    contigs = []
    lines = []
    for item in items:
        source = contig_by_id[str(item["contig_id"])]
        contig = json.loads(json.dumps(source))
        if _sha256_text(contig["sequence"]) != _sha256_text(source["sequence"]):
            raise SystemExit(f"positive {item['contig_id']} is not byte-identical to v2's")
        item["beside"] = False
        contigs.append(contig)
        lines.append(f"{item['contig_id']}\t{_sha256_text(contig['sequence'])}\n")
    return contigs, items, _sha256_text("".join(sorted(lines)))


def build_hardneg(
    *,
    v2: Mapping[str, Any],
    pools: Sequence[tuple[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    resolved_order: Mapping[str, Any],
    folds: Mapping[str, Collection[str]],
    context_by_id: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    """The A17 contig/item lists and scope, in exactly v2's contig and item schema.

    Returns ``(contigs, items, scope, positives_sha256)``. ``integration.two_stage`` reads only
    ``contig_id`` / ``sequence`` / ``truth_*`` from a contig, so its legs consume this manifest
    unchanged; ``precision fold`` reads the items' ``label`` / ``pool`` / ``block`` /
    ``seen_by`` / ``gated``, plus A17's ``beside`` and ``hardneg``.
    """
    pos_contigs, pos_items, positives_sha = hardneg_positives(v2)
    neg_contigs, neg_items = hardneg_negatives(pools, resolved_order=resolved_order)
    clash = {c["contig_id"] for c in pos_contigs} & {c["contig_id"] for c in neg_contigs}
    if clash:
        raise SystemExit(f"contig ids collide between positives and negatives: {sorted(clash)[:3]}")
    contigs, items = pos_contigs + neg_contigs, pos_items + neg_items
    seen = native_fold_overlap(contigs, items, folds, context_by_id)
    for item in neg_items:
        item["seen_by"] = {arm: item["contig_id"] in ids for arm, ids in sorted(seen.items())}
        item["host_seen_by_any_arm"] = any(item["seen_by"].values())
    seen_any = set().union(*seen.values()) if seen else set()
    graded = [item for item in neg_items if item["gated"]]
    beside = [item for item in neg_items if not item["gated"]]

    def by_pool(group: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            pool: sum(1 for item in group if item["pool"] == pool)
            for pool in sorted({item["pool"] for item in group})
        }

    scope = {
        "kind": "hardneg",
        "rung": v2["scope"]["rung"],
        "geometry": v2["scope"]["geometry"],
        "positive_population": (
            "benchmark_v2.json's graded positives (gate4_eval), contigs byte-identical "
            "(ADR-0005 A17 Pin 3)"
        ),
        "negative_construction": (
            "ADR-0005 A17 Pins 1–2: native 1,024-nt windows from reports/p3/hardneg_pool.json's "
            "graded and beside lists, the bytes the pool screened (sha256 asserted per item)"
        ),
        "n_positives": len(pos_items),
        "n_negatives": len(neg_items),
        "n_items": len(items),
        "n_blocks": len({item["block"] for item in items}),
        "n_positive_blocks": len({item["block"] for item in pos_items}),
        "n_negative_blocks": len({item["block"] for item in neg_items}),
        "negatives_by_pool": by_pool(neg_items),
        "n_beside": len(beside),
        "beside_by_pool": by_pool(beside),
        "training_fold_sizes": {arm: len(fold) for arm, fold in sorted(folds.items())},
        "seen_by_counts": {
            arm: {
                "positives": sum(1 for i in pos_items if i["seen_by"][arm]),
                "negatives": sum(1 for i in neg_items if i["seen_by"][arm]),
            }
            for arm in sorted(folds)
        },
        "seen_by_rule": HARDNEG_SEEN_BY_RULE,
        "host_pool": {
            "source": [mode for mode, _ in pools],
            "stage": "census" if len(pools) > 1 else "draw",
            "selection": HARDNEG_SELECTION,
            "n_hosts": len({item["hardneg"]["assembly"] for item in neg_items}),
            "overlap_with_training_folds": {
                "k": 32,
                "method": HARDNEG_OVERLAP_METHOD,
                "by_arm": {arm: len(ids) for arm, ids in sorted(seen.items())},
                "n_seen_by_any_arm": len(seen_any),
                "n_negatives": len(neg_items),
            },
        },
        **gated_scope(items),
        "gated_rule": HARDNEG_GATED_RULE,
    }
    if not graded:
        raise SystemExit("no graded negative: the A17 benchmark would have no denominator")
    return contigs, items, scope, positives_sha


def hardneg_manifest(
    *,
    contigs: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    scope: dict[str, Any],
    positives_sha256: str,
    pool_records: Sequence[Mapping[str, Any]],
    v2_path: str,
    v2_sha256: str,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """The A17 manifest: v2's contig/item schema plus ``kind`` and the pool bindings.

    ``source.hardneg_pools`` lists each pool report's path, sha256, mode and item files, in
    draw-then-census order; ``precision fold`` refuses a pool report whose digest is not the one
    listed here. The v2 benchmark's digest and the digest over the positives' sequences are
    recorded beside it, so the "byte-identical positives" claim is checkable from the manifest.
    """
    ids = [str(contig["contig_id"]) for contig in contigs]
    if len(set(ids)) != len(ids) or len(contigs) != len(items):
        raise SystemExit("contig/item lists diverged or repeat an id")
    scope = {**scope, "host_pool": {**scope["host_pool"]}}
    scope["host_pool"]["source"] = [str(record["path"]) for record in pool_records]
    return {
        "schema_version": HARDNEG_SCHEMA_VERSION,
        "kind": HARDNEG_KIND,
        "step": HARDNEG_STEP,
        "generated_by": "scripts/mint_p3_16_benchmark.py --hardneg",
        "adr": "ADR-0005 A17",
        "source": {
            "hardneg_pools": [dict(record) for record in pool_records],
            "benchmark_v2": v2_path,
            "benchmark_v2_sha256": v2_sha256,
            "positives_sha256": positives_sha256,
            "n_positives": scope["n_positives"],
            **dict(inputs),
            "embedded_by_pool": None,
        },
        "scope": scope,
        "items": [dict(item) for item in items],
        "contigs": [dict(contig) for contig in contigs],
    }


def main_hardneg(args: argparse.Namespace) -> int:
    """``--hardneg``: mint ``hardneg_benchmark.json`` from the pool(s) and v2's positives."""
    out = Path(args.out)
    violation = hardneg_out_violation(out)  # before any input is read
    if violation:
        raise SystemExit(violation)
    pools_in = [("draw", args.hardneg, args.hardneg_items)]
    if args.hardneg_census:
        if not args.hardneg_census_items:
            raise SystemExit("--hardneg-census needs --hardneg-census-items GRADED BESIDE")
        pools_in.append(("census", args.hardneg_census, args.hardneg_census_items))
    elif args.hardneg_census_items:
        raise SystemExit("--hardneg-census-items is given without --hardneg-census")
    if not args.hardneg_items or len(args.hardneg_items) != 2:
        raise SystemExit("--hardneg needs --hardneg-items GRADED BESIDE")
    names = ["split_table", "context", "labels", "host_orders", "hardneg_positives", "hardneg"]
    paths = [getattr(args, name) for name in names] + [
        path for _, report, files in pools_in for path in (report, *files)
    ]
    for value in paths:
        problem = _repo_relative_problem("an input path", value)
        if problem:
            raise SystemExit(f"{problem}; use --data-root / --reports-root to point at a checkout")
    code_root = Path(__file__).resolve().parents[1]
    root = Path(args.data_root) if args.data_root else code_root
    reports_root = Path(args.reports_root) if args.reports_root else code_root

    from tbox_finder.integration import hardneg_pool  # lazy: the pool's own validator

    def validate(report: Mapping[str, Any]) -> list[str]:
        return hardneg_pool.validate_report(report, root=root)

    pools: list[tuple[str, dict[str, list[dict[str, Any]]]]] = []
    pool_records: list[dict[str, Any]] = []
    for mode, report_path, (graded_path, beside_path) in pools_in:
        raw = (reports_root / report_path).read_bytes()
        report = json.loads(raw.decode("utf-8"))
        lists: dict[str, list[dict[str, Any]]] = {}
        digests: dict[str, tuple[str, int]] = {}
        files: dict[str, dict[str, Any]] = {}
        for role, rel in (("graded", graded_path), ("beside", beside_path)):
            item_raw = (root / rel).read_bytes()
            lists[role] = read_hardneg_items(item_raw, role=role, where=rel)
            digests[role] = (hashlib.sha256(item_raw).hexdigest(), len(lists[role]))
            files[role] = {"path": rel, "sha256": digests[role][0], "n": digests[role][1]}
        problems = hardneg_pool_refusals(report, mode=mode, item_digests=digests, validate=validate)
        if problems:
            raise SystemExit(f"refusing --hardneg {report_path}: " + "; ".join(problems))
        pools.append((mode, lists))
        pool_records.append(
            {
                "path": report_path,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "mode": mode,
                "item_files": files,
            }
        )

    v2_raw = (root / args.hardneg_positives).read_bytes()
    v2 = json.loads(v2_raw.decode("utf-8"))
    orders = pd.read_parquet(
        root / args.host_orders, columns=["assembly_accession", "resolved_order"]
    )
    resolved_order = dict(
        zip(orders["assembly_accession"].astype(str), orders["resolved_order"], strict=True)
    )
    folds = {
        arm: training_fold_ids(
            arm=arm,
            split_table=str(root / args.split_table),
            context=str(root / args.context),
            labels=str(root / args.labels),
        )
        for arm in ARM_EXCLUDE_GATE4_EVAL
    }
    context = pd.read_parquet(root / args.context, columns=["record_id", "context_seq"])
    context_by_id = dict(
        zip(context["record_id"].astype(str), context["context_seq"].astype(str), strict=True)
    )
    contigs, items, scope, positives_sha = build_hardneg(
        v2=v2, pools=pools, resolved_order=resolved_order, folds=folds, context_by_id=context_by_id
    )
    payload = hardneg_manifest(
        contigs=contigs,
        items=items,
        scope=scope,
        positives_sha256=positives_sha,
        pool_records=pool_records,
        v2_path=args.hardneg_positives,
        v2_sha256=hashlib.sha256(v2_raw).hexdigest(),
        inputs={
            **{name: getattr(args, name) for name in ("split_table", "context", "labels")},
            "host_orders": args.host_orders,
            "host_orders_sha256": sha256_file(root / args.host_orders),
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {out}: {scope['n_items']} items ({scope['n_positives']} pos / "
        f"{scope['n_negatives']} neg, {scope['n_beside']} beside) over {scope['n_blocks']} "
        f"blocks; graded {scope['n_items_gated']} items ({scope['n_negatives_gated']} neg) over "
        f"{scope['n_blocks_gated']} blocks; fold overlap "
        f"{scope['host_pool']['overlap_with_training_folds']['by_arm']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None, help="repo-relative path (v1/v2 mint)")
    parser.add_argument("--decoys", default=None, help="repo-relative path (v1/v2 mint)")
    parser.add_argument("--split-table", default="data/processed/splits/split_assignments.parquet")
    parser.add_argument("--context", default="data/interim/flank_context/context_v0.parquet")
    parser.add_argument("--labels", default="data/processed/labels/labels_v0.parquet")
    parser.add_argument("--mining-pool", default="data/processed/negatives/mining_pool_v0.parquet")
    parser.add_argument(
        "--host-screen",
        default=None,
        help=(
            "repo-relative ADR-0005 A15 screen report (reports/p3/host_screen.json). The draw "
            "then runs over the windows it admitted; omitted, the mint is v1's construction"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "checkout holding the DVC-tracked inputs; defaults to this script's own repo root. "
            "A worktree has no data/ of its own, so minting from one points this at the main "
            "checkout while the recorded provenance stays repo-relative"
        ),
    )
    parser.add_argument(
        "--reports-root",
        default=None,
        help=(
            "checkout holding the committed --host-screen report and its admitted list; "
            "defaults to this script's own repo root (committed reports live with the code, "
            "not with the DVC data --data-root points at)"
        ),
    )
    hardneg = parser.add_argument_group("ADR-0005 A17 hard-negative mint (--hardneg)")
    hardneg.add_argument(
        "--hardneg",
        default=None,
        help="repo-relative draw-mode pool report (reports/p3/hardneg_pool.json, --reports-root)",
    )
    hardneg.add_argument(
        "--hardneg-items",
        nargs=2,
        metavar=("GRADED", "BESIDE"),
        default=None,
        help="the draw pool's graded and beside item lists (--data-root)",
    )
    hardneg.add_argument("--hardneg-census", default=None, help="census-mode pool report")
    hardneg.add_argument("--hardneg-census-items", nargs=2, metavar=("GRADED", "BESIDE"))
    hardneg.add_argument("--hardneg-positives", default="data/interim/p3_16/benchmark_v2.json")
    hardneg.add_argument(
        "--host-orders", default="data/processed/mining/production_host_orders_v0.parquet"
    )
    args = parser.parse_args()
    if args.hardneg is not None:
        return main_hardneg(args)
    if args.dataset is None or args.decoys is None:
        parser.error("--dataset and --decoys are required (unless --hardneg)")

    # Before any input is read: refusing after a long build is how an output guard loses its
    # point ([[two-outputs-one-path-destroys-the-first]]).
    violation = v1_out_violation(Path(args.out))
    if violation:
        raise SystemExit(violation)

    # Repo-relative BY CONTRACT: these strings land in a provenance block, and an absolute
    # path there leaks this machine's layout into a public repo (the P3-10 review finding).
    names = ["dataset", "decoys", "split_table", "context", "labels", "mining_pool"]
    if args.host_screen is not None:
        names.append("host_screen")
    for name in names:
        value = Path(getattr(args, name))
        if value.is_absolute() or ".." in value.parts:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be repo-relative (got {getattr(args, name)!r}); "
                "use --data-root to point at the checkout that holds the data."
            )
    code_root = Path(__file__).resolve().parents[1]
    root = Path(args.data_root) if args.data_root else code_root
    reports_root = Path(args.reports_root) if args.reports_root else code_root

    dataset = pd.read_parquet(root / args.dataset)
    decoys = pd.read_parquet(root / args.decoys)
    folds = {
        arm: training_fold_ids(
            arm=arm,
            split_table=str(root / args.split_table),
            context=str(root / args.context),
            labels=str(root / args.labels),
        )
        for arm in ARM_EXCLUDE_GATE4_EVAL
    }
    embedded = check_embedding_derivation(decoys, folds)

    gate4_records, gate4_meta = wd.load_gate4_eval_records(
        context_parquet=str(root / args.context),
        labels_parquet=str(root / args.labels),
        split_table=str(root / args.split_table),
    )
    rung = dataset[dataset["fold_random"] == RUNG]
    n_nested_train_rows = int(
        ((rung["source"] == "corpus") & rung["nested_train"].astype("boolean").fillna(False)).sum()
    )
    wanted = set(rung.loc[rung["source"] == "decoy", "row_id"].astype(str))
    decoy_rows = select_decoy_rows(decoys, wanted)

    # Hosts: mined genomic windows that cleared the shipped P2-10d′-a admission, then
    # restricted to `parent_nested_train == False`. That constrains each window's PARENT, not
    # its DNA (see host_fold_overlap); under --host-screen the A15 screen then admits windows
    # on their own DNA. The loader's own flag only offers "require in-fold"/"no requirement",
    # so the out-of-fold selection is made here and counted, not hidden.
    mining_pool_sha256 = sha256_file(root / args.mining_pool)
    admitted, host_report = negatives_mod.load_admitted_pool_rows(
        str(root / args.mining_pool), require_parent_nested_train=False
    )
    hosts = [row for row in admitted if not row.get("parent_nested_train")]
    screen_record: dict[str, Any] | None = None
    admitted_ids: list[str] | None = None
    if args.host_screen is not None:
        admitted_ids, screen_record = read_host_screen(
            args.host_screen,
            reports_root=reports_root,
            mining_pool_sha256=mining_pool_sha256,
            n_candidates=len(hosts),
            # (source block, digest field, the digest of the file THIS mint was given). Only
            # the two sources whose digest the screen records as a FILE hash are bindable;
            # fold_contexts and mined_windows record counts and intervals, not a file digest.
            screened_dna_sha256={
                ("decoy_inserts", "sha256", sha256_file(root / args.decoys)),
                (
                    "stage2_population",
                    "stage2_dataset_sha256",
                    sha256_file(root / args.dataset),
                ),
            },
        )

    split = pd.read_parquet(root / args.split_table)
    all_corpus_ids = frozenset(split["record_id"].astype(str))
    parent_cluster = {
        str(record): int(cluster)
        for record, cluster in zip(split["record_id"], split["cluster_id"], strict=True)
        if not pd.isna(cluster)
    }
    positives = build_positive_windows(gate4_records)
    negatives, drawable = mint_negative_windows(
        decoy_rows,
        hosts,
        admitted_ids=admitted_ids,
        seed=args.seed,
        all_corpus_ids=all_corpus_ids,
        parent_cluster=parent_cluster,
    )
    contigs, items, scope = build(
        positives, negatives, decoys, folds, n_nested_train_rows, screened=screen_record is not None
    )
    corpus_records, _ = wd.load_corpus_records(
        context_parquet=str(root / args.context),
        labels_parquet=str(root / args.labels),
        split_table=str(root / args.split_table),
        exclude_selection_val=False,
        exclude_gate4_eval=False,
    )
    host_seen = host_fold_overlap(
        contigs, items, folds, {r.record_id: r.context_seq for r in corpus_records}
    )
    seen_any = set().union(*host_seen.values()) if host_seen else set()
    for item in items:
        if item["label"] == 0:
            item["host_seen_by_any_arm"] = item["contig_id"] in seen_any
    parent_meta = {
        str(row["record_id"]): {
            "nested_role": None if is_missing(row["nested_role"]) else str(row["nested_role"]),
            "is_designated_loo_holdout": (
                None
                if is_missing(row["is_designated_loo_holdout"])
                else bool(row["is_designated_loo_holdout"])
            ),
            "resolved_order": (
                None if is_missing(row["resolved_order"]) else str(row["resolved_order"])
            ),
        }
        for row in split[["record_id", *COMPOSITION_PARENT_FIELDS]].to_dict("records")
    }
    scope["host_pool"] = host_pool_scope(
        mining_pool=args.mining_pool,
        n_admitted=host_report["n_records"],
        n_out_of_fold=len(hosts),
        n_drawable=len(drawable),
        n_used=len(negatives[1]),
        screen=screen_record,
        composition=host_composition(
            items, {str(row["candidate_id"]): row for row in drawable}, parent_meta
        ),
        host_seen=host_seen,
        n_negatives=len(negatives[1]),
    )
    scope["insert_exposure"] = insert_exposure(
        items,
        dict(zip(decoys["decoy_id"].astype(str), decoys["sequence"], strict=True)),
        stage2_training_decoys(dataset),
    )
    if len(contigs) != len(items):
        raise SystemExit("contig/item lists diverged — they are written as one manifest")
    ids = [contig["contig_id"] for contig in contigs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate contig_id: it keys the payload hash and the Stage-1 lookup")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": "scripts/mint_p3_16_benchmark.py",
        "source": {
            "dataset": args.dataset,
            "decoys": args.decoys,
            "split_table": args.split_table,
            "context": args.context,
            "labels": args.labels,
            "mining_pool": args.mining_pool,
            "dataset_sha256": hashlib.sha256((root / args.dataset).read_bytes()).hexdigest(),
            "decoys_sha256": hashlib.sha256((root / args.decoys).read_bytes()).hexdigest(),
            "mining_pool_sha256": mining_pool_sha256,
            "host_screen": args.host_screen,
            "host_screen_sha256": None if screen_record is None else screen_record["sha256"],
            "gate4_eval_cluster_digest": gate4_meta["carve"]["cluster_digest"],
            "gate4_eval_n_records": gate4_meta["n_records"],
            "embedded_by_pool": embedded,
        },
        "scope": scope,
        "items": items,
        "contigs": contigs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {out}: {scope['n_items']} items "
        f"({scope['n_positives']} pos / {scope['n_negatives']} neg) over "
        f"{scope['n_blocks']} blocks; gated {scope['n_items_gated']} items "
        f"({scope['n_negatives_gated']} neg) over {scope['n_blocks_gated']} blocks; "
        f"host overlap {scope['host_pool']['overlap_with_training_folds']['by_arm']}; "
        f"seen_by={scope['seen_by_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
