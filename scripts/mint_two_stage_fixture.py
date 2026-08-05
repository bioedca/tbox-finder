#!/usr/bin/env python
"""P3-14 — mint ``tests/fixtures/two_stage/contigs.json`` from the real corpus.

The golden regression needs **real genomic contigs**, not synthesised ones (CLAUDE.md §8.7): a
harness that only ever ran on hand-built oracles would lock its own arithmetic and nothing else.
Every contig here is a slab of ``data/interim/flank_context/context_v0.parquet`` — the real NCBI
region around a curated T-box record — cropped to :data:`CONTIG_NT`.

Three deliberate properties, each of which the fixture would be useless without:

**1. Test-rung records only.** Records are drawn from ``fold_random == "test"`` with
``calib == False``. The fixture is a hash lock, not a measurement, but a train-fold contig would
still invite someone to read its posteriors as performance — so the draw comes from the rung
nothing was fitted on, and the report says it is not an evaluation regardless.

**2. Both orientations, because the source is orientation-degenerate.**
``flank_context`` reverse-complements minus-strand rows on fetch, so ``context_seq`` is *always*
in the locus's own 5′→3′ sense and every locus in it is on ``+``. A fixture built straight from
that would exercise the D15 resolver against a single answer and could not tell a working
resolver from one hardcoded to ``"+"``. Each record therefore contributes **two** contigs: the
slab as fetched (locus on ``+``) and its reverse complement (locus on ``-``), which is real
sequence either way.

**3. A composition-exact null.** Some contigs are the **reverse** (not the reverse complement) of
a slab — PRD §12's reversed-sequence null, chosen because reverse-complementing is useless
against an RC-equivariant scanner. They carry the same base composition and no T-box, so the
candidate table has rows that legitimately fail the operating point and ``confirmed`` is a
discriminative column rather than a constant.

Locus phase is jittered per record so the fixture does not sit at one window offset: P3-11
measured the *call* to be phase-invariant but the reported strength not, and a fixture pinned to
one phase would lock the easy case.

Deterministic end to end — a sort, a stride and an arithmetic jitter, no RNG and no seed.

Run (LOCAL, ``tbox-finder-data``)::

    PYTHONPATH=src python scripts/mint_two_stage_fixture.py \\
        --context data/interim/flank_context/context_v0.parquet \\
        --splits data/processed/splits/split_assignments.parquet \\
        --out tests/fixtures/two_stage/contigs.json \\
        [--data-root /path/to/the/checkout/holding/data]

``--context`` / ``--splits`` are **repo-relative by contract** — they are recorded verbatim in
the committed fixture, so an absolute one would leak this machine's layout — and ``--data-root``
says which checkout to read them from (a worktree has no ``data/`` of its own).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tbox_finder.infer.handoff import reverse_complement

#: Contig length. At window 1024 / stride 512 this tiles into exactly two windows (starts 0 and
#: 512), so the doubly-covered interior is positions 512–1023 and the locus — centred — sits
#: across the seam the ADR-0005 A3 reconciliation operator exists to remove. Short enough that
#: the committed per-window logits stay a few hundred kB.
CONTIG_NT = 1536

#: Records contributing a (forward, reverse-complement) pair, and records contributing a
#: reversed null. Kept small because each contig costs 2 × 1024 × 8 committed logits.
N_LOCUS_RECORDS = 8
N_NULL_RECORDS = 4

#: How many characters of a record_id become a contig_id. Short ids keep the committed table
#: readable; :func:`build` refuses a collision rather than letting two contigs share a key,
#: which would silently overwrite one of them in the Stage-1 archive and in every lookup keyed
#: on it (CodeRabbit r1).
ID_PREFIX_NT = 12

#: Phase jitter applied to the crop, in nt, cycled over the selected records. Spreads the locus
#: across the 512-nt window period instead of pinning every fixture locus to one offset.
PHASE_JITTER = (0, 137, 261, 389, 61, 199, 331, 453)


def select_records(context: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Test-rung, order-stratified, deterministic.

    Order stratification matters more than count here: the corpus is ~90 % Firmicutes, so an
    unstratified draw of eight would very likely be eight near-siblings and the fixture would
    exercise one clade's sequence statistics ([[uniform-cluster-draw-collapses-on-skew]]).
    """
    eligible = context[
        (context["status"] == "ok") & (context["context_seq"].str.len() >= CONTIG_NT + 512)
    ]
    # `resolved_order` is nullable, and a null is NOT a stratum: admitting one would let the
    # unresolved-lineage bucket claim a slot the way a null claims a block in `len(set(col))`
    # ([[nulls-inflate-block-counts]]). Dropped before stratification, not after.
    rung = splits[
        (splits["fold_random"] == "test")
        & (~splits["calib"].astype(bool))
        & splits["resolved_order"].notna()
    ]
    merged = eligible.merge(
        rung[["record_id", "resolved_order", "cluster_id"]], on="record_id", how="inner"
    )
    merged = merged.sort_values("record_id", kind="mergesort")
    picked: list[int] = []
    seen_orders: set[str] = set()
    for position, row in enumerate(merged.itertuples()):
        order = row.resolved_order
        if order in seen_orders:
            continue
        seen_orders.add(order)
        picked.append(position)
        if len(picked) >= N_LOCUS_RECORDS + N_NULL_RECORDS:
            break
    if len(picked) < N_LOCUS_RECORDS + N_NULL_RECORDS:
        raise SystemExit(
            f"only {len(picked)} distinct orders are eligible; the fixture needs "
            f"{N_LOCUS_RECORDS + N_NULL_RECORDS}"
        )
    return merged.iloc[picked].reset_index(drop=True)


def crop(row: pd.Series, jitter: int) -> tuple[str, int, int]:
    """Crop ``context_seq`` to :data:`CONTIG_NT` with the locus centred, then jittered."""
    sequence = row["context_seq"]
    offset, length = int(row["locus_offset"]), int(row["locus_length"])
    start = offset - (CONTIG_NT - length) // 2 + jitter
    start = max(0, min(start, len(sequence) - CONTIG_NT))
    contig = sequence[start : start + CONTIG_NT]
    locus_start = offset - start
    if not (locus_start >= 0 and locus_start + length <= CONTIG_NT):
        raise SystemExit(
            f"record {row['record_id']}: locus [{locus_start}, {locus_start + length}) falls "
            f"outside the {CONTIG_NT} nt crop; widen CONTIG_NT or drop the record"
        )
    return contig, locus_start, locus_start + length


def build(selected: pd.DataFrame) -> list[dict[str, object]]:
    contigs: list[dict[str, object]] = []
    for index, (_, row) in enumerate(selected.iterrows()):
        jitter = PHASE_JITTER[index % len(PHASE_JITTER)]
        sequence, start, end = crop(row, jitter)
        record_id = str(row["record_id"])
        if index < N_LOCUS_RECORDS:
            contigs.append(
                {
                    "contig_id": f"{record_id[:ID_PREFIX_NT]}_fwd",
                    "sequence": sequence,
                    "truth_strand": "+",
                    "truth_start": start,
                    "truth_end": end,
                    "truth_source_record_id": record_id,
                    "truth_transform": "forward",
                }
            )
            contigs.append(
                {
                    # The reverse complement of a real slab is a real slab read the other way:
                    # the locus is unchanged, its coordinates mirror, and its strand flips.
                    "contig_id": f"{record_id[:ID_PREFIX_NT]}_rc",
                    "sequence": reverse_complement(sequence),
                    "truth_strand": "-",
                    "truth_start": CONTIG_NT - end,
                    "truth_end": CONTIG_NT - start,
                    "truth_source_record_id": record_id,
                    "truth_transform": "reverse_complement",
                }
            )
        else:
            contigs.append(
                {
                    # PRD §12's composition-exact null. Reverse, NOT reverse-complement: an RC
                    # is useless against an RC-equivariant scanner, which would score it as the
                    # positive it is.
                    "contig_id": f"{record_id[:ID_PREFIX_NT]}_rev",
                    "sequence": sequence[::-1],
                    "truth_strand": None,
                    "truth_start": None,
                    "truth_end": None,
                    "truth_source_record_id": record_id,
                    "truth_transform": "reverse",
                }
            )
    seen: set[str] = set()
    for contig in contigs:
        contig_id = contig["contig_id"]
        if contig_id in seen:
            raise SystemExit(
                f"contig_id {contig_id!r} is not unique — two record_ids share their first "
                f"{ID_PREFIX_NT} characters. Every downstream lookup is keyed on it, so a "
                "collision would silently drop one contig; widen ID_PREFIX_NT"
            )
        seen.add(contig_id)
    return contigs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context", required=True, help="repo-relative path")
    parser.add_argument("--splits", required=True, help="repo-relative path")
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
    args = parser.parse_args()

    # --context/--splits are repo-relative BY CONTRACT, so what lands in the committed fixture's
    # provenance is portable by construction rather than by a rewrite that could silently
    # mis-resolve. --data-root says which checkout to read them from.
    for name in ("context", "splits"):
        candidate = Path(getattr(args, name))
        # `..` is refused alongside an absolute path: `../other-checkout/data/x.parquet` clears
        # an is_absolute() check, resolves outside the root, and is then recorded verbatim —
        # which is the exact property this guard exists to hold (CodeRabbit r2).
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SystemExit(
                f"--{name} must be repo-relative (got {getattr(args, name)!r}); it is recorded "
                "verbatim in a committed fixture and an absolute path leaks this machine's "
                "layout. Use --data-root to point at the checkout that holds the data."
            )
    root = Path(args.data_root) if args.data_root else Path(__file__).resolve().parents[1]
    selected = select_records(
        pd.read_parquet(root / args.context), pd.read_parquet(root / args.splits)
    )
    contigs = build(selected)
    payload = {
        "schema_version": "1.0",
        "step": "P3-14",
        "generated_by": "scripts/mint_two_stage_fixture.py",
        "source": {
            # Recorded exactly as given, and given repo-relative by contract (see --data-root):
            # a committed fixture that names this machine's home directory leaks a username and
            # resolves for nobody else (CodeRabbit r1).
            "context": args.context,
            "splits": args.splits,
            "rung": "fold_random == 'test' and not calib",
            "contig_nt": CONTIG_NT,
            "n_locus_records": N_LOCUS_RECORDS,
            "n_null_records": N_NULL_RECORDS,
            "orders": sorted(str(order) for order in selected["resolved_order"]),
            "record_ids": [str(value) for value in selected["record_id"]],
        },
        "contigs": contigs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(contigs)} contigs over {len(selected)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
