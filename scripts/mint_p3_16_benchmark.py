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
embedded in **both**; ``dinuc_shuffled`` splits by parent fold. Per the §7 decision all four
pools stay in the gated denominator, so the per-item exposure flags are what carries that
into the report as a disclosure instead of a silent bias.

Blocks (ADR-0005 D5)
--------------------
CIs resample at the homology-cluster level. Positives block on ``cluster_id``;
``dinuc_shuffled`` decoys inherit their parent's cluster (they are permutations of a real
locus, so they are correlated with it); the parentless pools get an explicit **singleton**
block id each. A null is never a block id — a NaN cluster collapsing into one shared block
would silently make thousands of items one exchangeable unit ([[nulls-inflate-block-counts]]).

Usage (from the checkout that holds the DVC data)::

    python scripts/mint_p3_16_benchmark.py \\
        --dataset data/processed/stage2_dataset.parquet \\
        --decoys data/processed/negatives/decoys_v0.parquet \\
        --out data/interim/p3_16/benchmark_v0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tbox_finder.data import embedding as emb
from tbox_finder.data import negatives as negatives_mod
from tbox_finder.data import window_dataset as wd

#: The rung the benchmark is drawn from. Stage-2 (P3-06) admits only ``fold_random ==
#: "train"``, so every ``test`` row is out of Stage-2's training set — measured 0 overlap.
RUNG = "test"

#: What the two committed Stage-1 runs recorded embedding, per pool. The derivation below
#: must reproduce these or the script refuses; see the module docstring.
EXPECTED_EMBEDDED: dict[str, dict[str, int]] = {
    "production": {"dinuc_shuffled": 702, "structured_rna": 2999},
    "twin": {"dinuc_shuffled": 602, "structured_rna": 2999},
}

#: ``exclude_gate4_eval`` per arm — the *one* override that separates the twin from the
#: shipped scanner (`checkpoints/stage1_gate4_twin/provenance.json`: "one override, nothing
#: else").
ARM_EXCLUDE_GATE4_EVAL: dict[str, bool] = {"production": False, "twin": True}

#: All four §9.1 classes enter the BENCHMARK denominator (PRD §2.3; §7 decision 2026-08-13),
#: unlike `embedding.TRAINING_DECOY_POOLS`, which is A7's two-pool *training* mix.
BENCHMARK_DECOY_POOLS: tuple[str, ...] = (
    "gc_background",
    "structured_rna",
    "dinuc_shuffled",
    "leader_decoy",
)

SCHEMA_VERSION = "1.0"
STEP = "P3-16"


def transcribe_to_dna(rna: str) -> str:
    """RNA → DNA. Stage 1 is a DNA model; the item table stores RNA (PRD §6 handoff, run
    backwards). Uppercased first so a lower-case ``u`` cannot survive as an ``u``."""
    return rna.upper().replace("U", "T")


def training_fold_ids(*, arm: str, split_table: str, context: str, labels: str) -> set[str]:
    """The corpus record ids one Stage-1 arm trained on, from the shipped loader."""
    records, _ = wd.load_corpus_records(
        context_parquet=context,
        labels_parquet=labels,
        split_table=split_table,
        exclude_selection_val=False,
        exclude_gate4_eval=ARM_EXCLUDE_GATE4_EVAL[arm],
    )
    return {record.record_id for record in records}


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


def block_id(row: pd.Series) -> str:
    """The resampling block (ADR-0005 D5). Never derived from a null."""
    cluster = row["cluster_id"]
    if cluster is not None and not pd.isna(cluster):
        return f"cluster:{int(cluster)}"
    return f"singleton:{row['row_id']}"


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
                "host_seen_by_any_arm": False,
            }
        )
    return contigs, items


def build_negative_windows(
    decoy_rows: Sequence[Mapping[str, Any]],
    hosts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    all_corpus_ids: Collection[str],
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

    Hosts are supplied by the caller already filtered to ``parent_nested_train == False``, so
    the DNA every negative is made of is DNA **neither** Stage-1 checkpoint trained on.
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
                # One host per negative (``unique_hosts=True``), so the host IS the block.
                "block": f"host:{entry.host_id}",
                "length": len(entry.sequence),
                "locus_length": entry.insert_len,
                "resolved_order": None,
                "host_id": entry.host_id,
                "host_seen_by_any_arm": False,
                "splice_phase": entry.phase,
            }
        )
    return contigs, items, report


def build(
    positives: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    negatives: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    decoys: pd.DataFrame,
    folds: dict[str, set[str]],
    n_nested_train_rows: int,
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
        # stream. The host DNA is separately clean for both arms by construction (hosts are
        # drawn `parent_nested_train == False`), which `host_seen_by_any_arm` records so the
        # two exposures are never conflated.
        item["seen_by"] = {
            arm: decoy_embedded(decoy_row, fold) for arm, fold in sorted(folds.items())
        }

    contigs = pos_contigs + neg_contigs
    items = pos_items + neg_items
    scope = {
        "rung": RUNG,
        "geometry": f"{wd.WINDOW_NT}-nt scan windows (ADR-0005 D3 / PRD §6 pinned geometry)",
        "positive_population": "gate4_eval (window_dataset.load_gate4_eval_records)",
        "negative_construction": (
            "data/embedding.py::embed_decoy_rows — the shipped ADR-0005 A7 splice, all four "
            "§9.1 pools, unique hosts, hosts drawn parent_nested_train == False"
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="repo-relative path")
    parser.add_argument("--decoys", required=True, help="repo-relative path")
    parser.add_argument("--split-table", default="data/processed/splits/split_assignments.parquet")
    parser.add_argument("--context", default="data/interim/flank_context/context_v0.parquet")
    parser.add_argument("--labels", default="data/processed/labels/labels_v0.parquet")
    parser.add_argument("--mining-pool", default="data/processed/negatives/mining_pool_v0.parquet")
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
    args = parser.parse_args()

    # Repo-relative BY CONTRACT: these strings land in a provenance block, and an absolute
    # path there leaks this machine's layout into a public repo (the P3-10 review finding).
    for name in ("dataset", "decoys", "split_table", "context", "labels", "mining_pool"):
        value = Path(getattr(args, name))
        if value.is_absolute() or ".." in value.parts:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be repo-relative (got {getattr(args, name)!r}); "
                "use --data-root to point at the checkout that holds the data."
            )
    root = Path(args.data_root) if args.data_root else Path(__file__).resolve().parents[1]

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
    decoy_rows = [
        {key: row[key] for key in decoys.columns}
        for _, row in decoys[decoys["decoy_id"].astype(str).isin(wanted)]
        .sort_values("decoy_id", kind="mergesort")
        .iterrows()
    ]
    if len(decoy_rows) != len(wanted):
        raise SystemExit(
            f"{len(wanted)} test-rung decoys but {len(decoy_rows)} matched in decoys_v0"
        )

    # Hosts: mined genomic windows that cleared the shipped P2-10d′-a admission, then
    # restricted to `parent_nested_train == False` — DNA NEITHER Stage-1 checkpoint trained
    # on. The loader's own flag only offers "require in-fold"/"no requirement", so the
    # out-of-fold selection is made here and counted, not hidden.
    admitted, host_report = negatives_mod.load_admitted_pool_rows(
        str(root / args.mining_pool), require_parent_nested_train=False
    )
    hosts = [row for row in admitted if not row.get("parent_nested_train")]
    if len(hosts) < len(decoy_rows):
        raise SystemExit(
            f"{len(hosts)} out-of-fold host windows for {len(decoy_rows)} decoys; "
            "unique_hosts=True needs at least one each"
        )

    all_corpus_ids = frozenset(pd.read_parquet(root / args.split_table)["record_id"].astype(str))
    positives = build_positive_windows(gate4_records)
    negatives = build_negative_windows(
        decoy_rows, hosts, seed=args.seed, all_corpus_ids=all_corpus_ids
    )
    contigs, items, scope = build(positives, negatives, decoys, folds, n_nested_train_rows)
    scope["host_pool"] = {
        "source": args.mining_pool,
        "n_admitted": host_report["n_records"],
        "n_out_of_fold_available": len(hosts),
        "n_used": len(negatives[1]),
        "selection": "parent_nested_train == False (clean for BOTH Stage-1 checkpoints)",
    }
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
            "dataset_sha256": hashlib.sha256((root / args.dataset).read_bytes()).hexdigest(),
            "decoys_sha256": hashlib.sha256((root / args.decoys).read_bytes()).hexdigest(),
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
        f"{scope['n_blocks']} blocks; seen_by={scope['seen_by_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
