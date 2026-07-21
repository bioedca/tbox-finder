"""Full-scale pre-flight junction measurement on the real substrate (P2-10d′-b, A7 pin 7).

Tier 1 of ADR-0005 A7's two-tier gate, run LOCAL on the actual pools rather than on the
CI fixture. Blocks the SLURM submit-ack: if the background→background junction is
separable from plain unspliced windows, the R2 embedding hands the model a
class-correlated boundary cue and no amount of GPU time fixes it.

Writes `reports/p2/junction_control.json`. Every arm uses **disjoint host windows** — see
`eval/junction_probe.cv_auroc` for why sharing them inverts the measurement.

    PYTHONPATH=src python scripts/measure_junction_control.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tbox_finder.data.embedding import (
    EXCLUDED_DECOY_POOLS,
    TRAINING_DECOY_POOLS,
    embed_decoy_rows,
    junction_control,
    load_decoy_rows,
)
from tbox_finder.data.negatives import load_admitted_pool_rows
from tbox_finder.eval.junction_probe import junction_clauses, junction_measurement

DEFAULT_POOL = "data/processed/negatives/mining_pool_v0.parquet"
DEFAULT_DECOYS = "data/processed/negatives/decoys_v0.parquet"
DEFAULT_OUT = "reports/p2/junction_control.json"
DEFAULT_SEED = 20260721


def _training_fold_ids(splits_path: str) -> set[str]:
    """Record ids of the §9.2 nested training fold — the decoy-parent admission set."""
    import pandas as pd

    frame = pd.read_parquet(splits_path)
    missing = [c for c in ("record_id", "nested_train") if c not in frame.columns]
    if missing:
        raise SystemExit(f"split table {splits_path} is missing column(s) {missing}")
    return {str(r) for r in frame.loc[frame["nested_train"].astype(bool), "record_id"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--decoys", default=DEFAULT_DECOYS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument(
        "--splits",
        default="data/processed/splits/split_assignments.parquet",
        help="split table, for the decoy's OWN §9.2 parent-fold admission rule",
    )
    ap.add_argument("--k", type=int, nargs="+", default=[4, 6])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    host_rows, pool_report = load_admitted_pool_rows(args.pool, window=args.window)
    # Deduplicate by SEQUENCE, not by candidate_id. The pool carries byte-identical
    # windows under distinct ids (a lead and a trail window of two nearby loci can cover
    # the same DNA), and a duplicate landing in two arms leaks across the cross-validation
    # folds exactly like a shared host would. Measured on the shipped pool, dropping them
    # costs a handful of windows and the count is reported rather than absorbed.
    seen: set[str] = set()
    unique_rows = []
    for row in host_rows:
        seq = str(row["sequence"])
        if seq not in seen:
            seen.add(seq)
            unique_rows.append(row)
    n_duplicate_windows = len(host_rows) - len(unique_rows)
    host_rows = unique_rows

    decoy_rows = load_decoy_rows(args.decoys)
    fold_ids = _training_fold_ids(args.splits)
    print(f"training-fold record ids: {len(fold_ids)}")
    # Size the spliced arms to the decoy pool: `unique_hosts=True` gives every embedded
    # window its own host, because two windows sharing a host are near-duplicates that a
    # cross-validated probe recognises across folds.
    n_embed = sum(
        1
        for r in decoy_rows
        if str(r.get("pool") or "") in TRAINING_DECOY_POOLS
        and not bool(r.get("masked"))
        and (
            not str(r.get("source_record_id") or "").strip()
            or str(r.get("source_record_id")).strip() in fold_ids
        )
    )
    n_plain = (len(host_rows) - 2 * n_embed) // 2
    if n_plain < 2:
        raise SystemExit(
            f"pool admits {len(host_rows)} windows; two spliced arms of {n_embed} leave "
            f"{n_plain} for each unspliced arm. Cap the decoy pool or widen the substrate."
        )
    picks = rng.choice(len(host_rows), size=2 * n_embed + 2 * n_plain, replace=False)
    embed_hosts = [host_rows[i] for i in picks[:n_embed]]
    control_hosts = [host_rows[i] for i in picks[n_embed : 2 * n_embed]]
    plain = [str(host_rows[i]["sequence"]) for i in picks[2 * n_embed : 2 * n_embed + n_plain]]
    null_reference = [str(host_rows[i]["sequence"]) for i in picks[2 * n_embed + n_plain :]]

    embedded, embed_report = embed_decoy_rows(
        decoy_rows,
        embed_hosts,
        seed=args.seed,
        window=args.window,
        unique_hosts=True,
        training_fold_record_ids=fold_ids,
    )
    # The paired control -- same hosts, same phases -- is what `arms_are_matched` verifies.
    matched_control = junction_control(embedded, embed_hosts, seed=args.seed, window=args.window)
    # The AUROC control is the same construction on DISJOINT hosts, because a classifier
    # fed a host window in both classes learns the host and inverts the comparison. Same
    # decoys => `plan_placement` draws the same phases, so the two control sets share a
    # phase and insert-length distribution and differ only in which windows they landed on.
    embedded_b, _ = embed_decoy_rows(
        decoy_rows,
        control_hosts,
        seed=args.seed,
        window=args.window,
        unique_hosts=True,
        training_fold_record_ids=fold_ids,
    )
    control = junction_control(embedded_b, control_hosts, seed=args.seed, window=args.window)
    host_sequences = {str(h["candidate_id"]): str(h["sequence"]) for h in embed_hosts}

    per_k = {}
    for k in args.k:
        measurement = junction_measurement(
            plain=plain,
            null_reference=null_reference,
            control=control,
            matched_control=matched_control,
            decoy=embedded,
            host_sequences=host_sequences,
            k=k,
            seed=args.seed,
        )
        clauses = junction_clauses(measurement)
        per_k[str(k)] = {
            "measurement": measurement,
            "clauses": clauses,
            # Retained verbatim and NOT gated on. See `gated_clauses` below for why, and
            # read this value rather than the pass flag when asking whether the junction
            # is visible: at k=6 it is.
            "junction_within_null_band_DIAGNOSTIC": clauses["junction_within_null_band"],
        }
        print(f"k={k}: {json.dumps(clauses)}")

    report = {
        "schema_version": "1.0",
        "step": "P2-10d'-b",
        "adr": "ADR-0005 A7 pin 7 (tier 1: pre-flight composition gate)",
        "seed": args.seed,
        "window_nt": args.window,
        "n_embedded_arm": n_embed,
        "n_unspliced_arm": n_plain,
        "n_admitted_pool_windows": len(host_rows),
        "n_duplicate_windows_dropped": n_duplicate_windows,
        "embedded_pools": list(TRAINING_DECOY_POOLS),
        "excluded_pools": dict(sorted(EXCLUDED_DECOY_POOLS.items())),
        "negative_pool_report": pool_report,
        "embedding_report": embed_report,
        "by_k": per_k,
        # ── What this artifact does and does not gate (ADR-0005 A7 pin 7, amended) ───
        # `junction_within_null_band` is measured, recorded, and DELIBERATELY NOT gated.
        # It fails at k=6, and that failure is the finding, not an inconvenience: across
        # four independent draws the junction arm read 0.5217 / 0.5264 / 0.5281 / 0.5328
        # against nulls of 0.4947-0.5042. A splice junction is a chimera of two genomes
        # and a 6-mer model can see it. Because the original R2 construction spliced only
        # negatives, that made "chimeric" a class-correlated cue.
        #
        # The response is NOT a widened band. The cue is removed by construction: the
        # amended pin splices POSITIVES at the negatives' own chimeric rate, so chimerism
        # carries no class information, and the gate that replaces this one is an exact
        # arithmetic identity over the emitted stream
        # (`eval/junction_probe.junction_symmetry_clauses`, enforced in the training
        # report and in tests/ml/test_junction_symmetry_stream.py). A construction
        # guarantee is strictly stronger than the underpowered AUROC it replaces -- but
        # the number stays here so no reader has to take that on trust.
        "gated_clauses": ["arms_are_matched", "probe_can_discriminate"],
        "ungated_diagnostic_clauses": ["junction_within_null_band"],
        "junction_is_visible_at_k6": True,
        "overall_pass": all(
            v["clauses"]["arms_are_matched"] and v["clauses"]["probe_can_discriminate"]
            for v in per_k.values()
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}  overall_pass={report['overall_pass']}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
