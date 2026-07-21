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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--decoys", default=DEFAULT_DECOYS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--window", type=int, default=1024)
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
    # Size the spliced arms to the decoy pool: `unique_hosts=True` gives every embedded
    # window its own host, because two windows sharing a host are near-duplicates that a
    # cross-validated probe recognises across folds.
    n_embed = sum(
        1
        for r in decoy_rows
        if str(r.get("pool") or "") in TRAINING_DECOY_POOLS and not bool(r.get("masked"))
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
        decoy_rows, embed_hosts, seed=args.seed, window=args.window, unique_hosts=True
    )
    # The paired control -- same hosts, same phases -- is what `arms_are_matched` verifies.
    matched_control = junction_control(embedded, embed_hosts, seed=args.seed, window=args.window)
    # The AUROC control is the same construction on DISJOINT hosts, because a classifier
    # fed a host window in both classes learns the host and inverts the comparison. Same
    # decoys => `plan_placement` draws the same phases, so the two control sets share a
    # phase and insert-length distribution and differ only in which windows they landed on.
    embedded_b, _ = embed_decoy_rows(
        decoy_rows, control_hosts, seed=args.seed, window=args.window, unique_hosts=True
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
        per_k[str(k)] = {
            "measurement": measurement,
            "clauses": junction_clauses(measurement),
        }
        print(f"k={k}: {json.dumps(per_k[str(k)]['clauses'])}")

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
        # The gate passes only if EVERY k passes: a cue invisible at k=4 but readable at
        # k=6 is still a cue, and reporting the friendlier k would be exactly the
        # "report only the flattering unit" failure ADR-0005 A5 names.
        "overall_pass": all(all(v["clauses"].values()) for v in per_k.values()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}  overall_pass={report['overall_pass']}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
