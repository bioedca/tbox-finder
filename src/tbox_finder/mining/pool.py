"""The coordinate-bearing mineable negative substrate (P2-10b).

PRD §9.1 masks all known T-box loci "+ a flank" from every negative pool and
requires the **residual contamination against the union denominator** to be
reported; ADR-0005 D14 makes that mask the first of the two guards standing
between a Stage-1 false positive and the mined hard-negative pool. Neither can
do any work on a pool whose records have no genomic coordinates —
:func:`~tbox_finder.mining.hard_negative.classify_candidate` refuses such a
candidate outright (``refused_no_coordinates``), which at P0 was **100 %** of
every mineable pool.

Of the four §9.1 pools, only one can be given real coordinates at P2:

===================  ==========================================================
``gc_background``    i.i.d. emitted at a matched GC — exists in no genome.
``dinuc_shuffled``   a permutation of a positive; its only coordinates are that
                     positive's, so carrying them masks 100 % of the pool
                     against its own parents (ADR-0005 A6 removes it from
                     ``MINEABLE_POOLS``; the parent link is recorded instead).
``leader_decoy``     opaque surrogate ids, no recoverable accession; re-sourcing
                     needs CDS/tRNA annotation the repo does not have.
``structured_rna``   coordinates recoverable from the Rfam headers — done in
                     :func:`tbox_finder.decoys.parse_structured_rna_locus`. But
                     the pool contains **0 true overlaps** with the union prior
                     (measured), so it cannot by itself demonstrate that the
                     mask fires.
===================  ==========================================================

This module supplies what none of them can: real genomic windows, carved from
the P2-00 ``flank_context`` regions, on the replicons that actually host T-boxes,
each carrying an exact ``(accession, locus_start, locus_end, strand)``. Because
those replicons host known loci, the mask has something to find — and because a
locus-centred window is *known* to overlap its own locus, the pool comes with a
**designed positive control that must mask at 100 %**. That control, not the
natural rate, is what proves the mask is live: a natural rate of zero is a
legitimate scientific outcome, whereas a control below 100 % is a broken frame.

This is **not** a fifth PRD §9.1 negative class. §9.1's four classes are
training/benchmark negatives feeding the ~10:1 seed mix; this is the P2 mining
substrate that ADR-0005 D14's loop consumes, and it is written to its own
artifact so the §9.1 pools' sizes, ratios, and golden digest are untouched.
Windows are annotation-blind — they are *not* verified 5′UTRs, and nothing here
may be described as a leader decoy (that is P2-10b′).

The pure carving/geometry helpers are stdlib-only so the unit tier runs in bare
CI; pandas is imported lazily in :func:`build`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder import masking, provenance
from tbox_finder.data.flank_context import forward_bounds

#: Pool name carried in the artifact's ``pool`` column and in ``MINEABLE_POOLS``.
POOL_GENOMIC_WINDOW = "genomic_window"

#: Output paths (mirrors the ``decoys`` module's layout).
_NEG_DIR = "data/processed/negatives"
_AUDIT_DIR = "data/processed/audits"
MINING_POOL_PARQUET = f"{_NEG_DIR}/mining_pool_v0.parquet"
MINING_POOL_PROVENANCE = f"{_NEG_DIR}/mining_pool_v0.provenance.json"
MINING_POOL_REPORT = f"{_AUDIT_DIR}/mining_pool_report.json"

#: Default window length (nt). Sits inside the corpus locus-length range
#: (104-550 nt, median 281) so a mined window is a plausible candidate rather
#: than a length outlier the scanner could separate on size alone.
DEFAULT_WINDOW_NT = 300

#: A flank must exceed the window by this margin before a window is carved from
#: it, so the window never abuts the locus it was carved beside.
DEFAULT_FLANK_MARGIN_NT = 50

#: Only ``status == "ok"`` context rows carry real geometry (flank_context pins
#: this: a non-anchored row's offsets are sentinels, not coordinates).
CONTEXT_STATUS_OK = "ok"

#: Which side of the locus a window was carved from.
SIDE_LEAD = "lead"
SIDE_TRAIL = "trail"

#: The designed positive control: a window centred on the record's own locus.
#: It overlaps a known T-box by construction, so it MUST mask.
SIDE_LOCUS_CONTROL = "locus_control"


class MiningPoolError(ValueError):
    """Raised when the substrate cannot be built or a designed control fails."""


def carve_window(
    row: Mapping[str, Any], side: str, *, window_nt: int, margin_nt: int
) -> dict[str, Any] | None:
    """Carve one coordinate-bearing window out of a ``flank_context`` row.

    ``side`` is :data:`SIDE_LEAD` / :data:`SIDE_TRAIL` (a flank window, taken at
    the outer edge so it is as far from the locus as the region allows) or
    :data:`SIDE_LOCUS_CONTROL` (the locus itself — the designed control).

    Returns ``None`` — never a truncated or invented window — when the row is not
    anchored or the requested side has less than ``window_nt + margin_nt`` of
    flank. ``margin_nt`` does not apply to the control, which is the locus.
    """
    if str(row.get("status")) != CONTEXT_STATUS_OK:
        return None
    seq = str(row.get("context_seq") or "")
    region_len = len(seq)
    locus_offset = int(row["locus_offset"])
    locus_length = int(row["locus_length"])
    if region_len <= 0 or locus_offset < 0:
        return None

    if side == SIDE_LOCUS_CONTROL:
        offset, length = locus_offset, locus_length
    elif side == SIDE_LEAD:
        if int(row["lead_flank"]) < window_nt + margin_nt:
            return None
        offset, length = 0, window_nt
    elif side == SIDE_TRAIL:
        if int(row["trail_flank"]) < window_nt + margin_nt:
            return None
        offset, length = region_len - window_nt, window_nt
    else:
        raise MiningPoolError(f"unknown side {side!r}")

    if offset < 0 or length < 1 or offset + length > region_len:
        return None
    lo, hi = forward_bounds(
        strand=int(row["strand"]),
        region_start=int(row["region_start"]),
        region_len=region_len,
        offset=offset,
        length=length,
    )
    return {
        "pool": POOL_GENOMIC_WINDOW,
        "candidate_id": f"{row['record_id']}:{side}",
        "side": side,
        "sequence": seq[offset : offset + length],
        "length": length,
        "accession": str(row["accession"]),
        "locus_start": lo,
        "locus_end": hi,
        "strand": int(row["strand"]),
        "source_record_id": str(row["record_id"]),
        "is_designed_control": side == SIDE_LOCUS_CONTROL,
    }


def carve_pool(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    window_nt: int = DEFAULT_WINDOW_NT,
    margin_nt: int = DEFAULT_FLANK_MARGIN_NT,
    n_controls: int = 0,
    sides: Sequence[str] = (SIDE_LEAD, SIDE_TRAIL),
) -> list[dict[str, Any]]:
    """Carve the substrate: flank windows from every eligible row, plus controls.

    ``n_controls`` locus-centred control windows are drawn from the anchored rows
    with a seeded RNG (0 disables them). They are tagged ``is_designed_control``
    so no report can fold them into the natural contamination rate — a control
    that is *built* to overlap a known locus would otherwise inflate exactly the
    number it exists to validate.
    """
    materialised = list(rows)
    records: list[dict[str, Any]] = []
    for row in materialised:
        for side in sides:
            window = carve_window(row, side, window_nt=window_nt, margin_nt=margin_nt)
            if window is not None:
                records.append(window)
    if n_controls > 0:
        anchored = [r for r in materialised if str(r.get("status")) == CONTEXT_STATUS_OK]
        rng = random.Random(f"{seed}:{POOL_GENOMIC_WINDOW}:controls")
        picked = rng.sample(anchored, min(int(n_controls), len(anchored)))
        for row in picked:
            control = carve_window(
                row, SIDE_LOCUS_CONTROL, window_nt=window_nt, margin_nt=margin_nt
            )
            if control is not None:
                records.append(control)
    return records


def mask_pool(
    records: Sequence[Mapping[str, Any]], index: masking.LocusIndex, *, flank: int
) -> dict[str, Any]:
    """Mask the substrate and split the numbers that must not be conflated.

    Three separations carry the P2-10b gate, each guarding a way the measurement
    could look green while measuring nothing:

    * **designed vs natural** — the control's mask rate proves the mask is live;
      the natural rate is the science. Pooling them lets a 100 % control hide a
      0 % natural rate, or a designed overlap masquerade as contamination.
    * **overlap vs proximity** — ``n_masked`` at a flank counts records *near* a
      known locus. ``n_overlapping`` (flank 0) is the subset that genuinely
      intersects one.
    * **namespace compatibility** — whether the pool's accessions can address the
      index at all. Without it, "0 masked" is unreadable.
    """
    if flank <= 0:
        raise MiningPoolError(
            f"flank must be positive (PRD §9.1 masks loci + a flank); got {flank}"
        )
    out: dict[str, Any] = {}
    for group, subset in (
        ("designed_control", [r for r in records if r["is_designed_control"]]),
        ("natural", [r for r in records if not r["is_designed_control"]]),
    ):
        n_masked = sum(
            1
            for r in subset
            if index.is_masked(r["accession"], r["locus_start"], r["locus_end"], flank=flank)
        )
        n_overlap = sum(
            1
            for r in subset
            if index.is_masked(r["accession"], r["locus_start"], r["locus_end"], flank=0)
        )
        out[group] = {
            "n_records": len(subset),
            "n_masked_at_flank": n_masked,
            "n_overlapping_at_flank_0": n_overlap,
            "masked_fraction": (n_masked / len(subset)) if subset else 0.0,
        }
    out["accession_namespace"] = masking.accession_namespace_report(
        (r["accession"] for r in records), index
    )
    return out


def control_gate(mask_summary: Mapping[str, Any]) -> dict[str, Any]:
    """The P2-10b non-vacuity gate, re-derived from the recorded evidence.

    Every clause is wrapped so that a *missing* measurement is FALSE, not
    vacuously TRUE: a gate assembled from the requested configuration rather than
    the found evidence passes exactly when the evidence is absent. The control
    clause demands **all** control windows mask, because each one overlaps a known
    T-box by construction — anything below 100 % is a coordinate-frame defect, not
    a biological result (an earlier minus-strand frame scored 23,147/23,535 and
    the corrected one 23,532/23,532, which is how the defect was found).
    """
    control = mask_summary.get("designed_control") or {}
    natural = mask_summary.get("natural") or {}
    namespace = mask_summary.get("accession_namespace") or {}
    n_control = int(control.get("n_records", 0))
    n_control_masked = int(control.get("n_masked_at_flank", 0))
    # Gate on the flank-0 **overlap** count, not the flanked mask count. A control
    # window IS the locus, so it must intersect its own known interval outright;
    # allowing the ±flank slack would let a frame error of up to ``flank`` nt pass
    # with a perfect-looking 100%. The stricter number is computed anyway, so
    # gating on the weaker one would have meant writing the evidence of the
    # failure into the report and then not reading it.
    n_control_overlap = int(control.get("n_overlapping_at_flank_0", 0))
    clauses = {
        "controls_present": bool(control) and n_control > 0,
        "all_controls_overlap": bool(control) and n_control > 0 and n_control_overlap == n_control,
        "all_controls_masked": bool(control) and n_control > 0 and n_control_masked == n_control,
        "natural_pool_present": bool(natural) and int(natural.get("n_records", 0)) > 0,
        "namespace_compatible": bool(namespace) and bool(namespace.get("namespace_compatible")),
    }
    return {
        "clauses": clauses,
        "n_control_windows": n_control,
        "n_control_masked": n_control_masked,
        "n_control_overlapping": n_control_overlap,
        "overall_pass": all(clauses.values()),
    }


def build(
    *,
    context_parquet: str | Path,
    union_parquet: str | Path,
    corpus_parquet: str | Path,
    out_parquet: str | Path = MINING_POOL_PARQUET,
    provenance_path: str | Path = MINING_POOL_PROVENANCE,
    report_path: str | Path = MINING_POOL_REPORT,
    seed: int = provenance.DEFAULT_SEED,
    window_nt: int = DEFAULT_WINDOW_NT,
    margin_nt: int = DEFAULT_FLANK_MARGIN_NT,
    n_controls: int = 500,
    flank_nt: int = masking.DEFAULT_FLANK_NT,
    env_lock: str | Path | None = None,
) -> int:
    """Carve, mask, gate, and write the coordinate-bearing mining substrate."""
    import pandas as pd

    context = pd.read_parquet(context_parquet)
    rows = context.to_dict("records")
    records = carve_pool(
        rows,
        seed=seed,
        window_nt=window_nt,
        margin_nt=margin_nt,
        n_controls=n_controls,
    )
    if not records:
        raise MiningPoolError(
            f"no windows carved from {context_parquet} — refusing to write an empty "
            "substrate that would make every downstream count vacuously zero"
        )

    union_loci, n_union_total, n_union_maskable = masking.load_union_loci(union_parquet)
    own_loci = masking.load_own_positive_loci(corpus_parquet)
    index = masking.LocusIndex.from_records(union_loci + own_loci)
    mask_summary = mask_pool(records, index, flank=flank_nt)
    gate = control_gate(mask_summary)

    for record in records:
        record["masked"] = index.is_masked(
            record["accession"], record["locus_start"], record["locus_end"], flank=flank_nt
        )

    df = pd.DataFrame.from_records(records)
    out_parquet = Path(out_parquet)

    report = {
        "n_records": len(df),
        "pool": POOL_GENOMIC_WINDOW,
        "window_nt": window_nt,
        "margin_nt": margin_nt,
        "flank_nt": flank_nt,
        "seed": seed,
        "n_context_rows": len(rows),
        "n_context_anchored": sum(1 for r in rows if str(r.get("status")) == CONTEXT_STATUS_OK),
        "side_counts": {
            side: int((df["side"] == side).sum())
            for side in (SIDE_LEAD, SIDE_TRAIL, SIDE_LOCUS_CONTROL)
        },
        "masking": mask_summary,
        "control_gate": gate,
        "union_denominator": n_union_total,
        "union_maskable_with_coords": n_union_maskable,
        "context_sha256": provenance.sha256_file(context_parquet),
        "union_prior_sha256": provenance.sha256_file(union_parquet),
        "notes": (
            "P2-10b mining substrate for the ADR-0005 D14 loop — NOT a fifth PRD §9.1 "
            "negative class and NOT annotation-verified 5′UTRs/leaders (that is "
            "P2-10b′). Windows are carved from the P2-00 flank_context regions, so "
            "they sit on replicons that host known T-boxes and the mask has something "
            "to find. designed_control windows are locus-centred and overlap a known "
            "locus by construction: they validate the coordinate frame and are "
            "excluded from the natural rate."
        ),
    }
    # The report is written first and unconditionally — it is the diagnosis — but
    # the parquet and its provenance are written **only after the gate passes**.
    # A gate-failed substrate left at the canonical path is the dangerous artifact:
    # `dvc add` does not consult the gate, so a later commit would ship a pool
    # whose coordinate frame is known-broken, provenance-stamped and indistinguishable
    # from a good one.
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if not gate["overall_pass"]:
        failed = sorted(k for k, v in gate["clauses"].items() if not v)
        raise MiningPoolError(
            f"P2-10b non-vacuity gate FAILED on {failed} — report written to "
            f"{report_path} for diagnosis; no substrate written (it is not usable "
            "for mining and must not be dvc-added)"
        )

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    provenance.write_provenance(
        provenance_path,
        rule="workflow/rules/data.smk :: build_mining_pool",
        script="src/tbox_finder/mining/pool.py",
        seed=seed,
        inputs=[context_parquet, union_parquet, corpus_parquet],
        outputs=[out_parquet, report_path],
        env_lock=env_lock,
        adr="ADR-0005",
        extra={
            "n_records": len(df),
            "control_gate_pass": gate["overall_pass"],
            "n_control_masked": gate["n_control_masked"],
            "n_control_overlapping": gate["n_control_overlapping"],
            "n_control_windows": gate["n_control_windows"],
            "natural_masked_fraction": mask_summary["natural"]["masked_fraction"],
        },
    )
    print(
        f"built {len(df)} mining-pool windows "
        f"(control {gate['n_control_masked']}/{gate['n_control_windows']} masked; "
        f"natural {mask_summary['natural']['n_masked_at_flank']}/"
        f"{mask_summary['natural']['n_records']})",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbox_finder.mining.pool")
    p.add_argument("--context", default="data/interim/flank_context/context_v0.parquet")
    p.add_argument("--union-prior", default="data/processed/priors/union_prior.parquet")
    p.add_argument("--corpus", default="data/processed/master_clean_v0.parquet")
    p.add_argument("--out", default=MINING_POOL_PARQUET)
    p.add_argument("--provenance", default=MINING_POOL_PROVENANCE)
    p.add_argument("--report", default=MINING_POOL_REPORT)
    p.add_argument("--config", default="conf/data/decoys.yaml")
    p.add_argument("--env-lock", default=None)
    a = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    # Sizing + seeding come from the same seeded config the §9.1 pools use, so the
    # substrate and the pools it is mined alongside cannot drift apart (CLAUDE.md §8.3).
    from tbox_finder.decoys import read_config

    cfg = read_config(a.config)
    return build(
        context_parquet=a.context,
        union_parquet=a.union_prior,
        corpus_parquet=a.corpus,
        out_parquet=a.out,
        provenance_path=a.provenance,
        report_path=a.report,
        seed=cfg.seed,
        window_nt=cfg.mining_window_nt,
        margin_nt=cfg.mining_margin_nt,
        n_controls=cfg.mining_n_controls,
        flank_nt=cfg.flank_nt,
        env_lock=a.env_lock,
    )


if __name__ == "__main__":
    raise SystemExit(main())
