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
    # Refuse a missing/blank accession rather than letting str() launder it. A
    # None/NaN/pd.NA accession stringifies to "None"/"nan"/"<NA>", none of which
    # masking.is_missing recognises, so the window would reach the mask carrying a
    # coordinate that addresses no replicon and be classified *minable* instead of
    # refused. Unreachable today (the only producer sets "" on the bad_name path,
    # which the status guard above already rejects) — but that guard is one
    # refactor away and this is the fail-open direction.
    if masking.is_missing(row.get("accession")) or not str(row.get("accession", "")).strip():
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


def window_is_masked(record: Mapping[str, Any], index: masking.LocusIndex, *, flank: int) -> bool:
    """Whether one carved window is masked — the single predicate for the whole module.

    :func:`mask_pool` computes the reported counts and :func:`build` stamps the
    per-record ``masked`` column; routing both through here is what stops the
    artifact's own column from silently disagreeing with the report that grades
    it, which is the failure a second inlined ``is_masked`` call invites
    (CodeRabbit r1).
    """
    return index.is_masked(
        record["accession"], record["locus_start"], record["locus_end"], flank=flank
    )


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
        n_masked = sum(1 for r in subset if window_is_masked(r, index, flank=flank))
        n_overlap = sum(1 for r in subset if window_is_masked(r, index, flank=0))
        n_exact = sum(
            1
            for r in subset
            if index.matches_interval_exactly(r["accession"], r["locus_start"], r["locus_end"])
        )
        out[group] = {
            "n_records": len(subset),
            "n_masked_at_flank": n_masked,
            "n_overlapping_at_flank_0": n_overlap,
            "n_exact_interval_match": n_exact,
            "masked_fraction": (n_masked / len(subset)) if subset else 0.0,
        }
    out["accession_namespace"] = masking.accession_namespace_report(
        (r["accession"] for r in records), index
    )
    return out


def control_gate(
    mask_summary: Mapping[str, Any], *, n_controls_requested: int | None = None
) -> dict[str, Any]:
    """The P2-10b non-vacuity gate, re-derived from the recorded evidence.

    Every clause is wrapped so that a *missing* measurement is FALSE, not
    vacuously TRUE: a gate assembled from the requested configuration rather than
    the found evidence passes exactly when the evidence is absent.

    **The binding clause is exact interval identity, not overlap.** A control
    window *is* the locus, so overlap survives a shift of up to the locus length:
    measured uniform shifts of ±160 nt kept 500/500 controls both masked and
    overlapping while the reported natural contamination moved 0.184 % → 0.473 %.
    Exact reproduction of the known interval has zero tolerance and holds on the
    real data (23,532/23,532), so **every** control discriminates a frame error
    rather than only the ~1.6 % a shift happens to push clear of the locus. The
    overlap and flank clauses are retained as strictly weaker corroboration.

    ``n_controls_requested`` closes the other fail-open direction: ``carve_pool``
    treats its ``n_controls`` as a *request*, dropping any row whose geometry does
    not fit, so a run that emitted 2 of 8 controls previously graded green on the
    survivors with the shortfall recorded nowhere. Pass it and the shortfall is a
    failure; omit it and the clause is FALSE rather than absent.
    """
    control = mask_summary.get("designed_control") or {}
    natural = mask_summary.get("natural") or {}
    namespace = mask_summary.get("accession_namespace") or {}
    n_control = int(control.get("n_records", 0))
    n_control_masked = int(control.get("n_masked_at_flank", 0))
    n_control_overlap = int(control.get("n_overlapping_at_flank_0", 0))
    n_control_exact = int(control.get("n_exact_interval_match", 0))
    have_control = bool(control) and n_control > 0
    clauses = {
        "controls_present": have_control,
        "all_controls_exact": have_control and n_control_exact == n_control,
        "all_controls_overlap": have_control and n_control_overlap == n_control,
        "all_controls_masked": have_control and n_control_masked == n_control,
        "all_requested_controls_carved": (
            have_control
            and n_controls_requested is not None
            and n_control == int(n_controls_requested)
        ),
        "natural_pool_present": bool(natural) and int(natural.get("n_records", 0)) > 0,
        "namespace_compatible": bool(namespace) and bool(namespace.get("namespace_compatible")),
    }
    return {
        "clauses": clauses,
        "n_control_windows": n_control,
        "n_controls_requested": n_controls_requested,
        "n_control_masked": n_control_masked,
        "n_control_overlapping": n_control_overlap,
        "n_control_exact_interval_match": n_control_exact,
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

    # A carved flank window must sit at least ``flank_nt`` from its own parent
    # locus, or the mask would count it as contamination against the very locus it
    # was carved beside — a construction artifact in the headline natural rate.
    # ``margin_nt`` provides that clearance and is satisfied today with exactly
    # 0 nt of slack (both are 50), so a one-line config edit — a masking-flank
    # sensitivity sweep, say — silently breaks it: measured, flank 51 adds 3
    # self-masked windows to the natural count where flank 50 has 0.
    if margin_nt < flank_nt:
        raise MiningPoolError(
            f"margin_nt ({margin_nt}) < flank_nt ({flank_nt}): a carved window would fall "
            "within the masking flank of its own parent locus and be counted as natural "
            "contamination. Raise mining_margin_nt to at least flank_nt in conf/data/decoys.yaml."
        )

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
    # The union prior must actually contribute. Every locus-centred control is
    # carved from a corpus positive, so it masks against ``own_loci`` whether or
    # not the union prior is live — measured: masking from union-only, own-only,
    # and both gives bit-identical counts (control 500/500, natural 85/46,091).
    # The designed control therefore cannot detect a dead union prior, and the
    # report's ``union_denominator`` is derived from the parquet's row count, not
    # from the loci loaded, so it would still read 24,160 with zero loci in hand.
    # That is the step's headline guard (ADR-0005 D14's first) degrading silently
    # to own-positives-only, so it is asserted here rather than inferred.
    if n_union_maskable > 0 and not union_loci:
        raise MiningPoolError(
            f"union prior {union_parquet} reports {n_union_maskable} maskable records but "
            "yielded 0 loci — the mask would silently degrade to own-positives-only while "
            "every gate clause stayed green"
        )
    if len(union_loci) != n_union_maskable:
        raise MiningPoolError(
            f"union prior loci ({len(union_loci)}) != maskable record count ({n_union_maskable}) "
            "— the loader dropped records the report will still count in its denominator"
        )
    if not own_loci:
        raise MiningPoolError(f"corpus {corpus_parquet} yielded 0 own-positive loci")
    index = masking.LocusIndex.from_records(union_loci + own_loci)
    mask_summary = mask_pool(records, index, flank=flank_nt)
    gate = control_gate(mask_summary, n_controls_requested=n_controls)

    for record in records:
        record["masked"] = window_is_masked(record, index, flank=flank_nt)

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
        "n_controls_requested": n_controls,
        "union_denominator": n_union_total,
        "union_maskable_with_coords": n_union_maskable,
        # Loci actually loaded, beside the denominator derived from row counts —
        # so a reader can tell a live prior from a dead one without re-running.
        "n_union_loci_loaded": len(union_loci),
        "n_own_positive_loci_loaded": len(own_loci),
        "context_sha256": provenance.sha256_file(context_parquet),
        "union_prior_sha256": provenance.sha256_file(union_parquet),
        # The corpus supplies own_loci, so it determines every masked/overlap count
        # in this report as much as the union prior does — it belongs in the
        # diagnosis, not only in provenance.json's inputs list (CodeRabbit r1).
        "corpus_sha256": provenance.sha256_file(corpus_parquet),
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
            "n_control_exact_interval_match": gate["n_control_exact_interval_match"],
            "n_union_loci_loaded": len(union_loci),
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
