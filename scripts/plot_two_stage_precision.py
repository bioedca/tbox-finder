#!/usr/bin/env python
"""P3-16 — draw the two-stage-vs-Stage-1-only AUPRC-by-prevalence figure (`@fig-twostage`).

Reads the **committed report only**. It re-derives nothing: every value plotted is a field
of `reports/two_stage_precision.json`, so the figure and the graded number cannot disagree,
and a re-run of the gate moves both together. The gated point is drawn as the pinned
prevalence, and the cluster-blocked interval on the gain is drawn **as it is** — including a
lower bound below zero, which is the honest shape of this measurement.

A schema-1.2 report (the ADR-0005 A15 v2 re-grade) is drawn the same way, with two
additions read from its own fields: the flank-familiarity baseline (A15 Pin 7(b)), drawn as
one marker at the pinned prevalence, which is the only prevalence it is graded at; and,
when ``gate.verdict_class`` is ``"underpowered"``, A15 Pin 5's sentence as an annotation, so
the figure cannot be read as "two-stage did not beat Stage-1-only". A 1.1 report (the
withdrawn v1 grade) draws exactly as before.

Usage::

    python scripts/plot_two_stage_precision.py \\
        --report reports/two_stage_precision.json \\
        --out figures/integration/two_stage_precision.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_LABEL = {"stage1_only": "Stage-1 only", "two_stage": "two-stage"}
ARM_STYLE = {
    "twin": dict(linestyle="-", linewidth=2.0),
    "production": dict(linestyle="--", linewidth=1.3),
}


def sweep_points(arm: dict) -> list[tuple[float, dict, float]]:
    """``(decoy_ratio, auprc-by-system, auprc_gain_pp)`` ordered by prevalence."""
    rows = [
        (float(node["decoy_ratio"]), node["auprc"], float(node["auprc_gain_pp"]))
        for node in arm["prevalence"].values()
    ]
    return sorted(rows, key=lambda row: row[0])


def underpowered_note(report: dict) -> str | None:
    """A15 Pin 5's sentence when the gate is underpowered, else ``None``.

    Read from the report's own ``gate`` fields, never composed here, so the figure says
    exactly what the graded artifact says.
    """
    gate = report.get("gate") or {}
    if gate.get("verdict_class") != "underpowered":
        return None
    return str(gate.get("underpowered_statement") or "underpowered")


def familiarity_point(report: dict) -> tuple[float, float] | None:
    """``(pinned prevalence, gated arm's familiarity AUPRC)`` for a 1.2 report, else ``None``."""
    baseline = report.get("familiarity_baseline")
    if not isinstance(baseline, dict):
        return None
    entry = (baseline.get("by_arm") or {}).get(report["gated_arm"]) or {}
    auprc = entry.get("auprc")
    if not isinstance(auprc, (int, float)) or isinstance(auprc, bool):
        return None
    return float(report["gate"]["decoy_prevalence"]), float(auprc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Imported here, not at module level, so the report-reading helpers above stay importable
    # (and unit-tested) where matplotlib is not installed: neither CI nor the local test env
    # carries it, and a module-level import would make any test of them skip green.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    gate = report["gate"]
    gated_arm = report["gated_arm"]
    pinned = float(gate["decoy_prevalence"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    for arm_name, arm in sorted(report["arms"].items()):
        style = ARM_STYLE.get(arm_name, {})
        rows = sweep_points(arm)
        ratios = [row[0] for row in rows]
        for system, colour in (("stage1_only", "#B45309"), ("two_stage", "#1D4ED8")):
            left.plot(
                ratios,
                [row[1][system] for row in rows],
                color=colour,
                marker="o" if arm_name == gated_arm else None,
                markersize=4,
                label=f"{SYSTEM_LABEL[system]} ({arm_name})",
                **style,
            )
        right.plot(
            ratios,
            [row[2] for row in rows],
            color="#047857",
            marker="o" if arm_name == gated_arm else None,
            markersize=4,
            label=f"gain ({arm_name})",
            **style,
        )

    ci = gate["gain_ci"]
    # The interval belongs to the GATED point only — it was resampled at the pinned
    # prevalence. Drawn as an error bar there, never as a band across the sweep: a band
    # would read as an interval on every prevalence, which no replicate estimated.
    right.errorbar(
        [pinned],
        [gate["gain_pp"]],
        yerr=[[gate["gain_pp"] - ci["lower"]], [ci["upper"] - gate["gain_pp"]]],
        fmt="none",
        ecolor="#047857",
        elinewidth=1.6,
        capsize=5,
        zorder=5,
        label=f"95% CI at {pinned:g}:1 (cluster-blocked)",
    )
    right.axhline(0.0, color="#374151", linewidth=0.8)
    for axis in (left, right):
        axis.set_xscale("log")
        axis.axvline(pinned, color="#6B7280", linewidth=0.9, linestyle=":")
        axis.set_xlabel("decoy : positive prevalence")
        axis.grid(alpha=0.25, linewidth=0.5)
    left.set_ylabel("AUPRC")
    left.set_title("AUPRC by decoy prevalence")
    right.set_ylabel("two-stage − Stage-1-only (pp)")
    right.set_title(
        f"gain; gated point {gate['gain_pp']:+.1f} pp " f"[{ci['lower']:+.1f}, {ci['upper']:+.1f}]"
    )
    familiar = familiarity_point(report)
    if familiar is not None:
        # Graded at the pinned prevalence only (A15 Pin 7(b)), so ONE marker, not a curve.
        left.plot(
            [familiar[0]],
            [familiar[1]],
            linestyle="none",
            marker="D",
            markersize=6,
            color="#6D28D9",
            label=f"flank-familiarity baseline ({gated_arm}, {familiar[0]:g}:1)",
        )
    left.legend(fontsize=7, loc="lower left")
    right.legend(fontsize=7, loc="upper left")
    note = underpowered_note(report)
    if note is not None:
        right.text(
            0.5,
            0.5,
            f"verdict_class: underpowered\n{note}",
            transform=right.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            wrap=True,
            zorder=10,
            bbox=dict(boxstyle="round", facecolor="#FEF3C7", edgecolor="#B45309"),
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(
        f"wrote {out}: gated arm {gated_arm}, pinned prevalence {pinned:g}:1"
        + (f", verdict_class {gate['verdict_class']}" if "verdict_class" in gate else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
