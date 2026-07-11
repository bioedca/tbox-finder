"""OOD-ECE min-N coverage simulation (P0-27) — pins the ADR-0005 D13 OOD-ECE
min-N admissibility floor.

The §13 genome-wide scan renders one calibration/drift verdict per deployment
corpus (PRD §7.2 / §12). A corpus is a **"calibrated-negative PASS"** only if
its nearest-relative leave-clade-out OOD-ECE (i) meets a drift bound, (ii) clears
a **min-N admissibility floor**, and (iii) clears a detection-power floor; else it
is a **"sensitivity-bounded / inconclusive negative *by rule*."** Condition (ii) —
the min-N floor — is what this module pins, and its coverage over the real
partition is what this simulation reports.

Following PRD §12's mandate ("a P0/P3 coverage simulation reports how many
held-out orders clear the min-N bar, so the adjudicable fraction of the scan is
known ex ante") this audit, over the leave-clade-out partition (P0-22), counts:

  * how many held-out **orders** (the leave-one-order-out benchmark unit) clear
    the pinned floor → the adjudicable calibration footprint;
  * the **ex-ante adjudicable fraction** and the **inconclusive-by-rule fraction**
    (sub-min-N positive-bearing orders + the named zero-positive scan superclades
    Archaea / DPANN / CPR / candidate phyla, PRD §7.2);
  * the predicted **per-corpus verdict-vector shape** feeding the §2.3
    project-level rollup (pinned in ADR-0006 at P0-29), quantifying why
    **"discovery-predominantly-inconclusive"** is a *pre-registered* modal
    terminal outcome, not a surprise.

It never fabricates counts (CLAUDE.md §10.3): every number is read from the real
P0-22 partition, and the per-order N is cross-checked byte-for-byte against the
signed P0-26 ``power_budget_report.json``.

Structure mirrors ``power.py``: pure, stdlib-only predicates (unit-testable
without numpy/pandas) on top; the heavy audit (numpy/pandas lazy) below. CLI
subcommands ``simulate`` (data env) and ``plot-figures`` (viz env).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tbox_finder import power, provenance, splits

# --------------------------------------------------------------------------- #
# Pinned values (ADR-0005; this module amends D13's min-N delegation)
# --------------------------------------------------------------------------- #

# ADR-0005 D18 delegation → pinned here (P0-27). The OOD-ECE / drift decision rule
# (D13) estimates each deployment corpus's leave-clade-out ECE with a small-N-robust
# estimator (proper-scoring-rule decomposition or kernel calibration, bootstrap CIs),
# admissible only at or above this many real positives in the corpus's calibration
# unit. Below it, no OOD-ECE is admitted → the corpus is "sensitivity-bounded /
# inconclusive negative *by rule*" (never a silent calibrated-negative PASS).
#
# Pinned = 20, internally consistent with the recall floor (``power.MIN_REAL_HOMOLOG_N``),
# the split's well-powered leave-one-order-out unit (conf/data/splits.yaml
# ``min_heldout_positives: 20``), and PRD §12 ("31 orders with ≥20 positives"; the
# ~20-positive floor is statistically unstable → the boundary of admissibility). The
# small-N-robust OOD estimator is chosen precisely so N = 20 is admissible (bootstrap
# CI reported, not per-bin filled). Blinded-frozen at P0 — a change needs ADR-0005
# re-sign-off (CLAUDE.md §7 item 2). Frozen in code (no CLI/config override) so no
# committed report can contradict the ADR-pinned threshold.
OOD_ECE_MIN_N = 20

# Reporting sweep — coverage sensitivity to the floor (the pinned value is 20; the
# sweep is diagnostic, not a threshold). Not configurable (report-shape stability).
FLOOR_SWEEP = (10, 15, 20, 30, 50)

# Named zero-positive scan superclades (PRD §7.2 targeted under-sampled clades /
# stretch targets). Each stands for an entire domain / superphylum of the scanned
# tree that carries **zero** known T-box positives in the CM-derived catalogue —
# "0/23,535 TBDB records are archaeal" is the catalogue's silence (PRD §7.2),
# verified here as measured absence from the partition's resolved phyla. They are
# inconclusive-by-rule ex ante (no leave-clade-out ECE is even computable).
NAMED_ZERO_POSITIVE_CLADES = ("Archaea", "DPANN", "CPR/Patescibacteria", "candidate-phyla")

DEFAULT_N_BOOT = 2000
DEFAULT_CI_LEVEL = 0.95

# Reused split-construction constants (single source of truth = splits.py).
INTERIM_TABLE = splits.INTERIM_TABLE
AUDIT_DIR = splits.AUDIT_DIR

POWER_REPORT = f"{AUDIT_DIR}/power_budget_report.json"
COVERAGE_REPORT = f"{AUDIT_DIR}/ood_ece_coverage_report.json"
COVERAGE_FIGURE_DATA = f"{AUDIT_DIR}/ood_ece_coverage_figure_data.json"
COVERAGE_CONFIG = "conf/data/coverage.yaml"
FIGURES_DIR = "figures/coverage"


# --------------------------------------------------------------------------- #
# Pure predicates (stdlib only — unit-testable without numpy/pandas)
# --------------------------------------------------------------------------- #


def orders_clearing_floor(order_counts: dict[str, int], floor: int = OOD_ECE_MIN_N) -> list[str]:
    """Sorted held-out orders whose real-positive N clears the admissibility floor."""
    return sorted(o for o, n in order_counts.items() if n >= floor)


def adjudicable_fraction(order_counts: dict[str, int], floor: int = OOD_ECE_MIN_N) -> float:
    """Fraction of positive-bearing held-out orders that clear the floor (0 if empty).

    This is the PRD §12 quantity — the adjudicable fraction *among orders that carry
    any held-out positive*. The scan-wide fraction is far lower (the zero-positive
    complement — Archaea/DPANN/CPR/candidate phyla + every T-box-free bacterial
    phylum — is inconclusive-by-rule); see :func:`verdict_vector_shape`.
    """
    if not order_counts:
        return 0.0
    return sum(1 for n in order_counts.values() if n >= floor) / len(order_counts)


def classify_order(n: int, floor: int = OOD_ECE_MIN_N) -> str:
    """Per-corpus admissibility class: ``"adjudicable"`` iff ``n >= floor``."""
    return "adjudicable" if n >= floor else "sub_min_n_inconclusive"


def floor_sweep_coverage(order_counts: dict[str, int], sweep=FLOOR_SWEEP) -> dict[str, dict]:
    """Adjudicable order count + fraction at each swept candidate floor.

    Monotone: ``n_clearing`` is non-increasing as the floor rises (visualizes the
    sensitivity of coverage to the pinned value).
    """
    total = len(order_counts)
    out: dict[str, dict] = {}
    for f in sweep:
        n_clear = sum(1 for n in order_counts.values() if n >= f)
        out[str(f)] = {
            "floor": int(f),
            "n_clearing": int(n_clear),
            "n_orders": int(total),
            "adjudicable_fraction": (n_clear / total if total else 0.0),
        }
    return out


def verdict_vector_shape(
    order_counts: dict[str, int],
    n_named_zero_positive_clades: int,
    floor: int = OOD_ECE_MIN_N,
) -> dict:
    """Predicted per-corpus verdict-vector shape (the §2.3 rollup input).

    Categories mirror PRD §2.3 / §12: ``adjudicable`` (min-N-clearing → eligible for
    a calibrated-negative PASS or a positive claim, subject to the D13 drift + power
    floors) vs ``inconclusive-by-rule`` (sub-min-N positive-bearing orders + the named
    zero-positive scan superclades). ADR-0006 (P0-29) pins the exact rollup function;
    this reports the *shape* it consumes.
    """
    adj = sum(1 for n in order_counts.values() if n >= floor)
    sub = sum(1 for n in order_counts.values() if n < floor)
    zero = int(n_named_zero_positive_clades)
    return {
        "adjudicable": int(adj),
        "sub_min_n_inconclusive": int(sub),
        "zero_positive_inconclusive_named_clades": zero,
        "inconclusive_by_rule_total": int(sub + zero),
    }


def modal_shape_is_inconclusive(shape: dict) -> bool:
    """``True`` iff inconclusive-by-rule is the modal per-corpus verdict of the scan.

    The named zero-positive superclades each stand for an entire domain / superphylum
    of the scanned prokaryotic tree (Archaea, DPANN, CPR, candidate phyla), so the
    inconclusive mass dominates whenever any such superclade is zero-positive **or**
    sub-min-N orders already outnumber adjudicable ones. Either condition makes the
    scan not predominantly adjudicable → "discovery-predominantly-inconclusive."
    """
    return (
        shape["zero_positive_inconclusive_named_clades"] > 0
        or shape["inconclusive_by_rule_total"] >= shape["adjudicable"]
    )


def predicted_gate3_modal_shape(shape: dict) -> str:
    """The §2.3 project-level rollup modal shape predicted ex ante (not the ADR-0006
    rollup itself — that is pinned at P0-29). Either ``discovery-predominantly-
    inconclusive`` or ``adjudicable-predominant``."""
    return (
        "discovery-predominantly-inconclusive"
        if modal_shape_is_inconclusive(shape)
        else "adjudicable-predominant"
    )


# --------------------------------------------------------------------------- #
# Heavy audit (numpy/pandas lazy)
# --------------------------------------------------------------------------- #


def _read_heldout_order_counts(
    table: str | Path = INTERIM_TABLE,
) -> tuple[dict[str, int], dict[str, str], dict[str, int]]:
    """Per-order held-out N, per-order majority phylum, and per-phylum held-out N.

    Mirrors ``power.build_report``'s grouping exactly: held-out subframe
    (``nested_role == "heldout"``) grouped by ``loo_order_unit`` — pandas' default
    ``groupby`` drops the NaN-order rows (the 34 externals + the 4 clade-crossing-swept
    corpus rows), leaving the resolved leave-one-order-out units. The per-phylum count
    is over held-out **corpus** positives (the calibration reference set).
    """
    import pandas as pd

    df = pd.read_parquet(table)
    heldout = df[df["nested_role"] == "heldout"]

    order_counts: dict[str, int] = {}
    order_phylum: dict[str, str] = {}
    for order, grp in heldout.groupby("loo_order_unit"):
        key = str(order)
        order_counts[key] = int(len(grp))
        phyla = Counter(p for p in grp["resolved_phylum"] if isinstance(p, str) and p)
        order_phylum[key] = phyla.most_common(1)[0][0] if phyla else ""

    corpus = heldout[heldout["source"] == "corpus"]
    phylum_counts = dict(
        Counter(p for p in corpus["resolved_phylum"] if isinstance(p, str) and p).most_common()
    )
    return order_counts, order_phylum, phylum_counts


def _bootstrap_fraction_ci(
    order_counts: dict[str, int],
    floor: int,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = provenance.DEFAULT_SEED,
) -> dict:
    """Seeded block bootstrap over held-out orders (the D5 resampling block) for the
    adjudicable fraction + clearing count. Resamples the orders with replacement.
    """
    import numpy as np

    counts = np.asarray(list(order_counts.values()), dtype=float)
    n = counts.size
    if n == 0:
        return {
            "n_boot": int(n_boot),
            "ci_level": float(ci_level),
            "seed": int(seed),
            "fraction_mean": 0.0,
            "fraction_lo": 0.0,
            "fraction_hi": 0.0,
            "n_clearing_lo": 0,
            "n_clearing_hi": 0,
        }
    rng = np.random.default_rng(seed)
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    fracs = np.empty(n_boot, dtype=float)
    clears = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(counts, size=n, replace=True)
        cleared = sample >= floor
        fracs[b] = float(cleared.mean())
        clears[b] = float(cleared.sum())
    return {
        "n_boot": int(n_boot),
        "ci_level": float(ci_level),
        "seed": int(seed),
        "fraction_mean": float(fracs.mean()),
        "fraction_lo": float(np.quantile(fracs, lo_q)),
        "fraction_hi": float(np.quantile(fracs, hi_q)),
        "n_clearing_lo": int(np.quantile(clears, lo_q)),
        "n_clearing_hi": int(np.quantile(clears, hi_q)),
    }


def _cross_check_power_report(
    order_counts: dict[str, int], power_report: str | Path = POWER_REPORT
) -> dict:
    """Cross-check per-order N against the signed P0-26 report (non-fabrication guard).

    Returns a match summary; the values must agree exactly (both group the same
    held-out frame by ``loo_order_unit``). Absent report → ``available: False``
    (never a fabricated pass).
    """
    path = Path(power_report)
    if not path.exists():
        return {"available": False, "power_report": str(power_report)}
    rounds = json.loads(path.read_text()).get("per_leave_one_order_out_round", {})
    ref = {k: int(v["n_heldout"]) for k, v in rounds.items()}
    n_matches = order_counts == ref
    mismatches = sorted(
        k for k in set(order_counts) | set(ref) if order_counts.get(k) != ref.get(k)
    )
    return {
        "available": True,
        "power_report": str(power_report),
        "n_orders_matches": len(order_counts) == len(ref),
        "per_order_n_matches": bool(n_matches),
        "n_mismatched_orders": len(mismatches),
        "mismatched_orders": mismatches[:10],
    }


def build_report(
    order_counts: dict[str, int],
    order_phylum: dict[str, str],
    phylum_counts: dict[str, int],
    boot: dict,
    cross_check: dict,
    *,
    floor: int = OOD_ECE_MIN_N,
    sweep=FLOOR_SWEEP,
    named_zero_positive_clades=NAMED_ZERO_POSITIVE_CLADES,
) -> dict:
    """Assemble the coverage-simulation report (pure over the count dicts)."""
    n_orders = len(order_counts)
    clearing = orders_clearing_floor(order_counts, floor)
    n_clearing = len(clearing)
    n_sub = n_orders - n_clearing

    # Adjudicable calibration footprint (phyla the clearing / positive-bearing orders span).
    phyla_positive = sorted({p for p in order_phylum.values() if p})
    phyla_clearing = sorted({order_phylum[o] for o in clearing if order_phylum.get(o)})

    # Named zero-positive superclades: measured absence from the resolved phyla.
    present = set(phyla_positive)
    named_present = sorted(c for c in named_zero_positive_clades if c in present)

    shape = verdict_vector_shape(order_counts, len(named_zero_positive_clades), floor)

    total_heldout = sum(order_counts.values())
    top_order_n = max(order_counts.values()) if order_counts else 0
    top_phylum_n = max(phylum_counts.values()) if phylum_counts else 0
    corpus_heldout = sum(phylum_counts.values())

    per_order = {
        o: {
            "n_heldout": int(order_counts[o]),
            "phylum": order_phylum.get(o, ""),
            "clears_floor": bool(order_counts[o] >= floor),
        }
        for o in sorted(order_counts)
    }

    return {
        "ood_ece_min_n": int(floor),
        "min_n_admissibility_rationale": (
            "OOD-ECE min-N admissibility floor (ADR-0005 D13). Below this many real "
            "positives in a corpus's leave-clade-out calibration unit the small-N-robust "
            "OOD-ECE estimator is inadmissible (the ~20-positive per-order ECE is "
            "statistically unstable, PRD §12) → the corpus is sensitivity-bounded / "
            "inconclusive-by-rule, never a silent calibrated-negative PASS. Pinned = 20, "
            "internally consistent with the recall floor MIN_REAL_HOMOLOG_N = "
            f"{power.MIN_REAL_HOMOLOG_N} (P0-26), the split's min_heldout_positives = 20, "
            "and PRD §12. Blinded-frozen at P0."
        ),
        "internal_consistency": {
            "ood_ece_min_n": int(floor),
            "recall_min_real_homolog_n": int(power.MIN_REAL_HOMOLOG_N),
            "consistent": bool(floor == power.MIN_REAL_HOMOLOG_N),
        },
        "floor_sweep": list(sweep),
        "floor_sweep_coverage": floor_sweep_coverage(order_counts, sweep),
        "n_orders_with_heldout": int(n_orders),
        "n_orders_clearing_floor": int(n_clearing),
        "adjudicable_fraction_among_positive_bearing_orders": (
            n_clearing / n_orders if n_orders else 0.0
        ),
        "adjudicable_fraction_ci": boot,
        "inconclusive_by_rule": {
            "n_sub_min_n_positive_bearing_orders": int(n_sub),
            "sub_min_n_fraction_among_positive_bearing": (n_sub / n_orders if n_orders else 0.0),
            "named_zero_positive_scan_clades": list(named_zero_positive_clades),
            "named_zero_positive_clades_present_in_corpus": named_present,
            "note": (
                "The scan-wide inconclusive-by-rule fraction is far larger than the "
                "positive-bearing-order fraction: every scanned corpus outside the "
                f"{len(phyla_positive)}-phylum calibration footprint (all Archaea / DPANN / "
                "CPR / candidate phyla + every T-box-free bacterial phylum) carries 0 "
                "held-out positives → inconclusive-by-rule ex ante. A precise scan-wide "
                "denominator is not pinnable here (the GTDB corpus enumeration is deferred "
                "to ADR-0003); it is bounded above by nearest-relative reach of this footprint."
            ),
        },
        "verdict_vector_shape": {
            **shape,
            "modal_shape_is_inconclusive": bool(modal_shape_is_inconclusive(shape)),
            "modal_shape_basis": (
                f"calibration footprint confined to {len(phyla_positive)} bacterial phyla "
                "(0 archaeal); the §13 scan (PRD §7.2) targets the whole prokaryotic tree "
                "incl. the named zero-positive superclades → inconclusive-by-rule is modal"
            ),
        },
        "predicted_gate3_modal_shape": predicted_gate3_modal_shape(shape),
        "adjudicable_footprint": {
            "n_phyla_positive_bearing": len(phyla_positive),
            "phyla_positive_bearing": phyla_positive,
            "n_phyla_clearing_floor": len(phyla_clearing),
            "phyla_clearing_floor": phyla_clearing,
            "per_phylum_heldout_n": phylum_counts,
            "top_order_share_of_heldout": (top_order_n / total_heldout if total_heldout else 0.0),
            "top_phylum_share_of_corpus_heldout": (
                top_phylum_n / corpus_heldout if corpus_heldout else 0.0
            ),
        },
        "per_order": per_order,
        "consistency_check": cross_check,
        "seed": int(boot.get("seed", provenance.DEFAULT_SEED)),
    }


def _figure_data(report: dict) -> dict:
    """Compact numeric payload for the viz-env plotter (no pandas)."""
    per_order = report["per_order"]
    ordered = sorted(per_order.items(), key=lambda kv: kv[1]["n_heldout"], reverse=True)
    sweep = report["floor_sweep_coverage"]
    shape = report["verdict_vector_shape"]
    return {
        "min_n": int(report["ood_ece_min_n"]),
        "per_order_labels": [k for k, _ in ordered],
        "per_order_n": [int(v["n_heldout"]) for _, v in ordered],
        "floor_sweep": [int(f) for f in report["floor_sweep"]],
        "floor_sweep_fraction": [
            sweep[str(f)]["adjudicable_fraction"] for f in report["floor_sweep"]
        ],
        "floor_sweep_n_clearing": [sweep[str(f)]["n_clearing"] for f in report["floor_sweep"]],
        "verdict_shape": {
            "adjudicable": int(shape["adjudicable"]),
            "sub_min_n_inconclusive": int(shape["sub_min_n_inconclusive"]),
            "zero_positive_inconclusive_named_clades": int(
                shape["zero_positive_inconclusive_named_clades"]
            ),
        },
    }


def run_sim(
    *,
    table: str | Path = INTERIM_TABLE,
    power_report: str | Path = POWER_REPORT,
    out_report: str | Path = COVERAGE_REPORT,
    figure_data: str | Path = COVERAGE_FIGURE_DATA,
    config: str | Path = COVERAGE_CONFIG,
    env_lock: str | None = None,
) -> int:
    """Heavy entry: read the partition, run the seeded bootstrap, write the report."""
    seed, n_boot, ci_level = _read_config(config)
    floor = OOD_ECE_MIN_N  # frozen — never from CLI/config (no report may contradict the pin)

    order_counts, order_phylum, phylum_counts = _read_heldout_order_counts(table)
    boot = _bootstrap_fraction_ci(order_counts, floor, n_boot=n_boot, ci_level=ci_level, seed=seed)
    cross_check = _cross_check_power_report(order_counts, power_report)
    report = build_report(order_counts, order_phylum, phylum_counts, boot, cross_check, floor=floor)

    out_report = Path(out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    figure_data = Path(figure_data)
    figure_data.parent.mkdir(parents=True, exist_ok=True)
    figure_data.write_text(json.dumps(_figure_data(report)) + "\n")

    provenance.write_provenance(
        out_report.with_suffix(".provenance.json"),
        rule="workflow/rules/data.smk :: ood_ece_coverage_sim",
        script="src/tbox_finder/coverage.py",
        seed=seed,
        inputs=[table, power_report, config],
        outputs=[out_report, figure_data],
        env_lock=env_lock,
        adr="ADR-0005",
        extra={
            "ood_ece_min_n": int(floor),
            "n_boot": int(n_boot),
            "ci_level": float(ci_level),
        },
    )

    summary = report["floor_sweep_coverage"][str(floor)]
    print(
        f"OOD-ECE coverage: floor={floor} → {report['n_orders_clearing_floor']}/"
        f"{report['n_orders_with_heldout']} held-out orders clear "
        f"(adjudicable fraction {summary['adjudicable_fraction']:.3f}); "
        f"modal GATE-3 shape = {report['predicted_gate3_modal_shape']}",
        file=sys.stderr,
    )
    if cross_check.get("available") and not cross_check.get("per_order_n_matches"):
        print(
            f"WARNING: per-order N disagrees with {power_report} "
            f"({cross_check.get('n_mismatched_orders')} orders) — investigate before trusting.",
            file=sys.stderr,
        )
    return 0


def _read_config(config: str | Path) -> tuple[int, int, float]:
    """Read seed / n_boot / ci_level from the seeded config (falls back to pins)."""
    path = Path(config)
    seed, n_boot, ci_level = provenance.DEFAULT_SEED, DEFAULT_N_BOOT, DEFAULT_CI_LEVEL
    if not path.exists():
        return seed, n_boot, ci_level
    text = path.read_text()
    # Minimal, dependency-free scalar reader (avoid a yaml dep in the data env for
    # three scalars): parse top-level ``key: value`` integer/float lines.
    for line in text.splitlines():
        raw = line.split("#", 1)[0].strip()
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        key, val = key.strip(), val.strip()
        if key == "seed" and val:
            seed = int(val)
        elif key == "n_boot" and val:
            n_boot = int(val)
        elif key == "ci_level" and val:
            ci_level = float(val)
    return seed, n_boot, ci_level


def plot_figures(
    *, figure_data: str | Path = COVERAGE_FIGURE_DATA, out_dir: str | Path = FIGURES_DIR
) -> int:
    """Render the coverage figures from the compact figure-data JSON (viz env)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = json.loads(Path(figure_data).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_n = fig["min_n"]

    # (1) Per-order held-out N (sorted) with the min-N admissibility line.
    n = fig["per_order_n"]
    f1, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2b8a3e" if v >= min_n else "#c92a2a" for v in n]
    ax.bar(range(len(n)), n, color=colors)
    ax.axhline(min_n, color="k", linestyle="--", linewidth=1, label=f"min-N = {min_n}")
    ax.set_yscale("symlog")
    ax.set_xlabel("held-out leave-one-order-out unit (sorted by N)")
    ax.set_ylabel("real held-out positives")
    ax.set_title("OOD-ECE per-order coverage (green ≥ min-N = adjudicable)")
    ax.legend()
    f1.tight_layout()
    f1.savefig(out_dir / "per_order_coverage.png", dpi=150)
    plt.close(f1)

    # (2) Adjudicable fraction vs candidate floor (coverage sensitivity).
    f2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fig["floor_sweep"], fig["floor_sweep_fraction"], "o-", color="#1c7ed6")
    ax.axvline(min_n, color="#c92a2a", linestyle="--", linewidth=1, label=f"pinned floor = {min_n}")
    ax.set_xlabel("candidate OOD-ECE min-N admissibility floor")
    ax.set_ylabel("adjudicable fraction (positive-bearing orders)")
    ax.set_ylim(0, 1)
    ax.set_title("Coverage sensitivity to the admissibility floor")
    ax.legend()
    f2.tight_layout()
    f2.savefig(out_dir / "floor_sweep_coverage.png", dpi=150)
    plt.close(f2)

    # (3) Predicted per-corpus verdict-vector shape.
    shape = fig["verdict_shape"]
    labels = ["adjudicable", "sub-min-N\ninconclusive", "zero-positive\nnamed clades"]
    vals = [
        shape["adjudicable"],
        shape["sub_min_n_inconclusive"],
        shape["zero_positive_inconclusive_named_clades"],
    ]
    f3, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["#2b8a3e", "#f08c00", "#c92a2a"])
    ax.set_ylabel("corpus count")
    ax.set_title("Predicted per-corpus verdict-vector shape (§2.3 rollup input)")
    f3.tight_layout()
    f3.savefig(out_dir / "verdict_vector_shape.png", dpi=150)
    plt.close(f3)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OOD-ECE min-N coverage simulation (P0-27)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="run the coverage simulation (data env)")
    s.add_argument("--table", default=INTERIM_TABLE)
    s.add_argument("--power-report", default=POWER_REPORT)
    s.add_argument("--out-report", default=COVERAGE_REPORT)
    s.add_argument("--figure-data", default=COVERAGE_FIGURE_DATA)
    s.add_argument("--config", default=COVERAGE_CONFIG)
    s.add_argument("--env-lock", default=None)

    g = sub.add_parser("plot-figures", help="render the coverage figures (viz env)")
    g.add_argument("--figure-data", default=COVERAGE_FIGURE_DATA)
    g.add_argument("--out-dir", default=FIGURES_DIR)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "simulate":
        return run_sim(
            table=args.table,
            power_report=args.power_report,
            out_report=args.out_report,
            figure_data=args.figure_data,
            config=args.config,
            env_lock=args.env_lock,
        )
    if args.cmd == "plot-figures":
        return plot_figures(figure_data=args.figure_data, out_dir=args.out_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
