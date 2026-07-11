"""GATE-1 power-budget audit (P0-26) — pins the minimum-real-homolog N into ADR-0005.

This module audits, over the real leave-clade-out partition (P0-22), **how many
real held-out positives the corpus actually supplies per pre-registered %-identity
bin** (ADR-0005 D1) and **per headline-certifying arm** (ADR-0005 D6), so the
project can decide *ex ante* — before any P4 result — which GATE-1 arms are
**powered** (clear the pinned min-N and are gated) versus **reported-not-gated**
(below min-N, disclosed, removed from the AND-of-powered-arms).

Why this step exists (PRD §5 mechanism 2; ADR-0005 D8): the leave-clade-out
positives are themselves CM-derived, so they cannot contain T-boxes below
``cmsearch``'s gathering cutoff — the genuinely CM-invisible region is reached
only by the synthetic-divergence arm and the §13 discovery campaign. This audit
decides whether the §1 CM-invisibility claim can rest on the (gated) synthetic
arm or must rest on the §13 campaign. It **never fabricates counts** (CLAUDE.md
§10.3) — it reports what the corpus actually supports.

The **pinned minimum-real-homolog N** (``MIN_REAL_HOMOLOG_N``) is delegated to
this step by ADR-0005 D18 and amends that ADR (CLAUDE.md §7 item 2 sign-off).

Identity is measured with the **same consensus-column metric the split was built
on** — this module reuses :func:`tbox_finder.splits.consensus_matrix` and
:func:`tbox_finder.splits._nearest_cross_fold_identity` so the binning is
byte-identical to the ADR-0004 D2 adequacy histogram (no second, divergent
identity definition). Held-out positives are binned by % nucleotide identity to
the nearest **nested-training-fold** member (the shipped checkpoint's training
set); a held-out record with no coverage-adequate training neighbour carries the
``-1`` sentinel (even more divergent than the ``<50`` bin — reported separately).

Structure: pure, stdlib-only predicates (unit-testable without numpy/pandas) on
top; the heavy audit (numpy/pandas lazy) below. CLI subcommands ``audit`` (data
env) and ``plot-figures`` (viz env), mirroring ``splits.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tbox_finder import provenance, splits

# --------------------------------------------------------------------------- #
# Pinned values (ADR-0005; this module amends D18's min-N delegation)
# --------------------------------------------------------------------------- #

# ADR-0005 D18 delegation → pinned here (P0-26). Below this many real held-out
# positives in a cell (identity bin or headline-certifying arm) the strong D4
# bar's block-resampled CI cannot resolve its +5 pp positive floor (1/N ≥ 5 pp at
# N < 20), so the cell is **reported-not-gated** and the D4 weak-clause fallback
# (CI lower bound > 0) is admissible. Chosen = 20 to cohere with the repo's
# already-pinned well-powered leave-one-order-out unit (conf/data/splits.yaml
# ``min_heldout_positives: 20``) and PRD §12 ("31 orders with ≥20 positives"; the
# ~20-positive floor is statistically unstable). Blinded-frozen at P0 — a change
# needs ADR-0005 re-sign-off (CLAUDE.md §7 item 2).
MIN_REAL_HOMOLOG_N = 20

# ADR-0005 D1 fixed, pre-registered identity-bin edges (percent identity to the
# nearest training member). Fixed for the life of the project — never quantile.
IDENTITY_BIN_EDGES_PCT = (50, 70, 85)
IDENTITY_BINS = ("<50", "50-70", "70-85", "85-100")
NO_NEIGHBOUR = "no_neighbour"  # -1 sentinel: no coverage-adequate training neighbour
# The "lowest-identity bins" (ADR-0005 D1/D6 arm 1) = the bottom two (< 70 %).
LOW_IDENTITY_BINS = ("<50", "50-70")

FIRMICUTES = "Firmicutes"

# Reused split-construction constants (single source of truth = splits.py).
COVERAGE_CUT = splits.COVERAGE_CUT
CLASS_I = splits.CLASS_I
CLASS_II = splits.CLASS_II
HOLDOUT_PHYLUM = splits.HOLDOUT_PHYLUM  # Actinobacteria (D9 within-phylum control)

INTERIM_TABLE = splits.INTERIM_TABLE
ALIGNED_DIR = splits.ALIGNED_DIR
AUDIT_DIR = splits.AUDIT_DIR
ANCHOR_REPORT = "data/external/gate1_anchor/gate1_anchor_report.json"
CLASSII_REPORT = "data/external/classII_positives/classII_report.json"
POWER_REPORT = f"{AUDIT_DIR}/power_budget_report.json"
POWER_FIGURE_DATA = f"{AUDIT_DIR}/power_figure_data.json"
FIGURES_DIR = "figures/power"


# --------------------------------------------------------------------------- #
# Pure predicates (stdlib only — unit-testable without numpy/pandas)
# --------------------------------------------------------------------------- #


def bin_identity(identity: float, edges=IDENTITY_BIN_EDGES_PCT) -> str:
    """Bin a fractional identity into the ADR-0005 D1 label (left-closed bins).

    ``identity`` is a fraction in ``[0, 1]``; ``edges`` are percent cut points.
    A negative value is the "no coverage-adequate training neighbour" sentinel
    (more divergent than any measurable bin — its own category).
    """
    if identity < 0:
        return NO_NEIGHBOUR
    pct = identity * 100.0
    lo, mid, hi = edges
    if pct < lo:
        return IDENTITY_BINS[0]  # <50
    if pct < mid:
        return IDENTITY_BINS[1]  # 50-70
    if pct < hi:
        return IDENTITY_BINS[2]  # 70-85
    return IDENTITY_BINS[3]  # 85-100


def bin_counts(identities, edges=IDENTITY_BIN_EDGES_PCT) -> dict[str, int]:
    """Count identities per D1 bin (all bins present, zeros included)."""
    counts = {b: 0 for b in IDENTITY_BINS}
    counts[NO_NEIGHBOUR] = 0
    for x in identities:
        counts[bin_identity(x, edges)] += 1
    return counts


def low_identity_count(counts: dict[str, int]) -> dict[str, int]:
    """Split the ``< 70 %`` (low-identity) region into measurable vs no-neighbour.

    The bottom two bins are the ADR-0005 D1/D6 lowest-identity region; the
    no-neighbour sentinels are even more divergent (no adequate training homolog).
    """
    measurable = sum(counts.get(b, 0) for b in LOW_IDENTITY_BINS)
    no_neighbour = counts.get(NO_NEIGHBOUR, 0)
    return {
        "measurable_below_70": measurable,
        "no_coverage_adequate_neighbour": no_neighbour,
        "total_low_identity_region": measurable + no_neighbour,
    }


def reachability(n: int, min_n: int = MIN_REAL_HOMOLOG_N) -> str:
    """``"powered"`` iff ``n >= min_n`` else ``"reported-not-gated"`` (ADR-0005 D6/D8)."""
    return "powered" if n >= min_n else "reported-not-gated"


def arm_verdict(n: int, *, min_n: int = MIN_REAL_HOMOLOG_N, n_blocks: int | None = None) -> dict:
    """Reachability verdict for one arm/cell: N vs min-N + optional block count.

    ``n_blocks`` is the number of independent resampling blocks (held-out orders /
    clusters) contributing to the arm — ADR-0005 D5 resamples at the block level,
    so a single-block arm cannot be block-bootstrapped (reported, not a second
    pinned floor).
    """
    v = {
        "n": int(n),
        "min_n": int(min_n),
        "powered": bool(n >= min_n),
        "status": reachability(n, min_n),
    }
    if n_blocks is not None:
        v["n_blocks"] = int(n_blocks)
        v["block_resamplable"] = bool(n_blocks >= 2)
    return v


# --------------------------------------------------------------------------- #
# Heavy audit (numpy/pandas lazy; reuses splits helpers for the identity metric)
# --------------------------------------------------------------------------- #


def compute_heldout_identities(
    split_df, aligned_dir: str | Path = ALIGNED_DIR, coverage_cut: float = COVERAGE_CUT
) -> dict[str, float]:
    """Nearest nested-training-fold identity per held-out record (per class).

    Reuses :func:`splits.consensus_matrix` + :func:`splits._nearest_cross_fold_identity`
    so the metric is identical to the ADR-0004 D2 histogram. Identity is computed
    **within each class alignment** (RF00230 vs TBDB001 consensus space; identity
    is undefined across the two CMs). Returns ``{seq_name: identity}`` for every
    ``nested_role == 'heldout'`` record (``-1.0`` = no coverage-adequate neighbour).
    """
    import numpy as np

    aligned_dir = Path(aligned_dir)
    role = dict(zip(split_df["seq_name"], split_df["nested_role"], strict=True))
    out: dict[str, float] = {}
    for klass in (CLASS_I, CLASS_II):
        names, codes = splits.consensus_matrix(aligned_dir / f"class_{klass}.sto")
        if not names:
            continue
        roles = np.array([role.get(n, "") for n in names])
        is_train = roles == "train"
        is_test = roles == "heldout"
        if not is_test.any():
            continue
        identity = splits._nearest_cross_fold_identity(codes, is_train, is_test, coverage_cut)
        test_names = [n for n, t in zip(names, is_test, strict=True) if t]
        for n, x in zip(test_names, identity, strict=True):
            out[n] = float(x)
    return out


def _read_counts(path: str | Path, keys: dict[str, str]) -> dict:
    """Read named integer/dict fields from a report JSON by dotted key path.

    ``keys`` maps output-name → dotted path (e.g. ``"leakage_report.n_coordinate_novel"``).
    A missing path raises (no silent 0 / no fabrication — CLAUDE.md §10.3).
    """
    doc = json.loads(Path(path).read_text())
    out = {}
    for name, dotted in keys.items():
        node = doc
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"{path}: missing key path '{dotted}'")
            node = node[part]
        out[name] = node
    return out


def anchor_counts(path: str | Path = ANCHOR_REPORT) -> dict:
    """P0-16 GATE-1 literature-anchor raw counts (arm c, ADR-0005 D6/D7)."""
    return _read_counts(
        path,
        {
            "raw_record_count": "raw_record_count",
            "counts_by_gtdb_phylum": "counts_by_gtdb_phylum",
            "n_corpus_coord_overlap": "leakage_report.n_corpus_coord_overlap",
            "n_coordinate_novel": "leakage_report.n_coordinate_novel",
        },
    )


def classII_counts(path: str | Path = CLASSII_REPORT) -> dict:
    """P0-17 additional independent non-Actinobacteria class-II raw counts (D9)."""
    return _read_counts(
        path,
        {
            "raw_positive_count": "raw_positive_count",
            "leads_count": "leads.count",
            "n_added_to_no_leakage_test": "leakage_report.n_added_to_no_leakage_test",
        },
    )


def build_report(
    split_df,
    identities: dict[str, float],
    anchor: dict,
    classii: dict,
    *,
    min_n: int = MIN_REAL_HOMOLOG_N,
    edges=IDENTITY_BIN_EDGES_PCT,
) -> dict:
    """Assemble the power-budget report (pure over a DataFrame + count dicts).

    Renders: per-%-identity-bin N of real held-out positives (pooled, per class,
    per leave-one-order-out round); the low-identity-homolog count; per-headline-arm
    reachability (bottom-two bins pooled, non-Firmicutes-order subgroup, arm-(c)
    literature anchor, class-II anti-mimicry); and the gated-vs-reported-not-gated
    determinations incl. the arm-(c) sourcing-fallback trigger.
    """
    heldout = split_df[split_df["nested_role"] == "heldout"].copy()
    heldout["identity"] = [identities.get(n, float("nan")) for n in heldout["seq_name"]]
    # Every held-out record must have a computed identity (else an alignment/join gap).
    missing = [n for n in heldout["seq_name"] if n not in identities]
    if missing:
        raise ValueError(
            f"{len(missing)} held-out records have no computed identity "
            f"(alignment/join gap); first few: {missing[:5]}"
        )
    ident = list(heldout["identity"])

    pooled = bin_counts(ident, edges)
    by_class = {
        klass: bin_counts(list(grp["identity"]), edges) for klass, grp in heldout.groupby("klass")
    }

    # Per leave-one-order-out round (ADR-0005 D8: "per round AND pooled").
    per_order = {}
    for order, grp in heldout.groupby("loo_order_unit"):
        key = "(unresolved-order)" if order is None or order == "" else str(order)
        counts = bin_counts(list(grp["identity"]), edges)
        phyla = Counter(p for p in grp["resolved_phylum"] if p)
        per_order[key] = {
            "n_heldout": int(len(grp)),
            "bins": counts,
            "phylum": phyla.most_common(1)[0][0] if phyla else "",
            "verdict": arm_verdict(len(grp), min_n=min_n),
        }
    n_orders_ge_min = sum(1 for v in per_order.values() if v["n_heldout"] >= min_n)

    # --- Headline-certifying arms (ADR-0005 D6) --------------------------------
    n_heldout = int(len(heldout))
    # Arm 1: lowest-identity bins pooled = bottom two + no-neighbour (all held-out,
    # since the leave-clade-out cut forces every held-out identity < 0.70).
    bottom_two = pooled[LOW_IDENTITY_BINS[0]] + pooled[LOW_IDENTITY_BINS[1]] + pooled[NO_NEIGHBOUR]
    heldout_orders = {o for o in heldout["loo_order_unit"] if o is not None and o != ""}
    upper_bins = pooled[IDENTITY_BINS[2]] + pooled[IDENTITY_BINS[3]]  # 70-85 + 85-100

    # Arm 2: non-Firmicutes-order subgroup (corpus held-out with a resolved
    # non-Firmicutes phylum) — the certified beyond-Firmicutes result.
    corpus_heldout = heldout[heldout["source"] == "corpus"]
    non_firm = corpus_heldout[
        (corpus_heldout["resolved_phylum"] != FIRMICUTES)
        & (corpus_heldout["resolved_phylum"].notna())
        & (corpus_heldout["resolved_phylum"] != "")
    ]
    non_firm_phyla = Counter(non_firm["resolved_phylum"])
    non_firm_orders = {o for o in non_firm["loo_order_unit"] if o is not None and o != ""}

    # Arm 3: class-II anti-mimicry (D9). Total held-out class-II is
    # Actinobacteria-heavy (single-phylum → within-phylum-memorization confound),
    # so the *independent-of-Actinobacteria* natural class-II is the scarce
    # evidence: P0-17 (0) + the non-Actinobacteria fraction of any blind set.
    heldout_classII = heldout[heldout["klass"] == CLASS_II].copy()
    # Normalize the phylum (None / NaN / "" → "(unresolved)") so an unresolved phylum
    # is never miscounted as independent non-Actinobacteria evidence and the Counter
    # keys stay JSON-sortable (a NaN float key breaks json.dumps(sort_keys=True)).
    heldout_classII["phylum_norm"] = [
        p if isinstance(p, str) and p else "(unresolved)"
        for p in heldout_classII["resolved_phylum"]
    ]
    blind_classII = heldout_classII[heldout_classII["source"] == "blind"]
    # A blind class-II counts as independent non-Actinobacteria evidence only when its
    # phylum is *resolved* and not Actinobacteria (an unresolved phylum cannot be
    # asserted non-Actino — conservative, no over-counting of independent evidence).
    blind_non_actino = blind_classII[
        (blind_classII["phylum_norm"] != HOLDOUT_PHYLUM)
        & (blind_classII["phylum_norm"] != "(unresolved)")
    ]
    independent_non_actino = int(classii["raw_positive_count"]) + int(len(blind_non_actino))

    arms = {
        "lowest_identity_bins_pooled": {
            **arm_verdict(bottom_two, min_n=min_n, n_blocks=len(heldout_orders)),
            "note": (
                "bottom-two D1 bins + no-neighbour sentinels; every leave-clade-out "
                "held-out positive falls here (max real identity 0.699 < 0.70 by "
                "whole-cluster holdout), so this arm = all held-out positives"
            ),
        },
        "upper_identity_bins_70_100": {
            "n": int(upper_bins),
            "powered": False,
            "status": "structurally-empty",
            "note": (
                "the 70-85 and 85-100 D1 bins are provably empty for the "
                "leave-clade-out arm (the split's homology cut caps held-out↔train "
                "identity below 0.70); these cells are reachable only by the D8 "
                "synthetic-divergence arm (built P2/P4) and the §13 campaign"
            ),
        },
        "non_firmicutes_order_subgroup": {
            **arm_verdict(int(len(non_firm)), min_n=min_n, n_blocks=len(non_firm_orders)),
            "phyla": dict(non_firm_phyla),
            "note": (
                "corpus held-out positives with a resolved non-Firmicutes phylum "
                "(incl. the Actinobacteria phylum holdout); the beyond-Firmicutes "
                "certification the D6 floor requires alongside arm (1)"
            ),
        },
        "arm_c_literature_anchor": {
            **arm_verdict(int(anchor["raw_record_count"]), min_n=min_n),
            "source": "P0-16 gate1_anchor (Vitreschak-2008 non-Firmicutes re-derivation)",
            "counts_by_phylum": anchor["counts_by_gtdb_phylum"],
            "n_coordinate_novel": int(anchor["n_coordinate_novel"]),
            "n_corpus_coord_overlap": int(anchor["n_corpus_coord_overlap"]),
        },
        "classII_anti_mimicry": {
            "heldout_classII_total": int(len(heldout_classII)),
            "heldout_classII_by_phylum": dict(Counter(heldout_classII["phylum_norm"])),
            "independent_non_actinobacteria": arm_verdict(independent_non_actino, min_n=min_n),
            "independent_sources": {
                "P0-17_additional_non_actino": int(classii["raw_positive_count"]),
                "blind_non_actino": int(len(blind_non_actino)),
                "blind_actino_single_phylum": int(len(blind_classII) - len(blind_non_actino)),
            },
            "note": (
                "total held-out class-II is Actinobacteria-dominated (single-phylum → "
                "within-Actinobacteria memorization confound, D9); the "
                "independent-of-Actinobacteria natural class-II is the scarce evidence"
            ),
        },
    }

    # --- Determinations (ADR-0005 D6/D7/D8/D9) --------------------------------
    arm_c_below = int(anchor["raw_record_count"]) < min_n
    classII_indep_below = independent_non_actino < min_n
    determinations = {
        "min_real_homolog_n_pinned": int(min_n),
        "lowest_identity_arm": reachability(bottom_two, min_n),
        "non_firmicutes_order_subgroup": reachability(int(len(non_firm)), min_n),
        "arm_c_literature_anchor_below_min_n": bool(arm_c_below),
        "arm_c_sourcing_fallback_trigger": bool(arm_c_below),  # ADR-0005 D7 → §7 stop-and-ask
        "classII_independent_natural_below_min_n": bool(classII_indep_below),
        "beyond_firmicutes_certification_rests_on": (
            "non_firmicutes_order_subgroup"
            if int(len(non_firm)) >= min_n
            else (
                "arm_c_literature_anchor"
                if not arm_c_below
                else "UNMET — both the non-Firmicutes-order subgroup and arm (c) below "
                "min-N (§7 stop-and-ask)"
            )
        ),
        "synthetic_divergence_arm": (
            "pre-registered precondition for the §1 CM-invisibility claim (ADR-0005 D8); "
            "the real leave-clade-out bins reach only to 0.699 identity, so the deep-"
            "divergence (<40%) cells are supplied only by the P2/P4 synthetic arm"
        ),
        "classII_anti_mimicry_rests_on": (
            "the P2 construction-powered synthetic-class-II recovery set (above min-N by "
            "construction, ADR-0005 D9) + the within-phylum-homology control; the natural "
            "independent-of-Actinobacteria class-II is below min-N"
            if classII_indep_below
            else "the natural independent-of-Actinobacteria class-II evidence (>= min-N) "
            "+ the D9 within-phylum-homology control"
        ),
    }

    return {
        "min_real_homolog_n": int(min_n),
        "min_n_rationale": (
            "20 real held-out positives per cell (identity bin or headline arm). Below "
            "this the D4 strong bar's block-resampled CI cannot resolve its +5 pp floor "
            "(1/N >= 5 pp at N < 20) so the cell is reported-not-gated and the D4 "
            "weak-clause fallback (CI lower > 0) is admissible. Coheres with the repo's "
            "well-powered leave-one-order-out unit (conf/data/splits.yaml "
            "min_heldout_positives=20) and PRD §12 (31 orders with >=20 positives; the "
            "~20-positive floor is statistically unstable). Blinded-frozen at P0."
        ),
        "identity_bin_edges_pct": list(edges),
        "identity_bins": list(IDENTITY_BINS),
        "coverage_cut": COVERAGE_CUT,
        "n_heldout_total": n_heldout,
        "heldout_identity_bins": {
            "pooled": pooled,
            "by_class": by_class,
        },
        "low_identity_homolog_count": low_identity_count(pooled),
        "per_leave_one_order_out_round": per_order,
        "per_order_summary": {
            "n_orders_with_heldout": len(per_order),
            "n_orders_ge_min_n": n_orders_ge_min,
            "min_n": int(min_n),
        },
        "headline_arms": arms,
        "determinations": determinations,
    }


def run_audit(
    *,
    table: str | Path = INTERIM_TABLE,
    aligned_dir: str | Path = ALIGNED_DIR,
    anchor_report: str | Path = ANCHOR_REPORT,
    classII_report: str | Path = CLASSII_REPORT,
    out_report: str | Path = POWER_REPORT,
    figure_data: str | Path = POWER_FIGURE_DATA,
    env_lock: str | Path | None = None,
) -> int:
    """Read the split partition, compute the audit, write report + figure data + provenance.

    The audit artifact is always produced at the **ADR-0005-pinned**
    ``MIN_REAL_HOMOLOG_N`` — there is no runtime override, so no committed report can
    contradict the frozen threshold (a recalibration goes through ADR re-sign-off, not
    a CLI flag). The ``min_n`` parameter is kept only on the pure ``build_report`` /
    predicate layer for unit testing.
    """
    import pandas as pd

    min_n = MIN_REAL_HOMOLOG_N
    split_df = pd.read_parquet(table)
    identities = compute_heldout_identities(split_df, aligned_dir)
    anchor = anchor_counts(anchor_report)
    classii = classII_counts(classII_report)
    report = build_report(split_df, identities, anchor, classii, min_n=min_n)

    out_report = Path(out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    # Numeric figure data → rendered to PNG by the viz-env ``plot-figures`` step
    # (data env has no matplotlib; figures are not on any leakage critical path).
    fig = {
        "identity_bins": list(IDENTITY_BINS) + [NO_NEIGHBOUR],
        "pooled_bin_counts": [report["heldout_identity_bins"]["pooled"][b] for b in IDENTITY_BINS]
        + [report["heldout_identity_bins"]["pooled"][NO_NEIGHBOUR]],
        "per_order_n": sorted(
            (v["n_heldout"] for v in report["per_leave_one_order_out_round"].values()),
            reverse=True,
        ),
        "min_n": int(min_n),
        "arms": {
            k: (v.get("n") if "n" in v else v.get("independent_non_actinobacteria", {}).get("n"))
            for k, v in report["headline_arms"].items()
        },
    }
    figure_data = Path(figure_data)
    figure_data.write_text(json.dumps(fig) + "\n")

    provenance.write_provenance(
        out_report.with_suffix(".provenance.json"),
        rule="workflow/rules/data.smk :: power_budget_audit",
        script="src/tbox_finder/power.py",
        seed=provenance.DEFAULT_SEED,
        inputs=[
            table,
            Path(aligned_dir) / "class_I.sto",
            Path(aligned_dir) / "class_II.sto",
            anchor_report,
            classII_report,
        ],
        outputs=[out_report, figure_data],
        env_lock=env_lock,
        adr="ADR-0005",
        extra={
            "min_real_homolog_n": int(min_n),
            "identity_bin_edges_pct": list(IDENTITY_BIN_EDGES_PCT),
        },
    )
    d = report["determinations"]
    print(
        f"[power.audit] held-out={report['n_heldout_total']} | "
        f"lowest-identity arm={d['lowest_identity_arm']} | "
        f"non-Firmicutes subgroup={d['non_firmicutes_order_subgroup']} | "
        f"arm-c anchor N={report['headline_arms']['arm_c_literature_anchor']['n']} "
        f"(<{min_n}: {d['arm_c_sourcing_fallback_trigger']}) | "
        f"classII independent N="
        f"{report['headline_arms']['classII_anti_mimicry']['independent_non_actinobacteria']['n']}"
    )
    return 0


def plot_figures(
    *, figure_data: str | Path = POWER_FIGURE_DATA, out_dir: str | Path = FIGURES_DIR
) -> int:
    """Render the audit figures from the numeric figure-data JSON (viz env)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = json.loads(Path(figure_data).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_n = fig["min_n"]

    # (1) Per-identity-bin held-out N — the collapse into the bottom two bins.
    f1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(fig["identity_bins"], fig["pooled_bin_counts"], color="#4477aa")
    ax1.axhline(min_n, color="#cc3311", ls="--", lw=1, label=f"min-N = {min_n}")
    ax1.set_yscale("symlog")
    ax1.set_ylabel("held-out positives (symlog)")
    ax1.set_xlabel("% identity to nearest training member (ADR-0005 D1)")
    ax1.set_title("GATE-1 held-out N per identity bin")
    ax1.legend()
    f1.tight_layout()
    f1.savefig(out_dir / "identity_bin_counts.png", dpi=150)
    plt.close(f1)

    # (2) Per-leave-one-order-out-round held-out N (sorted), min-N line.
    per_order = fig["per_order_n"]
    f2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(range(len(per_order)), per_order, color="#66ccee")
    ax2.axhline(min_n, color="#cc3311", ls="--", lw=1, label=f"min-N = {min_n}")
    ax2.set_yscale("symlog")
    ax2.set_ylabel("held-out positives (symlog)")
    ax2.set_xlabel("leave-one-order-out round (sorted)")
    n_ge = sum(1 for x in per_order if x >= min_n)
    ax2.set_title(f"Per-order held-out N ({n_ge}/{len(per_order)} orders ≥ min-N)")
    ax2.legend()
    f2.tight_layout()
    f2.savefig(out_dir / "per_order_counts.png", dpi=150)
    plt.close(f2)

    # (3) Headline-arm reachability vs min-N.
    arms = {k: (v if v is not None else 0) for k, v in fig["arms"].items()}
    f3, ax3 = plt.subplots(figsize=(7, 4))
    labels = list(arms.keys())
    vals = [arms[k] for k in labels]
    colours = ["#228833" if v >= min_n else "#ee6677" for v in vals]
    ax3.barh(range(len(labels)), [max(v, 0.5) for v in vals], color=colours)
    ax3.axvline(min_n, color="#cc3311", ls="--", lw=1, label=f"min-N = {min_n}")
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=7)
    ax3.set_xscale("symlog")
    ax3.set_xlabel("N real positives (symlog)")
    ax3.set_title("Headline-arm reachability (green ≥ min-N)")
    ax3.legend()
    f3.tight_layout()
    f3.savefig(out_dir / "arm_reachability.png", dpi=150)
    plt.close(f3)
    print(f"[power.plot] wrote 3 figures to {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="compute the GATE-1 power-budget audit (data env)")
    a.add_argument("--table", default=INTERIM_TABLE)
    a.add_argument("--aligned-dir", default=ALIGNED_DIR)
    a.add_argument("--anchor-report", default=ANCHOR_REPORT)
    a.add_argument("--classII-report", default=CLASSII_REPORT)
    a.add_argument("--out-report", default=POWER_REPORT)
    a.add_argument("--figure-data", default=POWER_FIGURE_DATA)
    a.add_argument("--env-lock", default=None)

    p = sub.add_parser("plot-figures", help="render the audit figures (viz env)")
    p.add_argument("--figure-data", default=POWER_FIGURE_DATA)
    p.add_argument("--out-dir", default=FIGURES_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "audit":
        return run_audit(
            table=args.table,
            aligned_dir=args.aligned_dir,
            anchor_report=args.anchor_report,
            classII_report=args.classII_report,
            out_report=args.out_report,
            figure_data=args.figure_data,
            env_lock=args.env_lock,
        )
    if args.cmd == "plot-figures":
        return plot_figures(figure_data=args.figure_data, out_dir=args.out_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
