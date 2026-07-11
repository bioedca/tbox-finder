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
import math
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

# --------------------------------------------------------------------------- #
# Blinded-frozen gate-default magnitude rationales (P0-28; ADR-0005 D18)
# --------------------------------------------------------------------------- #
#
# ADR-0005 D18 delegates the *authored magnitude-rationale text* for each
# blinded-frozen gate default to this step. The default *values* are pinned as
# constants here (single source of truth) so the rationale text and the ADR agree
# byte-for-byte — a unit test asserts the match. Each value is a **default** under
# the PRD §2.3 precedence carve-out and **blinded-frozen at P0**: it may not change
# after P4 unblinding, and any pre-P4 recalibration needs ADR re-sign-off
# (CLAUDE.md §7 item 2). The rationale is a smallest-effect-of-interest (SESOI) /
# power argument authored *before any P4 result*, grounded in the verified method
# citations below (CLAUDE.md §10.1) — never fit to a result.
RECALL_POINT_BAR_PP = 10  # ADR-0005 D4 — GATE-1 recall point-estimate bar
RECALL_CI_FLOOR_PP = 5  # ADR-0005 D4 — GATE-1 block-resampled CI lower floor
ECE_GATE = 0.05  # ADR-0005 D11 — in-distribution named-posterior ECE gate
FDR_GATE = 0.10  # ADR-0005 D12 — genome-scale FDP CI-upper-bound gate
DECOY_PREVALENCE = 100  # ADR-0005 D7 — benchmark decoy:positive prevalence
SWAP_ECE_MARGIN = 0.02  # ADR-0005 D17(c) — RiNALMo→RNA-FM OOD-ECE swap margin
SWAP_RECALL_MARGIN_PP = 3  # ADR-0005 D17(d) — recall@matched-precision swap margin
SWAP_AUPRC_MARGIN = 0.03  # ADR-0005 D17(d) — AUPRC swap margin
GATE4_F1_FLOOR = 0.80  # ADR-0004 D6 — GATE-4 per-nt per-element F1 floor

# Machine-traceable method citations (CLAUDE.md §10.1) grounding the SESOI / power
# arguments. Foundational method papers; every identifier verified (PubMed /
# publisher of record), not asserted from memory (CLAUDE.md §10.3).
CITATIONS = {
    "sesoi_primer": "DOI:10.1177/1948550617697177",  # Lakens 2017, SPPS 8(4):355-362
    "sesoi_tutorial": "DOI:10.1177/2515245918770963",  # Lakens, Scheel & Isager 2018, AMPPS
    "calibration_modern": "arXiv:1706.04599",  # Guo, Pleiss, Sun & Weinberger 2017, ICML
    "calibration_binning": "PMID:25927013",  # Naeini, Cooper & Hauskrecht 2015, AAAI (PMC4410090)
    "fdr_bh": "DOI:10.1111/j.2517-6161.1995.tb02031.x",  # Benjamini & Hochberg 1995, JRSS-B
    "fdr_genomewide": "DOI:10.1073/pnas.1530509100",  # Storey & Tibshirani 2003, PNAS
    "fdr_target_decoy": "PMID:17327847",  # Elias & Gygi 2007, Nat Methods (ADR-0005 D12)
    "divergence_freyhult": "DOI:10.1101/gr.5890907",  # Freyhult et al 2007 (ADR-0005 D1)
    "divergence_menzel": "DOI:10.1261/rna.1556009",  # Menzel, Gorodkin & Stadler 2009 (ADR-0005 D1)
}

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
# Gate-default magnitude rationales (P0-28; pure, stdlib math — no numpy/pandas)
# --------------------------------------------------------------------------- #
#
# These back the blinded-frozen defaults with numeric SESOI / power arguments and
# make them machine-checkable: a unit test asserts each argument is internally
# consistent (e.g. the +5 pp CI floor equals the per-positive granularity at the
# pinned min-N) and carries its verified method citations where applicable (the
# two project-internal design/reference defaults carry none). They never read a P4
# result — the arguments are effect-size / estimability logic authored ex ante.


def _require_count(value, name: str, *, allow_zero: bool = False) -> int:
    """Reject a non-integer / out-of-range count (``bool`` is not a count either).

    Counts (N of positives, candidates, min-N) must be true integers — a float
    like ``20.5`` would compute with 20.5 yet report ``int(20.5) == 20``, an
    inconsistent diagnostic (CodeRabbit P0-28 round-2).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def binomial_se(p: float, n: int) -> float:
    """Standard error of a binomial proportion estimate: ``sqrt(p(1-p)/n)``."""
    _require_count(n, "n")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    return math.sqrt(p * (1.0 - p) / n)


def recall_bar_resolution(min_n: int = MIN_REAL_HOMOLOG_N) -> dict:
    """Estimability backing for the D4 two-part recall bar at the pinned min-N.

    At ``min_n`` real held-out positives the per-positive recall granularity is
    ``1/min_n`` (in pp): a recall-difference floor finer than this cannot be
    resolved (the estimator moves in 1/N steps). The D4 CI floor (+5 pp) is set to
    exactly this granularity at ``min_n = 20``, and the point bar (+10 pp) to two
    granularities — so a passing arm's point estimate sits a full CI-floor-width
    above the +5 pp non-trivial-improvement boundary. This is the same
    ``1/N >= 5 pp`` argument that pinned ``MIN_REAL_HOMOLOG_N`` (Amendment A1), so
    the recall bar and the min-N floor are one internally-consistent construction.
    """
    _require_count(min_n, "min_n")
    granularity_pp = 100.0 / min_n
    return {
        "min_n": int(min_n),
        "per_positive_granularity_pp": granularity_pp,
        "ci_floor_pp": RECALL_CI_FLOOR_PP,
        "point_bar_pp": RECALL_POINT_BAR_PP,
        "ci_floor_matches_granularity": abs(RECALL_CI_FLOOR_PP - granularity_pp) < 1e-9,
        "point_is_two_ci_floors": RECALL_POINT_BAR_PP == 2 * RECALL_CI_FLOOR_PP,
    }


def min_detectable_effect_pp(baseline_recall: float, n: int, z: float = 1.96) -> float:
    """Illustrative minimum detectable recall gain (pp) at a baseline recall & N.

    A normal-approximation ``z * SE`` scaled to percentage points — used only to
    show the D4 bar sits in the detectable range for the effects the
    CM-invisibility hypothesis predicts; **not** a claim about the true effect size
    (CLAUDE.md §10.3). The gated inference is the block-resampled CI + the D5
    small-N exact/permutation test, not this normal approximation.
    """
    if not 0.0 < baseline_recall < 1.0:
        # The normal approximation is degenerate at the Bernoulli boundaries
        # (SE = 0 -> MDE = 0); reject rather than report a meaningless zero.
        raise ValueError("baseline_recall must be strictly between 0 and 1")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("z must be finite and positive")
    return z * binomial_se(baseline_recall, n) * 100.0


def expected_false_at_fdr(fdr: float, n_candidates: int) -> float:
    """Expected false discoveries in a candidate table of size N at a given FDP."""
    if not 0.0 <= fdr <= 1.0:
        raise ValueError("fdr must be in [0, 1]")
    _require_count(n_candidates, "n_candidates", allow_zero=True)
    return fdr * n_candidates


def calibration_fdr_budget_ratio(ece: float = ECE_GATE, fdr: float = FDR_GATE) -> float:
    """The ECE budget as a fraction of the FDR budget (ADR-0005 D11 vs D12).

    ECE <= 0.05 keeps the named-posterior calibration error at half the D12 FDR
    budget (0.05 / 0.10), so a GATE-2-passing calibration head cannot by itself
    exhaust the downstream FDR budget it feeds.
    """
    if not 0.0 <= ece <= 1.0:
        raise ValueError("ece must be in [0, 1]")
    if not 0.0 < fdr <= 1.0:
        raise ValueError("fdr must be in (0, 1]")
    return ece / fdr


def _rationale_records() -> dict:
    """The authored magnitude rationale for every delegated blinded-frozen default.

    Keyed by a stable slug; each record carries the default, its value, the
    argument kind, the SESOI / power argument, its verified method citations
    (empty for the two project-internal design/reference defaults —
    ``decoy_prevalence`` and ``gate4_f1_floor``), and the blinded-freeze flag.
    The QMD (``analyses/gate_default_rationales.qmd``)
    renders this verbatim so the durable doc has no hand-typed numbers.
    """
    return {
        "recall_point_bar": {
            "default": "GATE-1 recall point-estimate bar (ADR-0005 D4)",
            "value": f"+{RECALL_POINT_BAR_PP} pp",
            "kind": "SESOI+power",
            "argument": (
                f"The smallest recall gain over cmsearch at matched precision "
                f"that is both scientifically material and robustly separated "
                f"from sampling noise at the pinned min-N (20): +{RECALL_POINT_BAR_PP} pp "
                f"is two per-positive granularities (2 x 1/20), so a passing arm's "
                f"point estimate sits a full CI-floor-width above the "
                f"+{RECALL_CI_FLOOR_PP} pp non-trivial-improvement boundary. In the "
                f"bottom-two identity bins where sequence-based homology search is "
                f"documented to lose sensitivity, +{RECALL_POINT_BAR_PP} pp means "
                f"recovering >=1-in-10 more divergent T-boxes than the CM — the "
                f"smallest gain that materially expands the recoverable set. Read "
                f"as an equivalence-style lower bound (SESOI method)."
            ),
            "citations": [
                CITATIONS["sesoi_primer"],
                CITATIONS["sesoi_tutorial"],
                CITATIONS["divergence_freyhult"],
                CITATIONS["divergence_menzel"],
            ],
            "blinded_frozen": True,
        },
        "recall_ci_floor": {
            "default": "GATE-1 recall CI-lower-bound floor (ADR-0005 D4)",
            "value": f"+{RECALL_CI_FLOOR_PP} pp",
            "kind": "power",
            "argument": (
                f"The finest recall-difference lower bound the block-resampled CI "
                f"can resolve at the pinned min-N: 1/20 = {100.0 / MIN_REAL_HOMOLOG_N:.0f} pp "
                f"per held-out positive. A floor below this is below the estimator's "
                f"granularity, so +{RECALL_CI_FLOOR_PP} pp is the smallest positive "
                f"floor that is both estimable at the sample size the corpus supplies "
                f"and a non-trivial improvement (>=1-in-20 additional divergent "
                f"homologs). This is exactly the 1/N >= 5 pp argument that pinned "
                f"MIN_REAL_HOMOLOG_N (Amendment A1) — the bar and the min-N floor are "
                f"one construction. Below min-N the D4 weak clause (CI lower > 0) is "
                f"the disclosed fallback, never a convenience loosening."
            ),
            "citations": [CITATIONS["sesoi_primer"], CITATIONS["sesoi_tutorial"]],
            "blinded_frozen": True,
        },
        "ece_gate": {
            "default": "GATE-2 in-distribution named-posterior ECE (ADR-0005 D11)",
            "value": f"<= {ECE_GATE}",
            "kind": "SESOI+power",
            "argument": (
                f"A <= {ECE_GATE} average confidence-accuracy gap on the "
                f"temperature-scaled named posterior — a machinery-failure budget, "
                f"not a deployment claim. Two backings: (i) downstream-budget — the "
                f"calibrated posterior feeds the D12 FDP <= {FDR_GATE} operating "
                f"point, and {ECE_GATE} keeps calibration error at "
                f"{calibration_fdr_budget_ratio():.0%} of the FDR budget, so a "
                f"passing calibration head cannot by itself exhaust it; (ii) "
                f"estimator-resolution — with 15 equal-mass debiased bins the binned "
                f"ECE estimator resolves {ECE_GATE} above its own noise floor at "
                f"realistic calibration-set sizes, yet {ECE_GATE} is tight enough to "
                f"catch a broken head. Post-hoc temperature scaling routinely brings "
                f"deep models into this range, so it is achievable-yet-meaningful."
            ),
            "citations": [CITATIONS["calibration_modern"], CITATIONS["calibration_binning"]],
            "blinded_frozen": True,
        },
        "fdr_gate": {
            "default": "GATE-2 genome-scale FDP CI-upper bound (ADR-0005 D12)",
            "value": f"<= {FDR_GATE}",
            "kind": "SESOI",
            "argument": (
                f"A discovery-stage false-discovery proportion on the "
                f"pre-orthogonal candidate table — not a terminal error rate. The "
                f"candidates pass through the ADR-0006 orthogonal Tier-1/2/2N "
                f"validation (the terminal false-positive control), so the right "
                f"error rate here is a discovery FDR: tolerate more false positives "
                f"now, catch them downstream. FDR/q-value is the established error "
                f"rate for genome-wide discovery, explicitly 'a more liberal "
                f"criterion' than terminal thresholds; {FDR_GATE:.0%} is the SESOI "
                f"on the candidate table — tight enough that orthogonal validation "
                f"is not swamped (E[false] = {FDR_GATE} x table size, reported as a "
                f"density), loose enough to retain the divergent-loci recall that is "
                f"the model's whole advantage over the CM. Gated on the CI upper "
                f"bound so sampling error cannot sneak a >{FDR_GATE:.0%} table through."
            ),
            "citations": [
                CITATIONS["fdr_bh"],
                CITATIONS["fdr_genomewide"],
                CITATIONS["fdr_target_decoy"],
            ],
            "blinded_frozen": True,
        },
        "decoy_prevalence": {
            "default": "GATE-1 benchmark decoy:positive prevalence (ADR-0005 D7)",
            "value": f"{DECOY_PREVALENCE}:1",
            "kind": "design",
            "argument": (
                f"{DECOY_PREVALENCE}:1 sits ~1 decade above the ~10:1 training seed "
                f"ratio (§9.1) and 1-2 decades below the ~10^3-10^4:1 genome-scale "
                f"prevalence — a benchmark operating point, not a fairness choice. "
                f"Because the comparison is matched-precision (model recall read "
                f"where model precision equals cmsearch precision on the identical "
                f"negative pool), prevalence sets only the operating point; the "
                f"+10 pp gap is additionally reported as a full 10:1 -> 10^4:1 "
                f"prevalence-sensitivity sweep so robustness is visible. Not a "
                f"SESOI/power quantity; documented here per the D18 delegation."
            ),
            "citations": [],
            "blinded_frozen": True,
        },
        "swap_ece_margin": {
            "default": "RiNALMo->RNA-FM OOD-ECE swap margin (ADR-0005 D17c)",
            "value": f"> {SWAP_ECE_MARGIN} absolute ECE",
            "kind": "SESOI",
            "argument": (
                f"A backbone swap fires only if RiNALMo's post-calibration "
                f"leave-clade-out ECE exceeds RNA-FM's by > {SWAP_ECE_MARGIN}, "
                f"sustained across the held-out-order distribution (block-resampled). "
                f"{SWAP_ECE_MARGIN} is {SWAP_ECE_MARGIN / ECE_GATE:.0%} of the D11 "
                f"in-distribution ECE gate — a backbone difference that is a "
                f"meaningful fraction of the calibration budget, not estimator noise, "
                f"and 'sustained' guards against a single-order fluctuation."
            ),
            "citations": [CITATIONS["sesoi_primer"], CITATIONS["calibration_modern"]],
            "blinded_frozen": True,
        },
        "swap_recall_margin": {
            "default": "RiNALMo->RNA-FM recall@matched-precision swap margin (ADR-0005 D17d)",
            "value": f"> {SWAP_RECALL_MARGIN_PP} pp",
            "kind": "SESOI",
            "argument": (
                f"Swap if RiNALMo transfers worse on leave-clade-out "
                f"recall@matched-precision by > {SWAP_RECALL_MARGIN_PP} pp, required "
                f"*sustained across the held-out-order distribution* — a consistent "
                f"transfer deficit, not a single-arm point difference (which the "
                f"+5 pp CI floor already governs). A persistent "
                f">{SWAP_RECALL_MARGIN_PP} pp backbone gap is a material fraction of "
                f"the D4 headline margin — {100 * SWAP_RECALL_MARGIN_PP // RECALL_POINT_BAR_PP}% "
                f"of the +{RECALL_POINT_BAR_PP} pp point bar "
                f"({100 * SWAP_RECALL_MARGIN_PP // RECALL_CI_FLOOR_PP}% of the "
                f"+{RECALL_CI_FLOOR_PP} pp CI floor) — so it is the smallest sustained "
                f"deficit worth switching backbones over."
            ),
            "citations": [CITATIONS["sesoi_primer"], CITATIONS["sesoi_tutorial"]],
            "blinded_frozen": True,
        },
        "swap_auprc_margin": {
            "default": "RiNALMo->RNA-FM AUPRC swap margin (ADR-0005 D17d)",
            "value": f"> {SWAP_AUPRC_MARGIN} AUPRC",
            "kind": "SESOI",
            "argument": (
                f"A threshold-free ranking-quality deficit of comparable magnitude "
                f"to the recall margin: swap if RiNALMo's leave-clade-out AUPRC is "
                f"worse by > {SWAP_AUPRC_MARGIN} even if ECE matches, so a backbone "
                f"that ranks divergent positives materially worse is caught "
                f"independent of any single operating point."
            ),
            "citations": [CITATIONS["sesoi_primer"], CITATIONS["sesoi_tutorial"]],
            "blinded_frozen": True,
        },
        "gate4_f1_floor": {
            "default": "GATE-4 per-nt per-element F1 floor (ADR-0004 D6)",
            "value": f">= {GATE4_F1_FLOOR}",
            "kind": "SESOI",
            "argument": (
                "The smallest per-nucleotide F1 on the 3 core elements that (i) "
                "demonstrates the segmenter has learned element *extents* (not "
                "merely background-vs-foreground), and (ii) supports the downstream "
                "§13.1 locus construction and §13.3(d) sequence-read specifier the "
                "discovery pipeline depends on — while (iii) tolerating the ~1-2 nt "
                "boundary ambiguity intrinsic to projecting TBDB dot-bracket "
                "annotations onto individual nucleotides. It is a *reference* gate "
                "on the in-distribution split; the N<=9 PDB cross-source label-noise "
                "ceiling C (P0-21, reported non-gated, no CI) must not "
                "one-directionally lower it. Co-authored into ADR-0004 D6."
            ),
            "citations": [],
            "blinded_frozen": True,
        },
    }


def magnitude_rationale(key: str) -> dict:
    """Return the authored magnitude rationale for one delegated gate default.

    Keys: ``recall_point_bar``, ``recall_ci_floor``, ``ece_gate``, ``fdr_gate``,
    ``decoy_prevalence``, ``swap_ece_margin``, ``swap_recall_margin``,
    ``swap_auprc_margin``, ``gate4_f1_floor``. Raises ``KeyError`` for an unknown
    default so a caller cannot silently reference a non-existent rationale.
    """
    records = _rationale_records()
    if key not in records:
        raise KeyError(f"unknown gate default {key!r}; known: {sorted(records)}")
    return records[key]


def all_rationales() -> dict:
    """Every authored magnitude rationale, keyed by slug (ADR-0005 D18 set)."""
    return _rationale_records()


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
