"""P3-18 — the RiNALMo → RNA-FM **swap condition (c)**: a leave-clade-out ECE margin.

ADR-0005 **D17(c)** (PRD §10.2): swap the shipped Stage-2 backbone for RNA-FM *"if RiNALMo's
post-calibration leave-clade-out ECE exceeds RNA-FM's by > 0.02 (absolute ECE), sustained across
the held-out-order distribution (block-resampled)"*. The 0.02 is a blinded-frozen default
(:data:`tbox_finder.power.SWAP_ECE_MARGIN`, rationale ADR-0005 A-rationale item 6) and is read
here by attribute at call time — this module takes **no** margin argument, so there is no
override path for a caller to use.

What is compared
----------------
The two committed GATE-2 reports — the shipped RiNALMo-giga arm (P3-10) and the RNA-FM comparator
(P3-17) — each carry a per-held-out-order D13 OOD ECE for the same 30 leave-one-order-out units,
and each publishes a macro-average with a CI that resamples **orders** (``loo_order_unit``). Read
side by side those two marginal CIs overlap over almost their whole length, but that is not the
comparison D17(c) asks for: both arms were scored on the **same** rows of the **same** orders, so
the hard orders are hard for both and the marginal widths are mostly shared. The statistic here is
the paired one,

    Δ = mean over admissible held-out orders u of ( ECE_shipped(u) − ECE_comparator(u) ),

which equals the difference of the two published macros exactly (asserted, not assumed), and its
CI is the **same** order-blocked percentile bootstrap (``eval.resample.block_bootstrap``, the order
as the block) the macros themselves use — one draw of orders applied to both arms at once.

"Sustained" has no pinned operator — so this module refuses to choose one
-------------------------------------------------------------------------
D17(c) says *sustained … (block-resampled)* and pins no CI rule for it. Two readings respect both
words, and they are not the same test:

``ci_lower_above_margin``
    The order-blocked CI of Δ lies wholly above 0.02 — the *excess itself* survives resampling.
    This is the minimum-effect test against a SESOI, which is how A-rationale item 6 frames 0.02.
``point_above_margin_and_ci_lower_above_zero``
    The point Δ exceeds 0.02 and the CI excludes zero — the *direction* is sustained and the
    magnitude is read off the point.

The verdict is ``fired`` only when **every** reading fires and ``not_fired`` only when **none**
does; a disagreement is ``unadjudicated``, which is a CLAUDE.md §7 stop, never a silent pick
([[ambiguous-adr-prose-is-a-stop]]). Point-above-margin alone is recorded but is **not** a reading:
it drops the word "sustained", which is the clause A-rationale item 6 says exists to guard against
a single-order fluctuation.

What rides with the verdict
---------------------------
ADR-0002 **A15** obliges P3-18 to carry **both** confounds beside the number — the one-minor-version
``multimolecule`` difference and the two arms' different realised LR schedules (the shipped arm
under-annealed) — and not to present the margin as an apples-to-apples backbone comparison. They
are written into ``disclosures`` and a clause refuses a report that drops either.

A fired swap is a **§7 stop** (CLAUDE.md §7 item 2): the substituted RNA-FM Stage-2 would have to
re-clear the absolute GATE-1/GATE-2 thresholds and re-run the P4→P5 go/no-go (PRD §2.3, ADR-0005
D17). This module records that obligation; it never acts on it.

Inputs are bound before they are read: each GATE-2 report must pass its own validator and hash-bind
to the score sidecars it names (``gate2.sidecar_binding_problems``); the two arms' leave-clade-out
and in-distribution sidecars must describe one population row-for-row; and the CLI re-runs
``gate2.rederive_against_sidecars(select_bandwidth=True)`` on both reports and records the result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import power as PW
from tbox_finder import provenance as PROV
from tbox_finder.calib import gate2 as G2
from tbox_finder.eval import resample as RS
from tbox_finder.models import rna_backbone_registry as BR

SCHEMA_VERSION = "1"
STEP = "P3-18"
GENERATED_BY = "src/tbox_finder/calib/swap_check.py"
RULE = "workflow/rules/calibration.smk :: rnafm_swap_condition_c"
PRD = "PRD §2.3, §10.2 (condition (c) [P3])"
ADR = (
    "ADR-0005 D17(c) (0.02 ECE swap margin, blinded-frozen) + A-rationale item 6; "
    "ADR-0005 D13/A2 (the OOD ECE estimator and min-N the compared numbers come from); "
    "ADR-0002 D6 + A15 (the comparator and the two confounds it carries)"
)
ENV_LOCK = "envs/ml-rna.conda-lock.yml"

DEFAULT_SHIPPED_REPORT = G2.DEFAULT_REPORT
DEFAULT_COMPARATOR_REPORT = "reports/p3/gate2_rnafm_ece.json"
DEFAULT_REPORT = "reports/rnafm_swap_condition_c.json"

#: The order-level block — the same column both GATE-2 macros resample on.
UNIT_KEY = G2.OOD_UNIT_KEY
#: The shared resampler's default. The statistic is a mean of ~30 numbers, so the 200-replicate
#: D13 default (chosen because the *within-order* kernel bootstrap is expensive) has no reason to
#: apply here; fewer replicates than this is refused as a cost knob ([[cost-knobs-can-certify]]).
N_BOOT = RS.DEFAULT_N_BOOT
#: GATE-2's own seed, so the order draws are reproducible against the reports they compare.
SEED = G2.BOOTSTRAP_SEED
CI_LEVEL = 0.95

#: The coherent readings of D17(c)'s "sustained … (block-resampled)". See the module docstring.
READINGS: dict[str, str] = {
    "ci_lower_above_margin": (
        "the order-blocked CI of the paired difference lies wholly above the margin — the excess "
        "itself survives resampling of held-out orders (a minimum-effect test against the D18 "
        "SESOI, which is how ADR-0005 A-rationale item 6 frames 0.02)"
    ),
    "point_above_margin_and_ci_lower_above_zero": (
        "the point paired difference exceeds the margin AND its order-blocked CI excludes zero — "
        "the direction is sustained under resampling, the magnitude is the point"
    ),
}

#: Marker phrases a report's disclosures must carry — one per ADR-0002 A15 confound.
A15_CONFOUND_MARKERS: tuple[str, ...] = ("multimolecule 0.1.0", "realised LR schedule")

_CLAUSES: tuple[str, ...] = (
    "margin_is_the_pinned_d17c_value",
    "arms_are_the_shipped_backbone_and_the_d6_comparator",
    "both_gate2_reports_validate",
    "both_gate2_reports_bound_to_their_sidecars",
    "inputs_rederive_from_their_sidecars",
    "arms_graded_on_one_population",
    "point_is_the_difference_of_the_published_macros",
    "ci_is_order_block_resampled",
    "bootstrap_replicates_are_not_a_cost_knob",
    "verdict_follows_every_reading",
    "a15_confounds_disclosed",
)

_ABS_TOL = 1e-12


class PairingError(ValueError):
    """The two GATE-2 reports do not describe one paired comparison."""


# --------------------------------------------------------------------------- #
# Reading the two arms
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _backbone(report: Mapping[str, Any]) -> str | None:
    scoring = report.get("scoring")
    return G2.backbone_key_from_load(scoring.get("load") if isinstance(scoring, Mapping) else None)


def _admissible_values(report: Mapping[str, Any]) -> dict[str, float]:
    units = report["ood"]["units"]
    return {name: float(u["ood_ece"]) for name, u in units.items() if u.get("admissible") is True}


def _sidecar_population(path: Path, *, fields: Sequence[str]) -> dict[str, Any]:
    payload = _load_json(path)
    return {field: payload.get(field) for field in fields}


#: The population-defining fields of each sidecar kind. Logits are per-arm and are NOT compared.
_LOO_POPULATION_FIELDS = ("row_ids", "labels", "units", "blocks", "dataset_sha256")
_IN_DIST_POPULATION_FIELDS = ("row_ids", "labels", "rungs", "dataset_sha256")
#: OOD-block settings both arms must share for their per-unit ECEs to be one estimator's output.
_SHARED_OOD_SETTINGS = ("estimator", "unit_key", "block_key", "min_n", "n_boot", "bootstrap_seed")
#: Per-unit census fields that must agree unit-for-unit.
_SHARED_UNIT_CENSUS = ("n_records", "n_positives", "n_blocks", "admissible", "phylum")


def pairing_problems(
    shipped: Mapping[str, Any],
    comparator: Mapping[str, Any],
    *,
    repo_root: str | Path = G2._REPO_ROOT,
) -> list[str]:
    """Every way the two GATE-2 reports fail to describe ONE paired comparison.

    Checked against the reports AND the sidecar bytes they name: two reports can agree on every
    census count while having been scored on different rows
    ([[control-matchedness-must-be-asserted]]).
    """
    problems: list[str] = []
    root = Path(repo_root)
    for role, rep in (("shipped", shipped), ("comparator", comparator)):
        if rep.get("ood", {}).get("truncated_to_n_units") is not None:
            problems.append(f"the {role} report graded a truncated unit list")
    so, co = shipped["ood"], comparator["ood"]
    for key in _SHARED_OOD_SETTINGS:
        if so.get(key) != co.get(key):
            problems.append(
                f"ood.{key} differs: shipped {so.get(key)!r} vs comparator {co.get(key)!r}"
            )
    su, cu = so.get("units") or {}, co.get("units") or {}
    if sorted(su) != sorted(cu):
        problems.append(
            f"the two reports grade different held-out orders: only shipped "
            f"{sorted(set(su) - set(cu))}, only comparator {sorted(set(cu) - set(su))}"
        )
    for name in sorted(set(su) & set(cu)):
        for key in _SHARED_UNIT_CENSUS:
            if su[name].get(key) != cu[name].get(key):
                problems.append(
                    f"ood.units[{name!r}].{key} differs: {su[name].get(key)!r} vs "
                    f"{cu[name].get(key)!r}"
                )
    ss, cs = shipped["scoring"], comparator["scoring"]
    for kind, key, fields in (
        ("leave-clade-out", "loo_scores", _LOO_POPULATION_FIELDS),
        ("in-distribution", "in_distribution_scores", _IN_DIST_POPULATION_FIELDS),
    ):
        try:
            a = _sidecar_population(root / str(ss[key]), fields=fields)
            b = _sidecar_population(root / str(cs[key]), fields=fields)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            problems.append(f"the {kind} score sidecars could not be read for pairing: {exc}")
            continue
        for field in fields:
            if a[field] is None or b[field] is None:
                problems.append(f"a {kind} sidecar carries no {field!r}, so pairing is unproven")
            elif a[field] != b[field]:
                problems.append(f"the {kind} sidecars disagree on {field!r} — not one population")
    return problems


def _arm_summary(report: Mapping[str, Any], path: str) -> dict[str, Any]:
    gate = report["gate"]
    ood = report["ood"]
    return {
        "backbone": _backbone(report),
        "gate2_report": path,
        "env_lock": report.get("env_lock"),
        "arm": report["scoring"].get("arm"),
        "temperature": gate.get("calibration", {}).get("temperature"),
        "in_distribution_ece": gate.get("ece"),
        "loo_macro_ece": ood["macro_average"],
        "n_units": ood["n_units"],
        "n_units_admissible": ood["n_units_admissible"],
    }


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def readings_for(point: float, lower: float, margin: float) -> dict[str, bool]:
    """Evaluate every coherent reading of D17(c) on a paired point and its CI lower bound.

    Strict inequalities throughout: D17(c) says *exceeds … by > 0.02*.
    """
    return {
        "ci_lower_above_margin": lower > margin,
        "point_above_margin_and_ci_lower_above_zero": point > margin and lower > 0.0,
    }


def verdict_from(readings: Mapping[str, bool]) -> str:
    """``fired`` iff every reading fires, ``not_fired`` iff none does, else ``unadjudicated``."""
    values = [bool(v) for v in readings.values()]
    if not values:
        return "unadjudicated"
    if all(values):
        return "fired"
    if not any(values):
        return "not_fired"
    return "unadjudicated"


def paired_difference(
    shipped_values: Mapping[str, float],
    comparator_values: Mapping[str, float],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    ci_level: float = CI_LEVEL,
) -> dict[str, Any]:
    """The paired macro difference and its order-blocked percentile CI.

    One block per held-out order, holding that order's paired difference; a replicate draws
    ``len(units)`` orders with replacement and averages them, so the SAME draw of orders is
    applied to both arms — the pairing a pair of marginal CIs throws away.
    """
    if sorted(shipped_values) != sorted(comparator_values):
        raise PairingError("the two arms' admissible held-out orders differ")
    names = sorted(shipped_values)
    if not names:
        raise PairingError("no admissible held-out order is shared by the two arms")
    deltas = [(n, float(shipped_values[n]) - float(comparator_values[n])) for n in names]
    blocks = RS.blocks_by_key(deltas, [n for n, _ in deltas], key_name=UNIT_KEY)
    # `math.fsum`, not `sum`: Python 3.12 made float `sum()` compensated, so the same mean
    # differs in its last bits between 3.11 and 3.12+ and a committed report generated on one
    # would not re-derive bit-exactly on the other. `fsum` is exactly rounded on every version.
    ci = RS.block_bootstrap(
        blocks,
        lambda sample: math.fsum(d for _, d in sample) / len(sample) if sample else float("nan"),
        n_boot=n_boot,
        seed=seed,
        ci_level=ci_level,
    )
    return {"per_unit": dict(deltas), "ci": ci}


def _by_phylum(per_unit: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in per_unit.values():
        groups.setdefault(str(row["phylum"]), []).append(float(row["delta"]))
    n_total = sum(len(v) for v in groups.values())
    out: dict[str, Any] = {}
    for phylum in sorted(groups):
        deltas = groups[phylum]
        out[phylum] = {
            "n_units": len(deltas),
            "mean_delta": math.fsum(deltas) / len(deltas),
            "contribution_to_macro_delta": math.fsum(deltas) / n_total,
        }
    return out


def disclosures(*, readings: Mapping[str, bool], verdict: str) -> list[str]:
    """The qualifications that ride with the verdict (ADR-0002 A15 obliges the first two)."""
    out = [
        "ADR-0002 A15 confound (i): the comparator ran under multimolecule 0.2.0 while the shipped "
        "RiNALMo Stage-2 ran under multimolecule 0.1.0 — one minor version; every other conda pin "
        "and the tokenisation are identical.",
        "ADR-0002 A15 confound (ii): the two arms' weights were produced under different realised "
        "LR schedules — the shipped RiNALMo arm took 1,415 of 2,830 scheduled steps and stopped at "
        "0.5501 of peak LR (the P3-06 micro-batch cosine defect), the RNA-FM arm ran 1,420 of "
        "1,420 to ~0. Annealing is not calibration-neutral, so the difference measured here is "
        "NOT an apples-to-apples backbone comparison; the confound leaves the SHIPPED arm "
        "under-annealed, which is a reason to withhold a swap rather than to fire one.",
        "D17(c)'s 'sustained across the held-out-order distribution (block-resampled)' pins no CI "
        "rule. Every coherent reading is evaluated and the verdict is taken only when they agree; "
        f"here they read {dict(readings)} -> {verdict}. A disagreement would be a CLAUDE.md §7 "
        "stop, not a pick.",
        "The paired CI resamples held-out orders with each order's ECE held at its point — the "
        "same scheme as both GATE-2 macro CIs it is compared with. Within-order (cluster-level) "
        "uncertainty is not propagated into it.",
        "Condition (c) only. Condition (d) — leave-clade-out recall@matched-precision / AUPRC — is "
        "P4's, and nothing here adjudicates it; conditions (a)/(b) were resolved not-fired at P1.",
    ]
    if verdict == "fired":
        out.append(
            "SWAP FIRED: a CLAUDE.md §7 item-2 stop. The substituted RNA-FM Stage-2 must re-clear "
            "the absolute GATE-1 (+10 pp / +5 pp) and GATE-2 (in-dist ECE <= 0.05) thresholds and "
            "re-run the P4->P5 go/no-go before the GATE-3 scan (PRD §2.3; ADR-0005 D17)."
        )
    return out


def rnafm_swap_condition_c(
    shipped: Mapping[str, Any],
    comparator: Mapping[str, Any],
    *,
    shipped_path: str = DEFAULT_SHIPPED_REPORT,
    comparator_path: str = DEFAULT_COMPARATOR_REPORT,
    repo_root: str | Path = G2._REPO_ROOT,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    rederivation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate D17(c) on the two GATE-2 reports and return the swap-check report body.

    Raises :class:`PairingError` when the two reports are not one paired comparison — a
    difference between two populations is not a measurement of either backbone.
    """
    margin = float(PW.SWAP_ECE_MARGIN)  # attribute read at call time — no override path
    problems = pairing_problems(shipped, comparator, repo_root=repo_root)
    if problems:
        raise PairingError("; ".join(problems))
    sv, cv = _admissible_values(shipped), _admissible_values(comparator)
    paired = paired_difference(sv, cv, n_boot=n_boot, seed=seed)
    ci = paired["ci"]
    point, lower = float(ci["point"]), float(ci["lower"])
    readings = readings_for(point, lower, margin)
    verdict = verdict_from(readings)

    su = shipped["ood"]["units"]
    per_unit = {
        name: {
            "shipped": sv[name],
            "comparator": cv[name],
            "delta": delta,
            "phylum": su[name].get("phylum"),
            "n_positives": su[name].get("n_positives"),
            "n_records": su[name].get("n_records"),
        }
        for name, delta in paired["per_unit"].items()
    }
    deltas = [row["delta"] for row in per_unit.values()]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "prd": PRD,
        "adr": ADR,
        "env_lock": ENV_LOCK,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "condition": "c",
        "margin": {
            "value": margin,
            "unit": "absolute ECE",
            "inequality": "strict >",
            "source": "ADR-0005 D17(c); tbox_finder.power.SWAP_ECE_MARGIN",
            "blinded_frozen": True,
        },
        "arms": {
            "shipped": _arm_summary(shipped, shipped_path),
            "comparator": _arm_summary(comparator, comparator_path),
        },
        "pairing": {
            "n_units": len(su),
            "n_units_admissible": len(per_unit),
            "shared_ood_settings": {k: shipped["ood"].get(k) for k in _SHARED_OOD_SETTINGS},
            "population_checked_against": {
                "leave_clade_out_sidecars": [
                    shipped["scoring"]["loo_scores"],
                    comparator["scoring"]["loo_scores"],
                ],
                "in_distribution_sidecars": [
                    shipped["scoring"]["in_distribution_scores"],
                    comparator["scoring"]["in_distribution_scores"],
                ],
                "fields": {
                    "leave_clade_out": list(_LOO_POPULATION_FIELDS),
                    "in_distribution": list(_IN_DIST_POPULATION_FIELDS),
                },
            },
            "problems": [],
        },
        "statistic": {
            "name": "paired macro leave-clade-out ECE difference (shipped − comparator)",
            "definition": (
                "mean over the admissible held-out orders u of (ECE_shipped(u) − "
                "ECE_comparator(u)); equal to the difference of the two published macros"
            ),
            "point": point,
            "difference_of_published_macros": float(shipped["ood"]["macro_average"]["point"])
            - float(comparator["ood"]["macro_average"]["point"]),
        },
        "resampling": {
            "block_key": UNIT_KEY,
            "method": "eval.resample.block_bootstrap — seeded percentile, orders drawn with "
            "replacement, one draw applied to both arms",
            "n_boot_requested": int(n_boot),
            "seed": int(seed),
            "ci": ci,
        },
        "readings": {
            name: {"definition": READINGS[name], "fires": bool(readings[name])} for name in READINGS
        },
        "point_exceeds_margin": point > margin,
        "point_exceeds_margin_is": (
            "recorded, NOT a reading: a point above the margin with no resampling requirement "
            "drops D17(c)'s 'sustained', the clause A-rationale item 6 says guards against a "
            "single-order fluctuation"
        ),
        "verdict": verdict,
        "swap_fired": verdict == "fired",
        "requires_user_decision": verdict != "not_fired",
        "rationale": _rationale(point, ci, margin, readings, verdict, deltas),
        "descriptive": {
            "is": "per-order context for the verdict; no field here is a decision input",
            "per_unit": per_unit,
            "n_units_delta_above_margin": sum(1 for d in deltas if d > margin),
            "n_units_delta_positive": sum(1 for d in deltas if d > 0.0),
            "n_units_comparator_worse": sum(1 for d in deltas if d < 0.0),
            "by_phylum": _by_phylum(per_unit),
            "marginal_macro_cis": {
                "shipped": shipped["ood"]["macro_average"],
                "comparator": comparator["ood"]["macro_average"],
            },
        },
        "rederivation": (
            dict(rederivation)
            if rederivation is not None
            else {"ran": False, "select_bandwidth": None, "problems": {}}
        ),
        "disclosures": disclosures(readings=readings, verdict=verdict),
    }
    report["gate2_validation"] = {
        "shipped": G2.validate_report(shipped),
        "comparator": G2.validate_report(comparator),
    }
    report["sidecar_binding"] = {
        "shipped": G2.sidecar_binding_problems(shipped, repo_root=repo_root),
        "comparator": G2.sidecar_binding_problems(comparator, repo_root=repo_root),
    }
    report["clauses"] = derive_clauses(report)
    report["is_science"] = all(report["clauses"].values())
    return report


def _rationale(
    point: float,
    ci: Mapping[str, Any],
    margin: float,
    readings: Mapping[str, bool],
    verdict: str,
    deltas: Sequence[float],
) -> str:
    above = sum(1 for d in deltas if d > margin)
    return (
        f"Paired macro Δ = {point:+.6f} against the D17(c) margin {margin} (strict >); its "
        f"order-blocked {ci.get('ci_level')} CI is "
        f"[{ci.get('lower'):+.6f}, {ci.get('upper'):+.6f}] "
        f"over {ci.get('n_blocks')} held-out orders ({ci.get('n_boot')} replicates). The shipped "
        f"arm is worse by more than the margin in {above} of {len(deltas)} orders. Readings: "
        + ", ".join(f"{k}={'fires' if v else 'does not fire'}" for k, v in readings.items())
        + f" -> {verdict}."
    )


# --------------------------------------------------------------------------- #
# Clauses — every one re-derived from the report's own numbers
# --------------------------------------------------------------------------- #
def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """Re-derive every clause from the recorded values — never from a recorded boolean."""
    margin = report.get("margin") if isinstance(report.get("margin"), Mapping) else {}
    arms = report.get("arms") if isinstance(report.get("arms"), Mapping) else {}
    stat = report.get("statistic") if isinstance(report.get("statistic"), Mapping) else {}
    res = report.get("resampling") if isinstance(report.get("resampling"), Mapping) else {}
    ci = res.get("ci") if isinstance(res.get("ci"), Mapping) else {}
    desc = report.get("descriptive") if isinstance(report.get("descriptive"), Mapping) else {}
    per_unit = desc.get("per_unit") if isinstance(desc.get("per_unit"), Mapping) else {}
    rd = report.get("rederivation") if isinstance(report.get("rederivation"), Mapping) else {}
    rd_problems = rd.get("problems") if isinstance(rd.get("problems"), Mapping) else {}
    readings = report.get("readings") if isinstance(report.get("readings"), Mapping) else {}
    disc = report.get("disclosures") if isinstance(report.get("disclosures"), list) else []
    pairing = report.get("pairing") if isinstance(report.get("pairing"), Mapping) else {}

    shipped = arms.get("shipped") if isinstance(arms.get("shipped"), Mapping) else {}
    comparator = arms.get("comparator") if isinstance(arms.get("comparator"), Mapping) else {}

    deltas: list[float] = []
    units_ok = bool(per_unit)
    for row in per_unit.values():
        s, c, d = (
            _finite(row.get("shipped")) if isinstance(row, Mapping) else None,
            _finite(row.get("comparator")) if isinstance(row, Mapping) else None,
            _finite(row.get("delta")) if isinstance(row, Mapping) else None,
        )
        if s is None or c is None or d is None or abs((s - c) - d) > _ABS_TOL:
            units_ok = False
            continue
        deltas.append(d)
    point = _finite(stat.get("point"))
    macro_diff = _finite(stat.get("difference_of_published_macros"))
    lower, upper = _finite(ci.get("lower")), _finite(ci.get("upper"))
    value = _finite(margin.get("value"))

    point_ok = (
        units_ok
        and point is not None
        and macro_diff is not None
        and _finite(ci.get("point")) == point
        and abs(math.fsum(deltas) / len(deltas) - point) <= _ABS_TOL
        and abs(macro_diff - point) <= 1e-9
    )
    recorded_fires = {
        name: (row.get("fires") if isinstance(row, Mapping) else None)
        for name, row in readings.items()
    }
    if point is not None and lower is not None and value is not None:
        derived = readings_for(point, lower, value)
    else:
        derived = {}
    readings_ok = (
        bool(derived)
        and sorted(recorded_fires) == sorted(READINGS)
        and all(recorded_fires[k] is derived[k] for k in READINGS)
    )
    verdict = verdict_from(derived) if derived else None
    n_boot_requested = res.get("n_boot_requested")
    survived = ci.get("n_boot")

    return {
        "margin_is_the_pinned_d17c_value": value is not None and value == float(PW.SWAP_ECE_MARGIN),
        "arms_are_the_shipped_backbone_and_the_d6_comparator": (
            shipped.get("backbone") == BR.PRODUCTION_BACKBONE
            and comparator.get("backbone") == BR.COMPARATOR_BACKBONE
        ),
        "both_gate2_reports_validate": _all_empty(report.get("gate2_validation"), 2),
        "both_gate2_reports_bound_to_their_sidecars": _all_empty(report.get("sidecar_binding"), 2),
        # Bandwidth selection re-run, not merely the recorded bandwidths re-used: only the
        # former proves each unit's bandwidth is the selector's own choice (gate2 docstring).
        "inputs_rederive_from_their_sidecars": (
            rd.get("ran") is True
            and rd.get("select_bandwidth") is True
            and _all_empty(rd_problems, 2)
        ),
        "arms_graded_on_one_population": (
            pairing.get("problems") == []
            and isinstance(pairing.get("n_units_admissible"), int)
            and pairing.get("n_units_admissible") == len(per_unit) > 0
        ),
        "point_is_the_difference_of_the_published_macros": point_ok,
        "ci_is_order_block_resampled": (
            res.get("block_key") == UNIT_KEY
            and ci.get("n_blocks") == len(per_unit)
            and lower is not None
            and upper is not None
            and point is not None
            and lower <= point <= upper
        ),
        "bootstrap_replicates_are_not_a_cost_knob": (
            isinstance(n_boot_requested, int)
            and not isinstance(n_boot_requested, bool)
            and n_boot_requested >= N_BOOT
            and survived == n_boot_requested
        ),
        "verdict_follows_every_reading": (
            readings_ok
            and report.get("verdict") == verdict
            and report.get("swap_fired") is (verdict == "fired")
            and report.get("requires_user_decision") is (verdict != "not_fired")
        ),
        "a15_confounds_disclosed": all(
            any(isinstance(line, str) and marker in line for line in disc)
            for marker in A15_CONFOUND_MARKERS
        ),
    }


def _all_empty(block: Any, n: int) -> bool:
    """A {role: problems} block with exactly ``n`` roles, every one an empty list."""
    return (
        isinstance(block, Mapping)
        and len(block) == n
        and all(isinstance(v, list) and not v for v in block.values())
    )


def validate_report(
    report: Mapping[str, Any], *, repo_root: str | Path = G2._REPO_ROOT
) -> list[str]:
    """Problems with a swap-check report; never raises on a malformed one.

    Two layers. (1) Every clause is re-derived from the recorded numbers and compared with the
    recorded clause. (2) The report is re-derived from the two GATE-2 reports it names — which
    must still hash to what its provenance recorded — and every published number and the verdict
    must match; an internally consistent forgery cannot survive a re-run against its inputs
    ([[gate-must-bind-to-upstream-evidence]]). The expensive sidecar re-derivation is not re-run
    here; its recorded outcome is a clause.
    """
    problems: list[str] = []
    try:
        if report.get("schema_version") != SCHEMA_VERSION:
            problems.append(f"schema_version {report.get('schema_version')!r} != {SCHEMA_VERSION}")
        clauses = derive_clauses(report)
        recorded = report.get("clauses") if isinstance(report.get("clauses"), Mapping) else {}
        if sorted(recorded) != sorted(_CLAUSES) or sorted(clauses) != sorted(_CLAUSES):
            problems.append("the clause set is not the pinned one")
        for name in _CLAUSES:
            if recorded.get(name) is not clauses.get(name):
                problems.append(
                    f"clause {name}: recorded {recorded.get(name)!r}, "
                    f"re-derived {clauses.get(name)!r}"
                )
            elif clauses.get(name) is not True:
                problems.append(f"clause {name} is FALSE")
        if report.get("is_science") is not all(clauses.values()):
            problems.append("is_science does not equal the conjunction of the re-derived clauses")
        problems.extend(_rederivation_problems(report, repo_root=repo_root))
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        problems.append(f"the report could not be validated: {type(exc).__name__}: {exc}")
    return problems


#: Fields compared between the recorded report and its re-run against the named inputs.
_RERUN_FIELDS: tuple[tuple[str, ...], ...] = (
    ("statistic", "point"),
    ("statistic", "difference_of_published_macros"),
    ("resampling", "ci"),
    ("resampling", "n_boot_requested"),
    ("resampling", "seed"),
    ("descriptive", "per_unit"),
    ("arms",),
    ("readings",),
    ("verdict",),
    ("margin",),
)


def _rederivation_problems(report: Mapping[str, Any], *, repo_root: str | Path) -> list[str]:
    root = Path(repo_root)
    arms = report["arms"]
    paths = {role: arms[role]["gate2_report"] for role in ("shipped", "comparator")}
    inputs = report.get("provenance", {}).get("inputs", {})
    problems: list[str] = []
    for role, rel in paths.items():
        target = root / str(rel)
        if not target.is_file():
            return [f"the {role} GATE-2 report {rel!r} is not a file — nothing to re-derive from"]
        if inputs.get(rel) != PROV.sha256_file(target):
            problems.append(
                f"the {role} GATE-2 report {rel!r} does not hash to provenance.inputs — the "
                "verdict is not about these reports"
            )
    if problems:
        return problems
    rerun = rnafm_swap_condition_c(
        _load_json(root / paths["shipped"]),
        _load_json(root / paths["comparator"]),
        shipped_path=paths["shipped"],
        comparator_path=paths["comparator"],
        repo_root=root,
        n_boot=int(report["resampling"]["n_boot_requested"]),
        seed=int(report["resampling"]["seed"]),
        rederivation=report.get("rederivation"),
    )
    for path in _RERUN_FIELDS:
        got: Any = report
        want: Any = rerun
        for key in path:
            got = got.get(key) if isinstance(got, Mapping) else None
            want = want.get(key) if isinstance(want, Mapping) else None
        if got != want:
            problems.append(f"{'.'.join(path)} does not re-derive from the named GATE-2 reports")
    return problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tbox_finder.calib.swap_check",
        description="P3-18: evaluate ADR-0005 D17(c), the RiNALMo -> RNA-FM ECE swap margin.",
    )
    p.add_argument("--shipped-report", default=DEFAULT_SHIPPED_REPORT)
    p.add_argument("--comparator-report", default=DEFAULT_COMPARATOR_REPORT)
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument(
        "--recorded-bandwidths",
        action="store_true",
        help="re-derive at each unit's RECORDED kernel bandwidth (~35 s/report) instead of "
        "re-running the selection (~100 s/report); the report's is_science is then FALSE",
    )
    p.add_argument(
        "--skip-rederivation",
        action="store_true",
        help="skip gate2.rederive_against_sidecars (the report's is_science is then FALSE)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = G2._REPO_ROOT
    shipped = _load_json(root / args.shipped_report)
    comparator = _load_json(root / args.comparator_report)
    if args.skip_rederivation:
        rederivation: dict[str, Any] = {"ran": False, "select_bandwidth": None, "problems": {}}
    else:
        rederivation = {
            "ran": True,
            "select_bandwidth": not args.recorded_bandwidths,
            "problems": {
                role: G2.rederive_against_sidecars(
                    rep, repo_root=root, select_bandwidth=not args.recorded_bandwidths
                )
                for role, rep in (("shipped", shipped), ("comparator", comparator))
            },
        }
    report = rnafm_swap_condition_c(
        shipped,
        comparator,
        shipped_path=args.shipped_report,
        comparator_path=args.comparator_report,
        repo_root=root,
        rederivation=rederivation,
    )
    # Embedded provenance hashes INPUTS only — the report is its own output
    # ([[build-provenance-hashes-its-outputs]]).
    report["provenance"] = PROV.build_provenance(
        rule=RULE,
        script=GENERATED_BY,
        seed=SEED,
        inputs=[
            args.shipped_report,
            args.comparator_report,
            shipped["scoring"]["loo_scores"],
            comparator["scoring"]["loo_scores"],
            shipped["scoring"]["in_distribution_scores"],
            comparator["scoring"]["in_distribution_scores"],
        ],
        env_lock=ENV_LOCK,
        adr="ADR-0005",
        repo_root=root,
    )
    problems = validate_report(report, repo_root=root)
    valid = not problems
    target = Path(args.report) if valid else Path(args.report).with_suffix(".invalid.json")
    target = target if target.is_absolute() else root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    canonical = Path(args.report) if Path(args.report).is_absolute() else root / args.report
    if valid:
        canonical.with_suffix(".invalid.json").unlink(missing_ok=True)
    elif canonical.exists():
        canonical.unlink()
    print(report["rationale"])
    if report["requires_user_decision"]:
        print(f"STOP (CLAUDE.md §7): verdict {report['verdict']!r} needs a user decision.")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 0 if valid else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
