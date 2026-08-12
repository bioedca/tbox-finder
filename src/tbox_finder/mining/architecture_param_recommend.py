"""Re-derive criterion (b)'s seven ADR-0006 A4 rule parameters, over the whole grid.

`P3-15′-f` measured (b) on 278 de-novo consensuses of round-0 **false-positive-manifest**
candidates and put seven parameter values to the user; the user declined to pin them on
that evidence and asked for a **matched positive control** first (§7, 2026-08-09).
`P3-15′-g`/`-g-ii`/`-g-iii`/`-g-iv` built and measured that control.  `P3-15′-f`'s block
asked for its recommendation to be **re-derived, not re-quoted**, once the control landed.
This module is that re-derivation, and it changes what the choice is *about*.

Three things this module derives that no committed report states
================================================================
1. **The round's yield ceiling, before (b) is chosen at all.**  Sparing is a disjunction
   (ADR-0005 D14), so mining is a *conjunction*: a candidate is mined only if criterion
   (a) **and** (b) **and** (c) **and** — at P3 — the Stage-2 disjunct all resolve
   ``failed``.  Criterion (a) fails on 111 of 941 and criterion (c) on 96; their
   **intersection** is what (b) can ever act on.  Everything (b) decides outside that
   intersection changes no outcome.
2. **What (b)'s parameters are worth in mined candidates**, computed through the shipped
   :func:`tbox_finder.mining.spare_rule.is_mining_excluded` rather than a re-derived
   conjunction, over **every admissible point of the grid** — not the six named tuples
   the two measurement reports happen to carry.
3. **What each point costs on known T-boxes**, at the record level the matched control's
   pseudo-replication permits (149 of its 160 records contribute 2 queries), and — the
   quantity that actually decides label noise — the share of control records that fall
   through **(a) and (b) together**, which is not the same object as (b)'s own failure
   share and does not behave like it.

Why the recommendation is rule-first, and why that matters here
===============================================================
The admissible grid has thousands of points and the control has 68 producible records.
An argmax over that grid is a fitted value with no out-of-sample warrant, and ADR-0006 A4
requires the round to *supply* a value and record it — a **chosen rule**, not a tuned one.
So :data:`DECISION_FLOORS` is stated from ADR-0006 D3's own wording and ADR-0004 D1's
element vocabulary **before** any count is read, the floors are applied, and the winner is
whichever admissible survivor is **most permissive as measured** — permissiveness being
the fail-closed direction for a sparing disjunct, by A4's own argument.  The frontier of
higher-yield settings is reported in full so the reader can see exactly what the rule
declines to take, and why taking it would be fitting.

This module **pins nothing** (``pins_nothing: true``).  The value is the user's §7 call and
enters the round through the producer's seven ``${VAR:?}`` exports, recorded in provenance
per A4; D6/D17's P6 freeze is untouched.

Run::

    PYTHONPATH=src python -m tbox_finder.mining.architecture_param_recommend build-inputs \\
        --covariation-status <covariation_status.json> \\
        --synteny-status <synteny_status.json> \\
        --stage2-posteriors <stage2_posteriors.json> \\
        --out data/processed/mining/round0_fp_spare_rule_inputs_v0.json

    PYTHONPATH=src python -m tbox_finder.mining.architecture_param_recommend recommend \\
        --fp-msa-root <dir of <slug>/msa.sto from round_p3_15_supply> \\
        --control-msa-root <dir of <slug>/msa.sto from round_p3_15g_control> \\
        --out reports/p3/architecture_parameter_recommendation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tbox_finder.mining import architecture as arch
from tbox_finder.mining.architecture_param_control_compare import record_of
from tbox_finder.mining.architecture_param_measure import (
    BULGE_MAX_NT_SWEEP,
    BULGE_MIN_NT_SWEEP,
    MIN_HELIX_PAIRS_SWEEP,
    MIN_NAMED_HELICES_SWEEP,
    NCCA_PAIRING_NT_SWEEP,
    ParamTuple,
    SupplyItem,
    candidate_state,
    default_tuples,
    is_inside_repo,
    is_local_path_shaped,
    portable_path,
    read_supply,
    sha256_of,
    supply_digest,
)
from tbox_finder.mining.architecture_producer import candidate_id_of, candidate_msa_path
from tbox_finder.mining.curated_control_sizing import wilson_interval
from tbox_finder.mining.spare_rule import (
    DISJUNCT_STATUSES,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    SpareRuleEvidence,
    is_mining_excluded,
)
from tbox_finder.provenance import build_provenance

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-f"
ADR = (
    "ADR-0006 D3 (criterion (b)'s named-element predicate), D6 (the short-Stem-I "
    "relaxation), A4 (the seven rule parameters supplied per round, no defaults); "
    "ADR-0005 D14 (the spare rule: sparing is a disjunction, so mining is a "
    "conjunction; unavailable spares); ADR-0004 D1 (the element vocabulary D3 reuses)"
)

#: The sentinel on the ``bulge_max_nt`` axis that means "no upper bound at all".  A
#: setting carrying it has no bulge *size* filter: only ``bulge_min_nt`` decides.
BULGE_MAX_UNBOUNDED = 10_000

#: The two states a consensus can be in once it exists (``unavailable`` is its absence).
DECIDED_STATES: tuple[str, ...] = ("passed", "failed")

#: Stage-2 thresholds the yield is reported *across*.  ADR-0005 D3/D14 freeze the real
#: value at the §13.1 phase gate and this module pins none of them — the list exists so
#: the ceiling is quoted as a band rather than at one unowned number.
STAGE2_THRESHOLD_SENSITIVITY: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)


class RecommendError(ValueError):
    """The recommendation could not be derived from the inputs as given."""


# ═════════════════════════════════════════════════════════════════════════════
# The decision rule — stated from the documents, before any count is read
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DecisionFloor:
    """One admissibility floor on the grid, with the sentence it comes from.

    A floor is a *semantic* requirement on what the published predicate may be
    described as testing — never a threshold tuned against the arms.  ``predicate``
    is what the code applies; ``rationale`` and ``source`` are what the report
    prints beside it, so a reader can reject the floor rather than reverse-engineer it.
    """

    key: str
    predicate: Callable[[ParamTuple], bool]
    rationale: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {"floor": self.key, "rationale": self.rationale, "source": self.source}


DECISION_FLOORS: tuple[DecisionFloor, ...] = (
    DecisionFloor(
        key="min_named_helices >= 2",
        predicate=lambda p: p.min_named_helices >= 2,
        rationale=(
            "D3's pass condition is that 'the expected helices are present' — plural, over "
            "the four-element ADR-0004 D1 vocabulary (Stem_I, Stem_II, Stem_III, "
            "Antiterminator). A setting satisfied by ONE resolved helix of four is a "
            "'some helix is present' predicate; whatever its measured behaviour, the "
            "sentence the round would publish about it is not the one D3 licenses."
        ),
        source="ADR-0006 D3; ADR-0004 D1",
    ),
    DecisionFloor(
        key="min_helix_pairs >= 2",
        predicate=lambda p: p.min_helix_pairs >= 2,
        rationale=(
            "min_helix_pairs is the number of stacked pairs at which a run of pairing "
            "counts as a helix at all. At 1 a single isolated base pair is a named "
            "structural element, which the localizer's own stack-depth distribution shows "
            "is noise-level on these consensuses."
        ),
        source="ADR-0006 D3 (named-element presence)",
    ),
    DecisionFloor(
        key="ncca_pairing_nt >= 2",
        predicate=lambda p: p.ncca_pairing_nt >= 2,
        rationale=(
            "D3 requires an 'NCCA-PAIRING bulge'. One satisfied contact in the acceptor "
            "register is a coincidence at these alignment widths, not a register; two is "
            "the weakest reading under which the word 'pairing' is doing work."
        ),
        source="ADR-0006 D3",
    ),
    DecisionFloor(
        key="bulge_max_nt is bounded",
        predicate=lambda p: p.bulge_max_nt != BULGE_MAX_UNBOUNDED,
        rationale=(
            "bulge_max_nt = 10000 is the axis's no-upper-bound sentinel: the size filter "
            "stops filtering and bulge_min_nt alone decides. A round may choose that, but "
            "it may not then describe (b) as testing a bulge SIZE. ⚠ This floor is "
            "semantic, not decisive: on this supply the bounded value the rule selects "
            "measures IDENTICALLY to the sentinel (verified, see "
            "recommendation.floor_sensitivity)."
        ),
        source="ADR-0006 D3; ADR-0006 A4 (the value is recorded and must mean what it says)",
    ),
)

#: Applied to the survivors of :data:`DECISION_FLOORS`, in order, only to break an exact
#: measured tie.  Every entry resolves toward the value that CLAIMS LESS.
TIE_BREAK_ORDER: tuple[tuple[str, str], ...] = (
    (
        "allow_wobble",
        "False — the flag is proved structurally inert for the NCCA acceptor, so recording "
        "it as enabled would publish a relaxation that cannot fire",
    ),
    (
        "stem_i_nt_threshold",
        "the smallest value on the axis — the D6 relaxation is proved inert on this "
        "manifest (neither stem_i_extent_nt nor regulatory_mode is supplied), and the "
        "smallest value is the one under which it demonstrably never fires",
    ),
    (
        "bulge_max_nt",
        "the smallest bounded value among exact ties — a narrower stated filter, given the "
        "wider one measured identically",
    ),
)

DECISION_RULE_STATEMENT = (
    "Stated before any count was read. (1) Keep only points the SHIPPED localizer accepts "
    "on this supply — admissibility is measured by running it, not by re-encoding its "
    "guards. (2) Keep only points that clear every DECISION_FLOORS entry, each of which is "
    "a semantic requirement read off ADR-0006 D3's wording, not a tuned threshold. "
    "(3) Among the survivors choose the MOST PERMISSIVE as measured — the smallest number "
    "of FP-arm consensuses moved to 'failed'. Permissiveness is the fail-closed direction "
    "for a SPARING disjunct, and ADR-0006 A4 makes that argument itself: a relaxation "
    "'only ever makes (b) more likely to pass => more sparing => the fail-closed "
    "direction, so an imperfect value errs safe'. (4) Break exact ties by TIE_BREAK_ORDER, "
    "and REFUSE if a tie survives among parameters not proved inert. "
    "⚠ What this rule deliberately does NOT do is maximise yield over the grid: with "
    "thousands of admissible points and a control of 68 producible records, the argmax is "
    "a fitted value, and A4 asks the round to supply a chosen one."
)


# ═════════════════════════════════════════════════════════════════════════════
# The grid
# ═════════════════════════════════════════════════════════════════════════════
def grid_points(stem_i_nt_threshold: int = 1) -> tuple[ParamTuple, ...]:
    """Every point of the six-axis cross product the two measurement reports sweep.

    ``stem_i_nt_threshold`` is not an axis of that sweep — it is proved inert on both
    manifests (neither supplies ``stem_i_extent_nt`` nor ``regulatory_mode``), so the
    reports carry it as a fixed value and so does this grid.  It is a parameter here
    rather than a literal so a manifest that DID supply those fields could be swept
    without editing the enumeration.
    """
    out: list[ParamTuple] = []
    for mnh in MIN_NAMED_HELICES_SWEEP:
        for mhp in MIN_HELIX_PAIRS_SWEEP:
            for bmin in BULGE_MIN_NT_SWEEP:
                for bmax in BULGE_MAX_NT_SWEEP:
                    for ncca in NCCA_PAIRING_NT_SWEEP:
                        for wob in (False, True):
                            out.append(
                                ParamTuple(
                                    label=(
                                        f"mnh{mnh}_mhp{mhp}_bulge{bmin}-{bmax}"
                                        f"_ncca{ncca}_wob{int(wob)}"
                                    ),
                                    stem_i_nt_threshold=stem_i_nt_threshold,
                                    min_named_helices=mnh,
                                    min_helix_pairs=mhp,
                                    bulge_min_nt=bmin,
                                    bulge_max_nt=bmax,
                                    ncca_pairing_nt=ncca,
                                    allow_wobble=wob,
                                )
                            )
    return tuple(out)


def sweep_arm(
    items: Sequence[SupplyItem],
    params: ParamTuple,
) -> dict[str, str] | str:
    """Per-slug (b) verdicts at ``params``, or the refusal string the shipped path raised.

    Admissibility is **measured**: the localizer refuses a setting whose bulge size floor
    is shorter than its NCCA register (a bulge that passes the size filter but cannot hold
    the motif would read ``absent`` ⇒ minable rather than ``undetectable`` ⇒ spared).
    Re-encoding that guard here would let the two drift, and the drift direction is a grid
    point silently scored as if it were legal.
    """
    verdicts: dict[str, str] = {}
    for item in items:
        try:
            verdicts[item.slug] = candidate_state(item, params)
        except arch.ArchitectureError as exc:
            return str(exc)
    return verdicts


# ═════════════════════════════════════════════════════════════════════════════
# The spare-rule inputs (criterion (a), criterion (c), the Stage-2 posterior)
# ═════════════════════════════════════════════════════════════════════════════
def load_spare_rule_inputs(path: str | Path) -> dict[str, Any]:
    """The committed 941-row join, validated rather than trusted.

    Committed for the same reason ``curated_control_status_v0.json`` is: the yield
    arithmetic below is the load-bearing number of this step and it must be auditable
    without cluster access.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecommendError(f"{path}: cannot read spare-rule inputs ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise RecommendError(f"{path}: spare-rule inputs must be a JSON object")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RecommendError(f"{path}: spare-rule inputs carry no rows")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RecommendError(f"{path}: a spare-rule input row is not an object")
        cid = row.get("candidate_id")
        if not isinstance(cid, str) or not cid:
            raise RecommendError(f"{path}: a spare-rule input row carries no candidate_id")
        if cid in by_id:
            # A dict comprehension would have MERGED the duplicate and every derived
            # count would still reconcile ([[duplicate-key-merges-instead-of-colliding]]).
            raise RecommendError(f"{path}: duplicate candidate_id {cid!r}")
        for field in ("covariation_status", "synteny_status"):
            if row.get(field) not in DISJUNCT_STATUSES:
                raise RecommendError(
                    f"{path}: {cid} has {field}={row.get(field)!r}, "
                    f"expected one of {tuple(DISJUNCT_STATUSES)}"
                )
        posterior = row.get("stage2_posterior")
        if posterior is not None and not isinstance(posterior, (int, float)):
            raise RecommendError(f"{path}: {cid} has a non-numeric stage2_posterior")
        by_id[cid] = dict(row)
    payload = dict(payload)
    payload["by_id"] = by_id
    return payload


def assert_covers_manifest(by_id: Mapping[str, Any], manifest_ids: Sequence[str]) -> None:
    """Exact set equality with the manifest — an extra id is as fatal as a missing one."""
    want, have = set(manifest_ids), set(by_id)
    if want == have:
        return
    missing, extra = sorted(want - have), sorted(have - want)
    raise RecommendError(
        f"spare-rule inputs do not cover the manifest: {len(missing)} missing "
        f"(e.g. {missing[:2]}), {len(extra)} extra (e.g. {extra[:2]})"
    )


def evidence_for(
    row: Mapping[str, Any],
    architecture_status: str,
) -> SpareRuleEvidence:
    """One candidate's four-disjunct evidence, in the shipped dataclass."""
    return SpareRuleEvidence(
        relaxed_architecture=architecture_status,
        any_helix_rscape=str(row["covariation_status"]),
        downstream_aaRS_synteny=str(row["synteny_status"]),
        stage2_posterior=row.get("stage2_posterior"),
    )


def mined_ids(
    by_id: Mapping[str, Mapping[str, Any]],
    arch_status: Mapping[str, str],
    *,
    stage2_threshold: float | None,
) -> list[str]:
    """Every candidate the round would MINE, through the shipped spare rule.

    ``arch_status`` is keyed by candidate id and may omit ids: a candidate with no
    consensus has no (b) verdict, which the producer scores ``unavailable`` ⇒ spared.
    That default is supplied here rather than assumed by the caller, because the
    fail-open version of this function is the one that mines the whole corpus.
    """
    out: list[str] = []
    for cid, row in by_id.items():
        evidence = evidence_for(row, arch_status.get(cid, STATUS_UNAVAILABLE))
        if not is_mining_excluded(evidence, stage2_threshold=stage2_threshold):
            out.append(cid)
    return sorted(out)


def yield_ceiling(
    by_id: Mapping[str, Mapping[str, Any]],
    *,
    ids_with_consensus: Sequence[str],
) -> dict[str, Any]:
    """What the round can mine at MOST, over every possible criterion-(b) verdict.

    Computed by handing the shipped spare rule the two extreme (b) columns —
    ``failed`` everywhere (maximally minable) and ``passed`` everywhere (maximally
    sparing).  The gap between them is the whole of what choosing (b)'s parameters can
    move; the ceiling itself is set by (a), (c) and Stage-2 and no (b) value can raise it.
    """
    all_failed = {cid: STATUS_FAILED for cid in ids_with_consensus}
    all_passed = {cid: STATUS_PASSED for cid in ids_with_consensus}
    counts = {
        criterion: {
            state: sum(1 for r in by_id.values() if r[column] == state)
            for state in DISJUNCT_STATUSES
        }
        for criterion, column in (
            ("criterion_a_covariation", "covariation_status"),
            ("criterion_c_synteny", "synteny_status"),
        )
    }
    a_failed = {cid for cid, r in by_id.items() if r["covariation_status"] == STATUS_FAILED}
    c_failed = {cid for cid, r in by_id.items() if r["synteny_status"] == STATUS_FAILED}
    both = a_failed & c_failed
    band: dict[str, Any] = {}
    for threshold in (None, *STAGE2_THRESHOLD_SENSITIVITY):
        key = "stage2_not_declared" if threshold is None else f"stage2_threshold_{threshold}"
        band[key] = {
            "max_mined_if_b_failed_everywhere": len(
                mined_ids(by_id, all_failed, stage2_threshold=threshold)
            ),
            "min_mined_if_b_passed_everywhere": len(
                mined_ids(by_id, all_passed, stage2_threshold=threshold)
            ),
        }
    return {
        "n_candidates": len(by_id),
        "status_counts": counts,
        "n_criterion_a_failed": len(a_failed),
        "n_criterion_c_failed": len(c_failed),
        "n_failing_both_a_and_c": len(both),
        "n_failing_both_a_and_c_with_a_consensus": len(both & set(ids_with_consensus)),
        # Named, not just counted: 15 ids is small enough to publish, and a ceiling a
        # reader cannot check against the manifest is a number they have to take on trust.
        "candidate_ids_at_the_ceiling": mined_ids(by_id, all_failed, stage2_threshold=None),
        "by_stage2_threshold": band,
        "reading": (
            "sparing is a disjunction (ADR-0005 D14), so mining is a conjunction: only a "
            "candidate every disjunct scored 'failed' is mined. Criterion (b) can therefore "
            "act at all only inside the (a)-failed AND (c)-failed intersection; outside it "
            "(b)'s verdict changes no outcome. max_mined_if_b_failed_everywhere is the "
            "ceiling the parameter choice is bounded by — no setting can exceed it — and "
            "min_mined_if_b_passed_everywhere is the floor it cannot go below."
        ),
        "stage2_threshold_note": (
            "ADR-0005 D3/D14 freeze the Stage-2 threshold at the §13.1 phase gate; none is "
            "pinned here. 'stage2_not_declared' is the P2 accounting (the disjunct is not "
            "in play) and is an UPPER bound: declaring Stage-2 at any threshold can only "
            "spare more."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The control arm — record-level, because its queries are pseudo-replicated
# ═════════════════════════════════════════════════════════════════════════════
def control_records(status: Mapping[str, str]) -> dict[str, list[str]]:
    """``record -> its query ids``, from the merged control status table."""
    out: dict[str, list[str]] = {}
    for cid in sorted(status):
        out.setdefault(record_of(cid), []).append(cid)
    return out


def control_damage(
    status: Mapping[str, str],
    records: Mapping[str, Sequence[str]],
    arch_by_query: Mapping[str, str],
) -> dict[str, Any]:
    """What a setting costs on KNOWN T-boxes, in the two units that are defensible.

    ``all_disjuncts_failed`` is the quantity that decides label noise: a record is
    damaged only where **every** producible query of it fails (a) *and* (b), because
    any surviving pass spares it.  ⚠ Criterion (c) is **not** in this conjunction — the
    control's host genomes were never annotated, so (c) is unmeasured on this arm.  The
    number is therefore an **upper bound** on the damage, and the direction is stated
    rather than absorbed.
    """
    n_producible = 0
    n_records_ab_failed = 0
    n_records_b_failed = 0
    q_decided = 0
    q_b_failed = 0
    q_ab_failed = 0
    for queries in records.values():
        producible = [q for q in queries if status[q] in DECIDED_STATES]
        if not producible:
            continue
        n_producible += 1
        q_decided += len(producible)
        b_states = [arch_by_query.get(q, STATUS_UNAVAILABLE) for q in producible]
        q_b_failed += sum(1 for s in b_states if s == STATUS_FAILED)
        q_ab_failed += sum(
            1
            for q, s in zip(producible, b_states, strict=True)
            if s == STATUS_FAILED and status[q] == STATUS_FAILED
        )
        if all(s == STATUS_FAILED for s in b_states):
            n_records_b_failed += 1
        if all(
            s == STATUS_FAILED and status[q] == STATUS_FAILED
            for q, s in zip(producible, b_states, strict=True)
        ):
            n_records_ab_failed += 1
    lo, hi = wilson_interval(n_records_ab_failed, n_producible)
    lo_b, hi_b = wilson_interval(n_records_b_failed, n_producible)
    return {
        "n_records_producible": n_producible,
        "n_records_losing_b": n_records_b_failed,
        "share_records_losing_b": round(n_records_b_failed / n_producible, 6),
        "share_records_losing_b_ci95": [round(lo_b, 6), round(hi_b, 6)],
        "n_records_losing_a_and_b": n_records_ab_failed,
        "share_records_losing_a_and_b": round(n_records_ab_failed / n_producible, 6),
        "share_records_losing_a_and_b_ci95": [round(lo, 6), round(hi, 6)],
        "n_queries_decided": q_decided,
        "n_queries_b_failed": q_b_failed,
        "n_queries_a_and_b_failed": q_ab_failed,
        "query_level_ci95": None,
        "why_no_query_ci": (
            "the control's queries are pseudo-replicated — most records contribute more "
            "than one — so a binomial interval on them would be narrower than the data "
            "support. Every interval here is on the record level."
        ),
        "criterion_c_not_in_this_conjunction": (
            "the control's host genomes carry no GFF annotation, so criterion (c) is "
            "UNMEASURED on this arm and is absent from the conjunction. Real T-boxes are "
            "the population (c) is most likely to spare, so this damage figure is an UPPER "
            "bound; the FP arm's yield, by contrast, does include (c)."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# One grid point, both arms
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PointResult:
    """Everything measured at one admissible grid point."""

    params: ParamTuple
    fp_failed: int
    fp_passed: int
    fp_failed_with_a_failed: int
    mined_by_threshold: Mapping[str, int]
    control: Mapping[str, Any]

    @property
    def mined(self) -> int:
        """Yield with the Stage-2 disjunct not declared — the upper bound."""
        return int(self.mined_by_threshold["stage2_not_declared"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.as_dict(),
            "fp_arm": {
                "failed": self.fp_failed,
                "passed": self.fp_passed,
                "share_failed_of_decided": (
                    round(self.fp_failed / (self.fp_failed + self.fp_passed), 6)
                    if (self.fp_failed + self.fp_passed)
                    else None
                ),
                "failed_with_criterion_a_failed": self.fp_failed_with_a_failed,
            },
            "round_yield_n_mined": dict(self.mined_by_threshold),
            "control_arm": dict(self.control),
        }


def evaluate_point(
    params: ParamTuple,
    *,
    fp_items: Sequence[SupplyItem],
    fp_id_by_slug: Mapping[str, str],
    by_id: Mapping[str, Mapping[str, Any]],
    control_items: Sequence[SupplyItem],
    control_id_by_slug: Mapping[str, str],
    control_status: Mapping[str, str],
    control_record_index: Mapping[str, Sequence[str]],
) -> PointResult | str:
    """Both arms at one grid point, or the refusal string if the point is inadmissible."""
    fp_verdicts = sweep_arm(fp_items, params)
    if isinstance(fp_verdicts, str):
        return fp_verdicts
    control_verdicts = sweep_arm(control_items, params)
    if isinstance(control_verdicts, str):
        return control_verdicts

    arch_by_id = {fp_id_by_slug[slug]: state for slug, state in fp_verdicts.items()}
    mined = {
        ("stage2_not_declared" if t is None else f"stage2_threshold_{t}"): len(
            mined_ids(by_id, arch_by_id, stage2_threshold=t)
        )
        for t in (None, *STAGE2_THRESHOLD_SENSITIVITY)
    }
    fp_failed = sum(1 for s in fp_verdicts.values() if s == STATUS_FAILED)
    fp_with_a = sum(
        1
        for cid, state in arch_by_id.items()
        if state == STATUS_FAILED and by_id[cid]["covariation_status"] == STATUS_FAILED
    )
    control_by_query = {control_id_by_slug[s]: v for s, v in control_verdicts.items()}
    return PointResult(
        params=params,
        fp_failed=fp_failed,
        fp_passed=len(fp_verdicts) - fp_failed,
        fp_failed_with_a_failed=fp_with_a,
        mined_by_threshold=mined,
        control=control_damage(control_status, control_record_index, control_by_query),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Applying the rule
# ═════════════════════════════════════════════════════════════════════════════
def floors_passed(
    params: ParamTuple,
    floors: Sequence[DecisionFloor] = DECISION_FLOORS,
) -> tuple[str, ...]:
    """The keys of every floor ``params`` clears."""
    return tuple(f.key for f in floors if f.predicate(params))


def clears_all_floors(
    params: ParamTuple,
    floors: Sequence[DecisionFloor] = DECISION_FLOORS,
) -> bool:
    """The ONLY spelling of "this setting is one the rule may choose".

    ``apply_decision_rule`` filtered with its own inline ``all(f.predicate(...))`` until
    a sabotage showed the two could drift: neutering this function left the selection
    untouched, which is precisely the state in which the frontier and the recommendation
    could disagree about what "clears every floor" means
    ([[fixed-one-of-two-identical-things]]).
    """
    return len(floors_passed(params, floors)) == len(floors)


def _tie_break_key(result: PointResult) -> tuple[Any, ...]:
    p = result.params
    return (int(p.allow_wobble), p.stem_i_nt_threshold, p.bulge_max_nt)


def apply_decision_rule(
    results: Sequence[PointResult],
    *,
    floors: Sequence[DecisionFloor] = DECISION_FLOORS,
) -> dict[str, Any]:
    """The rule of :data:`DECISION_RULE_STATEMENT`, applied to the measured grid.

    ``floors`` is a parameter only so :func:`floor_sensitivity` can re-run the *same*
    selection with one floor removed.  A second copy of the selection there would be a
    diagnostic that could disagree with the decision it is describing.
    """
    survivors = [r for r in results if clears_all_floors(r.params, floors)]
    if not survivors:
        raise RecommendError(
            "no admissible grid point clears every decision floor; the floors and the "
            "shipped axes disagree, which is a decision to take, not a tie to break"
        )
    fewest = min(r.fp_failed for r in survivors)
    argmin = [r for r in survivors if r.fp_failed == fewest]
    ordered = sorted(argmin, key=_tie_break_key)
    chosen = ordered[0]
    # A surviving tie among parameters NOT proved inert would mean the rule did not
    # decide — refuse rather than let sort order pick a published parameter value.
    still_tied = [
        r
        for r in ordered[1:]
        if _tie_break_key(r) == _tie_break_key(chosen)
        and r.params.as_dict() != chosen.params.as_dict()
    ]
    if still_tied:
        raise RecommendError(
            "the decision rule left "
            f"{len(still_tied) + 1} settings tied after TIE_BREAK_ORDER: "
            f"{[r.params.label for r in [chosen, *still_tied]]}"
        )
    return {
        "n_admissible_points": len(results),
        "n_clearing_every_floor": len(survivors),
        "n_at_the_measured_minimum": len(argmin),
        "chosen": chosen,
        "tied_with": [r.params.as_dict() for r in ordered[1:]],
    }


def floor_sensitivity(
    results: Sequence[PointResult],
    chosen: PointResult,
) -> dict[str, Any]:
    """Which floors CHANGE the answer, and which merely rule out an exact tie.

    A floor that never changes the selected tuple is not load-bearing evidence, and a
    report that lists four floors as if each earned its place overstates the derivation.
    Each floor is dropped **alone** and the rule re-run — the one-at-a-time form, because
    dropping them together would only show that the conjunction bites, which is the
    weaker claim ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    out: dict[str, Any] = {}
    for dropped in DECISION_FLOORS:
        kept = [f for f in DECISION_FLOORS if f.key != dropped.key]
        try:
            alternative = apply_decision_rule(results, floors=kept)["chosen"]
        except RecommendError as exc:
            # Dropping the floor leaves the rule UNABLE to decide. That is the floor's
            # contribution and it must be reported as such — picking whichever tuple the
            # grid happened to enumerate first would attribute a decision to the rule
            # that the rule refused to make.
            out[dropped.key] = {
                "selection_without_this_floor": None,
                "changes_the_selection": None,
                "measured_identically_to_the_chosen_setting": None,
                "the_rule_cannot_decide_without_it": str(exc),
            }
            continue
        same_tuple = alternative.params.as_dict() == chosen.params.as_dict()
        out[dropped.key] = {
            "selection_without_this_floor": alternative.params.as_dict(),
            "changes_the_selection": not same_tuple,
            # ⚠ null, not True, when the selection did not move: comparing the chosen
            # setting with itself is vacuously identical, and a TRUE that cannot be FALSE
            # reads in the report as evidence when it is the absence of any
            # ([[clauses-must-guard-emptiness]]).
            "measured_identically_to_the_chosen_setting": (
                None
                if same_tuple
                else (
                    alternative.fp_failed == chosen.fp_failed
                    and alternative.control["n_records_losing_a_and_b"]
                    == chosen.control["n_records_losing_a_and_b"]
                    and alternative.mined == chosen.mined
                )
            ),
        }
    return out


def frontier(
    results: Sequence[PointResult],
    *,
    only_floor_clearing: bool = False,
) -> list[dict[str, Any]]:
    """Per achievable yield, the admissible point that costs the fewest control records.

    Reported in full so the reader can see what the rule declines to take.  ⚠ These are
    argmins over thousands of points against 68 control records: they are what the grid
    *contains*, not settings this module recommends.

    ``only_floor_clearing`` restricts the frontier to the region the decision rule is
    allowed to choose from.  Both are emitted: the unrestricted one is what the grid
    offers, and the restricted one is the trade actually on the table — reporting only
    the first would advertise yields the rule cannot take, and only the second would
    hide the cost of the floors.
    """
    best: dict[int, PointResult] = {}
    counts: dict[int, int] = {}
    for r in results:
        if only_floor_clearing and not clears_all_floors(r.params):
            continue
        counts[r.mined] = counts.get(r.mined, 0) + 1
        incumbent = best.get(r.mined)
        if incumbent is None or (
            r.control["n_records_losing_a_and_b"],
            _tie_break_key(r),
        ) < (
            incumbent.control["n_records_losing_a_and_b"],
            _tie_break_key(incumbent),
        ):
            best[r.mined] = r
    return [
        {
            "n_mined": n,
            "n_admissible_points_at_this_yield": counts[n],
            "fewest_control_records_losing_a_and_b": best[n].control["n_records_losing_a_and_b"],
            "of_n_producible_records": best[n].control["n_records_producible"],
            "ci95": best[n].control["share_records_losing_a_and_b_ci95"],
            "clears_every_decision_floor": clears_all_floors(best[n].params),
            "example_params": best[n].params.as_dict(),
        }
        for n in sorted(best)
    ]


def dominating_alternatives(
    results: Sequence[PointResult],
    chosen: PointResult,
) -> dict[str, Any]:
    """Floor-clearing settings that mine MORE at no greater measured control cost.

    The rule selects on permissiveness, not on yield, so it will walk past settings that
    look strictly better on this pair of measurements.  Those settings are enumerated
    here rather than left for a reader to find: a recommendation that quietly declines a
    dominating alternative is asking to be trusted instead of checked.

    ⚠ "Dominating" is on **two measured coordinates over 2,880 points against 68
    records**, and the damage coordinate omits criterion (c) entirely.  A setting that
    dominates here is a candidate for being *fitted*, not a candidate for being right —
    which is the whole reason the rule is stated before the counts are read.
    """
    better = [
        r
        for r in results
        if clears_all_floors(r.params)
        and r.mined > chosen.mined
        and r.control["n_records_losing_a_and_b"] <= chosen.control["n_records_losing_a_and_b"]
    ]
    by_yield: dict[int, PointResult] = {}
    for r in better:
        incumbent = by_yield.get(r.mined)
        if incumbent is None or _tie_break_key(r) < _tie_break_key(incumbent):
            by_yield[r.mined] = r
    return {
        "n_settings": len(better),
        "max_extra_mined": (max(r.mined for r in better) - chosen.mined) if better else 0,
        "examples_by_yield": [
            {
                "params": by_yield[n].params.as_dict(),
                "n_mined": n,
                "n_extra_mined_vs_recommendation": n - chosen.mined,
                "control_records_losing_a_and_b": by_yield[n].control["n_records_losing_a_and_b"],
                "of_n_producible_records": by_yield[n].control["n_records_producible"],
            }
            for n in sorted(by_yield)
        ],
        "why_the_rule_declines_them": (
            "they are selected BY the two arms, on a control of 68 records, from thousands "
            "of points; the extra yield is a handful of candidates out of 941 against a "
            "ceiling of 15; and the damage coordinate they tie on excludes criterion (c), "
            "which is the only disjunct with measured discrimination. ADR-0006 A4 asks the "
            "round to supply a chosen value and record it — taking one of these would "
            "record a tuned one instead, and nothing downstream would say so."
        ),
    }


def assert_supply_is_the_decided_set(
    label: str,
    supply_ids: Iterable[str],
    status: Mapping[str, str],
) -> None:
    """The consensuses on disk must be **exactly** the ids criterion (a) decided.

    A size comparison would pass on two sets of the same cardinality that overlap
    partially, and every count downstream would still reconcile — the report would be
    internally flawless and about a different corpus
    ([[gate-must-bind-to-upstream-evidence]]).  A candidate present in the supply but
    ``unavailable`` in the status table, or decided in the table with no consensus on
    disk, is a producer/retrieval mismatch and is refused rather than defaulted.
    """
    decided = {cid for cid, state in status.items() if state in DECIDED_STATES}
    have = set(supply_ids)
    if decided == have:
        return
    missing, extra = sorted(decided - have), sorted(have - decided)
    raise RecommendError(
        f"the {label} supply is not criterion (a)'s decided set: {len(missing)} decided "
        f"with no consensus on disk (e.g. {missing[:2]}), {len(extra)} consensuses whose "
        f"candidate (a) did not decide (e.g. {extra[:2]})"
    )


def anti_selectivity(result: PointResult, *, n_fp_decided: int) -> dict[str, Any]:
    """Does the (a)∧(b) conjunction fall through MORE often on known T-boxes?

    (b)'s own failure share is what the two measurement reports compare, but the object
    that decides what is mined is the **conjunction**: a candidate reaches mining only
    with (a) failed too.  If the conjunction falls through more often on the control than
    on the FP arm, the composite is selecting *against* the class it exists to protect —
    which the per-criterion comparison cannot show.
    """
    ctrl_n = result.control["n_queries_decided"]
    ctrl_k = result.control["n_queries_a_and_b_failed"]
    fp_k = result.fp_failed_with_a_failed
    ctrl_lo, ctrl_hi = wilson_interval(ctrl_k, ctrl_n) if ctrl_n else (None, None)
    fp_lo, fp_hi = wilson_interval(fp_k, n_fp_decided) if n_fp_decided else (None, None)
    ctrl_share = ctrl_k / ctrl_n if ctrl_n else None
    fp_share = fp_k / n_fp_decided if n_fp_decided else None
    return {
        "control_queries_failing_a_and_b": f"{ctrl_k}/{ctrl_n}",
        "control_share": None if ctrl_share is None else round(ctrl_share, 6),
        "control_ci95": None if ctrl_lo is None else [round(ctrl_lo, 6), round(ctrl_hi, 6)],
        "fp_candidates_failing_a_and_b": f"{fp_k}/{n_fp_decided}",
        "fp_share": None if fp_share is None else round(fp_share, 6),
        "fp_ci95": None if fp_lo is None else [round(fp_lo, 6), round(fp_hi, 6)],
        "fp_minus_control_pp": (
            None
            if ctrl_share is None or fp_share is None
            else round(100 * (fp_share - ctrl_share), 4)
        ),
        "control_falls_through_more": (
            None if ctrl_share is None or fp_share is None else bool(ctrl_share > fp_share)
        ),
        "intervals_disjoint": (
            None if ctrl_lo is None or fp_lo is None else bool(ctrl_lo > fp_hi or fp_lo > ctrl_hi)
        ),
        "unit_caveat": (
            "the control's unit is a QUERY (pseudo-replicated across records) and the FP "
            "arm's is a candidate; the shares are like-for-like as 'share of decided', "
            "and the record-level interval is carried separately above."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The report
# ═════════════════════════════════════════════════════════════════════════════
def recommend(
    *,
    fp_msa_root: str | Path,
    control_msa_root: str | Path,
    fp_manifest_path: str | Path,
    spare_rule_inputs_path: str | Path,
    control_status_path: str | Path,
    fp_report_path: str | Path,
    control_report_path: str | Path,
    comparison_report_path: str | Path,
    supply_origin: str | None = None,
) -> dict[str, Any]:
    """The whole re-derivation, as the report body."""
    if supply_origin is not None and is_local_path_shaped(supply_origin):
        raise RecommendError(
            f"--supply-origin {supply_origin!r} looks like a local absolute path; this "
            "report is public — name the cluster origin, not a home directory"
        )
    fp_items = read_supply(fp_msa_root)
    control_items = read_supply(control_msa_root)
    fp_digest = supply_digest(fp_items)
    control_digest = supply_digest(control_items)

    # The supplies are cluster-only, so bind them to the committed reports by digest:
    # a re-run against a different round directory of the same cardinality would
    # otherwise produce a tidy table about the wrong corpus.
    fp_report = json.loads(Path(fp_report_path).read_text(encoding="utf-8"))
    control_report = json.loads(Path(control_report_path).read_text(encoding="utf-8"))
    comparison = json.loads(Path(comparison_report_path).read_text(encoding="utf-8"))
    for label, report, digest in (
        ("FP", fp_report, fp_digest),
        ("control", control_report, control_digest),
    ):
        recorded = report.get("supply", {}).get("supply_digest_sha256")
        if recorded != digest:
            raise RecommendError(
                f"the {label} supply on disk digests to {digest} but the committed "
                f"report records {recorded} — this is a different supply"
            )

    manifest = json.loads(Path(fp_manifest_path).read_text(encoding="utf-8"))
    manifest_rows = manifest.get("candidates", manifest)
    manifest_ids = [candidate_id_of(r) for r in manifest_rows]
    inputs = load_spare_rule_inputs(spare_rule_inputs_path)
    by_id: dict[str, Any] = inputs["by_id"]
    assert_covers_manifest(by_id, manifest_ids)

    fp_id_by_slug = {candidate_msa_path(fp_msa_root, cid).parent.name: cid for cid in manifest_ids}
    control_status = json.loads(Path(control_status_path).read_text(encoding="utf-8"))["status"]
    control_id_by_slug = {
        candidate_msa_path(control_msa_root, cid).parent.name: cid for cid in control_status
    }
    for label, items, index in (
        ("FP", fp_items, fp_id_by_slug),
        ("control", control_items, control_id_by_slug),
    ):
        orphans = sorted(i.slug for i in items if i.slug not in index)
        if orphans:
            raise RecommendError(
                f"{len(orphans)} {label} consensus slug(s) resolve to no manifest "
                f"candidate, e.g. {orphans[:2]}; the supply and the manifest disagree"
            )
    ids_with_consensus = [fp_id_by_slug[i.slug] for i in fp_items]
    control_record_index = control_records(control_status)
    assert_supply_is_the_decided_set(
        "FP",
        ids_with_consensus,
        {cid: str(row["covariation_status"]) for cid, row in by_id.items()},
    )
    assert_supply_is_the_decided_set(
        "control",
        (control_id_by_slug[i.slug] for i in control_items),
        control_status,
    )

    admissible: list[PointResult] = []
    refusals: dict[str, int] = {}
    for params in grid_points():
        outcome = evaluate_point(
            params,
            fp_items=fp_items,
            fp_id_by_slug=fp_id_by_slug,
            by_id=by_id,
            control_items=control_items,
            control_id_by_slug=control_id_by_slug,
            control_status=control_status,
            control_record_index=control_record_index,
        )
        if isinstance(outcome, str):
            reason = outcome.split(":", 1)[0]
            refusals[reason] = refusals.get(reason, 0) + 1
            continue
        admissible.append(outcome)
    if not admissible:
        raise RecommendError("the shipped localizer refused every grid point")

    decision = apply_decision_rule(admissible)
    chosen: PointResult = decision["chosen"]
    n_fp_decided = chosen.fp_failed + chosen.fp_passed
    by_params = {tuple(sorted(r.params.as_dict().items())): r for r in admissible}
    named: list[dict[str, Any]] = []
    for tup in default_tuples():
        key = tuple(sorted(tup.as_dict().items()))
        result = by_params.get(key)
        if result is None:  # pragma: no cover - the named six are all admissible
            continue
        named.append(
            {
                "label": tup.label,
                "clears_every_decision_floor": clears_all_floors(tup),
                "floors_cleared": list(floors_passed(tup)),
                **result.as_dict(),
                "anti_selectivity": anti_selectivity(result, n_fp_decided=n_fp_decided),
            }
        )

    ceiling = yield_ceiling(by_id, ids_with_consensus=ids_with_consensus)
    chosen_verdicts = sweep_arm(fp_items, chosen.params)
    if isinstance(chosen_verdicts, str):  # pragma: no cover - it was admissible above
        raise RecommendError(
            f"the chosen setting became inadmissible on re-evaluation: {chosen_verdicts}"
        )
    chosen_mined = mined_ids(
        by_id,
        {fp_id_by_slug[slug]: state for slug, state in chosen_verdicts.items()},
        stage2_threshold=None,
    )
    return {
        "adr": ADR,
        "step": STEP,
        "schema_version": SCHEMA_VERSION,
        "pins_nothing": True,
        "disclosure": (
            "This report RECOMMENDS criterion (b)'s seven ADR-0006 A4 rule parameters and "
            "pins none of them. The value is a CLAUDE.md §7 decision for the project lead; "
            "it enters the round through the producer's seven required exports and is "
            "recorded in that round's provenance, per A4. D6/D17's P6 threshold freeze is "
            "untouched, and no ADR is amended by this step."
        ),
        "decision_rule": {
            "statement": DECISION_RULE_STATEMENT,
            "floors": [f.as_dict() for f in DECISION_FLOORS],
            "tie_break_order": [{"parameter": k, "resolves_to": v} for k, v in TIE_BREAK_ORDER],
        },
        "arms": {
            "fp": {
                "step": "P3-15'-f",
                "n_consensuses": len(fp_items),
                "supply_digest_sha256": fp_digest,
                "supply_origin": fp_report.get("supply", {}).get("supply_origin"),
                "ground_truth": (
                    "unknown — a round-0 false-positive MANIFEST, not verified negatives"
                ),
            },
            "control": {
                "step": "P3-15'-g-iv",
                "n_consensuses": len(control_items),
                "supply_digest_sha256": control_digest,
                "supply_origin": control_report.get("supply", {}).get("supply_origin"),
                "ground_truth": (
                    "believed positive — held-out curated TBDB records, Stage-1 re-detected"
                ),
            },
            "comparison_headline_carried_forward": comparison.get("headline", {}).get("reading"),
        },
        "grid": {
            "n_points_enumerated": len(grid_points()),
            "n_admissible": len(admissible),
            "n_refused_by_the_shipped_localizer": sum(refusals.values()),
            "refusal_reasons": dict(sorted(refusals.items())),
            "axes": {
                "min_named_helices": list(MIN_NAMED_HELICES_SWEEP),
                "min_helix_pairs": list(MIN_HELIX_PAIRS_SWEEP),
                "bulge_min_nt": list(BULGE_MIN_NT_SWEEP),
                "bulge_max_nt": list(BULGE_MAX_NT_SWEEP),
                "ncca_pairing_nt": list(NCCA_PAIRING_NT_SWEEP),
                "allow_wobble": [False, True],
            },
            "note": (
                "admissibility is MEASURED by running the shipped localizer, not by "
                "re-encoding its guards: it refuses any setting whose bulge size floor is "
                "shorter than its NCCA register."
            ),
        },
        "yield_ceiling": ceiling,
        "named_settings": named,
        "frontier_all_admissible": frontier(admissible),
        "frontier_clearing_every_floor": frontier(admissible, only_floor_clearing=True),
        "recommendation": {
            "params": chosen.params.as_dict(),
            "mined_candidate_ids": chosen_mined,
            "matches_named_setting": next(
                (t.label for t in default_tuples() if t.as_dict() == chosen.params.as_dict()),
                None,
            ),
            "n_admissible_points": decision["n_admissible_points"],
            "n_clearing_every_floor": decision["n_clearing_every_floor"],
            "n_at_the_measured_minimum": decision["n_at_the_measured_minimum"],
            "tied_with": decision["tied_with"],
            "floor_sensitivity": floor_sensitivity(admissible, chosen),
            "dominating_alternatives": dominating_alternatives(admissible, chosen),
            "measured": chosen.as_dict(),
            "anti_selectivity": anti_selectivity(chosen, n_fp_decided=n_fp_decided),
            "what_it_costs": (
                "the frontier above shows what a higher-yield setting would buy and what it "
                "would cost in control records; the rule declines those because selecting "
                "them is fitting a value to 68 records, not choosing one."
            ),
        },
        "limitations": {
            "the_fp_arm_is_not_a_verified_negative_set": (
                "it is a false-positive MANIFEST of unknown status. If a material share of "
                "it is really T-box, the two arms are not two populations and the "
                "anti-selectivity reading weakens accordingly."
            ),
            "criterion_c_is_unmeasured_on_the_control": (
                "the control's hosts carry no annotation, so every control damage figure "
                "omits the one criterion with measured discrimination and is an upper bound."
            ),
            "the_grid_is_not_the_parameter_space": (
                "the axes are the two measurement reports' sweep; a value off those axes is "
                "unmeasured here, and stem_i_nt_threshold is swept at one value because it "
                "is proved inert on both manifests."
            ),
            "the_control_is_68_records": (
                "adjacent grid points differing by a few candidates are not distinguishable "
                "at this n; the frontier is reported as what the grid contains, not as a "
                "ranking."
            ),
            "stage2_threshold_is_unpinned": ceiling["stage2_threshold_note"],
        },
    }


def build_inputs(
    *,
    covariation_status_path: str | Path,
    synteny_status_path: str | Path,
    stage2_posteriors_path: str | Path,
    fp_manifest_path: str | Path,
) -> dict[str, Any]:
    """Join the three per-candidate spare-rule columns into one committable table.

    All three producers write to cluster scratch or to a working directory; none of them
    is in git.  The yield arithmetic this step turns on must be auditable without cluster
    access, so the join is committed — the same reason `P3-15′-g-iv` committed
    ``curated_control_status_v0.json``.
    """
    manifest = json.loads(Path(fp_manifest_path).read_text(encoding="utf-8"))
    manifest_ids = [candidate_id_of(r) for r in manifest.get("candidates", manifest)]
    cov = json.loads(Path(covariation_status_path).read_text(encoding="utf-8"))["status"]
    syn = json.loads(Path(synteny_status_path).read_text(encoding="utf-8"))["status"]
    post = json.loads(Path(stage2_posteriors_path).read_text(encoding="utf-8"))["posteriors"]
    for label, table in (("covariation", cov), ("synteny", syn), ("stage2", post)):
        missing = [cid for cid in manifest_ids if cid not in table]
        if missing:
            raise RecommendError(
                f"the {label} table is missing {len(missing)} manifest candidate(s), "
                f"e.g. {missing[:2]}"
            )
        extra = sorted(set(table) - set(manifest_ids))
        if extra:
            raise RecommendError(
                f"the {label} table carries {len(extra)} id(s) absent from the manifest, "
                f"e.g. {extra[:2]}"
            )
    rows = [
        {
            "candidate_id": cid,
            "covariation_status": str(cov[cid]),
            "synteny_status": str(syn[cid]),
            "stage2_posterior": float(post[cid]),
        }
        for cid in sorted(manifest_ids)
    ]
    return {
        "step": STEP,
        "schema_version": SCHEMA_VERSION,
        "adr": ADR,
        "n_candidates": len(rows),
        "rows": rows,
        "sources": {
            "covariation_status": {
                "sha256": sha256_of(covariation_status_path),
                "origin": "two.amlab:$HOME/tbox-scratch/round_p3_15_supply/covariation_status.json",
                "produced_by": "P3-15'-e / -e-ii (SLURM jobs 1205 + 1254)",
                "anchored_by": (
                    "reports/p3/architecture_parameter_measurement.json records this digest "
                    "as its covariation_status external input"
                ),
            },
            "synteny_status": {
                "sha256": sha256_of(synteny_status_path),
                "origin": "LOCAL — tbox_finder.mining.synteny_producer (P3-15'-c-ii)",
                "produced_by": "P3-15'-c-ii",
                "anchored_by": (
                    "reports/p3/synteny_exclusion_diagnostic.json records this digest as its "
                    "synteny_status external input"
                ),
            },
            "stage2_posteriors": {
                "sha256": sha256_of(stage2_posteriors_path),
                "origin": "two.amlab:$HOME/tbox-scratch/round_p3_15b/stage2_posteriors.json",
                "produced_by": "P3-15'-b",
                "anchored_by": None,
                "warning": (
                    "⚠ NO committed report records this digest, and a second scoring of the "
                    "same 941 candidates exists whose posteriors differ by up to 3.8e-2 (a "
                    "CUDA forward is not bit-reproducible). The two agree on every candidate "
                    "at thresholds 0.5 / 0.9 / 0.95 and disagree on 3 at 0.99. This column is "
                    "therefore carried as a SENSITIVITY only; no headline number rests on it."
                ),
            },
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="architecture_param_recommend",
        description=(
            "Re-derive criterion (b)'s seven ADR-0006 A4 rule parameters over the whole "
            "admissible grid, against the matched positive control and the round's real "
            "spare-rule conjunction. Pins nothing."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-inputs", help="join the three spare-rule columns")
    b.add_argument("--covariation-status", required=True)
    b.add_argument("--synteny-status", required=True)
    b.add_argument("--stage2-posteriors", required=True)
    b.add_argument("--fp-manifest", default="data/processed/mining/round0_fp_manifest.json")
    b.add_argument("--out", default="data/processed/mining/round0_fp_spare_rule_inputs_v0.json")

    r = sub.add_parser("recommend", help="write the recommendation report")
    r.add_argument("--fp-msa-root", required=True)
    r.add_argument("--control-msa-root", required=True)
    r.add_argument("--fp-manifest", default="data/processed/mining/round0_fp_manifest.json")
    r.add_argument(
        "--spare-rule-inputs",
        default="data/processed/mining/round0_fp_spare_rule_inputs_v0.json",
    )
    r.add_argument(
        "--control-status", default="data/processed/mining/curated_control_status_v0.json"
    )
    r.add_argument("--fp-report", default="reports/p3/architecture_parameter_measurement.json")
    r.add_argument(
        "--control-report",
        default="reports/p3/architecture_parameter_measurement_control.json",
    )
    r.add_argument(
        "--comparison-report",
        default="reports/p3/architecture_parameter_control_comparison.json",
    )
    r.add_argument("--supply-origin", default=None)
    r.add_argument("--out", default="reports/p3/architecture_parameter_recommendation.json")
    return parser


def _write(body: Mapping[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-inputs":
            body: dict[str, Any] = build_inputs(
                covariation_status_path=args.covariation_status,
                synteny_status_path=args.synteny_status,
                stage2_posteriors_path=args.stage2_posteriors,
                fp_manifest_path=args.fp_manifest,
            )
            _write(body, args.out)
            print(f"joined {body['n_candidates']} spare-rule input rows -> {args.out}")
            return 0
        body = recommend(
            fp_msa_root=args.fp_msa_root,
            control_msa_root=args.control_msa_root,
            fp_manifest_path=args.fp_manifest,
            spare_rule_inputs_path=args.spare_rule_inputs,
            control_status_path=args.control_status,
            fp_report_path=args.fp_report,
            control_report_path=args.control_report,
            comparison_report_path=args.comparison_report,
            supply_origin=args.supply_origin,
        )
    except (RecommendError, ValueError, TypeError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    try:
        repo_inputs: list[str] = []
        external: dict[str, Any] = {
            "fp_msa_root": portable_path(args.fp_msa_root),
            "control_msa_root": portable_path(args.control_msa_root),
            "supply_origin": args.supply_origin,
            "fp_supply_digest_sha256": body["arms"]["fp"]["supply_digest_sha256"],
            "control_supply_digest_sha256": body["arms"]["control"]["supply_digest_sha256"],
        }
        for label, candidate in (
            ("fp_manifest", args.fp_manifest),
            ("spare_rule_inputs", args.spare_rule_inputs),
            ("control_status", args.control_status),
            ("fp_report", args.fp_report),
            ("control_report", args.control_report),
            ("comparison_report", args.comparison_report),
        ):
            if not candidate or not Path(candidate).is_file():
                continue
            if is_inside_repo(candidate):
                repo_inputs.append(portable_path(candidate))
            else:
                external[label] = {"name": Path(candidate).name, "sha256": sha256_of(candidate)}
        body["provenance"] = build_provenance(
            rule="P3-15'-f criterion-(b) parameter re-derivation",
            script=portable_path(__file__),
            inputs=sorted(repo_inputs),
            outputs=[],
            adr=ADR,
            extra={"schema_version": SCHEMA_VERSION, "external_inputs": external},
        )
    except (RecommendError, ValueError, TypeError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    try:
        _write(body, args.out)
    except OSError as exc:
        print(f"refused: cannot write report to {portable_path(args.out)}: {exc}", file=sys.stderr)
        return 3
    rec = body["recommendation"]
    ceiling = body["yield_ceiling"]["by_stage2_threshold"]["stage2_not_declared"]
    print(
        f"{body['grid']['n_admissible']} admissible grid points; "
        f"{rec['n_clearing_every_floor']} clear every floor; recommendation "
        f"{rec['matches_named_setting'] or rec['params']} mines "
        f"{rec['measured']['round_yield_n_mined']['stage2_not_declared']} of a ceiling of "
        f"{ceiling['max_mined_if_b_failed_everywhere']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
