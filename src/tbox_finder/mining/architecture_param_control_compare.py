"""Read criterion (b)'s grid on the MATCHED positive control against the FP arm.

`P3-15′-f` measured (b)'s seven rule parameters on 278 de-novo consensuses of
**round-0 false-positive-manifest** candidates — a population of *unknown* status.
`P3-15′-g` argued, and `-g-ii`/`-g-iii` built, the control that measurement was
missing: Stage-1 re-detected spans on **held-out curated T-box records**, one per
ADR-0004 cluster, order-stratified, searched and aligned by the *same* instrument
(SLURM job 1264).  This module reads the two arms against each other.

What the comparison is for
==========================
(b) is a **sparing** disjunct (ADR-0006 D3; sparing is a disjunction, so a candidate
(b) ``passed`` is spared whatever (a) and (c) say).  Its parameters therefore trade
two costs against each other:

* loosen them and fewer FP candidates reach mining — lost yield;
* tighten them and (b) stops protecting **real T-boxes**, which then enter Stage-1
  training as hard negatives — label noise on the very class being learned.

Only the second cost needs a positive control, and until job 1264 the repo had
exactly one de-novo positive (``certified_positive.sto``, n = 1), which
`P3-15′-f` correctly refused to choose seven parameters from.  This module supplies
the rate that n = 1 could not.

Three disciplines, each of them load-bearing here
=================================================
* **The grid is identical by construction, and that is CHECKED.**  Both arms are
  written by :mod:`tbox_finder.mining.architecture_param_measure`, so "identical
  grid" is a property of the code — but a comparison handed the *wrong* report file
  would still produce a tidy table.  :func:`assert_grids_identical` re-derives the
  claim from the two reports' own ``sweep`` blocks and their pairwise ``joint``
  labels/parameters, and refuses rather than aligning them.
* **The per-candidate verdicts come from the shipped path.**  Record-level
  aggregation needs each query's own (b) state, which the reports publish only as
  counts, so this module calls
  :func:`~tbox_finder.mining.architecture_param_measure.candidate_state` — the
  function `evaluate_tuple` itself calls.  Recomputing the counts and checking them
  against the control report is an **artifact-binding** check (does this report
  describe this supply?), not independent validation of the arithmetic; it is
  labelled that way in the output.
* **Every interval is stated on the RECORD-level n.**  On the job-1264 supply this
  module was written for, 149 of the 160 control records contribute 2 queries and 4
  contribute 3, so the 317 queries are pseudo-replicated and a binomial interval on
  them would be too narrow by construction.  Those sizes are *this* prose only: the
  report itself derives every such number from the corpus it was handed.
  :func:`compare` refuses to emit a query-level CI at all — the
  query-level *share* is reported for the like-for-like point comparison against the
  FP arm's candidate-level share, and the uncertainty is carried by the record-level
  interval alone.

Run::

    PYTHONPATH=src python -m tbox_finder.mining.architecture_param_control_compare compare \\
        --control-msa-root <dir of <slug>/msa.sto from round_p3_15g_control> \\
        --out reports/p3/architecture_parameter_control_comparison.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder.mining.architecture_param_measure import (
    ParamTuple,
    SupplyItem,
    arm_for_step,
    candidate_state,
    default_tuples,
    is_inside_repo,
    is_local_path_shaped,
    portable_path,
    read_supply,
    sha256_of,
)
from tbox_finder.mining.covariation_producer import candidate_slug
from tbox_finder.mining.curated_control_sizing import wilson_interval
from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.provenance import build_provenance

SCHEMA_VERSION = "1.0"
STEP = "P3-15'-g-iv"
ADR = (
    "ADR-0006 D3 (criterion (b) as a sparing disjunct), A2 (min_sequences floor), "
    "A4 (rule parameters supplied per round, no defaults); ADR-0005 D14 "
    "(unavailable spares), A9 (the round-0 scanner), A10 (the producer envelope); "
    "ADR-0004 D3/D5 (clusters and the nested carve the control is drawn from)"
)

#: The two states a consensus can be in once it exists.  ``unavailable`` is not one
#: of them: it is the *absence* of a consensus and is spared under ADR-0005 D14.
DECIDED_STATES: tuple[str, ...] = ("passed", "failed")

#: 95 % two-sided.
Z_95 = 1.96


class CompareError(ValueError):
    """The two arms could not be read against each other as given."""


# ═════════════════════════════════════════════════════════════════════════════
# Binding the two arms to each other
# ═════════════════════════════════════════════════════════════════════════════
def assert_grids_identical(control_report: Mapping[str, Any], fp_report: Mapping[str, Any]) -> None:
    """Refuse unless the two reports really swept the same grid.

    "Identical by construction" is true of the *module* and says nothing about the
    two **files** a reader hands this tool.  A stale FP report, a report from a
    different sweep, or simply the wrong path would otherwise be tabulated
    side-by-side and every difference read as biology.

    Three things are checked, and each can fail alone: the ``sweep`` axes; the
    ``joint`` labels **in order**; and each pair's full seven-parameter dict.  The
    last is the one that matters — two reports can agree on labels while the tuple
    behind a label has changed, and then the comparison silently contrasts
    ``sensitive_core`` with something else.
    """
    for name, report in (("control", control_report), ("fp", fp_report)):
        if not isinstance(report.get("joint"), list) or not report["joint"]:
            raise CompareError(f"the {name} report carries no 'joint' rows")
        # ⚠ The SIBLING of `load_status`'s row-shape guard, and it was missing here.
        # `c_row.get("label")` on a bare string raises AttributeError, which `main`
        # does not catch — the same exit-1-with-a-traceback escape, through the same
        # kind of operator-supplied path ([[fixed-one-of-two-identical-things]]).
        non_rows = [i for i, row in enumerate(report["joint"]) if not isinstance(row, Mapping)]
        if non_rows:
            raise CompareError(
                f"the {name} report has {len(non_rows)} 'joint' entr(ies) that are not "
                f"objects, e.g. index {non_rows[:2]}"
            )
        if not isinstance(report.get("sweep"), Mapping) or not report["sweep"]:
            raise CompareError(f"the {name} report carries no 'sweep' block")
    if control_report["sweep"] != fp_report["sweep"]:
        differing = sorted(
            k
            for k in set(control_report["sweep"]) | set(fp_report["sweep"])
            if control_report["sweep"].get(k) != fp_report["sweep"].get(k)
        )
        raise CompareError(
            f"the two arms swept different grids; axes that differ: {differing}. "
            "The comparison is only meaningful at an identical grid"
        )
    c_joint, f_joint = control_report["joint"], fp_report["joint"]
    if len(c_joint) != len(f_joint):
        raise CompareError(
            f"the control report has {len(c_joint)} joint rows and the FP report "
            f"{len(f_joint)}; they are not the same named settings"
        )
    # strict: the length check above already refused a mismatch, so this can only
    # fire if that check is ever weakened — a silent truncation is the failure mode.
    for i, (c_row, f_row) in enumerate(zip(c_joint, f_joint, strict=True)):
        if c_row.get("label") != f_row.get("label"):
            raise CompareError(
                f"joint row {i} is {c_row.get('label')!r} in the control report and "
                f"{f_row.get('label')!r} in the FP report"
            )
        if c_row.get("params") != f_row.get("params"):
            raise CompareError(
                f"joint row {i} ({c_row.get('label')!r}) carries different parameters in "
                "the two reports; the same label would contrast two different settings"
            )


def supply_block(report: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    """A report's ``supply`` object, refused rather than tracebacked.

    ⚠ THE THIRD SITE OF ONE ESCAPE CLASS. `load_status` guards its rows and
    `assert_grids_identical` guards its `joint` entries, and both guards were added a
    review round apart because I fixed the reported site and not its sibling
    ([[fixed-one-of-two-identical-things]]).  Every `supply` access now goes through
    here and :func:`distribution_block`, so the class is closed at every site rather
    than at the one a reviewer happened to name.
    """
    supply = report.get("supply")
    if not isinstance(supply, Mapping):
        raise CompareError(f"the {arm} report carries no 'supply' object")
    return supply


def distribution_block(report: Mapping[str, Any], arm: str, name: str) -> dict[str, Any]:
    """One ``supply`` sub-distribution, minus its per-value ``counts`` histogram."""
    node = supply_block(report, arm).get(name)
    if not isinstance(node, Mapping):
        raise CompareError(f"the {arm} report's 'supply.{name}' is not an object")
    return {k: v for k, v in node.items() if k != "counts"}


def record_of(candidate_id: str) -> str:
    """The source **record** a control query belongs to.

    A control ``candidate_id`` is ``<record_sha256>:c0:<window>:<start>-<end>``
    (P3-15′-g-ii), so the record is the first two colon-separated fields — the
    manifest's own ``accession``.  This function is never used to *derive* the
    grouping: :func:`load_control_records` reads ``accession`` from the manifest and
    uses this only to refuse a row whose two statements about its own record
    disagree.  Deriving a grouping key from a formatted string is how one record
    silently becomes two ([[duplicate-key-merges-instead-of-colliding]]).
    """
    parts = str(candidate_id).split(":")
    if len(parts) < 2:
        raise CompareError(
            f"candidate_id {candidate_id!r} is not '<record>:<context>:...' shaped; "
            "the record it belongs to cannot be read from it"
        )
    return ":".join(parts[:2])


def load_control_records(manifest_path: str | Path) -> dict[str, list[str]]:
    """``accession → [candidate_id, …]`` for the control manifest.

    Refuses a duplicate ``candidate_id`` (two queries collapsing onto one would
    shrink every denominator while the arithmetic still reconciled) and refuses a
    row whose ``accession`` disagrees with the record encoded in its own
    ``candidate_id``.
    """
    payload = json.loads(Path(manifest_path).read_text())
    rows = payload.get("candidates") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise CompareError(f"control manifest {portable_path(manifest_path)} has no candidates")
    by_record: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise CompareError("a control manifest row is not an object")
        cid = row.get("candidate_id")
        accession = row.get("accession")
        if cid is None or accession is None:
            raise CompareError(
                "a control manifest row is missing 'candidate_id' or 'accession'; the "
                "record-level denominator cannot be formed"
            )
        cid, accession = str(cid), str(accession)
        if cid in seen:
            raise CompareError(
                f"duplicate candidate_id {cid!r} in the control manifest; two queries "
                "would merge and every record-level denominator would be understated"
            )
        seen.add(cid)
        if record_of(cid) != accession:
            raise CompareError(
                f"candidate_id {cid!r} encodes record {record_of(cid)!r} but its row says "
                f"accession {accession!r}; the two disagree about which record it is"
            )
        by_record.setdefault(accession, []).append(cid)
    return by_record


def load_status(status_path: str | Path) -> tuple[dict[str, str], list[Mapping[str, Any]]]:
    """The control round's ``candidate_id → status`` map and its per-query rows."""
    payload = json.loads(Path(status_path).read_text())
    if not isinstance(payload, Mapping):
        raise CompareError(f"control status {portable_path(status_path)} is not an object")
    status = payload.get("status")
    rows = payload.get("rows")
    if not isinstance(status, Mapping) or not status:
        raise CompareError(
            f"control status {portable_path(status_path)} carries no 'status' map; the "
            "producible share would be silently empty"
        )
    if not isinstance(rows, list) or not rows:
        raise CompareError(
            f"control status {portable_path(status_path)} carries no 'rows'; the depth "
            "distribution and the self-hit caveat are re-derived from them"
        )
    # ⚠ The ROW SHAPE too, not just that `rows` is a non-empty list. A bare string or a
    # row missing a key raises TypeError/KeyError inside the comprehension below and in
    # `self_hit_floor_caveat`, which reads `msa_depth` and `n_homologs` off the same
    # rows — and `main` catches neither as a refusal, so an operator-supplied
    # `--control-status` would exit 1 with a traceback instead of 3. The shape of a file
    # another process wrote is an input, not an invariant.
    required = ("candidate_id", "status", "n_homologs", "msa_depth")
    malformed = [
        i
        for i, r in enumerate(rows)
        if not isinstance(r, Mapping) or any(k not in r for k in required)
    ]
    if malformed:
        raise CompareError(
            f"control status {portable_path(status_path)}: {len(malformed)} row(s) are not "
            f"objects carrying {list(required)}, e.g. index {malformed[:2]}"
        )
    # ⚠ The VALUES, not only the keys. `self_hit_floor_caveat` calls `int()` on both
    # depth fields, and `int(None)` / `int([])` raises TypeError — the same escape the
    # key check above exists to close, one level down. `bool` is excluded explicitly:
    # it is an `int` subclass, so `True` would otherwise pass as a homolog count.
    non_integer = [
        i
        for i, r in enumerate(rows)
        if not all(
            isinstance(r[k], int) and not isinstance(r[k], bool)
            for k in ("n_homologs", "msa_depth")
        )
    ]
    if non_integer:
        raise CompareError(
            f"control status {portable_path(status_path)}: {len(non_integer)} row(s) carry "
            f"a non-integer 'n_homologs'/'msa_depth', e.g. index {non_integer[:2]}; the "
            "depth relation and the flip band cannot be derived from them"
        )
    # ⚠ Built as a LIST first: a dict comprehension over rows sharing a `candidate_id`
    # keeps the LAST one, so the map/rows agreement check below could still pass while
    # a query's real status was overwritten — and every count downstream would still
    # reconcile ([[duplicate-key-merges-instead-of-colliding]]).
    seen_rows: dict[str, str] = {}
    repeated: list[str] = []
    for r in rows:
        cid = str(r["candidate_id"])
        if cid in seen_rows:
            repeated.append(cid)
        seen_rows[cid] = str(r["status"])
    if repeated:
        raise CompareError(
            f"control status {portable_path(status_path)}: {len(repeated)} duplicate "
            f"candidate_id(s) in 'rows', e.g. {sorted(set(repeated))[:2]}; the later row "
            "would overwrite the earlier one's status"
        )
    row_status = seen_rows
    if row_status != {str(k): str(v) for k, v in status.items()}:
        raise CompareError(
            f"control status {portable_path(status_path)}: the 'status' map disagrees "
            "with its own 'rows'"
        )
    return {str(k): str(v) for k, v in status.items()}, list(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Producibility, and the one caveat that cuts against the control
# ═════════════════════════════════════════════════════════════════════════════
def self_hit_floor_caveat(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Size the control's structural handicap **from the control's own rows**.

    An FP candidate is carved from an assembly that is itself in the searched
    database, so its query self-hits and its alignment is one sequence deeper than
    the true homolog set.  A curated query is not in that database, so at the *same*
    homolog set the control's alignment is one sequence shallower — and ADR-0006 A2
    floors the alignment at ``min_sequences``.  The control's producible share is
    therefore biased **downward**.

    ⚠ P3-15′-g-iii recorded that I had first sized this bias from the **FP** arm's
    median depth (20.0, exactly the floor) and called it "concentrated precisely
    where the cut falls".  The control's own distribution refutes that
    ([[caveat-size-must-come-from-its-own-arm]]): this function re-derives the size
    from these rows, and only from these rows.

    The arithmetic: ``msa_depth == n_homologs + 1`` is *verified* here rather than
    assumed, which is what makes the floor a floor on depth-with-query; a query that
    self-hit would reach ``n_homologs + 2``, so the queries that would flip are
    exactly the ``unavailable`` ones at ``n_homologs == min_sequences - 2``.
    """
    produced = [r for r in rows if str(r["status"]) in DECIDED_STATES]
    if not produced:
        raise CompareError("no produced rows: the depth relation cannot be verified")
    depth_is_homologs_plus_one = all(
        int(r["msa_depth"]) == int(r["n_homologs"]) + 1 for r in produced
    )
    flip_band = MIN_REAL_HOMOLOG_N - 2
    would_flip = [
        r
        for r in rows
        if str(r["status"]) not in DECIDED_STATES and int(r["n_homologs"]) == flip_band
    ]
    n_total = len(rows)
    return {
        "min_sequences_floor": MIN_REAL_HOMOLOG_N,
        "msa_depth_equals_n_homologs_plus_one": depth_is_homologs_plus_one,
        # ⚠ Emitted so a reader can see the claim is conditional: if the relation ever
        # failed, the flip band below is not derivable and the caveat is unsized.
        "flip_band_n_homologs": flip_band if depth_is_homologs_plus_one else None,
        "n_queries_that_would_flip": len(would_flip) if depth_is_homologs_plus_one else None,
        "pp_of_all_queries": (
            round(100.0 * len(would_flip) / n_total, 4) if depth_is_homologs_plus_one else None
        ),
        "n_homologs_median": statistics.median(int(r["n_homologs"]) for r in rows),
        "share_below_floor_by_n_homologs": round(
            sum(1 for r in rows if int(r["n_homologs"]) < MIN_REAL_HOMOLOG_N) / n_total, 6
        ),
        "direction": "downward — the control's producible share is a LOWER bound",
        "note": (
            "a curated query is absent from the searched database and so does not "
            "self-hit; at the same true homolog set the control's alignment is one "
            "sequence shallower than the FP arm's. Not corrected — the floor is a "
            "pinned ADR-0006 A2 value — and sized here from the control's OWN rows "
            "rather than from the FP arm's depth distribution."
        ),
    }


def producibility(
    *,
    by_record: Mapping[str, Sequence[str]],
    status: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    fp_report: Mapping[str, Any],
) -> dict[str, Any]:
    """What share of each arm the instrument could produce a consensus for at all."""
    decided = {cid for cid, s in status.items() if s in DECIDED_STATES}
    producible_records = {acc for acc, cids in by_record.items() if any(c in decided for c in cids)}
    fp_supply = supply_block(fp_report, "fp")
    n_fp_manifest = int(fp_supply["n_candidates_in_manifest"])
    n_fp_measured = int(fp_supply["n_consensuses_measured"])
    # ⚠ Refused, not divided: an FP report with no candidates is not a comparator, and
    # dividing would raise ZeroDivisionError — which `main` does not catch, so the CLI
    # would exit 1 with a traceback while every other bad input exits 3 with a refusal.
    if n_fp_manifest <= 0:
        raise CompareError(
            "the FP report's manifest carries no candidates; there is no producible "
            "share to compare the control against"
        )
    return {
        "control_query_level": {
            "n_queries": len(status),
            "n_producible": len(decided),
            "share": round(len(decided) / len(status), 6),
            "counts": {k: v for k, v in sorted(Counter(status.values()).items())},
        },
        "control_record_level": {
            "n_records": len(by_record),
            "n_records_with_a_producible_query": len(producible_records),
            "share": round(len(producible_records) / len(by_record), 6),
            "queries_per_record": {
                str(k): v for k, v in sorted(Counter(len(v) for v in by_record.values()).items())
            },
        },
        "fp_candidate_level": {
            "n_candidates": n_fp_manifest,
            "n_producible": n_fp_measured,
            "share": round(n_fp_measured / n_fp_manifest, 6),
        },
        "spared_unavailable": {
            "n_control_queries": len(status) - len(decided),
            "rule": (
                "ADR-0005 D14: a candidate with no consensus resolves 'unavailable' and "
                "is SPARED, so it never reaches mining and leaves every (b) denominator"
            ),
        },
        "self_hit_floor_caveat": self_hit_floor_caveat(rows),
        "reading": (
            "the control's query-level producible share is compared with the FP arm's "
            "candidate-level share: these are the units the two arms share. The "
            "record-level share is the one the intervals below are stated on."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The per-tuple comparison
# ═════════════════════════════════════════════════════════════════════════════
def _share_and_ci(successes: int, n: int) -> dict[str, Any]:
    """A share with its 95 % Wilson interval, or an explicit ``None`` at n = 0."""
    if n <= 0:
        return {"n": 0, "k": int(successes), "share": None, "ci95": None}
    lo, hi = wilson_interval(successes, n, z=Z_95)
    return {
        "n": int(n),
        "k": int(successes),
        "share": round(successes / n, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
    }


def control_states(
    items: Sequence[SupplyItem],
    slug_to_candidate: Mapping[str, str],
    params: ParamTuple,
) -> dict[str, str]:
    """``candidate_id → (b) state`` for every produced control consensus at ``params``."""
    out: dict[str, str] = {}
    for item in items:
        cid = slug_to_candidate.get(item.slug)
        if cid is None:  # pragma: no cover - refused up front by compare()
            raise CompareError(f"consensus dir {item.slug!r} matches no control candidate")
        out[cid] = candidate_state(item, params)
    return out


def stratify_by_criterion_a(
    control_row: Mapping[str, Any], fp_row: Mapping[str, Any]
) -> dict[str, Any] | None:
    """(b)'s failure share **within each criterion-(a) stratum**, both arms.

    The confound this closes: the two arms do not have the same (a) composition —
    (a) passes 30/76 of the control against 167/278 of the FP arm — and (a) and (b)
    read the same alignment, so they are correlated.  An unstratified similarity
    between the arms could therefore be a composition artifact rather than a
    statement about (b), and the arms' (b) shares could differ *within* every
    stratum while agreeing overall (Simpson's paradox).  Reported so the reading
    does not have to assume otherwise.

    ``None`` when either report was written without a covariation status, because
    then the stratification does not exist rather than being empty.
    """
    # ⚠ Coerced rather than returned on: a non-Mapping here is a type question, and
    # ONE place decides "absent ⇒ None" — the `if not out` below. An early return
    # beside it would be a branch no input can reach on its own, i.e. a guard that
    # cannot be shown to bite ([[pinned-constant-that-nothing-reads]]).
    c_by = control_row.get("by_covariation_status")
    f_by = fp_row.get("by_covariation_status")
    c_by = c_by if isinstance(c_by, Mapping) else {}
    f_by = f_by if isinstance(f_by, Mapping) else {}
    out: dict[str, Any] = {}
    for stratum in DECIDED_STATES:
        c_arm, f_arm = c_by.get(stratum), f_by.get(stratum)
        if not isinstance(c_arm, Mapping) or not isinstance(f_arm, Mapping):
            continue
        c_n, f_n = int(c_arm.get("n", 0)), int(f_arm.get("n", 0))
        c_share = round(int(c_arm["failed"]) / c_n, 6) if c_n else None
        f_share = round(int(f_arm["failed"]) / f_n, 6) if f_n else None
        out[f"criterion_a_{stratum}"] = {
            "control": {"failed": int(c_arm["failed"]), "n": c_n, "share_failed": c_share},
            "fp": {"failed": int(f_arm["failed"]), "n": f_n, "share_failed": f_share},
            "fp_minus_control": (
                round(f_share - c_share, 6) if c_share is not None and f_share is not None else None
            ),
        }
    if not out:
        return None
    out["reading"] = (
        "if the arms' (b) failure shares stay close INSIDE each (a) stratum, the "
        "overall similarity is not an artifact of the two arms having different (a) "
        "compositions. ⚠ (a) is not ground truth on either arm, and on the control it "
        "is a measurement of (a)'s own sensitivity on known T-boxes."
    )
    return out


def compare_tuple(
    *,
    params: ParamTuple,
    label: str,
    states: Mapping[str, str],
    by_record: Mapping[str, Sequence[str]],
    fp_row: Mapping[str, Any],
    control_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One named setting, both arms, with the interval on the record-level n.

    The record rule is **ANY**: a record is spared by (b) if at least one of its
    producible queries passes.  Each query is its own mining candidate and sparing
    is a disjunction, so a record whose element the instrument recognises *once* is
    protected; requiring all of them would measure the detector's consistency across
    its own overlapping calls rather than (b)'s sensitivity.  The strict **ALL**
    variant is reported beside it so the choice is visible rather than buried.
    """
    # ⚠ Derived, never typed. The first draft of this string said "149 of the 160
    # control records contribute 2 queries and 4 contribute 3" — true of the one supply
    # this module was first run on and false on every other, including its own test
    # fixture. A report that states a corpus it did not measure is the same defect as a
    # report that describes the wrong arm, and nothing internal can see either.
    n_records_total = len(by_record)
    n_queries_total = sum(len(cids) for cids in by_record.values())
    n_multi_query_records = sum(1 for cids in by_record.values() if len(cids) > 1)

    q_passed = sum(1 for s in states.values() if s == "passed")
    q_failed = sum(1 for s in states.values() if s == "failed")
    n_decided_q = q_passed + q_failed

    any_pass = all_pass = n_records_producible = 0
    for cids in by_record.values():
        decided = [states[c] for c in cids if c in states]
        if not decided:
            continue
        n_records_producible += 1
        if any(s == "passed" for s in decided):
            any_pass += 1
        if all(s == "passed" for s in decided):
            all_pass += 1

    fp_counts = fp_row["counts"]
    fp_decided = int(fp_counts["passed"]) + int(fp_counts["failed"])
    fp_fail = _share_and_ci(int(fp_counts["failed"]), fp_decided)

    # The two point estimates the arms genuinely share a unit for.
    control_query_fail_share = round(q_failed / n_decided_q, 6) if n_decided_q else None
    # ⚠ The record-level FAILURE share, i.e. the records (b) would hand to mining.
    record_fail = _share_and_ci(n_records_producible - any_pass, n_records_producible)

    contains = None
    if fp_fail["share"] is not None and record_fail["ci95"] is not None:
        contains = bool(record_fail["ci95"][0] <= fp_fail["share"] <= record_fail["ci95"][1])
    difference = None
    if fp_fail["share"] is not None and control_query_fail_share is not None:
        difference = round(fp_fail["share"] - control_query_fail_share, 6)

    return {
        "label": label,
        "params": params.as_dict(),
        "control_query_level": {
            "n_decided": n_decided_q,
            "passed": q_passed,
            "failed": q_failed,
            "share_failed": control_query_fail_share,
            "ci95": None,
            "why_no_ci": (
                f"{n_multi_query_records} of the {n_records_total} control records "
                f"contribute more than one query ({n_queries_total} queries in all), so "
                "the queries are pseudo-replicated; a binomial interval on them would be "
                "narrower than the data support. The interval is on the record level."
            ),
        },
        "control_record_level": {
            "rule": "ANY producible query of the record passes (b) => the record is spared",
            "n_records_producible": n_records_producible,
            "n_spared_any": any_pass,
            "n_spared_all": all_pass,
            "n_mined": n_records_producible - any_pass,
            "share_mined": record_fail["share"],
            "share_mined_ci95": record_fail["ci95"],
            "share_spared_all_variant": (
                round(all_pass / n_records_producible, 6) if n_records_producible else None
            ),
        },
        "fp_candidate_level": {
            "n_decided": fp_decided,
            "passed": int(fp_counts["passed"]),
            "failed": int(fp_counts["failed"]),
            "share_failed": fp_fail["share"],
            "ci95": fp_fail["ci95"],
            "ci_caveat": (
                "the FP arm's candidates are carved from far fewer source assemblies "
                "than there are candidates (P3-15'-g), so this interval is optimistic "
                "too; it is shown for scale, not for a test"
            ),
        },
        "stratified_by_criterion_a": (
            stratify_by_criterion_a(control_row, fp_row) if control_row is not None else None
        ),
        "discrimination": {
            "fp_minus_control_query_share_failed": difference,
            "control_record_ci_contains_fp_point": contains,
            "reading": (
                "(b) is a SPARING disjunct: 'failed' on the control means a known T-box "
                "loses (b)'s protection, 'failed' on the FP arm means a candidate becomes "
                "minable. A setting separates the two populations only if it fails the FP "
                "arm materially more often than the control."
            ),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# The report
# ═════════════════════════════════════════════════════════════════════════════
def compare(
    *,
    control_msa_root: str | Path,
    control_manifest_path: str | Path,
    control_status_path: str | Path,
    control_report_path: str | Path,
    fp_report_path: str | Path,
    detect_report_path: str | Path | None = None,
    supply_origin: str | None = None,
    tuples: Sequence[ParamTuple] | None = None,
) -> dict[str, Any]:
    """The whole comparison, as the report body (no provenance — ``main`` adds it)."""
    if supply_origin is not None and is_local_path_shaped(supply_origin):
        raise CompareError(
            f"supply_origin {supply_origin!r} is a local absolute path; it is recorded "
            "verbatim in a PUBLIC report — name the host and use $HOME"
        )
    control_report = json.loads(Path(control_report_path).read_text())
    fp_report = json.loads(Path(fp_report_path).read_text())
    if not isinstance(control_report, Mapping) or not isinstance(fp_report, Mapping):
        raise CompareError("a measurement report is not a JSON object")
    assert_grids_identical(control_report, fp_report)
    # ⚠ Each report's `ground_truth` is READ OFF the arm that wrote it, never restated
    # here: a second spelling of "is this supply believed positive" is a statement that
    # can disagree with the report it describes, and `SupplyArm.ground_truth` would
    # otherwise be a constant nothing reads ([[pinned-constant-that-nothing-reads]]).
    # It doubles as a binding: a report whose `step` no arm declares is refused.
    control_arm = arm_for_step(str(control_report.get("step")))
    fp_arm = arm_for_step(str(fp_report.get("step")))
    if control_arm.ground_truth == fp_arm.ground_truth:
        raise CompareError(
            f"both reports declare ground_truth {control_arm.ground_truth!r} "
            f"(steps {control_arm.step!r} and {fp_arm.step!r}); this is not a control "
            "read against a comparator, it is one arm read against itself"
        )

    by_record = load_control_records(control_manifest_path)
    status, rows = load_status(control_status_path)
    manifest_ids = {cid for cids in by_record.values() for cid in cids}
    if set(status) != manifest_ids:
        only_status = len(set(status) - manifest_ids)
        only_manifest = len(manifest_ids - set(status))
        raise CompareError(
            f"the control status table and the control manifest are not the same corpus "
            f"({only_status} id(s) only in the status table, {only_manifest} only in the "
            "manifest); every denominator below would be drawn from two different sets"
        )

    items = read_supply(control_msa_root)
    slug_to_candidate = {candidate_slug(cid): cid for cid in manifest_ids}
    if len(slug_to_candidate) != len(manifest_ids):
        raise CompareError(
            "two control candidate ids share a slug; one consensus directory would "
            "stand for two queries"
        )
    unknown = sorted(i.slug for i in items if i.slug not in slug_to_candidate)
    if unknown:
        raise CompareError(
            f"{len(unknown)} consensus dir(s) under the control msa-root match no control "
            f"candidate, e.g. {unknown[:2]}; this is not the control's supply"
        )
    # ⚠ The supply and criterion (a)'s decided set must be the SAME set, not merely the
    # same size: (b) reads the alignment (a) read (ADR-0006 A4). A mismatch means the
    # producible share and the (b) rates are being computed over different corpora.
    decided_ids = {cid for cid, s in status.items() if s in DECIDED_STATES}
    supply_ids = {slug_to_candidate[i.slug] for i in items}
    if supply_ids != decided_ids:
        raise CompareError(
            f"the control supply on disk ({len(supply_ids)} consensuses) is not the set "
            f"criterion (a) decided ({len(decided_ids)}); the two arms of the round "
            "disagree about which queries produced an alignment"
        )

    param_tuples = tuple(tuples) if tuples is not None else default_tuples()
    labels_in_report = [row["label"] for row in control_report["joint"]]
    if [p.label for p in param_tuples] != labels_in_report:
        raise CompareError(
            "the named settings this comparison sweeps are not the ones the control "
            f"report published ({[p.label for p in param_tuples]} vs {labels_in_report}); "
            "the table would contrast settings the reports never measured"
        )

    per_tuple: list[dict[str, Any]] = []
    binding: list[dict[str, Any]] = []
    for params, c_row, f_row in zip(
        param_tuples, control_report["joint"], fp_report["joint"], strict=True
    ):
        states = control_states(items, slug_to_candidate, params)
        row = compare_tuple(
            params=params,
            label=params.label,
            states=states,
            by_record=by_record,
            fp_row=f_row,
            control_row=c_row,
        )
        # Artifact binding, NOT independent validation: `candidate_state` is the very
        # function the control report's counts came from, so agreement is expected. What
        # it can catch is a --control-report that does not describe --control-msa-root
        # ([[gate-must-bind-to-upstream-evidence]]).
        binding.append(
            {
                "label": params.label,
                "report_counts": {k: int(c_row["counts"][k]) for k in ("passed", "failed")},
                "recomputed_counts": {
                    "passed": row["control_query_level"]["passed"],
                    "failed": row["control_query_level"]["failed"],
                },
                "agrees": (
                    int(c_row["counts"]["passed"]) == row["control_query_level"]["passed"]
                    and int(c_row["counts"]["failed"]) == row["control_query_level"]["failed"]
                ),
            }
        )
        per_tuple.append(row)
    disagreeing = [b["label"] for b in binding if not b["agrees"]]
    if disagreeing:
        raise CompareError(
            f"the control report's counts disagree with the supply at {disagreeing}; "
            "--control-report does not describe --control-msa-root"
        )

    prod = producibility(by_record=by_record, status=status, rows=rows, fp_report=fp_report)
    detect = load_detect(detect_report_path)
    matchedness = detect.get("matchedness_vs_fp_candidates") if detect else None

    n_no_separation = sum(
        1 for r in per_tuple if r["discrimination"]["control_record_ci_contains_fp_point"]
    )
    headline = {
        "n_settings": len(per_tuple),
        "n_settings_where_control_ci_contains_the_fp_point": n_no_separation,
        "max_abs_fp_minus_control_query_share_failed": max(
            (
                abs(r["discrimination"]["fp_minus_control_query_share_failed"])
                for r in per_tuple
                if r["discrimination"]["fp_minus_control_query_share_failed"] is not None
            ),
            default=None,
        ),
        "n_settings_where_control_fails_more_than_fp": sum(
            1
            for r in per_tuple
            if (r["discrimination"]["fp_minus_control_query_share_failed"] or 0) < 0
        ),
        "reading": (
            "criterion (b)'s failure share on KNOWN T-boxes, as the de-novo instrument "
            "resolves them, is read against its failure share on round-0 FP-manifest "
            "candidates at the identical grid. Where the control's record-level interval "
            "contains the FP arm's point estimate, this control detects no separation "
            "between the two populations at that setting — (b) behaves there as a "
            "severity knob on both arms rather than as a discriminator."
        ),
    }

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "pins_nothing": True,
        "disclosure": (
            "the matched de-novo positive control (P3-15'-g-ii draw, P3-15'-g-iii run, "
            "SLURM job 1264) read against the P3-15'-f FP arm at an identical grid. Both "
            "arms are measured by architecture_param_measure; this module adds only the "
            "record-level aggregation, the intervals and the two-arm contrast. It pins "
            "no parameter: the ADR-0006 A4 seven-parameter choice remains the open §7 "
            "decision P3-15'-f deferred."
        ),
        "arms": {
            "control": {
                "step": control_report.get("step"),
                "report": portable_path(control_report_path),
                "msa_root": portable_path(control_msa_root),
                "supply_origin": supply_origin,
                "supply_digest_sha256": supply_block(control_report, "control")[
                    "supply_digest_sha256"
                ],
                "n_consensuses": int(
                    supply_block(control_report, "control")["n_consensuses_measured"]
                ),
                "ground_truth": control_arm.ground_truth,
            },
            "fp": {
                "step": fp_report.get("step"),
                "report": portable_path(fp_report_path),
                "supply_origin": supply_block(fp_report, "fp").get("supply_origin"),
                "supply_digest_sha256": supply_block(fp_report, "fp")["supply_digest_sha256"],
                "n_consensuses": int(supply_block(fp_report, "fp")["n_consensuses_measured"]),
                "ground_truth": fp_arm.ground_truth,
            },
            "grid_identical": True,
            "grid_identical_checked_by": (
                "assert_grids_identical: the two reports' 'sweep' blocks, their 'joint' "
                "labels in order, and each pair's full seven-parameter dict"
            ),
            "report_binds_to_supply": binding,
        },
        "alignment_comparability": {
            "control_depth": distribution_block(control_report, "control", "alignment_depth"),
            "fp_depth": distribution_block(fp_report, "fp", "alignment_depth"),
            "control_width": distribution_block(control_report, "control", "consensus_width"),
            "fp_width": distribution_block(fp_report, "fp", "consensus_width"),
            "query_matchedness_from_the_detect_report": (
                {
                    k: matchedness[k]
                    for k in (
                        "ks_d",
                        "median_ratio_curated_over_fp",
                        "share_curated_inside_fp_range",
                        "baseline_raw_curated_ks_d",
                    )
                    if k in matchedness
                }
                if isinstance(matchedness, Mapping)
                else None
            ),
            "note": (
                "P3-15'-g-ii matched the two arms at the QUERY; these are the ALIGNMENTS "
                "those queries produced, which is what the localizer actually reads. "
                "Matchedness at the query does not guarantee it here, so it is reported "
                "rather than assumed — and it is also the one limitation the detect "
                "report left open (whether producibility correlates with span length, so "
                "that the produced subset differs from the population entering)."
            ),
        },
        "producibility": prod,
        "by_parameter_tuple": per_tuple,
        "headline": headline,
        "limitations": limitations(detect_report_path, detect, prod),
    }
    return body


def load_detect(detect_report_path: str | Path | None) -> Mapping[str, Any] | None:
    """`P3-15′-g-ii`'s detect report, or ``None`` if it was not supplied.

    Read rather than restated: the Stage-1 dropout and the query/FP matchedness are
    that leg's measurements, and a number retyped here would go stale the moment the
    draw is re-run without anything in this report showing it.
    """
    if detect_report_path is None or not Path(detect_report_path).is_file():
        return None
    payload = json.loads(Path(detect_report_path).read_text())
    if not isinstance(payload, Mapping):
        raise CompareError(f"detect report {portable_path(detect_report_path)} is not an object")
    return payload


def limitations(
    detect_report_path: str | Path | None,
    detect: Mapping[str, Any] | None,
    prod: Mapping[str, Any],
) -> dict[str, Any]:
    """Everything that bounds how far this rate may be carried.

    ``prod`` is threaded in so the one sentence here that quotes corpus sizes reads
    them from :func:`producibility`'s own output rather than restating them.
    """
    q_level = prod["control_query_level"]
    r_level = prod["control_record_level"]
    n_unproducible_records = r_level["n_records"] - r_level["n_records_with_a_producible_query"]
    out: dict[str, Any] = {
        "the_query_is_the_detectors_call": (
            "the record is a known T-box; the query is Stage-1's predicted span on it "
            "(P3-15'-g-ii). Every rate here is (b)'s verdict on a PARTIAL element — which "
            "is exactly what an FP candidate is, and why the arms are matched — but it is "
            "not a rate on the curated element."
        ),
        "not_a_two_sample_test": (
            "no p-value is computed. Both arms are pseudo-replicated (the control by "
            "record, the FP arm by source assembly), so a nominal two-proportion test "
            "would be anticonservative; the report states intervals and lets the reader "
            "see the overlap."
        ),
        "fp_arm_status_is_unknown_not_negative": (
            "the FP arm is a round-0 false-positive MANIFEST, not a verified-negative "
            "set. If a material share of it is really T-box, the two arms are not two "
            "populations and equal failure shares are the expected result rather than a "
            "finding about (b). This is the single largest threat to the reading above."
        ),
        "the_control_measures_the_instrument_not_the_biology": (
            "a control record that resolves no consensus "
            f"({q_level['n_queries'] - q_level['n_producible']} of {q_level['n_queries']} "
            f"queries, {n_unproducible_records} of {r_level['n_records']} records) is "
            "spared under ADR-0005 D14 and leaves the denominator; the rates are "
            "conditional on the instrument having produced an alignment at all."
        ),
    }
    if detect is not None:
        dropout = detect.get("dropout")
        out["stage1_dropout"] = {
            "source": portable_path(detect_report_path) if detect_report_path else None,
            "note": (
                "records where Stage-1 did not fire never entered the run, so the "
                "control's record denominator is already conditioned on detection. Read "
                "the other way, the same scan is Stage-1's per-locus recall on held-out "
                "curated records."
            ),
            **(
                {
                    k: dropout[k]
                    for k in (
                        "n_windows_scanned",
                        "n_records_with_a_query",
                        "n_records_dropped",
                        "dropout_share",
                    )
                    if k in dropout
                }
                if isinstance(dropout, Mapping)
                else {}
            ),
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="architecture_param_control_compare",
        description=(
            "Read criterion (b)'s grid on the matched de-novo positive control against "
            "the P3-15'-f FP arm at an identical grid."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="write the two-arm comparison report")
    c.add_argument(
        "--control-msa-root",
        required=True,
        help="directory of <slug>/msa.sto from round_p3_15g_control",
    )
    c.add_argument(
        "--control-manifest",
        default="data/processed/mining/curated_control_manifest_v0.json",
    )
    c.add_argument(
        "--control-status",
        default="data/processed/mining/curated_control_status_v0.json",
    )
    c.add_argument(
        "--control-report",
        default="reports/p3/architecture_parameter_measurement_control.json",
    )
    c.add_argument(
        "--fp-report",
        default="reports/p3/architecture_parameter_measurement.json",
    )
    c.add_argument(
        "--detect-report",
        default="reports/p3/curated_control_detect.json",
        help="P3-15'-g-ii's detect report; the Stage-1 dropout is read from it",
    )
    c.add_argument(
        "--supply-origin",
        default=None,
        help="free text naming where the control consensuses were PRODUCED",
    )
    c.add_argument(
        "--out",
        default="reports/p3/architecture_parameter_control_comparison.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "compare":  # pragma: no cover - argparse enforces
        raise CompareError(f"unknown command {args.command!r}")
    try:
        body = compare(
            control_msa_root=args.control_msa_root,
            control_manifest_path=args.control_manifest,
            control_status_path=args.control_status,
            control_report_path=args.control_report,
            fp_report_path=args.fp_report,
            detect_report_path=args.detect_report,
            supply_origin=args.supply_origin,
        )
    # Same convention as architecture_param_measure.main: every operator-supplied path
    # refuses with exit 3 rather than exiting 1 with a traceback.
    except (CompareError, ValueError, TypeError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    # ⚠ Inside a refusal too. `sha256_of` READS each external input and
    # `build_provenance` hashes the repo-relative ones, and `Path(...).is_file()`
    # above only proves the entry existed a moment ago — an unreadable file, or one
    # removed between the check and the read, raises OSError here. Outside a `try`
    # that exits 1 with a traceback, while every other bad input exits 3.
    try:
        repo_inputs: list[str] = []
        external: dict[str, Any] = {
            "control_msa_root": portable_path(args.control_msa_root),
            "supply_origin": args.supply_origin,
            "control_supply_digest_sha256": body["arms"]["control"]["supply_digest_sha256"],
            "fp_supply_digest_sha256": body["arms"]["fp"]["supply_digest_sha256"],
        }
        for label, candidate in (
            ("control_manifest", args.control_manifest),
            ("control_status", args.control_status),
            ("control_report", args.control_report),
            ("fp_report", args.fp_report),
            ("detect_report", args.detect_report),
        ):
            if not candidate or not Path(candidate).is_file():
                continue
            if is_inside_repo(candidate):
                repo_inputs.append(portable_path(candidate))
            else:
                external[label] = {"name": Path(candidate).name, "sha256": sha256_of(candidate)}
        body["provenance"] = build_provenance(
            rule="P3-15'-g-iv matched-control comparison",
            script=portable_path(__file__),
            inputs=sorted(repo_inputs),
            outputs=[],
            adr=ADR,
            extra={"schema_version": SCHEMA_VERSION, "external_inputs": external},
        )
    except (CompareError, ValueError, TypeError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"refused: cannot write report to {portable_path(out)}: {exc}", file=sys.stderr)
        return 3
    h = body["headline"]
    print(
        f"compared {body['arms']['control']['n_consensuses']} control vs "
        f"{body['arms']['fp']['n_consensuses']} FP consensuses over "
        f"{h['n_settings']} settings; the control's record-level CI contains the FP point "
        f"at {h['n_settings_where_control_ci_contains_the_fp_point']} of them -> {out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
