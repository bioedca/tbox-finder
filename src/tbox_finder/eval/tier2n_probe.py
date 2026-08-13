"""The Tier-2N probe set + per-round recall halt/rollback rule (P2-07).

ADR-0005 D14: *"A **Tier-2N probe set** (non-canonical + synthetic-Tier-2N
positives) is evaluated **each mining round**, and a per-round **recall drop on it
halts/rolls back** the iteration"* — so aggressive hard-negative mining cannot
directionally train the production scanner to reject the flagship class. The worst
case is then a *directionally-bounded* Tier-2N sensitivity, not an invalid
generalization claim.

Probe-set composition, and why the natural arm is empty
-------------------------------------------------------
The probe set is the union of two arms:

* the **natural** arm — real non-canonical (Tier-2N) positives; and
* the **synthetic** arm — :mod:`tbox_finder.synth.tier2n` output.

The natural arm is **empty at P2-07, by construction rather than by oversight**.
The corpus is 100 % TBDB/CM-derived, so it cannot contain a CM-invisible locus;
neither the corpus nor the committed split table carries a tier column to select
on; and the literature documents no genuinely CM-invisible T-box architecture —
by definition, since that is the class this project exists to discover. The arm is
therefore reported at N = 0 and **disclosed**, in the same "reported-not-gated"
spirit as ADR-0005 D6/D9, never quietly dropped from the accounting.

That places the whole min-N burden on the synthetic arm, which is why
:mod:`tbox_finder.synth.tier2n` makes probe eligibility a **measured discordant
pair** (parent CM-detected, variant CM-missed) rather than a count the generator
chooses. Without that, "probe-set size ≥ min-N" would gate a knob, not evidence.

Halt / rollback
---------------
Recall is measured on the probe set each round and compared against the
**best round so far**, not merely the previous round — otherwise a slow monotone
bleed of a few points per round never trips a
previous-round comparison while still destroying the class over an iteration.

Pure stdlib. PRD §9.1, §12; ADR-0005 D14; ADR-0006 D9.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.synth.tier2n import Tier2NVariant

#: The probe-set floor, imported from the ADR-0005 Amendment A1 pin.
TIER2N_PROBE_MIN_N = MIN_REAL_HOMOLOG_N

#: Absolute per-round recall drop, relative to the best round so far, that halts
#: the mining iteration and rolls back to the best checkpoint.
#:
#: Set to one probe-positive's worth of recall at the pinned floor
#: (``1 / MIN_REAL_HOMOLOG_N`` = 5 pp), matching the same ``1/N`` granularity
#: argument that pinned ``MIN_REAL_HOMOLOG_N`` and the ADR-0005 D4 +5 pp CI floor
#: — the smallest drop the probe set can actually resolve. A tighter bound would
#: be unmeasurable noise; a looser one would let a real regression pass.
#: Frozen: no CLI/config override.
TIER2N_RECALL_DROP_HALT = 1.0 / MIN_REAL_HOMOLOG_N

#: Tolerance for the halt comparison. Recall values are float ratios ``k/N``, so
#: a drop that is *exactly* one probe-positive can differ from
#: :data:`TIER2N_RECALL_DROP_HALT` by a few ulps in either direction. Far smaller
#: than any real recall difference the probe set can resolve (the finest is
#: ``1/N``), so it cannot mask a genuine regression — it only stops float
#: representation from deciding whether the rule fires.
HALT_COMPARISON_TOL = 1e-9

ROUND_CONTINUE = "continue"
ROUND_HALT_ROLLBACK = "halt_rollback"
ROUND_INADMISSIBLE = "inadmissible"


class Tier2NProbeError(ValueError):
    """Raised on a malformed probe set or round history."""


@dataclass(frozen=True)
class ProbeSet:
    """The evaluated Tier-2N probe set for a mining round.

    IDs must be unique **across both arms**. Otherwise :attr:`size` (which counts
    members) and :func:`probe_recall` (which de-duplicates into a set) would
    disagree, so a probe set could clear min-N on a count that recall does not
    recognise — a min-N pass on members that are not there.
    """

    natural: tuple[str, ...]
    synthetic: tuple[str, ...]

    def __post_init__(self) -> None:
        members = list(self.natural) + list(self.synthetic)
        if len(set(members)) != len(members):
            duplicates = sorted({m for m in members if members.count(m) > 1})
            raise Tier2NProbeError(
                f"probe-set IDs must be unique across both arms; duplicated: {duplicates}"
            )

    @property
    def size(self) -> int:
        return len(self.natural) + len(self.synthetic)

    def meets_min_n(self) -> bool:
        """Whether the probe set clears the ADR-0005 A1 floor.

        Guarded on non-emptiness so the clause cannot read true off an absent set.
        """
        return self.size > 0 and self.size >= TIER2N_PROBE_MIN_N


def build_probe_set(
    variants: list[Tier2NVariant],
    natural_ids: tuple[str, ...] = (),
) -> ProbeSet:
    """Assemble the probe set from the synthetic generator output + a natural arm.

    Only **probe-eligible** variants (measured discordant pairs) enter the
    synthetic arm; an emitted-but-unmeasured variant is excluded, so an unrun
    ``cmsearch`` shrinks the probe set toward failing min-N rather than inflating
    it toward passing.
    """
    synthetic = tuple(sorted(v.variant_id for v in variants if v.is_probe_eligible()))
    return ProbeSet(natural=tuple(sorted(natural_ids)), synthetic=synthetic)


# --------------------------------------------------------------------------- #
# Serialization — the write half of ``mining.remine.load_probe_set`` (P3-15′-i)
# --------------------------------------------------------------------------- #
#: Schema version of the serialized probe-set id file.
#:
#: :func:`tbox_finder.mining.remine.load_probe_set` shipped as a **reader with no
#: writer**: it reads ``{"natural": [...], "synthetic": [...]}`` and refuses an
#: empty set, but nothing in ``src/`` ever serialized a :class:`ProbeSet`, so the
#: round leg that consumes it could not start at all. These functions are that
#: missing half.
PROBE_SET_SCHEMA_VERSION = "1.0"

#: Shape of a synthetic probe id, mirrored from
#: :func:`tbox_finder.synth.tier2n.generate` (``f"tier2n:{family}:{record_id}"``).
_SYNTHETIC_ID_PREFIX = "tier2n"
_SYNTHETIC_ID_FIELDS = 3


def synthetic_probe_id_family(probe_id: str) -> str:
    """The family named by a synthetic probe id (``tier2n:<FAMILY>:<record>``).

    Raises rather than returning a sentinel. The only caller is the per-family
    reconciliation below; an unparsable id degraded to ``""`` would be counted
    into a bucket the report does not have, turning a genuine mismatch into an
    agreement about a family that does not exist.
    """
    parts = probe_id.split(":")
    # Both trailing fields are checked for emptiness, not only the family: an id like
    # ``tier2n:CLASS_II_PLATFORM_SWAP:`` names no record, so it would reconcile into
    # its family's count and be written as a member no scanner can ever recover.
    if (
        len(parts) != _SYNTHETIC_ID_FIELDS
        or parts[0] != _SYNTHETIC_ID_PREFIX
        or not parts[1]
        or not parts[2]
    ):
        raise Tier2NProbeError(
            f"synthetic probe id is not 'tier2n:<FAMILY>:<record>': {probe_id!r}"
        )
    return parts[1]


def _expected_int(
    report: Mapping[str, Any], key: str, problems: list[str], *, where: str = "counts report"
) -> int | None:
    """Read an integer count out of the counts report, or record why it could not.

    A missing or non-integer key is a **problem**, never a skipped clause: a
    reconciliation that quietly drops the comparison it could not make reports
    agreement it never measured. ``where`` names the block the key was read from,
    so a per-family failure says which family rather than which key alone.
    """
    if key not in report:
        problems.append(f"{where} has no {key!r} to reconcile the written ids against")
        return None
    value = report[key]
    if isinstance(value, bool) or not isinstance(value, int):
        problems.append(f"{where} {key!r} is not an integer: {value!r}")
        return None
    return value


def reconcile_probe_set_with_report(probe_set: ProbeSet, report: Mapping[str, Any]) -> list[str]:
    """Cross-check probe-set **members** against the counts report of the same run.

    Returns the disagreements; ``[]`` means the two agree.

    This is a real cross-check rather than a restatement: the ids come from
    :func:`build_probe_set` (which collects ``variant_id``s) and the counts come
    from :func:`tbox_finder.synth.tier2n.build_report` (which counts
    ``is_probe_eligible`` variants), so the two are computed by different code
    over the same variants. It is what turns *"these ids are the members those
    counts describe"* from a claim into an assertion — and it is checked
    **before** the file is written, so a mismatch leaves no artifact behind.
    """
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return [f"counts report is not an object: {type(report).__name__}"]

    for key, observed, what in (
        ("n_natural", len(probe_set.natural), "natural-arm ids"),
        ("n_synthetic", len(probe_set.synthetic), "synthetic-arm ids"),
        ("n_probe_eligible", len(probe_set.synthetic), "probe-eligible ids"),
        ("probe_set_size", probe_set.size, "probe-set members"),
    ):
        expected = _expected_int(report, key, problems)
        if expected is not None and expected != observed:
            problems.append(
                f"counts report {key}={expected} but the probe set has {observed} {what}"
            )

    per_family = report.get("per_family")
    if not isinstance(per_family, Mapping) or (not per_family and probe_set.synthetic):
        # Guarded on the synthetic arm being non-empty: an absent or empty
        # per-family block beside 45 ids would otherwise make the loop below
        # vacuous, and a vacuous loop reports agreement from nothing.
        problems.append(
            "counts report carries no usable 'per_family' block to reconcile "
            f"{len(probe_set.synthetic)} synthetic ids against"
        )
        return problems

    observed_by_family: dict[str, int] = {}
    for probe_id in probe_set.synthetic:
        try:
            family = synthetic_probe_id_family(probe_id)
        except Tier2NProbeError as exc:
            problems.append(str(exc))
            continue
        observed_by_family[family] = observed_by_family.get(family, 0) + 1

    for family in sorted(set(per_family) | set(observed_by_family)):
        observed = observed_by_family.get(family, 0)
        entry = per_family.get(family)
        if not isinstance(entry, Mapping):
            problems.append(
                f"counts report has no per-family entry for {family!r}, "
                f"but {observed} written id(s) name it"
            )
            continue
        expected = _expected_int(
            entry, "probe_eligible", problems, where=f"counts report per_family[{family!r}]"
        )
        if expected is not None and expected != observed:
            problems.append(
                f"counts report per_family[{family!r}].probe_eligible={expected} "
                f"but {observed} written id(s) name it"
            )
    return problems


def probe_set_payload(
    probe_set: ProbeSet, *, provenance: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The JSON object :func:`tbox_finder.mining.remine.load_probe_set` reads back.

    Both arms are emitted as lists even when empty — the natural arm is empty by
    construction here (see the module docstring), and a written ``[]`` says so,
    whereas an omitted key would leave a reader unable to distinguish *"no
    natural members"* from *"this writer did not know about the natural arm"*.

    No timestamp and no git SHA: the payload is a pure function of the run, so
    re-running the build over the same corpus and CM rewrites it byte-identically
    and a diff means the *members* moved.
    """
    payload: dict[str, Any] = {
        "schema_version": PROBE_SET_SCHEMA_VERSION,
        "natural": list(probe_set.natural),
        "synthetic": list(probe_set.synthetic),
    }
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    return payload


def write_probe_set(
    path: str | Path,
    probe_set: ProbeSet,
    report: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write the probe-set ids, refusing **before** the write rather than after it.

    Two refusals, both fail-closed — nothing is written when either fires:

    * the members disagree with the counts report of the same run
      (:func:`reconcile_probe_set_with_report`); and
    * the probe set is below the ADR-0005 Amendment A1 floor
      (``TIER2N_PROBE_MIN_N`` = 20).

    The min-N refusal is the load-bearing one. ``load_probe_set`` only refuses a
    *wholly empty* set, so without this a cheap build — ``--n-parents 30``, three
    eligible variants — would mint a well-formed probe file that a mining round
    accepts, and the round's per-round halt/rollback decision (ADR-0005 D14) would
    then rest on an underpowered instrument while every clause read green. The
    floor belongs at the point the artifact is minted, not only at the point it is
    read.
    """
    problems = reconcile_probe_set_with_report(probe_set, report)
    if problems:
        raise Tier2NProbeError(
            "probe-set ids disagree with the counts report of the same run, so "
            "nothing was written: " + "; ".join(problems)
        )
    if not probe_set.meets_min_n():
        raise Tier2NProbeError(
            f"probe set has {probe_set.size} member(s), below the ADR-0005 A1 floor "
            f"of {TIER2N_PROBE_MIN_N}; nothing was written, because a well-formed "
            "file with too few members is accepted by load_probe_set and would "
            "carry an underpowered probe into the round's halt/rollback decision"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = probe_set_payload(probe_set, provenance=provenance)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    # Written through a sibling temp file and ``os.replace`` (atomic on POSIX, same
    # filesystem by construction) rather than straight to ``out``: ``write_text``
    # truncates the destination first, so an I/O failure partway through would leave a
    # previously-valid probe set as partial JSON. A mining round reads this file to
    # decide what it may treat as a hard negative; half a file is not a safe input, and
    # the failure would surface at the next round rather than at the write.
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def probe_recall(probe_set: ProbeSet, recovered_ids: set[str]) -> float:
    """Fraction of the probe set the scanner still recovers this round.

    Raises on an empty probe set: a recall of ``0/0`` would otherwise be reported
    as a number (or a vacuous 1.0) for a measurement that never happened.
    """
    if probe_set.size == 0:
        raise Tier2NProbeError("cannot compute recall on an empty probe set")
    members = set(probe_set.natural) | set(probe_set.synthetic)
    return len(members & set(recovered_ids)) / len(members)


def round_decision(
    probe_set: ProbeSet,
    recall_this_round: float,
    recall_history: list[float],
) -> dict[str, Any]:
    """Decide whether the mining iteration continues, or halts and rolls back.

    The comparison baseline is the **best** recall observed so far, not the
    previous round. Returns a report whose clauses are re-derived from the
    arguments rather than accumulated by the caller.
    """
    if not 0.0 <= recall_this_round <= 1.0:
        raise Tier2NProbeError(f"recall must be in [0, 1], got {recall_this_round}")
    for value in recall_history:
        if not 0.0 <= value <= 1.0:
            raise Tier2NProbeError(f"recall history contains {value}, outside [0, 1]")

    admissible = probe_set.meets_min_n()
    best_prior = max(recall_history) if recall_history else None
    drop = None if best_prior is None else best_prior - recall_this_round
    # Compared with a tolerance because both operands are float subtractions of
    # k/N ratios: an exact one-probe-positive regression can land a few ulps
    # BELOW the threshold (0.95 - 0.90 == 0.04999999999999993 < 0.05) and a bare
    # ``>=`` would let the very regression this rule exists to catch continue.
    breached = bool(drop is not None and drop >= TIER2N_RECALL_DROP_HALT - HALT_COMPARISON_TOL)

    if not admissible:
        decision = ROUND_INADMISSIBLE
    elif breached:
        decision = ROUND_HALT_ROLLBACK
    else:
        decision = ROUND_CONTINUE

    return {
        "decision": decision,
        "probe_set_size": probe_set.size,
        "n_natural": len(probe_set.natural),
        "n_synthetic": len(probe_set.synthetic),
        "tier2n_probe_min_n": TIER2N_PROBE_MIN_N,
        "probe_set_meets_min_n": admissible,
        "recall_this_round": recall_this_round,
        "best_prior_recall": best_prior,
        "recall_drop_vs_best": drop,
        "halt_threshold": TIER2N_RECALL_DROP_HALT,
        "halt_threshold_breached": breached,
        "baseline_rule": "best round so far (not previous round) — catches slow monotone bleed",
        "natural_arm_disclosure": (
            "the natural Tier-2N arm is empty by construction: the corpus is "
            "100% CM-derived and so cannot contain a CM-invisible locus; reported "
            "at N=0 rather than dropped from the accounting"
        ),
    }
