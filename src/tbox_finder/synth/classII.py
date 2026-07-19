"""The construction-powered synthetic-class-II recovery set (P2-08).

Backs the **anti-mimicry sub-arm** of GATE-1 (PRD §2.3, §5 mechanism 3; ADR-0005
D9), graded **Stage-1-only on the class-II-CM-naive checkpoint** at P4 — never
routed through the TBDB-class-II-trained Stage-2, which would rescue class-II
hits and confound the test.

What this module does and does not claim
----------------------------------------
It emits **presentation variants** (window-phase offset, reverse complement) of
*real held-out class-II corpus records*. It asserts **no new biology**: every
variant is a re-presentation of a real T-box leader, so recovery of a variant is
evidence about the model's robustness and its class-II sensitivity, not about a
sequence anyone claims exists in nature.

That framing is deliberate. The construction the step's name most naturally
suggests — swapping a class-I transcriptional terminator platform for a class-II
sequestrator to mint synthetic class-II positives from the 8,281 class-I training
parents — was put through the CLAUDE.md §10.1 evidence gate at P2-08 and
**refused**:

* Platform modularity with respect to the sensing core is **contested**. Stem I
  architecture systematically covaries with class: transcriptional T-boxes
  generally carry long Stem I with the IDTM motif, translational ones are mostly
  intermediate or "ultrashort" and lack it (Suddala & Zhang 2020,
  DOI:10.1002/wrna.1600; accessed 2026-07-19), and class-II Stem I domains "lack
  several highly conserved elements that are essential for interaction with the
  tRNA ligand in other T box RNAs" (Sherwood et al. 2015, PMID:25583497). Two
  independent labs, both against the premise.
* **No published functional T-box platform-swap chimera could be verified.** The
  one review-level assertion of such a construct (PMID:31206978) cites a source
  that does not contain it. What *is* precedented is specifier-level
  re-engineering (PMID:9098057; PMID:21233158) — a different claim.

Encoding an unsupported premise directly beneath the headline anti-mimicry
pillar is what §10.2/§10.3 forbid, so the generator does not do it.

Where the power actually comes from
-----------------------------------
**Parents, not variants.** ADR-0005 D9 grades this arm with a *block-resampled*
floor (PRD §2.3: resampling at the homology-cluster / held-out-order level), so
emitting more variants per parent raises the record count without raising the
evidence count. A recovery set of 200 variants drawn from 22 parents spans the
same 20 clusters as the 22 parents do. The gate in :func:`build_report`
therefore keys on **block counts**, and the variant count is reported as what it
is — a presentation multiplier.

ADR-0004 Amendment A3 (P2-08, user sign-off 2026-07-19) scopes D7's
"training-fold parents only" rule to *training* augmentation and permits this
**evaluation-only** set to be parented on **held-out** records, which is what
buys the block power: 22 records / 20 clusters / 4 orders from the training fold
versus the held-out class-II pool measured in the report. D7's guarantee is
preserved by a stronger mechanism — an eval-only variant is never
training-eligible at all, rather than being made training-safe.

ADR-0005 Amendment A5 (same sign-off) records the honest limit: this set supplies
**min-N and block-level power, not phylogenetic independence**. Every high-count
parent lineage is Actinobacteria, so recovery *unaccompanied by a passing
within-phylum control* is not evidence of anti-mimicry.

The within-phylum-homology control
----------------------------------
D9 asks for a control separating anti-mimicry from within-Actinobacteria
memorization. Measured at P2-08: **zero Actinobacteria records of any class are
in the nested training fold**, so memorization by direct exposure is impossible —
the whole phylum is held out. The residual risk is *homology-mediated*: a
held-out Actinobacteria record close enough to a trained non-Actinobacteria one.
Each variant therefore carries its parent's **identity-to-nearest-training-fold
bin** (:func:`tbox_finder.power.bin_identity` over
:func:`tbox_finder.power.compute_heldout_identities`, the same metric as the
ADR-0004 D2 histogram — reused, not re-implemented), so P4 stratifies recovery by
homology. Recovery concentrated in the high-identity stratum is memorization;
recovery that holds in the low-identity and no-neighbour strata is not.

Pure stdlib at import time; ``pandas`` and the alignment backend are lazy.
PRD §2.3, §5, §7.1, §11; ADR-0005 D9 + A1 + A5; ADR-0004 D7 + A3.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.splits import INTERIM_TABLE, SYNTHETIC_CLASSII_SOURCE
from tbox_finder.synth import _common

#: Block units D9/D4/D5 resample over (PRD §2.3). The gate is graded at **both**;
#: reporting only the flattering one is how a green clause ends up measuring the
#: wrong quantity.
BLOCK_UNITS: tuple[str, ...] = ("cluster_id", "resolved_order")

#: The min-N floor. Imported from the ADR-0005 A1 pin, never re-declared locally —
#: a local literal drifts silently from the ADR it is supposed to enforce.
RECOVERY_MIN_N: int = MIN_REAL_HOMOLOG_N

#: Presentation transforms. ``phase`` re-tiles the leader at a different offset
#: within the model's window (PRD §11 window-phase-offset augmentation); ``rc``
#: presents the reverse complement, which the Stage-1 backbone is equivariant to
#: rather than invariant to (P1). Neither alters the underlying biology, which is
#: precisely why they are admissible here.
TRANSFORMS: tuple[str, ...] = ("phase", "rc", "phase_rc")

#: Identity-bin strata the D9 within-phylum-homology control reads. Sourced from
#: :mod:`tbox_finder.power` so the bin edges cannot drift from the D1/D2 pins.
CONTROL_STRATUM_KEY = "parent_identity_bin"

REPORT_PATH = Path("reports/p2/classII_recovery.json")
RECOVERY_TABLE_PATH = Path("data/processed/synth/classII_recovery.parquet")

#: Columns a parent row must carry for :func:`eligible_parents` to judge it.
REQUIRED_PARENT_COLUMNS: tuple[str, ...] = (
    "record_id",
    "source",
    "klass",
    "cluster_id",
    "resolved_phylum",
    "resolved_order",
    "nested_train",
    "nested_role",
)


class RecoverySetError(ValueError):
    """Raised when the recovery set cannot be built from what was supplied."""


@dataclass(frozen=True)
class RecoveryVariant:
    """One evaluation-only presentation variant of a real held-out class-II record."""

    variant_id: str
    parent_record_id: str
    transform: str
    phase_offset: int
    reverse_complement: bool
    parent_cluster_id: int
    parent_resolved_order: str | None
    parent_resolved_phylum: str | None
    #: Parent's identity-to-nearest-training-fold bin, or ``None`` when the
    #: alignment backend was not available. ``None`` is **not** a stratum: it
    #: leaves the control unmeasured, and :func:`build_report` says so rather
    #: than silently folding it into a bin.
    parent_identity_bin: str | None = None


def _require_columns(row: Mapping[str, Any], where: str) -> None:
    missing = [c for c in REQUIRED_PARENT_COLUMNS if c not in row]
    if missing:
        raise RecoverySetError(f"{where}: parent row is missing column(s) {missing}")


def is_eligible_parent(row: Mapping[str, Any]) -> bool:
    """Is ``row`` an admissible parent for the evaluation-only recovery set?

    Four conjuncts, each load-bearing:

    * ``klass == "II"`` — the arm is about translational T-boxes.
    * ``source == "corpus"`` — **excludes the 18-record blind set** (PRD §7.1).
      Those are the *natural* arm; parenting synthetic variants on them would
      make the synthetic and natural evidence non-independent, and the whole
      point of the synthetic set is to supplement the natural one.
    * ``nested_role == "heldout"`` and ``not nested_train`` — held out from the
      D5 nested training fold. Checked as a **conjunction** rather than inferring
      one from the other: ``nested_train`` is a bool whose negation means "in the
      LOO holdout", not "held out from training", and reasoning from that
      negation already produced one wrong fold in this repo (P2-06a).
    """
    _require_columns(row, "is_eligible_parent")
    return (
        row["klass"] == "II"
        and row["source"] == "corpus"
        and row["nested_role"] == "heldout"
        and not bool(row["nested_train"])
    )


def eligible_parents(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Admissible parents, in a deterministic order independent of input order."""
    kept = [dict(r) for r in rows if is_eligible_parent(r)]
    return sorted(kept, key=lambda r: str(r["record_id"]))


def variant_id(parent_record_id: str, transform: str, phase_offset: int) -> str:
    """Stable, collision-free id for a variant.

    Encodes the transform in the id so a reader of the split table alone can tell
    what a row is without joining to the recovery table.
    """
    if transform not in TRANSFORMS:
        raise RecoverySetError(f"unknown transform {transform!r} (known: {list(TRANSFORMS)})")
    return f"{parent_record_id}#c2{transform}{phase_offset}"


def build_recovery_set(
    parents: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    variants_per_parent: int = 2,
    max_phase_offset: int = 64,
    identity_bins: Mapping[str, str] | None = None,
) -> list[RecoveryVariant]:
    """Build the evaluation-only synthetic-class-II recovery set.

    ``parents`` are filtered through :func:`is_eligible_parent` here rather than
    trusted, so a caller that forgets to filter cannot smuggle a training-fold or
    blind-set parent in. ``identity_bins`` maps ``record_id`` → D1 identity bin
    for the D9 control; a parent absent from it carries ``None`` (unmeasured),
    never a fabricated stratum.

    Emission order and transform selection are derived from ``seed`` via
    :func:`tbox_finder.synth._common.stable_key`, so the set is reproducible and
    independent of the order rows happened to arrive in.
    """
    if variants_per_parent < 1:
        raise RecoverySetError(f"variants_per_parent must be >= 1, got {variants_per_parent}")
    if variants_per_parent > len(TRANSFORMS):
        raise RecoverySetError(
            f"variants_per_parent {variants_per_parent} exceeds the {len(TRANSFORMS)} "
            "distinct transforms available; a parent cannot carry two variants of the "
            "same transform without their ids depending on the phase draw for uniqueness"
        )
    if max_phase_offset < 1:
        raise RecoverySetError(f"max_phase_offset must be >= 1, got {max_phase_offset}")

    admissible = eligible_parents(parents)
    if not admissible:
        raise RecoverySetError(
            "no eligible class-II parents (need klass=='II', source=='corpus', "
            "nested_role=='heldout', not nested_train) — refusing to emit an empty "
            "recovery set silently"
        )

    bins = dict(identity_bins or {})
    ordered = sorted(
        admissible, key=lambda r: _common.stable_key(seed, "order", str(r["record_id"]))
    )

    out: list[RecoveryVariant] = []
    for parent in ordered:
        rid = str(parent["record_id"])
        # Distinct transforms per parent, chosen by a seed-derived permutation. Two
        # variants of one parent therefore differ in their id *by construction*
        # rather than relying on their phase draws to differ — with 3 transforms ×
        # 64 offsets, independent draws collide for roughly 1 parent in 192, which
        # is exactly the birthday collision the first draft hit.
        picks = sorted(TRANSFORMS, key=lambda t: _common.stable_key(seed, "tf", rid, t))
        for k, transform in enumerate(picks[:variants_per_parent]):
            offset = _common.stable_key(seed, "phase", rid, str(k)) % max_phase_offset
            out.append(
                RecoveryVariant(
                    variant_id=variant_id(rid, transform, offset),
                    parent_record_id=rid,
                    transform=transform,
                    phase_offset=int(offset),
                    reverse_complement=transform in ("rc", "phase_rc"),
                    parent_cluster_id=int(parent["cluster_id"]),
                    parent_resolved_order=parent.get("resolved_order"),
                    parent_resolved_phylum=parent.get("resolved_phylum"),
                    parent_identity_bin=bins.get(rid),
                )
            )

    ids = [v.variant_id for v in out]
    if len(set(ids)) != len(ids):
        raise RecoverySetError(
            "variant_id collision — two variants of one parent drew the same "
            "(transform, phase_offset); raise max_phase_offset or lower "
            "variants_per_parent"
        )
    return out


def block_counts(variants: Sequence[RecoveryVariant]) -> dict[str, int]:
    """Distinct block count per D9 resampling unit — the arm's real N."""
    parents = {v.parent_record_id: v for v in variants}.values()
    return {
        "cluster_id": len({p.parent_cluster_id for p in parents}),
        "resolved_order": len({p.parent_resolved_order for p in parents}),
    }


def control_strata(variants: Sequence[RecoveryVariant]) -> dict[str, int]:
    """Parent counts per D9 within-phylum-homology stratum (unmeasured tracked)."""
    parents = {v.parent_record_id: v for v in variants}.values()
    counts: dict[str, int] = {}
    for p in parents:
        key = p.parent_identity_bin if p.parent_identity_bin is not None else "unmeasured"
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_report(
    variants: Sequence[RecoveryVariant], *, seed: int, split_table: str | Path = INTERIM_TABLE
) -> dict[str, Any]:
    """Assemble the P2-08 recovery-set report + its gate clauses.

    Every count is re-derived here from ``variants`` rather than accumulated by
    the caller, so the report cannot describe a set other than the one it was
    handed — the P2-05 failure was a clause derived from the *requested* config
    rather than from found evidence, which reads vacuously TRUE exactly when the
    evidence is missing.

    The ``bool(...) and`` emptiness guards below are **defence in depth, not the
    load-bearing mechanism**, and sabotage at P2-08 confirmed it: removing one
    leaves all 54 unit tests green, because ``block_counts([])`` is ``0`` and
    ``0 >= RECOVERY_MIN_N`` is already False. They would only start to bite if
    the pinned floor were ever 0. Stated plainly here because a comment claiming
    a redundant guard is load-bearing is how the *next* reader stops looking for
    the real one. The absence branch is covered by asserting the **gate** over an
    empty set (``test_an_empty_set_fails_every_gate_clause``), not by trusting
    these guards.
    """
    parents = sorted({v.parent_record_id for v in variants})
    blocks = block_counts(variants)
    strata = control_strata(variants)
    measured_parents = sum(n for k, n in strata.items() if k != "unmeasured")

    per_unit_pass = {
        unit: bool(variants) and blocks[unit] >= RECOVERY_MIN_N for unit in BLOCK_UNITS
    }
    # Graded at the WEAKEST block unit, not the best: an arm powered at the
    # cluster level but not the order level is not a powered arm.
    blocks_meet_min_n = bool(variants) and all(per_unit_pass.values())
    control_measurable = bool(variants) and measured_parents >= RECOVERY_MIN_N

    return {
        "step": "P2-08",
        "seed": int(seed),
        "adr": ["ADR-0005 D9", "ADR-0005 A1", "ADR-0005 A5", "ADR-0004 D7", "ADR-0004 A3"],
        "interim_split_table": _common.repo_relative(split_table),
        "source_label": SYNTHETIC_CLASSII_SOURCE,
        "min_n": int(RECOVERY_MIN_N),
        "n_variants": len(variants),
        "n_parents": len(parents),
        "block_units": list(BLOCK_UNITS),
        "block_counts": blocks,
        "block_unit_meets_min_n": per_unit_pass,
        "transform_counts": {t: sum(1 for v in variants if v.transform == t) for t in TRANSFORMS},
        "control_strata_parent_counts": strata,
        "n_parents_with_measured_identity": measured_parents,
        "gate": {
            "blocks_meet_min_n": blocks_meet_min_n,
            "control_measurable": control_measurable,
            "overall_pass": bool(blocks_meet_min_n and control_measurable),
        },
        "eligibility_rule": (
            "klass == 'II' AND source == 'corpus' AND nested_role == 'heldout' AND "
            "NOT nested_train. source=='corpus' excludes the 18-record blind set "
            "(PRD §7.1) so the synthetic arm stays independent of the natural one."
        ),
        "power_disclosure": (
            "N for this arm is the BLOCK count, not the variant count: D9 resamples at "
            "the homology-cluster / held-out-order level (PRD §2.3), so variants of a "
            "shared parent add no evidence. n_variants is a presentation multiplier and "
            "is not the graded quantity."
        ),
        "independence_limitation": (
            "ADR-0005 A5: this set supplies min-N and block-level power, NOT "
            "phylogenetic independence — the parent pool is dominated by "
            "Actinobacteria. Recovery unaccompanied by a passing within-phylum-homology "
            "control is not evidence of anti-mimicry."
        ),
        "construction_disclosure": (
            "Presentation transforms (window-phase offset, reverse complement) of real "
            "held-out class-II records. No platform-swap chimera is emitted: the "
            "modularity premise it would require is contested (DOI:10.1002/wrna.1600; "
            "PMID:25583497) and no functional T-box platform-swap construct could be "
            "verified in the literature (CLAUDE.md §10.1 gate, P2-08, accessed "
            "2026-07-19)."
        ),
        "grading_scope": (
            "Stage-1-only on the class-II-CM-naive checkpoint (P2-11), never through "
            "the TBDB-class-II-trained Stage-2 (ADR-0005 D9). Evaluation-only: these "
            "rows are never training-eligible (ADR-0004 A3)."
        ),
    }


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Re-derive every gate clause from the report's own evidence.

    Returns a list of problems and **never raises** — a validator that dies on
    malformed input cannot report that the input was malformed. ``overall_pass``
    being an ``all(...)`` of stored clauses catches a clause flipped FALSE but
    never one fabricated TRUE, so each clause is recomputed here from the counts
    the report carries, and the evidence blocks those counts come from are
    *required* rather than defaulted.
    """
    problems: list[str] = []

    for key in ("n_variants", "n_parents", "min_n", "block_counts", "gate"):
        if key not in report:
            problems.append(f"missing required report block: {key!r}")
    if problems:
        return problems

    gate = report["gate"]
    blocks = report["block_counts"]
    if not isinstance(gate, Mapping) or not isinstance(blocks, Mapping):
        return ["'gate' and 'block_counts' must both be objects"]

    min_n = report["min_n"]
    if min_n != MIN_REAL_HOMOLOG_N:
        problems.append(
            f"min_n {min_n!r} != the ADR-0005 A1 pin {MIN_REAL_HOMOLOG_N} "
            "(the report must not carry a local copy that has drifted)"
        )

    n_variants = report["n_variants"]
    if isinstance(n_variants, bool) or not isinstance(n_variants, int) or n_variants < 0:
        problems.append(f"n_variants must be a non-negative int, got {n_variants!r}")
        return problems

    for unit in BLOCK_UNITS:
        if unit not in blocks:
            problems.append(f"block_counts is missing the {unit!r} resampling unit")
    if problems:
        return problems

    expected_blocks = bool(n_variants) and all(
        int(blocks[unit]) >= MIN_REAL_HOMOLOG_N for unit in BLOCK_UNITS
    )
    if _common.bad_bool(gate.get("blocks_meet_min_n"), expected_blocks):
        problems.append(
            f"gate.blocks_meet_min_n is {gate.get('blocks_meet_min_n')!r} but the "
            f"recorded block counts {dict(blocks)} against min-N {MIN_REAL_HOMOLOG_N} "
            f"give {expected_blocks}"
        )

    measured = report.get("n_parents_with_measured_identity")
    if isinstance(measured, bool) or not isinstance(measured, int):
        problems.append(f"n_parents_with_measured_identity must be an int, got {measured!r}")
    else:
        expected_control = bool(n_variants) and measured >= MIN_REAL_HOMOLOG_N
        if _common.bad_bool(gate.get("control_measurable"), expected_control):
            problems.append(
                f"gate.control_measurable is {gate.get('control_measurable')!r} but "
                f"{measured} parents carry a measured identity bin against min-N "
                f"{MIN_REAL_HOMOLOG_N} → {expected_control}"
            )

    clause_keys = ("blocks_meet_min_n", "control_measurable")
    expected_overall = all(gate.get(k) is True for k in clause_keys)
    if _common.bad_bool(gate.get("overall_pass"), expected_overall):
        problems.append(
            f"gate.overall_pass is {gate.get('overall_pass')!r} but its clauses "
            f"{ {k: gate.get(k) for k in clause_keys} } give {expected_overall}"
        )

    if report["n_parents"] > n_variants:
        problems.append(
            f"n_parents ({report['n_parents']}) exceeds n_variants ({n_variants}) — "
            "every parent must contribute at least one variant"
        )

    return problems


def variant_rows(variants: Sequence[RecoveryVariant]) -> list[dict[str, Any]]:
    """Sequence-free rows for the committed recovery table."""
    return [asdict(v) for v in variants]


# --------------------------------------------------------------------------- #
# CLI — heavy backends imported lazily so the module loads in the bare-CI tier.
# --------------------------------------------------------------------------- #


def _load_split_rows(interim_table: str | Path) -> list[dict[str, Any]]:
    """Committed-schema rows, projected from the DVC-interim table.

    Reads the **interim** table and runs it through the pinned
    :func:`tbox_finder.splits.build_split_table` projection rather than reading the
    committed parquet. Two reasons, both load-bearing:

    * **No dependency cycle.** ``split_assignment_table`` consumes this module's
      output, so this module must not consume ``split_assignment_table``'s.
    * **One source of truth for the schema.** Re-deriving the ``record_id`` /
      ``klass`` / fold columns here would be a second projection that could drift
      from the one the committed table is actually built with.
    """
    import pandas as pd

    from tbox_finder import splits

    projected = splits.build_split_table(pd.read_parquet(interim_table))
    return projected.to_dict("records")


def _identity_bins(aligned_dir: str | Path, interim_table: str | Path) -> dict[str, str]:
    """Parent → D1 identity bin, via the pinned ADR-0004 D2 metric.

    Reuses :func:`tbox_finder.power.compute_heldout_identities` +
    :func:`tbox_finder.power.bin_identity` rather than re-deriving identity, so
    the D9 control and the D1 histogram can never disagree. Returns ``{}`` when
    the alignment is unavailable — the caller then records the control as
    unmeasured instead of inventing strata.
    """
    import pandas as pd

    from tbox_finder import power

    aligned = Path(aligned_dir)
    if not (aligned / "class_II.sto").exists() or not Path(interim_table).exists():
        return {}
    interim = pd.read_parquet(interim_table)
    identities = power.compute_heldout_identities(interim, aligned)
    return {name: power.bin_identity(x) for name, x in identities.items()}


def build_report_from_artifacts(
    *,
    interim_table: str | Path = INTERIM_TABLE,
    aligned_dir: str | Path = "data/interim/splits/aligned",
    seed: int = 20260719,
    variants_per_parent: int = 2,
) -> tuple[dict[str, Any], list[RecoveryVariant]]:
    """Build the recovery set + report from the DVC-interim split artifacts."""
    rows = _load_split_rows(interim_table)
    parents = eligible_parents(rows)
    bins = _identity_bins(aligned_dir, interim_table)
    variants = build_recovery_set(
        parents, seed=seed, variants_per_parent=variants_per_parent, identity_bins=bins
    )
    return build_report(variants, seed=seed, split_table=interim_table), variants


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tbox_finder.synth.classII",
        description="Build the P2-08 construction-powered synthetic-class-II recovery set.",
    )
    parser.add_argument("--interim-table", default=str(INTERIM_TABLE))
    parser.add_argument("--aligned-dir", default="data/interim/splits/aligned")
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--table", default=str(RECOVERY_TABLE_PATH))
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--variants-per-parent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the recovery table + report. Exit code **is** the gate."""
    import pandas as pd

    args = _build_parser().parse_args(argv)
    report, variants = build_report_from_artifacts(
        interim_table=args.interim_table,
        aligned_dir=args.aligned_dir,
        seed=args.seed,
        variants_per_parent=args.variants_per_parent,
    )

    problems = validate_report(report)
    if problems:
        for p in problems:
            print(f"[classII] INVALID REPORT: {p}", file=sys.stderr)
        return 2

    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(variant_rows(variants)).to_parquet(table_path, index=False)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    blocks = report["block_counts"]
    print(
        f"[classII] {report['n_variants']} variants / {report['n_parents']} parents | "
        f"blocks: clusters={blocks['cluster_id']} orders={blocks['resolved_order']} "
        f"(min-N {report['min_n']}) | control strata={report['control_strata_parent_counts']} "
        f"| gate={'PASS' if report['gate']['overall_pass'] else 'FAIL'} → {report_path}"
    )
    return 0 if report["gate"]["overall_pass"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
