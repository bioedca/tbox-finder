"""P2-12 — the Stage-1 architecture-ablation table (RC-combination, head, size x context).

**What this is, and why it is not ``select_best``.** ``train/select_best.py`` reduces the
P2-06 sweep: it *selects a winner* on ``gate4_core_min_f1.min_f1`` over the γ × lr × α grid,
and its ``SWEPT_AXES`` is hardcoded ``("gamma", "lr", "class_weight_alpha")`` — so the axis
P2-12 varies is invisible to its ``_axes_of`` and every ablation leg would project onto the
same three (identical) values. Widening ``SWEPT_AXES`` in place would silently change the
committed ``reports/p2/sweep_selection.json`` artifact and the shape-lock test over it
(``tests/ml/test_sweep_selection.py``), so P2-12 ships a **separate reducer** over its own
axes and *reuses* select_best's point hygiene rather than forking it: ``point_problems``,
``load_reports`` and ``cross_point_consistency`` are imported, not reimplemented. A forked
copy means fixing one and shipping the bug in the other.

**It ranks on the interval, not on the point.** The P2-06 sweep already demonstrated why: its
winner's 95% CI (width ≈ 0.043) contained all four runners-up and the margin over second place
was 0.0011. A table that sorts by point estimate and stops there invites reading a
fortieth-of-a-CI gap as a finding. So rows are ordered by the **lower CI bound** (the
conservative statistic), the point estimate and the full interval are both shown, and every
row carries ``separated_from_baseline`` — TRUE only when the row's CI does **not** overlap the
baseline's. With 4 legs against one baseline on an 830-record fold, the expected answer is
"not separated", and saying so is the deliverable (PRD §10.1 asks which checkpoint/head to
ship, and "the axes do not separate at this power" is a legitimate, reportable answer).

**The baseline row costs 0 new GPU-h.** It is the promoted P2-06 sweep point
``reports/p2/sweep/g0.5_lr3e-4_a0.5.json`` — trained on the identical protocol
(``inner_train``, ``exclude_selection_val=true``, positives-only, DDP×8 fp32, seed 42,
γ=0.5 / lr=3e-4 / α=0.5, epochs 10, batch 8) and scored on the identical 830-record /
469-block ``selection_val`` fold — with ``rc_combine=concat`` and ``use_crf=false``, i.e.
exactly the (i)/(iii) baseline arm. It was trained at an **earlier commit**, which is a real
difference the artifact reports (``cross_point_consistency`` re-derives it) rather than
hides; it is disclosed, not gated, because re-training it would spend 1.9 GPU-h to reproduce
a committed measurement.

**The production checkpoint has no row and cannot get one.** It trained with ``selection_val``
*included* (P2-09), so it has no held-out val metric on this fold. That is why the baseline is
the promoted sweep point rather than the shipped model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder.models.backbone_registry import (
    PRODUCTION_CHECKPOINT,
    checkpoint_for_repo_id,
    resolve_checkpoint,
)
from tbox_finder.train.select_best import (
    REQUIRED_FOLD_SCOPE,
    SELECTION_STATISTIC,
    cross_point_consistency,
    load_reports,
    point_problems,
)

SCHEMA_VERSION = "1"
STEP = "P2-12"
GENERATED_BY = "src/tbox_finder/train/ablation_table.py"
ADR = "ADR-0003 A3; ADR-0002 A14; ADR-0005 D15 + D3/A3"
ENV_LOCK = "envs/ml-dna.conda-lock.yml"

#: The axes P2-12 varies (PRD §6/§10.1/§18.1; user decision 2026-07-29). Deliberately a
#: separate tuple from ``select_best.SWEPT_AXES`` — see the module docstring.
ABLATION_AXES: tuple[str, ...] = ("backbone", "rc_combine", "use_crf")

DEFAULT_BASELINE = "reports/p2/sweep/g0.5_lr3e-4_a0.5.json"
DEFAULT_ABLATION_DIR = "reports/p2/ablation"
DEFAULT_OUT = "reports/p2/ablation_table.json"

#: §10.3 disclosures the table must carry. They are properties of the *protocol*, not of any
#: one run, so they are constants here and are emitted with every artifact — a caveat that
#: lives only in a commit message does not reach the reader of the table.
DISCLOSURES: tuple[str, ...] = (
    "Every leg is POSITIVES-ONLY, while the shipped Stage-1 stack carries the ADR-0005 D14 "
    "10:1 mined-negative mix. This is the same protocol on which gamma/lr/alpha were selected "
    "and promoted (P2-06), which is what makes the baseline row reusable at 0 new GPU-h; it is "
    "not the shipped training distribution.",
    "The CRF leg trains a structured (CRF) objective but is GRADED under the frozen ADR-0005 "
    "D3+A3 reconcile->argmax operator: Viterbi decoding is NOT exercised. A CRF's benefit is "
    "partly a decode-time benefit, so this arm measures the CRF as a training signal only.",
    "The size axis moves parameter count AND pretraining context together — both smaller "
    "checkpoints are seqlen-1k while production is seqlen-131k (ADR-0002 A14). It is a joint "
    "size x context arm; no delta may be attributed to size alone.",
    "selection_val is 810/830 Firmicutes and is pooled MICRO (metrics.macro_average has zero "
    "src/ callers, so ADR-0005 D5's macro-over-held-out-orders half is unwired). The point "
    "estimate is Firmicutes-dominated; the CI is cluster-blocked over 469 homology clusters.",
    "The scan geometry is HELD at ADR-0005 D3+A3's 1024/512 for every leg, including the two "
    "seqlen-1k checkpoints, so no leg unpins the reconciliation geometry.",
    "The order-destroying averaged RC combination is NOT run. ADR-0005 D15 + PRD §6/§10.1 "
    "forbid it outright and models/rc_combine.py raises on it; it is not a negative control "
    "and its absence is not a gap in this table.",
    "This is a SELECTION-rung table, not GATE-4. GATE-4 is graded at P2-14 on the real "
    "leave-one-order-out holdout; no number here is a generalization result.",
)


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def axes_of(report: Mapping[str, Any]) -> dict[str, Any]:
    """The P2-12 axis values of one leg, **re-derived** off its own recorded evidence.

    ``backbone`` needs care. It became a config field at P2-12, so the 36 committed P2-06
    sweep reports — one of which *is* the baseline row — record no such key; they record only
    the ``backbone.repo_id`` / ``revision`` block the run echoed. Defaulting a missing axis to
    "production" would be an assumption of exactly the kind that lets a differently-trained
    point join the table unnoticed, so it is instead resolved from the recorded ``repo_id``
    through the allow-list. A report that records neither yields ``None``, and a ``None`` axis
    makes the row non-comparable (see :func:`row_problems`) rather than silently baseline.
    """
    diag = report.get("diagnostics")
    diag = diag if isinstance(diag, Mapping) else {}
    cfg = diag.get("config")
    cfg = cfg if isinstance(cfg, Mapping) else {}
    axes: dict[str, Any] = {a: cfg.get(a) for a in ABLATION_AXES}
    if axes.get("backbone") is None:
        block = report.get("backbone")
        block = block if isinstance(block, Mapping) else {}
        spec = checkpoint_for_repo_id(str(block.get("repo_id")))
        # Re-derive only when the RECORDED revision also matches the pin: a repo id alone
        # names a repository, not a checkpoint.
        if spec is not None and block.get("revision") == spec.revision:
            axes["backbone"] = spec.key
            axes["backbone_source"] = (
                "re-derived from backbone.repo_id + revision (pre-P2-12 schema)"
            )
    return axes


def _ci_of(report: Mapping[str, Any]) -> dict[str, Any] | None:
    metrics = report.get("eval_metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    ci = metrics.get("block_bootstrap_ci")
    return dict(ci) if isinstance(ci, Mapping) else None


def _overlaps(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool | None:
    """Do two CIs overlap? ``None`` when either interval is missing or malformed.

    ``None`` is load-bearing: "the intervals do not overlap" and "we could not tell" must not
    render identically, or an unmeasurable row reads as a separated one.
    """
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return None
    lo_a, hi_a, lo_b, hi_b = a.get("lower"), a.get("upper"), b.get("lower"), b.get("upper")
    if not all(_is_real(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return None
    return float(lo_a) <= float(hi_b) and float(lo_b) <= float(hi_a)


def row_problems(report: Mapping[str, Any]) -> list[str]:
    """Why this leg cannot join the table, or ``[]``.

    select_best's ``point_problems`` already enforces the comparability floor every row needs
    — gate passed, scored on the FULL ``selection_val`` fold, zero measured leakage against
    ``inner_train``, block count consistent with the CI. Reused wholesale; P2-12 adds only
    what is specific to an ablation row: a resolvable axis tuple and a usable interval.
    """
    problems = list(point_problems(report))
    axes = axes_of(report)
    for axis in ABLATION_AXES:
        if axes.get(axis) is None:
            problems.append(
                f"diagnostics.config.{axis} is absent and could not be re-derived — the row "
                "cannot be placed on the ablation axis it is supposed to occupy"
            )
    backbone = axes.get("backbone")
    if backbone is not None:
        try:
            resolve_checkpoint(backbone)
        except ValueError as exc:
            problems.append(f"backbone axis: {exc}")
    ci = _ci_of(report)
    if not isinstance(ci, Mapping) or not all(_is_real(ci.get(k)) for k in ("lower", "upper")):
        problems.append(
            "eval_metrics.block_bootstrap_ci: missing finite lower/upper — this table ranks "
            "on the interval, so a row without one cannot be ranked"
        )
    return problems


def _row(report: Mapping[str, Any], index: int) -> dict[str, Any]:
    metrics = report["eval_metrics"]
    gate4 = metrics["gate4_core_min_f1"]
    axes = axes_of(report)
    backbone = resolve_checkpoint(axes["backbone"])
    return {
        "leg": str(report.get("report_path") or report.get("point") or index),
        "axes": axes,
        "backbone_params": backbone.expected_param_count,
        "backbone_pretrain_seq_len": backbone.pretrain_seq_len,
        "score": float(gate4["min_f1"]),
        "ci": _ci_of(report),
        "per_element_f1": gate4.get("per_element_f1"),
        "n_eval_records": (report.get("eval_scope") or {}).get("n_records"),
        "n_eval_blocks": (report.get("eval_scope") or {}).get("n_blocks"),
        "git_sha": (report.get("provenance") or {}).get("git_sha"),
    }


def build_table(
    reports: Sequence[Mapping[str, Any]], *, baseline_leg: str | None = None
) -> dict[str, Any]:
    """Rank the ablation legs on the lower CI bound and compare each to the baseline.

    Returns ``{rows, baseline, n_rows, n_rejected, rejected, …}``. No winner is named: PRD
    §10.1's decision is which checkpoint/head to *ship*, and shipping on a point estimate
    whose CI contains every alternative is exactly the reading this reducer refuses to
    support. The table reports separation; the shipped-config choice is P2-14's (imp.md
    "Blocks: P2-14 (informs the shipped-config choice)").
    """
    rows: list[dict[str, Any]] = []
    accepted: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for i, r in enumerate(reports):
        label = str((r or {}).get("report_path") or (r or {}).get("point") or i)
        problems = row_problems(r if isinstance(r, Mapping) else {})
        if problems:
            rejected.append({"leg": label, "problems": problems})
            continue
        accepted.append(r)
        rows.append(_row(r, i))

    baseline = None
    if baseline_leg is not None:
        baseline = next((row for row in rows if row["leg"] == baseline_leg), None)
    if baseline is None:
        # Fall back to the row whose axes ARE the baseline arm (production backbone, concat,
        # no CRF) — re-derived from the axes rather than assumed to be a particular file.
        baseline = next(
            (
                row
                for row in rows
                if row["axes"].get("backbone") == PRODUCTION_CHECKPOINT
                and row["axes"].get("rc_combine") == "concat"
                and row["axes"].get("use_crf") is False
            ),
            None,
        )

    base_ci = baseline["ci"] if baseline else None
    for row in rows:
        row["is_baseline"] = bool(baseline is not None and row["leg"] == baseline["leg"])
        overlap = None if row["is_baseline"] else _overlaps(row["ci"], base_ci)
        row["ci_overlaps_baseline"] = overlap
        # TRUE only on a measured non-overlap. `None` (unmeasurable) must not read as
        # separated, and neither must the baseline's own row.
        row["separated_from_baseline"] = (overlap is False) if overlap is not None else None
        row["delta_vs_baseline"] = (
            float(row["score"]) - float(baseline["score"]) if baseline else None
        )

    # Rank on the LOWER CI bound (conservative), then the point, then the label — total and
    # deterministic, so a re-reduction of the same reports orders them identically (§8.3).
    rows.sort(key=lambda r: (-float(r["ci"]["lower"]), -float(r["score"]), str(r["leg"])))

    n_separated = sum(1 for r in rows if r.get("separated_from_baseline") is True)
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "env_lock": ENV_LOCK,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "is_science": False,
        "gate4_graded": False,
        "axes": list(ABLATION_AXES),
        "selected_on": SELECTION_STATISTIC,
        "ranked_on": "eval_metrics.block_bootstrap_ci.lower (the conservative bound), then "
        "the point estimate, then the leg label",
        "fold_scope": REQUIRED_FOLD_SCOPE,
        "baseline": baseline,
        "n_rows": len(rows),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "n_separated_from_baseline": n_separated,
        # The headline sentence, re-derived rather than narrated. With 4 legs on an
        # 830-record fold the expected value is 0, and reporting that plainly is the point.
        "any_axis_separated_from_baseline": bool(n_separated),
        "cross_point_consistency": cross_point_consistency(accepted),
        "disclosures": list(DISCLOSURES),
        "rows": rows,
    }


def artifact_problems(artifact: Mapping[str, Any], *, expect_rows: int) -> list[str]:
    """Validate a built table. Total — never raises, reports. Fails closed on emptiness.

    ``expect_rows`` is required, not optional: a table of zero rows is structurally valid and
    reads exactly like a table of clean rows in any ``all(...)``/``any(...)`` summary. The
    caller must say how many legs it expected to find ([[clauses-must-guard-emptiness]]).
    """
    problems: list[str] = []
    if not isinstance(artifact, Mapping):
        return [f"artifact must be a mapping, got {type(artifact).__name__}"]
    for key in ("schema_version", "step", "generated_by", "adr", "env_lock", "ranked_on"):
        if not isinstance(artifact.get(key), str) or not artifact.get(key):
            problems.append(f"{key}: missing or not a non-empty string")
    if artifact.get("is_science") is not False:
        problems.append("is_science: a selection-rung table is not a GATE-4 result (§10.3)")
    if artifact.get("gate4_graded") is not False:
        problems.append("gate4_graded: GATE-4 is graded at P2-14, never here")
    missing = [d for d in DISCLOSURES if d not in (artifact.get("disclosures") or [])]
    if missing:
        problems.append(f"disclosures: {len(missing)} required disclosure(s) missing")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        return [*problems, "rows: missing or not a list"]
    if len(rows) != expect_rows:
        problems.append(f"rows: {len(rows)} != expected {expect_rows}")
    if artifact.get("rejected"):
        problems.append(
            f"rejected: {len(artifact['rejected'])} leg(s) were not comparable — a partial "
            "ablation table must not be read as a complete one"
        )
    baseline = artifact.get("baseline")
    if not isinstance(baseline, Mapping):
        problems.append("baseline: absent — every row's separation is measured against it")
    seen: set[tuple] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            problems.append(f"rows[{i}]: not a mapping")
            continue
        axes = row.get("axes")
        if not isinstance(axes, Mapping):
            problems.append(f"rows[{i}]: axes missing")
            continue
        key = tuple(axes.get(a) for a in ABLATION_AXES)
        if key in seen:
            problems.append(f"rows[{i}]: duplicate axis tuple {key!r} — two runs of one leg")
        seen.add(key)
        if not _is_real(row.get("score")):
            problems.append(f"rows[{i}]: score is not a finite real")
        ci = row.get("ci")
        if not isinstance(ci, Mapping) or not all(
            _is_real(ci.get(k)) for k in ("lower", "point", "upper")
        ):
            problems.append(f"rows[{i}]: ci must carry finite lower/point/upper")
            continue
        if not float(ci["lower"]) <= float(ci["upper"]):
            problems.append(f"rows[{i}]: ci.lower > ci.upper")
        # Re-derive the separation verdict from the intervals rather than trusting the
        # stored boolean — a flag echoed by the builder is not evidence (P1-15/P1-16).
        if not row.get("is_baseline") and isinstance(baseline, Mapping):
            want = _overlaps(ci, baseline.get("ci"))
            want_sep = (want is False) if want is not None else None
            if row.get("separated_from_baseline") is not want_sep:
                problems.append(
                    f"rows[{i}]: separated_from_baseline {row.get('separated_from_baseline')!r} "
                    f"does not re-derive from the recorded intervals ({want_sep!r})"
                )
    n_sep = sum(1 for r in rows if isinstance(r, Mapping) and r.get("separated_from_baseline"))
    if artifact.get("n_separated_from_baseline") != n_sep:
        problems.append(
            f"n_separated_from_baseline {artifact.get('n_separated_from_baseline')!r} != the "
            f"{n_sep} row(s) that actually carry it"
        )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="the reused P2-06 point")
    parser.add_argument("--ablation-dir", default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--expect-rows",
        type=int,
        required=True,
        help="how many legs must appear (baseline included). No default: an empty table is "
        "structurally valid and must not pass for a complete one.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    leg_paths = sorted(Path(args.ablation_dir).glob("*.json"))
    paths = [Path(args.baseline), *leg_paths]
    reports = load_reports(paths)
    artifact = build_table(reports, baseline_leg=str(Path(args.baseline)))
    problems = artifact_problems(artifact, expect_rows=args.expect_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # An invalid table must NOT land on the canonical path. Returning 2 while having already
    # written it leaves an artifact that reads as the deliverable to anything that opens the
    # file rather than the exit code — and this artifact is git-committed evidence. So the
    # rejected build is diverted to a clearly-marked sibling, which keeps it inspectable
    # without letting it impersonate a validated table (CodeRabbit, P2-12 RESULT).
    if problems:
        out = out.with_suffix(".invalid.json")
        out.write_text(payload)
        print(f"wrote {out} — NOT PUBLISHED: the table failed its own validator")
    else:
        out.write_text(payload)
        print(f"wrote {out} ({artifact['n_rows']} rows, {artifact['n_rejected']} rejected)")
    for row in artifact["rows"]:
        axes = row["axes"]
        flag = "baseline" if row["is_baseline"] else f"sep={row['separated_from_baseline']}"
        print(
            f"  {row['score']:.4f} [{row['ci']['lower']:.4f}, {row['ci']['upper']:.4f}]  "
            f"{axes['backbone']:<28} rc={axes['rc_combine']:<7} crf={axes['use_crf']!s:<5} {flag}"
        )
    if problems:
        print("TABLE INVALID:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
