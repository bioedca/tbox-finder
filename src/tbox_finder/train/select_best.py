"""P2-06a — promote the best sweep config on the validation ladder (PRD §11:233).

**What this module is.** A pure, torch-free reducer over N ``train_stage1`` reports — one
per Hydra ``--multirun`` sweep point — that names the winner. It reads only what those
reports already record; nothing here re-computes a metric, and nothing here may invent one.

**The rung, and why it is the safe one.** Every report is scored on ``eval_scope.fold_scope
== "selection_val"``: the leak-free scheme-A (random, genus-stratified) val fold built by
``window_dataset.load_selection_val_records``. ADR-0004 D5 designates scheme-A the
*"detection-quality reference only; it is never the source of a generalization number"* —
which is precisely what makes it the right place to tune. Selecting here cannot contaminate
the PRD §12:241 leave-one-order-out headline, so that headline stays a genuine held-out
estimate. This is the inner rung of a nested design (user decision, 2026-07-17).

**The statistic.** ``eval_metrics.gate4_core_min_f1.min_f1`` — the minimum per-nt F1 over
the three core elements {Stem I, Specifier, Antiterminator} (ADR-0004 D6: a **min**, never
a mean; the elements are a commensurable per-nt-class unit). Two consequences are load-
bearing rather than incidental:

* It is **not** comparable to training loss, and that is the point. Focal-CE loss is a
  function of ``gamma`` and ``class_weight_alpha`` themselves, so ranking sweep points on
  loss across P2-06's γ/α grid would compare *objectives*, not *models*. GATE-4's statistic
  is objective-independent.
* NaN means an unmeasurable core element, and a NaN **never wins** — it is not a low score,
  it is an absent one. ``gate4_core_min_f1`` already refuses to certify on NaN; this module
  refuses to promote on it.

**What this module does NOT do.** It does not grade GATE-4 (that is P2-14, on the real
split, and the 0.80 floor is not applied here — a sweep's job is to rank, not to certify).
It does not touch the test set or the LOO-order holdout. It does not write a conf/ overlay;
P2-06 does that from the winner this returns.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: The statistic the ladder ranks on (ADR-0004 D6 via metrics.gate4_core_min_f1).
SELECTION_STATISTIC = "gate4_core_min_f1.min_f1"

#: The only fold scope a sweep point may be promoted on. A report scored on anything else
#: — the training fold, the LOO-order holdout, the test set — is rejected, not down-ranked.
REQUIRED_FOLD_SCOPE = "selection_val"

#: The axes P2-06 sweeps (user decision 2026-07-17). Recorded so the summary table names
#: the axis values that actually varied rather than dumping the whole config.
SWEPT_AXES: tuple[str, ...] = ("gamma", "lr", "class_weight_alpha")


def _is_real(v: Any) -> bool:
    """True for a non-bool finite real. Rejects bools, NaN and inf."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def point_problems(report: Mapping[str, Any]) -> list[str]:
    """Why ``report`` is not a promotable sweep point; ``[]`` when it is.

    Total by construction, and **fail-closed**: an unreadable or unscored point is excluded
    from the ranking rather than treated as a zero. A point silently scored 0.0 would still
    lose an honest comparison, but a point silently scored on the *wrong fold* would win one
    it had no right to — so the fold scope is checked before the number is read.
    """
    problems: list[str] = []
    if not isinstance(report, Mapping) or not report:
        return ["report: not a mapping, or empty"]

    gate = report.get("gate")
    if not isinstance(gate, Mapping) or not gate:
        problems.append("gate: block missing — an ungated run is not a sweep point")
    elif gate.get("overall_pass") is not True:
        failed = sorted(k for k, v in gate.items() if k != "overall_pass" and v is not True)
        problems.append(f"gate.overall_pass is not True (failed clauses: {failed or 'unknown'})")

    scope = report.get("eval_scope")
    if not isinstance(scope, Mapping) or not scope:
        problems.append("eval_scope: block missing — the point carries no val evidence")
    else:
        if scope.get("fold_scope") != REQUIRED_FOLD_SCOPE:
            problems.append(
                f"eval_scope.fold_scope = {scope.get('fold_scope')!r}, must be "
                f"{REQUIRED_FOLD_SCOPE!r} — a config promoted on any other fold is either "
                "train-on-train or contaminates the PRD §12:241 headline"
            )
        dis = scope.get("disjointness")
        if not isinstance(dis, Mapping) or not dis:
            problems.append("eval_scope.disjointness: block missing")
        else:
            for key in ("shared_record_ids_with_train", "shared_cluster_ids_with_train"):
                val = dis.get(key)
                if not isinstance(val, int) or isinstance(val, bool) or val != 0:
                    problems.append(f"eval_scope.disjointness.{key} = {val!r}, must be 0")

    metrics_block = report.get("eval_metrics")
    if not isinstance(metrics_block, Mapping) or not metrics_block:
        problems.append("eval_metrics: block missing — nothing to rank on")
    else:
        gate4 = metrics_block.get("gate4_core_min_f1")
        if not isinstance(gate4, Mapping) or not gate4:
            problems.append("eval_metrics.gate4_core_min_f1: block missing")
        elif not _is_real(gate4.get("min_f1")):
            problems.append(
                f"eval_metrics.gate4_core_min_f1.min_f1 = {gate4.get('min_f1')!r} is not a "
                "finite real — an unmeasurable core element cannot win a sweep"
            )
    return problems


def score_of(report: Mapping[str, Any]) -> float:
    """The ranked statistic. Raises on an unpromotable point rather than returning a
    sentinel — a sentinel score sorts, and a point that cannot be scored must not."""
    problems = point_problems(report)
    if problems:
        raise ValueError("not a promotable sweep point:\n  " + "\n  ".join(problems))
    return float(report["eval_metrics"]["gate4_core_min_f1"]["min_f1"])


def _axes_of(report: Mapping[str, Any]) -> dict[str, Any]:
    """The swept axis values of one point, off its recorded config."""
    diag = report.get("diagnostics")
    diag = diag if isinstance(diag, Mapping) else {}
    cfg = diag.get("config")
    cfg = cfg if isinstance(cfg, Mapping) else {}
    return {a: cfg.get(a) for a in SWEPT_AXES}


def select_best(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank sweep points and name the winner.

    Returns ``{winner, ranking, n_points, n_promotable, n_rejected, rejected, ...}``.
    ``winner`` is ``None`` when nothing is promotable — an empty ladder yields **no
    winner**, never an arbitrary first element (CLAUDE.md §10.3: withhold rather than emit
    an unfounded result).

    Ties break on the **lower** ``lr`` then the **lower** ``gamma``, deterministically, so a
    re-run of the same sweep promotes the same config (§8.3). The tie-break is a convention,
    not a finding: it prefers the more conservative point, and it is recorded in the output
    so a reader can see a tie happened rather than infer a decisive win.
    """
    promotable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for i, r in enumerate(reports):
        problems = point_problems(r)
        label = str((r or {}).get("report_path") or (r or {}).get("point") or i)
        if problems:
            rejected.append({"point": label, "problems": problems})
            continue
        axes = _axes_of(r)
        promotable.append(
            {
                "point": label,
                "score": float(r["eval_metrics"]["gate4_core_min_f1"]["min_f1"]),
                "axes": axes,
                "ci": r["eval_metrics"].get("block_bootstrap_ci"),
                "per_element_f1": r["eval_metrics"]["gate4_core_min_f1"].get("per_element_f1"),
            }
        )

    def _sort_key(p: Mapping[str, Any]) -> tuple:
        axes = p["axes"]
        lr = axes.get("lr")
        gamma = axes.get("gamma")
        return (
            -p["score"],
            float(lr) if _is_real(lr) else math.inf,
            float(gamma) if _is_real(gamma) else math.inf,
            str(p["point"]),
        )

    promotable.sort(key=_sort_key)
    winner = promotable[0] if promotable else None
    tied = [p for p in promotable if winner and p["score"] == winner["score"]]
    return {
        "schema_version": "1",
        "step": "P2-06a",
        "selected_on": SELECTION_STATISTIC,
        "fold_scope": REQUIRED_FOLD_SCOPE,
        "ladder_rung": "PRD §9.2 (a) random/genus-stratified — ADR-0004 D5 reference fold",
        "never_selected_on": ["test", "loo_order_unit"],
        "n_points": len(list(reports)),
        "n_promotable": len(promotable),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "winner": winner,
        "n_tied_at_winning_score": len(tied),
        "tie_break": "lower lr, then lower gamma, then point label (deterministic)",
        "ranking": promotable,
    }


def load_reports(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read sweep-point reports off disk, tagging each with the path it came from.

    An unreadable/unparseable file becomes a ``{}`` placeholder rather than an exception:
    ``select_best`` then rejects it *by name* in the summary. A sweep of 36 points must not
    lose 35 good results to one truncated JSON — but the loss must be visible, never silent.
    """
    out: list[dict[str, Any]] = []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            out.append({"report_path": str(p)})
            continue
        if isinstance(data, dict):
            data.setdefault("report_path", str(p))
            out.append(data)
        else:
            out.append({"report_path": str(p)})
    return out
