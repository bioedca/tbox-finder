"""P1-08 — Stage-1 segmentation-smoke **reproducibility** gate (ADR-0002 **A7**).

PRD §18.1's Phase-1 exit gate requires *"smoke run reproducible"*: a seeded re-run of
the P1-07 **binding** transfer go/no-go (ADR-0002 A6) must reproduce its go/no-go metrics.
The re-run is **not bit-exact** — the Caduceus Mamba ``selective_scan_cuda`` kernel
(ADR-0002 A2 C2) registers no deterministic algorithm, so the P1-07 harness runs
``torch.use_deterministic_algorithms(True, warn_only=True)`` (every RNG seed pinned,
TF32/cudnn autotune off; PRD §8.3/§11), but the fused scan may accumulate floating-point
differently run-to-run. The tolerance is therefore **metric-level, not bitwise**.

This module is the **pure-stdlib** comparison the P1-08 gate is built on (no numpy / torch
/ pandas — it imports bare and runs in CI). It compares the committed P1-07 reference
report (``reports/p1/seg_smoke_gonogo.json``) against the P1-08 re-run report
(``reports/p1/seg_smoke_repro.json``, produced by the **unchanged** ``stage1_smoke.py``
harness with only ``report_path=``/``checkpoint_dir=`` overridden) and decides
reproducibility at **two levels** (ADR-0002 A7):

1. **Primary (load-bearing):** the **go/no-go verdict reproduces** (re-run == the
   reference verdict, ``GO``). This is the exit-gate's meaning of *reproducible*: the
   transfer decision that gates P2 is stable under a seeded re-run.
2. **Secondary (determinism health-check):** the **max absolute difference** over the
   **go/no-go-decision** metrics (:data:`GATED_METRIC_KEYS` — ``min_core_f1``, the 3 core
   ``per_element_f1.*``, ``macro_f1``, ``micro_f1``) is **≤ τ** (:data:`REPRO_TOLERANCE`).
   The remaining per-class F1s (:data:`DIAGNOSTIC_METRIC_KEYS`) are reported as a
   **determinism diagnostic** (max |Δ| surfaced), **not** τ-gated — the rarer non-core
   classes have larger per-argmax-flip F1 sensitivity under the irreducibly-nondeterministic
   Mamba kernel (§8.3). All 14 metrics must still be **present** (a structural fail-closed
   floor); only the numeric τ comparison is scoped to the decision metrics.

The gate also asserts the two runs share the **same config** (seed / epochs / optimiser /
loss / rc-combine / … ) and the **same pinned backbone revision** — a re-run under a
*different* config is not a reproducibility test, so a config mismatch voids the gate
(fail-closed, §10.3).

**τ is pinned a priori** (ADR-0002 A7, before the re-run) and is **not weakenable on a
fail**: an observed diff > τ is a CLAUDE.md §7 stop-and-ask (chase the nondeterministic op,
or list it unit-test-only per §8.3), never a loosened tolerance (§8.5/§10.3). Every
predicate here **fails closed** — a missing key, a non-finite value, a mismatched key set,
or a differing config all yield *not reproducible*.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tbox_finder.labels import CLASS_ORDER, CORE_ELEMENTS

# --------------------------------------------------------------------------------------
# Pinned constants (ADR-0002 A7). Single-sourced in code — never re-declared in the test
# or loosened by config.
# --------------------------------------------------------------------------------------
#: ADR-0002 A7 metric-level reproducibility tolerance (absolute F1). Applies to the
#: **go/no-go-decision** metrics (:data:`GATED_METRIC_KEYS`): max|Δ| over them must be ≤ this
#: for the secondary determinism health-check to pass. Not weakenable on a fail (§8.5/§10.3).
REPRO_TOLERANCE: float = 1e-3

#: Sentinel for an absent value (a report block/key that isn't there). Distinguished from a
#: present ``None`` so a *missing* backbone/config datum voids the gate rather than matching
#: another missing one (``None == None`` would fail open).
_MISSING = object()

#: The **go/no-go-decision** metrics the secondary τ gate is scoped to (ADR-0002 A7, scoped by
#: the P1-08 re-sign-off): the ``min_core_f1`` the verdict rests on (the min over the 3 core
#: elements, ADR-0004 D6), the 3 core ``per_element_f1.*``, and the ``macro_f1``/``micro_f1``
#: aggregates. Single-sourced from the label vocabulary. These are exactly the quantities the
#: go/no-go decision depends on, and they must reproduce within τ.
GATED_METRIC_KEYS: tuple[str, ...] = ("macro_f1", "micro_f1", "min_core_f1") + tuple(
    f"per_element_f1.{name}" for name in CORE_ELEMENTS
)

#: The remaining reported per-class F1s — surfaced as a **determinism diagnostic** (their max
#: |Δ| is reported), **not** τ-gated: the rarer non-core classes (e.g. ``Stem_III``) span far
#: fewer positions, so a single argmax flip under the irreducibly-nondeterministic Mamba
#: kernel (§8.3, no deterministic variant) moves their F1 more than the core elements'. Their
#: drift is disclosed, not gated at the decision τ (P1-08 re-sign-off).
DIAGNOSTIC_METRIC_KEYS: tuple[str, ...] = tuple(f"per_class_f1.{name}" for name in CLASS_ORDER)

#: The full reported metric set (14 scalars = gated ∪ diagnostic). The gate requires **all**
#: of these **present + finite** in both reports (a structural fail-closed floor — a truncated
#: report, even a symmetric reduction, voids the gate); the numeric τ comparison is scoped to
#: :data:`GATED_METRIC_KEYS`, the rest reported as the diagnostic.
EXPECTED_METRIC_KEYS: tuple[str, ...] = GATED_METRIC_KEYS + DIAGNOSTIC_METRIC_KEYS

#: Config knobs that must be identical for the re-run to be a reproducibility test (the
#: :meth:`SmokeConfig.sanitized_knobs` set — paths are intentionally excluded, only the run
#: knobs). A mismatch on any voids the gate (fail-closed).
CONFIG_KEYS: tuple[str, ...] = (
    "seed",
    "epochs",
    "lr",
    "weight_decay",
    "grad_clip",
    "gamma",
    "rc_combine",
    "use_crf",
    "dropout",
    "batch_size",
)


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------
def _is_finite_number(v: Any) -> bool:
    """True iff ``v`` is a real, finite number (bools rejected)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a go/no-go / repro report JSON as a dict. Raises on missing/invalid JSON."""
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: report must be a JSON object, got {type(obj).__name__}")
    return obj


def flatten_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a report's ``metrics`` block to ``{dotted_key: value}``.

    Nested dicts (``per_element_f1``, ``per_class_f1``) expand to ``block.name`` keys; scalars
    stay as-is. Values may be ``None`` (a class absent from truth+pred → NaN → the harness
    sanitises to ``None``). Returns ``{}`` if ``metrics`` is missing/not a dict (→ the
    downstream key-set check fails closed).
    """
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    flat: dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, Mapping):
            for sub, subval in val.items():
                flat[f"{key}.{sub}"] = subval
        else:
            flat[key] = val
    return flat


def metric_abs_diffs(ref: Mapping[str, Any], rerun: Mapping[str, Any]) -> dict[str, float | None]:
    """Per-metric ``|ref − rerun|`` over the **union** of both reports' flattened metric keys.

    A key present in only one report, or whose value is non-finite in either, maps to
    ``None`` (uncomparable → fail-closed downstream). Both reports' full metric sets are
    covered so a *truncated* re-run report (fewer metrics) is caught, not silently ignored.
    """
    fref = flatten_metrics(ref)
    frer = flatten_metrics(rerun)
    diffs: dict[str, float | None] = {}
    for key in sorted(set(fref) | set(frer)):
        a = fref.get(key)
        b = frer.get(key)
        if key in fref and key in frer and _is_finite_number(a) and _is_finite_number(b):
            diffs[key] = abs(float(a) - float(b))
        else:
            diffs[key] = None
    return diffs


def verdict_of(report: Mapping[str, Any]) -> Any:
    """The go/no-go verdict (``report["gate"]["verdict"]``); ``None`` if absent."""
    gate = report.get("gate")
    return gate.get("verdict") if isinstance(gate, Mapping) else None


def _config_of(report: Mapping[str, Any]) -> dict[str, Any]:
    cfg = report.get("config")
    return dict(cfg) if isinstance(cfg, Mapping) else {}


def _backbone_revision_of(report: Mapping[str, Any]) -> Any:
    """The pinned backbone revision, or :data:`_MISSING` if the block/key is absent or null.

    Returns the sentinel (not ``None``) for an absent **or explicitly-null** revision so
    :func:`config_mismatches` treats *both* reports lacking it as a mismatch, not a match — a
    re-run whose backbone provenance is unknown is not a reproducibility test (fail-closed).
    """
    bb = report.get("backbone")
    revision = bb.get("revision") if isinstance(bb, Mapping) else None
    return _MISSING if revision is None else revision


def config_mismatches(ref: Mapping[str, Any], rerun: Mapping[str, Any]) -> list[str]:
    """Return a list of config/backbone keys that differ between the two runs (empty ⇒ same).

    A reproducibility test requires an **identical** config + pinned backbone; any difference
    (or a missing key) is reported so the gate can void itself (fail-closed).
    """
    cref = _config_of(ref)
    crer = _config_of(rerun)
    out: list[str] = []
    for key in CONFIG_KEYS:
        if key not in cref or key not in crer or cref.get(key) != crer.get(key):
            out.append(f"config.{key}")
    rev_ref = _backbone_revision_of(ref)
    rev_rerun = _backbone_revision_of(rerun)
    # A missing revision (_MISSING) on either side, or two differing revisions, is a mismatch.
    if rev_ref is _MISSING or rev_rerun is _MISSING or rev_ref != rev_rerun:
        out.append("backbone.revision")
    return out


def check_reproducibility(
    ref: Mapping[str, Any],
    rerun: Mapping[str, Any],
    *,
    tolerance: float = REPRO_TOLERANCE,
) -> dict[str, Any]:
    """Adjudicate the ADR-0002 A7 two-level reproducibility gate (fail-closed).

    Returns a structured result:

    - ``verdict_reproduces`` — both verdicts present and equal (**primary**, load-bearing).
    - ``config_ok`` / ``config_mismatches`` — the runs share the same config + pinned backbone.
    - ``per_metric_abs_diff`` — ``{key: |Δ| or None}`` over both reports' metrics.
    - ``all_metrics_comparable`` — every metric finite in both (no ``None``).
    - ``expected_metrics_present`` — the flattened metric set is **exactly** the 14
      :data:`EXPECTED_METRIC_KEYS` and every one finite in both (a structural floor: a
      truncation, a symmetric reduction, or an extra/renamed key is caught here, not passed).
    - ``max_abs_diff`` — max finite ``|Δ|`` over **all** metrics (for transparency).
    - ``gated_max_abs_diff`` — max ``|Δ|`` over :data:`GATED_METRIC_KEYS` (the τ-gated quantity).
    - ``diagnostic_max_abs_diff`` — max ``|Δ|`` over :data:`DIAGNOSTIC_METRIC_KEYS` (reported,
      **not** τ-gated: the rarer non-core per-class F1 determinism diagnostic).
    - ``within_tolerance`` — the full expected set present **and** ``gated_max_abs_diff ≤
      tolerance`` (**secondary**; scoped to the decision metrics, ADR-0002 A7 P1-08 re-sign-off).
    - ``reproducible`` — ``verdict_reproduces AND config_ok AND within_tolerance``.

    Any gap (absent verdict/gated-metric, non-finite value, a missing expected metric, a
    mismatched metric key set, differing config) forces ``reproducible = False`` — it never
    certifies reproducibility on missing evidence (§10.3). A large diagnostic drift alone does
    **not** fail the gate (it is disclosed, not gated).

    ``tolerance`` may only **tighten** the pinned :data:`REPRO_TOLERANCE`, never loosen it: it
    must be a finite number in ``[0, REPRO_TOLERANCE]``. A larger value (or ``inf``/negative/
    non-number) raises :class:`ValueError` — the ADR-0002 A7 τ is not weakenable on a fail
    (§8.5/§10.3), so a caller cannot pass a looser tolerance to force a pass.
    """
    if not (
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(tolerance)
        and 0.0 <= float(tolerance) <= REPRO_TOLERANCE
    ):
        raise ValueError(
            f"tolerance must be a finite number in [0, {REPRO_TOLERANCE}] — the ADR-0002 A7 τ "
            f"is not weakenable (§8.5/§10.3); got {tolerance!r}"
        )
    ref_verdict = verdict_of(ref)
    rerun_verdict = verdict_of(rerun)
    verdict_reproduces = ref_verdict is not None and ref_verdict == rerun_verdict

    mismatches = config_mismatches(ref, rerun)
    config_ok = not mismatches

    diffs = metric_abs_diffs(ref, rerun)
    finite = [d for d in diffs.values() if d is not None]
    all_comparable = bool(diffs) and all(d is not None for d in diffs.values())
    # Structural floor: the flattened metric set must be EXACTLY the 14 expected keys and every
    # one finite in both reports — a report that drops a metric (even symmetrically) or gains an
    # extra/renamed one voids the gate. The numeric τ comparison is then scoped to the
    # go/no-go-decision metrics (GATED_METRIC_KEYS); the per-class remainder is reported as a
    # determinism diagnostic, not τ-gated (P1-08 re-sign-off).
    metric_keyset_ok = set(diffs) == set(EXPECTED_METRIC_KEYS)
    expected_present = metric_keyset_ok and all_comparable
    gated_finite = [diffs.get(k) for k in GATED_METRIC_KEYS if diffs.get(k) is not None]
    diagnostic_finite = [diffs.get(k) for k in DIAGNOSTIC_METRIC_KEYS if diffs.get(k) is not None]
    max_abs_diff = max(finite) if finite else None
    gated_max_abs_diff = max(gated_finite) if gated_finite else None
    diagnostic_max_abs_diff = max(diagnostic_finite) if diagnostic_finite else None
    within_tolerance = (
        expected_present
        and gated_max_abs_diff is not None
        and gated_max_abs_diff <= float(tolerance)
    )

    return {
        "tolerance": float(tolerance),
        "ref_verdict": ref_verdict,
        "rerun_verdict": rerun_verdict,
        "verdict_reproduces": bool(verdict_reproduces),
        "config_ok": bool(config_ok),
        "config_mismatches": mismatches,
        "per_metric_abs_diff": diffs,
        "all_metrics_comparable": bool(all_comparable),
        "metric_keyset_ok": bool(metric_keyset_ok),
        "expected_metrics_present": bool(expected_present),
        "max_abs_diff": max_abs_diff,
        "gated_max_abs_diff": gated_max_abs_diff,
        "diagnostic_max_abs_diff": diagnostic_max_abs_diff,
        "within_tolerance": bool(within_tolerance),
        "reproducible": bool(verdict_reproduces and config_ok and within_tolerance),
    }
