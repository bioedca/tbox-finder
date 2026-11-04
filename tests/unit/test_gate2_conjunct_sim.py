"""ADR-0005 A18 Pin 2 — the committed error-rate record of GATE-2's plug-in conjunct.

``reports/p3/gate2_ece_conjunct_sim.json`` is the measurement A18 Pin 2 was signed on. Two
things are checked, and they are different things ([[artifact-pinning-test-cannot-see-the-code]]):

1. The committed record **re-derives** from the script and the SHIPPED estimator and
   predicate, so a change to ``metrics.binned_ece`` or ``metrics.gate2_ece_pass`` that moves
   a quoted rate fails here rather than leaving the ADR quoting a stale number.
2. The **claims A18 makes** about that record hold on it: no cost to a calibrated arm at the
   gated rung, the false passes it removes there, and the cost it discloses at small n.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from tbox_finder import metrics as M

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "measure_gate2_ece_conjunct.py"
_REPORT = _REPO / "reports" / "p3" / "gate2_ece_conjunct_sim.json"
GATE_RUNG_N = 3045  # P3-10's test rung (reports/gate2_p3_ece.json → gate.n)


def _script():
    spec = importlib.util.spec_from_file_location("measure_gate2_ece_conjunct", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _committed() -> dict:
    return json.loads(_REPORT.read_text(encoding="utf-8"))


def _cell(report: dict, n: int, scale: float) -> dict:
    (hit,) = [c for c in report["cells"] if c["n"] == n and math.isclose(c["scale"], scale)]
    return hit


def test_the_committed_record_rederives_from_the_shipped_estimator() -> None:
    committed = _committed()
    fresh = _script().measure()
    assert committed["design"] == fresh["design"]
    assert len(committed["cells"]) == len(fresh["cells"]) == 12
    for old, new in zip(committed["cells"], fresh["cells"], strict=True):
        assert (old["n"], old["scale"], old["reps"]) == (new["n"], new["scale"], new["reps"])
        # Means through a float sum: Python 3.12 compensates `sum()` and 3.11 does not
        # ([[py312-sum-breaks-bit-exact-rederivation]]), so these get a tolerance...
        for key in ("true_ece", "mean_debiased_ece", "mean_plugin_ece"):
            assert old[key] == pytest.approx(new[key], abs=1e-9), (old["n"], old["scale"], key)
        # ...and the pass RATES are counts over 200 replicates: they must match exactly.
        for key in ("p_pass_debiased_only", "p_pass_plugin_only", "p_pass_a18_conjunct"):
            assert old[key] == new[key], (old["n"], old["scale"], key)


def test_the_simulation_grades_through_the_shipped_conjunct_not_a_copy() -> None:
    """The conjunct rate is read off ``metrics.gate2_ece_pass`` itself. Since the plug-in is
    never below the debiased value, it must equal the plug-in-only rate in every cell; a
    script that re-implemented the predicate (or a predicate that dropped the plug-in) would
    break the identity in the cells where the two readings disagree."""
    report = _committed()
    disagree = 0
    for c in report["cells"]:
        assert c["p_pass_a18_conjunct"] == c["p_pass_plugin_only"], c
        assert c["p_pass_a18_conjunct"] <= c["p_pass_debiased_only"], c
        assert c["mean_plugin_ece"] >= c["mean_debiased_ece"], c
        disagree += c["p_pass_a18_conjunct"] != c["p_pass_debiased_only"]
    assert disagree >= 3, "too few discriminating cells to show the conjunct does anything"
    assert report["design"]["gate"] == M.ECE_GATE
    assert report["design"]["n_bins"] == M.ECE_N_BINS


def test_the_claims_a18_pin_2_makes_hold_on_the_committed_record() -> None:
    report = _committed()
    calibrated = _cell(report, GATE_RUNG_N, 1.0)
    near = _cell(report, GATE_RUNG_N, 1.4)
    over = _cell(report, GATE_RUNG_N, 1.6)
    small = _cell(report, 420, 1.0)
    # 1. No cost to a calibrated arm at the rung GATE-2 is graded on.
    assert calibrated["true_ece"] == pytest.approx(0.0, abs=1e-12)
    assert calibrated["p_pass_a18_conjunct"] == 1.0
    # 2. The false passes it removes there: arms whose TRUE error exceeds the gate.
    assert near["true_ece"] > M.ECE_GATE and over["true_ece"] > M.ECE_GATE
    assert near["p_pass_debiased_only"] > near["p_pass_a18_conjunct"]
    assert over["p_pass_debiased_only"] > 0.0 == over["p_pass_a18_conjunct"]
    # 3. The cost A18 discloses: at small n the conjunct fails calibrated arms on noise.
    assert small["p_pass_debiased_only"] == 1.0
    assert small["p_pass_a18_conjunct"] < 0.5
