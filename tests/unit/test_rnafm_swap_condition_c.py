"""P3-18 — the RiNALMo → RNA-FM swap condition (c) (ADR-0005 D17(c); PRD §10.2).

Three layers. (1) The decision logic on hand-computable inputs: strict inequalities, the
unanimity rule over the readings of "sustained", and an order-PAIRED CI (a constant per-order
difference over wildly varying per-order ECEs must give a zero-width interval — resampling the
two arms independently cannot). (2) The pairing refusals, against copies of the real sidecars.
(3) The committed report: it validates, and forgeries that keep it internally consistent are
caught by the re-run against the GATE-2 reports it names.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from tbox_finder import power as PW
from tbox_finder.calib import swap_check as SC
from tbox_finder.models import rna_backbone_registry as BR

_REPO = Path(__file__).resolve().parents[2]
_SHIPPED = _REPO / SC.DEFAULT_SHIPPED_REPORT
_COMPARATOR = _REPO / SC.DEFAULT_COMPARATOR_REPORT
_COMMITTED = _REPO / SC.DEFAULT_REPORT


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped() -> dict:
    return _load(_SHIPPED)


@pytest.fixture(scope="module")
def comparator() -> dict:
    return _load(_COMPARATOR)


@pytest.fixture(scope="module")
def committed() -> dict:
    return _load(_COMMITTED)


# --------------------------------------------------------------------------- #
# (1) the decision logic
# --------------------------------------------------------------------------- #
def test_margin_is_the_adr_pinned_value():
    assert PW.SWAP_ECE_MARGIN == 0.02


@pytest.mark.parametrize(
    ("point", "lower", "expected"),
    [
        # exactly at the margin: D17(c) says "> 0.02", so neither fires
        (
            0.02,
            0.02,
            {"ci_lower_above_margin": False, "point_above_margin_and_ci_lower_above_zero": False},
        ),
        (
            0.03,
            0.021,
            {"ci_lower_above_margin": True, "point_above_margin_and_ci_lower_above_zero": True},
        ),
        (
            0.03,
            0.01,
            {"ci_lower_above_margin": False, "point_above_margin_and_ci_lower_above_zero": True},
        ),
        # a CI touching zero does not exclude it
        (
            0.03,
            0.0,
            {"ci_lower_above_margin": False, "point_above_margin_and_ci_lower_above_zero": False},
        ),
        (
            0.03,
            -0.01,
            {"ci_lower_above_margin": False, "point_above_margin_and_ci_lower_above_zero": False},
        ),
        (
            0.019,
            0.001,
            {"ci_lower_above_margin": False, "point_above_margin_and_ci_lower_above_zero": False},
        ),
    ],
)
def test_readings_use_strict_inequalities(point, lower, expected):
    assert SC.readings_for(point, lower, 0.02) == expected


def test_verdict_requires_every_reading_to_agree():
    assert SC.verdict_from({"a": True, "b": True}) == "fired"
    assert SC.verdict_from({"a": False, "b": False}) == "not_fired"
    assert SC.verdict_from({"a": True, "b": False}) == "unadjudicated"
    assert SC.verdict_from({"a": False, "b": True}) == "unadjudicated"
    assert SC.verdict_from({}) == "unadjudicated"


def test_readings_are_the_documented_set():
    assert sorted(SC.readings_for(0.1, 0.05, 0.02)) == sorted(SC.READINGS)


def _units(n: int) -> list[str]:
    return [f"order_{i:02d}" for i in range(n)]


def test_constant_difference_gives_a_degenerate_interval_at_that_difference():
    # dyadic values so every replicate mean is exact: 0.3125 - 0.25 = 0.0625
    names = _units(30)
    out = SC.paired_difference({n: 0.3125 for n in names}, {n: 0.25 for n in names})
    assert out["ci"]["point"] == out["ci"]["lower"] == out["ci"]["upper"] == 0.0625
    assert out["ci"]["n_blocks"] == 30


def test_ci_is_paired_on_orders_not_two_marginals():
    """The per-order ECEs spread over [0, 0.75] but the difference is a constant 0.0625.

    One draw of orders applied to both arms gives a zero-width interval; resampling the two arms
    independently — the pair of marginal CIs — would give one ~0.3 wide.
    """
    names = _units(30)
    base = {n: (i % 13) * 0.0625 for i, n in enumerate(names)}
    out = SC.paired_difference({n: v + 0.0625 for n, v in base.items()}, base)
    assert out["ci"]["lower"] == out["ci"]["upper"] == 0.0625
    assert out["per_unit"] == {n: 0.0625 for n in names}


def test_a_split_reading_is_unadjudicated_not_picked():
    # Δ alternates 1/256 and 3/64: point 0.025390625 > 0.02, every replicate > 0, lower < 0.02
    names = _units(30)
    comp = {n: 0.25 for n in names}
    ship = {n: 0.25 + (0.00390625 if i % 2 else 0.046875) for i, n in enumerate(names)}
    ci = SC.paired_difference(ship, comp)["ci"]
    assert ci["point"] == pytest.approx(0.025390625)
    assert 0.0 < ci["lower"] < 0.02
    readings = SC.readings_for(ci["point"], ci["lower"], 0.02)
    assert readings == {
        "ci_lower_above_margin": False,
        "point_above_margin_and_ci_lower_above_zero": True,
    }
    assert SC.verdict_from(readings) == "unadjudicated"


def test_paired_difference_refuses_different_unit_sets():
    with pytest.raises(SC.PairingError, match="admissible held-out orders differ"):
        SC.paired_difference({"a": 0.1, "b": 0.2}, {"a": 0.1, "c": 0.2})
    with pytest.raises(SC.PairingError, match="no admissible"):
        SC.paired_difference({}, {})


# --------------------------------------------------------------------------- #
# (2) the real arms, and pairing refusals
# --------------------------------------------------------------------------- #
def test_real_arms_not_fired_and_point_is_the_macro_difference(shipped, comparator):
    out = SC.rnafm_swap_condition_c(shipped, comparator)
    macro_diff = (
        shipped["ood"]["macro_average"]["point"] - comparator["ood"]["macro_average"]["point"]
    )
    assert out["statistic"]["point"] == pytest.approx(macro_diff, abs=1e-12)
    assert out["point_exceeds_margin"] is True  # recorded …
    assert out["resampling"]["ci"]["lower"] < 0.0 < out["statistic"]["point"]
    assert out["verdict"] == "not_fired"  # … and not a reading
    assert out["swap_fired"] is False and out["requires_user_decision"] is False
    assert out["arms"]["shipped"]["backbone"] == BR.PRODUCTION_BACKBONE
    assert out["arms"]["comparator"]["backbone"] == BR.COMPARATOR_BACKBONE
    # no sidecar re-derivation was run in-test, so exactly that clause is FALSE
    assert {k for k, v in out["clauses"].items() if not v} == {
        "inputs_rederive_from_their_sidecars"
    }
    assert out["is_science"] is False


def test_margin_is_read_at_call_time(shipped, comparator, monkeypatch):
    # At -0.01 the CI-floor reading fires (lower ≈ -0.0075) and the zero-excluding one does not.
    monkeypatch.setattr(PW, "SWAP_ECE_MARGIN", -0.01)
    out = SC.rnafm_swap_condition_c(shipped, comparator)
    assert out["margin"]["value"] == -0.01
    assert out["verdict"] == "unadjudicated"
    assert out["clauses"]["margin_is_the_pinned_d17c_value"] is True  # pinned == the live attr


def test_reversed_arms_fail_the_role_clause(shipped, comparator):
    out = SC.rnafm_swap_condition_c(comparator, shipped)
    assert out["clauses"]["arms_are_the_shipped_backbone_and_the_d6_comparator"] is False
    assert out["statistic"]["point"] < 0.0


@pytest.mark.parametrize("which", ["shipped_is_the_comparator", "comparator_is_the_shipped"])
def test_each_role_is_checked_alone(shipped, comparator, which):
    """Reversing both arms breaks BOTH halves of the role conjunction, so it cannot show that
    either half is checked. Feeding one report as both arms breaks exactly one half."""
    pair = (comparator, comparator) if which == "shipped_is_the_comparator" else (shipped, shipped)
    out = SC.rnafm_swap_condition_c(*pair)
    roles = (out["arms"]["shipped"]["backbone"], out["arms"]["comparator"]["backbone"])
    broken = 0 if which == "shipped_is_the_comparator" else 1
    assert roles[broken] != (BR.PRODUCTION_BACKBONE, BR.COMPARATOR_BACKBONE)[broken]
    assert roles[1 - broken] == (BR.PRODUCTION_BACKBONE, BR.COMPARATOR_BACKBONE)[1 - broken]
    assert out["clauses"]["arms_are_the_shipped_backbone_and_the_d6_comparator"] is False


def test_refuses_a_comparator_missing_an_order(shipped, comparator):
    forged = copy.deepcopy(comparator)
    forged["ood"]["units"].pop(sorted(forged["ood"]["units"])[0])
    with pytest.raises(SC.PairingError, match="different held-out orders"):
        SC.rnafm_swap_condition_c(shipped, forged)


def test_refuses_a_census_disagreement(shipped, comparator):
    forged = copy.deepcopy(comparator)
    name = sorted(forged["ood"]["units"])[0]
    forged["ood"]["units"][name]["n_positives"] += 1
    with pytest.raises(SC.PairingError, match="n_positives differs"):
        SC.rnafm_swap_condition_c(shipped, forged)


def test_refuses_a_different_ood_estimator_setting(shipped, comparator):
    forged = copy.deepcopy(comparator)
    forged["ood"]["min_n"] = 10
    with pytest.raises(SC.PairingError, match="ood.min_n differs"):
        SC.rnafm_swap_condition_c(shipped, forged)


def _sidecar_root(tmp_path: Path, shipped: dict, comparator: dict) -> Path:
    for rep in (shipped, comparator):
        for key in ("loo_scores", "in_distribution_scores"):
            rel = rep["scoring"][key]
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_REPO / rel, tmp_path / rel)
    return tmp_path


def test_pairing_reads_the_sidecar_bytes_not_the_census(tmp_path, shipped, comparator):
    """Every census count agrees, but the comparator's LOO rows are re-ordered against labels."""
    root = _sidecar_root(tmp_path, shipped, comparator)
    assert SC.pairing_problems(shipped, comparator, repo_root=root) == []
    path = root / comparator["scoring"]["loo_scores"]
    payload = _load(path)
    payload["row_ids"][0], payload["row_ids"][1] = payload["row_ids"][1], payload["row_ids"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    problems = SC.pairing_problems(shipped, comparator, repo_root=root)
    assert problems == ["the leave-clade-out sidecars disagree on 'row_ids' — not one population"]


def test_pairing_checks_the_in_distribution_calib_population(tmp_path, shipped, comparator):
    root = _sidecar_root(tmp_path, shipped, comparator)
    path = root / comparator["scoring"]["in_distribution_scores"]
    payload = _load(path)
    payload["rungs"][0] = "test" if payload["rungs"][0] != "test" else "calib"
    path.write_text(json.dumps(payload), encoding="utf-8")
    problems = SC.pairing_problems(shipped, comparator, repo_root=root)
    assert problems == ["the in-distribution sidecars disagree on 'rungs' — not one population"]


# --------------------------------------------------------------------------- #
# (3) the committed report
# --------------------------------------------------------------------------- #
def test_committed_report_validates(committed):
    assert SC.validate_report(committed) == []
    assert committed["is_science"] is True
    assert committed["verdict"] == "not_fired"
    assert committed["rederivation"]["select_bandwidth"] is True


def test_validate_never_raises_on_garbage():
    for garbage in ({}, {"arms": 5}, {"resampling": {"ci": "x"}, "clauses": []}):
        problems = SC.validate_report(garbage)
        assert problems and all(isinstance(p, str) for p in problems)


def _fire(report: dict) -> None:
    """Make a forged report CONSISTENTLY fired, so only the re-run can see it."""
    for row in report["readings"].values():
        row["fires"] = True
    report["verdict"] = "fired"
    report["swap_fired"] = True
    report["requires_user_decision"] = True


def test_a_consistent_forged_interval_is_caught_only_by_the_rerun(committed):
    forged = copy.deepcopy(committed)
    forged["resampling"]["ci"]["lower"] = 0.0201
    _fire(forged)
    assert all(SC.derive_clauses(forged).values())  # internally consistent …
    problems = SC.validate_report(forged)
    assert (
        "resampling.ci does not re-derive from the named GATE-2 reports" in problems
    )  # … not true
    assert "verdict does not re-derive from the named GATE-2 reports" in problems


def test_a_forged_verdict_without_its_numbers_breaks_a_clause(committed):
    forged = copy.deepcopy(committed)
    _fire(forged)
    problems = SC.validate_report(forged)
    assert any("verdict_follows_every_reading" in p for p in problems)


def test_a_consistent_forged_order_is_caught(committed):
    forged = copy.deepcopy(committed)
    name = sorted(forged["descriptive"]["per_unit"])[0]
    row = forged["descriptive"]["per_unit"][name]
    row["shipped"] += 0.5
    row["delta"] += 0.5
    assert SC.derive_clauses(forged)["point_is_the_difference_of_the_published_macros"] is False
    assert "descriptive.per_unit does not re-derive from the named GATE-2 reports" in (
        SC.validate_report(forged)
    )


def test_dropping_an_a15_confound_is_refused(committed):
    for marker in SC.A15_CONFOUND_MARKERS:
        forged = copy.deepcopy(committed)
        forged["disclosures"] = [d for d in forged["disclosures"] if marker not in d]
        assert SC.derive_clauses(forged)["a15_confounds_disclosed"] is False


def test_a_moved_margin_is_refused(committed):
    forged = copy.deepcopy(committed)
    forged["margin"]["value"] = 0.03
    assert SC.derive_clauses(forged)["margin_is_the_pinned_d17c_value"] is False


def test_a_cheap_bootstrap_is_refused(committed):
    forged = copy.deepcopy(committed)
    forged["resampling"]["n_boot_requested"] = 200
    forged["resampling"]["ci"]["n_boot"] = 200
    assert SC.derive_clauses(forged)["bootstrap_replicates_are_not_a_cost_knob"] is False


def test_recorded_bandwidths_do_not_certify(committed):
    forged = copy.deepcopy(committed)
    forged["rederivation"]["select_bandwidth"] = False
    assert SC.derive_clauses(forged)["inputs_rederive_from_their_sidecars"] is False


def test_a_rebound_input_is_refused(committed):
    forged = copy.deepcopy(committed)
    forged["provenance"]["inputs"][SC.DEFAULT_COMPARATOR_REPORT] = "0" * 64
    problems = SC.validate_report(forged)
    assert any("does not hash to provenance.inputs" in p for p in problems)
