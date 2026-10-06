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
    """A repo-shaped copy of every file a swap report reads: reports, sidecars, env locks."""
    rels = [SC.DEFAULT_SHIPPED_REPORT, SC.DEFAULT_COMPARATOR_REPORT, SC.DEFAULT_REPORT, SC.ENV_LOCK]
    for rep in (shipped, comparator):
        rels += [rep["scoring"]["loo_scores"], rep["scoring"]["in_distribution_scores"]]
        rels.append(rep["env_lock"])
    for rel in rels:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO / rel, tmp_path / rel)
    return tmp_path


def _rewrite(path: Path, edit) -> None:
    payload = _load(path)
    edit(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _first_differing(values: list) -> int:
    return next(i for i, v in enumerate(values) if v != values[0])


_LOO_EDITS = {
    "row_ids": lambda d: d["row_ids"].__setitem__(0, d["row_ids"][1]) or None,
    "labels": lambda d: d["labels"].__setitem__(0, 1 - int(d["labels"][0])),
    "units": lambda d: d["units"].__setitem__(0, d["units"][_first_differing(d["units"])]),
    "blocks": lambda d: d["blocks"].__setitem__(0, d["blocks"][_first_differing(d["blocks"])]),
    "dataset_sha256": lambda d: d.__setitem__("dataset_sha256", "0" * 64),
}
_IN_DIST_EDITS = {
    "row_ids": lambda d: d["row_ids"].__setitem__(0, d["row_ids"][1]),
    "labels": lambda d: d["labels"].__setitem__(0, 1 - int(d["labels"][0])),
    "rungs": lambda d: d["rungs"].__setitem__(0, d["rungs"][_first_differing(d["rungs"])]),
    "dataset_sha256": lambda d: d.__setitem__("dataset_sha256", "0" * 64),
}


@pytest.mark.parametrize("field", sorted(_LOO_EDITS))
def test_every_loo_population_field_is_compared(tmp_path, shipped, comparator, field):
    assert sorted(_LOO_EDITS) == sorted(SC._LOO_POPULATION_FIELDS)
    root = _sidecar_root(tmp_path, shipped, comparator)
    _rewrite(root / comparator["scoring"]["loo_scores"], _LOO_EDITS[field])
    assert SC.pairing_problems(shipped, comparator, repo_root=root) == [
        f"the leave-clade-out sidecars disagree on {field!r} — not one population"
    ]


@pytest.mark.parametrize("field", sorted(_IN_DIST_EDITS))
def test_every_in_distribution_population_field_is_compared(tmp_path, shipped, comparator, field):
    assert sorted(_IN_DIST_EDITS) == sorted(SC._IN_DIST_POPULATION_FIELDS)
    root = _sidecar_root(tmp_path, shipped, comparator)
    _rewrite(root / comparator["scoring"]["in_distribution_scores"], _IN_DIST_EDITS[field])
    assert SC.pairing_problems(shipped, comparator, repo_root=root) == [
        f"the in-distribution sidecars disagree on {field!r} — not one population"
    ]


# Literals, not the module's tuples: parametrizing over `SC._SHARED_UNIT_CENSUS` would shrink
# the test along with the tuple it is meant to guard.
_CENSUS = ("n_records", "n_positives", "n_blocks", "admissible", "phylum")
_OOD_SETTINGS = ("estimator", "unit_key", "block_key", "min_n", "n_boot", "bootstrap_seed")


def test_the_pairing_tuples_are_the_documented_ones():
    assert SC._SHARED_UNIT_CENSUS == _CENSUS
    assert SC._SHARED_OOD_SETTINGS == _OOD_SETTINGS


@pytest.mark.parametrize("key", _CENSUS)
def test_every_census_field_is_compared(shipped, comparator, key):
    forged = copy.deepcopy(comparator)
    unit = forged["ood"]["units"][sorted(forged["ood"]["units"])[0]]
    value = unit[key]
    unit[key] = (
        (not value)
        if isinstance(value, bool)
        else (value + 1 if isinstance(value, int) else "Forged")
    )
    with pytest.raises(SC.PairingError, match=f"\\.{key} differs"):
        SC.rnafm_swap_condition_c(shipped, forged)


@pytest.mark.parametrize("key", _OOD_SETTINGS)
def test_every_shared_ood_setting_is_compared(shipped, comparator, key):
    forged = copy.deepcopy(comparator)
    value = forged["ood"][key]
    forged["ood"][key] = value + 1 if isinstance(value, int) else f"{value}_forged"
    with pytest.raises(SC.PairingError, match=f"ood\\.{key} differs"):
        SC.rnafm_swap_condition_c(shipped, forged)


def test_a_truncated_unit_list_is_refused(shipped, comparator):
    forged = copy.deepcopy(comparator)
    forged["ood"]["truncated_to_n_units"] = 29
    with pytest.raises(SC.PairingError, match="truncated"):
        SC.rnafm_swap_condition_c(shipped, forged)


def test_inadmissible_orders_are_left_out_of_the_difference(shipped, comparator):
    pair = [copy.deepcopy(shipped), copy.deepcopy(comparator)]
    name = sorted(shipped["ood"]["units"])[0]
    for rep in pair:
        rep["ood"]["units"][name]["admissible"] = False
    out = SC.rnafm_swap_condition_c(*pair)
    assert name not in out["descriptive"]["per_unit"]
    assert out["pairing"]["n_units_admissible"] == len(shipped["ood"]["units"]) - 1
    assert out["resampling"]["ci"]["n_blocks"] == len(shipped["ood"]["units"]) - 1


def test_a_gate2_report_failing_its_own_validator_fails_one_clause(shipped, comparator):
    forged = copy.deepcopy(shipped)
    first = sorted(forged["clauses"])[0]
    forged["clauses"][first] = not forged["clauses"][first]
    out = SC.rnafm_swap_condition_c(forged, comparator)
    assert {k for k, v in out["clauses"].items() if not v} == {
        "both_gate2_reports_validate",
        "inputs_rederive_from_their_sidecars",
    }


def test_a_sidecar_not_matching_its_gate2_report_fails_one_clause(tmp_path, shipped, comparator):
    """Logits are per-arm, so pairing holds; only the binding to the GATE-2 report breaks."""
    root = _sidecar_root(tmp_path, shipped, comparator)
    arm = comparator["scoring"]["arm"]
    _rewrite(
        root / comparator["scoring"]["loo_scores"],
        lambda d: d["arms"][arm]["logits"].__setitem__(0, d["arms"][arm]["logits"][0] * 5),
    )
    out = SC.rnafm_swap_condition_c(shipped, comparator, repo_root=root)
    assert {k for k, v in out["clauses"].items() if not v} == {
        "both_gate2_reports_bound_to_their_sidecars",
        "inputs_rederive_from_their_sidecars",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("pairing", "problems"), ["forged"]),
        (("pairing", "n_units_admissible"), 29),
    ],
)
def test_the_population_clause_reads_each_member(committed, path, value):
    forged = copy.deepcopy(committed)
    forged[path[0]][path[1]] = value
    assert SC.derive_clauses(forged)["arms_graded_on_one_population"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("resampling", "block_key"), "cluster_id"),
        (("resampling", "ci", "n_blocks"), 29),
        (("resampling", "ci", "upper"), -1.0),
    ],
)
def test_the_block_clause_reads_each_member(committed, path, value):
    forged = copy.deepcopy(committed)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert SC.derive_clauses(forged)["ci_is_order_block_resampled"] is False


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
    assert "resampling does not re-derive from the named GATE-2 reports" in problems  # … not true
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
    assert "descriptive does not re-derive from the named GATE-2 reports" in (
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


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("resampling", "n_boot_requested"), 200),  # a cheap bootstrap
        (("resampling", "n_boot_requested"), 2001),  # pinned, not floored
        (("resampling", "seed"), 390),  # a shopped seed
        (("resampling", "seed"), float(SC.SEED)),  # same value, not an int
        (("resampling", "ci", "n_boot"), 1999),  # replicates dropped
        (("resampling", "ci", "ci_level"), 0.9),
    ],
)
def test_the_bootstrap_scheme_is_pinned(committed, path, value):
    forged = copy.deepcopy(committed)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert SC.derive_clauses(forged)["bootstrap_is_the_pinned_scheme"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("select_bandwidth", False),  # recorded bandwidths do not certify
        ("ran", False),
        ("problems", {"shipped": [], "comparator": ["forged"]}),
        ("problems", {"shipped": []}),
    ],
)
def test_the_rederivation_clause_reads_each_member(committed, key, value):
    forged = copy.deepcopy(committed)
    forged["rederivation"][key] = value
    assert SC.derive_clauses(forged)["inputs_rederive_from_their_sidecars"] is False


def test_a_forged_rederivation_record_is_caught_when_rerun(
    tmp_path, shipped, comparator, committed
):
    """The record says the sidecars re-derive cleanly; on these bytes they do not.

    Both arms' LOO sidecars are edited and re-bound in the swap report's provenance, so layers
    (1)-(2) are satisfied by construction; each sidecar now fails its own GATE-2 binding, which
    is also what keeps the re-derivation fast (it stops at the binding check).
    """
    root = _sidecar_root(tmp_path, shipped, comparator)
    forged = copy.deepcopy(committed)
    for rep in (shipped, comparator):
        rel, arm = rep["scoring"]["loo_scores"], rep["scoring"]["arm"]
        _rewrite(root / rel, lambda d, arm=arm: d["arms"][arm]["logits"].__setitem__(0, 9.0))
        forged["provenance"]["inputs"][rel] = SC.PROV.sha256_file(root / rel)
    problems = SC.validate_report(forged, repo_root=root, rederive_sidecars=True)
    assert any(p.startswith("rederivation does not reproduce") for p in problems)


def test_a_rebound_input_is_refused(committed):
    forged = copy.deepcopy(committed)
    forged["provenance"]["inputs"][SC.DEFAULT_COMPARATOR_REPORT] = "0" * 64
    problems = SC.validate_report(forged)
    assert any("does not hash to provenance.inputs" in p for p in problems)


def test_a_stale_sidecar_under_an_unchanged_report_is_refused(
    tmp_path, shipped, comparator, committed
):
    """Every RNA-FM LOO logit x5; both GATE-2 reports and the swap report left untouched."""
    root = _sidecar_root(tmp_path, shipped, comparator)
    assert SC.validate_report(committed, repo_root=root) == []  # positive control
    rel = comparator["scoring"]["loo_scores"]
    arm = comparator["scoring"]["arm"]
    _rewrite(
        root / rel,
        lambda d: d["arms"][arm].__setitem__("logits", [v * 5 for v in d["arms"][arm]["logits"]]),
    )
    problems = SC.validate_report(committed, repo_root=root)
    assert (
        f"input {rel!r} does not hash to provenance.inputs — the verdict is not about these bytes"
        in problems
    )


def test_a_rebound_env_lock_is_refused(committed):
    forged = copy.deepcopy(committed)
    forged["provenance"]["env_lock_hash"] = "0" * 64
    assert f"provenance.env_lock_hash is not the sha256 of {SC.ENV_LOCK!r}" in SC.validate_report(
        forged
    )


def test_a_dropped_input_is_refused(committed):
    forged = copy.deepcopy(committed)
    forged["provenance"]["inputs"].pop(sorted(forged["provenance"]["inputs"])[-1])
    assert any(p.startswith("provenance.inputs names") for p in SC.validate_report(forged))


@pytest.mark.parametrize(
    "path",
    [
        ("arms", "comparator", "loo_macro_ece", "point"),
        ("descriptive", "n_units_delta_above_margin"),
        ("descriptive", "n_units_comparator_worse"),
        ("descriptive", "by_phylum", "Actinobacteria", "contribution_to_macro_delta"),
        ("descriptive", "marginal_macro_cis", "shipped", "lower"),
        ("pairing", "n_units"),
        ("pairing", "shared_ood_settings", "min_n"),
        ("readings", "ci_lower_above_margin", "definition"),
        ("margin", "source"),
        ("rationale",),
        ("disclosures", 1),
        ("gate2_validation", "shipped"),
    ],
)
def test_every_published_field_is_rerun(committed, path):
    forged = copy.deepcopy(committed)
    target = forged
    for key in path[:-1]:
        target = target[key]
    old = target[path[-1]]
    target[path[-1]] = (
        old + 1
        if isinstance(old, (int, float)) and not isinstance(old, bool)
        else (["forged"] if isinstance(old, list) else f"{old} (forged)")
    )
    assert f"{path[0]} does not re-derive from the named GATE-2 reports" in SC.validate_report(
        forged
    )


def test_negating_an_a15_confound_keeps_the_marker_but_is_caught(committed):
    forged = copy.deepcopy(committed)
    i = next(i for i, d in enumerate(forged["disclosures"]) if "realised LR schedule" in d)
    forged["disclosures"][i] = "The realised LR schedule was IDENTICAL in both arms; no confound."
    assert SC.derive_clauses(forged)["a15_confounds_disclosed"] is True  # the marker survives …
    assert "disclosures does not re-derive from the named GATE-2 reports" in SC.validate_report(
        forged
    )


@pytest.mark.parametrize(
    "path",
    [
        ("resampling", "n_boot_requested"),
        ("resampling", "seed"),
        ("statistic", "point"),
        ("margin", "value"),
        ("resampling", "ci", "lower"),
    ],
)
@pytest.mark.parametrize("value", [float("inf"), 10**400])
def test_validate_never_raises_on_overflow(committed, path, value):
    forged = copy.deepcopy(committed)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    problems = SC.validate_report(forged)
    assert problems and all(isinstance(p, str) for p in problems)


def test_finite_absorbs_an_overflowing_integer():
    # tested alone: validate_report's ArithmeticError catch would otherwise mask its removal
    assert SC._finite(10**400) is None
    assert SC._finite(float("inf")) is None
    assert SC._finite(0.25) == 0.25


def test_validate_reports_an_arithmetic_error_instead_of_raising(committed, monkeypatch):
    # tested alone: `_finite`'s own guard would otherwise mask the catch's removal
    def overflow(_report):
        raise OverflowError("forced")

    monkeypatch.setattr(SC, "derive_clauses", overflow)
    assert SC.validate_report(committed) == [
        "the report could not be validated: OverflowError: forced"
    ]
