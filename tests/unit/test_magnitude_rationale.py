"""P0-28 — gate-default magnitude rationales (blinded-freeze at P0).

Pure-stdlib tests over :mod:`tbox_finder.power`'s magnitude-rationale helpers.
They assert the authored SESOI / power arguments are **internally consistent**
(the +5 pp CI floor equals the per-positive granularity at the pinned min-N; the
point bar is two floors; the ECE budget is half the FDR budget), that **every**
delegated blinded-frozen default carries a rationale with a verified method
citation, and that the pinned constants match the ADR-pinned values byte-for-byte
(a drift between code and ADR-0005/ADR-0004 fails CI). No numpy/pandas — the
rationale helper is pure Python math, so the whole suite runs in the bare env.
"""

from __future__ import annotations

import pytest

from tbox_finder import power

# The full ADR-0005 D18 delegated set (+ ADR-0004 D6's 0.80), by slug.
ALL_KEYS = {
    "recall_point_bar",
    "recall_ci_floor",
    "ece_gate",
    "fdr_gate",
    "decoy_prevalence",
    "swap_ece_margin",
    "swap_recall_margin",
    "swap_auprc_margin",
    "gate4_f1_floor",
}

# Citation-identifier prefixes we allow in a durable artifact (CLAUDE.md §10.1).
_CITE_PREFIXES = ("PMID:", "DOI:", "arXiv:")


# --------------------------------------------------------------------------- #
# Pinned constants match the ADR-pinned values (code<->ADR drift guard)
# --------------------------------------------------------------------------- #


def test_constants_match_adr_pinned_values():
    assert power.RECALL_POINT_BAR_PP == 10  # ADR-0005 D4
    assert power.RECALL_CI_FLOOR_PP == 5  # ADR-0005 D4
    assert power.ECE_GATE == 0.05  # ADR-0005 D11
    assert power.FDR_GATE == 0.10  # ADR-0005 D12
    assert power.DECOY_PREVALENCE == 100  # ADR-0005 D7
    assert power.SWAP_ECE_MARGIN == 0.02  # ADR-0005 D17(c)
    assert power.SWAP_RECALL_MARGIN_PP == 3  # ADR-0005 D17(d)
    assert power.SWAP_AUPRC_MARGIN == 0.03  # ADR-0005 D17(d)
    assert power.GATE4_F1_FLOOR == 0.80  # ADR-0004 D6


# --------------------------------------------------------------------------- #
# Numeric backers
# --------------------------------------------------------------------------- #


def test_binomial_se_known_value():
    # sqrt(0.5*0.5/100) = 0.05
    assert power.binomial_se(0.5, 100) == pytest.approx(0.05)


@pytest.mark.parametrize("bad", [(0.5, 0), (0.5, -1)])
def test_binomial_se_rejects_nonpositive_n(bad):
    with pytest.raises(ValueError):
        power.binomial_se(*bad)


@pytest.mark.parametrize("bad_n", [10.5, 10.0, True])
def test_binomial_se_rejects_non_integer_n(bad_n):
    with pytest.raises(ValueError):
        power.binomial_se(0.5, bad_n)


@pytest.mark.parametrize("p", [-0.01, 1.01])
def test_binomial_se_rejects_out_of_range_p(p):
    with pytest.raises(ValueError):
        power.binomial_se(p, 20)


def test_recall_bar_resolution_consistent_at_pinned_min_n():
    res = power.recall_bar_resolution()  # min_n defaults to MIN_REAL_HOMOLOG_N (20)
    assert res["min_n"] == power.MIN_REAL_HOMOLOG_N == 20
    # 1/20 == 5 pp — the CI floor IS the estimator granularity at min-N.
    assert res["per_positive_granularity_pp"] == pytest.approx(5.0)
    assert res["ci_floor_matches_granularity"] is True
    # +10 pp point bar == two CI floors.
    assert res["point_is_two_ci_floors"] is True


def test_recall_bar_resolution_bite_at_wrong_min_n():
    # If min-N were 25, 1/25 = 4 pp < the pinned 5 pp floor => granularity no longer
    # matches the floor. The guard must flip (proves the assertion bites, §8.7).
    res = power.recall_bar_resolution(min_n=25)
    assert res["per_positive_granularity_pp"] == pytest.approx(4.0)
    assert res["ci_floor_matches_granularity"] is False


@pytest.mark.parametrize("bad", [0, -1, 20.5, 20.0, True])
def test_recall_bar_resolution_rejects_bad_min_n(bad):
    with pytest.raises(ValueError):
        power.recall_bar_resolution(min_n=bad)


@pytest.mark.parametrize("bad_z", [0.0, -1.0, float("inf"), float("nan")])
def test_min_detectable_effect_rejects_bad_z(bad_z):
    with pytest.raises(ValueError):
        power.min_detectable_effect_pp(0.5, 20, z=bad_z)


@pytest.mark.parametrize("bad_baseline", [0.0, 1.0, -0.1, 1.1])
def test_min_detectable_effect_rejects_boundary_baseline(bad_baseline):
    # SE = 0 at the Bernoulli boundaries -> a meaningless zero MDE; reject.
    with pytest.raises(ValueError):
        power.min_detectable_effect_pp(bad_baseline, 20)


def test_min_detectable_effect_is_positive_and_scales_with_n():
    mde_small = power.min_detectable_effect_pp(0.5, 20)
    mde_large = power.min_detectable_effect_pp(0.5, 200)
    assert mde_small > mde_large > 0.0  # more positives -> tighter MDE


def test_expected_false_at_fdr():
    assert power.expected_false_at_fdr(0.10, 1000) == pytest.approx(100.0)
    assert power.expected_false_at_fdr(0.0, 1000) == 0.0


@pytest.mark.parametrize("bad", [(-0.1, 10), (1.1, 10), (0.1, -1), (0.1, 5.5), (0.1, True)])
def test_expected_false_at_fdr_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        power.expected_false_at_fdr(*bad)


def test_calibration_fdr_budget_ratio_is_one_half():
    # ECE 0.05 is exactly half the FDR 0.10 budget.
    assert power.calibration_fdr_budget_ratio() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "ece,fdr",
    [
        (0.05, 0.0),  # fdr must be > 0
        (0.05, -0.1),  # fdr negative
        (0.05, 1.1),  # fdr > 1 (not a rate)
        (-0.01, 0.10),  # ece negative
        (1.5, 0.10),  # ece > 1 (not a rate)
        (float("nan"), 0.10),  # ece nan
        (0.05, float("nan")),  # fdr nan
    ],
)
def test_calibration_fdr_budget_ratio_rejects_out_of_domain(ece, fdr):
    with pytest.raises(ValueError):
        power.calibration_fdr_budget_ratio(ece, fdr)


# --------------------------------------------------------------------------- #
# Rationale registry — completeness, citations, freeze
# --------------------------------------------------------------------------- #


def test_all_delegated_defaults_have_a_rationale():
    assert set(power.all_rationales()) == ALL_KEYS


@pytest.mark.parametrize("key", sorted(ALL_KEYS))
def test_rationale_record_is_well_formed(key):
    rec = power.magnitude_rationale(key)
    # Required fields present and non-empty.
    for field in ("default", "value", "kind", "argument", "citations", "blinded_frozen"):
        assert field in rec
    assert rec["default"] and rec["value"] and rec["argument"]
    # Every default is blinded-frozen at P0 (PRD §2.3).
    assert rec["blinded_frozen"] is True


@pytest.mark.parametrize("key", sorted(ALL_KEYS))
def test_citations_are_machine_traceable(key):
    rec = power.magnitude_rationale(key)
    for cite in rec["citations"]:
        assert cite.startswith(_CITE_PREFIXES), f"{key}: untraceable cite {cite!r}"


def test_sesoi_and_power_defaults_carry_at_least_two_sources():
    # High-stakes SESOI/power conventions ship to reviewers => >=2 agreeing sources
    # (CLAUDE.md §10.1). The two pure-design/reference defaults (decoy prevalence,
    # the GATE-4 reference floor) are project-internal judgments and are exempt.
    exempt = {"decoy_prevalence", "gate4_f1_floor"}
    for key, rec in power.all_rationales().items():
        if key in exempt:
            continue
        assert len(rec["citations"]) >= 2, f"{key}: needs >=2 sources"


def test_every_cited_identifier_is_a_known_verified_citation():
    known = set(power.CITATIONS.values())
    for rec in power.all_rationales().values():
        for cite in rec["citations"]:
            assert cite in known, f"uncatalogued citation {cite!r}"


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        power.magnitude_rationale("not_a_gate_default")


def test_recall_rationales_reference_the_pinned_values():
    # The rendered value strings must echo the pinned constants (no drift).
    assert power.magnitude_rationale("recall_point_bar")["value"] == "+10 pp"
    assert power.magnitude_rationale("recall_ci_floor")["value"] == "+5 pp"
    assert power.magnitude_rationale("ece_gate")["value"] == "<= 0.05"
    assert power.magnitude_rationale("fdr_gate")["value"] == "<= 0.1"
    assert power.magnitude_rationale("decoy_prevalence")["value"] == "100:1"
    assert power.magnitude_rationale("gate4_f1_floor")["value"] == ">= 0.8"


def test_recall_prose_is_derived_from_min_n_not_hardcoded():
    # The recall arguments must interpolate MIN_REAL_HOMOLOG_N (=20) and its
    # 1/N granularity, so a future min-N change can't leave stale prose (the
    # round-3 drift guard). At min-N=20: "1/20", "1-in-20", granularity "5 pp".
    ci = power.magnitude_rationale("recall_ci_floor")["argument"]
    pt = power.magnitude_rationale("recall_point_bar")["argument"]
    assert f"1/{power.MIN_REAL_HOMOLOG_N}" in ci  # 1/20
    assert f"1-in-{power.MIN_REAL_HOMOLOG_N}" in ci  # 1-in-20
    assert f"1/{power.MIN_REAL_HOMOLOG_N}" in pt  # 2 x 1/20
    assert "5 pp per held-out positive" in ci  # granularity 100/20


@pytest.mark.parametrize(
    "call",
    [
        lambda: power.binomial_se(True, 20),
        lambda: power.expected_false_at_fdr(True, 20),
        lambda: power.calibration_fdr_budget_ratio(ece=True),
        lambda: power.calibration_fdr_budget_ratio(0.05, True),
        lambda: power.min_detectable_effect_pp(True, 20),
        lambda: power.min_detectable_effect_pp(0.5, 20, z=True),
    ],
)
def test_rate_params_reject_booleans(call):
    # bool is a subclass of int; a probability-like rate is never a boolean.
    with pytest.raises(ValueError):
        call()
