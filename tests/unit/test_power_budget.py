"""Unit tests for the P0-26 GATE-1 power-budget audit (`tbox_finder.power`).

The pinned ``MIN_REAL_HOMOLOG_N`` (ADR-0005 D18) and the reported-not-gated /
arm-(c)-sourcing-fallback machinery are load-bearing: they decide which GATE-1
arms are gated versus disclosed-as-underpowered, so each predicate is locked here
with a **clean-pass AND a below-min-N-fail** (the guard is proven to bite —
CLAUDE.md §8.7/§10.3). The identity binning follows the ADR-0005 D1 fixed edges.

Layering matches the repo pattern: the pure predicates (``bin_identity``,
``bin_counts``, ``reachability``, ``arm_verdict``, ``_read_counts``) are
stdlib-only and run in the bare CI env; the ``build_report`` aggregation is
pandas-gated (``importorskip``). No full-corpus recompute here — that runs at the
manual step-local gate (CLAUDE.md §8.5), not in CI.
"""

import json

import pytest

from tbox_finder import power

# --------------------------------------------------------------------------- #
# bin_identity — ADR-0005 D1 fixed edges (left-closed bins) + sentinel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "identity,expected",
    [
        (-1.0, power.NO_NEIGHBOUR),  # no coverage-adequate neighbour sentinel
        (0.0, "<50"),
        (0.4999, "<50"),
        (0.50, "50-70"),  # left-closed: 50 % lands in 50-70
        (0.6995, "50-70"),  # the observed max held-out↔train identity
        (0.70, "70-85"),
        (0.849, "70-85"),
        (0.85, "85-100"),
        (1.0, "85-100"),
    ],
)
def test_bin_identity_edges(identity, expected):
    assert power.bin_identity(identity) == expected


def test_bin_counts_all_bins_present_and_sentinel_separate():
    counts = power.bin_counts([-1.0, -1.0, 0.3, 0.55, 0.60, 0.72])
    assert counts == {"<50": 1, "50-70": 2, "70-85": 1, "85-100": 0, power.NO_NEIGHBOUR: 2}


def test_low_identity_count_splits_measurable_vs_no_neighbour():
    counts = power.bin_counts([0.3, 0.55, 0.60, -1.0, -1.0, -1.0])
    low = power.low_identity_count(counts)
    assert low == {
        "measurable_below_70": 3,
        "no_coverage_adequate_neighbour": 3,
        "total_low_identity_region": 6,
    }


# --------------------------------------------------------------------------- #
# reachability / arm_verdict — the min-N guard (clean-pass AND fail)
# --------------------------------------------------------------------------- #


def test_reachability_at_and_below_min_n():
    assert power.reachability(power.MIN_REAL_HOMOLOG_N) == "powered"  # boundary is inclusive
    assert power.reachability(power.MIN_REAL_HOMOLOG_N - 1) == "reported-not-gated"
    assert power.reachability(0) == "reported-not-gated"


def test_arm_verdict_powered_pass():
    v = power.arm_verdict(1635, n_blocks=34)
    assert v["powered"] is True
    assert v["status"] == "powered"
    assert v["block_resamplable"] is True


def test_arm_verdict_below_min_n_fail():
    # The P0-16 anchor's real N — the guard must flag it reported-not-gated.
    v = power.arm_verdict(16)
    assert v["powered"] is False
    assert v["status"] == "reported-not-gated"


def test_arm_verdict_single_block_not_resamplable():
    v = power.arm_verdict(50, n_blocks=1)
    assert v["powered"] is True  # enough records...
    assert v["block_resamplable"] is False  # ...but a single block can't be block-bootstrapped


def test_min_n_is_the_repo_well_powered_unit():
    # ADR-0005 D18 pin must equal the split's well-powered leave-one-order-out unit
    # (conf/data/splits.yaml min_heldout_positives) — one internally-consistent floor.
    from tbox_finder import splits

    assert power.MIN_REAL_HOMOLOG_N == splits.MIN_HELDOUT_POSITIVES == 20


# --------------------------------------------------------------------------- #
# _read_counts — dotted-key JSON reads; missing key must raise (no silent 0)
# --------------------------------------------------------------------------- #


def test_read_counts_dotted_paths(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"raw_record_count": 16, "leakage_report": {"n_coordinate_novel": 5}}))
    got = power._read_counts(
        p, {"n": "raw_record_count", "novel": "leakage_report.n_coordinate_novel"}
    )
    assert got == {"n": 16, "novel": 5}


def test_read_counts_missing_key_raises_no_fabrication(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"raw_record_count": 16}))
    with pytest.raises(KeyError):
        power._read_counts(p, {"x": "leakage_report.n_coordinate_novel"})


# --------------------------------------------------------------------------- #
# build_report — pandas-gated aggregation, clean-pass AND below-min-N-fail
# --------------------------------------------------------------------------- #
# pandas is gated per-test (fixture) so the pure predicate tiers above still run
# in the bare CI env (a module-level importorskip would skip the whole file).


@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


def _synthetic_split_df(pd):
    """A tiny held-out partition spanning both classes, phyla, and orders.

    30 Firmicutes class-I held-out in order 'Bacillales' (powered);
    25 Actinobacteria class-II held-out in order ' Corynebacteriales' (non-Firmicutes,
    single-phylum); a handful of train rows (ignored by the audit).
    """
    rows = []
    for i in range(30):
        rows.append(
            dict(
                seq_name=f"firm{i}",
                source="corpus",
                klass="I",
                nested_role="heldout",
                resolved_phylum="Firmicutes",
                loo_order_unit="Bacillales",
            )
        )
    for i in range(25):
        rows.append(
            dict(
                seq_name=f"act{i}",
                source="corpus",
                klass="II",
                nested_role="heldout",
                resolved_phylum="Actinobacteria",
                loo_order_unit="Corynebacteriales",
            )
        )
    for i in range(5):
        rows.append(
            dict(
                seq_name=f"train{i}",
                source="corpus",
                klass="I",
                nested_role="train",
                resolved_phylum="Firmicutes",
                loo_order_unit=None,
            )
        )
    return pd.DataFrame(rows)


def _identities(df):
    # All held-out at 0.60 identity (→ 50-70 bin); train rows carry no identity.
    names, roles = df["seq_name"], df["nested_role"]
    return {n: 0.60 for n, r in zip(names, roles, strict=True) if r == "heldout"}


def test_build_report_bins_and_powered_arms(pd):
    df = _synthetic_split_df(pd)
    ident = _identities(df)
    anchor = {
        "raw_record_count": 25,
        "counts_by_gtdb_phylum": {"Chloroflexota": 25},
        "n_coordinate_novel": 25,
        "n_corpus_coord_overlap": 0,
    }
    classii = {"raw_positive_count": 30, "leads_count": 0, "n_added_to_no_leakage_test": 0}
    rep = power.build_report(df, ident, anchor, classii)

    assert rep["min_real_homolog_n"] == 20
    assert rep["n_heldout_total"] == 55
    assert rep["heldout_identity_bins"]["pooled"]["50-70"] == 55
    # Clean-pass: anchor N=25 ≥ 20 → no sourcing-fallback trigger.
    assert rep["headline_arms"]["arm_c_literature_anchor"]["powered"] is True
    assert rep["determinations"]["arm_c_sourcing_fallback_trigger"] is False
    # Non-Firmicutes subgroup = the 25 Actinobacteria class-II → powered.
    assert rep["headline_arms"]["non_firmicutes_order_subgroup"]["n"] == 25
    assert rep["determinations"]["non_firmicutes_order_subgroup"] == "powered"
    # Upper bins are empty (all held-out at 0.60).
    assert rep["headline_arms"]["upper_identity_bins_70_100"]["n"] == 0


def test_build_report_below_min_n_triggers_fallback(pd):
    """The real-data case: anchor N=16 and independent class-II N=0 → guards bite."""
    df = _synthetic_split_df(pd)
    ident = _identities(df)
    anchor = {
        "raw_record_count": 16,
        "counts_by_gtdb_phylum": {"Chloroflexota": 9},
        "n_coordinate_novel": 5,
        "n_corpus_coord_overlap": 11,
    }
    classii = {"raw_positive_count": 0, "leads_count": 40, "n_added_to_no_leakage_test": 0}
    rep = power.build_report(df, ident, anchor, classii)

    assert rep["headline_arms"]["arm_c_literature_anchor"]["powered"] is False
    assert rep["determinations"]["arm_c_literature_anchor_below_min_n"] is True
    assert rep["determinations"]["arm_c_sourcing_fallback_trigger"] is True
    # No blind non-Actino + P0-17 = 0 → independent class-II below min-N.
    ci = rep["headline_arms"]["classII_anti_mimicry"]["independent_non_actinobacteria"]
    assert ci["n"] == 0 and ci["powered"] is False
    assert rep["determinations"]["classII_independent_natural_below_min_n"] is True


def test_build_report_raises_on_missing_identity(pd):
    df = _synthetic_split_df(pd)
    ident = _identities(df)
    ident.pop("firm0")  # drop one held-out record's identity → join gap
    anchor = {
        "raw_record_count": 16,
        "counts_by_gtdb_phylum": {},
        "n_coordinate_novel": 5,
        "n_corpus_coord_overlap": 11,
    }
    classii = {"raw_positive_count": 0, "leads_count": 40, "n_added_to_no_leakage_test": 0}
    with pytest.raises(ValueError, match="no computed identity"):
        power.build_report(df, ident, anchor, classii)


def test_build_report_counts_blind_non_actino_as_independent(pd):
    """A non-Actinobacteria blind class-II record counts toward independent evidence."""
    df = _synthetic_split_df(pd)
    extra = pd.DataFrame(
        [
            dict(
                seq_name="blind_cfx",
                source="blind",
                klass="II",
                nested_role="heldout",
                resolved_phylum="Chloroflexi",
                loo_order_unit=None,
            ),
        ]
    )
    df = pd.concat([df, extra], ignore_index=True)
    ident = _identities(df)
    ident["blind_cfx"] = 0.55
    anchor = {
        "raw_record_count": 16,
        "counts_by_gtdb_phylum": {},
        "n_coordinate_novel": 5,
        "n_corpus_coord_overlap": 11,
    }
    classii = {"raw_positive_count": 0, "leads_count": 40, "n_added_to_no_leakage_test": 0}
    rep = power.build_report(df, ident, anchor, classii)
    src = rep["headline_arms"]["classII_anti_mimicry"]["independent_sources"]
    assert src["blind_non_actino"] == 1


def test_build_report_unresolved_phylum_not_counted_and_json_safe(pd):
    """An unresolved-phylum blind class-II is NOT independent non-Actino, and the
    normalized phylum keeps the report JSON-serializable with sort_keys (a NaN key
    would break json.dumps(sort_keys=True))."""
    df = _synthetic_split_df(pd)
    extra = pd.DataFrame(
        [
            dict(
                seq_name="blind_unk",
                source="blind",
                klass="II",
                nested_role="heldout",
                resolved_phylum=None,  # unresolved — cannot be asserted non-Actino
                loo_order_unit=None,
            ),
        ]
    )
    df = pd.concat([df, extra], ignore_index=True)
    ident = _identities(df)
    ident["blind_unk"] = 0.55
    anchor = {
        "raw_record_count": 16,
        "counts_by_gtdb_phylum": {},
        "n_coordinate_novel": 5,
        "n_corpus_coord_overlap": 11,
    }
    classii = {"raw_positive_count": 0, "leads_count": 40, "n_added_to_no_leakage_test": 0}
    rep = power.build_report(df, ident, anchor, classii)
    src = rep["headline_arms"]["classII_anti_mimicry"]["independent_sources"]
    assert src["blind_non_actino"] == 0  # unresolved is conservatively excluded
    by_phylum = rep["headline_arms"]["classII_anti_mimicry"]["heldout_classII_by_phylum"]
    assert "(unresolved)" in by_phylum
    json.dumps(rep, sort_keys=True)  # must not raise on a normalized key set
