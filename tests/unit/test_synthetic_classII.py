"""P2-08 — the construction-powered synthetic-class-II recovery set.

Bare-CI tier: pure stdlib + pytest, no numpy/pandas/pyarrow. The parquet-touching
paths are exercised separately by ``tests/ml/test_no_leakage.py`` against the real
committed table.

Three failure modes this file exists to guard, each of which has bitten this repo
before:

1. **A gate clause that measures the wrong quantity.** D9 resamples at the block
   level, so a recovery set of 500 variants drawn from 3 parents must FAIL even
   though 500 >= min-N. ``test_the_gate_grades_blocks_not_variant_count``.
2. **A clause that reads vacuously TRUE on absent evidence** (P2-05: a green gate
   with zero measured points). Every emptiness branch is asserted on the *gate*,
   not merely on the field. ``test_an_empty_set_fails_every_gate_clause``.
3. **A fabricated-TRUE clause.** ``all(clauses)`` catches a clause flipped FALSE
   but never one flipped TRUE, so ``validate_report`` re-derives each clause from
   the report's own counts. ``test_validate_report_catches_a_fabricated_*``.
"""

from __future__ import annotations

import pytest

from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.splits import SYNTHETIC_CLASSII_SOURCE
from tbox_finder.synth import _common, classII

# --------------------------------------------------------------------------- #
# Fixtures — real committed-table column names, hand-laid values.
# --------------------------------------------------------------------------- #


def _parent(rid: str, **over):
    row = {
        "record_id": rid,
        "source": "corpus",
        "klass": "II",
        "cluster_id": abs(hash(rid)) % 10_000,
        "resolved_phylum": "Actinobacteria",
        "resolved_order": "Corynebacteriales",
        "nested_train": False,
        "nested_role": "heldout",
    }
    row.update(over)
    return row


def _parents(n: int, *, distinct_blocks: bool = True, **over):
    out = []
    for i in range(n):
        row = _parent(f"rec{i:04d}", **over)
        if distinct_blocks:
            row["cluster_id"] = 1000 + i
            row["resolved_order"] = f"Order{i:04d}"
        else:
            row["cluster_id"] = 7
            row["resolved_order"] = "OneOrder"
        out.append(row)
    return out


def _set(n_parents: int, *, per_parent: int = 2, seed: int = 11, **kw):
    return classII.build_recovery_set(
        _parents(n_parents, **kw), seed=seed, variants_per_parent=per_parent
    )


# --------------------------------------------------------------------------- #
# Pinned values come from the ADR, not from a local literal
# --------------------------------------------------------------------------- #


def test_min_n_is_the_adr_pin_not_a_local_literal():
    """A local copy of the floor drifts silently from the ADR it enforces."""
    assert classII.RECOVERY_MIN_N is MIN_REAL_HOMOLOG_N
    assert classII.RECOVERY_MIN_N == 20


def test_source_label_is_the_splits_constant():
    assert classII.SYNTHETIC_CLASSII_SOURCE is SYNTHETIC_CLASSII_SOURCE


def test_both_block_units_are_graded():
    """Grading only the flattering unit is how an underpowered arm reads green."""
    assert set(classII.BLOCK_UNITS) == {"cluster_id", "resolved_order"}


# --------------------------------------------------------------------------- #
# Eligibility — every conjunct load-bearing, reached in its own name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("override", "why"),
    [
        ({"klass": "I"}, "class-I parent would make the arm not about translational T-boxes"),
        (
            {"source": "blind"},
            "the 18-record blind set is the NATURAL arm — parenting on it "
            "would make synthetic and natural evidence non-independent",
        ),
        ({"source": "anchor"}, "arm-(c) literature anchor is likewise not a corpus parent"),
        ({"nested_role": "train"}, "a training-fold parent is what ADR-0004 A3 still forbids"),
        ({"nested_train": True}, "nested_train parent likewise"),
        ({"nested_role": "dropped"}, "dropped rows are neither train nor held out"),
    ],
)
def test_each_eligibility_conjunct_rejects(override, why):
    assert classII.is_eligible_parent(_parent("r", **override)) is False, why


def test_the_admissible_parent_is_accepted():
    """Without this the parametrized rejections above could pass vacuously."""
    assert classII.is_eligible_parent(_parent("r")) is True


def test_heldout_role_and_nested_train_are_checked_as_a_conjunction():
    """``not nested_train`` does NOT mean "held out from training" (P2-06a).

    A row with ``nested_train=False`` and ``nested_role="dropped"`` is in neither
    fold; inferring heldout-ness from the boolean's negation is exactly the error
    that produced an 88%-contaminated selection fold at P2-06a.
    """
    assert (
        classII.is_eligible_parent(_parent("r", nested_train=False, nested_role="dropped")) is False
    )


def test_a_parent_missing_a_required_column_raises_rather_than_defaulting():
    row = _parent("r")
    del row["nested_role"]
    with pytest.raises(classII.RecoverySetError, match="missing column"):
        classII.is_eligible_parent(row)


def test_build_recovery_set_filters_rather_than_trusting_its_caller():
    """A caller that forgets to filter must not be able to smuggle a bad parent in."""
    mixed = _parents(4) + [_parent("bad", nested_train=True, nested_role="train")]
    variants = classII.build_recovery_set(mixed, seed=3, variants_per_parent=1)
    assert "bad" not in {v.parent_record_id for v in variants}
    assert len(variants) == 4


def test_an_all_ineligible_input_refuses_rather_than_emitting_an_empty_set():
    with pytest.raises(classII.RecoverySetError, match="no eligible class-II parents"):
        classII.build_recovery_set(_parents(3, klass="I"), seed=1)


# --------------------------------------------------------------------------- #
# Determinism + uniqueness by construction
# --------------------------------------------------------------------------- #


def test_generation_is_deterministic_under_a_fixed_seed():
    a = [v.variant_id for v in _set(6, seed=99)]
    b = [v.variant_id for v in _set(6, seed=99)]
    assert a == b


def test_different_seeds_reorder_the_emission():
    seeds = {tuple(v.variant_id for v in _set(12, seed=s)) for s in range(8)}
    assert len(seeds) > 1


def test_emission_order_is_independent_of_input_order():
    forward = _parents(8)
    variants_a = classII.build_recovery_set(forward, seed=5, variants_per_parent=1)
    variants_b = classII.build_recovery_set(list(reversed(forward)), seed=5, variants_per_parent=1)
    assert [v.variant_id for v in variants_a] == [v.variant_id for v in variants_b]


def test_variant_ids_are_unique_by_construction_not_by_rejection():
    """Distinct transforms per parent, so ids differ without relying on the phase draw.

    The first draft drew transform and offset independently per variant; with 3
    transforms x 64 offsets that collides for roughly 1 parent in 192, and it did
    collide on the real 1,172-parent pool.
    """
    for seed in range(25):
        ids = [v.variant_id for v in _set(60, per_parent=3, seed=seed)]
        assert len(set(ids)) == len(ids), f"collision at seed {seed}"


def test_more_variants_than_transforms_is_refused():
    with pytest.raises(classII.RecoverySetError, match="exceeds the .* distinct transforms"):
        classII.build_recovery_set(
            _parents(3), seed=1, variants_per_parent=len(classII.TRANSFORMS) + 1
        )


def test_every_parent_contributes_the_requested_number_of_variants():
    variants = _set(9, per_parent=2)
    counts = {}
    for v in variants:
        counts[v.parent_record_id] = counts.get(v.parent_record_id, 0) + 1
    assert set(counts.values()) == {2}


def test_reverse_complement_flag_matches_the_transform():
    for v in _set(20, per_parent=3):
        assert v.reverse_complement is (v.transform in ("rc", "phase_rc"))


def test_variant_id_rejects_an_unknown_transform():
    with pytest.raises(classII.RecoverySetError, match="unknown transform"):
        classII.variant_id("r", "not_a_transform", 0)


# --------------------------------------------------------------------------- #
# THE central invariant: blocks are the evidence, variants are not
# --------------------------------------------------------------------------- #


def test_a_parent_without_an_order_contributes_no_order_block():
    """``None`` is not a block — counting it inflates the graded quantity.

    Measured on the real pool: exactly one eligible parent lacks
    ``resolved_order``, and the naive count reported 26 orders where 25 exist.
    """
    variants = _set(4, per_parent=1)
    import dataclasses

    variants = [
        dataclasses.replace(v, parent_resolved_order=None) if i < 2 else v
        for i, v in enumerate(variants)
    ]
    assert classII.block_counts(variants)["resolved_order"] == 2


def test_a_nan_order_also_contributes_no_order_block():
    """pandas hands back NaN, not None, for a missing string column."""
    import dataclasses

    variants = [
        dataclasses.replace(v, parent_resolved_order=float("nan")) for v in _set(3, per_parent=1)
    ]
    assert classII.block_counts(variants)["resolved_order"] == 0


def test_stable_key_rejects_the_delimiter_rather_than_colliding():
    """``('a|b',)`` and ``('a','b')`` would otherwise key identically."""
    with pytest.raises(ValueError, match="must not contain the '|' delimiter"):
        _common.stable_key(1, "a|b")


def test_repo_relative_keeps_the_directory_for_a_path_outside_cwd(tmp_path):
    """A path outside cwd's repo used to degrade to its bare basename."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "data").mkdir()
    target = repo / "data" / "x.parquet"
    target.write_text("")
    assert _common.repo_relative(target) == "data/x.parquet"


def test_block_counts_count_parents_not_variants():
    """Two variants of one parent span one block, not two."""
    variants = _set(5, per_parent=2, distinct_blocks=True)
    assert len(variants) == 10
    assert classII.block_counts(variants) == {"cluster_id": 5, "resolved_order": 5}


def test_the_gate_grades_blocks_not_variant_count():
    """Many variants inside ONE block must FAIL — this is the whole point of the arm.

    D9 resamples at the homology-cluster / held-out-order level (PRD §2.3), so
    variants of a shared parent add records without adding evidence.

    The variant count must **exceed min-N** for this to discriminate. An earlier
    draft emitted 9 variants, which a gate wrongly keyed on ``n_variants >= 20``
    would *also* have failed — so the test could not tell the two implementations
    apart, and the sabotage run that appeared to confirm it was in fact being
    caught by ``test_the_gate_grades_at_the_weakest_block_unit``. Caught by
    CodeRabbit, not by my own sabotage attribution.
    """
    variants = _set(15, per_parent=3, distinct_blocks=False)  # 45 variants, 1 block
    report = classII.build_report(variants, seed=1)
    assert report["n_variants"] > MIN_REAL_HOMOLOG_N, "must exceed min-N to discriminate"
    assert report["block_counts"] == {"cluster_id": 1, "resolved_order": 1}
    assert report["gate"]["blocks_meet_min_n"] is False
    assert report["gate"]["overall_pass"] is False
    assert validate_clean(report)


def test_the_gate_grades_at_the_weakest_block_unit():
    """Powered at the cluster level but not the order level is not powered."""
    parents = _parents(MIN_REAL_HOMOLOG_N + 5)
    for p in parents:
        p["resolved_order"] = "OneOrder"  # many clusters, a single order
    variants = classII.build_recovery_set(parents, seed=2, variants_per_parent=1)
    report = classII.build_report(variants, seed=2)
    assert report["block_counts"]["cluster_id"] >= MIN_REAL_HOMOLOG_N
    assert report["block_counts"]["resolved_order"] == 1
    assert report["block_unit_meets_min_n"] == {"cluster_id": True, "resolved_order": False}
    assert report["gate"]["blocks_meet_min_n"] is False


def test_a_genuinely_powered_set_passes():
    """Without this the FAIL assertions above could all pass for the wrong reason."""
    variants = _set(MIN_REAL_HOMOLOG_N + 2, per_parent=1)
    report = classII.build_report(_with_bins(variants), seed=1)
    assert report["gate"]["blocks_meet_min_n"] is True
    assert report["gate"]["control_measurable"] is True
    assert report["gate"]["overall_pass"] is True


# --------------------------------------------------------------------------- #
# The D9 within-phylum-homology control
# --------------------------------------------------------------------------- #


def _with_bins(variants, bin_label="<50"):
    """Re-emit ``variants`` carrying a measured identity bin."""
    import dataclasses

    return [dataclasses.replace(v, parent_identity_bin=bin_label) for v in variants]


def test_an_unmeasured_identity_is_its_own_bucket_not_folded_into_a_stratum():
    """``None`` must not be silently counted as a bin — that would fabricate a control."""
    variants = _set(4, per_parent=1)
    strata = classII.control_strata(variants)
    assert strata == {"unmeasured": 4}
    assert "<50" not in strata


def test_control_strata_count_parents_not_variants():
    variants = _with_bins(_set(6, per_parent=2))
    assert sum(classII.control_strata(variants).values()) == 6


def test_an_unmeasured_control_fails_the_gate_even_when_blocks_are_powered():
    """Blocks powered + control unmeasured is not a pass: A5 makes the control load-bearing."""
    variants = _set(MIN_REAL_HOMOLOG_N + 2, per_parent=1)  # no identity bins
    report = classII.build_report(variants, seed=1)
    assert report["gate"]["blocks_meet_min_n"] is True
    assert report["gate"]["control_measurable"] is False
    assert report["gate"]["overall_pass"] is False
    assert report["n_parents_with_measured_identity"] == 0


# --------------------------------------------------------------------------- #
# Emptiness guards — a clause over absent evidence must not read TRUE
# --------------------------------------------------------------------------- #


def test_an_empty_set_fails_every_gate_clause():
    """P2-05: a clause derived from the requested config rather than found evidence
    reads vacuously TRUE exactly when the evidence is missing."""
    report = classII.build_report([], seed=1)
    assert report["n_variants"] == 0
    assert report["gate"]["blocks_meet_min_n"] is False
    assert report["gate"]["control_measurable"] is False
    assert report["gate"]["overall_pass"] is False
    assert validate_clean(report)


def test_gate_clauses_are_real_bools_not_truthy_ints():
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1)), seed=1)
    for key, value in report["gate"].items():
        assert isinstance(value, bool), f"gate.{key} is {type(value).__name__}, not bool"


# --------------------------------------------------------------------------- #
# validate_report — total, and it catches a fabricated TRUE
# --------------------------------------------------------------------------- #


def validate_clean(report) -> bool:
    return classII.validate_report(report) == []


def test_a_faithful_report_validates_clean():
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 3, per_parent=2)), seed=7)
    assert classII.validate_report(report) == []


def test_validate_report_catches_a_fabricated_true_block_clause():
    """``all(clauses)`` never catches a clause flipped TRUE — re-derivation does."""
    report = classII.build_report(_set(2, per_parent=1, distinct_blocks=False), seed=1)
    assert report["gate"]["blocks_meet_min_n"] is False
    report["gate"]["blocks_meet_min_n"] = True
    problems = classII.validate_report(report)
    assert any("blocks_meet_min_n" in p for p in problems)


def test_validate_report_catches_a_fabricated_true_control_clause():
    report = classII.build_report(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1), seed=1)
    assert report["gate"]["control_measurable"] is False
    report["gate"]["control_measurable"] = True
    assert any("control_measurable" in p for p in classII.validate_report(report))


def test_validate_report_catches_an_overall_pass_inconsistent_with_its_clauses():
    report = classII.build_report([], seed=1)
    report["gate"]["overall_pass"] = True
    assert any("overall_pass" in p for p in classII.validate_report(report))


def test_validate_report_rejects_a_drifted_min_n():
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1)), seed=1)
    report["min_n"] = 3
    assert any("ADR-0005 A1 pin" in p for p in classII.validate_report(report))


def test_validate_report_rejects_a_bool_where_a_count_belongs():
    """``isinstance(True, int)`` holds — a bool must not pass as a count."""
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1)), seed=1)
    report["n_parents_with_measured_identity"] = True
    assert any("must be an int" in p for p in classII.validate_report(report))


@pytest.mark.parametrize(
    "report",
    [
        None,
        [],
        "bad report",
        42,
        {},
        {"gate": {}},
        {"n_variants": 1, "n_parents": 1, "min_n": 20, "block_counts": {}, "gate": {}},
        {"n_variants": 1, "n_parents": 1, "min_n": 20, "block_counts": None, "gate": None},
        {"n_variants": -1, "n_parents": 0, "min_n": 20, "block_counts": {}, "gate": {}},
        {"n_variants": "3", "n_parents": 1, "min_n": 20, "block_counts": {}, "gate": {}},
    ],
)
def test_validate_report_is_total_on_malformed_input(report):
    """A validator that dies on bad input cannot report that the input was bad."""
    problems = classII.validate_report(report)
    assert isinstance(problems, list) and problems


def test_validate_report_requires_the_evidence_block_not_just_the_gate():
    """Requiring only ``gate`` would let a report assert clauses over no evidence."""
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1)), seed=1)
    del report["block_counts"]
    assert any("block_counts" in p for p in classII.validate_report(report))


def test_validate_report_requires_every_block_unit_present():
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1)), seed=1)
    del report["block_counts"]["resolved_order"]
    assert any("resolved_order" in p for p in classII.validate_report(report))


# --------------------------------------------------------------------------- #
# The report is self-describing — a reader of the JSON alone cannot misread it
# --------------------------------------------------------------------------- #


def test_the_report_discloses_that_variant_count_is_not_the_graded_quantity():
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=2)), seed=1)
    assert "not the graded quantity" in report["power_disclosure"]
    assert "BLOCK count" in report["power_disclosure"]


def test_the_report_discloses_the_independence_limitation():
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1)), seed=1)
    assert "NOT" in report["independence_limitation"]
    assert "A5" in report["independence_limitation"]


def test_the_report_records_that_no_platform_swap_chimera_is_emitted():
    """The §10.1 refusal is carried in the artifact, not only in the module docstring."""
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1)), seed=1)
    assert "platform-swap" in report["construction_disclosure"]
    assert "contested" in report["construction_disclosure"]


def test_the_report_records_stage_1_only_grading():
    report = classII.build_report(_with_bins(_set(MIN_REAL_HOMOLOG_N + 1, per_parent=1)), seed=1)
    assert "Stage-1-only" in report["grading_scope"]
    assert "never" in report["grading_scope"]


def test_the_report_carries_a_repo_relative_split_table_path():
    """An absolute path in a committed report records the author's home directory."""
    report = classII.build_report(
        _with_bins(_set(3, per_parent=1)), seed=1, split_table="/tmp/x.parquet"
    )
    assert not report["interim_split_table"].startswith("/")


# --------------------------------------------------------------------------- #
# The promoted shared helpers
# --------------------------------------------------------------------------- #


def test_stable_key_is_deterministic_and_domain_separated():
    assert _common.stable_key(1, "a") == _common.stable_key(1, "a")
    assert _common.stable_key(1, "a") != _common.stable_key(2, "a")
    assert _common.stable_key(1, "a", "b") != _common.stable_key(1, "b", "a")


def test_stable_key_varies_in_every_argument_position():
    """A hash whose varying term comes last can collapse its outputs (P2-03)."""
    base = _common.stable_key(7, "x", "y", "z")
    assert _common.stable_key(7, "X", "y", "z") != base
    assert _common.stable_key(7, "x", "Y", "z") != base
    assert _common.stable_key(7, "x", "y", "Z") != base


def test_tier2n_delegates_to_the_promoted_helpers_rather_than_forking_them():
    """Promote-don't-duplicate is a correctness rule: two copies means fixing one."""
    from tbox_finder.synth import tier2n

    assert tier2n._stable_key(5, "q") == _common.stable_key(5, "q")
    assert tier2n._repo_relative("x.json") == _common.repo_relative("x.json")


def test_bad_bool_rejects_an_int_masquerading_as_a_bool():
    assert _common.bad_bool(1, True) is True
    assert _common.bad_bool(0, False) is True
    assert _common.bad_bool(True, True) is False
    assert _common.bad_bool(False, False) is False
    assert _common.bad_bool(True, False) is True
