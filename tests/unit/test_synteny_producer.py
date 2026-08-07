"""The criterion-(c) producer: the status table, the supply derivation, and the round wiring.

The committed diagnostic reports are treated as **evidence**, so their shape and their own
self-grading verdict are pinned here — a report that graded itself unpowered must not be able
to sit in the tree while the supply constant says the backend is live.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from tbox_finder.mining import gff3, mine_round, remine, synteny, synteny_producer
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED, STATUS_UNAVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = synteny_producer.SyntenyRunConfig(
    strand_policy="both", max_intervening_orfs=1, sub_threshold_orf_nt=150
)


def row(candidate_id: str, status: str, **extra) -> dict:
    base = {
        "candidate_id": candidate_id,
        "accession": "GCA_000000001.1:c0",
        "assembly": "GCA_000000001.1",
        "contig_index": 0,
        "seqid": "c0",
        "locus_start": 100,
        "locus_end": 200,
        "status": status,
        "reason": "",
        "per_strand": {},
        "note": "",
    }
    base.update(extra)
    return base


# ═════════════════════════════════════════════════════════════════════════════
# The status table
# ═════════════════════════════════════════════════════════════════════════════
class TestStatusTable:
    def test_status_map_and_rows_are_written_from_one_list(self) -> None:
        table = synteny_producer.build_status_table(
            [row("a", STATUS_PASSED), row("b", STATUS_FAILED)], config=CONFIG
        )
        assert table["status"] == {"a": STATUS_PASSED, "b": STATUS_FAILED}
        assert table["status_counts"] == {STATUS_FAILED: 1, STATUS_PASSED: 1}
        assert table["config"]["strand_policy"] == "both"

    def test_a_duplicate_candidate_id_raises_instead_of_merging(self) -> None:
        """[[duplicate-key-merges-instead-of-colliding]] — a silent overwrite loses a verdict
        while leaving every summed invariant satisfied."""
        with pytest.raises(synteny_producer.ProducerError, match="duplicate"):
            synteny_producer.build_status_table(
                [row("a", STATUS_PASSED), row("a", STATUS_FAILED)], config=CONFIG
            )

    def test_load_refuses_a_table_that_disagrees_with_itself(self, tmp_path: Path) -> None:
        table = synteny_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        table["status"]["a"] = STATUS_FAILED  # the map now contradicts the row
        path = tmp_path / "t.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        with pytest.raises(synteny_producer.ProducerError, match="disagrees"):
            synteny_producer.load_status_map(path)

    def test_positive_control_an_agreeing_table_loads(self, tmp_path: Path) -> None:
        """Without this, a loader that raised on *everything* would satisfy the test above."""
        table = synteny_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        path = tmp_path / "t.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        assert synteny_producer.load_status_map(path) == {"a": STATUS_PASSED}

    def test_load_refuses_an_unknown_status_value(self, tmp_path: Path) -> None:
        table = synteny_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        table["rows"][0]["status"] = "probably"
        table["status"]["a"] = "probably"
        path = tmp_path / "t.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        with pytest.raises(synteny_producer.ProducerError, match="unknown status"):
            synteny_producer.load_status_map(path)

    def test_merge_refuses_shards_that_disagree_on_the_run_config(self, tmp_path: Path) -> None:
        """Two shards run under different carve-out settings do not describe one round."""
        other = synteny_producer.SyntenyRunConfig(
            strand_policy="both", max_intervening_orfs=3, sub_threshold_orf_nt=150
        )
        paths = []
        for name, cfg, cid in (("a", CONFIG, "x"), ("b", other, "y")):
            p = tmp_path / f"{name}.json"
            p.write_text(
                json.dumps(
                    synteny_producer.build_status_table([row(cid, STATUS_PASSED)], config=cfg)
                ),
                encoding="utf-8",
            )
            paths.append(p)
        with pytest.raises(synteny_producer.ProducerError, match="disagree on the run config"):
            synteny_producer.merge_status_tables(paths)

    def test_merge_positive_control_matching_configs_merge(self, tmp_path: Path) -> None:
        paths = []
        for name, cid in (("a", "x"), ("b", "y")):
            p = tmp_path / f"{name}.json"
            p.write_text(
                json.dumps(
                    synteny_producer.build_status_table([row(cid, STATUS_PASSED)], config=CONFIG)
                ),
                encoding="utf-8",
            )
            paths.append(p)
        merged = synteny_producer.merge_status_tables(paths)
        assert set(merged["status"]) == {"x", "y"}


# ═════════════════════════════════════════════════════════════════════════════
# The round wiring — an absent id must SPARE, never mine
# ═════════════════════════════════════════════════════════════════════════════
class TestRoundWiring:
    def test_a_produced_status_reaches_the_disjunct(self) -> None:
        evidence = mine_round.candidate_evidence("x", None, {"x": STATUS_PASSED})
        assert evidence.downstream_aaRS_synteny == STATUS_PASSED

    def test_an_id_absent_from_the_map_is_unavailable_not_failed(self) -> None:
        """A dropped shard must cost sensitivity, never mine a real T-box."""
        evidence = mine_round.candidate_evidence("x", None, {"other": STATUS_PASSED})
        assert evidence.downstream_aaRS_synteny == STATUS_UNAVAILABLE

    def test_no_map_at_all_is_unavailable(self) -> None:
        assert mine_round.candidate_evidence("x", None).downstream_aaRS_synteny == (
            STATUS_UNAVAILABLE
        )

    def test_the_covariation_disjunct_is_unchanged_by_the_new_parameter(self) -> None:
        """Asserted by identity: a builder that wrote the synteny status into BOTH fields
        would still satisfy a test that only checked the synteny one."""
        evidence = mine_round.candidate_evidence("x", {"x": STATUS_FAILED}, {"x": STATUS_PASSED})
        assert evidence.any_helix_rscape == STATUS_FAILED
        assert evidence.downstream_aaRS_synteny == STATUS_PASSED
        assert evidence.relaxed_architecture == STATUS_UNAVAILABLE

    def test_the_p3_builder_threads_it_too(self) -> None:
        evidence = remine.remine_candidate_evidence(
            "x",
            covariation_status=None,
            stage2_posteriors=None,
            synteny_status={"x": STATUS_PASSED},
        )
        assert evidence.downstream_aaRS_synteny == STATUS_PASSED


# ═════════════════════════════════════════════════════════════════════════════
# The supply derivation — the constant has to prove itself, in BOTH directions
# ═════════════════════════════════════════════════════════════════════════════
class TestSupplyDerivation:
    def test_the_constant_agrees_with_the_live_derivation(self) -> None:
        derived = synteny_producer.derive_synteny_supply_available(repo_root=REPO_ROOT)
        assert derived["available"] == mine_round.SYNTENY_SUPPLY_AVAILABLE, derived["reasons"]

    def test_every_clause_is_true_in_this_checkout(self) -> None:
        derived = synteny_producer.derive_synteny_supply_available(repo_root=REPO_ROOT)
        assert derived["reasons"] == []
        assert set(derived["clauses"]) >= {name for name, _ in synteny_producer.SUPPLY_CLAUSES}

    def test_no_clause_reads_dvc_tracked_data(self, tmp_path: Path) -> None:
        """⚠ The trap ``derive_stage2_supply_available`` already hit — and the first version of
        this test could not have caught it.

        A clause over ``data/interim/production_annotations`` is ``False`` in CI and in any
        fresh clone, so the derivation would contradict the constant in exactly the
        environments that must agree.  The proof has to be a root where the DVC data **cannot**
        exist while the git-tracked evidence does; deriving against ``REPO_ROOT`` instead
        passes on any laptop that has run ``dvc pull``, i.e. it asserts nothing about the
        environment it names.
        """
        for _name, source in synteny_producer.SUPPLY_CLAUSES:
            target = tmp_path / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO_ROOT / source).read_bytes())
        assert not (tmp_path / synteny_producer.DEFAULT_ANNOTATION_DIR).exists()
        derived = synteny_producer.derive_synteny_supply_available(repo_root=tmp_path)
        assert derived["n_annotated_hosts_on_disk"] == 0, "the DVC corpus is absent by design"
        assert derived["n_annotated_hosts_in_acquisition_report"] > 0
        assert derived["available"] is True, derived["reasons"]

    @pytest.mark.parametrize("clause", [name for name, _ in synteny_producer.SUPPLY_CLAUSES])
    def test_breaking_ONE_clause_alone_flips_the_verdict(self, clause: str, tmp_path: Path) -> None:
        """[[all-true-fixture-cannot-test-a-conjunction]] — every clause is TRUE here, so a
        hardcoded ``True`` is behaviourally identical to ``all(clauses)`` unless each member
        is broken **on its own**."""
        rel = dict(synteny_producer.SUPPLY_CLAUSES)[clause]
        for name, source in synteny_producer.SUPPLY_CLAUSES:
            target = tmp_path / source
            target.parent.mkdir(parents=True, exist_ok=True)
            if name != clause:
                target.write_bytes((REPO_ROOT / source).read_bytes())
        assert not (tmp_path / rel).exists()
        derived = synteny_producer.derive_synteny_supply_available(repo_root=tmp_path)
        assert derived["available"] is False
        assert any(clause in reason for reason in derived["reasons"])

    def test_an_unpowered_false_pass_report_withdraws_the_supply(self, tmp_path: Path) -> None:
        """A measurement that graded ITSELF uninterpretable is not evidence of a backend."""
        for _name, source in synteny_producer.SUPPLY_CLAUSES:
            target = tmp_path / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO_ROOT / source).read_bytes())
        report = json.loads((tmp_path / synteny_producer.FALSE_PASS_REPORT).read_text())
        report["control"]["powered"] = False
        (tmp_path / synteny_producer.FALSE_PASS_REPORT).write_text(json.dumps(report))
        derived = synteny_producer.derive_synteny_supply_available(repo_root=tmp_path)
        assert derived["available"] is False
        assert any("false_pass_control_powered" in r for r in derived["reasons"])


# ═════════════════════════════════════════════════════════════════════════════
# CLI contracts
# ═════════════════════════════════════════════════════════════════════════════
class TestCli:
    @pytest.mark.parametrize(
        "flag", ["--strand-policy", "--max-intervening-orfs", "--sub-threshold-orf-nt"]
    )
    def test_the_undelegated_values_are_required_with_no_default(self, flag: str) -> None:
        """ADR-0006 pins none of these, so the round supplies them or the run refuses."""
        parser = synteny_producer.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run-shard", "--out", "x.json"])
        action = next(
            a
            for sub in parser._subparsers._group_actions  # type: ignore[union-attr]
            for p in sub.choices.values()
            for a in p._actions
            if flag in a.option_strings
        )
        assert action.required is True
        assert action.default is None or action.default is argparse.SUPPRESS

    def test_the_window_default_is_d4s_pinned_500(self) -> None:
        parser = synteny_producer.build_parser()
        args = parser.parse_args(
            [
                "run-shard",
                "--strand-policy",
                "both",
                "--max-intervening-orfs",
                "1",
                "--sub-threshold-orf-nt",
                "150",
                "--out",
                "x.json",
            ]
        )
        assert args.window_bp == 500

    def test_remine_carries_the_two_way_synteny_declaration(self) -> None:
        """A bare ``store_true`` cannot express a ``True`` default, so a checkout that cannot
        run the backend would have no way to say so (the hole P3-15′-a/-b each closed)."""
        parser = remine.build_parser()
        plan = parser._subparsers._group_actions[0].choices["plan"]  # type: ignore[union-attr]
        options = {opt for action in plan._actions for opt in action.option_strings}
        assert {"--synteny-available", "--no-synteny-available"} <= options
        positive = next(a for a in plan._actions if "--synteny-available" in a.option_strings)
        assert positive.default is mine_round.SYNTENY_SUPPLY_AVAILABLE

    def test_the_unevidenced_preflight_covers_the_synteny_declaration(self) -> None:
        """Declaring a supply this checkout cannot evidence must exit 4, not mine."""
        derivations = {
            "msa_supply_derivation": {"available": True},
            "stage2_supply_derivation": {"available": True},
            "synteny_supply_derivation": {"available": False, "reasons": ["no backend"]},
        }
        args = argparse.Namespace(
            msa_supply_available=True, stage2_supply_available=True, synteny_available=True
        )
        assert remine._refuse_unevidenced(args, derivations) == 4

    def test_positive_control_an_evidenced_declaration_does_not_refuse(self) -> None:
        derivations = {
            "msa_supply_derivation": {"available": True},
            "stage2_supply_derivation": {"available": True},
            "synteny_supply_derivation": {"available": True},
        }
        args = argparse.Namespace(
            msa_supply_available=True, stage2_supply_available=True, synteny_available=True
        )
        assert remine._refuse_unevidenced(args, derivations) is None


# ═════════════════════════════════════════════════════════════════════════════
# The committed diagnostic reports are evidence — pin their shape and verdicts
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def false_pass() -> dict:
    return json.loads((REPO_ROOT / synteny_producer.FALSE_PASS_REPORT).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exclusion() -> dict:
    return json.loads((REPO_ROOT / synteny_producer.EXCLUSION_REPORT).read_text(encoding="utf-8"))


class TestCommittedDiagnostics:
    def test_d4s_named_false_pass_arms_are_all_present(self, false_pass: dict) -> None:
        assert set(false_pass["arms"]) == {
            "clade_matched_random_leaders",
            "nine_one_five_prime_utr_decoys",
            "nine_one_five_prime_utr_decoys_excluding_d4_classes",
            "nine_one_trna_adjacent_decoys",
            "shipped_nine_one_decoy_rows",
        }

    def test_the_utr_arm_is_reported_under_BOTH_readings(self, false_pass: dict) -> None:
        """A 5′UTR upstream of an aaRS may itself be a T-box, so counting it as a *false*
        pass inflates the rate; excluding it stops the arm being a random leader.  Reporting
        one reading only would be a choice disguised as a measurement."""
        wide = false_pass["arms"]["nine_one_five_prime_utr_decoys"]
        strict = false_pass["arms"]["nine_one_five_prime_utr_decoys_excluding_d4_classes"]
        assert strict["n"] > 0 and wide["n"] > 0
        # ⚠ A filtered subset cannot outnumber its source population.  It did — 9,087 vs
        # 9,065 — because the two arms were independent draws each capped at n_per_host,
        # which are not nested.  The strict arm is now a filter of the wide arm's own sample.
        assert strict["n"] <= wide["n"], (strict["n"], wide["n"])
        # Guard on n_decided, not n: _arm reports None (not 0.0) for an arm that drew windows
        # and decided none of them, and `None <= float` raises in Python 3.
        assert strict["n_decided"] and wide["n_decided"], (strict, wide)
        assert strict["false_pass_rate"] <= wide["false_pass_rate"], (strict, wide)
        assert strict["note"]

    def test_every_arm_carries_the_same_shape(self, false_pass: dict) -> None:
        """Including the unavailable one — a reader must not special-case an arm."""
        for name, arm in false_pass["arms"].items():
            if "available" in arm:
                assert {"available", "n_usable_rows", "reason", "detail"} <= set(arm), name
            else:
                assert {"n", "status_counts", "false_pass_rate"} <= set(arm), name

    def test_the_control_is_powered_and_says_why(self, false_pass: dict) -> None:
        control = false_pass["control"]
        assert control["powered"] is True
        assert control["margin"] >= synteny_producer.CONTROL_MIN_MARGIN
        # ⚠ ``verdict`` is the field a reader actually consumes, and the one that would go
        # stale silently; the test named "…and says why" has to assert it or the name lies.
        assert "interpretable" in control["verdict"]
        assert "NOT separate" not in control["verdict"]

    def test_the_report_names_the_unit_of_each_count(self, exclusion: dict) -> None:
        """546 + 96 + 299 = 941 CANDIDATES; the distance block counts 554 STRAND EVALUATIONS.
        Not a discrepancy — a candidate passing on both strands contributes two — but a
        reader subtracting one from the other would conclude something was missing."""
        block = exclusion["passing_distance_sensitivity"]
        assert block["unit"] == "strand_evaluations"
        assert exclusion["passing_class_counts_unit"] == "strand_evaluations"
        assert (
            block["n"]
            == block["n_candidates_passing"] + block["n_candidates_passing_on_both_strands"]
        )
        per_clade_total = sum(b["n"] for b in exclusion["per_clade"].values())
        assert per_clade_total == exclusion["n_candidates"]

    def test_the_positive_context_arm_separates_from_the_background(self, false_pass: dict) -> None:
        """Without separation the false-pass rates measure nothing
        ([[control-matchedness-must-be-asserted]])."""
        background = false_pass["arms"]["clade_matched_random_leaders"]["false_pass_rate"]
        positive = false_pass["positive_context_control"]["pass_rate"]
        assert background is not None and positive is not None
        assert positive - background >= synteny_producer.CONTROL_MIN_MARGIN

    def test_an_unavailable_arm_carries_its_counts_not_just_a_flag(self, false_pass: dict) -> None:
        """[[clauses-must-guard-emptiness]] — an arm that silently vanishes reads as an arm
        that found nothing."""
        shipped = false_pass["arms"]["shipped_nine_one_decoy_rows"]
        assert shipped["available"] is False
        assert shipped["reason"]
        assert shipped["detail"]

    def test_the_joint_abc_arm_is_withheld_rather_than_computed(self, false_pass: dict) -> None:
        assert false_pass["joint_abc"]["available"] is False
        assert "P3-15" in false_pass["joint_abc"]["reason"]

    def test_the_exclusion_report_breaks_down_by_clade_and_reason(self, exclusion: dict) -> None:
        assert exclusion["per_clade"]
        for clade, block in exclusion["per_clade"].items():
            assert block["n"] == sum(block["status_counts"].values()), clade
            assert set(block["reasons"]) <= set(synteny_producer.EXCLUSION_REASONS), clade

    def test_the_pseudogene_diagnostic_is_not_vacuously_zero(self, exclusion: dict) -> None:
        """It reported 0 corpus-wide while the carve-out was consuming the population it sizes."""
        block = exclusion["pseudogene_diagnostic"]
        assert block["hmm_fallback_available"] is False
        assert block["n_unjudgeable_orfs_encountered"] > 0

    def test_the_exclusion_rate_reconciles_with_its_reason_totals(self, exclusion: dict) -> None:
        assert sum(exclusion["exclusion_reason_totals"].values()) == exclusion["n_unavailable"]
        assert exclusion["exclusion_rate"] == pytest.approx(
            exclusion["n_unavailable"] / exclusion["n_candidates"]
        )

    def test_the_distance_statistic_counts_only_passed_evaluations(self, exclusion: dict) -> None:
        """⚠ Collecting by function class alone put 554 entries with a 1,652 bp max into a
        block documented as "measured on the passing candidates" — including out-of-window
        hits that FAILED.  Every *decision* distance must sit inside the pad."""
        block = exclusion["passing_distance_sensitivity"]
        decision = block["decision_distance"]
        assert decision["max_bp"] is not None
        assert decision["max_bp"] <= block["window_bp"], decision
        assert decision["p99_bp"] <= block["window_bp"]
        # The element-relative series may exceed the pad, but ONLY via the tandem carve-out.
        assert block["max_bp"] is not None
        if block["max_bp"] > block["window_bp"]:
            assert block["n_passed_via_tandem_carve_out"] > 0, block

    def test_the_report_names_candidate_clades_it_could_not_sample(self, false_pass: dict) -> None:
        """A clade absent from the breakdown reads as "measured, found nothing"
        ([[clauses-must-guard-emptiness]])."""
        assert "clades_of_candidate_hosts_not_sampled" in false_pass
        assert "clades_with_no_background_windows" in false_pass

    def test_both_reports_describe_the_same_run_config(
        self, false_pass: dict, exclusion: dict
    ) -> None:
        """Two reports about different runs would let a reader draw a false-pass rate and an
        exclusion rate from incompatible corpora."""
        assert false_pass["config"] == exclusion["config"]
        assert false_pass["config"]["window_bp"] == synteny.DEFAULT_WINDOW_BP
        assert false_pass["config"]["hmm_fallback_available"] is False


def _apply_leg_options(module) -> set[str]:
    """Every option string on a module's ``apply-spare-rule`` subparser."""
    sub = module.build_parser()._subparsers._group_actions[0].choices["apply-spare-rule"]
    return {opt for action in sub._actions for opt in action.option_strings}


# ═════════════════════════════════════════════════════════════════════════════
# Review round 1 — the produced table must actually reach the round
# ═════════════════════════════════════════════════════════════════════════════
class TestSyntenyTableIsWired:
    def test_declaring_the_backend_without_a_table_refuses(self) -> None:
        """The silent form of a refusal: every candidate would read ``unavailable``, all would
        be spared, and the round would report a zero yield indistinguishable from an honest
        one — the same hole review found in ``apply_remine_spare_rule``'s ``probe_set``."""
        with pytest.raises(ValueError, match="no synteny status table"):
            mine_round.apply_spare_rule(
                "m.json",
                "s.json",
                rscape_installed=True,
                msa_supply_available=True,
                synteny_available=True,
            )

    def test_supplying_a_table_without_declaring_the_backend_refuses(self) -> None:
        """The opposite direction: `hard_negative.mine_round` raises on produced evidence for
        an undeclared backend, so catching it here names the cause instead."""
        with pytest.raises(ValueError, match="synteny_available=False"):
            mine_round.apply_spare_rule(
                "m.json",
                "s.json",
                rscape_installed=True,
                msa_supply_available=True,
                synteny_status_table="t.json",
                synteny_available=False,
            )

    def test_the_p3_round_carries_the_same_pair_of_guards(self) -> None:
        for kwargs, pattern in (
            ({"synteny_available": True}, "no synteny status table"),
            (
                {"synteny_available": False, "synteny_status_table": "t.json"},
                "synteny_available=False",
            ),
        ):
            with pytest.raises(remine.RemineError, match=pattern):
                remine.apply_remine_spare_rule(
                    "m.json",
                    "s.json",
                    "p.json",
                    stage2_threshold=0.9,
                    rscape_installed=True,
                    msa_supply_available=True,
                    stage2_supply_available=True,
                    probe_set=None,
                    **kwargs,
                )

    @pytest.mark.parametrize("module", [mine_round, remine])
    def test_both_cli_legs_expose_the_status_table(self, module) -> None:
        """A parameter unwired from the CLI the sbatch invokes is a parameter that never runs.

        ⚠ Asserted **structurally**.  The first version probed the P2 leg by parsing
        ``[..., "--synteny-status", "t.json", "--help"]`` and asserting exit 0 — but argparse
        handles ``--help`` before it rejects an unknown option, so that exits 0 even for a
        flag the parser has never heard of (verified by execution on a parser carrying no
        such option).  It passed whether or not the option existed;
        ``mine_round.build_parser`` was extracted so this one cannot.
        """
        options = _apply_leg_options(module)
        assert "--synteny-status" in options

    @pytest.mark.parametrize("module", [mine_round, remine])
    def test_negative_control_the_option_set_is_not_universal(self, module) -> None:
        """The control the vacuous version lacked: membership must be able to answer False."""
        assert "--synteny-status-typo" not in _apply_leg_options(module)


class TestDiagnosticsComputedNotJustCommitted:
    """⚠ The committed-artifact assertions above cannot see a source regression.

    Sabotaging the filter left them GREEN, because they read a report generated by the
    *fixed* code.  These call the report builders directly, so the rule itself is under test
    rather than one frozen output of it.
    """

    def _row(self, row_status: str, **detail) -> dict:
        base = {
            "status": STATUS_PASSED,
            "function_class": synteny.CLASS_AARS,
            "distance_bp": 40,
            "decision_distance_bp": 40,
            "is_pseudo": False,
            "n_pseudo_seen": 0,
            "n_unjudgeable_seen": 0,
            "n_intervening": 0,
            "carve_out_applied": False,
        }
        base.update(detail)
        return row(
            "c1",
            row_status,
            per_strand={
                "+": base,
                "-": dict(
                    base,
                    status=STATUS_FAILED,
                    function_class=None,
                    distance_bp=None,
                    decision_distance_bp=None,
                ),
            },
        )

    def test_a_passing_class_detail_that_FAILED_is_excluded_from_the_distance_series(self) -> None:
        failed_far = self._row(
            STATUS_FAILED,
            status=STATUS_FAILED,  # the strand detail's own status
            function_class=synteny.CLASS_AARS,
            distance_bp=1652,
            decision_distance_bp=1652,
        )
        report = synteny_producer.exclusion_report(
            [failed_far],
            clades={"GCA_000000001.1": "Bacillota"},
            config=CONFIG,
        )
        block = report["passing_distance_sensitivity"]
        assert block["n"] == 0, "an out-of-window aaRS hit that FAILED is not a passing distance"
        assert block["max_bp"] is None

    def test_a_passing_detail_IS_counted(self) -> None:
        """The positive control: a filter that dropped everything would satisfy the test above."""
        report = synteny_producer.exclusion_report(
            [self._row(STATUS_PASSED)],
            clades={"GCA_000000001.1": "Bacillota"},
            config=CONFIG,
        )
        block = report["passing_distance_sensitivity"]
        assert block["n"] == 1 and block["max_bp"] == 40
        assert block["decision_distance"]["max_bp"] == 40

    def test_an_out_of_repo_input_is_hashed_rather_than_named(self, tmp_path: Path) -> None:
        external = tmp_path / "round0" / "synteny_status.json"
        external.parent.mkdir(parents=True)
        external.write_text("{}", encoding="utf-8")
        inside, outside = synteny_producer._split_inputs(
            [external, REPO_ROOT / synteny_producer.FALSE_PASS_REPORT]
        )
        assert inside == [synteny_producer.FALSE_PASS_REPORT]
        assert [e["name"] for e in outside] == ["synteny_status.json"]
        assert len(outside[0]["sha256"]) == 64
        assert not any(str(tmp_path) in str(x) for x in inside + outside)


class TestProducerHardening:
    def test_merge_refuses_a_shard_that_contributes_no_rows(self, tmp_path: Path) -> None:
        """A dropped shard that merges cleanly costs coverage silently."""
        good = tmp_path / "a.json"
        good.write_text(
            json.dumps(
                synteny_producer.build_status_table([row("x", STATUS_PASSED)], config=CONFIG)
            )
        )
        empty = tmp_path / "b.json"
        payload = synteny_producer.build_status_table([row("y", STATUS_PASSED)], config=CONFIG)
        payload["rows"] = []
        empty.write_text(json.dumps(payload))
        with pytest.raises(synteny_producer.ProducerError, match="contributes no rows"):
            synteny_producer.merge_status_tables([good, empty])

    def test_merge_refuses_a_shard_with_an_incomplete_config(self, tmp_path: Path) -> None:
        path = tmp_path / "a.json"
        payload = synteny_producer.build_status_table([row("x", STATUS_PASSED)], config=CONFIG)
        del payload["config"]["window_bp"]
        path.write_text(json.dumps(payload))
        with pytest.raises(synteny_producer.ProducerError, match="missing"):
            synteny_producer.merge_status_tables([path])

    def test_provenance_never_records_an_absolute_local_path(self) -> None:
        """A committed record naming /home/<user>/… leaks a username and resolves for nobody.

        ⚠ The first version of this test was **vacuous on the very report it was written for**:
        both committed reports carry ``"inputs": {}`` (every input resolved outside the repo
        root at generation time), so the loop never executed and ``outputs`` was never looked
        at.  It now asserts the evidence is *somewhere* before checking its shape — the fifth
        vacuous-test finding in this step, and the fifth that was mine.
        """
        for report_path in (synteny_producer.FALSE_PASS_REPORT, synteny_producer.EXCLUSION_REPORT):
            provenance = json.loads((REPO_ROOT / report_path).read_text())["provenance"]
            external = provenance.get("extra", {}).get("external_inputs", [])
            assert provenance["inputs"] or external, (
                f"{report_path}: provenance records no input at all, so a path-shape "
                "assertion over it proves nothing"
            )
            for section in ("inputs", "outputs"):
                for key in provenance[section]:
                    # ⚠ ``Path(key).is_absolute()`` uses PurePosixPath semantics on a POSIX
                    # runner, so a Windows-style ``C:\Users\alice\...`` reads as *relative*
                    # and slips through — and a home-directory leak is the thing being caught.
                    assert not PurePosixPath(key).is_absolute(), (report_path, section, key)
                    assert not PureWindowsPath(key).is_absolute(), (report_path, section, key)
                    assert "/home/" not in key and "\\Users\\" not in key, (
                        report_path,
                        section,
                        key,
                    )
            for entry in external:
                assert "/" not in entry["name"], entry
                assert not Path(entry["name"]).is_absolute(), entry
                assert len(entry["sha256"]) == 64, entry


class TestReviewRoundTwo:
    def test_an_unknown_exclusion_reason_refuses(self) -> None:
        """The totals and the per-clade breakdown read ONE derivation, so the closed
        vocabulary cannot hold in one and be broken in the other."""
        with pytest.raises(synteny_producer.ProducerError, match="unknown exclusion reason"):
            synteny_producer.exclusion_report(
                [row("c1", STATUS_UNAVAILABLE, reason="something-invented")],
                clades={"GCA_000000001.1": "Bacillota"},
                config=CONFIG,
            )

    def test_positive_control_a_known_reason_is_accepted(self) -> None:
        report = synteny_producer.exclusion_report(
            [row("c1", STATUS_UNAVAILABLE, reason=synteny_producer.REASON_PSEUDOGENE)],
            clades={"GCA_000000001.1": "Bacillota"},
            config=CONFIG,
        )
        assert report["exclusion_reason_totals"] == {synteny_producer.REASON_PSEUDOGENE: 1}

    def test_the_totals_and_the_per_clade_reasons_use_the_same_key(self) -> None:
        """They disagreed: an empty reason fell back to "" in one and host_unannotated in the
        other, so the same row was counted under two different keys."""
        report = synteny_producer.exclusion_report(
            [row("c1", STATUS_UNAVAILABLE, reason="")],
            clades={"GCA_000000001.1": "Bacillota"},
            config=CONFIG,
        )
        assert report["exclusion_reason_totals"] == report["per_clade"]["Bacillota"]["reasons"]

    def test_merge_preserves_the_shard_recorded_fallback_flag(self, tmp_path: Path) -> None:
        """``config.as_dict()`` re-reads the LIVE module constant, so a merge on a checkout
        where ``HMM_FALLBACK_AVAILABLE`` had flipped would rewrite what the shards ran under."""
        path = tmp_path / "a.json"
        payload = synteny_producer.build_status_table([row("x", STATUS_PASSED)], config=CONFIG)
        payload["config"]["hmm_fallback_available"] = True  # what this shard actually ran under
        path.write_text(json.dumps(payload))
        merged = synteny_producer.merge_status_tables([path])
        assert merged["config"]["hmm_fallback_available"] is True
        assert synteny.HMM_FALLBACK_AVAILABLE is False, "the live constant differs, on purpose"


class TestStrictSubsampleIsASubset:
    """⚠ The committed-report assertion could not see this either.

    Sabotaging the derivation left it GREEN, because it reads a report generated by the fixed
    code — the third time in this step that an artifact-pinning test failed to bite.  The rule
    is now a named function so the property is testable directly.
    """

    def _cds(self, product: str, start: int) -> gff3.CdsFeature:
        return gff3.CdsFeature(
            seqid="c1",
            feature_id=f"f{start}",
            start=start,
            end=start + 300,
            strand="+",
            segments=((start, start + 300),),
            attributes={"product": (product,)},
        )

    def test_the_result_is_a_subset_of_the_input(self) -> None:
        sample = [
            self._cds("alanine--tRNA ligase", 100),
            self._cds("DNA gyrase subunit A", 500),
            self._cds("threonine synthase", 900),
            self._cds("hypothetical protein", 1300),
        ]
        strict = synteny_producer.strict_subsample(sample)
        assert len(strict) <= len(sample), "a filtered subset cannot outnumber its source"
        assert set(id(c) for c in strict) <= set(id(c) for c in sample), "same objects, filtered"

    def test_it_removes_exactly_the_d4_class_members(self) -> None:
        """Asserted by IDENTITY: a filter that dropped the wrong two would keep the count."""
        aars = self._cds("alanine--tRNA ligase", 100)
        gyrase = self._cds("DNA gyrase subunit A", 500)
        thrc = self._cds("threonine synthase", 900)
        hypo = self._cds("hypothetical protein", 1300)
        strict = synteny_producer.strict_subsample([aars, gyrase, thrc, hypo])
        assert [c.feature_id for c in strict] == [gyrase.feature_id, hypo.feature_id]

    def test_an_all_d4_sample_yields_nothing_rather_than_everything(self) -> None:
        sample = [self._cds("alanine--tRNA ligase", 100), self._cds("threonine synthase", 500)]
        assert synteny_producer.strict_subsample(sample) == []


class TestStrictArmReusesTheSameWindows:
    def test_the_two_utr_arms_share_a_single_draw(self, false_pass: dict) -> None:
        """⚠ The report claimed "the same windows, filtered" while the strict arm re-drew its
        spans with a fresh ``rng.choice(lengths)`` — that is a second draw over a nested CDS
        population, not a subset of one draw.  Both arms now read the same window objects."""
        wide = false_pass["arms"]["nine_one_five_prime_utr_decoys"]
        strict = false_pass["arms"]["nine_one_five_prime_utr_decoys_excluding_d4_classes"]
        assert strict["n"] <= wide["n"]
        # Every status bucket of the subset is bounded by its source's.
        for status, count in strict["status_counts"].items():
            assert count <= wide["status_counts"].get(status, 0), status
        assert "SAME window objects" in strict["note"]
        assert "strict subset" in strict["note"]

    def test_the_reported_rate_matches_its_own_counts(self, false_pass: dict) -> None:
        """A rate that does not reconcile with the counts beside it is a number nobody can
        check — and the PR description had already drifted from it once."""
        for name, arm in false_pass["arms"].items():
            if "n" not in arm:
                continue
            assert arm["n"] == sum(arm["status_counts"].values()), name
            # ⚠ The denominator is the DECIDED windows, not every drawn one: a window whose
            # (c) is unavailable is one the rule was never asked, not one it declined to
            # false-pass.  This assertion caught that contract change when it landed.
            # Guard n_decided, not n: an arm can draw windows and decide none of them, and
            # _arm then reports ``None`` rather than 0.0 — comparing that raises TypeError.
            if arm["n_decided"]:
                expected = arm["status_counts"].get(STATUS_PASSED, 0) / arm["n_decided"]
                assert arm["false_pass_rate"] == pytest.approx(expected), name
            else:
                assert arm["false_pass_rate"] is None, name
        control = false_pass["positive_context_control"]
        assert "false_pass_rate" not in control, (
            "the control arm's rate is a PASS rate; publishing it under false_pass_rate "
            "reports a 99.5% false-pass rate to anything that reads every such key"
        )
        assert control["pass_rate"] == pytest.approx(
            control["status_counts"].get(STATUS_PASSED, 0) / control["n_decided"]
        )
        background = false_pass["arms"]["clade_matched_random_leaders"]["false_pass_rate"]
        assert false_pass["control"]["margin"] == pytest.approx(control["pass_rate"] - background)


class TestUtrArmWindowsAreOneDraw:
    """The relation the committed-report assertion could not guard: strict ⊆ wide, by identity."""

    def _cds(self, product: str, start: int, strand: str = "+") -> gff3.CdsFeature:
        return gff3.CdsFeature(
            seqid="c1",
            feature_id=f"f{start}",
            start=start,
            end=start + 300,
            strand=strand,
            segments=((start, start + 300),),
            attributes={"product": (product,)},
        )

    def test_the_strict_list_is_a_sublist_of_the_wide_one(self) -> None:
        sample = [
            self._cds("alanine--tRNA ligase", 1000),
            self._cds("DNA gyrase subunit A", 2000),
            self._cds("threonine synthase", 3000),
            self._cds("hypothetical protein", 4000),
        ]
        wide, strict = utr_windows_for(sample)
        assert len(strict) < len(wide), "the D4-class members must actually be removed"
        # ⚠ ``is``, not ``==``.  Windows are plain tuples, so a producer that RECONSTRUCTS an
        # equal tuple — i.e. draws a second time and happens to land on the same span —
        # satisfies value equality while violating the "one draw" contract this test exists
        # to pin.  Object identity is the only assertion that distinguishes them.
        assert all(any(w is x for x in wide) for w in strict), "same objects, not equal ones"
        assert strict[0] is wide[1]
        assert strict[1] is wide[3]
        assert len(strict) == 2

    def test_an_all_non_d4_sample_keeps_every_window(self) -> None:
        """The positive control: a filter that removed everything would satisfy the test above."""
        sample = [self._cds("DNA gyrase subunit A", 1000), self._cds("hypothetical protein", 2000)]
        wide, strict = utr_windows_for(sample)
        assert len(strict) == len(wide)
        assert all(a is b for a, b in zip(strict, wide, strict=True))

    def test_a_span_count_mismatch_refuses(self) -> None:
        sample = [self._cds("DNA gyrase subunit A", 1000)]
        with pytest.raises(synteny_producer.ProducerError, match="one span per CDS"):
            synteny_producer.utr_arm_windows(sample, spans=[100, 200], extents={"c1": 99999})


def utr_windows_for(sample):
    return synteny_producer.utr_arm_windows(
        sample, spans=[120] * len(sample), extents={"c1": 999_999}
    )


class TestErrorsAreTranslatedAtTheMiningBoundary:
    """A contradictory status table is a fault the round must REPORT, not crash on."""

    def _bad_table(self, tmp_path: Path) -> Path:
        table = synteny_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        table["status"]["a"] = STATUS_FAILED  # contradicts its own rows
        path = tmp_path / "synteny_status.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        return path

    @pytest.mark.parametrize(
        ("payload", "label"),
        [
            ("{ truncated", "JSONDecodeError"),
            ("[1, 2, 3]", "non-object root -> ProducerError"),
            (
                '{"status": {"a": "passed"}, "rows": [{"no_id": 1}]}',
                "row missing keys -> ProducerError",
            ),
            ('{"status": {"a": "passed"}, "rows": "not-a-list"}', "wrong rows type"),
        ],
    )
    def test_every_input_fault_becomes_the_rounds_own_error(
        self, tmp_path: Path, payload: str, label: str
    ) -> None:
        """⚠ ``ProducerError`` is not the whole failure surface: ``load_status_map`` reads
        arbitrary JSON, so a truncated file, a non-object root and a malformed row each raise
        a *different* builtin — and every one of them would bypass the documented refusal path
        and surface as an unhandled traceback."""
        path = tmp_path / "synteny_status.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match="synteny status table unusable"):
            mine_round.apply_spare_rule(
                "m.json",
                "s.json",
                rscape_installed=True,
                msa_supply_available=True,
                synteny_status_table=path,
                synteny_available=True,
            )
        with pytest.raises(remine.RemineError, match="synteny status table unusable"):
            remine.apply_remine_spare_rule(
                "m.json",
                "s.json",
                "p.json",
                stage2_threshold=0.9,
                rscape_installed=True,
                msa_supply_available=True,
                stage2_supply_available=True,
                probe_set=None,
                synteny_status_table=path,
                synteny_available=True,
            )

    def test_the_p2_round_raises_its_own_error_type(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="synteny status table unusable"):
            mine_round.apply_spare_rule(
                "m.json",
                "s.json",
                rscape_installed=True,
                msa_supply_available=True,
                synteny_status_table=self._bad_table(tmp_path),
                synteny_available=True,
            )

    def test_the_p3_round_raises_its_own_error_type(self, tmp_path: Path) -> None:
        with pytest.raises(remine.RemineError, match="synteny status table unusable"):
            remine.apply_remine_spare_rule(
                "m.json",
                "s.json",
                "p.json",
                stage2_threshold=0.9,
                rscape_installed=True,
                msa_supply_available=True,
                stage2_supply_available=True,
                probe_set=None,
                synteny_status_table=self._bad_table(tmp_path),
                synteny_available=True,
            )


class TestReaderRefusesDuplicates:
    def test_a_duplicate_candidate_id_in_rows_is_refused_at_READ_time(self, tmp_path: Path) -> None:
        """⚠ A dict comprehension collapses a repeated id to its last occurrence, and the
        status↔rows cross-check would then agree with itself.  The writer refuses duplicates,
        but a table can reach the reader without having been written by it."""
        table = synteny_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        table["rows"].append(dict(table["rows"][0], status=STATUS_FAILED))
        path = tmp_path / "t.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        with pytest.raises(synteny_producer.ProducerError, match="duplicate candidate_id"):
            synteny_producer.load_status_map(path)

    def test_positive_control_distinct_ids_load(self, tmp_path: Path) -> None:
        table = synteny_producer.build_status_table(
            [row("a", STATUS_PASSED), row("b", STATUS_FAILED)], config=CONFIG
        )
        path = tmp_path / "t.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        assert synteny_producer.load_status_map(path) == {"a": STATUS_PASSED, "b": STATUS_FAILED}


class TestAnnotatedSetUsesTheSharedContract:
    """The accession→filename contract has one definition; a second copy undercounts silently."""

    def test_it_counts_the_materialized_hosts(self, tmp_path: Path) -> None:
        from tbox_finder.mining import annotation_fetch

        for accession in ("GCA_000296795.1", "GCF_000005845.2"):
            (tmp_path / annotation_fetch.destination_name(accession)).write_bytes(b"\x1f\x8b")
        (tmp_path / "README.md").write_text("not an annotation", encoding="utf-8")
        (tmp_path / "GCA_000296795.1.gff.gz.part").write_bytes(b"partial")
        found = synteny_producer._annotated_set(str(tmp_path))
        assert found == {"GCA_000296795.1", "GCF_000005845.2"}

    def test_a_missing_directory_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert synteny_producer._annotated_set(str(tmp_path / "absent")) == set()


class TestFalsePassDenominator:
    def test_undecided_windows_are_excluded_from_the_denominator(self) -> None:
        """⚠ A window whose (c) is ``unavailable`` is not one where the rule declined to
        false-pass — it is one where the rule was never asked.  Counting it dilutes the rate
        by exactly the annotation gaps the *other* diagnostic exists to report."""
        arm = synteny_producer._arm(
            Counter({STATUS_PASSED: 1, STATUS_FAILED: 1, STATUS_UNAVAILABLE: 8})
        )
        assert arm["n"] == 10 and arm["n_decided"] == 2 and arm["n_unavailable"] == 8
        assert arm["false_pass_rate"] == pytest.approx(0.5), "1 of 2 DECIDED, not 1 of 10"

    def test_an_arm_with_nothing_decided_reports_None_not_zero(self) -> None:
        """[[clauses-must-guard-emptiness]] — 0.0 would read as "measured, found nothing"."""
        arm = synteny_producer._arm(Counter({STATUS_UNAVAILABLE: 5}))
        assert arm["false_pass_rate"] is None

    def test_the_committed_arms_reconcile_against_the_decided_denominator(
        self, false_pass: dict
    ) -> None:
        for name, arm in false_pass["arms"].items():
            if "n" not in arm:
                continue
            assert arm["n_decided"] == arm["status_counts"].get(STATUS_PASSED, 0) + arm[
                "status_counts"
            ].get(STATUS_FAILED, 0), name
            if arm["n_decided"]:
                assert arm["false_pass_rate"] == pytest.approx(
                    arm["status_counts"].get(STATUS_PASSED, 0) / arm["n_decided"]
                ), name


class TestRoundTenHardening:
    def test_the_control_arm_publishes_a_pass_rate_not_a_false_pass_rate(
        self, false_pass: dict
    ) -> None:
        """⚠ Windows drawn upstream of a CDS already in a D4 class are SUPPOSED to pass.  At
        99.5 % under the key ``false_pass_rate``, any consumer that reads every such key in
        this file learns that criterion (c) false-passes almost always."""
        control = false_pass["positive_context_control"]
        assert "pass_rate" in control and "false_pass_rate" not in control
        assert control["pass_rate"] > 0.9
        for name, arm in false_pass["arms"].items():
            if "n" in arm:
                assert "pass_rate" not in arm, f"{name} is a decoy arm, not a control"

    def test_a_row_missing_its_keys_refuses_as_a_producer_error(self, tmp_path: Path) -> None:
        table = synteny_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        table["rows"][0].pop("status")
        path = tmp_path / "t.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        with pytest.raises(
            synteny_producer.ProducerError, match="lacks 'candidate_id' or 'status'"
        ):
            synteny_producer.load_status_map(path)

    def test_the_clade_join_refuses_a_duplicate_assembly(self, tmp_path: Path) -> None:
        """A repeat carrying a different phylum silently reassigns every candidate on that
        host — and the per-clade exclusion rate is the headline of the whole diagnostic."""
        pytest.importorskip("pyarrow")  # to_parquet needs an engine, absent in some envs
        pd = pytest.importorskip("pandas")
        path = tmp_path / "clades.parquet"
        pd.DataFrame(
            {
                "assembly_accession": ["GCA_1", "GCA_1", "GCA_2"],
                "phylum": ["Bacillota", "Chloroflexota", "Spirochaetota"],
            }
        ).to_parquet(path)
        with pytest.raises(synteny_producer.ProducerError, match="duplicate assembly_accession"):
            synteny_producer.load_clades(path)

    def test_positive_control_distinct_assemblies_join(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")  # to_parquet needs an engine, absent in some envs
        pd = pytest.importorskip("pandas")
        path = tmp_path / "clades.parquet"
        pd.DataFrame(
            {"assembly_accession": ["GCA_1", "GCA_2"], "phylum": ["Bacillota", "Spirochaetota"]}
        ).to_parquet(path)
        assert synteny_producer.load_clades(path) == {
            "GCA_1": "Bacillota",
            "GCA_2": "Spirochaetota",
        }


class TestControlArmIsRekeyedNotCopied:
    """⚠ The sixth artifact-pinning miss of this step: the committed-report assertion stayed
    green when the producer *copied* the key instead of renaming it, because the report it
    reads was written by the fixed code."""

    def test_the_false_pass_key_is_removed_not_duplicated(self) -> None:
        arm = synteny_producer._arm(Counter({STATUS_PASSED: 99, STATUS_FAILED: 1}))
        control = synteny_producer.as_control_arm(arm)
        assert control["pass_rate"] == pytest.approx(0.99)
        assert "false_pass_rate" not in control
        assert "false_pass_rate_denominator" not in control
        assert control["pass_rate_denominator"] == "decided_windows"

    def test_the_counts_survive_the_rekey(self) -> None:
        """Only the rate keys move; a re-key that dropped the evidence would pass the test
        above just as well."""
        arm = synteny_producer._arm(Counter({STATUS_PASSED: 99, STATUS_FAILED: 1}))
        control = synteny_producer.as_control_arm(arm)
        for key in ("n", "n_decided", "n_unavailable", "status_counts"):
            assert control[key] == arm[key], key

    def test_the_source_arm_is_not_mutated(self) -> None:
        arm = synteny_producer._arm(Counter({STATUS_PASSED: 1, STATUS_FAILED: 1}))
        synteny_producer.as_control_arm(arm)
        assert "false_pass_rate" in arm, "the decoy arms must keep their own key"


class TestOneBadFileDoesNotAbortTheRun:
    """⚠ `-c-i` recorded this exact defect: a read sitting above the handler written for it
    aborted all 339 verdicts on one bad file.  Here it would abort all 941."""

    def _host(self, tmp_path: Path, payload: bytes) -> synteny_producer.HostAnnotation:
        from tbox_finder.mining import annotation_fetch

        accession = "GCA_000296795.1"
        (tmp_path / annotation_fetch.destination_name(accession)).write_bytes(payload)
        genomes = tmp_path / "genomes"
        genomes.mkdir(exist_ok=True)
        (genomes / f"{accession}.fna").write_text(">c1 x\nACGT\n", encoding="utf-8")
        return synteny_producer.HostAnnotation(
            accession, annotation_dir=str(tmp_path), genome_dir=str(genomes)
        )

    def test_a_corrupt_gff_makes_that_host_unavailable(self, tmp_path: Path) -> None:
        host = self._host(tmp_path, b"this is not a gff and has no version directive\n")
        assert host.available is False
        assert host.reason == synteny_producer.REASON_ANNOTATION_UNREADABLE
        assert host.note, "the refusal must say which file and why"

    def test_positive_control_a_valid_gff_is_available(self, tmp_path: Path) -> None:
        """Without this, an __init__ that marked EVERY host unreadable would pass above."""
        gff = b"##gff-version 3\nc1\ts\tCDS\t10\t99\t.\t+\t0\tID=a;product=alanine--tRNA ligase\n"
        host = self._host(tmp_path, gff)
        assert host.available is True and host.reason == ""
        assert len(host.cds) == 1

    def test_the_reason_is_in_the_closed_vocabulary(self) -> None:
        assert synteny_producer.REASON_ANNOTATION_UNREADABLE in synteny_producer.EXCLUSION_REASONS

    def test_a_corrupt_host_is_counted_apart_from_an_unannotated_one(self) -> None:
        """A corrupt corpus must not hide inside a missing one — the per-clade exclusion rate
        would read identically either way."""
        report = synteny_producer.exclusion_report(
            [
                row("a", STATUS_UNAVAILABLE, reason=synteny_producer.REASON_ANNOTATION_UNREADABLE),
                row("b", STATUS_UNAVAILABLE, reason=synteny_producer.REASON_HOST_UNANNOTATED),
            ],
            clades={"GCA_000000001.1": "Bacillota"},
            config=CONFIG,
        )
        assert report["exclusion_reason_totals"] == {
            synteny_producer.REASON_ANNOTATION_UNREADABLE: 1,
            synteny_producer.REASON_HOST_UNANNOTATED: 1,
        }


class TestOneParsePerHostCoversBothArms:
    def test_the_host_caches_its_trna_features_too(self, tmp_path: Path) -> None:
        """⚠ Re-reading the file for the tRNA arm defeated the cache that is the reason the
        round takes seconds — and read a SECOND snapshot of a file the CDS arm had already
        committed to."""
        from tbox_finder.mining import annotation_fetch

        accession = "GCA_000296795.1"
        (tmp_path / annotation_fetch.destination_name(accession)).write_bytes(
            b"##gff-version 3\n"
            b"c1\ts\tCDS\t10\t99\t.\t+\t0\tID=a;product=alanine--tRNA ligase\n"
            b"c1\ts\ttRNA\t200\t280\t.\t+\t.\tID=t1\n"
            b"c1\ts\ttRNA\t400\t480\t.\t-\t.\tID=t2\n"
        )
        genomes = tmp_path / "g"
        genomes.mkdir()
        (genomes / f"{accession}.fna").write_text(">c1 x\nACGT\n", encoding="utf-8")
        host = synteny_producer.HostAnnotation(
            accession, annotation_dir=str(tmp_path), genome_dir=str(genomes)
        )
        assert [f.feature_id for f in host.trna] == ["t1", "t2"]
        assert len(host.cds) == 1, "the CDS arm is unchanged by the tRNA caching"

    def test_an_unavailable_host_exposes_an_empty_trna_list(self, tmp_path: Path) -> None:
        """The attribute must exist either way, or the tRNA arm raises on a skipped host."""
        host = synteny_producer.HostAnnotation(
            "GCA_000296795.1", annotation_dir=str(tmp_path), genome_dir=str(tmp_path)
        )
        assert host.available is False and host.trna == []


class TestRoundFifteenGuards:
    def test_an_empty_candidate_list_refuses(self) -> None:
        """Without candidates there is no length distribution to match, and every
        ``rng.choice(lengths)`` in the three control arms raises ``IndexError`` — which the CLI
        does not convert, so the run aborts with a traceback."""
        with pytest.raises(synteny_producer.ProducerError, match="no candidates"):
            synteny_producer.false_pass_report(
                [],
                annotation_dir="unused",
                genome_dir="unused",
                config=CONFIG,
                clades={},
                n_per_host=1,
                seed=1,
                decoys_parquet="absent.parquet",
                mining_pool_parquet="absent.parquet",
            )

    def test_a_scalar_manifest_row_refuses_as_a_producer_error(self, tmp_path: Path) -> None:
        """``dict(row)`` on a scalar raises ``TypeError``, which is not this module's contract."""
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"candidates": ["not-an-object"]}), encoding="utf-8")
        with pytest.raises(synteny_producer.ProducerError, match="not an object"):
            synteny_producer.read_manifest(path)

    def test_positive_control_a_well_formed_manifest_reads(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": "x",
                            "accession": "GCA_000000001.1:c0",
                            "locus_start": 1,
                            "locus_end": 2,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert synteny_producer.read_manifest(path)[0]["candidate_id"] == "x"

    def test_exclusion_report_takes_no_parameter_it_does_not_read(self) -> None:
        """A required parameter with no effect makes every call site assert something false."""
        import inspect

        params = set(inspect.signature(synteny_producer.exclusion_report).parameters)
        assert "annotation_dir" not in params and "genome_dir" not in params


class TestOneBadFastaDoesNotAbortTheRunEither:
    """Round 12 guarded the annotation read; the genome read beside it was still bare."""

    def _host(self, tmp_path: Path, fasta: bytes) -> synteny_producer.HostAnnotation:
        from tbox_finder.mining import annotation_fetch

        accession = "GCA_000296795.1"
        (tmp_path / annotation_fetch.destination_name(accession)).write_bytes(
            b"##gff-version 3\nc1\ts\tCDS\t10\t99\t.\t+\t0\tID=a;product=alanine--tRNA ligase\n"
        )
        genomes = tmp_path / "g"
        genomes.mkdir(exist_ok=True)
        (genomes / f"{accession}.fna").write_bytes(fasta)
        return synteny_producer.HostAnnotation(
            accession, annotation_dir=str(tmp_path), genome_dir=str(genomes)
        )

    def test_an_empty_fasta_makes_that_host_unavailable(self, tmp_path: Path) -> None:
        host = self._host(tmp_path, b"")
        assert host.available is False
        assert host.reason == synteny_producer.REASON_GENOME_UNREADABLE
        assert host.note

    def test_a_header_with_no_record_id_makes_that_host_unavailable(self, tmp_path: Path) -> None:
        host = self._host(tmp_path, b">\nACGT\n")
        assert host.reason == synteny_producer.REASON_GENOME_UNREADABLE

    def test_positive_control_a_valid_fasta_is_available(self, tmp_path: Path) -> None:
        host = self._host(tmp_path, b">c1 desc\nACGT\n")
        assert host.available is True and host.contig_ids == ["c1"]

    def test_unreadable_is_counted_apart_from_absent(self) -> None:
        """A corrupt genome and a missing one must not be indistinguishable in the diagnostic."""
        assert synteny_producer.REASON_GENOME_UNREADABLE != synteny_producer.REASON_GENOME_ABSENT
        assert synteny_producer.REASON_GENOME_UNREADABLE in synteny_producer.EXCLUSION_REASONS


class TestShardingKeepsHostsWhole:
    """⚠ A strided slice scatters one host across every shard, and `_host_cache` is built per
    ``run_shard`` call — so each shard re-parses almost every GFF, defeating the cache the
    module documents as the reason the round takes seconds."""

    def _rows(self, n_hosts: int, per_host: int) -> list[dict]:
        return [
            {"accession": f"GCA_{h:03d}.1:c0", "candidate_id": f"{h}-{k}"}
            for h in range(n_hosts)
            for k in range(per_host)
        ]

    def test_every_host_lands_in_exactly_one_shard(self) -> None:
        rows = self._rows(7, 4)
        shards = [synteny_producer.shard_by_host(rows, i, 3) for i in range(3)]
        seen: dict[str, int] = {}
        for index, shard in enumerate(shards):
            for row in shard:
                host = row["accession"].partition(":c")[0]
                assert seen.setdefault(host, index) == index, f"{host} spans shards"
        assert len(seen) == 7

    def test_the_partition_is_complete_and_disjoint(self) -> None:
        """A partition that dropped or duplicated candidates would still keep hosts whole."""
        rows = self._rows(7, 4)
        shards = [synteny_producer.shard_by_host(rows, i, 3) for i in range(3)]
        ids = [r["candidate_id"] for shard in shards for r in shard]
        assert sorted(ids) == sorted(r["candidate_id"] for r in rows)
        assert len(ids) == len(set(ids))

    def test_a_strided_slice_would_fail_the_first_assertion(self) -> None:
        """The control that shows the assertion has teeth: the old behaviour scatters hosts."""
        rows = self._rows(7, 4)
        strided = [rows[i::3] for i in range(3)]
        spanning = {
            r["accession"].partition(":c")[0]
            for index, shard in enumerate(strided)
            for r in shard
            if any(
                any(o["accession"] == r["accession"] for o in other)
                for j, other in enumerate(strided)
                if j != index
            )
        }
        assert spanning, "a strided slice must scatter at least one host"

    def test_an_out_of_range_shard_refuses(self) -> None:
        with pytest.raises(synteny_producer.ProducerError, match="out of range"):
            synteny_producer.shard_by_host(self._rows(2, 1), 3, 3)
