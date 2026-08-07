"""Unit tests for the criterion-(b) producer, its supply derivation, and its round wiring.

Every test here is written so that **breaking the thing it names** turns it red.

The trap this file exists to avoid: an assertion over the committed
``reports/p3/architecture_freeze.json`` cannot see a producer that writes the file
correctly for the wrong reason, because the bytes on disk were written by the code being
sabotaged.  So where the property is a relation — "the freeze measures the held-out carve",
"the derivation follows its clauses" — it is exercised on inputs built here, and the
committed report is checked only for the things a report genuinely owns (its shape, and
that it pinned nothing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder.mining import architecture, architecture_producer, mine_round, remine
from tbox_finder.mining.spare_rule import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = architecture_producer.ArchitectureRunConfig(
    stem_i_nt_threshold=60,
    min_named_helices=2,
    min_helix_pairs=1,
    bulge_min_nt=5,
    bulge_max_nt=9,
    ncca_pairing_nt=4,
)

ANTITERM_SEQ = "GCGGUGGCACCGCGAGUUCCCUUCUCGCCCGC"
ANTITERM_SS = "((((.......(((((.......)))))))))"


def write_msa(msa_dir: Path, candidate_id: str, seq: str, ss: str, *, n_rows: int = 24) -> Path:
    from tbox_finder.mining.covariation_producer import candidate_slug

    target = msa_dir / candidate_slug(candidate_id) / "msa.sto"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# STOCKHOLM 1.0"]
    lines += [f"s{i} {seq}" for i in range(n_rows)]
    lines += [f"#=GC SS_cons {ss}", "//"]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"candidates": rows}), encoding="utf-8")
    return path


def row(candidate_id: str, status: str, **extra) -> dict:
    return {"candidate_id": candidate_id, "status": status, **extra}


# ═════════════════════════════════════════════════════════════════════════════
class TestEvaluateCandidate:
    def test_a_candidate_with_a_canonical_consensus_passes(self, tmp_path: Path) -> None:
        write_msa(tmp_path, "c1", ANTITERM_SEQ, ANTITERM_SS)
        result = architecture_producer.evaluate_candidate(
            {"candidate_id": "c1", "stem_i_extent_nt": 96}, msa_dir=tmp_path, config=CONFIG
        )
        assert result["status"] == STATUS_PASSED
        assert result["msa_present"] is True

    def test_an_absent_consensus_is_unavailable_never_failed(self, tmp_path: Path) -> None:
        """ADR-0005 D14: a candidate the producer never scored is SPARED, not mined."""
        result = architecture_producer.evaluate_candidate(
            {"candidate_id": "missing"}, msa_dir=tmp_path, config=CONFIG
        )
        assert result["status"] == STATUS_UNAVAILABLE
        assert result["msa_present"] is False

    def test_a_corrupt_consensus_is_unavailable_never_failed(self, tmp_path: Path) -> None:
        write_msa(tmp_path, "c1", ANTITERM_SEQ, "((((" + "." * 28)  # unbalanced
        result = architecture_producer.evaluate_candidate(
            {"candidate_id": "c1"}, msa_dir=tmp_path, config=CONFIG
        )
        assert result["status"] == STATUS_UNAVAILABLE
        assert "unusable" in result["reason"]

    def test_a_shallow_alignment_is_unavailable_never_failed(self, tmp_path: Path) -> None:
        write_msa(tmp_path, "c1", ANTITERM_SEQ, ANTITERM_SS, n_rows=19)
        result = architecture_producer.evaluate_candidate(
            {"candidate_id": "c1"}, msa_dir=tmp_path, config=CONFIG
        )
        assert result["status"] == STATUS_UNAVAILABLE

    def test_a_real_negative_is_FAILED_not_unavailable(self, tmp_path: Path) -> None:
        """The distinction that decides whether the corpus can be mined at all."""
        broken = ANTITERM_SEQ[:4] + "AAAAAAA" + ANTITERM_SEQ[11:]
        write_msa(tmp_path, "c1", broken, ANTITERM_SS)
        result = architecture_producer.evaluate_candidate(
            {"candidate_id": "c1"}, msa_dir=tmp_path, config=CONFIG
        )
        assert result["status"] == STATUS_FAILED

    def test_a_row_without_a_candidate_id_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(architecture_producer.ProducerError, match="candidate_id"):
            architecture_producer.evaluate_candidate({}, msa_dir=tmp_path, config=CONFIG)

    def test_the_msa_path_uses_the_a_producers_own_slug(self, tmp_path: Path) -> None:
        """A forked slug would read no MSAs and report `unavailable` for the whole corpus."""
        from tbox_finder.mining.covariation_producer import candidate_slug

        path = architecture_producer.candidate_msa_path(tmp_path, "some/candidate:id")
        assert path.parent.name == candidate_slug("some/candidate:id")
        assert path.name == "msa.sto"


class TestStatusTable:
    def test_status_is_derived_from_rows_and_counts_reconcile(self) -> None:
        table = architecture_producer.build_status_table(
            [row("a", STATUS_PASSED), row("b", STATUS_FAILED), row("c", STATUS_UNAVAILABLE)],
            config=CONFIG,
        )
        assert table["status"] == {"a": "passed", "b": "failed", "c": "unavailable"}
        assert table["status_counts"] == {"failed": 1, "passed": 1, "unavailable": 1}
        assert table["n_candidates"] == 3

    def test_a_duplicate_candidate_id_refuses_rather_than_merging(self) -> None:
        """A duplicate key MERGES instead of colliding and every summed invariant still
        reconciles — the fault is invisible in the counts, so it is refused explicitly."""
        with pytest.raises(architecture_producer.ProducerError, match="duplicate"):
            architecture_producer.build_status_table(
                [row("a", STATUS_PASSED), row("a", STATUS_FAILED)], config=CONFIG
            )

    def test_the_config_is_stamped_and_records_that_no_cm_is_on_the_path(self) -> None:
        table = architecture_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        assert table["config"]["cmalign_on_path"] is False
        assert table["config"]["rf00230_on_path"] is False
        assert table["config"]["stem_i_nt_threshold"] == 60

    def test_load_refuses_a_table_that_disagrees_with_itself(self, tmp_path: Path) -> None:
        table = architecture_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        table["status"]["a"] = STATUS_FAILED
        path = tmp_path / "t.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        with pytest.raises(architecture_producer.ProducerError, match="disagrees"):
            architecture_producer.load_status_map(path)

    def test_positive_control_an_agreeing_table_loads(self, tmp_path: Path) -> None:
        table = architecture_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        path = tmp_path / "t.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        assert architecture_producer.load_status_map(path) == {"a": "passed"}

    def test_an_unknown_status_string_refuses(self, tmp_path: Path) -> None:
        payload = {"status": {"a": "probably"}, "rows": [row("a", "probably")]}
        with pytest.raises(architecture_producer.ProducerError, match="unknown statuses"):
            architecture_producer.validate_status_payload(payload, label="t")


class TestMerge:
    def _shard(self, tmp_path: Path, name: str, rows: list[dict], config=CONFIG) -> Path:
        path = tmp_path / name
        path.write_text(
            json.dumps(architecture_producer.build_status_table(rows, config=config)),
            encoding="utf-8",
        )
        return path

    def test_shards_concatenate(self, tmp_path: Path) -> None:
        a = self._shard(tmp_path, "a.json", [row("a", STATUS_PASSED)])
        b = self._shard(tmp_path, "b.json", [row("b", STATUS_FAILED)])
        merged = architecture_producer.merge_status_tables([a, b])
        assert merged["status"] == {"a": "passed", "b": "failed"}
        assert merged["n_candidates"] == 2

    def test_a_candidate_in_two_shards_refuses(self, tmp_path: Path) -> None:
        a = self._shard(tmp_path, "a.json", [row("x", STATUS_PASSED)])
        b = self._shard(tmp_path, "b.json", [row("x", STATUS_FAILED)])
        with pytest.raises(architecture_producer.ProducerError, match="more than one shard"):
            architecture_producer.merge_status_tables([a, b])

    def test_shards_produced_under_different_configs_refuse(self, tmp_path: Path) -> None:
        """Merging them would publish one round's parameters over another round's rows."""
        other = architecture_producer.ArchitectureRunConfig(
            stem_i_nt_threshold=999,
            min_named_helices=2,
            min_helix_pairs=1,
            bulge_min_nt=5,
            bulge_max_nt=9,
            ncca_pairing_nt=4,
        )
        a = self._shard(tmp_path, "a.json", [row("a", STATUS_PASSED)])
        b = self._shard(tmp_path, "b.json", [row("b", STATUS_PASSED)], config=other)
        with pytest.raises(architecture_producer.ProducerError, match="different configs"):
            architecture_producer.merge_status_tables([a, b])

    def test_positive_control_matching_configs_merge(self, tmp_path: Path) -> None:
        a = self._shard(tmp_path, "a.json", [row("a", STATUS_PASSED)])
        b = self._shard(tmp_path, "b.json", [row("b", STATUS_PASSED)])
        assert architecture_producer.merge_status_tables([a, b])["n_candidates"] == 2

    def test_merging_nothing_refuses(self) -> None:
        with pytest.raises(architecture_producer.ProducerError, match="no shard tables"):
            architecture_producer.merge_status_tables([])


class TestSharding:
    def test_every_candidate_lands_in_exactly_one_shard(self) -> None:
        cands = [{"candidate_id": f"c{i}"} for i in range(17)]
        seen: list[str] = []
        for s in range(4):
            seen += [c["candidate_id"] for c in architecture_producer.shard_candidates(cands, s, 4)]
        assert sorted(seen) == sorted(c["candidate_id"] for c in cands)
        assert len(seen) == len(set(seen))

    def test_a_bad_shard_index_refuses(self) -> None:
        with pytest.raises(architecture_producer.ProducerError, match="bad shard"):
            architecture_producer.shard_candidates([], 4, 4)


class TestSupplyDerivation:
    """Each clause is broken ALONE — an all-true fixture cannot test a conjunction."""

    def _repo(self, tmp_path: Path) -> Path:
        for _, rel in architecture_producer.SUPPLY_CLAUSES:
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        (tmp_path / architecture_producer.FREEZE_REPORT).write_text(
            json.dumps({"carve": {"n_with_discriminator_in_a_flanked_bulge": 8605}}),
            encoding="utf-8",
        )
        return tmp_path

    def test_the_real_checkout_evidences_the_supply(self) -> None:
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=REPO_ROOT)
        assert derivation["available"] is True, derivation["reasons"]
        assert derivation["n_heldout_canonical_measured"] > 0

    def test_the_constant_agrees_with_the_derivation(self) -> None:
        """A stale False fails here; a True on a checkout that cannot evidence it also fails."""
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=REPO_ROOT)
        assert derivation["available"] == mine_round.RELAXED_ARCH_SUPPLY_AVAILABLE

    @pytest.mark.parametrize("clause,rel", list(architecture_producer.SUPPLY_CLAUSES))
    def test_breaking_ONE_file_clause_alone_flips_the_verdict(
        self, tmp_path: Path, clause: str, rel: str
    ) -> None:
        repo = self._repo(tmp_path)
        (repo / rel).unlink()
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=repo)
        assert derivation["available"] is False
        assert derivation["clauses"][clause] is False

    def test_an_empty_freeze_carve_fails_its_own_clause(self, tmp_path: Path) -> None:
        """A freeze report that measured nothing certifies the predicate against nothing."""
        repo = self._repo(tmp_path)
        (repo / architecture_producer.FREEZE_REPORT).write_text(
            json.dumps({"carve": {"n_with_discriminator_in_a_flanked_bulge": 0}}), encoding="utf-8"
        )
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=repo)
        assert derivation["clauses"]["freeze_measured_a_nonempty_carve"] is False
        assert derivation["available"] is False

    def test_a_true_flag_is_not_accepted_as_a_count(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        (repo / architecture_producer.FREEZE_REPORT).write_text(
            json.dumps({"carve": {"n_with_discriminator_in_a_flanked_bulge": True}}),
            encoding="utf-8",
        )
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=repo)
        assert derivation["clauses"]["freeze_measured_a_nonempty_carve"] is False

    def test_a_malformed_freeze_report_fails_closed_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        repo = self._repo(tmp_path)
        (repo / architecture_producer.FREEZE_REPORT).write_text("not json", encoding="utf-8")
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=repo)
        assert derivation["available"] is False

    def test_it_delegates_to_the_a_supply_rather_than_probing_twice(self, tmp_path: Path) -> None:
        """A4: (b) reads the same consensus (a) does, so a checkout without the MSA supply
        cannot evidence (b). A second hand-written probe would be free to drift from it."""
        repo = self._repo(tmp_path)  # has (b)'s files but none of (a)'s evidence
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=repo)
        assert derivation["clauses"]["msa_supply_backs_it"] is False
        assert any("A4" in r or "same consensus" in r for r in derivation["reasons"])


class TestRoundWiring:
    def test_a_produced_status_reaches_the_candidates_b_disjunct(self) -> None:
        """A producer that ships but whose output the round drops is the no-op this catches."""
        evidence = mine_round.candidate_evidence("x", None, None, {"x": STATUS_PASSED})
        assert evidence.relaxed_architecture == STATUS_PASSED

    def test_an_absent_id_resolves_to_unavailable_not_failed(self) -> None:
        evidence = mine_round.candidate_evidence("x", None, None, {"other": STATUS_PASSED})
        assert evidence.relaxed_architecture == STATUS_UNAVAILABLE

    def test_no_table_at_all_resolves_to_unavailable(self) -> None:
        assert (
            mine_round.candidate_evidence("x", None, None, None).relaxed_architecture
            == STATUS_UNAVAILABLE
        )

    def test_a_produced_status_is_carried_INDEPENDENTLY_of_the_other_two(self) -> None:
        """A count-only assertion is blind to two disjuncts being swapped; assert identity."""
        evidence = mine_round.candidate_evidence(
            "x", {"x": STATUS_FAILED}, {"x": STATUS_UNAVAILABLE}, {"x": STATUS_PASSED}
        )
        assert evidence.relaxed_architecture == STATUS_PASSED
        assert evidence.any_helix_rscape == STATUS_FAILED
        assert evidence.downstream_aaRS_synteny == STATUS_UNAVAILABLE

    def test_availability_maps_the_b_flag_to_the_right_disjunct(self) -> None:
        avail = mine_round.build_round_availability(
            rscape_installed=False,
            msa_supply_available=False,
            relaxed_arch_available=True,
            synteny_available=False,
        )
        assert avail == {
            "relaxed_architecture": True,
            "any_helix_rscape": False,
            "downstream_aaRS_synteny": False,
        }

    def test_b_alone_still_does_NOT_make_a_round_ready(self) -> None:
        """ADR-0006 D9 row 5: (b) is False on every Tier-2N locus, so it protects nothing.
        Building its backend must not change that."""
        from tbox_finder.mining.spare_rule import mining_round_readiness

        readiness = mining_round_readiness(
            mine_round.build_round_availability(
                rscape_installed=False,
                msa_supply_available=False,
                relaxed_arch_available=True,
                synteny_available=False,
            )
        )
        assert readiness["ready"] is False
        assert "relaxed_architecture" in readiness["refusal_reason"]

    def test_b_is_still_absent_from_the_tier2n_protective_set(self) -> None:
        from tbox_finder.mining.spare_rule import TIER2N_PROTECTIVE_DISJUNCTS

        assert "relaxed_architecture" not in TIER2N_PROTECTIVE_DISJUNCTS


class TestApplySpareRulePairedRefusals:
    def test_declaring_b_available_without_a_table_refuses(self, tmp_path: Path) -> None:
        """The SILENT form of a refusal: every (b) status reads unavailable and all are spared."""
        with pytest.raises(mine_round.MineRoundError, match="no relaxed-architecture status"):
            mine_round.apply_spare_rule(
                tmp_path / "m.json",
                tmp_path / "s.json",
                rscape_installed=False,
                msa_supply_available=False,
                relaxed_arch_available=True,
                relaxed_arch_status_table=None,
            )

    def test_supplying_a_table_without_the_declaration_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(mine_round.MineRoundError, match="undeclared"):
            mine_round.apply_spare_rule(
                tmp_path / "m.json",
                tmp_path / "s.json",
                rscape_installed=False,
                msa_supply_available=False,
                relaxed_arch_available=False,
                relaxed_arch_status_table=tmp_path / "t.json",
            )


class TestCli:
    def test_every_rule_parameter_is_required_with_no_default(self) -> None:
        """A default would decide which candidates are mined without anyone choosing it."""
        parser = architecture_producer.build_parser()
        required = {
            "--stem-i-nt-threshold",
            "--min-named-helices",
            "--min-helix-pairs",
            "--bulge-min-nt",
            "--bulge-max-nt",
            "--ncca-pairing-nt",
        }
        run = next(
            a
            for a in parser._subparsers._group_actions[0].choices.values()  # type: ignore[attr-defined]
            if "--stem-i-nt-threshold" in {s for act in a._actions for s in act.option_strings}
        )
        for action in run._actions:
            for opt in action.option_strings:
                if opt in required:
                    assert action.required is True, f"{opt} is not required"
                    assert action.default is None, f"{opt} carries a default"
                    required.discard(opt)
        assert required == set(), f"never saw {required}"

    def test_run_shard_is_reachable_and_writes_a_table(self, tmp_path: Path) -> None:
        write_msa(tmp_path / "msa", "c1", ANTITERM_SEQ, ANTITERM_SS)
        m = manifest(tmp_path / "m.json", [{"candidate_id": "c1", "stem_i_extent_nt": 96}])
        out = tmp_path / "out.json"
        rc = architecture_producer.main(
            [
                "run-shard",
                "--manifest",
                str(m),
                "--msa-dir",
                str(tmp_path / "msa"),
                "--stem-i-nt-threshold",
                "60",
                "--min-named-helices",
                "2",
                "--min-helix-pairs",
                "1",
                "--bulge-min-nt",
                "5",
                "--bulge-max-nt",
                "9",
                "--ncca-pairing-nt",
                "4",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["status"] == {"c1": "passed"}
        assert payload["provenance"]["extra"]["stem_i_nt_threshold"] == 60

    def test_omitting_a_required_parameter_exits_nonzero(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            architecture_producer.main(["run-shard", "--out", str(tmp_path / "o.json")])
        assert exc.value.code != 0

    def test_a_data_fault_returns_one_rather_than_a_traceback(self, tmp_path: Path) -> None:
        rc = architecture_producer.main(
            [
                "run-shard",
                "--manifest",
                str(tmp_path / "absent.json"),
                "--stem-i-nt-threshold",
                "60",
                "--min-named-helices",
                "2",
                "--min-helix-pairs",
                "1",
                "--bulge-min-nt",
                "5",
                "--bulge-max-nt",
                "9",
                "--ncca-pairing-nt",
                "4",
                "--out",
                str(tmp_path / "o.json"),
            ]
        )
        assert rc == 1

    def test_derive_supply_is_reachable(self) -> None:
        assert architecture_producer.main(["derive-supply"]) == 0


class TestRemineCliCarriesTheTwoWayPair:
    def test_the_default_comes_from_the_module_constant(self) -> None:
        args = remine.build_parser().parse_args(
            ["plan", "--stage2-threshold", "0.9", "--out", "/dev/null"]
        )
        assert args.relaxed_arch_available is mine_round.RELAXED_ARCH_SUPPLY_AVAILABLE

    def test_the_negative_half_exists_and_reaches_False(self) -> None:
        """A bare store_true could not express a True default at all — the flip would be
        unreachable from the CLI the round actually invokes."""
        args = remine.build_parser().parse_args(
            [
                "plan",
                "--stage2-threshold",
                "0.9",
                "--out",
                "/dev/null",
                "--no-relaxed-arch-available",
            ]
        )
        assert args.relaxed_arch_available is False

    def test_both_halves_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            remine.build_parser().parse_args(
                [
                    "plan",
                    "--stage2-threshold",
                    "0.9",
                    "--out",
                    "/dev/null",
                    "--relaxed-arch-available",
                    "--no-relaxed-arch-available",
                ]
            )

    def test_an_unevidenced_b_declaration_is_refused_with_exit_4(self) -> None:
        """Declaring a supply this checkout cannot evidence is the fail-OPEN direction."""
        assert (
            remine.supply_declaration_unevidenced(declared=True, derivation={"available": False})
            is True
        )
        assert (
            remine.supply_declaration_unevidenced(declared=False, derivation={"available": False})
            is False
        )

    def test_the_b_supply_is_named_in_the_preflight_loop(self) -> None:
        """Adding the derivation without wiring it into `_refuse_unevidenced` would leave the
        exit-4 gate silent about (b)."""
        import inspect

        source = inspect.getsource(remine._refuse_unevidenced)
        assert "relaxed_arch_available" in source
        assert "relaxed_arch_supply_derivation" in source

    def test_a_derivation_the_collector_never_produced_refuses_rather_than_raising(self) -> None:
        """Adding an entry to the preflight loop without adding it to `_supply_derivations`
        used to raise KeyError out of a function documented as returning an exit code —
        which bypasses `_cmd_plan`'s report-writing path. It must fail CLOSED instead."""
        import argparse

        args = argparse.Namespace(
            msa_supply_available=True,
            stage2_supply_available=True,
            synteny_available=True,
            relaxed_arch_available=True,
        )
        assert remine._refuse_unevidenced(args, {}) == 4

    def test_positive_control_a_complete_derivation_set_does_not_refuse(self) -> None:
        import argparse

        args = argparse.Namespace(
            msa_supply_available=True,
            stage2_supply_available=True,
            synteny_available=True,
            relaxed_arch_available=True,
        )
        derivations = {
            k: {"available": True}
            for k in (
                "msa_supply_derivation",
                "stage2_supply_derivation",
                "synteny_supply_derivation",
                "relaxed_arch_supply_derivation",
            )
        }
        assert remine._refuse_unevidenced(args, derivations) is None

    def test_the_derivation_is_produced_by_the_supply_collector(self) -> None:
        import argparse

        derivations = remine._supply_derivations(argparse.Namespace())
        assert "relaxed_arch_supply_derivation" in derivations
        assert set(derivations["relaxed_arch_supply_derivation"]) >= {"available", "clauses"}


class TestTheCommittedFreeze:
    """Only the things a committed report genuinely owns — never a relation."""

    @pytest.fixture(scope="class")
    def freeze(self) -> dict:
        return json.loads(
            (REPO_ROOT / architecture_producer.FREEZE_REPORT).read_text(encoding="utf-8")
        )

    def test_it_measured_a_nonempty_held_out_carve(self, freeze: dict) -> None:
        assert freeze["carve"]["n_records"] > 0
        assert freeze["carve"]["n_with_discriminator_in_a_flanked_bulge"] > 0

    def test_it_joined_by_content_hash_not_position(self, freeze: dict) -> None:
        assert "record_hash" in freeze["carve"]["join"]

    def test_it_pins_nothing(self, freeze: dict) -> None:
        assert freeze["pins_nothing"] is True
        assert "NO value is frozen" in freeze["stem_i_extent_nt"]["note"]

    def test_it_discloses_that_the_freezing_corpus_is_CM_derived(self, freeze: dict) -> None:
        assert "CM-derived" in freeze["disclosure"]

    def test_the_recovery_shares_reconcile_against_their_own_counts(self, freeze: dict) -> None:
        """Every published rate is checked against the counts beside it — four numbers have
        drifted in this project from hand-copying a regenerated value."""
        denominator = freeze["carve"]["n_with_discriminator_in_a_flanked_bulge"]
        for k, arm in freeze["ncca_recovery"].items():
            assert arm["share"] == pytest.approx(arm["n"] / denominator, abs=1e-6), k
            assert 0 <= arm["n"] <= denominator, k

    def test_the_motif_in_the_report_is_the_one_the_code_derives(self, freeze: dict) -> None:
        assert freeze["acceptor_motif"] == architecture.acceptor_pairing_motif()


class TestFreezeIsComputedNotJustCommitted:
    """The relation the committed report cannot testify to: that the freeze MEASURES."""

    def test_an_empty_heldout_carve_refuses_rather_than_reporting_zero(
        self, tmp_path: Path
    ) -> None:
        import pandas as pd

        corpus = tmp_path / "c.parquet"
        splits = tmp_path / "s.parquet"
        pd.DataFrame({"Sequence": ["ACGU"], "Structure": ["(())"]}).to_parquet(corpus)
        pd.DataFrame(
            {"corpus_record_sha256": ["x"], "source": ["corpus"], "nested_role": ["train"]}
        ).to_parquet(splits)
        with pytest.raises(architecture_producer.ProducerError, match="held-out canonical set"):
            architecture_producer.heldout_canonical(corpus=corpus, split_table=splits)

    def test_a_carve_that_matches_nothing_refuses_rather_than_measuring_zero(
        self, tmp_path: Path
    ) -> None:
        """A silent empty join would publish a freeze over 0 records that still looks green."""
        import pandas as pd

        corpus = tmp_path / "c.parquet"
        splits = tmp_path / "s.parquet"
        pd.DataFrame({"Sequence": ["ACGU"], "Structure": ["(())"]}).to_parquet(corpus)
        pd.DataFrame(
            {
                "corpus_record_sha256": ["deadbeef"],
                "source": ["corpus"],
                "nested_role": ["heldout"],
            }
        ).to_parquet(splits)
        with pytest.raises(architecture_producer.ProducerError, match="do not describe the same"):
            architecture_producer.heldout_canonical(corpus=corpus, split_table=splits)

    def test_positive_control_a_real_hash_join_carves(self, tmp_path: Path) -> None:
        import pandas as pd

        from tbox_finder.ingest import record_hash

        frame = pd.DataFrame({"Sequence": ["ACGU"], "Structure": ["(())"]})
        corpus = tmp_path / "c.parquet"
        splits = tmp_path / "s.parquet"
        frame.to_parquet(corpus)
        digest = record_hash(next(frame.itertuples(index=False, name=None)))
        pd.DataFrame(
            {"corpus_record_sha256": [digest], "source": ["corpus"], "nested_role": ["heldout"]}
        ).to_parquet(splits)
        assert len(architecture_producer.heldout_canonical(corpus=corpus, split_table=splits)) == 1


class TestTheP3RoundCarriesBToo:
    """The gap CLI round 2 found: `mine_round` had the (b) status table and `remine` did not.

    Two apply paths, one fix — the P3 round would have declared the backend available and
    still read `unavailable` for every candidate, reporting a clean zero yield
    indistinguishable from an honest one. Each assertion below names the specific link in
    that chain rather than only the end state, so a partial re-break is attributable.
    """

    def test_remine_evidence_carries_a_produced_b_status(self) -> None:
        evidence = remine.remine_candidate_evidence(
            "x",
            covariation_status=None,
            stage2_posteriors=None,
            relaxed_arch_status={"x": STATUS_PASSED},
        )
        assert evidence.relaxed_architecture == STATUS_PASSED

    def test_remine_evidence_keeps_the_fail_closed_absent_id_rule(self) -> None:
        evidence = remine.remine_candidate_evidence(
            "x",
            covariation_status=None,
            stage2_posteriors=None,
            relaxed_arch_status={"other": STATUS_PASSED},
        )
        assert evidence.relaxed_architecture == STATUS_UNAVAILABLE

    def test_the_b_status_survives_alongside_a_stage2_posterior(self) -> None:
        """`dataclasses.replace` for the posterior must not drop the (b) status."""
        evidence = remine.remine_candidate_evidence(
            "x",
            covariation_status={"x": STATUS_FAILED},
            stage2_posteriors={"x": 0.99},
            synteny_status={"x": STATUS_UNAVAILABLE},
            relaxed_arch_status={"x": STATUS_PASSED},
        )
        assert evidence.relaxed_architecture == STATUS_PASSED
        assert evidence.any_helix_rscape == STATUS_FAILED
        assert evidence.downstream_aaRS_synteny == STATUS_UNAVAILABLE
        assert evidence.stage2_posterior == pytest.approx(0.99)

    def test_the_apply_parser_exposes_the_b_status_table(self) -> None:
        args = remine.build_parser().parse_args(
            [
                "apply-spare-rule",
                "--stage2-threshold",
                "0.9",
                "--out",
                "o",
                "--manifest",
                "m",
                "--status-table",
                "s",
                "--posteriors",
                "p",
                "--probe-set",
                "pr",
                "--relaxed-arch-status",
                "t.json",
            ]
        )
        assert args.relaxed_arch_status == "t.json"

    def test_read_remine_manifest_stamps_the_b_status(self, tmp_path: Path) -> None:
        """End to end through the manifest reader, not just the evidence builder."""
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": "c1",
                            "accession": "GCA_1:c0",
                            "locus_start": 0,
                            "locus_end": 100,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        candidates = remine.read_remine_manifest(path, relaxed_arch_status={"c1": STATUS_PASSED})
        assert [c.evidence.relaxed_architecture for c in candidates] == [STATUS_PASSED]

    def test_a_malformed_b_table_is_a_reported_round_fault_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(remine.RemineError, match="relaxed-architecture status table"):
            remine._load_relaxed_arch_status(bad)

    def test_positive_control_a_well_formed_b_table_loads(self, tmp_path: Path) -> None:
        good = tmp_path / "good.json"
        good.write_text(
            json.dumps(
                architecture_producer.build_status_table([row("c1", STATUS_PASSED)], config=CONFIG)
            ),
            encoding="utf-8",
        )
        assert remine._load_relaxed_arch_status(good) == {"c1": "passed"}

    def test_no_table_at_all_is_None_not_an_error(self) -> None:
        assert remine._load_relaxed_arch_status(None) is None


class TestBothApplyPathsPreflightTheBDeclaration:
    """`remine` refused an unevidenced (b) declaration; `mine_round`'s apply path did not.

    Two apply paths must agree about what "declared" costs, or the weaker one is simply the
    way around the gate.
    """

    def test_mine_round_apply_refuses_an_unevidenced_b_declaration_with_exit_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            architecture_producer,
            "derive_relaxed_arch_supply_available",
            lambda **_: {"available": False, "reasons": ["staged nowhere"]},
        )
        rc = mine_round.main(
            [
                "apply-spare-rule",
                "--manifest",
                str(tmp_path / "m.json"),
                "--status-table",
                str(tmp_path / "s.json"),
                "--out",
                str(tmp_path / "o.json"),
                "--rscape-installed",
                "false",
                "--relaxed-arch-available",
                "--relaxed-arch-status",
                str(tmp_path / "t.json"),
            ]
        )
        assert rc == 4

    def test_positive_control_not_declaring_it_does_not_hit_the_b_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guard that refuses on EVERYTHING also satisfies an exit-4 assertion."""
        monkeypatch.setattr(
            architecture_producer,
            "derive_relaxed_arch_supply_available",
            lambda **_: {"available": False, "reasons": ["staged nowhere"]},
        )
        # Getting as far as the missing manifest PROVES the (b) gate was not hit: the gate
        # returns 4 before any file is opened. Asserting `rc != 4` alone would also pass if
        # the command failed for an unrelated reason before reaching either.
        with pytest.raises(FileNotFoundError):
            mine_round.main(
                [
                    "apply-spare-rule",
                    "--manifest",
                    str(tmp_path / "m.json"),
                    "--status-table",
                    str(tmp_path / "s.json"),
                    "--out",
                    str(tmp_path / "o.json"),
                    "--rscape-installed",
                    "false",
                    "--no-msa-supply-available",
                ]
            )
