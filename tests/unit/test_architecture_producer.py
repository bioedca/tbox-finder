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

    def test_an_unevidenced_b_declaration_is_detected_by_the_predicate(self) -> None:
        """Declaring a supply this checkout cannot evidence is the fail-OPEN direction.

        ⚠ Named for the PREDICATE, not for exit 4: this body exercises only
        ``supply_declaration_unevidenced``. The exit code is asserted by
        ``test_a_derivation_the_collector_never_produced_refuses_rather_than_raising``.
        A failure here would otherwise point an operator at the exit-4 gate instead of at
        the predicate that feeds it.
        """
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
        # ⚠ Without this the loop body never runs on an empty mapping and the test passes
        # having checked nothing — the vacuous-assertion species this file names elsewhere,
        # in this file's own test. Assert the arm set exists before iterating it.
        assert freeze["ncca_recovery"], "no recovery arms to reconcile"
        assert denominator > 0, "an empty denominator makes every share vacuously checkable"
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


class TestRoundTwoProducerGuards:
    """CodeRabbit r2's producer-side findings, one test per finding."""

    def test_an_inverted_bulge_range_refuses_at_construction(self) -> None:
        """An empty range means NO bulge satisfies it, so every candidate resolves to
        `failed` ⇒ MINED — the fail-open direction, reachable because nothing is defaulted."""
        with pytest.raises(architecture_producer.ProducerError, match="bulge range is empty"):
            architecture_producer.ArchitectureRunConfig(
                stem_i_nt_threshold=60,
                min_named_helices=2,
                min_helix_pairs=1,
                bulge_min_nt=9,
                bulge_max_nt=5,
                ncca_pairing_nt=4,
            )

    @pytest.mark.parametrize(
        "field", ["min_named_helices", "min_helix_pairs", "ncca_pairing_nt", "bulge_min_nt"]
    )
    def test_a_non_positive_count_refuses(self, field: str) -> None:
        kwargs = {
            "stem_i_nt_threshold": 60,
            "min_named_helices": 2,
            "min_helix_pairs": 1,
            "bulge_min_nt": 5,
            "bulge_max_nt": 9,
            "ncca_pairing_nt": 4,
            field: 0,
        }
        with pytest.raises(architecture_producer.ProducerError, match=">= 1"):
            architecture_producer.ArchitectureRunConfig(**kwargs)

    def test_positive_control_a_well_formed_config_constructs(self) -> None:
        assert architecture_producer.ArchitectureRunConfig(
            stem_i_nt_threshold=60,
            min_named_helices=2,
            min_helix_pairs=1,
            bulge_min_nt=5,
            bulge_max_nt=9,
            ncca_pairing_nt=4,
        ).bulge_size_range == (5, 9)

    def test_an_unreadable_consensus_loses_ONE_candidate_not_the_shard(
        self, tmp_path: Path
    ) -> None:
        """OSError / UnicodeDecodeError from the open must not propagate: `main` would
        return 1 and the whole shard would produce no table at all."""
        from tbox_finder.mining.covariation_producer import candidate_slug

        target = tmp_path / candidate_slug("c1") / "msa.sto"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"# STOCKHOLM 1.0\n\xff\xfe not utf-8\n")
        result = architecture_producer.evaluate_candidate(
            {"candidate_id": "c1"}, msa_dir=tmp_path, config=CONFIG
        )
        assert result["status"] == STATUS_UNAVAILABLE
        assert "unusable" in result["reason"]

    def test_a_shard_of_two_survives_one_unreadable_file(self, tmp_path: Path) -> None:
        """The point of the widened except: the OTHER candidate must still be scored."""
        from tbox_finder.mining.covariation_producer import candidate_slug

        bad = tmp_path / candidate_slug("bad") / "msa.sto"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"\xff\xfe")
        write_msa(tmp_path, "good", ANTITERM_SEQ, ANTITERM_SS)
        rows = architecture_producer.run_shard(
            [{"candidate_id": "bad"}, {"candidate_id": "good"}], msa_dir=tmp_path, config=CONFIG
        )
        assert [r["status"] for r in rows] == [STATUS_UNAVAILABLE, STATUS_PASSED]

    def test_the_msa_filename_is_imported_from_the_producer_that_writes_it(self) -> None:
        """A second literal would let a rename in the (a) producer make every (b) candidate
        silently `unavailable` while the shard tables still merged cleanly."""
        from tbox_finder.mining import covariation_producer

        assert architecture_producer.MSA_FILENAME is covariation_producer.MSA_FILENAME

    def test_a_shard_table_without_a_config_refuses(self, tmp_path: Path) -> None:
        """Coerced to {}, every shard compares EQUAL and the merge publishes an empty
        config into provenance over rows scored under real parameters."""
        path = tmp_path / "a.json"
        path.write_text(
            json.dumps({"rows": [row("a", STATUS_PASSED)], "status": {"a": "passed"}}),
            encoding="utf-8",
        )
        with pytest.raises(architecture_producer.ProducerError, match="'config' is missing"):
            architecture_producer.merge_status_tables([path])

    def test_a_non_mapping_row_refuses_rather_than_raising_AttributeError(
        self, tmp_path: Path
    ) -> None:
        """`main` does not catch AttributeError, so the merge leg would exit with a
        traceback instead of FATAL/exit 1."""
        path = tmp_path / "a.json"
        path.write_text(
            json.dumps({"config": CONFIG.as_dict(), "rows": ["not-an-object"]}), encoding="utf-8"
        )
        with pytest.raises(architecture_producer.ProducerError, match="not an object"):
            architecture_producer.merge_status_tables([path])

    def test_a_row_without_a_candidate_id_names_ITS_OWN_fault(self, tmp_path: Path) -> None:
        """Read via .get(..., ''), two such rows reported 'appears in more than one shard
        table' — the wrong fault entirely."""
        path = tmp_path / "a.json"
        path.write_text(
            json.dumps({"config": CONFIG.as_dict(), "rows": [{"status": "passed"}]}),
            encoding="utf-8",
        )
        with pytest.raises(architecture_producer.ProducerError, match="no usable candidate_id"):
            architecture_producer.merge_status_tables([path])

    def test_a_row_without_a_status_refuses(self, tmp_path: Path) -> None:
        path = tmp_path / "a.json"
        path.write_text(
            json.dumps({"config": CONFIG.as_dict(), "rows": [{"candidate_id": "a"}]}),
            encoding="utf-8",
        )
        with pytest.raises(architecture_producer.ProducerError, match="no status"):
            architecture_producer.merge_status_tables([path])

    def test_a_value_mismatch_names_the_CONFLICTING_VALUES_not_only_keys(
        self, tmp_path: Path
    ) -> None:
        """On same-key/different-value drift both key-set lists are empty, so the operator
        would read '(declared-only [], rows-only [])' — which names no fault."""
        table = architecture_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG)
        table["status"]["a"] = STATUS_FAILED
        with pytest.raises(architecture_producer.ProducerError) as exc:
            architecture_producer.validate_status_payload(table, label="t")
        message = str(exc.value)
        assert "conflicting" in message
        assert "status says 'failed' but its row says 'passed'" in message


class TestRoundThreeGuards:
    """CodeRabbit r3. The first is the best finding of the review: a clause that could
    not fail, handing `all(clauses.values())` a hardcoded True."""

    def test_the_wiring_clause_is_BREAKABLE_not_vacuous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The clause it replaced imported the module already executing, so neither the
        import nor the hasattr could fail. This one measures the round's actual wiring, so
        breaking the wiring must flip it."""
        monkeypatch.setattr(
            mine_round,
            "candidate_evidence",
            lambda *a, **k: mine_round.SpareRuleEvidence(),  # never stamps (b)
        )
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=REPO_ROOT)
        assert derivation["clauses"]["producer_status_wired"] is False
        assert derivation["available"] is False
        assert any("producer_status_wired" in r for r in derivation["reasons"])

    def test_positive_control_the_real_wiring_satisfies_it(self) -> None:
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=REPO_ROOT)
        assert derivation["clauses"]["producer_status_wired"] is True

    def test_the_vacuous_clause_is_gone(self) -> None:
        """A clause that cannot fail must not be re-added: it inflates the conjunction."""
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=REPO_ROOT)
        assert "producer_entry_points_present" not in derivation["clauses"]
        assert not hasattr(architecture_producer, "PRODUCER_ENTRY_POINTS")

    def test_a_bool_count_cannot_disagree_with_its_own_clause(self, tmp_path: Path) -> None:
        """`isinstance(True, int)` is True, so `true` in the report made the clause read
        False while the reported count read `true` — two fields disagreeing in one payload."""
        for _, rel in architecture_producer.SUPPLY_CLAUSES:
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        (tmp_path / architecture_producer.FREEZE_REPORT).write_text(
            json.dumps({"carve": {"n_with_discriminator_in_a_flanked_bulge": True}}),
            encoding="utf-8",
        )
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=tmp_path)
        assert derivation["clauses"]["freeze_measured_a_nonempty_carve"] is False
        assert derivation["n_heldout_canonical_measured"] == 0

    def test_a_non_mapping_row_refuses_in_validate_too_not_only_in_merge(self) -> None:
        """`load_status_map` reads tables written by other processes, so the row shape is
        an input. AttributeError here escapes `main` as a traceback, not FATAL/exit 1."""
        payload = {"status": {"a": "passed"}, "rows": ["not-an-object"]}
        with pytest.raises(architecture_producer.ProducerError, match="are not objects"):
            architecture_producer.validate_status_payload(payload, label="t")

    def test_positive_control_object_rows_validate(self) -> None:
        payload = {"status": {"a": "passed"}, "rows": [row("a", STATUS_PASSED)]}
        assert architecture_producer.validate_status_payload(payload, label="t") == {"a": "passed"}


class TestRoundFourGuards:
    """CodeRabbit r4 — four minors, no majors for the first time this step."""

    def test_the_freeze_provenance_clause_is_breakable(self, tmp_path: Path) -> None:
        """The freeze is EVIDENCE, so how it was produced is part of the claim.
        `build_provenance` permits an empty `inputs` and a null env lock."""
        for _, rel in architecture_producer.SUPPLY_CLAUSES:
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        (tmp_path / architecture_producer.FREEZE_REPORT).write_text(
            json.dumps(
                {
                    "carve": {"n_with_discriminator_in_a_flanked_bulge": 8605},
                    "provenance": {"inputs": {}, "env_lock_hash": None, "git_sha": "abc"},
                }
            ),
            encoding="utf-8",
        )
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=tmp_path)
        assert derivation["clauses"]["freeze_provenance_names_its_inputs"] is False

    def test_sha256d_external_inputs_satisfy_the_clause(self, tmp_path: Path) -> None:
        """The corpus is DVC-tracked and absent from a worktree, so it is recorded as a
        hashed external input — MORE traceable than a path, and it must count."""
        for _, rel in architecture_producer.SUPPLY_CLAUSES:
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        (tmp_path / architecture_producer.FREEZE_REPORT).write_text(
            json.dumps(
                {
                    "carve": {"n_with_discriminator_in_a_flanked_bulge": 8605},
                    "provenance": {
                        "inputs": {},
                        "env_lock_hash": "deadbeef",
                        "git_sha": "abc",
                        "extra": {"external_inputs": [{"name": "x.parquet", "sha256": "aa"}]},
                    },
                }
            ),
            encoding="utf-8",
        )
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=tmp_path)
        assert derivation["clauses"]["freeze_provenance_names_its_inputs"] is True

    def test_the_PRODUCER_stamps_an_env_lock_hash(self, tmp_path: Path) -> None:
        """⚠ The first version of this read the COMMITTED report and stayed green under
        sabotage — the bytes on disk were written by the fixed code. An assertion over an
        artifact cannot see the producer that wrote it. Call the writer instead."""
        probe = tmp_path / "probe.txt"
        probe.write_text("x", encoding="utf-8")
        block = architecture_producer._provenance("freeze", [probe], {"pins_nothing": True})
        assert block["env_lock_hash"], "the provenance block names no environment"
        assert block["rule"] == "architecture_producer::freeze"

    def test_the_committed_freeze_carries_what_the_producer_stamps(self) -> None:
        """The artifact-side half: cheap, and it catches a report regenerated by older code."""
        freeze = json.loads(
            (REPO_ROOT / architecture_producer.FREEZE_REPORT).read_text(encoding="utf-8")
        )
        assert freeze["provenance"]["env_lock_hash"]
        assert freeze["provenance"]["extra"]["external_inputs"]

    def test_a_broken_round_import_is_a_FAILED_clause_not_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`derive-supply` follows the FATAL / exit 1 contract; a module-level failure in
        `mine_round` must not abort the leg with a traceback."""

        def boom(*_a, **_k):
            raise RuntimeError("mine_round is broken")

        monkeypatch.setattr(mine_round, "candidate_evidence", boom)
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=REPO_ROOT)
        assert derivation["clauses"]["producer_status_wired"] is False
        assert derivation["available"] is False


class TestAppReviewGuards:
    """Findings the GitHub app raised that the CLI did not — the reason both paths run."""

    def _table(self, tmp_path: Path, name: str, payload: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_a_shard_with_ZERO_ROWS_refuses_rather_than_merging_invisibly(
        self, tmp_path: Path
    ) -> None:
        """The invisible loss: a dropped shard merges cleanly, its candidates are simply
        absent, and each absent candidate reads `unavailable` ⇒ SPARED. Every count still
        reconciles, so nothing in the report looks wrong."""
        good = self._table(
            tmp_path,
            "a.json",
            architecture_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG),
        )
        empty = self._table(
            tmp_path, "b.json", {"config": CONFIG.as_dict(), "rows": [], "status": {}}
        )
        with pytest.raises(architecture_producer.ProducerError, match="carries no rows"):
            architecture_producer.merge_status_tables([good, empty])

    def test_positive_control_two_non_empty_shards_merge(self, tmp_path: Path) -> None:
        a = self._table(
            tmp_path,
            "a.json",
            architecture_producer.build_status_table([row("a", STATUS_PASSED)], config=CONFIG),
        )
        b = self._table(
            tmp_path,
            "b.json",
            architecture_producer.build_status_table([row("b", STATUS_FAILED)], config=CONFIG),
        )
        assert architecture_producer.merge_status_tables([a, b])["n_candidates"] == 2

    @pytest.mark.parametrize("key", architecture_producer.REQUIRED_CONFIG_KEYS)
    def test_a_config_missing_ONE_rule_key_refuses(self, tmp_path: Path, key: str) -> None:
        """A non-empty but incomplete config would publish partial rule parameters over
        rows scored under real ones. Each key broken ALONE."""
        cfg = {k: v for k, v in CONFIG.as_dict().items() if k != key}
        path = self._table(tmp_path, "a.json", {"config": cfg, "rows": [row("a", STATUS_PASSED)]})
        with pytest.raises(architecture_producer.ProducerError, match=key):
            architecture_producer.merge_status_tables([path])

    def test_the_required_config_keys_are_all_really_produced(self) -> None:
        """A required-key list naming something the producer never emits would refuse every
        real shard; one naming nothing would be vacuous."""
        emitted = set(CONFIG.as_dict())
        assert set(architecture_producer.REQUIRED_CONFIG_KEYS) <= emitted
        assert architecture_producer.REQUIRED_CONFIG_KEYS


class TestRoundFiveGuards:
    """CodeRabbit r5 — two minors, both about a shape being an INPUT rather than an
    invariant."""

    def test_a_scalar_provenance_inputs_fails_closed_not_TypeError(self, tmp_path: Path) -> None:
        """`len()` on a number raises TypeError, which `main` does not catch — the leg
        would exit with a traceback instead of the FATAL / exit 1 contract."""
        for _, rel in architecture_producer.SUPPLY_CLAUSES:
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        (tmp_path / architecture_producer.FREEZE_REPORT).write_text(
            json.dumps(
                {
                    "carve": {"n_with_discriminator_in_a_flanked_bulge": 8605},
                    "provenance": {"inputs": 5, "env_lock_hash": "aa", "git_sha": "bb"},
                }
            ),
            encoding="utf-8",
        )
        derivation = architecture_producer.derive_relaxed_arch_supply_available(repo_root=tmp_path)
        assert derivation["clauses"]["freeze_provenance_names_its_inputs"] is False
        assert derivation["available"] is False

    def test_a_numpy_integer_stem1_length_is_counted(self) -> None:
        """On an integer-dtype column `itertuples` yields np.int64, which is not a Python
        `int` — the narrower check silently dropped every value."""
        np = pytest.importorskip("numpy")
        from numbers import Real

        value = np.int64(96)
        assert not isinstance(value, int)
        assert isinstance(value, Real) and not isinstance(value, bool)

    def test_a_bool_is_still_excluded(self) -> None:
        from numbers import Real

        assert isinstance(True, Real)  # which is exactly why bool must be excluded
