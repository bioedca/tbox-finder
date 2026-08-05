"""P2-10e — unit tests for the mining round driver (readiness, FP adapter, Tier-2N gate).

The scan legs are torch and untested here; the pieces below are pure and RUN-relevant:
the honest MSA-producibility-gated readiness (the ADR-0006 A2 scope-guard correction), the
window→genome coordinate adapter (identity), and the per-round decision + degenerate guard.
"""

from __future__ import annotations

import json

import pytest

from tbox_finder.eval.mining_criterion import ProvisionalCriterionError
from tbox_finder.eval.tier2n_probe import ROUND_CONTINUE, ROUND_HALT_ROLLBACK, ProbeSet
from tbox_finder.infer.call import Candidate
from tbox_finder.mining import mine_round as mr
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED, STATUS_UNAVAILABLE


def _cand(start: int, end: int, *, peak: float = 0.95) -> Candidate:
    return Candidate(
        start=start,
        end=end,
        length=end - start,
        peak_p_elem=peak,
        mean_p_elem=peak,
        n_zero_flanked=0,
        dominant_class=1,
    )


def _probe_set(n: int) -> ProbeSet:
    return ProbeSet(natural=(), synthetic=tuple(f"v{i:03d}" for i in range(n)))


# --------------------------------------------------------------------------- #
# build_round_availability — covariation availability is gated on MSA-producibility,
# NOT on R-scape being installed (ADR-0006 A2 scope-guard correction).
# --------------------------------------------------------------------------- #
def test_covariation_requires_both_rscape_and_msa_supply():
    both = mr.build_round_availability(rscape_installed=True, msa_supply_available=True)
    assert both["any_helix_rscape"] is True

    installed_no_msa = mr.build_round_availability(
        rscape_installed=True, msa_supply_available=False
    )
    assert installed_no_msa["any_helix_rscape"] is False  # the integrity gap this closes

    msa_no_rscape = mr.build_round_availability(rscape_installed=False, msa_supply_available=True)
    assert msa_no_rscape["any_helix_rscape"] is False


def test_availability_names_every_disjunct():
    avail = mr.build_round_availability(rscape_installed=True, msa_supply_available=True)
    assert set(avail) == {
        "relaxed_architecture",
        "any_helix_rscape",
        "downstream_aaRS_synteny",
    }


# --------------------------------------------------------------------------- #
# plan_round — refused at P2, ready only via a protective backend.
# --------------------------------------------------------------------------- #
def test_plan_refuses_at_p2_no_protective_backend():
    # R-scape installed but no MSA supply and no synteny ⇒ no protective backend ⇒ refused.
    plan = mr.plan_round(rscape_installed=True, msa_supply_available=False)
    assert plan["ready"] is False
    assert plan["readiness"]["refusal_reason"]


def test_plan_ready_when_covariation_is_msa_producible():
    plan = mr.plan_round(rscape_installed=True, msa_supply_available=True)
    assert plan["ready"] is True
    assert plan["readiness"]["refusal_reason"] is None


def test_plan_ready_via_synteny_backend():
    plan = mr.plan_round(rscape_installed=False, msa_supply_available=False, synteny_available=True)
    assert plan["ready"] is True


def test_plan_relaxed_architecture_alone_is_not_enough():
    # (b) is False on every Tier-2N locus (D9 row 5) ⇒ non-protective ⇒ still refused.
    plan = mr.plan_round(
        rscape_installed=False, msa_supply_available=False, relaxed_arch_available=True
    )
    assert plan["ready"] is False


def test_msa_supply_default_is_available_since_p3_15a():
    # P3-15′-a: the supply landed (target DB job 741 + certified producer job 766), so the
    # default plan is READY on the protective covariation-(a) backend alone.
    assert mr.MSA_SUPPLY_AVAILABLE is True
    plan = mr.plan_round(rscape_installed=True)  # msa default = MSA_SUPPLY_AVAILABLE
    assert plan["ready"] is True


def test_plan_round_default_resolves_msa_flag_at_call_time(monkeypatch):
    # The default is a None sentinel resolved at call time (not bound at import), so the flag's
    # value reaches direct plan_round callers, not only the CLI. Asserted in BOTH directions:
    # with the flag forced False the same call refuses, so this cannot pass on a hardcoded True.
    assert mr.plan_round(rscape_installed=True)["ready"] is True
    monkeypatch.setattr(mr, "MSA_SUPPLY_AVAILABLE", False)
    assert mr.plan_round(rscape_installed=True)["ready"] is False


# --------------------------------------------------------------------------- #
# parse_window_name + the window→genome coordinate adapter.
# --------------------------------------------------------------------------- #
def test_parse_window_name():
    assert mr.parse_window_name("GCA_000220375.1:c2:1000") == ("GCA_000220375.1", 2, 1000)


def test_parse_window_name_rejects_malformed():
    with pytest.raises(mr.MineRoundError):
        mr.parse_window_name("GCA_000220375.1")


def test_parse_window_name_rejects_negative_contig_or_start():
    with pytest.raises(mr.MineRoundError):
        mr.parse_window_name("GCA_1:c-1:5")
    with pytest.raises(mr.MineRoundError):
        mr.parse_window_name("GCA_1:c0:-5")


def test_window_candidates_to_mining_maps_genome_coordinates():
    mined = mr.window_candidates_to_mining(
        [_cand(10, 70, peak=0.97), _cand(200, 260)],
        window_name="GCA_000220375.1:c2:1000",
    )
    assert len(mined) == 2
    first = mined[0]
    # Contig-scoped accession (`<assembly>:c<ci>`), NOT the bare assembly accession.
    assert first.accession == "GCA_000220375.1:c2"
    assert first.pool == "genomic_window"
    assert first.locus_start == 1010  # window_start 1000 + candidate.start 10
    assert first.locus_end == 1070
    assert first.candidate_id == "GCA_000220375.1:c2:1000:1010-1070"
    assert first.score == pytest.approx(0.97)
    # FP candidates enter with the default all-unavailable evidence (the spare rule decides).
    assert first.evidence.unavailable() == (
        "relaxed_architecture",
        "any_helix_rscape",
        "downstream_aaRS_synteny",
    )


def test_same_offset_on_different_contigs_does_not_collapse():
    # Two contigs of one assembly, an identical window offset + candidate span: the mask keys
    # on (accession, locus_start, locus_end), so a bare-assembly accession would collapse
    # these two distinct loci. The contig-scoped accession keeps them separable.
    c0 = mr.window_candidates_to_mining([_cand(10, 70)], window_name="GCA_1:c0:1000")[0]
    c1 = mr.window_candidates_to_mining([_cand(10, 70)], window_name="GCA_1:c1:1000")[0]
    assert (c0.locus_start, c0.locus_end) == (c1.locus_start, c1.locus_end)  # same coords
    assert c0.accession != c1.accession  # …but different mask keys ⇒ no collapse
    assert (c0.accession, c1.accession) == ("GCA_1:c0", "GCA_1:c1")


# --------------------------------------------------------------------------- #
# evaluate_probe_round — round-0 degenerate guard + the halt/continue decision.
# --------------------------------------------------------------------------- #
def test_round_zero_all_recovered_raises_degenerate():
    probe = _probe_set(20)
    recovered = set(probe.synthetic)  # all 20 ⇒ recall 1.0 ⇒ degenerate on the baseline
    with pytest.raises(ProvisionalCriterionError):
        mr.evaluate_probe_round(probe, recovered, [], round_index=0)


def test_round_zero_non_degenerate_continues():
    probe = _probe_set(20)
    recovered = set(list(probe.synthetic)[:18])  # 0.9, intermediate
    out = mr.evaluate_probe_round(probe, recovered, [], round_index=0)
    assert out["recall_this_round"] == pytest.approx(0.9)
    assert out["decision"]["decision"] == ROUND_CONTINUE
    assert out["degenerate_guard"]["degenerate"] is False


def test_later_round_recall_drop_halts_and_rolls_back():
    probe = _probe_set(20)
    recovered = set(list(probe.synthetic)[:18])  # 0.9 vs a best of 1.0 ⇒ drop 0.10 ≥ 0.05
    out = mr.evaluate_probe_round(probe, recovered, [1.0], round_index=1)
    assert out["decision"]["decision"] == ROUND_HALT_ROLLBACK
    assert out["degenerate_guard"] is None  # guard only on the round-0 baseline


def test_later_round_no_drop_continues():
    probe = _probe_set(20)
    recovered = set(list(probe.synthetic)[:19])  # 0.95 vs best 0.85 ⇒ no drop
    out = mr.evaluate_probe_round(probe, recovered, [0.85], round_index=1)
    assert out["decision"]["decision"] == ROUND_CONTINUE


def test_later_round_does_not_run_degenerate_guard():
    # All recovered on a later round is not a degenerate-guard case (the guard is a round-0
    # baseline check); it must not raise.
    probe = _probe_set(20)
    out = mr.evaluate_probe_round(probe, set(probe.synthetic), [1.0], round_index=1)
    assert out["degenerate_guard"] is None
    assert out["decision"]["decision"] == ROUND_CONTINUE


# --------------------------------------------------------------------------- #
# CLI — the unblock contract + strict bool parsing.
# --------------------------------------------------------------------------- #
def test_cli_default_tracks_msa_supply_flag(monkeypatch, capsys):
    # Absent a declaration, the CLI default is MSA_SUPPLY_AVAILABLE, so the one module flag
    # drives the preflight WITHOUT a CLI change. If the CLI hardcoded either value, the
    # documented unblock switch would be a dead letter — asserted in both directions.
    assert mr.main(["plan", "--rscape-installed", "true"]) == 0  # ready (flag True)
    monkeypatch.setattr(mr, "MSA_SUPPLY_AVAILABLE", False)
    assert mr.main(["plan", "--rscape-installed", "true"]) == 3  # refused, no CLI change
    capsys.readouterr()


def test_cli_can_declare_the_supply_unavailable(capsys):
    # The flip made the store_true declaration one-way: with the default True, a machine that
    # has NOT staged the homolog DB had no way to say so and would discover the gap in the
    # producer array, after the GPU legs. --no-msa-supply-available is that way out, and the
    # positive control beside it is the default run, which must still be ready.
    assert mr.main(["plan", "--rscape-installed", "true"]) == 0
    assert mr.main(["plan", "--rscape-installed", "true", "--no-msa-supply-available"]) == 3
    out = capsys.readouterr()
    assert "REFUSED at readiness" in out.err


def test_cli_refuses_both_msa_declarations_at_once(capsys):
    # Mutually exclusive: a script cannot assert available AND unavailable and silently get
    # whichever argparse saw last.
    with pytest.raises(SystemExit):
        mr.main(
            [
                "plan",
                "--rscape-installed",
                "true",
                "--msa-supply-available",
                "--no-msa-supply-available",
            ]
        )
    capsys.readouterr()


def test_cli_plan_is_fatal_when_the_declaration_is_unevidenced(monkeypatch, capsys, tmp_path):
    # Declared available + no evidence is a STAGING FAULT, and its exit code must differ from
    # the clean readiness refusal (3): mine_round.sbatch converts 3 into `exit 0`, so routing
    # this through 3 would turn a misconfigured checkout into a silent "no round today".
    monkeypatch.setattr(mr, "REPO_ROOT", tmp_path)  # an empty tree evidences nothing
    rc = mr.main(["plan", "--rscape-installed", "true"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "cannot evidence it" in err
    assert "target_db_versioned" in err  # the refusal names the clause that failed


def test_cli_plan_unevidenced_is_not_fatal_when_nothing_is_declared(monkeypatch, capsys, tmp_path):
    # The asymmetry: declaring the supply UNAVAILABLE on an unevidenced checkout is the
    # conservative direction and must stay a clean refusal (3), not the staging fault (4).
    monkeypatch.setattr(mr, "REPO_ROOT", tmp_path)
    assert mr.main(["plan", "--rscape-installed", "true", "--no-msa-supply-available"]) == 3
    capsys.readouterr()


def test_cli_plan_staging_fault_outranks_a_readiness_refusal(monkeypatch, capsys, tmp_path):
    # ORDER, not just the code: with R-scape absent the round is ALSO refused at readiness (3),
    # which mine_round.sbatch converts to `exit 0`. A checkout that declares a supply it cannot
    # evidence must not be able to leave through that door, so the evidence check runs FIRST.
    monkeypatch.setattr(mr, "REPO_ROOT", tmp_path)
    assert mr.main(["plan", "--rscape-installed", "false"]) == 4
    capsys.readouterr()


def test_cli_apply_spare_rule_refuses_before_reading_anything(monkeypatch, capsys, tmp_path):
    # The gate at the leg that actually MINES. It runs on a different job from `plan`, so it
    # cannot inherit the preflight's verdict. Paths that do not exist prove it refuses BEFORE
    # touching the manifest: a gate placed after the read would raise, not return 4.
    monkeypatch.setattr(mr, "REPO_ROOT", tmp_path)
    rc = mr.main(
        [
            "apply-spare-rule",
            "--manifest",
            str(tmp_path / "nope.json"),
            "--status-table",
            str(tmp_path / "nope_status.json"),
            "--out",
            str(tmp_path / "out.json"),
            "--rscape-installed",
            "true",
        ]
    )
    assert rc == 4
    assert "cannot evidence" in capsys.readouterr().err


def test_cli_rejects_invalid_rscape_override(capsys):
    # A typo must be an error, not a silent refuse. argparse exits (SystemExit 2) on a bad type.
    with pytest.raises(SystemExit):
        mr.main(["plan", "--rscape-installed", "treu"])
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# A10 producer wiring — window_candidates_to_mining stamps the produced covariation
# status onto any_helix_rscape; an absent id is fail-closed to 'unavailable' (⇒ spared),
# and the flag stays False until the Phase-2 pin flips it.
# --------------------------------------------------------------------------- #
def test_msa_supply_flag_agrees_with_the_derived_supply():
    # THE PIN (P3-15′-a, strengthened from a hard `is False`). The constant is a declaration;
    # this asserts it against an INDEPENDENT re-derivation from the shipped evidence, so it
    # cannot drift from reality in either direction: a supply that disappears fails here, and
    # so does a constant left behind after the supply lands. A bare `is True` would only have
    # caught the first.
    derived = mr.derive_msa_supply_available()
    assert derived["available"] is True, derived["reasons"]
    assert mr.MSA_SUPPLY_AVAILABLE is derived["available"]
    # All six clauses, named — so a future clause added to the conjunction cannot be satisfied
    # silently by the five that already hold ([[all-true-fixture-cannot-test-a-conjunction]]).
    assert derived["clauses"] == {
        "target_db_versioned": True,
        "producer_present": True,
        "producer_status_wired": True,
        "certification_green": True,
        "certification_matches_versioned_db": True,
        "certified_msa_intact": True,
    }
    assert derived["reasons"] == []


# --------------------------------------------------------------------------- #
# derive_msa_supply_available — the validator half of the constant. Every clause is
# broken ALONE (the all-TRUE real tree above is the positive control): a conjunction
# whose members are never individually falsified is satisfied by a hardcoded True.
# --------------------------------------------------------------------------- #
@pytest.fixture
def evidence_root(tmp_path):
    """A writable copy of the shipped supply evidence, so one clause can be broken alone."""
    for rel in (mr.HOMOLOG_DB_DVC, mr.HOMOLOG_MSA_PROVENANCE, mr.CERTIFIED_MSA):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((mr.REPO_ROOT / rel).read_bytes())
    return tmp_path


def _derive(root):
    return mr.derive_msa_supply_available(repo_root=root)


def _assert_only_failed(derived, clause):
    """`clause` is the ONLY False clause, and the verdict followed it."""
    assert derived["available"] is False
    assert derived["clauses"][clause] is False
    others = {k: v for k, v in derived["clauses"].items() if k != clause}
    assert all(others.values()), others
    assert any(r.startswith(f"{clause}:") for r in derived["reasons"]), derived["reasons"]


def test_evidence_root_fixture_is_the_all_true_positive_control(evidence_root):
    # Without this, every "broken" case below could be passing for the wrong reason (a
    # mis-copied fixture that never evidences anything).
    assert _derive(evidence_root)["available"] is True


def test_missing_evidence_fails_closed(tmp_path):
    derived = _derive(tmp_path)  # an empty tree: nothing readable
    assert derived["available"] is False
    assert derived["clauses"]["target_db_versioned"] is False
    assert derived["clauses"]["certification_green"] is False
    assert derived["clauses"]["certified_msa_intact"] is False


def test_clause_target_db_versioned_breaks_alone(evidence_root):
    # An EMPTY pointer (nfiles 0) rather than a deleted file: the md5 survives, so this
    # breaks the "DB is built" clause without also breaking the certification cross-check.
    pointer = evidence_root / mr.HOMOLOG_DB_DVC
    pointer.write_text(pointer.read_text().replace("nfiles: 10", "nfiles: 0"), encoding="utf-8")
    _assert_only_failed(_derive(evidence_root), "target_db_versioned")


def test_clause_target_db_versioned_rejects_a_non_dir_pointer(evidence_root):
    # A file pointer (no `.dir` suffix) is not a built directory DB — and the reader must
    # fail closed rather than accept the hash it can still see.
    pointer = evidence_root / mr.HOMOLOG_DB_DVC
    pointer.write_text(pointer.read_text().replace(".dir", ""), encoding="utf-8")
    assert mr.read_dvc_dir_pointer(pointer) is None
    assert _derive(evidence_root)["clauses"]["target_db_versioned"] is False


def test_read_dvc_dir_pointer_rejects_an_ambiguous_multi_output_file(evidence_root):
    # Two outputs in one pointer: there is no single DB version to cross-check the
    # certification against, so the reader must fail closed rather than pick the first.
    pointer = evidence_root / mr.HOMOLOG_DB_DVC
    pointer.write_text(pointer.read_text() * 2, encoding="utf-8")
    assert mr.read_dvc_dir_pointer(pointer) is None
    derived = _derive(evidence_root)
    assert derived["available"] is False
    assert derived["clauses"]["target_db_versioned"] is False


def test_clause_certification_matches_versioned_db_breaks_alone(evidence_root):
    # The DB rebuilt after certification: the pointer is fine, the certification is fine,
    # but the green was earned against a DIFFERENT database.
    path = evidence_root / mr.HOMOLOG_MSA_PROVENANCE
    doc = json.loads(path.read_text())
    key = next(k for k in doc["inputs"] if k.startswith("data/interim/homolog_db"))
    doc["inputs"][key] = "0" * 32 + ".dir"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    _assert_only_failed(_derive(evidence_root), "certification_matches_versioned_db")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda e: e.update(positive_status="failed"), id="positive_did_not_pass"),
        pytest.param(lambda e: e.update(shuffled_status="unavailable"), id="control_had_no_power"),
        pytest.param(
            lambda e: e["matched_control"].update(composition_matched=False), id="unmatched"
        ),
        # The vacuity case, and the reason the flags are named rather than iterated: a clause
        # read off the evidence's OWN key set is satisfied exactly when the evidence is gone —
        # an emptied matched_control has nothing left to be unmatched.
        pytest.param(lambda e: e.update(matched_control={}), id="matched_control_emptied"),
        pytest.param(
            lambda e: e["matched_control"].pop("ss_cons_matched"), id="one_dimension_dropped"
        ),
        pytest.param(lambda e: e.update(min_sequences_floor=5), id="floor_below_pin"),
        pytest.param(lambda e: e.update(n_records=3), id="depth_below_floor"),
    ],
)
def test_clause_certification_green_breaks_alone(evidence_root, mutate):
    # The must-fire matched control, re-derived. `control_had_no_power` is the one that
    # matters most: a shuffled twin reported `unavailable` means the detector had no power,
    # which reads identically to "no signal" and certifies NOTHING — it must not pass.
    path = evidence_root / mr.HOMOLOG_MSA_PROVENANCE
    doc = json.loads(path.read_text())
    mutate(doc["extra"])
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    _assert_only_failed(_derive(evidence_root), "certification_green")


def test_certification_floor_is_read_from_the_pinned_constant(evidence_root):
    # ADR-0006 A2 Pin 2 is MIN_REAL_HOMOLOG_N, imported — not a 20 re-typed here. Raising the
    # constant above the certified floor must break the clause, which a hardcoded 20 could not.
    from tbox_finder import power

    assert _derive(evidence_root)["clauses"]["certification_green"] is True
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(power, "MIN_REAL_HOMOLOG_N", 999)
        _assert_only_failed(_derive(evidence_root), "certification_green")


def test_clause_certified_msa_intact_breaks_alone(evidence_root):
    # The certified alignment must still be the bytes the certifying run hashed.
    msa = evidence_root / mr.CERTIFIED_MSA
    msa.write_bytes(msa.read_bytes() + b"\n")
    _assert_only_failed(_derive(evidence_root), "certified_msa_intact")


def test_clause_producer_present_breaks_alone(evidence_root, monkeypatch):
    from tbox_finder.mining import covariation_producer as cp

    monkeypatch.delattr(cp, "align_shard")  # the CM-free de-novo alignment leg
    derived = _derive(evidence_root)
    _assert_only_failed(derived, "producer_present")
    assert "align_shard" in " ".join(derived["reasons"])


def test_clause_producer_status_wired_breaks_alone(evidence_root, monkeypatch):
    # The claim the superseded provenance note made — a certified producer whose output the
    # round drops on the floor. Measured by CALLING the builder, not by reading a signature:
    # this fixture returns default-unavailable evidence exactly as the pre-producer code did.
    from tbox_finder.mining.spare_rule import SpareRuleEvidence

    monkeypatch.setattr(mr, "candidate_evidence", lambda cid, status: SpareRuleEvidence())
    _assert_only_failed(_derive(evidence_root), "producer_status_wired")


def test_plan_round_publishes_the_derivation_beside_the_claim(evidence_root):
    plan = mr.plan_round(rscape_installed=True, repo_root=evidence_root)
    assert plan["msa_supply_derivation"]["available"] is True
    assert plan["msa_supply_available"] is True
    assert mr.msa_supply_declaration_unevidenced(plan) is False


def test_declaration_unevidenced_is_asymmetric(tmp_path):
    # Declared available with no evidence = the fail-open direction, flagged.
    bad = mr.plan_round(rscape_installed=True, msa_supply_available=True, repo_root=tmp_path)
    assert mr.msa_supply_declaration_unevidenced(bad) is True
    # Declared UNavailable with no evidence = a conservative under-claim, not a fault.
    ok = mr.plan_round(rscape_installed=True, msa_supply_available=False, repo_root=tmp_path)
    assert mr.msa_supply_declaration_unevidenced(ok) is False


# --------------------------------------------------------------------------- #
# The two gates disagree, and that is the CORRECT state until (b)+(c) land.
# --------------------------------------------------------------------------- #
def test_ready_does_not_mean_minable_until_the_other_backends_land():
    # imp.md P3-15′-a's validation: readiness passes on covariation-(a) alone, while the P3
    # structural-yield gate still bounds the round at 0 mined because mining is a CONJUNCTION
    # and (b)+(c) have no backend. A future change that makes these two agree by accident —
    # in either direction — is a real regression, so both halves are asserted together.
    from tbox_finder.mining.remine import build_remine_availability, max_possible_yield

    plan = mr.plan_round(rscape_installed=True)
    assert plan["ready"] is True
    assert plan["availability"]["any_helix_rscape"] is True
    assert plan["availability"]["relaxed_architecture"] is False
    assert plan["availability"]["downstream_aaRS_synteny"] is False

    gate = max_possible_yield(
        build_remine_availability(
            rscape_installed=True,
            msa_supply_available=mr.MSA_SUPPLY_AVAILABLE,
            stage2_supply_available=False,
        ),
        stage2_threshold=0.9,
    )
    assert gate["yield_producible"] is False
    assert gate["max_mined"] == 0
    assert "relaxed_architecture" in gate["blocking_disjuncts"]
    assert "downstream_aaRS_synteny" in gate["blocking_disjuncts"]


def test_window_candidates_default_evidence_is_all_unavailable():
    # No status table (scan/collect legs) → the default all-'unavailable' evidence (spared).
    cands = mr.window_candidates_to_mining([_cand(10, 40)], window_name="GCA_1.1:c0:512")
    assert cands[0].evidence.any_helix_rscape == STATUS_UNAVAILABLE
    assert cands[0].candidate_id == "GCA_1.1:c0:512:522-552"


def test_window_candidates_apply_covariation_status_and_fail_closed_absent():
    win = "GCA_1.1:c0:512"
    present_id = f"{win}:522-552"
    status = {present_id: STATUS_FAILED}  # this candidate WAS scored (failed covariation)
    # Two candidates: one in the table (failed), one absent (must fail-closed to unavailable).
    cands = mr.window_candidates_to_mining(
        [_cand(10, 40), _cand(100, 130)], window_name=win, covariation_status=status
    )
    by_id = {c.candidate_id: c for c in cands}
    assert by_id[present_id].evidence.any_helix_rscape == STATUS_FAILED
    absent = by_id[f"{win}:612-642"]
    assert absent.evidence.any_helix_rscape == STATUS_UNAVAILABLE  # dropped shard cannot fail open


def test_candidate_evidence_passes_through_status():
    assert mr.candidate_evidence("x", None).any_helix_rscape == STATUS_UNAVAILABLE
    assert mr.candidate_evidence("x", {"x": STATUS_PASSED}).any_helix_rscape == STATUS_PASSED


def test_fp_manifest_roundtrip_and_status_stamp(tmp_path):
    win = "GCA_2.1:c1:0"
    cands = mr.window_candidates_to_mining([_cand(5, 35), _cand(50, 80)], window_name=win)
    path = tmp_path / "fp.json"
    mr.write_fp_manifest(cands, path)

    # The producer reads the FP manifest with its own reader (coords only; score/pool ignored).
    from tbox_finder.mining import covariation_producer as cp

    specs = cp.read_candidate_manifest(path)
    assert [s.candidate_id for s in specs] == [c.candidate_id for c in cands]
    assert specs[0].accession == "GCA_2.1:c1"

    # Retrain leg reloads with the merged status → stamps any_helix_rscape (absent ⇒ unavailable).
    first = cands[0].candidate_id
    reloaded = mr.read_fp_manifest(path, covariation_status={first: STATUS_PASSED})
    by_id = {c.candidate_id: c for c in reloaded}
    assert by_id[first].evidence.any_helix_rscape == STATUS_PASSED
    assert by_id[cands[1].candidate_id].evidence.any_helix_rscape == STATUS_UNAVAILABLE
    assert by_id[first].locus_start == 5 and by_id[first].locus_end == 35  # coords preserved


# --------------------------------------------------------------------------- #
# A10 Phase-1 scan-throughput probe — persistent, timeout-surviving instrumentation.
# The GPU scan itself is torch (untested); ScanThroughputLog + run_measured_scan are the
# pure accounting/persistence that turn a scan into win/s/GPU + FPs/window, so they carry
# the measurement's correctness and are the sabotage targets. A fake scan_window + an
# injected clock keep them deterministic and torch-free.
# --------------------------------------------------------------------------- #
class _Clock:
    """A deterministic monotonic clock: each call advances by ``dt`` (so ``wall_s`` > 0)."""

    def __init__(self, dt: float = 1.0) -> None:
        self._t = 0.0
        self._dt = dt

    def __call__(self) -> float:
        self._t += self._dt
        return self._t


def _fake_genome_windows(genomes):
    """``[(accession, n_windows), ...]`` → the ``(accession, [(window_name, seq), ...])`` stream."""
    for acc, n in genomes:
        yield acc, [(f"{acc}:c0:{i * 512}", "ACGT") for i in range(n)]


def _fp_scan_window(fp_map):
    """A fake ``scan_window`` emitting ``fp_map[window_name]`` placeholder FPs (0 if absent)."""

    def scan(window_name, seq):
        return [object() for _ in range(fp_map.get(window_name, 0))]

    return scan


def test_scan_throughput_log_math():
    log = mr.ScanThroughputLog()
    log.begin(100.0)
    log.begin_genome("GCA_1.1", 100.0)
    log.record_window(2)
    log.record_window(0)
    log.record_window(1)  # 3 windows, 3 FPs
    log.end_genome(110.0)  # genome wall = 10 s
    snap = log.snapshot(110.0, complete=True, note="completed")
    assert snap["windows_scanned"] == 3
    assert snap["candidates_found"] == 3
    assert snap["genomes_completed"] == 1
    assert snap["wall_s"] == 10.0
    assert snap["windows_per_s"] == 0.3  # 3 windows / 10 s — the headline win/s number
    assert snap["candidates_per_window"] == 1.0  # 3 FPs / 3 windows — the partial N₀ rate
    assert snap["per_genome"][0]["n_windows"] == 3
    assert snap["per_genome"][0]["windows_per_s"] == 0.3
    assert snap["complete"] is True and snap["note"] == "completed"


def test_scan_throughput_log_zero_guards():
    log = mr.ScanThroughputLog()
    snap = log.snapshot(0.0)  # nothing recorded → rates are None, never 0/0 or inf
    assert snap["windows_scanned"] == 0
    assert snap["wall_s"] == 0.0
    assert snap["windows_per_s"] is None
    assert snap["candidates_per_window"] is None
    log.begin(5.0)
    snap2 = log.snapshot(9.0)  # 4 s elapsed, still 0 windows
    assert snap2["wall_s"] == 4.0
    assert snap2["windows_per_s"] == 0.0  # 0 windows / 4 s = a real measured rate of 0
    assert snap2["candidates_per_window"] is None  # 0 windows → FP-rate undefined, not fabricated 0


def test_measured_scan_flushes_persistent_progress_before_interruption(tmp_path):
    out_path = tmp_path / "probe.json"
    calls = {"n": 0}

    def scan(window_name, seq):
        calls["n"] += 1
        if calls["n"] == 4:  # a stall→wall-kill lands on the 4th window
            raise RuntimeError("boom")
        return []

    with pytest.raises(RuntimeError):
        mr.run_measured_scan(
            _fake_genome_windows([("GCA_1.1", 10)]),
            scan,
            progress_out=out_path,
            flush_every_windows=2,
            now=_Clock(),
        )
    # The finally-flush left the last measurement on disk despite the mid-genome death.
    snap = json.loads(out_path.read_text())
    assert snap["complete"] is False
    assert snap["note"] == "interrupted"
    assert snap["windows_scanned"] == 3  # 3 succeeded before the 4th raised
    assert snap["in_progress_accession"] == "GCA_1.1"  # the offending genome is pinned


def test_measured_scan_window_cap_stops_and_marks_complete(tmp_path):
    out_path = tmp_path / "probe.json"
    out, _log = mr.run_measured_scan(
        _fake_genome_windows([("GCA_A", 5), ("GCA_B", 5)]),
        _fp_scan_window({}),
        progress_out=out_path,
        flush_every_windows=100,
        max_windows=3,
        now=_Clock(),
    )
    snap = json.loads(out_path.read_text())
    assert snap["windows_scanned"] == 3  # stopped exactly at the cap
    assert snap["complete"] is True  # an intentional stop, not an interruption
    assert snap["note"] == "window_cap"
    assert snap["genomes_completed"] == 1  # only GCA_A opened; GCA_B never scanned
    assert snap["per_genome"][0]["accession"] == "GCA_A"
    assert snap["per_genome"][0]["n_windows"] == 3  # honestly partial (not the 5-tile total)
    assert len(out) == 0


def test_measured_scan_counts_candidates_and_fp_rate(tmp_path):
    out_path = tmp_path / "probe.json"
    fp = {"GCA_A:c0:0": 2, "GCA_A:c0:512": 1}  # 3 FPs over 4 windows
    out, _log = mr.run_measured_scan(
        _fake_genome_windows([("GCA_A", 4)]),
        _fp_scan_window(fp),
        progress_out=out_path,
        flush_every_windows=100,
        now=_Clock(),
    )
    assert len(out) == 3
    snap = json.loads(out_path.read_text())
    assert snap["candidates_found"] == 3
    assert snap["windows_scanned"] == 4
    assert snap["candidates_per_window"] == 0.75  # 3 / 4
    assert snap["complete"] is True and snap["note"] == "completed"


def test_measured_scan_rejects_bad_bounds():
    with pytest.raises(mr.MineRoundError):
        mr.run_measured_scan(
            _fake_genome_windows([("A", 1)]), _fp_scan_window({}), flush_every_windows=0
        )
    with pytest.raises(mr.MineRoundError):
        mr.run_measured_scan(_fake_genome_windows([("A", 1)]), _fp_scan_window({}), max_windows=0)


def test_scan_progress_write_is_atomic(tmp_path):
    out_path = tmp_path / "probe.json"
    log = mr.ScanThroughputLog()
    log.begin(0.0)
    log.begin_genome("A", 0.0)
    log.record_window(1)
    log.end_genome(1.0)
    log.write(out_path, 1.0, complete=True, note="completed")
    assert out_path.exists()
    assert not (
        tmp_path / "probe.json.tmp"
    ).exists()  # the staging sibling was swapped in, not left
    json.loads(out_path.read_text())  # a full document (os.replace is atomic), never a half-write
