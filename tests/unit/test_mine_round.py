"""P2-10e — unit tests for the mining round driver (readiness, FP adapter, Tier-2N gate).

The scan legs are torch and untested here; the pieces below are pure and RUN-relevant:
the honest MSA-producibility-gated readiness (the ADR-0006 A2 scope-guard correction), the
window→genome coordinate adapter (identity), and the per-round decision + degenerate guard.
"""

from __future__ import annotations

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


def test_msa_supply_default_is_unavailable_at_p2():
    # The recorded RUN-blocker state: the per-candidate MSA supply is not built yet.
    assert mr.MSA_SUPPLY_AVAILABLE is False
    plan = mr.plan_round(rscape_installed=True)  # msa default = MSA_SUPPLY_AVAILABLE
    assert plan["ready"] is False


def test_plan_round_default_resolves_msa_flag_at_call_time(monkeypatch):
    # The default is a None sentinel resolved at call time (not bound at import), so flipping
    # MSA_SUPPLY_AVAILABLE at the unblock step reaches direct plan_round callers, not only the CLI.
    assert mr.plan_round(rscape_installed=True)["ready"] is False
    monkeypatch.setattr(mr, "MSA_SUPPLY_AVAILABLE", True)
    assert mr.plan_round(rscape_installed=True)["ready"] is True


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
    # Absent --msa-supply-available, the CLI default is MSA_SUPPLY_AVAILABLE, so flipping that
    # one module flag at the unblock step unblocks the preflight WITHOUT a CLI change. If the
    # CLI hardcoded False, the documented unblock switch would be a dead letter.
    assert mr.main(["plan", "--rscape-installed", "true"]) == 3  # refused at P2 (flag False)
    monkeypatch.setattr(mr, "MSA_SUPPLY_AVAILABLE", True)
    assert mr.main(["plan", "--rscape-installed", "true"]) == 0  # ready now, no CLI change
    capsys.readouterr()


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
def test_msa_supply_flag_stays_false_this_step():
    # The producer + wiring land now; MSA_SUPPLY_AVAILABLE is flipped only at the A10 Phase-2 pin,
    # after the measurement. If this reads True, a round is silently unblocked without the budget.
    assert mr.MSA_SUPPLY_AVAILABLE is False


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
