"""Unit tier — the P2-07 Tier-2N probe set, recall metric, and halt/rollback rule.

Guards the ADR-0005 D14 protection: a per-round recall drop on the Tier-2N probe
set must halt and roll back the mining iteration, so aggressive mining cannot
directionally train the production scanner to reject the flagship class.

Two specific failure modes are pinned:

* a **slow monotone bleed** — a few points of recall lost every round never trips
  a previous-round comparison, but destroys the class over an iteration; the
  baseline is therefore the *best* round so far;
* a **vacuous recall** — recall computed on an empty or sub-min-N probe set, which
  would report a number for a measurement that never happened.

Bare-CI tier: pure stdlib, no numpy/pandas/torch.
"""

from __future__ import annotations

import pytest

from tbox_finder.eval.tier2n_probe import (
    ROUND_CONTINUE,
    ROUND_HALT_ROLLBACK,
    ROUND_INADMISSIBLE,
    TIER2N_PROBE_MIN_N,
    TIER2N_RECALL_DROP_HALT,
    ProbeSet,
    Tier2NProbeError,
    build_probe_set,
    probe_recall,
    round_decision,
)
from tbox_finder.infernal import CmsearchHit, detection_map, write_fasta
from tbox_finder.infernal import build_report as cm_build_report
from tbox_finder.power import MIN_REAL_HOMOLOG_N
from tbox_finder.synth.tier2n import FAMILY_STEM_II_PK, Tier2NVariant


def _variant(
    vid: str,
    *,
    parent: bool | None = True,
    var: bool | None = False,
    ctl: bool | None = True,
) -> Tier2NVariant:
    return Tier2NVariant(
        variant_id=vid,
        parent_record_id=f"p_{vid}",
        family=FAMILY_STEM_II_PK,
        ablated_elements=("Stem_II",),
        sequence="AAAA",
        parent_sequence="AAAACCCC",
        control_sequence="CCCC",
        cm_detected_parent=parent,
        cm_detected_variant=var,
        cm_detected_control=ctl,
    )


def _probe_set(n: int) -> ProbeSet:
    return build_probe_set([_variant(f"v{i:03d}") for i in range(n)])


# --------------------------------------------------------------------------- #
# Probe-set assembly + the min-N floor
# --------------------------------------------------------------------------- #
def test_probe_min_n_is_the_adr_pin() -> None:
    assert TIER2N_PROBE_MIN_N == MIN_REAL_HOMOLOG_N


def test_only_probe_eligible_variants_enter_the_synthetic_arm() -> None:
    variants = [
        _variant("v1"),  # discordant → eligible
        _variant("v2", parent=False),  # parent already missed
        _variant("v3", var=True),  # ablation did not break detection
        _variant("v4", ctl=False),  # length-confounded
        _variant("v5", parent=None, var=None),  # unmeasured
    ]
    assert build_probe_set(variants).synthetic == ("v1",)


def test_natural_arm_is_empty_but_counted() -> None:
    """The natural arm is disclosed at N=0, not dropped from the accounting."""
    probe_set = _probe_set(MIN_REAL_HOMOLOG_N)
    assert probe_set.natural == ()
    decision = round_decision(probe_set, 1.0, [])
    assert decision["n_natural"] == 0
    assert decision["n_synthetic"] == MIN_REAL_HOMOLOG_N
    assert "empty by construction" in decision["natural_arm_disclosure"]


def test_sub_min_n_probe_set_is_inadmissible_not_a_pass() -> None:
    probe_set = _probe_set(MIN_REAL_HOMOLOG_N - 1)
    assert probe_set.meets_min_n() is False
    assert round_decision(probe_set, 1.0, [])["decision"] == ROUND_INADMISSIBLE


def test_empty_probe_set_does_not_meet_min_n() -> None:
    assert ProbeSet(natural=(), synthetic=()).meets_min_n() is False


def test_probe_set_at_exactly_min_n_is_admissible() -> None:
    assert _probe_set(MIN_REAL_HOMOLOG_N).meets_min_n() is True


# --------------------------------------------------------------------------- #
# Recall
# --------------------------------------------------------------------------- #
def test_recall_on_an_empty_probe_set_raises_rather_than_returning_a_number() -> None:
    with pytest.raises(Tier2NProbeError):
        probe_recall(ProbeSet(natural=(), synthetic=()), set())


def test_recall_is_the_recovered_fraction() -> None:
    probe_set = _probe_set(4)
    assert probe_recall(probe_set, {"v000", "v001"}) == 0.5
    assert probe_recall(probe_set, set()) == 0.0
    assert probe_recall(probe_set, {"v000", "v001", "v002", "v003"}) == 1.0


def test_recall_ignores_recovered_ids_outside_the_probe_set() -> None:
    """A scanner recovering unrelated loci must not inflate probe recall."""
    assert probe_recall(_probe_set(4), {"v000", "unrelated", "other"}) == 0.25


# --------------------------------------------------------------------------- #
# Halt / rollback
# --------------------------------------------------------------------------- #
def test_a_drop_at_the_threshold_halts_and_rolls_back() -> None:
    probe_set = _probe_set(MIN_REAL_HOMOLOG_N)
    decision = round_decision(probe_set, 1.0 - TIER2N_RECALL_DROP_HALT, [1.0])
    assert decision["decision"] == ROUND_HALT_ROLLBACK
    assert decision["halt_threshold_breached"] is True


def test_a_drop_below_the_threshold_continues() -> None:
    probe_set = _probe_set(MIN_REAL_HOMOLOG_N)
    decision = round_decision(probe_set, 1.0 - TIER2N_RECALL_DROP_HALT / 2, [1.0])
    assert decision["decision"] == ROUND_CONTINUE


def test_slow_monotone_bleed_trips_against_the_best_round_not_the_previous_one() -> None:
    """The failure mode a previous-round baseline cannot see.

    Each step loses less than the halt threshold, so a previous-round comparison
    never fires — but cumulatively the class is destroyed. Against the best round
    so far, it trips.
    """
    probe_set = _probe_set(MIN_REAL_HOMOLOG_N)
    step = TIER2N_RECALL_DROP_HALT / 2
    history = [1.0, 1.0 - step, 1.0 - 2 * step]
    this_round = 1.0 - 3 * step

    # Each consecutive step is under the bar ...
    assert history[-1] - this_round < TIER2N_RECALL_DROP_HALT
    # ... but the cumulative loss from the best round is not.
    assert round_decision(probe_set, this_round, history)["decision"] == ROUND_HALT_ROLLBACK


def test_first_round_has_no_baseline_and_continues() -> None:
    decision = round_decision(_probe_set(MIN_REAL_HOMOLOG_N), 0.4, [])
    assert decision["decision"] == ROUND_CONTINUE
    assert decision["best_prior_recall"] is None
    assert decision["recall_drop_vs_best"] is None


def test_recovering_more_than_the_best_round_continues() -> None:
    decision = round_decision(_probe_set(MIN_REAL_HOMOLOG_N), 0.95, [0.8, 0.7])
    assert decision["decision"] == ROUND_CONTINUE
    assert decision["recall_drop_vs_best"] == pytest.approx(-0.15)


def test_inadmissible_outranks_a_healthy_recall() -> None:
    """A sub-min-N probe set cannot certify a round even at perfect recall."""
    decision = round_decision(_probe_set(MIN_REAL_HOMOLOG_N - 1), 1.0, [1.0])
    assert decision["decision"] == ROUND_INADMISSIBLE


def test_halt_threshold_is_one_probe_positive_at_the_pinned_floor() -> None:
    """Pin the derivation, not the literal — the 1/N granularity argument.

    Exact equality, not ``approx``: both sides evaluate the identical float
    expression, so any difference means the threshold stopped being *derived* from
    the pinned floor — precisely what this test exists to catch. ``approx`` would
    tolerate a hand-edited literal that happened to land nearby.
    """
    assert TIER2N_RECALL_DROP_HALT == 1.0 / MIN_REAL_HOMOLOG_N


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_out_of_range_recall_is_rejected(bad: float) -> None:
    with pytest.raises(Tier2NProbeError):
        round_decision(_probe_set(MIN_REAL_HOMOLOG_N), bad, [])


def test_out_of_range_history_is_rejected() -> None:
    with pytest.raises(Tier2NProbeError):
        round_decision(_probe_set(MIN_REAL_HOMOLOG_N), 0.5, [0.5, 2.0])


# --------------------------------------------------------------------------- #
# Probe-set integrity (CodeRabbit round 1)
# --------------------------------------------------------------------------- #
def test_duplicate_ids_across_arms_are_rejected() -> None:
    """Otherwise ``size`` counts a member that ``probe_recall`` de-duplicates away,
    so a set could clear min-N on members recall does not recognise."""
    with pytest.raises(Tier2NProbeError, match="unique"):
        ProbeSet(natural=("shared",), synthetic=("shared",))


def test_duplicate_ids_within_one_arm_are_rejected() -> None:
    with pytest.raises(Tier2NProbeError, match="unique"):
        ProbeSet(natural=(), synthetic=("dup", "dup"))


def test_an_exact_one_positive_regression_halts_despite_float_error() -> None:
    """0.95 - 0.90 == 0.04999999999999993 in IEEE754 — a bare ``>=`` misses it.

    This is the exact-threshold regression the halt rule exists to catch, so it
    must not be decided by float representation.
    """
    assert TIER2N_RECALL_DROP_HALT > 0.95 - 0.90  # the trap, pinned explicitly
    decision = round_decision(_probe_set(MIN_REAL_HOMOLOG_N), 0.90, [0.95])
    assert decision["decision"] == ROUND_HALT_ROLLBACK


# --------------------------------------------------------------------------- #
# FASTA / tblout round-trip integrity — the fail-open path into the probe set
# --------------------------------------------------------------------------- #
def test_a_whitespace_bearing_name_is_refused(tmp_path) -> None:
    """cmsearch's tblout target column keeps only the first whitespace token.

    A header with a space would therefore never match its own key in
    ``detection_map``, read as "CM missed it", and manufacture a probe positive.
    """
    with pytest.raises(ValueError, match="single token"):
        write_fasta({"bad name": "ACGU"}, tmp_path / "x.fa")


def test_a_whitespace_bearing_sequence_is_refused(tmp_path) -> None:
    """Embedded whitespace would split one record into several."""
    with pytest.raises(ValueError, match="whitespace-free"):
        write_fasta({"ok": "ACGU\nACGU"}, tmp_path / "x.fa")


def test_an_empty_name_or_sequence_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_fasta({"": "ACGU"}, tmp_path / "x.fa")
    with pytest.raises(ValueError):
        write_fasta({"ok": "---"}, tmp_path / "x.fa")


def test_write_fasta_ungaps_and_uppercases(tmp_path) -> None:
    path = write_fasta({"r1": "ac-gu"}, tmp_path / "x.fa")
    assert path.read_text(encoding="utf-8") == ">r1\nACGU\n"


def test_detection_map_keys_off_submitted_records_not_the_hit_table() -> None:
    """A record drawing no hit must be an explicit False, never an absent key."""
    hits = [CmsearchHit(target="r1", score=100.0, evalue=1e-20)]
    assert detection_map({"r1": "ACGU", "r2": "ACGU"}, hits) == {"r1": True, "r2": False}


def test_caller_metadata_cannot_shadow_a_derived_count() -> None:
    """Otherwise a report's n_detected could come from the caller, not the hits."""
    report = cm_build_report({"r1": "ACGU"}, [], n_detected=999, arm="spoof")
    assert report["n_detected"] == 0
    assert report["arm"] == "spoof"


def test_write_fasta_leaves_a_prior_file_intact_when_a_record_is_invalid(tmp_path) -> None:
    """Validation runs before truncation, so a rejected batch cannot leave a
    partial FASTA that a later cmsearch would happily search."""
    path = tmp_path / "x.fa"
    write_fasta({"good": "ACGU"}, path)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        write_fasta({"good": "ACGU", "bad name": "ACGU"}, path)
    assert path.read_text(encoding="utf-8") == before
