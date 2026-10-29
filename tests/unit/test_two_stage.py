"""P3-14 unit — the two-stage harness's composition, refusals, and the branches the committed
fixture does not reach.

The golden (``tests/golden/test_two_stage_golden.py``) locks the harness's arithmetic against
real model output. It cannot cover everything: the fixture's loci are canonical corpus T-boxes,
so **every one of them resolves** and D15's ambiguous / both-strand-carry-through branch — the
recall-favouring half of the rule — emits nothing there. Those branches are exercised here, on
declared synthetic log-posteriors built to make a specific thing happen.

Where a test uses a hand-built posterior it says so; nothing in this file is a measurement of
either checkpoint, and no number here is a performance claim.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tbox_finder.infer import call as C
from tbox_finder.infer import locus as L
from tbox_finder.infer import strand as S
from tbox_finder.infer.reconcile import log_softmax
from tbox_finder.integration import two_stage as T
from tbox_finder.labels import CLASS_INDEX
from tbox_finder.models import rna_backbone_registry as BR

_SEQ_LEN = 1536
_WINDOW = 1024
_BG = CLASS_INDEX["background"]

#: A deliberately non-palindromic carrier — see :func:`_plant`.
_CARRIER = "AAGCTCGGATTACCTGAA"

_RULE = {
    "threshold_scope": "global",
    "threshold": 0.9,
    "min_span": 50,
    "gap_merge": 10,
    "min_distinct_elements": 2,
    "flank": 20,
    "min_order_margin": 1,
}
_CALIBRATION = {
    "temperature": 1.25,
    "stage2_operating_point": 0.5,
    "source_prior": None,
    "target_prior": None,
}


# ══════════════════════════════════════════════════════════════════════════════════════
# Declared synthetic oracle
# ══════════════════════════════════════════════════════════════════════════════════════
def _plant(
    spans: list[tuple[int, int, str]], *, seq_len: int = _SEQ_LEN, strong: float = 8.0
) -> tuple[np.ndarray, list[int], str]:
    """A declared stub Stage-1: window logits that make ``spans`` element-like and nothing else.

    Returns ``(window_logits, starts, sequence)`` in the shape ``reconcile_windows`` takes. The
    element logit is high enough that ``1 − P(background)`` clears any threshold used here, and
    the carrier is background everywhere else. This is window *geometry*, not a model: nothing
    in this file may be read as a property of the trained scanner.
    """
    per_position = np.zeros((seq_len, len(T.CLASS_ORDER)), dtype=np.float64)
    per_position[:, _BG] = strong
    for start, end, klass in spans:
        per_position[start:end, _BG] = 0.0
        per_position[start:end, CLASS_INDEX[klass]] = strong
    starts = [0, seq_len - _WINDOW]
    windows = np.zeros((len(starts), _WINDOW, len(T.CLASS_ORDER)), dtype=np.float64)
    for index, start in enumerate(starts):
        windows[index] = per_position[start : start + _WINDOW]
    # NOT "ACGT" repeated: that carrier is its own reverse complement, so the + and - payloads
    # of every locus come out byte-identical and every test that distinguishes them goes vacuous
    # (measured — two duplicate-agreement tests failed on exactly this before the carrier
    # changed). `_CARRIER` is asserted non-palindromic by test_the_carrier_is_not_self_rc.
    sequence = (_CARRIER * (seq_len // len(_CARRIER) + 1))[:seq_len]
    return windows, starts, sequence


def _contig(contig_id: str, sequence: str, **truth) -> dict[str, object]:
    return {"contig_id": contig_id, "sequence": sequence, **truth}


def _canonical_spans() -> list[tuple[int, int, str]]:
    """5′→3′ canonical order, so the resolver returns ``+`` with a positive margin.

    The three runs are **contiguous**, not gapped: at ``gap_merge=10`` a 20-nt gap would leave
    three single-class candidates, each of which ``min_distinct_elements=2`` discards, and every
    test below would run on an empty locus list.
    ``test_the_planted_oracle_actually_produces_a_locus`` is what caught that — a degenerate
    generator reads exactly like a working one.
    """
    return [(600, 700, "Stem_I"), (700, 780, "Specifier"), (780, 880, "Terminator")]


def _single_element_span() -> list[tuple[int, int, str]]:
    """One element only — D15's first ambiguous case: no pair exists to vote."""
    return [(600, 760, "Stem_I")]


def _run(spans, *, contig_id="c1", rule=None, **truth):
    windows, starts, sequence = _plant(spans)
    contigs = [_contig(contig_id, sequence, **truth)]
    stage1 = {contig_id: {"logits": windows, "starts": starts}}
    reconciled = T.reconcile_contig(stage1[contig_id], len(sequence))
    return contigs, stage1, T.run_contig(contigs[0], reconciled, **{**_RULE, **(rule or {})})


def _scores(runs, value: float = 6.0, overrides: dict[str, float] | None = None):
    logits = {entry["row_id"]: value for entry in T.payload_manifest(runs)}
    logits.update(overrides or {})
    return logits


# ══════════════════════════════════════════════════════════════════════════════════════
# The oracle is not degenerate
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_planted_oracle_actually_produces_a_locus() -> None:
    """A fixture that silently produced nothing would make every test below vacuous."""
    _, _, run = _run(_canonical_spans())
    assert len(run.loci) == 1
    assert run.loci[0].n_distinct_elements == 3
    assert run.calls[0].strand == "+"


def test_the_carrier_is_not_self_reverse_complementary() -> None:
    """Guards the guard: a self-RC carrier makes every +/- comparison in this file vacuous."""
    from tbox_finder.infer.handoff import reverse_complement

    _, _, sequence = _plant(_canonical_spans())
    assert reverse_complement(sequence) != sequence
    _, _, run = _run(_canonical_spans())
    plus = next(p for p in run.both.payloads if p.strand == "+")
    minus = next(p for p in run.both.payloads if p.strand == "-")
    assert plus.rna != minus.rna


def test_the_planted_oracle_covers_every_position_at_least_once() -> None:
    windows, starts, sequence = _plant(_canonical_spans())
    reconciled = T.reconcile_contig({"logits": windows, "starts": starts}, len(sequence))
    assert int(reconciled.coverage.min()) >= 1
    assert int(reconciled.coverage.max()) == 2


# ══════════════════════════════════════════════════════════════════════════════════════
# Signature / inventory discipline
# ══════════════════════════════════════════════════════════════════════════════════════
def test_no_rule_parameter_has_a_default() -> None:
    assert T.no_rule_parameter_has_a_default() is True


def test_the_predicate_rejects_a_stub_carrying_a_default() -> None:
    def stub(
        contigs,
        stage1,
        stage2_logits,
        *,
        threshold_scope,
        threshold,
        min_span,
        gap_merge,
        min_distinct_elements,
        flank,
        min_order_margin,
        temperature,
        stage2_operating_point,
        source_prior=None,
        target_prior=None,
    ): ...

    assert T.no_rule_parameter_has_a_default(stub) is False


def test_the_predicate_rejects_a_knob_missing_from_the_inventory() -> None:
    def stub(contigs, *, threshold_scope, threshold): ...

    assert T.no_rule_parameter_has_a_default(stub) is False


def test_the_promoted_predicate_is_the_one_all_three_rules_use() -> None:
    """The promotion, checked at the shared function rather than by comparing its callers.

    Asserting ``locus`` agrees with ``strand`` would be a tautology once both delegate
    ([[promote-dont-duplicate-is-a-correctness-rule]]), so the shared implementation is driven
    directly with a compliant and a non-compliant stub.
    """

    def compliant(a, *, x, y): ...
    def defaulted(a, *, x, y=1): ...

    assert C.rule_parameters_have_no_default(compliant, ("x", "y")) is True
    assert C.rule_parameters_have_no_default(defaulted, ("x", "y")) is False
    assert C.rule_parameters_have_no_default(compliant, ("x",)) is False
    # And the three live rules still hold, each against its own inventory.
    assert L.no_rule_parameter_has_a_default() is True
    assert S.no_rule_parameter_has_a_default() is True
    assert T.no_rule_parameter_has_a_default() is True


def test_every_rule_parameter_appears_in_the_report() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    result = T.run_two_stage(contigs, stage1, _scores([run]), **{**_RULE, **_CALIBRATION})
    assert set(T.RULE_PARAMETERS) <= set(result.report["rule"])


# ══════════════════════════════════════════════════════════════════════════════════════
# Payload identity
# ══════════════════════════════════════════════════════════════════════════════════════
def test_payload_key_changes_with_every_component() -> None:
    _, _, run = _run(_canonical_spans())
    payload = run.both.payloads[0]
    base = T.payload_key("c1", payload)
    assert T.payload_key("c2", payload) != base
    assert T.payload_key("c1", replace(payload, locus_index=payload.locus_index + 1)) != base
    assert T.payload_key("c1", replace(payload, strand="-")) != base
    assert T.payload_key("c1", replace(payload, start=payload.start + 1)) != base
    assert T.payload_key("c1", replace(payload, end=payload.end + 1)) != base
    # The RNA mutation is checked to BE a mutation first: the carrier happens to end in "A", so
    # an earlier `rna[:-1] + "A"` was a no-op and this assertion passed while testing nothing
    # ([[vacuous-test-perturbations]]).
    mutated_rna = payload.rna[:-1] + ("C" if payload.rna[-1] != "C" else "G")
    assert mutated_rna != payload.rna
    assert T.payload_key("c1", replace(payload, rna=mutated_rna)) != base
    assert T.payload_key("c1", payload) == base


def test_payload_key_is_stable_under_fields_stage2_never_sees() -> None:
    """Only what reaches Stage-2 is in the key — the flag rides in the table, not the address."""
    _, _, run = _run(_canonical_spans())
    payload = run.both.payloads[0]
    unchanged = replace(payload, low_order_confidence=not payload.low_order_confidence)
    assert T.payload_key("c1", unchanged) == T.payload_key("c1", payload)


# ══════════════════════════════════════════════════════════════════════════════════════
# The counterfactual comes out of the shipped handoff
# ══════════════════════════════════════════════════════════════════════════════════════
def test_both_strand_payloads_are_the_shipped_transcriber_output() -> None:
    from tbox_finder.infer.handoff import transcribe_to_rna

    _, _, run = _run(_canonical_spans())
    locus = run.loci[0]
    span = run.sequence[locus.start : locus.end]
    minus = next(p for p in run.both.payloads if p.strand == "-")
    assert minus.rna == transcribe_to_rna(span, strand="-")
    plus = next(p for p in run.both.payloads if p.strand == "+")
    assert plus.rna == transcribe_to_rna(span, strand="+")


def test_the_counterfactual_widens_strands_and_changes_nothing_else() -> None:
    _, _, run = _run(_canonical_spans())
    call = run.calls[0]
    assert call.strands == ("+",)
    assert len(run.emitted.payloads) == 1
    assert len(run.both.payloads) == 2
    # Every measured field of the call is untouched by the widening — the counterfactual asks
    # "what if the other strand", not "what if the vote had gone differently".
    for payload in run.both.payloads:
        assert payload.low_order_confidence is call.low_order_confidence
        assert (payload.start, payload.end) == (run.loci[0].start, run.loci[0].end)


# ══════════════════════════════════════════════════════════════════════════════════════
# D15's ambiguous branch — unreachable from the committed fixture
# ══════════════════════════════════════════════════════════════════════════════════════
def test_a_single_element_locus_is_ambiguous_and_emits_two_payloads() -> None:
    _, _, run = _run(_single_element_span(), rule={"min_distinct_elements": 1})
    call = run.calls[0]
    assert call.strand is None
    assert call.low_order_confidence is True
    assert call.reason == S.REASON_SINGLE_ELEMENT
    assert call.strands == ("+", "-")
    assert len(run.emitted.payloads) == 2
    assert {p.strand for p in run.emitted.payloads} == {"+", "-"}


def test_an_ambiguous_locus_reaches_the_table_as_two_rows() -> None:
    contigs, stage1, run = _run(_single_element_span(), rule={"min_distinct_elements": 1})
    result = T.run_two_stage(
        contigs,
        stage1,
        _scores([run]),
        **{**_RULE, "min_distinct_elements": 1},
        **_CALIBRATION,
    )
    assert len(result.rows) == 2
    assert {row["strand"] for row in result.rows} == {"+", "-"}
    assert all(row["low_order_confidence"] for row in result.rows)
    assert all(row["strands_carried"] == "+,-" for row in result.rows)
    diagnostic = result.report["strand_robustness"]
    assert diagnostic["ambiguity_path_exercised"] is True
    assert diagnostic["n_low_order_confidence"] == 1
    assert diagnostic["low_order_confidence_fraction"] == 1.0


def test_an_ambiguous_locus_is_confirmed_if_either_strand_confirms() -> None:
    """Recall-favouring, per D15: the locus survives on whichever strand is real."""
    contigs, stage1, run = _run(_single_element_span(), rule={"min_distinct_elements": 1})
    manifest = T.payload_manifest([run])
    plus = next(e["row_id"] for e in manifest if e["strand"] == "+")
    minus = next(e["row_id"] for e in manifest if e["strand"] == "-")
    logits = {plus: 8.0, minus: -8.0}
    result = T.run_two_stage(
        contigs, stage1, logits, **{**_RULE, "min_distinct_elements": 1}, **_CALIBRATION
    )
    diagnostic = result.report["strand_robustness"]
    assert diagnostic["n_confirmed_loci"] == 1
    assert diagnostic["n_verdict_disagreements"] == 1
    assert diagnostic["confirmation_invariance"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════════════
# The diagnostic's arithmetic
# ══════════════════════════════════════════════════════════════════════════════════════
def test_a_locus_confirmed_on_both_strands_is_invariant() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    result = T.run_two_stage(contigs, stage1, _scores([run], 8.0), **{**_RULE, **_CALIBRATION})
    diagnostic = result.report["strand_robustness"]
    assert diagnostic["n_confirmed_loci"] == 1
    assert diagnostic["n_confirmed_invariant"] == 1
    assert diagnostic["confirmation_invariance"] == 1.0
    assert diagnostic["n_verdict_disagreements"] == 0
    # ...and the liveness control correctly reports that nothing discriminated here, so a 1.0
    # from a strand-blind scorer is distinguishable from a 1.0 from a robust locus.
    assert diagnostic["max_abs_posterior_delta"] == 0.0


def test_confirmation_invariance_is_none_when_nothing_is_confirmed() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    result = T.run_two_stage(contigs, stage1, _scores([run], -8.0), **{**_RULE, **_CALIBRATION})
    diagnostic = result.report["strand_robustness"]
    assert diagnostic["n_confirmed_loci"] == 0
    assert diagnostic["confirmation_invariance"] is None


def _reversed_spans() -> list[tuple[int, int, str]]:
    """Canonical elements laid out **backwards**, so the resolver calls ``-``.

    Medians run Terminator 650 < Specifier 740 < Stem_I 830, the exact reverse of
    ``CANONICAL_ELEMENT_ORDER``, so every pair votes discordant and ``order_margin`` is negative.
    """
    return [(600, 700, "Terminator"), (700, 780, "Specifier"), (780, 880, "Stem_I")]


def test_the_reversed_oracle_really_makes_the_resolver_call_minus() -> None:
    """Without this the wrong-strand test below would be measuring the wrong thing."""
    _, _, run = _run(_reversed_spans())
    assert run.calls[0].strand == "-"
    assert run.calls[0].strands == ("-",)


def test_a_locus_confirmed_only_on_the_wrong_strand_is_counted() -> None:
    """The one outcome D15 forbids: a mis-resolution that becomes a false novelty claim.

    Built deliberately, and built to match the *emitted* reading (CodeRabbit r2): the elements
    are laid out in reverse so the resolver calls ``-`` while truth says ``+``, and Stage-2 then
    confirms the ``-`` payload the harness actually emits. That is a candidate reaching a table
    on the wrong strand — which is what the counter must catch. A locus whose merely
    *counterfactual* opposite strand scored well is **not** one, and an earlier version of this
    test asserted exactly that weaker thing.
    """
    contigs, stage1, run = _run(_reversed_spans(), truth_strand="+", truth_start=600, truth_end=880)
    manifest = T.payload_manifest([run])
    plus = next(e["row_id"] for e in manifest if e["strand"] == "+")
    minus = next(e["row_id"] for e in manifest if e["strand"] == "-")
    result = T.run_two_stage(contigs, stage1, {plus: -8.0, minus: 8.0}, **{**_RULE, **_CALIBRATION})
    truth = result.report["strand_robustness"]["truth"]
    assert truth["n_loci_overlapping_truth"] == 1
    assert truth["n_strand_calls_incorrect"] == 1
    assert truth["n_confirmed_on_wrong_strand_only"] == 1


def test_an_unemitted_counterfactual_is_not_a_wrong_strand_confirmation() -> None:
    """The distinction r2 drew, as its own test.

    The resolver calls ``+`` and truth agrees, so ``-`` is **never emitted**. Scoring that
    unemitted counterfactual high and the emitted ``+`` low is a locus the harness simply does
    not confirm — a bounded false negative — and must NOT be counted as a false novelty.
    """
    contigs, stage1, run = _run(
        _canonical_spans(), truth_strand="+", truth_start=600, truth_end=880
    )
    manifest = T.payload_manifest([run])
    plus = next(e["row_id"] for e in manifest if e["strand"] == "+")
    minus = next(e["row_id"] for e in manifest if e["strand"] == "-")
    result = T.run_two_stage(contigs, stage1, {plus: -8.0, minus: 8.0}, **{**_RULE, **_CALIBRATION})
    diagnostic = result.report["strand_robustness"]
    assert diagnostic["n_confirmed_loci"] == 0, "the emitted strand did not confirm"
    assert diagnostic["truth"]["n_confirmed_on_wrong_strand_only"] == 0


def test_the_wrong_strand_counter_stays_zero_when_the_right_strand_confirms() -> None:
    """The positive control for the test above: the identical setup, scores the other way."""
    contigs, stage1, run = _run(
        _canonical_spans(), truth_strand="+", truth_start=600, truth_end=880
    )
    manifest = T.payload_manifest([run])
    plus = next(e["row_id"] for e in manifest if e["strand"] == "+")
    minus = next(e["row_id"] for e in manifest if e["strand"] == "-")
    result = T.run_two_stage(contigs, stage1, {plus: 8.0, minus: -8.0}, **{**_RULE, **_CALIBRATION})
    assert result.report["strand_robustness"]["truth"]["n_confirmed_on_wrong_strand_only"] == 0


def test_truth_is_absent_unless_the_contig_declares_it() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    result = T.run_two_stage(contigs, stage1, _scores([run]), **{**_RULE, **_CALIBRATION})
    truth = result.report["strand_robustness"]["truth"]
    assert truth["n_loci_overlapping_truth"] == 0
    assert truth["n_confirmed_on_wrong_strand_only"] == 0


def test_a_locus_not_overlapping_truth_is_excluded_from_the_truth_block() -> None:
    contigs, stage1, run = _run(
        _canonical_spans(), truth_strand="+", truth_start=100, truth_end=200
    )
    result = T.run_two_stage(contigs, stage1, _scores([run]), **{**_RULE, **_CALIBRATION})
    assert result.report["strand_robustness"]["truth"]["n_loci_overlapping_truth"] == 0


# ══════════════════════════════════════════════════════════════════════════════════════
# Calibration is delegated, not re-implemented
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_named_posterior_is_the_key_the_calibration_stack_names() -> None:
    from tbox_finder.calib import recalibrate as R

    contigs, stage1, run = _run(_canonical_spans())
    logits = _scores([run], 2.5)
    result = T.run_two_stage(contigs, stage1, logits, **{**_RULE, **_CALIBRATION})
    payload = R.calibrated_posterior(np.array([2.5]), temperature=_CALIBRATION["temperature"])
    expected = float(payload[payload["gated_posterior_key"]][0])
    assert result.rows[0]["stage2_named_posterior"] == pytest.approx(expected)
    assert result.rows[0]["stage2_prior_shifted_posterior"] is None


def test_temperature_moves_the_posterior() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    logits = _scores([run], 2.0)
    hot = T.run_two_stage(
        contigs, stage1, logits, **{**_RULE, **{**_CALIBRATION, "temperature": 4.0}}
    )
    cold = T.run_two_stage(
        contigs, stage1, logits, **{**_RULE, **{**_CALIBRATION, "temperature": 1.0}}
    )
    assert hot.rows[0]["stage2_named_posterior"] < cold.rows[0]["stage2_named_posterior"]


def test_a_half_specified_prior_pair_is_refused_by_the_stack() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    with pytest.raises(ValueError, match="both source_prior and target_prior"):
        T.run_two_stage(
            contigs,
            stage1,
            _scores([run]),
            **{**_RULE, **{**_CALIBRATION, "source_prior": 0.5}},
        )


def test_the_prior_shift_populates_its_own_column_when_both_priors_are_given() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    result = T.run_two_stage(
        contigs,
        stage1,
        _scores([run], 2.0),
        **{**_RULE, **{**_CALIBRATION, "source_prior": 0.5, "target_prior": 0.001}},
    )
    row = result.rows[0]
    assert row["stage2_prior_shifted_posterior"] is not None
    assert row["stage2_prior_shifted_posterior"] < row["stage2_named_posterior"]
    # The gated column is the pre-shift one — a shifted posterior is miscalibrated at benchmark
    # prevalence by construction (PRD §12), so `confirmed` must not read it.
    assert row["confirmed"] is (row["stage2_named_posterior"] >= 0.5)


# ══════════════════════════════════════════════════════════════════════════════════════
# Fail-closed
# ══════════════════════════════════════════════════════════════════════════════════════
def test_a_missing_stage2_score_raises_and_the_same_call_succeeds_with_it() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    logits = _scores([run])
    victim = sorted(logits)[0]
    partial = {key: value for key, value in logits.items() if key != victim}
    with pytest.raises(T.TwoStageError, match="no Stage-2 logit"):
        T.run_two_stage(contigs, stage1, partial, **{**_RULE, **_CALIBRATION})
    assert T.run_two_stage(contigs, stage1, logits, **{**_RULE, **_CALIBRATION}).rows


def test_a_non_finite_stage2_score_is_refused() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    logits = _scores([run])
    logits[sorted(logits)[0]] = float("nan")
    with pytest.raises(T.TwoStageError, match="not finite"):
        T.run_two_stage(contigs, stage1, logits, **{**_RULE, **_CALIBRATION})


def test_a_contig_without_stage1_logits_is_refused() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    with pytest.raises(T.TwoStageError, match="no Stage-1 window logits"):
        T.run_two_stage(contigs, {}, _scores([run]), **{**_RULE, **_CALIBRATION})


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"sequence": "ACGT"}, "missing required key 'contig_id'"),
        ({"contig_id": "c"}, "missing required key 'sequence'"),
        ({"contig_id": "", "sequence": "ACGT"}, "non-empty string"),
        ({"contig_id": "c", "sequence": ""}, "carries no sequence"),
        ({"contig_id": "c", "sequence": "ACGT", "truth_strand": "?"}, "truth_strand"),
    ],
)
def test_malformed_contig_records_are_refused(record, match) -> None:
    with pytest.raises(T.TwoStageError, match=match):
        T.normalise_contig(record)


def test_a_well_formed_contig_record_is_accepted() -> None:
    """Positive control for the parametrised refusals above."""
    record = T.normalise_contig({"contig_id": "c", "sequence": "ACGT", "truth_strand": "-"})
    assert record["contig_id"] == "c"
    assert record["truth_strand"] == "-"
    assert record["truth_start"] is None


def test_a_stage1_entry_missing_starts_is_refused() -> None:
    windows, starts, sequence = _plant(_canonical_spans())
    with pytest.raises(T.TwoStageError, match="missing required key 'starts'"):
        T.reconcile_contig({"logits": windows}, len(sequence))


# ══════════════════════════════════════════════════════════════════════════════════════
# The digest
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_digest_covers_every_declared_column() -> None:
    """Dropping any one column from a row must move the digest, or the column is decoration."""
    contigs, stage1, run = _run(_canonical_spans())
    rows = [
        dict(row)
        for row in T.run_two_stage(
            contigs, stage1, _scores([run]), **{**_RULE, **_CALIBRATION}
        ).rows
    ]
    base = T.candidate_table_digest(rows)
    for column in T.CANDIDATE_COLUMNS:
        mutated = [dict(row) for row in rows]
        value = mutated[0][column]
        if isinstance(value, bool):
            mutated[0][column] = not value
        elif isinstance(value, str):
            mutated[0][column] = value + "x"
        elif isinstance(value, int):
            mutated[0][column] = value + 1
        elif isinstance(value, float):
            mutated[0][column] = value + 1.0
        else:  # None
            mutated[0][column] = 0
        assert T.candidate_table_digest(mutated) != base, column


def test_the_table_rows_carry_exactly_the_declared_columns() -> None:
    contigs, stage1, run = _run(_canonical_spans())
    rows = T.run_two_stage(contigs, stage1, _scores([run]), **{**_RULE, **_CALIBRATION}).rows
    assert set(rows[0]) == set(T.CANDIDATE_COLUMNS)


def test_a_non_finite_cell_is_refused_by_the_digest() -> None:
    with pytest.raises(T.TwoStageError, match="not finite"):
        T.candidate_table_digest([{column: float("inf") for column in T.CANDIDATE_COLUMNS}])


# ══════════════════════════════════════════════════════════════════════════════════════
# The report validator bites
# ══════════════════════════════════════════════════════════════════════════════════════
def _report():
    contigs, stage1, run = _run(_canonical_spans())
    return T.run_two_stage(contigs, stage1, _scores([run]), **{**_RULE, **_CALIBRATION}).report


def test_a_clean_report_has_no_problems() -> None:
    assert T.strand_robustness_problems(_report()) == []


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.__setitem__("is_science", True), "is_science must be false"),
        (lambda r: r.__setitem__("gated", True), "gated must be false"),
        (lambda r: r.__setitem__("step", "P9-99"), "step is"),
        (lambda r: r["rule"].__setitem__("pinned", True), "rule.pinned must be false"),
        (lambda r: r["rule"].pop("flank"), "missing knob 'flank'"),
        (
            lambda r: r["strand_robustness"].__setitem__("tier_invariance", 1.0),
            "tier_invariance must be null",
        ),
        (
            lambda r: r["strand_robustness"].__setitem__("tier_invariance_reason", ""),
            "must state the substitution",
        ),
        (
            lambda r: r["strand_robustness"].__setitem__("confirmation_invariance", 0.5),
            "does not re-derive",
        ),
        (
            lambda r: r["strand_robustness"].__setitem__("n_confirmed_invariant", 99),
            "exceeds n_confirmed_loci",
        ),
        (
            lambda r: r["strand_robustness"].__setitem__("n_low_order_confidence", 99),
            "exceeds n_loci",
        ),
        (
            lambda r: r["strand_robustness"].__setitem__("strand_call_reasons", {"resolved": 7}),
            "strand_call_reasons sums to",
        ),
        (
            lambda r: r["strand_robustness"].__setitem__("ambiguity_path_exercised", True),
            "ambiguity_path_exercised does not re-derive",
        ),
        (lambda r: r["candidate_table"].__setitem__("columns", ["a"]), "not the current"),
        (lambda r: r["candidate_table"].__setitem__("digest_quantum", 7), "digest_quantum"),
        (lambda r: r["candidate_table"].__setitem__("digest", "short"), "not a sha256"),
        (lambda r: r["candidate_table"].__setitem__("n_rows", 99), "n_emitted_payloads"),
        (lambda r: r["scope"].__setitem__("n_scored_payloads", 1), "2 \u00d7 n_loci"),
        (lambda r: r["scope"].__setitem__("n_loci", 99), "disagrees with scope.n_loci"),
        (lambda r: r.pop("disclosures"), "missing top-level key 'disclosures'"),
    ],
)
def test_the_report_validator_catches_each_inconsistency(mutate, match) -> None:
    report = _report()
    mutate(report)
    problems = T.strand_robustness_problems(report)
    assert any(match in problem for problem in problems), problems


def test_an_ambiguous_locus_makes_emitted_payloads_exceed_locus_count() -> None:
    """The clause that ties the two counts together, on the input that separates them."""
    contigs, stage1, run = _run(_single_element_span(), rule={"min_distinct_elements": 1})
    report = T.run_two_stage(
        contigs, stage1, _scores([run]), **{**_RULE, "min_distinct_elements": 1}, **_CALIBRATION
    ).report
    assert report["scope"]["n_emitted_payloads"] == 2
    assert report["scope"]["n_loci"] == 1
    assert T.strand_robustness_problems(report) == []
    report["scope"]["n_emitted_payloads"] = 1
    assert any(
        "n_loci + n_low_order_confidence" in problem
        for problem in T.strand_robustness_problems(report)
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# Duplicate-RNA agreement
# ══════════════════════════════════════════════════════════════════════════════════════
def test_duplicate_rna_agreement_reports_the_spread_it_is_given() -> None:
    contigs, stage1, run = _run(_canonical_spans(), contig_id="c1")
    manifest = T.payload_manifest([run])
    logits = {entry["row_id"]: 1.0 for entry in manifest}
    measured = T.duplicate_rna_agreement([run], logits)
    assert measured["n_distinct_rna_payloads"] == 2  # + and - of one locus differ in sequence
    assert measured["n_rna_scored_more_than_once"] == 0
    assert measured["max_abs_duplicate_logit_delta"] is None


def test_duplicate_rna_agreement_sees_the_same_rna_from_two_contigs() -> None:
    """Two contigs carrying the identical sequence give the identical RNA — the fwd/RC shape."""
    windows, starts, sequence = _plant(_canonical_spans())
    contigs = [_contig("a", sequence), _contig("b", sequence)]
    stage1 = {
        "a": {"logits": windows, "starts": starts},
        "b": {"logits": windows, "starts": starts},
    }
    runs = [
        T.run_contig(
            contig, T.reconcile_contig(stage1[contig["contig_id"]], len(sequence)), **_RULE
        )
        for contig in contigs
    ]
    manifest = T.payload_manifest(runs)
    logits = {entry["row_id"]: 1.0 for entry in manifest}
    logits[manifest[0]["row_id"]] = 1.5
    measured = T.duplicate_rna_agreement(runs, logits)
    assert measured["n_rna_scored_more_than_once"] == 2
    assert measured["max_abs_duplicate_logit_delta"] == pytest.approx(0.5)
    assert measured["n_duplicate_groups_disagreeing"] == 1


# ══════════════════════════════════════════════════════════════════════════════════════
# Temperature provenance
# ══════════════════════════════════════════════════════════════════════════════════════
def test_read_temperature_derives_from_the_gate2_report(tmp_path: Path) -> None:
    path = tmp_path / "gate2.json"
    path.write_text(json.dumps({"gate": {"calibration": {"temperature": 1.25}}}))
    assert T.read_temperature(path) == 1.25


def test_read_temperature_refuses_a_non_positive_value(tmp_path: Path) -> None:
    path = tmp_path / "gate2.json"
    path.write_text(json.dumps({"gate": {"calibration": {"temperature": 0.0}}}))
    with pytest.raises(T.TwoStageError, match="non-positive temperature"):
        T.read_temperature(path)


def test_the_committed_gate2_report_still_carries_the_temperature() -> None:
    """The derivation has a live target; a moved key would otherwise only fail at run time.

    ``reports/gate2_p3_ece.json`` is **git-tracked**, so its absence is a repository defect and
    this asserts rather than skips — the two CLI tests below read the same artifact unguarded,
    and one requirement stated once beats a skip in one place and a hard read in two
    (CodeRabbit r1).
    """
    report = Path(__file__).resolve().parents[2] / "reports" / "gate2_p3_ece.json"
    assert report.is_file(), "reports/gate2_p3_ece.json is git-tracked and must be present"
    assert T.read_temperature(report) > 0.0


# ══════════════════════════════════════════════════════════════════════════════════════
# Composition against the real operator contracts
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_harness_consumes_the_reconciled_contract_the_operator_returns() -> None:
    """dtypes included — float64 log-probs, bool zero_flanked, int32 coverage."""
    windows, starts, sequence = _plant(_canonical_spans())
    reconciled = T.reconcile_contig({"logits": windows, "starts": starts}, len(sequence))
    assert reconciled.log_probs.dtype == np.float64
    assert reconciled.zero_flanked.dtype == np.bool_
    assert reconciled.coverage.dtype == np.int32
    run = T.run_contig(_contig("c1", sequence), reconciled, **_RULE)
    assert run.loci[0].n_single_covered_span is not None, "coverage must reach the Locus"


def test_the_strand_call_reads_the_same_assignment_the_loci_were_called_from() -> None:
    """A different threshold on the two calls would orient a locus by evidence it was not called
    on; the harness passes one pair of values to both, so a divergence is impossible by wiring.
    """
    import inspect

    source = inspect.getsource(T.run_contig)
    assert "threshold_scope=threshold_scope" in source
    assert source.count("threshold=threshold,") == 2


def test_reconcile_windows_is_the_only_reduction_the_harness_applies() -> None:
    """No second log-sum-exp anywhere in the module — ADR-0005 A3 pins exactly one."""
    import inspect

    source = inspect.getsource(T)
    assert "logsumexp" not in source
    assert "log_softmax" not in source
    assert source.count("reconcile_windows(") >= 1


def test_planted_log_probs_are_normalised_by_the_operator() -> None:
    """Guards the oracle itself: raw planted logits are not a distribution until reconciled."""
    windows, starts, sequence = _plant(_canonical_spans())
    reconciled = T.reconcile_contig({"logits": windows, "starts": starts}, len(sequence))
    total = np.exp(reconciled.log_probs).sum(axis=1)
    np.testing.assert_allclose(total, np.ones_like(total), atol=1e-12)
    # ...and the un-reconciled planted rows are NOT normalised, so the assertion above is a
    # statement about the operator rather than about the fixture.
    assert not np.allclose(np.exp(windows[0]).sum(axis=1), 1.0)
    assert np.allclose(np.exp(log_softmax(windows[0])).sum(axis=1), 1.0)


# ══════════════════════════════════════════════════════════════════════════════════════
# The CLI writes what a public repo can carry
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_run_leg_records_repo_relative_paths(tmp_path: Path) -> None:
    """An absolute path in a committed report leaks a username and resolves for nobody else."""
    root = Path(__file__).resolve().parents[2]
    report, table = tmp_path / "r.json", tmp_path / "t.json"
    exit_code = T.main(
        [
            "run",
            "--contigs",
            str(root / "tests/fixtures/two_stage/contigs.json"),
            "--stage1",
            str(root / "tests/fixtures/two_stage/stage1_window_logits.npz"),
            "--stage2",
            str(root / "tests/fixtures/two_stage/stage2_logits.json"),
            "--report",
            str(report),
            "--table",
            str(table),
            "--threshold-scope",
            "global",
            "--threshold",
            "0.9",
            "--min-span",
            "50",
            "--gap-merge",
            "10",
            "--min-distinct-elements",
            "2",
            "--flank",
            "50",
            "--min-order-margin",
            "1",
            "--temperature-from",
            str(root / "reports/gate2_p3_ece.json"),
            "--stage2-operating-point",
            "0.5",
        ]
    )
    assert exit_code == 0
    written = json.loads(report.read_text())
    for value in (
        written["scope"]["stage1_source"],
        written["scope"]["stage2_source"],
        written["rule"]["temperature_source"],
    ):
        assert not Path(value).is_absolute(), value
        assert (root / value).is_file(), value


def test_a_path_outside_the_repository_is_refused(tmp_path: Path) -> None:
    """Positive control beside it: a repo path resolves, an outside one raises."""
    root = Path(__file__).resolve().parents[2]
    assert T.repo_relative(root / "reports" / "gate2_p3_ece.json") == "reports/gate2_p3_ece.json"
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}")
    with pytest.raises(T.TwoStageError, match="outside the repository"):
        T.repo_relative(outside)


def test_the_run_leg_fails_on_a_digest_mismatch(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    exit_code = T.main(
        [
            "run",
            "--contigs",
            str(root / "tests/fixtures/two_stage/contigs.json"),
            "--stage1",
            str(root / "tests/fixtures/two_stage/stage1_window_logits.npz"),
            "--stage2",
            str(root / "tests/fixtures/two_stage/stage2_logits.json"),
            "--report",
            str(tmp_path / "r.json"),
            "--table",
            str(tmp_path / "t.json"),
            "--threshold-scope",
            "global",
            "--threshold",
            "0.9",
            "--min-span",
            "50",
            "--gap-merge",
            "10",
            "--min-distinct-elements",
            "2",
            "--flank",
            "50",
            "--min-order-margin",
            "1",
            "--temperature-from",
            str(root / "reports/gate2_p3_ece.json"),
            "--stage2-operating-point",
            "0.5",
            "--expect-digest",
            "0" * 64,
        ]
    )
    assert exit_code == 1


def test_confirmed_reads_the_named_posterior_at_the_operating_point() -> None:
    """The boundary is ``>=``, and it is the *named* (pre-prior-shift) posterior it reads."""
    contigs, stage1, run = _run(_canonical_spans())
    manifest = T.payload_manifest([run])
    plus = next(e["row_id"] for e in manifest if e["strand"] == "+")
    minus = next(e["row_id"] for e in manifest if e["strand"] == "-")
    # temperature 1.0 makes the posterior exactly sigmoid(z), so z=0 lands exactly on 0.5.
    calibration = {**_CALIBRATION, "temperature": 1.0}
    result = T.run_two_stage(contigs, stage1, {plus: 0.0, minus: 0.0}, **{**_RULE, **calibration})
    assert result.rows[0]["stage2_named_posterior"] == pytest.approx(0.5)
    assert result.rows[0]["confirmed"] is True, "the operating point is inclusive"
    below = T.run_two_stage(
        contigs, stage1, {plus: -1e-6, minus: -1e-6}, **{**_RULE, **calibration}
    )
    assert below.rows[0]["confirmed"] is False


# ══════════════════════════════════════════════════════════════════════════════════════
# Two gaps the sabotage campaign found — both invisible on a one-locus, all-confirmed fixture
# ══════════════════════════════════════════════════════════════════════════════════════
def _two_canonical_loci() -> list[tuple[int, int, str]]:
    """Two separated canonical loci — the gap of 340 nt far exceeds ``gap_merge=10``."""
    return [
        (300, 380, "Stem_I"),
        (380, 460, "Specifier"),
        (460, 560, "Terminator"),
        (900, 980, "Stem_I"),
        (980, 1060, "Specifier"),
        (1060, 1160, "Terminator"),
    ]


def test_invariance_is_a_fraction_of_confirmed_loci_not_of_all_loci() -> None:
    """With every locus confirmed, ``/n_confirmed`` and ``/n_loci`` are the same number.

    Found by sabotage: swapping the denominator changed nothing on a one-locus, all-confirmed
    fixture, so the choice of denominator was untested. Here one of two loci is confirmed and
    the two denominators give 1.0 and 0.5.
    """
    contigs, stage1, run = _run(_two_canonical_loci())
    assert len(run.loci) == 2
    manifest = T.payload_manifest([run])
    logits = {entry["row_id"]: (8.0 if entry["locus_index"] == 0 else -8.0) for entry in manifest}
    diagnostic = T.run_two_stage(contigs, stage1, logits, **{**_RULE, **_CALIBRATION}).report[
        "strand_robustness"
    ]
    assert diagnostic["n_loci"] == 2
    assert diagnostic["n_confirmed_loci"] == 1
    assert diagnostic["n_confirmed_invariant"] == 1
    assert diagnostic["confirmation_invariance"] == 1.0


def test_the_wrong_strand_counter_does_not_fire_when_both_strands_confirm() -> None:
    """A locus confirmed on *both* strands is not a wrong-strand-only confirmation.

    Found by sabotage: relaxing the counter's ``and`` to an ``or`` was invisible while the
    fixtures only ever confirmed one strand, and this is the input that separates them — the
    ``or`` reading counts a both-strand confirmation as a false-novelty case, which is exactly
    backwards (a locus that survives either orientation is the *robust* one).
    """
    contigs, stage1, run = _run(
        _canonical_spans(), truth_strand="+", truth_start=600, truth_end=880
    )
    result = T.run_two_stage(contigs, stage1, _scores([run], 8.0), **{**_RULE, **_CALIBRATION})
    diagnostic = result.report["strand_robustness"]
    assert diagnostic["n_confirmed_on_plus"] == 1
    assert diagnostic["n_confirmed_on_minus"] == 1
    assert diagnostic["truth"]["n_confirmed_on_wrong_strand_only"] == 0


# ══════════════════════════════════════════════════════════════════════════════════════
# Review round 1 — the failure paths that raised untyped errors
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_payloads_leg_names_a_contig_with_no_stage1_logits() -> None:
    """It raised a bare ``KeyError`` where ``run_two_stage`` already raised ``TwoStageError``."""
    windows, starts, sequence = _plant(_canonical_spans())
    args = T.build_parser().parse_args(
        [
            "payloads",
            "--contigs",
            "x",
            "--stage1",
            "y",
            "--out",
            "z",
            "--threshold-scope",
            "global",
            "--threshold",
            "0.9",
            "--min-span",
            "50",
            "--gap-merge",
            "10",
            "--min-distinct-elements",
            "2",
            "--flank",
            "20",
            "--min-order-margin",
            "1",
        ]
    )
    assert args.command == "payloads"
    # The guard itself, driven directly — the leg's own I/O is exercised by the golden.
    with pytest.raises(T.TwoStageError, match="no Stage-1 window logits"):
        T.run_two_stage([_contig("absent", sequence)], {}, {}, **{**_RULE, **_CALIBRATION})


def test_the_cli_refuses_an_unset_rule_knob() -> None:
    """``argparse`` marks every rule knob required, so omission is an error, not a default."""
    for omitted in ("--threshold", "--min-span", "--flank", "--min-order-margin"):
        argv = [
            "payloads",
            "--contigs",
            "x",
            "--stage1",
            "y",
            "--out",
            "z",
            "--threshold-scope",
            "global",
            "--threshold",
            "0.9",
            "--min-span",
            "50",
            "--gap-merge",
            "10",
            "--min-distinct-elements",
            "2",
            "--flank",
            "20",
            "--min-order-margin",
            "1",
        ]
        index = argv.index(omitted)
        with pytest.raises(SystemExit):
            T.build_parser().parse_args(argv[:index] + argv[index + 2 :])
    # Positive control: the complete argv parses.
    assert (
        T.build_parser()
        .parse_args(
            [
                "payloads",
                "--contigs",
                "x",
                "--stage1",
                "y",
                "--out",
                "z",
                "--threshold-scope",
                "global",
                "--threshold",
                "0.9",
                "--min-span",
                "50",
                "--gap-merge",
                "10",
                "--min-distinct-elements",
                "2",
                "--flank",
                "20",
                "--min-order-margin",
                "1",
            ]
        )
        .min_span
        == 50
    )


def test_the_run_leg_requires_exactly_one_temperature_source() -> None:
    base = [
        "run",
        "--contigs",
        "a",
        "--stage1",
        "b",
        "--stage2",
        "c",
        "--report",
        "d",
        "--table",
        "e",
        "--threshold-scope",
        "global",
        "--threshold",
        "0.9",
        "--min-span",
        "50",
        "--gap-merge",
        "10",
        "--min-distinct-elements",
        "2",
        "--flank",
        "20",
        "--min-order-margin",
        "1",
        "--stage2-operating-point",
        "0.5",
    ]
    with pytest.raises(SystemExit):  # neither
        T.build_parser().parse_args(base)
    with pytest.raises(SystemExit):  # both
        T.build_parser().parse_args(base + ["--temperature", "1.0", "--temperature-from", "g"])
    assert T.build_parser().parse_args(base + ["--temperature", "1.25"]).temperature == 1.25


# ══════════════════════════════════════════════════════════════════════════════════════
# Review round 2
# ══════════════════════════════════════════════════════════════════════════════════════
def test_a_digest_mismatch_refuses_to_write(tmp_path: Path) -> None:
    """The guard must not destroy the artifact it guards.

    ``--expect-digest`` exists to protect the committed golden. Writing first and comparing
    second overwrote it and *then* reported the mismatch, so the one run the guard is for was
    the one that clobbered the file (CodeRabbit r2).
    """
    root = Path(__file__).resolve().parents[2]
    report, table = tmp_path / "r.json", tmp_path / "t.json"
    argv = [
        "run",
        "--contigs",
        str(root / "tests/fixtures/two_stage/contigs.json"),
        "--stage1",
        str(root / "tests/fixtures/two_stage/stage1_window_logits.npz"),
        "--stage2",
        str(root / "tests/fixtures/two_stage/stage2_logits.json"),
        "--report",
        str(report),
        "--table",
        str(table),
        "--threshold-scope",
        "global",
        "--threshold",
        "0.9",
        "--min-span",
        "50",
        "--gap-merge",
        "10",
        "--min-distinct-elements",
        "2",
        "--flank",
        "50",
        "--min-order-margin",
        "1",
        "--temperature-from",
        str(root / "reports/gate2_p3_ece.json"),
        "--stage2-operating-point",
        "0.5",
    ]
    assert T.main(argv + ["--expect-digest", "0" * 64]) == 1
    assert not report.exists(), "a mismatching run must leave the artifacts untouched"
    assert not table.exists()
    # Positive control: the same call with the right digest writes both.
    expected = (root / "tests/fixtures/two_stage/expected.sha256").read_text().strip()
    assert T.main(argv + ["--expect-digest", expected]) == 0
    assert report.is_file() and table.is_file()


def test_parse_threshold_accepts_a_scalar_and_a_per_class_mapping() -> None:
    assert T.parse_threshold("0.9") == 0.9
    assert T.parse_threshold(' {"Stem_I": 0.5, "Specifier": 0.7} ') == {
        "Stem_I": 0.5,
        "Specifier": 0.7,
    }
    for bad in ("[1,2]", "{not json}", "abc"):
        with pytest.raises(Exception):  # noqa: B017 - argparse.ArgumentTypeError or ValueError
            T.parse_threshold(bad)


def test_the_cli_can_express_the_per_class_scope() -> None:
    """``per_class`` was unreachable from the CLI: ``--threshold`` parsed only a float."""
    from tbox_finder.infer.locus import ELEMENT_CLASS_NAMES

    mapping = json.dumps(dict.fromkeys(ELEMENT_CLASS_NAMES, 0.9))
    args = T.build_parser().parse_args(
        [
            "payloads",
            "--contigs",
            "x",
            "--stage1",
            "y",
            "--out",
            "z",
            "--threshold-scope",
            "per_class",
            "--threshold",
            mapping,
            "--min-span",
            "50",
            "--gap-merge",
            "10",
            "--min-distinct-elements",
            "2",
            "--flank",
            "20",
            "--min-order-margin",
            "1",
        ]
    )
    assert args.threshold == dict.fromkeys(ELEMENT_CLASS_NAMES, 0.9)
    # ...and it runs end to end, which a float would not have.
    contigs, stage1, run = _run(_canonical_spans())
    result = T.run_two_stage(
        contigs,
        stage1,
        _scores([run]),
        **{**_RULE, "threshold_scope": "per_class", "threshold": args.threshold},
        **_CALIBRATION,
    )
    assert len(result.rows) == 1


def test_the_manifest_is_ordered_by_contig_regardless_of_input_order() -> None:
    """``handoff_loci`` fixes locus and strand order; nothing fixed contig order (r2)."""
    windows, starts, sequence = _plant(_canonical_spans())
    stage1 = {name: {"logits": windows, "starts": starts} for name in ("b", "a")}
    runs = [
        T.run_contig(
            _contig(name, sequence), T.reconcile_contig(stage1[name], len(sequence)), **_RULE
        )
        for name in ("b", "a")
    ]
    contigs = [entry["contig_id"] for entry in T.payload_manifest(runs)]
    assert contigs == sorted(contigs)
    result = T.run_two_stage(
        [_contig("b", sequence), _contig("a", sequence)],
        stage1,
        _scores(runs),
        **{**_RULE, **_CALIBRATION},
    )
    assert [row["contig_id"] for row in result.rows] == ["a", "b"]


# ══════════════════════════════════════════════════════════════════════════════════════
# Review round 3 — the one composition failure that produced a number instead of an error
# ══════════════════════════════════════════════════════════════════════════════════════
def test_a_duplicate_contig_id_is_refused() -> None:
    """Reproduced before it was fixed: two records sharing an id merged, silently.

    ``contig_id`` keys the payload hash, the diagnostic's per-locus map and the Stage-1 lookup,
    so a duplicate does not collide loudly — it overwrites. And nothing downstream sees it:
    ``n_loci`` and ``n_scored_payloads`` are both summed per run, so ``n_scored == 2 × n_loci``
    still held and ``strand_robustness_problems`` returned ``[]`` on the merged result.
    """
    windows, starts, sequence = _plant(_canonical_spans())
    contigs = [_contig("dup", sequence), _contig("dup", sequence[::-1])]
    stage1 = {"dup": {"logits": windows, "starts": starts}}
    with pytest.raises(T.TwoStageError, match="appears more than once"):
        T.normalise_contigs(contigs)
    with pytest.raises(T.TwoStageError, match="appears more than once"):
        T.run_two_stage(contigs, stage1, {}, **{**_RULE, **_CALIBRATION})


def test_distinct_contig_ids_are_accepted() -> None:
    """Positive control: the identical call with distinct ids runs and yields two rows."""
    windows, starts, sequence = _plant(_canonical_spans())
    contigs = [_contig("a", sequence), _contig("b", sequence)]
    stage1 = {
        "a": {"logits": windows, "starts": starts},
        "b": {"logits": windows, "starts": starts},
    }
    assert [c["contig_id"] for c in T.normalise_contigs(contigs)] == ["a", "b"]
    runs = [
        T.run_contig(c, T.reconcile_contig(stage1[c["contig_id"]], len(sequence)), **_RULE)
        for c in contigs
    ]
    result = T.run_two_stage(contigs, stage1, _scores(runs), **{**_RULE, **_CALIBRATION})
    assert len(result.rows) == 2
    assert len({row["payload_key"] for row in result.rows}) == 2


def test_the_committed_fixture_has_unique_contig_ids() -> None:
    root = Path(__file__).resolve().parents[2]
    contigs = T.read_contigs(root / "tests/fixtures/two_stage/contigs.json")
    ids = [c["contig_id"] for c in contigs]
    assert len(set(ids)) == len(ids) == 20


# ══════════════════════════════════════════════════════════════════════════════════════
# P3-16′-remint — Stage-2 identity, shards and the merge (ADR-0005 A15 Pin 6)
# ══════════════════════════════════════════════════════════════════════════════════════
# The manifest and the logits below are REAL: the committed P3-14 fixture's 20 contigs through
# the committed Stage-1 window logits at the golden's rule give the work list, and the
# committed `stage2_logits.json` is the production arm's measured output for exactly that list.
# Only the loader *record* is a declared stand-in (no checkpoint is loaded in a torch-free
# test); its synthetic fields are named as such where they are built.
_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "two_stage"
_GOLDEN_RULE = {**_RULE, "flank": 50}
_RINALMO_ROOT = "data/processed/checkpoints/stage2_rinalmo"
_RNAFM_ROOT = "data/processed/checkpoints/stage2_rnafm"
_ARM = "aux1.0_lr1e-4"
#: Declared synthetic: no heads file is read in this tier, so its digest is a stated stand-in.
_SYNTHETIC_HEADS_SHA = hashlib.sha256(b"declared synthetic stage2_heads.pt").hexdigest()


@functools.lru_cache(maxsize=1)
def _fixture_manifest_json() -> str:
    contigs = T.read_contigs(_FIXTURE / "contigs.json")
    stage1 = T.read_stage1(_FIXTURE / "stage1_window_logits.npz")
    runs = [
        T.run_contig(
            contig,
            T.reconcile_contig(stage1[contig["contig_id"]], len(contig["sequence"])),
            **_GOLDEN_RULE,
        )
        for contig in contigs
    ]
    return json.dumps(T.payload_manifest(runs))


def _manifest() -> list[dict]:
    """A fresh copy every call, so a test that mutates it cannot leak into the next."""
    return json.loads(_fixture_manifest_json())


def _bytes(payload) -> bytes:
    """The bytes :func:`T.write_json` would write, without touching disk."""
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _manifest_bytes(manifest=None) -> bytes:
    return _bytes(
        {"schema_version": T.SCHEMA_VERSION, "step": T.STEP, "payloads": manifest or _manifest()}
    )


@functools.lru_cache(maxsize=1)
def _fixture_stage2_json() -> str:
    return (_FIXTURE / "stage2_logits.json").read_text(encoding="utf-8")


def _scored(rows) -> list[dict]:
    """What ``stage2.eval.score_rows`` returned for these rows, read from the committed file."""
    committed = json.loads(_fixture_stage2_json())
    return [
        {
            "row_id": row["row_id"],
            "tbox_logit": committed["logits"][row["row_id"]],
            "n_tokens": committed["n_tokens"][row["row_id"]],
        }
        for row in rows
    ]


def _record(backbone: str = "rinalmo-giga") -> dict:
    """A stand-in for ``load_stage2_checkpoint``'s record, keyed exactly as the loader keys it.

    ``backbone`` / ``base_model_name_or_path`` / ``revision`` come from the ADR-0002 A15
    allow-list. ``adapter_sha256`` and ``n_lora_b_nonzero`` are the committed fixture's
    measured values. ``heads_sha256`` is declared synthetic.
    """
    spec = BR.resolve_backbone(backbone)
    committed = json.loads(_fixture_stage2_json())
    return {
        "backbone": spec.key,
        "base_model_name_or_path": spec.repo_id,
        "revision": spec.revision,
        "dtype": "bfloat16",
        "adapter_sha256": committed["adapter_sha256"],
        "heads_sha256": _SYNTHETIC_HEADS_SHA,
        "n_lora_b_nonzero": committed["n_lora_b_nonzero"],
        "attn_implementation": "flash_attention_2",
    }


def _identity(manifest_raw: bytes | None = None, **overrides) -> dict:
    kwargs = {
        "arm": _ARM,
        "arm_selected_by": "stage2.eval.production_arm_config()",
        "checkpoint_root": _RINALMO_ROOT,
        "checkpoint_dir": f"{_RINALMO_ROOT}/{_ARM}",
        "record": _record(),
        "trained_under": "flash_attention_2",
        "device": "cuda",
        "batch_size": 4,
        "payloads_sha256": hashlib.sha256(manifest_raw or _manifest_bytes()).hexdigest(),
    }
    kwargs.update(overrides)
    return T.stage2_identity(**kwargs)


def _shard_artifacts(count: int, manifest=None) -> list[dict]:
    manifest = manifest or _manifest()
    identity = _identity(_manifest_bytes(manifest))
    out = []
    for index in range(count):
        rows = T.shard_manifest(manifest, index, count)
        out.append(
            T.stage2_artifact(identity, payloads=rows, scored=_scored(rows), shard=(index, count))
        )
    return out


def _merge(artifacts, manifest=None) -> dict:
    return T.merge_stage2_shards(_manifest_bytes(manifest), [_bytes(a) for a in artifacts])


# ── the fixture is not degenerate ─────────────────────────────────────────────────────────
def test_the_shard_fixture_manifest_is_not_already_in_row_id_order() -> None:
    """Guards the guard: a row-id-sorted fixture would make dropping the sort invisible
    ([[sorted-fixture-hides-the-sort]]). ``payload_manifest`` orders by (contig, locus,
    strand), and the ids are content hashes, so the two orders differ."""
    ids = [entry["row_id"] for entry in _manifest()]
    assert len(ids) == 54
    assert ids != sorted(ids)


# ── shard_bounds / shard_manifest ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("n_rows", [0, 1, 5, 54, 55])
@pytest.mark.parametrize("count", [1, 2, 3, 7, 60])
def test_shard_bounds_are_contiguous_balanced_and_cover_everything(n_rows, count) -> None:
    bounds = [T.shard_bounds(n_rows, index, count) for index in range(count)]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n_rows
    for (_, stop), (start, _) in zip(bounds, bounds[1:], strict=False):
        assert start == stop, "chunks must be contiguous"
    sizes = [stop - start for start, stop in bounds]
    assert max(sizes) - min(sizes) <= 1
    # ...and the extra rows go to the FIRST `n_rows % count` chunks, not anywhere.
    extra = n_rows % count
    assert sizes == [n_rows // count + 1] * extra + [n_rows // count] * (count - extra)


def test_shards_partition_the_manifest() -> None:
    """The property `merge_stage2_shards` relies on in place of a separate union clause."""
    manifest = _manifest()
    ids = sorted(entry["row_id"] for entry in manifest)
    for count in (1, 2, 3, 5, 54):
        chunks = [
            [entry["row_id"] for entry in T.shard_manifest(manifest, index, count)]
            for index in range(count)
        ]
        assert [row_id for chunk in chunks for row_id in chunk] == ids
        assert max(map(len, chunks)) - min(map(len, chunks)) <= 1


def test_shard_assignment_is_independent_of_input_order() -> None:
    manifest = _manifest()
    rng = np.random.default_rng(42)
    shuffled = [manifest[i] for i in rng.permutation(len(manifest))]
    assert [e["row_id"] for e in shuffled] != [e["row_id"] for e in manifest]
    for order in (shuffled, manifest[::-1]):
        for index in range(3):
            assert T.shard_manifest(order, index, 3) == T.shard_manifest(manifest, index, 3)


def test_a_shard_carries_whole_payload_entries() -> None:
    """The slice is of payloads, not of ids: every field the scorer reads rides along."""
    manifest = _manifest()
    by_id = {entry["row_id"]: entry for entry in manifest}
    for entry in T.shard_manifest(manifest, 1, 4):
        assert entry == by_id[entry["row_id"]]


@pytest.mark.parametrize(
    ("index", "count"), [(3, 3), (-1, 3), (0, 0), (True, 2), (0, True), (0.0, 2), (1, 2.0)]
)
def test_shard_manifest_refuses_an_impossible_shard(index, count) -> None:
    with pytest.raises(T.TwoStageError, match="shard (index|count)"):
        T.shard_manifest(_manifest(), index, count)


def test_shard_manifest_refuses_a_duplicate_row_id() -> None:
    manifest = _manifest()
    manifest.append(dict(manifest[0]))
    with pytest.raises(T.TwoStageError, match="appears more than once"):
        T.shard_manifest(manifest, 0, 2)
    with pytest.raises(T.TwoStageError, match="appears more than once"):
        T.manifest_row_ids(manifest)
    # Positive control: without the duplicate the same call succeeds.
    assert T.shard_manifest(manifest[:-1], 0, 2)


@pytest.mark.parametrize("bad", [None, "", 7, "a\nb"])
def test_manifest_row_ids_refuses_an_unusable_row_id(bad) -> None:
    manifest = _manifest()
    manifest[3]["row_id"] = bad
    with pytest.raises(T.TwoStageError, match="no usable row_id"):
        T.manifest_row_ids(manifest)


def test_row_ids_sha256_is_order_free_and_newline_framed() -> None:
    assert T.row_ids_sha256(["b", "a"]) == T.row_ids_sha256(["a", "b"])
    assert T.row_ids_sha256(["a", "b"]) == hashlib.sha256(b"a\nb\n").hexdigest()
    assert T.row_ids_sha256(["ab"]) != T.row_ids_sha256(["a", "b"])


# ── --shard parsing ───────────────────────────────────────────────────────────────────────
def test_parse_shard_accepts_i_of_n() -> None:
    assert T.parse_shard("0/4") == (0, 4)
    assert T.parse_shard(" 3 / 4 ") == (3, 4)
    assert T.parse_shard("0/1") == (0, 1)


@pytest.mark.parametrize("text", ["4/4", "1/0", "0/0", "-1/3", "a/b", "1", "1/2/3", "1.0/2"])
def test_parse_shard_refuses_anything_else(text) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        T.parse_shard(text)


def test_the_stage2_leg_takes_a_shard_and_defaults_to_none() -> None:
    base = [
        "stage2",
        "--payloads",
        "p",
        "--checkpoint-root",
        "r",
        "--sweep-dir",
        "s",
        "--out",
        "o",
    ]
    assert T.build_parser().parse_args(base).shard is None
    assert T.build_parser().parse_args(base + ["--shard", "1/3"]).shard == (1, 3)
    with pytest.raises(SystemExit):
        T.build_parser().parse_args(base + ["--shard", "3/3"])


# ── env_lock derived from the backbone ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("backbone", "lock"),
    [
        ("rinalmo-giga", "envs/ml-rna.conda-lock.yml"),
        ("multimolecule/rinalmo-giga", "envs/ml-rna.conda-lock.yml"),
        ("rnafm", "envs/ml-rnafm.conda-lock.yml"),
        ("multimolecule/rnafm", "envs/ml-rnafm.conda-lock.yml"),
    ],
)
def test_the_env_lock_is_derived_from_the_backbone(backbone, lock) -> None:
    """ADR-0002 A15: the comparator runs under a lock the shipped arm's env cannot replace."""
    assert T.stage2_env_lock(backbone) == lock


@pytest.mark.parametrize(
    "backbone", ["multimolecule/rinalmo-mega", "RINALMO-GIGA", "", None, "ml-rna"]
)
def test_an_unknown_backbone_has_no_env_lock(backbone) -> None:
    with pytest.raises(T.TwoStageError, match="not in the ADR-0002 A15 allow-list"):
        T.stage2_env_lock(backbone)


def test_the_identity_names_the_comparators_lock_for_an_rnafm_checkpoint() -> None:
    """The pre-A15 leg stamped ``envs/ml-rna.conda-lock.yml`` whatever it loaded."""
    rnafm = _identity(
        checkpoint_root=_RNAFM_ROOT,
        checkpoint_dir=f"{_RNAFM_ROOT}/{_ARM}",
        record=_record("rnafm"),
    )
    assert rnafm["backbone"] == "rnafm"
    assert rnafm["env_lock"] == "envs/ml-rnafm.conda-lock.yml"
    assert _identity()["env_lock"] == "envs/ml-rna.conda-lock.yml"


# ── stage2_identity ───────────────────────────────────────────────────────────────────────
def test_a_clean_identity_records_every_field_from_the_loader_record() -> None:
    identity = _identity()
    record = _record()
    for field in ("backbone", "revision", "dtype", "adapter_sha256", "heads_sha256"):
        assert identity[field] == record[field], field
    assert identity["schema_version"] == T.STAGE2_SCHEMA_VERSION == "1.1"
    assert identity["checkpoint_root"] == _RINALMO_ROOT
    assert identity["attn_implementation_matches_training"] is True
    assert set(T.STAGE2_IDENTITY_FIELDS) | set(T.STAGE2_RUN_FIELDS) == set(identity)


def _without(field: str) -> dict:
    record = _record()
    record[field] = None
    return record


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"record": {**_record(), "backbone": "rinalmo-mega"}}, "allow-list"),
        ({"record": _without("heads_sha256")}, "'heads_sha256' is None"),
        ({"record": _without("revision")}, "'revision' is None"),
        ({"record": _without("dtype")}, "'dtype' is None"),
        ({"record": {**_record(), "adapter_sha256": "abc"}}, "adapter_sha256 is not a sha256"),
        ({"payloads_sha256": "0" * 63}, "payloads_sha256 is not a sha256"),
        ({"batch_size": 0}, "batch_size must be an integer"),
        ({"batch_size": True}, "batch_size must be an integer"),
        ({"checkpoint_root": "/home/someone/ckpt"}, "repo-relative"),
        ({"checkpoint_root": "../ckpt"}, "repo-relative"),
        ({"checkpoint_dir": "~/ckpt/" + _ARM}, "repo-relative"),
        ({"arm": ""}, "'arm' is ''"),
        # The P3-17 trap: RNA-FM's root holds an arm with the production arm's NAME.
        ({"checkpoint_root": _RNAFM_ROOT}, "is not checkpoint_root/arm"),
        ({"record": {**_record(), "base_model_name_or_path": "multimolecule/rnafm"}}, "base"),
    ],
)
def test_stage2_identity_refuses(overrides, match) -> None:
    with pytest.raises(T.TwoStageError, match=match):
        _identity(**overrides)


# ── stage2_artifact ───────────────────────────────────────────────────────────────────────
def test_an_unsharded_artifact_has_a_null_shard_and_layout() -> None:
    manifest = _manifest()
    artifact = T.stage2_artifact(
        _identity(), payloads=manifest, scored=_scored(manifest), shard=None
    )
    assert set(artifact) == set(T.STAGE2_ARTIFACT_KEYS)
    assert artifact["shard"] is None and artifact["shard_layout"] is None
    assert artifact["n_payloads"] == len(manifest) == len(artifact["logits"])


def test_a_shard_artifact_records_its_slice() -> None:
    manifest = _manifest()
    rows = T.shard_manifest(manifest, 2, 4)
    artifact = T.stage2_artifact(_identity(), payloads=rows, scored=_scored(rows), shard=(2, 4))
    assert artifact["shard"] == {
        "index": 2,
        "count": 4,
        "n_payloads": len(rows),
        "row_ids_sha256": T.row_ids_sha256(row["row_id"] for row in rows),
    }
    assert set(artifact["logits"]) == {row["row_id"] for row in rows}


def test_stage2_artifact_refuses_scores_that_are_not_the_work_list() -> None:
    manifest = _manifest()
    rows = T.shard_manifest(manifest, 0, 2)
    scored = _scored(rows)
    with pytest.raises(T.TwoStageError, match="unscored"):
        T.stage2_artifact(_identity(), payloads=rows, scored=scored[:-1], shard=(0, 2))
    with pytest.raises(T.TwoStageError, match="twice"):
        T.stage2_artifact(_identity(), payloads=rows, scored=scored + scored[:1], shard=(0, 2))
    for bad in (float("nan"), True, "1.0"):
        tampered = [dict(row) for row in scored]
        tampered[0]["tbox_logit"] = bad
        with pytest.raises(T.TwoStageError, match="not (finite|a number)"):
            T.stage2_artifact(_identity(), payloads=rows, scored=tampered, shard=(0, 2))
    # Positive control.
    assert T.stage2_artifact(_identity(), payloads=rows, scored=scored, shard=(0, 2))


# ── merge_stage2_shards ───────────────────────────────────────────────────────────────────
def test_a_clean_merge_reassembles_every_logit() -> None:
    manifest = _manifest()
    artifacts = _shard_artifacts(3, manifest)
    merged = _merge(artifacts, manifest)
    committed = json.loads(_fixture_stage2_json())
    ids = {entry["row_id"] for entry in manifest}
    assert set(merged["logits"]) == ids
    assert merged["logits"] == {row_id: committed["logits"][row_id] for row_id in ids}
    assert merged["n_tokens"] == {row_id: committed["n_tokens"][row_id] for row_id in ids}
    assert merged["n_payloads"] == len(ids)
    assert merged["shard"] is None
    layout = merged["shard_layout"]
    assert layout["count"] == 3
    assert [s["index"] for s in layout["shards"]] == [0, 1, 2]
    for shard, artifact in zip(layout["shards"], artifacts, strict=True):
        assert shard["file_sha256"] == hashlib.sha256(_bytes(artifact)).hexdigest()
        assert shard["row_ids_sha256"] == artifact["shard"]["row_ids_sha256"]
        assert shard["n_payloads"] == artifact["shard"]["n_payloads"]


def test_the_merge_does_not_depend_on_the_order_shards_are_given() -> None:
    manifest = _manifest()
    artifacts = _shard_artifacts(3, manifest)
    reordered = T.merge_stage2_shards(
        _manifest_bytes(manifest), [_bytes(artifacts[i]) for i in (2, 0, 1)]
    )
    assert reordered == _merge(artifacts, manifest)


def test_a_merged_artifact_has_the_shape_of_an_unsharded_one() -> None:
    manifest = _manifest()
    merged = _merge(_shard_artifacts(4, manifest), manifest)
    unsharded = T.stage2_artifact(
        _identity(), payloads=manifest, scored=_scored(manifest), shard=None
    )
    assert set(merged) == set(unsharded) == set(T.STAGE2_ARTIFACT_KEYS)
    for field in (*T.STAGE2_IDENTITY_FIELDS, *T.STAGE2_RUN_FIELDS):
        assert merged[field] == unsharded[field], field


def test_a_single_shard_merges() -> None:
    manifest = _manifest()
    merged = _merge(_shard_artifacts(1, manifest), manifest)
    assert merged["shard_layout"]["count"] == 1
    assert len(merged["logits"]) == len(manifest)


def test_the_merge_refuses_a_missing_index() -> None:
    artifacts = _shard_artifacts(3)
    with pytest.raises(T.TwoStageError, match=r"index\(es\) \[1\] of 3 are missing"):
        _merge([artifacts[0], artifacts[2]])


def test_the_merge_refuses_a_duplicated_index() -> None:
    artifacts = _shard_artifacts(3)
    with pytest.raises(T.TwoStageError, match=r"index\(es\) \[1\] were given more than once"):
        _merge([artifacts[0], artifacts[1], artifacts[1], artifacts[2]])
    # Even when the duplicate makes the COUNT of files match: [0, 1, 1] of 3.
    with pytest.raises(T.TwoStageError, match="more than once"):
        _merge([artifacts[0], artifacts[1], artifacts[1]])


@pytest.mark.parametrize("stray", [2, -1])
def test_the_merge_refuses_an_index_outside_the_count(stray) -> None:
    """Indices {0, 1} are complete for count 2, so neither 'missing' nor 'twice' fires."""
    artifacts = _shard_artifacts(2)
    forged = json.loads(json.dumps(artifacts[1]))
    forged["shard"]["index"] = stray
    with pytest.raises(T.TwoStageError, match=rf"\[{stray}\] lie outside \[0, 2\)"):
        _merge([*artifacts, forged])


def test_the_merge_refuses_disagreeing_shard_counts() -> None:
    artifacts = _shard_artifacts(3)
    artifacts[1]["shard"]["count"] = 4
    with pytest.raises(T.TwoStageError, match="different shard counts"):
        _merge(artifacts)


def _perturbed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    return f"{value}x"


@pytest.mark.parametrize(
    "field",
    [
        field
        for field in (*T.STAGE2_IDENTITY_FIELDS, *T.STAGE2_RUN_FIELDS)
        # These two have their own, earlier refusals (tested below).
        if field not in ("payloads_sha256", "schema_version")
    ],
)
def test_the_merge_refuses_shards_that_are_not_one_measurement(field) -> None:
    artifacts = _shard_artifacts(3)
    artifacts[2][field] = _perturbed(artifacts[2][field])
    with pytest.raises(T.TwoStageError, match=f"disagree on '{field}'"):
        _merge(artifacts)


def test_the_merge_compares_identity_by_type_not_only_by_value() -> None:
    """``4 == 4.0`` and ``1 == True`` in Python; as a record of a run they are not the same."""
    artifacts = _shard_artifacts(2)
    artifacts[1]["batch_size"] = 4.0
    with pytest.raises(T.TwoStageError, match="disagree on 'batch_size'"):
        _merge(artifacts)


def test_the_merge_refuses_an_identity_field_null_in_every_shard() -> None:
    """All-null agrees with itself; the clause must not pass on the absence of evidence."""
    artifacts = _shard_artifacts(2)
    for artifact in artifacts:
        artifact["heads_sha256"] = None
    with pytest.raises(T.TwoStageError, match="'heads_sha256' as null"):
        _merge(artifacts)


def test_the_merge_refuses_a_shard_holding_rows_its_index_was_not_assigned() -> None:
    """The union is still the whole manifest here; only the assignment is wrong."""
    artifacts = _shard_artifacts(2)
    a, b = artifacts
    moved_a = sorted(a["logits"])[0]
    moved_b = sorted(b["logits"])[0]
    for name in ("logits", "n_tokens"):
        a[name][moved_b] = b[name].pop(moved_b)
        b[name][moved_a] = a[name].pop(moved_a)
    union = set(a["logits"]) | set(b["logits"])
    assert union == {entry["row_id"] for entry in _manifest()}
    with pytest.raises(T.TwoStageError, match="keys are not the rows its index assigns"):
        _merge(artifacts)


def test_the_merge_refuses_a_shard_missing_one_of_its_rows() -> None:
    artifacts = _shard_artifacts(3)
    victim = sorted(artifacts[1]["logits"])[0]
    del artifacts[1]["logits"][victim]
    with pytest.raises(T.TwoStageError, match=r"logits keys .* \(1 assigned rows absent"):
        _merge(artifacts)


def test_the_merge_refuses_shards_cut_from_another_manifest() -> None:
    """``payloads_sha256`` is over the manifest's BYTES: re-serialising it changes the hash."""
    manifest = _manifest()
    artifacts = _shard_artifacts(2, manifest)
    reserialised = json.dumps({"payloads": manifest}).encode()
    assert json.loads(reserialised)["payloads"] == manifest
    with pytest.raises(T.TwoStageError, match="different work list"):
        T.merge_stage2_shards(reserialised, [_bytes(a) for a in artifacts])
    tampered = [dict(a) for a in artifacts]
    tampered[0]["payloads_sha256"] = "0" * 64
    with pytest.raises(T.TwoStageError, match="different work list"):
        _merge(tampered, manifest)


def test_the_merge_refuses_a_row_ids_sha256_that_does_not_rederive() -> None:
    artifacts = _shard_artifacts(2)
    artifacts[0]["shard"]["row_ids_sha256"] = "0" * 64
    with pytest.raises(T.TwoStageError, match="does not re-derive"):
        _merge(artifacts)


def test_the_merge_refuses_a_stated_n_payloads_that_does_not_rederive() -> None:
    artifacts = _shard_artifacts(2)
    artifacts[1]["shard"]["n_payloads"] += 1
    with pytest.raises(T.TwoStageError, match="states n_payloads"):
        _merge(artifacts)


def test_the_merge_refuses_a_non_finite_or_boolean_logit() -> None:
    for bad in (None, True):
        artifacts = _shard_artifacts(2)
        artifacts[0]["logits"][sorted(artifacts[0]["logits"])[0]] = bad
        with pytest.raises(T.TwoStageError, match="not a number"):
            _merge(artifacts)


def test_the_merge_refuses_a_schema_1_0_file() -> None:
    """The committed P3-14 fixture records no identity, so it cannot be certified."""
    with pytest.raises(T.TwoStageError, match="is schema '1.0'"):
        T.merge_stage2_shards(_manifest_bytes(), [_fixture_stage2_json().encode()])


def test_the_merge_refuses_an_unsharded_or_merged_artifact_as_a_shard() -> None:
    manifest = _manifest()
    unsharded = T.stage2_artifact(
        _identity(), payloads=manifest, scored=_scored(manifest), shard=None
    )
    with pytest.raises(T.TwoStageError, match="not one shard"):
        _merge([unsharded], manifest)
    merged = _merge(_shard_artifacts(2, manifest), manifest)
    as_shard = {**merged, "shard": {"index": 0, "count": 1, "n_payloads": 54}}
    with pytest.raises(T.TwoStageError, match="already a merged artifact"):
        _merge([as_shard], manifest)


def test_the_merge_refuses_no_shards() -> None:
    with pytest.raises(T.TwoStageError, match="no shard files"):
        T.merge_stage2_shards(_manifest_bytes(), [])


# ── read_stage2 across schema versions ────────────────────────────────────────────────────
def test_read_stage2_still_reads_the_committed_schema_1_0_file() -> None:
    raw = json.loads(_fixture_stage2_json())
    assert raw["schema_version"] == "1.0"
    assert "shard" not in raw and "backbone" not in raw
    logits = T.read_stage2(_FIXTURE / "stage2_logits.json")
    assert logits == {key: float(value) for key, value in raw["logits"].items()}
    assert len(logits) == 54


def test_read_stage2_reads_unsharded_and_merged_1_1_files(tmp_path: Path) -> None:
    manifest = _manifest()
    unsharded = T.stage2_artifact(
        _identity(), payloads=manifest, scored=_scored(manifest), shard=None
    )
    merged = _merge(_shard_artifacts(3, manifest), manifest)
    T.write_json(tmp_path / "u.json", unsharded)
    T.write_json(tmp_path / "m.json", merged)
    expected = T.read_stage2(_FIXTURE / "stage2_logits.json")
    assert T.read_stage2(tmp_path / "u.json") == expected
    assert T.read_stage2(tmp_path / "m.json") == expected


def test_read_stage2_refuses_a_single_shard(tmp_path: Path) -> None:
    shard = _shard_artifacts(3)[1]
    T.write_json(tmp_path / "s.json", shard)
    with pytest.raises(T.TwoStageError, match="shard 1 of 3 .* not a complete one"):
        T.read_stage2(tmp_path / "s.json")


# ── A15 Pin 5: no v2 artifact is written to a v1 path ─────────────────────────────────────
@pytest.mark.parametrize(
    "relative",
    [
        "data/interim/p3_16/v1_stage2_twin.json",
        "data/interim/p3_16/v1_stage1_twin.npz",
        "data/interim/p3_16/v1_payloads_twin_t0.5.json",
        "data/interim/p3_16/v1_stage2_production.json",
        "reports/two_stage_precision.json",
    ],
)
def test_a_v1_path_is_refused(tmp_path: Path, relative) -> None:
    with pytest.raises(T.TwoStageError, match="refusing to write"):
        T.refuse_v1_output(tmp_path / relative)


def test_a_v2_path_is_not_refused(tmp_path: Path) -> None:
    for relative in ("data/interim/p3_16/v2_stage2_twin.json", "data/interim/p3_16/x.json"):
        T.refuse_v1_output(tmp_path / relative)


@pytest.mark.parametrize(
    "relative",
    [
        "reports/two_stage_precision.json",
        "reports/p3/two_stage_precision_items.json",
        "data/interim/p3_16/v1_stage2_twin.json",
    ],
)
def test_the_replay_leg_refuses_a_v1_destination_for_EITHER_of_its_two_outputs(
    tmp_path: Path, relative
) -> None:
    """A15 Pin 5 at the entrypoint, not at the write.

    ``run`` writes two paths, so guarding one leaves the other open
    ([[two-outputs-one-path-destroys-the-first]]). The guard also has to fire BEFORE the replay:
    the five sibling entrypoints already check at entry, and these were the unguarded ones
    ([[fixed-one-of-two-identical-things]]).
    """
    safe = str(tmp_path / "reports/v2_ok.json")
    for report, table in ((str(tmp_path / relative), safe), (safe, str(tmp_path / relative))):
        with pytest.raises(T.TwoStageError, match="refusing to write"):
            T._cmd_run(SimpleNamespace(report=report, table=table))
    # Positive control: with both outputs on v2 paths the guard must let the leg through, or
    # a guard that refused everything would satisfy the cases above
    # ([[raises-test-needs-a-positive-control]]).
    with pytest.raises(AttributeError):
        T._cmd_run(SimpleNamespace(report=safe, table=safe))


# ── the merge-stage2 CLI ──────────────────────────────────────────────────────────────────
def _write_merge_inputs(tmp_path: Path, count: int = 3) -> tuple[Path, list[Path]]:
    manifest = _manifest()
    payloads = tmp_path / "v2_payloads_twin_t0.5.json"
    payloads.write_bytes(_manifest_bytes(manifest))
    shards = []
    for artifact in _shard_artifacts(count, manifest):
        path = tmp_path / f"v2_stage2_twin.shard{artifact['shard']['index']}.json"
        T.write_json(path, artifact)
        shards.append(path)
    return payloads, shards


def test_the_merge_leg_writes_a_readable_artifact(tmp_path: Path) -> None:
    payloads, shards = _write_merge_inputs(tmp_path)
    out = tmp_path / "v2_stage2_twin.json"
    argv = ["merge-stage2", "--payloads", str(payloads), "--out", str(out), "--shards"]
    assert T.main(argv + [str(path) for path in shards[::-1]]) == 0
    merged = json.loads(out.read_text())
    assert merged["shard"] is None
    assert [s["file_sha256"] for s in merged["shard_layout"]["shards"]] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in shards
    ]
    assert T.read_stage2(out) == T.read_stage2(_FIXTURE / "stage2_logits.json")


def test_the_merge_leg_refuses_to_overwrite_one_of_its_inputs(tmp_path: Path) -> None:
    payloads, shards = _write_merge_inputs(tmp_path)
    before = shards[0].read_bytes()
    argv = ["merge-stage2", "--payloads", str(payloads), "--out", str(shards[0]), "--shards"]
    with pytest.raises(T.TwoStageError, match="one of the merge's own inputs"):
        T.main(argv + [str(path) for path in shards])
    assert shards[0].read_bytes() == before


def test_the_merge_leg_refuses_a_v1_output_path(tmp_path: Path) -> None:
    payloads, shards = _write_merge_inputs(tmp_path)
    out = tmp_path / "data" / "interim" / "p3_16" / "v1_stage2_twin.json"
    argv = ["merge-stage2", "--payloads", str(payloads), "--out", str(out), "--shards"]
    with pytest.raises(T.TwoStageError, match="refusing to write"):
        T.main(argv + [str(path) for path in shards])
    assert not out.exists()


# ──────────────────────────────────────────────────────────────────────────────
# payloads-union: one Stage-2 work list for a τ sweep (P3-16′-remint)
# ──────────────────────────────────────────────────────────────────────────────
def test_the_union_deduplicates_by_row_id_and_is_independent_of_input_order():
    manifest = _manifest()
    assert len(manifest) > 10
    first, second = manifest[:30], manifest[20:]  # 10 rows shared
    assert [row["row_id"] for row in manifest] != sorted(row["row_id"] for row in manifest)
    union = T.union_payload_manifests([first, second])
    assert [row["row_id"] for row in union] == sorted(row["row_id"] for row in manifest)
    assert {row["row_id"]: row for row in union} == {row["row_id"]: row for row in manifest}
    assert T.union_payload_manifests([second, first]) == union


def test_the_union_refuses_one_row_id_carried_with_two_payloads():
    manifest = _manifest()
    clash = copy.deepcopy(manifest[:1])
    clash[0]["rna_sequence"] = clash[0]["rna_sequence"][:-1] + (
        "A" if clash[0]["rna_sequence"][-1] != "A" else "C"
    )
    with pytest.raises(T.TwoStageError, match="two different payloads"):
        T.union_payload_manifests([manifest, clash])
    with pytest.raises(T.TwoStageError, match="empty"):
        T.union_payload_manifests([[], []])


def test_the_union_leg_records_its_inputs_and_refuses_unsafe_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = _manifest()
    Path("a.json").write_bytes(_manifest_bytes(manifest[:30]))
    Path("b.json").write_bytes(_manifest_bytes(manifest[20:]))
    parser = T.build_parser()
    args = parser.parse_args(
        ["payloads-union", "--manifests", "a.json", "b.json", "--out", "union.json"]
    )
    assert args.func(args) == 0
    written = json.loads(Path("union.json").read_text(encoding="utf-8"))
    assert [row["row_id"] for row in written["payloads"]] == sorted(
        row["row_id"] for row in manifest
    )
    assert written["union_of"] == [
        {"path": name, "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest()}
        for name in ("a.json", "b.json")
    ]
    assert (
        T.parse_payload_manifest(Path("union.json").read_bytes(), where="u") == written["payloads"]
    )
    own_input = parser.parse_args(
        ["payloads-union", "--manifests", "a.json", "b.json", "--out", "a.json"]
    )
    with pytest.raises(T.TwoStageError, match="own inputs"):
        own_input.func(own_input)
    absolute = parser.parse_args(
        ["payloads-union", "--manifests", str(tmp_path / "a.json"), "--out", "u2.json"]
    )
    with pytest.raises(T.TwoStageError, match="repo-relative"):
        absolute.func(absolute)
    assert not Path("u2.json").exists()


# ──────────────────────────────────────────────────────────────────────────────
# stage1 --shard / merge-stage1 (P3-16″-hardneg: the A17 census's ~820k contigs per arm)
# ──────────────────────────────────────────────────────────────────────────────
def _fixture_contigs_unsorted() -> list[dict]:
    contigs = T.read_contigs(_FIXTURE / "contigs.json")
    shuffled = contigs[1::2][::-1] + contigs[0::2]
    ids = [contig["contig_id"] for contig in shuffled]
    assert ids != sorted(ids)  # the fixture arrives sorted; an unsorted input proves the sort
    return shuffled


def _write_stage1_shards(tmp_path: Path, count: int) -> list[Path]:
    contigs = T.read_contigs(_FIXTURE / "contigs.json")
    stage1 = T.read_stage1(_FIXTURE / "stage1_window_logits.npz")
    paths = []
    for index in range(count):
        arrays = {}
        for contig in T.shard_contigs(contigs, index, count):
            for field in ("logits", "starts"):
                arrays[f"{contig['contig_id']}/{field}"] = stage1[contig["contig_id"]][field]
        path = tmp_path / f"hardneg_census_stage1_twin.shard{index}of{count}.npz"
        np.savez_compressed(path, **arrays)
        paths.append(path)
    return paths


@pytest.mark.parametrize("count", [1, 2, 3, 7, 20, 21])
def test_stage1_shards_partition_the_contigs_by_sorted_id(count: int) -> None:
    contigs = _fixture_contigs_unsorted()
    ids = sorted(contig["contig_id"] for contig in contigs)
    slices = [[c["contig_id"] for c in T.shard_contigs(contigs, i, count)] for i in range(count)]
    assert [cid for chunk in slices for cid in chunk] == ids  # contiguous, in order, once each
    sizes = [len(chunk) for chunk in slices]
    assert max(sizes) - min(sizes) <= 1
    assert slices == [
        [c["contig_id"] for c in T.shard_contigs(contigs[::-1], i, count)] for i in range(count)
    ]


def test_stage1_shard_refuses_a_duplicate_contig_id() -> None:
    contigs = T.read_contigs(_FIXTURE / "contigs.json")
    with pytest.raises(T.TwoStageError, match="more than once"):
        T.shard_contigs(contigs + contigs[:1], 0, 2)


def test_stage1_parser_takes_a_shard_and_the_merge_leg() -> None:
    parser = T.build_parser()
    one = parser.parse_args(
        [
            "stage1",
            "--contigs",
            "c.json",
            "--checkpoint",
            "k.pt",
            "--out",
            "o.npz",
            "--shard",
            "2/8",
        ]
    )
    assert one.shard == (2, 8) and one.func is T._cmd_stage1
    assert (
        parser.parse_args(["stage1", "--contigs", "c", "--checkpoint", "k", "--out", "o"]).shard
        is None
    )
    merge = parser.parse_args(
        ["merge-stage1", "--contigs", "c.json", "--shards", "a.npz", "b.npz", "--out", "m.npz"]
    )
    assert merge.shards == ["a.npz", "b.npz"] and merge.func is T._cmd_merge_stage1
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["stage1", "--contigs", "c", "--checkpoint", "k", "--out", "o", "--shard", "8/8"]
        )


def test_the_stage1_merge_reassembles_the_fixture_archive(tmp_path: Path) -> None:
    shards = _write_stage1_shards(tmp_path, 3)
    out = tmp_path / "hardneg_census_stage1_twin.npz"
    argv = ["merge-stage1", "--contigs", str(_FIXTURE / "contigs.json"), "--out", str(out)]
    assert T.main(argv + ["--shards", *map(str, shards)]) == 0
    merged = T.read_stage1(out)
    reference = T.read_stage1(_FIXTURE / "stage1_window_logits.npz")
    assert sorted(merged) == sorted(reference)
    for contig_id, entry in reference.items():
        for field in ("logits", "starts"):
            assert merged[contig_id][field].dtype == entry[field].dtype
            assert np.array_equal(merged[contig_id][field], entry[field])


def test_the_stage1_merge_refuses_shards_out_of_index_order(tmp_path: Path) -> None:
    shards = _write_stage1_shards(tmp_path, 3)
    out = tmp_path / "hardneg_census_stage1_twin.npz"
    argv = ["merge-stage1", "--contigs", str(_FIXTURE / "contigs.json"), "--out", str(out)]
    with pytest.raises(T.TwoStageError, match="do not reassemble"):
        T.main(argv + ["--shards", *map(str, shards[::-1])])
    assert not out.exists()


def test_the_stage1_merge_refuses_a_shard_cut_at_another_count(tmp_path: Path) -> None:
    shards = _write_stage1_shards(tmp_path, 3)
    held = [T.stage1_npz_contigs(path) for path in shards]
    # The first two of three shards are a disjoint subset, so a union-only check would need the
    # count to catch them; the per-index re-derivation names the shard that does not fit.
    problems = T.merge_stage1_problems(T.read_contigs(_FIXTURE / "contigs.json"), held[:2])
    assert problems and all("/2 holds" in problem for problem in problems)
    whole = [sorted(cid for chunk in held for cid in chunk)]
    assert T.merge_stage1_problems(T.read_contigs(_FIXTURE / "contigs.json"), whole) == []


def test_the_stage1_merge_refuses_a_missing_or_doubled_contig() -> None:
    contigs = T.read_contigs(_FIXTURE / "contigs.json")
    held = [[c["contig_id"] for c in T.shard_contigs(contigs, i, 2)] for i in range(2)]
    assert T.merge_stage1_problems(contigs, held) == []
    missing = [held[0], held[1][:-1]]
    assert [
        "shard 1/2" in p and "1 missing" in p for p in T.merge_stage1_problems(contigs, missing)
    ] == [True]
    doubled = [held[0], held[1] + held[0][:1]]
    assert [
        "shard 1/2" in p and "1 unassigned" in p for p in T.merge_stage1_problems(contigs, doubled)
    ] == [True]
    assert T.merge_stage1_problems(contigs, []) == ["no Stage-1 shard was given"]


def test_the_stage1_merge_refuses_to_overwrite_a_shard_or_a_v1_path(tmp_path: Path) -> None:
    shards = _write_stage1_shards(tmp_path, 2)
    before = shards[0].read_bytes()
    argv = ["merge-stage1", "--contigs", str(_FIXTURE / "contigs.json"), "--shards"]
    with pytest.raises(T.TwoStageError, match="one of the --shards"):
        T.main(argv + [*map(str, shards), "--out", str(shards[0])])
    assert shards[0].read_bytes() == before
    v1 = tmp_path / "data" / "interim" / "p3_16" / "v1_stage1_twin.npz"
    with pytest.raises(T.TwoStageError, match="refusing to write"):
        T.main(argv + [*map(str, shards), "--out", str(v1)])
    assert not v1.exists()


def test_stage1_npz_contigs_refuses_a_torn_entry(tmp_path: Path) -> None:
    torn = tmp_path / "hardneg_torn.npz"
    np.savez_compressed(
        torn, **{"a/logits": np.zeros((1, 2, 3)), "a/starts": np.zeros(1), "b/logits": np.zeros(1)}
    )
    with pytest.raises(T.TwoStageError, match="lack logits or starts"):
        T.stage1_npz_contigs(torn)
    stray = tmp_path / "hardneg_stray.npz"
    np.savez_compressed(
        stray, **{"a/logits": np.zeros(1), "a/starts": np.zeros(1), "a/other": np.zeros(1)}
    )
    with pytest.raises(T.TwoStageError, match="unexpected array"):
        T.stage1_npz_contigs(stray)
