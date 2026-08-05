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

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tbox_finder.infer import call as C
from tbox_finder.infer import locus as L
from tbox_finder.infer import strand as S
from tbox_finder.infer.reconcile import log_softmax
from tbox_finder.integration import two_stage as T
from tbox_finder.labels import CLASS_INDEX

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


def test_a_locus_confirmed_only_on_the_wrong_strand_is_counted() -> None:
    """The one outcome D15 forbids: a mis-resolution that becomes a false novelty claim.

    Built deliberately — the resolver calls ``+`` (canonical order), truth says ``+``, and
    Stage-2 is then handed a score that confirms only ``-``. The counter must fire; if it never
    could, a real occurrence would be invisible.
    """
    contigs, stage1, run = _run(
        _canonical_spans(), truth_strand="+", truth_start=600, truth_end=880
    )
    manifest = T.payload_manifest([run])
    plus = next(e["row_id"] for e in manifest if e["strand"] == "+")
    minus = next(e["row_id"] for e in manifest if e["strand"] == "-")
    result = T.run_two_stage(contigs, stage1, {plus: -8.0, minus: 8.0}, **{**_RULE, **_CALIBRATION})
    truth = result.report["strand_robustness"]["truth"]
    assert truth["n_loci_overlapping_truth"] == 1
    assert truth["n_strand_calls_correct"] == 1
    assert truth["n_confirmed_on_wrong_strand_only"] == 1


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
    """Two separated canonical loci — the gap of 320 nt far exceeds ``gap_merge=10``."""
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
