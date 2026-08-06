"""P3-15′-b — the per-candidate Stage-2 posterior producer (CPU tier).

Everything here runs on the bare CI path: the module is import-torch-free by design, so
the composition, the refusals, the merge denominator and the supply derivation are all
testable without a GPU or a checkpoint. The one thing that genuinely needs the model —
the designed control that must fire — lives in ``tests/ml/test_stage2_producer_control.py``.

PRD §6/§9.1/§11; ADR-0005 D14, D11/A11.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from tbox_finder.decoys import dinucleotide_shuffle
from tbox_finder.mining import stage2_producer as SP
from tbox_finder.mining.covariation_producer import CandidateSpec
from tbox_finder.mining.homolog_db import HomologDbError
from tbox_finder.mining.remine import (
    STAGE2_SUPPLY_AVAILABLE,
    load_stage2_posteriors,
    remine_candidate_evidence,
)
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED, STATUS_UNAVAILABLE

# A deliberately NON-palindromic carrier: "ACGT" repeated is its own reverse complement,
# so a ± pair built from it comes out byte-identical and every strand assertion below
# would compare a sequence with itself (the P3-14 fixture degeneracy, one tier down).
_CARRIER = "AAGGTTCACGGATTCCAGGTTACAGGATTCACGGTTCCAAGGTTCACGGATTCCAGGTTACA"
_ASSEMBLY = "GCA_000000001.1"


def _write_genome(tmp_path: Path, *, contigs: int = 3) -> Path:
    genome_dir = tmp_path / "genomes"
    genome_dir.mkdir()
    lines = []
    for index in range(contigs):
        lines.append(f">contig_{index}")
        # Rotate the carrier per contig so a candidate resolved off the wrong contig
        # index produces different bytes rather than silently the same ones.
        lines.append((_CARRIER[index:] + _CARRIER[:index]) * 4)
    (genome_dir / f"{_ASSEMBLY}.fna").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return genome_dir


def _spec(index: int, start: int, end: int, *, contig: int = 0) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=f"{_ASSEMBLY}:c{contig}:0:{start}-{end}#{index}",
        accession=f"{_ASSEMBLY}:c{contig}",
        locus_start=start,
        locus_end=end,
    )


def _table(posteriors: dict[str, float], **overrides: object) -> dict[str, object]:
    table = {
        "schema_version": SP.SCHEMA_VERSION,
        "step": SP.STEP,
        "arm": "aux1.0_lr1e-4",
        "temperature": 1.140627294282911,
        "strand_policy": SP.POLICY_MAX_OVER_STRANDS,
        "unresolved": [],
        "strand_posteriors": {k: {"+": v, "-": v} for k, v in posteriors.items()},
        SP.POSTERIORS_KEY: dict(posteriors),
    }
    table.update(overrides)
    return table


# ═════════════════════════════════════════════════════════════════════════════
# The strand policy — the value the round supplies, never defaulted
# ═════════════════════════════════════════════════════════════════════════════
def test_strand_policy_has_no_default_on_the_cli() -> None:
    """``--strand-policy`` is required: it decides which loci get mined."""
    parser = SP.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["score-shard", "--shard", "s.json", "--out", "o.json"])
    args = parser.parse_args(
        ["score-shard", "--shard", "s.json", "--out", "o.json", "--strand-policy", "as_scanned"]
    )
    assert args.strand_policy == "as_scanned"


def test_emit_posterior_distinguishes_the_two_policies() -> None:
    """The two readings genuinely differ — asserted on the case where they disagree."""
    minus_wins = {"+": 0.01, "-": 0.99}
    assert SP.emit_posterior(minus_wins, strand_policy=SP.POLICY_AS_SCANNED) == 0.01
    assert SP.emit_posterior(minus_wins, strand_policy=SP.POLICY_MAX_OVER_STRANDS) == 0.99
    # …and the mirror, so a policy that always reads one slot cannot pass on symmetry.
    plus_wins = {"+": 0.99, "-": 0.01}
    assert SP.emit_posterior(plus_wins, strand_policy=SP.POLICY_AS_SCANNED) == 0.99
    assert SP.emit_posterior(plus_wins, strand_policy=SP.POLICY_MAX_OVER_STRANDS) == 0.99


def test_emit_posterior_refuses_an_unknown_policy_rather_than_falling_back() -> None:
    with pytest.raises(SP.Stage2ProducerError, match="strand_policy must be one of"):
        SP.emit_posterior({"+": 0.5, "-": 0.5}, strand_policy="whatever")
    # Positive control: a guard that raised on EVERYTHING would also satisfy the above.
    assert SP.emit_posterior({"+": 0.5, "-": 0.5}, strand_policy=SP.POLICY_AS_SCANNED) == 0.5


def test_emit_posterior_refuses_a_half_scored_candidate() -> None:
    """A policy applied to one strand would silently publish an unmeasured maximum."""
    with pytest.raises(SP.Stage2ProducerError, match="both strands must be scored"):
        SP.emit_posterior({"+": 0.9}, strand_policy=SP.POLICY_MAX_OVER_STRANDS)


def test_posterior_kind_discloses_that_a_max_is_a_selection() -> None:
    """The artifact must not let a reader take a max-of-two for the ECE-graded object."""
    as_scanned = SP.posterior_kind(SP.POLICY_AS_SCANNED)
    maxed = SP.posterior_kind(SP.POLICY_MAX_OVER_STRANDS)
    assert as_scanned != maxed
    assert "D11" in as_scanned and "D11" in maxed
    assert "SELECTION" in maxed and "SELECTION" not in as_scanned


# ═════════════════════════════════════════════════════════════════════════════
# build_rows — the CPU composition
# ═════════════════════════════════════════════════════════════════════════════
def test_build_rows_emits_both_strands_with_distinct_ids_and_payloads(tmp_path: Path) -> None:
    genome_dir = _write_genome(tmp_path)
    specs = [_spec(0, 4, 40), _spec(1, 8, 44)]
    rows, unresolved = SP.build_rows(specs, genome_dir=genome_dir)

    assert unresolved == []
    assert len(rows) == 2 * len(specs)
    # Unique row_ids: `score_rows` keys its output on str(row_id), so a duplicate would
    # collapse two candidates to one logit with the list length still matching.
    assert len({r["row_id"] for r in rows}) == len(rows)
    for spec in specs:
        pair = [r for r in rows if r["candidate_id"] == spec.candidate_id]
        assert {r["strand"] for r in pair} == {"+", "-"}
        # Selected BY VALUE, not by sort order: ord("+")==43 < ord("-")==45, so a
        # reverse sort yields ("-", "+") and binds both names to the wrong row. The
        # assertions below happen to be symmetric, so it passed while being wrong —
        # and any asymmetric assertion added later would have tested the other strand.
        plus = next(r for r in pair if r["strand"] == "+")
        minus = next(r for r in pair if r["strand"] == "-")
        assert plus["rna_sequence"] != minus["rna_sequence"]
        assert len(plus["rna_sequence"]) == len(minus["rna_sequence"]) == 36
        assert set(plus["rna_sequence"]) <= set("ACGU")
        assert "T" not in plus["rna_sequence"]


def test_build_rows_omits_an_unresolvable_candidate_instead_of_zeroing_it(
    tmp_path: Path,
) -> None:
    """Absent ⇒ unavailable ⇒ SPARED. A literal 0.0 would resolve to failed ⇒ MINED."""
    genome_dir = _write_genome(tmp_path)
    good = _spec(0, 4, 40)
    off_the_end = _spec(1, 10, 10_000)
    rows, unresolved = SP.build_rows([good, off_the_end], genome_dir=genome_dir)

    assert [u["candidate_id"] for u in unresolved] == [off_the_end.candidate_id]
    assert {r["candidate_id"] for r in rows} == {good.candidate_id}
    # …and that omission really is the spared direction, measured through the consumer.
    evidence = remine_candidate_evidence(
        off_the_end.candidate_id, covariation_status=None, stage2_posteriors={}
    )
    assert evidence.stage2_posterior is None


def test_build_rows_lets_a_missing_genome_abort_rather_than_spare_the_shard(
    tmp_path: Path,
) -> None:
    """HomologDbError is a CHECKOUT fault, not a candidate fault.

    Widening the ``except`` to cover it would turn an un-materialised DVC pull into a
    whole shard of ``unavailable`` — silently spared, and indistinguishable from a clean
    run. The two exception classes are siblings, so this only holds if the narrow one is
    caught by name.
    """
    empty = tmp_path / "no-genomes"
    empty.mkdir()
    with pytest.raises(HomologDbError):
        SP.build_rows([_spec(0, 4, 40)], genome_dir=empty)


# ═════════════════════════════════════════════════════════════════════════════
# score_to_posteriors — the D11 indirection and the join
# ═════════════════════════════════════════════════════════════════════════════
def _rows_for(ids: list[str]) -> list[dict[str, object]]:
    return [
        {"row_id": f"{cid}|{strand}", "candidate_id": cid, "strand": strand, "rna_sequence": "ACGU"}
        for cid in ids
        for strand in ("+", "-")
    ]


def test_score_to_posteriors_reads_the_gated_key_and_takes_the_max() -> None:
    rows = _rows_for(["a", "b"])
    # +8 => ~1.0, -8 => ~0.0 at T=1; candidate a is high on '-', b is high on '+'.
    scored = [{"tbox_logit": z} for z in (-8.0, 8.0, 8.0, -8.0)]
    posteriors, per_strand = SP.score_to_posteriors(
        rows, scored, temperature=1.0, strand_policy=SP.POLICY_MAX_OVER_STRANDS
    )
    assert set(posteriors) == {"a", "b"}
    assert posteriors["a"] > 0.99 and posteriors["b"] > 0.99
    assert per_strand["a"]["+"] < 0.01 and per_strand["a"]["-"] > 0.99
    # The published value IS the max of the two recorded ones, not a third number.
    for cid, per in per_strand.items():
        assert posteriors[cid] == max(per["+"], per["-"])


def test_score_to_posteriors_applies_the_temperature_it_is_given() -> None:
    """T is a real divisor here, not decoration — two temperatures must disagree."""
    rows = _rows_for(["a"])
    scored = [{"tbox_logit": 2.0}, {"tbox_logit": 2.0}]
    cold, _ = SP.score_to_posteriors(
        rows, scored, temperature=1.0, strand_policy=SP.POLICY_AS_SCANNED
    )
    warm, _ = SP.score_to_posteriors(
        rows, scored, temperature=4.0, strand_policy=SP.POLICY_AS_SCANNED
    )
    assert cold["a"] > warm["a"]
    assert cold["a"] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))


def test_score_to_posteriors_refuses_a_sheared_join() -> None:
    rows = _rows_for(["a", "b"])
    with pytest.raises(SP.Stage2ProducerError, match="the join would shear"):
        SP.score_to_posteriors(
            rows,
            [{"tbox_logit": 1.0}],
            temperature=1.0,
            strand_policy=SP.POLICY_AS_SCANNED,
        )


# ═════════════════════════════════════════════════════════════════════════════
# validate_posteriors — refuse what the consumer would, before writing
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    ("value", "match"),
    [
        (7.5, "outside"),
        (-0.1, "outside"),
        (True, "not a real number"),
        ("0.5", "not a real number"),
        (float("nan"), "not finite"),
    ],
)
def test_validate_posteriors_refuses(value: object, match: str) -> None:
    with pytest.raises(SP.Stage2ProducerError, match=match):
        SP.validate_posteriors({"c": value})


def test_validate_posteriors_accepts_the_closed_unit_interval() -> None:
    """The positive control: a guard that refused everything would pass the table above."""
    SP.validate_posteriors({"lo": 0.0, "hi": 1.0, "mid": 0.5})


# ═════════════════════════════════════════════════════════════════════════════
# merge — the denominator the consumer does not carry
# ═════════════════════════════════════════════════════════════════════════════
def _write_tables(tmp_path: Path, chunks: list[dict[str, float]], **overrides) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, chunk in enumerate(chunks):
        path = tmp_path / f"posteriors_{index:03d}.json"
        path.write_text(json.dumps(_table(chunk, **overrides)), encoding="utf-8")
        paths.append(path)
    return paths


def test_merge_refuses_a_truncated_run_and_accepts_a_complete_one(tmp_path: Path) -> None:
    """The load-bearing refusal: ``load_stage2_posteriors`` accepts ``{}`` silently.

    A table covering 2 of 10 leaves 8 candidates ``unavailable`` ⇒ spared, and the round
    reports success having decided almost nothing ([[cost-knobs-can-certify]]).
    """
    partial = _write_tables(tmp_path, [{"c0": 0.5}, {"c1": 0.5}])
    with pytest.raises(SP.Stage2ProducerError, match="below the required"):
        SP.merge_posterior_tables(partial, n_candidates=10)

    # POSITIVE CONTROL: the same code path with full coverage must pass, or the refusal
    # above is indistinguishable from a merge that refuses everything.
    full = _write_tables(
        tmp_path / "full", [{f"c{i}": 0.5 for i in range(5)}, {f"c{i}": 0.5 for i in range(5, 10)}]
    )
    merged = SP.merge_posterior_tables(full, n_candidates=10)
    assert merged["n_scored"] == 10
    assert merged["coverage"] == 1.0
    assert merged["n_shards"] == 2


def test_merge_refuses_a_candidate_claimed_by_two_shards(tmp_path: Path) -> None:
    tables = _write_tables(tmp_path, [{"c0": 0.5, "c1": 0.5}, {"c1": 0.9}])
    with pytest.raises(SP.Stage2ProducerError, match="appears in more than one shard"):
        SP.merge_posterior_tables(tables, n_candidates=2)


def test_merge_refuses_without_a_denominator(tmp_path: Path) -> None:
    tables = _write_tables(tmp_path, [{"c0": 0.5}])
    with pytest.raises(SP.Stage2ProducerError, match="n_candidates must be positive"):
        SP.merge_posterior_tables(tables, n_candidates=0)


@pytest.mark.parametrize("key", ["temperature", "strand_policy", "arm"])
def test_merge_refuses_shards_from_different_runs(tmp_path: Path, key: str) -> None:
    """Merging a max-over-strands shard with an as-scanned one publishes two objects."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(_table({"c0": 0.5})), encoding="utf-8")
    drifted = _table({"c1": 0.5})
    drifted[key] = "DRIFTED" if isinstance(drifted[key], str) else 99.0
    second.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(SP.Stage2ProducerError, match=f"disagree on {key!r}"):
        SP.merge_posterior_tables([first, second], n_candidates=2)


def test_merge_refuses_a_file_that_is_not_a_producer_table(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
    with pytest.raises(SP.Stage2ProducerError, match="not a producer table"):
        SP.merge_posterior_tables([path], n_candidates=1)


def test_the_merged_table_round_trips_through_the_real_consumer(tmp_path: Path) -> None:
    """The wrapper form is used because the FLAT form refuses metadata by name.

    Asserted end-to-end through ``remine.load_stage2_posteriors`` rather than by reading
    its source: this is the contract that makes the artifact useful at all.
    """
    tables = _write_tables(tmp_path, [{"c0": 0.25}, {"c1": 0.75}])
    merged = SP.merge_posterior_tables(tables, n_candidates=2)
    out = tmp_path / "stage2_posteriors.json"
    SP.write_json(out, merged)
    assert load_stage2_posteriors(out) == {"c0": 0.25, "c1": 0.75}
    # …and the sibling metadata really did ride along, which is the whole reason for the
    # wrapper: a flat table carrying `schema_version` is REFUSED by the consumer.
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["strand_policy"] == SP.POLICY_MAX_OVER_STRANDS
    assert reloaded["coverage"] == 1.0


def test_the_published_posterior_drives_the_spare_rule_verdict(tmp_path: Path) -> None:
    """The produced number reaches D14's three-valued status — passed/failed/unavailable."""
    tables = _write_tables(tmp_path, [{"high": 0.99, "low": 0.01}])
    merged = SP.merge_posterior_tables(tables, n_candidates=2)
    out = tmp_path / "p.json"
    SP.write_json(out, merged)
    posteriors = load_stage2_posteriors(out)

    from tbox_finder.mining.spare_rule import stage2_status

    def _status(candidate_id: str) -> str:
        # Routed through the REAL evidence builder, not a hand-made dataclass: the join
        # from the produced table to the rule's input is the half of the contract that a
        # constructed SpareRuleEvidence would skip over.
        evidence = remine_candidate_evidence(
            candidate_id, covariation_status=None, stage2_posteriors=posteriors
        )
        return stage2_status(evidence, stage2_threshold=0.9)

    assert _status("high") == STATUS_PASSED
    assert _status("low") == STATUS_FAILED
    # An id the producer never scored: absent ⇒ unavailable ⇒ SPARED, the fail-closed
    # direction that makes a dropped shard cost sensitivity and never a mined T-box.
    assert _status("never-scored") == STATUS_UNAVAILABLE


# ═════════════════════════════════════════════════════════════════════════════
# The temperature is read, never re-typed
# ═════════════════════════════════════════════════════════════════════════════
def test_no_module_in_src_hardcodes_the_fitted_temperature() -> None:
    """A second home for a number GATE-2 owns is free to go stale with nothing failing."""
    fitted = json.loads((SP.REPO_ROOT / SP.DEFAULT_GATE2_REPORT).read_text(encoding="utf-8"))[
        "gate"
    ]["calibration"]["temperature"]
    # Guard against the pin becoming vacuous if the report is ever re-fitted to 1.0.
    assert fitted != 1.0
    needle = repr(float(fitted))[:10]
    offenders = [
        str(path.relative_to(SP.REPO_ROOT))
        for path in (SP.REPO_ROOT / "src").rglob("*.py")
        if needle in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"the fitted temperature is hardcoded in {offenders}"


# ═════════════════════════════════════════════════════════════════════════════
# The control's null is a real null
# ═════════════════════════════════════════════════════════════════════════════
def test_control_flags_detect_a_null_that_is_a_copy_of_its_positive() -> None:
    """The matchedness computation, exercised where CI can reach it.

    A control whose "null" is the positive still separates nothing, and reports a perfect
    margin while doing so — "no power" reading as "no signal". This is the pure half of
    `run_control`, split out precisely so this case is checkable without a checkpoint.
    """
    positive = _CARRIER * 3
    shuffled = dinucleotide_shuffle(positive, random.Random(20260805))

    good = SP.control_flags(positive, shuffled)
    assert all(good[flag] for flag in SP.REQUIRED_CONTROL_FLAGS), good

    # (a) the null is a COPY — matched on every dimension EXCEPT the one that matters
    copied = SP.control_flags(positive, positive)
    assert copied["shuffle_differs_from_positive"] is False
    assert copied["dinucleotide_composition_matched"] is True
    assert copied["length_matched"] is True

    # (b) the null is a different sequence entirely — differs, but is NOT matched
    unmatched = SP.control_flags(positive, "A" * len(positive))
    assert unmatched["shuffle_differs_from_positive"] is True
    assert unmatched["dinucleotide_composition_matched"] is False
    assert unmatched["length_matched"] is True

    # (c) truncated
    assert SP.control_flags(positive, shuffled[:-5])["length_matched"] is False


def test_score_to_posteriors_follows_the_gated_key_when_it_moves() -> None:
    """The D11 indirection is load-bearing only if it is actually followed.

    Reading a literal ``"named_posterior"`` is indistinguishable from reading
    ``payload[payload["gated_posterior_key"]]`` while the two agree — which they do today.
    The property under test is what happens when they STOP agreeing: the calibrator's own
    declaration of which object is gated must win, or a move of the gated object would
    silently publish the ungated one.
    """
    import tbox_finder.mining.stage2_producer as module

    def _fake_calibrated_posterior(logits, *, temperature, **kwargs):
        return {
            "gated_posterior_key": "relocated_posterior",
            "relocated_posterior": [0.75 for _ in logits],
            # The old home still exists and carries a DIFFERENT value, so a literal read
            # produces a wrong number rather than a KeyError.
            "named_posterior": [0.25 for _ in logits],
        }

    original = module.calibrated_posterior
    module.calibrated_posterior = _fake_calibrated_posterior
    try:
        posteriors, _ = module.score_to_posteriors(
            _rows_for(["a"]),
            [{"tbox_logit": 1.0}, {"tbox_logit": 1.0}],
            temperature=1.0,
            strand_policy=SP.POLICY_AS_SCANNED,
        )
    finally:
        module.calibrated_posterior = original
    assert posteriors["a"] == 0.75, "the producer read a literal key, not the gated one"


def test_the_control_shuffle_is_matched_and_is_not_the_identity() -> None:
    """``dinucleotide_shuffle`` preserves the first AND last symbol.

    On a short or low-complexity span it can therefore return the input unchanged, and a
    "control" whose null equals its positive measures nothing while reporting a perfect
    separation ([[control-matchedness-must-be-asserted]]).
    """
    from collections import Counter

    positive = _CARRIER * 3
    shuffled = dinucleotide_shuffle(positive, random.Random(20260805))
    assert shuffled != positive
    assert len(shuffled) == len(positive)
    assert Counter(positive[i : i + 2] for i in range(len(positive) - 1)) == Counter(
        shuffled[i : i + 2] for i in range(len(shuffled) - 1)
    )


def test_read_control_positive_selects_by_name_and_returns_the_recorded_sequence() -> None:
    """Never a ``Name``-derived coordinate slice ([[tbdb-name-coords-untrustworthy]])."""
    csv_path = SP.REPO_ROOT / "tests/fixtures/ingest_sample/Master_tboxes_sample.csv"
    sequence = SP.read_control_positive(csv_path, name="CP020815.1:1322846-1323252")
    assert len(sequence) == 407
    assert set(sequence) <= set("ACGT")
    with pytest.raises(SP.Stage2ProducerError, match="no record named"):
        SP.read_control_positive(csv_path, name="not-a-record")


# ═════════════════════════════════════════════════════════════════════════════
# The supply derivation — six named, independently-breakable clauses
# ═════════════════════════════════════════════════════════════════════════════
_CLAUSES = (
    "gate2_calibration_wellformed",
    "production_arm_on_record",
    "producer_present",
    "producer_posterior_wired",
    "control_green",
    "control_matches_this_calibration",
)


def test_the_supply_derivation_is_green_and_names_every_clause() -> None:
    """The full named clause dict, not ``all(...)``.

    A hardcoded ``True`` satisfies a conjunction when every clause is TRUE, so the clause
    NAMES are asserted as a set — a clause silently dropped from the derivation would
    otherwise make the gate weaker while keeping it green
    ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    derived = SP.derive_stage2_supply_available()
    # Compared against this file's OWN literal tuple, never against SP.SUPPLY_CLAUSES:
    # the function now fills any unreported clause from that constant, so asserting
    # against it would be a tautology — dropping a clause from both would stay green
    # ([[promote-dont-duplicate-is-a-correctness-rule]]).
    assert set(derived["clauses"]) == set(_CLAUSES)
    assert set(SP.SUPPLY_CLAUSES) == set(_CLAUSES)
    assert derived["reasons"] == []
    assert all(derived["clauses"].values())
    assert derived["available"] is True


def test_a_clause_that_never_reports_a_verdict_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ABSENT clause must not read as a passing one.

    ``all(...)`` skips what is not in the map, so a branch that never runs is a silently
    satisfied clause. Reproduced by execution before the fix: with
    ``production_arm_config()`` returning ``None`` while ``sweep_fingerprint`` succeeded,
    neither branch set ``production_arm_on_record`` and the derivation returned
    ``available: True`` on five of six clauses ([[clauses-must-guard-emptiness]]).
    """
    monkeypatch.setattr(SP, "production_arm_config", lambda **kwargs: None)
    derived = SP.derive_stage2_supply_available()
    assert set(derived["clauses"]) == set(_CLAUSES), "a clause vanished from the map"
    assert derived["clauses"]["production_arm_on_record"] is False
    assert any("did not report a verdict" in reason for reason in derived["reasons"])
    assert derived["available"] is False


def test_an_unreachable_consumer_is_a_failed_clause_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clause 4 is documented fail-closed, so it must not propagate an import error."""
    import builtins

    real_import = builtins.__import__

    def _explode(name, globals=None, locals=None, fromlist=(), level=0):
        if "remine_candidate_evidence" in (fromlist or ()):
            raise RuntimeError("the consumer is unreachable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _explode)
    derived = SP.derive_stage2_supply_available()
    monkeypatch.undo()
    assert derived["clauses"]["producer_posterior_wired"] is False
    assert any("unreachable" in reason for reason in derived["reasons"])
    assert derived["available"] is False


def test_the_constant_and_its_derivation_cannot_drift_in_either_direction() -> None:
    derived = SP.derive_stage2_supply_available()
    assert STAGE2_SUPPLY_AVAILABLE is derived["available"], derived["reasons"]


@pytest.mark.parametrize("clause", ["control_green", "control_matches_this_calibration"])
def test_a_missing_control_report_fails_closed(tmp_path: Path, clause: str) -> None:
    """Fail-CLOSED on absent evidence, and both control clauses must notice."""
    (tmp_path / "reports/p3/sweep").mkdir(parents=True)
    for name in (SP.DEFAULT_GATE2_REPORT, "reports/p3/sweep/aux1.0_lr1e-4.json"):
        source = SP.REPO_ROOT / name
        (tmp_path / name).write_bytes(source.read_bytes())
    derived = SP.derive_stage2_supply_available(repo_root=tmp_path)
    assert derived["available"] is False
    assert derived["clauses"][clause] is False
    # Only the control clauses fail: the other four are unaffected, which is what makes
    # each one independently breakable rather than a single all-or-nothing read.
    assert derived["clauses"]["gate2_calibration_wellformed"] is True
    assert derived["clauses"]["producer_present"] is True
    assert derived["clauses"]["producer_posterior_wired"] is True


def test_a_control_certified_against_a_different_temperature_does_not_inherit_green(
    tmp_path: Path,
) -> None:
    """A re-fitted GATE-2 must invalidate a control scored at the old T."""
    _stage_evidence(tmp_path, control_overrides={"temperature": 99.0})
    derived = SP.derive_stage2_supply_available(repo_root=tmp_path)
    assert derived["clauses"]["control_matches_this_calibration"] is False
    assert derived["clauses"]["control_green"] is True  # the separation itself is untouched
    assert derived["available"] is False


def test_a_control_certified_against_a_retrained_arm_does_not_inherit_green(
    tmp_path: Path,
) -> None:
    """The checkpoint bytes are DVC-tracked and absent in CI; the run report is not."""
    fingerprint = dict(SP.sweep_fingerprint("aux1.0_lr1e-4"))
    fingerprint["saved_val_total"] = 0.123456
    _stage_evidence(tmp_path, control_overrides={"sweep_fingerprint": fingerprint})
    derived = SP.derive_stage2_supply_available(repo_root=tmp_path)
    assert derived["clauses"]["control_matches_this_calibration"] is False
    assert derived["available"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"positive_posterior": 0.5},
        {"shuffle_posterior": 0.5},
        {"green": False},
        {"n_control_rows": 1},
        {"flags": {"dinucleotide_composition_matched": True, "length_matched": True}},
        {
            "flags": {
                "shuffle_differs_from_positive": False,
                "dinucleotide_composition_matched": True,
                "length_matched": True,
            }
        },
    ],
)
def test_each_control_weakness_fails_the_clause_on_its_own(
    tmp_path: Path, override: dict[str, object]
) -> None:
    """One mutation at a time: a clause that only ever sees a fully-green record is untested.

    The last two cases are the ones that matter most — a control whose null is a COPY of
    its positive still separates nothing, so "no power" would read as "no signal".
    """
    _stage_evidence(tmp_path, control_overrides=override)
    derived = SP.derive_stage2_supply_available(repo_root=tmp_path)
    assert derived["clauses"]["control_green"] is False
    assert derived["available"] is False


def test_a_thresholds_only_control_cannot_pass_without_separation(tmp_path: Path) -> None:
    """Two thresholds without a margin are satisfiable by a scorer with no discrimination."""
    _stage_evidence(
        tmp_path,
        control_overrides={"positive_posterior": 0.95, "shuffle_posterior": 0.5},
    )
    derived = SP.derive_stage2_supply_available(repo_root=tmp_path)
    assert derived["clauses"]["control_green"] is False
    assert any("margin" in reason for reason in derived["reasons"])


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        ({"gate": {"graded_posterior_key": "prior_shifted_posterior"}}, "graded_posterior_key"),
        ({"gate": {"prior_shift_applied": True}}, "prior_shift_applied"),
        ({"gate": {"calibration": {"temperature": -1.0}}}, "not positive"),
        ({"gate": {"calibration": {"temperature": "hot"}}}, "not a real number"),
    ],
)
def test_the_calibration_clause_refuses_a_different_object(
    tmp_path: Path, mutation: dict, reason_fragment: str
) -> None:
    """This producer emits the PRE-prior-shift named object; anything else is not it."""
    _stage_evidence(tmp_path, gate2_mutation=mutation)
    derived = SP.derive_stage2_supply_available(repo_root=tmp_path)
    assert derived["clauses"]["gate2_calibration_wellformed"] is False
    assert any(reason_fragment in reason for reason in derived["reasons"])
    assert derived["available"] is False


def test_the_producer_posterior_wired_clause_measures_behaviour_not_a_signature() -> None:
    """A producer whose output the round drops on the floor is the no-op this refuses."""
    probe = "__stage2_supply_probe__"
    evidence = remine_candidate_evidence(
        probe, covariation_status=None, stage2_posteriors={probe: 0.875}
    )
    assert evidence.stage2_posterior == 0.875
    # The negative half: an id ABSENT from the map must stay None, or the clause would be
    # satisfied by a join that invents a value.
    assert (
        remine_candidate_evidence(
            probe, covariation_status=None, stage2_posteriors={"other": 0.875}
        ).stage2_posterior
        is None
    )


def test_the_producer_present_clause_names_the_whole_chain() -> None:
    for entry in SP.PRODUCER_ENTRY_POINTS:
        assert hasattr(SP, entry), entry
    assert set(SP.PRODUCER_ENTRY_POINTS) == {
        "build_rows",
        "score_shard",
        "merge_posterior_tables",
        "run_control",
    }


def test_a_producer_missing_an_entry_point_is_a_failed_clause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stub carrying only SOME of the chain must not satisfy ``producer_present``.

    The mutation empties the module of one named entry rather than swapping the tuple: a
    swap can leave the very name the clause looks for still present, and then the
    sabotage removes nothing (the P3-15′-a miss, one flag over).
    """
    _stage_evidence(tmp_path)
    monkeypatch.delattr(SP, "merge_posterior_tables", raising=True)
    derived = SP.derive_stage2_supply_available(repo_root=tmp_path)
    assert derived["clauses"]["producer_present"] is False
    assert any("merge_posterior_tables" in reason for reason in derived["reasons"])
    assert derived["available"] is False


def test_a_producer_that_cannot_import_is_a_failed_clause_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail-closed means fail-closed: a broken producer must not abort the derivation.

    Deliberately broader than ``ImportError`` — a module-level ``RuntimeError``/``OSError``
    anywhere in the producer's transitive imports would otherwise propagate out of a
    function documented as fail-closed on every clause. The failure is injected at
    ``__import__`` because ``from pkg import sub`` resolves through the parent package,
    so patching the submodule alone can be skipped entirely when the parent already
    carries the attribute ([[from-import-skips-when-parent-has-attr]]).
    """
    import builtins

    _stage_evidence(tmp_path)
    real_import = builtins.__import__

    def _explode(name, globals=None, locals=None, fromlist=(), level=0):
        if "stage2_producer" in (fromlist or ()):
            raise RuntimeError("module-level boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _explode)
    derived = SP.derive_stage2_supply_available(repo_root=tmp_path)
    monkeypatch.undo()
    assert derived["clauses"]["producer_present"] is False
    assert any("import failed" in reason for reason in derived["reasons"])
    assert derived["available"] is False


def _stage_evidence(
    tmp_path: Path,
    *,
    control_overrides: dict | None = None,
    gate2_mutation: dict | None = None,
) -> None:
    """Copy the three git-tracked evidence files into ``tmp_path``, optionally mutated."""
    (tmp_path / "reports/p3/sweep").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports/p3/sweep/aux1.0_lr1e-4.json").write_bytes(
        (SP.REPO_ROOT / "reports/p3/sweep/aux1.0_lr1e-4.json").read_bytes()
    )
    gate2 = json.loads((SP.REPO_ROOT / SP.DEFAULT_GATE2_REPORT).read_text(encoding="utf-8"))
    for key, patch in (gate2_mutation or {}).items():
        # The original nested dict must be captured BEFORE the shallow merge: that merge
        # replaces `gate.calibration` wholesale with the patch, so re-reading it afterwards
        # gave `{**patch, **patch}` and silently dropped beta/calib_prevalence/fitted_on.
        # The clause then failed partly because required keys were missing rather than
        # only because of the one value under test — the opposite of the
        # one-mutation-at-a-time discipline this helper exists to provide.
        original_nested = gate2.get(key, {}).get("calibration", {}) if key == "gate" else {}
        gate2[key] = {**gate2.get(key, {}), **patch} if isinstance(patch, dict) else patch
        if key == "gate" and isinstance(patch, dict) and "calibration" in patch:
            gate2["gate"]["calibration"] = {**original_nested, **patch["calibration"]}
    (tmp_path / SP.DEFAULT_GATE2_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / SP.DEFAULT_GATE2_REPORT).write_text(json.dumps(gate2), encoding="utf-8")

    control = json.loads((SP.REPO_ROOT / SP.CONTROL_REPORT).read_text(encoding="utf-8"))
    control.update(control_overrides or {})
    (tmp_path / SP.CONTROL_REPORT).write_text(json.dumps(control), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# The remine CLI — the flip must REACH the round, in both directions
# ═════════════════════════════════════════════════════════════════════════════
def test_the_remine_cli_defaults_both_supply_flags_from_their_module_constants() -> None:
    """A bare ``store_true`` could not express a ``True`` default at all.

    Without this the P3-15′-a and P3-15′-b flips are constants nothing reads: the CLI the
    round actually invokes would keep passing a hard ``False``
    ([[pinned-constant-that-nothing-reads]]).
    """
    from tbox_finder.mining.mine_round import MSA_SUPPLY_AVAILABLE
    from tbox_finder.mining.remine import build_parser

    args = build_parser().parse_args(["plan", "--stage2-threshold", "0.9", "--out", "o.json"])
    assert args.stage2_supply_available is STAGE2_SUPPLY_AVAILABLE
    assert args.msa_supply_available is MSA_SUPPLY_AVAILABLE


@pytest.mark.parametrize(
    ("flag", "dest"),
    [
        ("--no-stage2-supply-available", "stage2_supply_available"),
        ("--no-msa-supply-available", "msa_supply_available"),
    ],
)
def test_the_remine_cli_can_declare_a_supply_unavailable(flag: str, dest: str) -> None:
    """The conservative direction must be expressible on a machine that lacks the supply."""
    from tbox_finder.mining.remine import build_parser

    args = build_parser().parse_args(["plan", "--stage2-threshold", "0.9", "--out", "o.json", flag])
    assert getattr(args, dest) is False


def test_the_two_supply_flags_are_mutually_exclusive() -> None:
    """Asserting both would otherwise resolve to whichever argparse happened to see last."""
    from tbox_finder.mining.remine import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "plan",
                "--stage2-threshold",
                "0.9",
                "--out",
                "o.json",
                "--stage2-supply-available",
                "--no-stage2-supply-available",
            ]
        )


def test_a_declared_but_unevidenced_stage2_supply_exits_4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Declared-available on a checkout that cannot evidence it is a STAGING fault.

    Exit 4, deliberately distinct from the round's own honest refusals (1/2) that the
    sbatch branches on: routing this through one of those would turn a misconfigured
    checkout into a silent "no round today".
    """
    from tbox_finder.mining import remine

    monkeypatch.setattr(
        SP,
        "derive_stage2_supply_available",
        lambda **kwargs: {"available": False, "clauses": {}, "reasons": ["staged nothing"]},
    )
    out = tmp_path / "plan.json"
    rc = remine.main(
        [
            "plan",
            "--stage2-threshold",
            "0.9",
            "--out",
            str(out),
            "--stage2-supply-available",
        ]
    )
    assert rc == 4
    # It still WRITES, with may_run forced False and the override named: not writing would
    # leave an earlier good run's report at this path reading `may_run: true`, and that key
    # is what the §9.3 artifact verify reads ([[guard-runs-after-what-it-guards]]).
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["plan"]["may_run"] is False
    assert report["plan"]["may_run_overridden_by"] == "supply_declaration_unevidenced"


def test_declaring_a_supply_unavailable_is_never_the_staging_fault(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The asymmetry is the point — the positive control for the test above.

    Under-claiming is conservative (the round refuses; nothing is mined). Only the
    over-claim is fatal, so a guard that exited 4 on BOTH would also satisfy the
    assertion above while refusing every honest run.
    """
    from tbox_finder.mining import remine

    monkeypatch.setattr(
        SP,
        "derive_stage2_supply_available",
        lambda **kwargs: {"available": False, "clauses": {}, "reasons": ["staged nothing"]},
    )
    rc = remine.main(
        [
            "plan",
            "--stage2-threshold",
            "0.9",
            "--out",
            str(tmp_path / "plan.json"),
            "--no-stage2-supply-available",
        ]
    )
    assert rc != 4
