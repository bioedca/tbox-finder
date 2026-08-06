"""P3-15′-b validation gate — the designed Stage-2 control, run on the real model.

The unit tier (``tests/unit/test_stage2_producer.py``) proves the composition, the
refusals, the merge denominator and every clause of the supply derivation against
committed evidence. It cannot prove that the *checkpoint* still separates a T-box from a
matched null — that needs the model, which lives in ``envs/ml-rna.yml`` and a DVC-tracked
checkpoint, neither of which any CI job carries.

So this file carries the step's actual gate:

1. the committed control report reproduces on the checkpoint that is on disk **today** —
   the same positive, the same seed, the same temperature, the same posteriors;
2. the **designed control fires**: a real tbdb T-box scores high and its dinucleotide
   shuffle scores low, with a *margin*, not merely two thresholds;
3. the control has **power** — the null is a genuine shuffle of the positive (different
   bytes, identical length, identical dinucleotide composition), because a null that is a
   copy of its positive would report a perfect separation while measuring nothing, and
   "no power" is indistinguishable from "no signal" from the outside.

Clause 3 is the one that matters. Clause 2 alone is satisfied by any scorer that happens
to like this sequence; it is the matched null that turns a number into evidence.

⚠ **Disclosure.** The positive is a tbdb record from the committed ingest fixture, so it
may sit inside the Stage-2 training population. This gate is therefore a *liveness and
wiring* control — "the producer resolves, transcribes, scores and calibrates end to end,
and the head discriminates" — **not** a generalization measurement. Generalization is
graded at GATE-2 (``reports/gate2_p3_ece.json``) and by the P4 leave-clade-out arm.

Arming: ``TBOX_REQUIRE_STAGE2_PRODUCER=1``. Deliberately **not** set in
``.github/workflows/ci.yml`` — CI installs no torch, so arming it there would fail every
run. Run it locally inside the pinned env after any change to the producer, the
checkpoint, or the calibration::

    TBOX_REQUIRE_STAGE2_PRODUCER=1 PYTHONPATH=src \\
      python -m pytest tests/ml/test_stage2_producer_control.py

PRD §6/§9.1/§11; ADR-0005 D14, D11/A11; CLAUDE.md §8.5.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

import pytest

from tbox_finder.decoys import dinucleotide_shuffle
from tbox_finder.integration.two_stage import read_temperature
from tbox_finder.mining import stage2_producer as SP

_REPO = Path(__file__).resolve().parents[2]
_CONTROL_REPORT = _REPO / SP.CONTROL_REPORT
_GATE2_REPORT = _REPO / SP.DEFAULT_GATE2_REPORT
_POSITIVE_CSV = _REPO / "tests/fixtures/ingest_sample/Master_tboxes_sample.csv"

_REQUIRE = os.environ.get("TBOX_REQUIRE_STAGE2_PRODUCER") == "1"

#: The posteriors are bf16-backed and batch-position sensitive (P3-14 measured 0.0292
#: logits between two scorings of byte-identical RNA), so the reproduction check is a
#: bounded one, not an equality. It is still far tighter than the 0.999 separation.
_POSTERIOR_TOL = 1e-3


def _skip_or_fail(message: str) -> None:
    """Arming inverts the verdict: unarmed this is a skip, armed it is a failure.

    Without the inversion the gate is unrunnable-by-default *and* silently green when it
    is meant to be enforced — the shape that lets a whole tier skip itself.
    """
    if _REQUIRE:
        pytest.fail(f"TBOX_REQUIRE_STAGE2_PRODUCER=1 but {message}")
    pytest.skip(message)


#: Every module ``load_stage2_checkpoint`` lazy-imports. Probing ``torch`` alone is not
#: enough and the shortfall is not hypothetical: the local ``labvault`` interpreter has
#: torch but no ``peft``, so a torch-only probe sent it past the guard and it failed
#: three tests inside the loader — a missing dependency reported as a broken gate.
_STACK = ("torch", "peft", "safetensors", "multimolecule")


def _need_stack() -> dict:
    """The whole GPU stack and the checkpoint, or a skip/fail. Never a silent pass."""
    import importlib.util

    missing = [name for name in _STACK if importlib.util.find_spec(name) is None]
    if missing:
        _skip_or_fail(f"the Stage-2 stack is incomplete here: {missing} not importable")
    try:
        return SP.resolve_production_arm()
    except (FileNotFoundError, ValueError) as exc:
        _skip_or_fail(f"the Stage-2 production arm is not resolvable here: {exc}")
        raise  # pragma: no cover - _skip_or_fail always raises


def _device() -> str:
    """``"cuda"``, or a skip/fail — never a silent fall back to CPU.

    The six P3-06 arms trained under ``flash_attention_2`` and scoring reproduces the
    backend rather than re-resolving it, so a CPU run does not merely go slow: it raises
    ``NotImplementedError`` inside the attention kernel. Resolving the device here turns
    "this machine has no GPU" into an honest skip instead of three failures that read
    like a broken gate (measured: that is exactly what a hardcoded ``device=None`` did).
    """
    import torch

    if not torch.cuda.is_available():
        _skip_or_fail("no CUDA device; the pinned arm's flash_attention_2 has no CPU kernel")
    return "cuda"


def _committed() -> dict:
    """The committed control record.

    Its **absence is an AssertionError, never a skip**: the record is git-tracked, so a
    checkout that lacks it is a broken commit rather than an unequipped machine, and
    skipping would hide exactly that.
    """
    assert _CONTROL_REPORT.is_file(), f"{SP.CONTROL_REPORT} is git-tracked and must exist"
    return json.loads(_CONTROL_REPORT.read_text(encoding="utf-8"))


# ═════════════════════════════════════════════════════════════════════════════
# The committed record — checkable without the model
# ═════════════════════════════════════════════════════════════════════════════
def test_the_committed_control_record_is_green_and_matched() -> None:
    record = _committed()
    assert record["green"] is True
    assert record["positive_posterior"] >= SP.CONTROL_MIN_POSITIVE
    assert record["shuffle_posterior"] <= SP.CONTROL_MAX_SHUFFLE
    assert record["margin"] >= SP.CONTROL_MIN_MARGIN
    for flag in SP.REQUIRED_CONTROL_FLAGS:
        assert record["flags"][flag] is True, flag
    # The thresholds it was judged against ride WITH it, so a later loosening of the
    # module constants cannot silently re-bless an old record.
    assert record["thresholds"] == {
        "min_positive": SP.CONTROL_MIN_POSITIVE,
        "max_shuffle": SP.CONTROL_MAX_SHUFFLE,
        "min_margin": SP.CONTROL_MIN_MARGIN,
    }


def test_the_committed_control_names_the_calibration_that_ships_today() -> None:
    """A re-fit temperature or a re-trained arm must not inherit this green."""
    record = _committed()
    assert record["temperature"] == read_temperature(_GATE2_REPORT)
    assert record["sweep_fingerprint"] == SP.sweep_fingerprint(record["arm"])
    assert json.loads(_GATE2_REPORT.read_text(encoding="utf-8"))["scoring"]["arm"] == record["arm"]


def test_the_committed_nulls_matchedness_is_reproducible_without_the_model() -> None:
    """The shuffle is deterministic given the seed, so its matchedness is checkable in CI.

    This is the half of the control that does not need a GPU — and it is the half that
    decides whether the other half means anything.
    """
    record = _committed()
    positive = SP.read_control_positive(_POSITIVE_CSV, name=record["positive_name"])
    shuffled = dinucleotide_shuffle(positive, random.Random(record["seed"]))

    assert shuffled != positive, "the null is a COPY of the positive — the control has no power"
    assert len(shuffled) == len(positive) == record["n_nt"]
    assert Counter(positive[i : i + 2] for i in range(len(positive) - 1)) == Counter(
        shuffled[i : i + 2] for i in range(len(shuffled) - 1)
    )
    import hashlib

    assert (
        hashlib.sha256(positive.encode()).hexdigest() == record["positive_sequence_sha256"]
    ), "the committed record was produced from a different positive sequence"
    assert hashlib.sha256(shuffled.encode()).hexdigest() == record["shuffle_sequence_sha256"]


# ═════════════════════════════════════════════════════════════════════════════
# The gate — the model must actually separate them
# ═════════════════════════════════════════════════════════════════════════════
def test_gate_designed_control_fires_on_the_checkpoint_on_disk() -> None:
    _need_stack()
    committed = _committed()
    positive = SP.read_control_positive(_POSITIVE_CSV, name=committed["positive_name"])

    fresh = SP.run_control(
        positive,
        temperature=read_temperature(_GATE2_REPORT),
        seed=committed["seed"],
        device=_device(),
    )

    # (1) it fires
    assert fresh["positive_posterior"] >= SP.CONTROL_MIN_POSITIVE
    assert fresh["shuffle_posterior"] <= SP.CONTROL_MAX_SHUFFLE
    # (2) with a MARGIN — two thresholds alone are met by a scorer with no discrimination
    assert fresh["margin"] >= SP.CONTROL_MIN_MARGIN
    # (3) and the null had power — recomputed INDEPENDENTLY, not read back
    #
    # `run_control` reports its own matchedness, so reading `fresh["flags"]` alone would
    # be self-certification: a body that stopped calling `control_flags` and wrote
    # all-True would satisfy it. Recomputing from the seed and diffing is what makes the
    # flags evidence rather than a claim.
    shuffled = dinucleotide_shuffle(positive, random.Random(committed["seed"]))
    recomputed = SP.control_flags(positive, shuffled)
    assert fresh["flags"] == recomputed, "run_control's flags are not the computed ones"
    for flag in SP.REQUIRED_CONTROL_FLAGS:
        assert recomputed[flag] is True, flag
    assert fresh["green"] is True


def test_run_control_reports_a_POWERLESS_null_as_powerless() -> None:
    """The control for the control — the only case that can catch a fabricated flag.

    On the real positive every matchedness flag is True, so a ``run_control`` that
    stopped computing them and wrote all-True is **indistinguishable** from one that
    computes them ([[all-true-fixture-cannot-test-a-conjunction]]); an independent
    recomputation does not help, because it agrees. What separates them is an input whose
    flags are NOT all true.

    A homopolymer is exactly that: ``dinucleotide_shuffle`` preserves the first and last
    symbol and the exact dinucleotide composition, so on ``"A"*n`` it is the identity.
    A correct producer must report that null as powerless and refuse to certify —
    ``green`` must be False even though the two "arms" score identically, because a
    degenerate null is no evidence at all.
    """
    _need_stack()
    degenerate = SP.run_control(
        "A" * 60,
        temperature=read_temperature(_GATE2_REPORT),
        seed=1,
        device=_device(),
    )
    assert degenerate["flags"]["shuffle_differs_from_positive"] is False
    assert degenerate["green"] is False, "a null identical to its positive certified anyway"


def test_the_committed_record_reproduces_on_this_checkpoint() -> None:
    """Bounded, not exact: the head is bf16 and score_rows is batch-position sensitive."""
    _need_stack()
    committed = _committed()
    positive = SP.read_control_positive(_POSITIVE_CSV, name=committed["positive_name"])
    fresh = SP.run_control(
        positive,
        temperature=read_temperature(_GATE2_REPORT),
        seed=committed["seed"],
        device=_device(),
    )
    for key in ("positive_posterior", "shuffle_posterior"):
        assert fresh[key] == pytest.approx(committed[key], abs=_POSTERIOR_TOL), key
    # The bytes the record was earned against, not just the numbers.
    assert fresh["adapter_sha256"] == committed["adapter_sha256"]
    assert fresh["heads_sha256"] == committed["heads_sha256"]


def test_the_producer_scores_both_strands_of_a_real_locus_and_the_policy_picks_one() -> None:
    """End-to-end on the real model: the two orientations are genuinely different objects.

    This is the measurement the strand §7 decision rests on, reduced to one locus: a
    T-box read on its antisense strand is not a T-box, so a plus-only reading of a locus
    that happens to sit on the minus strand returns `failed` — a satisfied mining
    conjunct on a real T-box.
    """
    _need_stack()
    committed = _committed()
    positive = SP.read_control_positive(_POSITIVE_CSV, name=committed["positive_name"])

    from tbox_finder.infer.handoff import transcribe_to_rna
    from tbox_finder.stage2.eval import load_stage2_checkpoint, score_rows

    device = _device()
    arm = SP.resolve_production_arm()
    model, _record = load_stage2_checkpoint(
        arm["checkpoint_path"],
        device=device,
        attn_implementation=arm["attn_implementation"],
    )
    rows = [
        {"row_id": f"locus|{strand}", "rna_sequence": transcribe_to_rna(positive, strand=strand)}
        for strand in ("+", "-")
    ]
    assert rows[0]["rna_sequence"] != rows[1]["rna_sequence"], "the carrier is self-complementary"
    scored = score_rows(model, rows, batch_size=2, device=device)
    posteriors, per_strand = SP.score_to_posteriors(
        [{**row, "candidate_id": "locus", "strand": row["row_id"][-1]} for row in rows],
        scored,
        temperature=read_temperature(_GATE2_REPORT),
        strand_policy=SP.POLICY_MAX_OVER_STRANDS,
    )
    assert posteriors["locus"] == max(per_strand["locus"]["+"], per_strand["locus"]["-"])
    # The sense strand is the high one for a tbdb record, which is what makes the
    # antisense reading a false negative rather than noise.
    assert per_strand["locus"]["+"] > per_strand["locus"]["-"]


def test_the_supply_derivation_is_green_on_a_machine_that_has_the_model() -> None:
    """The derivation reads only git-tracked evidence, so it must agree here too."""
    from tbox_finder.mining.remine import STAGE2_SUPPLY_AVAILABLE

    derived = SP.derive_stage2_supply_available()
    assert derived["available"] is True, derived["reasons"]
    assert STAGE2_SUPPLY_AVAILABLE is derived["available"]
