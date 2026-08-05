"""P3-14 golden — the two-stage harness replayed from real model output must reproduce the
committed candidate-table digest.

What is real here and what is replayed
--------------------------------------
Everything in ``tests/fixtures/two_stage/`` came out of the shipped checkpoints on real data:
``contigs.json`` is 20 slabs of ``data/interim/flank_context/context_v0.parquet`` (test-rung
records, minted by ``scripts/mint_two_stage_fixture.py``); ``stage1_window_logits.npz`` is what
``data/processed/checkpoints/stage1_production/stage1.pt`` emitted for them; and
``stage2_logits.json`` is what the P3-06 production LoRA arm (``aux1.0_lr1e-4``, the arm
``stage2.eval.production_arm_config()`` names) emitted for every payload the harness derived.
No number in this fixture was invented.

CI installs no torch, so the regression **replays** those artifacts through the torch-free part
of the harness — reconcile → construct_loci → resolve_strands → handoff → calibrate → table —
and diffs the whole-table digest. That is not a weaker test than re-running the models; it is a
*different and stricter* one. Re-running Stage-2 could not produce a stable digest at all:
``determinism.max_abs_duplicate_logit_delta`` in the committed report measures **0.0292** of
logit spread between two scorings of byte-identical RNA (bf16 + flash-attention + length-sorted
batching), so a golden that re-scored would be flaky by construction. Replaying pins the
harness's arithmetic exactly, which is the thing this step ships.

Anti-tautology
--------------
A digest test passes trivially if the digest is insensitive to what it hashes. Three tests here
exist only to show it is not: a single shifted Stage-2 logit moves it, a shift one part in
``DIGEST_QUANTUM`` above the stated tolerance moves it, and reversing row order moves it. The
fixture *inputs* also carry byte-pins, so an edit to a fixture cannot be laundered by
regenerating ``expected.sha256`` alongside it.

CLAUDE.md §8.1 (golden-file regression), §8.7 (no mocking the science).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _ROOT / "tests" / "fixtures" / "two_stage"
_CONTIGS = _FIXTURE_DIR / "contigs.json"
_STAGE1 = _FIXTURE_DIR / "stage1_window_logits.npz"
_STAGE2 = _FIXTURE_DIR / "stage2_logits.json"
_EXPECTED = _FIXTURE_DIR / "expected.sha256"
_REPORT = _ROOT / "reports" / "strand_robustness.json"
_TABLE = _ROOT / "reports" / "p3" / "two_stage_candidates.json"

#: Byte-pins on the fixture inputs. Without these, an edit to a fixture could be laundered by
#: re-deriving ``expected.sha256`` from the edited input and the golden would go green on a
#: changed measurement. Same guard as ``test_reconcile_golden.py``'s ``geometry.csv`` pin.
_CONTIGS_SHA256 = "e28f6a8f976eac4b9d0e9a377b41bbf97a5c4fdc4cc24710ee50fc0693f36e97"
_STAGE1_SHA256 = "f6f8f758c5a8e879a10c2c33fb899a0b0553a7ee6689f2b5032a4b4991a038c1"
_STAGE2_SHA256 = "1ea8deaedde77cebacbcc5e30419ef5b343151c3fc6d4ecfecf030581de58389"

#: The rule the committed fixture was run under. Stated as literals **and** re-read from the
#: report (``test_committed_report_states_the_rule_this_fixture_was_built_under``), because the
#: two say different things: the literals say which rule produced this artifact, the report read
#: is what the replay actually executes. ADR-0005 D3/D15 freeze the production values at P5-01;
#: none of these is that freeze.
_RULE = {
    "threshold_scope": "global",
    "threshold": 0.9,
    "min_span": 50,
    "gap_merge": 10,
    "min_distinct_elements": 2,
    "flank": 50,
    "min_order_margin": 1,
}
_TEMPERATURE = 1.140627294282911  # P3-10's fit, read from reports/gate2_p3_ece.json at mint
_OPERATING_POINT = 0.5


def _replay(**overrides):
    """Replay the committed fixture through the harness, with optional surgical overrides."""
    from tbox_finder.integration import two_stage as T

    contigs = overrides.pop("contigs", None) or T.read_contigs(_CONTIGS)
    stage1 = overrides.pop("stage1", None) or T.read_stage1(_STAGE1)
    stage2 = overrides.pop("stage2", None) or T.read_stage2(_STAGE2)
    kwargs = {
        **_RULE,
        "temperature": _TEMPERATURE,
        "stage2_operating_point": _OPERATING_POINT,
        "source_prior": None,
        "target_prior": None,
    }
    kwargs.update(overrides)
    return T.run_two_stage(contigs, stage1, stage2, **kwargs)


# ── fixture presence (stdlib only — must not skip) ────────────────────────────────────
def test_fixture_present() -> None:
    for path in (_CONTIGS, _STAGE1, _STAGE2, _EXPECTED, _REPORT, _TABLE):
        assert path.is_file(), f"missing committed artifact {path}"
    assert len(_EXPECTED.read_text().strip()) == 64


def test_fixture_inputs_are_byte_pinned() -> None:
    for path, expected in (
        (_CONTIGS, _CONTIGS_SHA256),
        (_STAGE1, _STAGE1_SHA256),
        (_STAGE2, _STAGE2_SHA256),
    ):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{path.name} changed ({actual}); a fixture edit must not be laundered by "
            "re-deriving expected.sha256 from it"
        )


def test_stage1_fixture_is_not_a_git_lfs_pointer() -> None:
    """A checkout without git-lfs leaves a 132-byte pointer every existence check passes.

    ``.gitattributes`` deliberately keeps these paths out of LFS, so this asserts the policy
    rather than compensating for it — it is the tripwire that fires if a future ``*.npz`` or
    ``*.json`` LFS rule swallows the fixture, in CI and on the cluster alike.
    """
    assert not _STAGE1.read_bytes().startswith(b"version https://git-lfs")
    assert _STAGE1.stat().st_size > 100_000


# ── the digest ────────────────────────────────────────────────────────────────────────
def test_replay_reproduces_the_committed_digest() -> None:
    pytest.importorskip("numpy")
    result = _replay()
    assert result.digest == _EXPECTED.read_text().strip()


def test_committed_table_carries_the_committed_digest() -> None:
    pytest.importorskip("numpy")
    from tbox_finder.integration import two_stage as T

    table = json.loads(_TABLE.read_text())
    assert table["digest"] == _EXPECTED.read_text().strip()
    assert table["columns"] == list(T.CANDIDATE_COLUMNS)
    # Re-derive the digest from the committed rows themselves, so the file's own contents are
    # what the hash covers rather than a number it merely carries.
    assert T.candidate_table_digest(table["rows"]) == table["digest"]


def test_replayed_rows_are_the_committed_rows() -> None:
    pytest.importorskip("numpy")
    from tbox_finder.integration import two_stage as T

    committed = json.loads(_TABLE.read_text())["rows"]
    replayed = _replay().rows
    assert len(replayed) == len(committed)
    for fresh, stored in zip(replayed, committed, strict=True):
        for column in T.CANDIDATE_COLUMNS:
            if isinstance(fresh[column], float):
                assert fresh[column] == pytest.approx(stored[column], abs=1e-9), column
            else:
                assert fresh[column] == stored[column], column


# ── the digest actually bites ─────────────────────────────────────────────────────────
def test_digest_moves_if_one_emitted_stage2_logit_shifts() -> None:
    pytest.importorskip("numpy")
    from tbox_finder.integration import two_stage as T

    baseline = _replay()
    stage2 = T.read_stage2(_STAGE2)
    victim = baseline.rows[0]["payload_key"]
    stage2[victim] += 1.0
    assert _replay(stage2=stage2).digest != _EXPECTED.read_text().strip()


def test_the_table_digest_does_not_cover_the_counterfactual_scores() -> None:
    """A property worth stating, because the first draft of the test above assumed otherwise.

    Only **emitted** payloads become rows; the opposite-strand scores exist for the diagnostic
    alone. So perturbing a counterfactual-only logit leaves the candidate-table digest exactly
    where it was — the table is right to ignore it — and moves the diagnostic instead. Stated as
    a test rather than left implicit, because a shift test that happened to pick a
    counterfactual key would otherwise read as "the digest is insensitive to Stage-2", and the
    counterfactual half is covered by ``test_fresh_run_reproduces_the_committed_diagnostic``
    rather than by the hash.
    """
    pytest.importorskip("numpy")
    from tbox_finder.integration import two_stage as T

    baseline = _replay()
    emitted = {row["payload_key"] for row in baseline.rows}
    stage2 = T.read_stage2(_STAGE2)
    counterfactual = sorted(set(stage2) - emitted)
    assert counterfactual, "every scored payload is emitted; this fixture cannot make the point"
    stage2[counterfactual[0]] += 40.0
    perturbed = _replay(stage2=stage2)
    assert perturbed.digest == baseline.digest
    assert (
        perturbed.report["strand_robustness"] != baseline.report["strand_robustness"]
    ), "a counterfactual score must still reach the diagnostic"


def test_digest_moves_on_a_shift_just_above_the_stated_tolerance() -> None:
    """The digest quantises floats; this pins that the tolerance is no wider than advertised."""
    pytest.importorskip("numpy")
    from tbox_finder.integration import two_stage as T

    baseline = _replay().rows
    nudged = [dict(row) for row in baseline]
    # 2/DIGEST_QUANTUM guarantees the rounded integer moves regardless of where the untouched
    # value sat inside its bucket; 0.4/DIGEST_QUANTUM is inside the bucket and must not move it.
    nudged[0]["stage2_named_posterior"] += 2.0 / T.DIGEST_QUANTUM
    assert T.candidate_table_digest(nudged) != T.candidate_table_digest(baseline)
    inside = [dict(row) for row in baseline]
    inside[0]["stage2_named_posterior"] = (
        round(inside[0]["stage2_named_posterior"] * T.DIGEST_QUANTUM) / T.DIGEST_QUANTUM
        + 0.4 / T.DIGEST_QUANTUM
    )
    assert T.candidate_table_digest(inside) == T.candidate_table_digest(
        [
            {
                **row,
                "stage2_named_posterior": (
                    round(row["stage2_named_posterior"] * T.DIGEST_QUANTUM) / T.DIGEST_QUANTUM
                    if index == 0
                    else row["stage2_named_posterior"]
                ),
            }
            for index, row in enumerate(baseline)
        ]
    )


def test_digest_depends_on_row_order() -> None:
    pytest.importorskip("numpy")
    from tbox_finder.integration import two_stage as T

    rows = list(_replay().rows)
    assert T.candidate_table_digest(rows[::-1]) != T.candidate_table_digest(rows)


def test_replay_is_fail_closed_on_a_missing_score() -> None:
    """A payload the committed scores do not cover must raise, never be scored by a neighbour."""
    pytest.importorskip("numpy")
    from tbox_finder.integration import two_stage as T

    stage2 = T.read_stage2(_STAGE2)
    del stage2[sorted(stage2)[0]]
    with pytest.raises(T.TwoStageError, match="no Stage-2 logit"):
        _replay(stage2=stage2)
    # The positive control: the identical call with the score present succeeds, so the refusal
    # above is attributable to the missing key and not to a guard that refuses everything
    # ([[raises-test-needs-a-positive-control]]).
    assert _replay().digest == _EXPECTED.read_text().strip()


# ── the committed report ──────────────────────────────────────────────────────────────
def test_committed_report_re_derives() -> None:
    pytest.importorskip("numpy")
    from tbox_finder.integration import two_stage as T

    assert T.strand_robustness_problems(json.loads(_REPORT.read_text())) == []


def test_committed_report_states_the_rule_this_fixture_was_built_under() -> None:
    rule = json.loads(_REPORT.read_text())["rule"]
    for knob, value in _RULE.items():
        assert rule[knob] == value, knob
    assert rule["temperature"] == pytest.approx(_TEMPERATURE)
    assert rule["stage2_operating_point"] == _OPERATING_POINT
    assert rule["pinned"] is False, "ADR-0005 D3/D15 freeze these at P5-01; nothing here pins one"
    assert rule["source_prior"] is None and rule["target_prior"] is None


def test_committed_report_is_not_an_evaluation() -> None:
    report = json.loads(_REPORT.read_text())
    assert report["is_science"] is False
    assert report["gated"] is False
    assert report["strand_robustness"]["tier_invariance"] is None
    assert "P6" in report["strand_robustness"]["tier_invariance_reason"]


def test_fresh_run_reproduces_the_committed_diagnostic() -> None:
    pytest.importorskip("numpy")

    committed = json.loads(_REPORT.read_text())["strand_robustness"]
    fresh = _replay().report["strand_robustness"]
    assert fresh == committed


# ── the diagnostic's liveness controls: these MUST fire ───────────────────────────────
def test_stage2_is_strand_discriminating_on_this_fixture() -> None:
    """The designed control. A strand-blind Stage-2 reads a perfect 1.0 invariance.

    ``confirmation_invariance`` cannot be interpreted at all without knowing that the two
    strands were scored differently: a scorer that ignored orientation would agree with itself
    everywhere and report perfect robustness while measuring nothing. Both quantities below are
    zero in that world, so asserting them non-zero is what makes the fraction meaningful.
    """
    pytest.importorskip("numpy")

    diagnostic = _replay().report["strand_robustness"]
    assert diagnostic["max_abs_posterior_delta"] > 0.5
    assert diagnostic["n_verdict_disagreements"] > 0


def test_the_resolver_is_load_bearing_and_no_locus_confirms_on_the_wrong_strand() -> None:
    """D15's actual claim: a mis-resolution is a bounded false negative, never a false novelty.

    On this fixture the resolver calls every locus correctly and Stage-2 confirms only the
    correct strand — so ``n_confirmed_on_wrong_strand_only`` is 0, which is the outcome D15
    forbids being non-zero, and ``confirmation_invariance`` is 0.0, which says the resolver is
    load-bearing for every confirmed locus rather than that anything is broken.
    """
    pytest.importorskip("numpy")

    diagnostic = _replay().report["strand_robustness"]
    truth = diagnostic["truth"]
    assert truth["n_loci_overlapping_truth"] == diagnostic["n_loci"]
    assert truth["n_strand_calls_incorrect"] == 0
    assert truth["n_confirmed_on_wrong_strand_only"] == 0
    assert diagnostic["n_confirmed_loci"] > 0


def test_this_fixture_does_not_exercise_the_ambiguity_path() -> None:
    """Made visible, not hidden: the fixture's canonical loci all resolve.

    ``low_order_confidence_fraction`` is 0.0 here because no locus was ambiguous, **not**
    because the both-strand carry-through was tested and found to emit nothing. The branch is
    covered synthetically in ``tests/unit/test_two_stage.py``; this test exists so that if a
    future fixture *does* contain an ambiguous locus, the change is noticed rather than
    absorbed.
    """
    pytest.importorskip("numpy")

    diagnostic = _replay().report["strand_robustness"]
    assert diagnostic["ambiguity_path_exercised"] is False
    assert diagnostic["n_low_order_confidence"] == 0
    assert diagnostic["strand_call_reasons"] == {"resolved": diagnostic["n_loci"]}


def test_the_null_contigs_yield_no_locus() -> None:
    """PRD §12's reversed-sequence null: composition-exact, and Stage-1 finds nothing in it."""
    pytest.importorskip("numpy")

    result = _replay()
    nulls = [run for run in result.runs if run.contig_id.endswith("_rev")]
    assert len(nulls) == 4
    assert all(len(run.loci) == 0 for run in nulls)


def test_scoring_the_identical_rna_twice_is_not_bit_identical() -> None:
    """The measured reason this golden replays instead of re-scoring.

    A contig and its reverse complement hand Stage-2 byte-identical RNA, so the same sequence is
    scored from two batch positions. bf16 + flash-attention + length-sorted batching make that a
    ~0.03-logit difference rather than none — which is precisely why re-running the model inside
    a hash-diffing regression would be flaky, and why the *committed logits* are the fixture.
    """
    pytest.importorskip("numpy")

    determinism = _replay().report["determinism"]
    assert determinism["n_rna_scored_more_than_once"] > 0
    assert determinism["n_duplicate_groups_disagreeing"] > 0
    assert 0.0 < determinism["max_abs_duplicate_logit_delta"] < 0.1
