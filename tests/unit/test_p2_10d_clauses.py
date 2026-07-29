"""P2-10d — the ``negative_mix_realized`` and ``warm_start_loaded`` gate clauses.

Bare-CI tier: pure JSON, no torch.

Both clauses have the shape this repo has been bitten by four times: an evidence block
that a run may legitimately not produce, guarding a claim the run may nevertheless have
made. ``overall_pass = all(clauses)`` catches a clause flipped FALSE and structurally
cannot catch one fabricated TRUE (P1-15/P1-16), and a clause derived from the *requested
config* rather than the *found evidence* goes vacuously TRUE exactly when the evidence is
missing (P2-05).

So the absence branch of each clause is an **assertion**, not a skip:

* no ``negative_mix`` block ⇒ the run must also have requested no pool, no fraction, no
  warm start and no step-0 eval. A run that asked for a mix and recorded no measurement of
  it FAILS here.
* no ``warm_start`` block ⇒ ``init_from_checkpoint`` must be unset. A run that asked to
  continue a parent checkpoint and recorded nothing about the load FAILS here.

And the present branch of each is checked by **tampering**: every field the clause reads is
individually corrupted and the clause must go False, so a clause that quietly stopped
reading one of them cannot pass this file.
"""

from __future__ import annotations

import pytest

from tbox_finder.train.train_stage1 import (
    _negative_mix_ok,
    _rc_combine_ok,
    _warm_start_ok,
    derive_clauses,
    validate_report,
)


def _stream(n_pos: int = 100, n_neg: int = 1000, pool: int = 40) -> dict:
    return {
        "n_total": n_pos + n_neg,
        "n_negative": n_neg,
        "n_positive": n_pos,
        "realized_negative_fraction": n_neg / (n_pos + n_neg),
        "requested_negative_fraction": 10 / 11,
        "negative_pool_size": pool,
        "positive_pool_size": n_pos,
    }


def _mix(**over) -> dict:
    block = {
        "stream": _stream(),
        "consumed": {**_stream(), "n_total": 1100},
        "epochs": 1,
        "world_size": 1,
        "rank": 0,
    }
    block.update(over)
    return block


def _warm(**over) -> dict:
    block = {
        "checkpoint": "checkpoints/p2/stage1/stage1.pt",
        "checkpoint_sha256": "a" * 64,
        "n_checkpoint_tensors": 137,
        "n_model_tensors": 137,
        "n_missing_keys": 0,
        "n_unexpected_keys": 0,
        "n_shape_mismatch": 0,
        "n_tensors_differing_before": 137,
        "n_tensors_differing_after": 0,
        "strict": True,
    }
    block.update(over)
    return block


def _config(**over) -> dict:
    cfg = {
        "negative_pool_parquet": "data/processed/negatives/mining_pool_v0.parquet",
        "negative_fraction": 10 / 11,
        "negative_max_records": None,
        "init_from_checkpoint": "checkpoints/p2/stage1/stage1.pt",
        "eval_at_step0": True,
    }
    cfg.update(over)
    return cfg


def _report(*, mix=..., warm=..., config=None, **over) -> dict:
    rep: dict = {"diagnostics": {"config": _config(**(config or {}))}}
    if mix is not ...:
        rep["negative_mix"] = mix
    if warm is not ...:
        rep["warm_start"] = warm
    rep.update(over)
    return rep


# ══════════════════════════════════════════════════════════════════════════════════════
# negative_mix_realized
# ══════════════════════════════════════════════════════════════════════════════════════
def test_clause_passes_on_a_realized_mix() -> None:
    assert _negative_mix_ok(_report(mix=_mix(), warm=_warm())) is True


def test_gate_is_false_when_a_mix_was_requested_but_never_measured() -> None:
    """THE test this file exists for, half one.

    The run configured a pool and a fraction; the report carries no `negative_mix` block.
    A clause phrased as "no mix violation was found" returns True here — there is no
    measurement to violate. It must return False.
    """
    assert _negative_mix_ok(_report(mix=..., warm=...)) is False


def test_absence_is_allowed_only_for_a_run_that_requested_nothing() -> None:
    """A pre-P2-10d report stays valid; a P2-10d-configured one does not get the pass.

    P2-04's and P2-09's committed artifacts predate every one of these fields, and
    regenerating them to add a key would forge a measurement neither run made
    (CLAUDE.md §10.3). Their `diagnostics.config` therefore has none of these keys at all.
    """
    legacy = {"diagnostics": {"config": {"seed": 42, "epochs": 10}}}
    assert _negative_mix_ok(legacy) is True
    # ...but each individual leftover marker withdraws the pass.
    for over in (
        {"negative_pool_parquet": "x.parquet", "negative_fraction": 0.0},
        {"negative_fraction": 0.5, "negative_pool_parquet": None},
        {
            "init_from_checkpoint": "ckpt.pt",
            "negative_fraction": 0.0,
            "negative_pool_parquet": None,
        },
    ):
        cfg = {"seed": 42, **over}
        assert _negative_mix_ok({"diagnostics": {"config": cfg}}) is False, over
    # A stray warm_start / step0 block without a mix block is equally incoherent.
    assert _negative_mix_ok({"diagnostics": {"config": {"seed": 42}}, "warm_start": None}) is False
    assert (
        _negative_mix_ok({"diagnostics": {"config": {"seed": 42}}, "step0_requested": False})
        is False
    )


def test_gate_is_false_when_the_realized_fraction_is_not_the_requested_one() -> None:
    """The identity is exact: 100 positive draws at f = 10/11 means exactly 1000 negatives."""
    for n_neg in (999, 1001, 0, 500):
        rep = _report(mix=_mix(stream=_stream(n_neg=n_neg)), warm=_warm())
        assert _negative_mix_ok(rep) is False, n_neg


def test_gate_is_false_when_the_counts_do_not_add_up() -> None:
    bad = {**_stream(), "n_total": 999}
    assert _negative_mix_ok(_report(mix=_mix(stream=bad), warm=_warm())) is False


def test_gate_is_false_when_the_pool_is_empty_but_negatives_were_drawn() -> None:
    """Draws from a pool of size zero cannot have happened."""
    bad = _stream(pool=0)
    assert _negative_mix_ok(_report(mix=_mix(stream=bad), warm=_warm())) is False


def test_gate_is_false_when_a_zero_fraction_run_loaded_a_pool() -> None:
    """f = 0 asserts an empty pool and zero draws, rather than skipping the check."""
    cfg = {"negative_pool_parquet": None, "negative_fraction": 0.0, "init_from_checkpoint": None}
    ok = _report(mix=_mix(stream=_stream(n_neg=0, pool=0)), warm=None, config=cfg)
    assert _negative_mix_ok(ok) is True
    loaded = _report(mix=_mix(stream=_stream(n_neg=0, pool=40)), warm=None, config=cfg)
    assert _negative_mix_ok(loaded) is False


def test_gate_is_false_when_the_recorded_fraction_is_not_a_real_number() -> None:
    for bad in (None, "0.9", True, float("nan")):
        rep = _report(mix=_mix(), warm=_warm(), config={"negative_fraction": bad})
        assert _negative_mix_ok(rep) is False, bad


def test_gate_is_false_on_a_malformed_mix_block() -> None:
    for bad in ("a mix", [], 7, {"consumed": {}}, {"stream": "no"}):
        assert _negative_mix_ok(_report(mix=bad, warm=_warm())) is False, bad


# ══════════════════════════════════════════════════════════════════════════════════════
# warm_start_loaded
# ══════════════════════════════════════════════════════════════════════════════════════
def test_warm_clause_passes_on_a_verified_load() -> None:
    assert _warm_start_ok(_report(mix=_mix(), warm=_warm())) is True


def test_gate_is_false_when_a_warm_start_was_requested_but_never_recorded() -> None:
    """THE test this file exists for, half two."""
    assert _warm_start_ok(_report(mix=_mix(), warm=None)) is False
    assert _warm_start_ok(_report(mix=_mix(), warm=...)) is False


def test_a_fresh_build_must_record_no_warm_start() -> None:
    """The absence branch is an assertion in both directions."""
    fresh = {"init_from_checkpoint": None}
    assert _warm_start_ok(_report(mix=_mix(), warm=None, config=fresh)) is True
    # A warm-start block on a run that never asked for one means the two disagree about
    # what the checkpoint even was.
    assert _warm_start_ok(_report(mix=_mix(), warm=_warm(), config=fresh)) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_tensors_differing_after", 1),
        ("n_tensors_differing_before", 0),
        ("n_missing_keys", 1),
        ("n_unexpected_keys", 1),
        ("n_shape_mismatch", 1),
        ("n_model_tensors", 0),
        ("n_checkpoint_tensors", 136),
        ("checkpoint", "some/other.pt"),
        ("checkpoint_sha256", "short"),
        ("checkpoint_sha256", None),
    ],
)
def test_every_field_the_warm_clause_reads_can_break_it(field: str, value) -> None:
    """Tamper with one field at a time: a clause that stopped reading it would pass here.

    ``n_tensors_differing_before = 0`` is the designed control's own test — a load into a
    model that already matched satisfies every other clause while proving nothing.
    """
    rep = _report(mix=_mix(), warm=_warm(**{field: value}))
    assert _warm_start_ok(rep) is False, (field, value)


# ══════════════════════════════════════════════════════════════════════════════════════
# The clauses reach the gate, and the validator re-derives them
# ══════════════════════════════════════════════════════════════════════════════════════
def test_both_clauses_are_in_derive_clauses() -> None:
    clauses = derive_clauses({})
    assert "negative_mix_realized" in clauses
    assert "warm_start_loaded" in clauses
    # An empty report requests nothing, so both take their (asserted) absence branch.
    assert clauses["negative_mix_realized"] is True
    assert clauses["warm_start_loaded"] is True


def test_validator_catches_a_clause_fabricated_true() -> None:
    """`overall_pass = all(clauses)` cannot catch this; the re-derivation must."""
    rep = _report(mix=_mix(stream=_stream(n_neg=1)), warm=_warm())
    derived = derive_clauses(rep)
    assert derived["negative_mix_realized"] is False
    forged = {**rep, "gate": {**derived, "negative_mix_realized": True, "overall_pass": True}}
    problems = validate_report(forged)
    assert any("negative_mix_realized" in p for p in problems), problems


def test_validator_rejects_a_malformed_mix_block_shape() -> None:
    rep = _report(mix=_mix(stream={"n_total": "many"}), warm=_warm())
    problems = validate_report(rep)
    assert any("negative_mix.stream.n_total" in p for p in problems), problems


def test_validator_rejects_consuming_more_than_the_stream_holds() -> None:
    rep = _report(
        mix=_mix(consumed={**_stream(), "n_total": 5000}, epochs=1),
        warm=_warm(),
    )
    problems = validate_report(rep)
    assert any(p.startswith("negative_mix: consumed") for p in problems), problems


def test_validator_rejects_a_malformed_warm_block() -> None:
    rep = _report(mix=_mix(), warm=_warm(n_model_tensors=None, checkpoint=""))
    problems = validate_report(rep)
    assert any("warm_start.n_model_tensors" in p for p in problems), problems
    assert any("warm_start.checkpoint" in p for p in problems), problems


# ══════════════════════════════════════════════════════════════════════════════════════
# The schema-version excuse must not widen (it is the one fail-open shape here)
# ══════════════════════════════════════════════════════════════════════════════════════
def _graded(rep: dict, *, drop: str, schema: str) -> dict:
    """`rep` with a full re-derived gate, minus one clause, at a stated schema version."""
    derived = derive_clauses(rep)
    stored = {k: v for k, v in derived.items() if k != drop}
    return {
        **rep,
        "schema_version": schema,
        "gate": {**stored, "overall_pass": all(derived.values())},
    }


def test_a_current_schema_report_may_not_omit_a_clause() -> None:
    """The excuse is for artifacts written before the clause existed — nothing else.

    Without this, widening the excuse to "any missing clause is fine" would leave the whole
    suite green while every future report could drop a clause it failed.
    """
    from tbox_finder.train.train_stage1 import SCHEMA_VERSION

    rep = _report(mix=_mix(), warm=_warm())
    problems = validate_report(_graded(rep, drop="warm_start_loaded", schema=SCHEMA_VERSION))
    assert any("gate.warm_start_loaded: missing" in p for p in problems), problems


def test_an_older_schema_report_is_excused_only_when_the_clause_is_true() -> None:
    """An old artifact still cannot hide a FAILING clause behind an absent key."""
    passing = _report(mix=_mix(), warm=_warm())
    assert derive_clauses(passing)["warm_start_loaded"] is True
    assert not [
        p
        for p in validate_report(_graded(passing, drop="warm_start_loaded", schema="1"))
        if "warm_start_loaded" in p
    ]
    # Same schema, same missing key — but now the clause re-derives False.
    failing = _report(mix=_mix(), warm=None)
    assert derive_clauses(failing)["warm_start_loaded"] is False
    problems = validate_report(_graded(failing, drop="warm_start_loaded", schema="1"))
    assert any("gate.warm_start_loaded: missing" in p for p in problems), problems


def test_an_unparseable_schema_version_gets_no_excuse() -> None:
    """Fails closed: a report that cannot say when it was written is not excused."""
    rep = _report(mix=_mix(), warm=_warm())
    for schema in ("", "one", None):
        problems = validate_report(_graded(rep, drop="warm_start_loaded", schema=schema))
        assert any("gate.warm_start_loaded: missing" in p for p in problems), schema


def test_schema_version_ordering_is_numeric_not_lexicographic() -> None:
    """`"10" < "2"` as strings; a future schema 10 must not be excused as "older"."""
    from tbox_finder.train.train_stage1 import _schema_precedes

    assert _schema_precedes("1", "2") is True
    assert _schema_precedes("2", "2") is False
    assert _schema_precedes("10", "2") is False
    assert _schema_precedes(None, "2") is False
    assert _schema_precedes("v1", "2") is False


def test_the_warm_clause_compares_paths_not_spellings() -> None:
    """CodeRabbit r1: `warm_start` records `str(Path(...))`, so the clause must normalise.

    A config spelt `./ckpt.pt` would otherwise compare unequal to its own recorded path
    and fail a load that in fact succeeded — while a genuinely different checkpoint must
    still fail.
    """
    warm = _warm(checkpoint="checkpoints/p2/stage1/stage1.pt")
    for spelling in (
        "checkpoints/p2/stage1/stage1.pt",
        "./checkpoints/p2/stage1/stage1.pt",
        "checkpoints/p2//stage1/stage1.pt",
    ):
        rep = _report(mix=_mix(), warm=warm, config={"init_from_checkpoint": spelling})
        assert _warm_start_ok(rep) is True, spelling
    other = _report(
        mix=_mix(), warm=warm, config={"init_from_checkpoint": "checkpoints/p2/other.pt"}
    )
    assert _warm_start_ok(other) is False


# ══════════════════════════════════════════════════════════════════════════════════════
# rc_directionality_preserved (P2-12 RESULT) — the same absence-is-an-assertion shape
# ══════════════════════════════════════════════════════════════════════════════════════
# ADR-0005 D15 forbids the order-destroying symmetric fwd/RC average. Before this clause the
# constraint was enforced only on the mode NAME at config time — but "gate" is
# g*fwd + (1-g)*rc with a LEARNED g, and it *becomes* the forbidden average at g == 0.5.
# The name cannot see that, nothing in training prevents it (weight_decay pulls the logit
# toward 0), and the module property that claimed to check it returned a literal True with
# no callers. So the clause re-derives from a census of the trained gate.
def _gate_block(*, n_channels: int = 256, n_at_half: int = 0, mode: str = "gate") -> dict:
    return {
        "mode": mode,
        "structural": False,
        "n_channels": n_channels,
        "n_channels_at_half": n_at_half,
        "gate_min": 0.61,
        "gate_max": 0.63,
        "gate_mean": 0.62,
        "abs_dev_from_half_max": 0.13,
        "abs_dev_from_half_mean": 0.12,
        "atol": 1e-6,
    }


def _rc_report(*, block=..., mode: str = "gate") -> dict:
    rep = _report(mix=_mix(), warm=_warm(), config={"rc_combine": mode})
    if block is not ...:
        rep["rc_combine"] = block
    return rep


def test_a_gate_run_that_measured_its_gate_passes() -> None:
    assert _rc_combine_ok(_rc_report(block=_gate_block())) is True


def test_a_gate_collapsed_to_the_symmetric_average_fails() -> None:
    """The whole point: every channel at 0.5 IS the forbidden mean, under the name "gate"."""
    collapsed = _gate_block(n_channels=256, n_at_half=256)
    assert _rc_combine_ok(_rc_report(block=collapsed)) is False
    # One channel off 0.5 is no longer the exact symmetric average.
    assert _rc_combine_ok(_rc_report(block=_gate_block(n_at_half=255))) is True


def test_a_gate_run_that_recorded_no_census_fails_rather_than_passing_on_absence() -> None:
    """This is what keeps the unmeasured P2-12 rc_gate leg out, not a skip."""
    assert _rc_combine_ok(_rc_report(block=..., mode="gate")) is False


def test_a_concat_run_needs_no_census() -> None:
    """concat is directionality-preserving structurally — its absence is not a gap."""
    assert _rc_combine_ok(_rc_report(block=..., mode="concat")) is True
    concat_census = {"mode": "concat", "structural": True}
    assert _rc_combine_ok(_rc_report(block=concat_census, mode="concat")) is True


def test_a_block_describing_another_mode_is_not_evidence_about_this_run() -> None:
    """A concat census attached to a gate run would otherwise wave the gate through."""
    rep = _rc_report(block={"mode": "concat", "structural": True}, mode="gate")
    assert _rc_combine_ok(rep) is False


def test_a_forbidden_mode_name_in_the_block_fails() -> None:
    rep = _report(mix=_mix(), warm=_warm(), config={"rc_combine": "mean"})
    rep["rc_combine"] = {"mode": "mean", "structural": True}
    assert _rc_combine_ok(rep) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_channels", None),
        ("n_channels", 0),
        ("n_channels", "256"),
        ("n_channels_at_half", None),
        ("n_channels_at_half", -1),
        ("mode", ""),
    ],
)
def test_every_field_the_rc_clause_reads_can_break_it(field, value) -> None:
    block = _gate_block()
    block[field] = value
    assert _rc_combine_ok(_rc_report(block=block)) is False


def test_a_malformed_rc_block_fails_closed() -> None:
    assert _rc_combine_ok(_rc_report(block=["not", "a", "mapping"])) is False


def test_the_rc_clause_is_in_derive_clauses_and_is_version_gated() -> None:
    from tbox_finder.train.train_stage1 import CLAUSE_SCHEMA_VERSION

    assert "rc_directionality_preserved" in derive_clauses(_rc_report(block=_gate_block()))
    # Version-gated so the committed schema-3 reports stay valid without back-filling a
    # measurement nobody took (§10.3) — while a schema-3 GATE run still re-derives False.
    assert CLAUSE_SCHEMA_VERSION["rc_directionality_preserved"] == "4"
    assert derive_clauses(_rc_report(block=..., mode="concat"))["rc_directionality_preserved"]
    assert not derive_clauses(_rc_report(block=..., mode="gate"))["rc_directionality_preserved"]
