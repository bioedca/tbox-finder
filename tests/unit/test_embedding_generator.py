"""Sabotage gates for the junction-control GENERATOR (P2-10d′-b, ADR-0005 A7 pin 7).

`tests/unit/test_embedding.py` asserts that the control the generator emits *today* is
matched. That is not enough. Per [[control-matchedness-must-be-asserted]]: the arms could
be matched today and the generator still be wrong, and a later change would then ship a
non-matched control while every fixture-reading test stayed green.

So this file attacks from the other side. It builds the plausible-wrong controls a future
edit might produce — a copy of the treatment arm, a shifted phase, a re-composed insert,
an unspliced host — and asserts that :func:`arms_are_matched`, the checker the gate
actually consumes, **rejects each one**. A guard that cannot fail is not a guard, and
[[sabotage-attribution-must-name-the-test]] applies: each sabotage is asserted against the
specific clause it should trip, not merely against "something went red".
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from tbox_finder.data.embedding import (
    EmbeddedWindow,
    EmbeddingError,
    embed_decoy_rows,
    junction_control,
)
from tbox_finder.eval.junction_probe import arms_are_matched

WINDOW = 64
SEED = 20260721


def _hosts(n: int) -> list[dict[str, object]]:
    out = []
    for i in range(n):
        raw = hashlib.shake_256(f"host:{i}".encode()).digest(WINDOW)
        out.append(
            {
                "candidate_id": f"rec{i}:lead",
                "sequence": "".join("ACGT"[b & 3] for b in raw),
                "source_record_id": f"rec{i}",
            }
        )
    return out


def _host_map(hosts: list[dict[str, object]]) -> dict[str, str]:
    return {str(h["candidate_id"]): str(h["sequence"]) for h in hosts}


def _decoy_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool in ("dinuc_shuffled", "structured_rna"):
        for i in range(4):
            rows.append(
                {
                    "decoy_id": f"{pool}_{i}",
                    "pool": pool,
                    "sequence": "GGGGCCCCAAAATTTT"[: 10 + i],
                    "masked": False,
                }
            )
    for pool in ("gc_background", "leader_decoy"):
        rows.append(
            {"decoy_id": f"{pool}_0", "pool": pool, "sequence": "ACGTACGT", "masked": False}
        )
    return rows


def _arms() -> tuple[list[EmbeddedWindow], list[EmbeddedWindow], list[dict[str, object]]]:
    hosts = _hosts(24)
    decoy, _ = embed_decoy_rows(_decoy_rows(), hosts, seed=SEED, window=WINDOW)
    control = junction_control(decoy, hosts, seed=SEED, window=WINDOW)
    return decoy, control, hosts


# ── the must-fire baseline ───────────────────────────────────────────────────────────
def test_the_shipped_generator_produces_a_matched_pair() -> None:
    """MUST FIRE. If this is false every sabotage below is vacuous — they would all be
    rejecting an arm that was already broken ([[clauses-must-guard-emptiness]])."""
    decoy, control, _ = _arms()
    ok, detail = arms_are_matched(decoy, control, _host_map(_hosts(24)))
    assert ok, detail
    assert detail["n_pairs"] == len(decoy) > 0


# ── sabotage 1: the control is a copy of the treatment arm ───────────────────────────
def test_matchedness_rejects_a_control_copied_from_the_decoy() -> None:
    """The exact failure P2-10c shipped and caught: an arm the generator copies verbatim
    is matched on every geometric field and proves nothing."""
    decoy, _, _ = _arms()
    copied = [dataclasses.replace(w, arm="junction_control") for w in decoy]
    ok, detail = arms_are_matched(decoy, copied, _host_map(_hosts(24)))
    assert not ok
    assert detail["n_identical_inside"] == len(decoy), detail


# ── sabotage 2: the control never actually spliced ───────────────────────────────────
def test_matchedness_rejects_an_unspliced_host_as_a_control() -> None:
    """An identity splice carries no junction, so it would report the junction as
    invisible whatever the junction did — while passing every geometry assertion."""
    decoy, control, hosts = _arms()
    by_id = {h["candidate_id"]: h["sequence"] for h in hosts}
    unspliced = [dataclasses.replace(c, sequence=str(by_id[c.host_id])) for c in control]
    ok, detail = arms_are_matched(decoy, unspliced, _host_map(hosts))
    assert not ok
    # Name the clause that did the rejecting. The first version of this test asserted
    # `n_identical_inside == 0 or n_flank_mismatch == 0`, which is TRUE for a correctly
    # matched pair too — a disjunction of two zeros that could only fail if BOTH counters
    # were non-zero, i.e. it pinned nothing ([[sabotage-attribution-must-name-the-test]]).
    # The clause that actually catches an unspliced control is `n_control_unspliced`.
    assert detail["n_control_unspliced"] == len(decoy), detail
    assert detail["n_identical_inside"] == 0
    assert detail["n_flank_mismatch"] == 0
    assert detail["n_pairs"] == len(decoy)


# ── sabotage 3: the phase drifts ─────────────────────────────────────────────────────
def test_matchedness_rejects_a_control_at_a_different_phase() -> None:
    decoy, control, _ = _arms()
    shifted = [dataclasses.replace(c, phase=c.phase + 1) for c in control]
    ok, detail = arms_are_matched(decoy, shifted, _host_map(_hosts(24)))
    assert not ok
    assert detail["n_geometry_mismatch"] == len(decoy), detail


def test_matchedness_rejects_a_control_with_a_different_insert_length() -> None:
    decoy, control, _ = _arms()
    resized = [dataclasses.replace(c, insert_len=c.insert_len + 1) for c in control]
    ok, detail = arms_are_matched(decoy, resized, _host_map(_hosts(24)))
    assert not ok
    assert detail["n_geometry_mismatch"] == len(decoy), detail


# ── sabotage 4: the control gets a different host ────────────────────────────────────
def test_matchedness_rejects_a_control_built_on_a_different_host() -> None:
    """Different host = the flanking DNA differs, so the arms differ somewhere other than
    the interval under test and any separation is unattributable."""
    decoy, control, _ = _arms()
    rotated = [
        dataclasses.replace(control[(i + 1) % len(control)], phase=d.phase, insert_len=d.insert_len)
        for i, d in enumerate(decoy)
    ]
    ok, _ = arms_are_matched(decoy, rotated, _host_map(_hosts(24)))
    assert not ok


# ── sabotage 5: emptiness and shape ──────────────────────────────────────────────────
@pytest.mark.parametrize("bad", ["empty_decoy", "empty_control", "unequal"])
def test_matchedness_is_false_on_degenerate_arms(bad: str) -> None:
    """Never TRUE-by-vacuity: 'no pairs disagreed' must not read as 'the arms matched'."""
    decoy, control, _ = _arms()
    pair = {
        "empty_decoy": ([], control),
        "empty_control": (decoy, []),
        "unequal": (decoy, control[:-1]),
    }[bad]
    ok, detail = arms_are_matched(*pair, _host_map(_hosts(24)))
    assert not ok
    assert "reason" in detail


# ── the property that distinguishes a REAL donor from a synthesised one ──────────────
def test_control_insert_is_real_dna_lifted_from_another_window_at_the_same_offsets() -> None:
    """The discriminating clause for 'background→background'.

    A composition-matched *synthetic* donor would satisfy every matchedness test above
    while changing what the control measures: the arm has to be real DNA, because 'real
    DNA spliced into real DNA' is the null the junction claim is measured against. So the
    control's inserted segment must appear **verbatim, at the same offsets**, in some
    other window of the host pool.
    """
    _, control, hosts = _arms()
    by_id = {h["candidate_id"]: str(h["sequence"]) for h in hosts}
    for c in control:
        lo, hi = c.junctions
        assert c.insert_id in by_id, f"donor {c.insert_id!r} is not a real pool window"
        assert c.sequence[lo:hi] == by_id[c.insert_id][lo:hi]
        assert c.insert_id != c.host_id


def test_control_is_deterministic_and_moves_with_the_seed() -> None:
    decoy, _, hosts = _arms()
    a = junction_control(decoy, hosts, seed=SEED, window=WINDOW)
    b = junction_control(decoy, hosts, seed=SEED, window=WINDOW)
    c = junction_control(decoy, hosts, seed=SEED + 1, window=WINDOW)
    assert [w.sequence for w in a] == [w.sequence for w in b]
    assert [w.sequence for w in a] != [w.sequence for w in c]


def test_a_degenerate_host_pool_is_refused_rather_than_silently_unspliced() -> None:
    """If every donor would reproduce the host's own bases the control has no junction.
    The generator must say so — a quiet identity splice is the worst outcome, because it
    reports 'junction invisible' with no junction present."""
    flat = [
        {"candidate_id": f"rec{i}:lead", "sequence": "A" * WINDOW, "source_record_id": f"rec{i}"}
        for i in range(8)
    ]
    decoy, _ = embed_decoy_rows(_decoy_rows(), flat, seed=SEED, window=WINDOW)
    with pytest.raises(EmbeddingError, match="degenerate over that interval"):
        junction_control(decoy, flat, seed=SEED, window=WINDOW)


# ── sabotage 6: the host is unresolvable, so the splice claim cannot be checked ───────
def test_matchedness_rejects_arms_whose_host_cannot_be_resolved() -> None:
    """`n_host_unresolved` exists so that "could not check" never reads as "checked and
    clean". Without a test it is the one clause in the checker that could be deleted
    outright and leave every other assertion green — while a caller passing an empty or
    stale host map would get a silent pass on the splice check.
    """
    decoy, control, _ = _arms()
    ok, detail = arms_are_matched(decoy, control, {})
    assert not ok
    assert detail["n_host_unresolved"] == len(decoy), detail
    # And it must be THIS clause, not a coincidental geometry or flank failure.
    assert detail["n_geometry_mismatch"] == 0
    assert detail["n_flank_mismatch"] == 0
    assert detail["n_control_unspliced"] == 0


def test_a_stale_host_map_does_not_pass_by_resolving_the_wrong_dna() -> None:
    """A host map keyed correctly but carrying the WRONG sequences must not silently
    satisfy the splice check: the comparison is against the host's actual bases."""
    decoy, control, hosts = _arms()
    stale = {str(h["candidate_id"]): "A" * WINDOW for h in hosts}
    ok, detail = arms_are_matched(decoy, control, stale)
    assert ok is True  # the control genuinely differs from an all-A "host"
    assert detail["n_host_unresolved"] == 0
