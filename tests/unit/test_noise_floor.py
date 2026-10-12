"""P3-16′-remint unit — the matched-null noise floor's pure functions and its validator.

What is pinned here, and why each one can fail
----------------------------------------------
* **The mint's operation is an equal-length OVERWRITE.** ``splice`` is asserted piecewise
  (prefix, window, suffix) and by length, so an implementation that *inserted* would differ by a
  measurable 24 nt rather than by a flag. The matchedness property that follows from an
  overwrite — agreement **outside both** insert windows — is pinned against the obvious-but-wrong
  "excise the insert from both and compare the remainders", which is run here and shown to
  succeed on **0 of 6** correctly built items ([[matched-control-before-certifying]]).
* **The nine matchedness controls, each breakable alone.** A hand-built, internally consistent
  null set fires all nine; nine sabotages each leave the other eight firing, because a forge that
  breaks two coupled controls at once cannot say which one bit
  ([[all-true-fixture-cannot-test-a-conjunction]]). Several sabotages had to be constructed
  *inside* the union of the two insert windows precisely so the remaining controls stay satisfied.
* **Arm identity, not arm counts.** The two ``n2`` arms are equal-sized, so a count-only assert
  survives swapping them ([[symmetric-count-fixture-blind-to-inversion]]). ``n2_scramble1`` is
  pinned to the ``f"{SEED}:n2_scramble1"`` stream by replay, and asserted to *differ* from
  ``n2_scramble2``.
* **The padding rule.** ``cell_stats`` quantiles are over sequences searched, not over hits. The
  same hit set is passed with two different denominators and the median moves 5.5 → 10.0; both
  numbers are computed by hand in a comment, never by calling the function under test.
* **The bit column.** ``best_by_target`` is pinned on tblout rows where the best **bit** score
  carries the *worst* E-value and the *weaker* ``inc`` flag, so an implementation reading either
  of those columns returns a different, named value.
* **The validator.** A synthetic — and deliberately self-consistent — report validates clean, and
  then each clause is broken alone, with every other field adjusted to stay consistent.
* **The three CLI legs, over a ``tmp_path`` tree.** ``prep`` / ``search`` / ``report`` are driven
  end to end with no ``cmsearch`` and no real data: three *external boundaries* are stubbed (the
  parquet pool read, ``cmsearch -h``, and the binary) and nothing a leg test asserts about is
  replaced ([[in-process-no-ops-look-like-compliance]]). Eleven single-edit defects an
  adversarial audit found uncovered each fail a test that names the property — a validator that
  fails open, a matchedness block copied forward from ``prep.json`` instead of re-derived, a
  ``-T`` that is not 0, a failed prep that exits 0, FASTAs written before the certification that
  covers them, the two dropped digest guards, a report that exits 0 beside its own problems, a
  per-search instead of per-arm denominator, and the two defaults that put the artifacts where
  the next leg and ``host_screen`` look for them.

Nothing here measures the real nulls, and nothing here runs ``cmsearch``. The synthetic report
carries invented counts and is labelled as such, the planted tblout bit scores in the leg section
are invented likewise, and the measured floor lives in
``reports/p3/postmint_noise_floor.json``.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import random
import statistics
from pathlib import Path

import pytest

from tbox_finder import infernal, provenance
from tbox_finder.data import window_dataset as wd
from tbox_finder.integration import host_screen as hs
from tbox_finder.integration import noise_floor as nf

# ══════════════════════════════════════════════════════════════════════════════════════
# Synthetic sequence fixtures
# ══════════════════════════════════════════════════════════════════════════════════════
_HOST_NT = 400
_INSERT_NT = 24


def _random_dna(nt: int, seed: int) -> str:
    """A deterministic pseudo-random ACGT string.

    Pseudo-random rather than a repeat of ``ACGT``: a periodic host would make
    ``host[:p] + host[p + L:] == host[:m] + host[m + L:]`` for many ``(p, m)`` pairs, and the
    excise-and-compare trap below would then pass for the wrong reason.
    """
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(nt))


#: Two hosts, so the fixture carries both the shared-host case (items 0 and 1) that makes the
#: "a scramble reproduced a minted contig" sabotage constructible and a second host besides.
_HOSTS: dict[str, str] = {
    "cand:H0": _random_dna(_HOST_NT, 1),
    "cand:H1": _random_dna(_HOST_NT, 2),
}
_INSERTS: list[str] = [
    _random_dna(_INSERT_NT, 11),
    _random_dna(_INSERT_NT, 12),
    _random_dna(_INSERT_NT, 13),
]

_ITEMS: list[dict] = [
    {
        "contig_id": "dec:000",
        "host_id": "cand:H0",
        "splice_phase": 10,
        "locus_length": _INSERT_NT,
        "label": 0,
        "pool": "dinuc_shuffled",
    },
    {
        "contig_id": "dec:001",
        "host_id": "cand:H0",
        "splice_phase": 120,
        "locus_length": _INSERT_NT,
        "label": 0,
        "pool": "decoys_v0",
    },
    {
        "contig_id": "dec:002",
        "host_id": "cand:H1",
        "splice_phase": 230,
        "locus_length": _INSERT_NT,
        "label": 0,
        "pool": "decoys_v0",
    },
]

#: Hand-chosen scramble phases, one per item per arm. Every shift is ≥ MIN_PHASE_SHIFT and every
#: window fits: |200-10|=190, |300-120|=180, |50-230|=180 / |300-10|=290, |40-120|=80,
#: |100-230|=130. Hand-chosen rather than drawn, so a sabotage can move exactly one of them.
_SCRAMBLE1 = [200, 300, 50]
_SCRAMBLE2 = [300, 40, 100]

_ARM_PREFIX = (("n2_scramble1", "s1_", _SCRAMBLE1), ("n2_scramble2", "s2_", _SCRAMBLE2))

_CONTROL_NAMES = (
    "n1_covers_the_gated_negatives",
    "n2_length_matches_the_minted_contig",
    "n2_carries_its_insert_exactly_once",
    "n2_insert_is_the_minted_insert",
    "n2_agrees_outside_both_insert_windows",
    "n2_reproduces_no_minted_contig",
    "n2_phase_moved_far_enough",
    "n3_is_disjoint_from_the_drawn_hosts",
    "n3_is_the_requested_size",
)


@functools.lru_cache(maxsize=1)
def _pool_records() -> tuple[tuple[str, str], ...]:
    """``N_POOL_WINDOWS`` host-channel windows, built once.

    The real count, not a monkeypatched one: ``n3_is_the_requested_size`` compares against
    :data:`noise_floor.N_POOL_WINDOWS`, and relaxing the constant to shrink the fixture would
    monkeypatch the very precondition the control exists to assert
    ([[in-process-no-ops-look-like-compliance]]).
    """
    base = _random_dna(2000, 97)
    return tuple(
        (f"p{index:04d}", base[index % 1900 : index % 1900 + 40])
        for index in range(nf.N_POOL_WINDOWS)
    )


def _hand_built() -> dict:
    """A consistent null set, built by hand so every phase is known and separately breakable."""
    hosts = dict(_HOSTS)
    items = copy.deepcopy(_ITEMS)
    minted = {
        item["contig_id"]: nf.splice(hosts[item["host_id"]], _INSERTS[index], item["splice_phase"])
        for index, item in enumerate(items)
    }
    records: dict[str, dict[str, str]] = {arm: {} for arm in nf.ARMS}
    idmap: dict[str, dict[str, dict]] = {arm: {} for arm in nf.ARMS}
    for index, item in enumerate(items):
        tag = f"{index:04d}"
        records["n1_inserts"][f"i{tag}"] = _INSERTS[index]
        idmap["n1_inserts"][f"i{tag}"] = {
            "contig_id": item["contig_id"],
            "insert_nt": _INSERT_NT,
        }
        for arm, prefix, phases in _ARM_PREFIX:
            records[arm][f"{prefix}{tag}"] = nf.splice(
                hosts[item["host_id"]], _INSERTS[index], phases[index]
            )
            idmap[arm][f"{prefix}{tag}"] = {
                "contig_id": item["contig_id"],
                "host_id": item["host_id"],
                "minted_phase": item["splice_phase"],
                "scramble_phase": phases[index],
                "insert_nt": _INSERT_NT,
            }
    for name, sequence in _pool_records():
        records["n3_pool"][name] = sequence
        idmap["n3_pool"][name] = {"host_id": f"pool:{name}"}
    return {
        "hosts": hosts,
        "items": items,
        "minted": minted,
        "records": records,
        "idmap": idmap,
        "drawn": {"cand:H0", "cand:H1"},
    }


def _checks(fixture: dict) -> dict[str, dict]:
    rows = nf.matchedness(
        fixture["items"],
        fixture["minted"],
        fixture["records"],
        fixture["idmap"],
        fixture["drawn"],
    )
    return {row["name"]: row for row in rows}


# ══════════════════════════════════════════════════════════════════════════════════════
# splice — the mint's equal-length overwrite
# ══════════════════════════════════════════════════════════════════════════════════════
def test_splice_overwrites_the_window_and_does_not_displace_the_host():
    """An insertion would lengthen the contig by 24 nt and shift every downstream base.

    Asserted as three identities rather than as a length alone: a length check passes an
    implementation that overwrote the WRONG window.
    """
    host, insert, phase = _HOSTS["cand:H0"], _INSERTS[0], 10
    spliced = nf.splice(host, insert, phase)
    assert len(spliced) == len(host)
    assert spliced[:phase] == host[:phase]
    assert spliced[phase : phase + _INSERT_NT] == insert
    assert spliced[phase + _INSERT_NT :] == host[phase + _INSERT_NT :]
    # The insertion form is a different string of a different length, named here so the
    # distinction is pinned rather than implied.
    assert spliced != host[:phase] + insert + host[phase:]
    # The host bases under the window are destroyed, which is why the n2 matchedness test cannot
    # excise and compare remainders.
    assert spliced[phase : phase + _INSERT_NT] != host[phase : phase + _INSERT_NT]


def test_splice_accepts_an_insert_that_exactly_fills_the_tail_and_refuses_one_nt_more():
    """The boundary, both sides: a refusal that fired one nt early would truncate every host."""
    host = _HOSTS["cand:H0"]
    exact = nf.splice(host, "A" * 24, len(host) - 24)
    assert len(exact) == len(host) and exact.endswith("A" * 24)
    with pytest.raises(nf.NoiseFloorError, match="overruns the host"):
        nf.splice(host, "A" * 24, len(host) - 23)


def test_splice_refuses_a_negative_phase_rather_than_wrapping():
    """Python slicing would silently treat ``-5`` as an offset from the 3′ end."""
    with pytest.raises(nf.NoiseFloorError, match="overruns the host"):
        nf.splice(_HOSTS["cand:H0"], _INSERTS[0], -5)
    # Positive control: phase 0 is legal and is not confused with the negative case.
    assert nf.splice(_HOSTS["cand:H0"], _INSERTS[0], 0).startswith(_INSERTS[0])


# ══════════════════════════════════════════════════════════════════════════════════════
# scramble_phase
# ══════════════════════════════════════════════════════════════════════════════════════
def test_scramble_phase_is_always_far_enough_and_inside_the_window():
    """A phase outside ``[0, window - insert]`` would make the splice overrun the host."""
    for seed in range(40):
        for phase in (0, 10, 150, 376):
            moved = nf.scramble_phase(
                random.Random(seed), window_nt=_HOST_NT, insert_nt=_INSERT_NT, phase=phase
            )
            assert abs(moved - phase) >= nf.MIN_PHASE_SHIFT, (seed, phase, moved)
            assert 0 <= moved <= _HOST_NT - _INSERT_NT, (seed, phase, moved)


def test_scramble_phase_draws_exactly_the_legal_set_including_both_endpoints():
    """Pins the candidate range as inclusive at ``window - insert`` and the shift as ``>=``.

    window 120, insert 10 -> candidates 0..110 inclusive; |c - 60| >= 50 -> c <= 10 or c >= 110,
    i.e. exactly {0,…,10, 110} — 12 values, computed here by hand. A half-open upper bound would
    drop 110; a ``>`` shift test would drop both 10 and 110.
    """
    legal = set(range(0, 11)) | {110}
    drawn = {
        nf.scramble_phase(random.Random(seed), window_nt=120, insert_nt=10, phase=60)
        for seed in range(400)
    }
    assert drawn == legal


def test_scramble_phase_is_deterministic_for_a_given_seeded_random():
    """Two equally seeded streams must agree, or prep and report would describe different bytes."""
    first = nf.scramble_phase(
        random.Random("42:n2_scramble1"), window_nt=_HOST_NT, insert_nt=_INSERT_NT, phase=10
    )
    second = nf.scramble_phase(
        random.Random("42:n2_scramble1"), window_nt=_HOST_NT, insert_nt=_INSERT_NT, phase=10
    )
    assert first == second
    # And a different stream is a different draw, so the determinism above is not a constant.
    others = {
        nf.scramble_phase(
            random.Random(f"42:{tag}"), window_nt=_HOST_NT, insert_nt=_INSERT_NT, phase=10
        )
        for tag in ("n2_scramble1", "n2_scramble2")
    }
    assert len(others) == 2


def test_scramble_phase_refuses_a_geometry_with_no_legal_phase_rather_than_relaxing():
    """A control re-spliced 3 nt away is the same junction, and the relaxation is invisible.

    window 100, insert 60 -> candidates 0..40, so every candidate is within 40 nt of phase 20 and
    none clears MIN_PHASE_SHIFT = 50.
    """
    with pytest.raises(nf.NoiseFloorError, match="no phase"):
        nf.scramble_phase(random.Random(0), window_nt=100, insert_nt=60, phase=20)
    # Positive control: the SAME insert and phase in a window that leaves a legal phase is
    # accepted, so the refusal above is not a guard that refuses everything
    # ([[raises-test-needs-a-positive-control]]).
    moved = nf.scramble_phase(random.Random(0), window_nt=200, insert_nt=60, phase=20)
    assert moved >= 70 and moved <= 140


# ══════════════════════════════════════════════════════════════════════════════════════
# gated_negatives
# ══════════════════════════════════════════════════════════════════════════════════════
def _benchmark() -> dict:
    return {
        "items": [
            {"contig_id": "dec:000", "label": 0, "pool": "dinuc_shuffled"},
            {"contig_id": "pos:000", "label": 1, "pool": "corpus"},
            {"contig_id": "dec:leader", "label": 0, "pool": "leader_decoy"},
            {"contig_id": "dec:001", "label": 0, "pool": "decoys_v0"},
        ]
    }


def test_gated_negatives_keeps_label_0_in_benchmark_order_and_drops_the_leader_decoy():
    """The leader decoy is T-box-derived, so including it would put a T-box in the null."""
    items = nf.gated_negatives(_benchmark())
    assert [item["contig_id"] for item in items] == ["dec:000", "dec:001"]


def test_gated_negatives_returns_copies_so_a_caller_cannot_edit_the_benchmark():
    benchmark = _benchmark()
    nf.gated_negatives(benchmark)[0]["pool"] = "mutated"
    assert benchmark["items"][0]["pool"] == "dinuc_shuffled"


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"contig_id": "pos:000", "label": 1, "pool": "corpus"}],
        [{"contig_id": "dec:leader", "label": 0, "pool": "leader_decoy"}],
    ],
    ids=["no-items", "only-positives", "only-the-leader-decoy"],
)
def test_gated_negatives_refuses_a_benchmark_that_leaves_no_null(items):
    """An empty null arm is vacuously clean: 0 hits over 0 sequences is not a measurement."""
    with pytest.raises(nf.NoiseFloorError, match="no gated label-0 items"):
        nf.gated_negatives({"items": items})
    # Positive control: the same code path accepts a benchmark that does carry one.
    assert nf.gated_negatives(_benchmark())


# ══════════════════════════════════════════════════════════════════════════════════════
# build_nulls
# ══════════════════════════════════════════════════════════════════════════════════════
def _build_inputs() -> tuple[list[dict], dict[str, str], dict[str, str], list[tuple[str, str]]]:
    items = copy.deepcopy(_ITEMS)
    hosts = dict(_HOSTS)
    minted = {
        item["contig_id"]: nf.splice(hosts[item["host_id"]], _INSERTS[index], item["splice_phase"])
        for index, item in enumerate(items)
    }
    pool = [(f"pool:{name}", sequence) for name, sequence in _pool_records()]
    return items, minted, hosts, pool


def test_build_nulls_is_deterministic_across_two_calls():
    """Prep and a re-run must write the same bytes, or the search digests stop binding."""
    first_records, first_built = nf.build_nulls(*_build_inputs())
    second_records, second_built = nf.build_nulls(*_build_inputs())
    assert first_records == second_records
    assert first_built == second_built


def test_build_nulls_n1_is_the_insert_at_the_minted_phase_not_a_recomputed_decoy():
    """The insert is read out of the minted contig, which is the only place it exists as spliced."""
    items, minted, hosts, pool = _build_inputs()
    records, built = nf.build_nulls(items, minted, hosts, pool)
    for index, item in enumerate(items):
        phase, length = item["splice_phase"], item["locus_length"]
        expected = minted[item["contig_id"]][phase : phase + length]
        assert records["n1_inserts"][f"i{index:04d}"] == expected
        assert expected == _INSERTS[index]
        assert built["idmap"]["n1_inserts"][f"i{index:04d}"]["contig_id"] == item["contig_id"]


def test_build_nulls_scrambles_carry_the_insert_once_at_the_new_phase_and_host_bases_at_the_old():
    """The scramble is built from the POOL host, so the minted window holds host bases again.

    Asserted as an identity against the host, not as "differs from the insert": a scramble built
    from the minted contig would also differ from the insert at *some* positions.
    """
    items, minted, hosts, pool = _build_inputs()
    records, built = nf.build_nulls(items, minted, hosts, pool)
    for arm in ("n2_scramble1", "n2_scramble2"):
        for name, meta in built["idmap"][arm].items():
            scramble = records[arm][name]
            host = hosts[meta["host_id"]]
            phase, moved, length = (
                meta["minted_phase"],
                meta["scramble_phase"],
                meta["insert_nt"],
            )
            insert = minted[meta["contig_id"]][phase : phase + length]
            assert scramble[moved : moved + length] == insert
            assert scramble.count(insert) == 1
            assert scramble[phase : phase + length] == host[phase : phase + length]
            assert len(scramble) == len(host)


def test_build_nulls_binds_each_scramble_arm_to_its_own_named_seed_stream():
    """The two arms are equal-sized, so counts survive swapping them; this is an identity.

    ``n2_scramble1``'s phases are reproduced by replaying ``f"{SEED}:n2_scramble1"`` in item
    order, and asserted DIFFERENT from ``n2_scramble2``'s — an arm swap, or both arms sharing one
    stream, changes one of those two facts ([[symmetric-count-fixture-blind-to-inversion]]).
    """
    items, minted, hosts, pool = _build_inputs()
    _records, built = nf.build_nulls(items, minted, hosts, pool)
    replay = random.Random(f"{nf.SEED}:n2_scramble1")
    expected = [
        nf.scramble_phase(
            replay,
            window_nt=len(hosts[item["host_id"]]),
            insert_nt=item["locus_length"],
            phase=item["splice_phase"],
        )
        for item in items
    ]
    arm1 = [meta["scramble_phase"] for meta in built["idmap"]["n2_scramble1"].values()]
    arm2 = [meta["scramble_phase"] for meta in built["idmap"]["n2_scramble2"].values()]
    assert arm1 == expected
    assert arm2 != expected


def test_build_nulls_records_the_minimum_observed_shift_and_it_clears_the_pin():
    items, minted, hosts, pool = _build_inputs()
    _records, built = nf.build_nulls(items, minted, hosts, pool)
    shifts = [
        abs(meta["scramble_phase"] - meta["minted_phase"])
        for arm in ("n2_scramble1", "n2_scramble2")
        for meta in built["idmap"][arm].values()
    ]
    assert built["min_phase_shift_nt"] == min(shifts) >= nf.MIN_PHASE_SHIFT


def test_build_nulls_ids_are_zero_padded_so_insertion_order_is_id_order():
    """``write_fasta`` emits records in the mapping's own order, so the digest depends on it.

    Zero-padded tags are what make that order equal sorted-id order; ``i10`` beside ``i9`` would
    not, and the FASTA sha256 recorded in ``prep.json`` would then depend on the item count.
    """
    items, minted, hosts, pool = _build_inputs()
    records, _built = nf.build_nulls(items, minted, hosts, pool)
    for arm in nf.ARMS:
        names = list(records[arm])
        assert names == sorted(names), arm
        assert len({len(name) for name in names}) == 1, arm


def test_build_nulls_refuses_a_pool_host_of_a_different_length_from_the_minted_contig():
    """A host of another length is another window: every phase in the idmap would be off."""
    items, minted, hosts, pool = _build_inputs()
    hosts["cand:H0"] = hosts["cand:H0"][:-1]
    with pytest.raises(nf.NoiseFloorError, match="pool host is"):
        nf.build_nulls(items, minted, hosts, pool)
    # Positive control: the untruncated host is accepted by the same call.
    assert nf.build_nulls(*_build_inputs())[0]["n2_scramble1"]


def test_build_nulls_refuses_an_insert_the_minted_contig_cannot_hold():
    """A truncated insert would make the whole n1 arm shorter than the decoys it claims."""
    items, minted, hosts, pool = _build_inputs()
    items[0]["splice_phase"] = _HOST_NT - 4
    with pytest.raises(nf.NoiseFloorError, match="is truncated"):
        nf.build_nulls(items, minted, hosts, pool)


def test_build_nulls_names_the_pool_arm_in_order_and_keeps_each_host_id():
    items, minted, hosts, pool = _build_inputs()
    records, built = nf.build_nulls(items, minted, hosts, pool)
    assert len(records["n3_pool"]) == len(pool)
    for index, (host_id, sequence) in enumerate(pool):
        name = f"p{index:04d}"
        assert records["n3_pool"][name] == sequence
        assert built["idmap"]["n3_pool"][name]["host_id"] == host_id


# ══════════════════════════════════════════════════════════════════════════════════════
# choose_pool_windows
# ══════════════════════════════════════════════════════════════════════════════════════
def _pool_inputs(n_admitted: int = 1200) -> tuple[list[str], set[str], dict[str, str]]:
    """``n_admitted`` admitted ids in DELIBERATELY unsorted order, plus 3 drawn ones."""
    rows = {f"cand:{index:05d}": _random_dna(40, 500 + index) for index in range(n_admitted + 3)}
    admitted = list(rows)
    random.Random(7).shuffle(admitted)
    drawn = {f"cand:{index:05d}" for index in range(n_admitted, n_admitted + 3)}
    return admitted, drawn, rows


def test_choose_pool_windows_returns_n_windows_sorted_from_unsorted_admitted_input():
    """A producer that sorts must be handed UNSORTED input, or the sort is untested.

    ``_pool_inputs`` shuffles the admitted list, so a ``choose_pool_windows`` that dropped its
    ``sorted`` would return the shuffled order and this assert would bite
    ([[sorted-fixture-hides-the-sort]]).
    """
    admitted, drawn, rows = _pool_inputs()
    assert admitted != sorted(admitted), "the fixture must be unsorted or it tests nothing"
    chosen = nf.choose_pool_windows(admitted, drawn, rows, n=50)
    assert len(chosen) == 50
    names = [host_id for host_id, _ in chosen]
    assert names == sorted(names)
    # The sequence travels with its id, asserted as an identity rather than as a count.
    for host_id, sequence in chosen:
        assert sequence == rows[host_id]


def test_choose_pool_windows_does_not_inherit_the_admitted_files_order():
    """Two different admitted orderings must draw the same windows, or the draw is not seeded."""
    admitted, drawn, rows = _pool_inputs()
    first = nf.choose_pool_windows(admitted, drawn, rows, n=50)
    reshuffled = list(admitted)
    random.Random(99).shuffle(reshuffled)
    assert reshuffled != admitted
    assert nf.choose_pool_windows(reshuffled, drawn, rows, n=50) == first
    # And the same call twice is the same draw.
    assert nf.choose_pool_windows(admitted, drawn, rows, n=50) == first


def test_choose_pool_windows_never_returns_a_drawn_host():
    """A null sharing windows with the benchmark measures that benchmark, not its floor."""
    admitted, drawn, rows = _pool_inputs()
    chosen = nf.choose_pool_windows(admitted, drawn, rows, n=300)
    assert drawn and not ({host_id for host_id, _ in chosen} & drawn)


def test_choose_pool_windows_refuses_an_admitted_id_absent_from_the_pool():
    """A silently skipped id would shrink the eligible set without saying so."""
    admitted, drawn, rows = _pool_inputs()
    admitted.append("cand:ghost")
    with pytest.raises(nf.NoiseFloorError, match="absent from the candidate pool"):
        nf.choose_pool_windows(admitted, drawn, rows, n=50)
    # Positive control: without the ghost id the identical call succeeds.
    admitted.remove("cand:ghost")
    assert len(nf.choose_pool_windows(admitted, drawn, rows, n=50)) == 50


def test_choose_pool_windows_refuses_fewer_eligible_windows_than_requested():
    """Silently truncating the host arm would shrink the denominator the rate is read against."""
    admitted, drawn, rows = _pool_inputs(n_admitted=49)
    with pytest.raises(nf.NoiseFloorError, match="admitted-and-undrawn windows; need 50"):
        nf.choose_pool_windows(admitted, drawn, rows, n=50)
    # Positive control, one window either side of the boundary: 49 eligible satisfies n=49.
    assert len(nf.choose_pool_windows(admitted, drawn, rows, n=49)) == 49


def test_choose_pool_windows_counts_only_undrawn_windows_towards_the_requested_size():
    """The 3 drawn ids are present in ``rows`` and in ``admitted``; they must not make up the n."""
    admitted, drawn, rows = _pool_inputs(n_admitted=50)
    assert len(admitted) == 53
    with pytest.raises(nf.NoiseFloorError, match="only 50 admitted-and-undrawn"):
        nf.choose_pool_windows(admitted, drawn, rows, n=51)


# ══════════════════════════════════════════════════════════════════════════════════════
# matchedness — nine controls, each breakable alone
# ══════════════════════════════════════════════════════════════════════════════════════
def test_matchedness_fires_every_control_on_a_correctly_built_null_set():
    """The positive control for all nine sabotages below.

    Without it, a ``matchedness`` that returned ``fired: False`` for everything would satisfy
    every one of those sabotages ([[raises-test-needs-a-positive-control]]).
    """
    checks = _checks(_hand_built())
    assert tuple(checks) == _CONTROL_NAMES
    failed = [name for name, row in checks.items() if not row["fired"]]
    assert failed == []


def test_matchedness_derives_fired_from_expected_versus_observed():
    """A ``fired`` written independently of its own measurement is a report field, not a control."""
    for row in _checks(_hand_built()).values():
        assert row["fired"] == (row["observed"] == row["expected"]), row["name"]
        assert "mechanism" in row


def test_matchedness_counts_both_n2_arms_so_a_dropped_arm_cannot_pass():
    """``expected`` is ``n * 2``; an implementation looping over one arm would report half."""
    checks = _checks(_hand_built())
    assert checks["n2_length_matches_the_minted_contig"]["expected"] == len(_ITEMS) * 2 == 6
    assert checks["n3_is_the_requested_size"]["expected"] == nf.N_POOL_WINDOWS


def test_n2_matchedness_is_agreement_outside_both_windows_not_excise_and_compare():
    """The trap: the mint is an equal-length OVERWRITE, so excise-and-compare is the wrong test.

    Excising ``[phase, phase+L)`` from the contig leaves ``host`` minus one window; excising
    ``[moved, moved+L)`` from the scramble leaves ``host`` minus a *different* window. The two
    disagree on every item, so a ``matchedness`` rewritten that way would report 0 of 6 and the
    control would read as "no power" rather than as broken
    ([[control-matchedness-must-be-asserted]]).
    """
    fixture = _hand_built()
    row = _checks(fixture)["n2_agrees_outside_both_insert_windows"]
    assert row["observed"] == row["expected"] == 6 and row["fired"]

    agreed = 0
    for arm, _prefix, _phases in _ARM_PREFIX:
        for name, meta in fixture["idmap"][arm].items():
            contig = fixture["minted"][meta["contig_id"]]
            scramble = fixture["records"][arm][name]
            phase, moved, length = (
                meta["minted_phase"],
                meta["scramble_phase"],
                meta["insert_nt"],
            )
            excised_contig = contig[:phase] + contig[phase + length :]
            excised_scramble = scramble[:moved] + scramble[moved + length :]
            agreed += excised_contig == excised_scramble
    assert agreed == 0, "the wrong test must fail on every item, or this trap is not pinned"


def _break_n1_order(fixture: dict) -> None:
    """Reverse the n1 idmap, leaving its size intact: a count-only clause would not see this."""
    fixture["idmap"]["n1_inserts"] = dict(reversed(list(fixture["idmap"]["n1_inserts"].items())))


def _break_n2_length(fixture: dict) -> None:
    """Append 5 nt to one scramble. Appended at the 3′ end, so every original position — and so
    the insert, its single copy, and the agreement outside both windows — is untouched."""
    fixture["records"]["n2_scramble1"]["s1_0000"] += "GATCA"


def _break_insert_once(fixture: dict) -> None:
    """Plant a SECOND copy of the insert at the MINTED phase — inside the union of the two
    windows, so the outside-agreement control stays satisfied and only the copy count moves."""
    sequence = fixture["records"]["n2_scramble1"]["s1_0000"]
    phase = _ITEMS[0]["splice_phase"]
    fixture["records"]["n2_scramble1"]["s1_0000"] = (
        sequence[:phase] + _INSERTS[0] + sequence[phase + _INSERT_NT :]
    )


def _break_insert_identity(fixture: dict) -> None:
    """Re-splice a DIFFERENT decoy at the same phase: the arm is then not matched to its contig."""
    sequence = fixture["records"]["n2_scramble1"]["s1_0000"]
    moved = _SCRAMBLE1[0]
    fixture["records"]["n2_scramble1"]["s1_0000"] = (
        sequence[:moved] + _INSERTS[1] + sequence[moved + _INSERT_NT :]
    )


def _break_outside_agreement(fixture: dict) -> None:
    """Flip one base at position 390 — outside {10..33} and {200..223}, so only this control."""
    sequence = fixture["records"]["n2_scramble1"]["s1_0000"]
    replacement = "A" if sequence[390] != "A" else "C"
    fixture["records"]["n2_scramble1"]["s1_0000"] = sequence[:390] + replacement + sequence[391:]


def _break_no_minted_reproduction(fixture: dict) -> None:
    """Re-mint item 1 with item 0's insert at item 0's SCRAMBLE phase, so ``s1_0000`` reproduces
    ``minted['dec:001']`` exactly. Items 0 and 1 share host H0, which is what makes this
    constructible; every other control stays satisfied because item 1's own scrambles are rebuilt
    consistently around its new phase."""
    host = fixture["hosts"]["cand:H0"]
    fixture["items"][1]["splice_phase"] = _SCRAMBLE1[0]
    fixture["minted"]["dec:001"] = nf.splice(host, _INSERTS[0], _SCRAMBLE1[0])
    fixture["records"]["n1_inserts"]["i0001"] = _INSERTS[0]
    for arm, prefix, phases in _ARM_PREFIX:
        fixture["records"][arm][f"{prefix}0001"] = nf.splice(host, _INSERTS[0], phases[1])
        fixture["idmap"][arm][f"{prefix}0001"]["minted_phase"] = _SCRAMBLE1[0]


def _break_phase_shift(fixture: dict) -> None:
    """Move one junction 30 nt — a real re-splice, consistently recorded, just not far enough."""
    moved = _ITEMS[0]["splice_phase"] + 30
    fixture["records"]["n2_scramble1"]["s1_0000"] = nf.splice(
        fixture["hosts"]["cand:H0"], _INSERTS[0], moved
    )
    fixture["idmap"]["n2_scramble1"]["s1_0000"]["scramble_phase"] = moved


def _break_n3_disjointness(fixture: dict) -> None:
    fixture["drawn"].add(fixture["idmap"]["n3_pool"]["p0000"]["host_id"])


def _break_n3_size(fixture: dict) -> None:
    """Drop one record but not its idmap entry, so the disjointness control is unaffected."""
    fixture["records"]["n3_pool"].pop("p0000")


@pytest.mark.parametrize(
    "mutate,control",
    [
        (_break_n1_order, "n1_covers_the_gated_negatives"),
        (_break_n2_length, "n2_length_matches_the_minted_contig"),
        (_break_insert_once, "n2_carries_its_insert_exactly_once"),
        (_break_insert_identity, "n2_insert_is_the_minted_insert"),
        (_break_outside_agreement, "n2_agrees_outside_both_insert_windows"),
        (_break_no_minted_reproduction, "n2_reproduces_no_minted_contig"),
        (_break_phase_shift, "n2_phase_moved_far_enough"),
        (_break_n3_disjointness, "n3_is_disjoint_from_the_drawn_hosts"),
        (_break_n3_size, "n3_is_the_requested_size"),
    ],
)
def test_each_matchedness_control_is_breakable_alone(mutate, control):
    """Exactly one control must fall, or the nine are coupled and none is individually tested."""
    fixture = _hand_built()
    assert [name for name, row in _checks(fixture).items() if not row["fired"]] == []
    mutate(fixture)
    checks = _checks(fixture)
    failed = [name for name, row in checks.items() if not row["fired"]]
    assert failed == [control], (control, failed)
    assert checks[control]["observed"] != checks[control]["expected"]


def test_matchedness_reports_the_minimum_observed_shift_in_its_note():
    """A note that did not move with the measurement would certify the pin it describes."""
    fixture = _hand_built()
    assert "80 nt" in _checks(fixture)["n2_phase_moved_far_enough"]["note"]
    _break_phase_shift(fixture)
    assert "30 nt" in _checks(fixture)["n2_phase_moved_far_enough"]["note"]


def test_matchedness_on_a_build_nulls_output_fires_every_control():
    """The end-to-end shape: the bytes ``build_nulls`` writes satisfy the nine controls."""
    items, minted, hosts, pool = _build_inputs()
    records, built = nf.build_nulls(items, minted, hosts, pool)
    rows = nf.matchedness(items, minted, records, built["idmap"], {"cand:H0", "cand:H1"})
    assert [row["name"] for row in rows if not row["fired"]] == []


def test_an_empty_corpus_is_refused_rather_than_measured():
    """Eight of the nine controls are counting equalities, so on zero items every one reads
    ``0 == 0`` and the whole block certifies a null set that was never built
    ([[clauses-must-guard-emptiness]]).

    The positive control is the non-empty build below, which must NOT raise — a guard that
    refused everything would satisfy this test too ([[raises-test-needs-a-positive-control]]).
    """
    records, built = nf.build_nulls(
        [], {}, {}, [(f"pool:{name}", seq) for name, seq in _pool_records()]
    )
    assert built["min_phase_shift_nt"] is None
    with pytest.raises(nf.NoiseFloorError, match="would read 0 == 0"):
        nf.matchedness([], {}, records, built["idmap"], set())
    # Positive control: the same call on a real corpus returns nine fired controls.
    items, minted, hosts, pool = _build_inputs()
    good_records, good_built = nf.build_nulls(items, minted, hosts, pool)
    rows = nf.matchedness(items, minted, good_records, good_built["idmap"], set())
    assert [row["name"] for row in rows] == list(nf.MATCHEDNESS_CONTROLS)
    assert all(row["fired"] for row in rows)


def test_an_empty_null_arm_is_refused_even_when_the_corpus_is_not():
    """An arm whose FASTA came back empty would make its own controls read 0 == 0."""
    items, minted, hosts, pool = _build_inputs()
    records, built = nf.build_nulls(items, minted, hosts, pool)
    emptied = {**records, "n3_pool": {}}
    with pytest.raises(nf.NoiseFloorError, match="carry no sequences"):
        nf.matchedness(items, minted, emptied, built["idmap"], set())


# ══════════════════════════════════════════════════════════════════════════════════════
# best_by_target — the bit column, maximised per target
# ══════════════════════════════════════════════════════════════════════════════════════
def _row(target: str, score: float, *, evalue: float = 1.0e-10, inc: str = "!") -> str:
    """One real-shaped ``cmsearch --tblout --fmt 1`` data row: score is field 14, E-value 15,
    ``inc`` 16 — the layout the committed ``tests/fixtures/host_screen/*.tblout`` carry."""
    return f"{target} - T-box RF00230 cm 1 224 1 400 + no 1 0.40 0.0 " f"{score} {evalue} {inc} -"


def _tblout(*rows: str) -> str:
    header = "#target name accession query name accession mdl mdl from mdl to\n"
    return header + "\n".join(rows) + "\n"


@pytest.mark.parametrize("order", ["descending", "ascending"], ids=["worse-last", "worse-first"])
def test_best_by_target_takes_the_maximum_whatever_order_the_rows_arrive_in(order):
    """A ``last wins`` or ``first wins`` implementation gets one of these two orders wrong."""
    rows = [_row("s1_0000", 30.0), _row("s1_0000", 5.0)]
    text = _tblout(*(rows if order == "descending" else list(reversed(rows))))
    assert nf.best_by_target(text) == {"s1_0000": 30.0}


def test_best_by_target_reads_the_bit_column_and_not_the_evalue():
    """The best bit score here carries the WORST E-value, so an E-value-ranking implementation
    returns 5.0 (or 1e-30) instead of 30.0 — a named wrong answer, not just "something else"."""
    text = _tblout(
        _row("s1_0000", 5.0, evalue=1.0e-30),
        _row("s1_0000", 30.0, evalue=1.0),
    )
    assert nf.best_by_target(text) == {"s1_0000": 30.0}


def test_best_by_target_ignores_the_inc_flag_which_is_an_evalue_derived_call():
    """A15 Pin 2's rule. Here the only ``!`` row is the weak one, so an ``inc``-filtering
    implementation would report 5.0 and a floor 6 bits lower than the measured one."""
    text = _tblout(
        _row("s1_0000", 5.0, inc="!"),
        _row("s1_0000", 30.0, inc="?"),
    )
    assert nf.best_by_target(text) == {"s1_0000": 30.0}


def test_best_by_target_keys_each_target_separately_and_is_empty_without_rows():
    text = _tblout(_row("s1_0000", 12.0), _row("s1_0001", 7.5), _row("s1_0000", 9.0))
    assert nf.best_by_target(text) == {"s1_0000": 12.0, "s1_0001": 7.5}
    assert nf.best_by_target("") == {}
    assert nf.best_by_target("# a header and nothing else\n") == {}


# ══════════════════════════════════════════════════════════════════════════════════════
# cell_stats — the padding rule
# ══════════════════════════════════════════════════════════════════════════════════════
#: Twelve reported bit scores, used with two different denominators below.
_BEST = {
    f"s1_{index:04d}": score
    for index, score in enumerate(
        [30.0, 25.0, 20.0, 15.0, 12.0, 11.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    )
}
_GE_KEY = f"n_ge_{nf.RATE_AT_BITS:g}"
_RATE_KEY = f"rate_ge_{nf.RATE_AT_BITS:g}"
_SCORES_KEY = f"scores_ge_{nf.RATE_AT_BITS:g}"


def test_cell_stats_pads_unreported_sequences_at_zero_and_keeps_them_in_the_denominator():
    """20 sequences searched, 12 reported: the 8 silent ones are measurements of "below the floor".

    By hand, with padding: padded = [30,25,20,15,12,11,9,8,7,6,5,4] + [0]*8.
      n_ge_10   = 6   (30, 25, 20, 15, 12, 11)
      rate      = 6 / 20                                     = 0.3
      p99       = padded[int(0.01*20)] = padded[0]            = 30.0
      p95       = padded[int(0.05*20)] = padded[1]            = 25.0
      median    = mean of the 10th and 11th smallest of the 20 values. The 8 zeros take ranks
                  1-8, then 4.0 is rank 9, 5.0 rank 10, 6.0 rank 11, so the median is
                                                             (5.0 + 6.0) / 2 = 5.5
    """
    cell = nf.cell_stats(_BEST, 20)
    assert cell["n"] == 20
    assert cell["n_with_hit"] == 12
    assert cell[_GE_KEY] == 6
    assert cell[_RATE_KEY] == 0.3
    assert cell["max_bits"] == 30.0
    assert cell["p99_bits"] == 30.0
    assert cell["p95_bits"] == 25.0
    assert cell["median_bits"] == 5.5
    assert cell[_SCORES_KEY] == [30.0, 25.0, 20.0, 15.0, 12.0, 11.0]


def test_cell_stats_without_padding_gives_a_different_median_and_rate():
    """The same hits over 12 sequences instead of 20, so the padding rule is not a no-op.

    By hand, no padding: sorted ascending [4,5,6,7,8,9,11,12,15,20,25,30].
      rate   = 6 / 12                                  = 0.5
      median = (6th + 7th smallest) / 2 = (9.0 + 11.0) / 2 = 10.0
      p95    = padded[int(0.05*12)] = padded[0]            = 30.0
    A conditional-on-hitting implementation would report these numbers for the 20-sequence cell
    too, i.e. a floor nearly twice as high as the measured one.
    """
    cell = nf.cell_stats(_BEST, 12)
    assert cell[_RATE_KEY] == 0.5
    assert cell["median_bits"] == 10.0
    assert cell["p95_bits"] == 30.0


def test_cell_stats_of_a_silent_cell_is_zeros_and_not_a_missing_distribution():
    """Every sequence below the reporting threshold is still a measured cell, with ``n`` intact."""
    cell = nf.cell_stats({}, 7)
    assert cell["n"] == 7 and cell["n_with_hit"] == 0
    assert cell[_GE_KEY] == 0 and cell[_RATE_KEY] == 0.0
    assert cell["max_bits"] is None
    assert cell["median_bits"] == 0.0 and cell["p95_bits"] == 0.0


def test_cell_stats_counts_a_score_exactly_at_the_reporting_cut_as_at_or_above():
    """``>=``, not ``>``: TBDB001's real T-boxes start at exactly 10.0 bits."""
    cell = nf.cell_stats({"a": nf.RATE_AT_BITS, "b": nf.RATE_AT_BITS - 0.1}, 2)
    assert cell[_GE_KEY] == 1 and cell[_SCORES_KEY] == [10.0]


@pytest.mark.parametrize("n", [0, -1])
def test_cell_stats_refuses_a_cell_with_no_sequences(n):
    """A zero denominator would make ``rate_ge_10`` a ZeroDivisionError or, worse, 0/0 prose."""
    with pytest.raises(nf.NoiseFloorError, match="no sequences"):
        nf.cell_stats({"a": 12.0}, n)
    # Positive control: n = 1 with the same hit is accepted.
    assert nf.cell_stats({"a": 12.0}, 1)[_RATE_KEY] == 1.0


def test_cell_stats_refuses_more_reported_targets_than_sequences_searched():
    """More targets than sequences means the tblout is not the FASTA's: the pairing is broken."""
    with pytest.raises(nf.NoiseFloorError, match="targets reported from"):
        nf.cell_stats({"a": 12.0, "b": 11.0, "c": 10.0}, 2)
    # Positive control, exactly at the boundary: 3 targets from 3 sequences is legal.
    assert nf.cell_stats({"a": 12.0, "b": 11.0, "c": 10.0}, 3)["n_with_hit"] == 3


# ══════════════════════════════════════════════════════════════════════════════════════
# null_maxima
# ══════════════════════════════════════════════════════════════════════════════════════
def _cells(**overrides: float | None) -> dict:
    """Two models × four arms × two pipelines, every ``max_bits`` 5.0 unless overridden by a
    ``<model>__<arm>__<pipeline>`` keyword."""
    cells: dict[str, dict[str, dict[str, dict]]] = {}
    for model, _cm in hs.SCREEN_CMS:
        for arm in nf.ARMS:
            for pipeline, _flag in nf.PIPELINES:
                value = overrides.get(f"{model}__{arm}__{pipeline}", 5.0)
                cells.setdefault(model, {}).setdefault(arm, {})[pipeline] = {"max_bits": value}
    return cells


@pytest.mark.parametrize(
    "cell",
    [
        "RF00230__n1_inserts__default",
        "RF00230__n1_inserts__max",
        "RF00230__n2_scramble1__max",
        "RF00230__n2_scramble2__default",
        "RF00230__n3_pool__max",
    ],
)
def test_null_maxima_reaches_into_every_arm_and_every_pipeline(cell):
    """The unique maximum is placed in a different cell each time: an implementation that read
    only one arm, or only the ``default`` pipeline, would miss it in at least one of these."""
    maxima = nf.null_maxima(_cells(**{cell: 16.8}))
    assert maxima["RF00230"] == 16.8
    assert maxima["TBDB001"] == 5.0


def test_null_maxima_is_per_model_and_does_not_pool_the_two_score_scales():
    """RF00230 and TBDB001 differ ~5× in scale; one shared maximum would re-pin the pair."""
    maxima = nf.null_maxima(_cells(RF00230__n2_scramble1__max=16.8, TBDB001__n3_pool__default=9.9))
    assert maxima == {"RF00230": 16.8, "TBDB001": 9.9}


def test_null_maxima_refuses_a_model_whose_every_cell_is_empty():
    """Otherwise a model that silently searched nothing reports the most favourable floor there
    is — no measured maximum at all — and a threshold could be signed against it."""
    cells = _cells()
    for by_pipeline in cells["TBDB001"].values():
        for cell in by_pipeline.values():
            cell["max_bits"] = None
    with pytest.raises(nf.NoiseFloorError, match="every cell is empty"):
        nf.null_maxima(cells)
    # Positive control: one non-empty cell anywhere in that model is enough to measure a floor.
    cells["TBDB001"]["n3_pool"]["max"]["max_bits"] = 9.9
    assert nf.null_maxima(cells)["TBDB001"] == 9.9


def test_null_maxima_refuses_a_model_with_no_arms_at_all():
    """An unsearched model is an absence, not a floor of zero."""
    cells = _cells()
    cells["TBDB001"] = {}
    with pytest.raises(nf.NoiseFloorError, match="every cell is empty"):
        nf.null_maxima(cells)


def test_null_maxima_ignores_empty_cells_beside_non_empty_ones():
    cells = _cells(RF00230__n1_inserts__default=None, RF00230__n2_scramble1__max=16.8)
    assert nf.null_maxima(cells)["RF00230"] == 16.8


# ══════════════════════════════════════════════════════════════════════════════════════
# motif_counts
# ══════════════════════════════════════════════════════════════════════════════════════
def _motif_records(**planted: str) -> dict[str, dict[str, str]]:
    records = {
        arm: {f"{arm}_0": _random_dna(120, 300 + index)} for index, arm in enumerate(nf.ARMS)
    }
    for arm, sequence in planted.items():
        records[arm][f"{arm}_planted"] = sequence
    return records


def test_motif_counts_fires_both_strand_positive_controls_with_derived_fired():
    """``n_motif: 0`` is also what a scan returns when it never looked, so the measured zero is
    only publishable beside a detection that must be non-zero
    ([[namespace-mismatch-invisible-noop]]).
    """
    block = nf.motif_counts(_motif_records())
    controls = {entry["name"]: entry for entry in block["positive_controls"]}
    assert set(controls) == set(nf.MOTIF_CONTROL_NAMES)
    for entry in controls.values():
        assert entry["fired"] is True
        assert entry["fired"] == (entry["observed"] == entry["expected"])
    assert block["motif"] == hs.MOTIF
    # The asymmetry is the whole control. `hs.motif_present` is itself a both-strand scan, so
    # `motif_present(x) == motif_present(revcomp(x))` for EVERY x — a "reverse" control that
    # only reverse-complements its subject is the forward control twice and fires identically
    # with the reverse leg dead. The reverse control must therefore expect the both-strand
    # verdict and a forward-only scan to DISAGREE.
    assert controls[nf.MOTIF_CONTROL_NAMES[0]]["expected"] == [True, True]
    assert controls[nf.MOTIF_CONTROL_NAMES[1]]["expected"] == [True, False]
    forward = nf.MOTIF_CONTROL_FORWARD
    assert hs.motif_present(forward) == hs.motif_present(
        wd.reverse_complement(forward)
    ), "motif_present is both-strand, which is why a revcomp-only control tests nothing"


def test_motif_counts_records_a_denominator_beside_every_count():
    """A bare zero cannot be told from a scan that ran over nothing."""
    block = nf.motif_counts(_motif_records())
    assert set(block["per_arm"]) == set(nf.ARMS)
    assert block["n"] == sum(arm["n"] for arm in block["per_arm"].values()) == 4
    assert block["n_motif"] == 0
    assert block["rate"] == 0.0


def test_motif_counts_finds_a_motif_planted_on_the_reverse_strand():
    """The rule is a both-strand rule: a mined window carries no orientation.

    The planted sequence is the reverse complement of the forward control, and does NOT match on
    the forward strand — asserted here, so a forward-only scan fails this test rather than
    passing it on a sequence that happened to match both ways.
    """
    reverse_only = wd.reverse_complement(nf.MOTIF_CONTROL_FORWARD)
    assert not hs.MOTIF_RE.search(reverse_only)
    block = nf.motif_counts(_motif_records(n3_pool=reverse_only))
    # Attribution, not just a total: a scan that found it in the wrong arm would also total 1.
    assert block["per_arm"]["n3_pool"]["n_motif"] == 1
    assert [arm for arm, cell in block["per_arm"].items() if cell["n_motif"]] == ["n3_pool"]
    assert block["n_motif"] == 1 and block["n"] == 5
    assert block["rate"] == 1 / 5


def test_motif_counts_of_an_arm_with_no_sequences_reports_a_zero_denominator():
    """It must be visible as 0 of 0 rather than absent — ``validate_report`` refuses it below."""
    records = _motif_records()
    records["n1_inserts"] = {}
    block = nf.motif_counts(records)
    assert block["per_arm"]["n1_inserts"] == {"n": 0, "n_motif": 0}
    assert block["n"] == 3
    # And with nothing at all there is no rate to report, rather than a 0.0 that reads as a
    # measurement.
    assert nf.motif_counts({arm: {} for arm in nf.ARMS})["rate"] is None


# ══════════════════════════════════════════════════════════════════════════════════════
# wilson_interval
# ══════════════════════════════════════════════════════════════════════════════════════
def test_wilson_interval_lower_bound_is_exactly_zero_at_k_zero():
    """A Wald interval runs below zero at these rates and would read as a measurement.

    By hand, with z = 1.959963985 and z² = 3.84145882249708, at p = 0:
      centre = (z²/2n) / (1 + z²/n)
      spread = z·sqrt(z²/4n²) / (1 + z²/n) = (z²/2n) / (1 + z²/n) = centre
    so the lower bound is centre - spread = 0 exactly, and
      upper  = 2·centre = (z²/n) / (1 + z²/n)
             = 3.84145882249708/1000 / 1.00384145882249708
             = 0.0038267584873443055
    """
    low, high = nf.wilson_interval(0, 1000)
    assert low == 0.0
    assert high == pytest.approx(0.0038267584873443055, rel=1e-12)


def test_wilson_interval_upper_bound_is_exactly_one_at_k_equals_n():
    """At p = 1 the spread equals z²/2n / (1 + z²/n) and centre + spread is (1+z²/n)/(1+z²/n)."""
    low, high = nf.wilson_interval(7, 7)
    assert high == 1.0
    assert 0.0 < low < 1.0


@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (1, 3073), (14, 3073), (3073, 3073)])
def test_wilson_interval_is_bounded_in_the_unit_interval_and_brackets_the_point_estimate(k, n):
    low, high = nf.wilson_interval(k, n)
    assert 0.0 <= low <= k / n <= high <= 1.0


def test_wilson_interval_refuses_a_zero_denominator():
    """An interval over no observations is not a wide interval, it is no interval."""
    with pytest.raises(nf.NoiseFloorError, match="needs a denominator"):
        nf.wilson_interval(0, 0)
    # Positive control: one observation is enough. By the k = 0 closed form above,
    #   upper = (z²/n) / (1 + z²/n) = 3.84145882249708 / 4.84145882249708 = 0.7934506856996815
    # and the lower bound is 0 exactly — a single miss bounds the rate at 79 %, not at 0.
    assert nf.wilson_interval(0, 1) == [0.0, pytest.approx(0.7934506856996815, rel=1e-12)]


def test_wilson_interval_narrows_as_the_denominator_grows():
    """A width that did not move with n would make the pooled rate's interval decorative."""
    narrow = nf.wilson_interval(10, 10_000)
    wide = nf.wilson_interval(1, 1_000)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# ══════════════════════════════════════════════════════════════════════════════════════
# validate_report — a self-consistent synthetic report, then each clause alone
# ══════════════════════════════════════════════════════════════════════════════════════
def _cell_block(n: int = 100, n_with_hit: int = 3, scores: tuple[float, ...] = (12.0,)) -> dict:
    at_or_above = [value for value in scores if value >= nf.RATE_AT_BITS]
    return {
        "n": n,
        "n_with_hit": n_with_hit,
        _GE_KEY: len(at_or_above),
        _RATE_KEY: len(at_or_above) / n,
        "max_bits": max(scores) if scores else None,
        # Bounded by the cell's own max_bits, which the validator now re-derives: a p95 above
        # the maximum is arithmetically impossible and used to validate clean.
        "p99_bits": max(scores) if scores else 0.0,
        "p95_bits": 0.0,
        "median_bits": 0.0,
        _SCORES_KEY: [round(value, 1) for value in at_or_above],
    }


def _report() -> dict:
    """A synthetic — and deliberately self-consistent — noise-floor report.

    The counts are invented, not measured: the real ones live in
    ``reports/p3/postmint_noise_floor.json``. The unique maximum sits in ONE cell
    (RF00230 / n2_scramble1 / max, 16.8 bits) so ``null_max_bits`` has a single, hand-known
    source, and ``test_the_synthetic_reports_floor_is_what_null_maxima_derives`` ties the literal
    below to the producer.
    """
    cells: dict[str, dict[str, dict[str, dict]]] = {}
    for model, _cm in hs.SCREEN_CMS:
        for arm in nf.ARMS:
            for pipeline, _flag in nf.PIPELINES:
                cells.setdefault(model, {}).setdefault(arm, {})[pipeline] = _cell_block()
    cells["RF00230"]["n2_scramble1"]["max"] = _cell_block(n_with_hit=5, scores=(16.8, 11.0))
    per_arm = {arm: {"n": 100, "n_motif": 0} for arm in nf.ARMS}
    return {
        "schema_version": nf.SCHEMA_VERSION,
        "step": nf.STEP,
        "kind": nf.KIND,
        "generated_by": nf.GENERATED_BY,
        "adr": nf.ADR,
        "seed": nf.SEED,
        "search_threshold_bits": nf.REPORT_THRESHOLD_BITS,
        "rate_at_bits": nf.RATE_AT_BITS,
        # All NINE, by name. A fixture carrying three of them enshrined the gap the validator
        # now closes: a post-hoc artifact that dropped the six inconvenient controls validated
        # clean ([[new-gate-clause-invalidates-old-reports]]).
        "matchedness": [
            hs.control_entry("n1_covers_the_gated_negatives", "synthetic", [3, True], [3, True]),
            hs.control_entry("n2_length_matches_the_minted_contig", "synthetic", 6, 6),
            hs.control_entry("n2_carries_its_insert_exactly_once", "synthetic", 6, 6),
            hs.control_entry("n2_insert_is_the_minted_insert", "synthetic", 6, 6),
            hs.control_entry("n2_agrees_outside_both_insert_windows", "synthetic", 6, 6),
            hs.control_entry("n2_reproduces_no_minted_contig", "synthetic", 0, 0),
            hs.control_entry("n2_phase_moved_far_enough", "synthetic", True, True),
            hs.control_entry("n3_is_disjoint_from_the_drawn_hosts", "synthetic", 0, 0),
            hs.control_entry(
                "n3_is_the_requested_size", "synthetic", nf.N_POOL_WINDOWS, nf.N_POOL_WINDOWS
            ),
        ],
        "motif": {
            "motif": hs.MOTIF,
            "per_arm": per_arm,
            "n": sum(block["n"] for block in per_arm.values()),
            "n_motif": 0,
            "rate": 0.0,
            # The reverse control's expected value is ASYMMETRIC — [True, False] — because
            # `motif_present` is itself a both-strand scan, so a control whose subject is merely
            # the reverse complement fires identically with the reverse leg dead.
            "positive_controls": [
                hs.control_entry(
                    nf.MOTIF_CONTROL_NAMES[0], "synthetic", [True, True], [True, True]
                ),
                hs.control_entry(
                    nf.MOTIF_CONTROL_NAMES[1], "synthetic", [True, False], [True, False]
                ),
            ],
        },
        "cells": cells,
        "min_phase_shift_nt": nf.MIN_PHASE_SHIFT,
        "n_pool_windows": nf.N_POOL_WINDOWS,
        "null_max_bits": {"RF00230": 16.8, "TBDB001": 12.0},
        # One `n` per ARM (4 x 100), not one per search: summing per search would report 800
        # where 400 were searched, and the validator now re-derives this from the cells.
        "n_null_sequences": {model: 4 * 100 for model, _cm in hs.SCREEN_CMS},
    }


def test_the_synthetic_noise_floor_report_validates_clean():
    """The positive control for every mutation below: a validator that refused everything would
    satisfy all of them ([[raises-test-needs-a-positive-control]])."""
    assert nf.validate_report(_report()) == []


def test_the_synthetic_reports_floor_is_what_null_maxima_derives():
    """Ties the hand-written ``null_max_bits`` literals to the producer, so the fixture cannot
    drift away from the function whose output it stands in for."""
    report = _report()
    assert (
        nf.null_maxima(report["cells"])
        == report["null_max_bits"]
        == {
            "RF00230": 16.8,
            "TBDB001": 12.0,
        }
    )


def _forge_zero_denominator_arm(report: dict) -> None:
    """Zero one motif arm AND fix both sums, so only the positivity clause can bite."""
    report["motif"]["per_arm"]["n1_inserts"] = {"n": 0, "n_motif": 0}
    report["motif"]["n"] = sum(block["n"] for block in report["motif"]["per_arm"].values())
    report["motif"]["n_motif"] = sum(
        block["n_motif"] for block in report["motif"]["per_arm"].values()
    )


def _forge_ge_above_n_with_hit(report: dict) -> None:
    """Raise ``n_ge`` past ``n_with_hit`` while keeping the rate and the score list consistent,
    so the three clauses that read ``n_ge`` do not all fire at once."""
    cell = report["cells"]["RF00230"]["n1_inserts"]["default"]
    cell[_GE_KEY] = cell["n_with_hit"] + 1
    cell[_RATE_KEY] = cell[_GE_KEY] / cell["n"]
    cell[_SCORES_KEY] = [12.0] * cell[_GE_KEY]


def _control(report: dict, name: str) -> dict:
    """The matchedness row called ``name``. Addressed by name, never by index: the controls list
    is a named set and an index silently re-points when a control is added."""
    return next(
        row for row in report["matchedness"] if isinstance(row, dict) and row["name"] == name
    )


def _forge_matchedness_fired_flag(report: dict) -> None:
    """``fired: True`` beside a measurement that disagrees — a report field, not a control."""
    _control(report, "n2_agrees_outside_both_insert_windows")["observed"] = 5
    _control(report, "n2_agrees_outside_both_insert_windows")["fired"] = True


def _forge_failed_matchedness_control(report: dict) -> None:
    """A control that honestly did not fire. Distinct from the forged flag above."""
    _control(report, "n2_agrees_outside_both_insert_windows")["observed"] = 5
    _control(report, "n2_agrees_outside_both_insert_windows")["fired"] = False


def _forge_motif_control_that_did_not_fire(report: dict) -> None:
    report["motif"]["positive_controls"][1]["observed"] = False
    report["motif"]["positive_controls"][1]["fired"] = False


def _forge_empty_model(report: dict) -> None:
    for by_pipeline in report["cells"]["TBDB001"].values():
        for cell in by_pipeline.values():
            cell["max_bits"] = None


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda r: r.__setitem__("schema_version", "9.9"), "schema_version is not"),
        (lambda r: r.__setitem__("kind", "host_screen"), "kind is not"),
        (lambda r: r.__setitem__("step", "P3-16"), "step is not"),
        (lambda r: r.__setitem__("adr", "ADR-0005 D7"), "adr does not name"),
        # A null searched at the threshold it calibrates reports nothing below it, so the
        # measured maximum becomes an artefact of the flag rather than a floor.
        (
            lambda r: r.__setitem__("search_threshold_bits", 10.0),
            "search_threshold_bits is not 0.0",
        ),
        (lambda r: r.__setitem__("matchedness", []), "matchedness is missing"),
        (lambda r: r.pop("matchedness"), "matchedness is missing"),
        (_forge_matchedness_fired_flag, "fired is not observed == expected"),
        (_forge_failed_matchedness_control, "n2_agrees_outside_both_insert_windows' failed"),
        (lambda r: r.pop("motif"), "motif is missing"),
        (lambda r: r["motif"].__setitem__("motif", "GGTGGAACCGCG"), "motif.motif is not"),
        (
            lambda r: r["motif"]["per_arm"].pop("n3_pool"),
            "does not cover exactly the four null arms",
        ),
        (lambda r: r["motif"].__setitem__("n", 399), "motif.n is not the sum of its arms"),
        (
            lambda r: r["motif"].__setitem__("n_motif", 1),
            "motif.n_motif is not the sum of its arms",
        ),
        (_forge_zero_denominator_arm, "0 of 0 is not a measurement"),
        (
            lambda r: r["motif"].__setitem__(
                "positive_controls", r["motif"]["positive_controls"][:1]
            ),
            "positive_controls is missing both strands",
        ),
        (_forge_motif_control_that_did_not_fire, "did not fire"),
        (lambda r: r.pop("cells"), "cells is missing"),
        (
            lambda r: r["cells"]["TBDB001"].pop("n3_pool"),
            "cells.TBDB001.n3_pool is missing — an unsearched null arm",
        ),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"].pop("max"),
            "cells.RF00230.n1_inserts.max is missing",
        ),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"].__setitem__("n", 0),
            "cells.RF00230.n1_inserts.default.n is not positive",
        ),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"].__setitem__("n_with_hit", 101),
            "n_with_hit is outside [0, n]",
        ),
        (_forge_ge_above_n_with_hit, f"{_GE_KEY} exceeds n_with_hit"),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"].__setitem__(_RATE_KEY, 0.5),
            "rate is not n_ge / n",
        ),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"][_SCORES_KEY].append(11.0),
            "lists 2 scores for 1 hits",
        ),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"].__setitem__(
                _SCORES_KEY, [9.0]
            ),
            "lists a score below 10 bits",
        ),
        (lambda r: r.pop("null_max_bits"), "null_max_bits is missing"),
        (
            lambda r: r["null_max_bits"].pop("TBDB001"),
            "null_max_bits does not cover exactly the searched models",
        ),
        (
            lambda r: r["null_max_bits"].__setitem__("RF00230", 9.9),
            "null_max_bits.RF00230 is 9.9, not the measured maximum 16.8",
        ),
        (_forge_empty_model, "null_max_bits.TBDB001 rests on no non-empty cell"),
        (lambda r: r.pop("n_null_sequences"), "n_null_sequences is missing"),
        (
            lambda r: r["n_null_sequences"].__setitem__("RF00230", 0),
            "n_null_sequences.RF00230 is not positive",
        ),
        # ── the clauses added after the adversarial audit ────────────────────────────────
        # A clause set that re-derives only the rows it is GIVEN cannot see a dropped row.
        (
            lambda r: r["matchedness"].remove(_control(r, "n2_reproduces_no_minted_contig")),
            "matchedness is missing 1 of its 9 controls",
        ),
        (
            lambda r: r["matchedness"].append(
                hs.control_entry("n1_covers_the_gated_negatives", "dup", 1, 1)
            ),
            "duplicated control name",
        ),
        # The construction values: an artifact produced at another seed or another reporting
        # cut is not the measurement the threshold was signed against.
        (lambda r: r.__setitem__("seed", 999), "seed is 999"),
        (lambda r: r.__setitem__("rate_at_bits", 99.0), "rate_at_bits is 99.0"),
        (lambda r: r.__setitem__("min_phase_shift_nt", 5), "min_phase_shift_nt is 5"),
        (lambda r: r.__setitem__("n_pool_windows", 7), "n_pool_windows is 7"),
        # A forward control twice reads as both-strand evidence and is not.
        (
            lambda r: r["motif"]["positive_controls"][1].__setitem__(
                "name", nf.MOTIF_CONTROL_NAMES[0]
            ),
            "does not carry",
        ),
        # An invented arm would feed null_maxima and so the published floor.
        (
            lambda r: r["cells"]["RF00230"].__setitem__(
                "n5_bogus", copy.deepcopy(r["cells"]["RF00230"]["n1_inserts"])
            ),
            "arms that are not null arms",
        ),
        # A quantile above the cell's own maximum is arithmetically impossible.
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"].__setitem__("p95_bits", 999.0),
            "exceeds the cell's own max_bits",
        ),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"].pop("median_bits"),
            "median_bits is missing",
        ),
        # A string is a Sequence, so the old guard skipped it instead of refusing it.
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"].__setitem__(
                _SCORES_KEY, "12.0"
            ),
            "is a string",
        ),
        # The denominator is quoted beside an extreme-value maximum that rises with it.
        (
            lambda r: r["n_null_sequences"].__setitem__("RF00230", 4 * 100 * 2),
            "not the 400 its own cells report",
        ),
        (
            lambda r: r["n_null_sequences"].pop("TBDB001"),
            "does not cover exactly the models",
        ),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["max"].__setitem__("n", 101),
            "one arm searched at two different n",
        ),
    ],
)
def test_each_noise_floor_clause_is_breakable_alone(mutate, expected):
    """One mutation, one named problem, every other field left consistent.

    An all-TRUE fixture cannot test ``all(clauses)``, and two coupled fields broken together
    cannot say which clause bit ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    report = _report()
    assert nf.validate_report(report) == []
    mutate(report)
    problems = nf.validate_report(report)
    assert any(expected in problem for problem in problems), (expected, problems)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (_forge_zero_denominator_arm, "0 of 0 is not a measurement"),
        (_forge_ge_above_n_with_hit, f"{_GE_KEY} exceeds n_with_hit"),
        (_forge_matchedness_fired_flag, "fired is not observed == expected"),
        (
            lambda r: r["cells"]["RF00230"]["n1_inserts"]["default"].__setitem__(_RATE_KEY, 0.5),
            "rate is not n_ge / n",
        ),
        (
            lambda r: r["null_max_bits"].__setitem__("RF00230", 9.9),
            "null_max_bits.RF00230 is 9.9, not the measured maximum 16.8",
        ),
    ],
)
def test_the_coupling_prone_mutations_raise_exactly_one_problem(mutate, expected):
    """These five forges touch fields several clauses read; each must still produce ONE problem,
    or the ``breakable alone`` table above is reporting a message it did not isolate."""
    report = _report()
    mutate(report)
    problems = nf.validate_report(report)
    assert len(problems) == 1, problems
    assert expected in problems[0]


def test_a_missing_cells_block_stops_before_the_floor_clauses_rather_than_passing_them():
    """``null_max_bits`` is re-derived FROM ``cells``; with no cells the re-derivation would be
    vacuously satisfiable, so the validator must return early ([[clauses-must-guard-emptiness]])."""
    report = _report()
    report.pop("cells")
    assert nf.validate_report(report) == ["cells is missing"]


def test_a_forged_floor_is_refused_even_when_it_is_conservative():
    """A threshold is signed against this number, so a floor QUOTED HIGHER than measured is as
    much a fabrication as one quoted lower — the clause is equality, not an inequality."""
    report = _report()
    report["null_max_bits"]["RF00230"] = 30.0
    problems = nf.validate_report(report)
    assert any("not the measured maximum 16.8" in problem for problem in problems)


def test_a_matchedness_entry_that_is_not_an_object_is_named_and_the_rest_still_graded():
    """The ``continue`` must skip the bad entry, not abandon the remaining controls.

    A validator that stopped at the first malformed row would leave every later control
    unasserted while still reporting a problem, which reads as "one small defect".
    """
    report = _report()
    report["matchedness"].insert(1, "not an object")
    _control(report, "n2_agrees_outside_both_insert_windows")["observed"] = 5
    _control(report, "n2_agrees_outside_both_insert_windows")["fired"] = False
    problems = nf.validate_report(report)
    assert any("a matchedness entry is not an object" in problem for problem in problems)
    assert any("n2_agrees_outside_both_insert_windows' failed" in problem for problem in problems)


def test_validate_report_never_raises_on_a_malformed_payload():
    """A producer that raised would write nothing, and an artifact nobody can read cannot be
    adjudicated. Every one of these must come back as problems."""
    for payload in ({}, {"cells": "nope"}, {"matchedness": "nope", "motif": 3, "cells": {}}):
        problems = nf.validate_report(payload)
        assert problems and all(isinstance(problem, str) for problem in problems)


def test_copying_a_report_does_not_change_its_verdict():
    """The validator must not depend on object identity anywhere in the clause set."""
    report = _report()
    assert nf.validate_report(copy.deepcopy(report)) == nf.validate_report(report) == []


# ══════════════════════════════════════════════════════════════════════════════════════
# _pooled — the rate pooled over all four arms
# ══════════════════════════════════════════════════════════════════════════════════════
def test_pooled_sums_every_arm_per_model_and_pipeline():
    """The pooled rate must span all four arms, or it reports one channel as the floor.

    By hand over ``_report()``'s cells, where every cell is n = 100 with one hit at or above
    10 bits except RF00230 / n2_scramble1 / max, which has two:
      RF00230 default : k = 1+1+1+1 = 4 over n = 400 -> 0.01
      RF00230 max     : k = 1+2+1+1 = 5 over n = 400 -> 0.0125
      TBDB001 both    : k = 4       over n = 400 -> 0.01
    """
    pooled = nf._pooled(_report()["cells"])
    assert set(pooled) == {"RF00230", "TBDB001"}
    assert pooled["RF00230"]["default"]["n"] == 400
    assert pooled["RF00230"]["default"][_GE_KEY] == 4
    assert pooled["RF00230"]["default"][_RATE_KEY] == 0.01
    assert pooled["RF00230"]["max"][_GE_KEY] == 5
    assert pooled["RF00230"]["max"][_RATE_KEY] == 0.0125
    assert pooled["TBDB001"]["max"][_GE_KEY] == 4
    # The pooled maximum is the worst cell in that pipeline, not the worst cell overall: 16.8
    # sits in the ``max`` pipeline only.
    assert pooled["RF00230"]["max"]["max_bits"] == 16.8
    assert pooled["RF00230"]["default"]["max_bits"] == 12.0
    # The interval is the one `wilson_interval` computes for that k and n — pinned above against
    # hand arithmetic, so this binds the pooled block to it rather than re-deriving it here.
    assert pooled["RF00230"]["default"]["wilson95"] == nf.wilson_interval(4, 400)


def test_pooled_skips_a_pipeline_with_no_sequences_rather_than_dividing_by_zero():
    """A pipeline nobody searched must be absent, not present with a 0/0 rate."""
    cells = _report()["cells"]
    for arm in nf.ARMS:
        cells["TBDB001"][arm]["max"]["n"] = 0
    pooled = nf._pooled(cells)
    assert "max" not in pooled["TBDB001"]
    assert "default" in pooled["TBDB001"]


def test_pooled_reports_no_maximum_for_an_empty_pipeline_instead_of_raising():
    """``_pooled`` is descriptive and tolerates an empty cell; ``null_maxima`` — which a
    threshold is signed against — refuses one. Pinned so the asymmetry stays deliberate."""
    cells = _report()["cells"]
    for arm in nf.ARMS:
        cells["TBDB001"][arm]["default"]["max_bits"] = None
    assert nf._pooled(cells)["TBDB001"]["default"]["max_bits"] is None


# ══════════════════════════════════════════════════════════════════════════════════════
# FASTA round trip
# ══════════════════════════════════════════════════════════════════════════════════════
def test_write_fasta_then_read_fasta_round_trips_every_record(tmp_path):
    records = {"i0000": _INSERTS[0], "s1_0000": _HOSTS["cand:H0"]}
    digest = nf.write_fasta(tmp_path / "arm.fa", records, width=80)
    assert len(digest) == 64
    assert nf.read_fasta(tmp_path / "arm.fa") == records


def test_write_fasta_emits_records_in_the_mappings_own_order(tmp_path):
    """Pinned as an observation, not an endorsement: ``write_fasta`` iterates ``records.items()``,
    so the file — and therefore the sha256 ``prep.json`` records — depends on insertion order.
    Its docstring says "in id order", which holds only because ``build_nulls`` emits zero-padded
    ids in index order (pinned separately above). A caller handing it an out-of-order mapping
    gets an out-of-order FASTA, which is what this asserts.
    """
    path = tmp_path / "unsorted.fa"
    nf.write_fasta(path, {"i0002": "ACGT", "i0000": "TTTT", "i0001": "GGGG"})
    names = [
        line[1:] for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">")
    ]
    assert names == ["i0002", "i0000", "i0001"]
    assert names != sorted(names)


def test_read_fasta_refuses_a_duplicate_id_rather_than_merging_it(tmp_path):
    """Two records sharing one id would overwrite each other and every SUMMED invariant — the
    arm's ``n``, the cell's denominator — would still hold
    ([[duplicate-key-merges-instead-of-colliding]]).
    """
    path = tmp_path / "dup.fa"
    path.write_text(">i0000\nACGT\n>i0000\nTTTT\n", encoding="utf-8")
    with pytest.raises(nf.NoiseFloorError, match="duplicate FASTA id"):
        nf.read_fasta(path)
    # Positive control: the same two sequences under distinct ids are read as two records.
    path.write_text(">i0000\nACGT\n>i0001\nTTTT\n", encoding="utf-8")
    assert nf.read_fasta(path) == {"i0000": "ACGT", "i0001": "TTTT"}


def test_read_fasta_refuses_a_file_with_no_records(tmp_path):
    """An empty arm searched clean is the fail-open direction: 0 hits over 0 sequences."""
    path = tmp_path / "empty.fa"
    path.write_text("# nothing here\n\n", encoding="utf-8")
    with pytest.raises(nf.NoiseFloorError, match="no FASTA records"):
        nf.read_fasta(path)


def test_read_fasta_keys_on_the_first_whitespace_token_as_cmsearch_does(tmp_path):
    """The tblout's target column carries that token, and nothing after it."""
    path = tmp_path / "desc.fa"
    path.write_text(">s1_0000 scramble of dec:000\nACGTACGT\n", encoding="utf-8")
    assert nf.read_fasta(path) == {"s1_0000": "ACGTACGT"}


# ══════════════════════════════════════════════════════════════════════════════════════
# The pinned construction values
# ══════════════════════════════════════════════════════════════════════════════════════
def test_the_pinned_construction_values_are_the_ones_the_module_documents():
    """A silent change to any of these re-signs ``POSTMINT_THRESHOLD_BITS`` against a different
    measurement than the one the amendment names."""
    assert nf.SEED == 42
    assert nf.MIN_PHASE_SHIFT == 50
    assert nf.N_POOL_WINDOWS == 1000
    assert nf.REPORT_THRESHOLD_BITS == 0.0
    assert nf.RATE_AT_BITS == 10.0
    assert nf.PIPELINES == (("default", False), ("max", True))
    assert nf.ARMS == ("n1_inserts", "n2_scramble1", "n2_scramble2", "n3_pool")
    assert nf.KIND == hs.KIND_NOISE_FLOOR


def test_the_report_kind_is_the_one_host_screen_signs_its_thresholds_against():
    """If these two drifted apart, ``host_screen`` would validate a threshold against an artifact
    this module never produced ([[gate-must-bind-to-upstream-evidence]])."""
    assert nf.KIND == "postmint_noise_floor"
    assert nf.DEFAULT_REPORT == "reports/p3/postmint_noise_floor.json"
    assert set(hs.POSTMINT_THRESHOLD_BITS) == {model for model, _cm in hs.SCREEN_CMS}


def test_the_motif_positive_control_really_carries_the_consensus():
    """A control sequence that did not match would make the motif arm's zero unfalsifiable."""
    assert hs.motif_present(nf.MOTIF_CONTROL_FORWARD)
    assert hs.MOTIF_RE.search(nf.MOTIF_CONTROL_FORWARD)
    assert len(nf.MOTIF_CONTROL_FORWARD) == len(hs.MOTIF) + 8


def test_statistics_median_of_the_padded_distribution_is_not_the_median_of_the_hits():
    """Named here because the whole padding argument rests on it: 5.5 vs 10.0 on one hit set."""
    scores = sorted(_BEST.values(), reverse=True)
    assert statistics.median(scores + [0.0] * 8) == 5.5
    assert statistics.median(scores) == 10.0
    assert math.isclose(len([value for value in scores if value >= 10.0]) / 20, 0.3)


# ══════════════════════════════════════════════════════════════════════════════════════
# The three CLI legs — prep / search / report
# ══════════════════════════════════════════════════════════════════════════════════════
# Driven end to end over a ``tmp_path`` tree, with no ``cmsearch`` and no real data. Exactly
# three things are stubbed, every one of them an external boundary and none of them a subject of
# an assertion below ([[in-process-no-ops-look-like-compliance]]):
#
#   ``hs.load_candidates``       the pandas/parquet mining-pool read
#   ``hs.infernal_version_line`` ``cmsearch -h``, which is not on PATH in the test env
#   ``infernal.run_cmsearch``    the binary itself
#
# :func:`_fake_cmsearch` is a *faithful* stand-in rather than a no-op in the one respect the
# search leg checks: it echoes the options it was HANDED into the tblout's
# ``# Option settings:`` trailer, in Infernal's own order rather than ``search_argv``'s, so
# ``hs._search_one``'s argv-versus-executed comparison does real work and the ``-T`` the leg
# records is the ``-T`` the stub was asked for. A stub printing a fixed trailer would make the
# ``-T 0.0`` assertion vacuous.
_WORK_DIR = "work"
_BENCHMARK = "inputs/benchmark_v2.json"
_SCREEN_DIR = "inputs/host_screen"
_ADMITTED = "inputs/host_screen_admitted.txt"
_MINING_POOL = "data/processed/negatives/mining_pool_v0.parquet"
_REPORT_OUT = "out/postmint_noise_floor.json"
_VERSION_LINE = "INFERNAL 1.1.5 (Sep 2023)"

#: The leader decoy's host. Label 0, so it counts as DRAWN, and it is put in the admitted list
#: on purpose: the n3 arm must still exclude it.
_LEADER_HOST = "cand:H2"

#: One planted bit score per cell that is not at the floor. The two per-model maxima sit in
#: DIFFERENT arms and DIFFERENT pipelines, so a leg that searched one arm, or one pipeline,
#: could not reproduce both ([[symmetric-count-fixture-blind-to-inversion]]).
_PLANTED_BITS: dict[tuple[str, str, str], float] = {
    ("RF00230", "n2_scramble1", "max"): 16.8,
    ("TBDB001", "n3_pool", "default"): 9.9,
}
_FLOOR_BITS = 5.0


def _fake_cmsearch(cm, fasta, tblout, **kwargs) -> list:
    """``cmsearch``, stood in for at the binary boundary and nowhere above it.

    The planted hit's target is read out of the FASTA it was handed, so a tblout can only ever
    name a sequence that was actually searched.
    """
    path = Path(tblout)
    model, pipeline, arm = path.parent.name, path.parent.parent.name, path.stem
    target = next(iter(nf.read_fasta(fasta)))
    score = _PLANTED_BITS.get((model, arm, pipeline), _FLOOR_BITS)
    options = ["cmsearch", "-Z", repr(float(kwargs["search_space_mb"])), "--tblout", str(path)]
    if kwargs.get("cut_ga"):
        options.append("--cut_ga")
    if kwargs.get("toponly"):
        options.append("--toponly")
    if kwargs.get("no_filters"):
        options.append("--max")
    options += ["--noali", "-T", repr(float(kwargs["score_threshold"]))]
    options += ["--cpu", str(int(kwargs["cpu"])), str(cm), str(fasta)]
    text = (
        "# cmsearch :: search CM(s) against a sequence database\n"
        f"# {_VERSION_LINE}\n"
        f"# Option settings:     {' '.join(options)}\n"
        f"{_row(target, score)}\n"
        "# [ok]\n"
    )
    path.write_text(text, encoding="utf-8")
    return hs.parse_tblout_hits(text)


def _write_minted(screen_dir: Path, surrogates: dict[str, str], minted: dict[str, str]) -> None:
    """The post-mint label-0 FASTA, under the surrogate names its id map keys on."""
    nf.write_fasta(
        screen_dir / hs.FASTA_POSTMINT_LABEL0,
        {name: minted[contig_id] for name, contig_id in surrogates.items()},
    )


@pytest.fixture
def legs(tmp_path, monkeypatch) -> dict:
    """A complete, consistent input tree for the three legs, rooted at ``tmp_path``.

    The corpus is the SAME three gated negatives, two hosts and :data:`noise_floor.N_POOL_WINDOWS`
    pool windows the pure-function tests above use, so the legs run over a corpus whose
    properties are already pinned. The benchmark additionally carries a label-1 item and a
    ``leader_decoy`` item, so both of ``gated_negatives``' exclusions are live, and the leader
    decoy's host is admitted-but-drawn so the n3 arm has something to exclude.

    ``rows`` is handed to the stubbed pool loader by reference, so a test can break one host
    and leave everything else consistent.
    """
    items, minted, hosts, pool = _build_inputs()
    rows = {**dict(pool), **hosts, _LEADER_HOST: _random_dna(_HOST_NT, 4)}
    monkeypatch.chdir(tmp_path)
    for relative in (hs.ENV_LOCK, hs.ENV_LOCK_DATA, *(cm for _model, cm in hs.SCREEN_CMS)):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(f"# {relative} stand-in\n", encoding="utf-8")
    benchmark = {
        "schema_version": "2.0",
        "items": [
            *copy.deepcopy(items),
            {"contig_id": "pos:000", "host_id": "cand:X9", "label": 1, "pool": "corpus"},
            {
                "contig_id": "dec:leader",
                "host_id": _LEADER_HOST,
                "splice_phase": 5,
                "locus_length": _INSERT_NT,
                "label": 0,
                "pool": "leader_decoy",
            },
        ],
    }
    (tmp_path / _BENCHMARK).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / _BENCHMARK).write_text(json.dumps(benchmark, indent=1), encoding="utf-8")
    screen_dir = tmp_path / _SCREEN_DIR
    screen_dir.mkdir(parents=True, exist_ok=True)
    surrogates = {
        hs.surrogate("w", index): str(item["contig_id"]) for index, item in enumerate(items)
    }
    (screen_dir / "postmint_idmap.json").write_text(
        json.dumps(
            {"contigs": {name: {"contig_id": cid} for name, cid in surrogates.items()}}, indent=1
        ),
        encoding="utf-8",
    )
    _write_minted(screen_dir, surrogates, minted)
    admitted = [host_id for host_id, _sequence in pool] + ["cand:H0", "cand:H1", _LEADER_HOST]
    random.Random(3).shuffle(admitted)
    (tmp_path / _ADMITTED).write_text("\n".join(admitted) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        hs,
        "load_candidates",
        lambda root, mining_pool: (
            [{"candidate_id": cid, "sequence": sequence} for cid, sequence in rows.items()],
            {"max_records": None},
        ),
    )
    monkeypatch.setattr(hs, "infernal_version_line", lambda: _VERSION_LINE)
    monkeypatch.setattr(infernal, "run_cmsearch", _fake_cmsearch)
    return {
        "root": tmp_path,
        "items": items,
        "minted": minted,
        "hosts": hosts,
        "rows": rows,
        "surrogates": surrogates,
        "screen_dir": screen_dir,
        "out_dir": tmp_path / _WORK_DIR,
        "tblout_dir": tmp_path / _WORK_DIR / "tblout",
        "report": tmp_path / _REPORT_OUT,
    }


def _prep_argv(*extra: str) -> list[str]:
    return [
        "prep",
        "--out-dir",
        _WORK_DIR,
        "--data-root",
        ".",
        "--benchmark",
        _BENCHMARK,
        "--screen-dir",
        _SCREEN_DIR,
        "--admitted",
        _ADMITTED,
        "--mining-pool",
        _MINING_POOL,
        *extra,
    ]


def _search_argv(*extra: str) -> list[str]:
    """No ``--tblout-dir``: ``main``'s default is load-bearing in every leg test below."""
    return ["search", "--out-dir", _WORK_DIR, "--data-root", ".", *extra]


def _report_argv(*extra: str) -> list[str]:
    return [
        "report",
        "--out-dir",
        _WORK_DIR,
        "--data-root",
        ".",
        "--benchmark",
        _BENCHMARK,
        "--screen-dir",
        _SCREEN_DIR,
        "--out",
        _REPORT_OUT,
        *extra,
    ]


def _written(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prep_and_search(legs: dict) -> None:
    """The two legs the report reads, both at their defaults."""
    assert nf.main(_prep_argv()) == 0
    assert nf.main(_search_argv()) == 0


def _free_position(item_index: int) -> int:
    """A host position outside the item's minted window AND outside both scramble windows.

    Derived from ``build_nulls``' own idmap rather than hardcoded: a seed change that moved a
    window over a hardcoded position would silently put the one-base break below INSIDE a
    window, where no matchedness control reads it and the break would test nothing
    ([[green-sabotage-may-never-have-applied]]).
    """
    items, minted, hosts, pool = _build_inputs()
    _records, built = nf.build_nulls(items, minted, hosts, pool)
    item = items[item_index]
    occupied = set(range(item["splice_phase"], item["splice_phase"] + item["locus_length"]))
    for arm, prefix, _phases in _ARM_PREFIX:
        meta = built["idmap"][arm][f"{prefix}{item_index:04d}"]
        occupied |= set(range(meta["scramble_phase"], meta["scramble_phase"] + meta["insert_nt"]))
    free = sorted(set(range(len(hosts[item["host_id"]]))) - occupied)
    assert free, "the three windows cover the whole host; a break has nowhere outside to go"
    return free[-1]


def _flip_one_base(sequence: str, position: int) -> str:
    replacement = "A" if sequence[position] != "A" else "C"
    return sequence[:position] + replacement + sequence[position + 1 :]


def _diverge_the_minted_contigs(legs: dict) -> None:
    """Flip ONE base of ONE minted contig, outside its minted window and both scramble windows.

    The report re-reads the minted contigs, so after this the FASTAs that were searched no
    longer agree with them outside both insert windows — and only that one control moves, every
    other one staying satisfied ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    contig_id = str(legs["items"][2]["contig_id"])
    minted = dict(legs["minted"])
    minted[contig_id] = _flip_one_base(minted[contig_id], _free_position(2))
    assert minted != legs["minted"], "the flip must change the bytes or it tests nothing"
    _write_minted(legs["screen_dir"], legs["surrogates"], minted)


def _restore_the_minted_contigs(legs: dict) -> None:
    _write_minted(legs["screen_dir"], legs["surrogates"], legs["minted"])


# ── prep ─────────────────────────────────────────────────────────────────────────────────
def test_prep_writes_the_four_null_fastas_and_fires_all_nine_controls(legs):
    """The positive control for every prep sabotage below: a leg that failed on everything, or
    wrote nothing, would satisfy all of them ([[raises-test-needs-a-positive-control]])."""
    assert nf.main(_prep_argv()) == 0
    out_dir = legs["out_dir"]
    prep = _written(out_dir / nf.PREP_JSON)
    assert [entry["name"] for entry in prep["matchedness"]] == list(nf.MATCHEDNESS_CONTROLS)
    assert [entry["name"] for entry in prep["matchedness"] if not entry["fired"]] == []
    for arm in nf.ARMS:
        fasta = out_dir / f"{arm}.fa"
        assert prep["arms"][arm]["sha256"] == provenance.sha256_file(fasta), arm
        assert prep["arms"][arm]["n"] == len(nf.read_fasta(fasta)), arm
    assert {arm: prep["arms"][arm]["n"] for arm in nf.ARMS} == {
        "n1_inserts": 3,
        "n2_scramble1": 3,
        "n2_scramble2": 3,
        "n3_pool": nf.N_POOL_WINDOWS,
    }
    # The leader decoy's host is label 0, so it counts as DRAWN and the host arm must exclude it
    # even though the admitted list carries it.
    assert _LEADER_HOST in (legs["root"] / _ADMITTED).read_text(encoding="utf-8").split()
    assert _LEADER_HOST not in {str(meta["host_id"]) for meta in prep["idmap"]["n3_pool"].values()}


def test_prep_computes_matchedness_before_it_writes_the_fastas_it_certifies(legs, monkeypatch):
    """A certification computed after the bytes it covers is a report, not a guard
    ([[guard-runs-after-what-it-guards]]).

    Both calls are wrapped PASS-THROUGH — nothing is replaced, the nine controls are the real
    ones — and what is asserted is the order they ran in.
    """
    order: list[str] = []
    real_matchedness, real_write_fasta = nf.matchedness, nf.write_fasta

    def spy_matchedness(*args, **kwargs):
        order.append("matchedness")
        return real_matchedness(*args, **kwargs)

    def spy_write_fasta(path, records, **kwargs):
        order.append(f"write:{Path(path).name}")
        return real_write_fasta(path, records, **kwargs)

    monkeypatch.setattr(nf, "matchedness", spy_matchedness)
    monkeypatch.setattr(nf, "write_fasta", spy_write_fasta)
    assert nf.main(_prep_argv()) == 0
    assert order == ["matchedness"] + [f"write:{arm}.fa" for arm in nf.ARMS]
    # Not vacuous: the controls that one call computed are the ones the artifact carries, and
    # the four files it then wrote are on disk.
    prep = _written(legs["out_dir"] / nf.PREP_JSON)
    assert [entry["name"] for entry in prep["matchedness"] if not entry["fired"]] == []
    assert sorted(path.name for path in legs["out_dir"].glob("*.fa")) == sorted(
        f"{arm}.fa" for arm in nf.ARMS
    )


def test_prep_writes_no_fasta_when_matchedness_refuses(legs):
    """A refusal must leave NO bytes behind: FASTAs written before the certification that covers
    them sit on disk uncertified, and the search leg would search them
    ([[guard-runs-after-what-it-guards]]).

    ``matchedness``' own refusals — an empty corpus, an empty arm — are unreachable through this
    leg, because ``gated_negatives`` and ``choose_pool_windows`` refuse first, so the refusal is
    raised by a pass-through wrapper standing in for them. The return code is not the subject;
    the absence of the four FASTAs and of ``prep.json`` is.
    """
    out_dir = legs["out_dir"]
    real_matchedness = nf.matchedness

    def refuse(*args, **kwargs):
        real_matchedness(*args, **kwargs)
        raise nf.NoiseFloorError("these null arms carry no sequences: ['n3_pool']")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nf, "matchedness", refuse)
        with pytest.raises(nf.NoiseFloorError, match="carry no sequences"):
            nf.main(_prep_argv())
    assert sorted(path.name for path in out_dir.glob("*.fa")) == []
    assert not (out_dir / nf.PREP_JSON).exists()
    # Positive control: the identical call without the refusal writes all four, so the empty
    # listing above is an absence rather than the wrong directory.
    assert nf.main(_prep_argv()) == 0
    assert sorted(path.name for path in out_dir.glob("*.fa")) == sorted(
        f"{arm}.fa" for arm in nf.ARMS
    )


def test_prep_exits_nonzero_when_a_matchedness_control_fails(legs):
    """A failed control read as a successful prep lets the snakemake rule proceed, and the
    search leg then measures a null set that is not matched to the benchmark it calibrates.

    The break is one base of ONE pool host, outside that item's minted window and both of its
    scramble windows, so the scramble no longer agrees with its contig outside both windows and
    every other control stays satisfied ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    host_id = str(legs["items"][2]["host_id"])
    legs["rows"][host_id] = _flip_one_base(legs["hosts"][host_id], _free_position(2))
    assert nf.main(_prep_argv()) == 1
    prep = _written(legs["out_dir"] / nf.PREP_JSON)
    failed = {entry["name"]: entry for entry in prep["matchedness"] if not entry["fired"]}
    assert list(failed) == ["n2_agrees_outside_both_insert_windows"]
    assert failed["n2_agrees_outside_both_insert_windows"]["expected"] == 6
    assert failed["n2_agrees_outside_both_insert_windows"]["observed"] == 4
    # The artifact is still written, so the failure is inspectable rather than only printed.
    assert (legs["out_dir"] / nf.PREP_JSON).exists()
    # Positive control: the host restored, the identical call exits 0 — so the 1 above is not a
    # leg that always fails ([[raises-test-needs-a-positive-control]]).
    legs["rows"][host_id] = legs["hosts"][host_id]
    assert nf.main(_prep_argv()) == 0
    restored = _written(legs["out_dir"] / nf.PREP_JSON)["matchedness"]
    assert [entry["name"] for entry in restored if not entry["fired"]] == []


# ── search ───────────────────────────────────────────────────────────────────────────────
def test_search_records_minus_T_zero_on_every_search_in_both_the_argv_and_the_executed_line(legs):
    """A null searched at the threshold it calibrates reports nothing below that threshold, and
    the measured maximum becomes an artefact of the flag rather than a floor.

    ``search.json``'s header writes ``search_threshold_bits`` from the constant independently of
    the per-task flag, so a leg that searched at 10 bits would publish a SELF-CONSISTENT record
    saying 0.0 ([[gate-must-bind-to-upstream-evidence]]). The evidence is therefore the
    per-search argv and the option line ``cmsearch`` itself echoed, one assertion per search.
    """
    _prep_and_search(legs)
    record = _written(legs["tblout_dir"] / nf.SEARCH_JSON)
    searches = record["searches"]
    assert len(searches) == len(nf.ARMS) * len(hs.SCREEN_CMS) * len(nf.PIPELINES) == 16
    assert record["search_threshold_bits"] == nf.REPORT_THRESHOLD_BITS == 0.0
    for entry in searches:
        argv = entry["argv"]
        assert argv[argv.index("-T") + 1] == "0.0", entry["tblout"]
        assert "-T 0.0" in entry["option_settings"], entry["tblout"]
        assert entry["threshold_bits"] == 0.0, entry["tblout"]
    # "every search" is sixteen of sixteen, each cell exactly once — not three of sixteen.
    assert sorted((e["null_arm"], e["model"], e["pipeline"]) for e in searches) == sorted(
        (arm, model, pipeline)
        for arm in nf.ARMS
        for model, _cm in hs.SCREEN_CMS
        for pipeline, _flag in nf.PIPELINES
    )
    # The ``max`` pipeline really did pass ``--max``, so the two pipelines are not one run twice.
    assert {"--max" in e["argv"] for e in searches if e["pipeline"] == "max"} == {True}
    assert {"--max" in e["argv"] for e in searches if e["pipeline"] == "default"} == {False}
    # And it landed in main's default tblout tree, no ``--tblout-dir`` having been passed.
    assert (legs["out_dir"] / "tblout" / nf.SEARCH_JSON).exists()


def test_search_refuses_a_fasta_whose_sha256_is_not_the_one_prep_recorded(legs):
    """Without the guard the tblouts need not describe the FASTA ``prep`` wrote, and the pairing
    of a score to a sequence is positional: a mismatched FASTA reads as "no hits", the fail-open
    direction ([[gate-must-bind-to-upstream-evidence]])."""
    assert nf.main(_prep_argv()) == 0
    fasta = legs["out_dir"] / "n1_inserts.fa"
    recorded = _written(legs["out_dir"] / nf.PREP_JSON)["arms"]["n1_inserts"]["sha256"]
    records = nf.read_fasta(fasta)
    name = next(iter(records))
    nf.write_fasta(fasta, {**records, name: _flip_one_base(records[name], 0)})
    # Not vacuous: the bytes really do differ from the digest prep recorded.
    assert provenance.sha256_file(fasta) != recorded
    with pytest.raises(nf.NoiseFloorError, match="prep recorded"):
        nf.main(_search_argv())
    assert not (legs["tblout_dir"] / nf.SEARCH_JSON).exists()
    # Positive control: prep rewrites the FASTA and its digest, and the identical call succeeds.
    assert nf.main(_prep_argv()) == 0
    assert nf.main(_search_argv()) == 0
    assert len(_written(legs["tblout_dir"] / nf.SEARCH_JSON)["searches"]) == 16


# ── report ───────────────────────────────────────────────────────────────────────────────
def test_report_writes_the_measured_floor_validates_clean_and_exits_zero(legs):
    """The positive control for every report sabotage below."""
    _prep_and_search(legs)
    assert nf.main(_report_argv()) == 0
    written = _written(legs["report"])
    assert written["kind"] == nf.KIND == hs.KIND_NOISE_FLOOR
    assert written["problems"] == nf.validate_report(written) == []
    assert written["search_threshold_bits"] == 0.0
    assert [entry["name"] for entry in written["matchedness"] if not entry["fired"]] == []
    # The planted maxima, one per model, each in a different arm AND a different pipeline.
    assert written["null_max_bits"] == {"RF00230": 16.8, "TBDB001": 9.9}
    assert written["cells"]["RF00230"]["n2_scramble1"]["max"]["max_bits"] == 16.8
    assert written["cells"]["TBDB001"]["n3_pool"]["default"]["max_bits"] == 9.9


def test_report_problems_are_the_validators_verdict_on_the_artifact_it_wrote(legs):
    """``problems: []`` with nothing checked is the fail-open shape, and the most dangerous one
    here: ``host_screen`` refuses this artifact when the list is non-empty, so an empty list it
    did not earn signs ``POSTMINT_THRESHOLD_BITS`` against an unvalidated measurement
    ([[total-validator-can-fail-open]]).

    Re-derived, not read: the written list must equal ``validate_report`` run again on the
    written bytes, on a tree whose verdict is NON-empty — an all-clean tree cannot tell a real
    verdict from a hardcoded ``[]`` ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    _prep_and_search(legs)
    _diverge_the_minted_contigs(legs)
    # The exit code is `test_report_exits_nonzero_when_the_artifact_it_wrote_carries_problems`'
    # subject; asserting it here too would make one defect fail two tests and blur which bit.
    nf.main(_report_argv())
    written = _written(legs["report"])
    assert written["problems"] == nf.validate_report(written) != []
    assert any(
        "n2_agrees_outside_both_insert_windows' failed" in problem
        for problem in written["problems"]
    )
    # Positive control: the same clause set on the undiverged tree earns an empty list, so the
    # non-empty one above is not a validator that refuses everything.
    _restore_the_minted_contigs(legs)
    nf.main(_report_argv())
    clean = _written(legs["report"])
    assert clean["problems"] == nf.validate_report(clean) == []


def test_report_re_derives_matchedness_from_the_searched_fastas_not_from_preps_own_block(legs):
    """A matchedness block copied forward from ``prep.json`` certifies bytes nobody re-read
    ([[gate-clauses-need-re-derivation]]).

    The two blocks must therefore DISAGREE on a tree where the FASTAs no longer match the minted
    contigs: prep's is a true record of what prep measured, and the report's must be a
    measurement of what was searched.
    """
    _prep_and_search(legs)
    prep = _written(legs["out_dir"] / nf.PREP_JSON)
    _diverge_the_minted_contigs(legs)
    nf.main(_report_argv())  # the exit code is another test's subject, as above
    written = _written(legs["report"])
    name = "n2_agrees_outside_both_insert_windows"
    prep_row = next(entry for entry in prep["matchedness"] if entry["name"] == name)
    report_row = next(entry for entry in written["matchedness"] if entry["name"] == name)
    # An identity on each side, not just "the blocks differ": 6 of 6 was true when prep ran and
    # 4 of 6 is true of the bytes the report re-read.
    assert prep_row["observed"] == 6 and prep_row["fired"] is True
    assert report_row["observed"] == 4 and report_row["fired"] is False
    assert written["matchedness"] != prep["matchedness"]
    # Positive control: on the undiverged tree the re-derivation agrees with prep, so the
    # disagreement above is the divergence and not a report that never matches.
    _restore_the_minted_contigs(legs)
    nf.main(_report_argv())
    assert _written(legs["report"])["matchedness"] == prep["matchedness"]


def test_report_exits_nonzero_when_the_artifact_it_wrote_carries_problems(legs):
    """The exit code is the snakemake rule's only signal. Exiting 0 beside a non-empty problems
    list lets the phase proceed on an artifact this module's own validator refused."""
    _prep_and_search(legs)
    _diverge_the_minted_contigs(legs)
    assert nf.main(_report_argv()) == 1
    assert _written(legs["report"])["problems"] != []
    # Positive control: a clean artifact exits 0, so the 1 above is not a leg that always fails.
    _restore_the_minted_contigs(legs)
    assert nf.main(_report_argv()) == 0
    assert _written(legs["report"])["problems"] == []


def test_report_counts_null_sequences_once_per_arm_not_once_per_search(legs):
    """Two pipelines search each arm, so a per-SEARCH sum reports twice the sequences that
    exist — the 6,146-for-3,073 double count the module names. The figure is quoted beside an
    extreme-value maximum, which RISES with it, so a doubled denominator understates how hard
    the floor was looked for.
    """
    _prep_and_search(legs)
    # The exit code is asserted LAST here: the validator re-derives this same figure, so a rc
    # assertion up front would make the leg fail on its own verdict before the identity below
    # could say which number was wrong ([[sabotage-attribution-names-the-test]]).
    nf.main(_report_argv())
    written = _written(legs["report"])
    per_arm = {arm: len(nf.read_fasta(legs["out_dir"] / f"{arm}.fa")) for arm in nf.ARMS}
    assert per_arm == {
        "n1_inserts": 3,
        "n2_scramble1": 3,
        "n2_scramble2": 3,
        "n3_pool": nf.N_POOL_WINDOWS,
    }
    assert sum(per_arm.values()) == 1009
    for model, _cm in hs.SCREEN_CMS:
        assert written["n_null_sequences"][model] == sum(per_arm.values()), model
        # Named, not merely "not doubled": the per-search sum is 2018, two pipelines per arm.
        assert written["n_null_sequences"][model] != sum(per_arm.values()) * len(nf.PIPELINES)
    # The arms are NOT equal-sized, so this total is not one a doubling could coincide with
    # ([[symmetric-count-fixture-blind-to-inversion]]).
    assert len(set(per_arm.values())) > 1
    # And the validator's own re-derivation of the figure from the cells agrees, which is the
    # second, independent statement of the identity above.
    assert written["problems"] == []
    assert nf.main(_report_argv()) == 0


def test_report_refuses_a_tblout_whose_data_rows_the_search_record_does_not_cover(legs):
    """A cell computed from rows the search never recorded is a measurement of a file rather
    than of a search, and this one would raise the published RF00230 floor from 16.8 to 99.9
    ([[gate-must-bind-to-upstream-evidence]])."""
    _prep_and_search(legs)
    relative = "max/RF00230/n2_scramble1.tblout"
    tblout = legs["tblout_dir"] / relative
    recorded = next(
        entry
        for entry in _written(legs["tblout_dir"] / nf.SEARCH_JSON)["searches"]
        if entry["tblout"] == relative
    )
    text = tblout.read_text(encoding="utf-8")
    planted = f"{text}{_row('s1_0001', 99.9)}\n"
    tblout.write_text(planted, encoding="utf-8")
    # Not vacuous, twice over: the digest really moved, and the row really would change the cell.
    assert hs.tblout_digest(planted) != recorded["data_sha256"]
    assert max(nf.best_by_target(planted).values()) == 99.9
    with pytest.raises(nf.NoiseFloorError, match="data rows differ from the search record"):
        nf.main(_report_argv())
    assert not legs["report"].exists()
    # Positive control: the row removed, the identical call writes the measured floor.
    tblout.write_text(text, encoding="utf-8")
    assert nf.main(_report_argv()) == 0
    assert _written(legs["report"])["null_max_bits"]["RF00230"] == 16.8


# ── main and build_parser — where the artifacts land ─────────────────────────────────────
def test_main_defaults_the_tblout_dir_under_the_out_dir_for_both_legs(legs, monkeypatch):
    """Without the default, ``prep`` writes its FASTAs into one tree and ``search`` its tblouts
    into another, and ``report`` reads a tree the search never filled."""
    seen: dict[str, str | None] = {}

    def capture(args) -> int:
        seen[args.leg] = args.tblout_dir
        return 0

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nf, "_cmd_search", capture)
        patch.setattr(nf, "_cmd_report", capture)
        for leg, argv in (("search", _search_argv()), ("report", _report_argv())):
            assert nf.main(argv) == 0
            assert seen[leg] == str(Path(_WORK_DIR) / "tblout"), leg
        # Positive control: an explicit --tblout-dir is honoured, not overwritten by the default.
        assert nf.main(_search_argv("--tblout-dir", "elsewhere/tbl")) == 0
        assert seen["search"] == "elsewhere/tbl"
    # And the side effect, through the real legs: search writes, and report reads, the SAME
    # default tree, with no --tblout-dir passed to either.
    _prep_and_search(legs)
    assert (legs["out_dir"] / "tblout" / nf.SEARCH_JSON).exists()
    assert nf.main(_report_argv()) == 0
    assert _written(legs["report"])["problems"] == []


def test_the_report_out_default_is_the_path_host_screen_reads_the_floor_from(legs):
    """``host_screen`` reads the measured floor from its own ``DEFAULT_NOISE_FLOOR_REPORT``. A
    report written anywhere else leaves that path holding an older measurement, and
    ``POSTMINT_THRESHOLD_BITS`` is signed against whatever is sitting there
    ([[gate-must-bind-to-upstream-evidence]])."""
    assert nf.build_parser().parse_args(["report"]).out == nf.DEFAULT_REPORT
    assert nf.DEFAULT_REPORT == hs.DEFAULT_NOISE_FLOOR_REPORT
    assert nf.DEFAULT_REPORT == "reports/p3/postmint_noise_floor.json"
    # The side effect, through the real leg: with no --out, the artifact lands on that path.
    _prep_and_search(legs)
    argv = [token for token in _report_argv() if token not in ("--out", _REPORT_OUT)]
    assert "--out" not in argv and not legs["report"].exists()
    assert nf.main(argv) == 0
    assert _written(legs["root"] / nf.DEFAULT_REPORT)["kind"] == nf.KIND
