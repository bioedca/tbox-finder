"""Unit gates for the §9.1 decoy embedding + its matched junction control (P2-10d′-b).

The load-bearing claim this file defends is **matchedness**: that the junction control
differs from its decoy counterpart in nothing but the contents of the replaced interval.
Per [[control-matchedness-must-be-asserted]] the invariant the argument needs must be
asserted directly — a control the generator copies from the treatment arm, or one that
silently failed to splice, would keep every geometry test green while the separation the
step reports became meaningless. So the pairs are compared **base by base**, inside and
outside the interval, and the generator itself is sabotaged in
`tests/unit/test_embedding_generator.py`.
"""

from __future__ import annotations

import hashlib

import pytest

from tbox_finder.data.embedding import (
    ARM_CONTROL,
    ARM_DECOY,
    EXCLUDED_DECOY_POOLS,
    TRAINING_DECOY_POOLS,
    EmbeddedWindow,
    EmbeddingError,
    embed_decoy_rows,
    embedded_negative_records,
    junction_control,
    normalise_insert,
    plan_placement,
    splice,
)
from tbox_finder.data.negatives import is_negative_record

WINDOW = 64  # smoke-sized; every rule under test is width-agnostic
SEED = 20260721


def _host(i: int) -> dict[str, object]:
    """A synthetic 'mined window', pseudo-random per index.

    Deliberately NOT a tiled motif. The first version of this fixture was
    ``tag + "ACGT"*k``, which made two different hosts carry identical bases over most
    offsets — so the donor segment equalled the host segment, the control's splice was
    the identity, and `test_control_is_not_a_copy_of_its_host` failed. That exposed a
    real bug in :func:`junction_control` (now guarded), but a repetitive fixture would
    also have hidden the opposite failure ([[degenerate-fixture-generators]]): every
    arm looks alike when the alphabet has one word. shake_256 gives independent bases
    per position while keeping the fixture deterministic.
    """
    raw = hashlib.shake_256(f"host:{i}".encode()).digest(WINDOW)
    return {
        "candidate_id": f"rec{i}:lead",
        "sequence": "".join("ACGT"[b & 3] for b in raw),
        "source_record_id": f"rec{i}",
    }


def _hosts(n: int) -> list[dict[str, object]]:
    return [_host(i) for i in range(n)]


def _decoy_rows(n_per_pool: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool in TRAINING_DECOY_POOLS:
        for i in range(n_per_pool):
            rows.append(
                {
                    "decoy_id": f"{pool}_{i}",
                    "pool": pool,
                    "sequence": "GGGGCCCCAAAA"[: 8 + i],
                    "masked": False,
                }
            )
    # Every excluded pool must be REPRESENTED in the input, or the generator refuses:
    # an absent pool is indistinguishable from a renamed one that sailed through.
    for pool in EXCLUDED_DECOY_POOLS:
        rows.append(
            {"decoy_id": f"{pool}_0", "pool": pool, "sequence": "ACGTACGT", "masked": False}
        )
    return rows


# ── splice ───────────────────────────────────────────────────────────────────────────
def test_splice_replaces_and_preserves_length() -> None:
    host = "A" * 20
    out = splice(host, "GGGG", 5)
    assert len(out) == len(host)
    assert out == "A" * 5 + "GGGG" + "A" * 11


@pytest.mark.parametrize("phase", [0, 16])
def test_splice_admits_flush_placements(phase: int) -> None:
    """Flush-left and flush-right yield ONE junction, and must stay reachable.

    Excluding them would make 'exactly two junctions' an invariant of the negative class
    that the scan distribution does not share.
    """
    assert len(splice("A" * 20, "GGGG", phase)) == 20


def test_splice_refuses_overrun() -> None:
    with pytest.raises(EmbeddingError, match="runs off the end"):
        splice("A" * 20, "GGGG", 17)


def test_splice_refuses_empty_insert() -> None:
    with pytest.raises(EmbeddingError, match="non-empty"):
        splice("A" * 20, "", 0)


def test_normalise_insert_maps_u_to_t() -> None:
    """structured_rna carries 1,784 U; encode_bases would map each to the N id silently."""
    assert normalise_insert("acguACGU") == "ACGTACGT"


# ── placement determinism ────────────────────────────────────────────────────────────
def test_placement_is_deterministic_and_content_keyed() -> None:
    a = plan_placement(decoy_id="d1", insert_len=8, n_hosts=50, seed=SEED, window=WINDOW)
    b = plan_placement(decoy_id="d1", insert_len=8, n_hosts=50, seed=SEED, window=WINDOW)
    assert a == b


def test_placement_varies_with_decoy_and_with_seed() -> None:
    """A degenerate keying would map every decoy to one host/phase and the whole
    construction would collapse to a single window ([[degenerate-fixture-generators]])."""
    by_decoy = {
        plan_placement(decoy_id=f"d{i}", insert_len=8, n_hosts=50, seed=SEED, window=WINDOW)
        for i in range(40)
    }
    assert len(by_decoy) > 20, f"placements collapsed: {len(by_decoy)} distinct of 40"
    by_seed = {
        plan_placement(decoy_id="d1", insert_len=8, n_hosts=50, seed=s, window=WINDOW)
        for s in range(40)
    }
    assert len(by_seed) > 20, f"seed does not move the placement: {len(by_seed)} distinct"


def test_placement_phase_spans_the_whole_admissible_range() -> None:
    phases = [
        plan_placement(decoy_id=f"d{i}", insert_len=8, n_hosts=2, seed=SEED, window=WINDOW)[1]
        for i in range(400)
    ]
    assert min(phases) < WINDOW // 8 and max(phases) > (WINDOW - 8) - WINDOW // 8


def test_placement_refuses_an_insert_that_cannot_fit() -> None:
    with pytest.raises(EmbeddingError, match="does not fit"):
        plan_placement(decoy_id="d", insert_len=WINDOW + 1, n_hosts=2, seed=SEED, window=WINDOW)


# ── the A7 composition pins ──────────────────────────────────────────────────────────
def test_only_the_pinned_pools_are_embedded() -> None:
    windows, report = embed_decoy_rows(_decoy_rows(), _hosts(20), seed=SEED, window=WINDOW)
    assert {w.insert_pool for w in windows} == set(TRAINING_DECOY_POOLS)
    assert report["n_embedded"] == 6
    assert report["excluded_by_reason"]["pool_excluded_by_a7"] == len(EXCLUDED_DECOY_POOLS)


def test_gc_background_and_leader_decoy_are_excluded() -> None:
    """The two A7 share-0 pins, asserted by name rather than by arithmetic on a total."""
    assert set(EXCLUDED_DECOY_POOLS) == {"gc_background", "leader_decoy"}
    windows, _ = embed_decoy_rows(_decoy_rows(), _hosts(20), seed=SEED, window=WINDOW)
    assert not [w for w in windows if w.insert_pool in EXCLUDED_DECOY_POOLS]


def test_a_renamed_excluded_pool_is_an_error_not_a_silent_pass() -> None:
    """If `leader_decoy` vanished upstream, its records would enter under a new name and
    the exclusion would report success. Absence must be loud."""
    rows = [r for r in _decoy_rows() if r["pool"] != "leader_decoy"]
    with pytest.raises(EmbeddingError, match="excludes pool"):
        embed_decoy_rows(rows, _hosts(20), seed=SEED, window=WINDOW)


def test_a_missing_training_pool_is_an_error() -> None:
    rows = [r for r in _decoy_rows() if r["pool"] != "structured_rna"]
    with pytest.raises(EmbeddingError, match="pinned into the training mix"):
        embed_decoy_rows(rows, _hosts(20), seed=SEED, window=WINDOW)


def test_masked_decoys_are_refused() -> None:
    """Masking refuses the records it names — but not the whole pinned pool.

    Rewritten at P2-10d′-c, NOT relaxed. It used to mask *every* ``structured_rna`` row and
    assert ``n_embedded_by_pool["structured_rna"] == 0``, which is precisely the value the
    pandas-3 NaN bug produced — so the suite's only per-pool assertion certified the broken
    composition as correct. A pinned pool at share 0 is now an error in its own right (see
    :func:`test_a_pinned_pool_that_admits_nothing_is_refused`), so the refusal is asserted
    on a *subset* and the survivor is asserted too.
    """
    rows = _decoy_rows()
    masked = [r for r in rows if r["pool"] == "structured_rna"][:2]
    for r in masked:
        r["masked"] = True
    _, report = embed_decoy_rows(rows, _hosts(20), seed=SEED, window=WINDOW)
    assert report["excluded_by_reason"]["masked_known_locus"] == 2
    assert report["n_embedded_by_pool"]["structured_rna"] == 1


def test_a_pinned_pool_that_admits_nothing_is_refused() -> None:
    """A7 pin 3's shares are pool-proportional, and 0 is not a proportion.

    The pre-existing ``missing_wanted`` guard cannot catch this: ``seen_pools`` is stamped
    before every filter, so a pool whose every record is refused still counts as "seen".
    That gap is how ``structured_rna`` reached share 0 in a green run (P2-10d′-c).
    """
    rows = _decoy_rows()
    for r in rows:
        if r["pool"] == "structured_rna":
            r["masked"] = True
    with pytest.raises(EmbeddingError, match=r"every one of their records was refused"):
        embed_decoy_rows(rows, _hosts(20), seed=SEED, window=WINDOW)


# ── the decoy's OWN §9.2 parent rule, across pandas' two missing-value dialects ──────
# The bug these gates lock (P2-10d′-c): `str(row.get(col) or "")` yields `""` for pandas 2's
# `None` but the string `"nan"` for pandas 3's NaN sentinel — a present-looking parent id in
# no fold set. All 2,999 unmasked `structured_rna` decoys were refused as
# `decoy_parent_not_nested_train` in the training env while being admitted in CI's.
#
# These assertions use NaN as a LITERAL, not via a parquet round-trip, so they bite under
# pandas 2 and 3 alike; the round-trip test below locks the loader boundary but cannot
# discriminate under the pandas CI pins (it says so itself).
_FOLD = {"rec0", "rec1", "rec2", "rec3", "rec4", "rec5", "rec6", "rec7"}


def _parented(parent: object, n: int | None = None) -> list[dict[str, object]]:
    """The standard decoy rows with the first `n` `structured_rna` rows carrying `parent`.

    `n=None` means all of them. Tests that expect a REFUSAL must pass `n` < the pool size:
    a pinned pool refused down to zero is now an error in its own right, so refusing the
    whole pool would raise before the refusal *counts* could be asserted. Mixing admitted
    and refused rows is also the real situation — `dinuc_shuffled` ships 702 in-fold and
    1,298 out-of-fold.
    """
    rows = _decoy_rows()
    targets = [r for r in rows if r["pool"] == "structured_rna"]
    for r in targets if n is None else targets[:n]:
        r["source_record_id"] = parent
    return rows


@pytest.mark.parametrize(
    ("sentinel", "label"),
    [(None, "pandas-2 None"), (float("nan"), "pandas-3 NaN"), ("", "empty string")],
)
def test_a_missing_parent_reads_as_absent_not_as_out_of_fold(sentinel: object, label: str) -> None:
    """A decoy with NO parent has nothing to leak, and must be admitted in every dialect.

    `structured_rna` decoys are Rfam-derived, not permutations of corpus loci, so their
    `source_record_id` is null for all 3,000 of them. Reading that null as an id is what
    deleted the pool from the mix.
    """
    _, report = embed_decoy_rows(
        _parented(sentinel),
        _hosts(20),
        seed=SEED,
        window=WINDOW,
        training_fold_record_ids=_FOLD,
    )
    assert report["n_embedded_by_pool"]["structured_rna"] == 3, label
    assert "decoy_parent_not_nested_train" not in report["excluded_by_reason"], label


def test_a_parent_outside_the_fold_is_still_refused() -> None:
    """The other direction — the fix must not turn the §9.2 rule into a fail-open.

    Without this, collapsing every unrecognised parent to "absent" would admit exactly the
    held-out-locus permutations pin 6a exists to refuse.
    """
    _, report = embed_decoy_rows(
        _parented("rec_NOT_IN_FOLD", n=2),
        _hosts(20),
        seed=SEED,
        window=WINDOW,
        training_fold_record_ids=_FOLD,
    )
    assert report["excluded_by_reason"]["decoy_parent_not_nested_train"] == 2
    assert report["n_refused_decoy_parent_out_of_fold"] == 2
    assert report["n_embedded_by_pool"]["structured_rna"] == 1  # the parentless survivor


def test_a_parent_inside_the_fold_is_admitted() -> None:
    _, report = embed_decoy_rows(
        _parented("rec3"),
        _hosts(20),
        seed=SEED,
        window=WINDOW,
        training_fold_record_ids=_FOLD,
    )
    assert report["n_embedded_by_pool"]["structured_rna"] == 3
    assert report["n_refused_decoy_parent_out_of_fold"] == 0


def test_an_unverifiable_parent_is_refused_when_no_fold_set_is_supplied() -> None:
    """ "Could not check" must never read as "checked and clean" — the pre-existing rule,
    re-asserted here because the null-safety change touches the branch above it."""
    _, report = embed_decoy_rows(_parented("rec3", n=2), _hosts(20), seed=SEED, window=WINDOW)
    assert report["excluded_by_reason"]["decoy_parent_fold_unverifiable"] == 2
    assert report["decoy_parent_fold_checked"] is False
    assert report["n_embedded_by_pool"]["structured_rna"] == 1


def test_decoy_parent_nulls_survive_the_parquet_loader_boundary(tmp_path) -> None:
    """`load_decoy_rows` -> `embed_decoy_rows` on a REAL parquet with a null parent column.

    STATED LIMIT: under the pandas CI/data pins (2.3.3) the round-trip yields `None`, which
    the old code also handled, so this test CANNOT discriminate there — it is a boundary
    lock, not the bite. The parametrised NaN-literal test above is the one that fails under
    both dialects. Under pandas >= 3 this test additionally reproduces the original bug.
    """
    pd = pytest.importorskip("pandas", reason="the loader boundary is pandas-side")
    rows = _decoy_rows()
    for r in rows:
        r["source_record_id"] = None if r["pool"] == "structured_rna" else "rec3"
    path = tmp_path / "decoys.parquet"
    pd.DataFrame(rows).to_parquet(path)

    from tbox_finder.data.embedding import load_decoy_rows

    loaded = load_decoy_rows(path)
    assert len(loaded) == len(rows)
    _, report = embed_decoy_rows(
        loaded, _hosts(20), seed=SEED, window=WINDOW, training_fold_record_ids=_FOLD
    )
    assert report["n_embedded_by_pool"]["structured_rna"] == 3
    assert report["n_embedded_by_pool"]["dinuc_shuffled"] == 3
    assert "decoy_parent_not_nested_train" not in report["excluded_by_reason"]


def test_duplicate_decoy_id_is_refused() -> None:
    rows = _decoy_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(EmbeddingError, match="duplicate decoy_id"):
        embed_decoy_rows(rows, _hosts(20), seed=SEED, window=WINDOW)


def test_embedding_refuses_when_no_hosts_are_admitted() -> None:
    with pytest.raises(EmbeddingError, match="no admitted host windows"):
        embed_decoy_rows(_decoy_rows(), [], seed=SEED, window=WINDOW)


# ── matchedness: the claim the whole control rests on ────────────────────────────────
def _arms() -> tuple[list[EmbeddedWindow], list[EmbeddedWindow], list[dict[str, object]]]:
    hosts = _hosts(20)
    decoy, _ = embed_decoy_rows(_decoy_rows(), hosts, seed=SEED, window=WINDOW)
    control = junction_control(decoy, hosts, seed=SEED, window=WINDOW)
    return decoy, control, hosts


def test_control_matches_its_decoy_on_host_phase_and_length() -> None:
    decoy, control, _ = _arms()
    assert len(control) == len(decoy)
    for d, c in zip(decoy, control, strict=True):
        assert (c.host_id, c.phase, c.insert_len) == (d.host_id, d.phase, d.insert_len)
        assert c.junctions == d.junctions
        assert len(c.sequence) == len(d.sequence) == WINDOW


def test_control_and_decoy_agree_base_for_base_OUTSIDE_the_interval() -> None:
    """Geometry agreement is not enough — the flanking DNA must be literally identical,
    or the arms differ somewhere other than the thing under test."""
    decoy, control, _ = _arms()
    for d, c in zip(decoy, control, strict=True):
        lo, hi = d.junctions
        assert d.sequence[:lo] == c.sequence[:lo]
        assert d.sequence[hi:] == c.sequence[hi:]


def test_control_and_decoy_differ_INSIDE_the_interval() -> None:
    """The discriminating clause. A control that silently failed to splice would satisfy
    every test above and would report the junction as invisible no matter what it did."""
    decoy, control, _ = _arms()
    for d, c in zip(decoy, control, strict=True):
        lo, hi = d.junctions
        assert d.sequence[lo:hi] != c.sequence[lo:hi]


def test_control_is_not_a_copy_of_its_host() -> None:
    """If the donor segment came from the host itself the splice would be the identity,
    and the control would carry no junction at all."""
    _, control, hosts = _arms()
    by_id = {h["candidate_id"]: h["sequence"] for h in hosts}
    for c in control:
        assert c.sequence != by_id[c.host_id]


def test_control_donor_is_never_the_host() -> None:
    _, control, _ = _arms()
    for c in control:
        assert c.insert_id != c.host_id


def test_control_needs_at_least_two_windows() -> None:
    hosts = _hosts(20)
    decoy, _ = embed_decoy_rows(_decoy_rows(), hosts, seed=SEED, window=WINDOW)
    with pytest.raises(EmbeddingError, match="at least two mined windows"):
        junction_control(decoy, hosts[:1], seed=SEED, window=WINDOW)


# ── the records that actually reach training ─────────────────────────────────────────
def test_embedded_records_are_negatives_with_the_hosts_provenance() -> None:
    decoy, _, hosts = _arms()
    by_id = {h["candidate_id"]: h["source_record_id"] for h in hosts}
    records = embedded_negative_records(decoy, cluster_id_start=100, window=WINDOW)
    assert len(records) == len(decoy)
    for rec, w in zip(records, decoy, strict=True):
        assert is_negative_record(rec)
        # The §9.2 claim is the HOST's, and must be checkable against the split table.
        assert rec.source_record_id == by_id[w.host_id]
        assert set(rec.label_string) == {"."}
        assert len(rec.context_seq) == WINDOW


def test_embedded_cluster_ids_continue_the_plain_arms_namespace() -> None:
    decoy, _, _ = _arms()
    records = embedded_negative_records(decoy, cluster_id_start=100, window=WINDOW)
    ids = [r.cluster_id for r in records]
    assert ids == [-(100 + i + 1) for i in range(len(decoy))]
    assert all(i < 0 for i in ids)


def test_the_control_arm_is_refused_entry_to_training() -> None:
    """Training on the control would make the junction uninformative and destroy the
    measurement it exists to supply."""
    _, control, _ = _arms()
    with pytest.raises(EmbeddingError, match="only 'decoy' windows enter training"):
        embedded_negative_records(control, cluster_id_start=0, window=WINDOW)


def test_arm_names_are_distinct() -> None:
    assert ARM_DECOY != ARM_CONTROL
