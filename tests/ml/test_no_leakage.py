"""P0-24 — the load-bearing no-leakage gate (CLAUDE.md §8.2; ADR-0004 D7).

Homology leakage across the train/test boundary inflates the generalization metric
that *is* the paper's central result, so this test re-checks the **real** committed
split-assignment partition (P0-23; ``data/processed/splits/split_assignments.parquet``,
23,569 records = 23,535 corpus + 16 anchor + 18 blind) on **every PR**. It asserts,
for every §9.2 scheme, the ADR-0004 D7 contract:

- **Holdout-unit separation + cluster non-splitting** — no homology cluster is split
  across a scheme's folds, and the scheme's holdout unit (the cluster, and the taxon at
  the scheme's holdout rank) does not straddle train/test. This is deliberately **not**
  non-spanning at *every* rank (leave-one-order-out intentionally holds a held-out
  order's lower ranks together), so per-rank straddle is checked only where a whole
  taxon is designated held out (the LOO orders; the Actinobacteria phylum; the
  literature anchors) — never at a rank a training taxon legitimately shares (e.g.
  Firmicutes at phylum).
- **Variant→parent→fold provenance** — every augmented/synthetic variant inherits its
  parent record's fold across every scheme (currently vacuous — the committed base
  table is fully self-parented — but structurally enforced so P2 class-II augmentation
  cannot smuggle a held-out parent into training).
- **Defined behaviour on no-clade records (D4)** — a corpus record with no clade label
  cannot silently pass: ``resolved_order`` NULL ⟺ ``dropped_from_clade_holdout``, and a
  dropped record is never in the clade-holdout training fold, never a designated holdout
  unit, and is kept in the random-only bucket.
- **Independent held-out sets** — the 16-record P0-16 literature anchor, the 18-record
  class-II set, and every external's corpus cluster-twin are all held out of training.

Design (mirrors the repo's pure-core / thin-adapter pattern):
- The leakage predicates are **stdlib-only pure functions** over plain column lists;
  each is unit-tested against a hand-built clean table (0 violations) **and** a
  deliberately-leaky one (the guard fires) — so the gate is proven to bite, never
  weakened to pass (§8.7, §10.3).
- ``_load_committed_columns`` reads the parquet via ``pyarrow`` (the stdlib has no
  parquet reader). In CI the ``test`` job checks out with ``lfs: true`` and installs
  ``pyarrow``, and sets ``TBOX_REQUIRE_NO_LEAKAGE=1`` so an unreadable table **fails**
  the job rather than silently skipping — the gate cannot rot. Locally (env var unset)
  the real-table tier skips gracefully when pyarrow / the smudged table is absent; run
  it under the data env for the §8.5 manual gate.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tbox_finder import splits

_REPO = Path(__file__).resolve().parents[2]
_COMMITTED = _REPO / "data" / "processed" / "splits" / "split_assignments.parquet"
_PROVENANCE = _REPO / "data" / "processed" / "splits" / "split_assignments.provenance.json"
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

#: Roles on the held-out side of the nested single-checkpoint realization (ADR-0004 D5):
#: a held-out-clade member, or a training-clade member of a clade-crossing cluster that
#: D3 forces out of training. ``dropped`` is neither train nor held-out (random-only).
_HELDOUT_ROLES = frozenset({"heldout", "excluded_clade_crossing"})

#: The committed composition (P0-23): the two independent held-out sets that must be
#: present and non-leaking. P0-17's additional non-Actinobacteria class-II set is
#: VERIFIED-EMPTY (0 records) — there is nothing to include, and none is fabricated.
_EXPECTED_SOURCE_COUNTS = {"corpus": 23535, "anchor": 16, "blind": 18}
_EXPECTED_N_RECORDS = 23569

#: Every fold-scheme + nested column a variant must inherit from its parent (§9.2).
_FOLD_COLS = (*splits.FOLD_SCHEME_COLUMNS,)


# ========================================================================== #
# Pure leakage predicates — stdlib only, over ``cols`` (dict[str, list]).
# Each returns the (possibly empty) list of violations; empty ⇒ no leak.
# ========================================================================== #
def clusters_split_across_random_fold(cluster_id, fold_random):
    """Scheme A (random): a whole cluster must land in one ``fold_random`` (never split).

    ``None`` folds (the externals, held out of the random reference) do not participate.
    """
    folds = defaultdict(set)
    for cid, f in zip(cluster_id, fold_random, strict=True):
        if f is not None:
            folds[cid].add(f)
    return sorted(cid for cid, fs in folds.items() if len(fs) > 1)


def clusters_mixing_train_and_heldout(cluster_id, nested_role):
    """Every leave-clade-out scheme (nested realization): a cluster must not contain a
    ``train`` member alongside a held-out one — the general cluster-non-splitting guard."""
    roles = defaultdict(set)
    for cid, r in zip(cluster_id, nested_role, strict=True):
        roles[cid].add(r)
    return sorted(cid for cid, rs in roles.items() if "train" in rs and (rs & _HELDOUT_ROLES))


def nested_role_train_flag_inconsistent(nested_role, nested_train):
    """Internal consistency: ``nested_train`` ⟺ ``nested_role == 'train'``."""
    return [
        i
        for i, (r, t) in enumerate(zip(nested_role, nested_train, strict=True))
        if (r == "train") != bool(t)
    ]


def designated_holdouts_in_training(
    is_designated_loo, is_anchor_heldout, resolved_phylum, nested_train
):
    """Holdout-unit taxon separation: a *designated* held-out unit — a leave-one-order-out
    order, the Actinobacteria phylum holdout (D5 ii), or a literature anchor (arm c) —
    must never sit in the nested training fold."""
    out = []
    for i, trained in enumerate(nested_train):
        if not trained:
            continue
        if is_designated_loo[i] or is_anchor_heldout[i] or resolved_phylum[i] == "Actinobacteria":
            out.append(i)
    return out


def order_rank_train_test_straddle(is_designated_loo, resolved_order, nested_train):
    """Headline scheme (leave-one-order-out): the taxon at the holdout rank — the order —
    must not straddle. Designated held-out orders and trained orders must be disjoint.
    Checked at the *order* rank only, so Firmicutes (shared across held/train orders at
    the phylum rank) is correctly not flagged."""
    held = {
        o for d, o in zip(is_designated_loo, resolved_order, strict=True) if d and o is not None
    }
    trained = {o for t, o in zip(nested_train, resolved_order, strict=True) if t and o is not None}
    return sorted(held & trained)


def phylum_holdout_cluster_straddle(cluster_id, resolved_phylum, nested_train):
    """Phylum scheme (Actinobacteria): no cluster containing an Actinobacteria member may
    also contain a trained member (D3 forces the whole cluster out of training)."""
    held_clusters = {
        c for c, p in zip(cluster_id, resolved_phylum, strict=True) if p == "Actinobacteria"
    }
    return sorted(
        {c for c, t in zip(cluster_id, nested_train, strict=True) if t and c in held_clusters}
    )


def no_clade_records_silently_passing(
    source, resolved_order, dropped, nested_train, is_designated_loo, fold_random
):
    """D4 fail-closed: a corpus record lacking a clade label cannot silently pass.

    ``resolved_order`` NULL ⟺ ``dropped_from_clade_holdout`` (flagged, not silent), and a
    dropped record is never in clade-holdout training, never a designated holdout unit,
    and is kept in the random-only bucket (a non-null ``fold_random``). Externals may
    legitimately have a null order without being ``dropped`` (they are anchors), so the
    ⟺ is checked over corpus rows only.
    """
    viol = []
    for i, src in enumerate(source):
        if src != "corpus":
            continue
        if (resolved_order[i] is None) != bool(dropped[i]):
            viol.append(("flag_mismatch", i))
            continue
        if dropped[i]:
            if nested_train[i]:
                viol.append(("dropped_in_training", i))
            if is_designated_loo[i]:
                viol.append(("dropped_is_designated_holdout", i))
            if fold_random[i] is None:
                viol.append(("dropped_not_in_random", i))
    return viol


def external_positive_leaks(source, is_anchor_heldout, nested_train, nested_role, fold_random):
    """Arm (c): every independent positive (anchor / blind) must be flagged held out, out
    of the nested training fold, not a nested-``train`` role, and out of the random split."""
    return [
        i
        for i, src in enumerate(source)
        if src != "corpus"
        and (
            not is_anchor_heldout[i]
            or nested_train[i]
            or nested_role[i] == "train"
            or fold_random[i] is not None
        )
    ]


def external_cluster_training_twins(source, cluster_id, nested_train):
    """A corpus record that shares a homology cluster with an external held-out positive
    must not be in training (else a held-out anchor has a training twin)."""
    ext_clusters = {c for s, c in zip(source, cluster_id, strict=True) if s != "corpus"}
    return sorted(
        i
        for i, src in enumerate(source)
        if src == "corpus" and cluster_id[i] in ext_clusters and nested_train[i]
    )


def variant_parent_fold_mismatches(cols):
    """Variant→parent→fold: a variant (``parent_record_id != record_id``) must inherit its
    parent's fold across every scheme, and its parent must exist in the table."""
    index = {rid: i for i, rid in enumerate(cols["record_id"])}
    viol = []
    for i, (rid, pid) in enumerate(zip(cols["record_id"], cols["parent_record_id"], strict=True)):
        if pid is None or pid == rid:
            continue
        j = index.get(pid)
        if j is None:
            viol.append(("orphan_parent", i))
            continue
        for col in _FOLD_COLS:
            if cols[col][i] != cols[col][j]:
                viol.append((col, i))
                break
    return viol


# ========================================================================== #
# Pure-predicate unit tests — a clean table passes; a leaky one is caught.
# ========================================================================== #
def _clean_cols():
    """A tiny well-formed table exercising every role/scheme (all predicates ⇒ clean).

    Rows: 0,1 train (Bacillales, one cluster); 2,3 held-out designated order
    (Frankiales/Actinobacteria, one cluster); 4 no-clade dropped (random-only); 5 anchor.
    """
    return {
        "record_id": ["r0", "r1", "r2", "r3", "r4", "anchor:a5"],
        "parent_record_id": ["r0", "r1", "r2", "r3", "r4", "anchor:a5"],
        "corpus_record_sha256": ["h0", "h1", "h2", "h3", "h4", None],
        "source": ["corpus", "corpus", "corpus", "corpus", "corpus", "anchor"],
        "klass": ["I", "I", "I", "I", "I", "I"],
        "cluster_id": [0, 0, 1, 1, 2, 3],
        "resolved_phylum": [
            "Firmicutes",
            "Firmicutes",
            "Actinobacteria",
            "Actinobacteria",
            "Firmicutes",
            None,
        ],
        "resolved_class": ["Bacilli", "Bacilli", "Actinobacteria", "Actinobacteria", None, None],
        "resolved_order": ["Bacillales", "Bacillales", "Frankiales", "Frankiales", None, None],
        "resolved_genus": ["Bacillus", "Bacillus", "Frankia", "Frankia", None, None],
        "fold_random": ["train", "train", "val", "val", "test", None],
        "loo_order_unit": ["Bacillales", "Bacillales", "Frankiales", "Frankiales", None, None],
        "class_holdout_unit": [
            "Bacilli",
            "Bacilli",
            "Actinobacteria",
            "Actinobacteria",
            None,
            None,
        ],
        "phylum_holdout_unit": [
            "Firmicutes",
            "Firmicutes",
            "Actinobacteria",
            "Actinobacteria",
            None,
            None,
        ],
        "nested_train": [True, True, False, False, False, False],
        "nested_role": ["train", "train", "heldout", "heldout", "dropped", "heldout"],
        "is_designated_loo_holdout": [False, False, True, True, False, False],
        "is_anchor_heldout": [False, False, False, False, False, True],
        "clade_crossing_cluster": [False, False, False, False, False, False],
        "dropped_from_clade_holdout": [False, False, False, False, True, False],
    }


def _all_violations(c):
    """Every predicate over ``c`` — used by the clean-table test to assert 0 leaks."""
    return {
        "random_split": clusters_split_across_random_fold(c["cluster_id"], c["fold_random"]),
        "nested_mix": clusters_mixing_train_and_heldout(c["cluster_id"], c["nested_role"]),
        "role_flag": nested_role_train_flag_inconsistent(c["nested_role"], c["nested_train"]),
        "designated": designated_holdouts_in_training(
            c["is_designated_loo_holdout"],
            c["is_anchor_heldout"],
            c["resolved_phylum"],
            c["nested_train"],
        ),
        "order_straddle": order_rank_train_test_straddle(
            c["is_designated_loo_holdout"], c["resolved_order"], c["nested_train"]
        ),
        "phylum_straddle": phylum_holdout_cluster_straddle(
            c["cluster_id"], c["resolved_phylum"], c["nested_train"]
        ),
        "d4": no_clade_records_silently_passing(
            c["source"],
            c["resolved_order"],
            c["dropped_from_clade_holdout"],
            c["nested_train"],
            c["is_designated_loo_holdout"],
            c["fold_random"],
        ),
        "external": external_positive_leaks(
            c["source"],
            c["is_anchor_heldout"],
            c["nested_train"],
            c["nested_role"],
            c["fold_random"],
        ),
        "external_twin": external_cluster_training_twins(
            c["source"], c["cluster_id"], c["nested_train"]
        ),
        "variant": variant_parent_fold_mismatches(c),
    }


def test_clean_table_has_no_leaks():
    assert all(v == [] for v in _all_violations(_clean_cols()).values())


def test_random_cluster_split_is_caught():
    c = _clean_cols()
    c["fold_random"][1] = "val"  # cluster 0 now straddles train/val
    assert clusters_split_across_random_fold(c["cluster_id"], c["fold_random"]) == [0]


def test_nested_cluster_train_heldout_mix_is_caught():
    c = _clean_cols()
    c["nested_train"][1] = False
    c["nested_role"][1] = "heldout"  # cluster 0 now mixes train + heldout
    assert clusters_mixing_train_and_heldout(c["cluster_id"], c["nested_role"]) == [0]


def test_designated_holdout_in_training_is_caught():
    c = _clean_cols()
    c["nested_train"][2] = True  # a designated leave-one-order-out record now trains
    assert 2 in designated_holdouts_in_training(
        c["is_designated_loo_holdout"],
        c["is_anchor_heldout"],
        c["resolved_phylum"],
        c["nested_train"],
    )


def test_anchor_in_training_is_caught_as_designated():
    c = _clean_cols()
    c["nested_train"][5] = True  # the literature anchor now trains
    assert 5 in designated_holdouts_in_training(
        c["is_designated_loo_holdout"],
        c["is_anchor_heldout"],
        c["resolved_phylum"],
        c["nested_train"],
    )


def test_order_rank_straddle_is_caught():
    c = _clean_cols()
    c["resolved_order"][0] = "Frankiales"  # a trained record shares a held-out order
    assert order_rank_train_test_straddle(
        c["is_designated_loo_holdout"], c["resolved_order"], c["nested_train"]
    ) == ["Frankiales"]


def test_phylum_holdout_cluster_straddle_is_caught():
    c = _clean_cols()
    c["nested_train"][3] = True  # an Actinobacteria cluster member now trains
    assert phylum_holdout_cluster_straddle(
        c["cluster_id"], c["resolved_phylum"], c["nested_train"]
    ) == [1]


def test_firmicutes_at_phylum_rank_is_not_a_false_straddle():
    # Firmicutes spans trained (Bacillales) + held-out orders at the phylum rank — this is
    # the intended leave-one-order-out behaviour, NOT a leak (ADR-0004 D7).
    c = _clean_cols()
    assert (
        phylum_holdout_cluster_straddle(c["cluster_id"], c["resolved_phylum"], c["nested_train"])
        == []
    )


def test_role_train_flag_inconsistency_is_caught():
    c = _clean_cols()
    c["nested_train"][0] = False  # role says train, flag says not
    assert nested_role_train_flag_inconsistent(c["nested_role"], c["nested_train"]) == [0]


def test_no_clade_record_unflagged_is_caught():
    c = _clean_cols()
    c["dropped_from_clade_holdout"][4] = False  # order NULL but not flagged dropped
    assert ("flag_mismatch", 4) in no_clade_records_silently_passing(
        c["source"],
        c["resolved_order"],
        c["dropped_from_clade_holdout"],
        c["nested_train"],
        c["is_designated_loo_holdout"],
        c["fold_random"],
    )


def test_dropped_record_leaking_into_training_is_caught():
    c = _clean_cols()
    c["nested_train"][4] = True  # a no-clade record slips into clade-holdout training
    assert ("dropped_in_training", 4) in no_clade_records_silently_passing(
        c["source"],
        c["resolved_order"],
        c["dropped_from_clade_holdout"],
        c["nested_train"],
        c["is_designated_loo_holdout"],
        c["fold_random"],
    )


def test_dropped_record_out_of_random_bucket_is_caught():
    c = _clean_cols()
    c["fold_random"][4] = None  # a dropped record must stay in the random-only bucket
    assert ("dropped_not_in_random", 4) in no_clade_records_silently_passing(
        c["source"],
        c["resolved_order"],
        c["dropped_from_clade_holdout"],
        c["nested_train"],
        c["is_designated_loo_holdout"],
        c["fold_random"],
    )


def test_external_positive_training_leak_is_caught():
    c = _clean_cols()
    c["nested_train"][5] = True  # an external held-out positive trains
    assert external_positive_leaks(
        c["source"],
        c["is_anchor_heldout"],
        c["nested_train"],
        c["nested_role"],
        c["fold_random"],
    ) == [5]


def test_external_in_random_split_is_caught():
    c = _clean_cols()
    c["fold_random"][5] = "train"  # an external entered the in-distribution random split
    assert external_positive_leaks(
        c["source"],
        c["is_anchor_heldout"],
        c["nested_train"],
        c["nested_role"],
        c["fold_random"],
    ) == [5]


def test_external_cluster_training_twin_is_caught():
    c = _clean_cols()
    # a corpus record joins the anchor's cluster (3) and is trained → a training twin
    for col, val in {
        "record_id": "r6",
        "parent_record_id": "r6",
        "corpus_record_sha256": "h6",
        "source": "corpus",
        "klass": "I",
        "cluster_id": 3,
        "resolved_phylum": "Chloroflexi",
        "resolved_class": None,
        "resolved_order": "Dehalococcoidales",
        "resolved_genus": None,
        "fold_random": "train",
        "loo_order_unit": "Dehalococcoidales",
        "class_holdout_unit": None,
        "phylum_holdout_unit": "Chloroflexi",
        "nested_train": True,
        "nested_role": "train",
        "is_designated_loo_holdout": False,
        "is_anchor_heldout": False,
        "clade_crossing_cluster": True,
        "dropped_from_clade_holdout": False,
    }.items():
        c[col].append(val)
    assert external_cluster_training_twins(c["source"], c["cluster_id"], c["nested_train"]) == [6]


def test_variant_inheriting_wrong_fold_is_caught():
    c = _clean_cols()
    # append a class-II augmentation variant of r0 (a training parent) placed into heldout
    for col, val in {
        "record_id": "r0#aug1",
        "parent_record_id": "r0",
        "corpus_record_sha256": None,
        "source": "corpus",
        "klass": "II",
        "cluster_id": 0,
        "resolved_phylum": "Firmicutes",
        "resolved_class": "Bacilli",
        "resolved_order": "Bacillales",
        "resolved_genus": "Bacillus",
        "fold_random": "train",
        "loo_order_unit": "Bacillales",
        "class_holdout_unit": "Bacilli",
        "phylum_holdout_unit": "Firmicutes",
        "nested_train": False,
        "nested_role": "heldout",
        "is_designated_loo_holdout": False,
        "is_anchor_heldout": False,
        "clade_crossing_cluster": False,
        "dropped_from_clade_holdout": False,
    }.items():
        c[col].append(val)
    viol = variant_parent_fold_mismatches(c)
    assert any(i == 6 for _, i in viol)


def test_variant_inheriting_parent_fold_is_clean():
    c = _clean_cols()
    for col in c:  # a faithful variant of r0 — every fold field copied from the parent
        c[col].append({"record_id": "r0#aug1", "parent_record_id": "r0"}.get(col, c[col][0]))
    assert variant_parent_fold_mismatches(c) == []


# ========================================================================== #
# Real committed-table tier — the load-bearing gate over the full partition.
# ========================================================================== #
def _fail_or_skip(reason: str) -> None:
    """In CI (``TBOX_REQUIRE_NO_LEAKAGE=1``) an unreadable table FAILS the gate; locally it
    skips — so the CI leakage gate can never silently rot (§10.3)."""
    if os.environ.get("TBOX_REQUIRE_NO_LEAKAGE") == "1":
        pytest.fail(f"TBOX_REQUIRE_NO_LEAKAGE=1 but the split table is unusable: {reason}")
    pytest.skip(reason)


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(len(_LFS_MAGIC)) == _LFS_MAGIC


def _load_committed_columns():
    if not _COMMITTED.exists():
        _fail_or_skip("committed split-assignment table absent")
    if _is_lfs_pointer(_COMMITTED):
        _fail_or_skip("committed table is an unsmudged Git-LFS pointer (checkout without lfs)")
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - exercised only where pyarrow is absent
        _fail_or_skip("pyarrow not installed (needed to read the committed parquet)")
    return pq.read_table(_COMMITTED).to_pydict()


@pytest.fixture(scope="module")
def committed():
    return _load_committed_columns()


def test_committed_table_shape_and_composition(committed):
    c = committed
    assert set(splits.COMMITTED_TABLE_COLUMNS) <= set(c)
    assert not (splits.SEQUENCE_COLUMN_DENYLIST & set(c))  # structurally sequence-free
    assert len(c["record_id"]) == _EXPECTED_N_RECORDS
    assert len(set(c["record_id"])) == _EXPECTED_N_RECORDS  # record_id unique
    # The two independent held-out sets are actually present (a regression that dropped
    # them would fail here); P0-17's additional class-II set is verified-empty (absent).
    assert dict(Counter(c["source"])) == _EXPECTED_SOURCE_COUNTS
    if _PROVENANCE.exists():
        prov = json.loads(_PROVENANCE.read_text())
        assert len(c["record_id"]) == prov["extra"]["n_records"]


def test_scheme_random_cluster_non_splitting(committed):
    c = committed
    assert clusters_split_across_random_fold(c["cluster_id"], c["fold_random"]) == []


def test_scheme_leave_clade_out_cluster_non_splitting(committed):
    # nested realization ⇒ every leave-clade-out scheme (order / class / phylum): no
    # cluster mixes a trained and a held-out member.
    c = committed
    assert clusters_mixing_train_and_heldout(c["cluster_id"], c["nested_role"]) == []
    assert nested_role_train_flag_inconsistent(c["nested_role"], c["nested_train"]) == []


def test_scheme_leave_one_order_out_holdout_unit_separation(committed):
    c = committed
    assert (
        order_rank_train_test_straddle(
            c["is_designated_loo_holdout"], c["resolved_order"], c["nested_train"]
        )
        == []
    )
    assert (
        designated_holdouts_in_training(
            c["is_designated_loo_holdout"],
            c["is_anchor_heldout"],
            c["resolved_phylum"],
            c["nested_train"],
        )
        == []
    )


def test_scheme_phylum_actinobacteria_holdout_unit_separation(committed):
    c = committed
    assert (
        phylum_holdout_cluster_straddle(c["cluster_id"], c["resolved_phylum"], c["nested_train"])
        == []
    )


def test_no_clade_records_have_defined_behaviour(committed):
    c = committed
    assert (
        no_clade_records_silently_passing(
            c["source"],
            c["resolved_order"],
            c["dropped_from_clade_holdout"],
            c["nested_train"],
            c["is_designated_loo_holdout"],
            c["fold_random"],
        )
        == []
    )


def test_independent_positive_sets_are_held_out(committed):
    c = committed
    assert (
        external_positive_leaks(
            c["source"],
            c["is_anchor_heldout"],
            c["nested_train"],
            c["nested_role"],
            c["fold_random"],
        )
        == []
    )
    assert external_cluster_training_twins(c["source"], c["cluster_id"], c["nested_train"]) == []


def test_variant_parent_fold_inheritance(committed):
    assert variant_parent_fold_mismatches(committed) == []
