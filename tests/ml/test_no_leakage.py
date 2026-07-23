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

import inspect
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

#: The committed composition (P0-23; P2-08 appended the recovery set). The two
#: independent held-out sets must be present and non-leaking. P0-17's additional
#: non-Actinobacteria class-II set is VERIFIED-EMPTY (0 records) — there is nothing
#: to include, and none is fabricated.
#:
#: ``synthetic_classII`` = the P2-08 evaluation-only recovery set (ADR-0004 A3):
#: 2,344 presentation variants of 1,172 held-out class-II corpus parents. These are
#: **derived** rows — they inherit their parent's fold and are never
#: training-eligible — so they are judged by :func:`synthetic_variant_leaks`, not by
#: the external-positive predicate.
_EXPECTED_SOURCE_COUNTS = {
    "corpus": 23535,
    "anchor": 16,
    "blind": 18,
    "synthetic_classII": 2344,
}
_EXPECTED_N_RECORDS = 25913

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
    of the nested training fold, not a nested-``train`` role, and out of the random split.

    Scoped to :data:`splits.EXTERNAL_POSITIVE_SOURCES` rather than to "not corpus"
    (ADR-0004 A3, P2-08). The old ``src != "corpus"`` form conflated two different
    kinds of row: an *independent* positive that stands outside every scheme, and a
    *derived* variant that inherits its parent's fold. The ``fold_random is None``
    clause is right for the former and unsatisfiable for the latter — a D7-conforming
    variant inherits its parent's non-null fold, so under the old form every correct
    variant was a violation. Derived rows are held to
    :func:`synthetic_variant_leaks`, which is strictly stronger, not weaker.
    """
    return [
        i
        for i, src in enumerate(source)
        if src in splits.EXTERNAL_POSITIVE_SOURCES
        and (
            not is_anchor_heldout[i]
            or nested_train[i]
            or nested_role[i] == "train"
            or fold_random[i] is not None
        )
    ]


def unknown_sources(source):
    """Any ``source`` value outside the closed vocabulary is a violation.

    Without this, re-scoping :func:`external_positive_leaks` to a named allowlist
    would silently exempt a typo'd or newly-invented source from *every* predicate
    — turning a tightening into a hole.
    """
    known = {"corpus", *splits.EXTERNAL_POSITIVE_SOURCES, *splits.DERIVED_SOURCES}
    return sorted(i for i, src in enumerate(source) if src not in known)


def synthetic_variant_leaks(cols):
    """P2-08 derived rows (ADR-0004 A3): evaluation-only, never training-eligible.

    Strictly stronger than the external-positive predicate it replaces for these
    rows. Each derived row must:

    * be a real variant — ``parent_record_id != record_id`` (a self-parented
      "variant" would inherit nothing and silently become a base record);
    * be parented on a **held-out class-II corpus** row, so the eval set can never
      be built from a training-fold or blind-set parent;
    * itself be out of training (``not nested_train``, ``nested_role != "train"``).

    Fold-column inheritance is enforced separately by
    :func:`variant_parent_fold_mismatches`, which already covers every row whose
    parent differs from itself.
    """
    index = {rid: i for i, rid in enumerate(cols["record_id"])}
    viol = []
    for i, src in enumerate(cols["source"]):
        if src not in splits.DERIVED_SOURCES:
            continue
        rid, pid = cols["record_id"][i], cols["parent_record_id"][i]
        if pid is None or pid == rid:
            viol.append(("self_parented_variant", i))
            continue
        j = index.get(pid)
        if j is None:
            viol.append(("orphan_parent", i))
            continue
        if cols["source"][j] != "corpus":
            viol.append(("parent_not_corpus", i))
        elif cols["klass"][j] != "II":
            viol.append(("parent_not_class_ii", i))
        elif cols["nested_train"][j] or cols["nested_role"][j] == "train":
            viol.append(("parent_in_training_fold", i))
        elif cols["nested_train"][i] or cols["nested_role"][i] == "train":
            viol.append(("variant_in_training_fold", i))
    return viol


def external_cluster_training_twins(source, cluster_id, nested_train):
    """A corpus record that shares a homology cluster with an external held-out positive
    must not be in training (else a held-out anchor has a training twin).

    Scoped to :data:`splits.EXTERNAL_POSITIVE_SOURCES` (ADR-0004 A3): a derived
    variant shares its parent's cluster *by construction*, so counting derived rows
    here would re-flag the parent's own cluster as if a new independent positive had
    landed in it.
    """
    ext_clusters = {
        c for s, c in zip(source, cluster_id, strict=True) if s in splits.EXTERNAL_POSITIVE_SOURCES
    }
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
        "synthetic": synthetic_variant_leaks(c),
        "unknown_source": unknown_sources(c["source"]),
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


def _with_synthetic_variant(**over):
    """``_clean_cols()`` plus one P2-08 derived row parented on the held-out r2.

    The real committed table contains **no** violating row, so every
    :func:`synthetic_variant_leaks` clause is unreachable against it — disabling a
    clause there changes nothing and the gate stays green. (Established by
    sabotage at P2-08, not by reading: three clauses were confirmed non-biting on
    the real table.) These hand-built leaky fixtures are where the clauses are
    actually exercised.
    """
    c = {k: list(v) for k, v in _clean_cols().items()}
    row = {
        "record_id": "r2#c2phase7",
        "parent_record_id": "r2",
        "corpus_record_sha256": None,
        "source": splits.SYNTHETIC_CLASSII_SOURCE,
        "klass": "II",
        "cluster_id": 1,
        "resolved_phylum": "Actinobacteria",
        "resolved_class": "Actinobacteria",
        "resolved_order": "Frankiales",
        "resolved_genus": "Frankia",
        "fold_random": "val",
        "loo_order_unit": "Frankiales",
        "class_holdout_unit": "Actinobacteria",
        "phylum_holdout_unit": "Actinobacteria",
        "nested_train": False,
        "nested_role": "heldout",
        "is_designated_loo_holdout": True,
        "is_anchor_heldout": False,
        "clade_crossing_cluster": False,
        "dropped_from_clade_holdout": False,
    }
    row.update(over)
    for k in c:
        c[k].append(row[k])
    # The parent r2 is class II for this fixture's purposes.
    c["klass"][2] = "II"
    return c


def test_a_wellformed_synthetic_variant_is_clean():
    """Anchors the leaky cases below — without it they could pass for any reason."""
    c = _with_synthetic_variant()
    assert synthetic_variant_leaks(c) == []
    assert variant_parent_fold_mismatches(c) == []
    # ...and it is NOT judged by the external-positive predicate (ADR-0004 A3): it
    # inherits fold_random="val", which that predicate requires to be None.
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


def test_the_old_predicate_scope_would_have_failed_a_correct_variant():
    """Why A3's re-scope was necessary, pinned as a test rather than as prose.

    Under the pre-A3 ``src != "corpus"`` form, a D7-conforming variant — which
    inherits its parent's non-null ``fold_random`` — was a violation by
    construction. The two pinned contracts were in direct conflict.
    """
    c = _with_synthetic_variant()
    old_form = [
        i
        for i, src in enumerate(c["source"])
        if src != "corpus"
        and (
            not c["is_anchor_heldout"][i]
            or c["nested_train"][i]
            or c["nested_role"][i] == "train"
            or c["fold_random"][i] is not None
        )
    ]
    assert old_form == [6], "the correct variant would have been flagged by the old scope"


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"nested_train": True}, "variant_in_training_fold"),
        ({"nested_role": "train"}, "variant_in_training_fold"),
        ({"parent_record_id": "r2#c2phase7"}, "self_parented_variant"),
        ({"parent_record_id": "nope"}, "orphan_parent"),
    ],
)
def test_a_leaky_synthetic_variant_is_caught(override, reason):
    viol = synthetic_variant_leaks(_with_synthetic_variant(**override))
    assert viol and viol[0][0] == reason


def test_a_variant_parented_on_a_training_fold_row_is_caught():
    """ADR-0004 A3 relaxed *which* held-out parents are allowed, not the training ban.

    The parent is forced to ``klass == "II"`` so this reaches the
    ``parent_in_training_fold`` clause **in its own name**. Without that, r0 is
    class I, the earlier ``parent_not_class_ii`` branch of the elif chain fires
    first, and this test passes while the clause it is named for never executes —
    confirmed by sabotage (disabling the training-fold clause left it green).
    """
    c = _with_synthetic_variant(parent_record_id="r0", cluster_id=0, fold_random="train")
    c["klass"][0] = "II"
    viol = synthetic_variant_leaks(c)
    assert viol == [("parent_in_training_fold", 6)]


def test_a_variant_parented_on_a_nested_role_train_row_is_caught():
    """The second disjunct of the same clause (``nested_role == "train"``)."""
    c = _with_synthetic_variant(parent_record_id="r0", cluster_id=0, fold_random="train")
    c["klass"][0] = "II"
    c["nested_train"][0] = False  # only the role marks it as training
    viol = synthetic_variant_leaks(c)
    assert viol == [("parent_in_training_fold", 6)]


def test_a_variant_parented_on_the_blind_set_is_caught():
    """The 18-record natural arm must not become the parent of the synthetic arm."""
    c = _with_synthetic_variant(parent_record_id="anchor:a5", cluster_id=3, fold_random=None)
    viol = synthetic_variant_leaks(c)
    assert viol and viol[0][0] == "parent_not_corpus"


def test_a_variant_that_does_not_inherit_its_parents_fold_is_caught():
    """D7 inheritance, still enforced for derived rows (A3 did not relax it)."""
    c = _with_synthetic_variant(fold_random="test")
    assert variant_parent_fold_mismatches(c) == [("fold_random", 6)]


def test_no_unknown_source_values(committed):
    """The ``source`` vocabulary is closed.

    Load-bearing because :func:`external_positive_leaks` and
    :func:`synthetic_variant_leaks` are both keyed on named source allowlists
    (ADR-0004 A3): an unrecognised value would fall through *every* predicate and
    be checked by nothing.
    """
    assert unknown_sources(committed["source"]) == []


def test_synthetic_recovery_variants_are_evaluation_only(committed):
    """P2-08 derived rows: held-out class-II parents, never training-eligible."""
    c = committed
    assert synthetic_variant_leaks(c) == []
    # The set is actually present — a regression that dropped it would otherwise
    # leave this whole predicate vacuously green (the P2-07 lesson: a clause over
    # absent evidence reads TRUE exactly when the evidence is missing).
    n_synthetic = sum(1 for s in c["source"] if s in splits.DERIVED_SOURCES)
    assert n_synthetic == _EXPECTED_SOURCE_COUNTS["synthetic_classII"]
    assert n_synthetic > 0


def test_variant_parent_fold_inheritance(committed):
    assert variant_parent_fold_mismatches(committed) == []


# ========================================================================== #
# P2-10c′-f — the §9.2 held-out-order-negative gate (ADR-0004 A4; a1 + b2).
#
# The P0-24 predicates above read the committed split table, where the
# negatives live in **no row**: a mined/background negative window is injected
# at RUNTIME by ``data/negatives.py`` and the committed table holds only
# positives (+ their derived variants). So the leakage this section guards — a
# background window carved beside a *held-out-order* T-box entering training
# under a negative label — is invisible to them by construction
# ([[ci-leakage-gate-blind-to-runtime-augmentation]]).
#
# The signed resolution is **a1 + b2, loosening fallback OFF**:
#   a1 — admissibility keys on the window's PARENT corpus record's fold, which
#        inherits the leave-one-order-out holdout + the D3 clade-crossing
#        exclusion + the D4 dropped bucket (ADR-0004 D3/D4/D5), so it is at
#        least as strict as the positive side. For the current flank-carved
#        ``genomic_window`` substrate this equals a host-genome→order exclusion
#        *by the carving geometry* (every window's parent IS a catalogued
#        locus); that dependency is documented, not assumed away — a future
#        whole-genome background pool breaks the coincidence and would re-open
#        the a2 host-accession→order question.
#   b2 — make that runtime rule CI-VISIBLE here, over the real committed table,
#        so a loosened loader default OR a broken split→parent join turns the
#        gate RED instead of silently green-with-nothing-refused
#        ([[namespace-mismatch-invisible-noop]]).
# ========================================================================== #

#: Every clause the b2 gate re-derives (asserted complete, so a dropped/renamed
#: clause is a test failure, not a silent shrink).
_HELD_OUT_ORDER_NEGATIVE_CLAUSES = (
    "default_is_fail_closed",
    "out_of_fold_parents_refused",
    "join_resolves_every_parent",
    "join_finds_out_of_fold_parents",
    "loo_order_among_parents",
    "parents_are_a_spread",
    "loosening_admits_the_refused",
)

#: The shipped negative-loader entry points whose ``require_parent_nested_train``
#: default is the only thing standing between a held-out-order background window
#: and the training stream (Q-b: the committed table the P0-24 tier reads has no
#: negative rows). All four must default to True.
_PUBLIC_NEGATIVE_LOADERS = (
    "admit_pool_rows",
    "negative_records_from_rows",
    "load_admitted_pool_rows",
    "load_negative_records",
)


def held_out_order_negative_clauses(
    shipped_default_require,
    armed_report,
    loosened_report,
    stamp_summary,
    n_loo_order_parents,
):
    """The b2 §9.2 clauses, re-derived from the loader's own recorded evidence.

    Mirrors ``mining/pool.py::control_gate``: every clause is FALSE on a missing
    or degenerate measurement, never vacuously TRUE — a gate read off the
    *requested* configuration rather than the *found* evidence passes exactly
    when the evidence is absent ([[clauses-must-guard-emptiness]]).

    Args: the shipped loader default (``require_parent_nested_train``); the
    ``admit_pool_rows`` report with the rule ARMED (shipped default) and with it
    LOOSENED (``require_parent_nested_train=False``); the ``stamp_parent_folds``
    evidence block; and the count of fixture windows whose parent is a
    *designated* leave-one-order-out holdout order (the must-fire non-degeneracy
    witness — a real held-out order genuinely present among the parents).
    """
    n_records = int(stamp_summary.get("n_records", 0))
    n_out = int(stamp_summary.get("n_parent_out_of_training_fold", 0))
    n_unresolved = int(stamp_summary.get("n_unresolved_parent", -1))
    n_distinct = int(stamp_summary.get("n_distinct_parents", 0))
    refused_armed = int(armed_report.get("n_refused_parent_out_of_fold", 0))
    refused_loose = int(loosened_report.get("n_refused_parent_out_of_fold", -1))
    admitted_armed = int(armed_report.get("n_records", -1))
    admitted_loose = int(loosened_report.get("n_records", -1))
    return {
        # (i) the SHIPPED loader default is fail-closed — flipping it OFF is a
        # loosening, not a no-op, so the default is what the gate certifies.
        "default_is_fail_closed": (
            shipped_default_require is True
            and armed_report.get("require_parent_nested_train") is True
        ),
        # (ii) a substrate carrying out-of-fold (held-out-order) parents refuses
        # > 0 of them under that default.
        "out_of_fold_parents_refused": refused_armed > 0,
        # non-degeneracy companion (must-fire): the split→parent join is LIVE —
        # every window resolved a parent fold, so a namespace mismatch (which
        # would stamp *every* parent False and thus look like total
        # discrimination) cannot pass here ([[namespace-mismatch-invisible-noop]]).
        "join_resolves_every_parent": n_records > 0 and n_unresolved == 0,
        # … it actually finds out-of-fold parents (``parent_fold_discriminates``
        # shape; on its own this one is FOOLED by the namespace no-op, which is
        # exactly why it is paired with the two clauses around it).
        "join_finds_out_of_fold_parents": n_out > 0,
        # … a *designated LOO holdout order* is genuinely present among the
        # negative windows' parents — the exact material a1 governs.
        "loo_order_among_parents": n_loo_order_parents > 0,
        # … and the parents are a real spread, not one lucky record.
        "parents_are_a_spread": n_distinct > 1,
        # (iii) with the rule LOOSENED the same windows are admitted — proving
        # the armed refusal was the default's doing and nothing else's.
        "loosening_admits_the_refused": (
            refused_loose == 0 and admitted_loose == admitted_armed + refused_armed
        ),
    }


def _clean_held_out_inputs():
    """A consistent, all-clauses-TRUE evidence set: 10 in-fold + 5 held-out-order
    windows, join fully resolved. The bite tests below each perturb exactly one
    field and assert exactly the named clause flips FALSE (§8.7)."""
    stamp = {
        "n_records": 15,
        "n_parent_out_of_training_fold": 5,
        "n_unresolved_parent": 0,
        "n_distinct_parents": 15,
    }
    armed = {
        "require_parent_nested_train": True,
        "n_records": 10,
        "n_refused_parent_out_of_fold": 5,
        "n_refused_parent_unresolved": 0,
    }
    loosened = {
        "require_parent_nested_train": False,
        "n_records": 15,
        "n_refused_parent_out_of_fold": 0,
    }
    return {
        "shipped_default_require": True,
        "armed_report": armed,
        "loosened_report": loosened,
        "stamp_summary": stamp,
        "n_loo_order_parents": 5,
    }


def test_held_out_order_clean_inputs_pass():
    cl = held_out_order_negative_clauses(**_clean_held_out_inputs())
    assert set(cl) == set(_HELD_OUT_ORDER_NEGATIVE_CLAUSES)  # no clause dropped
    assert all(cl.values()), cl


def test_held_out_order_loosened_default_is_caught():
    """The banned loosening fallback (``require_parent_nested_train=False``)."""
    kw = _clean_held_out_inputs()
    kw["shipped_default_require"] = False
    kw["armed_report"] = {**kw["armed_report"], "require_parent_nested_train": False}
    cl = held_out_order_negative_clauses(**kw)
    assert [k for k, v in cl.items() if not v] == ["default_is_fail_closed"]


def test_held_out_order_namespace_noop_join_is_caught():
    """The join matched nothing — every parent stamped False (so ``n_out`` LOOKS
    like total discrimination), yet nothing was really refused. The companion
    (``join_resolves_every_parent`` + ``loo_order_among_parents``) catches it
    where a lone "found out-of-fold parents" clause would not."""
    kw = {
        "shipped_default_require": True,
        "armed_report": {
            "require_parent_nested_train": True,
            "n_records": 0,
            "n_refused_parent_out_of_fold": 0,
            "n_refused_parent_unresolved": 15,
        },
        "loosened_report": {"n_records": 15, "n_refused_parent_out_of_fold": 0},
        "stamp_summary": {
            "n_records": 15,
            "n_parent_out_of_training_fold": 15,  # the trap: all-False looks discriminating
            "n_unresolved_parent": 15,
            "n_distinct_parents": 15,
        },
        "n_loo_order_parents": 0,
    }
    cl = held_out_order_negative_clauses(**kw)
    assert cl["join_finds_out_of_fold_parents"] is True  # the fooled clause …
    assert cl["join_resolves_every_parent"] is False  # … caught by its companion
    assert cl["out_of_fold_parents_refused"] is False
    assert cl["loo_order_among_parents"] is False
    assert not all(cl.values())


def test_held_out_order_no_loo_witness_is_caught():
    """No *designated LOO holdout order* among the parents ⇒ the gate measured
    nothing about the material a1 governs, even if everything else is consistent."""
    kw = _clean_held_out_inputs()
    kw["n_loo_order_parents"] = 0
    cl = held_out_order_negative_clauses(**kw)
    assert [k for k, v in cl.items() if not v] == ["loo_order_among_parents"]


def test_held_out_order_loosening_that_still_refuses_is_caught():
    """If loosening the rule still refuses windows, the armed refusal was NOT the
    default's doing — the gate can no longer claim the default is what protects."""
    kw = _clean_held_out_inputs()
    kw["loosened_report"] = {
        **kw["loosened_report"],
        "n_records": 10,
        "n_refused_parent_out_of_fold": 5,
    }
    cl = held_out_order_negative_clauses(**kw)
    assert [k for k, v in cl.items() if not v] == ["loosening_admits_the_refused"]


def test_held_out_order_single_parent_is_caught():
    kw = _clean_held_out_inputs()
    kw["stamp_summary"] = {**kw["stamp_summary"], "n_distinct_parents": 1}
    cl = held_out_order_negative_clauses(**kw)
    assert [k for k, v in cl.items() if not v] == ["parents_are_a_spread"]


# ---- b2 real-loader tier: exercise the SHIPPED negative loader, not a re-impl ----
def _import_loader_or_fail_skip():
    """Import the shipped negative loader + mining pool, or FAIL under
    ``TBOX_REQUIRE_NO_LEAKAGE=1`` (never a silent skip — the runtime rule the
    committed-table tier cannot see must not go unchecked in CI)."""
    try:
        from tbox_finder.data import negatives
        from tbox_finder.mining import pool
    except ImportError as exc:  # pragma: no cover - only where numpy/pandas absent
        _fail_or_skip(f"negative loader not importable ({exc})")
    return negatives, pool


def test_shipped_negative_loader_defaults_are_fail_closed():
    """a1 + b2 (ADR-0004 A4): EVERY public negative-loader entry point defaults
    ``require_parent_nested_train=True``.

    A silent flip of any one default to False would re-open the held-out-order
    leak with every P0-24 assertion still green (the committed split table has
    no negative rows to catch it), so the default itself is asserted here — once
    per shipped entry point."""
    negatives, _pool = _import_loader_or_fail_skip()
    for name in _PUBLIC_NEGATIVE_LOADERS:
        fn = getattr(negatives, name)
        param = inspect.signature(fn).parameters.get("require_parent_nested_train")
        assert param is not None, f"{name} lost its require_parent_nested_train knob"
        assert param.default is True, (
            f"{name}(require_parent_nested_train=) defaults to {param.default!r}, not "
            "True — the §9.2 fail-closed default was silently loosened (ADR-0004 A4)"
        )


def _held_out_order_negative_fixture(committed, negatives, pool, *, n_in_fold=30, n_loo=20):
    """Synthesize a negative pool over REAL committed corpus parents and run it
    through the shipped ``stamp_parent_folds`` + ``admit_pool_rows`` path.

    ``fold_by_id`` is the corpus-only ``{record_id -> nested_train}`` projection
    ``mining/pool.py::load_parent_folds`` builds from this same table; the join
    under test is ``stamp_parent_folds``'s lookup on each window's
    ``source_record_id``. Selecting parents from the real table is what makes
    the non-degeneracy companion bite: fabricated ids would resolve to nothing.

    The two parent sets are DIFFERENT sizes (``n_in_fold`` ≠ ``n_loo``) and the
    admitted set is asserted downstream by IDENTITY, not count — so a fold-SENSE
    inversion (a one-token flip that refuses the in-fold windows and admits the
    held-out-order ones) changes both the counts and *which* windows survive and
    cannot slip through on a lucky symmetric count. The aggregate-report clauses
    alone are identity-blind here (stamp reports the true out-of-fold count while
    the inverted admit refuses the wrong windows), which is exactly why the
    identity assertion lives in the test, not the pure clause fn
    ([[degenerate-fixture-generators]]).
    """
    c = committed
    by_id, fold_by_id = {}, {}
    for rid, src, nt, loo, order in zip(
        c["record_id"],
        c["source"],
        c["nested_train"],
        c["is_designated_loo_holdout"],
        c["resolved_order"],
        strict=True,
    ):
        if src != pool.SPLIT_CORPUS_SOURCE:  # only corpus rows are valid parent keys
            continue
        by_id[str(rid)] = (bool(nt), bool(loo), order)
        fold_by_id[str(rid)] = bool(nt)
    in_fold = sorted(r for r, (nt, _l, _o) in by_id.items() if nt)[:n_in_fold]
    loo_parents = sorted(
        r for r, (nt, loo, order) in by_id.items() if (not nt) and loo and order is not None
    )[:n_loo]
    if not in_fold or not loo_parents:
        _fail_or_skip("committed table lacks both in-fold and designated-LOO corpus parents")
    rows = [
        {
            "candidate_id": f"neg_p2_10cf_{i}",
            "sequence": "A" * negatives.WINDOW_NT,
            "masked": False,
            "is_designed_control": False,
            "source_record_id": rid,
        }
        for i, rid in enumerate([*in_fold, *loo_parents])
    ]
    stamp_summary = pool.stamp_parent_folds(rows, fold_by_id)  # mutates rows in place
    n_loo_order_parents = sum(
        1 for row in rows if by_id[row["source_record_id"]][1] and by_id[row["source_record_id"]][2]
    )
    armed_rows, armed_report = negatives.admit_pool_rows(rows, window=negatives.WINDOW_NT)
    _loose_rows, loose_report = negatives.admit_pool_rows(
        rows, window=negatives.WINDOW_NT, require_parent_nested_train=False
    )
    return {
        "rows": rows,
        "stamp_summary": stamp_summary,
        "n_loo_order_parents": n_loo_order_parents,
        "armed_rows": armed_rows,
        "armed_report": armed_report,
        "loose_report": loose_report,
        "in_fold_ids": in_fold,
        "loo_ids": loo_parents,
    }


def test_held_out_order_negatives_refused_by_default(committed):
    """b2 (ADR-0004 A4): the SHIPPED loader default refuses a background window
    carved beside a held-out-order T-box — measured end-to-end over the real
    committed partition, asserted by IDENTITY (which windows, not just how many)
    so a fold-sense inversion cannot pass on symmetric counts, with the must-fire
    non-degeneracy companion so a broken split→parent join fails RED."""
    negatives, pool = _import_loader_or_fail_skip()
    fx = _held_out_order_negative_fixture(committed, negatives, pool)
    shipped_default = (
        inspect.signature(negatives.admit_pool_rows)
        .parameters["require_parent_nested_train"]
        .default
    )
    clauses = held_out_order_negative_clauses(
        shipped_default_require=shipped_default,
        armed_report=fx["armed_report"],
        loosened_report=fx["loose_report"],
        stamp_summary=fx["stamp_summary"],
        n_loo_order_parents=fx["n_loo_order_parents"],
    )
    assert all(clauses.values()), {k: v for k, v in clauses.items() if not v}

    in_fold, loo = set(fx["in_fold_ids"]), set(fx["loo_ids"])
    assert in_fold and loo and len(in_fold) != len(loo)  # asymmetric by construction
    # IDENTITY, not counts: the admitted windows' parents are EXACTLY the in-fold
    # set. A fold-sense inversion would admit the held-out-order parents instead —
    # which a symmetric-count fixture could not distinguish, but this can.
    admitted_parents = {r["source_record_id"] for r in fx["armed_rows"]}
    assert admitted_parents == in_fold
    assert admitted_parents.isdisjoint(loo)
    # Concrete counts (now asymmetric, so a swap also moves these):
    assert fx["armed_report"]["n_refused_parent_out_of_fold"] == len(loo)
    assert fx["armed_report"]["n_refused_parent_unresolved"] == 0
    assert fx["armed_report"]["n_records"] == len(in_fold)
    assert fx["loose_report"]["n_records"] == len(in_fold) + len(loo)
    assert fx["n_loo_order_parents"] == len(loo) > 0

    # A DELEGATING loader (not just admit_pool_rows) must honor the default at
    # RUNTIME: an internal delegation that silently passed
    # require_parent_nested_train=False would keep its own signature default True
    # yet leak here, so the signature tier alone under-checks it.
    _recs, deleg_report = negatives.negative_records_from_rows(
        fx["rows"], window=negatives.WINDOW_NT
    )
    assert deleg_report["n_records"] == len(in_fold)
    assert deleg_report["n_refused_parent_out_of_fold"] == len(loo)
