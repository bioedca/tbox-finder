"""Unit tests for P0-22 split construction (src/tbox_finder/splits.py).

Tiers:
- pure-logic (runs bare, no numpy/pandas): single-linkage, held-out-order
  designation, whole-cluster genus-stratified fold assignment, taxon spread;
- numpy tier (``importorskip numpy``): consensus-matrix extraction from a cmalign
  afa + the vectorised structure-aware clustering vs the reference union-find;
- pandas tier (``importorskip pandas``): the nested split ladder — whole clusters
  to a single fold, the D3 clade-crossing forced rule, no cluster straddling the
  train/heldout boundary, and the fail-loud split guard.

These lock the leakage-control invariants the headline generalization claim rests
on (ADR-0004 D2/D3/D5; CLAUDE.md §8.2) without needing cmalign or the real corpus.
"""

from __future__ import annotations

import pytest

from tbox_finder import splits


# --------------------------------------------------------------------------- #
# pure-logic tier
# --------------------------------------------------------------------------- #
def test_single_linkage_transitive_closure():
    # 0-1-2 chain (one component), 3-4 (a second), 5 isolated.
    labels = splits.single_linkage(6, [(0, 1), (1, 2), (3, 4)])
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3] != labels[5]
    assert len(set(labels)) == 3


def test_single_linkage_empty_edges_all_singletons():
    labels = splits.single_linkage(4, [])
    assert sorted(labels) == [0, 1, 2, 3]


def test_designate_heldout_orders_reserves_largest_to_floor():
    counts = {"Big": 7, "Small": 3, "Tiny": 1}
    heldout, reserved = splits.designate_heldout_orders(
        counts, min_positives=2, train_coverage_floor=0.60
    )
    # Big alone is 70% ≥ floor → reserved; Small (≥min) held out; Tiny (<min) neither.
    assert reserved == ["Big"]
    assert heldout == ["Small"]


def test_designate_heldout_orders_min_positives_excludes_small_orders():
    counts = {"A": 50, "B": 50, "C": 5}
    heldout, reserved = splits.designate_heldout_orders(
        counts, min_positives=20, train_coverage_floor=0.60
    )
    # reserve A (50/105=0.48<0.60) then B (100/105≥0.60) → reserved {A,B}; C<min → neither.
    assert set(reserved) == {"A", "B"}
    assert heldout == []


def test_assign_random_folds_whole_cluster_and_deterministic():
    records = {i: 10 for i in range(30)}
    genus = {i: f"g{i % 3}" for i in range(30)}
    a = splits.assign_random_folds(records, genus, seed=42)
    b = splits.assign_random_folds(records, genus, seed=42)
    assert a == b  # deterministic in the seed
    assert set(a.values()) <= {"train", "val", "test"}
    assert len(a) == 30  # every cluster placed in exactly one fold
    # train is the majority target (0.80) → gets the most records.
    from collections import Counter

    tally = Counter(a.values())
    assert tally["train"] >= tally["val"]
    assert tally["train"] >= tally["test"]


def test_cluster_taxon_spread_ignores_empty():
    spread = splits.cluster_taxon_spread([0, 0, 1, 1], ["X", "", "Y", "Z"])
    assert spread[0] == {"X"}
    assert spread[1] == {"Y", "Z"}


# --------------------------------------------------------------------------- #
# numpy tier
# --------------------------------------------------------------------------- #
def test_consensus_matrix_from_afa(tmp_path):
    pytest.importorskip("numpy")
    # 3 aligned rows: col layout = [match, match, insert(lowercase), match].
    afa = tmp_path / "x.afa"
    afa.write_text(">s0\nACgU\n>s1\nACaU\n>s2\nA-cU\n")
    names, codes = splits.consensus_matrix(afa)
    assert names == ["s0", "s1", "s2"]
    # insert column (index 2, lowercase) dropped → 3 consensus columns.
    assert codes.shape == (3, 3)
    # A,C,U → 0,1,3 ; the '-' in s2 col1 → -1 (missing).
    assert codes[0].tolist() == [0, 1, 3]
    assert codes[2].tolist() == [0, -1, 3]


def test_consensus_matrix_from_stockholm(tmp_path):
    pytest.importorskip("numpy")
    # Pfam/Stockholm block (the >10k class-I format): #= lines + // are skipped,
    # the #=GC RF line marks the insert column (col 2) with '.'.
    sto = tmp_path / "x.sto"
    sto.write_text(
        "# STOCKHOLM 1.0\n"
        "s0        ACgU\n"
        "s1        ACaU\n"
        "s2        A-cU\n"
        "#=GC RF   AC.U\n"
        "//\n"
    )
    names, codes = splits.consensus_matrix(sto)
    assert names == ["s0", "s1", "s2"]
    assert codes.shape == (3, 3)
    assert codes[0].tolist() == [0, 1, 3]
    assert codes[2].tolist() == [0, -1, 3]


def test_vectorized_clustering_two_disjoint_groups():
    np = pytest.importorskip("numpy")
    # seqs 0,1 identical; 2,3 identical; the two groups share 0 columns → id 0.
    codes = np.array(
        [
            [0, 1, 2, 3, 0, 1],
            [0, 1, 2, 3, 0, 1],
            [3, 2, 1, 0, 3, 2],
            [3, 2, 1, 0, 3, 2],
        ],
        dtype=np.int8,
    )
    labels = splits._cluster_consensus_matrix(codes, (0.70,), splits.COVERAGE_CUT)[0.70]
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert len(set(labels.tolist())) == 2


def test_vectorized_clustering_matches_reference_union_find():
    np = pytest.importorskip("numpy")
    # A near-identical chain 0≈1≈2 (each differs by 1 nt) plus a distant seq 3.
    codes = np.array(
        [
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 2],  # 90% id to 0
            [0, 1, 2, 3, 0, 1, 2, 3, 1, 2],  # 90% id to 1, 80% to 0
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # distant
        ],
        dtype=np.int8,
    )
    labels = splits._cluster_consensus_matrix(codes, (0.80,), splits.COVERAGE_CUT)[0.80]
    # 0-1 (0.90), 1-2 (0.90), 0-2 (0.80) all pass ≥0.80 → single-linkage merges 0,1,2.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_low_model_coverage_seq_is_forced_singleton():
    np = pytest.importorskip("numpy")
    # ADR-0004 D2 Amendment A1: coverage is vs the RF00230 model consensus (clen),
    # not the shorter member's span — so a sequence occupying only 1 of 10 model
    # columns can form no edge (co/clen ≤ 0.1 < 0.70) even if that 1 column matches
    # everyone. Under the old shorter-member coverage it would hub-bridge them all.
    codes = np.array(
        [
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],  # full, group A
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],  # full, identical to 0
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 1-column "hub", matches col 0
        ],
        dtype=np.int8,
    )
    labels = splits._cluster_consensus_matrix(codes, (0.70,), splits.COVERAGE_CUT)[0.70]
    assert labels[0] == labels[1]  # the two full sequences cluster
    assert labels[2] != labels[0]  # the 1-column sequence is a forced singleton
    assert len(set(labels.tolist())) == 2


# --------------------------------------------------------------------------- #
# pandas tier — the nested split ladder + leakage invariants
# --------------------------------------------------------------------------- #
def _toy_manifest():
    import pandas as pd

    rows = []

    def add(seq_name, source, klass, phylum, order, genus, cluster):
        rows.append(
            {
                "seq_name": seq_name,
                "source": source,
                "klass": klass,
                "record_sha256": seq_name if source == "corpus" else "",
                "resolved_phylum": phylum,
                "resolved_class": None,
                "resolved_order": order,
                "resolved_genus": genus,
                "dropped_from_clade_holdout": False,
            }
        )

    clusters = []
    # cluster 2: 4 pure-Big members → nested train.
    for i in range(4):
        add(f"big_train_{i}", "corpus", "I", "Firmicutes", "Big", "gA", 2)
        clusters.append(2)
    # cluster 0: 3 Big + 1 Small → clade-crossing (whole cluster held out).
    for i in range(3):
        add(f"big_cross_{i}", "corpus", "I", "Firmicutes", "Big", "gB", 0)
        clusters.append(0)
    add("small_cross", "corpus", "I", "Firmicutes", "Small", "gC", 0)
    clusters.append(0)
    # cluster 1: 2 Small → held out (leave-one-order-out unit).
    for i in range(2):
        add(f"small_{i}", "corpus", "I", "Firmicutes", "Small", "gC", 1)
        clusters.append(1)
    # cluster 3: an Actinobacteria phylum-holdout member.
    add("actino", "corpus", "II", "Actinobacteria", "Micrococcales", "gD", 3)
    clusters.append(3)
    # cluster 4: an independent anchor (always held out; scheme C).
    add("anchor:x", "anchor", "I", "Chloroflexota", None, None, 4)
    clusters.append(4)
    return pd.DataFrame(rows), clusters


def test_build_ladder_whole_cluster_and_clade_crossing():
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    manifest, clusters = _toy_manifest()
    table, diag = splits._build_ladder(
        manifest, clusters, seed=42, min_positives=2, train_coverage_floor=0.60
    )
    by_name = {r.seq_name: r for r in table.itertuples()}

    # D5 designation: Big reserved for training, Small a designated holdout.
    assert "Small" in diag["designated_heldout_orders"]
    assert "Big" in diag["train_reserved_orders"]

    # cluster 2 (pure Big) → all nested train.
    assert all(by_name[f"big_train_{i}"].nested_role == "train" for i in range(4))

    # cluster 0 (clade-crossing): Big members excluded (not train, not scored),
    # the Small member held out — so the cluster never straddles.
    assert all(by_name[f"big_cross_{i}"].nested_role == "excluded_clade_crossing" for i in range(3))
    assert by_name["small_cross"].nested_role == "heldout"
    assert all(by_name[f"big_cross_{i}"].clade_crossing_cluster for i in range(3))
    assert not by_name["big_train_0"].clade_crossing_cluster

    # Actinobacteria phylum-holdout + anchor are held out; anchor keeps no fold_random
    # (externals are not in scheme A → missing, stored as pandas NA).
    import pandas as pd

    assert by_name["actino"].nested_role == "heldout"
    assert by_name["anchor:x"].is_anchor_heldout
    assert pd.isna(by_name["anchor:x"].fold_random)
    assert by_name["big_train_0"].fold_random in {"train", "val", "test"}

    # The load-bearing invariant: no cluster straddles nested train/heldout.
    splits._assert_no_cluster_split(table)
    assert diag["clade_crossing"]["order"]["n_crossing_clusters"] >= 1


def test_assert_no_cluster_split_fails_loud_on_straddle():
    pd = pytest.importorskip("pandas")

    bad = pd.DataFrame(
        {
            "cluster_id": [7, 7],
            "nested_role": ["train", "heldout"],  # same cluster on both sides
        }
    )
    with pytest.raises(ValueError, match="straddle"):
        splits._assert_no_cluster_split(bad)
