"""P3-16′-remint — unit tests for the benchmark minter's ADR-0005 A15 changes.

``scripts/mint_p3_16_benchmark.py`` is tested at the PRODUCER, on small real sequences: the
host and insert DNA below is cut from ``tests/fixtures/two_stage/contigs.json`` (20 real
1,536-nt slabs of ``context_v0``, committed at P3-14). The decoy *rows* that carry it are
synthetic — a pool label and an id — because what is under test is the mint's bookkeeping
(the gated subset, the draw order, the screen filter, the refusals), not decoy content. The
45k-row mining pool is never read here.

What each block guards, and why a bare artifact check would not do
([[artifact-pinning-test-cannot-see-the-code]]):

* the gated flag and the ``*_gated`` scope counts (A15 Pin 4);
* the leader stays in the draw at its ``decoy_id``-sorted slot (A15 Pins 3–4);
* the screen filter runs BEFORE the seeded permutation (A15 Pin 3);
* a ``--host-screen`` report is refused unless it binds to this mint's pool and its controls
  fired (A15 Pins 1–2);
* the host-overlap measure skips non-ACGT k-mers (A15 Pin 1 alphabet rule);
* no v2 output is written to a v1 path (A15 Pin 5).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from tbox_finder.data import embedding as emb
from tbox_finder.integration import stage1_folds
from tbox_finder.stage2.cm_confirmer import CANONICAL_SOURCES

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mint_p3_16_benchmark.py"
SLABS = REPO / "tests" / "fixtures" / "two_stage" / "contigs.json"
WINDOW = 1024
SEED = 42
POOL_SHA = "e" * 64


@pytest.fixture(scope="module")
def mint():
    """The mint script, loaded by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("mint_p3_16_benchmark_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def slabs() -> list[dict]:
    return json.loads(SLABS.read_text(encoding="utf-8"))["contigs"]


@pytest.fixture(scope="module")
def hosts(slabs) -> list[dict]:
    """20 real 1,024-nt host windows, one per slab, shaped like admitted mining-pool rows."""
    rows = []
    for index, slab in enumerate(slabs):
        rows.append(
            {
                "candidate_id": f"host{index:02d}:{'lead' if index % 2 == 0 else 'trail'}",
                "sequence": slab["sequence"].upper()[:WINDOW],
                "side": "lead" if index % 2 == 0 else "trail",
                "source_record_id": slab["truth_source_record_id"],
                "parent_nested_train": False,
            }
        )
    return rows


#: Eight decoys over the four §9.1 pools. ``dec_d`` is the leader and sorts mid-list, so
#: three decoys draw AFTER it — the ones a dropped leader would re-index.
_DECOY_SPEC = (
    ("dec_a", "gc_background", 0, 1100, 90),
    ("dec_b", "structured_rna", 1, 1150, 72),
    ("dec_c", "dinuc_shuffled", 2, 1200, 120),
    ("dec_d", "leader_decoy", 3, 1100, 180),
    ("dec_e", "structured_rna", 4, 1250, 85),
    ("dec_f", "gc_background", 5, 1100, 95),
    ("dec_g", "dinuc_shuffled", 6, 1300, 110),
    ("dec_h", "structured_rna", 7, 1180, 77),
)
LEADER = "dec_d"


@pytest.fixture(scope="module")
def decoy_frame(slabs) -> pd.DataFrame:
    """The decoy rows, deliberately NOT in ``decoy_id`` order (the mint must sort them)."""
    rows = []
    for decoy_id, pool, slab_index, start, length in _DECOY_SPEC:
        slab = slabs[slab_index]
        rows.append(
            {
                "decoy_id": decoy_id,
                "pool": pool,
                "sequence": slab["sequence"].upper()[start : start + length],
                "masked": False,
                "source_record_id": (
                    slab["truth_source_record_id"] if pool == "dinuc_shuffled" else None
                ),
            }
        )
    return pd.DataFrame(rows[::-1])


@pytest.fixture(scope="module")
def corpus(slabs) -> tuple[frozenset[str], dict[str, int]]:
    ids = sorted({slab["truth_source_record_id"] for slab in slabs})
    return frozenset(ids), {record: 1000 + index for index, record in enumerate(ids)}


def _negatives(mint, decoy_frame, hosts, corpus, *, admitted_ids=None):
    all_ids, clusters = corpus
    rows = mint.select_decoy_rows(decoy_frame, set(decoy_frame["decoy_id"]))
    negatives, drawable = mint.mint_negative_windows(
        rows,
        hosts,
        admitted_ids=admitted_ids,
        seed=SEED,
        all_corpus_ids=all_ids,
        parent_cluster=clusters,
        window=WINDOW,
    )
    return rows, negatives, drawable


def _positives(slabs):
    contigs, items = [], []
    for index, slab in enumerate(slabs[:3]):
        contig_id = f"pos:{slab['truth_source_record_id']}-{index}"
        contigs.append({"contig_id": contig_id, "sequence": slab["sequence"][:WINDOW]})
        items.append(
            {
                "contig_id": contig_id,
                "label": 1,
                "source": "corpus",
                "pool": "corpus",
                "block": f"cluster:{index}",
                "length": WINDOW,
                "host_id": None,
                "host_seen_by_any_arm": None,
            }
        )
    return contigs, items


def _built(mint, slabs, decoy_frame, hosts, corpus):
    _, negatives, _ = _negatives(mint, decoy_frame, hosts, corpus)
    folds = {"production": set(), "twin": set()}
    return mint.build(
        copy.deepcopy(_positives(slabs)), copy.deepcopy(negatives), decoy_frame, folds, 3
    )


# ──────────────────────────────────────────────────────────────────────────────
# Promote, don't duplicate (A15 Pin 1)
# ──────────────────────────────────────────────────────────────────────────────
def test_a_screen_over_different_screened_dna_is_refused(mint):
    """The pool digest and the candidate count do not cover the screened DNA.

    Changing ``decoys_v0`` or ``stage2_dataset`` leaves both intact while altering which
    windows Pin 1 refuses, so the admitted list would be certified against a different union
    than the one that produced it ([[gate-must-bind-to-upstream-evidence]]).
    """
    report = {
        "problems": [],
        "controls_all_fired": True,
        "controls": [{"name": "c", "fired": True}],
        "content_screen": {"threshold_bits": 10.0},
        "pool": {"sha256": "p" * 64, "n_candidates": 7},
        "admitted": {"sha256": "a" * 64, "list": "reports/p3/host_screen_admitted.txt", "n": 5},
        "screened_dna": {
            "sources": {
                "decoy_inserts": {"sha256": "d" * 64},
                "stage2_population": {"stage2_dataset_sha256": "s" * 64},
            }
        },
    }
    base = dict(mining_pool_sha256="p" * 64, n_candidates=7, admitted_list_sha256="a" * 64)
    matching = {
        ("decoy_inserts", "sha256", "d" * 64),
        ("stage2_population", "stage2_dataset_sha256", "s" * 64),
    }
    # Positive control: the matching set must be accepted.
    assert mint.host_screen_refusals(report, **base, screened_dna_sha256=matching) == []
    for label, supplied in (
        ("a changed decoys_v0", {("decoy_inserts", "sha256", "0" * 64)}),
        (
            "a changed stage2_dataset",
            {("stage2_population", "stage2_dataset_sha256", "0" * 64)},
        ),
    ):
        problems = mint.host_screen_refusals(report, **base, screened_dna_sha256=supplied)
        assert any("different screened DNA" in p for p in problems), (label, problems)
    # A digest the screen never recorded is a refusal, not a skip: the absent case is the one
    # that reads as agreement ([[clauses-must-guard-emptiness]]).
    problems = mint.host_screen_refusals(
        report, **base, screened_dna_sha256={("fold_contexts", "sha256", "0" * 64)}
    )
    assert any("records no screened_dna.sources" in p for p in problems), problems


def test_the_fold_definition_is_the_promoted_one_not_a_local_copy(mint):
    """The screen and the mint must read ONE fold definition. A local copy that agrees today
    makes "the screen ran on the mint's folds" a coincidence
    ([[promote-dont-duplicate-is-a-correctness-rule]])."""
    assert mint.training_fold_ids is stage1_folds.training_fold_ids
    assert mint.ARM_EXCLUDE_GATE4_EVAL is stage1_folds.ARM_EXCLUDE_GATE4_EVAL


def test_the_manifest_schema_moves_with_the_new_fields(mint):
    assert mint.SCHEMA_VERSION == "1.1"


# ──────────────────────────────────────────────────────────────────────────────
# The gated subset (A15 Pin 4)
# ──────────────────────────────────────────────────────────────────────────────
def test_only_the_leader_pool_is_not_gated_and_the_scope_declares_the_subset(
    mint, slabs, decoy_frame, hosts, corpus
):
    contigs, items, scope = _built(mint, slabs, decoy_frame, hosts, corpus)
    assert all(isinstance(item["gated"], bool) for item in items)
    ungated = [item["contig_id"] for item in items if not item["gated"]]
    assert ungated == [f"dec:{LEADER}"]
    # positives are gated too — "gated" is not "is a negative"
    assert all(item["gated"] for item in items if item["label"] == 1)

    n_leader = 1
    assert scope["n_items"] == len(items)
    assert scope["n_items_gated"] == len(items) - n_leader
    assert scope["n_negatives_gated"] == scope["n_negatives"] - n_leader
    assert scope["negatives_by_pool_gated"] == {
        pool: count for pool, count in scope["negatives_by_pool"].items() if pool != "leader_decoy"
    }
    # the leader blocks on its unique host, so exactly one block leaves the gated set
    assert scope["n_blocks_gated"] == len(
        {i["block"] for i in items if i["pool"] != "leader_decoy"}
    )
    assert scope["n_blocks_gated"] == scope["n_blocks"] - 1
    assert scope["n_positive_blocks"] == 3
    assert "ADR-0005 A15 Pin 4" in scope["gated_rule"]


def test_the_gated_counts_are_read_off_the_flags_not_the_pool_names(mint):
    """A flag that disagrees with the rule must surface in the counts, not be papered over
    by a scope computed from pool names."""
    items = [
        {"contig_id": "pos:a", "label": 1, "pool": "corpus", "block": "cluster:1", "gated": True},
        {"contig_id": "dec:b", "label": 0, "pool": "gc_background", "block": "host:b"},
        {"contig_id": "dec:c", "label": 0, "pool": "leader_decoy", "block": "host:c"},
    ]
    items[1]["gated"] = False  # deliberately inconsistent with item_gated
    items[2]["gated"] = True
    scope = mint.gated_scope(items)
    assert scope["negatives_by_pool_gated"] == {"leader_decoy": 1}
    assert scope["n_items_gated"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# The leader stays in the draw at its decoy_id-sorted position (A15 Pins 3–4)
# ──────────────────────────────────────────────────────────────────────────────
def test_the_leader_keeps_its_sorted_slot_in_the_draw(mint, decoy_frame, hosts, corpus):
    rows, (_, items, report), drawable = _negatives(mint, decoy_frame, hosts, corpus)
    order = [row["decoy_id"] for row in rows]
    assert order == sorted(order), "the draw order is decoy_id order"
    assert LEADER in order, "the leader is minted, not filtered out of the draw"
    leader_index = order.index(LEADER)
    assert 0 < leader_index < len(order) - 1, "fixture: decoys draw on both sides of the leader"

    # The i-th decoy in decoy_id order takes host permutation[i] — the leader included.
    permutation = emb._host_permutation(len(drawable), SEED)
    expected = [drawable[int(permutation[i])]["candidate_id"] for i in range(len(order))]
    assert [item["host_id"] for item in items] == expected
    assert [item["contig_id"] for item in items] == [f"dec:{d}" for d in order]
    assert report["n_embedded_by_pool"]["leader_decoy"] == 1


def test_gating_changes_no_host_assignment_relative_to_the_v1_draw(
    mint, slabs, decoy_frame, hosts, corpus
):
    """With the leader out of the GATED denominator, the draw is still v1's: every item's host
    and phase equal what the shipped operator assigns with the leader present."""
    all_ids, _ = corpus
    rows = mint.select_decoy_rows(decoy_frame, set(decoy_frame["decoy_id"]))
    v1_windows, _ = emb.embed_decoy_rows(
        rows,
        hosts,
        seed=SEED,
        window=WINDOW,
        pools=mint.BENCHMARK_DECOY_POOLS,
        unique_hosts=True,
        training_fold_record_ids=all_ids,
    )
    v1 = {f"dec:{w.insert_id}": (w.host_id, w.phase) for w in v1_windows}
    _, items, _ = _built(mint, slabs, decoy_frame, hosts, corpus)
    minted = {i["contig_id"]: (i["host_id"], i["splice_phase"]) for i in items if i["label"] == 0}
    assert minted == v1

    # …and dropping the leader from the DRAW (the tempting "fix") would re-index every decoy
    # after it, which is why A15 keeps it there.
    without_leader, _ = emb.embed_decoy_rows(
        rows,
        hosts,
        seed=SEED,
        window=WINDOW,
        pools=tuple(p for p in mint.BENCHMARK_DECOY_POOLS if p != "leader_decoy"),
        unique_hosts=True,
        training_fold_record_ids=all_ids,
    )
    rehosted = {f"dec:{w.insert_id}": w.host_id for w in without_leader}
    after = sorted(d for d in v1 if d > f"dec:{LEADER}")
    assert after, "fixture: decoys draw after the leader"
    assert all(rehosted[d] != v1[d][0] for d in after)


# ──────────────────────────────────────────────────────────────────────────────
# The screen filter runs BEFORE the permutation (A15 Pin 3)
# ──────────────────────────────────────────────────────────────────────────────
def test_a_host_the_screen_did_not_admit_is_never_drawn(mint, decoy_frame, hosts, corpus):
    _, (_, unscreened, _), _ = _negatives(mint, decoy_frame, hosts, corpus)
    drawn = [item["host_id"] for item in unscreened]
    # Refuse two hosts the UNSCREENED draw uses, so the control can fire.
    refused = {drawn[0], drawn[-1]}
    admitted = sorted(row["candidate_id"] for row in hosts if row["candidate_id"] not in refused)

    _, (_, screened, report), drawable = _negatives(
        mint, decoy_frame, hosts, corpus, admitted_ids=admitted
    )
    assert not refused & {item["host_id"] for item in screened}
    assert {item["host_id"] for item in screened} <= set(admitted)
    # The permutation ran over the screened pool, in pool order.
    assert report["n_hosts_available"] == len(admitted)
    assert [row["candidate_id"] for row in drawable] == [
        row["candidate_id"] for row in hosts if row["candidate_id"] in set(admitted)
    ]
    permutation = emb._host_permutation(len(drawable), SEED)
    assert [item["host_id"] for item in screened] == [
        drawable[int(permutation[i])]["candidate_id"] for i in range(len(screened))
    ]


def test_the_screen_filter_keeps_pool_order_not_the_lists_sorted_order(mint, hosts):
    admitted = [row["candidate_id"] for row in hosts][::-1][:7]
    kept = mint.apply_host_screen(hosts, admitted)
    assert [row["candidate_id"] for row in kept] == [
        row["candidate_id"] for row in hosts if row["candidate_id"] in set(admitted)
    ]


def test_an_admitted_id_outside_the_out_of_fold_pool_is_refused(mint, hosts):
    good = [row["candidate_id"] for row in hosts[:10]]
    assert len(mint.apply_host_screen(hosts, good)) == 10  # positive control
    with pytest.raises(SystemExit, match="not in the out-of-fold candidate pool"):
        mint.apply_host_screen(hosts, [*good, "a-window-the-pool-never-carried:lead"])


def test_a_screen_that_leaves_fewer_hosts_than_decoys_is_refused(mint, decoy_frame, hosts, corpus):
    with pytest.raises(SystemExit, match="drawable host windows"):
        _negatives(
            mint,
            decoy_frame,
            hosts,
            corpus,
            admitted_ids=[row["candidate_id"] for row in hosts[:3]],
        )


# ──────────────────────────────────────────────────────────────────────────────
# --host-screen refusals (A15 Pins 1–2)
# ──────────────────────────────────────────────────────────────────────────────
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_screen(admitted: list[str]) -> tuple[dict, str]:
    listing = "".join(f"{cid}\n" for cid in sorted(admitted))
    report = {
        "schema_version": "1.0",
        "step": "P3-16′-remint",
        "adr": "ADR-0005 A15",
        "pool": {
            "source": "data/processed/negatives/mining_pool_v0.parquet",
            "sha256": POOL_SHA,
            "n_admitted": 40,
            "n_candidates": 20,
            "selection": "parent_nested_train == False",
        },
        "content_screen": {"threshold_bits": 10.0, "motif": "GGTGGNACCGCG"},
        "admitted": {
            "n": len(admitted),
            "list": "reports/p3/host_screen_admitted.txt",
            "sha256": _sha(listing),
        },
        "controls": [
            {"name": "kmer_fold_contexts", "mechanism": "kmer", "fired": True},
            {"name": "coordinate_v1_host", "mechanism": "coordinate", "fired": True},
        ],
        "controls_all_fired": True,
        "problems": [],
    }
    return report, listing


def _write_screen(root: Path, report: dict | str, listing: str | None) -> str:
    (root / "reports" / "p3").mkdir(parents=True, exist_ok=True)
    body = report if isinstance(report, str) else json.dumps(report)
    (root / "reports" / "p3" / "host_screen.json").write_text(body, encoding="utf-8")
    if listing is not None:
        (root / "reports" / "p3" / "host_screen_admitted.txt").write_text(listing, encoding="utf-8")
    return "reports/p3/host_screen.json"


def _read(mint, root: Path, screen: str):
    return mint.read_host_screen(
        screen, reports_root=root, mining_pool_sha256=POOL_SHA, n_candidates=20
    )


def test_a_valid_screen_is_accepted_and_recorded(mint, tmp_path, hosts):
    admitted = [row["candidate_id"] for row in hosts[:12]]
    report, listing = _valid_screen(admitted)
    ids, record = _read(mint, tmp_path, _write_screen(tmp_path, report, listing))
    assert ids == sorted(admitted)
    assert record["report"] == "reports/p3/host_screen.json"
    assert record["admitted_list"] == "reports/p3/host_screen_admitted.txt"
    assert record["admitted_sha256"] == _sha(listing)
    assert record["n_admitted"] == 12
    assert record["threshold_bits"] == 10.0
    assert (
        record["sha256"]
        == hashlib.sha256((tmp_path / "reports/p3/host_screen.json").read_bytes()).hexdigest()
    )


def _set(path: tuple, value):
    def mutate(report, listing):
        node = report
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return report, listing

    return mutate


def _unfire(report, listing):
    report["controls"][1]["fired"] = False  # the flag still says True
    return report, listing


def _relist(transform):
    """Rewrite the admitted list AND its recorded digest, so only the list's shape is wrong."""

    def mutate(report, listing):
        new = transform(listing)
        report["admitted"]["sha256"] = _sha(new)
        return report, new

    return mutate


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(_set(("problems",), ["a validator problem"]), "problems", id="problems"),
        pytest.param(
            lambda r, s: ({k: v for k, v in r.items() if k != "problems"}, s),
            "problems",
            id="problems-missing",
        ),
        pytest.param(_set(("controls_all_fired",), False), "controls_all_fired", id="flag"),
        pytest.param(_unfire, "did not fire", id="control-unfired-flag-true"),
        pytest.param(_set(("controls",), []), "no must-fire controls", id="no-controls"),
        pytest.param(_set(("content_screen", "threshold_bits"), 20.0), "threshold", id="20bits"),
        pytest.param(_set(("content_screen", "threshold_bits"), "10.0"), "threshold", id="str"),
        pytest.param(_set(("pool", "sha256"), "0" * 64), "--mining-pool", id="pool-sha"),
        pytest.param(_set(("pool", "n_candidates"), 19), "out-of-fold windows", id="pool-n"),
        pytest.param(_set(("admitted", "sha256"), "0" * 64), "hash to", id="list-sha"),
        pytest.param(
            lambda r, s: (r, s + "an-extra-window:lead\n"), "hash to", id="list-bytes-edited"
        ),
        pytest.param(_set(("admitted", "list"), "/abs/list.txt"), "repo-relative", id="abs"),
        pytest.param(
            _set(("admitted", "list"), "reports/p3/missing.txt"), "could not be read", id="gone"
        ),
        pytest.param(
            _relist(lambda s: "".join(sorted(s.splitlines(True))[::-1])),
            "not sorted",
            id="unsorted",
        ),
        pytest.param(_relist(lambda s: s + s.splitlines(True)[0]), "repeats", id="dup"),
        pytest.param(_relist(lambda s: s.rstrip("\n")), "newline", id="no-final-lf"),
        pytest.param(_set(("admitted", "n"), 11), "holds", id="count"),
    ],
)
def test_a_screen_that_does_not_bind_to_this_mint_is_refused(mint, tmp_path, hosts, mutate, match):
    report, listing = _valid_screen([row["candidate_id"] for row in hosts[:12]])
    report, listing = mutate(copy.deepcopy(report), listing)
    screen = _write_screen(tmp_path, report, listing)
    with pytest.raises(SystemExit, match=match):
        _read(mint, tmp_path, screen)


def test_the_screen_path_itself_must_be_repo_relative_json(mint, tmp_path, hosts):
    report, listing = _valid_screen([row["candidate_id"] for row in hosts[:12]])
    _write_screen(tmp_path, report, listing)
    with pytest.raises(SystemExit, match="repo-relative"):
        _read(mint, tmp_path, str(tmp_path / "reports/p3/host_screen.json"))
    with pytest.raises(SystemExit, match="not a JSON report"):
        _read(mint, tmp_path, _write_screen(tmp_path, "{not json", listing))


def test_the_refusal_validator_is_total_on_a_malformed_payload(mint):
    """It returns problems, never raises, on a payload this script did not write."""
    for payload in (None, [], "x", {"problems": None, "controls": "x", "pool": 7}):
        problems = mint.host_screen_refusals(
            payload, mining_pool_sha256=POOL_SHA, n_candidates=20, admitted_list_sha256=None
        )
        assert problems


# ──────────────────────────────────────────────────────────────────────────────
# scope.host_pool: the selection names the screen; composition of the drawn hosts
# ──────────────────────────────────────────────────────────────────────────────
def _pool_scope(mint, screen):
    return mint.host_pool_scope(
        mining_pool="data/processed/negatives/mining_pool_v0.parquet",
        n_admitted=40,
        n_out_of_fold=20,
        n_drawable=12 if screen else 20,
        n_used=8,
        screen=screen,
        composition={"n_hosts": 8},
        host_seen={"production": {"dec:a"}, "twin": set()},
        n_negatives=8,
    )


def test_the_screened_selection_names_the_screen_and_claims_no_cleanliness(mint):
    record = {"report": "reports/p3/host_screen.json", "threshold_bits": 10.0}
    screened = _pool_scope(mint, record)
    assert "ADR-0005 A15" in screened["selection"]
    assert screened["screen"] == record
    unscreened = _pool_scope(mint, None)
    assert "ADR-0005 A15" not in unscreened["selection"]
    assert unscreened["screen"] is None
    for block in (screened, unscreened):
        assert "clean for BOTH" not in block["selection"]
        overlap = block["overlap_with_training_folds"]
        assert overlap["by_arm"] == {"production": 1, "twin": 0}
        assert overlap["n_seen_by_any_arm"] == 1
        assert "non-ACGT" in overlap["method"]


def test_composition_counts_the_drawn_hosts_and_labels_every_gap(mint, hosts):
    by_id = {row["candidate_id"]: row for row in hosts}
    # four hosts with four DISTINCT parents (slabs come in fwd/rc pairs of one record)
    drawn = [hosts[0], hosts[3], hosts[4], hosts[7]]
    assert len({row["source_record_id"] for row in drawn}) == 4
    items = [{"contig_id": "pos:x", "label": 1, "host_id": None}] + [
        {"contig_id": f"dec:{i}", "label": 0, "host_id": row["candidate_id"]}
        for i, row in enumerate(drawn)
    ]
    parent_meta = {
        hosts[0]["source_record_id"]: {
            "nested_role": "heldout",
            "is_designated_loo_holdout": True,
            "resolved_order": "Bacillales",
        },
        hosts[3]["source_record_id"]: {
            "nested_role": "dropped",
            "is_designated_loo_holdout": False,
            "resolved_order": None,
        },
        hosts[4]["source_record_id"]: {
            "nested_role": "heldout",
            "is_designated_loo_holdout": True,
            "resolved_order": "Bacillales",
        },
        # hosts[7]'s parent is absent from the split table
    }
    assert hosts[7]["source_record_id"] not in parent_meta
    got = mint.host_composition(items, by_id, parent_meta)
    assert got["n_hosts"] == 4
    assert got["side"] == {"lead": 2, "trail": 2}
    assert got["parent_nested_role"] == {
        "absent_from_split_table": 1,
        "dropped": 1,
        "heldout": 2,
    }
    assert got["parent_is_designated_loo_holdout"] == {
        "absent_from_split_table": 1,
        "false": 1,
        "true": 2,
    }
    assert got["parent_resolved_order"] == {
        "Bacillales": 2,
        "absent_from_split_table": 1,
        "null": 1,
    }
    for block in ("side", "parent_nested_role", "parent_resolved_order"):
        assert sum(got[block].values()) == got["n_hosts"]


def test_the_v1_audit_and_the_v2_draw_tabulate_composition_on_the_same_axes(mint):
    """A15 Pin 7(e) reads v2's table beside v1's: the two producers must share axes and labels."""
    from tbox_finder.integration import host_screen

    assert host_screen.COMPOSITION_PARENT_FIELDS == mint.COMPOSITION_PARENT_FIELDS
    assert host_screen.ABSENT_PARENT == mint.ABSENT_PARENT
    for value in (None, float("nan"), True, False, "heldout"):
        assert host_screen._composition_label(value) == mint._composition_label(value)


# ──────────────────────────────────────────────────────────────────────────────
# Insert-level exposure (A15 Pin 7(c))
# ──────────────────────────────────────────────────────────────────────────────
def _exposure_items(slabs):
    """Three gated negatives over two pools plus the non-gated leader, on real slab DNA."""
    pools = ["structured_rna", "structured_rna", "gc_background", "leader_decoy"]
    items = [{"contig_id": "pos:p", "label": 1, "pool": "positive"}]
    sequences = {}
    for index, pool in enumerate(pools):
        decoy_id = f"d{index}"
        sequences[decoy_id] = slabs[index]["sequence"].upper()[200:420]
        items.append(
            {
                "contig_id": f"dec:{decoy_id}",
                "label": 0,
                "pool": pool,
                "gated": pool != "leader_decoy",
            }
        )
    return items, sequences


def test_insert_exposure_counts_sequence_duplicates_of_stage2_training_decoys(mint, slabs):
    items, sequences = _exposure_items(slabs)
    # Training holds d0's insert as RNA in lower case (the stage2_dataset form), and the
    # leader's: d0 must count; the leader is reported beside the gated count, not inside it.
    training = [sequences["d0"].lower().replace("t", "u"), sequences["d3"].replace("T", "U")]
    got = mint.insert_exposure(items, sequences, training)
    assert got["n_gated_negatives"] == 3
    assert got["n_test_inserts_duplicating_stage2_training_decoys"] == 1
    assert got["by_pool_gated"] == {
        "gc_background": {"n": 1, "n_duplicating": 0},
        "structured_rna": {"n": 2, "n_duplicating": 1},
    }
    assert got["non_gated"] == {"dec:d3": True}
    assert got["n_stage2_training_decoy_sequences"] == 2
    assert got["rule"] == mint.INSERT_EXPOSURE_RULE


def test_insert_exposure_is_by_sequence_never_by_id(mint, slabs):
    """A training decoy that shares the test id but not its bytes is not a duplicate."""
    items, sequences = _exposure_items(slabs)
    training = [slabs[10]["sequence"].upper()[200:420]]
    assert all(slabs[10]["sequence"].upper()[200:420] != seq for seq in sequences.values())
    got = mint.insert_exposure(items, sequences, training)
    assert got["n_test_inserts_duplicating_stage2_training_decoys"] == 0
    assert got["non_gated"] == {"dec:d3": False}


def test_insert_exposure_refuses_an_empty_population_and_an_unknown_insert(mint, slabs):
    items, sequences = _exposure_items(slabs)
    with pytest.raises(SystemExit, match="vacuous 0"):
        mint.insert_exposure(items, sequences, [])
    missing = {key: value for key, value in sequences.items() if key != "d1"}
    with pytest.raises(SystemExit, match="not in decoys_v0"):
        mint.insert_exposure(items, missing, [sequences["d0"]])


# ──────────────────────────────────────────────────────────────────────────────
# Host-fold overlap: N-free k-mers (A15 Pin 1 alphabet rule)
# ──────────────────────────────────────────────────────────────────────────────
def test_host_overlap_skips_kmers_with_non_acgt_and_never_spans_the_excision(mint, slabs):
    k = mint.HOST_OVERLAP_K
    host = slabs[0]["sequence"].upper()[:WINDOW]
    unrelated = slabs[16]["sequence"].upper()  # a different corpus record
    n_run = "N" * 40
    host = host[:100] + n_run + host[140:]
    insert = slabs[2]["sequence"].upper()[1200:1280]
    phase = 600
    contig = {"contig_id": "dec:x", "sequence": emb.splice(host, insert, phase)}
    item = {"contig_id": "dec:x", "label": 0, "splice_phase": phase, "locus_length": len(insert)}
    left, right = host[:phase], host[phase + len(insert) :]

    # Fixture guard: the unrelated record shares no ACGT k-mer with either flank, so every hit
    # below comes from the segment planted into it.
    flank_kmers = set(mint.acgt_kmers(left, k)) | set(mint.acgt_kmers(right, k))
    assert not flank_kmers & set(mint.acgt_kmers(unrelated, k))
    assert not flank_kmers & set(mint.acgt_kmers(insert, k))

    def carrying(segment: str) -> str:
        return unrelated[:500] + segment + unrelated[500:]

    contexts = {
        "n_run_only": carrying("A" + n_run + "C"),
        "real_flank": carrying(host[300:340]),
        "insert_only": carrying(insert),
        "junction": carrying(left[-16:] + right[:16]),
    }
    folds = {"a": {"n_run_only"}, "b": {"insert_only"}, "c": {"junction"}, "d": {"real_flank"}}
    seen = mint.host_fold_overlap([contig], [item], folds, contexts)
    assert seen["d"] == {"dec:x"}, "positive control: a real shared flank 32-mer is seen"
    assert seen["a"] == set(), "a shared N-run is an assembly gap, not shared DNA"
    assert seen["b"] == set(), "the spliced insert is not host DNA"
    assert seen["c"] == set(), "a k-mer across the excision exists in no genome"


def test_acgt_kmers_drops_every_kmer_that_touches_a_non_acgt_character(mint):
    assert list(mint.acgt_kmers("ACGTNACGT", 4)) == ["ACGT", "ACGT"]
    assert list(mint.acgt_kmers("ACGTACGT", 4)) == ["ACGT", "CGTA", "GTAC", "TACG", "ACGT"]
    assert list(mint.acgt_kmers("NNNN", 2)) == []


# ──────────────────────────────────────────────────────────────────────────────
# No v2 artifact is written to a v1 path (A15 Pin 5)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("relative", sorted(set(CANONICAL_SOURCES.values())))
def test_every_p3_17a_fixed_path_is_refused_as_an_output(mint, tmp_path, relative):
    assert mint.v1_out_violation(tmp_path / relative) is not None


def test_the_v1_benchmark_name_is_refused_anywhere_and_v2_is_not(mint, tmp_path):
    assert mint.v1_out_violation(tmp_path / "elsewhere" / "benchmark_v1.json") is not None
    # a path that only RESOLVES to the canonical one
    sneaky = tmp_path / "data" / "interim" / "p3_16" / "x" / ".." / "v1_stage2_twin.json"
    assert mint.v1_out_violation(sneaky) is not None
    # positive control: the v2 names A15 Pin 5 assigns are writable
    for name in ("benchmark_v2.json", "v2_stage1_twin.npz", "v2_stage2_twin.json"):
        assert mint.v1_out_violation(tmp_path / "data" / "interim" / "p3_16" / name) is None
