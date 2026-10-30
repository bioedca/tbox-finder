"""P3-16″-hardneg — ``integration/hardneg_pool.py`` (ADR-0005 A17 Pins 1–2).

Geometry and arithmetic run on small synthetic contigs: they test coordinate algebra, not
biology. The D4 partition is exercised through the shipped ``synteny`` classifier on real NCBI
product strings, and every validator clause has a positive control plus a sabotage that flips
that clause alone while the rest of the forge stays consistent
([[all-true-fixture-cannot-test-a-conjunction]]).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random

import pytest

from tbox_finder.data import window_dataset as wd
from tbox_finder.integration import hardneg_pool as hp
from tbox_finder.integration import host_screen as hs
from tbox_finder.mining import gff3


def _contig(n: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


# ── Pin 1 geometry ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("delta", [0, 10, 256])
def test_plus_strand_window_ends_delta_into_the_feature(delta):
    contig = _contig(5000)
    first1 = 2001  # 1-based first base → s0 = 2000
    start, end = hp.window_span(first1, "+", delta)
    assert (start, end) == (2000 - 1024 + delta, 2000 + delta)
    seq = hp.window_sequence(contig, {"start": start, "end": end, "strand": "+"})
    assert len(seq) == hp.WINDOW_NT
    assert seq[hp.WINDOW_NT - delta :] == contig[2000 : 2000 + delta]


@pytest.mark.parametrize("delta", [0, 10, 256])
def test_minus_strand_window_is_the_mirror_image(delta):
    contig = _contig(5000)
    first1 = 3000  # the − feature's first base is its highest coordinate, 0-based index 2999
    start, end = hp.window_span(first1, "-", delta)
    assert (start, end) == (3000 - delta, 3000 + 1024 - delta)
    seq = hp.window_sequence(contig, {"start": start, "end": end, "strand": "-"})
    assert len(seq) == hp.WINDOW_NT
    # The last δ bases in feature sense are the feature's own first δ bases, read 3′→5′ on +.
    assert seq[hp.WINDOW_NT - delta :] == wd.reverse_complement(contig[3000 - delta : 3000])
    if delta:
        assert seq[hp.WINDOW_NT - delta] == wd.reverse_complement(contig[2999])


@pytest.mark.parametrize("delta", [-1, 257])
def test_delta_outside_the_pinned_range_refuses(delta):
    with pytest.raises(hp.HardnegPoolError):
        hp.window_span(2001, "+", delta)


def test_unresolved_strand_refuses():
    with pytest.raises(hp.HardnegPoolError):
        hp.window_span(2001, ".", 0)


def _features() -> list[hp.Feature]:
    return sorted(
        [
            hp.Feature("GCA_1.1", "c1", 2001, "+", "CDS", "cds-a"),
            hp.Feature("GCA_1.1", "c1", 2001, "+", "tRNA", "rna-a"),  # same window as cds-a
            hp.Feature("GCA_1.1", "c1", 500, "+", "CDS", "cds-edge"),  # runs off the 5′ end
            hp.Feature("GCA_1.1", "c1", 4900, "-", "CDS", "cds-tail"),  # runs off the 3′ end
            hp.Feature("GCA_1.1", "c1", 3000, "-", "CDS", "cds-b"),
        ],
        key=lambda f: f.sort_key,
    )


def test_eligibility_drops_off_contig_and_keeps_the_first_duplicate():
    features = _features()
    eligible, status = hp.eligible_windows(features, [0] * len(features), {("GCA_1.1", "c1"): 5000})
    assert status == {"off_contig": 2, "duplicate_window": 1, "eligible": 2}
    kept = {w["feature_id"] for w in eligible}
    # cds-a sorts before rna-a (type "CDS" < "tRNA"), so the CDS keeps the shared window.
    assert kept == {"cds-a", "cds-b"}
    assert len({w["item_id"] for w in eligible}) == len(eligible)


def test_a_seqid_missing_from_the_genome_refuses():
    with pytest.raises(hp.HardnegPoolError):
        hp.eligible_windows(_features()[:1], [0], {})


# ── Pin 1 RNG order and selection ───────────────────────────────────────────────────────
def _eligible(n: int) -> list[dict]:
    return [
        {
            "item_id": f"A:c:{i}-{i + 1024}:+",
            "feature_type": "CDS",
            "feature_id": f"f{i}",
            "delta": 0,
        }
        for i in range(n)
    ]


def test_replay_reproduces_the_delta_then_draw_order():
    eligible = _eligible(50)
    rng = random.Random(hp.SEED)
    deltas = hp.assign_deltas(range(60), rng)
    drawn = hp.select(eligible, rng, mode=hp.MODE_DRAW, size=20)
    replayed_deltas, replayed = hp.replay_selection(60, eligible, mode=hp.MODE_DRAW, size=20)
    assert replayed_deltas == deltas and replayed == drawn
    # The order is load-bearing: drawing before the δs gives a different draw.
    other = random.Random(hp.SEED).sample(eligible, 20)
    assert other != drawn


def test_census_is_exactly_the_complement_of_the_draw():
    eligible = _eligible(50)
    _d, drawn = hp.replay_selection(10, eligible, mode=hp.MODE_DRAW, size=20)
    _d, census = hp.replay_selection(10, eligible, mode=hp.MODE_CENSUS, size=20)
    drawn_ids = {w["item_id"] for w in drawn}
    census_ids = {w["item_id"] for w in census}
    assert not drawn_ids & census_ids
    assert drawn_ids | census_ids == {w["item_id"] for w in eligible}


def test_a_draw_larger_than_the_pool_refuses():
    with pytest.raises(hp.HardnegPoolError):
        hp.select(_eligible(5), random.Random(1), mode=hp.MODE_DRAW, size=6)


# ── Pin 2 step 4: the D4-context partition ──────────────────────────────────────────────
def _cds(fid: str, start: int, end: int, strand: str, product: str) -> gff3.CdsFeature:
    return gff3.CdsFeature(
        seqid="c1",
        feature_id=fid,
        start=start,
        end=end,
        strand=strand,
        segments=((start, end),),
        attributes={"product": (product,)},
    )


def test_d4_intervals_keep_only_d4_classes_on_either_strand():
    intervals = hp.d4_intervals(
        [
            _cds("a", 101, 400, "+", "tyrosine--tRNA ligase"),
            _cds("b", 1001, 1300, "-", "tyrosine--tRNA ligase"),
            _cds("c", 2001, 2300, "+", "hypothetical protein"),
        ]
    )
    assert intervals == {"c1": [(100, 400), (1000, 1300)]}


@pytest.mark.parametrize(
    ("cds_end_half_open", "expected"),
    [(1500, False), (1501, True)],  # window start 2000 − 500 = 1500 is the exclusive bound
)
def test_d4_neighbourhood_boundary_is_half_open(cds_end_half_open, expected):
    window = {"seqid": "c1", "start": 2000, "end": 3024}
    assert hp.is_d4_context(window, {"c1": [(1000, cds_end_half_open)]}) is expected
    downstream = {"c1": [(3524 - (cds_end_half_open - 1500), 4000)]}
    assert hp.is_d4_context(window, downstream) is expected


def test_d4_context_reads_only_the_window_seqid():
    assert not hp.is_d4_context({"seqid": "c2", "start": 0, "end": 1024}, {"c1": [(0, 5000)]})


# ── audit sample and partition counts ────────────────────────────────────────────────────
def test_audit_sample_is_a_seeded_subset_of_graded():
    graded = [f"g{i:05d}" for i in range(3000)]
    sample = hp.audit_sample(reversed(graded))
    assert len(sample) == hp.AUDIT_SIZE and set(sample) <= set(graded)
    assert sample == hp.audit_sample(graded)  # input order does not matter
    assert hp.audit_sample(graded[:10]) == graded[:10]


def test_partition_counts_tally_every_window_once():
    items = {
        "a": {"partition": hp.P_GRADED, "sub_class": "cds_upstream"},
        "b": {"partition": hp.P_GRADED, "sub_class": "trna_upstream"},
        "c": {"partition": hp.P_GAP, "sub_class": "cds_upstream"},
    }
    counts = hp.partition_counts(items)
    assert sum(block["total"] for block in counts.values()) == 3
    assert counts[hp.P_GRADED] == {"cds_upstream": 1, "trna_upstream": 1, "total": 2}


# ── validate_report ─────────────────────────────────────────────────────────────────────
def _controls() -> list[dict]:
    names = [
        hs.CONTROL_HOST_CLEAN,
        hs.CONTROL_COORD_ONLY,
        hs.CONTROL_CONTENT,
        hs.CONTROL_CONTENT_HOST,
        "pin1_accession_as_given",
        *(f"pin1_kmer_{source}" for source in hs.SCREENED_SOURCES),
    ]
    return [hs.control_entry(name, "m", True, True) for name in names]


def _report() -> dict:
    counts = {
        hp.P_GAP: {"cds_upstream": 5_000, "total": 5_000},
        hp.P_FOLD: {"cds_upstream": 10_000, "total": 10_000},
        hp.P_CONTENT: {"cds_upstream": 1_000, "total": 1_000},
        hp.P_GRADED: {"cds_upstream": 78_000, "trna_upstream": 2_000, "total": 80_000},
        hp.P_BESIDE: {"cds_upstream": 4_000, "total": 4_000},
    }
    audit_ids = [f"g{i:05d}" for i in range(hp.AUDIT_SIZE)]
    calibration = {"null_max_bits": {"RF00230": 15.1, "TBDB001": 11.9}}
    rows = [{"item_id": "g00001", "model": "RF00230", "bits": 14.5}]
    return {
        "kind": hp.KIND_POOL,
        "schema_version": hp.SCHEMA_VERSION,
        "config": {
            "seed": hp.SEED,
            "delta_max": hp.DELTA_MAX,
            "window_nt": hp.WINDOW_NT,
            "pinned_draw": True,
            "max_hosts": None,
            "draw_size": hp.DRAW_SIZE,
            "mode": hp.MODE_DRAW,
        },
        "population": {"n_eligible": 900_000, "n_selected": hp.DRAW_SIZE},
        "partition_counts": counts,
        "controls": _controls(),
        "controls_all_fired": True,
        "audit_ids": audit_ids,
        "audit": hp.audit_block(rows, len(audit_ids), calibration),
        "files": {"features.tsv": {"path": "data/interim/p3_16/hardneg_pool/features.tsv"}},
    }


def test_a_valid_report_has_no_problems():
    assert hp.validate_report(_report()) == []


def _sabotage(mutate) -> list[str]:
    report = copy.deepcopy(_report())
    mutate(report)
    return hp.validate_report(report)


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda r: r.update(kind="host_screen"), "kind"),
        (lambda r: r["config"].update(seed=43), "config.seed"),
        (lambda r: r["config"].update(pinned_draw=False), "pinned draw"),
        (lambda r: r["config"].update(max_hosts=3), "pinned draw"),
        (lambda r: r["population"].update(n_selected=99_999), "draw selected"),
        (
            # one partition grows by 1, its sub-classes kept consistent: only conservation breaks
            lambda r: r["partition_counts"][hp.P_FOLD].update(cds_upstream=10_001, total=10_001),
            "conservation",
        ),
        (
            lambda r: r["partition_counts"][hp.P_GRADED].update(trna_upstream=1_999),
            "sub-classes sum",
        ),
        (lambda r: r["controls"][0].update(observed=False, fired=False), "did not fire"),
        (lambda r: r["controls"][0].update(fired=False), "fired is not observed == expected"),
        (lambda r: r.update(audit_ids=r["audit_ids"][:-1]), "audit has"),
        (lambda r: r["audit"].update(search_threshold_bits=10.0), "0-bit floor"),
        (lambda r: r["audit"].update(n_violations=1), "n_violations disagrees"),
        (lambda r: r["audit"].update(passes=False), "passes disagrees"),
        (
            lambda r: r["files"]["features.tsv"].update(path="/home/x/features.tsv"),
            "absolute",
        ),
    ],
)
def test_each_clause_refuses_alone(mutate, needle):
    problems = _sabotage(mutate)
    assert problems and any(needle in p for p in problems), problems


def test_an_audit_violation_fails_with_a_consistent_forge():
    def mutate(report):
        rows = [{"item_id": "g00002", "model": "RF00230", "bits": 20.0}]
        report["audit"] = hp.audit_block(rows, hp.AUDIT_SIZE, report["audit"]["calibration"])

    problems = _sabotage(mutate)
    assert any("audit FAILS" in p for p in problems), problems
    assert not any("disagrees" in p for p in problems)


def test_a_threshold_inside_its_floor_fails_the_audit():
    def mutate(report):
        calibration = {"null_max_bits": {"RF00230": 21.0, "TBDB001": 11.9}}
        report["audit"] = hp.audit_block([], hp.AUDIT_SIZE, calibration)

    assert any("does not clear" in p for p in _sabotage(mutate))


@pytest.mark.parametrize("bad", [None, [], "x", {"kind": hp.KIND_POOL}])
def test_malformed_input_returns_problems_and_never_raises(bad):
    assert hp.validate_report(bad)


def test_malformed_audit_row_is_a_problem():
    def mutate(report):
        report["audit"]["rows_at_or_above_10_bits"] = [{"model": "RF00230"}]

    assert any("malformed" in p for p in _sabotage(mutate))


@pytest.mark.parametrize("bad", [[15.1], "15.1", 15.1])
@pytest.mark.parametrize("where", ["calibration", "null_max_bits"])
def test_a_non_mapping_calibration_is_a_problem_and_never_raises(where, bad):
    def mutate(report):
        if where == "calibration":
            report["audit"]["calibration"] = bad
        else:
            report["audit"]["calibration"]["null_max_bits"] = bad

    assert any("does not clear" in p for p in _sabotage(mutate))


def test_a_bool_floor_does_not_clear_its_threshold():
    """``True`` is an int to ``isinstance`` and ``20.0 > True`` holds, so a bool floor would
    read as measured and cleared ([[bool-equals-one-passes-config-gates]])."""

    def mutate(report):
        calibration = {"null_max_bits": {"RF00230": True, "TBDB001": 11.9}}
        report["audit"] = hp.audit_block([], hp.AUDIT_SIZE, calibration)

    assert any("does not clear" in p for p in _sabotage(mutate))


# ── re-derivation against the data root ─────────────────────────────────────────────────
def _write_population(root, n_features=100_050):
    eligible = _eligible(n_features)
    deltas, drawn = hp.replay_selection(n_features, eligible, mode=hp.MODE_DRAW, size=hp.DRAW_SIZE)
    base = root / "data/interim/p3_16/hardneg_pool"
    base.mkdir(parents=True)
    (base / "features.tsv").write_text("".join(f"A\tc\t1\t+\tCDS\tf\t{d}\n" for d in deltas))
    text = "".join(hp.eligible_line(w) + "\n" for w in eligible)
    (base / "eligible.tsv").write_text(text)
    drawn_ids = sorted(w["item_id"] for w in drawn)
    (base / "drawn.txt").write_text("".join(f"{cid}\n" for cid in drawn_ids))
    return base, text, drawn_ids


def _rooted_report(root):
    base, text, drawn_ids = _write_population(root)
    report = _report()
    graded = drawn_ids[:80_000]
    beside = drawn_ids[80_000:84_000]
    report["audit_ids"] = hp.audit_sample(graded)
    report["audit"]["n_audited"] = len(report["audit_ids"])
    annex = root / "reports/p3/hardneg_pool_ids.tsv"
    annex.parent.mkdir(parents=True)
    annex.write_text(
        "".join(f"{cid}\t{hp.P_GRADED}\n" for cid in graded)
        + "".join(f"{cid}\t{hp.P_BESIDE}\n" for cid in beside)
    )
    report["ids_annex"] = {
        "path": "reports/p3/hardneg_pool_ids.tsv",
        "sha256": hashlib.sha256(annex.read_bytes()).hexdigest(),
    }
    rel = "data/interim/p3_16/hardneg_pool"
    report["files"] = {
        "features.tsv": {"path": f"{rel}/features.tsv"},
        "eligible.tsv": {
            "path": f"{rel}/eligible.tsv",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
        "drawn.txt": {"path": f"{rel}/drawn.txt"},
    }
    report["population"]["n_eligible"] = 100_050
    return report, base


def test_rederivation_passes_on_a_consistent_population(tmp_path):
    report, _base = _rooted_report(tmp_path)
    assert hp.validate_report(report, root=tmp_path) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(partition_counts=[1]),
        lambda r: r["partition_counts"].update({hp.P_GRADED: "80000"}),
        lambda r: r["partition_counts"].update({hp.P_BESIDE: [4_000]}),
    ],
    ids=["counts-not-a-mapping", "graded-block-a-string", "beside-block-a-list"],
)
def test_rederivation_on_non_mapping_partition_counts_is_a_problem_and_never_raises(
    tmp_path, mutate
):
    report, _base = _rooted_report(tmp_path)
    mutate(report)
    assert any("annex disagree" in p for p in hp.validate_report(report, root=tmp_path))


# ── publish_report: an annex is only ever replaced by one whose report validated ─────────
def _staged_case(root):
    """A rooted report whose annex sits at the STAGED path, beside an earlier accepted pair."""
    report, _base = _rooted_report(root)
    p3 = root / "reports/p3"
    annex, out = p3 / "hardneg_pool_ids.tsv", p3 / "hardneg_pool.json"
    staged = annex.with_suffix(".staged.tsv")
    annex.replace(staged)
    report["ids_annex"]["path"] = "reports/p3/hardneg_pool_ids.staged.tsv"
    annex.write_text("an earlier accepted annex\n")
    out.write_text('{"an": "earlier accepted report"}\n')
    return report, annex, out, staged


def test_an_accepted_report_publishes_its_annex_and_records_the_final_path(tmp_path):
    report, annex, out, staged = _staged_case(tmp_path)
    fresh = staged.read_bytes()
    assert hp.publish_report(report, out=out, annex=annex, staged=staged, root=tmp_path) == 0
    assert annex.read_bytes() == fresh and not staged.exists()
    written = json.loads(out.read_text())
    assert written["ids_annex"]["path"] == "reports/p3/hardneg_pool_ids.tsv"
    assert hp.validate_report(written, root=tmp_path) == []
    assert not out.with_suffix(".invalid.json").exists()


def test_a_refused_report_leaves_the_accepted_annex_and_removes_the_stale_report(tmp_path):
    report, annex, out, staged = _staged_case(tmp_path)
    fresh, earlier = staged.read_bytes(), annex.read_bytes()
    report["config"]["seed"] = 43  # a real clause refuses it
    assert hp.publish_report(report, out=out, annex=annex, staged=staged, root=tmp_path) == 3
    assert annex.read_bytes() == earlier, "a refused run overwrote the accepted annex"
    assert not out.exists(), "a stale accepted report would sit beside the fresh .invalid.json"
    rejected = annex.with_suffix(".invalid.tsv")
    assert rejected.read_bytes() == fresh and not staged.exists()
    invalid = json.loads(out.with_suffix(".invalid.json").read_text())
    assert invalid["ids_annex"]["path"] == "reports/p3/hardneg_pool_ids.invalid.tsv"
    assert any("config.seed" in p for p in invalid["problems"])


def test_a_tampered_delta_fails_the_replay(tmp_path):
    report, base = _rooted_report(tmp_path)
    rows = (base / "features.tsv").read_text().splitlines()
    head, delta = rows[0].rsplit("\t", 1)
    rows[0] = f"{head}\t{(int(delta) + 1) % 257}"
    (base / "features.tsv").write_text("\n".join(rows) + "\n")
    problems = hp.validate_report(report, root=tmp_path)
    assert any("δ column" in p for p in problems), problems


def test_a_tampered_draw_fails_the_replay(tmp_path):
    report, base = _rooted_report(tmp_path)
    drawn = (base / "drawn.txt").read_text().splitlines()
    (base / "drawn.txt").write_text("\n".join(drawn[1:]) + "\n")
    problems = hp.validate_report(report, root=tmp_path)
    assert any("drawn list" in p for p in problems), problems


def test_a_missing_population_file_is_a_problem_not_a_pass(tmp_path):
    report, base = _rooted_report(tmp_path)
    (base / "eligible.tsv").unlink()
    assert any("cannot re-derive" in p for p in hp.validate_report(report, root=tmp_path))


def test_a_data_file_missing_from_the_data_root_is_never_read_from_the_code_root(
    tmp_path, monkeypatch
):
    # A worktree can hold its own staged copy of the population at the same relative path. Plant
    # one (byte-identical, so its sha256 would even pass) in a stand-in code root, delete the
    # data root's, and the validator must still refuse rather than validate against the decoy.
    report, base = _rooted_report(tmp_path / "data_root")
    code_root = tmp_path / "code_root"
    decoy = code_root / "data/interim/p3_16/hardneg_pool/eligible.tsv"
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes((base / "eligible.tsv").read_bytes())
    monkeypatch.setattr(hp.hs, "CODE_ROOT", code_root)
    assert hp.validate_report(report, root=tmp_path / "data_root") == []
    (base / "eligible.tsv").unlink()
    found = hp.validate_report(report, root=tmp_path / "data_root")
    assert any("cannot re-derive" in p for p in found), found
