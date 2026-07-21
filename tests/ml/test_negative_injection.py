"""P2-10d — ``build_stream`` end to end with negatives, on the real DVC corpus.

The unit tier pins each piece; this is the wiring. A defect here — a mixer built over the
wrong index space, a boundary off by one, a scope block that counts the wrong population —
survives every unit test and is discovered only by an eight-hour cluster run.

**Tier:** reads `data/interim/flank_context/context_v0.parquet`, the labels table, the
committed split table and `data/processed/negatives/mining_pool_v0.parquet` — DVC-tracked,
so local/cluster only. No torch: `build_stream` is pandas + numpy. The one clause that
reads only the **git-tracked** `reports/p2/negative_injection.json` raises rather than
skips, so this file is not entirely unarmed in CI (see `_need_tracked`).

**P2-10d′-a changed what this file asserts.** P2-10d shipped with a 300-nt mining pool and
a committed measurement of **zero** injectable negatives; the executable form of that
blocker lived here as `test_the_real_mining_pool_is_refused_at_the_training_window`, whose
own docstring said "when a window-width pool lands this test changes". It has. The pool is
now carved at the training window (`conf/data/decoys.yaml::mining_window_nt = 1024`) off a
flank re-fetched at pad 1074, so the real pool is injectable and the assertions run in the
positive direction. The refusal is **kept, not deleted** — repointed at a deliberately
wrong-width pool (`test_a_wrong_width_pool_is_still_refused`), because a test that only
proves the good case cannot notice the re-carve being reverted.

⚠ `transport_pool` remains a **fixture, not a mineable pool**: it slices the outer 1024 nt
of a P2-00 region with no D14 margin, so it abuts its own parent locus and is not a
candidate the mining loop could ever legitimately emit. It is retained for the ratio and
label tests because its size is fixed at exactly 300, which pins `len(dataset)` and the
mix arithmetic against a moving artifact. It is filtered to real `nested_train` parents so
it does not have to lie about the §9.2 provenance the loader now checks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONTEXT = REPO / "data/interim/flank_context/context_v0.parquet"
MINING_POOL = REPO / "data/processed/negatives/mining_pool_v0.parquet"
SPLIT_TABLE = REPO / "data/processed/splits/split_assignments.parquet"
#: git-tracked, so it is present in every checkout — including CI's.
AUDIT_REPORT = REPO / "reports/p2/negative_injection.json"
WINDOW = 1024


def _need(path: Path) -> None:
    if not path.is_file():
        if os.environ.get("TBOX_REQUIRE_NEGATIVE_INJECTION") == "1":
            pytest.fail(f"TBOX_REQUIRE_NEGATIVE_INJECTION=1 but {path} is absent")
        pytest.skip(f"{path} absent (DVC-tracked; run `dvc pull`)")


def _need_tracked(path: Path) -> None:
    """A **git-tracked** input needs no arming var: its absence is a broken checkout.

    Strictly stronger than a ``TBOX_REQUIRE_*`` skip, and it cannot rot into an unarmed
    variable that silently skips green — the failure mode `TBOX_REQUIRE_NEGATIVE_INJECTION`
    itself has (it is set in no workflow, no sbatch, and no doc), which is why every other
    clause in this file has never executed in CI. Copied from
    `tests/ml/test_mining_pool_gate.py`'s two-tier `_need`.
    """
    if not path.is_file():
        raise AssertionError(f"git-tracked artifact missing: {path.relative_to(REPO)}")


@pytest.fixture(scope="module")
def transport_pool(tmp_path_factory):
    """A 300-row, 1024-nt all-background pool carved from real flank. See the ⚠ above."""
    _need(CONTEXT)
    _need(SPLIT_TABLE)
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    split = pd.read_parquet(SPLIT_TABLE, columns=["record_id", "nested_train", "source"])
    corpus = split[split["source"] == "corpus"]
    fold = dict(zip(corpus["record_id"], corpus["nested_train"], strict=True))

    frame = pd.read_parquet(CONTEXT, columns=["record_id", "status", "locus_offset", "context_seq"])
    rows = []
    for row in frame.itertuples(index=False):
        if row.status != "ok" or int(row.locus_offset) < WINDOW:
            continue
        # Only training-fold parents. The fixture could stamp True unconditionally and the
        # loader would accept it — which is exactly why it must not: a fixture that fakes
        # the column under test turns every provenance assertion below into a tautology.
        if not bool(fold.get(str(row.record_id), False)):
            continue
        seq = row.context_seq[:WINDOW]
        if len(seq) != WINDOW:
            continue
        rows.append(
            {
                "pool": "transport_fixture",
                "candidate_id": f"{row.record_id}:lead1024",
                "sequence": seq,
                "masked": False,
                "is_designed_control": False,
                "source_record_id": str(row.record_id),
                "parent_nested_train": True,
            }
        )
        if len(rows) == 300:
            break
    assert len(rows) == 300, f"only {len(rows)} carvable 1024-nt nested_train lead windows"
    out = tmp_path_factory.mktemp("negatives") / "transport_pool.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def _cfg(**over):
    from tbox_finder.train.train_stage1 import Stage1TrainConfig

    base = {
        "max_records": 100,
        "eval_val": False,
        "exclude_selection_val": True,
        "negative_fraction": 10 / 11,
        "negative_max_records": 200,
    }
    base.update(over)
    return Stage1TrainConfig(**base)


def test_build_stream_injects_negatives_at_the_measured_ratio(transport_pool) -> None:
    """The step's headline gate, on the real corpus: configured ratio == counted ratio."""
    from tbox_finder.data.negatives import is_negative_record, mix_clauses
    from tbox_finder.train.train_stage1 import batch_plan, build_stream

    cfg = _cfg(negative_pool_parquet=str(transport_pool))
    dataset, sampler, scope = build_stream(cfg)

    assert scope["n_records"] == 100  # positives only — the field's existing meaning
    assert scope["n_negative_records"] == 200
    assert scope["class_counts_include_negatives"] is True
    assert scope["negative_pool"]["n_records"] == 200
    assert len(dataset) == 300

    mixer = sampler.inner
    summary = mixer.mix_summary()
    assert summary["n_positive"] == 100
    assert summary["n_negative"] == 1000  # 100 * (10/11) / (1/11)
    assert all(mix_clauses(summary, negative_fraction=cfg.negative_fraction).values())

    # The boundary is real: every key the summary counted negative indexes a negative.
    negative_keys = [k for k in mixer if k[0] >= 100]
    assert len(negative_keys) == 1000
    assert all(is_negative_record(dataset.records[k[0]]) for k in negative_keys)

    # And the plan the training loop consumes is a subset of that stream.
    plan = batch_plan(sampler, cfg)
    assert len(plan) <= len(mixer)
    assert set(plan) <= set(mixer)


def test_injected_negative_windows_are_all_real_all_background(transport_pool) -> None:
    """The label contract, on windows carved by the production datamodule."""
    import numpy as np

    from tbox_finder.data.window_dataset import BACKGROUND_INDEX, IGNORE_INDEX
    from tbox_finder.train.train_stage1 import build_stream

    dataset, _sampler, _scope = build_stream(_cfg(negative_pool_parquet=str(transport_pool)))
    for index in (100, 150, 299):
        sample = dataset[index]
        labels = np.asarray(sample["labels"])
        assert set(np.unique(labels).tolist()) == {BACKGROUND_INDEX}
        assert IGNORE_INDEX not in labels
        assert bool(np.asarray(sample["real_mask"]).all())
        assert sample["zero_flanked"] is False
        assert sample["parent_record_id"].startswith("neg:")


def test_the_positive_stream_is_unchanged_by_injection(transport_pool) -> None:
    """Injecting negatives must not perturb which positives the curriculum draws.

    Same seed, same epoch: the positive sub-stream of the mixed sampler must be exactly the
    stream the positives-only run emits. If it is not, every P2-06 sweep point and the P2-09
    checkpoint became irreproducible for a reason no config records.
    """
    from tbox_finder.train.train_stage1 import build_stream

    plain_ds, plain_sampler, _ = build_stream(_cfg(negative_fraction=0.0))
    mixed_ds, mixed_sampler, _ = build_stream(_cfg(negative_pool_parquet=str(transport_pool)))
    assert [r.record_id for r in plain_ds.records] == [r.record_id for r in mixed_ds.records[:100]]
    plain_keys = list(plain_sampler.inner)
    mixed_positive = [k for k in mixed_sampler.inner if k[0] < 100]
    assert sorted(mixed_positive) == sorted(plain_keys)


def test_the_real_mining_pool_is_injectable_at_the_training_window() -> None:
    """The substrate gap, closed — and asserted rather than described.

    Supersedes `test_the_real_mining_pool_is_refused_at_the_training_window`, whose own
    docstring anticipated this change. P2-10b carved at 300 nt, so `background_record`'s
    exact-width rule refused **every** row and the committed audit recorded 0 injectable
    negatives; P2-10d′-a re-fetched the flank at pad 1074 and re-carved at 1024, so the real
    pool now supplies the §9.1 seed substrate.

    The refusals are asserted too, and asserted to be **non-empty**: a pool that yields
    records because it refused nothing is a pool whose masking and control columns stopped
    being read, and "0 injectable" would then be able to return as "0 rows examined"
    without a single clause noticing.
    """
    _need(MINING_POOL)
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from tbox_finder.data import negatives as neg
    from tbox_finder.train.train_stage1 import build_stream

    records, report = neg.load_negative_records(MINING_POOL, window=WINDOW)
    assert report["n_records"] > 0, report["excluded_by_reason"]
    assert len(records) == report["n_records"]
    assert report["n_rows_at_window_length"] > 0
    assert report["window_nt"] == WINDOW
    assert report["require_parent_nested_train"] is True

    excluded = report["excluded_by_reason"]
    # The P2-10d failure mode is *gone*, not merely outvoted: not one natural window is
    # refused for width any more. This is the clause that goes red if the pool is ever
    # re-carved at a width the trainer cannot inject.
    assert neg.REASON_TOO_SHORT not in excluded
    assert neg.REASON_TOO_LONG not in excluded
    # Both live guards still refuse real rows. Measured 2026-07-20: masked 2,312 (= 1,812
    # natural + all 500 designed controls, which mask by construction and therefore hit the
    # mask rung BEFORE the control rung — so `designed_control` correctly reads 0 and is not
    # asserted here); parent-out-of-fold 27,960.
    assert excluded.get(neg.REASON_MASKED, 0) > 500
    assert excluded.get(neg.REASON_PARENT_OUT_OF_FOLD, 0) > 0
    assert report["n_refused_parent_unresolved"] == 0

    # …and the wiring end: build_stream builds a real mixed stream off the real pool.
    _dataset, _sampler, scope = build_stream(_cfg(negative_pool_parquet=str(MINING_POOL)))
    assert scope["n_negative_records"] == 200  # the negative_max_records cap in `_cfg`
    assert scope["negative_pool"]["n_records"] == 200


def test_a_wrong_width_pool_is_still_refused(tmp_path) -> None:
    """The old blocker, kept executable so reverting the re-carve cannot pass silently.

    `test_the_real_mining_pool_is_injectable_at_the_training_window` proves the good case,
    and a suite that proves only the good case goes green the moment someone restores
    `mining_window_nt: 300` — the refusal path would simply stop being exercised. So the
    refusal is repointed at a deliberately 300-nt pool rather than deleted (CLAUDE.md §8.7).
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _need(CONTEXT)  # only so this test skips in the same environments as its sibling
    from tbox_finder.train.train_stage1 import build_stream

    wrong = tmp_path / "wrong_width_pool.parquet"
    pd.DataFrame(
        [
            {
                "pool": "genomic_window",
                "candidate_id": f"wrong{i}:lead",
                "sequence": "ACGT" * 75,  # 300 nt — P2-10b's width
                "masked": False,
                "is_designed_control": False,
                "source_record_id": f"parent{i}",
                "parent_nested_train": True,
            }
            for i in range(8)
        ]
    ).to_parquet(wrong)

    with pytest.raises(ValueError, match="supplied 0 injectable negatives"):
        build_stream(_cfg(negative_pool_parquet=str(wrong)))


def test_negative_max_records_caps_the_pool(transport_pool) -> None:
    from tbox_finder.train.train_stage1 import build_stream

    _ds, _sampler, scope = build_stream(
        _cfg(negative_pool_parquet=str(transport_pool), negative_max_records=17)
    )
    assert scope["n_negative_records"] == 17
    assert scope["negative_pool"]["max_records"] == 17


def test_the_committed_audit_records_a_live_fold_rule() -> None:
    """The git-tracked tier — the only clause in this file that runs in CI.

    `reports/p2/negative_injection.json` is the artifact P2-10e is pointed at. Three things
    must be readable from it without re-running anything: that the pool supplies records at
    all, that the §9.2 admission rule was **armed** (not merely absent), and that it
    **refused something**. The third is the non-vacuity clause: `excluded_by_reason` omits a
    reason that never fired, so "no `parent_not_nested_train` key" is ambiguous between "the
    rule refused nothing" and "the rule was never armed" — which is why the report carries
    the counts as explicit integers.
    """
    _need_tracked(AUDIT_REPORT)
    from tbox_finder.data import negatives as neg

    report = json.loads(AUDIT_REPORT.read_text())
    assert report["schema_version"] == neg.SCHEMA_VERSION, (
        "the committed audit predates the fold rule; regenerate it with "
        "`python -m tbox_finder.data.negatives`"
    )
    assert report["window_nt"] == WINDOW
    assert report["n_records"] > 0
    assert report["require_parent_nested_train"] is True
    assert report["n_refused_parent_out_of_fold"] > 0
    assert report["n_refused_parent_unresolved"] == 0
    assert (
        report["excluded_by_reason"].get(neg.REASON_PARENT_OUT_OF_FOLD, 0)
        == report["n_refused_parent_out_of_fold"]
    )
    # `n_records` is a count of built objects, not `n_rows_read - n_excluded`; with no cap
    # the two must agree exactly, and a disagreement means rows vanished uncounted.
    assert report["max_records"] is None
    assert report["n_records"] + report["n_excluded"] == report["n_rows_read"]
