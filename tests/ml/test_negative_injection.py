"""P2-10d — ``build_stream`` end to end with negatives, on the real DVC corpus.

The unit tier pins each piece; this is the wiring. A defect here — a mixer built over the
wrong index space, a boundary off by one, a scope block that counts the wrong population —
survives every unit test and is discovered only by an eight-hour cluster run.

**Tier:** reads `data/interim/flank_context/context_v0.parquet`, the labels table, the
committed split table and `data/processed/negatives/mining_pool_v0.parquet` — DVC-tracked,
so local/cluster only. No torch: `build_stream` is pandas + numpy.

⚠ **The 1024-nt pool built below is a TRANSPORT fixture, not a mineable pool.** It carves
real genomic sequence out of the P2-00 flank regions at the training window width, which is
the only way to exercise the injection path today — and it is *not* minable under
ADR-0005 D14, because a 1024-nt flank window has to abut its own parent locus (the P2-00
regions carry exactly 1024 nt of flank, so a window that wide leaves no margin) and the
union-prior mask at the D14 flank removes every one of them. That measurement is committed
in `reports/p2/negative_injection.json`; it is the named P2-10e input, and nothing here may
be read as supplying it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONTEXT = REPO / "data/interim/flank_context/context_v0.parquet"
MINING_POOL = REPO / "data/processed/negatives/mining_pool_v0.parquet"
WINDOW = 1024


def _need(path: Path) -> None:
    if not path.is_file():
        if os.environ.get("TBOX_REQUIRE_NEGATIVE_INJECTION") == "1":
            pytest.fail(f"TBOX_REQUIRE_NEGATIVE_INJECTION=1 but {path} is absent")
        pytest.skip(f"{path} absent (DVC-tracked; run `dvc pull`)")


@pytest.fixture(scope="module")
def transport_pool(tmp_path_factory):
    """A 1024-nt all-background pool carved from real flank sequence. See the ⚠ above."""
    _need(CONTEXT)
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = pd.read_parquet(CONTEXT, columns=["record_id", "status", "locus_offset", "context_seq"])
    rows = []
    for row in frame.itertuples(index=False):
        if row.status != "ok" or int(row.locus_offset) < WINDOW:
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
            }
        )
        if len(rows) == 300:
            break
    assert len(rows) == 300, f"only {len(rows)} carvable 1024-nt lead windows"
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


def test_the_real_mining_pool_is_refused_at_the_training_window() -> None:
    """The measured substrate gap, asserted rather than described.

    P2-10b's pool is carved at 300 nt (`mining/pool.py::DEFAULT_WINDOW_NT`). Injecting it
    would require padding a 300-nt window to 1024 — inventing a contig boundary — so
    `build_stream` refuses and names the reason. When P2-10e lands a window-width pool this
    test changes; until then it is the executable form of the blocker.
    """
    _need(MINING_POOL)
    from tbox_finder.train.train_stage1 import build_stream

    cfg = _cfg(negative_pool_parquet=str(MINING_POOL))
    with pytest.raises(ValueError, match="supplied 0 injectable negatives"):
        build_stream(cfg)


def test_negative_max_records_caps_the_pool(transport_pool) -> None:
    from tbox_finder.train.train_stage1 import build_stream

    _ds, _sampler, scope = build_stream(
        _cfg(negative_pool_parquet=str(transport_pool), negative_max_records=17)
    )
    assert scope["n_negative_records"] == 17
    assert scope["negative_pool"]["max_records"] == 17
