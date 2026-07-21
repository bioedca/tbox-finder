"""The ADR-0005 A7 composition pins, asserted against independent values (P2-10d′-b).

Two things this file deliberately does NOT do.

It does not check the mix identity by calling :func:`negative_draw_count` on both sides —
that is a tautology, and the P2-10c lesson is explicit that a test comparing a function to
the constant that produces it measures nothing
([[review-agents-mutate-code-under-review]]). The expected counts here are **closed-form
literals** derived from PRD §9.1's ratio by hand.

And it does not trust the shipped sbatch to still carry the tokens that make the run a
mixed one. `tests/unit/test_sbatch_overrides.py` proves the tokens *compose*; it cannot
notice that `negative_fraction` was dropped altogether, which would produce a green,
successful, positives-only run reporting a negative mix of zero — the exact P2-09
non-delivery this step exists to repair.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tbox_finder.data.embedding import EXCLUDED_DECOY_POOLS, TRAINING_DECOY_POOLS
from tbox_finder.data.negatives import (
    MIX_STREAM_SALT,
    admit_pool_rows,
    negative_draw_count,
    records_from_admitted_rows,
)

REPO = Path(__file__).resolve().parents[2]
SBATCH = REPO / "slurm" / "p2" / "train_production.sbatch"
STAGE1_YAML = REPO / "conf" / "train" / "stage1.yaml"

#: PRD §9.1's "~10:1" read as a STREAM share (ADR-0005 A7 pin 1): f = 10/(10+1).
SEED_RATIO_NEGATIVES_PER_POSITIVE = 10
A7_NEGATIVE_FRACTION = 10 / 11


# ── pin 1: the mix identity, against hand-derived literals ───────────────────────────
@pytest.mark.parametrize(
    "n_positive,expected_negative,expected_stream",
    [
        # 8,303 = the measured full ADR-0004 D5 training fold P2-09 streamed.
        (8303, 83030, 91333),
        # 7,472 = the P2-06a inner_train fold, in case a later run tunes on it.
        (7472, 74720, 82192),
        (1, 10, 11),
        (0, 0, 0),
    ],
)
def test_the_seed_ratio_emits_exactly_ten_negatives_per_positive(
    n_positive: int, expected_negative: int, expected_stream: int
) -> None:
    """The literals are `10 * n` and `11 * n`, written out rather than computed, so a
    change to the rounding rule or to f cannot move both sides of the comparison."""
    drawn = negative_draw_count(n_positive, A7_NEGATIVE_FRACTION)
    assert drawn == expected_negative
    assert n_positive + drawn == expected_stream
    assert drawn == SEED_RATIO_NEGATIVES_PER_POSITIVE * n_positive


def test_the_fraction_the_sbatch_actually_ships_is_exactly_ten_elevenths() -> None:
    """Read the literal OUT of the sbatch rather than restating it.

    The first version asserted `float("0.9090909090909091") == A7_NEGATIVE_FRACTION` —
    two constants defined in this file, an assertion no change to the sbatch could ever
    break. Parsing the shipped token is what makes the drift detectable: if it ever
    differs from the float nearest 10/11 the draw count stops being integer-exact and the
    gate's identity acquires a silent tolerance.
    """
    match = re.search(r"negative_fraction=([0-9.]+)", SBATCH.read_text())
    assert match is not None, "the sbatch no longer passes negative_fraction"
    shipped = float(match.group(1))
    assert shipped == A7_NEGATIVE_FRACTION
    assert negative_draw_count(8303, shipped) == 83030


# ── the shipped launcher actually asks for the mix ───────────────────────────────────
def test_the_production_sbatch_ships_every_token_the_mix_needs() -> None:
    text = SBATCH.read_text()
    for token in (
        "negative_pool_parquet=data/processed/negatives/mining_pool_v0.parquet",
        "negative_decoy_parquet=data/processed/negatives/decoys_v0.parquet",
        "negative_fraction=0.9090909090909091",
        # A7 pin: round 0 is COLD. Asserted, not assumed — a warm start here would make the
        # seed ratio a property of a checkpoint that never saw one.
        "init_from_checkpoint=null",
    ):
        assert token in text, f"{SBATCH.name} no longer passes {token!r}"


def test_the_production_sbatch_stages_both_negative_artifacts() -> None:
    """Both pools are DVC-tracked; an unstaged one surfaces as '0 injectable negatives'
    only AFTER the queue wait."""
    text = SBATCH.read_text()
    assert "assert_dvc_artifact data/processed/negatives/mining_pool_v0.parquet" in text
    assert "assert_dvc_artifact data/processed/negatives/decoys_v0.parquet" in text


def test_the_production_sbatch_walltime_covers_the_eleven_fold_stream() -> None:
    """f = 10/11 with `steps_per_epoch: null` makes the epoch stream 11x longer (1,290 ->
    14,270 steps). The P2-09 header asked for 1 h, which would kill this run mid-training."""
    match = re.search(r"^#SBATCH --time=(\d+):", SBATCH.read_text(), re.M)
    assert match is not None, "the sbatch has no --time header"
    assert int(match.group(1)) >= 3, "wall-time is below the ~2.2-2.6 h estimate for f=10/11"


def test_every_negative_config_key_exists_in_the_yaml() -> None:
    """A dataclass field with no YAML key raises `Key ... is not in struct` at Hydra
    compose time — it killed SLURM job 669 two minutes into a 14 h 36 m queue wait, and
    every dict-level unit test stayed green because they sit BELOW Hydra."""
    text = STAGE1_YAML.read_text()
    for key in (
        "negative_pool_parquet",
        "negative_fraction",
        "negative_max_records",
        "negative_decoy_parquet",
        "init_from_checkpoint",
    ):
        assert re.search(rf"^{key}:", text, re.M), f"conf/train/stage1.yaml has no {key!r} key"


# ── pins 3-5: which classes take a share ─────────────────────────────────────────────
def test_the_embedded_and_excluded_pools_partition_the_four_prd_classes() -> None:
    """All four §9.1 classes must be accounted for. A class in neither tuple would be
    silently absent from the mix with nothing recording that it was a decision."""
    prd_classes = {"gc_background", "structured_rna", "dinuc_shuffled", "leader_decoy"}
    assert set(TRAINING_DECOY_POOLS) | set(EXCLUDED_DECOY_POOLS) == prd_classes
    assert not set(TRAINING_DECOY_POOLS) & set(EXCLUDED_DECOY_POOLS)


def test_every_excluded_pool_carries_its_reason() -> None:
    """'Excluded' with no reason is indistinguishable from 'forgotten'."""
    for pool, reason in EXCLUDED_DECOY_POOLS.items():
        assert isinstance(reason, str) and len(reason) > 40, pool


# ── the shared admission filter (promote, don't duplicate) ───────────────────────────
def _rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f" rec{i}:lead ",
            "sequence": "acgt" * 16,
            "source_record_id": f" rec{i} ",
            "masked": False,
            "is_designed_control": False,
            "parent_nested_train": True,
        }
        for i in range(4)
    ]


def test_admitted_rows_are_normalised_so_both_consumers_see_the_same_bytes() -> None:
    """The embedding hosts its decoys on these rows; the plain arm builds records from
    them. If one saw ' rec0:lead ' and the other 'rec0:lead' the two arms would disagree
    about which window a negative came from."""
    admitted, report = admit_pool_rows(_rows(), window=64)
    assert report["n_records"] == 4
    for row in admitted:
        assert row["candidate_id"] == row["candidate_id"].strip()
        assert row["source_record_id"] == row["source_record_id"].strip()
        assert row["sequence"] == row["sequence"].upper()


def test_records_from_admitted_rows_continue_a_given_cluster_namespace() -> None:
    admitted, _ = admit_pool_rows(_rows(), window=64)
    first = records_from_admitted_rows(admitted, window=64)
    second = records_from_admitted_rows(admitted, window=64, cluster_id_start=len(first))
    assert [r.cluster_id for r in first] == [-1, -2, -3, -4]
    assert [r.cluster_id for r in second] == [-5, -6, -7, -8]
    assert not set(r.cluster_id for r in first) & set(r.cluster_id for r in second)


def test_the_mix_stream_salt_is_distinct_from_the_embedding_salts() -> None:
    """Shared RNG streams correlate across epochs; the embedding's host/phase draw and the
    control's donor draw must each be independent of the mix interleave."""
    from tbox_finder.data.embedding import CONTROL_STREAM_SALT, EMBED_STREAM_SALT

    assert len({MIX_STREAM_SALT, EMBED_STREAM_SALT, CONTROL_STREAM_SALT}) == 3


# ── the fail-closed config pairings (CodeRabbit r1) ──────────────────────────────────
def test_a_decoy_parquet_without_a_pool_is_refused() -> None:
    """`build_stream`'s decoy branch is nested inside the pool branch, so a decoy parquet
    with no pool would be silently ignored and the run would be positives-only while
    reporting `negative_embedding: null` — the §9.1 shortfall this step repairs,
    reintroduced one config key down. It must be unreachable from Hydra, not discovered."""
    from tbox_finder.train.train_stage1 import Stage1TrainConfig

    with pytest.raises(ValueError, match="negative_decoy_parquet is set but"):
        Stage1TrainConfig(negative_decoy_parquet="decoys_v0.parquet")


def test_the_shipped_pairing_is_accepted() -> None:
    """MUST FIRE: if the guard rejected the real configuration too, the test above would
    pass for the wrong reason."""
    from tbox_finder.train.train_stage1 import Stage1TrainConfig

    cfg = Stage1TrainConfig(
        negative_pool_parquet="data/processed/negatives/mining_pool_v0.parquet",
        negative_decoy_parquet="data/processed/negatives/decoys_v0.parquet",
        negative_fraction=A7_NEGATIVE_FRACTION,
    )
    assert cfg.negative_decoy_parquet and cfg.negative_pool_parquet
