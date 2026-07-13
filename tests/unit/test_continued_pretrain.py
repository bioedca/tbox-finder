"""P1-11 — unit gate for the probe-triggered GTDB continued-pretraining (MLM) harness.

Two tiers, mirroring ``tests/unit/test_nt_backbone.py`` and ``tests/ml/test_transfer_gonogo.py``:

- **Tier 1 (pure stdlib, runs + BLOCKS in bare CI):** the masking scheme
  (:func:`build_mlm_example` — deterministic, correct rate/label/80-10-10 split, specials
  never masked), the FASTA reader + windower, the report **validator** (every identity- and
  honesty-critical guard has a clean-pass **and** a violating-fail so it is proven to bite,
  §8.7), the conf↔code drift guard, and the **not-auto-run** lock. These import only stdlib +
  the bare ``continued_pretrain`` module (its heavy imports are lazy).
- **Tier 2 (torch; local/cluster only):** the ``-100``-ignored MLM cross-entropy and the full
  ``train_cpt`` loop against a **tiny fake LM** (no Caduceus/GPU needed — the loop structure,
  DDP shim, masking accounting, and checkpoint I/O are exercised on CPU). Gated by
  ``TBOX_REQUIRE_CONTINUED_PRETRAIN`` — fails closed under the env var, skips gracefully
  otherwise (bare CI has no torch, so this tier skips there). The real Caduceus MLM forward is
  validated only by the SLURM pass itself (``slurm/p1/gtdb_continued_pretrain.sbatch``), which
  is probe-triggered and — the P1-07 verdict being GO — not run.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path

import pytest

from tbox_finder.train import continued_pretrain as C

_REPO = Path(__file__).resolve().parents[2]
_CONF_TRAIN = _REPO / "conf" / "train" / "gtdb_cpt.yaml"
_CONF_OPTIM = _REPO / "conf" / "optim" / "adamw_cpt.yaml"
_TRAIN_CONF_DIR = _REPO / "conf" / "train"
_RULES_DIR = _REPO / "workflow" / "rules"

# The Caduceus char-tokenizer id layout (from the pinned checkpoint's config/tokenizer).
_SPECIALS = [0, 1, 2, 3, 4, 5, 6]  # [CLS][SEP][BOS][MASK][PAD][RESERVED][UNK]
_MASK_ID = 3
_BASE_IDS = [7, 8, 9, 10]  # A C G T


# ======================================================================================
# Tier 1 — pure stdlib (runs + blocks in bare CI)
# ======================================================================================
def test_identity_constants() -> None:
    assert C.STEP == "P1-11"
    assert C.OBJECTIVE == "masked_language_modeling"
    assert C.IGNORE_INDEX == -100
    # The re-clear thresholds are single-sourced from the P1-07 grader (ADR-0002 A6).
    from tbox_finder.train import stage1_smoke as S

    assert C.DEFAULT_GO_THRESHOLD == S.DEFAULT_GO_THRESHOLD == 0.30
    assert C.DEFAULT_NOGO_THRESHOLD == S.DEFAULT_NOGO_THRESHOLD == 0.10


def test_module_imports_bare() -> None:
    # The top-of-file `import continued_pretrain` already proves bare-importability (heavy
    # imports are lazy inside functions); assert the pure entry points are present.
    for name in ("run_cpt", "build_mlm_example", "window_sequences", "validate_report"):
        assert callable(getattr(C, name))


def test_validate_masking_bites() -> None:
    C.validate_masking(0.15, 0.8, 0.1)  # canonical → OK
    C.validate_masking(1.0, 1.0, 0.0)  # all-mask edge → OK
    with pytest.raises(ValueError):
        C.validate_masking(0.0, 0.8, 0.1)  # zero rate masks nothing
    with pytest.raises(ValueError):
        C.validate_masking(1.5, 0.8, 0.1)  # rate > 1
    with pytest.raises(ValueError):
        C.validate_masking(0.15, 0.8, 0.5)  # mask+random > 1
    with pytest.raises(ValueError):
        C.validate_masking(0.15, -0.1, 0.1)  # negative mask_replace_prob
    with pytest.raises(ValueError):
        C.validate_masking(0.15, 0.8, 1.5)  # random_replace_prob out of [0, 1]


def test_read_fasta_sequences(tmp_path: Path) -> None:
    p = tmp_path / "shard.fasta"
    p.write_text(">a\nACGTACGT\nNNNN\n>b\nGGGGCCCC\n")
    seqs = C.read_fasta_sequences(p)
    assert seqs == ["ACGTACGTNNNN", "GGGGCCCC"]


def test_read_fasta_gzip(tmp_path: Path) -> None:
    p = tmp_path / "shard.fasta.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(">g\nacgtACGT\n")
    assert C.read_fasta_sequences(p) == ["ACGTACGT"]  # upper-cased


def test_read_fasta_missing_and_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        C.read_fasta_sequences(tmp_path / "nope.fasta")
    empty = tmp_path / "empty.fasta"
    empty.write_text("\n\n")  # no records
    with pytest.raises(ValueError):
        C.read_fasta_sequences(empty)


def test_window_sequences() -> None:
    seq = "A" * 2500
    w = C.window_sequences([seq], window_nt=1000, stride_nt=1000, min_window_nt=128)
    # windows at 0, 1000, 2000 → the tail 2000:3000 is len 500 (>=128) → kept.
    assert [len(x) for x in w] == [1000, 1000, 500]
    # A short trailing chunk below min is dropped.
    w2 = C.window_sequences(["A" * 1050], window_nt=1000, stride_nt=1000, min_window_nt=128)
    assert [len(x) for x in w2] == [1000]  # tail 1000:1050 = 50 < 128 → dropped
    # A whole contig shorter than min is skipped.
    assert C.window_sequences(["ACGT"], window_nt=1000, stride_nt=1000, min_window_nt=128) == []


def test_window_sequences_overlapping_stride() -> None:
    # The PRODUCTION path (stride_nt < window_nt): overlapping windows. A regression advancing
    # by window_nt instead of stride_nt would yield 2 windows here, not 3.
    w = C.window_sequences(["A" * 2000], window_nt=1000, stride_nt=500, min_window_nt=128)
    assert [len(x) for x in w] == [1000, 1000, 1000]  # starts 0, 500, 1000


def test_window_sequences_bad_params() -> None:
    with pytest.raises(ValueError):
        C.window_sequences(["ACGT"], window_nt=0, stride_nt=1, min_window_nt=1)  # window_nt <= 0
    with pytest.raises(ValueError):
        C.window_sequences(["ACGT"], window_nt=10, stride_nt=0, min_window_nt=1)  # stride_nt <= 0
    with pytest.raises(ValueError):
        C.window_sequences(["ACGT"], window_nt=10, stride_nt=5, min_window_nt=0)  # min <= 0
    with pytest.raises(ValueError):
        C.window_sequences(["ACGT"], window_nt=10, stride_nt=5, min_window_nt=11)  # min > window


def test_build_mlm_example_deterministic() -> None:
    ids = [_BASE_IDS[i % 4] for i in range(200)]
    a = C.build_mlm_example(
        ids,
        special_ids=_SPECIALS,
        mask_id=_MASK_ID,
        base_ids=_BASE_IDS,
        mlm_probability=0.15,
        mask_replace_prob=0.8,
        random_replace_prob=0.1,
        rng=random.Random(7),
    )
    b = C.build_mlm_example(
        ids,
        special_ids=_SPECIALS,
        mask_id=_MASK_ID,
        base_ids=_BASE_IDS,
        mlm_probability=0.15,
        mask_replace_prob=0.8,
        random_replace_prob=0.1,
        rng=random.Random(7),
    )
    assert a == b  # same seed ⇒ identical example


def test_build_mlm_example_labels_and_specials() -> None:
    # Interleave specials (which must never be selected) with bases.
    ids = [_MASK_ID, 7, 4, 8, 0, 9, 10, 6]  # positions 0,2,4,7 are specials
    masked, labels = C.build_mlm_example(
        ids,
        special_ids=_SPECIALS,
        mask_id=_MASK_ID,
        base_ids=_BASE_IDS,
        mlm_probability=1.0,
        mask_replace_prob=0.0,
        random_replace_prob=0.0,  # all selected → keep
        rng=random.Random(0),
    )
    assert len(masked) == len(labels) == len(ids)
    for i, tid in enumerate(ids):
        if tid in _SPECIALS:
            assert labels[i] == C.IGNORE_INDEX  # specials never a target
            assert masked[i] == tid
        else:
            assert labels[i] == tid  # non-special, prob=1.0 → selected → label is original
            assert masked[i] == tid  # keep action (both replace probs 0) → unchanged input


def test_build_mlm_example_all_masked_at_prob_one() -> None:
    ids = [_BASE_IDS[i % 4] for i in range(50)]
    masked, labels = C.build_mlm_example(
        ids,
        special_ids=_SPECIALS,
        mask_id=_MASK_ID,
        base_ids=_BASE_IDS,
        mlm_probability=1.0,
        mask_replace_prob=1.0,
        random_replace_prob=0.0,  # all → [MASK]
        rng=random.Random(1),
    )
    assert all(v != C.IGNORE_INDEX for v in labels)  # every base selected
    assert all(m == _MASK_ID for m in masked)  # all replaced with [MASK]
    # The core MLM supervision invariant: the label is the ORIGINAL base, not the [MASK] id —
    # a regression recording the post-masking value would train the model to predict [MASK].
    assert labels == ids


def test_build_mlm_example_80_10_10_split() -> None:
    # A large window: the mask/random/keep split of selected positions ≈ 80/10/10.
    ids = [_BASE_IDS[i % 4] for i in range(20000)]
    masked, labels = C.build_mlm_example(
        ids,
        special_ids=_SPECIALS,
        mask_id=_MASK_ID,
        base_ids=_BASE_IDS,
        mlm_probability=1.0,
        mask_replace_prob=0.8,
        random_replace_prob=0.1,
        rng=random.Random(42),
    )
    selected = [i for i, v in enumerate(labels) if v != C.IGNORE_INDEX]
    assert len(selected) == len(ids)  # prob 1.0 → all selected
    n_mask = sum(1 for i in selected if masked[i] == _MASK_ID)
    n_keep = sum(1 for i in selected if masked[i] == ids[i])
    n_rand = len(selected) - n_mask - n_keep
    n = len(selected)
    assert 0.76 < n_mask / n < 0.84
    assert 0.06 < n_rand / n < 0.14
    assert 0.06 < n_keep / n < 0.14


def test_epoch_order_deterministic_and_hashseed_independent() -> None:
    a = C._epoch_order(100, 42, 3)
    b = C._epoch_order(100, 42, 3)
    assert a == b and sorted(a) == list(range(100))
    assert C._epoch_order(100, 42, 3) != C._epoch_order(100, 42, 4)  # epoch varies order


def test_window_index_stream_disjoint_covers_and_deterministic() -> None:
    # Over one epoch the per-process slices partition range(n): union == all, pairwise disjoint
    # (correct data-parallel coverage) and shuffled (not the identity order).
    n, procs = 20, 4
    per_epoch = n // procs
    slices = []
    for p in range(procs):
        s = C.window_index_stream(n, procs, p, seed=42)
        slices.append([next(s) for _ in range(per_epoch)])
    flat = [i for sl in slices for i in sl]
    assert sorted(flat) == list(range(n))  # covers every window exactly once
    for i in range(procs):
        for j in range(i + 1, procs):
            assert not (set(slices[i]) & set(slices[j]))  # disjoint across processes
    assert flat != list(range(n))  # shuffled, not the identity cycle
    # Deterministic: same (seed, process) ⇒ same first index.
    assert next(C.window_index_stream(n, procs, 1, 42)) == next(
        C.window_index_stream(n, procs, 1, 42)
    )


def test_window_index_stream_never_stalls_when_fewer_windows_than_procs() -> None:
    # n < num_processes: the empty-slice process still yields (no infinite spin).
    s = C.window_index_stream(2, 4, 3, seed=42)
    assert 0 <= next(s) < 2 and 0 <= next(s) < 2


# ------------------------------------------------------------------ report validator
def _valid_report() -> dict:
    cfg = C.CPTConfig()
    return C.build_report(
        cfg=cfg,
        n_records=10,
        n_windows=42,
        n_train_tokens=43008,
        steps=5000,
        final_loss=0.9,
        mean_last_loss=0.95,
        masked_token_count=12345,
        num_processes=4,
        wandb_run_id=None,
    )


def test_valid_report_passes() -> None:
    assert C.validate_report(_valid_report()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema_version="9"),
        lambda r: r.update(step="P1-09"),
        lambda r: r.update(objective="causal_lm"),
        lambda r: r.update(probe_triggered=False),
        lambda r: r["backbone"].update(revision="main"),
        lambda r: r["backbone"].update(repo_id="someone/else"),
        lambda r: r["masking"].update(mlm_probability=0.0),
        lambda r: r["masking"].update(mask_replace_prob=1.5),  # per-field _prob01 guard
        lambda r: r["masking"].update(
            mask_replace_prob=0.8, random_replace_prob=0.1, keep_prob=0.5
        ),
        lambda r: r["masking"].update(ignore_index=0),
        lambda r: r["gonogo_reference"].update(go_threshold=0.5),
        lambda r: r["gonogo_reference"].update(nogo_threshold=0.0),
        lambda r: r["budget"].update(kind="pinned"),
        lambda r: r["train"].update(final_loss=-1.0),
        lambda r: r["train"].update(final_loss=None),
        lambda r: r["train"].update(steps=-1),
        lambda r: r["train"].update(masked_token_count=0),
        lambda r: r["train"].update(masked_token_count=-1),  # non-negative guard
        # block-missing / not-a-mapping guards (honesty against a hand-edited report).
        lambda r: r.pop("backbone"),
        lambda r: r.pop("masking"),
        lambda r: r.pop("gonogo_reference"),
        lambda r: r.pop("budget"),
        lambda r: r.pop("train"),
    ],
)
def test_validator_bites(mutate) -> None:
    r = _valid_report()
    mutate(r)
    assert C.validate_report(r), f"validator failed to bite: {mutate}"


def test_validator_tolerates_unmeasured_report() -> None:
    # An unmeasured report skips the measured-train checks but still enforces identity/honesty.
    cfg = C.CPTConfig()
    r = C.build_report(
        cfg=cfg,
        n_records=0,
        n_windows=0,
        n_train_tokens=0,
        steps=0,
        final_loss=None,
        mean_last_loss=None,
        masked_token_count=0,
        num_processes=1,
        wandb_run_id=None,
        measured=False,
    )
    assert C.validate_report(r) == []  # None final_loss allowed when not measured
    r["probe_triggered"] = False
    assert C.validate_report(r)  # identity/honesty still enforced


# ------------------------------------------------------------------ conf ↔ code drift
def _scan_yaml_scalar(path: Path, key: str) -> str | None:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            value = stripped.split(":", 1)[1]
            return value.split("#", 1)[0].strip() or None
    return None


def test_conf_matches_code_pins() -> None:
    assert _CONF_TRAIN.exists() and _CONF_OPTIM.exists()
    # Numeric pins are compared as floats (0.80 and 0.8 are the same pin; a real drift in the
    # VALUE still bites — this only tolerates trailing-zero formatting).
    assert int(_scan_yaml_scalar(_CONF_TRAIN, "seed")) == 42
    assert float(_scan_yaml_scalar(_CONF_TRAIN, "mlm_probability")) == C.DEFAULT_MLM_PROBABILITY
    assert float(_scan_yaml_scalar(_CONF_TRAIN, "mask_replace_prob")) == C.DEFAULT_MASK_REPLACE_PROB
    assert (
        float(_scan_yaml_scalar(_CONF_TRAIN, "random_replace_prob"))
        == C.DEFAULT_RANDOM_REPLACE_PROB
    )
    assert int(_scan_yaml_scalar(_CONF_TRAIN, "window_nt")) == C.DEFAULT_WINDOW_NT
    assert int(_scan_yaml_scalar(_CONF_TRAIN, "stride_nt")) == C.DEFAULT_STRIDE_NT
    assert int(_scan_yaml_scalar(_CONF_TRAIN, "min_window_nt")) == C.DEFAULT_MIN_WINDOW_NT
    # Path/string pins are compared verbatim.
    assert _scan_yaml_scalar(_CONF_TRAIN, "shard_path") == C.DEFAULT_SHARD
    assert _scan_yaml_scalar(_CONF_TRAIN, "checkpoint_dir") == C.DEFAULT_CHECKPOINT_DIR
    assert _scan_yaml_scalar(_CONF_TRAIN, "report_path") == C.DEFAULT_REPORT
    # The CPT LR is the lower continued-pretraining rate the code defaults to.
    assert float(_scan_yaml_scalar(_CONF_OPTIM, "lr")) == C.CPTConfig().lr == 1.0e-4


def test_conf_is_global_package_with_group_defaults() -> None:
    text = _CONF_TRAIN.read_text()
    assert text.lstrip().startswith("# @package _global_")
    assert "/model: caduceus_stage1" in text
    assert "/optim: adamw_cpt" in text
    assert "/tracking: wandb" in text


def test_cpt_not_wired_as_default_and_not_a_snakemake_target() -> None:
    # Probe-triggered / off-roadmap (ADR-0002 D7): no OTHER training config selects the CPT
    # pass, and it is never a Snakemake `rule all` target (it is sbatch-driven, like P1-07).
    for cfg in _TRAIN_CONF_DIR.glob("*.yaml"):
        if cfg.name == "gtdb_cpt.yaml":
            continue
        for line in cfg.read_text().splitlines():
            no_comment = line.split("#", 1)[0]
            assert (
                "gtdb_cpt" not in no_comment and "continued_pretrain" not in no_comment
            ), f"{cfg.name} references the CPT pass, but it must not be wired as a default"
    if _RULES_DIR.exists():
        for smk in _RULES_DIR.glob("*.smk"):
            for line in smk.read_text().splitlines():
                no_comment = line.split("#", 1)[0]
                assert (
                    "continued_pretrain" not in no_comment and "gtdb_cpt" not in no_comment
                ), f"{smk.name} makes the CPT pass a Snakemake target — it must stay sbatch-only"


# ======================================================================================
# Tier 2 — torch (local/cluster only; the fake-LM loop needs no GPU/Caduceus)
# ======================================================================================
def _fail_or_skip(reason: str) -> None:
    """Fail closed under ``TBOX_REQUIRE_CONTINUED_PRETRAIN=1``; skip gracefully otherwise."""
    if os.environ.get("TBOX_REQUIRE_CONTINUED_PRETRAIN") == "1":
        pytest.fail(f"TBOX_REQUIRE_CONTINUED_PRETRAIN=1 but the torch tier is unrunnable: {reason}")
    pytest.skip(reason)


def _torch_or_skip():
    try:
        import torch  # noqa: F401
    except ImportError:
        _fail_or_skip("torch not installed (activate tbox-ml-dna)")
    import torch

    return torch


def test_mlm_cross_entropy_ignores_pad_and_rewards_correct() -> None:
    torch = _torch_or_skip()
    vocab = 12
    # A confident, correct logit at the one non-ignored position → near-zero loss.
    logits = torch.zeros(1, 3, vocab)
    logits[0, 1, 8] = 20.0  # position 1 strongly predicts class 8
    labels = torch.tensor([[C.IGNORE_INDEX, 8, C.IGNORE_INDEX]])
    loss = C.mlm_cross_entropy(logits, labels)
    assert float(loss) < 1e-3
    # All-ignored → clamped denom → finite 0 (not NaN).
    all_ignore = torch.tensor([[C.IGNORE_INDEX, C.IGNORE_INDEX, C.IGNORE_INDEX]])
    z = C.mlm_cross_entropy(logits, all_ignore)
    assert float(z) == 0.0 and torch.isfinite(z)


def test_mlm_cross_entropy_penalizes_wrong() -> None:
    torch = _torch_or_skip()
    logits = torch.zeros(1, 2, 12)
    logits[0, 0, 7] = 20.0  # predicts 7 but the label is 10 → large loss
    labels = torch.tensor([[10, C.IGNORE_INDEX]])
    assert float(C.mlm_cross_entropy(logits, labels)) > 5.0


class _FakeLM:
    """A tiny stand-in for CaduceusForMaskedLM: emits ``(B, L, V)`` logits from ids."""

    def __init__(self, torch, vocab: int = 12, hidden: int = 16):
        self.torch = torch
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)
        self._mods = torch.nn.ModuleList([self.emb, self.head])

    def parameters(self):
        return self._mods.parameters()

    def train(self):
        self._mods.train()
        return self

    def to(self, device):
        self._mods.to(device)
        return self

    def __call__(self, input_ids=None):
        h = self.emb(input_ids)
        logits = self.head(h)
        return type("Out", (), {"logits": logits})()

    def state_dict(self):
        return self._mods.state_dict()


def test_train_cpt_loop_on_fake_lm(tmp_path: Path) -> None:
    torch = _torch_or_skip()
    windows = ["ACGT" * 32 for _ in range(8)]  # 128-nt windows
    model = _FakeLM(torch)
    acc = C._SingleProcessAccelerator("cpu")

    def ids_of(seq: str) -> list[int]:
        table = {"A": 7, "C": 8, "G": 9, "T": 10}
        return [table[c] for c in seq]

    cfg = C.CPTConfig(max_steps=20, lr=1e-2, log_every=10, grad_clip=1.0)
    final, mean_last, masked, steps = C.train_cpt(
        model, acc, windows, ids_of, _SPECIALS, _MASK_ID, _BASE_IDS, cfg, log=lambda *_: None
    )
    import math

    assert steps == 20
    assert masked > 0  # something was masked
    assert final is not None and math.isfinite(final) and final >= 0.0
    assert final < 10.0  # loss stays bounded (no divergence) over the smoke loop
    assert mean_last is not None and math.isfinite(mean_last)


def test_lr_scheduler_warmup_then_decay() -> None:
    torch = _torch_or_skip()
    lin = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(lin.parameters(), lr=1.0)
    cfg = C.CPTConfig(max_steps=100, warmup_steps=10)
    sched = C._build_lr_scheduler(opt, cfg)
    lr0 = opt.param_groups[0]["lr"]  # step 0: (0+1)/10 = 0.1 of base
    assert abs(lr0 - 0.1) < 1e-6

    def _step():  # mirror the real loop order (opt.step() before sched.step())
        opt.step()
        sched.step()

    for _ in range(10):  # advance to the warmup peak (step 10 ⇒ full base LR)
        _step()
    assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-6
    for _ in range(80):  # decay well past warmup
        _step()
    assert opt.param_groups[0]["lr"] < 1.0  # decaying toward 0


def test_write_outputs_roundtrip(tmp_path: Path) -> None:
    torch = _torch_or_skip()
    shard = tmp_path / "shard.fasta"
    shard.write_text(">x\nACGTACGT\n")  # a real input so provenance can hash it
    cfg = C.CPTConfig(
        shard_path=str(shard),
        checkpoint_dir=str(tmp_path / "ckpt"),
        report_path=str(tmp_path / "gtdb_cpt.json"),
    )
    report = C.build_report(
        cfg=cfg,
        n_records=1,
        n_windows=1,
        n_train_tokens=8,
        steps=5,
        final_loss=0.5,
        mean_last_loss=0.6,
        masked_token_count=3,
        num_processes=1,
        wandb_run_id=None,
    )
    C._write_outputs(cfg, _FakeLM(torch), report, log=lambda *_: None)
    ckpt = tmp_path / "ckpt" / "caduceus_gtdb_cpt.pt"
    prov = tmp_path / "ckpt" / "provenance.json"
    assert ckpt.is_file() and ckpt.stat().st_size > 0
    assert prov.is_file() and prov.stat().st_size > 0
    written = json.loads((tmp_path / "gtdb_cpt.json").read_text())
    assert C.validate_report(written) == []  # the persisted report round-trips valid


def test_single_process_accelerator_shim() -> None:
    torch = _torch_or_skip()
    acc = C._SingleProcessAccelerator("cpu")
    assert acc.is_main_process and acc.num_processes == 1 and acc.process_index == 0
    lin = torch.nn.Linear(3, 3)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    m, o = acc.prepare(lin, opt)
    assert m is lin and o is opt
    x = torch.ones(2, 3, requires_grad=True)
    loss = (m(x) ** 2).mean()
    acc.backward(loss)
    acc.clip_grad_norm_(lin.parameters(), 1.0)
    assert acc.unwrap_model(lin) is lin


# ------------------------------------------------------------------ Caduceus MLM load tier
def test_caduceus_mlm_loader_rejects_non_pinned_revision() -> None:
    # Pure guard: a non-pinned revision is rejected BEFORE the transformers import (RCE guard).
    from tbox_finder.models import caduceus_backbone as cb

    with pytest.raises(ValueError):
        cb.load_caduceus_ps_for_masked_lm(revision="main")
