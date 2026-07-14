"""P1-13 — ML-gate regression for the RiNALMo mirror parity harness (ADR-0002 D5).

Three tiers:

- **Tier 1 (pure stdlib, runs + BLOCKS in bare CI):** the protocol-critical pure
  helpers of :mod:`tbox_finder.eval.rinalmo_parity` (the LinearLR schedule, the
  top-down gradual-unfreeze schedule, the RiNALMo greedy one-pair-per-base decode
  with its canonical + sharp-loop filters, threshold tuning, the ±1-slippage
  scoring), the pinned-revision guard, the config↔code drift guard, AND the
  verdict-core integration (nine synthetic per-fold JSONs → ``aggregate_parity`` →
  PASS / FAIL / telomerase-carveout). Every honesty guard has a clean-pass and a
  violating-fail so it is proven to bite (§8.7).
- **Tier 2 (torch; tiny synthetic model — no 2.5 GB download, no GPU):** the
  ``contact_logits`` shape / symmetry / **row-tiling invariance**, ``reference_matrix``,
  the two-group optimizer (head lr, encoder lr/10), the gradual-unfreeze
  ``requires_grad`` toggling, and that a few head steps reduce the BCE loss.
  Gated by ``TBOX_REQUIRE_PARITY_GATE`` — fails closed under the env var, skips
  gracefully otherwise (CI has no torch, so this tier skips there).
- **Tier 3 (real model + committed report; local/cluster only, gated):** loads
  ``multimolecule/rinalmo-giga`` and overfits one structure (recover self-F1), and
  validates the SLURM-produced ``reports/p1/rinalmo_parity.json`` once it lands.
  Both skip unless ``TBOX_REQUIRE_PARITY_GATE=1`` (they need the 2.5 GB checkpoint /
  the not-yet-produced report).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tbox_finder.eval import archiveii_lofo as A
from tbox_finder.eval import rinalmo_parity as R

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fail_or_skip(reason: str) -> None:
    """Fail closed under ``TBOX_REQUIRE_PARITY_GATE=1``; skip gracefully otherwise."""
    if os.environ.get("TBOX_REQUIRE_PARITY_GATE") == "1":
        pytest.fail(f"TBOX_REQUIRE_PARITY_GATE=1 but the parity torch tier is unrunnable: {reason}")
    pytest.skip(reason)


def _torch_or_skip():
    os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env-dependent
        _fail_or_skip(f"torch not importable ({exc})")
    import torch

    return torch


# ======================================================================== #
# Tier 1 — pure stdlib (CI)
# ======================================================================== #
def test_linear_lr_factor_schedule() -> None:
    assert R.linear_lr_factor(0) == 1.0
    assert abs(R.linear_lr_factor(R.LR_DECAY_STEPS) - R.LR_END_FACTOR) < 1e-12
    assert R.linear_lr_factor(2 * R.LR_DECAY_STEPS) == R.LR_END_FACTOR  # floored after
    mid = R.linear_lr_factor(R.LR_DECAY_STEPS // 2)
    assert R.LR_END_FACTOR < mid < 1.0


def test_unfreeze_schedule_top_down() -> None:
    assert R.unfrozen_layer_indices(0) == []
    assert R.unfrozen_layer_indices(2) == []  # head only for epochs 0..2
    assert R.unfrozen_layer_indices(3) == [30, 31, 32]  # top 3
    assert R.unfrozen_layer_indices(5) == [30, 31, 32]  # unchanged within the 3-epoch window
    assert R.unfrozen_layer_indices(6) == [27, 28, 29, 30, 31, 32]
    # by 15 epochs only the top 12 layers are unfrozen (epochs 0/3/6/9/12 fire)
    assert R.unfrozen_layer_indices(14) == list(range(21, 33))
    # every index is a valid encoder-layer index
    assert all(0 <= i < R.N_LAYERS for i in R.unfrozen_layer_indices(14))


def test_family_to_dir_roundtrip_and_unknown() -> None:
    for dirname, fam in A.FAMILY_DIRS.items():
        assert R._family_to_dir(fam) == dirname
    with pytest.raises(ValueError):
        R._family_to_dir("not_a_family")


def test_normalize_rna() -> None:
    assert R.normalize_rna("acgtACGT") == "ACGUACGU"


def test_decode_canonical_sharploop_greedy() -> None:
    seq = "GGGAAAACCC"  # (1,10)(2,9)(3,8) are GC, |i-j|>=4, canonical
    L = len(seq)
    prob = [[0.0] * L for _ in range(L)]
    for i, j in [(1, 10), (2, 9), (3, 8)]:
        prob[i - 1][j - 1] = prob[j - 1][i - 1] = 0.9
    assert R.decode_pairs(prob, seq, 0.5) == {(1, 10), (2, 9), (3, 8)}

    # sharp loop (|i-j| < 4) is filtered
    sharp = [[0.0] * L for _ in range(L)]
    sharp[0][2] = sharp[2][0] = 0.9  # (1,3), dist 2
    assert R.decode_pairs(sharp, seq, 0.5) == set()

    # non-canonical (G-A) is filtered
    nc = [[0.0] * L for _ in range(L)]
    nc[0][4] = nc[4][0] = 0.9  # (1,5): G-A
    assert R.decode_pairs(nc, seq, 0.5) == set()

    # greedy one-pair-per-base: the higher-probability pair wins the shared base
    comp = [[0.0] * L for _ in range(L)]
    comp[0][9] = comp[9][0] = 0.9  # (1,10)
    comp[0][7] = comp[7][0] = 0.8  # (1,8) shares base 1
    assert R.decode_pairs(comp, seq, 0.5) == {(1, 10)}


def test_score_and_tune_threshold() -> None:
    seq = "GGGAAAACCC"
    L = len(seq)
    ref = ((1, 10), (2, 9), (3, 8))
    prob = [[0.0] * L for _ in range(L)]
    for i, j in ref:
        prob[i - 1][j - 1] = prob[j - 1][i - 1] = 0.9
    preds = [(seq, prob, ref)]
    _p, _r, f = R.score_predictions(preds, 0.5)
    assert f == 1.0
    assert R.score_predictions([], 0.5) == (0.0, 0.0, 0.0)  # empty → 0
    t, best = R.tune_threshold(preds)
    assert best == 1.0 and t <= 0.5


def test_require_pinned_revision_bites() -> None:
    R._require_pinned_revision(R.REVISION)  # clean pass
    with pytest.raises(ValueError):
        R._require_pinned_revision("deadbeef")


# ---- verdict-core integration: synthetic per-fold JSONs → aggregate_parity ----
def _published() -> dict:
    return A.load_published_target(str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET))


def _write_fold(fold_dir: Path, family: str, f1: float) -> None:
    """A synthetic per-fold JSON carrying the SOURCED n_test (so it passes the
    dataset-integrity cross-check) — the shape a real fold job writes."""
    n_test = _published()["families"][family]["n_test"]
    (fold_dir / f"{family}.json").write_text(
        json.dumps(
            {
                "family": family,
                "measured": True,
                "non_weighted_mean_f1": f1,
                "n_test": n_test,
                "tuned_threshold": 0.1,
                "precision_mean": f1,
                "recall_mean": f1,
                "revision": R.REVISION,
                "git_sha": "abc123",
                "seed": R.SEED,
            }
        )
        + "\n"
    )


def test_aggregate_parity_pass(tmp_path: Path) -> None:
    pub = _published()["published_f1"]["per_family"]
    for fam in A.FAMILY_ORDER:
        _write_fold(tmp_path, fam, pub[fam])  # measured == published → PASS
    report = A.aggregate_parity(
        tmp_path,
        target_path=str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET),
        out_path=str(tmp_path / "parity.json"),
    )
    assert report["verdict"] == "PASS"
    assert report["fallback"] is None
    A.validate_parity_report(report)


def test_aggregate_parity_fail_on_stable_family(tmp_path: Path) -> None:
    pub = _published()["published_f1"]["per_family"]
    for fam in A.FAMILY_ORDER:
        _write_fold(tmp_path, fam, pub[fam])
    # knock one STABLE family 5 pp below published (> the ±2 pp margin) → FAIL
    _write_fold(tmp_path, "tRNA", pub["tRNA"] - 0.05)
    report = A.aggregate_parity(
        tmp_path,
        target_path=str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET),
        out_path=str(tmp_path / "parity.json"),
    )
    assert report["verdict"] == "FAIL"
    assert report["fallback"]  # pre-registered Zenodo fallback recorded
    A.validate_parity_report(report)


def test_aggregate_parity_telomerase_carveout(tmp_path: Path) -> None:
    """Telomerase (published 0.12) is carved out — a large swing there does NOT fail."""
    pub = _published()["published_f1"]["per_family"]
    for fam in A.FAMILY_ORDER:
        _write_fold(tmp_path, fam, pub[fam])
    _write_fold(tmp_path, "telomerase_RNA", pub["telomerase_RNA"] + 0.06)  # +6 pp, outside ±2
    report = A.aggregate_parity(
        tmp_path,
        target_path=str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET),
        out_path=str(tmp_path / "parity.json"),
    )
    assert report["verdict"] == "PASS"  # carve-out holds
    assert report["telomerase_carveout"]["gated"] is False


# ---- config ↔ code drift guard ----
def _yaml_scalars(path: Path, keys: set[str]) -> dict:
    """Tiny flat-yaml scalar reader (no PyYAML dep in bare CI)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip()
        if k in keys:
            out[k] = v.split("#", 1)[0].strip()
    return out


def test_config_matches_code_constants() -> None:
    cfg = _yaml_scalars(
        _REPO_ROOT / "conf/model/rinalmo_stage2.yaml",
        {"repo_id", "revision", "hidden_dim", "n_layer", "lr", "epochs", "lr_decay_steps"},
    )
    assert cfg["repo_id"] == R.REPO_ID
    assert cfg["revision"] == R.REVISION
    assert int(cfg["hidden_dim"]) == R.D_MODEL
    assert int(cfg["n_layer"]) == R.N_LAYERS
    assert float(cfg["lr"]) == R.LR
    assert int(cfg["epochs"]) == R.EPOCHS
    assert int(cfg["lr_decay_steps"]) == R.LR_DECAY_STEPS


def _write_pass_folds(tmp_path: Path) -> None:
    pub = _published()["published_f1"]["per_family"]
    for fam in A.FAMILY_ORDER:
        _write_fold(tmp_path, fam, pub[fam])


def test_run_parity_aggregate_entry(tmp_path: Path) -> None:
    """The ``run_parity --mode aggregate`` CLI (the sbatch's aggregate entry) reads the
    nine per-fold JSONs and writes a valid report."""
    _write_pass_folds(tmp_path)
    out = tmp_path / "parity.json"
    rc = A.run_parity(
        [
            "--mode",
            "aggregate",
            "--fold-dir",
            str(tmp_path),
            "--target",
            str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["verdict"] == "PASS"
    A.validate_parity_report(report, target=_published())


def test_module_main_dispatches_to_run_parity(tmp_path: Path) -> None:
    """``python -m tbox_finder.eval.archiveii_lofo --mode aggregate`` must reach
    ``run_parity`` — NOT the P1-12 staging ``main`` (whose argparse rejects ``--mode`` and
    would exit 2). This guards the __main__ dispatch the sbatch depends on."""
    _write_pass_folds(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(_REPO_ROOT / "src"))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tbox_finder.eval.archiveii_lofo",
            "--mode",
            "aggregate",
            "--fold-dir",
            str(tmp_path),
            "--target",
            str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET),
            "--out",
            str(tmp_path / "parity.json"),
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"dispatch failed: {proc.stderr}"
    assert "unrecognized arguments" not in proc.stderr  # would mean it hit staging main()
    assert (tmp_path / "parity.json").is_file()


def test_validate_parity_report_bites(tmp_path: Path) -> None:
    """Every fail-closed raise-path of the committed-report honesty gate is proven to
    bite (§8.4/§8.7/§10.3) — the emitting code never produces these, a tamper would."""
    _write_pass_folds(tmp_path)
    good = A.aggregate_parity(
        tmp_path,
        target_path=str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET),
        out_path=str(tmp_path / "parity.json"),
    )
    A.validate_parity_report(good)  # clean pass

    def _mutate(**over):
        r = json.loads(json.dumps(good))
        r.update(over)
        return r

    with pytest.raises(ValueError):  # unmeasured
        A.validate_parity_report(_mutate(measured=False))
    with pytest.raises(ValueError):  # bad verdict
        A.validate_parity_report(_mutate(verdict="MAYBE"))
    with pytest.raises(ValueError):  # PASS must not carry a fallback
        A.validate_parity_report(_mutate(fallback="something"))
    with pytest.raises(ValueError):  # FAIL must carry a fallback
        A.validate_parity_report(_mutate(verdict="FAIL", fallback=None))
    # per_family missing a family
    short = _mutate()
    short["per_family"].pop("tRNA")
    with pytest.raises(ValueError):
        A.validate_parity_report(short)
    # per-family measured F1 out of [0,1]
    oob = _mutate()
    oob["per_family"]["tRNA"]["measured"] = 1.5
    with pytest.raises(ValueError):
        A.validate_parity_report(oob)
    # summary flag contradicts the per-family table (the recompute guard, finding-2)
    liar = _mutate()
    liar["per_family"]["tRNA"]["measured"] = 0.10  # far below published → stable gate fails
    with pytest.raises(ValueError):
        A.validate_parity_report(liar)


def test_aggregate_rejects_wrong_n_test(tmp_path: Path) -> None:
    """A fold whose n_test != the sourced target's family count (ran on the wrong split)
    is rejected — the dataset-integrity guard (finding-3)."""
    pub = _published()["published_f1"]["per_family"]
    for fam in A.FAMILY_ORDER:
        _write_fold(tmp_path, fam, pub[fam])
    # corrupt tRNA's n_test to a value != the sourced 557
    tpath = tmp_path / "tRNA.json"
    d = json.loads(tpath.read_text())
    d["n_test"] = 999
    tpath.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="n_test"):
        A.aggregate_parity(
            tmp_path,
            target_path=str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET),
            out_path=str(tmp_path / "parity.json"),
        )


def test_aggregate_rejects_mixed_revision(tmp_path: Path) -> None:
    """Folds from different revisions are not a coherent parity measurement (finding-3)."""
    pub = _published()["published_f1"]["per_family"]
    for fam in A.FAMILY_ORDER:
        _write_fold(tmp_path, fam, pub[fam])
    tpath = tmp_path / "tRNA.json"
    d = json.loads(tpath.read_text())
    d["revision"] = "different-revision"
    tpath.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="revision"):
        A.aggregate_parity(
            tmp_path,
            target_path=str(_REPO_ROOT / A.DEFAULT_PUBLISHED_TARGET),
            out_path=str(tmp_path / "parity.json"),
        )


def test_tune_threshold_prefers_lower_on_tie() -> None:
    """When several thresholds achieve the max mean F1, the lower one wins (upstream
    sweeps 0.01→0.29 and keeps the first max)."""
    seq = "GGGAAAACCC"
    L = len(seq)
    ref = ((1, 10), (2, 9), (3, 8))
    prob = [[0.0] * L for _ in range(L)]
    for i, j in ref:
        prob[i - 1][j - 1] = prob[j - 1][i - 1] = 0.9  # perfect at every threshold < 0.9
    preds = [(seq, prob, ref)]
    t, best = R.tune_threshold(preds)
    assert best == 1.0
    assert t == R.THRESHOLDS[0]  # 0.01 — the lowest of the tied maxima


def test_crop_reindexes_pairs() -> None:
    """_crop drops pairs leaving the window and re-indexes the survivors 1-based."""
    seq = "AUGCAUGCAUGC"  # 1..12
    pairs = ((1, 12), (3, 10), (5, 8))
    cropped, new = R._crop(seq, pairs, start=2, length=6)  # keep 3..8
    assert cropped == seq[2:8]
    # (1,12) and (3,10) leave the window → dropped; (5,8) → (3,6) after shifting by start=2
    assert new == ((3, 6),)


# ======================================================================== #
def _tiny_model(torch, d: int = 8, channels: int = 4, n_layers: int = 6, vocab: int = 28):
    """A minimal stand-in exposing the exact attribute paths the harness drives:
    ``model.model.encoder.layer`` / ``.layer_norm`` / ``.__call__`` and
    ``model.ss_head.projection`` / ``.convnet`` / ``.prediction``."""
    import torch.nn as nn

    class _Out:
        def __init__(self, h):
            self.last_hidden_state = h

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
            self.layer_norm = nn.LayerNorm(d)

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(vocab, d)
            self.encoder = _Encoder()

        def forward(self, input_ids):
            h = self.embeddings(input_ids)
            for lyr in self.encoder.layer:
                h = h + lyr(h)
            return _Out(self.encoder.layer_norm(h))

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2 * d, channels)
            self.convnet = nn.Conv2d(channels, channels, kernel_size=1)
            self.prediction = nn.Conv2d(channels, 1, kernel_size=3, padding="same")

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Base()
            self.ss_head = _Head()

    return _Tiny()


def test_contact_logits_shape_symmetry_and_tiling() -> None:
    torch = _torch_or_skip()
    m = _tiny_model(torch)
    L = 7
    ids = torch.randint(5, 28, (1, L + 2))  # [cls, L tokens, eos]
    full = R.contact_logits(m, ids, tile_rows=10_000)
    assert full.shape == (1, L, L)
    # symmetric, zero diagonal
    assert torch.allclose(full, full.transpose(-1, -2), atol=1e-5)
    assert torch.allclose(torch.diagonal(full[0]), torch.zeros(L), atol=1e-6)
    # row-tiling invariance: a small tile gives the SAME logits as one tile
    tiled = R.contact_logits(m, ids, tile_rows=2)
    assert torch.allclose(full, tiled, atol=1e-5)


def test_reference_matrix() -> None:
    torch = _torch_or_skip()
    ref = R.reference_matrix([(1, 5), (2, 4)], 5, device="cpu", dtype=torch.float32)
    assert ref.shape == (1, 5, 5)
    assert ref[0, 0, 4] == 1.0 and ref[0, 4, 0] == 1.0
    assert ref[0, 1, 3] == 1.0 and ref[0, 3, 1] == 1.0
    assert ref.sum().item() == 4.0  # two symmetric pairs


def test_optimizer_two_groups_and_lr_scaling() -> None:
    torch = _torch_or_skip()
    m = _tiny_model(torch)
    opt = R.build_optimizer(m, lr=1e-4)
    assert len(opt.param_groups) == 2
    head_g, enc_g = opt.param_groups
    assert abs(head_g["_base_lr"] - 1e-4) < 1e-12
    assert abs(enc_g["_base_lr"] - 1e-4 / R.UNFREEZE_DENOM) < 1e-12
    R._apply_lr(opt, 0)  # factor 1.0
    assert abs(head_g["lr"] - 1e-4) < 1e-12
    assert abs(enc_g["lr"] - 1e-5) < 1e-12
    R._apply_lr(opt, R.LR_DECAY_STEPS)  # factor 0.1
    assert abs(head_g["lr"] - 1e-5) < 1e-12


def test_apply_unfreeze_toggles_requires_grad(monkeypatch) -> None:
    torch = _torch_or_skip()
    monkeypatch.setattr(R, "N_LAYERS", 6)  # match the tiny model's layer count
    m = _tiny_model(torch, n_layers=6)
    # epoch 0: head trainable, whole encoder frozen
    R.apply_unfreeze(m, 0)
    assert all(p.requires_grad for p in m.ss_head.parameters())
    assert not any(p.requires_grad for p in m.model.parameters())
    # epoch 3: top 3 layers (3,4,5) + final layer_norm unfrozen; layers 0-2 frozen
    idxs = R.apply_unfreeze(m, 3)
    assert idxs == [3, 4, 5]
    assert all(p.requires_grad for p in m.model.encoder.layer[5].parameters())
    assert not any(p.requires_grad for p in m.model.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in m.model.encoder.layer_norm.parameters())


def test_tiny_head_learns() -> None:
    torch = _torch_or_skip()
    import torch.nn as nn

    torch.manual_seed(0)
    m = _tiny_model(torch)
    R.apply_unfreeze(m, 0)  # head only
    opt = R.build_optimizer(m, lr=5e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    L = 8
    ids = torch.randint(5, 28, (1, L + 2))
    ref = R.reference_matrix([(1, 6), (2, 7)], L, device="cpu", dtype=torch.float32)
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    first = last = None
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        logits = R.contact_logits(m, ids)
        loss = loss_fn(logits[:, mask], ref[:, mask])
        loss.backward()
        R._apply_lr(opt, step)
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5  # the trainable head reduces the BCE loss


def test_eval_under_no_grad_builds_no_graph() -> None:
    """After training leaves encoder layers unfrozen, an eval forward under no_grad must
    NOT build an autograd graph (contact_logits respects the ambient grad state — the
    fix for the retained-encoder-graph OOM risk)."""
    torch = _torch_or_skip()
    m = _tiny_model(torch)
    for p in m.model.encoder.layer[0].parameters():  # simulate a post-training unfrozen layer
        p.requires_grad = True
    ids = torch.randint(5, 28, (1, 9))
    with torch.no_grad():
        out = R.contact_logits(m, ids)
    assert out.requires_grad is False  # outer no_grad wins → no retained graph at eval
    out_train = R.contact_logits(m, ids)  # ambient grad on + an unfrozen encoder param
    assert out_train.requires_grad is True  # training still builds the graph


# ======================================================================== #
# Tier 3 — real model + committed report (local/cluster only, gated)
# ======================================================================== #
def test_committed_parity_report_valid_if_present() -> None:
    """Once the SLURM run lands reports/p1/rinalmo_parity.json, it must be a
    self-consistent measured report (fail-closed). Skips while absent."""
    path = _REPO_ROOT / A.DEFAULT_PARITY_REPORT
    if not path.is_file():
        _fail_or_skip(f"{path} not produced yet (SLURM parity run pending)")
    report = json.loads(path.read_text())
    target = _published()
    A.validate_parity_report(report, target=target)
    assert report["verdict"] in {"PASS", "FAIL"}


@pytest.mark.skipif(
    os.environ.get("TBOX_REQUIRE_PARITY_GATE") != "1",
    reason="loads the 2.5 GB rinalmo-giga checkpoint; local/cluster only",
)
def test_real_model_overfits_one_structure() -> None:
    torch = _torch_or_skip()
    import torch.nn as nn

    model, tok, device = R.load_rinalmo_ss()
    R.apply_unfreeze(model, 0)
    model.train()
    seq = R.normalize_rna(
        "GGGCGAUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
    )
    ref_pairs = ((1, 72), (2, 71), (3, 70))
    ids = tok(seq, return_tensors="pt")["input_ids"].to(device)
    opt = R.build_optimizer(model, lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=(device != "cpu"))
    loss_fn = nn.BCEWithLogitsLoss()
    mask = torch.triu(torch.ones(len(seq), len(seq), dtype=torch.bool, device=device), diagonal=1)
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.split(":")[0], dtype=torch.float16, enabled=(device != "cpu")
        ):
            logits = R.contact_logits(model, ids)
            ref = R.reference_matrix(ref_pairs, len(seq), device=device, dtype=logits.dtype)
            loss = loss_fn(logits[:, mask], ref[:, mask])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    preds = R._predict_probs(
        model, [type("R", (), {"sequence": seq, "pairs": ref_pairs})()], tok, device
    )
    thr, _ = R.tune_threshold(preds)
    _p, _r, f = R.score_predictions(preds, thr)
    assert f > 0.5  # the real head learns the overfit structure
