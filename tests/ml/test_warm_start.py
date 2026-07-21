"""P2-10d — the warm-start path: a parent checkpoint's weights, byte-verified.

The imp.md gate reads *"a warm-started run reproduces its parent checkpoint's metrics at
step 0"*. That is checked here in the strongest form that is actually decidable: **the
warm-started model's every tensor equals the parent checkpoint's**, and its forward on a
fixed input is **bit-identical to the parent's**. Identical parameters on identical inputs
cannot produce different metrics, and unlike a metric comparison this needs no GPU, no
8-hour queue and no tolerance.

The model here is a real :class:`Stage1Segmenter` — the shipped RC-combine and
segmentation head — over a small deterministic stand-in for the Caduceus backbone. That is
not a mocked pipeline (§8.7): the object under test is ``train_stage1.warm_start`` and the
transport around it, and the pinned 16-layer Mamba stack needs CUDA the laptop tier does
not have. What the stand-in supplies is hidden states; every parameter the warm start
loads, and every parameter the forward consumes, is the real thing.

Three designed controls, because "the weights match" is trivially satisfiable:

* a **fresh** model must NOT match the checkpoint (else the check discriminates nothing);
* the fresh model's forward must DIFFER from the warm-started one (else the loaded weights
  never reach the computation — the state-dict equivalent of a no-op flag);
* ``n_tensors_differing_before`` must be > 0, which is the same control recorded in the
  report so the *gate* carries it, not only this file.

Tiers: torch (`TBOX_REQUIRE_WARM_START_TORCH`) and, when the DVC artifact is present, the
**real P2-09 round-0 production checkpoint** (`TBOX_REQUIRE_WARM_START_CKPT`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
#: The P2-09 round-0 production Stage-1 checkpoint (DVC-tracked, gitignored). Absent in CI
#: and in a fresh worktree; present after `dvc pull`.
PRODUCTION_CKPT = REPO / "data/processed/checkpoints/stage1_production/stage1.pt"

W = 32  # window; the stand-in backbone is O(W), so this stays instant


def _fail_or_skip(var: str, reason: str) -> None:
    """Fail when the tier is explicitly armed, skip otherwise (the repo's arming idiom)."""
    if os.environ.get(var) == "1":
        pytest.fail(f"{var}=1 but the tier is unrunnable: {reason}")
    pytest.skip(reason)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - bare CI
        _fail_or_skip("TBOX_REQUIRE_WARM_START_TORCH", f"torch not importable: {exc}")
    import torch

    return torch


def _stand_in_backbone(torch, *, d_model: int, seed: int):
    """A deterministic, parameterless hidden-state source shaped like Caduceus-PS.

    Caduceus-PS emits ``(B, L, 2*d_model)`` — the two strands concatenated, which is what
    ``RCCombine`` splits. This returns the same shape from a fixed table indexed by token
    id, so hidden states are a pure function of the input and cannot drift between the two
    models a comparison is run over.

    Parameterless on purpose: the warm start must be shown to move the **head** and
    RC-combine parameters, and a stand-in that carried its own would blur which tensors the
    checkpoint actually supplied.
    """

    class _StandIn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("table", torch.randn(16, 2 * d_model, generator=g))

        def forward(self, *, input_ids, **_):  # noqa: ANN001 - mirrors the real backbone
            return self.table[input_ids]

    return _StandIn()


def _segmenter(torch, *, seed: int, rc_combine: str = "concat"):
    from tbox_finder.models.stage1_segmenter import D_MODEL, Stage1Segmenter

    torch.manual_seed(seed)
    return Stage1Segmenter(
        backbone=_stand_in_backbone(torch, d_model=D_MODEL, seed=0),
        rc_combine=rc_combine,
        use_crf=False,
        dropout=0.0,
    )


def _inputs(torch):
    return torch.arange(2 * W, dtype=torch.long).remainder(16).reshape(2, W)


@pytest.fixture()
def parent(tmp_path):
    """A trained-ish parent checkpoint: a segmenter with non-default weights, saved."""
    torch = _require_torch()
    model = _segmenter(torch, seed=1234)
    # Perturb every parameter so the checkpoint is distinguishable from any fresh build —
    # the stand-in for "this ran 1,290 optimiser steps".
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.full_like(p, 0.37))
    path = tmp_path / "stage1.pt"
    torch.save(model.state_dict(), path)
    return model, path


def test_warm_start_makes_every_tensor_equal_to_the_checkpoint(parent) -> None:
    """The gate's core measurement, taken independently of the loader's own report."""
    torch = _require_torch()
    from tbox_finder.train.train_stage1 import warm_start

    _model, path = parent
    fresh = _segmenter(torch, seed=99)
    evidence = warm_start(fresh, path, device="cpu")

    saved = torch.load(path, map_location="cpu", weights_only=True)
    loaded = fresh.state_dict()
    assert set(loaded) == set(saved)
    for name, tensor in saved.items():
        assert torch.equal(loaded[name].cpu(), tensor), name

    assert evidence["n_tensors_differing_after"] == 0
    assert evidence["n_missing_keys"] == 0
    assert evidence["n_unexpected_keys"] == 0
    assert evidence["n_shape_mismatch"] == 0
    assert evidence["n_model_tensors"] == evidence["n_checkpoint_tensors"] == len(saved)
    assert len(evidence["checkpoint_sha256"]) == 64


def test_the_designed_control_a_fresh_model_does_not_match(parent) -> None:
    """Without this, "every tensor matches" could be true of any two models."""
    torch = _require_torch()
    from tbox_finder.train.train_stage1 import _state_digests

    _model, path = parent
    fresh = _segmenter(torch, seed=99)
    saved = _state_digests(torch.load(path, map_location="cpu", weights_only=True))
    before = _state_digests(fresh.state_dict())
    differing = [k for k, v in before.items() if saved.get(k) != v]
    assert differing, "a fresh build already equalled the checkpoint — the check is vacuous"


def test_warm_started_forward_reproduces_the_parent_exactly(parent) -> None:
    """ "Reproduces its parent's metrics at step 0", made decidable.

    Every metric in ``evaluate_selection_val`` is a deterministic function of the model's
    logits on the eval windows. Bit-identical logits on identical inputs therefore give
    bit-identical metrics — and this compares the logits, which is the claim, rather than
    a metric summary that could coincide.
    """
    torch = _require_torch()
    from tbox_finder.train.train_stage1 import warm_start

    model, path = parent
    x = _inputs(torch)
    model.eval()
    with torch.no_grad():
        parent_logits = model(input_ids=x)

    fresh = _segmenter(torch, seed=99)
    fresh.eval()
    with torch.no_grad():
        fresh_logits = fresh(input_ids=x)
    # Control: an un-warm-started model does NOT reproduce the parent, so the assertion
    # below is about the load and not about the stand-in backbone dominating the output.
    assert not torch.equal(fresh_logits, parent_logits)

    warm_start(fresh, path, device="cpu")
    fresh.eval()
    with torch.no_grad():
        warmed_logits = fresh(input_ids=x)
    assert torch.equal(warmed_logits, parent_logits)


def test_warm_start_refuses_a_missing_checkpoint(tmp_path) -> None:
    """A silent fall-back to a fresh build would report a round it never continued."""
    torch = _require_torch()
    from tbox_finder.train.train_stage1 import warm_start

    with pytest.raises(FileNotFoundError, match="does not exist"):
        warm_start(_segmenter(torch, seed=1), tmp_path / "nope.pt", device="cpu")


def test_warm_start_refuses_a_geometry_mismatch(parent) -> None:
    """`gate` RC-combine gives a (8, 256) head; `concat` gives (8, 512). Strict must catch it."""
    torch = _require_torch()
    from tbox_finder.train.train_stage1 import warm_start

    _model, path = parent
    wrong = _segmenter(torch, seed=5, rc_combine="gate")
    with pytest.raises(RuntimeError):
        warm_start(wrong, path, device="cpu")


def test_warm_start_refuses_a_whole_module_pickle(tmp_path) -> None:
    """`torch.save(model)` is not `weights_only` safe and carries no state_dict contract."""
    torch = _require_torch()
    from tbox_finder.train.train_stage1 import warm_start

    path = tmp_path / "notadict.pt"
    torch.save(torch.zeros(3), path)
    with pytest.raises(ValueError, match="not a state_dict"):
        warm_start(_segmenter(torch, seed=1), path, device="cpu")


def test_the_evidence_block_satisfies_the_gate_clause(parent) -> None:
    """The measured block must actually pass `_warm_start_ok` — clause and loader agree."""
    torch = _require_torch()
    from tbox_finder.train.train_stage1 import _warm_start_ok, warm_start

    _model, path = parent
    fresh = _segmenter(torch, seed=99)
    evidence = warm_start(fresh, path, device="cpu")
    report = {
        "warm_start": evidence,
        "diagnostics": {"config": {"init_from_checkpoint": str(path)}},
    }
    assert _warm_start_ok(report) is True


# ── The real P2-09 round-0 production checkpoint ─────────────────────────────────────
def test_the_production_checkpoint_is_a_bare_state_dict_this_module_can_load() -> None:
    """The artifact the first mining round will actually warm-start from.

    Key-level only: the pinned Caduceus backbone is a CUDA-only Mamba stack, so a full
    `Stage1Segmenter` cannot be built here. What is checkable — and what a warm start
    depends on — is that the file is a bare `state_dict` whose head keys and shapes fit the
    shipped `concat`/no-CRF configuration.
    """
    torch = _require_torch()
    if not PRODUCTION_CKPT.is_file():
        _fail_or_skip(
            "TBOX_REQUIRE_WARM_START_CKPT",
            f"{PRODUCTION_CKPT} absent (DVC-tracked; run `dvc pull`)",
        )
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter

    state = torch.load(PRODUCTION_CKPT, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    assert not any(k.startswith("module.") for k in state), "DDP prefix — save used the wrapper"

    skeleton = Stage1Segmenter(backbone=None, rc_combine="concat", use_crf=False)
    head_keys = {k for k in skeleton.state_dict() if not k.startswith("backbone.")}
    assert head_keys, "the skeleton has no head parameters to check against"
    for key in head_keys:
        assert key in state, key
        assert tuple(state[key].shape) == tuple(skeleton.state_dict()[key].shape), key
    assert any(k.startswith("backbone.") for k in state), "no backbone weights to continue from"
