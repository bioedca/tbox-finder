"""P2-10a — the Stage-1 scanner's torch surface: end-to-end transport + checkpoint fit.

Two tiers, the ``tests/ml`` convention, both **self-arming** — they run wherever their
prerequisite genuinely exists rather than behind a ``TBOX_REQUIRE_*`` flag nobody ever sets
(this repo has shipped five composition tests that skipped green for two phases that way):

* **Tier 1 — torch, CPU, no CUDA, no corpus, no download.** ``importorskip("torch")``. A stub
  ``nn.Module`` with an *analytic* logit function drives :func:`scan_sequence` end-to-end.
  This is the tier that covers what `tests/unit/test_infer_scan.py` structurally cannot: that
  tier has no torch, so it calls ``reconcile_windows`` directly and therefore never exercises
  ``scan_sequence``'s own wiring — above all the ``seq_len`` it hands the operator, which is
  the one place a pad *could* be let into the reduction.
* **Tier 2 — the committed checkpoint.** Skips only when the DVC-tracked ``stage1.pt`` is
  absent (CI), so it is armed by the artifact's presence, not by a flag. Needs **no** backbone
  and **no** CUDA: ``checkpoint_key_report`` builds the head with ``backbone=None``.

A clean fit is asserted **against a negative control** in both tiers. "0 missing, 0
unexpected" is worth nothing on its own — an empty comparison reports exactly that.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tbox_finder.data.window_dataset import PAD_TOKEN_ID, WINDOW_NT, encode_bases
from tbox_finder.infer.reconcile import NUM_CLASSES
from tbox_finder.infer.scan import (
    DEFAULT_CHECKPOINT,
    ScanError,
    checkpoint_key_report,
    scan_sequence,
)

_REPO = Path(__file__).resolve().parents[2]
_CKPT = _REPO / DEFAULT_CHECKPOINT


def _seq(n: int, *, seed: int = 3) -> str:
    rng = np.random.default_rng(seed)
    return "".join("ACGT"[i] for i in rng.integers(0, 4, size=n))


def _analytic_logit(token_id: int, cls: int) -> float:
    """Same closed form the unit tier pins — deterministic in the token id, class-asymmetric."""
    return ((token_id * 31 + cls * 17) % 13) / 4.0 - 1.5


def _hand_log_softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    denom = math.log(sum(math.exp(x - m) for x in logits)) + m
    return [x - denom for x in logits]


def _stub_model(torch):
    """An ``nn.Module`` whose logits depend only on the token id — no backbone, no CUDA.

    ``Stage1Segmenter.forward`` needs CUDA because the packaged Mamba does; the *transport*
    around it does not, and that is what these tests are about. The stub honours the same
    keyword-only ``input_ids=`` contract the real segmenter has, so a scanner that called it
    positionally would fail here rather than in production.
    """

    class _Stub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[int, ...]] = []
            table = torch.tensor(
                [[_analytic_logit(t, c) for c in range(NUM_CLASSES)] for t in range(16)],
                dtype=torch.float32,
            )
            self.register_buffer("table", table)

        def forward(self, *, input_ids):  # noqa: ANN001 - mirrors Stage1Segmenter
            self.calls.append(tuple(input_ids.shape))
            return self.table[input_ids]

    return _Stub()


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — the transport, end to end
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("seq_len", [150, 1024, 1600, 3000])
def test_scan_sequence_matches_the_hand_computed_closed_form(seq_len):
    """`scan_sequence` end-to-end against arithmetic computed here from first principles.

    Unlike the unit tier this drives the real entry point, so it also pins the ``seq_len``
    handed to the operator. That matters most below one window: a scanner passing ``window``
    instead of ``len(seq)`` would let 874 pad positions into the reduction for a 150-nt
    sequence and still return a perfectly well-formed distribution.
    """
    torch = pytest.importorskip("torch")
    model = _stub_model(torch)
    seq = _seq(seq_len)

    out = scan_sequence(model, seq, device="cpu", batch_size=4)

    assert out.log_probs.shape == (seq_len, NUM_CLASSES), "output must span the SEQUENCE"
    for p in (0, seq_len // 3, seq_len - 1):
        token = int(encode_bases(seq[p])[0])
        expected = _hand_log_softmax([_analytic_logit(token, c) for c in range(NUM_CLASSES)])
        np.testing.assert_allclose(out.log_probs[p], expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.exp(out.log_probs).sum(axis=1), 1.0, rtol=0, atol=1e-6)


def test_a_pad_position_never_reaches_the_reduction():
    """The negative control for the test above: prove PAD *would* have moved the value.

    Without this, "the closed form matches" is compatible with pads being averaged in and
    happening not to matter. Here the pad row is computed explicitly and shown to differ
    from the answer, so the match is evidence that pads were excluded.
    """
    torch = pytest.importorskip("torch")
    seq = _seq(150)
    out = scan_sequence(_stub_model(torch), seq, device="cpu")

    pad_row = _hand_log_softmax([_analytic_logit(PAD_TOKEN_ID, c) for c in range(NUM_CLASSES)])
    tail = out.log_probs[-1]
    assert not np.allclose(tail, pad_row, atol=1e-3), (
        "the last position of a 150-nt sequence sits against 874 pad positions; matching the "
        "PAD distribution means the pad logits were averaged in"
    )
    assert out.zero_flanked.all(), "every position of a sub-window sequence is zero-flanked"


def test_batching_changes_nothing_but_the_number_of_forwards():
    """`batch_size` is a throughput knob. If it moved the result, a scan's answer would
    depend on the machine it ran on."""
    torch = pytest.importorskip("torch")
    seq = _seq(3000)
    small, large = _stub_model(torch), _stub_model(torch)

    a = scan_sequence(small, seq, device="cpu", batch_size=1)
    b = scan_sequence(large, seq, device="cpu", batch_size=64)

    np.testing.assert_array_equal(a.log_probs, b.log_probs)
    assert len(small.calls) > len(large.calls), "the knob must actually have done something"
    assert a.n_windows == b.n_windows


def test_scan_restores_the_mode_it_found_and_scores_in_eval():
    """Nesting safety: the loop restores the OBSERVED mode, which is what makes the trainer's
    outer ``eval()`` survive a per-record call. Restoring a hardcoded ``train()`` would switch
    dropout back on midway through an evaluation."""
    torch = pytest.importorskip("torch")
    seq = _seq(200)

    training = _stub_model(torch)
    training.train()
    scan_sequence(training, seq, device="cpu")
    assert training.training is True, "a training model must be handed back in train mode"

    evaluating = _stub_model(torch)
    evaluating.eval()
    scan_sequence(evaluating, seq, device="cpu")
    assert evaluating.training is False, "an eval model must NOT be switched to train"


def test_a_model_emitting_the_wrong_class_count_is_refused():
    """An 8-class contract, checked. A 4-class head would otherwise reconcile happily into a
    4-column 'posterior' and only surface much later as a shape error."""
    torch = pytest.importorskip("torch")

    class _WrongWidth(torch.nn.Module):
        def forward(self, *, input_ids):  # noqa: ANN001
            return torch.zeros((*input_ids.shape, NUM_CLASSES - 1))

    with pytest.raises(ScanError, match="classes"):
        scan_sequence(_WrongWidth(), _seq(200), device="cpu")


@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_batch_size_is_refused(bad):
    torch = pytest.importorskip("torch")
    with pytest.raises(ScanError, match="batch_size must be positive"):
        scan_sequence(_stub_model(torch), _seq(200), device="cpu", batch_size=bad)


def test_an_empty_window_set_is_named_not_left_to_torch_cat():
    """CodeRabbit r1. `scan_sequence` cannot produce zero windows, but
    `scan_encoded_windows` is public and the trainer calls it directly. Unguarded, the batch
    loop just never runs and `torch.cat([])` raises "expected a non-empty list of Tensors" —
    an error naming neither the caller's mistake nor this module."""
    torch = pytest.importorskip("torch")
    from tbox_finder.infer.scan import scan_encoded_windows

    empty = np.empty((0, WINDOW_NT), dtype=np.int16)
    with pytest.raises(ScanError, match="no windows"):
        scan_encoded_windows(_stub_model(torch), empty, [], 100, device="cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2 — the committed P2-09 checkpoint actually fits the rebuilt architecture
# ═════════════════════════════════════════════════════════════════════════════
_needs_ckpt = pytest.mark.skipif(
    not _CKPT.is_file(),
    reason=f"{DEFAULT_CHECKPOINT} is DVC-tracked and absent here (CI); armed by its presence",
)


@_needs_ckpt
def test_the_committed_checkpoint_fits_the_rebuilt_segmenter():
    """THE P2-10a gate: `stage1.pt` carries no architecture metadata, so this is the only
    thing standing between a wrong geometry guess and a silently-wrong model."""
    pytest.importorskip("torch")
    report = checkpoint_key_report(_CKPT)

    assert report["rc_combine"] == "concat", "the P2-09 shipped config (ADR-0005 D15)"
    assert report["n_head_missing"] == 0, report["head_missing"]
    assert report["n_head_unexpected"] == 0, report["head_unexpected"]
    assert report["n_shape_mismatch"] == 0, report["shape_mismatch"]
    assert report["fits"] is True
    # Non-vacuity: a comparison over an empty key set would satisfy every line above.
    assert report["n_checkpoint_keys"] > 0


@_needs_ckpt
def test_the_wrong_architecture_guess_is_detected():
    """Guard the guard, against the MEASURED behaviour rather than an assumed one.

    An earlier draft of this test asserted that `concat` and `gate` "produce the same key
    names, so only the head weight's shape differs". That is false — `RCCombine` registers
    `gate_logit` only in gate mode — and the negative-control run printed the contradicting
    `rc_combine.gate_logit` while the claim was being written. Both signals are pinned here
    by exact count, so neither can quietly change.
    """
    pytest.importorskip("torch")

    wrong_mode = checkpoint_key_report(_CKPT, rc_combine="gate")
    assert wrong_mode["fits"] is False
    assert wrong_mode["head_missing"] == ["rc_combine.gate_logit"], (
        "gate mode registers a parameter concat does not, so this mismatch IS visible to a "
        "key-set check"
    )
    assert wrong_mode["n_shape_mismatch"] == 1, wrong_mode["shape_mismatch"]
    assert "head.classifier.weight" in wrong_mode["shape_mismatch"][0]

    wrong_head = checkpoint_key_report(_CKPT, use_crf=True)
    assert wrong_head["fits"] is False
    assert wrong_head["n_head_missing"] == 3, wrong_head["head_missing"]
    assert all(k.startswith("head.crf.") for k in wrong_head["head_missing"])
    assert wrong_head["n_shape_mismatch"] == 0, "a CRF head ADDS params; it reshapes none"


def _head_only_checkpoint(torch, tmp_path, *, source):
    """Write a head-only state_dict so the loader's guard can be tested one branch at a time.

    The point of the indirection: passing the FULL checkpoint alongside a stand-in backbone
    dumps all 307 `backbone.*` keys into `unexpected_keys`, which makes the loader raise no
    matter what the head geometry is — so `use_crf` becomes inert and the `missing` half of
    the guard goes untested. Stripping the backbone keys leaves exactly the parameters the
    architecture guess controls, so each branch can fail on its own merits.
    """
    state = torch.load(source, map_location="cpu", weights_only=True)
    head_only = {k: v for k, v in state.items() if not k.startswith("backbone.")}
    assert head_only, "the checkpoint must carry head parameters at all"
    assert len(head_only) < len(state), "the strip must have removed something"
    path = tmp_path / "head_only.pt"
    torch.save(head_only, path)
    return path


@_needs_ckpt
def test_load_accepts_the_checkpoint_when_the_geometry_is_right(tmp_path):
    """The positive path: a correct guess loads cleanly and comes back in eval mode.

    Without this, every assertion about the loader is a refusal, and a loader that raised
    unconditionally would satisfy all of them.
    """
    torch = pytest.importorskip("torch")
    from tbox_finder.infer.scan import load_stage1_checkpoint

    ckpt = _head_only_checkpoint(torch, tmp_path, source=_CKPT)
    model = load_stage1_checkpoint(ckpt, device="cpu", backbone=object())

    assert model.training is False, "a scanner must hand back a model in eval mode"
    assert model.rc_combine_mode == "concat"
    # The real trained weights actually landed — not a freshly initialised head.
    loaded = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert torch.equal(model.head.classifier.weight.detach(), loaded["head.classifier.weight"])


@_needs_ckpt
@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        ({"use_crf": True}, "missing"),
        ({"rc_combine": "gate"}, "shape"),
    ],
)
def test_load_refuses_each_wrong_geometry_on_its_own_merits(tmp_path, kwargs, expect):
    """Both refusal branches, isolated — neither is over-determined by a stand-in backbone.

    `use_crf=True` must fail on MISSING keys (a CRF head adds parameters); `rc_combine="gate"`
    must fail inside `load_state_dict` on a SHAPE mismatch, which torch raises even at
    `strict=False` (context7 /pytorch/pytorch). If the `missing` term were dropped from the
    loader's guard, the first parametrisation would go green.
    """
    torch = pytest.importorskip("torch")
    from tbox_finder.infer.scan import load_stage1_checkpoint

    ckpt = _head_only_checkpoint(torch, tmp_path, source=_CKPT)
    with pytest.raises((ScanError, RuntimeError)) as excinfo:
        load_stage1_checkpoint(ckpt, device="cpu", backbone=object(), **kwargs)

    message = str(excinfo.value)
    if expect == "missing":
        assert isinstance(excinfo.value, ScanError), message
        assert "does not fit" in message and "head.crf." in message
    else:
        assert "size mismatch" in message or "shape" in message, message


def test_a_missing_checkpoint_is_named_not_swallowed():
    pytest.importorskip("torch")
    with pytest.raises(ScanError, match="no Stage-1 checkpoint"):
        checkpoint_key_report(_REPO / "data" / "processed" / "checkpoints" / "nope.pt")
