"""P3-08 — the Stage-2 **score producer**: checkpoint back in, per-row logits out.

Two tiers, each armed by its own variable so a tier that cannot run skips loudly and a
tier that is *required* fails rather than skips (the P1-16 landmine):

1. **torch** (``TBOX_REQUIRE_STAGE2_TORCH``) — the round trip through a tiny
   same-architecture RiNALMo: save an adapter + heads exactly the way ``train_stage2``
   does, load them back, and assert the reloaded model scores **identically** to the
   one that was saved. Plus every refusal in :func:`load_stage2_checkpoint`.
2. **committed report** (``TBOX_REQUIRE_AUX_ABLATION``) — validates
   ``reports/stage2_aux_ablation.json`` once a real run has written one.

The refusals are the point of tier 1. ``PeftModel.from_pretrained`` *warns* about
missing adapter keys and PEFT initialises ``lora_B`` to zero, so the failure mode
being guarded against is not a crash — it is a model that loads cleanly, scores every
row, and is arithmetically the untuned backbone. Each refusal below is therefore
sabotage-shaped: a checkpoint corrupted in one specific, *plausible* way, next to the
identical uncorrupted checkpoint succeeding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tbox_finder.stage2 import eval as E
from tbox_finder.stage2 import train as T

_REPO = Path(__file__).resolve().parents[2]
_REPORT = _REPO / E.DEFAULT_REPORT


def _fail_or_skip(var: str, reason: str) -> None:
    if os.environ.get(var):
        pytest.fail(f"{var} is set but {reason}")
    pytest.skip(reason)


def _require_stack():
    if os.environ.get("CUDA_HOME") is None:
        _fail_or_skip("TBOX_REQUIRE_STAGE2_TORCH", "CUDA_HOME unset — multimolecule won't import")
    try:
        import multimolecule  # noqa: F401
        import peft  # noqa: F401
        import safetensors  # noqa: F401
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        _fail_or_skip("TBOX_REQUIRE_STAGE2_TORCH", f"the pinned ml-rna stack is unusable: {exc}")
    return torch


def _tiny_backbone(seed: int = 0):
    """A tiny same-architecture RiNALMo, **deterministically initialised**.

    The seed is load-bearing rather than tidy. A LoRA checkpoint stores adapters and
    heads and *not the backbone*, so "save, rebuild, load, compare" only isolates the
    adapter+head round trip if the two backbones are the same weights. Without the
    seed the round-trip test fails for a reason that has nothing to do with the code
    under test — which is exactly what it did the first time it was run.
    """
    import torch
    from multimolecule import RiNALMoConfig, RiNALMoModel

    cfg = RiNALMoConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128
    )
    cfg._attn_implementation = "sdpa"  # CPU tier; FA-2 needs a GPU
    torch.manual_seed(seed)
    return RiNALMoModel(cfg, add_pooling_layer=False)


def _rows(n: int = 6) -> list[dict[str, Any]]:
    """Deliberately ragged lengths, so batching really pads and really sorts."""
    lengths = [17, 41, 23, 58, 11, 34][:n]
    return [
        {"row_id": f"r{i}", "rna_sequence": "ACGU" * (length // 4) + "A" * (length % 4)}
        for i, length in enumerate(lengths)
    ]


def _write_checkpoint(tmp_path: Path, torch: Any) -> tuple[Path, Any]:
    """A checkpoint written the way ``train_stage2`` writes one; returns ``(dir, model)``.

    Mirrors ``train.py``'s writer exactly — ``backbone.save_pretrained`` for the adapter
    and a flat ``torch.save`` of ``{f"{attr}.{name}": tensor}`` for the heads — so the
    loader is graded against the real on-disk contract rather than one this test made up.
    """
    cfg = T.Stage2TrainConfig(batch_size=2, epochs=1, gradient_checkpointing=False)
    model, _ = T.build_model(cfg, base_model=_tiny_backbone())

    # A freshly-wrapped adapter has lora_B == 0, which is exactly the state the
    # "adapter is live" check must reject. Give it trained-looking weights so the
    # positive control is genuinely positive.
    with torch.no_grad():
        for name, param in model.backbone.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.05)

    ckpt = tmp_path / "arm"
    adapter_dir = ckpt / E.ADAPTER_SUBDIR
    adapter_dir.mkdir(parents=True)
    model.backbone.save_pretrained(str(adapter_dir))
    head_state = {
        f"{attr}.{name}": param.detach().cpu()
        for attr, module in model.head_modules.items()
        for name, param in module.state_dict().items()
    }
    torch.save(head_state, ckpt / E.HEADS_STATE_NAME)
    model.eval()
    return ckpt, model


def _load(ckpt: Path, torch: Any):
    return E.load_stage2_checkpoint(ckpt, base_model=_tiny_backbone(), attn_implementation="sdpa")


# --------------------------------------------------------------------------- #
# Tier 1 — the round trip
# --------------------------------------------------------------------------- #
def test_a_reloaded_checkpoint_scores_identically_to_the_model_that_was_saved(
    tmp_path: Path,
) -> None:
    """The whole contract in one assertion: same weights in, same logits out.

    This is what makes the producer trustworthy. An adapter that half-loaded, a head
    tensor that landed on the wrong module, a dtype that silently changed — every one
    of them moves these logits.
    """
    torch = _require_stack()
    ckpt, original = _write_checkpoint(tmp_path, torch)
    rows = _rows()

    before = E.score_rows(original, rows, batch_size=3, device="cpu")
    reloaded, record = _load(ckpt, torch)
    after = E.score_rows(reloaded, rows, batch_size=3, device="cpu")

    # Guard the guard: if the two backbones were NOT the same weights this comparison
    # would be meaningless, and it would still "pass" for any pair that happened to
    # agree. Assert the premise the seed is there to establish.
    original_base = dict(original.backbone.named_parameters())
    for name, param in reloaded.backbone.named_parameters():
        if "lora_" not in name:
            assert torch.equal(param, original_base[name]), f"base weight {name} differs"

    assert [s["row_id"] for s in before] == [r["row_id"] for r in rows]
    for a, b in zip(before, after, strict=True):
        assert a["row_id"] == b["row_id"]
        assert a["tbox_logit"] == pytest.approx(b["tbox_logit"], abs=1e-5)

    assert record["n_adapter_tensors_matched"] == record["n_adapter_tensors_in_file"] > 0
    assert record["n_adapter_tensors_mismatched"] == 0
    assert record["n_module_adapter_tensors_absent_from_file"] == 0
    assert record["n_lora_b_nonzero"] == record["n_lora_b_tensors"] > 0
    assert record["n_head_tensors_matched"] == record["n_head_tensors_in_file"] > 0


def test_an_all_zero_lora_b_is_refused_because_it_is_the_untuned_backbone(
    tmp_path: Path,
) -> None:
    """The silent failure this check exists for — and it is *not* a crash.

    Zeroing every ``lora_B`` leaves an adapter that loads without complaint and is
    mathematically the identity. Nothing about the resulting model looks wrong; it
    would produce a complete report of the base model's opinions.
    """
    torch = _require_stack()
    from safetensors.torch import load_file, save_file

    ckpt, _ = _write_checkpoint(tmp_path, torch)
    # Positive control FIRST: the untouched checkpoint loads.
    assert _load(ckpt, torch)[1]["n_lora_b_nonzero"] > 0

    adapter_file = ckpt / E.ADAPTER_SUBDIR / "adapter_model.safetensors"
    state = load_file(str(adapter_file))
    # `.clone()` is not defensive habit: load_file can return tensors backed by a
    # memory-map of the very file save_file is about to overwrite, so writing through
    # them can corrupt the weights being written.
    zeroed = {k: (torch.zeros_like(v) if ".lora_B." in k else v.clone()) for k, v in state.items()}
    save_file(zeroed, str(adapter_file), metadata={"format": "pt"})

    with pytest.raises(RuntimeError, match="every lora_B block is zero"):
        _load(ckpt, torch)


def test_an_adapter_tensor_missing_from_the_file_is_refused(tmp_path: Path) -> None:
    """A key the file does not carry never gets visited by a file-driven comparison.

    It stays at PEFT's initialisation — and ``lora_A``'s initialisation is *random*, not
    zero, so the resulting model is neither the trained one nor the base one. Only the
    module→file direction of the check can see this.
    """
    torch = _require_stack()
    from safetensors.torch import load_file, save_file

    ckpt, _ = _write_checkpoint(tmp_path, torch)
    adapter_file = ckpt / E.ADAPTER_SUBDIR / "adapter_model.safetensors"
    state = load_file(str(adapter_file))
    victim = next(k for k in state if ".lora_A." in k)
    save_file(
        {k: v.clone() for k, v in state.items() if k != victim},  # see the clone note above
        str(adapter_file),
        metadata={"format": "pt"},
    )
    with pytest.raises(RuntimeError, match="have no entry in the file"):
        _load(ckpt, torch)


def test_a_head_state_that_disagrees_with_the_spec_is_refused(tmp_path: Path) -> None:
    torch = _require_stack()
    ckpt, _ = _write_checkpoint(tmp_path, torch)
    heads_path = ckpt / E.HEADS_STATE_NAME
    state = torch.load(heads_path, map_location="cpu", weights_only=True)
    dropped = {k: v for k, v in state.items() if not k.startswith("trna_family_head.")}
    torch.save(dropped, heads_path)
    with pytest.raises(ValueError, match="the checkpoint and the head spec disagree"):
        _load(ckpt, torch)

    # Positive control: put it back and the identical path succeeds.
    torch.save(state, heads_path)
    assert _load(ckpt, torch)[1]["n_head_tensors_matched"] > 0


def test_a_head_tensor_of_the_wrong_shape_is_refused_not_warned(tmp_path: Path) -> None:
    """``strict=True``: a shape-mismatched head key raises rather than being skipped."""
    torch = _require_stack()
    ckpt, _ = _write_checkpoint(tmp_path, torch)
    heads_path = ckpt / E.HEADS_STATE_NAME
    state = torch.load(heads_path, map_location="cpu", weights_only=True)
    state["tbox_head.weight"] = state["tbox_head.weight"][:, :-1].contiguous()
    torch.save(state, heads_path)
    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        _load(ckpt, torch)


def test_an_adapter_trained_against_another_backbone_is_refused(tmp_path: Path) -> None:
    """The backbone is not in the checkpoint, so the wrong one applies cleanly and lies."""
    torch = _require_stack()
    ckpt, _ = _write_checkpoint(tmp_path, torch)
    config_path = ckpt / E.ADAPTER_SUBDIR / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = "some-other-org/some-other-rna-lm"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="never saw"):
        _load(ckpt, torch)

    # Positive control: the pinned repo id is accepted, and so is an empty one (a
    # locally-constructed base records no name and must not be refused for it).
    from tbox_finder.train import lora_harness as LH

    config["base_model_name_or_path"] = LH.REPO_ID
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert _load(ckpt, torch)[1]["n_adapter_tensors_matched"] > 0


def test_an_incomplete_checkpoint_directory_is_refused(tmp_path: Path) -> None:
    torch = _require_stack()
    ckpt, _ = _write_checkpoint(tmp_path, torch)
    (ckpt / E.HEADS_STATE_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="is not a Stage-2 checkpoint"):
        _load(ckpt, torch)


# --------------------------------------------------------------------------- #
# Tier 1 — score_rows
# --------------------------------------------------------------------------- #
def test_scores_are_invariant_to_the_caller_s_row_order(tmp_path: Path) -> None:
    """Internal length-sorting must not leak into the result.

    ``score_rows`` batches in ``(n_tokens, row_id)`` order for speed and determinism,
    then restores the caller's order. If the restore were positional rather than by
    ``row_id``, this permutation would silently pair every logit with the wrong row —
    and every downstream number would still look entirely reasonable.
    """
    torch = _require_stack()
    ckpt, _ = _write_checkpoint(tmp_path, torch)
    model, _ = _load(ckpt, torch)
    rows = _rows()

    straight = {s["row_id"]: s["tbox_logit"] for s in E.score_rows(model, rows, batch_size=2)}
    shuffled_rows = list(reversed(rows))
    shuffled = E.score_rows(model, shuffled_rows, batch_size=2)

    assert [s["row_id"] for s in shuffled] == [r["row_id"] for r in shuffled_rows]
    for scored in shuffled:
        assert scored["tbox_logit"] == pytest.approx(straight[scored["row_id"]], abs=1e-5)


def test_padding_does_not_change_a_row_s_score(tmp_path: Path) -> None:
    """Batch size changes who a row is padded against; it must not change its logit."""
    torch = _require_stack()
    ckpt, _ = _write_checkpoint(tmp_path, torch)
    model, _ = _load(ckpt, torch)
    rows = _rows()
    one_at_a_time = {s["row_id"]: s["tbox_logit"] for s in E.score_rows(model, rows, batch_size=1)}
    all_at_once = E.score_rows(model, rows, batch_size=len(rows))
    for scored in all_at_once:
        assert scored["tbox_logit"] == pytest.approx(one_at_a_time[scored["row_id"]], abs=1e-4)


def test_score_rows_refuses_rows_it_cannot_join_or_encode(tmp_path: Path) -> None:
    torch = _require_stack()
    ckpt, _ = _write_checkpoint(tmp_path, torch)
    model, _ = _load(ckpt, torch)
    with pytest.raises(ValueError, match="no row_id"):
        E.score_rows(model, [{"rna_sequence": "ACGU"}])
    with pytest.raises(ValueError, match="is missing or empty"):
        E.score_rows(model, [{"row_id": "x", "rna_sequence": ""}])
    with pytest.raises(ValueError, match="batch_size must be"):
        E.score_rows(model, _rows(2), batch_size=0)
    # Positive control: the same call, well-formed.
    assert len(E.score_rows(model, _rows(2), batch_size=1)) == 2


def test_the_scored_logit_is_the_raw_head_output_not_a_probability(tmp_path: Path) -> None:
    """The calibration stack must receive logits; a pre-squashed score cannot be scaled."""
    torch = _require_stack()
    ckpt, _ = _write_checkpoint(tmp_path, torch)
    model, _ = _load(ckpt, torch)
    rows = _rows()
    scored = E.score_rows(model, rows, batch_size=2)
    import torch as _t

    with _t.inference_mode():
        from tbox_finder.stage2 import tokenizer as TOK

        ids = TOK.encode(rows[0]["rna_sequence"])
        batch = _t.tensor([ids], dtype=_t.long)
        direct = model(input_ids=batch, attention_mask=_t.ones_like(batch))["tbox_logit"]
    assert scored[0]["tbox_logit"] == pytest.approx(float(direct.reshape(-1)[0]), abs=1e-5)
    assert scored[0]["n_tokens"] == len(ids)


# --------------------------------------------------------------------------- #
# Tier 2 — the committed report
# --------------------------------------------------------------------------- #
def test_the_committed_ablation_report_validates() -> None:
    """Armed by its own variable: a green bare-CI run must not imply the run happened."""
    if not _REPORT.is_file():
        _fail_or_skip("TBOX_REQUIRE_AUX_ABLATION", f"{_REPORT} has not been produced yet")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert E.validate_report(report) == []
    assert report["step"] == E.STEP
    # The §7 stop is settled (user sign-off 2026-08-03): the ABSOLUTE reading of D16
    # governs, so the verdict is derived from the with-aux arm's own ECE. The delta
    # tolerance stays unpinned, and `validate_report` still refuses a report that pins one.
    assert report["ablation"]["governing_reading"] == E.GOVERNING_READING
    assert report["ablation"]["verdict"] == E.verdict_from_absolute_reading(
        report["ablation"]["reading_absolute"]["passes"]
    )
    assert report["ablation"]["verdict"] in E.VERDICTS
    assert report["ablation"]["reading_delta"]["tolerance"] is None

    # NOT `overall_pass is True`. The committed run does not pass its own machinery
    # gate, and that is the finding: the no-aux arm's calib carve is perfectly
    # separated, so no temperature exists for it. Asserting a pass here would have
    # forced this file to either lie or be weakened once the real numbers arrived.
    # What is asserted instead is that a False gate is *explained* by its clauses.
    clauses = report["gate"]["clauses"]
    assert report["gate"]["overall_pass"] == all(clauses.values())
    if not report["gate"]["overall_pass"]:
        failing = {name for name, value in clauses.items() if not value}
        assert failing, "the gate is False but every clause is True"
        for arm in report["arms"].values():
            fitted = arm["calibration"]["fitted"]
            graded = arm["grades"][E.GRADE_RUNG]
            # An arm either has a temperature and a real ECE, or neither. A number
            # under the gated key with no fit behind it is the fabrication this whole
            # module is arranged to prevent.
            assert fitted == (graded["ece"] is not None)
            if not fitted:
                assert arm["calibration"]["refusal"]["classification"]
                assert graded["ece_unavailable_reason"]
