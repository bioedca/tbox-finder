"""Unit gate for the P3-03 Stage-2 multi-head model (:mod:`tbox_finder.stage2.model`).

Three tiers, each gated on what the running env actually has (there is no conftest and
no registered marker in this repo — the skip idiom is the try/except+``skipif`` one from
``tests/unit/test_rc_combination.py``):

* **bare** — the sequence-only prohibition, asserted off the *signature*, so it runs even
  where torch is absent;
* **torch** — head shapes, the token→nucleotide alignment, and masked pooling, all on
  synthetic hidden states with ``backbone=None`` (no 2.5 GB download);
* **torch + multimolecule/peft** — a real (tiny, randomly-initialised) ``RiNALMoModel``
  for the end-to-end forward and the LoRA composition.

The alignment tests assert **identity**, not counts. ``strip_special_tokens`` returns a
tensor of the right shape whether or not it removed the right positions, so a count-only
check passes an off-by-one; these encode each position's index into the hidden state and
read it back.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from tbox_finder.stage2 import heads as H

try:
    import torch  # noqa: E402

    from tbox_finder.stage2.model import (  # noqa: E402
        HEAD_OUTPUT_KEYS,
        OUTPUT_KEYS,
        Stage2Model,
        build_stage2_model,
    )

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in the bare CI env
    torch = None
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not installed (bare CI) — Stage-2 model tier runs under tbox-ml-rna",
)

_PDB_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pdb_element_extents"
    / "pdb_element_extents.json"
)


def _pdb_window_lengths() -> list[int]:
    """The nine real T-box window lengths the P3-03 validation gate names.

    The fixture carries element extents, not sequences (``tests/unit/test_stage2_tokenizer.py``
    documents the same limitation), so it is used for the one thing it can supply here:
    real, unequal locus lengths to shape the batch with.
    """
    entries = json.loads(_PDB_FIXTURE.read_text(encoding="utf-8"))["entries"]
    return [int(entry["window_length"]) for entry in entries]


def _spec() -> H.Stage2HeadSpec:
    return H.load_head_spec()


def _multimolecule_or_skip():
    """The ``tests/ml`` idiom: ``import multimolecule`` probes ``CUDA_HOME`` via deepspeed."""
    if os.environ.get("CUDA_HOME") is None:
        pytest.skip("CUDA_HOME unset — multimolecule's transitive deepspeed import needs it")
    return pytest.importorskip("multimolecule")


def _padded_batch(lengths, d_model, *, dtype=None):
    """Synthetic ``(hidden, attention_mask)`` for right-padded ``<cls> … <eos>`` rows.

    ``hidden[b, t, 0] == t`` so every position carries its own token index — that is what
    lets the alignment assertions check identity instead of shape.
    """
    n_special = H_TOKENS
    batch = len(lengths)
    max_tokens = max(lengths) + n_special
    hidden = torch.zeros(batch, max_tokens, d_model, dtype=dtype or torch.float32)
    mask = torch.zeros(batch, max_tokens, dtype=torch.long)
    for row, length in enumerate(lengths):
        used = length + n_special
        hidden[row, :used, 0] = torch.arange(used, dtype=hidden.dtype)
        mask[row, :used] = 1
    return hidden, mask


H_TOKENS = 2  # <cls> … <eos>; mirrors tokenizer.N_FLANKING_SPECIAL_TOKENS


# --------------------------------------------------------------------------- #
# Bare tier — the PRD §6/§10.2 sequence-only prohibition, enforced by the signature
# --------------------------------------------------------------------------- #


class TestSequenceOnlyContract:
    """PRD §10.2: "RiNALMo ingests **sequence only** (no structure-input channel)."

    Read off the shipped file's own bytes with :mod:`ast` rather than by importing it:
    ``model.py`` imports ``torch`` at module top, so an import-based check could only run
    where torch exists — i.e. never in CI, which is the one place this prohibition is
    checked on every push. Parsing the source keeps the guarantee in the bare tier.
    """

    @staticmethod
    def _forward_args():
        source = Path(__file__).resolve().parents[2] / "src" / "tbox_finder" / "stage2" / "model.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        stage2 = next(n for n in classes if n.name == "Stage2Model")
        return next(
            n.args for n in stage2.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )

    def test_forward_accepts_exactly_input_ids_and_attention_mask(self):
        args = self._forward_args()
        assert [a.arg for a in args.args] == ["self"]
        assert [a.arg for a in args.kwonlyargs] == ["input_ids", "attention_mask"]
        assert args.posonlyargs == []

    def test_forward_declares_no_var_kwargs(self):
        # **kwargs would let a structure channel in as an extra keyword and make the
        # prohibition a convention rather than a constraint.
        args = self._forward_args()
        assert args.kwarg is None
        assert args.vararg is None


@requires_torch
class TestSequenceOnlyContractTorch:
    def test_the_output_contract_is_logits_plus_a_mask(self):
        assert (*HEAD_OUTPUT_KEYS, "nucleotide_mask") == OUTPUT_KEYS
        assert len(HEAD_OUTPUT_KEYS) == 6

    def test_passing_a_structure_channel_is_a_type_error(self):
        model = Stage2Model(_spec(), d_model=8)
        ids = torch.ones(1, 5, dtype=torch.long)
        with pytest.raises(TypeError):
            model(input_ids=ids, pairing_dotbracket="((...))")

    def test_no_submodule_is_a_structure_encoder(self):
        model = Stage2Model(_spec(), d_model=8)
        forbidden = ("structure", "dotbracket", "pairing", "wuss", "basepair")
        offenders = [
            name
            for name, _ in model.named_modules()
            if any(token in name.lower() for token in forbidden)
        ]
        assert offenders == []


# --------------------------------------------------------------------------- #
# Torch tier — construction, head shapes, alignment, pooling
# --------------------------------------------------------------------------- #


@requires_torch
class TestConstruction:
    def test_head_widths_come_from_the_spec(self):
        spec = _spec()
        model = Stage2Model(spec, d_model=16)
        assert model.tbox_head.out_features == 1
        assert model.boundary_head.num_classes == spec.size(H.BOUNDARY_FIELD)
        assert model.regulatory_mode_head.out_features == spec.size(H.REGULATORY_MODE_FIELD)
        assert model.specifier_codon_head.out_features == spec.size(H.CODON_FIELD)
        assert model.cognate_aa_head.out_features == spec.size(H.AMINO_ACID_FIELD)
        assert model.trna_family_head.out_features == spec.size(H.TRNA_FAMILY_FIELD)

    def test_boundary_head_reuses_the_stage1_class_order(self):
        model = Stage2Model(_spec(), d_model=16)
        from tbox_finder.labels import CLASS_ORDER

        assert model.boundary_head.class_order == CLASS_ORDER

    def test_default_d_model_is_the_pinned_rinalmo_width(self):
        from tbox_finder.eval.rinalmo_parity import D_MODEL

        assert Stage2Model(_spec()).d_model == D_MODEL

    def test_a_boundary_vocabulary_of_the_wrong_size_is_refused(self):
        spec = H.Stage2HeadSpec(
            cognate_aa_classes=("ILE",),
            trna_family_classes=("ILE (GAU)",),
            boundary_classes=("background", "Stem_I"),
        )
        with pytest.raises(ValueError, match="ADR-0004 D1"):
            Stage2Model(spec, d_model=16)

    def test_forward_without_a_backbone_raises_but_the_heads_still_run(self):
        model = Stage2Model(_spec(), d_model=8)
        with pytest.raises(RuntimeError, match="no backbone"):
            model(input_ids=torch.ones(1, 5, dtype=torch.long))
        hidden, mask = _padded_batch([4], 8)
        assert set(model.heads_from_hidden(hidden, mask)) == set(OUTPUT_KEYS)


@requires_torch
class TestSpecialTokenRemoval:
    def test_the_nucleotide_axis_is_the_token_axis_minus_the_two_specials(self):
        lengths = [7, 3, 5]
        hidden, mask = _padded_batch(lengths, 4)
        nucleotide, nucleotide_mask = Stage2Model.strip_special_tokens(hidden, mask)
        assert nucleotide.shape == (3, max(lengths), 4)
        assert nucleotide_mask.sum(dim=1).tolist() == lengths

    def test_each_kept_position_is_the_right_token(self):
        # IDENTITY, not count: hidden[b, t, 0] == t, so after removing <cls> the kept
        # values must be exactly 1..L_b. A shape-only assertion passes an off-by-one.
        lengths = [7, 3, 5]
        hidden, mask = _padded_batch(lengths, 4)
        nucleotide, nucleotide_mask = Stage2Model.strip_special_tokens(hidden, mask)
        for row, length in enumerate(lengths):
            kept = nucleotide[row, :length, 0].tolist()
            assert kept == [float(i) for i in range(1, length + 1)]

    def test_the_eos_token_is_removed_not_merely_shifted(self):
        # The EOS sits at index L_b in the shifted frame; if it survived it would appear
        # as the value L_b + 1 somewhere in the row.
        lengths = [7, 3, 5]
        hidden, mask = _padded_batch(lengths, 4)
        nucleotide, _ = Stage2Model.strip_special_tokens(hidden, mask)
        for row, length in enumerate(lengths):
            assert float(length + 1) not in nucleotide[row, :, 0].tolist()

    def test_padded_positions_are_zeroed_and_unmasked(self):
        lengths = [7, 3]
        hidden, mask = _padded_batch(lengths, 4)
        nucleotide, nucleotide_mask = Stage2Model.strip_special_tokens(hidden, mask)
        assert nucleotide_mask[1, 3:].sum() == 0
        assert torch.equal(nucleotide[1, 3:, :], torch.zeros_like(nucleotide[1, 3:, :]))

    def test_a_row_with_no_nucleotide_is_refused(self):
        hidden = torch.zeros(1, 4, 4)
        mask = torch.tensor([[1, 1, 0, 0]])  # <cls><eos> only
        with pytest.raises(ValueError, match="cls \\+ >=1 nucleotide"):
            Stage2Model.strip_special_tokens(hidden, mask)

    def test_mismatched_mask_is_refused(self):
        with pytest.raises(ValueError, match="does not match"):
            Stage2Model.strip_special_tokens(
                torch.zeros(2, 6, 4), torch.ones(2, 5, dtype=torch.long)
            )


@requires_torch
class TestPooling:
    def test_padding_does_not_dilute_the_mean(self):
        # The same row, alone and inside a padded batch, must pool to the same vector.
        torch.manual_seed(0)
        lengths = [9, 4]
        hidden, mask = _padded_batch(lengths, 6)
        hidden = hidden + torch.randn_like(hidden) * mask.unsqueeze(-1)
        nucleotide, nucleotide_mask = Stage2Model.strip_special_tokens(hidden, mask)
        batched = Stage2Model.pool(nucleotide, nucleotide_mask)

        used = lengths[1] + H_TOKENS
        solo_hidden = hidden[1:2, :used, :]
        solo_mask = mask[1:2, :used]
        solo_nucleotide, solo_nucleotide_mask = Stage2Model.strip_special_tokens(
            solo_hidden, solo_mask
        )
        solo = Stage2Model.pool(solo_nucleotide, solo_nucleotide_mask)
        torch.testing.assert_close(batched[1], solo[0])

    def test_the_mean_is_over_nucleotides_only(self):
        # Values are 1..L, so the masked mean is (L + 1) / 2 exactly.
        lengths = [7, 3]
        hidden, mask = _padded_batch(lengths, 1)
        nucleotide, nucleotide_mask = Stage2Model.strip_special_tokens(hidden, mask)
        pooled = Stage2Model.pool(nucleotide, nucleotide_mask)
        assert pooled.squeeze(-1).tolist() == pytest.approx([4.0, 2.0])


@requires_torch
class TestHeadShapesOnRealWindowLengths:
    def test_every_head_is_correctly_shaped_at_the_nine_pdb_window_lengths(self):
        spec = _spec()
        lengths = _pdb_window_lengths()
        assert len(lengths) == 9
        assert len(set(lengths)) > 1, "the batch must be ragged or padding is untested"

        model = Stage2Model(spec, d_model=16)
        hidden, mask = _padded_batch(lengths, 16)
        out = model.heads_from_hidden(hidden, mask)

        batch, longest = len(lengths), max(lengths)
        assert set(out) == set(OUTPUT_KEYS)
        assert out["tbox_logit"].shape == (batch,)
        assert out["boundary_logits"].shape == (batch, longest, spec.size(H.BOUNDARY_FIELD))
        assert out["regulatory_mode_logits"].shape == (batch, spec.size(H.REGULATORY_MODE_FIELD))
        assert out["specifier_codon_logits"].shape == (batch, spec.size(H.CODON_FIELD))
        assert out["cognate_aa_logits"].shape == (batch, spec.size(H.AMINO_ACID_FIELD))
        assert out["trna_family_logits"].shape == (batch, spec.size(H.TRNA_FAMILY_FIELD))
        assert out["nucleotide_mask"].shape == (batch, longest)
        assert out["nucleotide_mask"].sum(dim=1).tolist() == lengths

    def test_the_boundary_axis_matches_the_label_string_length(self):
        # Head (b)'s logit axis is what a label_string is compared against position by
        # position, so its length must equal the locus length for every row.
        spec = _spec()
        lengths = _pdb_window_lengths()
        model = Stage2Model(spec, d_model=8)
        hidden, mask = _padded_batch(lengths, 8)
        out = model.heads_from_hidden(hidden, mask)
        for row, length in enumerate(lengths):
            assert int(out["nucleotide_mask"][row].sum()) == length
            targets = spec.encode_boundary_string("." * length)
            assert len(targets) == int(out["nucleotide_mask"][row].sum())

    def test_all_head_logits_are_finite(self):
        torch.manual_seed(0)
        model = Stage2Model(_spec(), d_model=16)
        hidden, mask = _padded_batch([102, 77], 16)
        hidden = hidden + torch.randn_like(hidden) * mask.unsqueeze(-1)
        out = model.heads_from_hidden(hidden, mask)
        for key in HEAD_OUTPUT_KEYS:
            assert torch.isfinite(out[key]).all(), key

    def test_hidden_states_in_another_dtype_are_cast_to_the_heads(self):
        # The encoder runs bf16 (PRD §10.3) while the heads keep their own dtype.
        model = Stage2Model(_spec(), d_model=8)
        hidden, mask = _padded_batch([12, 5], 8, dtype=torch.bfloat16)
        out = model.heads_from_hidden(hidden, mask)
        assert out["tbox_logit"].dtype == model.head_dtype


# --------------------------------------------------------------------------- #
# Torch + multimolecule/peft tier — a real (tiny) RiNALMo and the LoRA composition
# --------------------------------------------------------------------------- #


@requires_torch
class TestAgainstARealRiNALMoEncoder:
    @staticmethod
    def _tiny_backbone():
        multimolecule = _multimolecule_or_skip()
        torch.manual_seed(42)
        config = multimolecule.RiNALMoConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128
        )
        config._attn_implementation = "sdpa"  # CPU tier; FA-2 needs a GPU
        return multimolecule.RiNALMoModel(config, add_pooling_layer=False)

    def test_forward_end_to_end_through_the_real_tokenizer_contract(self):
        from tbox_finder.stage2 import tokenizer as tok

        spec = _spec()
        model = Stage2Model(spec, backbone=self._tiny_backbone())
        assert model.d_model == 64  # measured off the backbone, not defaulted

        sequences = ["ACGUACGUAC", "GGCCAUUGCAUUGGCC"]
        encoded = [tok.encode(s) for s in sequences]
        width = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(encoded), width), tok.PAD_ID, dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        out = model(input_ids=input_ids)  # attention_mask derived from PAD_ID
        assert out["boundary_logits"].shape == (2, max(len(s) for s in sequences), 8)
        assert out["tbox_logit"].shape == (2,)
        assert out["nucleotide_mask"].sum(dim=1).tolist() == [len(s) for s in sequences]

    def test_lora_wraps_the_backbone_and_leaves_every_head_trainable(self):
        pytest.importorskip("peft")
        spec = _spec()
        model, info = build_stage2_model(
            spec,
            base_model=self._tiny_backbone(),
            dtype="float32",
            attn_implementation="sdpa",
            gradient_checkpointing=False,
        )
        stage2 = info["stage2_heads"]
        assert stage2["d_model"] == 64
        assert stage2["head_sizes"] == spec.head_sizes
        # "all-linear" reaches every nn.Linear it can see, so a head inside the wrapper
        # would be LoRA-adapted instead of trained.
        assert stage2["heads_lora_adapted"] == []
        assert stage2["heads_outside_peft_wrapper"] is True
        assert stage2["heads_frozen"] == []
        assert stage2["all_heads_trainable"] is True
        # And the wrap itself really happened.
        assert info["n_adapter_sites"] > 0
        assert info["applied_lora"]["all_linear_fully_covered"] is True
        adapters = [n for n, _ in model.backbone.named_modules() if n.endswith(".lora_A.default")]
        assert adapters, "the backbone carries no LoRA adapter"
        assert not any(name.startswith(head) for head in model.head_modules for name in adapters)
