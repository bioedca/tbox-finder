"""Unit gate for the P3-05 optional structure-consistency loss (imp.md P3-05).

The gate imp.md names is two claims: *"pairing-partner target well-formed from the
dot-bracket"* and *"loss decreases on the hand-checked fixture"*. Both are here, plus the
refusals that make "well-formed" mean something — a target check that only ever sees
well-formed input proves nothing.

Tiers follow ``test_stage2_losses.py``'s idiom (no conftest, no registered marker in this
repo; ``exc.name != "torch"`` re-raised so a broken sibling cannot self-skip the tensor
tier green):

* **bare** — the target encoding, the three sentinels, the pseudoknot refusal, the config
  toggle and the dominance-cap arithmetic. All pure, so all of it runs in CI.
* **bare + hydra** — the two new YAML keys, composed for real (the job-669 struct-mode trap).
* **torch** — the head's shape/symmetry/masking contract, the term graded against torch's
  own ``F.cross_entropy``, the refusals, and the learning check.

Conventions carried from the P3-04 gate: fp64 with ``atol=1e-12`` for exact arithmetic, and
grading against an **external** reference rather than a recomputation of the implementation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tbox_finder.stage2 import dataset as DS
from tbox_finder.stage2 import heads as H
from tbox_finder.stage2 import losses as L

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in the bare CI env
    if exc.name != "torch":
        raise
    torch = None
    F = None
    PairingHead = Stage2Model = None
    ALL_HEAD_OUTPUT_KEYS = OPTIONAL_HEAD_OUTPUT_KEYS = STRUCTURE_OUTPUT_KEY = None
    _HAS_TORCH = False
else:
    # Outside the guard on purpose, naming the torch-tier siblings this file grades
    # against: a broken `stage2.model` must raise HERE rather than be swallowed into
    # `_HAS_TORCH = False` and skip the whole tensor tier green (the P1-15 failure mode).
    import torch.nn.functional as F

    from tbox_finder.stage2.model import (
        ALL_HEAD_OUTPUT_KEYS,
        OPTIONAL_HEAD_OUTPUT_KEYS,
        STRUCTURE_OUTPUT_KEY,
        PairingHead,
        Stage2Model,
    )

    _HAS_TORCH = True

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not installed (bare CI) — head (e) tensor tier runs under tbox-ml-rna",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"
LOSS_CONF = CONF_DIR / "loss" / "stage2.yaml"

#: One hand-checked fixture, used by the encoding tests and the learning test alike so the
#: two cannot drift. ``((..))`` — 0 pairs 5, 1 pairs 4, 2 and 3 unpaired.
HAND_DOT_BRACKET = "((..))"
#: Worked by hand from :data:`UNPAIRED_CLASS` = 0 and "class j+1 = paired with j":
#: 0→5 ⇒ 6, 1→4 ⇒ 5, 2 unpaired ⇒ 0, 3 unpaired ⇒ 0, 4→1 ⇒ 2, 5→0 ⇒ 1.
HAND_TARGET = [6, 5, 0, 0, 2, 1]


# ====================================================================================== #
# Bare tier — the target encoding (imp.md gate 1: "well-formed from the dot-bracket")
# ====================================================================================== #
class TestPairingTargetEncoding:
    def test_the_hand_checked_fixture_encodes_as_worked_out_by_hand(self):
        assert L.encode_pairing_target(HAND_DOT_BRACKET, length=6) == HAND_TARGET

    def test_the_encoding_is_the_shifted_partner_index_of_the_shared_parser(self):
        """Graded against ``dataset.dot_bracket_to_partners``, not a second parser here."""
        for text in ("((..))", "....", "(((...)))", ".((.)).", "(.)(.)"):
            partners = DS.dot_bracket_to_partners(text)
            encoded = L.encode_pairing_target(text, length=len(text))
            for i, partner in enumerate(partners):
                if partner == DS.UNPAIRED_PARTNER:
                    assert encoded[i] == L.UNPAIRED_CLASS
                else:
                    assert encoded[i] == partner + 1

    def test_unpaired_is_a_supervised_class_and_not_the_ignore_sentinel(self):
        """The trap: -1 means "supervise as unpaired", -100 means "teaches nothing"."""
        encoded = L.encode_pairing_target("....", length=4)
        assert encoded == [L.UNPAIRED_CLASS] * 4
        assert L.UNPAIRED_CLASS != L.IGNORE_INDEX
        assert L.UNPAIRED_CLASS != DS.UNPAIRED_PARTNER
        assert all(value >= 0 for value in encoded)

    def test_the_unpaired_class_is_first_so_a_target_does_not_depend_on_padding(self):
        """A row encoded alone and inside a longer batch must carry identical classes."""
        assert L.UNPAIRED_CLASS == 0
        alone = L.encode_pairing_target(HAND_DOT_BRACKET, length=6)
        in_batch = alone + [L.IGNORE_INDEX] * 14  # what a collator pads to L_pad = 20
        assert in_batch[:6] == alone

    @pytest.mark.parametrize("missing", [None, float("nan")])
    def test_a_row_with_no_dot_bracket_is_all_ignore(self, missing):
        """12,273 of 30,542 rows: 7,007 decoys + 5,266 unanchorable positives."""
        assert L.encode_pairing_target(missing, length=5) == [L.IGNORE_INDEX] * 5

    def test_a_length_mismatch_is_refused_rather_than_truncated(self):
        with pytest.raises(ValueError, match="misaligned"):
            L.encode_pairing_target("((..))", length=7)
        # positive control: the identical string at its own length succeeds
        assert L.encode_pairing_target("((..))", length=6) == HAND_TARGET

    @pytest.mark.parametrize("glyph", ["[", "]", "{", "}", "<", ">", "A", "a"])
    def test_a_pseudoknot_bracket_is_refused_not_silently_read_as_unpaired(self, glyph):
        """PRD §8: crossing pairs cannot be encoded in the nested dot-bracket.

        The corpus carries none, so this can only fire on a source change — and it must
        fire loudly rather than mapping an unknown glyph onto "unpaired".
        """
        with pytest.raises(ValueError, match="unexpected dot-bracket glyph"):
            L.encode_pairing_target(f"(.{glyph}.)", length=5)

    def test_an_unbalanced_dot_bracket_is_refused(self):
        with pytest.raises(ValueError, match="unbalanced|unexpected"):
            L.encode_pairing_target("((..)", length=5)

    def test_the_encoded_target_is_symmetric_and_self_free(self):
        for text in ("((..))", "(((...)))", ".((.)).", "(.)(.)"):
            encoded = L.encode_pairing_target(text, length=len(text))
            for i, cls in enumerate(encoded):
                if cls == L.UNPAIRED_CLASS:
                    continue
                partner = cls - 1
                assert partner != i, "a base cannot pair with itself"
                assert encoded[partner] == i + 1, "pairing must be symmetric"


# ====================================================================================== #
# Bare tier — the optional term's place in the objective
# ====================================================================================== #
class TestOptionalTermInventory:
    def test_structure_is_the_only_optional_term(self):
        assert L.OPTIONAL_TERMS == (L.STRUCTURE_TERM,)
        assert L.STRUCTURE_TERM in L.AUX_TERMS
        assert L.STRUCTURE_TERM not in (L.BINARY_TERM,)

    def test_it_is_per_nucleotide_and_last(self):
        assert L.STRUCTURE_TERM in L.PER_NUCLEOTIDE_TERMS
        assert L.AUX_TERMS[-1] == L.STRUCTURE_TERM
        assert L.TERMS[0] == L.BINARY_TERM

    def test_it_has_no_head_vocabulary_and_says_so(self):
        """Its classes are positions, so ``Stage2HeadSpec`` has nothing to hold."""
        assert L.NON_VOCAB_AUX_TERMS == (L.STRUCTURE_TERM,)
        assert L.STRUCTURE_TERM not in L.TERM_TO_FIELD
        assert set(L.TERM_TO_FIELD) == {t for t in L.AUX_TERMS if t != L.STRUCTURE_TERM}
        assert tuple(L.TERM_TO_FIELD[t] for t in L.AUX_TERMS if t in L.TERM_TO_FIELD) == (
            H.VOCAB_FIELDS
        )

    def test_it_reads_its_own_logit_key(self):
        assert L.TERM_TO_LOGITS[L.STRUCTURE_TERM] == "structure_logits"
        assert len(set(L.TERM_TO_LOGITS.values())) == len(L.TERMS)


class TestTheToggle:
    def test_the_shipped_default_is_off_so_the_default_objective_is_d16s_six_terms(self):
        """ADR-0005 D16 enumerates six heads; PRD §8/§11 call this one optional."""
        cfg = L.Stage2LossConfig()
        assert cfg.structure_enabled is False
        assert cfg.is_enabled(L.STRUCTURE_TERM) is False
        assert cfg.active_terms() == tuple(t for t in L.TERMS if t != L.STRUCTURE_TERM)
        assert len(cfg.active_terms()) == 6

    def test_enabling_it_makes_seven(self):
        cfg = L.Stage2LossConfig(structure_enabled=True)
        assert cfg.active_terms() == L.TERMS
        assert len(cfg.active_terms()) == 7

    def test_every_other_term_is_mandatory(self):
        cfg = L.Stage2LossConfig()
        for term in L.TERMS:
            if term == L.STRUCTURE_TERM:
                continue
            assert cfg.is_enabled(term) is True

    def test_an_unknown_term_is_refused_by_the_toggle_too(self):
        with pytest.raises(ValueError, match="unknown term"):
            L.Stage2LossConfig().is_enabled("secondary_structure")

    def test_a_disabled_term_leaves_the_dominance_sum_entirely(self):
        """Not merely zero-weighted: it must not tighten the D16 cap it does not use."""
        off = L.Stage2LossConfig()
        on = L.Stage2LossConfig(structure_enabled=True)
        assert off.base_weight(L.STRUCTURE_TERM) == 0.0
        assert on.base_weight(L.STRUCTURE_TERM) == on.structure_weight
        assert off.dominance()["aux_total"] == pytest.approx(0.7)
        assert on.dominance()["aux_total"] == pytest.approx(0.8)

    def test_enabling_it_lowers_the_d16_cap_from_1_4286_to_1_25(self):
        """The number P3-04's hand-off predicted, measured rather than restated."""
        assert L.Stage2LossConfig().max_aux_weight == pytest.approx(1.0 / 0.7)
        assert L.Stage2LossConfig(structure_enabled=True).max_aux_weight == pytest.approx(1.25)

    def test_the_cap_is_enforced_at_construction_not_reported_after_the_run(self):
        L.Stage2LossConfig(structure_enabled=True, aux_weight=1.25)  # admitted, at the bound
        with pytest.raises(ValueError, match="dominate"):
            L.Stage2LossConfig(structure_enabled=True, aux_weight=1.26)
        # positive control: the same aux_weight is fine with the optional term off
        L.Stage2LossConfig(structure_enabled=False, aux_weight=1.26)

    def test_on_with_zero_weight_is_refused_as_neither_on_nor_off(self):
        with pytest.raises(ValueError, match="structure_enabled=True with structure_weight=0"):
            L.Stage2LossConfig(structure_enabled=True, structure_weight=0.0)
        # positive control: both coherent readings of "off" are accepted
        L.Stage2LossConfig(structure_enabled=False, structure_weight=0.0)
        L.Stage2LossConfig(structure_enabled=True, structure_weight=0.1)

    @pytest.mark.parametrize("bad", ["false", "true", 0, 1, None])
    def test_a_non_bool_toggle_is_refused_rather_than_coerced(self, bad):
        """``structure_enabled="false"`` is truthy: coercion would turn it ON."""
        with pytest.raises(ValueError, match="structure_enabled must be a bool"):
            L.Stage2LossConfig(structure_enabled=bad)

    def test_the_no_aux_arm_still_collapses_to_the_binary_term_with_it_on(self):
        cfg = L.Stage2LossConfig(structure_enabled=True, aux_weight=0.0)
        effective = cfg.effective_weights()
        assert effective[L.STRUCTURE_TERM] == 0.0
        assert all(effective[t] == 0.0 for t in L.AUX_TERMS)
        assert effective[L.BINARY_TERM] == 1.0

    def test_gamma_routes_to_its_own_field(self):
        cfg = L.Stage2LossConfig(structure_gamma=1.5, aux_gamma=0.25, boundary_gamma=2.0)
        assert cfg.gamma(L.STRUCTURE_TERM) == 1.5
        assert cfg.gamma("boundary") == 2.0
        assert cfg.gamma("regulatory_mode") == 0.25

    def test_class_weighting_cannot_reach_a_positional_class_axis(self):
        cfg = L.Stage2LossConfig(structure_enabled=True, aux_class_weight_alpha=0.9)
        assert cfg.class_weight_alpha(L.STRUCTURE_TERM) == 0.0
        assert cfg.class_weight_alpha("cognate_aa") == 0.9

    def test_diagnostics_report_the_toggle_and_do_not_claim_d16_blessed_it(self):
        report = L.Stage2LossConfig(structure_enabled=True).diagnostics()
        assert report["structure_enabled"] is True
        assert report["optional_terms"] == [L.STRUCTURE_TERM]
        assert L.STRUCTURE_TERM in report["active_terms"]
        assert "PRD" in report["pinned"]["structure_term"]
        assert "not enumerated by ADR-0005 D16" in report["pinned"]["structure_term"]

    def test_a_disabled_terms_diagnostics_say_so(self):
        report = L.Stage2LossConfig().diagnostics()
        assert report["structure_enabled"] is False
        assert L.STRUCTURE_TERM not in report["active_terms"]
        assert report["effective_weights"][L.STRUCTURE_TERM] == 0.0


# ====================================================================================== #
# Bare + hydra tier — the two new keys must exist literally (the job-669 trap)
# ====================================================================================== #
class TestHydraKeys:
    @staticmethod
    def _compose(*overrides):
        """The same primary a trainer composes (``test_stage2_losses``'s ``_compose``)."""
        pytest.importorskip("hydra")
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
            return compose(config_name="train/lora_stage2", overrides=list(overrides))

    @pytest.mark.parametrize("field", ["structure_enabled", "structure_weight", "structure_gamma"])
    def test_the_new_fields_are_literal_yaml_keys(self, field):
        """A dataclass field with no YAML key is a hard ConfigCompositionException on
        override — the way SLURM job 669 was lost."""
        text = LOSS_CONF.read_text(encoding="utf-8")
        assert any(line.startswith(f"{field}:") for line in text.splitlines()), field

    def test_the_yaml_defaults_are_the_dataclass_defaults(self):
        text = LOSS_CONF.read_text(encoding="utf-8")
        values = {}
        for line in text.splitlines():
            if line.startswith(("structure_enabled:", "structure_weight:", "structure_gamma:")):
                key, raw = line.split(":", 1)
                raw = raw.split("#")[0].strip()
                values[key] = {"true": True, "false": False}.get(raw, raw)
        defaults = {f.name: f.default for f in dataclasses.fields(L.Stage2LossConfig)}
        assert values["structure_enabled"] == defaults["structure_enabled"] is False
        assert float(values["structure_weight"]) == defaults["structure_weight"]
        assert float(values["structure_gamma"]) == defaults["structure_gamma"]

    def test_the_toggle_is_overridable_as_a_sweep_axis(self):
        cfg = self._compose("loss.structure_enabled=true", "loss.structure_weight=0.25")
        assert cfg.loss.structure_enabled is True
        assert cfg.loss.structure_weight == pytest.approx(0.25)
        built = L.Stage2LossConfig(**dict(cfg.loss))
        assert built.is_enabled(L.STRUCTURE_TERM)

    def test_the_shipped_yaml_builds_a_config_with_the_term_off(self):
        cfg = self._compose()
        built = L.Stage2LossConfig(**dict(cfg.loss))
        assert built.structure_enabled is False
        assert built.active_terms() == tuple(t for t in L.TERMS if t != L.STRUCTURE_TERM)


# ====================================================================================== #
# Torch tier — head (e)'s contract
# ====================================================================================== #
def _hidden(batch=3, length=6, width=8, *, seed=0, ragged=True):
    torch.manual_seed(seed)
    hidden = torch.randn(batch, length, width, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(batch, length, dtype=torch.bool)
    if ragged:
        mask[-1, -2:] = False
    return hidden, mask


@requires_torch
class TestPairingHead:
    def test_the_class_axis_is_one_plus_the_position_axis(self):
        hidden, mask = _hidden()
        logits = PairingHead(8, proj_dim=4).double()(hidden, mask)
        assert tuple(logits.shape) == (3, 6, 7)

    def test_scores_are_symmetric_by_construction(self):
        """Base pairing is a symmetric relation; i→j and j→i must score identically.

        Over the **valid** sub-block: masking padded *columns* is deliberately
        one-directional (a padded position is never a legal partner, but its own query row
        is ignored anyway), so symmetry is asserted where it is claimed and not where it is
        not — the ragged row's valid×valid corner is checked separately below.
        """
        hidden, mask = _hidden()
        pair_block = PairingHead(8, proj_dim=4).double()(hidden, mask)[..., 1:]
        for row in range(pair_block.shape[0]):
            n = int(mask[row].sum())
            block = pair_block[row][:n, :n]
            assert torch.allclose(block, block.T, atol=1e-12), row
        assert int(mask[2].sum()) == 4, "the ragged row must actually be ragged"

    def test_symmetry_survives_the_pooling_of_two_different_projections(self):
        """Left and right are distinct Linears — symmetry is the transpose average, not
        an accident of tied weights."""
        head = PairingHead(8, proj_dim=4).double()
        assert head.left.weight is not head.right.weight
        assert not torch.allclose(head.left.weight, head.right.weight)
        hidden, mask = _hidden(ragged=False)
        block = head(hidden, mask)[0, :, 1:]
        assert torch.allclose(block, block.T, atol=1e-12)

    def test_self_pairing_and_padded_partners_are_forbidden_not_learned(self):
        hidden, mask = _hidden()
        logits = PairingHead(8, proj_dim=4).double()(hidden, mask)
        floor = torch.finfo(logits.dtype).min
        pair_block = logits[..., 1:]
        assert bool((pair_block.diagonal(dim1=1, dim2=2) == floor).all())
        assert bool((pair_block[2][:, 4:] == floor).all())  # row 2's last two are padding
        # ...and the probability mass they receive is exactly zero
        probs = torch.softmax(logits, dim=-1)
        assert float(probs[2][:, 5:].sum()) == pytest.approx(0.0, abs=1e-30)

    def test_column_zero_is_the_unpaired_logit_and_the_rest_are_partners(self):
        """The axis ORDER is the contract ``encode_pairing_target`` writes against.

        Named explicitly because it is otherwise only caught sideways: reversing the
        concatenation turns the symmetry and masking tests red without either of them
        saying *"the unpaired class moved"*.
        """
        head = PairingHead(8, proj_dim=4).double()
        hidden, mask = _hidden(ragged=False)
        logits = head(hidden, mask)
        assert L.UNPAIRED_CLASS == 0
        assert torch.allclose(
            logits[..., L.UNPAIRED_CLASS],
            head.unpaired(hidden).squeeze(-1),
            atol=1e-12,
        )
        # ...and column j+1 is the partner score for position j, symmetric in (i, j).
        assert logits.shape[-1] == logits.shape[-2] + 1
        assert torch.allclose(logits[0, 1, 1 + 3], logits[0, 3, 1 + 1], atol=1e-12)

    def test_the_unpaired_class_is_never_masked_so_no_row_is_fully_masked(self):
        hidden, mask = _hidden(length=2)
        logits = PairingHead(8, proj_dim=4).double()(hidden, mask)
        assert bool(torch.isfinite(logits[..., L.UNPAIRED_CLASS]).all())
        assert bool(torch.isfinite(torch.logsumexp(logits, dim=-1)).all())

    def test_it_reads_hidden_states_only_never_a_structure_string(self):
        """PRD §6: the head EMITS structure; nothing about it can consume one."""
        import inspect

        signature = inspect.signature(PairingHead.forward)
        assert list(signature.parameters) == ["self", "nucleotide_hidden", "nucleotide_mask"]
        assert PairingHead(8, proj_dim=4).left.in_features == 8

    def test_a_shape_mismatch_raises(self):
        hidden, mask = _hidden()
        head = PairingHead(8, proj_dim=4).double()
        with pytest.raises(ValueError, match="nucleotide_mask"):
            head(hidden, mask[:, :-1])
        with pytest.raises(ValueError, match=r"\(B, L, H\)"):
            head(hidden[0], mask[0])

    @pytest.mark.parametrize("bad", [0, -1])
    def test_degenerate_widths_are_refused(self, bad):
        with pytest.raises(ValueError):
            PairingHead(bad, proj_dim=4)
        with pytest.raises(ValueError):
            PairingHead(8, proj_dim=bad)


# ====================================================================================== #
# Torch tier — the term, graded against torch's own cross-entropy
# ====================================================================================== #
def _fixture_batch(seed=0):
    """Two rows: the hand-checked ``((..))`` and a ragged partly-unsupervised one."""
    torch.manual_seed(seed)
    length = 6
    mask = torch.ones(2, length, dtype=torch.bool)
    mask[1, -1:] = False
    logits = torch.randn(2, length, length + 1, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(
        [
            HAND_TARGET,
            # "(...)"+pad — 0 pairs 4, 1-3 unpaired, position 5 is padding
            [5, 0, 0, 0, 1, L.IGNORE_INDEX],
        ],
        dtype=torch.long,
    )
    return logits, target, mask


@requires_torch
class TestStructureTermReduction:
    def test_it_equals_torchs_own_masked_cross_entropy(self):
        logits, target, mask = _fixture_batch()
        value = L.structure_consistency_loss(logits, target, mask, gamma=0.0)
        reference = F.cross_entropy(
            logits.transpose(1, 2), target, ignore_index=L.IGNORE_INDEX, reduction="mean"
        )
        assert torch.allclose(value, reference, atol=1e-12)

    def test_an_unsupervised_position_does_not_enter_the_mean(self):
        logits, target, mask = _fixture_batch()
        value = L.structure_consistency_loss(logits, target, mask)
        per_position = F.cross_entropy(
            logits.transpose(1, 2), target, ignore_index=L.IGNORE_INDEX, reduction="none"
        )
        supervised = target != L.IGNORE_INDEX
        assert int(supervised.sum()) == 11
        assert torch.allclose(value, per_position[supervised].mean(), atol=1e-12)

    def test_an_all_ignored_batch_is_zero_with_a_count_of_zero_never_nan(self):
        """torch's own ``mean`` returns nan here — the witness that makes this non-vacuous."""
        logits, target, mask = _fixture_batch()
        blank = torch.full_like(target, L.IGNORE_INDEX)
        torch_value = F.cross_entropy(
            logits.transpose(1, 2), blank, ignore_index=L.IGNORE_INDEX, reduction="mean"
        )
        assert bool(torch.isnan(torch_value)), "the nan this term exists to avoid"
        value = L.structure_consistency_loss(logits, blank, mask)
        assert float(value) == 0.0
        assert bool(torch.isfinite(value))

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
    def test_the_gradient_is_finite_at_every_swept_gamma(self, gamma):
        logits, target, mask = _fixture_batch()
        value = L.structure_consistency_loss(logits, target, mask, gamma=gamma)
        value.backward()
        assert bool(torch.isfinite(logits.grad).all())
        assert float(logits.grad.abs().sum()) > 0.0

    def test_the_gradient_is_exactly_zero_at_an_unsupervised_position(self):
        logits, target, mask = _fixture_batch()
        L.structure_consistency_loss(logits, target, mask).backward()
        assert float(logits.grad[1, 5].abs().sum()) == 0.0
        assert float(logits.grad[0, 0].abs().sum()) > 0.0


# ====================================================================================== #
# Torch tier — imp.md gate 2: "loss decreases on the hand-checked fixture"
# ====================================================================================== #
@requires_torch
class TestItLearnsTheHandCheckedFixture:
    @staticmethod
    def _fit(steps=60, seed=0, lr=0.2):
        torch.manual_seed(seed)
        head = PairingHead(12, proj_dim=8).double()
        hidden = torch.randn(1, 6, 12, dtype=torch.float64)
        mask = torch.ones(1, 6, dtype=torch.bool)
        target = torch.tensor([HAND_TARGET], dtype=torch.long)
        optimiser = torch.optim.SGD(head.parameters(), lr=lr)
        history = []
        for _ in range(steps):
            optimiser.zero_grad()
            loss = L.structure_consistency_loss(head(hidden, mask), target, mask)
            loss.backward()
            optimiser.step()
            history.append(float(loss))
        return head, hidden, mask, target, history

    def test_the_loss_decreases(self):
        *_, history = self._fit()
        assert history[-1] < history[0], (history[0], history[-1])
        assert all(v > 0 for v in history)

    def test_it_recovers_the_hand_checked_pairing(self):
        """A falling loss on its own could be fitting anything; this names the answer."""
        head, hidden, mask, target, _ = self._fit(steps=400)
        predicted = head(hidden, mask).argmax(dim=-1)
        assert predicted.tolist() == [HAND_TARGET]
        assert target.tolist() == [HAND_TARGET]

    def test_an_untrained_head_does_not_already_predict_it(self):
        """The positive control for the control: the fixture is not trivially solved."""
        torch.manual_seed(0)
        head = PairingHead(12, proj_dim=8).double()
        hidden = torch.randn(1, 6, 12, dtype=torch.float64)
        mask = torch.ones(1, 6, dtype=torch.bool)
        assert head(hidden, mask).argmax(dim=-1).tolist() != [HAND_TARGET]


# ====================================================================================== #
# Torch tier — what the term refuses (each paired with the clean input succeeding)
# ====================================================================================== #
@requires_torch
class TestTheTermFailsClosed:
    def test_the_clean_fixture_succeeds(self):
        """The positive control every refusal below is measured against: a guard that
        raised on everything would satisfy ``pytest.raises`` too."""
        logits, target, mask = _fixture_batch()
        assert bool(torch.isfinite(L.structure_consistency_loss(logits, target, mask)))

    def test_a_labelled_padded_position_is_refused(self):
        logits, target, mask = _fixture_batch()
        corrupt = target.clone()
        corrupt[1, 5] = L.UNPAIRED_CLASS  # a class on a position the mask calls padding
        with pytest.raises(ValueError, match="carries a real pairing class"):
            L.structure_consistency_loss(logits, corrupt, mask)

    def test_a_real_position_may_carry_no_label(self):
        """The other direction is legitimate — 12,273 rows have no dot-bracket at all."""
        logits, target, mask = _fixture_batch()
        partial = target.clone()
        partial[0] = L.IGNORE_INDEX
        assert bool(torch.isfinite(L.structure_consistency_loss(logits, partial, mask)))

    def test_a_partner_that_is_padding_is_refused(self):
        logits, target, mask = _fixture_batch()
        corrupt = target.clone()
        corrupt[1, 0] = 6  # partner 5, which row 1's mask calls padding
        corrupt[1, 4] = L.UNPAIRED_CLASS
        with pytest.raises(ValueError, match="partner position"):
            L.structure_consistency_loss(logits, corrupt, mask)

    def test_an_asymmetric_pair_is_refused(self):
        """What a crop produces: one end of a pair kept, the other cut or repointed."""
        logits, target, mask = _fixture_batch()
        corrupt = target.clone()
        corrupt[0, 0] = 3  # 0 now claims 2, but 2 says unpaired
        with pytest.raises(ValueError, match="not symmetric"):
            L.structure_consistency_loss(logits, corrupt, mask)

    def test_a_self_pair_needs_its_own_refusal_because_it_is_trivially_symmetric(self):
        """``i → i`` satisfies "j claims i back" — the symmetry check alone cannot see it.

        Found by this test, not by review: before the dedicated guard, a self-pair reached
        ``cross_entropy`` against the masked diagonal and produced a huge but perfectly
        finite loss.
        """
        logits, target, mask = _fixture_batch()
        corrupt = target.clone()
        corrupt[0, 0] = 1  # 0 claims 0 — itself
        corrupt[0, 5] = L.UNPAIRED_CLASS
        assert corrupt[0, int(corrupt[0, 0]) - 1] == 1, "the self-pair IS symmetric"
        with pytest.raises(ValueError, match="pairs a position with itself"):
            L.structure_consistency_loss(logits, corrupt, mask)

    @pytest.mark.parametrize("bad", [7, 12, -3])
    def test_a_class_outside_the_axis_is_refused(self, bad):
        logits, target, mask = _fixture_batch()
        corrupt = target.clone()
        corrupt[0, 2] = bad
        with pytest.raises(ValueError, match=r"outside \[0, 6\]"):
            L.structure_consistency_loss(logits, corrupt, mask)

    def test_a_class_axis_that_is_not_one_plus_l_is_refused(self):
        """The off-by-one that would mislabel every base: an L-wide or L+2-wide axis."""
        logits, target, mask = _fixture_batch()
        with pytest.raises(ValueError, match="class axis"):
            L.structure_consistency_loss(logits[..., :-1], target, mask)
        with pytest.raises(ValueError, match="class axis"):
            L.structure_consistency_loss(torch.cat([logits, logits[..., :1]], dim=-1), target, mask)

    def test_shape_mismatches_are_refused(self):
        logits, target, mask = _fixture_batch()
        with pytest.raises(ValueError, match="structure target"):
            L.structure_consistency_loss(logits, target[:1], mask)
        with pytest.raises(ValueError, match="nucleotide_mask"):
            L.structure_consistency_loss(logits, target, mask[:1])
        with pytest.raises(ValueError, match=r"\(B, L, 1 \+ L\)"):
            L.structure_consistency_loss(logits[0], target[0], mask[0])

    @pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32"])
    def test_a_narrow_integer_target_is_promoted_before_the_partner_arithmetic(self, dtype_name):
        """``_check_supervised_targets`` admits int8/16/32, but ``j + 1`` and the gathers
        must not run in them: this term promotes to int64 first.

        The witness that the promotion is doing work rather than decorating a no-op is
        torch's own wrapping arithmetic in the narrow dtype, asserted below — ``-128 - 1``
        is ``127`` in int8, so a partner index computed in that dtype is not merely
        imprecise, it is a *different, valid-looking* position.
        """
        wrapped = torch.tensor([-128], dtype=torch.int8) - 1
        assert int(wrapped) == 127, "int8 wraps — that is what makes the promotion load-bearing"

        logits, target, mask = _fixture_batch()
        narrow = target.to(getattr(torch, dtype_name))
        value = L.structure_consistency_loss(logits, narrow, mask)
        assert torch.allclose(value, L.structure_consistency_loss(logits, target, mask))

    def test_promotion_reaches_a_partner_index_no_int8_could_hold(self):
        """L = 200, so the largest class (200) is outside int8 entirely.

        int16 can hold it; the arithmetic must be done after the promotion, or ``gather``
        would be handed a non-int64 index and the ``- 1`` a narrow accumulator.
        """
        torch.manual_seed(0)
        length = 200
        logits = torch.randn(1, length, length + 1, dtype=torch.float64)
        mask = torch.ones(1, length, dtype=torch.bool)
        text = "(" * 60 + "." * 80 + ")" * 60
        encoded = L.encode_pairing_target(text, length=length)
        assert max(encoded) == length, "the fixture must exercise the top of the class axis"
        assert max(encoded) > 127, "…and a value int8 cannot represent"
        target = torch.tensor([encoded], dtype=torch.long)
        value = L.structure_consistency_loss(logits, target.to(torch.int16), mask)
        assert torch.allclose(value, L.structure_consistency_loss(logits, target, mask))
        assert bool(torch.isfinite(value))

    def test_a_target_that_wrapped_upstream_is_caught_not_absorbed(self):
        """Promotion cannot undo a wrap that already happened in the collator — but the
        range and symmetry checks run *after* it, so the corruption still fails closed."""
        logits, target, mask = _fixture_batch()
        corrupt = target.clone()
        corrupt[0, 0] = -56  # what a wrapped narrow index looks like
        with pytest.raises(ValueError, match=r"outside \[0, 6\]"):
            L.structure_consistency_loss(logits, corrupt, mask)


# ====================================================================================== #
# Torch tier — the term inside the multi-task objective, and the model that feeds it
# ====================================================================================== #
def _model_batch(structure_head=True, batch=3, tokens=11, width=16, seed=0):
    spec = H.load_head_spec()
    torch.manual_seed(seed)
    model = Stage2Model(spec, d_model=width, structure_head=structure_head).double()
    hidden = torch.randn(batch, tokens, width, dtype=torch.float64)
    attention = torch.ones(batch, tokens, dtype=torch.long)
    attention[-1, -3:] = 0
    outputs = model.heads_from_hidden(hidden, attention)
    lengths = [int(n) for n in outputs["nucleotide_mask"].sum(dim=1)]
    padded = outputs["boundary_logits"].shape[1]

    boundary = torch.full((batch, padded), L.IGNORE_INDEX, dtype=torch.long)
    for row, n in enumerate(lengths):
        boundary[row, :n] = torch.randint(0, spec.size(H.BOUNDARY_FIELD), (n,))
    targets = {"binary": torch.randint(0, 2, (batch,)), "boundary": boundary}
    for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family"):
        targets[term] = torch.randint(0, spec.size(L.TERM_TO_FIELD[term]), (batch,))

    # Real dot-brackets of each row's own length, padded by the "collator" with IGNORE.
    structure = torch.full((batch, padded), L.IGNORE_INDEX, dtype=torch.long)
    for row, n in enumerate(lengths):
        text = "(" * (n // 3) + "." * (n - 2 * (n // 3)) + ")" * (n // 3)
        structure[row, :n] = torch.tensor(L.encode_pairing_target(text, length=n))
    targets[L.STRUCTURE_TERM] = structure
    return model, outputs, targets, lengths


@requires_torch
class TestInsideTheMultitaskObjective:
    def test_the_model_grows_exactly_one_key_and_the_loss_consumes_it(self):
        model, outputs, targets, lengths = _model_batch()
        assert model.has_structure_head is True
        assert model.output_keys[-2:] == (STRUCTURE_OUTPUT_KEY, "nucleotide_mask")
        assert STRUCTURE_OUTPUT_KEY in outputs
        padded = outputs["boundary_logits"].shape[1]
        assert tuple(outputs[STRUCTURE_OUTPUT_KEY].shape) == (3, padded, padded + 1)
        total, parts = L.multitask_loss(
            outputs, targets, config=L.Stage2LossConfig(structure_enabled=True)
        )
        assert bool(torch.isfinite(total))
        assert parts["n_supervised"][L.STRUCTURE_TERM] == sum(lengths)
        assert L.STRUCTURE_TERM in parts["included"]

    def test_the_default_model_carries_no_structure_head_at_all(self):
        model, outputs, _, _ = _model_batch(structure_head=False)
        assert model.has_structure_head is False
        assert model.structure_head is None
        assert STRUCTURE_OUTPUT_KEY not in outputs
        assert "structure_head" not in model.head_modules

    def test_the_gradient_reaches_head_e_and_the_backbone_free_path(self):
        model, outputs, targets, _ = _model_batch()
        total, _ = L.multitask_loss(
            outputs, targets, config=L.Stage2LossConfig(structure_enabled=True)
        )
        total.backward()
        for name, param in model.structure_head.named_parameters():
            assert param.grad is not None, name
            assert bool(torch.isfinite(param.grad).all()), name
        assert float(model.structure_head.left.weight.grad.abs().sum()) > 0.0

    def test_the_term_is_weighted_like_every_other_aux_term(self):
        model, outputs, targets, _ = _model_batch()
        cfg = L.Stage2LossConfig(structure_enabled=True, structure_weight=0.2, aux_weight=0.5)
        total, parts = L.multitask_loss(outputs, targets, config=cfg)
        rebuilt = sum(parts["weights"][term] * parts["terms"][term] for term in parts["included"])
        assert torch.allclose(total, rebuilt, atol=1e-12)
        assert parts["weights"][L.STRUCTURE_TERM] == pytest.approx(0.1)

    def test_the_no_aux_arm_removes_it_exactly(self):
        model, outputs, targets, _ = _model_batch()
        on = L.Stage2LossConfig(structure_enabled=True)
        off = L.Stage2LossConfig(structure_enabled=True, aux_weight=0.0)
        total_off, parts_off = L.multitask_loss(outputs, targets, config=off)
        binary_only, _ = L.multitask_loss(outputs, targets, config=on)
        assert L.STRUCTURE_TERM not in parts_off["included"]
        assert torch.allclose(total_off, parts_off["terms"][L.BINARY_TERM], atol=1e-14)
        assert not torch.allclose(total_off, binary_only)

    def test_an_all_decoy_batch_reports_the_emptiness_rather_than_hiding_it(self):
        """All 7,007 decoys carry no dot-bracket; a vacuous term must be visible."""
        model, outputs, targets, _ = _model_batch()
        targets = dict(targets)
        targets[L.STRUCTURE_TERM] = torch.full_like(targets[L.STRUCTURE_TERM], L.IGNORE_INDEX)
        total, parts = L.multitask_loss(
            outputs, targets, config=L.Stage2LossConfig(structure_enabled=True)
        )
        assert parts["n_supervised"][L.STRUCTURE_TERM] == 0
        assert L.STRUCTURE_TERM in parts["skipped_unsupervised"]
        assert L.STRUCTURE_TERM not in parts["included"]
        assert float(parts["terms"][L.STRUCTURE_TERM]) == 0.0
        assert bool(torch.isfinite(total))

    def test_a_head_without_its_term_is_refused_rather_than_left_gradient_free(self):
        _, outputs, targets, _ = _model_batch()
        with pytest.raises(ValueError, match="no gradient at all"):
            L.multitask_loss(outputs, targets, config=L.Stage2LossConfig())

    def test_a_term_without_its_head_is_refused(self):
        _, outputs, targets, _ = _model_batch(structure_head=False)
        with pytest.raises(ValueError, match="outputs is missing"):
            L.multitask_loss(outputs, targets, config=L.Stage2LossConfig(structure_enabled=True))

    def test_a_structure_target_for_a_disabled_term_is_reported_as_unused(self):
        """Unused *data* is reported; a dead *parameter* is refused. Both stay visible."""
        _, outputs, targets, _ = _model_batch(structure_head=False)
        total, parts = L.multitask_loss(outputs, targets, config=L.Stage2LossConfig())
        assert parts["unused_targets"] == [L.STRUCTURE_TERM]
        assert parts["disabled"] == [L.STRUCTURE_TERM]
        assert L.STRUCTURE_TERM not in parts["n_supervised"]
        assert bool(torch.isfinite(total))

    def test_class_weights_and_counts_are_refused_for_a_positional_axis(self):
        _, outputs, targets, _ = _model_batch()
        cfg = L.Stage2LossConfig(structure_enabled=True)
        with pytest.raises(ValueError, match="not meaningful"):
            L.multitask_loss(outputs, targets, config=cfg, class_weights={L.STRUCTURE_TERM: [1.0]})
        with pytest.raises(ValueError, match="cannot be used"):
            L.MultitaskLoss(cfg, class_counts={L.STRUCTURE_TERM: [1, 2]})

    def test_the_uncertainty_alternative_covers_it_too(self):
        _, outputs, targets, _ = _model_batch()
        cfg = L.Stage2LossConfig(weighting="uncertainty", structure_enabled=True)
        log_variances = L.uncertainty_log_variances()
        assert tuple(log_variances.shape) == (len(L.TERMS),) == (7,)
        total, parts = L.multitask_loss(outputs, targets, config=cfg, log_variances=log_variances)
        total.backward()
        assert L.STRUCTURE_TERM in parts["weights"]
        assert bool(torch.isfinite(log_variances.grad).all())
        assert float(log_variances.grad[L.TERMS.index(L.STRUCTURE_TERM)].abs()) > 0.0


# ====================================================================================== #
# Torch tier — drift guard against the model's own key inventory
# ====================================================================================== #
@requires_torch
class TestModelContractDrift:
    def test_the_optional_key_is_the_models_own(self):
        assert OPTIONAL_HEAD_OUTPUT_KEYS == (STRUCTURE_OUTPUT_KEY,)
        assert L.TERM_TO_LOGITS[L.STRUCTURE_TERM] == STRUCTURE_OUTPUT_KEY
        assert tuple(L.TERM_TO_LOGITS[term] for term in L.TERMS) == ALL_HEAD_OUTPUT_KEYS

    def test_head_e_is_a_head_and_not_part_of_the_backbone(self):
        """It must be in the LoRA-exclusion set, or ``all-linear`` would adapt it."""
        model, *_ = _model_batch()
        assert model.head_modules["structure_head"] is model.structure_head
        assert all(param.requires_grad for param in model.structure_head.parameters())
