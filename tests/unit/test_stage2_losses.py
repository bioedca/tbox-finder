"""Unit gate for the P3-04 Stage-2 multi-task objective (:mod:`tbox_finder.stage2.losses`).

Four tiers, gated on what the running env has (there is no conftest and no registered
marker in this repo — the skip idiom is the try/except+``skipif`` one from
``tests/unit/test_objective.py``, which also distinguishes ``exc.name != "torch"`` so a
broken sibling module cannot self-skip the whole tensor tier green):

* **bare** — the weighting scheme, the ADR-0005 D16 dominance rule, the closed method
  allow-list and the config validation. All pure, so all of it runs in CI.
* **bare + hydra** — the PRD §11 sweep axis, composed for real. CI installs hydra-core, so
  the job-669 struct-mode trap (a dataclass field with no YAML key) is caught in CI rather
  than by a GPU job dying in its first seconds.
* **torch** — per-term reduction, masking, weighting math and gradient flow on synthetic
  logits. No backbone, no download.
* **torch + drift** — that the term→logit-key map still matches ``Stage2Model``'s own
  output keys, which live in a torch-importing module.

Two conventions carried from ``test_objective.py``: fp64 fixtures with ``atol=1e-12`` for
anything asserting exact arithmetic, and grading against an **external** reference
(``F.binary_cross_entropy_with_logits``, ``F.cross_entropy``) rather than against the
module's own path — a test that recomputes the implementation is a tautology, and the
binary term's two-logit restatement is exactly the kind of clever step that needs an
outside witness.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from tbox_finder.stage2 import heads as H
from tbox_finder.stage2 import losses as L

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in the bare CI env
    if exc.name != "torch":
        raise
    torch = None
    F = None
    _HAS_TORCH = False
else:
    # Deliberately OUTSIDE the guard: a broken tbox_finder.train.objective must raise here,
    # not be swallowed into `_HAS_TORCH = False` and silently skip the tensor tier green.
    import torch.nn.functional as F

    _HAS_TORCH = True

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not installed (bare CI) — Stage-2 loss tensor tier runs under tbox-ml-rna",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"
LOSS_CONF = CONF_DIR / "loss" / "stage2.yaml"

#: Head widths, read off the committed vocabulary rather than retyped.
SPEC = H.load_head_spec()
WIDTHS = {
    "binary": 2,  # the two-logit restatement of head (a)'s single logit
    "boundary": SPEC.size(H.BOUNDARY_FIELD),
    "regulatory_mode": SPEC.size(H.REGULATORY_MODE_FIELD),
    "specifier_codon": SPEC.size(H.CODON_FIELD),
    "cognate_aa": SPEC.size(H.AMINO_ACID_FIELD),
    "trna_family": SPEC.size(H.TRNA_FAMILY_FIELD),
}


# ====================================================================================== #
# Bare tier — the term inventory
# ====================================================================================== #
class TestTermInventory:
    def test_terms_are_the_six_prd_heads_primary_first(self):
        assert L.BINARY_TERM == "binary"
        assert (L.BINARY_TERM, *L.AUX_TERMS) == L.TERMS
        assert len(L.TERMS) == 6
        assert set(L.TERM_TO_LOGITS) == set(L.TERMS)

    def test_aux_terms_pair_one_to_one_with_the_head_vocabularies(self):
        """The pairing is what makes a target the *right* target for a head.

        Asserted as an identity against ``heads.VOCAB_FIELDS``, not as a count: five terms
        and five fields would still line up if two of them were swapped, and two heads of
        the same width (there are none today, but `regulatory_mode` is 2 and the binary
        restatement is 2) would then train on each other's labels with every loss finite.
        """
        assert tuple(L.TERM_TO_FIELD[term] for term in L.AUX_TERMS) == H.VOCAB_FIELDS
        assert L.BINARY_TERM not in L.TERM_TO_FIELD

    def test_ignore_index_is_the_heads_sentinel(self):
        assert L.IGNORE_INDEX == H.IGNORE_INDEX == -100
        assert L.Stage2LossConfig().ignore_index == H.IGNORE_INDEX

    def test_only_the_boundary_term_is_per_nucleotide(self):
        assert L.PER_NUCLEOTIDE_TERMS == ("boundary",)
        assert set(L.PER_NUCLEOTIDE_TERMS) <= set(L.AUX_TERMS)

    def test_the_hydra_group_file_is_where_the_sweep_axis_lives(self):
        """Breaks loudly on a rename even in an env without hydra installed."""
        assert LOSS_CONF.is_file()
        assert "aux_weight:" in LOSS_CONF.read_text()


# ====================================================================================== #
# Bare tier — ADR-0005 D16: the weighting method and the dominance rule
# ====================================================================================== #
class TestWeightingMethod:
    def test_the_allow_list_is_exactly_d16s_two_methods(self):
        assert L.WEIGHTING_METHODS == ("fixed", "uncertainty")
        assert L.Stage2LossConfig().weighting == "fixed"

    @pytest.mark.parametrize("bad", ["gradnorm", "Fixed", "", "uniform", "none"])
    def test_an_unlisted_method_is_refused(self, bad):
        with pytest.raises(ValueError, match="ADR-0005 D16"):
            L.Stage2LossConfig(weighting=bad)

    def test_uncertainty_refuses_a_partial_aux_weight(self):
        """Scaling a learned weight by a manual one is neither of D16's two methods."""
        L.Stage2LossConfig(weighting="uncertainty", aux_weight=0.0)
        L.Stage2LossConfig(weighting="uncertainty", aux_weight=1.0)
        with pytest.raises(ValueError, match="hybrid"):
            L.Stage2LossConfig(weighting="uncertainty", aux_weight=0.5)


class TestDominance:
    def test_the_shipped_defaults_dominate_under_both_readings(self):
        margins = L.Stage2LossConfig().dominance()
        assert margins["holds"]
        assert margins["sum_margin"] >= 0.0
        assert margins["max_margin"] > 0.0

    def test_the_two_readings_are_reported_and_can_disagree(self):
        """The enforced rule is the sum one; the weaker per-term one is only reported.

        This is the measured divergence between the two readings of D16's qualitative
        "binary head weighted to dominate": at aux_weight=2 the shipped base weights put
        every individual aux term below the binary one (max_margin > 0) while their total is
        1.4x it (sum_margin < 0). The step encodes the sum reading; this test pins that the
        two are not the same rule, so the choice stays visible instead of being absorbed.
        """
        cfg = L.Stage2LossConfig()  # a valid config to read base weights off
        base_aux = sum(cfg.base_weight(t) for t in L.AUX_TERMS)
        assert cfg.max_aux_weight == pytest.approx(cfg.binary_weight / base_aux)
        with pytest.raises(ValueError, match="dominate"):
            L.Stage2LossConfig(aux_weight=2.0)
        # ... and the weaker reading would have admitted it:
        effective_aux = [2.0 * cfg.base_weight(t) for t in L.AUX_TERMS]
        assert max(effective_aux) < cfg.binary_weight
        assert sum(effective_aux) > cfg.binary_weight

    def test_the_no_aux_arm_is_admitted(self):
        cfg = L.Stage2LossConfig(aux_weight=0.0)
        assert cfg.dominance()["holds"]
        assert cfg.effective_weights() == {
            **{t: 0.0 for t in L.AUX_TERMS},
            L.BINARY_TERM: 1.0,
        }

    def test_a_zero_binary_weight_is_refused(self):
        """Fail-closed: zeroing the primary makes every reading of "dominate" false."""
        with pytest.raises(ValueError, match="dominate"):
            L.Stage2LossConfig(binary_weight=0.0)

    def test_the_cap_is_exactly_the_admissible_boundary(self):
        cfg = L.Stage2LossConfig()
        cap = cfg.max_aux_weight
        L.Stage2LossConfig(aux_weight=cap)  # the boundary itself is admitted (>=, not >)
        with pytest.raises(ValueError, match="dominate"):
            L.Stage2LossConfig(aux_weight=math.nextafter(cap, math.inf))

    def test_dominance_is_not_enforced_under_uncertainty(self):
        """The weights are learned there, so a static rule has nothing to check."""
        cfg = L.Stage2LossConfig(weighting="uncertainty", binary_weight=0.01)
        assert not cfg.dominance()["holds"]  # reported, and construction still succeeded


class TestConfigValidation:
    @pytest.mark.parametrize(
        "field",
        [
            "aux_weight",
            "binary_weight",
            "boundary_weight",
            "binary_gamma",
            "boundary_gamma",
            "aux_gamma",
            "aux_class_weight_alpha",
        ],
    )
    def test_nan_is_refused_on_every_scalar(self, field):
        """`x < 0` is False for NaN, so a NaN sweep value would sail through a bare check."""
        with pytest.raises(ValueError, match="finite"):
            L.Stage2LossConfig(**{field: float("nan")})

    @pytest.mark.parametrize("field", ["aux_weight", "binary_weight", "boundary_gamma"])
    def test_negative_is_refused(self, field):
        with pytest.raises(ValueError):
            L.Stage2LossConfig(**{field: -1.0})

    def test_gamma_and_alpha_route_per_term(self):
        cfg = L.Stage2LossConfig(binary_gamma=1.0, boundary_gamma=2.5, aux_gamma=0.5)
        assert cfg.gamma("binary") == 1.0
        assert cfg.gamma("boundary") == 2.5
        assert [cfg.gamma(t) for t in ("regulatory_mode", "cognate_aa")] == [0.5, 0.5]
        with pytest.raises(ValueError, match="unknown term"):
            cfg.gamma("nope")

    def test_diagnostics_separate_the_pinned_method_from_the_swept_numbers(self):
        diag = L.Stage2LossConfig().diagnostics()
        assert diag["pinned"]["weighting_method"] == "ADR-0005 D16"
        assert diag["pinned"]["dominance_rule"] == "ADR-0005 D16"
        assert diag["pinned"]["weight_values"] is False
        assert set(diag["effective_weights"]) == set(L.TERMS)


# ====================================================================================== #
# Bare tier — class weights are derived from measured counts, never assumed
# ====================================================================================== #
class TestClassWeightDerivation:
    def test_alpha_without_counts_raises(self):
        with pytest.raises(ValueError, match="never assumed"):
            L.MultitaskLoss(L.Stage2LossConfig(aux_class_weight_alpha=1.0))

    def test_alpha_zero_needs_nothing(self):
        loss_fn = L.MultitaskLoss()
        assert loss_fn.weights is None
        assert loss_fn.diagnostics()["class_weights"] is None

    def test_derived_weights_are_inverse_frequency_and_mean_one(self):
        counts = [90, 10]
        loss_fn = L.MultitaskLoss(
            L.Stage2LossConfig(binary_class_weight_alpha=1.0),
            class_counts={"binary": counts},
        )
        w = loss_fn.weights["binary"]
        assert sum(w) / len(w) == pytest.approx(1.0)
        # w ∝ 1/n, so the 9x rarer class takes 9x the weight — graded against the ratio, not
        # against a recomputation of the implementation.
        assert w[1] / w[0] == pytest.approx(9.0)

    def test_a_zero_count_class_is_refused_not_clamped(self):
        """A 64-wide codon axis the training fold does not fully realise cannot be weighted."""
        counts = {
            term: [5] * WIDTHS[term]
            for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family")
        }
        counts["specifier_codon"][-1] = 0
        with pytest.raises(ValueError, match="undefined"):
            L.MultitaskLoss(L.Stage2LossConfig(aux_class_weight_alpha=1.0), class_counts=counts)

    def test_unknown_term_in_counts_raises(self):
        with pytest.raises(ValueError, match="unknown terms"):
            L.MultitaskLoss(class_counts={"specifier": [1, 2]})


# ====================================================================================== #
# Bare + hydra tier — the PRD §11 sweep axis, composed for real
# ====================================================================================== #
class TestHydraSweepAxis:
    """The job-669 gate: a dataclass field with no YAML key is a hard compose failure.

    These compose the shipped ``conf/`` tree exactly as a trainer would, so a missing key is
    caught in CI (which installs hydra-core) rather than by a GPU job dying at second one.
    """

    @staticmethod
    def _compose(*overrides):
        pytest.importorskip("hydra")
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
            return compose(config_name="train/lora_stage2", overrides=list(overrides))

    def test_the_loss_group_composes_under_the_stage2_primary(self):
        cfg = self._compose()
        assert cfg.loss.weighting == "fixed"
        assert cfg.loss.aux_weight == 1.0

    def test_aux_weight_is_overridable_the_sweep_axis_p3_08_needs(self):
        cfg = self._compose("loss.aux_weight=0.0")
        assert cfg.loss.aux_weight == 0.0

    @pytest.mark.parametrize(
        "field",
        [f.name for f in dataclasses.fields(L.Stage2LossConfig) if f.name != "ignore_index"],
    )
    def test_every_config_field_has_a_literal_yaml_key(self, field):
        """Struct mode rejects an override for a key the YAML lacks — job 669's exact death.

        ``ignore_index`` is exercised separately: it must exist as a key too, but overriding
        it to a *different* value is not a supported operation.
        """
        value = "fixed" if field == "weighting" else "0.0"
        cfg = self._compose(f"loss.{field}={value}")
        assert field in cfg.loss

    def test_ignore_index_is_present_and_matches_the_code(self):
        cfg = self._compose()
        assert cfg.loss.ignore_index == H.IGNORE_INDEX

    def test_an_unknown_loss_key_is_refused(self):
        """Struct mode is what makes the presence test above meaningful."""
        with pytest.raises(Exception, match="not in struct|Could not override"):
            self._compose("loss.aux_wieght=0.5")

    def test_the_yaml_builds_a_valid_config_object(self):
        cfg = self._compose()
        from omegaconf import OmegaConf

        built = L.Stage2LossConfig(**OmegaConf.to_container(cfg.loss, resolve=True))
        assert built == L.Stage2LossConfig()

    def test_the_yaml_defaults_are_the_dataclass_defaults(self):
        """A config that silently disagreed with the code would misdescribe every run."""
        cfg = self._compose()
        from omegaconf import OmegaConf

        yaml_values = OmegaConf.to_container(cfg.loss, resolve=True)
        for f in dataclasses.fields(L.Stage2LossConfig):
            assert f.name in yaml_values, f.name
            assert yaml_values[f.name] == f.default, f.name


# ====================================================================================== #
# Torch tier — fixtures
# ====================================================================================== #
def _outputs(batch=4, length=7, *, seed=0, dtype=None):
    """Synthetic head logits of the real widths, plus a left-aligned nucleotide mask."""
    torch.manual_seed(seed)
    dtype = dtype or torch.float64
    lengths = [length - i % 2 for i in range(batch)]  # ragged, so padding is real
    mask = torch.zeros(batch, length, dtype=torch.bool)
    for row, n in enumerate(lengths):
        mask[row, :n] = True
    out = {
        "tbox_logit": torch.randn(batch, dtype=dtype, requires_grad=True),
        "boundary_logits": torch.randn(
            batch, length, WIDTHS["boundary"], dtype=dtype, requires_grad=True
        ),
        "nucleotide_mask": mask,
    }
    for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family"):
        out[L.TERM_TO_LOGITS[term]] = torch.randn(
            batch, WIDTHS[term], dtype=dtype, requires_grad=True
        )
    return out, lengths


def _targets(outputs, lengths, *, seed=0, supervise_aux=True):
    """Targets matching ``_outputs``; ``supervise_aux=False`` is the all-decoy batch."""
    torch.manual_seed(seed + 1)
    batch, length = outputs["nucleotide_mask"].shape
    boundary = torch.full((batch, length), L.IGNORE_INDEX, dtype=torch.long)
    for row, n in enumerate(lengths):
        boundary[row, :n] = torch.randint(0, WIDTHS["boundary"], (n,))
    targets = {
        "binary": torch.randint(0, 2, (batch,)),
        "boundary": boundary,
    }
    for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family"):
        if supervise_aux:
            targets[term] = torch.randint(0, WIDTHS[term], (batch,))
        else:
            targets[term] = torch.full((batch,), L.IGNORE_INDEX, dtype=torch.long)
    if not supervise_aux:
        targets["boundary"] = torch.full((batch, length), L.IGNORE_INDEX, dtype=torch.long)
    return targets


# ====================================================================================== #
# Torch tier — per-term reduction, graded against torch itself
# ====================================================================================== #
@requires_torch
class TestPerTermReduction:
    def test_binary_term_equals_torchs_own_bce_with_logits(self):
        """The two-logit restatement is exact, not approximate — the outside witness."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        _, parts = L.multitask_loss(out, tgt)
        reference = F.binary_cross_entropy_with_logits(
            out["tbox_logit"], tgt["binary"].to(out["tbox_logit"].dtype), reduction="mean"
        )
        assert torch.allclose(parts["terms"]["binary"], reference, atol=1e-12)

    def test_binary_gradient_equals_sigmoid_minus_target(self):
        """d/dz of BCE-with-logits. The fabricated zero channel must carry no gradient."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        cfg = L.Stage2LossConfig(aux_weight=0.0)
        total, _ = L.multitask_loss(out, tgt, config=cfg)
        total.backward()
        z = out["tbox_logit"]
        expected = (torch.sigmoid(z.detach()) - tgt["binary"].to(z.dtype)) / z.numel()
        assert torch.allclose(z.grad, expected, atol=1e-12)

    def test_boundary_term_equals_torchs_masked_cross_entropy(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        _, parts = L.multitask_loss(out, tgt, config=L.Stage2LossConfig(boundary_gamma=0.0))
        reference = F.cross_entropy(
            out["boundary_logits"].transpose(1, 2),
            tgt["boundary"],
            ignore_index=L.IGNORE_INDEX,
            reduction="mean",
        )
        assert torch.allclose(parts["terms"]["boundary"], reference, atol=1e-12)

    def test_record_terms_equal_torchs_cross_entropy(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        _, parts = L.multitask_loss(out, tgt)
        for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family"):
            reference = F.cross_entropy(
                out[L.TERM_TO_LOGITS[term]],
                tgt[term],
                ignore_index=L.IGNORE_INDEX,
                reduction="mean",
            )
            assert torch.allclose(parts["terms"][term], reference, atol=1e-12), term

    def test_padded_positions_do_not_enter_the_boundary_mean(self):
        """The reduction is over supervised positions, not over the padded rectangle."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        _, parts = L.multitask_loss(out, tgt, config=L.Stage2LossConfig(boundary_gamma=0.0))
        per_position = F.cross_entropy(
            out["boundary_logits"].transpose(1, 2),
            tgt["boundary"],
            ignore_index=L.IGNORE_INDEX,
            reduction="sum",
        )
        assert parts["n_supervised"]["boundary"] == sum(lengths)
        assert torch.allclose(parts["terms"]["boundary"], per_position / sum(lengths), atol=1e-12)
        assert sum(lengths) < out["boundary_logits"].shape[0] * out["boundary_logits"].shape[1]


# ====================================================================================== #
# Torch tier — masked, not dense
# ====================================================================================== #
@requires_torch
class TestMaskedNotDense:
    def test_an_all_decoy_batch_is_zero_with_a_zero_count_never_nan(self):
        """torch's own mean reduction returns nan here; the reported count says why it is 0."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths, supervise_aux=False)
        total, parts = L.multitask_loss(out, tgt)
        assert torch.isfinite(total)
        for term in L.AUX_TERMS:
            assert parts["n_supervised"][term] == 0
            assert float(parts["terms"][term]) == 0.0
            assert term in parts["skipped_unsupervised"]
            assert term not in parts["included"]
        # The witness that this is not vacuous: torch's own path DOES produce nan.
        naive = F.cross_entropy(
            out["cognate_aa_logits"], tgt["cognate_aa"], ignore_index=L.IGNORE_INDEX
        )
        assert torch.isnan(naive)

    def test_a_partially_supervised_term_reduces_over_the_supervised_rows_only(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        tgt["cognate_aa"][1] = L.IGNORE_INDEX
        tgt["cognate_aa"][3] = L.IGNORE_INDEX
        _, parts = L.multitask_loss(out, tgt)
        keep = [0, 2]
        reference = F.cross_entropy(
            out["cognate_aa_logits"][keep], tgt["cognate_aa"][keep], reduction="mean"
        )
        assert parts["n_supervised"]["cognate_aa"] == 2
        assert torch.allclose(parts["terms"]["cognate_aa"], reference, atol=1e-12)

    def test_a_fully_unsupervised_batch_still_backpropagates(self):
        """Degenerate batches must not blow up the caller's backward()."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths, supervise_aux=False)
        tgt["binary"] = torch.full((out["tbox_logit"].shape[0],), L.IGNORE_INDEX)
        total, parts = L.multitask_loss(out, tgt)
        assert parts["included"] == []
        assert float(total) == 0.0
        total.backward()
        assert torch.allclose(out["tbox_logit"].grad, torch.zeros_like(out["tbox_logit"]))

    def test_gradient_is_exactly_zero_at_an_ignored_record(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        tgt["trna_family"][2] = L.IGNORE_INDEX
        total, _ = L.multitask_loss(out, tgt)
        total.backward()
        grad = out["trna_family_logits"].grad
        assert torch.allclose(grad[2], torch.zeros(WIDTHS["trna_family"], dtype=grad.dtype))
        assert float(grad[0].abs().sum()) > 0.0

    def test_a_labelled_padded_position_is_refused(self):
        """One-directional by design: padding may not carry a class.

        Paired with its own positive control. Without the clean-batch leg this test passes
        under an *inverted* mask test too — a check that raises on everything raises here as
        well, and "it raised" would read as "it caught the corruption". Sabotage said so.
        """
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        total, _ = L.multitask_loss(out, tgt)  # the identical batch, uncorrupted
        assert torch.isfinite(total)
        row = lengths.index(min(lengths))
        tgt["boundary"][row, lengths[row]] = 0  # a 'background' label on a non-existent base
        with pytest.raises(ValueError, match="masked out by nucleotide_mask"):
            L.multitask_loss(out, tgt)

    def test_a_real_position_may_carry_no_label(self):
        """The other direction is the ordinary decoy case and must NOT raise."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        tgt["boundary"][0, :] = L.IGNORE_INDEX
        total, parts = L.multitask_loss(out, tgt)
        assert torch.isfinite(total)
        assert parts["n_supervised"]["boundary"] == sum(lengths) - lengths[0]


# ====================================================================================== #
# Torch tier — the weighting math
# ====================================================================================== #
@requires_torch
class TestFixedWeighting:
    def test_total_is_the_weighted_sum_of_the_reported_terms(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        cfg = L.Stage2LossConfig(aux_weight=0.5)
        total, parts = L.multitask_loss(out, tgt, config=cfg)
        expected = sum(
            (cfg.effective_weights()[term] * parts["terms"][term] for term in L.TERMS),
            start=torch.zeros((), dtype=total.dtype),
        )
        assert torch.allclose(total, expected, atol=1e-12)

    def test_aux_weight_scales_only_the_aux_terms(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        a, parts_a = L.multitask_loss(out, tgt, config=L.Stage2LossConfig(aux_weight=1.0))
        b, parts_b = L.multitask_loss(out, tgt, config=L.Stage2LossConfig(aux_weight=0.5))
        # The unweighted terms are identical; only the combination moved.
        for term in L.TERMS:
            assert torch.allclose(parts_a["terms"][term], parts_b["terms"][term], atol=1e-14)
        aux_a = a - parts_a["terms"]["binary"]
        aux_b = b - parts_b["terms"]["binary"]
        assert torch.allclose(aux_b, 0.5 * aux_a, atol=1e-12)

    def test_aux_weight_zero_is_exactly_the_binary_only_objective(self):
        """The P3-08 no-aux arm: the total must be the binary term, to the last bit."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        total, parts = L.multitask_loss(out, tgt, config=L.Stage2LossConfig(aux_weight=0.0))
        assert torch.allclose(total, parts["terms"]["binary"], atol=1e-14)
        assert parts["included"] == [L.BINARY_TERM]
        # ...and the aux terms are still *reported*, so "we turned them off" stays auditable.
        for term in L.AUX_TERMS:
            assert float(parts["terms"][term]) > 0.0
            assert parts["n_supervised"][term] > 0

    def test_no_aux_arm_leaves_the_aux_heads_without_gradient(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        total, _ = L.multitask_loss(out, tgt, config=L.Stage2LossConfig(aux_weight=0.0))
        total.backward()
        assert out["cognate_aa_logits"].grad is None
        assert float(out["tbox_logit"].grad.abs().sum()) > 0.0

    def test_class_weights_reach_the_term(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        # The realised targets must straddle DIFFERENT weight tiers, or weighted and
        # unweighted agree exactly and the "it changed the number" witness below is vacuous:
        # a weighted mean whose gathered weights are all equal IS the unweighted mean.
        tgt["cognate_aa"] = torch.arange(out["cognate_aa_logits"].shape[0])
        counts = {
            term: [n + 1 for n in range(WIDTHS[term])]
            for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family")
        }
        loss_fn = L.MultitaskLoss(
            L.Stage2LossConfig(aux_class_weight_alpha=1.0), class_counts=counts
        )
        _, parts = loss_fn(out, tgt)
        w = torch.tensor(loss_fn.weights["cognate_aa"], dtype=out["cognate_aa_logits"].dtype)
        reference = F.cross_entropy(
            out["cognate_aa_logits"],
            tgt["cognate_aa"],
            weight=w,
            ignore_index=L.IGNORE_INDEX,
            reduction="mean",
        )
        assert torch.allclose(parts["terms"]["cognate_aa"], reference, atol=1e-12)
        # Not a tautology: the weights genuinely changed the number.
        unweighted = F.cross_entropy(
            out["cognate_aa_logits"], tgt["cognate_aa"], ignore_index=L.IGNORE_INDEX
        )
        assert not torch.allclose(parts["terms"]["cognate_aa"], unweighted, atol=1e-6)


@requires_torch
class TestUncertaintyWeighting:
    def test_the_objective_is_the_kendall_log_variance_form(self):
        """total = Σ [ exp(-s_i)·L_i + s_i/2 ], graded against the formula, not the code."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        cfg = L.Stage2LossConfig(weighting="uncertainty")
        s = L.uncertainty_log_variances(dtype=torch.float64)
        with torch.no_grad():
            s.copy_(torch.tensor([0.3, -0.7, 0.1, 0.0, 1.2, -0.4], dtype=torch.float64))
        total, parts = L.multitask_loss(out, tgt, config=cfg, log_variances=s)
        expected = sum(
            (
                torch.exp(-s[i]) * parts["terms"][term] + 0.5 * s[i]
                for i, term in enumerate(L.TERMS)
            ),
            start=torch.zeros((), dtype=torch.float64),
        )
        assert torch.allclose(total, expected, atol=1e-12)

    def test_zero_log_variances_recover_unit_weights(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        s = L.uncertainty_log_variances(dtype=torch.float64)
        total, parts = L.multitask_loss(
            out, tgt, config=L.Stage2LossConfig(weighting="uncertainty"), log_variances=s
        )
        plain = sum(
            (parts["terms"][term] for term in L.TERMS),
            start=torch.zeros((), dtype=total.dtype),
        )
        assert torch.allclose(total, plain, atol=1e-12)
        assert float(parts["regulariser"]) == 0.0

    def test_the_log_variances_receive_gradient(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        s = L.uncertainty_log_variances(dtype=torch.float64)
        total, _ = L.multitask_loss(
            out, tgt, config=L.Stage2LossConfig(weighting="uncertainty"), log_variances=s
        )
        total.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()
        assert float(s.grad.abs().sum()) > 0.0

    def test_an_unsupervised_task_contributes_no_regulariser(self):
        """Otherwise the optimiser gets a free s_i/2 to minimise, driving that weight to ∞."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths, supervise_aux=False)
        s = L.uncertainty_log_variances(dtype=torch.float64)
        total, parts = L.multitask_loss(
            out, tgt, config=L.Stage2LossConfig(weighting="uncertainty"), log_variances=s
        )
        total.backward()
        assert parts["included"] == [L.BINARY_TERM]
        for index, term in enumerate(L.TERMS):
            if term == L.BINARY_TERM:
                assert float(s.grad[index].abs()) > 0.0
            else:
                assert float(s.grad[index]) == 0.0

    def test_the_no_aux_arm_drops_the_aux_tasks_entirely(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        s = L.uncertainty_log_variances(dtype=torch.float64)
        total, parts = L.multitask_loss(
            out,
            tgt,
            config=L.Stage2LossConfig(weighting="uncertainty", aux_weight=0.0),
            log_variances=s,
        )
        assert parts["included"] == [L.BINARY_TERM]
        assert torch.allclose(total, parts["terms"]["binary"], atol=1e-14)

    def test_missing_or_stray_log_variances_are_refused(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        with pytest.raises(ValueError, match="needs log_variances"):
            L.multitask_loss(out, tgt, config=L.Stage2LossConfig(weighting="uncertainty"))
        with pytest.raises(ValueError, match="silently ignored"):
            L.multitask_loss(
                out, tgt, log_variances=L.uncertainty_log_variances(dtype=torch.float64)
            )

    def test_wrong_sized_log_variances_are_refused(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        with pytest.raises(ValueError, match="TERMS order"):
            L.multitask_loss(
                out,
                tgt,
                config=L.Stage2LossConfig(weighting="uncertainty"),
                log_variances=torch.zeros(3, dtype=torch.float64),
            )


# ====================================================================================== #
# Torch tier — finiteness, gradient flow, and the fail-closed guards
# ====================================================================================== #
@requires_torch
class TestGradientFlowAndFiniteness:
    @pytest.mark.parametrize("gamma_name", ["0.0", "0.5", "2.0"])
    def test_every_term_is_finite_and_gradients_reach_every_head(self, gamma_name):
        """γ=0.5 is the regime whose pow() backward is nan without P2-02's clamp_min."""
        gamma = float(gamma_name)
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        cfg = L.Stage2LossConfig(binary_gamma=gamma, boundary_gamma=gamma, aux_gamma=gamma)
        total, parts = L.multitask_loss(out, tgt, config=cfg)
        for term in L.TERMS:
            assert torch.isfinite(parts["terms"][term]), term
        assert torch.isfinite(total)
        total.backward()
        for term in L.TERMS:
            grad = out[L.TERM_TO_LOGITS[term]].grad
            assert grad is not None, term
            assert torch.isfinite(grad).all(), term
            assert float(grad.abs().sum()) > 0.0, term

    def test_a_confident_correct_prediction_does_not_nan_the_gradient(self):
        """The P2-02 landmine, re-checked through the Stage-2 composition."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        with torch.no_grad():
            out["cognate_aa_logits"] *= 0.0
            out["cognate_aa_logits"][:, 0] = 60.0
        tgt["cognate_aa"] = torch.zeros_like(tgt["cognate_aa"])
        total, parts = L.multitask_loss(out, tgt, config=L.Stage2LossConfig(aux_gamma=0.5))
        assert torch.isfinite(parts["terms"]["cognate_aa"])
        total.backward()
        assert torch.isfinite(out["cognate_aa_logits"].grad).all()


@requires_torch
class TestFailClosedGuards:
    def test_a_missing_target_term_raises(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        del tgt["specifier_codon"]
        with pytest.raises(ValueError, match="missing terms"):
            L.multitask_loss(out, tgt)

    def test_a_mis_keyed_target_raises_rather_than_training_nothing(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        tgt["codon"] = tgt.pop("specifier_codon")
        with pytest.raises(ValueError, match="missing terms|unknown keys"):
            L.multitask_loss(out, tgt)

    def test_a_missing_output_raises(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        del out["nucleotide_mask"]
        with pytest.raises(ValueError, match="nucleotide_mask"):
            L.multitask_loss(out, tgt)

    def test_a_float_target_is_refused(self):
        """A float target would truncate toward zero and silently land on class 0."""
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        tgt["binary"] = tgt["binary"].to(torch.float32)
        with pytest.raises(ValueError, match="integer tensor"):
            L.multitask_loss(out, tgt)

    def test_an_out_of_range_binary_target_is_refused(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        tgt["binary"][0] = 3
        with pytest.raises(ValueError, match="binary targets"):
            L.multitask_loss(out, tgt)

    def test_the_boundary_crf_is_refused_not_silently_ignored(self):
        """A CE term would leave the CRF's transitions with no gradient at all."""
        from tbox_finder.models.seg_head import LinearChainCRF

        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        with pytest.raises(NotImplementedError, match="P3-06"):
            L.multitask_loss(out, tgt, boundary_crf=LinearChainCRF(WIDTHS["boundary"]))

    def test_a_shape_mismatch_raises(self):
        out, lengths = _outputs()
        tgt = _targets(out, lengths)
        tgt["regulatory_mode"] = tgt["regulatory_mode"][:2]
        with pytest.raises(ValueError, match="regulatory_mode target"):
            L.multitask_loss(out, tgt)


# ====================================================================================== #
# Torch tier — drift guard against the model's own output keys
# ====================================================================================== #
@requires_torch
class TestModelContractDrift:
    def test_the_logit_keys_are_the_models_own_head_outputs(self):
        """``losses`` names these by hand (it must stay torch-free); this is the guard."""
        from tbox_finder.stage2.model import HEAD_OUTPUT_KEYS, OUTPUT_KEYS

        assert tuple(L.TERM_TO_LOGITS[term] for term in L.TERMS) == HEAD_OUTPUT_KEYS
        assert L.NUCLEOTIDE_MASK_KEY in OUTPUT_KEYS
        assert set(OUTPUT_KEYS) == set(HEAD_OUTPUT_KEYS) | {L.NUCLEOTIDE_MASK_KEY}

    def test_a_real_model_forward_feeds_the_loss(self):
        """End-to-end shape contract: the model's dict is what multitask_loss consumes."""
        from tbox_finder.stage2.model import Stage2Model

        torch.manual_seed(0)
        model = Stage2Model(SPEC, d_model=16).double()
        batch, tokens = 3, 11
        hidden = torch.randn(batch, tokens, 16, dtype=torch.float64)
        attention_mask = torch.ones(batch, tokens, dtype=torch.long)
        attention_mask[2, -3:] = 0
        out = model.heads_from_hidden(hidden, attention_mask)
        length = out["boundary_logits"].shape[1]
        lengths = [int(n) for n in out["nucleotide_mask"].sum(dim=1)]
        boundary = torch.full((batch, length), L.IGNORE_INDEX, dtype=torch.long)
        for row, n in enumerate(lengths):
            boundary[row, :n] = torch.randint(0, WIDTHS["boundary"], (n,))
        targets = {"binary": torch.randint(0, 2, (batch,)), "boundary": boundary}
        for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family"):
            targets[term] = torch.randint(0, WIDTHS[term], (batch,))
        total, parts = L.multitask_loss(out, targets)
        assert torch.isfinite(total)
        assert parts["n_supervised"]["boundary"] == sum(lengths)
        total.backward()
        assert torch.isfinite(model.tbox_head.weight.grad).all()
