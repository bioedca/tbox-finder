"""Unit tests for the P2-02 Stage-1 objective (`tbox_finder.train.objective`).

Two tiers, mirroring `tests/unit/test_seg_head.py`:

* a **bare stdlib tier** that always runs in CI — the class-weight arithmetic and the
  loss-mass diagnostic are torch-free by design;
* a **torch tier** (`@requires_torch`) skipped in bare CI, run under `tbox-ml-dna` for
  the §8.5 manual gate. CPU is sufficient (a handful of small tensors).

The load-bearing test is `test_gamma_zero_weighted_equals_torch_weighted_ce`: it pins the
γ=0 reduction against **`torch.nn.functional.cross_entropy`** — an external reference —
rather than against this module's own unweighted path, which would be the tautology class
that bit P1-15 (`lora_config_exact`) and P2-01 (the RC golden case). An anti-tautology
test confirms the assertion actually bites when the denominator convention is wrong.
"""

from __future__ import annotations

import math

import pytest

from tbox_finder.labels import CLASS_ORDER, CORE_ELEMENTS
from tbox_finder.train.objective import (
    DEFAULT_CLASS_WEIGHT_ALPHA,
    DEFAULT_GAMMA,
    IGNORE_INDEX,
    NUM_CLASSES,
    Stage1Loss,
    Stage1LossConfig,
    class_weights_from_counts,
    core_mass_share,
    focal_cross_entropy,
    left_align_for_crf,
    loss_mass_share,
)

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in the bare CI env
    if exc.name != "torch":
        raise
    torch = None
    F = None
    LinearChainCRF = None
    _HAS_TORCH = False
else:
    # Deliberately OUTSIDE the guard: a broken tbox_finder.models.seg_head must raise here,
    # not be swallowed into `_HAS_TORCH = False` and silently skip the whole tensor tier
    # green. That self-skipping-green failure mode already bit P1-15.
    import torch.nn.functional as F

    from tbox_finder.models.seg_head import LinearChainCRF

    _HAS_TORCH = True

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not installed (bare CI) — objective tensor tier runs under tbox-ml-dna",
)

# The measured P2-02 training-stream class counts (8,303 nested_train windows of 1,024 nt;
# see the dev-log stanza). Written out as literals so a change in the datamodule that
# shifts the class distribution shows up here as a deliberate edit.
MEASURED_COUNTS = {
    "background": 6_646_421,
    "Stem_I": 769_403,
    "Specifier": 24_412,
    "Stem_II": 316_028,
    "Stem_III": 191_251,
    "Antiterminator_Tbox_seq": 231_557,
    "Terminator": 235_346,
    "Discriminator": 32_589,
}


# --------------------------------------------------------------------------- #
# Bare tier — class weights
# --------------------------------------------------------------------------- #
class TestClassWeights:
    def test_alpha_zero_is_all_ones(self):
        w = class_weights_from_counts(MEASURED_COUNTS, alpha=0.0)
        assert w == pytest.approx([1.0] * NUM_CLASSES)

    def test_weights_are_mean_normalised(self):
        for alpha in (0.0, 0.25, 0.5, 1.0):
            w = class_weights_from_counts(MEASURED_COUNTS, alpha=alpha)
            assert sum(w) / len(w) == pytest.approx(1.0)

    def test_alpha_one_equalises_loss_mass(self):
        """Full inverse-frequency ⇒ every class takes exactly 1/8 of the mass."""
        w = class_weights_from_counts(MEASURED_COUNTS, alpha=1.0)
        share = loss_mass_share(MEASURED_COUNTS, w)
        for name in CLASS_ORDER:
            assert share[name] == pytest.approx(1.0 / NUM_CLASSES)

    def test_rarer_class_gets_more_weight(self):
        w = class_weights_from_counts(MEASURED_COUNTS, alpha=1.0)
        idx = {name: i for i, name in enumerate(CLASS_ORDER)}
        # Specifier (24,412 nt) is ~272x rarer than background (6,646,421 nt).
        assert w[idx["Specifier"]] > w[idx["background"]]
        ratio = w[idx["Specifier"]] / w[idx["background"]]
        assert ratio == pytest.approx(
            MEASURED_COUNTS["background"] / MEASURED_COUNTS["Specifier"], rel=1e-9
        )

    def test_alpha_interpolates_monotonically(self):
        idx = {name: i for i, name in enumerate(CLASS_ORDER)}
        prev = None
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            w = class_weights_from_counts(MEASURED_COUNTS, alpha=alpha)
            spec = w[idx["Specifier"]]
            if prev is not None:
                assert spec > prev
            prev = spec

    def test_sequence_and_mapping_agree(self):
        seq = [MEASURED_COUNTS[n] for n in CLASS_ORDER]
        assert class_weights_from_counts(seq, alpha=0.5) == pytest.approx(
            class_weights_from_counts(MEASURED_COUNTS, alpha=0.5)
        )

    def test_zero_count_raises_rather_than_emitting_inf(self):
        counts = dict(MEASURED_COUNTS, Stem_III=0)
        with pytest.raises(ValueError, match="zero nucleotides"):
            class_weights_from_counts(counts, alpha=1.0)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            class_weights_from_counts(MEASURED_COUNTS, alpha=-1.0)

    def test_missing_class_raises(self):
        counts = {k: v for k, v in MEASURED_COUNTS.items() if k != "Terminator"}
        with pytest.raises(ValueError, match="missing classes"):
            class_weights_from_counts(counts, alpha=1.0)

    def test_unknown_class_raises(self):
        counts = dict(MEASURED_COUNTS, Pseudoknot=5)
        with pytest.raises(ValueError, match="unknown classes"):
            class_weights_from_counts(counts, alpha=1.0)

    def test_bool_count_rejected(self):
        """bool is an int subclass — the P1-15/P1-16 bool-as-count defect class."""
        counts = dict(MEASURED_COUNTS, Specifier=True)
        with pytest.raises(TypeError, match="got bool"):
            class_weights_from_counts(counts, alpha=1.0)

    def test_numpy_integer_counts_accepted(self):
        """np.bincount is how a caller tallies these; np.int64 is not a python int.

        A bare isinstance(value, int) check rejects the entire numpy pipeline — the exact
        way P2-04 will produce these counts.
        """
        np = pytest.importorskip("numpy")
        seq = np.array([MEASURED_COUNTS[n] for n in CLASS_ORDER], dtype=np.int64)
        got = class_weights_from_counts(seq, alpha=1.0)
        assert got == pytest.approx(class_weights_from_counts(MEASURED_COUNTS, alpha=1.0))

    def test_numpy_bincount_output_accepted(self):
        np = pytest.importorskip("numpy")
        labels = np.repeat(np.arange(NUM_CLASSES), [10, 9, 8, 7, 6, 5, 4, 3])
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        w = class_weights_from_counts(counts, alpha=1.0)
        assert len(w) == NUM_CLASSES
        assert all(x > 0 for x in w)

    def test_numpy_bool_and_float_counts_rejected(self):
        """np.bool_ / np.float64 must not slip through the integer protocol."""
        np = pytest.importorskip("numpy")
        with pytest.raises(TypeError, match="must be an integer"):
            class_weights_from_counts(dict(MEASURED_COUNTS, Specifier=np.bool_(True)), alpha=1.0)
        with pytest.raises(TypeError, match="must be an integer"):
            class_weights_from_counts(dict(MEASURED_COUNTS, Specifier=np.float64(5.0)), alpha=1.0)

    def test_float_and_str_counts_rejected(self):
        with pytest.raises(TypeError, match="must be an integer"):
            class_weights_from_counts(dict(MEASURED_COUNTS, Specifier=5.0), alpha=1.0)
        with pytest.raises(TypeError, match="must be an integer"):
            class_weights_from_counts(dict(MEASURED_COUNTS, Specifier="5"), alpha=1.0)

    def test_negative_count_rejected(self):
        counts = dict(MEASURED_COUNTS, Specifier=-5)
        with pytest.raises(ValueError, match="must be >= 0"):
            class_weights_from_counts(counts, alpha=1.0)

    def test_wrong_length_sequence_rejected(self):
        with pytest.raises(ValueError, match="must have 8 entries"):
            class_weights_from_counts([1, 2, 3], alpha=1.0)


# --------------------------------------------------------------------------- #
# Bare tier — the loss-mass consequence diagnostic
# --------------------------------------------------------------------------- #
class TestLossMassShare:
    def test_unweighted_share_is_the_class_frequency(self):
        share = loss_mass_share(MEASURED_COUNTS)
        total = sum(MEASURED_COUNTS.values())
        for name in CLASS_ORDER:
            assert share[name] == pytest.approx(MEASURED_COUNTS[name] / total)

    def test_shares_sum_to_one(self):
        for alpha in (0.0, 0.5, 1.0):
            w = class_weights_from_counts(MEASURED_COUNTS, alpha=alpha)
            assert sum(loss_mass_share(MEASURED_COUNTS, w).values()) == pytest.approx(1.0)

    def test_share_is_invariant_to_a_global_weight_rescale(self):
        w = class_weights_from_counts(MEASURED_COUNTS, alpha=0.5)
        scaled = [7.5 * x for x in w]
        a = loss_mass_share(MEASURED_COUNTS, w)
        b = loss_mass_share(MEASURED_COUNTS, scaled)
        for name in CLASS_ORDER:
            assert a[name] == pytest.approx(b[name])

    def test_dense_background_regime_is_real(self):
        """The regime PRD §11's objective exists to handle, pinned as measured."""
        share = loss_mass_share(MEASURED_COUNTS)
        assert share["background"] > 0.75
        assert share["Specifier"] < 0.005

    def test_alpha_one_hands_half_the_mass_to_the_four_gate4_excluded_classes(self):
        """ADR-0004 D6 excludes Stem_II/Stem_III/Terminator/Discriminator from GATE-4.

        Under full inverse-frequency they command 4/8 of the loss mass while the three
        graded core elements get 3/8 — the reason the default is alpha=0 and the reason
        this diagnostic exists. Reported, not gated: no ADR pins a floor.
        """
        w = class_weights_from_counts(MEASURED_COUNTS, alpha=1.0)
        share = loss_mass_share(MEASURED_COUNTS, w)
        excluded = ("Stem_II", "Stem_III", "Terminator", "Discriminator")
        assert sum(share[n] for n in excluded) == pytest.approx(0.5)
        assert core_mass_share(MEASURED_COUNTS, w) == pytest.approx(3.0 / 8.0)

    def test_core_mass_share_matches_the_adr0004_core_elements(self):
        share = loss_mass_share(MEASURED_COUNTS)
        assert core_mass_share(MEASURED_COUNTS) == pytest.approx(
            sum(share[n] for n in CORE_ELEMENTS)
        )

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match=r"weight\[0\] must be >= 0"):
            loss_mass_share(MEASURED_COUNTS, [-1.0] + [1.0] * 7)

    def test_wrong_weight_length_rejected(self):
        with pytest.raises(ValueError, match="must have 8 entries"):
            loss_mass_share(MEASURED_COUNTS, [1.0, 2.0])


# --------------------------------------------------------------------------- #
# Bare tier — constants + config
# --------------------------------------------------------------------------- #
class TestConstantsAndConfig:
    def test_ignore_index_matches_the_repo_wide_sentinel(self):
        from tbox_finder.data.seg_smoke import IGNORE_INDEX as SEG_IGNORE

        assert IGNORE_INDEX == SEG_IGNORE == -100

    def test_ignore_index_matches_the_datamodule_that_feeds_this_loss(self):
        from tbox_finder.data.window_dataset import IGNORE_INDEX as WD_IGNORE

        assert IGNORE_INDEX == WD_IGNORE

    def test_num_classes_is_single_sourced(self):
        assert NUM_CLASSES == len(CLASS_ORDER) == 8

    def test_default_gamma_is_the_prd_value(self):
        assert DEFAULT_GAMMA == 2.0

    def test_default_expresses_focal_or_inverse_frequency_not_both(self):
        """PRD §11 offers them as alternatives; the default must pick exactly one."""
        assert DEFAULT_CLASS_WEIGHT_ALPHA == 0.0
        assert Stage1Loss().weights is None

    def test_config_rejects_negative_values(self):
        for kwargs in ({"gamma": -1.0}, {"class_weight_alpha": -0.5}, {"crf_weight": -2.0}):
            with pytest.raises(ValueError):
                Stage1LossConfig(**kwargs)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("field", ["gamma", "class_weight_alpha", "crf_weight"])
    def test_config_rejects_non_finite_values(self, field, bad):
        """`x < 0` is False for NaN, so a bare range check lets nan through silently."""
        with pytest.raises(ValueError, match="must be finite"):
            Stage1LossConfig(**{field: bad})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_class_weights_rejects_non_finite_alpha(self, bad):
        with pytest.raises(ValueError, match="must be finite"):
            class_weights_from_counts(MEASURED_COUNTS, alpha=bad)

    def test_loss_mass_share_rejects_non_finite_weights(self):
        with pytest.raises(ValueError, match="must be finite"):
            loss_mass_share(MEASURED_COUNTS, [float("nan")] + [1.0] * 7)

    def test_alpha_without_counts_raises_rather_than_assuming_frequencies(self):
        with pytest.raises(ValueError, match="needs class_counts"):
            Stage1Loss(Stage1LossConfig(class_weight_alpha=1.0))

    def test_weights_are_built_when_counts_supplied(self):
        loss = Stage1Loss(Stage1LossConfig(class_weight_alpha=1.0), class_counts=MEASURED_COUNTS)
        assert loss.weights == pytest.approx(class_weights_from_counts(MEASURED_COUNTS, alpha=1.0))

    def test_diagnostics_reports_unpinned_and_omits_unmeasured_mass(self):
        bare = Stage1Loss().diagnostics()
        assert bare["pinned"] is False
        assert "loss_mass_share" not in bare  # nothing measured ⇒ nothing reported
        with_counts = Stage1Loss(class_counts=MEASURED_COUNTS).diagnostics()
        assert with_counts["loss_mass_share"]["background"] > 0.75
        assert with_counts["core_elements"] == list(CORE_ELEMENTS)


# --------------------------------------------------------------------------- #
# Torch tier — focal cross-entropy
# --------------------------------------------------------------------------- #
def _fixture(b=3, length=11, c=NUM_CLASSES, seed=0, n_ignored=4):
    torch.manual_seed(seed)
    logits = torch.randn(b, length, c, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, c, (b, length))
    flat = targets.view(-1)
    flat[torch.randperm(flat.numel())[:n_ignored]] = IGNORE_INDEX
    return logits, targets


@requires_torch
class TestFocalCrossEntropy:
    def test_gamma_zero_unweighted_equals_torch_cross_entropy(self):
        """γ=0 with no weights ≡ F.cross_entropy(reduction='mean') — torch as reference."""
        logits, targets = _fixture()
        got = focal_cross_entropy(logits, targets, gamma=0.0)
        want = F.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=IGNORE_INDEX, reduction="mean"
        )
        assert torch.allclose(got, want, atol=1e-12)

    def test_gamma_zero_weighted_equals_torch_weighted_ce(self):
        """The load-bearing gate (imp.md P2-02): γ=0 ≡ **weighted** CE within atol.

        Graded against torch's own weighted cross-entropy, which normalises the 'mean'
        by the total weight of the non-ignored targets. Comparing instead against this
        module's unweighted path would be a tautology.
        """
        logits, targets = _fixture()
        w = torch.tensor(class_weights_from_counts(MEASURED_COUNTS, alpha=1.0), dtype=torch.float64)
        got = focal_cross_entropy(logits, targets, gamma=0.0, weight=w)
        want = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            weight=w,
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        assert torch.allclose(got, want, atol=1e-12)

    def test_the_weighted_equivalence_bites_if_the_denominator_is_wrong(self):
        """Anti-tautology: the wrong (token-count) denominator must FAIL the assertion.

        This is the exact mistake the naive implementation makes — dividing by the valid
        token count instead of Σ w_y. If that still passed, the test above would be
        vacuous.
        """
        logits, targets = _fixture()
        w = torch.tensor(class_weights_from_counts(MEASURED_COUNTS, alpha=1.0), dtype=torch.float64)
        ce = F.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=IGNORE_INDEX, reduction="none"
        )
        valid = targets != IGNORE_INDEX
        wrong = (ce * w[targets.clamp_min(0)] * valid).sum() / valid.sum()
        want = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            weight=w,
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        assert not torch.allclose(wrong, want, atol=1e-6)

    def test_focal_matches_a_closed_form_hand_derivation(self):
        """The independent value pin for γ>0: FL = (1-p_t)^γ · (-log p_t), by hand.

        The expectation is derived with stdlib ``math`` from the Lin et al. formula and
        touches neither ``F.cross_entropy`` nor this module, so it pins the *value* of the
        focal term rather than just its agreement with a sibling implementation. Comparing
        against ``stage1_smoke.focal_cross_entropy`` cannot do this job — that function now
        delegates here, so such a test would compare this code to itself.
        """
        raw = [[2.0, 1.0, 0.5], [0.1, 0.2, 0.3]]
        tgt = [0, 2]
        for gamma in (0.0, 0.5, 1.0, 2.0):
            terms = []
            for row, t in zip(raw, tgt, strict=True):
                shift = max(row)
                exps = [math.exp(x - shift) for x in row]
                p_t = exps[t] / sum(exps)
                terms.append((1.0 - p_t) ** gamma * (-math.log(p_t)))
            want = sum(terms) / len(terms)
            got = focal_cross_entropy(
                torch.tensor([raw], dtype=torch.float64), torch.tensor([tgt]), gamma=gamma
            )
            assert float(got) == pytest.approx(want, abs=1e-12), f"gamma={gamma}"

    def test_weighted_focal_matches_a_closed_form_hand_derivation(self):
        """Same independent pin, with class weights: Σ w·(1-p_t)^γ·(-log p_t) / Σ w."""
        raw = [[2.0, 1.0, 0.5], [0.1, 0.2, 0.3]]
        tgt = [0, 2]
        w = [3.0, 1.0, 0.5]
        for gamma in (0.0, 0.5, 2.0):
            num = 0.0
            den = 0.0
            for row, t in zip(raw, tgt, strict=True):
                shift = max(row)
                exps = [math.exp(x - shift) for x in row]
                p_t = exps[t] / sum(exps)
                num += w[t] * (1.0 - p_t) ** gamma * (-math.log(p_t))
                den += w[t]
            got = focal_cross_entropy(
                torch.tensor([raw], dtype=torch.float64),
                torch.tensor([tgt]),
                gamma=gamma,
                weight=torch.tensor(w, dtype=torch.float64),
            )
            assert float(got) == pytest.approx(num / den, abs=1e-12), f"gamma={gamma}"

    def test_p1_smoke_entrypoint_still_matches_the_closed_form(self):
        """P1-06's public entry point is preserved by the delegation (same values).

        Pinned against the independent hand derivation, not against this module's
        focal_cross_entropy — after the promotion that comparison would be circular.
        """
        from tbox_finder.train.stage1_smoke import focal_cross_entropy as p1_focal

        raw = [[2.0, 1.0, 0.5], [0.1, 0.2, 0.3]]
        tgt = [0, 2]
        for gamma in (0.0, 0.5, 2.0):
            terms = []
            for row, t in zip(raw, tgt, strict=True):
                shift = max(row)
                exps = [math.exp(x - shift) for x in row]
                p_t = exps[t] / sum(exps)
                terms.append((1.0 - p_t) ** gamma * (-math.log(p_t)))
            got = p1_focal(
                torch.tensor([raw], dtype=torch.float64), torch.tensor([tgt]), gamma=gamma
            )
            assert float(got) == pytest.approx(sum(terms) / len(terms), abs=1e-12)

    def test_class_weight_is_applied(self):
        logits, targets = _fixture()
        plain = focal_cross_entropy(logits, targets, gamma=2.0)
        w = torch.tensor(class_weights_from_counts(MEASURED_COUNTS, alpha=1.0), dtype=torch.float64)
        weighted = focal_cross_entropy(logits, targets, gamma=2.0, weight=w)
        assert not torch.allclose(plain, weighted)

    def test_mean_reduction_is_invariant_to_a_global_weight_rescale(self):
        """Σ(k·w·ce)/Σ(k·w) cancels k — so mean-normalising the weights is cosmetic."""
        logits, targets = _fixture()
        w = torch.tensor(class_weights_from_counts(MEASURED_COUNTS, alpha=0.5), dtype=torch.float64)
        a = focal_cross_entropy(logits, targets, gamma=2.0, weight=w)
        b = focal_cross_entropy(logits, targets, gamma=2.0, weight=w * 13.7)
        assert torch.allclose(a, b, atol=1e-12)

    def test_focal_downweights_easy_examples(self):
        """γ>0 must shrink the loss of a confidently-correct position relative to γ=0."""
        logits = torch.zeros(1, 1, NUM_CLASSES, dtype=torch.float64)
        logits[0, 0, 3] = 10.0  # confident + correct
        targets = torch.tensor([[3]])
        ce = focal_cross_entropy(logits, targets, gamma=0.0)
        focal = focal_cross_entropy(logits, targets, gamma=2.0)
        assert focal < ce

    def test_focal_barely_touches_a_hard_example(self):
        """A confidently-WRONG position keeps ~its full loss (p_t→0 ⇒ (1-p_t)^γ→1)."""
        logits = torch.zeros(1, 1, NUM_CLASSES, dtype=torch.float64)
        logits[0, 0, 5] = 20.0  # confident, but the target is class 3
        targets = torch.tensor([[3]])
        ce = focal_cross_entropy(logits, targets, gamma=0.0)
        focal = focal_cross_entropy(logits, targets, gamma=2.0)
        assert focal / ce > 0.99

    def test_ignored_positions_contribute_nothing(self):
        """Flipping a logit under an ignored target must not move the loss.

        The perturbation must be **non-uniform** across the class dimension: softmax is
        shift-invariant, so bumping every class logit by the same constant is a no-op and
        the test would pass even if the ignored position *were* wrongly included. The
        control below proves the same perturbation does move the loss when the position is
        not ignored — without it this assertion would be vacuous.
        """
        logits, targets = _fixture(n_ignored=0)
        real_target = int(targets[0, 0])
        perturbed = logits.detach().clone()
        perturbed[0, 0, 0] += 100.0  # single class -> genuinely changes the softmax

        # Control: while the position is real, the perturbation *does* move the loss.
        assert not torch.allclose(
            focal_cross_entropy(logits, targets, gamma=2.0),
            focal_cross_entropy(perturbed, targets, gamma=2.0),
            atol=1e-9,
        ), "perturbation is inert — the ignore assertion below would be vacuous"
        assert real_target is not None

        # Now ignore it: the same perturbation must become invisible.
        targets[0, 0] = IGNORE_INDEX
        base = focal_cross_entropy(logits, targets, gamma=2.0)
        got = focal_cross_entropy(perturbed, targets, gamma=2.0)
        assert torch.allclose(base, got, atol=1e-12)

    def test_ignored_positions_excluded_from_the_weighted_denominator(self):
        logits, targets = _fixture(n_ignored=0)
        targets[0, 0] = IGNORE_INDEX
        w = torch.tensor(class_weights_from_counts(MEASURED_COUNTS, alpha=1.0), dtype=torch.float64)
        got = focal_cross_entropy(logits, targets, gamma=0.0, weight=w)
        want = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            weight=w,
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        assert torch.allclose(got, want, atol=1e-12)

    def test_all_ignored_batch_is_zero_not_nan(self):
        logits, targets = _fixture(n_ignored=0)
        targets[:] = IGNORE_INDEX
        for weight in (None, torch.ones(NUM_CLASSES, dtype=torch.float64)):
            got = focal_cross_entropy(logits, targets, gamma=2.0, weight=weight)
            assert torch.isfinite(got)
            assert float(got) == pytest.approx(0.0)

    def test_reduction_none_and_sum_are_consistent(self):
        logits, targets = _fixture()
        per_token = focal_cross_entropy(logits, targets, gamma=2.0, reduction="none")
        assert per_token.shape == targets.shape
        total = focal_cross_entropy(logits, targets, gamma=2.0, reduction="sum")
        assert torch.allclose(per_token.sum(), total, atol=1e-12)
        mean = focal_cross_entropy(logits, targets, gamma=2.0, reduction="mean")
        assert torch.allclose(total / (targets != IGNORE_INDEX).sum(), mean, atol=1e-12)

    def test_gradient_flows(self):
        logits, targets = _fixture()
        focal_cross_entropy(logits, targets, gamma=2.0).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert float(logits.grad.abs().sum()) > 0.0

    def test_gradient_is_zero_at_ignored_positions(self):
        logits, targets = _fixture(n_ignored=0)
        targets[1, 2] = IGNORE_INDEX
        focal_cross_entropy(logits, targets, gamma=2.0).backward()
        assert torch.allclose(
            logits.grad[1, 2], torch.zeros(NUM_CLASSES, dtype=torch.float64), atol=1e-14
        )

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 0.9, 0.99])
    def test_no_nan_gradient_for_fractional_gamma_on_a_confident_position(self, gamma):
        """Regression: 0 < γ < 1 silently NaN-poisoned every gradient.

        A confidently-correct position drives ce to *exactly* 0.0, so 1-p_t is exactly 0;
        pow's backward γ·x^(γ-1) is then inf, and grad_out there is ce = 0, giving
        inf·0 = nan — while the forward loss stays finite. γ<1 is on the swept axis
        (Lin et al.'s own grid includes 0.5), so this had to be fixed rather than banned.
        """
        logits = torch.zeros(1, 2, NUM_CLASSES, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 3] = 200.0  # confidently correct -> ce == 0.0 exactly
            logits[0, 1, 1] = 1.0
        targets = torch.tensor([[3, 1]])
        loss = focal_cross_entropy(logits, targets, gamma=gamma)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all(), f"gamma={gamma} produced non-finite grads"

    # NB: parametrise over dtype *names*, not torch.dtype objects — decorator arguments are
    # evaluated at collection time, when `torch` is None in bare CI, so a torch.float32 here
    # would raise AttributeError and take down collection of the whole file (skipif marks are
    # applied too late to help).
    @pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
    def test_no_nan_gradient_at_the_measured_ce_zero_threshold(self, dtype_name):
        """The trigger is reachable in the real training dtype, not just a synthetic one.

        Measured on this exact call path with the pinned torch 2.7.1+cu128: cross-entropy
        underflows to exactly 0.0 at a correct-class logit gap of 18 in both bf16 (the
        Stage-1 training dtype, P1-16) and fp32 — the K-dim kernel accumulates bf16 in
        fp32, so the two thresholds coincide. A margin of 18 is routine once the model is
        confident on the 78.7% background. The assert below fails loudly if a torch bump
        moves the threshold, rather than letting the regression test quietly stop testing.
        """
        gap = 18
        dtype = getattr(torch, dtype_name)
        logits = torch.zeros(1, 2, NUM_CLASSES, dtype=dtype, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 3] = float(gap)
            logits[0, 1, 1] = 1.0
        targets = torch.tensor([[3, 1]])
        ce = F.cross_entropy(
            logits.detach().transpose(1, 2), targets, reduction="none", ignore_index=IGNORE_INDEX
        )
        assert float(ce[0, 0]) == 0.0, "fixture no longer reaches the ce==0 regime"
        focal_cross_entropy(logits, targets, gamma=0.5).backward()
        assert torch.isfinite(logits.grad).all()

    def test_the_nan_guard_leaves_normal_gradients_bit_identical(self):
        """The eps clamp must not perturb training at the γ values that were already safe."""
        for gamma in (0.0, 1.0, 2.0):
            grads = []
            for clamp_floor in (0.0, torch.finfo(torch.float64).eps):
                torch.manual_seed(0)
                lg = torch.randn(3, 11, NUM_CLASSES, dtype=torch.float64, requires_grad=True)
                tg = torch.randint(0, NUM_CLASSES, (3, 11))
                ce = F.cross_entropy(
                    lg.transpose(1, 2), tg, ignore_index=IGNORE_INDEX, reduction="none"
                )
                pt = torch.exp(-ce)
                out = ((1.0 - pt).clamp_min(clamp_floor).pow(gamma) * ce).sum() / (
                    tg != IGNORE_INDEX
                ).sum()
                out.backward()
                grads.append(lg.grad.clone())
            assert torch.equal(grads[0], grads[1]), f"eps clamp moved gradients at gamma={gamma}"

    def test_p1_smoke_entrypoint_inherits_the_nan_fix(self):
        """The P1-06 alias carried the identical defect; the promotion fixes it there too."""
        from tbox_finder.train.stage1_smoke import focal_cross_entropy as p1_focal

        logits = torch.zeros(1, 2, NUM_CLASSES, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 3] = 200.0
            logits[0, 1, 1] = 1.0
        p1_focal(logits, torch.tensor([[3, 1]]), gamma=0.5).backward()
        assert torch.isfinite(logits.grad).all()

    def test_bad_shapes_and_args_raise(self):
        logits, targets = _fixture()
        with pytest.raises(ValueError, match="unknown reduction"):
            focal_cross_entropy(logits, targets, gamma=2.0, reduction="avg")
        with pytest.raises(ValueError, match="gamma must be >= 0"):
            focal_cross_entropy(logits, targets, gamma=-1.0)
        with pytest.raises(ValueError, match=r"logits must be \(B, L, C\)"):
            focal_cross_entropy(logits[0], targets, gamma=2.0)
        with pytest.raises(ValueError, match="targets shape"):
            focal_cross_entropy(logits, targets[:, :-1], gamma=2.0)
        with pytest.raises(ValueError, match="weight must be a 1-D tensor"):
            focal_cross_entropy(
                logits, targets, gamma=2.0, weight=torch.ones(3, dtype=torch.float64)
            )
        with pytest.raises(ValueError, match="non-negative"):
            focal_cross_entropy(
                logits, targets, gamma=2.0, weight=-torch.ones(NUM_CLASSES, dtype=torch.float64)
            )


# --------------------------------------------------------------------------- #
# Torch tier — CRF left-alignment
# --------------------------------------------------------------------------- #
@requires_torch
class TestLeftAlignForCrf:
    def test_left_flanked_window_would_crash_the_crf_unaligned(self):
        """The concrete failure the roll exists to fix (87/8303 real windows)."""
        crf = LinearChainCRF(NUM_CLASSES).double()
        emissions = torch.randn(1, 6, NUM_CLASSES, dtype=torch.float64)
        tags = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3, 0]])
        mask = tags != IGNORE_INDEX
        with pytest.raises(ValueError, match="left-aligned"):
            crf(emissions, tags.clamp_min(0), mask=mask)

    def test_roll_left_aligns_and_the_crf_then_accepts_it(self):
        crf = LinearChainCRF(NUM_CLASSES).double()
        emissions = torch.randn(1, 6, NUM_CLASSES, dtype=torch.float64)
        tags = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3, 0]])
        mask = tags != IGNORE_INDEX
        e, t, m = left_align_for_crf(emissions, tags, mask)
        assert m.tolist() == [[True, True, True, True, False, False]]
        assert t[0, :4].tolist() == [1, 2, 3, 0]
        nll = crf(e, t, mask=m, reduction="sum")
        assert torch.isfinite(nll)

    def test_roll_preserves_the_real_subsequence_exactly(self):
        emissions = torch.randn(2, 7, NUM_CLASSES, dtype=torch.float64)
        tags = torch.tensor(
            [
                [IGNORE_INDEX, 1, 2, 3, 4, 5, IGNORE_INDEX],
                [0, 1, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
            ]
        )
        mask = tags != IGNORE_INDEX
        e, t, m = left_align_for_crf(emissions, tags, mask)
        # row 0: real run is positions 1..5 -> rolled to 0..4
        assert t[0, :5].tolist() == [1, 2, 3, 4, 5]
        assert torch.allclose(e[0, :5], emissions[0, 1:6])
        assert m[0].tolist() == [True] * 5 + [False] * 2
        # row 1: already left-aligned -> unchanged
        assert t[1, :3].tolist() == [0, 1, 2]
        assert torch.allclose(e[1], emissions[1])

    def test_already_left_aligned_is_an_identity(self):
        emissions = torch.randn(2, 5, NUM_CLASSES, dtype=torch.float64)
        tags = torch.tensor([[0, 1, 2, 3, 4], [1, 1, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])
        mask = tags != IGNORE_INDEX
        e, t, m = left_align_for_crf(emissions, tags, mask)
        assert torch.allclose(e, emissions)
        assert torch.equal(m, mask)
        assert t[0].tolist() == [0, 1, 2, 3, 4]

    def test_crf_score_is_invariant_to_where_the_real_run_sits(self):
        """The roll is exact: the same real chain scores the same wherever it was carved.

        Left-flanked [pad, pad, A, B, C] and left-aligned [A, B, C, pad, pad] carrying the
        same emissions for A/B/C must yield the same CRF NLL.
        """
        torch.manual_seed(7)
        crf = LinearChainCRF(NUM_CLASSES).double()
        real_e = torch.randn(1, 3, NUM_CLASSES, dtype=torch.float64)
        real_t = [2, 5, 1]
        pad = [IGNORE_INDEX, IGNORE_INDEX]

        flanked_e = torch.cat([torch.randn(1, 2, NUM_CLASSES, dtype=torch.float64), real_e], dim=1)
        flanked_t = torch.tensor([pad + real_t])
        e1, t1, m1 = left_align_for_crf(flanked_e, flanked_t, flanked_t != IGNORE_INDEX)
        nll1 = crf(e1, t1, mask=m1, reduction="sum")

        aligned_e = torch.cat([real_e, torch.randn(1, 2, NUM_CLASSES, dtype=torch.float64)], dim=1)
        aligned_t = torch.tensor([real_t + pad])
        e2, t2, m2 = left_align_for_crf(aligned_e, aligned_t, aligned_t != IGNORE_INDEX)
        nll2 = crf(e2, t2, mask=m2, reduction="sum")

        assert torch.allclose(nll1, nll2, atol=1e-12)

    def test_interior_gap_raises_rather_than_silently_corrupting(self):
        emissions = torch.randn(1, 5, NUM_CLASSES, dtype=torch.float64)
        tags = torch.tensor([[0, IGNORE_INDEX, 2, 3, 4]])
        mask = tags != IGNORE_INDEX
        with pytest.raises(ValueError, match="non-contiguous"):
            left_align_for_crf(emissions, tags, mask)

    def test_fully_masked_sequence_raises(self):
        emissions = torch.randn(1, 4, NUM_CLASSES, dtype=torch.float64)
        tags = torch.full((1, 4), IGNORE_INDEX)
        with pytest.raises(ValueError, match="at least one unmasked"):
            left_align_for_crf(emissions, tags, tags != IGNORE_INDEX)

    def test_ignored_tags_are_clamped_into_range(self):
        emissions = torch.randn(1, 4, NUM_CLASSES, dtype=torch.float64)
        tags = torch.tensor([[1, 2, IGNORE_INDEX, IGNORE_INDEX]])
        _, t, _ = left_align_for_crf(emissions, tags, tags != IGNORE_INDEX)
        assert int(t.min()) >= 0
        assert int(t.max()) < NUM_CLASSES

    def test_bad_shapes_raise(self):
        emissions = torch.randn(1, 4, NUM_CLASSES, dtype=torch.float64)
        tags = torch.zeros(1, 4, dtype=torch.long)
        with pytest.raises(ValueError, match=r"emissions must be \(B, L, C\)"):
            left_align_for_crf(emissions[0], tags, tags == tags)
        with pytest.raises(ValueError, match="must both be"):
            left_align_for_crf(emissions, tags[:, :-1], tags == tags)


# --------------------------------------------------------------------------- #
# Torch tier — the composed objective
# --------------------------------------------------------------------------- #
@requires_torch
class TestStage1Loss:
    def test_default_matches_bare_focal(self):
        logits, targets = _fixture()
        got = Stage1Loss()(logits, targets)
        want = focal_cross_entropy(logits, targets, gamma=DEFAULT_GAMMA)
        assert torch.allclose(got, want, atol=1e-12)

    def test_weighted_config_applies_weights(self):
        logits, targets = _fixture()
        loss_fn = Stage1Loss(
            Stage1LossConfig(gamma=0.0, class_weight_alpha=1.0), class_counts=MEASURED_COUNTS
        )
        got = loss_fn(logits, targets)
        w = torch.tensor(class_weights_from_counts(MEASURED_COUNTS, alpha=1.0), dtype=torch.float64)
        want = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            weight=w,
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        assert torch.allclose(got, want, atol=1e-12)

    def test_crf_term_adds_and_is_per_token_scaled(self):
        logits, targets = _fixture(b=2, length=9, n_ignored=0)
        crf = LinearChainCRF(NUM_CLASSES).double()
        plain = Stage1Loss()(logits, targets)
        loss_fn = Stage1Loss(Stage1LossConfig(use_crf=True))
        total, parts = loss_fn(logits, targets, crf=crf, return_components=True)
        assert torch.allclose(parts["focal_ce"], plain, atol=1e-12)
        assert torch.allclose(total, parts["focal_ce"] + parts["crf_nll_per_token"], atol=1e-12)
        # Per-token normalisation: an un-normalised per-sequence NLL over L=9 would be
        # ~L times larger. Pin the relationship to the CRF's own sum reduction.
        raw = crf(logits, targets, mask=torch.ones_like(targets, dtype=torch.bool), reduction="sum")
        assert torch.allclose(parts["crf_nll_per_token"], raw / targets.numel(), atol=1e-12)

    def test_crf_weight_scales_the_term(self):
        logits, targets = _fixture(b=2, length=9, n_ignored=0)
        crf = LinearChainCRF(NUM_CLASSES).double()
        _, a = Stage1Loss(Stage1LossConfig(use_crf=True, crf_weight=1.0))(
            logits, targets, crf=crf, return_components=True
        )
        total_b, b = Stage1Loss(Stage1LossConfig(use_crf=True, crf_weight=0.5))(
            logits, targets, crf=crf, return_components=True
        )
        assert torch.allclose(a["crf_nll_per_token"], b["crf_nll_per_token"], atol=1e-12)
        assert torch.allclose(total_b, b["focal_ce"] + 0.5 * b["crf_nll_per_token"], atol=1e-12)

    def test_crf_zero_weight_recovers_the_ce_only_loss(self):
        logits, targets = _fixture(b=2, length=9, n_ignored=0)
        crf = LinearChainCRF(NUM_CLASSES).double()
        total = Stage1Loss(Stage1LossConfig(use_crf=True, crf_weight=0.0))(logits, targets, crf=crf)
        assert torch.allclose(total, Stage1Loss()(logits, targets), atol=1e-12)

    def test_crf_handles_a_left_flanked_window(self):
        """End-to-end: the zero-flanked-at-start case must train, not raise."""
        logits = torch.randn(1, 6, NUM_CLASSES, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3, 0]])
        crf = LinearChainCRF(NUM_CLASSES).double()
        total = Stage1Loss(Stage1LossConfig(use_crf=True))(logits, targets, crf=crf)
        assert torch.isfinite(total)
        total.backward()
        assert torch.isfinite(logits.grad).all()

    def test_crf_gradient_reaches_the_transitions(self):
        logits, targets = _fixture(b=2, length=9, n_ignored=0)
        crf = LinearChainCRF(NUM_CLASSES).double()
        Stage1Loss(Stage1LossConfig(use_crf=True))(logits, targets, crf=crf).backward()
        assert crf.transitions.grad is not None
        assert float(crf.transitions.grad.abs().sum()) > 0.0
        assert torch.isfinite(crf.start_transitions.grad).all()
        assert torch.isfinite(crf.end_transitions.grad).all()

    def test_crf_viterbi_decode_shape(self):
        """imp.md gate: CRF Viterbi shape."""
        emissions = torch.randn(2, 6, NUM_CLASSES, dtype=torch.float64)
        tags = torch.tensor(
            [[IGNORE_INDEX, 1, 2, 3, 4, 5], [0, 1, 2, 3, IGNORE_INDEX, IGNORE_INDEX]]
        )
        crf = LinearChainCRF(NUM_CLASSES).double()
        e, _, m = left_align_for_crf(emissions, tags, tags != IGNORE_INDEX)
        paths = crf.viterbi_decode(e, mask=m)
        assert [len(p) for p in paths] == [5, 4]
        assert all(0 <= tag < NUM_CLASSES for p in paths for tag in p)

    def test_real_mask_and_ignore_index_agree_by_construction(self):
        """P2-01 emits IGNORE_INDEX exactly at ~real_mask; passing either must match."""
        logits, targets = _fixture(b=2, length=8, n_ignored=0)
        targets[0, 0] = IGNORE_INDEX
        crf = LinearChainCRF(NUM_CLASSES).double()
        loss_fn = Stage1Loss(Stage1LossConfig(use_crf=True))
        a = loss_fn(logits, targets, crf=crf)
        b = loss_fn(logits, targets, crf=crf, real_mask=(targets != IGNORE_INDEX))
        assert torch.allclose(a, b, atol=1e-12)

    def test_inconsistent_real_mask_is_rejected_not_reinterpreted(self):
        """A mask calling an ignored position 'real' would train the CRF on a fake label.

        left_align_for_crf clamps the -100 tag to class 0, so without this check the chain
        would silently learn `background` at a position carrying no DNA.
        """
        logits, targets = _fixture(b=1, length=8, n_ignored=0)
        targets[0, 0] = IGNORE_INDEX
        crf = LinearChainCRF(NUM_CLASSES).double()
        loss_fn = Stage1Loss(Stage1LossConfig(use_crf=True))
        lying_mask = torch.ones_like(targets, dtype=torch.bool)  # claims the pad is real
        with pytest.raises(ValueError, match="real_mask disagrees"):
            loss_fn(logits, targets, crf=crf, real_mask=lying_mask)

    def test_crf_config_mismatches_fail_loud(self):
        logits, targets = _fixture()
        with pytest.raises(ValueError, match="no crf module was supplied"):
            Stage1Loss(Stage1LossConfig(use_crf=True))(logits, targets)
        crf = LinearChainCRF(NUM_CLASSES).double()
        with pytest.raises(ValueError, match="config.use_crf is False"):
            Stage1Loss()(logits, targets, crf=crf)

    def test_loss_is_finite_and_positive(self):
        logits, targets = _fixture()
        value = float(Stage1Loss()(logits, targets))
        assert math.isfinite(value)
        assert value > 0.0
