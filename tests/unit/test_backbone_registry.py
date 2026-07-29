"""P2-12 / ADR-0002 A14 — the closed Caduceus-PS checkpoint allow-list.

Bare-CI tier: :mod:`tbox_finder.models.backbone_registry` is torch-free on purpose, so the
pins that make the size ablation *reachable and honest* are locked without the GPU env — the
same posture :mod:`tbox_finder.models.rc_combine` takes for the RC-combination contract.

What is actually at stake. Before P2-12 the identity lived as two module constants and
``load_caduceus_ps`` took no repo argument, so the pre-registered PRD §10.1 size ablation could
not be run at all — and the *way* it could not be run was silent: a ``model.repo_id=`` override
composed, was dropped, and the report echoed the constants. These tests lock the replacement's
three load-bearing properties: the list is **closed**, every entry is **pinned to an immutable
SHA**, and each entry's ``expected_param_count`` is a **distinct measured integer** a run must
match at load.
"""

from __future__ import annotations

import re

import pytest

from tbox_finder.models import backbone_registry as R

_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_the_allow_list_is_not_empty_and_holds_the_three_prd_checkpoints():
    """Emptiness guard first: every other assertion here is vacuous over an empty table.

    PRD §10.1 names three points — the 7.73M family maximum and the two downward-search
    checkpoints — so the count is asserted, not merely "at least one".
    """
    assert len(R.CHECKPOINTS) == 3, sorted(R.CHECKPOINTS)
    assert tuple(R.CHECKPOINTS) == R.CHECKPOINT_KEYS
    assert R.PRODUCTION_CHECKPOINT in R.CHECKPOINTS
    roles = sorted(s.role for s in R.CHECKPOINTS.values())
    assert roles == ["ablation", "ablation", "production"], roles


@pytest.mark.parametrize("key", sorted(R.CHECKPOINTS))
def test_every_entry_is_pinned_to_an_immutable_sha(key: str):
    """``main`` is not a pin. ``trust_remote_code=True`` EXECUTES the modeling code at the
    revision, so a moving ref is both irreproducible weights and moving remote code."""
    spec = R.CHECKPOINTS[key]
    assert spec.key == key, "the mapping key and the entry's own key must agree"
    assert _SHA.match(spec.revision), f"{key}: revision {spec.revision!r} is not a commit SHA"
    assert spec.repo_id.startswith("kuleshov-group/caduceus-ps_"), spec.repo_id


@pytest.mark.parametrize("key", sorted(R.CHECKPOINTS))
def test_every_entry_carries_a_positive_integer_expectation(key: str):
    spec = R.CHECKPOINTS[key]
    for field in ("d_model", "n_layer", "pretrain_seq_len", "expected_param_count"):
        value = getattr(spec, field)
        assert isinstance(value, int) and not isinstance(value, bool), f"{key}.{field}={value!r}"
        assert value > 0, f"{key}.{field}={value}"


def test_the_three_entries_are_mutually_distinct():
    """Two entries sharing a repo id, a revision or a parameter count would make the axis
    unmeasurable — and ``checkpoint_for_repo_id`` ambiguous."""
    for field in ("repo_id", "revision", "expected_param_count"):
        values = [getattr(s, field) for s in R.CHECKPOINTS.values()]
        assert len(set(values)) == len(values), f"duplicate {field}: {values}"


def test_the_production_pin_is_the_one_the_backbone_module_publishes():
    """Single source of truth. ``caduceus_backbone`` derives REPO_ID / REVISION /
    EXPECTED_PARAM_COUNT / D_MODEL / N_LAYER from this table rather than restating them; a
    second literal is a second thing to forget to update."""
    from tbox_finder.models import caduceus_backbone as C

    spec = R.CHECKPOINTS[R.PRODUCTION_CHECKPOINT]
    assert spec.repo_id == C.REPO_ID
    assert spec.revision == C.REVISION
    assert spec.expected_param_count == C.EXPECTED_PARAM_COUNT == 7_725_312
    assert spec.d_model == C.D_MODEL == 256
    assert spec.n_layer == C.N_LAYER == 16


def test_the_size_axis_is_a_joint_size_by_context_axis():
    """ADR-0002 A14's second PRD wording correction, given a machine-readable referent.

    Both smaller checkpoints are ``seqlen-1k`` pretrained while production is ``seqlen-131k``,
    so the axis moves parameter count AND pretraining context together. If a future entry made
    them equal, the ablation table's size x context disclosure would be describing something
    that is no longer true — so the fact is asserted, not just written in prose.
    """
    production = R.CHECKPOINTS[R.PRODUCTION_CHECKPOINT]
    ablations = [s for s in R.CHECKPOINTS.values() if s.role == "ablation"]
    assert production.pretrain_seq_len == 131_072
    assert [s.pretrain_seq_len for s in ablations] == [1_024, 1_024]
    assert all(s.expected_param_count < production.expected_param_count for s in ablations)
    # The PRD §10.1 downward search: strictly decreasing, and the ~471k entry is the floor.
    counts = sorted((s.expected_param_count for s in R.CHECKPOINTS.values()), reverse=True)
    assert counts == [7_725_312, 1_934_592, 470_702], counts


def test_the_repo_ids_carry_the_lr_suffix_the_prd_omits():
    """ADR-0002 A14's first PRD wording correction. The PRD §10.1 ids are incomplete — the two
    ablation repos end ``_lr-8e-3`` — and a run built from the PRD's spelling would 404."""
    ablations = [s for s in R.CHECKPOINTS.values() if s.role == "ablation"]
    assert all(s.repo_id.endswith("_lr-8e-3") for s in ablations), [s.repo_id for s in ablations]


def test_resolve_checkpoint_is_closed():
    """An unknown key must RAISE, never fall back to the default. A selector that degrades to
    "load production" is how an ablation trains the pinned model and still passes its gate."""
    for key in sorted(R.CHECKPOINTS):
        assert R.resolve_checkpoint(key).key == key
    for bad in (
        "caduceus-ps-1k-d118-l5",  # a plausible typo
        "kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4",  # the repo id, not a key
        "",
        "production",  # a role, not a key
    ):
        with pytest.raises(ValueError, match="unknown backbone"):
            R.resolve_checkpoint(bad)
    for bad_type in (None, 7, 1.5, ["caduceus-ps-1k-d118-l4"]):
        with pytest.raises(ValueError, match="must be a string key"):
            R.resolve_checkpoint(bad_type)


def test_require_pinned_revision_generalises_the_pin_without_relaxing_it():
    """Per-checkpoint, and never "any revision of a known repo"."""
    prod = R.CHECKPOINTS[R.PRODUCTION_CHECKPOINT]
    small = R.CHECKPOINTS["caduceus-ps-1k-d118-l4"]
    assert R.require_pinned_revision(prod.key, prod.revision) is prod
    with pytest.raises(ValueError, match="code-pinned revision"):
        R.require_pinned_revision(prod.key, "main")
    # The cross-product must fail: a valid SHA for ANOTHER allow-listed checkpoint is still
    # not this one's, and accepting it would load different weights under a pinned name.
    with pytest.raises(ValueError, match="code-pinned revision"):
        R.require_pinned_revision(prod.key, small.revision)
    with pytest.raises(ValueError, match="code-pinned revision"):
        R.require_pinned_revision(small.key, prod.revision)


def test_checkpoint_for_repo_id_round_trips_and_refuses_a_stranger():
    """The pre-P2-12 re-derivation path: the 36 committed sweep reports record only a repo id,
    so the ablation table resolves the baseline's backbone axis through this."""
    for spec in R.CHECKPOINTS.values():
        assert R.checkpoint_for_repo_id(spec.repo_id) is spec
    assert R.checkpoint_for_repo_id("kuleshov-group/caduceus-ph_seqlen-131k") is None
    assert R.checkpoint_for_repo_id("") is None


def test_checkpoint_summary_is_json_safe_and_complete():
    spec = R.resolve_checkpoint("caduceus-ps-1k-d256-l4")
    summary = R.checkpoint_summary(spec)
    assert summary["key"] == spec.key
    assert summary["repo_id"] == spec.repo_id
    assert summary["revision"] == spec.revision
    assert summary["expected_param_count"] == spec.expected_param_count == 1_934_592
    assert summary["d_model"] == 256 and summary["n_layer"] == 4
    assert summary["pretrain_seq_len"] == 1_024
    assert all(isinstance(v, (str, int)) for v in summary.values()), summary


def test_the_allow_list_is_immutable_at_runtime():
    """A config cannot widen the list, and neither can a stray assignment. ADR-0002 D8: pins
    are added by amendment, never mutated in flight."""
    with pytest.raises(TypeError):
        R.CHECKPOINTS["rogue"] = R.CHECKPOINTS[R.PRODUCTION_CHECKPOINT]
    spec = R.CHECKPOINTS[R.PRODUCTION_CHECKPOINT]
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        spec.revision = "main"
