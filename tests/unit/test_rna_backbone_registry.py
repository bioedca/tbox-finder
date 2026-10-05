"""The Stage-2 RNA backbone allow-list (ADR-0002 D5/D6 + A15) is closed, pinned, and honest.

Stdlib-only by design. The registry exists so an unknown backbone is refused at **compose**
time on a login node — before ``sbatch``, before a GPU, before a multi-GB download — so its
tests must run in the bare tier that has no torch, or the guard's own guard would only run
where the guard is least needed.

What is being defended, in one sentence each:

* the allow-list is the ONLY way to name a Stage-2 backbone, so a typo cannot degrade into
  "load the production default" and turn a comparator into a very expensive re-run of the
  shipped model with a green gate;
* the production entry is **single-sourced** from the module P1-13's parity gate proved
  faithful, so this table cannot drift away from the checkpoint that actually ships;
* the two entries agree on tokenisation and disagree on width — the first is what makes the
  P3-18 ECE comparison clean, the second is what makes ``train.derive_clauses``'s
  ``backbone_pinned`` cross-check able to discriminate at all.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tbox_finder.models import rna_backbone_registry as R
from tbox_finder.stage2 import tokenizer as tok
from tbox_finder.stage2 import train as T

_REPO = Path(__file__).resolve().parents[2]
_MODULE = _REPO / "src" / "tbox_finder" / "models" / "rna_backbone_registry.py"


# --------------------------------------------------------------------------------------
# The allow-list is closed
# --------------------------------------------------------------------------------------
def test_the_allow_list_holds_exactly_the_two_adr_entries():
    assert R.BACKBONE_KEYS == ("rinalmo-giga", "rnafm")
    assert R.PRODUCTION_BACKBONE == "rinalmo-giga"
    assert R.COMPARATOR_BACKBONE == "rnafm"
    assert R.BACKBONES[R.PRODUCTION_BACKBONE].role == "production"
    assert R.BACKBONES[R.COMPARATOR_BACKBONE].role == "comparator"


@pytest.mark.parametrize(
    "bad", ["rinalmo", "rna-fm", "RNAFM", "", "multimolecule/rnafm", "caduceus-ps-131k-d256-l16"]
)
def test_an_unknown_key_raises_rather_than_defaulting(bad):
    """The failure this prevents: a misspelled comparator silently training production."""
    with pytest.raises(ValueError, match="unknown Stage-2 backbone"):
        R.resolve_backbone(bad)


@pytest.mark.parametrize("bad", [None, 3, 3.0, True, ["rnafm"], {"key": "rnafm"}])
def test_a_non_string_key_raises_rather_than_being_coerced(bad):
    with pytest.raises(ValueError, match="must be a string key"):
        R.resolve_backbone(bad)


def test_the_registry_is_immutable_from_outside():
    """A MappingProxyType, so a caller cannot add an entry at runtime and skip the ADR."""
    with pytest.raises(TypeError):
        R.BACKBONES["smuggled"] = R.BACKBONES[R.PRODUCTION_BACKBONE]


# --------------------------------------------------------------------------------------
# Revisions are pinned, immutable, and per-key
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", R.BACKBONE_KEYS)
def test_every_revision_is_a_commit_sha_never_a_branch(key):
    """ADR-0002 D2: an immutable commit, never ``main``."""
    spec = R.resolve_backbone(key)
    assert len(spec.revision) == R._SHA_LEN
    assert all(c in "0123456789abcdef" for c in spec.revision), spec.revision
    assert spec.revision != "main"


@pytest.mark.parametrize("key", R.BACKBONE_KEYS)
def test_require_pinned_revision_accepts_the_pin_and_none_and_nothing_else(key):
    spec = R.resolve_backbone(key)
    # None = "the caller did not ask for anything else" -> resolves.
    assert R.require_pinned_revision(key, None) is spec
    assert R.require_pinned_revision(key, spec.revision) is spec
    for bad in ("main", "b" * 40, "", spec.revision[:-1], spec.revision.upper()):
        with pytest.raises(ValueError, match="!= the pinned"):
            R.require_pinned_revision(key, bad)


def test_one_entrys_revision_is_not_accepted_for_the_other():
    """The pin is PER KEY. A shared check would let an RNA-FM revision authorise a RiNALMo
    load — which is not a hypothetical, since both live in the same `multimolecule` org and a
    copy-paste between the two entries is the obvious way to get it wrong."""
    prod = R.resolve_backbone(R.PRODUCTION_BACKBONE)
    comp = R.resolve_backbone(R.COMPARATOR_BACKBONE)
    assert prod.revision != comp.revision
    with pytest.raises(ValueError, match="!= the pinned"):
        R.require_pinned_revision(R.PRODUCTION_BACKBONE, comp.revision)
    with pytest.raises(ValueError, match="!= the pinned"):
        R.require_pinned_revision(R.COMPARATOR_BACKBONE, prod.revision)


# --------------------------------------------------------------------------------------
# The production entry is single-sourced, not re-typed
# --------------------------------------------------------------------------------------
def test_the_production_entry_is_the_parity_confirmed_checkpoint_itself():
    """Not "equal to" by coincidence — sourced from the module A9 signed off.

    Re-typing the repo id/revision here would let the shipped pin and this table drift, and
    the drift would be invisible: both would still be 40-hex strings naming a real checkpoint.
    """
    from tbox_finder.eval import rinalmo_parity as P

    spec = R.resolve_backbone(R.PRODUCTION_BACKBONE)
    assert spec.repo_id == P.REPO_ID
    assert spec.revision == P.REVISION
    assert spec.hidden_size == P.D_MODEL
    assert spec.n_layer == P.N_LAYERS


def test_the_module_imports_no_ml_stack_so_the_refusal_runs_on_a_login_node():
    """Torch-free by construction, asserted from the source rather than from ``sys.modules``.

    A ``sys.modules`` check would pass in a process that happened not to have imported torch
    yet; the contract is that this module *cannot* need it.
    """
    tree = ast.parse(_MODULE.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"torch", "transformers", "multimolecule", "peft", "numpy", "pandas"}
    assert not (imported & forbidden), sorted(imported & forbidden)


# --------------------------------------------------------------------------------------
# What makes the two arms comparable — and what makes the gate clause able to discriminate
# --------------------------------------------------------------------------------------
def test_both_backbones_share_the_tokenizer_the_repo_pins():
    """ADR-0002 A15's comparability claim, as an invariant rather than a sentence.

    RNA-FM's vocab.txt is byte-identical to rinalmo-giga's and to the repo's pinned mirror
    (measured 2026-08-21), which is why no second vocabulary was added and why a P3-18 ECE
    difference cannot be blamed on tokenisation. If an entry ever arrives whose vocabulary
    differs, this fails — and it *should*, because the shared `stage2.tokenizer` would then be
    silently mis-encoding one of the two arms.
    """
    for key in R.BACKBONE_KEYS:
        spec = R.resolve_backbone(key)
        assert spec.vocab_size == len(tok.VOCAB), (key, spec.vocab_size, len(tok.VOCAB))
        # The mirror's context cap must be reachable on every entry, or the shared
        # MAX_NUCLEOTIDE_TOKENS ceiling would over-run one model's position table.
        assert spec.max_position_embeddings >= tok.MAX_POSITION_EMBEDDINGS, key

    # ⚠ Round 10: the loop above is a CARDINALITY check. Two 28-token vocabularies in a
    # different ORDER agree on `len` and disagree on every token id, which is exactly the
    # confound A15 claims is absent — so `vocab_size` alone cannot be the evidence for
    # "tokenisation is provably identical". The identity claim is the shared mirror itself:
    # both arms encode through `stage2.tokenizer.VOCAB`, and A15 records that RNA-FM's
    # published `vocab.txt` is byte-identical to it. Pin the digest that claim names, so a
    # silent edit to the mirror (or a second table appearing for one arm) is a red test rather
    # than a comparison quietly made across two encodings
    # ([[symmetric-count-fixture-blind-to-inversion]]).
    assert tok.VOCAB_DIGEST == (
        "ac3cff22ff7eee31923e5b921470be247314794f3d283b3bc01f26049d3902b4"
    ), (
        "the shared tokenizer mirror changed. ADR-0002 A15 rests on RNA-FM's vocab.txt being "
        "byte-identical to this table; re-measure both hub vocabularies before moving it, or "
        "the D17(c) ECE comparison is confounded by tokenisation."
    )
    # ...and there really is only ONE table — a per-backbone vocabulary would break the claim.
    assert not any(
        getattr(R.resolve_backbone(key), "vocab", None) for key in R.BACKBONE_KEYS
    ), "a backbone grew its own vocabulary; the single-mirror premise of A15 no longer holds"


def test_the_two_entries_have_DIFFERENT_widths_so_the_gate_clause_can_discriminate():
    """`train.derive_clauses`'s `backbone_pinned` cross-checks the recorded identity against
    the head width MEASURED off the live module. That check is only a check while the two
    widths differ — if a future entry shared 1280, a run that recorded one backbone while
    adapting the other would satisfy the clause. Assert the discriminator exists.
    """
    widths = {R.resolve_backbone(k).hidden_size for k in R.BACKBONE_KEYS}
    assert len(widths) == len(R.BACKBONE_KEYS), f"widths collide: {widths}"


def test_param_counts_are_distinct_and_plausible_for_their_prd_labels():
    prod = R.resolve_backbone(R.PRODUCTION_BACKBONE)
    comp = R.resolve_backbone(R.COMPARATOR_BACKBONE)
    assert prod.expected_param_count != comp.expected_param_count
    # PRD §10.2 words them "650M" and "~100M". The labels are labels; the counts are measured.
    assert 6.0e8 < prod.expected_param_count < 7.0e8, prod.expected_param_count
    assert 9.0e7 < comp.expected_param_count < 1.1e8, comp.expected_param_count
    assert prod.prd_label == "650M" and comp.prd_label == "~100M"


# --------------------------------------------------------------------------------------
# Every entry is wired end-to-end
# --------------------------------------------------------------------------------------
def test_every_backbone_has_its_own_env_lock_and_the_files_exist():
    """ADR-0002 A15: the two arms genuinely run under different locks, and a run stamps the
    one it used. A missing file would make `provenance.env_lock_sha256` hash nothing."""
    locks = {}
    for key in R.BACKBONE_KEYS:
        spec = R.resolve_backbone(key)
        assert (_REPO / spec.env_lock).is_file(), spec.env_lock
        locks[key] = spec.env_lock
        assert T.env_lock_for(key) == spec.env_lock
    assert len(set(locks.values())) == len(locks), f"two arms share one env lock: {locks}"


def test_every_backbone_has_distinct_destinations():
    """A missing entry must RAISE, never inherit production's path — that inheritance is how a
    comparator run overwrites the six shipped P3-06 arms
    ([[two-outputs-one-path-destroys-the-first]])."""
    ckpts, reports = {}, {}
    for key in R.BACKBONE_KEYS:
        ckpts[key] = T.default_checkpoint_dir(key)
        reports[key] = T.default_report_path(key)
    assert len(set(ckpts.values())) == len(ckpts), ckpts
    assert len(set(reports.values())) == len(reports), reports
    with pytest.raises(ValueError, match="unknown Stage-2 backbone"):
        T.default_checkpoint_dir("not-a-backbone")
    with pytest.raises(ValueError, match="unknown Stage-2 backbone"):
        T.default_report_path("not-a-backbone")


def test_backbone_for_repo_id_round_trips_and_refuses_the_unknown():
    for key in R.BACKBONE_KEYS:
        spec = R.resolve_backbone(key)
        assert R.backbone_for_repo_id(spec.repo_id) is spec
    assert R.backbone_for_repo_id("multimolecule/rinalmo") is None
    assert R.backbone_for_repo_id("") is None


def test_backbone_summary_is_json_safe_and_carries_every_pinned_field():
    for key in R.BACKBONE_KEYS:
        spec = R.resolve_backbone(key)
        summary = R.backbone_summary(spec)
        json.dumps(summary)  # must not raise
        for field in (
            "key",
            "repo_id",
            "revision",
            "model_class",
            "hidden_size",
            "n_layer",
            "vocab_size",
            "expected_param_count",
            "env_lock",
            "role",
        ):
            assert field in summary, (key, field)
        assert summary["key"] == key
        assert isinstance(summary["hidden_size"], int)
        assert isinstance(summary["expected_param_count"], int)


def test_the_model_class_names_are_the_multimolecule_ones_the_loader_resolves():
    """Carried as strings so the module stays torch-free; the names still have to be right."""
    assert R.resolve_backbone(R.PRODUCTION_BACKBONE).model_class == "RiNALMoModel"
    assert R.resolve_backbone(R.COMPARATOR_BACKBONE).model_class == "RnaFmModel"
