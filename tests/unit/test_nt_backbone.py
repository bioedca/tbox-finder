"""Unit tests for the P1-10 NT-multispecies backbone fallback loader.

Two tiers (mirrors ``tests/unit/test_caduceus_backbone.py``):

* **Bare stdlib tier** — runs in CI with no torch: the pinned-identity constants, the
  pure ``random_dna`` helper, the revision guard, the schema validator (each mutator
  proven to *bite*), the committed-provenance artifact, the conf↔code drift guard, and the
  not-wired-as-default lock. ``nt_backbone`` imports torch-free (heavy imports are lazy).
* **Load tier** — loads the checkpoint + runs the interface-parity forward. Skips when
  torch/transformers/network are unavailable; ``TBOX_REQUIRE_NT=1`` turns a skip into a
  FAIL (so the drop-in cannot silently rot). NT is a stock ESM forward — **CPU is enough**
  (no GPU/CUDA kernel, unlike Caduceus/Mamba).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tbox_finder.labels import CLASS_ORDER
from tbox_finder.models import nt_backbone as nt

_REPO = Path(__file__).resolve().parents[2]
_PROVENANCE = _REPO / "data" / "interim" / "nt_multispecies_v2_250m" / "provenance.json"
_CONF = _REPO / "conf" / "model" / "nt_multispecies.yaml"
_TRAIN_CONF_DIR = _REPO / "conf" / "train"


# ========================================================================== #
# Pinned constants
# ========================================================================== #
def test_revision_is_immutable_commit_not_branch():
    assert nt._is_commit_sha(nt.REVISION), "REVISION must be a 40-hex commit SHA, never 'main'"


def test_hidden_dim_and_classes_frozen():
    # HIDDEN_DIM must be even (the concat-identity segmenter path splits it into halves) and
    # the class count single-sourced from the ADR-0004 label vocab (never redeclared).
    assert nt.HIDDEN_DIM % 2 == 0
    assert nt.NUM_CLASSES == len(CLASS_ORDER) == 8
    assert nt.KMER == 6  # the load-bearing tokenisation caveat (NT is not single-nt)


def test_loaders_reject_non_pinned_revision_before_import():
    # trust_remote_code=True executes remote code at the revision, so a non-pinned revision
    # must be rejected BEFORE any transformers import (arbitrary remote-code execution guard).
    for loader in (nt.load_nt_multispecies, nt.load_nt_tokenizer):
        with pytest.raises(ValueError, match="code-pinned REVISION"):
            loader(revision="main")
        with pytest.raises(ValueError, match="code-pinned REVISION"):
            loader(revision="0" * 40)


def test_pretraining_fact_carries_two_sources_and_no_archaea_overclaim():
    # The high-stakes domain claim needs >=2 agreeing sources (§10.1), and archaea must NOT
    # be asserted true (unresolved; §10.3 honesty).
    assert len(nt.PRETRAINING_CITATIONS) >= 2
    assert nt.PROKARYOTE_INCLUSIVE is True
    assert nt.ARCHAEA_INCLUDED is not True  # None (unknown) is allowed; True is an over-claim
    assert nt.MODEL_CARD_STATES_PROKARYOTES is False  # the cards are silent — record it


# ========================================================================== #
# Pure random_dna helper
# ========================================================================== #
def test_random_dna_deterministic_and_acgt():
    a = nt.random_dna(300, seed=42)
    b = nt.random_dna(300, seed=42)
    assert a == b and len(a) == 300
    assert set(a) <= set("ACGT")
    assert nt.random_dna(300, seed=43) != a  # a different seed gives a different window


def test_random_dna_rejects_nonpositive():
    with pytest.raises(ValueError):
        nt.random_dna(0, seed=42)


# ========================================================================== #
# Schema validator — a clean report is valid; each mutator bites
# ========================================================================== #
def good_report() -> dict:
    """A measured, self-consistent P1-10 report (validate_report -> [])."""
    return {
        "schema_version": nt.SCHEMA_VERSION,
        "step": "P1-10",
        "measured": True,
        "checkpoint": {
            "repo_id": nt.REPO_ID,
            "revision": nt.REVISION,
            "hidden_dim": nt.HIDDEN_DIM,
            "n_layer": nt.N_LAYER,
            "model_type": nt.MODEL_TYPE,
            "trust_remote_code": True,
        },
        "pretraining": {
            "prokaryote_inclusive": True,
            "archaea_included": None,
            "model_card_states_prokaryotes": False,
            "citations": [{"source": "a"}, {"source": "b"}],
        },
        "tokenization": {"kmer": nt.KMER},
        "env": {"torch": "x"},
        "interface_parity": {
            "hidden_shape": [1, 52, nt.HIDDEN_DIM],
            "seg_head_logits_shape": [1, 52, nt.NUM_CLASSES],
            "seg_head_logits_finite": True,
            "segmenter_logits_shape": [1, 52, nt.NUM_CLASSES],
            "segmenter_logits_finite": True,
            "num_classes": nt.NUM_CLASSES,
        },
        "gate": {
            "load_ok": True,
            "hidden_dim_ok": True,
            "seg_head_dropin_ok": True,
            "segmenter_dropin_ok": True,
            "overall_pass": True,
        },
    }


def test_good_report_is_valid():
    assert nt.validate_report(good_report()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        # structural
        lambda r: r.pop("gate"),
        lambda r: r.pop("interface_parity"),
        lambda r: r.pop("pretraining"),
        lambda r: r.update(checkpoint="not-a-mapping"),
        lambda r: r.update(schema_version="999"),
        lambda r: r.update(step="P1-09"),
        # checkpoint identity
        lambda r: r["checkpoint"].update(repo_id="someone-else/model"),
        lambda r: r["checkpoint"].update(revision="main"),
        lambda r: r["checkpoint"].update(revision="a" * 40),  # 40-hex but wrong pin
        lambda r: r["checkpoint"].update(hidden_dim=512),
        lambda r: r["checkpoint"].update(n_layer=12),
        lambda r: r["checkpoint"].update(model_type="bert"),
        lambda r: r["checkpoint"].update(trust_remote_code=False),
        # pretraining honesty (§10.1/§10.3)
        lambda r: r["pretraining"].update(prokaryote_inclusive=False),
        lambda r: r["pretraining"].update(archaea_included=True),  # over-claim
        lambda r: r["pretraining"].update(model_card_states_prokaryotes=True),  # cards are silent
        lambda r: r["pretraining"].update(citations=[{"source": "only-one"}]),  # < 2 sources
        # tokenisation
        lambda r: r["tokenization"].update(kmer=1),  # NT is 6-mer, not single-nt
        # measured consistency — the gate cannot claim a pass the shapes contradict
        lambda r: r["interface_parity"].update(hidden_shape=[1, 52, 512]),  # != HIDDEN_DIM
        lambda r: r["interface_parity"].update(num_classes=7),
        lambda r: r["interface_parity"].update(seg_head_logits_shape=[1, 52, 5]),  # != 8
        lambda r: r["interface_parity"].update(seg_head_logits_finite=False),  # non-finite logits
        lambda r: r["interface_parity"].update(segmenter_logits_shape=[1, 52, 3]),  # != 8
        lambda r: r["interface_parity"].update(
            seg_head_logits_shape=[1, 40, 8]
        ),  # token axis disagrees
        lambda r: r["gate"].update(overall_pass=False),  # AND is True -> inconsistent
        lambda r: r["gate"].update(load_ok="yes"),  # not a real bool
        lambda r: (r["gate"].update(seg_head_dropin_ok=False), r["gate"].update(overall_pass=True)),
        lambda r: r.update(
            measured=False
        ),  # unmeasured but gate still claims a pass -> fail-closed
    ],
)
def test_validator_bites(mutate):
    bad = good_report()
    mutate(bad)
    assert len(nt.validate_report(bad)) >= 1


def test_honest_dropin_failure_is_schema_valid():
    # An honestly-recorded failure (non-finite seg-head logits -> the drop-in did NOT pass)
    # must be schema-VALID — the validator flags inconsistency, never an honest failure (§10.3).
    r = good_report()
    r["interface_parity"]["seg_head_logits_finite"] = False
    r["gate"]["seg_head_dropin_ok"] = False
    r["gate"]["overall_pass"] = False
    assert nt.validate_report(r) == []


def test_unmeasured_report_without_pass_is_valid():
    # An unmeasured report is fine as long as it claims NO gate pass (a pass may only be
    # certified by a real measurement; §10.3). All gate flags False -> valid.
    r = good_report()
    r["measured"] = False
    for k in (
        "load_ok",
        "hidden_dim_ok",
        "seg_head_dropin_ok",
        "segmenter_dropin_ok",
        "overall_pass",
    ):
        r["gate"][k] = False
    assert nt.validate_report(r) == []


# ========================================================================== #
# Committed provenance artifact (a required, git-tracked P1-10 output)
# ========================================================================== #
def test_committed_provenance_valid_and_gate_passes():
    assert _PROVENANCE.exists(), "committed provenance.json is required (P1-10 artifact)"
    prov = json.loads(_PROVENANCE.read_text())
    assert prov["env_lock_hash"], "env_lock_hash must be recorded (ml-dna lock)"
    assert "extra" in prov
    report = prov["extra"]
    assert nt.validate_report(report) == []
    assert report["measured"] is True
    # The interface-parity drop-in actually passed on real hardware.
    assert report["gate"]["overall_pass"] is True
    assert report["interface_parity"]["hidden_shape"][-1] == nt.HIDDEN_DIM
    assert report["interface_parity"]["seg_head_logits_shape"][-1] == nt.NUM_CLASSES
    assert report["interface_parity"]["segmenter_logits_shape"][-1] == nt.NUM_CLASSES
    # Revision is the immutable pin; the domain claim is recorded + not over-claimed.
    assert report["checkpoint"]["revision"] == nt.REVISION
    assert report["pretraining"]["prokaryote_inclusive"] is True
    assert report["pretraining"]["archaea_included"] is not True
    assert len(report["pretraining"]["citations"]) >= 2
    # The 6-mer collapse is quantified (n_tokens < n_nucleotides).
    assert report["interface_parity"]["n_tokens"] < report["interface_parity"]["n_nucleotides"]


# ========================================================================== #
# conf/model drift guard + not-wired-as-default lock
# ========================================================================== #
def _scan_yaml_scalar(path: Path, key: str) -> str | None:
    """Return the scalar value of a top-level ``key:`` (stdlib; strips inline comments)."""
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            value = stripped.split(":", 1)[1]
            return value.split("#", 1)[0].strip() or None
    return None


def test_conf_model_matches_code_pins():
    assert _CONF.exists(), "conf/model/nt_multispecies.yaml is required (P1-10 artifact)"
    assert _scan_yaml_scalar(_CONF, "repo_id") == nt.REPO_ID
    assert _scan_yaml_scalar(_CONF, "revision") == nt.REVISION
    assert _scan_yaml_scalar(_CONF, "hidden_dim") == str(nt.HIDDEN_DIM)
    assert _scan_yaml_scalar(_CONF, "n_layer") == str(nt.N_LAYER)
    assert _scan_yaml_scalar(_CONF, "model_type") == nt.MODEL_TYPE
    assert _scan_yaml_scalar(_CONF, "kmer") == str(nt.KMER)
    assert _scan_yaml_scalar(_CONF, "max_tokens") == str(nt.MAX_TOKENS)
    assert _scan_yaml_scalar(_CONF, "trust_remote_code") == "true"


def test_nt_not_wired_as_default():
    # The fallback must NOT be selected by any training config (ADR-0002 A6: ladder not
    # triggered; the validation gate requires "not wired as default").
    train_configs = list(_TRAIN_CONF_DIR.glob("*.yaml")) if _TRAIN_CONF_DIR.exists() else []
    for cfg in train_configs:
        for line in cfg.read_text().splitlines():
            no_comment = line.split("#", 1)[0]
            assert (
                "nt_multispecies" not in no_comment
            ), f"{cfg.name} selects nt_multispecies, but it must not be wired as a default"


# ========================================================================== #
# Load tier — load + interface-parity forward (skips without deps; CPU is enough)
# ========================================================================== #
def _fail_or_skip(reason: str) -> None:
    """In ``TBOX_REQUIRE_NT=1`` an unusable NT backbone FAILS; otherwise it skips."""
    if os.environ.get("TBOX_REQUIRE_NT") == "1":
        pytest.fail(f"TBOX_REQUIRE_NT=1 but the NT-multispecies backbone is unusable: {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def loaded():
    try:
        import torch  # noqa: F401
    except ImportError:
        _fail_or_skip("torch not installed (activate tbox-ml-dna)")
    try:
        model = nt.load_nt_multispecies(device="cpu")
        tokenizer = nt.load_nt_tokenizer()
    except Exception as exc:  # noqa: BLE001 - network / transformers / remote-code failures
        _fail_or_skip(f"could not load checkpoint at pinned revision: {type(exc).__name__}: {exc}")
    return model, tokenizer


def test_load_forward_emits_hidden_dim(loaded):
    model, tokenizer = loaded
    dna = nt.random_dna(nt.DEFAULT_SEQ_LEN_NT, seed=42)
    ip = nt.interface_parity(model, tokenizer, dna=dna, seed=42, device="cpu")
    assert ip["hidden_shape"][-1] == nt.HIDDEN_DIM
    assert ip["hidden_shape"][0] == 1


def test_interface_parity_seg_head_and_segmenter_dropin(loaded):
    model, tokenizer = loaded
    dna = nt.random_dna(nt.DEFAULT_SEQ_LEN_NT, seed=42)
    ip = nt.interface_parity(model, tokenizer, dna=dna, seed=42, device="cpu")
    # Both drop-in paths emit per-position 8-class logits over the SAME token axis, all finite.
    assert ip["seg_head_logits_shape"][-1] == nt.NUM_CLASSES
    assert ip["segmenter_logits_shape"][-1] == nt.NUM_CLASSES
    assert (
        ip["hidden_shape"][1] == ip["seg_head_logits_shape"][1] == ip["segmenter_logits_shape"][1]
    )
    assert ip["seg_head_logits_finite"] is True
    assert ip["segmenter_logits_finite"] is True
    assert ip["segmenter_rc_combine"] == "concat"  # identity pass-through (NT is not RC)


def test_kmer_collapse_quantified(loaded):
    model, tokenizer = loaded
    dna = nt.random_dna(nt.DEFAULT_SEQ_LEN_NT, seed=42)
    ip = nt.interface_parity(model, tokenizer, dna=dna, seed=42, device="cpu")
    # The 6-mer tokeniser collapses ~6 nt/token -> far fewer tokens than nucleotides (the
    # documented per-nt label-alignment caveat). ~300 nt -> ~50 6-mer tokens (+ specials).
    assert ip["n_tokens"] < ip["n_nucleotides"]
    assert 4.0 < ip["kmer_collapse_ratio"] <= 6.0
