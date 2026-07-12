"""P1-02 — tests for the Caduceus-PS backbone loader + RC-equivariance provenance.

Two tiers (mirroring ``tests/unit/test_kernel_smoke_report.py`` + ``tests/ml/test_no_leakage.py``):

- **Bare stdlib tier** (runs in CI, no torch): the pure RC operator + report/schema
  validators are each proven to *bite* (a clean case passes, a mutated one fails), the
  pinned constants are checked, the ``conf/model`` spec is drift-guarded against the code,
  and the **committed** ``provenance.json`` (the real GPU-measured artifact) is validated
  and its gate asserted to pass.
- **GPU tier** (skips in CI / on a CPU-only box; runs under ``tbox-ml-dna`` for the §8.5
  manual gate): actually loads the checkpoint at the pinned revision, asserts the exact
  param count, and verifies RC-equivariance end-to-end (with the reverse-only negative
  control proving the check discriminates). ``TBOX_REQUIRE_CADUCEUS=1`` makes an unloadable
  backbone FAIL rather than skip, so the gate cannot silently rot (§10.3).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tbox_finder.models import caduceus_backbone as cb

_REPO = Path(__file__).resolve().parents[2]
_PROVENANCE = _REPO / "data" / "interim" / "caduceus_ps_131k" / "provenance.json"
_CONF = _REPO / "conf" / "model" / "caduceus_stage1.yaml"

# The Caduceus config complement_map (config.json): A(7)↔T(10), C(8)↔G(9), rest self-map.
_COMPLEMENT_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 10, 8: 9, 9: 8, 10: 7,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
}  # fmt: skip


# ========================================================================== #
# Pinned constants — frozen checkpoint identity (§10.3: no branch pins).
# ========================================================================== #
def test_revision_is_immutable_commit_not_branch():
    assert cb._is_commit_sha(cb.REVISION), "REVISION must be a 40-hex commit SHA, never 'main'"
    assert cb.REVISION != "main"


def test_expected_param_count_frozen():
    # The "7.73 M" of PRD §10.1 to the exact measured integer, frozen in code.
    assert cb.EXPECTED_PARAM_COUNT == 7_725_312


def test_pretraining_fact_carries_two_sources():
    assert cb.PRETRAINING_DOMAIN == "human reference genome"
    # Load-bearing fact (ships in the model card) → CLAUDE.md §10.1 needs ≥2 agreeing sources.
    assert len(cb.PRETRAINING_CITATIONS) >= 2
    joined = json.dumps(cb.PRETRAINING_CITATIONS)
    assert "2403.03234" in joined and "40567809" in joined and "huggingface.co" in joined


# ========================================================================== #
# Pure RC operator — stdlib, proven to bite.
# ========================================================================== #
def test_reverse_complement_ids_known_case():
    # A=7, C=8, G=9, T=10 → "ACGT" ids [7,8,9,10]; RC of ACGT is ACGT (palindrome under RC),
    # but the ORDER reverses: complement→[10,9,8,7], reverse→[7,8,9,10]. So RC(ACGT)==ACGT.
    assert cb.reverse_complement_ids([7, 8, 9, 10], _COMPLEMENT_MAP) == [7, 8, 9, 10]
    # AAAA (7,7,7,7) → complement TTTT (10,10,10,10), reversed is TTTT.
    assert cb.reverse_complement_ids([7, 7, 7, 7], _COMPLEMENT_MAP) == [10, 10, 10, 10]


def test_reverse_complement_ids_is_involution():
    ids = [7, 8, 8, 10, 9, 7, 10, 9]
    once = cb.reverse_complement_ids(ids, _COMPLEMENT_MAP)
    twice = cb.reverse_complement_ids(once, _COMPLEMENT_MAP)
    assert twice == ids  # RC∘RC == identity
    assert once != ids  # and a single RC actually changed it (the operator does something)


def test_reverse_complement_ids_bites_on_reverse_only():
    # A mutant that forgets to complement (pure reverse) must NOT equal the true RC.
    ids = [7, 8, 9, 8, 10]
    true_rc = cb.reverse_complement_ids(ids, _COMPLEMENT_MAP)
    reverse_only = list(reversed(ids))
    assert true_rc != reverse_only


# ========================================================================== #
# Report / schema validator — proven to bite (kernel_smoke idiom).
# ========================================================================== #
def good_report() -> dict:
    """A clean, self-consistent measured P1-02 report for guard-bites mutation."""
    return {
        "schema_version": "1",
        "step": "P1-02",
        "measured": True,
        "generated_by": "src/tbox_finder/models/caduceus_backbone.py",
        "checkpoint": {"repo_id": cb.REPO_ID, "revision": cb.REVISION, "hub_url": cb.HUB_URL},
        "pretraining": {"domain": cb.PRETRAINING_DOMAIN, "citations": cb.PRETRAINING_CITATIONS},
        "env": {"cuda_available": True, "is_sm86": True},
        "param_count": {
            "expected": cb.EXPECTED_PARAM_COUNT,
            "actual": cb.EXPECTED_PARAM_COUNT,
            "param_count_ok": True,
        },
        "rc_equivariance": {
            "atol": 1e-4,
            "max_abs_diff": 0.0,
            "neg_control_max_abs_diff": 2.2,
            "pass": True,
        },
        "gate": {
            "load_ok": True,
            "param_count_ok": True,
            "rc_equivariance_ok": True,
            "overall_pass": True,
        },
    }


def test_good_report_is_valid():
    assert cb.validate_report(good_report()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("gate"),
        lambda r: r["checkpoint"].__setitem__("revision", "main"),
        lambda r: r["checkpoint"].__setitem__("revision", "abc123"),  # not 40-hex
        lambda r: r["checkpoint"].__setitem__("repo_id", "someone/else"),
        lambda r: r["pretraining"].__setitem__("domain", "mouse genome"),
        lambda r: r["pretraining"].__setitem__("citations", []),
        lambda r: r["param_count"].__setitem__("expected", 123),
        lambda r: r["param_count"].__setitem__("actual", 999),  # actual != expected
        lambda r: r["rc_equivariance"].__setitem__("pass", False),  # inconsistent w/ diff<=atol
        lambda r: r["rc_equivariance"].__setitem__("max_abs_diff", 1.0),  # >atol but pass True
        # gate.rc_equivariance_ok=True while measured pass=False (consistent w/ diff>atol):
        lambda r: (
            r["rc_equivariance"].update({"max_abs_diff": 1.0, "pass": False}),
            r["gate"].__setitem__("rc_equivariance_ok", True),
        ),
        lambda r: r["gate"].__setitem__("param_count_ok", False),  # != pc.param_count_ok (True)
        lambda r: r["gate"].__setitem__("overall_pass", False),  # contradicts AND(subgates)
        lambda r: r["rc_equivariance"].pop("neg_control_max_abs_diff"),
    ],
)
def test_validator_bites(mutate):
    bad = good_report()
    mutate(bad)
    assert len(cb.validate_report(bad)) >= 1


def test_honest_param_count_failure_is_schema_valid():
    # A report that HONESTLY records a param-count mismatch (actual != expected,
    # param_count_ok=False, gate fails) is a valid failure report, NOT a schema error —
    # validate_report must not reject it (the CodeRabbit-caught consistency-vs-success bug).
    r = good_report()
    r["param_count"].update({"actual": 999, "param_count_ok": False})
    r["gate"].update({"param_count_ok": False, "overall_pass": False})
    assert cb.validate_report(r) == []


# ========================================================================== #
# conf/model spec drift guard — the declarative pin can't diverge from code.
# ========================================================================== #
def _scan_yaml_scalar(path: Path, key: str) -> str | None:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            val = stripped.split(":", 1)[1].strip()
            return val.split("#", 1)[0].strip()  # drop trailing comment
    return None


def test_conf_model_matches_code_pins():
    assert _CONF.exists(), "conf/model/caduceus_stage1.yaml missing"
    assert _scan_yaml_scalar(_CONF, "revision") == cb.REVISION
    assert _scan_yaml_scalar(_CONF, "repo_id") == cb.REPO_ID
    assert _scan_yaml_scalar(_CONF, "n_layer") == str(cb.N_LAYER)
    assert _scan_yaml_scalar(_CONF, "d_model") == str(cb.D_MODEL)


# ========================================================================== #
# Committed provenance.json — the real GPU-measured artifact (validated in CI).
# ========================================================================== #
def test_committed_provenance_valid_and_gate_passes():
    # The provenance.json is a required, git-tracked artifact of this step (not LFS/DVC),
    # so it is always present in a normal checkout — its absence is a regression, FAIL
    # (never skip), so the provenance gate cannot silently rot (§8.7/§10.3).
    assert _PROVENANCE.exists(), "committed provenance.json is required (P1-02 artifact)"
    prov = json.loads(_PROVENANCE.read_text())
    # Canonical sidecar fields (CLAUDE.md §11).
    assert prov["env_lock_hash"], "env_lock_hash must be recorded (ml-dna lock)"
    assert "extra" in prov
    report = prov["extra"]
    assert cb.validate_report(report) == []
    assert report["measured"] is True
    # The two P1-02 assertions actually passed on real hardware.
    assert report["param_count"]["actual"] == cb.EXPECTED_PARAM_COUNT
    assert report["rc_equivariance"]["max_abs_diff"] <= report["rc_equivariance"]["atol"]
    assert report["rc_equivariance"]["neg_control_max_abs_diff"] > report["rc_equivariance"]["atol"]
    assert report["gate"]["overall_pass"] is True
    # Revision is the immutable pin (never a branch).
    assert cb._is_commit_sha(report["checkpoint"]["revision"])
    assert report["checkpoint"]["revision"] == cb.REVISION


# ========================================================================== #
# GPU tier — load + measure end-to-end (skips without a GPU; §8.5 manual gate).
# ========================================================================== #
def _fail_or_skip(reason: str) -> None:
    """In ``TBOX_REQUIRE_CADUCEUS=1`` an unloadable backbone FAILS; otherwise it skips."""
    if os.environ.get("TBOX_REQUIRE_CADUCEUS") == "1":
        pytest.fail(f"TBOX_REQUIRE_CADUCEUS=1 but the Caduceus backbone is unusable: {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def loaded():
    try:
        import torch
    except ImportError:
        _fail_or_skip("torch not installed (activate tbox-ml-dna)")
    if not torch.cuda.is_available():
        _fail_or_skip("no CUDA device (the packaged Mamba forward requires a GPU, ADR-0002 A2 C2)")
    try:
        model = cb.load_caduceus_ps()
        tokenizer = cb.load_tokenizer()
    except Exception as exc:  # noqa: BLE001 - network / transformers / kernel failures
        _fail_or_skip(f"could not load checkpoint at pinned revision: {type(exc).__name__}: {exc}")
    return model, tokenizer


def test_gpu_param_count_matches_checkpoint(loaded):
    model, _ = loaded
    assert cb.count_parameters(model) == cb.EXPECTED_PARAM_COUNT


def test_gpu_rc_equivariance_holds_and_control_bites(loaded):
    model, tokenizer = loaded
    rc = cb.rc_equivariance(model, tokenizer, seq_len=128, seed=42, device="cuda")
    # f(RC(x)) == flip(f(x), (-2,-1)) within atol …
    assert rc["max_abs_diff"] <= cb.RC_EQUIVARIANCE_ATOL
    # … while the reverse-only control is far off — the check genuinely discriminates.
    assert rc["neg_control_max_abs_diff"] > 100 * cb.RC_EQUIVARIANCE_ATOL
