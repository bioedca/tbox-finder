"""P0-06: unit tests for the sm_86 wheel-availability classifier.

Exercises the network-free core of ``scripts/check_sm86_wheels.py`` — the wheel-filename
parser (``_WHEEL_RE`` / ``_LOCAL_RE``) and ``classify_assets`` — against fixture asset
lists, so the logic that produces the ADR-0002 verdict is tested without a live network
(the live probe itself is a one-shot metadata query, not a CI job).

The final test locks the **observed P0-06 finding** (mamba-ssm 2.2.6.post3 ships a prebuilt
``cu12torch2.7`` cp312 wheel → Option B needs no source build) as a regression fixture, so
a future refactor of the classifier can't silently re-break the matrix that the amendment
records. Stdlib + pytest only.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "check_sm86_wheels.py"

_spec = importlib.util.spec_from_file_location("check_sm86_wheels", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve the module via sys.modules (py3.12/3.13).
sys.modules["check_sm86_wheels"] = mod
_spec.loader.exec_module(mod)


def _classify(version, assets, *, sdist=True, exists=True, tag="v0"):
    return mod.classify_assets(
        "pkg",
        version,
        "org/repo",
        assets,
        sdist_on_pypi=sdist,
        pypi_version_exists=exists,
        github_tag=tag,
    )


def test_script_present():
    assert _SCRIPT.is_file()


def test_wheel_regex_parses_kernel_asset():
    m = mod._WHEEL_RE.match(
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    )
    assert m and m.group("py") == "cp312" and m.group("plat") == "linux_x86_64"
    local = mod._LOCAL_RE.search(m.group("ver"))
    assert local.group("cuda") == "12" and local.group("torch") == "2.7"
    assert local.group("abi") == "TRUE"


def test_wheel_present_at_both_anchors_records_abis():
    """mamba-like: prebuilt wheels at torch 2.7 and 2.5 (abiTRUE + abiFALSE)."""
    assets = [
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    ]
    r = _classify("2.2.6.post3", assets)
    b = r.anchor_status("2.7")
    a = r.anchor_status("2.5")
    assert b["status"] == "wheel-present" and b["abis_present"] == ["FALSE", "TRUE"]
    assert a["status"] == "wheel-present"
    assert r.torch_minors_with_wheels == ["2.5", "2.7"]


def test_source_build_required_when_anchor_missing_but_others_present():
    """causal-conv1d-like: torch 2.7 wheel exists, no torch 2.5 wheel -> A source-builds."""
    assets = [
        "causal_conv1d-1.6.2.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    ]
    r = _classify("1.6.2.post1", assets)
    assert r.anchor_status("2.7")["status"] == "wheel-present"
    assert r.anchor_status("2.5")["status"] == "source-build-required"


def test_sdist_only_is_source_build_required_at_all_anchors():
    r = _classify("9.9.9", [], sdist=True)
    assert r.anchor_status("2.7")["status"] == "source-build-required"
    assert r.anchor_status("2.5")["status"] == "source-build-required"


def test_truly_unavailable_when_no_wheel_and_no_sdist():
    r = _classify("9.9.9", [], sdist=False, exists=False, tag=None)
    assert r.anchor_status("2.7")["status"] == "unavailable"
    assert any("PyPI" in n for n in r.notes)
    assert any("no GitHub release" in n for n in r.notes)


def test_ngc_and_offtarget_tags_do_not_pollute_or_crash():
    """NGC-container (torch25.06), wrong-python, wrong-platform, and wrong-version
    assets are ignored; the int-tuple sort of the recorded minors must not crash."""
    assets = [
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch25.06cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp312-cp312-win_amd64.whl",
        "mamba_ssm-2.3.2.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    ]
    r = _classify("2.2.6.post3", assets)
    # Only the first (on-target, correct version) wheel qualifies for torch 2.7.
    assert r.anchor_status("2.7")["wheel_files"] == [
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    ]
    assert "2.7" in r.torch_minors_with_wheels and "25.06" in r.torch_minors_with_wheels


def test_cu11_wheels_excluded():
    r = _classify(
        "1.0.0", ["pkg-1.0.0+cu11torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"], sdist=True
    )
    # cu11 is not the cu12 target -> not counted as a wheel; sdist -> source-build-required.
    assert r.anchor_status("2.7")["status"] == "source-build-required"


def test_p0_06_finding_locked_mamba_torch27_wheel_present():
    """Regression lock on the observed P0-06 result: the real mamba-ssm 2.2.6.post3
    release ships a cu12torch2.7 cp312 abiTRUE wheel -> Option B is wheel-present, not
    a source build. Guards the claim the ADR-0002 amendment records."""
    real_mamba_assets = [
        "mamba_ssm-2.2.6.post3+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    ]
    r = _classify("2.2.6.post3", real_mamba_assets)
    status_b = r.anchor_status("2.7")
    assert status_b["status"] == "wheel-present"
    assert "TRUE" in status_b["abis_present"]
