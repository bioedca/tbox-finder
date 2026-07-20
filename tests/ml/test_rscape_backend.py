"""P2-10c validation gate — the real R-scape binary, run end-to-end (CLAUDE.md §8.5).

The unit tier (``tests/unit/test_covariation.py``) proves the parser and the
availability wiring against committed ``.helixcov`` fixtures. It cannot prove that
the *binary* still produces those numbers — that needs the binary, which lives in
``envs/rscape.yml`` and is installed in no CI job.

So this file carries the step's actual gate:

1. the installed R-scape reports the pinned version;
2. re-running it on the committed positive alignment reproduces the committed
   ``.helixcov`` byte-for-byte;
3. the **designed control fires**: the matched column-shuffled alignment — same
   sequences, same per-column composition, same ``#=GC SS_cons``, correlation
   between columns destroyed — scores zero covarying pairs while the positive
   clears the threshold, under both readings of the ADR prose.

Clause 3 is the one that matters. The natural rate cannot carry this gate: a
backend that had silently died would also report "no covariation", and that is
indistinguishable from a true negative. The control must fire **by construction**
or the gate is vacuous (the P2-10b lesson).

Arming: ``TBOX_REQUIRE_RSCAPE=1``. Not set in ``.github/workflows/ci.yml`` — CI
installs only the ``data`` env, so arming it there would fail every run. Run it
locally inside the pinned env after any change to the covariation backend::

    TBOX_REQUIRE_RSCAPE=1 conda run -n tbox-rscape python -m pytest tests/ml/test_rscape_backend.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tbox_finder.mining.covariation import (
    MIN_COVARYING_PAIRS,
    PINNED_RSCAPE_VERSION,
    RSCAPE_BINARY,
    AnyHelixCriterion,
    CovariationBackendError,
    backend_available,
    covariation_verdict,
    parse_helixcov,
    rscape_version,
    run_rscape,
)
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED

_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures" / "rscape"
_POSITIVE_STO = _FIXTURES / "classII_sub.sto"
_SHUFFLED_STO = _FIXTURES / "classII_sub.shuffled.sto"
_POSITIVE_HELIXCOV = _FIXTURES / "classII_sub.helixcov"
_SHUFFLED_HELIXCOV = _FIXTURES / "classII_sub.shuffled.helixcov"

_REQUIRE = os.environ.get("TBOX_REQUIRE_RSCAPE") == "1"

_BOTH_CRITERIA = list(AnyHelixCriterion)


def _need_binary() -> None:
    """Fail closed when armed; skip only when the operator did not ask for it."""
    if backend_available():
        return
    message = (
        f"{RSCAPE_BINARY} is not on PATH; run inside the pinned tbox-rscape env "
        "(envs/rscape.yml)"
    )
    if _REQUIRE:
        pytest.fail(f"TBOX_REQUIRE_RSCAPE=1 but {message}")
    pytest.skip(message)


def _need_fixtures() -> None:
    """Committed to git — a missing one is a broken checkout, so raise."""
    missing = [
        p
        for p in (_POSITIVE_STO, _SHUFFLED_STO, _POSITIVE_HELIXCOV, _SHUFFLED_HELIXCOV)
        if not p.is_file()
    ]
    if missing:
        raise AssertionError(f"committed fixtures missing: {missing}")


def test_installed_binary_matches_the_pin() -> None:
    """A different build would change verdicts silently.

    R-scape v2.0.5 recalculated the covariation power curves and lowered the
    CaCoFold power threshold, so the version is a methods-section fact, not an
    implementation detail (ADR-0002 A11).
    """
    _need_binary()
    assert rscape_version() == PINNED_RSCAPE_VERSION


def test_binary_resolves_inside_the_pinned_env() -> None:
    _need_binary()
    assert shutil.which(RSCAPE_BINARY) is not None


def test_committed_helixcov_reproduces_from_the_binary(tmp_path: Path) -> None:
    """The committed fixtures are what this binary actually emits, today."""
    _need_binary()
    _need_fixtures()

    for alignment, committed in (
        (_POSITIVE_STO, _POSITIVE_HELIXCOV),
        (_SHUFFLED_STO, _SHUFFLED_HELIXCOV),
    ):
        result = run_rscape(alignment, tmp_path / alignment.stem, outname="fx")
        expected = parse_helixcov(committed.read_text(encoding="utf-8"))
        assert result.helices == expected.helices, f"drift on {alignment.name}"


@pytest.mark.parametrize("criterion", _BOTH_CRITERIA, ids=lambda c: c.value)
def test_gate_designed_control_fires(criterion: AnyHelixCriterion, tmp_path: Path) -> None:
    """THE GATE — the control must mask by construction, the positive must clear it.

    Held fixed between the two arms: the sequences, every column's base-and-gap
    composition, the consensus structure, the alignment width, the E-value, and the
    binary. Varied: only the correlation *between* columns. So a separation here
    can only be covariation.
    """
    _need_binary()
    _need_fixtures()

    positive_status, positive_report = covariation_verdict(_POSITIVE_STO, criterion)
    control_status, control_report = covariation_verdict(_SHUFFLED_STO, criterion)

    assert positive_status == STATUS_PASSED, positive_report
    assert control_status == STATUS_FAILED, control_report

    assert control_report["total_covarying"] == 0
    assert control_report["max_covarying_in_one_helix"] == 0
    assert positive_report["max_covarying_in_one_helix"] > MIN_COVARYING_PAIRS
    assert positive_report["n_helices"] == control_report["n_helices"] > 0


def test_gate_control_is_matched_not_merely_weaker(tmp_path: Path) -> None:
    """The control's helices must be identical to the positive's.

    If the shuffle moved the helices, the arms would differ in geometry as well as
    covariation and the separation would no longer isolate the signal.
    """
    _need_binary()
    _need_fixtures()

    positive = run_rscape(_POSITIVE_STO, tmp_path / "pos", outname="fx")
    control = run_rscape(_SHUFFLED_STO, tmp_path / "ctl", outname="fx")

    assert [
        (h.left_start, h.left_end, h.right_start, h.right_end, h.n_basepairs)
        for h in positive.helices
    ] == [
        (h.left_start, h.left_end, h.right_start, h.right_end, h.n_basepairs)
        for h in control.helices
    ]
    assert positive.total_covarying > 0
    assert control.total_covarying == 0


def test_missing_alignment_raises_rather_than_failing_the_candidate() -> None:
    """An absent input must not read as "this candidate does not covary".

    ``failed`` makes a candidate minable; an I/O problem returning ``failed`` would
    mine away the Tier-2N loci the spare rule protects.
    """
    _need_binary()
    with pytest.raises(CovariationBackendError, match="not found"):
        run_rscape(Path("does-not-exist.sto"), Path("/tmp/tbox-rscape-missing"))


def test_alignment_without_ss_cons_raises(tmp_path: Path) -> None:
    """Helix-level aggregation needs ``#=GC SS_cons``; silence would be a false negative."""
    _need_binary()
    bare = tmp_path / "bare.sto"
    bare.write_text(
        "# STOCKHOLM 1.0\n\nseq1 ACGUACGUAC\nseq2 ACGUACGUAC\nseq3 AGGUACGUAC\n//\n",
        encoding="utf-8",
    )
    with pytest.raises(CovariationBackendError):
        run_rscape(bare, tmp_path / "out", outname="fx")
