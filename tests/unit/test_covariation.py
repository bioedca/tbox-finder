"""P2-10c — the R-scape covariation caller: parser, criteria, and availability wiring.

Pure-stdlib tier. Every clause here reads **git-tracked** fixtures
(``tests/fixtures/rscape/*.helixcov``, produced by ``scripts/make_rscape_fixture.py``
against the pinned R-scape 2.0.4.a), so a missing file is a broken checkout and
raises rather than skipping. No arming var — that is strictly stronger than a
``TBOX_REQUIRE_*`` which can rot into an unarmed skip-green.

The clauses that invoke the real binary live in ``tests/ml/test_rscape_backend.py``
behind ``TBOX_REQUIRE_RSCAPE``.
"""

from __future__ import annotations

import os
import stat
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
    round_backend_availability,
)
from tbox_finder.mining.spare_rule import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    SpareRuleEvidence,
    is_mining_excluded,
    mining_round_readiness,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "rscape"
_POSITIVE = _FIXTURES / "classII_sub.helixcov"
_SHUFFLED = _FIXTURES / "classII_sub.shuffled.helixcov"

_BOTH_CRITERIA = list(AnyHelixCriterion)


def _read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"committed fixture missing: {path}")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The designed control: matched on everything but the covariation signal
# --------------------------------------------------------------------------- #
def test_control_is_matched_on_helix_geometry() -> None:
    """The shuffled control must differ from the positive *only* in ``nbp_cov``.

    This is the clause that makes the separation below mean something. A control
    that also moved the helices, or changed how many base pairs each holds, would
    be measuring alignment geometry rather than covariation — the P2-10b lesson
    (a perturbation that changes more than the thing under test proves nothing
    about the thing under test).
    """
    positive = parse_helixcov(_read(_POSITIVE))
    control = parse_helixcov(_read(_SHUFFLED))

    assert positive.n_helices == control.n_helices > 0
    for helix_pos, helix_ctl in zip(positive.helices, control.helices, strict=True):
        assert (
            helix_pos.left_start,
            helix_pos.left_end,
            helix_pos.right_start,
            helix_pos.right_end,
            helix_pos.n_basepairs,
        ) == (
            helix_ctl.left_start,
            helix_ctl.left_end,
            helix_ctl.right_start,
            helix_ctl.right_end,
            helix_ctl.n_basepairs,
        )


@pytest.mark.parametrize("criterion", _BOTH_CRITERIA, ids=lambda c: c.value)
def test_designed_control_separates_under_every_criterion(
    criterion: AnyHelixCriterion,
) -> None:
    """Positive passes, matched control fails — under **both** readings.

    The fixture is deliberately outside the marginal band (60 sequences), so this
    separation is not a coin flip: a backend that silently stopped scoring would
    take the positive to ``failed`` and be caught here.
    """
    positive = parse_helixcov(_read(_POSITIVE))
    control = parse_helixcov(_read(_SHUFFLED))

    assert positive.status(criterion) == STATUS_PASSED
    assert control.status(criterion) == STATUS_FAILED


def test_control_has_zero_covariation_everywhere() -> None:
    """Not merely "below threshold" — the control must be flat zero.

    A control that scored 1 would pass ``< 2`` for the wrong reason and would stop
    discriminating the moment ``MIN_COVARYING_PAIRS`` changed.
    """
    control = parse_helixcov(_read(_SHUFFLED))
    assert control.total_covarying == 0
    assert control.max_covarying_in_one_helix == 0
    assert all(h.n_covarying == 0 for h in control.helices)


def test_positive_clears_the_threshold_with_margin() -> None:
    positive = parse_helixcov(_read(_POSITIVE))
    assert positive.max_covarying_in_one_helix > MIN_COVARYING_PAIRS
    assert positive.total_covarying > MIN_COVARYING_PAIRS


# --------------------------------------------------------------------------- #
# The two readings of "any-helix" are genuinely different operators
# --------------------------------------------------------------------------- #
def test_the_two_criteria_are_not_the_same_operator() -> None:
    """One covarying pair in each of two helices: looser passes, stricter fails.

    If this ever stops holding, the ``AnyHelixCriterion`` distinction has collapsed
    and the ADR-0006 decision it is waiting on has been made by accident.
    """
    two_helices_one_each = (
        "# RM 10-20 80-90, nbp = 4 nbp_cov = 1\n" "# RM 30-35 60-65, nbp = 3 nbp_cov = 1\n"
    )
    result = parse_helixcov(two_helices_one_each)
    assert result.satisfies(AnyHelixCriterion.TOTAL_ACROSS_HELICES) is True
    assert result.satisfies(AnyHelixCriterion.WITHIN_HELIX) is False


def test_report_carries_the_counterfactual() -> None:
    """A committed round report must record what the other operator would say."""
    result = parse_helixcov(
        "# RM 10-20 80-90, nbp = 4 nbp_cov = 1\n# RM 30-35 60-65, nbp = 3 nbp_cov = 1\n"
    )
    report = result.as_report(AnyHelixCriterion.WITHIN_HELIX)
    assert report["criterion"] == "within_helix"
    assert report["status"] == STATUS_FAILED
    assert report["other_criterion"] == "total_across_helices"
    assert report["status_under_other_criterion"] == STATUS_PASSED
    assert report["rscape_version"] == PINNED_RSCAPE_VERSION


def test_criterion_is_required_and_typed() -> None:
    """``covariation_verdict`` must not let a caller default into an operator."""
    with pytest.raises(TypeError):
        covariation_verdict(Path("unused.sto"))  # type: ignore[call-arg]
    with pytest.raises(CovariationBackendError):
        covariation_verdict(Path("unused.sto"), "within_helix")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parses_every_helix_record_in_the_fixture() -> None:
    result = parse_helixcov(_read(_POSITIVE))
    assert result.n_helices == sum(1 for line in _read(_POSITIVE).splitlines() if "nbp_cov" in line)
    assert result.n_helices == 8


def test_non_helix_lines_are_ignored() -> None:
    """``.helixcov`` interleaves ``# pvals:`` / ``# aggregated`` / blank lines."""
    result = parse_helixcov(
        "# RMs = 1 L = 100\n"
        "\n"
        "# RM 10-20 80-90, nbp = 4 nbp_cov = 3\n"
        "# pvals: 0.001,0.002,0.5,0.9\n"
        "# aggregated NONE\n"
    )
    assert result.n_helices == 1
    assert result.helices[0].n_covarying == 3


def test_empty_input_is_failed_not_passed() -> None:
    """No helices ⇒ no evidence of covariation ⇒ ``failed``, under both readings.

    ``failed`` (not an exception) is correct here because reaching the parser means
    R-scape ran; it is the *caller's* job to leave the disjunct ``unavailable``
    when it did not.
    """
    result = parse_helixcov("")
    assert result.n_helices == 0
    assert result.max_covarying_in_one_helix == 0
    for criterion in _BOTH_CRITERIA:
        assert result.status(criterion) == STATUS_FAILED


def test_impossible_helix_record_raises() -> None:
    """More covarying pairs than base pairs is a corrupt file, not a strong hit."""
    with pytest.raises(CovariationBackendError, match="nbp_cov"):
        parse_helixcov("# RM 10-20 80-90, nbp = 3 nbp_cov = 4\n")


def test_status_is_never_unavailable() -> None:
    for path in (_POSITIVE, _SHUFFLED):
        result = parse_helixcov(_read(path))
        for criterion in _BOTH_CRITERIA:
            assert result.status(criterion) != STATUS_UNAVAILABLE


# --------------------------------------------------------------------------- #
# Availability is probed, not asserted — the non-tautology clauses
# --------------------------------------------------------------------------- #
def _fake_binary(directory: Path) -> None:
    target = directory / RSCAPE_BINARY
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_availability_follows_the_real_path_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the genuine ``shutil.which`` code path by moving ``PATH``.

    Deliberately not a stub of ``backend_available`` itself: a test that
    monkeypatches the very precondition it exists to check proves nothing.
    """
    monkeypatch.setenv("PATH", "")
    assert backend_available() is False

    _fake_binary(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert backend_available() is True


def test_readiness_refuses_when_the_binary_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "")
    availability = round_backend_availability()
    assert availability["any_helix_rscape"] is False

    readiness = mining_round_readiness(availability)
    assert readiness["ready"] is False
    assert readiness["refusal_reason"]


def test_readiness_passes_only_when_the_binary_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_binary(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))

    availability = round_backend_availability()
    assert availability["any_helix_rscape"] is True

    readiness = mining_round_readiness(availability)
    assert readiness["ready"] is True
    assert readiness["refusal_reason"] is None
    assert readiness["protective_disjuncts_available"] == ["any_helix_rscape"]


def test_availability_cannot_be_forced_by_an_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the helper: ``any_helix_rscape`` has no override kwarg.

    ``mining_round_readiness`` accepts a plain ``dict[str, bool]``, so ``ready=true``
    is one keystroke away from being a literal. Availability produced through this
    constructor must track the environment even when the caller asks otherwise.
    """
    monkeypatch.setenv("PATH", "")
    with pytest.raises(TypeError):
        round_backend_availability(any_helix_rscape=True)  # type: ignore[call-arg]

    availability = round_backend_availability(
        relaxed_architecture=True, downstream_aaRS_synteny=True
    )
    assert availability["any_helix_rscape"] is False


def test_other_disjuncts_stay_caller_supplied() -> None:
    availability = round_backend_availability(relaxed_architecture=True)
    assert availability["relaxed_architecture"] is True
    assert availability["downstream_aaRS_synteny"] is False
    assert set(availability) == {
        "relaxed_architecture",
        "any_helix_rscape",
        "downstream_aaRS_synteny",
    }


# --------------------------------------------------------------------------- #
# The verdict lands in the spare rule with the right protective direction
# --------------------------------------------------------------------------- #
def test_a_passed_verdict_spares_the_candidate() -> None:
    positive = parse_helixcov(_read(_POSITIVE))
    evidence = SpareRuleEvidence(
        relaxed_architecture=STATUS_FAILED,
        any_helix_rscape=positive.status(AnyHelixCriterion.WITHIN_HELIX),
        downstream_aaRS_synteny=STATUS_FAILED,
    )
    assert is_mining_excluded(evidence) is True


def test_a_failed_verdict_alone_does_not_mine_a_candidate() -> None:
    """All three must have *run* and failed before a candidate becomes minable.

    A covariation ``failed`` next to two ``unavailable`` disjuncts must still spare
    — this is the fail-closed arm, and it is what keeps a half-instrumented round
    from mining away Tier-2N loci.
    """
    control = parse_helixcov(_read(_SHUFFLED))
    evidence = SpareRuleEvidence(any_helix_rscape=control.status(AnyHelixCriterion.WITHIN_HELIX))
    assert evidence.any_helix_rscape == STATUS_FAILED
    assert is_mining_excluded(evidence) is True

    fully_evaluated = SpareRuleEvidence(
        relaxed_architecture=STATUS_FAILED,
        any_helix_rscape=STATUS_FAILED,
        downstream_aaRS_synteny=STATUS_FAILED,
    )
    assert is_mining_excluded(fully_evaluated) is False


def test_pinned_version_constant_is_not_empty() -> None:
    assert PINNED_RSCAPE_VERSION
    assert os.path.basename(RSCAPE_BINARY) == RSCAPE_BINARY


def test_version_regex_skips_the_banner_title_line() -> None:
    """``# R-scape :: RNA Structural...`` precedes the version line in the banner.

    A bare ``\\S+`` after the tool name reports the version as ``::`` — caught by the
    armed gate, pinned here so the bare-CI tier holds it too.
    """
    from tbox_finder.mining.covariation import _VERSION_RE

    title = "# R-scape :: RNA Structural Covariation Above Phylogenetic Expectation"
    version = "# R-scape 2.0.4.a (Dec 2023)"
    assert _VERSION_RE.match(title) is None
    matched = _VERSION_RE.match(version)
    assert matched is not None
    assert matched.group("version") == "2.0.4.a"
