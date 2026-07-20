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
import re
import stat
from collections import Counter
from pathlib import Path

import pytest

from tbox_finder.mining import covariation
from tbox_finder.mining.covariation import (
    DEFAULT_EVALUE,
    MIN_COVARYING_PAIRS,
    PINNED_RSCAPE_VERSION,
    RSCAPE_BINARY,
    AnyHelixCriterion,
    CovariationBackendError,
    backend_available,
    covariation_verdict,
    parse_helixcov,
    round_backend_availability,
    rscape_version,
    run_rscape,
    stockholm_stats,
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
    assert report["rscape_version_pinned"] == PINNED_RSCAPE_VERSION
    assert report["evalue_pinned"] == DEFAULT_EVALUE


def test_report_never_claims_a_version_it_did_not_observe() -> None:
    """Parsed text is no evidence of which binary produced it.

    Stamping the pinned constant unconditionally would let a report built by a
    2.6.x binary claim 2.0.4.a — and since v2.0.5 recalculated the covariation
    power curves, that is the field a reader most needs to be true.
    """
    parsed = parse_helixcov(_read(_POSITIVE))
    report = parsed.as_report(AnyHelixCriterion.WITHIN_HELIX)
    assert report["rscape_version_observed"] is None
    assert report["rscape_version_matches_pin"] is False

    ran = parse_helixcov(_read(_POSITIVE), rscape_version=PINNED_RSCAPE_VERSION)
    ran_report = ran.as_report(AnyHelixCriterion.WITHIN_HELIX)
    assert ran_report["rscape_version_observed"] == PINNED_RSCAPE_VERSION
    assert ran_report["rscape_version_matches_pin"] is True

    other = parse_helixcov(_read(_POSITIVE), rscape_version="2.6.11")
    other_report = other.as_report(AnyHelixCriterion.WITHIN_HELIX)
    assert other_report["rscape_version_observed"] == "2.6.11"
    assert other_report["rscape_version_matches_pin"] is False


def test_criterion_is_required_and_typed() -> None:
    """``covariation_verdict`` must not let a caller default into an operator."""
    with pytest.raises(TypeError):
        covariation_verdict(Path("unused.sto"))  # type: ignore[call-arg]
    # min_sequences is keyword-required with no default, for the same reason.
    with pytest.raises(TypeError):
        covariation_verdict(  # type: ignore[call-arg]
            Path("unused.sto"), AnyHelixCriterion.WITHIN_HELIX
        )
    # `match=` is load-bearing: without it every layer below (PATH probe, missing
    # alignment) raises the same class, and deleting the isinstance guard entirely
    # left this test green.
    with pytest.raises(CovariationBackendError, match="criterion must be an AnyHelixCriterion"):
        covariation_verdict(  # type: ignore[arg-type]
            Path("unused.sto"), "within_helix", min_sequences=4
        )


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
def _fake_binary(directory: Path, version: str = PINNED_RSCAPE_VERSION) -> None:
    """A stand-in that prints R-scape's real two-line banner, version configurable."""
    target = directory / RSCAPE_BINARY
    target.write_text(
        "#!/bin/sh\n"
        "echo '# R-scape :: RNA Structural Covariation Above Phylogenetic Expectation'\n"
        f"echo '# R-scape {version} (Dec 2023)'\n"
        "exit 0\n",
        encoding="utf-8",
    )
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


def test_an_unpinned_build_reads_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presence is not enough — the version is part of availability.

    v2.0.5 recalculated the covariation power curves, which moves ``nbp_cov``, the
    single number every verdict is computed from. A round on an unpinned build
    would produce verdicts this repo cannot reproduce, so a mismatch must fail
    **closed**: unavailable ⇒ the readiness gate refuses the round.
    """
    _fake_binary(tmp_path, version="2.6.11")
    monkeypatch.setenv("PATH", str(tmp_path))

    assert rscape_version() == "2.6.11"
    assert backend_available() is False

    readiness = mining_round_readiness(round_backend_availability())
    assert readiness["ready"] is False


def test_an_unparseable_banner_reads_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unidentifiable build must not silently score candidates."""
    target = tmp_path / RSCAPE_BINARY
    target.write_text("#!/bin/sh\necho 'no banner here'\nexit 0\n", encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(CovariationBackendError, match="version banner"):
        rscape_version()
    assert backend_available() is False


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


def test_a_hung_binary_reads_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that times out must refuse the round, not crash the readiness check.

    ``backend_available`` catches only ``CovariationBackendError``, so every way the
    probe can fail has to arrive as one — a bare ``TimeoutExpired`` escaping here
    would take down the caller instead of returning False.
    """
    target = tmp_path / RSCAPE_BINARY
    target.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # `sleep` is resolved through PATH *inside* the script, so a PATH holding only
    # tmp_path makes the fake exit 127 instantly and this test would silently
    # exercise the no-banner branch instead of the timeout one. tmp_path stays
    # first, so `shutil.which` still resolves the fake.
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path), "/usr/bin", "/bin"]))
    monkeypatch.setattr(covariation, "VERSION_PROBE_TIMEOUT_S", 0.5)

    with pytest.raises(CovariationBackendError, match="timed out"):
        rscape_version()
    assert backend_available() is False
    assert mining_round_readiness(round_backend_availability())["ready"] is False


def test_an_unexecutable_binary_reads_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that resolves on PATH but cannot exec must also fail closed."""
    target = tmp_path / RSCAPE_BINARY
    target.write_text("\x7fELF not really\n", encoding="latin-1")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(CovariationBackendError, match="could not execute"):
        rscape_version()
    assert backend_available() is False


def test_run_refuses_a_pre_existing_helixcov(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover output must never be read as this candidate's verdict.

    R-scape exits 0 without writing a ``.helixcov`` when the alignment carries no
    ``SS_cons``. With a reused outdir that means an earlier candidate's file would
    be parsed as this one's — a stale ``passed`` on a candidate never scored, the
    fail-open direction the module exists to prevent.
    """
    _fake_binary(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))

    alignment = tmp_path / "aln.sto"
    alignment.write_text("# STOCKHOLM 1.0\n//\n", encoding="utf-8")
    outdir = tmp_path / "out"
    outdir.mkdir()
    stale = outdir / "fx.helixcov"
    stale.write_text("# RM 1-5 20-24, nbp = 5 nbp_cov = 5\n", encoding="utf-8")

    with pytest.raises(CovariationBackendError, match="pre-existing"):
        run_rscape(alignment, outdir, outname="fx")
    assert stale.read_text(encoding="utf-8").strip().endswith("nbp_cov = 5")


# --------------------------------------------------------------------------- #
# The control's matchedness, asserted from the alignments themselves
# --------------------------------------------------------------------------- #
def _read_alignment(path: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    if not path.is_file():
        raise AssertionError(f"committed fixture missing: {path}")
    seqs: list[tuple[str, str]] = []
    gc: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#=GC "):
            _, tag, val = line.split(None, 2)
            gc[tag] = val
        elif line.startswith("#") or line.strip() in {"", "//"}:
            continue
        else:
            name, aseq = line.split(None, 1)
            seqs.append((name, aseq.strip()))
    return seqs, gc


def test_the_control_alignment_is_genuinely_matched() -> None:
    """The gate's headline claim, asserted against the fixtures that carry it.

    ``tests/ml/test_rscape_backend.py`` claims "a separation here can only be
    covariation". That is only true if the two alignments differ in *nothing else*.
    Comparing R-scape's helix geometry does not establish it: geometry is a function
    of ``#=GC SS_cons`` and the gap pattern, which the generator copies verbatim
    into both arms — so an arm rebuilt by i.i.d. base resampling, a dinucleotide
    shuffle, a different seeded subsample, or a *row*-wise permutation would keep
    identical geometry, score zero everywhere, and leave every other test green
    while the separation silently became "composition" rather than "covariation".

    So this asserts the property directly: same sequence names in the same order,
    same width, same consensus annotation, and **per-column base-and-gap composition
    identical** — leaving inter-column correlation as the only difference.
    """
    positive, gc_pos = _read_alignment(_FIXTURES / "classII_sub.sto")
    control, gc_ctl = _read_alignment(_FIXTURES / "classII_sub.shuffled.sto")

    assert [n for n, _ in positive] == [n for n, _ in control]
    assert len(positive) == 60, "the fixture must stay clear of the marginal band"
    assert gc_pos == gc_ctl
    assert "SS_cons" in gc_pos

    widths = {len(a) for _, a in positive} | {len(a) for _, a in control}
    assert len(widths) == 1, f"ragged alignment: widths {sorted(widths)}"

    pos_columns = list(zip(*[a for _, a in positive], strict=True))
    ctl_columns = list(zip(*[a for _, a in control], strict=True))
    assert len(pos_columns) == len(ctl_columns) == widths.pop()
    mismatched = [
        i
        for i, (cp, cc) in enumerate(zip(pos_columns, ctl_columns, strict=True))
        if Counter(cp) != Counter(cc)
    ]
    assert not mismatched, f"{len(mismatched)} column(s) differ in composition: {mismatched[:5]}"


def test_the_control_is_not_simply_a_copy_of_the_positive() -> None:
    """Composition-identical is necessary but not sufficient — it must also differ.

    A control that *is* the positive would satisfy every matchedness clause above
    and then trivially "separate" nothing.
    """
    positive, _ = _read_alignment(_FIXTURES / "classII_sub.sto")
    control, _ = _read_alignment(_FIXTURES / "classII_sub.shuffled.sto")
    assert [a for _, a in positive] != [a for _, a in control]


# --------------------------------------------------------------------------- #
# Alignment depth: R-scape reports "no power" exactly as it reports "no signal"
# --------------------------------------------------------------------------- #
def test_stockholm_stats_counts_alignments_and_sequences() -> None:
    single = (_FIXTURES / "classII_sub.sto").read_text(encoding="utf-8")
    assert stockholm_stats(single) == (1, 60)

    doubled = single + single
    assert stockholm_stats(doubled) == (2, 60)


def test_min_sequences_is_required_and_validated(tmp_path: Path) -> None:
    """No default, for the same reason ``criterion`` has none — it is a threshold
    on Tier-2N protection, i.e. an ADR-0006 decision owed at P2-10e."""
    for bad in (0, -1, True, "4"):
        with pytest.raises(CovariationBackendError, match="min_sequences"):
            covariation_verdict(
                tmp_path / "x.sto",
                AnyHelixCriterion.WITHIN_HELIX,
                min_sequences=bad,  # type: ignore[arg-type]
            )


def test_depth_is_recorded_in_every_report() -> None:
    """A committed round must be re-gradable against whatever depth gets pinned."""
    parsed = parse_helixcov(_read(_POSITIVE))
    assert parsed.as_report(AnyHelixCriterion.WITHIN_HELIX)["n_sequences"] is None

    with_depth = parse_helixcov(_read(_POSITIVE), n_sequences=7)
    assert with_depth.as_report(AnyHelixCriterion.WITHIN_HELIX)["n_sequences"] == 7


# --------------------------------------------------------------------------- #
# The declared helix count is an in-band cross-check against silent skips
# --------------------------------------------------------------------------- #
def test_declared_helix_count_must_match_the_records_parsed() -> None:
    """A record the regex misses deflates nbp_cov into a false ``failed`` ⇒ minable.

    R-scape states its own count in the first line of every ``.helixcov``, so the
    oracle is already in the file; silently returning the smaller number is the
    fail-open direction.
    """
    with pytest.raises(CovariationBackendError, match="helix-count mismatch"):
        parse_helixcov(
            "# RMs = 3 L = 100\n"
            "# RM 10-20 80-90, nbp = 4 nbp_cov = 3\n"
            "# RM 30-35 60-65, nbp = 3 nbp_cov = 1\n"
        )


def test_committed_fixtures_agree_with_their_declared_count() -> None:
    for path in (_POSITIVE, _SHUFFLED):
        text = _read(path)
        declared = int(re.search(r"# RMs = (\d+)", text).group(1))
        assert parse_helixcov(text).n_helices == declared


# --------------------------------------------------------------------------- #
# run_rscape's guards, exercised in the bare tier with a scripted fake binary
# --------------------------------------------------------------------------- #
def _scripted_binary(directory: Path, body: str, version: str = PINNED_RSCAPE_VERSION) -> Path:
    """A fake that answers the `-h` probe, then runs ``body`` for a real invocation.

    The probe and the run go through the same executable, so a fake has to serve
    both: `-h` prints the banner and exits, anything else falls through to ``body``.
    """
    target = directory / RSCAPE_BINARY
    target.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-h" ]; then\n'
        "  echo '# R-scape :: RNA Structural Covariation Above Phylogenetic Expectation'\n"
        f"  echo '# R-scape {version} (Dec 2023)'\n"
        "  exit 0\n"
        "fi\n" + body,
        encoding="utf-8",
    )
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def _one_alignment(path: Path, n_seqs: int = 8, width: int = 12) -> Path:
    lines = ["# STOCKHOLM 1.0", ""]
    lines += [f"seq{i:03d} {'ACGU' * (width // 4)}" for i in range(n_seqs)]
    lines.append(f"#=GC SS_cons {'<' * (width // 2)}{'>' * (width // 2)}")
    lines.append("//")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_run_refuses_an_unpinned_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The version-mismatch refusal inside ``run_rscape`` — untested until now.

    ``backend_available``'s version gate is a *different* code path, and
    ``run_rscape`` is callable without ever consulting it.
    """
    _scripted_binary(tmp_path, "exit 0\n", version="2.6.11")
    monkeypatch.setenv("PATH", str(tmp_path))
    alignment = _one_alignment(tmp_path / "aln.sto")

    with pytest.raises(CovariationBackendError, match="envs/rscape.yml pins"):
        run_rscape(alignment, tmp_path / "out", outname="fx")


def test_run_refuses_a_multi_alignment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-scape rewrites <outname>.helixcov per alignment — last one wins.

    Without this guard the verdict returned belongs to whichever alignment happened
    to be last in the file, not to the candidate being scored.
    """
    _scripted_binary(tmp_path, "exit 0\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    one = _one_alignment(tmp_path / "one.sto").read_text(encoding="utf-8")
    two = tmp_path / "two.sto"
    two.write_text(one + one, encoding="utf-8")

    with pytest.raises(CovariationBackendError, match="2 alignment"):
        run_rscape(two, tmp_path / "out", outname="fx")


def test_run_surfaces_a_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted_binary(tmp_path, "echo 'boom' >&2\nexit 3\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    alignment = _one_alignment(tmp_path / "aln.sto")

    with pytest.raises(CovariationBackendError, match=r"failed \(rc=3\)"):
        run_rscape(alignment, tmp_path / "out", outname="fx")


def test_run_refuses_when_no_helixcov_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-scape exits 0 without writing this file when the alignment has no SS_cons.

    Silence must be an error: a missing file that read as ``failed`` would mine a
    candidate the backend never scored.
    """
    _scripted_binary(tmp_path, "exit 0\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    alignment = _one_alignment(tmp_path / "aln.sto")

    with pytest.raises(CovariationBackendError, match="produced no helix-level output"):
        run_rscape(alignment, tmp_path / "out", outname="fx")


def test_run_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted_binary(tmp_path, "sleep 30\n")
    # `sleep` resolves through PATH inside the script, so /bin must stay reachable
    # or the fake exits 127 and this test would silently grade a different branch.
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path), "/usr/bin", "/bin"]))
    alignment = _one_alignment(tmp_path / "aln.sto")

    with pytest.raises(CovariationBackendError, match="timed out"):
        run_rscape(alignment, tmp_path / "out", outname="fx", timeout_s=0.5)


def test_run_passes_the_requested_evalue_to_the_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report publishes ``evalue``; nothing else bound it to the argv.

    E is what *defines* a significant pair, so a report claiming 0.001 while the
    binary ran at its 0.05 default is the same provenance defect as an unobserved
    version stamp — and invisible to the byte-for-byte fixture check, because 0.05
    is R-scape's own default.
    """
    argv_dump = tmp_path / "argv.txt"
    _scripted_binary(tmp_path, f'printf "%s\\n" "$@" > {argv_dump}\nexit 7\n')
    monkeypatch.setenv("PATH", str(tmp_path))
    alignment = _one_alignment(tmp_path / "aln.sto")

    with pytest.raises(CovariationBackendError):
        run_rscape(alignment, tmp_path / "out", outname="fx", evalue=0.001)

    argv = argv_dump.read_text(encoding="utf-8").split("\n")
    assert "-E" in argv
    assert argv[argv.index("-E") + 1] == "0.001"
    assert "--onemsa" in argv


def _writer_body(nbp_cov: int) -> str:
    """Fake-binary body that honours the real ``--outdir``/``--outname`` it is given.

    ``covariation_verdict`` runs inside its own TemporaryDirectory, so a fake that
    writes to a path the test picked would never be found — the fake has to read its
    own argv, exactly as R-scape does.
    """
    return (
        'outdir=""; outname=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    --outdir) outdir="$2"; shift 2;;\n'
        '    --outname) outname="$2"; shift 2;;\n'
        "    *) shift;;\n"
        "  esac\n"
        "done\n"
        'echo "# RMs = 1 L = 12" > "$outdir/$outname.helixcov"\n'
        f'echo "# RM 1-3 10-12, nbp = 3 nbp_cov = {nbp_cov}" >> "$outdir/$outname.helixcov"\n'
        "exit 0\n"
    )


def test_depth_guard_bites_in_the_bare_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shallow-alignment refusal, exercised without the real binary.

    The ml-tier version of this needs R-scape and therefore skips in CI — so
    deleting the guard entirely left the bare tier green (caught by sabotage). A
    scripted fake that writes a real ``.helixcov`` closes that hole: below
    ``min_sequences`` this must raise (⇒ the caller records ``unavailable`` ⇒ the
    candidate is **spared**) rather than return the all-zero ``failed`` that would
    mine it.
    """
    _scripted_binary(tmp_path, _writer_body(0))
    monkeypatch.setenv("PATH", str(tmp_path))
    alignment = _one_alignment(tmp_path / "aln.sto", n_seqs=4)

    with pytest.raises(CovariationBackendError, match="below min_sequences"):
        covariation_verdict(alignment, AnyHelixCriterion.WITHIN_HELIX, min_sequences=8)


def test_depth_guard_admits_a_deep_enough_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """...and is not simply always-on: at or above the floor it returns a verdict."""
    _scripted_binary(tmp_path, _writer_body(3))
    monkeypatch.setenv("PATH", str(tmp_path))
    alignment = _one_alignment(tmp_path / "aln.sto", n_seqs=8)

    status, report = covariation_verdict(alignment, AnyHelixCriterion.WITHIN_HELIX, min_sequences=8)
    assert status == STATUS_PASSED
    assert report["n_sequences"] == 8
    assert report["rscape_version_matches_pin"] is True
