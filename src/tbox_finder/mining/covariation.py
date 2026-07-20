"""The (a) any-helix R-scape covariation backend of the §9.1 mining spare rule (P2-10c).

This module supplies the **covariation caller**: it probes for a pinned R-scape
binary, runs it on a *supplied* Stockholm alignment, and parses the helix-level
output into the ``passed`` / ``failed`` verdict that
:class:`tbox_finder.mining.spare_rule.SpareRuleEvidence` expects for the
``any_helix_rscape`` disjunct.

What this module is **not**
--------------------------
It is not the whole disjunct. R-scape scores an **alignment**, never a single
sequence, and ADR-0006 D2→D7 require that alignment to be the *model-independent*
one: a homolog set assembled by nhmmer / BLAST / cmsearch-from-candidate, aligned
by a CM-free de-novo structure-aware aligner. As of P2-10c the repo has **no
homolog-search target database** (``data/external`` is T-box-specific reference
FASTA, not RefSeq/GTDB/nt) and no ``hmmer``/``blast`` in any ``envs/*.yml``. That
half — the per-candidate MSA supply — is **P2-10c′**, sequenced with P6-01.

The consequence is load-bearing and deliberate: :func:`backend_available` reports
on the *binary*, and :func:`covariation_verdict` requires an alignment the caller
must already possess. Nothing here lets a round declare
``backend_availability["any_helix_rscape"] = True`` while being unable to produce
a per-candidate verdict — that combination spares every candidate and emits a
round report of zeroes indistinguishable from "there was nothing to mine", which
is the exact failure :func:`tbox_finder.mining.hard_negative.mine_round` says the
readiness gate exists to prevent. :func:`round_backend_availability` therefore
derives availability from a **probe**, never from a literal.

The two readings of "any-helix" (**unresolved — caller must choose**)
--------------------------------------------------------------------
ADR-0006 D11 names the spare-rule disjunct "**any-helix** R-scape covariation"
and D2's carve-out spells it "≥ 2 significant pairs in any conserved helix". That
prose admits two coherent operators, and they are not equivalent:

``WITHIN_HELIX``
    ≥ 2 significant covarying pairs inside a **single** helix
    (``max`` over helices of ``nbp_cov``).
``TOTAL_ACROSS_HELICES``
    ≥ 2 significant covarying pairs **summed** across helices
    (``sum`` over helices of ``nbp_cov``) — the looser, more *protective* reading.

Measured divergence on seeded subsamples of the class-II corpus alignment
(``data/interim/splits/aligned/class_II.sto``, R-scape 2.0.4.a, E ≤ 0.05): the two
agree on high-power alignments (0 % divergence at 5–8 and 60 sequences) and
disagree on **up to 33 %** of low-power ones (10–30 sequences: 17 %, 25 %, 17 %,
33 %, 8 %) — precisely the regime a real candidate's homolog set occupies. Every
divergent case is one the looser operator spares and the stricter one sends to the
mining pool, i.e. the choice is directly a Tier-2N-protection choice.

:class:`AnyHelixCriterion` therefore has **no default** and
:func:`covariation_verdict` requires it explicitly. Pinning one is an ADR-0006
decision owed at P2-10e, and until it is taken no caller can make it by accident.
:class:`CovariationResult` carries **both** statistics regardless, so a committed
round report records what the other operator would have said.

Tooling note
------------
R-scape's CLI flags and output-file grammar here were read off the **installed
binary** (``R-scape -h``; the ``.helixcov`` files it emits), not from
documentation — context7 carries no R-scape corpus, the same precedent recorded in
:mod:`tbox_finder.infernal` for Infernal's ``--fmt 1`` tblout fields.

PRD §9.1; ADR-0006 D2/D7/D11; ADR-0005 D14; ADR-0002 A11 (the ``rscape`` pin).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED

# --------------------------------------------------------------------------- #
# Pinned tool
# --------------------------------------------------------------------------- #
#: The R-scape executable name, as installed by ``envs/rscape.yml``.
RSCAPE_BINARY = "R-scape"

#: The version ``envs/rscape.yml`` pins. R-scape's own banner reports this string.
#: Asserted rather than assumed: the v2.0.5 changelog recalculated the covariation
#: power curves, so a silently-different build would change verdicts (ADR-0002 A11).
PINNED_RSCAPE_VERSION = "2.0.4.a"

#: E-value target for base-pair significance. ADR-0006 D2 pins **E ≤ 0.05**, which
#: is also R-scape's own default; passed explicitly so the committed value is the
#: one that runs rather than whatever a future build defaults to.
DEFAULT_EVALUE = 0.05

#: The ``any-helix`` criterion needs ≥ 2 significant covarying pairs (ADR-0006 D2
#: carve-out). Frozen here — no CLI/config override.
MIN_COVARYING_PAIRS = 2

#: Default subprocess timeout. The 120-sequence × 1,460-column sizing probe ran in
#: ~1.0 s, so this is a runaway guard, not a working budget.
DEFAULT_RSCAPE_TIMEOUT_S = 900.0

#: Timeout for the `-h` version probe. Separate from the run budget: printing a
#: banner is instantaneous, and the probe sits on the readiness path, so it must
#: not inherit a 15-minute ceiling.
VERSION_PROBE_TIMEOUT_S = 60.0

#: ``# RM <i>-<j> <k>-<l>, nbp = <n> nbp_cov = <m>`` — one helix per match in the
#: ``.helixcov`` file. ``nbp`` is the helix's base-pair count; ``nbp_cov`` is how
#: many of them covary significantly at the requested E-value.
_HELIX_RE = re.compile(
    r"^#\s*RM\s+(?P<i>\d+)-(?P<j>\d+)\s+(?P<k>\d+)-(?P<l>\d+)\s*,\s*"
    r"nbp\s*=\s*(?P<nbp>\d+)\s+nbp_cov\s*=\s*(?P<nbp_cov>\d+)\s*$"
)

#: ``# R-scape 2.0.4.a (Dec 2023)`` — the banner both ``-h`` and a real run print.
#: The version token must start with a digit: the line *above* it in the banner is
#: ``# R-scape :: RNA Structural Covariation Above Phylogenetic Expectation``, and a
#: bare ``\S+`` happily reports the version as ``::``.
_VERSION_RE = re.compile(r"^#\s*R-scape\s+(?P<version>\d\S*)")


class CovariationBackendError(RuntimeError):
    """Raised when the R-scape backend cannot produce a verdict.

    Deliberately an error rather than a ``failed`` verdict: ``failed`` means "the
    backend ran and this candidate does not covary", which makes the candidate
    **minable**. An absent or broken binary returning ``failed`` would therefore
    mine away exactly the Tier-2N loci the spare rule exists to protect — the
    fail-open direction. Callers that cannot run the backend must leave the
    disjunct ``unavailable`` (⇒ spared) instead.
    """


class AnyHelixCriterion(Enum):
    """Which reading of ADR-0006's "any-helix" prose to apply. No default.

    See the module docstring for the measured divergence between the two and why
    pinning one is an ADR-0006 decision rather than an implementation detail.
    """

    #: ≥ ``MIN_COVARYING_PAIRS`` significant pairs inside a single helix (stricter).
    WITHIN_HELIX = "within_helix"
    #: ≥ ``MIN_COVARYING_PAIRS`` significant pairs summed across helices (looser,
    #: more protective of Tier-2N).
    TOTAL_ACROSS_HELICES = "total_across_helices"


@dataclass(frozen=True)
class Helix:
    """One ``# RM`` record of a ``.helixcov`` file."""

    left_start: int
    left_end: int
    right_start: int
    right_end: int
    n_basepairs: int
    n_covarying: int


@dataclass(frozen=True)
class CovariationResult:
    """Parsed helix-level covariation for one alignment.

    Both operator statistics are carried unconditionally so a committed round
    report records what the *other* reading of the ADR prose would have said.
    """

    helices: tuple[Helix, ...]
    evalue: float
    #: Version the binary reported when it produced this result. ``None`` for a
    #: result parsed from a file, which carries no evidence about what ran.
    rscape_version: str | None = None

    @property
    def n_helices(self) -> int:
        return len(self.helices)

    @property
    def max_covarying_in_one_helix(self) -> int:
        """``WITHIN_HELIX`` statistic — 0 when the alignment has no helices."""
        return max((h.n_covarying for h in self.helices), default=0)

    @property
    def total_covarying(self) -> int:
        """``TOTAL_ACROSS_HELICES`` statistic."""
        return sum(h.n_covarying for h in self.helices)

    def satisfies(self, criterion: AnyHelixCriterion) -> bool:
        """Whether the any-helix disjunct passes under ``criterion``."""
        if criterion is AnyHelixCriterion.WITHIN_HELIX:
            return self.max_covarying_in_one_helix >= MIN_COVARYING_PAIRS
        if criterion is AnyHelixCriterion.TOTAL_ACROSS_HELICES:
            return self.total_covarying >= MIN_COVARYING_PAIRS
        raise CovariationBackendError(f"unknown any-helix criterion {criterion!r}")

    def status(self, criterion: AnyHelixCriterion) -> str:
        """``STATUS_PASSED`` / ``STATUS_FAILED`` — never ``STATUS_UNAVAILABLE``.

        A parsed result means the backend *ran*; "unavailable" is the caller's to
        record when it did not.
        """
        return STATUS_PASSED if self.satisfies(criterion) else STATUS_FAILED

    def as_report(self, criterion: AnyHelixCriterion) -> dict[str, Any]:
        """Auditable record of the verdict **and** its counterfactual.

        The version fields report what **actually ran** (``None`` for a result
        parsed from a file, which cannot certify a binary). Stamping the pinned
        constant here unconditionally would let a report built by a 2.6.x binary
        claim 2.0.4.a — and since v2.0.5 recalculated the covariation power curves,
        that is the one field a reader most needs to be true.
        """
        other = (
            AnyHelixCriterion.TOTAL_ACROSS_HELICES
            if criterion is AnyHelixCriterion.WITHIN_HELIX
            else AnyHelixCriterion.WITHIN_HELIX
        )
        return {
            "criterion": criterion.value,
            "status": self.status(criterion),
            "n_helices": self.n_helices,
            "max_covarying_in_one_helix": self.max_covarying_in_one_helix,
            "total_covarying": self.total_covarying,
            "min_covarying_pairs_required": MIN_COVARYING_PAIRS,
            "evalue": self.evalue,
            "rscape_version_observed": self.rscape_version,
            "rscape_version_pinned": PINNED_RSCAPE_VERSION,
            "rscape_version_matches_pin": self.rscape_version == PINNED_RSCAPE_VERSION,
            "status_under_other_criterion": self.status(other),
            "other_criterion": other.value,
        }


# --------------------------------------------------------------------------- #
# Availability — probed and version-gated, never asserted
# --------------------------------------------------------------------------- #
def _binary_on_path() -> bool:
    """Whether *some* ``R-scape`` resolves on ``PATH`` — says nothing about which."""
    return shutil.which(RSCAPE_BINARY) is not None


def rscape_version() -> str:
    """Version string reported by the installed binary.

    Raises :class:`CovariationBackendError` if the binary is absent or its banner
    is unparseable — an unidentifiable build must not silently score candidates.
    """
    if not _binary_on_path():
        raise CovariationBackendError(
            f"{RSCAPE_BINARY} is not on PATH; run inside the pinned tbox-rscape env "
            f"(envs/rscape.yml, rscape={PINNED_RSCAPE_VERSION})"
        )
    # Every way the probe can fail becomes CovariationBackendError, because
    # backend_available() catches exactly that and turns it into "unavailable".
    # A hung binary raising TimeoutExpired, or an unexecutable one raising OSError,
    # must make the round refuse — not crash the readiness probe itself.
    try:
        proc = subprocess.run(
            [RSCAPE_BINARY, "-h"],
            capture_output=True,
            text=True,
            check=False,
            timeout=VERSION_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise CovariationBackendError(
            f"`{RSCAPE_BINARY} -h` timed out after {VERSION_PROBE_TIMEOUT_S}s"
        ) from exc
    except OSError as exc:
        raise CovariationBackendError(f"could not execute {RSCAPE_BINARY}: {exc}") from exc
    for line in (proc.stdout + proc.stderr).splitlines():
        match = _VERSION_RE.match(line.strip())
        if match:
            return match.group("version")
    raise CovariationBackendError(f"could not read a version banner from `{RSCAPE_BINARY} -h`")


def backend_available() -> bool:
    """Whether the **pinned** R-scape is callable here — presence *and* version.

    Presence alone is not enough. `rscape` 2.0.5 reverted the null-model gap
    treatment and **recalculated the covariation power curves**, which moves
    ``nbp_cov`` — the single number every verdict is computed from. A round run on
    an unpinned build would produce verdicts this repo cannot reproduce or defend,
    so a mismatch reads as *unavailable*, and the readiness gate then refuses the
    round. That is the fail-closed direction: a refused round costs a step, an
    unpinned round costs the mined curriculum's provenance.
    """
    if not _binary_on_path():
        return False
    try:
        return rscape_version() == PINNED_RSCAPE_VERSION
    except CovariationBackendError:
        return False


def round_backend_availability(
    *,
    relaxed_architecture: bool = False,
    downstream_aaRS_synteny: bool = False,
) -> dict[str, bool]:
    """Build the ``mine_round`` availability map with (a) **probed**, not asserted.

    ``mining_round_readiness`` takes a caller-supplied ``dict[str, bool]``, so a
    round can be declared ready by typing ``True``. This helper is the honest
    constructor: the ``any_helix_rscape`` entry comes from :func:`backend_available`
    and cannot be overridden by an argument. The other two disjuncts stay
    caller-supplied because they have no backend yet (P2-10c′ / P6-01).
    """
    return {
        "relaxed_architecture": bool(relaxed_architecture),
        "any_helix_rscape": backend_available(),
        "downstream_aaRS_synteny": bool(downstream_aaRS_synteny),
    }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_helixcov(
    text: str,
    *,
    evalue: float = DEFAULT_EVALUE,
    rscape_version: str | None = None,
) -> CovariationResult:
    """Parse a ``.helixcov`` file body into a :class:`CovariationResult`.

    Pure text → dataclass; no I/O, no subprocess. This is the layer the bare-CI
    unit tier exercises against committed fixtures, following the
    :mod:`tbox_finder.infernal` precedent (parser pure, shell-out behind a probe).

    ``rscape_version`` defaults to ``None`` — text alone is no evidence of which
    binary produced it. :func:`run_rscape` supplies the probed value.
    """
    helices: list[Helix] = []
    for line in text.splitlines():
        match = _HELIX_RE.match(line.strip())
        if match is None:
            continue
        n_basepairs = int(match.group("nbp"))
        n_covarying = int(match.group("nbp_cov"))
        if n_covarying > n_basepairs:
            raise CovariationBackendError(
                f"helix reports nbp_cov={n_covarying} > nbp={n_basepairs}: {line.strip()!r}"
            )
        helices.append(
            Helix(
                left_start=int(match.group("i")),
                left_end=int(match.group("j")),
                right_start=int(match.group("k")),
                right_end=int(match.group("l")),
                n_basepairs=n_basepairs,
                n_covarying=n_covarying,
            )
        )
    return CovariationResult(
        helices=tuple(helices), evalue=float(evalue), rscape_version=rscape_version
    )


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_rscape(
    alignment: Path,
    outdir: Path,
    *,
    evalue: float = DEFAULT_EVALUE,
    outname: str = "rscape",
    timeout_s: float = DEFAULT_RSCAPE_TIMEOUT_S,
) -> CovariationResult:
    """Run R-scape on a Stockholm alignment and parse its helix-level output.

    The alignment must carry an ``#=GC SS_cons`` line — R-scape aggregates
    covariation *per helix* off that consensus structure, and without it no
    ``.helixcov`` records are emitted.
    """
    if not _binary_on_path():
        raise CovariationBackendError(
            f"{RSCAPE_BINARY} is not on PATH; run inside the pinned tbox-rscape env "
            f"(envs/rscape.yml, rscape={PINNED_RSCAPE_VERSION})"
        )
    # Probed before the run, and recorded on the result, so no report can stamp a
    # version the binary that produced it did not report.
    observed_version = rscape_version()
    if observed_version != PINNED_RSCAPE_VERSION:
        raise CovariationBackendError(
            f"{RSCAPE_BINARY} reports {observed_version!r} but envs/rscape.yml pins "
            f"{PINNED_RSCAPE_VERSION!r}; v2.0.5 recalculated the covariation power "
            "curves, so verdicts from another build are not comparable (ADR-0002 A11)"
        )
    alignment = Path(alignment)
    if not alignment.is_file():
        raise CovariationBackendError(f"alignment not found: {alignment}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    helixcov = outdir / f"{outname}.helixcov"
    # Refuse a pre-existing output rather than risk parsing it. If R-scape exits 0
    # without writing (it does exactly that on an alignment with no SS_cons), a
    # leftover file from an earlier run in the same outdir would be read as this
    # candidate's verdict — a stale `passed` on a candidate never scored, which is
    # the fail-open direction the whole module is built to avoid.
    if helixcov.exists():
        raise CovariationBackendError(
            f"refusing to reuse a pre-existing helix-level output: {helixcov}; "
            "pass a fresh outdir (covariation_verdict uses a temporary one)"
        )

    cmd = [
        RSCAPE_BINARY,
        "--nofigures",
        "-E",
        str(evalue),
        "--outdir",
        str(outdir),
        "--outname",
        outname,
        str(alignment),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise CovariationBackendError(
            f"{RSCAPE_BINARY} timed out after {timeout_s}s: {' '.join(cmd)}"
        ) from exc
    except OSError as exc:
        raise CovariationBackendError(f"could not execute {RSCAPE_BINARY}: {exc}") from exc
    if proc.returncode != 0:
        raise CovariationBackendError(
            f"{RSCAPE_BINARY} failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )

    if not helixcov.is_file():
        raise CovariationBackendError(
            f"{RSCAPE_BINARY} produced no helix-level output at {helixcov}; "
            "does the alignment carry an #=GC SS_cons line?"
        )
    return parse_helixcov(
        helixcov.read_text(encoding="utf-8"),
        evalue=evalue,
        rscape_version=observed_version,
    )


def covariation_verdict(
    alignment: Path,
    criterion: AnyHelixCriterion,
    *,
    evalue: float = DEFAULT_EVALUE,
    timeout_s: float = DEFAULT_RSCAPE_TIMEOUT_S,
) -> tuple[str, dict[str, Any]]:
    """Score one candidate's alignment → ``(status, report)``.

    ``status`` is ``STATUS_PASSED`` or ``STATUS_FAILED``, ready to drop into
    :class:`~tbox_finder.mining.spare_rule.SpareRuleEvidence`'s
    ``any_helix_rscape`` field. It is never ``STATUS_UNAVAILABLE``: reaching this
    function means the backend ran.

    ``criterion`` is positional and required — see the module docstring.
    """
    if not isinstance(criterion, AnyHelixCriterion):
        raise CovariationBackendError(
            f"criterion must be an AnyHelixCriterion, got {type(criterion).__name__}"
        )
    with tempfile.TemporaryDirectory(prefix="tbox-rscape-") as tmp:
        result = run_rscape(Path(alignment), Path(tmp), evalue=evalue, timeout_s=timeout_s)
    return result.status(criterion), result.as_report(criterion)
