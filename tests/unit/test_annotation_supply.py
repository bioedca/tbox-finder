"""Unit tier — P3-15′-c-i, the (c) annotation-supply measurement.

Four things are guarded here, in descending order of how expensive they would be to get
wrong:

1. **``unknown`` must never be read as ``unannotated``.** The route rule spends a cluster
   acquisition and possibly an ADR-0002 env amendment on the answer. A run that cannot reach
   NCBI must refuse, not report "nothing is annotated" and certify the gene-caller route —
   the [[matched-control-before-certifying]] shape, where the *absence of power* reads as a
   signal.

2. **The control must be able to fail.** A positive that must resolve-and-be-annotated **and**
   a negative that must not resolve. Either leg alone is satisfied by a degenerate probe (one
   that resolves everything, or one that resolves nothing), so both are asserted, and each is
   broken *alone* — an all-TRUE fixture cannot test a conjunction.

3. **A cost knob must not certify.** ``--limit`` truncates the sweep; the derivation refuses
   on an incomplete sweep rather than reading a 20-host answer as the 660-host answer.

4. **The host accession is not the manifest's ``accession`` field.** That field is
   ``<assembly>:c<contig>``; reading it whole inflates the candidate-host count from 76 to 228
   and would size an acquisition against the wrong denominator.

Bare-CI tier: pure stdlib, no network, no numpy/pandas/torch. Every transport test drives a
fake opener.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from tbox_finder.mining import annotation_supply as asup
from tbox_finder.mining.annotation_supply import (
    GENOMIC_GBFF_SUFFIX,
    GFF_SUFFIX,
    PROTEIN_FAA_SUFFIX,
    ROUTE_MIXED,
    ROUTE_NCBI_GFF,
    ROUTE_PRODIGAL,
    ROUTE_REFUSED,
    STATUS_ANNOTATED,
    STATUS_UNANNOTATED,
    STATUS_UNKNOWN,
    AnnotationSupplyError,
    accession_prefix_tally,
    assembly_basename,
    assembly_dir_url,
    candidate_host_accessions,
    classify_assembly,
    control_is_powered,
    derive_acquisition_route,
    fetch_file_manifest,
    load_source_urls,
    measure_annotation_supply,
    parse_md5_manifest,
    probe_assembly,
    run_control,
    sibling_url,
)


class _NetworkAccess(BaseException):
    """Raised when a test reaches the live network.

    Deliberately a **BaseException**, not an ``Exception``. The transport helpers wrap their
    calls in ``except Exception`` and convert a failure into a retry and then ``None``, so a
    guard that raised ``AssertionError`` would be *swallowed by the very code it is guarding* —
    measured: the suite still went to the network and still reported 89 passed, in 22.66 s
    instead of 0.17 s. A guard whose signal the subject can absorb is not a guard.
    """


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this file's "no network" claim enforced rather than asserted.

    Three end-to-end tests here *were* reaching live NCBI: they passed ``opener=`` but not
    ``header=``, and ``measure_sizes`` defaults on, so the size pass fell through to the real
    ``_urlhead_length``. Measured before fixing: **30 real requests, 45 s**. Nothing went red —
    a failed HEAD yields ``None`` and no assertion reads a size — so the cost was hidden
    runtime and offline flakiness behind a docstring that said the opposite. A prose claim that
    nothing enforces is exactly the shape this suite exists to refuse.
    """

    def _forbidden(url: str) -> object:
        raise _NetworkAccess(f"test reached the live network: {url}")

    monkeypatch.setattr(asup, "_urlhead_length", _forbidden)
    monkeypatch.setattr(asup, "_urlopen_text", _forbidden)


BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/296/795/GCA_000296795.1_ASM29679v1"
FNA_URL = f"{BASE}/GCA_000296795.1_ASM29679v1_genomic.fna.gz"
BASENAME = "GCA_000296795.1_ASM29679v1"
GFF_URL = f"{BASE}/{BASENAME}_genomic.gff.gz"
# The control positive's URL must belong to the control positive's accession — probe_assembly
# now refuses a mis-joined (accession, source_url) pair.
CTRL_BASENAME = f"{asup.CONTROL_POSITIVE_ACCESSION}_ASM718v1"
CTRL_BASE = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/007/185/{CTRL_BASENAME}"
CTRL_FNA_URL = f"{CTRL_BASE}/{CTRL_BASENAME}_genomic.fna.gz"
MD5 = "0" * 32


def _manifest_text(names: list[str]) -> str:
    return "\n".join(f"{MD5}  ./{n}" for n in names) + "\n"


def _annotated_names(basename: str = BASENAME) -> list[str]:
    return [
        f"{basename}_genomic.fna.gz",
        f"{basename}{GFF_SUFFIX}",
        f"{basename}{PROTEIN_FAA_SUFFIX}",
        f"{basename}{GENOMIC_GBFF_SUFFIX}",
        "annotation_hashes.txt",
    ]


def _unannotated_names(basename: str = BASENAME) -> list[str]:
    # A real unannotated GenBank assembly: a gbff IS served, a gff is NOT.
    return [
        f"{basename}_genomic.fna.gz",
        f"{basename}{GENOMIC_GBFF_SUFFIX}",
        f"{basename}_wgsmaster.gbff.gz",
        "annotation_hashes.txt",
    ]


# --------------------------------------------------------------------------- #
# Pure URL algebra
# --------------------------------------------------------------------------- #


def test_sibling_url_swaps_only_the_suffix() -> None:
    assert sibling_url(FNA_URL, GFF_SUFFIX) == f"{BASE}/{BASENAME}{GFF_SUFFIX}"
    assert sibling_url(FNA_URL, PROTEIN_FAA_SUFFIX) == f"{BASE}/{BASENAME}{PROTEIN_FAA_SUFFIX}"


@pytest.mark.parametrize(
    "bad",
    [
        f"{BASE}/{BASENAME}_genomic.fna",  # not gzipped
        f"{BASE}/{BASENAME}_protein.faa.gz",  # already a sibling
        "",
    ],
)
def test_sibling_url_refuses_a_url_it_cannot_anchor(bad: str) -> None:
    """A plausible-looking wrong URL is worse than a refusal: it 404s on every host and the
    sweep reports a uniformly unannotated corpus."""
    with pytest.raises(AnnotationSupplyError):
        sibling_url(bad, GFF_SUFFIX)


@pytest.mark.parametrize(
    "hostile",
    [
        "file:///etc/passwd/x_genomic.fna.gz",
        "ftp://ftp.ncbi.nlm.nih.gov/x/x_genomic.fna.gz",
        "http://ftp.ncbi.nlm.nih.gov/x/x_genomic.fna.gz",
        "https://evil.test/x/x_genomic.fna.gz",
        "https://evil.test/?x=ftp.ncbi.nlm.nih.gov/x_genomic.fna.gz",
        "https://ftp.ncbi.nlm.nih.gov.evil.test/x/x_genomic.fna.gz",
    ],
)
def test_url_helpers_refuse_a_scheme_or_host_outside_the_allowlist(hostile: str) -> None:
    """Both builders feed ``urlopen``. A ``file://`` or third-party ``source_url`` would become
    a local-file read or an off-host request, so the allowlist is enforced on **scheme and
    host** — not on a substring, which ``https://evil.test/?x=ftp.ncbi.nlm.nih.gov`` defeats,
    and not on a prefix, which ``ftp.ncbi.nlm.nih.gov.evil.test`` defeats.
    """
    with pytest.raises(AnnotationSupplyError, match="allowed NCBI URL"):
        sibling_url(hostile, GFF_SUFFIX)
    with pytest.raises(AnnotationSupplyError, match="allowed NCBI URL"):
        assembly_dir_url(hostile)


def test_transport_refuses_a_disallowed_url_it_was_handed_directly() -> None:
    """``fetch_file_manifest`` and ``head_content_length`` are public and take a URL, so
    guarding only the builders leaves the actual ``urlopen`` boundary open."""
    with pytest.raises(AnnotationSupplyError, match="allowed NCBI URL"):
        fetch_file_manifest("file:///etc", opener=lambda _u: "")
    with pytest.raises(AnnotationSupplyError, match="allowed NCBI URL"):
        asup.head_content_length("https://evil.test/a.gz", header=lambda _u: 1)


def test_redirect_handler_revalidates_every_hop() -> None:
    """``urlopen`` follows redirects, so validating only the URL handed to it leaves the guard
    one hop deep — a 302 moves the request off-host *after* the check passed."""
    handler = asup._AllowlistRedirectHandler()
    with pytest.raises(AnnotationSupplyError, match="allowed NCBI URL"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://evil.test/x.gz")


def test_the_transport_opener_installs_the_redirect_guard() -> None:
    """A second opener, or a bare ``urlopen``, would bypass the handler entirely — so the
    guard's presence on the shared opener is what makes it load-bearing."""
    assert any(
        isinstance(h, asup._AllowlistRedirectHandler) for h in asup._OPENER.handlers
    ), "the shared opener does not carry the allowlist redirect handler"


def test_an_allowlist_violation_is_not_retried_into_unknown() -> None:
    """A disallowed URL is a refusal, not a transient network failure. Swallowing it would
    retry four times and then report the host as ``unknown``, which reads as 'NCBI was flaky'
    rather than 'this run was pointed somewhere it must never go'."""
    calls: list[str] = []

    def _bad(url: str) -> str:
        calls.append(url)
        raise AnnotationSupplyError("not an allowed NCBI URL: pretend-redirect")

    with pytest.raises(AnnotationSupplyError, match="allowed NCBI URL"):
        fetch_file_manifest(BASE, opener=_bad, sleep=lambda _s: None)
    assert len(calls) == 1, "an allowlist violation must not be retried"


def test_assembly_dir_and_basename() -> None:
    assert assembly_dir_url(FNA_URL) == BASE
    assert assembly_basename(FNA_URL) == BASENAME


def test_assembly_basename_refuses_a_foreign_leaf() -> None:
    with pytest.raises(AnnotationSupplyError):
        assembly_basename(f"{BASE}/md5checksums.txt")


# --------------------------------------------------------------------------- #
# Manifest parsing
# --------------------------------------------------------------------------- #


def test_parse_md5_manifest_reads_names_and_checksums() -> None:
    parsed = parse_md5_manifest(_manifest_text(_annotated_names()))
    assert f"{BASENAME}{GFF_SUFFIX}" in parsed
    assert parsed[f"{BASENAME}{GFF_SUFFIX}"] == MD5


def test_parse_md5_manifest_refuses_a_malformed_line() -> None:
    """A partially-parsed manifest that happens to drop the GFF line is indistinguishable
    from an unannotated assembly — so a bad line raises rather than being skipped."""
    text = _manifest_text(_annotated_names()) + "this is not a checksum line\n"
    with pytest.raises(AnnotationSupplyError):
        parse_md5_manifest(text)


def test_parse_md5_manifest_refuses_an_empty_body() -> None:
    with pytest.raises(AnnotationSupplyError):
        parse_md5_manifest("   \n\n")


def test_parse_md5_manifest_tolerates_blank_lines() -> None:
    parsed = parse_md5_manifest("\n" + _manifest_text(_annotated_names()) + "\n")
    assert len(parsed) == len(_annotated_names())


# --------------------------------------------------------------------------- #
# Classification — the gbff trap
# --------------------------------------------------------------------------- #


def test_classify_assembly_annotated_iff_gff_present() -> None:
    assert classify_assembly(_annotated_names(), BASENAME) == STATUS_ANNOTATED
    assert classify_assembly(_unannotated_names(), BASENAME) == STATUS_UNANNOTATED


def test_classify_assembly_does_not_key_on_gbff() -> None:
    """NCBI serves ``_genomic.gbff.gz`` for unannotated assemblies too (measured: the real
    GCA_000372225.1 directory carries a gbff and a wgsmaster gbff and **no** gff). Keying on
    it would call every GenBank assembly annotated and the acquisition would download flat
    files with no CDS features."""
    names = _unannotated_names()
    assert any(n.endswith(GENOMIC_GBFF_SUFFIX) for n in names)
    assert classify_assembly(names, BASENAME) == STATUS_UNANNOTATED


def test_classify_assembly_requires_this_assemblys_gff() -> None:
    """A GFF belonging to a *different* basename in the same listing must not count."""
    foreign = [f"{BASENAME}_genomic.fna.gz", f"GCA_999999999.9_OTHER{GFF_SUFFIX}"]
    assert classify_assembly(foreign, BASENAME) == STATUS_UNANNOTATED


def test_classify_assembly_never_returns_unknown() -> None:
    """``unknown`` is a transport state, not a classification state."""
    assert classify_assembly([], BASENAME) in (STATUS_ANNOTATED, STATUS_UNANNOTATED)


# --------------------------------------------------------------------------- #
# Accession handling
# --------------------------------------------------------------------------- #


def test_accession_prefix_tally() -> None:
    assert accession_prefix_tally(["GCA_000296795.1", "GCF_000007185.1", "GCA_000372225.1"]) == {
        "GCA": 2,
        "GCF": 1,
    }


def test_accession_prefix_tally_refuses_a_malformed_accession() -> None:
    """A dropped accession understates the denominator every reported fraction is read
    against."""
    with pytest.raises(AnnotationSupplyError):
        accession_prefix_tally(["GCA_000296795.1", "GCA_000296795.1:c10"])


def test_candidate_host_accessions_strips_the_contig_index(tmp_path: Path) -> None:
    """The manifest's ``accession`` is ``<assembly>:c<contig>``. Reading it whole turns 2
    assemblies into 3 hosts — at real scale, 76 into 228."""
    p = tmp_path / "fp.json"
    p.write_text(
        json.dumps(
            {
                "candidates": [
                    {"accession": "GCA_001873345.1:c10", "candidate_id": "x"},
                    {"accession": "GCA_001873345.1:c11", "candidate_id": "y"},
                    {"accession": "GCF_000007185.1:c1", "candidate_id": "z"},
                ]
            }
        )
    )
    assert candidate_host_accessions(p) == ["GCA_001873345.1", "GCF_000007185.1"]


def test_candidate_host_accessions_refuses_a_malformed_host(tmp_path: Path) -> None:
    p = tmp_path / "fp.json"
    p.write_text(json.dumps({"candidates": [{"accession": "not-an-accession:c1"}]}))
    with pytest.raises(AnnotationSupplyError):
        candidate_host_accessions(p)


def test_load_source_urls_skips_non_ok_and_refuses_an_empty_result(tmp_path: Path) -> None:
    good = {"assembly_accession": "GCA_000296795.1", "status": "ok", "source_url": FNA_URL}
    bad = {"assembly_accession": "GCA_000372225.1", "status": "no_ftp_path", "source_url": ""}
    p = tmp_path / "rep.json"
    p.write_text(json.dumps({"per_genome": [good, bad]}))
    urls = load_source_urls(p)
    assert urls == {"GCA_000296795.1": FNA_URL}

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"per_genome": [bad]}))
    with pytest.raises(AnnotationSupplyError):
        load_source_urls(empty)


def test_load_source_urls_refuses_a_report_with_no_per_genome(tmp_path: Path) -> None:
    p = tmp_path / "rep.json"
    p.write_text(json.dumps({"per_genome": []}))
    with pytest.raises(AnnotationSupplyError):
        load_source_urls(p)


@pytest.mark.parametrize("payload", ["[]", '"a string"', "null", "42"])
def test_load_source_urls_refuses_a_non_object_root(tmp_path: Path, payload: str) -> None:
    """A shape assumed before it is validated escapes as an ``AttributeError`` traceback and
    exit 1, not as the documented refusal and exit 3 — ``main`` catches only
    ``AnnotationSupplyError``."""
    p = tmp_path / "rep.json"
    p.write_text(payload)
    with pytest.raises(AnnotationSupplyError):
        load_source_urls(p)


def test_candidate_host_accessions_refuses_a_mapping_without_candidates(tmp_path: Path) -> None:
    """``payload["candidates"]`` on an object that lacks the key is a ``KeyError`` — exit 1 with
    a traceback, not exit 3 with the refusal. ``load_source_urls`` already uses ``.get`` for
    exactly this reason."""
    p = tmp_path / "fp.json"
    p.write_text(json.dumps({"n_candidates": 0, "schema_version": "1.0"}))
    with pytest.raises(AnnotationSupplyError):
        candidate_host_accessions(p)


def test_load_source_urls_refuses_a_conflicting_duplicate_accession(tmp_path: Path) -> None:
    """Two ``ok`` rows for one accession with different URLs: the last silently wins, and the
    sweep then reports whichever directory survived as that accession's evidence with no record
    a conflict existed. ``probe_assembly``'s basename check cannot catch it — both rows share
    the accession prefix and differ only in the assembly-name part."""
    rows = [
        {"assembly_accession": "GCA_000296795.1", "status": "ok", "source_url": FNA_URL},
        {
            "assembly_accession": "GCA_000296795.1",
            "status": "ok",
            "source_url": FNA_URL.replace("ASM29679v1", "ASM29679v2"),
        },
    ]
    p = tmp_path / "rep.json"
    p.write_text(json.dumps({"per_genome": rows}))
    with pytest.raises(AnnotationSupplyError, match="conflicting source_url"):
        load_source_urls(p)


def test_load_source_urls_tolerates_an_identical_repeat(tmp_path: Path) -> None:
    """A byte-identical repeat discards nothing, so it is not a conflict."""
    row = {"assembly_accession": "GCA_000296795.1", "status": "ok", "source_url": FNA_URL}
    p = tmp_path / "rep.json"
    p.write_text(json.dumps({"per_genome": [row, dict(row)]}))
    assert load_source_urls(p) == {"GCA_000296795.1": FNA_URL}


@pytest.mark.parametrize("row", ["not-a-mapping", 7, None, {"score": 1.0}])
def test_candidate_host_accessions_refuses_a_row_with_no_accession(
    tmp_path: Path, row: object
) -> None:
    """Same shape one file over: ``row["accession"]`` raises ``TypeError``/``KeyError``, which
    escapes the refusal contract."""
    p = tmp_path / "fp.json"
    p.write_text(json.dumps({"candidates": [row]}))
    with pytest.raises(AnnotationSupplyError):
        candidate_host_accessions(p)


# --------------------------------------------------------------------------- #
# Transport — driven by a fake opener, never the network
# --------------------------------------------------------------------------- #


class _Opener:
    """A scripted ``url -> body`` opener. Any URL not in ``bodies`` raises ``code``."""

    def __init__(self, bodies: dict[str, str], code: int = 404) -> None:
        self.bodies = bodies
        self.code = code
        self.calls: list[str] = []

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        if url in self.bodies:
            return self.bodies[url]
        raise urllib.error.HTTPError(url, self.code, "no", None, None)  # type: ignore[arg-type]


def test_fetch_file_manifest_parses_a_served_manifest() -> None:
    url = f"{BASE}/{asup.MD5_MANIFEST_NAME}"
    op = _Opener({url: _manifest_text(_annotated_names())})
    manifest, note = fetch_file_manifest(BASE, opener=op)
    assert manifest is not None and note == ""
    assert f"{BASENAME}{GFF_SUFFIX}" in manifest


def test_fetch_file_manifest_returns_none_on_a_permanent_404_without_retrying() -> None:
    op = _Opener({})
    manifest, note = fetch_file_manifest(BASE, opener=op)
    assert manifest is None and "404" in note
    assert len(op.calls) == 1, "a permanent 4xx must not be retried"


def test_fetch_file_manifest_retries_a_transient_failure() -> None:
    op = _Opener({}, code=429)
    manifest, note = fetch_file_manifest(BASE, retries=3, opener=op, sleep=lambda _s: None)
    assert manifest is None and note
    assert len(op.calls) == 3, "a 429 is transient and must be retried"


def test_fetch_file_manifest_backs_off_between_transient_retries() -> None:
    """The rate limiter is not backoff — it meters the steady state at 3 req/s, so without an
    explicit wait a 429 would be re-sent four times inside ~1.3 s, which is exactly what the
    limiter exists to prevent. ``pilot_fetch`` uses ``1.5 * (attempt + 1)``; so does this."""
    slept: list[float] = []
    op = _Opener({}, code=429)
    fetch_file_manifest(BASE, retries=3, opener=op, sleep=slept.append)
    assert slept == [1.5, 3.0], f"backoff schedule was {slept}"


def test_fetch_file_manifest_does_not_back_off_on_a_permanent_failure() -> None:
    """A 404 is not retried, so it must not sleep either — 660 hosts × a pointless wait is the
    difference between a two-minute sweep and a twenty-minute one."""
    slept: list[float] = []
    fetch_file_manifest(BASE, retries=3, opener=_Opener({}), sleep=slept.append)
    assert slept == []


def test_head_content_length_backs_off_between_transient_retries() -> None:
    slept: list[float] = []

    def _busy(_url: str) -> int:
        raise urllib.error.HTTPError(_url, 429, "busy", None, None)  # type: ignore[arg-type]

    assert asup.head_content_length(GFF_URL, header=_busy, sleep=slept.append) is None
    assert slept == [1.5, 3.0], f"backoff schedule was {slept}"


def test_fetch_file_manifest_returns_none_on_an_unparseable_body() -> None:
    """A 200 carrying junk is *unknown*, not *unannotated*."""
    url = f"{BASE}/{asup.MD5_MANIFEST_NAME}"
    op = _Opener({url: "<html>503 backend error</html>"})
    manifest, note = fetch_file_manifest(BASE, opener=op)
    assert manifest is None and note.startswith("parse:")


def test_probe_assembly_unreachable_is_unknown_not_unannotated() -> None:
    """The single most expensive confusion this module can make."""
    op = _Opener({})
    row = probe_assembly("GCA_000296795.1", FNA_URL, opener=op)
    assert row["status"] == STATUS_UNKNOWN
    assert row["status"] != STATUS_UNANNOTATED
    assert row["gff_url"] == "" and row["n_files"] == 0


def test_probe_assembly_refuses_a_url_belonging_to_another_assembly() -> None:
    """The row is *labelled* with ``accession`` but *classified* from whatever directory
    ``source_url`` names. A mis-joined pair therefore produces well-formed evidence filed under
    the wrong assembly — it never shows up as ``unknown``, and it moves the very counts the
    route is derived from ([[namespace-mismatch-invisible-noop]])."""
    with pytest.raises(AnnotationSupplyError, match="does not belong to accession"):
        probe_assembly("GCF_000007185.1", FNA_URL, opener=_Opener({}))


def test_probe_assembly_annotated_carries_the_gff_url_and_md5() -> None:
    op = _Opener({f"{BASE}/{asup.MD5_MANIFEST_NAME}": _manifest_text(_annotated_names())})
    row = probe_assembly("GCA_000296795.1", FNA_URL, opener=op)
    assert row["status"] == STATUS_ANNOTATED
    assert row["gff_url"] == f"{BASE}/{BASENAME}{GFF_SUFFIX}"
    assert row["gff_md5"] == MD5
    assert row["faa_present"] is True


def test_probe_assembly_unannotated_publishes_no_gff_url() -> None:
    op = _Opener({f"{BASE}/{asup.MD5_MANIFEST_NAME}": _manifest_text(_unannotated_names())})
    row = probe_assembly("GCA_000296795.1", FNA_URL, opener=op)
    assert row["status"] == STATUS_UNANNOTATED
    assert row["gff_url"] == "", "an unannotated host must not publish a downloadable URL"


# --------------------------------------------------------------------------- #
# The control
# --------------------------------------------------------------------------- #


def _control_urls() -> dict[str, str]:
    return {asup.CONTROL_POSITIVE_ACCESSION: CTRL_FNA_URL}


def test_run_control_is_powered_when_both_legs_behave() -> None:
    op = _Opener(
        {f"{CTRL_BASE}/{asup.MD5_MANIFEST_NAME}": _manifest_text(_annotated_names(CTRL_BASENAME))}
    )
    control = run_control(_control_urls(), opener=op)
    assert control["positive_status"] == STATUS_ANNOTATED
    assert control["negative"]["resolved"] is False
    assert control_is_powered(control) is True


def test_control_is_unpowered_when_the_positive_leg_alone_breaks() -> None:
    """Break the positive ALONE: a probe that reaches nothing still leaves the negative leg
    satisfied, so the negative leg cannot carry the control by itself."""
    op = _Opener({})
    control = run_control(_control_urls(), opener=op)
    assert control["negative"]["resolved"] is False, "the negative leg still 'passes'"
    assert control["positive_status"] == STATUS_UNKNOWN
    assert control_is_powered(control) is False


def test_control_is_unpowered_when_the_negative_leg_alone_breaks() -> None:
    """Break the negative ALONE: an opener that resolves *every* URL keeps the positive leg
    satisfied, so the positive leg cannot carry the control by itself."""

    class _AlwaysOpen:
        def __call__(self, url: str) -> str:
            return _manifest_text(_annotated_names(CTRL_BASENAME))

    control = run_control(_control_urls(), opener=_AlwaysOpen())
    assert control["positive_status"] == STATUS_ANNOTATED, "the positive leg still 'passes'"
    assert control["negative"]["resolved"] is True
    assert control_is_powered(control) is False


def test_control_is_unpowered_when_the_positive_resolves_but_is_unannotated() -> None:
    """The exact degenerate run the control exists for: everything parses, nothing has a GFF.
    Without the positive leg this reports 660 ``unannotated`` and certifies the gene caller."""
    op = _Opener(
        {f"{CTRL_BASE}/{asup.MD5_MANIFEST_NAME}": _manifest_text(_unannotated_names(CTRL_BASENAME))}
    )
    control = run_control(_control_urls(), opener=op)
    assert control["positive_status"] == STATUS_UNANNOTATED
    assert control_is_powered(control) is False


def test_run_control_forwards_the_injected_clock_to_both_legs() -> None:
    """A parameter a function accepts and never reads is a parameter nothing tests. Both
    control legs retry, so a caller injecting a fake clock would still block on the real
    ``time.sleep`` — 9 s per leg on a transient failure — and no existing test noticed, because
    none of them made the control retry ([[pinned-constant-that-nothing-reads]])."""
    slept: list[float] = []
    control = run_control(_control_urls(), opener=_Opener({}, code=429), sleep=slept.append)
    assert control["positive_status"] == STATUS_UNKNOWN
    # Both legs retried and every wait went to the injected clock. Four attempts ⇒ three waits
    # per leg, 1.5 + 3.0 + 4.5 = 9.0 s of real time each if the clock is not forwarded.
    assert slept == [1.5, 3.0, 4.5, 1.5, 3.0, 4.5], f"waits were {slept}"
    assert sum(slept) == 18.0


def test_run_control_refuses_when_the_positive_accession_is_absent() -> None:
    """The message must name the control, not merely be *an* ``AnnotationSupplyError``.

    Found by sabotage: deleting the guard still raised — ``assembly_basename("")`` throws the
    same exception type one frame down — so a bare ``pytest.raises`` passed while the named
    refusal was gone. A ``raises`` test that does not pin *which* refusal fired is satisfied by
    any crash ([[raises-test-needs-a-positive-control]]).
    """
    with pytest.raises(AnnotationSupplyError, match=asup.CONTROL_POSITIVE_ACCESSION):
        run_control({}, opener=_Opener({}))


# --------------------------------------------------------------------------- #
# The pre-registered route rule
# --------------------------------------------------------------------------- #


def _report(
    *,
    annotated: int,
    unannotated: int = 0,
    unknown_admissible: int = 0,
    powered: bool = True,
    complete: bool = True,
) -> dict[str, object]:
    n_cand = annotated + unannotated
    return {
        "sweep_complete": complete,
        "n_candidate_hosts": n_cand,
        "n_candidate_hosts_probed": n_cand,
        "control": {
            "positive_status": STATUS_ANNOTATED if powered else STATUS_UNANNOTATED,
            "negative": {"resolved": False},
        },
        "admissible_status_counts": {
            STATUS_ANNOTATED: annotated,
            STATUS_UNANNOTATED: unannotated,
            STATUS_UNKNOWN: unknown_admissible,
        },
        "candidate_host_status_counts": {
            STATUS_ANNOTATED: annotated,
            STATUS_UNANNOTATED: unannotated,
            STATUS_UNKNOWN: 0,
        },
    }


def test_route_ncbi_gff_when_every_candidate_host_is_annotated() -> None:
    out = derive_acquisition_route(_report(annotated=76))
    assert out["route"] == ROUTE_NCBI_GFF
    assert out["adr_0002_amendment_required"] is False


def test_route_prodigal_when_no_candidate_host_is_annotated() -> None:
    out = derive_acquisition_route(_report(annotated=0, unannotated=76))
    assert out["route"] == ROUTE_PRODIGAL
    assert out["adr_0002_amendment_required"] is True


def test_route_mixed_when_some_are_annotated() -> None:
    out = derive_acquisition_route(_report(annotated=60, unannotated=16))
    assert out["route"] == ROUTE_MIXED
    assert out["n_candidate_hosts_annotated"] == 60 and out["n_candidate_hosts"] == 76


def test_route_refuses_while_any_admissible_host_is_unknown() -> None:
    """The load-bearing refusal: an unresolved host is not an unannotated host."""
    out = derive_acquisition_route(_report(annotated=76, unknown_admissible=1))
    assert out["route"] == ROUTE_REFUSED
    assert any("unknown is not unannotated" in r for r in out["reasons"])


def test_route_refuses_on_an_unpowered_control() -> None:
    out = derive_acquisition_route(_report(annotated=0, unannotated=76, powered=False))
    assert out["route"] == ROUTE_REFUSED
    assert "prodigal" not in out["route"]


def test_route_refuses_on_an_incomplete_sweep() -> None:
    """``--limit`` is a cost knob; a cost knob may not certify a route."""
    out = derive_acquisition_route(_report(annotated=76, complete=False))
    assert out["route"] == ROUTE_REFUSED
    assert any("sweep incomplete" in r for r in out["reasons"])


def test_route_refuses_when_a_candidate_host_was_not_probed() -> None:
    """``measure_annotation_supply`` drops a candidate host absent from the probed set, and
    ``sweep_complete`` only compares the *admissible* denominator — so without this clause a
    route is certified from an understated candidate denominator. The guard that already
    exists for the admissible set had no counterpart for the candidate set."""
    rep = _report(annotated=48, unannotated=28)
    rep["n_candidate_hosts"] = 76
    rep["n_candidate_hosts_probed"] = 70
    out = derive_acquisition_route(rep)
    assert out["route"] == ROUTE_REFUSED
    # the reason must state the measured pair, not a count derived from a sentinel
    assert any("probed (70) != n_candidate_hosts (76)" in r for r in out["reasons"])
    assert any("an unprobed host is not an unannotated host" in r for r in out["reasons"])


def test_route_refuses_a_report_that_omits_the_candidate_coverage_field() -> None:
    """A missing clause input must refuse, not be read as satisfied — an older report has no
    business certifying against a rule it was not measured under."""
    rep = _report(annotated=48, unannotated=28)
    del rep["n_candidate_hosts_probed"]
    out = derive_acquisition_route(rep)
    assert out["route"] == ROUTE_REFUSED
    # …and it must say the field is missing, not invent "77 hosts were not probed" from a
    # -1 sentinel. A refusal reason is read by whoever has to act on it.
    assert any("carries no n_candidate_hosts_probed" in r for r in out["reasons"])
    assert not any("!= n_candidate_hosts" in r for r in out["reasons"])


def test_route_refuses_on_an_unresolved_candidate_host() -> None:
    """The gate checked only the *admissible* unknown count. A report can declare 76 candidate
    hosts, 76 probed, zero admissible-unknown, and still carry an unknown candidate host — and
    the route came back ``mixed`` with no refusal. In a report this module writes that cannot
    happen (candidates are a subset of the probed rows), which is exactly the point: a gate that
    leans on an invariant enforced somewhere else is not a gate
    ([[gate-clauses-need-re-derivation]])."""
    rep = _report(annotated=48, unannotated=27)
    rep["candidate_host_status_counts"][STATUS_UNKNOWN] = 1
    rep["n_candidate_hosts"] = 76
    rep["n_candidate_hosts_probed"] = 76
    out = derive_acquisition_route(rep)
    assert out["route"] == ROUTE_REFUSED
    assert any("candidate-carrying host(s) unresolved" in r for r in out["reasons"])


def test_route_refuses_when_the_candidate_counts_do_not_sum_to_the_denominator() -> None:
    """``n_total`` is the base the annotated fraction is read against. If it disagrees with
    ``n_candidate_hosts``, "48 of 76 annotated" is being computed against a different 76."""
    rep = _report(annotated=48, unannotated=20)
    rep["n_candidate_hosts"] = 76
    rep["n_candidate_hosts_probed"] = 76
    out = derive_acquisition_route(rep)
    assert out["route"] == ROUTE_REFUSED
    assert any("wrong denominator" in r for r in out["reasons"])


def test_route_refuses_on_zero_candidate_hosts() -> None:
    out = derive_acquisition_route(_report(annotated=0))
    assert out["route"] == ROUTE_REFUSED


def test_route_refuses_a_report_with_no_counts() -> None:
    assert derive_acquisition_route({})["route"] == ROUTE_REFUSED


# --------------------------------------------------------------------------- #
# End-to-end, offline
# --------------------------------------------------------------------------- #


def _corpus(n_annotated: int, n_unannotated: int) -> tuple[dict[str, str], dict[str, str]]:
    """Build ``(source_urls, bodies)`` for a synthetic corpus + the control positive."""
    urls: dict[str, str] = {asup.CONTROL_POSITIVE_ACCESSION: CTRL_FNA_URL}
    bodies: dict[str, str] = {
        f"{CTRL_BASE}/{asup.MD5_MANIFEST_NAME}": _manifest_text(_annotated_names(CTRL_BASENAME))
    }
    for i in range(n_annotated + n_unannotated):
        acc = f"GCA_{i:09d}.1"
        base = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/x/{acc}_ASM{i}"
        bn = f"{acc}_ASM{i}"
        urls[acc] = f"{base}/{bn}_genomic.fna.gz"
        names = _annotated_names(bn) if i < n_annotated else _unannotated_names(bn)
        bodies[f"{base}/{asup.MD5_MANIFEST_NAME}"] = _manifest_text(names)
    return urls, bodies


def test_measure_end_to_end_all_annotated_routes_to_ncbi_gff() -> None:
    urls, bodies = _corpus(5, 0)
    admissible = [a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION]
    rep = measure_annotation_supply(
        admissible=sorted(admissible),
        candidate_hosts=sorted(admissible)[:2],
        source_urls=urls,
        workers=1,
        opener=_Opener(bodies),
        header=lambda _u: 1000,
    )
    assert rep["sweep_complete"] is True
    assert rep["admissible_status_counts"][STATUS_ANNOTATED] == 5
    assert rep["candidate_host_status_counts"][STATUS_ANNOTATED] == 2
    assert rep["route"]["route"] == ROUTE_NCBI_GFF


def test_measure_end_to_end_mixed_reports_both_denominators() -> None:
    urls, bodies = _corpus(3, 2)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    # candidate hosts: one annotated + one not
    cand = [admissible[0], admissible[-1]]
    rep = measure_annotation_supply(
        admissible=admissible,
        candidate_hosts=cand,
        source_urls=urls,
        workers=1,
        opener=_Opener(bodies),
        header=lambda _u: 1000,
    )
    assert rep["admissible_status_counts"] == {
        STATUS_ANNOTATED: 3,
        STATUS_UNANNOTATED: 2,
        STATUS_UNKNOWN: 0,
    }
    assert rep["route"]["route"] == ROUTE_MIXED
    assert rep["n_candidate_hosts"] == 2
    # assert the split this test's fixture actually builds, not just its size: the route is
    # derived from the candidate counts, and a regression in candidate classification would
    # otherwise hide behind the full admissible sweep.
    assert rep["candidate_host_status_counts"] == {
        STATUS_ANNOTATED: 1,
        STATUS_UNANNOTATED: 1,
        STATUS_UNKNOWN: 0,
    }


def test_measure_with_limit_cannot_certify() -> None:
    urls, bodies = _corpus(5, 0)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    rep = measure_annotation_supply(
        admissible=admissible,
        candidate_hosts=admissible[:2],
        source_urls=urls,
        workers=1,
        opener=_Opener(bodies),
        header=lambda _u: 1000,
        limit=2,
    )
    assert rep["n_probed"] == 2 and rep["sweep_complete"] is False
    assert rep["route"]["route"] == ROUTE_REFUSED


def test_measure_refuses_an_admissible_host_with_no_source_url() -> None:
    urls, bodies = _corpus(2, 0)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    with pytest.raises(AnnotationSupplyError):
        measure_annotation_supply(
            admissible=[*admissible, "GCA_999999999.9"],
            candidate_hosts=admissible[:1],
            source_urls=urls,
            workers=1,
            opener=_Opener(bodies),
        )


def test_measure_refuses_a_candidate_host_outside_the_admissible_set() -> None:
    """The CLI checked this; the exported API did not — so a direct caller could get
    ``sweep_complete=True`` and a certified route derived from a *subset* of the candidate
    hosts, the silently-understated denominator the admissible-set guard already prevents."""
    urls, bodies = _corpus(2, 0)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    with pytest.raises(AnnotationSupplyError, match="not in the admissible set"):
        measure_annotation_supply(
            admissible=admissible,
            candidate_hosts=[*admissible, "GCA_888888888.8"],
            source_urls=urls,
            workers=1,
            opener=_Opener(bodies),
        )


@pytest.mark.parametrize("which", ["admissible", "candidate_hosts"])
def test_measure_refuses_duplicate_accessions(which: str) -> None:
    """The mirror of the silently-dropped accession this module already refuses: a duplicate is
    probed twice, ``by_acc`` collapses it, and ``len(rows)`` counts it — so it inflates
    ``n_probed``, the prefix tally and the status counts, i.e. every denominator the route is
    read against, while doubling that host's request cost."""
    urls, bodies = _corpus(3, 0)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    cand = admissible[:2]
    if which == "admissible":
        admissible = [*admissible, admissible[0]]
    else:
        cand = [*cand, cand[0]]
    with pytest.raises(AnnotationSupplyError, match="duplicate accessions"):
        measure_annotation_supply(
            admissible=admissible,
            candidate_hosts=cand,
            source_urls=urls,
            workers=1,
            opener=_Opener(bodies),
        )


def test_measure_end_to_end_unreachable_corpus_refuses_rather_than_certifying_prodigal() -> None:
    """The whole point. Every request fails; the report must NOT read ``prodigal_required``."""
    urls, _bodies = _corpus(3, 0)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    rep = measure_annotation_supply(
        admissible=admissible,
        candidate_hosts=admissible[:1],
        source_urls=urls,
        workers=1,
        opener=_Opener({}),
    )
    assert rep["admissible_status_counts"][STATUS_UNKNOWN] == 3
    assert rep["route"]["route"] == ROUTE_REFUSED
    assert rep["route"]["route"] != ROUTE_PRODIGAL


# --------------------------------------------------------------------------- #
# The acquisition-cost pass — cost only, never a route input
# --------------------------------------------------------------------------- #


def test_head_content_length_reads_a_size() -> None:
    assert asup.head_content_length(GFF_URL, header=lambda _u: 4096) == 4096


def test_head_content_length_returns_none_rather_than_zero_on_failure() -> None:
    """A failed HEAD read as 0 understates the download budget a SLURM ack is sized against."""

    def _boom(_url: str) -> int:
        raise urllib.error.HTTPError("u", 404, "no", None, None)  # type: ignore[arg-type]

    assert asup.head_content_length(GFF_URL, header=_boom) is None


def test_head_content_length_returns_none_on_a_missing_header() -> None:
    """``Content-Length`` absent → the transport reports -1; that is unknown, not zero."""
    assert asup.head_content_length(GFF_URL, header=lambda _u: -1) is None


def test_head_content_length_is_none_for_an_empty_url() -> None:
    assert asup.head_content_length("", header=lambda _u: 1) is None


def test_size_pass_totals_only_known_sizes_and_reports_the_rest() -> None:
    urls, bodies = _corpus(3, 0)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    seen: list[str] = []

    def _header(url: str) -> int:
        seen.append(url)
        if len(seen) == 1:
            raise urllib.error.HTTPError(url, 404, "no", None, None)  # type: ignore[arg-type]
        return 1000

    rep = measure_annotation_supply(
        admissible=admissible,
        candidate_hosts=admissible[:1],
        source_urls=urls,
        workers=1,
        opener=_Opener(bodies),
        header=_header,
    )
    assert rep["n_gff_size_known"] == 2
    assert rep["n_gff_size_unknown"] == 1
    assert rep["gff_bytes_total_known"] == 2000, "an unknown size must not be summed as 0"
    assert rep["route"]["route"] == ROUTE_NCBI_GFF, "a size failure changes no route"


def test_size_pass_can_be_skipped_without_inventing_a_total() -> None:
    urls, bodies = _corpus(2, 0)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    rep = measure_annotation_supply(
        admissible=admissible,
        candidate_hosts=admissible[:1],
        source_urls=urls,
        workers=1,
        opener=_Opener(bodies),
        measure_sizes=False,
    )
    assert rep["gff_bytes_total_known"] == 0 and rep["n_gff_size_unknown"] == 2
    assert rep["route"]["route"] == ROUTE_NCBI_GFF


def test_size_pass_never_heads_an_unannotated_host() -> None:
    """An unannotated host publishes no GFF URL; HEADing one would be a 404 storm and would
    put a size on a file that does not exist."""
    urls, bodies = _corpus(1, 2)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    seen: list[str] = []
    measure_annotation_supply(
        admissible=admissible,
        candidate_hosts=admissible[:1],
        source_urls=urls,
        workers=1,
        opener=_Opener(bodies),
        header=lambda u: (seen.append(u), 10)[1],
    )
    assert len(seen) == 1, f"HEADed {len(seen)} URLs for 1 annotated host: {seen}"
    assert seen[0].endswith(GFF_SUFFIX)


# --------------------------------------------------------------------------- #
# The committed measurement — re-derived, not trusted
# --------------------------------------------------------------------------- #

COMMITTED_REPORT = Path(__file__).resolve().parents[2] / "reports/p3/annotation_supply.json"


def _committed() -> dict[str, object]:
    return json.loads(COMMITTED_REPORT.read_text(encoding="utf-8"))


def test_committed_report_route_is_rederivable_not_transcribed() -> None:
    """The published ``route`` must be what :func:`derive_acquisition_route` says about the
    published counts. A hand-edited verdict, or a rule change that silently leaves the old
    verdict in place, fails here — a gate that is read rather than re-derived is not a gate.
    """
    rep = _committed()
    assert derive_acquisition_route(rep)["route"] == rep["route"]["route"]


def test_committed_report_counts_agree_with_its_own_rows() -> None:
    """The headline counts must be the tally of ``per_assembly``, not a separately-carried
    number that can drift from the rows a reader would audit."""
    rep = _committed()
    rows = rep["per_assembly"]
    tallied = dict.fromkeys(asup.STATUS_VALUES, 0)
    for row in rows:
        tallied[row["status"]] += 1
    assert tallied == rep["admissible_status_counts"]
    assert len(rows) == rep["n_probed"] == rep["n_admissible_requested"]
    assert rep["sweep_complete"] is True


def test_committed_report_control_was_powered() -> None:
    """Without this, every count in the file could have come from a probe with no power."""
    assert control_is_powered(_committed()["control"]) is True


def test_committed_report_publishes_a_gff_url_exactly_for_annotated_hosts() -> None:
    rep = _committed()
    for row in rep["per_assembly"]:
        has_url = bool(row["gff_url"])
        assert has_url == (row["status"] == STATUS_ANNOTATED), row["accession"]


def test_committed_report_size_total_excludes_unknowns() -> None:
    rep = _committed()
    known = [r["gff_bytes"] for r in rep["per_assembly"] if isinstance(r.get("gff_bytes"), int)]
    assert len(known) == rep["n_gff_size_known"]
    assert sum(known) == rep["gff_bytes_total_known"]


# --------------------------------------------------------------------------- #
# The module pins no scientific value
# --------------------------------------------------------------------------- #


def test_module_pins_no_scientific_value() -> None:
    """P3-15′-c-i measures *supply*. D4's 500 bp window, D6's Stem-I threshold and the Pfam/KO
    cutoffs are pinned elsewhere and carry ADR sign-off; a numeric module constant appearing
    here would be an unsigned pin arriving through the back door.

    Prose is exempt — the docstring quotes D4's 500 bp *as a citation*. What is asserted is
    that no public module-level attribute holds a number, so nothing here can be read as a
    value. Adding ``WINDOW_BP = 500`` to the module turns this red.
    """
    from tbox_finder.data import flank_context as fc

    # The only numbers allowed to appear are the transport constants this module *re-exports*
    # from flank_context (promote-don't-duplicate). Each must still equal its original, so a
    # local copy that has drifted fails here too.
    reexported = {"NCBI_TIMEOUT_S", "RATE_LIMIT_NO_KEY"}
    numeric = {
        name: getattr(asup, name)
        for name in dir(asup)
        if not name.startswith("_")
        and isinstance(getattr(asup, name), (int, float))
        and not isinstance(getattr(asup, name), bool)
    }
    assert (
        set(numeric) <= reexported
    ), f"module-level numeric constant(s) with no ADR: {sorted(set(numeric) - reexported)}"
    for name in numeric:
        assert numeric[name] == getattr(fc, name), f"{name} has drifted from flank_context"
