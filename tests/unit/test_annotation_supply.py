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

BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/296/795/GCA_000296795.1_ASM29679v1"
FNA_URL = f"{BASE}/GCA_000296795.1_ASM29679v1_genomic.fna.gz"
BASENAME = "GCA_000296795.1_ASM29679v1"
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
    manifest, note = fetch_file_manifest(BASE, retries=3, opener=op)
    assert manifest is None and note
    assert len(op.calls) == 3, "a 429 is transient and must be retried"


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
    return {asup.CONTROL_POSITIVE_ACCESSION: FNA_URL}


def test_run_control_is_powered_when_both_legs_behave() -> None:
    op = _Opener({f"{BASE}/{asup.MD5_MANIFEST_NAME}": _manifest_text(_annotated_names())})
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
            return _manifest_text(_annotated_names())

    control = run_control(_control_urls(), opener=_AlwaysOpen())
    assert control["positive_status"] == STATUS_ANNOTATED, "the positive leg still 'passes'"
    assert control["negative"]["resolved"] is True
    assert control_is_powered(control) is False


def test_control_is_unpowered_when_the_positive_resolves_but_is_unannotated() -> None:
    """The exact degenerate run the control exists for: everything parses, nothing has a GFF.
    Without the positive leg this reports 660 ``unannotated`` and certifies the gene caller."""
    op = _Opener({f"{BASE}/{asup.MD5_MANIFEST_NAME}": _manifest_text(_unannotated_names())})
    control = run_control(_control_urls(), opener=op)
    assert control["positive_status"] == STATUS_UNANNOTATED
    assert control_is_powered(control) is False


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
    return {
        "sweep_complete": complete,
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
    urls: dict[str, str] = {asup.CONTROL_POSITIVE_ACCESSION: FNA_URL}
    bodies: dict[str, str] = {
        f"{BASE}/{asup.MD5_MANIFEST_NAME}": _manifest_text(_annotated_names())
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
    )
    assert rep["admissible_status_counts"] == {
        STATUS_ANNOTATED: 3,
        STATUS_UNANNOTATED: 2,
        STATUS_UNKNOWN: 0,
    }
    assert rep["route"]["route"] == ROUTE_MIXED
    assert rep["n_candidate_hosts"] == 2


def test_measure_with_limit_cannot_certify() -> None:
    urls, bodies = _corpus(5, 0)
    admissible = sorted(a for a in urls if a != asup.CONTROL_POSITIVE_ACCESSION)
    rep = measure_annotation_supply(
        admissible=admissible,
        candidate_hosts=admissible[:2],
        source_urls=urls,
        workers=1,
        opener=_Opener(bodies),
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
    assert asup.head_content_length("http://x/y.gz", header=lambda _u: 4096) == 4096


def test_head_content_length_returns_none_rather_than_zero_on_failure() -> None:
    """A failed HEAD read as 0 understates the download budget a SLURM ack is sized against."""

    def _boom(_url: str) -> int:
        raise urllib.error.HTTPError("u", 404, "no", None, None)  # type: ignore[arg-type]

    assert asup.head_content_length("http://x/y.gz", header=_boom) is None


def test_head_content_length_returns_none_on_a_missing_header() -> None:
    """``Content-Length`` absent → the transport reports -1; that is unknown, not zero."""
    assert asup.head_content_length("http://x/y.gz", header=lambda _u: -1) is None


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
