"""P3-15′-c-i (acquisition) — the md5-verified GFF fetch, its refusals, and its control.

**No test here touches the network.** The autouse ``_no_network`` guard raises a
``BaseException`` subclass rather than an ``Exception``, because ``download_gff`` catches
``Exception`` and retries: an ``AssertionError``-based guard is *swallowed by its own subject*
and the suite goes green while making real requests — the r4 major of PR #112, where three
end-to-end tests were quietly issuing 30 requests to live NCBI in 45 s behind a docstring
saying they did not ([[guard-exception-swallowed-by-subject]]).

The committed supply report (``reports/p3/annotation_supply.json``) is used as a real input in
several tests: it is git-tracked, neither DVC- nor LFS-shaped, so CI, the laptop and the
cluster all read the same 339 rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
from pathlib import Path

import pytest

from tbox_finder.mining import annotation_fetch as af
from tbox_finder.mining import annotation_supply as asup

REPO = Path(__file__).resolve().parents[2]
SUPPLY_REPORT = REPO / "reports" / "p3" / "annotation_supply.json"
FIXTURE_GFF = REPO / "tests" / "fixtures" / "annotation" / "GCA_002790315.1.gff.gz"

ACC = "GCA_000296795.1"
BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/296/795/GCA_000296795.1_ASM29679v1"
URL = f"{BASE}/GCA_000296795.1_ASM29679v1_genomic.gff.gz"
#: a second, equally well-formed assembly — the corpus a report can describe *instead*.
OTHER_ACC = "GCA_002790315.1"
OTHER_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/790/315/GCA_002790315.1_ASM279031v1"
OTHER_URL = f"{OTHER_BASE}/GCA_002790315.1_ASM279031v1_genomic.gff.gz"
PAYLOAD = b"##gff-version 3\nctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=cds-A;product=p\n"
PAYLOAD_MD5 = hashlib.md5(PAYLOAD, usedforsecurity=False).hexdigest()
N_PAYLOAD = len(PAYLOAD)


#: captured before ``_no_network`` replaces it, so the two transport-boundary tests can drive
#: the real function while the guard still protects every other test in the file.
_REAL_URLOPEN_BYTES = af._urlopen_bytes


class _NetworkAccess(BaseException):
    """Deliberately not an ``Exception`` — ``download_gff``'s retry loop must not absorb it."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(
        af, "_urlopen_bytes", lambda url: (_ for _ in ()).throw(_NetworkAccess(url))
    )
    monkeypatch.setattr(
        asup, "_urlopen_text", lambda url: (_ for _ in ()).throw(_NetworkAccess(url))
    )
    # The THIRD transport helper. Leaving it unguarded means any future test reaching a supply
    # code path that HEADs would issue a live request while this file's docstring still claims
    # it does not — the PR #112 shape the docstring itself cites.
    monkeypatch.setattr(
        asup, "_urlhead_length", lambda url: (_ for _ in ()).throw(_NetworkAccess(url))
    )


def _target(accession: str = ACC, url: str = URL, md5: str = PAYLOAD_MD5, n: int = N_PAYLOAD):
    return {"accession": accession, "gff_url": url, "gff_md5": md5, "expected_bytes": n}


def _opener(mapping):
    """A scripted opener: known URL → bytes, anything else → 404."""

    def _get(url):
        if url in mapping:
            value = mapping[url]
            if isinstance(value, BaseException):
                raise value
            return value
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    return _get


def _supply_payload():
    return json.loads(SUPPLY_REPORT.read_text(encoding="utf-8"))


def _minimal_supply(rows, *, annotated: int | None = None, bytes_total: int = 1, **over):
    n = len([r for r in rows if r.get("status") == "annotated"]) if annotated is None else annotated
    payload = {
        "sweep_complete": True,
        "route": {"route": "mixed"},
        "per_assembly": rows,
        "admissible_status_counts": {"annotated": n, "unannotated": 0, "unknown": 0},
        "gff_bytes_total_known": bytes_total,
    }
    payload.update(over)
    return payload


def _write_supply(tmp_path, payload) -> Path:
    path = tmp_path / "supply.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _annotated_row(accession=ACC, url=URL, md5=PAYLOAD_MD5, n_bytes=N_PAYLOAD, **over):
    row = {
        "accession": accession,
        "status": "annotated",
        "gff_url": url,
        "gff_md5": md5,
        "gff_bytes": n_bytes,
        "faa_present": True,
        "gbff_present": True,
        "n_files": 16,
        "note": "",
    }
    row.update(over)
    return row


# --------------------------------------------------------------------------- #
# The no-network guard must itself be effective
# --------------------------------------------------------------------------- #


def test_the_no_network_guard_is_not_swallowed_by_the_retry_loop():
    """If this ever goes green by *returning*, every test below may be hitting NCBI."""
    with pytest.raises(_NetworkAccess):
        af.download_gff(URL, retries=3, sleep=lambda _s: None)


# --------------------------------------------------------------------------- #
# load_annotation_targets — against the real committed report, then its refusals
# --------------------------------------------------------------------------- #


def test_loads_the_real_committed_supply_report():
    targets = af.load_annotation_targets(SUPPLY_REPORT)
    payload = _supply_payload()
    assert len(targets) == payload["admissible_status_counts"]["annotated"] == 339
    assert sum(t["expected_bytes"] for t in targets) == payload["gff_bytes_total_known"]


def test_the_pinned_control_assembly_is_in_the_real_target_set():
    """A control outside the target set is an unpowered control, and this is where it shows."""
    accessions = {t["accession"] for t in af.load_annotation_targets(SUPPLY_REPORT)}
    assert af.CONTROL_ACCESSION in accessions


def test_every_real_target_url_is_allowlisted_and_bound_to_its_accession():
    for target in af.load_annotation_targets(SUPPLY_REPORT):
        asup.require_allowed_url(target["gff_url"])
        af.check_url_binds_to_accession(target["accession"], target["gff_url"])


def test_unannotated_rows_are_not_targets(tmp_path):
    rows = [
        _annotated_row(),
        _annotated_row(
            accession="GCA_000372225.1",
            status="unannotated",
            gff_url="",
            gff_md5="",
            gff_bytes=None,
        ),
    ]
    path = _write_supply(tmp_path, _minimal_supply(rows, bytes_total=len(PAYLOAD)))
    assert [t["accession"] for t in af.load_annotation_targets(path)] == [ACC]


def test_refuses_a_partial_sweep(tmp_path):
    """``--limit`` on the *measurement* means the annotated set was never enumerated."""
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row()], sweep_complete=False))
    with pytest.raises(af.AnnotationFetchError, match="sweep_complete"):
        af.load_annotation_targets(path)


def test_refuses_a_refused_route(tmp_path):
    payload = _minimal_supply([_annotated_row()])
    payload["route"] = {"route": asup.ROUTE_REFUSED, "reasons": ["because"]}
    with pytest.raises(af.AnnotationFetchError, match=asup.ROUTE_REFUSED):
        af.load_annotation_targets(_write_supply(tmp_path, payload))


def test_refuses_a_missing_report(tmp_path):
    with pytest.raises(af.AnnotationFetchError, match="not found"):
        af.load_annotation_targets(tmp_path / "absent.json")


def test_refuses_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(af.AnnotationFetchError, match="not valid JSON"):
        af.load_annotation_targets(path)


def test_refuses_a_list_rooted_report(tmp_path):
    """A list root makes ``payload.get`` an ``AttributeError`` — a traceback, not a refusal."""
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(af.AnnotationFetchError, match="root is list"):
        af.load_annotation_targets(path)


def test_refuses_a_non_list_per_assembly(tmp_path):
    payload = _minimal_supply([_annotated_row()])
    payload["per_assembly"] = {"GCA_1": {}}
    with pytest.raises(af.AnnotationFetchError, match="per_assembly is dict"):
        af.load_annotation_targets(_write_supply(tmp_path, payload))


def test_refuses_a_non_mapping_row(tmp_path):
    payload = _minimal_supply([_annotated_row()])
    payload["per_assembly"] = ["GCA_000296795.1"]
    with pytest.raises(af.AnnotationFetchError, match="row is str"):
        af.load_annotation_targets(_write_supply(tmp_path, payload))


def test_refuses_a_missing_status_counts_block(tmp_path):
    payload = _minimal_supply([_annotated_row()])
    del payload["admissible_status_counts"]
    with pytest.raises(af.AnnotationFetchError, match="admissible_status_counts"):
        af.load_annotation_targets(_write_supply(tmp_path, payload))


@pytest.mark.parametrize("value", [True, "339", 3.0, None])
def test_refuses_a_non_int_annotated_count(tmp_path, value):
    """``bool`` is an ``int`` in Python; coercing ``"339"`` is the coerce-before-validate bug."""
    payload = _minimal_supply([_annotated_row()])
    payload["admissible_status_counts"]["annotated"] = value
    with pytest.raises(af.AnnotationFetchError, match="expected an int"):
        af.load_annotation_targets(_write_supply(tmp_path, payload))


def test_refuses_an_annotated_row_with_no_url(tmp_path):
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row(url="")]))
    with pytest.raises(af.AnnotationFetchError, match="gff_url is empty"):
        af.load_annotation_targets(path)


@pytest.mark.parametrize("md5", ["", "abc", "0" * 31, "0" * 33, "g" * 32, " " + "0" * 31])
def test_refuses_a_malformed_md5(tmp_path, md5):
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row(md5=md5)]))
    with pytest.raises(af.AnnotationFetchError, match="gff_md5"):
        af.load_annotation_targets(path)


def test_accepts_an_uppercase_md5_by_normalising_it(tmp_path):
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row(md5=PAYLOAD_MD5.upper())]))
    assert af.load_annotation_targets(path)[0]["gff_md5"] == PAYLOAD_MD5


@pytest.mark.parametrize("n_bytes", [0, -1, None, "163050", True, 1.5])
def test_refuses_a_non_positive_int_byte_count(tmp_path, n_bytes):
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row(n_bytes=n_bytes)]))
    with pytest.raises(af.AnnotationFetchError, match="gff_bytes"):
        af.load_annotation_targets(path)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/GCA_000296795.1_ASM29679v1_genomic.gff.gz",
        "http://ftp.ncbi.nlm.nih.gov/GCA_000296795.1_ASM29679v1_genomic.gff.gz",
        "https://evil.test/ftp.ncbi.nlm.nih.gov/GCA_000296795.1_ASM29679v1_genomic.gff.gz",
        "https://ftp.ncbi.nlm.nih.gov.evil.test/GCA_000296795.1_ASM29679v1_genomic.gff.gz",
    ],
)
def test_refuses_a_url_outside_the_allowlist(tmp_path, url):
    """Substring and prefix guards each lose to one of these; the message is asserted too.

    Every URL here **ends in the GFF suffix and carries the right accession**, so neither the
    suffix rule nor the binding rule can refuse it — only the allowlist can. Without that, a
    bare ``pytest.raises`` passes on whichever guard happens to fire first, and deleting the
    allowlist entirely leaves the test green.
    """
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row(url=url)]))
    with pytest.raises(af.AnnotationFetchError, match="not an allowed NCBI URL"):
        af.load_annotation_targets(path)


def test_refuses_a_url_that_is_not_a_gff(tmp_path):
    path = _write_supply(
        tmp_path, _minimal_supply([_annotated_row(url=f"{BASE}/x_genomic.fna.gz")])
    )
    with pytest.raises(af.AnnotationFetchError, match="does not end in"):
        af.load_annotation_targets(path)


def test_refuses_a_url_belonging_to_a_different_assembly(tmp_path):
    """A mis-join files a real GFF under the wrong accession and never looks like a failure."""
    other = f"{BASE}/GCA_999999999.1_ASMx_genomic.gff.gz"
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row(url=other)]))
    with pytest.raises(af.AnnotationFetchError, match="does not belong to"):
        af.load_annotation_targets(path)


def test_refuses_when_the_row_count_disagrees_with_the_reports_own_tally(tmp_path):
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row()], annotated=2))
    with pytest.raises(af.AnnotationFetchError, match="disagrees with itself"):
        af.load_annotation_targets(path)


def test_refuses_duplicate_annotated_accessions(tmp_path):
    """A duplicate key merges instead of colliding, and the summed totals stay satisfied."""
    path = _write_supply(
        tmp_path, _minimal_supply([_annotated_row(), _annotated_row()], annotated=2)
    )
    with pytest.raises(af.AnnotationFetchError, match="duplicate"):
        af.load_annotation_targets(path)


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #


def test_destination_name_is_accession_keyed():
    assert af.destination_name(ACC) == "GCA_000296795.1.gff.gz"


@pytest.mark.parametrize("bad", ["", "GCA_296795.1", "GCX_000296795.1", "GCA_000296795", ACC + " "])
def test_destination_name_refuses_a_malformed_accession(bad):
    with pytest.raises(af.AnnotationFetchError, match="malformed assembly accession"):
        af.destination_name(bad)


def test_md5_hex_agrees_with_hashlib():
    assert af.md5_hex(PAYLOAD) == hashlib.md5(PAYLOAD, usedforsecurity=False).hexdigest()


def test_derive_status_counts_refuses_an_unknown_status():
    with pytest.raises(af.AnnotationFetchError, match="unknown per-assembly status"):
        af.derive_status_counts([{"status": "probably_fine"}])


# --------------------------------------------------------------------------- #
# The verification control — each leg broken ALONE
# --------------------------------------------------------------------------- #


def test_control_passes_on_real_matching_bytes():
    control = af.verification_control(PAYLOAD, PAYLOAD_MD5)
    assert af.control_is_powered(control)
    assert control["observed_md5"] == PAYLOAD_MD5


def test_control_positive_leg_fails_on_the_wrong_expectation():
    control = af.verification_control(PAYLOAD, "0" * 32)
    assert control["positive_matches"] is False
    assert af.control_is_powered(control) is False


@pytest.mark.parametrize(
    "leg", ["positive_matches", "corrupt_expectation_fails", "corrupt_payload_fails"]
)
def test_control_is_unpowered_when_any_single_leg_is_false(leg):
    """Broken one at a time: any two legs alone are satisfied by a degenerate comparison."""
    control = dict(af.verification_control(PAYLOAD, PAYLOAD_MD5))
    control[leg] = False
    assert af.control_is_powered(control) is False


@pytest.mark.parametrize(
    "leg", ["positive_matches", "corrupt_expectation_fails", "corrupt_payload_fails"]
)
def test_control_is_unpowered_when_any_single_leg_is_MISSING(leg):
    """Absence must read as unpowered, not be skipped by the ``all``."""
    control = dict(af.verification_control(PAYLOAD, PAYLOAD_MD5))
    del control[leg]
    assert af.control_is_powered(control) is False


def test_control_is_unpowered_on_an_empty_mapping():
    assert af.control_is_powered({}) is False


def test_a_comparator_that_always_matches_makes_the_control_report_itself_unpowered(monkeypatch):
    """The point of routing the legs through ``digest_matches``.

    Inline, the two negative legs assert only that md5 has no collisions — they could not go
    False whatever the acquisition did, so a hardcoded ``True`` was indistinguishable from a
    computed one. Through the shared comparator, a broken comparison is visible *in the
    control*, which is where a broken comparison has to be visible.
    """
    monkeypatch.setattr(af, "digest_matches", lambda payload, expected: True)
    control = af.verification_control(PAYLOAD, PAYLOAD_MD5)
    assert control["corrupt_expectation_fails"] is False
    assert control["corrupt_payload_fails"] is False
    assert af.control_is_powered(control) is False


def test_a_comparator_that_never_matches_also_makes_the_control_unpowered(monkeypatch):
    monkeypatch.setattr(af, "digest_matches", lambda payload, expected: False)
    control = af.verification_control(PAYLOAD, PAYLOAD_MD5)
    assert control["positive_matches"] is False
    assert af.control_is_powered(control) is False


def test_digest_matches_is_the_comparator_the_fetch_path_uses(monkeypatch, tmp_path):
    """A control wired to a *different* comparator than the fetch certifies nothing."""
    seen: list[tuple[int, str]] = []

    def _spy(payload, expected):
        seen.append((len(payload), str(expected)))
        return af.md5_hex(payload) == str(expected).strip().lower()

    monkeypatch.setattr(af, "digest_matches", _spy)
    af.fetch_one(_target(), annotation_dir=tmp_path, opener=_opener({URL: PAYLOAD}))
    assert seen == [(len(PAYLOAD), PAYLOAD_MD5)]


@pytest.mark.parametrize("expected", [PAYLOAD_MD5.upper(), f"  {PAYLOAD_MD5}  "])
def test_digest_matches_normalises_case_and_surrounding_space(expected):
    assert af.digest_matches(PAYLOAD, expected) is True


def test_digest_matches_is_false_on_a_single_flipped_byte():
    assert af.digest_matches(PAYLOAD + b"\x00", PAYLOAD_MD5) is False


def test_control_flip_never_produces_the_original_md5():
    """The flipped expectation must actually differ, including when the md5 starts with 'b'."""
    for md5 in (PAYLOAD_MD5, "b" + "0" * 31, "c" + "0" * 31):
        control = af.verification_control(PAYLOAD, md5)
        assert control["corrupt_expectation_fails"] is True


# --------------------------------------------------------------------------- #
# download_gff — retries, backoff, and what must NOT be retried
# --------------------------------------------------------------------------- #


def test_download_returns_the_payload():
    payload, note, permanent = af.download_gff(URL, opener=_opener({URL: PAYLOAD}))
    assert payload == PAYLOAD and note == "" and permanent is False


def test_transient_failure_retries_with_the_declared_backoff_schedule():
    slept: list[float] = []
    err = urllib.error.HTTPError(URL, 429, "Too Many Requests", {}, None)  # type: ignore[arg-type]
    payload, note, permanent = af.download_gff(
        URL, retries=4, opener=_opener({URL: err}), sleep=slept.append
    )
    assert payload is None and permanent is False and "429" in note
    assert slept == [1.5, 3.0, 4.5]


def test_a_permanent_4xx_does_not_sleep_at_all():
    """339 pointless waits turn a two-minute acquisition into a twenty-minute one."""
    slept: list[float] = []
    err = urllib.error.HTTPError(URL, 404, "Not Found", {}, None)  # type: ignore[arg-type]
    payload, note, permanent = af.download_gff(
        URL, retries=4, opener=_opener({URL: err}), sleep=slept.append
    )
    assert payload is None and permanent is True and slept == []


def test_a_403_is_treated_as_transient_not_permanent():
    """NCBI rate-limits with 403 as well as 429; a permanent read would abandon the host."""
    slept: list[float] = []
    err = urllib.error.HTTPError(URL, 403, "Forbidden", {}, None)  # type: ignore[arg-type]
    _, _, permanent = af.download_gff(
        URL, retries=2, opener=_opener({URL: err}), sleep=slept.append
    )
    assert permanent is False and slept == [1.5]


def test_a_succeeding_retry_returns_the_payload():
    calls: list[int] = []

    def _flaky(url):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "x", {}, None)  # type: ignore[arg-type]
        return PAYLOAD

    payload, _, _ = af.download_gff(URL, opener=_flaky, sleep=lambda _s: None)
    assert payload == PAYLOAD and len(calls) == 2


def test_an_allowlist_violation_propagates_rather_than_being_retried():
    """A disallowed URL is a refusal, not a flaky host — retrying it four times is nonsense."""
    slept: list[float] = []

    def _guard(url):
        raise asup.AnnotationSupplyError("not an allowed NCBI URL")

    with pytest.raises(asup.AnnotationSupplyError):
        af.download_gff(URL, opener=_guard, sleep=slept.append)
    assert slept == []


# --------------------------------------------------------------------------- #
# fetch_one — the write path, the cache path, and what must never be written
# --------------------------------------------------------------------------- #


def test_fetch_one_writes_the_bytes_and_reports_ok(tmp_path):
    row = af.fetch_one(_target(), annotation_dir=tmp_path, opener=_opener({URL: PAYLOAD}))
    assert row["status"] == af.STATUS_OK and row["from_cache"] is False
    assert (tmp_path / "GCA_000296795.1.gff.gz").read_bytes() == PAYLOAD
    assert row["observed_md5"] == row["expected_md5"] == PAYLOAD_MD5


def test_a_mismatching_payload_is_NOT_written_to_disk(tmp_path):
    """The corrupt bytes must never land under a name a later step would happily parse."""
    row = af.fetch_one(
        _target(md5="0" * 32), annotation_dir=tmp_path, opener=_opener({URL: PAYLOAD})
    )
    assert row["status"] == af.STATUS_MD5_MISMATCH
    assert not (tmp_path / "GCA_000296795.1.gff.gz").exists()


def test_a_cached_file_is_re_hashed_and_not_re_downloaded(tmp_path):
    (tmp_path / "GCA_000296795.1.gff.gz").write_bytes(PAYLOAD)

    def _must_not_run(url):
        # _NetworkAccess, not AssertionError: ``download_gff`` catches ``Exception`` and
        # retries, so an AssertionError signal is absorbed by the very loop it is watching —
        # the test would then sleep the real 1.5/3.0/4.5 s schedule and fail on a status
        # assertion instead of this message ([[guard-exception-swallowed-by-subject]]).
        raise _NetworkAccess(f"downloaded {url} despite a valid cache hit")

    row = af.fetch_one(_target(), annotation_dir=tmp_path, opener=_must_not_run)
    assert row["status"] == af.STATUS_OK and row["from_cache"] is True


def test_the_cache_hit_guard_propagates_rather_than_being_retried(tmp_path):
    """Why the guard above must be a ``BaseException``, asserted where it is falsifiable.

    The cache-hit test alone cannot see the difference: when the cache path is correct the
    guard is never called, and when it regresses an ``AssertionError`` is swallowed by
    ``download_gff``'s ``except Exception`` and the test still goes red — just 9 s later, on a
    status assertion, with the wrong message. Here the guard is *reached* (``force=True``), so
    the exception type is load-bearing: an ``Exception`` subclass would be retried and this
    would return a row instead of raising.
    """
    (tmp_path / "GCA_000296795.1.gff.gz").write_bytes(PAYLOAD)
    slept: list[float] = []

    def _guard(url):
        raise _NetworkAccess(f"downloaded {url}")

    with pytest.raises(_NetworkAccess):
        af.fetch_one(
            _target(), annotation_dir=tmp_path, opener=_guard, sleep=slept.append, force=True
        )
    assert slept == []


def test_a_corrupted_cached_file_is_re_downloaded_not_trusted(tmp_path):
    dest = tmp_path / "GCA_000296795.1.gff.gz"
    dest.write_bytes(b"truncated")
    row = af.fetch_one(_target(), annotation_dir=tmp_path, opener=_opener({URL: PAYLOAD}))
    assert row["status"] == af.STATUS_OK and row["from_cache"] is False
    assert dest.read_bytes() == PAYLOAD
    assert "re-downloading" in row["note"]


def test_force_re_downloads_a_valid_cache_hit(tmp_path):
    (tmp_path / "GCA_000296795.1.gff.gz").write_bytes(PAYLOAD)
    row = af.fetch_one(
        _target(), annotation_dir=tmp_path, opener=_opener({URL: PAYLOAD}), force=True
    )
    assert row["from_cache"] is False


def test_a_permanent_failure_is_unavailable_and_a_transient_one_is_fetch_failed(tmp_path):
    gone = urllib.error.HTTPError(URL, 404, "Not Found", {}, None)  # type: ignore[arg-type]
    row = af.fetch_one(_target(), annotation_dir=tmp_path, opener=_opener({URL: gone}))
    assert row["status"] == af.STATUS_UNAVAILABLE

    busy = urllib.error.HTTPError(URL, 429, "Busy", {}, None)  # type: ignore[arg-type]
    row = af.fetch_one(
        _target(), annotation_dir=tmp_path, opener=_opener({URL: busy}), sleep=lambda _s: None
    )
    assert row["status"] == af.STATUS_FETCH_FAILED


def test_fetch_one_refuses_a_url_bound_to_another_accession(tmp_path):
    other = f"{BASE}/GCA_999999999.1_ASMx_genomic.gff.gz"
    with pytest.raises(af.AnnotationFetchError, match="does not belong to"):
        af.fetch_one(_target(url=other), annotation_dir=tmp_path, opener=_opener({}))


# --------------------------------------------------------------------------- #
# Orphans
# --------------------------------------------------------------------------- #


def test_an_unwritable_destination_becomes_a_failed_row_not_a_traceback(tmp_path, monkeypatch):
    """A full disk or a permission error must still produce the refusal report."""

    def _boom(self, data):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(Path, "write_bytes", _boom)
    row = af.fetch_one(_target(), annotation_dir=tmp_path, opener=_opener({URL: PAYLOAD}))
    assert row["status"] == af.STATUS_FETCH_FAILED and "PermissionError" in row["note"]


def test_an_unreadable_cache_file_becomes_a_failed_row_not_a_traceback(tmp_path, monkeypatch):
    (tmp_path / "GCA_000296795.1.gff.gz").write_bytes(PAYLOAD)

    def _boom(self):
        raise PermissionError("cannot read")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    row = af.fetch_one(_target(), annotation_dir=tmp_path, opener=_opener({URL: PAYLOAD}))
    assert row["status"] == af.STATUS_FETCH_FAILED and "PermissionError" in row["note"]


def test_the_destination_is_replaced_atomically_and_no_partial_is_left(tmp_path, monkeypatch):
    """A direct write leaves truncated bytes visible if the process stops mid-write."""
    seen: list[str] = []
    real_replace = os.replace

    def _spy(src, dst):
        seen.append(f"{Path(src).name}->{Path(dst).name}")
        return real_replace(src, dst)

    monkeypatch.setattr(af.os, "replace", _spy)
    af.fetch_one(_target(), annotation_dir=tmp_path, opener=_opener({URL: PAYLOAD}))
    assert len(seen) == 1 and seen[0].endswith("->GCA_000296795.1.gff.gz")
    assert [p.name for p in tmp_path.iterdir()] == ["GCA_000296795.1.gff.gz"]


def test_an_in_flight_temp_file_is_not_reported_as_an_orphan(tmp_path):
    """The property lives in the temp file's NAME: it must not end in ``.gff.gz``, or a
    concurrent orphan sweep would report a half-written file as foreign annotation."""
    dest = tmp_path / "GCA_000296795.1.gff.gz"
    tmp_name = af._temp_name(dest)
    assert not tmp_name.endswith(".gff.gz")
    (tmp_path / tmp_name).write_bytes(b"partial")
    assert af.find_orphans(tmp_path, [ACC]) == []


def test_orphans_are_reported_and_the_files_are_left_alone(tmp_path):
    (tmp_path / "GCA_000296795.1.gff.gz").write_bytes(PAYLOAD)
    stray = tmp_path / "GCA_111111111.1.gff.gz"
    stray.write_bytes(b"x")
    assert af.find_orphans(tmp_path, [ACC]) == ["GCA_111111111.1.gff.gz"]
    assert stray.exists()


def test_a_non_gff_file_is_not_an_orphan(tmp_path):
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")
    assert af.find_orphans(tmp_path, [ACC]) == []


def test_a_missing_annotation_directory_has_no_orphans(tmp_path):
    assert af.find_orphans(tmp_path / "absent", [ACC]) == []


# --------------------------------------------------------------------------- #
# validate_fetch_report — every clause broken ALONE
# --------------------------------------------------------------------------- #


def _clean_report(tmp_path, *, accession: str = ACC, url: str = URL):
    """A report that passes every clause — optionally about a *different* assembly.

    The accession/URL are parameters because the interesting failure is a report that is
    internally flawless and simply describes another corpus: nothing inside it can say so.
    """
    rows = [
        {
            "accession": accession,
            "status": af.STATUS_OK,
            "gff_url": url,
            "expected_md5": PAYLOAD_MD5,
            "observed_md5": PAYLOAD_MD5,
            "n_bytes": len(PAYLOAD),
            "path": str(tmp_path / f"{accession}.gff.gz"),
            "from_cache": False,
            "note": "",
        }
    ]
    return af.build_fetch_report(
        rows,
        targets=[_target(accession, url)],
        control=af.verification_control(PAYLOAD, PAYLOAD_MD5),
        orphans=[],
        n_annotated_in_supply=1,
        bytes_total_in_supply=len(PAYLOAD),
        sweep_complete=True,
    )


def test_a_clean_report_validates(tmp_path):
    assert af.validate_fetch_report(_clean_report(tmp_path), targets=[_target()]) == []


@pytest.mark.parametrize(
    ("clause", "mutate"),
    [
        ("sweep_complete", lambda r: r.update(sweep_complete=False)),
        ("targets_match_supply_report", lambda r: r.update(n_annotated_in_supply_report=2)),
        ("rows_match_targets", lambda r: r.update(n_targets=2)),
        (
            "no_duplicate_accessions",
            lambda r: r["per_assembly"].append(dict(r["per_assembly"][0], status="fetch_failed")),
        ),
        ("status_counts_rederive", lambda r: r["status_counts"].update(ok=99)),
        ("all_targets_ok", lambda r: r["per_assembly"][0].update(status="fetch_failed")),
        (
            "every_ok_row_md5_matches",
            lambda r: r["per_assembly"][0].update(observed_md5="0" * 32),
        ),
        ("bytes_total_rederives", lambda r: r.update(bytes_total=1)),
        ("bytes_total_matches_supply_report", lambda r: r.update(bytes_total_in_supply_report=1)),
        (
            "path_binds_to_accession",
            lambda r: r["per_assembly"][0].update(path="/tmp/GCA_999999999.1.gff.gz"),
        ),
        ("control_powered", lambda r: r["control"].update(corrupt_payload_fails=False)),
        ("no_orphans", lambda r: r.update(orphans=["GCA_111111111.1.gff.gz"])),
        ("clause_schema_current", lambda r: r.update(clause_schema_version="1")),
        (
            "rows_bind_to_targets",
            lambda r: r["per_assembly"][0].update(gff_url=OTHER_URL),
        ),
    ],
)
def test_each_clause_fails_alone(tmp_path, clause, mutate):
    """One clause broken at a time — an all-TRUE fixture cannot test a conjunction."""
    report = _clean_report(tmp_path)
    mutate(report)
    problems = af.validate_fetch_report(report, targets=[_target()])
    assert any(clause in p for p in problems), f"{clause} did not fail: {problems}"


def test_validation_evaluates_exactly_the_declared_clause_set(tmp_path):
    """A clause that is never evaluated contributes nothing to the ``all`` and passes.

    Asserted as set equality over the *declared* set. The opposite direction — a clause
    computed but never declared — cannot be seen from here, because ``named`` is built by
    iterating ``REQUIRED_CLAUSES``; it is asserted separately against the check
    ``validate_fetch_report`` now makes for it.
    """
    report = _clean_report(tmp_path)
    for key in ("sweep_complete", "n_targets", "control", "orphans"):
        report.pop(key, None)
    problems = af.validate_fetch_report(report, targets=[_target()])
    named = {c for c in af.REQUIRED_CLAUSES if any(c in p for p in problems)}
    assert named == {
        "sweep_complete",
        "targets_match_supply_report",
        "rows_match_targets",
        "all_targets_ok",
        "control_powered",
        "no_orphans",
    }


def test_a_clause_computed_but_never_declared_is_reported(monkeypatch, tmp_path):
    """The direction the set-equality test above cannot assert.

    ``validate_fetch_report``'s final comprehension iterates ``REQUIRED_CLAUSES``, so a clause
    the function computes but nobody declared is never read — it would be ignored whatever it
    evaluated to. The gate test's docstring claimed this was covered; it now is.
    """
    monkeypatch.setattr(af, "REQUIRED_CLAUSES", af.REQUIRED_CLAUSES - {"no_orphans"})
    problems = af.validate_fetch_report(_clean_report(tmp_path), targets=[_target()])
    assert problems == ["clause(s) computed but never declared: ['no_orphans']"]


def test_a_MALFORMED_n_bytes_fails_a_clause_instead_of_raising_a_traceback(tmp_path):
    """``int(r.get("n_bytes") or 0)`` coerced, and this function reads an ON-DISK report.

    ``parse_census`` validates the committed fetch report, so a hand-edited or truncated
    ``n_bytes`` aborted ``verify`` with a ``ValueError`` and exit 1 rather than returning the
    refusal the module promises — and ``"12"`` would have been accepted as 12, the
    coerce-before-validate shape ``strict_count`` exists to refuse.
    """
    for bad in ("68855450", 68855450.0, True, None, [1], {}):
        report = _clean_report(tmp_path)
        report["per_assembly"][0]["n_bytes"] = bad
        problems = af.validate_fetch_report(report, targets=[_target()])
        assert any("bytes_total_rederives" in p for p in problems), bad


def test_an_UNREADABLE_supply_report_is_a_refusal_not_a_traceback(tmp_path):
    """``FileNotFoundError`` was the only ``OSError`` converted to the documented refusal."""
    a_directory = tmp_path / "not-a-file"
    a_directory.mkdir()
    with pytest.raises(af.AnnotationFetchError, match="unreadable"):
        af.read_supply_report(a_directory)
    assert (
        af.main(
            [
                "verify",
                "--supply-report",
                str(a_directory),
                "--annotation-dir",
                str(tmp_path),
                "--out",
                str(tmp_path / "parse.json"),
            ]
        )
        == 3
    )


def test_a_NON_UTF8_supply_report_is_a_refusal_not_a_traceback(tmp_path):
    """``UnicodeDecodeError`` is a ``ValueError``, so no ``OSError`` clause would catch it."""
    path = tmp_path / "supply.json"
    path.write_bytes(b"\xff\xfe{}")
    with pytest.raises(af.AnnotationFetchError, match="not UTF-8"):
        af.read_supply_report(path)


def test_the_control_flip_survives_an_UPPERCASE_expectation(tmp_path):
    """``digest_matches`` lowercases; the flip did not, so the "corrupted" md5 compared EQUAL.

    The result was a control reporting itself unpowered on correct bytes — a control whose
    verdict is decided by the caller's capitalisation is not measuring the acquisition at all.
    """
    control = af.verification_control(PAYLOAD, PAYLOAD_MD5.upper())
    assert af.control_is_powered(control) is True
    # Both cases must agree leg-for-leg, or the control's answer depends on the input's case.
    assert {k: control[k] for k in ("positive_matches", "corrupt_expectation_fails")} == {
        "positive_matches": True,
        "corrupt_expectation_fails": True,
    }
    assert af.control_is_powered(af.verification_control(PAYLOAD, "")) is False


def test_strict_count_refuses_what_int_would_have_coerced():
    """``int("339")`` succeeds and ``int(None)`` raises — both are wrong answers here."""
    assert af.strict_count({"n": 339}, "n") == 339
    for bad in ("339", 339.0, True, None, [339], {}):
        with pytest.raises(af.AnnotationFetchError, match="expected an int"):
            af.strict_count({"n": bad}, "n")


def test_a_malformed_byte_total_refuses_instead_of_raising_a_traceback(tmp_path):
    """A non-numeric headline must exit 3 with the named refusal, not 1 with a ValueError."""
    path, url, payload = _control_supply(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["gff_bytes_total_known"] = "68855450"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(af.AnnotationFetchError, match="gff_bytes_total_known"):
        af.fetch_annotations(
            supply_report=path, annotation_dir=tmp_path / "ann", opener=_opener({url: payload})
        )


def test_the_supply_report_is_read_exactly_once_per_fetch(tmp_path, monkeypatch):
    """Two reads of one path can see different bytes, so the targets and the count they are
    checked against could describe different payloads."""
    path, url, payload = _control_supply(tmp_path)
    reads: list[str] = []
    real = af.read_supply_report

    def _spy(supply_report=af.DEFAULT_SUPPLY_REPORT):
        reads.append(str(supply_report))
        return real(supply_report)

    monkeypatch.setattr(af, "read_supply_report", _spy)
    af.fetch_annotations(
        supply_report=path,
        annotation_dir=tmp_path / "ann",
        opener=_opener({url: payload}),
        workers=1,
    )
    assert reads == [str(path)]


def test_a_missing_clause_key_is_reported_as_never_evaluated(monkeypatch, tmp_path):
    monkeypatch.setattr(af, "REQUIRED_CLAUSES", af.REQUIRED_CLAUSES | {"a_clause_nothing_computes"})
    problems = af.validate_fetch_report(_clean_report(tmp_path), targets=[_target()])
    assert problems == ["clause(s) never evaluated: ['a_clause_nothing_computes']"]


def test_validation_refuses_a_non_list_per_assembly():
    assert af.validate_fetch_report({"per_assembly": {}}, targets=[_target()]) == [
        "per_assembly is not a list of objects"
    ]


def test_a_FLAWLESS_report_about_a_DIFFERENT_CORPUS_is_refused(tmp_path):
    """Every numeric clause is satisfiable by a report about assemblies nobody asked for.

    The counts, the status tally, the byte total and the per-row md5 equality are all *internal*
    — they compare the report with itself, or with numbers its own writer copied in from the
    supply report. Swap the assembly and every one of them still passes. Asserted as an exact
    equality on the problem list, so this test says "only the identity clause can see it"
    rather than "something failed".
    """
    report = _clean_report(tmp_path, accession=OTHER_ACC, url=OTHER_URL)
    assert af.validate_fetch_report(report, targets=[_target(OTHER_ACC, OTHER_URL)]) == []
    assert af.validate_fetch_report(report, targets=[_target()]) == [
        "clause failed: rows_bind_to_targets"
    ]


def test_a_row_whose_URL_names_another_assembly_fails_the_binding_clause(tmp_path):
    """The row is *labelled* with the accession but the bytes came from the URL's directory."""
    report = _clean_report(tmp_path)
    report["per_assembly"][0]["gff_url"] = OTHER_URL
    assert af.validate_fetch_report(report, targets=[_target()]) == [
        "clause failed: rows_bind_to_targets"
    ]


def test_a_row_whose_EXPECTED_md5_was_rewritten_fails_only_the_binding_clause(tmp_path):
    """A row that agrees with itself passes every digest clause in the report.

    ``every_ok_row_md5_matches`` compares ``observed`` against ``expected`` — both fields of the
    same row — so rewriting the pair to any 32-hex value keeps it green. Only a comparison
    against the *supply report's* md5 catches an expectation that was moved to fit the bytes.
    """
    report = _clean_report(tmp_path)
    forged = "b" * 32
    report["per_assembly"][0].update(expected_md5=forged, observed_md5=forged)
    # Re-derive the digest too, so the forgery is *complete*: this is the strongest version of
    # the report a tamperer can write, and it is the only one worth testing against.
    report["corpus_digest"] = af.corpus_digest([(ACC, forged)])
    problems = af.validate_fetch_report(report, targets=[_target()])
    assert not any("every_ok_row_md5_matches" in p for p in problems)
    assert not any("corpus_digest_rederives" in p for p in problems)
    assert problems == ["clause failed: rows_bind_to_targets"]


def test_an_EMPTY_target_list_does_not_vacuously_satisfy_the_binding_clause(tmp_path):
    """Absent evidence must fail the clause it was evidence for.

    [[clauses-must-guard-emptiness]] — a clause read from a missing input is vacuously TRUE
    exactly when there is nothing to check it against.
    """
    problems = af.validate_fetch_report(_clean_report(tmp_path), targets=[])
    assert any("rows_bind_to_targets" in p for p in problems)


def test_a_report_written_under_an_OLDER_clause_set_does_not_read_as_a_pass(tmp_path):
    """Its ``validation_problems: []`` came from a weaker gate, and nothing else says so."""
    report = _clean_report(tmp_path)
    assert report["clause_schema_version"] == af.CLAUSE_SCHEMA_VERSION
    report["clause_schema_version"] = "1"
    assert af.validate_fetch_report(report, targets=[_target()]) == [
        "clause failed: clause_schema_current"
    ]


def test_the_committed_clause_set_and_its_version_are_declared_together():
    """A clause added without bumping the version leaves old reports looking current."""
    assert af.CLAUSE_SCHEMA_VERSION == "2"
    assert {"rows_bind_to_targets", "clause_schema_current"} <= af.REQUIRED_CLAUSES


def test_validate_fetch_report_has_NO_DEFAULT_for_targets():
    """A default would re-open the hole at every call site, silently.

    ``targets=None``/``()`` makes the identity clause unevaluable exactly where a caller forgot
    the evidence — and a clause with no evidence is the shape that reads as a pass. The pin is on
    the *signature*, because that is what a future caller inherits.
    """
    import inspect

    param = inspect.signature(af.validate_fetch_report).parameters["targets"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_an_md5_that_is_empty_on_both_sides_does_not_satisfy_the_match_clause(tmp_path):
    """``"" == ""`` is True; the clause must require a real 32-hex digest as well."""
    report = _clean_report(tmp_path)
    report["per_assembly"][0].update(observed_md5="", expected_md5="")
    assert any(
        "every_ok_row_md5_matches" in p
        for p in af.validate_fetch_report(report, targets=[_target()])
    )


# --------------------------------------------------------------------------- #
# fetch_annotations — the completeness and control preconditions
# --------------------------------------------------------------------------- #


def test_refuses_a_target_set_that_does_not_contain_the_control_assembly(tmp_path):
    path = _write_supply(tmp_path, _minimal_supply([_annotated_row()], bytes_total=len(PAYLOAD)))
    with pytest.raises(af.AnnotationFetchError, match="unpowered md5 control"):
        af.fetch_annotations(
            supply_report=path, annotation_dir=tmp_path / "ann", opener=_opener({})
        )


def _control_supply(tmp_path):
    payload = FIXTURE_GFF.read_bytes()
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    url = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/790/315/"
        "GCA_002790315.1_ASM279031v1/GCA_002790315.1_ASM279031v1_genomic.gff.gz"
    )
    row = _annotated_row(accession=af.CONTROL_ACCESSION, url=url, md5=md5, n_bytes=len(payload))
    path = _write_supply(tmp_path, _minimal_supply([row], bytes_total=len(payload)))
    return path, url, payload


def test_a_complete_single_target_run_validates(tmp_path):
    path, url, payload = _control_supply(tmp_path)
    report = af.fetch_annotations(
        supply_report=path,
        annotation_dir=tmp_path / "ann",
        opener=_opener({url: payload}),
        workers=1,
    )
    assert af.validate_fetch_report(report, targets=af.load_annotation_targets(path)) == []
    assert report["n_ok"] == 1 and report["n_downloaded"] == 1 and report["n_cached"] == 0
    assert report["bytes_total"] == len(payload)


def test_limit_sets_sweep_complete_false_and_the_validation_refuses(tmp_path):
    """A cost knob must not be able to hand a phase-exit path a green report."""
    path, url, payload = _control_supply(tmp_path)
    report = af.fetch_annotations(
        supply_report=path,
        annotation_dir=tmp_path / "ann",
        opener=_opener({url: payload}),
        workers=1,
        limit=1,
    )
    assert report["sweep_complete"] is False
    assert any(
        "sweep_complete" in p
        for p in af.validate_fetch_report(report, targets=af.load_annotation_targets(path))
    )


def test_a_limit_of_zero_leaves_the_control_unacquired_and_unpowered(tmp_path):
    path, url, payload = _control_supply(tmp_path)
    report = af.fetch_annotations(
        supply_report=path,
        annotation_dir=tmp_path / "ann",
        opener=_opener({url: payload}),
        workers=1,
        limit=0,
    )
    assert af.control_is_powered(report["control"]) is False
    assert any(
        "control_powered" in p
        for p in af.validate_fetch_report(report, targets=af.load_annotation_targets(path))
    )


def test_an_UNREADABLE_control_file_is_an_unpowered_control_not_a_traceback(tmp_path, monkeypatch):
    """A filesystem fault between writing the control and hashing it must not abort the run.

    Uncaught, the ``OSError`` escapes ``main``'s refusal handler as exit 1 **and no report is
    written at all** — so the one artifact that would have said "this acquisition cannot be
    trusted" is the thing the failure destroys. Fail-closed instead: the three legs stay False,
    ``control_powered`` fails, and the CLI exits 3 with a report that names the reason.
    """
    path, url, payload = _control_supply(tmp_path)
    ann = tmp_path / "ann"
    control_path = ann / af.destination_name(af.CONTROL_ACCESSION)
    real_read = Path.read_bytes

    def _refuse(self):
        if self == control_path:
            raise PermissionError(13, "Permission denied")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", _refuse)
    report = af.fetch_annotations(
        supply_report=path, annotation_dir=ann, opener=_opener({url: payload}), workers=1
    )
    assert af.control_is_powered(report["control"]) is False
    assert "unreadable" in report["control"]["note"]
    problems = af.validate_fetch_report(report, targets=af.load_annotation_targets(path))
    assert any("control_powered" in p for p in problems)


def test_a_failed_run_does_not_validate(tmp_path):
    path, url, _payload = _control_supply(tmp_path)
    report = af.fetch_annotations(
        supply_report=path,
        annotation_dir=tmp_path / "ann",
        opener=_opener({}),
        workers=1,
        sleep=lambda _s: None,
    )
    problems = af.validate_fetch_report(report, targets=af.load_annotation_targets(path))
    assert any("all_targets_ok" in p for p in problems)
    assert any("control_powered" in p for p in problems)


# --------------------------------------------------------------------------- #
# parse_census (the offline `verify` half)
# --------------------------------------------------------------------------- #


def _acquire(tmp_path):
    """Run a real single-target acquisition, then hand back what `verify` needs.

    The census tests go through the acquisition rather than dropping bytes on disk, because
    the ordering (`fetch` then `verify`) is now part of what is under test.
    """
    path, url, payload = _control_supply(tmp_path)
    ann = tmp_path / "ann"
    report = af.fetch_annotations(
        supply_report=path, annotation_dir=ann, opener=_opener({url: payload}), workers=1
    )
    fetch_path = tmp_path / "fetch.json"
    fetch_path.write_text(json.dumps(report), encoding="utf-8")
    return path, ann, fetch_path, payload


def test_parse_census_re_hashes_and_parses_the_real_fixture(tmp_path):
    path, ann, fetch_path, _payload = _acquire(tmp_path)
    census = af.parse_census(supply_report=path, annotation_dir=ann, fetch_report=fetch_path)
    assert census["corpus_matches_fetch_report"] is True
    assert census["corpus_check_reason"] == "ok"
    assert census["n_ok"] == 1 and census["n_failed"] == 0 and census["failures"] == []
    assert census["totals"]["n_cds"] == 455
    assert census["totals"]["n_cds_pseudo"] == 38
    assert census["totals"]["n_seqids"] == 41
    assert census["totals"]["n_with_fasta_section"] == 0


def test_the_census_parses_the_bytes_it_hashed_not_a_second_read(tmp_path, monkeypatch):
    """Between the hash and a re-read the file can change, and the two halves of the census
    would then describe different bytes. Asserted by making the SECOND read return something
    else: a path-reading census sees it, a snapshot-parsing one cannot."""
    path, ann, fetch_path, payload = _acquire(tmp_path)
    target = ann / af.destination_name(af.CONTROL_ACCESSION)
    real_read = Path.read_bytes
    calls = {"n": 0}

    def _swap(self):
        data = real_read(self)
        if self == target:
            calls["n"] += 1
            if calls["n"] > 1:  # every read after the one that was hashed
                import gzip as _gz

                return _gz.compress(b"##gff-version 3\n")
        return data

    monkeypatch.setattr(Path, "read_bytes", _swap)
    census = af.parse_census(supply_report=path, annotation_dir=ann, fetch_report=fetch_path)
    assert census["n_failed"] == 0
    assert census["per_assembly"][0]["n_cds"] == 455


def test_parse_census_reports_a_missing_file(tmp_path):
    path, ann, fetch_path, _payload = _acquire(tmp_path)
    (ann / af.destination_name(af.CONTROL_ACCESSION)).unlink()
    census = af.parse_census(supply_report=path, annotation_dir=ann, fetch_report=fetch_path)
    assert census["n_failed"] == 1 and census["failures"][0]["note"] == "missing"


def test_an_UNREADABLE_annotation_file_is_ONE_failure_row_not_a_dead_census(tmp_path, monkeypatch):
    """The read sat above the ``except OSError`` that was written for it.

    One unreadable file among 339 therefore aborted the whole census with a traceback and exit 1
    — losing the 338 verdicts that had nothing wrong with them — where the module's contract is
    a failure row and exit 3.
    """
    path, ann, fetch_path, _payload = _acquire(tmp_path)
    target = ann / af.destination_name(af.CONTROL_ACCESSION)
    real_read = Path.read_bytes

    def _refuse(self):
        if self == target:
            raise PermissionError(13, "Permission denied")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", _refuse)
    census = af.parse_census(supply_report=path, annotation_dir=ann, fetch_report=fetch_path)
    assert census["n_failed"] == 1
    assert "PermissionError" in census["failures"][0]["note"]


def test_parse_census_reports_a_tampered_file_rather_than_parsing_it(tmp_path):
    path, ann, fetch_path, payload = _acquire(tmp_path)
    (ann / af.destination_name(af.CONTROL_ACCESSION)).write_bytes(payload + b"\x00")
    census = af.parse_census(supply_report=path, annotation_dir=ann, fetch_report=fetch_path)
    assert census["n_failed"] == 1 and "md5" in census["failures"][0]["note"]


def test_a_SELF_CONSISTENT_fetch_report_about_other_bytes_is_caught_by_the_binding_clause(
    tmp_path,
):
    """The r2 forgery, re-asserted against the layer that now catches it FIRST.

    A fetch report whose md5s were all rewritten together used to reach the corpus-digest
    comparison as a *validating* report; with ``rows_bind_to_targets`` it no longer validates at
    all, because its expectations no longer equal the supply report's. The refusal must name the
    clause, not collapse into the digest's generic "different corpus" — a reader has to be able
    to tell "this report is about other assemblies" from "the files on disk moved".
    """
    path, ann, fetch_path, _payload = _acquire(tmp_path)
    other = json.loads(fetch_path.read_text(encoding="utf-8"))
    other_md5 = "a" * 32
    for row in other["per_assembly"]:
        row["observed_md5"] = row["expected_md5"] = other_md5
    other["corpus_digest"] = af.corpus_digest(
        (r["accession"], other_md5) for r in other["per_assembly"]
    )
    fetch_path.write_text(json.dumps(other), encoding="utf-8")
    census = af.parse_census(supply_report=path, annotation_dir=ann, fetch_report=fetch_path)
    assert census["n_failed"] == 0
    assert census["corpus_matches_fetch_report"] is False
    assert "rows_bind_to_targets" in census["corpus_check_reason"]


def test_the_corpus_digest_still_fires_when_the_FILES_moved_under_an_honest_report(tmp_path):
    """The digest's own remaining job: the disk state, not the report, is what changed.

    This is the r2 ordering defect in its real form — a census that describes a different set of
    bytes than the acquisition certified. The fetch report here is untouched and validates; only
    the corpus it describes is no longer the corpus on disk.
    """
    path, ann, fetch_path, payload = _acquire(tmp_path)
    (ann / af.destination_name(af.CONTROL_ACCESSION)).write_bytes(payload + b"\x00")
    census = af.parse_census(supply_report=path, annotation_dir=ann, fetch_report=fetch_path)
    assert census["n_failed"] == 1
    assert census["corpus_matches_fetch_report"] is False
    assert census["corpus_check_reason"] == "the fetch report certifies a different corpus"


def test_a_fetch_report_with_a_HAND_EDITED_corpus_digest_does_not_certify_itself(tmp_path):
    """Reading ``corpus_digest`` out of an unvalidated report lets it declare anything."""
    path, ann, fetch_path, _payload = _acquire(tmp_path)
    forged = json.loads(fetch_path.read_text(encoding="utf-8"))
    forged["corpus_digest"] = "0" * 64
    fetch_path.write_text(json.dumps(forged), encoding="utf-8")
    census = af.parse_census(supply_report=path, annotation_dir=ann, fetch_report=fetch_path)
    assert census["corpus_matches_fetch_report"] is False
    assert "corpus_digest_rederives" in census["corpus_check_reason"]


def test_an_ABSENT_fetch_report_does_not_silently_satisfy_the_corpus_check(tmp_path):
    """Empty must not equal empty — the [[clauses-must-guard-emptiness]] shape."""
    path, ann, _fetch_path, _payload = _acquire(tmp_path)
    census = af.parse_census(
        supply_report=path, annotation_dir=ann, fetch_report=tmp_path / "absent.json"
    )
    assert census["fetch_report_corpus_digest"] == ""
    assert census["corpus_matches_fetch_report"] is False
    # The reason must name the ABSENCE, not a mismatch: collapsing the failure modes would
    # make each of them unable to change an outcome on its own.
    assert census["corpus_check_reason"].startswith("the fetch report is unreadable")


def test_cli_verify_exits_3_when_the_corpus_does_not_match_the_fetch_report(tmp_path):
    path, ann, fetch_path, _payload = _acquire(tmp_path)
    stale = json.loads(fetch_path.read_text(encoding="utf-8"))
    stale["corpus_digest"] = "0" * 64
    fetch_path.write_text(json.dumps(stale), encoding="utf-8")
    code = af.main(
        [
            "verify",
            "--supply-report",
            str(path),
            "--annotation-dir",
            str(ann),
            "--fetch-report",
            str(fetch_path),
            "--out",
            str(tmp_path / "parse.json"),
        ]
    )
    assert code == 3


def test_corpus_digest_is_order_independent_and_content_sensitive():
    a = af.corpus_digest([("GCA_1", "aa"), ("GCA_2", "bb")])
    assert a == af.corpus_digest([("GCA_2", "BB"), ("GCA_1", "AA")])
    assert a != af.corpus_digest([("GCA_1", "aa"), ("GCA_2", "cc")])
    assert a != af.corpus_digest([("GCA_1", "aa")])


def test_parse_census_reports_an_unparseable_file(tmp_path):
    import gzip as _gz

    broken = _gz.compress(b"ctg1\tx\tCDS\tnot-a-number\t20\t.\t+\t0\tID=a\n")
    md5 = hashlib.md5(broken, usedforsecurity=False).hexdigest()
    url = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/790/315/"
        "GCA_002790315.1_ASM279031v1/GCA_002790315.1_ASM279031v1_genomic.gff.gz"
    )
    row = _annotated_row(accession=af.CONTROL_ACCESSION, url=url, md5=md5, n_bytes=len(broken))
    path = _write_supply(tmp_path, _minimal_supply([row], bytes_total=len(broken)))
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / af.destination_name(af.CONTROL_ACCESSION)).write_bytes(broken)

    census = af.parse_census(
        supply_report=path, annotation_dir=ann, fetch_report=tmp_path / "absent.json"
    )
    assert census["n_failed"] == 1 and "Gff3Error" in census["failures"][0]["note"]


def test_parse_census_refuses_a_file_that_never_declares_itself_gff3(tmp_path):
    """The census must go through the STRICT entry point, not the lenient line parser.

    These nine columns parse perfectly well as feature lines; what they lack is any claim to
    be GFF3 at all — which is what stops an arbitrary TSV entering the annotation corpus.
    """
    import gzip as _gz

    undeclared = _gz.compress(b"ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=a;product=p\n")
    md5 = hashlib.md5(undeclared, usedforsecurity=False).hexdigest()
    url = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/790/315/"
        "GCA_002790315.1_ASM279031v1/GCA_002790315.1_ASM279031v1_genomic.gff.gz"
    )
    row = _annotated_row(accession=af.CONTROL_ACCESSION, url=url, md5=md5, n_bytes=len(undeclared))
    path = _write_supply(tmp_path, _minimal_supply([row], bytes_total=len(undeclared)))
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / af.destination_name(af.CONTROL_ACCESSION)).write_bytes(undeclared)

    census = af.parse_census(
        supply_report=path, annotation_dir=ann, fetch_report=tmp_path / "absent.json"
    )
    assert census["n_failed"] == 1 and "##gff-version" in census["failures"][0]["note"]


def test_the_census_uses_the_PARSERS_fasta_rule_and_not_a_looser_one_of_its_own(tmp_path):
    """``has_fasta_section`` must answer about the file the reader actually read.

    A census recognising ``##FASTA`` more loosely than :func:`gff3.iter_gff3_features` would
    report a FASTA section on a file the parser read straight through — and the corpus total
    ``n_with_fasta_section`` would then describe neither the parse nor the file. Driven with a
    lowercase near-miss followed by a real CDS: the shared rule says "no FASTA section, two CDS",
    and any looser census rule says "FASTA section" while the parse says otherwise.
    """
    import gzip as _gz

    doc = (
        b"##gff-version 3\n"
        b"ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=a;product=p\n"
        b"##fasta\n"
        b"ctg1\tx\tCDS\t30\t40\t.\t+\t0\tID=b;product=q\n"
    )
    payload = _gz.compress(doc)
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    row = _annotated_row(
        accession=af.CONTROL_ACCESSION, url=OTHER_URL, md5=md5, n_bytes=len(payload)
    )
    path = _write_supply(tmp_path, _minimal_supply([row], bytes_total=len(payload)))
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / af.destination_name(af.CONTROL_ACCESSION)).write_bytes(payload)

    census = af.parse_census(
        supply_report=path, annotation_dir=ann, fetch_report=tmp_path / "absent.json"
    )
    assert census["n_failed"] == 0
    assert census["per_assembly"][0]["n_cds"] == 2
    assert census["per_assembly"][0]["has_fasta_section"] is False
    assert census["totals"]["n_with_fasta_section"] == 0


def test_the_census_reports_a_fasta_section_declared_in_the_IMPLIED_form(tmp_path):
    """The census and the parser must agree on the *widened* rule too, not just the narrow one."""
    import gzip as _gz

    doc = b"##gff-version 3\nctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=a;product=p\n>ctg1\nACGT\n"
    payload = _gz.compress(doc)
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    row = _annotated_row(
        accession=af.CONTROL_ACCESSION, url=OTHER_URL, md5=md5, n_bytes=len(payload)
    )
    path = _write_supply(tmp_path, _minimal_supply([row], bytes_total=len(payload)))
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / af.destination_name(af.CONTROL_ACCESSION)).write_bytes(payload)

    census = af.parse_census(
        supply_report=path, annotation_dir=ann, fetch_report=tmp_path / "absent.json"
    )
    assert census["n_failed"] == 0
    assert census["per_assembly"][0]["n_cds"] == 1
    assert census["per_assembly"][0]["has_fasta_section"] is True
    assert census["totals"]["n_with_fasta_section"] == 1


def test_parse_census_reports_a_file_that_parses_to_zero_cds(tmp_path):
    """Hashing correctly is not the same as being usable by (c)."""
    import gzip as _gz

    empty = _gz.compress(b"##gff-version 3\n")
    md5 = hashlib.md5(empty, usedforsecurity=False).hexdigest()
    url = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/790/315/"
        "GCA_002790315.1_ASM279031v1/GCA_002790315.1_ASM279031v1_genomic.gff.gz"
    )
    row = _annotated_row(accession=af.CONTROL_ACCESSION, url=url, md5=md5, n_bytes=len(empty))
    path = _write_supply(tmp_path, _minimal_supply([row], bytes_total=len(empty)))
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / af.destination_name(af.CONTROL_ACCESSION)).write_bytes(empty)

    census = af.parse_census(
        supply_report=path, annotation_dir=ann, fetch_report=tmp_path / "absent.json"
    )
    assert census["n_failed"] == 1 and census["failures"][0]["note"] == "parsed to zero CDS"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_requires_a_subcommand(capsys):
    with pytest.raises(SystemExit):
        af.main([])


def test_cli_fetch_validates_against_the_REAL_targets_and_exits_0(tmp_path):
    """The CLI must hand the validator the supply report's own targets, not a placeholder.

    ``targets=[]`` or a hand-built stand-in at this call site would make the identity clause
    unfalsifiable for every production run while every unit test — which passes its own targets —
    stayed green. Driven through ``main`` with the file already cached, so the acquisition is
    entirely offline and the autouse no-network guard stays armed.
    """
    path, _url, payload = _control_supply(tmp_path)
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / af.destination_name(af.CONTROL_ACCESSION)).write_bytes(payload)
    out = tmp_path / "fetch.json"
    code = af.main(
        ["fetch", "--supply-report", str(path), "--annotation-dir", str(ann), "--out", str(out)]
    )
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["validation_problems"] == []
    assert written["n_cached"] == 1 and written["n_downloaded"] == 0


def test_cli_verify_exits_0_on_a_clean_corpus(tmp_path, capsys):
    path, ann, fetch_path, _payload = _acquire(tmp_path)
    code = af.main(
        [
            "verify",
            "--supply-report",
            str(path),
            "--annotation-dir",
            str(ann),
            "--fetch-report",
            str(fetch_path),
            "--out",
            str(tmp_path / "parse.json"),
        ]
    )
    assert code == 0
    written = json.loads((tmp_path / "parse.json").read_text(encoding="utf-8"))
    assert written["n_ok"] == 1 and written["corpus_matches_fetch_report"] is True


def test_cli_verify_exits_3_when_a_file_is_missing(tmp_path):
    path, _url, _payload = _control_supply(tmp_path)
    code = af.main(
        [
            "verify",
            "--supply-report",
            str(path),
            "--annotation-dir",
            str(tmp_path / "empty"),
            "--out",
            str(tmp_path / "parse.json"),
        ]
    )
    assert code == 3


def test_cli_exits_3_on_a_refusal_rather_than_raising(tmp_path):
    code = af.main(
        [
            "verify",
            "--supply-report",
            str(tmp_path / "absent.json"),
            "--annotation-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "parse.json"),
        ]
    )
    assert code == 3


# --------------------------------------------------------------------------- #
# Module hygiene
# --------------------------------------------------------------------------- #


def test_module_pins_no_scientific_value():
    """D4's 500 bp window belongs to ADR-0006 and P3-15′-c-ii, not to an acquisition step.

    Prose is exempt — the module docstring quotes D4 *as a citation*. What is asserted is that
    no public module-level attribute holds a number, so nothing here can be **read** as a
    value. Adding ``WINDOW_BP = 500`` turns this red. Mirrors the same gate on
    ``annotation_supply``; the two transport constants re-exported from ``flank_context`` are
    exempt and are checked against their originals, so a drifted local copy fails here too.
    """
    from tbox_finder.data import flank_context as fc

    reexported = {"NCBI_TIMEOUT_S", "RATE_LIMIT_NO_KEY"}
    numeric = {
        name: getattr(af, name)
        for name in dir(af)
        if not name.startswith("_")
        and isinstance(getattr(af, name), (int, float))
        and not isinstance(getattr(af, name), bool)
    }
    assert (
        set(numeric) <= reexported
    ), f"module-level numeric constant(s) with no ADR: {sorted(set(numeric) - reexported)}"
    for name in numeric:
        assert numeric[name] == getattr(fc, name)


def test_the_acquisition_opener_carries_the_redirect_allowlist():
    """Building a bare opener would leave the guard one hop deep on a 302."""
    opener = asup.build_allowlisted_opener()
    assert any(isinstance(h, asup._AllowlistRedirectHandler) for h in opener.handlers)


def test_build_allowlisted_opener_returns_a_fresh_instance_each_call():
    assert asup.build_allowlisted_opener() is not asup.build_allowlisted_opener()


def test_the_committed_fixture_is_intact_gzip_of_the_expected_size(monkeypatch):
    """Raised three times as `major` by a reviewer whose sandbox reported 59,693 bytes with a
    `1f ef` header — `1f ef` is `1f 8b` after a lossy text round-trip, i.e. a mangled binary.
    A clean clone has the real bytes; this asserts all three properties so any environment
    that *does* mangle it says which one broke instead of failing on a downstream digest."""
    raw = FIXTURE_GFF.read_bytes()
    assert raw[:2] == b"\x1f\x8b", "fixture is not gzip — a text filter has mangled it"
    assert len(raw) == 33003
    assert hashlib.md5(raw, usedforsecurity=False).hexdigest() == "3afa0aff910cfd08f9f0163981656308"

    # Positive control. Deleting an assertion can never fail its own test, so the guard's power
    # is shown against an input that SHOULD trip it: the fixture put through the lossy text
    # round-trip that produces the reviewer's `1f ef` header and inflated size.
    mangled = raw.decode("utf-8", "replace").encode("utf-8")
    assert mangled[:2] != b"\x1f\x8b"
    assert len(mangled) != 33003
    assert hashlib.md5(mangled, usedforsecurity=False).hexdigest() != (
        "3afa0aff910cfd08f9f0163981656308"
    )


def test_the_binary_GET_actually_uses_the_allowlisted_opener(monkeypatch):
    """Testing the *factory* proves nothing if the transport calls bare ``urlopen`` instead.

    Asserted behaviourally rather than by grepping the source, so a rename cannot make it
    vacuous: the opener this module builds must be the one the request goes through.
    """
    used: list[str] = []

    class _Spy:
        def open(self, req, timeout=None):
            used.append(req.full_url)
            raise _NetworkAccess(req.full_url)

    monkeypatch.setattr(af, "build_allowlisted_opener", lambda: _Spy())
    with pytest.raises(_NetworkAccess):
        _REAL_URLOPEN_BYTES(URL)
    assert used == [URL]


def test_the_binary_GET_refuses_a_disallowed_url_before_opening_anything(monkeypatch):
    """The guard must fire ahead of the opener — a ``file://`` read must never be attempted."""
    opened: list[str] = []

    class _Spy:
        def open(self, req, timeout=None):
            opened.append(req.full_url)
            raise AssertionError("opener reached with a disallowed URL")

    monkeypatch.setattr(af, "build_allowlisted_opener", lambda: _Spy())
    with pytest.raises(asup.AnnotationSupplyError):
        _REAL_URLOPEN_BYTES("file:///etc/passwd")
    assert opened == []
