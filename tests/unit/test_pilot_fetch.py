"""P2-10c′-b — unit tests for the ρ-pilot whole-genome fetch (ADR-0003 D6).

Two tiers, mirroring ``test_flank_context``:

* **bare (stdlib only)** — the pure resolution/parse/normalize/status/digest helpers and the
  fail-closed :func:`validate_report`. These run in bare CI (no biopython, no pandas, no
  network), which is where the §10.3 honesty invariants must be enforced.
* **offline end-to-end** — ``build_pilot_fetch`` driven by a *fake* Entrez client (canned
  esearch/esummary) and a *fake* ``urlopen`` (canned gzip), so the resolution → download →
  gunzip → normalize → write → audit pipeline, its resumability, and its fail-closed raise
  are all exercised with no network. Guarded by ``importorskip`` so collection never breaks.

The validator tests are deliberate *bite* tests: each mutates one field of an otherwise
valid report and asserts :func:`validate_report` complains. The must-fire pair the step
turns on: :func:`test_validate_rejects_fabricated_ok` (a genome marked ``ok`` its evidence
does not support — the P1-15/P1-16 lesson that ``all(clauses)`` cannot catch a clause
fabricated TRUE) and :func:`test_validate_rejects_below_min_phyla` (the divergence span the
selection guaranteed silently lost to fetch failures).
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from tbox_finder.mining import pilot_fetch as pf

# --------------------------------------------------------------------------- #
# Fake Entrez (esearch/esummary) + fake urlopen (gzip) — offline
# --------------------------------------------------------------------------- #


class _Perm404(Exception):
    """An accession-specific (permanent) HTTP 4xx rejection, e.g. a 404 on the genome file."""

    code = 404


class _Transient503(Exception):
    """A transient HTTP 5xx."""

    code = 503


def _default_fasta(acc: str) -> str:
    return f">{acc}_c1 chromosome\n" + "ACGT" * 30 + "\n"


class FakeEntrez:
    """Drive esearch + esummary from a per-accession spec dict, with no network.

    ``spec[accession]`` may set: ``uid`` (assembly UID str, or absent ⇒ no_assembly);
    ``ftp`` (the esummary FtpPath — default a fake path derived from the accession; an
    explicit ``""`` ⇒ no_ftp_path); ``esearch_exc`` / ``esummary_exc`` (exceptions to raise).
    The genome bytes are served by the fake ``urlopen`` (see :func:`_make_fake_urlopen`).
    """

    def __init__(self, spec: dict[str, dict[str, Any]]) -> None:
        self.spec = spec
        self._uid_to_acc = {v["uid"]: k for k, v in spec.items() if v.get("uid")}
        self.calls: dict[str, int] = {"esearch": 0, "esummary": 0}

    def esearch(self, db: str, term: str):  # noqa: A002 - mirror Entrez signature
        self.calls["esearch"] += 1
        acc = term.split("[")[0]
        s = self.spec[acc]
        if s.get("esearch_exc"):
            raise s["esearch_exc"]
        return {"IdList": ([s["uid"]] if s.get("uid") else [])}

    def esummary(self, db: str, id: str):  # noqa: A002
        self.calls["esummary"] += 1
        acc = self._uid_to_acc[id]
        s = self.spec[acc]
        if s.get("esummary_exc"):
            raise s["esummary_exc"]
        ftp = s.get("ftp", f"ftp://fake/pub/{acc}_ASMx")
        ds: dict[str, str] = {}
        if ftp:
            key = "FtpPath_RefSeq" if acc.startswith("GCF_") else "FtpPath_GenBank"
            ds[key] = ftp
        return {"DocumentSummarySet": {"DocumentSummary": [ds]}}


class _FakeResp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def _make_fake_urlopen(spec: dict[str, dict[str, Any]]):
    def fake_urlopen(url, timeout=None):  # noqa: ANN001 - mirror urlopen
        acc = re.search(r"GC[AF]_\d+\.\d+", str(url)).group()  # type: ignore[union-attr]
        s = spec[acc]
        if s.get("download_exc"):
            raise s["download_exc"]
        if "raw_gz" in s:  # allow a deliberately corrupt/truncated gzip
            return _FakeResp(s["raw_gz"])
        fasta = s.get("fasta", _default_fasta(acc))
        return _FakeResp(gzip.compress(fasta.encode()))

    return fake_urlopen


def _patch_entrez(monkeypatch, spec: dict[str, dict[str, Any]]) -> FakeEntrez:
    fake = FakeEntrez(spec)
    monkeypatch.setattr(pf, "_entrez", lambda email, api_key: fake)
    # _read_xml is the only place Bio.Entrez.read runs; the fake returns parsed structures.
    monkeypatch.setattr(pf, "_read_xml", lambda handle: handle)
    monkeypatch.setattr("urllib.request.urlopen", _make_fake_urlopen(spec))
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)  # no real backoff sleeps
    return fake


def _write_manifest(path: Path, accessions: list[tuple[str, str, str]]) -> None:
    import pandas as pd

    pd.DataFrame(
        [{"assembly_accession": a, "domain": d, "phylum": p} for a, d, p in accessions]
    ).to_parquet(path, index=False)


def _manifest_of(tmp_path: Path, accs: list[tuple[str, str, str]]) -> Path:
    p = tmp_path / "pilot.parquet"
    _write_manifest(p, accs)
    return p


# --------------------------------------------------------------------------- #
# Bare-CI: pure resolution / parse / status helpers
# --------------------------------------------------------------------------- #


def test_parse_esearch_uids() -> None:
    assert pf.parse_esearch_uids({"IdList": ["1", "2"]}) == ["1", "2"]
    assert pf.parse_esearch_uids({"IdList": []}) == []
    assert pf.parse_esearch_uids({}) == []
    assert pf.parse_esearch_uids({"IdList": "notalist"}) == []


def test_parse_assembly_docsum() -> None:
    rec = {"DocumentSummarySet": {"DocumentSummary": [{"FtpPath_GenBank": "ftp://x/y"}]}}
    assert pf.parse_assembly_docsum(rec) == {"FtpPath_GenBank": "ftp://x/y"}
    assert pf.parse_assembly_docsum({"DocumentSummarySet": {"DocumentSummary": []}}) is None
    assert pf.parse_assembly_docsum({}) is None
    assert pf.parse_assembly_docsum("nope") is None  # type: ignore[arg-type]


def test_assembly_ftp_url_by_source_and_https() -> None:
    gca = {
        "FtpPath_GenBank": "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/009/842/505/GCA_009842505.1_ASM984250v1"
    }
    url = pf.assembly_ftp_url(gca, "GCA_009842505.1")
    assert url == (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/009/842/505/"
        "GCA_009842505.1_ASM984250v1/GCA_009842505.1_ASM984250v1_genomic.fna.gz"
    )
    # a RefSeq accession uses its OWN FtpPath_RefSeq, not GenBank
    gcf = {"FtpPath_GenBank": "ftp://host/gb/X", "FtpPath_RefSeq": "ftp://host/rs/GCF_1.1_ASM"}
    assert pf.assembly_ftp_url(gcf, "GCF_1.1").endswith("GCF_1.1_ASM/GCF_1.1_ASM_genomic.fna.gz")
    assert "rs" in pf.assembly_ftp_url(gcf, "GCF_1.1")


def test_assembly_ftp_url_empty_when_no_path() -> None:
    assert pf.assembly_ftp_url({}, "GCA_1.1") == ""
    assert pf.assembly_ftp_url({"FtpPath_GenBank": ""}, "GCA_1.1") == ""
    assert pf.assembly_ftp_url(None, "GCA_1.1") == ""


def test_iter_fasta_records_and_total_bp() -> None:
    txt = ">CP1 chr\nACGTacgt\nNNNN\n>CP2 plasmid\nGGGGCCCC\n"
    recs = pf.iter_fasta_records(txt)
    assert recs == [("CP1 chr", "ACGTACGTNNNN"), ("CP2 plasmid", "GGGGCCCC")]
    assert pf.fasta_total_bp(recs) == 20
    assert pf.iter_fasta_records("junk\n>H\n") == [("H", "")]  # pre-'>' junk ignored
    assert pf.iter_fasta_records("") == []


def test_normalize_fasta_deterministic_and_wraps() -> None:
    recs = [("CP1 chr", "ACGTACGTNNNN"), ("CP2 plasmid", "GGGGCCCC")]
    out = pf.normalize_fasta(recs, wrap=4)
    assert out == ">CP1 chr\nACGT\nACGT\nNNNN\n>CP2 plasmid\nGGGG\nCCCC\n"
    assert pf.normalize_fasta(recs, wrap=4) == out  # byte-identical on repeat
    with pytest.raises(ValueError):
        pf.normalize_fasta(recs, wrap=0)


def test_genome_status_evidence_chain() -> None:
    s = pf.genome_status
    ok = dict(
        esearch_ok=True,
        uid_found=True,
        esummary_ok=True,
        ftp_found=True,
        download_ok=True,
        total_bp=100,
    )
    assert s(**ok) == pf.STATUS_OK
    assert s(**{**ok, "uid_found": False}) == pf.STATUS_NO_ASSEMBLY
    assert s(**{**ok, "ftp_found": False}) == pf.STATUS_NO_FTP_PATH
    assert s(**{**ok, "esearch_ok": False}) == pf.STATUS_FETCH_FAILED
    assert s(**{**ok, "esummary_ok": False}) == pf.STATUS_FETCH_FAILED
    assert s(**{**ok, "total_bp": 0}) == pf.STATUS_FETCH_FAILED  # fetched but empty ⇒ transient
    assert s(**{**ok, "permanent_fail": True}) == pf.STATUS_UNAVAILABLE  # dominates


def test_genome_digest_order_independent_and_seq_sensitive() -> None:
    rows = [_ok_row(i) for i in range(4)]
    d = pf.genome_digest(rows)
    assert pf.genome_digest(list(reversed(rows))) == d  # order-independent
    mutated = copy.deepcopy(rows)
    mutated[0]["seq_sha256"] = "f" * 64
    assert pf.genome_digest(mutated) != d  # a changed sequence changes the digest


def test_derive_status_counts_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        pf.derive_status_counts([{"status": "banana"}])


# --------------------------------------------------------------------------- #
# Bare-CI: the fail-closed validator (bite tests)
# --------------------------------------------------------------------------- #


def _ok_row(i: int) -> dict[str, Any]:
    return {
        "assembly_accession": f"GCA_{i:07d}.1",
        "domain": "Bacteria" if i % 5 else "Archaea",
        "phylum": f"P{i}",
        "status": pf.STATUS_OK,
        "assembly_uid": str(1000 + i),
        "source_url": f"https://x/GCA_{i:07d}.1_A/GCA_{i:07d}.1_A_genomic.fna.gz",
        "n_replicons": 2,
        "total_bp": 1000 + i,
        "seq_sha256": hashlib.sha256(f"g{i}".encode()).hexdigest(),
        "fasta_path": f"data/interim/pilot_genomes/GCA_{i:07d}.1.fna",
    }


def _fail_row(i: int, status: str = pf.STATUS_NO_ASSEMBLY) -> dict[str, Any]:
    return {
        "assembly_accession": f"GCA_{9000000 + i}.1",
        "domain": "Archaea",
        "phylum": f"F{i}",
        "status": status,
        "assembly_uid": "",
        "source_url": "",
        "n_replicons": 0,
        "total_bp": 0,
        "seq_sha256": "",
        "fasta_path": "",
    }


def _valid_report(n_ok: int = 58, n_fail: int = 2) -> dict[str, Any]:
    rows = [_ok_row(i) for i in range(n_ok)] + [_fail_row(i) for i in range(n_fail)]
    return pf.build_report(rows, accessed="2026-07-22", errors=["some error"])


def test_valid_report_roundtrip() -> None:
    rep = _valid_report()
    assert pf.validate_report(rep) == []
    assert rep["n_ok"] == 58 and rep["n_genomes"] == 60
    assert rep["n_phyla_spanned_ok"] == 58
    assert rep["total_bp"] == sum(1000 + i for i in range(58))


def test_validate_rejects_fabricated_ok() -> None:
    """MUST-FIRE: a genome marked ok whose evidence (0 bp, 0 replicons) says it failed."""
    rep = _valid_report()
    rep["per_genome"][58]["status"] = pf.STATUS_OK  # a _fail_row, still 0 bp / 0 replicons
    problems = pf.validate_report(rep)
    assert any("total_bp <= 0" in p or "n_replicons <= 0" in p for p in problems), problems


def test_validate_rejects_inflated_success_rate() -> None:
    rep = _valid_report()
    rep["success_rate"] = 0.999
    assert any("success_rate" in p for p in pf.validate_report(rep))


def test_validate_rejects_inflated_total_bp() -> None:
    rep = _valid_report()
    rep["total_bp"] += 1000
    assert any("total_bp" in p for p in pf.validate_report(rep))


def test_validate_rejects_stale_status_counts() -> None:
    rep = _valid_report()
    rep["status_counts"][pf.STATUS_OK] += 1
    assert any("status_counts" in p for p in pf.validate_report(rep))


def test_validate_rejects_flipped_honesty_flag() -> None:
    for flag in ("rho_measured", "candidates_counted", "is_science", "sequences_synthetic"):
        rep = _valid_report()
        rep[flag] = True
        assert any(flag in p for p in pf.validate_report(rep)), flag


def test_validate_rejects_tampered_digest() -> None:
    rep = _valid_report()
    rep["digest"] = "0" * 64
    assert any("digest" in p for p in pf.validate_report(rep))


def test_validate_rejects_below_min_phyla() -> None:
    """MUST-FIRE: fetch failures eroded the divergence span below the floor."""
    rep = _valid_report(n_ok=pf.MIN_PHYLA_OK - 1, n_fail=0)
    assert any("MIN_PHYLA_OK" in p for p in pf.validate_report(rep))


def test_validate_rejects_below_min_success_rate() -> None:
    # 55 ok + 10 failed = 0.846 < 0.90; phyla stays >= floor so the success-rate clause bites
    rep = _valid_report(n_ok=55, n_fail=10)
    assert any("MIN_SUCCESS_RATE" in p for p in pf.validate_report(rep))


def test_validate_rejects_nonok_row_carrying_bp() -> None:
    rep = _valid_report()
    rep["per_genome"][58]["total_bp"] = 5  # a failure row must carry zero geometry
    assert any("total_bp != 0" in p for p in pf.validate_report(rep))


def test_validate_rejects_bad_source_block() -> None:
    rep = _valid_report()
    rep["source"]["genome_ftp_host"] = "evil.example.org"
    assert any("source.genome_ftp_host" in p for p in pf.validate_report(rep))
    rep = _valid_report()
    rep["source"]["accessed"] = ""
    assert any("accessed" in p for p in pf.validate_report(rep))


# --------------------------------------------------------------------------- #
# Offline end-to-end: build_pilot_fetch with a fake Entrez + fake urlopen
# --------------------------------------------------------------------------- #


def _relax_floors(monkeypatch) -> None:
    """Plumbing tests use a small fixture; the real floors are exercised by the validator
    bite tests above, so relax them here to isolate the orchestration."""
    monkeypatch.setattr(pf, "MIN_PHYLA_OK", 1)
    monkeypatch.setattr(pf, "MIN_SUCCESS_RATE", 0.5)


def test_load_manifest_rejects_bad_accessions(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    p = tmp_path / "m.parquet"
    _write_manifest(p, [("GCA_1.1", "Bacteria", "P1"), ("bogus", "Bacteria", "P2")])
    with pytest.raises(ValueError, match="non-GC"):
        pf.load_manifest(p)
    _write_manifest(p, [("GCA_1.1", "Bacteria", "P1"), ("GCA_1.1", "Bacteria", "P2")])
    with pytest.raises(ValueError, match="duplicate"):
        pf.load_manifest(p)


def test_build_end_to_end_with_fake(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _relax_floors(monkeypatch)
    accs = [(f"GCA_{i:07d}.1", "Bacteria", f"P{i}") for i in range(4)]
    accs.append(("GCF_0000009.1", "Archaea", "PA"))  # a RefSeq genome
    spec = {a: {"uid": str(100 + i)} for i, (a, _, _) in enumerate(accs)}
    fake = _patch_entrez(monkeypatch, spec)

    manifest = tmp_path / "pilot.parquet"
    _write_manifest(manifest, accs)
    gdir = tmp_path / "genomes"
    report_path = tmp_path / "report.json"
    prov_path = tmp_path / "prov.json"

    rep = pf.build_pilot_fetch(
        manifest_parquet=manifest,
        genome_dir=gdir,
        log_path=tmp_path / "log.jsonl",
        report_path=report_path,
        provenance_path=prov_path,
        email="me@example.org",
    )
    assert rep["n_ok"] == 5 and rep["n_genomes"] == 5
    assert pf.validate_report(rep) == []
    assert fake.calls["esearch"] == 5 and fake.calls["esummary"] == 5
    for a, _, _ in accs:
        fna = gdir / f"{a}.fna"
        assert fna.is_file() and fna.read_text().startswith(">")
    written = json.loads(report_path.read_text())
    assert pf.validate_report(written) == []
    prov = json.loads(prov_path.read_text())
    assert prov["extra"]["digest"] == rep["digest"]
    # the RefSeq genome resolved via its own FtpPath_RefSeq
    urls = {r["assembly_accession"]: r["source_url"] for r in rep["per_genome"]}
    assert urls["GCF_0000009.1"].endswith("_genomic.fna.gz")


def test_build_is_resumable(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _relax_floors(monkeypatch)
    accs = [(f"GCA_{i:07d}.1", "Bacteria", f"P{i}") for i in range(3)]
    spec = {a: {"uid": str(200 + i)} for i, (a, _, _) in enumerate(accs)}
    fake = _patch_entrez(monkeypatch, spec)

    manifest = tmp_path / "pilot.parquet"
    _write_manifest(manifest, accs)
    gdir = tmp_path / "genomes"
    kw = dict(
        manifest_parquet=manifest,
        genome_dir=gdir,
        log_path=tmp_path / "log.jsonl",
        report_path=tmp_path / "report.json",
        provenance_path=tmp_path / "prov.json",
        email="me@example.org",
    )
    pf.build_pilot_fetch(**kw)
    first = fake.calls["esummary"]
    assert first == 3

    pf.build_pilot_fetch(**kw)  # every genome cached + hashes match ⇒ no new resolve
    assert fake.calls["esummary"] == first

    (gdir / "GCA_0000000.1.fna").write_text(">tampered\nAAAA\n")  # hash no longer matches
    pf.build_pilot_fetch(**kw)
    assert fake.calls["esummary"] == first + 1  # exactly that one re-fetched


def test_build_fails_closed_when_degraded(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _relax_floors(monkeypatch)  # floor 0.5; make 3 of 4 fail ⇒ 0.25 < 0.5
    accs = [(f"GCA_{i:07d}.1", "Bacteria", f"P{i}") for i in range(4)]
    spec = {accs[0][0]: {"uid": "300"}}  # only genome 0 resolves
    for a, _, _ in accs[1:]:
        spec[a] = {}  # esearch returns empty IdList ⇒ no_assembly (deterministic, no retry)
    _patch_entrez(monkeypatch, spec)

    report_path = tmp_path / "report.json"
    prov_path = tmp_path / "prov.json"
    with pytest.raises(ValueError, match="failed validation"):
        pf.build_pilot_fetch(
            manifest_parquet=_manifest_of(tmp_path, accs),
            genome_dir=tmp_path / "genomes",
            log_path=tmp_path / "log.jsonl",
            report_path=report_path,
            provenance_path=prov_path,
            email="me@example.org",
        )
    assert not report_path.exists() and not prov_path.exists()  # nothing certified


def test_build_records_no_assembly_no_ftp_and_unavailable(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _relax_floors(monkeypatch)
    monkeypatch.setattr(pf, "MIN_SUCCESS_RATE", 0.3)  # 2 ok of 5 = 0.4; this test checks statuses
    accs = [
        ("GCA_0000000.1", "Bacteria", "P0"),  # ok
        ("GCA_0000001.1", "Bacteria", "P1"),  # ok
        ("GCA_0000002.1", "Archaea", "P2"),  # no_assembly (empty IdList)
        ("GCA_0000003.1", "Archaea", "P3"),  # no_ftp_path (esummary carries no FtpPath)
        ("GCA_0000004.1", "Archaea", "P4"),  # unavailable (permanent 404 on download)
    ]
    spec = {
        "GCA_0000000.1": {"uid": "400"},
        "GCA_0000001.1": {"uid": "401"},
        "GCA_0000002.1": {},
        "GCA_0000003.1": {"uid": "403", "ftp": ""},
        "GCA_0000004.1": {"uid": "404", "download_exc": _Perm404()},
    }
    _patch_entrez(monkeypatch, spec)
    rep = pf.build_pilot_fetch(
        manifest_parquet=_manifest_of(tmp_path, accs),
        genome_dir=tmp_path / "genomes",
        log_path=tmp_path / "log.jsonl",
        report_path=tmp_path / "report.json",
        provenance_path=tmp_path / "prov.json",
        email="me@example.org",
    )
    by = {r["assembly_accession"]: r["status"] for r in rep["per_genome"]}
    assert by["GCA_0000002.1"] == pf.STATUS_NO_ASSEMBLY
    assert by["GCA_0000003.1"] == pf.STATUS_NO_FTP_PATH
    assert by["GCA_0000004.1"] == pf.STATUS_UNAVAILABLE
    assert rep["status_counts"][pf.STATUS_OK] == 2


def test_truncated_gzip_is_retried_then_failed(tmp_path: Path, monkeypatch) -> None:
    """A truncated/corrupt gzip must NOT be accepted as a partial genome — gzip's CRC raises,
    so the download retries and, if persistent, becomes STATUS_FETCH_FAILED (never a partial
    ok with understated total_bp)."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _relax_floors(monkeypatch)
    accs = [("GCA_0000000.1", "Bacteria", "P0"), ("GCA_0000001.1", "Bacteria", "P1")]
    good_gz = gzip.compress((">c\n" + "ACGT" * 30 + "\n").encode())
    spec = {
        "GCA_0000000.1": {"uid": "500"},
        "GCA_0000001.1": {"uid": "501", "raw_gz": good_gz[: len(good_gz) // 2]},  # truncated
    }
    _patch_entrez(monkeypatch, spec)
    rep = pf.build_pilot_fetch(
        manifest_parquet=_manifest_of(tmp_path, accs),
        genome_dir=tmp_path / "genomes",
        log_path=tmp_path / "log.jsonl",
        report_path=tmp_path / "report.json",
        provenance_path=tmp_path / "prov.json",
        email="me@example.org",
    )
    by = {r["assembly_accession"]: r for r in rep["per_genome"]}
    assert by["GCA_0000001.1"]["status"] == pf.STATUS_FETCH_FAILED
    assert not (tmp_path / "genomes" / "GCA_0000001.1.fna").exists()  # no partial file written


def test_multi_replicon_written_in_sorted_order(tmp_path: Path, monkeypatch) -> None:
    """A multi-replicon genome sums every replicon's bp exactly and writes them in canonical
    (accession-sorted) order regardless of NCBI's FASTA record order (findings 5, 6)."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _relax_floors(monkeypatch)
    accs = [("GCA_0000000.1", "Bacteria", "P0")]
    # records deliberately unsorted (z3, z1, z2) to prove the on-disk order is canonicalized
    fasta = (
        ">z3 plasmid\n" + "G" * 20 + "\n>z1 chr\n" + "A" * 10 + "\n>z2 plasmid\n" + "C" * 15 + "\n"
    )
    spec = {"GCA_0000000.1": {"uid": "700", "fasta": fasta}}
    _patch_entrez(monkeypatch, spec)
    gdir = tmp_path / "genomes"
    rep = pf.build_pilot_fetch(
        manifest_parquet=_manifest_of(tmp_path, accs),
        genome_dir=gdir,
        log_path=tmp_path / "log.jsonl",
        report_path=tmp_path / "report.json",
        provenance_path=tmp_path / "prov.json",
        email="me@example.org",
    )
    row = rep["per_genome"][0]
    assert row["n_replicons"] == 3
    assert row["total_bp"] == 10 + 15 + 20  # exact, not >=
    headers = [h for h, _ in pf.iter_fasta_records((gdir / "GCA_0000000.1.fna").read_text())]
    assert headers == ["z1 chr", "z2 plasmid", "z3 plasmid"]  # accession-sorted


def test_build_rejects_orphan_fasta(tmp_path: Path, monkeypatch) -> None:
    """A stale .fna from a prior manifest, not in the current certified ok set, must block
    certification (it would ride along in the DVC artifact uncertified by the digest)."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _relax_floors(monkeypatch)
    accs = [("GCA_0000000.1", "Bacteria", "P0"), ("GCA_0000001.1", "Bacteria", "P1")]
    spec = {a: {"uid": str(800 + i)} for i, (a, _, _) in enumerate(accs)}
    _patch_entrez(monkeypatch, spec)
    gdir = tmp_path / "genomes"
    gdir.mkdir(parents=True)
    (gdir / "GCA_9999999.1.fna").write_text(">orphan\nACGT\n")  # a genome from a prior run
    report_path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="does not match the certified set"):
        pf.build_pilot_fetch(
            manifest_parquet=_manifest_of(tmp_path, accs),
            genome_dir=gdir,
            log_path=tmp_path / "log.jsonl",
            report_path=report_path,
            provenance_path=tmp_path / "prov.json",
            email="me@example.org",
        )
    assert not report_path.exists()  # nothing certified
