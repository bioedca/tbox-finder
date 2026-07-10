"""Unit tests for the P0-18 dataset-license-compatibility audit.

Verify the **audit contract** (PRD header / §2.1 G-E; ADR-0001 §D9): every upstream source
carries a verified license + a compatibility verdict; the encumbered sources
(share-alike / proprietary) route to reconstruct-from-source and are never embedded; the
resolved release path is the CC-BY-4.0 own-derived default consistent with ADR-0001. The
registry in ``scripts/reconstruct_encumbered.py`` is the single source of truth; the human
audit ``docs/license_audit.md`` must name every source and state the resolution.

Network-free: the reconstruct plan functions read the committed
``data/external/**/provenance.json`` pins (present in the repo via the CLAUDE.md §5.2
carve-out), never the network — so this runs in the bare CI test env.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "reconstruct_encumbered.py"
_AUDIT_DOC = REPO_ROOT / "docs" / "license_audit.md"

_spec = importlib.util.spec_from_file_location("reconstruct_encumbered", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve the module via sys.modules (py3.12/3.13).
sys.modules["reconstruct_encumbered"] = mod
_spec.loader.exec_module(mod)


# The audit is complete iff it covers exactly this set (imp.md P0-18 Inputs: every upstream
# source — TBDB, Rfam, crystals, Vitreschak, GTDB, RefSeq, sourced anchor/class-II sets — plus
# the class-II CM, the sub-element CMs, and the NCBI taxdump from PRD §7.1).
EXPECTED_SOURCE_IDS = {
    "tbdb_master",
    "rfam_rf00230",
    "tbox_scan_cm",
    "tboxevo_cms",
    "pdb_crystals",
    "vitreschak_supplement",
    "gtdb",
    "ncbi_refseq",
    "ncbi_taxonomy",
    "anchor_p0_16",
    "classII_p0_17",
}

VALID_VERDICTS = {mod.COMPATIBLE, mod.ENCUMBERED}
VALID_ROUTES = {mod.REDISTRIBUTE, mod.RECONSTRUCT}


def test_registry_covers_every_upstream_source() -> None:
    """The audit is exactly complete — no source is missing, none is stray."""
    got = {s.source_id for s in mod.SOURCES}
    missing = EXPECTED_SOURCE_IDS - got
    extra = got - EXPECTED_SOURCE_IDS
    assert got == EXPECTED_SOURCE_IDS, f"audit incomplete/drifted: missing {missing}, extra {extra}"


def test_source_ids_are_unique() -> None:
    ids = [s.source_id for s in mod.SOURCES]
    assert len(ids) == len(set(ids)), f"duplicate source_id in the registry: {ids}"


@pytest.mark.parametrize("s", mod.SOURCES, ids=[s.source_id for s in mod.SOURCES])
def test_every_source_has_a_license_and_a_verdict(s) -> None:
    """The core gate: every upstream source carries a license + a compatibility verdict."""
    assert s.upstream_license.strip(), f"{s.source_id}: empty license"
    assert s.license_spdx.strip(), f"{s.source_id}: empty license_spdx"
    assert s.verdict in VALID_VERDICTS, f"{s.source_id}: bad verdict {s.verdict!r}"
    assert s.route in VALID_ROUTES, f"{s.source_id}: bad route {s.route!r}"
    assert s.provenance_url.startswith("http"), f"{s.source_id}: no provenance URL"
    assert s.accessed, f"{s.source_id}: no license-verification date"


@pytest.mark.parametrize("s", mod.SOURCES, ids=[s.source_id for s in mod.SOURCES])
def test_verdict_route_invariants(s) -> None:
    """Encumbered ⇒ reconstruct-from-source AND not embedded; compatible ⇒ redistribute."""
    if s.verdict == mod.ENCUMBERED:
        assert s.route == mod.RECONSTRUCT, f"{s.source_id}: encumbered but route={s.route}"
        assert (
            not s.redistributed_in_release
        ), f"{s.source_id}: encumbered source must NOT be embedded in the release"
        assert (
            s.source_id in mod.RECONSTRUCTORS
        ), f"{s.source_id}: encumbered but no reconstructor registered"
    else:
        assert s.route == mod.REDISTRIBUTE, f"{s.source_id}: compatible but route={s.route}"


def test_the_only_encumbered_sources_are_gtdb_and_vitreschak() -> None:
    """Pin the audit's central finding: exactly two sources force the reconstruct route."""
    enc = {s.source_id for s in mod.encumbered_sources()}
    assert enc == {"gtdb", "vitreschak_supplement"}, f"unexpected encumbered set: {enc}"
    # ...and they are the only reconstruct targets.
    assert set(mod.RECONSTRUCTORS) == enc


def test_share_alike_and_proprietary_are_flagged_encumbered() -> None:
    """A CC-BY-SA (copyleft) or proprietary source can never be silently marked compatible."""
    for s in mod.SOURCES:
        spdx = s.license_spdx.lower()
        if "sa" in spdx.split("-") or spdx == "proprietary":
            assert (
                s.verdict == mod.ENCUMBERED
            ), f"{s.source_id}: {s.license_spdx} must be encumbered"


def test_resolved_release_path_is_the_ccby4_default() -> None:
    """The release path is decided and consistent with ADR-0001 §D9."""
    assert mod.RELEASE_LICENSE == "CC-BY-4.0"


# --- audit document cross-checks -------------------------------------------------------


def test_audit_doc_exists_and_names_every_source() -> None:
    assert _AUDIT_DOC.is_file(), f"missing audit doc: {_AUDIT_DOC}"
    text = _AUDIT_DOC.read_text(encoding="utf-8")
    for sid in EXPECTED_SOURCE_IDS:
        assert sid in text, f"docs/license_audit.md does not mention source {sid!r}"


def test_audit_doc_states_resolution_and_reconstruct_route() -> None:
    text = _AUDIT_DOC.read_text(encoding="utf-8")
    assert "CC-BY-4.0" in text, "audit doc must state the CC-BY-4.0 resolution"
    assert "CC-BY-SA-4.0" in text, "audit doc must name the GTDB share-alike license"
    assert "reconstruct_encumbered.py" in text, "audit doc must reference the reconstruct script"
    # ADR-0001 §D9 consistency — own-derived default is named.
    assert "own-derived" in text.lower()


# --- reconstruct plans are network-free and reference the pinned provenance -------------


def test_reconstruct_gtdb_plan_is_offline_and_pinned() -> None:
    plan = mod.reconstruct_gtdb(download=False)
    assert plan["source_id"] == "gtdb"
    assert plan["license"] and "CC-BY-SA-4.0" in plan["license"]
    assert plan["files"], "gtdb reconstruct plan lists no files"
    for f in plan["files"]:
        assert f["url"].startswith("https://data.gtdb.ecogenomic.org/"), f
        assert f.get("md5"), f"gtdb file has no pinned MD5: {f}"
    assert plan["downloaded"] == []  # plan-only: nothing fetched


def test_reconstruct_vitreschak_plan_is_offline_and_pinned() -> None:
    plan = mod.reconstruct_vitreschak(download=False)
    assert plan["source_id"] == "vitreschak_supplement"
    assert "RNA Society" in (plan["license"] or "")
    assert plan["files"], "vitreschak reconstruct plan lists no files"
    for f in plan["files"]:
        assert f["url"].startswith("http"), f
        assert f.get("sha256"), f"vitreschak file has no pinned SHA-256: {f}"
    assert plan["downloaded"] == []


def test_download_requires_dest() -> None:
    with pytest.raises(ValueError):
        mod.reconstruct_gtdb(download=True)  # no dest → fail loud, no network
