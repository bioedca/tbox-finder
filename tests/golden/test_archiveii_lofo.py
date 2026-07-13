"""Golden regression for the P1-12 ArchiveII nine-family LOFO benchmark.

Two independent digest paths lock the parse:

* the module's :func:`build_lofo` on the committed mini fam-fold fixture, and
* an independent stdlib re-parse of the same .ct files (a different code path).

Both must equal the committed ``expected.sha256``. The test also validates the
committed ``data/external/archiveii_lofo/provenance.json`` (the immutable-download
manifest) and the sourced ``reports/p1/rinalmo_published_target.json`` — the two
committed artifacts P1-13 depends on. Pure stdlib → runs in bare CI without the
34 785-file production archive (the ``ingest_sample`` reuse-a-slice precedent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder import ingest
from tbox_finder.eval import archiveii_lofo as al

_REPO = Path(__file__).resolve().parents[2]
_FIX = _REPO / "tests/fixtures/archiveii_lofo"
_MINI = _FIX / "mini/ct/fam-fold"
_EXPECTED = _FIX / "expected.sha256"
_PROV = _REPO / "data/external/archiveii_lofo/provenance.json"
_TARGET = _REPO / "reports/p1/rinalmo_published_target.json"


# --------------------------------------------------------------------------- #
# fixture presence
# --------------------------------------------------------------------------- #
def test_fixture_present() -> None:
    assert _MINI.is_dir()
    assert _EXPECTED.is_file()
    assert len(_EXPECTED.read_text().strip()) == 64
    assert len(list(_MINI.glob("*/test/*.ct"))) == 18  # 9 families × 2


# --------------------------------------------------------------------------- #
# golden digest — module path
# --------------------------------------------------------------------------- #
def test_build_lofo_digest_matches_expected() -> None:
    manifest = al.build_lofo(_MINI)
    al.validate_manifest(manifest, require_full=False)
    assert manifest.n_records == 18
    assert set(manifest.families) == set(al.FAMILY_ORDER)
    assert all(c == 2 for c in manifest.per_family_counts.values())
    assert manifest.lofo_digest == _EXPECTED.read_text().strip()


def _independent_stdlib_digest() -> str:
    """Re-derive the digest with a minimal, independent .ct parser."""
    records = []
    for dirname, family in al.FAMILY_DIRS.items():
        for ct in sorted((_MINI / dirname / "test").glob("*.ct")):
            lines = [ln for ln in ct.read_text().splitlines() if ln.strip()]
            seq_chars, pairs = [], set()
            for ln in lines[1:]:
                cols = ln.split()
                i, base, j = int(cols[0]), cols[1].upper(), int(cols[4])
                seq_chars.append(base)
                if j > 0:  # each pair added once via the canonical (min,max) key
                    pairs.add((min(i, j), max(i, j)))
            key = ";".join(f"{i}-{j}" for i, j in sorted(pairs))
            records.append((family, ct.stem, "".join(seq_chars), key))
    # sort by (family order, record_id) — mirror iter_test_records
    rank = {f: n for n, f in enumerate(al.FAMILY_ORDER)}
    records.sort(key=lambda r: (rank[r[0]], r[1]))
    per = [ingest.record_hash([rid, fam, seq, key]) for fam, rid, seq, key in records]
    return ingest.records_digest(per)


def test_independent_stdlib_digest_matches_expected() -> None:
    assert _independent_stdlib_digest() == _EXPECTED.read_text().strip()


# --------------------------------------------------------------------------- #
# committed provenance.json (data/external/ immutable-download manifest)
# --------------------------------------------------------------------------- #
def test_provenance_committed_and_consistent() -> None:
    prov = json.loads(_PROV.read_text())
    assert prov["source"]["sha256"] == al.SPLITS_SHA256
    assert prov["source"]["url"] == al.SPLITS_URL
    lofo = prov["lofo"]
    assert lofo["n_families"] == al.NUM_FAMILIES
    assert lofo["n_records"] == al.EXPECTED_TOTAL_RECORDS
    assert len(lofo["lofo_digest"]) == 64
    assert set(lofo["per_family_counts"]) == set(al.FAMILY_ORDER)
    assert sum(lofo["per_family_counts"].values()) == al.EXPECTED_TOTAL_RECORDS
    # every fold: test == that family; train+valid == the held-out complement
    for fam, counts in lofo["per_fold_counts"].items():
        assert counts["test"] == lofo["per_family_counts"][fam]
        rest = al.EXPECTED_TOTAL_RECORDS - lofo["per_family_counts"][fam]
        assert counts["train"] + counts["valid"] == rest


# --------------------------------------------------------------------------- #
# sourced RiNALMo parity target — recorded, not fabricated (§10.3)
# --------------------------------------------------------------------------- #
def test_published_target_sourced_and_agrees_with_adr() -> None:
    tgt = json.loads(_TARGET.read_text())
    # cited from the paper, not invented
    assert tgt["source"]["pmcid"] == "PMC12219582"
    assert tgt["source"]["doi"] == "10.1038/s41467-025-60872-5"
    assert tgt["source"]["metric_column"] == "F1_non_weighted"
    # no SD published → the ±N pp fallback governs (ADR-0002 D5)
    assert tgt["sd_reported"] is False
    assert tgt["parity_gate"]["margin_pp"] == 2.0

    pub = tgt["published_f1"]["per_family"]
    adr = tgt["adr_pinned_f1_2dp"]["per_family"]
    assert set(pub) == set(al.FAMILY_ORDER)
    assert set(adr) == set(al.FAMILY_ORDER)
    # published (3 dp) agrees with the ADR-pinned (2 dp) values to 0.01
    for fam in al.FAMILY_ORDER:
        assert round(pub[fam], 2) == adr[fam], fam
    assert (
        round(tgt["published_f1"]["mean_interfamily"], 2)
        == tgt["adr_pinned_f1_2dp"]["mean_interfamily"]
    )
    # telomerase is the carved-out outlier (ADR-0002 D5)
    assert tgt["parity_gate"]["telomerase_carveout"]["published_f1"] == 0.12
    assert "telomerase_RNA" not in tgt["parity_gate"]["stable_families"]
    assert len(tgt["parity_gate"]["stable_families"]) == 8


def test_published_target_hardest_and_easiest_families() -> None:
    tgt = json.loads(_TARGET.read_text())
    pub = tgt["published_f1"]["per_family"]
    # the published qualitative pattern: tRNA best, telomerase worst
    assert max(pub, key=pub.get) == "tRNA"
    assert min(pub, key=pub.get) == "telomerase_RNA"


def test_provenance_and_target_digests_and_counts_agree() -> None:
    """The two committed full-corpus constants CI cannot recompute must agree."""
    prov = json.loads(_PROV.read_text())
    tgt = json.loads(_TARGET.read_text())
    assert prov["lofo"]["lofo_digest"] == tgt["dataset"]["lofo_digest"]
    assert len(tgt["dataset"]["lofo_digest"]) == 64
    assert tgt["dataset"]["n_records"] == al.EXPECTED_TOTAL_RECORDS
    prov_counts = prov["lofo"]["per_family_counts"]
    tgt_counts = {fam: tgt["families"][fam]["n_test"] for fam in tgt["families"]}
    assert prov_counts == tgt_counts
    assert sum(tgt_counts.values()) == al.EXPECTED_TOTAL_RECORDS


# --------------------------------------------------------------------------- #
# validate_manifest(require_full=True) — the load-bearing production-shape gate
# (only reachable via the network-gated staging path, so covered here w/ a
# hand-built manifest so every fail-closed branch runs in bare CI).
# --------------------------------------------------------------------------- #
def _full_manifest(**overrides) -> al.LofoManifest:
    counts = {
        "5S_rRNA": 1283,
        "SRP_RNA": 918,
        "tRNA": 557,
        "tmRNA": 462,
        "RNaseP_RNA": 454,
        "group_I_intron": 74,
        "16S_rRNA": 67,
        "23S_rRNA": 15,
        "telomerase_RNA": 35,
    }
    total = sum(counts.values())  # 3865
    folds = {f: {"test": c, "train": total - c, "valid": 0} for f, c in counts.items()}
    m = al.LofoManifest(
        lofo_digest="0" * 64,
        n_records=total,
        families=sorted(counts, key=al.FAMILY_ORDER.index),
        per_family_counts=dict(counts),
        per_fold_counts=folds,
    )
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


def test_validate_full_manifest_passes() -> None:
    al.validate_manifest(_full_manifest(), require_full=True)


def test_validate_full_rejects_wrong_family_count() -> None:
    m = _full_manifest()
    del m.per_family_counts["telomerase_RNA"]
    del m.per_fold_counts["telomerase_RNA"]
    m.families = [f for f in m.families if f != "telomerase_RNA"]
    m.n_records = sum(m.per_family_counts.values())
    with pytest.raises(ValueError, match="expected 9 families"):
        al.validate_manifest(m, require_full=True)


def test_validate_full_rejects_wrong_total() -> None:
    m = _full_manifest()
    m.per_family_counts["5S_rRNA"] -= 1  # sum -> 3864, still self-consistent
    m.n_records -= 1
    with pytest.raises(ValueError, match="expected 3865 records"):
        al.validate_manifest(m, require_full=True)


def test_validate_full_rejects_bad_test_fold() -> None:
    m = _full_manifest()
    m.per_fold_counts["tRNA"]["test"] = 999  # != family size 557
    with pytest.raises(ValueError, match="test size != family size"):
        al.validate_manifest(m, require_full=True)


def test_validate_full_rejects_bad_train_valid_sum() -> None:
    m = _full_manifest()
    m.per_fold_counts["tRNA"]["train"] += 5  # train+valid != held-out complement
    with pytest.raises(ValueError, match=r"train\+valid != held-out complement"):
        al.validate_manifest(m, require_full=True)


def test_validate_rejects_count_mismatch_and_unknown_family() -> None:
    m = _full_manifest(n_records=9999)  # != sum(per_family)
    with pytest.raises(ValueError, match="n_records disagrees"):
        al.validate_manifest(m, require_full=False)
    m2 = _full_manifest()
    m2.families = [*m2.families, "bogus_family"]
    with pytest.raises(ValueError, match="unknown family"):
        al.validate_manifest(m2, require_full=False)
