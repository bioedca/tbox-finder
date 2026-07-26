"""P2-10c′-fetch: the ρ-sized PRODUCTION whole-genome fetch (ADR-0006 A1).

``mining/production_fetch.py`` reuses the pilot's whole-genome fetcher, its resumable NCBI
transport, and its fail-closed report/validator verbatim (promote-don't-duplicate). These tests
guard the three things that are genuinely its own and could silently drift:

* the **ADR-0006 A1 / operational pins** (step label, ADR / rule / script provenance, the
  re-scoped floors, the artifact paths) — a silent edit must fail a test,
* **reuse, not fork** — production ``build`` must delegate to ``pilot_fetch.build_pilot_fetch``,
  forwarding the ADR-0006 metadata + its own module as the provenance script + the re-scoped
  floors (a forked fetcher would fix one copy and ship the bug in the other),
* the **re-scoped floors actually bite** — production's ``MIN_PHYLA_OK`` (150) must reject a
  fetch that the pilot's 50 would pass, on the same delegation path (a floor read from the wrong
  constant is a vacuous guard),

plus a backward-compatibility guard that the generalization left the **pilot defaults unchanged**
(the pilot regression the P2-10c′-g production_genomes step also carries).

Stdlib-only: the fetcher's pure report/validator helpers import in bare CI (no biopython, no
pandas, no network), and the delegation test monkeypatches the network fetch away.
"""

from __future__ import annotations

from typing import Any

import pytest

from tbox_finder.mining import pilot_fetch as pf
from tbox_finder.mining import production_fetch as prod

# ---------------------------------------------------------------- ADR-0006 A1 / pin drift


def test_production_pins_match_adr0006_a1() -> None:
    # These pin the ADR-0006 A1 production fetch; a change to the ADR-scoped ones needs
    # re-sign-off, and the operational ones (floors/paths/labels) must not drift silently.
    assert prod.PRODUCTION_STEP == "P2-10c'-fetch"
    assert prod.PRODUCTION_ADR == "ADR-0006"
    assert prod.PRODUCTION_SCRIPT == "src/tbox_finder/mining/production_fetch.py"
    assert prod.PRODUCTION_RULE == (
        "slurm/p2/fetch_production_genomes.sbatch :: "
        "python -m tbox_finder.mining.production_fetch"
    )
    assert str(prod.PRODUCTION_MANIFEST) == "data/processed/mining/production_genomes_v0.parquet"
    assert str(prod.PRODUCTION_GENOME_DIR) == "data/interim/production_genomes"
    assert str(prod.PRODUCTION_REPORT) == "data/processed/audits/production_fetch_report.json"
    assert str(prod.PRODUCTION_PROVENANCE) == "data/interim/production_genomes.provenance.json"
    assert prod.PRODUCTION_MIN_SUCCESS_RATE == 0.90
    assert prod.PRODUCTION_MIN_PHYLA_OK == 150


def test_production_floors_are_stricter_than_or_equal_to_pilot() -> None:
    # A production floor that silently fell back to the pilot's would not fail-loud on a
    # collapsed fetch; the phyla floor must be the higher production one, the rate at least equal.
    assert prod.PRODUCTION_MIN_PHYLA_OK > pf.MIN_PHYLA_OK  # 150 > 50
    assert prod.PRODUCTION_MIN_SUCCESS_RATE >= pf.MIN_SUCCESS_RATE  # 0.90 >= 0.90


# --------------------------------------------------------------- reuse, not fork (delegation seam)


def test_build_delegates_to_shared_fetcher(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enforce the delegation SEAM directly: production build must call
    # pilot_fetch.build_pilot_fetch, forwarding the ADR-0006 metadata, its own module as the
    # provenance script, and the re-scoped floors — not a forked fetcher.
    calls: list[dict[str, Any]] = []

    def spy(**kwargs: object) -> dict[str, Any]:
        calls.append(dict(kwargs))
        return {"delegated": True}

    monkeypatch.setattr(pf, "build_pilot_fetch", spy)
    out = prod.build()
    assert out == {"delegated": True}
    assert len(calls) == 1, "production build did not delegate to pilot_fetch.build_pilot_fetch"
    kw = calls[0]
    assert kw["step"] == prod.PRODUCTION_STEP
    assert kw["adr"] == "ADR-0006"
    assert kw["rule_name"] == prod.PRODUCTION_RULE
    assert kw["script"] == prod.PRODUCTION_SCRIPT
    assert kw["min_success_rate"] == prod.PRODUCTION_MIN_SUCCESS_RATE
    assert kw["min_phyla_ok"] == prod.PRODUCTION_MIN_PHYLA_OK
    assert kw["manifest_parquet"] == prod.PRODUCTION_MANIFEST
    assert kw["genome_dir"] == prod.PRODUCTION_GENOME_DIR
    # The provenance note must name the production role, not leak the pilot's.
    assert "PRODUCTION" in kw["provenance_note"]
    assert "P2-10c′-fetch" in kw["provenance_note"]


# ------------------------------------------------------ the re-scoped phyla floor bites (must-fire)


def _ok_row(i: int, phylum: str) -> dict[str, Any]:
    """A minimal STATUS_OK evidence row that passes every per-row + count check."""
    acc = f"GCF_{i:09d}.1"
    return {
        "assembly_accession": acc,
        "domain": "Bacteria",
        "phylum": phylum,
        "status": pf.STATUS_OK,
        "assembly_uid": str(1000 + i),
        "source_url": f"https://ftp.ncbi.nlm.nih.gov/x/{acc}_genomic.fna.gz",
        "n_replicons": 1,
        "total_bp": 1000 + i,
        "seq_sha256": "a" * 64,
        "fasta_path": f"data/interim/production_genomes/{acc}.fna",
    }


def _valid_report_over_phyla(n_phyla: int) -> dict[str, Any]:
    # Two ok genomes per phylum, all fetched → success_rate 1.0 spanning exactly n_phyla phyla.
    rows = [_ok_row(i, f"Phylum{i % n_phyla}") for i in range(n_phyla * 2)]
    return pf.build_report(rows, accessed="2026-07-23", step=prod.PRODUCTION_STEP)


def test_rescoped_phyla_floor_is_the_one_on_the_production_path() -> None:
    # A 100-phylum fetch: the production floor (150) must REJECT it while the pilot default
    # (50) accepts it — proving production build's min_phyla_ok is the value that governs, and
    # that the floor is value-sensitive (not a vacuous constant).
    rep = _valid_report_over_phyla(100)
    assert rep["n_phyla_spanned_ok"] == 100
    assert pf.validate_report(rep) == []  # pilot default floor 50 → passes
    problems = pf.validate_report(rep, min_phyla_ok=prod.PRODUCTION_MIN_PHYLA_OK)
    assert any("MIN_PHYLA_OK" in p for p in problems), problems


def test_a_full_span_fetch_passes_the_production_floor() -> None:
    # The other direction: a 197-phylum fetch (the A1 selection's full span) certifies under
    # the production floor — the guard is a floor, not an unconditional reject.
    rep = _valid_report_over_phyla(197)
    assert pf.validate_report(rep, min_phyla_ok=prod.PRODUCTION_MIN_PHYLA_OK) == []


# ---------------------------------------------------------------- backward-compat (pilot unchanged)


def test_generalization_preserves_pilot_defaults() -> None:
    # The pilot regression: the generalization must leave the pilot's step label + floor
    # constants byte-identical, or P2-10c′-b's committed report/behaviour would drift.
    assert pf.PILOT_STEP == "P2-10c'-b"
    assert pf.DEFAULT_ADR == "ADR-0003"
    assert pf.MIN_PHYLA_OK == 50 and pf.MIN_SUCCESS_RATE == 0.90
    # build_report without a step arg still stamps the pilot label.
    rep = pf.build_report([_ok_row(0, "P0")], accessed="2026-07-23")
    assert rep["step"] == "P2-10c'-b"
    # ...and the same rows under the production step carry the production label.
    rep_prod = pf.build_report([_ok_row(0, "P0")], accessed="2026-07-23", step=prod.PRODUCTION_STEP)
    assert rep_prod["step"] == "P2-10c'-fetch"
