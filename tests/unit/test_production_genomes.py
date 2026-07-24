"""P2-10c′-g: the PRODUCTION homolog-DB + negative-window genome-selection manifest (ADR-0006 A1).

``mining/production_genomes.py`` reuses the pilot's pure selector and guarded builder
verbatim (promote-don't-duplicate); these tests guard the three things that are genuinely
its own and could silently drift:

* the **ADR-0006 A1 pins** (n_target=2500, per_phylum_cap=20, floors, paths) — a change
  needs ADR re-sign-off (§7 item 2), so a silent edit must fail a test,
* **reuse, not fork** — production ``build`` must produce a manifest byte-for-byte
  identical to the shared ``pilot_genomes.build`` on the same inputs; only the metadata
  (ADR-0006, rule, notes) may differ (a forked selector would ship a bug in one copy —
  MEMORY: promote-don't-duplicate-is-a-correctness-rule),
* the **high default floors are actually wired** — production's default ``min_phyla``/
  ``min_archaea`` must reject a small sample rather than pass it as the pilot's low floors
  would (a floor read from the wrong constant is a vacuous guard),
* a **committed-report lock** on the real R232 selection (2,500 reps / 197 phyla /
  2,109 bacteria / 391 archaea) so a regenerated manifest that drifts off the A1 pin is
  caught in CI.

Stdlib-only where possible (bare-CI safe); the parquet-writing tests ``importorskip``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbox_finder.mining import pilot_genomes as pg
from tbox_finder.mining import production_genomes as prod

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pilot_genomes"
    / "sp_clusters_pilot_sample.tsv"
)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Scaled floors for the 33-rep / 15-phylum fixture (the production defaults are for the
# full 199,923-rep R232 crosswalk and would — correctly — reject the fixture).
_FIXTURE_FLOORS = {"min_phyla": 10, "min_archaea": 2}


# ---------------------------------------------------------------- ADR-0006 A1 pin drift


def test_production_pins_match_adr0006_a1() -> None:
    # These are the ADR-0006 Amendment A1 pinned values; a change needs ADR re-sign-off.
    assert prod.PRODUCTION_N_TARGET == 2500
    assert prod.PRODUCTION_PER_PHYLUM_CAP == 20  # smallest cap reaching 2500 over 197 phyla
    assert prod.PRODUCTION_MIN_PHYLA == 150
    assert prod.PRODUCTION_MIN_ARCHAEA == 100
    assert prod.PRODUCTION_ADR == "ADR-0006"
    assert prod.PRODUCTION_RULE == "workflow/rules/data.smk :: select_production_genomes"
    assert prod.PRODUCTION_MANIFEST == "data/processed/mining/production_genomes_v0.parquet"
    assert prod.PRODUCTION_REPORT == "data/processed/audits/production_genomes_report.json"


def test_production_floors_are_stricter_than_pilot() -> None:
    # A production floor that silently fell back to the pilot's would not fail-loud on a
    # collapsed sample; the pins must be the higher production ones.
    assert prod.PRODUCTION_MIN_PHYLA > pg.DEFAULT_MIN_PHYLA
    assert prod.PRODUCTION_MIN_ARCHAEA > pg.DEFAULT_MIN_ARCHAEA
    assert prod.PRODUCTION_N_TARGET > pg.DEFAULT_N_TARGET


# --------------------------------------------------------------- reuse, not fork


def test_build_invokes_shared_pilot_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enforce the delegation SEAM directly (not just equivalent output, which a forked
    # copy that happens to match on a fixture would also pass — CodeRabbit): production
    # build must call pilot_genomes.build, forwarding the ADR-0006 metadata + its own
    # module as the provenance script.
    calls: list[dict] = []

    def spy(**kwargs: object) -> int:
        calls.append(dict(kwargs))
        return 0

    monkeypatch.setattr(pg, "build", spy)
    rc = prod.build(crosswalk_path=FIXTURE)
    assert rc == 0
    assert len(calls) == 1, "production build did not delegate to pilot_genomes.build"
    kw = calls[0]
    assert kw["adr"] == "ADR-0006"
    assert kw["rule_name"] == prod.PRODUCTION_RULE
    assert kw["notes"] == prod.PRODUCTION_NOTES
    assert kw["script"] == prod.PRODUCTION_SCRIPT
    assert kw["kind"] == "production"
    assert kw["n_target"] == prod.PRODUCTION_N_TARGET
    assert kw["per_phylum_cap"] == prod.PRODUCTION_PER_PHYLUM_CAP


def test_build_output_matches_shared_builder(tmp_path: Path) -> None:
    # Behavioral output-equivalence: same inputs → the production manifest is identical
    # to the pilot builder's (complements the seam test above; catches a metadata-only
    # wrapper that silently perturbs the selected rows).
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    def run(builder, adr: str) -> Path:
        dst = tmp_path / adr
        builder(
            crosswalk_path=FIXTURE,
            out_parquet=dst / "m.parquet",
            provenance_path=dst / "m.prov.json",
            report_path=dst / "r.json",
            seed=42,
            n_target=20,
            per_phylum_cap=3,
            **_FIXTURE_FLOORS,
        )
        return dst

    pilot_dst = run(pg.build, "pilot")
    prod_dst = run(prod.build, "prod")
    pd.testing.assert_frame_equal(
        pd.read_parquet(pilot_dst / "m.parquet"),
        pd.read_parquet(prod_dst / "m.parquet"),
    )


def test_build_stamps_adr0006_metadata(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    dst = tmp_path / "prod"
    prod.build(
        crosswalk_path=FIXTURE,
        out_parquet=dst / "m.parquet",
        provenance_path=dst / "m.prov.json",
        report_path=dst / "r.json",
        seed=42,
        n_target=20,
        per_phylum_cap=3,
        **_FIXTURE_FLOORS,
    )
    provrec = json.loads((dst / "m.prov.json").read_text())
    assert provrec["adr"] == "ADR-0006"
    assert provrec["rule"] == prod.PRODUCTION_RULE
    # Provenance must name the module the rule invokes, not the pilot module it delegates
    # to — else the artifact claims the wrong author (CodeRabbit major).
    assert provrec["script"] == prod.PRODUCTION_SCRIPT
    report = json.loads((dst / "r.json").read_text())
    # The production notes name its role; a pilot-notes leak (wrong `notes=` wiring) fails.
    assert "PRODUCTION genome-selection manifest" in report["notes"]
    assert "A8 clause (viii)" in report["notes"]
    assert "P2-10c′-a" not in report["notes"]  # not the pilot notes


# ------------------------------------------------------ default floors are wired (must-fire)


def test_default_floors_reject_small_sample() -> None:
    # Called with production DEFAULTS (no floor override), the 15-phylum fixture must be
    # rejected by min_phyla=150 — proving the high floor is the one on the default path,
    # not the pilot's 50. (importorskip pandas: build imports it before the guard raises.)
    pytest.importorskip("pandas")
    with pytest.raises(pg.PilotSelectionError, match="min_phyla"):
        prod.build(crosswalk_path=FIXTURE)


# ---------------------------------------------------------------- committed-report lock


def test_committed_report_locks_the_a1_selection() -> None:
    # The real R232 selection is deterministic from the frozen crosswalk (sha256 96414cdc…);
    # these headline numbers ARE the ADR-0006 A1 claim. A regenerated manifest that drifts
    # off the pin changes them → this fails in CI (the audit report is git-committed).
    report_path = REPO_ROOT / prod.PRODUCTION_REPORT
    assert report_path.is_file(), f"committed production report missing: {report_path}"
    rep = json.loads(report_path.read_text())
    assert rep["n_selected"] == 2500
    assert rep["n_target"] == 2500
    assert rep["per_phylum_cap"] == 20
    assert rep["seed"] == 42
    assert rep["n_phyla_available"] == 197
    assert rep["n_phyla_spanned"] == 197  # all R232 phyla spanned
    assert rep["n_bacteria"] == 2109
    assert rep["n_archaea"] == 391
    assert rep["n_reps_total"] == 199923
    assert rep["gtdb_release"] == "R232"
    assert rep["crosswalk_sha256"] == (
        "96414cdc04addaac0280593f3d32a7c8c30cbc2aa498a9fee81886e478ec9fc9"
    )
