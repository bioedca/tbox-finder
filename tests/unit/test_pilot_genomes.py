"""P2-10c′-a: the ρ-pilot genome-selection manifest (ADR-0003 D6).

Every test guards a way the selection could report a clean, ~100-genome-looking sample
while failing its one job — spanning divergent clades reproducibly:

* a crosswalk whose header is not a GTDB species-cluster file (wrong input, mined blind),
* a selection that collapsed to a handful of phyla (no cross-clade ρ variance),
* a selection that dropped the archaeal stretch (§7.2),
* a ``seed`` knob that does not actually change which genome represents a phylum
  (a vacuous perturbation — the softmax-shift-invariance trap, MEMORY §12),
* non-determinism across identical runs.

The pure selector tests are stdlib-only (bare-CI safe); the end-to-end ``build`` test —
which writes parquet — is guarded by ``importorskip`` so collection never breaks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tbox_finder.mining import pilot_genomes as pg

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pilot_genomes"
    / "sp_clusters_pilot_sample.tsv"
)


def _write_crosswalk(path: Path, rows: list[str]) -> Path:
    """Write a minimal GTDB sp_clusters TSV (real header + given raw data rows)."""
    header = FIXTURE.read_text().splitlines()[0]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def _fixture_rows() -> list[str]:
    return FIXTURE.read_text().splitlines()[1:]


# --------------------------------------------------------------------------- parse


def test_parse_species_reps_fixture() -> None:
    reps = pg.parse_species_reps(FIXTURE)
    assert len(reps) == 33
    # Source prefix stripped → bare NCBI assembly accession; raw kept alongside.
    for rep in reps:
        assert rep["assembly_accession"].startswith(("GCF_", "GCA_"))
        assert not rep["assembly_accession"].startswith(("RS_", "GB_"))
        assert rep["gtdb_accession"].startswith(("RS_", "GB_"))
        assert rep["domain"] in {"Bacteria", "Archaea"}
        assert rep["phylum"]  # non-empty
        assert rep["gtdb_taxonomy"].startswith("d__")
    assert {r["domain"] for r in reps} == {"Bacteria", "Archaea"}
    assert len({r["phylum"] for r in reps}) == 15


def test_parse_rejects_non_gtdb_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text("some_other_id\tname\tlineage\nx\ty\tz\n")
    with pytest.raises(pg.PilotSelectionError, match="not a GTDB species-cluster file"):
        pg.parse_species_reps(bad)


# ----------------------------------------------------------------------- selection


def test_selection_selects_all_when_target_exceeds_pool() -> None:
    reps = pg.parse_species_reps(FIXTURE)
    sel = pg.select_pilot_genomes(reps, n_target=100, per_phylum_cap=99, seed=42)
    assert len(sel) == 33  # the whole fixture
    assert len({r["phylum"] for r in sel}) == 15
    assert {r["domain"] for r in sel} == {"Bacteria", "Archaea"}
    # selection_rank is a dense 0..N-1 draw order.
    assert sorted(r["selection_rank"] for r in sel) == list(range(33))


def test_selection_respects_per_phylum_cap() -> None:
    reps = pg.parse_species_reps(FIXTURE)
    sel = pg.select_pilot_genomes(reps, n_target=100, per_phylum_cap=1, seed=42)
    counts: dict[str, int] = {}
    for r in sel:
        counts[r["phylum"]] = counts.get(r["phylum"], 0) + 1
    assert max(counts.values()) == 1
    assert len(sel) == 15  # one per phylum, all phyla present


def test_selection_is_breadth_first() -> None:
    # n_target == n_phyla with room to spare per phylum: pass 1 must cover every phylum
    # exactly once before any phylum is revisited.
    reps = pg.parse_species_reps(FIXTURE)
    sel = pg.select_pilot_genomes(reps, n_target=15, per_phylum_cap=5, seed=42)
    counts: dict[str, int] = {}
    for r in sel:
        counts[r["phylum"]] = counts.get(r["phylum"], 0) + 1
    assert len(sel) == 15
    assert set(counts.values()) == {1}  # every phylum exactly once


def test_selection_is_deterministic() -> None:
    reps = pg.parse_species_reps(FIXTURE)
    a = pg.select_pilot_genomes(reps, n_target=20, per_phylum_cap=3, seed=7)
    b = pg.select_pilot_genomes(reps, n_target=20, per_phylum_cap=3, seed=7)
    key = lambda s: [(r["assembly_accession"], r["selection_rank"]) for r in s]  # noqa: E731
    assert key(a) == key(b)


def test_seed_selects_the_phylum_representative() -> None:
    # A multi-rep phylum (Pseudomonadota has 4 reps in the fixture) with cap=1 draws ONE
    # rep; which one must depend on the seed, or the seed is a vacuous knob.
    reps = pg.parse_species_reps(FIXTURE)
    chosen = set()
    for seed in range(12):
        sel = pg.select_pilot_genomes(reps, n_target=100, per_phylum_cap=1, seed=seed)
        (rep,) = [r for r in sel if r["phylum"] == "Pseudomonadota"]
        chosen.add(rep["assembly_accession"])
    assert len(chosen) > 1, "seed does not change the phylum representative (vacuous knob)"


def test_selection_rejects_bad_params() -> None:
    reps = pg.parse_species_reps(FIXTURE)
    with pytest.raises(pg.PilotSelectionError):
        pg.select_pilot_genomes(reps, n_target=0)
    with pytest.raises(pg.PilotSelectionError):
        pg.select_pilot_genomes(reps, per_phylum_cap=0)


# --------------------------------------------------------------------------- build


def test_build_end_to_end(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    out = tmp_path / "pilot" / "manifest.parquet"
    prov = tmp_path / "pilot" / "manifest.provenance.json"
    report = tmp_path / "audits" / "report.json"
    rc = pg.build(
        crosswalk_path=FIXTURE,
        out_parquet=out,
        provenance_path=prov,
        report_path=report,
        seed=42,
        n_target=100,
        per_phylum_cap=5,
        min_phyla=10,
        min_archaea=2,
        env_lock=None,
    )
    assert rc == 0

    df = pd.read_parquet(out)
    assert list(df.columns) == [
        "assembly_accession",
        "gtdb_accession",
        "domain",
        "phylum",
        "gtdb_species",
        "gtdb_taxonomy",
        "selection_rank",
    ]
    assert len(df) == 33
    assert df["assembly_accession"].is_unique
    assert df["assembly_accession"].str.startswith(("GCF_", "GCA_")).all()

    import json

    rep = json.loads(report.read_text())
    assert rep["n_selected"] == 33
    assert rep["n_phyla_spanned"] == 15
    assert rep["n_archaea"] == 7
    assert rep["n_bacteria"] == 26
    assert rep["gtdb_release"] == "R232"
    assert len(rep["crosswalk_sha256"]) == 64

    provrec = json.loads(prov.read_text())
    for field in ("rule", "script", "git_sha", "seed", "inputs", "outputs"):
        assert field in provrec
    assert provrec["adr"] == "ADR-0003"
    assert provrec["extra"]["n_phyla_spanned"] == 15


def test_build_content_is_reproducible(tmp_path: Path) -> None:
    # Content (not raw parquet bytes — pyarrow may vary footer metadata) is stable.
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    def run(dst: Path):
        pg.build(
            crosswalk_path=FIXTURE,
            out_parquet=dst / "m.parquet",
            provenance_path=dst / "m.prov.json",
            report_path=dst / "r.json",
            seed=42,
            n_target=100,
            per_phylum_cap=5,
            min_phyla=10,
            min_archaea=2,
        )
        return pd.read_parquet(dst / "m.parquet")

    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    pd.testing.assert_frame_equal(a, b)


def test_build_succeeds_when_per_phylum_cap_binds(tmp_path: Path) -> None:
    # When the cap binds below phyla's rep totals the achievable count is
    # sum(min(cap, size)); the expected-count guard must use that, not the raw rep total,
    # or a valid cap-constrained selection would raise (CodeRabbit r1 regression guard).
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    out = tmp_path / "m.parquet"
    rc = pg.build(
        crosswalk_path=FIXTURE,
        out_parquet=out,
        provenance_path=tmp_path / "m.prov.json",
        report_path=tmp_path / "r.json",
        seed=42,
        n_target=100,
        per_phylum_cap=2,  # binds: the fixture has phyla with 3-4 reps
        min_phyla=10,
        min_archaea=2,
    )
    assert rc == 0
    df = pd.read_parquet(out)
    # 15 phyla, per-phylum sizes capped at 2 sum to 26 (11 phyla with >=2 reps, 4 with 1).
    assert len(df) == 26
    assert df["phylum"].value_counts().max() == 2


# --------------------------------------------------------- must-fire guard sabotage


def test_build_guard_fires_on_collapsed_phyla(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    # All rows from one phylum → a non-divergent sample. The guard must reject it rather
    # than ship a manifest that silently understates cross-clade ρ variance.
    single = [r for r in _fixture_rows() if "p__Pseudomonadota" in r]
    assert len(single) >= 2
    cw = _write_crosswalk(tmp_path / "one_phylum.tsv", single)
    with pytest.raises(pg.PilotSelectionError, match="min_phyla"):
        pg.build(
            crosswalk_path=cw,
            out_parquet=tmp_path / "m.parquet",
            provenance_path=tmp_path / "m.prov.json",
            report_path=tmp_path / "r.json",
            n_target=100,
            min_phyla=5,
            min_archaea=0,
        )


def test_build_guard_fires_on_missing_archaea(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    bacteria_only = [r for r in _fixture_rows() if "d__Bacteria" in r.split("\t")[2]]
    cw = _write_crosswalk(tmp_path / "bac_only.tsv", bacteria_only)
    with pytest.raises(pg.PilotSelectionError, match="archaeal stretch"):
        pg.build(
            crosswalk_path=cw,
            out_parquet=tmp_path / "m.parquet",
            provenance_path=tmp_path / "m.prov.json",
            report_path=tmp_path / "r.json",
            n_target=100,
            min_phyla=3,
            min_archaea=1,
        )
