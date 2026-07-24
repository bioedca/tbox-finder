"""pilot_genomes.py — the ρ-pilot genome-selection manifest (ADR-0003 D6; P2-10c′-a).

ADR-0003 D6 pins a scan-sizing pilot: *"On a ~100-genome sample spanning divergent
clades, measure … **Stage-1 candidates/Mbp ρ** …"*. ρ is the pivot of the P2-10e /
P2-10c′ homolog-DB + negative-window fetch, whose size the roadmap prices at
**0.7–68 GB pivoting on ρ** — nobody has measured it, so nothing downstream (N, the
fetch, the mining GPU-h budget) can be sized until it is.

This module implements the **selection half** of that pilot, and *only* that half: it
turns the pinned GTDB R232 species-representative crosswalk (``sp_clusters_r232.tsv``,
one row per species cluster → representative genome + GTDB lineage) into a reproducible,
phylum-stratified ~100-genome **manifest** (accession + lineage). It deliberately does
**not** fetch the genomes (that is the LOCAL fetch sub-step) and does **not** scan them
or choose a Stage-1 detection threshold (that is the SLURM scan sub-step, where ρ is
finally measured). No ρ value, no candidate count, and no detection threshold is decided
here — so this step carries no §10.3 fabrication surface and pins no ADR value; it
concretises the ADR-0003 D6 pilot that is already pinned.

**Selection scheme (deterministic, seeded, divergence-spanning).** Species reps are
grouped by GTDB phylum; within each phylum a seeded shuffle fixes a reproducible
representative order; phyla are ordered by descending rep-count (ties by name) and drawn
**round-robin** — one rep per phylum per pass — until ``n_target`` is reached. Because a
single pass covers every phylum before any phylum is revisited, the sample spans the
maximum number of divergent clades the target allows (at ``n_target``=100 over R232's 197
phyla: 100 phyla, one genome each, both domains). ``per_phylum_cap`` bounds any one
phylum should ``n_target`` exceed the phylum count. The scheme captures cross-clade ρ
*variance* — the right basis for a **conservative** (high-percentile) fetch size — rather
than matching the scan target's phylum composition.

Mirrors ``mining/pool.py``: a pure, unit-tested selector (:func:`select_pilot_genomes`),
an I/O ``build`` that writes manifest + report + ``provenance.json`` behind must-fire
non-vacuity guards, and a ``main`` CLI. Stdlib + a lazy pandas import (parquet only), so
the module imports in a bare env.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder import provenance

#: Canonical artifact paths (one path per artifact — imp.md §"Canonical artifact paths").
PILOT_DIR = "data/processed/pilot"
PILOT_MANIFEST = f"{PILOT_DIR}/pilot_genomes_v0.parquet"
PILOT_PROVENANCE = f"{PILOT_DIR}/pilot_genomes_v0.provenance.json"
PILOT_REPORT = "data/processed/audits/pilot_genomes_report.json"

#: The pinned GTDB R232 species-rep crosswalk (staged by ``pin_gtdb_release``; ADR-0003 A1).
DEFAULT_CROSSWALK = "data/external/gtdb/sp_clusters_r232.tsv"

#: ADR-0003 D6 pins "~100-genome sample"; the cap only binds when n_target > n_phyla.
DEFAULT_N_TARGET = 100
DEFAULT_PER_PHYLUM_CAP = 5

#: Provenance metadata for the pilot invocation (generalised so the production selection —
#: ``mining/production_genomes.py``, ADR-0006 A1 — can reuse this guarded builder verbatim
#: rather than fork it; only the ADR / rule / notes differ, not the selector or its guards).
DEFAULT_ADR = "ADR-0003"
DEFAULT_RULE_NAME = "workflow/rules/data.smk :: select_pilot_genomes"
_PILOT_NOTES = (
    "P2-10c′-a: the SELECTION half of the ADR-0003 D6 ρ-pilot. A reproducible, "
    "phylum-stratified, divergence-spanning ~100-genome sample of GTDB R232 species "
    "representatives, drawn round-robin across phyla (breadth-first; one rep per "
    "phylum per pass) so the sample spans the maximum number of divergent clades the "
    "target allows. This manifest is an ACCESSION LIST only — it fetches no genomes "
    "(the LOCAL fetch sub-step does that) and measures no ρ, no candidate count, and "
    "chooses no Stage-1 detection threshold (the SLURM scan sub-step does that, where "
    "ρ is finally measured). It therefore pins no ADR value and carries no scientific "
    "claim. ρ sizes the shared GTDB/RefSeq homolog-DB + negative-window fetch "
    "(P2-10c′/P2-10e), priced at 0.7–68 GB pivoting on ρ."
)
#: Non-vacuity floors for the production run (197 phyla available in R232). A divergence-
#: spanning pilot that collapsed to a handful of clades — or dropped the archaeal stretch
#: (§7.2) — must fail loud rather than ship a sample that silently understates ρ variance.
DEFAULT_MIN_PHYLA = 50
DEFAULT_MIN_ARCHAEA = 3

#: GTDB ``sp_clusters`` column layout (header-validated at parse time; see taxonomy.py).
_REP_COL = 0
_SPECIES_COL = 1
_TAXONOMY_COL = 2
_HEADER_PREFIX = "Representative genome"

#: GTDB accessions carry an assembly-source prefix (``RS_`` RefSeq / ``GB_`` GenBank) on
#: top of the NCBI assembly accession; the fetch step wants the bare ``GC[AF]_…`` id.
_SOURCE_PREFIXES = ("RS_", "GB_")

_RANK_DOMAIN = "d__"
_RANK_PHYLUM = "p__"


class PilotSelectionError(ValueError):
    """Raised when the crosswalk is malformed or the selection fails a non-vacuity guard."""


def _rank_value(lineage: str, prefix: str) -> str:
    """Return the ``prefix``-tagged rank value from a GTDB ``;``-joined lineage, or ``""``.

    e.g. ``_rank_value("d__Bacteria;p__Pseudomonadota;c__…", "p__") == "Pseudomonadota"``.
    """
    for seg in lineage.split(";"):
        seg = seg.strip()
        if seg.startswith(prefix):
            return seg[len(prefix) :]
    return ""


def _strip_source_prefix(gtdb_accession: str) -> str:
    """Strip a GTDB ``RS_``/``GB_`` source prefix, yielding the NCBI assembly accession."""
    for pref in _SOURCE_PREFIXES:
        if gtdb_accession.startswith(pref):
            return gtdb_accession[len(pref) :]
    return gtdb_accession


def parse_species_reps(crosswalk_path: str | Path) -> list[dict[str, str]]:
    """Stream the GTDB ``sp_clusters`` crosswalk into one record per species rep.

    Each data row is one species cluster (one representative genome). Returns records with
    ``gtdb_accession`` (raw, prefixed), ``assembly_accession`` (bare ``GC[AF]_…``),
    ``domain``, ``phylum``, ``gtdb_species``, and ``gtdb_taxonomy`` (full lineage). The
    ~50 MB file is streamed (never loaded whole), matching taxonomy.py's idiom.

    Raises :class:`PilotSelectionError` if the header is not a GTDB species-cluster file —
    a wrong input must fail loud, never be selected from silently (CLAUDE.md §10.3).
    """
    path = Path(crosswalk_path)
    reps: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header.startswith(_HEADER_PREFIX):
            raise PilotSelectionError(
                f"unexpected sp_clusters header (not a GTDB species-cluster file): "
                f"{header[:60]!r} from {path}"
            )
        for line in fh:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= _TAXONOMY_COL:
                continue
            gtdb_accession = fields[_REP_COL].strip()
            lineage = fields[_TAXONOMY_COL].strip()
            reps.append(
                {
                    "gtdb_accession": gtdb_accession,
                    "assembly_accession": _strip_source_prefix(gtdb_accession),
                    "domain": _rank_value(lineage, _RANK_DOMAIN),
                    "phylum": _rank_value(lineage, _RANK_PHYLUM),
                    "gtdb_species": (
                        fields[_SPECIES_COL].strip() if len(fields) > _SPECIES_COL else ""
                    ),
                    "gtdb_taxonomy": lineage,
                }
            )
    return reps


def select_pilot_genomes(
    reps: Iterable[Mapping[str, str]],
    *,
    n_target: int = DEFAULT_N_TARGET,
    per_phylum_cap: int = DEFAULT_PER_PHYLUM_CAP,
    seed: int = provenance.DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Deterministically pick a phylum-stratified, divergence-spanning sample of reps.

    Pure and reproducible: same ``reps`` + ``seed`` → identical selection and order.
    Reps with no phylum are ignored (they cannot contribute a clade). Returns the selected
    records (copied), each carrying ``selection_rank`` (0-based draw order).

    Round-robin over phyla (ordered by descending rep-count, ties by name), one rep per
    phylum per pass, within-phylum order fixed by a per-phylum seeded shuffle, bounded by
    ``per_phylum_cap`` per phylum, until ``n_target`` reps are drawn or the pool is dry.
    """
    if n_target <= 0:
        raise PilotSelectionError(f"n_target must be positive, got {n_target}")
    if per_phylum_cap <= 0:
        raise PilotSelectionError(f"per_phylum_cap must be positive, got {per_phylum_cap}")

    by_phylum: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rep in reps:
        phylum = str(rep.get("phylum", "")).strip()
        if not phylum:
            continue
        by_phylum[phylum].append(dict(rep))

    # Within-phylum: a seeded shuffle keyed on (seed, phylum) fixes a reproducible pick
    # that is not merely "first alphabetically" — so the seed genuinely selects which
    # genome represents a multi-rep phylum (a no-op seed would be a vacuous knob, §12).
    for phylum, members in by_phylum.items():
        members.sort(key=lambda r: r["assembly_accession"])  # stable base order
        random.Random(f"{seed}:{phylum}").shuffle(members)

    # Phylum draw order: descending rep-count then name. Pass 1 covers every phylum before
    # any is revisited, so breadth (divergent-clade span) is maximal regardless of order;
    # the order only decides which phyla fill the final slots when n_target < n_phyla.
    phyla = sorted(by_phylum, key=lambda p: (-len(by_phylum[p]), p))

    selected: list[dict[str, Any]] = []
    used = {p: 0 for p in phyla}
    rank = 0
    progressed = True
    while len(selected) < n_target and progressed:
        progressed = False
        for phylum in phyla:
            if len(selected) >= n_target:
                break
            members = by_phylum[phylum]
            if used[phylum] < len(members) and used[phylum] < per_phylum_cap:
                rec = dict(members[used[phylum]])
                rec["selection_rank"] = rank
                selected.append(rec)
                used[phylum] += 1
                rank += 1
                progressed = True
    return selected


def _selection_summary(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-domain / per-phylum tallies over a selection (report + guard inputs)."""
    per_domain: dict[str, int] = defaultdict(int)
    per_phylum: dict[str, int] = defaultdict(int)
    for rec in selected:
        per_domain[str(rec.get("domain", ""))] += 1
        per_phylum[str(rec.get("phylum", ""))] += 1
    return {
        "n_selected": len(selected),
        "n_phyla": len(per_phylum),
        "n_bacteria": per_domain.get("Bacteria", 0),
        "n_archaea": per_domain.get("Archaea", 0),
        "per_domain": dict(sorted(per_domain.items())),
        "per_phylum": dict(sorted(per_phylum.items())),
    }


def build(
    *,
    crosswalk_path: str | Path = DEFAULT_CROSSWALK,
    out_parquet: str | Path = PILOT_MANIFEST,
    provenance_path: str | Path = PILOT_PROVENANCE,
    report_path: str | Path = PILOT_REPORT,
    seed: int = provenance.DEFAULT_SEED,
    n_target: int = DEFAULT_N_TARGET,
    per_phylum_cap: int = DEFAULT_PER_PHYLUM_CAP,
    min_phyla: int = DEFAULT_MIN_PHYLA,
    min_archaea: int = DEFAULT_MIN_ARCHAEA,
    env_lock: str | Path | None = None,
    adr: str = DEFAULT_ADR,
    rule_name: str = DEFAULT_RULE_NAME,
    notes: str | None = None,
    kind: str = "pilot",
) -> int:
    """Parse, select, guard, and write a phylum-stratified genome manifest + report + provenance.

    Defaults produce the ADR-0003 D6 ρ-pilot manifest. ``adr`` / ``rule_name`` / ``notes``
    are the only per-invocation metadata that differ for the ADR-0006 A1 **production**
    selection (``mining/production_genomes.py``), which reuses this guarded builder — and
    thus the shared :func:`select_pilot_genomes` selector and every must-fire guard below —
    verbatim (promote-don't-duplicate; the guards are load-bearing correctness logic).
    """
    import pandas as pd

    reps = parse_species_reps(crosswalk_path)
    if not reps:
        raise PilotSelectionError(
            f"no species reps parsed from {crosswalk_path} — refusing to write an empty "
            "manifest that would make every downstream sizing count vacuously zero"
        )
    selected = select_pilot_genomes(
        reps, n_target=n_target, per_phylum_cap=per_phylum_cap, seed=seed
    )
    summary = _selection_summary(selected)

    n_reps_with_phylum = sum(1 for r in reps if str(r.get("phylum", "")).strip())
    phylum_sizes: dict[str, int] = defaultdict(int)
    for rep in reps:
        ph = str(rep.get("phylum", "")).strip()
        if ph:
            phylum_sizes[ph] += 1
    # The round-robin can draw at most ``per_phylum_cap`` from any one phylum, so the
    # *achievable* count — not the raw rep total — is the ceiling this guard checks
    # against; otherwise a cap that binds below the phylum total makes a correct
    # cap-constrained selection raise (CodeRabbit r1).
    max_achievable = sum(min(per_phylum_cap, c) for c in phylum_sizes.values())
    expected = min(n_target, max_achievable)
    accessions = [r["assembly_accession"] for r in selected]

    # Must-fire guards (CLAUDE.md §8.5/§12): each catches a distinct way the selection
    # could silently degrade to a non-divergence-spanning or malformed sample.
    if summary["n_selected"] != expected:
        raise PilotSelectionError(
            f"selected {summary['n_selected']} != expected {expected} "
            f"(min(n_target={n_target}, max_achievable={max_achievable} at "
            f"per_phylum_cap={per_phylum_cap} over {len(phylum_sizes)} phyla)) — "
            "the round-robin did not fill the target"
        )
    if summary["n_phyla"] < min_phyla:
        raise PilotSelectionError(
            f"selection spans {summary['n_phyla']} phyla < min_phyla ({min_phyla}) — not a "
            "divergent-clade sample; ρ measured on it would not bound cross-clade variance"
        )
    if summary["n_archaea"] < min_archaea:
        raise PilotSelectionError(
            f"selection has {summary['n_archaea']} archaeal genomes < min_archaea "
            f"({min_archaea}) — the §7.2 archaeal stretch is unrepresented"
        )
    if len(set(accessions)) != len(accessions):
        raise PilotSelectionError("duplicate assembly accessions in the selection")
    bad = [a for a in accessions if not (a.startswith(("GCF_", "GCA_")) and "." in a)]
    if bad:
        raise PilotSelectionError(
            f"{len(bad)} accessions are not GC[AF]_<n>.<v> assembly ids: {bad[:5]}"
        )

    columns = [
        "assembly_accession",
        "gtdb_accession",
        "domain",
        "phylum",
        "gtdb_species",
        "gtdb_taxonomy",
        "selection_rank",
    ]
    df = (
        pd.DataFrame.from_records(selected)[columns]
        .sort_values("selection_rank")
        .reset_index(drop=True)
    )

    report = {
        "n_selected": summary["n_selected"],
        "n_target": n_target,
        "per_phylum_cap": per_phylum_cap,
        "seed": seed,
        "min_phyla": min_phyla,
        "min_archaea": min_archaea,
        "n_reps_total": len(reps),
        "n_reps_with_phylum": n_reps_with_phylum,
        "n_phyla_available": len({r["phylum"] for r in reps if str(r.get("phylum", "")).strip()}),
        "n_phyla_spanned": summary["n_phyla"],
        "n_bacteria": summary["n_bacteria"],
        "n_archaea": summary["n_archaea"],
        "per_domain": summary["per_domain"],
        "per_phylum": summary["per_phylum"],
        "gtdb_release": "R232",
        "gtdb_license": "CC-BY-SA-4.0 (GTDB data; https://gtdb.ecogenomic.org/downloads)",
        "crosswalk_sha256": provenance.sha256_file(crosswalk_path),
        "notes": _PILOT_NOTES if notes is None else notes,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    provenance.write_provenance(
        provenance_path,
        rule=rule_name,
        script="src/tbox_finder/mining/pilot_genomes.py",
        seed=seed,
        inputs=[crosswalk_path],
        outputs=[out_parquet, report_path],
        env_lock=env_lock,
        adr=adr,
        extra={
            "n_selected": summary["n_selected"],
            "n_phyla_spanned": summary["n_phyla"],
            "n_bacteria": summary["n_bacteria"],
            "n_archaea": summary["n_archaea"],
        },
    )
    print(
        f"selected {summary['n_selected']} {kind} genomes spanning {summary['n_phyla']} phyla "
        f"({summary['n_bacteria']} bacteria, {summary['n_archaea']} archaea) from "
        f"{len(reps)} GTDB R232 species reps",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbox_finder.mining.pilot_genomes")
    p.add_argument("--crosswalk", default=DEFAULT_CROSSWALK)
    p.add_argument("--out", default=PILOT_MANIFEST)
    p.add_argument("--provenance", default=PILOT_PROVENANCE)
    p.add_argument("--report", default=PILOT_REPORT)
    p.add_argument("--seed", type=int, default=provenance.DEFAULT_SEED)
    p.add_argument("--n-target", type=int, default=DEFAULT_N_TARGET)
    p.add_argument("--per-phylum-cap", type=int, default=DEFAULT_PER_PHYLUM_CAP)
    p.add_argument("--min-phyla", type=int, default=DEFAULT_MIN_PHYLA)
    p.add_argument("--min-archaea", type=int, default=DEFAULT_MIN_ARCHAEA)
    p.add_argument("--env-lock", default=None)
    a = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    return build(
        crosswalk_path=a.crosswalk,
        out_parquet=a.out,
        provenance_path=a.provenance,
        report_path=a.report,
        seed=a.seed,
        n_target=a.n_target,
        per_phylum_cap=a.per_phylum_cap,
        min_phyla=a.min_phyla,
        min_archaea=a.min_archaea,
        env_lock=a.env_lock,
    )


if __name__ == "__main__":
    raise SystemExit(main())
