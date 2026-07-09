"""priors.py — reconcile the union novelty prior + NCBI→GTDB projection + audit (P0-14).

The **union novelty prior** (PRD §7.2) is the reference set of *already-known* T-box
occurrences against which every genome-wide discovery is scored for novelty (§13.3,
GATE-3) and from which every negative/decoy pool is masked (§9.1). It is the **union** of
three sources, deliberately broader than TBDB alone so §5's ascertainment bias is not
re-encoded as ground truth:

  1. **TBDB** — the ingested master corpus (P0-12; ``master_clean_v0.parquet``), which
     already subsumes the annotated RF00230 and Vitreschak sub-datasets as tagged rows.
  2. **RF00230-only hits** — Rfam RF00230 loci (``RF00230_master.fa``) whose accession is
     absent from TBDB: additional *masking loci* (§9.1 "at locus resolution where
     coordinates exist"); they carry no in-repo taxonomy, so they credit no clade.
  3. A **curated literature-occurrence-by-clade artifact** built here from published
     non-CM occurrences (Vitreschak 2008, DOI:10.1261/rna.819308).

**Projection (PRD §7.2:163).** Every prior record is projected into the one governing
GTDB release (R232, pinned in P0-13) with NCBI names *demoted to a display label*, so a
known-T-box lineage cannot be mis-scored novel through an NCBI→GTDB renaming/splitting
artifact (Firmicutes→Bacillota\\*; Proteobacteria→Pseudomonadota/Myxococcota/
Desulfobacterota, routed by class). The corpus carries **no genome-assembly accession**
(only a nucleotide accession + NCBI ``TaxId``), and genome-resolution GTDB-Tk placement
is a P6 operation (§13.2); so P0-14 does the §7.2 *"matched at the finest available
taxonomic resolution"* tier — **name projection at phylum rank** — for every record.
Split taxa are **over-credited** (all GTDB daughter phyla of a split NCBI phylum), because
conservatism is directional: over-crediting a prior costs recall (safe); under-crediting
fabricates novelty (unsafe). The authoritative P0-14 deliverable is the **no-prior-record
phylum list**; sub-phylum (class/order) novelty is finalized at P6 with GTDB-Tk placement.

**§10.1 evidence gate (Stops).** The four non-CM occurrence-by-clade claims are
high-stakes distribution facts requiring ≥2 agreeing sources. A multi-agent literature
verification (2026-07-09) confirmed three at ≥2 mutually-independent sources — δ-proteo →
**Desulfobacterota** (Geobacter/Pelobacter; Vitreschak 2008 + Gutiérrez-Preciado 2009,
DOI:10.1128/MMBR.00026-08), Deinococcus–Thermus → **Deinococcota** (Vitreschak 2008 +
Green/Grundy/Henkin 2010, DOI:10.1016/j.febslet.2009.11.056), Chloroflexi →
**Chloroflexota** (Vitreschak 2008 + Gutiérrez-Preciado 2009) — and found **Dictyoglomi**
(Dictyoglomota) to be **single-source** (Vitreschak 2008 only; two later surveys omit it).
Per the user decision (2026-07-09, CLAUDE.md §10.1), Dictyoglomi is **withheld** from the
curated literature artifact and flagged in the audit; Dictyoglomota nonetheless remains
has-prior via 10 TBDB corpus loci (*D. thermophilum* / *D. turgidum*, three sub-datasets),
so the no-prior list is unchanged either way.

Stdlib-only at import (like :mod:`tbox_finder.taxonomy`): the pure projection logic is
unit-tested in a bare CI env; :mod:`pandas` is imported lazily inside
:func:`reconcile_union_prior` for the parquet join.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tbox_finder import taxonomy
from tbox_finder.provenance import SCHEMA_VERSION, git_sha, write_provenance

# --- Standard I/O paths (overridable on the CLI) --------------------------------------
CORPUS_PARQUET = Path("data/processed/master_clean_v0.parquet")
RF00230_FASTA = Path("data/external/refs/RF00230_master.fa")
GTDB_DIR = taxonomy.GTDB_DIR
PRIORS_DIR = Path("data/processed/priors")
AUDIT_DIR = Path("data/processed/audits")
UNION_PRIOR_NAME = "union_prior.parquet"
UNION_PRIOR_PROVENANCE = "union_prior.provenance.json"
AUDIT_REPORT_NAME = "union_prior_report.json"

ACCESSED_DATE = taxonomy.ACCESSED_DATE  # "2026-07-09"
VITRESCHAK_DOI = "DOI:10.1261/rna.819308"

# GTDB taxonomy TSVs (genome → lineage) fetched on demand to build the R232 clade universe.
_TAXONOMY_FILES = ("bac120_taxonomy_r232.tsv", "ar53_taxonomy_r232.tsv")

# --- NCBI → GTDB phylum projection map ------------------------------------------------
# Values are GTDB phylum *base* names; a record is credited every universe phylum equal to
# a base or prefixed ``base + "_"`` (so Firmicutes credits Bacillota, Bacillota_A, …).
# Over-crediting the polyphyletic suffixes is the safe direction (§7.2:163). Every credited
# name is intersected with the real R232 universe, so a base absent from R232 is dropped
# and a *fully* absent mapping is surfaced (:func:`validate_projection_map`).
NCBI_TO_GTDB_PHYLUM_BASES: dict[str, tuple[str, ...]] = {
    "Firmicutes": ("Bacillota",),
    # GTDB folds the Tenericutes/Mollicutes into Bacillota (Mycoplasmatota is the ICNP
    # alt name); intersect-with-universe keeps whichever R232 actually uses.
    "Tenericutes": ("Bacillota", "Mycoplasmatota"),
    "Actinobacteria": ("Actinomycetota", "Actinobacteriota"),
    "Chloroflexi": ("Chloroflexota",),
    "Deinococcus-Thermus": ("Deinococcota",),
    "Synergistetes": ("Synergistota",),
    "Dictyoglomi": ("Dictyoglomota",),
    "Nitrospirae": ("Nitrospirota",),
    "Bacteroidetes": ("Bacteroidota",),
    "Spirochaetes": ("Spirochaetota",),
    "Fusobacteria": ("Fusobacteriota",),
    "Cyanobacteria": ("Cyanobacteriota",),
    "Elusimicrobia": ("Elusimicrobiota",),
    "Armatimonadetes": ("Armatimonadota",),
    "Candidatus Atribacteria": ("Atribacterota",),
    "Candidatus Omnitrophica": ("Omnitrophota", "Omnitrophica"),
    # CPR (Candidate Phyla Radiation) → GTDB phylum Patescibacteriota (R232 ICNP name).
    "Candidatus Saccharibacteria": ("Patescibacteriota",),
    "Candidatus Collierbacteria": ("Patescibacteriota",),
    "Candidatus Wolfebacteria": ("Patescibacteriota",),
    "Candidatus Campbellbacteria": ("Patescibacteriota",),
    "Candidatus Gottesmanbacteria": ("Patescibacteriota",),
    "Candidatus Woesebacteria": ("Patescibacteriota",),
}

# Proteobacteria is polyphyletic in GTDB — route by NCBI class. Deltaproteobacteria is
# over-credited across all its GTDB daughter phyla (the literature occurrence is confirmed
# only for Desulfobacterota, but over-crediting the projection is the safe direction).
_PROTEO_CLASS_TO_GTDB_BASES: dict[str, tuple[str, ...]] = {
    "Alphaproteobacteria": ("Pseudomonadota",),
    "Betaproteobacteria": ("Pseudomonadota",),
    "Gammaproteobacteria": ("Pseudomonadota",),
    "Epsilonproteobacteria": ("Campylobacterota",),
    # NCBI Deltaproteobacteria splits across several GTDB phyla; the PRD §7.2:163 names
    # Myxococcota + Desulfobacterota. Over-credit both (the confirmed literature occurrence
    # is Desulfobacterota — Geobacter/Pelobacter — but the projection stays safe-side).
    "Deltaproteobacteria": ("Desulfobacterota", "Myxococcota"),
}
# An unknown/absent proteobacterial class over-credits the union of every daughter phylum.
_PROTEO_ALL_BASES: tuple[str, ...] = tuple(
    dict.fromkeys(b for bases in _PROTEO_CLASS_TO_GTDB_BASES.values() for b in bases)
)

# NCBI phyla that are eukaryotic (absent from GTDB by construction): a T-box annotation
# here is a mis-assignment and is correctly unprojectable — it credits no bacterial clade.
EUKARYOTIC_NCBI_PHYLA: frozenset[str] = frozenset(
    {
        "Arthropoda",
        "Ascomycota",
        "Basidiomycota",
        "Chordata",
        "Nematoda",
        "Streptophyta",
        "Chlorophyta",
        "Mucoromycota",
    }
)

#: GTDB phylum bases that MUST end up has-prior — the corpus's known-T-box lineages. The
#: no-false-novelty runtime gate (§10.3) fails loud if any is missing from has-prior.
KNOWN_TBOX_GTDB_BASES: tuple[str, ...] = (
    "Bacillota",
    "Actinomycetota",
    "Chloroflexota",
    "Deinococcota",
    "Dictyoglomota",
    "Desulfobacterota",
    "Pseudomonadota",
    "Synergistota",
)


# --- Curated literature-occurrence-by-clade artifact (§7.2; §10.1 gate) ----------------
@dataclass(frozen=True)
class LiteratureClade:
    """One published non-CM occurrence-by-clade record with its ≥2-source evidence."""

    ncbi_clade: str
    gtdb_phylum: str  # the confirmed GTDB target (validated against the R232 universe)
    evidence_status: str  # "confirmed_2plus_independent" | "single_source"
    sources: tuple[str, ...]
    note: str
    withheld: bool = False  # True → recorded in the audit, excluded from the credited set


#: Verified 2026-07-09 (CLAUDE.md §10.1, ≥2 mutually-independent agreeing sources).
LITERATURE_CLADES: tuple[LiteratureClade, ...] = (
    LiteratureClade(
        ncbi_clade="delta-proteobacteria",
        gtdb_phylum="Desulfobacterota",
        evidence_status="confirmed_2plus_independent",
        sources=(VITRESCHAK_DOI, "DOI:10.1128/MMBR.00026-08"),
        note=(
            "Confirmed for the Geobacter/Pelobacter lineage (Desulfuromonadales; GTDB "
            "Desulfobacterota/Desulfuromonadia). Myxococcota is NOT independently "
            "supported — the projection over-credits it for false-novelty safety, but "
            "the literature occurrence credits only Desulfobacterota."
        ),
    ),
    LiteratureClade(
        ncbi_clade="Deinococcus-Thermus",
        gtdb_phylum="Deinococcota",
        evidence_status="confirmed_2plus_independent",
        sources=(VITRESCHAK_DOI, "DOI:10.1016/j.febslet.2009.11.056"),
        note="Gelfand/Vitreschak 2008 + independent Merino-Henkin program (FEBS 2010).",
    ),
    LiteratureClade(
        ncbi_clade="Chloroflexi",
        gtdb_phylum="Chloroflexota",
        evidence_status="confirmed_2plus_independent",
        sources=(VITRESCHAK_DOI, "DOI:10.1128/MMBR.00026-08"),
        note="Two independent comparative-genomic pipelines converge on Chloroflexus.",
    ),
    LiteratureClade(
        ncbi_clade="Dictyoglomi",
        gtdb_phylum="Dictyoglomota",
        evidence_status="single_source",
        sources=(VITRESCHAK_DOI,),
        note=(
            "SINGLE-SOURCE: only Vitreschak 2008; Green/Grundy/Henkin 2010 "
            "(DOI:10.1016/j.febslet.2009.11.056) and TBDB/Zhu 2021 omit it. WITHHELD "
            "from the curated literature set per §10.1 (user decision 2026-07-09). "
            "Dictyoglomota remains has-prior via 10 TBDB corpus loci "
            "(D. thermophilum / D. turgidum, three sub-datasets)."
        ),
        withheld=True,
    ),
)


# --- Pure projection logic (stdlib-only; unit-tested) ---------------------------------
@dataclass(frozen=True)
class ProjectionResult:
    """The GTDB projection of one record's NCBI lineage."""

    credited_phyla: frozenset[str]  # GTDB phyla this record credits (∅ if unprojectable)
    method: str
    projectable: bool


def parse_gtdb_lineage(lineage: str) -> dict[str, str]:
    """Parse a GTDB ``d__…;p__…;c__…;o__…;f__…;g__…;s__…`` string into a rank→name dict.

    Empty ranks (``p__`` with nothing after) are omitted. Rank keys are the full words
    ``domain/phylum/class/order/family/genus/species``.
    """
    ranks = {
        "d": "domain",
        "p": "phylum",
        "c": "class",
        "o": "order",
        "f": "family",
        "g": "genus",
        "s": "species",
    }
    out: dict[str, str] = {}
    for token in lineage.strip().split(";"):
        token = token.strip()
        if len(token) < 3 or token[1:3] != "__":
            continue
        prefix, name = token[0], token[3:].strip()
        rank = ranks.get(prefix)
        if rank and name:
            out[rank] = name
    return out


def load_clade_universe(taxonomy_lines: Iterable[str]) -> dict[str, set[str]]:
    """Build ``{rank: set of GTDB clade names}`` from genome→lineage TSV lines.

    Each line is ``<genome_accession>\\t<d__…;s__…>``. Streams the (large) taxonomy files
    and enumerates the release's clade universe at phylum/class/order (and finer), which
    is the complement space for the no-prior-record list.
    """
    universe: dict[str, set[str]] = {
        "domain": set(),
        "phylum": set(),
        "class": set(),
        "order": set(),
        "family": set(),
        "genus": set(),
    }
    for line in taxonomy_lines:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        for rank, name in parse_gtdb_lineage(parts[1]).items():
            if rank in universe:
                universe[rank].add(name)
    return universe


def _match_universe(bases: Sequence[str], universe_phyla: set[str]) -> frozenset[str]:
    """GTDB phyla equal to, or a suffix-split (``base_…``) of, any base — real ones only."""
    return frozenset(p for p in universe_phyla for b in bases if p == b or p.startswith(b + "_"))


def _norm(value: object) -> str | None:
    """Normalise a possibly-NaN/None/blank cell to a stripped ``str`` or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    return text


def project_ncbi_phylum(
    ncbi_phylum: object,
    ncbi_class: object,
    *,
    universe_phyla: set[str],
) -> ProjectionResult:
    """Project one record's NCBI (phylum, class) to its credited GTDB phyla.

    Conservative + over-crediting on split taxa (§7.2:163). ``method`` records how the
    projection was made / why it failed, for the audit:
    ``name_projection`` / ``name_projection_class_routed`` (projectable);
    ``null_lineage`` (no NCBI phylum), ``eukaryotic_nonbacterial`` (absent from GTDB),
    ``unmapped_ncbi_phylum`` (bacterial but not in the map — unprojectable-but-known),
    ``mapped_but_absent_in_release`` (mapped, but no candidate exists in R232) — all
    unprojectable.
    """
    phylum = _norm(ncbi_phylum)
    if phylum is None:
        return ProjectionResult(frozenset(), "null_lineage", False)
    if phylum in EUKARYOTIC_NCBI_PHYLA:
        return ProjectionResult(frozenset(), "eukaryotic_nonbacterial", False)

    if phylum == "Proteobacteria":
        cls = _norm(ncbi_class)
        bases = (
            _PROTEO_CLASS_TO_GTDB_BASES.get(cls, _PROTEO_ALL_BASES) if cls else _PROTEO_ALL_BASES
        )
        method = "name_projection_class_routed"
    elif phylum in NCBI_TO_GTDB_PHYLUM_BASES:
        bases = NCBI_TO_GTDB_PHYLUM_BASES[phylum]
        method = "name_projection"
    else:
        return ProjectionResult(frozenset(), "unmapped_ncbi_phylum", False)

    credited = _match_universe(bases, universe_phyla)
    if not credited:
        return ProjectionResult(frozenset(), "mapped_but_absent_in_release", False)
    return ProjectionResult(credited, method, True)


def validate_projection_map(universe_phyla: set[str]) -> dict[str, list[str]]:
    """Return ``{ncbi_phylum: [absent bases]}`` for every mapping with NO R232 target.

    A non-empty result means the curated map is broken for a phylum that could carry
    records; the load-bearing subset is additionally gated at runtime by
    :func:`assert_no_false_novelty`.
    """
    broken: dict[str, list[str]] = {}
    for ncbi, bases in NCBI_TO_GTDB_PHYLUM_BASES.items():
        if not _match_universe(bases, universe_phyla):
            broken[ncbi] = list(bases)
    for cls, bases in _PROTEO_CLASS_TO_GTDB_BASES.items():
        if not _match_universe(bases, universe_phyla):
            broken.setdefault(f"Proteobacteria/{cls}", list(bases))
    return broken


def credited_literature_phyla(universe_phyla: set[str]) -> frozenset[str]:
    """GTDB phyla credited by the *non-withheld* literature clades (validated vs R232)."""
    return frozenset(
        lc.gtdb_phylum
        for lc in LITERATURE_CLADES
        if not lc.withheld and lc.gtdb_phylum in universe_phyla
    )


def derive_prior_phyla(
    credited_per_record: Iterable[frozenset[str]],
    literature_phyla: frozenset[str],
    universe_phyla: set[str],
) -> tuple[list[str], list[str]]:
    """Return ``(has_prior_phyla, no_prior_phyla)`` — the re-derived no-prior list.

    ``has_prior`` = the union of every record's credited phyla ∪ the credited literature
    phyla, intersected with the real R232 universe. ``no_prior`` = the universe minus
    has-prior (the novel-eligible phyla). Both sorted for determinism.
    """
    has_prior: set[str] = set(literature_phyla)
    for credited in credited_per_record:
        has_prior |= credited
    has_prior &= universe_phyla
    no_prior = universe_phyla - has_prior
    return sorted(has_prior), sorted(no_prior)


def assert_no_false_novelty(
    has_prior_phyla: Iterable[str],
    universe_phyla: set[str],
    known_bases: Iterable[str] = KNOWN_TBOX_GTDB_BASES,
) -> None:
    """Fail-loud (§10.3) unless every known-T-box GTDB phylum is has-prior.

    This is the anti-false-novelty gate: a known-T-box lineage present in the corpus must
    never land on the no-prior list through an NCBI→GTDB renaming/splitting artifact. The
    check requires **every** GTDB split-daughter of each known base (e.g. both Bacillota
    and Bacillota_I) to be has-prior — the projection over-credits all splits, so a missing
    one signals a regression, not a real absence.
    """
    has_prior = set(has_prior_phyla)
    missing: list[str] = []
    for base in known_bases:
        candidates = {p for p in universe_phyla if p == base or p.startswith(base + "_")}
        if not candidates:
            missing.append(f"{base} (absent from R232 universe)")
        else:
            missing.extend(sorted(candidates - has_prior))
    if missing:
        raise ValueError(
            "no-false-novelty gate FAILED (CLAUDE.md §10.3): known-T-box GTDB phyla "
            f"missing from has-prior: {missing}. A renaming/splitting artifact would "
            "mis-score these lineages novel — do NOT ship this union prior."
        )


def parse_fasta_loci(fasta_lines: Iterable[str]) -> list[tuple[str, int | None, int | None]]:
    """Parse ``>accession.version:start-end`` headers → ``(accession, start, end)``.

    The accession is versionless (``rsplit('.', 1)[0]``, matching the corpus
    ``accession_name``). ``start``/``end`` are ``None`` when a header carries no coords.
    """
    loci: list[tuple[str, int | None, int | None]] = []
    for line in fasta_lines:
        if not line.startswith(">"):
            continue
        fields = line[1:].strip().split()
        if not fields:
            continue  # bare ">" header line — no accession to parse
        header = fields[0]
        token, _, coords = header.partition(":")
        accession = token.rsplit(".", 1)[0] if "." in token else token
        start = end = None
        if "-" in coords:
            a, _, b = coords.partition("-")
            if a.isdigit() and b.isdigit():
                start, end = int(a), int(b)
        loci.append((accession, start, end))
    return loci


# --- Reconciliation (pandas lazy) -----------------------------------------------------
@dataclass
class ReconcileOutputs:
    """Paths written by :func:`reconcile_union_prior`."""

    union_prior: Path
    audit_report: Path
    provenance: Path
    audit: dict = field(default_factory=dict)


def reconcile_union_prior(
    *,
    corpus_parquet: str | Path = CORPUS_PARQUET,
    rf00230_fasta: str | Path = RF00230_FASTA,
    gtdb_dir: str | Path = GTDB_DIR,
    priors_dir: str | Path = PRIORS_DIR,
    audit_dir: str | Path = AUDIT_DIR,
    download: bool = True,
) -> ReconcileOutputs:
    """Build ``union_prior.parquet`` + the unprojectable audit + provenance (P0-14).

    Reads the TBDB corpus, fetches (MD5-verified) the R232 taxonomy TSVs to build the GTDB
    clade universe, name-projects every record, folds in the RF00230-only masking loci and
    the curated literature clades, and writes the per-record union prior + the audit report
    (unprojectable fraction, has-prior/no-prior phylum lists, projection map, literature
    evidence). Runs the no-false-novelty gate (fail-loud) before writing.
    """
    import pandas as pd  # lazy — keeps module import stdlib-only for the unit tests

    corpus_parquet = Path(corpus_parquet)
    rf00230_fasta = Path(rf00230_fasta)
    gtdb_dir = Path(gtdb_dir)
    priors_dir = Path(priors_dir)
    audit_dir = Path(audit_dir)

    # 1. GTDB clade universe (fetch on demand; MD5-verified via taxonomy.ensure_file).
    tax_shas: dict[str, str] = {}
    universe_lines: list[str] = []
    for gf in taxonomy.FILES:
        if gf.name not in _TAXONOMY_FILES:
            continue
        if download:
            sha, _ = taxonomy.ensure_file(gf, gtdb_dir)
            tax_shas[gf.name] = sha
        path = gtdb_dir / gf.name
        with path.open("r", encoding="utf-8") as fh:
            universe_lines.extend(fh)
    universe = load_clade_universe(universe_lines)
    universe_phyla = universe["phylum"]
    map_gaps = validate_projection_map(universe_phyla)

    # 2. Project every TBDB corpus record (finest-available-rank = phylum name projection).
    # Iterate positional tuples over the needed columns: ``class`` is a Python keyword, so
    # attribute-style ``itertuples`` would mangle it (silently dropping the class routing).
    df = pd.read_parquet(corpus_parquet)
    needed = [
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "GBSeq_organism",
        "type",
        "source",
        "accession_name",
        "locus_start",
        "locus_end",
        "TaxId",
    ]
    records: list[dict] = []
    credited_per_record: list[frozenset[str]] = []
    method_counts: dict[str, int] = {}
    for (
        phylum,
        ncbi_class,
        order,
        family,
        genus,
        organism,
        tbox_type,
        source,
        accession,
        lstart,
        lend,
        taxid,
    ) in df[needed].itertuples(index=False, name=None):
        proj = project_ncbi_phylum(phylum, ncbi_class, universe_phyla=universe_phyla)
        credited_per_record.append(proj.credited_phyla)
        method_counts[proj.method] = method_counts.get(proj.method, 0) + 1
        records.append(
            {
                "record_kind": "tbdb_locus",
                "source": _norm(source),
                "accession": _norm(accession),
                "locus_start": _int_or_none(lstart),
                "locus_end": _int_or_none(lend),
                "taxid": _int_or_none(taxid),
                "ncbi_phylum": _norm(phylum),
                "ncbi_class": _norm(ncbi_class),
                "ncbi_order": _norm(order),
                "ncbi_family": _norm(family),
                "ncbi_genus": _norm(genus),
                "ncbi_organism": _norm(organism),
                "tbox_type": _norm(tbox_type),
                "gtdb_phyla_credited": ";".join(sorted(proj.credited_phyla)),
                "projection_method": proj.method,
                "projectable": proj.projectable,
                "has_prior_credit": bool(proj.credited_phyla),
                "citation": None,
                "evidence_status": None,
            }
        )
    n_tbdb = len(records)

    # 3. RF00230-only masking loci — accessions absent from the corpus (no taxonomy).
    corpus_accessions = {a for a in (_norm(x) for x in df["accession_name"]) if a}
    rf_loci = parse_fasta_loci(_read_lines(rf00230_fasta))
    rf_only_accessions: set[str] = set()
    n_rf_only_loci = 0
    for accession, start, end in rf_loci:
        if accession in corpus_accessions:
            continue
        rf_only_accessions.add(accession)
        n_rf_only_loci += 1
        records.append(
            {
                "record_kind": "rf00230_only_locus",
                "source": "RF00230_master.fa",
                "accession": accession,
                "locus_start": start,
                "locus_end": end,
                "taxid": None,
                "ncbi_phylum": None,
                "ncbi_class": None,
                "ncbi_order": None,
                "ncbi_family": None,
                "ncbi_genus": None,
                "ncbi_organism": None,
                "tbox_type": None,
                "gtdb_phyla_credited": "",
                "projection_method": "accession_only",
                "projectable": False,
                "has_prior_credit": False,
                "citation": None,
                "evidence_status": None,
            }
        )

    # 4. Curated literature clades (§10.1). Withheld clades are audited, not credited.
    literature_phyla = credited_literature_phyla(universe_phyla)
    for lc in LITERATURE_CLADES:
        if lc.withheld:
            continue
        in_universe = lc.gtdb_phylum in universe_phyla
        records.append(
            {
                "record_kind": "literature_clade",
                "source": f"literature:{lc.ncbi_clade}",
                "accession": None,
                "locus_start": None,
                "locus_end": None,
                "taxid": None,
                "ncbi_phylum": lc.ncbi_clade,
                "ncbi_class": None,
                "ncbi_order": None,
                "ncbi_family": None,
                "ncbi_genus": None,
                "ncbi_organism": None,
                "tbox_type": None,
                "gtdb_phyla_credited": lc.gtdb_phylum if in_universe else "",
                "projection_method": "literature",
                "projectable": in_universe,
                "has_prior_credit": in_universe,
                "citation": " ; ".join(lc.sources),
                "evidence_status": lc.evidence_status,
            }
        )

    # 5. Re-derive has-prior / no-prior phyla; run the no-false-novelty gate.
    has_prior_phyla, no_prior_phyla = derive_prior_phyla(
        credited_per_record, literature_phyla, universe_phyla
    )
    assert_no_false_novelty(has_prior_phyla, universe_phyla)

    # 6. Assemble + write the union prior parquet.
    priors_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = priors_dir / UNION_PRIOR_NAME
    prior_df = pd.DataFrame.from_records(records)
    for col in ("locus_start", "locus_end", "taxid"):
        prior_df[col] = prior_df[col].astype("Int64")
    prior_df.to_parquet(out_parquet, index=False)

    # 7. Audit report + provenance.
    unprojectable_taxonomic = sum(
        c
        for m, c in method_counts.items()
        if m not in {"name_projection", "name_projection_class_routed"}
    )
    audit = _build_audit(
        n_tbdb=n_tbdb,
        n_rf_only_loci=n_rf_only_loci,
        n_rf_only_accessions=len(rf_only_accessions),
        method_counts=method_counts,
        unprojectable_taxonomic=unprojectable_taxonomic,
        has_prior_phyla=has_prior_phyla,
        no_prior_phyla=no_prior_phyla,
        universe=universe,
        map_gaps=map_gaps,
        tax_shas=tax_shas,
        total_records=len(records),
    )
    out_audit = audit_dir / AUDIT_REPORT_NAME
    out_audit.write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    out_provenance = priors_dir / UNION_PRIOR_PROVENANCE
    write_provenance(
        out_provenance,
        rule="workflow/rules/data.smk :: reconcile_union_prior",
        script="src/tbox_finder/priors.py",
        inputs=[corpus_parquet, rf00230_fasta, gtdb_dir / "provenance.json"],
        outputs=[out_parquet, out_audit],
        env_lock="envs/data.conda-lock.yml",
        adr="PRD §7.2 (ADR-0006 pending, P0-29)",
        extra={
            "gtdb_release": taxonomy.GTDB_RELEASE,
            "union_prior_records": len(records),
            "unprojectable_fraction_taxonomic": audit["projection"][
                "unprojectable_fraction_taxonomic"
            ],
            "n_no_prior_phyla": len(no_prior_phyla),
        },
    )
    return ReconcileOutputs(out_parquet, out_audit, out_provenance, audit)


def _build_audit(
    *,
    n_tbdb: int,
    n_rf_only_loci: int,
    n_rf_only_accessions: int,
    method_counts: Mapping[str, int],
    unprojectable_taxonomic: int,
    has_prior_phyla: Sequence[str],
    no_prior_phyla: Sequence[str],
    universe: Mapping[str, set[str]],
    map_gaps: Mapping[str, list[str]],
    tax_shas: Mapping[str, str],
    total_records: int,
) -> dict:
    """Assemble the union-prior audit report (the §7.2 unprojectable audit + no-prior list)."""
    projectable_taxonomic = method_counts.get("name_projection", 0) + method_counts.get(
        "name_projection_class_routed", 0
    )
    denom = n_tbdb  # the taxonomic records (TBDB corpus) the projection applies to
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "P0-14 union novelty prior reconciliation + NCBI→GTDB projection + "
            "unprojectable audit. The union prior (TBDB + RF00230-only loci + curated "
            "literature) projected into the governing GTDB release at finest-available "
            "(phylum) resolution; genome-resolution GTDB-Tk placement is a P6 op (§13.2)."
        ),
        "rule": "workflow/rules/data.smk :: reconcile_union_prior",
        "script": "src/tbox_finder/priors.py",
        "prd": "§7.2, §4, §13.3",
        "adr": "PRD §7.2 (ADR-0006 pending, P0-29)",
        "git_sha": git_sha(),
        "accessed_date": ACCESSED_DATE,
        "gtdb_release": taxonomy.GTDB_RELEASE,
        "gtdb_taxonomy_sha256": dict(tax_shas),
        "totals": {
            "union_prior_records": total_records,
            "tbdb_locus": n_tbdb,
            "rf00230_only_locus": n_rf_only_loci,
            "rf00230_only_accessions": n_rf_only_accessions,
            "literature_clade": sum(1 for lc in LITERATURE_CLADES if not lc.withheld),
        },
        "projection": {
            "denominator_taxonomic_records": denom,
            "projectable_taxonomic": projectable_taxonomic,
            "unprojectable_taxonomic": unprojectable_taxonomic,
            "unprojectable_fraction_taxonomic": (
                round(unprojectable_taxonomic / denom, 6) if denom else 0.0
            ),
            "by_method": dict(sorted(method_counts.items())),
            "note": (
                "RF00230-only loci are accession-only masking loci (no taxonomy) and are "
                "reported separately, not in the taxonomic unprojectable fraction."
            ),
        },
        "has_prior_phyla": list(has_prior_phyla),
        "n_has_prior_phyla": len(has_prior_phyla),
        "no_prior_phyla": list(no_prior_phyla),
        "n_no_prior_phyla": len(no_prior_phyla),
        "r232_universe": {
            "n_phyla": len(universe.get("phylum", ())),
            "n_class": len(universe.get("class", ())),
            "n_order": len(universe.get("order", ())),
        },
        "ncbi_to_gtdb_phylum_map": {k: list(v) for k, v in NCBI_TO_GTDB_PHYLUM_BASES.items()},
        "proteobacteria_class_routing": {
            k: list(v) for k, v in _PROTEO_CLASS_TO_GTDB_BASES.items()
        },
        "projection_map_gaps": dict(map_gaps),
        "literature_occurrence_artifact": [
            {
                "ncbi_clade": lc.ncbi_clade,
                "gtdb_phylum": lc.gtdb_phylum,
                "evidence_status": lc.evidence_status,
                "sources": list(lc.sources),
                "note": lc.note,
            }
            for lc in LITERATURE_CLADES
            if not lc.withheld
        ],
        "single_source_flagged_withheld": [
            {
                "ncbi_clade": lc.ncbi_clade,
                "gtdb_phylum": lc.gtdb_phylum,
                "evidence_status": lc.evidence_status,
                "sources": list(lc.sources),
                "note": lc.note,
                "has_prior_via_corpus": lc.gtdb_phylum in set(has_prior_phyla),
            }
            for lc in LITERATURE_CLADES
            if lc.withheld
        ],
        "conservative_rule": (
            "unprojectable-but-known clade treated as having a prior (§7.2:163); split "
            "taxa over-credited across GTDB daughter phyla (over-crediting costs recall — "
            "safe; under-crediting fabricates novelty — unsafe)."
        ),
        "sub_phylum_deferral": (
            "class/order no-prior determination deferred to P6 GTDB-Tk genome-resolution "
            "placement (§13.2/§13.3); P0-14 publishes the authoritative phylum no-prior list."
        ),
    }


def _int_or_none(value: object) -> int | None:
    """Coerce a possibly-NaN numeric cell to ``int`` or ``None`` (no fabricated 0)."""
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except (TypeError, ValueError):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return fh.readlines()


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m tbox_finder.priors --priors-dir data/processed/priors``."""
    parser = argparse.ArgumentParser(
        description="Reconcile the union novelty prior + NCBI→GTDB projection + audit (P0-14)."
    )
    parser.add_argument("--corpus", default=str(CORPUS_PARQUET))
    parser.add_argument("--rf00230", default=str(RF00230_FASTA))
    parser.add_argument("--gtdb-dir", default=str(GTDB_DIR))
    parser.add_argument("--priors-dir", default=str(PRIORS_DIR))
    parser.add_argument("--audit-dir", default=str(AUDIT_DIR))
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="use already-fetched GTDB taxonomy TSVs; do not touch the network",
    )
    args = parser.parse_args(argv)
    out = reconcile_union_prior(
        corpus_parquet=args.corpus,
        rf00230_fasta=args.rf00230,
        gtdb_dir=args.gtdb_dir,
        priors_dir=args.priors_dir,
        audit_dir=args.audit_dir,
        download=not args.no_download,
    )
    proj = out.audit["projection"]
    print(
        f"union prior: {out.audit['totals']['union_prior_records']} records -> {out.union_prior}\n"
        f"  unprojectable(taxonomic): {proj['unprojectable_taxonomic']}/"
        f"{proj['denominator_taxonomic_records']} "
        f"({proj['unprojectable_fraction_taxonomic']:.4f})\n"
        f"  has-prior phyla: {out.audit['n_has_prior_phyla']}; "
        f"no-prior phyla: {out.audit['n_no_prior_phyla']}  -> {out.audit_report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
