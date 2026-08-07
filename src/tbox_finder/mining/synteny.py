"""ADR-0006 **D4** — criterion (c), downstream-aaRS synteny, over the acquired NCBI GFFs.

P3-15′-c-i put 339 md5-verified ``_genomic.gff.gz`` files on disk and a pure-stdlib GFF3
reader beside them (:mod:`tbox_finder.mining.gff3`).  This module is the half that makes a
*judgement*: given a mined candidate's contig span, walk the annotation downstream on the
resolved strand and decide whether D4's pass condition holds.

D4 (locked rule)
    Pass = the first downstream **same-strand** CDS start within **500 bp** of the T-box
    element 3′ end encodes an **aaRS / amino-acid-biosynthesis / transport / transamidation**
    function.  The 3′ end is terminator-inclusive for class-I (transcriptional) T-boxes;
    class-II (translational) T-boxes have no terminator and abut/overlap the start codon
    (distance ≈ 0), handled here as the ``distance <= 0`` case.

Three things D4 deliberately does **not** specify, and where each one lives here
-------------------------------------------------------------------------------
1. **The product-name vocabulary.**  ADR-0006 pins the four functional *classes* and no
   attribute keys, no product-name list and no Pfam/KO cutoffs — so the judgement is this
   step's, and widening it has to be a visible diff.  It lives in :data:`CLASS_RULES` and
   :data:`EXCLUSIONS`, built **precision-first** and grounded in the T-box literature:

   * the four classes are the ones the comparative-genomics and structural literature
     reports for T-box-regulated operons — *"biosynthesis, transport, aminoacylation,
     transamidation"* [DOI:10.1002/wrna.1273] (accessed 2026-08-07); *"the majority of
     T-box-regulated genes encode aminoacyl-tRNA synthetases … two other groups are amino
     acid biosynthetic genes and transporters"* [PMID:18359782] (accessed 2026-08-07);
     *"aminoacyl-tRNA synthetases, proteins of amino acid biosynthetic pathways,
     transporters"* [PMID:19258532] (accessed 2026-08-07); the tRNA-dependent
     transamidation operon [DOI:10.1073/pnas.1220991110] (accessed 2026-08-07).
   * :data:`EXCLUSIONS` is **not** guesswork either — every entry was measured on this
     corpus before it was written down.  An "amino-acid word + enzyme word" rule looks
     reasonable and is badly wrong here: over 12 of the 48 candidate-carrying annotated
     hosts it swallows ``HAMP domain-containing histidine kinase`` (38 CDS, two-component
     signalling), ``serine/threonine protein kinase``, ``glutamine amidotransferase``
     (purine/His/Trp pathways, **not** tRNA-dependent transamidation),
     ``methylated-DNA--[protein]-cysteine S-methyltransferase`` (DNA repair) and
     ``peptide-methionine (S)-S-oxide reductase`` (protein repair).  D4 says specificity
     comes from the gene-identity requirement rather than the distance, so a vocabulary
     that admits those families would defeat the criterion rather than implement it.

2. **The tandem / intervening-ORF carve-out.**  D4 asks for one and pins no number, so it is
   *not* baked in: :func:`resolve_downstream_gene` takes ``max_intervening_orfs`` and
   ``sub_threshold_orf_nt`` as **required keyword arguments with no default** — the round
   supplies them and the value it used rides in the producer's report.  A default here
   would decide which candidates are mined without anyone choosing it.

3. **The Pfam/KO fallback for hypothetical / pseudogenized ORFs.**  D4 routes those to
   targeted HMM profiles.  The profile database is a §10.2 acquisition that has **not**
   landed, so :data:`HMM_FALLBACK_AVAILABLE` is ``False`` and such an ORF resolves to
   :data:`FN_UNJUDGEABLE` ⇒ the candidate's disjunct is ``unavailable`` ⇒ under ADR-0005
   D14 it is **spared, never mined**.  That is the fail-closed direction, and D4's own
   symmetric pseudogene diagnostic is what measures the recall it costs.

The two coordinate conventions this module bridges
--------------------------------------------------
* A mining candidate carries ``accession = "<assembly>:c<contig_index>"`` plus
  ``locus_start``/``locus_end`` that are **0-based half-open** contig coordinates
  (:func:`tbox_finder.mining.homolog_msa.resolve_candidate_sequence`).
* GFF3 columns 4/5 are **1-based inclusive both ends** and its ``seqid`` is the NCBI
  contig accession — *not* ``:c<ci>``.

The bridge is by **identity, never by index**: ``records[ci]``'s FASTA id is looked up among
the GFF's seqids.  Measured on all 48 candidate-carrying annotated hosts, **23 of them serve
their GFF seqids in a different order than the genomic FASTA serves its records**, so an
index-to-index join would silently read the wrong contig on roughly half the corpus; the id
join resolves 134 of 134 needed (host, contig) pairs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from tbox_finder.mining import gff3
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED, STATUS_UNAVAILABLE

__all__ = [
    "CLASS_AARS",
    "CLASS_AA_BIOSYNTHESIS",
    "CLASS_AA_TRANSPORT",
    "CLASS_TRANSAMIDATION",
    "DEFAULT_WINDOW_BP",
    "FN_UNJUDGEABLE",
    "HMM_FALLBACK_AVAILABLE",
    "PASSING_CLASSES",
    "STRAND_POLICIES",
    "DownstreamGene",
    "SyntenyError",
    "classify_gene_identity",
    "gene_symbols",
    "contig_seqid",
    "criterion_c",
    "downstream_cds_on_strand",
    "load_contig_ids",
    "locus_three_prime_1based",
    "resolve_downstream_gene",
    "synteny_status",
]


class SyntenyError(ValueError):
    """A criterion-(c) evaluation could not be carried out as specified."""


# ═════════════════════════════════════════════════════════════════════════════
# D4's four functional classes + the unjudgeable sentinel
# ═════════════════════════════════════════════════════════════════════════════
#: The four classes D4 names.  ``criterion_c`` passes on exactly these.
CLASS_AARS = "aars"
CLASS_AA_BIOSYNTHESIS = "aa_biosynthesis"
CLASS_AA_TRANSPORT = "aa_transport"
CLASS_TRANSAMIDATION = "transamidation"

PASSING_CLASSES: frozenset[str] = frozenset(
    {CLASS_AARS, CLASS_AA_BIOSYNTHESIS, CLASS_AA_TRANSPORT, CLASS_TRANSAMIDATION}
)

#: The ORF exists and is same-strand and in-window, but its function cannot be *decided* from
#: the annotation — ``hypothetical protein``, no identity text at all, or ``pseudo=true``.
#: D4 routes exactly this population to Pfam/KO profiles; see :data:`HMM_FALLBACK_AVAILABLE`.
#: Distinct from ``None`` (no downstream ORF at all), which is an honest ``failed``.
FN_UNJUDGEABLE = "unjudgeable"

#: ADR-0006 D4's blinded-frozen default window.
DEFAULT_WINDOW_BP = 500

#: D4's Pfam/KO fallback for hypothetical / pseudogenized first-downstream ORFs.  The profile
#: database is an unmet §10.2 acquisition (`TODO.md`, the bakta/prokka arm), so the fallback
#: is declared **unavailable** rather than silently skipped: an ORF that needs it resolves to
#: :data:`FN_UNJUDGEABLE`, the disjunct reads ``unavailable``, and ADR-0005 D14 spares the
#: candidate.  Flipping this to ``True`` without wiring a profile search is the fail-open
#: direction and the unit tests refuse it.
HMM_FALLBACK_AVAILABLE = False

#: Strand policies the producer may be run under.  The round0 manifest carries **no strand**
#: (Stage-1 is RC-equivariant and the manifest predates the §6 resolver's output being
#: joined), so "the" downstream gene of a candidate is not defined until one is chosen.
#: ``both`` implements ADR-0005 D15's locked rule for orientation-ambiguous loci — *"carried
#: through on both strands"* — and is the sparing-favouring direction (a pass on either
#: strand spares).  ``plus`` reads the a2 substrate's forward tiling literally.
STRAND_POLICIES: tuple[str, ...] = ("both", "plus", "minus")


# ═════════════════════════════════════════════════════════════════════════════
# The gene-identity vocabulary (D4 delegates it here; see the module docstring)
# ═════════════════════════════════════════════════════════════════════════════
_AA_NAME = (
    r"alanine|arginine|asparagine|aspartate|cysteine|cystine|glutamate|glutamine|glycine|"
    r"histidine|isoleucine|leucine|lysine|methionine|phenylalanine|proline|serine|threonine|"
    r"tryptophan|tyrosine|valine|ornithine|citrulline|homoserine|diaminopimelate|"
    r"branched-chain amino acid|polar amino acid|basic amino acid|neutral amino acid|"
    r"acidic amino acid|amino acid|glutamyl|aspartyl"
)

_AARS_STEM = (
    r"alanyl|arginyl|asparaginyl|aspartyl|cysteinyl|glutamyl|glutaminyl|glycyl|histidyl|"
    r"isoleucyl|leucyl|lysyl|methionyl|phenylalanyl|prolyl|seryl|threonyl|tryptophanyl|"
    r"tyrosyl|valyl"
)

#: Families that a naive "amino-acid word + enzyme word" rule admits and that are **not** any
#: of D4's four classes.  Applied *before* every include rule, so precedence is explicit
#: rather than emergent from regex ordering.  Every entry was observed on this corpus.
EXCLUSIONS: tuple[tuple[str, str], ...] = (
    # Two-component signalling / protein kinases — the single biggest contaminant measured
    # (``HAMP domain-containing histidine kinase`` alone is 38 CDS over 12 hosts).
    (r"histidine kinase", "two-component sensor kinase, not histidine biosynthesis"),
    (r"serine/threonine[- ]protein kinase|serine/threonine kinase", "protein kinase"),
    (r"protein kinase|protein[- ]arginine kinase|autokinase", "protein kinase"),
    (r"response regulator|two-component|HAMP domain|PAS domain", "signal transduction"),
    # Generic hydrolase/protease families named for their catalytic residue.
    (
        r"serine hydrolase|cysteine hydrolase|serine protease|cysteine protease",
        "catalytic-residue family name",
    ),
    (
        r"aspartic protease|serine/threonine phosphatase|serine recombinase",
        "catalytic-residue family name",
    ),
    # Protein / DNA repair and modification.
    (r"methylated-DNA", "DNA repair"),
    (
        r"peptide-methionine|Msr[AB]\b|isoaspartate|protein-glutamate|CheR",
        "protein repair/chemotaxis",
    ),
    (r"ribosomal[- ]protein|ribosomal protein S|ribosome", "ribosome modification"),
    # Pathways that borrow an amino acid as substrate or donor but are not amino-acid
    # biosynthesis, transport, aminoacylation or tRNA-dependent transamidation.
    (
        r"GMP synthase|phosphoribosylformylglycinamidine|carbamoyl-phosphate synthase",
        "purine/pyrimidine",
    ),
    (r"aspartate carbamoyltransferase|dihydroorotase|orotate", "pyrimidine"),
    (r"Pdx[AT]\b|pyridoxal|folylpolyglutamate|dihydrofolate|folate", "vitamin/folate"),
    (r"CoaBC|phosphopantothen|pantothenate|aspartate 1-decarboxylase", "pantothenate/CoA"),
    (r"CDP-diacylglycerol|phosphatidyl", "phospholipid"),
    (r"polysaccharide biosynthesis|capsul|Cps[A-Z]|teichoic", "capsule/cell-envelope"),
    (r"racemase", "D-amino-acid / peptidoglycan"),
    (r"glycine cleavage|Gcv[TPH]", "glycine catabolism / one-carbon"),
    (
        r"adenosylmethionine decarboxylase|methionine adenosyltransferase|spermidine",
        "SAM/polyamine",
    ),
    # Deliberately generic NCBI family labels: they name a fold, not a pathway role.
    (r"aminotransferase family protein|amino acid[- ]binding protein$", "generic family label"),
)

_EXCLUSION_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), why) for p, why in EXCLUSIONS
)

#: aaRS gene symbols — the closed set for the 20 amino acids plus the split subunits.  Checked
#: as exact, case-insensitive symbol matches, so ``argS`` (aaRS) can never be mistaken for the
#: ``arg*`` biosynthesis operon and ``glyQ``/``glyS`` (aaRS) never for ``glyA`` (Gly synthesis).
AARS_GENE_SYMBOLS: frozenset[str] = frozenset(
    {
        "alas",
        "args",
        "asns",
        "asps",
        "cyss",
        "glns",
        "gltx",
        "glyq",
        "glys",
        "hiss",
        "iles",
        "leus",
        "lyss",
        "metg",
        "mets",
        "phes",
        "phet",
        "pros",
        "sers",
        "thrs",
        "trps",
        "tyrs",
        "vals",
        "asps2",
        "lyss1",
        "lyss2",
        "glys1",
        "glys2",
        "pylsn",
    }
)

#: tRNA-dependent transamidation (GatCAB / GatDE).
TRANSAMIDATION_GENE_SYMBOLS: frozenset[str] = frozenset({"gata", "gatb", "gatc", "gatd", "gate"})

#: Amino-acid transport gene symbols observed in the standard operon nomenclature.
TRANSPORT_GENE_SYMBOLS: frozenset[str] = frozenset(
    {
        "livf",
        "livg",
        "livh",
        "livj",
        "livk",
        "livm",
        "brnq",
        "alst",
        "cyca",
        "prop",
        "putp",
        "metn",
        "meti",
        "metq",
        "metp",
        "glnh",
        "glnp",
        "glnq",
        "tcya",
        "tcyb",
        "tcyc",
        "tcyj",
        "tcyk",
        "tcyl",
        "tcym",
        "tcyn",
        "artp",
        "artq",
        "artm",
        "artj",
        "hisj",
        "hism",
        "hisp",
        "hisq",
        "aapj",
        "aapq",
        "aapm",
        "aapp",
        "opuaa",
        "opuab",
    }
)

#: Amino-acid biosynthesis gene symbols, by pathway.  Exact symbol match only.
BIOSYNTHESIS_GENE_SYMBOLS: frozenset[str] = frozenset(
    {
        # branched-chain (Ile/Leu/Val)
        "ilva",
        "ilvb",
        "ilvc",
        "ilvd",
        "ilve",
        "ilvh",
        "ilvn",
        "leua",
        "leub",
        "leuc",
        "leud",
        # aromatic (Trp/Tyr/Phe) + shikimate
        "trpa",
        "trpb",
        "trpc",
        "trpd",
        "trpe",
        "trpf",
        "trpg",
        "trps2",
        "aroa",
        "arob",
        "aroc",
        "arod",
        "aroe",
        "arof",
        "arog",
        "aroh",
        "arok",
        "aroq",
        "phea",
        "tyra",
        # histidine
        "hisa",
        "hisb",
        "hisc",
        "hisd",
        "hise",
        "hisf",
        "hisg",
        "hish",
        "hisi",
        "hisn",
        "hisz",
        # aspartate family (Lys/Thr/Met/DAP)
        "asd",
        "dapa",
        "dapb",
        "dapd",
        "dape",
        "dapf",
        "dapl",
        "lysa",
        "lysc",
        "hom",
        "homb",
        "thra",
        "thrb",
        "thrc",
        "meta",
        "metb",
        "metc",
        "mete",
        "metf",
        "meth",
        "meti2",
        "metx",
        "metz",
        "mmum",
        # glutamate/glutamine/proline/arginine
        "glna",
        "glnii",
        "gltb",
        "gltd",
        "gdha",
        "proa",
        "prob",
        "proc",
        "arga",
        "argb",
        "argc",
        "argd",
        "argf",
        "argg",
        "argh",
        "argi",
        "argj",
        # serine/glycine/cysteine
        "sera",
        "serb",
        "serc",
        "glya",
        "cysd",
        "cyse",
        "cysh",
        "cysi",
        "cysj",
        "cysk",
        "cysm",
        "cysn",
        # asparagine
        "asna",
        "asnb",
        "asno",
    }
)

#: Product-name rules, evaluated in this order **after** :data:`EXCLUSIONS`.  ``aars`` and
#: ``transamidation`` come first because both are tRNA-anchored and unambiguous; the
#: transamidation rules all require the literal ``tRNA`` because a bare ``amidotransferase``
#: match picks up the glutamine amidotransferase domains of purine, His and Trp biosynthesis
#: (measured: ``type 1 glutamine amidotransferase``, ``Anthranilate synthase, amidotransferase
#: component``, ``Imidazole glycerol phosphate synthase amidotransferase subunit``).
CLASS_RULES: tuple[tuple[str, str], ...] = (
    (CLASS_TRANSAMIDATION, r"Asp-tRNA\(Asn\)|Glu-tRNA\(Gln\)|GatCAB"),
    (CLASS_TRANSAMIDATION, rf"(?:{_AARS_STEM}|asparaginyl)[-/].*tRNA.*amidotransferase"),
    (
        CLASS_TRANSAMIDATION,
        r"tRNA[- ]dependent amidotransferase|tRNA\([A-Za-z]{3}\) amidotransferase",
    ),
    # ``aspartate--tRNA(Asn) ligase`` is the real product name of a non-discriminating AspRS;
    # the optional parenthesised acceptor is measured, not defensive.
    (CLASS_AARS, rf"(?:{_AA_NAME})--tRNA(?:\([A-Za-z]{{3}}\))? ligase"),
    (CLASS_AARS, rf"(?:{_AARS_STEM})-tRNA synthetase"),
    (
        CLASS_AARS,
        r"aminoacyl-tRNA synthetase|class (?:I|II) (?:aminoacyl-)?tRNA (?:ligase|synthetase)",
    ),
    (CLASS_AARS, r"tRNA ligase family protein"),
    # Transport: an amino-acid token AND a transport token, in either order.
    (
        CLASS_AA_TRANSPORT,
        rf"(?:{_AA_NAME}|betaine)[^;]{{0,60}}?"
        r"(?:ABC transporter|permease|symporter|antiporter|transporter|transport system|"
        r"carrier protein|substrate-binding protein|uptake|translocator|exporter)",
    ),
    (
        CLASS_AA_TRANSPORT,
        rf"(?:ABC transporter|permease|symporter|antiporter|transporter)"
        rf"[^;]{{0,60}}?(?:{_AA_NAME})",
    ),
    # Biosynthesis: pathway-diagnostic enzyme names only.  Nothing here is a generic fold.
    (
        CLASS_AA_BIOSYNTHESIS,
        r"threonine synthase|cysteine synthase|tryptophan synthase|anthranilate synthase|"
        r"anthranilate phosphoribosyltransferase|phosphoribosylanthranilate isomerase|"
        r"indole-3-glycerol[- ]phosphate synthase|chorismate synthase|chorismate mutase|"
        r"prephenate dehydratase|prephenate dehydrogenase|arogenate dehydrogenase|"
        r"shikimate kinase|shikimate dehydrogenase|3-deoxy-7-phosphoheptulonate synthase|"
        r"3-dehydroquinate|3-phosphoshikimate 1-carboxyvinyltransferase|"
        r"aspartate kinase|aspartokinase|aspartate-semialdehyde dehydrogenase|"
        r"homoserine dehydrogenase|"
        r"homoserine kinase|homoserine O-(?:succinyl|acetyl)transferase|"
        r"diaminopimelate (?:decarboxylase|epimerase|dehydrogenase)|"
        r"dihydrodipicolinate (?:synthase|reductase)|4-hydroxy-tetrahydrodipicolinate|"
        r"LL-diaminopimelate aminotransferase|"
        r"tetrahydrodipicolinate N-(?:succinyl|acetyl)transferase|"
        r"cystathionine (?:beta|gamma)-(?:synthase|lyase)|methionine synthase|"
        r"homocysteine S?-?methyltransferase|"
        r"O-acetylhomoserine|serine O-acetyltransferase|"
        r"phosphoglycerate dehydrogenase|"
        r"phosphoserine (?:phosphatase|aminotransferase|transaminase)|"
        r"3-phosphoserine/phosphohydroxythreonine transaminase|"
        r"serine hydroxymethyltransferase|glycine hydroxymethyltransferase|"
        r"glutamine synthetase|glutamate synthase|glutamate dehydrogenase|"
        r"glutamate 5-kinase|glutamate-5-semialdehyde dehydrogenase|"
        r"pyrroline-5-carboxylate reductase|acetylglutamate kinase|"
        r"N-acetyl-gamma-glutamyl-phosphate reductase|"
        r"acetylornithine (?:aminotransferase|deacetylase)|"
        r"ornithine carbamoyltransferase|argininosuccinate (?:synthase|lyase)|"
        r"glutamate N-acetyltransferase|ornithine acetyltransferase|N-acetylglutamate synthase|"
        r"amino-acid N-acetyltransferase|"
        r"ATP phosphoribosyltransferase|"
        r"histidinol[- ](?:phosphate )?(?:dehydrogenase|aminotransferase|transaminase|phosphatase)|"
        r"imidazole ?glycerol[- ]phosphate (?:synthase|dehydratase)|"
        r"phosphoribosyl-AMP cyclohydrolase|"
        r"phosphoribosyl-ATP (?:pyrophosphohydrolase|pyrophosphatase|diphosphatase)|"
        r"phosphoribosylformimino|imidazole-4-carboxamide isomerase|"
        r"acetolactate synthase|ketol-acid reductoisomerase|dihydroxy-acid dehydratase|"
        r"branched-chain[- ]amino[- ]acid (?:transaminase|aminotransferase)|"
        r"2-isopropylmalate synthase|3-isopropylmalate deh(?:ydrogenase|ydratase)|"
        r"threonine (?:ammonia-lyase|dehydratase)|"
        r"asparagine synth(?:ase|etase)|aspartate--ammonia ligase|glutamate--ammonia ligase|"
        r"amino[- ]acid biosynthes",
    ),
)

_CLASS_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(p, re.IGNORECASE)) for name, p in CLASS_RULES
)

#: The product strings NCBI uses when it has nothing to say.  D4 routes these to Pfam/KO.
_UNINFORMATIVE = re.compile(
    r"^(?:hypothetical protein|uncharacteri[sz]ed protein|"
    r"(?:DUF\d+ )?(?:domain-containing )?protein|protein of unknown function.*)$",
    re.IGNORECASE,
)

#: ⚠ **Identifier-shaped values are not gene identity, and treating them as such silently
#: disables D4's whole hypothetical/pseudogene route.**  ``gff3.gene_identity_text`` collects
#: ``Dbxref``, ``Name`` and friends as well as ``product``, and on NCBI CDS rows those are
#: accessions — ``Genbank:WP_012345678.1``, ``cds-OIP74746.1``.  ``Dbxref`` is present on
#: 29,852 of 30,273 measured CDS, so before this filter existed **every** ``hypothetical
#: protein`` still had a non-empty "informative" text, never resolved to
#: :data:`FN_UNJUDGEABLE`, and was scored a decided ``failed`` instead of being routed to the
#: Pfam/KO fallback — the exclusion diagnostic reported **zero** unjudgeable ORFs as a result.
#: GO terms are included here deliberately: ``GO:0004812`` *is* functional information, but
#: this step resolves no GO table, so calling it readable would be claiming a judgement that
#: was never made.
_IDENTIFIER_LIKE = re.compile(
    r"^(?:Genbank|RefSeq|Protein|UniProtKB(?:/\w+)?|NCBI_GP|InterPro|GeneID|EnsemblGenomes[-\w]*|"
    r"GO|SO|EC|KEGG|COG|Pfam|TIGRFAM|HAMAP)[:_]"
    r"|^(?:cds-|gene-|rna-|id-)"
    r"|^[A-Z]{1,3}[_-]?\d{5,}(?:\.\d+)?$",
    re.IGNORECASE,
)


def gene_symbols(cds: gff3.CdsFeature) -> frozenset[str]:
    """Lower-cased ``gene`` / ``Name`` symbols, for the exact-match symbol routes."""
    out: set[str] = set()
    for key in ("gene", "Name", "gene_synonym"):
        for value in cds.attributes.get(key, ()):
            token = value.strip().lower()
            if token and len(token) <= 8:
                out.add(token)
    return frozenset(out)


def classify_gene_identity(
    identity_text: Sequence[str], *, gene_symbols: Iterable[str] = ()
) -> str | None:
    """Map a CDS's gene-identity text onto one of D4's four classes.

    ``identity_text`` is :func:`tbox_finder.mining.gff3.gene_identity_text`'s output — every
    non-empty value over the consulted attribute keys, unescaped.

    **The curated product name outranks the gene symbol**, and that order was corrected by
    measurement rather than assumed: gene symbols are unambiguous *within* a genome but not
    *across* taxa, and ``hisJ`` is the clearest case in this corpus — the histidine ABC
    transporter's periplasmic binding protein in the enterics, but **histidinol phosphatase**
    (a histidine *biosynthesis* enzyme) in Firmicutes.  Symbol-first mislabelled the real
    record ``histidinol-phosphatase HisJ`` as transport.  Symbols therefore run only as the
    fallback for a CDS whose product name is uninformative — which on NCBI GFFs is rare
    (``product`` is present on every one of the 30,273 CDS measured) but will not be once the
    commissioned bakta/prokka arm starts emitting annotations.

    Returns the class name, :data:`FN_UNJUDGEABLE` when the text exists but says nothing a
    function can be read from, or ``None`` when the text is present and readable but names
    none of D4's classes (an honest **failure** of the criterion, not an absence of evidence).

    Exclusions are applied **before** every product-name include rule, so a string that
    matches both is excluded — the precedence is stated, not emergent from regex order.
    """
    texts = [t for t in identity_text if t and t.strip()]
    informative = [
        t
        for t in texts
        if not _UNINFORMATIVE.match(t.strip()) and not _IDENTIFIER_LIKE.match(t.strip())
    ]
    for text in informative:
        if any(pattern.search(text) for pattern, _why in _EXCLUSION_RES):
            continue
        for name, pattern in _CLASS_RES:
            if pattern.search(text):
                return name

    if informative:
        # A curated product name that names no D4 class is a *decided* non-match.  Consulting
        # the symbol here instead of stopping is what produced every remaining false positive
        # when it was measured: ``PTS galactitol transporter subunit IIB`` reached
        # ``transamidation`` through the ``gatB`` symbol collision, ``cytochrome C-552``
        # reached transport through ``cycA``, and ``homocitrate synthase`` reached aaRS
        # through ``lysS`` (which is homocitrate synthase in the α-aminoadipate lysine
        # pathway and lysyl-tRNA synthetase everywhere else).  24 of 4,991 matches over the
        # 48 candidate-carrying hosts came from this route and 3 of the 24 were wrong.
        return None

    symbols = {s.strip().lower() for s in gene_symbols if s and s.strip()}
    if symbols & AARS_GENE_SYMBOLS:
        return CLASS_AARS
    if symbols & TRANSAMIDATION_GENE_SYMBOLS:
        return CLASS_TRANSAMIDATION
    if symbols & TRANSPORT_GENE_SYMBOLS:
        return CLASS_AA_TRANSPORT
    if symbols & BIOSYNTHESIS_GENE_SYMBOLS:
        return CLASS_AA_BIOSYNTHESIS
    return FN_UNJUDGEABLE


def excluded_by(text: str) -> str | None:
    """The reason string of the first :data:`EXCLUSIONS` entry ``text`` matches, else ``None``.

    Exposed so the diagnostics can report *which* family suppressed a match rather than only
    that something did — an exclusion list nobody can audit is a vocabulary nobody can review.
    """
    for pattern, why in _EXCLUSION_RES:
        if pattern.search(text):
            return why
    return None


# ═════════════════════════════════════════════════════════════════════════════
# The ADR-0006-frozen predicate, verbatim
# ═════════════════════════════════════════════════════════════════════════════
def criterion_c(
    downstream_gene_fn: str | None,
    downstream_gene_distance_bp: int | None,
    strand_same: bool,
    *,
    window_bp: int = DEFAULT_WINDOW_BP,
) -> bool:
    """ADR-0006 D4's frozen predicate — signature and semantics exactly as pinned.

    ``True`` iff ``strand_same`` **and** ``distance <= window_bp`` **and**
    ``downstream_gene_fn`` is one of :data:`PASSING_CLASSES`.

    Class II is the ``distance ≈ 0`` case: a translational T-box abuts or overlaps the start
    codon, so a **negative** distance (the CDS start lies inside the element) is in-window,
    not out of it.  ``downstream_gene_distance_bp is None`` means no downstream CDS was found
    at all ⇒ ``False``.  :data:`FN_UNJUDGEABLE` is **not** a passing class — it is the input
    that makes :func:`synteny_status` report ``unavailable``, and it must never reach here as
    a pass.
    """
    if not strand_same:
        return False
    if downstream_gene_fn is None or downstream_gene_fn not in PASSING_CLASSES:
        return False
    if downstream_gene_distance_bp is None:
        return False
    return int(downstream_gene_distance_bp) <= int(window_bp)


# ═════════════════════════════════════════════════════════════════════════════
# Coordinates: candidate span (0-based half-open) → GFF3 (1-based inclusive)
# ═════════════════════════════════════════════════════════════════════════════
def locus_three_prime_1based(locus_start: int, locus_end: int, strand: str) -> int:
    """The element's 3′ end as a **1-based inclusive** contig coordinate.

    ``locus_start``/``locus_end`` are the manifest's 0-based half-open span.  On ``+`` the 3′
    end is the right edge, whose 1-based inclusive coordinate is ``locus_end`` unchanged (the
    half-open end and the inclusive end coincide).  On ``-`` it is the left edge, whose
    1-based coordinate is ``locus_start + 1``.  D4's *terminator-inclusive* boundary is the
    called locus edge — the Stage-1 locus constructor already spans the terminator, so nothing
    is re-derived here.
    """
    if locus_start < 0 or locus_end <= locus_start:
        raise SyntenyError(f"bad candidate span [{locus_start}, {locus_end})")
    if strand == gff3.STRAND_PLUS:
        return int(locus_end)
    if strand == gff3.STRAND_MINUS:
        return int(locus_start) + 1
    raise SyntenyError(f"unresolved strand {strand!r}; expected {sorted(gff3.STRANDS_RESOLVED)}")


def load_contig_ids(genome_fasta: str) -> list[str]:
    """The genome FASTA's record ids, in file order — index ``ci`` is the candidate's contig.

    Only the ``>`` headers are read (first whitespace-delimited token), so a 600-contig MAG
    costs a scan rather than a full sequence load.
    """
    ids: list[str] = []
    with open(genome_fasta, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.startswith(">"):
                continue
            token = line[1:].split()
            if not token:
                raise SyntenyError(f"{genome_fasta}:{line_no}: FASTA header has no record id")
            ids.append(token[0])
    if not ids:
        raise SyntenyError(f"{genome_fasta}: no FASTA records")
    return ids


def contig_seqid(contig_ids: Sequence[str], contig_index: int) -> str:
    """``records[ci]``'s id — the join key into the GFF's ``seqid`` column.

    The bridge is by identity because it cannot be by index: 23 of the 48 candidate-carrying
    annotated hosts order their GFF seqids differently from their FASTA records.
    """
    if not 0 <= contig_index < len(contig_ids):
        raise SyntenyError(f"contig index {contig_index} out of range ({len(contig_ids)} records)")
    return contig_ids[contig_index]


# ═════════════════════════════════════════════════════════════════════════════
# The downstream walk + D4's tandem / intervening-ORF carve-out
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DownstreamGene:
    """What the downstream walk found, and how it got there."""

    function_class: str | None
    #: Distance from the **element's** 3′ end — the reportable quantity, and the one D4's
    #: p95/p99 sensitivity check is about.
    distance_bp: int | None
    #: Distance from the **anchor the walk actually judged against** — identical to
    #: ``distance_bp`` when no carve-out fired, and measured from the last intervening ORF's
    #: 3′ end when one did.  ⚠ This is what :func:`criterion_c` must see: D4 says the tandem
    #: carve-out *"extends the window past a downstream leader or sub-threshold ORF"*, so a
    #: carve-out that re-anchors the search but not the decision fires only when it was not
    #: needed, and turns exactly the tandem loci it exists for into false FAILs.
    decision_distance_bp: int | None
    feature_id: str | None
    identity_text: tuple[str, ...]
    is_pseudo: bool
    n_intervening: int
    carve_out_applied: bool
    #: Pseudogenized / unjudgeable ORFs seen anywhere in the walk, **including the ones the
    #: tandem carve-out hopped past**.  Counting only the ORF the walk stopped on made D4's
    #: pseudogene diagnostic report 0 for the whole corpus — the carve-out was consuming
    #: exactly the population the diagnostic exists to size.
    n_pseudo_seen: int = 0
    n_unjudgeable_seen: int = 0
    note: str = ""


def downstream_cds_on_strand(
    cds_features: Iterable[gff3.CdsFeature],
    *,
    seqid: str,
    strand: str,
    three_prime: int,
    element_span_nt: int,
) -> list[tuple[int, gff3.CdsFeature]]:
    """Same-contig, same-strand CDS **downstream** of ``three_prime``, nearest first.

    "Downstream" follows the strand: increasing coordinates on ``+``, decreasing on ``-``.
    The returned distance is ``start - three_prime`` (``+``) or ``three_prime - start``
    (``-``) where ``start`` is :func:`tbox_finder.mining.gff3.cds_start_position` — the
    feature's ``end`` on the minus strand, which is the whole reason that helper exists.

    ⚠ **The negative side has to be bounded, and getting that wrong is not a small error.**
    D4's class-II case is a translational T-box that *abuts or overlaps the start codon*
    (``distance ≈ 0``), so a slightly negative distance must be admissible.  But leaving the
    negative side open admits a CDS whose start is hundreds of bp **behind** the element —
    i.e. the element is buried inside that gene, which is the opposite of the gene being
    downstream of the element.  Measured, that error is enormous rather than marginal: with
    no lower bound, **541 of 941 (57.5 %)** hard-negative candidates "passed" criterion (c),
    against a per-CDS background density of 4.1 % for D4's four classes — a bacterial genome
    is ~88 % coding, so almost every random window sits inside some gene and simply inherits
    that gene's identity.

    The bound pins no new number: the CDS start must lie at or after the **element's own 5′
    end**, i.e. ``distance >= -element_span_nt``.  A gene whose start codon falls inside the
    element is the class-II signature; a gene that started before the element did is a gene
    the element is inside of.
    """
    if element_span_nt < 0:
        raise SyntenyError(f"element_span_nt must be >= 0, got {element_span_nt}")
    out: list[tuple[int, gff3.CdsFeature]] = []
    for cds in cds_features:
        if cds.seqid != seqid or cds.strand != strand:
            continue
        start = gff3.cds_start_position(cds)
        distance = start - three_prime if strand == gff3.STRAND_PLUS else three_prime - start
        if distance < -element_span_nt:
            continue
        out.append((distance, cds))
    out.sort(key=lambda item: (item[0], item[1].feature_id))
    return out


def resolve_downstream_gene(
    cds_features: Iterable[gff3.CdsFeature],
    *,
    seqid: str,
    strand: str,
    three_prime: int,
    element_span_nt: int,
    window_bp: int = DEFAULT_WINDOW_BP,
    max_intervening_orfs: int,
    sub_threshold_orf_nt: int,
) -> DownstreamGene:
    """Walk downstream and resolve the inputs :func:`criterion_c` consumes.

    ``max_intervening_orfs`` and ``sub_threshold_orf_nt`` are **required and have no
    default** — D4 commissions a *"tandem / intervening-ORF carve-out"* and pins no number
    for either, so the round states them and they ride in the report.  A default here would
    silently decide which candidates are mined.

    The carve-out fires only on an ORF that cannot carry the criterion either way: one whose
    identity is :data:`FN_UNJUDGEABLE` (hypothetical / no text / pseudogenized) **or** that is
    shorter than ``sub_threshold_orf_nt``.  When it fires, the window is re-anchored at that
    ORF's own 3′ end and the walk continues — the tandem-locus reading of D4 (*"extends the
    window past a downstream leader or sub-threshold ORF"*), not a blanket window widening.
    A CDS that is judgeable and simply not in one of D4's classes **stops the walk**: that is
    a real downstream gene of another function, i.e. an honest criterion failure.
    """
    if max_intervening_orfs < 0:
        raise SyntenyError(f"max_intervening_orfs must be >= 0, got {max_intervening_orfs}")
    if sub_threshold_orf_nt < 0:
        raise SyntenyError(f"sub_threshold_orf_nt must be >= 0, got {sub_threshold_orf_nt}")

    ordered = downstream_cds_on_strand(
        cds_features,
        seqid=seqid,
        strand=strand,
        three_prime=three_prime,
        element_span_nt=element_span_nt,
    )
    anchor = int(three_prime)
    n_intervening = 0
    n_pseudo_seen = 0
    n_unjudgeable_seen = 0
    for distance_from_locus, cds in ordered:
        start = gff3.cds_start_position(cds)
        distance = start - anchor if strand == gff3.STRAND_PLUS else anchor - start
        if distance > window_bp:
            return DownstreamGene(
                function_class=None,
                distance_bp=None,
                decision_distance_bp=None,
                feature_id=None,
                identity_text=(),
                is_pseudo=False,
                n_intervening=n_intervening,
                carve_out_applied=n_intervening > 0,
                n_pseudo_seen=n_pseudo_seen,
                n_unjudgeable_seen=n_unjudgeable_seen,
                note=f"nearest same-strand CDS start is {distance} bp past the anchor",
            )
        pseudo = gff3.is_pseudo(cds)
        texts = gff3.gene_identity_text(cds)
        function = (
            FN_UNJUDGEABLE
            if pseudo
            else classify_gene_identity(texts, gene_symbols=gene_symbols(cds))
        )
        unjudgeable = function == FN_UNJUDGEABLE
        n_pseudo_seen += int(pseudo)
        n_unjudgeable_seen += int(unjudgeable)
        # ⚠ The carve-out fires only on an ORF that **cannot carry the criterion either
        # way**, and the length branch has to enforce that too: a short CDS that already
        # names one of D4's four classes decides the criterion, so hopping past it would
        # discard a real pass in favour of whatever sits further downstream.
        carryable = function in PASSING_CLASSES
        # ``coding_length_bp``, not ``length_bp``: the genomic span of a frameshifted CDS
        # includes the gap between its segments, so a short two-part ORF would read as long.
        sub_threshold = cds.coding_length_bp < sub_threshold_orf_nt and not carryable
        if (unjudgeable or sub_threshold) and n_intervening < max_intervening_orfs:
            n_intervening += 1
            # ⚠ Clamped to the element's own 3′ end.  The walk admits a CDS starting up to
            # ``element_span_nt`` BEHIND ``three_prime`` (D4's class-II overlap), and if such a
            # CDS triggers the carve-out its far edge can also lie behind the element — moving
            # the anchor upstream and judging the next CDS against a window WIDER than the
            # 500 bp D4 pins.  The carve-out may extend the window forward, never backward.
            moved = cds.end if strand == gff3.STRAND_PLUS else cds.start
            # Clamped against the CURRENT anchor, not the fixed element 3′ end: with more than
            # one hop allowed, a later ORF can also lie behind an anchor the walk already
            # advanced, and comparing to ``three_prime`` would not catch that.
            anchor = max(moved, anchor) if strand == gff3.STRAND_PLUS else min(moved, anchor)
            continue
        return DownstreamGene(
            function_class=function,
            distance_bp=distance_from_locus,
            decision_distance_bp=distance,
            feature_id=cds.feature_id,
            identity_text=texts,
            is_pseudo=pseudo,
            n_intervening=n_intervening,
            carve_out_applied=n_intervening > 0,
            n_pseudo_seen=n_pseudo_seen,
            n_unjudgeable_seen=n_unjudgeable_seen,
            note="pseudogene routed to the (absent) Pfam/KO fallback" if pseudo else "",
        )
    return DownstreamGene(
        function_class=None,
        distance_bp=None,
        decision_distance_bp=None,
        feature_id=None,
        identity_text=(),
        is_pseudo=False,
        n_intervening=n_intervening,
        carve_out_applied=n_intervening > 0,
        n_pseudo_seen=n_pseudo_seen,
        n_unjudgeable_seen=n_unjudgeable_seen,
        note="no same-strand CDS downstream of the element on this contig",
    )


def synteny_status(resolved: DownstreamGene, *, window_bp: int = DEFAULT_WINDOW_BP) -> str:
    """``passed`` / ``failed`` / ``unavailable`` for the ``downstream_aaRS_synteny`` disjunct.

    ``unavailable`` is reserved for *"the criterion could not be evaluated"*, which under
    ADR-0005 D14 **spares** the candidate.  Exactly one thing produces it here: the first
    judgeable-or-not ORF that D4 routes to Pfam/KO profiles while
    :data:`HMM_FALLBACK_AVAILABLE` is ``False``.  Everything else is a decided outcome —
    including *no downstream gene at all*, which is a real ``failed`` and not an absence of
    machinery.
    """
    if resolved.function_class == FN_UNJUDGEABLE:
        if HMM_FALLBACK_AVAILABLE:  # pragma: no cover - guarded by test_synteny.py
            raise SyntenyError(
                "HMM_FALLBACK_AVAILABLE is True but no Pfam/KO profile search is wired; "
                "declaring the fallback available without it is the fail-open direction"
            )
        return STATUS_UNAVAILABLE
    # ⚠ ``decision_distance_bp``, NOT ``distance_bp``.  With no carve-out the two are equal;
    # with one, the element-relative distance can exceed the window while the re-anchored one
    # does not — and D4's carve-out exists precisely to admit that case.
    passed = criterion_c(
        resolved.function_class, resolved.decision_distance_bp, True, window_bp=window_bp
    )
    return STATUS_PASSED if passed else STATUS_FAILED


def combine_strand_statuses(per_strand: Mapping[str, str], *, policy: str) -> str:
    """Fold the per-strand statuses into the one the disjunct carries.

    ``both`` is ADR-0005 D15's locked rule for orientation-ambiguous loci (*"carried through
    on both strands"*) resolved in the sparing direction: any ``passed`` ⇒ ``passed``; else
    any ``unavailable`` ⇒ ``unavailable``; else ``failed``.  That is the same three-valued
    fold the spare rule itself uses, so a candidate can never be mined because the orientation
    the manifest never recorded happened to be the unlucky one.
    """
    if policy not in STRAND_POLICIES:
        raise SyntenyError(f"unknown strand policy {policy!r}; expected {list(STRAND_POLICIES)}")
    missing = [s for s in (gff3.STRAND_PLUS, gff3.STRAND_MINUS) if s not in per_strand]
    if missing:
        raise SyntenyError(f"per-strand statuses missing {missing}; both strands are required")
    if policy == "plus":
        return per_strand[gff3.STRAND_PLUS]
    if policy == "minus":
        return per_strand[gff3.STRAND_MINUS]
    values = [per_strand[s] for s in (gff3.STRAND_PLUS, gff3.STRAND_MINUS)]
    if STATUS_PASSED in values:
        return STATUS_PASSED
    if STATUS_UNAVAILABLE in values:
        return STATUS_UNAVAILABLE
    return STATUS_FAILED
