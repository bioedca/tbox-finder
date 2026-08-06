"""P3-15′-c-i — the GFF3 reader, and the coordinate conventions D4 will be evaluated in.

Two tiers, both required:

* **Synthetic** lines pin each rule in isolation, so a failure names the rule.
* **A real committed NCBI GFF** (``GCA_002790315.1``, the smallest annotated host in the
  corpus, byte-identical to what NCBI serves — its md5 is the one in
  ``reports/p3/annotation_supply.json``) pins the whole parse against a golden hash. Synthetic
  lines only ever contain the shapes their author thought of; the real file is where the
  escaped comma, the 38 pseudogenes and the 41 contigs actually came from.

No network, no third-party imports: this runs in CI's bare tier, which installs no biopython.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tbox_finder.mining import gff3

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "annotation" / "GCA_002790315.1.gff.gz"

#: sha256 of the canonical JSON of every parsed CDS in FIXTURE (see ``_golden_digest``).
#: Regenerate ONLY when the fixture or the parse contract deliberately changes — a diff here
#: means coordinates, strands, grouping or attribute decoding moved.
GOLDEN_DIGEST = "8e5df62b32f892c62f94c6ee75dc29aea57407f136de18bc3e9287da20bb4d48"

CDS_PLUS = "ctg1\tGenbank\tCDS\t100\t400\t.\t+\t0\tID=cds-A;product=alanine--tRNA ligase"
CDS_MINUS = "ctg1\tGenbank\tCDS\t500\t900\t.\t-\t0\tID=cds-B;product=hypothetical protein"


def _feature(line: str) -> gff3.GffFeature:
    return gff3.parse_gff3_line(line, line_no=1)


def _cds(*lines: str) -> list[gff3.CdsFeature]:
    return gff3.parse_gff3_cds(lines)


def _golden_digest(cds: list[gff3.CdsFeature]) -> str:
    """A stable digest of the whole parse — coordinates, strand, segments, attributes."""
    payload = [
        {
            "seqid": c.seqid,
            "feature_id": c.feature_id,
            "start": c.start,
            "end": c.end,
            "strand": c.strand,
            "segments": [list(s) for s in c.segments],
            "attributes": {k: list(v) for k, v in sorted(c.attributes.items())},
        }
        for c in cds
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Coordinates — the convention this module exists to pin
# --------------------------------------------------------------------------- #


def test_coordinates_are_carried_through_1_based_inclusive():
    f = _feature(CDS_PLUS)
    assert (f.start, f.end) == (100, 400)


def test_length_bp_is_inclusive_of_both_endpoints():
    (c,) = _cds("ctg1\tx\tCDS\t7\t7\t.\t+\t0\tID=one")
    assert c.length_bp == 1
    (c2,) = _cds("ctg1\tx\tCDS\t100\t400\t.\t+\t0\tID=two")
    assert c2.length_bp == 301


def test_cds_start_position_is_the_low_coordinate_on_plus_strand():
    (c,) = _cds(CDS_PLUS)
    assert gff3.cds_start_position(c) == 100


def test_cds_start_position_is_the_HIGH_coordinate_on_minus_strand():
    """D4 measures to the CDS *start codon*; on ``-`` that is the feature's ``end``.

    The whole-gene off-by-one this guards is invisible in a count-based assertion, so it is
    asserted as an identity against the other endpoint.
    """
    (c,) = _cds(CDS_MINUS)
    assert gff3.cds_start_position(c) == 900
    assert gff3.cds_start_position(c) != c.start


@pytest.mark.parametrize("strand", [".", "?"])
def test_cds_start_position_refuses_an_unresolved_strand(strand):
    (c,) = _cds(f"ctg1\tx\tCDS\t10\t20\t.\t{strand}\t0\tID=u")
    with pytest.raises(gff3.Gff3Error, match="unresolved strand"):
        gff3.cds_start_position(c)


def test_unresolved_strand_is_parsed_not_dropped():
    """It must reach the caller as a feature — refusing at *parse* would hide it entirely."""
    (c,) = _cds("ctg1\tx\tCDS\t10\t20\t.\t.\t0\tID=u")
    assert c.strand == "."


# --------------------------------------------------------------------------- #
# Line-level refusals — malformed input never degrades into a plausible feature
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "line",
    [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0",  # 8 columns
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=a\textra",  # 10 columns
        "ctg1 x CDS 10 20 . + 0 ID=a",  # spaces, not tabs
    ],
)
def test_refuses_a_wrong_column_count(line):
    with pytest.raises(gff3.Gff3Error, match="tab-separated columns"):
        _feature(line)


@pytest.mark.parametrize("raw", ["abc", "1.5", "-3", "", " ", "1e3"])
def test_refuses_a_non_integer_start(raw):
    with pytest.raises(gff3.Gff3Error, match="start"):
        _feature(f"ctg1\tx\tCDS\t{raw}\t20\t.\t+\t0\tID=a")


def test_refuses_a_zero_coordinate_because_gff3_is_1_based():
    with pytest.raises(gff3.Gff3Error, match="1-based"):
        _feature("ctg1\tx\tCDS\t0\t20\t.\t+\t0\tID=a")


def test_refuses_start_greater_than_end():
    with pytest.raises(gff3.Gff3Error, match="start 40 > end 20"):
        _feature("ctg1\tx\tCDS\t40\t20\t.\t+\t0\tID=a")


def test_refuses_an_unknown_strand_token():
    with pytest.raises(gff3.Gff3Error, match="unknown strand"):
        _feature("ctg1\tx\tCDS\t10\t20\t.\t*\t0\tID=a")


@pytest.mark.parametrize("phase", ["3", "-1", "x"])
def test_refuses_an_out_of_range_phase(phase):
    with pytest.raises(gff3.Gff3Error, match="phase"):
        _feature(f"ctg1\tx\tCDS\t10\t20\t.\t+\t{phase}\tID=a")


def test_a_non_cds_feature_may_omit_its_phase():
    f = _feature("ctg1\tx\tgene\t10\t20\t.\t+\t.\tID=a")
    assert f.phase is None and f.score is None


def test_a_CDS_MUST_carry_a_phase():
    """GFF3 requires it for CDS specifically; 0 of the corpus's 897,369 CDS lines omit it."""
    with pytest.raises(gff3.Gff3Error, match="must carry a phase"):
        _feature("ctg1\tx\tCDS\t10\t20\t.\t+\t.\tID=a")


def test_refuses_a_score_that_is_neither_dot_nor_a_float():
    with pytest.raises(gff3.Gff3Error, match="score"):
        _feature("ctg1\tx\tCDS\t10\t20\tgood\t+\t0\tID=a")


def test_parses_a_real_float_score():
    assert _feature("ctg1\tx\tCDS\t10\t20\t12.5\t+\t0\tID=a").score == 12.5


# --------------------------------------------------------------------------- #
# Attributes — escaping, multi-value ordering, repeats
# --------------------------------------------------------------------------- #


def test_parses_key_value_pairs():
    attrs = gff3.parse_attributes("ID=cds-A;product=alanine--tRNA ligase;locus_tag=X_001")
    assert gff3.attribute_first(attrs, "product") == "alanine--tRNA ligase"
    assert gff3.attribute_first(attrs, "locus_tag") == "X_001"


def test_unescapes_percent_encoding():
    attrs = gff3.parse_attributes("country=USA: Green River%2C Utah")
    assert gff3.attribute_first(attrs, "country") == "USA: Green River, Utah"


def test_comma_split_happens_BEFORE_unescaping():
    """An escaped comma must stay inside one value, and a bare comma must separate two.

    Unescaping first would shatter ``glycine%2C serine transporter`` into two bogus products —
    and the resulting gene-identity text would then contain a fragment that matches nothing.
    """
    attrs = gff3.parse_attributes("product=glycine%2C serine transporter;Dbxref=a:1,b:2")
    assert attrs["product"] == ("glycine, serine transporter",)
    assert attrs["Dbxref"] == ("a:1", "b:2")


def test_unescape_does_not_treat_plus_as_a_space():
    """``unquote_plus`` would corrupt ``NAD(P)+ transhydrogenase`` into ``NAD(P)  …``."""
    assert gff3.unescape("NAD(P)+ transhydrogenase") == "NAD(P)+ transhydrogenase"
    attrs = gff3.parse_attributes("product=NAD(P)+ transhydrogenase")
    assert gff3.attribute_first(attrs, "product") == "NAD(P)+ transhydrogenase"


def test_a_repeated_key_merges_rather_than_overwriting():
    attrs = gff3.parse_attributes("Note=first;Note=second")
    assert attrs["Note"] == ("first", "second")


@pytest.mark.parametrize("col9", ["", ".", "   "])
def test_an_empty_attribute_column_is_not_an_error(col9):
    assert gff3.parse_attributes(col9) == {}


def test_refuses_an_attribute_fragment_with_no_equals():
    with pytest.raises(gff3.Gff3Error, match="no '='"):
        gff3.parse_attributes("ID=a;garbage")


def test_refuses_an_empty_attribute_key():
    with pytest.raises(gff3.Gff3Error, match="empty key"):
        gff3.parse_attributes("=value")


def test_attribute_first_collapses_absent_and_empty():
    attrs = gff3.parse_attributes("product=;gene=hemB")
    assert gff3.attribute_first(attrs, "product") is None
    assert gff3.attribute_first(attrs, "missing") is None
    assert gff3.attribute_first(attrs, "gene") == "hemB"


# --------------------------------------------------------------------------- #
# ##FASTA, comments, blank lines
# --------------------------------------------------------------------------- #


def test_fasta_directive_terminates_the_feature_section():
    """bakta/prokka append their contigs; a 9-column split would mangle those into features."""
    lines = [
        "##gff-version 3",
        CDS_PLUS,
        "##FASTA",
        ">ctg1",
        "ACGT\tACGT\tCDS\t1\t9\t.\t+\t0\tID=not-a-feature",
    ]
    got = gff3.parse_gff3_cds(lines)
    assert [c.feature_id for c in got] == ["cds-A"]


def test_fasta_directive_is_matched_case_insensitively_and_with_surrounding_space():
    lines = ["##gff-version 3", CDS_PLUS, "  ##fasta  ", ">c", "ACGT"]
    assert len(gff3.parse_gff3_cds(lines)) == 1


def test_comments_directives_and_blank_lines_are_skipped():
    lines = [
        "##gff-version 3",
        "#!processor NCBI annotwriter",
        "##sequence-region ctg1 1 5000",
        "",
        "   ",
        CDS_PLUS,
    ]
    assert len(gff3.parse_gff3_cds(lines)) == 1


def test_non_cds_features_are_filtered_out_of_the_cds_view():
    lines = [
        "ctg1\tx\tgene\t100\t400\t.\t+\t.\tID=gene-A",
        "ctg1\tx\ttRNA\t600\t700\t.\t+\t.\tID=rna-A",
        CDS_PLUS,
    ]
    assert [c.feature_id for c in gff3.parse_gff3_cds(lines)] == ["cds-A"]


def test_iter_gff3_features_with_no_type_filter_yields_everything():
    lines = ["ctg1\tx\tgene\t1\t9\t.\t+\t.\tID=g", CDS_PLUS]
    assert [f.type for f in gff3.iter_gff3_features(lines)] == ["gene", "CDS"]


def test_crlf_line_endings_parse():
    (c,) = gff3.parse_gff3_cds([CDS_PLUS + "\r\n"])
    assert c.end == 400


# --------------------------------------------------------------------------- #
# Grouping — the multi-row CDS that is real in this corpus
# --------------------------------------------------------------------------- #


def test_rows_sharing_an_id_merge_into_one_cds_spanning_both():
    """``GCF_002895085.1`` really carries ``cds-WP_011096872.1`` on two rows (frameshift).

    A per-row reading would report two genes and put a spurious "CDS start" in the middle.
    """
    lines = [
        "ctg1\tx\tCDS\t173622\t173690\t.\t+\t0\tID=cds-W;product=p",
        "ctg1\tx\tCDS\t173692\t174729\t.\t+\t0\tID=cds-W;product=p",
    ]
    (c,) = gff3.parse_gff3_cds(lines)
    assert c.segments == ((173622, 173690), (173692, 174729))
    assert (c.start, c.end) == (173622, 174729)
    assert gff3.cds_start_position(c) == 173622


def test_grouping_preserves_first_appearance_order():
    lines = [
        "ctg1\tx\tCDS\t500\t600\t.\t+\t0\tID=cds-B",
        "ctg1\tx\tCDS\t100\t200\t.\t+\t0\tID=cds-A",
        "ctg1\tx\tCDS\t610\t700\t.\t+\t0\tID=cds-B",
    ]
    assert [c.feature_id for c in gff3.parse_gff3_cds(lines)] == ["cds-B", "cds-A"]


def test_refuses_one_cds_id_spanning_two_contigs():
    lines = [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=cds-A",
        "ctg2\tx\tCDS\t30\t40\t.\t+\t0\tID=cds-A",
    ]
    with pytest.raises(gff3.Gff3Error, match="spans two contigs"):
        gff3.parse_gff3_cds(lines)


def test_refuses_one_cds_id_carrying_two_strands():
    lines = [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=cds-A",
        "ctg1\tx\tCDS\t30\t40\t.\t-\t0\tID=cds-A",
    ]
    with pytest.raises(gff3.Gff3Error, match="two strands"):
        gff3.parse_gff3_cds(lines)


def test_id_less_rows_do_not_merge_into_one_giant_pseudo_gene():
    """The silent failure here is a *merge*, not a crash — so it is asserted as a count."""
    lines = [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tproduct=a",
        "ctg1\tx\tCDS\t3000\t4000\t.\t+\t0\tproduct=b",
    ]
    got = gff3.parse_gff3_cds(lines)
    assert len(got) == 2
    assert got[0].end == 20 and got[1].start == 3000


def test_two_ID_LESS_rows_at_IDENTICAL_coordinates_do_not_collapse():
    """The case the *different*-coordinates test above cannot reach.

    The coordinate-derived fallback is a display name, not an identity: two ID-less rows
    sharing contig/coordinates/strand/type produce the same fallback key and merge, losing the
    second row's attributes entirely. A merge, not a crash — so it is asserted as a count AND
    as the surviving attributes ([[duplicate-key-merges-instead-of-colliding]]).
    """
    lines = [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tproduct=first;locus_tag=A",
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tproduct=second;locus_tag=B",
    ]
    got = gff3.parse_gff3_cds(lines)
    assert len(got) == 2
    assert [gff3.attribute_first(c.attributes, "product") for c in got] == ["first", "second"]
    assert got[0].feature_id != got[1].feature_id


def test_declared_ids_still_group_across_rows_after_the_id_less_fix():
    """The fix must not turn every row into its own feature — declared IDs still merge."""
    lines = [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=cds-A",
        "ctg1\tx\tCDS\t30\t40\t.\t+\t0\tID=cds-A",
    ]
    (c,) = gff3.parse_gff3_cds(lines)
    assert c.segments == ((10, 20), (30, 40)) and c.feature_id == "cds-A"


def test_an_empty_ID_attribute_is_treated_as_undeclared():
    """``ID=`` is no identity at all; grouping on ``""`` would merge every such row."""
    lines = [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=;product=first",
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=;product=second",
    ]
    assert len(gff3.parse_gff3_cds(lines)) == 2


def test_parse_gff3_document_requires_the_version_directive():
    """Any nine-column TSV would otherwise reach the annotation census as GFF3."""
    with pytest.raises(gff3.Gff3Error, match="not '##gff-version'"):
        gff3.parse_gff3_document(CDS_PLUS + "\n")


def test_parse_gff3_document_refuses_a_directive_with_no_separator():
    """``##gff-version3`` satisfies ``startswith`` and is not the required directive."""
    with pytest.raises(gff3.Gff3Error, match="not '##gff-version'"):
        gff3.parse_gff3_document("##gff-version3\n" + CDS_PLUS + "\n")


def test_two_distinct_invalid_utf8_ids_do_not_decode_to_the_same_string():
    """``errors="replace"`` maps both to U+FFFD, and ``group_cds`` then merges them."""
    for escape in ("%FF", "%FE"):
        with pytest.raises(gff3.Gff3Error, match="not valid UTF-8"):
            gff3.unescape(escape)


def test_a_stray_percent_is_refused_because_gff3_requires_it_escaped():
    with pytest.raises(gff3.Gff3Error, match="malformed percent-escape"):
        gff3.unescape("50% identity")
    assert gff3.unescape("50%25 identity") == "50% identity"


def test_valid_multibyte_percent_escapes_still_decode():
    assert gff3.unescape("%CE%B2-lactamase") == "β-lactamase"


def test_decode_gff3_bytes_handles_both_plain_and_gzipped_snapshots():
    text = "##gff-version 3\n" + CDS_PLUS + "\n"
    assert gff3.decode_gff3_bytes(text.encode("utf-8")) == text
    assert gff3.decode_gff3_bytes(gzip.compress(text.encode("utf-8"))) == text


def test_parse_gff3_document_refuses_a_non_gff3_version():
    with pytest.raises(gff3.Gff3Error, match="unsupported GFF version"):
        gff3.parse_gff3_document("##gff-version 2\n" + CDS_PLUS + "\n")


def test_parse_gff3_document_refuses_an_empty_document():
    with pytest.raises(gff3.Gff3Error, match="empty document"):
        gff3.parse_gff3_document("\n  \n")


def test_parse_gff3_document_accepts_a_declared_document_and_a_minor_version():
    assert len(gff3.parse_gff3_document("##gff-version 3\n" + CDS_PLUS + "\n")) == 1
    assert len(gff3.parse_gff3_document("##gff-version 3.1.26\n" + CDS_PLUS + "\n")) == 1


def test_parse_gff3_document_refuses_a_directive_that_is_not_first():
    with pytest.raises(gff3.Gff3Error, match="not '##gff-version'"):
        gff3.parse_gff3_document(CDS_PLUS + "\n##gff-version 3\n")


def test_the_real_fixture_is_a_declared_gff3_document():
    assert gff3.require_gff3_version(gff3.read_gff3_text(FIXTURE).splitlines()) == "3"
    assert len(gff3.parse_gff3_document(gff3.read_gff3_text(FIXTURE))) == 455


def test_attributes_from_EVERY_segment_survive_the_merge():
    """A flag written only on the second row of a frameshifted CDS must not be discarded.

    First-wins merging would silently undo ``is_pseudo``'s read-every-value contract one
    layer up: the CDS would carry no ``pseudo`` at all and be read as a normal gene.
    """
    lines = [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=cds-A;product=first",
        "ctg1\tx\tCDS\t30\t40\t.\t+\t0\tID=cds-A;product=second;pseudo=true",
    ]
    (c,) = gff3.parse_gff3_cds(lines)
    assert c.attributes["product"] == ("first", "second")
    assert gff3.is_pseudo(c) is True
    assert gff3.attribute_first(c.attributes, "product") == "first"


def test_merged_attributes_keep_the_first_rows_value():
    lines = [
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=cds-A;product=first",
        "ctg1\tx\tCDS\t30\t40\t.\t+\t0\tID=cds-A;product=second;Note=only-on-second",
    ]
    (c,) = gff3.parse_gff3_cds(lines)
    assert gff3.attribute_first(c.attributes, "product") == "first"
    assert gff3.attribute_first(c.attributes, "Note") == "only-on-second"


# --------------------------------------------------------------------------- #
# Pseudogenes + gene-identity text (D4's two routes)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("col9", "expected"),
    [
        ("ID=a;pseudo=true", True),
        ("ID=a;pseudo=TRUE", True),
        ("ID=a;pseudo=", True),  # written but empty — conservative reading (see is_pseudo)
        ("ID=a;pseudo=false", False),
        ("ID=a;pseudo=no", False),
        ("ID=a", False),
        # Repeated keys merge (parse_attributes preserves both), so reading values[0] alone
        # calls these normal genes — and undercounts n_cds_pseudo in the offline census.
        ("ID=a;pseudo=false;pseudo=true", True),
        ("ID=a;pseudo=;pseudo=false", True),
        ("ID=a;pseudo=false;pseudo=no", False),
    ],
)
def test_is_pseudo_reads_the_flag_not_its_presence_alone(col9, expected):
    (c,) = _cds(f"ctg1\tx\tCDS\t10\t20\t.\t+\t0\t{col9}")
    assert gff3.is_pseudo(c) is expected


def test_a_bare_pseudo_tag_with_no_equals_is_refused_not_read_as_set():
    """GFF3 column 9 is ``tag=value``; a bare tag is malformed input, not a boolean flag."""
    with pytest.raises(gff3.Gff3Error, match="no '='"):
        _cds("ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=a;pseudo")


def test_gene_identity_text_collects_in_declared_key_order():
    (c,) = _cds(
        "ctg1\tx\tCDS\t10\t20\t.\t+\t0\t"
        "ID=a;Note=n;product=alanine--tRNA ligase;gene=alaS;Dbxref=GO:0004813"
    )
    got = gff3.gene_identity_text(c)
    assert got[0] == "alanine--tRNA ligase"
    assert got.index("alaS") < got.index("n")
    assert "GO:0004813" in got


def test_gene_identity_text_skips_empty_values_and_is_empty_when_nothing_describes_the_cds():
    (c,) = _cds("ctg1\tx\tCDS\t10\t20\t.\t+\t0\tID=a;product=;locus_tag=X")
    assert gff3.gene_identity_text(c) == ()


# --------------------------------------------------------------------------- #
# read_gff3_text
# --------------------------------------------------------------------------- #


def test_gzip_is_detected_by_magic_bytes_not_by_suffix(tmp_path):
    plain = tmp_path / "misnamed.gff.gz"
    plain.write_text(CDS_PLUS + "\n", encoding="utf-8")
    assert gff3.read_gff3_text(plain).startswith("ctg1")

    zipped = tmp_path / "misnamed.gff"
    zipped.write_bytes(gzip.compress((CDS_PLUS + "\n").encode("utf-8")))
    assert gff3.read_gff3_text(zipped).startswith("ctg1")


def test_read_gff3_text_decodes_utf8_product_names(tmp_path):
    path = tmp_path / "u.gff"
    path.write_text("ctg1\tx\tCDS\t1\t9\t.\t+\t0\tID=a;product=β-lactamase\n", encoding="utf-8")
    (c,) = gff3.parse_gff3_cds(gff3.read_gff3_text(path).splitlines())
    assert gff3.attribute_first(c.attributes, "product") == "β-lactamase"


# --------------------------------------------------------------------------- #
# The real committed NCBI GFF — golden + invariants
# --------------------------------------------------------------------------- #


def test_fixture_is_the_real_ncbi_bytes():
    """Its md5 is the value ``reports/p3/annotation_supply.json`` recorded from NCBI.

    If this fails the fixture has been re-generated or normalized, and every assertion below
    it is describing a different file than the corpus does.
    """
    import hashlib as _h

    digest = _h.md5(FIXTURE.read_bytes(), usedforsecurity=False).hexdigest()
    assert digest == "3afa0aff910cfd08f9f0163981656308"


def test_real_gff_parses_to_the_golden_digest():
    cds = gff3.parse_gff3_cds(gff3.read_gff3_text(FIXTURE).splitlines())
    assert _golden_digest(cds) == GOLDEN_DIGEST


def test_real_gff_census_matches_the_measured_shape():
    cds = gff3.parse_gff3_cds(gff3.read_gff3_text(FIXTURE).splitlines())
    assert len(cds) == 455
    assert sum(1 for c in cds if gff3.is_pseudo(c)) == 38
    assert len({c.seqid for c in cds}) == 41
    assert all(c.strand in gff3.STRANDS_RESOLVED for c in cds)


def test_real_gff_carries_an_escaped_attribute_value():
    """The ``%2C`` in this file is why the split order in ``parse_attributes`` is load-bearing."""
    regions = list(
        gff3.iter_gff3_features(gff3.read_gff3_text(FIXTURE).splitlines(), types={"region"})
    )
    countries = [gff3.attribute_first(f.attributes, "country") for f in regions]
    assert any(c and ", " in c for c in countries)


def test_real_gff_minus_strand_cds_start_is_its_high_coordinate():
    cds = gff3.parse_gff3_cds(gff3.read_gff3_text(FIXTURE).splitlines())
    minus = [c for c in cds if c.strand == gff3.STRAND_MINUS]
    assert minus, "fixture has no minus-strand CDS — the assertion below would be vacuous"
    assert all(gff3.cds_start_position(c) == c.end for c in minus)


def test_real_gff_every_cds_carries_a_product():
    cds = gff3.parse_gff3_cds(gff3.read_gff3_text(FIXTURE).splitlines())
    assert all(gff3.attribute_first(c.attributes, "product") for c in cds)
