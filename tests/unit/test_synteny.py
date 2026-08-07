"""ADR-0006 D4's criterion (c): the predicate, the vocabulary and the downstream walk.

Every test here is written so that *breaking the thing it names* turns it red — the
sabotage campaign for this step checks each one individually, because a suite that only
fails as a block proves some test bit, not which.
"""

from __future__ import annotations

import inspect

import pytest

from tbox_finder.mining import gff3, synteny
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED, STATUS_UNAVAILABLE


def cds(
    *,
    start: int,
    end: int,
    strand: str = "+",
    seqid: str = "c1",
    product: str | None = None,
    gene: str | None = None,
    pseudo: bool = False,
    feature_id: str | None = None,
    dbxref: str | None = "Genbank:WP_000000001.1",
) -> gff3.CdsFeature:
    attrs: dict[str, tuple[str, ...]] = {}
    if product is not None:
        attrs["product"] = (product,)
    if gene is not None:
        attrs["gene"] = (gene,)
    if pseudo:
        attrs["pseudo"] = ("true",)
    if dbxref is not None:
        attrs["Dbxref"] = (dbxref,)
    return gff3.CdsFeature(
        seqid=seqid,
        feature_id=feature_id or f"{seqid}:{start}-{end}:{strand}",
        start=start,
        end=end,
        strand=strand,
        segments=((start, end),),
        attributes=attrs,
    )


# ═════════════════════════════════════════════════════════════════════════════
# criterion_c — the ADR-frozen predicate
# ═════════════════════════════════════════════════════════════════════════════
class TestCriterionC:
    def test_signature_matches_the_adr_0006_pin(self) -> None:
        """ADR-0006 freezes the signature; a re-ordered or defaulted parameter is a departure."""
        sig = inspect.signature(synteny.criterion_c)
        names = list(sig.parameters)
        assert names == [
            "downstream_gene_fn",
            "downstream_gene_distance_bp",
            "strand_same",
            "window_bp",
        ]
        for positional in names[:3]:
            assert sig.parameters[positional].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert sig.parameters[positional].default is inspect.Parameter.empty
        window = sig.parameters["window_bp"]
        assert window.kind is inspect.Parameter.KEYWORD_ONLY
        assert window.default == 500, "D4's blinded-frozen window is 500 bp"

    @pytest.mark.parametrize("function_class", sorted(synteny.PASSING_CLASSES))
    def test_passes_on_each_of_d4s_four_classes(self, function_class: str) -> None:
        assert synteny.criterion_c(function_class, 100, True) is True

    def test_refuses_the_opposite_strand(self) -> None:
        assert synteny.criterion_c(synteny.CLASS_AARS, 10, False) is False

    def test_refuses_beyond_the_window_and_accepts_exactly_at_it(self) -> None:
        assert synteny.criterion_c(synteny.CLASS_AARS, 501, True) is False
        assert synteny.criterion_c(synteny.CLASS_AARS, 500, True) is True

    def test_class_ii_overlap_is_in_window_not_out_of_it(self) -> None:
        """A translational T-box abuts/overlaps the start codon — distance ≈ 0, D4."""
        assert synteny.criterion_c(synteny.CLASS_AARS, 0, True) is True
        assert synteny.criterion_c(synteny.CLASS_AARS, -30, True) is True

    def test_no_downstream_gene_is_a_failure_not_a_pass(self) -> None:
        assert synteny.criterion_c(None, None, True) is False
        assert synteny.criterion_c(synteny.CLASS_AARS, None, True) is False

    def test_unjudgeable_is_never_a_pass(self) -> None:
        """It is the input that makes the disjunct ``unavailable`` — it must not pass here."""
        assert synteny.criterion_c(synteny.FN_UNJUDGEABLE, 1, True) is False

    def test_an_unrelated_function_fails(self) -> None:
        assert synteny.criterion_c("dna_polymerase", 1, True) is False


# ═════════════════════════════════════════════════════════════════════════════
# The gene-identity vocabulary
# ═════════════════════════════════════════════════════════════════════════════
class TestClassifyGeneIdentity:
    @pytest.mark.parametrize(
        "product",
        [
            "alanine--tRNA ligase",
            "phenylalanine--tRNA ligase subunit beta",
            "aspartate--tRNA(Asn) ligase",
            "Seryl-tRNA synthetase",
            "class I tRNA ligase family protein",
        ],
    )
    def test_both_aars_naming_conventions(self, product: str) -> None:
        assert synteny.classify_gene_identity((product,)) == synteny.CLASS_AARS

    @pytest.mark.parametrize(
        "product",
        [
            "Asp-tRNA(Asn)/Glu-tRNA(Gln) amidotransferase subunit GatA",
            "aspartyl/glutamyl-tRNA amidotransferase subunit B",
            "Aspartyl-tRNA(Asn) amidotransferase subunit C",
        ],
    )
    def test_trna_dependent_transamidation(self, product: str) -> None:
        assert synteny.classify_gene_identity((product,)) == synteny.CLASS_TRANSAMIDATION

    @pytest.mark.parametrize(
        "product",
        [
            "type 1 glutamine amidotransferase",
            "glutamine amidotransferase family protein",
            "Phosphoribosylformylglycinamidine synthase, glutamine amidotransferase subunit",
        ],
    )
    def test_a_bare_glutamine_amidotransferase_is_NOT_transamidation(self, product: str) -> None:
        """These are purine/His/Trp glutamine amidotransferase domains, all real in this corpus.

        D4's class is the **tRNA-dependent** transamidation pathway; a bare
        ``amidotransferase`` rule would sweep these in and inflate the criterion.
        """
        assert synteny.classify_gene_identity((product,)) != synteny.CLASS_TRANSAMIDATION

    @pytest.mark.parametrize(
        "product",
        [
            "HAMP domain-containing histidine kinase",
            "serine/threonine protein kinase",
            "methylated-DNA--[protein]-cysteine S-methyltransferase",
            "peptide-methionine (S)-S-oxide reductase MsrA",
            "protein-glutamate O-methyltransferase CheR",
            "polysaccharide biosynthesis tyrosine autokinase",
            "ribosomal-protein-alanine N-acetyltransferase",
            "glutamine-hydrolyzing GMP synthase",
            "alanine racemase",
        ],
    )
    def test_the_measured_false_positive_families_are_excluded(self, product: str) -> None:
        """An "amino-acid word + enzyme word" rule admits every one of these.  None is a D4
        class."""
        assert synteny.classify_gene_identity((product,)) not in synteny.PASSING_CLASSES
        assert synteny.excluded_by(product) is not None

    def test_exclusion_outranks_a_matching_include_rule(self) -> None:
        """Precedence is stated, not emergent: a string matching both is excluded."""
        both = "histidine kinase / threonine synthase fusion protein"
        assert synteny.excluded_by(both) is not None
        assert synteny.classify_gene_identity((both,)) not in synteny.PASSING_CLASSES

    @pytest.mark.parametrize(
        ("product", "expected"),
        [
            ("branched-chain amino acid ABC transporter permease", synteny.CLASS_AA_TRANSPORT),
            ("sodium:alanine symporter family protein", synteny.CLASS_AA_TRANSPORT),
            ("threonine synthase", synteny.CLASS_AA_BIOSYNTHESIS),
            ("imidazoleglycerol-phosphate dehydratase HisB", synteny.CLASS_AA_BIOSYNTHESIS),
            ("argininosuccinate lyase", synteny.CLASS_AA_BIOSYNTHESIS),
        ],
    )
    def test_transport_and_biosynthesis(self, product: str, expected: str) -> None:
        assert synteny.classify_gene_identity((product,)) == expected

    def test_an_identifier_is_not_gene_identity(self) -> None:
        """⚠ The regression that made D4's whole Pfam/KO route dead.

        ``gene_identity_text`` collects ``Dbxref``/``Name`` as well as ``product``, and on NCBI
        CDS those are accessions.  Counting them as readable identity meant a ``hypothetical
        protein`` was scored a decided ``failed`` instead of being routed to the fallback, and
        the exclusion diagnostic reported **zero** unjudgeable ORFs across the whole corpus.
        """
        texts = ("hypothetical protein", "Genbank:WP_012345678.1", "WP_012345678.1")
        assert synteny.classify_gene_identity(texts) == synteny.FN_UNJUDGEABLE

    def test_positive_control_the_same_identifiers_do_not_suppress_a_real_product(self) -> None:
        """The filter must remove identifiers, not remove judgement.

        Without this, an ``_IDENTIFIER_LIKE`` that matched everything would satisfy the test
        above while silently making the whole vocabulary unreachable
        ([[raises-test-needs-a-positive-control]]).
        """
        texts = ("alanine--tRNA ligase", "Genbank:WP_012345678.1", "WP_012345678.1")
        assert synteny.classify_gene_identity(texts) == synteny.CLASS_AARS

    def test_a_readable_but_unrelated_product_is_a_decided_failure_not_unjudgeable(self) -> None:
        assert synteny.classify_gene_identity(("DNA gyrase subunit A",)) is None

    def test_no_identity_text_at_all_is_unjudgeable(self) -> None:
        assert synteny.classify_gene_identity(()) == synteny.FN_UNJUDGEABLE

    def test_the_symbol_route_runs_only_when_the_product_says_nothing(self) -> None:
        """``gatB`` is Asp-tRNA(Asn) amidotransferase — and PTS galactitol subunit IIB.

        Consulting the symbol on a CDS that already has a curated product name is what put
        ``PTS galactitol transporter subunit IIB`` into the transamidation class when it was
        measured.
        """
        assert (
            synteny.classify_gene_identity(
                ("PTS galactitol transporter subunit IIB",), gene_symbols=("gatB",)
            )
            is None
        )
        assert (
            synteny.classify_gene_identity(("hypothetical protein",), gene_symbols=("gatB",))
            == synteny.CLASS_TRANSAMIDATION
        )

    def test_aars_symbols_never_read_as_their_biosynthesis_operon(self) -> None:
        """``argS`` is arginyl-tRNA synthetase; ``argG`` is argininosuccinate synthase."""
        assert synteny.classify_gene_identity((), gene_symbols=("argS",)) == synteny.CLASS_AARS
        assert (
            synteny.classify_gene_identity((), gene_symbols=("argG",))
            == synteny.CLASS_AA_BIOSYNTHESIS
        )


# ═════════════════════════════════════════════════════════════════════════════
# Coordinates and the namespace bridge
# ═════════════════════════════════════════════════════════════════════════════
class TestCoordinates:
    def test_three_prime_end_differs_by_strand(self) -> None:
        # 0-based half-open [100, 200) → 1-based inclusive 101..200.
        assert synteny.locus_three_prime_1based(100, 200, "+") == 200
        assert synteny.locus_three_prime_1based(100, 200, "-") == 101

    @pytest.mark.parametrize("strand", [".", "?", "plus", ""])
    def test_refuses_an_unresolved_strand(self, strand: str) -> None:
        with pytest.raises(synteny.SyntenyError):
            synteny.locus_three_prime_1based(100, 200, strand)

    def test_refuses_a_degenerate_span(self) -> None:
        with pytest.raises(synteny.SyntenyError):
            synteny.locus_three_prime_1based(200, 200, "+")

    def test_contig_seqid_joins_by_identity_not_position(self) -> None:
        ids = ["JAVCCE010000065.1", "JAVCCE010000001.1"]
        assert synteny.contig_seqid(ids, 0) == "JAVCCE010000065.1"
        assert synteny.contig_seqid(ids, 1) == "JAVCCE010000001.1"

    def test_contig_index_out_of_range_refuses(self) -> None:
        with pytest.raises(synteny.SyntenyError):
            synteny.contig_seqid(["a", "b"], 2)


# ═════════════════════════════════════════════════════════════════════════════
# The downstream walk
# ═════════════════════════════════════════════════════════════════════════════
class TestDownstreamWalk:
    def test_minus_strand_start_is_the_features_end(self) -> None:
        left = cds(start=500, end=900, strand="-", product="alanine--tRNA ligase")
        got = synteny.downstream_cds_on_strand(
            [left], seqid="c1", strand="-", three_prime=1000, element_span_nt=100
        )
        assert [d for d, _ in got] == [1000 - 900]

    def test_a_buried_element_does_not_inherit_the_gene_it_sits_inside(self) -> None:
        """⚠ The defect that made 541 of 941 hard negatives "pass" (57.5 % vs a 4.1 % floor).

        A bacterial genome is ~88 % coding, so with no lower bound on the distance almost
        every window finds the gene it is inside of and adopts that gene's identity.
        """
        buried = cds(start=1, end=3000, product="alanine--tRNA ligase")
        got = synteny.downstream_cds_on_strand(
            [buried], seqid="c1", strand="+", three_prime=2000, element_span_nt=100
        )
        assert got == []

    def test_the_bound_is_the_elements_own_span_so_class_ii_overlap_survives(self) -> None:
        overlapping = cds(start=1960, end=2400, product="alanine--tRNA ligase")
        got = synteny.downstream_cds_on_strand(
            [overlapping], seqid="c1", strand="+", three_prime=2000, element_span_nt=100
        )
        assert [d for d, _ in got] == [-40]

    def test_other_contigs_and_other_strands_are_ignored(self) -> None:
        elsewhere = cds(start=2100, end=2400, seqid="c2", product="alanine--tRNA ligase")
        other_strand = cds(start=2100, end=2400, strand="-", product="alanine--tRNA ligase")
        got = synteny.downstream_cds_on_strand(
            [elsewhere, other_strand],
            seqid="c1",
            strand="+",
            three_prime=2000,
            element_span_nt=100,
        )
        assert got == []


class TestTandemCarveOut:
    def _walk(self, features, **kwargs):
        params = {
            "seqid": "c1",
            "strand": "+",
            "three_prime": 1000,
            "element_span_nt": 100,
            "max_intervening_orfs": 1,
            "sub_threshold_orf_nt": 150,
        }
        params.update(kwargs)
        return synteny.resolve_downstream_gene(features, **params)

    def test_hops_an_unjudgeable_orf_and_reanchors_the_window(self) -> None:
        leader = cds(start=1050, end=1200, product="hypothetical protein")
        target = cds(start=1400, end=2000, product="alanine--tRNA ligase")
        got = self._walk([leader, target])
        assert got.function_class == synteny.CLASS_AARS
        assert got.n_intervening == 1 and got.carve_out_applied is True
        # The reported distance stays measured from the ELEMENT, not the re-anchored window.
        assert got.distance_bp == 400

    def test_the_reanchoring_is_real_a_target_past_the_original_window_is_reached(self) -> None:
        leader = cds(start=1050, end=1450, product="hypothetical protein")
        target = cds(start=1520, end=2200, product="alanine--tRNA ligase")
        assert self._walk([leader, target]).function_class == synteny.CLASS_AARS
        # …and with the carve-out disabled it is not.
        assert self._walk([leader, target], max_intervening_orfs=0).function_class == (
            synteny.FN_UNJUDGEABLE
        )

    def test_a_judgeable_non_matching_gene_STOPS_the_walk(self) -> None:
        """A real downstream gene of another function is a criterion failure, not an obstacle."""
        blocker = cds(start=1050, end=1400, product="DNA gyrase subunit A")
        target = cds(start=1450, end=2000, product="alanine--tRNA ligase")
        got = self._walk([blocker, target])
        assert got.function_class is None
        assert got.n_intervening == 0 and got.carve_out_applied is False

    def test_a_sub_threshold_orf_is_hopped_even_when_judgeable(self) -> None:
        tiny = cds(start=1050, end=1120, product="DNA gyrase subunit A")  # 71 bp < 150
        target = cds(start=1200, end=2000, product="alanine--tRNA ligase")
        assert self._walk([tiny, target]).function_class == synteny.CLASS_AARS

    def test_pseudogenes_are_counted_through_the_hop(self) -> None:
        """⚠ Counting only the ORF the walk stopped on reported 0 pseudogenes corpus-wide.

        The carve-out consumes exactly the population D4's pseudogene diagnostic exists to
        size, so the count has to accumulate over the whole walk.
        """
        dead = cds(start=1050, end=1200, product="alanine--tRNA ligase", pseudo=True)
        target = cds(start=1300, end=2000, product="DNA gyrase subunit A")
        got = self._walk([dead, target])
        assert got.n_pseudo_seen == 1
        assert got.n_unjudgeable_seen == 1
        assert got.is_pseudo is False, "the stopped-on ORF is the live one"

    def test_a_pseudogenized_target_routes_to_the_absent_fallback(self) -> None:
        dead = cds(start=1050, end=1400, product="alanine--tRNA ligase", pseudo=True)
        got = self._walk([dead], max_intervening_orfs=0)
        assert got.function_class == synteny.FN_UNJUDGEABLE
        assert got.is_pseudo is True

    def test_nothing_downstream_at_all(self) -> None:
        got = self._walk([])
        assert got.function_class is None and got.distance_bp is None

    def test_out_of_window_is_reported_without_a_gene(self) -> None:
        far = cds(start=1600, end=2000, product="alanine--tRNA ligase")
        got = self._walk([far])
        assert got.function_class is None and got.distance_bp is None

    @pytest.mark.parametrize("bad", [{"max_intervening_orfs": -1}, {"sub_threshold_orf_nt": -1}])
    def test_negative_carve_out_parameters_refuse(self, bad: dict) -> None:
        with pytest.raises(synteny.SyntenyError):
            self._walk([], **bad)

    def test_the_carve_out_parameters_have_no_defaults(self) -> None:
        """ADR-0006 pins no number for either, so the round must state them."""
        params = inspect.signature(synteny.resolve_downstream_gene).parameters
        for name in ("max_intervening_orfs", "sub_threshold_orf_nt"):
            assert params[name].default is inspect.Parameter.empty, name
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


# ═════════════════════════════════════════════════════════════════════════════
# Status mapping + the strand fold
# ═════════════════════════════════════════════════════════════════════════════
class TestSyntenyStatus:
    def _resolved(self, function_class, distance):
        return synteny.DownstreamGene(
            function_class=function_class,
            distance_bp=distance,
            feature_id="f",
            identity_text=(),
            is_pseudo=False,
            n_intervening=0,
            carve_out_applied=False,
        )

    def test_passing_failing_and_unavailable(self) -> None:
        assert synteny.synteny_status(self._resolved(synteny.CLASS_AARS, 10)) == STATUS_PASSED
        assert synteny.synteny_status(self._resolved(None, None)) == STATUS_FAILED
        assert (
            synteny.synteny_status(self._resolved(synteny.FN_UNJUDGEABLE, 10)) == STATUS_UNAVAILABLE
        )

    def test_out_of_window_is_failed_not_unavailable(self) -> None:
        """ "No qualifying gene" is a decided outcome; ``unavailable`` means "could not decide"."""
        assert synteny.synteny_status(self._resolved(synteny.CLASS_AARS, 900)) == STATUS_FAILED

    def test_declaring_the_hmm_fallback_available_without_wiring_it_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fail-open direction has to be a hard error, not a silent pass-through."""
        monkeypatch.setattr(synteny, "HMM_FALLBACK_AVAILABLE", True)
        with pytest.raises(synteny.SyntenyError):
            synteny.synteny_status(self._resolved(synteny.FN_UNJUDGEABLE, 10))

    def test_the_shipped_fallback_flag_is_false(self) -> None:
        """The Pfam/KO profile DB is an unmet §10.2 acquisition; the flag must say so."""
        assert synteny.HMM_FALLBACK_AVAILABLE is False


class TestStrandFold:
    def test_both_resolves_toward_sparing(self) -> None:
        fold = synteny.combine_strand_statuses
        assert fold({"+": STATUS_PASSED, "-": STATUS_FAILED}, policy="both") == STATUS_PASSED
        assert (
            fold({"+": STATUS_UNAVAILABLE, "-": STATUS_FAILED}, policy="both") == STATUS_UNAVAILABLE
        )
        assert fold({"+": STATUS_FAILED, "-": STATUS_FAILED}, policy="both") == STATUS_FAILED
        assert fold({"+": STATUS_PASSED, "-": STATUS_UNAVAILABLE}, policy="both") == STATUS_PASSED

    def test_the_single_strand_policies_read_that_strand_only(self) -> None:
        """Asserted by IDENTITY, not by count: a fold that swapped the two arms would keep
        every count identical ([[symmetric-count-fixture-blind-to-inversion]])."""
        per_strand = {"+": STATUS_PASSED, "-": STATUS_FAILED}
        assert synteny.combine_strand_statuses(per_strand, policy="plus") == STATUS_PASSED
        assert synteny.combine_strand_statuses(per_strand, policy="minus") == STATUS_FAILED

    def test_an_unknown_policy_refuses(self) -> None:
        with pytest.raises(synteny.SyntenyError):
            synteny.combine_strand_statuses({"+": STATUS_PASSED, "-": STATUS_PASSED}, policy="max")
