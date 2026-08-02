"""Unit gate for the P3-03 Stage-2 head vocabularies (:mod:`tbox_finder.stage2.heads`).

Torch-free, so this whole file runs in the bare CI env — which is the point: the
cardinalities it locks are the numbers :class:`~tbox_finder.stage2.model.Stage2Model`
sizes its logit axes from, and CI is the only place that check runs on every push.

Two properties get most of the attention:

* the canonical axes are **derived**, not re-typed — asserted by comparing against the
  upstream constant itself, so a change to :data:`tbox_finder.labels.CLASS_ORDER`
  propagates rather than leaving a stale literal behind;
* an unmappable label **masks** rather than guessing, including the pandas-3 ``NaN``
  that ``str(x or "")`` would have turned into the string ``"nan"`` (P2-10d′-c).
"""

from __future__ import annotations

import json
import re

import pytest

from tbox_finder import ingest, labels
from tbox_finder.stage2 import heads as H

# --------------------------------------------------------------------------- #
# The canonical axes are imported/enumerated, never re-typed here.
# --------------------------------------------------------------------------- #


class TestCanonicalVocabularies:
    def test_boundary_axis_is_the_adr0004_class_order(self):
        # Compared against the constant, not a copy of it: this is what makes the head
        # width follow labels.py instead of going stale beside it.
        assert H.BOUNDARY_CLASSES == labels.CLASS_ORDER
        assert len(H.BOUNDARY_CLASSES) == 8
        assert H.BOUNDARY_CLASSES[0] == "background"

    def test_boundary_indices_match_the_labels_module(self):
        spec = _spec()
        assert spec.index_map(H.BOUNDARY_FIELD) == labels.CLASS_INDEX

    def test_regulatory_modes_are_the_two_prd_8_modes(self):
        assert tuple(ingest.TYPE_TO_CLASS) == H.REGULATORY_MODE_CLASSES
        assert H.REGULATORY_MODE_CLASSES == ("Transcriptional", "Translational")
        # "Unknown" is an admissible *record* state (ingest.REGULATION_ALLOWED) but not a
        # mode to predict, so it must not have a logit.
        assert "Unknown" in ingest.REGULATION_ALLOWED
        assert "Unknown" not in H.REGULATORY_MODE_CLASSES

    def test_codon_axis_is_the_64_rna_triplets(self):
        codons = H.SPECIFIER_CODON_CLASSES
        assert len(codons) == 64
        assert len(set(codons)) == 64
        assert all(len(c) == 3 and set(c) <= set(H.RNA_BASES) for c in codons)
        assert codons[0] == "AAA" and codons[-1] == "UUU"
        # T→U alphabet: a DNA triplet must not be in the vocabulary.
        assert "AAT" not in codons


# --------------------------------------------------------------------------- #
# Stage2HeadSpec construction + validation
# --------------------------------------------------------------------------- #


def _spec(**overrides) -> H.Stage2HeadSpec:
    kwargs = {
        "cognate_aa_classes": ("ILE", "LEU", "TRP"),
        "trna_family_classes": ("ILE (GAU)", "LEU (GAG)"),
    }
    kwargs.update(overrides)
    return H.Stage2HeadSpec(**kwargs)


class TestSpecValidation:
    def test_head_sizes_cover_every_categorical_head(self):
        spec = _spec()
        assert spec.head_sizes == {
            "label_string": 8,
            "regulatory_mode": 2,
            "specifier_codon": 64,
            "cognate_aa": 3,
            "trna_family": 2,
        }
        assert set(spec.head_sizes) == set(H.VOCAB_FIELDS)

    def test_lists_are_coerced_to_tuples(self):
        # A frozen dataclass holding a list is mutable through the back door.
        spec = _spec(cognate_aa_classes=["ILE", "LEU"])
        assert isinstance(spec.cognate_aa_classes, tuple)

    def test_empty_vocabulary_is_refused(self):
        with pytest.raises(ValueError, match="vocabulary is empty"):
            _spec(cognate_aa_classes=())

    def test_duplicate_class_is_refused(self):
        with pytest.raises(ValueError, match="duplicate class label"):
            _spec(cognate_aa_classes=("ILE", "ILE"))

    def test_blank_class_is_refused(self):
        with pytest.raises(ValueError, match="blank class label"):
            _spec(cognate_aa_classes=("ILE", "   "))

    def test_unknown_field_is_refused(self):
        with pytest.raises(KeyError, match="unknown head field"):
            _spec().classes("not_a_column")


# --------------------------------------------------------------------------- #
# Encoding: mapped, or masked — never guessed
# --------------------------------------------------------------------------- #


class TestEncoding:
    def test_known_value_maps_to_its_index(self):
        spec = _spec()
        assert spec.encode(H.AMINO_ACID_FIELD, "LEU") == 1
        assert spec.encode(H.CODON_FIELD, "AAA") == 0
        assert spec.encode(H.REGULATORY_MODE_FIELD, "Translational") == 1

    @pytest.mark.parametrize(
        "value",
        [None, float("nan"), "", "   "],
        ids=["none", "nan", "empty", "whitespace"],
    )
    def test_missing_values_mask(self, value):
        # float("nan") is the load-bearing case: under the training env's pandas 3 a null
        # string cell arrives as a TRUTHY NaN, so `str(x or "")` yields "nan" — a
        # present-looking value in no vocabulary. is_missing() is what catches it.
        assert _spec().encode(H.AMINO_ACID_FIELD, value) == H.IGNORE_INDEX

    def test_the_nan_gotcha_masks_even_when_nan_is_a_vocabulary_entry(self):
        # The pandas-3 hazard has two halves and both must mask: the float NaN (caught by
        # is_missing) and the string it renders to (caught by is_admissible_label). A spec
        # that somehow carries "nan" as a class must still refuse the value, or a corpus
        # built under pandas 3 would train the head on its own missing-value sentinel.
        spec = _spec(cognate_aa_classes=("ILE", "nan"))
        assert spec.encode(H.AMINO_ACID_FIELD, float("nan")) == H.IGNORE_INDEX
        assert spec.encode(H.AMINO_ACID_FIELD, "nan") == H.IGNORE_INDEX
        assert spec.encode(H.AMINO_ACID_FIELD, "ILE") == 0

    def test_out_of_vocabulary_value_masks(self):
        spec = _spec()
        # TBDB's gap-bearing "codons" are real corpus values and must not become a class.
        assert spec.encode(H.CODON_FIELD, "CC-") == H.IGNORE_INDEX
        assert spec.encode(H.CODON_FIELD, "7]*") == H.IGNORE_INDEX
        assert spec.encode(H.AMINO_ACID_FIELD, "SEC") == H.IGNORE_INDEX

    def test_ignore_index_is_the_cross_entropy_default(self):
        assert H.IGNORE_INDEX == -100


class TestBoundaryStringEncoding:
    def test_codes_map_through_the_canonical_table(self):
        spec = _spec()
        encoded = spec.encode_boundary_string(".1S2")
        assert encoded == [
            labels.CLASS_INDEX["background"],
            labels.CLASS_INDEX["Stem_I"],
            labels.CLASS_INDEX["Specifier"],
            labels.CLASS_INDEX["Stem_II"],
        ]

    def test_every_class_code_round_trips(self):
        spec = _spec()
        codes = "".join(labels.CLASS_CODE[name] for name in labels.CLASS_ORDER)
        assert spec.encode_boundary_string(codes) == list(range(len(labels.CLASS_ORDER)))

    @pytest.mark.parametrize("value", [None, float("nan"), ""], ids=["none", "nan", "empty"])
    def test_a_row_without_a_label_string_yields_no_targets(self, value):
        # Every decoy row is this case; it must be empty, not an exception.
        assert _spec().encode_boundary_string(value) == []

    def test_an_unknown_code_inside_a_present_string_raises(self):
        # Unlike a whole-row absence, a stray glyph means the upstream derivation changed.
        with pytest.raises(ValueError, match="unknown class code"):
            _spec().encode_boundary_string("..X..")


# --------------------------------------------------------------------------- #
# derive_head_spec
# --------------------------------------------------------------------------- #

#: Deliberately ASYMMETRIC (3 amino acids vs 2 tRNA families, different row counts) and
#: asserted by IDENTITY, not by count: two equal-sized vocabularies compared only by
#: length pass under a swap of the two fields.
_CORPUS = [
    {"cognate_aa": "TRP", "trna_family": "TRP (CCA)"},
    {"cognate_aa": "ILE", "trna_family": "ILE (GAU)"},
    {"cognate_aa": "ILE", "trna_family": "ILE (GAU)"},
    {"cognate_aa": "LEU", "trna_family": "ILE (GAU)"},
    {"cognate_aa": None, "trna_family": None},
    {"cognate_aa": float("nan"), "trna_family": float("nan")},
    {"cognate_aa": "", "trna_family": "  "},
    {},
]


class TestDeriveHeadSpec:
    def test_derived_vocabularies_are_the_observed_values_sorted(self):
        spec = H.derive_head_spec(_CORPUS)
        assert spec.cognate_aa_classes == ("ILE", "LEU", "TRP")
        assert spec.trna_family_classes == ("ILE (GAU)", "TRP (CCA)")

    def test_missing_and_blank_cells_contribute_nothing(self):
        spec = H.derive_head_spec(_CORPUS)
        assert "" not in spec.cognate_aa_classes
        assert "nan" not in spec.cognate_aa_classes
        assert len(spec.cognate_aa_classes) == 3

    def test_row_order_does_not_change_the_indices(self):
        forward = H.derive_head_spec(_CORPUS)
        backward = H.derive_head_spec(list(reversed(_CORPUS)))
        assert forward == backward

    def test_only_the_two_uncanonical_axes_are_derived(self):
        spec = H.derive_head_spec(_CORPUS)
        assert spec.boundary_classes == H.BOUNDARY_CLASSES
        assert spec.regulatory_mode_classes == H.REGULATORY_MODE_CLASSES
        assert spec.specifier_codon_classes == H.SPECIFIER_CODON_CLASSES
        assert H.DERIVED_VOCAB_FIELDS == ("cognate_aa", "trna_family")


# --------------------------------------------------------------------------- #
# coverage_report — what got masked must be countable
# --------------------------------------------------------------------------- #


class TestCoverageReport:
    def test_out_of_vocabulary_is_counted_apart_from_absent(self):
        spec = _spec()
        rows = [
            {"specifier_codon": "AAA", "label_string": ".1S"},  # encoded
            {"specifier_codon": "CC-", "label_string": None},  # present but OOV
            {"specifier_codon": "7]*", "label_string": ""},  # present but OOV
            {"specifier_codon": None, "label_string": "..."},  # absent
        ]
        report = H.coverage_report(spec, rows)
        codon = report["fields"]["specifier_codon"]
        assert report["n_rows"] == 4
        assert codon["present"] == 3
        assert codon["encoded"] == 1
        assert codon["out_of_vocabulary"] == 2
        assert codon["stringified_null"] == 0
        assert codon["masked"] == 3  # 2 OOV + 1 absent
        assert codon["examples"] == ["7]*", "CC-"]

    def test_boundary_rows_and_positions_are_counted(self):
        report = H.coverage_report(
            _spec(),
            [{"label_string": ".1S"}, {"label_string": None}, {"label_string": "12"}],
        )
        boundary = report["label_string"]
        assert boundary == {"rows_with_label": 2, "rows_masked": 1, "n_positions": 5}

    def test_a_fully_masked_field_reports_zero_encoded(self):
        # "no signal" and "no power" must not read identically: a field where nothing
        # encodes has to be visible as encoded == 0, not as an absent key.
        report = H.coverage_report(_spec(), [{"cognate_aa": None} for _ in range(5)])
        assert report["fields"]["cognate_aa"]["encoded"] == 0
        assert report["fields"]["cognate_aa"]["masked"] == 5


# --------------------------------------------------------------------------- #
# Stringified nulls — a null that survived being rendered into text
# --------------------------------------------------------------------------- #


class TestStringifiedNulls:
    @pytest.mark.parametrize("text", ["None", "none", "NaN", "nan", "NA", "<NA>", "null", "N/A"])
    def test_a_whole_cell_null_token_is_inadmissible(self, text):
        assert not H.is_admissible_label(H.AMINO_ACID_FIELD, text)
        assert _spec().encode(H.AMINO_ACID_FIELD, text) == H.IGNORE_INDEX

    def test_a_trna_family_with_a_null_amino_acid_is_inadmissible(self):
        # TBDB composes f"{amino_acid} ({anticodon})"; a record with no amino acid gives
        # the literal below. It is well-formed text, so no missing check can see it.
        assert not H.is_admissible_label(H.TRNA_FAMILY_FIELD, "None (UCA)")

    def test_a_real_trna_family_is_admissible(self):
        assert H.is_admissible_label(H.TRNA_FAMILY_FIELD, "ILE (GAU)")

    def test_the_guard_is_scoped_to_the_trna_family_axis(self):
        # An amino acid literally spelled "None" is caught by the whole-cell rule, but a
        # value that merely CONTAINS a null token elsewhere must not be refused.
        assert H.is_admissible_label(H.AMINO_ACID_FIELD, "ASN")
        assert H.is_admissible_label(H.CODON_FIELD, "AAA")

    def test_a_stringified_null_never_enters_a_derived_vocabulary(self):
        spec = H.derive_head_spec(
            [
                {"cognate_aa": "ILE", "trna_family": "ILE (GAU)"},
                {"cognate_aa": "", "trna_family": "None (UCA)"},
                {"cognate_aa": "None", "trna_family": "None"},
            ]
        )
        assert spec.trna_family_classes == ("ILE (GAU)",)
        assert spec.cognate_aa_classes == ("ILE",)

    def test_encoding_refuses_it_even_if_a_spec_carries_it(self):
        # Build and encode must agree: a spec hand-built with the bad class must still
        # mask the value, or the two disagree about what the corpus contains.
        spec = _spec(trna_family_classes=("ILE (GAU)", "None (UCA)"))
        assert spec.encode(H.TRNA_FAMILY_FIELD, "None (UCA)") == H.IGNORE_INDEX
        assert spec.encode(H.TRNA_FAMILY_FIELD, "ILE (GAU)") == 0

    def test_coverage_counts_it_apart_from_ordinary_absence(self):
        report = H.coverage_report(
            _spec(),
            [
                {"trna_family": "ILE (GAU)"},
                {"trna_family": "None (UCA)"},
                {"trna_family": None},
            ],
        )
        bucket = report["fields"]["trna_family"]
        assert bucket["encoded"] == 1
        assert bucket["stringified_null"] == 1
        assert bucket["out_of_vocabulary"] == 0
        assert bucket["masked"] == 2
        assert bucket["examples"] == ["None (UCA)"]


# --------------------------------------------------------------------------- #
# The committed vocabulary artifact
# --------------------------------------------------------------------------- #

#: The committed head widths. Derived from the full 30,542-row Stage-2 dataset, so a
#: change here is a change to every Stage-2 checkpoint's logit shape: regenerate
#: head_vocab.json and edit this literal in the SAME commit, never one without the other.
_COMMITTED_SIZES = {
    "label_string": 8,
    "regulatory_mode": 2,
    "specifier_codon": 64,
    "cognate_aa": 20,
    "trna_family": 61,
}


class TestCommittedHeadVocab:
    def test_document_loads_into_a_spec(self):
        spec = H.load_head_spec()
        assert spec.head_sizes == _COMMITTED_SIZES

    def test_sizes_agree_with_the_vocabularies_they_describe(self):
        document = H.read_head_vocab()
        for field, size in document["sizes"].items():
            assert size == len(document["vocabularies"][field]), field

    def test_canonical_axes_match_the_in_repo_constants(self):
        vocabularies = H.read_head_vocab()["vocabularies"]
        assert tuple(vocabularies["label_string"]) == labels.CLASS_ORDER
        assert tuple(vocabularies["regulatory_mode"]) == H.REGULATORY_MODE_CLASSES
        assert tuple(vocabularies["specifier_codon"]) == H.SPECIFIER_CODON_CLASSES

    def test_derived_axes_are_sorted_unique_and_well_shaped(self):
        vocabularies = H.read_head_vocab()["vocabularies"]
        for field in H.DERIVED_VOCAB_FIELDS:
            entries = vocabularies[field]
            assert entries == sorted(entries), field
            assert len(set(entries)) == len(entries), field
        # 20 standard amino acids, TBDB's uppercase three-letter spelling.
        assert all(re.fullmatch(r"[A-Z]{3}", aa) for aa in vocabularies["cognate_aa"])
        # tRNA families are "<AA> (<anticodon>)" over the RNA alphabet. The amino-acid
        # component is a real amino acid, never the literal "None" TBDB renders when the
        # record has none (see TestStringifiedNulls).
        assert all(
            re.fullmatch(r"[A-Z]{3} \([ACGU]{3}\)", fam) for fam in vocabularies["trna_family"]
        )
        amino_acids = set(vocabularies["cognate_aa"])
        assert all(fam.split(" ", 1)[0] in amino_acids for fam in vocabularies["trna_family"])

    def test_provenance_pins_the_corpus_the_vocabularies_came_from(self):
        provenance = H.read_head_vocab()["provenance"]
        assert provenance["source_parquet"] == "data/processed/stage2_dataset.parquet"
        assert provenance["derived_by"] == "tbox_finder.stage2.heads.derive_head_spec"
        assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_dataset_digest"])
        assert provenance["source_n_rows"] == 30542
        assert provenance["coverage"]["n_rows"] == provenance["source_n_rows"]

    def test_the_masked_specifier_codons_are_disclosed_not_silent(self):
        # 202 rows carry a gap-bearing or corrupt "codon"; head (d1) never sees them. The
        # number is in the artifact so it can be read, rather than inferred from a shape.
        coverage = H.read_head_vocab()["provenance"]["coverage"]["fields"]["specifier_codon"]
        assert coverage["out_of_vocabulary"] == 202
        assert coverage["encoded"] == 22941
        assert "---" in coverage["examples"]

    def test_the_stringified_null_trna_families_are_disclosed_not_silent(self):
        # Nine rows render a missing amino acid as the family "None (UCA)". They are
        # refused, counted apart from ordinary absence, and named in the artifact.
        fields = H.read_head_vocab()["provenance"]["coverage"]["fields"]
        assert fields["trna_family"]["stringified_null"] == 9
        assert fields["trna_family"]["examples"] == ["None (UCA)"]

    def test_a_trna_family_is_encoded_exactly_when_its_amino_acid_is(self):
        # The family string is COMPOSED from the amino acid, so the two heads must see
        # the same rows. They diverged by exactly the 9 "None (UCA)" rows before the
        # guard, which is what made the defect visible.
        fields = H.read_head_vocab()["provenance"]["coverage"]["fields"]
        assert fields["trna_family"]["encoded"] == fields["cognate_aa"]["encoded"] == 22932


class TestLoadHeadSpecRefusals:
    def _write(self, tmp_path, mutate):
        document = H.read_head_vocab()
        mutate(document)
        path = tmp_path / "head_vocab.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_inconsistent_sizes_are_refused(self, tmp_path):
        def mutate(doc):
            doc["sizes"]["cognate_aa"] = 999

        with pytest.raises(ValueError, match="internally inconsistent"):
            H.load_head_spec(self._write(tmp_path, mutate))

    def test_a_drifted_canonical_axis_is_refused(self, tmp_path):
        def mutate(doc):
            doc["vocabularies"]["label_string"] = list(reversed(labels.CLASS_ORDER))

        # Same length, different order — every per-nt target would be silently re-indexed.
        with pytest.raises(ValueError, match="drifted from its canonical"):
            H.load_head_spec(self._write(tmp_path, mutate))

    def test_another_schema_version_is_refused(self, tmp_path):
        # A version field nothing reads records intent and enforces nothing.
        def mutate(doc):
            doc["schema_version"] = "9.9"

        with pytest.raises(ValueError, match="schema_version"):
            H.load_head_spec(self._write(tmp_path, mutate))

    def test_a_sizes_key_with_no_vocabulary_is_an_inconsistency(self, tmp_path):
        def mutate(doc):
            doc["sizes"]["not_a_column"] = 3

        # Not a bare KeyError on the lookup — the document is what is wrong.
        with pytest.raises(ValueError, match="internally inconsistent"):
            H.load_head_spec(self._write(tmp_path, mutate))

    def test_a_vocabulary_with_no_sizes_entry_is_refused(self, tmp_path):
        # Otherwise that axis skips the checksum entirely and the redundancy the sizes
        # block exists to provide is silently absent for it.
        def mutate(doc):
            doc["sizes"].pop("trna_family")

        with pytest.raises(ValueError, match="internally inconsistent"):
            H.load_head_spec(self._write(tmp_path, mutate))

    def test_a_missing_corpus_derived_axis_is_named(self, tmp_path):
        def mutate(doc):
            doc["vocabularies"].pop("cognate_aa")
            doc["sizes"].pop("cognate_aa")

        # The three canonical axes have defaults; these two have none, so the error must
        # say which axis and how to rebuild it, not just which dict key was looked up.
        with pytest.raises(KeyError, match="derive_head_spec"):
            H.load_head_spec(self._write(tmp_path, mutate))

    def test_from_dict_falls_back_only_for_the_canonical_axes(self):
        spec = H.Stage2HeadSpec.from_dict({"cognate_aa": ["ILE"], "trna_family": ["ILE (GAU)"]})
        assert spec.boundary_classes == H.BOUNDARY_CLASSES
        assert spec.regulatory_mode_classes == H.REGULATORY_MODE_CLASSES
        assert spec.specifier_codon_classes == H.SPECIFIER_CODON_CLASSES
        with pytest.raises(KeyError, match="trna_family"):
            H.Stage2HeadSpec.from_dict({"cognate_aa": ["ILE"]})
