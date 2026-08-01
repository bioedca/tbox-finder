"""Unit tests for the P3-01 Stage-2 supervised-dataset assembly (PRD §6/§8/§11; ADR-0004).

The tests that matter here are the **fail-closed** ones. Every way this builder can go
wrong is a way a fold, a parent link, or a structure target becomes fiction:

* a derived decoy floating free of its parent's fold (ADR-0004 D7),
* a parentless decoy handed an invented clade holdout unit,
* a consensus pair re-based onto the wrong nucleotide when its partner is deleted,
* a structure target placed at an arbitrary offset when the anchor is ambiguous,
* a per-nucleotide label vector that no longer spans its sequence.

Fold-inheritance is asserted by **identity** (the derived row's every scheme value equals
its parent's, and the fixture gives the parent values no other row has), never by "is not
null" — a count- or nullity-shaped assertion passes just as happily when the builder
copies the wrong row.
"""

from __future__ import annotations

import pytest

from tbox_finder.stage2 import dataset as ds
from tbox_finder.stage2 import tokenizer as tok

pd = pytest.importorskip("pandas")


# --------------------------------------------------------------------- WUSS primitives


def test_wuss_pairs_parses_nested_brackets() -> None:
    assert ds.wuss_pairs("<<..>>") == [(0, 5), (1, 4)]
    assert ds.wuss_pairs("<->_<:>") == [(0, 2), (4, 6)]
    assert ds.wuss_pairs("......") == []


def test_wuss_pairs_rejects_rather_than_repairs() -> None:
    assert ds.wuss_pairs("<<.>") is None, "unclosed"
    assert ds.wuss_pairs("<.>>") is None, "unopened"
    assert ds.wuss_pairs("<(.)>") is None, "unexpected glyph"
    assert ds.wuss_pairs("<A.a>") is None, "pseudoknot glyphs are not silently ignored"


def test_dot_bracket_to_partners_round_trips() -> None:
    assert ds.dot_bracket_to_partners("((..))") == [5, 4, -1, -1, 1, 0]
    assert ds.dot_bracket_to_partners("....") == [-1, -1, -1, -1]
    with pytest.raises(ValueError, match="unbalanced"):
        ds.dot_bracket_to_partners("((.)")
    with pytest.raises(ValueError, match="unexpected dot-bracket glyph"):
        ds.dot_bracket_to_partners("(<)")


# ------------------------------------------------------------------ structure projection


def test_projection_places_pairs_at_the_anchored_offset() -> None:
    # aligned row: GG-CC  (a gap column), structure <<->>  -> pairs (0,4) and (1,3)
    # ungapped GGCC -> DNA GGCC; locus has it once, at offset 3.
    db, status, dropped = ds.project_structure_to_locus(
        aligned_sequence="GG-CC",
        aligned_structure="<<->>",
        locus_dna="AAAGGCCTTT",
    )
    assert status == ds.PAIRING_ANCHORED
    assert dropped == 0
    assert db == "...(())..."
    assert len(db) == 10
    assert ds.dot_bracket_to_partners(db)[3] == 6


def test_projection_drops_a_pair_whose_partner_is_a_deleted_column() -> None:
    """The consensus pair is not formed in this member — dropped and counted, not moved.

    The fixture is asymmetric on purpose: the surviving pair and the dropped pair have
    different spans, so a builder that re-based the dropped pair onto the neighbouring
    nucleotide (rather than dropping it) produces a *different string*, not merely a
    different count.
    """
    # aligned  G  G  -  C  C     structure  <  <  >  >  -
    # consensus pairs are (0,3) and (1,2); column 2 is a gap in this member, so (1,2)
    # has a deleted partner while (0,3) survives.
    db, status, dropped = ds.project_structure_to_locus(
        aligned_sequence="GG-CC",
        aligned_structure="<<>>-",
        locus_dna="GGCC",
    )
    assert status == ds.PAIRING_ANCHORED
    assert dropped == 1
    assert db == "(.).", "only the pair with both partners present is emitted"
    assert ds.dot_bracket_to_partners(db) == [2, -1, 0, -1]


def test_projection_refuses_an_ambiguous_anchor() -> None:
    db, reason, dropped = ds.project_structure_to_locus(
        aligned_sequence="GC",
        aligned_structure="<>",
        locus_dna="GCGC",  # two occurrences
    )
    assert db is None
    assert reason == ds.REJECT_MULTI_ANCHOR
    assert dropped == 0


def test_projection_refuses_an_OVERLAPPING_ambiguous_anchor() -> None:
    """``str.count`` is non-overlapping — ``"AAA".count("AA") == 1``.

    A count-based uniqueness check would call this probe unique and silently place the
    structure at offset 0, which is the mis-placement the rule exists to prevent
    (CodeRabbit r1, PR #93). The disjoint-occurrence case above cannot catch it.
    """
    assert "AAA".count("AA") == 1, "the premise: str.count under-counts overlaps"
    db, reason, _ = ds.project_structure_to_locus(
        aligned_sequence="AA",
        aligned_structure="<>",
        locus_dna="AAA",  # occurrences at offsets 0 and 1, overlapping
    )
    assert db is None
    assert reason == ds.REJECT_MULTI_ANCHOR


@pytest.mark.parametrize(
    ("seq", "struct", "locus", "reason"),
    [
        (None, "<>", "GC", ds.REJECT_NULL_FIELD),
        ("GC", None, "GC", ds.REJECT_NULL_FIELD),
        ("GCA", "<>", "GCA", ds.REJECT_LENGTH_MISMATCH),
        ("G*C", "<->", "GC", ds.REJECT_SEQ_ALPHABET),
        ("G[C", "<->", "GC", ds.REJECT_SEQ_ALPHABET),
        ("GC", "<(", "GC", ds.REJECT_STRUCT_ALPHABET),
        ("GC", "<<", "GC", ds.REJECT_UNBALANCED),
        ("GC", "<>", "AAAA", ds.REJECT_NO_ANCHOR),
    ],
)
def test_projection_reject_reasons_are_specific(
    seq: str | None, struct: str | None, locus: str, reason: str
) -> None:
    db, got, _ = ds.project_structure_to_locus(
        aligned_sequence=seq, aligned_structure=struct, locus_dna=locus
    )
    assert db is None
    assert got == reason
    assert got in ds.REJECT_REASONS


def test_projection_transcribes_before_anchoring() -> None:
    """A U-bearing alignment row must anchor into a T-bearing DNA locus."""
    db, status, _ = ds.project_structure_to_locus(
        aligned_sequence="GUUC", aligned_structure="<..>", locus_dna="AGTTCA"
    )
    assert status == ds.PAIRING_ANCHORED
    assert db == ".(..)."


# ---------------------------------------------------------------------- decoy folding


def test_decoy_fold_is_deterministic_and_order_independent() -> None:
    ids = [f"gcbg_{i:06d}" for i in range(500)]
    first = [ds.decoy_fold(i) for i in ids]
    assert [ds.decoy_fold(i) for i in reversed(ids)][::-1] == first
    assert set(first) == set(ds.FOLD_RANDOM_VALUES), "all three folds are reachable"
    # majority train, per DECOY_FOLD_FRACTIONS — a broken key would flatten this
    assert first.count("train") > first.count("val") + first.count("test")


def test_decoy_fold_depends_on_the_seed() -> None:
    ids = [f"gcbg_{i:06d}" for i in range(200)]
    assert [ds.decoy_fold(i) for i in ids] != [ds.decoy_fold(i, seed=1) for i in ids]


# ----------------------------------------------------------------------- the assembly


#: 16 nt each; the alignment probes ("GGCC" / "AAAA") occur exactly once in their locus,
#: so both positives anchor and the assembly tests are not silently exercising the
#: unanchorable path (the reject reasons have their own parametrised test above).
_LOCUS_A = "GGCCAATTAATTGGAA"
_LOCUS_B = "TTTTAAAACCCCGGGG"


def _corpus(n: int = 2) -> pd.DataFrame:
    loci = [_LOCUS_A, _LOCUS_B][:n]
    return pd.DataFrame(
        {
            "FASTA_sequence": loci,
            "Sequence": ["GGCC", "AAAA"][:n],
            "Structure": ["<..>", "<..>"][:n],
            "tbox_length": [len(s) for s in loci],
        }
    )


def _fixture_frames() -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    """A 2-positive corpus with **asymmetric** fold values, so a wrong-row copy bites."""
    corpus = _corpus()
    from tbox_finder import ingest

    ids = ingest.compute_record_hashes(corpus)
    labels = pd.DataFrame(
        {
            ingest.RECORD_HASH_COL: ids,
            "label_string": ["1" * 16, "A" * 16],
            "regulatory_mode": ["Transcriptional", "Translational"],
            "specifier_codon": ["UGG", None],
            "cognate_aa": ["Trp", None],
            "trna_family": ["Trp", None],
        }
    )
    splits = pd.DataFrame(
        {
            "record_id": ids,
            "parent_record_id": ids,
            "source": ["corpus", "corpus"],
            "klass": ["I", "II"],
            "cluster_id": [11, 22],
            "resolved_phylum": ["Firmicutes", "Actinobacteria"],
            "resolved_class": ["Bacilli", "Actinobacteria"],
            "resolved_order": ["Bacillales", "Corynebacteriales"],
            "resolved_genus": ["Bacillus", "Corynebacterium"],
            "fold_random": ["train", "test"],
            "loo_order_unit": ["Bacillales", "Corynebacteriales"],
            "class_holdout_unit": ["Bacilli", "Actinobacteria"],
            "phylum_holdout_unit": ["Firmicutes", "Actinobacteria"],
            "nested_train": [True, False],
            "nested_role": ["train", "heldout"],
            "is_designated_loo_holdout": [False, True],
            "dropped_from_clade_holdout": [False, False],
            # ADR-0004 A7: the carve is computed by `splits.py` and only *read* here.
            # The training-fold record is in it; the held-out one is not — so the
            # fixture exercises both branches of the inheritance the decoys rely on.
            "calib": [True, False],
        }
    )
    return corpus, ids, labels, splits


def _decoys(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pool": ["dinuc_shuffled", "gc_background", "structured_rna"],
            "decoy_id": ["dinuc_000001_0", "gcbg_000000", "SAM|RF00162|X/1-9"],
            "sequence": ["CCGGTTAA", "ATATATAT", "GCGCGCGC"],
            "length": [8, 8, 8],
            "gc": [0.5, 0.0, 1.0],
            "source": ["dinuc", "gc", "rfam"],
            "accession": [None, None, None],
            "locus_start": [None, None, None],
            "locus_end": [None, None, None],
            "strand": [None, None, None],
            # the SECOND corpus record is the parent, and it is the held-out one:
            # a builder that grabbed row 0 would produce train/nested_train=True here.
            "source_record_id": [ids[1], None, None],
            "masked": [False, False, False],
            "mask_reason": [None, None, None],
        }
    )


def _build(**kwargs):
    corpus, ids, labels, splits = _fixture_frames()
    decoys = kwargs.pop("decoys", None)
    if decoys is None:
        decoys = _decoys(ids)
    return (
        ds.build_dataset(
            corpus=kwargs.pop("corpus", corpus),
            labels=kwargs.pop("labels", labels),
            splits=kwargs.pop("splits", splits),
            decoys=decoys,
            **kwargs,
        ),
        ids,
    )


def test_schema_is_exactly_the_declared_columns() -> None:
    (frame, _), _ = _build()
    assert list(frame.columns) == list(ds.OUTPUT_COLUMNS)
    assert len(frame) == 5  # 2 positives + 3 unmasked decoys


def test_positives_carry_their_own_fold_and_transcribed_rna() -> None:
    (frame, report), ids = _build()
    pos = frame[frame["is_tbox"]].set_index("row_id")
    assert set(pos.index) == set(ids)
    assert pos.loc[ids[0], "rna_sequence"] == _LOCUS_A.replace("T", "U")
    assert pos.loc[ids[0], "fold_random"] == "train"
    assert pos.loc[ids[1], "fold_random"] == "test"
    assert bool(pos.loc[ids[1], "is_designated_loo_holdout"]) is True
    assert pos.loc[ids[0], "fold_basis"] == ds.FOLD_BASIS_CORPUS
    assert pos.loc[ids[0], "n_tokens"] == len(_LOCUS_A) + 2
    assert pos.loc[ids[0], "specifier_codon"] == "UGG"
    assert pos.loc[ids[1], "specifier_codon"] is None
    assert report["n_positives"] == 2


def test_derived_decoy_inherits_every_scheme_value_from_its_parent() -> None:
    """ADR-0004 D7 on the negative side, asserted by identity against the parent row."""
    (frame, _), ids = _build()
    parent = frame[frame["row_id"] == ids[1]].iloc[0]
    child = frame[frame["row_id"] == "dinuc_000001_0"].iloc[0]
    assert child["parent_record_id"] == ids[1]
    assert child["fold_basis"] == ds.FOLD_BASIS_PARENT
    for column in ds.SPLIT_CARRIED_COLUMNS:
        assert child[column] == parent[column], f"{column} did not inherit"
    # the discriminating check: the parent is the held-out record, not the train one
    assert child["fold_random"] == "test"
    assert bool(child["nested_train"]) is False


def test_parentless_decoys_are_random_only_and_invent_no_clade() -> None:
    (frame, report), _ = _build()
    free = frame[frame["fold_basis"] == ds.FOLD_BASIS_DECOY_RANDOM]
    assert set(free["row_id"]) == {"gcbg_000000", "SAM|RF00162|X/1-9"}
    for column in (
        "klass",
        "cluster_id",
        "resolved_phylum",
        "resolved_class",
        "resolved_order",
        "resolved_genus",
        "loo_order_unit",
        "class_holdout_unit",
        "phylum_holdout_unit",
        "nested_train",
        "nested_role",
        "is_designated_loo_holdout",
        "dropped_from_clade_holdout",
    ):
        assert free[column].isna().all(), f"{column} was invented for a parentless decoy"
    assert (~free["clade_holdout_eligible"]).all()
    assert free["fold_random"].notna().all()
    assert set(free["fold_random"]) <= set(ds.FOLD_RANDOM_VALUES)
    # ADR-0004 A7.4: `nested_train` stays null (that is P3-03's call), but `calib` is
    # DECIDED — never null. The whole point of A7.4 is that a parentless decoy's
    # calibration membership is a stated rule, not a join default.
    assert free["nested_train"].isna().all()
    assert free["calib"].notna().all()
    assert report["fold_basis_counts"][ds.FOLD_BASIS_DECOY_RANDOM] == 2


# --------------------------------------------------------------------------- #
# ADR-0004 A7.4 — the parentless-decoy calibration draw
# --------------------------------------------------------------------------- #
def test_decoy_calib_refuses_every_decoy_outside_the_train_portion() -> None:
    """The clause that makes the carve structurally unable to reach the graded split."""
    for fold in ("val", "test"):
        assert not any(
            ds.decoy_calib(f"decoy_{i}", fold_random=fold) for i in range(2000)
        ), f"a {fold}-fold decoy was admitted to calib"


def test_decoy_calib_is_deterministic_and_id_keyed() -> None:
    ids = [f"gcbg_{i:06d}" for i in range(500)]
    first = [ds.decoy_calib(i, fold_random="train") for i in ids]
    # Same answer on a re-run, and on a *reordered* pool — the draw is keyed on the id,
    # not on position, so re-running the build over a superset cannot reshuffle it.
    assert first == [ds.decoy_calib(i, fold_random="train") for i in ids]
    assert dict(zip(ids, first, strict=True)) == {
        i: ds.decoy_calib(i, fold_random="train") for i in reversed(ids)
    }
    assert any(first), "the draw admitted nothing over 500 ids"
    assert not all(first), "the draw admitted everything over 500 ids"


def test_decoy_calib_is_independent_of_the_fold_draw() -> None:
    """A distinct hash domain: calib membership must not be a function of the fold draw.

    Sharing one digest would make `calib` the low tail of the same unit interval
    `decoy_fold` already partitions — i.e. a deterministic *slice* of the train fold
    rather than an independent draw across it.
    """
    ids = [f"gcbg_{i:06d}" for i in range(4000)]
    train = [i for i in ids if ds.decoy_fold(i) == "train"]
    drawn = [i for i in train if ds.decoy_calib(i, fold_random="train")]
    assert drawn, "no decoy was drawn"
    rate = len(drawn) / len(train)
    # Binomial noise around the pinned rate; the assertion is deliberately loose because
    # the *realised* count is a measurement, not a target to tune (CLAUDE.md §10.3).
    assert 0.5 * ds.DECOY_CALIB_RATE < rate < 2.0 * ds.DECOY_CALIB_RATE, rate


def test_negatives_carry_no_positive_only_field() -> None:
    (frame, _), _ = _build()
    neg = frame[~frame["is_tbox"]]
    assert len(neg) == 3
    for column in (
        "label_string",
        "regulatory_mode",
        "specifier_codon",
        "cognate_aa",
        "trna_family",
        "pairing_dotbracket",
    ):
        assert neg[column].isna().all()
    assert (neg["pairing_status"] == ds.PAIRING_NOT_APPLICABLE).all()


def test_masked_decoys_are_excluded() -> None:
    corpus, ids, labels, splits = _fixture_frames()
    decoys = _decoys(ids)
    decoys.loc[1, "masked"] = True
    decoys.loc[1, "mask_reason"] = "union_prior_or_own_positive_flank"
    frame, report = ds.build_dataset(corpus=corpus, labels=labels, splits=splits, decoys=decoys)
    assert "gcbg_000000" not in set(frame["row_id"])
    assert report["n_decoys_masked_out"] == 1
    assert report["n_negatives"] == 2


def test_derived_decoy_with_an_unknown_parent_is_refused() -> None:
    corpus, ids, labels, splits = _fixture_frames()
    decoys = _decoys(ids)
    decoys.loc[0, "source_record_id"] = "0" * 64
    with pytest.raises(ValueError, match="absent from the split table"):
        ds.build_dataset(corpus=corpus, labels=labels, splits=splits, decoys=decoys)
    frame, report = ds.build_dataset(
        corpus=corpus, labels=labels, splits=splits, decoys=decoys, require_full_join=False
    )
    assert report["n_decoys_refused_no_parent"] == 1
    assert "dinuc_000001_0" not in set(frame["row_id"])


def test_corpus_record_without_a_split_row_is_refused() -> None:
    corpus, ids, labels, splits = _fixture_frames()
    with pytest.raises(ValueError, match="no labels/split row"):
        ds.build_dataset(corpus=corpus, labels=labels, splits=splits.iloc[:1], decoys=_decoys(ids))
    frame, report = ds.build_dataset(
        corpus=corpus,
        labels=labels,
        splits=splits.iloc[:1],
        decoys=_decoys(ids).iloc[1:],
        require_full_join=False,
    )
    assert report["n_corpus_refused_unjoined"] == 1
    assert report["n_positives"] == 1


def test_a_null_parent_link_is_refused_not_stringified() -> None:
    """A missing ``parent_record_id`` must reach the null gate, not arrive as ``"nan"``.

    ``str(cell)`` turns a null into the *present-looking* strings ``"None"`` (pandas 2) or
    ``"nan"`` (pandas 3), either of which sails past a not-null check as a fabricated
    parent link — the P3-01 gate requires every row to name its parent for real
    (CodeRabbit r1, PR #93).
    """
    corpus, ids, labels, splits = _fixture_frames()
    splits.loc[0, "parent_record_id"] = None
    with pytest.raises(ValueError, match="parent_record_id is null"):
        ds.build_dataset(corpus=corpus, labels=labels, splits=splits, decoys=_decoys(ids))


def test_a_nonzero_flank_is_refused() -> None:
    corpus, ids, labels, splits = _fixture_frames()
    with pytest.raises(ValueError, match="bare locus"):
        ds.build_dataset(
            corpus=corpus, labels=labels, splits=splits, decoys=_decoys(ids), flank_nt=50
        )


def test_a_label_vector_that_does_not_span_its_locus_is_refused() -> None:
    corpus, ids, labels, splits = _fixture_frames()
    labels.loc[0, "label_string"] = "1" * 15  # one short
    with pytest.raises(ValueError, match="per-nucleotide target would be misaligned"):
        ds.build_dataset(corpus=corpus, labels=labels, splits=splits, decoys=_decoys(ids))


def test_a_locus_over_the_context_window_is_refused() -> None:
    corpus, ids, labels, splits = _fixture_frames()
    long_locus = "A" * (tok.MAX_NUCLEOTIDE_TOKENS + 1)
    corpus.loc[0, "FASTA_sequence"] = long_locus
    corpus.loc[0, "tbox_length"] = len(long_locus)
    from tbox_finder import ingest

    new_ids = ingest.compute_record_hashes(corpus)
    labels[ingest.RECORD_HASH_COL] = new_ids
    labels.loc[0, "label_string"] = "1" * len(long_locus)
    splits["record_id"] = new_ids
    splits["parent_record_id"] = new_ids
    with pytest.raises(ValueError, match="exceeds RiNALMo"):
        ds.build_dataset(corpus=corpus, labels=labels, splits=splits, decoys=_decoys(new_ids))


def test_report_pairing_accounting_adds_up() -> None:
    (frame, report), _ = _build()
    counts = report["pairing_status_counts"]
    assert sum(counts.values()) == report["n_rows"]
    assert counts[ds.PAIRING_NOT_APPLICABLE] == report["n_negatives"]
    assert (
        sum(report["pairing_reject_reasons"].values()) == counts[ds.PAIRING_UNANCHORABLE]
    ), "every unanchorable row names its reason"
    assert report["digest"] == ds.dataset_digest(frame)


def test_digest_is_order_independent_but_content_sensitive() -> None:
    (frame, _), _ = _build()
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    assert ds.dataset_digest(shuffled) == ds.dataset_digest(frame)
    mutated = frame.copy()
    mutated.loc[0, "fold_random"] = "val"
    assert ds.dataset_digest(mutated) != ds.dataset_digest(frame)
