"""P3-16′-remint unit — the decoy-host screen's two rules, its verdicts, and its validator.

What is pinned here, and why each one can fail
----------------------------------------------
* **The k-mer kernel.** A15 Pin 1's rule 1 is "any N-free canonical 32-mer on either strand".
  Canonicality is asserted as an *identity* (a sequence and its reverse complement yield the
  same multiset), not as a count: equal counts survive a strand-sense inversion
  ([[symmetric-count-fixture-blind-to-inversion]]). The non-ACGT skip is asserted by the exact
  number of windows an ``N`` removes, so a rule that stopped skipping would differ by a
  measurable amount rather than by a flag.
* **The tblout parse, on real Infernal 1.1.5 output.** ``tests/fixtures/host_screen/`` holds two
  real tblouts over two real catalogued RF00230 T-boxes and one real mined genomic window
  (CLAUDE.md §8.7: no mocked tool output). They pin that the verdict reads the **bit** column
  and that the tblout's query-name column is the CM's own ``NAME`` (``T-box`` / ``seed_ILE``) —
  the string that would silently zero every per-model score if a verdict keyed on it.
* **The executed command line.** ``search_argv`` is a mirror of ``infernal.run_cmsearch``'s
  construction, so it is pinned against the ``# Option settings:`` line the binary itself wrote
  into the committed fixture ([[verify-the-line-you-ship]]).
* **The validator.** A synthetic — and deliberately self-consistent — report validates clean,
  and then each clause is broken **alone**, with every other field left consistent: a forge that
  breaks two coupled fields at once cannot tell which clause bit
  ([[all-true-fixture-cannot-test-a-conjunction]]).

Nothing here is a measurement of the real pool. The synthetic reports carry invented counts and
are labelled as such; the real screen's numbers live in ``reports/p3/host_screen.json``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from tbox_finder import masking
from tbox_finder.data import window_dataset as wd
from tbox_finder.integration import host_screen as H

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO / "tests/fixtures/host_screen"


def _panel_meta() -> dict:
    """The committed real-sequence fixture: two catalogued RF00230 T-boxes and a mined window.

    Plain JSON rather than a ``.fa``: ``*.fa`` is git-LFS-tracked in this repo, and a fixture CI
    reads as REAL sequences must not arrive as a 132-byte pointer
    ([[git-lfs-pointers-in-ci]]) — the same reason the curated-control query FASTA and
    ``tests/fixtures/cm_confirmer/**`` carry LFS carve-outs. The committed tblouts were produced
    by searching a FASTA written from exactly these records.
    """
    return json.loads((_FIXTURE / "panel.json").read_text(encoding="utf-8"))


def _panel() -> dict[str, str]:
    """``{name: sequence}`` over the fixture's real sequences."""
    return {name: record["sequence"] for name, record in _panel_meta()["records"].items()}


def test_the_fixture_sequences_are_the_lengths_the_fixture_declares():
    """A fixture whose sequences were truncated would make every bit score below unreadable."""
    for name, record in _panel_meta()["records"].items():
        assert len(record["sequence"]) == record["length"], name
        assert set(record["sequence"]) <= set("ACGTN"), name


def _tblout(model: str) -> str:
    return (_FIXTURE / f"{model}.tblout").read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════════════
# normalise_dna / canonical_kmers / kmer_hits
# ══════════════════════════════════════════════════════════════════════════════════════
def test_normalise_dna_upper_cases_before_transcribing():
    assert H.normalise_dna("acgu") == "ACGT"
    assert H.normalise_dna("AcGu-n") == "ACGT-N"


def test_canonical_kmers_counts_every_window_of_a_clean_sequence():
    sequence = _panel()["w000003"]
    assert len(sequence) == 1024
    assert H.canonical_kmers(sequence).size == 1024 - H.KMER_K + 1


def test_canonical_kmers_is_strand_canonical_as_an_identity():
    """Not a count: equal counts also survive a fold-SENSE inversion.

    The multiset of canonical k-mers of a sequence must be *identical* to that of its reverse
    complement, which is what makes one sorted array screen both strands.
    """
    sequence = _panel()["w000001"]
    forward = np.sort(H.canonical_kmers(sequence))
    reverse = np.sort(H.canonical_kmers(wd.reverse_complement(sequence)))
    assert forward.size > 0
    assert np.array_equal(forward, reverse)


def test_canonical_kmers_folds_each_window_to_the_smaller_strand():
    """A hand-checked k = 4 value, so the 2-bit code itself is pinned, not just its symmetry."""
    # AAAA -> 0b00000000 = 0; its reverse complement TTTT -> 0b11111111 = 255; min is 0.
    assert H.canonical_kmers("AAAA", 4).tolist() == [0]
    assert H.canonical_kmers("TTTT", 4).tolist() == [0]
    # ACGT is its own reverse complement: 0b00011011 = 27.
    assert H.canonical_kmers("ACGT", 4).tolist() == [0b00011011]


def test_canonical_kmers_skips_exactly_the_windows_an_n_spans():
    """A15 Pin 1's alphabet rule, as a count: an interior N removes k windows, no more."""
    clean = "A" * 40 + "C" * 40
    k = 8
    assert H.canonical_kmers(clean, k).size == len(clean) - k + 1
    dirty = clean[:40] + "N" + clean[41:]
    assert H.canonical_kmers(dirty, k).size == len(clean) - k + 1 - k


def test_canonical_kmers_is_32_wide_and_collision_free_at_k_32():
    """Two 32-mers differing only at the last base must not collide (the full 64-bit code)."""
    first = H.canonical_kmers("A" * 31 + "C", 32)
    second = H.canonical_kmers("A" * 31 + "G", 32)
    assert first.size == second.size == 1
    assert first[0] != second[0]
    assert first.dtype == np.uint64


def test_canonical_kmers_refuses_a_k_a_uint64_cannot_hold():
    with pytest.raises(H.HostScreenError):
        H.canonical_kmers("A" * 40, 33)


def test_canonical_kmers_transcribes_so_rna_and_dna_sources_intersect():
    """The Stage-2 source is RNA; a screen that did not transcribe would share nothing."""
    dna = _panel()["w000001"][:100]
    rna = dna.replace("T", "U")
    assert np.array_equal(H.canonical_kmers(dna), H.canonical_kmers(rna))


def test_sorted_unique_kmers_equals_one_unique_over_the_concatenation():
    sequences = [_panel()["w000001"], _panel()["w000002"]]
    incremental = H.sorted_unique_kmers(sequences, compress_at=64)
    at_once = np.unique(np.concatenate([H.canonical_kmers(s) for s in sequences]))
    assert np.array_equal(incremental, at_once)


def test_kmer_hits_counts_shared_windows_and_is_zero_on_disjoint_dna():
    source = H.sorted_unique_kmers([_panel()["w000001"]])
    shared = H.canonical_kmers(_panel()["w000001"][:100])
    assert H.kmer_hits(shared, source) == shared.size
    assert H.kmer_hits(H.canonical_kmers("ACGT" * 40), source) == 0
    assert H.kmer_hits(np.empty(0, dtype=np.uint64), source) == 0
    assert H.kmer_hits(shared, np.empty(0, dtype=np.uint64)) == 0


def test_kmer_hits_sees_a_reverse_complement_insert():
    """The both-strand claim, end to end: an insert placed in its RC is still shared DNA."""
    source = H.sorted_unique_kmers([_panel()["w000001"]])
    reversed_insert = wd.reverse_complement(_panel()["w000001"][200:260])
    assert H.kmer_hits(H.canonical_kmers(reversed_insert), source) > 0


def test_familiarity_fraction_is_none_without_a_single_clean_kmer():
    source = H.sorted_unique_kmers([_panel()["w000001"]])
    assert H.familiarity_fraction("N" * 100, source) is None
    assert H.familiarity_fraction(_panel()["w000001"], source) == 1.0
    assert H.familiarity_fraction("ACGT" * 40, source) == 0.0


# ══════════════════════════════════════════════════════════════════════════════════════
# intervals_overlap (A15 Pin 1 rule 2)
# ══════════════════════════════════════════════════════════════════════════════════════
def test_intervals_overlap_is_closed_and_strand_folded():
    assert H.intervals_overlap("X", 100, 200, "X", 200, 300) is True  # touching endpoints
    assert H.intervals_overlap("X", 100, 200, "X", 201, 300) is False
    # The repo encodes strand as start > end; the fold must not change the verdict.
    assert H.intervals_overlap("X", 200, 100, "X", 150, 160) is True


def test_intervals_overlap_normalises_the_insdc_version_on_both_sides():
    assert H.intervals_overlap("NZ_CP009247.1", 10, 20, "NZ_CP009247", 15, 25) is True
    assert H.intervals_overlap("NZ_CP009247", 10, 20, "NZ_CP009247.3", 15, 25) is True
    assert H.intervals_overlap("NZ_CP009247.1", 10, 20, "OTHER.1", 15, 25) is False


def test_intervals_overlap_refuses_a_missing_accession_rather_than_matching_everything():
    assert H.intervals_overlap(None, 10, 20, "X", 10, 20) is False
    assert H.intervals_overlap("X", 10, 20, float("nan"), 10, 20) is False


# ══════════════════════════════════════════════════════════════════════════════════════
# The tblout parse and the verdict (real Infernal 1.1.5 output)
# ══════════════════════════════════════════════════════════════════════════════════════
def test_parse_tblout_reads_the_bit_column_of_real_output():
    hits = {hit.target: hit for hit in H.parse_tblout_hits(_tblout("RF00230"))}
    assert set(hits) == {"w000001", "w000002"}
    # The two catalogued T-boxes; the mined genomic window drew no hit at -T 10.
    assert hits["w000001"].score == pytest.approx(115.1)
    assert hits["w000002"].score == pytest.approx(102.0)
    # The bit column is NOT the E-value column, which is ~1e-27 for the same row.
    assert hits["w000001"].evalue < 1e-20
    assert hits["w000001"].strand == "+"
    assert hits["w000001"].trunc == "no"
    assert hits["w000001"].inc == "!"


def test_parse_tblout_keeps_the_cms_own_name_not_the_repo_model_key():
    """The field a per-model dict must NOT be keyed on."""
    assert {hit.model for hit in H.parse_tblout_hits(_tblout("RF00230"))} == {"T-box"}
    assert {hit.model for hit in H.parse_tblout_hits(_tblout("TBDB001"))} == {"seed_ILE"}
    assert H.CM_QUERY_NAMES == {"RF00230": "T-box", "TBDB001": "seed_ILE"}


def test_check_query_names_refuses_a_tblout_attributed_to_the_wrong_model():
    hits = H.parse_tblout_hits(_tblout("RF00230"))
    H.check_query_names(hits, "RF00230")  # the pairing the fixture was produced under
    with pytest.raises(H.HostScreenError):
        H.check_query_names(hits, "TBDB001")
    with pytest.raises(H.HostScreenError):
        H.check_query_names(hits, "RF00231")


def test_best_bits_attributes_by_the_search_that_produced_the_hits():
    grouped = {
        model: H.hits_by_target(_tblout(model), model=model) for model in ("RF00230", "TBDB001")
    }
    bits = H.best_bits({model: grouped[model].get("w000002", []) for model in grouped})
    assert bits["RF00230"] == pytest.approx(102.0)
    assert bits["TBDB001"] == pytest.approx(25.9)
    # The window with no TBDB001 hit gets None, not a floor.
    assert H.best_bits({m: grouped[m].get("w000001", []) for m in grouped})["TBDB001"] is None
    assert H.best_bits({m: grouped[m].get("w000003", []) for m in grouped}) == {
        "RF00230": None,
        "TBDB001": None,
    }


def test_parse_tblout_refuses_a_short_row_rather_than_dropping_it():
    with pytest.raises(H.HostScreenError):
        H.parse_tblout_hits("target - T-box RF00230 cm 1 224\n")


def test_parse_tblout_refuses_a_non_numeric_score():
    row = [line for line in _tblout("RF00230").splitlines() if not line.startswith("#")][0]
    fields = row.split()
    fields[14] = "not-a-score"
    with pytest.raises(H.HostScreenError):
        H.parse_tblout_hits(" ".join(fields) + "\n")


def test_content_refused_reads_the_score_and_never_the_inc_flag():
    """A hit marked ``?`` at 10 bits refuses; a hit marked ``!`` at 9.9 does not."""
    assert H.content_refused({"RF00230": 10.0}, False) is True
    assert H.content_refused({"RF00230": 9.9}, False) is False
    assert H.content_refused({"RF00230": None, "TBDB001": 12.0}, False) is True
    assert H.content_refused({"RF00230": None, "TBDB001": None}, False) is False
    # The motif alone refuses, with no CM hit at all.
    assert H.content_refused({"RF00230": None}, True) is True


def test_at_least_does_not_read_a_zero_bit_hit_as_no_hit():
    """``(bits or -inf) >= t`` is the trap: 0.0 is falsy, so a real zero-bit hit vanishes.

    The audit searches run at ``-T 0``, where a hit of exactly 0.0 bits is a reported row.
    """
    assert H.at_least(0.0, 0.0) is True
    assert H.at_least(0.0, 10.0) is False
    assert H.at_least(None, 0.0) is False
    assert H.at_least(-1.0, 0.0) is False
    assert H.at_least(10.0, 10.0) is True
    assert H.at_least(True, 1.0) is False  # a bool is not a bit score
    assert H.at_least(float("nan"), 0.0) is False
    assert H.content_refused({"RF00230": 0.0}, False, threshold=0.0) is True


def test_content_refused_does_not_read_a_bool_as_a_bit_score():
    """``float(True)`` is 1.0; at a threshold of 1.0 a bool must still not refuse."""
    assert H.content_refused({"RF00230": True}, False, threshold=1.0) is False


def test_the_real_tbox_fixtures_are_refused_and_the_real_host_window_is_not():
    """The content screen's verdict on real data, at the pinned 10.0 bits."""
    grouped = {
        model: H.hits_by_target(_tblout(model), model=model) for model in ("RF00230", "TBDB001")
    }
    panel = _panel()
    for name in ("w000001", "w000002"):
        bits = H.best_bits({m: grouped[m].get(name, []) for m in grouped})
        assert H.content_refused(bits, H.motif_present(panel[name])) is True
    host_bits = H.best_bits({m: grouped[m].get("w000003", []) for m in grouped})
    assert H.content_refused(host_bits, H.motif_present(panel["w000003"])) is False


# ══════════════════════════════════════════════════════════════════════════════════════
# The motif (A15 Pin 2's CM-free backstop)
# ══════════════════════════════════════════════════════════════════════════════════════
def test_motif_fires_on_a_real_catalogued_tbox_that_carries_the_consensus():
    panel, meta = _panel(), _panel_meta()
    assert meta["records"]["w000002"]["kind"] == "catalogued_tbox_with_motif"
    assert H.motif_present(panel["w000002"]) is True
    # The other real T-box does not carry an exact match — A15's "backstop, not a sensitive
    # test" in one fixture.
    assert H.motif_present(panel["w000001"]) is False
    assert H.motif_present(panel["w000003"]) is False


def test_motif_reads_both_strands():
    assert H.motif_present("TTT" + H.MOTIF.replace("N", "A") + "TTT") is True
    assert H.motif_present("TTT" + wd.reverse_complement(H.MOTIF.replace("N", "A")) + "TTT")


def test_motif_n_means_any_of_acgt_and_not_the_letter_n():
    assert H.motif_present("GGTGGCACCGCG") is True
    assert H.motif_present("GGTGGNACCGCG") is False


def test_motif_transcribes_before_matching():
    """Lower-case RNA must match: the screened DNA includes RNA-alphabet sources."""
    rna = H.MOTIF.replace("N", "A").replace("T", "U").lower()
    assert rna == "gguggaaccgcg"
    assert H.motif_present(rna) is True


# ══════════════════════════════════════════════════════════════════════════════════════
# The command line (A15 Pin 2's pinned invocation)
# ══════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model", ["RF00230", "TBDB001"])
def test_search_argv_mirrors_the_option_line_cmsearch_itself_wrote(model):
    """The mirror, pinned against the binary's own record in the committed fixture."""
    executed = H.option_settings_options(H.option_settings_line(_tblout(model)))
    mirrored = H.option_settings_options(
        " ".join(
            H.search_argv(
                cm=f"data/external/refs/{model}.cm",
                fasta="panel.fa",
                tblout=f"{model}.tblout",
                threshold_bits=10.0,
                cpu=2,
            )
        )
    )
    assert executed == mirrored
    assert executed["-T"] == "10.0"
    assert executed["-Z"] == "1.0"
    assert "--toponly" not in executed
    assert "--cut_ga" not in executed
    assert "--max" not in executed


def test_option_settings_options_is_order_insensitive_but_not_value_blind():
    """A sorted token list would make these two compare equal; a mapping does not."""
    assert H.option_settings_options("-T 10.0 -Z 1.0") == H.option_settings_options(
        "-Z 1.0 -T 10.0"
    )
    assert H.option_settings_options("-T 10.0 -Z 1.0") != H.option_settings_options(
        "-T 1.0 -Z 10.0"
    )


def test_option_settings_options_drops_paths_so_no_machine_layout_is_recorded():
    line = H.option_settings_line(_tblout("RF00230"))
    options = H.option_settings_options(line)
    assert all(not str(value).endswith(".cm") for value in options.values())
    assert all(not str(value).endswith(".fa") for value in options.values())
    assert options["--tblout"] is True


def test_the_committed_fixtures_carry_no_absolute_path():
    """A committed tool-output fixture leaks a username and a layout unless it is sanitised.

    Iterates the directory rather than a hand-written list, so a fixture added later is covered
    without anyone remembering to add it here.
    """
    for name in sorted(path.name for path in _FIXTURE.iterdir()):
        text = (_FIXTURE / name).read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "/exports/" not in text


def test_search_argv_adds_max_only_for_the_filter_off_arm():
    assert "--max" not in H.search_argv(cm="c", fasta="f", tblout="t", threshold_bits=10.0, cpu=1)
    assert "--max" in H.search_argv(
        cm="c", fasta="f", tblout="t", threshold_bits=10.0, cpu=1, no_filters=True
    )


def test_tblout_digest_ignores_the_header_and_not_the_rows():
    text = _tblout("RF00230")
    rehashed = text.replace("# Date:            (sanitised: committed fixture)", "# Date: other")
    assert H.tblout_digest(rehashed) == H.tblout_digest(text)
    dropped = "\n".join(line for line in text.splitlines() if not line.startswith("w000002"))
    assert H.tblout_digest(dropped) != H.tblout_digest(text)


# ══════════════════════════════════════════════════════════════════════════════════════
# truncate_window (the sensitivity panel's geometry)
# ══════════════════════════════════════════════════════════════════════════════════════
def test_truncate_window_drops_a_fraction_of_the_LOCUS_from_the_5_prime_end():
    sequence = "".join("ACGT"[i % 4] for i in range(100))
    # Locus at [20, 60): L = 40, so f = 0.25 starts at 20 + 10 = 30 and runs to the window end.
    assert H.truncate_window(sequence, 20, 60, 0.25) == sequence[30:]
    assert H.truncate_window(sequence, 20, 60, 0.50) == sequence[40:]
    assert H.truncate_window(sequence, 20, 60, 0.0) == sequence[20:]


def test_truncate_window_floors_rather_than_rounding():
    sequence = "A" * 100
    # L = 7, f = 0.25 -> floor(1.75) = 1.
    assert len(H.truncate_window(sequence, 10, 17, 0.25)) == len(sequence) - 11


def test_truncate_window_refuses_a_degenerate_locus_or_fraction():
    with pytest.raises(H.HostScreenError):
        H.truncate_window("A" * 50, 10, 10, 0.25)
    with pytest.raises(H.HostScreenError):
        H.truncate_window("A" * 50, 10, 20, 1.0)
    with pytest.raises(H.HostScreenError):
        H.truncate_window("A" * 50, 60, 70, 0.25)


# ══════════════════════════════════════════════════════════════════════════════════════
# screen_window and the must-fire controls
# ══════════════════════════════════════════════════════════════════════════════════════
def _screened_dna(**sequences: list[str]) -> H.ScreenedDna:
    """A :class:`ScreenedDna` over real sequence slices, one list per A15 source."""
    kmers = {name: H.sorted_unique_kmers(values) for name, values in sequences.items()}
    loci = {
        H.SOURCE_FOLD_CONTEXTS: masking.LocusIndex.from_records([("NZ_CP009247.1", 5000, 6000)]),
        H.SOURCE_MINED_WINDOWS: masking.LocusIndex.from_records([("OTHER", 1, 100)]),
    }
    return H.ScreenedDna(kmers=kmers, loci=loci, blocks={}, sequences=dict(sequences))


def _four_sources() -> H.ScreenedDna:
    panel = _panel()
    host = panel["w000003"]
    return _screened_dna(
        **{
            H.SOURCE_FOLD_CONTEXTS: [panel["w000001"]],
            H.SOURCE_MINED_WINDOWS: [panel["w000002"]],
            H.SOURCE_DECOY_INSERTS: [host[:300]],
            H.SOURCE_STAGE2: [host[700:]],
        }
    )


def test_screen_window_attributes_a_shared_kmer_to_the_source_that_carries_it():
    dna = _four_sources()
    panel = _panel()
    verdict = H.screen_window(panel["w000001"], "UNRELATED", 1, 1024, dna)
    assert verdict.rule_kmer is True
    assert verdict.rule_coord is False
    assert verdict.attributed_to == [H.SOURCE_FOLD_CONTEXTS]
    assert verdict.refused is True


def test_screen_window_passes_dna_no_source_carries():
    dna = _four_sources()
    clean = "".join("ACGTTGCA"[(i * 5 + i // 7) % 8] for i in range(1024))
    verdict = H.screen_window(clean, "UNRELATED", 1, 1024, dna)
    assert verdict.kmer_sources == dict.fromkeys(H.SCREENED_SOURCES, 0)
    assert verdict.refused is False


def test_screen_window_fires_rule_2_without_rule_1():
    """The coordinate-only mechanism: an overlap with no shared k-mer at all."""
    dna = _four_sources()
    clean = "".join("ACGTTGCA"[(i * 5 + i // 7) % 8] for i in range(1024))
    verdict = H.screen_window(clean, "NZ_CP009247.1", 5500, 5600, dna)
    assert verdict.rule_kmer is False
    assert verdict.rule_coord is True
    assert verdict.coord_sources == (H.SOURCE_FOLD_CONTEXTS,)
    assert verdict.refused is True


def test_screen_window_fires_rule_2_through_the_accession_version_normalisation():
    dna = _four_sources()
    clean = "".join("ACGTTGCA"[(i * 5 + i // 7) % 8] for i in range(1024))
    for accession in ("NZ_CP009247", "NZ_CP009247.1", "NZ_CP009247.7"):
        assert H.screen_window(clean, accession, 5500, 5600, dna).rule_coord is True


def test_screen_window_evaluates_both_rules_so_only_counts_are_measurable():
    """No short-circuit: a window refused by one rule still records the other."""
    dna = _four_sources()
    verdict = H.screen_window(_panel()["w000001"], "NZ_CP009247.1", 5500, 5600, dna)
    assert verdict.rule_kmer is True
    assert verdict.rule_coord is True


def test_control_segment_prefers_an_exclusive_segment_and_reports_when_it_cannot():
    panel = _panel()
    dna = _four_sources()
    segment, exclusive = H.control_segment(
        dna.sequences[H.SOURCE_FOLD_CONTEXTS], H.SOURCE_FOLD_CONTEXTS, dna, segment_nt=200
    )
    assert len(segment) == 200
    assert exclusive is True
    assert H.kmer_hits(H.canonical_kmers(segment), dna.kmers[H.SOURCE_FOLD_CONTEXTS]) > 0

    # A source wholly contained in another cannot offer an exclusive segment — a property of
    # the sources, which the helper reports rather than papers over.
    contained = _screened_dna(
        **{
            H.SOURCE_FOLD_CONTEXTS: [panel["w000003"]],
            H.SOURCE_MINED_WINDOWS: [panel["w000003"][:500]],
            H.SOURCE_DECOY_INSERTS: [panel["w000001"]],
            H.SOURCE_STAGE2: [panel["w000002"]],
        }
    )
    _segment, exclusive = H.control_segment(
        contained.sequences[H.SOURCE_MINED_WINDOWS], H.SOURCE_MINED_WINDOWS, contained
    )
    assert exclusive is False


def test_control_segment_is_none_when_the_source_index_is_dead():
    """The sabotage shape the per-source control must catch: an empty index."""
    panel = _panel()
    dna = _screened_dna(
        **{
            H.SOURCE_FOLD_CONTEXTS: [],
            H.SOURCE_MINED_WINDOWS: [panel["w000002"]],
            H.SOURCE_DECOY_INSERTS: [panel["w000003"][:300]],
            H.SOURCE_STAGE2: [panel["w000003"][700:]],
        }
    )
    assert H.control_segment([panel["w000001"]], H.SOURCE_FOLD_CONTEXTS, dna) is None


def _control_host() -> dict:
    """A Pin-1-clean host row built from DNA no source in :func:`_four_sources` carries."""
    return {
        "candidate_id": "test_host:lead",
        "sequence": "".join("ACGTTGCA"[(i * 5 + i // 7) % 8] for i in range(1024)),
        "accession": "UNRELATED",
        "locus_start": 1,
        "locus_end": 1024,
    }


def test_pin1_controls_fire_on_a_live_index_and_name_their_source():
    dna = _four_sources()
    controls, sequences = H.pin1_controls(dna=dna, host=_control_host(), coordinate_only=None)
    by_name = {entry["name"]: entry for entry in controls}
    assert by_name[H.CONTROL_HOST_CLEAN]["fired"] is True
    for source in H.SCREENED_SOURCES:
        entry = by_name[f"pin1_kmer_{source}"]
        assert entry["fired"] is True, entry
        assert entry["observed"]["contains_source"] is True
        assert f"pin1_kmer_{source}" in sequences
    # The coordinate-only control is absent from the pool here, and says so rather than passing.
    assert by_name[H.CONTROL_COORD_ONLY]["fired"] is False


def test_a_pin1_control_fails_when_its_own_source_index_is_dead():
    """Each source's control must be breakable ALONE, with the other three left live."""
    panel = _panel()
    dna = _four_sources()
    dead = H.ScreenedDna(
        kmers={**dna.kmers, H.SOURCE_STAGE2: np.empty(0, dtype=np.uint64)},
        loci=dna.loci,
        blocks={},
        sequences=dna.sequences,
    )
    # The segment is still drawn from the live sequence list, so only the INDEX is dead.
    segment = H.normalise_dna(panel["w000003"][700:900])
    spliced = "".join(
        [_control_host()["sequence"][:300], segment, _control_host()["sequence"][500:]]
    )
    verdict = H.screen_window(spliced, "UNRELATED", 1, 1024, dead)
    assert H.SOURCE_STAGE2 not in verdict.attributed_to
    live = H.screen_window(spliced, "UNRELATED", 1, 1024, dna)
    assert live.attributed_to == [H.SOURCE_STAGE2]


def test_pin1_controls_carry_the_accession_variants_when_the_candidate_is_present():
    dna = _four_sources()
    clean = "".join("ACGTTGCA"[(i * 5 + i // 7) % 8] for i in range(1024))
    verdicts = {
        "4ef2d573abc:lead": {
            **H.screen_window(clean, "NZ_CP009247.1", 5500, 5600, dna).as_json(),
            "accession": "NZ_CP009247.1",
            "locus_start": 5500,
            "locus_end": 5600,
        }
    }
    found = H.coordinate_only_candidate(verdicts, "4ef2d573")
    assert found["candidate_id"] == "4ef2d573abc:lead"
    assert set(found["accession_variants"]) == {"as_given", "stripped"}
    controls, _ = H.pin1_controls(dna=dna, host=_control_host(), coordinate_only=found)
    by_name = {entry["name"]: entry for entry in controls}
    assert by_name[H.CONTROL_COORD_ONLY]["fired"] is True
    variants = [name for name in by_name if name.startswith("pin1_accession_")]
    assert len(variants) == 2
    assert all(by_name[name]["fired"] is True for name in variants)


def test_coordinate_only_candidate_is_none_when_the_prefix_matches_nothing():
    assert H.coordinate_only_candidate({"abc": {}}, "4ef2d573") is None


def test_composition_buckets_sum_to_n_and_name_a_null_role():
    """A15 Pin 7(e)'s table. A null role becomes the token ``"null"``, never a ``None`` bucket."""
    records = [
        {
            "side": "lead",
            "parent_nested_role": "heldout",
            "parent_is_designated_loo_holdout": "true",
            "parent_resolved_order": "Bacillales",
        },
        {
            "side": "trail",
            "parent_nested_role": "heldout",
            "parent_is_designated_loo_holdout": "false",
            "parent_resolved_order": "Bacillales",
        },
        {
            "side": "trail",
            "parent_nested_role": "null",
            "parent_is_designated_loo_holdout": "null",
            "parent_resolved_order": H.ABSENT_PARENT,
        },
        {"side": "trail"},
    ]
    block = H._composition(records)
    assert block["n"] == 4
    assert block["side"] == {"lead": 1, "trail": 3}
    assert block["parent_nested_role"] == {"heldout": 2, "null": 1, "unresolved": 1}
    assert block["parent_is_designated_loo_holdout"] == {
        "false": 1,
        "null": 1,
        "true": 1,
        "unresolved": 1,
    }
    assert block["parent_resolved_order"] == {
        H.ABSENT_PARENT: 1,
        "Bacillales": 2,
        "unresolved": 1,
    }
    for axis in H.COMPOSITION_AXES:
        assert sum(block[axis].values()) == block["n"]
    assert H._composition([]) == {"n": 0, **{axis: {} for axis in H.COMPOSITION_AXES}}


def test_parent_fields_label_an_absent_parent_on_every_axis_and_keep_a_present_one():
    """A parent the split table lacks is a labelled bucket on all three axes, never dropped."""
    meta = {
        "rec_1": {
            "parent_nested_role": "heldout",
            "parent_is_designated_loo_holdout": "true",
            "parent_resolved_order": "Lactobacillales",
        }
    }
    assert H._parent_fields(meta, "rec_1") == meta["rec_1"]
    assert H._parent_fields(meta, "rec_missing") == {
        f"parent_{field}": H.ABSENT_PARENT for field in H.COMPOSITION_PARENT_FIELDS
    }
    assert H._composition_label(None) == "null"
    assert H._composition_label(float("nan")) == "null"
    assert H._composition_label(True) == "true"
    assert H._composition_label(np.bool_(False)) == "false"
    assert H._composition_label("Bacillales") == "Bacillales"


@pytest.mark.parametrize("axis", ["parent_is_designated_loo_holdout", "parent_resolved_order"])
def test_a_composition_axis_that_does_not_sum_to_its_n_is_refused(axis):
    """Every Pin 7(e) axis must sum to the window count, not only side and nested_role."""
    report = _synthetic_report()
    assert not [p for p in H.validate_report(report) if f".{axis} sums to" in p]
    forged = copy.deepcopy(report)
    forged["composition"]["admitted"][axis] = {"null": 2}
    assert f"composition.admitted.{axis} sums to 2, not 3" in H.validate_report(forged)


def test_an_output_is_recorded_against_the_code_checkout_before_the_data_root(tmp_path):
    """A worktree run reading another checkout's data must still record ``reports/p3/...``."""
    data_root = H.CODE_ROOT.parents[2]  # the main checkout a worktree is nested under
    inside_code = H.CODE_ROOT / "reports/p3/host_screen_admitted.txt"
    assert H.recorded_path(inside_code, data_root) == "reports/p3/host_screen_admitted.txt"
    assert H.recorded_path(inside_code, H.CODE_ROOT) == "reports/p3/host_screen_admitted.txt"
    only_data = data_root / "data/processed/negatives/mining_pool_v0.parquet"
    assert H.recorded_path(only_data, data_root) == (
        "data/processed/negatives/mining_pool_v0.parquet"
    )
    outside = tmp_path / "scratch" / "host_screen.json"
    assert H.recorded_path(outside, data_root) == "host_screen.json"


def test_a_v1_audit_composition_that_is_not_its_hosts_table_is_refused():
    """The v1 column D reads beside v2's must re-derive from the per-host entries."""
    report = _synthetic_report()
    assert not [p for p in H.validate_report(report) if "v1_audit.composition" in p]
    forged = copy.deepcopy(report)
    forged["v1_audit"]["composition"]["parent_resolved_order"] = {"Bacillales": 2}
    assert "v1_audit.composition is not the table its own hosts give" in H.validate_report(forged)


def test_control_entry_derives_fired_rather_than_taking_it():
    entry = H.control_entry("x", "m", {"a": 1}, {"a": 1})
    assert entry["fired"] is True
    assert H.control_entry("x", "m", {"a": 1}, {"a": 2})["fired"] is False


# ══════════════════════════════════════════════════════════════════════════════════════
# decoy_embedded (A15 Pin 7(b)'s insert set)
# ══════════════════════════════════════════════════════════════════════════════════════
def test_decoy_embedded_mirrors_a7s_rule_including_the_pandas_spellings_of_missing():
    fold = {"parent_in_fold"}
    assert H.decoy_embedded(
        {"pool": "structured_rna", "masked": False, "source_record_id": None}, fold
    )
    assert H.decoy_embedded(
        {"pool": "dinuc_shuffled", "masked": False, "source_record_id": "parent_in_fold"}, fold
    )
    assert not H.decoy_embedded(
        {"pool": "dinuc_shuffled", "masked": False, "source_record_id": "held_out"}, fold
    )
    assert not H.decoy_embedded(
        {"pool": "gc_background", "masked": False, "source_record_id": None}, fold
    )
    assert not H.decoy_embedded(
        {"pool": "structured_rna", "masked": True, "source_record_id": None}, fold
    )
    # "nan"/"None" as a string is MISSING, not a parent id: str(x or "") would admit it.
    assert H.decoy_embedded(
        {"pool": "structured_rna", "masked": False, "source_record_id": float("nan")}, fold
    )


def test_expected_embedded_matches_the_minters_committed_counts():
    assert H.EXPECTED_EMBEDDED == {
        "production": {"dinuc_shuffled": 702, "structured_rna": 2999},
        "twin": {"dinuc_shuffled": 602, "structured_rna": 2999},
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════════════
def test_require_repo_relative_refuses_absolute_and_climbing_paths():
    assert H.require_repo_relative("data/x.parquet", what="--x") == "data/x.parquet"
    for bad in ("/data/x.parquet", "../x.parquet", "a/../../x"):
        with pytest.raises(H.HostScreenError):
            H.require_repo_relative(bad, what="--x")


def test_provenance_record_keys_every_input_repo_relatively(tmp_path):
    """An absolute path in a committed artifact publishes this machine's layout."""
    root = tmp_path / "repo"
    (root / "data/processed").mkdir(parents=True)
    data = root / "data/processed/pool.parquet"
    data.write_bytes(b"not really a parquet")
    scratch = tmp_path / "scratch/prep.json"
    scratch.parent.mkdir(parents=True)
    scratch.write_text("{}", encoding="utf-8")
    record = H.provenance_record(leg="prep", inputs=[data, scratch], root=root)
    assert set(record["inputs"]) == {"data/processed/pool.parquet", "prep.json"}
    assert all(not Path(key).is_absolute() for key in record["inputs"])
    assert (
        record["inputs"]["data/processed/pool.parquet"]
        == hashlib.sha256(b"not really a parquet").hexdigest()
    )
    assert record["extra"]["leg"] == "prep"
    assert record["adr"] == H.ADR
    assert record["seed"] == 42
    for field in ("rule", "script", "git_sha", "env_lock_hash", "seed", "inputs", "outputs"):
        assert field in record


def test_recorded_path_is_relative_inside_the_root_and_a_basename_outside(tmp_path):
    root = tmp_path / "repo"
    (root / "reports").mkdir(parents=True)
    inside_path = root / "reports/host_screen.json"
    assert H.recorded_path(inside_path, root) == "reports/host_screen.json"
    outside = tmp_path / "scratch/host_screen.json"
    assert H.recorded_path(outside, root) == "host_screen.json"
    assert H.inside(inside_path, root) is True
    assert H.inside(outside, root) is False


# ══════════════════════════════════════════════════════════════════════════════════════
# validate_report — the screen report
# ══════════════════════════════════════════════════════════════════════════════════════
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_FAKE_SHA = "0" * 64


def _synthetic_report(tmp_path: Path | None = None) -> dict:
    """A self-consistent synthetic screen report. **Every number here is invented.**

    Synthetic so the clause set can be broken one field at a time; the real screen's numbers are
    in ``reports/p3/host_screen.json``. The arithmetic is kept consistent everywhere, because a
    forge that is inconsistent in two places cannot show which clause bit
    ([[all-true-fixture-cannot-test-a-conjunction]]).
    """
    admitted_ids = ["cand_a:lead", "cand_b:trail", "cand_c:lead"]
    admitted_text = "\n".join(admitted_ids) + "\n"
    argv = {}
    options = {}
    for fasta in H.SCREEN_FASTAS:
        for model, cm in H.SCREEN_CMS:
            tokens = H.search_argv(
                cm=cm,
                fasta=fasta,
                tblout="t.tblout",
                threshold_bits=H.FASTA_THRESHOLDS[fasta],
                cpu=4,
            )
            argv[f"{fasta}:{model}"] = tokens
            rendered = H.option_settings_options(" ".join(tokens))
            options[f"{fasta}:{model}"] = " ".join(
                flag if value is True else f"{flag} {value}"
                for flag, value in sorted(rendered.items())
            )
    controls = [
        H.control_entry(H.CONTROL_HOST_CLEAN, "m", {"rule_kmer": False}, {"rule_kmer": False}),
        *(H.control_entry(f"pin1_kmer_{source}", "m", True, True) for source in H.SCREENED_SOURCES),
        H.control_entry(H.CONTROL_COORD_ONLY, "m", True, True),
        H.control_entry("pin1_accession_as_given", "m", True, True),
        H.control_entry(H.CONTROL_CONTENT, "m", True, True),
        H.control_entry(H.CONTROL_CONTENT_HOST, "m", False, False),
    ]
    hosts = {
        "host_1:lead": {
            "bits": {"RF00230": 120.0, "TBDB001": 30.0},
            "motif": True,
            "refused": True,
            "refused_fold": True,
            "refused_content": True,
            "side": "lead",
            "parent_nested_role": "heldout",
            "parent_is_designated_loo_holdout": "true",
            "parent_resolved_order": "Bacillales",
        },
        "host_2:trail": {
            "bits": {"RF00230": 2.0, "TBDB001": None},
            "motif": False,
            "refused": False,
            "refused_fold": False,
            "refused_content": False,
            "side": "trail",
            "parent_nested_role": "null",
            "parent_is_designated_loo_holdout": "null",
            "parent_resolved_order": H.ABSENT_PARENT,
        },
    }
    report = {
        "schema_version": H.SCHEMA_VERSION,
        "step": H.STEP,
        "kind": H.KIND_SCREEN,
        "generated_by": H.GENERATED_BY,
        "adr": H.ADR,
        "pool": {
            "source": "data/x.parquet",
            "sha256": _FAKE_SHA,
            "n_admitted": 10,
            "n_candidates": 6,
        },
        "screened_dna": {
            "k": H.KMER_K,
            "alphabet_rule": H.ALPHABET_RULE,
            "sources": {
                H.SOURCE_FOLD_CONTEXTS: {"n_records": 3, "n_kmers": 100},
                H.SOURCE_MINED_WINDOWS: {"n": 4, "n_kmers": 200},
                H.SOURCE_DECOY_INSERTS: {"n": 5, "n_kmers": 300, "sha256": _FAKE_SHA},
                H.SOURCE_STAGE2: {
                    "n": H.STAGE2_EXPECTED_ADMITTED,
                    "n_kmers": 400,
                    "predicate": H.STAGE2_PREDICATE,
                    "provenance_n_admitted": H.STAGE2_EXPECTED_ADMITTED,
                    "stage2_dataset_sha256": _FAKE_SHA,
                },
            },
        },
        "fold_screen": {
            "n_refused_kmer": 2,
            "n_refused_coord": 1,
            "n_refused_kmer_only": 1,
            "n_refused_coord_only": 0,
            "n_refused": 2,
            "n_survivors": 4,
            "per_source_windows": {name: {"kmer": 1, "coord": 0} for name in H.SCREENED_SOURCES},
        },
        "content_screen": {
            "threshold_bits": H.CONTENT_THRESHOLD_BITS,
            "motif": H.MOTIF,
            "models": {model: {"cm": cm, "cm_sha256": _FAKE_SHA} for model, cm in H.SCREEN_CMS},
            "argv": argv,
            "option_settings": options,
            "infernal_version": "INFERNAL 1.1.5 (Sep 2023)",
            "env_lock": H.ENV_LOCK,
            "env_lock_sha256": _FAKE_SHA,
            "n_searched": 4,
            "n_refused_cm": {
                "RF00230": 1,
                "TBDB001": 0,
                "either": 1,
                "marginal_RF00230": 1,
                "marginal_TBDB001": 0,
            },
            "n_refused_motif": 1,
            "n_refused_motif_only": 0,
            "n_refused": 1,
            "n_admitted": 3,
        },
        "admitted": {
            "n": 3,
            "list": "reports/p3/host_screen_admitted.txt",
            "sha256": _sha(admitted_text),
        },
        "refused_annex": {
            "path": "reports/p3/host_screen_refused.json",
            "sha256": _FAKE_SHA,
            "n": 3,
        },
        "controls": controls,
        "controls_all_fired": True,
        "sensitivity_panel": {
            "threshold_bits": H.CONTENT_THRESHOLD_BITS,
            "variants": {
                variant: {
                    "n": 4,
                    "RF00230": {"n_hits": 3, "recall": 0.75},
                    "TBDB001": {"n_hits": 1, "recall": 0.25},
                    "motif": {"n_hits": 2, "recall": 0.5},
                    "either": {"n_hits": 3, "recall": 0.75},
                }
                for variant in ("full", "t25", "t50")
            },
        },
        "v1_audit": {
            "bit_model": "RF00230",
            "threshold_bits": H.CONTENT_THRESHOLD_BITS,
            "counts": {
                "n_hosts": 2,
                "n_ge_93": 1,
                "n_ge_20": 1,
                "n_ge_10": 1,
                "n_motif": 1,
                "n_refused_fold": 1,
                "n_refused_content": 1,
                "n_refused": 1,
            },
            "composition": H._composition(list(hosts.values())),
            "hosts": hosts,
        },
        "composition": {
            "admitted": {
                "n": 3,
                "side": {"lead": 2, "trail": 1},
                "parent_nested_role": {"null": 3},
                "parent_is_designated_loo_holdout": {"null": 3},
                "parent_resolved_order": {"null": 3},
            },
            "refused": {
                "n": 3,
                "side": {"lead": 1, "trail": 2},
                "parent_nested_role": {"null": 3},
                "parent_is_designated_loo_holdout": {"false": 1, "null": 2},
                "parent_resolved_order": {"Bacillales": 1, "null": 2},
            },
        },
        "problems": [],
    }
    if tmp_path is not None:
        path = tmp_path / "reports/p3/host_screen_admitted.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(admitted_text, encoding="utf-8")
    return report


def test_the_synthetic_screen_report_validates_clean():
    assert H.validate_report(_synthetic_report()) == []


def test_validate_report_binds_the_admitted_list_file_when_a_root_is_given(tmp_path):
    report = _synthetic_report(tmp_path)
    assert H.validate_report(report, root=tmp_path) == []
    (tmp_path / "reports/p3/host_screen_admitted.txt").write_text("other\n", encoding="utf-8")
    problems = H.validate_report(report, root=tmp_path)
    assert any("sha256 does not match" in problem for problem in problems)
    # Without a root the validator touches no file and the clause is simply not asserted.
    assert H.validate_report(report) == []


def test_validate_report_refuses_an_unknown_kind():
    report = _synthetic_report()
    report["kind"] = "something_else"
    assert any("unknown report kind" in problem for problem in H.validate_report(report))


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda r: r.__setitem__("schema_version", "9.9"), "schema_version"),
        (lambda r: r.__setitem__("step", "P3-16"), "step"),
        (lambda r: r.__setitem__("adr", "ADR-0005 D7"), "adr"),
        (lambda r: r["screened_dna"].__setitem__("k", 31), "screened_dna.k"),
        (
            lambda r: r["screened_dna"].__setitem__("alphabet_rule", "upper-case only"),
            "alphabet_rule",
        ),
        (
            lambda r: r["content_screen"].__setitem__("threshold_bits", 20.0),
            "threshold_bits",
        ),
        (lambda r: r["content_screen"].__setitem__("motif", "GGTGGAACCGCG"), "motif"),
        (
            lambda r: r["screened_dna"]["sources"][H.SOURCE_STAGE2].__setitem__("n", 9058),
            f"sources.{H.SOURCE_STAGE2}.n",
        ),
        (
            lambda r: r["screened_dna"]["sources"][H.SOURCE_STAGE2].__setitem__(
                "provenance_n_admitted", 9060
            ),
            "provenance count",
        ),
        (
            lambda r: r["screened_dna"]["sources"][H.SOURCE_DECOY_INSERTS].__setitem__(
                "n_kmers", 0
            ),
            "n_kmers is not positive",
        ),
        (
            lambda r: r["screened_dna"]["sources"].pop(H.SOURCE_MINED_WINDOWS),
            f"sources.{H.SOURCE_MINED_WINDOWS} is missing",
        ),
        (lambda r: r["fold_screen"].__setitem__("n_refused_kmer", 0), "refused nothing"),
        (lambda r: r["fold_screen"].__setitem__("n_survivors", 5), "n_survivors + n_refused"),
        (
            lambda r: r["fold_screen"].__setitem__("n_refused_kmer_only", 2),
            "n_refused != n_refused_coord + n_refused_kmer_only",
        ),
        (
            lambda r: r["fold_screen"]["per_source_windows"].pop(H.SOURCE_STAGE2),
            f"per_source_windows.{H.SOURCE_STAGE2} is missing",
        ),
        (
            lambda r: r["content_screen"].__setitem__("n_searched", 3),
            "n_searched != fold_screen.n_survivors",
        ),
        (
            lambda r: r["content_screen"].__setitem__("n_admitted", 2),
            "admitted.n != content_screen.n_admitted",
        ),
        (
            lambda r: r["content_screen"]["n_refused_cm"].__setitem__("marginal_TBDB001", 1),
            "marginal_TBDB001",
        ),
        (
            lambda r: r["content_screen"]["n_refused_cm"].__setitem__("RF00230", 2),
            "exceeds n_refused_cm.either",
        ),
        (
            lambda r: r["content_screen"].__setitem__("infernal_version", "INFERNAL 1.1.4"),
            "infernal_version",
        ),
        (lambda r: r["content_screen"].__setitem__("env_lock", "envs/ml.yml"), "env_lock"),
        (
            lambda r: r["content_screen"].__setitem__("env_lock_sha256", "nope"),
            "env_lock_sha256",
        ),
        (
            lambda r: r["content_screen"]["models"]["RF00230"].__setitem__("cm_sha256", "nope"),
            "cm_sha256",
        ),
        (lambda r: r["refused_annex"].__setitem__("n", 2), "refused_annex.n"),
        (lambda r: r["composition"]["admitted"].__setitem__("n", 4), "composition.admitted.n"),
        (
            lambda r: r["composition"]["refused"].__setitem__("side", {"lead": 1}),
            "composition.refused.side sums",
        ),
        (lambda r: r["v1_audit"]["counts"].__setitem__("n_ge_20", 2), "v1_audit.counts.n_ge_20"),
        (lambda r: r["v1_audit"]["counts"].__setitem__("n_hosts", 3), "n_hosts"),
        (
            lambda r: r["sensitivity_panel"]["variants"]["full"]["motif"].__setitem__(
                "recall", 1.0
            ),
            "motif.recall is not n_hits / n",
        ),
        (
            lambda r: r["sensitivity_panel"]["variants"].pop("t50"),
            "variants has no 't50' row",
        ),
        (
            lambda r: r["sensitivity_panel"].__setitem__("threshold_bits", 20.0),
            "sensitivity_panel.threshold_bits",
        ),
    ],
)
def test_each_screen_clause_is_breakable_alone(mutate, expected):
    report = _synthetic_report()
    assert H.validate_report(report) == []
    mutate(report)
    problems = H.validate_report(report)
    assert any(expected in problem for problem in problems), (expected, problems)


@pytest.mark.parametrize(
    "flag,expected",
    [
        ("--toponly", "passes --toponly"),
        ("--cut_ga", "passes --cut_ga"),
        ("--max", "passes --max"),
    ],
)
def test_a_forbidden_cmsearch_flag_is_caught_in_the_argv_and_in_the_executed_line(flag, expected):
    for field in ("argv", "option_settings"):
        report = _synthetic_report()
        key = f"{H.FASTA_SURVIVORS}:RF00230"
        if field == "argv":
            report["content_screen"]["argv"][key] = [*report["content_screen"]["argv"][key], flag]
        else:
            report["content_screen"]["option_settings"][key] += f" {flag}"
        problems = H.validate_report(report)
        assert any(expected in problem and field in problem for problem in problems), problems


@pytest.mark.parametrize("flag,value", [("-T", "20.0"), ("-Z", "2.0")])
def test_a_wrong_threshold_or_search_space_is_caught_on_both_records(flag, value):
    report = _synthetic_report()
    key = f"{H.FASTA_SURVIVORS}:TBDB001"
    tokens = list(report["content_screen"]["argv"][key])
    tokens[tokens.index(flag) + 1] = value
    report["content_screen"]["argv"][key] = tokens
    problems = H.validate_report(report)
    assert any(f"does not pass {flag}" in problem for problem in problems), problems


def test_a_missing_searched_fasta_is_caught():
    report = _synthetic_report()
    report["content_screen"]["argv"].pop(f"{H.FASTA_PANEL}:RF00230")
    assert any(
        f"no entry for {H.FASTA_PANEL}:RF00230" in problem for problem in H.validate_report(report)
    )


def test_the_audit_fastas_are_required_to_be_searched_at_zero_bits():
    """The v1/panel searches record full distributions; at -T 10 they would be censored."""
    report = _synthetic_report()
    key = f"{H.FASTA_V1_HOSTS}:RF00230"
    tokens = list(report["content_screen"]["argv"][key])
    tokens[tokens.index("-T") + 1] = repr(H.CONTENT_THRESHOLD_BITS)
    report["content_screen"]["argv"][key] = tokens
    assert any("does not pass -T 0.0" in problem for problem in H.validate_report(report))


@pytest.mark.parametrize(
    "name",
    [
        H.CONTROL_HOST_CLEAN,
        H.CONTROL_COORD_ONLY,
        H.CONTROL_CONTENT,
        H.CONTROL_CONTENT_HOST,
        *(f"pin1_kmer_{source}" for source in H.SCREENED_SOURCES),
    ],
)
def test_a_missing_must_fire_control_is_caught_by_name(name):
    report = _synthetic_report()
    report["controls"] = [entry for entry in report["controls"] if entry["name"] != name]
    problems = H.validate_report(report)
    assert any(name in problem for problem in problems), problems


def test_the_other_pin1_identity_is_breakable_alone():
    """``n_refused == n_refused_kmer + n_refused_coord_only`` — the identity whose counterpart
    the parametrized case above breaks. Both must hold, or the per-rule attribution could be
    scrambled behind a correct total."""
    report = _synthetic_report()
    fold = report["fold_screen"]
    # coord_only 0 -> 1, with coord raised to keep `coord >= coord_only` satisfied, so the only
    # clause left standing on is the kmer + coord_only identity.
    fold["n_refused_coord_only"] = 1
    fold["n_refused_coord"] = 2
    problems = H.validate_report(report)
    assert any(
        "n_refused != n_refused_kmer + n_refused_coord_only" in problem for problem in problems
    ), problems


def test_a_screen_that_left_no_pool_is_refused():
    report = _synthetic_report()
    report["fold_screen"].update(
        {
            "n_survivors": 0,
            "n_refused": 6,
            "n_refused_kmer": 6,
            "n_refused_kmer_only": 5,
            "n_refused_coord": 1,
            "n_refused_coord_only": 0,
        }
    )
    assert any(
        "there is no screened pool to draw from" in problem for problem in H.validate_report(report)
    )


def test_the_content_screens_own_total_is_breakable_alone():
    report = _synthetic_report()
    # Admitted and refused must account for everything searched; both places that carry the
    # admitted count are moved together, so only the sum clause is left to fire.
    report["content_screen"]["n_admitted"] = 2
    report["admitted"]["n"] = 2
    problems = H.validate_report(report)
    assert any("n_admitted + n_refused != n_searched" in problem for problem in problems), problems


def test_a_motif_only_count_above_the_motif_count_is_caught():
    report = _synthetic_report()
    report["content_screen"]["n_refused_motif_only"] = 2
    report["content_screen"]["n_refused"] = 3  # keep the total identity satisfied
    report["content_screen"]["n_admitted"] = 1
    report["admitted"]["n"] = 1
    problems = H.validate_report(report)
    assert any("n_refused_motif_only exceeds n_refused_motif" in problem for problem in problems)


def test_the_v1_audit_refusal_count_is_rederived_from_its_hosts():
    report = _synthetic_report()
    report["v1_audit"]["counts"]["n_refused"] = 2
    assert any(
        "v1_audit.counts.n_refused is not the 1" in problem for problem in H.validate_report(report)
    )


def test_a_missing_accession_control_is_caught():
    report = _synthetic_report()
    report["controls"] = [
        entry for entry in report["controls"] if not entry["name"].startswith("pin1_accession_")
    ]
    assert any(
        "accession-normalisation control" in problem for problem in H.validate_report(report)
    )


def test_a_control_that_did_not_fire_refuses_the_report():
    report = _synthetic_report()
    report["controls"][1]["observed"] = "something else"
    report["controls"][1]["fired"] = False
    report["controls_all_fired"] = False
    problems = H.validate_report(report)
    assert any("did not fire" in problem for problem in problems), problems


def test_controls_all_fired_cannot_be_asserted_against_its_own_controls():
    """The flag is re-derived; a TRUE beside a failed control is itself a problem."""
    report = _synthetic_report()
    report["controls"][1]["fired"] = False
    report["controls"][1]["observed"] = "no"
    assert report["controls_all_fired"] is True
    problems = H.validate_report(report)
    assert any("controls_all_fired disagrees" in problem for problem in problems), problems


def test_a_fired_flag_inconsistent_with_its_own_measurement_is_caught():
    report = _synthetic_report()
    report["controls"][0]["observed"] = {"rule_kmer": True}
    assert any(
        "fired is not observed == expected" in problem for problem in H.validate_report(report)
    )


def test_validate_report_never_raises_on_a_malformed_payload():
    for payload in (
        {"kind": H.KIND_SCREEN},
        {"kind": H.KIND_SCREEN, "fold_screen": "not a mapping", "controls": "x"},
        {"kind": H.KIND_POSTMINT},
        {"kind": H.KIND_POSTMINT, "violations": "x", "exempt": 5, "arms": "y"},
    ):
        assert isinstance(H.validate_report(payload), list)


def test_validate_report_does_not_read_the_problems_field_it_produces():
    report = _synthetic_report()
    report["problems"] = ["a stale problem from an earlier run"]
    assert H.validate_report(report) == []


# ══════════════════════════════════════════════════════════════════════════════════════
# The post-mint assertion (A15 Pin 3)
# ══════════════════════════════════════════════════════════════════════════════════════
def _postmint_search(arm: str, model: str, fasta: str) -> dict:
    tokens = H.search_argv(
        cm=dict(H.SCREEN_CMS)[model],
        fasta=fasta,
        tblout="t.tblout",
        threshold_bits=H.CONTENT_THRESHOLD_BITS,
        cpu=4,
        no_filters=arm == "filter_off",
    )
    rendered = H.option_settings_options(" ".join(tokens))
    return {
        "fasta": fasta,
        "model": model,
        "arm": arm,
        "tblout": f"{arm}/{model}/{Path(fasta).stem}.tblout",
        "argv": tokens,
        "option_settings": " ".join(
            flag if value is True else f"{flag} {value}" for flag, value in sorted(rendered.items())
        ),
    }


#: Per model, the query accession and CM length a real tblout row carries. Taken from the
#: committed tblouts: RF00230 is ``T-box RF00230 cm 1 224``, TBDB001 is ``seed_ILE - cm 1 233``.
#: TBDB001 has no Rfam accession, so a fixture that hardcoded ``RF00230`` in that column would
#: read as a bug to a reviewer and would not match the bytes the real search produces.
_CM_ACCESSION: dict[str, str] = {"RF00230": "RF00230", "TBDB001": "-"}
_CM_LENGTH: dict[str, int] = {"RF00230": 224, "TBDB001": 233}


def _tblout_row(target: str, model: str, seq_from: int, seq_to: int, score: float) -> str:
    query = H.CM_QUERY_NAMES[model]
    accession, length = _CM_ACCESSION[model], _CM_LENGTH[model]
    return (
        f"{target} - {query} {accession} cm 1 {length} {seq_from} {seq_to} + no 1 0.40 0.0 "
        f"{score} 1.0e-10 ! -"
    )


def _postmint_inputs(score: float, span: tuple[int, int], *, pool: str = "leader_decoy") -> tuple:
    contigs = {
        "v000001": {
            "contig_id": "dec:leader",
            "label": 0,
            "pool": pool,
            "splice_phase": 331,
            "locus_length": 180,
            "motif": False,
        },
        "v000002": {
            "contig_id": "pos:real",
            "label": 1,
            "pool": "corpus",
            "splice_phase": None,
            "locus_length": 243,
            "motif": True,
        },
    }
    # The FULL coverage A15 Pin 3 requires — every D2 model over every FASTA each arm covers.
    # This fixture used to build RF00230 alone, which is exactly the gap the coverage clause now
    # closes: with TBDB001 absent the suite could not tell a two-model assertion from a
    # one-model one, and a report with every TBDB001 search deleted validated clean.
    searches = [
        _postmint_search(arm, model, fasta)
        for arm, fastas in sorted(H.REQUIRED_POSTMINT_SEARCHES.items())
        for fasta in fastas
        for model, _cm in H.SCREEN_CMS
    ]
    hit = f"screen/RF00230/{Path(H.FASTA_POSTMINT_ALL).stem}.tblout"
    text = "\n".join([_tblout_row("v000001", "RF00230", span[0], span[1], score)]) + "\n"
    tblouts = {entry["tblout"]: (text if entry["tblout"] == hit else "") for entry in searches}
    for entry in searches:
        entry["data_sha256"] = H.tblout_digest(tblouts[entry["tblout"]])
    return contigs, searches, tblouts


def _postmint_both_models(
    hits: Sequence[tuple[str, float, tuple[int, int]]],
    *,
    pool: str = "dinuc_shuffled",
) -> tuple:
    """A label-0 contig carrying ``hits`` as ``(model, score, span)``, with BOTH models searched.

    ``_postmint_inputs`` searches RF00230 only, so before this helper existed TBDB001 appeared in
    no post-mint tblout, arm, violation or exemption anywhere in the suite — and a per-model
    threshold applied to only one of the two models left the whole file green. Every search is
    recorded for both models and both arms, so the arm clauses stay satisfied while the verdict
    is driven purely by which model's row carries which score.
    """
    contigs = {
        "v000001": {
            "contig_id": "dec:synthetic",
            "label": 0,
            "pool": pool,
            "splice_phase": 331,
            "locus_length": 180,
            "motif": False,
        },
    }
    searches = [
        _postmint_search(arm, model, fasta)
        for arm, fasta in (
            ("screen", H.FASTA_POSTMINT_ALL),
            ("filter_off", H.FASTA_POSTMINT_LABEL0),
        )
        for model, _ in H.SCREEN_CMS
    ]
    rows: dict[str, list[str]] = {entry["tblout"]: [] for entry in searches}
    stem = Path(H.FASTA_POSTMINT_ALL).stem
    screen = {model: f"screen/{model}/{stem}.tblout" for model, _ in H.SCREEN_CMS}
    for model, score, span in hits:
        rows[screen[model]].append(_tblout_row("v000001", model, span[0], span[1], score))
    tblouts = {key: ("\n".join(value) + "\n" if value else "") for key, value in rows.items()}
    for entry in searches:
        entry["data_sha256"] = H.tblout_digest(tblouts[entry["tblout"]])
    return contigs, searches, tblouts


def test_the_postmint_verdict_is_keyed_on_the_HITTING_MODEL_not_one_shared_number():
    """A15 Pin 3 Correction 2: the same score is noise on RF00230 and a violation on TBDB001.

    One fixture, one contig, one score, two models — so the only thing that can produce a
    different disposition is the per-model lookup. A change applied to one model and forgotten
    on the other fails here, which is the gap the single-model fixtures left open.
    """
    contigs, searches, tblouts = _postmint_both_models(
        [("RF00230", 15.0, (100, 200)), ("TBDB001", 15.0, (100, 200))]
    )
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert [entry["model"] for entry in findings["violations"]] == ["TBDB001"]
    assert [entry["model"] for entry in findings["noise_residue"]] == ["RF00230"]


@pytest.mark.parametrize(
    "model,score,violation",
    [
        # The pinned sense is `>=`, so a hit sitting EXACTLY on its threshold is a violation.
        # No pre-existing fixture scored at a boundary, so a `<=`/`<` slip was invisible.
        ("RF00230", 20.0, True),
        ("RF00230", 19.9, False),
        ("RF00230", 16.8, False),  # the measured matched-null maximum itself
        ("TBDB001", 12.0, True),
        ("TBDB001", 11.9, False),  # the measured TBDB001 matched-null maximum itself
        ("TBDB001", 10.0, False),  # inside its floor since Correction 2's amendment
        ("TBDB001", 20.0, True),
    ],
)
def test_each_models_threshold_boundary_is_exact(model, score, violation):
    contigs, searches, tblouts = _postmint_both_models([(model, score, (100, 200))])
    findings = H.postmint_findings(contigs, searches, tblouts)
    if violation:
        assert len(findings["violations"]) == 1, findings
        assert findings["noise_residue"] == []
    else:
        assert findings["violations"] == []
        assert len(findings["noise_residue"]) == 1, findings


def test_a_sub_threshold_hit_is_disclosed_with_the_threshold_that_tolerated_it():
    """The residue is published, not dropped: the band is what the re-sign now tolerates."""
    contigs, searches, tblouts = _postmint_both_models([("RF00230", 14.5, (357, 427))])
    findings = H.postmint_findings(contigs, searches, tblouts)
    (entry,) = findings["noise_residue"]
    assert entry["bits"] == 14.5
    assert entry["assertion_threshold_bits"] == H.POSTMINT_THRESHOLD_BITS["RF00230"]
    assert entry["label"] == 0 and entry["contig_id"] == "dec:synthetic"


def test_a_sub_threshold_leader_hit_is_residue_rather_than_exempt():
    """Sub-threshold is tested before the exemption, so `n_exempt` tracks interval arithmetic
    alone and does not move when a threshold moves."""
    contigs, searches, tblouts = _postmint_both_models(
        [("RF00230", 12.0, (332, 511))], pool="leader_decoy"
    )
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert findings["exempt"] == []
    assert len(findings["noise_residue"]) == 1


def test_a_leader_hit_wholly_inside_its_own_insert_is_exempt_not_a_violation():
    # The insert is [331, 511) 0-based; a 1-based inclusive hit at 332..511 is wholly inside.
    contigs, searches, tblouts = _postmint_inputs(50.0, (332, 511))
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert findings["violations"] == []
    assert len(findings["exempt"]) == 1
    assert findings["exempt"][0]["exempt_interval"] == [331, 511]


@pytest.mark.parametrize("span", [(331, 511), (332, 512), (169, 487), (1, 332), (511, 700)])
def test_a_leader_hit_overlapping_its_own_insert_is_exempt(span):
    """A15 Pin 3 as corrected 2026-10-11: OVERLAP, not containment.

    The real leader's RF00230 alignment runs from 169 to 487 against an insert of [331, 511) —
    162 nt upstream of the splice point — because a CM aligning the leader's own element does
    not stop at the junction. Under the signed "wholly inside" reading every such hit was a
    violation, which made the assertion unsatisfiable while the leader stayed in the draw.
    The 1-based inclusive hit [lo, hi] occupies [lo-1, hi), so 1..332 touches position 331 and
    511..700 touches position 510: both overlap, and both are exempt.
    """
    contigs, searches, tblouts = _postmint_inputs(50.0, span)
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert findings["violations"] == []
    assert len(findings["exempt"]) == 1
    assert findings["exempt"][0]["exempt_interval"] == [331, 511]


@pytest.mark.parametrize("span", [(1, 331), (512, 700)])
def test_a_leader_hit_DISJOINT_from_its_own_insert_is_still_a_violation(span):
    """The exemption is an interval test, never pool membership.

    One base either side of the insert and the hit is pure host DNA, which is exactly what the
    assertion exists to catch — the leader's pool does not excuse it. [1, 331] occupies
    [0, 331), ending where the insert begins; [512, 700] starts at 511, one past its end.
    """
    contigs, searches, tblouts = _postmint_inputs(50.0, span)
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert findings["exempt"] == []
    assert len(findings["violations"]) == 1


def test_only_the_leader_pool_is_exempt():
    contigs, searches, tblouts = _postmint_inputs(50.0, (332, 511), pool="structured_rna")
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert findings["exempt"] == []
    assert len(findings["violations"]) == 1


def test_a_sub_threshold_hit_is_neither_a_violation_nor_exempt():
    contigs, searches, tblouts = _postmint_inputs(9.9, (700, 800))
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert findings["violations"] == []
    assert findings["exempt"] == []


def test_a_motif_on_a_label0_contig_is_a_violation_with_no_cm_hit_at_all():
    contigs, searches, tblouts = _postmint_inputs(9.9, (700, 800))
    contigs["v000001"]["motif"] = True
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert [entry["arm"] for entry in findings["violations"]] == ["motif"]


def test_a_leader_motif_is_exempt_only_where_every_occurrence_overlaps_the_insert():
    """The motif gets the same interval test as a CM hit, so it needs positions, not a flag.

    A motif inside the leader's own T-box-derived insert is the insert being itself; one in its
    host flank is a violation, and a contig carrying both is a violation.
    """
    inside, outside = [[340, 352]], [[40, 52]]
    for spans, exempt in ((inside, True), (outside, False), (inside + outside, False)):
        contigs, searches, tblouts = _postmint_inputs(9.9, (700, 800))
        contigs["v000001"]["motif"] = True
        contigs["v000001"]["motif_spans"] = spans
        findings = H.postmint_findings(contigs, searches, tblouts)
        assert bool(findings["exempt"]) is exempt, spans
        assert bool(findings["violations"]) is (not exempt), spans
    # A motif with no recorded position cannot be tested, so it is never exempted.
    contigs, searches, tblouts = _postmint_inputs(9.9, (700, 800))
    contigs["v000001"]["motif"] = True
    assert [e["arm"] for e in H.postmint_findings(contigs, searches, tblouts)["violations"]] == [
        "motif"
    ]


def test_motif_spans_locates_both_strands_in_forward_coordinates():
    """A minus-strand match is reported at forward coordinates, so it is comparable to a hit."""
    motif = "GGTGGAACCGCG"
    seq = "AAAA" + motif + "TTTT"
    assert H.motif_spans(seq) == [(4, 16)]
    rc = wd.reverse_complement(seq)
    assert H.motif_spans(rc) == [(4, 16)]
    assert H.motif_spans("ACGT" * 10) == []
    assert H.motif_spans(motif + "A" * 20 + motif) == [(0, 12), (32, 44)]
    assert H.spans_overlap((0, 5), (4, 9)) and not H.spans_overlap((0, 4), (4, 9))
    assert not H.spans_overlap((4, 9), (0, 4))


def test_a_positive_contigs_hit_is_recorded_and_not_asserted_on():
    contigs, searches, tblouts = _postmint_inputs(9.9, (700, 800))
    text = _tblout_row("v000002", "RF00230", 100, 300, 140.0) + "\n"
    tblouts[searches[0]["tblout"]] = text
    searches[0]["data_sha256"] = H.tblout_digest(text)
    findings = H.postmint_findings(contigs, searches, tblouts)
    assert findings["violations"] == []
    assert len(findings["positive_hits"]) == 1


def test_read_searches_refuses_tblouts_searched_from_another_runs_fasta(tmp_path):
    """The fail-open pairing: unsearched windows would read as "no hits", i.e. admitted."""
    tblout_root = tmp_path / "tblout"
    (tblout_root / "RF00230").mkdir(parents=True)
    text = _tblout_row("w000001", "RF00230", 1, 240, 115.1) + "\n"
    (tblout_root / "RF00230/survivors.tblout").write_text(text, encoding="utf-8")
    search = {
        "searches": [
            {
                "fasta": H.FASTA_SURVIVORS,
                "fasta_sha256": "a" * 64,
                "model": "RF00230",
                "arm": "screen",
                "tblout": "RF00230/survivors.tblout",
                "data_sha256": H.tblout_digest(text),
            }
        ]
    }
    # The digest the search recorded is the one prep wrote: accepted.
    grouped = H.read_searches(tblout_root, search, fasta_digests={H.FASTA_SURVIVORS: "a" * 64})
    assert list(grouped) == [(H.FASTA_SURVIVORS, "RF00230")]
    # A different FASTA, same surrogate names: refused rather than read as a clean pool.
    with pytest.raises(H.HostScreenError):
        H.read_searches(tblout_root, search, fasta_digests={H.FASTA_SURVIVORS: "b" * 64})
    with pytest.raises(H.HostScreenError):
        H.read_searches(tblout_root, search, fasta_digests={})
    # Without the prep record the binding is simply not asserted (the postmint leg's path).
    assert H.read_searches(tblout_root, search)


def test_read_searches_refuses_a_tblout_whose_rows_changed_after_the_search(tmp_path):
    """An edited hit row must not reach the report as though the recorded argv produced it."""
    tblout_root = tmp_path / "tblout"
    (tblout_root / "RF00230").mkdir(parents=True)
    path = tblout_root / "RF00230/survivors.tblout"
    text = _tblout_row("w000001", "RF00230", 1, 240, 115.1) + "\n"
    path.write_text(text, encoding="utf-8")
    search = {
        "searches": [
            {
                "fasta": H.FASTA_SURVIVORS,
                "model": "RF00230",
                "arm": "screen",
                "tblout": "RF00230/survivors.tblout",
                "data_sha256": H.tblout_digest(text),
            }
        ]
    }
    assert H.read_searches(tblout_root, search)
    # A row appended (or edited) after the search: refused, not parsed.
    path.write_text(text + _tblout_row("w000002", "RF00230", 1, 99, 60.0) + "\n", encoding="utf-8")
    with pytest.raises(H.HostScreenError):
        H.read_searches(tblout_root, search)
    # A header-only change leaves the data digest intact, so it is accepted.
    path.write_text(
        "# a comment the committed fixture sanitiser rewrote\n" + text, encoding="utf-8"
    )
    assert H.read_searches(tblout_root, search)


def test_read_searches_refuses_a_tblout_whose_query_name_is_another_cm(tmp_path):
    tblout_root = tmp_path / "tblout"
    (tblout_root / "RF00230").mkdir(parents=True)
    text = _tblout_row("w000001", "TBDB001", 1, 240, 115.1) + "\n"
    (tblout_root / "RF00230/survivors.tblout").write_text(text, encoding="utf-8")
    with pytest.raises(H.HostScreenError):
        H.read_searches(
            tblout_root,
            {
                "searches": [
                    {
                        "fasta": H.FASTA_SURVIVORS,
                        "model": "RF00230",
                        "arm": "screen",
                        "tblout": "RF00230/survivors.tblout",
                        "data_sha256": H.tblout_digest(text),
                    }
                ]
            },
        )


def test_postmint_findings_refuses_a_tblout_that_changed_after_the_search():
    contigs, searches, tblouts = _postmint_inputs(50.0, (332, 511))
    tblouts[searches[0]["tblout"]] += _tblout_row("v000001", "RF00230", 1, 50, 60.0) + "\n"
    with pytest.raises(H.HostScreenError):
        H.postmint_findings(contigs, searches, tblouts)


def test_postmint_findings_refuses_a_hit_for_an_unknown_contig():
    contigs, searches, tblouts = _postmint_inputs(50.0, (332, 511))
    text = _tblout_row("v999999", "RF00230", 10, 20, 50.0) + "\n"
    tblouts[searches[0]["tblout"]] = text
    searches[0]["data_sha256"] = H.tblout_digest(text)
    with pytest.raises(H.HostScreenError):
        H.postmint_findings(contigs, searches, tblouts)


def test_leader_exempt_interval_refuses_a_row_that_cannot_state_its_insert():
    assert H.leader_exempt_interval(
        {"pool": "leader_decoy", "splice_phase": 1, "locus_length": 2}
    ) == (1, 3)
    assert (
        H.leader_exempt_interval({"pool": "structured_rna", "splice_phase": 1, "locus_length": 2})
        is None
    )
    assert (
        H.leader_exempt_interval({"pool": "leader_decoy", "splice_phase": None, "locus_length": 2})
        is None
    )
    assert (
        H.leader_exempt_interval({"pool": "leader_decoy", "splice_phase": True, "locus_length": 2})
        is None
    )


#: The matched-null maxima the committed calibration artifact measures, as a synthetic stand-in.
#: Both sit BELOW their model's pinned threshold, which is the relation the validator re-derives.
_CALIBRATION_FLOOR: dict[str, float] = {"RF00230": 15.1, "TBDB001": 11.9}


def _postmint_report(score: float = 12.0, span: tuple[int, int] = (700, 800)) -> dict:
    """A synthetic post-mint report at the re-signed per-model thresholds.

    The default score is 12.0, not the old 9.9: the search records ``-T 10.0``, so a 9.9-bit row
    is a row that search could not have reported, and the band clause now refuses it rather than
    skipping it silently. 12.0 is in-band for RF00230, so the clean baseline exercises the
    residue clauses instead of leaving them vacuous.
    """
    contigs, searches, tblouts = _postmint_inputs(score, span)
    findings = H.postmint_findings(contigs, searches, tblouts)
    return {
        "schema_version": H.POSTMINT_SCHEMA_VERSION,
        "step": H.STEP,
        "kind": H.KIND_POSTMINT,
        "adr": H.ADR,
        "threshold_bits": dict(H.POSTMINT_THRESHOLD_BITS),
        "search_threshold_bits": H.CONTENT_THRESHOLD_BITS,
        "calibration": {
            "path": H.DEFAULT_NOISE_FLOOR_REPORT,
            "sha256": "0" * 64,
            "schema_version": "1.0",
            "null_max_bits": dict(_CALIBRATION_FLOOR),
            "n_null_sequences": {"RF00230": 3073, "TBDB001": 3073},
        },
        "motif": H.MOTIF,
        "models": {model: {"cm": cm, "cm_sha256": "0" * 64} for model, cm in H.SCREEN_CMS},
        "arms": findings["arms"],
        "scope": {"n_contigs": 2, "n_label0": 1, "n_positive_hits": 0},
        "violations": findings["violations"],
        "n_violations": len(findings["violations"]),
        "exempt": findings["exempt"],
        "n_exempt": len(findings["exempt"]),
        "noise_residue": findings["noise_residue"],
        "n_noise_residue": len(findings["noise_residue"]),
        "positive_hits": findings["positive_hits"],
    }


def test_the_synthetic_postmint_report_validates_clean():
    assert H.validate_report(_postmint_report()) == []


# ── A15 Pin 3 Correction 2: the per-model threshold accessor ────────────────────────────────
def test_the_pinned_table_is_exactly_what_the_amendment_signed():
    assert H.POSTMINT_THRESHOLD_BITS == {"RF00230": 20.0, "TBDB001": 12.0}
    assert set(H.POSTMINT_THRESHOLD_BITS) == {model for model, _ in H.SCREEN_CMS}


def test_an_unpinned_model_is_refused_rather_than_given_a_default():
    """A model the amendment never priced is an unsigned configuration.

    Both plausible defaults are wrong: the screen's 10.0 re-imposes the floor the measurement
    removed, and the strictest pinned value grades a new model on a number measured for another.
    """
    with pytest.raises(H.HostScreenError, match="no A15 Pin 3 assertion threshold"):
        H.postmint_threshold("RF04321")


def test_a_threshold_below_the_screens_own_is_refused():
    """The structural rule: an assertion more sensitive than its screen cannot pass, however
    sound both are ([[assertion-stronger-than-its-screen-cannot-pass]])."""
    with pytest.raises(H.HostScreenError, match="stronger than its own screen"):
        H.postmint_threshold("RF00230", {"RF00230": H.CONTENT_THRESHOLD_BITS - 0.1})


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "20.0", None])
def test_a_threshold_that_is_not_a_bit_score_is_refused(value):
    """`float(True)` is 1.0 and would order against a comparison like any other number."""
    with pytest.raises(H.HostScreenError, match="not a bit score"):
        H.postmint_threshold("RF00230", {"RF00230": value})


def test_an_injected_threshold_table_is_honoured():
    """Pinned as the read path: the accessor must not reach past its argument to the global,
    or a caller's own table would be silently ignored."""
    assert H.postmint_threshold("RF00230", {"RF00230": 42.0}) == 42.0


def test_the_accessor_reads_the_global_at_CALL_time():
    """A module global is not seen by a def-time default or a `from`-import rebind
    ([[pinned-constant-has-two-read-paths]]), so the live value must be read per call."""
    original = dict(H.POSTMINT_THRESHOLD_BITS)
    try:
        H.POSTMINT_THRESHOLD_BITS["RF00230"] = 33.0
        assert H.postmint_threshold("RF00230") == 33.0
        contigs, searches, tblouts = _postmint_both_models([("RF00230", 25.0, (100, 200))])
        findings = H.postmint_findings(contigs, searches, tblouts)
        assert findings["violations"] == [], "the verdict did not follow the live global"
        assert len(findings["noise_residue"]) == 1
    finally:
        H.POSTMINT_THRESHOLD_BITS.clear()
        H.POSTMINT_THRESHOLD_BITS.update(original)


def test_the_amendment_prose_states_the_values_the_code_pins():
    """A15 Pin 3's prose and `POSTMINT_THRESHOLD_BITS` can otherwise diverge indefinitely: no
    test reads the ADR, and A13's gated statistic is threshold-free, so CI stays green."""
    adr = (
        Path(__file__).resolve().parents[2] / "docs/decisions/ADR-0005-non-circular-eval-design.md"
    ).read_text(encoding="utf-8")
    assert "POSTMINT_THRESHOLD_BITS" in adr
    for model, bits in H.POSTMINT_THRESHOLD_BITS.items():
        assert f"{model} **≥ {bits:.1f}**" in adr or f"{model} ≥ {bits:.1f}" in adr, model
    # Pin 2 must still read 10.0: Correction 2 changed the assertion, never the screen.
    pin2 = f"refused if either model reports a hit of **≥ {H.CONTENT_THRESHOLD_BITS:.1f} bits**"
    assert pin2 in adr


@pytest.mark.parametrize(
    "mutate,expected",
    [
        # Retargeted for Correction 2. The old case assigned the scalar 20.0, which after the
        # per-model change fails on TYPE rather than on value — green for the wrong reason, and
        # no longer testing the clause. Break ONE model's value and leave the other consistent
        # ([[all-true-fixture-cannot-test-a-conjunction]]).
        (
            lambda r: r["threshold_bits"].__setitem__("RF00230", H.CONTENT_THRESHOLD_BITS),
            "pinned table",
        ),
        (lambda r: r["threshold_bits"].__setitem__("TBDB001", 20.0), "pinned table"),
        (lambda r: r.__setitem__("threshold_bits", H.CONTENT_THRESHOLD_BITS), "pinned table"),
        (lambda r: r["threshold_bits"].pop("TBDB001"), "no Pin 3 assertion threshold"),
        (lambda r: r.__setitem__("search_threshold_bits", 20.0), "search_threshold_bits"),
        (lambda r: r.__setitem__("schema_version", H.SCHEMA_VERSION), "schema_version"),
        # The calibration binding: the thresholds must clear the measured floor, and must name
        # a specific artifact to have been measured at all.
        (lambda r: r.pop("calibration"), "calibration is missing"),
        (lambda r: r["calibration"].__setitem__("sha256", ""), "calibration.sha256"),
        (
            lambda r: r["calibration"]["null_max_bits"].__setitem__("RF00230", 21.0),
            "does not clear the measured matched-null maximum",
        ),
        (
            # Equal is not enough: a threshold ON the worst null sequence still fires on it.
            lambda r: r["calibration"]["null_max_bits"].__setitem__("TBDB001", 12.0),
            "does not clear the measured matched-null maximum",
        ),
        (
            # The 0.1-bit margin, pinned. TBDB001's measured floor is 11.9 and its threshold is
            # 12.0, so a floor re-measured only 0.05 bits higher already refuses it — which is
            # how the first TBDB001 value of 10.0 was caught against a floor of 11.9, and why
            # the artifact publishes n_null_sequences beside every maximum.
            lambda r: r["calibration"]["null_max_bits"].__setitem__("TBDB001", 12.05),
            "does not clear the measured matched-null maximum",
        ),
        (
            lambda r: r["calibration"]["null_max_bits"].pop("RF00230"),
            "signed against no measured null",
        ),
        # The partition between violations and residue, re-derived per row.
        (
            lambda r: r["violations"].append({**r["noise_residue"][0], "model": "RF00230"})
            or r.__setitem__("n_violations", 1),
            "belongs in noise_residue",
        ),
        (
            lambda r: r["noise_residue"][0].__setitem__("bits", 25.0),
            "outside the disclosed band",
        ),
        (
            lambda r: r["noise_residue"][0].__setitem__("bits", 9.5),
            "outside the disclosed band",
        ),
        (
            lambda r: r["noise_residue"][0].__setitem__("assertion_threshold_bits", 10.0),
            "records assertion threshold",
        ),
        (lambda r: r.__setitem__("n_noise_residue", 9), "n_noise_residue != len"),
        (lambda r: r.pop("noise_residue"), "noise_residue is missing"),
        (lambda r: r.__setitem__("motif", "GGTGG"), "motif"),
        (lambda r: r.__setitem__("n_violations", 1), "n_violations != len(violations)"),
        (lambda r: r.__setitem__("n_exempt", 7), "n_exempt != len(exempt)"),
        (lambda r: r["arms"].pop("filter_off"), "no filter-off"),
        (lambda r: r["arms"].pop("screen"), "no screen arm"),
        # Coverage: the per-search clauses iterate what is PRESENT, so they cannot see a search
        # that was removed. Each of these validated CLEAN before the coverage clause existed.
        (
            lambda r: [
                b["argv"].pop(k)
                for b in r["arms"].values()
                for k in [x for x in list(b["argv"]) if "TBDB001" in x]
            ],
            "required search(es)",
        ),
        (lambda r: r["arms"]["filter_off"].__setitem__("argv", {}), "required search(es)"),
        (
            lambda r: r["arms"]["screen"]["argv"].pop(f"{H.FASTA_POSTMINT_ALL}:RF00230"),
            "required search(es)",
        ),
        (lambda r: r["scope"].__setitem__("n_label0", 0), "n_label0 is not positive"),
        (
            lambda r: r["arms"]["filter_off"]["argv"].__setitem__(
                f"{H.FASTA_POSTMINT_LABEL0}:RF00230",
                [t for t in next(iter(r["arms"]["filter_off"]["argv"].values())) if t != "--max"],
            ),
            "does not pass --max",
        ),
        (
            lambda r: r["arms"]["screen"]["argv"].__setitem__(
                f"{H.FASTA_POSTMINT_ALL}:RF00230",
                [*next(iter(r["arms"]["screen"]["argv"].values())), "--max"],
            ),
            "arms.screen.argv",
        ),
    ],
)
def test_each_postmint_clause_is_breakable_alone(mutate, expected):
    report = _postmint_report()
    assert H.validate_report(report) == []
    mutate(report)
    problems = H.validate_report(report)
    assert any(expected in problem for problem in problems), (expected, problems)


def test_a_violation_is_reported_as_a_failed_assertion():
    report = _postmint_report(score=50.0, span=(700, 800))
    problems = H.validate_report(report)
    assert any("the post-mint assertion failed" in problem for problem in problems), problems


def test_an_exempt_entry_disjoint_from_the_leader_insert_is_caught_in_the_report_too():
    """A forged exemption must be refused by the report's own clause set, not only by the leg.

    The forged span is DISJOINT from the insert, not merely wider than it: under the corrected
    overlap rule (A15 Pin 3, 2026-10-11) a hit spanning the whole contig does overlap the insert
    and is legitimately exempt, so the old [1, 1000] forge would no longer bite.
    """
    report = _postmint_report(score=50.0, span=(332, 511))
    assert H.validate_report(report) == []
    report["exempt"][0]["span"] = [1, 331]
    assert any("does not overlap" in problem for problem in H.validate_report(report))
    report["exempt"][0]["span"] = [512, 700]
    assert any("does not overlap" in problem for problem in H.validate_report(report))


def test_an_exempt_entry_for_another_pool_is_caught_in_the_report_too():
    report = _postmint_report(score=50.0, span=(332, 511))
    report["exempt"][0]["pool"] = "gc_background"
    assert any(
        "only the leader insert is exempt" in problem for problem in H.validate_report(report)
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# The CLI's shape
# ══════════════════════════════════════════════════════════════════════════════════════
def test_every_leg_is_reachable_and_defaults_to_repo_relative_paths():
    parser = H.build_parser()
    for leg in ("prep", "search", "report", "familiarity", "postmint-prep", "postmint-report"):
        assert any(
            leg in action.choices
            for action in parser._subparsers._group_actions  # noqa: SLF001 - argparse has no API
            if hasattr(action, "choices")
        )
    args = parser.parse_args(["prep"])
    assert args.mining_pool == H.DEFAULT_MINING_POOL
    assert not Path(args.mining_pool).is_absolute()
    assert args.data_root is None


def test_the_model_pair_is_the_adr_0005_d2_pair():
    assert [model for model, _ in H.SCREEN_CMS] == ["RF00230", "TBDB001"]
    assert dict(H.SCREEN_CMS)["RF00230"].endswith("RF00230.cm")
    assert set(H.CM_QUERY_NAMES) == {model for model, _ in H.SCREEN_CMS}


def test_the_pinned_a15_construction_values():
    assert H.KMER_K == 32
    assert H.CONTENT_THRESHOLD_BITS == 10.0
    assert H.AUDIT_THRESHOLD_BITS == 0.0
    assert H.SEARCH_SPACE_MB == 1.0
    assert H.MOTIF == "GGTGGNACCGCG"
    assert H.STAGE2_EXPECTED_ADMITTED == 9059
    assert H.FASTA_THRESHOLDS[H.FASTA_SURVIVORS] == 10.0
    assert H.FASTA_THRESHOLDS[H.FASTA_V1_HOSTS] == 0.0


def test_disclosures_carry_the_circularity_and_lower_bound_sentences():
    ids = {entry["id"] for entry in H.disclosures()}
    assert {"lower_bound", "cm_circularity", "threshold_choice"} <= ids
    text = " ".join(entry["text"] for entry in H.disclosures())
    assert "false positive" in text
    assert "lower-bounds" in text


def test_the_module_imports_without_pandas_so_the_search_leg_can_run_in_the_infernal_env():
    """The import surface, read off the module's own source.

    A runtime ``"pandas" not in sys.modules`` assertion would be vacuous here — another test
    module in the same session imports pandas — so what is pinned is that no *module-level*
    import mentions it and that the data tier's import is marked lazy. The infernal env, which
    has cmsearch and no pandas, is where that claim is cashed.
    """
    source = Path(H.__file__).read_text(encoding="utf-8")
    top_level = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "pandas" in line
    ]
    assert top_level == [], top_level
    assert "import pandas as pd  # lazy" in source


def test_copying_a_report_does_not_change_its_verdict():
    report = _synthetic_report()
    assert H.validate_report(copy.deepcopy(report)) == []
