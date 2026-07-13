"""Unit tests for the ArchiveII secondary-structure base-pair-F1 metric + .ct parser.

Pure-stdlib (runs in bare CI). Covers the RiNALMo-protocol base-pair F1 (±1 nt
slippage), the canonical-pair / sharp-loop helpers, the pair-set utilities, and
the fail-loud .ct parser — with hand-checked expected values and mutation bites.
"""

from __future__ import annotations

import io
import tarfile

import pytest

from tbox_finder.eval import archiveii_lofo as al


# --------------------------------------------------------------------------- #
# base_pair_prf — hand-checked precision / recall / F1
# --------------------------------------------------------------------------- #
def test_perfect_match_is_one() -> None:
    ref = {(1, 20), (2, 19), (3, 18)}
    p, r, f = al.base_pair_prf(ref, ref, slippage=False)
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_two_of_three_no_slippage() -> None:
    ref = {(1, 10), (2, 9), (3, 8)}
    pred = {(1, 10), (2, 9), (4, 7)}  # (4,7) is not a ±1 neighbor of any ref pair
    p, r, f = al.base_pair_prf(pred, ref, slippage=False, seq_len=10)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)
    assert f == pytest.approx(2 / 3)


def test_slippage_flips_near_miss_to_hit() -> None:
    # A single pair shifted by one nt: a miss without slippage, perfect with it.
    ref = {(1, 10)}
    pred = {(2, 10)}
    assert al.base_pair_prf(pred, ref, slippage=False, seq_len=10) == (0.0, 0.0, 0.0)
    assert al.base_pair_prf(pred, ref, slippage=True, seq_len=10) == (1.0, 1.0, 1.0)


def test_slippage_only_tolerates_one_nt() -> None:
    # Shift of two nt is beyond the ±1 tolerance → still a miss even with slippage.
    ref = {(1, 10)}
    pred = {(3, 10)}
    assert al.base_pair_prf(pred, ref, slippage=True, seq_len=10) == (0.0, 0.0, 0.0)


def test_both_defined_zero_match_is_f1_zero_not_nan() -> None:
    # Predictions and reference both exist but share nothing → F1 is 0.0, not nan.
    p, r, f = al.base_pair_prf({(4, 7)}, {(1, 10)}, slippage=False, seq_len=10)
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_empty_prediction_scores_zero_rinalmo_parity() -> None:
    # RiNALMo uses sklearn zero_division=0.0 → an empty prediction scores 0.0
    # (never nan), pulling the family mean toward 0.0 rather than poisoning it.
    assert al.base_pair_prf(set(), {(1, 10)}, slippage=False) == (0.0, 0.0, 0.0)


def test_empty_reference_scores_zero_rinalmo_parity() -> None:
    assert al.base_pair_prf({(1, 10)}, set(), slippage=False) == (0.0, 0.0, 0.0)


def test_no_pairs_on_either_side_scores_zero() -> None:
    # both empty → 0.0 (RiNALMo parity; ArchiveII records always have pairs so
    # this branch is dormant in production but must not return nan).
    assert al.base_pair_prf(set(), set(), slippage=True) == (0.0, 0.0, 0.0)


def test_precision_and_recall_can_differ_under_slippage() -> None:
    # One ref pair; two predicted pairs, one within ±1 of the ref, one far.
    ref = {(5, 20)}
    pred = {(5, 21), (1, 2)}  # (5,21) matches ref±1; (1,2) is spurious
    p, r, f = al.base_pair_prf(pred, ref, slippage=True, seq_len=25)
    assert p == pytest.approx(0.5)  # 1 of 2 predicted pairs is precise
    assert r == pytest.approx(1.0)  # the single ref pair is recalled
    assert f == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))


# --------------------------------------------------------------------------- #
# relax_pairs — the ±1 neighborhood
# --------------------------------------------------------------------------- #
def test_relax_pairs_neighborhood() -> None:
    got = al.relax_pairs({(5, 10)}, seq_len=20)
    assert got == {(5, 10), (4, 10), (6, 10), (5, 9), (5, 11)}


def test_relax_pairs_respects_bounds_and_drops_degenerate() -> None:
    # (1,2): i-1=0 dropped (below 1); i+1=2 gives (2,2) degenerate dropped.
    got = al.relax_pairs({(1, 2)}, seq_len=3)
    assert (0, 2) not in got and (2, 2) not in got
    assert (1, 2) in got and (1, 3) in got  # (1,1) degenerate excluded
    assert (1, 1) not in got
    # upper bound respected
    assert all(i <= 3 and j <= 3 for i, j in got)


# --------------------------------------------------------------------------- #
# canonical_pairs_only + pairs_key
# --------------------------------------------------------------------------- #
def test_canonical_pairs_only_filters_noncanonical_and_sharp() -> None:
    seq = "GGGGAAAACCCC"  # 1..12; G-C canonical, A-A not
    pairs = [
        (1, 12),  # G-C canonical, dist 11 -> keep
        (5, 6),  # A-A non-canonical AND sharp -> drop
        (2, 4),  # G-G non-canonical -> drop
        (1, 3),  # G-G + dist 2 < 4 sharp -> drop
    ]
    kept = al.canonical_pairs_only(seq, pairs)
    assert kept == ((1, 12),)


def test_canonical_pairs_only_keeps_gu_wobble() -> None:
    seq = "GAAAAAU"  # G(1) .. U(7); G-U wobble, dist 6
    assert al.canonical_pairs_only(seq, [(1, 7)]) == ((1, 7),)


def test_pairs_key_is_sorted_and_pseudoknot_safe() -> None:
    # crossing pairs (a pseudoknot) survive losslessly in a canonical order
    key = al.pairs_key([(10, 20), (1, 15)])
    assert key == "1-15;10-20"


# --------------------------------------------------------------------------- #
# parse_ct — hand-checked + fail-loud
# --------------------------------------------------------------------------- #
_MINI_CT = (
    "5\ttoy\n"
    "1\tG\t0\t2\t5\t1\n"
    "2\tC\t1\t3\t0\t2\n"
    "3\tA\t2\t4\t0\t3\n"
    "4\tG\t3\t5\t0\t4\n"
    "5\tC\t4\t0\t1\t5\n"
)


def test_parse_ct_sequence_and_pairs() -> None:
    rec = al.parse_ct(_MINI_CT, record_id="toy")
    assert rec.sequence == "GCAGC"
    assert rec.length == 5
    assert rec.pairs == ((1, 5),)  # symmetric 1<->5, canonicalized i<j
    assert rec.title == "toy"


def test_parse_ct_rejects_length_mismatch() -> None:
    bad = "9\tx\n" + "\n".join(f"{i}\tA\t{i-1}\t{i+1}\t0\t{i}" for i in range(1, 4))
    with pytest.raises(ValueError, match="header says"):
        al.parse_ct(bad, record_id="short")


def test_parse_ct_rejects_asymmetric_pairing() -> None:
    # residue 1 says it pairs with 3, but residue 3 says unpaired -> asymmetric
    bad = "3\tx\n" "1\tG\t0\t2\t3\t1\n" "2\tA\t1\t3\t0\t2\n" "3\tC\t2\t0\t0\t3\n"
    with pytest.raises(ValueError, match="asymmetric"):
        al.parse_ct(bad, record_id="asym")


def test_parse_ct_rejects_noncontiguous_index() -> None:
    bad = (
        "2\tx\n"
        "1\tG\t0\t2\t0\t1\n"
        "9\tC\t1\t3\t0\t2\n"  # index jumps to 9
    )
    with pytest.raises(ValueError, match="non-contiguous"):
        al.parse_ct(bad, record_id="jump")


def test_parse_ct_rejects_self_pairing() -> None:
    # a residue that pairs to itself (j == i) must fail loud, not silently vanish
    # under the i<j pair filter
    bad = "2\ttoy\n1\tG\t0\t2\t1\t1\n2\tC\t1\t0\t0\t2\n"  # residue 1 pairs to 1
    with pytest.raises(ValueError, match="pairs to itself"):
        al.parse_ct(bad, record_id="selfpair")


def test_parse_ct_rejects_empty_file() -> None:
    with pytest.raises(ValueError, match="empty .ct file"):
        al.parse_ct("   \n\t\n", record_id="blank")


def test_parse_ct_rejects_malformed_residue_line() -> None:
    # a residue row with < 6 columns fails loud, not with an opaque IndexError
    bad = "1\ttoy\n1\tG\t0\t2\n"  # body row has 4 fields
    with pytest.raises(ValueError, match="malformed residue line"):
        al.parse_ct(bad, record_id="short-row")


# --------------------------------------------------------------------------- #
# _safe_extract — tar path-traversal guard (external-download security)
# --------------------------------------------------------------------------- #
def _tar_with_member(name: str, data: bytes = b"x") -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def test_safe_extract_blocks_path_traversal(tmp_path) -> None:
    buf = _tar_with_member("../escape.txt")
    with (
        tarfile.open(fileobj=buf, mode="r") as tf,
        pytest.raises(ValueError, match="unsafe tar member"),
    ):
        al._safe_extract(tf, tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_allows_benign_member(tmp_path) -> None:
    dest = tmp_path / "dest"
    buf = _tar_with_member("ct/fam-fold/ok.txt", b"hello")
    with tarfile.open(fileobj=buf, mode="r") as tf:
        al._safe_extract(tf, dest)
    assert (dest / "ct/fam-fold/ok.txt").read_bytes() == b"hello"


# --------------------------------------------------------------------------- #
# real fixture record — parser correctness against hand-read .ct
# --------------------------------------------------------------------------- #
def test_real_fixture_srp_record() -> None:
    from pathlib import Path

    p = (
        Path(__file__).resolve().parents[1]
        / "fixtures/archiveii_lofo/mini/ct/fam-fold/srp/test/srp_Shig.flex._CP000266.ct"
    )
    rec = al.parse_ct_file(p)
    assert rec.sequence == "CCGUCAGGUCCGGAAGGAAGCAGCGGUA"
    assert rec.length == 28
    assert rec.pairs == ((1, 26), (2, 25), (3, 24), (4, 23), (9, 18), (10, 17), (11, 16))
    # self-scores perfectly
    assert al.base_pair_prf(rec.pairs, rec.pairs, slippage=True, seq_len=rec.length)[2] == 1.0
