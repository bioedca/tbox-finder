"""Golden-file regression for P2-01 Stage-1 window carving + label alignment.

CLAUDE.md §8.1: a 50-200-record fixture input + a committed ``expected.sha256``;
CI re-runs the stage and diffs the hash. This locks the load-bearing P2-01
invariant — *where the labels land inside a window* — against silent drift in the
window geometry, the label projection, the zero-flank rule, or the RC transform.

**Fixture** (``tests/fixtures/window_labels/context_sample.csv``, 56 real records):
a deterministic stratified pick from the P2-00 anchored corpus, taken in
``record_id`` order as the union of the first 12 ``klass == "II"``, the first 10
``clipped_start``, the first 10 ``clipped_end``, the first 28 unclipped
``klass == "I"``, and the first 8 ``nested_train`` records, deduplicated and sorted
by ``record_id``. It joins the real ``context_v0.parquet`` geometry (P2-00), the
real ``labels_v0.parquet`` ``label_string`` (P0-20), and the real
``split_assignments.parquet`` strata (P0-23) — no synthetic sequence (§10.3).
P2-06a appended the real ``is_designated_loo_holdout`` column from the same split
table (24 of the 56 are True). It is not part of what this golden pins —
``windows_digest`` hashes ids, tokens, targets and padding only — so the window
digest is unchanged by its arrival; only the input's own byte-pin moved.

**Two independent paths** (the ``test_archiveii_lofo.py`` precedent): the module
path (:func:`window_dataset.carve_window` + :func:`window_dataset.windows_digest`)
and an inline stdlib re-implementation that shares no code with the module beyond
the ``ingest`` hash primitives. A bug in either alone breaks the test.

Stdlib + numpy only (no pandas, no torch) — runs green in bare CI.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tbox_finder import ingest
from tbox_finder import labels as labels_mod
from tbox_finder.data import window_dataset as wd

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "window_labels"
_SAMPLE = _FIXTURE_DIR / "context_sample.csv"
_EXPECTED = _FIXTURE_DIR / "expected.sha256"


def _load_fixture() -> list[dict[str, str]]:
    with _SAMPLE.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(text: str) -> bool:
    """Parse the CSV's pandas-written booleans strictly (no truthiness fallback)."""
    if text in ("True", "true"):
        return True
    if text in ("False", "false"):
        return False
    raise ValueError(f"not a boolean: {text!r}")


def _fixture_records() -> list[wd.CorpusRecord]:
    records = []
    for row in _load_fixture():
        records.append(
            wd.CorpusRecord(
                record_id=row["record_id"],
                context_seq=row["context_seq"],
                locus_offset=int(row["locus_offset"]),
                locus_length=int(row["locus_length"]),
                label_string=row["label_string"],
                clipped_start=_as_bool(row["clipped_start"]),
                clipped_end=_as_bool(row["clipped_end"]),
                klass=row["klass"],
                phylum=row["resolved_phylum"],
                cognate_aa=row["cognate_aa"],
                cluster_id=int(row["cluster_id"]),
                nested_train=_as_bool(row["nested_train"]),
                # Real values, joined from the committed split table at P2-06a — NOT a
                # convenience default. 24 of these 56 records genuinely ARE designated
                # LOO holdout, so `False` would have written a falsehood into a fixture
                # that exists to be trusted. The join also re-confirmed the fixture's own
                # `nested_train` against the table (all 56 agree).
                is_designated_loo_holdout=_as_bool(row["is_designated_loo_holdout"]),
                folds=tuple(row[c] for c in wd.FOLD_SCHEME_COLUMNS),
            )
        )
    return records


def _module_digest() -> str:
    """Path 1 — the production module: dataset eval-mode carve + windows_digest."""
    ds = wd.Stage1WindowDataset(_fixture_records(), augment=False)
    return wd.windows_digest([ds.window_at(i) for i in range(len(ds))])


def _rc_windows() -> list[wd.Window]:
    """The same eval-mode windows carved on the reverse strand."""
    windows = []
    for r in _fixture_records():
        rng = wd.window_lead_range(
            locus_offset=r.locus_offset,
            locus_length=r.locus_length,
            context_length=len(r.context_seq),
            clipped_start=r.clipped_start,
            clipped_end=r.clipped_end,
        )
        assert rng is not None
        lead = wd.deterministic_lead(rng, window=1024, locus_length=r.locus_length)
        windows.append(
            wd.carve_window(
                context_seq=r.context_seq,
                locus_offset=r.locus_offset,
                locus_length=r.locus_length,
                label_string=r.label_string,
                lead=lead,
                record_id=r.record_id,
                clipped_start=r.clipped_start,
                clipped_end=r.clipped_end,
                rc=True,
            )
        )
    return windows


def _independent_digest() -> str:
    """Path 2 — an inline re-implementation of the carve, sharing no module code.

    Deliberately re-derives the honest lead range, the window geometry, the label
    projection, and the zero-flank padding from the fixture's raw columns with a
    plain per-position loop, using only the class vocabulary and the ``ingest``
    hash primitives — so it cannot inherit a bug from ``window_dataset``.
    """
    window = 1024
    base_ids = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
    pad_id, ignore = 4, -100
    code_to_index = {
        code: labels_mod.CLASS_INDEX[name] for name, code in labels_mod.CLASS_CODE.items()
    }

    keyed: list[tuple[str, str]] = []
    for row in _load_fixture():
        seq = row["context_seq"]
        off, m = int(row["locus_offset"]), int(row["locus_length"])
        lstr = row["label_string"]
        cs, ce = _as_bool(row["clipped_start"]), _as_bool(row["clipped_end"])
        clen = len(seq)

        # Honest lead range, re-derived from first principles.
        lo = 0 if ce else max(0, off + window - clen)
        hi = (window - m) if cs else min(window - m, off)
        assert lo <= hi, f"fixture record {row['record_id']} admits no honest window"
        lead = min(max((window - m) // 2, lo), hi)  # eval-mode: centred, clamped

        start = off - lead
        ids, lab = [], []
        for k in range(window):
            p = start + k
            if 0 <= p < clen:
                ids.append(base_ids.get(seq[p], base_ids["N"]))
                # real DNA: background unless the locus covers this position
                lab.append(code_to_index[lstr[p - off]] if off <= p < off + m else 0)
            else:
                ids.append(pad_id)
                lab.append(ignore)

        vid = f"{row['record_id']}#w{lead}"
        keyed.append(
            (
                vid,
                ingest.record_hash(
                    [
                        vid,
                        row["record_id"],
                        np.asarray(ids, dtype=np.int16).tobytes().hex(),
                        np.asarray(lab, dtype=np.int16).tobytes().hex(),
                        max(0, -start),
                        max(0, (start + window) - clen),
                    ]
                ),
            )
        )
    keyed.sort(key=lambda kv: kv[0])  # the module sorts windows by variant id
    return ingest.records_digest([h for _, h in keyed])


def test_fixture_present() -> None:
    """Stdlib guard: the fixture + expectation are committed and well-formed."""
    assert _SAMPLE.is_file(), f"missing golden fixture input: {_SAMPLE}"
    assert _EXPECTED.is_file(), f"missing golden expectation: {_EXPECTED}"
    assert len(_EXPECTED.read_text().strip()) == 64
    rows = _load_fixture()
    assert 50 <= len(rows) <= 200, f"fixture must hold 50-200 records, got {len(rows)}"


def test_fixture_covers_the_material_geometry() -> None:
    """The fixture must exercise both classes and both zero-flank directions.

    A golden hash over an all-unclipped class-I sample would lock nothing about
    the two paths most likely to drift (the contig-end zero-flank and the scarce
    class-II records).
    """
    rows = _load_fixture()
    assert {r["klass"] for r in rows} == {"I", "II"}
    assert sum(_as_bool(r["clipped_start"]) for r in rows) >= 5
    assert sum(_as_bool(r["clipped_end"]) for r in rows) >= 5
    assert sum(_as_bool(r["nested_train"]) for r in rows) >= 5


def test_window_digest_matches_expected() -> None:
    """Path 1: the production carve reproduces the committed digest."""
    assert _module_digest() == _EXPECTED.read_text().strip()


def test_independent_reimplementation_agrees() -> None:
    """Path 2: an inline re-derivation reproduces the same digest."""
    assert _independent_digest() == _EXPECTED.read_text().strip()


def test_label_alignment_holds_on_every_fixture_record() -> None:
    """The P2-01 invariant, asserted directly rather than only via the hash.

    Within a window: the locus slice equals ``label_string``; every real-DNA flank
    position is ``background``; every zero-flanked position is ``IGNORE_INDEX`` and
    carries ``[PAD]``; and the window's real sequence matches ``context_seq``.
    """
    ds = wd.Stage1WindowDataset(_fixture_records(), augment=False)
    for i, rec in enumerate(ds.records):
        w = ds.window_at(i)
        assert w.input_ids.shape == (1024,)
        assert w.labels.shape == (1024,)

        expect = np.asarray(labels_mod.label_string_to_indices(rec.label_string), dtype=np.int16)
        locus = w.labels[w.lead : w.lead + rec.locus_length]
        assert np.array_equal(locus, expect), f"label misalignment on {rec.record_id}"

        # Real flank is background — never ignored.
        flank = w.real_mask.copy()
        flank[w.lead : w.lead + rec.locus_length] = False
        assert np.all(w.labels[flank] == wd.BACKGROUND_INDEX)
        assert not np.any(w.labels[flank] == wd.IGNORE_INDEX)

        # Zero-flank is ignored — never background.
        pad = ~w.real_mask
        assert np.all(w.labels[pad] == wd.IGNORE_INDEX)
        assert np.all(w.input_ids[pad] == wd.PAD_TOKEN_ID)

        # The real span reproduces context_seq exactly.
        start = rec.locus_offset - w.lead
        real = rec.context_seq[max(0, start) : start + 1024]
        assert np.array_equal(w.input_ids[w.real_mask], wd.encode_bases(real))


def test_expected_digest_is_stable_across_repeat_runs() -> None:
    """Eval-mode carving is deterministic (no hidden RNG in the non-augmented path)."""
    assert _module_digest() == _module_digest()


def test_rc_windows_are_locked_against_an_independent_projection() -> None:
    """Lock the RC transform on real records, independently of the module.

    The forward digest above never exercises `rc=True` (eval mode is
    forward-strand), so without this the reverse-strand carve — half of the
    both-strand handling imp.md asks for — would be golden-unlocked.

    Every expectation here is built from the **fixture's raw columns** with a
    local complement table and a plain per-position loop. It deliberately does
    NOT call `wd.reverse_complement_ids` or any other RC production helper: an
    expectation computed with the function under test would pass no matter what
    that function did.
    """
    window = 1024
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    base_ids = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
    pad_id, ignore = 4, -100
    code_to_index = {
        code: labels_mod.CLASS_INDEX[name] for name, code in labels_mod.CLASS_CODE.items()
    }

    for row, rev in zip(_load_fixture(), _rc_windows(), strict=True):
        seq = row["context_seq"]
        off, m = int(row["locus_offset"]), int(row["locus_length"])
        lstr = row["label_string"]
        cs, ce = _as_bool(row["clipped_start"]), _as_bool(row["clipped_end"])
        clen = len(seq)

        lo = 0 if ce else max(0, off + window - clen)
        hi = (window - m) if cs else min(window - m, off)
        lead = min(max((window - m) // 2, lo), hi)
        start = off - lead

        # Build the forward window from raw characters, then reverse-complement it
        # by hand: walk the window backwards, complementing each base.
        exp_ids, exp_lab, exp_mask = [], [], []
        for k in range(window - 1, -1, -1):
            p = start + k
            if 0 <= p < clen:
                exp_ids.append(base_ids[comp[seq[p]]] if seq[p] in comp else base_ids["N"])
                exp_lab.append(code_to_index[lstr[p - off]] if off <= p < off + m else 0)
                exp_mask.append(True)
            else:
                exp_ids.append(pad_id)
                exp_lab.append(ignore)
                exp_mask.append(False)

        assert np.array_equal(rev.input_ids, np.asarray(exp_ids, dtype=np.int16))
        assert np.array_equal(rev.labels, np.asarray(exp_lab, dtype=np.int16))
        assert np.array_equal(rev.real_mask, np.asarray(exp_mask, dtype=bool))
        # `lead` must describe the EMITTED window, not the forward one.
        assert rev.lead == window - lead - m
        assert rev.pad_left == max(0, (start + window) - clen)  # forward's pad_right
        assert rev.pad_right == max(0, -start)
        # The locus, read off the emitted window, is the reversed label string.
        expect = [code_to_index[c] for c in reversed(lstr)]
        got = rev.labels[rev.lead : rev.lead + m]
        assert np.array_equal(
            got, np.asarray(expect, dtype=np.int16)
        ), f"RC label misalignment on {row['record_id']}"


def test_rc_digest_is_stable_and_differs_from_the_forward_digest() -> None:
    """The RC carve is deterministic and is genuinely a different window set."""
    assert wd.windows_digest(_rc_windows()) == wd.windows_digest(_rc_windows())
    assert wd.windows_digest(_rc_windows()) != _module_digest()


def test_digest_moves_if_a_label_shifts() -> None:
    """Anti-tautology: the golden hash must actually bite on a 1-nt label shift."""
    records = _fixture_records()
    victim = records[0]
    shifted = wd.CorpusRecord(
        **{
            **victim.__dict__,
            "label_string": victim.label_string[1:] + victim.label_string[:1],
        }
    )
    if shifted.label_string == victim.label_string:
        pytest.skip("fixture record 0 has a rotation-invariant label_string")
    ds = wd.Stage1WindowDataset([shifted, *records[1:]], augment=False)
    mutated = wd.windows_digest([ds.window_at(i) for i in range(len(ds))])
    assert mutated != _EXPECTED.read_text().strip()


def test_sample_csv_is_byte_stable() -> None:
    """The fixture input itself is hash-pinned, so a silent edit to it is visible.

    Distinct from ``expected.sha256`` (which pins the *carved windows*): if the
    input silently changed, both it and the window digest would move together and
    a regenerated expectation would hide it. `.gitattributes` marks
    `tests/fixtures/** -text`, so these bytes are stable across checkouts.
    """
    assert (
        hashlib.sha256(_SAMPLE.read_bytes()).hexdigest()
        == "8989ddc55c2c275d46e689198d28471b2db4c8866e78608eeab4e40c5de3cf57"
    )
