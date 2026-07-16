"""P2-03 golden regression — the frozen window-reconciliation operator (ADR-0005 D3 + A3).

What is pinned
--------------
`tests/fixtures/reconcile/geometry.csv` fixes the tiling geometries; the per-window
logits are synthesised from a **SHAKE-256 stream keyed by (case_id, window index)**. The
operator is pure arithmetic over logits — content-independent — and no trained Stage-1
checkpoint exists before P2-04, so synthetic logits are the only available input and a
legitimate one: this golden pins the operator's *arithmetic*, and asserts nothing
biological (the P1-14 synthetic-RNA-latency-probe precedent; §10.3).

The **geometry is real**: every `sampled_real_context_length` row is a context length that
actually occurs in the committed 56-row `tests/fixtures/window_labels/context_sample.csv`
(drawn from the P2-00 corpus), and a test re-checks that claim against the file rather than
trusting the CSV's own `source` column. They are real lengths, **not corpus extremes** —
the full corpus spans 183-2598 nt — and the `note` column is scoped to the sample.

SHAKE-256 (FIPS 202) is used rather than `numpy.random.default_rng` deliberately: NumPy
does not guarantee `Generator` stream stability across releases, so an RNG-seeded fixture
could drift under a numpy bump and be silently "repaired" by regenerating the digest.

Two independent paths
---------------------
`_module_digest()` runs `infer.reconcile`; `_independent_digest()` re-derives every value
in stdlib `math`/`hashlib` with explicit per-position loops, sharing nothing with the
module. Both are asserted against the same committed digest, and `test_digest_bites_*`
prove the digest moves when the operator is sabotaged in the two ways that matter.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from tbox_finder import ingest
from tbox_finder.data import window_dataset as wd
from tbox_finder.infer import reconcile as rc

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "reconcile"
_GEOMETRY = _FIXTURE_DIR / "geometry.csv"
_EXPECTED = _FIXTURE_DIR / "expected.sha256"
_CORPUS_SAMPLE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "window_labels" / "context_sample.csv"
)

#: byte-pin: a silent edit to the geometry cannot be laundered by regenerating the digest.
_GEOMETRY_SHA256 = "84b7c455e7b14fef704de716ee88e39831a96b7482fdedc3404b16b41b0fd538"

#: Logit range of the synthetic stream: wide enough that softmax spans near-0..near-1.
_LOGIT_LO, _LOGIT_HI = -8.0, 8.0


# --------------------------------------------------------------------------------------
# fixture I/O + the pinned synthetic-logit stream (shared input; the two paths differ
# only in how they *reconcile* it)
# --------------------------------------------------------------------------------------


def _cases() -> list[dict[str, object]]:
    with _GEOMETRY.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [
        {
            "case_id": r["case_id"],
            "seq_len": int(r["seq_len"]),
            "window": int(r["window"]),
            "stride": int(r["stride"]),
            "source": r["source"],
        }
        for r in rows
    ]


def _stream(case_id: str, w: int, n_bytes: int) -> bytes:
    """SHAKE-256 keyed byte stream — bit-stable across platforms and library versions."""
    return hashlib.shake_256(f"{case_id}|{w}".encode()).digest(n_bytes)


def _logits_numpy(case: dict[str, object]) -> np.ndarray:
    """Vectorised read of the pinned stream."""
    window = int(case["window"])
    n = len(wd.tile_windows(int(case["seq_len"]), window=window, stride=int(case["stride"])))
    out = np.empty((n, window, rc.NUM_CLASSES), dtype=np.float64)
    per_window = window * rc.NUM_CLASSES
    for w in range(n):
        raw = np.frombuffer(_stream(str(case["case_id"]), w, per_window * 8), dtype=">u8")
        unit = raw.astype(np.float64) / 2.0**64
        out[w] = (_LOGIT_LO + (_LOGIT_HI - _LOGIT_LO) * unit).reshape(window, rc.NUM_CLASSES)
    return out


def _logits_stdlib(case: dict[str, object]) -> list[list[list[float]]]:
    """Independent stdlib read of the same spec — no numpy, no frombuffer."""
    window = int(case["window"])
    n = len(wd.tile_windows(int(case["seq_len"]), window=window, stride=int(case["stride"])))
    out: list[list[list[float]]] = []
    for w in range(n):
        raw = _stream(str(case["case_id"]), w, window * rc.NUM_CLASSES * 8)
        rows: list[list[float]] = []
        for j in range(window):
            row: list[float] = []
            for c in range(rc.NUM_CLASSES):
                off = (j * rc.NUM_CLASSES + c) * 8
                unit = int.from_bytes(raw[off : off + 8], "big") / 2.0**64
                row.append(_LOGIT_LO + (_LOGIT_HI - _LOGIT_LO) * unit)
            rows.append(row)
        out.append(rows)
    return out


# --------------------------------------------------------------------------------------
# digest construction
# --------------------------------------------------------------------------------------

#: log-probs are quantised to 1e-6 before hashing: floating-point noise across libm
#: builds is ~1e-15, while ANY semantic change to the operator (dropping the coverage
#: normalisation, averaging raw logits, per-window arg-max) moves them by >> 1e-6.
_QUANT = 1_000_000


def _case_hash(
    case: dict[str, object],
    starts: list[int],
    log_probs: np.ndarray,
    prediction: np.ndarray,
    coverage: np.ndarray,
    zero_flanked: np.ndarray,
) -> str:
    quantised = np.rint(np.asarray(log_probs, dtype=np.float64) * _QUANT)
    return ingest.record_hash(
        [
            case["case_id"],
            case["seq_len"],
            case["window"],
            case["stride"],
            len(starts),
            ",".join(str(s) for s in starts),
            # big-endian everywhere: the digest must not depend on host byte order
            quantised.astype(">i8").tobytes().hex(),
            np.asarray(prediction, dtype=">i2").tobytes().hex(),
            np.asarray(coverage, dtype=">i4").tobytes().hex(),
            np.asarray(zero_flanked, dtype=np.uint8).tobytes().hex(),
        ]
    )


def _module_digest() -> str:
    """Path A — the production operator."""
    per_case: list[str] = []
    for case in _cases():
        starts = wd.tile_windows(
            int(case["seq_len"]), window=int(case["window"]), stride=int(case["stride"])
        )
        out = rc.reconcile_windows(_logits_numpy(case), starts, int(case["seq_len"]))
        per_case.append(
            _case_hash(case, starts, out.log_probs, out.prediction, out.coverage, out.zero_flanked)
        )
    return ingest.records_digest(sorted(per_case))


def _independent_reconcile(
    logits: list[list[list[float]]], starts: list[int], window: int, seq_len: int
) -> tuple[list[list[float]], list[int], list[int], list[int]]:
    """Path B — log(mean of per-window posteriors) then argmax, in stdlib `math`."""
    log_probs: list[list[float]] = []
    prediction: list[int] = []
    coverage: list[int] = []
    flagged: list[int] = []
    for p in range(seq_len):
        acc = [0.0] * rc.NUM_CLASSES
        n = 0
        flag = 0
        for k, start in enumerate(starts):
            if not (start <= p < start + window):
                continue
            n += 1
            if start < 0 or start + window > seq_len:
                flag = 1
            row = logits[k][p - start]
            m = max(row)
            denom = sum(math.exp(v - m) for v in row)
            for c, v in enumerate(row):
                acc[c] += math.exp(v - m) / denom  # softmax, one class at a time
        lp = [math.log(v / n) for v in acc]
        log_probs.append(lp)
        prediction.append(max(range(rc.NUM_CLASSES), key=lambda c: lp[c]))
        coverage.append(n)
        flagged.append(flag)
    return log_probs, prediction, coverage, flagged


def _independent_digest() -> str:
    per_case: list[str] = []
    for case in _cases():
        seq_len, window = int(case["seq_len"]), int(case["window"])
        starts = wd.tile_windows(seq_len, window=window, stride=int(case["stride"]))
        lp, pred, cov, flag = _independent_reconcile(_logits_stdlib(case), starts, window, seq_len)
        per_case.append(
            _case_hash(
                case, starts, np.asarray(lp), np.asarray(pred), np.asarray(cov), np.asarray(flag)
            )
        )
    return ingest.records_digest(sorted(per_case))


# --------------------------------------------------------------------------------------
# fixture integrity
# --------------------------------------------------------------------------------------


def test_fixture_present_and_well_formed() -> None:
    assert _GEOMETRY.exists() and _EXPECTED.exists()
    digest = _EXPECTED.read_text().strip()
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    cases = _cases()
    assert 10 <= len(cases) <= 200
    assert len({c["case_id"] for c in cases}) == len(cases)


def test_geometry_csv_is_byte_stable() -> None:
    """A silent input edit must not be launderable by regenerating `expected.sha256`."""
    assert hashlib.sha256(_GEOMETRY.read_bytes()).hexdigest() == _GEOMETRY_SHA256


def test_sampled_real_lengths_really_occur_in_the_sample() -> None:
    """The CSV's `source` column is a claim; check it against the real data (§10.2).

    Scope, stated precisely: this checks membership in the committed 56-row
    `context_sample.csv` (itself drawn from the P2-00 corpus), which is what a bare CI tier
    can read — `context_v0.parquet` is DVC-tracked and absent there. The `note` column
    likewise describes that **sample**, not the 23,535-record corpus: the corpus spans
    183-2598 nt (3 zero-length records; 279 under one window), so these rows are real
    lengths but not corpus extremes.
    """
    with _CORPUS_SAMPLE.open(newline="") as fh:
        real_lengths = {len(r["context_seq"]) for r in csv.DictReader(fh)}
    claimed = [c for c in _cases() if c["source"] == "sampled_real_context_length"]
    assert claimed, "the fixture must exercise real scan geometry, not only edge cases"
    for case in claimed:
        assert case["seq_len"] in real_lengths, case["case_id"]


def test_fixture_covers_the_material_geometry() -> None:
    """The cases must exercise every path that can drift, or the digest guards nothing."""
    cases = _cases()
    seen_coverage: set[int] = set()
    saw_zero_flank = saw_clean = saw_tail_anchor = False
    for case in cases:
        seq_len, window, stride = (int(case[k]) for k in ("seq_len", "window", "stride"))
        starts = wd.tile_windows(seq_len, window=window, stride=stride)
        cov = np.zeros(seq_len, dtype=int)
        for s in starts:
            cov[max(s, 0) : min(s + window, seq_len)] += 1
        seen_coverage |= set(cov.tolist())
        if any(s < 0 or s + window > seq_len for s in starts):
            saw_zero_flank = True
        else:
            saw_clean = True
        if len(starts) > 1 and (starts[-1] - starts[-2]) != stride:
            saw_tail_anchor = True
    assert {1, 2, 3} <= seen_coverage  # single, doubled, and tail-anchored triple cover
    assert saw_zero_flank and saw_clean and saw_tail_anchor
    assert any(int(c["window"]) == rc.WINDOW_NT for c in cases)  # the real pinned tiling


def test_the_synthetic_logit_stream_is_pinned() -> None:
    """Hard-pin the stream so a generator change cannot silently rewrite the golden."""
    first = _logits_numpy({"case_id": "edge_small", "seq_len": 100, "window": 40, "stride": 16})
    np.testing.assert_allclose(
        first[0, 0],
        [
            -1.8657277954300033,
            -4.350477216731877,
            -6.320289843893297,
            0.33021783543650507,
            -1.462264889925323,
            -2.2682246408444025,
            -1.9141518598436775,
            3.3213989636444587,
        ],
        rtol=0,
        atol=1e-12,
    )


def test_the_two_logit_readers_agree_bit_for_bit() -> None:
    """The shared input is not itself a source of divergence between the two paths."""
    case = {"case_id": "edge_small", "seq_len": 100, "window": 40, "stride": 16}
    assert np.array_equal(_logits_numpy(case), np.asarray(_logits_stdlib(case)))


def test_logit_stream_spans_the_declared_range() -> None:
    arr = _logits_numpy({"case_id": "real_01380", "seq_len": 1380, "window": 1024, "stride": 512})
    assert arr.min() >= _LOGIT_LO and arr.max() < _LOGIT_HI
    assert arr.min() < _LOGIT_LO + 0.5 and arr.max() > _LOGIT_HI - 0.5  # actually spans it


# --------------------------------------------------------------------------------------
# the golden itself — two independent paths against one committed digest
# --------------------------------------------------------------------------------------


def test_reconcile_digest_matches_expected() -> None:
    assert _module_digest() == _EXPECTED.read_text().strip()


def test_independent_reimplementation_agrees() -> None:
    assert _independent_digest() == _EXPECTED.read_text().strip()


def test_digest_is_deterministic() -> None:
    assert _module_digest() == _module_digest()


# --------------------------------------------------------------------------------------
# anti-tautology: the digest must move when the operator is sabotaged
# --------------------------------------------------------------------------------------


def _sabotaged_digest(mode: str) -> str:
    per_case: list[str] = []
    for case in _cases():
        seq_len, window = int(case["seq_len"]), int(case["window"])
        starts = wd.tile_windows(seq_len, window=window, stride=int(case["stride"]))
        logits = _logits_numpy(case)
        coverage = np.zeros(seq_len, dtype=np.int64)
        zero_flanked = np.zeros(seq_len, dtype=bool)
        acc = np.full((seq_len, rc.NUM_CLASSES), -np.inf)
        stack: list[tuple[int, int, np.ndarray]] = []
        for k, s in enumerate(starts):
            lo, hi = max(s, 0), min(s + window, seq_len)
            piece = logits[k, lo - s : hi - s]
            if mode != "raw_logits":
                piece = rc.log_softmax(piece, axis=-1)
            stack.append((lo, hi, piece))
            coverage[lo:hi] += 1
            if s < 0 or s + window > seq_len:
                zero_flanked[lo:hi] = True
            np.maximum(acc[lo:hi], piece, out=acc[lo:hi])
        total = np.zeros((seq_len, rc.NUM_CLASSES))
        for lo, hi, piece in stack:
            total[lo:hi] += np.exp(piece - acc[lo:hi])
        if mode == "no_coverage_norm":
            log_probs = acc + np.log(total)  # bare log-sum-exp: NO / coverage
        else:
            log_probs = acc + np.log(total / coverage[:, None])
        prediction = np.argmax(log_probs, axis=1).astype(np.int16)
        per_case.append(_case_hash(case, starts, log_probs, prediction, coverage, zero_flanked))
    return ingest.records_digest(sorted(per_case))


@pytest.mark.parametrize("mode", ["no_coverage_norm", "raw_logits"])
def test_digest_bites_a_sabotaged_operator(mode: str) -> None:
    """Both plausible misreadings of the D3 rule must fail the committed digest."""
    assert _sabotaged_digest(mode) != _EXPECTED.read_text().strip()


def test_the_sabotage_harness_reproduces_the_golden_when_not_sabotaged() -> None:
    """Anti-tautology partner: the sabotage differs ONLY in the sabotage, so a mismatch
    above is attributable to the operator change and not to a broken harness."""
    assert _sabotaged_digest("faithful") == _EXPECTED.read_text().strip()


def test_digest_moves_if_a_single_logit_shifts() -> None:
    case = _cases()[0]
    seq_len, window = int(case["seq_len"]), int(case["window"])
    starts = wd.tile_windows(seq_len, window=window, stride=int(case["stride"]))
    logits = _logits_numpy(case)
    baseline = rc.reconcile_windows(logits, starts, seq_len)
    logits[0, 0, 0] += 5.0
    mutated = rc.reconcile_windows(logits, starts, seq_len)
    assert not np.array_equal(baseline.log_probs, mutated.log_probs)
    assert _case_hash(
        case,
        starts,
        mutated.log_probs,
        mutated.prediction,
        mutated.coverage,
        mutated.zero_flanked,
    ) != _case_hash(
        case,
        starts,
        baseline.log_probs,
        baseline.prediction,
        baseline.coverage,
        baseline.zero_flanked,
    )


# --------------------------------------------------------------------------------------
# invariants asserted directly — the hash is not the only guard
# --------------------------------------------------------------------------------------


def test_every_case_reconciles_to_a_normalised_distribution() -> None:
    for case in _cases():
        seq_len, window = int(case["seq_len"]), int(case["window"])
        starts = wd.tile_windows(seq_len, window=window, stride=int(case["stride"]))
        out = rc.reconcile_windows(_logits_numpy(case), starts, seq_len)
        assert out.log_probs.shape == (seq_len, rc.NUM_CLASSES), case["case_id"]
        np.testing.assert_allclose(np.exp(out.log_probs).sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert int(out.coverage.min()) >= 1
        assert np.array_equal(out.prediction, np.argmax(out.log_probs, axis=1).astype(np.int16))


def test_zero_flank_fires_exactly_on_the_contigs_shorter_than_one_window() -> None:
    """Under tail-anchored tiling the flag is all-or-nothing, and `seq_len < window` is
    the only way to get it: `seq_len == window` fits exactly and is NOT zero-flanked."""
    saw_short = saw_exact = saw_long = False
    for case in _cases():
        seq_len, window = int(case["seq_len"]), int(case["window"])
        starts = wd.tile_windows(seq_len, window=window, stride=int(case["stride"]))
        out = rc.reconcile_windows(_logits_numpy(case), starts, seq_len)
        if seq_len < window:
            assert bool(out.zero_flanked.all()), case["case_id"]
            saw_short = True
        else:
            assert not bool(out.zero_flanked.any()), case["case_id"]
            saw_exact |= seq_len == window
            saw_long |= seq_len > window
    assert saw_short and saw_exact and saw_long
