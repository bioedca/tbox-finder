"""Mined-negative injection into the Stage-1 training stream (P2-10d).

PRD §9.1 pins the Stage-1 negative curriculum: sample windows at a fixed moderate
**~10:1 seed ratio**, then **iteratively mine Stage-1 false positives** as hard
negatives. ADR-0005 **D14** pins the guards between a false positive and the mined pool
(union-prior locus masking, then the three-valued spare rule). P2-10a built the scanner,
P2-10b the coordinate-bearing substrate, P2-10c the first protective spare-rule backend.
This module is the missing **training-side hook**: it turns a mined-negative pool into
:class:`~tbox_finder.data.window_dataset.CorpusRecord` objects the P2-01 datamodule can
already window, and mixes them into the draw stream at a fraction that is **measured off
the emitted stream, never echoed from the request**.

Three rules govern it, and each is a refusal rather than a repair.

1. **A negative is exactly one honest window of real DNA.** `window_dataset` paints every
   *real* flank position `background` and every *zero-flanked* position `IGNORE_INDEX`
   (its rules 1–2). A mined negative shorter than the window could only be made to fit by
   padding, which asserts a contig boundary that does not exist and hands the model a
   trivially separable "PAD-heavy window ⇒ background" cue. So
   :func:`background_record` **requires ``len(sequence) == window``** and raises
   otherwise, naming the shortfall. It never pads, never tiles, never concatenates two
   non-contiguous windows into one.

2. **Negatives live in a private stratum and cluster namespace.** The PRD §11 curriculum
   weights are inverse-frequency over `(phylum, klass, cognate_aa)`, computed by
   ``Counter`` over *all* records in one pool. Landing negatives in an existing bucket —
   including the `UNKNOWN` bucket that real positives with unresolved lineage already
   occupy — changes those positives' counts and therefore their weights. With the private
   keys below, every positive-vs-positive relative weight is preserved exactly (asserted
   in ``tests/unit/test_negatives.py``). ``cluster_id`` is negative for the same reason:
   positive homology-cluster ids are non-negative, so a negative's id can never be swept
   into (or out of) the P2-06a selection-val carve or counted as a bootstrap block.

3. **The mix ratio is a property of the emitted stream, not of the config.** The
   curriculum weights do *not* give the ratio you configured: a single-stratum negative
   pool of size N contributes raw mass ``N**(1 - 3*alpha)`` — at the shipped
   ``alpha = 0.25`` that is ``N**0.25``, so a 10x larger pool moves the realized share by
   only ~1.78x. Reading the ratio off the pool size would therefore be a config echo of a
   number the sampler contradicts — the recurring `basis_point` defect. So
   :class:`MixedIndexSampler` does **not** route negatives through the curriculum at all:
   it draws exactly :func:`negative_draw_count` of them, interleaves, and
   :meth:`MixedIndexSampler.mix_summary` **counts the emitted keys** to report the
   realized fraction. The gate is an exact integer identity re-derived from the recorded
   counts, so it carries no tolerance knob to loosen.

**What this module does not do.** It does not run a mining round (P2-10e), does not
decide which candidates are minable (that is `mining/hard_negative.py` + the D14 spare
rule), and does not source the pool. :func:`load_negative_records` reads whatever pool it
is pointed at, applies the geometry contract, and **reports every refusal with its
reason** — so a pool that cannot supply window-length negatives produces a report saying
so and zero records, not a quietly smaller run.

Bare-CI importable: pandas is imported lazily, so the geometry and mixing tiers run
stdlib+numpy-only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder.data.window_dataset import (
    DEFAULT_SEED,
    WINDOW_NT,
    CorpusRecord,
    WeightedIndexSampler,
)

SCHEMA_VERSION = "1.0"
STEP = "P2-10d"

# ── The private negative namespace (rule 2) ──────────────────────────────────────────
#: Stratum keys for injected negatives. Deliberately not ``UNKNOWN``: that bucket is
#: already populated by real positives whose phylum / specifier AA could not be resolved
#: (``window_dataset._normalise_stratum``), and sharing it would move *their* weights.
#: Deliberately not ``"I"`` / ``"II"`` either — a negative has no T-box class.
NEGATIVE_KLASS = "NEGATIVE"
NEGATIVE_PHYLUM = "NEGATIVE"
NEGATIVE_AA = "NEGATIVE"

#: Negatives take ``cluster_id = -(1 + ordinal)``. Positive homology-cluster ids are
#: non-negative integers assigned by ``splits.py``, so the two namespaces cannot collide
#: and a negative can never be drawn into the P2-06a selection-val cluster carve nor
#: counted as an ADR-0005 D5 bootstrap block.
NEGATIVE_CLUSTER_SIGN = -1

#: The six fold-scheme values a negative carries. Every entry is ``None`` except
#: ``nested_train`` — a mined negative belongs to no split scheme, and stating ``None``
#: is the honest answer where stating ``"train"`` would let a leakage check believe it had
#: audited something. ``nested_train=True`` is the one real claim: the negative is part of
#: the training stream. Ordered as ``window_dataset.FOLD_SCHEME_COLUMNS``.
NEGATIVE_FOLDS: tuple[Any, ...] = (None, None, None, None, True, None)

#: ``record_id`` prefix, so a negative is identifiable in any provenance row, digest, or
#: W&B table without a join.
NEGATIVE_ID_PREFIX = "neg"

# ── Pool-column names (mining/pool.py's artifact schema) ─────────────────────────────
CANDIDATE_ID_COL = "candidate_id"
SEQUENCE_COL = "sequence"
MASKED_COL = "masked"
POOL_COL = "pool"
CONTROL_COL = "is_designed_control"

#: Refusal reasons recorded by :func:`load_negative_records`. Each is a *count*, not a
#: silent drop: a pool that supplies nothing must say which contract it failed.
REASON_MASKED = "masked_known_locus"
REASON_DESIGNED_CONTROL = "designed_control"
REASON_TOO_SHORT = "shorter_than_window"
REASON_TOO_LONG = "longer_than_window"
REASON_NON_ACGTN = "non_acgtn_characters"
REASON_EMPTY_ID = "missing_candidate_id"

_ACGTN = frozenset("ACGTN")

#: Salt for the mix shuffle's RNG key, so the interleave stream is independent of both
#: the curriculum draw (keyed ``[seed, epoch]``) and the per-window augmentation draw
#: (keyed ``[seed, epoch, index, occurrence]``). Without it the shuffle would share a
#: stream with the positive draw and the two would be correlated across epochs.
MIX_STREAM_SALT = 0x10D


class NegativeInjectionError(ValueError):
    """Raised when a negative cannot be built honestly, or a mix cannot be realized."""


# ═════════════════════════════════════════════════════════════════════════════════════
# The all-background record constructor
# ═════════════════════════════════════════════════════════════════════════════════════
def background_record(
    *,
    record_id: str,
    sequence: str,
    cluster_id: int,
    window: int = WINDOW_NT,
) -> CorpusRecord:
    """Build one all-background :class:`CorpusRecord` from a window-length negative.

    The record declares a *pseudo-locus* co-extensive with the whole sequence whose
    ``label_string`` is all ``background``. That is not a trick: ``carve_window`` already
    paints real flank ``background`` and then lets the locus paint its own classes, so a
    locus that paints background yields a window every one of whose 1024 positions is a
    real nucleotide labelled ``background``, with **no** ``IGNORE_INDEX`` and **no**
    ``[PAD]``. It is the same code path a positive takes, which is the point — a negative
    that reached the model through a different carve would differ from a positive in ways
    the label does not name.

    ``len(sequence)`` must equal ``window`` **exactly**.

    * Shorter is refused because the only way to fill the window is padding, and padding
      asserts a contig boundary that does not exist (``window_dataset`` honesty rule 1)
      while teaching "``[PAD]``-heavy window ⇒ background" — a cue absent from the scan
      distribution except at genuine contig ends.
    * Longer is refused because it would silently discard sequence: with a pseudo-locus of
      length ``L < window`` the honest lead range is ``min(o, window-L, C-window, C-o-L)``
      wide, so the emitted window would cover an unpredictable sub-span of the mined
      region while the record's id still claimed the whole of it. A caller that wants
      sub-windows must carve them itself, and say so in the ids.

    ``clipped_start`` / ``clipped_end`` are both ``False``: a mined negative is an
    interior genomic window, and asserting a contig end would buy lead freedom by lying
    (it lets the carve step off the sequence). The consequence is deliberate — the record
    admits exactly one lead, so offset augmentation is a no-op for negatives. Strand
    augmentation still applies (``both_strands`` draws the reverse complement), which is
    the augmentation that matters here: a negative is one specific hard window, and
    re-phasing it would slide the frame into DNA that was never mined or masked.
    """
    if window <= 0:
        raise NegativeInjectionError(f"window must be positive; got {window}")
    rid = str(record_id).strip()
    if not rid:
        raise NegativeInjectionError("record_id must be a non-empty string")
    # Upper-casing is deliberate and lossless: soft-masking is a repeat *annotation*, not a
    # different nucleotide, and every genomic FASTA source in this project may deliver it.
    # It is the one normalisation applied — everything else outside ACGTN is refused below,
    # because `encode_bases` would map it to the N id without a word.
    seq = str(sequence).upper()
    if len(seq) != window:
        raise NegativeInjectionError(
            f"negative {rid!r}: sequence is {len(seq)} nt but the window is {window} nt. "
            "A negative must be exactly one window of real DNA — padding a short one "
            "would assert a contig boundary that does not exist (window_dataset honesty "
            "rule 1), and truncating a long one would emit an unnamed sub-span. Carve the "
            "pool at the training window width instead."
        )
    bad = sorted(set(seq) - _ACGTN)
    if bad:
        raise NegativeInjectionError(
            f"negative {rid!r}: sequence carries non-ACGTN characters {bad!r}. "
            "encode_bases maps every unknown character to the N id silently, so a "
            "soft-masked or IUPAC-coded window would train as a run of Ns without a "
            "single warning."
        )
    if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
        raise NegativeInjectionError(
            f"negative {rid!r}: cluster_id must be an int; got {cluster_id!r}"
        )
    if cluster_id >= 0:
        raise NegativeInjectionError(
            f"negative {rid!r}: cluster_id must be negative (the private namespace); got "
            f"{cluster_id}. A non-negative id collides with the positive homology clusters "
            "and can be swept into the P2-06a selection-val carve."
        )
    return CorpusRecord(
        record_id=rid,
        context_seq=seq,
        locus_offset=0,
        locus_length=window,
        label_string="." * window,
        clipped_start=False,
        clipped_end=False,
        klass=NEGATIVE_KLASS,
        phylum=NEGATIVE_PHYLUM,
        cognate_aa=NEGATIVE_AA,
        cluster_id=cluster_id,
        nested_train=True,
        # A mined negative is not in the leave-one-order-out holdout: it is not a corpus
        # positive at all, and `splits.py`'s indicator is `(order in heldout) and is_corpus`.
        # Stated explicitly because the field has no default, by design.
        is_designated_loo_holdout=False,
        folds=NEGATIVE_FOLDS,
    )


def is_negative_record(record: CorpusRecord) -> bool:
    """Whether ``record`` came from :func:`background_record`.

    Keyed on the **cluster namespace**, not on the id prefix: the prefix is cosmetic and a
    caller can pass any ``record_id``, whereas a negative ``cluster_id`` is enforced at
    construction and is the property the split/bootstrap machinery actually reads.
    """
    return int(record.cluster_id) < 0


def negative_records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = WINDOW_NT,
    id_prefix: str = NEGATIVE_ID_PREFIX,
    skip_masked: bool = True,
    skip_designed_controls: bool = True,
    max_records: int | None = None,
) -> tuple[list[CorpusRecord], dict[str, Any]]:
    """Turn mined-pool rows into negatives, counting every refusal by reason.

    Returns ``(records, report)``. The report's ``excluded_by_reason`` is the whole point:
    a pool whose windows are the wrong width yields **zero** records and a report that
    names the width, rather than a training run that silently shrinks.

    ``skip_masked`` drops rows the union-prior locus mask flagged (ADR-0005 D14's first
    guard); ``skip_designed_controls`` drops P2-10b's locus-centred control windows, which
    overlap a known T-box **by construction** and exist to prove the mask fires — training
    on them would be label poisoning of the most direct kind. Both default on and both are
    counted, so a pool loaded with either disabled says so in its own report.
    """
    if window <= 0:
        raise NegativeInjectionError(f"window must be positive; got {window}")
    records: list[CorpusRecord] = []
    excluded: dict[str, int] = {}
    lengths: dict[int, int] = {}
    n_rows = 0
    seen_ids: set[str] = set()

    def _drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for row in rows:
        n_rows += 1
        seq = str(row.get(SEQUENCE_COL) or "").upper()
        lengths[len(seq)] = lengths.get(len(seq), 0) + 1
        cid = str(row.get(CANDIDATE_ID_COL) or "").strip()
        if not cid:
            _drop(REASON_EMPTY_ID)
            continue
        if skip_masked and bool(row.get(MASKED_COL)):
            _drop(REASON_MASKED)
            continue
        if skip_designed_controls and bool(row.get(CONTROL_COL)):
            _drop(REASON_DESIGNED_CONTROL)
            continue
        if len(seq) < window:
            _drop(REASON_TOO_SHORT)
            continue
        if len(seq) > window:
            _drop(REASON_TOO_LONG)
            continue
        if set(seq) - _ACGTN:
            _drop(REASON_NON_ACGTN)
            continue
        if cid in seen_ids:
            raise NegativeInjectionError(
                f"duplicate candidate_id {cid!r} in the negative pool — a duplicated hard "
                "negative is silently oversampled, and the mix ratio would describe a pool "
                "that does not exist"
            )
        seen_ids.add(cid)
        records.append(
            background_record(
                record_id=f"{id_prefix}:{cid}",
                sequence=seq,
                cluster_id=NEGATIVE_CLUSTER_SIGN * (len(records) + 1),
                window=window,
            )
        )
        if max_records is not None and len(records) >= max_records:
            break

    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "window_nt": int(window),
        "n_rows_read": int(n_rows),
        "n_records": len(records),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "n_excluded": int(sum(excluded.values())),
        "skip_masked": bool(skip_masked),
        "skip_designed_controls": bool(skip_designed_controls),
        "max_records": None if max_records is None else int(max_records),
        # The measurement that explains a zero-record pool without anyone having to guess:
        # the width distribution the pool actually carries, against the width required.
        "sequence_length_counts": {str(k): int(v) for k, v in sorted(lengths.items())},
        "n_rows_at_window_length": int(lengths.get(window, 0)),
    }
    return records, report


def load_negative_records(
    parquet_path: str | Path,
    *,
    window: int = WINDOW_NT,
    max_records: int | None = None,
    skip_masked: bool = True,
    skip_designed_controls: bool = True,
) -> tuple[list[CorpusRecord], dict[str, Any]]:
    """Read a mined-negative parquet and build the negative records it can honestly supply.

    The pool schema is `mining/pool.py`'s: ``candidate_id``, ``sequence``, ``masked``,
    ``is_designed_control`` (extra columns are ignored). Rows arrive in file order and the
    resulting ``cluster_id``s are assigned in that order, so the same file yields the same
    records — no shuffling here, the mix owns that.
    """
    import pandas as pd  # lazy: keeps the geometry tier bare-CI importable

    path = Path(parquet_path)
    if not path.exists():
        raise NegativeInjectionError(f"negative pool not found: {path}")
    frame = pd.read_parquet(path)
    missing = [c for c in (CANDIDATE_ID_COL, SEQUENCE_COL) if c not in frame.columns]
    if missing:
        raise NegativeInjectionError(
            f"negative pool {path} is missing required column(s) {missing}; found "
            f"{sorted(frame.columns)}"
        )
    for optional in (MASKED_COL, CONTROL_COL):
        if optional not in frame.columns:
            raise NegativeInjectionError(
                f"negative pool {path} has no {optional!r} column. Loading without it "
                "would treat every row as unmasked / not-a-control, i.e. fail OPEN on the "
                "ADR-0005 D14 guard the column exists to record."
            )
    records, report = negative_records_from_rows(
        frame.to_dict("records"),
        window=window,
        max_records=max_records,
        skip_masked=skip_masked,
        skip_designed_controls=skip_designed_controls,
    )
    report["source_parquet"] = str(path)
    report["pools"] = (
        {str(k): int(v) for k, v in sorted(frame[POOL_COL].value_counts().items())}
        if POOL_COL in frame.columns
        else {}
    )
    return records, report


# ═════════════════════════════════════════════════════════════════════════════════════
# The mix
# ═════════════════════════════════════════════════════════════════════════════════════
def negative_draw_count(n_positive_draws: int, negative_fraction: float) -> int:
    """How many negative draws make the emitted stream ``negative_fraction`` negative.

    Solves ``n / (n_pos + n) == f`` for the integer ``n``, i.e.
    ``n = round(n_pos * f / (1 - f))``. Exposed (and re-derived by the gate) rather than
    inlined so the clause and the sampler cannot disagree: the clause recomputes this from
    the *recorded* positive-draw count and compares it to the *counted* negative keys.

    ``f`` is the share of the **emitted stream**, not a pool-size ratio — PRD §9.1's
    "~10:1" is ``f = 10/11 ≈ 0.909``.
    """
    if not isinstance(negative_fraction, (int, float)) or isinstance(negative_fraction, bool):
        raise NegativeInjectionError(
            f"negative_fraction must be a real number; got {negative_fraction!r}"
        )
    f = float(negative_fraction)
    if not 0.0 <= f < 1.0:
        raise NegativeInjectionError(
            f"negative_fraction must lie in [0, 1); got {f}. 1.0 would ask for a stream "
            "with no positives at all, which is not a mix."
        )
    if not isinstance(n_positive_draws, int) or isinstance(n_positive_draws, bool):
        raise NegativeInjectionError(f"n_positive_draws must be an int; got {n_positive_draws!r}")
    if n_positive_draws < 0:
        raise NegativeInjectionError(f"n_positive_draws must be >= 0; got {n_positive_draws}")
    if f == 0.0:
        return 0
    # Banker's rounding would make the count depend on parity; round half up so the
    # identity the gate re-derives is the same one on both sides.
    exact = n_positive_draws * f / (1.0 - f)
    return int(np.floor(exact + 0.5))


def positive_only_sampler(
    dataset: Any, *, n_positive: int, num_samples: int | None = None
) -> WeightedIndexSampler:
    """The PRD §11 curriculum sampler over a mixed dataset's **positive prefix** only.

    ``Stage1WindowDataset.sampler()`` weights every record it holds, so calling it on a
    dataset that also carries negatives would route the negatives through the curriculum —
    the exact coupling :class:`MixedIndexSampler` exists to avoid, and one that would make
    the realized mix a sublinear function of pool size. Restricting the weights to the
    prefix removes the coupling *by construction* rather than by relying on the negatives'
    private stratum keys.

    The prefix boundary is **verified, not assumed**: every record below ``n_positive``
    must not be a negative and every record at or above it must be. An off-by-one here
    would silently train on a positive as an all-background window, or weight a negative by
    a phylum it does not have — neither of which any downstream count would reveal.
    """
    from tbox_finder.data.window_dataset import curriculum_weights

    records = list(dataset.records)
    n = int(n_positive)
    if not 0 < n <= len(records):
        raise NegativeInjectionError(
            f"n_positive={n} is outside the dataset's {len(records)} records"
        )
    misplaced_positive = [r.record_id for r in records[:n] if is_negative_record(r)]
    misplaced_negative = [r.record_id for r in records[n:] if not is_negative_record(r)]
    if misplaced_positive or misplaced_negative:
        raise NegativeInjectionError(
            "the positive/negative boundary is not at n_positive="
            f"{n}: {len(misplaced_positive)} negative(s) below it "
            f"{misplaced_positive[:3]}, {len(misplaced_negative)} positive(s) above it "
            f"{misplaced_negative[:3]}"
        )
    prefix = records[:n]
    cfg = dataset.config
    weights = curriculum_weights(
        phyla=[r.phylum for r in prefix],
        klasses=[r.klass for r in prefix],
        amino_acids=[r.cognate_aa for r in prefix],
        phylum_alpha=cfg.phylum_alpha,
        klass_alpha=cfg.klass_alpha,
        aa_alpha=cfg.aa_alpha,
    )
    return WeightedIndexSampler(
        weights, num_samples=num_samples if num_samples is not None else n, seed=cfg.seed
    )


class MixedIndexSampler:
    """Interleave a positive draw stream with uniform negative draws at a set fraction.

    Yields the same ``(index, occurrence)`` keys :class:`WeightedIndexSampler` does, over
    **one** index space: the dataset holds positives at ``[0, n_positive)`` and negatives
    at ``[n_positive, n_positive + n_negative)``, so ``Stage1WindowDataset`` needs no
    negative-awareness at all and every window — positive or negative — takes the same
    carve, the same augmentation RNG and the same collate.

    Negatives are drawn **uniformly, outside the PRD §11 curriculum**, and that is a
    decision with a measured reason. The curriculum's three inverse-frequency terms
    multiply over ``(phylum, klass, cognate_aa)``; a negative pool sharing one bucket on
    all three axes contributes raw mass ``N**(1 - 3*alpha)`` — ``N**0.25`` at the shipped
    ``alpha = 0.25``. Routing negatives through it would make the realized ratio a
    sublinear function of pool size that no config states, so the configured number and
    the trained stream would disagree by construction. Negatives carry no phylum and no
    T-box class, so there is nothing for the curriculum to balance in the first place.

    ``occurrence`` for a negative is its ordinal **within the negative draw stream**, so a
    negative drawn twice in an epoch still gets two independent strand draws (the same
    property ``WeightedIndexSampler`` supplies for oversampled positives).
    """

    def __init__(
        self,
        positive: WeightedIndexSampler,
        *,
        n_positive_records: int,
        n_negative_records: int,
        negative_fraction: float,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if n_positive_records <= 0:
            raise NegativeInjectionError("n_positive_records must be positive")
        if n_negative_records < 0:
            raise NegativeInjectionError("n_negative_records must be >= 0")
        self.positive = positive
        self.n_positive_records = int(n_positive_records)
        self.n_negative_records = int(n_negative_records)
        self.negative_fraction = float(negative_fraction)
        self.n_negative_draws = negative_draw_count(len(positive), self.negative_fraction)
        if self.n_negative_draws > 0 and self.n_negative_records == 0:
            raise NegativeInjectionError(
                f"negative_fraction={self.negative_fraction} asks for "
                f"{self.n_negative_draws} negative draws but the pool is EMPTY. Refusing "
                "rather than training a run that reports a mix it never sampled — an "
                "empty pool must fail loudly, not degrade to positives-only."
            )
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance both streams' epochs together (either alone would freeze the other)."""
        self._epoch = int(epoch)
        self.positive.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.positive) + self.n_negative_draws

    def __iter__(self) -> Iterator[tuple[int, int]]:
        keys: list[tuple[int, int]] = list(self.positive)
        if self.n_negative_draws:
            rng = np.random.default_rng([self.seed, self._epoch, MIX_STREAM_SALT])
            drawn = rng.integers(0, self.n_negative_records, size=self.n_negative_draws)
            keys.extend((self.n_positive_records + int(j), occ) for occ, j in enumerate(drawn))
            # Shuffle the concatenation so negatives are spread across the epoch rather
            # than arriving as one tail block: a tail block would make the last batches
            # all-negative, and with DDP sharding it would also hand entire ranks a
            # single-class stream.
            order = np.random.default_rng(
                [self.seed, self._epoch, MIX_STREAM_SALT + 1]
            ).permutation(len(keys))
            keys = [keys[i] for i in order]
        return iter(keys)

    # ── The measurement ──────────────────────────────────────────────────────────────
    def mix_summary(self, keys: Sequence[Any] | None = None) -> dict[str, Any]:
        """Count the emitted stream's composition — the number the gate reads.

        ``keys`` defaults to this sampler's own full emission; pass a truncated key list
        (what DDP sharding + ``steps_per_epoch`` actually leave) to measure what a rank
        consumed. Either way the counts come from **classifying the keys**, never from
        ``negative_fraction`` or from the pool size.
        """
        emitted = list(self) if keys is None else list(keys)
        n_total = len(emitted)
        n_negative = sum(1 for k in emitted if _key_index(k) >= self.n_positive_records)
        return {
            "n_total": int(n_total),
            "n_negative": int(n_negative),
            "n_positive": int(n_total - n_negative),
            "realized_negative_fraction": (float(n_negative / n_total) if n_total else None),
            "requested_negative_fraction": float(self.negative_fraction),
            "negative_pool_size": int(self.n_negative_records),
            "positive_pool_size": int(self.n_positive_records),
        }


def _key_index(key: Any) -> int:
    """The dataset index of a sampler key (``(index, occurrence)`` or a bare index)."""
    if isinstance(key, tuple):
        return int(key[0])
    return int(key)


def mix_clauses(summary: Mapping[str, Any] | None, *, negative_fraction: float) -> dict[str, bool]:
    """Re-derive the negative-mix gate clauses from a recorded :meth:`mix_summary`.

    Shared by the report builder and its validator so the two cannot disagree, and
    **total**: every clause is wrapped so a *missing* measurement is FALSE rather than
    vacuously TRUE (the P2-05 defect — a clause derived from the requested config is true
    exactly when the evidence is absent).

    The requested-zero branch is asserted, not skipped: ``negative_fraction == 0`` must
    come with a recorded stream that carries **no** negative draws and **no** pool. A run
    that configured no negatives and silently loaded some would fail here.
    """
    f = float(negative_fraction) if isinstance(negative_fraction, (int, float)) else float("nan")
    # A missing/malformed summary becomes an empty mapping, and every field then reads as
    # `None`, which `_is_count` rejects — so absence propagates to FALSE through the count
    # check rather than through a separate presence flag. One path, not two that could
    # disagree: a `have` flag alongside this would be dead weight that *looked* load-bearing.
    summary = summary if isinstance(summary, Mapping) else {}
    n_total = summary.get("n_total")
    n_neg = summary.get("n_negative")
    n_pos = summary.get("n_positive")
    pool = summary.get("negative_pool_size")
    ints_ok = all(_is_count(v) for v in (n_total, n_neg, n_pos, pool))
    consistent = bool(ints_ok) and n_total == n_neg + n_pos and n_total > 0
    # The exact identity: the counted negative keys must equal the count re-derived from
    # the counted positive keys and the requested fraction. No tolerance — the sampler
    # draws an integer, so the gate compares integers.
    try:
        expected = negative_draw_count(int(n_pos), f) if consistent else None
    except NegativeInjectionError:
        expected = None
    requested_zero = consistent and f == 0.0
    return {
        "negative_mix_measured": bool(consistent),
        "negative_mix_matches_request": bool(
            consistent and expected is not None and n_neg == expected
        ),
        # Requested > 0 ⇒ the pool and the draws must both be non-empty; requested == 0 ⇒
        # both must be exactly zero. Either way the clause reads the evidence.
        "negative_mix_pool_consistent": bool(
            consistent
            and ((pool == 0 and n_neg == 0) if requested_zero else (pool > 0 and n_neg > 0))
        ),
    }


def _is_count(value: Any) -> bool:
    """A non-negative int that is not a bool (``isinstance(True, int)`` is True)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# ═════════════════════════════════════════════════════════════════════════════════════
# The substrate audit (CLI)
# ═════════════════════════════════════════════════════════════════════════════════════
DEFAULT_MINING_POOL = "data/processed/negatives/mining_pool_v0.parquet"
DEFAULT_CONTEXT_PARQUET = "data/interim/flank_context/context_v0.parquet"
DEFAULT_AUDIT_REPORT = "reports/p2/negative_injection.json"

#: Margins the flank-capacity measurement is reported at. 50 is `mining/pool.py`'s
#: ``DEFAULT_FLANK_MARGIN_NT`` **and** `masking.DEFAULT_FLANK_NT`, which is not a
#: coincidence: the carve margin exists so a flank window does not sit inside the
#: union-prior mask flank of its own parent locus. 0 is reported alongside it to separate
#: "no room at all" from "no room once D14's flank is honoured".
AUDIT_MARGINS_NT: tuple[int, ...] = (0, 50)


def flank_capacity(
    context_parquet: str | Path = DEFAULT_CONTEXT_PARQUET,
    *,
    window: int = WINDOW_NT,
    margins: Sequence[int] = AUDIT_MARGINS_NT,
) -> dict[str, Any]:
    """How many ``window``-nt flank windows the P2-00 context regions could supply.

    This is the measurement that turns "the substrate does not exist yet" from an
    assertion into a number. A mined negative must be exactly one training window of real
    DNA (:func:`background_record`), so the pool has to be carved at the training width —
    and whether that is *possible* is a property of how much real flank P2-00 fetched, not
    of anything this module can decide.

    Reported per margin because the margin is load-bearing: a flank window carved with no
    margin abuts its own parent locus, so it falls inside that locus's union-prior mask
    flank and ADR-0005 D14's first guard removes it. Capacity at margin 0 without capacity
    at the D14 flank is capacity that cannot be mined.
    """
    import pandas as pd  # lazy

    frame = pd.read_parquet(context_parquet)
    ok = frame[frame["status"] == "ok"]
    context_len = ok["context_seq"].str.len()
    lead = ok["locus_offset"].astype(int)
    trail = context_len.astype(int) - lead - ok["locus_length"].astype(int)
    per_margin = {}
    for margin in margins:
        need = int(window) + int(margin)
        per_margin[str(int(margin))] = {
            "required_flank_nt": need,
            "n_rows_with_lead_flank": int((lead >= need).sum()),
            "n_rows_with_trail_flank": int((trail >= need).sum()),
            "n_windows_carvable": int((lead >= need).sum() + (trail >= need).sum()),
        }
    return {
        "context_parquet": str(context_parquet),
        "window_nt": int(window),
        "n_rows": int(len(frame)),
        "n_rows_anchored": int(len(ok)),
        "max_lead_flank_nt": int(lead.max()) if len(ok) else 0,
        "max_trail_flank_nt": int(trail.max()) if len(ok) else 0,
        "by_margin_nt": per_margin,
    }


def audit_pool(
    parquet_path: str | Path = DEFAULT_MINING_POOL,
    *,
    window: int = WINDOW_NT,
    context_parquet: str | Path | None = DEFAULT_CONTEXT_PARQUET,
) -> dict[str, Any]:
    """Measure what a pool can supply at ``window``, without building a training run.

    This is the honest answer to "can P2-10e mine into training today?" — it reports the
    width distribution the pool carries, the count that survives each contract clause, and
    (when the P2-00 context is available) whether a wider pool could be carved at all. A
    zero is then a measurement with a named cause rather than an assertion.
    """
    _records, report = load_negative_records(parquet_path, window=window)
    if context_parquet is not None and Path(context_parquet).exists():
        report["flank_capacity"] = flank_capacity(context_parquet, window=window)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m tbox_finder.data.negatives`` — write the substrate audit."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit a mined-negative pool for injection.")
    parser.add_argument("--pool", default=DEFAULT_MINING_POOL)
    parser.add_argument("--context", default=DEFAULT_CONTEXT_PARQUET)
    parser.add_argument("--window", type=int, default=WINDOW_NT)
    parser.add_argument("--out", default=DEFAULT_AUDIT_REPORT)
    args = parser.parse_args(argv)

    report = audit_pool(args.pool, window=args.window, context_parquet=args.context)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}: {report['n_records']} injectable negatives at window={args.window}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
