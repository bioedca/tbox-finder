"""Stage-1 windowed per-nucleotide segmentation datamodule + curriculum sampling (P2-01).

PRD §6 pins the Stage-1 scanner to a **1024-nt window / 512-nt stride** over genomic
contigs; PRD §11 pins the training curriculum: rare-class oversampling (class II +
rare specifier amino acids) **constrained to training-fold parents**, **window-phase
offset augmentation** of training positives, and **balanced-phylum sampling** to avoid
the Firmicutes overfit (§12). This module builds the dataset those pins describe.

The window substrate is **real genomic context** (`data/interim/flank_context/
context_v0.parquet`, P2-00): each record carries `context_seq` — the locus embedded in
up to ±1024 nt of *real* NCBI flank — and `locus_offset`, the 0-based index of the
locus inside it. `label_string` (P0-20, `labels_v0.parquet`) is **co-extensive with the
locus**, so within a window the labels occupy `[locus_offset, locus_offset +
locus_length)` and **every real-DNA flank position is `background`** — never
`IGNORE_INDEX`. That is deliberate: the flank is real DNA, and real background is
exactly what the scanner must learn to reject (imp.md P2-01; PRD §9.1).

Three honesty rules govern window construction (CLAUDE.md §10.3; PRD §5/§10.3 — a
synthesised flank hands the model a trivially separable boundary cue and would inflate
GATE-4/GATE-1):

1. **A window never steps outside `context_seq` onto invented sequence.** It may extend
   past an end only when that end is a *real contig boundary* — i.e. only when P2-00's
   `clipped_start` / `clipped_end` says so. Those positions are zero-flanked (`[PAD]`)
   and flagged, per PRD §6.
2. **Zero-flanked positions are labelled `IGNORE_INDEX`, not `background`.** A padded
   position carries no DNA; training it as background would teach the model that "no
   sequence" is a negative. Only *real* flank is background (rule above).
3. **Records whose geometry admits no honest window are excluded and counted**, never
   patched with synthetic flank.

Scope (imp.md P2-01 `Outputs:`): this module ships the **positive**-window dataset, the
offset-augmentation and reverse-complement transforms, and the three samplers. The
§9.1 decoy/negative pools are **not** mixed in here — the seed ~10:1 negative ratio and
hard-negative mining are P2-07/P2-09, and the P0 `gc_background` pool is a 0th-order
composition null with no genomic context (`data/processed/audits/decoys_report.json`),
so splicing it into a 1024-nt window would be the very cue rule 1 exists to prevent.

Sampling weights are **implementer choices** (`WEIGHTS_PINNED = False`): PRD §11 and
ADR-0004/ADR-0005 pin *that* the three samplers exist, not their strength. The defaults
(`alpha = 0.25` per axis) were **chosen on a measured sweep over the real training
fold**, not assumed — see :data:`DEFAULT_PHYLUM_ALPHA` for the table and the
concentration pathology that rules the larger values out. `conf/data/stage1.yaml` echoes
them and a drift-guard test asserts config == code (the ADR-0002 A5 precedent).

Bare-CI importable: pandas and torch are imported lazily inside the functions that need
them, so `tests/unit/test_window_dataset.py` runs stdlib+numpy-only in CI.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder import ingest
from tbox_finder import labels as labels_mod
from tbox_finder import splits as splits_mod

# ── Schema / provenance ──────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"
STEP = "P2-01"

# ── Window geometry (PRD §6) ─────────────────────────────────────────────────
WINDOW_NT = 1_024
STRIDE_NT = 512

# ── Label / token conventions ────────────────────────────────────────────────
# -100 is the seg-head cross-entropy ignore value (ADR-0005); duplicated here
# rather than imported so this module stays bare-importable, with a drift-guard
# test asserting it equals ``seg_smoke.IGNORE_INDEX`` (the repo-wide pattern).
IGNORE_INDEX = -100
BACKGROUND_INDEX = labels_mod.CLASS_INDEX["background"]  # 0
NUM_CLASSES = len(labels_mod.CLASS_ORDER)  # 8

# Caduceus-PS tokenizer vocabulary (kuleshov-group/caduceus-ps_seqlen-131k_d_model-256
# _n_layer-16 @ d89eeb85..., ``tokenization_caduceus.py``). Pinned here as the
# zero-flank pad id + base ids; a drift-guard test re-reads them off the real
# tokenizer when it is available.
PAD_TOKEN_ID = 4
BASE_TO_ID: dict[str, int] = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
UNKNOWN_BASE_ID = BASE_TO_ID["N"]
_COMPLEMENT: dict[str, str] = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}

# ── Curriculum defaults (implementer choices — NOT pinned by PRD/ADR) ────────
# The three alpha terms MULTIPLY (see ``curriculum_weights``), and on this corpus
# the three axes are strongly correlated — a class-II record is also rare by
# phylum and usually by specifier AA — so the effective strength on the joint
# stratum is ~the SUM of the alphas, not any single one. Measured on the real
# 8,303-record ADR-0004 D5 training fold (class II natural rate 0.27%):
#
#   alpha (each)   class-II share   enrichment   Firmicutes   top-10-record mass
#   0.25            2.39%            9.0x         92.1%        1.9%
#   0.50           22.82%           86.1x         58.5%       22.5%
#   0.75           71.32%          269.2x         11.0%       72.5%
#   1.00           91.88%          346.8x          1.2%       93.0%
#
# 0.25 is the default: it materially lifts class II and softens the Firmicutes
# skew while no single locus dominates. Above it the stream collapses onto the
# 22 unique class-II loci — that is memorisation, not learning, and the class-II
# anti-mimicry evidence does not rest on this sampler anyway (it rests on the
# class-II-CM-naive checkpoint + the P2-08 construction-powered recovery set,
# ADR-0005 D9). ``test_curriculum_bounds_the_mass_on_any_single_locus`` pins the
# consequence, so a future alpha change cannot silently re-create the pathology.
# P2-06 sweeps these as ordinary hyperparameters.
WEIGHTS_PINNED = False
DEFAULT_SEED = 42
DEFAULT_PHYLUM_ALPHA = 0.25
DEFAULT_KLASS_ALPHA = 0.25
DEFAULT_AA_ALPHA = 0.25
ALPHA_MIN, ALPHA_MAX = 0.0, 1.0
UNKNOWN_STRATUM = "UNKNOWN"

# ── Column names (exact, as written by the upstream producers) ───────────────
RECORD_ID_COL = "record_id"
PARENT_ID_COL = "parent_record_id"
CONTEXT_SEQ_COL = "context_seq"
LOCUS_OFFSET_COL = "locus_offset"
LOCUS_LENGTH_COL = "locus_length"
CLIPPED_START_COL = "clipped_start"
CLIPPED_END_COL = "clipped_end"
STATUS_COL = "status"
LABEL_STRING_COL = "label_string"
LABELS_ID_COL = "record_sha256"
COGNATE_AA_COL = "cognate_aa"
KLASS_COL = "klass"
PHYLUM_COL = "resolved_phylum"
CLUSTER_COL = "cluster_id"
NESTED_TRAIN_COL = "nested_train"
FOLD_RANDOM_COL = "fold_random"
#: splits.py's dedicated leave-one-order-out indicator (splits.py:751,
#: ``(order in heldout_set) and is_corpus``). The **exact** predicate — `nested_role`
#: is not, because its ``"heldout"`` value conflates three populations (LOO-order + the
#: Actinobacteria phylum arm + external anchors), and the negation of ``nested_train``
#: is not, because it is what shipped the P2-06a defect. Carried onto every
#: :class:`CorpusRecord` so a fold's LOO count is **measured off what was emitted**.
LOO_HOLDOUT_COL = "is_designated_loo_holdout"
SOURCE_COL = "source"
STATUS_OK = "ok"
POSITIVE_SOURCE = "corpus"

# Scheme A's val fold value (splits.RANDOM_RATIOS' key).
#
# ⚠ NOT the P2-06a selection rung, and the reason is the whole point of this module's
# second decision. `fold_random` and `nested_train` are **independent schemes on one
# table**, so `fold_random == "val"` says nothing about scheme B: filtering it (even
# after removing the training fold) lands 88.4% inside the **designated
# leave-one-order-out holdout** — the population PRD §12:241 reports as the headline.
# See `load_selection_val_records` for the rung that replaced it.
FOLD_RANDOM_VAL = "val"

#: Fold scopes a loader can return.
#:
#: * ``train`` — the full ADR-0004 D5 nested training fold (8,303 rec / 4,775 clusters).
#: * ``inner_train`` — ``train`` **minus** the selection-val clusters: what a P2-06 sweep
#:   point actually trains on.
#: * ``selection_val`` — the P2-06a inner rung carved from *inside* ``train``: what P2-06
#:   promotes its best config on (:func:`load_selection_val_records`).
FOLD_SCOPE_TRAIN = "train"
FOLD_SCOPE_INNER_TRAIN = "inner_train"
FOLD_SCOPE_SELECTION_VAL = "selection_val"
FOLD_SCOPE_ALL = "all"

#: The step that owns the selection-val loader (its report's provenance).
STEP_SELECTION_VAL = "P2-06a"

#: The P2-06a inner-rung carve (user decision 2026-07-17, re-taken after the first
#: definition was measured to be 88.4% designated-LOO-holdout).
#:
#: **The rule.** Whole ``cluster_id`` groups are drawn — seeded, deterministically — from
#: **inside** the ADR-0004 D5 ``nested_train`` fold until ~``SELECTION_VAL_FRACTION`` of
#: its *records* are held. Cluster-grouped because a homology cluster split across the
#: boundary is leakage (PRD §9.2); from *inside* ``nested_train`` because that fold's
#: complement **is** the holdout — ``nested_train = is_corpus & has_order & ~linked``
#: (splits.py:706), so "not nested_train" means "in the LOO holdout", not "held out from
#: training". Selecting there optimises the headline. Measured on the committed table,
#: ``nested_train`` contains **0** designated-LOO records, so this carve is LOO-free by
#: construction — and :func:`load_selection_val_records` **verifies** it off the emitted
#: records rather than resting on that sentence.
SELECTION_VAL_FRACTION = 0.10
SELECTION_VAL_SEED = 20260717

# The six fold-per-scheme columns (splits.FOLD_SCHEME_COLUMNS). A variant must
# inherit every one from its parent (ADR-0004 D7).
FOLD_SCHEME_COLUMNS: tuple[str, ...] = (
    "fold_random",
    "loo_order_unit",
    "class_holdout_unit",
    "phylum_holdout_unit",
    "nested_train",
    "nested_role",
)

# ── Default artifact paths ───────────────────────────────────────────────────
DEFAULT_CONTEXT = Path("data/interim/flank_context/context_v0.parquet")
DEFAULT_LABELS = Path("data/processed/labels/labels_v0.parquet")
DEFAULT_SPLIT_TABLE = Path("data/processed/splits/split_assignments.parquet")
CONFIG_PATH = Path("conf/data/stage1.yaml")


# ═════════════════════════════════════════════════════════════════════════════
# Sequence encoding
# ═════════════════════════════════════════════════════════════════════════════
def encode_bases(seq: str) -> np.ndarray:
    """Encode a DNA string to Caduceus token ids as an ``(L,)`` int16 array.

    Any character outside ``ACGTN`` maps to the ``N`` id — the corpus is
    uppercase ``ACGTN`` (P2-00 measured), so this is a defensive fallback, not a
    silent cleaner.
    """
    return np.fromiter(
        (BASE_TO_ID.get(ch, UNKNOWN_BASE_ID) for ch in seq),
        dtype=np.int16,
        count=len(seq),
    )


def reverse_complement(seq: str) -> str:
    """Reverse-complement a DNA string (``ACGTN``; unknown chars → ``N``)."""
    return "".join(_COMPLEMENT.get(ch, "N") for ch in reversed(seq))


def reverse_complement_ids(ids: np.ndarray) -> np.ndarray:
    """Reverse-complement an encoded id array, preserving ``[PAD]`` as ``[PAD]``.

    A/T and C/G swap, N and [PAD] are self-complementary, and the array is
    reversed — so a zero-flank at the 5' end becomes a zero-flank at the 3' end,
    which is what reading the other strand actually means.
    """
    ids = np.asarray(ids)
    swapped = ids.copy()
    a, c, g, t = BASE_TO_ID["A"], BASE_TO_ID["C"], BASE_TO_ID["G"], BASE_TO_ID["T"]
    swapped[ids == a] = t
    swapped[ids == t] = a
    swapped[ids == c] = g
    swapped[ids == g] = c
    return swapped[::-1].copy()


def label_indices(label_string: str) -> np.ndarray:
    """``label_string`` → an ``(m,)`` int16 array of 8-class softmax indices."""
    return np.asarray(labels_mod.label_string_to_indices(label_string), dtype=np.int16).reshape(-1)


# ═════════════════════════════════════════════════════════════════════════════
# Window geometry
# ═════════════════════════════════════════════════════════════════════════════
def tile_windows(seq_len: int, *, window: int = WINDOW_NT, stride: int = STRIDE_NT) -> list[int]:
    """Window start offsets tiling ``seq_len`` at ``stride`` (PRD §6 scan tiling).

    The final window is anchored to the sequence end so the 3' tail is always
    covered; every interior nucleotide is covered by >= 2 windows when
    ``stride <= window // 2``. This is the deterministic *scan/eval* tiling that
    P2-03's logit-reconciliation operator consumes — training positives use
    :func:`window_lead_range` offset augmentation instead.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if seq_len <= window:
        return [0]
    starts = list(range(0, seq_len - window + 1, stride))
    last = seq_len - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def window_lead_range(
    *,
    locus_offset: int,
    locus_length: int,
    context_length: int,
    clipped_start: bool,
    clipped_end: bool,
    window: int = WINDOW_NT,
) -> tuple[int, int] | None:
    """The honest range of window-phase *leads* for one record.

    ``lead`` is the number of window positions preceding the locus, so the window
    is ``[locus_offset - lead, locus_offset - lead + window)`` in ``context_seq``
    coordinates and the locus sits at ``[lead, lead + locus_length)`` within the
    window.

    Three constraints intersect:

    * **Containment** — the locus must fit inside the window: ``0 <= lead <= window
      - locus_length``.
    * **No invented 5' sequence** — the window may start before ``context_seq``
      only if ``clipped_start`` (that boundary is a real contig start), so
      otherwise ``lead <= locus_offset``.
    * **No invented 3' sequence** — symmetrically, unless ``clipped_end``,
      ``lead >= locus_offset + window - context_length``.

    Returns the inclusive ``(lo, hi)`` lead range, or ``None`` when no honest
    window exists (the caller must then exclude and count the record — never pad
    it into existence; CLAUDE.md §10.3).
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if locus_length <= 0:
        raise ValueError(f"locus_length must be positive, got {locus_length}")
    if locus_length > window:
        return None
    if locus_offset < 0 or context_length < 0:
        return None
    if locus_offset + locus_length > context_length:
        raise ValueError(
            "locus runs past the end of context_seq: "
            f"offset={locus_offset} length={locus_length} context={context_length}"
        )

    lo = 0 if clipped_end else max(0, locus_offset + window - context_length)
    hi = (window - locus_length) if clipped_start else min(window - locus_length, locus_offset)
    if lo > hi:
        return None
    return lo, hi


def deterministic_lead(lead_range: tuple[int, int], *, window: int, locus_length: int) -> int:
    """The eval-mode lead: centre the locus, clamped into the honest range."""
    lo, hi = lead_range
    centred = (window - locus_length) // 2
    return min(max(centred, lo), hi)


def sample_lead(rng: np.random.Generator, lead_range: tuple[int, int]) -> int:
    """Draw a window-phase lead uniformly from the honest range (PRD §6, §11)."""
    lo, hi = lead_range
    return int(rng.integers(lo, hi + 1))


@dataclass(frozen=True)
class Window:
    """One carved window: model input, per-nt target, and its provenance."""

    input_ids: np.ndarray  # (window,) int16 — [PAD] at zero-flanked positions
    labels: np.ndarray  # (window,) int16 — IGNORE_INDEX at zero-flanked positions
    real_mask: np.ndarray  # (window,) bool — True where a real nucleotide sits
    record_id: str  # the variant id (this window)
    parent_record_id: str  # the source corpus record
    lead: int
    reverse_complement: bool
    pad_left: int
    pad_right: int

    @property
    def zero_flanked(self) -> bool:
        """True when the window was zero-flanked at a real contig end (PRD §6)."""
        return bool(self.pad_left or self.pad_right)


def variant_id(record_id: str, *, lead: int, rc: bool) -> str:
    """Deterministic id for an augmented variant of ``record_id``.

    Distinct from the parent's id (so `parent_record_id != record_id` marks it a
    variant, matching the ADR-0004 D7 predicate) and a pure function of the
    augmentation, so the same draw always names the same variant.
    """
    return f"{record_id}#w{lead}{'rc' if rc else ''}"


def carve_window(
    *,
    context_seq: str,
    locus_offset: int,
    locus_length: int,
    label_string: str,
    lead: int,
    record_id: str,
    clipped_start: bool,
    clipped_end: bool,
    window: int = WINDOW_NT,
    rc: bool = False,
) -> Window:
    """Carve one ``window``-nt training window at window-phase ``lead``.

    Labels follow imp.md P2-01 exactly: the locus contributes ``label_string``;
    **every real-DNA flank position is `background`**; only zero-flanked (padded)
    positions — which exist solely at real contig ends — take ``IGNORE_INDEX``.
    """
    if len(label_string) != locus_length:
        raise ValueError(
            f"label_string length {len(label_string)} != locus_length {locus_length} "
            f"for record {record_id}"
        )
    context_length = len(context_seq)
    rng_range = window_lead_range(
        locus_offset=locus_offset,
        locus_length=locus_length,
        context_length=context_length,
        clipped_start=clipped_start,
        clipped_end=clipped_end,
        window=window,
    )
    if rng_range is None:
        raise ValueError(f"record {record_id} admits no honest window at window={window}")
    lo, hi = rng_range
    if not lo <= lead <= hi:
        raise ValueError(f"lead {lead} outside the honest range [{lo}, {hi}] for {record_id}")

    start = locus_offset - lead
    stop = start + window
    # Slice the real sequence, then zero-flank whatever fell off a contig end.
    real_start, real_stop = max(0, start), min(context_length, stop)
    pad_left, pad_right = real_start - start, stop - real_stop

    ids = np.full(window, PAD_TOKEN_ID, dtype=np.int16)
    lab = np.full(window, IGNORE_INDEX, dtype=np.int16)
    mask = np.zeros(window, dtype=bool)

    real = context_seq[real_start:real_stop]
    end = window - pad_right
    ids[pad_left:end] = encode_bases(real)
    mask[pad_left:end] = True
    # Real DNA is background by default; the locus then paints its own classes.
    lab[pad_left:end] = BACKGROUND_INDEX
    locus_in_window = lead  # == locus_offset - start
    lab[locus_in_window : locus_in_window + locus_length] = label_indices(label_string)

    emitted_lead = lead
    if rc:
        ids = reverse_complement_ids(ids)
        lab = lab[::-1].copy()
        mask = mask[::-1].copy()
        pad_left, pad_right = pad_right, pad_left
        # Reversing the array moves position p to (window - 1 - p), so the locus
        # now begins at `window - lead - locus_length`. `lead` must describe the
        # EMITTED window (it is the locus's offset within it), or every consumer
        # slicing `labels[lead : lead + locus_length]` reads the wrong span.
        emitted_lead = window - lead - locus_length

    return Window(
        input_ids=ids,
        labels=lab,
        real_mask=mask,
        record_id=variant_id(record_id, lead=emitted_lead, rc=rc),
        parent_record_id=record_id,
        lead=emitted_lead,
        reverse_complement=rc,
        pad_left=pad_left,
        pad_right=pad_right,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Curriculum weights (PRD §11)
# ═════════════════════════════════════════════════════════════════════════════
def _normalise_stratum(value: Any) -> str:
    """Map a missing/blank stratum key to an explicit ``UNKNOWN`` bucket.

    Records with no resolved phylum or no called specifier amino acid form their
    own stratum rather than silently joining the majority one.
    """
    if value is None:
        return UNKNOWN_STRATUM
    if isinstance(value, float) and math.isnan(value):
        return UNKNOWN_STRATUM
    text = str(value).strip()
    return text if text else UNKNOWN_STRATUM


def inverse_frequency_weights(keys: Sequence[Any], *, alpha: float) -> np.ndarray:
    """Per-record weights ``w_i ∝ (1 / n_{stratum(i)}) ** alpha``, mean-normalised.

    ``alpha = 0`` reproduces the natural distribution; ``alpha = 1`` fully
    balances the strata; the default ``0.25`` is a partial rebalance that lifts
    the rare strata without letting a 3-record phylum dominate an 8,182-record
    one. Note the caller multiplies three such terms — see
    :data:`DEFAULT_PHYLUM_ALPHA` for why the default is 0.25 and not higher.

    ``alpha`` is an **implementer choice** (``WEIGHTS_PINNED is False``): PRD §11
    pins that balanced-phylum and rare-class sampling happen, not their strength.
    """
    if not ALPHA_MIN <= alpha <= ALPHA_MAX:
        raise ValueError(f"alpha must be in [{ALPHA_MIN}, {ALPHA_MAX}], got {alpha}")
    if len(keys) == 0:
        return np.zeros(0, dtype=np.float64)
    strata = [_normalise_stratum(k) for k in keys]
    counts = Counter(strata)
    w = np.asarray([counts[s] ** (-alpha) for s in strata], dtype=np.float64)
    return w / w.mean()


def curriculum_weights(
    *,
    phyla: Sequence[Any],
    klasses: Sequence[Any],
    amino_acids: Sequence[Any],
    phylum_alpha: float = DEFAULT_PHYLUM_ALPHA,
    klass_alpha: float = DEFAULT_KLASS_ALPHA,
    aa_alpha: float = DEFAULT_AA_ALPHA,
) -> np.ndarray:
    """The combined PRD §11 sampling weight per training record.

    Three inverse-frequency terms multiply: **balanced-phylum** (guards the
    ~90%-Firmicutes overfit, §12), **rare-class** (class II is scarce and
    Actinobacteria-skewed, §8), and **rare specifier amino acid** (the Trp/Leu/Ile
    vs Lys/Glu/Gln skew, §8 [DOI:10.1093/nar/gkaa721]). Mean-normalised, so the
    expected epoch size is unchanged.

    The terms **compound**: the axes are correlated on this corpus, so the joint
    strength is roughly ``phylum_alpha + klass_alpha + aa_alpha``. See the
    measured sweep at :data:`DEFAULT_PHYLUM_ALPHA` before raising any of them.
    """
    n = len(phyla)
    if not (len(klasses) == len(amino_acids) == n):
        raise ValueError(
            f"phyla/klasses/amino_acids length mismatch: {n}/{len(klasses)}/{len(amino_acids)}"
        )
    w = (
        inverse_frequency_weights(phyla, alpha=phylum_alpha)
        * inverse_frequency_weights(klasses, alpha=klass_alpha)
        * inverse_frequency_weights(amino_acids, alpha=aa_alpha)
    )
    if n == 0:
        return w
    return w / w.mean()


class WeightedIndexSampler:
    """Deterministic with-replacement weighted sampler over dataset indices.

    Torch-free by design (numpy ``Generator``), so the whole curriculum is unit-
    testable in bare CI, and duck-type compatible with ``DataLoader(sampler=...)``
    — which only requires an iterable of **keys** plus ``__len__``, and passes each
    key straight through to ``dataset[key]``. Seeding follows the same contract as
    ``torch.utils.data.WeightedRandomSampler(generator=...)``: a fixed seed
    reproduces the draw exactly (CLAUDE.md §8.3).

    **Yields ``(index, occurrence)`` keys, not bare indices.** Oversampling draws
    a rare record many times per epoch; ``occurrence`` is the ordinal of the draw
    within the epoch, so :class:`Stage1WindowDataset` can give each draw an
    independent window phase and strand. Without it, ``dataset[index]`` is a pure
    function of the index and the 9x-oversampled class-II records would arrive as
    9 **identical** windows — which is exactly the memorisation that pairing
    oversampling with offset augmentation (PRD §11) exists to prevent.
    """

    def __init__(
        self,
        weights: Sequence[float] | np.ndarray,
        *,
        num_samples: int | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {w.shape}")
        if w.size == 0:
            raise ValueError("weights must be non-empty")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must all be finite")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = w
        self.probabilities = w / total
        self.num_samples = int(num_samples) if num_samples is not None else int(w.size)
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-seed per epoch so successive epochs draw differently but reproducibly."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, self._epoch])
        draw = rng.choice(
            self.probabilities.size, size=self.num_samples, replace=True, p=self.probabilities
        )
        return iter((int(i), occ) for occ, i in enumerate(draw))

    def indices(self) -> list[int]:
        """The drawn indices without their occurrence ordinals (for diagnostics)."""
        return [i for i, _ in self]


# ═════════════════════════════════════════════════════════════════════════════
# Corpus loading
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class CorpusRecord:
    """One training-eligible corpus record: geometry + labels + strata."""

    record_id: str
    context_seq: str
    locus_offset: int
    locus_length: int
    label_string: str
    clipped_start: bool
    clipped_end: bool
    klass: str
    phylum: str
    cognate_aa: str
    cluster_id: int
    nested_train: bool
    #: splits.py's designated leave-one-order-out indicator, carried per-record so a
    #: loader's LOO count is re-derivable from the records it actually emitted rather
    #: than from the filter that chose them. **No default**: a record that cannot say
    #: whether it is in the headline holdout must not be constructible, and `False` is
    #: exactly the value a fabricating default would supply (CLAUDE.md §10.3).
    is_designated_loo_holdout: bool
    folds: tuple[Any, ...]
    #: The **corpus record whose genomic context this record's DNA came from**. For a
    #: positive it is ``record_id`` itself; for a mined negative it is the parent locus
    #: the window was carved beside, which is a *different* record and the only thing
    #: that can say whether the DNA is training-fold DNA. `negatives.background_record`
    #: stamps ``nested_train=True`` unconditionally — an assertion about the negative,
    #: not a check on its origin — so without this field a window carved from the flank
    #: of a leave-one-order-out holdout locus is, in the type, indistinguishable from an
    #: in-fold one (P2-10d′-a). **No default**, for the reason
    #: ``is_designated_loo_holdout`` has none: a record that cannot name its origin must
    #: not be constructible, and the fabricating default here is `record_id`, which is
    #: right for positives and silently wrong for exactly the records that matter.
    source_record_id: str


def _exclusion_reason(row: Mapping[str, Any], *, window: int) -> str | None:
    """Why a record cannot yield an honest window, or ``None`` if it can."""
    if row[STATUS_COL] != STATUS_OK:
        # P2-00 could not anchor the locus in real context (2 multi_hit +
        # 1 suppressed accession). There is no context_seq to window.
        return f"unanchored:{row[STATUS_COL]}"
    if (
        window_lead_range(
            locus_offset=int(row[LOCUS_OFFSET_COL]),
            locus_length=int(row[LOCUS_LENGTH_COL]),
            context_length=len(row[CONTEXT_SEQ_COL]),
            clipped_start=bool(row[CLIPPED_START_COL]),
            clipped_end=bool(row[CLIPPED_END_COL]),
            window=window,
        )
        is None
    ):
        # Real context too short to fill the window, and the missing sequence is
        # NOT at a flagged contig end — so padding it would invent a boundary.
        return "no_honest_window"
    return None


def _merge_corpus_tables(
    *,
    context_parquet: str | Path,
    labels_parquet: str | Path,
    split_table: str | Path,
) -> Any:
    """Join the P2-00 context, the P0-20 labels, and the ADR-0004 split table.

    All three key on the same 64-hex corpus record hash (``record_id`` in the
    context table and — for ``source == "corpus"`` rows — in the split table;
    ``record_sha256`` in the labels table).

    Factored out so every fold scope reads **one** join implementation:
    :func:`load_corpus_records` (the ADR-0004 D5 training fold) and
    :func:`load_selection_val_records` (the P2-06a selection val) must agree on
    record identity, or a "disjoint" claim compares two different populations.
    """
    import pandas as pd

    ctx = pd.read_parquet(context_parquet)
    lab = pd.read_parquet(labels_parquet)
    spl = pd.read_parquet(split_table)

    positives = spl[spl[SOURCE_COL] == POSITIVE_SOURCE]
    keep = [
        RECORD_ID_COL,
        PARENT_ID_COL,
        KLASS_COL,
        PHYLUM_COL,
        CLUSTER_COL,
        LOO_HOLDOUT_COL,
        *FOLD_SCHEME_COLUMNS,
    ]
    merged = ctx.merge(
        lab[[LABELS_ID_COL, LABEL_STRING_COL, COGNATE_AA_COL]],
        left_on=RECORD_ID_COL,
        right_on=LABELS_ID_COL,
        how="inner",
        validate="one_to_one",
    ).merge(positives[keep], on=RECORD_ID_COL, how="inner", validate="one_to_one")
    if len(merged) != len(ctx):
        raise ValueError(
            f"context/labels/split join lost rows: {len(ctx)} context -> {len(merged)} joined"
        )
    return merged


def _corpus_record(row: Mapping[str, Any]) -> CorpusRecord:
    """Build one :class:`CorpusRecord` from a merged table row."""
    return CorpusRecord(
        record_id=str(row[RECORD_ID_COL]),
        context_seq=str(row[CONTEXT_SEQ_COL]),
        locus_offset=int(row[LOCUS_OFFSET_COL]),
        locus_length=int(row[LOCUS_LENGTH_COL]),
        label_string=str(row[LABEL_STRING_COL]),
        clipped_start=bool(row[CLIPPED_START_COL]),
        clipped_end=bool(row[CLIPPED_END_COL]),
        klass=str(row[KLASS_COL]),
        phylum=_normalise_stratum(row[PHYLUM_COL]),
        cognate_aa=_normalise_stratum(row[COGNATE_AA_COL]),
        cluster_id=int(row[CLUSTER_COL]),
        nested_train=bool(row[NESTED_TRAIN_COL]),
        is_designated_loo_holdout=bool(row[LOO_HOLDOUT_COL]),
        folds=tuple(row[c] for c in FOLD_SCHEME_COLUMNS),
        # A positive IS its own origin: the window is carved from the context fetched
        # for this very locus, so the fold the record carries and the fold of the DNA
        # are the same fold.
        source_record_id=str(row[RECORD_ID_COL]),
    )


def _training_fold_cluster_sizes(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """``cluster_id -> record count`` over ``nested_train`` rows only.

    The carve's input, built **identically** for both callers — the sizes decide which
    clusters are drawn, so two callers computing them from different populations would
    silently carve two different buckets from the same seed.
    """
    return dict(Counter(int(r[CLUSTER_COL]) for r in rows if bool(r[NESTED_TRAIN_COL])))


def selection_val_cluster_ids(
    cluster_sizes: Mapping[int, int],
    *,
    fraction: float = SELECTION_VAL_FRACTION,
    seed: int = SELECTION_VAL_SEED,
) -> frozenset[int]:
    """Choose the whole clusters forming the P2-06a selection-val bucket.

    ``cluster_sizes`` maps ``cluster_id -> record count`` over the **training fold only**
    (the caller's job: the carve draws from inside ``nested_train``, never from the table
    at large). Whole clusters are drawn in a seeded permutation until ``fraction`` of the
    fold's *records* is reached, so the bucket is cluster-grouped by construction — no
    homology cluster can straddle the inner boundary (PRD §9.2).

    **One rule, one implementation, two callers.** Both :func:`load_selection_val_records`
    (which *returns* the bucket) and :func:`load_corpus_records` (which must *remove* it
    from training) call this. A second copy of the rule would let the two folds disagree
    about what was carved — i.e. train-on-val — while each self-consistently reported
    success ([[promote-dont-duplicate-is-a-correctness-rule]]: a forked helper means
    fixing one copy and shipping the bug in the other).

    Deterministic in the mapping's *content*, not its iteration order: ids are sorted
    before the seeded permutation, so two callers building the mapping differently still
    get the identical bucket (§8.3).
    """
    if not isinstance(fraction, float) or not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be a float in (0, 1), got {fraction!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {seed!r}")
    ids = sorted(int(c) for c in cluster_sizes)
    if not ids:
        raise ValueError("cluster_sizes is empty — there is no training fold to carve")
    sizes = {int(c): int(cluster_sizes[c]) for c in cluster_sizes}
    if any(n <= 0 for n in sizes.values()):
        raise ValueError("cluster_sizes carries a non-positive count")
    total = sum(sizes.values())
    target = fraction * total

    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    held = 0
    for idx in rng.permutation(len(ids)):
        if held >= target:
            break
        cid = ids[int(idx)]
        chosen.add(cid)
        held += sizes[cid]
    if not chosen or len(chosen) == len(ids):
        raise ValueError(
            f"carve degenerate: {len(chosen)} of {len(ids)} clusters chosen — a bucket "
            "that is empty or the whole fold is not a selection rung"
        )
    return frozenset(chosen)


def load_corpus_records(
    *,
    context_parquet: str | Path = DEFAULT_CONTEXT,
    labels_parquet: str | Path = DEFAULT_LABELS,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    window: int = WINDOW_NT,
    training_fold_only: bool = True,
    exclude_selection_val: bool | None = None,
    selection_val_fraction: float = SELECTION_VAL_FRACTION,
    selection_val_seed: int = SELECTION_VAL_SEED,
) -> tuple[list[CorpusRecord], dict[str, Any]]:
    """Load the corpus records for the **training** fold.

    With ``training_fold_only`` (the default), only ``nested_train`` records are
    returned: the ADR-0004 D5 single most-restrictive nested training fold. This
    is the mechanism behind PRD §11's "class-II augmentation **constrained to
    training-fold parents**" — a held-out parent is never even loaded, so no
    sampler can oversample it into training (ADR-0004 D7).

    ``exclude_selection_val`` (default: **on** whenever ``training_fold_only`` is) then
    removes the :func:`selection_val_cluster_ids` bucket, yielding the ``inner_train``
    fold a P2-06 sweep point trains on. The default is on because the two failure
    directions are not symmetric: training on the selection fold makes every sweep number
    train-on-train and is **silent**, whereas losing 10% of training data is loud and
    recorded. A caller that genuinely wants the full D5 fold — the P2-09 production
    retrain once the config is fixed, and :mod:`sizing`, whose committed P2-05 figures are
    full-fold extrapolations — must say ``exclude_selection_val=False`` and mean it.

    ⚠ ``training_fold_only=False`` returns train+val+test **pooled**, not a val fold; the
    carve is undefined there (it is a partition *of the training fold*), so passing
    ``exclude_selection_val=True`` alongside it raises rather than silently no-opping.

    Returns ``(records, report)``; the report counts every exclusion by reason
    (CLAUDE.md §10.3 — excluded records are reported, never silently dropped).
    """
    if exclude_selection_val is None:
        exclude_selection_val = bool(training_fold_only)
    elif exclude_selection_val and not training_fold_only:
        raise ValueError(
            "exclude_selection_val=True requires training_fold_only=True — the P2-06a "
            "carve partitions the nested training fold, so removing it from a pooled "
            "train+val+test load is not a defined operation"
        )
    merged = _merge_corpus_tables(
        context_parquet=context_parquet,
        labels_parquet=labels_parquet,
        split_table=split_table,
    )

    n_total = len(merged)
    rows = merged.to_dict("records")
    val_clusters: frozenset[int] = frozenset()
    if exclude_selection_val:
        val_clusters = selection_val_cluster_ids(
            _training_fold_cluster_sizes(rows),
            fraction=selection_val_fraction,
            seed=selection_val_seed,
        )

    excluded: Counter[str] = Counter()
    records: list[CorpusRecord] = []
    n_out_of_fold = 0
    n_selection_val = 0
    for row in rows:
        if training_fold_only and not bool(row[NESTED_TRAIN_COL]):
            n_out_of_fold += 1
            continue
        if exclude_selection_val and int(row[CLUSTER_COL]) in val_clusters:
            n_selection_val += 1
            continue
        reason = _exclusion_reason(row, window=window)
        if reason is not None:
            excluded[reason] += 1
            continue
        records.append(_corpus_record(row))
    records.sort(key=lambda r: r.record_id)

    if training_fold_only:
        fold_scope = FOLD_SCOPE_INNER_TRAIN if exclude_selection_val else FOLD_SCOPE_TRAIN
    else:
        fold_scope = FOLD_SCOPE_ALL
    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "window_nt": int(window),
        "n_context_records": int(n_total),
        "training_fold_only": bool(training_fold_only),
        "fold_scope": fold_scope,
        "exclude_selection_val": bool(exclude_selection_val),
        # Whole clusters, so this is also the count the val fold must NOT be able to
        # reach. Measured on the pass that emitted `records`, not predicted from the rule.
        "n_selection_val_excluded": int(n_selection_val),
        "n_selection_val_clusters": len(val_clusters),
        "n_out_of_training_fold": int(n_out_of_fold),
        "n_excluded": int(sum(excluded.values())),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "n_records": len(records),
        "klass_counts": dict(sorted(Counter(r.klass for r in records).items())),
        "phylum_counts": dict(sorted(Counter(r.phylum for r in records).items())),
        "synthetic_flank": False,
    }
    return records, report


def load_selection_val_records(
    *,
    context_parquet: str | Path = DEFAULT_CONTEXT,
    labels_parquet: str | Path = DEFAULT_LABELS,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    window: int = WINDOW_NT,
    fraction: float = SELECTION_VAL_FRACTION,
    seed: int = SELECTION_VAL_SEED,
) -> tuple[list[CorpusRecord], dict[str, Any]]:
    """Load the **P2-06a model-selection val fold**: the inner rung of the nested design.

    This is the population P2-06 promotes its best config on (user decision 2026-07-17,
    **re-taken** after the first definition was measured — see below). It is a
    cluster-grouped, seeded ~``fraction`` carve from **inside** the ADR-0004 D5
    ``nested_train`` fold, and it is disjoint from what the model trains on because
    :func:`load_corpus_records` removes the identical bucket, by the identical rule
    (:func:`selection_val_cluster_ids`), from training.

    Filters:

    1. ``nested_train`` — the carve draws only from **inside** the training fold.
    2. ``cluster_id`` in the seeded carve bucket — whole clusters, so nothing straddles
       the inner boundary (PRD §9.2's cluster non-splitting, applied to the inner rung).
    3. ``len(context_seq) >= window`` — the eval tiles the **full context** with
       :func:`tile_windows`, which for ``seq_len >= window`` emits only starts in
       ``[0, seq_len - window]``: every window is then wholly real DNA and no position is
       padded. Requiring it means the eval never zero-flanks a boundary that is not a real
       contig end (PRD §6; §10.3) — shorter contexts are **excluded and counted**, never
       padded into existence. A record dropped here is dropped from training too (its
       whole cluster is carved), so it is lost, not leaked.

    ⚠ **The defect this replaced, recorded because the name still invites it.** The first
    version filtered ``fold_random == "val" & not nested_train``. ``fold_random`` and
    ``nested_train`` are **independent schemes on one table**, and
    ``nested_train = is_corpus & has_order & ~linked`` (splits.py:706) — so ``not
    nested_train`` means *"in the leave-one-order-out holdout"*, **not** *"held out from
    training"*. Measured on the emitted fold: **778 / 880 = 88.4%** were designated LOO
    holdout, **687 of 726** blocks were LOO clusters, and its top order — Lactobacillales,
    58% of the fold alone — is itself a designated holdout order. Tuning there would have
    optimised the exact population P2-14 reports as the PRD §12:241 headline.

    Its ``disjointness`` block was **true and useless**: it proved disjointness *from
    train* while the fold sat inside the *holdout*. The two are not complements, and no
    amount of checking the first detects a violation of the second — which is why the
    ``leakage`` block below measures ``n_designated_loo_holdout`` **directly**, off
    splits.py's dedicated indicator, off the records actually emitted.

    Returns ``(records, report)``. Every number in ``report["leakage"]`` is measured back
    off the emitted records, never asserted from the filter that produced them — a filter
    and its own echo cannot disagree (CLAUDE.md §10.3; the P1-15/P1-16 re-derivation rule).
    """
    merged = _merge_corpus_tables(
        context_parquet=context_parquet,
        labels_parquet=labels_parquet,
        split_table=split_table,
    )

    rows = merged.to_dict("records")
    fold_sizes = _training_fold_cluster_sizes(rows)
    val_clusters = selection_val_cluster_ids(fold_sizes, fraction=fraction, seed=seed)

    n_total = len(rows)
    excluded: Counter[str] = Counter()
    records: list[CorpusRecord] = []
    # The complement, built in the SAME pass under the SAME rule, so the disjointness
    # below compares this fold against the very population training will see.
    inner_train_ids: set[str] = set()
    inner_train_clusters: set[int] = set()
    for row in rows:
        if not bool(row[NESTED_TRAIN_COL]):
            excluded["not_in_training_fold"] += 1
            continue
        if int(row[CLUSTER_COL]) not in val_clusters:
            if _exclusion_reason(row, window=window) is None:
                inner_train_ids.add(str(row[RECORD_ID_COL]))
                inner_train_clusters.add(int(row[CLUSTER_COL]))
            excluded["inner_train_cluster"] += 1
            continue
        if len(str(row[CONTEXT_SEQ_COL])) < window:
            excluded["context_shorter_than_window"] += 1
            continue
        reason = _exclusion_reason(row, window=window)
        if reason is not None:
            excluded[reason] += 1
            continue
        records.append(_corpus_record(row))
    records.sort(key=lambda r: r.record_id)

    # ── Leakage evidence, MEASURED off the emitted records (never off the filter) ──
    got_ids = {r.record_id for r in records}
    got_clusters = {r.cluster_id for r in records}
    klass_counts = Counter(r.klass for r in records)
    n_ii = int(klass_counts.get("II", 0))
    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP_SELECTION_VAL,
        "window_nt": int(window),
        "fold_scope": FOLD_SCOPE_SELECTION_VAL,
        "scheme": (
            "P2-06a inner rung — cluster-grouped seeded carve from INSIDE the ADR-0004 D5 "
            "nested training fold; NOT a PRD §9.2 ladder scheme and NOT scheme-A val"
        ),
        "carve": {
            "fraction": float(fraction),
            "seed": int(seed),
            "rule": "whole clusters, seeded permutation, filled to fraction of fold records",
            "n_fold_records": int(sum(fold_sizes.values())),
            "n_fold_clusters": len(fold_sizes),
            "n_val_clusters": len(val_clusters),
            # Pins WHICH clusters were drawn, so a re-run is checkable against this run
            # rather than merely re-derivable in principle (§8.3, §11 provenance).
            "cluster_digest": hashlib.sha256(
                ",".join(str(c) for c in sorted(val_clusters)).encode()
            ).hexdigest(),
        },
        "n_context_records": int(n_total),
        "n_records": len(records),
        "n_blocks": len(got_clusters),
        "n_excluded": int(sum(excluded.values())),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "klass_counts": dict(sorted(klass_counts.items())),
        "phylum_counts": dict(sorted(Counter(r.phylum for r in records).items())),
        "leakage": {
            # ── THE clause. The P2-06a defect, measured directly rather than argued
            # from a fold name. splits.py's own indicator, counted over what was emitted.
            "n_designated_loo_holdout": sum(1 for r in records if r.is_designated_loo_holdout),
            # The carve came from inside the training fold, or it is not this rung.
            "n_not_nested_train": sum(1 for r in records if not r.nested_train),
            # Disjointness from what training ACTUALLY sees (inner_train), not from the
            # full D5 fold — the old block's mistake was checking the wrong population.
            "shared_record_ids_with_inner_train": len(got_ids & inner_train_ids),
            "shared_cluster_ids_with_inner_train": len(got_clusters & inner_train_clusters),
            "n_inner_train_records": len(inner_train_ids),
            "n_inner_train_blocks": len(inner_train_clusters),
        },
        # Reported, NOT gated — and the distinction is the point. `min_heldout_positives`
        # is splits.py's min-N floor for a *held-out generalization unit* (PRD §12's
        # per-order ECE rule); this is a *selection* fold whose statistic is the GATE-4
        # core min-F1 over {Stem I, Specifier, Antiterminator} — structural elements, to
        # which class-I and class-II records both contribute. Gating a selection rung on
        # class-II count asserted it could rank configs on class-II signal; nothing does,
        # at any carve size (the whole D5 fold holds 22). The user accepted this cost
        # explicitly on 2026-07-17. Recorded so a reader sees the limit, not a silence.
        "min_heldout_positives": int(splits_mod.MIN_HELDOUT_POSITIVES),
        "class_ii_records": n_ii,
        "class_ii_below_min_heldout_positives": bool(n_ii < int(splits_mod.MIN_HELDOUT_POSITIVES)),
        "synthetic_flank": False,
    }
    return records, report


def selection_val_problems(report: Mapping[str, Any]) -> list[str]:
    """Re-derive the P2-06a selection-val invariants from ``report``; ``[]`` when clean.

    Total by construction: a missing block is a problem, never a pass. The clause an
    absent-evidence bug would fabricate TRUE is the one that matters, so every check below
    is guarded on the evidence actually being **present** ([[clauses-must-guard-emptiness]]
    — a P2-05 finding paid for in a green gate that rested on zero measurements).

    ⚠ **Being total was not enough, and that is P2-06a's own lesson.** The first version of
    this function was total, guarded, and green — and the fold it certified was 88.4%
    designated LOO holdout. It checked disjointness *from the training fold*, which was
    true, while the real question — whether the fold sits in the *headline holdout* — went
    unasked. A clause can only catch what it names. ``leakage.n_designated_loo_holdout``
    names it.
    """
    problems: list[str] = []
    leak = report.get("leakage")
    if not isinstance(leak, Mapping) or not leak:
        return ["leakage: block missing — a selection set with no leak evidence is not usable"]

    #: Every one of these must be **0**, and each is measured off the emitted records.
    #: The first is the clause P2-06a's original definition would have failed at 778.
    _ZERO_CLAUSES = {
        "n_designated_loo_holdout": (
            "the selection set sits in the designated leave-one-order-out holdout; "
            "tuning here optimises the population PRD §12:241 reports as the headline"
        ),
        "n_not_nested_train": (
            "the selection set reaches outside the ADR-0004 D5 training fold, whose "
            "complement IS the holdout (splits.py:706)"
        ),
        "shared_record_ids_with_inner_train": (
            "the selection set overlaps what the model trains on; selecting a config "
            "here is train-on-train (CLAUDE.md §8.2)"
        ),
        "shared_cluster_ids_with_inner_train": (
            "a homology cluster straddles the inner boundary; the selection set is "
            "homology-leaky (PRD §9.2 cluster non-splitting)"
        ),
    }
    for key, why in _ZERO_CLAUSES.items():
        val = leak.get(key)
        if not isinstance(val, int) or isinstance(val, bool):
            problems.append(f"leakage.{key}: must be an int, got {val!r}")
        elif val != 0:
            problems.append(f"leakage.{key} = {val} — {why}")
    n_inner = leak.get("n_inner_train_records")
    if not isinstance(n_inner, int) or isinstance(n_inner, bool) or n_inner <= 0:
        problems.append(
            f"leakage.n_inner_train_records: must be a positive int, got {n_inner!r} — a "
            "disjointness of 0 against an EMPTY training fold is vacuously true, which is "
            "exactly how a clause fabricates TRUE on absent evidence"
        )

    carve = report.get("carve")
    if not isinstance(carve, Mapping) or not carve:
        problems.append("carve: block missing — the rung's provenance is unreproducible")
    else:
        frac = carve.get("fraction")
        if not isinstance(frac, float) or not 0.0 < frac < 1.0:
            problems.append(f"carve.fraction: must be a float in (0, 1), got {frac!r}")
        if not isinstance(carve.get("seed"), int) or isinstance(carve.get("seed"), bool):
            problems.append(f"carve.seed: must be an int, got {carve.get('seed')!r}")
        n_val_c = carve.get("n_val_clusters")
        n_fold_c = carve.get("n_fold_clusters")
        if not (isinstance(n_val_c, int) and not isinstance(n_val_c, bool) and n_val_c > 0):
            problems.append(f"carve.n_val_clusters: must be a positive int, got {n_val_c!r}")
        elif isinstance(n_fold_c, int) and not isinstance(n_fold_c, bool) and n_val_c >= n_fold_c:
            problems.append(
                f"carve.n_val_clusters = {n_val_c} >= n_fold_clusters = {n_fold_c} — the "
                "carve took the whole fold; nothing is left to train on"
            )

    n_records = report.get("n_records")
    if not isinstance(n_records, int) or isinstance(n_records, bool) or n_records <= 0:
        problems.append(f"n_records: must be a positive int, got {n_records!r}")
    n_blocks = report.get("n_blocks")
    if not isinstance(n_blocks, int) or isinstance(n_blocks, bool) or n_blocks < 2:
        problems.append(
            f"n_blocks: must be >= 2, got {n_blocks!r} — fewer than 2 blocks is not "
            "block-resamplable (ADR-0005 Amendment A1)"
        )
    # class_ii_records is REPORTED, never gated — see load_selection_val_records' comment.
    # The floor it was gated against is splits.py's min-N rule for a held-out
    # GENERALIZATION unit, and this is a SELECTION fold ranked on the GATE-4 core
    # elements. It must still be a well-formed count: a fold that cannot say how many
    # class-II it holds is not reporting, it is guessing.
    n_ii = report.get("class_ii_records")
    if not isinstance(n_ii, int) or isinstance(n_ii, bool) or n_ii < 0:
        problems.append(f"class_ii_records: must be a non-negative int, got {n_ii!r}")
    return problems


def context_labels(record: CorpusRecord) -> np.ndarray:
    """The per-nt truth over a record's **full context** (P2-01's labelling rule).

    Real flank is ``background``; the locus paints its own classes. No position is
    ``IGNORE_INDEX``: this is the *eval* truth over real DNA only — the caller
    guarantees the context is real (:func:`load_selection_val_records` filter 4),
    so there is nothing padded to ignore.
    """
    lab = np.full(len(record.context_seq), BACKGROUND_INDEX, dtype=np.int16)
    lab[record.locus_offset : record.locus_offset + record.locus_length] = label_indices(
        record.label_string
    )
    return lab


def encode_eval_window(record: CorpusRecord, start: int, *, window: int = WINDOW_NT) -> np.ndarray:
    """Encode the ``[start, start + window)`` slice of a record's context.

    The eval counterpart of :func:`carve_window`'s input half, for the
    :func:`tile_windows` geometry. Raises rather than padding: a window running
    off the context would zero-flank a boundary that may not be a contig end, and
    filter 4 of :func:`load_selection_val_records` exists so this cannot happen.
    """
    if start < 0 or start + window > len(record.context_seq):
        raise ValueError(
            f"eval window [{start}, {start + window}) runs off record {record.record_id}'s "
            f"{len(record.context_seq)}-nt context — it would invent a boundary (PRD §6)"
        )
    return encode_bases(record.context_seq[start : start + window])


def records_digest(records: Sequence[CorpusRecord]) -> str:
    """Golden digest over the loaded records' identity + geometry + labels."""
    per = [
        ingest.record_hash(
            [
                r.record_id,
                r.context_seq,
                r.locus_offset,
                r.locus_length,
                r.label_string,
                r.clipped_start,
                r.clipped_end,
            ]
        )
        for r in sorted(records, key=lambda x: x.record_id)
    ]
    return ingest.records_digest(per)


def windows_digest(windows: Sequence[Window]) -> str:
    """Golden digest over carved windows — the P2-01 label-alignment invariant.

    Hashes the emitted variant id, the token ids, and the per-nt targets, so any
    drift in window carving, label placement, zero-flanking, or the RC transform
    moves the digest.
    """
    per = [
        ingest.record_hash(
            [
                w.record_id,
                w.parent_record_id,
                w.input_ids.tobytes().hex(),
                w.labels.tobytes().hex(),
                w.pad_left,
                w.pad_right,
            ]
        )
        for w in sorted(windows, key=lambda x: x.record_id)
    ]
    return ingest.records_digest(per)


# ═════════════════════════════════════════════════════════════════════════════
# The dataset
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Stage1DataConfig:
    """Stage-1 datamodule configuration (echoed by ``conf/data/stage1.yaml``)."""

    window_nt: int = WINDOW_NT
    stride_nt: int = STRIDE_NT
    seed: int = DEFAULT_SEED
    offset_augmentation: bool = True
    both_strands: bool = True
    phylum_alpha: float = DEFAULT_PHYLUM_ALPHA
    klass_alpha: float = DEFAULT_KLASS_ALPHA
    aa_alpha: float = DEFAULT_AA_ALPHA


class Stage1WindowDataset:
    """Map-style dataset of 1024-nt windows with per-nucleotide 8-class targets.

    Duck-typed for ``torch.utils.data.DataLoader``: a map-style dataset needs only
    ``__len__`` and ``__getitem__``, which lets this class stay torch-free and
    fully unit-testable in bare CI. Pair it with :func:`collate_windows` to batch
    into tensors.

    Each ``__getitem__`` re-draws the window phase (and strand) from a generator
    seeded by ``(seed, epoch, index)``, so augmentation is **stochastic across
    epochs but exactly reproducible** for a given seed (CLAUDE.md §8.3) and is
    safe under DataLoader worker parallelism (no shared RNG state).

    Every emitted sample carries ``parent_record_id`` — the corpus record it was
    augmented from — so the ADR-0004 D7 variant→parent→fold invariant is checkable
    on the dataset's own output (:meth:`provenance_rows`).
    """

    def __init__(
        self,
        records: Sequence[CorpusRecord],
        *,
        config: Stage1DataConfig | None = None,
        augment: bool | None = None,
    ) -> None:
        if len(records) == 0:
            raise ValueError("Stage1WindowDataset needs at least one record")
        self.records = list(records)
        self.config = config or Stage1DataConfig()
        self.augment = self.config.offset_augmentation if augment is None else bool(augment)
        self._epoch = 0
        for r in self.records:
            if (
                window_lead_range(
                    locus_offset=r.locus_offset,
                    locus_length=r.locus_length,
                    context_length=len(r.context_seq),
                    clipped_start=r.clipped_start,
                    clipped_end=r.clipped_end,
                    window=self.config.window_nt,
                )
                is None
            ):
                raise ValueError(
                    f"record {r.record_id} admits no honest window — it must be excluded "
                    "by load_corpus_records, not padded into existence"
                )

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation epoch (re-draws phases reproducibly)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, index: int, occurrence: int) -> np.random.Generator:
        return np.random.default_rng([self.config.seed, self._epoch, index, occurrence])

    def window_at(self, index: int, occurrence: int = 0) -> Window:
        """Carve the window for ``index`` under the current epoch's augmentation.

        ``occurrence`` distinguishes repeated draws of the same record within an
        epoch (the sampler supplies it), so an oversampled record is re-phased on
        every draw instead of yielding identical copies. It is part of the RNG
        key, so the result stays exactly reproducible.
        """
        r = self.records[index]
        lead_range = window_lead_range(
            locus_offset=r.locus_offset,
            locus_length=r.locus_length,
            context_length=len(r.context_seq),
            clipped_start=r.clipped_start,
            clipped_end=r.clipped_end,
            window=self.config.window_nt,
        )
        assert lead_range is not None  # guaranteed by __init__
        if self.augment:
            rng = self._rng(index, occurrence)
            lead = sample_lead(rng, lead_range)
            rc = bool(rng.integers(2)) if self.config.both_strands else False
        else:
            lead = deterministic_lead(
                lead_range, window=self.config.window_nt, locus_length=r.locus_length
            )
            rc = False
        return carve_window(
            context_seq=r.context_seq,
            locus_offset=r.locus_offset,
            locus_length=r.locus_length,
            label_string=r.label_string,
            lead=lead,
            record_id=r.record_id,
            clipped_start=r.clipped_start,
            clipped_end=r.clipped_end,
            window=self.config.window_nt,
            rc=rc,
        )

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        """Fetch one sample.

        ``key`` is either a bare index (occurrence 0 — direct/eval access) or an
        ``(index, occurrence)`` pair as yielded by :class:`WeightedIndexSampler`.
        ``DataLoader`` passes sampler keys through verbatim, so both forms arrive
        here unchanged.
        """
        if isinstance(key, tuple):
            index, occurrence = key
        else:
            index, occurrence = key, 0
        w = self.window_at(index, occurrence)
        return {
            "input_ids": w.input_ids,
            "labels": w.labels,
            "real_mask": w.real_mask,
            "record_id": w.record_id,
            "parent_record_id": w.parent_record_id,
            "lead": w.lead,
            "reverse_complement": w.reverse_complement,
            "zero_flanked": w.zero_flanked,
        }

    # ── Curriculum ───────────────────────────────────────────────────────────
    def weights(self) -> np.ndarray:
        """The PRD §11 per-record curriculum weights for this dataset."""
        return curriculum_weights(
            phyla=[r.phylum for r in self.records],
            klasses=[r.klass for r in self.records],
            amino_acids=[r.cognate_aa for r in self.records],
            phylum_alpha=self.config.phylum_alpha,
            klass_alpha=self.config.klass_alpha,
            aa_alpha=self.config.aa_alpha,
        )

    def sampler(self, *, num_samples: int | None = None) -> WeightedIndexSampler:
        """The seeded rare-class + balanced-phylum sampler over this dataset."""
        return WeightedIndexSampler(self.weights(), num_samples=num_samples, seed=self.config.seed)

    # ── Provenance (ADR-0004 D7) ─────────────────────────────────────────────
    def provenance_rows(self) -> list[dict[str, Any]]:
        """One variant→parent→fold row per dataset index.

        Each row carries the emitted variant's ``record_id``, its
        ``parent_record_id``, and the parent's six fold-scheme values — the exact
        shape ``tests/ml/test_no_leakage.py::variant_parent_fold_mismatches``
        checks. Because a variant is a re-windowing of its parent (never a new
        locus), it inherits the parent's fold by construction; this method makes
        that inheritance explicit and testable rather than merely asserted.
        """
        rows = []
        for i, r in enumerate(self.records):
            w = self.window_at(i)
            row = {
                RECORD_ID_COL: w.record_id,
                PARENT_ID_COL: w.parent_record_id,
                CLUSTER_COL: r.cluster_id,
            }
            row.update(dict(zip(FOLD_SCHEME_COLUMNS, r.folds, strict=True)))
            rows.append(row)
        return rows


def collate_windows(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate :class:`Stage1WindowDataset` samples into a batch of torch tensors.

    ``input_ids`` → ``(B, W)`` int64 (embedding-ready), ``labels`` → ``(B, W)``
    int64 with ``IGNORE_INDEX`` preserved at zero-flanked positions (the seg head's
    ``ignore_index=-100`` consumes it directly), ``real_mask`` → ``(B, W)`` bool.
    Torch is imported lazily so the module stays bare-CI importable.
    """
    import torch

    if len(batch) == 0:
        raise ValueError("collate_windows needs a non-empty batch")
    return {
        "input_ids": torch.as_tensor(
            np.stack([np.asarray(b["input_ids"]) for b in batch]), dtype=torch.long
        ),
        "labels": torch.as_tensor(
            np.stack([np.asarray(b["labels"]) for b in batch]), dtype=torch.long
        ),
        "real_mask": torch.as_tensor(
            np.stack([np.asarray(b["real_mask"]) for b in batch]), dtype=torch.bool
        ),
        "record_id": [b["record_id"] for b in batch],
        "parent_record_id": [b["parent_record_id"] for b in batch],
    }
