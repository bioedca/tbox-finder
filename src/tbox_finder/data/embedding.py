"""The §9.1 decoy embedding and its matched junction control (P2-10d′-b, ADR-0005 A7).

PRD §9.1 names four training-negative classes, but every record in `decoys_v0.parquet`
is carved at *locus* length — 57–550 nt — while Stage-1 trains on a **1024-nt** window,
and :func:`~tbox_finder.data.negatives.background_record` refuses a length mismatch by
design. So "sample windows at ~10:1" had no referent for three of the four classes. The
user's **R2** decision (2026-07-20) resolves it: a decoy is **embedded at a random phase
inside a 1024-nt real genomic window**, every position labelled background.

**The objection R2 has to answer, and the measurement that answers it.** This repo has
already ruled, in `workflow/rules/data.smk`'s `p2_flank_context` docstring, that "a
0th-order background splice would hand the model a trivially separable boundary cue and
inflate GATE-4/GATE-1" — which is *why* P2-00 spent 23,535 NCBI requests sourcing real
flank rather than synthesising it. Embedding reintroduces two junctions per window **in
negatives only**, so if the junction were readable the cue would be perfectly
class-correlated. It was therefore measured before being adopted (ADR-0005 A7; k-mer
logistic regression, 5-fold CV AUROC, n = 2,500/arm, seed 20260721):

===========================================================  =======  =======
arm                                                              k=4      k=6
===========================================================  =======  =======
null — two disjoint sets of *unspliced* real windows            0.493    0.520
junction alone — background→background, same two junctions      0.487    0.519
junction alone, local 128-nt window centred on the junction     0.487        —
===========================================================  =======  =======

The junction sits **on the null**. The earlier ruling was about a *0th-order synthetic*
splice; a real-DNA-into-real-DNA junction leaves no compositional trace, and R2's
construction never abuts synthetic DNA against real DNA at a labelled boundary.

That is a **composition** bound and this module does not pretend otherwise: a k-mer
frequency over a 128-nt window dilutes a single-position discontinuity that a
convolutional or state-space receptive field could still read. Hence the two-tier gate
(A7 pin 7) — this module builds the arms, :mod:`tbox_finder.eval.junction_probe` scores
the pre-flight composition tier, and the post-run tier scores the trained checkpoint on
the same arms with the pass rule "the decoy must beat its own junction null".

**The arms.** Three windows share a host, a phase and an insert length, and differ only
in what occupies ``[phase, phase + insert_len)``:

``PLAIN``
    the host window, unspliced — no junction, no decoy.
``CONTROL``
    the segment replaced by **real DNA taken from the same offsets of a different mined
    window** — two junctions, no decoy. This is the matched control ADR-0005 A7 makes
    mandatory, and matchedness is *asserted*, not assumed: `tests/unit/test_embedding.py`
    pins that treatment and control agree on host, phase, insert length, junction offsets
    and flanking bases, and that they differ **only** inside the replaced interval. The
    generator is sabotage-tested too ([[control-matchedness-must-be-asserted]]) — a
    control the generator copies from the treatment arm proves nothing, and a control
    whose donor segment came from the host itself would be no splice at all.
``DECOY``
    the segment replaced by a real §9.1 decoy. This arm, and only this arm, enters
    training.

**Which classes are embedded (ADR-0005 A7 pin 3/4/5, user sign-off 2026-07-21).**

``dinuc_shuffled``, ``structured_rna``
    embedded, one window per admitted decoy record.
``gc_background``
    **share 0.** §9.1's class 1 is "random windows matched to T-box GC + length (dominant
    real-world negative)"; this pool is a seeded 0th-order i.i.d. composition null built
    at P0 *because no genome panel existed then* (`decoys.py`'s own docstring says so).
    One does now: the 15,708 mined `genomic_window` records are real 1024-nt DNA carved
    from the same replicons as the loci — better matched than a bootstrap, and measured
    to carry **no** compositional shortcut (they are the null arm above). So class 1 is
    served by real DNA, and the synthetic surrogate is retired from *training* only. The
    pool itself is untouched and remains the GATE-1 benchmark denominator (ADR-0005 D7's
    pinned 100:1 prevalence).
``leader_decoy``
    **share 0 — a label collision, not a decoy.** All 8 records are T-box-derived, and 2
    are *exact substrings of training-corpus positives* (`neg_tbdb_-1baL_Tx` inside 2 of
    the 23,535 `master_clean_v0` records, `neg_tbdb_GwZsgp7s` inside 2 — verified against
    the corpus at P2-10d′-b). `neg_anchor_6UFM` / `neg_anchor_6UFG` are strand 0 of PDB
    6UFM / 6UFG, which this repo's own `tests/fixtures/pdb_element_extents/README.md`
    classifies as **class-II ultrashort translational T-box riboswitches**; the remaining
    four are TBDB v1 T-box records. They were negatives relative to *tboxevo's IDTM
    covariance model*, never relative to "is a T-box". Embedding them would label verbatim
    positive sequence as background — a direct contradiction, and a breach of §9.1's own
    masking rule ("all known T-box loci … masked from every negative/decoy pool"). It
    would also hand the model an AUROC-1.000 memorisation target: 8 distinct inserts
    across 83,030 draws per epoch. ADR-0005 **A7 re-scopes D14's retention clause** to
    "the leader pool *as re-sourced at P2-10b′*", with the non-delivery disclosed.

Bare-CI importable: numpy + stdlib only (pandas is the caller's problem).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder.data.negatives import (
    NEGATIVE_CLUSTER_SIGN,
    NEGATIVE_ID_PREFIX,
    NegativeInjectionError,
    background_record,
)
from tbox_finder.data.window_dataset import WINDOW_NT, CorpusRecord

#: Bumped whenever the embedding report gains a field a reader could otherwise mistake
#: for a rule that ran. "1.0" is P2-10d′-b's first shape.
SCHEMA_VERSION = "1.0"
STEP = "P2-10d'-b"

# ── Arm names ────────────────────────────────────────────────────────────────────────
ARM_PLAIN = "plain"
ARM_CONTROL = "junction_control"
ARM_DECOY = "decoy"

# ── decoys_v0.parquet columns (decoys.py::_record) ───────────────────────────────────
DECOY_ID_COL = "decoy_id"
DECOY_POOL_COL = "pool"
DECOY_SEQUENCE_COL = "sequence"
DECOY_MASKED_COL = "masked"

#: The §9.1 classes embedded into the round-0 training stream (ADR-0005 A7 pin 3).
TRAINING_DECOY_POOLS: tuple[str, ...] = ("dinuc_shuffled", "structured_rna")

#: The §9.1 classes deliberately given share 0, each with the reason a reader needs in
#: order to tell "excluded on purpose" from "the join silently found nothing". A pool
#: named here that is *absent* from the parquet is itself an error — see
#: :func:`embed_decoy_rows` — because a rename upstream would otherwise read as a
#: successful exclusion ([[clauses-must-guard-emptiness]]).
EXCLUDED_DECOY_POOLS: dict[str, str] = {
    "gc_background": (
        "superseded for training by the 15,708 real mined genomic windows; a 0th-order "
        "i.i.d. surrogate built at P0 when no genome panel existed. Retained unchanged "
        "as the ADR-0005 D7 GATE-1 benchmark denominator."
    ),
    "leader_decoy": (
        "label collision: all 8 records are T-box-derived and 2 are exact substrings of "
        "training-corpus positives. Re-sourcing deferred to P2-10b′ (ADR-0005 A7 "
        "re-scopes D14's leader-pool retention clause)."
    ),
}

#: Refusal reasons, counted rather than silently dropped.
REASON_EXCLUDED_POOL = "pool_excluded_by_a7"
REASON_MASKED = "masked_known_locus"
REASON_EMPTY_ID = "missing_decoy_id"
REASON_TOO_LONG = "insert_longer_than_window"
REASON_EMPTY_INSERT = "empty_insert"
REASON_NON_ACGTN = "non_acgtn_characters"

_ACGTN = frozenset("ACGTN")

#: Salt separating the embedding's host/phase draw from every other RNG stream in the
#: project (the curriculum draw, the mix interleave `MIX_STREAM_SALT`, and the per-window
#: augmentation draw). Shared streams correlate across epochs.
EMBED_STREAM_SALT = 0x10E1

#: … and the control arm's donor draw, which must be independent of the host draw or a
#: donor could track its host.
CONTROL_STREAM_SALT = 0x10E2

#: How many donors the control arm may try before refusing. Only exhausted when the host
#: pool is degenerate over the requested interval; see :func:`junction_control`.
_MAX_DONOR_DRAWS = 64


class EmbeddingError(ValueError):
    """Raised when an embedded window or its matched control cannot be built honestly."""


@dataclass(frozen=True)
class EmbeddedWindow:
    """One window of an arm, carrying every field its matched counterpart must agree on.

    ``host_id`` / ``phase`` / ``insert_len`` are the matchedness contract: a
    :data:`ARM_CONTROL` window and its :data:`ARM_DECOY` counterpart share all three (and
    therefore both junction offsets), and differ only in ``sequence[phase:phase+insert_len]``.
    ``insert_id`` names what was actually spliced in — the decoy id for the treatment arm,
    the donor window's candidate id for the control arm — so no arm is identifiable only
    by its position in a list.
    """

    arm: str
    sequence: str
    host_id: str
    host_parent_record_id: str
    insert_id: str
    insert_pool: str
    phase: int
    insert_len: int

    @property
    def junctions(self) -> tuple[int, int]:
        """The two splice offsets, as half-open interval bounds into ``sequence``."""
        return (self.phase, self.phase + self.insert_len)


def normalise_insert(sequence: str) -> str:
    """Upper-case and map ``U`` → ``T``.

    Not cosmetic. `structured_rna` carries 1,784 ``U`` (it is subsampled from Rfam FASTA)
    and `leader_decoy` is *entirely* RNA-alphabet. ``background_record`` refuses anything
    outside ACGTN, and `encode_bases` would otherwise map every ``U`` to the ``N`` id
    without a word — so an un-normalised structured-RNA insert would train as a run of Ns
    in the middle of real DNA, which is both wrong and trivially separable.
    """
    return str(sequence).upper().replace("U", "T")


def splice(host: str, insert: str, phase: int) -> str:
    """Replace ``host[phase:phase+len(insert)]`` with ``insert``.

    **Replacement, not insertion.** The window must stay exactly ``len(host)`` nt:
    ``background_record`` requires it, and an insertion would either lengthen the window
    (refused) or force a truncation elsewhere that no id names. Length invariance is what
    lets the embedded negative take the *same* ``carve_window`` path a positive takes.
    """
    if phase < 0:
        raise EmbeddingError(f"phase must be non-negative; got {phase}")
    n = len(insert)
    if n == 0:
        raise EmbeddingError("insert must be non-empty")
    if phase + n > len(host):
        raise EmbeddingError(
            f"insert of {n} nt at phase {phase} runs off the end of a {len(host)}-nt host. "
            "The splice is a replacement, so the insert must fit entirely inside the window."
        )
    out = host[:phase] + insert + host[phase + n :]
    # Cheap, and it is the invariant every downstream length rule depends on.
    if len(out) != len(host):
        raise EmbeddingError(
            f"splice changed the window length {len(host)} -> {len(out)}; this is a bug"
        )
    return out


def _stable_stream(*parts: Any, salt: int) -> np.random.Generator:
    """A generator keyed on the *content* of ``parts``, not on iteration order.

    Keying on a list index would make every draw depend on the pool's file order, so a
    re-sorted or filtered pool would silently re-pair every decoy with a different host
    while the report still claimed the same seed.
    """
    digest = hashlib.shake_256("\x1f".join(str(p) for p in parts).encode()).digest(8)
    return np.random.default_rng([salt, int.from_bytes(digest, "big")])


def plan_placement(
    *, decoy_id: str, insert_len: int, n_hosts: int, seed: int, window: int = WINDOW_NT
) -> tuple[int, int]:
    """Choose ``(host_index, phase)`` for one decoy, deterministically.

    ``phase ~ U[0, window - insert_len]`` inclusive, so every admissible placement —
    including flush-left and flush-right, which are the two that produce a *single*
    junction rather than two — is reachable. Excluding them would make "two junctions"
    an invariant of the negative class that the scan distribution does not share.
    """
    if insert_len <= 0 or insert_len > window:
        raise EmbeddingError(
            f"decoy {decoy_id!r}: insert length {insert_len} does not fit a {window}-nt window"
        )
    if n_hosts <= 0:
        raise EmbeddingError(f"decoy {decoy_id!r}: no hosts available")
    rng = _stable_stream(seed, decoy_id, salt=EMBED_STREAM_SALT)
    host_index = int(rng.integers(0, n_hosts))
    phase = int(rng.integers(0, window - insert_len + 1))
    return host_index, phase


def _host_permutation(n_hosts: int, seed: int) -> np.ndarray:
    """A seeded permutation of host indices, for the measurement's one-host-per-window mode."""
    return np.random.default_rng([EMBED_STREAM_SALT, int(seed)]).permutation(n_hosts)


def embed_decoy_rows(
    decoy_rows: Iterable[Mapping[str, Any]],
    hosts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    window: int = WINDOW_NT,
    pools: Sequence[str] = TRAINING_DECOY_POOLS,
    unique_hosts: bool = False,
) -> tuple[list[EmbeddedWindow], dict[str, Any]]:
    """Build one :data:`ARM_DECOY` window per admitted decoy record.

    ``hosts`` are mined-pool rows that have **already passed** the P2-10d′-a admission
    rules (unmasked, not a designed control, ``parent_nested_train``) — this function does
    not re-derive them, it inherits them, so an embedded negative's §9.2 provenance is the
    host's and cannot be weaker than a plain negative's.

    **Host reuse (``unique_hosts=False``, the training default).** Each decoy draws its
    host independently, so a host window may appear once unspliced in the plain arm and
    once or more decoy-bearing. That is deliberate for *training*: it makes the decoy the
    only difference between the two, and it is the composition ADR-0005 A7 pin 3 signs off
    (15,708 plain + 5,000 embedded = 20,708 admitted records, drawn uniformly, so shares
    follow pool sizes and no per-class weight knob exists). Repetition inside the negative
    class creates no positive-vs-negative cue.

    **``unique_hosts=True`` is for the pre-flight MEASUREMENT, not for training.** A
    cross-validated k-mer probe is not indifferent to repetition: two windows sharing a
    host are near-duplicates, and when one lands in the training fold and the other in the
    test fold the classifier recognises the host rather than the splice. With the spliced
    arms carrying such pairs and the plain arm carrying none, that asymmetry alone drove
    the fixture's junction AUROC to **0.70** while nothing about the junction had changed.
    Under this flag hosts are assigned by a seeded permutation, one per window, and the
    caller must supply at least as many hosts as there are admitted decoys.

    Returns ``(windows, report)``; every refusal is counted by reason.
    """
    if window <= 0:
        raise EmbeddingError(f"window must be positive; got {window}")
    if not hosts:
        raise EmbeddingError(
            "no admitted host windows — an embedded negative's DNA is its host's, so a "
            "pool that admits nothing cannot supply one. Check the P2-10d′-a filters."
        )
    wanted = tuple(pools)
    out: list[EmbeddedWindow] = []
    excluded: dict[str, int] = {}
    seen_pools: set[str] = set()
    per_pool: dict[str, int] = {}
    seen_ids: set[str] = set()

    def _drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for row in decoy_rows:
        pool = str(row.get(DECOY_POOL_COL) or "").strip()
        seen_pools.add(pool)
        did = str(row.get(DECOY_ID_COL) or "").strip()
        if not did:
            _drop(REASON_EMPTY_ID)
            continue
        if pool not in wanted:
            _drop(REASON_EXCLUDED_POOL)
            continue
        if bool(row.get(DECOY_MASKED_COL)):
            _drop(REASON_MASKED)
            continue
        insert = normalise_insert(row.get(DECOY_SEQUENCE_COL) or "")
        if not insert:
            _drop(REASON_EMPTY_INSERT)
            continue
        if len(insert) > window:
            _drop(REASON_TOO_LONG)
            continue
        if set(insert) - _ACGTN:
            _drop(REASON_NON_ACGTN)
            continue
        if did in seen_ids:
            raise EmbeddingError(
                f"duplicate decoy_id {did!r} — a duplicated insert is silently oversampled "
                "and the realized per-class share would describe a pool that does not exist"
            )
        seen_ids.add(did)
        host_index, phase = plan_placement(
            decoy_id=did, insert_len=len(insert), n_hosts=len(hosts), seed=seed, window=window
        )
        if unique_hosts:
            if len(out) >= len(hosts):
                raise EmbeddingError(
                    f"unique_hosts=True needs at least one host per admitted decoy; "
                    f"{len(hosts)} hosts cannot carry {len(out) + 1}. Supply more hosts or "
                    "cap the decoy pool."
                )
            host_index = int(_host_permutation(len(hosts), seed)[len(out)])
        host = hosts[host_index]
        host_seq = str(host.get("sequence") or "").upper()
        if len(host_seq) != window:
            raise EmbeddingError(
                f"host {host.get('candidate_id')!r} is {len(host_seq)} nt, not {window}; "
                "hosts must be admitted window-length mined records"
            )
        out.append(
            EmbeddedWindow(
                arm=ARM_DECOY,
                sequence=splice(host_seq, insert, phase),
                host_id=str(host.get("candidate_id") or "").strip(),
                host_parent_record_id=str(host.get("source_record_id") or "").strip(),
                insert_id=did,
                insert_pool=pool,
                phase=phase,
                insert_len=len(insert),
            )
        )
        per_pool[pool] = per_pool.get(pool, 0) + 1

    # A pool A7 names as excluded that is not in the parquet at all means the upstream
    # name moved. Silence there would read as a successful exclusion while the records
    # sailed through under a new name, so it is an error, not a warning.
    missing_excluded = sorted(set(EXCLUDED_DECOY_POOLS) - seen_pools)
    if missing_excluded:
        raise EmbeddingError(
            f"ADR-0005 A7 excludes pool(s) {missing_excluded} from the training mix, but no "
            "row in this decoy pool carries that name. Either the pool was renamed upstream "
            "— in which case its records are entering the mix unrecognised — or the file is "
            "not decoys_v0. Refusing rather than reporting a vacuous exclusion."
        )
    missing_wanted = sorted(set(wanted) - seen_pools)
    if missing_wanted:
        raise EmbeddingError(
            f"pool(s) {missing_wanted} are pinned into the training mix but absent from the "
            "decoy pool; the realized composition would silently differ from A7 pin 3."
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "window_nt": int(window),
        "seed": int(seed),
        "embedded_pools": list(wanted),
        "excluded_pools": {k: v for k, v in sorted(EXCLUDED_DECOY_POOLS.items())},
        "n_embedded": len(out),
        "n_embedded_by_pool": dict(sorted(per_pool.items())),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "n_excluded": int(sum(excluded.values())),
        "n_hosts_available": len(hosts),
        "n_distinct_hosts_used": len({w.host_id for w in out}),
        "unique_hosts": bool(unique_hosts),
        "insert_len_min": min((w.insert_len for w in out), default=0),
        "insert_len_max": max((w.insert_len for w in out), default=0),
    }
    return out, report


def load_decoy_rows(parquet_path: str | Path) -> list[dict[str, Any]]:
    """Read `decoys_v0.parquet` and return its rows, refusing a file of the wrong shape.

    Column presence is checked at **file** level rather than per row: a decoy parquet
    missing ``masked`` would treat every record as unmasked and fail OPEN on the §9.1
    locus-mask guard, and a per-row refusal would report that as thousands of identical
    data problems instead of one stale file.
    """
    import pandas as pd  # lazy: keeps the geometry tier bare-CI importable

    path = Path(parquet_path)
    if not path.exists():
        raise EmbeddingError(f"decoy pool not found: {path}")
    frame = pd.read_parquet(path)
    missing = [
        c
        for c in (DECOY_ID_COL, DECOY_POOL_COL, DECOY_SEQUENCE_COL, DECOY_MASKED_COL)
        if c not in frame.columns
    ]
    if missing:
        raise EmbeddingError(
            f"decoy pool {path} is missing required column(s) {missing}; found "
            f"{sorted(frame.columns)}. Loading without {DECOY_MASKED_COL!r} in particular "
            "would fail OPEN on the §9.1 union-prior mask."
        )
    return list(frame.to_dict("records"))


def junction_control(
    embedded: Sequence[EmbeddedWindow],
    hosts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    window: int = WINDOW_NT,
) -> list[EmbeddedWindow]:
    """Build the matched background→background control for each embedded window.

    Held fixed against the treatment arm: the **host window**, the **phase**, the
    **insert length**, and therefore both junction offsets and every base outside
    ``[phase, phase+insert_len)``. Varied: **only** what occupies that interval — real DNA
    here, a decoy there. So a model that separates the two arms is reading decoy content,
    and a model that separates *this* arm from plain background is reading the junction.

    The donor segment is taken from **the same offsets of a different mined window**
    (``donor[phase:phase+insert_len]``), not from a re-sampled or shuffled sequence: a
    composition-matched synthetic donor would make the control a different experiment —
    the arm has to be real DNA, because "real DNA spliced into real DNA" is the null the
    junction claim is measured against.

    The donor is refused if it resolves to the host itself: splicing a window's own
    offsets back into it is the identity, and a control that is silently *not spliced*
    would report the junction as invisible no matter what the junction did.
    """
    if len(hosts) < 2:
        raise EmbeddingError(
            "the junction control needs at least two mined windows — the donor segment "
            "must come from a window other than the host"
        )
    by_id = {str(h.get("candidate_id") or "").strip(): h for h in hosts}
    out: list[EmbeddedWindow] = []
    for w in embedded:
        rng = _stable_stream(seed, w.insert_id, w.host_id, salt=CONTROL_STREAM_SALT)
        host_row = by_id.get(w.host_id)
        if host_row is None:
            raise EmbeddingError(
                f"host {w.host_id!r} of an embedded window is not in the host pool; the "
                "control cannot be matched to a host it cannot find"
            )
        host_seq = str(host_row.get("sequence") or "").upper()
        segment = None
        for _ in range(_MAX_DONOR_DRAWS):
            cand = hosts[int(rng.integers(0, len(hosts)))]
            cid = str(cand.get("candidate_id") or "").strip()
            if not cid or cid == w.host_id:
                continue
            donor_seq = str(cand.get("sequence") or "").upper()
            if len(donor_seq) != window:
                raise EmbeddingError(f"donor {cid!r} is {len(donor_seq)} nt, not {window}")
            candidate_segment = donor_seq[w.phase : w.phase + w.insert_len]
            # A donor drawn from a *different* window can still carry the *same* bases at
            # these offsets — short inserts inside a repeat make it likely, and it happened
            # in the first fixture written for this function. The resulting "control" is
            # the identity splice: it carries no junction, so it would report the junction
            # as invisible whatever the junction actually did, and every geometry
            # assertion about it would still pass. Re-draw rather than emit it.
            if candidate_segment != host_seq[w.phase : w.phase + w.insert_len]:
                donor, segment = cand, candidate_segment
                break
        if segment is None:
            raise EmbeddingError(
                f"no donor in {_MAX_DONOR_DRAWS} draws differs from host {w.host_id!r} "
                f"over [{w.phase}, {w.phase + w.insert_len}). Every candidate would have "
                "produced an identity splice — a control with no junction. This means the "
                "host pool is degenerate over that interval, not that the junction is safe."
            )
        out.append(
            EmbeddedWindow(
                arm=ARM_CONTROL,
                sequence=splice(host_seq, segment, w.phase),
                host_id=w.host_id,
                host_parent_record_id=w.host_parent_record_id,
                insert_id=str(donor.get("candidate_id") or "").strip(),
                insert_pool=ARM_CONTROL,
                phase=w.phase,
                insert_len=w.insert_len,
            )
        )
    return out


def embedded_negative_records(
    embedded: Sequence[EmbeddedWindow],
    *,
    cluster_id_start: int,
    window: int = WINDOW_NT,
    id_prefix: str = NEGATIVE_ID_PREFIX,
) -> list[CorpusRecord]:
    """Turn embedded windows into all-background :class:`CorpusRecord`s.

    ``cluster_id_start`` is the ordinal the plain mined arm stopped at, so embedded and
    plain negatives share one contiguous negative cluster namespace and no id is reused.
    ``source_record_id`` is the **host's** parent corpus record: the DNA is the host's, so
    the §9.2 provenance claim is the host's and is checkable against the same split table.
    """
    if cluster_id_start < 0:
        raise EmbeddingError(f"cluster_id_start must be non-negative; got {cluster_id_start}")
    records: list[CorpusRecord] = []
    for i, w in enumerate(embedded):
        if w.arm != ARM_DECOY:
            raise EmbeddingError(
                f"only {ARM_DECOY!r} windows enter training; got arm {w.arm!r}. The "
                f"{ARM_CONTROL!r} arm is a diagnostic — training on it would make the "
                "junction uninformative and destroy the measurement it exists to supply."
            )
        try:
            records.append(
                background_record(
                    record_id=f"{id_prefix}:emb:{w.insert_pool}:{w.insert_id}@{w.host_id}:{w.phase}",
                    sequence=w.sequence,
                    cluster_id=NEGATIVE_CLUSTER_SIGN * (cluster_id_start + i + 1),
                    source_record_id=w.host_parent_record_id,
                    window=window,
                )
            )
        except NegativeInjectionError as exc:
            raise EmbeddingError(
                f"embedded window {w.insert_id!r}@{w.host_id!r} is not an honest negative: {exc}"
            ) from exc
    return records
