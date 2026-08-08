"""The per-candidate criterion-(b) producer, and D3's held-out-canonical freeze.

Two jobs, deliberately in one module because they must agree about the predicate:

``run-shard`` / ``merge``
    Read each candidate's A3/A4 ``msa.sto`` comparative consensus, localize D3's two
    inputs through :mod:`tbox_finder.mining.architecture`, and write the
    ``candidate_id → status`` table the round's ``relaxed_architecture`` disjunct
    reads — in the identical shape the (a) and (c) producers write.

``freeze``
    ADR-0006 D3 requires the predicate be *"frozen on the held-out canonical set and
    unit-tested"*.  This measures it there — on the ``source == "corpus"`` ∧
    ``nested_role == "heldout"`` carve of the committed ADR-0004 split table — and
    writes the distributions.  ⚠ **It pins nothing** (§10.3): no threshold in this
    module has a default, the round supplies every value, and the value it used rides
    in the report's provenance.  A4 ships D6's short-Stem-I threshold the same way and
    explicitly does **not** consume D17's P6 delegation.

Why the freeze can run before the production MSAs exist
-------------------------------------------------------
The freeze reads each held-out canonical record's **curated** ``Structure`` (WUSS over
its own ``Sequence``), not a de-novo consensus — the canonical set is exactly the
CM-derived corpus, which is what D3 names as the freezing set.  The **production**
path never touches it: ``run-shard`` reads only ``msa.sto``.  ⚠ That asymmetry is the
point and it is a disclosed limitation, not a leak: calibrating on curated structure
measures whether the *localizer* recovers a known architecture, and says nothing about
how often a de-novo consensus resolves one.  The second number is only measurable once
the full-corpus (a)/(b) supply run has happened, and it is **not** claimed here.

Coordinates are never trusted
-----------------------------
``master_clean_v0``'s element coordinates index a different frame than its own
``Sequence`` column — measured: ``discrim_start``/``discrim_end`` reproduces the
``discriminator`` on **5 of 2,000** rows, and ``ingest.COORD_COLS_ZERO_SENTINEL``
records that a literal ``0`` means "TBDB found no value".  The freeze therefore
anchors the curated discriminator by **exact substring match** into the element
sequence, never by coordinate.

PRD §13.2/§13.3(b); ADR-0006 D3/D6/D9/A3/A4; ADR-0005 D14 (the spare rule this feeds).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from tbox_finder.mining import architecture
from tbox_finder.mining.covariation_producer import MSA_FILENAME as _MSA_FILENAME
from tbox_finder.mining.spare_rule import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
)
from tbox_finder.power import MIN_REAL_HOMOLOG_N

__all__ = [
    "ArchitectureRunConfig",
    "DEFAULT_CORPUS",
    "DEFAULT_FP_MANIFEST",
    "DEFAULT_MSA_DIR",
    "DEFAULT_SPLIT_TABLE",
    "FREEZE_REPORT",
    "ProducerError",
    "REQUIRED_CONFIG_KEYS",
    "SUPPLY_CLAUSES",
    "build_parser",
    "build_status_table",
    "candidate_msa_path",
    "derive_relaxed_arch_supply_available",
    "evaluate_candidate",
    "freeze_report",
    "heldout_canonical",
    "load_status_map",
    "main",
    "merge_status_tables",
    "read_manifest",
    "run_shard",
    "shard_candidates",
    "validate_status_payload",
]

STEP = "P3-15'-d"
ADR = "ADR-0006 D3/D6 (criterion (b)), A3/A4 (the de-novo consensus instrument); ADR-0005 D14"
SCHEMA_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_FP_MANIFEST = "data/processed/mining/round0_fp_manifest.json"
DEFAULT_MSA_DIR = "data/interim/homolog_msa/rounds"
DEFAULT_CORPUS = "data/processed/master_clean_v0.parquet"
DEFAULT_SPLIT_TABLE = "data/processed/splits/split_assignments.parquet"
FREEZE_REPORT = "reports/p3/architecture_freeze.json"

#: The env this producer's legs run in (CLAUDE.md §3.2 rule = env). Recorded in every
#: provenance block so a report names the environment that produced it — ``stage2_producer``
#: sets this and it is the better precedent; ``synteny_producer`` leaves it ``None``.
ENV_LOCK = "envs/data.conda-lock.yml"

#: The status-table filename the round's ``--relaxed-arch-status`` expects.
DEFAULT_STATUS_TABLE = "architecture_status.json"

#: The per-candidate consensus ``covariation_producer.align_shard`` writes — **imported**
#: from the module that writes it, not re-typed.  A second literal would let a rename in
#: the (a) producer make every (b) candidate silently ``unavailable``: the shard tables
#: would still merge, the round would still run, and the yield would read as an honest
#: zero.  Importing it means a rename surfaces as an ImportError instead.
MSA_FILENAME = _MSA_FILENAME


#: The rule parameters every shard table's ``config`` must name.  Listed explicitly rather
#: than derived from whatever keys a payload happens to carry: a required-key set read from
#: its own evidence is vacuously satisfied exactly when the evidence is missing.
REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "stem_i_nt_threshold",
    "min_named_helices",
    "min_helix_pairs",
    "bulge_min_nt",
    "bulge_max_nt",
    "ncca_pairing_nt",
)


class ProducerError(RuntimeError):
    """A criterion-(b) production step could not be carried out as specified."""


# ═════════════════════════════════════════════════════════════════════════════
# Run configuration — every value required, none defaulted
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ArchitectureRunConfig:
    """The values a (b) round must choose, recorded beside every status it produces.

    None of them has a default anywhere in this step.  ADR-0006 D3 assigns the
    predicate's freezing to the held-out canonical set and **D17** freezes D6's
    threshold at P6; A4 resolves that by having the round supply the value and
    recording it, which is the discipline already used by ``--strand-policy``,
    ``substrate_prescan``'s ``score_threshold``, and criterion (c)'s three required
    parameters.  A default here would decide which candidates are mined without
    anyone choosing it.
    """

    stem_i_nt_threshold: int
    min_named_helices: int
    min_helix_pairs: int
    bulge_min_nt: int
    bulge_max_nt: int
    ncca_pairing_nt: int
    allow_wobble: bool = False
    min_sequences: int = MIN_REAL_HOMOLOG_N

    def __post_init__(self) -> None:
        """Refuse a mis-specified round rather than mining the corpus under it.

        An inverted bulge range (``--bulge-min-nt 9 --bulge-max-nt 5``) is empty, so **no**
        bulge satisfies it, every scored candidate resolves to ``failed`` ⇒ **mined**, and
        the round reports a full sweep it never actually evaluated.  That is the fail-OPEN
        direction this module exists to refuse, and it is reachable from the CLI because
        none of these values has a default.  The same holds for a non-positive count.
        """
        if self.bulge_min_nt > self.bulge_max_nt:
            raise ProducerError(
                f"bulge range is empty: min {self.bulge_min_nt} > max {self.bulge_max_nt} — "
                "no bulge could satisfy it and every candidate would resolve to 'failed' "
                "⇒ mined"
            )
        non_positive = {
            name: value
            for name, value in (
                ("min_named_helices", self.min_named_helices),
                ("min_helix_pairs", self.min_helix_pairs),
                ("ncca_pairing_nt", self.ncca_pairing_nt),
                ("bulge_min_nt", self.bulge_min_nt),
                ("min_sequences", self.min_sequences),
            )
            if int(value) < 1
        }
        if non_positive:
            raise ProducerError(f"these must each be >= 1: {dict(sorted(non_positive.items()))}")

    @property
    def bulge_size_range(self) -> tuple[int, int]:
        return (int(self.bulge_min_nt), int(self.bulge_max_nt))

    def as_dict(self) -> dict[str, Any]:
        return {
            "stem_i_nt_threshold": int(self.stem_i_nt_threshold),
            "min_named_helices": int(self.min_named_helices),
            "min_helix_pairs": int(self.min_helix_pairs),
            "bulge_min_nt": int(self.bulge_min_nt),
            "bulge_max_nt": int(self.bulge_max_nt),
            "ncca_pairing_nt": int(self.ncca_pairing_nt),
            "allow_wobble": bool(self.allow_wobble),
            "min_sequences": int(self.min_sequences),
            # Stamped, not inferred: a reader must be able to see that no `cmalign`
            # bit-score is on this path without re-reading the code (D3; A4).
            "instrument": "de-novo mlocarna #=GC SS_cons (ADR-0006 A3/A4)",
            "cmalign_on_path": False,
            "rf00230_on_path": False,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Production: manifest → per-candidate status
# ═════════════════════════════════════════════════════════════════════════════
def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProducerError(f"{path}: cannot read manifest ({exc})") from exc
    rows = payload.get("candidates") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ProducerError(f"{path}: manifest carries no candidate list")
    return [dict(r) for r in rows if isinstance(r, Mapping)]


def candidate_msa_path(msa_dir: str | Path, candidate_id: str) -> Path:
    """Where the (a) producer's promoted consensus for ``candidate_id`` lives.

    The slug is **imported** from the (a) producer rather than re-derived: the two
    must agree exactly or this producer silently reads no MSAs and reports
    ``unavailable`` for the whole corpus — which spares everything and looks like a
    clean run.
    """
    from tbox_finder.mining.covariation_producer import candidate_slug

    return Path(msa_dir) / candidate_slug(str(candidate_id)) / MSA_FILENAME


def evaluate_candidate(
    candidate: Mapping[str, Any],
    *,
    msa_dir: str | Path,
    config: ArchitectureRunConfig,
) -> dict[str, Any]:
    """One manifest row → one status-table row.

    Every failure to *evaluate* — absent consensus, unreadable Stockholm, no
    ``SS_cons``, unbalanced structure, depth below the A2 Pin 2 floor — resolves to
    ``unavailable`` ⇒ **spared** under ADR-0005 D14, never to ``failed`` ⇒ mined.
    """
    candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "")
    if not candidate_id:
        raise ProducerError(f"manifest row carries no candidate_id: {dict(candidate)!r}")

    msa = candidate_msa_path(msa_dir, candidate_id)
    row: dict[str, Any] = {"candidate_id": candidate_id, "msa_present": msa.is_file()}
    if not msa.is_file():
        row.update({"status": STATUS_UNAVAILABLE, "reason": "no per-candidate consensus on disk"})
        return row

    try:
        consensus = architecture.parse_stockholm(msa)
        localization = architecture.localize(
            candidate_id,
            consensus,
            row=0,
            stem_i_extent_nt=_as_int_or_none(candidate.get("stem_i_extent_nt")),
            regulatory_mode=candidate.get("regulatory_mode"),
            stem_i_nt_threshold=config.stem_i_nt_threshold,
            class_ii=bool(candidate.get("class_ii", False)),
            min_named_helices=config.min_named_helices,
            min_helix_pairs=config.min_helix_pairs,
            bulge_size_range=config.bulge_size_range,
            ncca_pairing_nt=config.ncca_pairing_nt,
            allow_wobble=config.allow_wobble,
        )
    except (architecture.ArchitectureError, OSError, ValueError) as exc:
        # The load-bearing conversion is UPSTREAM: `parse_stockholm` already catches
        # OSError/UnicodeError/ValueError from the read and re-raises ArchitectureError, so
        # a permission fault, an I/O error, a file removed between the `is_file()` check
        # above and the open, or non-UTF-8 bytes all arrive here as ArchitectureError and
        # resolve ONE candidate to `unavailable` rather than losing the whole shard to
        # `main`'s exit 1. Review asked for the wider clause here; it is kept as defence for
        # any future caller that reaches `localize` without going through `parse_stockholm`,
        # but it is NOT what provides the guarantee — `test_parse_stockholm_converts_read_
        # errors` pins the line that does, and narrowing THIS clause alone does not regress
        # the behaviour (verified by sabotage: it stayed green, which is why the test moved).
        row.update({"status": STATUS_UNAVAILABLE, "reason": f"consensus unusable: {exc}"})
        return row

    status, detail = architecture.architecture_status(
        localization, min_sequences=config.min_sequences
    )
    row.update({"status": status, "n_sequences": localization.n_sequences, **detail})
    return row


def _as_int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def shard_candidates(
    candidates: Sequence[Mapping[str, Any]], shard_index: int, n_shards: int
) -> list[Mapping[str, Any]]:
    """Deterministic round-robin partition. Every candidate lands in exactly one shard."""
    if n_shards < 1 or not 0 <= shard_index < n_shards:
        raise ProducerError(f"bad shard {shard_index}/{n_shards}")
    return [c for i, c in enumerate(candidates) if i % n_shards == shard_index]


def run_shard(
    candidates: Sequence[Mapping[str, Any]],
    *,
    msa_dir: str | Path,
    config: ArchitectureRunConfig,
) -> list[dict[str, Any]]:
    return [evaluate_candidate(c, msa_dir=msa_dir, config=config) for c in candidates]


def build_status_table(
    rows: Sequence[Mapping[str, Any]], *, config: ArchitectureRunConfig
) -> dict[str, Any]:
    """The ``candidate_id → status`` table, in the (a)/(c) producers' shape."""
    status = {str(r["candidate_id"]): str(r["status"]) for r in rows}
    if len(status) != len(rows):
        seen = Counter(str(r["candidate_id"]) for r in rows)
        dupes = sorted(k for k, v in seen.items() if v > 1)
        raise ProducerError(f"duplicate candidate_id in shard rows: {dupes[:5]}")
    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "config": config.as_dict(),
        "n_candidates": len(rows),
        "status_counts": dict(sorted(Counter(status.values()).items())),
        "status": status,
        "rows": [dict(r) for r in rows],
    }


def merge_status_tables(
    table_paths: Sequence[str | Path], *, out_path: str | Path | None = None
) -> dict[str, Any]:
    """Concatenate shard tables; a ``candidate_id`` in two shards is an error, not a merge.

    A duplicate key would **merge instead of colliding** and every summed invariant
    would still reconcile — the failure mode is invisible in the counts, so it is
    refused explicitly.
    """
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in table_paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProducerError(f"{path}: cannot read shard table ({exc})") from exc
        if not isinstance(payload, Mapping):
            raise ProducerError(f"{path}: shard table is not an object")
        # ⚠ These tables are written by OTHER processes, so their shape is an input, not an
        # invariant. Three separate faults, each named rather than left to surface as the
        # wrong error somewhere downstream:
        #   (1) a missing/empty `config` coerced to {} makes every shard compare EQUAL, so
        #       the mismatch guard passes and the merge publishes an empty config into
        #       provenance over rows that were scored under real parameters;
        #   (2) a non-Mapping row makes `.get` raise AttributeError, which `main` does not
        #       catch — the merge leg exits with a traceback instead of FATAL/exit 1;
        #   (3) a row missing `candidate_id` read via `.get(..., "")` passes the duplicate
        #       check as "" and then raises KeyError later; a SECOND such row reports
        #       "appears in more than one shard table", naming the wrong fault entirely.
        cfg = payload.get("config")
        missing_keys = (
            [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
            if isinstance(cfg, Mapping)
            else list(REQUIRED_CONFIG_KEYS)
        )
        if not isinstance(cfg, Mapping) or not cfg or missing_keys:
            raise ProducerError(
                f"{path}: shard table's 'config' is missing {missing_keys or 'entirely'}; "
                "merging it would publish an incomplete config over rows scored under real "
                "parameters, and an all-empty set would compare equal across shards"
            )
        configs.append(dict(cfg))
        shard_rows = payload.get("rows")
        if not isinstance(shard_rows, list) or not shard_rows:
            # ⚠ An EMPTY shard is the invisible loss. A dropped or truncated shard merges
            # cleanly, every candidate it should have carried is simply absent from the
            # merged table, and `mine_round` reads each absent candidate as `unavailable`
            # ⇒ spared. The counts still reconcile against each other, so nothing in the
            # report looks wrong — the round just quietly decided less than it claims.
            raise ProducerError(
                f"{path}: shard table carries no rows; a dropped shard would merge cleanly "
                "and its candidates would silently read 'unavailable' ⇒ spared"
            )
        for row in shard_rows:
            if not isinstance(row, Mapping):
                raise ProducerError(f"{path}: a row is {type(row).__name__}, not an object")
            cid = row.get("candidate_id")
            if not isinstance(cid, str) or not cid:
                raise ProducerError(f"{path}: a row carries no usable candidate_id ({cid!r})")
            if not row.get("status"):
                raise ProducerError(f"{path}: row {cid!r} carries no status")
            if cid in seen:
                raise ProducerError(f"candidate_id {cid!r} appears in more than one shard table")
            seen.add(cid)
            rows.append(dict(row))
    if not configs:
        raise ProducerError("no shard tables supplied")
    first = configs[0]
    mismatched = [c for c in configs[1:] if c != first]
    if mismatched:
        raise ProducerError(
            "shard tables were produced under different configs; merging them would "
            f"publish one config over another's rows: {first} vs {mismatched[0]}"
        )
    merged = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "config": first,
        "n_candidates": len(rows),
        "status_counts": dict(sorted(Counter(str(r["status"]) for r in rows).items())),
        "status": {str(r["candidate_id"]): str(r["status"]) for r in rows},
        "rows": rows,
    }
    if out_path is not None:
        _write(out_path, merged)
    return merged


def validate_status_payload(payload: Any, *, label: str) -> dict[str, str]:
    """Re-derive ``status`` from ``rows`` and refuse a table that disagrees with itself."""
    if not isinstance(payload, Mapping):
        raise ProducerError(f"{label}: status table is not an object")
    status = payload.get("status")
    rows = payload.get("rows")
    if not isinstance(status, Mapping) or not isinstance(rows, list):
        raise ProducerError(f"{label}: status table lacks 'status' / 'rows'")
    # Same input-not-invariant reasoning as `merge_status_tables`: these tables are written
    # by other processes. A row that is a string or a number makes `.get` raise
    # AttributeError, which `main` does not catch, so the leg exits with a traceback
    # instead of FATAL / exit 1.
    bad_rows = [i for i, r in enumerate(rows) if not isinstance(r, Mapping)]
    if bad_rows:
        raise ProducerError(f"{label}: rows {bad_rows[:5]} are not objects")
    rederived = {str(r.get("candidate_id", "")): str(r.get("status", "")) for r in rows}
    declared = {str(k): str(v) for k, v in status.items()}
    if rederived != declared:
        only_declared = sorted(set(declared) - set(rederived))[:5]
        only_rows = sorted(set(rederived) - set(declared))[:5]
        # The COMMON disagreement is same-key/different-value drift, and on that path both
        # key-set lists are empty — the operator would read "(declared-only [], rows-only
        # [])", which names no fault at all.
        conflicting = sorted(
            f"{k}: status says {declared[k]!r} but its row says {rederived[k]!r}"
            for k in set(declared) & set(rederived)
            if declared[k] != rederived[k]
        )[:5]
        raise ProducerError(
            f"{label}: 'status' disagrees with 'rows' "
            f"(declared-only {only_declared}, rows-only {only_rows}, "
            f"conflicting {conflicting})"
        )
    known = {STATUS_PASSED, STATUS_FAILED, STATUS_UNAVAILABLE}
    bad = sorted(set(declared.values()) - known)
    if bad:
        raise ProducerError(f"{label}: status table carries unknown statuses {bad}")
    return declared


def load_status_map(table_path: str | Path) -> dict[str, str]:
    try:
        payload = json.loads(Path(table_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProducerError(f"{table_path}: cannot read status table ({exc})") from exc
    return validate_status_payload(payload, label=str(table_path))


# ═════════════════════════════════════════════════════════════════════════════
# D3's freeze: the held-out canonical measurement
# ═════════════════════════════════════════════════════════════════════════════
def heldout_canonical(
    *,
    corpus: str | Path = DEFAULT_CORPUS,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
) -> Any:
    """The held-out canonical carve, joined by content hash — never positionally.

    ``split_assignments``'s ``corpus_record_sha256`` is exactly
    :func:`tbox_finder.ingest.record_hash` over the corresponding ``master_clean``
    row's cell values (verified on every row, not sampled), so the join is
    deterministic and assertable.  A positional join would look identical on a good
    day and be silently wrong after any re-order.

    "Held out" is ``source == "corpus"`` ∧ ``nested_role == "heldout"`` — the
    ADR-0004 leakage-controlled split's designated holdout, which is what D3 names.
    """
    import pandas as pd  # lazy: the CI unit tier has pandas, the predicate module needs nothing

    from tbox_finder.ingest import record_hash

    frame = pd.read_parquet(corpus)
    splits = pd.read_parquet(split_table, columns=["corpus_record_sha256", "source", "nested_role"])
    heldout = splits[(splits["source"] == "corpus") & (splits["nested_role"] == "heldout")]
    wanted = set(heldout["corpus_record_sha256"].dropna())
    if not wanted:
        raise ProducerError(
            f"{split_table}: no source=='corpus' rows with nested_role=='heldout' — the "
            "held-out canonical set is empty, so a freeze over it would measure nothing"
        )
    hashes = [record_hash(row) for row in frame.itertuples(index=False, name=None)]
    frame = frame.assign(_record_sha256=hashes)
    carved = frame[frame["_record_sha256"].isin(wanted)]
    if carved.empty:
        raise ProducerError(
            f"the content-hash join matched 0 of {len(wanted)} held-out records — "
            f"{corpus} and {split_table} do not describe the same corpus"
        )
    return carved


def freeze_report(
    *,
    corpus: str | Path = DEFAULT_CORPUS,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    max_ncca_pairing_nt: int = 4,
) -> dict[str, Any]:
    """Measure the localizer on the held-out canonical set. Pins nothing.

    For each held-out canonical record the curated WUSS ``Structure`` is decomposed
    exactly as a de-novo consensus would be, and three things are reported:

    ``bulge_size_nt``
        the length distribution of the flanked unpaired run that **contains the
        curated ``discriminator``** — i.e. what size range a round must admit for the
        real antiterminator bulge to be in scope at all.
    ``ncca_recovery``
        at each ``ncca_pairing_nt`` from 1 to ``max_ncca_pairing_nt``, the share of
        records whose containing bulge satisfies the derived acceptor motif.  This is
        the detector's sensitivity on known architecture.
    ``n_helices``
        the helix-count distribution, at each ``min_helix_pairs`` — what
        ``min_named_helices`` a round can ask for without excluding canonical loci.

    ⚠ No value from this report is written into any default (§10.3).  It is evidence
    for the round's choice, and the choice stays the round's.
    """
    import pandas as pd

    carved = heldout_canonical(corpus=corpus, split_table=split_table)
    want_cols = ["Sequence", "Structure", "discriminator", "type", "stem1_length"]
    missing = [c for c in want_cols if c not in carved.columns]
    if missing:
        raise ProducerError(f"{corpus}: held-out carve lacks {missing}")

    sizes: Counter = Counter()
    helix_counts: dict[int, Counter] = {k: Counter() for k in (1, 2, 3)}
    ncca_hits: Counter = Counter()
    n_considered = 0
    n_discriminator_in_a_bulge = 0
    n_unparseable = 0
    stem1: list[int] = []

    for rec in carved.itertuples(index=False):
        seq = getattr(rec, "Sequence", None)
        struct = getattr(rec, "Structure", None)
        discrim = getattr(rec, "discriminator", None)
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct):
            n_unparseable += 1
            continue
        try:
            pairs = architecture.pair_table(struct)
        except architecture.ArchitectureError:
            n_unparseable += 1
            continue
        n_considered += 1
        for k in helix_counts:
            helix_counts[k][len(architecture.find_helices(pairs, min_pairs=k))] += 1
        s1 = getattr(rec, "stem1_length", None)
        # `numbers.Real`, not `(int, float)`: on an integer-dtype column `itertuples`
        # yields `np.int64`, which is NOT a Python `int`, so the narrower check silently
        # DROPPED every value and the extent range would read as if the column were empty.
        # The current corpus happens to yield floats (the column carries NaN), which is why
        # the committed report has 8,578 values — a latent fault, not an active one.
        if isinstance(s1, Real) and not isinstance(s1, bool) and not pd.isna(s1) and int(s1) > 0:
            stem1.append(int(s1))

        if not isinstance(discrim, str) or len(discrim) != len(architecture.TRNA_ACCEPTOR_3PRIME):
            continue
        # Anchor by exact subsequence — the coordinates do not index this frame.
        containing_bulge = None
        containing = None
        for bulge in architecture.find_bulges(pairs):
            residues = architecture.degapped_span(seq, bulge.start, bulge.end)
            if discrim.upper() in residues:
                containing_bulge, containing = bulge, residues
                break
        if containing is None or containing_bulge is None:
            continue
        n_discriminator_in_a_bulge += 1
        sizes[len(containing)] += 1
        for k in range(1, int(max_ncca_pairing_nt) + 1):
            # ⚠ Scored on the LOCATED bulge only (`target=`). Scanning every flanked bulge —
            # what an earlier version did, with a range of (1, max(len, 64)) — let an
            # UNRELATED bulge satisfy the motif and increment this numerator, while the
            # denominator counted only records whose discriminator-bearing bulge was found.
            # A size range cannot substitute: two bulges of equal length are equally
            # admissible. The recovery rate is a claim about the antiterminator's own bulge,
            # so it must be measured on that bulge.
            state, _ = architecture.ncca_bulge_status(
                seq,
                pairs,
                bulge_size_range=(len(containing), len(containing)),
                ncca_pairing_nt=k,
                target=containing_bulge,
            )
            if state == architecture.BULGE_DETECTED:
                ncca_hits[k] += 1

    def _share(n: int) -> float | None:
        return round(n / n_discriminator_in_a_bulge, 6) if n_discriminator_in_a_bulge else None

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "adr": ADR,
        "carve": {
            "definition": "source == 'corpus' AND nested_role == 'heldout'",
            "join": "ingest.record_hash(master_clean row) == split.corpus_record_sha256",
            "n_records": int(len(carved)),
            "n_structure_parseable": n_considered,
            "n_unparseable": n_unparseable,
            "n_with_discriminator_in_a_flanked_bulge": n_discriminator_in_a_bulge,
        },
        "bulge_size_nt": {
            "counts": {str(k): v for k, v in sorted(sizes.items())},
            "modal_size": max(sizes, key=lambda k: (sizes[k], -k)) if sizes else None,
            "note": (
                "length of the flanked unpaired run CONTAINING the curated discriminator, "
                "in the record's own residues"
            ),
        },
        "ncca_recovery": {
            str(k): {"n": ncca_hits.get(k, 0), "share": _share(ncca_hits.get(k, 0))}
            for k in range(1, int(max_ncca_pairing_nt) + 1)
        },
        "n_helices": {
            f"min_helix_pairs={k}": {str(n): c for n, c in sorted(counter.items())}
            for k, counter in helix_counts.items()
        },
        "stem_i_extent_nt": {
            "n": len(stem1),
            "min": min(stem1) if stem1 else None,
            "max": max(stem1) if stem1 else None,
            "note": "D6's threshold is supplied per round (ADR-0006 A4); NO value is frozen here",
        },
        "acceptor_motif": architecture.acceptor_pairing_motif(),
        "pins_nothing": True,
        "disclosure": (
            "measured on CURATED structure, which is CM-derived: this is the localizer's "
            "sensitivity on known architecture, NOT the rate at which a de-novo consensus "
            "resolves one. The second number needs the full-corpus (a)/(b) supply run and is "
            "not claimed here."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Supply derivation — the validator half of RELAXED_ARCH_SUPPLY_AVAILABLE
# ═════════════════════════════════════════════════════════════════════════════
#: Git-tracked evidence only — the same constraint ``derive_synteny_supply_available``
#: documents.  A clause reading a DVC- or scratch-shaped path would be ``False`` in CI
#: and in every fresh clone, making the derivation disagree with the constant in
#: exactly the environments that must agree.
SUPPLY_CLAUSES: tuple[tuple[str, str], ...] = (
    ("predicate_module_present", "src/tbox_finder/mining/architecture.py"),
    ("producer_module_present", "src/tbox_finder/mining/architecture_producer.py"),
    ("freeze_report_present", FREEZE_REPORT),
)


def _json_or_none(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def derive_relaxed_arch_supply_available(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Can *this checkout* evidence a criterion-(b) supply?  Fail-closed, clause by clause.

    Mirrors :func:`tbox_finder.mining.mine_round.derive_msa_supply_available` and
    :func:`tbox_finder.mining.synteny_producer.derive_synteny_supply_available`.

    The load-bearing clause is the **last** one.  A4 disclosed that (b) now reads the
    same MSA (a) does, so a (b) supply is not producible until the per-candidate
    consensus corpus exists — and the *predicate* shipping is not the same claim as the
    *supply* existing.  ``msa_supply_backs_it`` therefore delegates to the (a)
    derivation: a checkout that cannot evidence the MSA supply cannot evidence (b)
    either, by construction rather than by a second hand-written probe that could drift
    from it.

    ⚠ Clause names are listed exhaustively from :data:`SUPPLY_CLAUSES` rather than
    iterated off whatever keys happen to be present: a clause set read from its own
    evidence is vacuously satisfied exactly when the evidence is missing.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    clauses = {name: (root / rel).is_file() for name, rel in SUPPLY_CLAUSES}

    # ⚠ There was a ``producer_entry_points_present`` clause here and it was **vacuous**:
    # it imported the module that was already executing, so neither the import nor the
    # ``hasattr`` could ever fail, and ``all(clauses.values())`` was being handed a
    # hardcoded True. That is the precise failure mode this function's own docstring warns
    # about — and the parametrized clause test never caught it, because it iterates
    # SUPPLY_CLAUSES and this clause was not in that tuple.
    #
    # It is replaced by the clause that ``derive_msa_supply_available`` uses for the same
    # job and which IS breakable: measure that the round actually **stamps** a produced (b)
    # status onto the candidate, by calling the wiring rather than by reading a signature.
    # A producer that ships but whose output the round drops on the floor is the silent
    # no-op the whole gate exists to refuse.
    probe_id = "__relaxed_arch_supply_probe__"
    try:
        from tbox_finder.mining.mine_round import candidate_evidence

        stamped: Any = candidate_evidence(
            probe_id, None, None, {probe_id: STATUS_PASSED}
        ).relaxed_architecture
    except Exception as exc:  # noqa: BLE001 - a broken round is a FAILED clause, not a crash
        # Deliberately broader than ImportError: a module-level failure anywhere in
        # `mine_round` or its transitive imports (RuntimeError, OSError, AttributeError)
        # would otherwise propagate out of a function documented as fail-closed on every
        # clause, aborting `derive-supply` with a traceback instead of the FATAL / exit 1
        # contract every other leg follows. `derive_msa_supply_available` already treats a
        # broken producer import this way, for the same reason.
        stamped = f"probe failed: {exc!r}"
    clauses["producer_status_wired"] = stamped == STATUS_PASSED
    wiring_reason = (
        ""
        if clauses["producer_status_wired"]
        else (
            f"a produced {STATUS_PASSED!r} status reached the candidate as {stamped!r} — "
            "the producer's output is not composed into the round"
        )
    )

    # The freeze must have MEASURED something, not merely exist: a report whose carve
    # is empty certifies the predicate against nothing.
    freeze = _json_or_none(root / FREEZE_REPORT)
    carve = freeze.get("carve") if isinstance(freeze, Mapping) else None
    n_measured = (
        carve.get("n_with_discriminator_in_a_flanked_bulge") if isinstance(carve, Mapping) else None
    )
    clauses["freeze_measured_a_nonempty_carve"] = bool(
        isinstance(n_measured, int) and not isinstance(n_measured, bool) and n_measured > 0
    )

    # The freeze is EVIDENCE, so the record of how it was produced is part of the claim.
    # `build_provenance` permits an empty `inputs` and a null env lock, so a report written
    # by a run that hashed nothing would otherwise satisfy every other clause here.
    provenance = freeze.get("provenance") if isinstance(freeze, Mapping) else None
    provenance = provenance if isinstance(provenance, Mapping) else {}
    extra = provenance.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}

    # ⚠ The corpus and the split table are DVC-tracked and are NOT materialized in a
    # worktree, so the freeze reads them by absolute path from the main checkout and
    # `_split_inputs` records them as sha256'd ``external_inputs`` rather than as
    # repo-relative ``inputs``. Either form satisfies the contract — the hashed form is
    # strictly MORE traceable, since a reader can verify the bytes without knowing where
    # they sat. What must never pass is a report that names NEITHER.
    # ⚠ `len()` accepts only sized objects, and the freeze report is an on-disk artifact
    # written by another process — its shape is an INPUT, not an invariant, the same
    # reasoning `merge_status_tables` states. A scalar `inputs` would raise TypeError out
    # of a function documented as fail-closed on every clause, and `main` does not catch
    # TypeError, so `derive-supply` would exit with a traceback instead of FATAL / exit 1.
    def _n_named(value: Any) -> int:
        return len(value) if isinstance(value, (Mapping, list, tuple)) else 0

    named_inputs = _n_named(provenance.get("inputs")) + _n_named(extra.get("external_inputs"))
    clauses["freeze_provenance_names_its_inputs"] = bool(
        named_inputs > 0 and provenance.get("env_lock_hash") and provenance.get("git_sha")
    )

    # The delegation needs the SAME fail-closed handling as the probe above, and for the
    # same reason: `derive_msa_supply_available` runs its own unwrapped `candidate_evidence`
    # probe, so a broken round takes THIS function down through the delegation even when the
    # local probe is guarded. Found by the test for the local guard failing here instead —
    # one of two call sites again.
    try:
        from tbox_finder.mining.mine_round import derive_msa_supply_available

        msa = derive_msa_supply_available(repo_root=root)
    except Exception as exc:  # noqa: BLE001 - a broken (a) derivation is a FAILED clause
        msa = {"available": False, "reasons": [f"(a) derivation raised: {exc!r}"]}
    clauses["msa_supply_backs_it"] = bool(msa.get("available"))

    reasons = [f"{name} is missing or false" for name, ok in sorted(clauses.items()) if not ok]
    if wiring_reason:
        reasons.append(f"producer_status_wired: {wiring_reason}")
    if not clauses["msa_supply_backs_it"]:
        reasons.append(
            "msa_supply_backs_it: (b) reads the same consensus (a) does (ADR-0006 A4), so "
            f"it inherits (a)'s supply — {'; '.join(msa.get('reasons') or ['unavailable'])}"
        )
    return {
        "available": all(clauses.values()),
        "clauses": dict(sorted(clauses.items())),
        # Reuse the CLAUSE's verdict rather than re-testing: `isinstance(True, int)` is
        # True, so a report carrying `true` here would have made the clause read False
        # while this field reported `true` — two fields disagreeing inside one payload.
        "n_heldout_canonical_measured": (
            n_measured if clauses["freeze_measured_a_nonempty_carve"] else 0
        ),
        "reasons": reasons,
        "repo_root": str(root),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def _split_inputs(inputs: Sequence[str | Path]) -> tuple[list[str], list[dict[str, str]]]:
    """In-repo inputs (hashed by ``build_provenance``) vs external ones (hashed here).

    A committed provenance record must never carry an absolute local path — it leaks a
    username and names a machine-specific location no reader can resolve.
    """
    from tbox_finder.provenance import sha256_file

    inside: list[str] = []
    external: list[dict[str, str]] = []
    for item in inputs:
        resolved = Path(item).resolve()
        try:
            inside.append(str(resolved.relative_to(REPO_ROOT)))
        except ValueError:
            external.append({"name": resolved.name, "sha256": sha256_file(resolved)})
    return inside, external


def _provenance(entry: str, inputs: Sequence[str | Path], extra: Mapping[str, Any]) -> dict:
    """CLAUDE.md §11 provenance.  ``inputs`` only — never the report being written.

    ``build_provenance`` hashes whatever it is given, and a not-yet-written output
    raises at the END of the run that would have produced it.
    """
    from tbox_finder.provenance import build_provenance

    inside, external = _split_inputs([i for i in inputs if Path(i).exists()])
    payload = dict(extra)
    if external:
        payload["external_inputs"] = external
    return build_provenance(
        rule=f"architecture_producer::{entry}",
        script="src/tbox_finder/mining/architecture_producer.py",
        inputs=inside,
        adr=ADR,
        env_lock=ENV_LOCK,
        extra=payload,
    )


def _write(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ProducerError(f"{path}: cannot write ({exc})") from exc
    return target


def _config_from(args: argparse.Namespace) -> ArchitectureRunConfig:
    return ArchitectureRunConfig(
        stem_i_nt_threshold=args.stem_i_nt_threshold,
        min_named_helices=args.min_named_helices,
        min_helix_pairs=args.min_helix_pairs,
        bulge_min_nt=args.bulge_min_nt,
        bulge_max_nt=args.bulge_max_nt,
        ncca_pairing_nt=args.ncca_pairing_nt,
        allow_wobble=bool(args.allow_wobble),
        min_sequences=args.min_sequences,
    )


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    """Every rule parameter REQUIRED with no default (ADR-0006 A4; §10.3)."""
    parser.add_argument(
        "--stem-i-nt-threshold",
        type=int,
        required=True,
        help=(
            "REQUIRED, no default: D6's short-Stem-I nt threshold, supplied per round and "
            "recorded in provenance (ADR-0006 A4; D17's P6 freeze is NOT consumed)"
        ),
    )
    parser.add_argument(
        "--min-named-helices",
        type=int,
        required=True,
        help="REQUIRED, no default: how many distinct helices count as the expected set (D3)",
    )
    parser.add_argument(
        "--min-helix-pairs",
        type=int,
        required=True,
        help="REQUIRED, no default: minimum stacked pairs for a helix to count",
    )
    parser.add_argument("--bulge-min-nt", type=int, required=True, help="REQUIRED, no default")
    parser.add_argument("--bulge-max-nt", type=int, required=True, help="REQUIRED, no default")
    parser.add_argument(
        "--ncca-pairing-nt",
        type=int,
        required=True,
        help="REQUIRED, no default: how many acceptor-end contacts must be evidenced",
    )
    parser.add_argument("--allow-wobble", action="store_true", help="admit G·U at the interface")
    parser.add_argument(
        "--min-sequences",
        type=int,
        default=MIN_REAL_HOMOLOG_N,
        help=f"ADR-0006 A2 Pin 2 alignment-depth floor (default {MIN_REAL_HOMOLOG_N})",
    )
    parser.add_argument("--manifest", default=DEFAULT_FP_MANIFEST)
    parser.add_argument("--msa-dir", default=DEFAULT_MSA_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tbox_finder.mining.architecture_producer",
        description="ADR-0006 D3 criterion-(b) producer + its held-out-canonical freeze.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-shard", help="evaluate a shard (or the whole manifest)")
    _add_config_flags(run)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--n-shards", type=int, default=1)
    run.add_argument("--out", required=True)

    merge = sub.add_parser("merge", help="concatenate shard tables into the status table")
    merge.add_argument("--table", action="append", required=True)
    merge.add_argument("--out", required=True)

    freeze = sub.add_parser("freeze", help="D3's held-out-canonical measurement (pins nothing)")
    freeze.add_argument("--corpus", default=DEFAULT_CORPUS)
    freeze.add_argument("--split-table", default=DEFAULT_SPLIT_TABLE)
    freeze.add_argument("--out", default=FREEZE_REPORT)

    sub.add_parser("derive-supply", help="print the supply derivation for this checkout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run-shard":
            config = _config_from(args)
            candidates = shard_candidates(
                read_manifest(args.manifest), args.shard_index, args.n_shards
            )
            table = build_status_table(
                run_shard(candidates, msa_dir=args.msa_dir, config=config), config=config
            )
            table["provenance"] = _provenance(
                "run-shard",
                [args.manifest],
                {"shard_index": args.shard_index, "n_shards": args.n_shards, **config.as_dict()},
            )
            _write(args.out, table)
            print(f"{args.out}: {table['status_counts']}")
            return 0

        if args.command == "merge":
            merged = merge_status_tables(args.table)
            validate_status_payload(merged, label="merged")
            merged["provenance"] = _provenance("merge", args.table, merged["config"])
            _write(args.out, merged)
            print(f"{args.out}: {merged['n_candidates']} candidates {merged['status_counts']}")
            return 0

        if args.command == "freeze":
            report = freeze_report(corpus=args.corpus, split_table=args.split_table)
            report["provenance"] = _provenance(
                "freeze", [args.corpus, args.split_table], {"pins_nothing": True}
            )
            _write(args.out, report)
            print(f"{args.out}: {report['carve']}")
            return 0

        if args.command == "derive-supply":
            derivation = derive_relaxed_arch_supply_available()
            print(json.dumps(derivation, indent=2, sort_keys=True))
            return 0
    except (ProducerError, architecture.ArchitectureError, OSError, ValueError, KeyError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
