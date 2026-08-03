"""P3-06 — the Stage-2 RiNALMo-giga LoRA fine-tune entrypoint (PRD §10.2/§10.3/§11).

``python -m tbox_finder.stage2.train`` under ``torchrun``, in the ``ml-rna`` env
(ADR-0002 A4). It composes ``conf/train/stage2.yaml``, selects the training population,
builds the P3-03 model + the P3-04/P3-05 objective, runs DDP×8, and writes a
self-validating JSON report plus a LoRA-adapter checkpoint. **A run that fails its own
gate raises**, so ``rc != 0`` and the §9.3 artifact check sees no ``DONE`` marker.

Three tiers, mirroring ``train_stage1``:

* **pure** — eligibility, config validation, the LR schedule, the report clauses and the
  validator. No ``torch``, no ``pandas``, no ``hydra``; it runs and *blocks* in bare CI.
* **torch** — the dataset/collator, the model/optimizer wiring, the loop.
* **hydra** — config composition, exercised by the P3-06 unit tier.

── THE TRAINING POPULATION (user decision, AskUserQuestion 2026-08-03) ──────────────────
``imp.md`` hands the train-eligibility policy to this step. Two things were decided and
are encoded in :func:`row_eligibility`:

1. **The fold is the ADR-0004 D5 nested training fold intersected with the scheme-A
   (random, genus-stratified) *train* rung, minus ``calib``.** Not the full D5 fold.
   Stage-2 must sit inside D5 because PRD §12 grades leave-one-order-out recall on the
   **two-stage** system — a Stage-2 that saw a held-out order contaminates the headline
   just as a Stage-1 that did would. The extra ``fold_random == "train"`` restriction is
   what keeps the scheme-A ``val``/``test`` rungs untrained, so P3-07 fits its temperature
   on the disjoint ``calib`` split and P3-10 grades GATE-2's in-distribution ECE on **this
   checkpoint** rather than on an ADR-0004 A6-style eval twin. For GATE-4 (segmentation
   quality) a twin was acceptable; for a *calibration* gate it is much weaker, because
   calibration is a property of the model that ships. Measured cost of the restriction:
   8 records across 5 tiny orders and 3 rare phyla (Arthropoda 4, Spirochaetes 2,
   Streptophyta 2), i.e. 8,084 → 5,199 corpus rows.
2. **The 5,007 parentless decoys are admitted.** They carry ``nested_train = NULL``
   because they have no corpus parent — ``structured_rna`` (2,999), ``gc_background``
   (2,000) and ``leader_decoy`` (8) are built, not carved, so there is no order to hold
   out and the ADR-0004 A4/A5 ``require_parent_nested_train`` rule (which governs the
   *mined genomic-window* substrate) has nothing to key on. Refusing them fail-closed
   would collapse the negative class to the 404 parent-inheriting dinuc-shuffled rows —
   a ~12:1 imbalance in which the binary head only ever sees the null PRD §12 itself
   calls the optimistic lower bound.

Everything else is refused, **fail-closed and by name**: ``nested_train is False`` (the
LOO/clade holdout, whatever its ``nested_role``), a row on another scheme-A rung, a
``calib`` row, and any row whose fold basis this module does not recognise. The refusal
counts are recorded and two of them are *must-fire* gate clauses — a rule that refuses
nothing is a rule that is not running ([[namespace-mismatch-invisible-noop]]).

── DECISIONS P3-03/P3-04 HANDED HERE, RESOLVED ──────────────────────────────────────────
* **``boundary_use_crf`` → False, and ``True`` is refused at construction.**
  :func:`~tbox_finder.stage2.losses.multitask_loss` raises on any non-``None``
  ``boundary_crf`` because computing the plain cross-entropy instead would leave the CRF's
  transition matrix in the optimiser with no gradient reaching it. Refusing here means the
  refusal lands on the login node, before the queue wait, rather than after it.
* **Sequence pooling stays masked-mean** (:meth:`Stage2Model.pool`). The checkpoint carries
  no pooler and ``load_rinalmo_backbone`` passes ``add_pooling_layer=False`` precisely so
  PEFT is never handed a randomly-initialised module to adapt. Recorded, not a knob.

── WHAT IS SWEPT, AND WHAT IS NOT ───────────────────────────────────────────────────────
PRD §11 names "γ, LoRA rank, learning rate, window/stride, RC-combination, Stage-2
aux-loss weight" as ``--multirun`` axes. Of those, **``loss.aux_weight`` and ``optim.lr``
are swept here**; the focal γs are exposed (``loss.*_gamma``) but held at their P3-04
defaults; window/stride and RC-combination are Stage-1 axes with no Stage-2 analogue (a
Stage-2 input is a whole locus, not a tiled window). **LoRA rank is deliberately NOT
swept**: PRD §10.3 pins ``LoraConfig(r=16, α=32, dropout=0.05, target_modules="all-linear")``
and ``lora_harness`` freezes those four as module constants with a drift-guard test
asserting config == code, so varying ``r`` would mean editing that frozen contract — an
ADR-0002 question, not a sweep value. Named here rather than silently dropped.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import labels, masking
from tbox_finder.stage2 import heads as H
from tbox_finder.stage2 import losses as L
from tbox_finder.stage2 import tokenizer as tok
from tbox_finder.stage2.dataset import FOLD_BASIS_DECOY_RANDOM, SOURCE_DECOY
from tbox_finder.train import ddp

__all__ = [
    "ADMITTED",
    "CONFIG_PATH",
    "DEFAULT_CKPT_DIR",
    "DEFAULT_DATASET",
    "DEFAULT_REPORT",
    "ENTRYPOINT",
    "ENV_LOCK",
    "REFUSAL_REASONS",
    "SCHEMA_VERSION",
    "STEP",
    "EpochShuffleSampler",
    "Stage2TrainConfig",
    "build_report",
    "derive_clauses",
    "diagnostics",
    "lr_scale",
    "row_eligibility",
    "select_rows",
    "train_stage2",
    "validate_report",
]

# --------------------------------------------------------------------------------------
# Provenance constants (CLAUDE.md §11). Single-sourced here; conf/ echoes them.
# --------------------------------------------------------------------------------------
SCHEMA_VERSION = "1"
STEP = "P3-06"
ENTRYPOINT = "tbox_finder.stage2.train"
#: ADR-0002 A4 — Stage-2 is the RNA closure, NOT ml-dna's.
ENV_LOCK = "envs/ml-rna.conda-lock.yml"
CONFIG_PATH = "conf/train/stage2.yaml"

DEFAULT_DATASET = "data/processed/stage2_dataset.parquet"
DEFAULT_REPORT = "reports/p3/train_stage2.json"
DEFAULT_CKPT_DIR = "data/processed/checkpoints/stage2_rinalmo"

#: File names inside a checkpoint directory. The adapter is PEFT's own layout; the heads
#: are a plain state-dict because they live OUTSIDE the PEFT wrapper by construction
#: (``build_stage2_model`` measures that and reports ``heads_outside_peft_wrapper``).
ADAPTER_SUBDIR = "lora_adapter"
HEADS_STATE_NAME = "stage2_heads.pt"

# --------------------------------------------------------------------------------------
# Eligibility — the pure, torch-free heart of the step
# --------------------------------------------------------------------------------------
#: The scheme-A (random, genus-stratified) rung a row must sit on to train.
TRAIN_RUNG = "train"
#: The scheme-A rung the in-training monitor evaluates on. Disjoint from TRAIN_RUNG by
#: construction, and asserted as an identity (not a count) by a gate clause.
VAL_RUNG = "val"

ADMITTED = "admitted"

#: Refused because the row is in the P3-02 calibration split. Checked FIRST, so a calib
#: row can never be counted as admitted by some later branch.
REFUSE_CALIB = "calibration_split"
#: Refused because the row sits on another scheme-A rung (val/test) — the in-distribution
#: rungs GATE-2 is graded on.
REFUSE_OTHER_RUNG = "other_scheme_a_rung"
#: Refused because ``nested_train`` is False: the row is in the ADR-0004 D5 complement,
#: i.e. the leave-one-order-out / phylum / literature-anchor holdout, a clade-crossing
#: exclusion, or a dropped record. The `nested_role` breakdown is recorded beside the count.
REFUSE_LOO_HOLDOUT = "d5_holdout_or_excluded"
#: Refused because ``nested_train`` is missing and the row is not a recognised
#: pool-randomised decoy — fail-closed on anything this module cannot place in a fold.
REFUSE_UNPLACEABLE = "unplaceable_missing_nested_train"

REFUSAL_REASONS: tuple[str, ...] = (
    REFUSE_CALIB,
    REFUSE_OTHER_RUNG,
    REFUSE_LOO_HOLDOUT,
    REFUSE_UNPLACEABLE,
)

#: Admitted rows, split by how they earned it. Two disjoint routes, counted separately so
#: the report can show that BOTH fired — an eligibility rule that silently admitted only
#: corpus rows would still look healthy on a single total.
ADMIT_NESTED_TRAIN = "nested_train"
ADMIT_PARENTLESS_DECOY = "parentless_decoy"
ADMIT_ROUTES: tuple[str, ...] = (ADMIT_NESTED_TRAIN, ADMIT_PARENTLESS_DECOY)


def _bool_or_none(value: Any) -> bool | None:
    """``nested_train`` as a tri-state, with pandas' several spellings of missing.

    ``masking.is_missing`` rather than ``value or None``: under the ``ml-rna`` env's
    pandas the string form of a missing cell is ``"nan"``, which is truthy, and that exact
    difference once deleted 60% of a training mix while every clause stayed green
    ([[pandas-3-nan-truthy-in-training-env]]).
    """
    if masking.is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    # numpy.bool_ / 0 / 1 / "True" all arrive here from parquet round-trips.
    text = masking.row_text(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    if not text or text in H.NULL_TOKENS:
        # An empty cell, or one whose STRING form is a null token, carries no fold
        # assignment — so it is missing, and routes to the fail-closed refusal rather than
        # raising. The literal `"nan"` matters here and is not hypothetical: it is what a
        # missing cell stringifies to under the ml-rna env's pandas, which is exactly how a
        # missingness check that only knew about `None` once deleted 60% of a training mix
        # ([[pandas-3-nan-truthy-in-training-env]]). `H.NULL_TOKENS` is reused rather than
        # respelled so the two null vocabularies cannot drift apart.
        return None
    raise ValueError(f"nested_train={value!r} is neither missing nor boolean")


def row_eligibility(row: Mapping[str, Any], *, rung: str, admit_parentless_decoys: bool) -> str:
    """:data:`ADMITTED`, or the named reason this row is refused.

    Fail-closed: every branch that is not an explicit admit returns a refusal, and an
    unrecognised fold basis is a refusal rather than a fallthrough.

    Raises:
        ValueError: if a row claims the pool-randomised decoy fold basis but is not a
            decoy with a missing ``nested_train``. That combination cannot occur in the
            committed dataset, and if it ever did, the parentless-decoy admit branch would
            be swallowing a corpus record whose order *is* known — the one way this
            function could leak a held-out clade into training.
    """
    fold_basis = masking.row_text(row.get("fold_basis"))
    nested = _bool_or_none(row.get("nested_train"))
    source = masking.row_text(row.get("source"))

    if fold_basis == FOLD_BASIS_DECOY_RANDOM and not (nested is None and source == SOURCE_DECOY):
        raise ValueError(
            f"row {row.get('row_id')!r} has fold_basis={FOLD_BASIS_DECOY_RANDOM!r} but "
            f"nested_train={row.get('nested_train')!r} / source={source!r}. The "
            "parentless-decoy admit branch exists for rows with NO corpus parent; a row "
            "with a fold assignment must be placed by that assignment, not by this branch."
        )

    if _bool_or_none(row.get("calib")) is True:
        return REFUSE_CALIB
    if masking.row_text(row.get("fold_random")) != rung:
        return REFUSE_OTHER_RUNG
    if nested is True:
        return ADMITTED
    if nested is False:
        return REFUSE_LOO_HOLDOUT
    # nested_train is missing from here on.
    if admit_parentless_decoys and fold_basis == FOLD_BASIS_DECOY_RANDOM:
        return ADMITTED
    return REFUSE_UNPLACEABLE


def _admit_route(row: Mapping[str, Any]) -> str:
    return (
        ADMIT_NESTED_TRAIN
        if _bool_or_none(row.get("nested_train")) is True
        else ADMIT_PARENTLESS_DECOY
    )


def select_rows(
    rows: Sequence[Mapping[str, Any]], *, rung: str, admit_parentless_decoys: bool = True
) -> tuple[list[int], dict[str, Any]]:
    """``(admitted row indices, census)`` over ``rows`` for one scheme-A rung.

    The census is the evidence every eligibility gate clause is re-derived from — it
    carries the per-reason refusal counts, the per-route admit counts, the ``nested_role``
    breakdown of the D5 refusals, and the admitted positive/negative split. Nothing in it
    restates the *request*; a clause that read the requested config back would be
    vacuously true exactly when the evidence was missing
    ([[clauses-must-guard-emptiness]]).
    """
    admitted: list[int] = []
    refused = dict.fromkeys(REFUSAL_REASONS, 0)
    by_route = dict.fromkeys(ADMIT_ROUTES, 0)
    holdout_roles: dict[str, int] = {}
    n_positive = 0

    for i, row in enumerate(rows):
        verdict = row_eligibility(row, rung=rung, admit_parentless_decoys=admit_parentless_decoys)
        if verdict == ADMITTED:
            admitted.append(i)
            by_route[_admit_route(row)] += 1
            if _bool_or_none(row.get("is_tbox")) is True:
                n_positive += 1
            continue
        refused[verdict] += 1
        if verdict == REFUSE_LOO_HOLDOUT:
            role = masking.row_text(row.get("nested_role")) or "unlabelled"
            holdout_roles[role] = holdout_roles.get(role, 0) + 1

    census = {
        "rung": rung,
        "admit_parentless_decoys": bool(admit_parentless_decoys),
        "n_rows_scanned": len(rows),
        "n_admitted": len(admitted),
        "admitted_by_route": by_route,
        "refused": refused,
        "d5_refusal_roles": dict(sorted(holdout_roles.items())),
        "n_admitted_positive": n_positive,
        "n_admitted_negative": len(admitted) - n_positive,
    }
    return admitted, census


# --------------------------------------------------------------------------------------
# Learning-rate schedule — pure, so the bare tier can grade its shape
# --------------------------------------------------------------------------------------
def lr_scale(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup → cosine decay to zero, as a multiplier on the base LR.

    Written here rather than imported so it is exercised in the bare CI tier: an LR
    schedule that silently returns a constant is invisible in a loss curve for the first
    few hundred steps and expensive to discover on a GPU node.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError(f"warmup_steps must be in [0, {total_steps}], got {warmup_steps}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step < warmup_steps:
        # step 0 must not be a zero LR (that wastes the step); ramp over [1, warmup].
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class EpochShuffleSampler:
    """A seeded, epoch-dependent permutation of ``n`` indices; yields bare ints.

    Deliberately minimal — Stage-2 has no oversampling curriculum (its records are whole
    loci, not windows, and PRD §11's rare-class oversampling is a Stage-1 device). It
    exposes ``set_epoch``/``__len__``/``__iter__`` so
    :class:`~tbox_finder.train.ddp.ShardedSampler` can shard it, which is the whole reason
    it is a class rather than a call to ``random.shuffle``.
    """

    def __init__(self, n: int, *, seed: int) -> None:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self._n = int(n)
        self._seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[int]:
        import random

        order = list(range(self._n))
        # A fresh Random seeded on (seed, epoch): reaching for the global RNG here would
        # couple the shuffle to every other draw in the process, and reusing one Random
        # across epochs would make the permutation depend on how many times __iter__ had
        # been called rather than on the epoch it claims to key on.
        # A string key, not a tuple: `random.Random` seeds a str through SHA-512, so the
        # permutation is identical on every rank and every machine. `hash((seed, epoch))`
        # would be PYTHONHASHSEED-dependent for anything but ints, which is precisely the
        # class of "reproducible until it isn't" §8.3 exists to keep out.
        random.Random(f"{ENTRYPOINT}:{self._seed}:{self._epoch}").shuffle(order)
        return iter(order)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage2TrainConfig:
    """Every field is a literal key in ``conf/train/stage2.yaml`` (or ``conf/optim``/``loss``).

    Hydra composes in struct mode, so a CLI override against a key the YAML lacks is a hard
    ``ConfigCompositionException`` — which is how SLURM job 669 was lost after a 14 h queue
    wait ([[hydra-struct-mode-override-needs-yaml-key]]). ``tests/unit/test_sbatch_overrides.py``
    composes every override this repo's sbatch files write, so the guard is CI-visible.
    """

    seed: int = 42
    pythonhashseed: int = 0

    # Data
    dataset_parquet: str = DEFAULT_DATASET
    train_rung: str = TRAIN_RUNG
    val_rung: str = VAL_RUNG
    admit_parentless_decoys: bool = True
    max_records: int | None = None
    eval_max_records: int | None = None
    num_workers: int = 2

    # Optimisation
    epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1

    # Model
    revision: str | None = None
    dtype: str | None = None
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    dropout: float = 0.1
    boundary_use_crf: bool = False
    structure_head: bool = False
    pairing_proj_dim: int = 64

    # Objective (the /loss group is its ONE home — do not restate any of its keys here)
    loss: L.Stage2LossConfig = field(default_factory=L.Stage2LossConfig)

    # Eval / IO
    eval_val: bool = True
    save_checkpoint: bool = True
    checkpoint_dir: str = DEFAULT_CKPT_DIR
    report_path: str = DEFAULT_REPORT
    log_every: int = 20
    device: str | None = None

    # Tracking (the /tracking group)
    wandb_mode: str = "offline"
    wandb_project: str = "tbox-finder"
    wandb_entity: str | None = None
    wandb_dir: str = "wandb"

    def __post_init__(self) -> None:
        for name, minimum in (
            ("epochs", 1),
            ("batch_size", 1),
            ("eval_batch_size", 1),
            ("gradient_accumulation_steps", 1),
            ("pairing_proj_dim", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
        for name in ("lr", "weight_decay", "grad_clip", "dropout", "warmup_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not 0.0 <= float(self.warmup_ratio) < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.num_workers < 0 or isinstance(self.num_workers, bool):
            raise ValueError(f"num_workers must be an int >= 0, got {self.num_workers!r}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.train_rung == self.val_rung:
            raise ValueError(
                f"train_rung and val_rung are both {self.train_rung!r}; the monitor would "
                "score the model on its own training rows and report them as held out."
            )
        if not isinstance(self.loss, L.Stage2LossConfig):
            raise TypeError(f"loss must be a Stage2LossConfig, got {type(self.loss).__name__}")
        if bool(self.structure_head) != bool(self.loss.structure_enabled):
            raise ValueError(
                f"structure_head={self.structure_head} but loss.structure_enabled="
                f"{self.loss.structure_enabled}. A head with no term sits in the optimiser "
                "receiving no gradient; a term with no head has no logits. "
                "multitask_loss refuses the pairing too — this refuses it on the login "
                "node, before the queue wait."
            )
        if self.boundary_use_crf:
            raise ValueError(
                "boundary_use_crf=True is refused (the P3-03/P3-04 hand-off, decided at "
                "P3-06). losses.multitask_loss raises on any boundary_crf because the "
                "alternative — computing the plain cross-entropy and calling it a CRF — "
                "leaves the transition matrix in the optimiser with no gradient reaching "
                "it. Implementing the CRF term is a step of its own, not a flag."
            )
        for name in ("max_records", "eval_max_records"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be None or an int >= 1, got {value!r}")


def diagnostics(cfg: Stage2TrainConfig) -> dict[str, Any]:
    """The run's configuration as recorded — what was asked for, labelled as such."""
    payload = {k: v for k, v in asdict(cfg).items() if k != "loss"}
    payload["loss"] = cfg.loss.diagnostics()
    payload["pooling"] = "masked_mean"  # P3-03 hand-off, resolved here (see the docstring)
    payload["boundary_use_crf"] = False
    return payload


# --------------------------------------------------------------------------------------
# Report + gate
# --------------------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _env_lock_sha256(path: str | Path = ENV_LOCK) -> str | None:
    import hashlib

    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, bool, int, type(None))):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return str(obj)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pos_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """Re-derive every gate clause from the report's own recorded evidence.

    Nothing here reads back a *requested* setting. ``all(clauses)`` catches a clause
    flipped false but never one fabricated true, so each clause is recomputed from the
    numbers the run measured ([[gate-clauses-need-re-derivation]]), and the two population
    clauses are **must-fire**: they require the refusal counters to be non-zero, because a
    filter that refuses nothing is indistinguishable from a filter that never ran.
    """
    data = report.get("data") or {}
    train_census = data.get("train_census") or {}
    val_census = data.get("val_census") or {}
    refused = train_census.get("refused") or {}
    routes = train_census.get("admitted_by_route") or {}
    steps = report.get("steps") or {}
    wrap = report.get("wrap") or {}
    applied = wrap.get("applied_lora") or {}
    objective = report.get("objective") or {}
    ckpt = report.get("checkpoint") or {}
    prov = report.get("provenance") or {}
    losses = report.get("losses") or {}

    n_admitted = train_census.get("n_admitted")
    n_scanned = train_census.get("n_rows_scanned")

    clauses: dict[str, bool] = {}

    # -- population ------------------------------------------------------------------ #
    clauses["no_d5_holdout_trained"] = (
        _pos_int(refused.get(REFUSE_LOO_HOLDOUT))
        and _pos_int(n_admitted)
        and n_admitted + sum(v for v in refused.values() if isinstance(v, int)) == n_scanned
    )
    clauses["calib_refused"] = _pos_int(refused.get(REFUSE_CALIB))
    clauses["both_admit_routes_fired"] = (
        all(_pos_int(routes.get(name)) for name in ADMIT_ROUTES)
        and sum(v for v in routes.values() if isinstance(v, int)) == n_admitted
    )
    # Completeness, beside the correctness clauses: a `max_records` truncation leaves every
    # rule-shaped clause green over a handful of rows ([[cost-knobs-can-certify]]).
    clauses["full_population"] = (
        _pos_int(data.get("n_train_rows")) and data.get("n_train_rows") == n_admitted
    )
    # Identity, not counts: equal-sized arms with a fold-sense inversion satisfy a count
    # check ([[symmetric-count-fixture-blind-to-inversion]]).
    clauses["val_disjoint_from_train"] = (
        data.get("n_train_val_row_id_overlap") == 0 and _pos_int(val_census.get("n_admitted"))
        if report.get("eval_val")
        else data.get("n_train_val_row_id_overlap") is None
    )

    # -- optimisation ---------------------------------------------------------------- #
    clauses["steps_ran"] = (
        _pos_int(steps.get("n_steps"))
        and _pos_int(steps.get("n_optimizer_steps"))
        and _pos_int(steps.get("world_size"))
        and steps.get("n_steps") == steps.get("expected_n_steps")
    )
    clauses["losses_finite"] = (
        _finite(losses.get("final_train_total"))
        and bool(losses.get("per_term_final"))
        and all(_finite(v) for v in (losses.get("per_term_final") or {}).values())
        and losses.get("n_nonfinite_steps") == 0
    )

    # -- the §10.3 LoRA contract, read back off the model --------------------------- #
    clauses["lora_contract_held"] = (
        applied.get("r") == 16
        and applied.get("lora_alpha") == 32
        and float(applied.get("lora_dropout") or -1) == 0.05
        and applied.get("all_linear_fully_covered") is True
        and wrap.get("base_frozen") is True
        and wrap.get("n_base_trainable_params") == 0
        and _pos_int(wrap.get("n_adapter_sites"))
    )
    heads = wrap.get("stage2_heads") or {}
    clauses["heads_outside_wrapper"] = (
        heads.get("heads_outside_peft_wrapper") is True
        and heads.get("all_heads_trainable") is True
        and _pos_int(heads.get("n_head_parameters"))
    )
    # Verified, not requested: a checkpointing flag that no-ops looks exactly like one that
    # works ([[in-process-no-ops-look-like-compliance]]).
    clauses["gradient_checkpointing_verified"] = (
        _pos_int(wrap.get("n_modules_with_checkpointing"))
        and wrap.get("checkpoint_use_reentrant") is False
        if wrap.get("gradient_checkpointing")
        else wrap.get("n_modules_with_checkpointing") == 0
    )

    # -- the objective actually optimised -------------------------------------------- #
    # `terms_seen` is the union of what `multitask_loss` actually INCLUDED in the total over
    # the whole run, so an active term that never once found supervision — a head trained on
    # nothing while every loss stayed finite — fails here rather than passing quietly.
    active = objective.get("active_terms") or []
    weights = objective.get("effective_weights") or {}
    binary_w = weights.get(L.BINARY_TERM)
    aux_w = [weights.get(term) for term in L.AUX_TERMS]
    # A term contributes only where it carries a NON-ZERO effective weight — that is what
    # `aux_weight=0` means and why the P3-08 no-aux arm is a real arm rather than a
    # differently-weighted one. So the expectation is re-derived from the recorded weights
    # instead of being the active set: demanding `terms_seen == active_terms` would fail the
    # no-aux arm by construction, and demanding nothing would let an aux head that never
    # once found supervision pass as trained.
    expected = [term for term in active if _finite(weights.get(term)) and weights[term] != 0.0]
    # Dominance is likewise re-derived from the recorded weights, NOT read back from the
    # config's own `holds` flag: Stage2LossConfig refuses a non-dominant config at
    # construction, so a clause echoing that flag could never fail.
    clauses["objective_terms_match"] = (
        bool(active)
        and expected == objective.get("terms_seen")
        and _finite(binary_w)
        and all(_finite(w) for w in aux_w)
        and binary_w >= sum(aux_w)
    )

    # -- artifacts + provenance ------------------------------------------------------ #
    clauses["checkpoint_written"] = (
        bool(ckpt.get("adapter_dir"))
        and _pos_int(ckpt.get("adapter_bytes"))
        and _pos_int(ckpt.get("heads_bytes"))
        and _pos_int(ckpt.get("n_saved_parameters"))
        if report.get("save_checkpoint")
        else ckpt.get("adapter_dir") is None
    )
    clauses["provenance_recorded"] = all(
        bool(prov.get(key)) for key in ("git_sha", "env_lock_sha256", "config_path", "step")
    ) and isinstance(prov.get("seed"), int)

    return {k: bool(v) for k, v in clauses.items()}


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Structural problems with ``report``; empty means it is well-formed AND passing."""
    problems: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {report.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    if report.get("step") != STEP:
        problems.append(f"step {report.get('step')!r} != {STEP!r}")
    for block in ("data", "steps", "wrap", "objective", "losses", "provenance", "gate"):
        if not isinstance(report.get(block), Mapping):
            problems.append(f"missing or non-mapping block {block!r}")
    gate = report.get("gate")
    if not isinstance(gate, Mapping):
        return problems
    recomputed = derive_clauses(report)
    recorded = gate.get("clauses")
    if not isinstance(recorded, Mapping):
        problems.append("gate.clauses is missing")
        return problems
    if set(recorded) != set(recomputed):
        problems.append(f"gate.clauses keys {sorted(recorded)} != re-derived {sorted(recomputed)}")
    for name, value in recomputed.items():
        if bool(recorded.get(name)) != value:
            problems.append(
                f"gate.clauses[{name!r}]={recorded.get(name)!r} but re-derivation says {value}"
            )
    if bool(gate.get("overall_pass")) != all(recomputed.values()):
        problems.append(
            f"gate.overall_pass={gate.get('overall_pass')!r} but the re-derived clauses give "
            f"{all(recomputed.values())}"
        )
    return problems


def build_report(
    cfg: Stage2TrainConfig,
    *,
    data: Mapping[str, Any],
    steps: Mapping[str, Any],
    wrap: Mapping[str, Any],
    objective: Mapping[str, Any],
    losses: Mapping[str, Any],
    val: Mapping[str, Any] | None,
    checkpoint: Mapping[str, Any],
    git_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the run's self-validating report.

    ``git_snapshot`` is taken **once, on rank 0, before training starts** and passed in.
    It is deliberately *not* re-read here: ``build_report`` runs on every DDP rank, and a
    clause that re-reads ``git status`` while rank 0 writes this run's own git-tracked
    report is a race that costs the entire run at the finish line
    ([[gate-reads-state-the-run-mutates]]). For the same reason the working-tree state is
    recorded as provenance and is **not** a gate clause — dirtiness at submit time is an
    operator fact, not a property of the training.
    """
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "prd": "PRD §10.2, §10.3, §11",
        "adr": "ADR-0002 (D2/D3/D5 backbone+kernels+LoRA); ADR-0004 D5 (fold); ADR-0005 D16",
        "written_at": datetime.now(UTC).isoformat(),
        "eval_val": bool(cfg.eval_val),
        "save_checkpoint": bool(cfg.save_checkpoint),
        "config": diagnostics(cfg),
        "data": _sanitize(data),
        "steps": _sanitize(steps),
        "wrap": _sanitize(wrap),
        "objective": _sanitize(objective),
        "losses": _sanitize(losses),
        "val": _sanitize(val) if val is not None else None,
        "checkpoint": _sanitize(checkpoint),
        "provenance": {
            "step": STEP,
            "config_path": CONFIG_PATH,
            "entrypoint": ENTRYPOINT,
            "script": "src/tbox_finder/stage2/train.py",
            "seed": int(cfg.seed),
            "env_lock": ENV_LOCK,
            "env_lock_sha256": _env_lock_sha256(),
            "dataset_parquet": cfg.dataset_parquet,
            **_sanitize(git_snapshot),
        },
    }
    clauses = derive_clauses(report)
    report["gate"] = {
        "clauses": clauses,
        "overall_pass": all(clauses.values()),
        "failed": sorted(name for name, ok in clauses.items() if not ok),
    }
    return report


# --------------------------------------------------------------------------------------
# Torch tier
# --------------------------------------------------------------------------------------
def load_rows(parquet: str | Path) -> list[dict[str, Any]]:
    """The Stage-2 dataset as plain dicts — the form the pure eligibility tier eats."""
    import pandas as pd

    frame = pd.read_parquet(parquet)
    return frame.to_dict(orient="records")


class Stage2SequenceDataset:
    """One admitted row → tokens + one target per active term.

    A ``torch.utils.data.Dataset`` by duck-typing (``__len__``/``__getitem__``); the base
    class is not subclassed so this module stays importable without torch, which is what
    lets the eligibility and collation contracts be graded in bare CI.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        spec: H.Stage2HeadSpec,
        *,
        loss_config: L.Stage2LossConfig,
    ) -> None:
        self._rows = list(rows)
        self._spec = spec
        self._cfg = loss_config
        self._terms = loss_config.active_terms()
        # Derived through the shipped mapping, never hardcoded as 0: `CLASS_CODE` →
        # `CODE_TO_CLASS` → this spec's own boundary vocabulary. A literal index would go
        # stale the moment ADR-0004 D1's class order changed, with nothing failing.
        self._background_index = spec.encode_boundary_string(labels.CLASS_CODE["background"])[0]

    @staticmethod
    def _is_unlabelled_negative(row: Mapping[str, Any]) -> bool:
        """A decoy, which by construction carries no per-nucleotide element annotation.

        Both conditions are required, and the check is fail-closed on purpose: a *corpus*
        row with an empty ``label_string``, or a row marked positive, must still hit the
        alignment guard rather than be quietly handed a fabricated all-background target.
        """
        return (
            masking.row_text(row.get("source")) == SOURCE_DECOY
            and _bool_or_none(row.get("is_tbox")) is False
        )

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[Mapping[str, Any]]:
        return self._rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        sequence = masking.row_text(row.get("rna_sequence"))
        if not sequence:
            raise ValueError(f"row {row.get('row_id')!r} has no rna_sequence")
        tok.assert_within_context(sequence, row_id=masking.row_text(row.get("row_id")))
        input_ids = tok.encode(sequence)
        n_nt = len(input_ids) - tok.N_FLANKING_SPECIAL_TOKENS

        item: dict[str, Any] = {"input_ids": input_ids, "row_id": row.get("row_id")}
        item[L.BINARY_TERM] = 1 if _bool_or_none(row.get("is_tbox")) is True else 0

        if "boundary" in self._terms:
            boundary = self._spec.encode_boundary_string(row.get(L.TERM_TO_FIELD["boundary"]))
            if not boundary and self._is_unlabelled_negative(row):
                # All 7,007 decoys carry `label_string = None` — a decoy has no T-box
                # element annotation, and none of the four pools ever did. They are
                # supervised as **all-background**, not ignored, following the Stage-1
                # convention this shares a SegmentationHead with: `data/negatives.py`
                # builds a negative with "every real nucleotide labelled `background`, with
                # no IGNORE_INDEX … That is not a trick". So head (b) learns that a
                # structured non-T-box RNA contains no T-box element anywhere — real
                # anti-mimicry signal, and the dense-background regime PRD §11's γ=2 focal
                # loss is written for. (User decision, 2026-08-03, after job 1036 died here.)
                boundary = [self._background_index] * n_nt
            if len(boundary) != n_nt:
                # Silent misalignment would train every position against its neighbour's
                # label while the loss stayed finite; that is the failure this catches.
                raise ValueError(
                    f"row {row.get('row_id')!r}: label_string is {len(boundary)} long but the "
                    f"sequence tokenises to {n_nt} nucleotides"
                )
            item["boundary"] = boundary

        for term in ("regulatory_mode", "specifier_codon", "cognate_aa", "trna_family"):
            if term in self._terms:
                item[term] = self._spec.encode(
                    L.TERM_TO_FIELD[term], row.get(L.TERM_TO_FIELD[term])
                )

        if L.STRUCTURE_TERM in self._terms:
            item[L.STRUCTURE_TERM] = L.encode_pairing_target(
                row.get("pairing_dotbracket"), length=n_nt
            )
        return item


def collate_stage2(batch: Sequence[Mapping[str, Any]], *, ignore_index: int = H.IGNORE_INDEX):
    """Pad a batch to ``(B, T)`` tokens and ``(B, T-2)`` nucleotide targets.

    ``T - 2`` because :meth:`Stage2Model.strip_special_tokens` drops ``<cls>`` and each
    row's own ``<eos>`` and leaves every row's nucleotides at indices ``0 … L_i-1``, so a
    per-nucleotide target padded to the same width is index-aligned with no per-row gather.
    Padding uses ``ignore_index`` — :func:`structure_consistency_loss` checks that it did.
    """
    import torch

    if not batch:
        raise ValueError("collate_stage2 received an empty batch")
    width = max(len(item["input_ids"]) for item in batch)
    n_nt = width - tok.N_FLANKING_SPECIAL_TOKENS

    input_ids = torch.full((len(batch), width), tok.PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1

    targets: dict[str, Any] = {}
    present = [key for key in L.TERMS if key in batch[0]]
    for term in present:
        if term in L.PER_NUCLEOTIDE_TERMS:
            padded = torch.full((len(batch), n_nt), ignore_index, dtype=torch.long)
            for i, item in enumerate(batch):
                values = item[term]
                padded[i, : len(values)] = torch.tensor(values, dtype=torch.long)
            targets[term] = padded
        else:
            targets[term] = torch.tensor([int(item[term]) for item in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "row_ids": [item["row_id"] for item in batch],
    }


def _count_checkpointed_modules(module: Any) -> int:
    """Modules whose ``gradient_checkpointing`` flag is actually on — a measurement."""
    return sum(1 for m in module.modules() if getattr(m, "gradient_checkpointing", False))


def build_model(cfg: Stage2TrainConfig, *, base_model: Any = None):
    """``(model, wrap_info)`` — the P3-03 model, LoRA-wrapped, checkpointing verified.

    Gradient checkpointing is re-enabled with ``use_reentrant=False``. That is not a taste
    call: this run needs ``find_unused_parameters=True`` (a batch of pure decoys supervises
    nothing but the binary and boundary terms, so ``multitask_loss`` drops the four
    record-level terms and their heads receive no gradient that step), and PyTorch's DDP
    documents "activation checkpointing when model has unused parameters" as unsupported
    for the *reentrant* variant outside ``static_graph=True``. ``static_graph`` is wrong
    here for exactly the same reason — the used/unused set changes from batch to batch, so
    telling DDP it is static would be a false statement it optimises against.
    """
    from tbox_finder.stage2.model import build_stage2_model

    spec = H.load_head_spec()
    kwargs: dict[str, Any] = {
        "base_model": base_model,
        "gradient_checkpointing": bool(cfg.gradient_checkpointing),
        "dropout": float(cfg.dropout),
        "boundary_use_crf": False,
        "structure_head": bool(cfg.structure_head),
        "pairing_proj_dim": int(cfg.pairing_proj_dim),
        "device": cfg.device,
    }
    if cfg.revision is not None:
        kwargs["revision"] = cfg.revision
    if cfg.dtype is not None:
        kwargs["dtype"] = cfg.dtype
    if cfg.attn_implementation is not None:
        kwargs["attn_implementation"] = cfg.attn_implementation

    model, info = build_stage2_model(spec, **kwargs)

    use_reentrant: bool | None = None
    if cfg.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        use_reentrant = False
        n_ckpt = _count_checkpointed_modules(model.backbone)
        if n_ckpt == 0:
            # Refuse rather than train a silently-unwrapped backbone at 4× the VRAM: a
            # flag that no-ops looks exactly like one that works.
            raise RuntimeError(
                "gradient_checkpointing was requested but no module carries the flag after "
                "gradient_checkpointing_enable(); the backbone would train unwrapped."
            )
    else:
        n_ckpt = _count_checkpointed_modules(model.backbone)

    info = {
        **info,
        "n_modules_with_checkpointing": int(n_ckpt),
        "checkpoint_use_reentrant": use_reentrant,
    }
    return model, info


def _trainable_parameters(model: Any) -> list[Any]:
    return [p for p in model.parameters() if p.requires_grad]


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "targets": {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()},
        "row_ids": batch["row_ids"],
    }


def _batched(dataset: Any, order: Iterable[int], *, batch_size: int, ignore_index: int):
    """Materialise ``order`` into equal-size batches; the trailing partial batch is kept.

    Batch *count* is what must match across ranks, and it does: every rank shards the same
    stream to the same length (:class:`~tbox_finder.train.ddp.ShardedSampler`), so the
    per-rank batch counts are identical even when the last batch is short.
    """
    chunk: list[Any] = []
    for index in order:
        chunk.append(dataset[index])
        if len(chunk) == batch_size:
            yield collate_stage2(chunk, ignore_index=ignore_index)
            chunk = []
    if chunk:
        yield collate_stage2(chunk, ignore_index=ignore_index)


def _n_batches(n_items: int, batch_size: int) -> int:
    return (n_items + batch_size - 1) // batch_size


def evaluate(model: Any, loss_fn: Any, dataset: Any, *, cfg: Stage2TrainConfig, device: Any):
    """Forward-only pass over the val rung → mean total loss, per-term means, accuracy."""
    import torch

    model.eval()
    order = list(range(len(dataset)))
    if cfg.eval_max_records is not None:
        order = order[: cfg.eval_max_records]

    total = 0.0
    per_term: dict[str, float] = {}
    n_batches = 0
    n_correct = 0
    n_scored = 0
    with torch.no_grad():
        for batch in _batched(
            dataset, order, batch_size=cfg.eval_batch_size, ignore_index=cfg.loss.ignore_index
        ):
            batch = _to_device(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, components = loss_fn(outputs, batch["targets"])
            total += float(loss.detach())
            for term, value in (components.get("terms") or {}).items():
                per_term[term] = per_term.get(term, 0.0) + float(value.detach())
            n_batches += 1
            predicted = (outputs["tbox_logit"].detach().float() > 0).long()
            truth = batch["targets"][L.BINARY_TERM]
            n_correct += int((predicted == truth).sum())
            n_scored += int(truth.numel())
    model.train()
    if n_batches == 0:
        return {
            "n_batches": 0,
            "n_records": 0,
            "total": None,
            "per_term": {},
            "binary_accuracy": None,
        }
    return {
        "n_batches": n_batches,
        "n_records": len(order),
        "total": total / n_batches,
        "per_term": {k: v / n_batches for k, v in sorted(per_term.items())},
        "binary_accuracy": n_correct / n_scored if n_scored else None,
    }


def _init_wandb(cfg: Stage2TrainConfig, *, run_name: str) -> Any:
    """Offline W&B on rank 0 only; a missing wandb is a warning, never a run-killer."""
    os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)
    os.environ.setdefault("WANDB_DIR", cfg.wandb_dir)
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        mode=cfg.wandb_mode,
        dir=cfg.wandb_dir,
        config=diagnostics(cfg),
        reinit=True,
    )


def train_stage2(
    cfg: Stage2TrainConfig, *, log: Any = print, base_model: Any = None
) -> dict[str, Any]:
    """Run the fine-tune and return the report. Raises if the run fails its own gate."""
    import torch
    import torch.distributed as dist

    ddp.check_pythonhashseed(cfg.pythonhashseed, entrypoint=ENTRYPOINT)

    from tbox_finder.train.repro import set_determinism

    set_determinism(cfg.seed)

    world_size = ddp.ddp_world_size()
    rank = ddp.ddp_rank()
    local_rank = ddp.ddp_local_rank()
    primary = ddp.is_primary()

    # Snapshot git ONCE, on rank 0, before anything writes — see build_report's docstring.
    git_snapshot: dict[str, Any] = {}
    if primary:
        status = _git("status", "--porcelain", "--untracked-files=no")
        git_snapshot = {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
            "git_dirty_paths": sorted(
                line[3:] for line in (status or "").splitlines() if len(line) > 3
            ),
        }

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(cfg.device or "cpu")

    # -- population ------------------------------------------------------------------- #
    rows = load_rows(cfg.dataset_parquet)
    train_idx, train_census = select_rows(
        rows, rung=cfg.train_rung, admit_parentless_decoys=cfg.admit_parentless_decoys
    )
    val_idx, val_census = (
        select_rows(rows, rung=cfg.val_rung, admit_parentless_decoys=cfg.admit_parentless_decoys)
        if cfg.eval_val
        else ([], {})
    )
    if cfg.max_records is not None:
        train_idx = train_idx[: cfg.max_records]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    if not train_rows:
        raise RuntimeError(
            f"the eligibility rule admitted no rows on rung {cfg.train_rung!r}; refusals "
            f"were {train_census.get('refused')}"
        )

    overlap = (
        len(
            {masking.row_text(r.get("row_id")) for r in train_rows}
            & {masking.row_text(r.get("row_id")) for r in val_rows}
        )
        if cfg.eval_val
        else None
    )

    spec = H.load_head_spec()
    train_ds = Stage2SequenceDataset(train_rows, spec, loss_config=cfg.loss)
    val_ds = Stage2SequenceDataset(val_rows, spec, loss_config=cfg.loss) if cfg.eval_val else None

    # -- model + objective ------------------------------------------------------------ #
    model, wrap = build_model(cfg, base_model=base_model)
    model.to(device)
    loss_fn = L.MultitaskLoss(cfg.loss)

    trainable = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    sampler = ddp.ShardedSampler(
        EpochShuffleSampler(len(train_ds), seed=cfg.seed), rank=rank, world_size=world_size
    )
    per_epoch = _n_batches(len(sampler), cfg.batch_size)
    total_steps = per_epoch * cfg.epochs
    warmup_steps = int(round(cfg.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_scale(s, warmup_steps=warmup_steps, total_steps=total_steps)
    )

    ddp_model = model
    if world_size > 1:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            # Required, not tuned: `multitask_loss` drops a term with no supervision in the
            # batch, so a pure-decoy batch leaves the four record-level heads without a
            # gradient. `static_graph=True` would be the faster answer and is FALSE here —
            # the used/unused set genuinely changes batch to batch.
            find_unused_parameters=True,
        )

    run = _init_wandb(cfg, run_name=f"{STEP}-{Path(cfg.checkpoint_dir).name}") if primary else None

    # -- the loop ---------------------------------------------------------------------- #
    n_steps = 0
    n_optimizer_steps = 0
    n_nonfinite = 0
    terms_seen: set[str] = set()
    per_term_final: dict[str, float] = {}
    final_total = float("nan")
    best_val: float | None = None
    best_epoch: int | None = None
    val_history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(
            _batched(
                train_ds,
                iter(sampler),
                batch_size=cfg.batch_size,
                ignore_index=cfg.loss.ignore_index,
            )
        ):
            batch = _to_device(batch, device)
            outputs = ddp_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            loss, components = loss_fn(outputs, batch["targets"])
            terms_seen.update(components.get("included") or [])
            value = float(loss.detach())
            if not math.isfinite(value):
                n_nonfinite += 1
            (loss / cfg.gradient_accumulation_steps).backward()
            n_steps += 1
            if n_steps % cfg.gradient_accumulation_steps == 0:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                n_optimizer_steps += 1
            final_total = value
            per_term_final = {
                term: float(t.detach()) for term, t in (components.get("terms") or {}).items()
            }
            if primary and n_steps % cfg.log_every == 0:
                log(f"epoch {epoch} step {i} loss {value:.4f} lr {scheduler.get_last_lr()[0]:.2e}")
                if run is not None:
                    run.log(
                        {
                            "train/total": value,
                            "train/lr": scheduler.get_last_lr()[0],
                            **{f"train/{k}": v for k, v in per_term_final.items()},
                        },
                        step=n_steps,
                    )

        if cfg.eval_val and val_ds is not None:
            metrics = evaluate(model, loss_fn, val_ds, cfg=cfg, device=device)
            metrics["epoch"] = epoch
            val_history.append(metrics)
            if primary:
                log(f"epoch {epoch} val {metrics['total']}")
                if run is not None:
                    run.log(
                        {
                            "val/total": metrics["total"],
                            "val/binary_accuracy": metrics["binary_accuracy"],
                        },
                        step=n_steps,
                    )
            if metrics["total"] is not None and (best_val is None or metrics["total"] < best_val):
                best_val, best_epoch = metrics["total"], epoch

    elapsed = time.time() - started

    # -- checkpoint (rank 0 only) ------------------------------------------------------ #
    checkpoint: dict[str, Any] = {
        "adapter_dir": None,
        "heads_path": None,
        "adapter_bytes": 0,
        "heads_bytes": 0,
        "n_saved_parameters": 0,
        "best_val_total": best_val,
        "best_val_epoch": best_epoch,
        "saved_from_epoch": cfg.epochs - 1,
    }
    if cfg.save_checkpoint and primary:
        out = Path(cfg.checkpoint_dir)
        adapter_dir = out / ADAPTER_SUBDIR
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        model.backbone.save_pretrained(str(adapter_dir))
        head_state = {
            name: param.detach().cpu()
            for attr, module in model.head_modules.items()
            for name, param in ((f"{attr}.{n}", p) for n, p in module.state_dict().items())
        }
        heads_path = out / HEADS_STATE_NAME
        torch.save(head_state, heads_path)
        checkpoint.update(
            adapter_dir=str(adapter_dir),
            heads_path=str(heads_path),
            adapter_bytes=sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file()),
            heads_bytes=heads_path.stat().st_size,
            n_saved_parameters=sum(int(t.numel()) for t in head_state.values())
            + sum(int(p.numel()) for n, p in model.backbone.named_parameters() if "lora_" in n),
        )

    report = build_report(
        cfg,
        data={
            "dataset_parquet": cfg.dataset_parquet,
            "n_dataset_rows": len(rows),
            "n_train_rows": len(train_rows),
            "n_val_rows": len(val_rows),
            "train_census": train_census,
            "val_census": val_census or None,
            "n_train_val_row_id_overlap": overlap,
        },
        steps={
            "n_steps": n_steps,
            "expected_n_steps": per_epoch * cfg.epochs,
            "n_optimizer_steps": n_optimizer_steps,
            "batches_per_epoch_per_rank": per_epoch,
            "world_size": world_size,
            "warmup_steps": warmup_steps,
            "total_scheduled_steps": total_steps,
            "elapsed_seconds": round(elapsed, 3),
        },
        wrap=wrap,
        objective={
            "active_terms": list(cfg.loss.active_terms()),
            # TERMS order on both sides, so the clause compares sets by a stable ordering
            # rather than accidentally comparing sorted-alphabetically to declaration order.
            "terms_seen": [t for t in L.TERMS if t in terms_seen],
            "dominance": cfg.loss.dominance(),
            "weighting": cfg.loss.weighting,
            "effective_weights": cfg.loss.effective_weights(),
            "max_aux_weight": cfg.loss.max_aux_weight,
        },
        losses={
            "final_train_total": final_total,
            "per_term_final": per_term_final,
            "n_nonfinite_steps": n_nonfinite,
        },
        val=(
            {"history": val_history, "best_total": best_val, "best_epoch": best_epoch}
            if cfg.eval_val
            else None
        ),
        checkpoint=checkpoint,
        git_snapshot=git_snapshot,
    )

    if primary:
        out = Path(cfg.report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if run is not None:
            run.finish()

    problems = validate_report(report)
    if problems:
        raise RuntimeError(f"P3-06 report is malformed: {problems}")
    if not report["gate"]["overall_pass"]:
        raise RuntimeError(
            f"P3-06 gate FAILED (clauses: {', '.join(report['gate']['failed'])}). The report "
            f"was written to {cfg.report_path} for inspection, but a run that fails its own "
            "gate must not exit 0."
        )
    return report


# --------------------------------------------------------------------------------------
# Hydra entry
# --------------------------------------------------------------------------------------
_LOSS_FIELDS = frozenset(f.name for f in fields(L.Stage2LossConfig))


def _cfg_from_mapping(cfg: Mapping[str, Any]) -> Stage2TrainConfig:
    """Build the config from a resolved Hydra tree.

    Unknown keys inside the ``/loss`` and ``/optim`` groups **raise**. Stage-1's flattener
    ignores unknown keys because its composed tree legitimately carries group metadata it
    reads elsewhere; here a stray ``loss.aux_wieght=0.0`` on a sweep line would train the
    default weighting and self-consistently report the value it was handed, which is the
    exact shape of a sweep that measures nothing.
    """
    from tbox_finder.eval.rinalmo_parity import REPO_ID, REVISION  # bare-importable

    optim = cfg.get("optim") or {}
    tracking = cfg.get("tracking") or {}
    loss_block = cfg.get("loss") or {}
    model_block = cfg.get("model") or {}

    # The /model group is a RECORD (P1-13), not an input: the checkpoint identity is frozen
    # in code so no config can weaken it. That makes `model.revision=<x>` on a sweep line an
    # override that changes nothing — so refuse the drift rather than let the run report a
    # checkpoint it did not load. Overriding the checkpoint is `revision=<x>` at top level,
    # which does reach the loader and is itself gated by lora_harness's pinned-revision check.
    for key, pinned in (("repo_id", REPO_ID), ("revision", REVISION)):
        if key in model_block and masking.row_text(model_block[key]) != pinned:
            raise ValueError(
                f"conf/model/rinalmo_stage2.yaml records {key}={model_block[key]!r} but the "
                f"code pins {pinned!r}. The /model group is a record of the frozen checkpoint "
                "identity, not an input — a value only this file carries would name a "
                "checkpoint the run never loaded."
            )

    unknown_loss = sorted(set(loss_block) - _LOSS_FIELDS)
    if unknown_loss:
        raise ValueError(
            f"conf/loss/stage2.yaml carries keys Stage2LossConfig does not define: "
            f"{unknown_loss}. A mis-keyed loss override changes nothing while the run "
            "reports it as set."
        )

    known = frozenset(Stage2TrainConfig.__dataclass_fields__) - {"loss"}
    values: dict[str, Any] = {k: v for k, v in cfg.items() if k in known}
    for source, key, target in (
        (optim, "lr", "lr"),
        (optim, "weight_decay", "weight_decay"),
        (optim, "warmup_ratio", "warmup_ratio"),
        (optim, "grad_clip", "grad_clip"),
        (tracking, "mode", "wandb_mode"),
        (tracking, "project", "wandb_project"),
        (tracking, "entity", "wandb_entity"),
        (tracking, "dir", "wandb_dir"),
    ):
        if isinstance(source, Mapping) and key in source:
            values[target] = source[key]
    return Stage2TrainConfig(loss=L.Stage2LossConfig(**dict(loss_block)), **values)


def main() -> None:
    """Hydra entry: ``python -m tbox_finder.stage2.train``.

    The primary config lives at ``conf/train/stage2.yaml`` and is selected by its *slashed*
    group path — a ``conf/<group>/`` primary needs ``@package _global_`` plus leading-slash
    ``/group`` defaults to reach its sibling groups, or composition silently no-ops.
    """
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../../../conf", config_name="train/stage2")
    def _entry(cfg: Any) -> None:
        train_stage2(_cfg_from_mapping(OmegaConf.to_container(cfg, resolve=True)))

    _entry()


if __name__ == "__main__":  # pragma: no cover
    main()
