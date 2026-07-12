"""P1-05 — Transfer go/no-go part (i): frozen-embedding binary linear-separability
pre-filter (cheap, **non-binding**).

The public Caduceus-PS checkpoints are **human-reference pretrained**, while the
discovery scan is fully prokaryotic/archaeal/MAG (PRD §10.1; ADR-0002 D7). This module
answers the *cheap* half of the two-part Stage-1 transfer go/no-go: **is there
above-chance linear separability in the frozen embeddings?** A scikit-learn logistic
linear probe is fit over frozen (no-fine-tune) Caduceus-PS embeddings of

  - **positives** — held-out (leave-clade-out) bacterial T-box windows from the P0
    split table (``nested_role == "heldout"`` & ``source == "corpus"``), and
  - **negatives** — GC-matched prokaryotic background windows (§9.1 class 1;
    ``decoys_v0.parquet`` pool ``gc_background``).

and evaluated on a **cluster-grouped, homology-safe probe split** (``GroupShuffleSplit``
by ``cluster_id``, so no sequence cluster spans the probe train/test boundary — the
load-bearing §8.2 anti-leakage unit). Positives come from the P0 leave-clade-out
held-out fold; ``resolved_order`` overlap across the probe boundary is informational
(distinct clusters of one order are not homology leakage). The metric reported is
balanced-accuracy + AUROC (with a seeded bootstrap CI) on the held-out probe split.

**This gate is non-binding** (ADR-0002 D7 part i is an *advisory* screen): a low
pre-filter can coexist with a passing full fine-tune, and the *binding* gate is the
P1-07 per-nucleotide segmentation smoke — a binary probe can pass while the embeddings
lack the positional structure segmentation needs, or fail while full fine-tuning
succeeds. The ADR pins **no numeric separability floor** for the pre-filter; the pinned
criterion is qualitative — *above-chance* separability. We therefore operationalize the
verdict against chance (0.5) via the bootstrap CI (see :func:`classify_separability`);
this is an implementer operationalization of the ADR's qualitative criterion, **not** a
new pinned gate number.

**Two-stage by environment (no env change → no ADR-0002 D8 bump).** No single committed
env carries both ``torch`` and ``scikit-learn``: extraction runs under ``tbox-ml-dna``
(``--stage extract`` → frozen forward → embeddings ``.npz`` cache), the probe runs under
``tbox-data`` (``--stage probe`` → sklearn fit → report). ``--stage all`` runs both when
an env provides both. All heavy imports are lazy so the module imports in a bare env.

Rule/Entry: :func:`run_prefilter`.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbox_finder import provenance
from tbox_finder.models.caduceus_backbone import D_MODEL, REPO_ID, REVISION

# --------------------------------------------------------------------------------------
# Constants (gate-critical values are frozen in code, not config).
# --------------------------------------------------------------------------------------
SCHEMA_VERSION = "1"
STEP = "P1-05"
GENERATED_BY = "src/tbox_finder/probes/frozen_linear_probe.py"
ADR = "ADR-0002"

#: Frozen Caduceus-PS embedding width == 2*d_model (fwd‖rc channels), single-sourced
#: from :data:`tbox_finder.models.caduceus_backbone.D_MODEL`.
EMB_DIM: int = 2 * D_MODEL

#: The ADR-0002 D7 part-(i) pre-filter criterion is *above-chance* separability. Chance
#: for balanced-accuracy and AUROC on a binary task is 0.5.
CHANCE_LEVEL: float = 0.5
VERDICTS = ("PASS", "borderline", "FAIL")

# Data selection (P0 split table + §9.1 decoys).
HELDOUT_ROLE = "heldout"  # nested_role value for the leave-clade-out held-out fold
POSITIVE_SOURCE = "corpus"  # exclude blind/anchor rows
BACKGROUND_POOL = "gc_background"  # §9.1 class-1 GC+length-matched composition null
ORDER_COL = "resolved_order"
CLUSTER_COL = "cluster_id"

# Defaults (repo-relative; overridable on the CLI).
DEFAULT_SPLIT_TABLE = "data/processed/splits/split_assignments.parquet"
DEFAULT_DECOYS = "data/processed/negatives/decoys_v0.parquet"
DEFAULT_FASTA_DIR = "data/interim/splits/inputs"
DEFAULT_OUT = "reports/p1/prefilter_separability.json"
DEFAULT_EMB_CACHE = "data/interim/probes/prefilter_embeddings.npz"

# Probe hyper-parameters (frozen defaults; deterministic).
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.25
DEFAULT_C = 1.0
DEFAULT_MAX_ITER = 1000
DEFAULT_N_BOOTSTRAP = 2000

# Env-lock files recorded in provenance (§11).
EXTRACT_ENV_LOCK = "envs/ml-dna.conda-lock.yml"
PROBE_ENV_LOCK = "envs/data.conda-lock.yml"

_BASE_TOKENS = "ACGTN"  # Caduceus char vocab bases we map explicitly


# --------------------------------------------------------------------------------------
# Pure stdlib helpers (bare-testable — no numpy/sklearn/torch).
# --------------------------------------------------------------------------------------
def _sanitize(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` so the JSON is strict (no NaN/Inf)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _finite_number(v: Any) -> bool:
    """True iff ``v`` is a real, finite number (bools rejected)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _prob01(v: Any) -> bool:
    """True iff ``v`` is a finite number in the closed unit interval."""
    return _finite_number(v) and 0.0 <= float(v) <= 1.0


def _bad_bool(value: Any, expected: bool) -> bool:
    """True iff ``value`` is not the exact boolean ``expected`` (int 0/1 rejected)."""
    return not (isinstance(value, bool) and value is expected)


def classify_separability(auroc: Any, auroc_ci_lower: Any, *, chance: float = CHANCE_LEVEL) -> str:
    """Operationalize the ADR-0002 D7 *above-chance* pre-filter criterion into a verdict.

    - ``PASS`` — the AUROC bootstrap-CI **lower bound** is strictly above chance
      (separability is statistically above chance).
    - ``borderline`` — the AUROC point estimate is above chance but the CI includes
      chance (weak/uncertain separability). Per ADR-0002 a borderline read on the
      *binding* go/no-go is a §7 stop-and-ask; here (non-binding) it is an advisory
      label only.
    - ``FAIL`` — the AUROC point estimate is at or below chance, or unmeasurable
      (non-finite → fail-closed, §10.3).

    This is an implementer operationalization of the ADR's qualitative criterion, not a
    new pinned gate number (P0-31 precedent).
    """
    if not _finite_number(auroc):
        return "FAIL"
    if float(auroc) <= chance:
        return "FAIL"
    if _finite_number(auroc_ci_lower) and float(auroc_ci_lower) > chance:
        return "PASS"
    return "borderline"


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a list of schema problems for a pre-filter report (empty ⇒ valid).

    Never raises. Fails **closed**: a report whose **cluster**-leakage flag indicates a
    sequence cluster spanning the probe boundary (§8.2 homology guard) is invalid, as is
    a non-finite/out-of-range metric, an unknown verdict, a chance level != 0.5, or a
    report that claims to be binding. ``order_spans_boundary`` is informational (a
    cluster-safe split may still put distinct clusters of one order on both sides — not
    homology leakage), so it must be present and boolean but its value is not gated.
    """
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if report.get("step") != STEP:
        errors.append(f"step must be {STEP!r}")
    if _bad_bool(report.get("measured"), True):
        errors.append("measured must be boolean True")

    floor = report.get("floor")
    if not isinstance(floor, dict):
        errors.append("floor block missing")
    else:
        if floor.get("chance_level") != CHANCE_LEVEL:
            errors.append("floor.chance_level must be 0.5")
        if _bad_bool(floor.get("non_binding"), True):
            errors.append("floor.non_binding must be boolean True")

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics block missing")
    else:
        for key in ("balanced_accuracy", "auroc"):
            if not _prob01(metrics.get(key)):
                errors.append(f"metrics.{key} must be a finite number in [0, 1]")
        for key in ("auroc_ci_lower", "auroc_ci_upper"):
            # CI bounds may be null when the bootstrap is degenerate; if present they
            # must be finite in [0, 1].
            val = metrics.get(key)
            if val is not None and not _prob01(val):
                errors.append(f"metrics.{key} must be null or a finite number in [0, 1]")
        lo, hi = metrics.get("auroc_ci_lower"), metrics.get("auroc_ci_upper")
        if _prob01(lo) and _prob01(hi) and float(lo) > float(hi):
            errors.append("metrics.auroc_ci_lower must be <= metrics.auroc_ci_upper")

    leakage = report.get("leakage")
    if not isinstance(leakage, dict):
        errors.append("leakage block missing")
    else:
        if _bad_bool(leakage.get("cluster_spans_boundary"), False):
            errors.append(
                "leakage.cluster_spans_boundary must be boolean False (§8.2 homology guard)"
            )
        if not isinstance(leakage.get("order_spans_boundary"), bool):
            errors.append("leakage.order_spans_boundary must be present and boolean")
        if _bad_bool(leakage.get("positives_heldout_only"), True):
            errors.append("leakage.positives_heldout_only must be boolean True (§9.2)")

    gate = report.get("gate")
    if not isinstance(gate, dict):
        errors.append("gate block missing")
    else:
        if gate.get("verdict") not in VERDICTS:
            errors.append(f"gate.verdict must be one of {VERDICTS}")
        if _bad_bool(gate.get("binding"), False):
            errors.append("gate.binding must be boolean False (non-binding pre-filter)")
        if gate.get("chance_level") != CHANCE_LEVEL:
            errors.append("gate.chance_level must be 0.5")
        # Internal consistency: the verdict + above_chance must follow from the metrics,
        # so a hand-edited gate that contradicts the numbers is rejected.
        if isinstance(metrics, dict) and _prob01(metrics.get("auroc")):
            expected = classify_separability(metrics.get("auroc"), metrics.get("auroc_ci_lower"))
            if gate.get("verdict") != expected:
                errors.append(f"gate.verdict inconsistent with metrics (expected {expected!r})")
            if _bad_bool(gate.get("above_chance"), expected in ("PASS", "borderline")):
                errors.append("gate.above_chance inconsistent with the verdict")
    return errors


# --------------------------------------------------------------------------------------
# Stage A — frozen embedding extraction (lazy torch/transformers; GPU required).
# --------------------------------------------------------------------------------------
def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a two-line-per-record FASTA into ``{record_id: sequence}`` (stdlib)."""
    seqs: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    seqs[header] = "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            elif header is not None:
                chunks.append(line)
    if header is not None:
        seqs[header] = "".join(chunks)
    return seqs


def extract_embeddings(
    sequences: Sequence[str],
    *,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    log=lambda _m: None,
):
    """Frozen Caduceus-PS embeddings — mean-pool over positions → ``(N, EMB_DIM)``.

    Loads the code-pinned backbone in eval mode, freezes every parameter, and runs a
    single-sequence (batch=1) forward per window under ``torch.no_grad()`` so the
    variable-length windows are pooled over exactly their real positions (no pad-token
    contamination). Deterministic: no dropout in eval, no sampling, TF32 disabled.

    Returns a ``numpy.ndarray`` of dtype float32, shape ``(len(sequences), EMB_DIM)``.
    """
    import numpy as np  # lazy
    import torch  # lazy

    from tbox_finder.models.caduceus_backbone import (
        _hidden_states,
        load_caduceus_ps,
        load_tokenizer,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_caduceus_ps(device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = load_tokenizer()

    id_map: dict[str, int] = {}
    for base in _BASE_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(base)
        if tid is not None and tid != tokenizer.unk_token_id:
            id_map[base] = int(tid)
    unk = tokenizer.unk_token_id
    if unk is None:
        unk = id_map["A"]

    def ids_of(seq: str) -> list[int]:
        return [id_map.get(c, unk) for c in seq.upper()]

    out = np.empty((len(sequences), EMB_DIM), dtype=np.float32)
    n = len(sequences)
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            x = torch.tensor([ids_of(seq)], device=device)
            h = _hidden_states(model(input_ids=x))  # (1, L, EMB_DIM)
            out[i] = h.mean(dim=1).squeeze(0).float().cpu().numpy()
            if (i + 1) % 1000 == 0 or (i + 1) == n:
                log(f"  extracted {i + 1}/{n} windows")
    return out


def _select_positive_records(split_table: str | Path):
    """Held-out (leave-clade-out) bacterial-T-box positive rows from the P0 split table.

    Returns a pandas DataFrame with ``record_id``, ``klass``, ``resolved_order``,
    ``cluster_id`` for rows with ``nested_role == "heldout"`` and
    ``source == "corpus"`` (§9.2 held-out-fold only; blind/anchor excluded).
    """
    import pandas as pd  # lazy

    df = pd.read_parquet(split_table)
    mask = (df["nested_role"] == HELDOUT_ROLE) & (df["source"] == POSITIVE_SOURCE)
    cols = ["record_id", "klass", ORDER_COL, CLUSTER_COL]
    return df.loc[mask, cols].reset_index(drop=True)


def _select_background_records(decoys: str | Path):
    """§9.1 class-1 GC-matched background rows (``pool == gc_background``)."""
    import pandas as pd  # lazy

    df = pd.read_parquet(decoys)
    bg = df.loc[df["pool"] == BACKGROUND_POOL, ["decoy_id", "sequence", "gc", "length"]]
    return bg.reset_index(drop=True)


def run_extract(
    *,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    decoys: str | Path = DEFAULT_DECOYS,
    fasta_dir: str | Path = DEFAULT_FASTA_DIR,
    emb_cache: str | Path = DEFAULT_EMB_CACHE,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    log=print,
) -> Path:
    """Stage A: select records, extract frozen embeddings, write the ``.npz`` cache."""
    import numpy as np  # lazy

    pos = _select_positive_records(split_table)
    fasta_dir = Path(fasta_dir)
    fasta = read_fasta(fasta_dir / "class_I.fa")
    fasta.update(read_fasta(fasta_dir / "class_II.fa"))

    pos_seq: list[str] = []
    keep = np.ones(len(pos), dtype=bool)
    for i, rid in enumerate(pos["record_id"].tolist()):
        seq = fasta.get(rid)
        if seq is None:
            keep[i] = False
            continue
        pos_seq.append(seq)
    if not keep.all():
        missing = int((~keep).sum())
        log(f"WARNING: {missing} positive record_id(s) had no FASTA sequence; dropped")
        pos = pos.loc[keep].reset_index(drop=True)

    bg = _select_background_records(decoys)
    neg_seq = bg["sequence"].tolist()

    log(f"extracting positives: {len(pos_seq)} held-out windows")
    x_pos = extract_embeddings(pos_seq, device=device, seed=seed, log=log)
    log(f"extracting background: {len(neg_seq)} gc_background windows")
    x_neg = extract_embeddings(neg_seq, device=device, seed=seed, log=log)

    emb_cache = Path(emb_cache)
    emb_cache.parent.mkdir(parents=True, exist_ok=True)
    # String columns are stored as fixed-width Unicode (dtype=str), NOT object, so the
    # cache loads with allow_pickle=False (no pickle-deserialization surface).
    pos_order = pos[ORDER_COL].where(pos[ORDER_COL].notna(), "(unresolved)").to_numpy(dtype=str)
    # Metadata pinned into the cache so run_probe validates + reports what was ACTUALLY
    # extracted (revision/seed/emb-dim/source hashes) rather than reconstructing from
    # current values — a stale/mismatched cache is then detectable.
    extract_inputs = {}
    for path in (split_table, decoys, fasta_dir / "class_I.fa", fasta_dir / "class_II.fa"):
        try:
            extract_inputs[str(path)] = provenance.sha256_file(path)
        except Exception:
            extract_inputs[str(path)] = "unavailable"
    meta = {
        "revision": REVISION,
        "backbone": REPO_ID,
        "emb_dim": EMB_DIM,
        "seed": int(seed),
        "pooling": "mean-over-positions",
        "extract_inputs": extract_inputs,
    }
    np.savez(
        emb_cache,
        x_pos=x_pos,
        x_neg=x_neg,
        pos_order=pos_order,
        pos_cluster=pos[CLUSTER_COL].to_numpy(),
        pos_record_id=pos["record_id"].to_numpy(dtype=str),
        pos_klass=pos["klass"].to_numpy(dtype=str),
        neg_gc=bg["gc"].to_numpy(dtype=float),
        neg_length=bg["length"].to_numpy(),
        seed=np.int64(seed),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    log(f"wrote embeddings cache: {emb_cache} " f"(pos {x_pos.shape}, neg {x_neg.shape})")
    return emb_cache


# --------------------------------------------------------------------------------------
# Stage B — logistic linear probe + report (lazy scikit-learn/numpy; CPU).
# --------------------------------------------------------------------------------------
def _bootstrap_auroc_ci(
    y_true, y_score, *, seed: int, n_bootstrap: int
) -> tuple[float, float, int]:
    """Seeded percentile 95% CI for AUROC via test-set resampling (both bounds)."""
    import numpy as np  # lazy
    from sklearn.metrics import roc_auc_score  # lazy

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    vals: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # single-class resample — AUROC undefined; skip
            continue
        vals.append(float(roc_auc_score(yt, y_score[idx])))
    if len(vals) < max(10, n_bootstrap // 10):
        return math.nan, math.nan, len(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def build_probe_report(
    x_pos,
    x_neg,
    pos_order: Sequence[Any],
    pos_cluster: Sequence[Any],
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    c: float = DEFAULT_C,
    max_iter: int = DEFAULT_MAX_ITER,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> dict[str, Any]:
    """Fit the logistic linear probe on a homology-safe split; return the core report.

    The positive windows are split by ``cluster_id`` (``GroupShuffleSplit``) so **no
    sequence cluster spans the probe train/test boundary** — the load-bearing §8.2
    anti-leakage unit, since near-duplicate homologs in both train and test would
    inflate a separability estimate. (Grouping by ``resolved_order`` instead cannot also
    guarantee this here: ~41 clusters span >1 order, so an order-grouped split leaks
    clusters; those 41 clusters carry ~20% of the held-out positives, too lossy to drop.
    ``order_spans_boundary`` is therefore *reported* but not gated — two distinct
    clusters of one order on opposite sides is not homology leakage.) Positives are the
    P0 leave-clade-out held-out fold (``nested_role == heldout``), so the *source* is
    clade-held-out. The background windows are split by a seeded random draw. A
    ``StandardScaler`` + ``LogisticRegression(class_weight="balanced")`` pipeline is fit
    on train and scored on the held-out probe split; reports balanced-accuracy + AUROC
    (seeded bootstrap CI) and the ADR-0002 above-chance verdict.
    """
    import numpy as np  # lazy
    from sklearn.linear_model import LogisticRegression  # lazy
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # lazy
    from sklearn.model_selection import GroupShuffleSplit  # lazy
    from sklearn.pipeline import make_pipeline  # lazy
    from sklearn.preprocessing import StandardScaler  # lazy

    x_pos = np.asarray(x_pos, dtype=np.float64)
    x_neg = np.asarray(x_neg, dtype=np.float64)
    pos_order = np.asarray(pos_order, dtype=object)
    pos_cluster = np.asarray(pos_cluster)

    # Homology-safe split of the positives (grouped by cluster_id — §8.2: no sequence
    # cluster may span the probe train/test boundary).
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    pos_train_idx, pos_test_idx = next(gss.split(x_pos, groups=pos_cluster))

    # Random seeded split of the background (no taxonomy → free to split).
    rng = np.random.default_rng(seed)
    neg_perm = rng.permutation(len(x_neg))
    n_neg_test = int(round(len(x_neg) * test_size))
    neg_test_idx = neg_perm[:n_neg_test]
    neg_train_idx = neg_perm[n_neg_test:]

    x_train = np.vstack([x_pos[pos_train_idx], x_neg[neg_train_idx]])
    y_train = np.concatenate([np.ones(len(pos_train_idx), int), np.zeros(len(neg_train_idx), int)])
    x_test = np.vstack([x_pos[pos_test_idx], x_neg[neg_test_idx]])
    y_test = np.concatenate([np.ones(len(pos_test_idx), int), np.zeros(len(neg_test_idx), int)])

    # Leakage checks (§8.2): no clade or cluster spans the positive train/test boundary.
    orders_train = set(pos_order[pos_train_idx].tolist())
    orders_test = set(pos_order[pos_test_idx].tolist())
    shared_orders = sorted(orders_train & orders_test)
    clusters_train = set(pos_cluster[pos_train_idx].tolist())
    clusters_test = set(pos_cluster[pos_test_idx].tolist())
    shared_clusters = sorted(clusters_train & clusters_test)

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced", random_state=seed),
    )
    pipe.fit(x_train, y_train)
    y_score = pipe.predict_proba(x_test)[:, 1]
    y_pred = pipe.predict(x_test)

    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    auroc = float(roc_auc_score(y_test, y_score))
    ci_lo, ci_hi, n_valid = _bootstrap_auroc_ci(y_test, y_score, seed=seed, n_bootstrap=n_bootstrap)
    verdict = classify_separability(auroc, ci_lo)

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "measured": True,
        "generated_by": GENERATED_BY,
        "adr": ADR,
        "floor": {
            "criterion": "above-chance linear separability (ADR-0002 D7 part i)",
            "chance_level": CHANCE_LEVEL,
            "non_binding": True,
            "operationalization": (
                "PASS iff AUROC 95% bootstrap CI lower bound > 0.5; borderline iff "
                "point estimate > 0.5 but the CI includes 0.5; FAIL iff point <= 0.5 "
                "or unmeasurable"
            ),
        },
        "probe": {
            "model": "logistic_regression",
            "scaler": "standard",
            "class_weight": "balanced",
            "C": float(c),
            "max_iter": int(max_iter),
            "seed": int(seed),
            "test_size": float(test_size),
            "split_scheme": (
                "homology-safe (GroupShuffleSplit by cluster_id); positives from the "
                "P0 leave-clade-out held-out fold"
            ),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_train_pos": int(len(pos_train_idx)),
            "n_test_pos": int(len(pos_test_idx)),
            "n_train_neg": int(len(neg_train_idx)),
            "n_test_neg": int(len(neg_test_idx)),
            "n_clusters_train": int(len(clusters_train)),
            "n_clusters_test": int(len(clusters_test)),
            "n_orders_train": int(len(orders_train)),
            "n_orders_test": int(len(orders_test)),
        },
        "leakage": {
            # cluster_spans_boundary is the load-bearing §8.2 homology guard (gated to
            # False in validate_report); order_spans_boundary is informational (distinct
            # clusters of one order on opposite sides is not homology leakage).
            "cluster_spans_boundary": bool(shared_clusters),
            "order_spans_boundary": bool(shared_orders),
            "n_shared_clusters": int(len(shared_clusters)),
            "n_shared_orders": int(len(shared_orders)),
            "positives_heldout_only": True,
        },
        "metrics": {
            "balanced_accuracy": bal_acc,
            "auroc": auroc,
            "auroc_ci_lower": ci_lo if math.isfinite(ci_lo) else None,
            "auroc_ci_upper": ci_hi if math.isfinite(ci_hi) else None,
            "n_bootstrap": int(n_bootstrap),
            "n_bootstrap_valid": int(n_valid),
            "test_pos": int(len(pos_test_idx)),
            "test_neg": int(n_neg_test),
        },
        "gate": {
            "verdict": verdict,
            "binding": False,
            "chance_level": CHANCE_LEVEL,
            "above_chance": verdict in ("PASS", "borderline"),
        },
    }


def _env_block() -> dict[str, Any]:
    """Best-effort runtime env fingerprint (never raises)."""
    block: dict[str, Any] = {
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    for name in ("numpy", "sklearn", "torch", "transformers", "pandas"):
        try:
            mod = __import__(name)
            block[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            block[name] = None
    return block


def run_probe(
    *,
    emb_cache: str | Path = DEFAULT_EMB_CACHE,
    out: str | Path = DEFAULT_OUT,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    decoys: str | Path = DEFAULT_DECOYS,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    c: float = DEFAULT_C,
    max_iter: int = DEFAULT_MAX_ITER,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    log=print,
) -> dict[str, Any]:
    """Stage B: load the embedding cache, fit the probe, write the JSON report."""
    import numpy as np  # lazy

    emb_cache = Path(emb_cache)
    # allow_pickle=False: the cache stores only numeric + fixed-width Unicode arrays, so
    # loading has no pickle-deserialization surface.
    data = np.load(emb_cache, allow_pickle=False)
    x_pos, x_neg = data["x_pos"], data["x_neg"]
    pos_order, pos_cluster = data["pos_order"], data["pos_cluster"]

    # Cache-recorded extraction metadata is the source of truth for what was ACTUALLY
    # embedded; reject a stale/mismatched cache rather than silently probing it.
    meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
    cache_dim = int(meta.get("emb_dim", x_pos.shape[1]))
    if cache_dim != EMB_DIM or int(x_pos.shape[1]) != EMB_DIM:
        raise ValueError(
            f"embedding cache dim mismatch: meta={cache_dim}, array={x_pos.shape[1]}, "
            f"expected {EMB_DIM} — re-run --stage extract"
        )
    extract_seed = int(meta.get("seed", seed))

    report = build_probe_report(
        x_pos,
        x_neg,
        pos_order,
        pos_cluster,
        seed=seed,
        test_size=test_size,
        c=c,
        max_iter=max_iter,
        n_bootstrap=n_bootstrap,
    )

    # Enrich with data-provenance / embedding / env / provenance blocks.
    report["data"] = {
        "positives": {
            "n": int(x_pos.shape[0]),
            "source": POSITIVE_SOURCE,
            "role": HELDOUT_ROLE,
            "n_orders": int(len(set(np.asarray(pos_order, dtype=object).tolist()))),
            "n_clusters": int(len(set(np.asarray(pos_cluster).tolist()))),
        },
        "background": {
            "n": int(x_neg.shape[0]),
            "pool": BACKGROUND_POOL,
            "gc_mean": float(np.mean(data["neg_gc"])) if "neg_gc" in data else None,
            "length_mean": float(np.mean(data["neg_length"])) if "neg_length" in data else None,
        },
    }
    # Embedding provenance is taken from the cache metadata (what was actually used).
    report["embedding"] = {
        "backbone": meta.get("backbone", REPO_ID),
        "revision": meta.get("revision", REVISION),
        "hidden_dim": EMB_DIM,
        "pooling": meta.get("pooling", "mean-over-positions"),
        "frozen": True,
        "extract_seed": extract_seed,
    }
    report["env"] = _env_block()

    inputs = {}
    for path in (split_table, decoys, emb_cache):
        try:
            inputs[str(path)] = provenance.sha256_file(path)
        except Exception:
            inputs[str(path)] = "unavailable"
    report["provenance"] = {
        "git_sha": provenance.git_sha(),
        "probe_seed": int(seed),
        "extract_seed": extract_seed,
        "extract_env_lock": _safe_env_lock(EXTRACT_ENV_LOCK),
        "probe_env_lock": _safe_env_lock(PROBE_ENV_LOCK),
        "inputs": inputs,
        "extract_inputs": meta.get("extract_inputs", {}),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    errors = validate_report(report)
    if errors:
        sys.stderr.write(json.dumps({"schema_errors": errors}, indent=2) + "\n")
        raise ValueError(f"report failed validation: {errors}")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    log(
        f"wrote report: {out}  verdict={report['gate']['verdict']}  "
        f"AUROC={report['metrics']['auroc']:.4f}  "
        f"bal_acc={report['metrics']['balanced_accuracy']:.4f}"
    )
    return report


def _safe_env_lock(lock_path: str) -> str:
    try:
        return provenance.env_lock_hash(lock_path)
    except Exception:
        return "unavailable"


def run_prefilter(
    *,
    split_table: str | Path = DEFAULT_SPLIT_TABLE,
    decoys: str | Path = DEFAULT_DECOYS,
    fasta_dir: str | Path = DEFAULT_FASTA_DIR,
    emb_cache: str | Path = DEFAULT_EMB_CACHE,
    out: str | Path = DEFAULT_OUT,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    c: float = DEFAULT_C,
    max_iter: int = DEFAULT_MAX_ITER,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    reuse_embeddings: bool = False,
    log=print,
) -> dict[str, Any]:
    """End-to-end pre-filter (Stage A then Stage B) — requires torch **and** sklearn.

    In the committed envs no single env carries both, so production runs use
    ``--stage extract`` (``tbox-ml-dna``) then ``--stage probe`` (``tbox-data``). This
    helper is the ``--stage all`` path for any env that provides both.
    """
    if not (reuse_embeddings and Path(emb_cache).exists()):
        run_extract(
            split_table=split_table,
            decoys=decoys,
            fasta_dir=fasta_dir,
            emb_cache=emb_cache,
            device=device,
            seed=seed,
            log=log,
        )
    return run_probe(
        emb_cache=emb_cache,
        out=out,
        split_table=split_table,
        decoys=decoys,
        seed=seed,
        test_size=test_size,
        c=c,
        max_iter=max_iter,
        n_bootstrap=n_bootstrap,
        log=log,
    )


# --------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("extract", "probe", "all"),
        default="all",
        help="extract (torch env) | probe (sklearn env) | all (needs both)",
    )
    p.add_argument("--split-table", default=DEFAULT_SPLIT_TABLE)
    p.add_argument("--decoys", default=DEFAULT_DECOYS)
    p.add_argument("--fasta-dir", default=DEFAULT_FASTA_DIR)
    p.add_argument("--emb-cache", default=DEFAULT_EMB_CACHE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--C", dest="c", type=float, default=DEFAULT_C)
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--reuse-embeddings", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "extract":
        run_extract(
            split_table=args.split_table,
            decoys=args.decoys,
            fasta_dir=args.fasta_dir,
            emb_cache=args.emb_cache,
            device=args.device,
            seed=args.seed,
        )
        return 0

    if args.stage == "probe":
        report = run_probe(
            emb_cache=args.emb_cache,
            out=args.out,
            split_table=args.split_table,
            decoys=args.decoys,
            seed=args.seed,
            test_size=args.test_size,
            c=args.c,
            max_iter=args.max_iter,
            n_bootstrap=args.n_bootstrap,
        )
    else:
        report = run_prefilter(
            split_table=args.split_table,
            decoys=args.decoys,
            fasta_dir=args.fasta_dir,
            emb_cache=args.emb_cache,
            out=args.out,
            device=args.device,
            seed=args.seed,
            test_size=args.test_size,
            c=args.c,
            max_iter=args.max_iter,
            n_bootstrap=args.n_bootstrap,
            reuse_embeddings=args.reuse_embeddings,
        )
    # Non-binding: always exit 0 on a valid measured report (the verdict is advisory).
    print(json.dumps({"gate": report["gate"], "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
