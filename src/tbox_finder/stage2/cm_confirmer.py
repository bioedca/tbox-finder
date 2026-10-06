"""P3-17a — the CM + learned-calibration Stage-2 confirmer, built as a pre-registered ablation.

PRD §6 permits two Stage-2 ablations / fallbacks: (i) RNA-FM (P3-17 / P3-18) and (ii) "a
covariance-model + learned-calibration confirmer (continuity with the existing pipeline)"
(ADR-0001:36). This module builds (ii) and grades it beside the shipped RiNALMo-giga
Stage-2 on the **same rows**, calibrated on the **same split** and graded by the **same
estimators** (ADR-0005 A14 says why the temperature stage itself cannot apply). It is an ablation.
PRD §10.2 pins that the CM stays an **orthogonal cross-validator, never the sole Stage-2**,
so nothing here can promote it to the shipped confirmer, and the report carries no gate.

The confirmer
-------------
1. **Score.** Every Stage-2 input RNA is searched with the two ADR-0005 D2 canonical models
   (``RF00230.cm`` class I, ``TBDB001.cm`` class II) by Infernal 1.1.5 ``cmsearch --toponly
   --max -T -1000``. The confirmer's raw score for a sequence is its **best hit bit score
   over both models**. ``--toponly`` because the Stage-2 input is already oriented RNA.
   ``--max`` because without it the HMM filters discard ~99 % of decoys before the CM stage,
   which would leave them with no score at all. Measured on a 200-row smoke, the filtered
   pipeline scored 1 of 100 decoys even at ``-T -1000``. Filter-off also makes the score
   independent of ``-Z`` and of how queries are sharded: a bit score is a per-sequence
   log-odds, and with no filters there is no ``Z``-dependent threshold left to change which
   sequences reach the CM. ``-Z`` is still pinned, so the E-value column is
   shard-invariant too.
2. **Learn the calibration (ADR-0005 A14).** Platt scaling, a logistic map ``z = a·s + b``
   from bit score to log-odds, fitted on the ``calib`` rung: the disjoint rows D11 fits
   RiNALMo's temperature on. It uses Platt's smoothed targets ``(N₊+1)/(N₊+2)`` and
   ``1/(N₋+2)`` [Platt 1999; Lin, Lin & Weng 2007, DOI:10.1007/s10994-007-5018-6 (accessed
   2026-10-06)], which keep the minimiser finite when the score separates the classes
   perfectly. On ``calib`` it does: the CM's positives are CM-derived (PRD §5). ``a > 0`` is
   required, so the posterior is monotone in bit score.
3. **No temperature stage.** As first built, Platt was fitted on the training rows and P3-07's
   ``temperature_scale`` was to fit ``T`` on ``calib``. It refused: the CM logit misclassifies
   no ``calib`` row, so the NLL has no minimiser. A14 pins the fix: for a model with no training
   phase, the learned calibration *is* the fit on the calibration split. The posterior
   ``σ(z)`` is recorded as ``cm_platt_posterior``, never D11's ``named_posterior``. It is then
   graded with D11's and D13's estimators, the identical functions behind RiNALMo's numbers.

What is compared
----------------
* **Calibration.** In-distribution ECE of ``cm_platt_posterior`` on the ``test`` rung and per-order
  leave-clade-out ECE on the 30 holdout orders, paired against RiNALMo's committed GATE-2
  report by P3-18's order-blocked :func:`swap_check.paired_difference`.
* **Precision.** The P3-16 benchmark's gated ``twin`` arm is replayed, and the CM confirmer
  re-ranks the identical Stage-1 candidate loci. AUPRC at the D7 100:1 prevalence (ADR-0005
  A13's statistic) and precision at P3-16's matched recall R* come from P3-16's own
  :func:`precision.arm_metrics`, with the same blocks, seed and replicate count.

⚠ **Circularity disclosure (PRD §5).** The positives are CM-derived. A CM scored against
truth built with CMs measures agreement with itself, so its discrimination on this
benchmark is a ceiling by construction, not evidence of generalization. RF00230 and TBDB001
were also built from seed sequences that are not order-held-out. The CM's leave-clade-out
row is therefore **not** a leave-clade-out test of the CM; only the calibrator is
order-held-out. The report states both and a clause refuses a report that drops them.

Envs: :func:`build_queries` and :func:`run_ablation` run in ``data`` (pandas / pyarrow);
:func:`search` runs in ``infernal``. Module top imports stdlib + numpy only, so all three
import in either env.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tbox_finder import infernal

SCHEMA_VERSION = "1"
STEP = "P3-17a"
GENERATED_BY = "src/tbox_finder/stage2/cm_confirmer.py"
RULE = "workflow/rules/stage2.smk :: cm_confirmer_ablation"
PRD = "§6 ablation (ii); §10.2; §12; §5"
ADR = "ADR-0001:36; ADR-0005 D2, D5, D7, D11, D13, A13"

#: The two ADR-0005 D2 canonical models, keyed by the name the score table uses.
CONFIRMER_CMS: tuple[tuple[str, Path], ...] = (
    ("RF00230", Path("data/external/refs/RF00230.cm")),
    ("TBDB001", Path("data/external/refs/TBDB001.cm")),
)

#: ``-T`` for the score search, in bits. Low enough that every query reports its best
#: window. A query with no hit at all is counted and refused by the report's completeness
#: clause; it is never silently given a floor.
SEARCH_THRESHOLD_BITS = -1000.0
#: ``-Z`` in Mb. It only affects the E-value column, which the confirmer does not read, but
#: pinning it keeps every shard's tblout comparable.
SEARCH_SPACE_MB = 1.0
DEFAULT_N_SHARDS = 22

#: The alphabet a Stage-2 input may carry: the RNA nucleotides plus IUPAC ambiguity codes.
#: A ``T`` means a DNA string reached the RNA confirmer, which is refused.
RNA_ALPHABET = frozenset("ACGUNRYKMSWBDHV")

QUERY_MANIFEST_SCHEMA = "1"


class ConfirmerError(RuntimeError):
    """Raised when an input or an intermediate violates the confirmer's contract."""


# ══════════════════════════════════════════════════════════════════════════════════════
# Queries
# ══════════════════════════════════════════════════════════════════════════════════════
def normalise_rna(sequence: str) -> str:
    """Upper-case one Stage-2 input and refuse anything that is not RNA.

    The confirmer scores **RNA sequence only** (PRD §6), the same string RiNALMo ingests.
    A ``T`` or a gap would mean the caller handed over DNA or an alignment row, so it
    raises rather than being transcribed here.
    """
    seq = str(sequence).strip().upper()
    if not seq:
        raise ConfirmerError("empty RNA sequence")
    bad = set(seq) - RNA_ALPHABET
    if bad:
        raise ConfirmerError(f"RNA sequence carries non-RNA characters {sorted(bad)!r}")
    return seq


def query_id(sequence: str) -> str:
    """Content address of one query: sha256 of its normalised RNA.

    Keying on the sequence, not on a row id, means identical inputs are scored once and
    always receive the same score whichever table they came from.
    """
    return hashlib.sha256(normalise_rna(sequence).encode("ascii")).hexdigest()


def collect_queries(sequences: Sequence[str]) -> dict[str, str]:
    """``{query_id: rna}`` over ``sequences``, deduplicated by content, sorted by id."""
    out: dict[str, str] = {}
    for sequence in sequences:
        seq = normalise_rna(sequence)
        out[hashlib.sha256(seq.encode("ascii")).hexdigest()] = seq
    return dict(sorted(out.items()))


def shard_of(qid: str, n_shards: int) -> int:
    """Deterministic shard for a query id (its leading 32 bits, mod ``n_shards``)."""
    if n_shards < 1:
        raise ConfirmerError(f"n_shards must be >= 1, got {n_shards}")
    return int(qid[:8], 16) % n_shards


def read_payload_sequences(path: str | Path) -> list[str]:
    """Every Stage-2 payload RNA in a P3-16 ``payloads`` file (both strands of every locus)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload["payloads"] if isinstance(payload, Mapping) else payload
    return [str(entry["rna_sequence"]) for entry in entries]


def read_dataset_sequences(path: str | Path) -> list[str]:
    """Every ``rna_sequence`` in the Stage-2 dataset (data env: needs pandas + pyarrow)."""
    import pandas as pd

    frame = pd.read_parquet(path, columns=["rna_sequence"])
    return [str(v) for v in frame["rna_sequence"]]


def write_query_shards(
    queries: Mapping[str, str],
    out_dir: str | Path,
    *,
    n_shards: int,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Write ``{query_id: rna}`` as ``n_shards`` FASTA files plus ``manifest.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards: dict[int, dict[str, str]] = {k: {} for k in range(n_shards)}
    for qid, seq in sorted(queries.items()):
        if qid != query_id(seq):
            raise ConfirmerError(f"query {qid!r} is not the content address of its sequence")
        shards[shard_of(qid, n_shards)][qid] = seq
    files = []
    for k in range(n_shards):
        if not shards[k]:
            raise ConfirmerError(f"shard {k} is empty; lower n_shards")
        files.append(infernal.write_fasta(shards[k], out / f"shard_{k:03d}.fa").name)
    manifest = {
        "schema_version": QUERY_MANIFEST_SCHEMA,
        "step": STEP,
        "n_queries": len(queries),
        "n_residues": sum(len(s) for s in queries.values()),
        "n_shards": n_shards,
        "shards": files,
        "query_ids_sha256": hashlib.sha256("\n".join(sorted(queries)).encode("ascii")).hexdigest(),
        "sources": dict(sources),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_queries(
    *,
    dataset: str | Path,
    payloads: str | Path,
    out_dir: str | Path,
    n_shards: int = DEFAULT_N_SHARDS,
) -> dict[str, Any]:
    """Every Stage-2 dataset row and every P3-16 payload, deduplicated, as query shards.

    Only the ``rna_sequence`` column is read: the confirmer sees the RNA and nothing else.
    """
    from tbox_finder import provenance as PROV

    queries = collect_queries([*read_dataset_sequences(dataset), *read_payload_sequences(payloads)])
    return write_query_shards(
        queries,
        out_dir,
        n_shards=n_shards,
        sources={
            "dataset": {"path": str(dataset), "sha256": PROV.sha256_file(dataset)},
            "payloads": {"path": str(payloads), "sha256": PROV.sha256_file(payloads)},
        },
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# Search (infernal env)
# ══════════════════════════════════════════════════════════════════════════════════════
def search(
    *,
    query_dir: str | Path,
    out_dir: str | Path,
    jobs: int,
    cms: Sequence[tuple[str, Path]] = CONFIRMER_CMS,
    timeout_s: float = 4 * 3600.0,
) -> dict[str, Any]:
    """Search every shard against every confirmer CM, ``jobs`` cmsearch processes at once.

    Each process is single-threaded (``--cpu 0``): Infernal's worker threads parallelise
    over a sequence block, and for a shard of short queries one worker did all the work
    in the smoke, so parallelism comes from the shards instead. A ``DONE`` marker is
    written only after every (model, shard) pair exited 0.
    """
    qdir, out = Path(query_dir), Path(out_dir)
    manifest = json.loads((qdir / "manifest.json").read_text(encoding="utf-8"))
    tasks = []
    for name, cm in cms:
        (out / name).mkdir(parents=True, exist_ok=True)
        for shard in manifest["shards"]:
            tasks.append((name, Path(cm), qdir / shard, out / name / f"{Path(shard).stem}.tblout"))

    def _one(task: tuple[str, Path, Path, Path]) -> tuple[str, str, int]:
        name, cm, fasta, tblout = task
        hits = infernal.run_cmsearch(
            cm,
            fasta,
            tblout,
            cut_ga=False,
            cpu=0,
            timeout_s=timeout_s,
            toponly=True,
            no_filters=True,
            score_threshold=SEARCH_THRESHOLD_BITS,
            search_space_mb=SEARCH_SPACE_MB,
        )
        return name, fasta.name, len(hits)

    counts: dict[str, dict[str, int]] = {name: {} for name, _ in cms}
    with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for name, shard, n_hits in pool.map(_one, tasks):
            counts[name][shard] = n_hits
            print(f"{name} {shard}: {n_hits} hits", flush=True)
    done = {
        "step": STEP,
        "query_manifest_sha256": hashlib.sha256((qdir / "manifest.json").read_bytes()).hexdigest(),
        "flags": search_flags(),
        "cms": {name: str(cm) for name, cm in cms},
        "hit_counts": counts,
    }
    (out / "DONE.json").write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    return done


def search_flags() -> list[str]:
    """The cmsearch options every confirmer search runs with, as recorded in the report."""
    return [
        "--noali",
        "--cpu",
        "0",
        "--toponly",
        "--max",
        "-T",
        repr(SEARCH_THRESHOLD_BITS),
        "-Z",
        repr(SEARCH_SPACE_MB),
    ]


def _read_fasta(path: Path) -> dict[str, str]:
    """``{name: sequence}`` from a FASTA :func:`infernal.write_fasta` wrote (one line each)."""
    out: dict[str, str] = {}
    name: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            if name in out:
                raise ConfirmerError(f"{path}: duplicate query {name!r}")
            out[name] = ""
        elif name is not None:
            out[name] += line.strip()
    return out


# ══════════════════════════════════════════════════════════════════════════════════════
# Scores
# ══════════════════════════════════════════════════════════════════════════════════════
def read_search(
    *,
    query_dir: str | Path,
    tblout_dir: str | Path,
    cms: Sequence[tuple[str, Path]] = CONFIRMER_CMS,
) -> dict[str, Any]:
    """Parse every shard's tblout into ``{cm: {query_id: best bit score}}``.

    Bound to the query set it searched: ``DONE.json`` must name this manifest's sha256 and
    the pinned flags, and every hit must belong to a query of the shard that reported it.
    A hit for an unknown id would mean the tblouts and the manifest are from different runs.
    """
    qdir, tdir = Path(query_dir), Path(tblout_dir)
    manifest_path = qdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done_path = tdir / "DONE.json"
    if not done_path.exists():
        raise ConfirmerError(f"{done_path} is missing: the search did not finish")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if done.get("query_manifest_sha256") != manifest_sha:
        raise ConfirmerError("the tblouts were searched from a different query manifest")
    if done.get("flags") != search_flags():
        raise ConfirmerError(f"the tblouts ran with {done.get('flags')!r}, not {search_flags()!r}")

    queries: dict[str, str] = {}
    by_shard: dict[str, set[str]] = {}
    for shard in manifest["shards"]:
        records = _read_fasta(qdir / shard)
        by_shard[shard] = set(records)
        queries.update(records)
    if len(queries) != int(manifest["n_queries"]):
        raise ConfirmerError(
            f"the shards hold {len(queries)} queries, the manifest says {manifest['n_queries']}"
        )
    best: dict[str, dict[str, float]] = {name: {} for name, _ in cms}
    for name, _ in cms:
        for shard in manifest["shards"]:
            path = tdir / name / f"{Path(shard).stem}.tblout"
            for hit in infernal.parse_tblout(path.read_text(encoding="utf-8")):
                if hit.target not in by_shard[shard]:
                    raise ConfirmerError(f"{path}: hit for {hit.target!r}, not a query of {shard}")
                previous = best[name].get(hit.target)
                best[name][hit.target] = hit.score if previous is None else max(previous, hit.score)
    return {"queries": queries, "bits": best, "manifest": manifest, "done": done}


def score_table(searched: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per query: its length, each model's best bit score (``None`` = no hit) and the max."""
    out: dict[str, dict[str, Any]] = {}
    for qid, seq in searched["queries"].items():
        per_cm = {name: bits.get(qid) for name, bits in searched["bits"].items()}
        present = [v for v in per_cm.values() if v is not None]
        out[qid] = {"length": len(seq), **per_cm, "best": max(present) if present else None}
    return out


# ══════════════════════════════════════════════════════════════════════════════════════
# The learned calibration
# ══════════════════════════════════════════════════════════════════════════════════════
PLATT_MAX_ITER = 100
#: Converged when every gradient component is at most this times the row count.
PLATT_GRAD_TOL = 1e-10
#: Smallest Armijo step before the line search gives up.
PLATT_MIN_STEP = 1e-10
#: Ridge on the Hessian, as Lin et al. (2007) add, so the Newton system stays solvable
#: when the posterior saturates.
PLATT_RIDGE = 1e-12


@dataclass(frozen=True)
class PlattFit:
    """``z = slope · bits + intercept``, the learned bit-score → log-odds map."""

    slope: float
    intercept: float
    n_positive: int
    n_negative: int
    target_positive: float
    target_negative: float
    n_iterations: int
    converged: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "platt_smoothed_targets",
            "slope": self.slope,
            "intercept": self.intercept,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "target_positive": self.target_positive,
            "target_negative": self.target_negative,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "citation": "Platt 1999; Lin, Lin & Weng 2007, DOI:10.1007/s10994-007-5018-6",
        }


def platt_targets(labels: Sequence[int]) -> tuple[np.ndarray, int, int, float, float]:
    """Platt's smoothed targets: ``(N₊+1)/(N₊+2)`` for positives, ``1/(N₋+2)`` for negatives."""
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    return np.where(y == 1, t_pos, t_neg), n_pos, n_neg, t_pos, t_neg


def _platt_loss(z: np.ndarray, t: np.ndarray) -> float:
    """Cross-entropy against soft targets, ``Σ log(1 + eᶻ) − t·z``, overflow-free."""
    return float(np.sum(np.logaddexp(0.0, z) - t * z))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def fit_platt(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    max_iter: int = PLATT_MAX_ITER,
    grad_tol: float = PLATT_GRAD_TOL,
) -> PlattFit:
    """Fit Platt scaling by damped Newton (Lin, Lin & Weng 2007, Algorithm 1).

    The score is standardised for the solve and the coefficients mapped back, so a bit-score
    range of hundreds does not condition the Hessian. The smoothed targets make the
    objective strictly convex with a finite minimiser even under perfect separation. A
    non-positive slope would make the posterior *fall* as the CM score rises, so it raises
    rather than shipping a confirmer that ranks backwards.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if s.ndim != 1 or s.size == 0 or s.shape != y.shape:
        raise ConfirmerError(f"need equal-length 1-D scores and labels, got {s.shape} / {y.shape}")
    if not np.all(np.isfinite(s)):
        raise ConfirmerError("a bit score is non-finite")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ConfirmerError("labels must be 0/1")
    t, n_pos, n_neg, t_pos, t_neg = platt_targets(y.astype(np.int64))
    if n_pos == 0 or n_neg == 0:
        raise ConfirmerError(f"need both classes to calibrate, got {n_pos} pos / {n_neg} neg")
    mu, sd = float(s.mean()), float(s.std())
    if not sd > 0.0:
        raise ConfirmerError("every bit score is identical; there is nothing to calibrate")
    x = (s - mu) / sd

    a, b = 0.0, math.log((n_pos + 1.0) / (n_neg + 1.0))
    loss = _platt_loss(a * x + b, t)
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        p = _sigmoid(a * x + b)
        r = p - t
        g = np.array([float(np.dot(r, x)), float(r.sum())])
        if float(np.max(np.abs(g))) <= grad_tol * x.size:
            converged = True
            break
        w = p * (1.0 - p)
        h = np.array(
            [[float(np.dot(w, x * x)), float(np.dot(w, x))], [float(np.dot(w, x)), float(w.sum())]]
        ) + PLATT_RIDGE * np.eye(2)
        d = -np.linalg.solve(h, g)
        gd = float(g @ d)
        step = 1.0
        accepted = False
        while step >= PLATT_MIN_STEP:
            na, nb = a + step * float(d[0]), b + step * float(d[1])
            new_loss = _platt_loss(na * x + nb, t)
            if new_loss <= loss + 1e-4 * step * gd:
                accepted = True
                break
            step /= 2.0
        if not accepted:
            break
        a, b, loss = na, nb, new_loss
    slope = a / sd
    intercept = b - a * mu / sd
    if not slope > 0.0:
        raise ConfirmerError(f"the learned slope is {slope!r}: the posterior would not be monotone")
    return PlattFit(
        slope=float(slope),
        intercept=float(intercept),
        n_positive=n_pos,
        n_negative=n_neg,
        target_positive=t_pos,
        target_negative=t_neg,
        n_iterations=iterations,
        converged=converged,
    )


def platt_logit(scores: Sequence[float], fit: PlattFit) -> list[float]:
    """The confirmer's log-odds for each bit score — the logit the P3-07 stack scales."""
    s = np.asarray(scores, dtype=np.float64)
    return [float(v) for v in fit.slope * s + fit.intercept]


# ══════════════════════════════════════════════════════════════════════════════════════
# Calibration grade (the shared P3-07 / P3-10 stack)
# ══════════════════════════════════════════════════════════════════════════════════════
ARM = "cm_confirmer"
#: The confirmer's graded posterior (ADR-0005 A14). Never D11's ``named_posterior``: no
#: temperature is fitted, because on ``calib`` none exists.
POSTERIOR_KEY = "cm_platt_posterior"
#: ``grade_ood_units`` divides the logit by the ``T`` it is handed. The confirmer's logit is
#: already Platt's calibrated log-odds, so it is handed the identity — no fit stands behind it,
#: and the report says so rather than recording a temperature.
IDENTITY_TEMPERATURE = 1.0


def platt_posterior(logits: Sequence[float]) -> list[float]:
    """``σ(z)`` through P3-07's own posterior function, at the identity temperature."""
    from tbox_finder.calib import recalibrate as R

    payload = R.calibrated_posterior(list(logits), temperature=IDENTITY_TEMPERATURE)
    return [float(v) for v in payload[R.NAMED_POSTERIOR_KEY]]


def grade_in_distribution_a14(
    *,
    scores: Mapping[str, Any],
    blocks_by_row: Mapping[str, str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """D11's in-distribution read of the Platt posterior, on the ``test`` rung.

    The same estimator calls, in the same order, as ``gate2.grade_in_distribution`` after
    its temperature fit: debiased equal-mass binned ECE, the plug-in beside it, the
    reliability bins, and the cluster-blocked bootstrap CI through ``gate2._blocks``. Only
    the posterior differs (ADR-0005 A14). There is no ``passes`` field: nothing is gated.
    """
    from tbox_finder import metrics as M
    from tbox_finder.calib import ece as ECE
    from tbox_finder.calib import gate2 as G2

    rungs = scores["rungs"]
    posterior = platt_posterior(scores["logits"])
    idx = [i for i, rung in enumerate(rungs) if rung == G2.GATE_RUNG]
    if not idx:
        raise ConfirmerError(f"no rows on the {G2.GATE_RUNG!r} rung")
    y = [int(scores["labels"][i]) for i in idx]
    p = [posterior[i] for i in idx]
    keys = [blocks_by_row[scores["row_ids"][i]] for i in idx]
    n_bins = G2.ECE_N_BINS
    ece = M.binned_ece(y, p, n_bins, debias=True)
    reliability = M.reliability_bins(y, p, n_bins)
    ci = M.block_bootstrap_ci(
        G2._blocks(list(zip(y, p, strict=True)), keys),
        lambda sample: M.binned_ece([a for a, _ in sample], [b for _, b in sample], n_bins),
        n_boot=n_boot,
        seed=seed,
    )
    n_pos = int(sum(y))
    return {
        "graded_rung": G2.GATE_RUNG,
        "graded_posterior_key": POSTERIOR_KEY,
        "graded_object": (
            "cm_platt_posterior: sigma(a*bits + b), Platt fitted on calib (ADR-0005 A14); "
            "NOT D11's named_posterior, and no temperature stage"
        ),
        "temperature_stage": "not_applied",
        "n": len(idx),
        "n_positive": n_pos,
        "n_negative": len(idx) - n_pos,
        "prevalence": n_pos / len(idx),
        "ece": ece,
        "ece_plugin": M.binned_ece(y, p, n_bins, debias=False),
        "ece_ci": ci,
        "ece_n_bins": int(n_bins),
        "ece_binning": "equal_mass",
        "ece_debiased": True,
        "estimator": ECE.IN_DISTRIBUTION_ESTIMATOR,
        "n_blocks": len(set(keys)),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "reliability": reliability,
        "gated": False,
    }


def grade_calibration(
    *,
    in_dist_scores: Mapping[str, Any],
    loo_scores: Mapping[str, Any],
    split_rows: Sequence[Mapping[str, Any]],
    dataset: str | Path,
    rinalmo_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Grade the Platt posterior with GATE-2's estimators, at RiNALMo's own settings.

    ``n_boot``, ``ood_n_boot`` and the bootstrap seed are read off RiNALMo's committed report
    rather than retyped, so the two arms are one estimator's output by construction.
    """
    from tbox_finder.calib import gate2 as G2
    from tbox_finder.calib import swap_check as SC
    from tbox_finder.stage2 import eval as E

    scoring = rinalmo_report["scoring"]
    seed = int(scoring["bootstrap_seed"])
    keys, _ = E.block_keys(split_rows)
    blocks_by_row = {row[G2._ROW_ID]: key for row, key in zip(split_rows, keys, strict=True)}
    holdout, census = G2.loo_holdout_rows(dataset)
    rows_by_id = {row[G2._ROW_ID]: row for row in holdout}

    gate = grade_in_distribution_a14(
        scores=in_dist_scores,
        blocks_by_row=blocks_by_row,
        n_boot=int(scoring["n_boot"]),
        seed=seed,
    )
    ood = G2.grade_ood_units(
        scores=loo_scores,
        rows_by_id=rows_by_id,
        temperature=IDENTITY_TEMPERATURE,
        n_boot=int(scoring["ood_n_boot"]),
        seed=seed,
    )
    rinalmo_values = SC._admissible_values(rinalmo_report)
    cm_values = SC._admissible_values({"ood": ood})
    paired = SC.paired_difference(rinalmo_values, cm_values)
    rin_ood = rinalmo_report["ood"]
    return {
        "posterior_key": POSTERIOR_KEY,
        "temperature_stage": (
            "not applied (ADR-0005 A14): on calib the CM logit misclassifies no row, so D11's "
            "temperature has no minimiser; grade_ood_units is handed the identity T = 1.0 on "
            "Platt's log-odds, which is not a fitted temperature"
        ),
        "in_distribution": {
            ARM: gate,
            "rinalmo": {
                "graded_posterior_key": rinalmo_report["gate"]["graded_posterior_key"],
                "ece": rinalmo_report["gate"]["ece"],
                "ece_ci": rinalmo_report["gate"]["ece_ci"],
                "temperature": rinalmo_report["gate"]["calibration"]["temperature"],
                "n": rinalmo_report["gate"]["n"],
            },
            "rinalmo_minus_cm_confirmer": float(rinalmo_report["gate"]["ece"]) - float(gate["ece"]),
            "note": (
                "both ECEs are D11's debiased 15-bin equal-mass estimator on the same test rows, "
                "each arm calibrated on calib (RiNALMo: T; CM: a two-parameter Platt map, A14); "
                "the two CIs are marginal, not paired"
            ),
        },
        "leave_clade_out": {
            ARM: ood,
            "rinalmo": {"macro_average": rin_ood["macro_average"]},
            "paired": {
                "definition": (
                    "per held-out order, RiNALMo OOD ECE minus CM-confirmer OOD ECE; positive = "
                    "the CM confirmer is better calibrated on that order. P3-18's order-blocked "
                    "bootstrap (swap_check.paired_difference), one draw of orders for both arms"
                ),
                **paired,
            },
            "ood_settings": {
                name: {ARM: ood.get(name), "rinalmo": rin_ood.get(name)}
                for name in SC._SHARED_OOD_SETTINGS
            },
            "census": census,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# The P3-16 benchmark, re-ranked by the confirmer
# ══════════════════════════════════════════════════════════════════════════════════════
#: The P3-16 locus / Stage-1 knobs. Values come from the Snakemake config (the same keys and
#: defaults ``integration.smk::two_stage_eval`` uses), and :func:`benchmark_items` binds them
#: by reproducing P3-16's committed item scores exactly: a wrong knob changes the candidate
#: set and refuses, so a retyped value cannot drift silently.
REPLAY_KNOBS = (
    "threshold_scope",
    "threshold",
    "min_span",
    "gap_merge",
    "min_distinct_elements",
    "flank",
    "min_order_margin",
)


def replay_benchmark(
    *,
    benchmark: str | Path,
    stage1: str | Path,
    stage2: str | Path,
    rinalmo_report: str | Path,
    operating_point: float,
    knobs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    """Re-run P3-16's gated arm from its replayed model outputs (no model, ~2 s)."""
    from tbox_finder.integration import two_stage as TS

    if sorted(knobs) != sorted(REPLAY_KNOBS):
        raise ConfirmerError(f"replay knobs {sorted(knobs)} != {sorted(REPLAY_KNOBS)}")
    temperature = TS.read_temperature(rinalmo_report)
    result = TS.run_two_stage(
        TS.read_contigs(benchmark),
        TS.read_stage1(stage1),
        TS.read_stage2(stage2),
        temperature=temperature,
        stage2_operating_point=operating_point,
        source_prior=None,
        target_prior=None,
        **knobs,
    )
    return [dict(row) for row in result.rows], temperature


def _item_key(item: Mapping[str, Any]) -> tuple:
    return (
        item["contig_id"],
        int(item["n_rows"]),
        float(item["stage1_only"]),
        float(item["two_stage"]),
    )


def benchmark_items(
    *,
    committed: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cm_posterior_by_row: Mapping[str, float],
    arm: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The benchmark's items scored by both re-rankers over the **same** candidate rows.

    First the replayed RiNALMo rows are folded with P3-16's own :func:`precision.item_scores`
    and must equal the committed items exactly (row count, Stage-1 score, two-stage score,
    every item). Only then is each row's Stage-2 posterior swapped for the CM confirmer's
    and folded again, so the two systems differ in the re-ranker and nothing else.
    """
    from tbox_finder.integration import precision as P

    reference = P.arm_items(committed, arm)
    replayed = P.item_scores(reference, rows)
    want = {item["contig_id"]: _item_key(item) for item in reference}
    got = {item["contig_id"]: _item_key(item) for item in replayed}
    mismatched = sorted(k for k in want if want[k] != got.get(k))
    if mismatched or set(got) != set(want):
        raise ConfirmerError(
            f"the replay does not reproduce P3-16's committed {arm} items "
            f"({len(mismatched)} differ, e.g. {mismatched[:3]}); a knob or an input changed"
        )
    missing = sorted({row["payload_key"] for row in rows} - set(cm_posterior_by_row))
    if missing:
        raise ConfirmerError(f"{len(missing)} candidate rows have no CM-confirmer posterior")
    cm_rows = [
        {**row, "stage2_named_posterior": cm_posterior_by_row[row["payload_key"]]} for row in rows
    ]
    cm_items = {item["contig_id"]: item for item in P.item_scores(reference, cm_rows)}
    items = [
        {
            "contig_id": item["contig_id"],
            "label": item["label"],
            "pool": item["pool"],
            "block": item["block"],
            "seen_by": item["seen_by"],
            "n_rows": item["n_rows"],
            "rinalmo": item["two_stage"],
            ARM: cm_items[item["contig_id"]]["two_stage"],
        }
        for item in replayed
    ]
    binding = {
        "arm": arm,
        "n_items": len(items),
        "n_candidate_rows": len(rows),
        "n_items_mismatched_vs_committed": len(mismatched),
    }
    return items, binding


#: ``precision.arm_metrics`` names its two systems ``two_stage`` and ``stage1_only``. The
#: confirmer reuses it unchanged (same R*, blocks, seed, replicate count and estimators) by
#: passing RiNALMo as ``two_stage`` and the CM confirmer as ``stage1_only``; every key is
#: then renamed, so each ``*gain*`` field reads **RiNALMo minus CM confirmer**.
_RELABEL = (("two_stage", "rinalmo"), ("stage1_only", ARM))


def _relabel(node: Any) -> Any:
    if isinstance(node, Mapping):
        out = {}
        for key, value in node.items():
            new = str(key)
            for old, repl in _RELABEL:
                new = new.replace(old, repl)
            out[new] = _relabel(value)
        return out
    if isinstance(node, list):
        return [_relabel(v) for v in node]
    return node


def benchmark_metrics(
    items: Sequence[Mapping[str, Any]], *, operating_point: float, n_boot: int, seed: int
) -> dict[str, Any]:
    """AUPRC at D7's 100:1 + precision at P3-16's R*, RiNALMo vs CM confirmer, blocked CIs."""
    from tbox_finder import power as PW
    from tbox_finder.integration import precision as P

    scored = [
        {
            "contig_id": item["contig_id"],
            "label": int(item["label"]),
            "pool": item["pool"],
            "block": item["block"],
            "seen_by": item["seen_by"],
            "n_rows": int(item["n_rows"]),
            "two_stage": float(item["rinalmo"]),
            "stage1_only": float(item[ARM]),
        }
        for item in items
    ]
    metrics = P.arm_metrics(
        scored,
        P.GATED_ARM,
        stage2_operating_point=operating_point,
        decoy_prevalence=PW.DECOY_PREVALENCE,
        prevalence_sweep=P.PREVALENCE_SWEEP,
        recall_grid=P.RECALL_GRID,
        n_boot=n_boot,
        seed=seed,
    )
    # Not a gate: the ablation grades nothing (PRD §10.2), so P3-16's verdict fields go.
    metrics.pop("passes", None)
    metrics.pop("gated", None)
    return _relabel(metrics)


# ══════════════════════════════════════════════════════════════════════════════════════
# The report
# ══════════════════════════════════════════════════════════════════════════════════════
#: The clause set. ``is_science`` is their AND, and :func:`validate_report` re-derives each
#: from the body rather than trusting the recorded booleans.
CLAUSES = (
    "every_query_scored",
    "calibrator_monotone",
    "calibrator_converged",
    "calibrator_fit_disjoint_from_every_graded_row",
    "calibrator_fit_on_the_whole_calib_rung",
    "graded_object_is_the_a14_posterior",
    "in_distribution_population_is_rinalmos",
    "leave_clade_out_population_is_rinalmos",
    "ood_settings_are_rinalmos",
    "replay_reproduces_p3_16_items",
    "rinalmo_benchmark_reproduces_p3_16_report",
    "every_benchmark_item_scored",
    "circularity_disclosed",
    "never_the_shipped_stage2",
)

#: Disclosures a report must carry; the ``circularity_disclosed`` clause checks every id.
REQUIRED_DISCLOSURES = (
    "cm_derived_positives",
    "cm_not_order_held_out",
    "orthogonal_cross_validator_only",
    "score_is_not_d2_detection",
    "gated_arm_only",
    "a14_two_parameter_calibrator",
)


def disclosures() -> list[dict[str, str]]:
    """The fixed caveats every confirmer report carries (PRD §5, §10.2; ADR-0005 D2)."""
    return [
        {
            "id": "cm_derived_positives",
            "text": (
                "PRD §5: the positives are CM-derived (TBDB / Rfam), so a CM scored against them "
                "measures agreement with the models that built the truth. Its discrimination "
                "here is a ceiling by construction, not evidence of generalization, and a "
                "RiNALMo-minus-CM gap below zero does not mean the CM finds what RiNALMo misses."
            ),
        },
        {
            "id": "cm_not_order_held_out",
            "text": (
                "RF00230.cm and TBDB001.cm were built from seed alignments that are not "
                "order-held-out. On the leave-clade-out rows only the CALIBRATOR is held out "
                "(fit on the calib rung, which shares no row and no order with the 30 holdout "
                "orders); the scorer is not. The CM's leave-clade-out ECE is therefore not a "
                "leave-clade-out test of the CM."
            ),
        },
        {
            "id": "orthogonal_cross_validator_only",
            "text": (
                "PRD §10.2: the CM stays an orthogonal cross-validator and is never the sole "
                "Stage-2. This is the pre-registered ablation (ii) of PRD §6. It carries no "
                "gate and cannot swap the shipped backbone."
            ),
        },
        {
            "id": "score_is_not_d2_detection",
            "text": (
                "The confirmer score is the best bit score from cmsearch --toponly --max -T -1000 "
                "(filters off, every query scored). It is not ADR-0005 D2's canonical detection "
                "operating point (RF00230 GA 93 bits, default filtered pipeline), which stays "
                "GATE-1's baseline at P4."
            ),
        },
        {
            "id": "gated_arm_only",
            "text": (
                "The precision comparison replays P3-16's gated twin arm only. The production "
                "arm's Stage-1 candidates are not re-ranked here."
            ),
        },
        {
            "id": "a14_two_parameter_calibrator",
            "text": (
                "ADR-0005 A14: on calib the CM logit misclassifies no row (a 26.6-bit gap at the "
                "time of signing), so D11's temperature has no minimiser. The CM arm is "
                "calibrated by a two-parameter Platt map (scale and offset) fitted on the same "
                "calib rows D11 fits RiNALMo's one-parameter T on. Its posterior is "
                "cm_platt_posterior, not D11's named_posterior, and no T = 1 is substituted."
            ),
        },
    ]


def _population_matches(ours: Mapping[str, Any], theirs: Mapping[str, Any], fields) -> bool:
    return all(ours.get(f) == theirs.get(f) for f in fields)


def derive_clauses(report: Mapping[str, Any]) -> dict[str, bool]:
    """Every clause, recomputed from the report body."""
    conf = report["confirmer"]
    cal = report["calibration"]
    prec = report["precision"]
    fit = conf["calibrator"]
    graded = cal["in_distribution"][ARM]
    settings = cal["leave_clade_out"]["ood_settings"]
    ids = {d.get("id") for d in report.get("disclosures", []) if str(d.get("text") or "").strip()}
    rin_ref = prec["p3_16_reference"]
    rin_now = prec["metrics"]["prevalence"][prec["metrics"]["gated_prevalence_key"]]["auprc"]
    return {
        "every_query_scored": conf["n_queries_unscored"] == 0 and conf["n_queries"] > 0,
        "calibrator_monotone": float(fit["slope"]) > 0.0,
        "calibrator_converged": bool(fit["converged"]),
        "calibrator_fit_disjoint_from_every_graded_row": conf["n_fit_rows_also_graded"] == 0
        and conf["n_fit_rows"] > 0,
        "calibrator_fit_on_the_whole_calib_rung": fit.get("fitted_on") == "calib"
        and conf["n_fit_rows"] == conf["n_calib_rows_in_sidecar"],
        "graded_object_is_the_a14_posterior": graded.get("graded_posterior_key") == POSTERIOR_KEY
        and cal.get("posterior_key") == POSTERIOR_KEY
        and graded.get("temperature_stage") == "not_applied"
        and graded.get("gated") is False,
        "in_distribution_population_is_rinalmos": bool(conf["population_match"]["in_distribution"]),
        "leave_clade_out_population_is_rinalmos": bool(conf["population_match"]["leave_clade_out"]),
        "ood_settings_are_rinalmos": bool(settings)
        and all(v.get(ARM) == v.get("rinalmo") for v in settings.values()),
        "replay_reproduces_p3_16_items": prec["binding"]["n_items_mismatched_vs_committed"] == 0,
        "rinalmo_benchmark_reproduces_p3_16_report": rin_now.get("rinalmo")
        == rin_ref["auprc_at_pinned_prevalence"]
        and prec["metrics"]["operating_point"]["target_recall"] == rin_ref["target_recall"],
        "every_benchmark_item_scored": prec["binding"]["n_items"] == prec["n_items_expected"],
        "circularity_disclosed": set(REQUIRED_DISCLOSURES) <= ids,
        "never_the_shipped_stage2": report.get("gated") is False
        and report.get("role") == "ablation",
    }


def build_report(
    *,
    confirmer: Mapping[str, Any],
    calibration: Mapping[str, Any],
    precision: Mapping[str, Any],
    provenance: Mapping[str, Any],
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    from datetime import UTC, datetime

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "generated_by": GENERATED_BY,
        "rule": RULE,
        "prd": PRD,
        "adr": ADR,
        "role": "ablation",
        "gated": False,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "confirmer": dict(confirmer),
        "calibration": dict(calibration),
        "precision": dict(precision),
        "disclosures": disclosures(),
        "provenance": dict(provenance),
    }
    clauses = derive_clauses(report)
    report["clauses"] = clauses
    report["is_science"] = all(clauses.values())
    return report


def _canonical(node: Any) -> Any:
    return json.loads(json.dumps(node, sort_keys=True))


def validate_report(
    report: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
    require_all_inputs: bool = True,
    rederive_precision: bool = True,
) -> list[str]:
    """Problems with a confirmer report; ``[]`` means it is internally and externally sound.

    * the header, the clause set, and ``is_science`` as their AND;
    * every clause re-derived from the body;
    * every recorded input re-hashed. ``require_all_inputs=False`` skips inputs absent from
      this checkout, which in CI are the DVC dataset and the local-only P3-16 replay files.
      The git-committed inputs are always checked;
    * the precision block re-derived from the committed items file (``rederive_precision``).
    """
    problems: list[str] = []
    for key, want in (
        ("schema_version", SCHEMA_VERSION),
        ("step", STEP),
        ("generated_by", GENERATED_BY),
    ):
        if report.get(key) != want:
            problems.append(f"{key} is {report.get(key)!r}, expected {want!r}")
    clauses = report.get("clauses")
    if not isinstance(clauses, Mapping) or tuple(sorted(clauses)) != tuple(sorted(CLAUSES)):
        problems.append(f"clause set {sorted(clauses or {})} != {sorted(CLAUSES)}")
        return problems
    if report.get("is_science") is not all(clauses.values()):
        problems.append("is_science is not the AND of the clauses")
    try:
        derived = derive_clauses(report)
    except (KeyError, TypeError, ValueError) as exc:
        return [*problems, f"clauses cannot be re-derived: {exc!r}"]
    for name in CLAUSES:
        if derived[name] != clauses[name]:
            problems.append(
                f"clause {name} records {clauses[name]} but the body gives {derived[name]}"
            )

    from tbox_finder import provenance as PROV

    root = Path(repo_root)
    inputs = (report.get("provenance") or {}).get("inputs") or {}
    if not inputs:
        problems.append("provenance records no inputs")
    for rel, digest in inputs.items():
        path = root / rel
        if not path.exists():
            if require_all_inputs:
                problems.append(f"input {rel} is absent")
            continue
        if PROV.sha256_file(path) != digest:
            problems.append(f"input {rel} no longer hashes to the recorded {digest[:12]}…")

    if rederive_precision:
        prec = report["precision"]
        items_path = root / prec["items"]
        if not items_path.exists():
            problems.append(f"items file {prec['items']} is absent")
        else:
            payload = json.loads(items_path.read_text(encoding="utf-8"))
            again = benchmark_metrics(
                payload["items"],
                operating_point=float(prec["operating_point"]),
                n_boot=int(prec["n_boot"]),
                seed=int(prec["seed"]),
            )
            if _canonical(again) != _canonical(prec["metrics"]):
                problems.append("the precision block does not re-derive from its items file")
    return problems


# ══════════════════════════════════════════════════════════════════════════════════════
# The run (data env)
# ══════════════════════════════════════════════════════════════════════════════════════
def _write_json(path: str | Path, payload: Any, *, indent: int | None = 2) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _separation(bits: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """How far the raw score is from separating the classes: the overlap counts PRD §5 predicts."""
    pos = sorted(b for b, y in zip(bits, labels, strict=True) if y == 1)
    neg = sorted(b for b, y in zip(bits, labels, strict=True) if y == 0)
    if not pos or not neg:
        return {"n_positive": len(pos), "n_negative": len(neg)}
    max_neg, min_pos = neg[-1], pos[0]
    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "min_positive_bits": min_pos,
        "max_negative_bits": max_neg,
        "n_positives_at_or_below_max_negative": sum(1 for b in pos if b <= max_neg),
        "n_negatives_at_or_above_min_positive": sum(1 for b in neg if b >= min_pos),
        "perfectly_separated": min_pos > max_neg,
    }


def run_ablation(
    *,
    dataset: str | Path,
    query_dir: str | Path,
    tblout_dir: str | Path,
    rinalmo_scores: str | Path,
    rinalmo_loo_scores: str | Path,
    rinalmo_report: str | Path,
    benchmark: str | Path,
    stage1: str | Path,
    stage2: str | Path,
    payloads: str | Path,
    precision_items: str | Path,
    precision_report: str | Path,
    knobs: Mapping[str, Any],
    out_report: str | Path,
    out_scores: str | Path,
    out_loo_scores: str | Path,
    out_items: str | Path,
    out_confirmer_dir: str | Path,
    env_lock: str | Path,
    infernal_env_lock: str | Path,
    generated_at_utc: str | None = None,
) -> tuple[dict[str, Any], list[str], Path]:
    """Score → learn the calibration → grade both ways → write every artifact + the report."""
    import pandas as pd

    from tbox_finder import provenance as PROV
    from tbox_finder.calib import gate2 as G2
    from tbox_finder.calib import recalibrate as R
    from tbox_finder.calib import swap_check as SC
    from tbox_finder.integration import precision as P

    rel = G2._recorded_path
    searched = read_search(query_dir=query_dir, tblout_dir=tblout_dir)
    table = score_table(searched)
    unscored = sorted(q for q, r in table.items() if r["best"] is None)
    if unscored:
        raise ConfirmerError(
            f"{len(unscored)} queries drew no hit even at -T {SEARCH_THRESHOLD_BITS}; a floor "
            "score would be invented, so the run stops"
        )

    # ── the Stage-2 rows → their scores ────────────────────────────────────────────────
    split_rows, _ = G2._read_split_table(dataset, with_sequences=True)
    bits_by_row: dict[str, float] = {}
    label_by_row: dict[str, int] = {}
    for row in split_rows:
        qid = query_id(row[G2._SEQUENCE])
        if qid not in table:
            raise ConfirmerError(f"row {row[G2._ROW_ID]!r} was never searched")
        bits_by_row[row[G2._ROW_ID]] = float(table[qid]["best"])
        label_by_row[row[G2._ROW_ID]] = int(bool(row[G2._LABEL]))

    dataset_sha = PROV.sha256_file(dataset)
    rin_in = json.loads(Path(rinalmo_scores).read_text(encoding="utf-8"))
    rin_loo = json.loads(Path(rinalmo_loo_scores).read_text(encoding="utf-8"))
    for name, side in (("in-distribution", rin_in), ("leave-clade-out", rin_loo)):
        if side.get("dataset_sha256") != dataset_sha:
            raise ConfirmerError(f"the RiNALMo {name} sidecar was scored on a different dataset")
        bad = [
            r
            for r, y in zip(side["row_ids"], side["labels"], strict=True)
            if label_by_row.get(r) != int(y)
        ]
        if bad:
            raise ConfirmerError(
                f"the RiNALMo {name} sidecar disagrees with the dataset on {len(bad)} labels"
            )

    # ── the learned calibration: Platt on the D11 calib rung (ADR-0005 A14) ─────────────
    fit_ids = [
        r
        for r, rung in zip(rin_in["row_ids"], rin_in["rungs"], strict=True)
        if rung == R.CALIB_RUNG
    ]
    graded_ids = {
        r
        for r, rung in zip(rin_in["row_ids"], rin_in["rungs"], strict=True)
        if rung == G2.GATE_RUNG
    } | set(rin_loo["row_ids"])
    fit_bits = [bits_by_row[r] for r in fit_ids]
    fit_labels = [label_by_row[r] for r in fit_ids]
    fit = fit_platt(fit_bits, fit_labels)
    train_ids = [row[G2._ROW_ID] for row in split_rows if G2.training_admission(row)]

    cm_records = {
        name: {"path": str(cm), "sha256": PROV.sha256_file(cm)} for name, cm in CONFIRMER_CMS
    }
    load = {
        "confirmer": "cm_bit_score_platt",
        "cms": cm_records,
        "search_flags": search_flags(),
        "calibrator": fit.as_dict(),
    }

    def _sidecar(
        side: Mapping[str, Any], fields: Sequence[str], source: str | Path
    ) -> dict[str, Any]:
        bits = [bits_by_row[r] for r in side["row_ids"]]
        return {
            **{f: side[f] for f in fields},
            "step": STEP,
            "generated_by": GENERATED_BY,
            "population_from": rel(source),
            "arms": {ARM: {"logits": platt_logit(bits, fit), "bit_scores": bits, "load": load}},
        }

    in_side = _sidecar(rin_in, SC._IN_DIST_POPULATION_FIELDS, rinalmo_scores)
    loo_side = _sidecar(rin_loo, SC._LOO_POPULATION_FIELDS, rinalmo_loo_scores)
    _write_json(out_scores, in_side, indent=None)
    _write_json(out_loo_scores, loo_side, indent=None)

    # ── calibration grade, read back from the files just written ────────────────────────
    rin_report = json.loads(Path(rinalmo_report).read_text(encoding="utf-8"))
    calibration = grade_calibration(
        in_dist_scores=G2.load_scores(out_scores, ARM),
        loo_scores=G2.load_scores(out_loo_scores, ARM),
        split_rows=split_rows,
        dataset=dataset,
        rinalmo_report=rin_report,
    )

    # ── the P3-16 benchmark ─────────────────────────────────────────────────────────────
    p3_16 = json.loads(Path(precision_report).read_text(encoding="utf-8"))
    twin = p3_16["arms"][P.GATED_ARM]
    if float(knobs["threshold"]) != float(p3_16["sources"]["stage1_threshold"]):
        raise ConfirmerError(
            f"replay threshold {knobs['threshold']!r} is not P3-16's recorded Stage-1 threshold "
            f"{p3_16['sources']['stage1_threshold']!r}"
        )
    operating_point = float(twin["operating_point"]["stage2_operating_point"])
    n_boot = int(twin["gain_ci"]["n_boot"])
    seed = int(p3_16["provenance"]["seed"])
    rows, t_rin = replay_benchmark(
        benchmark=benchmark,
        stage1=stage1,
        stage2=stage2,
        rinalmo_report=rinalmo_report,
        operating_point=operating_point,
        knobs=knobs,
    )
    payload_rna = {
        str(e["row_id"]): str(e["rna_sequence"])
        for e in json.loads(Path(payloads).read_text(encoding="utf-8"))["payloads"]
    }
    row_bits: dict[str, float] = {}
    for row in rows:
        key = row["payload_key"]
        rna = payload_rna.get(key)
        if rna is None:
            raise ConfirmerError(f"candidate row {key!r} has no payload RNA")
        # The RNA scored here must be the RNA this row handed to Stage 2, byte for byte.
        if hashlib.sha256(rna.encode("ascii")).hexdigest() != row["rna_sha256"]:
            raise ConfirmerError(f"candidate row {key!r}: payload RNA is not the row's handoff")
        qid = query_id(rna)
        if qid not in table:
            raise ConfirmerError(f"candidate row {key!r} was never searched")
        row_bits[key] = float(table[qid]["best"])
    ids = sorted(row_bits)
    posterior = platt_posterior(platt_logit([row_bits[i] for i in ids], fit))
    cm_posterior = dict(zip(ids, posterior, strict=True))
    committed_items = json.loads(Path(precision_items).read_text(encoding="utf-8"))
    items, binding = benchmark_items(
        committed=committed_items, rows=rows, cm_posterior_by_row=cm_posterior, arm=P.GATED_ARM
    )
    _write_json(
        out_items,
        {
            "schema_version": SCHEMA_VERSION,
            "step": STEP,
            "generated_by": GENERATED_BY,
            "arm": P.GATED_ARM,
            "systems": ["rinalmo", ARM],
            "source": {"items": rel(precision_items), "benchmark": rel(benchmark)},
            "items": items,
        },
        indent=None,
    )
    metrics = benchmark_metrics(items, operating_point=operating_point, n_boot=n_boot, seed=seed)
    gated_key = twin["gated_prevalence_key"]

    # ── the DVC-tracked confirmer: per-query scores + the fitted calibrator ─────────────
    cdir = Path(out_confirmer_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in CONFIRMER_CMS]
    frame = pd.DataFrame(
        {
            "query_id": list(table),
            "length": [r["length"] for r in table.values()],
            **{f"{n}_bits": [r[n] for r in table.values()] for n in names},
            "best_bits": [r["best"] for r in table.values()],
        }
    )
    scores_path = cdir / "query_scores.parquet"
    frame.to_parquet(scores_path, index=False)
    calibrator_path = _write_json(
        cdir / "calibrator.json",
        {
            "step": STEP,
            "calibrator": fit.as_dict(),
            "fitted_on": R.CALIB_RUNG,
            "posterior_key": POSTERIOR_KEY,
            "temperature_stage": "not_applied (ADR-0005 A14)",
            "cms": cm_records,
            "search_flags": search_flags(),
        },
    )
    seed_record = int(rin_report["scoring"]["bootstrap_seed"])
    dir_prov = PROV.build_provenance(
        rule="workflow/rules/stage2.smk :: cm_confirmer_ablation",
        script=GENERATED_BY,
        seed=seed_record,
        inputs=[dataset, *(cm for _, cm in CONFIRMER_CMS)],
        outputs=[scores_path, calibrator_path],
        env_lock=env_lock,
        adr=ADR,
        extra={"infernal_env_lock_sha256": PROV.env_lock_hash(infernal_env_lock)},
    )
    dir_prov["inputs"] = {rel(k): v for k, v in dir_prov["inputs"].items()}
    dir_prov["outputs"] = {rel(k): v for k, v in dir_prov["outputs"].items()}
    _write_json(cdir / "provenance.json", dir_prov)

    confirmer = {
        "cms": cm_records,
        "search_flags": search_flags(),
        "score_definition": (
            "per query, the best hit bit score over RF00230 and TBDB001 from cmsearch "
            "--toponly --max -T -1000 (Infernal 1.1.5)"
        ),
        "n_queries": len(table),
        "n_queries_unscored": len(unscored),
        "n_residues": int(searched["manifest"]["n_residues"]),
        "query_manifest_sha256": searched["done"]["query_manifest_sha256"],
        "calibrator": {**fit.as_dict(), "fitted_on": R.CALIB_RUNG},
        "n_fit_rows": len(fit_ids),
        "n_calib_rows_in_sidecar": sum(1 for rung in rin_in["rungs"] if rung == R.CALIB_RUNG),
        "n_fit_rows_also_graded": len(set(fit_ids) & graded_ids),
        "n_fit_rows_in_training_admission": len(set(fit_ids) & set(train_ids)),
        "separation": {
            "calib_rows": _separation(fit_bits, fit_labels),
            "training_admission_rows": _separation(
                [bits_by_row[r] for r in train_ids], [label_by_row[r] for r in train_ids]
            ),
            "misclassified_at_zero_on_calib": sum(
                1
                for z, y in zip(platt_logit(fit_bits, fit), fit_labels, strict=True)
                if (z > 0) != (y == 1)
            ),
        },
        "population_match": {
            "in_distribution": _population_matches(in_side, rin_in, SC._IN_DIST_POPULATION_FIELDS),
            "leave_clade_out": _population_matches(loo_side, rin_loo, SC._LOO_POPULATION_FIELDS),
        },
        "artifact_dir": rel(cdir),
    }
    precision = {
        "arm": P.GATED_ARM,
        "items": rel(out_items),
        "operating_point": operating_point,
        "n_boot": n_boot,
        "seed": seed,
        "n_items_expected": len(committed_items["items"]),
        "binding": binding,
        "posteriors": {
            "rinalmo": f"named_posterior at T = {t_rin!r} (P3-16's two-stage replay)",
            ARM: f"{POSTERIOR_KEY} (Platt on calib, no temperature; ADR-0005 A14)",
        },
        "p3_16_reference": {
            "auprc_at_pinned_prevalence": twin["prevalence"][gated_key]["auprc"]["two_stage"],
            "target_recall": twin["operating_point"]["target_recall"],
        },
        "reading": (
            "P3-16's precision.arm_metrics with RiNALMo in its two_stage slot and the CM "
            "confirmer in its stage1_only slot, keys renamed after: every *gain* field is "
            "RiNALMo minus CM confirmer (pp). R* is RiNALMo's recall at P3-16's operating "
            "point. ADR-0005 A13's statistic is AUPRC at the D7 100:1 prevalence. Nothing "
            "here is gated."
        ),
        "metrics": metrics,
    }
    prov = PROV.build_provenance(
        rule=RULE,
        script=GENERATED_BY,
        seed=seed,
        inputs=[
            dataset,
            rinalmo_scores,
            rinalmo_loo_scores,
            rinalmo_report,
            *(cm for _, cm in CONFIRMER_CMS),
            precision_items,
            precision_report,
            benchmark,
            stage1,
            stage2,
            payloads,
            out_scores,
            out_loo_scores,
            out_items,
            Path(query_dir) / "manifest.json",
            Path(tblout_dir) / "DONE.json",
            cdir / "provenance.json",
        ],
        env_lock=env_lock,
        adr=ADR,
        extra={
            "declared_outputs": [rel(out_report)],
            "infernal_env_lock_sha256": PROV.env_lock_hash(infernal_env_lock),
            "replay_knobs": dict(knobs),
        },
    )
    prov["inputs"] = {rel(k): v for k, v in prov["inputs"].items()}
    report = build_report(
        confirmer=confirmer,
        calibration=calibration,
        precision=precision,
        provenance=prov,
        generated_at_utc=generated_at_utc,
    )
    problems = validate_report(report, repo_root=".")
    out = Path(out_report)
    if problems or not report["is_science"]:
        out = out.with_suffix(".invalid.json")
    _write_json(out, report)
    return report, problems, out


# ══════════════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tbox_finder.stage2.cm_confirmer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("queries", help="write the deduplicated query shards (data env)")
    q.add_argument("--dataset", required=True)
    q.add_argument("--payloads", required=True)
    q.add_argument("--out-dir", required=True)
    q.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS)

    s = sub.add_parser("search", help="cmsearch every shard against both CMs (infernal env)")
    s.add_argument("--query-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--jobs", type=int, required=True)

    a = sub.add_parser("ablation", help="calibrate, grade and write the report (data env)")
    for flag in (
        "--dataset",
        "--query-dir",
        "--tblout-dir",
        "--rinalmo-scores",
        "--rinalmo-loo-scores",
        "--rinalmo-report",
        "--benchmark",
        "--stage1",
        "--stage2",
        "--payloads",
        "--precision-items",
        "--precision-report",
        "--out-report",
        "--out-scores",
        "--out-loo-scores",
        "--out-items",
        "--out-confirmer-dir",
        "--env-lock",
        "--infernal-env-lock",
        "--threshold-scope",
    ):
        a.add_argument(flag, required=True)
    a.add_argument("--threshold", type=float, required=True)
    for flag in (
        "--min-span",
        "--gap-merge",
        "--min-distinct-elements",
        "--flank",
        "--min-order-margin",
    ):
        a.add_argument(flag, type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "queries":
        manifest = build_queries(
            dataset=args.dataset,
            payloads=args.payloads,
            out_dir=args.out_dir,
            n_shards=args.n_shards,
        )
        print(f"{manifest['n_queries']} queries, {manifest['n_residues']} nt")
        return 0
    if args.cmd == "search":
        search(query_dir=args.query_dir, out_dir=args.out_dir, jobs=args.jobs)
        return 0
    if args.cmd == "ablation":
        report, problems, out = run_ablation(
            dataset=args.dataset,
            query_dir=args.query_dir,
            tblout_dir=args.tblout_dir,
            rinalmo_scores=args.rinalmo_scores,
            rinalmo_loo_scores=args.rinalmo_loo_scores,
            rinalmo_report=args.rinalmo_report,
            benchmark=args.benchmark,
            stage1=args.stage1,
            stage2=args.stage2,
            payloads=args.payloads,
            precision_items=args.precision_items,
            precision_report=args.precision_report,
            knobs={knob: getattr(args, knob) for knob in REPLAY_KNOBS},
            out_report=args.out_report,
            out_scores=args.out_scores,
            out_loo_scores=args.out_loo_scores,
            out_items=args.out_items,
            out_confirmer_dir=args.out_confirmer_dir,
            env_lock=args.env_lock,
            infernal_env_lock=args.infernal_env_lock,
        )
        print(f"wrote {out}")
        for name, ok in sorted(report["clauses"].items()):
            if not ok:
                print(f"  FALSE clause: {name}")
        for problem in problems:
            print(f"  REPORT PROBLEM: {problem}")
        return 3 if problems or not report["is_science"] else 0
    raise AssertionError(args.cmd)  # pragma: no cover - argparse enforces the choices


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
