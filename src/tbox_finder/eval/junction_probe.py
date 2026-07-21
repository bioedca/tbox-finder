"""The two-tier junction gate for the §9.1 decoy embedding (P2-10d′-b, ADR-0005 A7 pin 7).

R2 embeds a decoy inside a real 1024-nt background window, which puts two splice
junctions into **negatives only**. If a junction were readable, it would be a perfectly
class-correlated cue — exactly the failure `workflow/rules/data.smk` already ruled against
for a 0th-order synthetic splice, and strictly worse, because there it would have applied
to every window rather than to one class. So the construction is gated twice:

**Tier 1 — pre-flight, composition (this module, LOCAL, blocks the submit-ack).** A k-mer
logistic regression is asked to separate the matched background→background control from
plain unspliced windows. The reference is not a chosen number but a **measured null**: the
same classifier separating two *disjoint sets of unspliced windows*, whose true AUROC is
0.5 by construction. The junction arm must not deviate from chance by more than the null
does, allowing for the AUROC's own sampling error.

**Tier 2 — post-run, model-side.** The trained checkpoint scores the same three arms and
must satisfy :func:`model_side_pass` — the decoy must beat its own junction null. Its rule
lives here, pinned now, so the number the run produces is graded against a criterion
written before the run rather than after it.

**Why the must-fire clause is not optional.** "The junction is invisible" and "the probe
sees nothing" produce identical numbers ([[control-matchedness-must-be-asserted]]: R-scape
reports "no power" exactly as it reports "no signal"). A featuriser that returned constant
features would pass the junction clause perfectly. So :func:`junction_clauses` also
requires the probe to separate the **decoy** arm from the control arm — a separation that
must exist by construction, since those two differ by an entire decoy — and a run where
that clause is FALSE is reported as *underpowered*, never as *clean*.

sklearn is imported lazily so the geometry tier stays importable in a bare environment.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tbox_finder.data.embedding import EmbeddedWindow

SCHEMA_VERSION = "1.0"
STEP = "P2-10d'-b"

#: Alphabet the k-mer featuriser indexes. Non-ACGT k-mers are dropped rather than folded
#: into an N bucket: an N run is a property of the *host*, shared by every arm, so giving
#: it a feature would add noise to all three arms equally and dilute nothing but power.
ALPHABET = "ACGT"

#: The pre-flight k. k=4 (256 features) is the smoke-sized tier; the full measurement
#: reported in ADR-0005 A7 also ran k=6 (4096 features), which is the more sensitive of
#: the two and moved the junction arm no further from chance.
DEFAULT_K = 4

#: How many standard errors of slack the junction arm gets over the measured null. Three
#: is the conventional two-sided ~99.7 % envelope; it is applied to the **null-relative**
#: deviation, so it is slack around a measured quantity, not an absolute pass bar.
NULL_BAND_SIGMA = 3.0


class JunctionProbeError(ValueError):
    """Raised when the probe cannot produce an honest measurement."""


def auroc_null_stderr(n_pos: int, n_neg: int) -> float:
    """Standard error of AUROC under H0 (Hanley & McNeil's null variance).

    ``sqrt((n1 + n2 + 1) / (12 * n1 * n2))`` — the distribution-free null used to size the
    band the junction arm must sit inside. Returns ``inf`` for an empty arm so an
    unmeasurable comparison widens the band to "anything", and the clause that consumes it
    fails on the emptiness guard rather than passing on a divide-by-zero.
    """
    if n_pos <= 0 or n_neg <= 0:
        return math.inf
    return math.sqrt((n_pos + n_neg + 1.0) / (12.0 * n_pos * n_neg))


def kmer_frequencies(sequences: Sequence[str], k: int = DEFAULT_K) -> np.ndarray:
    """Frequency-normalised k-mer counts, one row per sequence."""
    if k <= 0:
        raise JunctionProbeError(f"k must be positive; got {k}")
    if not sequences:
        raise JunctionProbeError("no sequences to featurise")
    index = {"".join(p): i for i, p in enumerate(itertools.product(ALPHABET, repeat=k))}
    out = np.zeros((len(sequences), len(index)), dtype=np.float32)
    for row, seq in enumerate(sequences):
        s = str(seq).upper()
        for i in range(len(s) - k + 1):
            j = index.get(s[i : i + k])
            if j is not None:
                out[row, j] += 1.0
    totals = out.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return out / totals


def cv_auroc(
    positive: Sequence[str], negative: Sequence[str], *, k: int = DEFAULT_K, seed: int
) -> float:
    """Cross-validated AUROC separating two sets of sequences by k-mer composition.

    Cross-validated, not in-sample: with 4096 features and a few thousand rows an
    in-sample fit separates anything. **The two arms must not share host windows** — a
    first version of this measurement built all arms on one host set, and because each
    spliced window is derived from its own unspliced counterpart, a classifier that
    memorised a host scored its paired counterpart in the opposite direction and every
    junction AUROC came back *below* chance (0.25–0.45). That is a pairing artifact, not a
    signal; :func:`junction_measurement` allocates disjoint hosts per arm.
    """
    # BEFORE the lazy sklearn import, not after: the module promises to stay importable
    # in a bare environment, and an import-ordered guard turns a named refusal into a
    # ModuleNotFoundError for every caller without sklearn.
    if len(positive) < 2 or len(negative) < 2:
        raise JunctionProbeError(
            f"each arm needs at least 2 sequences; got {len(positive)} / {len(negative)}"
        )

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = kmer_frequencies(list(positive) + list(negative), k=k)
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    n_splits = int(min(5, len(positive), len(negative)))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs"))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    scores = cross_val_predict(model, features, labels, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, scores))


def deviation(auroc: float) -> float:
    """Distance from chance. AUROC is symmetric about 0.5 — 0.2 is as separable as 0.8."""
    return abs(float(auroc) - 0.5)


def arms_are_matched(
    decoy: Sequence[EmbeddedWindow],
    control: Sequence[EmbeddedWindow],
    host_sequences: Mapping[str, str],
) -> tuple[bool, dict[str, Any]]:
    """Re-derive the matchedness contract from the windows themselves.

    Not a restatement of what the generator promised: this reads ``sequence`` and checks
    that each control/decoy pair agrees on host, phase and insert length, that every base
    **outside** the replaced interval is identical, and that at least one base **inside**
    it differs.

    ``host_sequences`` is **required**, and the clause it feeds is the one that stops the
    gate from being a tautology. Pair-wise matchedness alone cannot tell a genuine
    background→background splice from a control that is simply *the unspliced host*: both
    agree with the decoy arm on the flank and differ from it inside the interval. But an
    unspliced control carries **no junction**, so "junction control vs plain" would be two
    samples of the same distribution and the AUROC would sit at the null *by construction*
    — a clean-looking verdict that measured nothing. The first version of this function
    omitted the clause and the sabotage test in
    `tests/unit/test_embedding_generator.py` caught it.
    """
    detail: dict[str, Any] = {
        "n_pairs": min(len(decoy), len(control)),
        "n_length_mismatch": 0,
        "n_geometry_mismatch": 0,
        "n_flank_mismatch": 0,
        "n_identical_inside": 0,
        "n_control_unspliced": 0,
        "n_host_unresolved": 0,
    }
    if not decoy or not control or len(decoy) != len(control):
        detail["reason"] = "arms are empty or unequal in length"
        return False, detail
    for d, c in zip(decoy, control, strict=True):
        if d.host_id != c.host_id or d.phase != c.phase or d.insert_len != c.insert_len:
            detail["n_geometry_mismatch"] += 1
            continue
        if len(d.sequence) != len(c.sequence):
            detail["n_length_mismatch"] += 1
            continue
        lo, hi = d.junctions
        if d.sequence[:lo] != c.sequence[:lo] or d.sequence[hi:] != c.sequence[hi:]:
            detail["n_flank_mismatch"] += 1
        if d.sequence[lo:hi] == c.sequence[lo:hi]:
            detail["n_identical_inside"] += 1
        host = host_sequences.get(c.host_id)
        if host is None:
            # An unresolvable host is not a pass: the splice claim becomes uncheckable,
            # and "could not check" must never read the same as "checked and clean".
            detail["n_host_unresolved"] += 1
        elif str(host)[lo:hi] == c.sequence[lo:hi]:
            detail["n_control_unspliced"] += 1
    ok = (
        detail["n_geometry_mismatch"] == 0
        and detail["n_length_mismatch"] == 0
        and detail["n_flank_mismatch"] == 0
        and detail["n_identical_inside"] == 0
        and detail["n_control_unspliced"] == 0
        and detail["n_host_unresolved"] == 0
    )
    return bool(ok), detail


def junction_measurement(
    *,
    plain: Sequence[str],
    null_reference: Sequence[str],
    control: Sequence[EmbeddedWindow],
    matched_control: Sequence[EmbeddedWindow],
    decoy: Sequence[EmbeddedWindow],
    host_sequences: Mapping[str, str],
    k: int = DEFAULT_K,
    seed: int,
) -> dict[str, Any]:
    """Measure the three AUROCs the pre-flight gate is built from.

    **Two roles the control arm cannot play at once, so it is passed twice.**

    ``matched_control`` is the *true paired* control: same host, same phase, same insert
    length as its ``decoy`` counterpart. That pairing is what makes it a control at all,
    and :func:`arms_are_matched` verifies it.

    ``control`` is the arm the **AUROC** is measured on, and it must be built on hosts
    **disjoint** from ``decoy``. Those requirements are in direct conflict: a
    cross-validated classifier fed a host window in both classes learns the host and then
    scores its counterpart in the opposite direction — which is why the first version of
    this measurement, with all arms on one host set, returned junction AUROCs *below*
    chance (0.25–0.45). Splitting the roles keeps the matchedness assertion exact while
    leaving the distributional comparison unpaired; the two control sets are drawn from
    the same pool with the same phase and insert-length distributions, so they differ only
    in which windows they landed on.

    ``plain`` and ``null_reference`` are two further **disjoint** sets of unspliced mined
    windows; separating them is the measured null.
    """
    if not plain or not null_reference:
        raise JunctionProbeError("the null needs two non-empty sets of unspliced windows")
    overlap = set(plain) & set(null_reference)
    if overlap:
        raise JunctionProbeError(
            f"{len(overlap)} sequence(s) appear in both null arms; a shared window makes "
            "the null optimistic and the band it defines too wide"
        )
    control_seqs = [w.sequence for w in control]
    decoy_seqs = [w.sequence for w in decoy]
    shared_hosts = {w.host_id for w in control} & {w.host_id for w in decoy}
    if shared_hosts:
        raise JunctionProbeError(
            f"{len(shared_hosts)} host window(s) appear in both the decoy and the AUROC "
            "control arm. Paired arms invert the measurement (a classifier that memorises "
            "a host scores its counterpart in the opposite direction). Pass the paired "
            "control as `matched_control` and build `control` on disjoint hosts."
        )
    matched, matched_detail = arms_are_matched(decoy, matched_control, host_sequences)

    null_auroc = cv_auroc(plain, null_reference, k=k, seed=seed)
    junction_auroc = cv_auroc(control_seqs, plain, k=k, seed=seed)
    content_auroc = cv_auroc(decoy_seqs, control_seqs, k=k, seed=seed)

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "k": int(k),
        "seed": int(seed),
        "n_plain": len(plain),
        "n_null_reference": len(null_reference),
        "n_control": len(control_seqs),
        "n_matched_control": len(matched_control),
        "n_decoy": len(decoy_seqs),
        "auroc_null_unspliced_vs_unspliced": null_auroc,
        "auroc_junction_control_vs_plain": junction_auroc,
        "auroc_decoy_vs_junction_control": content_auroc,
        "null_stderr": auroc_null_stderr(len(plain), len(null_reference)),
        "junction_stderr": auroc_null_stderr(len(control_seqs), len(plain)),
        "content_stderr": auroc_null_stderr(len(decoy_seqs), len(control_seqs)),
        "arms_matched": bool(matched),
        "arms_matched_detail": matched_detail,
        "null_band_sigma": float(NULL_BAND_SIGMA),
    }


def junction_clauses(report: dict[str, Any] | None) -> dict[str, bool]:
    """Re-derive the pre-flight gate's clauses from a measurement.

    Total: a missing, malformed or partial report yields every clause FALSE rather than
    raising, so a gate reading a truncated artifact fails closed
    ([[gate-clauses-need-re-derivation]]). Every clause is guarded on the *presence* of
    the evidence it summarises, so none can be vacuously TRUE when the measurement is the
    thing that is missing ([[clauses-must-guard-emptiness]]).
    """
    clauses = {
        "junction_measurement_present": False,
        "arms_are_matched": False,
        "probe_can_discriminate": False,
        "junction_within_null_band": False,
    }
    if not isinstance(report, dict):
        return clauses

    def _num(key: str) -> float | None:
        v = report.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return float(v)

    null = _num("auroc_null_unspliced_vs_unspliced")
    junction = _num("auroc_junction_control_vs_plain")
    content = _num("auroc_decoy_vs_junction_control")
    null_se = _num("null_stderr")
    counts = [report.get(c) for c in ("n_plain", "n_null_reference", "n_control", "n_decoy")]
    counts_ok = all(isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in counts)
    if None in (null, junction, content, null_se) or not counts_ok:
        return clauses
    assert null is not None and junction is not None
    assert content is not None and null_se is not None
    if not math.isfinite(null_se):
        return clauses

    clauses["junction_measurement_present"] = True
    clauses["arms_are_matched"] = report.get("arms_matched") is True

    raw_sigma = report.get("null_band_sigma", NULL_BAND_SIGMA)
    sigma = (
        float(raw_sigma)
        if isinstance(raw_sigma, (int, float)) and not isinstance(raw_sigma, bool)
        else NULL_BAND_SIGMA
    )
    band = deviation(null) + sigma * null_se

    # MUST FIRE. A probe that cannot separate a decoy-bearing window from its own matched
    # control has no power, and its verdict on the junction is not evidence of absence.
    clauses["probe_can_discriminate"] = deviation(content) > band
    clauses["junction_within_null_band"] = deviation(junction) <= band
    return clauses


def junction_symmetry_clauses(
    summary: dict[str, Any] | None, realized: dict[str, Any] | None = None
) -> dict[str, bool]:
    """Re-derive the junction-SYMMETRY gate (ADR-0005 A7 pin 7, amended 2026-07-21).

    This replaces the original `junction_within_null_band` pass/fail, and it is worth
    being exact about why, because dropping a clause that was failing is otherwise
    indistinguishable from loosening a bar. The original clause asked whether the splice
    junction was *invisible* to a k-mer probe. Measured on the real substrate it is not:
    across four independent draws the k=6 junction arm read 0.5217 / 0.5264 / 0.5281 /
    0.5328 against nulls of 0.4947–0.5042 — small, but reproducible and one-directional.
    A splice junction is a chimera of two genomes and a 6-mer model can see it.

    So the cue is no longer *bounded by measurement*; it is *removed by construction*.
    Positives receive the same background→background splice, at the same rate and from the
    same insert-length distribution, which makes "this window is chimeric" carry no class
    information. What must then be true is arithmetic, not statistics:

    ``rate_is_derived``
        the positive splice rate equals ``n_chimeric_negatives / n_negative_records``
        exactly — re-derived here from the recorded counts, never read from a config echo.
    ``labels_unchanged``
        the splice moved zero per-nucleotide targets. It overwrites only positions already
        labelled background and already real, and it is equal-length, so any non-zero count
        means the interval selector is wrong.
    ``realized_rates_agree``
        over an actually-emitted epoch stream, the positive and negative chimeric rates
        agree within three standard errors of their difference. This is the clause that
        catches an augmentation configured but never firing.
    ``control_fires_when_disabled``
        with the augmentation off, ``realized_rates_agree`` must be FALSE. Without it a
        broken measurement that reports both rates as zero would pass every clause above
        ([[control-matchedness-must-be-asserted]]).
    """
    clauses = {
        "rate_is_derived": False,
        "labels_unchanged": False,
        "realized_rates_agree": False,
        "control_fires_when_disabled": False,
    }
    if not isinstance(summary, dict):
        return clauses

    def _count(key: str, src: dict[str, Any]) -> int | None:
        v = src.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return None
        return v

    n_chim = _count("n_chimeric_negatives", summary)
    n_neg = _count("n_negative_records", summary)
    rate = summary.get("rate")
    if (
        n_chim is None
        or n_neg is None
        or n_neg == 0
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
    ):
        return clauses
    clauses["rate_is_derived"] = abs(float(rate) - n_chim / n_neg) < 1e-12

    moved = _count("n_labels_changed", summary)
    n_checked = _count("n_windows_label_checked", summary)
    clauses["labels_unchanged"] = moved == 0 and n_checked is not None and n_checked > 0

    if not isinstance(realized, dict):
        return clauses
    on = realized.get("augmented")
    off = realized.get("control_disabled")
    clauses["realized_rates_agree"] = _rates_agree(on)
    # MUST FIRE in the opposite direction: the disabled arm has to be separable, or the
    # measurement has no power to detect a non-firing augmentation.
    disabled_agrees = _rates_agree(off)
    clauses["control_fires_when_disabled"] = disabled_agrees is False and off is not None
    return clauses


def _rates_agree(arm: Any) -> bool:
    """True when a measured arm's positive and negative chimeric rates are compatible."""
    if not isinstance(arm, dict):
        return False
    try:
        n_pos = int(arm["n_positive_draws"])
        n_neg = int(arm["n_negative_draws"])
        c_pos = int(arm["n_positive_chimeric"])
        c_neg = int(arm["n_negative_chimeric"])
    except (KeyError, TypeError, ValueError):
        return False
    if n_pos <= 0 or n_neg <= 0:
        return False
    p, q = c_pos / n_pos, c_neg / n_neg
    se = math.sqrt(p * (1 - p) / n_pos + q * (1 - q) / n_neg)
    if se == 0.0:
        return p == q
    return abs(p - q) <= 3.0 * se


def model_side_pass(*, auroc_decoy_vs_control: float, auroc_control_vs_plain: float) -> bool:
    """Tier-2 rule: the decoy must beat its own junction null (ADR-0005 A7 pin 7).

    Scored on the trained checkpoint, over the same three arms. PASS iff the trained model
    separates decoy-bearing windows from their matched junction controls **more** than it
    separates those controls from plain unspliced windows — i.e. it is reading decoy
    content rather than the splice.

    Threshold-free on purpose. Pinning an absolute bar would invite the P2-07 failure in
    reverse: there, a perturbation was certified against a fixed number while a
    length-matched random excision broke detection *more often* than the real element, so
    the measured effect was smaller than its own null and the clause still passed
    ([[matched-control-before-certifying]]). Here the null travels with the measurement.
    """
    return deviation(float(auroc_decoy_vs_control)) > deviation(float(auroc_control_vs_plain))
