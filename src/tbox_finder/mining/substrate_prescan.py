"""substrate_prescan.py — the off-catalogue cmsearch masking gate (ADR-0005 A8, P2-10c'-d).

Why this exists
---------------
The union-prior mask (:mod:`tbox_finder.masking`, ADR-0005 A6/A7) is **fail-open off
catalogue**: :func:`~tbox_finder.masking.is_masked` /
:func:`~tbox_finder.masking.matches_interval_exactly` return ``False`` for any accession
not keyed in ``union_prior.parquet`` [masking.py:194-198, :207-214, :249-252], and every
GTDB/RefSeq accession is off-catalogue. Drawing an independent-genome window as *unmasked
background* would label a real, uncatalogued T-box as a negative and directionally train the
production scanner to reject the flagship discovery class — the exact harm ADR-0005 D14's
spare rule exists to prevent [ADR-0005:129-131]. This module is the catalogue-independent
gate D14 presumes but does not deliver: a **model-independent, SHARD-AWARE** ``cmsearch``
pre-scan of the fetched substrate, whose own versioned report carries same-artifact must-fire
clauses so ``overall_pass`` is ``False`` **by construction** on a dead scan, a dead shard, a
degraded CM, or a silently-shrunken fetch.

What the number is (and is not)
-------------------------------
The reported ``substrate_removal_rate`` is a **removal/detection YIELD on the production
substrate**, NOT a residual. Every detected window is *refused*, so what escapes the gate *at
the detector's sensitivity* is ~0 by construction — but ``cmsearch`` misses ~27.6 % of
catalogued T-boxes at ``--cut_ga`` [reports/p2/tier2n_probe.json:measured_confound_baselines]
and misses Tier-2N loci **by definition** [tier2n_probe.json:n_natural=0], so ``removal_rate``
can only *understate* the true contamination-window fraction. A **low or zero** production
removal rate is EXPECTED and ADMISSIBLE (few CM-detectable canonical leaders — the common
case) and is **never** a dead-scan flag: detector liveness is guaranteed only by the per-shard
spike/null power controls and the per-invocation CM-identity check. The residual RISK is
one-sided **lower**-bounded, never bounded above; the retained D14 Tier-2N halt is the
directional backstop, unmeasured above. This module asserts **no** contamination value and
pins **no** numeric threshold — the recall-favouring detection cutoff is a *rule* here, its
value frozen at the P2/P6 phase gate [ADR-0005:41 D3 precedent], so ``score_threshold`` is a
keyword-required parameter with **no default** (§10.3).

Compute model
-------------
There is no per-window fast path and, at ρ-scale, no single global invocation: the scan is a
SLURM array (``slurm/p2/substrate_prescan.sbatch``). Each shard runs **one** batched
``cmsearch --tblout`` over a uniquely-named window FASTA that folds in that shard's **own**
spike/null control arms, then :func:`build_shard_segment` reduces the one tblout to a segment.
:func:`reduce` aggregates the per-shard segments into ``reports/p2/substrate_prescan.json``.
The ``cmsearch`` call itself is the single caller in :mod:`tbox_finder.infernal` (reused, not
forked); everything in this module's report/clause path is pure stdlib so it certifies on the
bare-CI tier without ``infernal`` on PATH.

Pure stdlib + :mod:`tbox_finder.infernal` + :mod:`tbox_finder.provenance`.
PRD §9.1/§9.2; ADR-0005 A8 (amends D14); ADR-0006 D7/D2; ADR-0002 A11.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tbox_finder import infernal, provenance

# ═════════════════════════════════════════════════════════════════════════════
# Pinned constants — frozen in code, NO CLI/config override (ADR-0002 A1 precedent
# [ADR-0005:222]), so no committed report can silently contradict the pin.
# ═════════════════════════════════════════════════════════════════════════════
#: On-disk report schema. Bump on a non-additive change.
SCHEMA_VERSION = "1.0"

#: The REQUIRED-clause-set version. A new gate clause bumps this; an older report then
#: re-validates FALSE unless every current key re-derives present (the A7 schema-versioning
#: precedent [ADR-0005:436], [[new-gate-clause-invalidates-old-reports]]). There are no
#: committed substrate_prescan reports yet, so v1 is authoritative and complete.
CLAUSE_SCHEMA_VERSION = 1

STEP = "P2-10c'-d"

#: SHA-256 of the pinned ``RF00230.cm`` this gate scans against — the CM-identity anchor
#: (clause vii), re-derived per invocation. Sourced from
#: ``reports/p2/tier2n_probe.json:cm_sha256`` and matched by ``sha256sum data/external/refs/
#: RF00230.cm``. A truncated/swapped/wrong-path CM changes the recorded sha256 → clause FALSE,
#: independently of co-invocation (clause vi).
RF00230_CM_SHA256 = "082616efe0616479dc9b9f565b5bcab7648d7ea1d629430940319f247af52cea"

#: Pinned tiling geometry — the FASTA-independent expected-window basis (pin 4/5(i)). Matches
#: ``data/processed/pilot/rho_pilot_report.json:geometry``. The per-shard ``expected`` window
#: count is read from the frozen fetch/geometry manifest (produced by the canonical tiling
#: code), NOT re-counted from the FASTA the scan reads — so a FASTA-write truncation moves the
#: *consumed* side while ``expected`` stays put and clause (i) turns FALSE.
WINDOW_NT = 1024
STRIDE_NT = 512

#: Operational power floors + strict separation margin — control-admissibility parameters, NOT
#: the (unpinned) cmsearch detection threshold. Loosening these is not a threshold change; they
#: are the analogue of :data:`tbox_finder.infer.rho_pilot.MIN_GLOBAL_MAX_P_ELEM`. ``MARGIN`` is
#: on the *difference* of removal rates, so it is robust to the operating threshold: at any
#: cutoff a live detector removes strictly more spike than matched null. ``0.0`` with strict
#: ``>`` is the weakest defensible separation (a copied/dead control gives exactly ``0``).
MIN_SCAN_N = 1
MIN_SPIKE_N = 20
MIN_NULL_N = 20
MARGIN = 0.0

#: The enumerated, versioned REQUIRED clause set. ``overall_pass = all(clauses[k] for k in
#: REQUIRED_CLAUSES)`` — never ``all()`` over whatever keys happen to be present. A MISSING key
#: is a HARD FAIL (:func:`validate_report`), never silently dropped ([[clauses-must-guard-
#: emptiness]], [[new-gate-clause-invalidates-old-reports]]).
REQUIRED_CLAUSES: tuple[str, ...] = (
    "production_denominator_consistent",  # (i)   loaded==expected, per shard + aggregate
    "detector_live_every_shard",  # (ii)  spike removal > 0 in every shard
    "power_arms_min_n",  # (iii) matched spike/null arms, min-N + matchedness
    "spike_null_separation",  # (iv)  removal(spike) − removal(null) > MARGIN, strict
    "hit_window_mapping_total",  # (v)   unique naming + every hit joins a window
    "shard_completeness_coinvocation",  # (vi)  shard count + SHARD_OK + one-tblout + id-equality
    "cm_identity",  # (vii) loaded CM sha256 == pinned, every shard
    "genome_completeness",  # (viii) scanned genomes == git-frozen selection, no shrink
)

ARMS: tuple[str, ...] = ("production", "spike", "null")

DEFAULT_CM = infernal.RF00230_CM
DEFAULT_REPORT = Path("reports/p2/substrate_prescan.json")
DEFAULT_PROVENANCE = Path("reports/p2/substrate_prescan.provenance.json")
DEFAULT_SELECTION_MANIFEST = Path("data/processed/pilot/pilot_genomes_v0.parquet")

#: The lower-bound disclosure carried on every reported value (pin 3).
LOWER_BOUND_DISCLOSURE = (
    "substrate_removal_rate is a removal/detection YIELD, not a residual: it lower-bounds the "
    "true removable-contamination fraction (cmsearch misses ~27.6% of catalogued T-boxes and "
    "misses Tier-2N by definition), so a low value is NOT a clean-pool claim. Residual Tier-2N "
    "contamination is UNMEASURED above; the D14 per-round recall-drop halt is the directional "
    "backstop, not a sufficiency claim."
)


class SubstratePrescanError(ValueError):
    """Raised on malformed scan evidence, a mis-shaped segment, or a non-certifying report."""


# ═════════════════════════════════════════════════════════════════════════════
# Window model + small pure helpers (composition/length matchedness, hit join)
# ═════════════════════════════════════════════════════════════════════════════
def base_composition(seq: str) -> dict[str, int]:
    """Base counts ``{A,C,G,T,other}`` of an upper-cased sequence — the matchedness fingerprint.

    ``other`` folds every non-ACGT symbol (N, IUPAC ambiguity) so a shuffle that preserves the
    ACGT multiset fingerprints identically to its source, and any base substitution shows up.
    """
    counts = {"A": 0, "C": 0, "G": 0, "T": 0, "other": 0}
    for ch in seq.upper():
        counts[ch if ch in "ACGT" else "other"] += 1
    return counts


def _length_multiset_sha(lengths: Sequence[int]) -> str:
    """Order-independent SHA-256 over a window-length multiset (matchedness leg iii)."""
    payload = json.dumps(sorted(int(x) for x in lengths))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tblout_identity(tblout_text: str) -> str:
    """The identity of ONE ``--tblout`` — SHA-256 of its DATA rows (comment lines excluded).

    Derived from the tblout's own content, never a builder-stamped string (pin 5(vi)), so a
    shard that split production and control into separate ``cmsearch`` calls cannot forge a
    shared id. The ``cmsearch`` header comments carry the input path, command line, and a
    timestamp — excluding them makes the id reproducible (path/date-independent) while still
    proving all three arms were parsed from the same real hit table; rows are sorted so any
    thread-order nondeterminism in ``cmsearch`` cannot perturb it.
    """
    rows = sorted(
        ln.strip() for ln in tblout_text.splitlines() if ln.strip() and not ln.startswith("#")
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def arm_metadata(windows: Mapping[str, str]) -> dict[str, Any]:
    """Reduce ``{name: seq}`` for one arm to its recorded, re-derivable metadata.

    Names are validated as single whitespace-free tokens here too (the
    :func:`tbox_finder.infernal.write_fasta` invariant), because a name with a space would be
    truncated by ``cmsearch`` at the first token and silently collapse the hit→window join
    (pin 4/5(v), the deflated-rate harm).
    """
    total = {"A": 0, "C": 0, "G": 0, "T": 0, "other": 0}
    lengths: list[int] = []
    for name, seq in windows.items():
        if not name or any(c.isspace() for c in name):
            raise SubstratePrescanError(
                f"window name must be a single whitespace-free token: {name!r}"
            )
        lengths.append(len(seq))
        for base, n in base_composition(seq).items():
            total[base] += n
    return {
        "n_windows": len(windows),
        "total_bp": sum(lengths),
        "length_multiset_sha": _length_multiset_sha(lengths),
        "composition": total,
    }


def arms_matched(spike: Mapping[str, Any], null: Mapping[str, Any]) -> bool:
    """Whether the spike and null arms differ ONLY in the embedded locus (clause iii matchedness).

    Equal window count, equal length multiset (so ``cmsearch`` sees the same amount of
    sequence), and identical base composition (a composition-preserving shuffle of the spike
    window is the canonical matched null). A copied-spike null trivially matches here — that
    sabotage is caught by the *separation* clause (iv), not this one; a length- or
    composition-perturbed null fails here.
    """
    return (
        int(spike["n_windows"]) == int(null["n_windows"])
        and str(spike["length_multiset_sha"]) == str(null["length_multiset_sha"])
        and dict(spike["composition"]) == dict(null["composition"])
    )


def _removed_and_joined(
    hits: Sequence[infernal.CmsearchHit],
    arm_windows: Mapping[str, Mapping[str, str]],
    *,
    score_threshold: float,
) -> tuple[dict[str, int], int]:
    """Join ``cmsearch`` hits back to windows by EXACT target name (pin 5(v)).

    Returns ``({arm: n_windows_removed}, n_hits_joined)``. A window is *removed* iff it draws
    at least one hit with ``score >= score_threshold`` (recall-favouring: a lower threshold
    refuses more windows, so the gate errs toward removal). ``n_hits_joined`` counts EVERY
    reported hit (threshold-independent) whose target maps to a scanned window in ANY arm — a
    reported hit that joins nothing means the name mapping is degraded and the rate is
    deflated, so the caller asserts ``n_hits_reported == n_hits_joined``. The match is on the
    exact name (no normalisation), so mutating a ``.N`` suffix on one side drops that hit's
    join — the versioned-vs-unversioned no-op guard [masking.py:370-405], per shard.
    """
    known = set()
    for arm in ARMS:
        known |= set(arm_windows.get(arm, {}))
    removed_by = {arm: set() for arm in ARMS}
    n_joined = 0
    for hit in hits:
        if hit.target not in known:
            continue  # unjoined — surfaces as n_hits_reported > n_hits_joined
        n_joined += 1
        if hit.score >= score_threshold:
            for arm in ARMS:
                if hit.target in arm_windows.get(arm, {}):
                    removed_by[arm].add(hit.target)
    return {arm: len(removed_by[arm]) for arm in ARMS}, n_joined


# ═════════════════════════════════════════════════════════════════════════════
# One shard → one segment (pure — the unit-tested join + count core)
# ═════════════════════════════════════════════════════════════════════════════
def build_shard_segment(
    shard: int,
    *,
    arm_windows: Mapping[str, Mapping[str, str]],
    hits: Sequence[infernal.CmsearchHit],
    tblout_text: str,
    cm_sha256: str,
    expected_production_windows: int,
    score_threshold: float,
    shard_ok: bool,
    genome_windows: Mapping[str, int],
    n_tblout_files: int = 1,
) -> dict[str, Any]:
    """Reduce ONE shard's ONE ``cmsearch`` invocation to a recorded segment.

    Every count is derived from the scan evidence (the parsed ``hits`` and the submitted
    ``arm_windows``), never from a config value. The per-arm ``invocation_id`` is the SHA-256
    of the shard's single ``--tblout`` **text** — so all three arms carry the same id *because
    they were parsed from one file*, not because a builder stamped three equal strings (pin
    5(vi)). ``genome_windows`` maps each production genome accession → its consumed production
    window count, for the clause-(viii) genome-completeness / within-genome-shrink checks.
    """
    invocation_id = tblout_identity(tblout_text)
    all_windows: dict[str, str] = {}
    for arm in ARMS:
        all_windows.update(arm_windows.get(arm, {}))

    removed_by, n_joined = _removed_and_joined(
        hits, arm_windows, score_threshold=float(score_threshold)
    )
    arms: dict[str, Any] = {}
    for arm in ARMS:
        meta = arm_metadata(arm_windows.get(arm, {}))
        meta["n_removed"] = int(removed_by[arm])
        meta["invocation_id"] = invocation_id
        arms[arm] = meta

    # Production consumed count is the number of production windows actually SUBMITTED to this
    # scan (the real window stream), never a manifest row count — pin 4.
    consumed_production = arms["production"]["n_windows"]
    if int(sum(genome_windows.values())) != consumed_production:
        raise SubstratePrescanError(
            f"shard {shard}: genome_windows sum {sum(genome_windows.values())} != consumed "
            f"production windows {consumed_production}"
        )
    return {
        "kind": "shard_segment",
        "step": STEP,
        "shard": int(shard),
        "shard_ok": bool(shard_ok),
        "n_tblout_files": int(n_tblout_files),
        "invocation_id": invocation_id,
        "cm_sha256": str(cm_sha256),
        "score_threshold": float(score_threshold),
        "expected_production_windows": int(expected_production_windows),
        "n_windows_scanned": len(all_windows),
        "n_distinct_names": len(set(all_windows)),
        "n_hits_reported": len(hits),
        "n_hits_joined": int(n_joined),
        "arms": arms,
        "genomes": sorted(genome_windows),
        "genome_windows": {str(k): int(v) for k, v in genome_windows.items()},
    }


# ═════════════════════════════════════════════════════════════════════════════
# The gate — a builder/validator-SHARED derivation of the 8 clauses from recorded
# counts. Called by build_report (to record) AND validate_report (to re-check),
# so a clause can never be read back from the requested config
# ([[gate-clauses-need-re-derivation]], [[promote-dont-duplicate-is-a-correctness-rule]]).
# ═════════════════════════════════════════════════════════════════════════════
def derive_clauses(
    segments: Sequence[Mapping[str, Any]],
    *,
    n_shards_expected: int,
    selected_accessions: Iterable[str],
    prior_genome_windows: Mapping[str, int] | None,
) -> dict[str, bool]:
    """Re-derive the 8 REQUIRED clauses from recorded segment counts + git-frozen external inputs.

    ``selected_accessions`` and ``prior_genome_windows`` are the git-frozen selection manifest
    and the prior committed fetch report — supplied EXTERNALLY, never read from the report
    under test (clause viii is the anti-shrink backstop and must consult the frozen truth, not
    the report's own copy). Every clause defaults FALSE and is set only from evidence.
    """
    segs = list(segments)
    selected = {str(a) for a in selected_accessions}
    clauses = dict.fromkeys(REQUIRED_CLAUSES, False)
    if not segs:
        return clauses  # no shard scanned anything ⇒ every clause FALSE

    # (i) production denominator — strict loaded==expected + emptiness, per shard AND aggregate
    agg_consumed = agg_expected = 0
    denom_ok = True
    for s in segs:
        consumed = int(s["arms"]["production"]["n_windows"])
        expected = int(s["expected_production_windows"])
        if expected > 0 and consumed == 0:  # decoys.py:720-724 emptiness arm
            denom_ok = False
        if consumed != expected:  # decoys.py:725-728 strict-equality arm
            denom_ok = False
        agg_consumed += consumed
        agg_expected += expected
    clauses["production_denominator_consistent"] = bool(
        denom_ok and agg_consumed == agg_expected and agg_consumed >= MIN_SCAN_N
    )

    # (ii) detector LIVE in every shard — spike arm removes > 0
    clauses["detector_live_every_shard"] = all(
        int(s["arms"]["spike"]["n_removed"]) > 0 for s in segs
    )

    # (iii) power arms min-N + matchedness, per shard
    clauses["power_arms_min_n"] = all(
        int(s["arms"]["spike"]["n_windows"]) >= MIN_SPIKE_N
        and int(s["arms"]["null"]["n_windows"]) >= MIN_NULL_N
        and arms_matched(s["arms"]["spike"], s["arms"]["null"])
        for s in segs
    )

    # (iv) STRICT spike-vs-null separation at the operating threshold, per shard
    def _sep(s: Mapping[str, Any]) -> bool:
        sp, nu = s["arms"]["spike"], s["arms"]["null"]
        if int(sp["n_windows"]) <= 0 or int(nu["n_windows"]) <= 0:
            return False
        return (int(sp["n_removed"]) / int(sp["n_windows"])) - (
            int(nu["n_removed"]) / int(nu["n_windows"])
        ) > MARGIN

    clauses["spike_null_separation"] = all(_sep(s) for s in segs)

    # (v) hit→window mapping totality + unique naming, per shard, all three arms
    clauses["hit_window_mapping_total"] = all(
        int(s["n_distinct_names"]) == int(s["n_windows_scanned"])
        and int(s["n_hits_reported"]) == int(s["n_hits_joined"])
        for s in segs
    )

    # (vi) shard completeness + one-tblout co-invocation (all arm ids equal, one file)
    n_recorded = len(segs)
    distinct_shards = len({int(s["shard"]) for s in segs}) == n_recorded
    coinv = all(
        int(s["n_tblout_files"]) == 1
        and s["arms"]["production"]["invocation_id"]
        == s["arms"]["spike"]["invocation_id"]
        == s["arms"]["null"]["invocation_id"]
        == s["invocation_id"]
        for s in segs
    )
    clauses["shard_completeness_coinvocation"] = bool(
        n_recorded == int(n_shards_expected)
        and n_recorded > 0
        and distinct_shards
        and all(bool(s["shard_ok"]) for s in segs)
        and coinv
    )

    # (vii) CM identity — every shard loaded the pinned RF00230.cm (fires without relying on vi)
    clauses["cm_identity"] = all(str(s["cm_sha256"]) == RF00230_CM_SHA256 for s in segs)

    # (viii) genome completeness — scanned set == git-frozen selection; no within-genome shrink
    scanned: set[str] = set()
    genome_windows: dict[str, int] = {}
    for s in segs:
        for acc in s["genomes"]:
            scanned.add(str(acc))
        for acc, nw in s.get("genome_windows", {}).items():
            genome_windows[str(acc)] = genome_windows.get(str(acc), 0) + int(nw)
    set_ok = bool(selected) and scanned == selected
    if prior_genome_windows is None:  # maiden fetch — the frozen set-equality carries it
        shrink_ok = True
    else:  # no genome's window total may have shrunk vs the prior committed fetch report
        prior = {str(k): int(v) for k, v in prior_genome_windows.items()}
        shrink_ok = all(genome_windows.get(acc, 0) >= prior.get(acc, 0) for acc in prior) and all(
            acc in genome_windows for acc in prior
        )
    clauses["genome_completeness"] = bool(set_ok and shrink_ok)
    return clauses


# ═════════════════════════════════════════════════════════════════════════════
# Aggregate report — every headline re-derived from the segments
# ═════════════════════════════════════════════════════════════════════════════
def build_report(
    segments: Sequence[Mapping[str, Any]],
    *,
    n_shards_expected: int,
    selected_accessions: Iterable[str],
    prior_genome_windows: Mapping[str, int] | None,
    score_threshold: float,
    substrate_scanned: bool,
    source: Mapping[str, Any] | None = None,
    accessed: str,
) -> dict[str, Any]:
    """Assemble the aggregate ``substrate_prescan.json`` — clauses via :func:`derive_clauses`."""
    segs = sorted(segments, key=lambda s: int(s["shard"]))
    selected = sorted({str(a) for a in selected_accessions})
    clauses = derive_clauses(
        segs,
        n_shards_expected=n_shards_expected,
        selected_accessions=selected,
        prior_genome_windows=prior_genome_windows,
    )
    overall_pass = all(clauses[k] for k in REQUIRED_CLAUSES)

    def _arm_total(arm: str, field: str) -> int:
        return sum(int(s["arms"][arm][field]) for s in segs)

    consumed_prod = _arm_total("production", "n_windows")
    removed_prod = _arm_total("production", "n_removed")
    expected_prod = sum(int(s["expected_production_windows"]) for s in segs)
    scanned_genomes = sorted({str(a) for s in segs for a in s["genomes"]})

    return {
        "schema_version": SCHEMA_VERSION,
        "clause_schema_version": CLAUSE_SCHEMA_VERSION,
        "step": STEP,
        "n_shards": len(segs),
        "n_shards_expected": int(n_shards_expected),
        "geometry": {"window_nt": WINDOW_NT, "stride_nt": STRIDE_NT},
        "cm_pinned_sha256": RF00230_CM_SHA256,
        "score_threshold": float(score_threshold),
        # Three DISJOINT window sets, three separate keys (pin 4 directive).
        "n_production_windows": consumed_prod,
        "n_spike_windows": _arm_total("spike", "n_windows"),
        "n_null_windows": _arm_total("null", "n_windows"),
        "expected_production_windows": expected_prod,
        "consumed_production_windows": consumed_prod,
        "n_production_windows_removed": removed_prod,
        # The YIELD, never a residual (pin 3/4). Named substrate_removal_rate so no reader
        # reads "high = bad" or "low = clean".
        "substrate_removal_rate": (removed_prod / consumed_prod) if consumed_prod else None,
        "lower_bound_disclosure": LOWER_BOUND_DISCLOSURE,
        "n_genomes_scanned": len(scanned_genomes),
        "n_genomes_selected": len(selected),
        "scanned_accessions": scanned_genomes,
        "selected_accessions": selected,
        "clauses": clauses,
        "overall_pass": bool(overall_pass),
        # Honesty flags (§10.3), validator-enforced.
        "is_removal_yield_not_residual": True,
        "residual_risk_lower_bounded_only": True,
        "tier2n_residual_unmeasured_above": True,
        "pins_no_adr_value": True,
        "is_science": False,
        "substrate_scanned": bool(substrate_scanned),
        "source": dict(source) if source else {},
        "accessed": accessed,
        "segments": segs,
    }


def validate_report(
    report: Mapping[str, Any],
    *,
    selected_accessions: Iterable[str],
    prior_genome_windows: Mapping[str, int] | None = None,
) -> list[str]:
    """Re-derive every clause + headline from the report's own evidence; list failures.

    Never raises; ``[] == valid``. ``selected_accessions`` / ``prior_genome_windows`` come from
    the git-frozen selection manifest and prior fetch report (NOT from the report under test) —
    that is what makes clause (viii) an anti-shrink backstop. A MISSING required clause key is a
    HARD FAIL (the clause-set-completeness gate); ``overall_pass`` must equal ``all(required
    re-derived clauses)``; every aggregate headline is re-summed from the segments.
    """
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version != {SCHEMA_VERSION!r} (got {report.get('schema_version')!r})"
        )
    if report.get("clause_schema_version") != CLAUSE_SCHEMA_VERSION:
        problems.append(
            f"clause_schema_version != {CLAUSE_SCHEMA_VERSION} "
            f"(got {report.get('clause_schema_version')!r}) — a new clause invalidates old reports"
        )
    if report.get("step") != STEP:
        problems.append(f"step != {STEP!r} (got {report.get('step')!r})")

    segs = report.get("segments")
    if not isinstance(segs, Sequence) or isinstance(segs, (str, bytes)) or not segs:
        return problems + ["segments missing or empty — a gate over zero shards certifies nothing"]

    # ── clause-set completeness: every REQUIRED key present + boolean (missing = HARD FAIL) ──
    recorded = report.get("clauses")
    if not isinstance(recorded, Mapping):
        return problems + ["clauses block missing"]
    for k in REQUIRED_CLAUSES:
        if k not in recorded:
            problems.append(f"REQUIRED clause {k!r} MISSING — hard fail (never silently dropped)")
        elif not isinstance(recorded[k], bool):
            problems.append(f"clause {k!r} is not a bool (got {recorded[k]!r})")

    # ── re-derive the clauses from the recorded counts + the git-frozen external truth ──
    try:
        rederived = derive_clauses(
            segs,
            n_shards_expected=int(report.get("n_shards_expected", -1)),
            selected_accessions=selected_accessions,
            prior_genome_windows=prior_genome_windows,
        )
    except (KeyError, TypeError, ValueError, SubstratePrescanError) as exc:
        return problems + [f"segments are mis-shaped, clauses not re-derivable: {exc}"]
    for k in REQUIRED_CLAUSES:
        if k in recorded and bool(recorded[k]) is not rederived[k]:
            problems.append(f"clause {k!r} recorded {recorded[k]} != re-derived {rederived[k]}")

    want_overall = all(rederived[k] for k in REQUIRED_CLAUSES)
    if bool(report.get("overall_pass")) is not want_overall:
        problems.append(
            f"overall_pass {report.get('overall_pass')} != all(required re-derived) {want_overall}"
        )
    # A consistent report whose gate did NOT pass is honest but NON-CERTIFYING: []==valid must
    # mean "consistent AND admissible", so :func:`reduce` refuses to write it (the rho_pilot
    # reduce_partials precedent — never emit a report the substrate would be admitted on unless
    # every REQUIRED clause passed). The failing clauses are named so the §9.3 stop is actionable.
    if not want_overall:
        failed = [k for k in REQUIRED_CLAUSES if not rederived[k]]
        problems.append(f"gate did NOT pass — substrate inadmissible; failing clauses: {failed}")

    # ── the report's recorded selection must equal the git-frozen selection (no shrink lie) ──
    frozen = sorted({str(a) for a in selected_accessions})
    if [str(a) for a in report.get("selected_accessions", [])] != frozen:
        problems.append("selected_accessions in report != git-frozen selection manifest")
    if int(report.get("n_genomes_selected", -1)) != len(frozen):
        problems.append("n_genomes_selected != |git-frozen selection|")

    # ── aggregate headlines re-summed from the segments (never read back) ──
    consumed_prod = sum(int(s["arms"]["production"]["n_windows"]) for s in segs)
    removed_prod = sum(int(s["arms"]["production"]["n_removed"]) for s in segs)
    expected_prod = sum(int(s["expected_production_windows"]) for s in segs)
    for key, want in (
        ("n_production_windows", consumed_prod),
        ("consumed_production_windows", consumed_prod),
        ("n_production_windows_removed", removed_prod),
        ("expected_production_windows", expected_prod),
        ("n_spike_windows", sum(int(s["arms"]["spike"]["n_windows"]) for s in segs)),
        ("n_null_windows", sum(int(s["arms"]["null"]["n_windows"]) for s in segs)),
        ("n_shards", len(segs)),
    ):
        if int(report.get(key, -1)) != want:
            problems.append(f"{key} {report.get(key)} != re-summed {want}")
    want_rate = (removed_prod / consumed_prod) if consumed_prod else None
    got_rate = report.get("substrate_removal_rate")
    if want_rate is None:
        if got_rate is not None:
            problems.append(
                "substrate_removal_rate must be null when 0 production windows consumed"
            )
    elif got_rate is None or abs(float(got_rate) - want_rate) > 1e-12:
        problems.append(f"substrate_removal_rate {got_rate} != removed/consumed {want_rate}")

    if str(report.get("cm_pinned_sha256")) != RF00230_CM_SHA256:
        problems.append("cm_pinned_sha256 != the pinned RF00230.cm sha256")

    # ── honesty flags (§10.3) ──
    for flag, want in (
        ("is_removal_yield_not_residual", True),
        ("residual_risk_lower_bounded_only", True),
        ("tier2n_residual_unmeasured_above", True),
        ("pins_no_adr_value", True),
        ("is_science", False),
    ):
        if bool(report.get(flag)) is not want:
            problems.append(f"honesty flag {flag} must be {want}")
    if not str(report.get("lower_bound_disclosure", "")).strip():
        problems.append("lower_bound_disclosure missing — every reported value must carry pin-3")
    return problems


# ═════════════════════════════════════════════════════════════════════════════
# Runtime legs — scan one shard (cluster: needs cmsearch), reduce shards → report
# ═════════════════════════════════════════════════════════════════════════════
def scan_shard(
    shard: int,
    *,
    arm_windows: Mapping[str, Mapping[str, str]],
    genome_windows: Mapping[str, int],
    expected_production_windows: int,
    score_threshold: float,
    cm: str | Path = DEFAULT_CM,
    workdir: str | Path,
    cpu: int = 8,
) -> dict[str, Any]:
    """Scan ONE shard with ONE ``cmsearch`` invocation and reduce it to a segment (cluster leg).

    Builds a single FASTA of the shard's production + spike + null windows (uniquely named,
    whitespace-free — :func:`infernal.write_fasta` enforces it), runs **one** ``cmsearch``
    without ``--cut_ga`` (the recall-favouring rule; the numeric ``score_threshold`` is applied
    downstream in :func:`build_shard_segment`, not pinned here), records the loaded CM's sha256,
    and derives the segment from that one tblout. Requires ``cmsearch`` on PATH (the
    ``tbox-infernal`` env) — an absent binary raises rather than manufacturing an empty scan.
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    for arm in ARMS:
        for name, seq in arm_windows.get(arm, {}).items():
            if name in merged:
                raise SubstratePrescanError(
                    f"shard {shard}: duplicate window name across arms: {name!r}"
                )
            merged[name] = seq
    fasta = infernal.write_fasta(merged, work / f"shard_{shard}.fna")
    tblout = work / f"shard_{shard}.tblout"
    # cut_ga=False → recall-favouring: report weak hits too; the operating cutoff is applied
    # as score_threshold in build_shard_segment (unpinned rule, §10.3).
    hits = infernal.run_cmsearch(cm, fasta, tblout, cut_ga=False, cpu=cpu)
    return build_shard_segment(
        shard,
        arm_windows=arm_windows,
        hits=hits,
        tblout_text=Path(tblout).read_text(encoding="utf-8"),
        cm_sha256=provenance.sha256_file(cm),
        expected_production_windows=expected_production_windows,
        score_threshold=score_threshold,
        shard_ok=True,
        genome_windows=genome_windows,
    )


def load_segments(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read + shape-check per-shard segment JSONs; raise on a missing/duplicate/mis-typed shard."""
    if not paths:
        raise SubstratePrescanError("no shard segments given to reduce")
    segs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for p in paths:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        if obj.get("kind") != "shard_segment" or obj.get("step") != STEP:
            raise SubstratePrescanError(f"{p} is not a {STEP} shard segment")
        sh = int(obj["shard"])
        if sh in seen:
            raise SubstratePrescanError(f"duplicate shard {sh} among segments")
        seen.add(sh)
        segs.append(obj)
    return segs


def load_selection_accessions(manifest: str | Path = DEFAULT_SELECTION_MANIFEST) -> list[str]:
    """The git-frozen selected-genome accession set (``assembly_accession`` column).

    Reads the committed selection manifest (P2-10c′-a ``pilot_genomes_v0.parquet`` for the
    pilot; the production selection manifest for the full fetch). Imported lazily so the
    stdlib report/clause path stays free of the pandas/pyarrow stack (bare-CI tier).
    """
    import pandas as pd  # local import: keep the gate logic stdlib-only

    df = pd.read_parquet(manifest, columns=["assembly_accession"])
    return sorted(str(a) for a in df["assembly_accession"].tolist())


def write_report(report: Mapping[str, Any], out_path: str | Path = DEFAULT_REPORT) -> Path:
    """Write the aggregate report as pretty JSON (parents created)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return p


def prior_genome_windows_from_fetch_report(path: str | Path) -> dict[str, int]:
    """Per-genome production-window totals from a prior committed fetch report (clause-viii base).

    Uses ``per_genome[].assembly_accession`` → its recorded window total. The pilot fetch report
    (``pilot_fetch_report.json``) records ``total_bp`` per genome, not ``n_windows``; the
    production fetch manifest records ``n_windows`` per genome directly. This reads whichever the
    report carries (``n_windows`` preferred), so the within-genome-shrink leg diffs against the
    previous report, never a floor ([[ncbi-refetch-429-silent-shrink]]).
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for g in obj.get("per_genome", []):
        acc = str(g.get("assembly_accession", ""))
        if not acc:
            continue
        if "n_windows" in g:
            out[acc] = int(g["n_windows"])
        elif "total_bp" in g:  # window count is deterministic in bp given the pinned geometry
            out[acc] = int(g["total_bp"])
    return out


# ═════════════════════════════════════════════════════════════════════════════
# The fetch → prescan contract: a shard-spec JSON (production windows for a shard,
# emitted by the tiling/fetch step) + a control-manifest JSON (spike/null arms,
# emitted by scripts/make_substrate_prescan_control.py). Kept a small explicit
# contract so this module never depends on the fetch's internal FASTA layout.
# ═════════════════════════════════════════════════════════════════════════════
def read_shard_spec(path: str | Path) -> dict[str, Any]:
    """Read one shard's production-window spec: ``{shard, expected_production_windows,
    genome_windows: {acc: n}, production: {name: seq}}``."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("shard", "expected_production_windows", "genome_windows", "production"):
        if key not in obj:
            raise SubstratePrescanError(f"{path} shard-spec missing {key!r}")
    return obj


def read_control_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    """Read the control arms: ``{spike: {name: seq}, null: {name: seq}}`` (from the generator)."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    arms = obj.get("arms", obj)
    for arm in ("spike", "null"):
        if arm not in arms or not isinstance(arms[arm], Mapping):
            raise SubstratePrescanError(f"{path} control manifest missing {arm!r} arm")
    return {"spike": dict(arms["spike"]), "null": dict(arms["null"])}


def run_shard_from_specs(
    shard_spec_path: str | Path,
    control_manifest_path: str | Path,
    *,
    score_threshold: float,
    cm: str | Path,
    workdir: str | Path,
    out_segment: str | Path,
    ok_marker: str | Path,
) -> Path:
    """Cluster leg for one SLURM-array task: scan a shard, write its segment + ``SHARD_OK``."""
    spec = read_shard_spec(shard_spec_path)
    control = read_control_manifest(control_manifest_path)
    seg = scan_shard(
        int(spec["shard"]),
        arm_windows={
            "production": dict(spec["production"]),
            "spike": control["spike"],
            "null": control["null"],
        },
        genome_windows={str(k): int(v) for k, v in spec["genome_windows"].items()},
        expected_production_windows=int(spec["expected_production_windows"]),
        score_threshold=score_threshold,
        cm=cm,
        workdir=workdir,
    )
    out = Path(out_segment)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(seg, indent=2, sort_keys=True) + "\n")
    # SHARD_OK is written ONLY on the success path (CLAUDE.md §9.3), after the segment lands.
    Path(ok_marker).parent.mkdir(parents=True, exist_ok=True)
    Path(ok_marker).write_text(f"shard {seg['shard']} ok\n")
    return out


def reduce(
    segment_paths: Sequence[str | Path],
    *,
    n_shards_expected: int,
    selected_accessions: Iterable[str],
    prior_genome_windows: Mapping[str, int] | None,
    score_threshold: float,
    accessed: str,
    out_report: str | Path = DEFAULT_REPORT,
    out_provenance: str | Path = DEFAULT_PROVENANCE,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-shard segments → the certified report; raise if it does not certify.

    Mirrors :func:`tbox_finder.infer.rho_pilot.reduce_partials`: the report is validated with
    the same builder/validator-shared clauses and is **never written** if it fails to certify,
    so a broken scan cannot leave a healthy-looking artifact on disk (§10.3).
    """
    segs = load_segments(segment_paths)
    report = build_report(
        segs,
        n_shards_expected=n_shards_expected,
        selected_accessions=selected_accessions,
        prior_genome_windows=prior_genome_windows,
        score_threshold=score_threshold,
        substrate_scanned=True,
        source=source,
        accessed=accessed,
    )
    problems = validate_report(
        report,
        selected_accessions=selected_accessions,
        prior_genome_windows=prior_genome_windows,
    )
    if problems:
        raise SubstratePrescanError(
            "substrate_prescan report failed to certify:\n  - " + "\n  - ".join(problems)
        )
    write_report(report, out_report)
    provenance.write_provenance(
        out_provenance,
        rule="slurm/p2/substrate_prescan.sbatch :: tbox_finder.mining.substrate_prescan reduce",
        script="src/tbox_finder/mining/substrate_prescan.py",
        inputs=[Path(p) for p in segment_paths],
        outputs=[out_report],
        env_lock="envs/infernal.conda-lock.yml",
        adr="ADR-0005",
        extra={"step": STEP, "clause_schema_version": CLAUSE_SCHEMA_VERSION},
    )
    return report


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Off-catalogue cmsearch substrate pre-scan (ADR-0005 A8)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scan-shard", help="scan one shard (cluster; needs cmsearch)")
    sc.add_argument("--shard-spec", required=True)
    sc.add_argument("--control", required=True)
    sc.add_argument("--score-threshold", type=float, required=True)
    sc.add_argument("--cm", default=str(DEFAULT_CM))
    sc.add_argument("--workdir", required=True)
    sc.add_argument("--out-segment", required=True)
    sc.add_argument("--ok-marker", required=True)

    rd = sub.add_parser("reduce", help="aggregate shard segments → the certified report")
    rd.add_argument("--segment", action="append", required=True, dest="segments")
    rd.add_argument("--n-shards-expected", type=int, required=True)
    rd.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    rd.add_argument("--prior-fetch-report", default=None)
    rd.add_argument("--score-threshold", type=float, required=True)
    rd.add_argument("--accessed", required=True)
    rd.add_argument("--out-report", default=str(DEFAULT_REPORT))

    args = ap.parse_args(argv)
    if args.cmd == "scan-shard":
        run_shard_from_specs(
            args.shard_spec,
            args.control,
            score_threshold=args.score_threshold,
            cm=args.cm,
            workdir=args.workdir,
            out_segment=args.out_segment,
            ok_marker=args.ok_marker,
        )
        return 0
    prior = (
        prior_genome_windows_from_fetch_report(args.prior_fetch_report)
        if args.prior_fetch_report
        else None
    )
    reduce(
        args.segments,
        n_shards_expected=args.n_shards_expected,
        selected_accessions=load_selection_accessions(args.selection_manifest),
        prior_genome_windows=prior,
        score_threshold=args.score_threshold,
        accessed=args.accessed,
        out_report=args.out_report,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(_main())
