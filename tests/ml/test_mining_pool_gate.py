"""P2-10b mask-non-vacuity gate, re-derived from the shipped artifacts (CLAUDE.md §8.5).

The step's gate is "a known-locus-overlapping candidate is actually masked, and
residual contamination is reported against the union denominator". Reading that
off ``mining_pool_report.json`` alone would only check that the builder wrote the
numbers it computed; these tests **re-derive** the load-bearing clauses from the
real artifacts so a report that disagrees with the data cannot pass.

Two arming vars, because the clauses have two different input tiers:

* the clauses reading only the **git-tracked** audit JSONs need no arming var at
  all: a missing committed file is a broken checkout, so they raise rather than
  skip. That is strictly stronger than a var, and it cannot rot into an unarmed
  ``TBOX_REQUIRE_*`` that silently skips green.
* ``TBOX_REQUIRE_MINING_POOL`` — the clauses re-deriving from the **DVC-tracked**
  parquets. CI has no ``dvc pull``, so this is the CLAUDE.md §8.5 step-local gate
  and is deliberately **not** armed in ``ci.yml``; run it locally after any change
  to the decoy/masking/context pipeline.

Splitting them is the point: a single var covering both meant the whole file
skipped green whenever the parquets were absent, which is the default state in
CI, on a fresh cluster checkout, and on any un-``dvc pull``-ed laptop — 6 of the
9 gate clauses never executed and nothing said so (P2-10b review finding).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CONTEXT = _REPO / "data/interim/flank_context/context_v0.parquet"
_UNION = _REPO / "data/processed/priors/union_prior.parquet"
_CORPUS = _REPO / "data/processed/master_clean_v0.parquet"
_DECOYS = _REPO / "data/processed/negatives/decoys_v0.parquet"
_POOL = _REPO / "data/processed/negatives/mining_pool_v0.parquet"
_POOL_REPORT = _REPO / "data/processed/audits/mining_pool_report.json"
_DECOY_REPORT = _REPO / "data/processed/audits/decoys_report.json"

_REQUIRE = os.environ.get("TBOX_REQUIRE_MINING_POOL") == "1"
#: The audit JSONs are committed to git, so they are present in any checkout.
_TRACKED = {_POOL_REPORT, _DECOY_REPORT}


def _need(*paths: Path) -> None:
    pytest.importorskip("pandas")
    missing = [p for p in paths if not p.is_file()]
    if not missing:
        return
    names = [str(p.relative_to(_REPO)) for p in missing]
    if all(p in _TRACKED for p in missing):
        # Committed files: absence is a broken checkout, never a skip.
        raise AssertionError(f"git-tracked audit reports missing: {names}")
    message = f"DVC-tracked inputs absent: {names} (dvc pull to run this tier)"
    if _REQUIRE:
        pytest.fail(f"TBOX_REQUIRE_MINING_POOL=1 but {message}")
    pytest.skip(message)


@pytest.fixture(scope="module")
def index():
    _need(_UNION, _CORPUS)
    from tbox_finder import masking

    union_loci, _, _ = masking.load_union_loci(_UNION)
    own_loci = masking.load_own_positive_loci(_CORPUS)
    return masking.LocusIndex.from_records(union_loci + own_loci)


# --------------------------------------------------------------------------- #
# The designed control — the clause that proves the mask is live
# --------------------------------------------------------------------------- #
def test_every_locus_centred_control_window_masks(index) -> None:
    """A locus-centred window overlaps a known T-box by construction: 100% or a bug.

    This is the falsifier the natural rate cannot be: a natural rate of zero is a
    legitimate biological result, so it can never distinguish "nothing overlaps"
    from "the mask is structurally dead". Re-derived here from ``context_v0``
    rather than read from the report.
    """
    _need(_CONTEXT)
    import pandas as pd

    from tbox_finder.mining import pool as mining_pool

    context = pd.read_parquet(_CONTEXT)
    rows = context.to_dict("records")
    controls = [
        w
        for w in (
            mining_pool.carve_window(r, mining_pool.SIDE_LOCUS_CONTROL, window_nt=300, margin_nt=50)
            for r in rows
        )
        if w is not None
    ]
    assert len(controls) > 20_000, f"only {len(controls)} anchored rows — fixture drifted"
    unmasked = [
        c
        for c in controls
        if not index.is_masked(c["accession"], c["locus_start"], c["locus_end"], flank=50)
    ]
    assert not unmasked, (
        f"{len(unmasked)}/{len(controls)} locus-centred controls did NOT mask — the "
        "coordinate frame is wrong, not the biology "
        f"(first: {unmasked[0]['candidate_id']} {unmasked[0]['accession']} "
        f"{unmasked[0]['locus_start']}-{unmasked[0]['locus_end']})"
    )


def test_the_wrong_coordinate_frame_would_fail_this_gate(index) -> None:
    """The control has teeth only if a plausible-but-wrong frame breaks it.

    Anchoring minus-strand windows to ``region_start`` (ignoring the server-side
    reverse-complement) is the natural mistake; it must be visibly worse.
    """
    _need(_CONTEXT)
    import pandas as pd

    context = pd.read_parquet(_CONTEXT)
    ok = context[context["status"] == "ok"]
    wrong = 0
    for row in ok.itertuples():
        lo = row.region_start + row.locus_offset
        if not index.is_masked(row.accession, lo, lo + row.locus_length - 1, flank=50):
            wrong += 1
    # A threshold of 1 would accept a state in which the control has essentially
    # no power: only 386 of 23,532 rows (1.64 %) distinguish the naive frame, so a
    # 500-row sample contains ~5 of them. Pin the measured margin instead.
    assert wrong >= 300, (
        f"only {wrong}/{len(ok)} rows discriminate the naive forward-only frame "
        "(was 386) — the flank-window control has lost its power"
    )


# --------------------------------------------------------------------------- #
# Namespace compatibility — the measured silent no-op
# --------------------------------------------------------------------------- #
def test_the_union_index_and_the_decoy_accessions_share_a_namespace(index) -> None:
    """0 as-is vs 269 normalised was the measured P2-10b defect."""
    _need(_DECOYS)
    import pandas as pd

    from tbox_finder import masking

    decoys = pd.read_parquet(_DECOYS, columns=["pool", "accession"])
    structured = decoys[decoys["pool"] == "structured_rna"]["accession"].dropna()
    assert len(structured) > 0, "structured_rna carries no coordinates — P2-10b regressed"
    report = masking.accession_namespace_report(structured, index)
    assert report["namespace_compatible"] is True
    assert report["n_intersect_normalized"] > 0


def test_normalisation_is_what_makes_the_namespaces_compatible(index) -> None:
    """Sabotage-equivalent: without version stripping the intersection is empty.

    The assertion has to be written against ``normalize_accession`` itself, not
    against the two key sets. On this data ``raw_accession_keys ==
    accession_keys`` (no union-prior accession carries a version), so any
    comparison between them is invariant to the function under test — an earlier
    form of this test asserted exactly that and stayed green under the sabotage
    it was named for.
    """
    _need(_DECOYS)
    import pandas as pd

    from tbox_finder.masking import normalize_accession

    decoys = pd.read_parquet(_DECOYS, columns=["pool", "accession"])
    structured = {str(a) for a in decoys[decoys["pool"] == "structured_rna"]["accession"].dropna()}
    # As written, the decoy accessions address nothing: they are versioned.
    assert structured & index.accession_keys == set()
    # Normalised, they address the index. This clause is False iff the
    # normalisation stops stripping versions.
    normalised = {normalize_accession(a) for a in structured}
    assert normalised & index.accession_keys != set()


# --------------------------------------------------------------------------- #
# The shipped reports must agree with the data
# --------------------------------------------------------------------------- #
def test_the_mining_pool_report_gate_passed_and_is_not_vacuous() -> None:
    _need(_POOL_REPORT)
    report = json.loads(_POOL_REPORT.read_text())
    gate = report["control_gate"]
    assert gate["overall_pass"] is True
    assert gate["n_control_windows"] > 0
    assert gate["n_control_masked"] == gate["n_control_windows"]
    # The gate must rest on true overlap, not on flank proximity: a control window
    # IS the locus, so ±flank slack would let a frame error up to flank_nt through.
    assert gate["n_control_overlapping"] == gate["n_control_windows"]
    assert report["masking"]["natural"]["n_records"] > 0
    # The designed controls must not be folded into the natural rate. Asserting
    # side_counts == n_control_windows only restates how both were derived; the
    # binding check is that the two partitions are disjoint and exhaust the pool.
    assert report["side_counts"]["locus_control"] == gate["n_control_windows"]
    assert (
        report["masking"]["natural"]["n_records"] + gate["n_control_windows"] == report["n_records"]
    )
    assert report["masking"]["natural"]["n_records"] == (
        report["side_counts"]["lead"] + report["side_counts"]["trail"]
    )


def test_the_decoy_report_records_pool_side_coverage_not_only_the_union_residual() -> None:
    """The union-side residual is insensitive to pool maskability; it cannot be the gate."""
    _need(_DECOY_REPORT)
    report = json.loads(_DECOY_REPORT.read_text())
    coverage = report["coordinate_coverage"]
    assert coverage["structured_rna"]["n_with_coordinates"] > 0
    # Honest, measured, and asserted so a future refactor cannot quietly invent them:
    # these pools have no genomic coordinates even in principle.
    assert coverage["gc_background"]["n_with_coordinates"] == 0
    assert coverage["dinuc_shuffled"]["n_with_coordinates"] == 0
    assert coverage["leader_decoy"]["n_with_coordinates"] == 0


def test_the_decoy_report_does_not_claim_overlaps_it_does_not_have() -> None:
    """structured_rna has 0 true overlaps; the report must say so, not merely imply it.

    ``n_overlapping <= n_masked`` is a monotonicity identity (flank 0 ⊆ flank 50)
    that no input can falsify — asserting it would test nothing. The claim that
    matters is the *value*: ADR-0005 A6 rests the whole ``genomic_window`` design
    on structured_rna having **zero** true overlaps, so a resubsample that
    introduces real contamination must break this test rather than hide inside
    the flank count.
    """
    _need(_DECOY_REPORT)
    report = json.loads(_DECOY_REPORT.read_text())
    structured = report["coordinate_coverage"]["structured_rna"]
    assert structured["n_overlapping_at_flank_0"] == 0, (
        "structured_rna now has true union-prior overlaps — ADR-0005 A6's premise "
        "for the designed-control design has changed and must be re-derived"
    )
    assert structured["n_masked_at_flank"] >= structured["n_overlapping_at_flank_0"]


def test_the_mining_pool_artifact_matches_its_report() -> None:
    _need(_POOL, _POOL_REPORT)
    import pandas as pd

    df = pd.read_parquet(_POOL)
    report = json.loads(_POOL_REPORT.read_text())
    assert len(df) == report["n_records"]
    assert int(df["is_designed_control"].sum()) == report["control_gate"]["n_control_windows"]
    assert df["accession"].notna().all()
    assert df["locus_start"].notna().all()
    assert df["locus_end"].notna().all()
    assert (df["locus_start"] <= df["locus_end"]).all()


def test_every_mining_pool_record_is_classifiable_rather_than_refused(index) -> None:
    """The P2-10b headline: ``refused_no_coordinates`` was 100% of every mineable pool."""
    _need(_POOL)
    import pandas as pd

    from tbox_finder.mining.hard_negative import (
        OUTCOME_UNMASKABLE,
        MiningCandidate,
        classify_candidate,
    )

    df = pd.read_parquet(_POOL).head(2_000)
    refused = 0
    for row in df.itertuples():
        candidate = MiningCandidate(
            candidate_id=str(row.candidate_id),
            pool=str(row.pool),
            # Deliberately NOT str()/int()-coerced: those would turn a null
            # accession into the literal "nan" and a null coordinate into a
            # TypeError-or-garbage, i.e. they would launder away the exact
            # missingness this test exists to detect.
            accession=row.accession,
            locus_start=row.locus_start,
            locus_end=row.locus_end,
            score=1.0,
        )
        if classify_candidate(candidate, index)[0] == OUTCOME_UNMASKABLE:
            refused += 1
    assert refused == 0, f"{refused}/{len(df)} substrate records still refused for lack of coords"


# --------------------------------------------------------------------------- #
# The union prior must actually contribute — the step's headline guard
# --------------------------------------------------------------------------- #
def test_the_union_prior_contributes_loci_beyond_the_corpus_positives() -> None:
    """Every designed control masks against ``own_loci`` alone, so it cannot see this.

    Measured: masking from union-only, own-only, and both gives bit-identical
    counts (control 500/500, natural 85/46,091). A dead union prior would leave
    every gate clause green and the report still printing a 24,160 denominator,
    silently degrading ADR-0005 D14's first guard to own-positives-only. The
    union prior's *distinct contribution* therefore has to be asserted directly.
    """
    _need(_UNION, _CORPUS)
    from tbox_finder import masking

    union_loci, n_total, n_maskable = masking.load_union_loci(_UNION)
    own_loci = masking.load_own_positive_loci(_CORPUS)
    assert union_loci, "union prior yielded 0 loci — the mask would be a no-op"
    assert own_loci, "corpus yielded 0 own-positive loci"
    # The denominator the report publishes must equal the loci actually loaded,
    # or a partial load reads as a healthy full one.
    assert len(union_loci) == n_maskable
    assert n_maskable <= n_total

    union_keys = {masking.normalize_accession(a) for a, _, _ in union_loci}
    own_keys = {masking.normalize_accession(a) for a, _, _ in own_loci}
    union_only = union_keys - own_keys
    assert len(union_only) >= 400, (
        f"only {len(union_only)} accessions are contributed by the union prior alone "
        "(was 414) — it has stopped adding reach beyond the training corpus"
    )
    # Those accessions must be addressable in an index built from the union prior.
    union_index = masking.LocusIndex.from_records(union_loci)
    assert union_only <= union_index.accession_keys


def test_the_report_records_the_loci_actually_loaded_not_only_the_denominator() -> None:
    """``union_denominator`` comes from a row count and stays healthy on a dead load."""
    _need(_POOL_REPORT)
    report = json.loads(_POOL_REPORT.read_text())
    assert report["n_union_loci_loaded"] > 0
    assert report["n_own_positive_loci_loaded"] > 0
    assert report["n_union_loci_loaded"] == report["union_maskable_with_coords"]


# --------------------------------------------------------------------------- #
# Exact frame identity + the control contract
# --------------------------------------------------------------------------- #
def test_every_control_window_reproduces_its_known_interval_exactly(index) -> None:
    """Zero-tolerance frame check — overlap alone tolerates ±165 nt.

    23,532/23,532 holds on the real data, and a ±1 nt shift takes it to 0, so
    every control discriminates instead of only the ~1.6 % a shift pushes clear.
    """
    _need(_CONTEXT)
    import pandas as pd

    from tbox_finder.mining import pool as mining_pool

    rows = pd.read_parquet(_CONTEXT).to_dict("records")
    controls = [
        w
        for w in (
            mining_pool.carve_window(r, mining_pool.SIDE_LOCUS_CONTROL, window_nt=300, margin_nt=50)
            for r in rows
        )
        if w is not None
    ]
    inexact = [
        c
        for c in controls
        if not index.matches_interval_exactly(c["accession"], c["locus_start"], c["locus_end"])
    ]
    assert not inexact, f"{len(inexact)}/{len(controls)} controls did not reproduce their interval"
    # The check has teeth: a one-nucleotide shift must break all of them.
    still = sum(
        1
        for c in controls
        if index.matches_interval_exactly(c["accession"], c["locus_start"] + 1, c["locus_end"] + 1)
    )
    assert still == 0, f"{still} controls survive a +1 nt shift — exact match is not exact"


def test_the_shipped_gate_carved_every_control_it_requested() -> None:
    _need(_POOL_REPORT)
    gate = json.loads(_POOL_REPORT.read_text())["control_gate"]
    assert gate["n_controls_requested"] is not None
    assert gate["n_control_windows"] == gate["n_controls_requested"]
    assert gate["n_control_exact_interval_match"] == gate["n_control_windows"]
    assert gate["clauses"]["all_controls_exact"] is True
    assert gate["clauses"]["all_requested_controls_carved"] is True


# --------------------------------------------------------------------------- #
# Strand frames: the sequence and the coordinates are deliberately different
# --------------------------------------------------------------------------- #
def test_a_control_window_carries_its_locus_sequence_in_element_orientation() -> None:
    """``sequence`` is element-oriented; ``locus_start``/``end`` are forward-genome.

    Both frames are intentional but they differ on the minus strand, and nothing
    checked it: reverse-complementing every minus-strand slice left the whole
    suite green. A control window is the locus, so its sequence must equal the
    corpus record verbatim on both strands.
    """
    _need(_CONTEXT, _CORPUS)
    import pandas as pd

    from tbox_finder import ingest
    from tbox_finder.mining import pool as mining_pool

    corpus = pd.read_parquet(_CORPUS)
    by_id = dict(zip(ingest.compute_record_hashes(corpus), corpus["FASTA_sequence"], strict=True))
    rows = pd.read_parquet(_CONTEXT).to_dict("records")
    checked = {1: 0, 2: 0}
    for row in rows:
        window = mining_pool.carve_window(
            row, mining_pool.SIDE_LOCUS_CONTROL, window_nt=300, margin_nt=50
        )
        if window is None or window["source_record_id"] not in by_id:
            continue
        assert (
            window["sequence"] == str(by_id[window["source_record_id"]]).upper()
        ), f"{window['candidate_id']} (strand {window['strand']}) does not match its corpus record"
        checked[window["strand"]] += 1
    assert checked[1] > 1_000 and checked[2] > 1_000, f"strand coverage too thin: {checked}"


def test_every_source_record_id_resolves_to_a_real_corpus_record() -> None:
    """The dinuc parent link is content-addressed; a narrowed corpus read breaks it silently.

    Reading only the 3 columns the generators need changes the record hash, which
    re-points all 2,000 links while leaving them non-null and the golden digest
    untouched.
    """
    _need(_DECOYS, _CORPUS)
    import pandas as pd

    from tbox_finder import ingest

    corpus = pd.read_parquet(_CORPUS)
    valid = set(ingest.compute_record_hashes(corpus))
    decoys = pd.read_parquet(_DECOYS, columns=["pool", "source_record_id"])
    linked = decoys[decoys["source_record_id"].notna()]
    assert len(linked) > 0, "no decoy carries a parent link — P2-10b regressed"
    assert set(linked["pool"]) == {"dinuc_shuffled"}
    unknown = set(linked["source_record_id"]) - valid
    assert not unknown, f"{len(unknown)} source_record_ids resolve to no corpus record"
