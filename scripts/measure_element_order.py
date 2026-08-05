#!/usr/bin/env python3
"""Measure the 5′→3′ order of the seven annotated T-box elements on the curated corpus.

This is the reproducible provenance for :data:`tbox_finder.infer.strand.CANONICAL_ELEMENT_ORDER`
— the rank the **PRD §6 / ADR-0005 D15 strand-resolver** compares an observed element order
against to decide which strand a candidate locus lies on. That constant must not be typed from
memory, and it must not be borrowed from ``labels.CLASS_ORDER``: ``CLASS_ORDER`` is the
*softmax index* order, and it disagrees with the measured 5′→3′ order (it places
``Discriminator`` last; the corpus places it 5′ of the antiterminator's midpoint in essentially
every record, because the ~3-nt discriminator sits inside the 5′ half of the ~30-nt
antiterminator extent — the nesting PRD §8 documents). Reusing it would have mis-ranked one
element in seven with nothing failing.

What is measured, over ``data/processed/master_clean_v0.parquet`` (the P0-12 ingest):

  * ``median_midpoint`` — per element, the median of ``(lo + hi) / 2`` across the records that
    annotate it. The **midpoint**, not the start, because that is the statistic the resolver
    itself uses: it reduces each observed element to the median position of its predicted
    nucleotides, which is a midpoint for a contiguous run and is robust to ``Stem_I`` being
    split around the ``Specifier`` nested inside it.
  * ``pairs[].frac_a_before_b`` — for each unordered pair, the fraction of records annotating
    **both** in which A's midpoint precedes B's, plus ``n_both`` and the tie count. This is the
    per-pair support for the total order; a rank adjacency resting on a weak majority is
    visible rather than implied.

Coordinates are per-record local (each record carries its own origin), so every comparison is
made **within** a record between that record's own elements and only then aggregated — no
cross-record coordinate comparison is made, and the medians in ``elements[]`` are reported for
orientation only. The total order is derived from the pairwise matrix, never from the medians.

Absence is decided by the shared
``scripts/measure_overlap_prevalence.py::element_intervals`` reader (null or negative
coordinate ⇒ absent) so this measurement and the ADR-0004 precedence measurement cannot drift
apart on what "annotated" means.

The literature this agrees with (CLAUDE.md §10.1, ≥2 independent agreeing sources for a
high-stakes claim; all accessed 2026-08-05): 5′ Stem I carrying the specifier, then the 3′
antiterminator/antisequestrator (discriminator) domain — Zhang & Ferré-D'Amaré 2013,
"Stem I contains the specifier trinucleotide … 3′ to stem I is the antiterminator domain"
[PMID:23892783]; Suddala & Zhang 2019, "a 5′ Stem I domain … and a 3′
antiterminator/antisequestrator (or discriminator) domain" [PMID:31206978;
DOI:10.1002/iub.2098]; Zhang 2020, "the Stem I, Stem II, and Discriminator domains, which
collectively compose the T-box riboswitches" [PMID:32633085; DOI:10.1002/wrna.1600]. The first
two already carry in-repo citations in PRD §3. The literature places the discriminator *within*
the 3′ antiterminator domain and so defines no order between the two; the
``Discriminator`` < ``Antiterminator_Tbox_seq`` rank below is a measurement about **annotation
extents**, not a claim about domain order.

Pure-stdlib import surface; pandas/numpy are imported lazily so the module stays
bare-importable in the CI test env (matching ``scripts/measure_overlap_prevalence.py``).

Usage::

    python scripts/measure_element_order.py \
        --corpus data/processed/master_clean_v0.parquet \
        --out tests/fixtures/element_order/element_order_prevalence.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

# Both inserts exist so the documented command works **as written**, from a bare checkout with
# no editable install: `scripts/` for the sibling import below, and `src/` because
# :func:`measure` reads ``tbox_finder.labels.ELEMENT_COORDS``. The env that actually holds this
# corpus (``tbox-finder-data``) has no editable install, so without the second insert the usage
# line above dies with ModuleNotFoundError after the argument parse — verified, not assumed.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR.parent / "src"))

from measure_overlap_prevalence import element_intervals  # noqa: E402

SCHEMA_VERSION = "1.0"
STEP = "P3-13"


#: Repo root, for recording paths relative to it. The report is **committed**, so an absolute
#: path would leak this machine's layout (OS user name, worktree location) into a public repo
#: and make the artifact un-diffable across machines — the exact defect a P3-10 review caught
#: in a committed report. A path outside the repo is recorded as given rather than as a pile of
#: ``..`` segments.
_REPO_ROOT = _SCRIPTS_DIR.parent


def _repo_relative(path: Path) -> str:
    if not path.is_absolute():
        return str(path)  # already repo-relative by the documented invocation
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        # Outside the checkout (a linked worktree reading the main clone's DVC data, say).
        # The sha256 beside this field is the artifact's real identity; the absolute path is
        # dropped rather than committed.
        return f"<outside-repo>/{path.name}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_order(df, columns: dict[str, tuple[str, str]]) -> dict:
    """Pure core: per-element midpoint summary + all-pairs before/after prevalence.

    Kept separate from I/O so it is unit-testable on a small hand-built frame without the
    gitignored corpus.
    """
    import numpy as np  # noqa: PLC0415 (lazy — keep module bare-importable)

    extents = element_intervals(df, columns)
    mid = {name: (lo + hi) / 2.0 for name, (lo, hi, _) in extents.items()}
    present = {name: mask for name, (_, _, mask) in extents.items()}

    n_records = int(len(df))
    elements = []
    for name in columns:
        mask = present[name]
        n = int(mask.sum())
        elements.append(
            {
                "element": name,
                "n_annotated": n,
                "pct_annotated": (round(100.0 * n / n_records, 2) if n_records else 0.0),
                "median_midpoint": (float(np.median(mid[name][mask])) if n else None),
            }
        )

    pairs = []
    for a, b in itertools.combinations(columns, 2):
        both = present[a] & present[b]
        n_both = int(both.sum())
        record = {
            "A": a,
            "B": b,
            "n_both": n_both,
            "n_a_before_b": 0,
            "n_b_before_a": 0,
            "n_tied": 0,
            "frac_a_before_b": None,
        }
        if n_both:
            ma, mb = mid[a][both], mid[b][both]
            n_ab = int(np.count_nonzero(ma < mb))
            n_ba = int(np.count_nonzero(mb < ma))
            record.update(
                n_a_before_b=n_ab,
                n_b_before_a=n_ba,
                n_tied=n_both - n_ab - n_ba,
                frac_a_before_b=round(n_ab / n_both, 6),
            )
        pairs.append(record)

    return {"n_records": n_records, "elements": elements, "pairs": pairs}


def derive_order(report: dict) -> tuple[str, ...]:
    """5′→3′ element order derived from a report's **pairwise** matrix.

    Each element is ranked by how many of its pairwise comparisons it wins — A precedes B
    when a **strict majority** of the records annotating both put A's midpoint first, i.e.
    ``2 * n_a_before_b > n_both``. That is a Copeland score over the measured tournament, so
    the order follows the per-pair evidence rather than the medians (which are aggregated
    across records with different origins and are reported for orientation only).

    A majority, **not a plurality**, and the difference is only invisible while ties are rare.
    ``n_a_before_b > n_b_before_a`` is the same test whenever ``n_tied == 0``, but the third
    outcome is real here — two elements share a midpoint whenever one is a single-nucleotide
    annotation centred in the other, and the committed corpus does record ties (13 for
    ``Stem_I``/``Specifier``, 1 for ``Antiterminator_Tbox_seq``/``Discriminator``). On a
    40 / 39 / 21 split the plurality rule awards A a pairwise win on **40 %** support and the
    order that comes out is one most co-annotating records do not attest. The two rules agree
    on all 21 pairs of the committed measurement, so nothing shipped moves; the exposure is a
    re-measurement on a different corpus — a different GTDB release, the P5 scan's own calls —
    silently deriving a rank the stated rule does not support.

    Refuses to return an order that is not a strict total one: a pair with no strict majority
    (a tie, or a three-way split where the leader is short of half), a pair with no
    co-annotating record, or a tied Copeland score leaves the rank of at least two elements
    undetermined, and a silently-arbitrary tie-break would put an unmeasured ordering into the
    strand-resolver. This is also the check that the constant in
    :mod:`tbox_finder.infer.strand` is re-derived from, so a corpus that stopped supporting a
    total order would fail the unit test rather than quietly re-order one element.
    """
    names = [e["element"] for e in report["elements"]]
    wins = dict.fromkeys(names, 0)

    # The pair set must be the COMPLETE tournament, each pair exactly once. Copeland scores are
    # only a ranking if every element met every other: with a duplicated A-vs-B and no A-vs-C,
    # A scores 2, B 1, C 0 — all distinct, so the strict-total-order check below passes and an
    # order is returned for a pair that was never measured. Cheap to state, and this function
    # decides the constant that orients every reported locus.
    expected_pairs = {frozenset(pair) for pair in itertools.combinations(names, 2)}
    seen_pairs: set[frozenset] = set()

    for pair in report["pairs"]:
        a, b, n_both = pair["A"], pair["B"], pair["n_both"]
        if a not in wins or b not in wins:
            unknown = sorted({a, b} - set(wins))
            raise ValueError(f"pair names an element absent from elements[]: {unknown}")
        if a == b:
            raise ValueError(f"{a} is paired with itself; that is not a comparison")
        key = frozenset((a, b))
        if key in seen_pairs:
            raise ValueError(f"{a} vs {b} appears more than once; it would be counted twice")
        seen_pairs.add(key)
        if n_both <= 0:
            raise ValueError(
                f"{a} vs {b}: no record annotates both, so their 5′→3′ order is unmeasured"
            )
        n_ab, n_ba = pair["n_a_before_b"], pair["n_b_before_a"]
        # ``n_both`` is now the DENOMINATOR of the award, not just a presence check, so it has
        # to be a real one. A report where the directional counts overrun it (or go negative)
        # would make ``2 * n_ab > n_both`` true on arithmetic rather than on evidence.
        if n_ab < 0 or n_ba < 0 or n_ab + n_ba > n_both:
            raise ValueError(
                f"{a} vs {b}: {n_ab} + {n_ba} directional records do not fit in {n_both} "
                "co-annotating ones, so the majority denominator is not the one measured"
            )
        if 2 * n_ab > n_both:
            wins[a] += 1
        elif 2 * n_ba > n_both:
            wins[b] += 1
        else:
            raise ValueError(
                f"{a} vs {b}: no strict majority over {n_both} co-annotating records "
                f"({n_ab} before, {n_ba} after, {n_both - n_ab - n_ba} tied) — order undecided. "
                "A plurality is not a majority: the leader here is attested by "
                f"{100.0 * max(n_ab, n_ba) / n_both:.1f} % of the records that saw both"
            )
    if seen_pairs != expected_pairs:
        missing = sorted(tuple(sorted(p)) for p in expected_pairs - seen_pairs)
        raise ValueError(f"the pairwise matrix is incomplete; unmeasured pair(s): {missing}")
    if len(set(wins.values())) != len(names):
        raise ValueError(f"pairwise scores do not give a strict total order: {wins}")
    return tuple(sorted(names, key=lambda n: -wins[n]))


def measure(corpus: Path) -> dict:
    """Read the corpus parquet and compute the element-order report with provenance."""
    import pandas as pd  # noqa: PLC0415 (lazy — keep module bare-importable)

    from tbox_finder.labels import ELEMENT_COORDS  # noqa: PLC0415

    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus} — `dvc pull` the P0-12 artifact first")
    df = pd.read_parquet(corpus)  # compute_order validates columns fail-loud
    report = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "prd": "§6 (strand resolution), §8 (label derivation)",
        "adr": "ADR-0005 D15 (strand resolver)",
        "corpus": _repo_relative(corpus),
        "corpus_sha256": _sha256(corpus),
        "element_columns": {k: list(v) for k, v in ELEMENT_COORDS.items()},
        "statistic": "median of (start + end) / 2 per record; pairwise comparison within record",
        **compute_order(df, ELEMENT_COORDS),
    }
    report["derived_order_5p_to_3p"] = list(derive_order(report))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/master_clean_v0.parquet"),
        help="curated corpus parquet (P0-12)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/element_order/element_order_prevalence.json"),
        help="destination measurement JSON (committed; the unit test re-derives the order from it)",
    )
    args = p.parse_args(argv)

    # Never let --out alias --corpus: writing the JSON report would destroy the input parquet.
    if args.out.resolve() == args.corpus.resolve() or (
        args.out.exists() and args.corpus.exists() and args.out.samefile(args.corpus)
    ):
        p.error("--out must not alias --corpus (would overwrite the input artifact)")

    report = measure(args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out}")
    print("5′→3′ order: " + " < ".join(report["derived_order_5p_to_3p"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
