"""Column-permuted matched control for a covariation alignment (promoted, shared home).

The R-scape covariation backend (:mod:`tbox_finder.mining.covariation`) can only be
*certified* — proven to answer, not merely to run — against a **matched control**: the
same alignment with every column independently permuted across rows. That permutation
holds each column's exact base-and-gap composition, the alignment width, the row depth,
and the ``#=GC SS_cons`` consensus structure fixed, and destroys only the correlation
*between* columns — i.e. the covariation signal and nothing else. A dead or powerless
backend cannot tell the two apart; a working one must (P2-10b: 13 covarying pairs on the
positive vs **0** on the shuffled twin).

These three helpers were first written in ``scripts/make_rscape_fixture.py`` to mint the
committed P2-10b fixtures. They are promoted here verbatim (byte-for-byte identical
output — the shuffle is ``random.Random(seed)`` deterministic) so the runtime
certification of the P2-10e homolog-MSA supply (:mod:`tbox_finder.mining.homolog_msa`)
and the fixture minter share **one** implementation rather than two that can drift
([[promote-dont-duplicate-is-a-correctness-rule]]). The module is stdlib-only and
pandas/torch-free, so it imports cleanly in every pinned env (tbox-homology / tbox-infernal
/ tbox-rscape).
"""

from __future__ import annotations

import random
from pathlib import Path

__all__ = [
    "read_pfam_alignment",
    "write_pfam_alignment",
    "shuffle_alignment_columns",
]


def read_pfam_alignment(source: str | Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Read a Stockholm alignment, concatenating interleaved (wrapped) blocks per row.

    Returns ``([(name, aligned_seq), ...], {gc_tag: value})`` in first-seen row order. Sequence
    rows and ``#=GC`` lines are accumulated by name / tag and joined across blocks, so BOTH a
    one-line-per-sequence (Pfam) alignment AND mlocarna's multi-block interleaved ``result.stk``
    — 24 rows wrapped over N column-blocks (e.g. widths 120/120/49) — parse to full-width rows.
    A naive "one line == one row" read instead yields ``rows × blocks`` ragged rows and crashes
    the column shuffle's ``zip(..., strict=True)``. The ``#=GC`` block (which carries ``SS_cons``)
    is captured separately so a shuffle can hold it fixed. ``source`` may be a path or the
    alignment text itself (a leading ``# STOCKHOLM`` marks text).

    For an already-unwrapped alignment each name / tag appears once, so the join is the identity
    and the output is byte-for-byte the pre-existing behaviour — the committed P2-10b fixtures are
    unwrapped, so their round-trip and seeded-shuffle locks are unchanged.
    """
    text = _read_text(source)
    seq_names: list[str] = []
    seq_frags: dict[str, list[str]] = {}
    gc_tags: list[str] = []
    gc_frags: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("#=GC "):
            _, tag, val = line.split(None, 2)
            if tag not in gc_frags:
                gc_frags[tag] = []
                gc_tags.append(tag)
            gc_frags[tag].append(val)
        elif line.startswith("#") or line.strip() in {"", "//"}:
            continue
        else:
            name, aseq = line.split(None, 1)
            if name not in seq_frags:
                seq_frags[name] = []
                seq_names.append(name)
            seq_frags[name].append(aseq.strip())
    seqs = [(name, "".join(seq_frags[name])) for name in seq_names]
    gc = {tag: "".join(gc_frags[tag]) for tag in gc_tags}
    return seqs, gc


def write_pfam_alignment(seqs: list[tuple[str, str]], gc: dict[str, str]) -> str:
    """Render ``seqs`` + ``gc`` as a one-line-per-sequence (Pfam-format) Stockholm string."""
    width = max([len(n) for n, _ in seqs] + [len("#=GC " + t) for t in gc])
    lines = ["# STOCKHOLM 1.0", ""]
    lines += [f"{name.ljust(width)} {aseq}" for name, aseq in seqs]
    lines += [f"{('#=GC ' + tag).ljust(width)} {val}" for tag, val in gc.items()]
    lines.append("//")
    return "\n".join(lines) + "\n"


def shuffle_alignment_columns(seqs: list[tuple[str, str]], *, seed: int) -> list[tuple[str, str]]:
    """Permute every alignment column independently across rows (the matched control).

    Names, row order, alignment width and every column's exact composition are preserved;
    only the inter-column correlation (the covariation signal) is destroyed. Per-column
    permutation also removes the *phylogenetic* correlation between rows, which *raises*
    R-scape's apparent power rather than lowering it — so zero covarying pairs on the
    shuffle is a stronger negative than a phylogeny-matched null would give.
    """
    rng = random.Random(seed)
    names = [n for n, _ in seqs]
    shuffled_columns = []
    for column in zip(*[aseq for _, aseq in seqs], strict=True):
        cells = list(column)
        rng.shuffle(cells)
        shuffled_columns.append(cells)
    return list(
        zip(names, ["".join(row) for row in zip(*shuffled_columns, strict=True)], strict=True)
    )


def _read_text(source: str | Path) -> str:
    """Accept a path or the alignment text itself."""
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if "\n" in source or source.lstrip().startswith("# STOCKHOLM"):
        return source
    return Path(source).read_text(encoding="utf-8")
