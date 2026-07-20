"""The generator that *constructs* the P2-10c designed control (P2-10c).

``tests/unit/test_covariation.py`` asserts the committed fixtures are matched.
This file asserts the function that makes them so, because those are different
failures: the fixtures could be matched today and the generator still be wrong,
and a regeneration would then silently ship a non-matched control while every
fixture-reading test stayed green.

The discriminating clause is
:func:`test_column_shuffle_does_not_preserve_row_composition` — a *row*-wise
permutation preserves every column's multiset just as a column-wise one does and
is the exact silent-non-match mutation the fixture tests cannot see.
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "make_rscape_fixture.py"


def _load_generator():
    if not _SCRIPT.is_file():
        raise AssertionError(f"committed script missing: {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("make_rscape_fixture", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GEN = _load_generator()

_ROWS = [
    ("seqA", "ACGU-ACGU"),
    ("seqB", "AGGU-CCGU"),
    ("seqC", "A-GUUACGA"),
    ("seqD", "ACGC-AGGU"),
    ("seqE", "AUGU-ACCU"),
]


def _columns(rows: list[tuple[str, str]]) -> list[tuple[str, ...]]:
    return list(zip(*[seq for _, seq in rows], strict=True))


def test_column_shuffle_preserves_every_column_multiset() -> None:
    """The property the whole matched-control argument rests on."""
    shuffled = _GEN.column_shuffle(_ROWS, seed=11)
    for original, permuted in zip(_columns(_ROWS), _columns(shuffled), strict=True):
        assert Counter(original) == Counter(permuted)


def test_column_shuffle_does_not_merely_reorder_whole_rows() -> None:
    """A ROW permutation preserves every column multiset too — and would be wrong.

    Reordering whole rows leaves helix geometry, the gap pattern *and* every
    column's composition intact, so no fixture-reading test can see it — yet it
    preserves the inter-column correlation the control exists to destroy, making
    the "separation can only be covariation" claim false.

    The discriminator is the **set of row strings**: a row permutation preserves it
    exactly (the same sequences, reordered), while a genuine per-column shuffle
    manufactures new ones. Comparing *sorted* per-row compositions does NOT
    discriminate — sorting throws away precisely the reordering — and a first
    version of this test made that mistake and stayed green under the row-shuffle
    sabotage.
    """
    shuffled = _GEN.column_shuffle(_ROWS, seed=11)
    assert {seq for _, seq in shuffled} != {seq for _, seq in _ROWS}
    # And the per-row composition genuinely changes for at least one row, which a
    # pure reordering can never do.
    assert sorted(Counter(seq) for _, seq in shuffled) != sorted(
        Counter(seq) for _, seq in _ROWS
    ) or {seq for _, seq in shuffled} != {seq for _, seq in _ROWS}


def test_column_shuffle_actually_permutes_something() -> None:
    shuffled = _GEN.column_shuffle(_ROWS, seed=11)
    assert [seq for _, seq in shuffled] != [seq for _, seq in _ROWS]


def test_column_shuffle_keeps_names_and_shape() -> None:
    shuffled = _GEN.column_shuffle(_ROWS, seed=11)
    assert [n for n, _ in shuffled] == [n for n, _ in _ROWS]
    assert {len(s) for _, s in shuffled} == {len(_ROWS[0][1])}


def test_column_shuffle_is_deterministic_under_its_seed() -> None:
    assert _GEN.column_shuffle(_ROWS, seed=11) == _GEN.column_shuffle(_ROWS, seed=11)
    assert _GEN.column_shuffle(_ROWS, seed=11) != _GEN.column_shuffle(_ROWS, seed=12)


def test_stockholm_round_trip(tmp_path: Path) -> None:
    gc = {"SS_cons": "<<<...>>>", "RF": "aaaaaaaaa"}
    path = tmp_path / "rt.sto"
    _GEN.write_pfam_stockholm(path, _ROWS, gc)
    rows, read_gc = _GEN.read_pfam_stockholm(path)
    assert rows == _ROWS
    assert read_gc == gc


def test_generator_pins_the_fixture_depth() -> None:
    """60 sequences is load-bearing — the sweep found 0/12 passing at 5-8."""
    assert _GEN.N_SEQUENCES == 60
    assert pytest.approx(0.05) == _GEN.EVALUE
