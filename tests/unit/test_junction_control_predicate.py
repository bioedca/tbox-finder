"""P2-10d′-e — the last pandas-dialect idiom, in `scripts/measure_junction_control.py`.

The script sizes its spliced arm by counting how many decoys `embed_decoy_rows` would embed.
It used to do that with `str(row.get("source_record_id") or "")`, which is
pandas-version-dependent: under pandas 3 the default string dtype's missing sentinel is `NaN`
(truthy), so a parentless decoy read as the literal parent `"nan"` — in no fold set — and the
arm was sized at 702 while `embed_decoy_rows` (already on `row_text`) admitted 3,701, dying
with "702 hosts cannot carry 703" under `unique_hosts=True` on the training env (P2-10d′-c).

`decoy_sizes_embedded_arm` now reads every cell through `masking.row_text`. These assertions
pass a LITERAL `float("nan")` (dialect-free, the `test_masking.py` house idiom), so they bite
under whichever pandas is installed and pin the fix rather than the ambient dtype.

The script imports torch-free (`embedding`/`negatives`/`junction_probe` are numpy-only at
module scope), so this runs in bare CI — loaded by path because `scripts/` is not a package.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "measure_junction_control.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("measure_junction_control", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MJC = _load_script()
POOL = MJC.TRAINING_DECOY_POOLS[0]  # a real embeddable pool name


def test_a_parentless_decoy_is_sized_in_not_read_as_a_parent_named_nan() -> None:
    """The exact failure mode in one assertion: a NaN parent must not look present.

    Under the old `str(... or "")` idiom this row's parent read as `"nan"` (pandas 3),
    excluded it from the count, and shrank the arm 3,701 → 702. `row_text(NaN) == ""`, so the
    parent is absent and the decoy is sized in regardless of the fold set.
    """
    row = {"pool": POOL, "masked": False, "source_record_id": float("nan")}
    assert MJC.decoy_sizes_embedded_arm(row, set()) is True


def test_a_none_parent_is_also_parentless() -> None:
    row = {"pool": POOL, "masked": False, "source_record_id": None}
    assert MJC.decoy_sizes_embedded_arm(row, set()) is True


def test_a_real_parent_must_be_in_the_training_fold() -> None:
    """A parented decoy (dinuc_shuffled) is admitted only when its parent is in-fold."""
    in_fold = {"pool": POOL, "masked": False, "source_record_id": "rec_A"}
    out_fold = {"pool": POOL, "masked": False, "source_record_id": "rec_B"}
    assert MJC.decoy_sizes_embedded_arm(in_fold, {"rec_A"}) is True
    assert MJC.decoy_sizes_embedded_arm(out_fold, {"rec_A"}) is False


def test_masked_and_excluded_pools_are_not_sized_in() -> None:
    masked = {"pool": POOL, "masked": True, "source_record_id": None}
    wrong_pool = {"pool": "gc_background", "masked": False, "source_record_id": None}
    assert MJC.decoy_sizes_embedded_arm(masked, set()) is False
    # gc_background is a real pool but not in TRAINING_DECOY_POOLS (it is an excluded pool).
    assert "gc_background" not in MJC.TRAINING_DECOY_POOLS
    assert MJC.decoy_sizes_embedded_arm(wrong_pool, set()) is False


def test_a_nan_pool_cell_is_not_read_as_a_pool_named_nan() -> None:
    """The other nullable read on the same predicate: a NaN pool must not match either."""
    row = {"pool": float("nan"), "masked": False, "source_record_id": None}
    assert MJC.decoy_sizes_embedded_arm(row, set()) is False


def test_the_predicate_uses_row_text_not_the_pandas_dialect_idiom() -> None:
    """Guard the fix at the source: the `str(... or "")` idiom must not creep back in.

    Parse the function and inspect only its EXECUTABLE statements (drop the docstring, which
    legitimately quotes the banned idiom to explain it) so the tripwire is dialect-proof.
    """
    tree = ast.parse(_SCRIPT.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "decoy_sizes_embedded_arm"
    )
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # strip the docstring node
    code = "\n".join(ast.unparse(stmt) for stmt in body)
    assert "row_text(" in code, "the predicate must read cells through masking.row_text"
    assert (
        'or ""' not in code
    ), 'the pandas-dialect idiom `str(... or "")` reintroduces the pandas-3 NaN bug (P2-10d′-e).'
    assert "str(" not in code, (
        "the predicate must read cells via row_text, never str() — str(NaN) is 'nan' under "
        "pandas 3 (P2-10d′-e)."
    )
