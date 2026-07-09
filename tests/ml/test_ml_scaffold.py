"""Scaffold placeholder for the ML-behavioral gate suite (CLAUDE.md §8.2-8.4).

Keeps ``pytest tests/ml`` green (exit 0, not exit 5 "no tests collected") until
the load-bearing gates land and become blocking here:
  - the no-leakage test (§8.2, P0-24) — no cluster/taxon spans a split boundary,
    run over the committed full-corpus split-assignment table;
  - the eval-gate regression (§8.4, P0-31) — the smoke model + fixture verify the
    PRD §2 metrics (element-F1, boundary IoU, AUPRC, ECE, recall@matched-precision).
Delete this module when a real ``tests/ml/test_*.py`` is added.
"""

from pathlib import Path


def test_ml_gate_scaffold_present() -> None:
    # tests/ml/ is the anchor for the §8.2 no-leakage + §8.4 eval-gate tests.
    # This trivial check is replaced by those gates as P0-24 / P0-31 land.
    assert Path(__file__).parent.is_dir()
