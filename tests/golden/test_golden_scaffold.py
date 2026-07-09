"""Scaffold placeholder for the golden-file regression suite (CLAUDE.md §8.1).

Keeps ``pytest tests/golden`` green (exit 0, not exit 5 "no tests collected")
until the first real per-stage golden fixture lands with the data pipeline: a
50-200-record input under ``tests/fixtures/`` plus a committed ``expected.sha256``
that CI re-verifies. Delete this module when a real ``tests/golden/test_*.py``
is added.
"""

from pathlib import Path


def test_golden_suite_scaffold_present() -> None:
    # tests/golden/ is the anchor for committed per-stage expected.sha256
    # fixtures (CLAUDE.md §8.1). This trivial check is replaced by real
    # per-stage hash-diff regressions as the data pipeline lands.
    assert Path(__file__).parent.is_dir()
