"""P2-10d′-e — train_production.sbatch: RESTORE the tracked report, and IGNORE its run signals.

Two deterministic-sibling defects of the DDP report-write race, both fixable in the sbatch and
`.gitignore` rather than in the training code:

1. **Restore, never delete.** A successful run overwrites the git-tracked report in place, so
   run #2's `build_report` git snapshot (on ALL 8 ranks) reads that leftover dirt as modified
   code outside `_DATA_STAGING_PREFIXES` and re-derives `provenance_complete` FALSE on every
   rank — a deterministic ~20 GPU-h loss, no race needed. The sbatch must `git checkout HEAD`
   the report before training (the P2-10d′-c ruling forbids `rm`, which dirties the tree too).

2. **Ignore the run signals.** Every sibling job ignores its own DONE / lock / SLURM logs;
   train_production did not, so a `git add -A` on the cluster checkout would commit them. The
   committed report itself must stay tracked.

Pure-subprocess `git` — runs in bare CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SBATCH = REPO / "slurm" / "p2" / "train_production.sbatch"

# The tracked deliverable (NOT a run signal) and the four run signals it must ignore.
_COMMITTED_REPORT = "reports/p2/train_stage1_production.json"
_RUN_SIGNALS = [
    "reports/p2/train_stage1_production.DONE",
    "reports/p2/.train_production.lock",
    "reports/p2/train_production_678.out",  # the %j-stamped SLURM logs
    "reports/p2/train_production_678.err",
]


def _is_ignored(rel_path: str) -> bool:
    """True iff a .gitignore rule matches (exit 0 = ignored).

    `--no-index` evaluates the ignore RULES regardless of whether the path is tracked. Without
    it, `git check-ignore` reports a tracked path as *not* ignored even when a rule matches — so
    a run signal that got accidentally committed would slip past `test_every_..._gitignored`,
    and `test_the_committed_report_is_not_gitignored` would answer "not ignored" for the wrong
    reason (it is tracked) rather than the right one (no rule matches it). CodeRabbit, r1.
    """
    return (
        subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", "--no-index", "-q", rel_path],
            capture_output=True,
        ).returncode
        == 0
    )


def _first_executable_line(needle: str) -> int | None:
    """Index of the first non-comment sbatch line with `needle` (None if only in a comment)."""
    for i, line in enumerate(SBATCH.read_text().splitlines()):
        if needle in line and not line.lstrip().startswith("#"):
            return i
    return None


def test_the_sbatch_restores_the_tracked_report_to_head() -> None:
    """The report is restored (never rm'd) BEFORE training, so re-runs start from a clean tree.

    Line-based, not a bare substring: the restore must be an EXECUTABLE line (a comment
    mentioning it does not restore anything) and must precede the training invocation (a restore
    placed after `torchrun` cleans the tree too late — build_report has already snapshotted).
    CodeRabbit, r1.
    """
    restore = _first_executable_line('git checkout HEAD -- "$REPORT"')
    train = _first_executable_line("torchrun")
    assert restore is not None, (
        "train_production.sbatch must RESTORE $REPORT to HEAD as an executable line before "
        "training — otherwise run #2's build_report snapshot reads the leftover-dirty tracked "
        "report as modified code and fails provenance_complete on all 8 ranks (P2-10d′-e)."
    )
    assert train is not None, "expected a torchrun training invocation in train_production.sbatch."
    assert restore < train, (
        "the $REPORT restore must run BEFORE torchrun, or the tree is still dirty at $REPORT when "
        "build_report snapshots git status on all 8 ranks (P2-10d′-e)."
    )


def test_the_sbatch_never_deletes_the_tracked_report() -> None:
    """`rm`-ing $REPORT dirties the tree exactly like leaving it dirty does — restore instead.

    Belt-and-braces beside `test_sbatch_rm_targets.py`'s census: name the specific regression.
    """
    body = SBATCH.read_text()
    for token in ('rm -f "$REPORT"', 'rm "$REPORT"', 'rm -rf "$REPORT"'):
        assert token not in body, (
            f"train_production.sbatch must not delete the git-tracked report ({token!r}); "
            "deleting it dirties the tree outside _DATA_STAGING_PREFIXES too (P2-10d′-c)."
        )


def test_every_production_run_signal_is_gitignored() -> None:
    """Each of DONE / lock / .out / .err is ignored — a `git add -A` cannot commit a run signal."""
    not_ignored = [s for s in _RUN_SIGNALS if not _is_ignored(s)]
    assert not not_ignored, (
        "these train_production run signals are not gitignored, so `git add -A` on the cluster "
        "checkout would commit them (every sibling job ignores its own): " + ", ".join(not_ignored)
    )


def test_the_committed_report_is_not_gitignored() -> None:
    """The deliverable stays tracked — ignoring it would silently drop the run's evidence."""
    assert not _is_ignored(
        _COMMITTED_REPORT
    ), f"{_COMMITTED_REPORT} is the committed deliverable and must remain tracked, not ignored."
