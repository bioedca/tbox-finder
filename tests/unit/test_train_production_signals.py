"""P2-10d′-e — the DDP training sbatches: RESTORE the tracked report, and IGNORE the run signals.

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

Both properties are checked for **every** DDP training sbatch, not just the one whose defect
named them: P2-14's `train_gate4_twin.sbatch` is a clone of `train_production.sbatch` and
arrived with its four run signals un-ignored (CodeRabbit, r1). A per-file copy of these tests
is how one gets fixed and the other ships the bug, so the cases are parametrised over a
registry and a new sbatch is added by adding a row.

Pure-subprocess `git` — runs in bare CI.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainSbatch:
    """One DDP training sbatch: its file, its committed deliverable, and its run signals."""

    name: str
    path: Path
    committed_report: str
    run_signals: tuple[str, ...]


_JOB = "678"  # a concrete %j stamp — `check-ignore` needs a literal path, not a glob

SBATCHES = [
    TrainSbatch(
        name="train_production",
        path=REPO / "slurm" / "p2" / "train_production.sbatch",
        committed_report="reports/p2/train_stage1_production.json",
        run_signals=(
            "reports/p2/train_stage1_production.DONE",
            "reports/p2/.train_production.lock",
            f"reports/p2/train_production_{_JOB}.out",
            f"reports/p2/train_production_{_JOB}.err",
        ),
    ),
    TrainSbatch(
        name="train_gate4_twin",
        path=REPO / "slurm" / "p2" / "train_gate4_twin.sbatch",
        committed_report="reports/p2/train_stage1_gate4_twin.json",
        run_signals=(
            "reports/p2/train_stage1_gate4_twin.DONE",
            "reports/p2/.train_gate4_twin.lock",
            f"reports/p2/train_gate4_twin_{_JOB}.out",
            f"reports/p2/train_gate4_twin_{_JOB}.err",
        ),
    ),
]

_IDS = [s.name for s in SBATCHES]


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


def _first_executable_line(sbatch: Path, needle: str) -> int | None:
    """Index of the first non-comment sbatch line with `needle` (None if only in a comment)."""
    for i, line in enumerate(sbatch.read_text().splitlines()):
        if needle in line and not line.lstrip().startswith("#"):
            return i
    return None


@pytest.mark.parametrize("sb", SBATCHES, ids=_IDS)
def test_the_sbatch_restores_the_tracked_report_to_head(sb: TrainSbatch) -> None:
    """The report is restored (never rm'd) BEFORE training, so re-runs start from a clean tree.

    Line-based, not a bare substring: the restore must be an EXECUTABLE line (a comment
    mentioning it does not restore anything) and must precede the training invocation (a restore
    placed after `torchrun` cleans the tree too late — build_report has already snapshotted).
    CodeRabbit, r1.
    """
    restore = _first_executable_line(sb.path, 'git checkout HEAD -- "$REPORT"')
    train = _first_executable_line(sb.path, "torchrun")
    assert restore is not None, (
        f"{sb.name}.sbatch must RESTORE $REPORT to HEAD as an executable line before "
        "training — otherwise run #2's build_report snapshot reads the leftover-dirty tracked "
        "report as modified code and fails provenance_complete on all 8 ranks (P2-10d′-e)."
    )
    assert train is not None, f"expected a torchrun training invocation in {sb.name}.sbatch."
    assert restore < train, (
        "the $REPORT restore must run BEFORE torchrun, or the tree is still dirty at $REPORT when "
        "build_report snapshots git status on all 8 ranks (P2-10d′-e)."
    )


@pytest.mark.parametrize("sb", SBATCHES, ids=_IDS)
def test_the_sbatch_never_deletes_the_tracked_report(sb: TrainSbatch) -> None:
    """`rm`-ing $REPORT dirties the tree exactly like leaving it dirty does — restore instead.

    Belt-and-braces beside `test_sbatch_rm_targets.py`'s census: name the specific regression.
    """
    body = sb.path.read_text()
    for token in ('rm -f "$REPORT"', 'rm "$REPORT"', 'rm -rf "$REPORT"'):
        assert token not in body, (
            f"{sb.name}.sbatch must not delete the git-tracked report ({token!r}); "
            "deleting it dirties the tree outside _DATA_STAGING_PREFIXES too (P2-10d′-c)."
        )


@pytest.mark.parametrize("sb", SBATCHES, ids=_IDS)
def test_every_run_signal_is_gitignored(sb: TrainSbatch) -> None:
    """Each of DONE / lock / .out / .err is ignored — a `git add -A` cannot commit a run signal."""
    not_ignored = [s for s in sb.run_signals if not _is_ignored(s)]
    assert not not_ignored, (
        f"these {sb.name} run signals are not gitignored, so `git add -A` on the cluster "
        "checkout would commit them (every sibling job ignores its own): " + ", ".join(not_ignored)
    )


@pytest.mark.parametrize("sb", SBATCHES, ids=_IDS)
def test_the_committed_report_is_not_gitignored(sb: TrainSbatch) -> None:
    """The deliverable stays tracked — ignoring it would silently drop the run's evidence."""
    assert not _is_ignored(sb.committed_report), (
        f"{sb.committed_report} is {sb.name}'s committed deliverable and must remain tracked, "
        "not ignored."
    )


def _sbatch_log_directives(body: str) -> dict[str, str]:
    """`{"output": <path>, "error": <path>}` from the `#SBATCH --output=/--error=` headers."""
    found: dict[str, str] = {}
    for line in body.splitlines():
        m = re.match(r"#SBATCH\s+--(output|error)[=\s]+(\S+)", line.strip())
        if m and m.group(1) not in found:
            found[m.group(1)] = m.group(2)
    return found


def test_the_run_signal_paths_are_the_ones_the_sbatch_actually_writes() -> None:
    """The registry above is evidence only if it names the sbatch's OWN paths.

    Without this, a typo'd or drifted entry (a renamed DONE marker, a re-pointed `--output`)
    makes every case above pass while the real signals stay un-ignored — the registry would be
    self-certifying rather than checking the file.

    The SLURM logs are compared as WHOLE PATHS, not stems: `--output=logs/train_gate4_twin_%j.out`
    still contains the stem `train_gate4_twin`, so a stem check would bless a registry claiming
    `reports/p2/…` while the job writes somewhere the `/reports/p2/…` ignore rule cannot reach —
    the exact drift this test exists to catch. CodeRabbit, r2.
    """
    for sb in SBATCHES:
        body = sb.path.read_text()

        directives = _sbatch_log_directives(body)
        assert set(directives) == {"output", "error"}, (
            f"{sb.name}.sbatch must declare both #SBATCH --output and --error; found "
            f"{sorted(directives)}."
        )
        registered = {
            Path(s).suffix.lstrip("."): s
            for s in sb.run_signals
            if Path(s).suffix in (".out", ".err")
        }
        for stream, suffix in (("output", "out"), ("error", "err")):
            declared = directives[stream].replace("%j", _JOB)
            assert declared == registered.get(suffix), (
                f"{sb.name}.sbatch writes its {stream} to {directives[stream]!r} but the registry "
                f"locks {registered.get(suffix)!r}. The ignore rule is path-anchored, so a "
                "re-pointed log directory leaves the real logs un-ignored while these tests stay "
                "green."
            )

        for signal in sb.run_signals:
            if Path(signal).suffix in (".out", ".err"):
                continue  # already compared as a whole path above
            assert Path(signal).name in body, (
                f"{sb.name}.sbatch never mentions {Path(signal).name!r}, so the run signal "
                f"{signal!r} this test claims to lock is not a path that sbatch writes — the "
                "registry has drifted."
            )
        assert Path(sb.committed_report).name in body, (
            f"{sb.name}.sbatch never mentions {Path(sb.committed_report).name!r} — the committed "
            "deliverable this test protects is not the one the job writes."
        )
