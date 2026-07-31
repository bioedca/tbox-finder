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


def _strip_inline_comment(line: str) -> str:
    """The executable part of a shell line — everything before an unquoted `#`.

    Quote-aware on purpose: a bare `line.split("#")` would truncate at a `#` inside a
    quoted string (a heredoc's python, a URL), which is the kind of silent mis-parse that
    makes a test look stricter than it is.
    """
    out: list[str] = []
    quote: str | None = None
    for j, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (j == 0 or line[j - 1].isspace()):
            break
        out.append(ch)
    return "".join(out)


def _first_executable_line(sbatch: Path, needle: str) -> int | None:
    """Index of the first sbatch line whose EXECUTABLE part contains `needle`.

    Comments are excluded twice over: a whole-line comment, and — the case that actually
    bit — a *trailing* comment on a real command. `export OMP_NUM_THREADS=2  # …torchrun
    warns…` is not a comment line, so the previous substring test matched it and reported
    the training invocation ~34 lines earlier than it is; the ordering assertion below then
    held for the wrong reason and would have false-alarmed on a restore placed between the
    two. Matching on the executable part is also why the needle stays a substring rather
    than a prefix: the real call is `PYTHONHASHSEED=0 PYTHONPATH=src torchrun …`, which no
    `startswith("torchrun")` would ever find. CodeRabbit, r8.
    """
    for i, line in enumerate(sbatch.read_text().splitlines()):
        if line.lstrip().startswith("#"):
            continue
        if needle in _strip_inline_comment(line):
            return i
    return None


_VAR_ASSIGN = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"\s*$', re.M)
_VAR_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def _expand_shell_vars(body: str) -> str:
    """Substitute the sbatch's own `VAR="literal"` assignments through its text.

    The run signals are written as `"$OUT_DIR/train_stage1_gate4_twin.DONE"`, so their whole
    path never appears literally and a basename test was the only thing on offer. That test
    has a real blind spot: `.gitignore` rules for these signals are **path-anchored**
    (`/reports/p2/…`), so re-pointing `OUT_DIR` moves every signal out from under its ignore
    rule while the basenames — and the test — stay exactly the same. Expanding first lets the
    registry be compared as whole paths, the way the `--output`/`--error` directives already
    are. CodeRabbit, r8.
    """

    def _sub(text: str, env: dict[str, str]) -> str:
        # `env` is a parameter, not a closed-over loop variable: binding it late would make
        # every pass substitute from whichever dict the loop happened to end on (ruff B023).
        return _VAR_REF.sub(lambda m: env.get(m.group(1), m.group(0)), text)

    env = dict(_VAR_ASSIGN.findall(body))
    for _ in range(5):  # resolve nesting; a fixed point in one or two passes in practice
        resolved = {k: _sub(v, env) for k, v in env.items()}
        if resolved == env:
            break
        env = resolved
    return _sub(body, env)


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

        expanded = _expand_shell_vars(body)
        for signal in sb.run_signals:
            if Path(signal).suffix in (".out", ".err"):
                continue  # already compared as a whole path above
            assert signal in expanded, (
                f"{sb.name}.sbatch never writes the whole path {signal!r} (basename "
                f"{Path(signal).name!r} alone is not enough: the ignore rules are "
                "path-anchored, so a re-pointed $OUT_DIR leaves the real signals un-ignored "
                "while a basename test stays green) — the registry has drifted."
            )
        assert sb.committed_report in expanded, (
            f"{sb.name}.sbatch never writes the whole path {sb.committed_report!r} — the "
            "committed deliverable this test protects is not the one the job writes."
        )


# ══════════════════════════════════════════════════════════════════════════════
# The two parsers the assertions above lean on. A test whose helper silently
# matches the wrong line is weaker than it reads, and the way that stays
# invisible is never testing the helper ([[tests-can-specify-the-bug]]).
# ══════════════════════════════════════════════════════════════════════════════
def test_first_executable_line_ignores_a_trailing_comment_mention(tmp_path):
    """The case that actually bit: a real command whose *inline comment* names the needle."""
    sbatch = tmp_path / "x.sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "# torchrun is mentioned in this whole-line comment\n"
        "export OMP_NUM_THREADS=2   # 16 cpus / 8 ranks; torchrun warns and defaults to 1.\n"
        "PYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \\\n"
    )
    assert _first_executable_line(sbatch, "torchrun") == 3  # the invocation, not the export


def test_first_executable_line_matches_a_command_that_is_not_the_line_prefix(tmp_path):
    """`torchrun` is reached through an env prefix, so a prefix match would find nothing."""
    sbatch = tmp_path / "x.sbatch"
    sbatch.write_text("#!/bin/bash\nPYTHONHASHSEED=0 PYTHONPATH=src torchrun --standalone\n")
    assert _first_executable_line(sbatch, "torchrun") == 1
    assert _first_executable_line(sbatch, "nothing-here") is None


def test_strip_inline_comment_keeps_a_hash_inside_quotes():
    """A bare split on '#' would truncate a quoted hash and quietly shorten the match text."""
    assert _strip_inline_comment('echo "a # b"   # real comment').strip() == 'echo "a # b"'
    assert _strip_inline_comment("cmd 'x #y'").strip() == "cmd 'x #y'"
    assert _strip_inline_comment("cmd   # tail").strip() == "cmd"


def test_expand_shell_vars_resolves_the_declared_signal_paths():
    """`$OUT_DIR/x.DONE` must compare as a whole path — the ignore rules are path-anchored."""
    body = 'OUT_DIR="reports/p2"\nDONE="$OUT_DIR/train_stage1_gate4_twin.DONE"\n'
    expanded = _expand_shell_vars(body)
    assert "reports/p2/train_stage1_gate4_twin.DONE" in expanded
    # …and the blind spot it closes: re-pointing OUT_DIR must change the compared path.
    moved = _expand_shell_vars(body.replace('OUT_DIR="reports/p2"', 'OUT_DIR="logs/p2"'))
    assert "reports/p2/train_stage1_gate4_twin.DONE" not in moved
    assert "logs/p2/train_stage1_gate4_twin.DONE" in moved


def test_expand_shell_vars_leaves_runtime_only_variables_alone():
    """`$SLURM_JOB_ID` has no literal assignment; it must survive rather than vanish."""
    assert "$SLURM_JOB_ID" in _expand_shell_vars('B="/tmp/$USER-$SLURM_JOB_ID"\n')
