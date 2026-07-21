"""No shipped sbatch may delete a git-TRACKED path.

This gate exists because `slurm/p2/train_production.sbatch` cleared its stale outputs with::

    rm -f "$DONE" "$REPORT" "$CKPT_DIR/stage1.pt" "$CKPT_DIR/provenance.json"

and ``$REPORT`` — ``reports/p2/train_stage1_production.json`` — became a **git-tracked** file
at P2-09 (082f8c7). Deleting a tracked path leaves the working tree dirty at a path outside
``train_stage1._DATA_STAGING_PREFIXES = ("data/",)``. ``build_report`` snapshots
``git status`` **once**, while the deletion is live, so:

    ``_provenance_complete`` -> False -> ``overall_pass`` -> False -> ``torch.save`` SKIPPED
    -> ``RuntimeError`` -> rc != 0 -> the EXIT trap wipes the node-local build dir

i.e. the whole ~20 GPU-h run is destroyed **at the finish line**, after training completed,
and the only symptom before the fact is a line of shell nobody reads as dangerous. Measured
on the cluster: dirty=[staged parquet] -> ``provenance_complete`` True; dirty=[staged
parquet, deleted report] -> **False**.

Nothing caught it. ``sbatch --test-only`` validates SLURM resources, never the body;
``tests/unit/test_sbatch_overrides.py`` parses the *launch* line only; and the clause that
fires shipped in the same commit that made the report tracked, so no prior run exercised it.

The gate is deliberately about the *class*, not the one path: any sbatch that removes a
tracked file has the same effect on the same clause.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "slurm"

#: ``VAR=value`` / ``VAR="value"`` alone on a line — the assignment forms the shipped sbatch
#: bodies actually use. Command substitutions are not resolved (see `_resolve`).
_ASSIGN = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=(".*?"|\'.*?\'|\S*)\s*$')
#: A deletion. `rm` with any flags, capturing the rest of the (logical) command.
_RM = re.compile(r"(?:^|\s|;|&&|\|\|)rm\s+((?:-[A-Za-z]+\s+)*)(.+)$")
#: `$VAR` or `${VAR}`.
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _sbatch_files() -> list[Path]:
    return sorted(SLURM.rglob("*.sbatch"))


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """``(first physical line number, joined command)``, backslash-continuations merged.

    Shell commands are not physical lines. A file-wide scan that reads them as such misses::

        rm -f \\
          "$REPORT"

    which deletes exactly what the un-continued form does. Joining first is what makes the
    gate about *commands* rather than about formatting (CodeRabbit, P2-10d′-c r1).
    """
    out: list[tuple[int, str]] = []
    buf = ""
    start: int | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if start is None:
            start = lineno
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        out.append((start, buf + stripped))
        buf, start = "", None
    if buf:
        out.append((start or 1, buf))
    return out


def _resolve(value: str, env: dict[str, str]) -> str:
    """Expand known vars; leave unknown ones as an UNRESOLVED marker.

    Unresolved is deliberately *not* silently dropped: a path this parser cannot resolve
    must not masquerade as a safe literal, so it is reported rather than skipped.
    """
    for _ in range(5):  # bounded: assignments here nest at most a level or two
        new = _VAR.sub(lambda m: env.get(m.group(1) or m.group(2), "\x00UNRESOLVED"), value)
        if new == value:
            break
        value = new
    return value


def _tracked_paths() -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {p for p in out.split("\0") if p}


def _rm_targets(text: str) -> list[tuple[int, str]]:
    """``(line number, resolved path)`` for every argument of every `rm`, in EXECUTION ORDER.

    The environment is built **as the scan advances**, so each `rm` is resolved against the
    values its variables hold *at that point*. Resolving against a whole-file dict of final
    values lets a later reassignment hide an earlier deletion::

        REPORT="reports/p2/train_stage1_production.json"
        rm -f "$REPORT"          # deletes a TRACKED file …
        REPORT="/tmp/scratch"    # … but a file-wide dict scores it as /tmp/scratch

    (CodeRabbit, P2-10d′-c r1.)
    """
    env: dict[str, str] = {}
    targets: list[tuple[int, str]] = []
    for line_no, line in _logical_lines(text):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # a comment, not a command — the fix's own rationale lives in one
        assign = _ASSIGN.match(line)
        if assign:
            raw = assign.group(2)
            quoted = len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]
            env[assign.group(1)] = _resolve(raw[1:-1] if quoted else raw, env)
            continue
        for _flags, rest in _RM.findall(line):
            try:
                words = shlex.split(rest, comments=True)
            except ValueError:
                continue  # unbalanced quotes; not a shape we ship
            for word in words:
                if word.startswith("-"):
                    continue
                targets.append((line_no, _resolve(word, env)))
    return targets


def test_sbatch_files_are_discovered() -> None:
    """A glob that silently matched nothing would make every gate below vacuous."""
    files = _sbatch_files()
    assert len(files) >= 3, files
    assert any(f.name == "train_production.sbatch" for f in files)


def test_the_parser_finds_the_rm_lines_it_is_meant_to_police() -> None:
    """Anti-vacuity: prove the regex actually extracts resolved paths from the real file."""
    text = (SLURM / "p2" / "train_production.sbatch").read_text()
    targets = [p for _, p in _rm_targets(text)]
    assert "reports/p2/train_stage1_production.DONE" in targets, targets
    assert "data/processed/checkpoints/stage1_production/stage1.pt" in targets, targets


# ── the two evasions a naive parser admits (CodeRabbit r1) ───────────────────────────
# Both are asserted on synthetic text rather than on a shipped file, so they keep biting
# after the real sbatch files change shape.
def test_a_later_reassignment_cannot_hide_an_earlier_deletion() -> None:
    """Execution order, not final values: the `rm` must score against the value it HAD."""
    text = (
        'REPORT="reports/p2/train_stage1_production.json"\n'
        'rm -f "$REPORT"\n'
        'REPORT="/tmp/scratch/harmless.json"\n'
    )
    assert ("reports/p2/train_stage1_production.json") in [p for _, p in _rm_targets(text)]


def test_a_backslash_continued_rm_is_still_seen() -> None:
    """`rm -f \\` + a continuation line deletes exactly what the one-liner deletes."""
    text = 'DONE="reports/p2/x.DONE"\nrm -f \\\n  "$DONE" \\\n  "other.json"\n'
    found = [p for _, p in _rm_targets(text)]
    assert "reports/p2/x.DONE" in found and "other.json" in found


def test_an_unresolvable_variable_is_not_scored_as_a_safe_literal() -> None:
    """A path the parser cannot resolve must be visibly unresolved, never a clean miss."""
    (target,) = [p for _, p in _rm_targets('rm -f "$UNDEFINED_VAR"\n')]
    assert "UNRESOLVED" in target


#: Known-defective sbatch files, named rather than skipped. `sizing_smoke.sbatch:98` clears
#: `reports/p2/sizing_smoke.json` **and** `reports/p2/sizing/*.json` (16 tracked files) and
#: runs the same `train_stage1` entrypoint, so it carries the identical latent failure — a
#: re-run would die on its own `provenance_complete` clause. It is NOT fixed here because
#: the fix is not the one-token change `train_production.sbatch` took: line 301 aggregates
#: `POINTS=("$POINT_DIR"/*.json)` by GLOB, so simply not deleting would silently aggregate a
#: previous run's points into this run's report — strictly worse than the bug. The correct
#: fix redirects `POINT_DIR` to node-local scratch and copies the points back after
#: aggregating, and no part of that can be executed or verified from the laptop. Shipping an
#: unverified edit to a SLURM script is precisely the failure this file exists to catch, so
#: it is recorded instead (P2-10d′-c; TODO.md).
#:
#: `strict=True` is load-bearing: when sizing_smoke IS fixed this test XPASSes, which
#: **fails** the suite and forces this entry to be deleted. The exemption cannot rot green.
_KNOWN_DEFECTIVE = {"sizing_smoke.sbatch": "P2-10d′-c: needs POINT_DIR moved to scratch"}


@pytest.mark.parametrize("name", sorted(_KNOWN_DEFECTIVE))
def test_a_known_defective_sbatch_cannot_run_at_all(name: str) -> None:
    """An exemption from the gate above must come with the job being UNRUNNABLE.

    Otherwise `xfail` is just permission: the file stays submittable while the suite is
    green about it (CodeRabbit, P2-10d′-c r1). The refusal must be unconditional (column 0,
    so not nested inside an `if`) and must precede the first deletion, so the job cannot
    destroy anything on its way out.
    """
    (path,) = [p for p in _sbatch_files() if p.name == name]
    text = path.read_text()
    first_rm = min(line for line, _ in _rm_targets(text))
    guards = [
        lineno
        for lineno, line in _logical_lines(text)
        if re.match(r"^exit\s+[1-9]", line) and lineno < first_rm
    ]
    assert guards, (
        f"{name} is exempted from the tracked-path gate but has no unconditional "
        f"`exit <non-zero>` before its first deletion at line {first_rm}. Either fix the "
        "file and remove its _KNOWN_DEFECTIVE entry, or disable it outright."
    )


def _param(path: Path):
    reason = _KNOWN_DEFECTIVE.get(path.name)
    marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
    return pytest.param(path, marks=marks, id=path.name)


@pytest.mark.parametrize("path", [_param(p) for p in _sbatch_files()])
def test_no_sbatch_deletes_a_git_tracked_path(path: Path) -> None:
    tracked = _tracked_paths()
    offenders = [
        f"{path.relative_to(REPO)}:{line} rm's tracked path {target!r}"
        for line, target in _rm_targets(path.read_text())
        if target in tracked
    ]
    assert not offenders, (
        "Deleting a tracked path dirties the tree outside `_DATA_STAGING_PREFIXES`, so "
        "`_provenance_complete` re-derives FALSE and the run fails its own gate AFTER "
        "training. Clear untracked markers instead, and fingerprint tracked outputs "
        "(md5 before/after) to prove freshness:\n  " + "\n  ".join(offenders)
    )
