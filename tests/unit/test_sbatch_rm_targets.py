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

import posixpath
import re
import shlex
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "slurm"

#: ``VAR=value`` / ``VAR="value"`` alone on a line — the assignment forms the shipped sbatch
#: bodies actually use. Command substitutions are not resolved (see `_resolve`).
_ASSIGN = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\'[^\']*\'|\S*)\s*$')
#: Command names that mean `rm`, once the path prefix and the alias-bypassing backslash
#: are stripped. `\rm`, `/bin/rm` and `command rm` all delete exactly what `rm` deletes.
_RM_NAMES = {"rm"}
#: Tokens that END one command and begin another. Operands are scored per command, so
#: `rm -f "$REPORT"; :` cannot smuggle a trailing `;` into the path (CodeRabbit r5).
_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "\n"}
#: Marker left in place of a variable this parser cannot resolve. Never silently dropped.
_UNRESOLVED = "\x00UNRESOLVED"
#: `$VAR`, `${VAR}`, or `${VAR:-default}` (the default is taken when VAR is unknown — that
#: is what makes `${SLURM_SUBMIT_DIR:-$HOME/tbox-finder}` resolvable).
_VAR = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-[^}]*)?\}"  # ${VAR} / ${VAR:-default}
    r"|\$([A-Za-z_][A-Za-z0-9_]*)"  # $VAR
    r"|\$\{[^}]*\}"  # ${ARR[i]} and other forms this parser cannot expand
    r"|\$\([^)]*\)"  # $(command substitution) — never guessed at
    r"|`[^`]*`"  # `command substitution`
)
#: Variables whose meaning is fixed by how these jobs are submitted (§9.3: `sbatch` is run
#: FROM the repo root, so SLURM sets SLURM_SUBMIT_DIR to it). Seeding them as "." is what
#: lets `${SLURM_SUBMIT_DIR:-...}/reports/p1/x.json` be scored as the repo-relative path it
#: actually is, instead of as an unresolvable mystery.
_SEED_ENV = {"SLURM_SUBMIT_DIR": ".", "REPO": "."}
#: Absolute roots that cannot be the checkout whatever their variables expand to. An
#: absolute path is NOT safe merely for being absolute: `/work/$USER/tbox-finder/reports/...`
#: is unresolved AND absolute AND very possibly the repo (CodeRabbit, P2-10d′-c r4).
_SCRATCH_ROOTS = ("/tmp/", "/scratch/", "/dev/shm/", "/var/tmp/")


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

    def _one(m: re.Match) -> str:
        name = m.group(1) or m.group(3)
        if name is None:
            return _UNRESOLVED  # `${ARR[i]}` and friends: a form this parser cannot expand
        default = m.group(2)
        value = env.get(name)
        # bash `:-` takes the default when VAR is unset OR EMPTY. Returning the empty
        # value instead would collapse `${X:-reports/p2/r.json}` to "" — a path that
        # matches nothing and so scores clean (CodeRabbit, P2-10d′-c r3).
        if value:
            return value
        if default is not None:
            return default[2:]
        return _UNRESOLVED if value is None else ""

    for _ in range(5):  # bounded: assignments here nest at most a level or two
        new = _VAR.sub(_one, value)
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


def _store(env: dict[str, str], name: str, raw: str) -> None:
    """Record an assignment, storing UNRESOLVED unless the value FULLY expanded.

    A value that still contains `$` or a backtick did not resolve — `FAM=$(python -c ...)`
    tokenizes into `FAM=$` plus the substitution's own words, and storing that bare `$`
    made every later `$FAM` expand to an innocuous-looking literal, silently un-flagging
    nine tracked deletions. Storing the marker instead keeps the later use fail-CLOSED
    (CodeRabbit, P2-10d′-c r6).
    """
    quoted = len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]
    value = _resolve(raw[1:-1] if quoted else raw, env)
    env[name] = _UNRESOLVED if ("$" in value or "`" in value) else value


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
    env: dict[str, str] = dict(_SEED_ENV)
    targets: list[tuple[int, str]] = []
    for line_no, line in _logical_lines(text):
        if line.lstrip().startswith("#"):
            continue  # a comment, not a command — the fix's own rationale lives in one
        assign = _ASSIGN.match(line)
        if assign:
            _store(env, assign.group(1), assign.group(2))
            continue
        for command in _commands(line, line_no):
            words = list(command)
            prefixes: list[str] = []
            while words and _ASSIGN.match(words[0]):
                prefixes.append(words.pop(0))
            if not words:
                # An assignment-ONLY command — `REPORT="x.json"; rm -f "$REPORT"` splits
                # into two commands on the `;`, and this is the first. That assignment
                # PERSISTS, unlike the `FOO=bar rm ...` per-command prefix above it, which
                # bash scopes to the one command. Dropping it left the later `rm` resolving
                # against an absent value — which fails CLOSED (the marker widens to `*`
                # and flags everything), so it was never a missed detection, but it would
                # fire spuriously the first time a shipped script used the idiom
                # (CodeRabbit, P2-10d′-c r6).
                for prefix in prefixes:
                    kv = _ASSIGN.match(prefix)
                    if kv:
                        _store(env, kv.group(1), kv.group(2))
                continue
            if words[0] == "command":
                words.pop(0)
            if not words:
                continue
            name = words[0].lstrip("\\")
            if posixpath.basename(name) not in _RM_NAMES or name.endswith("/"):
                continue
            for word in words[1:]:
                if word.startswith("-"):
                    continue
                targets.append((line_no, _resolve(word, env)))
    return targets


def _commands(line: str, line_no: int) -> list[list[str]]:
    """Tokenize one logical line into commands, split on shell separators.

    Scoring "everything after `rm`" treats `;` and `&&` as part of the last operand and
    hides both a trailing separator and any LATER `rm` on the same line. `shlex` with
    ``punctuation_chars`` is the tokenizer that gets this right (CodeRabbit r5).
    """
    lex = shlex.shlex(line, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        tokens = list(lex)
    except ValueError:
        # Unbalanced quotes. Fail CLOSED if the line could be a deletion at all: an
        # unparseable `rm` must not read the same as no `rm`.
        return [["rm", _UNRESOLVED]] if re.search(r"(?:^|\s|/)rm\b", line) else []
    out: list[list[str]] = [[]]
    for token in tokens:
        if token in _SEPARATORS:
            out.append([])
        else:
            out[-1].append(token)
    return [c for c in out if c]


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


@pytest.mark.parametrize(
    "cmd",
    ["/bin/rm -f", "/usr/bin/rm -rf", "\\rm -f", "command rm -f"],
    ids=["abs-path", "abs-path-rf", "escaped", "command-builtin"],
)
def test_path_qualified_and_escaped_rm_are_still_seen(cmd: str) -> None:
    """`rm` is not always spelled `rm`. Each of these deletes exactly what plain rm does."""
    text = f'{cmd} "reports/p2/train_stage1_production.json"\n'
    assert "reports/p2/train_stage1_production.json" in [p for _, p in _rm_targets(text)]


def test_an_unresolved_absolute_path_is_not_safe_merely_for_being_absolute() -> None:
    """`/work/$USER/tbox-finder/...` can BE the checkout at runtime — fail closed."""
    text = 'rm -f "/work/$USER/tbox-finder/reports/p2/train_stage1_production.json"\n'
    (target,) = [p for _, p in _rm_targets(text)]
    assert _offence(target, _tracked_paths()) is not None


def test_node_local_scratch_stays_exempt() -> None:
    """The other direction — the build dir must NOT be flagged, or the gate is noise."""
    text = 'BUILD="/tmp/${USER}-${SLURM_JOB_ID:-local}"\nrm -rf "$BUILD"\n'
    (target,) = [p for _, p in _rm_targets(text)]
    assert _offence(target, _tracked_paths()) is None


def test_a_separator_joined_assignment_persists() -> None:
    """`X="..."; rm -f "$X"` — the assignment before the `;` is a real shell assignment."""
    text = 'REPORT="reports/p2/train_stage1_production.json"; rm -f "$REPORT"\n'
    assert "reports/p2/train_stage1_production.json" in [p for _, p in _rm_targets(text)]


def test_a_per_command_env_prefix_does_NOT_persist() -> None:
    """The other direction: bash scopes `FOO=bar cmd` to that one command.

    Without this, "persist assignments" would over-correct and let a transient prefix leak
    into later commands — resolving a path against a value that never existed there.
    """
    text = 'REPORT="/tmp/scratch.json" :\nrm -f "$REPORT"\n'
    (target,) = [p for _, p in _rm_targets(text)]
    assert _UNRESOLVED in target, target


def test_a_separator_cannot_hide_a_tracked_deletion() -> None:
    """`rm -f "$REPORT"; :` must not smuggle the `;` into the path (CodeRabbit r5)."""
    text = 'REPORT="reports/p2/train_stage1_production.json"\nrm -f "$REPORT"; :\n'
    assert "reports/p2/train_stage1_production.json" in [p for _, p in _rm_targets(text)]


def test_a_second_rm_on_the_same_line_is_not_swallowed() -> None:
    text = "rm -f /tmp/scratch.json && rm -f reports/p2/train_stage1_production.json\n"
    assert "reports/p2/train_stage1_production.json" in [p for _, p in _rm_targets(text)]


def test_an_unparseable_rm_line_fails_closed() -> None:
    """Unbalanced quotes must not read the same as "there was no rm here"."""
    targets = [p for _, p in _rm_targets('rm -f "unterminated\n')]
    assert targets and all(_UNRESOLVED in t for t in targets)


def test_a_command_substitution_is_never_guessed_at() -> None:
    """`$(...)` left as a literal matches nothing and would score clean — fail-open."""
    (target,) = [p for _, p in _rm_targets('rm -f "$(dirname x)/report.json"\n')]
    assert _UNRESOLVED in target


def test_an_empty_value_falls_through_to_the_default_like_bash() -> None:
    """`${VAR:-default}` uses the default when VAR is set-but-EMPTY, not just when unset."""
    text = 'X=""\nrm -f "${X:-reports/p2/train_stage1_production.json}"\n'
    assert "reports/p2/train_stage1_production.json" in [p for _, p in _rm_targets(text)]


def test_an_unresolvable_variable_is_not_scored_as_a_safe_literal() -> None:
    """A path the parser cannot resolve must be visibly unresolved, never a clean miss."""
    (target,) = [p for _, p in _rm_targets('rm -f "$UNDEFINED_VAR"\n')]
    assert _UNRESOLVED in target


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
_KNOWN_DEFECTIVE = {
    # P2 — this step's blast radius. Disabled outright (see _MUST_BE_DISABLED).
    "sizing_smoke.sbatch": "P2-10d′-c: deletes sizing_smoke.json + 15 tracked point reports",
    "sweep_stage1.sbatch": "P2-10d′-c: deletes 36 tracked sweep point reports",
    # P1 — the same defect, found by this gate the day it was written. These jobs are
    # COMPLETE and unscheduled, and disabling seven of them is a scope decision for the
    # user, not a side effect of a P2 bug fix — so they are recorded here and raised in
    # TODO.md for a remediation step, not silently fixed or silently ignored.
    "kernel_smoke.sbatch": "P2-10d′-c census: deletes tracked reports/p1/kernel_smoke.json",
    "lora_smoke.sbatch": "P2-10d′-c census: deletes tracked reports/p1/lora_vram_smoke.json",
    "rinalmo_parity.sbatch": "P2-10d′-c census: deletes 9 tracked parity fold reports",
    "rinalmo_throughput.sbatch": "P2-10d′-c census: deletes tracked rinalmo_throughput.json",
    "seg_smoke.sbatch": "P2-10d′-c census: deletes a tracked report AND a tracked provenance",
    "seg_smoke_repro.sbatch": "P2-10d′-c census: deletes tracked reports/p1/seg_smoke_repro.json",
}

#: Of the above, the ones that must additionally be UNRUNNABLE. An exemption alone is just
#: permission — the file stays submittable while the suite is green about it (CodeRabbit,
#: P2-10d′-c r1) — so every P2 job in this step's blast radius carries a hard refusal. The
#: P1 entries are a census awaiting a scheduled remediation step; they are listed above so
#: their `strict=True` xfail forces them out the moment they are fixed.
_MUST_BE_DISABLED = {"sizing_smoke.sbatch", "sweep_stage1.sbatch"}


@pytest.mark.parametrize("name", sorted(_MUST_BE_DISABLED))
def test_a_known_defective_sbatch_cannot_run_at_all(name: str) -> None:
    """A disabled-for-cause job must refuse BEFORE it can destroy anything.

    The refusal must be unconditional (column 0, so not nested inside an `if`) and must
    precede the first deletion.
    """
    assert name in _KNOWN_DEFECTIVE, f"{name} must also carry a gate exemption"
    (path,) = [p for p in _sbatch_files() if p.name == name]
    text = path.read_text()
    first_rm = min(line for line, _ in _rm_targets(text))
    guards = [n for n, line in _logical_lines(text) if _is_unconditional_exit(text, n, line)]
    assert any(n < first_rm for n in guards), (
        f"{name} is exempted from the tracked-path gate but has no UNCONDITIONAL "
        f"`exit <non-zero>` before its first deletion at line {first_rm}. Either fix the "
        "file and remove its _KNOWN_DEFECTIVE entry, or disable it outright."
    )


#: Shell words that open / close a control block. Column zero is NOT enough on its own:
#: `if "$ALLOW"; then` / `exit 1` / `fi` puts an `exit` at column zero that the normal path
#: never reaches, so the job stays runnable behind a green test (CodeRabbit r5).
_BLOCK_OPEN = ("if ", "for ", "while ", "until ", "case ")
_BLOCK_CLOSE = ("fi", "done", "esac", "}")


def _block_depth_at(text: str, target_lineno: int) -> int:
    """Control-block nesting depth immediately before `target_lineno`."""
    depth = 0
    for lineno, line in _logical_lines(text):
        if lineno >= target_lineno:
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith(_BLOCK_OPEN):
            depth += 1
        elif stripped in _BLOCK_CLOSE or stripped.startswith(("fi ", "done ", "esac ")):
            depth = max(0, depth - 1)
        elif stripped.endswith("() {") or stripped.endswith("()  {"):
            depth += 1
    return depth


def _is_unconditional_exit(text: str, lineno: int, line: str) -> bool:
    """A non-zero `exit` that the normal path cannot avoid."""
    return bool(re.match(r"^exit\s+[1-9]", line)) and _block_depth_at(text, lineno) == 0


def _param(path: Path):
    reason = _KNOWN_DEFECTIVE.get(path.name)
    marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
    return pytest.param(path, marks=marks, id=path.name)


def _offence(target: str, tracked: set[str]) -> str | None:
    """Why `target` is not allowed to be deleted, or None.

    Three ways a deletion reaches a tracked path, all fail-open if only the first is
    checked (CodeRabbit, P2-10d′-c r2):

    1. It names one literally.
    2. It is a **glob** that matches one. `rm -f "$POINT_DIR"/*.json` deletes 15 tracked
       files while the literal string `reports/p2/sizing/*.json` is in no index.
    3. It could not be **resolved**. An unresolved relative path may be anything at
       runtime, so scoring it clean is a guess in the fail-open direction. Absolute
       unresolved paths are exempt — `/tmp/$USER-$SLURM_JOB_ID` is node-local scratch and
       cannot be a repo path whatever it expands to.
    """
    path = posixpath.normpath(target)
    if path in tracked:
        return "tracked path"
    if path.startswith(_SCRATCH_ROOTS):
        return None  # node-local scratch can never be the checkout
    if path.startswith("/"):
        if _UNRESOLVED not in path:
            return None  # a fully-resolved absolute path outside the repo
        return "unresolved absolute path (could be inside the checkout at runtime)"
    # A glob and an unresolved variable are the same question — "could this match a tracked
    # file?" — so they get the same answer. `*` for the unknown part is the WIDEST reading,
    # which is the fail-CLOSED direction: `reports/p2/sweep/g${GAMMAS[i]}_...json` becomes
    # `reports/p2/sweep/g*_...json` and matches the committed sweep points, as it should.
    pattern = path.replace(_UNRESOLVED, "*")
    if pattern != path or any(ch in path for ch in "*?["):
        hits = sorted(t for t in tracked if fnmatch(t, pattern))
        if hits:
            return f"pattern matching {len(hits)} tracked path(s), e.g. {hits[0]}"
    return None


@pytest.mark.parametrize("path", [_param(p) for p in _sbatch_files()])
def test_no_sbatch_deletes_a_git_tracked_path(path: Path) -> None:
    tracked = _tracked_paths()
    offenders = [
        f"{path.relative_to(REPO)}:{line} rm's {why}: {target.replace(_UNRESOLVED, '<?>')!r}"
        for line, target in _rm_targets(path.read_text())
        if (why := _offence(target, tracked))
    ]
    assert not offenders, (
        "Deleting a tracked path dirties the tree outside `_DATA_STAGING_PREFIXES`, so "
        "`_provenance_complete` re-derives FALSE and the run fails its own gate AFTER "
        "training. Clear untracked markers instead, and fingerprint tracked outputs "
        "(md5 before/after) to prove freshness:\n  " + "\n  ".join(offenders)
    )
