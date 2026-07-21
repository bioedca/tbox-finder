"""P2-10d′-e — the DDP barrier that orders every rank's git snapshot before rank 0's write.

`build_report` runs on EVERY rank and snapshots `git status` as its first act; rank 0 alone
then writes the git-tracked `cfg.report_path`. Without a synchronisation point a straggler
rank still short of its own snapshot when that write lands reads the freshly dirtied report as
modified code outside `_DATA_STAGING_PREFIXES`, re-derives `provenance_complete` FALSE, and
fails the whole DDP run after ~20 GPU-h (P2-10d′-c Blocker B, re-armed by the WRITE instead of
the pre-run `rm`).

Two tiers, per the house convention (there is no conftest; gating is per-file, fail-closed):

* **Structural** (Tier 1, bare-CI): `train_stage1` imports torch-free, so `inspect.getsource`
  can assert the barrier call sits between `validate_report` and the `is_primary()` write, and
  that the helper is `dist.is_initialized()`-guarded. This bites when the CALL is removed or
  moved — it is the binding between the mechanism and the real epilogue.
* **Behavioral** (Tier 2, needs torch + gloo, runs LOCAL in `tbox-ml-dna`): a real 2-rank gloo
  group replays the race on a real temp git repo. The straggler rank sleeps before its snapshot;
  WITHOUT the barrier it observes rank 0's write (dirty), WITH `_barrier_before_primary_write`
  its snapshot provably precedes the write (clean). This is the "rank-ordering forced" test the
  step's Validation gate names — it fails without the barrier and passes with it.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tbox_finder.train.train_stage1 as T

REPO_ROOT = Path(__file__).resolve().parents[2]


def _function_ast(name: str) -> ast.FunctionDef:
    """The AST of a top-level function in train_stage1.py — code only, docstring inert.

    Inspecting the AST rather than `inspect.getsource` text is deliberate: the helper's
    docstring quotes `dist.is_initialized()` to explain the guard, so a substring check over
    the source would pass even if the guard were deleted from the CODE (a vacuous assertion —
    the exact trap [[degenerate-fixture-generators]] warns about).
    """
    tree = ast.parse(Path(T.__file__).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


# ── Tier 1 — structural (bare CI, torch-free) ────────────────────────────────────────────
def test_epilogue_barriers_between_validate_and_the_primary_write() -> None:
    """The barrier CALL sits between `validate_report(report)` and `if is_primary():`.

    Order matters absolutely: a barrier after the write, or before build_report, closes
    nothing. Asserting the source order binds the mechanism (`_barrier_before_primary_write`,
    exercised behaviourally below) to the one place it must be called.
    """
    src = inspect.getsource(T._train_stage1_inner)
    i_validate = src.index("problems = validate_report(report)")
    i_barrier = src.index("_barrier_before_primary_write()", i_validate)
    i_write = src.index("if is_primary():", i_validate)
    assert i_validate < i_barrier < i_write, (
        "the report-write barrier must be called AFTER validate_report and BEFORE the "
        "is_primary() write, or a straggler rank's git snapshot can still observe rank 0's "
        "freshly-written report (P2-10d′-e)."
    )


def test_the_barrier_helper_is_guarded_on_an_initialised_process_group() -> None:
    """EVERY `dist.barrier()` in the helper sits inside an `if dist.is_initialized():` guard.

    An unguarded `dist.barrier()` raises on the laptop smoke (`train_stage1` runs a single
    process with no `init_process_group`). AST-checked so removing the guard from the code —
    but not from the docstring — still trips this (P2-10d′-e).
    """

    def _attr(call: ast.Call) -> str:
        return getattr(call.func, "attr", "")  # method name for `x.method(...)`, else ""

    fn = _function_ast("_barrier_before_primary_write")
    all_barriers = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _attr(n) == "barrier"]
    guarded_barriers = [
        n
        for guard in ast.walk(fn)
        if isinstance(guard, ast.If)
        and isinstance(guard.test, ast.Call)
        and _attr(guard.test) == "is_initialized"
        for n in ast.walk(guard)
        if isinstance(n, ast.Call) and _attr(n) == "barrier"
    ]
    assert all_barriers, "the helper must call dist.barrier() — otherwise it synchronises nothing."
    assert len(guarded_barriers) == len(all_barriers), (
        "every dist.barrier() must be nested inside `if dist.is_initialized():`, or the "
        "single-process local smoke raises with no process group (P2-10d′-e)."
    )


def test_the_helper_is_a_noop_without_a_process_group() -> None:
    """Behavioural check of the guard that needs no DDP: torch present, group NOT initialised.

    Torch-gated but CPU-only and single-process, so it runs anywhere torch is installed.
    """
    dist = pytest.importorskip("torch.distributed")
    assert not dist.is_initialized()  # no group in this bare process
    T._barrier_before_primary_write()  # must return cleanly, not raise


# ── Tier 2 — behavioral (torch + gloo, LOCAL) ────────────────────────────────────────────
# The child driver runs in a fresh interpreter per rank so gloo rendezvous is real. It is
# emitted as a string rather than a module function so the two ranks are unambiguously
# separate OS processes with their own cwd (the snapshot is cwd-sensitive), matching the
# house `subprocess.run([sys.executable, "-c", code])` idiom (test_priors, test_release_pin).
_RANK_DRIVER = r"""
import os, sys, time, json
from pathlib import Path
import torch.distributed as dist
import tbox_finder.train.train_stage1 as T

rank = int(sys.argv[1])
world = int(sys.argv[2])
init_file = sys.argv[3]
repo = sys.argv[4]
report = sys.argv[5]
results = Path(sys.argv[6])
use_barrier = sys.argv[7] == "1"
straggle_s = float(sys.argv[8])

os.chdir(repo)  # _git_status_snapshot reads `git status` in the process cwd
dist.init_process_group(backend="gloo", init_method="file://" + init_file,
                        rank=rank, world_size=world)
try:
    # The straggler reaches its snapshot late. Rank 0 (no sleep) snapshots clean, then either
    # waits at the barrier (use_barrier) or races straight to the write.
    if rank != 0:
        time.sleep(straggle_s)
    snap = T._git_status_snapshot()          # the REAL function under test's neighbourhood
    if use_barrier:
        T._barrier_before_primary_write()    # the REAL production helper
    if rank == 0:
        Path(report).write_text('{"schema_version": 2, "changed": true}\n')  # dirties tracked path
    dirty = T._git_dirty_paths(snap)         # verdict from THIS rank's snapshot instant
    (results / ("rank%d.json" % rank)).write_text(json.dumps(dirty))
finally:
    dist.destroy_process_group()
"""


def _init_git_repo(repo: Path, report_rel: str) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    git("config", "commit.gpgsign", "false")
    report = repo / report_rel
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"schema_version": 1}\n')  # the committed baseline
    git("add", "-A")
    git("commit", "-q", "-m", "seed")


def _run_two_rank_race(tmp_path: Path, *, use_barrier: bool, straggle_s: float = 1.0) -> dict:
    """Spawn a real 2-rank gloo group replaying the snapshot/write race; return rank verdicts."""
    repo = tmp_path / ("repo_barrier" if use_barrier else "repo_nobarrier")
    repo.mkdir()
    report_rel = "reports/p2/train_stage1_production.json"
    _init_git_repo(repo, report_rel)
    results = tmp_path / ("res_barrier" if use_barrier else "res_nobarrier")
    results.mkdir()
    init_file = tmp_path / ("rdv_barrier" if use_barrier else "rdv_nobarrier")

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    procs = []
    for rank in (0, 1):
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _RANK_DRIVER,
                    str(rank),
                    "2",
                    str(init_file),
                    str(repo),
                    str(repo / report_rel),
                    str(results),
                    "1" if use_barrier else "0",
                    str(straggle_s),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    verdicts: dict[int, list] = {}
    for rank, p in enumerate(procs):
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, f"rank {rank} failed (rc={p.returncode}):\n{err}"
        verdicts[rank] = json.loads((results / f"rank{rank}.json").read_text())
    return verdicts


def test_the_barrier_closes_the_snapshot_write_race_and_its_absence_reproduces_it(
    tmp_path: Path,
) -> None:
    """WITHOUT the barrier the straggler sees rank 0's write (dirty); WITH it, clean.

    This is the load-bearing behavioural test: it fails when `_barrier_before_primary_write`
    is a no-op (the sabotage), because then the barrier arm reproduces the race too.
    """
    pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_gloo_available():
        pytest.skip("gloo backend unavailable")

    report_rel = "reports/p2/train_stage1_production.json"

    # No barrier: rank 0 snapshots clean then writes immediately; rank 1 wakes late and its
    # snapshot observes the dirtied report. The race, reproduced.
    no_barrier = _run_two_rank_race(tmp_path, use_barrier=False)
    assert report_rel in no_barrier[1], (
        "control arm did not reproduce the race — the straggler's snapshot should observe "
        "rank 0's write when nothing orders them. Check the straggle margin."
    )

    # With the barrier: rank 1's snapshot is taken before the barrier releases rank 0 to write,
    # so it is clean. THIS is the property the fix guarantees.
    barrier = _run_two_rank_race(tmp_path, use_barrier=True)
    assert barrier[1] == [], (
        "the barrier did not close the race: the straggler still observed a dirty report. If "
        "this fails while the control arm reproduced the race, _barrier_before_primary_write "
        "is not actually synchronising the ranks (P2-10d′-e)."
    )
    # Rank 0's own snapshot is clean in both arms (it snapshots before it writes) — a sanity
    # anchor that the harness is not trivially reporting "clean" everywhere.
    assert barrier[0] == []
    assert no_barrier[0] == []
