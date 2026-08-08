"""The per-candidate mlocarna wall-clock bound (P3-15'-e-ii) and its fail-closed resolution.

WHY THIS EXISTS. The full-corpus covariation-(a) supply run (SLURM job 1205) lost shard 016 to the
12 h wall having completed **1 of its 8 alignable candidates**: one ``mlocarna`` ran 6h11m without
finishing. Measured over the 272 alignments that succeeded in the 47 surviving shards, 271 finished
within 353 s and one took 8500 s at MSA depth 844 — a single candidate accounting for 69 % of all
align wall-clock in the corpus. LocARNA's cost climbs steeply in depth (ADR-0005 A10 Pin 2's own
"scaling caveat"), so the tail has no useful bound and a longer ``--time`` is not a fix.

WHAT MUST HOLD, and what each of these tests would let through if it did not:
  1. The bound actually kills the tool  — else the shard still dies at the wall.
  2. It kills the tool's **children**   — ``mlocarna`` is a Perl driver that dispatches ``locarna``
     workers, and ``subprocess.run(timeout=…)`` reaps only the process it spawned. Orphaned workers
     keep burning the node's cores; the shard would look "bounded" while the cost survived.
  3. A timeout leaves **no MSA**        — absence of ``msa.sto`` is what makes the score stage read
     ``unavailable`` ⇒ **spared** (ADR-0005 D14). An MSA left behind by a killed alignment would be
     promoted by the sbatch (which tests ``[ -s … ]``, not this run's verdict) and scored as a real
     de-novo consensus — a fabricated result, §10.3.
  4. A timeout is **distinguishable**   — from "the tool answered no" and from "too few homologs";
     all three spare, but only one of them is a choice this run made.
  5. Every caller **names** the bound   — keyword-required-no-default (the ADR-0006 A4 rule-param
     shape) and ``required=True`` at the CLI, so a forgetful sbatch fails at argparse instead of
     silently re-running job 1205 unbounded.
  6. A non-positive bound is **refused** — because a timeout SPARES, ``--align-timeout-s 0`` would
     mark all 941 candidates ``unavailable`` and read as a clean, producible-nothing round
     ([[cost-knobs-can-certify]]).

The subprocesses here are real (``sys.executable``), not mocks: a timeout is a property of process
management, and a monkeypatched ``subprocess`` would assert only that this file agrees with itself.
"""

from __future__ import annotations

import errno
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tbox_finder.mining import covariation_producer as cp
from tbox_finder.mining import homolog_db as hdb
from tbox_finder.mining.homolog_db import HomologDbError, ToolTimeoutError

REPO = Path(__file__).resolve().parents[2]
PRODUCER_SBATCH = REPO / "slurm" / "p2" / "mine_round_producer.sbatch"
ORCHESTRATOR_SBATCH = REPO / "slurm" / "p2" / "mine_round.sbatch"
MEASURE_SBATCH = REPO / "slurm" / "p2" / "mine_round_measure.sbatch"

#: Long enough that a machine hiccup cannot make a *fast* command look slow, short enough that the
#: bounded tests stay inside the CI budget (§8.6, whole ml tier < ~8 min).
SLOW = 30.0
BOUND = 1.0


def _sleeper(seconds: float) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


# ═════════════════════════════════════════════════════════════════════════════
# 1-2. The transport: the bound fires, and it takes the whole process group
# ═════════════════════════════════════════════════════════════════════════════
def test_an_unbounded_run_is_unchanged():
    """``timeout_s=None`` is the pre-P3-15'-e-ii behaviour — the certified search/index paths
    (job 741 / job 766) must not acquire a bound by accident."""
    proc = hdb._run([sys.executable, "-c", "print('ok')"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "ok"


def test_a_command_inside_its_bound_returns_normally():
    """The positive control. Without it, a `_run` that raised on EVERYTHING would satisfy every
    `pytest.raises` below ([[raises-test-needs-a-positive-control]])."""
    proc = hdb._run([sys.executable, "-c", "print('fast')"], timeout_s=SLOW)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "fast"


def test_a_command_exceeding_its_bound_raises_tool_timeout():
    t0 = time.perf_counter()
    with pytest.raises(ToolTimeoutError) as excinfo:
        hdb._run(_sleeper(SLOW), timeout_s=BOUND)
    elapsed = time.perf_counter() - t0
    # It must return PROMPTLY, not merely eventually: a `_run` that waited out the child and then
    # reported a timeout would raise exactly this exception while saving no wall-clock at all.
    assert elapsed < SLOW / 2, f"the bound did not cut the call short (took {elapsed:.1f}s)"
    assert "bound" in str(excinfo.value)


def test_a_timeout_is_distinguishable_from_a_nonzero_exit():
    """Both are HomologDbError, but only one means "the tool never answered". align_shard branches
    on the difference to label the row, so collapsing them would erase the diagnosis."""
    with pytest.raises(HomologDbError) as failed:
        hdb._run([sys.executable, "-c", "raise SystemExit(3)"])
    assert not isinstance(failed.value, ToolTimeoutError)
    assert issubclass(ToolTimeoutError, HomologDbError)


def test_the_timeout_kills_the_childs_children_not_just_the_child(tmp_path: Path):
    """THE LOAD-BEARING ONE. ``mlocarna`` forks workers (``--threads`` is a real parallelism knob,
    LocARNA docs); ``subprocess.run(timeout=…)`` kills only the direct child. If the group is not
    killed, the grandchild outlives the timeout and keeps consuming the node — the shard would be
    bounded on paper while the CPU burn it exists to stop continued.

    The grandchild writes a file *after* the parent is already dead. The file's absence is the
    proof; a plain ``proc.kill()`` implementation leaves it present.
    """
    beacon = tmp_path / "grandchild_survived.txt"
    # Parent spawns a detached grandchild that sleeps past the bound, then writes the beacon.
    parent = [
        sys.executable,
        "-c",
        (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', "
            f"\"import time; time.sleep(3); open(r'{beacon}', 'w').write('alive')\"])\n"
            "time.sleep(60)\n"
        ),
    ]
    with pytest.raises(ToolTimeoutError):
        hdb._run(parent, timeout_s=BOUND)
    time.sleep(5)  # outlast the grandchild's own sleep — if it lived, it has now written
    assert not beacon.exists(), (
        "the grandchild outlived the timeout: the bound killed the direct child only, so a "
        "timed-out mlocarna would orphan its locarna workers on the compute node"
    )


def test_the_killed_group_is_actually_reaped(tmp_path: Path):
    """A kill that leaves the process alive-but-unwaited is a zombie, and the shard's task would
    still hold it. Assert the pid is gone, not merely signalled."""
    pidfile = tmp_path / "pid.txt"
    staging = tmp_path / "pid.txt.tmp"
    # Written atomically: `open(...).write(...)` CREATES the file and then fills it, so a timeout
    # landing between those two steps leaves an empty pid.txt and `int("")` fails the test for a
    # reason that has nothing to do with reaping.
    cmd = [
        sys.executable,
        "-c",
        (
            "import os, time\n"
            f"open(r'{staging}', 'w').write(str(os.getpid()))\n"
            f"os.replace(r'{staging}', r'{pidfile}')\n"
            "time.sleep(60)\n"
        ),
    ]
    with pytest.raises(ToolTimeoutError):
        hdb._run(cmd, timeout_s=BOUND)
    raw = pidfile.read_text().strip()
    assert raw, "the child never recorded its pid — the test cannot say anything about reaping"
    pid = int(raw)
    time.sleep(0.5)
    with pytest.raises(OSError) as exc:  # ESRCH once reaped; a zombie would still answer signal 0
        os.kill(pid, 0)
    assert exc.value.errno == errno.ESRCH, f"pid {pid} still exists after the timeout"


@pytest.mark.parametrize("bad", [0, 0.0, -1, -600.0, float("nan"), float("inf")])
def test_a_non_positive_bound_is_refused(bad: float):
    """A 0/negative bound times out EVERY alignment. Because a timeout spares rather than fails,
    that would be silent: a full round would report `unavailable` on all 941 candidates and look
    like an honest producible-nothing result ([[cost-knobs-can-certify]])."""
    with pytest.raises(ValueError, match="positive"):
        hdb._run([sys.executable, "-c", "pass"], timeout_s=bad)


def test_the_group_kill_falls_back_when_the_group_is_already_gone():
    """`_kill_process_group` must not explode on a race where the group died on its own between
    the timeout and the kill — that would turn a spared candidate into a crashed shard."""

    class _AlreadyGone:
        pid = 2**30  # a pid that cannot exist ⇒ os.getpgid raises ProcessLookupError
        killed = False

        def kill(self) -> None:
            self.killed = True

    popen = _AlreadyGone()
    hdb._kill_process_group(popen)  # must not raise
    assert popen.killed, "the fallback must still try to kill the direct child"


def test_the_group_kill_refuses_to_kill_its_own_group(monkeypatch: pytest.MonkeyPatch):
    """The safety interlock. If `start_new_session=True` were dropped from `_run_bounded`, the
    child would share OUR process group and the killpg would SIGKILL the SLURM array task itself —
    a dead shard instead of a spared candidate. Kill only the child in that case."""
    monkeypatch.setattr(hdb.os, "getpgid", lambda pid: os.getpgrp())

    def _must_not_run(pgid: int, sig: int) -> None:  # pragma: no cover - the failure case
        raise AssertionError(f"killpg({pgid}) would have killed our own process group")

    monkeypatch.setattr(hdb.os, "killpg", _must_not_run)

    class _SameGroup:
        pid = 123
        killed = False

        def kill(self) -> None:
            self.killed = True

    popen = _SameGroup()
    hdb._kill_process_group(popen)
    assert popen.killed, "the child must still be killed, just not the whole group"


def test_the_group_kill_uses_sigkill_on_the_group(monkeypatch: pytest.MonkeyPatch):
    """Pin the mechanism, not just the outcome: SIGTERM is catchable and mlocarna's Perl driver
    could trap it, which is exactly how a bound silently stops binding."""
    seen: list[tuple[int, int]] = []
    monkeypatch.setattr(hdb.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(hdb.os, "killpg", lambda pgid, sig: seen.append((pgid, sig)))

    class _Fake:
        pid = 99

        def kill(self) -> None:  # pragma: no cover - must not be reached
            raise AssertionError("fell back to the direct child despite a live group")

    hdb._kill_process_group(_Fake())
    assert seen == [(4242, signal.SIGKILL)]


# ═════════════════════════════════════════════════════════════════════════════
# 3-4. align_shard: a timeout is recorded, leaves no MSA, and is spared
# ═════════════════════════════════════════════════════════════════════════════
def _seed_sufficient_candidate(tmp_path: Path) -> tuple[cp.CandidateSpec, Path]:
    """A candidate whose search stage reported a sufficient homolog set, so align_shard runs it."""
    spec = cp.CandidateSpec(
        candidate_id="GCA_000000001.1:c0:0:10-20",
        accession="GCA_000000001.1:c0",
        locus_start=10,
        locus_end=20,
    )
    wd = cp.candidate_workdir(tmp_path, spec.candidate_id)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "search.json").write_text('{"sufficient": true, "n_homologs": 42}\n', encoding="utf-8")
    return spec, wd


def test_a_timed_out_alignment_is_recorded_as_align_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec, wd = _seed_sufficient_candidate(tmp_path)

    def _timeout(**_kwargs: object) -> None:
        raise ToolTimeoutError("command exceeded its 600s bound and was killed: mlocarna …")

    monkeypatch.setattr(cp, "align_candidate", _timeout)
    rows = cp.align_shard([spec], workroot=tmp_path, align_timeout_s=600.0)

    (row,) = rows
    assert row["aligned"] is False
    assert row["reason"] == "align_timeout", (
        "a timeout must be labelled distinctly from align_failed — both spare, but only the "
        "timeout is a consequence of a knob this run chose"
    )
    assert not (wd / cp.MSA_FILENAME).exists(), "a timed-out alignment must leave no MSA behind"


def test_a_timed_out_alignment_does_not_abort_the_rest_of_the_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The whole point: one pathological candidate must cost ONE candidate, not the shard. Job 1205
    lost 20 candidates to one alignment because the failure was fatal rather than spared."""
    specs = []
    for i in range(3):
        spec = cp.CandidateSpec(
            candidate_id=f"GCA_00000000{i}.1:c0:0:10-20",
            accession=f"GCA_00000000{i}.1:c0",
            locus_start=10,
            locus_end=20,
        )
        wd = cp.candidate_workdir(tmp_path, spec.candidate_id)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "search.json").write_text('{"sufficient": true}\n', encoding="utf-8")
        specs.append(spec)

    calls: list[str] = []

    def _first_one_hangs(*, out_sto: Path, **kwargs: object) -> Path:
        calls.append(str(out_sto))
        if len(calls) == 1:
            raise ToolTimeoutError("bound exceeded")
        Path(out_sto).write_text("# STOCKHOLM 1.0\n//\n", encoding="utf-8")
        return Path(out_sto)

    monkeypatch.setattr(cp, "align_candidate", _first_one_hangs)
    rows = cp.align_shard(specs, workroot=tmp_path, align_timeout_s=600.0)

    assert len(rows) == 3, "the shard must run to completion"
    assert [r.get("reason") for r in rows] == ["align_timeout", None, None]
    assert [r["aligned"] for r in rows] == [False, True, True]


def test_a_stale_msa_is_discarded_when_the_alignment_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A leftover msa.sto under a re-used workdir would be promoted by the sbatch (which tests
    `[ -s … ]`, not this run's verdict) and scored as a de-novo consensus this run never produced
    — a fabricated result (§10.3). Absence must be enforced, not assumed."""
    spec, wd = _seed_sufficient_candidate(tmp_path)
    stale = wd / cp.MSA_FILENAME
    stale.write_text("# STOCKHOLM 1.0\n#=GC SS_cons <<>>\n//\n", encoding="utf-8")

    def _timeout(**_kwargs: object) -> None:
        raise ToolTimeoutError("bound exceeded")

    monkeypatch.setattr(cp, "align_candidate", _timeout)
    cp.align_shard([spec], workroot=tmp_path, align_timeout_s=600.0)

    assert not stale.exists(), "the stale MSA survived a timed-out alignment"


def test_a_stale_msa_is_discarded_when_the_alignment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The sibling branch — `align_failed` relies on the same absence. Fixing one of two identical
    things is this repo's most-repeated defect ([[fixed-one-of-two-identical-things]])."""
    spec, wd = _seed_sufficient_candidate(tmp_path)
    stale = wd / cp.MSA_FILENAME
    stale.write_text("# STOCKHOLM 1.0\n//\n", encoding="utf-8")

    def _boom(**_kwargs: object) -> None:
        raise cp.HomologMsaError("mlocarna produced no SS_cons")

    monkeypatch.setattr(cp, "align_candidate", _boom)
    cp.align_shard([spec], workroot=tmp_path, align_timeout_s=600.0)

    assert not stale.exists(), "the stale MSA survived a failed alignment"


def test_the_bound_reaches_align_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A parameter that is accepted and then dropped is worse than none — the round would record a
    bound in provenance that never reached mlocarna ([[pinned-constant-that-nothing-reads]])."""
    spec, _wd = _seed_sufficient_candidate(tmp_path)
    seen: dict[str, object] = {}

    def _capture(*, out_sto: Path, **kwargs: object) -> Path:
        seen.update(kwargs)
        Path(out_sto).write_text("# STOCKHOLM 1.0\n//\n", encoding="utf-8")
        return Path(out_sto)

    monkeypatch.setattr(cp, "align_candidate", _capture)
    cp.align_shard([spec], workroot=tmp_path, align_timeout_s=137.5)
    assert seen["timeout_s"] == 137.5


def test_align_candidate_forwards_its_bound_to_the_transport(monkeypatch: pytest.MonkeyPatch):
    """The other half of the same thread — homolog_msa must hand the bound to `_run`, not just
    accept it. Sabotaged by dropping `timeout_s=` from the `_run(mlocarna_argv(...))` call."""
    from tbox_finder.mining import homolog_msa as hm

    seen: dict[str, object] = {}
    monkeypatch.setattr(hm, "assert_mlocarna_version", lambda *a, **k: "LocARNA 2.0.1")
    monkeypatch.setattr(hm, "_assert_candidate_in_homologs", lambda *a, **k: None)
    # The binary is not installed in the test env; resolve it by name so the REAL mlocarna_argv
    # still builds the argv (the thing under test is the timeout thread, not the tool lookup).
    monkeypatch.setattr(hm, "tool_path", lambda name: name)

    def _fake_run(cmd: list[str], *, timeout_s: float | None = None) -> None:
        seen["timeout_s"] = timeout_s
        raise ToolTimeoutError("bound exceeded")

    monkeypatch.setattr(hm, "_run", _fake_run)
    with pytest.raises(ToolTimeoutError):
        hm.align_candidate(
            candidate_fasta="c.fa",
            homologs_fasta="h.fa",
            out_sto="o.sto",
            work_dir="w",
            timeout_s=42.0,
        )
    assert seen["timeout_s"] == 42.0


# ═════════════════════════════════════════════════════════════════════════════
# 5. Every caller names the bound
# ═════════════════════════════════════════════════════════════════════════════
def test_align_shard_refuses_to_run_without_a_bound(tmp_path: Path):
    """Keyword-required, NO default (the ADR-0006 A4 rule-parameter shape). A default would let a
    caller inherit an unbounded align silently — which is the state that lost shard 016."""
    spec = cp.CandidateSpec(
        candidate_id="GCA_000000001.1:c0:0:10-20",
        accession="GCA_000000001.1:c0",
        locus_start=10,
        locus_end=20,
    )
    with pytest.raises(TypeError, match="align_timeout_s"):
        cp.align_shard([spec], workroot=tmp_path)  # type: ignore[call-arg]


def test_the_cli_requires_the_bound(capsys: pytest.CaptureFixture[str]):
    """argparse must reject an align-shard invocation that omits it, so a forgetful sbatch dies at
    once instead of re-running job 1205 unbounded for 12 h."""
    with pytest.raises(SystemExit) as exc:
        cp.main(["align-shard", "--shard", "s.json", "--workroot", "w"])
    assert exc.value.code == 2
    assert "--align-timeout-s" in capsys.readouterr().err


@pytest.mark.parametrize("given", ["137.5", "911"])
def test_the_cli_passes_the_bound_through_as_a_number(
    given: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`--align-timeout-s` must reach align_shard as a float, AND must VARY with the input.

    ⚠ This test was originally written with 600 — the same value a plausible sabotage hardcodes
    into `_cmd_align_shard` — so a CLI that parsed the flag and then passed a constant stayed
    green. Two distinct non-round values are what actually distinguish "threaded through" from
    "equal to a constant" ([[pinned-constant-that-nothing-reads]]: flip the value to prove it
    bites). A string bound would reach `communicate(timeout="600")`, which raises, so every
    alignment would crash its shard rather than being spared."""
    manifest = tmp_path / "shard.json"
    cp.write_candidate_manifest(
        [
            cp.CandidateSpec(
                candidate_id="GCA_000000001.1:c0:0:10-20",
                accession="GCA_000000001.1:c0",
                locus_start=10,
                locus_end=20,
            )
        ],
        manifest,
    )
    seen: dict[str, object] = {}

    def _capture(specs: object, **kwargs: object) -> list[dict[str, object]]:
        seen.update(kwargs)
        return []

    monkeypatch.setattr(cp, "align_shard", _capture)
    rc = cp.main(
        [
            "align-shard",
            "--shard",
            str(manifest),
            "--workroot",
            str(tmp_path),
            "--align-timeout-s",
            given,
        ]
    )
    assert rc == 0
    assert isinstance(seen["align_timeout_s"], float)
    assert seen["align_timeout_s"] == float(given)


# ═════════════════════════════════════════════════════════════════════════════
# The sbatch side — the bound must be required and forwarded, in bytes
# ═════════════════════════════════════════════════════════════════════════════
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _align_shard_invocation(sbatch: Path) -> str:
    """The single EXECUTABLE `… covariation_producer align-shard …` command in an sbatch.

    Anchored on the module path and filtered to non-comment lines. Both matter: these files
    *discuss* `align-shard` in their headers, and a substring search happily matched a sentence
    about the bound instead of the command that carries it — a check that would then have passed
    on prose while the real invocation stayed unbounded.
    """
    text = _read(sbatch)
    matches = [
        m.group(0)
        for m in re.finditer(r"covariation_producer align-shard\b((?:[^\n]*\\\s*\n)*[^\n]*)", text)
        if not text[text.rfind("\n", 0, m.start()) + 1 : m.start()].lstrip().startswith("#")
    ]
    assert len(matches) == 1, (
        f"expected exactly one executable `covariation_producer align-shard` command in "
        f"{sbatch.name}, found {len(matches)}"
    )
    return matches[0]


@pytest.mark.parametrize(
    "sbatch",
    [PRODUCER_SBATCH, ORCHESTRATOR_SBATCH, MEASURE_SBATCH],
    ids=lambda p: p.name,
)
def test_every_sbatch_that_aligns_requires_the_bound(sbatch: Path):
    """`${ALIGN_TIMEOUT_S:?…}` — not `:-`, which would restore the silent unbounded default that
    `--test-only` cannot catch (it validates headers only)."""
    # BOTH assertions read the comment-stripped body. These headers DISCUSS the flags at length, so
    # a positive check against the raw text passes on a header that merely QUOTES
    # `${ALIGN_TIMEOUT_S:?…}` while the executable body declares nothing, and the negative check
    # fails on a header documenting the anti-pattern. Splitting them was itself an instance of
    # [[fixed-one-of-two-identical-things]] — twice over: the raw-text positive survived the round
    # that fixed the negative.
    text = _read(sbatch)
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert re.search(r'ALIGN_TIMEOUT_S="\$\{ALIGN_TIMEOUT_S:\?', code), (
        f"{sbatch.name} must declare ALIGN_TIMEOUT_S as a REQUIRED export (:?), so a submit that "
        f"omits it aborts instead of aligning unbounded"
    )
    assert not re.search(
        r"ALIGN_TIMEOUT_S:-", code
    ), f"{sbatch.name} supplies a default bound — the round must name it explicitly"


@pytest.mark.parametrize("sbatch", [PRODUCER_SBATCH, MEASURE_SBATCH], ids=lambda p: p.name)
def test_every_align_shard_invocation_passes_the_bound(sbatch: Path):
    """The required export is worthless if the CLI call does not carry it — that combination
    reads as compliance while aligning unbounded ([[in-process-no-ops-look-like-compliance]])."""
    call = _align_shard_invocation(sbatch)
    assert (
        "--align-timeout-s" in call
    ), f"{sbatch.name} declares ALIGN_TIMEOUT_S but its align-shard call does not pass it"
    assert (
        "$ALIGN_TIMEOUT_S" in call
    ), f"{sbatch.name} passes a literal bound rather than the exported one"


def test_the_invocation_extractor_ignores_a_commented_command(tmp_path: Path):
    """Negative control for `_align_shard_invocation`. Without the comment filter it matched a
    HEADER SENTENCE about `align-shard` and asserted against prose — green, and blind to the real
    command. Sabotage: drop the `startswith("#")` filter and this test goes red."""
    fake = tmp_path / "fake.sbatch"
    fake.write_text(
        "#!/bin/bash\n"
        "# `covariation_producer align-shard` is discussed here, unbounded and uncalled\n"
        "#   python -m tbox_finder.mining.covariation_producer align-shard --shard x\n"
        "python -m tbox_finder.mining.covariation_producer align-shard \\\n"
        '  --shard "$S" --align-timeout-s "$ALIGN_TIMEOUT_S"\n',
        encoding="utf-8",
    )
    call = _align_shard_invocation(fake)
    assert "--align-timeout-s" in call
    assert "discussed here" not in call


def test_the_orchestrator_forwards_the_bound_to_the_producer_array():
    """mine_round.sbatch leg (c) submits the producer, which declares ALIGN_TIMEOUT_S `:?`. If the
    --export omits it, EVERY array task aborts on its own config line — a 48-task no-op that
    reports nothing. This is the sibling call site P3-15'-e's cpus-per-task fix had to chase."""
    text = _read(ORCHESTRATOR_SBATCH)
    submit = re.search(r"sbatch[^\n]*(?:\\\s*\n[^\n]*)*?mine_round_producer\.sbatch", text)
    assert submit, "no producer submit found in mine_round.sbatch"
    assert "ALIGN_TIMEOUT_S=" in submit.group(
        0
    ), "leg (c) does not forward ALIGN_TIMEOUT_S — the producer array would abort every task"


def test_the_producer_bound_is_documented_as_sparing_not_failing():
    """A reader deciding the value must see the consequence. `unavailable` ⇒ spared costs YIELD;
    if it were read as `failed` ⇒ mined, a short bound would inject false hard negatives."""
    text = _read(PRODUCER_SBATCH).lower()
    assert "unavailable" in text and "spared" in text


def test_a_bounded_run_still_reports_a_nonzero_exit_as_a_failure():
    """Regression seam: `_run`'s bounded arm rebuilds the CompletedProcess by hand, so it could
    lose the returncode check that the unbounded arm gets from subprocess.run."""
    with pytest.raises(HomologDbError) as exc:
        hdb._run([sys.executable, "-c", "raise SystemExit(7)"], timeout_s=SLOW)
    assert not isinstance(exc.value, ToolTimeoutError)
    assert "(7)" in str(exc.value)


def test_a_bounded_run_captures_stdout_and_stderr():
    """The bounded arm must be a drop-in: HomologDbError embeds proc.stderr, and R-scape/nhmmer
    parsers read proc.stdout."""
    proc = hdb._run(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        timeout_s=SLOW,
    )
    assert proc.stdout.strip() == "out"
    assert proc.stderr.strip() == "err"
    assert isinstance(proc, subprocess.CompletedProcess)


# ═════════════════════════════════════════════════════════════════════════════
# Review round 1 (CodeRabbit CLI) — the four behaviours those findings named
# ═════════════════════════════════════════════════════════════════════════════
def test_a_nan_bound_cannot_masquerade_as_a_bound():
    """THE ONE A `<= 0` CHECK LETS THROUGH. Every comparison against `nan` is False, so
    `nan <= 0` is False and a naive guard waves it past; `communicate(timeout=nan)` then never
    fires. The bound would be *declared, recorded in the round's provenance, and inert* — the
    unbounded behaviour that lost shard 016, wearing the fix's clothes."""
    assert not (float("nan") <= 0), "premise: nan slips past a <=0 guard, which is why isfinite"
    with pytest.raises(ValueError, match="finite"):
        hdb.assert_usable_timeout(float("nan"))


def test_a_stale_msa_is_discarded_when_the_homolog_set_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """THE THIRD SPARE BRANCH — I had fixed two of three. `insufficient_homologs` `continue`s
    before the try/except, so a stale msa.sto under a re-used workdir survived it and would be
    promoted and scored ([[fixed-one-of-two-identical-things]])."""
    spec = cp.CandidateSpec(
        candidate_id="GCA_000000001.1:c0:0:10-20",
        accession="GCA_000000001.1:c0",
        locus_start=10,
        locus_end=20,
    )
    wd = cp.candidate_workdir(tmp_path, spec.candidate_id)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "search.json").write_text('{"sufficient": false, "n_homologs": 3}\n', encoding="utf-8")
    stale = wd / cp.MSA_FILENAME
    stale.write_text("# STOCKHOLM 1.0\n#=GC SS_cons <<>>\n//\n", encoding="utf-8")

    def _must_not_align(**_kwargs: object) -> None:  # pragma: no cover - the failure case
        raise AssertionError("an insufficient homolog set must not be aligned")

    monkeypatch.setattr(cp, "align_candidate", _must_not_align)
    (row,) = cp.align_shard([spec], workroot=tmp_path, align_timeout_s=600.0)

    assert row["reason"] == "insufficient_homologs"
    assert not stale.exists(), "the stale MSA survived an insufficient-homologs skip"


def test_an_unusable_bound_is_refused_before_any_candidate_is_touched(tmp_path: Path):
    """Validated once up front, not on the first sufficient candidate — otherwise a shard would
    align an arbitrary number of candidates before discovering its bound is unusable."""
    spec = cp.CandidateSpec(
        candidate_id="GCA_000000001.1:c0:0:10-20",
        accession="GCA_000000001.1:c0",
        locus_start=10,
        locus_end=20,
    )
    # No workdir, no search.json: reaching the loop at all would raise something OTHER than this.
    with pytest.raises(ValueError, match="finite"):
        cp.align_shard([spec], workroot=tmp_path, align_timeout_s=float("nan"))


@pytest.mark.parametrize("bad", ["0", "-5", "nan", "inf", "abc"])
def test_the_cli_refuses_an_unusable_bound(bad: str, capsys: pytest.CaptureFixture[str]):
    """argparse `type=float` alone accepts nan/inf. Refusing at the parser kills the shard before
    the ~37 min search stage rather than after it."""
    with pytest.raises(SystemExit) as exc:
        cp.main(["align-shard", "--shard", "s.json", "--workroot", "w", "--align-timeout-s", bad])
    assert exc.value.code == 2
    assert "--align-timeout-s" in capsys.readouterr().err


def test_the_post_kill_drain_is_bounded(monkeypatch: pytest.MonkeyPatch):
    """A grandchild that escaped the process group (e.g. it called setsid itself) still holds the
    inherited stdout/stderr write ends, so an unbounded `communicate()` would read toward an EOF
    that never arrives — an unbounded wait INSIDE the code path whose job is to bound one."""
    calls: list[float | None] = []

    class _Popen:
        returncode = -9

        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _Popen:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            calls.append(timeout)
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)

    monkeypatch.setattr(hdb.subprocess, "Popen", _Popen)
    monkeypatch.setattr(hdb, "_kill_process_group", lambda popen: None)
    with pytest.raises(ToolTimeoutError):
        hdb._run_bounded(["x"], 1.0)

    assert calls == [1.0, hdb.DRAIN_TIMEOUT_S], (
        "the post-kill drain must carry a bound of its own; an unbounded second communicate() "
        "can hang forever on a pipe held open by a process that escaped the killed group"
    )


@pytest.mark.parametrize(
    "sbatch", [PRODUCER_SBATCH, ORCHESTRATOR_SBATCH, MEASURE_SBATCH], ids=lambda p: p.name
)
def test_the_documented_submit_line_exports_the_bound(sbatch: Path):
    """An operator follows the header's SUBMIT line. If it omits a `:?` export, the documented
    command aborts the job it documents — and for mine_round.sbatch that abort happens only after
    leg (a) has already spent 8 GPUs on the scan."""
    lines = _read(sbatch).splitlines()
    # The documented command spans backslash continuations, so a per-line search would look at
    # `# ssh two ... sbatch \\` alone and never see the --export on the next comment line.
    blocks: list[str] = []
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if not (stripped.startswith("#") and "sbatch" in ln and "--test-only" not in ln):
            continue
        block, j = [ln], i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            block.append(lines[j])
        blocks.append("\n".join(block))
    joined = "\n".join(blocks)
    assert blocks, f"no documented sbatch submit command found in {sbatch.name}"
    assert "ALIGN_TIMEOUT_S=" in joined, (
        f"{sbatch.name}'s documented submit command does not export ALIGN_TIMEOUT_S, which the "
        f"file itself declares required"
    )
