"""The covariation-producer array's SLURM envelope must equal ADR-0005 A10's pinned one.

Why this file exists
--------------------
``slurm/p2/mine_round_producer.sbatch`` shipped ``#SBATCH --cpus-per-task=4`` from
P2-10e-msa-producer until P3-15'-e, while **ADR-0005 A10 Phase-2** — signed 2026-07-28 —
pins *"Array width = 48, cpus-per-task = 2 for ``slurm/p2/mine_round_producer.sbatch`` on
one 96-core gpu-partition node … All 48 tasks run concurrently (48 x 2 = 96 cores, no
queueing)"*.  ``slurm/p2/mine_round.sbatch``'s leg (c) submits the array **without** a
``--cpus-per-task`` override, so the header was authoritative and a full-width submit
would have asked for 48 x 4 = 192 cores on a 96-core node.  The pinned no-queueing
property could not hold, and nothing in the suite said so: the envelope lived only in
prose on one side and in ``#SBATCH`` bytes on the other, with no test between them.

The pin is **re-derived from the ADR's own text**, never retyped
---------------------------------------------------------------
A test that hardcodes ``2`` pins this file to a number a future amendment could move,
and would then enforce a stale value against a correctly-updated sbatch.  So the numbers
come out of ``docs/decisions/ADR-0005-non-circular-eval-design.md`` at read time and the
sbatch is compared to *them*.

That makes the parse load-bearing, and a regex that silently matches nothing would make
every comparison vacuous.  Two guards:

* :func:`test_the_adr_pin_is_actually_parsed` is the positive control — it asserts the
  parse found a pin at all, so a reworded ADR fails loudly instead of passing emptily.
* The ADR states the envelope **twice**, in the Phase-2 pin bullet and again in the
  Phase-2 sign-off line, written independently of each other.
  :func:`test_the_adrs_two_statements_of_the_pin_agree` parses both and requires them to
  agree, so a single edited sentence cannot quietly move the pin this file enforces.

What is deliberately NOT pinned here
------------------------------------
``--mem``.  A10 prices CPU only and no per-task RSS has ever been measured (the K=50
Phase-1 job ran a single task at ``--mem=128G``), so 16G is an unmeasured operational
value, not a pinned one; asserting it would dress a guess as a decision.  Its real
consequence — memory, not CPU, bounds how many of the 48 tasks co-schedule — is recorded
in the sbatch header rather than enforced here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ADR = REPO / "docs" / "decisions" / "ADR-0005-non-circular-eval-design.md"
PRODUCER = REPO / "slurm" / "p2" / "mine_round_producer.sbatch"
ORCHESTRATOR = REPO / "slurm" / "p2" / "mine_round.sbatch"

#: The A10 Phase-2 pin bullet: "**Array width = 48, cpus-per-task = 2** for
#: `slurm/p2/mine_round_producer.sbatch`".  The sbatch path is part of the match so a
#: sizing sentence about some *other* array can never be read as this one's envelope.
_PIN_BULLET = re.compile(
    r"Array width\s*=\s*(?P<width>\d+),\s*cpus-per-task\s*=\s*(?P<cpus>\d+)\*\*\s*"
    r"for\s*`slurm/p2/mine_round_producer\.sbatch`"
)
#: The A10 Phase-2 sign-off restatement: "array width 48 x cpus-per-task 2".  Written as
#: its own sentence in the sign-off line, so it is an independent witness of the pin.
_PIN_SIGNOFF = re.compile(r"array width\s*(?P<width>\d+)\s*×\s*cpus-per-task\s*(?P<cpus>\d+)")
#: An ``#SBATCH --flag=value`` / ``#SBATCH --flag value`` directive.
_SBATCH = re.compile(r"^#SBATCH\s+(--[A-Za-z0-9-]+)(?:[=\s]+(\S+))?\s*$", re.M)
#: A ``--cpus-per-task`` passed on a ``sbatch`` command line inside another script.
_SUBMIT_CPUS = re.compile(r"--cpus-per-task[=\s]+(\S+)")


def _read(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(REPO)} is missing"
    return path.read_text(encoding="utf-8")


def _pin(pattern: re.Pattern[str]) -> tuple[int, int] | None:
    match = pattern.search(_read(ADR))
    if match is None:
        return None
    return int(match.group("width")), int(match.group("cpus"))


def _directives(text: str) -> dict[str, str | None]:
    """``#SBATCH`` directives as ``{flag: value}``; a bare flag maps to ``None``."""
    return {flag: value for flag, value in _SBATCH.findall(text)}


# ═════════════════════════════════════════════════════════════════════════════
# The positive controls — without these every comparison below can pass vacuously
# ═════════════════════════════════════════════════════════════════════════════
def test_the_adr_pin_is_actually_parsed():
    """A regex that matches nothing would make every other test in this file empty."""
    assert _pin(_PIN_BULLET) is not None, (
        "the ADR-0005 A10 Phase-2 pin bullet did not parse — this file's comparisons "
        "would all be vacuous. Re-read the amendment and fix _PIN_BULLET rather than "
        "deleting the assertion."
    )
    assert _pin(_PIN_SIGNOFF) is not None, (
        "the ADR-0005 A10 Phase-2 sign-off restatement did not parse — the "
        "two-witness cross-check below would be vacuous."
    )


def test_the_adrs_two_statements_of_the_pin_agree():
    """The bullet and the sign-off are independent sentences; they must say one thing.

    If they ever disagree the pin is ambiguous, and an sbatch cannot be certified
    against an ambiguous pin (CLAUDE.md §7: an ambiguous ADR is a stop, not a
    judgement call).
    """
    bullet, signoff = _pin(_PIN_BULLET), _pin(_PIN_SIGNOFF)
    assert bullet == signoff, (
        f"ADR-0005 A10 Phase-2 states its envelope twice and the two disagree: "
        f"pin bullet {bullet} vs sign-off {signoff}"
    )


def test_the_producer_sbatch_is_the_file_the_adr_names():
    assert PRODUCER.is_file(), "the ADR pins an envelope for a file that does not exist"


# ═════════════════════════════════════════════════════════════════════════════
# The envelope itself
# ═════════════════════════════════════════════════════════════════════════════
def test_cpus_per_task_equals_the_adr_pin():
    """The regression P3-15'-e fixed: the header read 4 against a pinned 2."""
    _, pinned_cpus = _pin(_PIN_BULLET)
    directives = _directives(_read(PRODUCER))
    assert "--cpus-per-task" in directives, (
        "the producer array declares no --cpus-per-task, so the cluster default decides "
        "how many cores each nhmmer/mlocarna shard gets"
    )
    assert int(directives["--cpus-per-task"]) == pinned_cpus, (
        f"--cpus-per-task={directives['--cpus-per-task']} contradicts ADR-0005 A10's "
        f"pinned {pinned_cpus}; at the pinned array width that changes how many tasks "
        f"co-schedule on a 96-core node"
    )


def test_the_allocation_covers_the_threads_the_body_actually_asks_for():
    """The pin is only sound because the body consumes exactly that many threads.

    ``align-shard`` is invoked with an explicit ``--cpu``; ``homolog_msa.nhmmer_argv``
    passes no ``--cpu``, so nhmmer runs at HMMER's own default.  Whatever the align
    stage asks for must fit inside the allocation, or the pin under-provisions the run
    it is meant to size.
    """
    text = _read(PRODUCER)
    allocated = int(_directives(text)["--cpus-per-task"])
    align_cpu = re.search(r"align-shard\b[^\n]*(?:\\\s*\n[^\n]*)*?--cpu\s+(\d+)", text)
    assert align_cpu is not None, (
        "no `align-shard … --cpu N` found in the producer sbatch — either the align "
        "stage stopped declaring its thread count, or this pattern went stale and the "
        "check below is vacuous"
    )
    assert int(align_cpu.group(1)) <= allocated, (
        f"align-shard asks for {align_cpu.group(1)} threads but the task is allocated "
        f"{allocated} cores"
    )


def test_the_orchestrator_does_not_contradict_the_pin():
    """Leg (c)'s submit line is the other place the effective cpus-per-task is decided.

    It currently passes none, which is why the header is authoritative and why the
    shipped 4 mattered.  If a future edit adds an override it must carry the pinned
    value, not a fresh one.
    """
    _, pinned_cpus = _pin(_PIN_BULLET)
    text = _read(ORCHESTRATOR)
    leg_c = re.search(r"sbatch[^\n]*(?:\\\s*\n[^\n]*)*mine_round_producer\.sbatch", text)
    assert leg_c is not None, (
        "leg (c)'s submit of mine_round_producer.sbatch was not found in "
        "mine_round.sbatch — this check would be vacuous"
    )
    override = _SUBMIT_CPUS.search(leg_c.group(0))
    if override is not None:
        assert int(override.group(1)) == pinned_cpus, (
            f"leg (c) submits the producer array with --cpus-per-task="
            f"{override.group(1)}, overriding the header and contradicting ADR-0005 "
            f"A10's pinned {pinned_cpus}"
        )


@pytest.mark.parametrize("flag", ["--gres", "--nodelist", "--account", "--qos"])
def test_the_producer_array_declares_no_forbidden_directive(flag: str):
    """A10: CPU-only (**no `--gres`**). CLAUDE.md §9.2/§13: never pin a node, no account/QOS.

    ``--gres`` is the one that costs science rather than tidiness: a GPU allocation on a
    CPU-only leg idles a scarce A4000 for the whole array while cryosparc already keeps
    both nodes fragmented.
    """
    assert flag not in _directives(
        _read(PRODUCER)
    ), f"{flag} is declared in slurm/p2/mine_round_producer.sbatch"


def test_the_producer_array_runs_on_the_gpu_partition():
    """§9.2: ``compute`` is node `zero` alone (no GPU, shared gateway) and is forbidden.

    CPU-only work still belongs on ``gpu`` — it is where node `two`'s 96 cores live.
    """
    assert _directives(_read(PRODUCER)).get("--partition") == "gpu"
