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
#:
#: **Anchored on the sign-off's own label.** Unanchored, this phrase would match the same
#: wording in *any* later amendment's sign-off — a sizing decision about some other array
#: could then be read as agreeing, or disagreeing, with A10's producer envelope, and the
#: two-witness cross-check below would be comparing two different pins.
_PIN_SIGNOFF = re.compile(
    r"\*\*Sign-off \(A10 Phase-2\):\*\*[^\n]*?"
    r"array width\s*(?P<width>\d+)\s*×\s*cpus-per-task\s*(?P<cpus>\d+)"
)
#: The ADR's own arithmetic for *why* the width is 48: "48 × 2 = 96 cores, no queueing" on
#: "one 96-core gpu-partition node".  Parsed so the pinned width is checked against the
#: reason it was pinned rather than merely read and discarded.
#:
#: The ADR writes U+00D7; ASCII ``x`` is accepted too so a later prose normalisation is a
#: non-event.  Either way the failure is fail-CLOSED — an unparseable sentence trips
#: :func:`test_the_pinned_width_is_checked_against_the_reason_it_was_pinned`'s own
#: not-None assertion rather than passing vacuously.
_PIN_ARITHMETIC = re.compile(
    r"All (?P<width>\d+) tasks run concurrently \((?P<w2>\d+)\s*[x×]\s*(?P<cpus>\d+)\s*=\s*"
    r"(?P<cores>\d+) cores, no queueing\)"
)
#: An ``#SBATCH`` directive.  Short flags are matched too: ``#SBATCH -G 1`` is a GPU
#: request that a ``--``-only pattern does not see at all, so the forbidden-directive
#: check below would pass on a leg that had quietly acquired a GPU.
_SBATCH = re.compile(r"^#SBATCH\s+(--?[A-Za-z0-9-]+)(?:[=\s]+(\S+))?\s*$", re.M)
#: ``--cpus-per-task`` **or its short form ``-c``** on an ``sbatch`` command line.  sbatch
#: accepts both, so a pattern that reads only the long form certifies a submit line that
#: can override the pin in one character.
_SUBMIT_CPUS = re.compile(r"(?<![\w-])(?:--cpus-per-task|-c)[=\s]+(\S+)")
#: ``--partition`` or its short form ``-p`` on an ``sbatch`` command line.
_SUBMIT_PARTITION = re.compile(r"(?<![\w-])(?:--partition|-p)[=\s]+(\S+)")
#: Every way an sbatch line can ask for a GPU.  ``--gres`` is only the one this repo
#: happens to use; a submit line or header carrying any of the others would take an A4000
#: for a CPU-only leg just as effectively.
_GPU_REQUEST_FLAGS: tuple[str, ...] = (
    "--gres",
    "--gpus",
    "--gpus-per-node",
    "--gpus-per-socket",
    "--gpus-per-task",
    "-G",
)
#: Directives the producer array must never carry, **long and short spelling both**.
#: sbatch's short forms are not aliases this file may ignore: ``-w`` pins a node just as
#: ``--nodelist`` does, and ``-A``/``-q`` name an account/QOS that does not exist here.
#: Listing only the long forms is the same blind spot ``-G`` had one round ago.
_FORBIDDEN_DIRECTIVES: tuple[str, ...] = (
    *_GPU_REQUEST_FLAGS,
    "--nodelist",
    "-w",
    "--account",
    "-A",
    "--qos",
    "-q",
)


def _read(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(REPO)} is missing"
    return path.read_text(encoding="utf-8")


def _pin(pattern: re.Pattern[str]) -> tuple[int, int] | None:
    match = pattern.search(_read(ADR))
    if match is None:
        return None
    return int(match.group("width")), int(match.group("cpus"))


def _directives(text: str) -> dict[str, str | None]:
    """``#SBATCH`` directives as ``{flag: value}``; a bare flag maps to ``None``.

    ``re.findall`` yields ``""`` for a group that did not participate, so the ``or None``
    is what makes the annotation true.  Without it a bare ``#SBATCH --cpus-per-task``
    reaches :func:`int` as ``""`` and the value check dies on ``ValueError`` instead of
    reporting which directive is malformed.
    """
    return {flag: (value or None) for flag, value in _SBATCH.findall(text)}


def _leg_c_submit() -> str:
    """The single ``sbatch … mine_round_producer.sbatch`` command in the orchestrator.

    Every check about how the producer is *submitted* must read this one span.  Scanning
    the whole file instead lets an unrelated line elsewhere satisfy a check while leg (c)
    itself does something else — which is exactly how a required ``ARRAY_WIDTH`` and a
    dynamic ``--array`` can both be present while the producer is submitted with a literal
    width.
    """
    text = _read(ORCHESTRATOR)
    # Comment lines are excluded rather than anchoring on "the line starts with sbatch":
    # the real command starts `PRODUCER_JOB="$(sbatch …`, so a start-of-line anchor would
    # match nothing and the "exactly one" assertion would fail on correct code.  This
    # file's sibling sbatch documents a standalone submit *in a comment*, so a commented
    # command is a realistic thing to accidentally validate instead of the executed one.
    matches = [
        m.group(0)
        for m in re.finditer(r"sbatch[^\n]*(?:\\\s*\n[^\n]*)*?mine_round_producer\.sbatch", text)
        if not text[text.rfind("\n", 0, m.start()) + 1 : m.start()].lstrip().startswith("#")
    ]
    assert len(matches) == 1, (
        f"expected exactly one executable `sbatch … mine_round_producer.sbatch` in "
        f"mine_round.sbatch, found {len(matches)}; with none every submit-line check is "
        f"vacuous, and with several this file would certify only the first"
    )
    return matches[0]


def _allocated_cpus(text: str) -> int:
    """The producer's declared CPU allocation — one declaration, either spelling, valued.

    Both the pin check and the thread-headroom check need this number, and reading it
    twice by hand is how one of the two ends up without the guard.

    Three things this cannot do, each of which would let the pin go unenforced:

    * **read only the long spelling.**  :data:`_SBATCH` sees ``-c`` — a lookup keyed on
      ``--cpus-per-task`` alone would call ``#SBATCH -c 2`` *missing*.
    * **read through a dict.**  Two declarations of the same flag collapse to one key, so
      a duplicate is invisible; sbatch applies the *later* one, which need not be the one
      that matched the pin.
    * **tolerate a mix of spellings.**  ``--cpus-per-task=2`` next to ``-c 4`` would
      satisfy a long-form check while the job actually runs at 4.
    """
    declarations = [
        (flag, value) for flag, value in _SBATCH.findall(text) if flag in ("--cpus-per-task", "-c")
    ]
    assert len(declarations) == 1, (
        f"the producer array declares the CPU allocation {len(declarations)} times "
        f"({declarations or 'not at all'}); none leaves the cluster default deciding, "
        f"and several means sbatch applies the last one, not the one checked here"
    )
    flag, value = declarations[0]
    assert value, (
        f"{flag} is declared with no value; sbatch would reject the job and the pin "
        f"would be unenforced"
    )
    return int(value)


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


def test_the_pinned_width_is_checked_against_the_reason_it_was_pinned():
    """The width must decide something here, or parsing it is decoration.

    It cannot be asserted against the sbatch header: ``#SBATCH --array=0`` is a
    deliberate one-task placeholder that the submit overrides
    (``mine_round.sbatch`` builds ``0-$((ARRAY_WIDTH - 1))`` from a required
    ``${ARRAY_WIDTH:?}`` export), so requiring 48 in the header would enforce a width
    the design keeps out of the file on purpose.

    What the width *does* decide is A10's own justification — ``48 x 2 = 96 cores, no
    queueing`` on ``one 96-core gpu-partition node``.  Checking the pin against that
    arithmetic means an amendment that moves the width to 24 while leaving the
    "no queueing" sentence standing fails here instead of shipping a pin whose stated
    reason no longer holds.
    """
    pinned_width, pinned_cpus = _pin(_PIN_BULLET)
    match = _PIN_ARITHMETIC.search(_read(ADR))
    assert match is not None, (
        "A10 Phase-2's concurrency arithmetic did not parse — the width would then be "
        "read from the ADR and never checked against anything"
    )
    width, w2, cpus, cores = (int(match.group(k)) for k in ("width", "w2", "cpus", "cores"))
    assert (width, cpus) == (pinned_width, pinned_cpus), (
        f"A10's concurrency sentence describes {width} x {cpus} but the pin bullet says "
        f"{pinned_width} x {pinned_cpus}"
    )
    assert width == w2, f"A10's own sentence names {width} tasks then multiplies {w2}"
    assert width * cpus == cores, (
        f"A10's no-queueing claim does not multiply out: {width} x {cpus} = "
        f"{width * cpus}, not the {cores} cores it states"
    )


def test_the_array_width_reaches_the_producer_only_through_a_required_export():
    """Nothing may supply the width silently — the operator must state it.

    The producer header's ``--array=0`` is a placeholder, so if leg (c) could fall back
    to a default width a short array would run, write a complete-looking set of shard
    tables, and every unprocessed candidate would resolve ``unavailable`` => spared:
    the round would decide less than it reports while every count still reconciled.
    """
    assert re.search(r'ARRAY_WIDTH="?\$\{ARRAY_WIDTH:\?', _read(ORCHESTRATOR)), (
        "mine_round.sbatch no longer requires ARRAY_WIDTH with no default; a defaulted "
        "width can silently under-run the array"
    )
    # Read from leg (c)'s OWN command, not from the file at large: a dynamic --array
    # somewhere else in the script would otherwise satisfy this while the producer is
    # submitted with a literal or defaulted width.
    submit = _leg_c_submit()
    array = re.search(r"(?<![\w-])(?:--array|-a)[=\s]+(\S+)", submit)
    assert array is not None, "leg (c) submits the producer array with no --array at all"
    assert "ARRAY_WIDTH" in array.group(1), (
        f"leg (c) submits --array={array.group(1)}, which does not derive from "
        f"ARRAY_WIDTH — the required export and the submitted width could then disagree"
    )


# ═════════════════════════════════════════════════════════════════════════════
# The envelope itself
# ═════════════════════════════════════════════════════════════════════════════
def test_cpus_per_task_equals_the_adr_pin():
    """The regression P3-15'-e fixed: the header read 4 against a pinned 2."""
    _, pinned_cpus = _pin(_PIN_BULLET)
    allocated = _allocated_cpus(_read(PRODUCER))
    assert allocated == pinned_cpus, (
        f"--cpus-per-task={allocated} contradicts ADR-0005 A10's pinned {pinned_cpus}; "
        f"at the pinned array width that changes how many tasks co-schedule on a "
        f"96-core node"
    )


def test_the_allocation_covers_the_threads_the_body_actually_asks_for():
    """The pin is only sound because the body consumes exactly that many threads.

    ``align-shard`` is invoked with an explicit ``--cpu``; ``homolog_msa.nhmmer_argv``
    passes no ``--cpu``, so nhmmer runs at HMMER's own default.  Whatever the align
    stage asks for must fit inside the allocation, or the pin under-provisions the run
    it is meant to size.
    """
    text = _read(PRODUCER)
    allocated = _allocated_cpus(text)
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
    submit = _leg_c_submit()
    override = _SUBMIT_CPUS.search(submit)
    if override is not None:
        assert int(override.group(1)) == pinned_cpus, (
            f"leg (c) submits the producer array with --cpus-per-task="
            f"{override.group(1)}, overriding the header and contradicting ADR-0005 "
            f"A10's pinned {pinned_cpus}"
        )
    # Every header directive this file certifies is overridable on the submit line, so
    # checking the header alone certifies a file the orchestrator can contradict.
    partition = _SUBMIT_PARTITION.search(submit)
    if partition is not None:
        assert partition.group(1) == "gpu", (
            f"leg (c) submits the producer array with --partition={partition.group(1)}, "
            f"overriding the header; §9.2 forbids `compute` (= node `zero`, no GPU)"
        )
    for flag in _FORBIDDEN_DIRECTIVES:
        assert not re.search(rf"(?<![\w-]){re.escape(flag)}[=\s]", submit), (
            f"leg (c) submits the producer array with {flag}, overriding the header on a "
            f"directive ADR-0005 A10 / CLAUDE.md §9.2 forbid for this leg"
        )


#: ``--nodelist`` pins a node (§9.2: `one` is often down); ``--account``/``--qos`` do not
#: exist here (accounting disabled); the GPU forms all take an A4000 A10 says not to take.
@pytest.mark.parametrize("flag", _FORBIDDEN_DIRECTIVES)
def test_the_producer_array_declares_no_forbidden_directive(flag: str):
    """A10: CPU-only (**no `--gres`**). CLAUDE.md §9.2/§13: never pin a node, no account/QOS.

    The GPU forms are the ones that cost science rather than tidiness: an allocation on a
    CPU-only leg idles a scarce A4000 for the whole array while cryosparc already keeps
    both nodes fragmented.  ``--gres`` is only the spelling this repo happens to use —
    ``-G 1`` requests a GPU just as effectively, and is invisible to a ``--``-only
    directive pattern, so :data:`_SBATCH` matches short flags too.
    """
    assert flag not in _directives(
        _read(PRODUCER)
    ), f"{flag} is declared in slurm/p2/mine_round_producer.sbatch"


def test_the_directive_parser_sees_short_flags():
    """Positive control for the check above: a pattern that cannot see ``-G`` passes it.

    Without this, narrowing :data:`_SBATCH` back to ``--``-only flags would leave every
    short-form GPU request silently unassertable while the suite stayed green.
    """
    parsed = _directives("#SBATCH -G 1\n#SBATCH --mem=8G\n")
    assert parsed == {"-G": "1", "--mem": "8G"}, (
        f"_SBATCH does not parse short flags; got {parsed}. The forbidden-directive "
        f"check would then be blind to `#SBATCH -G 1`."
    )


def test_the_submit_line_patterns_see_short_flags():
    """Positive control for the submit-line checks: ``-c 4`` overrides the pin too.

    Without this, narrowing :data:`_SUBMIT_CPUS` / :data:`_SUBMIT_PARTITION` back to the
    long spellings would leave a one-character override unassertable while the suite
    stayed green — the same blind spot ``-G`` had in the headers.  The negative legs are
    what make it a control rather than a restatement: a flag *embedded* in a longer token
    must not match, or the checks would fire on unrelated arguments.
    """
    line = "sbatch --parsable -p compute -c 4 -a 0-7 x.sbatch"
    assert _SUBMIT_CPUS.search(line).group(1) == "4"
    assert _SUBMIT_PARTITION.search(line).group(1) == "compute"
    for embedded in ("sbatch --export=ALL,MY-c=1 x.sbatch", "sbatch --no-p=2 x.sbatch"):
        assert _SUBMIT_CPUS.search(embedded) is None or "-c" not in embedded.split()[1]
        assert _SUBMIT_PARTITION.search(embedded) is None, (
            f"a partition pattern matched inside {embedded!r}; it would fire on "
            f"arguments that are not a partition"
        )


def test_the_cpu_allocation_reader_handles_aliases_and_duplicates():
    """Positive control for :func:`_allocated_cpus`'s three refusals.

    Each leg is a *fabricated* header rather than the real file, so the control keeps
    working when the real file is correct — which is exactly when a broken reader is
    invisible.  Review r3 raised all three: ``-c`` read as missing, duplicates hidden by
    a dict, and mixed spellings satisfying a long-form check while sbatch runs the other.
    """
    assert _allocated_cpus("#SBATCH -c 2\n") == 2, "the short alias must be read, not skipped"
    assert _allocated_cpus("#SBATCH --cpus-per-task=2\n") == 2

    for header, why in (
        ("#SBATCH --mem=8G\n", "no declaration at all"),
        ("#SBATCH --cpus-per-task=2\n#SBATCH --cpus-per-task=4\n", "a duplicate long form"),
        ("#SBATCH --cpus-per-task=2\n#SBATCH -c 4\n", "mixed spellings; sbatch takes the last"),
    ):
        with pytest.raises(AssertionError):
            _allocated_cpus(header)
        assert why  # named for the failure message, not asserted on


def test_the_leg_c_extractor_ignores_a_commented_submit():
    """A commented command must not be certified in place of the executed one.

    The sibling producer sbatch documents a standalone submit *in a comment*, so this is
    a realistic mistake rather than a hypothetical.  The real leg (c) line begins
    ``PRODUCER_JOB="$(sbatch …``, which is why the extractor filters comments instead of
    anchoring on a line that starts with ``sbatch``.
    """
    submit = _leg_c_submit()
    assert not submit.lstrip().startswith("#")
    assert "mine_round_producer.sbatch" in submit
    assert "ARRAY_WIDTH" in submit, (
        "the extracted span is not leg (c)'s real command — it does not even mention the "
        "width export the submit builds its --array from"
    )


def test_the_producer_array_runs_on_the_gpu_partition():
    """§9.2: ``compute`` is node `zero` alone (no GPU, shared gateway) and is forbidden.

    CPU-only work still belongs on ``gpu`` — it is where node `two`'s 96 cores live.
    """
    assert _directives(_read(PRODUCER)).get("--partition") == "gpu"
