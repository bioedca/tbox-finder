"""Schema-age arithmetic for report clause sets, in one place.

A clause set is part of a report's shape, so adding a clause invalidates every committed
report ([[new-gate-clause-invalidates-old-reports]]). When the artifact cannot be regenerated —
the P3-17 sizing report needs an A4000, and no local environment carries ``torch`` **and**
``pyarrow`` together — the alternative is to *version* the clause set and grade each report
against the set of its own schema.

Three modules do that now (``stage2.sizing``, ``stage2.eval``, ``calib.gate2``) and the age
arithmetic was copied into each. Copies drift: if one module gained a rule the others did not,
the same legacy artifact would grade differently depending on which validator read it. The
tables stay per-module — they are per-module facts — and only the arithmetic lives here.

The same file carries the **shape** half of a validator's contract: :func:`as_mapping` and
:func:`shape_problems`. A validator's job is a verdict, and a report that is not even shaped
like the producer's output is the case it most needs one for; an ``AttributeError`` out of
``(x or {}).get`` is not a verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

__all__ = [
    "EACH",
    "MALFORMED",
    "as_mapping",
    "check_schema_tables",
    "clauses_not_required_at",
    "is_count",
    "is_list",
    "is_mapping",
    "is_real",
    "nullable",
    "shape_problems",
]

#: A path step that fans out over every entry of a list or every value of a mapping — e.g.
#: every arm under ``arms``, or every point under ``measurements``.
EACH = "*"

#: The prefix every :func:`shape_problems` message carries, so a reader can tell a malformed
#: report from a well-formed one whose clauses disagree.
MALFORMED = "malformed:"


def clauses_not_required_at(
    schema: str,
    *,
    known: Sequence[str],
    first_required_at: Mapping[str, frozenset[str]],
) -> frozenset[str]:
    """Clauses introduced AFTER ``schema``, which a report at that schema cannot carry.

    ``known`` is oldest-first and is *indexed*, never compared: ``"3" < "4"`` holds only while
    both are one character, which is the string-version-compare trap this repo has already been
    bitten by once. ``schema`` must be an actual ``str``: a JSON number ``1`` is not schema
    ``"1"``, and coercing it would hand a report an exemption its recorded version never
    claimed.

    An **unknown** schema excuses nothing — a report this validator does not recognise is
    graded against the whole current clause set, and its version is flagged separately.
    """
    # `schema is not a str` is checked, not coerced. `str(1)` == `"1"`, so a JSON *number*
    # would otherwise collect the schema-1 exemptions — omission of a clause included — while
    # the version check flagged only the type. A value this validator cannot recognise excuses
    # nothing, whatever it looks like once stringified.
    if not isinstance(schema, str) or schema not in known:
        return frozenset()
    age = list(known).index(schema)
    return frozenset().union(
        *(first_required_at.get(newer, frozenset()) for newer in list(known)[age + 1 :]),
        frozenset(),
    )


def check_schema_tables(
    *,
    known: Sequence[str],
    first_required_at: Mapping[str, frozenset[str]],
    current: str,
    module: str,
) -> None:
    """Refuse a table that cannot do its job, at import time.

    Two silent failures are possible and both surface only as a legacy artifact that suddenly
    fails validation: a schema key outside ``known`` is ignored by
    :func:`clauses_not_required_at`, and the current schema listing itself excuses a clause the
    current code requires. Clause *names* are checked in the unit tests, where
    ``derive_clauses`` can be called on a real report.
    """
    unknown = sorted(set(first_required_at) - set(known))
    if unknown:
        raise ValueError(
            f"{module}: CLAUSES_FIRST_REQUIRED_AT names schemas {unknown!r} that KNOWN_SCHEMAS "
            f"{tuple(known)!r} does not list, so they would be silently ignored"
        )
    if current not in known:
        raise ValueError(
            f"{module}: SCHEMA_VERSION {current!r} is not in KNOWN_SCHEMAS {tuple(known)!r}"
        )
    if clauses_not_required_at(current, known=known, first_required_at=first_required_at):
        raise ValueError(
            f"{module}: the current schema {current!r} excuses clauses, which means "
            "KNOWN_SCHEMAS lists a schema newer than SCHEMA_VERSION"
        )


def as_mapping(value: Any) -> Mapping[str, Any]:
    """``value`` when it is a mapping, else an empty one.

    The ``x or {}`` idiom raises at the next ``.get`` on a TRUTHY non-mapping (a string, a
    number, a list). Reading it as absent instead is a verdict every clause already fails
    closed on, and on every value the idiom did not raise on the two agree — so swapping one
    for the other moves no clause. Absence is not acceptance: :func:`shape_problems` reports
    the malformed block separately, so the validator still objects to it.
    """
    return value if isinstance(value, Mapping) else {}


def is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def is_list(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def is_count(value: Any) -> bool:
    """A non-negative ``int`` — never a ``bool``, which ``isinstance(True, int)`` lets through."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_real(value: Any) -> bool:
    """A number a float can hold — never a ``bool``.

    ``float(10**400)`` and ``math.isfinite(10**400)`` RAISE ``OverflowError``, so an over-large
    JSON integer is refused here instead of crashing the caller. NaN and ±inf ARE floats and
    pass: whether one is acceptable is a clause's call — a diverged temperature fit is an
    honest failure, not a malformed report.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        float(value)
    except OverflowError:
        return False
    return True


def nullable(ok: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """``ok``, widened to accept ``None`` — for a field the producer legitimately writes as
    null (an OOM point's peak, a block a legacy schema left empty)."""
    return lambda value: value is None or ok(value)


def _walk(node: Any, path: Sequence[str], seen: tuple[str, ...]) -> Iterator[tuple[str, Any]]:
    if not path:
        yield "/".join(seen), node
        return
    step, rest = path[0], path[1:]
    if step == EACH:
        if isinstance(node, Mapping):
            for key, child in node.items():
                yield from _walk(child, rest, (*seen, str(key)))
        elif is_list(node):
            for index, child in enumerate(node):
                yield from _walk(child, rest, (*seen, str(index)))
    elif isinstance(node, Mapping) and step in node:
        yield from _walk(node[step], rest, (*seen, step))


def shape_problems(
    report: Mapping[str, Any],
    rules: Sequence[tuple[Sequence[str], Callable[[Any], bool], str]],
) -> list[str]:
    """One problem per PRESENT value, at each rule's path, that is not the kind it names.

    Absence is not a shape problem: whether a field is required is a clause's business, and
    legacy schemas legitimately lack fields the current producer writes. ``None`` IS one,
    unless the rule wraps its predicate in :func:`nullable` — a null count is not "no count",
    and reading it as one is how a validator that stopped raising would start passing. A path
    whose parent is malformed is not descended into; the parent is reported at its own rule.
    """
    problems: list[str] = []
    for path, ok, kind in rules:
        for where, value in _walk(report, path, ()):
            if not ok(value):
                problems.append(f"{MALFORMED} {where} is {type(value).__name__}, not {kind}")
    return problems
