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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["clauses_not_required_at", "check_schema_tables"]


def clauses_not_required_at(
    schema: str,
    *,
    known: Sequence[str],
    first_required_at: Mapping[str, frozenset[str]],
) -> frozenset[str]:
    """Clauses introduced AFTER ``schema``, which a report at that schema cannot carry.

    ``known`` is oldest-first and is *indexed*, never compared: ``"3" < "4"`` holds only while
    both are one character, which is the string-version-compare trap this repo has already been
    bitten by once.

    An **unknown** schema excuses nothing — a report this validator does not recognise is
    graded against the whole current clause set, and its version is flagged separately.
    """
    if schema not in known:
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
