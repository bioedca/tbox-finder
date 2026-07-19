"""Helpers shared by the ``synth`` generators (promoted at P2-08).

These were private to :mod:`tbox_finder.synth.tier2n` (P2-07). P2-08 needs the
same deterministic keying and the same provenance-path sanitiser, so they are
**promoted here and delegated to** rather than copied: a forked helper means
fixing one copy and shipping the bug in the other, which is a correctness rule
in this repo, not a tidiness preference.

Pure stdlib — importable in the bare-CI unit tier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_key(seed: int, *parts: str) -> int:
    """Deterministic 64-bit key from ``seed`` + ``parts`` (no global RNG state).

    ``shake_256`` over a ``|``-joined string. Callers fold a domain-separation
    token into ``parts`` so that two draws for the same record (e.g. an ordering
    draw and a control draw) are independent rather than perfectly correlated.

    A cheap polynomial hash is deliberately *not* used here: at P2-03 one whose
    varying term came last produced 8 logits within 1.3e-08 of each other, and
    the resulting fixture could not express what its assertions measured.
    """
    digest = hashlib.shake_256(("|".join((str(seed), *parts))).encode("utf-8")).digest(8)
    return int.from_bytes(digest, "big")


def repo_relative(path: str | Path) -> str:
    """``path`` relative to the repo root when it is inside it, else its basename.

    Never returns an absolute path: a committed report that embedded one would
    record the author's home directory as if it were provenance.
    """
    resolved = Path(path).resolve()
    for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (parent / ".git").exists():
            try:
                return str(resolved.relative_to(parent))
            except ValueError:
                break
    return resolved.name


def bad_bool(value: object, expected: bool) -> bool:
    """True when ``value`` is not a real ``bool`` equal to ``expected``.

    ``isinstance(True, int)`` holds, so a validator that merely compares ``==``
    accepts an ``int`` 0/1 (or a numpy ``bool_``) where a JSON ``true``/``false``
    was required. Gate clauses are read by humans and by CI from the committed
    JSON, so the type is part of the contract.
    """
    return not isinstance(value, bool) or value is not expected
