"""Hard-negative-mining infrastructure for the Stage-1 scanner (P2-07+).

The subpackage top-level stays import-light (pure stdlib) so the unit tier runs
in bare CI without pandas/numpy/torch. Heavy backends (parquet readers, the
Infernal/R-scape shell-outs) are imported lazily inside the functions that need
them.

Modules
-------
``spare_rule``
    The ADR-0005 D14 / ADR-0006 D11 mining-exclusion **spare rule**, lifted from
    two-valued booleans to **three-valued disjunct evidence** (passed / failed /
    **unavailable**) plus the per-round **readiness gate**. The pure predicate it
    delegates to lives in :mod:`tbox_finder.masking`; this module never re-derives
    it (one rule, one implementation).
``hard_negative``
    Collection of Stage-1 false positives from the §9.1 negative/decoy pools into
    a mined hard-negative pool, after union-prior locus masking and the spare-rule
    gate. PRD §9.1; ADR-0005 D14.
"""

from __future__ import annotations
