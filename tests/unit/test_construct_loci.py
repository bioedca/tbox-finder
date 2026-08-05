"""P3-12 — the ADR-0005 D3 locus-construction rule (`infer/locus.py::construct_loci`).

What this file is for
---------------------
D3 [ADR-0005:46] lists five knobs. `infer/call.py::call_candidates` shipped three of them at
P2-10c′-c-i (global threshold, gap-merge, min span); this step adds the **threshold scope**,
the **required-element co-occurrence** and the **flank**, and composes the existing merge
rather than forking it. So this file has to prove three separate things, and they need
different kinds of test:

1. **The arithmetic is right** — spans, merges, span cuts, co-occurrence counts and flank
   clipping, on a hand-checked fixture whose every expected number is written out below.
2. **The composition is real** — `locus.py` calls the *shared* merge with the *right
   arguments*, and no second merge exists in the tree. AST pins, compared by **contents**
   (`ast.unparse` against the literal call) rather than by shape: P3-11's review found a pin
   that verified one argument of three, which `f(other_logits, stale_starts, seq_len)` passed.
3. **The rule is recall-favouring in the way D3 means** — the co-occurrence predicate is a
   *count*, structurally unable to name a canonical element set (PRD §5/§13.3), and at
   `min_distinct_elements <= 1` it discards nothing the bare caller found.

The reference-and-sabotage discipline
-------------------------------------
Following `test_infer_call.py`: every ordering invariant is paired with an independent
reference implementation (`_ref_loci`) plus **variants with one decision flipped**, each shown
to actually disagree on this fixture. A boundary test no wrong operator can fail is a
tautology, not a guard. The reference walks the *layout spec* in pure Python, never the
shipped arrays, so it and `construct_loci` share no code — only the fixture's probability
formula, which `test_fixture_assignment_reproduces_the_requested_layout` pins independently.

`numpy`-only and torch-free: this runs in the bare CI Tier-1 environment, like the module it
tests. PRD §6, §13.1, §5/§13.3; ADR-0005 D3 + A3.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from collections.abc import Sequence

import numpy as np
import pytest

from tbox_finder.infer import call as cl
from tbox_finder.infer import locus as lc
from tbox_finder.labels import CLASS_INDEX

# ══════════════════════════════════════════════════════════════════════════════════════════
# Fixture: a layout spec → reconciled log-posteriors
# ══════════════════════════════════════════════════════════════════════════════════════════

#: Share of a position's element mass held by its named class. 0.94 leaves 0.01 of the mass on
#: each of the other six, so every probability is strictly positive (no `log(0)`, which
#: `element_posterior` refuses) and the named class is the unambiguous arg-max.
_SHARE = 0.94

#: Element posterior on a background position. Spread evenly over the seven element classes,
#: so a background position has no dominant element and cannot clear any threshold used here.
_BG_P_ELEM = 0.05

#: Element posterior on an element position.
_EL_P_ELEM = 0.95

_ELEM = tuple(int(i) for i in cl.ELEMENT_INDICES)
_N_EL = len(_ELEM)


def _element_probs(name: str | None, p_elem: float, share: float = _SHARE) -> np.ndarray:
    """The seven element-class probabilities, in `cl.ELEMENT_INDICES` order, summing to p_elem."""
    if name is None:
        return np.full(_N_EL, p_elem / _N_EL, dtype=np.float64)
    probs = np.full(_N_EL, (1.0 - share) * p_elem / (_N_EL - 1), dtype=np.float64)
    probs[_ELEM.index(CLASS_INDEX[name])] = share * p_elem
    return probs


def _rows(spec: Sequence[tuple[str | None, int, float]]) -> list[tuple[str | None, float]]:
    """Expand `(class name or None, n positions, p_elem)` segments into one row per position."""
    out: list[tuple[str | None, float]] = []
    for name, n, p_elem in spec:
        out.extend([(name, p_elem)] * n)
    return out


def _log_probs(rows: Sequence[tuple[str | None, float]]) -> np.ndarray:
    """`(len(rows), NUM_CLASSES)` log-posterior — a normalised distribution per position."""
    lp = np.empty((len(rows), cl.NUM_CLASSES), dtype=np.float64)
    for i, (name, p_elem) in enumerate(rows):
        probs = np.empty(cl.NUM_CLASSES, dtype=np.float64)
        probs[cl.BACKGROUND_INDEX] = 1.0 - p_elem
        probs[cl.ELEMENT_INDICES] = _element_probs(name, p_elem)
        lp[i] = np.log(probs)
    return lp


# The hand-checked fixture. Every expected number in this file is derived from these seven
# segments by hand; nothing is read back out of the implementation.
#
#   positions    0.. 9   background   (10)
#   positions   10..39   Stem_I       (30)
#   positions   40..44   background   ( 5)   <- gap of 5 to the next element
#   positions   45..69   Specifier    (25)
#   positions   70..109  background   (40)   <- gap of 40
#   positions  110..121  Terminator   (12)
#   positions  122..139  background   (18)
#                                    ----
#                              seq_len = 140
_SPEC: tuple[tuple[str | None, int, float], ...] = (
    (None, 10, _BG_P_ELEM),
    ("Stem_I", 30, _EL_P_ELEM),
    (None, 5, _BG_P_ELEM),
    ("Specifier", 25, _EL_P_ELEM),
    (None, 40, _BG_P_ELEM),
    ("Terminator", 12, _EL_P_ELEM),
    (None, 18, _BG_P_ELEM),
)
_ROWS = _rows(_SPEC)
_LP = _log_probs(_ROWS)
_SEQ_LEN = 140

#: The three raw element runs — the numbers the whole file rests on. Each scope selects exactly
#: these over its own useful range, and the two ranges are **not** the same, which is the point
#: of the scope being a knob: global for τ in (0.05, 0.95], because an element position carries
#: ``1 − P(background) = 0.95`` whatever its class; per-class (uniform map) for τ in (0.00714,
#: 0.893), because the own-class posterior 0.94 × 0.95 round-trips to 0.8929999999999999 and so
#: never attains 0.893. Both boundaries measured, not reasoned about.
_RUNS = ((10, 40), (45, 70), (110, 122))
_TAU = 0.5
_STEM_I = CLASS_INDEX["Stem_I"]
_SPECIFIER = CLASS_INDEX["Specifier"]
_TERMINATOR = CLASS_INDEX["Terminator"]

#: A per-class map equivalent to the global τ on this fixture: an element position carries
#: 0.94 × 0.95 = 0.893 on its own class, a background position 0.05/7 ≈ 0.00714 on each.
_TAU_MAP_EQUIV = dict.fromkeys(lc.ELEMENT_CLASS_NAMES, 0.5)


def _kw(**over):
    """The rule parameters, with `over` applied — no defaults exist on `construct_loci`."""
    base = dict(
        threshold_scope="global",
        threshold=_TAU,
        min_span=1,
        gap_merge=0,
        min_distinct_elements=0,
        flank=0,
    )
    base.update(over)
    return base


def _cores(loci) -> list[tuple[int, int]]:
    return [(lo.candidate.start, lo.candidate.end) for lo in loci]


def _spans(loci) -> list[tuple[int, int]]:
    return [(lo.start, lo.end) for lo in loci]


# ══════════════════════════════════════════════════════════════════════════════════════════
# The independent reference, and the variants that must disagree with it
# ══════════════════════════════════════════════════════════════════════════════════════════


def _ref_assign(row: tuple[str | None, float], scope: str, threshold) -> int | None:
    """Reference per-position assignment, re-derived from the layout row in pure Python."""
    name, p_elem = row
    probs = _element_probs(name, p_elem)
    if scope == "global":
        if p_elem < float(threshold):
            return None
        return _ELEM[int(np.argmax(probs))]
    taus = [float(threshold[n]) for n in lc.ELEMENT_CLASS_NAMES]
    cleared = [j for j in range(_N_EL) if probs[j] >= taus[j]]
    if not cleared:
        return None
    return _ELEM[max(cleared, key=lambda j: (probs[j], -j))]


def _ref_loci(
    rows,
    *,
    scope,
    threshold,
    min_span,
    gap_merge,
    k,
    flank,
    co_occurrence_over="core",
    span_filter_on="core",
    flank_when="after",
):
    """A deliberately naive reference for the D3 rule; the keyword variants are the sabotages.

    `co_occurrence_over="span"`, `span_filter_on="flanked"` and `flank_when="before_merge"`
    each flip exactly one ordering decision the shipped rule documents, and each is shown
    below to change the answer on this fixture — which is what makes the agreement test above
    them evidence rather than decoration.
    """
    assign = [_ref_assign(r, scope, threshold) for r in rows]
    n = len(rows)

    runs, i = [], 0
    while i < n:
        if assign[i] is None:
            i += 1
            continue
        j = i
        while j < n and assign[j] is not None:
            j += 1
        runs.append((i, j))
        i = j

    if flank_when == "before_merge":
        runs = [(max(0, s - flank), min(n, e + flank)) for s, e in runs]

    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap_merge:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged]

    out = []
    for s, e in merged:
        fs, fe = (s, e) if flank_when == "before_merge" else (max(0, s - flank), min(n, e + flank))
        if (fe - fs if span_filter_on == "flanked" else e - s) < min_span:
            continue
        lo, hi = (fs, fe) if co_occurrence_over == "span" else (s, e)
        classes = {a for a in assign[lo:hi] if a is not None}
        if len(classes) < k:
            continue
        out.append({"core": (s, e), "span": (fs, fe), "classes": frozenset(classes)})
    return out


def _ref_kwargs(point, **over):
    scope, thr, min_span, gap_merge, k, flank = point
    kwargs = dict(
        scope=scope, threshold=thr, min_span=min_span, gap_merge=gap_merge, k=k, flank=flank
    )
    kwargs.update(over)
    return kwargs


# A grid broad enough that a flipped ordering decision has somewhere to show itself.
_GRID = [
    (scope, thr, min_span, gap_merge, k, flank)
    for scope, thr in (("global", _TAU), ("per_class", _TAU_MAP_EQUIV))
    for min_span in (1, 13, 25, 31)
    for gap_merge in (0, 4, 5, 40)
    for k in (0, 1, 2, 3)
    for flank in (0, 10, 25)
]


def _shipped(scope, thr, min_span, gap_merge, k, flank):
    return lc.construct_loci(
        _LP,
        **_kw(
            threshold_scope=scope,
            threshold=thr,
            min_span=min_span,
            gap_merge=gap_merge,
            min_distinct_elements=k,
            flank=flank,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════════════════
# 1. The fixture itself
# ══════════════════════════════════════════════════════════════════════════════════════════


def test_fixture_assignment_reproduces_the_requested_layout():
    """The layout the rest of the file assumes is the layout `element_assignment` sees.

    Guards the degenerate-fixture failure: a generator that quietly made every position
    identical would let every downstream test compare uniform to uniform and pass.
    """
    for scope, thr in (("global", _TAU), ("per_class", _TAU_MAP_EQUIV)):
        got = lc.element_assignment(_LP, threshold_scope=scope, threshold=thr)
        assert got.shape == (_SEQ_LEN,)
        assert np.count_nonzero(got >= 0) == 30 + 25 + 12
        assert set(got[10:40].tolist()) == {_STEM_I}
        assert set(got[45:70].tolist()) == {_SPECIFIER}
        assert set(got[110:122].tolist()) == {_TERMINATOR}
        for s, e in ((0, 10), (40, 45), (70, 110), (122, 140)):
            assert set(got[s:e].tolist()) == {lc.NOT_ELEMENT}, (scope, s, e)


def test_fixture_element_posterior_is_the_requested_p_elem():
    """`1 − P(background)` is exactly the layout's `p_elem`, so τ = 0.5 means what it says."""
    p = cl.element_posterior(_LP)
    assert np.allclose(p[10:40], _EL_P_ELEM)
    assert np.allclose(p[0:10], _BG_P_ELEM)
    # …and the three classes are genuinely distinguishable, not three names for one vector.
    probs = np.exp(_LP)
    assert probs[10, _STEM_I] == pytest.approx(_SHARE * _EL_P_ELEM)
    assert probs[45, _SPECIFIER] == pytest.approx(_SHARE * _EL_P_ELEM)
    assert probs[10, _SPECIFIER] < 0.02


# ══════════════════════════════════════════════════════════════════════════════════════════
# 2. The composition is real — AST pins on the shared merge
# ══════════════════════════════════════════════════════════════════════════════════════════

_SRC_ROOT = pathlib.Path(cl.__file__).resolve().parent.parent
_MERGE_NAMES = ("candidates_from_mask", "_merge_runs", "_true_runs")


def _defined_functions(tree: ast.AST) -> set[str]:
    """Every function name defined anywhere in the tree — `async def` included.

    P3-11's review found guards matching only `ast.FunctionDef`, so one keyword walked past
    them: a refusal a keyword evades is not a refusal.
    """
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _called_names(tree: ast.AST) -> set[str]:
    """Every name the tree *calls*, bare (`f(...)`) or qualified (`np.diff(...)`)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
            names.add(ast.unparse(node.func))
    return names


def test_exactly_one_module_in_the_tree_defines_the_merge():
    """`candidates_from_mask` / `_merge_runs` / `_true_runs` live in `infer/call.py`, nowhere else.

    ADR-0005 D3 pins one locus-construction operator. A second definition anywhere under
    `src/` means fixing one copy and shipping the bug in the other. (A second definition
    *inside* `call.py` is plain shadowing, which ruff F811 rejects as a CI-blocking error.)
    """
    owners = {name: [] for name in _MERGE_NAMES}
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        defined = _defined_functions(ast.parse(path.read_text(encoding="utf-8")))
        for name in _MERGE_NAMES:
            if name in defined:
                owners[name].append(path.name)
    for name in _MERGE_NAMES:
        assert owners[name] == ["call.py"], f"{name} is defined in {owners[name]}"


def test_locus_derives_no_run_extraction_or_merge_of_its_own():
    """`locus.py` supplies a mask and reads the result; it never re-derives the merge.

    Checks both what the module *defines* and what it *calls* — a guard that looked only at
    local `def`s would let an imported `np.diff` / `np.flatnonzero` run-extraction through,
    which is exactly the gap P3-11's second review round found in its sibling guard.
    """
    tree = ast.parse(pathlib.Path(lc.__file__).read_text(encoding="utf-8"))
    defined = _defined_functions(tree)
    called = _called_names(tree)

    assert not {n for n in defined if "merge" in n or "run" in n}, defined
    assert "candidates_from_mask" in called
    for forbidden in ("_merge_runs", "_true_runs", "diff", "flatnonzero", "np.diff"):
        assert forbidden not in called, f"locus.py calls {forbidden}"


def _return_call_source(module, func_name: str, callee: str) -> list[str]:
    """`ast.unparse` of every `callee(...)` call inside `func_name`, in source order."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == func_name
    )
    return [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee
    ]


def test_both_entry_points_call_the_shared_merge_with_the_arguments_they_claim():
    """Pinned by **contents**, not by shape.

    A guard that checks a call's *arity* is satisfied by a call wrong in exactly the way the
    guard exists to catch — `candidates_from_mask(other_mask, stale_zf, ...)` has the same
    shape. Comparing the unparsed source means a regression that reconciled the wrong mask
    would turn this red instead of leaving the tier green.
    """
    assert _return_call_source(cl, "call_candidates", "candidates_from_mask") == [
        "candidates_from_mask(log_probs, p_elem >= float(threshold), zero_flanked, "
        "min_span=min_span, gap_merge=gap_merge)"
    ]
    assert _return_call_source(lc, "construct_loci", "candidates_from_mask") == [
        "candidates_from_mask(log_probs, assignment >= 0, zero_flanked, "
        "min_span=min_span, gap_merge=gap_merge)"
    ]


# ══════════════════════════════════════════════════════════════════════════════════════════
# 3. The recall floor: at k <= 1 the rule discards nothing the caller found
# ══════════════════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("tau", [0.05, 0.5, 0.89])
@pytest.mark.parametrize("min_span", [1, 13, 25])
@pytest.mark.parametrize("gap_merge", [0, 5, 40])
def test_global_scope_at_k_le_1_loses_nothing_the_caller_found(k, tau, min_span, gap_merge):
    """D3's Stage-1 per-locus recall floor, as a property of the rule rather than a promise.

    Under `threshold_scope="global"` the mask is bit-identical to `call_candidates`'s, and
    every merged run contains at least one masked position — so every position gets an element
    class and `n_distinct_elements >= 1` always. The cores are therefore the *same spans*, not
    merely a similar count.
    """
    caller = cl.call_candidates(_LP, None, threshold=tau, min_span=min_span, gap_merge=gap_merge)
    loci = _shipped("global", tau, min_span, gap_merge, k, 0)
    assert _cores(loci) == [(c.start, c.end) for c in caller]
    assert all(lo.n_distinct_elements >= 1 for lo in loci)


def test_the_carried_candidate_is_the_callers_candidate_exactly():
    """`Locus` holds the caller's `Candidate`, not a re-derived copy — so it cannot drift."""
    caller = cl.call_candidates(_LP, None, threshold=_TAU, min_span=1, gap_merge=0)
    loci = _shipped("global", _TAU, 1, 0, 1, 25)
    assert [lo.candidate for lo in loci] == caller


def test_co_occurrence_is_monotone_nested_in_k():
    """Raising the co-occurrence count can only ever remove loci, never add or move one."""
    prev = None
    for k in range(cl.NUM_ELEMENT_CLASSES + 1):
        cores = set(_cores(_shipped("global", _TAU, 1, 5, k, 0)))
        if prev is not None:
            assert cores <= prev, k
        prev = cores


# ══════════════════════════════════════════════════════════════════════════════════════════
# 4. Core geometry — the hand-checked numbers
# ══════════════════════════════════════════════════════════════════════════════════════════


def test_the_three_raw_runs_are_the_hand_checked_spans():
    assert _cores(_shipped("global", _TAU, 1, 0, 0, 0)) == list(_RUNS)


@pytest.mark.parametrize(
    ("gap_merge", "expected"),
    [
        (0, [(10, 40), (45, 70), (110, 122)]),
        (4, [(10, 40), (45, 70), (110, 122)]),  # the gap is 5; 4 does not reach it
        (5, [(10, 70), (110, 122)]),  # …5 does, inclusively
        (39, [(10, 70), (110, 122)]),  # the second gap is 40
        (40, [(10, 122)]),
    ],
)
def test_gap_merge_bridges_at_the_gap_and_not_one_below(gap_merge, expected):
    assert _cores(_shipped("global", _TAU, 1, gap_merge, 0, 0)) == expected


@pytest.mark.parametrize(
    ("min_span", "expected"),
    [
        (12, [(10, 40), (45, 70), (110, 122)]),  # the 12-nt run survives an inclusive cut
        (13, [(10, 40), (45, 70)]),
        (26, [(10, 40)]),
        (31, []),
    ],
)
def test_min_span_is_inclusive_and_applies_to_the_core(min_span, expected):
    assert _cores(_shipped("global", _TAU, min_span, 0, 0, 0)) == expected


def test_min_span_is_applied_to_the_core_not_the_flanked_span():
    """A generous flank must not rescue a run that failed the span cut.

    The 12-nt Terminator run plus a 10-nt flank each side spans 32 nt; at `min_span = 25` it
    is still gone. Flank is context handed to Stage-2, not evidence of extent.
    """
    assert _cores(_shipped("global", _TAU, 25, 0, 0, 10)) == [(10, 40), (45, 70)]


# ══════════════════════════════════════════════════════════════════════════════════════════
# 5. Co-occurrence — a count, never an identity
# ══════════════════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (0, [(10, 70), (110, 122)]),
        (1, [(10, 70), (110, 122)]),
        (2, [(10, 70)]),  # only the Stem_I+Specifier merge carries two classes
        (3, []),
    ],
)
def test_co_occurrence_counts_distinct_classes_in_the_core(k, expected):
    """At `gap_merge = 5` the first two runs are one core carrying two classes; the third, one.

    Deliberately asymmetric — two survivors at k ≤ 1, one at k = 2, none at k = 3 — so a rule
    that returned the right *count* of the *wrong* loci cannot pass by counting alone.
    """
    loci = _shipped("global", _TAU, 1, 5, k, 0)
    assert _cores(loci) == expected
    if k <= 2 and loci:
        assert loci[0].element_classes == (_STEM_I, _SPECIFIER)
        assert loci[0].n_distinct_elements == 2


def test_bridged_gap_positions_contribute_no_class():
    """The five background positions gap-merged into the core are not an eighth element.

    They are assigned `NOT_ELEMENT` and drop out of the count, so `n_distinct_elements` is 2,
    not 3 — a rule reading `np.unique` without excluding the sentinel would say 3.
    """
    (locus,) = _shipped("global", _TAU, 1, 5, 2, 0)
    assert locus.element_classes == (_STEM_I, _SPECIFIER)
    assert lc.NOT_ELEMENT not in locus.element_classes


def _permuted(lp: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Move the mass on element slot `j` to element slot `perm[j]`; background untouched."""
    out = lp.copy()
    for j, dest in enumerate(perm):
        out[:, cl.ELEMENT_INDICES[dest]] = lp[:, cl.ELEMENT_INDICES[j]]
    return out


_PERM = (1, 2, 3, 4, 5, 6, 0)  # a 7-cycle: no element class keeps its own index


@pytest.mark.parametrize("scope", ["global", "per_class"])
def test_co_occurrence_is_class_identity_blind(scope):
    """**The §5/§13.3 anti-bias gate.**

    D3 warns that "mandating canonical elements would re-impose the §5/§13.3 bias", and the
    flagship discovery class is the non-canonical Tier-2N locus. Relabelling which element
    class is which must therefore leave the surviving loci untouched — which is a property a
    count has and a named-set predicate does not. Any rule that acquired a "must contain
    Stem I" clause turns this red.
    """
    thr = _TAU if scope == "global" else _TAU_MAP_EQUIV
    permuted_thr = (
        thr
        if scope == "global"
        else {
            lc.ELEMENT_CLASS_NAMES[dest]: thr[lc.ELEMENT_CLASS_NAMES[j]]
            for j, dest in enumerate(_PERM)
        }
    )
    for k in range(cl.NUM_ELEMENT_CLASSES + 1):
        base = lc.construct_loci(
            _LP, **_kw(threshold_scope=scope, threshold=thr, gap_merge=5, min_distinct_elements=k)
        )
        moved = lc.construct_loci(
            _permuted(_LP, _PERM),
            **_kw(
                threshold_scope=scope,
                threshold=permuted_thr,
                gap_merge=5,
                min_distinct_elements=k,
            ),
        )
        assert _cores(moved) == _cores(base), k
        assert [lo.n_distinct_elements for lo in moved] == [lo.n_distinct_elements for lo in base]
        relabel = {
            int(cl.ELEMENT_INDICES[j]): int(cl.ELEMENT_INDICES[d]) for j, d in enumerate(_PERM)
        }
        assert [lo.element_classes for lo in moved] == [
            tuple(sorted(relabel[c] for c in lo.element_classes)) for lo in base
        ]


def test_the_rule_exposes_no_way_to_name_a_required_class():
    """The co-occurrence knob is an integer count, and there is no sibling that names classes.

    Structural, not stylistic: if a `required_classes=` (or any other) parameter were added,
    the keyword-only set would no longer equal `RULE_PARAMETERS` and this fails.
    """
    params = inspect.signature(lc.construct_loci).parameters
    assert params["min_distinct_elements"].annotation == "int"
    assert {n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY} == set(
        lc.RULE_PARAMETERS
    )
    with pytest.raises(TypeError):
        lc.construct_loci(_LP, **_kw(), required_classes=("Stem_I",))
    # positive control: the identical call without the invented knob succeeds
    assert lc.construct_loci(_LP, **_kw())


# ══════════════════════════════════════════════════════════════════════════════════════════
# 6. Flank
# ══════════════════════════════════════════════════════════════════════════════════════════


def test_flank_expands_symmetrically_when_the_sequence_allows_it():
    loci = _shipped("global", _TAU, 1, 5, 1, 8)
    assert _spans(loci) == [(2, 78), (102, 130)]
    assert all(lo.flank == 8 for lo in loci)
    assert all(lo.flank_short_left == 0 and lo.flank_short_right == 0 for lo in loci)
    assert [lo.length for lo in loci] == [76, 28]


def test_flank_clips_at_both_ends_and_records_the_shortfall():
    """Core 1 starts at 10 and core 2 ends at 122 of 140, so a 20-nt flank runs off both ends.

    The shortfall is recorded rather than left to be inferred from the span: a locus truncated
    at a contig edge stays visible downstream instead of looking like a shorter request.
    """
    loci = _shipped("global", _TAU, 1, 5, 1, 20)
    assert _spans(loci) == [(0, 90), (90, 140)]
    assert (loci[0].flank_short_left, loci[0].flank_short_right) == (10, 0)
    assert (loci[1].flank_short_left, loci[1].flank_short_right) == (0, 2)
    assert not any(lo.overlaps_neighbour for lo in loci)  # abutting is not overlapping


def test_flank_overlap_is_flagged_and_the_cores_are_not_re_merged():
    """Flanking runs after gap-merging, so padded spans can intersect; that is reported.

    Re-merging here would silently undo the gap-merge decision — two loci 40 nt apart would
    become one because of context, not evidence. They stay separate and both carry the flag.
    """
    loci = _shipped("global", _TAU, 1, 5, 1, 25)
    assert _cores(loci) == [(10, 70), (110, 122)]
    assert _spans(loci) == [(0, 95), (85, 140)]
    assert all(lo.overlaps_neighbour for lo in loci)


@pytest.mark.parametrize("flank", [0, 1, 10, 25, 500])
def test_flank_never_changes_which_loci_survive(flank):
    """Co-occurrence is read over the core, so widening the flank cannot admit or drop a locus.

    At `flank = 10` the first core's padded span reaches position 49 — inside the Specifier
    run — so a rule counting classes over the *span* would flip this at k = 2. It does not.
    """
    for k in range(cl.NUM_ELEMENT_CLASSES + 1):
        assert _cores(_shipped("global", _TAU, 1, 0, k, flank)) == _cores(
            _shipped("global", _TAU, 1, 0, k, 0)
        ), (flank, k)


def test_a_flank_wider_than_the_sequence_saturates_at_the_ends():
    loci = _shipped("global", _TAU, 1, 40, 1, 500)
    assert _spans(loci) == [(0, _SEQ_LEN)]
    assert (loci[0].flank_short_left, loci[0].flank_short_right) == (490, 482)


# ══════════════════════════════════════════════════════════════════════════════════════════
# 7. Threshold scope
# ══════════════════════════════════════════════════════════════════════════════════════════


def test_per_class_scope_reproduces_global_when_the_thresholds_are_equivalent():
    """A per-class map at 0.5 selects the same positions as the global τ on this fixture.

    Not a tautology — the two take different routes (`1 − P(background)` versus each `P(c)`)
    and this fixture is built so both land on the same mask, which is the control the next
    test's divergence is read against.
    """
    assert _cores(_shipped("per_class", _TAU_MAP_EQUIV, 1, 5, 1, 0)) == _cores(
        _shipped("global", _TAU, 1, 5, 1, 0)
    )


def test_per_class_scope_expresses_a_mask_no_global_threshold_can():
    """Raising one class's τ removes that class's positions and leaves every other alone.

    The global scope cannot do this at **any** τ, and the reason is that it never sees a class:
    every element position carries ``1 − P(background) = 0.95`` whatever class it belongs to,
    so τ ≤ 0.95 keeps the Specifier run along with the other two and τ > 0.95 loses all three
    together. Both sides of that boundary are swept below, so "cannot" is asserted rather than
    argued. That is what makes the scope a real D3 knob rather than a re-parameterisation — and
    why the choice is frozen at the phase gate.
    """
    suppress = dict(_TAU_MAP_EQUIV) | {"Specifier": 0.95}
    assert _cores(_shipped("per_class", suppress, 1, 5, 1, 0)) == [(10, 40), (110, 122)]
    for tau in (0.05, 0.5, 0.89, 0.893, 0.9, 0.95, 0.96, 1.0):
        assert _cores(_shipped("global", tau, 1, 5, 1, 0)) != [(10, 40), (110, 122)]


def test_per_class_assignment_names_only_a_class_that_cleared_its_own_threshold():
    """With every τ_c at 1.0 bar one, only that class can ever be assigned."""
    only_terminator = dict.fromkeys(lc.ELEMENT_CLASS_NAMES, 1.0) | {"Terminator": 0.5}
    got = lc.element_assignment(_LP, threshold_scope="per_class", threshold=only_terminator)
    assert set(got[got >= 0].tolist()) == {_TERMINATOR}
    assert _cores(_shipped("per_class", only_terminator, 1, 5, 1, 0)) == [(110, 122)]


def test_per_class_assignment_names_the_cleared_class_not_the_strongest_one():
    """The case the test above cannot see, and the reason it needed a partner.

    Masking the arg-max to the classes that cleared is only observable where the **strongest**
    class is one that did *not* clear while a **weaker** one did. Every threshold set used
    elsewhere in this file makes those the same class, so an unmasked arg-max would agree with
    the correct answer everywhere and the guard would be decoration.

    Here, on a Stem_I position (P = 0.893 against 0.0095 for every other class): τ_Stem_I is
    0.95, which Stem_I fails, while τ_Specifier is 0.005, which Specifier clears. The position
    is element-like — something cleared — and it must be named **Specifier**. An arg-max taken
    over the unmasked posteriors would name Stem_I: a class the operator just refused.
    """
    thr = dict.fromkeys(lc.ELEMENT_CLASS_NAMES, 1.0) | {"Stem_I": 0.95, "Specifier": 0.005}
    got = lc.element_assignment(_LP, threshold_scope="per_class", threshold=thr)
    assert set(got[10:40].tolist()) == {_SPECIFIER}, "the strongest class did not clear its τ"
    assert set(got[45:70].tolist()) == {_SPECIFIER}  # Specifier clears on its own run too
    assert set(got[110:122].tolist()) == {_SPECIFIER}
    # …and the background positions (0.00714 each) sit below τ_Specifier = 0.005? No — they
    # clear it, which is the point: this threshold set is deliberately liberal, so what is
    # being read here is *which* class is named, not how many positions were selected.
    assert np.count_nonzero(got >= 0) == _SEQ_LEN


def test_dominant_class_is_the_element_that_drives_the_call():
    """Descriptive, and it must at least be one of the classes actually present in the core."""
    for locus in _shipped("global", _TAU, 1, 5, 1, 0):
        assert locus.candidate.dominant_class in locus.element_classes
    # the merged core is 30 nt of Stem_I against 25 of Specifier — no tie, and Stem_I wins
    assert _shipped("global", _TAU, 1, 5, 2, 0)[0].candidate.dominant_class == _STEM_I


# ══════════════════════════════════════════════════════════════════════════════════════════
# 8. Contig-end and coverage flags — kept and flagged, never dropped
# ══════════════════════════════════════════════════════════════════════════════════════════


def test_zero_flanked_is_counted_over_the_span_and_the_core_separately():
    """Two intervals, two differently-named fields, and the locus is kept either way."""
    zf = np.zeros(_SEQ_LEN, dtype=bool)
    zf[:20] = True  # the first 20 nt were scored with synthetic pad
    (first, second) = lc.construct_loci(
        _LP, zf, **_kw(gap_merge=5, min_distinct_elements=1, flank=8)
    )
    assert first.candidate.n_zero_flanked == 10  # core starts at 10; 10..19 are padded
    assert first.n_zero_flanked_span == 18  # span starts at 2; 2..19
    assert second.n_zero_flanked_span == 0
    assert len(_shipped("global", _TAU, 1, 5, 1, 8)) == 2  # nothing was dropped for it


def test_coverage_populates_single_covered_and_its_absence_is_none_not_zero():
    """`None` means *not measured*; a real zero is a different statement and reads differently.

    P3-11 left this measured-but-invisible: `Candidate` has no coverage field, so a locus
    called from one window's evidence looked identical downstream to one called from two.
    `Locus` is a new record, so carrying it costs no P2 schema change.
    """
    cov = np.full(_SEQ_LEN, 2, dtype=np.int32)
    cov[:64] = 1  # a tail-anchored contig start, singly covered
    (first, second) = lc.construct_loci(
        _LP, None, cov, **_kw(gap_merge=5, min_distinct_elements=1, flank=8)
    )
    assert first.n_single_covered_span == 62  # span (2, 78) ∩ [0, 64)
    assert second.n_single_covered_span == 0
    assert all(lo.n_single_covered_span is None for lo in _shipped("global", _TAU, 1, 5, 1, 8))


# ══════════════════════════════════════════════════════════════════════════════════════════
# 9. Refusals — every one paired with the identical clean call succeeding
# ══════════════════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"threshold_scope": "per-class"}, "threshold_scope must be one of"),
        ({"threshold_scope": "global", "threshold": _TAU_MAP_EQUIV}, "mapping is the per_class"),
        ({"threshold_scope": "per_class", "threshold": 0.5}, "takes a mapping"),
        (
            {"threshold_scope": "per_class", "threshold": {"Stem_I": 0.5}},
            "missing=",
        ),
        (
            {
                "threshold_scope": "per_class",
                "threshold": _TAU_MAP_EQUIV | {"background": 0.5},
            },
            "unknown=",
        ),
        (
            {"threshold_scope": "per_class", "threshold": _TAU_MAP_EQUIV | {"Stem_I": 1.5}},
            "must be in .0, 1.",
        ),
        ({"threshold": 1.5}, "threshold must be in"),
        ({"threshold": float("nan")}, "threshold must be in"),
        ({"min_distinct_elements": -1}, "min_distinct_elements must be in"),
        ({"min_distinct_elements": cl.NUM_ELEMENT_CLASSES + 1}, "min_distinct_elements must be in"),
        ({"flank": -1}, "flank must be >= 0"),
    ],
)
def test_refuses_a_malformed_rule_parameter(over, match):
    with pytest.raises(lc.LocusError, match=match):
        lc.construct_loci(_LP, **_kw(**over))


def test_the_clean_call_the_refusals_perturb_actually_succeeds():
    """The positive control for every case above.

    `pytest.raises` asserts only *that* it raised — a guard rejecting everything satisfies all
    eleven cases at once. This is the half that distinguishes a correct refusal from a broken
    one.
    """
    assert len(lc.construct_loci(_LP, **_kw())) == 3
    assert len(lc.construct_loci(_LP, **_kw(threshold_scope="per_class", threshold=_TAU_MAP_EQUIV)))


@pytest.mark.parametrize(
    ("coverage", "match"),
    [
        (np.full(_SEQ_LEN, 2.0), "must be an integer array"),
        (np.full(_SEQ_LEN, True), "must be an integer array"),
        (np.full(_SEQ_LEN - 1, 2, dtype=np.int32), "must be .seq_len,."),
        (np.zeros(_SEQ_LEN, dtype=np.int32), "covered by no window"),
    ],
)
def test_refuses_a_malformed_coverage_array(coverage, match):
    with pytest.raises(lc.LocusError, match=match):
        lc.construct_loci(_LP, None, coverage, **_kw())
    # positive control: the same call with a well-formed coverage succeeds
    assert lc.construct_loci(_LP, None, np.full(_SEQ_LEN, 2, dtype=np.int32), **_kw())


def test_the_shared_merge_refuses_a_float_mask_rather_than_coercing_it():
    """Handing the merge the *scores* instead of the *decision* is a wrong answer that runs clean.

    `np.asarray(p_elem, dtype=bool)` maps every non-zero posterior to True — on this fixture
    that is all 140 positions, one candidate spanning the whole sequence, no error anywhere.
    """
    p_elem = cl.element_posterior(_LP)
    with pytest.raises(cl.CandidateError, match="must be a boolean array"):
        cl.candidates_from_mask(_LP, p_elem, None, min_span=1, gap_merge=0)
    # positive control, and the demonstration of what the coercion would have produced
    assert len(cl.candidates_from_mask(_LP, p_elem >= _TAU, None, min_span=1, gap_merge=0)) == 3
    assert (
        len(
            cl.candidates_from_mask(
                _LP, np.asarray(p_elem, dtype=bool), None, min_span=1, gap_merge=0
            )
        )
        == 1
    )


def test_a_malformed_log_probs_is_refused_by_the_shared_validator():
    """`locus` inherits `call`'s input contract rather than re-checking it — and it fires."""
    bad = _LP.copy()
    bad[3, 0] = np.nan
    with pytest.raises(cl.CandidateError, match="non-finite"):
        lc.construct_loci(bad, **_kw())
    with pytest.raises(cl.CandidateError, match="min_span must be"):
        lc.construct_loci(_LP, **_kw(min_span=0))
    assert lc.construct_loci(_LP, **_kw())  # positive control


# ══════════════════════════════════════════════════════════════════════════════════════════
# 10. Pin discipline — no rule value has a default
# ══════════════════════════════════════════════════════════════════════════════════════════


def test_no_rule_parameter_has_a_default():
    """ADR-0005 D3 freezes the locus values at the §13.1 phase gate; a default pre-empts that."""
    assert lc.no_rule_parameter_has_a_default()
    for name in lc.RULE_PARAMETERS:
        with pytest.raises(TypeError):
            kwargs = _kw()
            kwargs.pop(name)
            lc.construct_loci(_LP, **kwargs)


def test_the_default_check_bites_on_a_signature_that_has_one():
    """The positive control for the check itself.

    A predicate that returned True unconditionally would satisfy the test above. Applied to a
    stub carrying the same six knobs with one defaulted, it must return False.
    """

    def _stub(
        log_probs,
        *,
        threshold_scope,
        threshold,
        min_span,
        gap_merge,
        min_distinct_elements,
        flank=50,
    ):
        return None

    def _stub_clean(
        log_probs, *, threshold_scope, threshold, min_span, gap_merge, min_distinct_elements, flank
    ):
        return None

    def _stub_extra_knob(
        log_probs,
        *,
        threshold_scope,
        threshold,
        min_span,
        gap_merge,
        min_distinct_elements,
        flank,
        required_classes,
    ):
        return None

    assert not lc.no_rule_parameter_has_a_default(_stub)
    assert lc.no_rule_parameter_has_a_default(_stub_clean)
    # The set comparison, not the default scan, is what catches a knob that was *added* — an
    # unlisted `required_classes` has no default to find, so a subset check would wave it
    # through and the D3 knob inventory would silently stop being the inventory.
    assert not lc.no_rule_parameter_has_a_default(_stub_extra_knob)


def test_the_module_pins_no_locus_value():
    """No module-level constant here is a threshold, span, gap or flank a caller could take."""
    numeric = {
        name: value
        for name, value in vars(lc).items()
        if not name.startswith("__")
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    }
    # Both survivors are structural — a sentinel and a class count derived from `CLASS_ORDER`.
    # Neither is a threshold, span, gap or flank, and there is nothing else for a later reader
    # to mistake for a frozen D3 value.
    assert numeric == {"NOT_ELEMENT": -1, "NUM_ELEMENT_CLASSES": cl.NUM_ELEMENT_CLASSES}, numeric


# ══════════════════════════════════════════════════════════════════════════════════════════
# 11. The reference, and the sabotages that must bite
# ══════════════════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("point", _GRID)
def test_the_shipped_rule_agrees_with_the_independent_reference(point):
    ref = _ref_loci(_ROWS, **_ref_kwargs(point))
    got = _shipped(*point)
    assert [r["core"] for r in ref] == _cores(got)
    assert [r["span"] for r in ref] == _spans(got)
    assert [sorted(r["classes"]) for r in ref] == [list(lo.element_classes) for lo in got]


@pytest.mark.parametrize(
    "flip",
    [
        {"co_occurrence_over": "span"},
        {"span_filter_on": "flanked"},
        {"flank_when": "before_merge"},
    ],
)
def test_each_reference_variant_disagrees_somewhere_on_the_grid(flip):
    """Each flipped ordering decision must actually change the answer.

    Without this, the agreement test above would be evidence only that two implementations of
    the *same* ordering agree — true of a reference that never exercised the decision at all.
    """
    disagreements = [
        point
        for point in _GRID
        if [r["core"] for r in _ref_loci(_ROWS, **_ref_kwargs(point, **flip))]
        != _cores(_shipped(*point))
        or [r["span"] for r in _ref_loci(_ROWS, **_ref_kwargs(point, **flip))]
        != _spans(_shipped(*point))
    ]
    assert disagreements, flip
