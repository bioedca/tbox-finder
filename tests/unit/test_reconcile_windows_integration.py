"""P3-11 — the frozen D3+A3 reconciliation operator *in the P3 integration path*.

`tests/unit/test_reconcile.py` (P2-03) gates the operator in isolation and
`tests/golden/test_reconcile_golden.py` pins its arithmetic to a digest. Neither asks the
question this step exists to answer, which is a question about the **path**, not the
reduction:

    does a locus that happens to straddle a 512-nt window boundary get the *same call*
    as one that does not?

That is D3's stated purpose in the PRD's own words — *"a seam-free operator … so the
Stage-1 per-locus recall floor holds under overlapping-tile scanning and boundary IoU
(GATE-4) is not a 512-grid artifact"* (PRD §6; ADR-0005 D3, disambiguated by A3). It is a
property of `tile_windows` → `encode_scan_window` → `reconcile_windows` →
`call_candidates` composed, and no test in the repo composed them.

What is real here and what is stubbed
------------------------------------
Real and shipped: the tiling (`scan.scan_window_ids`, which is `tile_windows` +
`encode_scan_window`), the reduction (`reconcile.reconcile_windows`), and the along-sequence
caller (`call.call_candidates`). Stubbed: the model forward, and only the model forward.

The forward is stubbed because it is the one link that needs torch, which the bare CI tier
does not have — `tests/ml/test_scan_checkpoint.py` covers `scan_sequence` end-to-end through
a real `nn.Module` instead. That leaves a seam of exactly one line between what this file
executes and what production executes, so
:func:`test_the_integration_path_hands_its_logits_to_the_P2_operator` pins that line by
AST: `scan_encoded_windows` must *end* in
``reconcile_windows(logits, np.asarray(starts), seq_len)``. Without that pin this tier would
be testing a re-typed equivalent of the shipped path rather than the shipped path.

The stub is a **content oracle**, not a model
---------------------------------------------
It reads a window's *token ids* — so the real encoder is inside the loop, pad convention
included — and maps a base to a class: ``C`` → `Stem_I`, ``G`` → `Specifier`, ``T`` →
`Antiterminator_Tbox_seq`, ``A``/``N`` → `background`. The background is therefore poly-``A``:
any random ACGT carrier would light the oracle up everywhere, and the subject under test is
the window *geometry*, not the sequence. :func:`test_the_oracle_reads_token_ids_not_positions`
keeps that from becoming a fiction by mutating the carrier and requiring the call to move.

Its one interesting property is **context-quality degradation**: a position near a window's
edge is scored from truncated context, so the oracle attenuates its element logit there.
That is the mechanism D3 exists to defeat, declared openly rather than assumed. Two forms
are swept — a hard truncation (``step``) and a linear ramp (``ramp``) — so no conclusion
here rests on the shape of the degradation.

Why the geometry makes seam-freeness work at all (and where it stops)
---------------------------------------------------------------------
At window 1024 / stride 512 an interior position ``p`` is covered by exactly two windows,
at in-window offsets ``o`` and ``o + 512`` for ``o = p mod 512``. Their distances to the
nearer window edge are ``o`` and ``511 - o`` — **a constant sum of 511**. One covering
window is close to an edge exactly when the other is not, which is why averaging the two
posteriors recovers the confident call at every phase. It is also why the guarantee is
*interior-only*: at the pinned tail-anchored tiling the first and last 512 nt of a contig
are covered **once**, there is no second window to compensate, and
:func:`test_the_seam_free_guarantee_is_interior_only_and_the_terminus_carries_no_flag`
measures that limit rather than letting it be discovered downstream.

Nothing in this file pins a value
---------------------------------
``_TAU`` / ``_MIN_SPAN`` / ``_GAP_MERGE`` are test-local. ADR-0005 D3 freezes the production
Stage-1 threshold and locus geometry **at the phase gate** (§13.1), and `infer/call.py`
deliberately gives those three parameters no defaults. The invariance asserted below is
shown to hold across a measured band of thresholds precisely so that it cannot be read as a
property of one lucky choice.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tbox_finder.data.window_dataset import BASE_TO_ID, PAD_TOKEN_ID, STRIDE_NT, WINDOW_NT
from tbox_finder.infer import call as call_mod
from tbox_finder.infer import reconcile as reconcile_mod
from tbox_finder.infer import scan as scan_mod
from tbox_finder.infer.call import Candidate, call_candidates, element_posterior
from tbox_finder.infer.reconcile import NUM_CLASSES, reconcile_windows
from tbox_finder.infer.scan import scan_window_ids
from tbox_finder.labels import CLASS_INDEX

# ═════════════════════════════════════════════════════════════════════════════
# The fixture: a content oracle over real encoded windows
# ═════════════════════════════════════════════════════════════════════════════

_BG = CLASS_INDEX["background"]

#: Base → class. The carrier is poly-``A``; the three signal bases name three real element
#: classes so the run has genuine multi-class structure and ``dominant_class`` means something.
_BASE_CLASS = {
    "A": _BG,
    "N": _BG,
    "C": CLASS_INDEX["Stem_I"],
    "G": CLASS_INDEX["Specifier"],
    "T": CLASS_INDEX["Antiterminator_Tbox_seq"],
}

#: Token-id → class, built through the *shipped* base→id table so a change to the encoder's
#: alphabet cannot leave this oracle quietly mapping the wrong ids. ``-1`` marks "not a base"
#: (i.e. ``PAD_TOKEN_ID``), which the oracle must never score as DNA.
_TOKEN_CLASS = np.full(max([PAD_TOKEN_ID, *BASE_TO_ID.values()]) + 1, -1, dtype=np.int64)
for _base, _cls in _BASE_CLASS.items():
    _TOKEN_CLASS[BASE_TO_ID[_base]] = _cls
assert set(_BASE_CLASS) == set(BASE_TO_ID), "the oracle must name every base the encoder emits"
assert _TOKEN_CLASS[PAD_TOKEN_ID] < 0, "the pad id must not map to a class"

_BACKGROUND_LOGIT = 4.0

#: Must exceed ``2 × _BACKGROUND_LOGIT``. Under the ``ramp`` oracle the two windows covering a
#: position sit at qualities summing to ~1, so the worst case is ``q1 = q2 = 0.5``; there the
#: reconciled element mass is ``exp(_ELEMENT_LOGIT / 2)`` against a background ``exp(
#: _BACKGROUND_LOGIT)``. At exactly ``2 ×`` those tie, ``np.argmax`` resolves the tie to
#: `background` (index 0), and the arg-max half of D3 would look phase-dependent for a reason
#: belonging to the fixture rather than to the operator. Measured at ``8.0`` — the tie is real
#: and lands exactly at the ramp's midpoint offset — so the constant is set clear of it.
_ELEMENT_LOGIT = 12.0

#: Width of the truncated-context band at each window edge, for the ``step`` oracle. Any
#: value ``< 256`` keeps the "at most one covering window is degraded" property (the two
#: edge distances sum to 511), which is the geometry the operator exploits.
_EDGE_NT = 64

#: The planted locus: 100 nt Stem_I + 30 nt Specifier + 30 nt Antiterminator. The Stem_I run
#: is long enough that ``dominant_class`` survives a fully degraded band anywhere in the
#: locus — otherwise a phase-dependent flip would be a property of the *fixture*, not a
#: finding about the operator.
_MOTIF = "C" * 100 + "G" * 30 + "T" * 30
_MOTIF_LEN = len(_MOTIF)
_MOTIF_CLASS = CLASS_INDEX["Stem_I"]

#: Test-local caller parameters. These pin NOTHING — see the module docstring.
_TAU = 0.4
_MIN_SPAN = 40
_GAP_MERGE = 5

#: A sequence long enough for a full 512-nt phase period to sit inside the coverage-2
#: interior: ``tile_windows(3072) == [0, 512, 1024, 1536, 2048]``, interior = ``[512, 2560)``.
_SEQ_LEN = 3072

#: The 5′ anchor of the phase sweep. ``_ANCHOR + 511 + _MOTIF_LEN < 2560``, so the locus stays
#: interior at every phase.
_ANCHOR = 1024

#: One full 512-nt period, sampled at both edges, both quarters, and the centre.
_PHASES = (0, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 480, 511)


def _quality_step(window: int) -> np.ndarray:
    """Hard truncation: a position within ``_EDGE_NT`` of a window edge is context-blind."""
    d = np.minimum(np.arange(window), window - 1 - np.arange(window))
    return (d >= _EDGE_NT).astype(np.float64)


def _quality_ramp(window: int) -> np.ndarray:
    """Graded degradation: confidence falls off linearly toward each window edge."""
    d = np.minimum(np.arange(window), window - 1 - np.arange(window))
    return d / ((window - 1) / 2.0)


_QUALITY = {"step": _quality_step, "ramp": _quality_ramp}


#: What the oracle writes at a pad offset. ``zero`` is the neutral default; the other two are
#: probes for :func:`test_pad_logits_never_reach_the_reduction_across_the_encode_hop`, and they
#: catch *different* leaks — see that test.
_PAD_FILL = {
    "zero": 0.0,
    "nan": np.nan,
    "loud": 60.0,
}


def _oracle_logits(window_ids: np.ndarray, quality: str, *, pads: str = "zero") -> np.ndarray:
    """``(n_windows, window)`` token ids → ``(n_windows, window, 8)`` logits.

    A position's element logit is ``_ELEMENT_LOGIT * quality(offset)``; background carries a
    constant ``_BACKGROUND_LOGIT``. Pad positions are filled per ``pads`` and carry **no**
    background logit — a pad describes no DNA, and the reduction must never see it whatever
    it holds.
    """
    n_windows, window = window_ids.shape
    q = _QUALITY[quality](window)

    cls = _TOKEN_CLASS[np.asarray(window_ids, dtype=np.int64)]
    logits = np.zeros((n_windows, window, NUM_CLASSES), dtype=np.float64)
    logits[..., _BG] = _BACKGROUND_LOGIT

    # Select "is an element class" explicitly rather than as `cls > _BG`: the latter is only
    # equivalent while `background` holds the LOWEST class index. Were `CLASS_ORDER` reordered,
    # every class below it would silently stop receiving `_ELEMENT_LOGIT` and be scored as
    # background — the whole file would then measure a weaker fixture without failing.
    rows, cols = np.nonzero((cls >= 0) & (cls != _BG))
    logits[rows, cols, cls[rows, cols]] = _ELEMENT_LOGIT * q[cols]

    logits[cls < 0] = _PAD_FILL[pads]
    return logits


def _sequence(seq_len: int, motif_start: int) -> str:
    """A poly-``A`` carrier with ``_MOTIF`` written in at ``motif_start``."""
    bases = ["A"] * seq_len
    bases[motif_start : motif_start + _MOTIF_LEN] = list(_MOTIF)
    return "".join(bases)


def _reconciled(seq: str, quality: str, *, pads: str = "zero"):
    """The shipped path, model forward excepted: tile → encode → oracle → reconcile."""
    window_ids, starts = scan_window_ids(seq)
    logits = _oracle_logits(window_ids, quality, pads=pads)
    return reconcile_windows(logits, np.asarray(starts), len(seq))


def _call(reconciled, *, threshold: float = _TAU) -> list[Candidate]:
    return call_candidates(
        reconciled.log_probs,
        reconciled.zero_flanked,
        threshold=threshold,
        min_span=_MIN_SPAN,
        gap_merge=_GAP_MERGE,
    )


def _relative(cands: list[Candidate], motif_start: int) -> tuple[tuple[int, int, int], ...]:
    """Calls as ``(start, end, dominant_class)`` **relative to the planted locus**.

    Relative coordinates are what make the phase sweep an equality assertion: the locus moves
    with the phase, so absolute spans differ trivially while ``(0, _MOTIF_LEN, _MOTIF_CLASS)``
    is the statement "the call recovered exactly the locus, whatever the grid did".
    """
    return tuple((c.start - motif_start, c.end - motif_start, c.dominant_class) for c in cands)


# ═════════════════════════════════════════════════════════════════════════════
# The matched control: the naive per-tile stitch (the 512-grid artifact itself)
# ═════════════════════════════════════════════════════════════════════════════
def _stitched(seq: str, quality: str) -> np.ndarray:
    """The reduction D3 rejects: score every position from the ONE window whose tile owns it.

    This is the classic overlapping-scan stitch — tile ``[512k, 512(k+1))`` is read off the
    window starting at ``512k`` — and it is the *matched* control, not a straw man: same
    sequence, same encoder, same oracle, same caller parameters. The only difference is that
    it picks one covering window instead of averaging them, so a position lands in the **left
    half** of its window and the first ``_EDGE_NT`` nt of every 512-nt tile are scored from
    truncated context. A locus whose 5′ end falls on a 512 boundary loses exactly that band —
    the artifact.

    **The tail is the exception**, and it is not the left half. ``tile_windows`` is
    tail-anchored, so on this file's 3,072-nt sequence there is no window starting at 2560:
    positions 2560–3071 fall back to owner 2048 at in-window offsets 512–1023, i.e. the
    **right** half, where the truncated band sits at the 3′ end instead. Measured — 2,560
    left-half positions and exactly the last 512 right-half. The phase sweep lives in
    ``[1024, 1696)``, entirely left-half, so no assertion here depends on the tail; the case
    is stated only so a later reader does not read "left half" as an invariant of the tiling.

    A control has to be able to produce the right answer, or "it disagrees" says nothing;
    :func:`test_the_stitch_control_agrees_wherever_the_grid_happens_to_be_kind` is that leg.
    """
    window_ids, starts = scan_window_ids(seq)
    logits = _oracle_logits(window_ids, quality)
    start_list = list(starts)
    out = np.empty((len(seq), NUM_CLASSES), dtype=np.float64)
    for p in range(len(seq)):
        owner = STRIDE_NT * (p // STRIDE_NT)
        if owner not in start_list:
            owner = max(s for s in start_list if s <= p)
        row = logits[start_list.index(owner), p - owner]
        m = row.max()
        out[p] = row - (m + np.log(np.exp(row - m).sum()))
    return out


def _stitch_call(seq: str, quality: str, *, threshold: float = _TAU) -> list[Candidate]:
    return call_candidates(
        _stitched(seq, quality),
        None,
        threshold=threshold,
        min_span=_MIN_SPAN,
        gap_merge=_GAP_MERGE,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. The wiring — the operator is IMPORTED from P2, never rebuilt
# ═════════════════════════════════════════════════════════════════════════════
_SRC_ROOT = Path(reconcile_mod.__file__).resolve().parents[1]

#: Both node types, always. A guard that matches only ``ast.FunctionDef`` is evaded by a
#: single keyword: an ``async def reconcile_windows`` would be an unreported fork, and an
#: ``async def scan_encoded_windows`` would report as *missing* rather than as unwired. A
#: refusal that one keyword walks past is not a refusal. CodeRabbit, r1.
_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _module_ast(module) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _function_def(module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(_module_ast(module)):
        if isinstance(node, _FUNC_NODES) and node.name == name:
            return node
    raise AssertionError(f"{module.__name__} defines no function {name!r}")


def test_the_integration_path_hands_its_logits_to_the_P2_operator():
    """`scan_encoded_windows` must END in the P2 operator — the one line this tier stubs past.

    Pinned by AST rather than by calling it, because calling it needs torch. The assertion is
    deliberately narrow and structural: the last statement returns ``reconcile_windows(...)``
    with three positional arguments whose last is ``seq_len``. That is what makes the
    hand-assembled path in this file the shipped arithmetic instead of a plausible copy of it
    — the failure mode that shipped a dead SLURM job once already.
    """
    fn = _function_def(scan_mod, "scan_encoded_windows")
    last = fn.body[-1]
    assert isinstance(last, ast.Return), (
        "scan_encoded_windows must end by returning the reconciliation; a trailing statement "
        "after it would be arithmetic this tier never sees"
    )
    assert isinstance(last.value, ast.Call)
    assert isinstance(last.value.func, ast.Name)
    assert last.value.func.id == "reconcile_windows"
    assert len(last.value.args) == 3 and not last.value.keywords
    assert isinstance(last.value.args[2], ast.Name)
    assert last.value.args[2].id == "seq_len", (
        "the third argument is the sequence length that decides which positions are pad; "
        "handing the operator a window length here is how a pad reaches the reduction"
    )


def test_the_operator_is_imported_from_the_P2_module_not_redefined_in_the_scanner():
    """imp.md P3-11: *operator imported from P2, not rebuilt*."""
    imports = [
        node
        for node in ast.walk(_module_ast(scan_mod))
        if isinstance(node, ast.ImportFrom)
        and node.module == "tbox_finder.infer.reconcile"
        and any(a.name == "reconcile_windows" for a in node.names)
    ]
    assert len(imports) == 1, "the scanner must import the operator exactly once, from P2"
    assert scan_mod.reconcile_windows is reconcile_mod.reconcile_windows


def test_exactly_one_module_in_the_tree_defines_the_operator():
    """A second `reconcile_windows` anywhere under `src/` is a fork of a frozen operator.

    ADR-0005 A3 freezes the reduction *in code* with no config override; a fork would let one
    copy be fixed while the other keeps shipping — and `calib/temperature.py` legitimately
    needs to re-reconcile after applying T, which is exactly the pressure that produces one.
    It calls the P2 function; this test is what keeps that true.

    Scope: this collects the *modules* that define the name, so it owns the **cross-module**
    fork. A second definition inside `infer/reconcile.py` itself is a different failure —
    plain shadowing — and `ruff` F811 rejects it as a CI-blocking error (verified by
    execution, not assumed), so it is covered rather than missed.
    """
    definers = sorted(
        path.relative_to(_SRC_ROOT).as_posix()
        for path in _SRC_ROOT.rglob("*.py")
        if any(
            isinstance(node, _FUNC_NODES) and node.name == "reconcile_windows"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    )
    assert definers == ["infer/reconcile.py"]


def test_the_caller_consumes_a_reconciled_distribution_and_derives_none_of_its_own():
    """`call_candidates` must merge a distribution it was *given* (D3: reconcile, then merge).

    D3 orders the two operations — reconciliation *"before along-sequence element merging"*.
    The ordering is structural here (the caller takes ``log_probs`` and has no window axis to
    reduce), and it stays structural only while the caller owns no reduction of its own.
    """
    defined = {
        node.name for node in ast.walk(_module_ast(call_mod)) if isinstance(node, _FUNC_NODES)
    }
    assert not defined & {"logsumexp", "log_softmax", "softmax", "reconcile_windows"}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Interior seam-freeness — the claim P3-11 exists to gate
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("quality", sorted(_QUALITY))
@pytest.mark.parametrize("phase", _PHASES)
def test_a_locus_straddling_a_512_boundary_calls_identically_to_one_inside_a_window(
    quality: str, phase: int
):
    """The whole point: the reconciled call does not know where the 512 grid fell.

    The same locus is planted at every phase of one full 512-nt period — phase 0 puts its 5′
    end exactly on a window boundary, the worst case — and every phase must recover exactly
    ``[0, 160)`` relative to it, with the same dominant class. Measured: 13 phases × 2
    degradation models, one outcome.
    """
    motif_start = _ANCHOR + phase
    seq = _sequence(_SEQ_LEN, motif_start)
    calls = _relative(_call(_reconciled(seq, quality)), motif_start)
    assert calls == ((0, _MOTIF_LEN, _MOTIF_CLASS),)


@pytest.mark.parametrize("quality", sorted(_QUALITY))
def test_the_reconciled_call_is_the_same_object_at_every_phase(quality: str):
    """The sweep collapsed to a set — one outcome, not thirteen that happen to be checked."""
    outcomes = set()
    for phase in _PHASES:
        motif_start = _ANCHOR + phase
        outcomes.add(
            _relative(_call(_reconciled(_sequence(_SEQ_LEN, motif_start), quality)), motif_start)
        )
    assert outcomes == {((0, _MOTIF_LEN, _MOTIF_CLASS),)}


#: Context either side of the locus that the arg-max pattern is compared over, so the test
#: sees the element→background transitions and not just the locus interior.
_PATTERN_FLANK = 64


def _expected_prediction_pattern() -> tuple[int, ...]:
    """The per-position arg-max the locus should produce, derived from ``_MOTIF`` itself."""
    return (
        (_BG,) * _PATTERN_FLANK
        + tuple(_BASE_CLASS[base] for base in _MOTIF)
        + (_BG,) * _PATTERN_FLANK
    )


@pytest.mark.parametrize("quality", sorted(_QUALITY))
def test_the_reconciled_argmax_prediction_is_phase_invariant_too(quality: str):
    """D3's *other* half — and the one GATE-4 reads.

    ADR-0005 D3 pins the operator as *"averaged in log-sum-exp space then arg-maxed into one
    per-position prediction"*. The candidate caller does not use that arg-max: it thresholds
    ``1 − P(background)`` (D3's separate Stage-1-threshold rule), so every assertion above
    leaves ``Reconciled.prediction`` untouched. But `eval/gate4.py` grades per-nucleotide F1
    and boundary IoU **on the arg-max**, and PRD §6 names boundary IoU as the quantity that
    must not be a 512-grid artifact — so it needs its own phase sweep.

    Compared as the exact per-position class vector over the locus plus ``_PATTERN_FLANK`` nt
    of carrier either side, so the element→background transitions are inside the comparison.
    """
    patterns = set()
    for phase in _PHASES:
        motif_start = _ANCHOR + phase
        prediction = _reconciled(_sequence(_SEQ_LEN, motif_start), quality).prediction
        patterns.add(
            tuple(
                int(c)
                for c in prediction[
                    motif_start - _PATTERN_FLANK : motif_start + _MOTIF_LEN + _PATTERN_FLANK
                ]
            )
        )
    assert patterns == {_expected_prediction_pattern()}


def test_the_argmax_phase_sweep_has_power_because_the_naive_stitch_fails_it():
    """The matched control for the arg-max half: without reconciliation the boundary moves.

    Same sequences, same oracle, same comparison window — the only change is the reduction.
    A stitched arg-max yields several distinct per-position patterns across the sweep, which
    is precisely a 512-grid-dependent boundary and therefore a 512-grid-dependent IoU.
    """
    patterns = set()
    for phase in _PHASES:
        motif_start = _ANCHOR + phase
        stitched = np.argmax(_stitched(_sequence(_SEQ_LEN, motif_start), "step"), axis=1)
        patterns.add(
            tuple(
                int(c)
                for c in stitched[
                    motif_start - _PATTERN_FLANK : motif_start + _MOTIF_LEN + _PATTERN_FLANK
                ]
            )
        )
    assert len(patterns) > 1
    assert _expected_prediction_pattern() in patterns, (
        "the control never reproduces the right pattern — it is broken by construction "
        "rather than grid-dependent"
    )


@pytest.mark.parametrize("quality", sorted(_QUALITY))
@pytest.mark.parametrize("threshold", (0.2, 0.3, 0.4, 0.5))
def test_the_invariance_is_not_a_property_of_one_lucky_threshold(quality: str, threshold: float):
    """Seam-freeness must not be an artifact of ``_TAU`` sitting in a convenient gap.

    Measured band over the whole sweep: the reconciled posterior never falls below 0.5567 on
    a locus position nor rises above 0.1137 on a carrier position, so every threshold in
    ``(0.114, 0.556]`` yields the identical call at every phase. Four are checked.
    """
    outcomes = {
        _relative(
            _call(_reconciled(_sequence(_SEQ_LEN, _ANCHOR + phase), quality), threshold=threshold),
            _ANCHOR + phase,
        )
        for phase in _PHASES
    }
    assert outcomes == {((0, _MOTIF_LEN, _MOTIF_CLASS),)}


@pytest.mark.parametrize("quality", sorted(_QUALITY))
def test_the_separation_band_the_invariance_rests_on_is_measured_not_assumed(quality: str):
    """Record the margin, so a future change that narrows it fails here and not silently."""
    worst_locus, best_carrier = 1.0, 0.0
    for phase in _PHASES:
        motif_start = _ANCHOR + phase
        p_elem = element_posterior(_reconciled(_sequence(_SEQ_LEN, motif_start), quality).log_probs)
        locus = p_elem[motif_start : motif_start + _MOTIF_LEN]
        carrier = np.delete(p_elem, np.arange(motif_start, motif_start + _MOTIF_LEN))
        worst_locus = min(worst_locus, float(locus.min()))
        best_carrier = max(best_carrier, float(carrier.max()))
    assert best_carrier < 0.12 < 0.55 < worst_locus, (
        f"the locus/carrier separation collapsed: worst locus position {worst_locus:.4f}, "
        f"best carrier position {best_carrier:.4f}"
    )


# ── the matched control: without reconciliation, the same sweep is grid-dependent ──
@pytest.mark.parametrize("quality", sorted(_QUALITY))
def test_the_phase_sweep_has_power_because_the_naive_stitch_fails_it(quality: str):
    """If the stitch were also invariant, the test above would be measuring nothing.

    Measured on the identical sweep: the per-tile stitch yields **8** distinct outcomes under
    each model, including phases where it recovers a truncated locus, phases where it names
    the wrong dominant class, and — under ``ramp`` — phases where it finds no candidate at
    all. Against the reconciled path's **one**.
    """
    outcomes = {
        _relative(_stitch_call(_sequence(_SEQ_LEN, _ANCHOR + phase), quality), _ANCHOR + phase)
        for phase in _PHASES
    }
    assert len(outcomes) > 1, "the control is inert — it cannot show the operator does anything"
    assert outcomes != {((0, _MOTIF_LEN, _MOTIF_CLASS),)}


def test_the_naive_stitch_can_miss_the_locus_entirely_and_can_misname_its_element():
    """Name the two failure modes explicitly, so "differs" is not doing hidden work.

    ``ramp``: at phases 0, 16, 480 and 511 the stitch returns **no candidate at all** — the
    locus is simply absent from the scan. ``step``: at phase 480 it returns a 64-nt fragment
    whose dominant class is `Antiterminator_Tbox_seq`, not `Stem_I` — a locus that would reach
    Stage 2 with the wrong predicted element, which is what the §6 strand-resolver reads to
    orient it.
    """
    missed = [
        phase for phase in _PHASES if not _stitch_call(_sequence(_SEQ_LEN, _ANCHOR + phase), "ramp")
    ]
    assert missed, "the ramp control never misses the locus; it is weaker than measured"

    misnamed = [
        phase
        for phase in _PHASES
        for call in _relative(
            _stitch_call(_sequence(_SEQ_LEN, _ANCHOR + phase), "step"), _ANCHOR + phase
        )
        if call[2] != _MOTIF_CLASS
    ]
    assert misnamed, "the step control never misnames the element; it is weaker than measured"


@pytest.mark.parametrize("phase", (64, 96, 128, 192, 256, 320))
def test_the_stitch_control_agrees_wherever_the_grid_happens_to_be_kind(phase: int):
    """The control's positive leg: it *can* produce the right answer.

    At these phases the locus clears the truncated band of its owning tile, and the stitch
    recovers exactly what reconciliation does. A control that failed everywhere would be
    broken by construction rather than grid-dependent.
    """
    motif_start = _ANCHOR + phase
    seq = _sequence(_SEQ_LEN, motif_start)
    assert _relative(_stitch_call(seq, "step"), motif_start) == ((0, _MOTIF_LEN, _MOTIF_CLASS),)


@pytest.mark.parametrize("quality", sorted(_QUALITY))
def test_the_call_is_phase_invariant_but_the_reported_strength_is_not(quality: str):
    """A caveat the call-level invariance hides, and P3-12 inherits it.

    Seam-freeness is a statement about the **call** — the span and the element. It is *not* a
    statement about the strength reported beside it: averaging a truncated-context window in
    genuinely dilutes the posterior over the degraded band, and how much it dilutes depends
    on the shape of the degradation. Measured over the same sweep:

    ==========  =====================  ==================
    model       ``peak_p_elem``        ``mean_p_elem``
    ==========  =====================  ==================
    ``step``    one value (0.9997)     0.6453 → 0.9997
    ``ramp``    0.6078 → 0.8817        0.5709 → 0.8368
    ==========  =====================  ==================

    Mean strength moves by ~0.27–0.35 under **both** models. The peak is invariant only under
    hard truncation, where the locus always contains a position both windows see perfectly;
    under a ramp no position is scored perfectly at every phase, so even the peak moves.
    Neither is a defect of the operator — averaging a truncated-context window in genuinely
    dilutes the posterior. But any later rule that thresholds a locus on *mean* (or, under a
    graded model, *peak*) strength re-imports the 512-grid dependence that thresholding per
    position does not have. Recorded for P3-12's locus rule.
    """
    peaks, means = [], []
    for phase in _PHASES:
        (cand,) = _call(_reconciled(_sequence(_SEQ_LEN, _ANCHOR + phase), quality))
        peaks.append(cand.peak_p_elem)
        means.append(cand.mean_p_elem)

    assert max(means) - min(means) > 0.2, "strength barely moved — the caveat is overstated"
    if quality == "step":
        assert len(set(round(p, 9) for p in peaks)) == 1
    else:
        assert max(peaks) - min(peaks) > 0.2


# ═════════════════════════════════════════════════════════════════════════════
# 3. Where the guarantee stops — the coverage-1 contig terminus
# ═════════════════════════════════════════════════════════════════════════════
def test_the_pinned_tiling_leaves_the_first_and_last_512_nt_of_a_contig_single_covered():
    """PRD §6 says every *interior* nucleotide is scored by ≥ 2 windows. Measure the rest.

    ``tile_windows`` is tail-anchored and adds no window past either end, so on a 3,072-nt
    contig **1,024 positions** — the first and last 512 — are covered once. This is the
    arithmetic that bounds the claim above, stated as a number rather than left implicit.
    """
    reconciled = _reconciled(_sequence(_SEQ_LEN, _ANCHOR), "step")
    coverage = reconciled.coverage
    assert sorted(set(coverage.tolist())) == [1, 2]
    single = np.flatnonzero(coverage == 1)
    assert single.size == 2 * STRIDE_NT == 1024
    assert np.array_equal(single[:STRIDE_NT], np.arange(STRIDE_NT))
    assert np.array_equal(single[STRIDE_NT:], np.arange(_SEQ_LEN - STRIDE_NT, _SEQ_LEN))


@pytest.mark.parametrize("quality", sorted(_QUALITY))
def test_the_seam_free_guarantee_is_interior_only_and_the_terminus_carries_no_flag(quality: str):
    """The honest boundary of the P3-11 claim, and a gap P3-12 has to decide about.

    Planted in the coverage-1 5′ terminus instead of the interior, the same locus is **not**
    phase-invariant: there is no second window whose context compensates, so over eight
    phases the call takes **4** distinct forms under ``step`` (truncated by 64, 48 or 32 nt,
    or intact) and **5** under ``ramp`` — including one phase where nothing is called at all
    and two where the element is misnamed. Two consequences worth carrying:

    1. Seam-freeness is a property of the *doubly-covered interior*, exactly as PRD §6 words
       it — not of the operator alone.
    2. Nothing marks it. ``zero_flanked`` fires only for a window that ran off an end, and at
       the tail-anchored tiling no window does on a contig longer than one window — so on
       this sequence the flag is identically ``False`` while 1,024 nt are single-covered.
       ``Reconciled.coverage`` records it, but ``Candidate`` carries no coverage field, so a
       locus called from single-window evidence is indistinguishable downstream from one
       called from two.
    """
    outcomes = {
        _relative(_call(_reconciled(_sequence(_SEQ_LEN, phase), quality)), phase)
        for phase in (0, 16, 32, 64, 128, 200, 256, 320)
    }
    assert len(outcomes) > 1, (
        "the terminus behaved like the interior — either the tiling changed or this "
        "limitation has been closed, and the docstring above is now wrong"
    )
    assert ((0, _MOTIF_LEN, _MOTIF_CLASS),) in outcomes

    reconciled = _reconciled(_sequence(_SEQ_LEN, 0), quality)
    assert not reconciled.zero_flanked.any(), (
        "a long contig has no zero-flanked position at the pinned tiling; if that changed, "
        "the unflagged-single-coverage gap this test records may have been closed"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Contig ends — the zero-flank flag has to survive the reconcile → call hop
# ═════════════════════════════════════════════════════════════════════════════
def test_a_contig_shorter_than_one_window_flags_every_position_and_the_flag_reaches_the_candidate():
    """PRD §6 / D3: *contig ends are zero-flanked and flagged*. Flagged **where it is read**.

    `reconcile_windows` sets ``Reconciled.zero_flanked``; the claim only pays off if the value
    survives into the object locus construction hands downstream. On a 400-nt contig — one
    padded window, every position zero-flanked — the called candidate must carry
    ``n_zero_flanked == length``.
    """
    seq = _sequence(400, 100)
    reconciled = _reconciled(seq, "step")
    assert reconciled.n_windows == 1
    assert reconciled.zero_flanked.all()

    (cand,) = _call(reconciled)
    assert (cand.start, cand.end) == (100, 100 + _MOTIF_LEN)
    assert cand.n_zero_flanked == cand.length == _MOTIF_LEN


def test_an_interior_locus_carries_exactly_zero_zero_flanked_positions():
    """The control for the test above: the flag must not fire on everything.

    Without this, a `zero_flanked` hardwired to ``True`` would pass the contig-end test and
    silently mark every locus in the genome as pad-contaminated.
    """
    (cand,) = _call(_reconciled(_sequence(_SEQ_LEN, _ANCHOR), "step"))
    assert cand.n_zero_flanked == 0


@pytest.mark.parametrize("pads", ("nan", "loud"))
def test_pad_logits_never_reach_the_reduction_across_the_encode_hop(pads: str):
    """Poison every pad offset and require the result to be **bit-identical**.

    `encode_scan_window` pads a short contig's window tail with ``PAD_TOKEN_ID``, and those
    offsets describe no DNA. `reconcile_windows` slices them away before any arithmetic, so
    whatever sits there must be inert. Two probes, because they catch different leaks:

    * ``nan`` — the hazard `reconcile_windows` guards explicitly: ``np.argmax`` treats NaN as
      the maximum, so a leak would not merely propagate, it would *select a class*. A leak
      trips the operator's own non-finite refusal, i.e. this probe fails loudly.
    * ``loud`` — a large **finite** element logit. This is the probe the NaN one cannot
      replace: a finite leak raises no guard at all, and comparing two runs that both leak
      the same finite garbage would agree with each other and prove nothing. Here the leaked
      value would have to move the reconciled posterior, and bit-equality says it did not.

    `tests/unit/test_infer_scan.py` records that its own closed-form fixture cannot see a pad
    leak; this is the integration-side check it points at.
    """
    seq = _sequence(400, 100)
    window_ids, _ = scan_window_ids(seq)
    pad_mask = window_ids == PAD_TOKEN_ID
    assert int(pad_mask.sum()) == WINDOW_NT - 400 > 0

    # The probe must be live and targeted: it has to actually change the pad logits, and it
    # has to change nothing else. Without this a `_PAD_FILL` that quietly became the neutral
    # value would leave both assertions below trivially true.
    clean_logits = _oracle_logits(window_ids, "step")
    probe_logits = _oracle_logits(window_ids, "step", pads=pads)
    assert not np.array_equal(clean_logits[pad_mask], probe_logits[pad_mask])
    assert np.array_equal(clean_logits[~pad_mask], probe_logits[~pad_mask])

    clean = _reconciled(seq, "step")
    poisoned = _reconciled(seq, "step", pads=pads)
    assert np.array_equal(clean.log_probs, poisoned.log_probs)
    assert np.array_equal(clean.prediction, poisoned.prediction)
    assert _call(clean) == _call(poisoned)


# ═════════════════════════════════════════════════════════════════════════════
# 5. The fixture is not lying to itself
# ═════════════════════════════════════════════════════════════════════════════
def test_the_oracle_reads_token_ids_not_positions():
    """The encoder must be genuinely inside the loop.

    An oracle keyed on the offset rather than the base would make every assertion above a
    statement about `tile_windows` alone, with the encoder — and its pad convention — never
    exercised. Writing a second element run into the carrier must produce a second call at
    exactly its coordinates.
    """
    seq = list(_sequence(_SEQ_LEN, _ANCHOR))
    before = _call(_reconciled("".join(seq), "step"))
    seq[1400:1500] = ["C"] * 100
    after = _call(_reconciled("".join(seq), "step"))

    assert [(c.start, c.end) for c in before] == [(_ANCHOR, _ANCHOR + _MOTIF_LEN)]
    assert [(c.start, c.end) for c in after] == [(_ANCHOR, _ANCHOR + _MOTIF_LEN), (1400, 1500)]


def test_the_context_degradation_the_whole_file_rests_on_is_real():
    """If the oracle did not degrade at window edges, every control here would be vacuous.

    Both quality models must actually attenuate: the ``step`` model has a band of exactly
    ``_EDGE_NT`` blind offsets at each edge, the ``ramp`` model reaches 0 at the edges and 1
    at the centre, and in both the two offsets covering an interior position have edge
    distances summing to 511 — the geometric fact that makes averaging recover the call.
    """
    step = _quality_step(WINDOW_NT)
    ramp = _quality_ramp(WINDOW_NT)
    assert step[:_EDGE_NT].max() == 0.0 and step[-_EDGE_NT:].max() == 0.0
    assert step[_EDGE_NT] == 1.0 and step[WINDOW_NT // 2] == 1.0
    assert ramp[0] == 0.0 and ramp[WINDOW_NT // 2 - 1] == pytest.approx(1.0, abs=1e-3)

    for offset in (0, 1, 128, 255, 256, 511):
        near = min(offset, WINDOW_NT - 1 - offset)
        far = min(offset + STRIDE_NT, WINDOW_NT - 1 - (offset + STRIDE_NT))
        assert near + far == STRIDE_NT - 1


def test_the_geometry_under_test_is_the_pinned_one():
    """A sweep over a 512-nt period only means anything at stride 512 / window 1024."""
    assert (WINDOW_NT, STRIDE_NT) == (1024, 512)
    assert len(_PHASES) > 1 and min(_PHASES) == 0 and max(_PHASES) == STRIDE_NT - 1
    assert _ANCHOR >= STRIDE_NT
    assert _ANCHOR + max(_PHASES) + _MOTIF_LEN < _SEQ_LEN - STRIDE_NT
    assert _PATTERN_FLANK > 0, (
        "the arg-max comparison must include carrier either side of the locus, or it stops "
        "seeing the element→background transitions — which is where a boundary artifact lives"
    )
    assert _ELEMENT_LOGIT > 2 * _BACKGROUND_LOGIT, (
        "at or below 2x, the ramp's midpoint offset ties element against background and "
        "np.argmax resolves the tie to background — a fixture artifact, not an operator one"
    )


def test_the_element_mask_does_not_assume_background_is_the_lowest_class_index(monkeypatch):
    """Make the CodeRabbit r1 class-ordering fix non-vacuous.

    Under the shipped `CLASS_ORDER`, `background` is index 0, so ``cls > _BG`` and "is an
    element class" happen to coincide and the fix is a no-op — which is exactly the shape of
    change that gets reverted later because nothing shows it mattered. Re-point the oracle at
    a table where `background` sits mid-order and the coincidence breaks: a class *below* it
    must still receive ``_ELEMENT_LOGIT`` rather than being scored as carrier.
    """
    fake_bg = 3
    table = np.full_like(_TOKEN_CLASS, -1)
    table[BASE_TO_ID["A"]] = fake_bg
    table[BASE_TO_ID["N"]] = fake_bg
    table[BASE_TO_ID["C"]] = 1  # deliberately BELOW the background index
    table[BASE_TO_ID["G"]] = 5
    table[BASE_TO_ID["T"]] = 6
    monkeypatch.setitem(globals(), "_TOKEN_CLASS", table)
    monkeypatch.setitem(globals(), "_BG", fake_bg)

    window_ids = np.full((1, WINDOW_NT), BASE_TO_ID["A"], dtype=np.int16)
    window_ids[0, STRIDE_NT] = BASE_TO_ID["C"]
    logits = _oracle_logits(window_ids, "step")

    assert logits[0, STRIDE_NT, 1] == _ELEMENT_LOGIT
    assert logits[0, STRIDE_NT, fake_bg] == _BACKGROUND_LOGIT
    assert logits[0, 0, 1] == 0.0
