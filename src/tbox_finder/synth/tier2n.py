"""Synthetic **non-canonical-architecture** (Tier-2N) generator (P2-07).

ADR-0006 D9 row 5 routes **Tier-2N** — the flagship "CM-invisible" class — as
``(c)✓ ∧ (a)✓ ∧ (b)✗``: downstream-aaRS synteny and any-helix covariation hold,
while **relaxed-architecture detection fails**. Tier-2N is therefore defined by an
*architectural* departure, not by a score. This module builds synthetic Tier-2N
examples to back the ADR-0005 D14 probe set, the P5 spike-in recovery, and P6.

What the literature actually licenses (CLAUDE.md §10.1 evidence gate, 2026-07-18)
--------------------------------------------------------------------------------
An adversarial ≥2-independent-source review of the non-canonical T-box literature
admitted exactly **two** architecture families. Sixteen further candidate variants
were rejected — most collapsed under *lab* independence once reviews were stopped
from double-counting their own primary papers, and two are affirmatively
contradicted. Only these two are implemented:

``CLASS_II_PLATFORM_SWAP``
    The class-II / **translational** T-box: no intrinsic terminator; mutually
    exclusive sequestrator/antisequestrator helices occlude or expose the
    Shine-Dalgarno sequence, replacing the canonical expression platform.
    Actinobacteria, *ileS*. PMID:18359782 / DOI:10.1261/rna.819308;
    PMID:25583497 / DOI:10.1073/pnas.1424175112; PMID:31740853 /
    DOI:10.1038/s41594-019-0327-6; PMID:39097611 / DOI:10.1038/s41467-024-50885-x;
    PMID:41290600 / DOI:10.1038/s41467-025-65388-6; PMID:32882008 /
    DOI:10.1093/nar/gkaa721 (accessed 2026-07-18) — five non-overlapping labs.
``STEM_II_PK_DELETION``
    The glycyl / *glyQS* type: **Stem II and the Stem IIA/B H-type pseudoknot are
    jointly absent** (they co-delete as one event), leaving Stem I with its distal
    T-loop module intact, a lengthened interdomain linker, Stem III, and the
    antiterminator. Firmicutes glycyl leaders. PMID:28621923 /
    DOI:10.1021/acs.biochem.7b00284; PMID:28531275 / DOI:10.1093/nar/gkx518;
    PMID:38981655 / DOI:10.1261/rna.080071.124 (accessed 2026-07-18) — three
    independent structural/biochemical labs.

**Hard constraint** (Suddala & Zhang 2019, PMID:31206978, stated twice): no known
T-box lacks *all three* of {Stem I distal T-loop module, Stem II, Stem IIA/B}.
:data:`FORBIDDEN_JOINT_ABLATION` enforces it; that combination may be emitted only
as an explicit negative control, never as a Tier-2N positive. Two further variants
are affirmatively contradicted and have **no** generator bin: "Stem III absent"
(≥3 sources against) and "Stem I absent" (Stem I is obligate; TBDB located the
antiterminator 5'-UGGN-3' bulge in all but 48 of 23,535 sequences).

Why eligibility is a **triple**, and why a bare CM miss is not Tier-2N
----------------------------------------------------------------------
Two confounds sit between "the covariance model missed it" and "its architecture
is non-canonical", and both were measured in-repo at P2-07 rather than assumed.

**Confound 1 — parent divergence.** Over 500 corpus records (seed 42,
``RF00230.cm --cut_ga``) **27.6 % of real, unablated TBDB T-boxes are already
missed** (362/500 detected; independently reproduced at 219/300 = 27.0 % on the
generator's own parent sample). Labelling "the CM missed it" as Tier-2N would
therefore sweep in roughly a quarter of the ordinary corpus, grading *sequence
divergence* — the separate GATE-1 divergence arm — as *architectural* novelty.
Requiring the **parent to be CM-detected** removes it.

**Confound 2 — excision length.** This one is not intuitive and it invalidated
the first construction. Every ablation *shortens* the leader, and a covariance
model is an alignment: measured on 599 length-matched controls, excising an
equally long segment from a **random non-element position** breaks detection
**more** often (79.1 %) than removing the actual Stem II or Terminator (66.7 %).
So a ``--cut_ga`` miss is largely explained by excision *per se*. Requiring the
**length-matched control to remain CM-detected** removes it, keeping only variants
whose detection loss is attributable to *which* element was removed.

Eligibility is therefore a **triple**, and every variant carries its own control
so one cannot be omitted:

======================================  ===================================
parent CM-**detected**                  the parent is visible to begin with
variant CM-**missed**                   the ablation broke detection
length-matched control CM-**detected**  excision alone does *not* break it
======================================  ===================================

Measured yield at 300 parents (seed 20260719, this module's own code path): 599
emitted → **45 pass the triple** (21 class-II, 24 stem-II; both clear min-N
independently). The discards partition the emission exactly — 161 parent already
CM-missed + 200 ablation did not break detection + **193 length-confounded** + 45
eligible + 0 unmeasured = 599 — and each is counted under its own cause rather
than dropped. The 193 length-confounded are precisely what a parent/variant pair
filter would have admitted as probe positives; that filter yields 238 here and is
*not* a probe set. The per-family split is reported because a family sitting on
the floor would otherwise be invisible inside a pooled total.

*(Attribution is to the excised element, not yet to "relaxed-architecture
detection (b)" in the ADR-0006 D9 sense: no (b) backend exists in-repo until
P6-01/P6-11 — the element-level ``stem2_structure_only.cm`` was tested and is not
one, hitting 1/300 parents even at E ≤ 1000. The CM-based triple is a disclosed
proxy, pre-registered for re-derivation against the real (b) detector at P6.)*

Determinism: every draw derives from ``seed`` via :mod:`hashlib`-keyed selection,
never :mod:`random` global state, so a fixed seed reproduces the set byte-for-byte.

Honest limitation, restated in the dev-log and any card that ships this set: the
synthetic corpus samples **two** literature-grounded architecture departures, not
the space of natural non-canonical T-boxes; its diversity is a **lower bound**.
The claim that covariance models miss these architectures is *procedural* in the
literature (TBDB needed a second, class-II-specific CM at corpus scale —
PMID:32882008) — **no published miss rate exists**, so every recall figure this
project reports on Tier-2N is measured in-repo, first-party.

Pure stdlib at import time; ``pandas`` and the Infernal shell-out are lazy.
PRD §9.1, §12; ADR-0005 D14; ADR-0006 D9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tbox_finder.labels import CLASS_II_TYPE, CORPUS_PARQUET, ELEMENT_COORDS
from tbox_finder.power import MIN_REAL_HOMOLOG_N

# --------------------------------------------------------------------------- #
# The canonical architecture (pinned; PMID:31206978, corroborated PMID:32882008)
# --------------------------------------------------------------------------- #
#: Canonical element order along the leader, 5'→3', for *B. subtilis tyrS* — the
#: reference architecture named by Suddala & Zhang 2019 (PMID:31206978 /
#: DOI:10.1002/iub.2098, corroborated PMID:32882008 / DOI:10.1093/nar/gkaa721;
#: accessed 2026-07-18). Names match :data:`tbox_finder.labels.CLASS_ORDER`.
#:
#: Note the two review flags carried with this pin: the antiterminator's verified
#: functional core is the 4-nt 5'-UGGN-3' pairing the tRNA 5'-NCCA-3' end (the
#: "7-nt bulge" figure is lower-confidence); and the K-turn and the AG bulge are
#: **distinct** motifs on opposite sides of the Specifier — a conflation common in
#: secondary sources and deliberately not reproduced here.
CANONICAL_ELEMENT_ORDER: tuple[str, ...] = (
    "Stem_I",
    "Specifier",
    "Stem_II",
    "Stem_III",
    "Antiterminator_Tbox_seq",
    "Terminator",
)

#: Elements that are **obligate** in every documented natural T-box. An ablation
#: family may never remove one; doing so leaves the region of "unsupported but
#: plausible" constructs, which is a false-positive sink rather than a probe.
OBLIGATE_ELEMENTS: tuple[str, ...] = ("Stem_I", "Stem_III", "Antiterminator_Tbox_seq")

#: The joint ablation no known natural T-box exhibits (PMID:31206978). Emitted
#: only as a labelled negative control via ``allow_forbidden=True``.
FORBIDDEN_JOINT_ABLATION: frozenset[str] = frozenset({"Stem_I_DTM", "Stem_II", "Stem_IIA_B"})

# --------------------------------------------------------------------------- #
# The two literature-grounded ablation families (frozen; no CLI/config override)
# --------------------------------------------------------------------------- #
FAMILY_CLASS_II = "CLASS_II_PLATFORM_SWAP"
FAMILY_STEM_II_PK = "STEM_II_PK_DELETION"

#: family → the citations that cleared the ≥2-independent-source bar. Carried into
#: every emitted variant so a downstream consumer can trace the construct's
#: grounding without re-reading this module.
FAMILY_CITATIONS: dict[str, tuple[str, ...]] = {
    FAMILY_CLASS_II: (
        "PMID:18359782",
        "PMID:25583497",
        "PMID:31740853",
        "PMID:39097611",
        "PMID:41290600",
        "PMID:32882008",
    ),
    FAMILY_STEM_II_PK: ("PMID:28621923", "PMID:28531275", "PMID:38981655"),
}

#: family → the elements its ablation removes from the parent, by
#: :data:`tbox_finder.labels.ELEMENT_COORDS` key.
FAMILY_ABLATED_ELEMENTS: dict[str, tuple[str, ...]] = {
    # Class II replaces the expression platform: the intrinsic terminator is gone
    # (a sequestrator/antisequestrator pair takes its place). The decoding module
    # — Stem I, Specifier, Stem III, antiterminator — is retained.
    FAMILY_CLASS_II: ("Terminator",),
    # glyQS: Stem II and the IIA/B pseudoknot co-delete. The repo annotates the
    # pair as one ``stem2_region`` extent, so removing it removes both.
    FAMILY_STEM_II_PK: ("Stem_II",),
}

VALIDATED_FAMILIES: tuple[str, ...] = (FAMILY_CLASS_II, FAMILY_STEM_II_PK)

#: The Tier-2N probe-set floor. Imported from the ADR-0005 Amendment A1 pin —
#: never re-declared here (one number, one definition).
TIER2N_PROBE_MIN_N = MIN_REAL_HOMOLOG_N

#: Identifier prefix for a variant's length-matched excision control.
CONTROL_ID_PREFIX = "ctl:"

#: Measured confound baselines, recorded so a future reader can see what the
#: control arm is for without re-running the experiment (P2-07, RF00230.cm
#: --cut_ga). ``parent_miss_rate`` is the divergence confound; the two excision
#: rates are the length confound — note the **random** excision is the more
#: destructive of the two, which is what invalidated the pair-only construction.
MEASURED_CONFOUND_BASELINES: dict[str, float] = {
    "parent_miss_rate": 0.276,  # 362/500 detected, n=500, seed 42
    "element_excision_miss_rate": 0.667,  # 200/600 detected, n=600, seed 20260719
    "random_excision_miss_rate": 0.791,  # 125/599 detected, n=599, seed 20260719
}


class Tier2NGeneratorError(ValueError):
    """Raised on a malformed parent record or an unsupported ablation request."""


@dataclass(frozen=True)
class Tier2NVariant:
    """One synthetic non-canonical variant, with its parent and its length control.

    ``control_sequence`` is a length-matched excision from a **non-element**
    position of the same parent, built at emission time so it cannot be omitted.
    The three ``cm_detected_*`` fields are filled by :func:`classify_pairs` after
    the covariance-model run; until then they are ``None`` and
    :meth:`is_probe_eligible` is ``False`` — an unmeasured variant is never
    probe-eligible.
    """

    variant_id: str
    parent_record_id: str
    family: str
    ablated_elements: tuple[str, ...]
    sequence: str
    parent_sequence: str
    control_sequence: str = ""
    citations: tuple[str, ...] = ()
    unvalidated: bool = False
    cm_detected_parent: bool | None = None
    cm_detected_variant: bool | None = None
    cm_detected_control: bool | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def control_id(self) -> str:
        """Identifier of this variant's length-matched control construct."""
        return f"{CONTROL_ID_PREFIX}{self.variant_id}"

    def is_probe_eligible(self) -> bool:
        """True iff the **triple** holds: parent detected, variant missed, control detected.

        All three measurements must be present. ``None`` (never measured) is not
        treated as ``False`` — an unmeasured leg makes the variant ineligible
        rather than silently counting as a CM miss, which would let an unrun
        cmsearch inflate the probe set to any size.

        The control leg is what makes this an *architecture* claim rather than an
        *excision* claim: a length-matched cut elsewhere in the same parent must
        leave detection intact (module docstring, confound 2).
        """
        if (
            self.cm_detected_parent is None
            or self.cm_detected_variant is None
            or self.cm_detected_control is None
        ):
            return False
        return (
            bool(self.cm_detected_parent)
            and not bool(self.cm_detected_variant)
            and bool(self.cm_detected_control)
        )


def _stable_key(seed: int, *parts: str) -> int:
    """Deterministic 64-bit key from ``seed`` + ``parts`` (no global RNG state)."""
    digest = hashlib.shake_256(("|".join((str(seed), *parts))).encode("utf-8")).digest(8)
    return int.from_bytes(digest, "big")


def _element_span(parent: dict[str, Any], element: str) -> tuple[int, int] | None:
    """0-based half-open span of ``element`` within the parent window, or ``None``."""
    cols = ELEMENT_COORDS.get(element)
    if cols is None:
        raise Tier2NGeneratorError(f"no coordinate columns registered for element {element!r}")
    start_col, end_col = cols
    start, end = parent.get(start_col), parent.get(end_col)
    if start is None or end is None:
        return None
    try:
        lo, hi = int(start), int(end)
    except (TypeError, ValueError):
        return None
    if lo != lo or hi != hi:  # NaN
        return None
    lo, hi = min(lo, hi), max(lo, hi)
    if lo < 0 or hi <= lo:
        return None
    return lo, hi


def ablate(
    parent: dict[str, Any],
    family: str,
    *,
    sequence_key: str = "FASTA_sequence",
    allow_forbidden: bool = False,
) -> str:
    """Excise ``family``'s elements from the parent sequence by annotated extent.

    The ablation is a **deletion of real annotated coordinates**, not synthesised
    sequence — the construct is a subsequence of a natural leader, so nothing is
    fabricated. Raises if the parent lacks the extent the family targets (a
    silently-unablated "variant" would be an unperturbed copy of its parent and
    would corrupt the discordant-pair measurement).
    """
    if family not in FAMILY_ABLATED_ELEMENTS:
        raise Tier2NGeneratorError(f"unknown family {family!r}; expected {VALIDATED_FAMILIES}")
    sequence = parent.get(sequence_key)
    if not isinstance(sequence, str) or not sequence:
        raise Tier2NGeneratorError(f"parent has no usable {sequence_key!r}")

    targets = FAMILY_ABLATED_ELEMENTS[family]
    for element in targets:
        if element in OBLIGATE_ELEMENTS and not allow_forbidden:
            raise Tier2NGeneratorError(
                f"family {family!r} would ablate the obligate element {element!r}"
            )

    spans: list[tuple[int, int]] = []
    for element in targets:
        span = _element_span(parent, element)
        if span is None:
            raise Tier2NGeneratorError(
                f"parent lacks an annotated {element!r} extent, so family {family!r} "
                "cannot be applied without leaving the sequence unperturbed"
            )
        spans.append(span)

    keep: list[str] = []
    cursor = 0
    for lo, hi in sorted(spans):
        lo, hi = max(lo, 0), min(hi, len(sequence))
        if lo >= len(sequence) or hi <= cursor:
            continue
        keep.append(sequence[cursor:lo])
        cursor = max(cursor, hi)
    keep.append(sequence[cursor:])
    ablated = "".join(keep)
    if ablated == sequence:
        raise Tier2NGeneratorError(
            f"family {family!r} left the parent sequence unchanged (extents out of range)"
        )
    if not ablated:
        raise Tier2NGeneratorError(f"family {family!r} removed the entire parent sequence")
    return ablated


def length_matched_control(
    parent: dict[str, Any],
    family: str,
    *,
    n_removed: int,
    seed: int,
    sequence_key: str = "FASTA_sequence",
) -> str:
    """Excise ``n_removed`` nt from a **non-element** position of the same parent.

    The control answers "would cutting this much out anywhere have broken
    detection?". Its excision window is chosen deterministically from ``seed`` and
    is constrained not to overlap the extent ``family`` targets, so the only
    difference from the real variant is *which* nucleotides were removed — length,
    parent, and composition context are held fixed.

    Raises if no non-overlapping window of the required length exists; a control
    that silently fell back to overlapping the element would defeat its purpose.
    """
    sequence = parent.get(sequence_key)
    if not isinstance(sequence, str) or not sequence:
        raise Tier2NGeneratorError(f"parent has no usable {sequence_key!r}")
    if n_removed <= 0 or n_removed >= len(sequence):
        raise Tier2NGeneratorError(
            f"n_removed must be in (0, len(sequence)); got {n_removed} for a "
            f"{len(sequence)}-nt parent"
        )
    element = FAMILY_ABLATED_ELEMENTS[family][0]
    span = _element_span(parent, element)
    if span is None:
        raise Tier2NGeneratorError(f"parent lacks an annotated {element!r} extent")
    lo, hi = span

    starts = [s for s in range(0, len(sequence) - n_removed + 1) if s + n_removed <= lo or s >= hi]
    if not starts:
        raise Tier2NGeneratorError(
            f"no {n_removed}-nt window avoids the {element!r} extent in this parent"
        )
    key = _stable_key(seed, "control", family, str(parent.get("record_id", "")))
    start = starts[key % len(starts)]
    return sequence[:start] + sequence[start + n_removed :]


def generate(
    parents: list[dict[str, Any]],
    *,
    seed: int,
    families: tuple[str, ...] = VALIDATED_FAMILIES,
    sequence_key: str = "FASTA_sequence",
    record_id_key: str = "record_id",
    max_per_family: int | None = None,
) -> list[Tier2NVariant]:
    """Emit synthetic non-canonical variants, deterministically under ``seed``.

    Parents are visited in a seed-derived stable order (not input order), so the
    emitted set is reproducible but not an artefact of how the corpus happened to
    be sorted. A parent that cannot carry a family's ablation is skipped for that
    family and retained for the others.

    The class-II family is applied only to parents that are **not already class
    II** — swapping a translational platform onto a translational leader is a
    no-op, and counting it would inflate the set with unperturbed copies.
    """
    for family in families:
        if family not in VALIDATED_FAMILIES:
            raise Tier2NGeneratorError(
                f"unknown family {family!r}; only {VALIDATED_FAMILIES} cleared the "
                "CLAUDE.md §10.1 evidence gate"
            )

    out: list[Tier2NVariant] = []
    for family in families:
        ordered = sorted(
            parents,
            key=lambda p: _stable_key(seed, family, str(p.get(record_id_key, ""))),
        )
        emitted = 0
        for parent in ordered:
            if max_per_family is not None and emitted >= max_per_family:
                break
            record_id = str(parent.get(record_id_key, ""))
            if not record_id:
                continue
            if family == FAMILY_CLASS_II and str(parent.get("Type", "")) == CLASS_II_TYPE:
                continue
            parent_sequence = str(parent.get(sequence_key) or "")
            try:
                sequence = ablate(parent, family, sequence_key=sequence_key)
                # Built here, not by the caller: a variant without its control
                # cannot be probe-eligible, so the two must be emitted together.
                control = length_matched_control(
                    parent,
                    family,
                    n_removed=len(parent_sequence) - len(sequence),
                    seed=seed,
                    sequence_key=sequence_key,
                )
            except Tier2NGeneratorError:
                continue
            out.append(
                Tier2NVariant(
                    variant_id=f"tier2n:{family}:{record_id}",
                    parent_record_id=record_id,
                    family=family,
                    ablated_elements=tuple(FAMILY_ABLATED_ELEMENTS[family]),
                    sequence=sequence,
                    parent_sequence=parent_sequence,
                    control_sequence=control,
                    citations=FAMILY_CITATIONS[family],
                    notes={
                        "parent_length": len(parent_sequence),
                        "variant_length": len(sequence),
                        "control_length": len(control),
                        "nt_removed": len(parent_sequence) - len(sequence),
                    },
                )
            )
            emitted += 1
    return out


def classify_pairs(
    variants: list[Tier2NVariant],
    parent_detected: dict[str, bool],
    variant_detected: dict[str, bool],
    control_detected: dict[str, bool] | None = None,
) -> list[Tier2NVariant]:
    """Attach the three covariance-model verdicts of the triple to each variant.

    ``parent_detected`` is keyed by ``parent_record_id``, ``variant_detected`` by
    ``variant_id``, ``control_detected`` by ``control_id``. A variant missing from
    any map keeps ``None`` for that leg and stays probe-ineligible — absence of a
    measurement is never read as a miss.

    ``control_detected=None`` means the control arm was not run at all, which
    leaves **every** variant ineligible rather than silently degrading to the
    confounded pair filter.
    """
    controls = control_detected or {}
    out: list[Tier2NVariant] = []
    for variant in variants:
        out.append(
            Tier2NVariant(
                variant_id=variant.variant_id,
                parent_record_id=variant.parent_record_id,
                family=variant.family,
                ablated_elements=variant.ablated_elements,
                sequence=variant.sequence,
                parent_sequence=variant.parent_sequence,
                control_sequence=variant.control_sequence,
                citations=variant.citations,
                unvalidated=variant.unvalidated,
                cm_detected_parent=parent_detected.get(variant.parent_record_id),
                cm_detected_variant=variant_detected.get(variant.variant_id),
                cm_detected_control=controls.get(variant.control_id),
                notes=dict(variant.notes),
            )
        )
    return out


def build_report(variants: list[Tier2NVariant], *, seed: int) -> dict[str, Any]:
    """Summarise an emitted + classified set, with the probe-set min-N gate.

    Every count is re-derived from ``variants`` here rather than accumulated by
    the caller, so the report cannot describe a set other than the one it was
    handed. ``probe_set_meets_min_n`` is guarded on the probe set being non-empty:
    a clause derived from the *requested* configuration rather than the *found*
    evidence is vacuously true exactly when the evidence is missing.
    """

    def _measured(v: Tier2NVariant) -> bool:
        return (
            v.cm_detected_parent is not None
            and v.cm_detected_variant is not None
            and v.cm_detected_control is not None
        )

    eligible = [v for v in variants if v.is_probe_eligible()]
    unmeasured = [v for v in variants if not _measured(v)]
    parent_already_missed = [v for v in variants if _measured(v) and not v.cm_detected_parent]
    ablation_did_not_break = [
        v for v in variants if _measured(v) and v.cm_detected_parent and v.cm_detected_variant
    ]
    # The confound the control arm exists to catch: detection was lost, but a
    # length-matched cut elsewhere loses it too, so the loss is not attributable
    # to the element. Counted explicitly — this is the discard that the pair-only
    # construction silently admitted as a probe positive.
    length_confounded = [
        v
        for v in variants
        if _measured(v)
        and v.cm_detected_parent
        and not v.cm_detected_variant
        and not v.cm_detected_control
    ]

    per_family: dict[str, dict[str, int]] = {}
    for family in sorted({v.family for v in variants}):
        fam = [v for v in variants if v.family == family]
        fam_eligible = [v for v in fam if v.is_probe_eligible()]
        per_family[family] = {
            "emitted": len(fam),
            "probe_eligible": len(fam_eligible),
            # Reported per family because a family sitting on the floor would be
            # invisible inside a pooled total.
            "meets_min_n_alone": bool(fam_eligible) and len(fam_eligible) >= TIER2N_PROBE_MIN_N,
        }

    n_probe = len(eligible)
    meets_min_n = bool(eligible) and n_probe >= TIER2N_PROBE_MIN_N

    return {
        "seed": seed,
        "n_emitted": len(variants),
        "n_probe_eligible": n_probe,
        "n_unmeasured": len(unmeasured),
        "n_discarded_parent_already_cm_missed": len(parent_already_missed),
        "n_discarded_ablation_did_not_break_detection": len(ablation_did_not_break),
        "n_discarded_length_confounded": len(length_confounded),
        "per_family": per_family,
        "families_validated": list(VALIDATED_FAMILIES),
        "family_citations": {k: list(v) for k, v in FAMILY_CITATIONS.items()},
        "tier2n_probe_min_n": TIER2N_PROBE_MIN_N,
        "probe_set_meets_min_n": meets_min_n,
        "eligibility_rule": (
            "probe-eligible iff parent CM-detected AND variant CM-missed AND "
            "length-matched control CM-detected (the triple). The parent leg "
            "controls the divergence confound (27.6% of unablated corpus records "
            "are already missed); the control leg controls the excision-length "
            "confound (random equal-length excision misses 79.1%, MORE than the "
            "66.7% of real element ablations) — without it the filter measures "
            "excision, not architecture"
        ),
        "measured_confound_baselines": dict(MEASURED_CONFOUND_BASELINES),
        "limitation": (
            "samples two literature-grounded architecture departures, not the space "
            "of natural non-canonical T-boxes; diversity is a lower bound. "
            "Attribution is to the excised element via a CM proxy, not to "
            "ADR-0006 D9's relaxed-architecture detector (b), which has no in-repo "
            "backend until P6 — pre-registered for re-derivation there"
        ),
    }


# --------------------------------------------------------------------------- #
# CLI — build the probe set end-to-end so the committed report has a producer
# --------------------------------------------------------------------------- #
#: Default output path for the committed probe-set report.
REPORT_PATH = Path("reports/p2/tier2n_probe.json")

#: Columns a parent must carry for BOTH families to be applicable.
REQUIRED_PARENT_COLUMNS: tuple[str, ...] = (
    "FASTA_sequence",
    "stem2_region_start",
    "stem2_region_end",
    "term_start",
    "term_end",
)


def build_probe_report(
    *,
    corpus_parquet: str | Path = CORPUS_PARQUET,
    cm: str | Path | None = None,
    n_parents: int = 300,
    seed: int = 20260719,
    min_len: int = 120,
    max_len: int = 600,
    cpu: int = 6,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate variants + controls, run the three cmsearch arms, build the report.

    Heavy imports (``pandas``) and the Infernal shell-out are deferred to here so
    the module stays importable in the bare-CI unit tier.
    """
    import tempfile

    import pandas as pd

    from tbox_finder.infernal import (
        RF00230_CM,
        detection_map,
        run_cmsearch,
        write_fasta,
    )
    from tbox_finder.infernal import (
        build_report as cm_report,
    )

    cm_path = Path(cm) if cm is not None else RF00230_CM
    frame = pd.read_parquet(corpus_parquet)
    subset = frame.dropna(subset=list(REQUIRED_PARENT_COLUMNS)).copy()
    lengths = subset["FASTA_sequence"].str.len()
    subset = subset[(lengths >= min_len) & (lengths <= max_len)]
    subset = subset.sample(n=min(n_parents, len(subset)), random_state=seed)
    subset["record_id"] = [f"p{i:05d}" for i in range(len(subset))]
    parents = subset.to_dict("records")

    variants = generate(parents, seed=seed)

    parent_fa = {str(p["record_id"]): str(p["FASTA_sequence"]) for p in parents}
    variant_fa = {v.variant_id: v.sequence for v in variants}
    control_fa = {v.control_id: v.control_sequence for v in variants}

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="tier2n-"))
    tmp.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, Any]] = {}
    detections: dict[str, dict[str, bool]] = {}
    for name, records in (
        ("parents", parent_fa),
        ("variants", variant_fa),
        ("controls", control_fa),
    ):
        fasta = write_fasta(records, tmp / f"{name}.fa")
        hits = run_cmsearch(cm_path, fasta, tmp / f"{name}.tbl", cpu=cpu)
        detections[name] = detection_map(records, hits)
        arms[name] = cm_report(records, hits, arm=name)

    classified = classify_pairs(
        variants, detections["parents"], detections["variants"], detections["controls"]
    )
    report = build_report(classified, seed=seed)
    report["arms"] = arms
    report["n_parents"] = len(parents)
    report["cm"] = str(cm_path)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tbox_finder.synth.tier2n")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the Tier-2N probe set + report")
    build.add_argument("--corpus", default=str(CORPUS_PARQUET))
    build.add_argument("--cm", default=None)
    build.add_argument("--n-parents", type=int, default=300)
    build.add_argument("--seed", type=int, default=20260719)
    build.add_argument("--cpu", type=int, default=6)
    build.add_argument("--out", default=str(REPORT_PATH))
    build.add_argument("--workdir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_probe_report(
        corpus_parquet=args.corpus,
        cm=args.cm,
        n_parents=args.n_parents,
        seed=args.seed,
        cpu=args.cpu,
        workdir=args.workdir,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"tier2n probe set: {report['n_probe_eligible']} eligible "
        f"(min-N {report['tier2n_probe_min_n']}) -> "
        f"{'PASS' if report['probe_set_meets_min_n'] else 'FAIL'}; wrote {out}"
    )
    return 0 if report["probe_set_meets_min_n"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
