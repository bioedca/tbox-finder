"""Unit tier — serializing the Tier-2N probe set (P3-15′-i).

``mining.remine.load_probe_set`` shipped as a **reader with no writer**: it reads
``{"natural": [...], "synthetic": [...]}``, refuses an empty set by design, and
nothing in ``src/`` ever serialized a :class:`ProbeSet`. The round leg that needs
the members (``remine apply-spare-rule --probe-set``) therefore could not start at
all. These tests drive the writer that closes that gap.

Three properties carry the weight:

* the written file **round-trips through the real reader**, by member identity —
  not by count, since a count can agree while the members do not;
* the members are **reconciled against the counts report of the same run**, and a
  disagreement leaves **no file behind** (the guard runs before its side effect);
* a below-min-N probe set is **refused at the point it is minted**. ``load_probe_set``
  only refuses a wholly empty set, so a cheap ``--n-parents 30`` build would
  otherwise mint a well-formed file carrying an underpowered instrument into the
  ADR-0005 D14 halt/rollback decision, with every clause reading green.

The committed-artifact checks at the end are deliberately *additional* to the
function-level ones: a test that only reads a committed report cannot see the code
that produced it, so it can stay green through any rewrite of the writer.

Bare-CI tier: pure stdlib, no numpy/pandas/torch.
"""

from __future__ import annotations

import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any

import pytest

from tbox_finder.eval.tier2n_probe import (
    PROBE_SET_SCHEMA_VERSION,
    TIER2N_PROBE_MIN_N,
    ProbeSet,
    Tier2NProbeError,
    build_probe_set,
    probe_set_payload,
    reconcile_probe_set_with_report,
    synthetic_probe_id_family,
    write_probe_set,
)
from tbox_finder.mining.remine import RemineError, load_probe_set, probe_member_ids
from tbox_finder.synth import tier2n
from tbox_finder.synth.tier2n import (
    FAMILY_CLASS_II,
    FAMILY_STEM_II_PK,
    PROBE_IDS_PATH,
    REPORT_PATH,
    Tier2NBuild,
    Tier2NVariant,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Deliberately **asymmetric** family sizes summing past the min-N floor. Equal
#: arms would let a swapped or mis-attributed per-family count reconcile anyway.
N_CLASS_II = 12
N_STEM_II_PK = 9

#: Byte pin on the committed probe-ids artifact, produced by the 831 s P3-15′-i
#: regeneration. Update it in the same commit that regenerates the artifact.
COMMITTED_PROBE_IDS_SHA256 = "ea8c3f7ce31940c6cd4fa5beb958e07d43e3353a469df551e493850842f776e3"


def _variant(vid: str, family: str, *, eligible: bool = True) -> Tier2NVariant:
    return Tier2NVariant(
        variant_id=vid,
        parent_record_id=f"p_{vid}",
        family=family,
        ablated_elements=("Stem_II",),
        sequence="AAAA",
        parent_sequence="AAAACCCC",
        control_sequence="CCCC",
        cm_detected_parent=True,
        cm_detected_variant=not eligible,
        cm_detected_control=True,
    )


def _variants() -> list[Tier2NVariant]:
    """Emitted deliberately **out of sorted order**, and interleaved across families.

    ``generate`` visits parents in a seed-derived ``_stable_key`` hash order, so the
    real eligible ids never arrive sorted. A fixture that happens to be emitted in
    ascending order makes ``build_probe_set``'s ``sorted()`` — the only thing that
    makes the committed artifact byte-stable across regenerations — invisible to
    every assertion in this file.
    """
    stem = [
        _variant(f"tier2n:{FAMILY_STEM_II_PK}:p{i:05d}", FAMILY_STEM_II_PK)
        for i in reversed(range(100, 100 + N_STEM_II_PK))
    ]
    class_ii = [
        _variant(f"tier2n:{FAMILY_CLASS_II}:p{i:05d}", FAMILY_CLASS_II)
        for i in reversed(range(N_CLASS_II))
    ]
    out: list[Tier2NVariant] = []
    for pair in zip_longest(stem, class_ii):
        out += [v for v in pair if v is not None]
    # An ineligible variant: it must reach neither the ids nor the per-family counts,
    # so the reconciliation is over the *eligible* set.
    out.append(_variant(f"tier2n:{FAMILY_CLASS_II}:p09999", FAMILY_CLASS_II, eligible=False))
    return out


def _probe_set() -> ProbeSet:
    return build_probe_set(_variants())


def _report(**overrides: Any) -> dict[str, Any]:
    report = tier2n.build_report(_variants(), seed=20260719)
    report.update(overrides)
    return report


# --------------------------------------------------------------------------- #
# The fixture itself must be able to fail the assertions below
# --------------------------------------------------------------------------- #
def test_the_fixture_is_asymmetric_and_clears_min_n() -> None:
    """Guards every test below: symmetric arms hide swaps, sub-min-N hides the floor."""
    assert N_CLASS_II != N_STEM_II_PK
    probe_set = _probe_set()
    assert probe_set.size == N_CLASS_II + N_STEM_II_PK
    assert probe_set.size >= TIER2N_PROBE_MIN_N
    report = _report()
    assert report["per_family"][FAMILY_CLASS_II]["probe_eligible"] == N_CLASS_II
    assert report["per_family"][FAMILY_STEM_II_PK]["probe_eligible"] == N_STEM_II_PK


# --------------------------------------------------------------------------- #
# Round-trip through the real reader
# --------------------------------------------------------------------------- #
def test_the_written_file_round_trips_through_load_probe_set(tmp_path: Path) -> None:
    """Identity, not count: an equal-sized set of different members is not a pass."""
    probe_set = _probe_set()
    path = write_probe_set(tmp_path / "ids.json", probe_set, _report())

    reloaded = load_probe_set(path)
    assert reloaded.synthetic == probe_set.synthetic
    assert reloaded.natural == probe_set.natural
    assert probe_member_ids(reloaded) == set(probe_set.synthetic)


def test_the_empty_natural_arm_survives_the_round_trip(tmp_path: Path) -> None:
    """``[]`` is written, so a reader cannot mistake "no members" for "no arm"."""
    path = write_probe_set(tmp_path / "ids.json", _probe_set(), _report())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["natural"] == []
    assert load_probe_set(path).natural == ()


def test_the_counts_report_itself_is_still_refused_as_a_probe_set(tmp_path: Path) -> None:
    """The negative control the writer exists to make unnecessary — and must not break."""
    counts_only = tmp_path / "counts.json"
    counts_only.write_text(json.dumps(_report()), encoding="utf-8")
    with pytest.raises(RemineError, match="probe set is empty"):
        load_probe_set(counts_only)


def test_the_payload_is_a_pure_function_of_the_run(tmp_path: Path) -> None:
    """No timestamp, no git SHA: a diff on this file means the *members* moved."""
    first = write_probe_set(tmp_path / "a.json", _probe_set(), _report()).read_bytes()
    second = write_probe_set(tmp_path / "b.json", _probe_set(), _report()).read_bytes()
    assert first == second


def test_the_payload_carries_its_schema_version_as_a_json_list() -> None:
    payload = probe_set_payload(_probe_set())
    assert payload["schema_version"] == PROBE_SET_SCHEMA_VERSION
    assert isinstance(payload["synthetic"], list)
    assert "provenance" not in payload


def test_the_arms_are_sorted_not_merely_emission_ordered() -> None:
    """``sorted()`` is what makes the artifact byte-stable, so the fixture must be unsorted.

    Asserting ``arm == sorted(arm)`` against an already-ascending fixture holds whether
    or not the code sorts. The guard below refuses that fixture outright.
    """
    variants = _variants()
    emitted = [v.variant_id for v in variants if v.is_probe_eligible()]
    assert emitted != sorted(emitted), "fixture is sorted, so the assert below is vacuous"
    payload = probe_set_payload(build_probe_set(variants))
    assert payload["synthetic"] == sorted(emitted)


def test_provenance_is_carried_verbatim_when_supplied() -> None:
    payload = probe_set_payload(_probe_set(), provenance={"source_report_sha256": "abc"})
    assert payload["provenance"] == {"source_report_sha256": "abc"}


# --------------------------------------------------------------------------- #
# The reconciliation — each clause broken ALONE
# --------------------------------------------------------------------------- #
def test_a_consistent_pair_reconciles_clean() -> None:
    """The positive control: a guard that refused everything would satisfy the rest."""
    assert reconcile_probe_set_with_report(_probe_set(), _report()) == []


@pytest.mark.parametrize(
    "key",
    ["n_synthetic", "n_probe_eligible", "probe_set_size", "n_natural"],
)
def test_each_top_level_count_is_reconciled_alone(key: str) -> None:
    problems = reconcile_probe_set_with_report(_probe_set(), _report(**{key: 999}))
    assert any(key in p for p in problems), problems


@pytest.mark.parametrize("key", ["n_synthetic", "n_probe_eligible", "probe_set_size", "n_natural"])
def test_a_missing_count_is_a_problem_not_a_skipped_clause(key: str) -> None:
    report = _report()
    del report[key]
    problems = reconcile_probe_set_with_report(_probe_set(), report)
    assert any(key in p and "no" in p for p in problems), problems


def test_a_non_integer_count_is_a_problem() -> None:
    problems = reconcile_probe_set_with_report(_probe_set(), _report(n_synthetic="21"))
    assert any("not an integer" in p for p in problems), problems


def test_a_boolean_count_is_not_accepted_as_an_integer() -> None:
    """``True == 1`` in Python, so a bool must be refused explicitly."""
    problems = reconcile_probe_set_with_report(_probe_set(), _report(n_natural=False))
    assert any("not an integer" in p for p in problems), problems


def test_one_family_count_alone_breaks_the_reconciliation() -> None:
    """Break a single family: an all-wrong fixture cannot distinguish AND from OR."""
    report = _report()
    report["per_family"][FAMILY_STEM_II_PK]["probe_eligible"] = N_STEM_II_PK + 1
    problems = reconcile_probe_set_with_report(_probe_set(), report)
    assert any(FAMILY_STEM_II_PK in p for p in problems), problems
    assert not any(FAMILY_CLASS_II in p for p in problems), problems


def test_swapped_family_counts_are_caught() -> None:
    """The asymmetric fixture's reason for existing."""
    report = _report()
    report["per_family"][FAMILY_CLASS_II]["probe_eligible"] = N_STEM_II_PK
    report["per_family"][FAMILY_STEM_II_PK]["probe_eligible"] = N_CLASS_II
    problems = reconcile_probe_set_with_report(_probe_set(), report)
    assert len(problems) == 2, problems


def test_an_absent_per_family_block_is_refused_rather_than_looped_over_vacuously() -> None:
    problems = reconcile_probe_set_with_report(_probe_set(), _report(per_family={}))
    assert any("per_family" in p for p in problems), problems


def test_a_per_family_entry_missing_its_count_names_the_family() -> None:
    """A diagnostic that said only "no 'probe_eligible'" would not say which one."""
    report = _report()
    del report["per_family"][FAMILY_CLASS_II]["probe_eligible"]
    problems = reconcile_probe_set_with_report(_probe_set(), report)
    assert any(FAMILY_CLASS_II in p and "probe_eligible" in p for p in problems), problems


def test_a_family_present_in_the_ids_but_absent_from_the_report_is_caught() -> None:
    report = _report()
    del report["per_family"][FAMILY_CLASS_II]
    problems = reconcile_probe_set_with_report(_probe_set(), report)
    assert any(FAMILY_CLASS_II in p for p in problems), problems


def test_a_family_present_in_the_report_but_absent_from_the_ids_is_caught() -> None:
    report = _report()
    report["per_family"]["A_THIRD_FAMILY"] = {"emitted": 3, "probe_eligible": 3}
    problems = reconcile_probe_set_with_report(_probe_set(), report)
    assert any("A_THIRD_FAMILY" in p for p in problems), problems


def test_a_report_that_is_not_an_object_is_refused() -> None:
    problems = reconcile_probe_set_with_report(_probe_set(), ["not", "a", "mapping"])  # type: ignore[arg-type]
    assert any("not an object" in p for p in problems), problems


@pytest.mark.parametrize(
    "bad_id", ["p00001", "tier2n:p00001", "tier2n::p00001", "x:FAM:p1", "tier2n:FAM:"]
)
def test_an_unparsable_synthetic_id_raises_rather_than_bucketing_into_nothing(
    bad_id: str,
) -> None:
    with pytest.raises(Tier2NProbeError, match="tier2n:<FAMILY>:<record>"):
        synthetic_probe_id_family(bad_id)


def test_an_unparsable_id_in_the_probe_set_is_a_reconciliation_problem() -> None:
    probe_set = ProbeSet(natural=(), synthetic=(*_probe_set().synthetic, "not-a-probe-id"))
    problems = reconcile_probe_set_with_report(probe_set, _report())
    assert any("not-a-probe-id" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# The refusals leave nothing behind (the guard precedes its side effect)
# --------------------------------------------------------------------------- #
def test_a_reconciliation_failure_writes_no_file(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    with pytest.raises(Tier2NProbeError, match="disagree with the counts report"):
        write_probe_set(path, _probe_set(), _report(n_synthetic=999))
    assert not path.exists()


def test_a_sub_min_n_probe_set_is_refused_and_writes_no_file(tmp_path: Path) -> None:
    """A cheap run must not be able to mint a well-formed probe file."""
    small = [
        _variant(f"tier2n:{FAMILY_CLASS_II}:p{i:05d}", FAMILY_CLASS_II)
        for i in range(TIER2N_PROBE_MIN_N - 1)
    ]
    probe_set = build_probe_set(small)
    report = tier2n.build_report(small, seed=20260719)
    assert reconcile_probe_set_with_report(probe_set, report) == []  # consistent, just too small
    path = tmp_path / "ids.json"
    with pytest.raises(Tier2NProbeError, match="below the ADR-0005 A1 floor"):
        write_probe_set(path, probe_set, report)
    assert not path.exists()


def test_a_failed_write_leaves_the_previous_artifact_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write is atomic, so a partial file can never reach a mining round.

    A plain ``write_text`` truncates the destination first: an I/O failure partway
    through would replace a valid probe set with partial JSON, and the round that
    reads it would be the one to find out.
    """
    path = write_probe_set(tmp_path / "ids.json", _probe_set(), _report())
    good = path.read_bytes()

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("tbox_finder.eval.tier2n_probe.os.replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        write_probe_set(path, _probe_set(), _report())
    assert path.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir()] == ["ids.json"], "a temp file was left behind"


def test_exactly_min_n_is_accepted(tmp_path: Path) -> None:
    """The floor's positive control: the refusal must not fire one member too early."""
    at_floor = [
        _variant(f"tier2n:{FAMILY_CLASS_II}:p{i:05d}", FAMILY_CLASS_II)
        for i in range(TIER2N_PROBE_MIN_N)
    ]
    path = write_probe_set(
        tmp_path / "ids.json",
        build_probe_set(at_floor),
        tier2n.build_report(at_floor, seed=20260719),
    )
    assert load_probe_set(path).size == TIER2N_PROBE_MIN_N


# --------------------------------------------------------------------------- #
# The shipped CLI bytes
# --------------------------------------------------------------------------- #
def test_the_build_subcommand_defaults_the_ids_path_beside_the_report() -> None:
    """Pinned to the literal paths: comparing against the constants is a tautology."""
    args = tier2n._build_parser().parse_args(["build"])
    assert args.out == "reports/p2/tier2n_probe.json"
    assert args.probe_ids_out == "reports/p2/tier2n_probe_ids.json"
    assert Path(args.probe_ids_out).parent == Path(args.out).parent
    assert (str(REPORT_PATH), str(PROBE_IDS_PATH)) == (args.out, args.probe_ids_out)


def _stub_build(variants: list[Tier2NVariant]) -> Tier2NBuild:
    """A canned build result, so ``main`` itself runs without pandas or ``cmsearch``."""
    report = tier2n.build_report(variants, seed=20260719)
    report.update({"n_parents": 300, "cm": "RF00230.cm", "cm_sha256": "0" * 64})
    return Tier2NBuild(report=report, probe_set=build_probe_set(variants))


def _stub_run(monkeypatch: pytest.MonkeyPatch, variants: list[Tier2NVariant]) -> dict[str, Any]:
    """Replace the heavy build with a canned one so ``main`` itself is executed."""
    build = _stub_build(variants)
    monkeypatch.setattr(tier2n, "build_probe_run", lambda **_kwargs: build)
    return build.report


def test_main_writes_both_artifacts_and_the_ids_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_run(monkeypatch, _variants())
    ids_path = tmp_path / "ids.json"
    rc = tier2n.main(
        ["build", "--out", str(tmp_path / "report.json"), "--probe-ids-out", str(ids_path)]
    )
    assert rc == 0
    assert (tmp_path / "report.json").exists()
    assert load_probe_set(ids_path).size == N_CLASS_II + N_STEM_II_PK


def test_main_binds_the_ids_to_the_digest_of_the_report_it_just_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    _stub_run(monkeypatch, _variants())
    report_path = tmp_path / "report.json"
    ids_path = tmp_path / "ids.json"
    tier2n.main(["build", "--out", str(report_path), "--probe-ids-out", str(ids_path)])
    provenance = json.loads(ids_path.read_text(encoding="utf-8"))["provenance"]
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert provenance["source_report_sha256"] == digest
    assert provenance["seed"] == 20260719
    assert provenance["tier2n_probe_min_n"] == TIER2N_PROBE_MIN_N


def test_main_exits_nonzero_and_writes_no_ids_for_a_sub_min_n_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cheap-knob path, driven end-to-end through the shipped entry point."""
    small = [
        _variant(f"tier2n:{FAMILY_CLASS_II}:p{i:05d}", FAMILY_CLASS_II)
        for i in range(TIER2N_PROBE_MIN_N - 1)
    ]
    _stub_run(monkeypatch, small)
    ids_path = tmp_path / "ids.json"
    rc = tier2n.main(
        ["build", "--out", str(tmp_path / "report.json"), "--probe-ids-out", str(ids_path)]
    )
    assert rc == 1
    assert not ids_path.exists()
    assert (tmp_path / "report.json").exists()  # the counts report is still the record


def test_main_refuses_to_write_both_artifacts_to_one_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ids would destroy the report whose digest they record, and exit 0."""
    calls: list[int] = []
    monkeypatch.setattr(
        tier2n, "build_probe_run", lambda **_k: calls.append(1) or _stub_build(_variants())
    )
    both = tmp_path / "same.json"
    rc = tier2n.main(["build", "--out", str(both), "--probe-ids-out", str(both)])
    assert rc == 2
    assert not both.exists()
    assert calls == [], "the refusal must precede the ~14-minute build, not follow it"


def test_the_same_path_refusal_sees_through_a_relative_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comparing the raw strings would miss ``./x.json`` against ``x.json``."""
    monkeypatch.setattr(tier2n, "build_probe_run", lambda **_k: _stub_build(_variants()))
    monkeypatch.chdir(tmp_path)
    rc = tier2n.main(["build", "--out", "same.json", "--probe-ids-out", "./same.json"])
    assert rc == 2
    assert not (tmp_path / "same.json").exists()


def test_main_refuses_when_the_ids_and_the_counts_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variants = _variants()
    report = tier2n.build_report(variants, seed=20260719)
    report.update({"n_parents": 300, "cm": "RF00230.cm", "cm_sha256": "0" * 64})
    report["n_synthetic"] = 999
    monkeypatch.setattr(
        tier2n,
        "build_probe_run",
        lambda **_kwargs: Tier2NBuild(report=report, probe_set=build_probe_set(variants)),
    )
    ids_path = tmp_path / "ids.json"
    rc = tier2n.main(
        ["build", "--out", str(tmp_path / "report.json"), "--probe-ids-out", str(ids_path)]
    )
    assert rc == 1
    assert not ids_path.exists()


# --------------------------------------------------------------------------- #
# The delegation — one pipeline, not two
# --------------------------------------------------------------------------- #
def test_the_run_pairs_the_report_with_the_probe_set_of_the_same_variants() -> None:
    """The one line of the heavy build path the bare-CI tier can still reach."""
    variants = _variants()
    report = tier2n.build_report(variants, seed=20260719)
    build = tier2n.pair_report_with_probe_set(variants, report)
    assert build.report is report
    assert build.probe_set.synthetic == build_probe_set(variants).synthetic
    assert build.probe_set.size == N_CLASS_II + N_STEM_II_PK
    assert reconcile_probe_set_with_report(build.probe_set, build.report) == []


def test_build_probe_report_delegates_to_build_probe_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forked copy would let the counts and the ids drift run-to-run."""
    calls: list[dict[str, Any]] = []
    sentinel = {"marker": "the delegated report"}

    def _fake(**kwargs: Any) -> Tier2NBuild:
        calls.append(kwargs)
        return Tier2NBuild(report=sentinel, probe_set=ProbeSet(natural=(), synthetic=("x",)))

    monkeypatch.setattr(tier2n, "build_probe_run", _fake)
    assert tier2n.build_probe_report(n_parents=7, seed=11) is sentinel
    assert len(calls) == 1
    assert calls[0]["n_parents"] == 7 and calls[0]["seed"] == 11


# --------------------------------------------------------------------------- #
# The committed artifacts (additional to — never instead of — the above)
# --------------------------------------------------------------------------- #
def test_the_committed_probe_ids_load_and_reconcile_with_the_committed_counts() -> None:
    ids_path = REPO_ROOT / PROBE_IDS_PATH
    report_path = REPO_ROOT / REPORT_PATH
    probe_set = load_probe_set(ids_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert reconcile_probe_set_with_report(probe_set, report) == []
    assert probe_set.meets_min_n()
    assert probe_set.natural == ()
    assert len(set(probe_set.synthetic)) == len(probe_set.synthetic)


def test_the_probe_exclusion_will_be_zero_for_a_structural_reason_not_an_empty_set() -> None:
    """Measures the disjointness the round report has to state, instead of asserting it.

    Probe ids are ``tier2n:<FAMILY>:<record>`` synthetic-variant names; the P3
    re-mining substrate is the committed 941-row FP manifest, whose
    ``candidate_id``s are genomic coordinates. ``exclude_probe_members`` therefore
    removes **0** — but for a namespace reason, not because the probe set was
    empty, and those two zeros must never be reported as the same finding. If a
    future substrate ever shares the namespace this test goes red, which is the
    point at which the round's exclusion stops being vacuous.
    """
    manifest = json.loads(
        (REPO_ROOT / "data/processed/mining/round0_fp_spare_rule_inputs_v0.json").read_text(
            encoding="utf-8"
        )
    )
    substrate = {row["candidate_id"] for row in manifest["rows"]}
    probe_set = load_probe_set(REPO_ROOT / PROBE_IDS_PATH)
    assert substrate, "the FP manifest is empty, so the disjointness below is vacuous"
    assert probe_member_ids(probe_set) & substrate == set()


def test_the_committed_probe_ids_name_the_committed_counts_bytes() -> None:
    """States exactly what the digest binding proves — and what it does not.

    It proves *which report bytes this id list was written against*. It does **not**
    prove the members came from that run: the counts report carries no id list, so the
    reconciliation above is over cardinality and the per-family split only. Rewriting
    the 45 members while preserving the 21/24 split would satisfy both. That gap is
    closed by the byte pin below, not by this test.
    """
    payload = json.loads((REPO_ROOT / PROBE_IDS_PATH).read_text(encoding="utf-8"))
    report_bytes = (REPO_ROOT / REPORT_PATH).read_bytes()
    provenance = payload["provenance"]
    assert provenance["source_report"] == str(REPORT_PATH)
    assert provenance["source_report_sha256"] == hashlib.sha256(report_bytes).hexdigest()


def test_the_committed_probe_ids_are_pinned_by_digest() -> None:
    """A golden byte pin on the artifact, in the repo's ``expected.sha256`` idiom.

    The members are only ever compared to the counts report by *cardinality*, so an
    edit that swaps all 45 ids for fabricated ones with the same 21/24 split passes
    every other check here. This pin is what makes such an edit red. Regenerating the
    probe set legitimately therefore requires updating this constant in the same
    commit — deliberately, exactly as a golden fixture does.
    """
    digest = hashlib.sha256((REPO_ROOT / PROBE_IDS_PATH).read_bytes()).hexdigest()
    assert digest == COMMITTED_PROBE_IDS_SHA256
