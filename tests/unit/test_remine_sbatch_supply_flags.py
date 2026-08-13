"""Unit tier — the produced (b)/(c) evidence must have a PATH INTO the P3 re-mining round.

P3-15′-c-ii, -d, -e and -f each built one leg of that path and each ended with a table on
cluster scratch. Nothing then read those tables, and the two reasons were invisible from
either end:

1. **`slurm/p3/stage1_remine.sbatch` never passed `--relaxed-arch-status` /
   `--synteny-status`.** It composed `--relaxed-arch-available` into ``$PLAN_FLAGS`` and
   reused that string verbatim at leg (1), where ``apply_remine_spare_rule`` *requires* the
   paired table. A round declaring either backend died with a ``RemineError`` — after the
   queue wait, and only there.

2. **The plan leg exited 2 the moment the round became runnable.** ``_cmd_plan`` writes a
   report with ``round_report=None`` by design, and the shared clause set said *"a round
   that may run carries no mining outcome"*. While the backends were missing, ``may_run``
   was False and the clause never ran; the first green plan hit it, and the sbatch reads any
   non-zero rc as a refusal. Measured, not reasoned: ``plan`` with the four supplies and
   ``--rscape-installed`` returned **2** before this step and **0** after.

Both are the same shape — a seam that is unreachable until the thing it gates starts
working, so no earlier run could have exercised it. What is guarded here is therefore the
*composition itself*: the tokens are lifted from the shipped sbatch and **executed**, then
fed to the shipped parser, then through the shipped rule. A retyped equivalent would prove
only that this file agrees with itself ([[verify-the-line-you-ship]]).

Bare-CI tier: pure stdlib + a ``bash`` subprocess.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest

from tbox_finder.eval.tier2n_probe import ProbeSet
from tbox_finder.masking import LocusIndex
from tbox_finder.mining import mine_round as mine_round_module
from tbox_finder.mining.architecture_producer import (
    DEFAULT_STATUS_TABLE as ARCH_STATUS_FILENAME,
)
from tbox_finder.mining.mine_round import refuse_status_tables_that_decide_nothing
from tbox_finder.mining.remine import (
    LEG_PLAN,
    LEG_ROUND,
    REMINE_LEGS,
    RemineError,
    apply_remine_spare_rule,
    build_parser,
    build_remine_report,
    remine_problems,
)
from tbox_finder.mining.remine import main as remine_main
from tbox_finder.mining.spare_rule import STATUS_FAILED, STATUS_PASSED
from tbox_finder.mining.synteny_producer import (
    DEFAULT_STATUS_TABLE as SYNTENY_STATUS_FILENAME,
)

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "slurm"
REMINE_SBATCH = SLURM / "p3" / "stage1_remine.sbatch"

#: The two shipped sbatch files that invoke an ``apply-spare-rule`` leg. Enumerated so the
#: sweep below cannot go vacuous by matching nothing, and so a THIRD round script has to
#: come here and declare which side of the pairing rule it is on
#: ([[fixed-one-of-two-identical-things]]).
APPLY_SPARE_RULE_SBATCH = {
    "slurm/p3/stage1_remine.sbatch",
    "slurm/p2/mine_round_retrain.sbatch",
}

THRESHOLD = 0.9
EMPTY_MASK = LocusIndex.from_records([])


def _read(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(REPO)} is missing"
    return path.read_text(encoding="utf-8")


def _slice(text: str, start: str, end: str) -> str:
    """The shipped lines from the one starting ``start`` to the one starting ``end``.

    ``end`` is exclusive. Both markers must match exactly once — a marker that matched
    zero times would silently yield an empty fragment, and every assertion made about
    that fragment would pass on nothing.
    """
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(start)]
    ends = [i for i, ln in enumerate(lines) if ln.startswith(end)]
    assert len(starts) == 1, f"marker {start!r} matched {len(starts)} lines, expected 1"
    assert len(ends) == 1, f"marker {end!r} matched {len(ends)} lines, expected 1"
    assert starts[0] < ends[0], f"{start!r} does not precede {end!r}"
    return "\n".join(lines[starts[0] : ends[0]])


def _declaration_fragment() -> str:
    """The sbatch's own status-path defaults + its whole declaration/pairing block.

    Two slices, both verbatim: the ``*_STATUS=`` defaults (which name the producers'
    canonical filenames) and the block that turns this job's environment into CLI tokens.
    """
    text = _read(REMINE_SBATCH)
    defaults = _slice(text, "RELAXED_ARCH_STATUS=", "PROBE_SET=")
    block = _slice(text, "# ── The four supply declarations", "# ── LEG (0)")
    return defaults + "\n" + block


#: Read the two composed arrays out of the fragment. ``${ARR[@]+...}`` so an EMPTY array
#: prints nothing at all — a bare ``printf 'X|%s\n' "${ARR[@]}"`` runs its format once with
#: no argument and would report a phantom empty token.
_PROBE = """
printf 'NPLAN|%s\\n' "${#PLAN_FLAGS[@]}"
for _t in ${PLAN_FLAGS[@]+"${PLAN_FLAGS[@]}"}; do printf 'PLAN|%s\\n' "$_t"; done
printf 'NSTATUS|%s\\n' "${#STATUS_FLAGS[@]}"
for _t in ${STATUS_FLAGS[@]+"${STATUS_FLAGS[@]}"}; do printf 'STATUS|%s\\n' "$_t"; done
"""


def _compose(round_dir: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """Run the shipped fragment under ``set -eo pipefail`` — the sbatch's own options."""
    script = "set -eo pipefail\n" + _declaration_fragment() + _PROBE
    return subprocess.run(
        ["bash", "-c", script],
        env={"PATH": os.environ["PATH"], "ROUND_DIR": str(round_dir), **env},
        capture_output=True,
        text=True,
    )


def _tokens(proc: subprocess.CompletedProcess[str]) -> tuple[list[str], list[str]]:
    assert proc.returncode == 0, f"fragment failed rc={proc.returncode}: {proc.stderr}"
    plan = [ln[5:] for ln in proc.stdout.splitlines() if ln.startswith("PLAN|")]
    status = [ln[7:] for ln in proc.stdout.splitlines() if ln.startswith("STATUS|")]
    # The counts the fragment printed itself, so a token lost to word-splitting is caught
    # rather than absorbed into a shorter list.
    n_plan = [ln[6:] for ln in proc.stdout.splitlines() if ln.startswith("NPLAN|")]
    n_status = [ln[8:] for ln in proc.stdout.splitlines() if ln.startswith("NSTATUS|")]
    assert n_plan == [str(len(plan))] and n_status == [str(len(status))]
    return plan, status


def _all_available() -> dict[str, str]:
    return {
        "RSCAPE_INSTALLED": "1",
        "MSA_SUPPLY_AVAILABLE": "1",
        "STAGE2_SUPPLY_AVAILABLE": "1",
        "RELAXED_ARCH_AVAILABLE": "1",
        "SYNTENY_AVAILABLE": "1",
    }


def _status_table(path: Path, statuses: dict[str, str]) -> Path:
    """A producer status table in the shape both ``load_status_map`` readers demand."""
    path.write_text(
        json.dumps(
            {
                "status": dict(statuses),
                "rows": [{"candidate_id": k, "status": v} for k, v in statuses.items()],
            }
        ),
        encoding="utf-8",
    )
    return path


# ═════════════════════════════════════════════════════════════════════════════
# 1. The tokens the shipped sbatch composes
# ═════════════════════════════════════════════════════════════════════════════
def test_the_declared_backends_come_with_their_produced_tables(tmp_path: Path) -> None:
    """The defect itself: declare (b)/(c) and the round must be HANDED (b)/(c).

    Asserted on the composed tokens, not on the file's text, because the flag names are
    built at run time from the ``declare_supply`` stem — a substring search for
    ``--relaxed-arch-available`` finds nothing in the shipped file and would pass on a
    script that composes no flags at all.
    """
    arch = _status_table(tmp_path / ARCH_STATUS_FILENAME, {"a": STATUS_PASSED})
    syn = _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"a": STATUS_FAILED})
    plan, status = _tokens(_compose(tmp_path, **_all_available()))

    assert "--relaxed-arch-available" in plan
    assert "--synteny-available" in plan
    # Paired, in order, at the producers' own canonical filenames under $ROUND_DIR.
    assert status == [
        "--relaxed-arch-status",
        str(arch),
        "--synteny-status",
        str(syn),
    ]


def test_an_undeclared_backend_is_handed_no_table(tmp_path: Path) -> None:
    """The other half of the same guard — and the half a one-sided fix would miss.

    ``apply_remine_spare_rule`` refuses a table for a backend the round did not declare,
    because that is produced evidence the round said it did not have. So the pairing has
    to be an equivalence, not an implication.
    """
    _status_table(tmp_path / ARCH_STATUS_FILENAME, {"a": STATUS_PASSED})
    _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"a": STATUS_FAILED})
    env = {**_all_available(), "RELAXED_ARCH_AVAILABLE": "0"}
    plan, status = _tokens(_compose(tmp_path, **env))

    assert "--no-relaxed-arch-available" in plan and "--relaxed-arch-available" not in plan
    assert "--relaxed-arch-status" not in status
    # ...and (c), still declared, still carries its table: the two are independent.
    assert status == ["--synteny-status", str(tmp_path / SYNTENY_STATUS_FILENAME)]


def test_an_unset_env_var_declares_the_supply_UNAVAILABLE(tmp_path: Path) -> None:
    """The fail-open direction this step closed, stated as the thing it must never do.

    Every one of the four CLI flags is a two-way pair defaulting to its MODULE constant,
    and all four constants are ``True``. So *omitting* the flag meant AVAILABLE, and the
    block that emitted only the positive half made an unset (or explicitly ``0``) env var
    declare the supply available anyway. The negative half is what makes this job's own
    environment decide.
    """
    plan, status = _tokens(_compose(tmp_path))  # nothing exported at all
    assert plan == [
        "--no-msa-supply-available",
        "--no-stage2-supply-available",
        "--no-relaxed-arch-available",
        "--no-synteny-available",
    ]
    assert status == []
    # And the parser resolves them to False — the flags are not merely spelled, they bite.
    args = build_parser().parse_args(
        ["plan", "--stage2-threshold", "0.9", *plan, "--out", "x.json"]
    )
    assert args.msa_supply_available is False
    assert args.stage2_supply_available is False
    assert args.relaxed_arch_available is False
    assert args.synteny_available is False


def test_a_staged_table_for_an_undeclared_backend_is_reported(tmp_path: Path) -> None:
    """The reverse mis-staging, which is silent by construction and cost-free to say.

    An undeclared backend legitimately passes no table, so this is not a refusal — a round
    that deliberately switches a disjunct off is supported. But a table sitting in
    ``$ROUND_DIR`` means produced evidence the round will never read, and the consequence
    is invisible in the outcome: the disjunct resolves ``unavailable`` for every candidate,
    all are spared, and ``n_mined: 0`` is published with nothing to distinguish it from an
    honest zero. The job log is the only place that can say so, so it does.
    """
    staged = _status_table(tmp_path / ARCH_STATUS_FILENAME, {"a": STATUS_PASSED})
    _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"a": STATUS_FAILED})
    env = {**_all_available(), "RELAXED_ARCH_AVAILABLE": "0"}
    proc = _compose(tmp_path, **env)
    plan, status = _tokens(proc)

    assert "WARNING" in proc.stderr and str(staged) in proc.stderr
    # It is a warning, not a refusal, and it changes nothing about what is composed.
    assert proc.returncode == 0
    assert "--no-relaxed-arch-available" in plan
    assert "--relaxed-arch-status" not in status
    # ...and it stays quiet when there is nothing staged, so it cannot become noise every
    # operator learns to ignore.
    staged.unlink()
    quiet = _compose(tmp_path, **env)
    _tokens(quiet)
    assert "WARNING" not in quiet.stderr


def test_a_declared_backend_with_no_table_refuses_before_the_preflight(tmp_path: Path) -> None:
    """A mis-staged supply must cost nothing — not a queue wait, and not GPU legs.

    The refusal is the sbatch's own, executed: exit 3, naming the missing path and the
    producer that writes it.
    """
    _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"a": STATUS_FAILED})  # (c) staged, (b) not
    proc = _compose(tmp_path, **_all_available())
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "architecture_status.json" in proc.stderr
    assert "--relaxed-arch-status" in proc.stderr


def test_the_composed_tokens_are_the_ones_the_shipped_parser_accepts(tmp_path: Path) -> None:
    """The sbatch's bytes, through the real ``build_parser`` — no retyped token list.

    ``tests/unit/test_mining_spare_rule.py`` already parses a HAND-WRITTEN copy of these
    spellings; that catches a renamed flag only if somebody remembers to retype it here
    too. These tokens come out of the file the cluster runs.
    """
    _status_table(tmp_path / ARCH_STATUS_FILENAME, {"a": STATUS_PASSED})
    _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"a": STATUS_FAILED})
    plan, status = _tokens(_compose(tmp_path, **_all_available()))

    args = build_parser().parse_args(
        [
            "apply-spare-rule",
            "--stage2-threshold",
            str(THRESHOLD),
            "--manifest",
            "m.json",
            "--status-table",
            "s.json",
            "--posteriors",
            "p.json",
            "--probe-set",
            "probe.json",
            *plan,
            *status,
            "--out",
            "o.json",
        ]
    )
    assert args.relaxed_arch_available is True and args.synteny_available is True
    assert args.relaxed_arch_status == str(tmp_path / ARCH_STATUS_FILENAME)
    assert args.synteny_status == str(tmp_path / SYNTENY_STATUS_FILENAME)
    # The plan leg takes the declarations but no tables — the same tokens must parse there.
    build_parser().parse_args(["plan", "--stage2-threshold", "0.9", *plan, "--out", "o.json"])


def _argv_of_leg(tmp_path: Path, start: str, end: str, **env: str) -> list[str]:
    """The argv the shipped leg hands the CLI, captured from a stub ``python`` on PATH.

    Composing the right tokens and then not passing them are two different bugs, and a
    test that reads only the composition cannot see the second — it would stay green with
    ``"${STATUS_FLAGS[@]}"`` deleted from the invocation, which is the very shape of the
    defect this file exists for ([[artifact-pinning-test-cannot-see-the-code]]). So the
    invocation is lifted from the file too, and run.
    """
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "python"
    stub.write_text('#!/bin/bash\nfor a in "$@"; do printf "ARGV|%s\\n" "$a"; done\n')
    stub.chmod(0o755)
    script = (
        "set -eo pipefail\n"
        + _declaration_fragment()
        + "\n"
        + _slice(_read(REMINE_SBATCH), start, end)
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "ROUND_DIR": str(tmp_path),
            "STAGE2_THRESHOLD": str(THRESHOLD),
            "FP_MANIFEST": str(tmp_path / "fps.json"),
            "PLAN_REPORT": str(tmp_path / "plan.json"),
            "ROUND_REPORT": str(tmp_path / "round.json"),
            "STATUS_TABLE": str(tmp_path / "covariation_status.json"),
            "POSTERIORS": str(tmp_path / "post.json"),
            "PROBE_SET": str(tmp_path / "probe.json"),
            **env,
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}: {proc.stdout}{proc.stderr}"
    return [ln[5:] for ln in proc.stdout.splitlines() if ln.startswith("ARGV|")]


def test_the_apply_leg_hands_the_cli_both_composed_arrays(tmp_path: Path) -> None:
    """The declarations AND the tables reach the process the cluster starts.

    ``$PLAN_FLAGS`` alone was what shipped: correct declarations, no tables, and a
    ``RemineError`` at the far end of a queue wait.
    """
    arch = _status_table(tmp_path / ARCH_STATUS_FILENAME, {"a": STATUS_PASSED})
    syn = _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"a": STATUS_FAILED})
    argv = _argv_of_leg(
        tmp_path,
        "PYTHONPATH=src python -m tbox_finder.mining.remine apply-spare-rule",
        "N_MINED=",
        **_all_available(),
    )
    assert argv[:2] == ["-m", "tbox_finder.mining.remine"]
    assert "apply-spare-rule" in argv
    for token in (
        "--rscape-installed",
        "--msa-supply-available",
        "--stage2-supply-available",
        "--relaxed-arch-available",
        "--synteny-available",
    ):
        assert token in argv, f"{token} never reached the CLI"
    # Paths adjacent to their own flag — an argv carrying both flags but the tables
    # transposed would satisfy a membership-only assertion.
    assert argv[argv.index("--relaxed-arch-status") + 1] == str(arch)
    assert argv[argv.index("--synteny-status") + 1] == str(syn)
    # ⚠ MEMBERSHIP IS NOT ENOUGH, and adversarial review found this exact gap: capturing
    # argv and only searching it leaves several ways to break the shipped invocation
    # invisible — a missing required argument, a subcommand typo, a value that lands on the
    # wrong option. The captured argv is therefore RE-PARSED by the CLI it was composed for,
    # and the RESOLVED namespace is asserted, not the token list.
    args = build_parser().parse_args(argv[2:])
    assert args.cmd == "apply-spare-rule"
    assert args.stage2_threshold == pytest.approx(THRESHOLD)
    assert (args.relaxed_arch_available, args.synteny_available) == (True, True)
    assert (args.msa_supply_available, args.stage2_supply_available) == (True, True)
    assert args.rscape_installed is True
    assert args.relaxed_arch_status == str(arch)
    assert args.synteny_status == str(syn)
    # Every required argument the leg is supposed to hand over, resolved — a dropped
    # `--probe-set` or `--posteriors` would exit argparse on the cluster, after the sync.
    assert args.manifest and args.status_table and args.posteriors and args.probe_set
    assert args.out.endswith("round.json")


def test_the_plan_leg_hands_the_cli_the_declarations(tmp_path: Path) -> None:
    """Leg (0) declares the same four supplies leg (1) will mine under.

    Two legs, one ``$PLAN_FLAGS``: a preflight that judged a different availability set
    from the one the mining leg runs with would be a preflight of nothing.
    """
    _status_table(tmp_path / ARCH_STATUS_FILENAME, {"a": STATUS_PASSED})
    _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"a": STATUS_FAILED})
    argv = _argv_of_leg(
        tmp_path,
        "PYTHONPATH=src python -m tbox_finder.mining.remine plan",
        "PLAN_RC=",
        **_all_available(),
    )
    assert "plan" in argv and "--relaxed-arch-available" in argv
    assert "--synteny-available" in argv and "--rscape-installed" in argv
    # The plan subcommand takes no status tables; passing one would be an argparse error
    # on the cluster, after the sync.
    assert "--relaxed-arch-status" not in argv and "--synteny-status" not in argv
    # Re-parsed, not merely searched (see the apply-leg test for why membership is weak).
    args = build_parser().parse_args(argv[2:])
    assert args.cmd == "plan"
    assert args.stage2_threshold == pytest.approx(THRESHOLD)
    assert (args.relaxed_arch_available, args.synteny_available) == (True, True)
    assert (args.msa_supply_available, args.stage2_supply_available) == (True, True)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Those tokens, through the shipped rule — the evidence must DECIDE something
# ═════════════════════════════════════════════════════════════════════════════
def test_the_produced_b_and_c_tables_reach_the_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End of the path: the sbatch's own tokens, the shipped rule, a decided candidate.

    Only the union-prior mask is substituted (it needs the DVC-tracked corpus parquet).
    The fixture is asymmetric on purpose — ``spared-by-b`` is the only candidate whose
    (b) status is ``passed``, so its spare REASON names ``relaxed_architecture``. A
    count-only assertion would be satisfied by a round that never read the (b) table at
    all, which is precisely the state this step found.
    """
    arch = _status_table(
        tmp_path / ARCH_STATUS_FILENAME,
        {"spared-by-b": STATUS_PASSED, "spared-by-c": STATUS_FAILED},
    )
    syn = _status_table(
        tmp_path / SYNTENY_STATUS_FILENAME,
        {"spared-by-b": STATUS_FAILED, "spared-by-c": STATUS_PASSED},
    )
    _, status = _tokens(_compose(tmp_path, **_all_available()))
    assert status == ["--relaxed-arch-status", str(arch), "--synteny-status", str(syn)]

    manifest = tmp_path / "fps.json"
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": cid,
                        "accession": "GCA_1:c0",
                        "locus_start": start,
                        "locus_end": start + 50,
                        "score": 0.9,
                        "pool": "genomic_window",
                    }
                    for cid, start in (("spared-by-b", 100), ("spared-by-c", 300))
                ]
            }
        ),
        encoding="utf-8",
    )
    covariation = tmp_path / "covariation_status.json"
    covariation.write_text(
        json.dumps({"status": {"spared-by-b": STATUS_FAILED, "spared-by-c": STATUS_FAILED}}),
        encoding="utf-8",
    )
    posteriors = tmp_path / "post.json"
    posteriors.write_text(
        json.dumps({"posteriors": {"spared-by-b": 0.01, "spared-by-c": 0.01}}), encoding="utf-8"
    )
    monkeypatch.setattr(mine_round_module, "load_union_mask", lambda **_kw: EMPTY_MASK)

    report = apply_remine_spare_rule(
        manifest,
        covariation,
        posteriors,
        stage2_threshold=THRESHOLD,
        rscape_installed=True,
        msa_supply_available=True,
        stage2_supply_available=True,
        relaxed_arch_available=True,
        synteny_available=True,
        # The two paths come out of the fragment the cluster runs, positionally paired
        # with their own flags — not retyped from the fixture above.
        relaxed_arch_status_table=status[status.index("--relaxed-arch-status") + 1],
        synteny_status_table=status[status.index("--synteny-status") + 1],
        probe_set=ProbeSet(natural=("not-in-substrate",), synthetic=()),
    )
    # Each candidate spared by the DIFFERENT disjunct — so a round that read one table for
    # both, or neither, cannot produce this pair.
    assert report["reasons"]["spared-by-b"] == "passed:relaxed_architecture"
    assert report["reasons"]["spared-by-c"] == "passed:downstream_aaRS_synteny"
    assert report["n_mined"] == 0


def test_the_pairing_guard_is_what_the_sbatch_would_have_hit(tmp_path: Path) -> None:
    """The RemineError a declared-(b) round used to die on, reproduced.

    This is the failure the shipped sbatch produced for every round that declared the
    backend — kept as an executable record of *why* the ``--*-status`` arguments exist,
    so deleting them from the sbatch cannot look like a cleanup.
    """
    with pytest.raises(RemineError, match="no relaxed-architecture status table"):
        apply_remine_spare_rule(
            tmp_path / "absent.json",
            tmp_path / "absent.json",
            tmp_path / "absent.json",
            stage2_threshold=THRESHOLD,
            rscape_installed=True,
            msa_supply_available=True,
            stage2_supply_available=True,
            relaxed_arch_available=True,
            probe_set=ProbeSet(natural=("x",), synthetic=()),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. The plan leg must exit 0 once the round can actually run
# ═════════════════════════════════════════════════════════════════════════════
def test_a_green_plan_leg_exits_zero(tmp_path: Path) -> None:
    """Measured regression: this returned **2** before P3-15′-h.

    ``_cmd_plan`` writes ``round: null`` by design, and the shared clause set demanded a
    mining outcome from any report whose ``may_run`` was True. Nothing caught it because
    ``may_run`` could not BE True until the (b)/(c) backends landed — the clause and the
    condition that reaches it shipped years apart in step terms. The sbatch treats every
    non-zero rc as a refusal, so this made the round unrunnable at leg (0).
    """
    out = tmp_path / "plan.json"
    rc = remine_main(
        [
            "plan",
            "--stage2-threshold",
            str(THRESHOLD),
            "--rscape-installed",
            "--msa-supply-available",
            "--stage2-supply-available",
            "--relaxed-arch-available",
            "--synteny-available",
            "--out",
            str(out),
        ]
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert (rc, written["may_run"], written["problems"]) == (0, True, [])
    assert written["leg"] == LEG_PLAN
    assert written["round"] is None


def test_the_plan_leg_declaration_is_falsifiable(tmp_path: Path) -> None:
    """The excusing clause must not be a blanket amnesty.

    ``leg`` is a caller DECLARATION, so the clause set checks it against the payload in
    both directions. Deriving it from ``round is None`` instead would make the plan branch
    read its own conclusion — a clause that can only catch a value flipped false
    ([[gate-clauses-need-re-derivation]]).
    """
    plan = {
        "readiness": {"ready": True},
        "yield": {"yield_producible": True, "blocking_disjuncts": []},
        "may_run": True,
    }
    outcome = {
        "n_probe_members_considered": 1,
        "excluded_probe_member_ids": [],
        "n_excluded_probe_members": 0,
    }
    # A plan-leg report that carries a mining outcome is claiming to have mined from the
    # leg that spends no GPU time.
    lying_plan = build_remine_report(
        plan=plan,
        round_report=outcome,
        probe_trace=None,
        stage2_threshold=THRESHOLD,
        leg=LEG_PLAN,
    )
    assert any("carries a mining outcome" in p for p in remine_problems(lying_plan))
    # ...and the round leg still owes one. Same payload, other declaration.
    silent_round = build_remine_report(
        plan=plan,
        round_report=None,
        probe_trace=None,
        stage2_threshold=THRESHOLD,
        leg=LEG_ROUND,
    )
    assert any("carries no mining outcome" in p for p in remine_problems(silent_round))
    # An unknown leg is refused rather than defaulted: neither branch could check it.
    unknown = {**silent_round, "leg": "whatever", "round": outcome}
    assert any("is not one of" in p for p in remine_problems(unknown))
    assert LEG_PLAN in REMINE_LEGS and LEG_ROUND in REMINE_LEGS
    # ...and the DEFAULT is the strict half, read off the live signature. A caller that
    # forgets to declare its leg must inherit the clause that DEMANDS an outcome, not the
    # one that excuses its absence — defaulting to LEG_PLAN would hand every forgetful
    # caller a blanket amnesty, and no other assertion here would notice.
    assert inspect.signature(build_remine_report).parameters["leg"].default == LEG_ROUND


def test_the_round_leg_stamps_itself_as_the_round_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OTHER declaration — the one adversarial review found untested.

    ``_cmd_apply_spare_rule`` declares ``leg=LEG_ROUND`` at two call sites, and flipping
    both to ``LEG_PLAN`` was invisible to every other test here: the plan-leg tests pin
    the plan side, and the falsifiability test builds its reports by hand. A round leg
    that called itself ``plan`` would take the excusing branch and stop owing a mining
    outcome at all — the amnesty this step was careful not to hand out
    ([[fixed-one-of-two-identical-things]], on the two halves of one declaration).
    """
    arch = _status_table(tmp_path / ARCH_STATUS_FILENAME, {"c1": STATUS_FAILED})
    syn = _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"c1": STATUS_FAILED})
    manifest = tmp_path / "fps.json"
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "accession": "GCA_1:c0",
                        "locus_start": 100,
                        "locus_end": 150,
                        "score": 0.9,
                        "pool": "genomic_window",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    covariation = tmp_path / "covariation_status.json"
    covariation.write_text(json.dumps({"status": {"c1": STATUS_FAILED}}), encoding="utf-8")
    posteriors = tmp_path / "post.json"
    posteriors.write_text(json.dumps({"posteriors": {"c1": 0.01}}), encoding="utf-8")
    probe = tmp_path / "probe.json"
    probe.write_text(
        json.dumps({"natural": ["not-in-substrate"], "synthetic": []}), encoding="utf-8"
    )
    monkeypatch.setattr(mine_round_module, "load_union_mask", lambda **_kw: EMPTY_MASK)
    out = tmp_path / "round.json"

    rc = remine_main(
        [
            "apply-spare-rule",
            "--stage2-threshold",
            str(THRESHOLD),
            "--manifest",
            str(manifest),
            "--status-table",
            str(covariation),
            "--posteriors",
            str(posteriors),
            "--probe-set",
            str(probe),
            "--rscape-installed",
            "--msa-supply-available",
            "--stage2-supply-available",
            "--relaxed-arch-available",
            "--relaxed-arch-status",
            str(arch),
            "--synteny-available",
            "--synteny-status",
            str(syn),
            "--out",
            str(out),
        ]
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert (rc, written["leg"], written["problems"]) == (0, LEG_ROUND, [])
    # Every disjunct ran and failed, so this candidate is MINED — the round is not a
    # structural zero any more, which is the whole point of the arc this step closes.
    assert written["round"]["n_mined"] == 1


def test_a_table_that_decides_none_of_these_candidates_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One step past the pairing guard, and the state adversarial review demonstrated.

    A table that is present and well-formed but decides **zero** of this round's
    candidates — an empty ``{"status": {}, "rows": []}`` (the shipped per-shard
    ``build_status_table([])`` writes exactly that, 379 valid bytes), or a table from a
    different round — passes `[ -s ]`, passes `load_status_map`, and then resolves
    ``unavailable`` for every candidate. All are spared, ``n_mined`` is 0, the leg exits
    0 and the sbatch writes its DONE marker: a zero-yield round indistinguishable from an
    honest one. A PARTIAL overlap is left alone — 663 of 941 candidates legitimately
    resolved ``unavailable`` for criterion (b) at job 1288.
    """
    manifest = tmp_path / "fps.json"
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "accession": "GCA_1:c0",
                        "locus_start": 100,
                        "locus_end": 150,
                        "score": 0.9,
                        "pool": "genomic_window",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    covariation = tmp_path / "covariation_status.json"
    covariation.write_text(json.dumps({"status": {"c1": STATUS_FAILED}}), encoding="utf-8")
    posteriors = tmp_path / "post.json"
    posteriors.write_text(json.dumps({"posteriors": {"c1": 0.01}}), encoding="utf-8")
    monkeypatch.setattr(mine_round_module, "load_union_mask", lambda **_kw: EMPTY_MASK)

    # (c) is held FIXED and healthy across every arm, so the only thing varying is the (b)
    # table. Without it the control could not mine for a reason that has nothing to do with
    # the guard under test — an undeclared (c) makes every candidate spared anyway.
    syn = _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"c1": STATUS_FAILED})

    def _round(arch_rows: dict[str, str]) -> dict[str, object]:
        return apply_remine_spare_rule(
            manifest,
            covariation,
            posteriors,
            stage2_threshold=THRESHOLD,
            rscape_installed=True,
            msa_supply_available=True,
            stage2_supply_available=True,
            relaxed_arch_available=True,
            relaxed_arch_status_table=_status_table(tmp_path / "arch.json", arch_rows),
            synteny_available=True,
            synteny_status_table=syn,
            probe_set=ProbeSet(natural=("not-in-substrate",), synthetic=()),
        )

    # Empty, and wrong-round — the same fault, and neither is a result.
    with pytest.raises(RemineError, match="none of them is one of this round's"):
        _round({})
    with pytest.raises(RemineError, match="none of them is one of this round's"):
        _round({"some-other-round-id": STATUS_FAILED})
    # ...and the honest partial case still runs and still spares nobody wrongly: the one
    # candidate IS decided, so no refusal fires. Without this half the guard could be a
    # blanket refusal and the two `raises` above would still pass
    # ([[raises-test-needs-a-positive-control]]).
    report = _round({"c1": STATUS_FAILED, "another-round-id": STATUS_PASSED})
    assert report["n_mined"] == 1

    # ⚠ AND EVERY OTHER DISJUNCT, because guarding only the two NEW backends is how this
    # defect survived its first fix: review found the covariation arm — same shape, same
    # consequence — still unguarded after (b) and (c) were closed. The Stage-2 posterior
    # map is the same again: empty ⇒ every posterior is None ⇒ `unavailable` ⇒ spared.
    empty_covariation = tmp_path / "empty_covariation.json"
    empty_covariation.write_text(json.dumps({"status": {}}), encoding="utf-8")
    with pytest.raises(RemineError, match="any_helix_rscape"):
        apply_remine_spare_rule(
            manifest,
            empty_covariation,
            posteriors,
            stage2_threshold=THRESHOLD,
            rscape_installed=True,
            msa_supply_available=True,
            stage2_supply_available=True,
            relaxed_arch_available=True,
            relaxed_arch_status_table=_status_table(tmp_path / "arch.json", {"c1": STATUS_FAILED}),
            synteny_available=True,
            synteny_status_table=syn,
            probe_set=ProbeSet(natural=("not-in-substrate",), synthetic=()),
        )
    empty_posteriors = tmp_path / "empty_post.json"
    empty_posteriors.write_text(json.dumps({"posteriors": {}}), encoding="utf-8")
    with pytest.raises(RemineError, match="high_stage2_posterior"):
        apply_remine_spare_rule(
            manifest,
            covariation,
            empty_posteriors,
            stage2_threshold=THRESHOLD,
            rscape_installed=True,
            msa_supply_available=True,
            stage2_supply_available=True,
            relaxed_arch_available=True,
            relaxed_arch_status_table=_status_table(tmp_path / "arch.json", {"c1": STATUS_FAILED}),
            synteny_available=True,
            synteny_status_table=syn,
            probe_set=ProbeSet(natural=("not-in-substrate",), synthetic=()),
        )


def test_the_p2_leg_guards_every_disjunct_it_declares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sibling call site, driven — not assumed to agree because it shares a function.

    ``mine_round.apply_spare_rule`` is the other leg carrying the identical guard pair, and
    "the same helper is imported there" is not evidence that it is *called* with the right
    map. Fixing one of two identical call sites is the class this project keeps re-finding,
    so the P2 leg is exercised here rather than reasoned about.
    """
    manifest = tmp_path / "p2_fps.json"
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "accession": "GCA_1:c0",
                        "locus_start": 100,
                        "locus_end": 150,
                        "score": 0.9,
                        "pool": "genomic_window",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "p2_empty_covariation.json"
    empty.write_text(json.dumps({"status": {}}), encoding="utf-8")
    monkeypatch.setattr(mine_round_module, "load_union_mask", lambda **_kw: EMPTY_MASK)

    with pytest.raises(mine_round_module.MineRoundError, match="any_helix_rscape"):
        mine_round_module.apply_spare_rule(
            manifest,
            empty,
            rscape_installed=True,
            msa_supply_available=True,
        )
    # Positive control: the same leg on a table that DOES decide this candidate runs.
    decided = tmp_path / "p2_covariation.json"
    decided.write_text(json.dumps({"status": {"c1": STATUS_FAILED}}), encoding="utf-8")
    report = mine_round_module.apply_spare_rule(
        manifest,
        decided,
        rscape_installed=True,
        msa_supply_available=True,
    )
    assert report["n_candidates"] == 1


def test_no_declared_disjunct_can_be_exempt_from_the_zero_overlap_refusal() -> None:
    """The anti-drift half: a fifth backend cannot inherit the hole the first four had.

    The sweep is over the availability map, and a disjunct present there with no evidence
    entry is refused rather than silently skipped. Without this, adding a backend would
    reproduce exactly the state review found — a guard that covers the disjuncts somebody
    remembered ([[fixed-one-of-two-identical-things]]).
    """
    with pytest.raises(RemineError, match="no evidence map was offered"):
        refuse_status_tables_that_decide_nothing(
            availability={"any_helix_rscape": True, "a_fifth_backend": True},
            evidence_maps={"any_helix_rscape": {"c1": STATUS_FAILED}},
            candidate_ids=["c1"],
            error=RemineError,
        )
    # The positive control: the same call with the entry present does NOT raise, so the
    # guard is not simply refusing everything handed to it.
    refuse_status_tables_that_decide_nothing(
        availability={"any_helix_rscape": True, "a_fifth_backend": True},
        evidence_maps={
            "any_helix_rscape": {"c1": STATUS_FAILED},
            "a_fifth_backend": {"c1": STATUS_FAILED},
        },
        candidate_ids=["c1"],
        error=RemineError,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. The class, not the one file
# ═════════════════════════════════════════════════════════════════════════════
def test_the_sbatch_status_paths_are_the_producers_own_filenames(tmp_path: Path) -> None:
    """Bind the sbatch's defaults to the producers' constants instead of retyping them.

    CodeRabbit's finding: the filenames are repeated in the sbatch and in this file, and
    nothing tied either to ``architecture_producer.DEFAULT_STATUS_TABLE`` /
    ``synteny_producer.DEFAULT_STATUS_TABLE``. A rename there leaves the sbatch pointing at
    a path the producer no longer writes, and the symptom is `exit 3` on a round whose
    supply WAS staged correctly.

    The tables are staged under the **constants'** names and the composed paths are read
    out of the shipped file, so this fails on a drift in either direction: rename the
    constant and the staged filename moves out from under the sbatch (`require_status`
    exits 3, `_tokens` raises); change the sbatch's literal and the composed path stops
    matching.
    """
    arch = _status_table(tmp_path / ARCH_STATUS_FILENAME, {"a": STATUS_PASSED})
    syn = _status_table(tmp_path / SYNTENY_STATUS_FILENAME, {"a": STATUS_FAILED})
    _, status = _tokens(_compose(tmp_path, **_all_available()))
    assert status == ["--relaxed-arch-status", str(arch), "--synteny-status", str(syn)]
    # The constants are what the producers' merge legs write, so a stale sbatch is a
    # staging failure rather than a science one — say which is which.
    assert arch.name == ARCH_STATUS_FILENAME and syn.name == SYNTENY_STATUS_FILENAME


def test_every_apply_spare_rule_sbatch_is_on_one_side_of_the_pairing_rule() -> None:
    """Any round script that CAN declare (b)/(c) must also compose their tables.

    The enumeration is explicit so the sweep cannot pass by matching nothing, and so a
    third round script has to come here and say which side it is on. ``mine_round_retrain``
    is the other member today: it declares neither backend, which is the P2 round's
    documented under-declaration, and the assertion below is what would notice if that
    changed without the tables following ([[fixed-one-of-two-identical-things]]).
    """
    found = {
        str(p.relative_to(REPO))
        for p in sorted(SLURM.rglob("*.sbatch"))
        if "apply-spare-rule" in _read(p)
    }
    assert found == APPLY_SPARE_RULE_SBATCH

    for rel in sorted(found):
        text = _read(REPO / rel)
        code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        for backend, table in (
            ("relaxed-arch-available", "--relaxed-arch-status"),
            ("synteny-available", "--synteny-status"),
        ):
            if backend in code:
                assert table in code, (
                    f"{rel} can declare {backend} but never composes {table}; the round "
                    "would raise RemineError, or spare every candidate on an "
                    "'unavailable' the operator believed was produced"
                )
