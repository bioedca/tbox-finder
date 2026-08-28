"""P3-17 review round 10 — the PRODUCERS behind the gate clauses, pinned at their call sites.

Round 10 found six guards whose only tests exercised the *helper* and never the *writer*, so
each could be reverted to a constant with the whole suite green. The pattern is the one this
repo keeps re-learning: a decision function can be exhaustively unit-tested and still never be
called ([[artifact-pinning-test-cannot-see-the-code]], [[pinned-constant-that-nothing-reads]]).

These are AST pins because the producers need `torch` and CI installs none — but an AST pin
must check *contents*, not shape ([[ast-pin-must-check-contents-not-shape]]): every assertion
below names the exact expression that must appear, so a shape-preserving substitution
(`resolved_backbone` -> `BR.PRODUCTION_BACKBONE`, `base_model is None` -> `True`) turns it red.

Each test carries the concrete sabotage that motivated it. Re-apply any of them and exactly
the named test must fail.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "tbox_finder"


def _tree(rel: str) -> ast.Module:
    return ast.parse((SRC / rel).read_text(encoding="utf-8"))


def _functions(tree: ast.Module, name: str) -> list[ast.FunctionDef]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name
    ]


def _calls(node: ast.AST, func_name: str) -> list[ast.Call]:
    out = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        target = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if target == func_name:
            out.append(n)
    return out


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _dict_value(node: ast.AST, key: str) -> list[ast.expr]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values, strict=False):
                if isinstance(k, ast.Constant) and k.value == key:
                    out.append(v)
    return out


# --------------------------------------------------------------------------------------- #
# 1. The score sidecar's `load.backbone` — root evidence of two gate clauses
# --------------------------------------------------------------------------------------- #
def test_the_sidecars_backbone_is_the_RESOLVED_one_not_a_constant() -> None:
    """SABOTAGE: `"backbone": resolved_backbone` -> `"backbone": BR.PRODUCTION_BACKBONE`.

    Every RNA-FM sidecar would then record `rinalmo-giga`, so `env_lock_for_scored_arms`
    returns the production lock and both `env_lock` clauses compare that lock against itself
    and pass — while the report names an environment that cannot load the scored weights.
    """
    tree = _tree("stage2/eval.py")
    values = _dict_value(tree, "backbone")
    assert values, "stage2/eval.py writes no `backbone` dict entry at all"
    # ⚠ Round 11: assert EVERY writer, not that at least one is right. The first version
    # required only that SOME dict said `resolved_backbone` and then merely rejected
    # `ast.Constant` — but `BR.PRODUCTION_BACKBONE` is an ast.Attribute, so a SECOND write
    # path stamping the module constant passed ([[fixed-one-of-two-identical-things]]).
    for value in values:
        assert isinstance(value, ast.Name) and value.id == "resolved_backbone", (
            f"the `backbone` dict entry at line {value.lineno} is not the resolver's "
            f"output: {ast.unparse(value)!r}"
        )


def test_the_resolver_is_actually_called_before_the_sidecar_is_written() -> None:
    """A name is only evidence if something assigned it from the decision function."""
    tree = _tree("stage2/eval.py")
    assigned = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "resolved_backbone" for t in n.targets)
    ]
    assert assigned, "`resolved_backbone` is never assigned"
    assert any(
        _calls(a.value, "resolve_checkpoint_backbone") for a in assigned
    ), "`resolved_backbone` is assigned from something other than `resolve_checkpoint_backbone`"


# --------------------------------------------------------------------------------------- #
# 2. The training report's own env lock
# --------------------------------------------------------------------------------------- #
def test_the_training_report_stamps_the_BACKBONES_lock_not_the_module_constant() -> None:
    """SABOTAGE: `"env_lock": env_lock_for(cfg.backbone)` -> `"env_lock": ENV_LOCK`.

    Both RNA-FM sweep reports would publish `envs/ml-rna.conda-lock.yml` and an
    `env_lock_sha256` digesting that wrong file, with `gate.overall_pass` still true.
    """
    tree = _tree("stage2/train.py")
    values = _dict_value(tree, "env_lock")
    assert values, "stage2/train.py writes no `env_lock` key at all"
    # ⚠ Round 11: EVERY `env_lock` value, not just the ones that happen to be calls. The
    # first version asserted `calls` was non-empty and then inspected only those, so a second
    # writer using the bare `ENV_LOCK` constant survived untouched.
    for value in values:
        assert isinstance(value, ast.Call), (
            f"the `env_lock` value at line {value.lineno} is not derived from the config — "
            f"the module constant is back: {ast.unparse(value)!r}"
        )
        assert ast.unparse(value) == "env_lock_for(cfg.backbone)", ast.unparse(value)
    # ...and the digest must hash THAT file, not the constant.
    digests = _dict_value(tree, "env_lock_sha256")
    assert digests, "no `env_lock_sha256` is written"
    for digest in digests:
        assert "env_lock_for(cfg.backbone)" in ast.unparse(digest), ast.unparse(digest)


# --------------------------------------------------------------------------------------- #
# 3 + 6. Sizing measures the backbone it was ASKED for, at EVERY call site
# --------------------------------------------------------------------------------------- #
def test_every_measure_batch_call_site_sizes_the_requested_backbone() -> None:
    """SABOTAGE (a): `backbone=backbone` -> `backbone=PRODUCTION_BACKBONE` in `measure_batch`.
    SABOTAGE (b): delete `backbone=backbone` from the off-comparison call at ~line 670.

    (a) makes `run_sizing --backbone rnafm` measure RiNALMo-giga's VRAM and step time while
    every field in the report says `rnafm`. (b) computes `saving_ratio` across two different
    models. The existing AST test walks these same calls and inspects only
    `gradient_checkpointing` ([[fixed-one-of-two-identical-things]]).
    """
    tree = _tree("stage2/sizing.py")
    # the config `measure_batch` builds must carry the parameter it was handed
    measure = _functions(tree, "measure_batch")
    assert len(measure) == 1, "measure_batch is defined more than once"
    configs = _calls(measure[0], "Stage2TrainConfig")
    assert configs, "measure_batch no longer builds a Stage2TrainConfig"
    for cfg in configs:
        value = _kwarg(cfg, "backbone")
        assert value is not None, "Stage2TrainConfig is built without a `backbone` kwarg"
        assert isinstance(value, ast.Name) and value.id == "backbone", (
            "measure_batch sizes a backbone other than the one it was asked for: "
            f"{ast.unparse(value)!r}"
        )
    # ...and every caller threads it through
    sites = _calls(tree, "measure_batch")
    assert (
        len(sites) >= 2
    ), f"expected the sweep and the off-comparison call sites, saw {len(sites)}"
    for site in sites:
        value = _kwarg(site, "backbone")
        assert value is not None, (
            f"the measure_batch call at line {site.lineno} passes no `backbone`, so it measures "
            "the parameter default while the report names the requested key"
        )
        assert (
            isinstance(value, ast.Name) and value.id == "backbone"
        ), f"line {site.lineno}: {ast.unparse(value)!r}"


# --------------------------------------------------------------------------------------- #
# 4. `loaded_from_registry` — the conjunct that stops a toy fixture certifying
# --------------------------------------------------------------------------------------- #
def test_loaded_from_registry_is_derived_from_the_injected_base_model() -> None:
    """SABOTAGE: `backbone_loaded_from_registry = base_model is None` -> `= True`.

    A 2-layer toy wrap would then record a full production identity block with
    `loaded_from_registry: true`, and `backbone_pinned` would grade TRUE for it. The two tests
    that cover this today both skip in CI (one needs CUDA_HOME + multimolecule, the other an
    env var CI does not arm), so nothing guarded the writer.
    """
    tree = _tree("train/lora_harness.py")
    assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "backbone_loaded_from_registry" for t in n.targets
        )
    ]
    assert assigns, "`backbone_loaded_from_registry` is never assigned"
    for assign in assigns:
        rendered = ast.unparse(assign.value)
        assert rendered == "base_model is None", (
            "the registry flag is no longer derived from whether a base model was injected: "
            f"{rendered!r}"
        )


# --------------------------------------------------------------------------------------- #
# 5. `gate2.main`'s sidecar-mismatch refusal
# --------------------------------------------------------------------------------------- #
def test_gate2_main_raises_on_a_sidecar_identity_mismatch() -> None:
    """SABOTAGE: `if mismatched:` -> `if False:`.

    `gate2 grade` would then join an in-distribution ECE from one checkpoint to a
    leave-clade-out ECE from a different adapter and publish them as one arm's grade.
    """
    tree = _tree("calib/gate2.py")
    main = _functions(tree, "main")
    assert len(main) == 1
    assigned = [
        n
        for n in ast.walk(main[0])
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "mismatched" for t in n.targets)
    ]
    assert assigned, "`main` no longer computes `mismatched`"
    assert any(
        _calls(a.value, "sidecar_identity_mismatches") for a in assigned
    ), "`mismatched` is computed from something other than `sidecar_identity_mismatches`"
    guards = [
        n
        for n in ast.walk(main[0])
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "mismatched"
    ]
    assert guards, "nothing in `main` branches on `mismatched`, so the refusal is unreachable"
    for guard in guards:
        assert any(
            isinstance(s, ast.Raise) for s in ast.walk(guard)
        ), f"the `if mismatched:` at line {guard.lineno} does not raise"


# --------------------------------------------------------------------------------------- #
# The negative control — it must exercise the REAL pins, not a copy of them.
#
# ⚠ Round 11 replaced the first version, which parsed a synthetic string and re-ran a
# hand-copied version of each predicate. That certified a copy: deleting the assertion from
# the pin above left the "control" green, so it read as evidence the pin bites while being
# incapable of noticing that it had stopped ([[artifact-pinning-test-cannot-see-the-code]]).
# This version writes the sabotaged source to a temp tree and calls the pin functions
# themselves against it, so a pin that stops biting turns the control red.
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("rel", "old", "new", "pin"),
    [
        (
            "stage2/eval.py",
            '"backbone": resolved_backbone,',
            '"backbone": BR.PRODUCTION_BACKBONE,',
            "test_the_sidecars_backbone_is_the_RESOLVED_one_not_a_constant",
        ),
        (
            "stage2/train.py",
            '"env_lock": env_lock_for(cfg.backbone),',
            '"env_lock": ENV_LOCK,',
            "test_the_training_report_stamps_the_BACKBONES_lock_not_the_module_constant",
        ),
        (
            "train/lora_harness.py",
            "backbone_loaded_from_registry = base_model is None",
            "backbone_loaded_from_registry = True",
            "test_loaded_from_registry_is_derived_from_the_injected_base_model",
        ),
        (
            "calib/gate2.py",
            "if mismatched:",
            "if False:",
            "test_gate2_main_raises_on_a_sidecar_identity_mismatch",
        ),
        (
            "stage2/sizing.py",
            "backbone=backbone,",
            "backbone=PRODUCTION_BACKBONE,",
            "test_every_measure_batch_call_site_sizes_the_requested_backbone",
        ),
    ],
)
def test_each_pin_goes_red_on_its_own_sabotage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rel: str, old: str, new: str, pin: str
) -> None:
    """Apply the sabotage to a COPY of src/, point the pins at it, and require the named one
    to fail. A pin that cannot go red is not a pin ([[raises-test-needs-a-positive-control]]).
    """
    import shutil

    sandbox = tmp_path / "tbox_finder"
    shutil.copytree(SRC, sandbox)
    target = sandbox / rel
    source = target.read_text(encoding="utf-8")
    assert old in source, f"the sabotage anchor is gone from {rel}: {old!r}"
    target.write_text(source.replace(old, new), encoding="utf-8")
    assert target.read_text(encoding="utf-8") != source, "the sabotage never applied"

    monkeypatch.setattr(sys.modules[__name__], "SRC", sandbox)
    with pytest.raises(AssertionError):
        globals()[pin]()


def test_every_pin_passes_on_the_real_tree() -> None:
    """...and the same pins must PASS unmodified, or the control above proves nothing."""
    for name, fn in sorted(globals().items()):
        if name.startswith("test_the_") or name.startswith("test_every_measure"):
            if name in {"test_every_pin_passes_on_the_real_tree"}:
                continue
            if fn.__code__.co_argcount:
                continue
            fn()


# --------------------------------------------------------------------------------------- #
# The schema_version coercion fixes — two of three had no test at all.
#
# `str(report.get("schema_version"))` made the JSON *number* `1` indistinguishable from schema
# `"1"` and handed it that schema's clause exemptions. Round 8 removed the coercion in three
# validators; only `calib.gate2`'s removal was covered, so reverting either of the other two
# restored the exemption-laundering bug with the whole suite green.
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rel",
    ["calib/gate2.py", "stage2/eval.py", "stage2/sizing.py"],
)
def test_no_validator_coerces_schema_version_to_a_string(rel: str) -> None:
    """SABOTAGE: `schema = report.get("schema_version")` -> `= str(report.get(...))`."""
    tree = _tree(rel)
    validators = _functions(tree, "validate_report")
    assert validators, f"{rel} defines no validate_report"
    for fn in validators:
        assigns = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "schema" for t in n.targets)
        ]
        assert assigns, f"{rel}: validate_report no longer reads a `schema` local"
        for assign in assigns:
            rendered = ast.unparse(assign.value)
            assert "str(" not in rendered, (
                f"{rel}:{assign.lineno} coerces schema_version — the JSON number 1 would then "
                f"collect schema '1' exemptions: {rendered!r}"
            )
            # `ast.unparse` renders string literals single-quoted; compare the AST-normalised
            # form rather than the source spelling.
            assert rendered == "report.get('schema_version')", rendered


@pytest.mark.parametrize(
    "module",
    [
        "tbox_finder.calib.gate2",
        "tbox_finder.stage2.eval",
        # ⚠ Round 11: sizing was named by the AST half above and omitted here, so the third
        # round-8 coercion fix had no behavioural cover. Its validator is torch-free and runs
        # in CI's stack, so there was never a reason to skip it. A dead second parametrize
        # element (`attr`, always None, never referenced) is dropped with it.
        "tbox_finder.stage2.sizing",
    ],
)
def test_a_numeric_schema_version_is_refused_by_each_validator(module: str) -> None:
    """The behavioural half: a numeric version is refused, and gets no exemption."""
    import importlib

    mod = importlib.import_module(module)
    problems = mod.validate_report({"schema_version": 1})
    assert any("is not one of" in p for p in problems), (module, problems)
