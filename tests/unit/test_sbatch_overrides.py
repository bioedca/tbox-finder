"""Every Hydra CLI override in every shipped sbatch must actually compose.

This gate exists because SLURM job 669 (the P2-09 production Stage-1 run) died in its
first seconds, after a ~9 h queue wait, on::

    Could not override 'exclude_selection_val'.
    Key 'exclude_selection_val' is not in struct

P2-09 added ``exclude_selection_val`` as a :class:`Stage1TrainConfig` field and passed it
on the ``torchrun`` line, but never added the key to ``conf/train/stage1.yaml``. Hydra
composes in **struct mode**, so an override naming a key the config lacks is a hard
``ConfigCompositionException`` — before any training starts.

Nothing in the suite caught it. ``tests/unit/test_production_fold.py`` exercises
``_cfg_from_mapping({"exclude_selection_val": False, ...})`` — a plain dict, one layer
BELOW Hydra — and ``sbatch --test-only`` validates only SLURM resources, never the
command. The blind spot was the composition layer itself.

So this gate reads each sbatch's **own bytes**, extracts the override tokens from the
launch line, resolves the entrypoint's ``config_name`` by statically parsing its
``@hydra.main`` decorator (no import, no torch), and runs real composition.

Two deliberate limits, stated rather than hidden:

* Values that are shell-interpolated (``report_path="$REPORT"``) cannot be known
  statically and are replaced with a placeholder. The invariant under test is
  **key existence under struct mode**, which is exactly the failure class above.
* A module with no ``@hydra.main`` decorator is not Hydra-driven and is skipped —
  but :func:`test_hydra_entrypoints_are_discovered` asserts the discovery actually
  found the known entrypoints, so a regex that silently matches nothing fails loudly
  instead of vacuously passing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONF = REPO / "conf"
SLURM = REPO / "slurm"
SRC = REPO / "src"

hydra = pytest.importorskip("hydra", reason="hydra-core drives the run-config composition")

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

#: A Hydra CLI override: optional leading ``+``, a dotted key, then ``=``.
_OVERRIDE = re.compile(r"^\+?[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*=")
#: ``config_name="train/stage1"`` inside an ``@hydra.main(...)`` decorator.
_CONFIG_NAME = re.compile(r"@hydra\.main\([^)]*config_name=\"([^\"]+)\"", re.S)
#: Any in-repo module launch, whatever the launcher. Deliberately NOT a list of launcher
#: names: the repo already uses three (``torchrun``, ``python -m``, ``accelerate launch``)
#: and an unknown fourth must not silently drop out of this gate — as
#: ``accelerate launch`` did on the first cut of this file.
_MODULE_LAUNCH = re.compile(r"(?:^|\s)-m\s+(tbox_finder\.[A-Za-z0-9_.]+)")


class Block:
    """One launch invocation found in an sbatch file."""

    def __init__(self, path: Path, lineno: int, module: str, overrides: list[str]):
        self.path = path
        self.lineno = lineno
        self.module = module
        self.overrides = overrides

    @property
    def config_name(self) -> str | None:
        """The entrypoint's Hydra ``config_name``, or None if it is not Hydra-driven."""
        source = SRC.joinpath(*self.module.split(".")).with_suffix(".py")
        if not source.is_file():
            return None
        found = _CONFIG_NAME.search(source.read_text())
        return found.group(1) if found else None

    def __repr__(self) -> str:
        return f"{self.path.relative_to(REPO)}:{self.lineno}[{self.module}]"


def _logical_lines(path: Path) -> list[tuple[int, str]]:
    """Join ``\\`` continuations into logical lines, dropping comments.

    Returns ``(1-based lineno of the first physical line, joined text)``.
    """
    lines = path.read_text().splitlines()
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        start = i
        chunk: list[str] = []
        while i < len(lines):
            chunk.append(lines[i].rstrip())
            if not lines[i].rstrip().endswith("\\"):
                break
            i += 1
        joined = " ".join(c.rstrip("\\") for c in chunk if not c.lstrip().startswith("#"))
        if joined.strip():
            out.append((start + 1, joined))
        i += 1
    return out


def _blocks(path: Path) -> list[Block]:
    """Extract every in-repo module launch from ``path``, whatever the launcher."""
    out: list[Block] = []
    for lineno, text in _logical_lines(path):
        found = _MODULE_LAUNCH.search(text)
        if not found:
            continue
        module = found.group(1)
        # Overrides are ONLY what follows `-m <module>`; tokens before the launcher
        # are shell env prefixes (PYTHONHASHSEED=0 PYTHONPATH=src …).
        tail = text[found.end() :].split()
        overrides = [t for t in tail if _OVERRIDE.match(t)]
        out.append(Block(path, lineno, module, overrides))
    return out


def _placeholder(token: str) -> str:
    """Neutralise shell interpolation; the key is what this gate tests."""
    key, _, value = token.partition("=")
    return f"{key}=_shell_" if "$" in value else token


def _all_blocks() -> list[Block]:
    return [b for f in sorted(SLURM.rglob("*.sbatch")) for b in _blocks(f)]


def _gated() -> list[Block]:
    """Launch blocks that are Hydra-driven AND carry at least one override."""
    return [b for b in _all_blocks() if b.config_name and b.overrides]


GATED = _gated()


def test_hydra_entrypoints_are_discovered():
    """Emptiness guard: the discovery must actually find the known entrypoints.

    Without this, a regex that matches nothing turns every parametrised composition
    case below into a vacuous pass — green because it tested nothing.
    """
    assert GATED, "no Hydra-driven sbatch launch blocks discovered at all"
    configs = {b.config_name for b in GATED}
    assert "train/stage1" in configs, f"train/stage1 entrypoint not discovered: {configs}"
    modules = {b.module for b in GATED}
    assert "tbox_finder.train.train_stage1" in modules, modules


def test_no_sbatch_launch_is_silently_dropped():
    """Every sbatch that launches an in-repo module must yield a discovered block.

    The first cut of this file keyed discovery off a list of launcher names
    (``torchrun``/``python -m``) and silently dropped
    ``slurm/p1/gtdb_continued_pretrain.sbatch``, which uses ``accelerate launch`` —
    a Hydra entrypoint with three overrides, ungated and invisible. A gate that
    quietly covers less than it claims is worse than no gate, so assert the file-level
    coverage directly rather than trusting the extractor.
    """
    found = {b.path for b in _all_blocks()}
    expected = {f for f in sorted(SLURM.rglob("*.sbatch")) if "-m tbox_finder." in f.read_text()}
    missing = {f.relative_to(REPO) for f in expected - found}
    assert not missing, f"sbatch files launch a module but yielded no block: {missing}"


def test_production_block_carries_the_override_that_broke_job_669():
    """The specific regression: P2-09's sbatch passes ``exclude_selection_val``.

    Pins the fixture to the real failure so this file cannot drift into testing a
    launch line that no longer contains the override class that caused the outage.
    """
    production = [b for b in GATED if b.path.name == "train_production.sbatch"]
    assert production, "slurm/p2/train_production.sbatch has no gated launch block"
    tokens = [t for b in production for t in b.overrides]
    assert "exclude_selection_val=false" in tokens, tokens


@pytest.mark.parametrize("block", GATED, ids=repr)
def test_every_sbatch_override_composes(block: Block):
    """Real struct-mode composition of the overrides as the sbatch actually spells them."""
    overrides = [_placeholder(t) for t in block.overrides]
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(CONF), version_base=None):
            compose(config_name=block.config_name, overrides=overrides)
    except Exception as exc:  # noqa: BLE001 - any composition failure is the defect
        pytest.fail(
            f"{block} does not compose against conf/{block.config_name}.yaml\n"
            f"  overrides: {' '.join(overrides)}\n"
            f"  {type(exc).__name__}: {str(exc).strip().splitlines()[0]}"
        )
    finally:
        GlobalHydra.instance().clear()


def test_exclude_selection_val_is_a_config_key_and_a_dataclass_field():
    """The two-sided contract job 669 had only one side of.

    The field existed on ``Stage1TrainConfig``; the config key did not. Either half
    alone is a run that dies on the cluster, so assert both — the config key by
    composition (not by reading the YAML, which would not prove Hydra accepts it).
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(CONF), version_base=None):
            cfg = compose(config_name="train/stage1", overrides=[])
            assert "exclude_selection_val" in cfg, list(cfg.keys())
            # Default must hold the val rung OUT — the sweep ranks on it.
            assert cfg.exclude_selection_val is True
            overridden = compose(
                config_name="train/stage1", overrides=["exclude_selection_val=false"]
            )
            assert overridden.exclude_selection_val is False
    finally:
        GlobalHydra.instance().clear()

    # Parse the AST rather than regex the text: an indented `exclude_selection_val: bool`
    # anywhere in the module — another dataclass, a local annotation, a docstring example —
    # would satisfy a regex while Stage1TrainConfig itself had lost the field. Only the
    # class-level annotation ON Stage1TrainConfig is the contract. (AST, not import: the
    # module pulls in torch, which this unit tier does not have.)
    tree = ast.parse((SRC / "tbox_finder" / "train" / "train_stage1.py").read_text())
    cls = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "Stage1TrainConfig"
        ),
        None,
    )
    assert cls is not None, "Stage1TrainConfig class not found in train_stage1.py"
    annotated = {
        n.target.id: ast.unparse(n.annotation)
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    assert (
        "exclude_selection_val" in annotated
    ), f"Stage1TrainConfig lost the exclude_selection_val field; has {sorted(annotated)}"
    assert annotated["exclude_selection_val"] == "bool", annotated["exclude_selection_val"]


#: The P2-10d CLI-overridable fields, with their expected dataclass annotations. Job 669's
#: lesson generalised: the moment a step adds a knob, both halves must exist — and no
#: sbatch passes these yet (P2-10e will), so an sbatch-driven gate would not cover them.
P2_10D_FIELDS: dict[str, str] = {
    "negative_pool_parquet": "str | None",
    "negative_fraction": "float",
    "negative_max_records": "int | None",
    "init_from_checkpoint": "str | None",
    "eval_at_step0": "bool",
}


def test_p2_10d_fields_are_config_keys_and_dataclass_fields():
    """Both halves of the job-669 contract for every knob P2-10d added.

    The config key is asserted by **real composition**, including an actual CLI-shaped
    override — reading the YAML would prove the text exists, not that Hydra accepts the
    override under struct mode, which is precisely the distinction job 669 died on.
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(CONF), version_base=None):
            cfg = compose(config_name="train/stage1", overrides=[])
            for key in P2_10D_FIELDS:
                assert key in cfg, f"{key} missing from conf/train/stage1.yaml: {list(cfg.keys())}"
            # The shipped defaults are the pre-P2-10d behaviour: positives only, fresh build.
            assert cfg.negative_pool_parquet is None
            assert cfg.negative_fraction == 0.0
            assert cfg.negative_max_records is None
            assert cfg.init_from_checkpoint is None
            assert cfg.eval_at_step0 is False
            # And a realistic P2-10e launch line composes as a whole, not key by key.
            mined = compose(
                config_name="train/stage1",
                overrides=[
                    "negative_pool_parquet=data/processed/negatives/mined_round1.parquet",
                    "negative_fraction=0.9090909090909091",
                    "negative_max_records=20000",
                    "init_from_checkpoint=data/processed/checkpoints/stage1_production/stage1.pt",
                    "eval_at_step0=true",
                ],
            )
            assert mined.eval_at_step0 is True
            assert mined.negative_max_records == 20000
    finally:
        GlobalHydra.instance().clear()

    tree = ast.parse((SRC / "tbox_finder" / "train" / "train_stage1.py").read_text())
    cls = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "Stage1TrainConfig"
        ),
        None,
    )
    assert cls is not None, "Stage1TrainConfig class not found in train_stage1.py"
    annotated = {
        n.target.id: ast.unparse(n.annotation)
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    for key, annotation in P2_10D_FIELDS.items():
        assert key in annotated, f"Stage1TrainConfig lost {key}; has {sorted(annotated)}"
        assert annotated[key] == annotation, (key, annotated[key])
