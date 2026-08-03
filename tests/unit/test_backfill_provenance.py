"""The P3-06 provenance backfill — and the drift guard its self-containment requires.

`scripts/backfill_stage2_provenance.py` defines its own `checkpoint_output_files` instead of
importing `train`'s. That is deliberate and constrained, not laziness: the script must run
against the checkout the RUN used (its git_sha guard refuses otherwise, so a moved tree cannot
misattribute the run), and that checkout predates the shared helper. A repair script cannot
depend on code newer than the tree it repairs.

The cost of that self-containment is a fork, so this pins the two implementations together:
they must agree on identical input, including both refusal paths. If someone changes one, this
fails rather than the sidecars quietly recording a different output set than the sbatch does.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "backfill_stage2_provenance.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("_backfill", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(root: Path) -> Path:
    ckpt = root / "aux1.0_lr1e-4"
    adapter = ckpt / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00")
    (adapter / "README.md").write_text("peft")
    (ckpt / "stage2_heads.pt").write_bytes(b"\x00")
    return ckpt


def test_the_backfill_enumerator_agrees_with_the_shared_helper() -> None:
    """One behaviour, two homes — asserted, not assumed."""
    backfill = _load_script()
    from tbox_finder.stage2 import train as T

    with tempfile.TemporaryDirectory() as root:
        ckpt = _checkpoint(Path(root))
        assert backfill.checkpoint_output_files(ckpt) == T.checkpoint_output_files(ckpt)
        # And it is the real contract, not a coincidence of both returning nothing.
        assert len(backfill.checkpoint_output_files(ckpt)) == 4


def test_both_enumerators_refuse_the_same_two_ways() -> None:
    """A fork that agreed on the happy path and diverged on refusals would still be a fork."""
    backfill = _load_script()
    from tbox_finder.stage2 import train as T

    with tempfile.TemporaryDirectory() as empty:
        for fn in (backfill.checkpoint_output_files, T.checkpoint_output_files):
            with pytest.raises(FileNotFoundError):
                fn(empty)
    for fn in (backfill.checkpoint_output_files, T.checkpoint_output_files):
        with pytest.raises(NotADirectoryError):
            fn(_SCRIPT)


def test_the_adapter_directory_is_never_an_output() -> None:
    """The exact value that raised IsADirectoryError on all six points of job 1064."""
    backfill = _load_script()
    with tempfile.TemporaryDirectory() as root:
        ckpt = _checkpoint(Path(root))
        outputs = backfill.checkpoint_output_files(ckpt)
        assert str(ckpt / "lora_adapter") not in outputs
        assert all(Path(o).is_file() for o in outputs)


def test_the_backfill_excludes_its_own_sidecar_so_write_is_idempotent() -> None:
    """A second --write would otherwise enumerate the provenance.json the first one wrote.

    The record would then declare itself as one of its own outputs, and that entry's hash can
    never match after the file is written — a self-referential provenance record.
    """
    backfill = _load_script()
    with tempfile.TemporaryDirectory() as root:
        ckpt = _checkpoint(Path(root))
        before = backfill.checkpoint_output_files(ckpt)
        (ckpt / "provenance.json").write_text("{}")
        after = backfill.checkpoint_output_files(ckpt)
        assert after == before, "the sidecar leaked into its own outputs list"
        assert not any(Path(o).name == "provenance.json" for o in after)
