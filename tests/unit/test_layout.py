"""P0-01: the repository tree matches the PRD §16 layout.

Directory-presence gate (imp.md P0-01 Validation). Guards against a later
step accidentally deleting a scaffold directory the workflow depends on.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DIRS = [
    "src/tbox_finder",
    "workflow/rules",
    "workflow/profiles/slurm",
    "conf/model",
    "conf/data",
    "conf/optim",
    "envs",
    "slurm",
    "data/raw",
    "data/external",
    "data/interim",
    "data/processed",
    "tests/unit",
    "tests/golden",
    "tests/ml",
    "tests/fixtures",
    "analyses",
    "figures",
    "paper",
    "app",
    "docker",
    "docs/decisions",
]

EXPECTED_FILES = [
    "pyproject.toml",
    "README.md",
    "workflow/Snakefile",
    "src/tbox_finder/__init__.py",
    "analyses/phase0_log.qmd",
]


def test_expected_directories_exist():
    missing = [d for d in EXPECTED_DIRS if not (REPO_ROOT / d).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_expected_files_exist():
    missing = [f for f in EXPECTED_FILES if not (REPO_ROOT / f).is_file()]
    assert not missing, f"missing files: {missing}"
