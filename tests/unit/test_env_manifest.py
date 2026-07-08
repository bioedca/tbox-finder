"""P0-05: the conda env manifest is complete and every Snakemake `conda:` resolves.

Two guards (imp.md P0-05 Validation; CLAUDE.md §3.2 "rule = environment"):
  1. The five canonical per-env specs (ADR-0002 D1) all exist and are well-formed.
  2. Every `conda:` directive in `workflow/rules/*.smk` points at a file that exists
     (resolved relative to the .smk that declares it, as Snakemake does). Vacuously
     true until the first rule lands (P0-08+); it then locks the rule=env contract.

Stdlib-only (no PyYAML) so it runs in any CI test env.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVS_DIR = REPO_ROOT / "envs"
RULES_DIR = REPO_ROOT / "workflow" / "rules"

# The five environments pinned by ADR-0002 D1 (one lockfile each, never aggregate).
EXPECTED_ENVS = ["data", "infernal", "ml", "viz", "app"]

REQUIRED_YAML_KEYS = ("channels:", "dependencies:")

# Matches both `conda: "../../envs/data.yml"` and the block form
#     conda:
#         "../../envs/data.yml"
# \s includes newlines, so the block form is covered; the path must end in .yml/.yaml.
CONDA_DIRECTIVE_RE = re.compile(r"""conda:\s*['"]([^'"]+\.ya?ml)['"]""")


def test_all_five_env_specs_exist():
    missing = [e for e in EXPECTED_ENVS if not (ENVS_DIR / f"{e}.yml").is_file()]
    assert not missing, f"missing envs/*.yml specs: {missing}"


def test_env_specs_are_well_formed():
    """Each spec declares a name, channels, and dependencies (text-level check)."""
    problems = []
    for e in EXPECTED_ENVS:
        text = (ENVS_DIR / f"{e}.yml").read_text()
        if not re.search(r"^name:\s*\S+", text, re.MULTILINE):
            problems.append(f"{e}.yml: no `name:`")
        for key in REQUIRED_YAML_KEYS:
            if key not in text:
                problems.append(f"{e}.yml: no `{key}`")
    assert not problems, f"malformed env specs: {problems}"


def test_cpu_env_lockfiles_exist():
    """The four CPU envs are locked at P0-05; `ml` is deferred to P0-06 (ADR-0002 D3)."""
    cpu_envs = [e for e in EXPECTED_ENVS if e != "ml"]
    missing = [e for e in cpu_envs if not (ENVS_DIR / f"{e}.conda-lock.yml").is_file()]
    assert not missing, f"missing per-env lockfiles: {missing}"


def test_snakemake_conda_directives_resolve():
    """Every `conda:` path in a rule file resolves to an existing spec."""
    if not RULES_DIR.is_dir():
        return
    unresolved = []
    for smk in sorted(RULES_DIR.glob("*.smk")):
        for rel in CONDA_DIRECTIVE_RE.findall(smk.read_text()):
            target = (smk.parent / rel).resolve()
            if not target.is_file():
                unresolved.append(f"{smk.name} -> {rel}")
    assert not unresolved, f"unresolved conda: directives: {unresolved}"
