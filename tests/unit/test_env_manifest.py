"""P0-05: the conda env manifest is complete and every Snakemake `conda:` resolves.

Two guards (imp.md P0-05 Validation; CLAUDE.md §3.2 "rule = environment"):
  1. The six canonical per-env specs (ADR-0002 D1 + A4 ml split) all exist and are
     well-formed, and both GPU envs pin torch by direct URL (A4 cu128-closure invariant).
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

# The environments pinned by ADR-0002 D1 (one lockfile each, never aggregate). The `ml`
# env was split into `ml-dna` + `ml-rna` at P0-06c (ADR-0002 A4): transformers 4.57.5
# (Caduceus trust_remote_code ceiling) is mutually exclusive with `multimolecule` 0.0.9,
# which at import needs transformers 5.x.
EXPECTED_ENVS = ["data", "infernal", "ml-dna", "ml-rna", "viz", "app", "rscape"]

# The two GPU envs (ADR-0002 D2/D3/A4); the torch-URL / no-`--extra-index-url` guard
# applies to both, since both carry the same cu128 URL-pinned closure.
ML_ENVS = ["ml-dna", "ml-rna"]

REQUIRED_YAML_KEYS = ("channels:", "dependencies:")

# Matches both `conda: "../../envs/data.yml"` and the block form
#     conda:
#         "../../envs/data.yml"
# \s includes newlines, so the block form is covered; the path must end in .yml/.yaml.
CONDA_DIRECTIVE_RE = re.compile(r"""conda:\s*['"]([^'"]+\.ya?ml)['"]""")


def test_all_env_specs_exist():
    missing = [e for e in EXPECTED_ENVS if not (ENVS_DIR / f"{e}.yml").is_file()]
    assert not missing, f"missing envs/*.yml specs: {missing}"
    # The superseded single `ml` env spec must be gone (A4 replaced it with ml-dna + ml-rna).
    assert not (ENVS_DIR / "ml.yml").is_file(), "envs/ml.yml must be deleted (ADR-0002 A4)"


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


def test_all_env_lockfiles_exist():
    """All six envs are locked: the four CPU envs at P0-05, both GPU envs at P0-06c (A4).

    ml-dna / ml-rna are lockable on the laptop only via a full URL-pinned cu128 closure
    (see `test_ml_envs_pin_torch_by_url_not_index`); conda-lock 4.0.2 cannot use an index.
    """
    missing = [e for e in EXPECTED_ENVS if not (ENVS_DIR / f"{e}.conda-lock.yml").is_file()]
    assert not missing, f"missing per-env lockfiles: {missing}"


def test_ml_envs_pin_torch_by_url_not_index():
    """Lock the ADR-0002 A4 invariant for BOTH GPU envs' cu128 pip stack.

    conda-lock 4.0.2 rejects an inline `--extra-index-url` line in a pip: block (PEP 508
    parse error) and its wheel-tag matcher omits `manylinux_2_27` (torch+cu128's nvidia deps
    ship only as `_2_27`). So each ml env must obtain torch via a direct wheel URL and must
    NOT carry an `--extra-index-url` line. Guarding both ml-dna and ml-rna prevents a
    well-meaning "simplify to an index" edit from silently breaking `conda-lock`.
    """
    problems = []
    for env in ML_ENVS:
        text = (ENVS_DIR / f"{env}.yml").read_text()
        # Strip inline `#` comments (they legitimately explain the --extra-index-url history)
        # so the guard only inspects real YAML content, not prose.
        code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        if re.search(r"^\s*-\s*--extra-index-url", code, re.MULTILINE):
            problems.append(f"{env}.yml: has an `--extra-index-url` pip entry (A4 forbids it)")
        if not re.search(r"^\s*-\s*torch\s*@\s*https://\S+\.whl\s*$", code, re.MULTILINE):
            problems.append(f"{env}.yml: does not pin `torch` by a direct wheel URL")
    assert not problems, "ml torch-URL invariant violated (ADR-0002 A4): " + "; ".join(problems)


def test_ml_rna_pins_multimolecule_0_1_0():
    """Lock the ADR-0002 A8 fix: ml-rna pins `multimolecule==0.1.0`, never 0.0.9.

    multimolecule 0.0.9's RiNALMo forward calls transformers'
    `create_bidirectional_mask(input_embeds=...)`, whose compat alias transformers
    removed in 5.9.0 → TypeError under the pinned transformers 5.13.0 (the P1-13
    forward gate). 0.1.0 is the first release passing `inputs_embeds=` (the v5
    masking API). This guard fails closed on a regression to the broken 0.0.9 and
    confirms the lockfile matches the spec.
    """
    yml = (ENVS_DIR / "ml-rna.yml").read_text()
    code = "\n".join(line.split("#", 1)[0] for line in yml.splitlines())
    assert re.search(
        r"^\s*-\s*multimolecule==0\.1\.0\s*$", code, re.MULTILINE
    ), "ml-rna.yml must pin `multimolecule==0.1.0` (ADR-0002 A8 transformers-5 forward fix)"
    assert not re.search(
        r"^\s*-\s*multimolecule==0\.0\.9\s*$", code, re.MULTILINE
    ), "ml-rna.yml must NOT pin the broken multimolecule==0.0.9 (fails on transformers 5.13.0)"
    # the regenerated lockfile must record 0.1.0 (spec ↔ lock consistency).
    lock = (ENVS_DIR / "ml-rna.conda-lock.yml").read_text()
    assert re.search(
        r"^- name: multimolecule\n\s+version: 0\.1\.0\s*$", lock, re.MULTILINE
    ), "ml-rna.conda-lock.yml must lock multimolecule 0.1.0 (re-solve after the A8 spec bump)"


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
