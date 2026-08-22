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
# which at import needs transformers 5.x. `rscape` was added at P2-10c (A11),
# `homology` at P2-10c′-e (A12: hmmer+blast, no infernal co-pin), `locarna` at
# P2-10e-msa (A13: locarna 2.0.1 / mlocarna, the D7 CM-free comparative-consensus aligner),
# and `ml-rnafm` at P3-17 (A15: `multimolecule` 0.2.0 for the D6 RNA-FM comparator, because
# ml-rna's 0.1.0 cannot load that checkpoint at all) — ten envs total.
EXPECTED_ENVS = [
    "data",
    "infernal",
    "ml-dna",
    "ml-rna",
    "viz",
    "app",
    "rscape",
    "homology",
    "locarna",
    "ml-rnafm",
]

# The two GPU envs (ADR-0002 D2/D3/A4); the torch-URL / no-`--extra-index-url` guard
# applies to all three, since they carry the same cu128 URL-pinned closure. `ml-rnafm` is
# `ml-rna` with ONE pin changed (ADR-0002 A15), so it inherits the invariant unchanged — and
# omitting it here would leave the comparator's env the only GPU spec nothing checks.
ML_ENVS = ["ml-dna", "ml-rna", "ml-rnafm"]

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
    """All ten envs are locked: four CPU envs at P0-05, both GPU envs at P0-06c (A4),
    `rscape` at P2-10c (A11 — a separate env so the GATE-1-load-bearing `infernal`
    lock is not re-solved by `rscape`'s gnuplot/Qt closure), `homology` at
    P2-10c′-e (A12 — hmmer/blast with NO infernal co-pin, so `envs/infernal.conda-lock.yml`
    is likewise never re-solved and the `esl-*` two-provider clobber is out of reach), and
    `locarna` at P2-10e-msa (A13 — locarna 2.0.1/mlocarna, again a separate env so the
    infernal lock stays byte-frozen), and `ml-rnafm` at P3-17 (A15 — `multimolecule` 0.2.0
    for the D6 RNA-FM comparator, hand-merged from `ml-rna`'s lock so nothing else moved).

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


def _pip_pins(env: str) -> list[str]:
    """The `- name==version` pip lines of a spec, comments stripped."""
    text = (ENVS_DIR / f"{env}.yml").read_text()
    code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return [m.group(1).strip() for m in re.finditer(r"^\s*-\s*(\S+==\S+)\s*$", code, re.MULTILINE)]


def test_ml_rnafm_pins_multimolecule_0_2_0_and_ml_rna_does_not_move():
    """Lock ADR-0002 A15 in **both** directions — that is the whole point of the split.

    A15 added `envs/ml-rnafm.yml` (`multimolecule` 0.2.0) rather than bumping `ml-rna`,
    because 0.1.0 cannot load `multimolecule/rnafm` at all (`configuration_rnafm` demands
    `vocab_size == 26`; the published checkpoint declares 28) and bumping in place would fire
    CLAUDE.md §8.5 across every published RiNALMo Stage-2 number.

    So this asserts the comparator env moved AND that the shipped one did **not**. A guard on
    the new env alone would stay green through exactly the change A15 declined to make.
    """
    code = "\n".join(
        line.split("#", 1)[0] for line in (ENVS_DIR / "ml-rnafm.yml").read_text().splitlines()
    )
    assert re.search(
        r"^\s*-\s*multimolecule==0\.2\.0\s*$", code, re.MULTILINE
    ), "ml-rnafm.yml must pin `multimolecule==0.2.0` (ADR-0002 A15 — the RNA-FM comparator)"
    assert not re.search(
        r"^\s*-\s*multimolecule==0\.1\.0\s*$", code, re.MULTILINE
    ), "ml-rnafm.yml must NOT pin 0.1.0 — it cannot load multimolecule/rnafm"

    lock = (ENVS_DIR / "ml-rnafm.conda-lock.yml").read_text()
    assert re.search(
        r"^- name: multimolecule\n\s+version: 0\.2\.0\s*$", lock, re.MULTILINE
    ), "ml-rnafm.conda-lock.yml must lock multimolecule 0.2.0 (spec ↔ lock consistency)"

    # ...and the shipped env is UNMOVED. `test_ml_rna_pins_multimolecule_0_1_0` asserts the
    # same thing from A8's side; restated here so the A15 split's own invariant is complete in
    # one place rather than resting on a neighbour that could be edited for another reason.
    rna = "\n".join(
        line.split("#", 1)[0] for line in (ENVS_DIR / "ml-rna.yml").read_text().splitlines()
    )
    assert re.search(
        r"^\s*-\s*multimolecule==0\.1\.0\s*$", rna, re.MULTILINE
    ), "ml-rna.yml must STILL pin multimolecule==0.1.0 — A15 split precisely to avoid bumping it"


def test_ml_rnafm_differs_from_ml_rna_in_exactly_one_dependency():
    """A15's minimize-divergence design, asserted rather than described.

    The comparator's value rests on differing from the shipped arm in the backbone alone. If
    `ml-rnafm` drifted on torch, the kernel wheels or transformers, a P3-18 ECE difference
    would no longer be attributable to the backbone — and the drift would be invisible, since
    both envs would still install and both would still train something.

    ⚠ This is exactly the failure a *fresh* `conda-lock` re-solve produced while A15 was being
    authored: solving `ml-rnafm.yml` from scratch moved 55 packages (gcc 14.3→14.4,
    libstdcxx 15.2→16.1, mkl, pandas, setuptools, and multimolecule's own danling/chanfig).
    The committed lock is hand-merged from `ml-rna`'s for that reason, and this test is what
    keeps a future re-solve from quietly undoing it.
    """
    rna = _pip_pins("ml-rna")
    fm = _pip_pins("ml-rnafm")
    assert rna, "ml-rna.yml yielded no `name==version` pip pins — the extractor is broken"
    only_rna = sorted(set(rna) - set(fm))
    only_fm = sorted(set(fm) - set(rna))
    assert only_rna == ["multimolecule==0.1.0"], only_rna
    assert only_fm == ["multimolecule==0.2.0"], only_fm

    # The URL-pinned lines (torch + its cu128 closure + the three sm_86 kernel wheels) must be
    # byte-identical: those are the pins ADR-0002 D8 calls load-bearing.
    def urls(env: str) -> list[str]:
        text = (ENVS_DIR / f"{env}.yml").read_text()
        code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        return sorted(
            m.group(1).strip()
            for m in re.finditer(r"^\s*-\s*(\S+\s*@\s*https://\S+)\s*$", code, re.MULTILINE)
        )

    assert urls("ml-rna"), "ml-rna.yml yielded no URL-pinned wheels — the extractor is broken"
    assert urls("ml-rnafm") == urls("ml-rna"), (
        "ml-rnafm.yml's URL-pinned wheel closure differs from ml-rna's; ADR-0002 A15 pins them "
        "identical so a P3-18 ECE difference is attributable to the backbone alone"
    )


def test_the_two_ml_rna_locks_differ_only_by_multimolecule_and_matplotlib():
    """The lockfile half of the same invariant (ADR-0002 A15).

    The spec-level test above cannot see a lock that drifted on its own — and a lock IS what
    gets installed. 0.2.0 adds `matplotlib` (and its subtree) over 0.1.0's dependencies; every
    other locked package must be identical in name AND version.
    """
    pat = re.compile(r"^- name: (\S+)\n  version: (\S+)", re.MULTILINE)

    def locked(env: str) -> dict[str, str]:
        text = (ENVS_DIR / f"{env}.conda-lock.yml").read_text()
        return {m.group(1): m.group(2) for m in pat.finditer(text)}

    a, b = locked("ml-rna"), locked("ml-rnafm")
    assert len(a) > 100, f"ml-rna lock parsed only {len(a)} packages — the extractor is broken"
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    assert removed == [], f"ml-rnafm's lock DROPPED packages ml-rna has: {removed}"
    assert changed == [
        "multimolecule"
    ], f"ml-rnafm's lock changed versions beyond the pin: {changed}"
    assert added == [
        "contourpy",
        "cycler",
        "fonttools",
        "kiwisolver",
        "matplotlib",
        "pillow",
        "pyparsing",
    ], f"ml-rnafm's lock added packages beyond multimolecule 0.2.0's matplotlib subtree: {added}"


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


def test_homology_env_pins_hmmer_blast_and_no_infernal():
    """Lock the ADR-0002 A12 invariant: the 8th `homology` env pins the two absent
    sequence-search binaries (`hmmer` 3.4 = nhmmer, `blast` 2.17.0 = BLAST+) and
    **must NOT co-pin `infernal`**.

    The no-infernal rule is the load-bearing decision, not a style choice: `hmmer`
    and `infernal` each ship their own `esl-*` Easel miniapps into `bin/` and neither
    vendors Easel, so co-installing them is a genuine two-provider file clobber
    (A11's vendored-R-scape co-install never faced it). Keeping `infernal` out of
    this env also leaves the GATE-1-load-bearing `envs/infernal.conda-lock.yml`
    byte-frozen. This guard fails closed on a future edit that folds `infernal` in
    or drifts the pins, and confirms spec ↔ lock consistency.
    """
    spec = (ENVS_DIR / "homology.yml").read_text()
    # Strip inline `#` comments so the header prose (which legitimately explains the
    # esl-*/infernal rationale) is not inspected — only real YAML dependency lines.
    code = "\n".join(line.split("#", 1)[0] for line in spec.splitlines())
    problems = []
    for pat, why in (
        (r"^\s*-\s*hmmer=3\.4\s*$", "must pin `hmmer=3.4` (provides nhmmer)"),
        (r"^\s*-\s*blast=2\.17\.0\s*$", "must pin `blast=2.17.0` (BLAST+)"),
        (r"^\s*-\s*python=3\.12\s*$", "must pin `python=3.12` (ADR-0002 D1)"),
        (r"^\s*-\s*biopython=1\.87\s*$", "must pin `biopython=1.87` (sibling envs)"),
    ):
        if not re.search(pat, code, re.MULTILINE):
            problems.append(f"homology.yml: {why}")
    if re.search(r"^\s*-\s*infernal\b", code, re.MULTILINE):
        problems.append("homology.yml: must NOT co-pin `infernal` (A12 esl-* clobber avoidance)")
    assert not problems, "homology env A12 invariant violated: " + "; ".join(problems)

    # spec ↔ lock consistency + the same no-infernal invariant in the solved closure.
    lock = (ENVS_DIR / "homology.conda-lock.yml").read_text()
    lock_problems = []
    if not re.search(r"^- name: hmmer\n\s+version: '?3\.4'?\s*$", lock, re.MULTILINE):
        lock_problems.append("homology.conda-lock.yml must lock hmmer 3.4")
    if not re.search(r"^- name: blast\n\s+version: '?2\.17\.0'?\s*$", lock, re.MULTILINE):
        lock_problems.append("homology.conda-lock.yml must lock blast 2.17.0")
    if not re.search(r"^- name: python\n\s+version: '?3\.12\.\d+'?\s*$", lock, re.MULTILINE):
        lock_problems.append("homology.conda-lock.yml must lock python 3.12.x")
    if not re.search(r"^- name: biopython\n\s+version: '?1\.87'?\s*$", lock, re.MULTILINE):
        lock_problems.append("homology.conda-lock.yml must lock biopython 1.87")
    if re.search(r"^- name: infernal$", lock, re.MULTILINE):
        lock_problems.append(
            "homology.conda-lock.yml must NOT contain infernal (A12 esl-* clobber avoidance)"
        )
    assert not lock_problems, "; ".join(lock_problems)


def test_locarna_env_pins_locarna_2_0_1_and_no_infernal():
    """Lock the ADR-0002 A13 invariant: the 9th `locarna` env pins `locarna` 2.0.1 (the package
    shipping `mlocarna`, D7's first-named CM-free de-novo comparative-consensus aligner) over
    python 3.12 + biopython 1.87, with `viennarna` pulled transitively (mlocarna → RNAalifold).

    A SEPARATE env with **no `infernal` co-pin** — the load-bearing reason mirrors A11/A12: adding
    any dependency to `envs/infernal.yml` would re-solve the GATE-1-load-bearing
    `envs/infernal.conda-lock.yml` (byte-frozen at `776610ae…`). This guard fails closed on a
    drifted pin or an `infernal` fold-in, and confirms spec ↔ lock consistency.
    """
    spec = (ENVS_DIR / "locarna.yml").read_text()
    # Strip inline `#` comments so the header prose (which explains the isolation rationale) is not
    # inspected — only real YAML dependency lines.
    code = "\n".join(line.split("#", 1)[0] for line in spec.splitlines())
    problems = []
    for pat, why in (
        (r"^\s*-\s*locarna=2\.0\.1\s*$", "must pin `locarna=2.0.1` (provides mlocarna)"),
        (r"^\s*-\s*python=3\.12\s*$", "must pin `python=3.12` (ADR-0002 D1)"),
        (r"^\s*-\s*biopython=1\.87\s*$", "must pin `biopython=1.87` (sibling envs)"),
    ):
        if not re.search(pat, code, re.MULTILINE):
            problems.append(f"locarna.yml: {why}")
    if re.search(r"^\s*-\s*infernal\b", code, re.MULTILINE):
        problems.append(
            "locarna.yml: must NOT co-pin `infernal` (A13 keeps the infernal lock frozen)"
        )
    assert not problems, "locarna env A13 invariant violated: " + "; ".join(problems)

    # spec ↔ lock consistency + the same no-infernal invariant + the transitive viennarna runtime.
    lock = (ENVS_DIR / "locarna.conda-lock.yml").read_text()
    lock_problems = []
    if not re.search(r"^- name: locarna\n\s+version: '?2\.0\.1'?\s*$", lock, re.MULTILINE):
        lock_problems.append("locarna.conda-lock.yml must lock locarna 2.0.1")
    if not re.search(r"^- name: python\n\s+version: '?3\.12\.\d+'?\s*$", lock, re.MULTILINE):
        lock_problems.append("locarna.conda-lock.yml must lock python 3.12.x")
    if not re.search(r"^- name: biopython\n\s+version: '?1\.87'?\s*$", lock, re.MULTILINE):
        lock_problems.append("locarna.conda-lock.yml must lock biopython 1.87")
    if not re.search(r"^- name: viennarna\n", lock, re.MULTILINE):
        lock_problems.append("locarna.conda-lock.yml must pull viennarna (mlocarna → RNAalifold)")
    if re.search(r"^- name: infernal$", lock, re.MULTILINE):
        lock_problems.append("locarna.conda-lock.yml must NOT contain infernal (A13 frozen lock)")
    assert not lock_problems, "; ".join(lock_problems)
