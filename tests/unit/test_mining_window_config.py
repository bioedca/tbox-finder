"""P2-10d′-a: the mining window is a *pin*, not a local default (config ↔ code ↔ workflow).

``mining_window_nt`` is read by three independent consumers with three
independent fallbacks — ``decoys.read_config``'s hand-rolled line splitter,
``mining.pool.DEFAULT_WINDOW_NT``, and ``workflow/rules/common.smk::flank_pad_nt``
via ``yaml.safe_load`` — and until this file nothing tied any of them together
(``grep mining_window_nt tests/`` returned 0 hits). P2-10d measured what that
costs: the shipped context was padded to 1024 rather than 1024 + 50, the pool was
carved at 300, ``background_record`` refuses any sequence that is not exactly one
1024-nt training window, and the result was a substrate from which **no** §9.1
negative could be injected at all — with every test green.

Kept apart from ``tests/unit/test_mining_pool.py`` deliberately. That file is
stdlib-only window *geometry* (it says so in its own docstring); these are
config/workflow-wiring pins that need ``yaml`` and — for the training-width
comparison — ``window_dataset``, which pulls numpy and the splits machinery. The
two concerns fail for different reasons and should not share an import graph.

``conf/data/decoys.yaml`` and the two ``workflow/rules/*.smk`` files are **git-tracked**, so
their absence is a broken checkout and raises rather than skipping (the stronger
tier of the two-var pattern in ``tests/ml/test_mining_pool_gate.py``).

One dependency note: CI has ``PyYAML`` only *transitively*, via the
``omegaconf==2.3.0`` that ``hydra-core`` pulls in (verified — dropping those two
from the pip line leaves ``import yaml`` failing). The two tests below that need
it therefore import it **locally and unguarded**: if it ever disappears they must
ERROR loudly, never ``importorskip`` green, because a silently skipped drift
check is the failure this whole file exists to remove. ``common.smk`` imports
``yaml`` at module scope, so the ``dag`` job would be dead in that world anyway.
"""

from __future__ import annotations

import re
from pathlib import Path

from tbox_finder import decoys
from tbox_finder.mining import pool as mining_pool

REPO = Path(__file__).resolve().parents[2]
DECOYS_CONF = REPO / "conf/data/decoys.yaml"
DATA_SMK = REPO / "workflow/rules/data.smk"
#: The helper lives here, not beside the rule: `snakemake --lint` refuses "Mixed
#: rules and functions in same snakefile" and a red lint is CI-blocking.
COMMON_SMK = REPO / "workflow/rules/common.smk"

#: PRD §6 / ADR-0001 D31 / ADR-0005 D3 — the Stage-1 training window, pinned here
#: as an independent literal. Without it every assertion below is "A == B" between
#: two constants that a single edit moves together.
TRAINING_WINDOW_NT = 1024


def _read(path: Path) -> str:
    """Read a git-tracked file, raising (never skipping) when it is absent.

    A ``TBOX_REQUIRE_*``-style skip would go green on exactly the checkout where
    the pin matters. These two files travel with the repo.
    """
    if not path.is_file():
        raise AssertionError(f"git-tracked file missing: {path.relative_to(REPO)}")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# conf/data/decoys.yaml ↔ the code that consumes it
# --------------------------------------------------------------------------- #
def test_the_shipped_mining_window_is_the_training_window() -> None:
    """The pin ``decoys._Config`` and ``mining.pool`` both name in their docstrings.

    ``negatives.background_record`` refuses a sequence whose length is not exactly
    one training window, so a mining pool carved at any other width is not "a bit
    off" — it is 100 % uninjectable, and the §9.1 seed mix cannot be built from it.
    """
    # Lazy: ``window_dataset`` pulls numpy + ingest/labels/splits, which this
    # file otherwise has no need of.
    from tbox_finder.data.window_dataset import WINDOW_NT

    assert WINDOW_NT == TRAINING_WINDOW_NT
    assert decoys.read_config(DECOYS_CONF).mining_window_nt == WINDOW_NT


def test_the_bare_config_fallback_is_the_mining_pool_default() -> None:
    """``read_config(None)`` and a missing file both return the bare ``_Config``.

    ``build`` takes its width from whichever of the two it gets, so a default that
    disagreed with ``mining.pool``'s would silently rebuild the pool at a width no
    negative can be injected at — the P2-10d failure reached through the other door.
    """
    fallbacks = [
        decoys._Config(),
        decoys.read_config(None),
        decoys.read_config(REPO / "conf/data/does-not-exist.yaml"),
    ]
    for cfg in fallbacks:
        assert cfg.mining_window_nt == mining_pool.DEFAULT_WINDOW_NT == TRAINING_WINDOW_NT
        assert cfg.mining_margin_nt == mining_pool.DEFAULT_FLANK_MARGIN_NT


def test_the_two_independent_readers_of_the_decoy_config_agree() -> None:
    """``decoys.py`` hand-rolls a line splitter; ``data.smk`` uses ``yaml.safe_load``.

    Two parsers over one file is a real drift hole: a value the splitter cannot
    parse (a quoted scalar, a flow mapping, an anchor) is silently dropped and
    ``read_config`` returns its built-in default while the workflow reads the file
    value, so the pool and the flank pad would be derived from different numbers.
    """
    import yaml

    loaded = yaml.safe_load(_read(DECOYS_CONF)) or {}
    cfg = decoys.read_config(DECOYS_CONF)
    assert loaded["mining_window_nt"] == cfg.mining_window_nt
    assert loaded["mining_margin_nt"] == cfg.mining_margin_nt


def test_the_reader_agreement_check_is_not_vacuous(tmp_path: Path) -> None:
    """Control for the test above: on the shipped file both readers return 1024
    even if ``read_config`` ignored the file completely, because its fallback is
    also 1024. Only a config whose values differ from the built-in defaults can
    show that the splitter reads the file at all — including through the trailing
    ``#`` comments the shipped file uses.
    """
    import yaml

    path = tmp_path / "decoys.yaml"
    path.write_text(
        "mining_window_nt: 777  # not the built-in default\nmining_margin_nt: 13\n",
        encoding="utf-8",
    )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = decoys.read_config(path)
    assert cfg.mining_window_nt == loaded["mining_window_nt"] == 777
    assert cfg.mining_margin_nt == loaded["mining_margin_nt"] == 13
    assert mining_pool.DEFAULT_WINDOW_NT != 777  # the values really are distinguishing


# --------------------------------------------------------------------------- #
# workflow/rules/data.smk — the flags the rules must actually pass
# --------------------------------------------------------------------------- #
_RULE_START = re.compile(r"^rule\s+(\w+)\s*:", re.MULTILINE)


def _rule_block(text: str, name: str) -> str:
    """The source of one Snakemake rule, from its ``rule <name>:`` to the next rule."""
    starts = [(match.group(1), match.start()) for match in _RULE_START.finditer(text)]
    assert name in {rule for rule, _ in starts}, f"rule {name} not found in data.smk"
    for i, (rule, start) in enumerate(starts):
        if rule == name:
            end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
            return text[start:end]
    raise AssertionError("unreachable")  # pragma: no cover


def _shell_block(text: str, name: str) -> str:
    _, separator, shell = _rule_block(text, name).partition("\n    shell:\n")
    assert separator, f"rule {name} has no shell: block"
    return shell


def test_the_rule_slicer_discriminates_between_rules() -> None:
    """Control for the two scans below: a slicer that returned the whole file
    would find every flag in every rule, so both would pass regardless of which
    rule actually carries which flag."""
    smk = _read(DATA_SMK)
    assert "--pad-nt" not in _shell_block(smk, "build_mining_pool")
    assert "--union-prior" not in _shell_block(smk, "p2_flank_context")


def test_the_flank_rule_passes_a_pad_derived_from_the_mining_geometry() -> None:
    """``p2_flank_context`` has always *accepted* ``--pad-nt``; the rule never passed it.

    The shipped context was therefore padded to the CLI default and yielded **zero**
    carvable 1024-nt windows at margin 50, which ``build_mining_pool`` reported as a
    hard failure only after the whole NCBI fetch had run.
    """
    smk = _read(DATA_SMK)
    assert "--pad-nt {params.pad_nt:q}" in _shell_block(smk, "p2_flank_context")
    assert "pad_nt=flank_pad_nt()" in _rule_block(smk, "p2_flank_context")
    # …and the helper derives the number from the config rather than restating it.
    common = _read(COMMON_SMK)
    assert 'conf["mining_window_nt"]' in common
    assert 'conf["mining_margin_nt"]' in common


def test_the_flank_pad_is_never_written_as_a_literal() -> None:
    """1074 must exist nowhere in the ``.smk`` files — it is ``window + margin``, derived.

    A duplicated pad drifts silently the next time either number moves, and the
    failure mode is a silently empty mining pool rather than an error.
    """
    literal = decoys.read_config(DECOYS_CONF).mining_window_nt + mining_pool.DEFAULT_FLANK_MARGIN_NT
    assert literal == 1074  # the value that must NOT appear, computed independently
    for path in (DATA_SMK, COMMON_SMK):
        text = _read(path)
        assert not re.search(rf"(?<![\w.]){literal}(?![\w.])", text), (
            f"{literal} appears literally in {path.name} — the flank pad must stay "
            "derived from conf/data/decoys.yaml via common.smk::flank_pad_nt()"
        )


def test_the_mining_pool_rule_passes_the_split_table() -> None:
    """Without ``--split-table`` the builder falls back to its own default path.

    Parent-fold stamping is what makes admissibility data on the window rather
    than a promise by whoever loads it; a rule that never passes the table hands
    the gate a default that may not be the table the run was configured with.
    """
    smk = _read(DATA_SMK)
    assert "--split-table {params.split_table:q}" in _shell_block(smk, "build_mining_pool")
    block = _rule_block(smk, "build_mining_pool")
    assert "split_table=config.get(" in block
    assert mining_pool.DEFAULT_SPLIT_TABLE in block
