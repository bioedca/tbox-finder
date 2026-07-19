"""P2-06 — the sweep result, the promotion, and the overlay that carries it.

`tests/unit/test_selection_val.py` already proves `select_best` behaves correctly on
hand-built points. Nothing proved it behaves correctly on the **real 36**, and nothing tied
the `conf/` overlay to the winner those reports name. Both gaps are the same shape: the
selection logic can be perfect while the thing actually shipped — a YAML file typed by hand
from a number read off a terminal — disagrees with it. A config is promoted by the file
that gets loaded, not by the reducer that ranked it.

Tiering follows the P1-16 split the CI comments spell out: the committed-report tier is pure
JSON/text and is ARMED in CI (`TBOX_REQUIRE_SWEEP_SELECTION=1`), while the Hydra composition
tier carries its own var (`TBOX_REQUIRE_SWEEP_HYDRA`, deliberately NOT armed — CI installs no
hydra). One var guarding both tiers could not be armed at all.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tbox_finder.train import select_best as SB

_REPO = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO / SB.DEFAULT_SWEEP_DIR
_ARTIFACT = _REPO / SB.DEFAULT_OUT
_OVERLAY = _REPO / "conf/train/stage1_best.yaml"
_OPTIM_OVERLAY = _REPO / "conf/optim/stage1_best.yaml"

#: The grid P2-06 swept: γ{0.5,1,2,3} × lr{3e-5,1e-4,3e-4} × α{0,0.25,0.5}.
_EXPECT_POINTS = 36


def _fail_or_skip(var: str, reason: str) -> None:
    if os.environ.get(var) == "1":
        pytest.fail(f"{var}=1 but the tier is unrunnable: {reason}")
    pytest.skip(reason)


def _require_artifact() -> dict:
    if not _ARTIFACT.is_file():
        _fail_or_skip("TBOX_REQUIRE_SWEEP_SELECTION", f"no committed artifact at {_ARTIFACT}")
    return json.loads(_ARTIFACT.read_text())


def _report_paths() -> list[Path]:
    paths = sorted(_SWEEP_DIR.glob("*.json"))
    if not paths:
        _fail_or_skip("TBOX_REQUIRE_SWEEP_SELECTION", f"no committed reports under {_SWEEP_DIR}")
    return paths


def _top_level_scalar(path: Path, key: str) -> str:
    """The value of a top-level ``key: value`` line in a hand-authored YAML file.

    Deliberately text-based rather than a YAML parse: CI installs neither PyYAML nor Hydra,
    and this tier must stay arm-able there. Anchored at column 0 so a commented mention
    (``# lr — SELECTED…``) or an indented key inside another block cannot match, and it
    requires EXACTLY one hit — two definitions of the same key is itself the bug, and
    "first match wins" would hide it.
    """
    hits = re.findall(rf"^{re.escape(key)}:[ \t]*(\S+)", path.read_text(), flags=re.MULTILINE)
    assert len(hits) == 1, f"expected exactly one top-level `{key}:` in {path.name}, got {hits}"
    return hits[0]


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — the committed sweep (pure JSON/text; armed in CI)
# ═══════════════════════════════════════════════════════════════════════════
def test_the_committed_sweep_is_complete_and_every_point_is_promotable() -> None:
    """36/36, and none of them rejected — the sweep the selection claims to have reduced."""
    paths = _report_paths()
    assert len(paths) == _EXPECT_POINTS, f"expected {_EXPECT_POINTS} points, found {len(paths)}"
    out = SB.select_best(SB.load_reports(paths))
    assert out["n_rejected"] == 0, f"unpromotable points: {out['rejected']}"
    assert out["n_promotable"] == _EXPECT_POINTS


def test_the_grid_is_the_grid_that_was_signed_off() -> None:
    """Every (γ, lr, α) cell present exactly once — 36 files could still be 12 points
    written three times, or one corner swept twice and another never run."""
    reports = SB.load_reports(_report_paths())
    cells = sorted(
        (
            r["diagnostics"]["config"]["gamma"],
            r["diagnostics"]["config"]["lr"],
            r["diagnostics"]["config"]["class_weight_alpha"],
        )
        for r in reports
    )
    expected = sorted(
        (g, lr, a)
        for g in (0.5, 1.0, 2.0, 3.0)
        for lr in (3e-5, 1e-4, 3e-4)
        for a in (0.0, 0.25, 0.5)
    )
    assert cells == expected


def test_the_committed_artifact_re_derives_from_the_committed_reports() -> None:
    """The shipped artifact must be what the code produces from the shipped evidence.

    A JSON file is editable, and a winner is one keystroke from being a different winner.
    Re-running the reducer over the same reports is the only thing that makes the committed
    numbers evidence rather than assertion.
    """
    artifact = _require_artifact()
    rebuilt = SB.build_selection_artifact(_report_paths(), expect_points=_EXPECT_POINTS)
    committed_sel, rebuilt_sel = artifact["selection"], rebuilt["selection"]

    assert committed_sel["winner"]["axes"] == rebuilt_sel["winner"]["axes"]
    assert committed_sel["winner"]["score"] == rebuilt_sel["winner"]["score"]
    assert Path(committed_sel["winner"]["point"]).name == Path(rebuilt_sel["winner"]["point"]).name
    # The whole ranking, not just its head — a reordered tail is still a corrupted record.
    assert [p["score"] for p in committed_sel["ranking"]] == [
        p["score"] for p in rebuilt_sel["ranking"]
    ]
    assert committed_sel["n_promotable"] == rebuilt_sel["n_promotable"] == _EXPECT_POINTS


def test_no_config_was_promoted_on_the_headline_holdout() -> None:
    """The §8.2-adjacent invariant for the selection rung: tuning must not touch the
    leave-one-order-out population the PRD §12:241 headline is measured on."""
    artifact = _require_artifact()
    hyg = artifact["selection"]["selection_hygiene"]
    assert hyg["n_points_measured"] == _EXPECT_POINTS, (
        "hygiene must be measured on every point — 'all clean' and 'nothing checked' render "
        "identically in any all()/max() summary"
    )
    assert hyg["all_points_zero_loo_holdout"] is True
    assert hyg["max_designated_loo_holdout_over_points"] == 0
    assert hyg["fold_scopes_seen"] == [SB.REQUIRED_FOLD_SCOPE]


def test_the_points_are_mutually_comparable() -> None:
    """A ranking across points built by different code or scored on a different fold ranks
    runs, not configs."""
    artifact = _require_artifact()
    cons = artifact["consistency"]
    assert cons["n_points"] == _EXPECT_POINTS
    assert cons["all_agree"] is True
    for field in SB.CONSISTENCY_FIELDS:
        per = cons["per_field"][field]
        assert per["n_points_carrying"] == _EXPECT_POINTS, f"{field} missing from some point"
        assert len(per["distinct"]) == 1, f"{field} disagrees across points: {per['distinct']}"


def test_the_winning_score_is_the_min_over_the_three_core_elements() -> None:
    """ADR-0004 D6: a MIN, never a mean. A mean of the same three numbers would score
    higher and rank differently, so the statistic is re-derived, not trusted."""
    artifact = _require_artifact()
    winner = artifact["selection"]["winner"]
    per_element = winner["per_element_f1"]
    assert len(per_element) == 3, f"expected the 3 GATE-4 core elements, got {sorted(per_element)}"
    assert winner["score"] == min(per_element.values())


def test_the_sweep_ran_from_a_dirty_tree_and_says_so() -> None:
    """A disclosed caveat that must not rot into silence.

    All 36 points record ``git_dirty: true``. Audited on the cluster at retrieval: the only
    TRACKED modification was ``data/processed/splits/split_assignments.parquet``, whose
    working-tree sha256 (``fb231fe4…``, 4,922,605 B) matches the committed git-LFS pointer's
    own ``oid``/``size`` exactly — the cluster has no git-lfs, so ``reset --hard`` leaves a
    132-byte pointer and the rsync that restores the real table registers as a modification
    ([[git-lfs-pointers-in-ci]]). Zero tracked ``*.py``/``*.yaml``/``*.sbatch`` differed, so
    ``git_sha 4c723fd`` does identify the code that ran.

    This test pins the FLAG, not the audit — if a future re-run lands clean reports, it
    fails and forces the caveat above to be re-checked rather than left standing as folklore.
    """
    dirty = {json.loads(p.read_text())["provenance"]["git_dirty"] for p in _report_paths()}
    assert dirty == {True}, (
        "the recorded git_dirty flags changed — re-audit the working tree and update the "
        f"caveat in this docstring and the dev-log stanza (observed: {dirty})"
    )


def test_the_winner_is_not_statistically_separated_from_its_runners_up() -> None:
    """The honesty gate on the claim the overlay header makes.

    `stage1_best.yaml` tells its reader the win is a ranking, not a demonstrated
    superiority. That is a factual claim about these numbers, so it is tested: the winner's
    95% block-bootstrap CI must overlap the runner-up's. If a future sweep ever DOES
    separate them, this fails — and the header stops under-claiming.
    """
    artifact = _require_artifact()
    ranking = artifact["selection"]["ranking"]
    winner, runner_up = ranking[0], ranking[1]
    assert winner["ci"]["lower"] <= runner_up["ci"]["upper"], "CIs are disjoint"
    assert runner_up["ci"]["lower"] <= winner["ci"]["upper"], "CIs are disjoint"


# ── The overlay is only a promotion if it carries the winning values ──────────────────────
def test_the_overlay_pins_exactly_the_winning_axes() -> None:
    """The load-bearing test of this step.

    `select_best` names a winner; `conf/train/stage1_best.yaml` + `conf/optim/stage1_best.yaml`
    are typed by hand. If they drift — a digit dropped, an axis forgotten, the sweep re-run
    with a new winner and the YAML left behind — every downstream run trains a config the
    sweep never selected while every document says otherwise. This is not a tautology: one
    side is hand-authored YAML, the other is measured data.
    """
    winner_axes = _require_artifact()["selection"]["winner"]["axes"]
    assert float(_top_level_scalar(_OVERLAY, "gamma")) == winner_axes["gamma"]
    assert (
        float(_top_level_scalar(_OVERLAY, "class_weight_alpha"))
        == winner_axes["class_weight_alpha"]
    )
    # lr lives in the optim GROUP, never in the train primary: _cfg_from_mapping fills
    # top-level fields first and the group loop second, so a top-level `lr:` here would be
    # silently overwritten by cfg.optim.lr and the run would train at the group's value.
    assert float(_top_level_scalar(_OPTIM_OVERLAY, "lr")) == winner_axes["lr"]
    assert not re.search(r"^lr:", _OVERLAY.read_text(), flags=re.MULTILINE), (
        "a top-level `lr:` in the train overlay is silently overwritten by cfg.optim.lr — "
        "set the learning rate through the optim group only"
    )


def test_the_overlay_inherits_rather_than_forks_the_entrypoint() -> None:
    """A forked config means fixing every later Stage-1 change twice and shipping the bug in
    whichever copy was missed ([[promote-dont-duplicate-is-a-correctness-rule]])."""
    text = _OVERLAY.read_text()
    # `(?:\s*#.*)?$` — the defaults entries carry trailing comments; the value itself must
    # still end there, so `- /train: stage1_smoke` cannot satisfy the `stage1` pattern.
    assert re.search(
        r"^\s*-\s*/train:\s*stage1(?:\s*#.*)?$", text, flags=re.MULTILINE
    ), "the overlay must inherit conf/train/stage1.yaml through the defaults list"
    assert re.search(
        r"^\s*-\s*override\s+/optim:\s*stage1_best(?:\s*#.*)?$", text, flags=re.MULTILINE
    ), "re-selecting a group the inherited config already chose requires `override`"
    assert text.startswith(
        "# @package _global_"
    ), "a conf/<group>/ primary without @package _global_ silently fails to compose"


# ── The artifact's own guards must bite ───────────────────────────────────────────────────
def test_the_expect_points_guard_bites_on_a_partial_glob() -> None:
    """A 4-of-36 glob yields an internally flawless summary of the wrong evidence."""
    paths = _report_paths()[:4]
    with pytest.raises(ValueError, match="4 but 36 were expected"):
        SB.build_selection_artifact(paths, expect_points=_EXPECT_POINTS)


def test_the_consistency_guard_bites_when_a_point_came_from_different_code(tmp_path) -> None:
    reports = SB.load_reports(_report_paths())
    reports[0]["provenance"]["git_sha"] = "deadbeef" * 5
    cons = SB.cross_point_consistency(reports)
    assert cons["all_agree"] is False
    assert cons["per_field"]["git_sha"]["agrees"] is False
    assert len(cons["per_field"]["git_sha"]["distinct"]) == 2


def test_an_empty_sweep_is_not_a_clean_sweep() -> None:
    """[[clauses-must-guard-emptiness]] — the absence branch must FAIL, not pass vacuously.

    Zero points trivially satisfies "every point is clean", "all fields agree" and "no point
    touched the holdout". Asserting on the fields is not enough; the GATE has to reject.
    """
    cons = SB.cross_point_consistency([])
    assert cons["all_agree"] is False, "a sweep of nothing is not a consistent sweep"
    with pytest.raises(ValueError) as exc:
        SB.build_selection_artifact([], expect_points=0)
    msg = str(exc.value)
    assert "n_promotable" in msg and "n_points_measured" in msg


def test_a_missing_expect_points_is_not_defaultable() -> None:
    """The count must be stated by the caller — a default would re-introduce exactly the
    partial-glob hole the argument exists to close."""
    with pytest.raises(TypeError):
        SB.build_selection_artifact(_report_paths())  # type: ignore[call-arg]


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Hydra composition (TBOX_REQUIRE_SWEEP_HYDRA; CI installs no hydra)
# ═══════════════════════════════════════════════════════════════════════════
def _require_hydra():
    try:
        from hydra import compose, initialize_config_dir  # noqa: F401
    except ImportError as exc:
        _fail_or_skip("TBOX_REQUIRE_SWEEP_HYDRA", f"hydra not importable: {exc}")
    from hydra import compose, initialize_config_dir

    return compose, initialize_config_dir


def test_the_overlay_composes_and_the_winning_axes_reach_the_dataclass() -> None:
    """The end of the chain: not "the YAML says 3e-4" but "the run trains at 3e-4".

    Reading the composed dict alone would prove only that Hydra read the file. The values
    are followed through `_cfg_from_mapping` — the group loop that silently overwrites a
    top-level `lr` — into the dataclass the training loop is actually built from.
    """
    compose, initialize_config_dir = _require_hydra()
    from omegaconf import OmegaConf

    from tbox_finder.train import train_stage1 as T

    with initialize_config_dir(version_base=None, config_dir=str(_REPO / "conf")):
        raw = OmegaConf.to_container(compose(config_name="train/stage1_best"), resolve=True)
    cfg = T._cfg_from_mapping(raw)

    winner_axes = _require_artifact()["selection"]["winner"]["axes"]
    assert cfg.gamma == winner_axes["gamma"]
    assert cfg.lr == winner_axes["lr"]
    assert cfg.class_weight_alpha == winner_axes["class_weight_alpha"]
    # The overlay must not have cost us anything the inherited entrypoint pins.
    assert cfg.window_nt == 1024 and cfg.stride_nt == 512, "ADR-0005 D3+A3 geometry"
    assert cfg.rc_combine == "concat" and cfg.wandb_mode == "offline"
    assert cfg.seed == 42 and cfg.gradient_checkpointing is True
    assert (
        cfg.eval_val is True and cfg.eval_max_records is None
    ), "a capped eval records full_fold: false and select_best rejects the point"
    # And it must not overwrite the committed P2-04 smoke measurement.
    assert cfg.report_path != T.DEFAULT_REPORT
