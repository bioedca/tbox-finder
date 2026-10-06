"""P3-15′-recon-b — the ADR-0006 A5 checkpoint → phase → gate → release map.

The committed ADR table is validated against each checkpoint's own ``provenance.json``,
reached through the committed ``.dvc`` pointers and the md5-addressed mini-cache under
``tests/fixtures/checkpoint_map``. Every check in ``validate_map`` is then broken alone, with
the table consistent everywhere else, and must name its own problem. Weights depth (re-hashing
the weight blobs) is exercised on a synthetic repo here and on the real cache by the
existence-guarded test at the bottom.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from tbox_finder import checkpoint_map as M

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO / M.FIXTURE_CACHE
_ADR = (_REPO / M.ADR_PATH).read_text(encoding="utf-8")
_TRACKED = M.git_tracked(_REPO)

_PROD = "data/processed/checkpoints/stage1_production"
_TWIN = "data/processed/checkpoints/stage1_gate4_twin"
_SHIPPED_S2 = "data/processed/checkpoints/stage2_rinalmo/aux1.0_lr1e-4"


def _validate(adr_text: str = _ADR, cache: Path = _FIXTURE, tracked=None, **kw) -> list[str]:
    return M.validate_map(M.parse_map(adr_text), _REPO, cache, tracked or _TRACKED, **kw)


def _row_line(adr_text: str, checkpoint: str) -> str:
    (line,) = [ln for ln in adr_text.splitlines() if ln.startswith(f"| `{checkpoint}` |")]
    return line


def _edit_cell(adr_text: str, checkpoint: str, col: int, new: str) -> str:
    line = _row_line(adr_text, checkpoint)
    cells = M._split_row(line)
    assert cells[col] != new, "the edit must change the cell"
    cells[col] = new
    return adr_text.replace(line, "| " + " | ".join(cells) + " |")


# ------------------------------------------------------------------ the committed map


def test_committed_map_rederives_from_provenance():
    assert _validate() == []


def test_the_map_is_parsed_non_vacuously():
    rows = M.parse_map(_ADR)
    by = {r.checkpoint: r for r in rows}
    # Positive control: the rows the rest of this file edits are really parsed, and every
    # non-future row carries at least one weight digest for the validator to compare.
    assert {_PROD, _TWIN, _SHIPPED_S2} <= set(by)
    assert all(r.weights for r in rows if not r.declared_future)
    assert by[_PROD].ships == "yes" and by[_TWIN].ships == "no"
    # Stage-2 adapters live one directory down, so their key is a path, not a basename.
    assert "lora_adapter/adapter_model.safetensors" in by[_SHIPPED_S2].weights


def test_weight_outputs_key_by_the_path_inside_the_checkpoint():
    """The manifest addresses ``lora_adapter/adapter_model.safetensors``, not its basename."""
    pointer = M.owning_pointer(_SHIPPED_S2, M.committed_pointers(_REPO, _TRACKED))
    prov, blobs = M.load_provenance(_REPO, _FIXTURE, _SHIPPED_S2, pointer)
    keys = set(M.weight_outputs(prov, _SHIPPED_S2))
    assert keys == {"lora_adapter/adapter_model.safetensors", "stage2_heads.pt"}
    assert {f"{_SHIPPED_S2}/{k}" for k in keys} <= set(blobs)


def test_every_committed_checkpoint_pointer_is_in_scope():
    pointers = M.committed_pointers(_REPO, _TRACKED)
    # Without this, a CHECKPOINT_ROOTS typo would make completeness vacuous.
    assert {p.dvc_file for p in pointers} >= {
        "checkpoints/p1/seg_smoke/seg_smoke.pt.dvc",
        "data/processed/checkpoints/stage1_production.dvc",
        "data/processed/checkpoints/stage2_rinalmo.dvc",
    }


def test_fixture_is_exactly_the_manifests_and_provenance_blobs():
    """The mini-cache holds exactly what manifest depth reads, each blob self-addressed."""
    on_disk = {p.relative_to(_FIXTURE).as_posix() for p in _FIXTURE.rglob("*") if p.is_file()}
    needed: set[str] = set()
    for p in M.committed_pointers(_REPO, _TRACKED):
        if not p.is_dir:
            continue
        needed.add(f"files/md5/{p.md5[:2]}/{p.md5[2:]}.dir")
        for rel, md5 in M.dir_manifest(_FIXTURE, p).items():
            if rel.endswith("provenance.json"):
                needed.add(f"files/md5/{md5[:2]}/{md5[2:]}")
    assert on_disk == needed
    for rel in on_disk:  # read_blob re-checks every md5
        md5 = rel.split("/")[2] + rel.split("/")[3].removesuffix(".dir")
        M.read_blob(_FIXTURE, md5, suffix=".dir" if rel.endswith(".dir") else "")
    # No weight blob was copied into the public repo.
    assert max((_FIXTURE / r).stat().st_size for r in on_disk) < 64 * 1024


# ------------------------------------------------- each check broken alone, names itself


def test_a_wrong_weight_digest_is_refused():
    row = M.parse_map(_ADR)[[r.checkpoint for r in M.parse_map(_ADR)].index(_PROD)]
    (name, digest) = next(iter(row.weights.items()))
    flipped = ("0" if digest[0] != "0" else "1") + digest[1:]
    bad = _edit_cell(_ADR, _PROD, 3, f"`{name}` = `{flipped}`")
    problems = _validate(bad)
    assert len(problems) == 1 and problems[0].startswith(f"{_PROD}: Weights")


def test_an_omitted_weight_is_refused():
    line = _row_line(_ADR, _SHIPPED_S2)
    cell = M._split_row(line)[3]
    bad = _edit_cell(_ADR, _SHIPPED_S2, 3, cell.split("<br>")[0])
    problems = _validate(bad)
    assert len(problems) == 1 and problems[0].startswith(f"{_SHIPPED_S2}: Weights")


def test_a_wrong_step_is_refused():
    bad = _edit_cell(_ADR, _TWIN, 1, "P2-10d'-b")
    assert _validate(bad) == [f"{_TWIN}: Step \"P2-10d'-b\" != provenance extra.step 'P2-14'"]


def test_a_wrong_dvc_md5_is_refused():
    bad = _edit_cell(_ADR, _TWIN, 2, "`" + "0" * 32 + "`")
    problems = _validate(bad)
    assert len(problems) == 1 and problems[0].startswith(f"{_TWIN}: DVC md5")


def test_ships_yes_on_a_checkpoint_provenance_says_is_not_shipped():
    bad = _edit_cell(_ADR, _TWIN, 5, "yes")
    assert _validate(bad) == [f"{_TWIN}: Ships 'yes' but its provenance records it as not shipped"]


def test_a_comparator_cannot_be_marked_shipped():
    rnafm = "data/processed/checkpoints/stage2_rnafm/aux1.0_lr1e-4"
    assert _validate(_edit_cell(_ADR, rnafm, 5, "yes")) == [
        f"{rnafm}: Ships 'yes' but its provenance records it as not shipped"
    ]


def test_a_missing_row_fails_completeness():
    bad = _ADR.replace(_row_line(_ADR, _TWIN) + "\n", "")
    assert _validate(bad) == [f"{_TWIN}: has a provenance.json in {_TWIN}.dvc but no row"]


def test_a_missing_sweep_point_row_fails_completeness():
    point = "data/processed/checkpoints/stage2_rinalmo/aux0.5_lr3e-4"
    bad = _ADR.replace(_row_line(_ADR, point) + "\n", "")
    assert _validate(bad) == [
        f"{point}: has a provenance.json in {Path(point).parent.as_posix()}.dvc but no row"
    ]


def test_a_missing_single_file_pointer_row_fails_completeness():
    smoke = "checkpoints/p1/seg_smoke"
    bad = _ADR.replace(_row_line(_ADR, smoke) + "\n", "")
    assert _validate(bad) == [
        f"{smoke}/seg_smoke.pt.dvc: committed checkpoint pointer named by no row"
    ]


def test_an_untracked_pointer_neither_satisfies_nor_escapes_the_map():
    tracked = [p for p in _TRACKED if p != f"{_TWIN}.dvc"]
    assert _validate(tracked=tracked) == [f"{_TWIN}: no committed .dvc pointer tracks this path"]


def test_declared_future_must_be_absent():
    line = _row_line(_ADR, _TWIN)
    fut = line.replace(M._split_row(line)[2], f"`{M.DECLARED_FUTURE}`")
    cells = M._split_row(fut)
    cells[3] = "—"
    bad = _ADR.replace(line, "| " + " | ".join(cells) + " |")
    problems = _validate(bad)
    assert f"{_TWIN}: declared-future, but the artifact or a pointer exists" in problems


def test_declared_future_row_for_an_absent_path_is_accepted():
    """Positive control for the declared-future branch, on the real map."""
    future = "data/processed/checkpoints/stage1_future_example"
    assert not (_REPO / future).exists()
    row = f"| `{future}` | P9-99 | `{M.DECLARED_FUTURE}` | — | nothing yet | no |"
    good = _ADR.replace(M.MAP_END, row + "\n" + M.MAP_END)
    assert len(M.parse_map(good)) == len(M.parse_map(_ADR)) + 1
    assert _validate(good) == []
    shipped = good.replace(row, row.replace("| no |", "| yes |"))
    assert _validate(shipped) == [f"{future}: a declared-future artifact cannot ship"]


def test_a_corrupted_provenance_blob_is_refused(tmp_path):
    cache = tmp_path / "cache"
    shutil.copytree(_FIXTURE, cache)
    pointer = M.owning_pointer(_TWIN, M.committed_pointers(_REPO, _TRACKED))
    md5 = M.dir_manifest(cache, pointer)["provenance.json"]
    blob = cache / "files/md5" / md5[:2] / md5[2:]
    blob.write_bytes(blob.read_bytes().replace(b'"P2-14"', b'"P2-15"'))
    problems = _validate(cache=cache)
    assert len(problems) == 1
    assert problems[0].startswith(f"{_TWIN}: provenance unreadable") and "hashes to" in problems[0]


def test_an_absent_cache_fails_closed(tmp_path):
    problems = _validate(cache=tmp_path / "no-cache")
    assert problems and all("unreadable" in p for p in problems)
    # Every directory-pointer row, and the completeness pass, report — none is skipped.
    n_dir_rows = sum(1 for r in M.parse_map(_ADR) if not r.checkpoint.startswith("checkpoints/"))
    assert len([p for p in problems if "provenance unreadable" in p]) == n_dir_rows


# ----------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda t: t.replace(M.MAP_END, ""), "exactly one"),
        (lambda t: t + "\n" + M.MAP_BEGIN, "exactly one"),
        (lambda t: t.replace("| Checkpoint | Step |", "| Checkpoint | Stage |"), "header"),
        (
            lambda t: t.replace(
                _row_line(t, _TWIN), _row_line(t, _TWIN) + "\n" + _row_line(t, _TWIN)
            ),
            "two rows",
        ),
        (lambda t: _edit_cell(t, _TWIN, 5, "maybe"), "Ships"),
        (lambda t: _edit_cell(t, _TWIN, 2, "`7869eec1`"), "32 hex"),
        (lambda t: _edit_cell(t, _TWIN, 3, "`stage1.pt` = `0140a8a3`"), "weight item"),
        (lambda t: _edit_cell(t, _TWIN, 0, _TWIN), "Checkpoint cell"),
    ],
)
def test_malformed_tables_are_refused_while_parsing(mutate, match):
    with pytest.raises(M.MapError, match=match):
        M.parse_map(mutate(_ADR))


def test_parse_accepts_the_unmutated_table():
    """Positive control for the parametrized refusals above."""
    assert len(M.parse_map(_ADR)) >= 12


# --------------------------------------------------------- weights depth, synthetic repo


def _md5(b: bytes) -> str:
    return hashlib.md5(b, usedforsecurity=False).hexdigest()


def _put(cache: Path, data: bytes, suffix: str = "") -> str:
    h = _md5(data)
    p = cache / "files/md5" / h[:2] / (h[2:] + suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return h


def _synthetic(tmp_path: Path, *, claimed_sha: str | None = None):
    """A repo with one directory pointer holding ``ck/{stage1.pt, provenance.json}``."""
    root, cache = tmp_path / "repo", tmp_path / "cache"
    ck = "data/processed/checkpoints/toy"
    weights = b"toy-weights" * 1000
    sha = claimed_sha or hashlib.sha256(weights).hexdigest()
    prov = json.dumps({"extra": {"step": "P9-01"}, "outputs": {f"{ck}/stage1.pt": sha}}).encode()
    manifest = [
        {"md5": _put(cache, prov), "relpath": "provenance.json"},
        {"md5": _put(cache, weights), "relpath": "stage1.pt"},
    ]
    dir_md5 = _put(cache, json.dumps(manifest).encode(), ".dir")
    (root / "data/processed/checkpoints").mkdir(parents=True)
    (root / f"{ck}.dvc").write_text(f"outs:\n- md5: {dir_md5}.dir\n  nfiles: 2\n  path: toy\n")
    row = f"| `{ck}` | P9-01 | `{dir_md5}` | `stage1.pt` = `{sha}` | toy gate | no |"
    adr = "\n".join(
        [
            M.MAP_BEGIN,
            "| " + " | ".join(M.COLUMNS) + " |",
            "|---|---|---|---|---|---|",
            row,
            M.MAP_END,
        ]
    )
    return root, cache, adr, [f"{ck}.dvc"], weights


def test_weights_depth_rehashes_every_blob(tmp_path):
    root, cache, adr, tracked, _ = _synthetic(tmp_path)
    stats: dict[str, int] = {}
    assert (
        M.validate_map(M.parse_map(adr), root, cache, tracked, verify_weights=True, stats=stats)
        == []
    )
    assert stats == {"rehashed": 1}


def test_weights_depth_refuses_a_false_provenance_claim(tmp_path):
    lie = "f" * 64
    root, cache, adr, tracked, weights = _synthetic(tmp_path, claimed_sha=lie)
    # Manifest depth cannot see the lie: the row agrees with provenance.
    assert M.validate_map(M.parse_map(adr), root, cache, tracked) == []
    stats: dict[str, int] = {}
    problems = M.validate_map(
        M.parse_map(adr), root, cache, tracked, verify_weights=True, stats=stats
    )
    real = hashlib.sha256(weights).hexdigest()
    assert problems == [
        f"data/processed/checkpoints/toy: stage1.pt hashes to {real}, provenance says {lie}"
    ]
    assert stats == {"rehashed": 0}


def test_a_row_without_weights_on_a_weightless_provenance_is_refused(tmp_path):
    """Row and provenance agree on 'no weights', so only the emptiness clause can refuse."""
    root, cache, adr, tracked, _ = _synthetic(tmp_path)
    ck = "data/processed/checkpoints/toy"
    pointer = M.read_pointer(root, f"{ck}.dvc")
    entries = json.loads(M.read_blob(cache, pointer.md5, suffix=".dir"))
    prov = json.dumps({"extra": {"step": "P9-01"}, "outputs": {}}).encode()
    entries[0]["md5"] = _put(cache, prov)
    new_dir = _put(cache, json.dumps(entries).encode(), ".dir")
    (root / f"{ck}.dvc").write_text(f"outs:\n- md5: {new_dir}.dir\n  nfiles: 2\n  path: toy\n")
    line = [ln for ln in adr.splitlines() if ln.startswith(f"| `{ck}`")][0]
    cells = M._split_row(line)
    cells[2], cells[3] = f"`{new_dir}`", "—"
    adr = adr.replace(line, "| " + " | ".join(cells) + " |")
    assert M.validate_map(M.parse_map(adr), root, cache, tracked) == [
        f"{ck}: provenance names no weight output under the checkpoint"
    ]


def test_weights_depth_refuses_a_missing_weight_blob(tmp_path):
    root, cache, adr, tracked, weights = _synthetic(tmp_path)
    h = _md5(weights)
    (cache / "files/md5" / h[:2] / h[2:]).unlink()
    problems = M.validate_map(M.parse_map(adr), root, cache, tracked, verify_weights=True)
    assert len(problems) == 1 and "stage1.pt unreadable from the cache" in problems[0]


def test_single_file_pointer_needs_a_tracked_provenance(tmp_path):
    root = tmp_path / "repo"
    ck = "checkpoints/p9/toy"
    (root / ck).mkdir(parents=True)
    weights = b"w" * 10
    sha = hashlib.sha256(weights).hexdigest()
    (root / ck / "toy.pt.dvc").write_text(f"outs:\n- md5: {_md5(weights)}\n  path: toy.pt\n")
    (root / ck / "provenance.json").write_text(
        json.dumps({"extra": {"step": "P9-02"}, "outputs": {f"{ck}/toy.pt": sha}})
    )
    row = f"| `{ck}` | P9-02 | `{_md5(weights)}` | `toy.pt` = `{sha}` | toy | no |"
    adr = "\n".join(
        [
            M.MAP_BEGIN,
            "| " + " | ".join(M.COLUMNS) + " |",
            "|---|---|---|---|---|---|",
            row,
            M.MAP_END,
        ]
    )
    rows = M.parse_map(adr)
    tracked = [f"{ck}/toy.pt.dvc", f"{ck}/provenance.json"]
    assert M.validate_map(rows, root, tmp_path / "cache", tracked) == []
    assert M.validate_map(rows, root, tmp_path / "cache", tracked[:1]) == [
        f"{ck}: single-file pointer without a git-tracked provenance.json"
    ]
    _put(tmp_path / "cache", weights)
    stats: dict[str, int] = {}
    assert (
        M.validate_map(rows, root, tmp_path / "cache", tracked, verify_weights=True, stats=stats)
        == []
    )
    assert stats == {"rehashed": 1}


def test_read_pointer_refuses_a_multi_output_file(tmp_path):
    (tmp_path / "x.dvc").write_text(
        "outs:\n- md5: " + "a" * 32 + "\n  path: a\n- md5: " + "b" * 32 + "\n  path: b\n"
    )
    with pytest.raises(M.MapError, match="expected one md5 and one path"):
        M.read_pointer(tmp_path, "x.dvc")


# ------------------------------------------------- weights depth on the real DVC cache

_REAL_CACHE = Path(os.environ.get("TBOX_DVC_CACHE", _REPO / ".dvc" / "cache"))
_PROBE = _REAL_CACHE / "files/md5/91/bd49dd91997b8a94b6ba262afab071.dir"
_REQUIRE = os.environ.get("TBOX_REQUIRE_DVC_CACHE") == "1"


@pytest.mark.skipif(
    not _PROBE.exists() and not _REQUIRE,
    reason="real DVC cache absent (CI does no dvc pull); TBOX_REQUIRE_DVC_CACHE=1 makes this fail",
)
def test_committed_map_weights_rehash_on_the_real_cache():
    stats: dict[str, int] = {}
    assert _validate(cache=_REAL_CACHE, verify_weights=True, stats=stats) == []
    assert stats["rehashed"] == sum(len(r.weights) for r in M.parse_map(_ADR))


def test_the_adr_names_the_validator():
    """The amendment's prose and the code agree on where the map lives and who checks it."""
    a5 = _ADR.split("## Amendment A5", 1)[1]
    assert "checkpoint_map.py::validate_map" in a5
    assert re.search(r"tests/unit/test_checkpoint_map\.py", a5)
