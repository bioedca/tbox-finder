"""checkpoint_map.py — the ADR-0006 A5 checkpoint → phase → gate → release map, re-derived.

PRD §18.3 delegates the *checkpoint → phase → gate → release map* to ADR-0006; Amendment A5
writes it as a table between two HTML-comment markers. This module is the map's validator,
and it restates nothing. Each row's producing step and weight digests are **re-read from the
checkpoint's own** ``provenance.json``, reached through the committed ``.dvc`` pointer and
the DVC cache, and every blob read is checked against the md5 that addresses it. So a row is
bound to git through one hash chain:

    ``<ptr>.dvc`` (git) → ``md5.dir`` manifest → ``provenance.json`` blob → ``outputs`` sha256

A single-file pointer (the P1 smoke) keeps its ``provenance.json`` in git beside the pointer,
so the chain there is the git blob itself.

Two depths:

* **manifest depth** (``verify_weights=False``) needs only the ``.dir`` manifests and the
  ``provenance.json`` blobs. CI runs at this depth against a committed mini-cache
  (``tests/fixtures/checkpoint_map/dvc_cache``), so the map is re-derived in CI rather than
  skipped there.
* **weights depth** (``verify_weights=True``) also re-hashes every weight blob (≈0.5 GB)
  and checks provenance's sha256 claim about it. This needs the real cache, so it runs
  locally as the step's validation gate.

Completeness is checked as well. Every committed checkpoint pointer must have a row, and so
must every ``provenance.json`` inside a directory pointer, so a new sweep point cannot be left
out of the map without the check failing.

Stdlib-only, like :mod:`tbox_finder.provenance`, so it imports in the bare CI test env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: The ADR file that carries the map (PRD §18.3's delegatee).
ADR_PATH = "docs/decisions/ADR-0006-validation-decision-rule-and-tiering.md"
MAP_BEGIN = "<!-- checkpoint-map:begin -->"
MAP_END = "<!-- checkpoint-map:end -->"

#: The table's columns, in order. Parsing refuses any other header.
COLUMNS = ("Checkpoint", "Step", "DVC md5", "Weights (sha256)", "Graded by", "Ships")

#: Where trained checkpoints live. Every row must name a path under one of these roots, and
#: every committed ``.dvc`` pointer under them must be named by a row.
CHECKPOINT_ROOTS = ("data/processed/checkpoints", "checkpoints")

#: File suffixes that count as weights. Every provenance output with one of these suffixes
#: under a row's checkpoint must be named in that row's Weights cell, and every tracked file
#: with one must be named by some row.
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".safetensors")

#: The Step/DVC-md5 cell value of a row whose artifact does not exist yet.
DECLARED_FUTURE = "declared-future"

#: The committed CI mini-cache (``.dir`` manifests + ``provenance.json`` blobs only).
FIXTURE_CACHE = "tests/fixtures/checkpoint_map/dvc_cache"

_MD5 = re.compile(r"^[0-9a-f]{32}$")
_WEIGHT_ITEM = re.compile(r"^`([^`]+)`\s*=\s*`([0-9a-f]{64})`$")
_CODE = re.compile(r"^`([^`]+)`$")


@dataclass(frozen=True)
class MapRow:
    """One parsed table row. ``weights`` maps a file's path inside the checkpoint → sha256."""

    checkpoint: str
    step: str
    dvc_md5: str
    weights: dict[str, str]
    graded_by: str
    ships: str

    @property
    def declared_future(self) -> bool:
        return self.dvc_md5 == DECLARED_FUTURE


@dataclass(frozen=True)
class Pointer:
    """A committed single-output ``.dvc`` pointer."""

    dvc_file: str  # repo-relative path of the .dvc file
    out_path: str  # repo-relative path of the tracked file or directory
    md5: str  # bare 32-hex md5, without the ``.dir`` suffix
    is_dir: bool


class MapError(ValueError):
    """The map table is malformed (refused while parsing, before any validation)."""


# --------------------------------------------------------------------------- parsing


def _cell_code(cell: str, what: str) -> str:
    m = _CODE.match(cell)
    if not m:
        raise MapError(f"{what} cell must be one backticked value, got {cell!r}")
    return m.group(1)


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        raise MapError(f"not a table row: {line!r}")
    return [c.strip() for c in s[1:-1].split("|")]


def parse_map(adr_text: str) -> list[MapRow]:
    """Parse the A5 table between :data:`MAP_BEGIN` and :data:`MAP_END`.

    Raises :class:`MapError` on a missing or duplicated marker, a wrong header, a malformed
    cell, or a duplicated checkpoint. A malformed table is refused here rather than reported
    as a validation problem, because nothing downstream can be trusted to read it.
    """
    if adr_text.count(MAP_BEGIN) != 1 or adr_text.count(MAP_END) != 1:
        raise MapError("the ADR must carry exactly one begin and one end checkpoint-map marker")
    body = adr_text.split(MAP_BEGIN, 1)[1].split(MAP_END, 1)[0]
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise MapError("the checkpoint map has no rows")
    header = tuple(_split_row(lines[0]))
    if header != COLUMNS:
        raise MapError(f"header must be {COLUMNS}, got {header}")
    if not all(set(c) <= set("-: ") and c for c in _split_row(lines[1])):
        raise MapError("second line must be the table's delimiter row")
    rows: list[MapRow] = []
    for line in lines[2:]:
        cells = _split_row(line)
        if len(cells) != len(COLUMNS):
            raise MapError(f"row has {len(cells)} cells, expected {len(COLUMNS)}: {line!r}")
        checkpoint = _cell_code(cells[0], "Checkpoint")
        step = cells[1]
        dvc_md5 = _cell_code(cells[2], "DVC md5")
        if dvc_md5 != DECLARED_FUTURE and not _MD5.match(dvc_md5):
            raise MapError(f"{checkpoint}: DVC md5 must be 32 hex or {DECLARED_FUTURE!r}")
        weights: dict[str, str] = {}
        if cells[3] not in ("—", "-"):
            for item in cells[3].split("<br>"):
                m = _WEIGHT_ITEM.match(item.strip())
                if not m:
                    raise MapError(f"{checkpoint}: weight item must be `name` = `sha256`: {item!r}")
                if m.group(1) in weights:
                    raise MapError(f"{checkpoint}: weight {m.group(1)!r} named twice")
                weights[m.group(1)] = m.group(2)
        ships = cells[5]
        if ships not in ("yes", "no"):
            raise MapError(f"{checkpoint}: Ships must be 'yes' or 'no', got {ships!r}")
        if any(r.checkpoint == checkpoint for r in rows):
            raise MapError(f"{checkpoint}: named by two rows")
        rows.append(MapRow(checkpoint, step, dvc_md5, weights, cells[4], ships))
    return rows


# ------------------------------------------------------------------- DVC + provenance


def read_pointer(repo_root: Path, dvc_file: str) -> Pointer:
    """Parse a committed single-output ``.dvc`` pointer (md5 + path; nothing else is read).

    Not :func:`tbox_finder.mining.mine_round.read_dvc_dir_pointer`: that reader accepts
    **directory** pointers only, by contract, and the P1 smoke checkpoint is a single-file
    pointer. This one reads both shapes and refuses a multi-output file rather than guessing.
    """
    text = (repo_root / dvc_file).read_text(encoding="utf-8")
    md5s = re.findall(r"^\s*-?\s*md5:\s*([0-9a-f]{32})(\.dir)?\s*$", text, flags=re.M)
    paths = re.findall(r"^\s*-?\s*path:\s*(\S+)\s*$", text, flags=re.M)
    if len(md5s) != 1 or len(paths) != 1:
        raise MapError(f"{dvc_file}: expected one md5 and one path, got {len(md5s)}/{len(paths)}")
    out = (Path(dvc_file).parent / paths[0]).as_posix()
    return Pointer(dvc_file, out, md5s[0][0], bool(md5s[0][1]))


def committed_pointers(repo_root: Path, tracked: list[str]) -> list[Pointer]:
    """Every tracked ``.dvc`` pointer under :data:`CHECKPOINT_ROOTS`."""
    roots = tuple(r.rstrip("/") + "/" for r in CHECKPOINT_ROOTS)
    return [
        read_pointer(repo_root, p)
        for p in sorted(tracked)
        if p.endswith(".dvc") and p.startswith(roots)
    ]


def owning_pointer(checkpoint: str, pointers: list[Pointer]) -> Pointer | None:
    """The pointer that tracks ``checkpoint``: itself, or the directory pointer above it."""
    hits = [
        p
        for p in pointers
        if checkpoint == p.out_path
        or (p.is_dir and checkpoint.startswith(p.out_path + "/"))
        or (not p.is_dir and Path(p.out_path).parent.as_posix() == checkpoint)
    ]
    if len(hits) > 1:
        raise MapError(
            f"{checkpoint}: owned by more than one pointer: {[p.dvc_file for p in hits]}"
        )
    return hits[0] if hits else None


def read_blob(cache_dir: Path, md5: str, *, suffix: str = "") -> bytes:
    """Read a DVC-3 cache blob (``files/md5/xx/rest``) and check that it hashes to ``md5``."""
    path = cache_dir / "files" / "md5" / md5[:2] / (md5[2:] + suffix)
    data = path.read_bytes()
    got = hashlib.md5(data, usedforsecurity=False).hexdigest()
    if got != md5:
        raise MapError(f"cache blob {path} hashes to {got}, not {md5}")
    return data


def dir_manifest(cache_dir: Path, pointer: Pointer) -> dict[str, str]:
    """``relpath → md5`` for a directory pointer, from its md5-checked ``.dir`` manifest."""
    entries = json.loads(read_blob(cache_dir, pointer.md5, suffix=".dir"))
    manifest = {e["relpath"]: e["md5"] for e in entries}
    if len(manifest) != len(entries):
        raise MapError(f"{pointer.dvc_file}: .dir manifest repeats a relpath")
    return manifest


def _member(checkpoint: str, pointer: Pointer) -> str:
    """``checkpoint``'s path inside a directory pointer ('' when it is the pointer itself)."""
    return "" if checkpoint == pointer.out_path else checkpoint[len(pointer.out_path) + 1 :]


def load_provenance(
    repo_root: Path, cache_dir: Path, checkpoint: str, pointer: Pointer
) -> tuple[dict, dict[str, str]]:
    """Return ``(provenance, blobs)`` for ``checkpoint``.

    ``blobs`` maps each file of the checkpoint (repo-relative) to its DVC md5, which is what
    weights depth re-hashes. A directory pointer's provenance is read out of the cache through
    the manifest; a single-file pointer's provenance is the git file beside the pointer.
    """
    if pointer.is_dir:
        manifest = dir_manifest(cache_dir, pointer)
        member = _member(checkpoint, pointer)
        prefix = member + "/" if member else ""
        rel = prefix + "provenance.json"
        if rel not in manifest:
            raise MapError(f"{checkpoint}: no provenance.json in {pointer.dvc_file}'s manifest")
        prov = json.loads(read_blob(cache_dir, manifest[rel]))
        blobs = {
            f"{pointer.out_path}/{r}": m
            for r, m in manifest.items()
            if r.startswith(prefix) and r != rel
        }
        return prov, blobs
    prov = json.loads((repo_root / checkpoint / "provenance.json").read_text(encoding="utf-8"))
    return prov, {pointer.out_path: pointer.md5}


def weight_outputs(prov: dict, checkpoint: str) -> dict[str, str]:
    """Provenance ``outputs`` that are weights under ``checkpoint``.

    Keyed by the path **inside** the checkpoint (``lora_adapter/adapter_model.safetensors``),
    not the basename, because that path is what addresses the blob in the ``.dir`` manifest.
    """
    prefix = checkpoint + "/"
    return {
        path[len(prefix) :]: digest
        for path, digest in prov.get("outputs", {}).items()
        if path.startswith(prefix) and path.endswith(WEIGHT_SUFFIXES)
    }


def _provenance_says_not_shipped(prov: dict) -> bool:
    extra = prov.get("extra", {})
    shipped = str(extra.get("shipped", "")).strip().lower()
    return shipped.startswith("no") or extra.get("role") == "comparator"


def _is_number(value: object) -> bool:
    """A JSON number: ``int`` or ``float``, and never ``bool`` (a subclass of ``int``)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True)
class ShippedRule:
    """Which checkpoints ship, derived from the code that uses them rather than typed.

    ``stage1`` is the canonical Stage-1 checkpoint directory (ADR-0005 A12), the one the
    scanner loads by default. A Stage-2 row ships when it lives under ``stage2_root`` and its
    own provenance records the ``(aux_weight, lr)`` the repo's ``conf/`` ships. ``None`` for
    a stage means nothing of that stage ships (synthetic repos in tests); the production rule
    sets both, and then exactly one row of each stage must ship.
    """

    stage1: str | None
    stage2_root: str | None
    aux_weight: float
    lr: float

    def is_stage2(self, checkpoint: str) -> bool:
        return self.stage2_root is not None and checkpoint.startswith(self.stage2_root + "/")

    def ships(self, checkpoint: str, prov: dict) -> bool:
        if checkpoint == self.stage1:
            return True
        if self.is_stage2(checkpoint):
            extra = prov.get("extra", {})
            got = (extra.get("loss_aux_weight"), extra.get("optim_lr"))
            # Compared as the JSON numbers the trainer writes, never coerced: ``float("1e-4")``
            # would accept a string, and a bare ``==`` already accepts ``True`` as ``1.0``.
            return all(_is_number(v) for v in got) and got == (self.aux_weight, self.lr)
        return False


def default_shipped_rule(repo_root: Path) -> ShippedRule:
    """The rule as the shipped code defines it: ``infer.scan.DEFAULT_CHECKPOINT`` and
    ``stage2.eval.production_arm_config()``. Imported lazily, because both modules pull in
    numpy and this module is otherwise stdlib-only."""
    from tbox_finder.infer.scan import DEFAULT_CHECKPOINT
    from tbox_finder.stage2.eval import (
        DEFAULT_CKPT_ROOT,
        LOSS_CONF,
        OPTIM_CONF,
        production_arm_config,
    )

    cfg = production_arm_config(loss_conf=repo_root / LOSS_CONF, optim_conf=repo_root / OPTIM_CONF)
    return ShippedRule(
        stage1=Path(DEFAULT_CHECKPOINT).parent.as_posix(),
        stage2_root=Path(DEFAULT_CKPT_ROOT).as_posix(),
        aux_weight=cfg["aux_weight"],
        lr=cfg["lr"],
    )


def _members(cache_dir: Path, pointer: Pointer) -> list[str]:
    """Repo-relative paths of every file a pointer tracks."""
    if not pointer.is_dir:
        return [pointer.out_path]
    return [f"{pointer.out_path}/{rel}" for rel in dir_manifest(cache_dir, pointer)]


# ------------------------------------------------------------------------ validation


def validate_map(
    rows: list[MapRow],
    repo_root: Path,
    cache_dir: Path,
    tracked: list[str],
    *,
    verify_weights: bool = False,
    stats: dict[str, int] | None = None,
    rule: ShippedRule | None = None,
) -> list[str]:
    """Return every problem with ``rows``; ``[]`` means each row re-derives.

    ``tracked`` is the list of git-tracked paths (``git ls-files``). The pointer set is taken
    from it, so an untracked ``.dvc`` file can neither satisfy nor escape the map. ``rule``
    defaults to :func:`default_shipped_rule`. When ``stats`` is given, ``stats["rehashed"]``
    counts the weight blobs whose sha256 matched.
    """
    if stats is not None:
        stats["rehashed"] = 0
    rule = rule or default_shipped_rule(repo_root)
    roots = tuple(r.rstrip("/") + "/" for r in CHECKPOINT_ROOTS)
    problems: list[str] = []
    pointers = committed_pointers(repo_root, tracked)
    tracked_set = set(tracked)
    covered: set[str] = set()
    named_weights: set[str] = set()
    owned: set[str] = set()
    shipped_stage1 = shipped_stage2 = 0
    stage2_unread = False
    for row in rows:
        ck = row.checkpoint
        if Path(ck).is_absolute() or ".." in Path(ck).parts:
            problems.append(f"{ck}: checkpoint path must be repo-relative")
            continue
        if not ck.startswith(roots):
            problems.append(f"{ck}: outside the checkpoint roots {CHECKPOINT_ROOTS}")
            continue
        try:
            pointer = owning_pointer(ck, pointers)
        except MapError as exc:
            problems.append(str(exc))
            continue
        if row.declared_future:
            # A declared-future row must be genuinely absent. Once a pointer or file exists,
            # the row has to be promoted, so the map cannot hide a real artifact.
            if pointer is not None or (repo_root / ck).exists():
                problems.append(f"{ck}: declared-future, but the artifact or a pointer exists")
            if row.weights:
                problems.append(f"{ck}: declared-future rows carry no weight digests")
            if row.ships != "no":
                problems.append(f"{ck}: a declared-future artifact cannot ship")
            continue
        if pointer is None:
            problems.append(f"{ck}: no committed .dvc pointer tracks this path")
            continue
        covered.add(ck)
        owned.add(pointer.dvc_file)
        named_weights.update(f"{ck}/{w}" for w in row.weights)
        if row.dvc_md5 != pointer.md5:
            problems.append(f"{ck}: DVC md5 {row.dvc_md5} != {pointer.dvc_file}'s {pointer.md5}")
            continue
        if not pointer.is_dir and f"{ck}/provenance.json" not in tracked_set:
            problems.append(f"{ck}: single-file pointer without a git-tracked provenance.json")
            continue
        try:
            prov, blobs = load_provenance(repo_root, cache_dir, ck, pointer)
        except (OSError, MapError, ValueError, KeyError) as exc:
            problems.append(f"{ck}: provenance unreadable: {exc}")
            stage2_unread |= rule.is_stage2(ck)
            continue
        step = prov.get("extra", {}).get("step")
        if row.step != step:
            problems.append(f"{ck}: Step {row.step!r} != provenance extra.step {step!r}")
        derived = weight_outputs(prov, ck)
        if not derived:
            problems.append(f"{ck}: provenance names no weight output under the checkpoint")
        if row.weights != derived:
            problems.append(f"{ck}: Weights {row.weights} != provenance outputs {derived}")
        # Ships is derived, in both directions: the yes-set is exactly what the rule says.
        ships = rule.ships(ck, prov)
        shipped_stage1 += ships and ck == rule.stage1
        shipped_stage2 += ships and rule.is_stage2(ck)
        if (row.ships == "yes") != ships:
            want = "yes" if ships else "no"
            problems.append(f"{ck}: Ships {row.ships!r}, but the shipped rule derives {want!r}")
        if row.ships == "yes" and _provenance_says_not_shipped(prov):
            problems.append(f"{ck}: Ships 'yes' but its provenance records it as not shipped")
        for name, digest in derived.items():
            # Membership needs only the manifest, so it runs at both depths.
            md5 = blobs.get(f"{ck}/{name}")
            if md5 is None:
                problems.append(f"{ck}: {name} is not in the pointer's manifest")
                continue
            if not verify_weights:
                continue
            try:
                got = hashlib.sha256(read_blob(cache_dir, md5)).hexdigest()
            except (OSError, MapError) as exc:
                problems.append(f"{ck}: {name} unreadable from the cache: {exc}")
                continue
            if got != digest:
                problems.append(f"{ck}: {name} hashes to {got}, provenance says {digest}")
            elif stats is not None:
                stats["rehashed"] += 1

    stage1_bad = not any(p.startswith(f"{rule.stage1}:") for p in problems)
    if rule.stage1 is not None and shipped_stage1 != 1 and stage1_bad:
        problems.append(f"{rule.stage1}: the shipped Stage-1 is named by {shipped_stage1} rows")
    if rule.stage2_root is not None and shipped_stage2 != 1 and not stage2_unread:
        problems.append(
            f"{rule.stage2_root}: {shipped_stage2} rows match the shipped Stage-2 config "
            f"(aux_weight={rule.aux_weight}, lr={rule.lr}); exactly one must"
        )

    # Completeness: every committed checkpoint pointer must be named by a non-future row, and
    # so must every weight file and every provenance.json it tracks. A checkpoint whose
    # sidecar was lost (job 1064's failure mode) is caught by its weights, not its sidecar.
    for p in pointers:
        if p.dvc_file not in owned:
            problems.append(f"{p.dvc_file}: committed checkpoint pointer named by no row")
            continue
        try:
            members = _members(cache_dir, p)
        except (OSError, MapError, ValueError, KeyError) as exc:
            problems.append(f"{p.dvc_file}: manifest unreadable: {exc}")
            continue
        for path in members:
            if path.endswith(WEIGHT_SUFFIXES) and path not in named_weights:
                problems.append(f"{path}: a weight file in {p.dvc_file} that no row names")
            if path.endswith("/provenance.json") and p.is_dir:
                ck = path[: -len("/provenance.json")]
                if ck not in covered:
                    problems.append(f"{ck}: has a provenance.json in {p.dvc_file} but no row")
    return problems


# ------------------------------------------------------------------------------- CLI


def git_tracked(repo_root: Path) -> list[str]:
    """Every git-tracked path under ``repo_root`` (``git ls-files``)."""
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"], check=True, capture_output=True
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def export_fixture(rows: list[MapRow], repo_root: Path, cache_dir: Path, dest: Path) -> list[str]:
    """Copy the ``.dir`` manifests + ``provenance.json`` blobs the map needs into ``dest``.

    Every blob is md5-checked as it is read, so the fixture is a byte-exact subset of the
    real cache, and no weight blob is copied.
    """
    pointers = committed_pointers(repo_root, git_tracked(repo_root))
    written: list[str] = []

    def put(md5: str, suffix: str) -> None:
        target = dest / "files" / "md5" / md5[:2] / (md5[2:] + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read_blob(cache_dir, md5, suffix=suffix))
        written.append(target.relative_to(dest).as_posix())

    for p in pointers:
        if not p.is_dir:
            continue
        put(p.md5, ".dir")
        for rel, md5 in dir_manifest(cache_dir, p).items():
            if rel == "provenance.json" or rel.endswith("/provenance.json"):
                put(md5, "")
    return sorted(set(written))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--cache", type=Path, default=None, help="DVC cache dir (default: the fixture)")
    ap.add_argument("--verify-weights", action="store_true", help="re-hash every weight blob")
    ap.add_argument(
        "--export-fixture", type=Path, default=None, help="write the CI mini-cache here"
    )
    args = ap.parse_args(argv)
    root = args.repo_root.resolve()
    cache = (args.cache or root / FIXTURE_CACHE).resolve()
    rows = parse_map((root / ADR_PATH).read_text(encoding="utf-8"))
    if args.export_fixture is not None:
        print(
            json.dumps(
                {"written": export_fixture(rows, root, cache, args.export_fixture)}, indent=2
            )
        )
        return 0
    stats: dict[str, int] = {}
    problems = validate_map(
        rows, root, cache, git_tracked(root), verify_weights=args.verify_weights, stats=stats
    )
    n_weights = sum(len(r.weights) for r in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "declared_future": sum(r.declared_future for r in rows),
                "weights_named": n_weights,
                "weights_rehashed": stats["rehashed"],
                "problems": problems,
            },
            indent=2,
        )
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
