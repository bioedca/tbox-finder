"""Regenerate the P2-10c R-scape gate fixtures (positive + matched shuffled control).

The gate in ``tests/ml/test_rscape_backend.py`` needs a control that must fire by
construction, so it uses a **matched pair**: a seeded subsample of the real
class-II corpus alignment, and the *same* subsample with every alignment column
independently permuted across rows. The permutation preserves each column's exact
base-and-gap composition and the ``#=GC SS_cons`` consensus structure verbatim,
and destroys only the correlation *between* columns — i.e. the covariation signal
itself, and nothing else. A natural "low covariation" negative could not falsify a
dead backend; this one must.

Note the control is conservative in the right direction: per-column permutation
also removes the phylogenetic correlation between sequences, which *raises*
R-scape's apparent power rather than lowering it. Finding zero covarying pairs on
it is therefore a stronger result than finding zero on a phylogeny-matched null.

Usage (inside the pinned env, from the repo root)::

    conda run -n tbox-rscape python scripts/make_rscape_fixture.py

Inputs  : data/interim/splits/aligned/class_II.sto  (DVC-tracked; `dvc pull` first)
Outputs : tests/fixtures/rscape/{classII_sub.sto,classII_sub.shuffled.sto,
                                 classII_sub.helixcov,classII_sub.shuffled.helixcov}
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tbox_finder.mining.covariation import PINNED_RSCAPE_VERSION, rscape_version

SOURCE = Path("data/interim/splits/aligned/class_II.sto")
OUTDIR = Path("tests/fixtures/rscape")

#: 60 sequences sits well clear of the power threshold — a seed sweep over the same
#: corpus passed the covariation criterion in 12/12 subsamples at N=60 and 12/12 at
#: N=30 under the looser operator, versus 0/12 at N=5-8. The fixture must not sit in
#: the marginal band, or a real regression would be indistinguishable from noise.
N_SEQUENCES = 60
SEED = 20260720
EVALUE = 0.05


def read_pfam_stockholm(path: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Read a one-line-per-sequence (pfam-format) Stockholm alignment."""
    seqs: list[tuple[str, str]] = []
    gc: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#=GC "):
            _, tag, val = line.split(None, 2)
            gc[tag] = val
        elif line.startswith("#") or line.strip() in {"", "//"}:
            continue
        else:
            name, aseq = line.split(None, 1)
            seqs.append((name, aseq.strip()))
    return seqs, gc


def write_pfam_stockholm(path: Path, seqs: list[tuple[str, str]], gc: dict[str, str]) -> None:
    width = max([len(n) for n, _ in seqs] + [len("#=GC " + t) for t in gc])
    lines = ["# STOCKHOLM 1.0", ""]
    lines += [f"{name.ljust(width)} {aseq}" for name, aseq in seqs]
    lines += [f"{('#=GC ' + tag).ljust(width)} {val}" for tag, val in gc.items()]
    lines.append("//")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def column_shuffle(seqs: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    """Permute every alignment column independently across rows."""
    rng = random.Random(seed)
    names = [n for n, _ in seqs]
    shuffled_columns = []
    for column in zip(*[aseq for _, aseq in seqs], strict=True):
        cells = list(column)
        rng.shuffle(cells)
        shuffled_columns.append(cells)
    return list(
        zip(names, ["".join(row) for row in zip(*shuffled_columns, strict=True)], strict=True)
    )


def run_rscape_to(alignment: Path, helixcov_out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="tbox-rscape-fixture-") as tmp:
        proc = subprocess.run(
            [
                "R-scape",
                "--nofigures",
                "-E",
                str(EVALUE),
                "--outdir",
                tmp,
                "--outname",
                "fx",
                str(alignment),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(f"R-scape failed (rc={proc.returncode}): {proc.stderr.strip()}")
        produced = Path(tmp) / "fx.helixcov"
        if not produced.is_file():
            raise SystemExit(f"R-scape produced no {produced}")
        helixcov_out.write_text(produced.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"{SOURCE} not found — run `dvc pull` first")
    # Refuse before touching tracked fixtures: regenerating outside the pinned env
    # would commit outputs from a different build, and v2.0.5 recalculated the
    # covariation power curves that produce every `nbp_cov` in these files.
    observed = rscape_version()
    if observed != PINNED_RSCAPE_VERSION:
        raise SystemExit(
            f"R-scape reports {observed!r} but envs/rscape.yml pins "
            f"{PINNED_RSCAPE_VERSION!r}; run inside the tbox-rscape env"
        )
    OUTDIR.mkdir(parents=True, exist_ok=True)

    seqs, gc = read_pfam_stockholm(SOURCE)
    if "SS_cons" not in gc:
        raise SystemExit(f"{SOURCE} carries no #=GC SS_cons; R-scape needs one")
    picked = sorted(random.Random(SEED).sample(range(len(seqs)), N_SEQUENCES))
    subsample = [seqs[i] for i in picked]

    # Stage all four in a temp dir and move them in only once BOTH R-scape runs
    # succeed. Writing incrementally means a failure on the second run leaves a
    # fresh positive beside a stale control — a mismatched pair that still looks
    # like a matched one, which is the single thing the gate cannot tolerate.
    with tempfile.TemporaryDirectory(prefix="tbox-rscape-stage-") as stage_dir:
        stage = Path(stage_dir)
        positive = stage / "classII_sub.sto"
        shuffled = stage / "classII_sub.shuffled.sto"
        write_pfam_stockholm(positive, subsample, gc)
        write_pfam_stockholm(shuffled, column_shuffle(subsample, SEED + 1), gc)

        run_rscape_to(positive, stage / "classII_sub.helixcov")
        run_rscape_to(shuffled, stage / "classII_sub.shuffled.helixcov")

        for name in (
            "classII_sub.sto",
            "classII_sub.shuffled.sto",
            "classII_sub.helixcov",
            "classII_sub.shuffled.helixcov",
        ):
            shutil.move(str(stage / name), str(OUTDIR / name))

    print(f"wrote {N_SEQUENCES} seqs x {len(subsample[0][1])} cols to {OUTDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
