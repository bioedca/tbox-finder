#!/usr/bin/env python
"""make_substrate_prescan_control.py — the designed spike/null control for the A8 gate.

The substrate pre-scan (:mod:`tbox_finder.mining.substrate_prescan`, ADR-0005 A8) folds a
matched spike/null control into **every** SLURM-array shard's single ``cmsearch`` invocation.
This script generates that control, deterministically (``shake_256`` — no RNG, the repo fixture
idiom), so the same shard always gets the same arms and the whole gate is reproducible.

- **spike** arm: known, cmsearch-detectable T-box loci (real records from
  ``data/external/refs/RF00230_master.fa``) embedded in deterministic background — a *live*
  detector MUST recover these (clause ii/iv). Their native accessions are irrelevant to the
  join; the arm is named ``spike_<shard>_<i>`` (unique, whitespace-free).
- **null** arm: a composition-preserving *mononucleotide shuffle* of the matched spike window —
  identical length and base composition, structure destroyed, so cmsearch does NOT detect it.
  Only the embedded locus varies (clause iii matchedness); the spike-minus-null separation
  (clause iv) is therefore attributable to the locus, not to length/composition.

``emit-control`` writes one shard's control manifest (the cluster path). ``mint-fixture`` builds
the committed golden test fixture end-to-end with a **real** local ``cmsearch`` run (CLAUDE.md
§9.1 lists cmsearch on small sets as LOCAL) — real records, real pipeline output, no mocking
(§8.7). CLAUDE.md §10.3: this control asserts no contamination value; it only proves the
detector is live and separating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from tbox_finder import infernal
from tbox_finder.mining import substrate_prescan as sp

MASTER_FA = Path("data/external/refs/RF00230_master.fa")

#: cmsearch writes its input path, full command line, working dir, and a timestamp into the
#: tblout header comments. Those leak a username + absolute worktree paths into a committed
#: fixture (CodeRabbit PR #76), so they are dropped before the tblout is committed. Only comment
#: metadata is removed — every data row (which `parse_tblout` reads) and the format/column
#: headers are kept, so the committed file stays a real cmsearch tblout (§8.7).
_SENSITIVE_TBLOUT_COMMENT = re.compile(
    r"^#.*(?:/home/|/tmp/|/exports/|Target file:|Option settings:|Current dir:|Date:"
    r"|target sequence database:)"
)


def sanitize_tblout(text: str) -> str:
    """Drop environment-specific comment lines from a cmsearch tblout (keeps all data rows)."""
    kept = [ln for ln in text.splitlines() if not _SENSITIVE_TBLOUT_COMMENT.search(ln)]
    return "\n".join(kept) + "\n"


def deterministic_acgt(n: int, key: str) -> str:
    """``n`` deterministic ACGT bases keyed on ``key`` (``shake_256`` — reproducible, no RNG)."""
    raw = hashlib.shake_256(key.encode("utf-8")).digest(n)
    return "".join("ACGT"[b & 3] for b in raw)


def mono_shuffle(seq: str, key: str) -> str:
    """Composition-preserving deterministic shuffle (Fisher–Yates over a ``shake_256`` stream).

    Preserves the exact base multiset (so the null fingerprints identically to its spike source
    in :func:`tbox_finder.mining.substrate_prescan.base_composition`) while destroying all
    sequence order, hence the covariation structure ``cmsearch`` keys on.
    """
    chars = list(seq)
    stream = hashlib.shake_256(key.encode("utf-8")).digest(8 * len(chars))
    for i in range(len(chars) - 1, 0, -1):
        # 8 bytes of entropy per step → a uniform-enough index in [0, i]
        r = int.from_bytes(stream[8 * i : 8 * i + 8], "big")
        j = r % (i + 1)
        chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def read_real_tboxes(
    n: int, *, master: Path = MASTER_FA, min_len: int = 90, max_len: int = 320
) -> list[str]:
    """First ``n`` real T-box sequences (length-bounded) from the RF00230 master alignment.

    Sequences only (not coordinates — TBDB ``Name`` coords are untrustworthy,
    [[tbdb-name-coords-untrustworthy]]); upper-cased, gaps stripped, U→T. Length-bounded so a
    compact fixture window stays small while carrying a genuine, cmsearch-detectable locus.
    """
    seqs: list[str] = []
    name: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if name is not None:
            s = "".join(chunks).upper().replace("-", "").replace("U", "T")
            if min_len <= len(s) <= max_len and set(s) <= set("ACGT"):
                seqs.append(s)

    with Path(master).open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                name = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        flush()
    if len(seqs) < n:
        raise RuntimeError(f"only {len(seqs)} usable T-boxes in {master}, need {n}")
    return seqs[:n]


def build_control_arms(
    shard: int, *, n: int, flank_nt: int, tbox_offset: int = 0
) -> dict[str, dict[str, str]]:
    """One shard's matched spike/null arms; asserts matchedness before returning.

    ``tbox_offset`` selects a disjoint slice of real T-boxes per shard so different shards embed
    different loci. Raises if the arms are not matched (equal count, equal length multiset,
    identical composition) — the generator refuses to emit a control that would certify vacuously
    ([[control-matchedness-must-be-asserted]]).
    """
    tboxes = read_real_tboxes(tbox_offset + n)[tbox_offset : tbox_offset + n]
    spike: dict[str, str] = {}
    null: dict[str, str] = {}
    for i, tb in enumerate(tboxes):
        left = deterministic_acgt(flank_nt, f"spike:{shard}:{i}:L")
        right = deterministic_acgt(flank_nt, f"spike:{shard}:{i}:R")
        window = left + tb + right
        spike[f"spike_{shard}_{i}"] = window
        null[f"null_{shard}_{i}"] = mono_shuffle(window, f"null:{shard}:{i}")
    sm = sp.arm_metadata(spike)
    nm = sp.arm_metadata(null)
    if not sp.arms_matched(sm, nm):
        raise RuntimeError(
            f"shard {shard}: generated spike/null arms are NOT matched — refusing to emit"
        )
    return {"spike": spike, "null": null}


def emit_control(shard: int, *, n: int, flank_nt: int, out: Path) -> Path:
    """Write one shard's control manifest (the cluster path — folded into that shard's scan)."""
    arms = build_control_arms(shard, n=n, flank_nt=flank_nt, tbox_offset=shard * n)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "kind": "substrate_prescan_control",
                "step": sp.STEP,
                "shard": shard,
                "n_per_arm": n,
                "flank_nt": flank_nt,
                "arms": arms,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return out


def _cmsearch_bin() -> str:
    """The ``cmsearch`` binary — from PATH, or the pinned local ``tbox-finder-infernal`` env."""
    import shutil

    if shutil.which("cmsearch"):
        return "cmsearch"
    local = Path.home() / "miniconda3/envs/tbox-finder-infernal/bin/cmsearch"
    if local.exists():
        return str(local)
    raise RuntimeError("cmsearch not found on PATH or in the pinned tbox-finder-infernal env")


def mint_fixture(
    out_dir: Path, *, n_shards: int, n_per_arm: int, n_production: int, flank_nt: int
) -> None:
    """Build the committed golden fixture end-to-end with a REAL local cmsearch run (§8.7)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shard_specs").mkdir(exist_ok=True)
    (out_dir / "tblout").mkdir(exist_ok=True)
    cm = str(infernal.RF00230_CM)
    cm_sha = hashlib.sha256(Path(cm).read_bytes()).hexdigest()
    score_threshold = 30.0  # recall-favouring: below GA=93 (unpinned in the ADR; fixture-local)
    cms = _cmsearch_bin()
    segments: list[dict[str, Any]] = []
    selected: list[str] = []

    for shard in range(n_shards):
        arms = build_control_arms(
            shard, n=n_per_arm, flank_nt=flank_nt, tbox_offset=shard * n_per_arm
        )
        # production: locus-free deterministic background windows (the admissible low-removal case),
        # one per fixture "genome" so genome_windows sums to the consumed production count.
        production: dict[str, str] = {}
        genome_windows: dict[str, int] = {}
        for i in range(n_production):
            acc = f"GCA_FIX{shard:02d}{i:03d}.1"
            production[f"prod_{shard}_{i}"] = deterministic_acgt(
                flank_nt * 2 + 200, f"prod:{shard}:{i}"
            )
            genome_windows[acc] = 1
            selected.append(acc)
        merged = {**production, **arms["spike"], **arms["null"]}
        fasta = out_dir / f"shard_{shard}.fna"
        infernal.write_fasta(merged, fasta)
        tblout = out_dir / "tblout" / f"shard_{shard}.tblout"
        proc = subprocess.run(
            [cms, "--noali", "--cpu", "2", "--tblout", str(tblout), cm, str(fasta)],
            capture_output=True,
            text=True,
            check=True,
        )
        if proc.stderr.strip():
            raise RuntimeError(f"cmsearch shard {shard} stderr non-empty: {proc.stderr[:200]}")
        # sanitize BEFORE committing (drop username/absolute paths) and use the sanitized text
        # for the segment too, so the committed tblout is exactly what build_shard_segment saw.
        tblout_text = sanitize_tblout(tblout.read_text(encoding="utf-8"))
        tblout.write_text(tblout_text)
        hits = infernal.parse_tblout(tblout_text)
        seg = sp.build_shard_segment(
            shard,
            arm_windows={"production": production, "spike": arms["spike"], "null": arms["null"]},
            hits=hits,
            tblout_text=tblout_text,
            cm_sha256=cm_sha,
            expected_production_windows=n_production,
            score_threshold=score_threshold,
            shard_ok=True,
            genome_windows=genome_windows,
        )
        segments.append(seg)
        (out_dir / "shard_specs" / f"shard_{shard}.json").write_text(
            json.dumps(
                {
                    "shard": shard,
                    "expected_production_windows": n_production,
                    "genome_windows": genome_windows,
                    "production": production,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        # per-shard control manifest (every shard, so the ml gate can rebuild EACH shard from its
        # own real tblout + control — CodeRabbit PR #76 asked for shard 1 to be covered too).
        (out_dir / f"control_shard_{shard}.json").write_text(
            json.dumps(
                {
                    "kind": "substrate_prescan_control",
                    "step": sp.STEP,
                    "shard": shard,
                    "arms": arms,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        # keep the FASTA out of the committed fixture (rebuildable from the specs); remove it
        fasta.unlink()

    report = sp.build_report(
        segments,
        n_shards_expected=n_shards,
        selected_accessions=selected,
        prior_genome_windows=None,  # maiden fetch — the frozen selection set-equality carries it
        score_threshold=score_threshold,
        substrate_scanned=True,
        source={
            "origin": "make_substrate_prescan_control.py mint-fixture",
            "note": "real local cmsearch",
        },
        accessed="2026-07-23",
    )
    problems = sp.validate_report(report, selected_accessions=selected, prior_genome_windows=None)
    if problems:
        raise RuntimeError("minted golden report does NOT certify:\n  - " + "\n  - ".join(problems))
    if not report["overall_pass"]:
        raise RuntimeError("minted golden report overall_pass is False")

    (out_dir / "golden_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out_dir / "selection_accessions.json").write_text(
        json.dumps(sorted(selected), indent=2, sort_keys=True) + "\n"
    )
    # a compact summary for the human minting it
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "n_shards": report["n_shards"],
                "n_production_windows": report["n_production_windows"],
                "n_spike_windows": report["n_spike_windows"],
                "n_null_windows": report["n_null_windows"],
                "substrate_removal_rate": report["substrate_removal_rate"],
                "per_shard_spike_removed": [s["arms"]["spike"]["n_removed"] for s in segments],
                "per_shard_null_removed": [s["arms"]["null"]["n_removed"] for s in segments],
                "clauses": report["clauses"],
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ec = sub.add_parser("emit-control", help="write one shard's spike/null control manifest")
    ec.add_argument("--shard", type=int, required=True)
    ec.add_argument("--n-per-arm", type=int, default=sp.MIN_SPIKE_N)
    ec.add_argument("--flank-nt", type=int, default=120)
    ec.add_argument("--out", required=True)

    mf = sub.add_parser(
        "mint-fixture", help="build the committed golden fixture (real local cmsearch)"
    )
    mf.add_argument("--out-dir", required=True)
    mf.add_argument("--n-shards", type=int, default=2)
    mf.add_argument("--n-per-arm", type=int, default=sp.MIN_SPIKE_N)
    mf.add_argument("--n-production", type=int, default=6)
    mf.add_argument("--flank-nt", type=int, default=120)

    args = ap.parse_args(argv)
    if args.cmd == "emit-control":
        emit_control(args.shard, n=args.n_per_arm, flank_nt=args.flank_nt, out=Path(args.out))
        return 0
    mint_fixture(
        Path(args.out_dir),
        n_shards=args.n_shards,
        n_per_arm=args.n_per_arm,
        n_production=args.n_production,
        flank_nt=args.flank_nt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
