"""Infernal ``cmsearch`` runner + ``--tblout`` parser (P2-07).

The repo already shells out to ``cmalign`` (:func:`tbox_finder.splits.run_cmalign`);
``cmsearch`` had no caller until P2-07 needed covariance-model *detection* verdicts
for the Tier-2N discordant-pair construction. This module is the single such
caller — P6's covariation/architecture work extends it rather than forking it.

Pinned tool: ``infernal=1.1.5`` (``envs/infernal.yml``), which is **not** on the
base PATH; callers run inside the ``tbox-infernal`` env. The tabular format parsed
here is ``--fmt 1`` (the default), whose 18 whitespace-delimited fields were read
off the installed binary rather than from documentation — Context7 carries no
Infernal corpus, so the authoritative source is the pinned binary itself.

Detection semantics
-------------------
``--cut_ga`` applies the model's own **GA gathering cutoff** as the reporting
threshold. For ``RF00230.cm`` that is **93.00 bits** (``GA``/``TC`` both 93.00,
``NC`` 92.90, ``CLEN`` 224). This is a deliberately strict operating point and it
is *not* a proxy for "is a T-box": measured in-repo at P2-07 over 500 corpus
records (seed 42), **only 362/500 = 72.4 % of real TBDB T-boxes clear it** — a
27.6 % miss rate on unablated natural positives. Any consumer treating a
``--cut_ga`` miss as evidence of non-canonical *architecture* must control for
that baseline (see :mod:`tbox_finder.synth.tier2n`, which pairs every variant
against its own parent).

Pure stdlib. PRD §9.1, §12; ADR-0005 D14.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Field count of a ``cmsearch --tblout --fmt 1`` data row, from the installed
#: 1.1.5 header: target name / accession / query name / accession / mdl /
#: mdl from / mdl to / seq from / seq to / strand / trunc / pass / gc / bias /
#: score / E-value / inc / description. The description field may contain spaces,
#: so rows are split with ``maxsplit`` and the tail kept whole.
TBLOUT_FIXED_FIELDS = 17

#: 0-based column of the target (sequence) name in a ``--fmt 1`` data row.
TBLOUT_TARGET_COL = 0
#: 0-based column of the bit score.
TBLOUT_SCORE_COL = 14
#: 0-based column of the E-value.
TBLOUT_EVALUE_COL = 15

#: Default CM used for the T-box detection verdict.
RF00230_CM = Path("data/external/refs/RF00230.cm")

#: Wall-clock ceiling for one ``cmsearch`` invocation. Generous relative to the
#: measured cost of the sets this module searches (500 corpus records against
#: RF00230 takes ~4 min on 4 threads), so it bounds a hang without truncating
#: legitimate work. Callers searching larger sets should raise it explicitly
#: rather than relying on the default.
DEFAULT_CMSEARCH_TIMEOUT_S = 3600.0


class InfernalNotAvailableError(RuntimeError):
    """Raised when ``cmsearch`` is not on PATH.

    Deliberately an error rather than a silent empty result: an absent binary that
    returned "no hits" would mark **every** sequence CM-missed, which is exactly
    the fail-open direction that would let an unrun search manufacture a Tier-2N
    probe set of arbitrary size.
    """


@dataclass(frozen=True)
class CmsearchHit:
    """One reported hit from a ``--tblout`` table."""

    target: str
    score: float
    evalue: float


def cmsearch_available() -> bool:
    """Whether the pinned ``cmsearch`` binary is callable in this environment."""
    return shutil.which("cmsearch") is not None


def parse_tblout(text: str) -> list[CmsearchHit]:
    """Parse ``cmsearch --tblout`` (``--fmt 1``) text into hits.

    Comment lines (``#``) and blanks are skipped; a short row raises rather than
    being dropped, so a format change surfaces as a failure instead of silently
    shrinking the hit set (which would read as "CM missed it").
    """
    hits: list[CmsearchHit] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=TBLOUT_FIXED_FIELDS)
        if len(fields) < TBLOUT_FIXED_FIELDS:
            raise ValueError(
                f"tblout line {lineno} has {len(fields)} fields, expected at least "
                f"{TBLOUT_FIXED_FIELDS}: {line!r}"
            )
        try:
            score = float(fields[TBLOUT_SCORE_COL])
            evalue = float(fields[TBLOUT_EVALUE_COL])
        except ValueError as exc:  # pragma: no cover - malformed numeric field
            raise ValueError(f"tblout line {lineno} has a non-numeric score/E-value") from exc
        hits.append(CmsearchHit(target=fields[TBLOUT_TARGET_COL], score=score, evalue=evalue))
    return hits


def write_fasta(records: dict[str, str], path: str | Path) -> Path:
    """Write ``{name: sequence}`` to a FASTA file, uppercased and ungapped.

    Names and sequences are validated rather than escaped, because both failure
    modes are **fail-open toward a larger probe set**. ``cmsearch``'s tblout
    ``target name`` column holds only the first whitespace-delimited token of a
    header, so a name containing a space would be looked up in
    :func:`detection_map` under its full form, never match, and read as
    "CM missed it". Whitespace inside a sequence would likewise split one record
    into several. Either would silently manufacture Tier-2N probe positives, so
    both raise here instead.
    """
    # Validate everything BEFORE truncating the destination, so a rejected record
    # leaves any previous FASTA intact rather than replacing it with a partial
    # file that a later cmsearch would happily search.
    cleaned_records: list[tuple[str, str]] = []
    for name, sequence in records.items():
        if not name or any(char.isspace() for char in name):
            raise ValueError(f"FASTA record name must be a non-empty single token: {name!r}")
        cleaned = sequence.replace("-", "").upper()
        if not cleaned or any(char.isspace() for char in cleaned):
            raise ValueError(f"FASTA sequence for {name!r} must be non-empty and whitespace-free")
        cleaned_records.append((name, cleaned))

    out = Path(path)
    with out.open("w", encoding="utf-8") as handle:
        for name, cleaned in cleaned_records:
            handle.write(f">{name}\n{cleaned}\n")
    return out


def run_cmsearch(
    cm: str | Path,
    fasta: str | Path,
    tblout: str | Path,
    *,
    cut_ga: bool = True,
    cpu: int = 4,
    timeout_s: float = DEFAULT_CMSEARCH_TIMEOUT_S,
) -> list[CmsearchHit]:
    """Run ``cmsearch`` and return its parsed hits.

    Raises :class:`InfernalNotAvailableError` if the binary is absent and
    ``RuntimeError`` on a non-zero exit — never an empty hit list, which a caller
    could not distinguish from a genuine no-hit result.
    """
    if not cmsearch_available():
        raise InfernalNotAvailableError(
            "cmsearch is not on PATH; run inside the pinned tbox-infernal env "
            "(envs/infernal.yml, infernal=1.1.5)"
        )
    cm_path, fasta_path, tbl_path = Path(cm), Path(fasta), Path(tblout)
    if not cm_path.exists():
        raise FileNotFoundError(f"CM file not found: {cm_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    cmd = ["cmsearch", "--noali", "--cpu", str(cpu), "--tblout", str(tbl_path)]
    if cut_ga:
        cmd.append("--cut_ga")
    cmd += [str(cm_path), str(fasta_path)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        # Routed into the same failure path as a non-zero exit: a hung search must
        # never surface as "no hits", which downstream reads as "the CM missed it".
        raise RuntimeError(f"cmsearch timed out after {timeout_s}s: {' '.join(cmd)}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"cmsearch failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return parse_tblout(tbl_path.read_text(encoding="utf-8"))


def detection_map(records: dict[str, str], hits: list[CmsearchHit]) -> dict[str, bool]:
    """Map every submitted record name → whether the CM reported a hit for it.

    Keyed off the **submitted** records, not the hit table, so a record that drew
    no hit is an explicit ``False`` rather than an absent key that a caller might
    resolve to ``None``/unmeasured.
    """
    detected = {hit.target for hit in hits}
    return {name: name in detected for name in records}


def build_report(records: dict[str, str], hits: list[CmsearchHit], **extra: Any) -> dict[str, Any]:
    """Summarise a detection run (counts re-derived from the inputs)."""
    detection = detection_map(records, hits)
    n_detected = sum(1 for v in detection.values() if v)
    n_total = len(detection)
    # ``extra`` is spread FIRST so caller metadata can never shadow a derived
    # count — a report whose n_detected came from the caller rather than from the
    # hit table would be a fabricated metric wearing a measured field's name.
    return {
        **extra,
        "n_submitted": n_total,
        "n_detected": n_detected,
        "n_missed": n_total - n_detected,
        "detected_fraction": (n_detected / n_total) if n_total else None,
    }
