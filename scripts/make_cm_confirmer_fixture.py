"""Regenerate the P3-17a CM-confirmer search fixture from its committed query list.

Reads ``tests/fixtures/cm_confirmer/queries.json`` (8 real Stage-2 rows: 4 T-boxes, 4
decoys), writes them as two query shards with the module's own shard writer, runs the
module's own :func:`cm_confirmer.search` with the pinned flags, then drops the tblout header
lines that carry machine paths (the ``make_substrate_prescan_control`` sanitiser). Every data
row is kept, so the committed files stay real Infernal 1.1.5 output (CLAUDE.md §8.7).

Runs in the ``infernal`` env (cmsearch on PATH)::

    PYTHONPATH=src python scripts/make_cm_confirmer_fixture.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_substrate_prescan_control import sanitize_tblout  # noqa: E402

from tbox_finder.stage2 import cm_confirmer as C  # noqa: E402

FIXTURE = Path("tests/fixtures/cm_confirmer")
N_SHARDS = 2


def main() -> int:
    spec = json.loads((FIXTURE / "queries.json").read_text(encoding="utf-8"))
    queries = C.collect_queries([r["rna_sequence"] for r in spec["records"]])
    out = FIXTURE / "search"
    if out.exists():
        shutil.rmtree(out)
    C.write_query_shards(
        queries,
        out / "queries",
        n_shards=N_SHARDS,
        sources={"queries": C.recorded_path(FIXTURE / "queries.json")},
    )
    C.search(query_dir=out / "queries", out_dir=out / "tblout", jobs=4)
    for tbl in sorted((out / "tblout").glob("*/*.tblout")):
        tbl.write_text(sanitize_tblout(tbl.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
