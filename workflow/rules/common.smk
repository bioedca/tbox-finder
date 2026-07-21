"""common.smk — helper functions shared by the rule modules.

Functions live here rather than beside the rules that call them because
``snakemake --lint`` refuses "Mixed rules and functions in same snakefile" and a
red lint is CI-blocking (`.github/workflows/ci.yml` runs `snakemake --lint`).
The Snakefile globs ``workflow/rules/*.smk`` in sorted order, so ``common``
(c) is included before ``data`` (d) and its names are in scope there.

Only helpers that are not one-liners belong here — the lint's own guidance is
that a small single-use function should be a lambda in the rule instead, which
is the form ``data.smk``'s path params already use.
"""

import yaml

#: The §9.1 negative/mining sizing config, read by `flank_pad_nt` below.
DECOYS_CONF = "conf/data/decoys.yaml"


def flank_pad_nt(path: str = DECOYS_CONF) -> int:
    """Flank pad (nt) ``p2_flank_context`` must request, from the mining geometry.

    Derived, never duplicated. ``mining/pool.py::carve_window`` refuses to carve a
    window unless the record's realized flank exceeds ``mining_window_nt`` by
    ``mining_margin_nt`` (ADR-0005 D14's 50-nt margin), so the P2-00 fetch has to
    pad by at least their sum. A second copy of that sum hardcoded beside the rule
    would drift silently the next time either number moves, and P2-10d measured what
    that costs: at pad 1024 the corpus yields **0** carvable 1024-nt windows at
    margin 50, and ``build_mining_pool`` hard-fails rather than degrading — so the
    pad is load-bearing, and its drift is not a warning but an empty substrate.

    Overridable **upward** via ``--config flank_pad_nt=<int>``; the derived value is
    the default and is what every unattended run gets. An override *below* the
    geometry is refused rather than honoured: it would spend the whole NCBI fetch
    and then produce a context from which ``build_mining_pool`` carves nothing, so
    the cheap failure is here and the expensive one is an hour later (CodeRabbit r1).
    A larger pad is allowed — it costs bandwidth, not correctness.
    """
    with open(path, encoding="utf-8") as handle:
        conf = yaml.safe_load(handle) or {}
    required_nt = int(conf["mining_window_nt"]) + int(conf["mining_margin_nt"])
    override = config.get("flank_pad_nt")
    if override is None:
        return required_nt
    pad_nt = int(override)
    if pad_nt < required_nt:
        raise ValueError(
            f"--config flank_pad_nt={pad_nt} is below the mining geometry "
            f"({conf['mining_window_nt']} + {conf['mining_margin_nt']} = {required_nt}): "
            "the fetched flank could not yield a single carvable mining window"
        )
    return pad_nt
