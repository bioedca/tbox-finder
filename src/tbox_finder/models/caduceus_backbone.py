"""P1-02 — Caduceus-PS Stage-1 backbone loader + human-reference pretraining provenance.

Loads the **default Stage-1 checkpoint** — Caduceus-PS
``kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16`` (7.73 M params,
``d_model`` 256, 16 MambaDNA layers, RC-equivariant) — as Hub *remote code*
(``trust_remote_code=True``), pinned to an **immutable commit revision** (never
``main``) so the backbone is byte-reproducible (PRD §6/§10.1; ADR-0002 D2/D7).

It then runs the two P1-02 assertions and writes a ``provenance.json`` sidecar:

  1. **Parameter count** equals the checkpoint's (``EXPECTED_PARAM_COUNT``) — a load /
     config-integrity check.
  2. **RC-equivariance forward-invariance.** Caduceus-PS is reverse-complement
     equivariant: for a hidden state ``h = f(x)`` the model card prescribes
     ``f(rc(x)) == h.flip(dims=(-2, -1))`` (flip along the sequence-length and channel
     dims), where ``rc(x)`` reverse-complements the input tokens via the config
     ``complement_map``. This is the precondition the §6 strand-resolver relies on (the
     RC-combination ablation must stay directionality-preserving), so it is asserted
     here on a short DNA fixture + its reverse-complement. The raw max-abs-diff is
     always recorded (no fabricated tolerance; §10.3).

**Pretraining-domain provenance (load-bearing, ships in the model card).** These public
Caduceus checkpoints are **pretrained on the human reference genome**, whereas the
tbox-finder discovery scan is fully prokaryotic/archaeal/MAG — so Stage-1 cross-domain
transfer is a *hypothesis to validate*, not an assumption (ADR-0002 D7; the P1 two-part
go/no-go). The provenance records this domain verbatim from the two authoritative
agreeing sources below so the model card can state it (CLAUDE.md §10.1 evidence gate):

  - HF model card ``kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16``:
    *"This model was pre-trained on the human reference genome with sequence length
    131,072 for 50k steps"* (accessed 2026-07-12).
  - Schiff et al., *Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence
    Modeling* — arXiv:2403.03234; PMID:40567809 (PMC12189541; Proc Mach Learn Res 2024).

**Compute.** LOCAL, on a **GPU**: the packaged ``Mamba`` block hard-imports
``selective_scan_cuda`` and calls the CUDA kernel even with ``use_fast_path=False``
(ADR-0002 A2 C2), so there is **no** pure-PyTorch CPU forward — the RC-equivariance
smoke runs on the local GPU (the ``tbox-ml-dna`` env; ADR-0002 A4). This supersedes the
imp.md "pure-PyTorch CPU forward" phrasing, per CLAUDE.md ADR > imp.md precedence.

All ``torch`` / ``transformers`` imports are lazy (inside functions): the module imports
cleanly with no torch present (the pure functions + constants stay stdlib-only).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from tbox_finder.provenance import write_provenance

SCHEMA_VERSION = "1"

# --- Pinned checkpoint identity (the reproducibility anchor) ---------------------------
#: The default Stage-1 checkpoint (PRD §6/§10.1; ADR-0002 D7).
REPO_ID = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
#: IMMUTABLE commit revision — never ``main`` (ADR-0002 D2). Resolved from the Hub
#: ``main`` branch head on 2026-07-12; pinning the SHA freezes the weights + remote code.
REVISION = "d89eeb853136ea64da7feb3d0c8e909771b17ae6"
#: Canonical source URL recorded in the provenance / model card.
HUB_URL = f"https://huggingface.co/{REPO_ID}"

# --- Frozen checkpoint facts (asserted; no report may contradict them) -----------------
#: Measured parameter count of the ``AutoModel`` (Caduceus base) at ``REVISION`` — the
#: "7.73 M" of PRD §10.1, to the exact integer. Frozen in CODE (not config) so a
#: generated provenance record can never silently drift from the checkpoint identity
#: (the coverage.py::OOD_ECE_MIN_N precedent).
EXPECTED_PARAM_COUNT = 7_725_312
D_MODEL = 256
N_LAYER = 16

#: Pretraining domain, verbatim from the two authoritative sources (see module docstring).
PRETRAINING_DOMAIN = "human reference genome"
PRETRAINING_SEQ_LEN = 131_072
PRETRAINING_CITATIONS = [
    {
        "source": "HF model card",
        "id": HUB_URL,
        "quote": (
            "This model was pre-trained on the human reference genome with sequence "
            "length 131,072 for 50k steps"
        ),
        "accessed": "2026-07-12",
    },
    {
        "source": "Schiff et al. 2024, Caduceus (Proc Mach Learn Res)",
        "id": "arXiv:2403.03234; PMID:40567809; PMC12189541",
        "accessed": "2026-07-12",
    },
]

#: RC-equivariance parity tolerance — an implementer choice (ADR-0002 pins no such
#: tolerance; the raw max-abs-diff is always recorded, §10.3). Observed max-abs-diff is
#: 0.0 (bit-exact) in fp32 on sm_86/sm_89, so this is a generous margin that still bites:
#: the reverse-only negative control differs by O(1).
RC_EQUIVARIANCE_ATOL = 1e-4
DEFAULT_SEQ_LEN = 128
DEFAULT_SEED = 42

DEFAULT_OUT = "data/interim/caduceus_ps_131k/provenance.json"

# The provenance ``extra`` blocks a schema-valid P1-02 report must carry.
_REPORT_BLOCKS = ("checkpoint", "pretraining", "env", "param_count", "rc_equivariance", "gate")


# ====================================================================================== #
# Pure helpers — stdlib only (import in any env; unit-tested to bite).
# ====================================================================================== #
def reverse_complement_ids(ids: Sequence[int], complement_map: Mapping[Any, Any]) -> list[int]:
    """Reverse-complement a token-id sequence.

    Complement each id through ``complement_map`` (the Caduceus config's id→id map:
    A↔T, C↔G, specials self-map), **then** reverse the order — the RC operator that
    turns a DNA window into the input the RC-equivariance smoke feeds the model, and
    the token-level analogue of the §6 strand handling.

    Pure/stdlib so the operator is unit-testable with no torch present.
    """
    cmap = {int(k): int(v) for k, v in complement_map.items()}
    return [cmap[int(i)] for i in ids][::-1]


def _sanitize(obj: Any) -> Any:
    """Map non-finite floats (NaN/Inf) → None so the report is always strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _is_commit_sha(rev: Any) -> bool:
    """True iff ``rev`` is a full 40-hex git commit SHA (an immutable pin, not a branch)."""
    return isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a (possibly empty) list of schema/consistency errors for a P1-02 report.

    Enforced regardless of whether a GPU produced it, so the committed provenance's
    ``extra`` block can be validated in a bare CI env (mirrors the kernel_smoke report
    validator). Empty list ⇒ structurally valid.
    """
    errs: list[str] = []
    for blk in _REPORT_BLOCKS:
        if blk not in report:
            errs.append(f"missing block: {blk}")
    if errs:
        return errs

    ckpt = report["checkpoint"]
    if ckpt.get("repo_id") != REPO_ID:
        errs.append("checkpoint.repo_id != pinned REPO_ID")
    if not _is_commit_sha(ckpt.get("revision")):
        errs.append(
            "checkpoint.revision is not a 40-hex commit SHA (must not be a branch like 'main')"
        )

    pre = report["pretraining"]
    if pre.get("domain") != PRETRAINING_DOMAIN:
        errs.append("pretraining.domain != 'human reference genome'")
    if not pre.get("citations"):
        errs.append("pretraining.citations is empty (the load-bearing fact needs a cite)")

    pc = report["param_count"]
    if pc.get("expected") != EXPECTED_PARAM_COUNT:
        errs.append("param_count.expected != EXPECTED_PARAM_COUNT (code/report drift)")

    rc = report["rc_equivariance"]
    for k in ("atol", "max_abs_diff", "neg_control_max_abs_diff", "pass"):
        if k not in rc:
            errs.append(f"rc_equivariance missing key: {k}")

    gate = report["gate"]
    for k in ("load_ok", "param_count_ok", "rc_equivariance_ok", "overall_pass"):
        if k not in gate:
            errs.append(f"gate missing key: {k}")

    # Consistency (only checkable once a GPU has measured the numbers).
    if report.get("measured"):
        if pc.get("actual") != pc.get("expected"):
            errs.append("param_count.actual != expected but param_count_ok cannot be true")
        if pc.get("param_count_ok") is False and gate.get("param_count_ok") is True:
            errs.append("gate.param_count_ok contradicts the block")
        md, at = rc.get("max_abs_diff"), rc.get("atol")
        if isinstance(md, (int, float)) and isinstance(at, (int, float)):
            expect_pass = bool(math.isfinite(md) and md <= at)
            if bool(rc.get("pass")) != expect_pass:
                errs.append("rc_equivariance.pass inconsistent with max_abs_diff vs atol")
        want_overall = bool(
            gate.get("load_ok") and gate.get("param_count_ok") and gate.get("rc_equivariance_ok")
        )
        if bool(gate.get("overall_pass")) != want_overall:
            errs.append("gate.overall_pass != AND(load_ok, param_count_ok, rc_equivariance_ok)")
    return errs


# ====================================================================================== #
# Heavy loaders — lazy torch / transformers (inside functions only).
# ====================================================================================== #
def load_caduceus_ps(*, revision: str = REVISION, device: str | None = None, dtype: Any = None):
    """Load the Caduceus-PS backbone (``AutoModel``) at the pinned ``revision``.

    Args:
        revision: the immutable commit SHA to pin (defaults to :data:`REVISION`); a
            branch name (e.g. ``main``) is rejected — the pin must be reproducible.
        device: torch device string; defaults to ``"cuda"`` if available else ``"cpu"``
            (note: a forward requires CUDA — ADR-0002 A2 C2).
        dtype: optional torch dtype override.

    Returns:
        The Caduceus base model in ``.eval()`` mode on ``device``.
    """
    if not _is_commit_sha(revision):
        raise ValueError(f"revision must be a 40-hex commit SHA (never a branch), got {revision!r}")
    import torch  # lazy
    from transformers import AutoModel  # lazy

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(REPO_ID, revision=revision, trust_remote_code=True)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval()


def load_tokenizer(*, revision: str = REVISION):
    """Load the Caduceus char tokenizer at the pinned ``revision`` (remote code)."""
    if not _is_commit_sha(revision):
        raise ValueError(f"revision must be a 40-hex commit SHA (never a branch), got {revision!r}")
    from transformers import AutoTokenizer  # lazy

    return AutoTokenizer.from_pretrained(REPO_ID, revision=revision, trust_remote_code=True)


def count_parameters(model) -> int:
    """Total number of parameters in ``model``."""
    return int(sum(p.numel() for p in model.parameters()))


def _hidden_states(output):
    """Extract the per-position hidden states ``(B, L, 2*d_model)`` from a Caduceus output."""
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def rc_equivariance(model, tokenizer, *, seq_len: int, seed: int, device: str) -> dict[str, Any]:
    """Measure RC-equivariance forward-invariance on a seeded DNA fixture + its RC.

    Builds a length-``seq_len`` random DNA token sequence, forms its reverse-complement
    in token space (``reverse_complement_ids`` via the config ``complement_map``),
    forwards both through the model, and compares ``f(rc(x))`` to
    ``f(x).flip(dims=(-2, -1))`` (the model-card RC transform on hidden states). A
    **reverse-only** (no complement) negative control is measured too, so the check is
    demonstrably discriminative.

    Returns a dict of the raw measurements (no pass/fail decision — the caller applies
    the atol).
    """
    import random

    import torch  # lazy

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    complement_map = dict(model.config.complement_map)
    acgt = [tokenizer.convert_tokens_to_ids(c) for c in "ACGT"]
    rng = random.Random(seed)
    ids = [rng.choice(acgt) for _ in range(seq_len)]
    rc_ids = reverse_complement_ids(ids, complement_map)
    rev_only = list(reversed(ids))  # negative control: reverse WITHOUT complement

    x = torch.tensor([ids], device=device)
    xrc = torch.tensor([rc_ids], device=device)
    xrev = torch.tensor([rev_only], device=device)

    with torch.no_grad():
        h = _hidden_states(model(input_ids=x))
        hrc = _hidden_states(model(input_ids=xrc))
        hrev = _hidden_states(model(input_ids=xrev))

    expected = h.flip(dims=(-2, -1))  # model-card RC transform on hidden states
    max_abs_diff = (hrc - expected).abs().max().item()
    mean_abs_diff = (hrc - expected).abs().mean().item()
    neg_control = (hrev - expected).abs().max().item()
    return {
        "seq_len": int(seq_len),
        "seed": int(seed),
        "hidden_shape": [int(v) for v in h.shape],
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "neg_control_max_abs_diff": neg_control,
        "hidden_abs_max": h.abs().max().item(),
    }


# ====================================================================================== #
# Orchestration — build the P1-02 report.
# ====================================================================================== #
def _env_block() -> dict[str, Any]:
    import torch  # lazy
    import transformers  # lazy

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": None,
        "device_capability": None,
        "is_sm86": None,
        "hostname": platform.node(),
    }
    if info["cuda_available"]:
        cap = torch.cuda.get_device_capability(0)
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = [int(cap[0]), int(cap[1])]
        info["is_sm86"] = list(info["device_capability"]) == [8, 6]
    return info


def build_report(*, revision: str, seq_len: int, seed: int, atol: float) -> dict[str, Any]:
    """Load the checkpoint, run the two P1-02 assertions, and return the report dict."""
    env = _env_block()
    device = "cuda" if env["cuda_available"] else "cpu"

    model = load_caduceus_ps(revision=revision, device=device)
    tokenizer = load_tokenizer(revision=revision)
    cfg = model.config

    actual_params = count_parameters(model)
    param_ok = actual_params == EXPECTED_PARAM_COUNT

    rc = rc_equivariance(model, tokenizer, seq_len=seq_len, seed=seed, device=device)
    rc["atol"] = atol
    md = rc["max_abs_diff"]
    rc["pass"] = bool(math.isfinite(md) and md <= atol)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": "P1-02",
        "measured": True,
        "generated_by": "src/tbox_finder/models/caduceus_backbone.py",
        "checkpoint": {
            "repo_id": REPO_ID,
            "revision": revision,
            "hub_url": HUB_URL,
            "d_model": int(getattr(cfg, "d_model", D_MODEL)),
            "n_layer": int(getattr(cfg, "n_layer", N_LAYER)),
            "rcps": bool(getattr(cfg, "rcps", True)),
            "config_transformers_version": getattr(cfg, "transformers_version", None),
            "torch_device": device,
        },
        "pretraining": {
            "domain": PRETRAINING_DOMAIN,
            "sequence_length": PRETRAINING_SEQ_LEN,
            "transfer_note": (
                "Human-reference pretrained; the tbox-finder scan is prokaryotic/archaeal/MAG "
                "→ Stage-1 transfer is a hypothesis validated by the P1 go/no-go (ADR-0002 D7)."
            ),
            "citations": PRETRAINING_CITATIONS,
        },
        "env": env,
        "param_count": {
            "expected": EXPECTED_PARAM_COUNT,
            "actual": actual_params,
            "param_count_ok": param_ok,
        },
        "rc_equivariance": rc,
        "gate": None,
    }
    load_ok = True
    report["gate"] = {
        "load_ok": load_ok,
        "param_count_ok": bool(param_ok),
        "rc_equivariance_ok": bool(rc["pass"]),
        "overall_pass": bool(load_ok and param_ok and rc["pass"]),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="P1-02 Caduceus-PS backbone load + RC-equivariance gate"
    )
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, dest="seq_len")
    p.add_argument("--atol", type=float, default=RC_EQUIVARIANCE_ATOL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seq_len < 2:
        parser.error("--seq-len must be >= 2")

    report = build_report(
        revision=args.revision, seq_len=args.seq_len, seed=args.seed, atol=args.atol
    )

    errs = validate_report(report)
    if errs:  # a self-inconsistent report must not be written as if valid
        print(json.dumps({"schema_errors": errs}, indent=2), file=sys.stderr)
        return 2

    # Write the canonical provenance.json sidecar (CLAUDE.md §11 fields + the P1-02
    # report folded under ``extra``). The HF checkpoint is the input (pinned by revision,
    # not a local file), so ``inputs``/``outputs`` are empty and the identity lives in
    # ``extra.checkpoint``.
    write_provenance(
        args.out,
        rule="src/tbox_finder/models/caduceus_backbone.py :: main (P1-02, LOCAL GPU)",
        script="src/tbox_finder/models/caduceus_backbone.py",
        seed=args.seed,
        env_lock="envs/ml-dna.conda-lock.yml",
        adr="ADR-0002 (D2/D7; A2 C2)",
        extra=_sanitize(report),
    )
    g = report["gate"]
    print(json.dumps({"gate": g, "out": args.out}, indent=2))
    return 0 if g["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
