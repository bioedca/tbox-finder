"""P1-10 — Nucleotide-Transformer-multispecies Stage-1 backbone (fallback/ablation drop-in).

The **last rung** of the ADR-0002 D7 transfer-fallback ladder (full fine-tune →
GTDB continued-pretraining → *this* prokaryotic/multispecies DNA backbone). The P1-07
transfer go/no-go returned **GO**, so this ladder is **NOT triggered** (ADR-0002 A6);
P1-10 nonetheless **builds** the drop-in — a stub-free loader with the same per-position
hidden-state contract the seg head consumes — so a P2 transfer-driven GATE-4 failure can
invoke it without new plumbing. It is **not wired as any default** (a unit test locks
that: no ``conf/train/*.yaml`` selects it).

What "drop-in" means here
-------------------------
The P1-03 :class:`~tbox_finder.models.seg_head.SegmentationHead` is width-parameterised:
it maps ``(B, L, input_dim) → (B, L, 8)`` for *any* ``input_dim``. So a backbone is a
drop-in iff it emits a per-position hidden state ``(B, L, d_model)`` the head can size to.
This loader exposes exactly that (:func:`hidden_states` → ``(B, L, HIDDEN_DIM)``), and the
interface-parity smoke feeds it through **both** the head directly *and* the existing
:class:`~tbox_finder.models.stage1_segmenter.Stage1Segmenter` assembly (see below), on a
real forward — so the compatibility is *measured*, not asserted (§10.3).

**The k-mer tokenisation caveat (load-bearing for P2, documented here).** Unlike the
default Caduceus-PS backbone (single-nucleotide char tokeniser → one token per
nucleotide, so ``L_token == L_nt``), the Nucleotide Transformer uses a **6-mer**
tokeniser: a length-``L_nt`` window becomes ``≈ L_nt / 6`` tokens (with single-nt
fallback tokens for ``N`` or a non-multiple-of-6 tail; PMID:39609566). The seg head is
therefore per-**token** (per-6-mer), **not** per-nucleotide — so the GATE-4 per-nt 8-class
label scheme (ADR-0004) does **not** align 1:1 with NT tokens. Consuming this backbone at
P2 requires a token→nucleotide label broadcast/upsampling (6× per token) — that mapping
is a P2 concern only realised **if this fallback is triggered**; P1-10 delivers the
loader + the interface-parity proof + this caveat, not the relabelling.

Pretraining-domain provenance (the §10.1 scientific-evidence gate, load-bearing)
--------------------------------------------------------------------------------
The premise that justifies this backbone as a *domain-shift remedy* is that its
pretraining corpus is **multispecies and prokaryote-inclusive** (the tbox-finder scan is
prokaryotic/archaeal/MAG, whereas the default Caduceus-PS is human-reference pretrained).
Verified against **two independent, agreeing authoritative sources** (CLAUDE.md §10.1,
≥2 for a high-stakes biological claim):

  - **Peer-reviewed paper** — Dalla-Torre et al., *Nucleotide Transformer: building and
    evaluating robust foundation models for human genomics*, Nat Methods 2025;22:287-297.
    PMID:39609566; PMCID:PMC11810778; DOI:10.1038/s41592-024-02523-z. Methods, "The
    Multispecies dataset": the 850-genome collection was sampled from RefSeq groupings
    *"archaea, fungi, vertebrate_mammalian, vertebrate_other, etc."* with *"a random
    subset"* of the *"large number of available bacterial genomes"* (accessed 2026-07-13).
  - **Pretraining-corpus dataset card** — HF ``InstaDeepAI/multi_species_genomes`` (the
    actual pretraining data): composition table lists **Bacteria = 667 species / 17.1 B
    nucleotides** (~78% of the 850 species by count), alongside fungi/invertebrate/
    protozoa/vertebrate rows (accessed 2026-07-13).

**Asserted claim (both sources agree):** the corpus is multispecies and
**prokaryote-inclusive via bacteria**. Two honesty caveats we do **not** paper over
(§10.3): (1) **archaea specifically is unresolved** — the paper lists "archaea" among the
sampled RefSeq groupings, but the released corpus card has **no archaea row** and its rows
sum to 850 species without one, so we do **NOT** assert archaeal inclusion; (2) the
per-checkpoint **model cards are silent** on prokaryotes (they say only "850 genomes …
including model and non-model organisms" and exclude plants+viruses) — the load-bearing
claim rests on the paper + the corpus dataset card, not the model card.

Checkpoint choice (implementer decision — ADR-0002 pins no NT checkpoint)
------------------------------------------------------------------------
``InstaDeepAI/nucleotide-transformer-v2-250m-multi-species`` — a v2 multispecies model
(rotary embeddings, ~2048-token / ~12 kb receptive field vs v1's ~6 kb; ESM architecture,
``hidden_size`` 768, 24 layers, ~250 M params). Materially larger than the 7.73 M
Caduceus-PS default yet small enough to load on the laptop for this smoke and to
fine-tune on a single A4000 (16 GB) at P2. The size is a one-line swap (``REPO_ID`` /
``REVISION``) to the 500 m sibling if a triggered P2 fine-tune needs more capacity.

**Loading (ADR-0002 D2 ceiling).** v2 NT checkpoints ship an ``auto_map`` to repo-local
modeling code (a bias-free gated-FFN ESM variant), so they load with
``trust_remote_code=True`` — same security posture as Caduceus. As with the Caduceus
loader, ``trust_remote_code=True`` **executes** the Hub modeling code at ``REVISION`` on
load, so the loaders **reject any non-pinned revision before the import** (re-pinning is a
code change + re-sign-off, never a runtime argument). Must load under the ``ml-dna`` env's
``transformers`` 4.57.5 ceiling (ADR-0002 D2/A4).

**Compute.** LOCAL. NT is a stock-forward ESM (no ``selective_scan_cuda`` kernel), so the
interface-parity forward runs on **CPU** — unlike Caduceus/Mamba it needs no GPU. All
``torch`` / ``transformers`` / seg-head imports are lazy (inside functions): the module
imports cleanly with no torch present (the pure helpers + constants + validator stay
stdlib-only, so the bare-CI tier runs).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from collections.abc import Mapping
from typing import Any

from tbox_finder.labels import CLASS_ORDER
from tbox_finder.provenance import write_provenance

SCHEMA_VERSION = "1"

# --- Pinned checkpoint identity (the reproducibility anchor; ADR-0002 D2) --------------
#: The NT-multispecies fallback checkpoint (implementer choice — ADR pins none; see the
#: module docstring for the rationale).
REPO_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
#: IMMUTABLE commit revision — never ``main`` (ADR-0002 D2). The current ``main`` head on
#: the Hub (HF API ``/api/models/{REPO_ID}`` ``.sha``); pinning the SHA freezes the weights
#: + the remote modeling code that ``trust_remote_code=True`` executes.
REVISION = "c0f0359229f36ff6bc3a021247eefe0a9c344bd1"
#: Canonical source URL recorded in the provenance / model card.
HUB_URL = f"https://huggingface.co/{REPO_ID}"

# --- Frozen checkpoint config facts (from the checkpoint's config.json; asserted) ------
#: Per-position hidden width (``config.json`` ``hidden_size``) — the ``d_model`` the seg
#: head is sized to for the NT drop-in (there is no RC concatenation: NT is single-strand,
#: so this is ``d_model`` itself, not ``2*d_model`` as for RC-equivariant Caduceus-PS).
HIDDEN_DIM = 768
#: ``config.json`` ``num_hidden_layers``.
N_LAYER = 24
#: The ESM ``model_type`` these checkpoints report.
MODEL_TYPE = "esm"
#: The tokeniser k-mer size — the load-bearing tokenisation caveat (NOT single-nt).
KMER = 6
#: Usable token context (``max_position_embeddings`` 2050 − 2 specials).
MAX_TOKENS = 2048

#: Number of segmentation classes (ADR-0004 D1) — single-sourced from the label vocab so
#: the head width the drop-in verifies against can never silently drift from the scheme.
NUM_CLASSES = len(CLASS_ORDER)

#: Pretraining-domain facts, verbatim-sourced (see module docstring). We assert
#: prokaryote-inclusion **via bacteria**; archaea is deliberately left ``None`` (unresolved).
PRETRAINING_COLLECTION = "InstaDeepAI multispecies (850 genomes sampled from RefSeq)"
PROKARYOTE_INCLUSIVE = True  # via bacteria (667 species / 17.1 B nt) — paper + corpus card
ARCHAEA_INCLUDED: bool | None = None  # UNRESOLVED — not asserted (§10.3)
MODEL_CARD_STATES_PROKARYOTES = (
    False  # the per-checkpoint model cards are silent; paper + corpus card carry it
)
PRETRAINING_CITATIONS = [
    {
        "source": "Dalla-Torre et al. 2025, Nucleotide Transformer (Nat Methods)",
        "id": "PMID:39609566; PMCID:PMC11810778; DOI:10.1038/s41592-024-02523-z",
        "quote": (
            "we randomly selected one genome at the genus level from each of the main "
            "groupings available in RefSeq (archaea, fungi, vertebrate_mammalian, "
            "vertebrate_other, etc.) ... due to the large number of available bacterial "
            "genomes, we opted to include only a random subset of them"
        ),
        "supports": "multispecies + prokaryote-inclusive (bacteria)",
        "accessed": "2026-07-13",
    },
    {
        "source": "HF dataset card InstaDeepAI/multi_species_genomes (pretraining corpus)",
        "id": "https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes",
        "quote": "Bacteria: 667 species / 17.1 B nucleotides (of 850 species total)",
        "supports": "prokaryote-inclusive (bacteria)",
        "accessed": "2026-07-13",
    },
]

#: A deterministic DNA fixture length for the interface-parity smoke — a multiple of
#: :data:`KMER` so the 6-mer tokeniser has no ragged tail (the smoke checks shapes, not a
#: specific sequence, so any seeded ACGT window suffices).
DEFAULT_SEQ_LEN_NT = 300
DEFAULT_SEED = 42
DEFAULT_OUT = "data/interim/nt_multispecies_v2_250m/provenance.json"

# The provenance ``extra`` blocks a schema-valid P1-10 report must carry.
_REPORT_BLOCKS = ("checkpoint", "pretraining", "tokenization", "env", "interface_parity", "gate")


# ====================================================================================== #
# Pure helpers — stdlib only (import in any env; unit-tested to bite).
# ====================================================================================== #
def random_dna(seq_len: int, seed: int) -> str:
    """A deterministic length-``seq_len`` ACGT string (seeded; stdlib, torch-free)."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(seq_len))


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


def _bad_bool(value: Any, expected: bool) -> bool:
    """True (a violation) iff ``value`` is not a real ``bool`` equal to ``expected`` — so a
    tampered JSON cannot slip a truthy string/number past a verdict check (§8.7/§10.3)."""
    return not isinstance(value, bool) or value != expected


def _is_shape(v: Any, *, ndim: int, last: int | None = None) -> bool:
    """True iff ``v`` is a list of ``ndim`` positive ints (optionally with ``last`` last dim)."""
    if not isinstance(v, list) or len(v) != ndim:
        return False
    if not all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in v):
        return False
    return last is None or v[-1] == last


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return a (possibly empty) list of schema/consistency errors for a P1-10 report.

    Enforced regardless of whether torch produced it, so the committed provenance's
    ``extra`` block validates in a bare CI env (mirrors the Caduceus / kernel_smoke
    validators). Robust to a malformed/tampered report — never raises, always returns an
    error list. Empty list ⇒ structurally valid.
    """
    errs: list[str] = []
    for blk in _REPORT_BLOCKS:
        if blk not in report:
            errs.append(f"missing block: {blk}")
        elif not isinstance(report[blk], Mapping):
            errs.append(f"block is not a mapping: {blk}")
    if errs:  # a non-mapping block would crash the .get() checks below — stop here
        return errs

    if report.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version != {SCHEMA_VERSION!r}")
    if report.get("step") != "P1-10":
        errs.append("step != 'P1-10'")

    ckpt = report["checkpoint"]
    if ckpt.get("repo_id") != REPO_ID:
        errs.append("checkpoint.repo_id != pinned REPO_ID")
    if not _is_commit_sha(ckpt.get("revision")):
        errs.append(
            "checkpoint.revision is not a 40-hex commit SHA (must not be a branch like 'main')"
        )
    elif ckpt.get("revision") != REVISION:
        errs.append("checkpoint.revision != pinned REVISION")
    if ckpt.get("hidden_dim") != HIDDEN_DIM:
        errs.append("checkpoint.hidden_dim != HIDDEN_DIM")
    if ckpt.get("n_layer") != N_LAYER:
        errs.append("checkpoint.n_layer != N_LAYER")
    if ckpt.get("model_type") != MODEL_TYPE:
        errs.append("checkpoint.model_type != MODEL_TYPE")
    # trust_remote_code MUST be recorded True — v2 NT ships remote modeling code (ADR-0002 D2).
    if _bad_bool(ckpt.get("trust_remote_code"), True):
        errs.append("checkpoint.trust_remote_code must be True (v2 NT is Hub remote code)")

    pre = report["pretraining"]
    if _bad_bool(pre.get("prokaryote_inclusive"), True):
        errs.append("pretraining.prokaryote_inclusive must be True (the domain-shift premise)")
    # Honesty guards (§10.3): archaea must NOT be asserted true, and the model-card-silent
    # fact must be recorded, so a future edit cannot quietly over-claim the domain.
    if pre.get("archaea_included") is True:
        errs.append("pretraining.archaea_included must not be True (archaea is unresolved)")
    if _bad_bool(pre.get("model_card_states_prokaryotes"), False):
        errs.append("pretraining.model_card_states_prokaryotes must be False (cards are silent)")
    cites = pre.get("citations")
    if not isinstance(cites, list) or len(cites) < 2:
        errs.append("pretraining.citations needs >= 2 sources (high-stakes claim, §10.1)")

    tok = report["tokenization"]
    if tok.get("kmer") != KMER:
        errs.append("tokenization.kmer != KMER (NT is 6-mer, not single-nt)")

    ip = report["interface_parity"]
    for k in ("hidden_shape", "seg_head_logits_shape", "segmenter_logits_shape", "num_classes"):
        if k not in ip:
            errs.append(f"interface_parity missing key: {k}")

    gate = report["gate"]
    for k in (
        "load_ok",
        "hidden_dim_ok",
        "seg_head_dropin_ok",
        "segmenter_dropin_ok",
        "overall_pass",
    ):
        if k not in gate:
            errs.append(f"gate missing key: {k}")
    if errs:  # missing sub-keys → the consistency checks below can't run meaningfully
        return errs

    # Consistency (only checkable once torch has measured the shapes). These flag an
    # internally *inconsistent* report, never an honestly-recorded failure. Verdicts must
    # be real booleans and shapes well-formed, so a hand-tampered JSON cannot slip a
    # truthy string / mis-sized tensor past the gate (§8.7/§10.3).
    if report.get("measured"):
        hidden_ok_shape = _is_shape(ip.get("hidden_shape"), ndim=3, last=HIDDEN_DIM)
        if _bad_bool(gate.get("hidden_dim_ok"), hidden_ok_shape):
            errs.append("gate.hidden_dim_ok inconsistent with interface_parity.hidden_shape")
        if ip.get("num_classes") != NUM_CLASSES:
            errs.append(f"interface_parity.num_classes != {NUM_CLASSES}")
        seg_ok = _is_shape(ip.get("seg_head_logits_shape"), ndim=3, last=NUM_CLASSES) and (
            ip.get("seg_head_logits_finite") is True
        )
        if _bad_bool(gate.get("seg_head_dropin_ok"), seg_ok):
            errs.append("gate.seg_head_dropin_ok inconsistent with seg_head_logits_shape/finite")
        segr_ok = _is_shape(ip.get("segmenter_logits_shape"), ndim=3, last=NUM_CLASSES) and (
            ip.get("segmenter_logits_finite") is True
        )
        if _bad_bool(gate.get("segmenter_dropin_ok"), segr_ok):
            errs.append("gate.segmenter_dropin_ok inconsistent with segmenter_logits_shape/finite")
        # The token axis must agree across the two drop-in paths and the hidden state
        # (both heads classify the SAME positions the backbone emitted).
        hs, sh, sm = (
            ip.get("hidden_shape"),
            ip.get("seg_head_logits_shape"),
            ip.get("segmenter_logits_shape"),
        )
        if all(_is_shape(s, ndim=3) for s in (hs, sh, sm)) and not (hs[1] == sh[1] == sm[1]):
            errs.append("interface_parity token axis (L) disagrees across hidden/head/segmenter")
        for k in ("load_ok", "hidden_dim_ok", "seg_head_dropin_ok", "segmenter_dropin_ok"):
            if not isinstance(gate.get(k), bool):
                errs.append(f"gate.{k} must be a bool")
        want_overall = bool(
            gate.get("load_ok")
            and gate.get("hidden_dim_ok")
            and gate.get("seg_head_dropin_ok")
            and gate.get("segmenter_dropin_ok")
        )
        if _bad_bool(gate.get("overall_pass"), want_overall):
            errs.append(
                "gate.overall_pass != AND(load_ok, hidden_dim_ok, seg_head_dropin_ok, "
                "segmenter_dropin_ok)"
            )
    return errs


# ====================================================================================== #
# Heavy loaders — lazy torch / transformers (inside functions only).
# ====================================================================================== #
def _require_pinned_revision(revision: str) -> None:
    """Reject any revision other than the code-pinned :data:`REVISION`.

    ``trust_remote_code=True`` **executes** the Hub modeling code at ``revision`` on load,
    so an arbitrary revision is arbitrary remote-code execution. The revision is pinned in
    code (ADR-0002 D2); re-pinning is a code change to :data:`REVISION` + re-sign-off, never
    a runtime argument — so the loaders accept only the pinned value (mirrors the Caduceus
    loader guard).
    """
    if revision != REVISION:
        raise ValueError(
            f"revision must be the code-pinned REVISION {REVISION!r}; loading remote code "
            f"(trust_remote_code=True) from another revision is not allowed, got {revision!r}"
        )


def load_nt_multispecies(*, revision: str = REVISION, device: str | None = None, dtype: Any = None):
    """Load the NT-multispecies backbone (masked-LM head) at the code-pinned ``revision``.

    Loaded as ``AutoModelForMaskedLM`` (the class the v2 model card + ``auto_map`` expose)
    with ``trust_remote_code=True``; the per-position **backbone** hidden state is read from
    ``output_hidden_states`` (see :func:`hidden_states`), so the LM head is never used for a
    prediction — only its encoder trunk feeds the seg head. Runs on **CPU** (stock ESM
    forward, no CUDA kernel required).

    Args:
        revision: must equal :data:`REVISION` (the only accepted value); anything else is
            rejected before the remote-code import.
        device: torch device string; defaults to ``"cuda"`` if available else ``"cpu"``.
        dtype: optional torch dtype override.

    Returns:
        The NT model in ``.eval()`` mode on ``device`` (config ``output_hidden_states=True``).
    """
    _require_pinned_revision(revision)
    import torch  # lazy
    from transformers import AutoModelForMaskedLM  # lazy

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForMaskedLM.from_pretrained(
        REPO_ID, revision=revision, trust_remote_code=True, output_hidden_states=True
    )
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval()


def load_nt_tokenizer(*, revision: str = REVISION):
    """Load the NT **6-mer** tokenizer at the code-pinned ``revision``."""
    _require_pinned_revision(revision)
    from transformers import AutoTokenizer  # lazy

    return AutoTokenizer.from_pretrained(REPO_ID, revision=revision, trust_remote_code=True)


def hidden_states(output):
    """Extract the final-layer per-position hidden state ``(B, L_tok, HIDDEN_DIM)``.

    NT is loaded with ``output_hidden_states=True``, so the masked-LM output carries the
    per-layer ``hidden_states`` tuple (embeddings + one per encoder layer); ``[-1]`` is the
    final encoder-layer output — the backbone trunk the seg head consumes (the LM head is
    skipped). Falls back to ``last_hidden_state`` / positional access so a base-model output
    also works. This mirrors :func:`caduceus_backbone._hidden_states`' role.
    """
    hs = getattr(output, "hidden_states", None)
    if hs is not None:
        return hs[-1]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def count_parameters(model) -> int:
    """Total number of parameters in ``model`` (recorded in provenance; not gated)."""
    return int(sum(p.numel() for p in model.parameters()))


def interface_parity(model, tokenizer, *, dna: str, seed: int, device: str) -> dict[str, Any]:
    """Measure the drop-in interface parity on a real forward over a DNA fixture.

    Tokenises ``dna`` with the NT 6-mer tokeniser, runs the backbone, then feeds the
    resulting per-position hidden state through **both** drop-in paths and records their
    output shapes + finiteness (no pass/fail — the caller applies the gate):

      1. **Direct seg-head drop-in** — :class:`SegmentationHead` sized to
         :data:`HIDDEN_DIM`: ``(B, L_tok, HIDDEN_DIM) → (B, L_tok, 8)``. This is the
         primary contract (the head is width-parameterised).
      2. **Stage1Segmenter assembly** — the existing segmenter's backbone-free
         ``logits_from_hidden`` path with ``rc_combine="concat"`` and ``d_model =
         HIDDEN_DIM // 2``. NB: ``"concat"`` is a pure **identity pass-through** (its
         ``forward`` returns the hidden state unchanged); NT is **single-strand, not
         RC-equivariant**, so the RC-strand interpretation does **not** apply — the
         ``d_model = HIDDEN_DIM//2`` merely sizes the head to ``2*d_model == HIDDEN_DIM``.
         This shows the *same* segmenter assembly accepts an arbitrary-width backbone.

    Also records ``n_nucleotides`` vs ``n_tokens`` — quantifying the ~6× 6-mer collapse
    that breaks per-nt label alignment (the documented caveat).
    """
    import torch  # lazy

    from tbox_finder.models.seg_head import SegmentationHead  # lazy (torch at import)
    from tbox_finder.models.stage1_segmenter import Stage1Segmenter  # lazy

    torch.manual_seed(seed)
    enc = tokenizer(dna, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    n_tokens = int(input_ids.shape[1])

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = hidden_states(out)  # (B, L_tok, HIDDEN_DIM)

    if hidden.dim() != 3:
        raise RuntimeError(f"expected (B, L, d) hidden states, got shape {tuple(hidden.shape)}")
    d = int(hidden.shape[-1])
    if d % 2 != 0:
        raise RuntimeError(
            f"HIDDEN_DIM must be even for the concat-identity segmenter path, got {d}"
        )

    # 1) direct seg-head drop-in
    head = SegmentationHead(d).to(device).eval()
    with torch.no_grad():
        head_logits = head(hidden)
    # 2) existing Stage1Segmenter assembly (concat == identity pass-through; NT is not RC)
    segmenter = (
        Stage1Segmenter(backbone=None, rc_combine="concat", d_model=d // 2).to(device).eval()
    )
    with torch.no_grad():
        seg_logits = segmenter.logits_from_hidden(hidden)

    return {
        "dna_len_nt": len(dna),
        "n_nucleotides": len(dna),
        "n_tokens": n_tokens,
        "kmer_collapse_ratio": round(len(dna) / n_tokens, 4) if n_tokens else None,
        "hidden_shape": [int(v) for v in hidden.shape],
        "hidden_dtype": str(hidden.dtype),
        "seg_head_logits_shape": [int(v) for v in head_logits.shape],
        "seg_head_logits_finite": bool(torch.isfinite(head_logits).all().item()),
        "segmenter_logits_shape": [int(v) for v in seg_logits.shape],
        "segmenter_logits_finite": bool(torch.isfinite(seg_logits).all().item()),
        "segmenter_rc_combine": segmenter.rc_combine_mode,
        "num_classes": int(head.num_classes),
        "seed": int(seed),
    }


# ====================================================================================== #
# Orchestration — build the P1-10 report.
# ====================================================================================== #
def _env_block() -> dict[str, Any]:
    import torch  # lazy
    import transformers  # lazy

    # No hostname is recorded (public repo; a machine name is a needless personal identifier).
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": None,
        "device_capability": None,
    }
    if info["cuda_available"]:
        cap = torch.cuda.get_device_capability(0)
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = [int(cap[0]), int(cap[1])]
    return info


def build_report(
    *, revision: str = REVISION, seq_len: int, seed: int, device: str | None = None
) -> dict[str, Any]:
    """Load the checkpoint, run the interface-parity smoke, and return the report dict."""
    import torch  # lazy

    env = _env_block()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_nt_multispecies(revision=revision, device=device)
    tokenizer = load_nt_tokenizer(revision=revision)
    cfg = model.config

    hidden_dim = int(getattr(cfg, "hidden_size", HIDDEN_DIM))
    n_layer = int(getattr(cfg, "num_hidden_layers", N_LAYER))
    model_type = str(getattr(cfg, "model_type", MODEL_TYPE))

    dna = random_dna(seq_len, seed)
    ip = interface_parity(model, tokenizer, dna=dna, seed=seed, device=device)

    load_ok = True
    hidden_dim_ok = ip["hidden_shape"][-1] == HIDDEN_DIM and hidden_dim == HIDDEN_DIM
    seg_head_dropin_ok = (
        ip["seg_head_logits_shape"][-1] == NUM_CLASSES and ip["seg_head_logits_finite"]
    )
    segmenter_dropin_ok = (
        ip["segmenter_logits_shape"][-1] == NUM_CLASSES and ip["segmenter_logits_finite"]
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": "P1-10",
        "measured": True,
        "generated_by": "src/tbox_finder/models/nt_backbone.py",
        "checkpoint": {
            "repo_id": REPO_ID,
            "revision": revision,
            "hub_url": HUB_URL,
            "hidden_dim": hidden_dim,
            "n_layer": n_layer,
            "model_type": model_type,
            "trust_remote_code": True,
            "param_count": count_parameters(model),
            "config_transformers_version": getattr(cfg, "transformers_version", None),
            "torch_device": device,
            "role": "Stage-1 transfer-fallback backbone (ADR-0002 D7 rung 3); NOT wired as default",
        },
        "pretraining": {
            "collection": PRETRAINING_COLLECTION,
            "prokaryote_inclusive": PROKARYOTE_INCLUSIVE,
            "prokaryote_basis": "bacteria (667 species / 17.1 B nt of 850)",
            "archaea_included": ARCHAEA_INCLUDED,
            "model_card_states_prokaryotes": MODEL_CARD_STATES_PROKARYOTES,
            "domain_shift_note": (
                "Multispecies + prokaryote-inclusive (bacteria); the domain-shift remedy "
                "premise for a prokaryotic scan where the default Caduceus-PS is "
                "human-reference pretrained (ADR-0002 D7). Not triggered — the P1-07 "
                "transfer go/no-go returned GO (ADR-0002 A6)."
            ),
            "citations": PRETRAINING_CITATIONS,
        },
        "tokenization": {
            "kmer": KMER,
            "max_tokens": MAX_TOKENS,
            "caveat": (
                "6-mer tokeniser: L_token ~= L_nt/6, so the per-nt 8-class label scheme (ADR-0004) "
                "does NOT align 1:1 with NT tokens. Consuming this backbone at P2 requires a "
                "token->nucleotide label broadcast (6x/token); realised only if this fallback is "
                "triggered."
            ),
        },
        "env": env,
        "interface_parity": ip,
        "gate": None,
    }
    report["gate"] = {
        "load_ok": load_ok,
        "hidden_dim_ok": bool(hidden_dim_ok),
        "seg_head_dropin_ok": bool(seg_head_dropin_ok),
        "segmenter_dropin_ok": bool(segmenter_dropin_ok),
        "overall_pass": bool(
            load_ok and hidden_dim_ok and seg_head_dropin_ok and segmenter_dropin_ok
        ),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="P1-10 NT-multispecies backbone load + interface-parity drop-in gate"
    )
    p.add_argument("--out", default=DEFAULT_OUT)
    # No --revision flag: the checkpoint revision is code-pinned (REVISION; ADR-0002 D2) and
    # must not be a runtime argument, since trust_remote_code=True executes it (see loaders).
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN_NT, dest="seq_len")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default=None)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seq_len < KMER:
        parser.error(f"--seq-len must be >= {KMER} (the tokeniser k-mer size)")

    report = build_report(seq_len=args.seq_len, seed=args.seed, device=args.device)

    errs = validate_report(report)
    if errs:  # a self-inconsistent report must not be written as if valid
        print(json.dumps({"schema_errors": errs}, indent=2), file=sys.stderr)
        return 2

    # Weights are pulled from HF by revision (not a local file), so inputs/outputs are empty
    # and the checkpoint identity lives in extra.checkpoint (mirrors the P1-02 sidecar).
    write_provenance(
        args.out,
        rule="src/tbox_finder/models/nt_backbone.py :: main (P1-10, LOCAL)",
        script="src/tbox_finder/models/nt_backbone.py",
        seed=args.seed,
        env_lock="envs/ml-dna.conda-lock.yml",
        adr="ADR-0002 (D7 rung 3; D2 trust_remote_code ceiling; A4/A6)",
        extra=_sanitize(report),
    )
    g = report["gate"]
    print(json.dumps({"gate": g, "out": args.out}, indent=2))
    return 0 if g["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
