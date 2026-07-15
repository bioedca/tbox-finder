"""RiNALMo mirror parity — the heavy per-fold SS fine-tune + decode (P1-13).

This is the torch + ``multimolecule`` half of the RiNALMo mirror parity gate
(PRD §10.2; ADR-0002 D5). The pure-stdlib verdict/aggregation logic lives in the
sibling :mod:`tbox_finder.eval.archiveii_lofo`, which lazily calls
:func:`run_single_fold` here (``--mode fold``) once per leave-one-family-out
(LOFO) fold and then decides parity from the nine per-fold results
(``--mode aggregate``). Keeping the heavy path here means the verdict logic +
report validator stay bare-CI-importable (CLAUDE.md §8.4); **every torch /
multimolecule import in this module is lazy** (inside a function), so importing
``rinalmo_parity`` in bare CI does not pull torch.

Faithful reproduction of the published protocol
-----------------------------------------------
The gate reproduces RiNALMo's published ArchiveII nine-family inter-family LOFO
base-pair-F1 (Penić et al. 2025, Nat. Commun.; DOI:10.1038/s41467-025-60872-5;
PMC12219582). Every hyperparameter below is sourced from the authoritative
``lbcb-sci/RiNALMo`` repo (commit ``2c2c5c14``) + the paper Methods, cross-checked
against the ``multimolecule`` port whose fresh SS head is the one we fine-tune:

* **SS head = the mirror's own** ``RiNALMoSecondaryStructurePredictionHead``
  (freshly initialised on the pretrained encoder): pairwise-concat ``(i‖j)`` →
  ``Linear(2*1280 -> 64)`` → 2× bottleneck ResNet2D (Conv 1×1/3×3/1×1,
  ``InstanceNorm2d``, ReLU, bias-free, residual) → ``Conv2d(64 -> 1, k=3)`` →
  ``triu(1)+transpose`` symmetrise → ``(1, L, L)`` logits. Fine-tuning the *mirror's*
  head (not a re-implementation) is what certifies the mirror. The pairwise-concat
  is applied through a **row-tiled** projection so the ``L*L*2560`` intermediate is
  never materialised whole (a safety net for long inputs; the dl-rna splits are
  ≤512 nt, so a single tile is the norm).
* **Optimizer** = ``Adam`` (NOT AdamW), ``weight_decay=0`` — SS fine-tuning uses no
  decay (the 0.01 WD in RiNALMo ``config.py`` is pretraining-only).
* **Learning rate = 1e-4** — the paper Methods value ("10⁻⁴"). NB the released
  script's argparse ``--lr`` default is ``5e-4`` (the run-with-no-flag trap) and the
  wrapper class default ``1e-5`` is a decoy that the CLI overrides; **neither is the
  paper value**. Parity reproduces the paper, so ``1e-4`` governs.
* **LR schedule** = ``LinearLR`` 1.0×→0.1× over a hardcoded **7000 optimizer steps**,
  stepped per-step, **no warmup**; the factor floors at 0.1× thereafter.
* **15 epochs**, **batch size 1**, **fp16 mixed** precision, **no gradient clipping**.
* **Gradual unfreezing**, top-down, **3 layers every 3 epochs** starting epoch 3
  (epochs 0–2 train the head only). At 15 epochs the top 12 encoder layers +
  the final encoder ``layer_norm`` are unfrozen; blocks 0–20 + embeddings stay
  frozen. Newly-unfrozen encoder params train at **lr/10** (a second Adam param
  group). RiNALMo-giga has **33** transformer layers (``encoder.layer``).
* **Loss** = ``BCEWithLogitsLoss`` on the **strict upper triangle** of the predicted
  vs reference ``L×L`` contact map (no ``pos_weight``; the reference is the raw
  ``.ct`` pairing incl. pseudoknots).
* **Decoding** = sigmoid → mask the diagonal, sharp loops (``|i-j| < 4``) and
  non-canonical (AU/GC/GU) cells → threshold → **greedy one-pair-per-base** (take
  the highest-probability surviving pair, forbid its two bases, iterate). This is
  the RiNALMo ``_clean_sec_struct``: one pair per base, **pseudoknots allowed**
  (no non-crossing constraint). The threshold is tuned on the fold's ``valid`` set
  by maximising mean base-pair F1 over candidates ``0.01..0.29`` (final model; no
  early stopping / best-epoch selection in RiNALMo).
* **Metric** = ±1-nt-slippage base-pair F1 per structure, averaged **un-weighted**
  within the held-out family (the "non-weighted" column) — computed by the shared
  :func:`tbox_finder.eval.archiveii_lofo.base_pair_prf` (``zero_division=0.0``).

Env: ``tbox-ml-rna`` (transformers 5.13.0 + multimolecule 0.1.0; ADR-0002 A8).
``import multimolecule`` needs ``CUDA_HOME`` set (its deepspeed transitive import
probes it) — the sbatch + the local runner export ``CUDA_HOME=$CONDA_PREFIX``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from tbox_finder import provenance
from tbox_finder.eval import archiveii_lofo as A

# --------------------------------------------------------------------------- #
# Pinned checkpoint (ADR-0002 D5/A8) + protocol constants
# --------------------------------------------------------------------------- #
REPO_ID = "multimolecule/rinalmo-giga"
#: IMMUTABLE commit revision — never ``main`` (ADR-0002 D2/A8). Resolved + verified
#: against the HF Hub 2026-07-13; the mirror's config/classes are unchanged across
#: multimolecule 0.0.9→0.1.0 so the weights load byte-identically here.
REVISION = "2a71f6f98fb41dd2e6542a5e131d3778111d1468"
HUB_URL = f"https://huggingface.co/{REPO_ID}"

#: RiNALMo-giga architecture facts (config.json; asserted at load).
D_MODEL = 1280
N_LAYERS = 33

# --- fine-tuning protocol (lbcb-sci/RiNALMo @2c2c5c14 + paper Methods) --------
LR = 1e-4  # paper "10⁻⁴" (NOT the 5e-4 argparse trap nor the 1e-5 decoy)
WEIGHT_DECAY = 0.0  # Adam, no decay for SS fine-tuning
EPOCHS = 15
BATCH_SIZE = 1  # hardcoded upstream
LR_DECAY_STEPS = 7000  # LinearLR total_iters (hardcoded upstream)
LR_END_FACTOR = 0.1
UNFREEZE_START_EPOCH = 3  # epochs 0..2 = head only
UNFREEZE_EVERY = 3  # every 3 epochs
UNFREEZE_BLOCKS = 3  # 3 top layers per step
UNFREEZE_DENOM = 10.0  # newly-unfrozen encoder params at lr/10
SEED = 42
#: Threshold candidates tuned on valid F1 (upstream ``range(1, 30)/100``).
THRESHOLDS = tuple(round(i / 100, 2) for i in range(1, 30))  # 0.01..0.29
#: Guard: the dl-rna splits are ≤512 nt; a longer training input is cropped to
#: this (paper: 1024-token cap). Never fires on the pinned splits.
MAX_TRAIN_LEN = 1022
#: Row-tile size for the pairwise-concat projection (long-seq VRAM safety net).
HEAD_TILE_ROWS = 256

DEFAULT_FOLD_DIR = A.DEFAULT_FOLD_DIR
CANONICAL_PAIRS = A.CANONICAL_PAIRS
SHARP_LOOP_MIN_DIST = A.SHARP_LOOP_MIN_DIST


# --------------------------------------------------------------------------- #
# Pure helpers (torch-free → bare-CI testable)
# --------------------------------------------------------------------------- #
def _require_pinned_revision(revision: str) -> None:
    """Reject any revision other than the code-pinned :data:`REVISION`.

    The parity claim certifies *these exact weights*; a divergent revision would
    silently swap the checkpoint under a passing gate (mirrors the Caduceus/NT
    loader guards, CLAUDE.md §10.3).
    """
    if revision != REVISION:
        raise ValueError(
            f"revision must be the code-pinned REVISION {REVISION!r}; got {revision!r}"
        )


def _family_to_dir(family: str) -> str:
    """Canonical family key -> the archive directory name (inverse of FAMILY_DIRS)."""
    for dirname, fam in A.FAMILY_DIRS.items():
        if fam == family:
            return dirname
    raise ValueError(f"unknown family {family!r}; expected one of {list(A.FAMILY_ORDER)}")


def normalize_rna(sequence: str) -> str:
    """Upper-case + DNA→RNA (T→U) so the tokenizer sees canonical RNA letters."""
    return sequence.upper().replace("T", "U")


def linear_lr_factor(
    step: int, total_iters: int = LR_DECAY_STEPS, end: float = LR_END_FACTOR
) -> float:
    """``LinearLR`` multiplier: 1.0 at step 0 → ``end`` at ``total_iters``, floored after.

    Matches ``torch.optim.lr_scheduler.LinearLR(start_factor=1.0, end_factor=end,
    total_iters=total_iters)`` stepped per optimizer step.
    """
    if step >= total_iters:
        return end
    return 1.0 + (end - 1.0) * (step / total_iters)


def unfrozen_layer_indices(epoch: int) -> list[int]:
    """Top-down gradual-unfreeze schedule → the encoder-layer indices unfrozen by
    the *start* of ``epoch`` (0-based). Head-only for epochs 0..2; then 3 top
    layers every 3 epochs (33→...). Blocks not listed stay frozen.
    """
    if epoch < UNFREEZE_START_EPOCH:
        return []
    n_steps = (epoch - UNFREEZE_START_EPOCH) // UNFREEZE_EVERY + 1
    unfrozen: list[int] = []
    for s in range(n_steps):
        top = N_LAYERS - 1 - s * UNFREEZE_BLOCKS  # 32, 29, 26, ...
        for k in range(UNFREEZE_BLOCKS):
            idx = top - k
            if 0 <= idx < N_LAYERS:
                unfrozen.append(idx)
    return sorted(set(unfrozen))


def decode_pairs(prob, sequence: str, threshold: float) -> set[tuple[int, int]]:
    """RiNALMo ``_clean_sec_struct`` decode → a 1-based base-pair set.

    ``prob`` is an ``L×L`` symmetric probability map (any 2-D indexable: ``prob[i][j]``
    for python-``float`` cells). Applies, in order: the diagonal / sharp-loop
    (``|i-j| < SHARP_LOOP_MIN_DIST``) / non-canonical (AU/GC/GU) masks, the
    ``prob > threshold`` cut, then **greedy one-pair-per-base** — repeatedly take the
    highest-probability surviving pair and forbid both bases (pseudoknots allowed;
    no non-crossing constraint). Returns ``{(i, j)}`` with ``1 <= i < j <= L``.
    """
    seq = normalize_rna(sequence)
    n = len(seq)
    cands: list[tuple[float, int, int]] = []
    for i in range(n):
        bi = seq[i]
        row = prob[i]
        for j in range(i + SHARP_LOOP_MIN_DIST, n):
            if (bi, seq[j]) not in CANONICAL_PAIRS:
                continue
            p = float(row[j])
            if p > threshold:
                cands.append((p, i, j))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))  # highest prob first; deterministic ties
    used = [False] * n
    pairs: set[tuple[int, int]] = set()
    for _p, i, j in cands:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        pairs.add((i + 1, j + 1))  # 1-based to match reference pairs
    return pairs


def score_predictions(
    predictions: list[tuple[str, object, tuple[tuple[int, int], ...]]],
    threshold: float,
) -> tuple[float, float, float]:
    """Mean (precision, recall, F1) over structures at ``threshold`` — the RiNALMo
    ±1-slippage, ``zero_division=0.0`` base-pair metric, averaged un-weighted.

    ``predictions`` = ``[(sequence, prob_map, ref_pairs), ...]``. Empty input → 0.0.
    """
    if not predictions:
        return 0.0, 0.0, 0.0
    ps: list[float] = []
    rs: list[float] = []
    fs: list[float] = []
    for seq, prob, ref in predictions:
        pred = decode_pairs(prob, seq, threshold)
        p, r, f = A.base_pair_prf(pred, ref, slippage=True, seq_len=len(seq))
        ps.append(p)
        rs.append(r)
        fs.append(f)
    n = len(predictions)
    return sum(ps) / n, sum(rs) / n, sum(fs) / n


def tune_threshold(
    predictions: list[tuple[str, object, tuple[tuple[int, int], ...]]],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> tuple[float, float]:
    """Pick the threshold maximising mean base-pair F1 on ``predictions`` (RiNALMo
    tunes on the fold's ``valid`` set). Returns ``(best_threshold, best_mean_f1)``;
    ties break to the lower threshold (upstream sweep order 0.01→0.29)."""
    best_t = thresholds[0]
    best_f = -1.0
    for t in thresholds:
        _p, _r, f = score_predictions(predictions, t)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# --------------------------------------------------------------------------- #
# Data loading (reuses the P1-12 .ct parser)
# --------------------------------------------------------------------------- #
def load_fold_records(fam_fold_root: str | Path, family: str):
    """Load the fold's ``(train, valid, test)`` :class:`CtRecord` lists.

    For LOFO fold ``family``: ``test`` is that family held out; ``train``+``valid`` are
    the eight-family complement (dl-rna already routes the longest RNAs to
    ``valid``). Records are sorted by id for a deterministic epoch order.
    """
    root = Path(fam_fold_root)
    dirname = _family_to_dir(family)
    out: dict[str, list] = {}
    for role in ("train", "valid", "test"):
        d = root / dirname / role
        recs = [A.parse_ct_file(p) for p in sorted(d.glob("*.ct"))]
        if not recs:
            raise FileNotFoundError(f"no .ct records under {d}")
        out[role] = recs
    return out["train"], out["valid"], out["test"]


# --------------------------------------------------------------------------- #
# Model (all torch / multimolecule imports are lazy)
# --------------------------------------------------------------------------- #
def load_rinalmo_ss(*, revision: str = REVISION, device: str | None = None, seed: int = SEED):
    """Load ``RiNALMoForSecondaryStructurePrediction`` at the pinned revision.

    Returns ``(model, tokenizer, device)``. The pretrained encoder loads from the
    checkpoint; the ``ss_head`` is **freshly initialised** (its weights are absent
    from the base checkpoint) — exactly a per-fold fresh SS head. The RNG is seeded
    **before** ``from_pretrained`` so that fresh-head init is reproducible (§8.3). No
    ``trust_remote_code`` (multimolecule is transformers-native)."""
    _require_pinned_revision(revision)
    os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
    import torch  # lazy
    from multimolecule import RiNALMoForSecondaryStructurePrediction, RnaTokenizer  # lazy

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)  # reproducible fresh ss_head init
    tokenizer = RnaTokenizer.from_pretrained(REPO_ID, revision=revision)
    model = RiNALMoForSecondaryStructurePrediction.from_pretrained(REPO_ID, revision=revision)
    cfg = model.config
    if (
        getattr(cfg, "hidden_size", None) != D_MODEL
        or getattr(cfg, "num_hidden_layers", None) != N_LAYERS
    ):
        raise ValueError(
            f"checkpoint arch drift: hidden_size={getattr(cfg, 'hidden_size', None)} "
            f"num_hidden_layers={getattr(cfg, 'num_hidden_layers', None)} "
            f"(expected {D_MODEL}/{N_LAYERS})"
        )
    model.to(device)
    return model, tokenizer, device


def _encoder_and_head(model):
    """The base encoder (``RiNALMoModel``) + the SS-head submodules of the wrapper."""
    return model.model, model.ss_head


def contact_logits(model, input_ids, *, tile_rows: int = HEAD_TILE_ROWS):
    """Predicted symmetric ``(L, L)`` base-pair logits for one sequence (batch 1).

    Runs the encoder, strips the CLS/EOS tokens, then drives the mirror's own
    ``ss_head`` submodules (``projection`` → ``convnet`` → ``prediction``) with a
    **row-tiled** pairwise-concat+projection so the ``L*L*2*d_model`` intermediate is
    never materialised whole. Symmetrised (``triu(1)+transpose``) exactly as the head.
    Gradients flow to the encoder only if some encoder param requires grad (else the
    embeddings are detached — saves memory in the head-only epochs).
    """
    import torch  # lazy

    enc, head = _encoder_and_head(model)
    # Respect an enclosing ``no_grad`` (eval): ``set_grad_enabled(True)`` nested inside
    # ``no_grad`` would otherwise re-enable autograd and retain the whole 33-layer encoder
    # graph during inference (wasted VRAM). Only build a graph when grad is ambiently on
    # AND some encoder param is unfrozen (training with ≥1 layer unfrozen).
    enc_needs_grad = torch.is_grad_enabled() and any(p.requires_grad for p in enc.parameters())
    with torch.set_grad_enabled(enc_needs_grad):
        hidden = enc(input_ids).last_hidden_state  # (1, L+2, d)
    if input_ids.shape[1] < 2:
        raise ValueError("input_ids must include the CLS/EOS special tokens")
    emb = hidden[:, 1:-1, :]  # strip CLS + EOS → (1, L, d)
    if not enc_needs_grad:
        emb = emb.detach()
    _b, seq_len, d = emb.shape
    proj = head.projection  # Linear(2d -> C)
    rows: list = []
    step = max(1, int(tile_rows))
    for i0 in range(0, seq_len, step):
        i1 = min(i0 + step, seq_len)
        left = emb[:, i0:i1, :].unsqueeze(2).expand(1, i1 - i0, seq_len, d)
        right = emb.unsqueeze(1).expand(1, i1 - i0, seq_len, d)
        block = proj(torch.cat((left, right), dim=-1))  # (1, rows, L, C)
        rows.append(block)
    feat = torch.cat(rows, dim=1).permute(0, 3, 1, 2)  # (1, C, L, L)
    feat = head.convnet(feat)
    logits = head.prediction(feat)  # (1, 1, L, L)
    logits = logits.squeeze(1)  # (1, L, L)
    tri = torch.triu(logits, diagonal=1)
    return tri + tri.transpose(-1, -2)  # symmetric, zero diagonal


def reference_matrix(pairs, seq_len: int, *, device, dtype):
    """Symmetric ``(1, L, L)`` binary reference contact map from 1-based pairs."""
    import torch  # lazy

    m = torch.zeros((1, seq_len, seq_len), device=device, dtype=dtype)
    for i, j in pairs:
        m[0, i - 1, j - 1] = 1.0
        m[0, j - 1, i - 1] = 1.0
    return m


def _predict_probs(model, records, tokenizer, device, *, tile_rows: int = HEAD_TILE_ROWS):
    """Forward each record → ``(sequence, prob_map_as_lists, ref_pairs)`` for decode.

    Probabilities are moved to CPU python lists so threshold tuning + decode are
    torch-free (and memory stays bounded — the maps are ≤512×512)."""
    import torch  # lazy

    model.eval()
    out: list[tuple[str, object, tuple[tuple[int, int], ...]]] = []
    with torch.no_grad():
        for rec in records:
            seq = normalize_rna(rec.sequence)
            ids = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
            with torch.autocast(
                device_type=device.split(":")[0], dtype=torch.float16, enabled=(device != "cpu")
            ):
                logits = contact_logits(model, ids, tile_rows=tile_rows)
            prob = torch.sigmoid(logits.float())[0].cpu().tolist()
            out.append((seq, prob, rec.pairs))
    return out


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #
def _set_requires_grad(module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def apply_unfreeze(model, epoch: int) -> list[int]:
    """Set ``requires_grad`` per the gradual-unfreeze schedule for ``epoch``; the head
    is always trainable. Returns the sorted list of unfrozen encoder-layer indices."""
    enc, head = _encoder_and_head(model)
    _set_requires_grad(head, True)
    _set_requires_grad(enc, False)  # freeze everything, then unfreeze the schedule
    idxs = unfrozen_layer_indices(epoch)
    layers = enc.encoder.layer
    for idx in idxs:
        _set_requires_grad(layers[idx], True)
    if idxs:  # the final encoder layer_norm rides with the first unfreeze step
        _set_requires_grad(enc.encoder.layer_norm, True)
    return idxs


def build_optimizer(model, *, lr: float = LR, weight_decay: float = WEIGHT_DECAY):
    """Two Adam param groups: the SS head at ``lr``; ALL encoder params at ``lr/10``
    (only the currently-unfrozen ones receive gradients). Mirrors RiNALMo's
    ``initial_denom_lr=10`` for newly-unfrozen layers."""
    import torch  # lazy

    enc, head = _encoder_and_head(model)
    groups = [
        {"params": list(head.parameters()), "lr": lr, "_base_lr": lr},
        {
            "params": list(enc.parameters()),
            "lr": lr / UNFREEZE_DENOM,
            "_base_lr": lr / UNFREEZE_DENOM,
        },
    ]
    return torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay)


def _apply_lr(optimizer, step: int) -> float:
    """Scale every group's ``lr`` by the shared LinearLR factor for ``step``."""
    factor = linear_lr_factor(step)
    for g in optimizer.param_groups:
        g["lr"] = g["_base_lr"] * factor
    return factor


def train_fold(
    model,
    train_recs,
    tokenizer,
    device,
    *,
    epochs=EPOCHS,
    lr=LR,
    seed=SEED,
    tile_rows=HEAD_TILE_ROWS,
):
    """Fine-tune one LOFO fold in place (batch 1, fp16-mixed, BCE upper-tri).

    Returns the number of optimizer steps taken. Determinism: seeded shuffle +
    ``torch.manual_seed`` (the Mamba-free RiNALMo path has no non-deterministic
    kernel dependency, but eval parity is at the metric level regardless)."""
    import torch  # lazy
    import torch.nn as nn

    torch.manual_seed(seed)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device != "cpu"))
    rng = random.Random(seed)
    step = 0
    order = list(range(len(train_recs)))
    for epoch in range(epochs):
        apply_unfreeze(model, epoch)
        model.train()
        rng.shuffle(order)
        for k in order:
            rec = train_recs[k]
            seq = normalize_rna(rec.sequence)
            if len(seq) > MAX_TRAIN_LEN:  # paper 1024-token cap (never fires on dl-rna ≤512)
                start = rng.randint(0, len(seq) - MAX_TRAIN_LEN)
                seq, pairs = _crop(seq, rec.pairs, start, MAX_TRAIN_LEN)
            else:
                pairs = rec.pairs
            ids = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
            _apply_lr(optimizer, step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.split(":")[0], dtype=torch.float16, enabled=(device != "cpu")
            ):
                logits = contact_logits(model, ids, tile_rows=tile_rows)  # (1, L, L)
                ref = reference_matrix(pairs, len(seq), device=device, dtype=logits.dtype)
                mask = torch.triu(
                    torch.ones(len(seq), len(seq), dtype=torch.bool, device=device), diagonal=1
                )
                loss = loss_fn(logits[:, mask], ref[:, mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
    return step


def _crop(seq: str, pairs, start: int, length: int):
    """Crop ``seq[start:start+length]`` and re-index (drop pairs leaving the window)."""
    end = start + length
    new_pairs = tuple(
        (i - start, j - start) for (i, j) in pairs if start < i <= end and start < j <= end
    )
    return seq[start:end], new_pairs


# --------------------------------------------------------------------------- #
# The fold entry (called by archiveii_lofo.run_parity --mode fold)
# --------------------------------------------------------------------------- #
def run_single_fold(
    family: str,
    *,
    fam_fold_root: str | Path | None = None,
    fold_dir: str | Path = DEFAULT_FOLD_DIR,
    epochs: int = EPOCHS,
    lr: float = LR,
    seed: int = SEED,
    revision: str = REVISION,
    device: str | None = None,
    tile_rows: int = HEAD_TILE_ROWS,
) -> dict:
    """Fine-tune + score one LOFO fold; write ``fold_dir/<family>.json``; return it.

    The returned dict is the contract :func:`archiveii_lofo.aggregate_parity`
    consumes: ``family``, ``measured=True``, ``non_weighted_mean_f1`` ∈ [0,1],
    ``n_test``, ``tuned_threshold``, ``precision_mean``, ``recall_mean``,
    ``revision``, ``git_sha``, ``seed`` (+ ``_path``)."""
    if family not in A.FAMILY_ORDER:
        raise ValueError(f"unknown family {family!r}")
    if fam_fold_root is None:
        raise ValueError("fam_fold_root is required (extracted ct/fam-fold root)")
    _require_pinned_revision(revision)

    train_recs, valid_recs, test_recs = load_fold_records(fam_fold_root, family)
    model, tokenizer, device = load_rinalmo_ss(revision=revision, device=device, seed=seed)

    train_fold(
        model, train_recs, tokenizer, device, epochs=epochs, lr=lr, seed=seed, tile_rows=tile_rows
    )

    val_preds = _predict_probs(model, valid_recs, tokenizer, device, tile_rows=tile_rows)
    threshold, _val_f1 = tune_threshold(val_preds)

    test_preds = _predict_probs(model, test_recs, tokenizer, device, tile_rows=tile_rows)
    p_mean, r_mean, f1_mean = score_predictions(test_preds, threshold)

    result = {
        "schema_version": A.PARITY_SCHEMA_VERSION,
        "step": "P1-13",
        "family": family,
        "measured": True,
        "non_weighted_mean_f1": round(f1_mean, 6),
        "precision_mean": round(p_mean, 6),
        "recall_mean": round(r_mean, 6),
        "tuned_threshold": threshold,
        "n_test": len(test_recs),
        "n_train": len(train_recs),
        "n_valid": len(valid_recs),
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "revision": revision,
        "repo_id": REPO_ID,
        "git_sha": provenance.git_sha(),
        "device": device,
    }
    if not (0.0 <= result["non_weighted_mean_f1"] <= 1.0):
        raise ValueError(f"fold {family}: F1 out of [0,1]: {result['non_weighted_mean_f1']}")

    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    out_path = fold_dir / f"{family}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    result["_path"] = str(out_path)
    return result


def main(argv: list[str] | None = None) -> int:
    """Thin CLI (the canonical entry is ``archiveii_lofo.run_parity --mode fold``)."""
    parser = argparse.ArgumentParser(description="RiNALMo mirror parity — one LOFO fold (P1-13).")
    parser.add_argument("--family", required=True, choices=list(A.FAMILY_ORDER))
    parser.add_argument("--fam-fold-root", required=True)
    parser.add_argument("--fold-dir", default=str(DEFAULT_FOLD_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    out = run_single_fold(
        args.family,
        fam_fold_root=args.fam_fold_root,
        fold_dir=args.fold_dir,
        epochs=args.epochs,
        seed=args.seed,
    )
    f1 = out["non_weighted_mean_f1"]
    print(f"fold {args.family}: non_weighted_mean_f1={f1:.4f} -> {out['_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
