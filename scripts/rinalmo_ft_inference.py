"""P1-13 Test 2 — official released-checkpoint inference parity (diagnostic).

Loads RiNALMo's OWN released per-family SS-fine-tuned checkpoint
(rinalmo_giga_ss_archiveII-<fam>_ft.pt) — the fine-tuned ENCODER + trained SS head
that produced the PUBLISHED numbers — overlays it onto the multimolecule mirror
model, and runs INFERENCE-ONLY decode with OUR exact forward + decode + metric on
the held-out family. No training.

If the released weights reproduce the published F1 on our splits/metric, then our
forward + decode + metric + splits are all faithful, and the mirror-run parity FAIL
is purely the fresh-head REFIT (not a backbone or pipeline defect). If they also
underperform, a forward-code difference (e.g. native inference TokenDropout rescale)
is implicated.

Run (under tbox-ml-rna, CUDA_HOME set; PYTHONPATH=<worktree>/src):
  python rinalmo_ft_inference.py <family> <ft_ckpt.pt> <fam_fold_root> [out.json]
"""

import json
import sys

# --- args ---
FAMILY = sys.argv[1]  # canonical family key, e.g. "tRNA", "5S_rRNA", "23S_rRNA"
FT_CKPT = sys.argv[2]  # path to rinalmo_giga_ss_archiveII-<fam>_ft.pt
FAM_FOLD_ROOT = sys.argv[3]  # extracted ct/fam-fold root
OUT_JSON = sys.argv[4] if len(sys.argv) > 4 else f"ft_infer_{FAMILY}.json"

PUBLISHED = {  # non-weighted F1, Supp. Table S2 (rinalmo_published_target.json)
    "5S_rRNA": 0.884,
    "SRP_RNA": 0.700,
    "tRNA": 0.931,
    "tmRNA": 0.801,
    "RNaseP_RNA": 0.798,
    "group_I_intron": 0.657,
    "16S_rRNA": 0.736,
    "23S_rRNA": 0.848,
    "telomerase_RNA": 0.120,
}
# our measured (fresh-head refit) mirror F1 for comparison
REFIT = {
    "5S_rRNA": 0.842408,
    "SRP_RNA": 0.682393,
    "tRNA": 0.888564,
    "tmRNA": 0.781139,
    "RNaseP_RNA": 0.773337,
    "group_I_intron": 0.680905,
    "16S_rRNA": 0.704994,
    "23S_rRNA": 0.815154,
    "telomerase_RNA": 0.095082,
}


def ss_name_map(key):
    """Verbatim multimolecule 0.1.0 _convert_checkpoint replaces (task='ss'), through
    the Wqkv split. Returns a list of (name, tensor) — Wqkv expands to 3."""
    if "inv_freq" in key or key in {"threshold"}:
        return []
    k = key
    k = k.replace("lm.", "")
    k = k.replace("gamma", "weight")
    k = k.replace("beta", "bias")
    k = k.replace("embedding", "model.embeddings.word_embeddings")
    k = k.replace("transformer", "model")
    k = k.replace("blocks", "encoder.layer")
    k = k.replace("mh_attn", "attention")
    k = k.replace("attn_layer_norm", "attention.layer_norm")
    k = k.replace("out_proj", "output.dense")
    k = k.replace("out_layer_norm", "layer_norm")
    k = k.replace("transition.0", "intermediate")
    k = k.replace("transition.2", "output.dense")
    k = k.replace("final_layer_norm", "encoder.layer_norm")
    k = k.replace("lm_mask_head.linear1", "lm_head.transform.dense")
    k = k.replace("lm_mask_head.layer_norm", "lm_head.transform.layer_norm")
    k = k.replace("lm_mask_head.linear2", "lm_head.decoder")
    k = k.replace("pred_head.linear_in", "ss_head.projection")
    k = k.replace("pred_head.resnet.encoder.layer", "ss_head.convnet.layers")
    k = k.replace("conv_net.0", "conv1")
    k = k.replace("conv_net.3", "conv2")
    k = k.replace("conv_net.6", "conv3")
    k = k.replace("pred_head.conv_out", "ss_head.prediction")
    return [k]  # Wqkv handled by caller (needs the tensor)


def main():
    import torch

    from tbox_finder.eval import rinalmo_parity as R

    # 1. Mirror model (correct 28-row word_embeddings + fresh ss_head) + tokenizer/device.
    model, tokenizer, device = R.load_rinalmo_ss()
    msd = model.state_dict()
    print(f"device={device}; model params={sum(p.numel() for p in model.parameters())}", flush=True)

    # 2. Load released ft checkpoint; name-map encoder + ss_head; overlay.
    raw = torch.load(FT_CKPT, map_location="cpu", weights_only=True)
    raw = raw.get("model", raw) if isinstance(raw, dict) else raw
    native_thr = None
    for kk in ("threshold", "lm.threshold"):
        if isinstance(raw, dict) and kk in raw:
            try:
                native_thr = float(raw[kk])
            except Exception:
                native_thr = None
    ss_native = sorted(k for k in raw if "pred_head" in k or "ss_head" in k)
    print(f"ft native SS-head keys (sample): {ss_native[:6]}", flush=True)

    overlay = {}
    for key, value in raw.items():
        mapped = ss_name_map(key)
        if not mapped:
            continue
        k = mapped[0]
        if "Wqkv" in k:
            q, kk_, v = (
                k.replace("Wqkv", "self.query"),
                k.replace("Wqkv", "self.key"),
                k.replace("Wqkv", "self.value"),
            )
            qv, kv, vv = value.chunk(3, dim=0)
            overlay[q], overlay[kk_], overlay[v] = qv, kv, vv
        else:
            overlay[k] = value

    # Keep only fine-tuned encoder + trained ss_head (drop word_embeddings/lm_head — the
    # mirror already carries correct-shape embeddings; embeddings were frozen in SS FT).
    keep = {
        k: v
        for k, v in overlay.items()
        if (k.startswith("model.encoder.") or k.startswith("ss_head."))
        and "word_embeddings" not in k
    }
    # Shape-check every kept tensor against the model before loading (fail loud).
    bad = [
        (k, tuple(v.shape), tuple(msd[k].shape))
        for k, v in keep.items()
        if k not in msd or msd[k].shape != v.shape
    ]
    if bad:
        print("SHAPE/NAME MISMATCH (first 10):", bad[:10], flush=True)
        raise SystemExit("ft->model name/shape map is wrong; inspect ss_native above")
    n_enc = sum(1 for k in keep if k.startswith("model.encoder."))
    n_head = sum(1 for k in keep if k.startswith("ss_head."))
    model.load_state_dict(keep, strict=False)
    # Assert the whole SS head + all encoder layers were overlaid.
    model_head = [k for k in msd if k.startswith("ss_head.")]
    model_enc = [k for k in msd if k.startswith("model.encoder.")]
    missing_head = [k for k in model_head if k not in keep]
    missing_enc = [k for k in model_enc if k not in keep]
    print(
        f"overlaid encoder tensors={n_enc}/{len(model_enc)}  ss_head={n_head}/{len(model_head)}",
        flush=True,
    )
    if missing_head or missing_enc:
        raise SystemExit(
            f"NOT fully overlaid: missing_head={missing_head[:5]} " f"missing_enc={missing_enc[:5]}"
        )

    # 3. Held-out family valid + test.
    _train, valid_recs, test_recs = R.load_fold_records(FAM_FOLD_ROOT, FAMILY)
    print(f"{FAMILY}: valid={len(valid_recs)} test={len(test_recs)}", flush=True)

    # 4. Inference: tune threshold on valid (our method), score test; also native threshold.
    def predict(records, tile_rows=48):
        """Memory-robust per-structure forward: small tile + empty_cache; on a CUDA
        error (WSL2 8GB card, long seqs) fall back to a CPU forward for that structure."""
        model.eval()
        out = []
        cpu_model = None
        for n, rec in enumerate(records):
            seq = R.normalize_rna(rec.sequence)
            try:
                ids = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=device.split(":")[0],
                        dtype=torch.float16,
                        enabled=(device != "cpu"),
                    ),
                ):
                    logits = R.contact_logits(model, ids, tile_rows=tile_rows)
                prob = torch.sigmoid(logits.float())[0].cpu().tolist()
            except RuntimeError as e:
                torch.cuda.empty_cache()
                if cpu_model is None:
                    cpu_model = model.to("cpu")  # move once; big seqs go on CPU
                ids = tokenizer(seq, return_tensors="pt")["input_ids"]
                with torch.no_grad():
                    logits = R.contact_logits(cpu_model, ids, tile_rows=tile_rows)
                prob = torch.sigmoid(logits.float())[0].tolist()
                print(f"  [cpu-fallback] rec {n} len={len(seq)} ({str(e)[:40]})", flush=True)
            out.append((seq, prob, rec.pairs))
            if device != "cpu" and cpu_model is None:
                torch.cuda.empty_cache()
        if cpu_model is not None:
            model.to(device)
        return out

    val_preds = predict(valid_recs)
    tuned_t, val_f1 = R.tune_threshold(val_preds)
    test_preds = predict(test_recs)
    p_t, r_t, f1_t = R.score_predictions(test_preds, tuned_t)
    native_row = None
    if native_thr is not None:
        p_n, r_n, f1_n = R.score_predictions(test_preds, native_thr)
        native_row = {
            "threshold": native_thr,
            "precision": round(p_n, 6),
            "recall": round(r_n, 6),
            "f1": round(f1_n, 6),
        }

    pub = PUBLISHED.get(FAMILY)
    report = {
        "test": "P1-13 Test 2 — official released *_ft.pt inference parity",
        "family": FAMILY,
        "ft_checkpoint": FT_CKPT,
        "n_valid": len(valid_recs),
        "n_test": len(test_recs),
        "our_tuned_threshold": tuned_t,
        "valid_f1_at_tuned": round(val_f1, 6),
        "test_f1_at_tuned_threshold": round(f1_t, 6),
        "test_precision_at_tuned": round(p_t, 6),
        "test_recall_at_tuned": round(r_t, 6),
        "test_at_native_threshold": native_row,
        "published_f1": pub,
        "our_fresh_head_refit_f1": REFIT.get(FAMILY),
        "delta_official_vs_published_pp": (round((f1_t - pub) * 100, 3) if pub else None),
        "delta_refit_vs_published_pp": (round((REFIT.get(FAMILY) - pub) * 100, 3) if pub else None),
        "device": device,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
