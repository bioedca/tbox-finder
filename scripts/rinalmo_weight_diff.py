"""P1-13 Test 1 — official-vs-mirror ENCODER weight-tensor diff (diagnostic).

Converts the official Zenodo `rinalmo_giga_pretrained.pt` native state_dict into
multimolecule names using the EXACT installed-0.1.0 name-map + Wqkv split (copied
verbatim from multimolecule/models/rinalmo/convert_checkpoint.py::_convert_checkpoint),
then diffs every ENCODER tensor against the published `multimolecule/rinalmo-giga`
mirror. Vocab-indexed rows (word_embeddings / lm_head.decoder) are EXCLUDED — those
are legitimately remapped by convert_word_embeddings; the encoder is pure rename+split.

Identical encoder tensors  => the mirror faithfully carries the official ENCODER
weights (the parity gap is NOT the encoder weights). Material divergence => a
conversion/source divergence is confirmed. CPU-only; no forward, no flash_attn.
"""

import json
import sys

import torch

OFFICIAL_PT = sys.argv[1] if len(sys.argv) > 1 else "rinalmo_giga_pretrained.pt"
OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else "rinalmo_weight_diff.json"
MIRROR_REPO = "multimolecule/rinalmo-giga"
MIRROR_REV = "2a71f6f98fb41dd2e6542a5e131d3778111d1468"


def convert_names(orig_sd):
    """Verbatim encoder/LM name-map from multimolecule 0.1.0 _convert_checkpoint
    (task=None path; the ss-only replaces are inert on a pretrained backbone)."""
    out = {}
    for key, value in orig_sd.items():
        if "inv_freq" in key or key in {"threshold"}:
            continue
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
        if "Wqkv" in k:
            q, kk, v = (
                k.replace("Wqkv", "self.query"),
                k.replace("Wqkv", "self.key"),
                k.replace("Wqkv", "self.value"),
            )
            out[q], out[kk], out[v] = value.chunk(3, dim=0)
        else:
            out[k] = value
    return out


def is_encoder_key(k):
    # The encoder = the 33 transformer layers + the final encoder layer_norm.
    # Exclude the vocab-remapped word_embeddings and the LM head entirely.
    if "word_embeddings" in k or "lm_head" in k:
        return False
    return k.startswith("model.encoder.") or k.startswith("encoder.")


def strip_model_prefix(k):
    return k[len("model.") :] if k.startswith("model.") else k


def main():
    print(f"Loading official native checkpoint: {OFFICIAL_PT}", flush=True)
    ckpt = torch.load(OFFICIAL_PT, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    conv = convert_names(ckpt)
    print(f"  official tensors: {len(ckpt)} -> converted: {len(conv)}", flush=True)

    print(f"Loading mirror {MIRROR_REPO}@{MIRROR_REV[:8]} (from HF cache)", flush=True)
    from multimolecule import RiNALMoForPreTraining  # noqa: PLC0415

    mirror = RiNALMoForPreTraining.from_pretrained(MIRROR_REPO, revision=MIRROR_REV)
    mirror_sd = mirror.state_dict()
    print(f"  mirror tensors: {len(mirror_sd)}", flush=True)

    # Align on model.-prefix-stripped names so RiNALMoModel-vs-ForPreTraining prefixes match.
    conv_enc = {strip_model_prefix(k): v for k, v in conv.items() if is_encoder_key(k)}
    mir_enc = {strip_model_prefix(k): v for k, v in mirror_sd.items() if is_encoder_key(k)}

    shared = sorted(set(conv_enc) & set(mir_enc))
    only_conv = sorted(set(conv_enc) - set(mir_enc))
    only_mir = sorted(set(mir_enc) - set(conv_enc))

    per_key = []  # shape mismatches (a name/arch drift, not a value diff)
    worst = []
    n_identical = 0  # within atol 1e-6
    n_bit_identical = 0  # exact: same dtype + shape + values (torch.equal)
    global_max = 0.0
    for k in shared:
        ao, bo = conv_enc[k], mir_enc[k]  # ORIGINAL dtype (both fp32) for the exact check
        if ao.shape != bo.shape or ao.dtype != bo.dtype:
            per_key.append(
                {
                    "key": k,
                    "mismatch": [list(ao.shape), str(ao.dtype), list(bo.shape), str(bo.dtype)],
                }
            )
            continue
        n_bit_identical += int(torch.equal(ao, bo))
        a, b = ao.float(), bo.float()
        mad = (a - b).abs().max().item()
        n_identical += int(mad <= 1e-6)
        global_max = max(global_max, mad)
        worst.append((mad, k))
    worst.sort(reverse=True)

    complete_keyset = len(shared) > 0 and not only_conv and not only_mir and not per_key
    report = {
        "test": "P1-13 Test 1 — official-vs-mirror encoder weight diff",
        "official_pt": OFFICIAL_PT,
        "mirror": f"{MIRROR_REPO}@{MIRROR_REV}",
        "n_encoder_keys_shared": len(shared),
        "n_encoder_keys_identical_atol_1e-6": n_identical,
        "n_encoder_keys_bit_identical_torch_equal": n_bit_identical,
        "encoder_global_max_abs_diff": global_max,
        # BYTE-identical requires the full key-set to match AND every shared tensor
        # to be exactly equal (same dtype + values) — not merely within tolerance.
        "all_encoder_bit_identical": complete_keyset and n_bit_identical == len(shared),
        "all_encoder_identical_atol_1e-6": complete_keyset and n_identical == len(shared),
        "shape_or_dtype_mismatches": per_key,
        "worst_10_encoder_max_abs_diff": [{"key": k, "max_abs_diff": m} for (m, k) in worst[:10]],
        "keys_only_in_converted_official": only_conv[:20],
        "keys_only_in_mirror": only_mir[:20],
        "n_only_conv": len(only_conv),
        "n_only_mir": len(only_mir),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
