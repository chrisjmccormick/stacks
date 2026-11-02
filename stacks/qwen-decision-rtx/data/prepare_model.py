r"""Bank Qwen3.5-9B's text stack in the layout train_decision.py loads, so the trainer opens one
file and gets tensors already in the shapes its hand-written forward and backward use.

What changes, and nothing else:

  - per-layer matrices are **stacked** into (L, ...) banks: one set for the 24 Gated DeltaNet
    layers (`deltanet_*`), one for the 8 softmax-attention layers (`attention_*`);
  - each layer's input projections are **concatenated** into one GEMM: [qkv | z | b | a] in a
    DeltaNet layer, [q | gate | k | v] in an attention layer (the checkpoint interleaves q and
    gate per head; here each is one contiguous block), [gate | up] for every MLP;
  - the zero-centred RMSNorm weights get their 1 added, in fp32;
  - the convolution's taps are transposed to (4, channels), tap j reading x[t - 3 + j]; the
    decay is stored as -exp(A_log);
  - the LM head is reduced to its rows for the 255 answer codes (A..Z, AA.., each one token),
    which initialize the trainer's readout.

The vision tower, the MTP layer and the rest of the LM head stay on disk. Same values: bf16
where the checkpoint is bf16, the small fp32 derivations exact.

    python data/prepare_model.py

No command line: every knob is a constant below, the same rule train_decision.py applies to
its config. Host-only, no CUDA: reads the checkpoint's shards a tensor at a time and writes
one safetensors file, a JSON sidecar (arch, shapes, the code ids, provenance) and the
tokenizer into ~/.cache/qwen-decision-rtx/model/. PUSH = True also uploads them, with a
generated model card, to REPO_ID (needs a write-scoped HF_TOKEN).
"""
from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

MODEL_ID, MODEL_REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"   # the post-trained 9B; d3-flash's base too
REPO_ID = "ChrisMcCormick/qwen-decision-rtx"       # public model repo
PRIVATE = False
PUSH = False                                       # True = upload after building
OUT_DIR = Path.home() / ".cache" / "qwen-decision-rtx" / "model"
LICENSE = "apache-2.0"                             # Qwen3.5-9B's
SOURCE_REF = "qwen-decision-rtx"                   # the branch the card links the trainer on

# Qwen3.5-9B's text stack, asserted against the checkpoint's config.json below and echoed into
# the sidecar, where train_decision.py asserts them against its own config: the arch is pinned
# on both sides of the file.
ARCH = dict(n_layers=32, d_model=4096, d_mlp=12288, d_embed=248320, rms_eps=1e-6,
            # Gated DeltaNet layers: every layer but each fourth
            n_qk_heads=16, d_qk=128, n_v_heads=32, d_v=128, d_conv=4,
            # softmax attention layers, every fourth
            n_qo_heads=16, n_kv_heads=4, d_head=256, d_rotary=64, rope_theta=1e7)

N_CODES = 255

# What each bank is, one line each: the sidecar's table and the model card's are generated.
BANK_DOC = {
    "embed":               "token embedding table (the LM head is separate in Qwen3.5)",
    "final_norm":          "final RMSNorm weight, 1 + the checkpoint's zero-centred weight",
    "lm_head_codes":       "the LM head's rows for the 255 answer codes, in code order",
    "deltanet_in_norm":    "pre-mixer RMSNorm weight, 1 + zero-centred",
    "deltanet_in_proj":    "fused mixer input projection, rows [qkv (q 16x128, k 16x128, v 32x128) | z 4096 | b 32 | a 32]",
    "deltanet_conv":       "depthwise causal convolution taps, (4, 8192): tap j reads x[t - 3 + j]",
    "deltanet_decay":      "-exp(A_log), the per-head decay scale on softplus(a + dt_bias)",
    "deltanet_dt_bias":    "bias on a before the softplus",
    "deltanet_gate_norm":  "the gated RMSNorm's weight on the delta rule's output, per value dim",
    "deltanet_out_proj":   "mixer output projection",
    "deltanet_post_norm":  "pre-MLP RMSNorm weight, 1 + zero-centred",
    "deltanet_mlp_in":     "fused SwiGLU input projection, rows [gate | up]",
    "deltanet_mlp_out":    "SwiGLU output projection",
    "attention_in_norm":   "pre-mixer RMSNorm weight, 1 + zero-centred",
    "attention_in_proj":   "fused QKV projection, rows [q 16x256 | gate 16x256 | k 4x256 | v 4x256]",
    "attention_q_norm":    "per-head q RMSNorm weight, 1 + zero-centred",
    "attention_k_norm":    "per-head k RMSNorm weight, 1 + zero-centred",
    "attention_out_proj":  "attention output projection",
    "attention_post_norm": "pre-MLP RMSNorm weight, 1 + zero-centred",
    "attention_mlp_in":    "fused SwiGLU input projection, rows [gate | up]",
    "attention_mlp_out":   "SwiGLU output projection",
}


def answer_codes(tok):
    """d3's codes: A..Z then AA..ZZ, each kept when it is one token of its own not taken by an
    earlier code, the first 255. The data prep derives the same list and the trainer asserts
    the two agree."""
    codes, ids = [], []
    letters = string.ascii_uppercase
    for code in [*letters, *(a + b for a in letters for b in letters)]:
        alone = tok.encode(code, add_special_tokens=False).ids
        if len(alone) == 1 and alone[0] not in ids:
            codes.append(code); ids.append(alone[0])
            if len(codes) == N_CODES:
                return codes, ids
    raise ValueError(f"only {len(codes)} usable codes")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def md_table(rows: list[tuple], header: tuple) -> str:
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(str(header[i])), *(len(r[i]) for r in cells)) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (n + 2) for n in w) + "|"]
                     + [line(r) for r in cells])


def model_card(meta: dict, banks_name: str) -> str:
    rows = [(f"`{k}`", " x ".join(str(d) for d in v["shape"]), v["dtype"], BANK_DOC[k])
            for k, v in meta["banks"].items()]
    return f"""---
license: {LICENSE}
base_model: {MODEL_ID}
library_name: safetensors
inference: false
tags:
- qwen3_5
- decision-model
- lora
---

# qwen-decision-rtx: Qwen3.5-9B's text stack, banked

**Not a fine-tune.** These are {MODEL_ID}'s own weights (revision `{MODEL_REV[:8]}`), re-containered
into the layout that
[`train_decision.py`](https://github.com/chrisjmccormick/stacks/blob/{SOURCE_REF}/stacks/qwen-decision-rtx/train_decision.py)
loads: per-layer matrices **stacked** into `(L, ...)` banks, one set per layer kind, each layer's
input projections **concatenated** into one GEMM, the zero-centred norms with their 1 added, and
the LM head reduced to its rows for the 255 answer codes. {meta['n_params']:,} parameters, the
same values. The vision tower and the MTP layer are left out.

That script fine-tunes Qwen3.5-9B into a decision model with LoRA on one RTX PRO 6000, with a
hand-written forward and backward and no `transformers` dependency: it opens this one file and
gets tensors whose names, shapes and dtypes are the ones its code uses.

## Files

- `{banks_name}`: the banks ({meta['file_mb']} MB, sha256 `{meta['file_sha256']}`)
- `{banks_name.replace('.safetensors', '.json')}`: sidecar with the arch, every bank's shape and
  dtype, the answer codes and their token ids, and provenance. The trainer asserts the arch
  fields and the code ids against its own config and data.
- `tokenizer.json`: the source repo's tokenizer, verbatim (sha256 `{meta['tokenizer_sha256']}`).

## Banks

`L` = 24 for `deltanet_*` (layers 0, 1, 2, 4, 5, 6, ... in checkpoint order), 8 for
`attention_*` (layers 3, 7, 11, ...):

{md_table(rows, ("bank", "shape", "dtype", "what it is"))}

## Reproducing it

`data/prepare_model.py` in the repo above, from the pinned checkpoint. The file's sha256 is in
the sidecar.

## License

{LICENSE}, inherited from {MODEL_ID}. Cite Qwen3.5 for the weights.
"""


def main():
    model_dir = Path(snapshot_download(MODEL_ID, revision=MODEL_REV,
                                       allow_patterns=["*.safetensors", "*.json"]))
    hf = json.loads((model_dir / "config.json").read_text())["text_config"]
    rope = hf["rope_parameters"]
    assert (hf["num_hidden_layers"], hf["hidden_size"], hf["intermediate_size"], hf["vocab_size"], hf["rms_norm_eps"],
            hf["linear_num_key_heads"], hf["linear_key_head_dim"], hf["linear_num_value_heads"], hf["linear_value_head_dim"],
            hf["linear_conv_kernel_dim"], hf["num_attention_heads"], hf["num_key_value_heads"], hf["head_dim"],
            int(hf["head_dim"] * rope["partial_rotary_factor"]), rope["rope_theta"]) \
        == tuple(ARCH.values()), f"{MODEL_ID} does not match the hardcoded Qwen3.5-9B arch"
    assert hf["attn_output_gate"] and hf["full_attention_interval"] == 4
    assert hf["layer_types"] == ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(ARCH["n_layers"])]

    # Every tensor by name, read from its shard on demand (mmap), so only the banks sit in memory.
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {file: safe_open(str(model_dir / file), framework="pt", device="cpu") for file in sorted(set(index.values()))}
    unread = set(index)
    def get(name):
        unread.discard(name)
        return shards[index[name]].get_tensor(name)
    f32 = lambda name: get(name).float()                                 # exact: the checkpoint is bf16

    L = ARCH["n_layers"]
    deltanet_layers = [i for i in range(L) if i % 4 != 3]
    attention_layers = [i for i in range(L) if i % 4 == 3]
    stack = lambda layers, make: torch.stack([make(f"model.language_model.layers.{i}.") for i in layers])

    out = {}
    out["embed"] = get("model.language_model.embed_tokens.weight")
    out["final_norm"] = 1 + f32("model.language_model.norm.weight")

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    codes, code_ids = answer_codes(tok)
    out["lm_head_codes"] = get("lm_head.weight")[code_ids].contiguous()

    out["deltanet_in_norm"] = stack(deltanet_layers, lambda p: 1 + f32(p + "input_layernorm.weight"))
    out["deltanet_in_proj"] = stack(deltanet_layers, lambda p: torch.cat([get(p + "linear_attn.in_proj_qkv.weight"),
                                                                          get(p + "linear_attn.in_proj_z.weight"),
                                                                          get(p + "linear_attn.in_proj_b.weight"),
                                                                          get(p + "linear_attn.in_proj_a.weight")]))
    # conv1d.weight is (channels, 1, 4), tap 3 the current token: (4, channels), tap j reading x[t - 3 + j]
    out["deltanet_conv"] = stack(deltanet_layers, lambda p: f32(p + "linear_attn.conv1d.weight")[:, 0, :].T.contiguous())
    out["deltanet_decay"] = stack(deltanet_layers, lambda p: -f32(p + "linear_attn.A_log").exp())
    out["deltanet_dt_bias"] = stack(deltanet_layers, lambda p: f32(p + "linear_attn.dt_bias"))
    out["deltanet_gate_norm"] = stack(deltanet_layers, lambda p: f32(p + "linear_attn.norm.weight"))   # not zero-centred
    out["deltanet_out_proj"] = stack(deltanet_layers, lambda p: get(p + "linear_attn.out_proj.weight"))
    out["deltanet_post_norm"] = stack(deltanet_layers, lambda p: 1 + f32(p + "post_attention_layernorm.weight"))
    out["deltanet_mlp_in"] = stack(deltanet_layers, lambda p: torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]))
    out["deltanet_mlp_out"] = stack(deltanet_layers, lambda p: get(p + "mlp.down_proj.weight"))

    # q_proj's rows are [q | gate] per head; split them into one q block and one gate block.
    def attention_in_proj(p):
        q_gate = get(p + "self_attn.q_proj.weight").view(ARCH["n_qo_heads"], 2, ARCH["d_head"], ARCH["d_model"])
        return torch.cat([q_gate[:, 0].reshape(-1, ARCH["d_model"]), q_gate[:, 1].reshape(-1, ARCH["d_model"]),
                          get(p + "self_attn.k_proj.weight"), get(p + "self_attn.v_proj.weight")])
    out["attention_in_norm"] = stack(attention_layers, lambda p: 1 + f32(p + "input_layernorm.weight"))
    out["attention_in_proj"] = stack(attention_layers, attention_in_proj)
    out["attention_q_norm"] = stack(attention_layers, lambda p: 1 + f32(p + "self_attn.q_norm.weight"))
    out["attention_k_norm"] = stack(attention_layers, lambda p: 1 + f32(p + "self_attn.k_norm.weight"))
    out["attention_out_proj"] = stack(attention_layers, lambda p: get(p + "self_attn.o_proj.weight"))
    out["attention_post_norm"] = stack(attention_layers, lambda p: 1 + f32(p + "post_attention_layernorm.weight"))
    out["attention_mlp_in"] = stack(attention_layers, lambda p: torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]))
    out["attention_mlp_out"] = stack(attention_layers, lambda p: get(p + "mlp.down_proj.weight"))
    out = {k: v.contiguous() for k, v in out.items()}

    # Everything in the text stack was read; what remains is the vision tower, the MTP layer and the LM head.
    assert all(n.startswith(("model.visual.", "mtp.")) or n == "lm_head.weight" for n in unread), sorted(unread)[:5]

    D, F, R = ARCH["d_model"], ARCH["d_mlp"], len(code_ids)
    d_qkv = 2 * ARCH["n_qk_heads"] * ARCH["d_qk"] + ARCH["n_v_heads"] * ARCH["d_v"]
    assert out["embed"].shape == (ARCH["d_embed"], D) and out["lm_head_codes"].shape == (R, D)
    assert out["deltanet_in_proj"].shape == (24, d_qkv + ARCH["n_v_heads"] * ARCH["d_v"] + 2 * ARCH["n_v_heads"], D)
    assert out["deltanet_conv"].shape == (24, ARCH["d_conv"], d_qkv)
    assert out["deltanet_gate_norm"].shape == (24, ARCH["d_v"])
    assert out["attention_in_proj"].shape == (8, 2 * ARCH["n_qo_heads"] * ARCH["d_head"] + 2 * ARCH["n_kv_heads"] * ARCH["d_head"], D)
    assert out["attention_q_norm"].shape == (8, ARCH["d_head"])
    assert out["deltanet_mlp_in"].shape == (24, 2 * F, D) and out["attention_mlp_out"].shape == (8, D, F)
    for name, t in out.items():
        assert t.dtype == (torch.float32 if name.endswith(("_norm", "_conv", "_decay", "_dt_bias")) else torch.bfloat16), name
        assert torch.isfinite(t).all(), name

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "qwen3.5-9b-decision.safetensors"
    save_file(out, str(path), metadata=dict(model_id=MODEL_ID, revision=MODEL_REV))

    tok_path = OUT_DIR / "tokenizer.json"
    tok_path.write_bytes((model_dir / "tokenizer.json").read_bytes())

    n_params = sum(t.numel() for t in out.values()) - out["lm_head_codes"].numel()
    meta = dict(model_id=MODEL_ID, revision=MODEL_REV, repo_id=REPO_ID, arch=ARCH, n_params=n_params,
                n_params_non_embedding=n_params - out["embed"].numel(),
                banks={k: dict(shape=list(t.shape), dtype=str(t.dtype).removeprefix("torch.")) for k, t in out.items()},
                codes=codes, code_ids=code_ids,
                file_mb=round(path.stat().st_size / 2**20), file_sha256=sha256_of(path), tokenizer_sha256=sha256_of(tok_path))
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=1))

    card = model_card(meta, path.name)
    card_path = OUT_DIR / "model_card.md"
    card_path.write_text(card, encoding="utf-8")

    print(f"[{MODEL_ID}] {n_params:,} params ({meta['n_params_non_embedding']:,} non-embedding) in {len(out)} banks")
    print(f"  codes: {codes[0]}..{codes[-1]} ({len(codes)}), ids {code_ids[:3]}..")
    print(f"  sha256: {meta['file_sha256']}")
    print(f"  -> {path} ({meta['file_mb']} MB)\n  -> {sidecar}\n  -> {tok_path}\n  -> {card_path}\n")
    print(md_table([(f"`{k}`", " x ".join(str(d) for d in v["shape"]), v["dtype"]) for k, v in meta["banks"].items()],
                   ("bank", "shape", "dtype")))

    if not PUSH:
        print("\nPUSH = False: not uploading.")
        return

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=REPO_ID, commit_message="model card")
    api.upload_folder(folder_path=str(OUT_DIR), repo_id=REPO_ID, allow_patterns=[path.name, sidecar.name, tok_path.name],
                      commit_message=f"banks from {MODEL_ID} @ {MODEL_REV[:8]} (sha {meta['file_sha256'][:12]})")
    print(f"\n-> https://huggingface.co/{REPO_ID} ({'private' if PRIVATE else 'public'})")


if __name__ == "__main__":
    main()
