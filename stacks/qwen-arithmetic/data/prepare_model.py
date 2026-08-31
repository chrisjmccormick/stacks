r"""Bank the HF checkpoint in **fp16** and publish it to the hub, so
`train_qwen_arithmetic.py` DOWNLOADS its weights instead of building them.

The raw-H100 baseline has its own bf16 banker
(`baselines/20260813_raw-h100/data/prepare_model.py`) rather than this taking a
dtype flag: the T4 artifact is allowed to diverge from the H100 one. Today it
differs only in dtype; a Turing-specific layout (bank order, padding, a
pre-transposed GEMM operand) would land here without touching that baseline.

fp16 rather than bf16, because that is what the T4 actually runs — it has no
bf16 tensor cores, so the trainer's live weights, activations and KV cache are
all fp16. Storing the banks in the run dtype means:

  - the trainer's load is a straight `load_file(...)` to device, with no
    1 GB bf16 -> fp16 transient on a 16 GB card;
  - nothing in the T4 path ever touches a bf16 kernel on sm75.

It changes NO number. The checkpoint is bf16, bf16's 7 explicit mantissa bits
fit inside fp16's 10, and the trainer already cast bf16 -> fp16 at load — so
`checkpoint -> fp16` here is bit-for-bit the live tensor the H100-banked path
produced (asserted below, per bank). Both directions lose the same handful of
tiny weights to fp16's subnormal floor; the count is recorded in the sidecar.

    python data/prepare_model.py

No command line: every knob is a constant below, the same rule
train_qwen_arithmetic.py applies to T4Config. Set PUSH = False to build the
banks without touching the hub.

Host-only — no CUDA. Writes into the same regenerable cache the baseline
scripts use, under a distinct `banks_fp16_*` name so the two live side by side,
then
uploads banks + sidecar + tokenizer + a generated model card to REPO_ID.

Publishing needs a write-scoped HF token (`HF_TOKEN` in the environment).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REPO_ID = "ChrisMcCormick/qwen-arithmetic-t4"      # public model repo
PRIVATE = False
PUSH = True                                        # False = build locally only
OUT_DIR = Path.home() / ".cache" / "qwen-arithmetic" / "data"
DTYPE = torch.float16
LICENSE = "apache-2.0"                             # Qwen2.5-0.5B-Instruct's
SOURCE_REF = "qwen-gsm8k"                          # branch the card links the
                                                   # trainer on -> "main" once
                                                   # this one merges (rerun to
                                                   # republish the card)

# Qwen2.5-0.5B, asserted against the checkpoint's config.json below and echoed
# into the sidecar, where the trainer asserts them against T4Config — so the
# arch is pinned on both sides of the file.
ARCH = dict(n_layers=24, d_model=896, n_qo_heads=14, n_kv_heads=2, d_head=64,
            d_mlp=4864, d_vocab=151936, rope_theta=1_000_000.0, rms_eps=1e-6)

# Trainer-side name -> the HF key(s) that make it. Several keys concatenate
# along dim 0 into one fused GEMM; a name with `{i}` banks over layers.
# Identical to prepare_model.py: the LAYOUT is shared, only the dtype is not.
BANKS = {
    "embed":      ["embed_tokens.weight"],                       # tied: table AND lm_head
    "W_QKV":      ["layers.{i}.self_attn.q_proj.weight",
                   "layers.{i}.self_attn.k_proj.weight",
                   "layers.{i}.self_attn.v_proj.weight"],
    "b_QKV":      ["layers.{i}.self_attn.q_proj.bias",           # Qwen2.5 has QKV biases
                   "layers.{i}.self_attn.k_proj.bias",
                   "layers.{i}.self_attn.v_proj.bias"],
    "W_O":        ["layers.{i}.self_attn.o_proj.weight"],
    "W_GU":       ["layers.{i}.mlp.gate_proj.weight",            # [gate | up]
                   "layers.{i}.mlp.up_proj.weight"],
    "W_down":     ["layers.{i}.mlp.down_proj.weight"],
    "attn_norm":  ["layers.{i}.input_layernorm.weight"],
    "mlp_norm":   ["layers.{i}.post_attention_layernorm.weight"],
    "final_norm": ["norm.weight"],
}

# What the bank is, one line each — the model card's table is generated, and
# a shape alone does not say which GEMM it feeds.
BANK_DOC = {
    "embed":      "token embedding table; TIED, so it is also the lm_head",
    "W_QKV":      "fused QKV projection, rows [Q | K | V] (14 Q heads, 2 KV)",
    "b_QKV":      "fused QKV bias, same row split",
    "W_O":        "attention output projection",
    "W_GU":       "fused SwiGLU input projection, rows [gate | up]",
    "W_down":     "SwiGLU output projection",
    "attn_norm":  "pre-attention RMSNorm weight (input_layernorm)",
    "mlp_norm":   "pre-MLP RMSNorm weight (post_attention_layernorm)",
    "final_norm": "final RMSNorm weight",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def md_table(rows: list[tuple], header: tuple) -> str:
    """A markdown table, generated — no hand-typed numbers anywhere downstream."""
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(str(header[i])), *(len(r[i]) for r in cells)) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (n + 2) for n in w) + "|"]
                     + [line(r) for r in cells])


def model_card(meta: dict, banks_name: str) -> str:
    rows = [(f"`{k}`", " x ".join(str(d) for d in v), f"{BANK_DOC[k]}")
            for k, v in meta["banks"].items()]
    return f"""---
license: {LICENSE}
base_model: {MODEL_ID}
library_name: safetensors
inference: false
tags:
- qwen2
- grpo
- speedrun
- t4
---

# qwen-arithmetic-t4 — banked fp16 weights

**Not a fine-tune.** These are {MODEL_ID}'s own weights, re-containered into
the layout that
[`train_qwen_arithmetic.py`](https://github.com/chrisjmccormick/stacks/blob/{SOURCE_REF}/stacks/qwen-arithmetic/train_qwen_arithmetic.py)
loads: per-layer matrices **stacked** into `(L, ...)` banks, QKV and gate/up
**concatenated** into one GEMM each, cast to **fp16**. Same
{meta['n_params']:,} parameters, same values.

That script is a single-file GRPO speedrun of Qwen2.5-0.5B-Instruct on
arithmetic for ONE Tesla T4 — the free Colab GPU. It has no `transformers`
dependency and no `nn.Module`: it opens one file and gets tensors whose names,
shapes and dtype are already the ones its handwritten forward/backward and its
CUDA-graph decode engine use. This repo is that file, so a Colab session spends
its first minute downloading ~{meta['file_mb']} MB instead of pulling the
checkpoint and rebuilding the banks on two vCPUs.

The trainer fetches this automatically — there is nothing to do by hand:

```bash
colab run --gpu T4 train_qwen_arithmetic.py --timeout 1h
```

## Files

- `{banks_name}` — the banks ({meta['file_mb']} MB, sha256 `{meta['file_sha256']}`)
- `{banks_name.replace('.safetensors', '.json')}` — sidecar: arch, shapes,
  provenance. The trainer asserts every arch field in it against its own config
  at load, so a mismatched bank file fails loudly instead of silently.
- `tokenizer.json` — the source repo's tokenizer, verbatim (sha256
  `{meta['tokenizer_sha256']}`), so neither the dataset prep nor the trainer
  touches another repo.

## Banks

{meta['n_params']:,} parameters ({meta['n_params_non_embedding']:,}
non-embedding) in {len(meta['banks'])} tensors, `L` = {ARCH['n_layers']}:

{md_table(rows, ("bank", "shape", "what it is"))}

## Why fp16

The T4 (sm75) has no bf16 tensor cores, so the trainer runs fp16 live weights
against fp32 masters. Storing the banks in the run dtype keeps a 1 GB bf16
transient off a 16 GB card at load and means nothing in the T4 path touches a
bf16 kernel.

The cast is free of consequence: the checkpoint is bf16, whose 7 explicit
mantissa bits fit inside fp16's 10, so these tensors are bit-for-bit what the
trainer used to produce by casting at load. {meta['underflow_to_zero']:,} of
{meta['n_params']:,} values ({100 * meta['underflow_to_zero'] / meta['n_params']:.4f}%)
land under fp16's subnormal floor and become zero — the same ones, either way.
Largest magnitude in the checkpoint is {meta['max_abs']:g}, against fp16's
65504 ceiling.

## Reproducing it

`data/prepare_model.py` in the repo above, from the pinned `{MODEL_ID}`
checkpoint. Both files' sha256 are in the sidecar.

## License

{LICENSE}, inherited from {MODEL_ID}. Cite Qwen2.5 for the weights.
"""


def main():
    model_dir = Path(snapshot_download(
        MODEL_ID, allow_patterns=["*.safetensors", "config.json", "tokenizer.json"]))
    hf = json.loads((model_dir / "config.json").read_text())
    assert (hf["num_hidden_layers"], hf["hidden_size"], hf["num_attention_heads"],
            hf["num_key_value_heads"], hf["intermediate_size"], hf["vocab_size"]) \
        == (ARCH["n_layers"], ARCH["d_model"], ARCH["n_qo_heads"],
            ARCH["n_kv_heads"], ARCH["d_mlp"], ARCH["d_vocab"]), \
        f"{MODEL_ID} does not match the hardcoded Qwen2.5-0.5B arch"
    assert hf.get("tie_word_embeddings", False), "expected tied embeddings"
    assert hf["rope_theta"] == ARCH["rope_theta"] and hf["rms_norm_eps"] == ARCH["rms_eps"]

    sd = load_file(str(next(model_dir.glob("*.safetensors"))))
    sd = {k.removeprefix("model."): v for k, v in sd.items()}

    out = {}
    n_underflow = 0
    max_abs = 0.0
    for name, keys in BANKS.items():
        if "{i}" in keys[0]:
            rows = [torch.cat([sd[k.format(i=i)] for k in keys], dim=0) if len(keys) > 1
                    else sd[keys[0].format(i=i)] for i in range(ARCH["n_layers"])]
            t = torch.stack(rows)
        else:
            t = sd[keys[0]]
        h = t.to(DTYPE).contiguous()
        # The equivalence the whole file rests on: this is EXACTLY the tensor
        # the trainer used to build by loading the raw-H100 baseline's bf16 bank and
        # casting it at load. Trivially true while the checkpoint is bf16 --
        # which is the point: it fails the day that stops being true.
        assert torch.equal(t.to(torch.bfloat16).to(DTYPE), h), f"{name}: bf16 detour differs"
        assert torch.isfinite(h).all(), f"{name}: overflowed fp16's 65504 ceiling"
        n_underflow += int(((t != 0) & (h == 0)).sum())
        max_abs = max(max_abs, float(t.abs().max()))
        out[name] = h
    del sd

    L, D, V = ARCH["n_layers"], ARCH["d_model"], ARCH["d_vocab"]
    d_qkv = ARCH["n_qo_heads"] * ARCH["d_head"] + 2 * ARCH["n_kv_heads"] * ARCH["d_head"]
    assert out["embed"].shape == (V, D)
    assert out["W_QKV"].shape == (L, d_qkv, D) and out["b_QKV"].shape == (L, d_qkv)
    assert out["W_O"].shape == (L, D, D)
    assert out["W_GU"].shape == (L, 2 * ARCH["d_mlp"], D)
    assert out["W_down"].shape == (L, D, ARCH["d_mlp"])
    assert out["attn_norm"].shape == out["mlp_norm"].shape == (L, D)
    assert out["final_norm"].shape == (D,)
    assert all(t.dtype == DTYPE for t in out.values())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"banks_fp16_{MODEL_ID.split('/')[-1]}.safetensors"
    save_file(out, str(path))

    # The tokenizer travels with the weights. data/prepare_arithmetic.py renders
    # the prompts with it and the trainer decodes completions with it, both off
    # this local copy -- which the trainer now gets from REPO_ID, not the hub's
    # copy of the source model.
    tok_path = OUT_DIR / "tokenizer.json"
    tok_path.write_bytes((model_dir / "tokenizer.json").read_bytes())

    n_params = sum(t.numel() for t in out.values())
    meta = dict(model_id=MODEL_ID, repo_id=REPO_ID, **ARCH, n_params=n_params,
                n_params_non_embedding=n_params - out["embed"].numel(),
                banks={k: list(t.shape) for k, t in out.items()},
                dtype=str(DTYPE).removeprefix("torch."),
                source_dtype="bfloat16", underflow_to_zero=n_underflow,
                max_abs=max_abs, file_mb=round(path.stat().st_size / 2**20),
                file_sha256=sha256_of(path), tokenizer_sha256=sha256_of(tok_path))
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=1))

    card = model_card(meta, path.name)
    card_path = OUT_DIR / "model_card.md"          # uploaded as the repo's README.md
    card_path.write_text(card, encoding="utf-8")

    print(f"[{MODEL_ID}] {n_params:,} params "
          f"({meta['n_params_non_embedding']:,} non-embedding) in {len(out)} banks, "
          f"{meta['dtype']} from {meta['source_dtype']}")
    print(f"  {n_underflow:,} values ({100 * n_underflow / n_params:.4f}%) underflowed "
          f"to zero | max |w| {max_abs:g} of fp16's 65504")
    print(f"  sha256: {meta['file_sha256']}")
    print(f"  -> {path} ({meta['file_mb']} MB)")
    print(f"  -> {sidecar}")
    print(f"  -> {tok_path} ({tok_path.stat().st_size / 2**20:.1f} MB, "
          f"sha256 {meta['tokenizer_sha256']})")
    print(f"  -> {card_path}")
    print()
    print(md_table([(f"`{k}`", " x ".join(str(d) for d in v)) for k, v in meta["banks"].items()],
                   ("bank", "shape")))

    if not PUSH:
        print("\nPUSH = False -- not uploading.")
        return

    # Idempotent: the hub skips a file whose hash already matches, so a rerun
    # after a partial upload costs a HEAD per file.
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=REPO_ID, commit_message="model card")
    api.upload_folder(folder_path=str(OUT_DIR), repo_id=REPO_ID,
                      allow_patterns=[path.name, sidecar.name, tok_path.name],
                      commit_message=f"banks {meta['dtype']} from {MODEL_ID} "
                                     f"(sha {meta['file_sha256'][:12]})")
    print(f"\n-> https://huggingface.co/{REPO_ID} "
          f"({'private' if PRIVATE else 'public'})")


if __name__ == "__main__":
    main()
