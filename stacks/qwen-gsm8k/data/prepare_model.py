r"""Download the HF checkpoint and convert it into the banked layout the
trainer runs on, once, to a local safetensors file.

The trainer's weights are not an nn.Module and not the checkpoint's tensors:
per-layer matrices are STACKED into (L, ...) banks, and QKV / gate-in are
CONCATENATED into one GEMM each — (L, 1152, 896) and (L, 9728, 896) — which is
~72 fewer kernel launches per decode step and free throughput at 0.5B, where the
step is launch-bound rather than bandwidth-bound. That reshape used to happen
inside train_qwen_gsm8k.py on every run: a hub round-trip, a config.json audit,
24 x 9 dict lookups and a stack per bank. It is a property of the checkpoint,
not of the run, so it belongs here — the trainer now opens one file and gets
tensors whose names, shapes and dtype are already the ones it uses.

Deliberately NOT committed: ~940 MB, and fully determined by MODEL_ID. It lands
in the same regenerable cache as the tokenized dataset; the sidecar JSON records
what produced it, including a sha256 of the file.

    python data/prepare_model.py

No command line: every knob is a constant below, the same rule
train_qwen_gsm8k.py applies to GSM8KConfig.

Host-only — no CUDA. The banks are assembled and written on the CPU; the trainer
loads them straight to device.

It also drops the repo's `tokenizer.json` next to the banks, so neither the
dataset prep nor the trainer touches the hub.

What stays with the trainer: the fp32 masters / AdamW moments (run state, and
zeros at init), and the rotary cos/sin caches (their length is `cfg.t_row`, a
training-budget quantity, not a checkpoint one).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = Path.home() / ".cache" / "qwen-gsm8k" / "data"

# Qwen2.5-0.5B, asserted against the checkpoint's config.json below and echoed
# into the sidecar, where the trainer asserts them against GSM8KConfig — so the
# arch is pinned on both sides of the file.
ARCH = dict(n_layers=24, d_model=896, n_qo_heads=14, n_kv_heads=2, d_head=64,
            d_mlp=4864, d_vocab=151936, rope_theta=1_000_000.0, rms_eps=1e-6)

# Trainer-side name -> the HF key(s) that make it. Several keys concatenate
# along dim 0 into one fused GEMM; a name with `{i}` banks over layers.
BANKS = {
    "embed":      ["embed_tokens.weight"],                       # tied: table AND lm_head
    "W_QKV":      ["layers.{i}.self_attn.q_proj.weight",
                   "layers.{i}.self_attn.k_proj.weight",
                   "layers.{i}.self_attn.v_proj.weight"],
    "b_QKV":      ["layers.{i}.self_attn.q_proj.bias",           # Qwen2.5 has QKV biases
                   "layers.{i}.self_attn.k_proj.bias",
                   "layers.{i}.self_attn.v_proj.bias"],
    "W_O":        ["layers.{i}.self_attn.o_proj.weight"],
    "W_gin":      ["layers.{i}.mlp.gate_proj.weight",            # [gate | in]
                   "layers.{i}.mlp.up_proj.weight"],               # HF: up_proj
    "W_out":      ["layers.{i}.mlp.down_proj.weight"],             # HF: down_proj
    "attn_norm":  ["layers.{i}.input_layernorm.weight"],
    "mlp_norm":   ["layers.{i}.post_attention_layernorm.weight"],
    "final_norm": ["norm.weight"],
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


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
    for name, keys in BANKS.items():
        if "{i}" in keys[0]:
            rows = [torch.cat([sd[k.format(i=i)] for k in keys], dim=0) if len(keys) > 1
                    else sd[keys[0].format(i=i)] for i in range(ARCH["n_layers"])]
            t = torch.stack(rows)
        else:
            t = sd[keys[0]]
        # The checkpoint is already bf16, so this is a no-op cast that documents
        # the invariant: the live weights ARE the checkpoint, bit-exact, and the
        # trainer's fp32 masters start with an all-zero mantissa because of it.
        out[name] = t.to(torch.bfloat16).contiguous()
    del sd

    L, D, V = ARCH["n_layers"], ARCH["d_model"], ARCH["d_vocab"]
    d_qkv = ARCH["n_qo_heads"] * ARCH["d_head"] + 2 * ARCH["n_kv_heads"] * ARCH["d_head"]
    assert out["embed"].shape == (V, D)
    assert out["W_QKV"].shape == (L, d_qkv, D) and out["b_QKV"].shape == (L, d_qkv)
    assert out["W_O"].shape == (L, D, D)
    assert out["W_gin"].shape == (L, 2 * ARCH["d_mlp"], D)
    assert out["W_out"].shape == (L, D, ARCH["d_mlp"])
    assert out["attn_norm"].shape == out["mlp_norm"].shape == (L, D)
    assert out["final_norm"].shape == (D,)
    assert all(t.dtype == torch.bfloat16 for t in out.values())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"banks_{MODEL_ID.split('/')[-1]}.safetensors"
    save_file(out, str(path))

    # The tokenizer travels with the weights. data/prepare_gsm8k.py renders the
    # prompts with it and the trainer decodes completions with it, both off this
    # local copy.
    tok_path = OUT_DIR / "tokenizer.json"
    tok_path.write_bytes((model_dir / "tokenizer.json").read_bytes())

    n_params = sum(t.numel() for t in out.values())
    meta = dict(model_id=MODEL_ID, **ARCH, n_params=n_params,
                n_params_non_embedding=n_params - out["embed"].numel(),
                banks={k: list(t.shape) for k, t in out.items()},
                dtype="bfloat16", file_sha256=sha256_of(path),
                tokenizer_sha256=sha256_of(tok_path))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    print(f"[{MODEL_ID}] {n_params:,} params "
          f"({meta['n_params_non_embedding']:,} non-embedding) in {len(out)} banks")
    print(f"  sha256: {meta['file_sha256']}")
    print(f"  -> {path} ({path.stat().st_size / 2**20:.0f} MB)")
    print(f"  -> {tok_path} ({tok_path.stat().st_size / 2**20:.1f} MB, "
          f"sha256 {meta['tokenizer_sha256']})")


if __name__ == "__main__":
    main()
