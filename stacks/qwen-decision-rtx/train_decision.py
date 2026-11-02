"""Qwen3.5-9B fine-tuned into a decision model with LoRA, its forward and backward written by hand, on one RTX PRO 6000.

Each question is one row: the state, then the question in encoder-rtx's special-token format, read out at the
readout token that ends each option (README). Rows are packed end to end into a token budget per step; the
optimizer steps each layer's LoRA weights inside the backward, as soon as their gradients exist, so a step is one
pack and there is no gradient accumulation. The activations the backward needs are kept from the forward rather
than recomputed.

    python train_decision.py            train, then evaluate on the panel (knobs below, each printed)
    CHECK=1 python train_decision.py    the forward against transformers' Qwen3.5 and the hand-written gradients
                                        against autograd, on a tiny random model in fp64 (GPU, no download)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

# ==============================================================================
# § Knobs
# ==============================================================================
# Read from the environment and printed, so a log carries the whole configuration.

def knob(name, default, cast=str):
    value = cast(os.environ.get(name, default))
    print(f"knob {name}={value}")
    return value

CHECK = knob("CHECK", "0") == "1"
RUN_NAME = knob("RUN_NAME", time.strftime("run-%m%d-%H%M"))
TRAIN_SPLITS = knob("TRAIN_SPLITS", "train,phase1_train").split(",")
VAL_SPLITS = knob("VAL_SPLITS", "decision_dev,transfer_dev,phase1_holdout,phase1_jevbench,cookbook_dev").split(",")
EPOCHS = knob("EPOCHS", "1", float)
TOKENS = knob("TOKENS", "16384", int)            # tokens per training pack (= per step)
EVAL_TOKENS = knob("EVAL_TOKENS", "32768", int)  # tokens per evaluation pack
LR = knob("LR", "2e-4", float)
RANK = knob("RANK", "16", int)
ALPHA = knob("ALPHA", "32", float)               # LoRA's update is (ALPHA / RANK) B A
SOFT_WEIGHT = knob("SOFT_WEIGHT", "1.0", float)  # target = SOFT_WEIGHT x a soft target (where one exists) + the rest on the label
STATE_CAP = knob("STATE_CAP", "8192", int)
VAL_EVERY = knob("VAL_EVERY", "0", int)          # 0: evaluate at the end only
ABORT_STEP = knob("ABORT_STEP", "0", int)        # stop after this many steps (timing runs)
SEED = knob("SEED", "42", int)
USE_WANDB = knob("USE_WANDB", "1") == "1"
SAVE_CHECKPOINT = knob("SAVE_CHECKPOINT", "1") == "1"

MODEL_REPO, MODEL_REV = "Qwen/Qwen3.5-9B", None   # pinned once the box has resolved it
CACHE = Path.home() / ".cache" / "qwen-decision-rtx"
DATA_DIR = CACHE / "decision"
RUN_DIR = CACHE / "runs" / RUN_NAME

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda", 0)

# ==============================================================================
# § Model
# ==============================================================================
# Qwen3.5-9B's text stack: 32 layers, every fourth softmax attention (causal, with a sigmoid output gate and
# per-head q / k norms), the rest Gated DeltaNet; one RMSNorm before each mixer and each MLP. The frozen weights
# live in flat per-layer lists (agent-ops-stacks encoder-rtx/2026-11-01_1135am_d3-varlen-engine/d3_varlen.py,
# checked there against transformers), with the projections that read one input fused into one matrix. The LM
# head is dropped: a single learned readout vector scores each option's readout token.

cfg = SimpleNamespace()
KIND = []                        # "linear" | "full" per layer
W_IN, W_POST = [], []            # (D,) fp32: 1 + the zero-centred RMSNorm weight
W_MIX_IN, W_MIX_OUT = [], []     # the mixer's fused input projection (qkv|z|b|a, or q+gate|k|v) and its output projection
CONV_W, A_NEG, DT_BIAS, W_GNORM = [], [], [], []   # Gated DeltaNet: (4, C) taps, -exp(A_log), dt_bias, the gated norm's weight
W_QN, W_KN = [], []              # attention: 1 + the per-head q / k RMSNorm weights (Dh,)
W_MLP_IN, W_MLP_OUT = [], []     # gate|up (2F, D) and down (D, F)
EMBED = W_FINAL = INV_FREQ = None
DT = torch.bfloat16              # the model's dtype (fp64 in the checks)
ACC = torch.float32              # the norms', decay's and convolution's dtype

# The trained parameters, fp32: LoRA's (A, B) per layer per projection, the readout vector, the special tokens'
# embedding rows. Each has AdamW moments beside it.
LORA_NAMES = ("mix_in", "mix_out", "mlp_in", "mlp_out")
LORA = []                        # per layer: {name: [A (r, in), B (out, r)]}
READOUT = None                   # (D,)
SPECIAL_ROWS = None              # (n_special, D)
SPECIAL_BASE = None              # the first special token's id
ADAM = {}                        # id(param) -> (m, v)
FA2 = None


def set_cfg(text):
    get = text.get if isinstance(text, dict) else lambda k, d=None: getattr(text, k, d)
    cfg.d = get("hidden_size"); cfg.f = get("intermediate_size"); cfg.layers = get("num_hidden_layers")
    cfg.hq = get("num_attention_heads"); cfg.hkv = get("num_key_value_heads"); cfg.dh = get("head_dim")
    cfg.hk = get("linear_num_key_heads"); cfg.hv = get("linear_num_value_heads")
    cfg.dk = get("linear_key_head_dim"); cfg.dv = get("linear_value_head_dim")
    cfg.conv = get("linear_conv_kernel_dim"); cfg.eps = get("rms_norm_eps")
    rope = get("rope_parameters") or {}
    cfg.rot = int(cfg.dh * rope.get("partial_rotary_factor", get("partial_rotary_factor", 1.0)))
    cfg.theta = rope.get("rope_theta", 1e7)
    KIND[:] = ["full" if t == "full_attention" else "linear" for t in get("layer_types")[:cfg.layers]]
    assert cfg.conv == 4 and cfg.hv % cfg.hk == 0 and cfg.hq % cfg.hkv == 0


def load_weights(get):
    """Fill the weight lists from `get(name) -> tensor` (checkpoint names after `language_model.`)."""
    global EMBED, W_FINAL, INV_FREQ
    f32 = lambda name: get(name).to(device, ACC)
    mat = lambda *names: torch.cat([get(n).to(device, DT) for n in names], 0)
    for i in range(cfg.layers):
        p = f"layers.{i}."
        W_IN.append(1 + f32(p + "input_layernorm.weight"))
        W_POST.append(1 + f32(p + "post_attention_layernorm.weight"))
        if KIND[i] == "linear":
            m = p + "linear_attn."
            W_MIX_IN.append(mat(m + "in_proj_qkv.weight", m + "in_proj_z.weight", m + "in_proj_b.weight", m + "in_proj_a.weight"))
            W_MIX_OUT.append(mat(m + "out_proj.weight"))
            CONV_W.append(f32(m + "conv1d.weight").reshape(-1, cfg.conv).T.contiguous())   # (4, C): tap j reads x[t - 3 + j]
            A_NEG.append(-f32(m + "A_log").exp())
            DT_BIAS.append(f32(m + "dt_bias"))
            W_GNORM.append(f32(m + "norm.weight"))
            W_QN.append(None); W_KN.append(None)
        else:
            m = p + "self_attn."
            W_MIX_IN.append(mat(m + "q_proj.weight", m + "k_proj.weight", m + "v_proj.weight"))
            W_MIX_OUT.append(mat(m + "o_proj.weight"))
            W_QN.append(1 + f32(m + "q_norm.weight")); W_KN.append(1 + f32(m + "k_norm.weight"))
            CONV_W.append(None); A_NEG.append(None); DT_BIAS.append(None); W_GNORM.append(None)
        W_MLP_IN.append(mat(p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"))
        W_MLP_OUT.append(mat(p + "mlp.down_proj.weight"))
    EMBED = get("embed_tokens.weight").to(device, DT)
    W_FINAL = 1 + f32("norm.weight")
    INV_FREQ = 1.0 / cfg.theta ** (torch.arange(0, cfg.rot, 2, device=device).to(ACC) / cfg.rot)


def init_trained(special_base, special_init):
    """LoRA at rank RANK on every projection (A Kaiming-uniform, B zero, so training starts from the base
    model), the readout vector at zero (every option starts equal), and the special tokens' rows from
    `special_init` (n_special, D)."""
    global READOUT, SPECIAL_ROWS, SPECIAL_BASE
    LORA.clear(); ADAM.clear()
    pdt = torch.float64 if DT == torch.float64 else torch.float32
    for i in range(cfg.layers):
        shapes = {"mix_in": W_MIX_IN[i].shape, "mix_out": W_MIX_OUT[i].shape,
                  "mlp_in": W_MLP_IN[i].shape, "mlp_out": W_MLP_OUT[i].shape}
        layer = {}
        for name, (out_dim, in_dim) in shapes.items():
            A = torch.empty(RANK, in_dim, device=device, dtype=pdt)
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            layer[name] = [A, torch.zeros(out_dim, RANK, device=device, dtype=pdt)]
        LORA.append(layer)
    READOUT = torch.zeros(cfg.d, device=device, dtype=pdt)
    SPECIAL_ROWS = special_init.to(device, pdt).clone()
    SPECIAL_BASE = special_base
    for p in all_params():
        ADAM[id(p)] = (torch.zeros_like(p), torch.zeros_like(p))


def all_params():
    return [p for layer in LORA for name in LORA_NAMES for p in layer[name]] + [READOUT, SPECIAL_ROWS]


def load_qwen():
    """Qwen3.5-9B's text weights onto the GPU (the vision tower, the MTP layer and the LM head stay on disk),
    and the FlashAttention-2 kernels."""
    global FA2
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    root = Path(snapshot_download(MODEL_REPO, revision=MODEL_REV, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"]))
    config = json.loads((root / "config.json").read_text())
    set_cfg(config.get("text_config", config))
    tensors = {}
    for file in sorted(root.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "language_model." in name and ".mtp" not in name:
                    tensors[name.split("language_model.", 1)[1]] = f.get_tensor(name)
    load_weights(tensors.pop)
    assert not tensors, f"unused weights: {sorted(tensors)[:5]}"
    torch.cuda.empty_cache()
    from kernels import get_kernel
    k = get_kernel("kernels-community/flash-attn2", version=1)
    FA2 = k if hasattr(k, "flash_attn_varlen_func") else k.flash_attn_interface
    return root


# ------------------------------------------------------------------------------
# §§ Kernels
# ------------------------------------------------------------------------------
# The two segment-aware ops: causal softmax attention within each row (FlashAttention-2 varlen) and the Gated
# DeltaNet recurrence within each row (FLA's chunk kernel). The backward uses each library's own backward:
# `kernel` runs the op on detached inputs under a local autograd graph and keeps that graph, which saves
# exactly what the op's backward reads (q, k, v, out and the LSE; FLA's chunk inputs); `kernel_bwd` runs it.
# Everything around them is differentiated by hand. On the checks' CPU-free fp64 path they are per-row
# references in plain torch.

def kernel(fn, inputs, keep):
    if not keep:
        return fn(*inputs), None
    leaves = [x.detach().requires_grad_(True) for x in inputs]
    with torch.enable_grad():
        out = fn(*leaves)
    return out.detach(), (out, leaves)


def kernel_bwd(saved, dout):
    out, leaves = saved
    return torch.autograd.grad(out, leaves, dout)


def attn_fn(pk):
    """Causal softmax attention within each row: q (T, Hq, Dh), k and v (T, Hkv, Dh), grouped by Hq / Hkv."""
    scale = cfg.dh ** -0.5
    if DT != torch.float64:
        def fa(q, k, v):
            out = FA2.flash_attn_varlen_func(q, k, v, pk.cu, pk.cu, pk.longest, pk.longest, softmax_scale=scale, causal=True)
            return out[0] if isinstance(out, (tuple, list)) else out
        return fa

    def reference(q, k, v):
        rep = q.shape[1] // k.shape[1]
        outs = []
        for s, e in zip(pk.cu_cpu[:-1].tolist(), pk.cu_cpu[1:].tolist()):
            qq = q[s:e].transpose(0, 1)[None]
            kk, vv = (x[s:e].transpose(0, 1).repeat_interleave(rep, 0)[None] for x in (k, v))
            outs.append(F.scaled_dot_product_attention(qq, kk, vv, scale=scale, is_causal=True)[0].transpose(0, 1))
        return torch.cat(outs, 0)
    return reference


def gdn_fn(pk):
    """Gated DeltaNet within each row, the state reset at each boundary: q, k (T, Hk, Dk), v (T, Hv, Dv),
    g fp32 and beta (T, Hv); q and k are L2-normalized inside, and each Hv / Hk value heads share a q / k head."""
    if DT != torch.float64:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        def fla(q, k, v, g, beta):
            out, _ = chunk_gated_delta_rule(q[None], k[None], v[None], g[None], beta[None], cu_seqlens=pk.cu_long,
                                            cu_seqlens_cpu=pk.cu_cpu_long, use_qk_l2norm_in_kernel=True)
            return out[0]
        return fla

    def reference(q, k, v, g, beta):
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        rep = v.shape[1] // q.shape[1]
        outs = []
        for s, e in zip(pk.cu_cpu[:-1].tolist(), pk.cu_cpu[1:].tolist()):
            outs.append(torch_chunk_gated_delta_rule(q[None, s:e].repeat_interleave(rep, 2), k[None, s:e].repeat_interleave(rep, 2),
                                                     v[None, s:e], g[None, s:e], beta[None, s:e], use_qk_l2norm_in_kernel=True)[0][0])
        return torch.cat(outs, 0)
    return reference


# ------------------------------------------------------------------------------
# §§ Pieces and their backwards
# ------------------------------------------------------------------------------

def rmsnorm(x, w1):
    """RMSNorm in fp32 with the (1 + w) weight, back to x's dtype -> (y, rstd (.., 1) fp32)."""
    xf = x.to(ACC)
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.eps)
    return (xf * rstd * w1).to(x.dtype), rstd


def rmsnorm_bwd(dy, x, rstd, w1):
    """dx (fp32) for y = x * rstd * w1, w1 frozen (None for no weight)."""
    xhat = x.to(ACC) * rstd
    g = dy.to(ACC) * (w1 if w1 is not None else 1.0)
    return rstd * (g - xhat * (g * xhat).mean(-1, keepdim=True))


def silu_grad(x):
    s = torch.sigmoid(x.to(ACC))
    return s * (1 + x.to(ACC) * (1 - s))


def rope(x, cos, sin):
    """Rotary position embedding on the first cfg.rot dims of each head (the rotate-half form)."""
    r, rest = x[..., :cfg.rot], x[..., cfg.rot:]
    half = cfg.rot // 2
    return torch.cat((r * cos + torch.cat((-r[..., half:], r[..., :half]), -1) * sin, rest), -1)


def rope_bwd(dy, cos, sin):
    """The transpose of the rotation: the same rotation with -sin."""
    return rope(dy, cos, -sin)


def lora_mm(x, i, name, W):
    """x (T, in) -> x W^T + (ALPHA/RANK) (x A^T) B^T, and u = x A^T for the backward."""
    A, B = LORA[i][name]
    u = x @ A.to(x.dtype).T
    return torch.addmm(x @ W.T, u, B.to(x.dtype).T, alpha=ALPHA / RANK), u


def lora_mm_bwd(dy, x, u, i, name, W):
    """-> dx, and the fp32 gradients of A and B."""
    A, B = LORA[i][name]
    dyB = dy @ B.to(dy.dtype)                                                       # (T, r)
    dx = torch.addmm(dy @ W, dyB, A.to(dy.dtype), alpha=ALPHA / RANK)
    return dx, (dyB.T @ x).to(A.dtype) * (ALPHA / RANK), (dy.T @ u).to(A.dtype) * (ALPHA / RANK)


# ------------------------------------------------------------------------------
# §§ Forward
# ------------------------------------------------------------------------------
# One pack -> the final-norm hidden states at the readout rows. With keep=True each layer leaves what its
# backward reads in `ctx`; otherwise nothing is kept (evaluation, and the checks' autograd reference).

def forward(pk, keep):
    T = pk.tokens.shape[0]
    special = pk.tokens >= SPECIAL_BASE
    x = torch.where(special[:, None], SPECIAL_ROWS.to(DT)[(pk.tokens - SPECIAL_BASE).clamp(min=0)], EMBED[pk.tokens.clamp(max=SPECIAL_BASE - 1)])
    freqs = pk.pos.to(ACC)[:, None] * INV_FREQ[None]
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos().to(DT)[:, None], emb.sin().to(DT)[:, None]                # (T, 1, rot)
    # Convolution tap j reads x[t - 3 + j]; it stays inside the row iff pos >= 3 - j (zero padding before it).
    tap_keep = (pk.pos[:, None] >= torch.arange(3, -1, -1, device=pk.pos.device)).to(DT)   # (T, 4)
    attn, gdn = attn_fn(pk), gdn_fn(pk)
    ctx = []
    for i in range(cfg.layers):
        c = dict(x=x) if keep else None
        h, rstd = rmsnorm(x, W_IN[i])
        proj, u_in = lora_mm(h, i, "mix_in", W_MIX_IN[i])
        if KIND[i] == "linear":
            kd, vd = cfg.hk * cfg.dk, cfg.hv * cfg.dv
            qkv, z, b, a = proj.split([2 * kd + vd, vd, cfg.hv, cfg.hv], -1)
            conv = torch.zeros(qkv.shape, dtype=ACC, device=x.device)
            xp = F.pad(qkv, (0, 0, 3, 0))
            for j in range(4):
                conv = conv + (xp[j:j + T] * tap_keep[:, j:j + 1]).to(ACC) * CONV_W[i][j]
            conv = conv.to(DT)
            q, k, v = F.silu(conv).split([kd, kd, vd], -1)
            q, k = q.unflatten(-1, (cfg.hk, cfg.dk)).contiguous(), k.unflatten(-1, (cfg.hk, cfg.dk)).contiguous()
            v = v.unflatten(-1, (cfg.hv, cfg.dv)).contiguous()
            beta = torch.sigmoid(b)
            g = A_NEG[i] * F.softplus(a.to(ACC) + DT_BIAS[i])
            core, k_saved = kernel(gdn, (q, k, v, g, beta), keep)                     # (T, Hv, Dv)
            cf = core.to(ACC)
            core_rstd = torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + cfg.eps)
            zf = z.to(ACC).unflatten(-1, (cfg.hv, cfg.dv))
            mixed = (cf * core_rstd * W_GNORM[i] * F.silu(zf)).to(DT).flatten(1)
            if keep:
                # z and a copied out of the fused projection, so the projection itself can be freed
                c.update(conv=conv, z=z.contiguous(), a=a.contiguous(), beta=beta, core=core, core_rstd=core_rstd, kernel=k_saved)
        else:
            qg, k, v = proj.split([2 * cfg.hq * cfg.dh, cfg.hkv * cfg.dh, cfg.hkv * cfg.dh], -1)
            qg = qg.unflatten(-1, (cfg.hq, 2 * cfg.dh))
            q_raw, gate = qg[..., :cfg.dh], qg[..., cfg.dh:]
            k_raw = k.unflatten(-1, (cfg.hkv, cfg.dh))
            qn, q_rstd = rmsnorm(q_raw, W_QN[i])
            kn, k_rstd = rmsnorm(k_raw, W_KN[i])
            q, k = rope(qn, cos, sin), rope(kn, cos, sin)
            att, k_saved = kernel(attn, (q.contiguous(), k.contiguous(), v.unflatten(-1, (cfg.hkv, cfg.dh)).contiguous()), keep)
            mixed = att.flatten(1) * torch.sigmoid(gate.flatten(1))
            if keep:
                c.update(q_raw=q_raw, k_raw=k_raw, q_rstd=q_rstd, k_rstd=k_rstd, gate=gate, att=att, kernel=k_saved)
        out, u_out = lora_mm(mixed, i, "mix_out", W_MIX_OUT[i])
        x = x + out
        h2, rstd2 = rmsnorm(x, W_POST[i])
        gu, u_mlp_in = lora_mm(h2, i, "mlp_in", W_MLP_IN[i])
        act = F.silu(gu[:, :cfg.f]) * gu[:, cfg.f:]
        down, u_mlp_out = lora_mm(act, i, "mlp_out", W_MLP_OUT[i])
        if keep:
            c.update(rstd=rstd, u_in=u_in, mixed=mixed, u_out=u_out, x_mid=x, rstd2=rstd2, gu=gu,
                     u_mlp_in=u_mlp_in, u_mlp_out=u_mlp_out)
            ctx.append(c)
        x = x + down
    hf, rstd_f = rmsnorm(x[pk.readout], W_FINAL)
    if keep:
        ctx.append(dict(x_rows=x[pk.readout], rstd_f=rstd_f, hf=hf, special=special))
    return hf, ctx


def readout_loss(hf, pk):
    """Cross-entropy of each question's options against its target, averaged over the pack's questions."""
    logits = hf.to(READOUT.dtype) @ READOUT                                         # (R,)
    padded = logits[pk.opt_idx.clamp(min=0)].masked_fill(pk.opt_idx < 0, float("-inf"))   # (Q, K)
    logp = torch.log_softmax(padded, -1)
    loss = -(pk.target * logp.masked_fill(pk.opt_idx < 0, 0)).sum(-1).mean()
    return loss, logp.exp()


# ------------------------------------------------------------------------------
# §§ Backward
# ------------------------------------------------------------------------------
# The forward in reverse, a layer at a time. Each layer's LoRA gradients go straight into AdamW (`step_in`),
# and the layer's saved activations are freed as soon as its backward has read them.

def backward(pk, ctx, probs, step_in):
    last = ctx.pop()
    Q = probs.shape[0]
    dpad = (probs - pk.target).masked_fill(pk.opt_idx < 0, 0) / Q                   # dloss / dlogits, (Q, K)
    valid = pk.opt_idx >= 0
    dlogits = torch.zeros(pk.readout.shape[0], device=probs.device, dtype=READOUT.dtype)
    dlogits.index_add_(0, pk.opt_idx[valid], dpad[valid].to(READOUT.dtype))
    d_readout = dlogits @ last["hf"].to(READOUT.dtype)
    dhf = dlogits[:, None] * READOUT[None]
    step_in([READOUT], [d_readout])
    dx = torch.zeros(pk.tokens.shape[0], cfg.d, device=probs.device, dtype=ACC)
    dx[pk.readout] = rmsnorm_bwd(dhf, last["x_rows"], last["rstd_f"], W_FINAL)
    for i in reversed(range(cfg.layers)):
        c = ctx.pop()
        grads = {}
        # MLP: x_out = x_mid + down(silu(gate) * up)
        h2 = rmsnorm(c["x_mid"], W_POST[i])[0]
        g_, up = c["gu"][:, :cfg.f], c["gu"][:, cfg.f:]
        act = F.silu(g_) * up
        dact, *grads["mlp_out"] = lora_mm_bwd(dx.to(DT), act, c["u_mlp_out"], i, "mlp_out", W_MLP_OUT[i])
        dgu = torch.cat(((dact.to(ACC) * up.to(ACC) * silu_grad(g_)).to(DT), (dact.to(ACC) * F.silu(g_.to(ACC))).to(DT)), -1)
        dh2, *grads["mlp_in"] = lora_mm_bwd(dgu, h2, c["u_mlp_in"], i, "mlp_in", W_MLP_IN[i])
        dx = dx + rmsnorm_bwd(dh2, c["x_mid"], c["rstd2"], W_POST[i])
        # Mixer: x_mid = x + out(mixed)
        dmixed, *grads["mix_out"] = lora_mm_bwd(dx.to(DT), c["mixed"], c["u_out"], i, "mix_out", W_MIX_OUT[i])
        if KIND[i] == "linear":
            zf = c["z"].to(ACC).unflatten(-1, (cfg.hv, cfg.dv))
            cf = c["core"].to(ACC)
            n = cf * c["core_rstd"]
            dm = dmixed.to(ACC).unflatten(-1, (cfg.hv, cfg.dv))
            dz = dm * n * W_GNORM[i] * silu_grad(zf)
            dcore = rmsnorm_bwd(dm * W_GNORM[i] * F.silu(zf), cf, c["core_rstd"], None)
            dq, dk, dv, dg, dbeta = kernel_bwd(c["kernel"], dcore.to(DT))
            beta = c["beta"].to(ACC)
            db = dbeta.to(ACC) * beta * (1 - beta)
            da = dg.to(ACC) * A_NEG[i] * torch.sigmoid(c["a"].to(ACC) + DT_BIAS[i])
            dconv = torch.cat((dq.flatten(1), dk.flatten(1), dv.flatten(1)), -1).to(ACC) * silu_grad(c["conv"])
            # conv[t] = sum_j qkv[t - 3 + j] * keep[t, j] * w[j]  =>  dqkv[t - 3 + j] += dconv[t] * keep[t, j] * w[j]
            T = dconv.shape[0]
            tap_keep = (pk.pos[:, None] >= torch.arange(3, -1, -1, device=pk.pos.device)).to(ACC)
            dxp = torch.zeros(T + 3, dconv.shape[1], device=dconv.device, dtype=ACC)
            for j in range(4):
                dxp[j:j + T] += dconv * tap_keep[:, j:j + 1] * CONV_W[i][j]
            dproj = torch.cat((dxp[3:], dz.flatten(1), db, da), -1).to(DT)
        else:
            sig = torch.sigmoid(c["gate"].flatten(1).to(ACC))
            datt = (dmixed.to(ACC) * sig).to(DT).unflatten(-1, (cfg.hq, cfg.dh))
            dgate = (dmixed.to(ACC) * c["att"].flatten(1).to(ACC) * sig * (1 - sig)).unflatten(-1, (cfg.hq, cfg.dh))
            dq, dk, dv = kernel_bwd(c["kernel"], datt)
            cos, sin = pk.cos, pk.sin
            dq_raw = rmsnorm_bwd(rope_bwd(dq, cos, sin), c["q_raw"], c["q_rstd"], W_QN[i])
            dk_raw = rmsnorm_bwd(rope_bwd(dk, cos, sin), c["k_raw"], c["k_rstd"], W_KN[i])
            dqg = torch.cat((dq_raw, dgate), -1).flatten(1)
            dproj = torch.cat((dqg, dk_raw.flatten(1), dv.flatten(1).to(ACC)), -1).to(DT)
        h = rmsnorm(c["x"], W_IN[i])[0]
        dh, *grads["mix_in"] = lora_mm_bwd(dproj, h, c["u_in"], i, "mix_in", W_MIX_IN[i])
        dx = dx + rmsnorm_bwd(dh, c["x"], c["rstd"], W_IN[i])
        del c
        step_in([p for name in LORA_NAMES for p in LORA[i][name]], [gr for name in LORA_NAMES for gr in grads[name]])
    special = last["special"]
    d_special = torch.zeros_like(SPECIAL_ROWS)
    d_special.index_add_(0, pk.tokens[special] - SPECIAL_BASE, dx[special])
    step_in([SPECIAL_ROWS], [d_special])


# ==============================================================================
# § Optimizer
# ==============================================================================
# AdamW on the fp32 trained parameters, stepped in place from inside the backward. Warmup over the first 5% of
# steps, then cosine to 10% of LR. No weight decay (LoRA's B starts at zero and its A carries no scale).

def adamw_step(params, grads, step, num_steps):
    warm = max(1, int(0.05 * num_steps))
    progress = (step - warm) / max(1, num_steps - warm)
    lr = LR * (step + 1) / warm if step < warm else LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = step + 1
    for p, g in zip(params, grads):
        m, v = ADAM[id(p)]
        m.lerp_(g, 1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        p.addcdiv_(m, (v / (1 - b2 ** t)).sqrt_().add_(eps), value=-lr / (1 - b1 ** t))
    return lr


# ==============================================================================
# § Data
# ==============================================================================
# Rows from the data prep's records: one per question, the state (its first STATE_CAP ids) then the question's
# prefix and options. Training shuffles a choice question's options every time its row is built.

def load_split(split):
    t = pq.read_table(DATA_DIR / f"{split}.parquet").to_pydict()
    rows = []
    for r in range(len(t["id"])):
        state = t["state_ids"][r][:STATE_CAP]
        for j in range(len(t["q_name"][r])):
            rows.append(SimpleNamespace(split=split, record=r, source=t["source"][r], clean=t["variant"][r] == "clean",
                                        state=state, kind=t["q_type"][r][j], prefix=t["q_prefix"][r][j],
                                        options=t["q_options"][r][j], label=t["q_label"][r][j],
                                        target=t["q_target"][r][j] or None,
                                        length=len(state) + len(t["q_prefix"][r][j]) + sum(len(o) for o in t["q_options"][r][j])))
    return rows


def row_target(row, order):
    """The training target in option order `order`: SOFT_WEIGHT x the soft target + the rest on the label."""
    hard = np.zeros(len(order), np.float32)
    hard[order.index(row.label)] = 1
    if row.target is None:
        return hard
    soft = np.asarray([row.target[o] for o in order], np.float32)
    return SOFT_WEIGHT * soft + (1 - SOFT_WEIGHT) * hard


def make_pack(rows, train):
    """Rows end to end -> the forward's inputs: tokens, positions within each row, cu_seqlens, the readout rows,
    and per question its option rows (padded with -1) and target."""
    tokens, pos, readout, opt_rows, targets, labels, orders = [], [], [], [], [], [], []
    n = 0
    for row in rows:
        order = list(range(len(row.options)))
        if train and row.kind == "choice":
            random.shuffle(order)
        ids = row.state + row.prefix
        mine = []
        for o in order:
            ids = ids + row.options[o]
            mine.append(n + len(ids) - 1)
        tokens.append(ids); pos.append(np.arange(len(ids)))
        q_rows = list(range(len(readout), len(readout) + len(mine)))
        readout += mine; opt_rows.append(q_rows); orders.append(order)
        targets.append(row_target(row, order) if train else None)
        n += len(ids)
    K = max(len(o) for o in opt_rows)
    opt_idx = np.full((len(rows), K), -1, np.int64)
    target = np.zeros((len(rows), K), np.float32)
    for q, rr in enumerate(opt_rows):
        opt_idx[q, :len(rr)] = rr
        if targets[q] is not None:
            target[q, :len(rr)] = targets[q]
    lengths = [len(t) for t in tokens]
    cu_cpu = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.int32)
    pk = SimpleNamespace(tokens=torch.from_numpy(np.concatenate(tokens)).to(device),
                         pos=torch.from_numpy(np.concatenate(pos).astype(np.int32)).to(device),
                         cu=cu_cpu.to(device), cu_cpu=cu_cpu, cu_long=cu_cpu.long().to(device), cu_cpu_long=cu_cpu.long(),
                         longest=max(lengths),
                         readout=torch.tensor(readout, device=device), opt_idx=torch.from_numpy(opt_idx).to(device),
                         target=torch.from_numpy(target).to(device), rows=rows, orders=orders, n_tokens=n)
    freqs = pk.pos.to(ACC)[:, None] * INV_FREQ[None]
    emb = torch.cat((freqs, freqs), -1)
    pk.cos, pk.sin = emb.cos().to(DT)[:, None], emb.sin().to(DT)[:, None]
    return pk


def plan_packs(rows, budget, shuffle):
    """Rows into packs of at most `budget` tokens: in random order, filled greedily (training), or first-fit
    decreasing (evaluation)."""
    assert max(r.length for r in rows) <= budget, f"a {max(r.length for r in rows)}-token row is over the {budget}-token budget"
    if shuffle:
        order = list(range(len(rows))); random.shuffle(order)
        packs, cur, used = [], [], 0
        for i in order:
            if used + rows[i].length > budget:
                packs.append(cur); cur, used = [], 0
            cur.append(rows[i]); used += rows[i].length
        return packs + [cur]
    packs, room = [], []
    for i in sorted(range(len(rows)), key=lambda i: -rows[i].length):
        for p in range(len(packs)):
            if rows[i].length <= room[p]:
                packs[p].append(rows[i]); room[p] -= rows[i].length
                break
        else:
            packs.append([rows[i]]); room.append(budget - rows[i].length)
    return packs


# ==============================================================================
# § Evaluation
# ==============================================================================
# Accuracy, Brier score and log loss against the label, and KL from d3-flash's distribution where the data prep
# joined one, on each split's clean rows; Cookbook-dev also by source.

@torch.no_grad()
def evaluate(splits, rows_by_split, final):
    results = {}
    for split in splits:
        rows = rows_by_split[split]
        stats = []
        for pack_rows in plan_packs(rows, EVAL_TOKENS, shuffle=False):
            pk = make_pack(pack_rows, train=False)
            hf, _ = forward(pk, keep=False)
            _, probs = readout_loss(hf, pk)
            probs = probs.cpu().numpy()
            for q, row in enumerate(pack_rows):
                p = probs[q, :len(row.options)]
                onehot = np.eye(len(p))[row.label]
                kl = None
                if row.target is not None:
                    t = np.asarray(row.target)
                    kl = float((t * (np.log(np.maximum(t, 1e-12)) - np.log(np.maximum(p, 1e-12)))).sum())
                stats.append(dict(clean=row.clean, source=row.source, acc=float(p.argmax() == row.label),
                                  brier=float(((p - onehot) ** 2).sum()), logloss=float(-np.log(max(p[row.label], 1e-12))),
                                  kl=kl, p=p.round(4).tolist(), label=row.label, record=row.record))
        clean = [s for s in stats if s["clean"]]
        kls = [s["kl"] for s in clean if s["kl"] is not None]
        res = dict(n=len(clean), acc=np.mean([s["acc"] for s in clean]), brier=np.mean([s["brier"] for s in clean]),
                   logloss=np.mean([s["logloss"] for s in clean]), kl_d3=np.mean(kls) if kls else None)
        if final:
            by_source = {}
            for s in clean:
                by_source.setdefault(s["source"], []).append(s["acc"])
            res["by_source"] = {k: (int(sum(v)), len(v)) for k, v in sorted(by_source.items())}
            res["questions"] = stats
        results[split] = res
        kl_text = f" kl_d3 {res['kl_d3']:.3f}" if res["kl_d3"] is not None else ""
        print(f"  val {split}: n {res['n']} acc {res['acc']:.4f} brier {res['brier']:.4f} logloss {res['logloss']:.4f}{kl_text}", flush=True)
        if final and split == "cookbook_dev":
            print("    by source: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res["by_source"].items()))
    return results


# ==============================================================================
# § Checks
# ==============================================================================
# CHECK=1: a tiny random Qwen3.5 in fp64 on the GPU, with the per-row reference kernels. (1) With LoRA's B at zero
# the forward's final hidden states match transformers' Qwen3_5TextModel. (2) With B random, the hand-written
# backward's gradients match autograd's through the same forward, for every trained parameter.

def run_checks():
    global DT, ACC, SPECIAL_BASE
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    DT, ACC = torch.float64, torch.float64
    tc = Qwen3_5TextConfig(   # agent-ops-stacks encoder-rtx/2026-11-01_1135am_d3-varlen-engine/check_tiny.py's
        hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, attn_output_gate=True,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, vocab_size=512, rms_norm_eps=1e-6, max_position_embeddings=4096,
        rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25,
                             mrope_section=[11, 11, 10], mrope_interleaved=True))
    tc._attn_implementation = "sdpa"
    torch.manual_seed(0)
    ref = Qwen3_5TextModel(tc).eval()
    with torch.no_grad():
        for p in ref.parameters():        # every weight random, the zero-centred norms included
            p.add_(torch.randn_like(p) * 0.3)
    ref = ref.to(device, torch.float64)
    set_cfg(tc)
    sd = ref.state_dict()
    load_weights(lambda name: sd[name])
    n_special = 18
    init_trained(tc.vocab_size - n_special, torch.randn(n_special, cfg.d, dtype=torch.float64) * 0.1)
    READOUT_ = torch.randn(cfg.d, device=device, dtype=torch.float64)

    rng = np.random.default_rng(0)
    rows = []
    for length in (23, 9, 40):
        ids = rng.integers(0, tc.vocab_size - n_special, size=length).tolist()
        ids[3] = tc.vocab_size - n_special + 2                                    # a special token mid-row
        rows.append(SimpleNamespace(state=ids[:length - 6], prefix=ids[length - 6:length - 4],
                                    options=[ids[length - 4:length - 2], ids[length - 2:]], kind="choice", label=1,
                                    target=None, length=length))

    # (1) the forward against transformers, every row on its own
    pk = make_pack(rows, train=False)
    with torch.no_grad():
        SPECIAL_ROWS.copy_(ref.embed_tokens.weight[SPECIAL_BASE:SPECIAL_BASE + n_special])
        T = pk.tokens.shape[0]
        pk.readout = torch.arange(T, device=device)
        hf, _ = forward(pk, keep=False)
        worst = 0.0
        for s, e in zip(pk.cu_cpu[:-1].tolist(), pk.cu_cpu[1:].tolist()):
            out = ref(input_ids=pk.tokens[None, s:e]).last_hidden_state[0]
            worst = max(worst, (out - hf[s:e]).abs().max().item() / out.abs().max().item())
    # transformers computes the delta rule, the gated norm, the log decay and every RMSNorm in fp32 whatever the
    # model dtype, so fp32 rounding is the floor here (check_tiny.py widens those to reach fp64 agreement)
    print(f"check forward vs transformers: max relative error {worst:.2e}")
    assert worst < 1e-4

    # (2) the hand-written gradients against autograd's
    for layer in LORA:
        for name in LORA_NAMES:
            layer[name][1].normal_(0, 0.05)
    global READOUT
    READOUT = READOUT_
    pk = make_pack(rows, train=True)
    params = all_params()
    for p in params:
        p.requires_grad_(True)
    with torch.enable_grad():
        hf, _ = forward(pk, keep=False)
        loss, _ = readout_loss(hf, pk)
        ref_grads = torch.autograd.grad(loss, params)
    for p in params:
        p.requires_grad_(False)
    mine = {}
    hf, ctx = forward(pk, keep=True)
    _, probs = readout_loss(hf, pk)
    backward(pk, ctx, probs, lambda ps, gs: mine.update({id(p): g for p, g in zip(ps, gs)}))
    worst = 0.0
    for p, g in zip(params, ref_grads):
        err = (mine[id(p)] - g).abs().max().item() / max(g.abs().max().item(), 1e-30)
        worst = max(worst, err)
    print(f"check hand-written gradients vs autograd: {len(params)} tensors, max relative error {worst:.2e}")
    assert worst < 1e-8


# ==============================================================================
# § Train
# ==============================================================================

def special_init_rows(root):
    """Each special token's row starts as the mean of the embeddings of the word that names it."""
    from tokenizers import Tokenizer
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    words = {"<|state|>": "state", "<|noul|>": "yes or no", "<|score|>": "score", "<|choice|>": "choice", '<|"|>': '"',
             "<|yes|>": "yes", "<|no|>": "no", "<|option|>": "option", **{f"<|level_{i}|>": str(i) for i in range(10)}}
    names = sorted(meta["special_tokens"], key=meta["special_tokens"].get)
    base = meta["special_tokens"][names[0]]
    assert [meta["special_tokens"][n] for n in names] == list(range(base, base + len(names)))
    rows = torch.stack([EMBED[tok.encode(" " + words[n] if words[n][0].isalnum() else words[n], add_special_tokens=False).ids].float().mean(0)
                        for n in names])
    return base, rows


def main():
    if CHECK:
        run_checks()
        return
    t0 = time.time()
    root = load_qwen()
    print(f"loaded {MODEL_REPO} in {time.time() - t0:.0f}s; {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    special_base, special_rows = special_init_rows(root)
    init_trained(special_base, special_rows)
    n_lora = sum(p.numel() for layer in LORA for name in LORA_NAMES for p in layer[name])
    print(f"LoRA rank {RANK}: {n_lora / 1e6:.1f}M parameters", flush=True)

    train_rows = [r for s in TRAIN_SPLITS for r in load_split(s)]
    val_rows = {s: load_split(s) for s in VAL_SPLITS}
    n_tokens = sum(r.length for r in train_rows)
    epochs = [plan_packs(train_rows, TOKENS, shuffle=True) for _ in range(math.ceil(EPOCHS))]
    packs = [p for ep in epochs for p in ep][:int(round(EPOCHS * len(epochs[0])))]
    num_steps = min(len(packs), ABORT_STEP) if ABORT_STEP else len(packs)
    print(f"train: {len(train_rows):,} questions, {n_tokens / 1e6:.1f}M tokens a pass, {len(epochs[0])} packs a pass; "
          f"{num_steps} steps", flush=True)

    if USE_WANDB:
        import wandb
        wandb.init(project="qwen_decision_rtx_dev", name=RUN_NAME,
                   config={k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))})
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    step_time, seen = 0.0, 0
    for step in range(num_steps):
        torch.cuda.synchronize(); ts = time.time()
        pk = make_pack(packs[step], train=True)
        lr = [None]
        with torch.no_grad():
            hf, ctx = forward(pk, keep=True)
            loss, probs = readout_loss(hf, pk)
            # ← the step: the backward, with AdamW stepping each layer's weights as their gradients arrive
            backward(pk, ctx, probs, lambda ps, gs: lr.__setitem__(0, adamw_step(ps, gs, step, num_steps)))
        torch.cuda.synchronize()
        dt = time.time() - ts
        if step >= 3:
            step_time += dt; seen += 1
        msg = (f"step {step + 1}/{num_steps} loss {loss.item():.4f} rows {len(pk.rows)} tokens {pk.n_tokens} "
               f"{dt:.2f}s ({pk.n_tokens / dt:,.0f} tok/s) lr {lr[0]:.2e} peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        print(msg, flush=True)
        if USE_WANDB:
            wandb.log(dict(loss=loss.item(), lr=lr[0], step_s=dt, tok_s=pk.n_tokens / dt), step=step + 1)
        if VAL_EVERY and (step + 1) % VAL_EVERY == 0 and step + 1 < num_steps:
            res = evaluate(VAL_SPLITS, val_rows, final=False)
            if USE_WANDB:
                wandb.log({f"{s}/{k}": v for s, r in res.items() for k, v in r.items() if isinstance(v, float)}, step=step + 1)
    if seen:
        print(f"mean step {step_time / seen:.2f}s over steps 4-{num_steps}; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    t_eval = time.time()
    res = evaluate(VAL_SPLITS, val_rows, final=True)
    print(f"eval {time.time() - t_eval:.0f}s", flush=True)
    if USE_WANDB:
        wandb.log({f"{s}/{k}": v for s, r in res.items() for k, v in r.items() if isinstance(v, float)}, step=num_steps)
        wandb.finish()
    summary = {s: {k: v for k, v in r.items() if k != "questions"} for s, r in res.items()}
    (RUN_DIR / "result.json").write_text(json.dumps(dict(summary=summary, steps=num_steps, mean_step_s=step_time / max(seen, 1)), indent=1, default=float))
    (RUN_DIR / "questions.json").write_text(json.dumps({s: r["questions"] for s, r in res.items()}, default=float))
    if SAVE_CHECKPOINT:
        torch.save(dict(lora=LORA, readout=READOUT, special_rows=SPECIAL_ROWS, special_base=SPECIAL_BASE, rank=RANK, alpha=ALPHA),
                   RUN_DIR / "trained.pt")
    print(f"-> {RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
