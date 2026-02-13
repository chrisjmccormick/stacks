import os
import io
import contextlib
import copy
import faulthandler
import math
import re
import random
import signal
import tempfile
import time
import warnings
import multiprocessing
import pickle
import platform
import urllib.request
from contextlib import contextmanager
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from dataclasses import dataclass
from kernels import get_kernel

flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface

# Module-level globals, initialized by _init_distributed()
rank = 0
world_size = 1
device = None
DATASET_NAME = None

def _init_distributed():
    """Set module globals from the DDP environment."""
    global rank, world_size, device
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))

def print0(*args, **kwargs):
    """Print only on rank 0."""
    if rank == 0:
        print(*args, **kwargs)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinearT(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features) which
    addresses the slow kernel that was used for gradient accumulation. @chrisjmccormick
    """
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight) # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out = torch.ops.nanogpt.mm_t(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return x @ self.weight.type_as(x)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def rotary(self, x_BTHD, offset=0):
        """Apply rotary embeddings. offset is the starting position for KV cache inference."""
        T = x_BTHD.size(-3)
        assert self.factor1.size(0) >= offset + T, f"Rotary cache too small: {self.factor1.size(0)} < {offset + T}"
        factor1, factor2 = (
            self.factor1[None, offset:offset + T, None, :],
            self.factor2[None, offset:offset + T, None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)

        theta = torch.outer(t, angular_freq)
        self.factor1 = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.factor2 = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )

        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)

        theta = torch.outer(t, self.angular_freq)
        self.factor1.copy_(theta.cos())
        self.factor2.copy_(theta.sin())
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

@dataclass
class GPTConfig:
    """Config object expected by engine.py for inference."""
    n_head: int = 6
    n_kv_head: int = 6  # Same as n_head (no GQA in this model)
    n_embd: int = 768
    n_layer: int = 11
    sequence_len: int = 2048


class GPTInference(nn.Module):
    """
    Inference-only model that mirrors the training GPT architecture.
    All weights are stored in parameter banks -- no per-layer sub-modules.
    Built from a trained GPT via from_training_model().
    """

    def __init__(self, vocab_size, num_layers, num_heads, head_dim, model_dim, max_seq_len, bigram_vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = num_heads * head_dim

        # Config object expected by engine.py for inference
        self.config = GPTConfig(
            n_head=num_heads,
            n_kv_head=num_heads,  # No GQA
            n_embd=model_dim,
            n_layer=num_layers,
            sequence_len=max_seq_len,
        )

        self.smear_gate = nn.Linear(12, 1, bias=False)
        self.skip_gate = nn.Linear(12, 1, bias=False)

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.Parameter(torch.zeros(5 * vocab_size, model_dim, dtype=torch.bfloat16))

        # parameter banks for attention and value embedding gate weights
        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12))  # 10 layers
        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12))    # 5 unique gates

        # -----------------------------------
        # Parameter banks for sharded optimization, by @chrisjmccormick

        # Identify which layers have attention/MLP
        # Attention is skipped in layer 6 by @YouJiacheng
        attn_layer_indices = [i for i in range(num_layers) if i != 6]
        # All layers have MLP (At 11 layers--dropped first layer @EmelyanenkoK)
        mlp_layer_indices = list(range(num_layers))

        hdim = num_heads * head_dim
        mlp_hdim = 4 * model_dim

        # Create index mappings: layer_idx -> bank_idx
        self.layer_to_attn_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(attn_layer_indices)}
        self.layer_to_mlp_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(mlp_layer_indices)}

        # Attention bank: stores QKVO weights for all attention layers
        # merged QKVO weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # Simplified layout by @chrisjmccormick
        # Shape: (num_attn_layers, 4*model_dim, hdim) = (10, 3072, 768)
        self.attn_bank = nn.Parameter(torch.empty(len(attn_layer_indices), 4 * model_dim, hdim))

        # MLP bank: stores c_fc and c_proj for all MLP layers
        # Shape: (num_mlp_layers + padding, 2, mlp_hdim, model_dim) = (12, 2, 3072, 768)
        # We add 1 padding layer (index 11) to get 12*2=24 matrices for even distribution across 8 GPUs
        num_mlp_with_padding = len(mlp_layer_indices) + 1  # 11 + 1 = 12
        self.mlp_bank = nn.Parameter(torch.empty(num_mlp_with_padding, 2, mlp_hdim, model_dim))

        self.yarn = Yarn(head_dim, max_seq_len)

        # Transposed weight storage for lm_head (inference never uses FP8)
        self.lm_head = CastedLinearT(model_dim, vocab_size, use_fp8=False)

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.bigram_embed = nn.Embedding(bigram_vocab_size, model_dim)

        # x0_lambdas separated out for different optimizer treatment (no beta smoothing)
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))

        self.scalars = nn.Parameter(torch.zeros(4 * num_layers + 3))

    @classmethod
    def from_training_model(cls, training_model, dev):
        """Build an inference model by copying weights from a trained GPT."""
        tm = training_model
        cfg = tm.config
        m = cls(
            vocab_size=tm.vocab_size,
            num_layers=tm.num_layers,
            num_heads=cfg.n_head,
            head_dim=cfg.n_embd // cfg.n_head,
            model_dim=cfg.n_embd,
            max_seq_len=cfg.sequence_len,
            bigram_vocab_size=tm.bigram_embed.num_embeddings,
        )

        # Copy all weight tensors from the training model
        with torch.no_grad():
            m.smear_gate.weight.copy_(tm.smear_gate.weight)
            m.skip_gate.weight.copy_(tm.skip_gate.weight)
            m.value_embeds.copy_(tm.value_embeds)
            m.attn_gate_bank.copy_(tm.attn_gate_bank)
            m.ve_gate_bank.copy_(tm.ve_gate_bank)
            m.attn_bank.copy_(tm.attn_bank)
            m.mlp_bank.copy_(tm.mlp_bank)
            m.lm_head.weight.copy_(tm.lm_head.weight)
            m.embed.weight.copy_(tm.embed.weight)
            m.bigram_embed.weight.copy_(tm.bigram_embed.weight)
            m.x0_lambdas.copy_(tm.x0_lambdas)
            # scalars may have training padding; copy the meaningful portion
            n = m.scalars.numel()
            m.scalars.copy_(tm.scalars[:n])

        # Copy Yarn state (factor1, factor2, attn_scale)
        m.yarn.factor1 = nn.Buffer(tm.yarn.factor1.clone(), persistent=False)
        m.yarn.factor2 = nn.Buffer(tm.yarn.factor2.clone(), persistent=False)
        m.yarn.attn_scale = tm.yarn.attn_scale

        m = m.to(dev)
        m.eval()
        return m

    def get_device(self):
        """Return the device of the model (for engine.py compatibility)."""
        return self.embed.weight.device

    def forward(self, input_seq: Tensor, *, kv_cache=None):
        """
        Inference forward pass for text generation (compatible with engine.py).
        All attention and MLP logic is inlined -- no sub-module dispatch.

        Args:
            input_seq: Token indices of shape (B, T)
            kv_cache: KVCache object for incremental decoding (required)

        Returns:
            logits: Output logits of shape (B, T, vocab_size)
        """
        assert kv_cache is not None, "kv_cache is required for inference"
        idx = input_seq
        B, T = idx.size()

        # Skip connection and backout config (matches training)
        skip_in = [3]   # save hidden state at layer 3
        skip_out = [6]  # apply skip connection at layer 6
        backout_layer = 7

        # Get scalars for residual connections and gates
        resid_lambdas = self.scalars[:self.num_layers]
        x0_lambdas = self.x0_lambdas
        sa_lambdas = self.scalars[self.num_layers:3 * self.num_layers].view(-1, 2)
        bigram_lambdas = self.scalars[3 * self.num_layers:4 * self.num_layers]
        smear_lambda = self.scalars[4 * self.num_layers]
        backout_lambda = self.scalars[4 * self.num_layers + 1]
        skip_lambda = self.scalars[4 * self.num_layers + 2]

        # Compute bigram indices using XOR hash (must match get_bigram_hash exactly)
        rand_int_1 = 36313
        rand_int_2 = 27191
        mod = self.bigram_embed.num_embeddings - 1

        if T > 1:
            # Prefill: Compute for all adjacent pairs in the sequence
            curr = idx[:, 1:].to(torch.int32)
            prev = idx[:, :-1].to(torch.int32)

            # Initialize with the "reserved" token for the first position
            bigram_idx = torch.full_like(idx, mod, dtype=torch.int32)

            # Apply XOR hash to the rest
            bigram_idx[:, 1:] = torch.bitwise_xor(rand_int_1 * curr, rand_int_2 * prev) % mod

            # Cache the last token for subsequent decode steps
            kv_cache.prev_token = idx[:, -1:].clone()
        else:
            # Decode: Use cached previous token
            if kv_cache.prev_token is not None:
                curr = idx.to(torch.int32)
                prev = kv_cache.prev_token.to(torch.int32)

                # Apply XOR hash
                bigram_idx = torch.bitwise_xor(rand_int_1 * curr, rand_int_2 * prev) % mod
            else:
                # No previous token (start of generation if T=1 passed initially)
                bigram_idx = torch.full_like(idx, mod, dtype=torch.int32)

            # Update cache with current token
            kv_cache.prev_token = idx.clone()

        # Compute bigram embedding
        x0_bigram = self.bigram_embed(bigram_idx)  # (B, T, model_dim)

        # Embedding lookup
        x = self.embed(idx)

        # Smear gate: blend each token's embedding with previous token's embedding
        # During prefill (T > 1): apply smear gate like training
        # During decode (T == 1): use cached previous embedding from kv_cache
        if T > 1:
            # Prefill: apply smear gate across the sequence
            smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :12]))
            x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)
            # Cache the last embedding for subsequent decode steps
            kv_cache.prev_embed = x[:, -1:].clone()
        else:
            # Decode: use cached previous embedding
            if kv_cache.prev_embed is not None:
                smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[..., :12]))
                x = x + smear_gate_out * kv_cache.prev_embed
            # Update cache with current embedding
            kv_cache.prev_embed = x.clone()

        x = x0 = norm(x)

        # Value embeddings - single Parameter viewed as 5 embeddings, same pattern as training
        # Pattern: 01 ... 234 structure on token value embeddings by @photomz
        ve_all = self.value_embeds.view(5, self.vocab_size, -1)[:, idx]  # (5, B, T, D)
        ve = [ve_all[0], ve_all[1]] + [None] * (self.num_layers - 5) + [ve_all[2], ve_all[3], ve_all[4]]

        # VE gate bank - same pattern as training
        veg = self.ve_gate_bank.unbind(0)
        ve_gates = [veg[0], veg[1]] + [None] * (self.num_layers - 5) + [veg[2], veg[3], veg[4]]

        # Attention gate bank - same pattern as training (layer 6 has no attention)
        ag = self.attn_gate_bank.unbind(0)
        attn_gates = ag[:6] + [None] + ag[6:]

        # Use full context window for inference (no sliding window)
        window_size = self.config.sequence_len

        # Get weights from parameter banks
        attn_weights = self.attn_bank.unbind(0)
        mlp_fcs = self.mlp_bank[:, 0, :, :].unbind(0)
        mlp_projs = self.mlp_bank[:, 1, :, :].unbind(0)

        # Skip connection and backout buffers
        skip_buffer = None
        x_backout = None

        # Forward through transformer layers
        rotary_offset = kv_cache.get_pos()
        for i in range(self.num_layers):
            # Apply skip connection at layer 6 (before residual scaling)
            if i in skip_out and skip_buffer is not None:
                skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :12]))
                x = x + skip_gate_out * skip_buffer

            # Apply residual scaling with bigram embedding (matches training)
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x + bigram_lambdas[0] * x0_bigram
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram

            # --- Attention (inline) ---
            if i in self.layer_to_attn_idx:
                qkvo_w = attn_weights[self.layer_to_attn_idx[i]]
                h = norm(x)

                # Compute Q, K, V projections
                q, k, v = F.linear(h, sa_lambdas[i][0] * qkvo_w[:self.dim * 3].type_as(h)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
                q, k = norm(q), norm(k)  # QK norm @Grad62304977

                # Apply rotary embeddings with position offset from KV cache
                q = self.yarn.rotary(q, offset=rotary_offset)
                k = self.yarn.rotary(k, offset=rotary_offset)

                # Value embeddings
                if ve[i] is not None:
                    ve_gate_out = 2 * torch.sigmoid(F.linear(h[..., :12], ve_gates[i])).view(B, T, self.num_heads, 1)
                    v = v + ve_gate_out * ve[i].view_as(v)  # @KoszarskyB & @Grad62304977

                # Flash Attention with KV cache
                k_cache, v_cache = kv_cache.get_layer_cache(i)
                y = flash_attn_interface.flash_attn_with_kvcache(
                    q, k_cache, v_cache,
                    k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    softmax_scale=self.yarn.attn_scale,
                    window_size=(window_size, 0),
                )

                # Advance cache position after last layer
                if i == kv_cache.n_layers - 1:
                    kv_cache.advance(T)

                # Attention gate - sparse gated attention for context-based no-op
                y = y * torch.sigmoid(F.linear(h[..., :12], attn_gates[i])).view(B, T, self.num_heads, 1)

                # Output projection
                y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
                y = F.linear(y, sa_lambdas[i][1] * qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
                x = x + y

            # --- MLP (inline) ---
            # relu(x)^2: https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU
            # suggested by @SKYLINEZ007 and @Grad62304977
            if i in self.layer_to_mlp_idx:
                c_fc = mlp_fcs[self.layer_to_mlp_idx[i]]
                c_proj = mlp_projs[self.layer_to_mlp_idx[i]]
                x = x + F.linear(F.relu(F.linear(norm(x), c_fc)).square(), c_proj)

            # Save skip connection at layer 3
            if i in skip_in:
                skip_buffer = x.clone()

            # Save backout at layer 7
            if i == backout_layer:
                x_backout = x.clone()

        # Backout: subtract scaled layer 7 output from final hidden state
        # This removes "context-building" contributions not needed for direct prediction
        x = x - backout_lambda * x_backout

        # Final norm and lm_head
        x = norm(x)
        logits = self.lm_head(x)

        # Apply softcap (inference version: 23 * sigmoid((logits + 5) / 7.5))
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)

        return logits

# =============================================================================
# KV Cache
# =============================================================================

class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous embedding for smear gate
        self.prev_embed = None
        # Previous token for bigram embedding
        self.prev_token = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embed = None
        self.prev_token = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy prev_embed for smear gate (expand to batch size if needed)
        if other.prev_embed is not None:
            self.prev_embed = other.prev_embed.expand(self.batch_size, -1, -1).clone()
        else:
            self.prev_embed = None
        # Copy prev_token for bigram embedding (expand to batch size if needed)
        if other.prev_token is not None:
            self.prev_token = other.prev_token.expand(self.batch_size, -1).clone()
        else:
            self.prev_token = None


# =============================================================================
# Sampling
# =============================================================================

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# =============================================================================
# Generation engine
# =============================================================================

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Generate tokens autoregressively. Prefills batch=1, then decodes num_samples in parallel."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        dtype = torch.bfloat16  # CUDA-only, always bfloat16
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

# =============================================================================
# ChatTokenizer -- minimal tiktoken-based tokenizer for generative eval
# Ported from nanochat/nanochat/tokenizer.py (RustBPETokenizer, inference-only)
# =============================================================================

class ChatTokenizer:
    """Minimal tokenizer for chat inference: encode/decode + conversation rendering."""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        import tiktoken  # lazy import -- only needed at eval time
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text):
        if isinstance(text, str):
            return self.enc.encode_ordinary(text)
        elif isinstance(text, list):
            return self.enc.encode_ordinary_batch(text, num_threads=8)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return self.enc.decode(ids)

    def render_conversation(self, conversation, max_tokens=2048):
        """Tokenize a chat conversation into (ids, mask). mask=1 for assistant tokens."""
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Merge system message into user message if present
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]

        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is {message['role']} but expected {must_be_from}"
            content = message["content"]
            if message["role"] == "user":
                assert isinstance(content, str)
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    add_tokens(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                add_tokens(assistant_end, 1)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        """Render conversation primed for assistant completion (no assistant message, ends with <|assistant_start|>)."""
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, mask = self.render_conversation(conversation)
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids

# =============================================================================
# Generative eval tasks (ported from nanochat/tasks/)
# =============================================================================

class Task:
    """Base class for evaluation tasks. Supports lightweight slicing."""
    def __init__(self, start=0, stop=None, step=1):
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def eval_type(self):
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def evaluate(self, problem, completion):
        raise NotImplementedError

    def __len__(self):
        stop = self.num_examples() if self.stop is None else self.stop
        span = stop - self.start
        return (span + self.step - 1) // self.step

    def __getitem__(self, index):
        physical_index = self.start + index * self.step
        return self.get_example(physical_index)


# --- GSM8K ---

_GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def _extract_gsm_answer(completion):
    match = _GSM_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


class GSM8K(Task):
    def __init__(self, subset="main", split="test", **kwargs):
        super().__init__(**kwargs)
        from datasets import load_dataset
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question, answer = row['question'], row['answer']
        assistant_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                assistant_parts.append({"type": "python", "text": expr})
                assistant_parts.append({"type": "python_output", "text": result})
            else:
                assistant_parts.append({"type": "text", "text": part})
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_parts},
            ]
        }

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str)
        last_text_part = conversation['messages'][-1]['content'][-1]['text']
        ref_num = _extract_gsm_answer(last_text_part)
        pred_num = _extract_gsm_answer(assistant_response)
        return int(pred_num == ref_num)


# --- HumanEval ---

def _extract_imports(prompt):
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            break
    return '\n'.join(imports)


def _extract_program(completion):
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


# --- Sandboxed code execution for HumanEval (from nanochat/execution.py) ---

@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    error: str = None
    timeout: bool = False
    memory_exceeded: bool = False

@contextlib.contextmanager
def _time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

@contextlib.contextmanager
def _capture_io():
    class WriteOnlyStringIO(io.StringIO):
        def read(self, *a, **k): raise IOError
        def readline(self, *a, **k): raise IOError
        def readlines(self, *a, **k): raise IOError
        def readable(self, *a, **k): return False

    class _redirect_stdin(contextlib._RedirectStream):
        _stream = "stdin"

    stdout_cap = io.StringIO()
    stderr_cap = io.StringIO()
    stdin_block = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stdout_cap):
        with contextlib.redirect_stderr(stderr_cap):
            with _redirect_stdin(stdin_block):
                yield stdout_cap, stderr_cap

def _reliability_guard(max_mem_bytes):
    if platform.uname().system != "Darwin":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (max_mem_bytes, max_mem_bytes))
        resource.setrlimit(resource.RLIMIT_STACK, (max_mem_bytes, max_mem_bytes))
    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for attr in ['kill','system','putenv','remove','removedirs','rmdir','fchdir',
                 'setuid','fork','forkpty','killpg','rename','renames','truncate',
                 'replace','unlink','fchmod','fchown','chmod','chown','chroot',
                 'lchflags','lchmod','lchown','getcwd','chdir']:
        if hasattr(os, attr):
            setattr(os, attr, None)
    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    import subprocess
    subprocess.Popen = None

def _unsafe_execute(code, timeout_sec, max_mem, result_dict):
    with tempfile.TemporaryDirectory() as dirname:
        cwd = os.getcwd()
        os.chdir(dirname)
        import shutil as _shutil
        _rmtree = _shutil.rmtree
        _chdir = os.chdir
        _reliability_guard(max_mem)
        result_dict.update({"success": False, "stdout": "", "stderr": "", "timeout": False, "memory_exceeded": False, "error": None})
        try:
            with _capture_io() as (stdout_cap, stderr_cap):
                with _time_limit(timeout_sec):
                    exec(code, {})
            result_dict.update({"success": True, "stdout": stdout_cap.getvalue(), "stderr": stderr_cap.getvalue()})
        except TimeoutError:
            result_dict.update({"timeout": True, "error": "Execution timed out"})
        except MemoryError as e:
            result_dict.update({"memory_exceeded": True, "error": f"Memory limit exceeded: {e}"})
        except BaseException as e:
            result_dict.update({"error": f"{type(e).__name__}: {e}"})
        _shutil.rmtree = _rmtree
        os.chdir = _chdir
        _chdir(cwd)

def execute_code(code, timeout=5.0, max_memory_bytes=256*1024*1024):
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    p = multiprocessing.Process(target=_unsafe_execute, args=(code, timeout, max_memory_bytes, result_dict))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        return ExecutionResult(success=False, stdout="", stderr="", error="Execution timed out (process killed)", timeout=True)
    if not result_dict:
        return ExecutionResult(success=False, stdout="", stderr="", error="Execution failed (no result)", timeout=True)
    return ExecutionResult(
        success=result_dict["success"], stdout=result_dict["stdout"], stderr=result_dict["stderr"],
        error=result_dict["error"], timeout=result_dict["timeout"], memory_exceeded=result_dict["memory_exceeded"],
    )


class HumanEval(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from datasets import load_dataset
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt, solution = row['prompt'], row['canonical_solution']
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"{prompt}\n{solution}"},
            ],
            "entry_point": row['entry_point'],
            "test": row['test'],
        }

    def evaluate(self, conversation, completion):
        imports = _extract_imports(conversation['messages'][0]['content'])
        code = _extract_program(completion)
        program = (
            imports + "\n\n" + code + "\n\n"
            + conversation['test'] + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program)
        return result.success


# --- SpellingBee ---

_SPELLINGBEE_WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
_SPELLINGBEE_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_SPELLINGBEE_TEST_SEED_OFFSET = 10_000_000

_SPELLINGBEE_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    "\u00bfCu\u00e1ntas {letter} hay en {word}?",
    "\u00bfCu\u00e1ntas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "\u00bfCu\u00e1ntas letras {letter} tiene {word}?",
    "{word}\u4e2d\u6709\u591a\u5c11\u4e2a{letter}",
    "{word}\u91cc\u6709\u51e0\u4e2a{letter}",
    "\u6570\u4e00\u4e0b{word}\u4e2d\u7684{letter}",
    "{word}\u8fd9\u4e2a\u8bcd\u91cc\u6709\u591a\u5c11{letter}",
    "{word}\uc5d0 {letter}\uac00 \uba87 \uac1c \uc788\ub098\uc694",
    "{word}\uc5d0\uc11c {letter}\uc758 \uac1c\uc218\ub294",
    "{word}\uc5d0 {letter}\uac00 \uba87 \ubc88 \ub098\uc624\ub098\uc694",
    "{word}\ub77c\ub294 \ub2e8\uc5b4\uc5d0 {letter}\uac00 \uba87 \uac1c",
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} appara\u00eet dans {word}",
    "Compte les {letter} dans {word}",
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Z\u00e4hle die {letter} in {word}",
    "{word}\u306b{letter}\u306f\u4f55\u500b\u3042\u308a\u307e\u3059\u304b",
    "{word}\u306e\u4e2d\u306b{letter}\u304c\u3044\u304f\u3064",
    "{word}\u306b{letter}\u304c\u4f55\u56de\u51fa\u3066\u304f\u308b",
]

class SpellingBee(Task):
    def __init__(self, size=256, split="test", **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.split = split
        # Download word list (rank 0 downloads, others wait)
        self._word_list_path = os.path.join(
            os.environ.get("DATA_PATH", "."),
            f"data/{DATASET_NAME}/words_alpha.txt"
        )
        if rank == 0 and not os.path.exists(self._word_list_path):
            urllib.request.urlretrieve(_SPELLINGBEE_WORD_LIST_URL, self._word_list_path)
        dist.barrier()
        with open(self._word_list_path, 'r', encoding='utf-8') as f:
            self.words = [line.strip() for line in f]

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else _SPELLINGBEE_TEST_SEED_OFFSET + index
        rng = random.Random(seed)
        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(_SPELLINGBEE_LETTERS)
        count = word.count(letter)
        template = rng.choice(_SPELLINGBEE_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', "'", '"']
        letter_wrapped = f"{rng.choice(quote_options)}{letter}{rng.choice(quote_options)}"
        word_wrapped = f"{rng.choice(quote_options)}{word}{rng.choice(quote_options)}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5:
            user_msg += "?"
        # Build assistant response parts
        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.\n\nFirst spell the word out:\n{word}:{word_letters}\n\nThen count the occurrences of '{letter}':\n"
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"
        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        assistant_parts.append({"type": "python", "text": f"'{word}'.count('{letter}')"})
        assistant_parts.append({"type": "python_output", "text": str(count)})
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_parts},
            ]
        }

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str)
        last_text_part = conversation['messages'][-1]['content'][-1]['text']
        ref_num = _extract_gsm_answer(last_text_part)
        pred_num = _extract_gsm_answer(assistant_response)
        return int(pred_num == ref_num)

# =============================================================================

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)


def run_generative_eval(task_object, tokenizer, model, engine,
                        num_samples, max_new_tokens, temperature, top_k,
                        max_problems=None):
    """Run generative evaluation on a single task, distributed across DDP ranks."""
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_passed, total = 0, 0

    for i in range(rank, num_problems, world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)
        print(f"\r\033[KRank {rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()  # finish progress line

    # Aggregate across ranks
    num_passed_t = torch.tensor([num_passed], dtype=torch.long, device=device)
    total_t = torch.tensor([total], dtype=torch.long, device=device)
    dist.all_reduce(num_passed_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
    num_passed = num_passed_t.item()
    total = total_t.item()
    accuracy = num_passed / total if total > 0 else 0.0
    return accuracy


def evaluate_chat_generative(training_model, dev, dataset_name):
    """
    Run generative chat evaluation on GSM8K, HumanEval, SpellingBee.
    Creates an inference-only model from the training model, then runs benchmarks.
    Returns dict with results, centered_results, and generative_chatcore_metric.
    """
    global DATASET_NAME
    DATASET_NAME = dataset_name
    _init_distributed()

    # Build inference model from trained weights
    inference_model = GPTInference.from_training_model(training_model, dev)

    # Load tokenizer and create engine
    tokenizer_dir = os.path.join(
        os.environ.get("DATA_PATH", "."),
        f"data/{DATASET_NAME}/tokenizer"
    )
    tokenizer = ChatTokenizer.from_directory(tokenizer_dir)
    engine = Engine(inference_model, tokenizer)

    eval_params = dict(num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50)

    task_configs = [
        ("GSM8K", lambda: GSM8K(subset="main", split="test"), 0.0),
        ("HumanEval", lambda: HumanEval(), 0.0),
        ("SpellingBee", lambda: SpellingBee(size=256, split="test"), 0.0),
    ]

    results = {}
    centered_results = {}

    for label, task_fn, baseline in task_configs:
        print0(f"Chat generative eval: {label}...")
        start_time = time.time()
        task_object = task_fn()
        accuracy = run_generative_eval(task_object, tokenizer, inference_model, engine, **eval_params)
        centered = (accuracy - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0
        results[label] = accuracy
        centered_results[label] = centered
        elapsed = time.time() - start_time
        print0(f"  {label}: accuracy={accuracy:.4f} | centered={centered:.4f} | time={elapsed:.2f}s")

    generative_chatcore = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "generative_chatcore_metric": generative_chatcore,
    }
