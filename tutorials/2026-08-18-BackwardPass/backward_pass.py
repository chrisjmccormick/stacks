# ==============================================================================
# Part 1: the production kernels
# ==============================================================================


# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

import os
import sys
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import torch
import torch.nn.functional as F
from torch import Tensor


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

class Config:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    data_dir: Path = Path.home() / ".cache" / "qwen-arithmetic" / "data"

    # Model (Qwen2.5-0.5B -- asserted against the banks at load)
    n_layers:   int = 24
    d_model:    int = 896
    n_qo_heads: int = 14
    n_kv_heads: int = 2
    d_head:     int = 64
    d_mlp:      int = 4864
    d_vocab:    int = 151936
    rope_theta: float = 1_000_000.0
    rms_eps:    float = 1e-6

    # Part 1 training: SFT on the model's own CORRECT generations
    n_steps:       int = 20
    docs_per_step: int = 32
    seed:          int = 1337

    # Optimizer -- AdamW on fp32 masters; LR ramps linearly over the first
    # warmup_steps, then holds (Adam's first steps from zero moments are
    # sign-like and jolt the loss without it; 1e-5 still jolts it to ~2x
    # before recovering, 5e-6 does not)
    lr:           float = 5e-6
    warmup_steps: int   = 5
    weight_decay: float = 0.01
    beta1:        float = 0.9
    beta2:        float = 0.999
    adam_eps:     float = 1e-8
    loss_scale:   float = 4096.0      # static fp16 loss scale (backoff on overflow)

    # Training packs (packed varlen; a pack is one fwd/bwd)
    train_t:      int = 2048          # tokens per pack (cap)
    sel_cap:      int = 1536          # completion positions per pack (cap)
    ce_chunk:     int = 512           # lm_head/CE row chunk
    max_docs:     int = 64            # docs per pack (cap)
    pack_quantum: int = 256           # T trimmed to a multiple of this

    # Part 2: a few docs -> one pack, small enough that the full (T, V)
    # logits of the unchunked lm_head fit comfortably
    part2_docs:   int = 8

cfg = Config()

assert cfg.sel_cap % cfg.ce_chunk == 0 and cfg.sel_cap <= cfg.train_t
assert cfg.train_t % cfg.pack_quantum == 0
assert 1 <= cfg.warmup_steps <= cfg.n_steps
cfg.d_q   = cfg.n_qo_heads * cfg.d_head    # 896
cfg.d_kv  = cfg.n_kv_heads * cfg.d_head    # 128
cfg.d_qkv = cfg.d_q + 2 * cfg.d_kv         # 1152 -- fused QKV rows: [Q | K | V]
cfg.half  = cfg.d_head // 2
cfg.group = cfg.n_qo_heads // cfg.n_kv_heads   # 7 query heads per KV head
cfg.rope_t = cfg.train_t                   # rotary cache length (positions restart per doc)


# ------------------------------------------------------------------------------
# Prepare
# ------------------------------------------------------------------------------

IM_END = 151645       # <|im_end|>    ends the assistant turn
ENDOFTEXT = 151643    # <|endoftext|> ends the document; doubles as the pack pad
PAD_ID = ENDOFTEXT

BANKS_REPO = "ChrisMcCormick/qwen-arithmetic-t4"
DATA_REPO = "ChrisMcCormick/basic-arithmetic"
GEN_SPLITS = ("val", "test_id", "test_ood")

_BANKS_PATH = cfg.data_dir / "banks_fp16_Qwen2.5-0.5B-Instruct.safetensors"
_TOK_PATH = cfg.data_dir / "tokenizer.json"
_FA_DIR = cfg.data_dir / "fa_turing"

from huggingface_hub import hf_hub_download, snapshot_download

print(f"fetching {BANKS_REPO} + {DATA_REPO} ...", flush=True)
cfg.data_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(BANKS_REPO, local_dir=str(cfg.data_dir),
                  allow_patterns=[_TOK_PATH.name, _BANKS_PATH.name])
assert _BANKS_PATH.exists(), f"{BANKS_REPO} is missing {_BANKS_PATH.name}"
_gen_paths = {s: hf_hub_download(DATA_REPO, f"baseline_eval/{s}_generations.parquet",
                                 repo_type="dataset") for s in GEN_SPLITS}

# @markdown Fetch (or build) the flash-attention-turing extension

# fa_turing.py lives next to the wheels in the model repo. It picks the wheel
# matching THIS runtime's ABI and extracts it (a few seconds); if Colab has
# moved to an image we have no wheel for yet, it builds one here instead of
# leaving you stuck (~12 min, and it says so).
_fa_py = hf_hub_download(BANKS_REPO, "fa_turing/fa_turing.py")
sys.path.insert(0, str(Path(_fa_py).parent))
from fa_turing import ensure as _ensure_fa

_fa_info = _ensure_fa(BANKS_REPO, cache_dir=_FA_DIR)


# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

import pyarrow.parquet as pq
from tokenizers import Tokenizer as _RustTokenizer

tokenizer = _RustTokenizer.from_file(str(_TOK_PATH))
assert tokenizer.token_to_id("<|im_end|>") == IM_END


def encode(text: str) -> list[int]:
    # add_special_tokens=False adds no template tokens; the ChatML specials in
    # the text itself still map to their single ids.
    return tokenizer.encode(text, add_special_tokens=False).ids


def decode(ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


docs: list[tuple[list[int], list[int]]] = []   # (prompt_ids, completion_ids)
doc_text: list[tuple[str, str]] = []           # the same, as text (for display)
n_gen = n_wrong = n_trunc = 0
for _s in GEN_SPLITS:
    _t = pq.read_table(_gen_paths[_s]).to_pydict()
    for _pt, _g, _r in zip(_t["prompt_text"], _t["generation"], _t["reward"]):
        n_gen += 1
        if _r != 1.0:
            n_wrong += 1
            continue
        _gids = encode(_g)
        if len(_gids) >= 256:          # ran into the generation budget: no natural end
            n_trunc += 1
            continue
        docs.append((encode(_pt), _gids + [IM_END]))
        doc_text.append((_pt, _g))
del _t
_n_tok = sum(len(p) + len(g) for p, g in docs)
print(f"{n_gen} generations -> {len(docs)} training docs (dropped {n_wrong} wrong, "
      f"{n_trunc} truncated) | {_n_tok:,} tokens | prompts {min(len(p) for p, _ in docs)}-"
      f"{max(len(p) for p, _ in docs)} tok, completions {min(len(g) for _, g in docs)}-"
      f"{max(len(g) for _, g in docs)} tok", flush=True)


# ------------------------------------------------------------------------------
# Packing
# ------------------------------------------------------------------------------

def _roundup(n: int, q: int) -> int:
    return -(-n // q) * q


def plan_packs(docs: list[tuple[list[int], list[int], float]]):
    """docs: (prompt_ids, gen_ids, weight). Returns (packs, pack_stats); each
    pack is a dict of numpy arrays:
      idx (T,) int32       packed inputs (per doc: seq[:-1])
      pos (T,) int64       rotary positions, restarting at each doc
      cu  (n_seg+1,) int32 segment boundaries: the docs, then the pad tail
      sel (S,) int64       positions of completion targets (lm_head runs
                           only here); padded with 0
      tgt (S,) int64       targets at sel; padded with 0
      w   (S,) fp32        per-token loss weight at sel; padded with 0
      max_seg              the longest segment (for the attention kernel)
    First-fit-decreasing over three caps: train_t tokens, sel_cap completion
    positions, max_docs docs. T is trimmed to a multiple of pack_quantum; S is
    pinned at sel_cap (the CE chunk loop is a python range over S, so a second
    value would compile a second graph). The pad tail is a real attended
    segment carrying zero weight, with VARYING ids and positions -- a constant
    pad segment can NaN the attention backward."""
    order = sorted(range(len(docs)), key=lambda i: -(len(docs[i][0]) + len(docs[i][1])))
    packs_docs, packs_tok, packs_selc = [], [], []
    for i in order:
        p, g, _ = docs[i]
        n_tok, n_sel = len(p) + len(g) - 1, len(g)
        assert n_tok <= cfg.train_t and n_sel <= cfg.sel_cap, "doc exceeds pack caps"
        for j in range(len(packs_docs)):
            if (packs_tok[j] + n_tok <= cfg.train_t and packs_selc[j] + n_sel <= cfg.sel_cap
                    and len(packs_docs[j]) < cfg.max_docs):
                packs_docs[j].append(i)
                packs_tok[j] += n_tok
                packs_selc[j] += n_sel
                break
        else:
            packs_docs.append([i])
            packs_tok.append(n_tok)
            packs_selc.append(n_sel)

    packs = []
    pad_tokens = cap_tokens = 0
    for members, n_tok, n_sel in zip(packs_docs, packs_tok, packs_selc):
        T = min(cfg.train_t, _roundup(n_tok, cfg.pack_quantum))
        S = cfg.sel_cap
        idx = np.full(T, PAD_ID, dtype=np.int32)
        pos = np.zeros(T, dtype=np.int64)
        cu = [0]
        sel = np.zeros(S, dtype=np.int64)
        tgt = np.zeros(S, dtype=np.int64)
        w = np.zeros(S, dtype=np.float32)
        o = s = 0
        max_seg = 0
        for i in members:
            p, g, wt = docs[i]
            seq = np.asarray(p + g, dtype=np.int64)
            n = len(seq) - 1
            idx[o:o + n] = seq[:-1]
            pos[o:o + n] = np.arange(n)
            ng = len(g)
            sel[s:s + ng] = o + n - ng + np.arange(ng)   # targets seq[1:]; the
            tgt[s:s + ng] = seq[-ng:]                    # last ng are the completion
            w[s:s + ng] = wt
            s += ng
            o += n
            cu.append(o)
            max_seg = max(max_seg, n)
        if o < T:
            n_pad = T - o
            idx[o:] = 1 + (np.arange(n_pad) % 4096)
            pos[o:] = np.arange(n_pad) % cfg.rope_t
            cu.append(T)
            max_seg = max(max_seg, n_pad)
            pad_tokens += n_pad
        cap_tokens += T
        packs.append(dict(idx=idx, pos=pos, cu=np.asarray(cu, dtype=np.int32),
                          sel=sel, tgt=tgt, w=w, max_seg=max_seg,
                          n_tok=o, n_sel=s, n_docs=len(members)))
    return packs, dict(n_packs=len(packs), pad_tokens=pad_tokens,
                       cap_tokens=max(1, cap_tokens))


# ------------------------------------------------------------------------------
# CUDA init
# ------------------------------------------------------------------------------

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda", 0)
torch.cuda.set_device(device)
try:   # the env var above only counts if torch was not already imported (Colab)
    torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
except Exception:
    pass
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
torch.set_grad_enabled(False)          # nothing here uses autograd -- that is the point
# fp16 GEMMs accumulate in fp32 (cuBLAS default) -- never let split-K reduce
# in fp16.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
_cc = torch.cuda.get_device_capability()
_gpu_name = torch.cuda.get_device_name(0)
print(f"{_gpu_name} (sm_{_cc[0]}{_cc[1]}, {torch.cuda.mem_get_info()[1] / 2**30:.1f} GB) | "
      f"torch {torch.__version__} | cuda {torch.version.cuda}", flush=True)
assert _cc >= (7, 0), "fp16 tensor cores required (sm70+)"


# ------------------------------------------------------------------------------
# Attention kernel
# ------------------------------------------------------------------------------

import flash_attn_turing

_ATTN_SCALE = cfg.d_head ** -0.5


@torch.library.custom_op("t4::fa_fwd", mutates_args=())
def _fa_fwd_op(q: Tensor, k: Tensor, v: Tensor, cu: Tensor,
               max_seg: int) -> tuple[Tensor, Tensor]:
    return flash_attn_turing.varlen_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(), cu, cu,
        max_seg, max_seg, _ATTN_SCALE, True)


@_fa_fwd_op.register_fake
def _(q, k, v, cu, max_seg):
    return (torch.empty_like(q),
            q.new_empty((cu.shape[0] - 1, q.shape[1], max_seg), dtype=torch.float32))


@torch.library.custom_op("t4::fa_bwd", mutates_args=())
def _fa_bwd_op(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor,
               lse: Tensor, cu: Tensor,
               max_seg: int) -> tuple[Tensor, Tensor, Tensor]:
    return flash_attn_turing.varlen_bwd(
        q, k, v, out, lse, dout.contiguous(), cu, cu,
        max_seg, max_seg, _ATTN_SCALE, True)


@_fa_bwd_op.register_fake
def _(dout, q, k, v, out, lse, cu, max_seg):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def attn_varlen_fwd(q: Tensor, k: Tensor, v: Tensor, cu: Tensor, max_seg: int):
    """Packed causal attention. q (T, H_q, Dh), k/v (T, H_kv, Dh) fp16;
    cu (n_seg+1,) int32 on device. -> (out (T, H_q, Dh),
    lse (n_seg, H_q, max_seg) fp32), the lse passed straight back to
    attn_varlen_bwd."""
    return _fa_fwd_op(q, k, v, cu, max_seg)


def attn_varlen_bwd(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor,
                    lse: Tensor, cu: Tensor, max_seg: int):
    """Backward for attn_varlen_fwd -> (dq (T,H_q,Dh), dk (T,H_kv,Dh), dv).
    dK/dV arrive at the KV head count already: GQA is inside the kernel."""
    return _fa_bwd_op(dout, q, k, v, out, lse, cu, max_seg)


# ------------------------------------------------------------------------------
# Model load
# ------------------------------------------------------------------------------

class Model:
    embed:      Tensor   # (V, D) fp16 -- tied: input table AND lm_head
    W_QKV:      Tensor   # (L, 1152, 896)
    b_QKV:      Tensor   # (L, 1152)       Qwen2.5 QKV biases
    W_O:        Tensor   # (L, 896, 896)
    W_GU:       Tensor   # (L, 9728, 896)  [gate | up]
    W_down:     Tensor   # (L, 896, 4864)
    attn_norm:  Tensor   # (L, 896)  input_layernorm weights
    mlp_norm:   Tensor   # (L, 896)  post_attention_layernorm weights
    final_norm: Tensor   # (896,)
    cos: Tensor          # rotary caches (not trained)
    sin: Tensor

    weight_names = ("embed", "W_QKV", "b_QKV", "W_O", "W_GU", "W_down",
                    "attn_norm", "mlp_norm", "final_norm")
    big_names = ("embed", "W_QKV", "W_O", "W_GU", "W_down")

    def __iter__(self):
        return (getattr(self, n) for n in self.weight_names)


print(f"loading {cfg.model_id} banks ...", flush=True)
from safetensors.torch import load_file

m = Model()
_sd = load_file(str(_BANKS_PATH), device=str(device))   # fp16, straight to device
for _n in Model.weight_names:
    setattr(m, _n, _sd[_n].to(torch.float16).contiguous())
del _sd
torch.cuda.empty_cache()

assert m.embed.shape == (cfg.d_vocab, cfg.d_model)
assert m.W_QKV.shape == (cfg.n_layers, cfg.d_qkv, cfg.d_model)
assert m.W_GU.shape == (cfg.n_layers, 2 * cfg.d_mlp, cfg.d_model)

for _n in Model.weight_names:
    p = getattr(m, _n)
    p.pname = _n
    p.master = p.float()                       # == live, exactly, at init
    gd = torch.float16 if _n in Model.big_names else torch.float32
    p.gacc = torch.zeros(p.shape, dtype=gd, device=device)
    p.exp_avg = torch.zeros(p.shape, dtype=torch.float32, device=device)
    p.exp_avg_sq = torch.zeros(p.shape, dtype=torch.float32, device=device)
    if p.dim() >= 2 and _n != "embed":
        p.grad_slices = list(p.gacc.unbind(0))
del p

# Rotary cache -- HF/Qwen convention (rotate_half, non-interleaved): channel j
# pairs with j + head_dim/2; cos/sin are (T, head_dim/2) and broadcast over
# both halves. Forward rotation: y1 = q1*cos - q2*sin ; y2 = q2*cos + q1*sin.
_inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2, dtype=torch.float32, device=device) / cfg.d_head))
_freqs = torch.outer(torch.arange(cfg.rope_t, dtype=torch.float32, device=device), _inv_freq)
m.cos = _freqs.cos().to(torch.float16)    # (rope_t, 32)
m.sin = _freqs.sin().to(torch.float16)
del _inv_freq, _freqs

_n_params = sum(p.numel() for p in m)
_state_gb = sum(p.numel() * (14 + (2 if p.pname in Model.big_names else 4)) for p in m) / 2**30
print(f"loaded: {_n_params:,} params | live fp16 + master fp32 + fp32 moments "
      f"+ grads = {_state_gb:.1f} GB", flush=True)


# ------------------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------------------

class AdamWTabs(NamedTuple):
    wd_mul: Tensor           # 1 - lr*wd            decoupled weight decay
    one_minus_beta1: Tensor  # exp_avg lerp weight
    one_minus_beta2: Tensor  # exp_avg_sq lerp weight
    rsqrt_bias2: Tensor      # 1/sqrt(1 - beta2^t)
    step_size: Tensor        # lr / (1 - beta1^t)


def build_schedules(n_steps: int):
    N = max(1, n_steps)
    t1 = np.arange(1, N + 1, dtype=np.float64)
    lr = cfg.lr * np.minimum(1.0, t1 / cfg.warmup_steps)   # linear warmup, then constant
    dev = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return SimpleNamespace(
        tabs=AdamWTabs(
            wd_mul          = dev(1.0 - lr * cfg.weight_decay),
            one_minus_beta1 = dev(np.full(N, 1.0 - cfg.beta1)),
            one_minus_beta2 = dev(np.full(N, 1.0 - cfg.beta2)),
            rsqrt_bias2     = dev(1.0 / np.sqrt(1.0 - cfg.beta2 ** t1)),
            step_size       = dev(lr / (1.0 - cfg.beta1 ** t1)),
        ),
        lr_host=lr, num_steps=N)


sched = build_schedules(cfg.n_steps)
# Shape (1,), NOT 0-D: a 1-element index tensor is plain advanced indexing
# that dynamo can trace; a 0-D index is a data-dependent select it cannot.
t_step = torch.zeros(1, dtype=torch.int64, device=device)   # advanced on-device
inv_scale = torch.tensor(1.0 / cfg.loss_scale, dtype=torch.float32, device=device)
opt_t = 0   # steps taken (host mirror of t_step)


@torch.compile(dynamic=False, fullgraph=True)
def _adamw_all(c: AdamWTabs, t: Tensor, scale: Tensor, eps: float) -> None:
    """AdamW over every trained tensor: fp32 master of record, fp16 live copy
    re-derived by round-to-nearest, and the gradient zeroed while it is still
    in registers. One compiled graph, one kernel per tensor."""
    for p in m:
        g = p.gacc.float() * scale
        p.master.mul_(c.wd_mul[t])
        p.exp_avg.lerp_(g, c.one_minus_beta1[t])
        p.exp_avg_sq.lerp_(g.square(), c.one_minus_beta2[t])
        denom = p.exp_avg_sq.sqrt() * c.rsqrt_bias2[t] + eps
        p.master.sub_(c.step_size[t] * (p.exp_avg / denom))
        p.copy_(p.master)        # fp32 -> fp16, round-to-nearest
        p.gacc.zero_()           # fused: the grad is already resident here


def optimizer_step() -> None:
    """One step, then advance t on device. Leaves the gradients zeroed."""
    global opt_t
    _adamw_all(sched.tabs, t_step, inv_scale, cfg.adam_eps)
    t_step.add_(1)
    opt_t += 1


@torch.compile(dynamic=False, fullgraph=True)
def _grad_sq_sum() -> Tensor:
    tot = torch.zeros((), dtype=torch.float32, device=device)
    for p in m:
        tot += p.gacc.float().square().sum()
    return tot


def grad_global_norm() -> float:
    """sqrt(sum g^2) over every trained tensor (still carrying the loss
    scale) -- nan/inf if any grad overflowed. The one host sync of the
    training step."""
    return math.sqrt(float(_grad_sq_sum()))


def zero_grads() -> None:
    for p in m:
        p.gacc.zero_()


# ------------------------------------------------------------------------------
# Forward and backward
# ------------------------------------------------------------------------------

fp16 = lambda x: x.to(torch.float16)


def _rms_fwd(x):
    """Unweighted rms_norm + 1/rms (fp32), Qwen eps."""
    r = (x.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
    return fp16(x.float() * r), r


def _rms_bwd(d_hat, x_hat, r):
    xf, df = x_hat.float(), d_hat.float()
    return fp16(r * (df - xf * (xf * df).mean(dim=-1, keepdim=True)))


class LayerStash(NamedTuple):
    """One layer's forward activations held for backward."""
    xb_hat:     Tensor   # (T, D)        attn-norm output, unweighted
    xb_inv_rms: Tensor   # (T, 1) fp32
    q:          Tensor   # (T, 14, 64)   post-rope (what attention consumed)
    k:          Tensor   # (T, 2, 64)    post-rope
    v:          Tensor   # (T, 2, 64)
    y:          Tensor   # (T, 14, 64)   attn out
    lse:        Tensor   # (n_seg, 14, max_seg) fp32   softmax lse
    xm:         Tensor   # (T, D)        post-attn residual (mlp norm recomputed)
    gu:         Tensor   # (T, 9728)     fused gate|up pre-activation


def forward_backward(idx, pos, cu, sel, tgt_sel, w_sel, max_seg: int):
    """One pack: forward, stash, explicit backward into .gacc. Returns the
    summed weighted CE (carrying the loss scale like everything else)."""
    T = idx.size(0)
    Hq, Hkv, Dh = cfg.n_qo_heads, cfg.n_kv_heads, cfg.d_head
    cos = m.cos[pos].unsqueeze(1)   # (T, 1, 32) -- broadcasts over heads
    sin = m.sin[pos].unsqueeze(1)

    # ---- Forward ----
    x = F.embedding(idx, m.embed)
    stash = []
    for i in range(cfg.n_layers):
        xb_hat, xb_r = _rms_fwd(x)
        xbn = xb_hat * m.attn_norm[i]
        qkv = torch.addmm(m.b_QKV[i], xbn, m.W_QKV[i].mT)
        q = qkv[:, :cfg.d_q].view(T, Hq, Dh)
        k = qkv[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(T, Hkv, Dh)
        v = qkv[:, cfg.d_q + cfg.d_kv:].view(T, Hkv, Dh).contiguous()
        q1, q2 = q[..., :cfg.half], q[..., cfg.half:]
        k1, k2 = k[..., :cfg.half], k[..., cfg.half:]
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        y, lse = attn_varlen_fwd(q, k, v, cu, max_seg)
        y = y.contiguous()
        xm = torch.addmm(x, y.view(T, -1), m.W_O[i].mT)
        xm_hat, xm_r = _rms_fwd(xm)
        xmn = xm_hat * m.mlp_norm[i]
        gu = xmn @ m.W_GU[i].mT                  # (T, 9728)
        g, u = gu[:, :cfg.d_mlp], gu[:, cfg.d_mlp:]
        x = torch.addmm(xm, F.silu(g) * u, m.W_down[i].mT)
        stash.append(LayerStash(xb_hat=xb_hat, xb_inv_rms=xb_r, q=q, k=k, v=v,
                                y=y, lse=lse, xm=xm, gu=gu))
    xf_hat, xf_r = _rms_fwd(x)
    xfn = xf_hat * m.final_norm

    # ---- LM head + weighted CE (chunked over the gathered completion rows) ----
    xfn_sel = xfn.index_select(0, sel)           # (S, D)
    S = sel.size(0)
    loss = torch.zeros((), dtype=torch.float32, device=idx.device)
    sel_grads = []
    ar = torch.arange(cfg.ce_chunk, device=idx.device)
    for c0 in range(0, S, cfg.ce_chunk):
        xs = xfn_sel[c0:c0 + cfg.ce_chunk]           # (c, D)
        tg = tgt_sel[c0:c0 + cfg.ce_chunk]
        wc = w_sel[c0:c0 + cfg.ce_chunk]
        c = xs.size(0)
        lg = (xs @ m.embed.mT).float()               # (c, V) fp32 -- tied head
        cmax = lg.amax(dim=1, keepdim=True)
        lg.sub_(cmax)
        ly = lg.gather(1, tg.unsqueeze(1)).squeeze(1)    # logit_y - cmax
        lg.exp_()                                        # e
        ssum = lg.sum(dim=1, keepdim=True)
        loss += (wc * (ssum.log().squeeze(1) - ly)).sum()   # lse - logit_y
        lg.div_(ssum)                                    # p
        lg[ar[:c], tg] -= 1.0                            # p - onehot
        lg.mul_(wc.unsqueeze(1))                         # weighted
        lgh = fp16(lg)                                   # (c, V) fp16
        del lg
        m.embed.gacc.addmm_(lgh.mT, xs)                  # (V, c) @ (c, D)
        sel_grads.append(lgh @ m.embed)                  # (c, D)
    xfn_grad = torch.zeros_like(xfn)
    xfn_grad.index_add_(0, sel, torch.cat(sel_grads))   # w=0 pads land as zeros at row 0
    del sel_grads, xfn_sel

    # ---- Backward ----
    m.final_norm.gacc.add_((xfn_grad.float() * xf_hat.float()).sum(dim=0))
    stream_grad = _rms_bwd(xfn_grad * m.final_norm, xf_hat, xf_r)
    del xfn_grad, xfn, xf_hat
    for i in reversed(range(cfg.n_layers)):
        st = stash[i]
        # MLP backward (SwiGLU)
        xm_hat, xm_r = _rms_fwd(st.xm)
        xmn = xm_hat * m.mlp_norm[i]
        g, u = st.gu[:, :cfg.d_mlp], st.gu[:, cfg.d_mlp:]
        sg = torch.sigmoid(g)
        silu_g = g * sg
        a = silu_g * u
        m.W_down.grad_slices[i].addmm_(stream_grad.mT, a)
        a_grad = stream_grad @ m.W_down[i]
        del a
        u_grad = a_grad * silu_g
        g_grad = a_grad * u * (sg * (1 + g * (1 - sg)))   # d silu / dg
        del a_grad, sg, silu_g
        gu_grad = torch.cat([g_grad, u_grad], dim=1)
        del g_grad, u_grad
        m.W_GU.grad_slices[i].addmm_(gu_grad.mT, xmn)
        xmn_grad = gu_grad @ m.W_GU[i]
        del gu_grad
        m.mlp_norm.grad_slices[i].add_((xmn_grad.float() * xm_hat.float()).sum(dim=0))
        xm_grad = stream_grad + _rms_bwd(xmn_grad * m.mlp_norm[i], xm_hat, xm_r)
        del xmn_grad, xm_hat, xmn
        # Attention backward
        xbn = st.xb_hat * m.attn_norm[i]
        m.W_O.grad_slices[i].addmm_(xm_grad.mT, st.y.view(T, -1))
        y_grad = (xm_grad @ m.W_O[i]).view(T, Hq, Dh)
        q_grad, k_grad, v_grad = attn_varlen_bwd(
            y_grad, st.q, st.k, st.v, st.y, st.lse, cu, max_seg)
        del y_grad
        q1g, q2g = q_grad[..., :cfg.half], q_grad[..., cfg.half:]
        k1g, k2g = k_grad[..., :cfg.half], k_grad[..., cfg.half:]
        q_grad = torch.cat([q1g * cos + q2g * sin, q2g * cos - q1g * sin], dim=-1)
        k_grad = torch.cat([k1g * cos + k2g * sin, k2g * cos - k1g * sin], dim=-1)
        qkv_grad = torch.cat([q_grad.reshape(T, cfg.d_q), k_grad.reshape(T, cfg.d_kv),
                              v_grad.reshape(T, cfg.d_kv)], dim=1)
        del q_grad, k_grad, v_grad
        m.b_QKV.grad_slices[i].add_(qkv_grad.sum(dim=0, dtype=torch.float32))
        m.W_QKV.grad_slices[i].addmm_(qkv_grad.mT, xbn)
        xbn_grad = qkv_grad @ m.W_QKV[i]
        del qkv_grad, xbn
        m.attn_norm.grad_slices[i].add_((xbn_grad.float() * st.xb_hat.float()).sum(dim=0))
        stream_grad = xm_grad + _rms_bwd(xbn_grad * m.attn_norm[i], st.xb_hat, st.xb_inv_rms)
        del xbn_grad, xm_grad
        stash[i] = None                          # free as we go
    # token embedding scatter (the tied table's second gradient path)
    m.embed.gacc.add_(torch.ops.aten.embedding_dense_backward(
        stream_grad, idx, cfg.d_vocab, -1, False))
    return loss


# Compiled: dynamic=True holds ONE graph across the pack's varying token count.
# The compile is a few minutes on Colab's 2 vCPUs and happens on the first call.
fb = torch.compile(forward_backward, dynamic=True, fullgraph=True)


# ------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------

loss_scale = cfg.loss_scale


def balanced_batches(rng: random.Random) -> list[list[int]]:
    """One epoch of batches with the SAME mix of doc lengths in each (sort by
    length, snake-deal into bins, shuffle the bin order) -- so the per-step
    loss is comparable step to step instead of tracking which batch happened
    to hold the long explanations. Same trick as the trainer's round schedule.
    The remainder (len(docs) % docs_per_step) is dropped each epoch."""
    n_b = len(docs) // cfg.docs_per_step
    order = list(range(len(docs)))
    rng.shuffle(order)
    order = sorted(order[:n_b * cfg.docs_per_step],
                   key=lambda i: len(docs[i][0]) + len(docs[i][1]), reverse=True)
    bins = [[] for _ in range(n_b)]
    for j, i in enumerate(order):
        lap, off = divmod(j, n_b)
        bins[off if lap % 2 == 0 else n_b - 1 - off].append(i)
    rng.shuffle(bins)
    return bins


_rng = random.Random(cfg.seed)
batches = []
while len(batches) < cfg.n_steps:
    batches += balanced_batches(_rng)
batches = batches[:cfg.n_steps]
print(f"\n== Part 1: {cfg.n_steps} AdamW steps x {cfg.docs_per_step} docs, lr {cfg.lr:g} "
      f"({cfg.warmup_steps}-step warmup), loss scale {loss_scale:g} (step 0 includes the compile) ==",
      flush=True)
part1_log = []
for step in range(cfg.n_steps):
    t0 = time.perf_counter()
    batch = [docs[i] for i in batches[step]]
    n_tok = sum(len(g) for _, g in batch)
    packs, pst = plan_packs([(p, g, loss_scale / n_tok) for p, g in batch])
    loss_sum = 0.0
    for pk in packs:
        args = [torch.from_numpy(pk[k]).to(device, non_blocking=True)
                for k in ("idx", "pos", "cu", "sel", "tgt", "w")]
        loss_sum += float(fb(*args, pk["max_seg"]))
    gnorm = grad_global_norm()
    if math.isfinite(gnorm):
        optimizer_step()                     # zeroes the grads as it steps
        ok = 1
    else:
        ok = 0
        print(f"  !! non-finite gradient at loss scale {loss_scale:g} -- step skipped, "
              f"scale -> {loss_scale / 2:g}", flush=True)
        loss_scale /= 2.0
        inv_scale.fill_(1.0 / loss_scale)
        zero_grads()
    torch.cuda.synchronize()
    row = dict(step=step, loss=loss_sum / loss_scale, grad_norm=gnorm / loss_scale,
               lr=float(sched.lr_host[min(max(0, opt_t - 1), sched.num_steps - 1)]),
               n_docs=len(batch), n_tok=n_tok, n_packs=pst["n_packs"],
               pad_pct=100.0 * pst["pad_tokens"] / pst["cap_tokens"], step_ok=ok,
               s=time.perf_counter() - t0)
    part1_log.append(row)
    print(f"  [{step:2d}/{cfg.n_steps}] loss {row['loss']:.4f} | grad norm {row['grad_norm']:7.3f} | "
          f"lr {row['lr']:.1e} | {row['n_docs']} docs {row['n_tok']:5d} tok in {row['n_packs']} pack(s) "
          f"(pad {row['pad_pct']:4.1f}%) | {row['s']:6.2f}s"
          + ("" if ok else " | SKIPPED"), flush=True)

torch.cuda.synchronize()
print(f"== Part 1 done: {opt_t} optimizer steps taken | loss {part1_log[0]['loss']:.4f} -> "
      f"{part1_log[-1]['loss']:.4f} | peak mem {torch.cuda.max_memory_reserved() / 2**30:.1f} GB ==",
      flush=True)
torch.cuda.empty_cache()


# ==============================================================================
# Part 2: one pack, spelled out
# ==============================================================================


# ------------------------------------------------------------------------------
# Visualizers
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import torch

def show_heatmap(tensor, title=None, figsize=(10, 8)):
    """Displays a 2D PyTorch tensor as a heatmap."""
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
        raise ValueError(f"Expected a 2D PyTorch tensor, got shape {getattr(tensor, 'shape', type(tensor))}")

    # Move to CPU, ensure it's float, and convert to numpy
    data = tensor.detach().cpu().float().numpy()

    plt.figure(figsize=figsize)
    # interpolation='none' prevents smoothing between pixels.
    # origin='upper' (default) keeps the tensor's native row/col orientation.
    plt.imshow(data, interpolation='none', aspect='auto', cmap='viridis')
    plt.colorbar()

    if title:
        plt.title(title)

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 0")
    plt.show()


# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

packs2, _ = plan_packs([(p, g, 1.0) for p, g in docs[:cfg.part2_docs]])
pk = packs2[0]                     # one pack is all we need; any spill is unused

idx = torch.from_numpy(pk["idx"]).to(device).long()   # int64: we index with it
pos = torch.from_numpy(pk["pos"]).to(device)
cu = torch.from_numpy(pk["cu"]).to(device)            # int32 -- what the kernel wants
sel = torch.from_numpy(pk["sel"]).to(device)
tgt_sel = torch.from_numpy(pk["tgt"]).to(device)
w_sel = torch.from_numpy(pk["w"]).to(device)
T, max_seg = int(idx.numel()), int(pk["max_seg"])

# plan_packs took a per-doc weight of 1.0, so w_sel is 1 on a real completion
# target and 0 on the padding. Rescale to the token-mean weight Part 1 uses:
# 1/N per target (N = targets in this pack), times the fp16 loss scale.
n_targets = int((w_sel > 0).sum())
w_sel *= loss_scale / n_targets

# plan_packs hands the lm_head a GATHERED list of the completion positions
# (`sel`) so the head runs only where the loss lives. Part 2 runs the head over
# every position instead -- simpler to read, affordable at this T -- so scatter
# the targets and weights back out to full length. The unused rows of `sel` are
# padding that points at position 0 with weight 0, and position 0 is never a
# real completion target (every doc starts with a prompt), so those writes are
# zeros landing on a zero.
tgt = torch.zeros(T, dtype=torch.int64, device=device)
w = torch.zeros(T, dtype=torch.float32, device=device)
tgt[sel] = tgt_sel
w[sel] = w_sel

torch.cuda.reset_peak_memory_stats()
peak_gb = lambda: (f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB alloc / "
                   f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved")
print(f"\n== Part 2: a pack of {pk['n_docs']} docs, T = {T} tokens, {n_targets} completion "
      f"targets, loss scale {loss_scale:g} | {torch.cuda.memory_reserved() / 2**30:.1f} GB "
      f"resident (weights + optimizer state) ==", flush=True)
print(f"  cu = {pk['cu'].tolist()}")
print(f"  e.g. {doc_text[0][0][-60:]!r} -> {doc_text[0][1]!r}")

# Rotary tables for these positions: (T, 1, 32), broadcasting over heads.
cos = m.cos[pos].unsqueeze(1)
sin = m.sin[pos].unsqueeze(1)

Hq, Hkv, Dh, D = cfg.n_qo_heads, cfg.n_kv_heads, cfg.d_head, cfg.d_model
L = cfg.n_layers - 1          # the layer we flatten below
rms_eps = cfg.rms_eps
ar_T = torch.arange(T, device=device)

# Calculate T manually for the first 8 docs without needing plan_packs
part2_docs_slice = docs[:8]

# Sum of (prompt + completion - 1) for each doc
total_tokens = sum(len(p) + len(g) - 1 for p, g in part2_docs_slice)

pack_quantum = 256
train_t = 2048

# Round up to the nearest pack_quantum (256)
T_val = min(train_t, -(-total_tokens // pack_quantum) * pack_quantum)

print(f"Total actual tokens in the first 8 docs: {total_tokens}")
print(f"T (rounded up to the nearest {pack_quantum}): {T_val}")

t0 = time.perf_counter()
x = m.embed[idx]                     # (T, D) fp16 -- the input embedding lookup
stash = []
for i in range(L):
    xb_hat, xb_r = _rms_fwd(x)
    xbn = xb_hat * m.attn_norm[i]
    qkv_i = xbn @ m.W_QKV[i].mT + m.b_QKV[i]
    q_i = qkv_i[:, :cfg.d_q].view(T, Hq, Dh)
    k_i = qkv_i[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(T, Hkv, Dh)
    v_i = qkv_i[:, cfg.d_q + cfg.d_kv:].view(T, Hkv, Dh).contiguous()
    q1, q2 = q_i[..., :cfg.half], q_i[..., cfg.half:]
    k1, k2 = k_i[..., :cfg.half], k_i[..., cfg.half:]
    q_i = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_i = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    y_i, lse_i = attn_varlen_fwd(q_i, k_i, v_i, cu, max_seg)
    y_i = y_i.contiguous()
    xm_i = x + y_i.view(T, -1) @ m.W_O[i].mT
    xm_hat, _ = _rms_fwd(xm_i)
    gu_i = (xm_hat * m.mlp_norm[i]) @ m.W_GU[i].mT
    g_i, u_i = gu_i[:, :cfg.d_mlp], gu_i[:, cfg.d_mlp:]
    x = xm_i + (F.silu(g_i) * u_i) @ m.W_down[i].mT
    # Same stash Part 1 keeps: xm and gu are held, xm_hat / xmn / silu(g) * u
    # are recomputed in the backward rather than stored (they are the big ones).
    stash.append(LayerStash(xb_hat=xb_hat, xb_inv_rms=xb_r, q=q_i, k=k_i, v=v_i,
                            y=y_i, lse=lse_i, xm=xm_i, gu=gu_i))
# `x` is now the residual stream entering the last layer.


# ------------------------------------------------------------------------------
# Forward, Step-by-Step
# ------------------------------------------------------------------------------

# ======== Pre-norm ========

# The model primarily operates in 16-bit, but 32-bit precision is important here
# (TODO - why?)
x_f32 = x.float()  # (T, D)

# Calculate 1/rms using the root-square-root function `.rsqrt()`.
inv_rms1 = (x_f32.square().mean(dim=-1, keepdim=True) + rms_eps).rsqrt()   # (T, 1)  1 / rms(x)

# Normalize each stream by its own rms.
x_hat = (x_f32 * inv_rms1).half()  # (T, D)  x / rms(x)

xn = x_hat * m.attn_norm[L]  # (T, D)  weighted

# ======== QKV and RoPE ========

qkv = xn @ m.W_QKV[L].mT + m.b_QKV[L]                              # (T, 1152)
q = qkv[:, :cfg.d_q].view(T, Hq, Dh)                               # (T, 14, 64)
k = qkv[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(T, Hkv, Dh)            # (T, 2, 64)
v = qkv[:, cfg.d_q + cfg.d_kv:].view(T, Hkv, Dh).contiguous()      # (T, 2, 64)

# Rotary position embedding (rotate_half convention): channel j is paired with
# channel j + 32 and the pair is rotated by the position-dependent angle.
q1, q2 = q[..., :cfg.half], q[..., cfg.half:]
k1, k2 = k[..., :cfg.half], k[..., cfg.half:]
q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)   # (T, 14, 64)
k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)   # (T, 2, 64)

# ======== Attention ========

y_heads, lse = attn_varlen_fwd(q_rot, k_rot, v, cu, max_seg)       # (T, 14, 64), lse
y_heads = y_heads.contiguous()
y = y_heads.view(T, Hq * Dh)                                       # (T, 896) heads side by side

# ======== Output projection ========

xm = x + y @ m.W_O[L].mT                                           # (T, D) post-attention residual stream

# ======== MLP ========

xm_f32 = xm.float()
inv_rms2 = (xm_f32.square().mean(dim=-1, keepdim=True) + rms_eps).rsqrt()   # (T, 1)
xm_hat = (xm_f32 * inv_rms2).half()                                # (T, D)
xmn = xm_hat * m.mlp_norm[L]                                       # (T, D)

gu = xmn @ m.W_GU[L].mT                                            # (T, 9728)
g, u = gu[:, :cfg.d_mlp], gu[:, cfg.d_mlp:]                        # (T, 4864) each
sg = torch.sigmoid(g)                                              # (T, 4864)
silu_g = g * sg                                                    # silu(g) = g * sigmoid(g)
a = silu_g * u                                                     # (T, 4864)
x_out = xm + a @ m.W_down[L].mT                                    # (T, D) leaving the last layer

# ======== LM head and loss ========

xo_f32 = x_out.float()
inv_rms_f = (xo_f32.square().mean(dim=-1, keepdim=True) + rms_eps).rsqrt()   # (T, 1)
xf_hat = (xo_f32 * inv_rms_f).half()                               # (T, D)
xfn = xf_hat * m.final_norm                                        # (T, D)

logits = (xfn @ m.embed.mT).float()                                # (T, V) fp32
lse_ce = torch.logsumexp(logits, dim=-1)                           # (T,)   log sum_v exp(logit_v)
logit_y = logits.gather(1, tgt.unsqueeze(1)).squeeze(1)            # (T,)   the target's logit
ce = lse_ce - logit_y                                              # (T,)   -log p(target) per position
loss = (w * ce).sum()                                              # scalar; w carries 1/N and the loss scale
torch.cuda.synchronize()
fwd_s = time.perf_counter() - t0
print(f"  forward: loss (mean CE over {n_targets} completion tokens) = {float(loss) / loss_scale:.4f} "
      f"| {fwd_s:.2f}s | peak {peak_gb()}", flush=True)

# What the model was asked to predict, and how surprised it was (CE in nats;
# p = exp(-CE)), for the first stretch of trained positions in the pack.
_rows = torch.nonzero(w > 0).squeeze(1)[:24].tolist()
print("  completion targets:          input -> target      CE     p(target)")
for _t in _rows:
    print(f"    {decode([int(idx[_t])])!r:>14} -> {decode([int(tgt[_t])])!r:<14} "
          f"{float(ce[_t]):6.3f}   {math.exp(-float(ce[_t])):.3f}")


# ------------------------------------------------------------------------------
# Backward
# ------------------------------------------------------------------------------

# ======== Cross-entropy and LM head ========

t0 = time.perf_counter()
probs = torch.softmax(logits, dim=-1)                              # (T, V) fp32
dlogits = probs                                                    # reuse the buffer
dlogits[ar_T, tgt] -= 1.0                                          # p - onehot(target)
dlogits *= w.unsqueeze(1)                                          # times the per-position weight (0 off-target)
dlogits = dlogits.half()                                           # (T, V) fp16 for the GEMMs
m.embed.gacc.addmm_(dlogits.mT, xfn)                               # d embed (as head) += dlogits^T @ xfn   (V, D)
dxfn = dlogits @ m.embed                                           # (T, D) d loss / d xfn
del probs, dlogits, logits

# ======== Final norm ========

m.final_norm.gacc.add_((dxfn.float() * xf_hat.float()).sum(dim=0))          # (D,)
d_hat = (dxfn * m.final_norm).float()                                        # (T, D)
xf_hat32 = xf_hat.float()
dx_out = (inv_rms_f * (d_hat - xf_hat32 * (xf_hat32 * d_hat).mean(dim=-1, keepdim=True))).half()   # (T, D)

# ======== MLP ========

m.W_down.grad_slices[L].addmm_(dx_out.mT, a)                       # dW_down += dx_out^T @ a      (896, 4864)
da = dx_out @ m.W_down[L]                                          # (T, 4864)
du = da * silu_g                                                   # d/du of silu(g)*u
dg = da * u * (sg * (1 + g * (1 - sg)))                            # d silu(g)/dg = sig(g) * (1 + g * (1 - sig(g)))
dgu = torch.cat([dg, du], dim=1)                                   # (T, 9728) -- same layout as gu
m.W_GU.grad_slices[L].addmm_(dgu.mT, xmn)                          # dW_GU += dgu^T @ xmn        (9728, 896)
dxmn = dgu @ m.W_GU[L]                                             # (T, D)

m.mlp_norm.grad_slices[L].add_((dxmn.float() * xm_hat.float()).sum(dim=0))
d_hat = (dxmn * m.mlp_norm[L]).float()
xm_hat32 = xm_hat.float()
dxm = dx_out + (inv_rms2 * (d_hat - xm_hat32 * (xm_hat32 * d_hat).mean(dim=-1, keepdim=True))).half()   # residual + norm path

# ======== Output projection ========

m.W_O.grad_slices[L].addmm_(dxm.mT, y)                             # dW_O += dxm^T @ y          (896, 896)
dy = dxm @ m.W_O[L]                                                # (T, 896)

# ======== Attention ========

dq_rot, dk_rot, dv = attn_varlen_bwd(dy.view(T, Hq, Dh), q_rot, k_rot, v,
                                     y_heads, lse, cu, max_seg)    # (T,14,64), (T,2,64), (T,2,64)

# ======== RoPE and QKV ========

dq1, dq2 = dq_rot[..., :cfg.half], dq_rot[..., cfg.half:]
dk1, dk2 = dk_rot[..., :cfg.half], dk_rot[..., cfg.half:]
dq = torch.cat([dq1 * cos + dq2 * sin, dq2 * cos - dq1 * sin], dim=-1)   # (T, 14, 64)
dk = torch.cat([dk1 * cos + dk2 * sin, dk2 * cos - dk1 * sin], dim=-1)   # (T, 2, 64)

# Back through the fused projection: dqkv has the same [Q | K | V] layout as qkv.
dqkv = torch.cat([dq.reshape(T, cfg.d_q), dk.reshape(T, cfg.d_kv), dv.reshape(T, cfg.d_kv)], dim=1)   # (T, 1152)
m.b_QKV.grad_slices[L].add_(dqkv.sum(dim=0, dtype=torch.float32))  # db_QKV += sum_t dqkv          (1152,)
m.W_QKV.grad_slices[L].addmm_(dqkv.mT, xn)                         # dW_QKV += dqkv^T @ xn         (1152, 896)
dxn = dqkv @ m.W_QKV[L]                                            # (T, D)

m.attn_norm.grad_slices[L].add_((dxn.float() * x_hat.float()).sum(dim=0))
d_hat = (dxn * m.attn_norm[L]).float()
x_hat32 = x_hat.float()
dx = dxm + (inv_rms1 * (d_hat - x_hat32 * (x_hat32 * d_hat).mean(dim=-1, keepdim=True))).half()   # (T, D)
# `dx` is now the gradient w.r.t. the residual stream ENTERING the last layer.

# ======== Layers 22-0 ========

for i in reversed(range(L)):
    st = stash[i]
    # MLP: xm_hat / xmn / silu(g)*u were not stashed, so recompute them here.
    xm_hat_i, xm_r_i = _rms_fwd(st.xm)
    xmn_i = xm_hat_i * m.mlp_norm[i]
    g_i, u_i = st.gu[:, :cfg.d_mlp], st.gu[:, cfg.d_mlp:]
    sg_i = torch.sigmoid(g_i)
    silu_i = g_i * sg_i
    m.W_down.grad_slices[i].addmm_(dx.mT, silu_i * u_i)
    da_i = dx @ m.W_down[i]
    du_i = da_i * silu_i
    dg_i = da_i * u_i * (sg_i * (1 + g_i * (1 - sg_i)))
    dgu_i = torch.cat([dg_i, du_i], dim=1)
    m.W_GU.grad_slices[i].addmm_(dgu_i.mT, xmn_i)
    dxmn_i = dgu_i @ m.W_GU[i]
    m.mlp_norm.grad_slices[i].add_((dxmn_i.float() * xm_hat_i.float()).sum(dim=0))
    dxm_i = dx + _rms_bwd(dxmn_i * m.mlp_norm[i], xm_hat_i, xm_r_i)
    # Attention
    xbn_i = st.xb_hat * m.attn_norm[i]
    m.W_O.grad_slices[i].addmm_(dxm_i.mT, st.y.view(T, -1))
    dy_i = (dxm_i @ m.W_O[i]).view(T, Hq, Dh)
    dq_i, dk_i, dv_i = attn_varlen_bwd(dy_i, st.q, st.k, st.v, st.y, st.lse, cu, max_seg)
    dq1_i, dq2_i = dq_i[..., :cfg.half], dq_i[..., cfg.half:]
    dk1_i, dk2_i = dk_i[..., :cfg.half], dk_i[..., cfg.half:]
    dq_i = torch.cat([dq1_i * cos + dq2_i * sin, dq2_i * cos - dq1_i * sin], dim=-1)
    dk_i = torch.cat([dk1_i * cos + dk2_i * sin, dk2_i * cos - dk1_i * sin], dim=-1)
    dqkv_i = torch.cat([dq_i.reshape(T, cfg.d_q), dk_i.reshape(T, cfg.d_kv),
                        dv_i.reshape(T, cfg.d_kv)], dim=1)
    m.b_QKV.grad_slices[i].add_(dqkv_i.sum(dim=0, dtype=torch.float32))
    m.W_QKV.grad_slices[i].addmm_(dqkv_i.mT, xbn_i)
    dxn_i = dqkv_i @ m.W_QKV[i]
    m.attn_norm.grad_slices[i].add_((dxn_i.float() * st.xb_hat.float()).sum(dim=0))
    dx = dxm_i + _rms_bwd(dxn_i * m.attn_norm[i], st.xb_hat, st.xb_inv_rms)
    stash[i] = None                              # free as we go

# ======== Embedding ========

m.embed.gacc.index_add_(0, idx, dx)
torch.cuda.synchronize()
bwd_s = time.perf_counter() - t0
gsq = sum(float(p.gacc.float().square().sum()) for p in m)
print(f"  backward: grad norm {math.sqrt(gsq) / loss_scale:.4f} | {bwd_s:.2f}s | peak {peak_gb()}",
      flush=True)


# ------------------------------------------------------------------------------
# Parity check
# ------------------------------------------------------------------------------

mine = {n: getattr(m, n).gacc.clone() for n in Model.weight_names}
zero_grads()
ref_loss = forward_backward(idx, pos, cu, sel, tgt_sel, w_sel, max_seg)
torch.cuda.synchronize()
print(f"  reference loss {float(ref_loss) / loss_scale:.4f} vs ours {float(loss) / loss_scale:.4f}")
print(f"  {'tensor':<12} {'rel L2 err':>10} {'cosine':>8}")
worst = 0.0
for n in Model.weight_names:
    a_ref, b_ref = mine[n].float(), getattr(m, n).gacc.float()     # fp32, one tensor at a time
    rel = float((a_ref - b_ref).norm() / b_ref.norm())              # ||ours - ref|| / ||ref||
    cosv = float(torch.dot(a_ref.flatten(), b_ref.flatten()) / (a_ref.norm() * b_ref.norm()))
    del a_ref, b_ref
    worst = max(worst, rel)
    print(f"  {n:<12} {rel:10.2e} {cosv:8.5f}")
print(f"  peak {peak_gb()}")
assert worst < 5e-2, f"hand-rolled gradient disagrees with the reference (rel err {worst:.3g})"
# Put OUR gradients back so the optimizer step below consumes them.
for n in Model.weight_names:
    getattr(m, n).gacc.copy_(mine[n])
del mine


# ------------------------------------------------------------------------------
# One AdamW step
# ------------------------------------------------------------------------------

lr, wd, b1, b2, adam_eps = cfg.lr, cfg.weight_decay, cfg.beta1, cfg.beta2, cfg.adam_eps
t = opt_t + 1                             # continues Part 1's count (past warmup, so lr is just cfg.lr)
p = m.W_down
t0 = time.perf_counter()

g_p = p.gacc.float() / loss_scale                      # unscale; fp32 from here on
p.master.mul_(1 - lr * wd)                             # decoupled weight decay
p.exp_avg.mul_(b1).add_(g_p, alpha=1 - b1)             # m_t
p.exp_avg_sq.mul_(b2).addcmul_(g_p, g_p, value=1 - b2) # v_t
del g_p
m_hat = p.exp_avg / (1 - b1 ** t)                      # bias-corrected first moment
denom = (p.exp_avg_sq / (1 - b2 ** t)).sqrt_() + adam_eps   # sqrt(bias-corrected second moment) + eps
p.master.addcdiv_(m_hat, denom, value=-lr)             # master -= lr * m_hat / denom
del m_hat, denom

before = p.clone()
p.copy_(p.master)                     # fp32 master -> fp16 live weight (round-to-nearest)
p.gacc.zero_()
n_changed = int((p != before).sum())
del before
torch.cuda.synchronize()
print(f"  AdamW step {t} on {p.pname} {tuple(p.shape)}: {time.perf_counter() - t0:.2f}s | peak {peak_gb()} | "
      f"{n_changed:,} of {p.numel():,} live fp16 weights changed bits ({100 * n_changed / p.numel():.1f}%) "
      f"-- every fp32 master moved, but at lr {lr:g} most updates round away in fp16", flush=True)


# ------------------------------------------------------------------------------
# After
# ------------------------------------------------------------------------------

x2 = m.embed[idx]
for i in range(cfg.n_layers):
    xb2, _ = _rms_fwd(x2)
    qkv2 = (xb2 * m.attn_norm[i]) @ m.W_QKV[i].mT + m.b_QKV[i]
    q2 = qkv2[:, :cfg.d_q].view(T, Hq, Dh)
    k2 = qkv2[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(T, Hkv, Dh)
    v2 = qkv2[:, cfg.d_q + cfg.d_kv:].view(T, Hkv, Dh).contiguous()
    q2a, q2b = q2[..., :cfg.half], q2[..., cfg.half:]
    k2a, k2b = k2[..., :cfg.half], k2[..., cfg.half:]
    q2 = torch.cat([q2a * cos - q2b * sin, q2b * cos + q2a * sin], dim=-1)
    k2 = torch.cat([k2a * cos - k2b * sin, k2b * cos + k2a * sin], dim=-1)
    y2, _ = attn_varlen_fwd(q2, k2, v2, cu, max_seg)
    xm2 = x2 + y2.contiguous().view(T, -1) @ m.W_O[i].mT
    xmh2, _ = _rms_fwd(xm2)
    gu2 = (xmh2 * m.mlp_norm[i]) @ m.W_GU[i].mT
    g2, u2 = gu2[:, :cfg.d_mlp], gu2[:, cfg.d_mlp:]
    x2 = xm2 + (F.silu(g2) * u2) @ m.W_down[i].mT
xf2, _ = _rms_fwd(x2)
lg2 = ((xf2 * m.final_norm) @ m.embed.mT).float()
ce2 = torch.logsumexp(lg2, dim=-1) - lg2.gather(1, tgt.unsqueeze(1)).squeeze(1)
loss2 = float((w * ce2).sum()) / loss_scale
del x2, xf2, lg2, ce2
l1 = float(loss) / loss_scale
print(f"  loss on this pack: {l1:.6f} before the step -> {loss2:.6f} after "
      f"({loss2 - l1:+.6f})")
print(f"\n== done | Part 2 peak {peak_gb()} ==", flush=True)
