import os
import sys
import time as _time
run_wall_t0 = _time.perf_counter()
del _time

with open(sys.argv[0], 'r') as f:
    code = f.read()

import copy
import glob
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import gc
import wandb

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

torch.empty(1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True).backward()

from kernels import get_kernel

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Distributed training setup
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = 8 // world_size
grad_scale = 1 / grad_accum_steps
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0)

COMPUTE_DTYPE = torch.bfloat16

# -----------------------------------------------------------------------------
# Wandb helpers

class DummyWandb:
    """No-op wandb replacement when logging is disabled."""
    def log(self, *args, **kwargs): pass
    def finish(self): pass

def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()
    _PEAK_FLOPS_TABLE = (
        (["gb200"], 2.5e15), (["grace blackwell"], 2.5e15), (["b200"], 2.25e15), (["b100"], 1.8e15),
        (["h200", "nvl"], 836e12), (["h200", "pcie"], 836e12), (["h200"], 989e12),
        (["h100", "nvl"], 835e12), (["h100", "pcie"], 756e12), (["h100"], 989e12),
        (["h800", "nvl"], 989e12), (["h800"], 756e12),
        (["a100"], 312e12), (["a800"], 312e12), (["a40"], 149.7e12), (["a30"], 165e12),
        (["l40s"], 362e12), (["l40-s"], 362e12), (["l40 s"], 362e12), (["l4"], 121e12),
        (["mi355"], 2.5e15), (["mi325"], 1.3074e15), (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12), (["mi250x"], 383e12), (["mi250"], 362.1e12),
        (["5090"], 209.5e12), (["4090"], 165.2e12), (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    return float('inf')

_cc_major, _ = torch.cuda.get_device_capability()
if _cc_major < 9:  # pre-Hopper (Ampere sm80, Ada sm89)
    flash_attn_interface = get_kernel("kernels-community/flash-attn2").flash_attn_interface
else:  # Hopper sm90, Blackwell sm100+
    flash_attn_interface = get_kernel("kernels-community/flash-attn3").flash_attn_interface

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

def print0(*args, console=False, **kwargs):
    if master_process:
        print(*args, **kwargs)

# -----------------------------------------------------------------------------
# optim.py

"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill all the 0-D tensors with current values
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)

# -----------------------------------------------------------------------------
# gpt.py


"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Max total tokens in a single forward pass (packed batch). Determines RoPE cache size.
    # 0 = auto: sequence_len * 10 (only safe when packed batches are small)
    max_packed_tokens: int = 0


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, seqlens, max_seq_len):
        B, T, C = x.size()  # B=1 for varlen packed sequences

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.15  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.15

        # Flash Attention 3 with varlen packed sequences
        # q[0]/k[0]/v[0] squeeze the B=1 dim to get (T, H, D) for flash_attn_varlen_func
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        y = flash_attn_interface.flash_attn_varlen_func(
            q[0], k[0], v[0],
            cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
            max_seqlen_q=max_seq_len, max_seqlen_k=max_seq_len,
            causal=True, window_size=window_size)

        # Re-assemble the heads and project back to residual stream
        y = y.view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, seqlens, max_seq_len):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, seqlens, max_seq_len)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        self.bigram_lambdas = nn.Parameter(0.1 * torch.ones(config.n_layer))  # fake init, real init in init_weights()
        # Bigram embedding (bigram_vocab_size defined in Hyperparameters, resolved at instantiation time)
        self.bigram_embed = nn.Embedding(args.bigram_vocab_size, config.n_embd)
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # Rotary cache must cover the largest packed-batch token count seen in any forward pass.
        self.rotary_seq_len = config.max_packed_tokens if config.max_packed_tokens > 0 else config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.5, s * 0.5)  # 0.5x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding
        # Bigram embedding and lambdas
        self.bigram_lambdas.fill_(0.1)
        torch.nn.init.zeros_(self.bigram_embed.weight)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)
            self.bigram_embed.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 3 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.bigram_embed.weight.numel() +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        bigram_embed = self.bigram_embed.weight.numel()
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.bigram_lambdas.numel()
        total = wte + value_embeds + bigram_embed + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp = dist.is_initialized()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        bigram_lambdas_params = [self.bigram_lambdas]
        bigram_embed_params = list(self.bigram_embed.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(bigram_lambdas_params) + len(bigram_embed_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=bigram_lambdas_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=bigram_embed_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, input_seq, target_seq=None, seqlens=None, bigram_input_seq=None, loss_reduction='mean'):
        assert input_seq.ndim == 1  # 1D packed sequence: (total_tokens,)
        T = input_seq.size(0)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        max_seq_len = self.config.sequence_len  # upper bound for flash_attn_varlen_func

        # Forward the trunk of the Transformer
        x = self.transformer.wte(input_seq)  # (T, model_dim)
        x = norm(x[None])  # (1, T, model_dim) — add batch dim for rotary/attention compat
        x0 = x  # save initial normalized embedding for x0 residual

        # Bigram embedding
        x0_bigram = self.bigram_embed(bigram_input_seq)[None] if bigram_input_seq is not None else None

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if x0_bigram is not None:
                x = x + self.bigram_lambdas[i] * x0_bigram
            ve = self.value_embeds[str(i)](input_seq).to(x.dtype)[None] if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], seqlens, max_seq_len)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x)  # (1, T, padded_vocab_size)
        logits = logits[..., :self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        if target_seq is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: return logits (used by CORE eval)
            return logits

# -----------------------------------------------------------------------------
# Dataset download

NUM_TRAIN_SHARDS = 20
DATASET_NAME = "fineweb_edu_32k_8_370"
HF_REPO_ID = f"ChrisMcCormick/{DATASET_NAME}"
_data_path = os.environ.get("DATA_PATH", ".")
DATASET_DIR = os.path.join(_data_path, f"data/{DATASET_NAME}")
_config_path = os.path.join(DATASET_DIR, "config.json")

if not os.path.exists(_config_path):
    if master_process:
        from huggingface_hub import HfApi, hf_hub_download, login
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"=== Downloading dataset from {HF_REPO_ID} ===")
        api = HfApi()
        train_prefix = "fineweb_edu/train_"
        for fname in api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"):
            if fname.startswith(train_prefix) and int(fname[len(train_prefix):].split(".")[0]) >= NUM_TRAIN_SHARDS:
                continue
            if not os.path.exists(os.path.join(DATASET_DIR, fname)):
                hf_hub_download(repo_id=HF_REPO_ID, filename=fname, repo_type="dataset", local_dir=DATASET_DIR)
        assert os.path.exists(_config_path), "config.json missing after download"
        print("  Done.")
    dist.barrier()

# Verify chat eval data is present (fail fast rather than after hours of training)
_chat_eval_dir = os.path.join(DATASET_DIR, "chat_eval")
_chat_eval_config_path = os.path.join(_chat_eval_dir, "config.json")
assert os.path.exists(_chat_eval_config_path), (
    f"Chat eval config not found at {_chat_eval_config_path}. "
    f"Run `python data/chat_eval_dataset.py` to generate chat eval data."
)
with open(_chat_eval_config_path) as f:
    _chat_eval_config = json.load(f)
for _task_info in _chat_eval_config['tasks']:
    _pt_path = os.path.join(_chat_eval_dir, _task_info['file'])
    assert os.path.exists(_pt_path), (
        f"Chat eval data not found: {_pt_path}. "
        f"Run `python data/chat_eval_dataset.py` to generate chat eval data."
    )
del _chat_eval_config, _chat_eval_dir, _chat_eval_config_path, _task_info, _pt_path

# Load vocab config
with open(_config_path) as f:
    _vocab_config = json.load(f)
VOCAB_SIZE = _vocab_config["vocab_size"]
BOS_ID = _vocab_config["bos_id"]

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

def get_bigram_hash(x):
    """
    Computes bigram hash for each position using [prev_token, curr_token].
    Multiply by arbitary large ints to get even spread over int32 range.
    Position 0 is mapped to the reserved index (vocab_size - 1).
    BOS_tokens within the batch will hash based on last token of prior doc. Masking this ran slower and showed no improvement.
    """
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = args.bigram_vocab_size-1
    x = x.to(torch.int32).clone()
    x[0] = mod
    x[1:] = torch.bitwise_xor(rand_int_1 * x[1:], rand_int_2 * x[:-1]) % mod
    return x

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        shard = Shard(tokens, world_size)
        next_shard_getter = Shard.load_async(next(file_iter), world_size)
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                shard = next_shard_getter()
                tokens = shard.tokens
                try:
                    next_shard_getter = Shard.load_async(next(file_iter), world_size)
                except StopIteration:
                    next_shard_getter = None  # no more shards to preload
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _bigram_inputs = get_bigram_hash(_inputs)

        new_params = yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
            _bigram_inputs.to(device="cuda", non_blocking=True)
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len

# -----------------------------------------------------------------------------
# CORE Evaluation
"""
CORE evaluation using pre-tokenized benchmark data.

The CORE metric (from the DCLM paper, https://arxiv.org/abs/2406.11794) evaluates
a base model on in-context learning tasks using logit-based scoring (no generation).

Pre-tokenized .pt files are produced by data/core_dataset.py and loaded at eval time.
Sequences are packed into fixed-size 1D buffers with cu_seqlens marking boundaries,
enabling batched evaluation through the compiled varlen flash attention model.
"""

# -----------------------------------------------------------------------------
# Packed CORE evaluation: batch multiple examples into fixed-length 1D buffers

def pack_for_eval(sequences, buffer_size, bos_id):
    """
    Pack pre-tokenized sequences into fixed-size 1D buffers for batched evaluation.

    Args:
        sequences: list of (tokens, start_idx, end_idx, example_idx, seq_idx_within_example)
        buffer_size: fixed buffer size (must be multiple of 16)
        bos_id: BOS token id for padding

    Returns:
        list of dicts with keys: input_ids, cu_seqlens, bigram_hash, metadata
    """
    assert buffer_size % 16 == 0
    # CORE eval sequences can be short (~50-200 tokens), so allow many more per buffer
    # than training's //300 estimate. Use //8 for generous headroom (memory is negligible).
    max_num_seqs = next_multiple_of_n(buffer_size // 8, n=128)

    buffers = []
    cur_tokens = []
    cur_cu = [0]
    cur_meta = []
    cur_pos = 0

    for tokens, start_idx, end_idx, example_idx, seq_idx in sequences:
        seq_len = len(tokens)
        if seq_len > buffer_size:
            continue  # should not happen after truncation

        if cur_pos + seq_len > buffer_size:
            # Finalize current buffer
            _finalize_eval_buffer(buffers, cur_tokens, cur_cu, cur_meta,
                                  buffer_size, max_num_seqs, bos_id)
            cur_tokens, cur_cu, cur_meta, cur_pos = [], [0], [], 0

        # Track answer span in global buffer coordinates
        global_start = cur_pos + start_idx
        global_end = cur_pos + end_idx
        cur_meta.append((example_idx, seq_idx, global_start, global_end))
        cur_tokens.extend(tokens)
        cur_pos += seq_len
        cur_cu.append(cur_pos)

    if cur_tokens:
        _finalize_eval_buffer(buffers, cur_tokens, cur_cu, cur_meta,
                              buffer_size, max_num_seqs, bos_id)

    return buffers


def _finalize_eval_buffer(buffers, cur_tokens, cur_cu, cur_meta,
                          buffer_size, max_num_seqs, bos_id):
    """Pad and finalize a packed eval buffer."""
    total_packed = len(cur_tokens)
    pad_count = buffer_size - total_packed

    # Input tokens: packed sequences + BOS padding
    input_ids = torch.full((buffer_size,), bos_id, dtype=torch.int32)
    input_ids[:total_packed] = torch.tensor(cur_tokens, dtype=torch.int32)

    # cu_seqlens: [0, end1, end2, ..., total_packed, buffer_size, buffer_size, ...]
    if pad_count > 0:
        cur_cu.append(buffer_size)  # ghost sequence for padding region
    cu_seqlens = torch.full((max_num_seqs,), buffer_size, dtype=torch.int32)
    cu_seqlens[:len(cur_cu)] = torch.tensor(cur_cu, dtype=torch.int32)

    bigram_hash = get_bigram_hash(input_ids)

    buffers.append({
        'input_ids': input_ids,
        'cu_seqlens': cu_seqlens,
        'bigram_hash': bigram_hash,
        'metadata': cur_meta,
    })


@torch.no_grad()
def forward_eval_packed(model, input_ids, cu_seqlens, bigram_hash):
    """
    Forward a packed 1D eval buffer through the compiled model.
    Returns logits of shape (buffer_size, vocab_size).
    """
    logits_3d = model(input_ids, target_seq=None, seqlens=cu_seqlens,
                      bigram_input_seq=bigram_hash)
    return logits_3d[0]  # (buffer_size, vocab_size)


@torch.no_grad()
def evaluate_task_packed(model, task_data, device, bos_id,
                         buffer_size=16384):
    """Evaluate one task using pre-tokenized sequences and packed batched evaluation."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    task_type = task_data['task_type']
    num_examples = task_data['num_examples']
    all_sequences = task_data['sequences']
    num_seqs_per_example = task_data['num_seqs_per_example']
    gold_labels = task_data['gold_labels']

    # Step 1: Select this rank's share of pre-tokenized sequences
    rank_examples = set(range(rank, num_examples, world_size))
    sequences = [
        (s['tokens'], s['start_idx'], s['end_idx'], s['example_idx'], s['seq_idx'])
        for s in all_sequences if s['example_idx'] in rank_examples
    ]

    # Step 2: Pack into fixed-size buffers
    packed_buffers = pack_for_eval(sequences, buffer_size, bos_id)

    # Step 3: Forward pass each buffer and collect per-sequence results
    seq_results = {}

    for buf in packed_buffers:
        input_ids = buf['input_ids'].to(device)
        cu_seqlens = buf['cu_seqlens'].to(device)
        bigram_hash = buf['bigram_hash'].to(device)

        logits = forward_eval_packed(model, input_ids, cu_seqlens, bigram_hash)

        # Per-position losses: loss[j] = -log p(input_ids[j+1] | context up to j)
        target_ids = torch.roll(input_ids.long(), shifts=-1)
        all_losses = F.cross_entropy(logits.float(), target_ids, reduction='none')
        all_predictions = logits.argmax(dim=-1)

        for example_idx, seq_idx, gs, ge in buf['metadata']:
            # Answer span [gs, ge): logits at [gs-1, ge-1) predict tokens at [gs, ge)
            seq_results[(example_idx, seq_idx)] = {
                'losses': all_losses[gs - 1 : ge - 1],
                'predictions': all_predictions[gs - 1 : ge - 1],
                'input_ids': input_ids[gs : ge].long(),
            }

    # Step 4: Evaluate per-example correctness
    correct = torch.zeros(num_examples, dtype=torch.float32, device=device)

    for idx in range(rank, num_examples, world_size):
        if task_type == 'language_modeling':
            r = seq_results[(idx, 0)]
            is_correct = torch.all(r['predictions'] == r['input_ids']).item()
        elif task_type in ['multiple_choice', 'schema']:
            mean_losses = []
            for seq_j in range(num_seqs_per_example[idx]):
                r = seq_results[(idx, seq_j)]
                mean_losses.append(r['losses'].mean().item())
            pred_idx = mean_losses.index(min(mean_losses))
            is_correct = pred_idx == gold_labels[idx]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        correct[idx] = float(is_correct)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


@torch.no_grad()
def evaluate_chat_task_packed(model, task_data, device, bos_id,
                              buffer_size=16384):
    """Evaluate one chat categorical task using packed batched evaluation.

    Unlike CORE eval (which compares losses across multiple sequences per example),
    chat eval checks single-token logits at the answer position against letter choices.
    Each sequence ends with the prompt (including <|assistant_start|>), and we check
    what the model predicts as the next token, restricted to the valid answer letters.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    all_sequences = task_data['sequences']
    num_examples = task_data['num_examples']

    # Step 1: Select this rank's share and convert to pack_for_eval format.
    # We store answer_pos as start_idx (end_idx = start_idx + 1 for tuple compat)
    # and keep letter_token_ids / gold in a side table.
    sequences = []
    example_meta = {}  # example_idx -> (letter_token_ids, gold)
    for s in all_sequences:
        idx = s['example_idx']
        if idx % world_size != rank:
            continue
        answer_pos = s['answer_pos']
        sequences.append((s['tokens'], answer_pos, answer_pos + 1, idx, 0))
        example_meta[idx] = (s['letter_token_ids'], s['gold'])

    # Step 2: Pack into fixed-size buffers (reuse CORE eval packing infrastructure)
    packed_buffers = pack_for_eval(sequences, buffer_size, bos_id)

    # Step 3: Forward pass each buffer and score
    correct = 0
    total = 0

    for buf in packed_buffers:
        input_ids = buf['input_ids'].to(device)
        cu_seqlens = buf['cu_seqlens'].to(device)
        bigram_hash = buf['bigram_hash'].to(device)

        logits = forward_eval_packed(model, input_ids, cu_seqlens, bigram_hash)

        for example_idx, seq_idx, gs, ge in buf['metadata']:
            # gs = global position of answer_pos in the buffer.
            # logits[gs] predicts the token AFTER position gs — i.e. the assistant's answer.
            # (This differs from CORE's logits[gs-1:ge-1] convention because here the
            # answer token is NOT in the sequence — we want what the model predicts next.)
            answer_logits = logits[gs]  # (vocab_size,)
            letter_ids, gold = example_meta[example_idx]
            focus_logits = answer_logits[letter_ids]  # (num_choices,)
            pred = focus_logits.argmax().item()
            correct += int(pred == gold)
            total += 1

    # Step 4: Aggregate across ranks
    if world_size > 1:
        correct_t = torch.tensor([correct], dtype=torch.long, device=device)
        total_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        correct = correct_t.item()
        total = total_t.item()

    return correct / total if total > 0 else 0.0


def evaluate_chat_categorical(model, device, bos_id):
    """
    Evaluate a chat model on categorical benchmarks (MMLU, ARC-Easy, ARC-Challenge)
    using pre-tokenized data from chat_eval_dataset.py.
    Returns dict with results, centered_results, and chatcore_metric.
    """
    data_path = os.environ.get("DATA_PATH", ".")
    chat_eval_dir = os.path.join(data_path, f"data/{DATASET_NAME}/chat_eval")
    config_path = os.path.join(chat_eval_dir, "config.json")

    assert os.path.exists(config_path), f"Chat eval config not found: {config_path}"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task_info in config['tasks']:
        torch.cuda.synchronize()
        start_time = time.time()
        label = task_info['label']

        pt_path = os.path.join(chat_eval_dir, task_info['file'])
        assert os.path.exists(pt_path), f"Chat eval data not found: {pt_path}"
        task_data = torch.load(pt_path, weights_only=False)
        print0(f"Chat eval: {label} ({task_data['num_examples']} examples)... ", console=True)

        accuracy = evaluate_chat_task_packed(model, task_data, device, bos_id)
        torch.cuda.synchronize()
        results[label] = accuracy
        random_baseline = task_data['random_baseline']
        centered_result = (accuracy - random_baseline) / (1.0 - random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s", console=True)

    chatcore_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "chatcore_metric": chatcore_metric,
    }
    return out


def evaluate_core(model, device, bos_id):
    """
    Evaluate a base model on the CORE benchmark using pre-tokenized data.
    Returns dict with results, centered_results, and core_metric.
    """
    data_path = os.environ.get("DATA_PATH", ".")
    core_eval_dir = os.path.join(data_path, f"data/{DATASET_NAME}/core_eval")
    config_path = os.path.join(core_eval_dir, "config.json")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task_info in config['tasks']:
        torch.cuda.synchronize()
        start_time = time.time()
        label = task_info['label']

        task_data = torch.load(os.path.join(core_eval_dir, task_info['file']),
                               weights_only=False)
        print0(f"Evaluating: {label} ({task_data['task_type']}, "
               f"{task_data['num_examples']} examples)... ", console=True)

        accuracy = evaluate_task_packed(model, task_data, device, bos_id)
        torch.cuda.synchronize()
        results[label] = accuracy
        random_baseline = task_data['random_baseline']
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s", console=True)

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out



# -----------------------------------------------------------------------------
# Training Management

import datetime

@dataclass
class Hyperparameters:
    # data
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, f"data/{DATASET_NAME}/fineweb_edu/train_*.bin")
    val_files: str = os.path.join(data_path, f"data/{DATASET_NAME}/fineweb_edu/val_*.bin")
    val_tokens: int = 10485760
    # batch sizes
    train_max_seq_len: int = 2048
    val_batch_size: int = 4 * 64 * 1024 * 8
    # schedule
    num_iterations: int = 4000
    # evaluation and logging
    run_id: str = f"{str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))}-d24"
    val_loss_every: int = 250
    save_checkpoint: bool = True
    # wandb logging ("dummy" disables wandb)
    wandb_run: str = f"{str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))}-d24"
    wandb_project: str = "decoderstack"
    # bigram hash embedding
    bigram_vocab_size: int = VOCAB_SIZE * 5

args = Hyperparameters()

# -----------------------------------------------------------------------------
# int main

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

print0(code)
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

# -----------------------------------------------------------------------------
# Initialize the Model (hardcoded medium: depth=24, model_dim=1536, 12 heads)

max_packed = args.val_batch_size // (grad_accum_steps * world_size)
config = GPTConfig(
    sequence_len=2048, vocab_size=VOCAB_SIZE,
    n_layer=24, n_head=12, n_kv_head=12, n_embd=1536,
    window_pattern="SSSL",
    max_packed_tokens=max_packed,
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

orig_model = model
model = torch.compile(model, dynamic=False)

# --- Compute stats for MFU and wandb config ---
num_params = sum(p.numel() for p in model.parameters())
num_flops_per_token = model.estimate_flops()
gpu_device_name = torch.cuda.get_device_name(0)
gpu_peak_flops = get_peak_flops(gpu_device_name)
print0(f"Model parameters: {num_params:,} | FLOPs/token: {num_flops_per_token:e}", console=True)
print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}", console=True)

total_batch_size = 8 * args.train_max_seq_len * 8  # 8 sequences * 2048 tokens * 8 GPUs = 131072
weight_decay_scaled = 0.28  # fixed for medium model

# --- Initialize the Optimizer ---
optimizer = model.setup_optimizer(
    unembedding_lr=0.008,
    embedding_lr=0.3,
    scalar_lr=0.5,
    matrix_lr=0.02,
    weight_decay=weight_decay_scaled,
)

# -----------------------------------------------------------------------------
# Warmup kernels (compile the model by running a few dummy steps)

warmup_t0 = time.perf_counter()
print0("Compiling model and warming up kernels...", console=True)
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizer=copy.deepcopy(optimizer.state_dict()))
train_loader = distributed_data_generator(args.train_files, total_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)

for step in [0, 1, 2]:
    model.eval()
    with torch.no_grad():
        inputs, targets, cum_seqlens, bigram_inputs = next(val_loader)
        model(inputs, targets, cum_seqlens, bigram_inputs)
    model.train()
    for idx in range(grad_accum_steps):
        inputs, targets, cum_seqlens, bigram_inputs = next(train_loader)
        (model(inputs, targets, cum_seqlens, bigram_inputs) * grad_scale).backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)

# Warmup CORE eval compiled path (eval mode, target_seq=None, fixed buffer size)
_core_buf_size = 16384
_core_max_seqs = next_multiple_of_n(_core_buf_size // 8, n=128)
model.eval()
with torch.no_grad():
    _dummy_ids = torch.zeros(_core_buf_size, dtype=torch.int32, device=device)
    _dummy_cu = torch.full((_core_max_seqs,), _core_buf_size, dtype=torch.int32, device=device)
    _dummy_cu[0] = 0
    _dummy_cu[1] = _core_buf_size
    _dummy_bigram = get_bigram_hash(_dummy_ids)
    model(_dummy_ids, target_seq=None, seqlens=_dummy_cu, bigram_input_seq=_dummy_bigram)
model.train()

# Reset state so warmup steps don't count
print0("Resetting model after warmup", console=True)
model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
optimizer.load_state_dict(initial_state["optimizer"])
del val_loader, train_loader, initial_state
model.train()
warmup_elapsed = time.perf_counter() - warmup_t0
print0(f"Warmup/compilation complete in {warmup_elapsed:.1f}s ({warmup_elapsed/60:.1f}m)", console=True)

# --- wandb logging init (after warmup since most crashes occur there) ---
use_dummy_wandb = args.wandb_run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project=args.wandb_project, name=args.wandb_run,
    config={
        "num_params": num_params,
        "num_flops_per_token": num_flops_per_token,
        "n_layer": 24, "n_head": 12, "n_embd": 1536,
        "train_steps": args.num_iterations,
        "total_batch_size": total_batch_size,
        "val_loss_every": args.val_loss_every,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
    },
)
if not use_dummy_wandb:
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
wandb_run.log({"step": 0, "timing/warmup_seconds": warmup_elapsed})

# -----------------------------------------------------------------------------
# Training and validation

with open(f"data/{DATASET_NAME}/tokenizer/token_bytes.pt", "rb") as f:
    token_bytes = torch.load(f, map_location=device)

train_steps = args.num_iterations
train_loader = distributed_data_generator(args.train_files, total_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)

# LR schedule: linear warmup, constant, linear warmdown
def get_lr_multiplier(it):
    warmup_iters = 40
    warmdown_iters = round(0.65 * train_steps)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= train_steps - warmdown_iters:
        return 1.0
    else:
        progress = (train_steps - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * 0.05

# Momentum scheduler for Muon (warms up to 0.97 over first 400 steps)
def get_muon_momentum(it):
    frac = min(it / 400, 1)
    return (1 - frac) * 0.85 + frac * 0.97

# Weight decay scheduler for Muon (cosine decay to zero)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / train_steps))

gc.collect()

training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()

for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_t0 = time.perf_counter()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens, bigram_inputs = next(val_loader)
                loss_flat = model(inputs, targets, cum_seqlens, bigram_inputs, loss_reduction='none')
                num_bytes_flat = token_bytes[targets]
                total_nats += (loss_flat * (num_bytes_flat > 0).float()).sum()
                total_bytes += num_bytes_flat.sum()
        del val_loader
        if world_size > 1:
            dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        val_bpb = total_nats.item() / (math.log(2) * total_bytes.item()) if total_bytes.item() > 0 else float('inf')
        val_elapsed = time.perf_counter() - val_t0
        print0(f"step:{step}/{train_steps} val_bpb:{val_bpb:.4f} val_time:{val_elapsed:.2f}s train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        wandb_run.log({"step": step, "val/bpb": val_bpb, "val/eval_seconds": val_elapsed, "total_training_time_ms": training_time_ms})
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizer=optimizer.state_dict())
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # --------------- CORE EVALUATION -----------------
        model.eval()
        core_eval_t0 = time.time()
        core_out = evaluate_core(model, device, bos_id=BOS_ID)
        core_eval_elapsed = time.time() - core_eval_t0
        print0(f"CORE metric: {core_out['core_metric']:.4f} | total CORE eval time: {core_eval_elapsed:.2f}s", console=True)
        for label, acc in core_out['results'].items():
            centered = core_out['centered_results'][label]
            print0(f"  {label}: accuracy={acc:.4f} centered={centered:.4f}", console=True)
        wandb_run.log({
            "step": step,
            "core_metric": core_out["core_metric"],
            **{f"core/{label}/accuracy": acc for label, acc in core_out["results"].items()},
            **{f"core/{label}/centered": c for label, c in core_out["centered_results"].items()},
            "timing/pretrain_seconds": training_time_ms / 1000,
            "timing/core_eval_seconds": core_eval_elapsed,
        })
        break

    # --------------- TRAINING SECTION -----------------
    torch.cuda.synchronize()
    step_t0 = time.perf_counter()
    for idx in range(grad_accum_steps):
        inputs, targets, cum_seqlens, bigram_inputs = next(train_loader)
        (model(inputs, targets, cum_seqlens, bigram_inputs) * grad_scale).backward()
    # Update LR, momentum, weight decay schedules
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - step_t0

    # logging
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * world_size)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    pct_done = 100 * (step + 1) / train_steps
    if step > 5:
        avg_step_ms = approx_training_time_ms / (step + 1)
        remaining_steps = train_steps - step - 1
        eta_seconds = remaining_steps * avg_step_ms / 1000
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    print0(f"step {step+1:05d}/{train_steps:05d} ({pct_done:.2f}%) | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}% | total time: {approx_training_time_ms/1000/60:.2f}m{eta_str}", console=True)
    wandb_run.log({
        "step": step,
        "train/dt": dt,
        "train/tok_per_sec": tok_per_sec,
        "train/mfu": mfu,
        "total_training_time_ms": approx_training_time_ms,
    })

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)


# ========================================================================================
#                                   SFT TRAINING                                     
# ========================================================================================
# After pre-training, we fine-tune on SFT conversation data (pre-tokenized .bin shards
# produced by data/sft_dataset.py). The model architecture and compiled graph are reused;
# we create a fresh optimizer with SFT-appropriate learning rates.
#
# Data format: identical to pre-training .bin shards (BOS-delimited conversations).
# No masking is applied — all tokens contribute to the loss (same as nanochat).
# ========================================================================================

print0("", console=True)
print0("=" * 80, console=True)
print0("                        SFT TRAINING PHASE", console=True)
print0("=" * 80, console=True)

# --- SFT Configuration ---
sft_train_files = os.path.join(_data_path, f"data/{DATASET_NAME}/sft/sft_train_*.bin")
sft_val_files = os.path.join(_data_path, f"data/{DATASET_NAME}/sft/sft_val_*.bin")
sft_total_batch_size = 524288       # total tokens per optimizer step (all ranks, all accum steps)
sft_max_seq_len = 2048              # max conversation length
sft_num_iterations = -1             # -1 = one full epoch through all shards
sft_eval_every = 150                # evaluate val bpb every N steps (0 = only at end)
sft_eval_tokens = 5 * 524288       # ~2.6M tokens for val evaluation
sft_grad_accum_steps = grad_accum_steps  # same as pre-training: 8 // world_size
sft_grad_scale = 1.0 / sft_grad_accum_steps

print0(f"SFT config: total_batch={sft_total_batch_size:,} max_seq_len={sft_max_seq_len} "
       f"grad_accum={sft_grad_accum_steps}", console=True)

# --- SFT Data Generator ---
def sft_data_generator(filename_pattern, total_batch_size, max_seq_len, ga_steps=1):
    """
    Data generator for SFT .bin shards. Single-epoch: yields data from all shards
    in order, then returns (generator exhaustion = end of epoch).

    Yields (inputs, targets, cum_seqlens, bigram_inputs) — same format as
    distributed_data_generator, compatible with the model's training forward path.

    Args:
        filename_pattern: glob pattern for .bin shard files
        total_batch_size: total tokens per optimizer step (all ranks, all accum steps)
        max_seq_len: maximum sequence length for BOS-aligned packing
        ga_steps: gradient accumulation steps (each yield = one micro-batch)
    """
    ws = dist.get_world_size() if dist.is_initialized() else 1
    rk = dist.get_rank() if dist.is_initialized() else 0
    assert total_batch_size % (ws * ga_steps) == 0
    num_tokens_local = total_batch_size // (ws * ga_steps)
    max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)

    files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No SFT shard files found for: {filename_pattern}")
    print0(f"  SFT data: {len(files)} shards from {filename_pattern}", console=True)

    file_idx = 0
    tokens = _load_data_shard(files[file_idx])
    shard = Shard(tokens, ws)
    file_idx += 1
    next_getter = Shard.load_async(files[file_idx], ws) if file_idx < len(files) else None

    while True:
        try:
            seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
        except StopIteration:
            # Current shard exhausted — try to switch to next
            if next_getter is None:
                return  # All shards consumed → end of epoch
            shard = next_getter()
            tokens = shard.tokens
            file_idx += 1
            next_getter = Shard.load_async(files[file_idx], ws) if file_idx < len(files) else None
            continue

        start_idxs = torch.tensor(seq_starts[rk])
        end_idxs = torch.tensor(seq_ends[rk])
        buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
        _inputs = buf[:-1]
        _targets = buf[1:]
        end_idxs[-1] -= 1  # last doc was too long to account for _targets offset
        cum_lengths = (end_idxs - start_idxs).cumsum(0)

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _bigram_inputs = get_bigram_hash(_inputs)

        yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
            _bigram_inputs.to(device="cuda", non_blocking=True),
        )

# --- SFT Validation ---
def evaluate_sft_bpb():
    """Evaluate BPB on SFT validation shards (same method as pre-training val eval)."""
    model.eval()
    val_steps = sft_grad_accum_steps * sft_eval_tokens // sft_total_batch_size
    val_loader = distributed_data_generator(
        sft_val_files, sft_total_batch_size, -1,
        grad_accum_steps=sft_grad_accum_steps, align_to_bos=False,
    )
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets, cum_seqlens, bigram_inputs = next(val_loader)
            loss_flat = model(inputs, targets, cum_seqlens, bigram_inputs,
                              loss_reduction='none')
            num_bytes_flat = token_bytes[targets]
            total_nats += (loss_flat * (num_bytes_flat > 0).float()).sum()
            total_bytes += num_bytes_flat.sum()
    del val_loader
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    bpb = total_nats.item() / (math.log(2) * total_bytes.item()) if total_bytes.item() > 0 else float('inf')
    model.train()
    return bpb

# --- Check for SFT data ---
_sft_shard_files = sorted(glob.glob(sft_train_files))
assert _sft_shard_files, f"No SFT training shards found at {sft_train_files}"
# print0(f"No SFT training shards found at {sft_train_files} — skipping SFT phase.", console=True)
# print0("Run `python data/sft_dataset.py` to generate SFT data.", console=True)

# Exact step count from a prior run (BOS-aligned packing means raw token count / batch_size
# overestimates by ~2x, so we hardcode the observed value instead).
# NOTE: this is calibrated for dataset='{DATASET_NAME}' with sft_total_batch_size={sft_total_batch_size:,}.
#       If either changes, re-run once with sft_estimated_steps = -1 and note the final step count.
_sft_known_steps = 838  # observed on dataset='fineweb_edu_32k_8_370', batch=524288
sft_estimated_steps = _sft_known_steps
if sft_num_iterations > 0:
    sft_estimated_steps = min(sft_estimated_steps, sft_num_iterations)
print0(f"SFT steps: {sft_estimated_steps} (known for dataset='{DATASET_NAME}', "
       f"batch={sft_total_batch_size:,})", console=True)

# --- Fresh Optimizer ---
# Re-create optimizer with fresh state for SFT.
# Same param group structure as pre-training (preserving per-parameter LR ratios),
# but with zero weight decay (following nanochat SFT convention).
sft_optimizer = model.setup_optimizer(
    unembedding_lr=0.008,
    embedding_lr=0.3,
    scalar_lr=0.5,
    matrix_lr=0.02,
    weight_decay=0.0,
)

# LR scheduler: 80% flat, then linear decay to 0
def sft_get_lr_multiplier(progress):
    return 1.0 if progress < 0.8 else max(0.0, 1.0 - (progress - 0.8) / 0.2)

# Muon momentum warmup: ramp from 0.85 to 0.95 over 300 steps
def sft_get_muon_momentum(step, warmup_steps=300):
    frac = min(step / warmup_steps, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95

# --- Instantiate SFT train data loader ---
sft_train_loader = sft_data_generator(
    sft_train_files, sft_total_batch_size, sft_max_seq_len, ga_steps=sft_grad_accum_steps,
)

# --- SFT Training Loop ---
model.train()
sft_step = 0
sft_last_step = False
sft_smooth_loss = 0.0
sft_ema_beta = 0.9
sft_min_val_bpb = float('inf')
sft_progress = 0.0

# Prefetch first batch
try:
    sft_batch = next(sft_train_loader)
except StopIteration:
    print0("ERROR: SFT training data generator yielded nothing!", console=True)
    sft_batch = None

if sft_batch is not None:
    print0(f"SFT training started. First batch prefetched.", console=True)
    torch.cuda.synchronize()
    sft_wall_t0 = time.perf_counter()  # wall clock start (includes compilation warmup)

    while not sft_last_step:
        # --- Synchronize last_step across all ranks ---
        if world_size > 1:
            _ls = torch.tensor(int(sft_last_step), dtype=torch.int32, device=device)
            dist.all_reduce(_ls, op=dist.ReduceOp.MAX)
            sft_last_step = bool(_ls.item())
            if sft_last_step:
                break

        # --- Evaluation ---
        if sft_eval_every > 0 and sft_step % sft_eval_every == 0:
            val_bpb = evaluate_sft_bpb()
            sft_min_val_bpb = min(sft_min_val_bpb, val_bpb)
            elapsed = time.perf_counter() - sft_wall_t0
            print0(f"SFT step {sft_step:05d} | val_bpb: {val_bpb:.4f} | "
                    f"elapsed: {elapsed:.1f}s", console=True)
            wandb_run.log({"step": train_steps + sft_step, "sft/val_bpb": val_bpb})

        # --- Training step: accumulate gradients ---
        torch.cuda.synchronize()
        step_t0 = time.perf_counter()
        train_loss_accum = 0.0
        for micro_step in range(sft_grad_accum_steps):
            inputs, targets, cum_seqlens, bigram_inputs = sft_batch
            loss = model(inputs, targets, cum_seqlens, bigram_inputs)
            train_loss_accum += loss.detach().item()
            (loss * sft_grad_scale).backward()
            # Prefetch next batch
            try:
                sft_batch = next(sft_train_loader)
            except StopIteration:
                sft_last_step = True
                break

        # Check num_iterations limit
        if sft_num_iterations > 0 and sft_step + 1 >= sft_num_iterations:
            sft_last_step = True

        # --- LR and momentum scheduling ---
        sft_progress = min((sft_step + 1) / max(sft_estimated_steps, 1), 1.0)
        lrm = sft_get_lr_multiplier(sft_progress)
        muon_mom = sft_get_muon_momentum(sft_step)
        for group in sft_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_mom

        # --- Optimizer step ---
        sft_optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_dt = time.perf_counter() - step_t0

        sft_step += 1

        # --- Logging ---
        avg_loss = train_loss_accum / sft_grad_accum_steps
        sft_smooth_loss = sft_ema_beta * sft_smooth_loss + (1 - sft_ema_beta) * avg_loss
        debiased_loss = sft_smooth_loss / (1 - sft_ema_beta ** (sft_step + 1))
        tok_per_sec = int(sft_total_batch_size / step_dt) if step_dt > 0 else 0
        sft_flops_per_sec = num_flops_per_token * sft_total_batch_size / step_dt if step_dt > 0 else 0
        sft_mfu = 100 * sft_flops_per_sec / (gpu_peak_flops * world_size)
        pct_done = 100 * sft_progress
        sft_elapsed = time.perf_counter() - sft_wall_t0
        # ETA based on average step time
        if sft_step > 3 and sft_estimated_steps > sft_step:
            sft_eta_seconds = (sft_elapsed / sft_step) * (sft_estimated_steps - sft_step)
            sft_eta_str = f" | eta: {sft_eta_seconds/60:.1f}m"
        else:
            sft_eta_str = ""
        print0(f"SFT step {sft_step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.4f} | "
                f"lrm: {lrm:.3f} | dt: {step_dt * 1000:.0f}ms | tok/s: {tok_per_sec:,} | "
                f"mfu: {sft_mfu:.2f}% | total time: {sft_elapsed/60:.2f}m{sft_eta_str}",
                console=True)
        wandb_run.log({
            "step": train_steps + sft_step,
            "sft/loss": debiased_loss,
            "sft/lrm": lrm,
            "sft/dt": step_dt,
            "sft/tok_per_sec": tok_per_sec,
            "sft/mfu": sft_mfu,
        })

    # --- End of SFT: final evaluation ---
    sft_total_time = time.perf_counter() - sft_wall_t0
    print0("", console=True)
    print0(f"SFT training complete. {sft_step} steps in {sft_total_time:.1f}s ({sft_total_time/60:.1f}m).",
            console=True)

    # Final val BPB
    val_bpb = evaluate_sft_bpb()
    sft_min_val_bpb = min(sft_min_val_bpb, val_bpb)
    print0(f"SFT final val_bpb: {val_bpb:.4f} | min_val_bpb: {sft_min_val_bpb:.4f}", console=True)
    wandb_run.log({
        "step": train_steps + sft_step,
        "sft/final_val_bpb": val_bpb,
        "sft/min_val_bpb": sft_min_val_bpb,
        "sft/total_steps": sft_step,
        "sft/total_time_seconds": sft_total_time,
        "timing/sft_seconds": sft_total_time,
    })

    # Save checkpoint
    if master_process and args.save_checkpoint:
        sft_log = dict(
            sft_step=sft_step,
            code=code,
            model=model.state_dict(),
            sft_optimizer=sft_optimizer.state_dict(),
            sft_val_bpb=val_bpb,
            sft_min_val_bpb=sft_min_val_bpb,
        )
        os.makedirs(f"logs/{run_id}", exist_ok=True)
        torch.save(sft_log, f"logs/{run_id}/sft_state_step{sft_step:06d}.pt")
        print0(f"SFT checkpoint saved to logs/{run_id}/sft_state_step{sft_step:06d}.pt", console=True)

    print0(f"SFT peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", console=True)

# =============================================================================
# Phase 4: Chat Categorical Evaluation (MMLU, ARC-Easy, ARC-Challenge)
# =============================================================================

print0("", console=True)
print0("=" * 60, console=True)
print0("Chat Categorical Evaluation", console=True)
print0("=" * 60, console=True)

model.eval()
chat_eval_t0 = time.time()
chat_out = evaluate_chat_categorical(model, device, bos_id=BOS_ID)
chat_eval_elapsed = time.time() - chat_eval_t0

print0(f"ChatCORE metric: {chat_out['chatcore_metric']:.4f} | total chat eval time: {chat_eval_elapsed:.2f}s", console=True)
for label, acc in chat_out['results'].items():
    centered = chat_out['centered_results'][label]
    print0(f"  {label}: accuracy={acc:.4f} centered={centered:.4f}", console=True)
wandb_run.log({
    "step": train_steps + sft_step if sft_batch is not None else train_steps,
    "chat/chatcore_metric": chat_out["chatcore_metric"],
    **{f"chat/{label}/accuracy": acc for label, acc in chat_out["results"].items()},
    **{f"chat/{label}/centered": c for label, c in chat_out["centered_results"].items()},
    "timing/chat_eval_seconds": chat_eval_elapsed,
})


# ============ TEMPORARY HACK: Load checkpoint and run CORE eval only ============
if False:  # Set to False to disable this hack
    _ckpt_path = "/home/ubuntu/stacks/logs/2026-02-12_063725-4000-steps/state_step004000.pt"
    print0(f"HACK: Loading checkpoint from {_ckpt_path}", console=True)
    _ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(_ckpt["model"])
    print0(f"HACK: Loaded model state from step {_ckpt.get('step', '?')}", console=True)
    del _ckpt


if False:
    # Run CORE eval on loaded checkpoint
    core_eval_t0 = time.time()
    model.eval()
    print0("HACK: Running CORE eval", console=True)
    core_out = evaluate_core(model, device, bos_id=BOS_ID)
    core_eval_elapsed = time.time() - core_eval_t0
    print0(f"CORE metric: {core_out['core_metric']:.4f} | total CORE eval time: {core_eval_elapsed:.2f}s", console=True)
    for label, acc in core_out['results'].items():
        centered = core_out['centered_results'][label]
        print0(f"  {label}: accuracy={acc:.4f} centered={centered:.4f}", console=True)
    dist.destroy_process_group()
    sys.exit(0)
# ============ END TEMPORARY HACK ============


# =============================================================================
# Phase 5: Chat Generative Evaluation (GSM8K, HumanEval, SpellingBee)
# =============================================================================
from generation_medium import evaluate_chat_generative, set_flash_attn
set_flash_attn(flash_attn_interface)
#set_fused_mlp(FusedLinearReLUSquareFunction)

print0("", console=True)
print0("=" * 60, console=True)
print0("Chat Generative Evaluation", console=True)
print0("=" * 60, console=True)

# Run the benchmark tasks
gen_eval_t0 = time.time()
gen_out = evaluate_chat_generative(model, device, DATASET_NAME)
gen_eval_elapsed = time.time() - gen_eval_t0

print0(f"Generative ChatCORE: {gen_out['generative_chatcore_metric']:.4f} | time: {gen_eval_elapsed:.2f}s", console=True)
for label, acc in gen_out['results'].items():
    centered = gen_out['centered_results'][label]
    print0(f"  {label}: accuracy={acc:.4f} centered={centered:.4f}", console=True)

# Log generative results to wandb
wandb_run.log({
    "step": train_steps + sft_step if sft_batch is not None else train_steps,
    "chat_gen/generative_chatcore_metric": gen_out["generative_chatcore_metric"],
    **{f"chat_gen/{label}/accuracy": acc for label, acc in gen_out["results"].items()},
    **{f"chat_gen/{label}/centered": c for label, c in gen_out["centered_results"].items()},
    "timing/gen_eval_seconds": gen_eval_elapsed,
})

# Compute combined ChatCORE metric across all 6 tasks (3 categorical + 3 generative)
all_centered = {}
all_centered.update(chat_out["centered_results"])  # MMLU, ARC-Easy, ARC-Challenge
all_centered.update(gen_out["centered_results"])    # GSM8K, HumanEval, SpellingBee
combined_chatcore = sum(all_centered.values()) / len(all_centered)
print0(f"Combined ChatCORE (all 6 tasks): {combined_chatcore:.4f}", console=True)
wandb_run.log({
    "step": train_steps + sft_step if sft_batch is not None else train_steps,
    "chat/combined_chatcore_metric": combined_chatcore,
})

# --- Final timing summary ---
run_wall_total = time.perf_counter() - run_wall_t0
print0(f"Total end-to-end wall time: {run_wall_total:.1f}s ({run_wall_total/60:.1f}m)", console=True)
wandb_run.log({
    "step": train_steps + sft_step if sft_batch is not None else train_steps,
    "timing/total_seconds": run_wall_total,
})

# --- wandb cleanup ---
if not use_dummy_wandb:
    wandb.save(sys.argv[0])  # save the training script
    if logfile:
        wandb.save(logfile)
wandb_run.finish()

dist.destroy_process_group()
