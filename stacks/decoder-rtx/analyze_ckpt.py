"""What a trained bigram slot holds: a run's table at a checkpoint, against the batches
that trained it.

    python analyze_ckpt.py <run_name> <step> [cpu|cuda]     # from ~/stacks/stacks/decoder-rtx

Loads logs/<run>/model_step<step>.pt (bf16 live weights), replays the run's data order to
get every slot's hits, (prev, curr) pairs and next tokens over steps 0..step-1, and prints,
by hits per step: how far the slots moved, how many are still at zero, what the slot's
vector points at (the checkpoint's own E[curr*], E[prev*], E[next*], H[next*]; the
count-derived direction H^T (p_hat(.|slot) - p_b(.|curr))), and the common-shift and
low-rank shares. The heavy parts run on the CPU by default so it can sit beside a training
run; the data loader still needs the GPU for its (small) device tensors.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
from utils import DATASET_DIR, data_generator

run, step = sys.argv[1], int(sys.argv[2])
dev = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cpu")
V, D, S = 32768, 768, 5 * 32768
seq_len, micro_tokens, grad_accum, num_steps = 2048, 2 ** 18, 2, 1035
t0 = time.perf_counter()

ckpt = torch.load(f"logs/{run}/model_step{step:06d}.pt", map_location="cpu", weights_only=False)
assert ckpt["step"] == step
T = ckpt["weights"]["bigram_embeds"].float().to(dev)          # (S, D)
E = ckpt["weights"]["input_embeds"].float().to(dev)           # (V, D)
H = ckpt["weights"]["lm_head"].float().to(dev)                # (V, D) bf16 live; the master's low bits do not matter here
lam_b = ckpt["weights"]["bigram_lambdas"].float().tolist()
lam_x = ckpt["weights"]["x0_lambdas"].float().tolist()
del ckpt
print(f"# {run} @ step {step}: the bigram table against the batches of steps 0..{step - 1}\n")
print(f"bigram_lambdas: " + " ".join(f"{v:+.3f}" for v in lam_b))
print(f"x0_lambdas:     " + " ".join(f"{v:+.3f}" for v in lam_x))

# ---- replay the data order: hits, (prev, curr) pairs and (slot, next) per position ----
hits = np.zeros(S, dtype=np.int64)
pair_chunks, next_chunks = [], []
loader = data_generator("train", seq_len, micro_tokens, num_steps * grad_accum)
for i in range(step * grad_accum):
    inputs, targets, _ = next(loader)
    idx = inputs.cpu().numpy().astype(np.int64)
    tgt = targets.cpu().numpy().astype(np.int64)
    slots = ((36313 * idx[1:]) ^ (27191 * idx[:-1])) % (S - 1)   # forward_backward's hash, positions 1..T-1
    hits += np.bincount(slots, minlength=S)
    pair_chunks.append(idx[:-1] * V + idx[1:])
    next_chunks.append(slots * V + tgt[1:])
loader.close()
hits[S - 1] = 0                                                    # the reserved position-0 slot
num_tokens = int(hits.sum())
print(f"\nreplayed {step * grad_accum} micro-batches, {num_tokens:,} positions ({time.perf_counter() - t0:.0f}s)")

pk, pc = np.unique(np.concatenate(pair_chunks), return_counts=True); del pair_chunks
nk, nc = np.unique(np.concatenate(next_chunks), return_counts=True); del next_chunks
pk, pc, nk, nc = (torch.from_numpy(a).to(dev) for a in (pk, pc, nk, nc))
h = torch.from_numpy(hits).to(dev)
prev_p, curr_p = pk // V, pk % V
slot_p = ((36313 * curr_p) ^ (27191 * prev_p)) % (S - 1)
n_pairs = torch.bincount(slot_p, minlength=S)
max_pc = torch.zeros(S, dtype=pc.dtype, device=dev).scatter_reduce_(0, slot_p, pc, reduce="amax", include_self=False)
is_dom = pc == max_pc[slot_p]
dom_pair = torch.zeros(S, dtype=torch.int64, device=dev); dom_pair[slot_p[is_dom]] = pk[is_dom]
prev_star, curr_star = dom_pair // V, dom_pair % V
dom_share = max_pc.float() / h.clamp_min(1).float()
slot_n, next_n = nk // V, nk % V
n_nexts = torch.bincount(slot_n, minlength=S)
max_nc = torch.zeros(S, dtype=nc.dtype, device=dev).scatter_reduce_(0, slot_n, nc, reduce="amax", include_self=False)
is_domn = nc == max_nc[slot_n]
next_star = torch.zeros(S, dtype=torch.int64, device=dev); next_star[slot_n[is_domn]] = next_n[is_domn]
next_share = max_nc.float() / h.clamp_min(1).float()
print(f"pair and next-token tables: {pk.numel():,} distinct (prev, curr) pairs, {nk.numel():,} distinct (slot, next) ({time.perf_counter() - t0:.0f}s)")

# ---- the count-derived direction: H^T (p_hat(.|slot) - p_b(.|curr)) with the corpus bigram ----
bg = np.load(os.path.join(DATASET_DIR, "tokenizer/bigram_counts.npz"))
crow = torch.from_numpy(bg["train_indptr"]).to(dev)
bcol = torch.from_numpy(bg["train_indices"].astype(np.int64)).to(dev)
bval = torch.from_numpy(bg["train_data"].astype(np.float32)).to(dev)
del bg
row_tot = torch.zeros(V, device=dev).index_add_(0, torch.repeat_interleave(torch.arange(V, device=dev), crow[1:] - crow[:-1]), bval)
col_tot = torch.zeros(V, device=dev).index_add_(0, bcol, bval)
unigram = (col_tot + 0.5) / (col_tot.sum() + 0.5 * V)
CH = torch.sparse_csr_tensor(crow, bcol, bval, size=(V, V)) @ H
Hbar_b = (CH + 50.0 * (unigram @ H)) / (row_tot + 50.0)[:, None]
del CH, crow, bcol, bval
G_hat = torch.zeros(S, D, device=dev)
for s in range(0, nc.numel(), 4_000_000):
    e = s + 4_000_000
    G_hat.index_add_(0, slot_n[s:e], (nc[s:e].float() / h[slot_n[s:e]].clamp_min(1).float())[:, None] * H[next_n[s:e]])
Hbar_slot = torch.zeros(S, D, device=dev)
for s in range(0, pc.numel(), 4_000_000):
    e = s + 4_000_000
    Hbar_slot.index_add_(0, slot_p[s:e], (pc[s:e].float() / h[slot_p[s:e]].clamp_min(1).float())[:, None] * Hbar_b[curr_p[s:e]])
g_excess = G_hat - Hbar_slot
g_dom = H[next_star] - Hbar_slot
del G_hat, Hbar_slot
print(f"count-derived directions built ({time.perf_counter() - t0:.0f}s)", flush=True)

# ---- the table by hits per step ----
rate = h.float() / step
buckets = [(0, 0.01, "< 0.01 / step"), (0.01, 0.1, "0.01-0.1"), (0.1, 1, "0.1-1"), (1, 10, "1-10"), (10, 100, "10-100"), (100, 1e9, "100+")]
header = " | ".join(label for _, _, label in buckets) + " | all touched"
touched = h > 0
hN = h.float()

def by_bucket(values, weights=None, fmt="{:.3f}"):
    weights = hN if weights is None else weights
    cells = []
    for lo, hi, _ in buckets + [(0, 1e9, "all")]:
        sel = touched & (rate >= lo) & (rate < hi)
        ww = weights[sel]
        cells.append(fmt.format((values[sel] * ww).sum().item() / ww.sum().item()) if ww.numel() and ww.sum() > 0 else "-")
    return " | ".join(cells)

norms = T.norm(dim=1)
print(f"\nslots touched: {touched.sum().item():,} of {S - 1:,}; slots with a nonzero vector: {(norms > 0).sum().item():,}; "
      f"untouched slots with a nonzero vector: {((~touched) & (norms > 0)).sum().item():,} (the reserved slot aside)")
print(f"\n## the table by the slot's hits per step (hits-weighted means unless marked)\n")
print(f"| | {header} |")
print("|---|" + "---|" * (len(buckets) + 1))
print("| slots | " + " | ".join(f"{(touched & (rate >= lo) & (rate < hi)).sum().item():,}" for lo, hi, _ in buckets) + f" | {touched.sum().item():,} |")
print("| share of positions | " + " | ".join(f"{hN[touched & (rate >= lo) & (rate < hi)].sum().item() / num_tokens:.1%}" for lo, hi, _ in buckets) + " | 100% |")
print(f"| distinct (prev, curr) pairs in the slot | {by_bucket(n_pairs.float(), fmt='{:.1f}')} |")
print(f"| dominant pair's share of the slot's hits | {by_bucket(dom_share)} |")
print(f"| dominant next token's share | {by_bucket(next_share)} |")
print(f"| norm of the slot's vector | {by_bucket(norms, fmt='{:.1f}')} |")
print(f"| norm, unweighted mean | {by_bucket(norms, torch.ones_like(hN), fmt='{:.1f}')} |")
for label, X in (("E[curr*]: the current token's embedding", E[curr_star]), ("E[prev*]", E[prev_star]), ("E[next*]", E[next_star]),
                 ("H[curr*]", H[curr_star]), ("H[next*]: the dominant continuation's head row", H[next_star]),
                 ("g_excess = H^T (p_hat(.|slot) - p_b(.|curr))", g_excess), ("g_dom = H[next*] - E_b[H | curr]", g_dom)):
    print(f"| cos(T, {label}) | {by_bucket(F.cosine_similarity(T, X, dim=1))} |")
idx = touched.nonzero().squeeze(1)
Tt, hT = T[idx], hN[idx]
mu = (hT[:, None] * Tt).sum(0) / hT.sum()
share_mu = (hT.sum() * mu.square().sum() / (hT[:, None] * Tt.square()).sum()).item()
sw = hT.sqrt()[:, None]
_, sv, _ = torch.svd_lowrank(sw * Tt, q=24, niter=4)
tot = (sw * Tt).square().sum()
_, sg, _ = torch.svd_lowrank(sw * torch.randn(idx.numel(), D, device=dev), q=24, niter=4)
print(f"\ncommon shift: energy share {share_mu:.3f}, |mu| {mu.norm().item():.2f}; cos(mu, mean E row) {F.cosine_similarity(mu, E.mean(0), dim=0).item():+.3f}, "
      f"cos(mu, mean H row) {F.cosine_similarity(mu, H.mean(0), dim=0).item():+.3f}")
print(f"sqrt(hits)-weighted spectrum over touched slots: top-1 share {(sv[0].square() / tot).item():.4f}, top-8 {(sv[:8].square().sum() / tot).item():.4f} "
      f"(Gaussian, same shape and weighting: {(sg[0].square() / (sw.square().sum() * D)).item():.4f}, {(sg[:8].square().sum() / (sw.square().sum() * D)).item():.4f})")

# The direct path's reading of a slot: does T[slot] @ H^T favour the slot's continuations
# beyond the bigram's? Rank of next* among T[slot] @ H^T, for the most-hit slots.
top = idx[hT.argsort(descending=True)[:2000]]
logits = T[top] @ H.T                                              # (2000, V)
rank_next = (logits > logits[torch.arange(top.numel(), device=dev), next_star[top]][:, None]).sum(1)
print(f"\n2,000 most-hit slots: the dominant continuation's rank in T[slot] @ H^T -- median {rank_next.float().median().item():.0f}, "
      f"top-1 {(rank_next == 0).float().mean().item():.1%}, top-10 {(rank_next < 10).float().mean().item():.1%}, top-100 {(rank_next < 100).float().mean().item():.1%} of {V:,}")
print(f"\ndone in {time.perf_counter() - t0:.0f}s")

# For the record: the three traced slots.
print("\n| slot | hits/step | pairs | dominant pair share | next* share | norm | cos(T, E[curr*]) | cos(T, H[next*]) | cos(T, g_excess) |")
print("|---|---|---|---|---|---|---|---|---|")
for tid, word in ((262, "the"), (4736, "quantum"), (32741, "morbidity")):
    s = ((36313 * tid) ^ (27191 * 285)) % (S - 1)
    print(f"| ' of {word}' ({s}) | {rate[s].item():.2f} | {n_pairs[s].item()} | {dom_share[s].item():.3f} | {next_share[s].item():.3f} | {norms[s].item():.1f} | "
          f"{F.cosine_similarity(T[s], E[curr_star[s]], dim=0).item():+.3f} | {F.cosine_similarity(T[s], H[next_star[s]], dim=0).item():+.3f} | "
          f"{F.cosine_similarity(T[s], g_excess[s], dim=0).item():+.3f} |")
