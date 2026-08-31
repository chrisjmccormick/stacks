"""scaling.py -- recomputes every one-off derived constant hardcoded in
decoderstack_medium_pt-sft.py, with the reasoning attached. Pure
python, no torch; run anywhere:  python scaling.py

The pipeline file states values; this file shows where they came from (and
asserts they still agree). When a shape changes, rerun this and update the
hardcoded numbers.
"""
import math

# =============================================================================
# Model shapes (mirror StackConfig)
# =============================================================================

D24 = dict(n_layers=24, d_model=1536, d_vocab=32768, d_mlp=4 * 1536,
           n_q_heads=12, n_kv_heads=12, n_o_heads=12, d_qk=128, d_vo=128,
           num_ves=12, d_ve_gate=12, d_smr_gate=24)

# nanochat's d12 reference (where the AdamW LRs and weight decay were tuned):
# depth 12 x aspect 64 -> d_model 768, 6 heads of 128; VE on 6 alternating
# layers; d_mlp = 4x.
D12 = dict(n_layers=12, d_model=768, d_vocab=32768, d_mlp=4 * 768,
           n_q_heads=6, n_kv_heads=6, n_o_heads=6, d_qk=128, d_vo=128,
           num_ves=6, d_ve_gate=12, d_smr_gate=24)


def param_counts(n_layers, d_model, d_vocab, d_mlp, n_q_heads, n_kv_heads,
                 n_o_heads, d_qk, d_vo, num_ves, d_ve_gate, d_smr_gate):
    """Every trained weight, by group (matches the pipeline's weight_names)."""
    return {
        "input_embeds": d_vocab * d_model,
        "value_embeds": num_ves * d_vocab * (n_kv_heads * d_vo),
        "lm_head":      d_vocab * d_model,
        "W_Q":   n_layers * (n_q_heads * d_qk) * d_model,
        "W_K":   n_layers * (n_kv_heads * d_qk) * d_model,
        "W_V":   n_layers * (n_kv_heads * d_vo) * d_model,
        "W_O":   n_layers * d_model * (n_o_heads * d_vo),
        "W_in":  n_layers * d_mlp * d_model,
        "W_out": n_layers * d_model * d_mlp,
        "ve_gate": num_ves * n_kv_heads * d_ve_gate,
        # resid_lambdas + x0_lambdas + smear_gate + smear_lambda + backout_lambda
        "scalars": n_layers + n_layers + d_smr_gate + 1 + 1,
    }


def matmul_params(c):
    """Weights that participate in a matmul: the matrix banks + ve_gate +
    lm_head. Identical to nanochat's "scaling params" set (which drives the
    horizon), since embeddings are lookups and scalars are pointwise."""
    return sum(c[k] for k in ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out",
                              "ve_gate", "lm_head"])


counts = param_counts(**D24)
total_params = sum(counts.values())
mm_params = matmul_params(counts)

print("== d24 parameter counts ==")
for k, v in counts.items():
    print(f"  {k:14s} {v:>15,}")
print(f"  {'TOTAL':14s} {total_params:>15,}")
print(f"  matmul / scaling params: {mm_params:,}")

# =============================================================================
# FLOPs per token (nanochat gpt.estimate_flops)
# =============================================================================
# 6 FLOPs per matmul-weight parameter (2 forward: multiply+accumulate; 2x that
# in backward), plus the in-attention score/value matmuls:
# 12 * heads * head_dim * effective_seq per layer, where sliding windows cap
# effective_seq (PaLM appendix convention). Embedding lookups and the per-layer
# scalars contribute no matmul FLOPs.

seq_len, short_win = 2048, 768
full_ctxt_layers = [3, 7, 11, 15, 19, 23]                  # "sssL"
windows = [seq_len if i in full_ctxt_layers else short_win for i in range(24)]
attn_flops = sum(12 * D24["n_q_heads"] * D24["d_qk"] * min(w, seq_len) for w in windows)
flops_per_token = 6 * mm_params + attn_flops

print("\n== FLOPs/token ==")
print(f"  6 * matmul params        = {6 * mm_params:,}")
print(f"  attention (18xS768/6xL2048) = {attn_flops:,}")
print(f"  flops/token              = {flops_per_token:,}")

# =============================================================================
# Batch size, horizon, LR/WD corrections
# =============================================================================
# Batch: nanochat's Power Lines auto-compute (Bopt ∝ D^0.383 from the d12
# reference B_REF = 2^19) rounds to 2^20 for d24. Micro-batch: 16 seqs x 2048
# = 32,768 tokens per rank.
# Horizon: the d24 speedrun spec trains at data:param ratio 8 over the scaling
# params (runs/speedrun.sh --target-param-data-ratio=8).
B_REF, B = 2**19, 2**20
ratio = 8
target_tokens = ratio * mm_params
num_iterations = target_tokens // B

# LR: eta ∝ sqrt(B/B_ref) for AdamW (standard), assumed for Muon too.
batch_lr_scale = math.sqrt(B / B_REF)

# Weight decay: T_epoch framework (arXiv:2405.13698) -- keep
# T_epoch = B/(eta*lambda*D) constant => lambda = 0.28 * sqrt(B/B_ref) * (D_ref/D).
# D_ref/D = ratio*scaling(d12) / (ratio*scaling(d24)) -- the ratio cancels.
d12_mm = matmul_params(param_counts(**D12))
weight_decay = 0.28 * batch_lr_scale * (d12_mm / mm_params)

print("\n== batch / horizon / corrections ==")
print(f"  d12 scaling params  = {d12_mm:,}")
print(f"  target tokens       = {ratio} x {mm_params:,} = {target_tokens:,}")
print(f"  num_iterations      = {target_tokens:,} // {B:,} = {num_iterations:,}")
print(f"  batch_lr_scale      = sqrt({B}/{B_REF}) = {batch_lr_scale!r}")
print(f"  weight_decay        = 0.28 * {batch_lr_scale:.6f} * {d12_mm / mm_params:.6f} = {weight_decay:.6f}")
print( "                        (baked as 0.059738 -- nanochat's rounded printout,")
print( "                         which the validated reference/smoke runs used)")

# =============================================================================
# Peak BF16 FLOPS (MFU denominator)
# =============================================================================
# The pipeline inlines just {GH200, H100, A100}. GH200 carries the H100-class
# SXM5 die -> 989 TFLOPS dense BF16 (the old lookup only matched it through
# the "h200" substring by accident). Full table kept here for reference
# (dense, no sparsity; vendor datasheets):
PEAK_FLOPS_TABLE = {
    "GB200": 2.5e15, "B200": 2.25e15, "B100": 1.8e15,
    "H200 NVL": 836e12, "H200": 989e12,
    "H100 NVL": 835e12, "H100 PCIe": 756e12, "H100 (SXM)": 989e12,
    "GH200": 989e12,
    "A100": 312e12, "A800": 312e12, "A40": 149.7e12,
    "L40S": 362e12, "L4": 121e12,
    "MI300X": 1.3074e15, "MI250X": 383e12,
    "RTX 5090": 209.5e12, "RTX 4090": 165.2e12, "RTX 3090": 71e12,
}
print("\n== peak BF16 FLOPS (dense) ==")
for k in ("GH200", "H100 (SXM)", "A100"):
    print(f"  {k:12s} {PEAK_FLOPS_TABLE[k]:.3e}")

# =============================================================================
# cu_seqlens sizing (loader max_num_docs)
# =============================================================================
# Fixed-size cu_seqlens: measured over the climbmix train+val shards
# (scan_max_docs.py, 2026-07-31), the densest run of docs packs 82 into one
# 32,768-token micro-batch (upper bound). Baked 96 = ~17% headroom; the loader
# asserts on overflow. (nanochat's estimate 32768 // 400 rounded up to 16 also
# lands on 96.)
measured_worst = 82
max_num_docs = 96
print("\n== cu_seqlens (max_num_docs) ==")
print(f"  measured worst-case docs/micro-batch = {measured_worst} (scan_max_docs.py)")
print(f"  baked max_num_docs                   = {max_num_docs}")

# =============================================================================
# Cross-check against the constants baked into the pipeline file
# =============================================================================
assert total_params == 1_384_122_122
assert mm_params == 729_810_624
assert flops_per_token == 4_860_160_128
assert num_iterations == 5568
assert batch_lr_scale == 1.4142135623730951
assert d12_mm == 110_100_912
print("\nAll baked constants check out.")
