"""
Benchmark script for tuning XXT, XTX, ba_plus_cAA Triton kernels
for medium-track dimensions (n_embd=1536, 24 layers, 12 heads).
"""

import torch
import triton
import time
import itertools
import sys

from triton_kernels_medium import XXT_kernel, XTX_kernel, ba_plus_cAA_kernel

torch.manual_seed(42)
device = "cuda"


def bench_XXT(A, out, config, warmup=5, iters=50):
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = config
    M = A.shape[-2]
    K = A.shape[-1]
    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)

    for _ in range(warmup):
        XXT_kernel[grid](
            A_ptr=A, C_ptr=out, M=M, K=K,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=8, LOWER_UPPER=1,
            num_stages=num_stages, num_warps=num_warps,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        XXT_kernel[grid](
            A_ptr=A, C_ptr=out, M=M, K=K,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=8, LOWER_UPPER=1,
            num_stages=num_stages, num_warps=num_warps,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed


def bench_XTX(A, out, config, warmup=5, iters=50):
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = config
    M = A.shape[-2]
    K = A.shape[-1]
    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = (batch_size * triton.cdiv(K, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_N),)

    for _ in range(warmup):
        XTX_kernel[grid](
            A_ptr=A, C_ptr=out, M=M, K=K,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=8, LOWER_UPPER=1,
            num_stages=num_stages, num_warps=num_warps,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        XTX_kernel[grid](
            A_ptr=A, C_ptr=out, M=M, K=K,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=8, LOWER_UPPER=1,
            num_stages=num_stages, num_warps=num_warps,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed


def bench_ba_plus_cAA(A, out, config, warmup=5, iters=50):
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = config
    M = A.shape[-2]
    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)

    alpha, beta = 0.5, 1.0

    for _ in range(warmup):
        ba_plus_cAA_kernel[grid](
            A_ptr=A, C_ptr=out, M=M,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
            alpha=alpha, beta=beta,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=8, LOWER_UPPER=1,
            num_stages=num_stages, num_warps=num_warps,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        ba_plus_cAA_kernel[grid](
            A_ptr=A, C_ptr=out, M=M,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
            alpha=alpha, beta=beta,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=8, LOWER_UPPER=1,
            num_stages=num_stages, num_warps=num_warps,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed


def run_sweep(kernel_name, bench_fn, A, out, configs):
    print(f"\n{'='*80}")
    print(f"Benchmarking {kernel_name}")
    print(f"  Input shape: {tuple(A.shape)}, Output shape: {tuple(out.shape)}")
    print(f"  Testing {len(configs)} configurations...")
    print(f"{'='*80}")

    results = []
    for i, cfg in enumerate(configs):
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = cfg
        try:
            t = bench_fn(A, out, cfg)
            results.append((cfg, t))
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(configs)}] M={BLOCK_SIZE_M:3d} N={BLOCK_SIZE_N:3d} "
                      f"K={BLOCK_SIZE_K:3d} stages={num_stages} warps={num_warps} => {t*1000:.3f} ms")
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] M={BLOCK_SIZE_M:3d} N={BLOCK_SIZE_N:3d} "
                  f"K={BLOCK_SIZE_K:3d} stages={num_stages} warps={num_warps} => FAILED: {e}")

    results.sort(key=lambda x: x[1])
    print(f"\nTop 10 configs for {kernel_name}:")
    for rank, (cfg, t) in enumerate(results[:10]):
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = cfg
        print(f"  #{rank+1}: M={BLOCK_SIZE_M:3d} N={BLOCK_SIZE_N:3d} K={BLOCK_SIZE_K:3d} "
              f"stages={num_stages} warps={num_warps} => {t*1000:.3f} ms")

    best_cfg, best_t = results[0]
    baseline_cfg = (128, 128, 64, 4, 4 if kernel_name != "XTX" else 8)
    baseline_t = None
    for cfg, t in results:
        if cfg == baseline_cfg:
            baseline_t = t
            break
    if baseline_t is None:
        baseline_t = bench_fn(A, out, baseline_cfg)

    print(f"\n  Baseline (128,128,64,4,{'4' if kernel_name != 'XTX' else '8'}): {baseline_t*1000:.3f} ms")
    print(f"  Best:     {best_cfg}: {best_t*1000:.3f} ms")
    if baseline_t > 0:
        print(f"  Speedup:  {baseline_t/best_t:.3f}x")

    return best_cfg, best_t


def main():
    block_mn = [64, 128, 256]
    block_k = [32, 64, 128]
    stages = [2, 3, 4, 5]
    warps = [4, 8]

    configs = list(itertools.product(block_mn, block_mn, block_k, stages, warps))
    print(f"Total configs per kernel: {len(configs)}")

    # =========================================================================
    # XXT - Square attention weights: (96, 1536, 1536) => (96, 1536, 1536)
    # =========================================================================
    A_sq = torch.randn(96, 1536, 1536, device=device, dtype=torch.bfloat16)
    out_sq = torch.empty(96, 1536, 1536, device=device, dtype=torch.bfloat16)
    best_xxt_sq, _ = run_sweep("XXT_square (96, 1536, 1536)", bench_XXT, A_sq, out_sq, configs)
    del A_sq, out_sq
    torch.cuda.empty_cache()

    # =========================================================================
    # XXT - Wide MLP down: (24, 1536, 6144) => (24, 1536, 1536)
    # =========================================================================
    A_wide = torch.randn(24, 1536, 6144, device=device, dtype=torch.bfloat16)
    out_wide = torch.empty(24, 1536, 1536, device=device, dtype=torch.bfloat16)
    best_xxt_wide, _ = run_sweep("XXT_wide (24, 1536, 6144)", bench_XXT, A_wide, out_wide, configs)
    del A_wide, out_wide
    torch.cuda.empty_cache()

    # =========================================================================
    # XTX - Tall MLP up: (24, 6144, 1536) => (24, 1536, 1536)
    # =========================================================================
    A_tall = torch.randn(24, 6144, 1536, device=device, dtype=torch.bfloat16)
    out_tall = torch.empty(24, 1536, 1536, device=device, dtype=torch.bfloat16)
    best_xtx_tall, _ = run_sweep("XTX_tall (24, 6144, 1536)", bench_XTX, A_tall, out_tall, configs)
    del A_tall, out_tall
    torch.cuda.empty_cache()

    # =========================================================================
    # ba_plus_cAA - Square: (120, 1536, 1536) => (120, 1536, 1536)
    # =========================================================================
    A_ba = torch.randn(120, 1536, 1536, device=device, dtype=torch.bfloat16)
    out_ba = torch.empty(120, 1536, 1536, device=device, dtype=torch.bfloat16)
    best_ba, _ = run_sweep("ba_plus_cAA (120, 1536, 1536)", bench_ba_plus_cAA, A_ba, out_ba, configs)
    del A_ba, out_ba
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY - Best configs for medium model")
    print("="*80)
    print(f"XXT (square, attention): BLOCK_M={best_xxt_sq[0]}, BLOCK_N={best_xxt_sq[1]}, "
          f"BLOCK_K={best_xxt_sq[2]}, stages={best_xxt_sq[3]}, warps={best_xxt_sq[4]}")
    print(f"XXT (wide, MLP down):    BLOCK_M={best_xxt_wide[0]}, BLOCK_N={best_xxt_wide[1]}, "
          f"BLOCK_K={best_xxt_wide[2]}, stages={best_xxt_wide[3]}, warps={best_xxt_wide[4]}")
    print(f"XTX (tall, MLP up):      BLOCK_M={best_xtx_tall[0]}, BLOCK_N={best_xtx_tall[1]}, "
          f"BLOCK_K={best_xtx_tall[2]}, stages={best_xtx_tall[3]}, warps={best_xtx_tall[4]}")
    print(f"ba_plus_cAA:             BLOCK_M={best_ba[0]}, BLOCK_N={best_ba[1]}, "
          f"BLOCK_K={best_ba[2]}, stages={best_ba[3]}, warps={best_ba[4]}")

    # Check if XXT needs per-shape configs or a single config works
    if best_xxt_sq == best_xxt_wide:
        print("\nXXT: Same config works for both square and wide -- use single config.")
    else:
        print("\nXXT: Different configs optimal for square vs wide -- consider shape-dependent dispatch.")


if __name__ == "__main__":
    main()
