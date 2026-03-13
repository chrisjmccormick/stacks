# Experiment Log

A running summary documenting some experiments and findings. Started March 13, 2026.

---

## 2026-03-13: Tune Polar Express Kernel Configs for Medium

Swept 216 configs per kernel (BLOCK_{M,N} in {64,128,256}, BLOCK_K in {32,64,128}, stages in {2..5}, warps in {4,8}) across four workloads matching actual medium-model batched tensor shapes. Updated `triton_kernels_medium.py` with winning configs. End-to-end training improved from 2131.86ms to 2126.29ms per step (~0.3%).

### Motivation

The Triton Polar Express kernels (XXT, XTX, ba_plus_cAA) were copied from the small track with configs tuned for 768x768. The medium track operates on 1536x1536 outputs with reduction dims up to 6144, so SM utilization and tile efficiency differ significantly.

### What changed

Updated three launcher configs in `triton_kernels_medium.py`:

| Kernel | Old Config | New Config | Microbench Speedup |
|--------|-----------|------------|-------------------|
| XXT | M=128 N=128 K=64 stages=4 warps=4 | M=128 N=256 K=64 stages=4 warps=8 | 1.06x (square), 1.26x (wide K=6144) |
| XTX | M=128 N=128 K=64 stages=4 warps=8 | M=128 N=256 K=32 stages=4 warps=8 | 1.08x |
| ba_plus_cAA | M=128 N=128 K=64 stages=4 warps=4 | M=128 N=256 K=64 stages=3 warps=8 | 1.07x |

Common pattern: all three kernels preferred wider N tiles (256 vs 128) and 8 warps. This doubles grid parallelism from 78 to ~150 blocks after symmetry skip, better saturating the H100's 132 SMs.

### Observations

- **Shared memory is the binding constraint.** Many large configs (256x256, 128xN with K=128 at stages>=3) hit the 232KB smem limit. BLOCK_SIZE_K=128 only works with stages<=3 and one small tile dim.
- **End-to-end impact is small** because Polar Express runs once per optimizer step and is not the bottleneck -- the forward/backward matmuls dominate. The fused MLP kernel (Task 2) should have much larger impact since it's on the critical path.
- **XXT sees the same optimal config** for both square (K=1536) and wide (K=6144) workloads, so no shape-dependent dispatch is needed.
- **XTX prefers smaller K tiles** (32 vs 64) despite having the deepest reduction (M=6144), likely because smaller K tiles leave more smem budget for pipelining.

### Benchmark script

`dev/bench_polar_kernels.py` -- run from `stacks/` dir with `micromamba run -n stacks python dev/bench_polar_kernels.py`.

### Validation

50-step training: step_avg **2126.29ms** (baseline 2131.86ms, delta -5.6ms, -0.26%). MFU ~39.6%.

---