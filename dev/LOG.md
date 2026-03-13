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

## 2026-03-13: Dataset upgrade: FineWeb-EDU 100B → ClimbMix 400B

Switched the pretraining dataset from FineWeb-EDU 100B to ClimbMix 400B, following nanochat's commit `324e69c` which achieved a 27% reduction in GPT-2 speedrun time (2h46m to 2h01m) from this single change.

### What is ClimbMix?

ClimbMix 400B is a curated 400B-token pretraining mixture from [NVIDIA Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix). It is a blend of high-quality web text, code, math, and other sources. The repackaged version used here is hosted at `karpathy/climbmix-400b-shuffle` (6542 shards, up from 1822 for FineWeb-EDU).

### What changed

**Data pipeline (`data/dataset_and_vocab.py`):**
- Source URL: `karpathy/fineweb-edu-100b-shuffle` → `karpathy/climbmix-400b-shuffle`
- Default shards: 370 → 170 (~150 needed for GPT-2 capability + 20 padding)
- Output dataset name: `fineweb_edu_32k_8_370` → `climbmix_32k_8_170`
- Bin files written to `climbmix/` subfolder (was `fineweb_edu/`) inside the dataset dir
- Tokenizer retrained on ClimbMix data (vocab size unchanged at 32768)

**Training script (`decoderstack_small_pt-sft.py`):**
- `DATASET_NAME`, `train_prefix`, `train_files`/`val_files` all point to new dataset
- SFT step calibration flagged for recalibration

**Auxiliary scripts:**
- `download_dataset.py`, `core_dataset.py`, `sft_dataset.py`, `chat_eval_dataset.py` defaults updated

**Not changed (yet):** `decoderstack_medium_pt-sft.py` -- still uses `fineweb_edu_32k_8_370`.

All old values preserved in comments (e.g. `# To use FineWeb-EDU instead: DATASET_NAME = "fineweb_edu_32k_8_370"`).

### New HF repo

`ChrisMcCormick/climbmix_32k_8_170` -- 214 files total:
- 91 training shards + 1 val shard (100M tokens each, ~9.1B training tokens)
- 23 CORE eval `.pt` files
- 4 chat eval files (MMLU, ARC-Easy, ARC-Challenge)
- 91 SFT files (token + mask shard pairs)
- Tokenizer (tokenizer.pkl + token_bytes.pt)

### Tokenizer comparison (ClimbMix-trained vs GPT-2 baseline)

| Text Type | GPT-2 Ratio | Ours Ratio | Relative |
|-----------|-------------|------------|----------|
| news      | 4.50        | 4.49       | -0.2%    |
| code      | 2.19        | 3.17       | +31.1%   |
| math      | 1.96        | 2.01       | +2.6%    |
| science   | 4.28        | 4.56       | +6.2%    |
| cm-train  | 4.67        | 4.74       | +1.4%    |

The ClimbMix-trained tokenizer shows a large improvement on code (+31%) since ClimbMix includes code in its mixture, unlike FineWeb-EDU which was text-only.

---