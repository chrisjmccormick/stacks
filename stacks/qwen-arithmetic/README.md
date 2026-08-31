# qwen-arithmetic

**A GRPO speedrun you can run for free.** Qwen2.5-0.5B-Instruct learns
difficulty-balanced 2-operand arithmetic on ONE Tesla T4 — the free Colab GPU —
in about 12 minutes, with nothing to install.

`train_qwen_arithmetic.py` is a single file: handwritten forward/backward (no
autograd, no `torch.optim`, no `nn.Module`), a CUDA-graph decode engine fused
into the same process as training, full-parameter AdamW, one optimizer step per
round. It is the sibling of [`qwen-gsm8k`](../qwen-gsm8k/), re-derived in fp16
for a 16 GB Turing card.

The dataset is
[ChrisMcCormick/basic-arithmetic](https://huggingface.co/datasets/ChrisMcCormick/basic-arithmetic):
train 10,000 / val 200 / test in-distribution 400 / test out-of-distribution 400.

```bash
colab run --gpu T4 train_qwen_arithmetic.py --timeout 1h
python train_qwen_arithmetic.py            # or any CUDA box with >= 15 GB
```

Nothing to install — **pip never runs**. It needs only what Colab already ships:
torch (2.11+cu128, which the prebuilt attention extension is compiled against and
asserts), numpy, pyarrow, tokenizers, safetensors, huggingface_hub, and wandb
(optional). The file is ASCII-only on purpose, because it is shipped through
Colab's CLI, whose reader uses the local default encoding.

## What it downloads instead of building

On first run it builds nothing — it pulls three finished artifacts into
`~/.cache/qwen-arithmetic/data/` and starts:

- **weights already banked and already fp16**, from
  [ChrisMcCormick/qwen-arithmetic-t4](https://huggingface.co/ChrisMcCormick/qwen-arithmetic-t4)
  (~940 MB), instead of the checkpoint plus a bank-and-cast pass;
- **the dataset already tokenized**, from
  [basic-arithmetic](https://huggingface.co/datasets/ChrisMcCormick/basic-arithmetic)
  `pretokenized/` (0.4 MB), instead of four split parquets plus a render and
  encode pass;
- **the attention kernel already compiled** — flash-attention-turing for sm75,
  from the same model repo (4.9 MB, `fa_turing/`), instead of ~12 minutes of
  nvcc on two vCPUs.

Every one of those passes is cheap on an H100 box and not on Colab. Dropping the
second one also took the prompt template out of the script: it reads token ids
and never renders a prompt, so it cannot drift from the baselines' phrasing — a
prompt change is a dataset change.

The two publishers live in `data/` and are the maintainer's path, not the
runner's — you do not need them to train:

| script | builds | publishes to |
|---|---|---|
| `data/prepare_model.py` | fp16 banks + tokenizer | [`qwen-arithmetic-t4`](https://huggingface.co/ChrisMcCormick/qwen-arithmetic-t4) (model) |
| `data/prepare_arithmetic.py` | the pre-tokenized parquet | [`basic-arithmetic`](https://huggingface.co/datasets/ChrisMcCormick/basic-arithmetic) `pretokenized/`, back into the repo the splits came from |

Both need the uv env that `setup.sh` builds, and both are regenerable — nothing
under `~/.cache/qwen-arithmetic/` is committed, and the sidecar JSONs record what
produced what.

## Rules

Same as qwen-gsm8k: **no command line, no config env.** Every knob is a field of
`T4Config` at the top of the script, edited in place — a run is defined by its
source. If an idea needs a switch, it runs as a labelled arm (edit, run, revert);
the defaults are the measured winners.

Useful modes:

- `host_test` — the host-only self-tests, no GPU (takes the data, skips the
  940 MB of weights);
- `fixed_problems` + `rounds_cap` — single-problem overfit smoke;
- `eval_every = 0` + `final_eval = False` — train-only.

Telemetry is one console line per round, full rows to wandb (`wandb = False`, or
no API key, disables it), and `metrics_<tag>.csv` / `evals_<tag>.csv` /
`evals_detail_<tag>.csv` / `result_<tag>.json` in the cwd.

## Baselines — where this came from

Two H100 predecessors are frozen in [`baselines/`](baselines/), and the table
there is worth reading before you take any single number here seriously:

| Baseline | Stack | val best | test ID | test OOD | wall |
|---|---|---|---|---|---|
| [`20260813_raw-h100`](baselines/20260813_raw-h100/) | handwritten fwd/bwd + CUDA-graph decode, full-parameter | **94.0%** | **89.0%** | 86.75% | 3.7 min |
| [`20260813_hf-vllm`](baselines/20260813_hf-vllm/) | HF `transformers` + in-process vLLM + LoRA | 89.0% | 87.2% | 86.2% | ~7.5 min |

The hf-vllm run is the off-the-shelf-stack ceiling; the raw run is what replacing
that stack with a handwritten one bought (2x faster at +1.8 test ID). This script
is the raw one re-derived for a card with no bf16 tensor cores and 16 GB — the
differences, all measured, are in
[How the T4 port differs](#how-the-t4-port-differs-all-measured) below.

`setup.sh` and `pyproject.toml` at this level build the env those baselines and
the `data/` publishers need. **The shipped speedrun needs neither.**

## Task layer (vs. qwen-gsm8k)

- **Reward:** the LAST number in the completion, compared as an integer — no
  `\boxed{}`, no format channel — plus a method-vocabulary bonus (weight 1.0,
  gated on correctness) that pays for actually working the problem. Measured
  in the experiment line: method reward at 1.0 beat 0.5 beat absent.
- **No trim-to-answer surgery:** its anchor was the boxed span; "the last
  number" anchors nothing, and at budget 640 truncation is 0-1% — the
  run-to-the-budget attractor doesn't form here.
- **Eval:** greedy, K=1 (the experiment line's protocol), so in-loop val and
  final test ID/OOD numbers compare directly across all three scripts. Note what
  OOD means here: `test_ood` is the *same 400 problems* as `test_id`, at the
  same indices, re-worded from a disjoint template set (verified operand for
  operand). The ID/OOD gap is sensitivity to phrasing with difficulty held
  exactly fixed — not harder arithmetic.
- **Operands are columns, not prose.** The dataset carries `a` and `b` in
  operator order (`a op b == answer`), so nothing downstream has to regex them
  back out of the question — which matters because 49 of `test_ood`'s rows
  phrase it "take {b} from {a}" and would come out backwards.
- **Optimizer departure (raw script only):** fp32 masters start mid-bin
  (mantissa `0x8000`) instead of qwen-gsm8k's all-zero init — the bin-edge
  start turns the run's first update into a ~2⁻⁹-relative signSGD kick that
  collapses this task's policy in one round. `TECHNIQUES.md` § The mantissa
  first-step kick has the measurements.

## Model

Qwen2.5-0.5B (`config.json`, asserted against the banks' sidecar at load):
24 layers, d_model 896, 14 Q heads / 2 KV heads, head_dim 64, MLP 4864
(SwiGLU), vocab 151,936, rope_theta 1e6, rms eps 1e-6, QKV biases (o/mlp
none), TIED embeddings (embed table == lm_head), no qk-norm.

## Measured (Tesla T4, Colab, shipped defaults)

200 rounds of 16 problems x K=16 @ budget 320, lr 1e-6, seed 1337:

| | val best | test ID | test OOD | wall (loop) | peak mem |
|---|---|---|---|---|---|
| **T4, this script** (200 rounds) | 89.0% @ round 180 | 81.75% | 82.25% | **12.3 min** | 11.6 GB of 14.6 |
| [raw H100 baseline](baselines/20260813_raw-h100/), for reference (272 rounds) | 94.0% @ round 270 | 89.0% | 86.75% | 3.7 min | 28.7 GB |

Val over the run — and the wall clock at which each point landed, which is the
question a free GPU actually poses:

| round | 0 | 20 | 40 | 60 | 100 | 120 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| min | 0.1 | 1.4 | 2.4 | 3.6 | 6.0 | 7.2 | 9.6 | 10.8 | 12.1 |
| val % | 52.5 | 85.5 | 88.0 | 84.5 | 83.5 | 85.5 | 83.5 | **89.0** | 85.5 |

The median round is 3.20 s (1.82 gen + 1.39 train); no non-finite gradient in
the run, so the loss scale never left 4096.

**Read the wall clock as the result and the accuracy column as a sample.**
Four 200-round runs of this line exist, and they span **val 84.0-92.0 and test
ID 79.0-86.5** on identical config and seed. Nothing about the task is
stochastic at that scale by design — the runs diverge because the backward's
reductions are not deterministic, a few ulps move the weights, and what
actually differs is whether the policy escapes the short-answer attractor in
the first ~20 rounds. One run (the fastest, at 6.6 min) never escaped it and
finished at 79.0/77.0. So single-run accuracy comparisons on this task are not
informative at this budget, and every speed claim below is measured at
*matched work* — round 0, where two runs generate the same rollouts and train
on the same packs — rather than from wall clock.

For the record, in order: the eager script ran 16.6 min (val 91.0 / test ID
86.5); compiling the training step gave 13.1 min (92.0 / 85.25); adding the
fused optimizer gives the 12.3 min above.

Round 0 reads 52.5% here against the raw-H100 baseline's 61.5%: greedy on this model is a
coin flip on the `medium_hard` band (four greedy decodes of the same weights
measured 49-53.5%), plus 15.5% of round-0 completions are greedy repetition
loops truncated at the budget. The trained numbers are what matter.

## Features

T4 straggler cut: once 90% of a round's
rows have retired, the rest get at most
this many more macro windows (8 x 8 =
64 tokens) and are then force-truncated
(scored on their partial text, like a
budget truncation; counted as n_tail).
Measured: t90 ~1.2 s vs gen ~3.9 s at
budget 320 -- the last ~5% of rows cost
2.7 s a round at ~9 ms/step. Training
rounds only; eval runs the full budget.
0 = off.


## How the T4 port differs (all measured)

- **fp16 live weights + fp32 master shadow + fp16 grads under a loss scale +
  fp32 AdamW moments.** The T4 has no bf16 tensor cores (a bf16 matmul falls
  back to the fp32 CUDA cores at ~1/8 the speed), so the live weights,
  activations, KV cache, activation gradients and the big weight-gradient
  buffers are fp16. fp16 carries an 11-bit mantissa (bf16: 8) but a 65,504
  ceiling and a 6e-5 normal floor, so:
  - **fp32 masters, a full shadow copy.** qwen-gsm8k's mantissa trick is a
    *bf16* trick (bf16 == fp32's top half); fp16 has its own exponent width,
    so the update math runs on a real fp32 master and the fp16 live copy is
    re-derived by round-to-nearest after every step. No first-step kick to
    worry about either. State (live + master + moments + grads) = 7.4 GB.
  - **a loss scale (4096)** on the per-token weights: the pg-loss coefficient
    is ~advantage / round-tokens ~ 3e-5, under fp16's normal floor. The
    backward runs scaled, the optimizer divides it back out, and a non-finite
    gradient skips the step and halves the scale. (It never fired in the run
    of record.)
  - **fp32 moments**: on a T4 memory bandwidth is the round's cheap resource
    (one full fp32 AdamW step over 494M params is ~60 ms), so precision here
    is free.
  - **the banks ship fp16 too.** The weights are downloaded in the run dtype,
    so the load is a straight `load_file` to device — no 1 GB bf16 transient
    on a 16 GB card, and nothing in the T4 path touches a bf16 kernel on sm75.
    Bit-identical to casting the raw-H100 baseline's bf16 banks at load, which is
    what this script used to do: bf16's 7 explicit mantissa bits fit inside fp16's
    10, so the cast just happens earlier now.
  - every reduction fp16 cannot hold runs in fp32: rms_norm variance,
    softmax/logsumexp (inside the attention kernel and the CE block),
    bias/norm gradient sums, the sampler.
- **Attention = two kernels, split by what each can do.** Everything varlen —
  the training pack's forward and backward, and generation's prefill — runs on
  [flash-attention-turing](https://github.com/ssiu/flash-attention-turing), a
  FlashAttention-2 written for sm75: fp16, head_dim 64, causal, and both varlen
  and **GQA native**. That last part deletes three pieces of scaffolding the
  CUTLASS path needed — expanding K/V to the 14 query heads, summing dK/dV back
  over each group of 7, and sanitizing the uninitialized tail of its
  per-segment LSE. **Decode cannot move**: flash-attention-turing has no
  KV-cache path, so it keeps PyTorch's vendored CUTLASS FMHA
  (`torch.ops.aten._efficient_attention_forward`, xformers' FMHA inside torch)
  in its padded-KV mode (`cu_seqlens_k` = row starts, `seqlen_k` = per-row live
  length), so each row reads only its live keys, with GQA folded in as 7 query
  ROWS per KV head. Both kernels are checked against fp32 references on the
  card.

  It is still a download, not a build: the extension arrives prebuilt from the
  same model repo as the weights (4.9 MB; the script extracts one `.so` from
  the wheel into its cache and puts that on `sys.path` — pip never runs), with
  a sidecar recording the Python ABI tag, torch version, C++ ABI flag and CUDA
  it was compiled against, asserted at startup so a runtime move fails loudly
  instead of as an `undefined symbol`. Colab pins its runtime and keeps past
  versions for a year, so one wheel serves until the pin moves;
  `agent-ops/stacks/2026-08-16_0131pm_t4-turing-fa-wheel/build_wheel.py`
  rebuilds it inside the runtime it targets (~12 min of nvcc, once).

  **On speed, be honest: this is not why it is here.** Measured on the card,
  flash-attention-turing runs the attention call **1.03–1.34x** at this task's
  pack shapes — but attention is only ~6% of the pack's fwd/bwd, so end to end
  the change sits inside the ±2% run-to-run spread (two pack-level A/Bs
  disagreed in sign). The win scales with segment length — **1.31x at 2048
  tokens against ~1.0x at 64** — and this task's docs are 84–224 tokens, so the
  kernel is simply not where the round's time goes. It is the simpler API and
  the smaller code; the speed is a wash.
- **torch.compile on the training step and the optimizer; the decode body
  stays eager under CUDA graphs.** The pack forward/backward is compiled with `dynamic=True`, which
  holds ONE graph across the pack's varying token count: **measured 1.35x
  against eager** on identical packs (1.29-1.39x over four pack shapes,
  445→321 ms at a full 2048-token pack), worth ~12% of the round. It costs ~4
  minutes of inductor time on Colab's 2 vCPUs cold (~75 s against a warm FX
  cache), once, in warmup — speedrun timings exclude compile. Two details are
  what make it one graph instead of a dozen:
  - the CE row count `S` is pinned to `sel_cap` rather than trimmed to the
    content like `T` is. The chunk loop is a python `range` over `S`, so a
    symbolic trip count specializes — a full recompile per distinct value. The
    padded rows carry weight 0.
  - the attention backward is wrapped in a `torch.library.custom_op` with its
    own fake implementation, because torch 2.11's **meta function for
    `aten._efficient_attention_backward` mis-binds `scale`** and dies during
    fake-tensor propagation (the op itself is fine eagerly). Without the
    wrapper the script still compiles, but as ~10 graph-broken segments, and
    that version measured *slower* per round than plain eager.

  The **optimizer** is compiled too, in the shape the H100 script has always
  used: one fused kernel per tensor with every coefficient gathered from a
  device-side per-step table, so the step sets nothing on the host and never
  recompiles as the step count advances. The Adam bias corrections are folded
  into the tables rather than applied in the kernel — `lr_t =
  lr·√(1−β₂ᵗ)/(1−β₁ᵗ)` and, because the denominator's correction must follow
  it, an eps schedule `eps_t = eps·√(1−β₂ᵗ)` — so the kernel reads raw moments,
  and the betas (0.9/0.999) are literals in the compiled update rather than
  config. Verified same-math: from identical state on identical gradients the
  folded and standard forms agree on the fp32 masters to ≤1e-9 absolute
  (4 of 6M fp16 live weights land 1 ULP apart, the rounding-boundary cases),
  and the round-0 gate reproduces the pre-fold script exactly
  (`agent-ops/stacks/2026-08-20_0639pm_fold-bias-schedules/`). Adapted to fp16, which needs a real
  fp32 master (the raw-H100 baseline's mantissa trick is a bf16 trick), it divides the
  loss scale out from a device scalar and **zeroes each gradient inside the
  same kernel that reads it**, which removes a whole 1 GB pass. Measured
  against the chunked eager version it replaced, at identical work (round 0,
  same rollouts, same packs): **5.98 → 5.53 s, i.e. 0.45 s off every round**,
  ~12% of the median round, and constant — the optimizer touches the same
  7.4 GB whatever the policy is doing. The gradient-norm pass is compiled on
  the same principle: **41.2 → 4.7 ms**, which is the bandwidth floor for
  reading 1 GB of fp16, because eager it materialized a full fp32 copy of every
  gradient buffer before reducing.

  The model itself is not compiled: a T4 decode step is GEMM/bandwidth-bound
  rather than kernel-count-bound (the weights alone are 1 GB / 270 GB/s = ~4
  ms per step, vs ~0.3 ms on an H100), and CUDA graphs already take the launch
  overhead off the critical path. Compiling the decode body before capturing
  it — which the H100 baseline does do — was tried here and is **20x slower**
  (round-0 generation 4.4 → 80.7 s, peak memory +2.3 GB): this engine writes
  each step's K/V into a *view of the dense cache* inside the compiled region,
  and that appears to become a clone-and-copy-back of the cache per layer per
  step. The H100 engine writes through a block table into a paged pool and
  doesn't hit it. The one exception is the **Gumbel-max sampler**, compiled in
  13-17 s: eager it materialized ~5 fp32 (B, V) temporaries per step — ~1.9 GB
  of traffic at B=256, as much as the GEMMs.
- **Dense KV cache, no paging:** `(L, rows, t_row, H_kv, Dh)` x2, one static
  allocation, each row owns its slot — the same static ownership the H100
  engine had behind its block table, without the table. A bucket drop compacts
  survivors' KV rows to the front (~10 ms/round).
- **T4 round shape:** `max_tokens` 320 (H100: 640) and a straggler cut
  (`tail_windows` 8 — once 90% of a round's rows retire, the rest get at most
  64 more tokens and are then scored on their partial text). A T4 decode step
  floors at ~7-9 ms, so a round's last few rows running to the budget cost
  more than the other 250 rows together: t90 ~1.2 s against gen 3.9 s at
  budget 320, and 6.6-9 s at 640. With the cut, gen is ~1.0-1.6 s early and
  ~2.8 s once completions grow. Trained, p99 completion length is ~75 tokens,
  so the 320 budget truncates nothing past the first rounds. Eval always runs
  the full budget.

---

_T4 results and the port's provenance:
`agent-ops/stacks/2026-08-15_0540pm_qwen-arithmetic-t4-port/` — the run of
record is `run_run2.log` + `out_run2/` + `report_run2.md`; the 1.35x compile
measurement is `train_t4-05-benchpack.py` + `run_bench.log`; the round-0 parity
gate for every arm is `report_arms.md`._
