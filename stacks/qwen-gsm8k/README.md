# GSM8K Speedrun

Single-file GRPO speedrun: 
- Model: Qwen2.5 500M 
- Dataset: GSM8K
- GPU: 1xH100 

Current best:
- TODO

## Implementation

- Handwritten forward/backward
- Single process generation + training, same weights
- CUDA-graph captured decode engine 

Weights:
- "Live weights" are bf16
- Separate fp32 master weights preserve precision for optimizer
- TODO:
   - Grad precision?
   - Mantissa trick?
 
Decode engine:
- Fused QKV and MLP gate/in GEMMs to reduce kernel launch count.
- Compiled decode bodies
- CUDA graph-capture, "macro-window replays" means only one host transfer every 8 decode steps.
- "bucket ladder" - Separate graphs for different numbers of active generations. 
    - Go down a bucket size once enough generations are complete. 
- Gumbel-max sampler (one elementwise pass + argmax). 
- KV cache is a simple static allocation; every generation fits at once.
- Prefix sharing: every context prefills once (packed varlen) and broadcasts into its K sibling rows.

## verl Baseline

I ran the benchmark using the `verl` library, just to get some sense of expected performance.

- verl v0.8.0 + vLLM 0.12
- 1 epoch = 467 rounds of 16 problems x K=16 
- Max completion 512 tokens
- lr 1e-6
- temp 1.0

Test set:
- "mean@8" - Accuracy at K=8 : `30.2 -> 54.8`
- "best@8" - Any correct at K=8: `60.4 -> 75.2`

Timing:
~77 min (6.39 s/round)

## Training

Beyond systems changes, I've already started to change the task / approach:
- Qwen is good at producing boxed answers, and teaching it to format with `####` is a boring objective that complicates the rewards, so the prompt says to box the answer (`\boxed{}`), and this is the only accepted form. 
- Truncated-but-correct rollouts are trimmed to the end of their answer, then we append the EOS token to teach the model to stop.

- Sampling: temperature 1.0, no top-k, no top-p  (TODO - "Gumbel-max over logits")
- GRPO advantage: per-group (r - mean) / (std_{ddof=1} + 1e-6), z-scored - TODO
- loss: token-mean over ALL response tokens in the round
- no PPO ratio/clip (on-policy, 1 step/round), no KL / reference model, 
- no entropy bonus (TODO - I'm unfamiliar)
- Full fine-tuning
- AdamW at a fixed lr of 1e-6 (no warmup / schedule); the bias corrections are
  folded into per-step lr/eps tables (`lr_bias_corr` / `eps_bias_corr`), so the
  compiled update reads RAW moments — see § Schedules in the script. The betas
  are `cfg.beta1` / `cfg.beta2`, but the compiled update still carries `1-beta`
  as `lerp_` literals, so they live in two places and an assert beside cfg
  enforces the pairing.
- "Fixed lr" is misleading and the run page now says so. At betas (0.9, 0.999)
  over 467 steps the bias-corrected rate is 0.32x of nominal at step 1, dips to
  **0.15x by step 12**, and reaches only **0.61x** at the last step — the run
  never gets to its own learning rate. 0.999 is sized for a far longer horizon.
- **Tested, and it made no difference.** `beta2` 0.99 gives v a 69-step half-life
  instead of 693 (longer than the whole run) and **2.05x** the total step
  distance. One paired epoch: full-test mean@8 53.80 vs 55.08, paired
  -1.28 pp with a 95% CI of -2.71..+0.15, last-100 solve rate 0.656 vs 0.659.
  Inside the run-to-run band on every measure, so **this config is not
  learning-rate-limited near this range** — see
  `agent-ops/stacks/2026-08-21_0933pm_gsm8k-beta2-099/`.
- weight decay is enabled at wd 0.01 to mirror verl, but at lr*wd = 1e-8 a
  step's decay moves an fp32 master by under half a ULP, which always rounds
  back — provably a no-op, kept as reference

For reference, Qwen2.5-0.5B architecture:
- 24 layers
- d_model 896
- 14 Q heads / 2 KV heads
  - biases on QKV, none on O or MLP.
  - head_dim 64
- MLP 4864 (SwiGLU)
- Vocab 151,936 with tied embeddings

- rope_theta 1e6, rms eps 1e-6

## How to Run
Run (after clone + cd into this folder):

```
bash setup.sh                    # uv env + prepare_model + prepare_gsm8k
python train_qwen_gsm8k.py       # reads ~/.cache/qwen-gsm8k/data/
```

There are no command line or env arguments; all settings are in a single GSM8KConfig (§ Config) object.
It's held as a global object named `cfg`, and referenced directly rather than passed.

Some modes that exist (for now):
- `host_test` (host-only self-tests, no GPU) 
- `fixed_problems` + `rounds_cap` (single-problem overfit smoke),
- `eval_every = 0` + `final_eval = False` (train-only). 

**Logging**

- Each gen+train round is one output line in the log. 
- Full detailed logged to `wandb`.
- And metrics_<tag>.csv / evals_<tag>.csv / evals_detail_<tag>.csv / result_<tag>.json.

## Docs

Claude tends to journal about its experiences and insights by inlining these reflections as "code comments".

I try to strip these. For genuinely tricky sections, where future Claudes might get tripped up, I route the insights to TECHNIQUES.md and leave a reference note in the code.

## FlashAttention

FA3 and FA2 are retrieved via the HF kernels hub.
(TODO - I think FA2 is used for faster decode, even on H100)
