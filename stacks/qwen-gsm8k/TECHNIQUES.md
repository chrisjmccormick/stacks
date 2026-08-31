# Techniques

Write-ups of the tricky details in `train_qwen_gsm8k.py` — the invariants that
are non-obvious, cost real debugging time to rediscover, and would bloat the
code if spelled out inline. The code carries a one-line pointer to the section
name; the reasoning lives here.

- [Padded varlen](#padded-varlen)
- [Boxed answers](#boxed-answers)

---

## Padded varlen

Training packs are a fixed compiled shape (`cfg.train_t` = 16,384 tokens), so
the last pack of a round is under-filled and carries a pad tail. FlashAttention
varlen has three separate requirements on that tail. Violating any one of them
produces the **same signature: every weight gradient goes NaN while the reported
loss stays finite.** That combination is what makes it hard to find — the loss
is computed only over selected completion positions, and those are all healthy.

### 1. The pad tail must be a real `cu_seqlens` segment

FA-varlen only touches rows that fall inside a `cu_seqlens` segment. Rows past
the last segment are never written, so they come back as whatever was in the
buffer — routinely NaN.

The loss never reads them (their weight is 0), but the **weight-gradient
matmuls do**: `grad.mT @ x` reduces over *every* row of the pack, and `0 * NaN`
is NaN. One orphan row poisons every shared parameter bank.

So `plan_packs` closes the tail as its own attended segment
(`cu[len(members) + 1] = train_t`) rather than leaving it outside the last doc.

The remaining `cu` entries are ghost segments: `cu` is a fixed `(max_docs + 2,)`
array filled with `train_t`, so unused trailing entries describe zero-length
segments. Those are legal and cost nothing, and the fixed size is what keeps the
compiled shape static.

### 2. `max_seqlen` must bound the pad tail, not the longest document

`max_seqlen` is FA's per-segment launch bound: it launches
`ceil(max_seqlen / BLOCK_M)` m-blocks per segment, and rows beyond that bound
are left unwritten — the same 0\*NaN poisoning as above, arriving by a different
route.

The longest segment in a pack is the **pad tail**, which can span nearly the
whole pack. So the bound is `cfg.train_t` (16,384), *not* the longest document
and not `cfg.t_row` (768, the decode row capacity). Passing `t_row` here was
the original bug: correct for generation, catastrophically short for training.

Over-estimating is safe — surplus m-blocks early-return — and `train_t` is a
compile-time constant, so it does not specialize anything beyond the single
static shape the packs already have.

Both the forward (`flash_attn_varlen_fwd_lse`) and the backward
(`flash_attn_varlen_bwd`) take this bound, and both must get `train_t`.

### 3. The pad tail must be non-degenerate

It is not enough for the tail to be attended; its **contents must vary**.

RoPE rotates Q and K but not V. A long run of identical token ids at identical
positions therefore yields identical V rows across the segment, the softmax over
it is uniform, and FA's **backward** returns NaN on that degenerate case once the
segment is large. The forward stays finite, which is why this one survives a
forward-only smoke test.

So the tail is filled with a benign varying causal run:

```python
idx[o:] = 1 + (np.arange(n_pad) % 4096)   # ids vary, and stay inside any vocab
pos[o:] = np.arange(n_pad) % cfg.t_row    # positions vary, cycled to stay inside
                                          # the rotary cache
```

Targets and weights on those rows stay 0, so they contribute nothing to the loss
— the varying content exists purely to keep attention's backward well-posed.

### Why prefill does not need any of this

`Engine.run_round`'s prefill pack pads with a constant `PAD_ID` at position 0 and
is *fine*, because prefill is forward-only: there is no backward to blow up on a
degenerate softmax. It still closes its pad tail as a real segment and passes
`cfg.prefill_t` as `max_seqlen`, so requirements 1 and 2 hold there too; only
requirement 3 is specific to the training path.

### Where this comes from

The same doctrine, hit independently and written up in more depth, lives in
`guided-rl/src/guided_rl/dataloader.py` (module docstring and
`build_reinforce_packs`). In this repo it was rediscovered at round 0 of the
first GSM8K training run (commit `e7a7732`).

---

## Boxed answers

The reward is boxed-only: **the last parseable `\boxed{}` holds the gold
answer, or the rollout scores zero.** No `####`, no trailing bare number. This
section is why, and why the box is not required to be the last thing said.

### Why boxed-only, and not the looser channels

Measured on 31,656 untrained rollouts, asking for `\boxed{}` in the prompt
rather than `####`:

| | answer-mode (`####`, then box, then last number) | boxed-only |
|---|---|---|
| solve rate | 29.7% | **33.4%** |
| dead groups (all-zero, no gradient) | 27.6% | 30.1% |

Boxed-only is the **more accurate** scorer and barely harder to train on. Two
supporting numbers from the same probe:

- Asking for `\boxed{}` moves box usage 22.4% -> 78.9%, and lenient solve
  29.7 -> 36.2. The model can do it; it just has to be asked.
- Of the rollouts answer-mode calls correct, the last box disagrees with gold
  in **0.0%** of cases. When this model boxes something, the box is the answer.

So the loose channels were not buying accuracy, they were buying noise: a bare
last number rewards a truncated ramble that happens to end near the right
digits — exactly the attractor `trim_to_answer` exists to drain.

An answer-mode scorer ran alongside the real reward as telemetry for several
sessions, to keep the headline comparable to a verl reference run. That
comparison is no longer a goal and the scorer is gone; the trainer computes one
number.

### The box is almost never final

Requiring the box to be the last thing said, or the turn to have ended, leaves
**97% of groups all-zero** — no spread, no advantage, no gradient. Qwen opens a
LaTeX environment and has to close it, so a correct completion routinely ends
`\(\boxed{18}\).` or `\[\n\boxed{18}\n\]`. `extract_answer` therefore scans for
the *last parseable* box anywhere in the text.

The same fact drives `anchored_answer_end`: a trim that halts at the first
whitespace past the box orphans the `\]`. Measured on the probe's rollouts,
cutting there left **unbalanced LaTeX in 34% of trims** (647 of 1,882), which
teaches the policy to open a math environment and stop inside it. Swallowing one
following whitespace-and-closer group removes ~84% of those.

### `norm_answer` and the non-finite overflow

`float()` does not raise on a long digit run — it overflows to `inf`, and
`int(inf)` raises. That crashed a verl run inside a ray worker. A non-finite
value is never a valid GSM8K answer, so `norm_answer` keeps the cleaned digits
instead: they compare unequal to any gold, which scores the rollout wrong, the
right outcome.

### Where this comes from

`agent-ops/stacks/2026-08-09_0150pm_boxed-format-probe/`.
