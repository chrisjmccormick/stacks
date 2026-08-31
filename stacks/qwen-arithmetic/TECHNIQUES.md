# Techniques

Write-ups of the tricky details in this sub-repo's scripts — the invariants
that are non-obvious, cost real debugging time to rediscover, and would bloat
the code if spelled out inline.

> **Scope note.** The one section here so far belongs to the **raw H100
> baseline** ([`baselines/20260813_raw-h100/`](baselines/20260813_raw-h100/)),
> not to the shipped T4 speedrun. It is a *bf16* problem: the mantissa trick it
> describes only works because bf16 is fp32's top half. `train_qwen_arithmetic.py`
> runs fp16, which has its own exponent width, so it carries a real fp32 master
> and has no first-step kick to fix. The frozen baseline script says this file
> is "in this folder" — it was, at the commit that froze it.

The code carries a one-line pointer to the section name; the reasoning lives
here. (The pack-layout doctrine these scripts also rely on — pad tails as real
varlen segments with varying content — is qwen-gsm8k's `TECHNIQUES.md`
§ Padded varlen; it is not repeated here.)

- [The mantissa first-step kick](#the-mantissa-first-step-kick)

---

## The mantissa first-step kick

**Symptom:** with machinery copied verbatim from qwen-gsm8k, three optimizer
steps at lr 1e-6 destroyed the model. Val accuracy went 61.5% → 23.5% → 0.5%;
completions became fluent English wrapping digit salad ("The result of 3956
plus 6284 is 102400000"). Gradient norms were normal (~2.3), no NaNs, and the
loss looked healthy — nothing in the telemetry said "diverged."

### The mechanism

The mantissa-trick optimizer stores fp32 masters as
`master_bits = (live_bf16 << 16) | mantissa`, and the writeback after each
update is a bit **truncation** (round-to-nearest could carry into the top bits
and break the lossless pairing).

The trap: at init the mantissa was **all-zero**, which parks every master
exactly **on** its bf16 bin boundary. From there, the very first
negative-signed update — of *any* size, because truncation borrows on any
decrease — drops the live weight a full bf16 ULP. The positive half moves
nothing (the master needs a full ULP of accumulated increase to carry). Net
effect of step one, regardless of lr:

> a one-shot **signSGD step at ~2⁻⁹ relative magnitude** on the
> negative-gradient half of every tensor — roughly 1000× the nominal 1e-6
> update.

Three follow-up facts pin the diagnosis:

1. **The optimizer math is exact.** Compiled and eager `adamw_step_fused`
   reproduce a plain fp32 AdamW reference bit-for-bit on the master.
2. **The magnitude alone is harmless.** Applying *random* ±1-ULP truncation
   jitter of the same shape to every parameter leaves the model solving all
   probes perfectly. The damage requires the *coherent gradient direction* —
   a fixed-norm perturbation aligned with the loss landscape's steep direction
   is a huge functional step; the same norm in a random direction is noise.
3. **qwen-gsm8k takes the same kick.** Measured at its HEAD, one round moves
   its val tracker mean@8 33.0 → 22.4 and fmt 79% → 98%. It *survives* because
   its kick lands productively (everyone boxes; training solve rises) and 467
   rounds re-earn the rest. Arithmetic's kick lands on "answer short" — the
   dominant easy direction when 61% of short answers are already correct and
   the long ones are disproportionately wrong — and the policy collapses to
   ~18-token digit salad it never recovers from.

FA3's varlen backward was also cleared explicitly (fwd/bwd match an SDPA
autograd reference, including with ghost-heavy `cu_seqlens` layouts), since a
"coherent wrong gradient with a normal norm" fit an attention-backward bug
just as well.

### The fix

Initialize the mantissa **mid-bin**:

```python
p.mantissa = torch.full(p.shape, 0x8000, dtype=torch.uint16, device=device)
```

The master starts half a ULP *inside* its bin (in magnitude, so it is correct
for both signs under fp32's sign-magnitude layout), and small updates now
accumulate honestly: the live weight flips only after a genuine half-ULP of
net movement. The pairing stays lossless — truncation still recovers the live
bits exactly, so live == checkpoint at init, unchanged.

With the fix, the same 3-round smoke holds val 61.5% → 62.5% with training
solve rising 61% → 68% — and the first full 272-round run of the defaults
went on to val 94.0% / test ID 89.0% / OOD 86.75% in a 3.7-minute loop,
beating the [hf-vllm baseline](baselines/20260813_hf-vllm/) on every number.

### Why this stays a local departure

qwen-gsm8k's published results were produced *with* the kick; its first-step
jolt fast-forwards format compliance and the run's trajectory is calibrated
around that. Porting mid-bin init there would change proven dynamics — worth
trying as a labelled arm (the kick plausibly costs it the ~10 points it spends
early rounds re-earning), but it is that folder's experiment to run, not this
one's to impose.
