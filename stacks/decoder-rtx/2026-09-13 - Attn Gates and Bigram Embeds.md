# 2026-09-13 — Attention Gates and Bigram Hash Embeddings

Attempts to port two modded-nanogpt features into decoder-rtx. **Neither is working yet.**
Both are implemented and verified correct against autograd, but no configuration tried so
far matches or beats the control. These are attempts-in-progress, not verdicts on the
techniques.

Branch: `sep-14-baseline`. All runs are 1035 steps, 512K tokens/step, one RTX PRO 6000.

## Results

| run | change | val_bpb | slack | s/step | peak mem |
|-----|--------|---------|-------|--------|----------|
| A | control (`sep-7-baseline`, renamed vars only) | 0.900104 | −104 | 1.502 | 85.6 GB |
| B | + attention output gates | 0.900147 | −147 | 1.527 | ~86.9 GB |
| C | + gates + bigram embeds (lr 0.3) | 0.900492 | −492 | 1.565 | 88.2 GB |
| D | C with bigram lr 0.9 | *pending* | | 1.566 | 89.1 GB |

Slack = (0.900000 − val_bpb) × 1e6. Stated noise floor is ±35; one step is worth ~70.

Note the control itself lands at −104, i.e. it does not quite reach 0.900 in 1035 steps.
The README's +410 figure describes the older 1000-step `2026-09-04_BigramInit` config, so
it isn't the right comparison for this branch.

## What was verified

Neither result appears to be an implementation bug:

- **Variable renames are semantics-preserving.** `sep-7-baseline` and the renamed file are
  token-identical up to a consistent 1:1 identifier map (8629 tokens each). The three
  non-1:1 cases are name reassignments (`x_out`/`x0` → `x`/`x0`, `qr_grad`/`q_grad` → `qb`,
  `smeared_grad`/`x0_grad` → `x0b`), each checked by hand.
- **Attention-gate backward** matches autograd in float64 to round-off on all four grads
  (`W_O`, `attn_gate`, `y`, `x_biased_hat`).
- **Bigram hash** reproduces the archive's `get_bigram_hash` bit-for-bit, including the
  quirk where position 1 pairs against the reserved slot rather than token 0.
- **Bigram blend backward** matches autograd in float64 (`x_bigram`, `bigram_lambdas`,
  and the existing `x0`/`x0_lambdas` grads).
- The bigram table is genuinely in the optimizer — the +2.3 GB memory step matches the
  predicted table + Adam state exactly.

## Attention gates (run B)

Per-(token, head) sigmoid on `y` before the `W_O` projection, reading the same 12-dim slice
of the normed layer input as the VE gate. All 11 layers get one; the archived decoder-small
skips its layer 6 only because that layer has no attention.

B tracked the control almost exactly (Δ ≤ 43 points throughout, ending 43 worse — at the
stated ±35 noise floor). Step 0 is bit-identical, which is expected: `W_O` initialises to
zero, so the gate multiplies something that contributes nothing.

One candidate explanation, untested: a constant ~0.5 gate is exactly absorbable into
`W_O`'s scale, so the gate can only pay off once it becomes strongly *input-dependent*, and
at Muon lr 0.02 on a 6×12 matrix it may not move far enough within 1035 steps. **The gate
learning rate has not been swept** — that is the obvious next thing to try before drawing
any conclusion.

## Bigram hash embeddings (runs C, D)

Port of [modded-nanogpt PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201)
(ClassicLarry, merged 2026-01-20), which reported −5.1s and a 1765 → 1600 step cut (−9.35%).

Matched from the PR: zero-init table, `5 × vocab` slots, `bigram_lambdas = 0.1`, the hash,
and raw un-normed addition into the per-layer blend.

### Known differences from the PR

Ordered by guessed impact, all untested except the first:

1. **Bigram table lr.** PR gives it the value-embedding group — Adam base 0.008 × `lr_mul`
   75 × schedule ≤1.73 = **0.60–1.04** effective. Run C used 0.30 (decoder-rtx's
   input-embedding row). Both optimizers are normalized AdamW (`w += lr·m/(√v+ε)`), so
   these compare directly in weight-delta units: C was running ~2–3.5× cold. Run D tests
   0.9. *The 75× multiplier is not the gap — the PR's Adam base is simply 37× colder than
   decoder-rtx's embedding lr.*
2. **β2.** PR uses 0.95 on every embedding; decoder-rtx uses 0.995. Unchanged in C and D.
3. **Weight decay.** PR gives the bigram table 5× base; C/D use the input-embed value,
   which is 10× less than decoder-rtx's own `value_embeds`.
4. **`x0_lambdas` init.** PR inits them to **zero**, so the bigram path is the only live
   re-read into the stream at init. decoder-rtx inits `linspace(0.20, 0.05)`, so the
   bigram table arrives as a competitor to an already-active x0. This is pre-existing
   architecture, not a knob to flip casually.
5. **`cooldown_frac`.** The PR raised it 0.50 → 0.55 *together with* the step cut, noting
   "I find it works to increase this as step count decreases." Not yet relevant — there is
   no step cut to make. decoder-rtx is already at 0.65, though the schedule shapes differ.

### Confounds that make the PR a loose reference

decoder-rtx and modded-nanogpt differ in ways that have nothing to do with this feature:

- **Vocab** — 32,768 vs 50,304
- **Dataset** — Climbmix vs FineWeb
- **Loss target** — 0.900 val bpb vs 3.28 CE, and a much shorter token budget here
- **Bigram distribution init** — decoder-rtx folds corpus bigram statistics into the input
  embedding and lm_head (`embed_prior`/`log_bigram`, credited with removing the 40-step
  warmup). modded-nanogpt has no counterpart. A learned bigram hash table may overlap with
  information this model already has, but that is a hypothesis and has not been tested —
  doing so would mean running the hash table with the bigram init disabled.
- A pile of independently-tuned lr/beta/wd choices that were never synced between the two.

Any of these could be why the port hasn't landed yet, and several could be acting at once.

## Where to pick up

- Sweep the attention-gate lr — cheapest untested knob, and B's near-perfect overlap with
  the control is consistent with the gate simply not moving.
- If run D's hotter lr doesn't help, the redundancy question (bigram init vs bigram hash
  table) is the one worth isolating, since it is the structural difference rather than a
  tuning one.
- Adding 126M zero-init parameters to a 543M-token run is a real cost regardless; the PR
  itself notes the model ends up with more parameters than training tokens, which it could
  afford and this budget may not.
- Attention gates cost ~1.7% step time and bigram a further ~2.5%. If neither earns that
  back, they should come out of the baseline — but that is a decision to make after the
  lr sweeps, not now.
