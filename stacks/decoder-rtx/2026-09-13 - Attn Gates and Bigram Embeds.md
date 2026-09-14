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
| B | + attention output gates | 0.900147 | −147 | 1.550 | 86.0 GB |
| C | + gates + bigram embeds (lr 0.3) | 0.900492 | −492 | 1.565 | 88.2 GB |
| D | C with bigram lr 0.9 | 0.900460 | −460 | 1.566 | 88.2 GB |

Slack = (0.900000 − val_bpb) × 1e6. Stated noise floor is ±35; one step is worth ~70.

Full val_bpb trajectories (the box was torn down after run D, so these are kept here;
the runs are also in wandb under project `decoderstack_rtx`):

| step | A control | B gates | C bigram lr .3 | D bigram lr .9 |
|------|-----------|---------|----------------|----------------|
| 0    | 1.841550 | 1.841550 | 1.841546 | 1.841546 |
| 125  | 1.168177 | 1.165672 | 1.160861 | 1.170663 |
| 250  | 1.065258 | 1.064685 | 1.064310 | 1.071816 |
| 375  | 1.028779 | 1.028428 | 1.030188 | 1.029369 |
| 500  | 0.986016 | 0.986253 | 0.986629 | 0.986430 |
| 625  | 0.956121 | 0.956116 | 0.956268 | 0.956244 |
| 750  | 0.933081 | 0.933099 | 0.933449 | 0.933210 |
| 875  | 0.915088 | 0.915057 | 0.915435 | 0.915419 |
| 1000 | 0.902273 | 0.902324 | 0.902678 | 0.902622 |
| 1035 | 0.900104 | 0.900147 | 0.900492 | 0.900460 |

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

   **Run D result:** 3× the lr moved the final number by +32 points (−492 → −460), inside
   the ±35 noise floor. Its early trajectory was actually *worse* than C's (step 125:
   1.170663 vs 1.160861) before converging back by step 375. So this particular lr change
   did not recover the gap. That is one alternative value, not a sweep — a larger change,
   or the lr not being the operative difference at all, both remain open.
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
6. **Batch-size schedule.** modded ramps the batch in thirds of its scheduled steps —
   131,072 → 262,144 → 393,216 tokens/step — with the lr stepping alongside it
   (`(16/8)**0.6`, `(24/8)**0.5`). decoder-rtx runs flat at 524,288. So their first 520
   steps are at a quarter of our batch size, and they take 1600 optimizer steps to our
   1035. A zero-init sparse table gets many more updates per token early on there, which
   is plausibly when it most needs to move. This is independent of the lr *value* and was
   not varied in any run here.

### Confounds that make the PR a loose reference

decoder-rtx and modded-nanogpt differ in ways that have nothing to do with this feature:

- **Vocab** — 32,768 vs 50,304
- **Dataset** — Climbmix vs FineWeb
- **Loss target** — 0.900 val bpb vs 3.28 CE
- **Token budget** — decoder-rtx trains on **~543M** tokens (1035 × 2 × 262,144; the
  planner reports 542,900,224 actually trained). modded post-PR trains on **~425M**
  (520 × 131,072 + 520 × 262,144 + 560 × 393,216). So **we train on ~1.28× more tokens
  than they do**, over fewer optimizer steps (1035 vs 1600).

  This kills an earlier guess that ours is the tighter budget for 126M cold parameters —
  the ratio favours us. Their post-bigram model is roughly 548M params against 425M
  tokens (**0.77 tokens/param**, matching the PR's own "more parameters than training
  tokens" remark); runs C/D are 412M params against 543M tokens (**1.32**). They took the
  win at the worse ratio, so "too many cold params for the budget" is not a live
  explanation and should not be carried forward.
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
- For bigram, the lr looks less promising than it did before run D, so the redundancy
  question (bigram distribution init vs. a learned bigram hash table) is probably the more
  informative thing to isolate next — it is the structural difference rather than a tuning
  one. Concretely: run the hash table with `embed_prior`/`head_prior` disabled and see
  whether it earns its keep when the model no longer starts with bigram statistics baked
  into the embedding and lm_head.
- **Find an init strategy for the bigram table.** The table presumably captures more than
  the raw bigram distribution — decoder-rtx already has that baked in via `embed_prior`,
  so whatever the table earns in modded is likely something else. The approach that worked
  for the bigram *init* applies here: analyse a trained model to see what the table
  actually learned, then initialise it to that instead of to zeros. Concretely — train a
  run with the table in, then look at which slots move at all vs. stay near zero, how row
  norms track pair frequency, whether learned rows align with the existing `embed_prior`
  direction for the same pair (that would measure the redundancy directly rather than
  inferring it), and whether rows cluster by anything interpretable — position in a word,
  multi-token names, common collocations. A good init would also sidestep the
  "zero-init table needs many updates to climb out" problem that the lr and batch-schedule
  differences both bear on.
- Also still untested for bigram: β2 (0.995 here vs the PR's 0.95), weight decay (10× less
  than decoder-rtx's own `value_embeds`), the batch-size ramp, and a much larger lr
  than 0.9.
- Adding 126M zero-init parameters to a 543M-token run is a real cost regardless; the PR
  itself notes the model ends up with more parameters than training tokens, which it could
  afford and this budget may not.
- Attention gates cost ~1.7% step time and bigram a further ~2.5%. If neither earns that
  back, they should come out of the baseline — but that is a decision to make after the
  lr sweeps, not now.

## 2026-09-14 update: the table's lr was the problem, in the other direction

The 09-13 runs B–D trained the table frozen (its parameters were missing from the
optimizer loops; fixed in `dc089dd`). With the table actually stepping at lr 0.9, Adam's
normalized update (~lr per element per step, whatever the gradient) put every slot's vector
at a norm of 200–290 by the end of the run against 44 for an input-embedding row, the
rarest slots largest, and the model settled `bigram_lambdas` at ~0.1 / 0.0 / 0.06 — the
table switched nearly off — finishing 0.901611 against the head-init baseline's 0.899560.
Lowering the lr, with `bigram_lambdas`' lr at 0.1 so their first gradient no longer
overshoots them to −0.26:

| table peak lr | val bpb | slack |
|---|---|---|
| none (baseline) | 0.899560 | +440 |
| 0.9 | 0.901611 | −1,611 |
| 0.1 | 0.897176 | +2,824 |
| 0.03 | 0.895198 | +4,802 |
| 0.01 | 0.895067 | +4,933 |

Ahead at every validation from step 125, under 0.90 by step 875, +1.1% step time; the
attention gates are set aside on this branch (`d75f72d`, the last commit with them is
`e8e6870`). A hit-rows-only Adam and a count-derived table init did not help. Provenance:
`agent-ops-stacks/decoder-rtx/2026-09-14_0744am_bigram-hash-study/`.
