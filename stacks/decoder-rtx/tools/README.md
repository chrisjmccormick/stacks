# decoder-rtx / tools

Offline analysis of the torch-profiler traces `train_stack.py` writes with `PROFILE = True`.
Both tools are **stdlib-only** — no torch, no numpy, no browser — so they run on a laptop or
over ssh on the box.

These are development tools, not part of a timed run. `AGENTS.md` rule 2 ("all code in
`train_stack.py` and `utils.py`") is about what the submitted script contains; nothing here
is imported by it.

| tool | what it answers |
|---|---|
| `kernel_totals.py` | *Where did the milliseconds go, and what moved?* Per-kernel ms/step, and the delta between two traces. |
| `trace_strip.py` | *What is the step actually doing, in order?* The kernel strip drawn to scale, coloured by what each kernel computes, several traces stacked on one time axis. |
| `kernel_map.json` | The map `trace_strip.py` colours from — which part of the model each kernel belongs to. |

```bash
python kernel_totals.py control.json.gz arm.json.gz

python trace_strip.py prof.json.gz --map kernel_map.json -o strip.html
python trace_strip.py base.json.gz arm.json.gz --map kernel_map.json \
    --labels "fold,arm" --region 'bwd.layer[5]'
```

Each `trace_strip.py` call writes an `.html` (hover for a kernel's name, role and duration,
plus a per-kernel delta table) and a `.png`. **The PNG carries its own text** — title, axis,
region band and legend, in a 5x7 bitmap font — because it is the only form an agent can read
over ssh.

## The map, and why it is positional

Kernel *names* do not say what a kernel is computing. `W_Q`, `W_K`, `W_V` and `W_O` are one
name at one duration (~620 µs), and which name that is flips with the autotune cache —
`cutlass::Kernel2` when the shape stays on cuBLAS, `triton_tem_fused_mm_select_transpose_N`
when inductor's template wins it. Position does say. So the map is read **positionally**,
anchored on the flash-attention kernels: the one clearly-labelled landmark in the step, with
a known thing either side of it.

`segment()` cuts the step into regions off those anchors — `fwd.input`, `fwd.layer[i]`,
`fwd.head`, `bwd.head`, `bwd.layer[i]`, `optim` — and each region gets an
`offset from its anchor → [roles]` table. The tables are consulted most specific first
(`fwd.layer.1`, then `fwd.layer.ve`, then `fwd.layer`), because the layers are genuinely not
interchangeable.

**Every entry carries an `expect` substring**, and applies only if the kernel sitting at that
offset really is the one the map was written against. A failed check leaves the kernel dark
grey and counts against the coverage line. That is the whole design: a map that guessed would
look complete and be wrong, and there is no way to tell those apart by looking at the picture.
**Grey is the honest answer, and it is the thing to go fix.**

## Keeping the map current — what is automatic and what is not

The model changes every week, so the map goes stale by design. It is built to make that cheap
rather than to avoid it:

- **The prior map does the bulk, and checks itself.** Wherever the kernel sequence is
  unchanged, the roles apply. Wherever the model moved, `expect` fails and the kernel goes
  grey. You are labelling the delta, not a thousand kernels.
- **The residue is a manual read.** `--emit-map` writes a scaffold — every region, every
  offset, the kernel that actually sits there with its duration in a `seen` field, and an
  empty `roles` list. Fill in `roles` by reading it against `train_stack.py`; only `roles` is
  read back, so `seen` stays as the note of what the map was built against.
- **No script infers roles.** Nothing here tries to. The value is in a human or an agent
  having read the source next to the trace, and in the `expect` guards preserving that work
  across the next several model changes.

```bash
python trace_strip.py prof.json.gz --map kernel_map.json --emit-map draft_map.json
```

When the layer count or layout changes, update `model` in `kernel_map.json` first —
`n_layers`, `ve_layers`, `full_ctxt_layers`, `backout_layer` — from `StackConfig`. The FA
anchor count is asserted against `n_layers`, so a mismatch fails loudly rather than
mis-segmenting.

## What the map currently covers

Built against **baseline 5** (`sep-28-baseline`: 8 layers, value embeds at 1/2/5/6/7, full
context at 3/4/7, backout at 4, the FFN width ramp 2,2,3,3,5,5,6,6), from
`inl-00-parent-prof_trace.json.gz`, step 13, 1,027 kernels / 418.3 ms.

| region | state |
|---|---|
| `fwd.input`, `fwd.layer[0..7]`, `fwd.head`, `bwd.head` | **complete** — zero grey, every kernel verified against `train_stack.py`'s forward |
| `bwd.layer[0..7]` | the **spine** — the FA triple, the MLP backward pair, the norm backward, `W_O`, and (in VE layers) both gate backwards and both table gradients |
| `optim` | one `_default` role, ~475 kernels in 10.1 ms |

**Left to do: the backward's QKV / gate / RoPE tail**, ~106 ms of grey across the eight
layers. `--emit-map` is the scaffold for exactly that, and the place to do it is the box,
where you can change one thing and watch which bar moves.

Two structural facts worth knowing before you read a strip:

- **A forward layer is ten kernels** — the layer-entry kernel (residual scale, `x0` gate,
  bigram read and RMSNorm, all one fused kernel), `W_Q`, `W_K`, RoPE-already-fused-into-the-q/k-norm,
  `W_V`, FA, `W_O`, the MLP input norm, `W_in` **with `relu²` fused into it**, `W_out`. A
  value-embedding layer is thirteen, adding the two tiny gate GEMMs and one kernel that does
  both table reads and applies both gates to `v`. Layer 1, the first VE layer, is fifteen:
  it is where the VE table read and `pair_code_v` are materialised.
- **Backward layers are not uniform** — 10 to 136 kernels — because the `W_in`/`W_out` width
  banks complete at different layers and the sparse table updates ride along. Layer 0 carries
  the whole gradient tail, so the optimizer does not begin until +131 past its anchor; that
  offset is in the map and **wants re-checking whenever the gradient plumbing changes**,
  since without it the tail falls into `optim` and is blanket-labelled with it.

## Provenance

`trace_strip.py` was built across
`agent-ops-stacks/decoder-rtx/2026-09-20_1233pm_headroom-and-step-time/` and
`2026-09-20_0323pm_trace-strip-kernel-map/`; `kernel_totals.py` in
`2026-09-20_0927am_gpu-fusion-opts/`. The map was rebuilt for baseline 5 in
`2026-09-22_0136pm_promote-trace-tools/`.
