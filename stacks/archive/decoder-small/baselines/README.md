# Baselines

The history of DecoderStack-small, in the spirit of modded-nanogpt's `records/`
folder: one dated, named folder per baseline, holding a frozen copy of the code
that produced it plus everything it needed to run.

| Baseline | Date | What changed | Result |
|---|---|---|---|
| _none frozen yet_ | | | |

**No baseline has been cut for this track yet** — there is no full-horizon run
with a reported val bpb or CORE number to pin one to. The working copy at the
folder root is the only version, unchanged since `cd4b8ba` (2026-03-13). Cut the
first baseline when there's a result to attach to it.

## reference/

Not our baselines — nanochat's own d12 speedrun logs, kept as the external
reference this track is measured against:

| | |
|---|---|
| `speedrun_nanochat-d12-fp8.log` | nanochat d12, fp8 |
| `20260214-170301-nanochat-d12-bs32-fp8.log` | nanochat d12, batch size 32, fp8 |

⚠️ **These are gitignored** (`*.log`) and have never been tracked, so they exist
only on machines that already had them — a fresh clone gets an empty
`reference/`. Force-add them, or move them to `agent-ops`, if they're worth
keeping around.

## Adding a baseline

1. `baselines/YYYYMMDD_name/` — the date is the commit that froze it, the name
   says what changed relative to the previous baseline.
2. Copy in the training script **at that commit** and every file it imports
   (`triton_kernels.py`, `generation_small.py`), plus the run script. Don't copy
   from the working tree if it has already drifted.
3. Write a `README.md`: the pinning commit, the files, the config, the measured
   result with its error bars, and what distinguishes it from the last baseline.
4. Add a row to the table above.

The baseline-vs-record policy — and why a single-GPU speedrun is worth the
statistical awkwardness — is in the repo-root [`BASELINES.md`](../../../BASELINES.md).
