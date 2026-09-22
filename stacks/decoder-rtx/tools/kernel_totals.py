#!/usr/bin/env python3
"""Per-step GPU kernel totals from a torch-profiler trace, and the delta between two.

The step a kernel belongs to is decided by the CPU op that launched it -- the runtime
launch event carrying the same correlation id -- not by the trace's gpu_user_annotation,
which on these traces covers only the tail of the step.

    python kernel_totals.py control.json.gz                 # one step's kernels, by name
    python kernel_totals.py control.json.gz arm.json.gz     # ... with the arm's delta
"""
import gzip, json, sys
from collections import defaultdict

GPU_CATS = ("kernel", "gpu_memset", "gpu_memcpy")
LAUNCH_CATS = ("cuda_runtime", "cuda_driver")


def per_step(path):
    """{kernel name: ms per step}, the step count, and the mean step wall-ms."""
    events = json.load(gzip.open(path) if path.endswith(".gz") else open(path))["traceEvents"]

    # The CPU-side ProfilerStep annotations, in order.
    steps = sorted((e["ts"], e["ts"] + e.get("dur", 0.0), e["name"])
                   for e in events
                   if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("ProfilerStep#"))
    assert steps, f"no CPU-side ProfilerStep annotation in {path}"

    def which(ts):
        for i, (t0, t1, _) in enumerate(steps):
            if t0 <= ts < t1:
                return i
        return None

    launch_step = {}   # correlation id -> step index
    for e in events:
        if e.get("cat") in LAUNCH_CATS and "correlation" in e.get("args", {}):
            launch_step[e["args"]["correlation"]] = which(e["ts"])

    totals, counts = defaultdict(float), defaultdict(int)
    for e in events:
        if e.get("cat") not in GPU_CATS or e.get("ph") != "X":
            continue
        if launch_step.get(e.get("args", {}).get("correlation")) is None:
            continue
        totals[e["name"]] += e["dur"] / 1000.0
        counts[e["name"]] += 1

    n = len(steps)
    wall = sum(t1 - t0 for t0, t1, _ in steps) / 1000.0 / n
    return {k: v / n for k, v in totals.items()}, {k: v / n for k, v in counts.items()}, n, wall


def short(name, width=78):
    name = name.split("(")[0]
    return name if len(name) <= width else name[:width - 1] + "…"


paths = sys.argv[1:]
assert paths, __doc__
base, base_n, n0, wall0 = per_step(paths[0])
others = [per_step(p) for p in paths[1:]]

print(f"{paths[0]}: {n0} steps, {wall0:.1f} ms/step wall, {sum(base.values()):.1f} ms/step of kernels")
for p, (t, _, n, wall) in zip(paths[1:], others):
    print(f"{p}: {n} steps, {wall:.1f} ms/step wall, {sum(t.values()):.1f} ms/step of kernels")
print()

names = set(base)
for t, _, _, _ in others:
    names |= set(t)
rows = sorted(names, key=lambda k: -max([base.get(k, 0.0)] + [t.get(k, 0.0) for t, _, _, _ in others]))

head = f"{'ms/step':>9} {'calls':>6}" + "".join(f"{'delta':>9}" for _ in others) + "  kernel"
print(head)
print("-" * len(head))
for k in rows:
    b = base.get(k, 0.0)
    if b < 0.05 and all(t.get(k, 0.0) < 0.05 for t, _, _, _ in others):
        continue
    c = base_n.get(k, 0.0) or max(cn.get(k, 0.0) for _, cn, _, _ in others)
    line = f"{b:>9.2f} {c:>6.0f}" + "".join(f"{t.get(k, 0.0) - b:>+9.2f}" for t, _, _, _ in others)
    print(f"{line}  {short(k)}")
