"""Stack the GPU kernel strip of one or more torch-profiler traces on one time axis, and
colour every kernel by *what it computes* rather than by what family of kernel it is.

Reads the `.json.gz` chrome traces `train_stack.py` writes with PROFILE = True, keeps only
the device streams, aligns every trace at t = 0, and draws the perfetto picture -- runs
stacked so the same kernel lands in the same column.

Writes an `.html` (hover for the kernel name, its role and its duration, plus a per-kernel
table with deltas) and a `.png` of the same strips. **The PNG carries its own text** -- the
title, the region band and the legend are drawn with a 5x7 bitmap font -- because reading
the PNG is the only way an agent sees the picture at all: no browser, no display, no
extension, over ssh on a GPU box.

    python trace_strip.py prof.json.gz --map kernel_map.json -o strip.html
    python trace_strip.py base.json.gz arm.json.gz --map kernel_map.json --region 'fwd.layer[2]'
    python trace_strip.py prof.json.gz --emit-map draft_map.json     # the scaffold to fill in

--- The map --------------------------------------------------------------------------

Kernel *names* do not say what a kernel is doing: W_Q, W_K, W_V and W_O are one name at
one duration (~620 us), whether that name is `cutlass::Kernel2` or, once the autotune
cache picks a triton template for the shape, `triton_tem_fused_mm_select_transpose_N`.
Position does say, so the map is read positionally, anchored on the flash-attention
kernels -- the one clearly-labelled landmark, with a known thing before and after it.

`segment()` cuts the step into regions off those anchors (`fwd.input`, `fwd.layer[i]`,
`fwd.head`, `bwd.head`, `bwd.layer[i]`, `optim`), and the map gives each region a
`offset from its anchor -> [roles]` table. One table covers the plain forward layers;
layers with value embeddings get their own, since they carry three extra kernels, and
layer 1 -- the first of them, where the tables are materialised -- gets its own again.

A kernel with several roles is drawn as horizontal bands, one per role, because inductor
fuses (the layer-entry kernel really is the residual add *and* the bigram read *and* the
x0 gate *and* the RMSNorm). Anything the map does not cover is drawn dark grey and counted
in the coverage line, so the gaps are visible rather than silently plausible.

    python trace_strip.py prof.json.gz --emit-map draft.json

writes the scaffold: every region, every offset, the kernel that sits there and its
duration in a `seen` field, and an empty `roles` list to fill in. Only `roles` is read
back, so the `seen` fields stay as the note of what the map was built against.
"""

import argparse
import binascii
import gzip
import html
import json
import math
import os
import re
import struct
import sys
import zlib
from collections import defaultdict

GPU_CATS = ("kernel", "gpu_memset", "gpu_memcpy")
LAUNCH_CATS = ("cuda_runtime", "cuda_driver")

# ==============================================================================
# § The palette
# ==============================================================================

# role -> (label, hue, saturation %, lightness % forward, lightness % backward).
# Backward ops are the same hue a shade darker than their forward counterpart, so a
# column reads as "the same thing, going the other way".
ROLES = {
    "rmsnorm":  ("RMSNorm",        2,  72, 63, 40),   # reds -- every RMS normalisation
    "qknorm":   ("QK norm",      355,  62, 70, 46),   # ... its q/k sibling, a touch pinker
    "rope":     ("RoPE",         344,  28, 44, 30),   # in the q/k family but plainly not a norm
    "gate":     ("sigmoid gate",   0,   0, 62, 40),   # dark greys
    "ve":       ("value embeds",  48,  88, 62, 40),   # yellow
    "pair":     ("pair values",   60,  55, 55, 35),   # yellow-olive: the pair table sibling
    "bigram":   ("bigram hash",   26,  52, 42, 28),   # brown
    "mlp_in":   ("MLP in",       216,  72, 62, 40),   # blues
    "mlp_act":  ("relu^2",       200,  62, 54, 34),
    "mlp_out":  ("MLP out",      228,  66, 70, 46),
    "fa":       ("FlashAttn",    278,  58, 64, 42),   # purples
    "wq":       ("W_Q",          320,  72, 64, 42),   # magenta
    "wk":       ("W_K",          136,  56, 58, 36),   # green
    "wv":       ("W_V",           30,  88, 60, 38),   # orange
    "wo":       ("W_O",          186,  62, 58, 37),   # teal -- not in the brief, needed one
    "embed":    ("token embed",   96,  42, 52, 33),
    "lmhead":   ("LM head",      250,  30, 60, 38),   # quiet: the block is unmistakable by size
    "loss":     ("loss / CE",    250,  16, 46, 30),
    "smear":    ("smear",         82,  40, 52, 33),
    "table":    ("table rows",   166,  36, 48, 32),   # sort / compaction / scatter
    "optim":    ("optimizer",      0,   0, 52, 52),
    "muon":     ("Muon step",     18,  85, 66, 50),   # coral -- momentum, norm, variance reduction, the update
    "muon_pe":  ("Muon Polar Express", 348, 70, 58, 42),   # rose -- its Gram / A@A / combine GEMMs
    "adam":     ("AdamW step",    42,  80, 60, 46),   # amber
    "copy":     ("copy / cast",  210,  14, 46, 46),
    "unmapped": ("unmapped",       0,   0, 26, 26),
}

# Dark mode. The strip is the point, so everything else stays quiet.
INK        = (222, 222, 216)
INK_DIM    = (129, 132, 138)
BG         = (18, 20, 24)
LANE_BG    = (31, 34, 40)
GRID       = (54, 58, 66)
EDGE_MINOR = (72, 78, 88)     # layer boundaries
EDGE_MAJOR = (196, 172, 92)   # fwd | bwd | optim


def rgb_of(role, phase):
    """(r, g, b) for a role, shaded by whether it runs in the forward or the backward."""
    _, hue, sat, light_fwd, light_bwd = ROLES.get(role, ROLES["unmapped"])
    h, s, l = hue / 360.0, sat / 100.0, (light_bwd if phase == "bwd" else light_fwd) / 100.0
    c = (1 - abs(2 * l - 1)) * s
    xc = c * (1 - abs((h * 6) % 2 - 1))
    r, g, b = [(c, xc, 0), (xc, c, 0), (0, c, xc),
               (0, xc, c), (xc, 0, c), (c, 0, xc)][int(h * 6) % 6]
    m = l - c / 2
    return tuple(round((v + m) * 255) for v in (r, g, b))


# ==============================================================================
# § Reading the trace
# ==============================================================================

def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)["traceEvents"]


def short_name(name):
    """Drop template arguments and the `void` return so kernels stay recognisable."""
    for _ in range(4):
        name = re.sub(r"<[^<>]*>", "", name)
    name = re.sub(r"^void\s+", "", name)
    name = re.sub(r"\(.*\)$", "", name)
    name = name.replace("(anonymous namespace)::", "")
    return name.strip() or "?"


def kernel_step(events):
    """{correlation id: step index} from the CPU op that launched each kernel.

    The CPU-side ProfilerStep annotations bracket the launches; a kernel belongs to the
    step whose annotation contains the launch, whenever the kernel itself ran. The
    device-side `gpu_user_annotation` span covers only 36 ms of a 656 ms step, so
    cropping to it would drop five kernels in six.
    """
    steps = sorted((e["ts"], e["ts"] + e.get("dur", 0.0), int(e["name"].split("#")[1]))
                   for e in events
                   if e.get("cat") == "user_annotation"
                   and str(e.get("name", "")).startswith("ProfilerStep#"))
    assert steps, "no CPU-side ProfilerStep annotation in this trace"

    def which(ts):
        for t0, t1, n in steps:
            if t0 <= ts < t1:
                return n
        return None

    return {e["args"]["correlation"]: which(e["ts"])
            for e in events
            if e.get("cat") in LAUNCH_CATS and "correlation" in e.get("args", {})}, \
           [n for _, _, n in steps]


def collect(path, step, cats):
    """(step number, [(t_ms, dur_ms, stream, name)]) for one trace, sorted, t=0 at the start."""
    events = load_trace(path)
    launched_in, step_ids = kernel_step(events)
    n = step if step is not None else max(step_ids)

    picked = []
    for e in events:
        if e.get("cat") not in cats or e.get("ph") != "X":
            continue
        if launched_in.get(e.get("args", {}).get("correlation")) != n:
            continue
        picked.append((e["ts"], e.get("dur", 0.0), e["tid"], e["name"]))
    if not picked:
        sys.exit(f"{path}: no GPU events found (step {n}, cats {','.join(cats)})")
    picked.sort()

    zero = picked[0][0]
    return n, [((ts - zero) / 1000.0, dur / 1000.0, tid, name) for ts, dur, tid, name in picked]


# ==============================================================================
# § Segmenting the step on the flash-attention anchors
# ==============================================================================

def segment(names, model, spans):
    """[(region key, label, first idx, last idx, anchor idx, phase)] over the whole step.

    The forward's `flash_fwd_kernel`s and the backward's `flash_bwd_dot_do_o_kernel`s are
    the anchors: one per layer, in source order forward and in reverse backward. A layer's
    extent is the anchor plus the offset window the map declares for it, so the map and
    the picture cannot drift apart.
    """
    fwd_at = [i for i, nm in enumerate(names) if "flash_fwd_kernel" in nm]
    bwd_at = [i for i, nm in enumerate(names) if "flash_bwd_dot_do_o" in nm]
    n_layers = model["n_layers"]
    assert len(fwd_at) == n_layers, f"{len(fwd_at)} forward FA kernels, expected {n_layers}"
    assert len(bwd_at) == n_layers, f"{len(bwd_at)} backward FA kernels, expected {n_layers}"

    ve = set(model.get("ve_layers", []))
    out = []

    def window(key, layer, anchor, prev_anchor):
        """(resolved key, lo offset, hi offset) for one layer.

        The map is consulted most specific first -- `fwd.layer.1`, then `fwd.layer.ve`,
        then `fwd.layer` -- because the layers really are not interchangeable: the value
        embedding layers carry three extra kernels, and layer 1, the first of them,
        carries two more again where the VE and pair-code tables are first materialised.

        With no map yet -- how the scaffold is first emitted -- layers tile instead: each
        runs from just after the previous anchor of its kind up to and including its own,
        so every kernel between two anchors is offered for labelling exactly once.
        """
        for candidate in (f"{key}.{layer}", key + ".ve" if layer in ve else key, key):
            table = spans.get(candidate)
            if table:
                offsets = [int(k) for k in table if not k.startswith("_")]
                return candidate, min(offsets), max(offsets)
        fallback = f"{key}.ve" if layer in ve else key
        return fallback, (0 if prev_anchor is None else prev_anchor - anchor + 1), 0

    for layer, anchor in enumerate(fwd_at):
        key, lo, hi = window("fwd.layer", layer, anchor, fwd_at[layer - 1] if layer else None)
        out.append((key, f"L{layer}", anchor + lo, anchor + hi, anchor, "fwd"))
    for n, anchor in enumerate(bwd_at):
        layer = n_layers - 1 - n
        key, lo, hi = window("bwd.layer", layer, anchor, bwd_at[n - 1] if n else None)
        out.append((key, f"L{layer}", anchor + lo, anchor + hi, anchor, "bwd"))
    out.sort(key=lambda r: r[2])

    # The unclaimed stretches. Only three are real regions of their own: the input
    # before the first forward layer, the head between the forward and the backward, and
    # the optimizer after the last backward layer. Everything else is slack between two
    # layers of the same phase -- the layers are not all the same length -- and belongs to
    # the layer it trails, so it is swallowed rather than named.
    fwd_end = max(r[3] for r in out if r[5] == "fwd")
    bwd_start = min(r[2] for r in out if r[5] == "bwd")

    gaps, cursor = [], 0
    for _, _, first, last, _, _ in out:
        if first > cursor:
            gaps.append((cursor, first - 1))
        cursor = max(cursor, last + 1)
    if cursor < len(names):
        gaps.append((cursor, len(names) - 1))

    named, swallow = [], {}
    for first, last in gaps:
        if last < out[0][2]:
            named.append(("fwd.input", "input", first, last, first, "fwd"))
        elif first > out[-1][3]:
            named.append(("optim", "optimizer", first, last, first, "optim"))
        elif fwd_end < first and last < bwd_start:
            # The LM head: its forward runs to the loss reduction, and the map says how
            # many kernels that is; the rest is the LM head backward.
            cut = first + (model.get("head_fwd_kernels") or (last - first) // 2)
            named.append(("fwd.head", "LM head", first, min(cut - 1, last), first, "fwd"))
            named.append(("bwd.head", "LM head bwd", cut, last, cut, "bwd"))
        else:
            swallow[first - 1] = last          # trailing slack, folded into the layer above

    out = [(key, label, first, swallow.get(last, last), anchor, phase)
           for key, label, first, last, anchor, phase in out]
    return sorted(out + named, key=lambda r: r[2])


def roles_for(regions, spans, by_name, names, count):
    """[[role, ...]] per kernel index, from the map's offset tables.

    Two escape hatches, both there because the step is not as regular as it looks. A
    region's `_default` covers every offset it does not name, which is what keeps the
    optimizer's ~400 kernels from needing 400 lines. And an entry may carry an `expect`
    substring: the roles apply only if the kernel sitting at that offset really is the one
    the map was written against. **Backward layers do not all have the same kernel count**
    -- layer 0 spends two GEMMs on the W_O backward where layer 9 spends one -- so without
    the check a fixed offset would confidently mislabel half the layers. A failed check
    leaves the kernel grey, which is the honest answer and shows up in the coverage count.
    """
    def positional(entry, name):
        """Roles from one offset entry: a dict, or a list of candidates tried in order."""
        for candidate in (entry if isinstance(entry, list) else [entry]):
            expect = candidate.get("expect")
            if not expect or expect in name:
                return list(candidate.get("roles") or [])
        return None                        # nothing the map describes sits here

    out = [[] for _ in range(count)]
    for key, _, first, last, anchor, phase in regions:
        table = spans.get(key, {})
        fallback = (table.get("_default") or {}).get("roles") or []
        for i in range(max(first, 0), min(last + 1, count)):
            entry = table.get(str(i - anchor))
            roles = positional(entry, names[i]) if entry is not None else None
            if roles is None:
                roles = next((list(v) for needle, v in by_name.items()
                              if needle in names[i]), list(fallback))
            out[i] = roles
    return out


# ==============================================================================
# § A 5x7 bitmap font, so the PNG can carry its own labels
# ==============================================================================

# Each glyph is five column bytes, bit 0 the top row. Uppercase only -- lowercase is
# folded up, which keeps the table half the size and stays legible at this scale.
FONT = {
    " ": "0000000000", "-": "0808080808", "+": "08083E0808", ".": "0000600000",
    ":": "0000240000", ",": "0050300000", "/": "4030080601", "|": "0000770000",
    "=": "1414141414", "*": "14083E0814", "#": "147F147F14", "%": "2313086462",
    "(": "001C224100", ")": "0041221C00", "[": "007F414100", "]": "0041417F00",
    "<": "0814224100", ">": "0041221408", "?": "0201510906", "!": "00005F0000",
    "_": "4040404040", "^": "0402010204", "'": "0000070000", '"': "0007000700",
    "0": "3E5149453E", "1": "00427F4000", "2": "4261514946", "3": "2141454B31",
    "4": "1814127F10", "5": "2745454539", "6": "3C4A494930", "7": "0171090503",
    "8": "3649494936", "9": "064949291E",
    "A": "7E1111117E", "B": "7F49494936", "C": "3E41414122", "D": "7F4141221C",
    "E": "7F49494941", "F": "7F09090901", "G": "3E4149497A", "H": "7F0808087F",
    "I": "00417F4100", "J": "2040413F01", "K": "7F08142241", "L": "7F40404040",
    "M": "7F020C027F", "N": "7F0408107F", "O": "3E4141413E",
    "P": "7F09090906", "Q": "3E41415E60", "R": "7F09192946", "S": "4649494931",
    "T": "01017F0101", "U": "3F4040403F", "V": "1F2040201F",
    "W": "7F2018207F", "X": "6314081463", "Y": "0708700807", "Z": "6151494543",
}
GLYPH_W, GLYPH_H = 6, 7          # 5 columns plus one of spacing


def text_fills(s, x, y, rgb, scale=1):
    """[(x0, y0, x1, y1, rgb)] painting `s` with its top-left at (x, y)."""
    out = []
    for ch in str(s).upper():
        cols = FONT.get(ch, FONT["?"])
        for col in range(5):
            bits = int(cols[col * 2:col * 2 + 2], 16)
            for row in range(GLYPH_H):
                if bits >> row & 1:
                    px, py = x + col * scale, y + row * scale
                    out.append((px, py, px + scale, py + scale, rgb))
        x += GLYPH_W * scale
    return out


def text_w(s, scale=1):
    return len(str(s)) * GLYPH_W * scale


def write_png(path, width, height, fills):
    """Paint `fills` -- (x0, y0, x1, y1, rgb) in pixels, in order -- and write an RGB PNG.

    Hand-rolled on zlib so the tool needs nothing installed: the agent reading the picture
    on a GPU box has no browser and no display.
    """
    rows = [bytearray(bytes(BG) * width) for _ in range(height)]
    for x0, y0, x1, y1, rgb in fills:
        x0, x1 = max(0, int(x0)), min(width, int(math.ceil(x1)))
        if x1 <= x0:                      # a kernel thinner than a pixel still gets one
            x0, x1 = min(max(x0, 0), width - 1), min(max(x0, 0) + 1, width)
        span = bytes(rgb) * (x1 - x0)
        for row in rows[max(0, int(y0)):min(height, int(y1))]:
            row[x0 * 3:x1 * 3] = span

    chunk = lambda tag, data: (struct.pack(">I", len(data)) + tag + data +
                               struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"".join(b"\x00" + bytes(r) for r in rows), 6))
                + chunk(b"IEND", b""))


# ==============================================================================
# § The map file
# ==============================================================================

DEFAULT_MODEL = {"n_layers": 11, "ve_layers": [], "full_ctxt_layers": [],
                 "backout_layer": None, "head_fwd_kernels": None}


def load_map(path):
    if not path:
        return DEFAULT_MODEL, {}, {}
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    model = dict(DEFAULT_MODEL, **doc.get("model", {}))
    return model, doc.get("regions", {}), doc.get("by_name", {})


def emit_map(path, names, durs, regions, spans, model, source):
    """Write the scaffold: every region, every offset, what sits there, roles to fill in."""
    out = {"note": f"kernel map for {os.path.basename(source)} -- fill in each `roles` list."
                   " `seen` is what the offset held when the scaffold was written and is"
                   " never read back. Roles: " + ", ".join(sorted(ROLES)),
           "model": model, "regions": {}}
    for key, label, first, last, anchor, phase in regions:
        table = out["regions"].setdefault(key, {})
        for i in range(max(first, 0), min(last + 1, len(names))):
            off = str(i - anchor)
            if off in table:
                continue                  # the first layer of a kind writes the template
            known = spans.get(key, {}).get(off)
            if isinstance(known, list):
                table[off] = known            # a list of candidates: carry it through whole
                continue
            table[off] = {"roles": list((known or {}).get("roles") or []),
                          "seen": f"{names[i][:64]} ({durs[i] * 1000:.0f} us)"}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    n_off = sum(len(v) for v in out["regions"].values())
    print(f"{path}: {len(out['regions'])} regions, {n_off} offsets to label")


# ==============================================================================
# § Drawing
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="chrome traces (.json or .json.gz), baseline first")
    ap.add_argument("-o", "--out", default="trace_strip.html")
    ap.add_argument("--map", help="kernel map JSON (see --emit-map)")
    ap.add_argument("--emit-map", metavar="PATH", help="write a scaffold map and exit")
    ap.add_argument("--labels", help="comma-separated row labels (default: file names)")
    ap.add_argument("--step", type=int, help="ProfilerStep# to crop to (default: the last one)")
    ap.add_argument("--region", help="crop to a region key, e.g. 'fwd.layer[2]' or 'optim'")
    ap.add_argument("--window", nargs=2, type=float, metavar=("START", "END"),
                    help="ms relative to t=0 (default: the whole step)")
    ap.add_argument("--title", help="a note drawn above the strip, saying what we are looking at")
    ap.add_argument("--cats", default=",".join(GPU_CATS), help="event categories to keep")
    ap.add_argument("--width", type=int, default=1600, help="plot width in px")
    ap.add_argument("--top", type=int, default=20, help="rows in the summary table")
    ap.add_argument("--no-png", action="store_true", help="skip the .png companion")
    args = ap.parse_args()

    cats = tuple(args.cats.split(","))
    labels = args.labels.split(",") if args.labels else [os.path.basename(p) for p in args.traces]
    if len(labels) != len(args.traces):
        sys.exit("--labels count does not match the number of traces")
    model, spans, by_name = load_map(args.map)

    rows = []
    for path, label in zip(args.traces, labels):
        n, events = collect(path, args.step, cats)
        names = [short_name(nm) for _, _, _, nm in events]
        durs = [d for _, d, _, _ in events]
        regions = segment(names, model, spans)
        rows.append({"label": label.strip(), "path": path, "step": n, "events": events,
                     "names": names, "regions": regions,
                     "roles": roles_for(regions, spans, by_name, names, len(events))})

    if args.emit_map:
        r = rows[0]
        emit_map(args.emit_map, r["names"], [d for _, d, _, _ in r["events"]],
                 r["regions"], spans, model, r["path"])
        return

    # ---- the window ----
    for r in rows:
        r["shift"] = 0.0
    if args.region:
        key = args.region
        want_layer = None
        if "[" in key:
            key, want_layer = key.split("[")[0], key.split("[")[1].rstrip("]")
        stem = lambda k: ".".join(p for p in k.split(".") if not p.isdigit() and p != "ve")

        def matching(row):
            return [r for r in row["regions"] if stem(r[0]) == stem(key)
                    and (want_layer is None or r[1] == f"L{want_layer}" or r[1] == want_layer)]

        hits = matching(rows[0])
        if not hits:
            sys.exit(f"--region {args.region!r} matched nothing; regions are "
                     + ", ".join(sorted({r[0] for r in rows[0]['regions']})))
        # Each trace is shifted onto its *own* copy of the region. An arm that added or
        # dropped kernels earlier in the step has its layer 2 at a different wall time,
        # and comparing it against the baseline's clock would slide the two apart for a
        # reason that has nothing to do with the region under the microscope.
        # The alignment point is the region's anchor (its flash-attention kernel), not its first
        # kernel: what sits at the window's first offset can differ between arms when the scheduler
        # moves work around, and the anchor is the one kernel every row has in the same place.
        base_zero = rows[0]["events"][hits[0][4]][0]
        for r in rows[1:]:
            mine = matching(r)
            if mine:
                r["shift"] = r["events"][mine[0][4]][0] - base_zero
        ev = rows[0]["events"]
        t_start = ev[hits[0][2]][0]
        t_end = ev[hits[-1][3]][0] + ev[hits[-1][3]][1]
        pad = (t_end - t_start) * 0.02
        t_start, t_end = t_start - pad, t_end + pad
        # Say which layer this is and what it carries: they are not interchangeable.
        if args.title is None and len(hits) == 1 and re.fullmatch(r"L\d+", hits[0][1]):
            n_layer = int(hits[0][1][1:])
            carries = ([f"value embeds (bank {model['ve_layers'].index(n_layer)})"]
                       if n_layer in model.get("ve_layers", []) else [])                     + (["full context"] if n_layer in model.get("full_ctxt_layers", [])
                       else [f"sliding window"])                     + (["backout"] if n_layer == model.get("backout_layer") else [])
            args.title = (f"{'forward' if hits[0][5] == 'fwd' else 'backward'} layer "
                          f"{n_layer} of {model['n_layers']} -- " + ", ".join(carries))
        elif args.title is None:
            args.title = f"{hits[0][1]} ({args.region})"
    elif args.window:
        t_start, t_end = args.window
    else:
        t_start = min(e[0] for r in rows for e in r["events"])
        t_end = max(e[0] + e[1] for r in rows for e in r["events"])
    span = t_end - t_start
    if span <= 0:
        sys.exit("empty window")

    scale = args.width / span
    x = lambda t: (t - t_start) * scale

    # ---- lay the strips out ----
    TITLE_H, AXIS_H, BAND_H, LABEL_H, LANE_H, ROW_GAP, PAD = 22, 24, 13, 16, 34, 24, 14
    svg, legend, used_roles = [], {}, {}
    fills, guide_fills, over_fills = [], [], []      # under the strips / strips / over them
    y = TITLE_H + AXIS_H + PAD
    n_drawn = 0

    for r in rows:
        svg.append(f'<text class="lbl" x="0" y="{y + 11}">{html.escape(r["label"])}'
                   f'<tspan class="dim"> &#183; step {r["step"]}</tspan></text>')
        over_fills += text_fills(r["label"], 0, y + 3, INK)
        y += LABEL_H

        # The region band: one cell per region, with the layer's own features named.
        for key, label, first, last, anchor, phase in r["regions"]:
            ev = r["events"]
            if last >= len(ev) or first >= len(ev):
                continue
            rx0, rx1 = x(ev[first][0] - r["shift"]), x(ev[last][0] + ev[last][1] - r["shift"])
            if rx1 < 0 or rx0 > args.width:
                continue
            clipped = rx0 < 0                 # its start is off-window: no room to name it
            rx0, rx1 = max(rx0, 0), min(rx1, args.width)
            major = key in ("fwd.head", "bwd.head", "optim")
            edge = EDGE_MAJOR if major else EDGE_MINOR
            fills.append((rx0, y, rx1, y + BAND_H - 3, (38, 42, 50) if major else (28, 31, 37)))
            over_fills.append((rx0, y, rx0 + (2 if major else 1), y + BAND_H - 3, edge))
            # Guide line down through the lanes below.
            guide_fills.append((rx0, y + BAND_H, rx0 + (2 if major else 1), 10 ** 6, edge))

            tag = label
            if key.startswith(("fwd.layer", "bwd.layer")):
                n_layer = int(label[1:])
                feats = ("VE" if n_layer in model.get("ve_layers", []) else "") \
                        + ("+FULL" if n_layer in model.get("full_ctxt_layers", []) else "") \
                        + ("+BKOUT" if n_layer == model.get("backout_layer") else "")
                tag = label + (" " + feats if feats else "")
            if not clipped and text_w(tag) + 6 < rx1 - rx0:
                over_fills += text_fills(tag, rx0 + 4, y + 2, INK_DIM)
            svg.append(f'<rect x="{rx0:.1f}" y="{y}" width="{max(rx1 - rx0, 1):.1f}" '
                       f'height="{BAND_H - 3}" fill="{"#262a32" if major else "#1c1f25"}"/>'
                       f'<rect x="{rx0:.1f}" y="{y}" width="{2 if major else 1}" '
                       f'height="{BAND_H - 3}" fill="rgb{edge}"/>'
                       + ('' if clipped else
                          f'<text class="band" x="{rx0 + 4:.1f}" y="{y + BAND_H - 6}">'
                          f'{html.escape(tag)}</text>'))
        y += BAND_H

        # The lane.
        svg.append(f'<rect class="lane" x="0" y="{y}" width="{args.width}" height="{LANE_H}"/>')
        fills.append((0, y, args.width, y + LANE_H, LANE_BG))
        r["busy"] = 0.0
        phase_of = {}
        for key, _, first, last, _, phase in r["regions"]:
            for i in range(first, last + 1):
                phase_of[i] = phase

        for i, (t, dur, tid, name) in enumerate(r["events"]):
            t -= r["shift"]
            if t + dur < t_start or t > t_end:
                continue
            r["busy"] += min(t + dur, t_end) - max(t, t_start)
            x0, x1 = max(x(t), 0.0), min(x(t + dur), args.width)
            w = max(x1 - x0, 0.6)
            sname = r["names"][i]
            these = r["roles"][i] or ["unmapped"]
            phase = phase_of.get(i, "fwd")

            # Several roles -> horizontal bands, one per role, top to bottom.
            band_h = (LANE_H - 2) / len(these)
            titles = " + ".join(ROLES.get(role, ROLES["unmapped"])[0] for role in these)
            for b, role in enumerate(these):
                rgb = rgb_of(role, phase)
                y0 = y + 1 + b * band_h
                fills.append((x0, y0, x0 + w, y0 + band_h, rgb))
                svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" '
                           f'height="{band_h:.2f}" fill="rgb{rgb}">'
                           f'<title>{html.escape(sname)}\n{titles}\n'
                           f'{dur * 1000:.1f} us @ {t:.3f} ms &#183; #{i} &#183; {phase}</title></rect>')
                used_roles.setdefault(role, phase)
            key = (sname, tuple(these))
            legend.setdefault(key, {"roles": these, "family": titles, "t": defaultdict(float)})
            legend[key]["t"][r["label"]] += min(t + dur, t_end) - max(t, t_start)
            n_drawn += 1
        y += LANE_H + ROW_GAP

    strip_bottom = y - ROW_GAP
    guide_fills = [(a, b, c, min(d, strip_bottom), e) for a, b, c, d, e in guide_fills]

    # ---- the legend, under the strips ----
    legend_y = y
    shown = [role for role in ROLES if role in used_roles]
    col_w = args.width // max(1, min(6, len(shown)))
    for k, role in enumerate(shown):
        cx = (k % 6) * col_w
        cy = legend_y + (k // 6) * 15
        for j, phase in enumerate(("fwd", "bwd")):
            over_fills.append((cx + j * 9, cy, cx + j * 9 + 8, cy + 8, rgb_of(role, phase)))
        over_fills += text_fills(ROLES[role][0], cx + 23, cy + 1, INK_DIM)
        svg.append(f'<rect x="{cx}" y="{cy}" width="8" height="8" fill="rgb{rgb_of(role, "fwd")}"/>'
                   f'<rect x="{cx + 9}" y="{cy}" width="8" height="8" fill="rgb{rgb_of(role, "bwd")}"/>'
                   f'<text class="band" x="{cx + 22}" y="{cy + 8}">'
                   f'{html.escape(ROLES[role][0])}</text>')
    y = legend_y + ((len(shown) + 5) // 6) * 15 + 4

    # ---- the axis, at the top ----
    ticks_at = []
    tick = 10 ** (len(str(int(span))) - 1) or 1
    while span / tick > 12:
        tick *= 2
    while span / tick < 4:
        tick /= 2.0
    t = tick * (int(t_start / tick) - 1)
    while t <= t_end:
        if t >= t_start:
            guide_fills.append((x(t), TITLE_H + AXIS_H - 6, x(t) + 1, strip_bottom, GRID))
            svg.append(f'<line class="tick" x1="{x(t):.1f}" y1="{TITLE_H + AXIS_H - 6}" '
                       f'x2="{x(t):.1f}" y2="{strip_bottom}"/>')
            svg.append(f'<text class="ax" x="{x(t):.1f}" y="{TITLE_H + AXIS_H - 10}">{t:g}</text>')
            over_fills += text_fills(f"{t:g}", x(t) + 2, TITLE_H + AXIS_H - 18, INK_DIM)
            ticks_at.append(t)
        t += tick

    covered = sum(1 for r in rows[0]["roles"] if r)
    note = args.title or (args.region or "whole step")
    head = (f"{note}  |  {t_start:.1f}-{t_end:.1f} MS  |  {n_drawn} KERNELS  |  "
            f"{covered}/{len(rows[0]['roles'])} MAPPED  |  MS ->")
    over_fills += text_fills(head, 0, 4, INK)

    svg = ([f'<rect x="0" y="0" width="{args.width}" height="{y}" fill="rgb{BG}"/>'] +
           [f'<line class="guide" x1="{a:.1f}" y1="{b:.1f}" x2="{a:.1f}" y2="{d:.1f}" '
            f'stroke="rgb{e}"/>' for a, b, c, d, e in guide_fills] + svg)

    # ---- the summary table ----
    order = sorted(legend, key=lambda k: -sum(legend[k]["t"].values()))[:args.top]
    base = rows[0]["label"]
    thead = "".join(f"<th>{html.escape(r['label'])}</th>" +
                    ("" if i == 0 else "<th>&#916;</th>") for i, r in enumerate(rows))
    body = []
    for key in order:
        cells = []
        for i, r in enumerate(rows):
            v = legend[key]["t"].get(r["label"], 0.0)
            cells.append(f"<td>{v:.3f}</td>")
            if i:
                d = v - legend[key]["t"].get(base, 0.0)
                cls = "up" if d > 0.0005 else ("down" if d < -0.0005 else "flat")
                cells.append(f'<td class="{cls}">{d:+.3f}</td>')
        sw = "".join(f'<span class="sw" style="background:rgb{rgb_of(role, "fwd")}"></span>'
                     for role in legend[key]["roles"])
        body.append(f'<tr><td>{sw}<code>{html.escape(key[0][:70])}</code></td>'
                    f'<td class="dim">{html.escape(legend[key]["family"])}</td>'
                    f'{"".join(cells)}</tr>')

    totals = []
    for i, r in enumerate(rows):
        totals.append(f"<td><b>{r['busy']:.3f}</b></td>")
        if i:
            d = r["busy"] - rows[0]["busy"]
            totals.append(f'<td class="{"up" if d > 0 else "down"}"><b>{d:+.3f}</b></td>')

    out = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(' vs '.join(labels))}</title>
<style>
 body {{ font: 13px/1.45 ui-monospace, Menlo, Consolas, monospace; margin: 22px;
        background: rgb{BG}; color: rgb{INK}; }}
 h1 {{ font-size: 13px; font-weight: 600; margin: 0 0 12px; color: rgb{INK}; }}
 .dim {{ fill: rgb{INK_DIM}; color: rgb{INK_DIM}; font-weight: 400; }}
 .lbl {{ font: 600 12px ui-monospace, monospace; fill: rgb{INK}; }}
 .band {{ font: 9px ui-monospace, monospace; fill: rgb{INK_DIM}; }}
 .ax {{ font: 10px ui-monospace, monospace; fill: rgb{INK_DIM}; text-anchor: middle; }}
 .lane {{ fill: rgb{LANE_BG}; }}
 .tick {{ stroke: rgb{GRID}; stroke-width: 1; }}
 .guide {{ stroke-width: 1; }}
 table {{ border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
 th, td {{ padding: 2px 9px; text-align: right; border-bottom: 1px solid #2a2e36; }}
 th:first-child, td:first-child {{ text-align: left; }}
 th {{ font-weight: 600; color: rgb{INK_DIM}; }}
 code {{ font-size: 11px; }}
 .sw {{ display: inline-block; width: 9px; height: 9px; margin-right: 3px; border-radius: 2px; }}
 .up {{ color: #e5808a; }} .down {{ color: #5fbf7f; }} .flat {{ color: rgb{INK_DIM}; }}
 tfoot td {{ border-top: 2px solid #3a3f49; border-bottom: none; }}
</style>
<h1>{html.escape(head)}</h1>
<svg width="{args.width}" height="{y}" font-family="ui-monospace, monospace">
{chr(10).join(svg)}
</svg>
<table>
<thead><tr><th>kernel</th><th>role</th>{thead}</tr></thead>
<tbody>
{chr(10).join(body)}
</tbody>
<tfoot><tr><td><b>kernel-ms in window</b></td><td></td>{''.join(totals)}</tr></tfoot>
</table>
"""
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)

    png = re.sub(r"\.html?$", "", args.out) + ".png"
    if not args.no_png:
        write_png(png, args.width, int(y), guide_fills + fills + over_fills)

    print(f"{args.out}: {n_drawn} kernels, {t_start:.2f} to {t_end:.2f} ms"
          + ("" if args.no_png else f"\n{png}: {args.width}x{int(y)} -- carries its own text"))
    for r in rows:
        miss = sum(1 for role in r["roles"] if not role)
        print(f"  {r['label']:<26} step {r['step']:<4} {r['busy']:8.3f} kernel-ms  "
              f"{len(r['roles']) - miss}/{len(r['roles'])} mapped")
    print(f"  gridlines at {', '.join(f'{v:g}' for v in ticks_at)} ms")


if __name__ == "__main__":
    main()
