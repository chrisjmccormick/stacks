# Self-contained test for convert_ckpt_to_nanochat.py. No pytest, no fixtures:
#
#   NANOCHAT_PATH=~/nanochat python utils/test_convert_ckpt_to_nanochat.py
#
# Builds a synthetic DecoderStack capture (weights + optimizer state, with real
# mantissas) at a toy config, converts it, and checks the result against a REAL
# nanochat GPT and a REAL MuonAdamW -- so the assertions are about nanochat's
# actual behaviour, not a second copy of my assumptions about it.
#
# The toy config deliberately mirrors d24's SHAPE RELATIONSHIPS rather than just
# being small: n_head * head_dim == n_embd (W_O square) and n_kv_head == 12 ==
# d_ve_gate (ve_gate square). Those are exactly the conditions under which the two
# codebases' NorMuon second-moment axes agree; a config that breaks either is
# meant to be REFUSED, and the last check covers that.
#
# Requires a nanochat checkout on the fa-varlen branch (or any branch whose GPT
# uses the modular transformer.h.N.* layout).
import math
import os
import sys

import torch

_NC = os.environ.get("NANOCHAT_PATH")
if not _NC or not os.path.isdir(os.path.join(os.path.expanduser(_NC), "nanochat")):
    sys.exit("set NANOCHAT_PATH to a nanochat checkout, e.g. NANOCHAT_PATH=~/nanochat")
sys.path.insert(0, os.path.expanduser(_NC))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# COMPUTE_DTYPE auto-detects to fp32 with no CUDA; force the bf16 a GPU box would
# pick, so the embedding dtypes we compare against are the ones a real load wants.
os.environ["NANOCHAT_DTYPE"] = "bfloat16"
# nanochat's fused kernels are @torch.compile'd and inductor's CPU backend wants a
# host compiler. Run them eager -- the bodies are plain PyTorch, so the shapes and
# the math are still exercised, just unfused.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from nanochat.gpt import GPT, GPTConfig                       # noqa: E402
from convert_ckpt_to_nanochat import (convert, convert_optimizer,  # noqa: E402
                                      ve_layers, _matrix_params)

NL, DM, NH, HD, V = 8, 192, 12, 16, 384
KV, MLP = NH * HD, 4 * DM
VE, D_VE_GATE, D_SMR = ve_layers(NL), 12, 24
NVE = len(VE)
LRS = dict(unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, scalar_lr=0.5,
           weight_decay=0.0)

upper = lambda x: (x.contiguous().view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
lower = lambda x: (x.contiguous().view(torch.int32)).to(torch.int16).view(torch.uint16)

torch.manual_seed(0)

# ---------------------------------------------------------------- synthetic capture
BANKS = {
    "input_embeds": (V, DM), "value_embeds": (NVE, V, KV), "lm_head": (V, DM),
    "W_Q": (NL, KV, DM), "W_K": (NL, KV, DM), "W_V": (NL, KV, DM), "W_O": (NL, DM, KV),
    "W_in": (NL, MLP, DM), "W_out": (NL, DM, MLP), "ve_gate": (NVE, NH, D_VE_GATE),
}
fp32_src, weights, state = {}, {}, {}
for name, shape in BANKS.items():                 # bf16 live + uint16 mantissa
    t = torch.randn(*shape)
    fp32_src[name], weights[name], state[f"{name}.mantissa"] = t, upper(t), lower(t)
for name, shape in [("resid_lambdas", (NL,)), ("x0_lambdas", (NL,)),
                    ("smear_gate", (1, D_SMR)), ("smear_lambda", (1,)),
                    ("backout_lambda", (1,))]:    # fp32 live, no mantissa
    weights[name] = torch.randn(*shape)

# AdamW moments: param-shaped, except value_embeds, which the real capture holds
# over the FLATTENED (slot * vocab) row axis. Every element gets a distinct value
# so a misrouted or misaligned shard is loud, but the SCALE stays realistic --
# with a second moment of ~1e7 the resulting update lands below fp32 epsilon and
# the "every param moved" check below fails on the (1,) scalars for reasons that
# have nothing to do with the conversion.
for t_id, (name, shape) in enumerate([("input_embeds", (V, DM)), ("lm_head", (V, DM)),
                                      ("value_embeds", (NVE * V, KV)),
                                      ("resid_lambdas", (NL,)), ("x0_lambdas", (NL,)),
                                      ("smear_gate", (1, D_SMR)), ("smear_lambda", (1,)),
                                      ("backout_lambda", (1,))]):
    for a_id, attr in enumerate(("exp_avg", "exp_avg_sq")):
        n = math.prod(shape)
        base = (t_id * 2 + a_id) * n
        state[f"{name}.{attr}"] = (
            (torch.arange(n, dtype=torch.float32) + base) * 1e-7).reshape(*shape)
# Muon moments. The factored second moment follows DecoderStack's EXPLICIT
# residual_dim: -1 for QKV / W_in / ve_gate, -2 for W_O / W_out.
SND = {"W_Q": (NL, KV, 1), "W_K": (NL, KV, 1), "W_V": (NL, KV, 1), "W_O": (NL, 1, KV),
       "W_in": (NL, MLP, 1), "W_out": (NL, 1, MLP), "ve_gate": (NVE, NH, 1)}
for name, shape in SND.items():
    b = BANKS[name]
    state[f"{name}.frst_mntm"] = torch.stack(
        [torch.full(b[1:], float(i + 1)) for i in range(b[0])])
    state[f"{name}.scnd_mntm"] = torch.stack(
        [torch.full(shape[1:], float(i + 1) * 0.5) for i in range(shape[0])])

model_data = {"step": 42, "code": "# toy\n", "weights": weights}
optim_data = {"step": 42, "t_step": 42, "state": state}

cfg = GPTConfig(sequence_len=256, vocab_size=V, n_layer=NL, n_head=NH, n_kv_head=NH,
                n_embd=DM, window_pattern="SSSL")
new_opt = lambda m: m.setup_optimizer(unembedding_lr=LRS["unembedding_lr"],
                                      embedding_lr=LRS["embedding_lr"],
                                      matrix_lr=LRS["matrix_lr"],
                                      weight_decay=LRS["weight_decay"],
                                      scalar_lr=LRS["scalar_lr"])

# ---------------------------------------------------------------- 1. weights
with torch.device("meta"):
    ref = GPT(cfg)
ref_sd = ref.state_dict()
for tag, mant in [("with mantissas", state), ("model only", None)]:
    sd = convert(model_data, mant)
    assert set(sd) == set(ref_sd), (f"key mismatch: missing "
                                    f"{sorted(set(ref_sd) - set(sd))[:4]}, extra "
                                    f"{sorted(set(sd) - set(ref_sd))[:4]}")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    want_dtype = {k: v.dtype for k, v in model.state_dict().items()}
    model.load_state_dict(sd, strict=True, assign=True)   # strict: shapes + names
    bad = {k: (sd[k].dtype, want_dtype[k]) for k in sd if sd[k].dtype != want_dtype[k]}
    assert not bad, f"dtype mismatch vs a fresh model: {bad}"
    for key, src, idx in [("lm_head.weight", "lm_head", None),
                          ("transformer.h.2.attn.c_q.weight", "W_Q", 2),
                          ("transformer.h.2.attn.c_proj.weight", "W_O", 2),
                          ("transformer.h.3.mlp.c_fc.weight", "W_in", 3),
                          (f"transformer.h.{VE[1]}.attn.ve_gate.weight", "ve_gate", 1)]:
        full = fp32_src[src] if idx is None else fp32_src[src][idx]
        want = full if mant else (upper(fp32_src[src]) if idx is None
                                  else upper(fp32_src[src])[idx]).float()
        assert torch.equal(sd[key], want), f"{tag}: {key} value mismatch"
    for key, src, idx in [("transformer.wte.weight", "input_embeds", None),
                          (f"value_embeds.{VE[0]}.weight", "value_embeds", 0)]:
        want = weights[src] if idx is None else weights[src][idx]
        assert torch.equal(sd[key], want), f"{tag}: {key} should be the bf16 live weight"
    print(f"[OK] weights, {tag}: {len(sd)} tensors, strict load, values verified")

ve_keys = sorted(k for k in ref_sd if k.endswith("ve_gate.weight"))
assert ve_keys == [f"transformer.h.{i}.attn.ve_gate.weight" for i in VE], ve_keys
print(f"[OK] ve slot->layer map: slots 0..{NVE - 1} -> layers {VE}")

# ---------------------------------------------------------------- 2. optimizer
ref_opt = new_opt(model)
param_key = {id(p): k for k, p in model.named_parameters()}
flat = [p for g in ref_opt.param_groups for p in g["params"]]
index_of_key = {param_key[id(p)]: i for i, p in enumerate(flat)}
print(f"reference optimizer: {len(ref_opt.param_groups)} groups, {len(flat)} params")

ROLE = {"W_Q": "attn.c_q", "W_K": "attn.c_k", "W_V": "attn.c_v", "W_O": "attn.c_proj",
        "W_in": "mlp.c_fc", "W_out": "mlp.c_proj"}
def key_of(entry):
    name, slot = entry
    return {"lm_head": "lm_head.weight", "input_embeds": "transformer.wte.weight",
            "smear_gate": "smear_gate.weight"}.get(name) \
        or (f"value_embeds.{VE[slot]}.weight" if name == "value_embeds"
            else f"transformer.h.{VE[slot]}.attn.ve_gate.weight" if name == "ve_gate"
            else name if slot is None
            else f"transformer.h.{slot}.{ROLE[name]}.weight")

ADAMW = ([("lm_head", None), ("input_embeds", None)]
         + [("value_embeds", j) for j in range(NVE)]
         + [("resid_lambdas", None), ("x0_lambdas", None), ("smear_gate", None),
            ("smear_lambda", None), ("backout_lambda", None)])
matrix = _matrix_params(NL, VE)
shape_of = lambda e: tuple(weights[e[0]].shape[1:])

# world_size 3 is deliberate: it leaves the Muon groups RAGGED (the ve_gate
# group hands rank 2 nothing, the attention group one padding slot), which is
# exactly what d24's 12-param ve_gate group does across 8 ranks.
for W in (1, 2, 3, 4):
    shards = [convert_optimizer(model_data, optim_data, W, r, LRS) for r in range(W)]

    for sd_r in shards:                       # structure: a real optimizer accepts it
        opt = new_opt(model)
        opt.load_state_dict(sd_r)             # raises on group/param-count mismatch
        for g in opt.param_groups:
            if g["kind"] != "muon":
                continue
            assert "momentum_buffer" in opt.state[g["params"][0]], "muon state off params[0]"
            assert not any(opt.state.get(p) for p in g["params"][1:]), "muon state leaked"

    for entry in ADAMW:                       # AdamW: shards must reassemble exactly
        pidx = index_of_key[key_of(entry)]
        full = state[f"{entry[0]}.exp_avg"]
        if entry[1] is not None:
            full = full.view(NVE, -1, full.shape[-1])[entry[1]]
        pieces = [s["state"][pidx]["exp_avg"] for s in shards]
        got = pieces[0] if pieces[0].shape == full.shape else torch.cat(pieces, 0)
        assert torch.equal(got, full), f"W={W} {key_of(entry)}: adamw state misassembled"
        assert all(s["state"][pidx]["step"] == 42 for s in shards), "step not carried"

    for shape in sorted({shape_of(e) for e in matrix}):   # Muon: chunked by group
        members = [e for e in matrix if shape_of(e) == shape]
        first = index_of_key[key_of(members[0])]
        chunk = -(-len(members) // W)
        for n, e in enumerate(members):
            r, k = n // chunk, n % chunk
            assert torch.equal(shards[r]["state"][first]["momentum_buffer"][k],
                               state[f"{e[0]}.frst_mntm"][e[1]]), f"W={W} {e}: momentum"
            got = shards[r]["state"][first]["second_momentum_buffer"][k]
            ours = state[f"{e[0]}.scnd_mntm"][e[1]]
            if tuple(got.shape) == tuple(ours.shape):
                assert torch.equal(got, ours), f"W={W} {e}: second moment"
            else:                              # the square-c_proj mean fallback
                assert e[0] == "W_O" and shape[-2] == shape[-1], f"unexpected fallback {e}"
                assert torch.allclose(got, ours.mean().expand_as(got)), f"W={W} {e}: fallback"
        for r in range(W):                     # padding slots stay zero
            for k in range(chunk):
                if r * chunk + k >= len(members):
                    assert not shards[r]["state"][first]["momentum_buffer"][k].any(), "padding"
    print(f"[OK] optimizer, world_size={W}: structure + routing + reassembly verified")

# ---------------------------------------------------------------- 3. it actually steps
# At world_size=1 the non-distributed optimizer's buffers have exactly the shapes
# DistMuonAdamW's rank-0 shard carries, so this runs the real update kernels over
# our tensors rather than only checking bookkeeping.
opt = new_opt(model)
opt.load_state_dict(convert_optimizer(model_data, optim_data, 1, 0, LRS))
for p in model.parameters():
    p.grad = torch.randn_like(p) * 1e-3
before = {k: v.detach().clone() for k, v in model.named_parameters()}
mom_before = {i: opt.state[g["params"][0]]["momentum_buffer"].clone()
              for i, g in enumerate(opt.param_groups) if g["kind"] == "muon"}
opt.step()
stuck = [k for k, v in model.named_parameters() if torch.equal(v, before[k])]
assert not stuck, f"params did not move: {stuck[:5]}"
for i, m0 in mom_before.items():
    assert not torch.equal(m0, opt.state[opt.param_groups[i]["params"][0]]["momentum_buffer"]), \
        f"muon group {i}: momentum buffer did not advance"
assert all(opt.state[g["params"][0]]["step"] == 43
           for g in opt.param_groups if g["kind"] == "adamw"), "step did not advance from 42"
assert all(torch.isfinite(v).all() for _, v in model.named_parameters()), "non-finite param"
print(f"[OK] warm-started MuonAdamW.step(): all {len(before)} params updated, "
      "buffers advanced, step 42 -> 43")

# ---------------------------------------------------------------- 4. the guard bites
# Outside the square case the two second-moment axes carry different information,
# and there is no faithful conversion. Refuse rather than quietly mean-fill.
bad_state = dict(state)
bad_state["W_in.scnd_mntm"] = torch.zeros(NL, 1, DM)   # wrong axis, NON-square bank
try:
    convert_optimizer(model_data, {"step": 42, "t_step": 42, "state": bad_state}, 1, 0, LRS)
    sys.exit("FAIL: a non-square second-moment axis mismatch was accepted")
except AssertionError as e:
    assert "non-square" in str(e), e
print("[OK] non-square second-moment axis mismatch refused")
print("PASS")
