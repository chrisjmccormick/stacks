# Convert a DecoderStack-medium (d24) capture into a nanochat checkpoint.
#
# DecoderStack writes its own two-file capture (see § Checkpoint capture in
# decoderstack_medium_pt-sft.py):
#
#   model_stepNNNNNN.pt   {step, code, weights: {name: tensor}}
#                         bf16 live weights + fp32 scalars, banked over layers.
#   optim_stepNNNNNN.pt   {step, t_step, state: {"name.attr": tensor}}
#                         mantissa / frst_mntm / scnd_mntm / exp_avg / exp_avg_sq,
#                         all-gathered to full size (world-agnostic).
#
# nanochat wants one flat state_dict per capture, in its own module-path key
# names, next to a meta_NNNNNN.json:
#
#   <out_dir>/model_NNNNNN.pt   torch.save(model.state_dict())
#   <out_dir>/meta_NNNNNN.json  {"step", "val_bpb", "model_config", ...}
#
# This script does that translation. The architectures are the same model --
# DecoderStack-medium is a flattened port of nanochat d24 -- so every tensor has
# a home and nothing is reshaped or transposed: DecoderStack's banks index the
# layer on dim 0 and each slice already uses F.linear's (out, in) convention.
# The only real work is naming, unbanking, and dtype.
#
# DTYPE, and why the optimizer file matters
# -----------------------------------------
# nanochat holds fp32 master weights for everything except the two embedding
# tables, which init_weights() casts to COMPUTE_DTYPE (bf16). DecoderStack holds
# bf16 LIVE weights everywhere plus a uint16 `mantissa` in the optimizer file --
# the fp32 master's bit pattern is (live_bf16_bits << 16) | mantissa. So:
#
#   nanochat fp32 params  <- fp32 master  = live + mantissa   (needs --optim)
#   nanochat bf16 params  <- live bf16 as-is                  (mantissa dropped,
#                            which is correct: nanochat's bf16 embedding IS its
#                            master, it has no lower bits to carry)
#
# Without --optim the fp32 params are filled with the bf16 live values upcast to
# fp32 -- the right dtype, but only bf16 precision. That is what you get if you
# only downloaded the model file, and it is fine for eval: on a held-out English
# paragraph the d24 step-5568 capture scores 2.4828 nats/token with the masters
# and 2.4833 without (bpb 0.7252 vs 0.7254). Pass --optim when you want the
# exact master anyway -- it is a bit-exact reconstruction, not an approximation.
#
# OPTIMIZER STATE
# ---------------
# --world-size N additionally writes optim_NNNNNN_rank{0..N-1}.pt, the ZeRO-2
# shards nanochat's DistMuonAdamW expects, so chat_sft can warm-start its
# optimizer instead of printing "starting with fresh optimizer (slightly worse)".
# It is optional: SFT is correct without it. Pre-training resume is out of reach
# either way -- see OPTIMIZER-STATE NOTES at the bottom of this file.
#
# Usage:
#   python utils/convert_ckpt_to_nanochat.py \
#       --model checkpoints/model_step005568.pt \
#       --optim checkpoints/optim_step005568.pt \
#       --meta  base_checkpoints/d24_decoderstack/meta_005568.json \
#       --out   ~/.cache/nanochat/base_checkpoints/d24_decoderstack \
#       --world-size 8            # optional: also emit the optimizer shards
#
# Then, in nanochat (branch fa-varlen):
#   from nanochat.checkpoint_manager import build_model
#   model, tokenizer, meta = build_model(checkpoint_dir, 5568, device, "eval")
#
# Verify a change to this file with:
#   NANOCHAT_PATH=~/nanochat python utils/test_convert_ckpt_to_nanochat.py
#
# The tokenizer is NOT interchangeable with other nanochat d24 releases -- see
# the model card. DecoderStack trained on the 32k vocab shipped with the
# ChrisMcCormick/climbmix_32k_8_170 dataset repo; pairing these weights with a
# different 32k tokenizer produces garbage, not slightly-worse text.
import argparse
import json
import os
import shutil

import torch


def fp32_master(live: torch.Tensor, mantissa: torch.Tensor | None) -> torch.Tensor:
    """Rebuild the fp32 master from bf16 live bits + the stashed lower 16 bits.

    Mirrors fp32_master() in the training script. int32 rather than uint32
    because CUDA has no uint32 shifts as of torch 2.9; the truncating .to(int16)
    and the <<16 discard of sign-extension bits make the two equivalent. With no
    mantissa this degrades to a plain upcast (the mantissa bits read as zero),
    which is exactly the bf16-precision fallback documented above.
    """
    if mantissa is None:
        return live.float()
    assert mantissa.shape == live.shape, f"mantissa {tuple(mantissa.shape)} != live {tuple(live.shape)}"
    bits = (live.view(torch.int16).to(torch.int32) << 16) | \
           (mantissa.view(torch.int16).to(torch.int32) & 0xFFFF)
    return bits.view(torch.float32)


def _adamw_groups(ve_slots: int):
    """The AdamW half of setup_optimizer()'s group list, in its exact order.

    Each entry is (lr_key, [(bank_name, bank_slot), ...], betas, eps, weight_decay).
    bank_slot is None for a whole tensor, an int to index a bank's dim 0. The
    betas/eps/wd here are constants in BOTH codebases -- nanochat hardcodes them
    in setup_optimizer, DecoderStack passes the same numbers to build_schedules --
    so they are not a guess about this run, they are the shared values.
    """
    return [
        ("lm_head",      [("lm_head", None)],        (0.8,  0.96),  1e-10, 0.01),
        ("embedding",    [("input_embeds", None)],   (0.8,  0.995), 1e-10, 0.001),
        ("value_embeds", [("value_embeds", j) for j in range(ve_slots)],
                                                     (0.8,  0.995), 1e-10, 0.01),
        ("resid",        [("resid_lambdas", None)],  (0.8,  0.95),  1e-10, 0.05),
        ("x0",           [("x0_lambdas", None)],     (0.96, 0.95),  1e-10, 0.0),
        ("smear",        [("smear_gate", None), ("smear_lambda", None),
                          ("backout_lambda", None)], (0.8,  0.95),  1e-10, 0.0),
    ]


def _matrix_params(n_layer: int, ve: list[int]):
    """setup_optimizer()'s `matrix_params`, in list(transformer.h.parameters()) order.

    Module registration order gives, per block: attn.c_q, c_k, c_v, c_proj,
    [ve_gate], then mlp.c_fc, mlp.c_proj.
    """
    out = []
    for i in range(n_layer):
        out += [("W_Q", i), ("W_K", i), ("W_V", i), ("W_O", i)]
        if i in ve:
            out.append(("ve_gate", ve.index(i)))   # ve_gate banks by SLOT, not layer
        out += [("W_in", i), ("W_out", i)]
    return out


def ve_layers(n_layer: int) -> list[int]:
    """Layers carrying a value embedding, in bank-slot order.

    nanochat's has_ve(): alternating layers, last layer always included. Identical
    to StackConfig.ve_layers, and the ascending order matches the VE bank's slot
    order (cfg.ve_index), so slot j belongs to layer ve_layers(n_layer)[j].
    """
    return [i for i in range(n_layer) if i % 2 == (n_layer - 1) % 2]


def convert(model_data: dict, optim_state: dict | None) -> dict:
    """DecoderStack weights dict -> nanochat state_dict."""
    w = model_data["weights"]
    mant = {} if optim_state is None else optim_state

    def master(name, i=None):
        """fp32 param: live + mantissa. The mantissa of a sharded weight was
        all-gathered to full size at capture, so it already lines up 1:1. Pass
        `i` to rebuild one slice of a bank -- worth it on the MLP banks, where
        materializing the whole thing in fp32 would cost ~0.9 GB per bank."""
        live, m = w[name], mant.get(f"{name}.mantissa")
        if i is not None:
            live, m = live[i], (None if m is None else m[i])
        return fp32_master(live, m)

    n_layer = w["W_Q"].shape[0]
    sd = {}

    # --- Embeddings: bf16 in nanochat, so the live weights go in untouched. ---
    sd["transformer.wte.weight"] = w["input_embeds"]
    for slot, layer in enumerate(ve_layers(n_layer)):
        sd[f"value_embeds.{layer}.weight"] = w["value_embeds"][slot]

    # --- lm_head: AdamW in both, but nanochat keeps it fp32 (it is a Linear,
    #     not an Embedding, so init_weights() never casts it). ---
    sd["lm_head.weight"] = master("lm_head")

    # --- Per-layer matrices: unbank dim 0. Every slice is already (out, in). ---
    banks = {
        "W_Q":   "transformer.h.{i}.attn.c_q.weight",
        "W_K":   "transformer.h.{i}.attn.c_k.weight",
        "W_V":   "transformer.h.{i}.attn.c_v.weight",
        "W_O":   "transformer.h.{i}.attn.c_proj.weight",
        "W_in":  "transformer.h.{i}.mlp.c_fc.weight",
        "W_out": "transformer.h.{i}.mlp.c_proj.weight",
    }
    for name, template in banks.items():
        for i in range(n_layer):
            sd[template.format(i=i)] = master(name, i).contiguous()

    # --- VE gates: banked by VE SLOT, not by layer, so they unbank through the
    #     same slot->layer map as the value embeddings. ---
    for slot, layer in enumerate(ve_layers(n_layer)):
        sd[f"transformer.h.{layer}.attn.ve_gate.weight"] = master("ve_gate", slot).contiguous()

    # --- Scalars: fp32-live in both, no mantissa, same names. ---
    sd["resid_lambdas"] = w["resid_lambdas"]
    sd["x0_lambdas"] = w["x0_lambdas"]
    sd["smear_gate.weight"] = w["smear_gate"]
    sd["smear_lambda"] = w["smear_lambda"]
    sd["backout_lambda"] = w["backout_lambda"]
    return sd


def convert_optimizer(model_data: dict, optim_data: dict, world_size: int, rank: int,
                      lrs: dict) -> dict:
    """DecoderStack optimizer capture -> one rank's nanochat optimizer state_dict.

    nanochat's state_dict is keyed by flattened param INDEX over setup_optimizer()'s
    groups, and DistMuonAdamW shards that state per rank. Our capture all-gathered
    everything to full size, so this is re-slicing, not reconstruction.

    Returns the dict to torch.save as optim_NNNNNN_rank{rank}.pt. Call once per rank
    rather than building them all: at d24/world=8 each shard is ~1 GB.
    """
    w, st = model_data["weights"], optim_data["state"]
    t_step = optim_data["t_step"]
    n_layer = w["W_Q"].shape[0]
    ve = ve_layers(n_layer)
    n_embd = w["input_embeds"].shape[1]
    d_scale = (n_embd / 768) ** -0.5   # setup_optimizer's 1/sqrt(dmodel) AdamW LR scale

    def bank(name, slot, attr):
        """One param's full-size optimizer state. value_embeds is the odd one out:
        its AdamW state is shaped over the FLATTENED (slot * vocab) row axis, so it
        has to be folded back to 3-D before a slot can be indexed."""
        t = st[f"{name}.{attr}"]
        if slot is None:
            return t
        if name == "value_embeds":
            return t.view(len(ve), -1, t.shape[-1])[slot]
        return t[slot]

    adamw_lr = {
        "lm_head": lrs["unembedding_lr"] * d_scale,
        "embedding": lrs["embedding_lr"] * d_scale,
        "value_embeds": lrs["embedding_lr"] * d_scale * 0.5,
        "resid": lrs["scalar_lr"] * 0.01,
        "x0": lrs["scalar_lr"],
        "smear": 0.2,   # hardcoded in setup_optimizer, not scaled
    }

    # --- Build the group plan exactly as setup_optimizer() would: AdamW groups in
    #     a fixed order, then Muon groups keyed by `sorted({p.shape})`. ---
    plan = []   # (kind, [(name, slot), ...], hyperparams dict)
    for lr_key, params, betas, eps, wd in _adamw_groups(len(ve)):
        plan.append(("adamw", params, dict(kind="adamw", lr=adamw_lr[lr_key],
                                           betas=list(betas), eps=eps, weight_decay=wd)))
    matrix = _matrix_params(n_layer, ve)
    shape_of = lambda p: tuple(w[p[0]].shape[1:])
    for shape in sorted({shape_of(p) for p in matrix}):
        plan.append(("muon", [p for p in matrix if shape_of(p) == shape],
                     dict(kind="muon", lr=lrs["matrix_lr"], momentum=0.95, ns_steps=5,
                          beta2=0.9, weight_decay=lrs["weight_decay"])))

    # Param indices are assigned by walking the groups in order.
    index, idx = {}, 0
    for _, params, _ in plan:
        for p in params:
            index[p] = idx
            idx += 1
    assert idx == 7 + len(ve) + len(matrix), f"param count {idx} does not add up"

    state, groups = {}, []
    for kind, params, hp in plan:
        groups.append({**hp, "initial_lr": hp["lr"],
                       "params": [index[p] for p in params]})
        if kind == "adamw":
            for p in params:
                exp_avg = bank(*p, "exp_avg")
                # ZeRO-2: params with >= 1024 elements are row-sharded over dim 0 by
                # rank; smaller ones are replicated (nanochat batches those into an
                # all_reduce instead of a reduce_scatter).
                if exp_avg.numel() >= 1024:
                    assert exp_avg.shape[0] % world_size == 0, \
                        f"{p}: dim 0 ({exp_avg.shape[0]}) must divide world_size {world_size}"
                    rows = exp_avg.shape[0] // world_size
                    cut = lambda t, n=rows: t[rank * n:(rank + 1) * n].clone()
                else:
                    cut = lambda t: t.clone()
                state[index[p]] = {
                    "step": t_step,
                    "exp_avg": cut(exp_avg),
                    "exp_avg_sq": cut(bank(*p, "exp_avg_sq")),
                }
        else:
            # Muon state is one stacked buffer per GROUP, held under the first
            # param's entry, chunked across ranks and zero-padded when the group
            # does not divide evenly.
            shape = shape_of(params[0])
            chunk = -(-len(params) // world_size)
            start = rank * chunk
            owned = min(chunk, max(0, len(params) - start))
            mom = torch.zeros(chunk, *shape, dtype=torch.float32)
            # nanochat factors the second moment along whichever axis its shape
            # heuristic calls the neuron axis; ours is set explicitly per bank.
            nc_shape = (shape[-2], 1) if shape[-2] >= shape[-1] else (1, shape[-1])
            snd = torch.zeros(chunk, *nc_shape, dtype=torch.float32)
            for k in range(owned):
                p = params[start + k]
                mom[k] = bank(*p, "frst_mntm")
                ours = bank(*p, "scnd_mntm")
                if tuple(ours.shape) == nc_shape:
                    snd[k] = ours
                else:
                    # Only W_O lands here, and only because nanochat's shape
                    # heuristic picks the other axis on a SQUARE c_proj. That is
                    # benign: polar express returns a ~orthonormal update, whose
                    # neuron norms are ~uniform along either axis, so their mean is
                    # the right common value. On a non-square bank the two axes
                    # would carry genuinely different information -- refuse.
                    assert shape[-2] == shape[-1], (
                        f"{p}: second-moment axis differs on a non-square bank "
                        f"{shape} (ours {tuple(ours.shape)}, nanochat {nc_shape}); "
                        "no faithful conversion exists")
                    snd[k] = ours.mean()
            state[index[params[0]]] = {"momentum_buffer": mom,
                                       "second_momentum_buffer": snd}
    return {"state": state, "param_groups": groups}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="DecoderStack model_stepNNNNNN.pt")
    p.add_argument("--optim", default=None,
                   help="matching optim_stepNNNNNN.pt; supplies the mantissas that make "
                        "the fp32 params exact. Omit for a bf16-precision conversion.")
    p.add_argument("--out", required=True, help="output checkpoint dir (nanochat model_tag dir)")
    p.add_argument("--meta", default=None, help="meta_NNNNNN.json to copy alongside the model")
    p.add_argument("--dump-code", action="store_true",
                   help="also write the training script embedded in the capture's `code` field")
    p.add_argument("--world-size", type=int, default=0, metavar="N",
                   help="also write optim_NNNNNN_rank{0..N-1}.pt for an N-GPU run "
                        "(requires --optim). Omit to convert weights only.")
    # Group hyperparameters for the emitted optimizer. torch's load_state_dict
    # REPLACES param_group dicts with the saved ones, so whatever goes here becomes
    # the optimizer's policy on load. Defaults are setup_optimizer()'s own, with
    # weight_decay=0.0 -- both the SFT setting and where DecoderStack's cosine-to-
    # zero Muon decay actually lands (4.8e-9 at step 5568). nanochat's chat_sft
    # restores its own lr right after loading and schedules momentum per step, so
    # in practice only betas/eps/weight_decay/ns_steps come from here.
    p.add_argument("--unembedding-lr", type=float, default=0.004)
    p.add_argument("--embedding-lr", type=float, default=0.2)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--scalar-lr", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    args = p.parse_args()
    if args.world_size and not args.optim:
        p.error("--world-size needs --optim (the optimizer state lives in that file)")

    # mmap so a 2.8 GB model / 11 GB optimizer file is paged, not slurped.
    model_data = torch.load(args.model, map_location="cpu", mmap=True, weights_only=True)
    step = model_data["step"]
    print(f"loaded {args.model}: step {step}, {len(model_data['weights'])} weights")

    optim_data = optim_state = None
    if args.optim:
        optim_data = torch.load(args.optim, map_location="cpu", mmap=True, weights_only=True)
        assert optim_data["step"] == step, f"optim step {optim_data['step']} != model step {step}"
        optim_state = optim_data["state"]
        n_mant = sum(1 for k in optim_state if k.endswith(".mantissa"))
        print(f"loaded {args.optim}: {len(optim_state)} state tensors, {n_mant} mantissas")
    else:
        print("no --optim: fp32 params will carry bf16 precision (upcast, not exact masters)")

    sd = convert(model_data, optim_state)
    total = sum(t.numel() for t in sd.values())
    by_dtype = {}
    for t in sd.values():
        by_dtype[t.dtype] = by_dtype.get(t.dtype, 0) + t.numel()
    print(f"converted: {len(sd)} tensors, {total:,} params "
          + ", ".join(f"{n:,} {str(d).replace('torch.', '')}" for d, n in by_dtype.items()))

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, f"model_{step:06d}.pt")
    torch.save(sd, model_path)
    print(f"wrote {model_path} ({os.path.getsize(model_path):,} bytes)")

    if args.meta:
        meta_path = os.path.join(args.out, f"meta_{step:06d}.json")
        shutil.copyfile(args.meta, meta_path)
        with open(meta_path, encoding="utf-8") as f:
            meta_step = json.load(f)["step"]
        assert meta_step == step, f"meta step {meta_step} != model step {step}"
        print(f"wrote {meta_path}")
    else:
        print(f"NOTE: nanochat also needs meta_{step:06d}.json in {args.out} "
              "(model_config lives there, not in the .pt)")

    if args.world_size:
        lrs = dict(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
                   matrix_lr=args.matrix_lr, scalar_lr=args.scalar_lr,
                   weight_decay=args.weight_decay)
        # One rank at a time -- holding all of them would cost the whole optimizer.
        for r in range(args.world_size):
            shard = convert_optimizer(model_data, optim_data, args.world_size, r, lrs)
            path = os.path.join(args.out, f"optim_{step:06d}_rank{r:d}.pt")
            torch.save(shard, path)
            print(f"wrote {path} ({os.path.getsize(path):,} bytes)")
            if r == 0:
                g = shard["param_groups"]
                print(f"  {len(g)} groups "
                      f"({sum(1 for x in g if x['kind'] == 'adamw')} adamw / "
                      f"{sum(1 for x in g if x['kind'] == 'muon')} muon), "
                      f"{sum(len(x['params']) for x in g)} params, "
                      f"step {optim_data['t_step']}")
            del shard

    if args.dump_code:
        code_path = os.path.join(args.out, f"code_{step:06d}.py")
        with open(code_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(model_data["code"])
        print(f"wrote {code_path} (the exact training script for this capture)")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# OPTIMIZER-STATE NOTES
# -----------------------------------------------------------------------------
# SFT CONTINUATION WORKS WITH THE MODEL ALONE. nanochat's chat_sft builds a fresh
# optimizer via model.setup_optimizer() and only optionally warm-starts it from
# load_optimizer_state(); when the shard is absent it prints "optimizer
# checkpoint not found, starting with fresh optimizer (slightly worse)" and
# carries on. --world-size exists to remove that "slightly worse", not to unlock
# anything.
#
# PRE-TRAINING RESUME is genuinely out of reach, and it is the dataloader that
# closes the door, not the optimizer: nanochat's resume needs
# meta_data["dataloader_state_dict"] to put its tokenizing loader back in the
# stream, and DecoderStack reads pre-tokenized binary shards through a loader
# with no equivalent state to hand over. The data order could not be continued no
# matter what the optimizer held.
#
# HYPERPARAMETERS ARE POLICY, NOT STATE. torch's Optimizer.load_state_dict
# REPLACES each param_group dict with the saved one, keeping only 'params' -- so
# whatever this script writes becomes the optimizer's lr/betas/wd on load. That
# is why chat_sft saves and restores its own LRs around the call. The emitted
# groups use setup_optimizer()'s defaults (overridable on the command line), with
# weight_decay=0.0: both the SFT setting and where DecoderStack's cosine-to-zero
# Muon decay actually lands (4.8e-9 at step 5568). betas/eps/adamw-wd are not a
# guess -- they are identical constants in both codebases.
#
# Every buffer we keep has a nanochat counterpart, and the precisions line up on
# everything except the two embedding tables:
#
#   DecoderStack              nanochat (MuonAdamW / DistMuonAdamW)     precision
#   ------------------------  --------------------------------------  ---------
#   .frst_mntm      fp32      Muon  state["momentum_buffer"]           fp32 both
#   .scnd_mntm      fp32      Muon  state["second_momentum_buffer"]    fp32 both
#   .exp_avg        fp32      AdamW state["exp_avg"]                   see below
#   .exp_avg_sq     fp32      AdamW state["exp_avg_sq"]                see below
#   .mantissa     uint16      (no counterpart -- nanochat's fp32 param IS
#                              the master; consumed above to rebuild it)
#
# nanochat allocates its Muon buffers as `dtype=p.dtype` and its AdamW buffers as
# `torch.zeros_like(p)`. Its Muon params and lm_head are fp32, so those match us.
# But wte and value_embeds are bf16 PARAMS, so THEIR AdamW moments are bf16 --
# where ours are fp32. That is the one precision difference, and ours is the more
# precise of the two, deliberately: it is only the GRADIENTS that are bf16 for
# those two tables (they are the biggest tensors in the model, so fp32 grads
# would double their scatter and comm traffic, and bf16 matches the autograd
# baseline's numerics). The moment math stays fp32 -- adamw_step_fused upcasts on
# the way in, `grad = grad.to(exp_avg.dtype)`. There is no bf16 AdamW variant in
# the file: the two AdamW kernels differ in whether the param carries a mantissa
# (adamw_step_fused vs adamw_step_fused_fp32), not in moment dtype.
#
# W_O's reduction axis differs between the two, and at d24 it costs nothing.
# NorMuon's factored second moment is a per-neuron mean-square; nanochat infers
# the neuron axis from the shape (`red_dim = -1 if shape[-2] >= shape[-1] else
# -2`) while DecoderStack states it (m.W_O.residual_dim = -2), so the two
# disagree on a square c_proj -- ours is (1, 1536) where nanochat's is (1536, 1).
# But polar express returns a ~orthonormal update, and a square orthonormal
# matrix has ~uniform neuron norms along either axis: there is no variance to
# reduce, the rescale is a ~no-op, and the run is unaffected by the choice.
# The explicit axis earns its keep only when n_heads * d_head != d_model. Above
# d_model, W_O looks like an MLP projection and the shape heuristic happens to
# agree; below it the heuristic picks the wrong axis, and since W_O stores its
# heads transposed relative to QKV, the right answer is not one a shape alone
# can give. Every other bank agrees at d24 (W_in -1, W_out -2, QKV -1).
#
# ve_gate is the second place the axes can diverge, and it is worth knowing about
# because it is NOT square in general. DecoderStack banks it (num_ves, n_kv_heads,
# d_ve_gate) with residual_dim = -1, so the neurons are the n_kv_heads rows;
# nanochat's Linear(ve_gate_channels=12, n_kv_head) hits the same heuristic and
# agrees only when n_kv_head >= 12. At d24 n_kv_head == 12 == d_ve_gate, so the
# bank is square and the two land together. A model with fewer than 12 KV heads
# would disagree for real -- convert_optimizer() asserts rather than papering
# over it, since outside the square case the two axes carry different
# information.
#
# --world-size implements the mapping below. It is mechanical but fiddly, because
# nanochat's state_dict is keyed by flattened param INDEX and is sharded per rank:
#   - Param order is setup_optimizer()'s group order: lm_head, wte,
#     value_embeds.*, resid_lambdas, x0_lambdas, [smear_gate.weight,
#     smear_lambda, backout_lambda], then the Muon groups in `sorted({shapes})`
#     order -- (12,12) ve_gates, (1536,1536) c_q/c_k/c_v/c_proj interleaved in
#     block order, (1536,6144) mlp.c_proj, (6144,1536) mlp.c_fc.
#   - AdamW state for params with >= 1024 elements is sliced over dim 0 by rank;
#     smaller ones are replicated. Ours is captured all-gathered to full size,
#     so it just needs re-slicing (value_embeds first reshaped from its flattened
#     (num_ves * vocab, kv_dim) row axis back to (num_ves, vocab, kv_dim)).
#   - Muon state is stacked per group and chunked: rank r owns params
#     [r*ceil(K/W) : (r+1)*ceil(K/W)] of the group, zero-padded when K % W != 0.
