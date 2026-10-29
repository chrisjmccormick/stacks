import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import torch


# ------------------------------------------------------------------------------
# FlashAttention
# ------------------------------------------------------------------------------
# FA3 is fundamentally not for Blackwell; it uses FA2.
# Note that FA2 needs d_qk == d_vo and head_dim <= 128

from kernels import get_kernel

_k = get_kernel("kernels-community/flash-attn2", version=1)
# The raw ops live in flash_attn_interface; the top level only re-exports
# the varlen/kvcache wrappers.
fa2 = getattr(_k, "flash_attn_interface", _k)


def flash_attn_varlen_fwd_lse(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, window_size):
    """Bidirectional attention forward that also returns the softmax LSE (H, T) fp32.
    Query segment b attends to key segment b; the two tables differ for the
    question-to-state call, whose rows outside its segments come back unwritten."""
    # FA2 takes dropout_p and softmax_scale positionally, before causal.
    out, softmax_lse, *_ = fa2._flash_attn_varlen_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        0.0,                        # dropout_p
        q.shape[-1] ** -0.5,        # softmax_scale
        False,                      # causal
        window_size_left=window_size[0], window_size_right=window_size[1])
    return out, softmax_lse


def flash_attn_varlen_bwd(dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k, window_size):
    """Attention backward for flash_attn_varlen_fwd_lse: returns (dq, dk, dv).
    FA2's varlen backward writes the grads into pre-allocated dq/dk/dv (they
    sit right after the saved tensors) and returns softmax_d. They start at zero:
    FA2 writes only the rows inside its segments, and the question-to-state
    call's grads are added to the self call's on every row."""
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    fa2._flash_attn_varlen_backward(
        dout, q, k, v, out, softmax_lse,
        dq, dk, dv,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        0.0,                        # dropout_p
        q.shape[-1] ** -0.5,        # softmax_scale
        False,                      # is_causal
        window_size[0], window_size[1],
        0.0,                        # softcap
        None,                       # alibi_slopes
        False,                      # deterministic
        None,                       # rng_state
    )
    return dq, dk, dv


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

REPO_ID = "ChrisMcCormick/climbmix_32k_8_170"
DATASET_DIR = os.path.expanduser("~/.cache/encoder-rtx/climbmix_32k_8_170")
KEV_DIR = os.path.expanduser("~/.cache/encoder-rtx/data")   # data/prepare_kev_suites.py writes here

NUM_TRAIN_SHARDS = 10      # 100M raw tokens each; the filler reads about 4 of them


def download_dataset():
    """The shards, tokenizer and config into DATASET_DIR, if not already there."""
    from huggingface_hub import HfApi, hf_hub_download
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("=== Downloading dataset files ===")
    for fname in HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset"):
        if not (fname.startswith("climbmix/") or fname.startswith("tokenizer/") or fname == "config.json"):
            continue
        # Skip over excess training shards.
        if fname.startswith("climbmix/train_") and int(fname[len("climbmix/train_"):].split(".")[0]) > NUM_TRAIN_SHARDS:
            continue
        # Download everything else.
        if not os.path.exists(os.path.join(DATASET_DIR, fname)):
            hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset", local_dir=DATASET_DIR)
    print("  Done.")
    assert os.path.exists(os.path.join(KEV_DIR, "kev_meta.json")), \
        f"{KEV_DIR} has no decision data -- run data/prepare_kev_suites.py first"


def _load_shard(path, pin=False):
    """The uint16 token tensor of one .bin shard (256-int32 header)."""
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, f"magic number mismatch in {path}"
    assert header[1] == 1, f"unsupported version in {path}"
    num_tokens = int(header[2])  # number of tokens (claimed)
    tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=pin)
    with open(path, "rb", buffering=0) as f:
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
    assert nbytes == 2 * num_tokens, f"number of tokens read does not match header in {path}"
    return tokens


def doc_stream(split, bos_id):
    """The "train" or "val" shards as one stream of documents in corpus order, each
    a uint16 array beginning with BOS. Documents split across two shards are stitched."""
    names = ([f"climbmix/train_{k:06d}.bin" for k in range(1, NUM_TRAIN_SHARDS + 1)] if split == "train"
             else ["climbmix/val_000000.bin"])
    carry = None
    for name in names:
        toks = _load_shard(os.path.join(DATASET_DIR, name)).numpy()
        bos = np.flatnonzero(toks == bos_id)
        if bos.size == 0:
            carry = toks if carry is None else np.concatenate([carry, toks])
            continue
        if carry is not None:
            yield np.concatenate([carry, toks[:bos[0]]])
        elif bos[0] != 0:
            raise ValueError(f"{name} starts mid-document with no preceding shard")
        for i in range(bos.size - 1):
            yield toks[bos[i]:bos[i + 1]]
        carry = toks[bos[-1]:]
    if carry is not None:
        yield carry


# ------------------------------------------------------------------------------
# Decision Records
# ------------------------------------------------------------------------------

def load_records(split):
    """One split of Kev's suites (data/prepare_kev_suites.py) as a list of records:
    the state's ids, and per question its prefix ids, its options' ids (each
    ending in its readout token) and a target distribution over the options --
    the record's soft target where it has one, else the label one-hot."""
    import pyarrow.parquet as pq
    records = []
    for r in pq.read_table(os.path.join(KEV_DIR, f"kev_{split}.parquet")).to_pylist():
        questions = []
        for kind, prefix, options, label, target in zip(r["q_type"], r["q_prefix"], r["q_options"],
                                                        r["q_label"], r["q_target"]):
            questions.append(dict(
                kind=kind, label=label,
                prefix=np.array(prefix, dtype=np.int32),
                options=[np.array(o, dtype=np.int32) for o in options],
                target=(np.array(target, dtype=np.float32) if target
                        else np.eye(len(options), dtype=np.float32)[label])))
        records.append(dict(split=split, source=r["source"], variant=r["variant"],
                            state=np.array(r["state_ids"], dtype=np.int32), questions=questions))
    return records


def record_tokens(r):
    return r["state"].size + sum(q["prefix"].size + sum(o.size for o in q["options"]) for q in r["questions"])


# ------------------------------------------------------------------------------
# Packs
# ------------------------------------------------------------------------------
# One pack is one micro-batch of `tokens_per_pack` tokens, laid out in three regions:
#
#   [ state 1 | state 2 | ... ][ questions of record 1 | of record 2 | ... ][ filler documents ]
#
# Every state, question and filler document is its own segment of the self
# attention call (cu_self). Each record's questions also attend to its state in
# a second call, whose query segments are the records' question blocks (cu_q)
# and whose key segments are their states (cu_k). A question's positions
# continue its state's, so every question of a record reads it as if it were the
# only one. Filler is ClimbMix, trained on MLM; the decision records fill a
# `decision_frac` share of the pack.

class Pack(NamedTuple):
    """One micro-batch, as forward_backward consumes it."""
    idx:       torch.Tensor   # (T,)      int32  token ids, MLM-corrupted in the filler
    pos:       torch.Tensor   # (T,)      int32  position within the segment (questions continue their state's)
    rows:      torch.Tensor   # (7, T)    int32  each position's table rows (see table_rows)
    seg_start: torch.Tensor   # (T,)      bool   first token of a segment: no smear, no [prev, curr] pair
    is_q:      torch.Tensor   # (T,)      bool   question rows, whose attention merges both calls
    cu_self:   torch.Tensor   # (W,)      int32  every segment, ghost-padded with T
    cu_q:      torch.Tensor   # (Wr,)     int32  each record's question block, ghost-padded with its end
    cu_k:      torch.Tensor   # (Wr,)     int32  each record's state, ghost-padded with its end
    mlm_pos:   torch.Tensor   # (M,)      int64  masked filler positions
    mlm_tgt:   torch.Tensor   # (M,)      int64  their original tokens
    ro_pos:    torch.Tensor   # (Rc,)     int64  readout positions, padded with 0
    ro_qid:    torch.Tensor   # (Rc,)     int64  each readout's question in the pack, padded with Qc
    ro_tgt:    torch.Tensor   # (Rc,)     fp32   target probability of each option, padded with 0
    n_q:       torch.Tensor   # (1,)      fp32   questions in the pack


def plan_packs(records, docs, tokens_per_pack, decision_frac, seq_len, num_packs, rng):
    """Choose each pack's records (with a choice question's option order) and its
    filler documents. Training (num_packs set) cycles shuffled epochs of the
    records and permutes every choice's options; evaluation (num_packs None)
    takes each record once, in order, as stored. A pack takes records until the
    next would pass its decision budget (none at all when the budget is 0: MLM-only
    packs), then filler: each document's first seq_len tokens, the last one cut to
    fill the pack exactly."""
    budget = decision_frac * tokens_per_pack
    order, k = [], 0
    plans = []
    while num_packs is None or len(plans) < num_packs:
        chosen, used = [], 0
        while True:
            if k == len(order):
                if num_packs is None and order:
                    break                      # evaluation: every record placed
                order, k = (list(rng.permutation(len(records))) if num_packs else list(range(len(records)))), 0
            r = records[order[k]]
            n = record_tokens(r)
            if (chosen or budget <= 0) and used + n > budget:
                break
            perms = [rng.permutation(len(q["options"])) if (num_packs and q["kind"] == "choice")
                     else np.arange(len(q["options"])) for q in r["questions"]]
            chosen.append((r, perms))
            used += n
            k += 1
        if not chosen and num_packs is None:
            break                          # evaluation: nothing left to place
        filler, pos = [], used
        while pos < tokens_per_pack:
            doc = next(docs)
            n = min(doc.size, seq_len, tokens_per_pack - pos)
            filler.append(doc[:n])
            pos += n
        plans.append((chosen, filler))
        if num_packs is None and k == len(order):
            break
    return plans


def pack_caps(plans):
    """The static sizes every pack is padded to, from its plans: cu_self width,
    records (cu_q/cu_k width - 1), readouts, questions, and the longest segment,
    question block and state (FA2's max_seqlen bounds), and the fewest filler tokens."""
    caps = dict(segments=0, records=0, readouts=0, questions=0, seg_len=0, qblock_len=0, state_len=0,
                min_filler=math.inf)
    for chosen, filler in plans:
        n_q = sum(len(r["questions"]) for r, _ in chosen)
        caps["segments"] = max(caps["segments"], len(chosen) + n_q + len(filler))
        caps["records"] = max(caps["records"], len(chosen))
        caps["readouts"] = max(caps["readouts"], sum(len(q["options"]) for r, _ in chosen for q in r["questions"]))
        caps["questions"] = max(caps["questions"], n_q)
        for r, _ in chosen:
            q_lens = [q["prefix"].size + sum(o.size for o in q["options"]) for q in r["questions"]]
            caps["seg_len"] = max(caps["seg_len"], r["state"].size, *q_lens)
            caps["qblock_len"] = max(caps["qblock_len"], sum(q_lens))
            caps["state_len"] = max(caps["state_len"], r["state"].size)
        caps["seg_len"] = max([caps["seg_len"], *(d.size for d in filler)])
        caps["min_filler"] = min(caps["min_filler"], sum(d.size for d in filler))
    return caps


def materialize_packs(plans, tokens_per_pack, cfg, table_rows, mask_id, rng, mlm_per_pack, pin=True):
    """Lay out every planned pack into pinned host tensors (see Pack), and return
    them with each pack's questions' (record, question) for scoring. cfg carries
    the static sizes: cu_width, rec_width, ro_cap, q_cap, bpe_vocab; mlm_per_pack is
    the masked positions per pack (the evaluation packs have their own count, fixed
    across runs, so their validation is comparable whatever the training share)."""
    T, n = tokens_per_pack, len(plans)
    alloc = lambda *shape, dtype: torch.empty(*shape, dtype=dtype, pin_memory=pin)
    out = dict(idx=alloc(n, T, dtype=torch.int32), pos=alloc(n, T, dtype=torch.int32),
               rows=alloc(n, 7, T, dtype=torch.int32), seg_start=alloc(n, T, dtype=torch.bool),
               is_q=alloc(n, T, dtype=torch.bool),
               cu_self=alloc(n, cfg.cu_width, dtype=torch.int32),
               cu_q=alloc(n, cfg.rec_width, dtype=torch.int32), cu_k=alloc(n, cfg.rec_width, dtype=torch.int32),
               mlm_pos=alloc(n, mlm_per_pack, dtype=torch.int64), mlm_tgt=alloc(n, mlm_per_pack, dtype=torch.int64),
               ro_pos=alloc(n, cfg.ro_cap, dtype=torch.int64), ro_qid=alloc(n, cfg.ro_cap, dtype=torch.int64),
               ro_tgt=alloc(n, cfg.ro_cap, dtype=torch.float32), n_q=alloc(n, 1, dtype=torch.float32))
    np_ = {k: v.numpy() for k, v in out.items()}
    scored = [None] * n
    seeds = rng.integers(2**63, size=n)   # one generator per pack: the thread pool can't change the draws

    def fill(i):
        chosen, filler = plans[i]
        g = np.random.default_rng(seeds[i])
        idx, pos, seg_start = np_["idx"][i], np_["pos"][i], np_["seg_start"][i]
        seg_start[:] = False
        ends, o = [], 0
        # States
        state_starts = []
        for r, _ in chosen:
            L = r["state"].size
            idx[o:o + L], pos[o:o + L] = r["state"], np.arange(L)
            state_starts.append(o); seg_start[o] = True
            o += L; ends.append(o)
        s_end = o
        # Questions, each record's as one block; readouts at each option's last token
        q_starts, ro_pos, ro_qid, ro_tgt, questions = [], [], [], [], []
        for r, perms in chosen:
            q_starts.append(o)
            L_state = r["state"].size
            for q, perm in zip(r["questions"], perms):
                opts = [q["options"][j] for j in perm]
                toks = np.concatenate([q["prefix"], *opts])
                L = toks.size
                idx[o:o + L], pos[o:o + L] = toks, L_state + np.arange(L)
                seg_start[o] = True
                ro_pos += list(o + q["prefix"].size + np.cumsum([x.size for x in opts]) - 1)
                ro_qid += [len(questions)] * len(opts)
                ro_tgt += list(q["target"][perm])
                questions.append((r, q, perm))
                o += L; ends.append(o)
        q_end = o
        # Filler
        for doc in filler:
            idx[o:o + doc.size], pos[o:o + doc.size] = doc, np.arange(doc.size)
            seg_start[o] = True
            o += doc.size; ends.append(o)
        assert o == T, f"pack {i} holds {o} tokens, not {T}"
        assert len(ro_pos) <= cfg.ro_cap and len(questions) <= cfg.q_cap
        assert len(ends) < cfg.cu_width and len(chosen) < cfg.rec_width

        np_["is_q"][i][:] = False
        np_["is_q"][i][s_end:q_end] = True
        np_["cu_self"][i][:] = T
        np_["cu_self"][i][0] = 0
        np_["cu_self"][i][1:len(ends) + 1] = ends
        np_["cu_q"][i][:] = q_end
        np_["cu_q"][i][:len(chosen)] = q_starts
        np_["cu_k"][i][:] = s_end
        np_["cu_k"][i][:len(chosen)] = state_starts
        np_["ro_pos"][i][:] = 0
        np_["ro_pos"][i][:len(ro_pos)] = ro_pos
        np_["ro_qid"][i][:] = cfg.q_cap
        np_["ro_qid"][i][:len(ro_qid)] = ro_qid
        np_["ro_tgt"][i][:] = 0.0
        np_["ro_tgt"][i][:len(ro_tgt)] = ro_tgt
        np_["n_q"][i][0] = len(questions)

        # MLM on the filler: BERT's 80/10/10 -- [MASK], a random token, or unchanged.
        mlm = q_end + g.choice(T - q_end, mlm_per_pack, replace=False)
        mlm.sort()
        np_["mlm_pos"][i][:] = mlm
        np_["mlm_tgt"][i][:] = idx[mlm]
        roll = g.random(mlm.size)
        idx[mlm[roll < 0.8]] = mask_id
        swap = mlm[(roll >= 0.8) & (roll < 0.9)]
        idx[swap] = g.integers(0, cfg.bpe_vocab - 9, size=swap.size)   # below nanochat's 9 specials

        np_["rows"][i] = table_rows(idx, seg_start)
        scored[i] = questions

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        list(pool.map(fill, range(n)))
    return out, scored


def pack_loader(packs, start=0):
    """The materialized packs, one Pack of device tensors per call, in order from `start`."""
    for i in range(start, packs["idx"].size(0)):
        yield Pack(**{k: v[i].to("cuda", non_blocking=True) for k, v in packs.items()})
