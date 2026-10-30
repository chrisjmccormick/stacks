# utils_decision.py -- the data module and FA2 wrappers for train_decision.py: the MLM
# base's 16K Unigram ClimbMix shards and its validation loader (utils_mlm.py, unchanged),
# the decision records of data/prepare_decision_data.py, and the packs that put records
# and ClimbMix filler into one micro-batch (utils.py's, from the Phase-0 fork).
import glob
import json
import math
import os
import threading
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
    question-to-state call, whose rows outside its segments come back unwritten.
    window_size is (left, right): local attention each way within a segment;
    (-1, -1) is the whole segment."""
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

REPO_ID = "ChrisMcCormick/climbmix_unigram16k_14_170"
DATASET_DIR = os.path.expanduser("~/.cache/encoder-rtx/climbmix_unigram16k_14_170")
DECISION_DIR = os.path.expanduser("~/.cache/encoder-rtx/decision16k")   # data/prepare_decision_data.py writes here

NUM_TRAIN_SHARDS   = 10      # 100M raw tokens each; the MLM base read 1-7
EVAL_BUFFER_TOKENS = 65536   # tokens per MLM validation micro-batch


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
    assert os.path.exists(os.path.join(DECISION_DIR, "meta.json")), \
        f"{DECISION_DIR} has no decision data -- run data/prepare_decision_data.py first"


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


def doc_stream(split, bos_id, first_shard=1):
    """The "train" shards from `first_shard` on, or the "val" shard, as one stream of
    documents in corpus order, each a uint16 array beginning with BOS. Documents split
    across two shards are stitched; a stream that starts mid-shard-sequence skips the
    leading partial document."""
    names = ([f"climbmix/train_{k:06d}.bin" for k in range(first_shard, NUM_TRAIN_SHARDS + 1)] if split == "train"
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
        for i in range(bos.size - 1):
            yield toks[bos[i]:bos[i + 1]]
        carry = toks[bos[-1]:]
    if carry is not None:
        yield carry


class Shard:
    def __init__(self, tokens, bos_id: int):
        self.tokens = tokens
        self.size = tokens.numel()
        self.bos_id = bos_id
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == bos_id).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == self.bos_id).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens: int, max_seq_len: int):
        """Returns (starts, ends) of BOS-aligned document slices totalling exactly
        num_tokens tokens, or None if this shard is exhausted."""
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = []
        ends = []

        idx = self.i
        cur_len = 0
        while cur_len < num_tokens:
            if idx >= n:
                return None
            cur = self.bos_idx[idx]
            starts.append(cur)
            end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                      cur + max_seq_len,
                      cur + num_tokens - cur_len)
            ends.append(end)
            cur_len += end - cur
            idx += 1

        assert cur_len == num_tokens
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file, bos_id: int):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            result['shard'] = Shard(_load_shard(file, pin=True), bos_id)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get


def val_generator(seq_len, tokens_per_micro):
    """
    The MLM base's validation micro-batches (utils_mlm.data_generator("val"), unchanged), so
    val MLM here scores the same positions as in the base's own run: EVAL_BUFFER_TOKENS
    tokens each, as (tokens, cu_seqlens) device tensors. Single-epoch: sequences are
    BOS-aligned and only returned from their beginning; tokens past `seq_len` are discarded
    (the next sequence starts at the next BOS). The generator ends when the shard runs out.
    """
    # This is to set the fixed size of 'cu_seqlens' for varlen.
    # Estimating 192 docs per 64K tokens.
    max_num_docs = 192 * math.ceil(tokens_per_micro / EVAL_BUFFER_TOKENS)

    with open(os.path.join(DATASET_DIR, "config.json")) as f:
        bos_id = json.load(f)["bos_id"]

    num_tokens = EVAL_BUFFER_TOKENS

    # Get the list of shard files.
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "climbmix/val_*.bin")))
    if not files:
        raise FileNotFoundError(f"No val shards found under {DATASET_DIR}")

    # Load the first shard.
    shard = Shard(_load_shard(files[0], pin=True), bos_id)
    remaining_files = files[1:]
    next_shard_idx = 0
    next_shard_getter = Shard.load_async(remaining_files[0], bos_id) if remaining_files else None

    while True:
        # Get the start and end indices (within the shard) of the sequences to use for
        # the current micro-batch.
        result = shard.next_batch(num_tokens, seq_len)

        # If this shard is exhausted,
        if result is None:
            # If there are no more shards, kill the dataloader.
            if next_shard_getter is None:
                return

            # Load the next shard.
            shard = next_shard_getter()
            next_shard_idx += 1
            next_shard_getter = Shard.load_async(remaining_files[next_shard_idx], bos_id) if next_shard_idx < len(remaining_files) else None

            # Re-start the loop.
            continue

        # Locations of the documents in the shard.
        start_idxs = torch.tensor(result[0])
        end_idxs = torch.tensor(result[1])

        # The sequences may not be contiguous within the shard (some are truncated), so
        # slice them and re-concatenate into a single tensor.
        buf = torch.cat([shard.tokens[i:j] for i, j in zip(start_idxs, end_idxs)])

        # The documents' start indices within `buf`, by accumulating their lengths.
        cum_lengths = (end_idxs - start_idxs).cumsum(0)

        # One entry per doc plus the leading 0 must fit the fixed buffer.
        assert len(cum_lengths) < max_num_docs, \
            f"micro-batch packed {len(cum_lengths)} docs; cu_seqlens holds only {max_num_docs}"

        # The fixed buffer, filled with "empty documents" starting one past the end of `buf`.
        _cum_lengths = torch.full((max_num_docs,), num_tokens)

        # Then copy in the lengths, inserting the first document (index 0).
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        yield (
            buf.to(dtype=torch.int32).to(device="cuda", non_blocking=True),
            _cum_lengths.to(dtype=torch.int32).to(device="cuda", non_blocking=True),
        )


# ------------------------------------------------------------------------------
# Decision Records
# ------------------------------------------------------------------------------

def load_records(split, state_cap, soft_weight):
    """One split of data/prepare_decision_data.py's output as a list of records: the
    state's ids (its first state_cap tokens), and per question its prefix ids, its
    options' ids (each ending in its readout token) and a target distribution over the
    options -- soft_weight x the record's soft target + the rest on the label, where it
    has a soft target, else the label one-hot."""
    import pyarrow.parquet as pq
    records = []
    for r in pq.read_table(os.path.join(DECISION_DIR, f"{split}.parquet")).to_pylist():
        questions = []
        for name, kind, prefix, options, label, target in zip(r["q_name"], r["q_type"], r["q_prefix"],
                                                              r["q_options"], r["q_label"], r["q_target"]):
            hard = np.eye(len(options), dtype=np.float32)[label]
            questions.append(dict(
                name=name, kind=kind, label=label,
                prefix=np.array(prefix, dtype=np.int32),
                options=[np.array(o, dtype=np.int32) for o in options],
                target=(soft_weight * np.array(target, dtype=np.float32) + (1 - soft_weight) * hard
                        if target else hard)))
        records.append(dict(split=split, id=r["id"], source=r["source"], variant=r["variant"],
                            state=np.array(r["state_ids"][:state_cap], dtype=np.int32),
                            state_len=len(r["state_ids"]), questions=questions))
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
# only one. Filler is ClimbMix, trained on MLM (corrupted on the GPU, from
# filler_start on); the decision records fill a `decision_frac` share of the pack.

class Pack(NamedTuple):
    """One micro-batch, as forward_backward consumes it (the MLM corruption aside)."""
    tokens:       torch.Tensor   # (T,)   int32  clean token ids
    pos:          torch.Tensor   # (T,)   int32  RoPE position within the segment (questions continue their state's)
    is_q:         torch.Tensor   # (T,)   bool   question rows, whose attention merges both calls
    cu_self:      torch.Tensor   # (W,)   int32  every segment, ghost-padded with T
    cu_q:         torch.Tensor   # (Wr,)  int32  each record's question block, ghost-padded with its end
    cu_k:         torch.Tensor   # (Wr,)  int32  each record's state, ghost-padded with its end
    filler_start: torch.Tensor   # (1,)   int64  the first filler position: MLM corrupts from here on
    ro_pos:       torch.Tensor   # (Rc,)  int64  readout positions, padded with 0
    ro_qid:       torch.Tensor   # (Rc,)  int64  each readout's question in the pack, padded with Qc
    ro_tgt:       torch.Tensor   # (Rc,)  fp32   target probability of each option, padded with 0
    n_q:          torch.Tensor   # (1,)   fp32   questions in the pack


def plan_packs(records, docs, tokens_per_pack, decision_frac, seq_len, rng, epochs=None):
    """Choose each pack's records (with a choice question's option order) and its
    filler documents. Training (epochs set) cycles shuffled epochs of the records,
    permutes every choice's options, and plans packs until `epochs` x the records are
    placed, plus one more (the loop prefetches a pack after its last step);
    evaluation (epochs None) takes each record once, in order, as stored. A pack
    takes records until the next would pass its decision budget (at least one),
    then filler: each document's first seq_len tokens, the last one cut to fill the
    pack exactly."""
    training = epochs is not None
    budget = decision_frac * tokens_per_pack
    target = round(epochs * len(records)) if training else len(records)
    order, k, placed = [], 0, 0

    def one_pack(limit):
        nonlocal order, k, placed
        chosen, used = [], 0
        while placed < limit:
            if k == len(order):
                order, k = (list(rng.permutation(len(records))) if training else list(range(len(records)))), 0
            r = records[order[k]]
            n = record_tokens(r)
            if chosen and used + n > budget:
                break
            perms = [rng.permutation(len(q["options"])) if (training and q["kind"] == "choice")
                     else np.arange(len(q["options"])) for q in r["questions"]]
            chosen.append((r, perms))
            used += n
            k += 1
            placed += 1
        filler, pos = [], used
        while pos < tokens_per_pack:
            doc = next(docs, None)
            assert doc is not None, f"the filler ran out after {len(plans)} packs: raise NUM_TRAIN_SHARDS (utils_decision.py)"
            n = min(doc.size, seq_len, tokens_per_pack - pos)
            filler.append(doc[:n])
            pos += n
        return chosen, filler

    plans = []
    while placed < target:
        plans.append(one_pack(target))
    if training:
        plans.append(one_pack(math.inf))   # the prefetched pack: planned, never trained on
    return plans


def pack_caps(plans):
    """The static sizes every pack is padded to, from its plans: cu_self width,
    records (cu_q/cu_k width - 1), readouts, questions, and the longest segment,
    question block and state (FA2's max_seqlen bounds), and the fewest non-BOS filler
    tokens (the MLM corruption's room)."""
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
        caps["min_filler"] = min(caps["min_filler"], sum(d.size for d in filler) - len(filler))
    return caps


def materialize_packs(plans, tokens_per_pack, cfg, pin=True):
    """Lay out every planned pack into pinned host tensors (see Pack), and return
    them with each pack's questions' (record, question, option order) for scoring.
    cfg carries the static sizes: cu_width, rec_width, ro_cap, q_cap."""
    T, n = tokens_per_pack, len(plans)
    alloc = lambda *shape, dtype: torch.empty(*shape, dtype=dtype, pin_memory=pin)
    out = dict(tokens=alloc(n, T, dtype=torch.int32), pos=alloc(n, T, dtype=torch.int32),
               is_q=alloc(n, T, dtype=torch.bool),
               cu_self=alloc(n, cfg.cu_width, dtype=torch.int32),
               cu_q=alloc(n, cfg.rec_width, dtype=torch.int32), cu_k=alloc(n, cfg.rec_width, dtype=torch.int32),
               filler_start=alloc(n, 1, dtype=torch.int64),
               ro_pos=alloc(n, cfg.ro_cap, dtype=torch.int64), ro_qid=alloc(n, cfg.ro_cap, dtype=torch.int64),
               ro_tgt=alloc(n, cfg.ro_cap, dtype=torch.float32), n_q=alloc(n, 1, dtype=torch.float32))
    np_ = {k: v.numpy() for k, v in out.items()}
    scored = [None] * n

    def fill(i):
        chosen, filler = plans[i]
        tokens, pos = np_["tokens"][i], np_["pos"][i]
        ends, o = [], 0
        # States
        state_starts = []
        for r, _ in chosen:
            L = r["state"].size
            tokens[o:o + L], pos[o:o + L] = r["state"], np.arange(L)
            state_starts.append(o)
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
                tokens[o:o + L], pos[o:o + L] = toks, L_state + np.arange(L)
                ro_pos += list(o + q["prefix"].size + np.cumsum([x.size for x in opts]) - 1)
                ro_qid += [len(questions)] * len(opts)
                ro_tgt += list(q["target"][perm])
                questions.append((r, q, perm))
                o += L; ends.append(o)
        q_end = o
        # Filler
        for doc in filler:
            tokens[o:o + doc.size], pos[o:o + doc.size] = doc, np.arange(doc.size)
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
        np_["filler_start"][i][0] = q_end
        np_["ro_pos"][i][:] = 0
        np_["ro_pos"][i][:len(ro_pos)] = ro_pos
        np_["ro_qid"][i][:] = cfg.q_cap
        np_["ro_qid"][i][:len(ro_qid)] = ro_qid
        np_["ro_tgt"][i][:] = 0.0
        np_["ro_tgt"][i][:len(ro_tgt)] = ro_tgt
        np_["n_q"][i][0] = len(questions)
        scored[i] = questions

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        list(pool.map(fill, range(n)))
    return out, scored


def pack_loader(packs, start=0):
    """The materialized packs, one Pack of device tensors per call, in order from `start`."""
    for i in range(start, packs["tokens"].size(0)):
        yield Pack(**{k: v[i].to("cuda", non_blocking=True) for k, v in packs.items()})
