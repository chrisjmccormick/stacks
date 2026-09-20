import glob
import json
import math
import os
import threading
import time

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


def flash_attn_varlen_fwd_lse(q, k, v, cu_seqlens, max_seqlen, window_size):
    """Attention forward that also returns the softmax LSE (H, T) fp32."""
    # FA2 takes dropout_p and softmax_scale positionally, before causal.
    out, softmax_lse, *_ = fa2._flash_attn_varlen_forward(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        0.0,                        # dropout_p
        q.shape[-1] ** -0.5,        # softmax_scale
        True,                       # causal
        window_size_left=window_size[0], window_size_right=window_size[1])
    return out, softmax_lse


def flash_attn_varlen_bwd(dout, q, k, v, out, softmax_lse, cu_seqlens, max_seqlen, window_size,
                          dq=None, dk=None, dv=None):
    """Attention backward for flash_attn_varlen_fwd_lse: returns (dq, dk, dv).
    FA2's varlen backward writes the grads into pre-allocated dq/dk/dv (they
    sit right after the saved tensors) and returns softmax_d.

    Pass dq/dk/dv to have it write into buffers the caller owns. The kernel takes its
    strides as arguments and only needs the head dimension contiguous, so a slice of a
    wider tensor works -- which is how train_stack.py gets the three backward streams to
    come out side by side in one (T, 3 * d_model) buffer, so their input-gradient matmul
    is a single K = 3 * d_model reduction instead of three K = d_model ones."""
    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv
    fa2._flash_attn_varlen_backward(
        dout, q, k, v, out, softmax_lse,
        dq, dk, dv,
        cu_seqlens, cu_seqlens,     # cu_seqlens_q, cu_seqlens_k
        max_seqlen, max_seqlen,
        0.0,                        # dropout_p
        q.shape[-1] ** -0.5,        # softmax_scale
        True,                       # is_causal
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
DATASET_DIR = "./data/climbmix_32k_8_170"

NUM_TRAIN_SHARDS   = 10      # the 988-step plan reads 8 (100M raw tokens each)
EVAL_BUFFER_TOKENS = 65536   # tokens per validation micro-batch
CAP0               = 256     # document prefix cap at micro-batch 0
CAP_RAMP_FRAC      = 0.5     # fraction of the run over which it reaches seq_len


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


def _doc_stream(bos_id):
    """The train shards as one stream of documents in corpus order, each a uint16
    array beginning with BOS. Documents split across two shards are stitched."""
    carry = None
    for k in range(1, NUM_TRAIN_SHARDS + 1):
        toks = _load_shard(os.path.join(DATASET_DIR, f"climbmix/train_{k:06d}.bin")).numpy()
        bos = np.flatnonzero(toks == bos_id)
        if bos.size == 0:
            carry = toks if carry is None else np.concatenate([carry, toks])
            continue
        if carry is not None:
            yield np.concatenate([carry, toks[:bos[0]]])
        elif bos[0] != 0:
            raise ValueError(f"shard {k} starts mid-document with no preceding shard")
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
        """Returns (starts, ends), or None if this shard is exhausted."""
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = []
        ends = []

        idx = self.i
        cur_len = 0
        while cur_len <= num_tokens:
            if idx >= n:
                return None
            cur = self.bos_idx[idx]
            starts.append(cur)
            end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                      cur + max_seq_len,
                      cur + num_tokens - cur_len + 1)
            ends.append(end)
            cur_len += end - cur
            idx += 1

        assert cur_len == num_tokens + 1
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


# ------------------------------------------------------------------------------
# Data Loader
# ------------------------------------------------------------------------------

def data_generator(split, seq_len, tokens_per_micro, table_rows, total_micro_steps=None):
    """
    Generator (i.e., yields rather than returns) of one micro-batch per call:
    `tokens_per_micro` tokens for "train" and EVAL_BUFFER_TOKENS for "val", as
    (inputs, targets, rows, cu_seqlens) device tensors -- the packed varlen
    layout the forward passes consume. `table_rows` maps a micro-batch's input
    ids to its lookup tables' rows, a (tables, tokens) int32 array, on the host.
    "train" plans and stages all `total_micro_steps` + 1 micro-batches up front,
    placing each document's first cap(i) tokens (see CAP0) and cutting at most
    one document per micro-batch to fill it exactly.
    "val" is single-epoch: sequences are BOS-aligned and only returned from their
    beginning; tokens past `seq_len` are discarded (the next sequence starts at
    the next BOS). The generator ends when the shards run out.
    """
    # This is to set the fixed size of 'cu_seqlens' for varlen.
    # Estimating 192 docs per 64K tokens.
    max_num_docs = 192 * math.ceil(tokens_per_micro / EVAL_BUFFER_TOKENS)

    with open(os.path.join(DATASET_DIR, "config.json")) as f:
        bos_id = json.load(f)["bos_id"]

    # --------------------------- Training ---------------------------
    if split == "train":
        num_tokens = tokens_per_micro
        num_micro = total_micro_steps + 1
        ramp = max(1, round(CAP_RAMP_FRAC * total_micro_steps))
        print(f"=== Planning {num_micro} micro-batches of {num_tokens:,}: document prefix cap "
              f"{CAP0} -> {seq_len} in 64-token steps over the first {ramp} micro-batches, then {seq_len} ===")

        inputs = torch.empty((num_micro, num_tokens), dtype=torch.int32, pin_memory=True)
        targets = torch.empty((num_micro, num_tokens), dtype=torch.int64, pin_memory=True)
        rows = torch.empty((num_micro, 3, num_tokens), dtype=torch.int32, pin_memory=True)
        inp_np, tgt_np, rows_np = inputs.numpy(), targets.numpy(), rows.numpy()  # views: write straight into pinned memory
        starts = [[] for _ in range(num_micro)]

        docs = _doc_stream(bos_id)
        num_docs, raw_tokens = 0, 0
        t0 = time.perf_counter()
        for i in range(num_micro):
            # Rounded to a multiple of 64.
            cap = seq_len if i >= ramp else 64 * round(CAP0 * (seq_len / CAP0) ** (i / ramp) / 64)
            pos = 0
            while pos < num_tokens:
                doc = next(docs, None)
                assert doc is not None, \
                    f"document stream exhausted: {NUM_TRAIN_SHARDS} train shards do not cover this horizon"
                L = doc.size
                num_docs += 1
                raw_tokens += L
                n = min(L, cap, num_tokens - pos)   # capped, or cut to fill
                inp_np[i, pos:pos + n] = doc[:n]
                if n < L:                           # cut short
                    tgt_np[i, pos:pos + n] = doc[1:n + 1]
                else:                               # ran to its end
                    tgt_np[i, pos:pos + n - 1] = doc[1:n]
                    tgt_np[i, pos + n - 1] = bos_id
                starts[i].append(pos)
                pos += n

        # cu_seqlens (checked against the BOS positions in the inputs) and the tables' rows.
        docs_per_micro = np.array([len(s) for s in starts])
        cu_width = max(max_num_docs, 64 * math.ceil((int(docs_per_micro.max()) + 1) / 64))
        cu = torch.full((num_micro, cu_width), num_tokens, dtype=torch.int32, pin_memory=True)
        for i in range(num_micro):
            b = np.flatnonzero(inp_np[i] == bos_id)
            assert np.array_equal(b, np.array(starts[i])), f"micro-batch {i}: BOS positions != planned starts"
            cu[i, :b.size] = torch.from_numpy(b.astype(np.int32))
            rows_np[i] = table_rows(inp_np[i])

        train_tokens = num_micro * num_tokens
        print(f"  planned in {time.perf_counter() - t0:.1f}s "
              f"({(inputs.numel() * 4 + targets.numel() * 8 + rows.numel() * 4) / 2**30:.1f} GiB pinned): "
              f"{num_docs:,} docs, {raw_tokens:,} raw tokens -> {train_tokens:,} trained "
              f"({100 * (1 - train_tokens / raw_tokens):.1f}% discarded by the cap and the fill)")
        print(f"  docs per micro-batch: first {docs_per_micro[0]}, mean {docs_per_micro.mean():.0f}, "
              f"max {docs_per_micro.max()} (cu_seqlens width {cu_width})")

        for i in range(num_micro):
            yield (inputs[i].to("cuda", non_blocking=True),
                   targets[i].to("cuda", non_blocking=True),
                   rows[i].to("cuda", non_blocking=True),
                   cu[i].to("cuda", non_blocking=True))
        return

    # -------------------------- Validation --------------------------
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

        # Locations of the documents in the shard. Only specifies the
        # number of documents needed, not max.
        start_idxs = torch.tensor(result[0])
        end_idxs = torch.tensor(result[1])

        # `shard.tokens` holds the entire shard. The sequences defined by the starts and
        # ends may or may not be contiguous within it, due to some sequences being
        # truncated, so we slice them and then re-concatenate into a single tensor.
        buf = torch.cat([shard.tokens[i:j] for i, j in zip(start_idxs, end_idxs)])

        # `buf` contains `num_tokens + 1` tokens to allow for the inputs vs.
        # targets offset.
        _inputs = buf[:-1] # All tokens minus the last
        _targets = buf[1:] # Shift the tokens to the left, so that targets contains the
                           # next token for each input token.

        # The final document includes an extra token that is the target of the last
        # token in the last document. Now that we have our `_targets`, we can remove it.
        end_idxs[-1] -= 1

        # Calculate the start indices of the documents within `_inputs`. (flashattention
        # start_idxs are relative to the `tokens` buffer, so we convert them by
        # accumulating the document lengths.
        # cum_lengths starts with the second document, so we'll shift
        cum_lengths = (end_idxs - start_idxs).cumsum(0)

        # One entry per doc plus the leading 0 must fit the fixed buffer.
        assert len(cum_lengths) < max_num_docs, \
            f"micro-batch packed {len(cum_lengths)} docs; cu_seqlens holds only {max_num_docs}"

        # We allocate that buffer here and fill it with "empty documents", i.e., setting
        # their start index to one past the end of the `_inputs` buffer.
        _cum_lengths = torch.full((max_num_docs,), num_tokens)

        # Then copy in the lengths, inserting the first document (index 0).
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 / int64 on the CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _rows = torch.from_numpy(table_rows(_inputs.numpy()))

        yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _rows.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
        )
        # Execution resumes here on the next call.
