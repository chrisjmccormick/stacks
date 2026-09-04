import glob
import threading
from pathlib import Path
import torch
from torch import Tensor


# ------------------------------------------------------------------------------
# FlashAttention
# ------------------------------------------------------------------------------
# FA3 is fundamentally not for Blackwell; it uses FA2.
# Note that FA2 needs d_qk == d_vo and head_dim <= 128

from kernels import get_kernel

_k = get_kernel("kernels-community/flash-attn2")
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


def flash_attn_varlen_bwd(dout, q, k, v, out, softmax_lse, cu_seqlens, max_seqlen, window_size):
    """Attention backward for flash_attn_varlen_fwd_lse: returns (dq, dk, dv).
    FA2's varlen backward writes the grads into pre-allocated dq/dk/dv (they
    sit right after the saved tensors) and returns softmax_d."""
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
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
# Data Loader
# ------------------------------------------------------------------------------

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

class Shard:
    def __init__(self, tokens: Tensor, bos_id: int):
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
    def load_async(file: Path, bos_id: int):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, bos_id)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

def data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int,
                   bos_id: int, max_num_docs: int):
    """
    Generator (i.e., yields rather than returns) of one micro-batch per call:
    `num_tokens` tokens, as (inputs, targets, cu_seqlens) device tensors -- the
    packed varlen layout the forward passes consume.
    Sequences are BOS-aligned and only returned from their beginning; tokens
    past max_seq_len are discarded (the next sequence starts at the next BOS).
    Single-epoch: the generator ends when the shards run out.
    Serves training (micro_batch_tokens) and validation (eval_buffer_tokens).
    """
    # Get the list of shard files and wrap in an iterator.
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    file_iter = iter(files)

    # Load the first shard.
    tokens = _load_data_shard(next(file_iter))

    shard = Shard(tokens, bos_id)
    remaining_files = list(file_iter)
    next_shard_idx = 0
    next_shard_getter = Shard.load_async(remaining_files[0], bos_id) if remaining_files else None

    while True:
        # Get the start and end indices (within `tokens`) of the sequences to use for
        # the current micro-batch.
        result = shard.next_batch(num_tokens, max_seq_len)

        # If this shard is exhausted,
        if result is None:
            # If there are no more shards, kill the dataloader.
            if next_shard_getter is None:
                return

            # Load the next shard.
            shard = next_shard_getter()
            tokens = shard.tokens
            next_shard_idx += 1
            next_shard_getter = Shard.load_async(remaining_files[next_shard_idx], bos_id) if next_shard_idx < len(remaining_files) else None

            # Re-start the loop.
            continue

        # Locations of the documents in `tokens`. Only specifies the
        # number of documents needed, not max.
        start_idxs = torch.tensor(result[0])
        end_idxs = torch.tensor(result[1])

        # `tokens` contains the entire shard. The sequences defined by the starts and ends
        # may or may not be contiguous within `tokens`, due to some sequences being
        # truncated, so we slice them and then re-concatenate into a single tensor.
        buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])

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

        # The actual cu_seqlens array always needs to contain `max_num_docs` elements so we
        # the compiler can build a single graph.
        # We allocate that buffer here and fill it with "empty documents", i.e., setting their start index
        # to one past the end of the `_inputs` buffer.
        _cum_lengths = torch.full((max_num_docs,), num_tokens)

        # Then copy in the lengths, inserting the first document (index 0).
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 / int64 on the CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)

        yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
        )
        # Execution resumes here on the next call.
