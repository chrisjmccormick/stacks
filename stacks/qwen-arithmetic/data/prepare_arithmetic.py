r"""Pre-tokenize ChrisMcCormick/basic-arithmetic into one parquet, ready for
training.

Moves every tokenizer-shaped decision — the ChatML render, the integer gold —
out of the training loop and into an artifact you can inspect. The training
script then loads token ids: it never renders a prompt.

The dataset ships its own `prompt_text` (question through Qwen's chat template
with the default system message). This script RE-RENDERS from `question` and
asserts equality, so the prompt contract is pinned in this source rather than
trusted from the artifact — a dataset revision that changed the phrasing would
fail loudly here instead of silently moving the token ids.

Deliberately NOT committed to git: the output is ~1 MB and fully determined by
(tokenizer, dataset revision). It IS published, back into the dataset repo it
came from, under `pretokenized/` — so `train_qwen_arithmetic.py` can
download the finished artifact instead of re-deriving it on a cold Colab box
(the same trade data/prepare_model.py makes for the weights). Every box can
still regenerate it; the sidecar JSON records what produced it, and
`ids_sha256` says whether two copies are the same token stream.

Reads `tokenizer.json` out of the local cache, where data/prepare_model.py put
it, so run that first.

    python data/prepare_arithmetic.py

Set PUSH = False to build without republishing. Publishing needs a
write-scoped HF token (`HF_TOKEN` in the environment).

No command line: every knob is a constant below, edited in place — the same
rule train_qwen_arithmetic.py applies to ArithConfig.

NOTE the folder is `data/` — the repo-wide convention for pre-processing
scripts — and deliberately not `datasets/`, which as a namespace portion on
sys.path would shadow HF packages.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token, snapshot_download
from tokenizers import Tokenizer

DATASET_ID = "ChrisMcCormick/basic-arithmetic"
SPLITS = ("train", "val", "test_id", "test_ood")
OUT_DIR = Path.home() / ".cache" / "qwen-arithmetic" / "data"
TOKENIZER = OUT_DIR / "tokenizer.json"   # placed by data/prepare_model.py
MAX_PROMPT = 256   # the trainer's assumption; measured max is ~52 tokens

# Where the finished artifact goes back to. A subfolder, not the repo root:
# the root parquets ARE the dataset's splits, and dropping a fifth one there
# with a different schema would show up as a phantom split in the viewer.
# The name is the tokenizer's, not the model's — every Qwen2.5 size shares it.
PUSH = True
PRETOK = "pretokenized/qwen2.5"          # .parquet + .json in the dataset repo

# For the operand check below. The dataset ships `a`, `b` in OPERATOR order,
# which for 49 of test_ood's rows is not the order the question names them in.
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
       "*": lambda a, b: a * b, "/": lambda a, b: a / b if b else None}

IM_END = 151645       # <|im_end|>    ends the assistant turn
ENDOFTEXT = 151643    # <|endoftext|> ends the document

# Qwen2.5-Instruct's chat template inserts this system message when the caller
# supplies none. Spelled out here rather than read from the jinja: it is a
# prompt contract, and it should be pinned in the source alongside the run.
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def render(question: str) -> str:
    """One user turn through Qwen's ChatML, primed for the assistant — must
    match the dataset's own `prompt_text` byte for byte (asserted below)."""
    return (f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main():
    assert TOKENIZER.exists(), (f"{TOKENIZER} not found — build it with:\n"
                                f"    python data/prepare_model.py")
    tok = Tokenizer.from_file(str(TOKENIZER))
    assert (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")) \
        == (IM_END, ENDOFTEXT), "tokenizer disagrees on the Qwen special ids"

    # The split parquets land beside this script's output; the hf-vllm sibling
    # reads them raw from the same place.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(DATASET_ID, repo_type="dataset", local_dir=str(OUT_DIR),
                      allow_patterns=[f"{s}.parquet" for s in SPLITS])

    cols = {c: [] for c in ("split", "idx", "question", "gold", "op", "a", "b",
                            "difficulty", "prompt_ids", "prompt_len")}
    for split in SPLITS:
        t = pq.read_table(OUT_DIR / f"{split}.parquet").to_pydict()
        assert "a" in t and "b" in t, (
            f"{split}.parquet has no operand columns — the dataset revision "
            f"predates them (agent-ops/stacks/2026-08-16_1133am_"
            f"qwen-arithmetic-t4-banks/add_operands.py adds them)")
        for i, (q, ans, op, a, b, diff, pt) in enumerate(zip(
                t["question"], t["answer"], t["op"], t["a"], t["b"],
                t["difficulty"], t["prompt_text"])):
            # The reward compares integers; a non-integer gold would be
            # unscorable, so it is a dataset defect, not a row to skip.
            assert ans == int(ans), f"non-integer answer {ans!r} in {split}[{i}]"
            # Operands are carried through, not re-parsed — but they are
            # re-checked here, the same way the prompt is re-rendered rather
            # than trusted: `a op b == answer` is the property every consumer
            # will assume, so it is pinned in this source.
            got = OPS[op](a, b)
            assert got is not None and abs(got - ans) < 1e-9, \
                f"operands disagree at {split}[{i}]: {a} {op} {b} != {ans}"
            text = render(q)
            assert text == pt, f"prompt contract drifted at {split}[{i}]: {pt!r}"
            # add_special_tokens=False adds no template tokens; the ChatML
            # specials in the text itself still map to their single ids.
            ids = tok.encode(text, add_special_tokens=False).ids
            cols["split"].append(split)
            cols["idx"].append(i)
            cols["question"].append(q)
            cols["gold"].append(int(ans))
            cols["op"].append(op)
            cols["a"].append(int(a))
            cols["b"].append(int(b))
            cols["difficulty"].append(diff)
            cols["prompt_ids"].append(ids)
            cols["prompt_len"].append(len(ids))

    lens = cols["prompt_len"]
    assert max(lens) <= MAX_PROMPT, f"prompt of {max(lens)} exceeds {MAX_PROMPT}"
    assert min(lens) >= 2, "prompt too short for the forced-last-token split"
    # Spot-check the decode round-trip: the trainer reads completions back
    # with skip_special_tokens=False, so encode/decode must agree on specials.
    for ids, q in list(zip(cols["prompt_ids"], cols["question"]))[::500]:
        assert tok.decode(ids, skip_special_tokens=False) == render(q)

    out = OUT_DIR / "arithmetic.parquet"
    pq.write_table(pa.table({
        "split":      pa.array(cols["split"], pa.string()),
        "idx":        pa.array(cols["idx"], pa.int32()),
        "question":   pa.array(cols["question"], pa.string()),
        "gold":       pa.array(cols["gold"], pa.int64()),
        "op":         pa.array(cols["op"], pa.string()),
        "a":          pa.array(cols["a"], pa.int64()),
        "b":          pa.array(cols["b"], pa.int64()),
        "difficulty": pa.array(cols["difficulty"], pa.string()),
        "prompt_ids": pa.array(cols["prompt_ids"], pa.list_(pa.int32())),
        "prompt_len": pa.array(cols["prompt_len"], pa.int32()),
    }), out, compression="zstd")

    n_split = {s: sum(x == s for x in cols["split"]) for s in SPLITS}
    meta = dict(dataset=DATASET_ID, system=QWEN_SYSTEM,
                im_end=IM_END, endoftext=ENDOFTEXT, **n_split,
                prompt_len_min=min(lens), prompt_len_max=max(lens),
                prompt_len_mean=round(sum(lens) / len(lens), 1),
                # Pins the exact token stream: a tokenizer or prompt change moves this.
                ids_sha256=hashlib.sha256(
                    repr(cols["prompt_ids"]).encode()).hexdigest()[:32])
    side = OUT_DIR / "arithmetic.json"
    side.write_text(json.dumps(meta, indent=1))

    print(" | ".join(f"{s} {n_split[s]:,}" for s in SPLITS))
    print(f"  prompt tokens min {meta['prompt_len_min']} "
          f"mean {meta['prompt_len_mean']} max {meta['prompt_len_max']}")
    print(f"  ids sha256: {meta['ids_sha256']}")
    print(f"  -> {out} ({out.stat().st_size / 2**20:.1f} MB)")

    if not PUSH:
        print("  PUSH = False -- not publishing.")
        return
    # setup.sh runs this on every fresh GPU box, most of which have no
    # write-scoped token and do not need one -- the artifact is already
    # published, and this rebuild is a local no-op. So a missing token is a
    # note, not a failure; a token that is present but cannot write still
    # raises, because that IS a surprise.
    if get_token() is None:
        print("  no HF token -- built locally, not published.")
        return
    # Back into the dataset repo the splits came from. Idempotent: the hub
    # skips a file whose hash already matches, so rerunning after a no-op
    # rebuild costs two HEADs.
    api = HfApi()
    for src, ext in ((out, ".parquet"), (side, ".json")):
        api.upload_file(path_or_fileobj=str(src), path_in_repo=PRETOK + ext,
                        repo_id=DATASET_ID, repo_type="dataset",
                        commit_message=f"pre-tokenized copy (ids sha {meta['ids_sha256'][:12]})")
    print(f"  -> https://huggingface.co/datasets/{DATASET_ID}/tree/main/"
          f"{PRETOK.rsplit('/', 1)[0]}")


if __name__ == "__main__":
    main()
