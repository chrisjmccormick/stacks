r"""Pre-tokenize GSM8K into one parquet per prompt variant, ready for training.

Moves every tokenizer-shaped decision — chat template, prompt suffix, gold
normalization, the end-anchored answer suffix — AND the choice of validation
problems out of the training loop and into an artifact you can inspect. The
training script then loads token ids: it never renders a prompt and never
decides what to validate on.

Deliberately NOT committed: the output is a couple of MB per variant and fully
determined by (tokenizer, prompt, gsm8k revision), so it is cheap to regenerate
and pointless to version. Regenerate on each box; the sidecar JSON records what
produced it.

Reads `tokenizer.json` out of the local cache, where data/prepare_model.py put
it, so run that first.

    python data/prepare_gsm8k.py

No command line: every knob is a constant below, edited in place — the same rule
train_qwen_gsm8k.py applies to GSM8KConfig. All three prompt variants are built
in one pass (seconds each), so `cfg.prompt` can name any of them without a
regeneration step.

NOTE the folder is `data/` — the repo-wide convention for pre-processing
scripts — and deliberately not `datasets/`, which as a namespace portion on
sys.path would shadow the HF `datasets` package this script imports.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tokenizers import Tokenizer

OUT_DIR = Path.home() / ".cache" / "qwen-gsm8k" / "data"
TOKENIZER = OUT_DIR / "tokenizer.json"   # placed by data/prepare_model.py
MAX_PROMPT = 512   # verl's prompt-length filter; assert nothing exceeds it

IM_END = 151645       # <|im_end|>    ends the assistant turn
ENDOFTEXT = 151643    # <|endoftext|> ends the document

# Qwen2.5-Instruct's chat template inserts this system message when the caller
# supplies none. Spelled out here rather than read from the jinja: it is a
# prompt contract, and it should be pinned in the source alongside the run.
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# The prompt is part of the dataset, not the trainer: changing it changes the
# token ids, so it names the artifact.
#   hash       — verl's stock preprocessor, the reference baseline
#   boxed      — the minimal edit of that sentence to \boxed{}
#   boxed_qwen — the phrasing Qwen2.5-Math was post-trained on
PROMPTS = {
    "hash":       " Let's think step by step and output the final answer after \"####\".",
    "boxed":      " Let's think step by step and put your final answer within \\boxed{}.",
    "boxed_qwen": " Please reason step by step, and put your final answer within \\boxed{}.",
}

# The in-loop validation set, flagged into the parquet as `is_val`: 256 TEST
# problems sampled from the MOVERS of a reference training trajectory — problems
# whose solve rate actually changes — with proportional allocation over step-0
# difficulty bins. A dead always-0/8 (or 8/8) item contributes a constant to the
# mean, so dropping it costs zero correlation with the full-test number and only
# raises signal-to-noise. Keyed by sha1(question)[:16] so the set survives any
# reindexing of the HF dataset.
#
# Selected in agent-ops/nanochat/2026-08-04_0413pm_verl-gsm8k-reference
# (select_eval_subset.py, OPTIMIZATION.md § 11), which found that SIZE matters
# more than selection and recommended n=256. Re-scored on an independent
# trajectory in agent-ops/stacks/2026-08-09_0432pm_horizon-sweep, predicting the
# full-test number over 11 checkpoints:
#
#     n=128  residual 0.89pp,  LOO mean/max 0.98 / 2.65
#     n=256  residual 0.64pp,  LOO mean/max 0.70 / 2.20   <- this set
#     a RANDOM 256 (0.85) still beats the hand-selected 128 (0.89)
#
# BEWARE what the number means. It tracks the trajectory; it does not stand in
# for the full test. It over-reports GAINS — the affine map is
# full ~= 10.50 + 0.738 x subset — so the subset-minus-full gap WIDENS as the
# score rises (measured -1.2pp early, +8.4pp late on a 128-problem set). A
# constant offset is the wrong model.
VAL_QIDS = """
b8bf5ec3b13b51d1 b7269b7bbca3a2fd dcf04c89e50933cd 5dc9fb4d42fff12d 0a46ac603b89702c
42551aa157349b2e f44770261678b47f 7505c6a6c78700c7 719d9ab214fd5f2a 5a33532be2f4c509
eade53c151c9e85e b9fcfb64c1eac9e2 c8b8091ba339cb74 4baf18694c20b18d 7017227e9c199f44
52efeefb5c8d96bd 897513910dad8c57 deb43aee72d586cf e64a5ee3035f94c6 38e6c41a6067b3a0
74c6e4b9d5095115 1dc309eb3fe61379 475c701016c7919c e9ee81f9b6669299 b6940c2c479cf0ba
12c896e50107368c 92284b53892edece bc81853cf6cf722c 40d1c6d764ffc511 7f53a94856b64776
a8187ba4bb9cb0d3 beff955d2365a8ad 8df469f1653d4dd5 217f09f2fc232ddc 4d550745c057cb02
15faca8a11cd047a afe63467e8c3f64a 102b97a31a015a10 cf0f5f8489dbb3e2 a0ee9f99726e8748
503934bd06d6528e 6f8e4ab2a47c2d63 84c2df5d4d8578ac d0f97b2b5a83422f 524ae69eda124ac9
621e8cf6e269903e 4d58bf747df75153 8f344f2eb8d7da7f 9871e6f98efd0f3b 36696fa858005c71
80cdff0fd74e0566 f1815b1f71fe4b58 68d1eb1ecec57044 77fd023cfdd0554b fa110de857d572e9
0d60b8892e08414d 80aca3ab084c060c 02a919262b8c6a1f 1262c067228137f2 abb93ec7c040e32d
592f7d4c5a5139ac a9df7adfe56bf8e4 9eb5a53e755cf4f7 93502e5a9b633708 0034d757745b7b67
d0006e7cb1942d8e afff6c6e55d6f28f 77b141f7b38d6d1f 5a7ea1946e6f4d82 f52498d0434d5282
76474afff297d0c8 ee2963603f043034 106444ee8f789cd3 4d6a1cbd5ac9b89b c84cf2a191982935
531051ff89627921 5a59cab01e14df0e 385694b2261bffcf c0165a13489cd5b6 96cd26025cdacc3f
2084c08fc7cef9d1 1bb44ee7c3651f7d 71fb433a1ece230c b476c01e6e8cb28e ae93757c15aac78f
26ea2daf2e0106f5 6b0a1e250700908b c1559e7303ed5168 69ee26f38cf71f54 2dc25b1b66d1fc28
c0b83e044c20f7ec fce638941bbcd9c6 bb15962871374527 4e3c00c973bdf5c9 cf180baba52447dc
e06b7791d4ce8491 ca4b6a837db3a05c 68fb156d39025083 9632bba84f204132 f20827846dd5dafb
cb434c87934c3895 8271a7f1ea8ffd2f ee9718fe727a6d2a c95e9b96c5149d22 f625f5bf10953102
e321b0b5d35942b1 4c02bab50866ea86 f869df2a72a6d1a6 d0b6d688ce2f5806 06e18b2f25c7d2ec
60925f8082f7c9a9 a81215ea31e91c4c 1277f3fdbbb5fdc0 1fc5b7f415166cb5 40df555a53993e4d
48fc09e49f87939d 13de9ec57f1ee3d2 25bacf2e9ec32848 c81e62e786b215e0 f4428dec7cc485fc
92a42b9dd575de84 7d102ff34bbb0dd2 7132709c7b25a2e8 d083d86e55e27b40 f15d2cb7edbcca59
dc0af2bdfebe7143 ed5a5ee81b99227f a65c94a578fe1230 a52a4e161b65abb7 b2c145be62289f05
e28fb8a153d8d719 ca0f679b5b5cd8ee 5dab50f773b81be4 16633ad9483b0b23 5900d720bc12af2f
9c381ddea4cabafe 1c811badfe826f9d 8255302673e6dbd9 991e2ca116abe37e a2dc1c3de7bf889f
5a06a3c5894e5358 d382013a47e97d46 2b5d2f1f660ddade 3b1eee4ebb866855 04157051480e633a
be6c02ad0f63d6d8 4294812abbc868d2 ec338c9488a3f299 b3cf5970b2aee90e 6dbe8d912a5455da
c3b8f7a7b459f1b9 50fef1f327cff6f5 ab4be2991a25955b de7c116bdc7ec391 24e97e4963a111dd
31e606775d93c8ec d0a4e6beb6675589 279f69c9c9144cb2 f96a38bc681a72e7 b4693495fa3a1c0f
3a6497d7aa5466d8 0c63fbc631c8a1df d62b943cd388b3e6 966ed29677c51872 9df7525ee555dae0
e82c1b565fee8639 8db84239fc20f5f9 9abcd1bd455e12fc acd88185417bfb57 1464d900c287dfbd
0549bf32fda215da f9365303ac275df9 ed3b100d3382a8a3 317157e12a5b9a7f e6c6e3588355635e
d9f901a1c028d340 edb3ed4fb3a59e7f e2983bfcf28080ca 5a65b559815c0996 19e28dc0d6b9ce69
57f9c71cf482f36b 23bdb2fe7ed4ef8a 10dfdafe84cd3a3d bfcc209fbfd4e7ea 08ec8c968599ee35
365fc081a02e8b80 34bcabae57d76178 e1df157cdd59d76d 290d87096b3514cb 9c2b7ab9cbc102ac
389c696982d474a1 ea00c72f0fb07a37 153b62877316ac3d d32073da1845d440 f7feb7b7d76d3952
ad78adf1538870cd 8369448341de80a3 c826e38bb4bfac47 1f974f1c74041e37 4dfdb87b7896103f
158d4e2231078ddf 3e25b1995ab115a6 163fb24c18ba9c00 5f0973e6ec3c8045 6566211ce85ecd1d
86856317da988346 f7179089014be48e e37ffd27f4d22c87 cabf13b64bc8e0e5 bdd7e13b72ebc508
1eac763019568e2f ef70313131442cb5 390e8bd1fe16a6d7 9dd0431e964f43af 5236d96fd36d9027
788b6ef6222e8e6b 927d712d3e30c3a1 bea9c37be79c80e9 47f268eb215ecc1c 5c72b44b1bc0c9fd
222237e4fcca1544 46f5db890c105032 e0dafe7c15cc6533 9068e069266e1ca5 6beae1dd8e336b47
3daf8af5dae9064f b5c0a9421ef7f85c 8e9a535d7907f093 53b5941f4efcad5b 06a204f3c77d2375
d325fba71a23ab3d b7a60af52254e42b e92c398334fbecfc e0742ba9f3d36b6b 8f79e693a02f0d60
a24cb4136ebdf621 ae6fd0e58aa7f44c 7c0a4fbcf06e55fe 4e0c9fe82a8d7cfd e858f093c75b24fa
ff09b7b64075b0b4 c8548c503eaceccc 2e8e0e948bbcc580 94187b6b153a201b 1e2269f5fce21297
e4415f5faf566bca 7fe567b35871fc8f 44f9bfd9e2462853 728aecb32e648503 d815d115072f9fe7
8d0eda562a8094ef 6efe65d5b64ba3ab fa6b6c6a22069197 34a5ea0e48d085ef bc8861eeacf79623
686c67ea2cc03b88
""".split()
N_VAL = 256
assert len(set(VAL_QIDS)) == len(VAL_QIDS) == N_VAL

_GOLD = re.compile(r"^-?\d+$")


def gold_of(answer_field: str) -> str:
    """GSM8K's reference answer is whatever follows the final `####`. Every one
    of the 8,792 is a plain integer (asserted): keeping it exact matters,
    because the reward compares the model's digits against these."""
    raw = answer_field.split("####")[-1].strip().replace(",", "").replace("$", "")
    assert _GOLD.match(raw), f"non-integer gold {raw!r} in {answer_field[-60:]!r}"
    return str(int(raw))


def build(prompt: str, tok: Tokenizer, ds) -> None:
    """One prompt variant -> gsm8k_<prompt>.parquet + its sidecar JSON."""
    suffix = PROMPTS[prompt]
    val = set(VAL_QIDS)
    cols = {c: [] for c in ("split", "idx", "qid", "question", "gold", "is_val",
                            "prompt_ids", "prompt_len", "answer_suffix_ids")}
    for split in ("train", "test"):
        qs, ans = ds[split]["question"], ds[split]["answer"]
        for i, (q, ansr) in enumerate(zip(qs, ans)):
            gold = gold_of(ansr)
            qid = hashlib.sha1(q.strip().encode("utf-8")).hexdigest()[:16]
            # One user turn through Qwen's ChatML, primed for the assistant.
            ids = tok.encode(f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n"
                             f"<|im_start|>user\n{q}{suffix}<|im_end|>\n"
                             f"<|im_start|>assistant\n", add_special_tokens=False).ids
            cols["split"].append(split)
            cols["idx"].append(i)
            cols["qid"].append(qid)
            cols["question"].append(q)
            cols["gold"].append(gold)
            # The validation tracker is drawn from the test split only — a train
            # problem carrying the flag would mean a qid collision across splits.
            cols["is_val"].append(split == "test" and qid in val)
            cols["prompt_ids"].append(ids)
            cols["prompt_len"].append(len(ids))
            # What a completion must END with under the end-anchored reward.
            # The leading backslash is deliberately absent, and the [1:] drops
            # its token: `\` MERGES with whatever precedes it, so `\boxed` is
            # [59, 79075] at a line start but [1124, 79075] after a space — and
            # after a space is the common case. From the `boxed` token onward
            # the ids are identical in every context checked.
            cols["answer_suffix_ids"].append(
                tok.encode(rf"\boxed{{{gold}}}", add_special_tokens=False).ids[1:] + [IM_END])

    lens = cols["prompt_len"]
    assert max(lens) <= MAX_PROMPT, f"prompt of {max(lens)} exceeds {MAX_PROMPT}"
    assert min(lens) >= 2, "prompt too short for the forced-last-token split"
    n_qid = len(set(cols["qid"]))
    assert n_qid == len(cols["qid"]), f"qid collision: {n_qid} unique of {len(cols['qid'])}"
    n_val = sum(cols["is_val"])
    assert n_val == N_VAL, f"val set matched {n_val}/{N_VAL} qids — dataset revision moved?"

    out = OUT_DIR / f"gsm8k_{prompt}.parquet"
    pq.write_table(pa.table({
        "split":             pa.array(cols["split"], pa.string()),
        "idx":               pa.array(cols["idx"], pa.int32()),
        "qid":               pa.array(cols["qid"], pa.string()),
        "question":          pa.array(cols["question"], pa.string()),
        "gold":              pa.array(cols["gold"], pa.string()),
        "is_val":            pa.array(cols["is_val"], pa.bool_()),
        "prompt_ids":        pa.array(cols["prompt_ids"], pa.list_(pa.int32())),
        "prompt_len":        pa.array(cols["prompt_len"], pa.int32()),
        "answer_suffix_ids": pa.array(cols["answer_suffix_ids"], pa.list_(pa.int32())),
    }), out, compression="zstd")

    meta = dict(prompt=prompt, prompt_suffix=suffix, system=QWEN_SYSTEM,
                dataset="openai/gsm8k:main", im_end=IM_END, endoftext=ENDOFTEXT,
                n_train=sum(s == "train" for s in cols["split"]),
                n_test=sum(s == "test" for s in cols["split"]),
                n_val=n_val,
                prompt_len_min=min(lens), prompt_len_max=max(lens),
                prompt_len_mean=round(sum(lens) / len(lens), 1),
                # Pins the exact token stream: a tokenizer or prompt change moves this.
                ids_sha256=hashlib.sha256(
                    repr(cols["prompt_ids"]).encode()).hexdigest()[:32])
    (OUT_DIR / f"gsm8k_{prompt}.json").write_text(json.dumps(meta, indent=1))

    print(f"[{prompt}] {meta['n_train']:,} train + {meta['n_test']:,} test "
          f"({n_val} val) | prompt tokens min {meta['prompt_len_min']} "
          f"mean {meta['prompt_len_mean']} max {meta['prompt_len_max']}")
    print(f"  suffix: {suffix.strip()}")
    print(f"  ids sha256: {meta['ids_sha256']}")
    print(f"  -> {out} ({out.stat().st_size / 2**20:.1f} MB)")


def main():
    assert TOKENIZER.exists(), (f"{TOKENIZER} not found — build it with:\n"
                                f"    python data/prepare_model.py")
    tok = Tokenizer.from_file(str(TOKENIZER))
    assert (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")) \
        == (IM_END, ENDOFTEXT), "tokenizer disagrees on the Qwen special ids"

    ds = load_dataset("openai/gsm8k", "main")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for prompt in PROMPTS:
        build(prompt, tok, ds)


if __name__ == "__main__":
    main()
