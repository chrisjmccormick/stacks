r"""Tokenize Kev's decision suites into encoder-rtx's input format, one parquet per split.

Kev (github.com/jaredpalmer/kev, Apache-2.0) publishes the data its 4B model trained on:
ten public classification datasets with their own labels, generated policy and rule
cases, CFPB complaint narratives, generated skill families and developer-tooling sets.
It also publishes the development and test splits its model cards report. This script
renders each record (one state, its named questions) into token ids, so the training
script never renders text. Every tokenizer-shaped decision lives here.

The format (see agent-ops-stacks/encoder-rtx/.../PROPOSAL.md):

    <|state|>{state}
    <|choice|>{instructions}\n<|"|>name<|"|> description<|option|><|"|>name<|"|><|option|>...
    <|noul|>{instructions}\n{true criteria}<|yes|>{false criteria}<|no|>
    <|score|>{instructions}\n{level 0 text}<|level_0|>{level 1 text}<|level_1|>...

Each question is stored as a prefix (its type token and instructions) plus a list of
options, each ending in its readout token. The trainer concatenates them, so it can
permute a choice's options without re-tokenizing. JSON states, instructions and
descriptions keep their braces and keys as text, with string values in super quotes:
{policy:<|"|>...<|"|>,case:<|"|>...<|"|>}.

The vocabulary is decoder-rtx's 32,768-token BPE (ChrisMcCormick/climbmix_32k_8_170),
unchanged; our 19 special tokens take ids 32,768-32,786, inserted directly between
encoded text pieces. BPE never emits them, so user text can't forge a delimiter.

Outputs, in ~/.cache/encoder-rtx/data/ (regenerable, so not committed):
    kev_<split>.parquet   one row per record; the splits are below
    kev_meta.json         special token ids, sources and their hashes, counts, lengths

No command line: every knob is a constant below. On the Mac:

    uv run --with tiktoken --with pyarrow --with huggingface_hub python data/prepare_kev_suites.py

On a GPU box the `stacks` env already has all three: `python data/prepare_kev_suites.py`.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

CACHE = Path.home() / ".cache" / "encoder-rtx"
RAW_DIR = CACHE / "kev-suites"
OUT_DIR = CACHE / "data"

# decoder-rtx's tokenizer: nanochat's rustbpe vocabulary, pickled as a tiktoken Encoding.
CLIMBMIX_REPO, CLIMBMIX_REV = "ChrisMcCormick/climbmix_32k_8_170", "cb86ce96bce6cc942a976768603f483046c5d3e4"
TOKENIZER = CACHE / "climbmix_32k_8_170" / "tokenizer" / "tokenizer.pkl"

# Kev's suites, pinned. The large train files are on the Hub; the rest are in the GitHub
# repo's evals/ folder.
KEV_HF_REPO, KEV_HF_REV = "jaredpalmer/kev-suites", "cc4bac803e73112689ec327ffa481c519cbc7a05"
KEV_GH = "https://raw.githubusercontent.com/jaredpalmer/kev/2074cc44256200bb37540103fa4572901faa10f3/evals"

# Output split -> the files it concatenates, each (where, path). "train" is exactly what
# Kev-4B trained on (its card's five stages). transfer_test is Kev's locked test: keep it
# out of development reads.
SPLITS = {
    "train": [("hf", "v7/decision-v7/train.jsonl"),          # ten public datasets + generated policy/rules
              ("gh", "night2/dates_unknowable.jsonl"),       # dates, and cases with the evidence removed
              ("hf", "documents-v1/train.jsonl"),            # CFPB complaint narratives
              ("hf", "hard-v1/train.jsonl"),                 # seven generated skill families
              ("gh", "devtools-v1/train.jsonl")],            # developer-tooling datasets
    "decision_dev":   [("gh", "v7/decision-v7/development.jsonl")],
    "decision_test":  [("gh", "v7/decision-v7/test.jsonl")],
    "calibration":    [("gh", "v7/decision-v7/calibration.jsonl")],
    "transfer_dev":   [("gh", "v4/transfer-v4/development.jsonl")],
    "transfer_test":  [("gh", "v4/transfer-v4/test.jsonl")],
}

# ------------------------------------------------------------------------------
# Special tokens
# ------------------------------------------------------------------------------

BPE_VOCAB = 32768    # decoder-rtx's vocabulary; nanochat's 9 specials sit at 32,759-32,767
SPECIAL = (["<|state|>", "<|noul|>", "<|score|>", "<|choice|>", '<|"|>', "<|mask|>",
            "<|yes|>", "<|no|>", "<|option|>"]
           + [f"<|level_{i}|>" for i in range(10)])
SPECIAL_IDS = {name: BPE_VOCAB + i for i, name in enumerate(SPECIAL)}
D_VOCAB = 32832      # embedding rows: BPE + specials, padded to a multiple of 64
assert BPE_VOCAB + len(SPECIAL) <= D_VOCAB and D_VOCAB % 64 == 0

STATE, QUOTE = SPECIAL_IDS["<|state|>"], SPECIAL_IDS['<|"|>']
TYPE_TOKEN = {"noul": SPECIAL_IDS["<|noul|>"], "score": SPECIAL_IDS["<|score|>"],
              "choice": SPECIAL_IDS["<|choice|>"]}
YES, NO, OPTION = SPECIAL_IDS["<|yes|>"], SPECIAL_IDS["<|no|>"], SPECIAL_IDS["<|option|>"]
LEVELS = [SPECIAL_IDS[f"<|level_{i}|>"] for i in range(10)]


# ------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------

def json_pieces(value) -> list:
    """A JSON value as pieces: str for BPE text, int for a special id. Braces, brackets,
    keys and non-string scalars are text; string values go in super quotes."""
    if isinstance(value, str):
        return [QUOTE, value, QUOTE]
    if isinstance(value, dict):
        pieces = ["{"]
        for i, (k, v) in enumerate(value.items()):
            pieces += ([","] if i else []) + [f"{k}:"] + json_pieces(v)
        return pieces + ["}"]
    if isinstance(value, list):
        pieces = ["["]
        for i, v in enumerate(value):
            pieces += ([","] if i else []) + json_pieces(v)
        return pieces + ["]"]
    return [json.dumps(value)]   # numbers, booleans, null


def text_pieces(value) -> list:
    """A field that is either plain text (rendered as is) or JSON."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else json_pieces(value)


def encode(enc, pieces: list) -> list[int]:
    """Adjacent text pieces are joined before BPE, so the merges see whole strings."""
    ids, text = [], ""
    for p in pieces:
        if isinstance(p, str):
            text += p
        else:
            if text:
                ids += enc.encode_ordinary(text)
                text = ""
            ids.append(p)
    if text:
        ids += enc.encode_ordinary(text)
    return ids


def render_question(enc, q: dict):
    """-> (prefix ids, [option ids, each ending in its readout token], label, target).
    label indexes the options; target is a probability per option, or [] when the
    record has only a hard label."""
    kind = q["type"]
    # The newline marks where the instructions end: a yes/no or score criterion is plain
    # text, and would otherwise run straight on from the question.
    instructions = text_pieces(q.get("instructions"))
    prefix = encode(enc, [TYPE_TOKEN[kind]] + instructions + (["\n"] if instructions else []))
    crit = q.get("criteria")
    if kind == "noul":
        crit = crit or {}
        names = ["true", "false"]
        options = [encode(enc, text_pieces(crit.get("true")) + [YES]),
                   encode(enc, text_pieces(crit.get("false")) + [NO])]
        label = 0 if q["label"] else 1
    elif kind == "score":
        assert 1 <= len(crit) <= len(LEVELS), f"score with {len(crit)} levels"
        names = [str(i) for i in range(len(crit))]
        options = [encode(enc, text_pieces(level) + [LEVELS[i]]) for i, level in enumerate(crit)]
        label = q["label"]
    else:
        names = list(crit)
        options = [encode(enc, [QUOTE, name, QUOTE]
                          + ([" "] + text_pieces(desc) if desc else []) + [OPTION])
                   for name, desc in crit.items()]
        label = names.index(q["label"])
    assert 0 <= label < len(options)
    target = []
    if "target" in q:
        assert set(q["target"]) == set(names), f"target keys {sorted(q['target'])} vs {names}"
        target = [float(q["target"][n]) for n in names]
        assert abs(sum(target) - 1) < 1e-6
    return prefix, options, label, target


# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

def fetch(where: str, path: str) -> Path:
    """The raw file, downloaded once into RAW_DIR at the pinned revision."""
    local = RAW_DIR / path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        if where == "hf":
            hf_hub_download(KEV_HF_REPO, path, repo_type="dataset", revision=KEV_HF_REV,
                            local_dir=RAW_DIR)
        else:
            urllib.request.urlretrieve(f"{KEV_GH}/{path}", local)
    return local


def load_tokenizer():
    if not TOKENIZER.exists():
        hf_hub_download(CLIMBMIX_REPO, "tokenizer/tokenizer.pkl", repo_type="dataset",
                        revision=CLIMBMIX_REV, local_dir=TOKENIZER.parent.parent)
    with open(TOKENIZER, "rb") as f:
        enc = pickle.load(f)
    assert enc.n_vocab == BPE_VOCAB, f"tokenizer has {enc.n_vocab} ids, expected {BPE_VOCAB}"
    assert enc.encode_single_token("<|bos|>") == 32759
    return enc


def decode(enc, ids: list[int]) -> str:
    """Readable rendering, special tokens by name: for the sample printout."""
    out, run = [], []
    for i in ids:
        if i >= BPE_VOCAB:
            out.append(enc.decode(run)); run = []
            out.append(SPECIAL[i - BPE_VOCAB])
        else:
            run.append(i)
    out.append(enc.decode(run))
    return "".join(out)


def pct(xs: list[int], p: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------

def build(enc, split: str, files: list) -> dict:
    cols = {c: [] for c in ("source", "variant", "group", "state_ids", "q_name", "q_type",
                            "q_prefix", "q_options", "q_label", "q_target")}
    sources, by_type, by_source = {}, Counter(), Counter()
    state_lens, question_lens = [], []
    for where, path in files:
        local = fetch(where, path)
        sources[path] = dict(where=where, sha256=hashlib.sha256(local.read_bytes()).hexdigest())
        for line in local.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            meta = r.get("_meta", {})
            state = encode(enc, [STATE] + text_pieces(r["state"]))
            cols["source"].append(meta.get("source", "?"))
            cols["variant"].append(meta.get("variant", "clean"))
            cols["group"].append(str(meta.get("group_id") or meta.get("id") or meta.get("family_id") or ""))
            cols["state_ids"].append(state)
            state_lens.append(len(state))
            names, types, prefixes, options, labels, targets = [], [], [], [], [], []
            for name, q in r["questions"].items():
                prefix, opts, label, target = render_question(enc, q)
                names.append(name); types.append(q["type"]); prefixes.append(prefix)
                options.append(opts); labels.append(label); targets.append(target)
                question_lens.append(len(prefix) + sum(len(o) for o in opts))
                by_type[q["type"]] += 1
                by_source[meta.get("source", "?")] += 1
            for c, v in zip(("q_name", "q_type", "q_prefix", "q_options", "q_label", "q_target"),
                            (names, types, prefixes, options, labels, targets)):
                cols[c].append(v)

    flat = [i for s in cols["state_ids"] for i in s]
    flat += [i for ps in cols["q_prefix"] for p in ps for i in p]
    flat += [i for qs in cols["q_options"] for opts in qs for o in opts for i in o]
    assert max(flat) < BPE_VOCAB + len(SPECIAL)
    # BPE text never lands on nanochat's specials (32,759-32,767): only our formatter
    # emits ids at or above the BPE vocabulary.
    assert not any(32759 <= i < BPE_VOCAB for i in flat)

    out = OUT_DIR / f"kev_{split}.parquet"
    ints = pa.list_(pa.int32())
    pq.write_table(pa.table({
        "source":    pa.array(cols["source"], pa.string()),
        "variant":   pa.array(cols["variant"], pa.string()),
        "group":     pa.array(cols["group"], pa.string()),
        "state_ids": pa.array(cols["state_ids"], ints),
        "q_name":    pa.array(cols["q_name"], pa.list_(pa.string())),
        "q_type":    pa.array(cols["q_type"], pa.list_(pa.string())),
        "q_prefix":  pa.array(cols["q_prefix"], pa.list_(ints)),
        "q_options": pa.array(cols["q_options"], pa.list_(pa.list_(ints))),
        "q_label":   pa.array(cols["q_label"], ints),
        "q_target":  pa.array(cols["q_target"], pa.list_(pa.list_(pa.float32()))),
    }), out, compression="zstd")

    n_questions = sum(len(n) for n in cols["q_name"])
    stats = dict(
        records=len(cols["state_ids"]), questions=n_questions,
        clean_records=sum(v == "clean" for v in cols["variant"]),
        soft_targets=sum(bool(t) for ts in cols["q_target"] for t in ts),
        by_type=dict(by_type), by_source=dict(sorted(by_source.items())),
        state_tokens=sum(state_lens), question_tokens=sum(question_lens),
        state_len_p50=pct(state_lens, .5), state_len_p90=pct(state_lens, .9), state_len_max=max(state_lens),
        question_len_p50=pct(question_lens, .5), question_len_max=max(question_lens),
        sources=sources,
        # Pins the exact token stream: a tokenizer or format change moves this.
        ids_sha256=hashlib.sha256(repr((cols["state_ids"], cols["q_prefix"], cols["q_options"])).encode()).hexdigest()[:32],
    )
    print(f"[{split}] {stats['records']:,} records, {n_questions:,} questions "
          f"({', '.join(f'{k} {v:,}' for k, v in sorted(by_type.items()))}); "
          f"state tokens {stats['state_tokens']/1e6:.2f}M (p50 {stats['state_len_p50']}, "
          f"p90 {stats['state_len_p90']}, max {stats['state_len_max']}), "
          f"question tokens {stats['question_tokens']/1e6:.2f}M (p50 {stats['question_len_p50']}, "
          f"max {stats['question_len_max']}) -> {out.name} ({out.stat().st_size / 2**20:.1f} MB)")

    # Three rendered records per split, to read the format by eye.
    for i in range(0, len(cols["state_ids"]), max(1, len(cols["state_ids"]) // 3))[:3]:
        print(f"  --- {cols['source'][i]} ({cols['variant'][i]})")
        print("  " + decode(enc, cols["state_ids"][i])[:400])
        for name, prefix, opts, label in zip(cols["q_name"][i], cols["q_prefix"][i],
                                             cols["q_options"][i], cols["q_label"][i]):
            shown = decode(enc, prefix + [t for o in opts[:6] for t in o])
            more = f" ... (+{len(opts) - 6} options)" if len(opts) > 6 else ""
            print(f"  [{name}, label {label}] {shown[:400]}{more}")
    return stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = load_tokenizer()
    meta = dict(bpe_vocab=BPE_VOCAB, d_vocab=D_VOCAB, special_tokens=SPECIAL_IDS,
                tokenizer=f"{CLIMBMIX_REPO}@{CLIMBMIX_REV}:tokenizer/tokenizer.pkl",
                kev_hf=f"{KEV_HF_REPO}@{KEV_HF_REV}", kev_github=KEV_GH, splits={})
    for split, files in SPLITS.items():
        meta["splits"][split] = build(enc, split, files)
    (OUT_DIR / "kev_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"-> {OUT_DIR / 'kev_meta.json'}")


if __name__ == "__main__":
    main()
