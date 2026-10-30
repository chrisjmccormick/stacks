r"""Tokenize decision records (Kev's suite schema) into train_decision.py's input format, one
parquet per split, with the 16K Unigram tokenizer of the baseline-6 MLM base.

This is data/prepare_kev_suites.py (the Phase-0 fork's 32K-BPE version) moved to the
tokenizer of ChrisMcCormick/climbmix_unigram16k_14_170, which the MLM base
(agent-ops-stacks/encoder-rtx/2026-10-29_0740am_b6-mlm-speedrun, `runs/mlm-base4gb/`)
was trained on. The rendering is unchanged except that a state now opens with BOS, as
every document the base saw does:

    <|bos|><|state|>{state}
    <|choice|>{instructions}\n<|"|>name<|"|> description<|option|><|"|>name<|"|><|option|>...
    <|noul|>{instructions}\n{true criteria}<|yes|>{false criteria}<|no|>
    <|score|>{instructions}\n{level 0 text}<|level_0|>{level 1 text}<|level_1|>...

Each question is stored as a prefix (its type token and instructions) plus a list of
options, each ending in its readout token, so the trainer can permute a choice's options
without re-tokenizing. JSON states, instructions and descriptions keep their braces and
keys as text, with string values in super quotes: {policy:<|"|>...<|"|>,case:<|"|>...<|"|>}.

Special tokens. The 16K vocabulary has 15 unused ids after BOS (16,368); the MLM base took
the first for [MASK] (16,369), and the decision format reuses it as <|mask|>. The other 18
special tokens take ids 16,384-16,401, past the tokenizer's vocabulary: the trainer grows
its id-indexed input tables (the input embeddings and the value embeddings) to 16,448 rows
and leaves the LM head at 16,384, since MLM never targets a special token. The tokenizer
never emits an id at or above BOS, so user text can't forge a delimiter.

Sources are anything in Kev's suite schema: one JSON record per line, {state, questions,
_meta}. Each question may carry a soft `target` (option name -> probability); a labels
table can add one too (see LABELS below), and the trainer mixes it with the hard label.

Outputs, in ~/.cache/encoder-rtx/decision16k/ (regenerable, so not committed):
    <split>.parquet   one row per record: ids, source/variant/group, the state, per
                      question its name, type, prefix, options, label and target
    meta.json         special token ids, sources and their hashes, counts, lengths

No command line: every knob is a constant below. On a GPU box, in the `stacks` env:

    python data/prepare_decision_data.py
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

CACHE = Path.home() / ".cache" / "encoder-rtx"
RAW_DIR = CACHE / "kev-suites"
OUT_DIR = CACHE / "decision16k"

# The MLM base's tokenizer: a byte-level Unigram model in HF tokenizers' JSON.
CLIMBMIX_REPO, CLIMBMIX_REV = "ChrisMcCormick/climbmix_unigram16k_14_170", "a42ad58143e636b2ed7ee60f661ad7fdf1df5bf7"
TOKENIZER = CACHE / "climbmix_unigram16k_14_170" / "tokenizer" / "tokenizer.json"

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

# Soft labels from a teacher, joined onto the records by (record id, question name): split ->
# a parquet with columns `id`, `question` and `probs` (option name -> probability, as JSON
# text or a map). A question with a table entry gets that target in place of its own.
LABELS: dict[str, Path] = {}

# ------------------------------------------------------------------------------
# Special tokens
# ------------------------------------------------------------------------------

BOS_ID = 16368       # the tokenizer's <|bos|>, which opens every training document
MASK_ID = 16369      # the MLM base's [MASK], the first unused id after BOS
D_TOKENIZER = 16384  # the tokenizer's ids, and the LM head's rows
SPECIAL = (["<|state|>", "<|noul|>", "<|score|>", "<|choice|>", '<|"|>',
            "<|yes|>", "<|no|>", "<|option|>"]
           + [f"<|level_{i}|>" for i in range(10)])
SPECIAL_IDS = {"<|mask|>": MASK_ID, **{name: D_TOKENIZER + i for i, name in enumerate(SPECIAL)}}
D_INPUT_VOCAB = 16448   # input-table rows: the tokenizer's plus the specials, padded to a multiple of 64
assert D_TOKENIZER + len(SPECIAL) <= D_INPUT_VOCAB and D_INPUT_VOCAB % 64 == 0

STATE, QUOTE = SPECIAL_IDS["<|state|>"], SPECIAL_IDS['<|"|>']
TYPE_TOKEN = {"noul": SPECIAL_IDS["<|noul|>"], "score": SPECIAL_IDS["<|score|>"],
              "choice": SPECIAL_IDS["<|choice|>"]}
YES, NO, OPTION = SPECIAL_IDS["<|yes|>"], SPECIAL_IDS["<|no|>"], SPECIAL_IDS["<|option|>"]
LEVELS = [SPECIAL_IDS[f"<|level_{i}|>"] for i in range(10)]


# ------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------

def json_pieces(value) -> list:
    """A JSON value as pieces: str for tokenizer text, int for a special id. Braces,
    brackets, keys and non-string scalars are text; string values go in super quotes."""
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


def encode(tok, pieces: list) -> list[int]:
    """Adjacent text pieces are joined before tokenizing, so the model sees whole strings."""
    ids, text = [], ""
    for p in pieces:
        if isinstance(p, str):
            text += p
        else:
            if text:
                ids += tok.encode(text, add_special_tokens=False).ids
                text = ""
            ids.append(p)
    if text:
        ids += tok.encode(text, add_special_tokens=False).ids
    return ids


def render_question(tok, q: dict):
    """-> (prefix ids, [option ids, each ending in its readout token], label, target, names).
    label indexes the options; target is a probability per option, or [] when the
    record has only a hard label."""
    kind = q["type"]
    # The newline marks where the instructions end: a yes/no or score criterion is plain
    # text, and would otherwise run straight on from the question.
    instructions = text_pieces(q.get("instructions"))
    prefix = encode(tok, [TYPE_TOKEN[kind]] + instructions + (["\n"] if instructions else []))
    crit = q.get("criteria")
    if kind == "noul":
        crit = crit or {}
        names = ["true", "false"]
        options = [encode(tok, text_pieces(crit.get("true")) + [YES]),
                   encode(tok, text_pieces(crit.get("false")) + [NO])]
        label = 0 if q["label"] else 1
    elif kind == "score":
        assert 1 <= len(crit) <= len(LEVELS), f"score with {len(crit)} levels"
        names = [str(i) for i in range(len(crit))]
        options = [encode(tok, text_pieces(level) + [LEVELS[i]]) for i, level in enumerate(crit)]
        label = q["label"]
    else:
        names = list(crit)
        options = [encode(tok, [QUOTE, name, QUOTE]
                          + ([" "] + text_pieces(desc) if desc else []) + [OPTION])
                   for name, desc in crit.items()]
        label = names.index(q["label"])
    assert 0 <= label < len(options)
    target = []
    if "target" in q:
        target = target_list(q["target"], names)
    return prefix, options, label, target, names


def target_list(probs: dict, names: list) -> list[float]:
    """A soft target (option name -> probability) in option order. A noul question's
    options are named "true" / "false"; a teacher may also key them as booleans."""
    probs = {str(k).lower() if isinstance(k, bool) else str(k): float(v) for k, v in probs.items()}
    assert set(probs) == set(names), f"target keys {sorted(probs)} vs {names}"
    target = [probs[n] for n in names]
    assert abs(sum(target) - 1) < 1e-4, f"target sums to {sum(target)}"
    return target


# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

def fetch(where: str, path: str) -> Path:
    """The raw file, downloaded once into RAW_DIR at the pinned revision (or a local path)."""
    if where == "local":
        return Path(path).expanduser()
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
        hf_hub_download(CLIMBMIX_REPO, "tokenizer/tokenizer.json", repo_type="dataset",
                        revision=CLIMBMIX_REV, local_dir=TOKENIZER.parent.parent)
    tok = Tokenizer.from_file(str(TOKENIZER))
    assert tok.get_vocab_size() == BOS_ID, f"tokenizer has {tok.get_vocab_size()} ids, expected {BOS_ID} before BOS"
    return tok


def load_labels(split: str) -> dict:
    """(record id, question name) -> {option name: probability}, from LABELS[split]."""
    if split not in LABELS:
        return {}
    table = {}
    for row in pq.read_table(LABELS[split]).to_pylist():
        probs = row["probs"]
        probs = json.loads(probs) if isinstance(probs, str) else dict(probs)
        table[(row["id"], row["question"])] = probs
    return table


def decode(tok, ids: list[int]) -> str:
    """Readable rendering, special tokens by name: for the sample printout."""
    names = {v: k for k, v in SPECIAL_IDS.items()} | {BOS_ID: "<|bos|>"}
    out, run = [], []
    for i in ids:
        if i >= BOS_ID:
            out.append(tok.decode(run)); run = []
            out.append(names[i])
        else:
            run.append(i)
    out.append(tok.decode(run))
    return "".join(out)


def pct(xs: list[int], p: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------

def build(tok, split: str, files: list) -> dict:
    cols = {c: [] for c in ("id", "file", "line", "source", "variant", "group", "state_ids", "q_name",
                            "q_type", "q_prefix", "q_options", "q_label", "q_target")}
    labels = load_labels(split)
    sources, by_type, by_source = {}, Counter(), Counter()
    state_lens, question_lens = [], []
    joined = 0
    for where, path in files:
        local = fetch(where, path)
        sources[path] = dict(where=where, sha256=hashlib.sha256(local.read_bytes()).hexdigest())
        for line_no, line in enumerate(local.open(encoding="utf-8")):
            if not line.strip():
                continue
            r = json.loads(line)
            meta = r.get("_meta", {})
            state = encode(tok, [BOS_ID, STATE] + text_pieces(r["state"]))
            cols["id"].append(str(meta.get("id", f"{path}:{line_no}")))
            cols["file"].append(path)
            cols["line"].append(line_no)
            cols["source"].append(meta.get("source", "?"))
            cols["variant"].append(meta.get("variant", "clean"))
            cols["group"].append(str(meta.get("group_id") or meta.get("id") or meta.get("family_id") or ""))
            cols["state_ids"].append(state)
            state_lens.append(len(state))
            names, types, prefixes, options, q_labels, targets = [], [], [], [], [], []
            for name, q in r["questions"].items():
                prefix, opts, label, target, opt_names = render_question(tok, q)
                if (cols["id"][-1], name) in labels:
                    target = target_list(labels[(cols["id"][-1], name)], opt_names)
                    joined += 1
                names.append(name); types.append(q["type"]); prefixes.append(prefix)
                options.append(opts); q_labels.append(label); targets.append(target)
                question_lens.append(len(prefix) + sum(len(o) for o in opts))
                by_type[q["type"]] += 1
                by_source[meta.get("source", "?")] += 1
            for c, v in zip(("q_name", "q_type", "q_prefix", "q_options", "q_label", "q_target"),
                            (names, types, prefixes, options, q_labels, targets)):
                cols[c].append(v)

    flat = [i for s in cols["state_ids"] for i in s[1:]]
    flat += [i for ps in cols["q_prefix"] for p in ps for i in p]
    flat += [i for qs in cols["q_options"] for opts in qs for o in opts for i in o]
    assert max(flat) < D_TOKENIZER + len(SPECIAL)
    # Text never lands on BOS or the unused ids: only our formatter emits ids at or above BOS.
    assert not any(BOS_ID <= i < D_TOKENIZER for i in flat)

    out = OUT_DIR / f"{split}.parquet"
    ints = pa.list_(pa.int32())
    pq.write_table(pa.table({
        "id":        pa.array(cols["id"], pa.string()),
        "file":      pa.array(cols["file"], pa.string()),
        "line":      pa.array(cols["line"], pa.int32()),
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
        soft_targets=sum(bool(t) for ts in cols["q_target"] for t in ts), soft_targets_joined=joined,
        by_type=dict(by_type), by_source=dict(sorted(by_source.items())),
        state_tokens=sum(state_lens), question_tokens=sum(question_lens),
        state_len_p50=pct(state_lens, .5), state_len_p90=pct(state_lens, .9), state_len_max=max(state_lens),
        states_over_2048=sum(L > 2048 for L in state_lens),
        state_tokens_past_2048=sum(max(0, L - 2048) for L in state_lens),
        question_len_p50=pct(question_lens, .5), question_len_max=max(question_lens),
        sources=sources,
        # Pins the exact token stream: a tokenizer or format change moves this.
        ids_sha256=hashlib.sha256(repr((cols["state_ids"], cols["q_prefix"], cols["q_options"])).encode()).hexdigest()[:32],
    )
    print(f"[{split}] {stats['records']:,} records, {n_questions:,} questions "
          f"({', '.join(f'{k} {v:,}' for k, v in sorted(by_type.items()))}); "
          f"state tokens {stats['state_tokens']/1e6:.2f}M (p50 {stats['state_len_p50']}, "
          f"p90 {stats['state_len_p90']}, max {stats['state_len_max']}; {stats['states_over_2048']:,} over 2,048 "
          f"holding {stats['state_tokens_past_2048']/1e6:.2f}M past it), "
          f"question tokens {stats['question_tokens']/1e6:.2f}M (p50 {stats['question_len_p50']}, "
          f"max {stats['question_len_max']}); soft targets {stats['soft_targets']:,} ({joined:,} from a labels table) "
          f"-> {out.name} ({out.stat().st_size / 2**20:.1f} MB)")

    # Three rendered records per split, to read the format by eye.
    for i in range(0, len(cols["state_ids"]), max(1, len(cols["state_ids"]) // 3))[:3]:
        print(f"  --- {cols['id'][i]} ({cols['source'][i]}, {cols['variant'][i]})")
        print("  " + decode(tok, cols["state_ids"][i])[:400])
        for name, prefix, opts, label in zip(cols["q_name"][i], cols["q_prefix"][i],
                                             cols["q_options"][i], cols["q_label"][i]):
            shown = decode(tok, prefix + [t for o in opts[:6] for t in o])
            more = f" ... (+{len(opts) - 6} options)" if len(opts) > 6 else ""
            print(f"  [{name}, label {label}] {shown[:400]}{more}")
    return stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    meta = dict(bos_id=BOS_ID, d_tokenizer=D_TOKENIZER, d_input_vocab=D_INPUT_VOCAB, special_tokens=SPECIAL_IDS,
                tokenizer=f"{CLIMBMIX_REPO}@{CLIMBMIX_REV}:tokenizer/tokenizer.json",
                kev_hf=f"{KEV_HF_REPO}@{KEV_HF_REV}", kev_github=KEV_GH,
                labels={s: str(p) for s, p in LABELS.items()}, splits={})
    for split, files in SPLITS.items():
        meta["splits"][split] = build(tok, split, files)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"-> {OUT_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
