
### `data/sft_dataset.py` — SFT data pre-tokenization

This script loads all 6 data sources from the nanochat SFT mixture, formats conversations, tokenizes them, and writes paired `.bin` shards:

**Data sources (train):**
- **SmolTalk** train (~460K) — passthrough conversations from `HuggingFaceTB/smol-smoltalk`
- **MMLU** auxiliary_train (~100K) — formatted via `render_mc`, assistant answers with letter
- **GSM8K** train x2 (~16K) — parsed `<<expr=result>>` tool calls into python/output parts
- **CustomJSON** x2 (~2K) — identity conversations from local JSONL file (optional)
- **SimpleSpelling** (200K) — synthetic: "Spell the word: X" -> "X:a,p,p,l,e"
- **SpellingBee** (80K) — synthetic: letter counting with manual + Python verification

**Data sources (val):** SmolTalk test (24K), MMLU test (capped 5.2K), GSM8K test (capped 420)

**Key design:**
- Uses `tokenizer.render_conversation()` which returns `(ids, mask)` — both saved
- Shard pairs: `sft_train_000000.bin` (tokens) + `sft_train_000000_mask.bin` (supervision mask)
- Uses identical `write_datafile()` format (magic=20240520, version=1, uint16)
- Deterministic shuffle (seed=42) matches TaskMixture behavior
- Val processed first, then train (clean split boundary)

---

### `data/chat_eval_dataset.py` — Chat categorical eval pre-tokenization

This script pre-tokenizes MMLU, ARC-Easy, and ARC-Challenge for single-token logit evaluation:

**Tasks:**
- **MMLU** (all, test) — ~14K problems, 4 answer letters each
- **ARC-Easy** (test) — ~2.4K problems, variable answer letters
- **ARC-Challenge** (test) — ~1.2K problems, variable answer letters

**Per problem:** Formats as chat conversation via `render_mc`, calls `tokenizer.render_for_completion()` (strips assistant message, appends `<|assistant_start|>`), records `answer_pos` (last token), `letter_token_ids`, and `gold` index.

**Key difference from CORE eval:** Creates 1 sequence per problem (not N per problem). At eval time, the training script compares single-token logits at the answer position against the letter token IDs — simpler and more natural for chat models.