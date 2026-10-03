#!/usr/bin/env python3
"""Record TinyLlama's own-tokenizer ids in the chat-template edge pack.

`fixtures/tokenizer/tinyllama-chat-template-edge-cases.json` holds llama.cpp's ids
for each case (`tokens`). Every token score in this GGUF is 0.0, so a tokenizer
that merges by score, as llama.cpp does, falls back to leftmost-first merging and
does not reproduce the model's own SentencePiece tokenizer. This script adds that
tokenizer's ids to each case as `model_tokenizer_tokens`, tokenizing the way the
llama.cpp run did: BOS first, `<s>` and `</s>` parsed as control tokens, and every
text segment encoded by SentencePiece, which applies its dummy prefix.

usage: python3 scripts/gen-tinyllama-sentencepiece-reference.py [--check]
  --check  regenerate in memory and fail if the committed fixture differs
requires: sentencepiece, huggingface_hub
"""
import hashlib
import importlib.metadata
import json
import os
import re
import sys

import sentencepiece
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "fixtures", "tokenizer", "tinyllama-chat-template-edge-cases.json")
HF_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CONTROL = {"<s>": 1, "</s>": 2}

model_path = hf_hub_download(HF_REPO, "tokenizer.model")
with open(model_path, "rb") as fh:
    model_sha256 = hashlib.sha256(fh.read()).hexdigest()
processor = sentencepiece.SentencePieceProcessor(model_file=model_path)


def encode(text):
    ids = [1]
    for part in re.split(r"(<s>|</s>)", text):
        if part in CONTROL:
            ids.append(CONTROL[part])
        elif part:
            ids.extend(processor.encode(part))
    return ids


with open(FIXTURE, encoding="utf-8") as fh:
    committed = fh.read()
fixture = json.loads(committed)
fixture["model_tokenizer_reference"] = {
    "source": f"{HF_REPO} tokenizer.model",
    "tokenizer_model_sha256": model_sha256,
    "tool": f"sentencepiece {importlib.metadata.version('sentencepiece')}",
    "script": "scripts/gen-tinyllama-sentencepiece-reference.py",
    "segmentation": "BOS (1) first; <s> and </s> parsed as control tokens (1, 2); "
                    "each text segment encoded by SentencePieceProcessor.encode, which applies the dummy prefix",
    "why": "Every score in this GGUF is 0.0, so score-ordered merging (llama.cpp) falls back to "
           "leftmost-first and its ids in `tokens` differ from the model's own tokenizer. "
           "camelid merges this GGUF by its merge ranks and is held to `model_tokenizer_tokens`.",
}
for case in fixture["cases"].values():
    case["model_tokenizer_tokens"] = encode(case["expected_prompt"])

generated = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
if "--check" in sys.argv:
    if generated != committed:
        sys.exit("fixture differs from what this script generates; rerun without --check")
    print("fixture matches the model's own tokenizer")
else:
    with open(FIXTURE, "w", encoding="utf-8") as fh:
        fh.write(generated)
    for name, case in fixture["cases"].items():
        same = case["model_tokenizer_tokens"] == case["tokens"]
        print(f"{name}: {len(case['model_tokenizer_tokens'])} ids, {'same as' if same else 'differs from'} llama.cpp")
