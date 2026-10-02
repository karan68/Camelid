#!/usr/bin/env python3
"""Test the proposed SPM fix in Python before touching Rust: when a llama-SPM GGUF's scores
are all equal, merge by the GGUF's merge ranks (HF tokenizers BPE semantics). Compare that,
today's HF fast tokenizer and raw sentencepiece against the fixture, case by case.

usage: spm_rank_probe.py <gguf> <fixture.json>
"""
import json
import sys

sys.path.insert(0, "/tmp/f2r")
from spm_probe_lib import read_gguf_metadata  # noqa: E402

GGUF, FIXTURE = sys.argv[1], sys.argv[2]
_, meta = read_gguf_metadata(GGUF)
tokens = meta["tokenizer.ggml.tokens"]
token_id = {t: i for i, t in enumerate(tokens)}
ranks = {}
for rank, merge in enumerate(meta["tokenizer.ggml.merges"]):
    left, right = merge.split(" ", 1)
    ranks.setdefault((left, right), rank)


def byte_fallback(piece):
    return [token_id[f"<0x{b:02X}>"] for b in piece.encode("utf-8")]


def rank_bpe(text):
    word = "▁" + text.replace(" ", "▁")
    symbols = list(word)
    while len(symbols) > 1:
        best = min(((ranks.get((symbols[i], symbols[i + 1]), None), i) for i in range(len(symbols) - 1)),
                   key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0, item[1]))
        if best[0] is None:
            break
        i = best[1]
        symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
    out = []
    for s in symbols:
        out.extend([token_id[s]] if s in token_id else byte_fallback(s))
    return out


fixture = json.load(open(FIXTURE))
print("fixture meta:", {k: v for k, v in fixture.items() if k != "corpus"})
from transformers import AutoTokenizer  # noqa: E402
import transformers, tokenizers  # noqa: E402,E401
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
fast = AutoTokenizer.from_pretrained(fixture["hf_repo"], use_fast=True)
try:
    import sentencepiece as spm  # noqa: E402
    from huggingface_hub import hf_hub_download  # noqa: E402
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(fixture["hf_repo"], "tokenizer.model"))
except Exception as error:  # noqa: BLE001
    sp = None
    print("sentencepiece unavailable:", error)

agree = {"rank_bpe": 0, "hf_fast_now": 0, "sentencepiece": 0}
for case in fixture["corpus"]:
    want = case["ids"]
    got = {"rank_bpe": rank_bpe(case["text"]), "hf_fast_now": fast.encode(case["text"], add_special_tokens=False)}
    if sp is not None:
        got["sentencepiece"] = sp.encode(case["text"])
    for name, ids in got.items():
        agree[name] += ids == want
    if any(ids != want for ids in got.values()):
        print(f"case {case['text']!r:.70}")
        print(f"   fixture       {want[:14]}")
        for name, ids in got.items():
            if ids != want:
                print(f"   {name:13} {ids[:14]}")
print(f"of {len(fixture['corpus'])} cases:", agree)
