#!/usr/bin/env python3
"""Compare tokenizers on the corpus against the model's own tokenizer (sentencepiece,
TinyLlama's tokenizer.model). No special tokens are added or parsed anywhere.

usage: compare_spm.py <corpus.json> <out.json> [camelid:<label>:<binary>]... [llamacpp:<label>:<llama-tokenize>]
env: GGUF (the TinyLlama GGUF), HF_HOME
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

import sentencepiece as spm
from huggingface_hub import hf_hub_download

CORPUS, OUT, SPECS = sys.argv[1], sys.argv[2], sys.argv[3:]
GGUF = os.environ["GGUF"]
texts = json.load(open(CORPUS, encoding="utf-8"))
sources = json.load(open(CORPUS.replace(".json", "-sources.json")))
REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_file = hf_hub_download(REPO, "tokenizer.model")
sp = spm.SentencePieceProcessor(model_file=model_file)
reference = [sp.encode(text) for text in texts]


def camelid(binary):
    out = subprocess.run([binary, "tokenize", "--model", GGUF, "--file", CORPUS, "--no-add-special"],
                         capture_output=True, text=True, check=True).stdout
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert len(rows) == len(texts), (len(rows), len(texts))
    return [row["ids"] for row in rows]


def llamacpp(binary):
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prompt.txt")
        for text in texts:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            out = subprocess.run([binary, "-m", GGUF, "-f", path, "--ids", "--no-bos", "--no-escape", "--no-parse-special",
                                  "--log-disable"],
                                 capture_output=True, text=True, check=True).stdout.strip().splitlines()
            results.append(ast.literal_eval(out[-1]) if out else [])
    return results


report = {"reference": f"sentencepiece {spm.__version__} on {REPO} tokenizer.model", "gguf": os.path.basename(GGUF),
          "texts": len(texts), "tokenizers": {}}
all_ids = {}
for spec in SPECS:
    kind, label, binary = spec.split(":", 2)
    ids = camelid(binary) if kind == "camelid" else llamacpp(binary)
    all_ids[label] = ids
    agree = [a == b for a, b in zip(ids, reference)]
    by_source = {}
    for source, ok in zip(sources, agree):
        entry = by_source.setdefault(source, [0, 0])
        entry[0] += ok
        entry[1] += 1
    report["tokenizers"][label] = {
        "agree": sum(agree), "by_source": by_source,
        "first_disagreements": [{"text": texts[i][:80], "reference": reference[i][:16], "got": ids[i][:16]}
                                for i, ok in enumerate(agree) if not ok][:8],
    }
    print(f"{label:28} agrees with sentencepiece on {sum(agree)}/{len(texts)}  {by_source}", flush=True)
labels = list(all_ids)
report["identical_ids_between"] = [
    {"a": a, "b": b, "texts": sum(x == y for x, y in zip(all_ids[a], all_ids[b]))}
    for i, a in enumerate(labels) for b in labels[i + 1:]
]
print(report["identical_ids_between"])
json.dump(report, open(OUT, "w"), indent=1, ensure_ascii=False)
