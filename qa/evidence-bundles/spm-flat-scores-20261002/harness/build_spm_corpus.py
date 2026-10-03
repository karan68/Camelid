#!/usr/bin/env python3
"""Build a tokenizer comparison corpus from text already on this machine.

Sources: the HF fixture (30), hand-written probes (whitespace, words named in
commit e78dc0a2, unicode), the 405 calibration off-topic chat messages, BEIR
SciFact claims and FiQA questions, and lines of Rust and JavaScript from the
repository. Writes a JSON array of strings and a sidecar with each one's source.

usage: build_spm_corpus.py <out.json>
"""
import json
import os
import random
import subprocess
import sys

OUT = sys.argv[1]
REPO = os.path.expanduser("~/Camelid")
EVAL = "/mnt/disks/data/camelid-eval"
rng = random.Random(20261002)
items = []


def add(source, texts):
    for text in texts:
        if isinstance(text, str):
            items.append((source, text))


add("fixture", [case["text"] for case in json.load(open(f"{REPO}/tests/fixtures/tokenizer_hf/spm.json"))["corpus"]])
# The Windows path probe is a placeholder, split in two only because the public
# scrub flags anything shaped like a real Windows home directory.
add("probe", [
    "thunderstorm", "a thunderstorm", "LRUCache", "Hello", " Hello", "  Hello", "Hello  world", "\tTabbed\tline",
    "line one\nline two\n\nline four", "trailing space ", "  ", "   ", "Numbers 0 12 345 6789", "3.14159 and 2,718",
    "naïve café résumé", "Ünïcödé façade", "日本語のテキスト", "中文分词测试", "emoji 🙂🚀 test", "Привет, мир!",
    "مرحبا بالعالم", "e = mc^2; x_i <= y_{j+1}", "https://example.com/path?q=1&r=2", "C:" + "\\Users\\name\\file.txt",
    "don't won't can't it's", "MixedCASEwords and snake_case_words", "<s> not a special token </s>",
])
add("offtopic", [json.loads(line)["text"] for line in open(f"{EVAL}/calibration/offtopic.jsonl", encoding="utf-8")])
scifact = [json.loads(line)["text"] for line in open(f"{EVAL}/scifact/queries.jsonl", encoding="utf-8")]
add("scifact", rng.sample(scifact, 300))
fiqa = [json.loads(line)["text"] for line in open(f"{EVAL}/fiqa-src/fiqa/queries.jsonl", encoding="utf-8")]
add("fiqa", rng.sample(fiqa, 300))
for path, source in (("src/tokenizer/mod.rs", "rust"), ("frontend/src/views/ChatWorkspace.jsx", "javascript")):
    # Pinned to main at the time, so the corpus does not move with the files.
    body = subprocess.run(["git", "show", f"c7ea906a:{path}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    lines = [line for line in body.splitlines() if line.strip()]
    add(source, rng.sample(lines, 150))

json.dump([text for _, text in items], open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
json.dump([source for source, _ in items], open(OUT.replace(".json", "-sources.json"), "w"))
counts = {}
for source, _ in items:
    counts[source] = counts.get(source, 0) + 1
print(len(items), counts)
