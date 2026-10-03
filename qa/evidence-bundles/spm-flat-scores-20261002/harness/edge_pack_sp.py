#!/usr/bin/env python3
"""Tokenize the TinyLlama chat-template edge pack with the model's own sentencepiece
tokenizer, the way llama.cpp's reference run did (BOS added, `</s>` and `<s>` parsed
as control tokens, sentencepiece's dummy prefix on every text segment), and compare
with the pack's llama.cpp ids and with a camelid build.

usage: edge_pack_sp.py <pack.json> <camelid binary> <gguf> <out.json>
"""
import json
import re
import subprocess
import sys
import tempfile

import sentencepiece as spm
from huggingface_hub import hf_hub_download

PACK, BINARY, GGUF, OUT = sys.argv[1:5]
sp = spm.SentencePieceProcessor(model_file=hf_hub_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tokenizer.model"))
SPECIAL = {"<s>": 1, "</s>": 2}


def sentencepiece_ids(text):
    ids = [1]
    for part in re.split(r"(<s>|</s>)", text):
        if part in SPECIAL:
            ids.append(SPECIAL[part])
        elif part:
            ids.extend(sp.encode(part))
    return ids


cases = json.load(open(PACK))["cases"]
names = list(cases)
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump([cases[name]["expected_prompt"] for name in names], fh)
camelid_rows = [json.loads(line) for line in subprocess.run(
    [BINARY, "tokenize", "--model", GGUF, "--file", fh.name, "--parse-special"],
    capture_output=True, text=True, check=True).stdout.splitlines() if line.startswith("{")]
report = []
for name, row in zip(names, camelid_rows):
    reference = sentencepiece_ids(cases[name]["expected_prompt"])
    llama_cpp = cases[name]["tokens"]
    camelid_ids = row["ids"]
    entry = {"case": name, "sentencepiece_vs_llama_cpp": reference == llama_cpp,
             "sentencepiece_vs_camelid": reference == camelid_ids}
    for label, ids in (("llama_cpp", llama_cpp), ("camelid", camelid_ids)):
        if ids != reference:
            at = next((i for i, (a, b) in enumerate(zip(ids, reference)) if a != b), min(len(ids), len(reference)))
            entry[f"{label}_first_difference"] = {"index": at, label: ids[at:at + 4], "sentencepiece": reference[at:at + 4],
                                                  "pieces_sentencepiece": [sp.id_to_piece(i) for i in reference[at:at + 4]],
                                                  f"pieces_{label}": [sp.id_to_piece(i) for i in ids[at:at + 4]]}
    report.append(entry)
    print(json.dumps(entry, ensure_ascii=False))
json.dump({"binary": BINARY, "cases": report}, open(OUT, "w"), indent=1, ensure_ascii=False)
