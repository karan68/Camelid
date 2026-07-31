#!/usr/bin/env python3
"""Build the tokenizer parity case list.

Over-inclusive by design: every string value found anywhere in the committed
prompt packs becomes a case, plus a block of synthetic adversarial strings that
stress the special-token matcher (boundaries, overlaps, partial markers,
multi-byte neighbours). Both parse_special modes are exercised per case by the
Rust probe, so the gate covers the special path AND the plain path.
"""
import json
import os
import sys

PACK_DIR = sys.argv[1] if len(sys.argv) > 1 else "qa/prompt-packs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "cases.json"

# The three packs the task names, first, so they lead the case list.
NAMED = [
    "gemma3-chat-template-shapes-v1.json",
    "gemma3-chat-gate-pack-v1.json",
    "gemma3-windowed-context-pack-v1.json",
]

cases = []
seen = set()


def add(name, text):
    if not isinstance(text, str) or text == "":
        return
    key = text
    if key in seen:
        return
    seen.add(key)
    cases.append({"name": name, "text": text})


def walk(node, path, name_prefix):
    if isinstance(node, str):
        add(f"{name_prefix}:{path}", node)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}[{i}]", name_prefix)
    elif isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else k, name_prefix)


pack_files = NAMED + sorted(
    f for f in os.listdir(PACK_DIR) if f.endswith(".json") and f not in NAMED
)
for fname in pack_files:
    p = os.path.join(PACK_DIR, fname)
    if not os.path.exists(p):
        print(f"WARN missing pack {p}", file=sys.stderr)
        continue
    with open(p) as fh:
        try:
            data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN unparsable {p}: {exc}", file=sys.stderr)
            continue
    walk(data, "", fname[:-5])

pack_case_count = len(cases)

# ---- synthetic adversarial cases -------------------------------------------
# Marker vocabulary spanning every family in tree. A marker that does not exist
# in a given model's vocab is still a useful case: it must tokenize as plain
# text identically before and after.
MARKERS = [
    "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<pad>", "<unk>",
    "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>",
    "</tool_call>", "<|begin_of_text|>", "<|end_of_text|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    "[INST]", "[/INST]", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|system|>", "<turn|>", "<|turn>", "<unused0>", "<mask>", "<2mass>",
]

for m in MARKERS:
    add(f"syn:bare:{m}", m)
    add(f"syn:lead:{m}", m + "hello world")
    add(f"syn:trail:{m}", "hello world" + m)
    add(f"syn:wrap:{m}", m + "hello world" + m)
    add(f"syn:nl:{m}", m + "\nhello\n" + m + "\n")
    add(f"syn:space:{m}", " " + m + " x " + m + " ")
    add(f"syn:adjacent:{m}", m + m + m)
    add(f"syn:partial-open:{m}", m[:-1] + "hello")
    add(f"syn:partial-close:{m}", "hello" + m[1:])
    add(f"syn:utf8:{m}", "éé" + m + "中文\U0001F600" + m + "ß")
    add(f"syn:tab:{m}", m + "\t\t  x")

# Overlap / longest-match stress: prefixes of one marker inside another.
add("syn:overlap:think", "<th<think>ink</think></th>")
add("syn:overlap:im", "<|im_<|im_start|>start|><|im_end|>")
add("syn:overlap:turn", "<start_of_<start_of_turn>turn>")
add("syn:lt-run", "<<<<<<<<<<>>>>>>>>>>")
add("syn:angle-soup", "a<b<c<|d|>e<f>g</h>i<start_of_turn>j")
add("syn:only-lt", "<")
add("syn:only-gt", ">")
add("syn:pipe", "|>|<|")
add("syn:ws-only", "   \n\t  ")
add("syn:ctrl-chars", "a\x01b\x1fc\x7fd")
add("syn:emoji", "\U0001F600\U0001F1EC\U0001F1E7 hello")
add("syn:combining", "áé nfc/nfd")
add("syn:long-plain", "The quick brown fox jumps over the lazy dog. " * 40)
add(
    "syn:long-mixed",
    ("<start_of_turn>user\nThe quick brown fox jumps over the lazy dog.<end_of_turn>\n"
     "<start_of_turn>model\nA reply sentence here.<end_of_turn>\n") * 20,
)
add(
    "syn:long-chatml",
    ("<|im_start|>user\nExplain gradient descent briefly.<|im_end|>\n"
     "<|im_start|>assistant\nIt walks downhill.<|im_end|>\n") * 20,
)
add(
    "syn:long-llama3",
    ("<|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>"
     "<|start_header_id|>assistant<|end_header_id|>\n\n4<|eot_id|>") * 20,
)
add("syn:long-inst", ("<s>[INST] Say hi. [/INST] Hi.</s>") * 30)

with open(OUT, "w") as fh:
    json.dump({"cases": cases}, fh)

print(f"{len(cases)} cases ({pack_case_count} from packs, "
      f"{len(cases) - pack_case_count} synthetic) -> {OUT}")
