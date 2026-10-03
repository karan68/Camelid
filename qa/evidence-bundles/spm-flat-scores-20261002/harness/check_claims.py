#!/usr/bin/env python3
"""Check that every number and quoted result in this bundle's README comes from the
bundle's own data, and that SHA256SUMS matches the files.

usage: python3 harness/check_claims.py
"""
import hashlib
import json
import os
import re
import sys

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []
checked = 0


def path(name):
    return os.path.join(BUNDLE, name)


def load(name):
    with open(path(name), encoding="utf-8") as fh:
        return json.load(fh)


def text(name):
    with open(path(name), encoding="utf-8") as fh:
        return fh.read()


def normalize(value):
    return re.sub(r"\s+", " ", value).strip()


README = normalize(text("README.md"))


def claim(label, needle):
    global checked
    checked += 1
    if normalize(needle) not in README:
        failures.append(f"README does not state {label}: {needle!r}")


def fact(label, condition, detail=""):
    global checked
    checked += 1
    if not condition:
        failures.append(f"{label}: {detail}" if detail else label)


# ---- the GGUFs -------------------------------------------------------------------------------------
meta = {row["file"]: row for row in map(json.loads, text("data/gguf-tokenizer-metadata.jsonl").splitlines())}
tiny = meta["tinyllama-1.1b-chat-v1.0.Q8_0.gguf"]
fact("TinyLlama is a llama SPM GGUF", tiny["model"] == "llama")
claim("TinyLlama's merges", f"carries {tiny['merges']:,} merges")
claim("TinyLlama's vocabulary", f"its {tiny['vocab']:,} scores hold {tiny['distinct_scores']} distinct value")
fact("every TinyLlama score is 0.0", tiny["score_range"] == [0.0, 0.0], tiny["score_range"])
mistral = meta["Mistral-7B-Instruct-v0.3-Q8_0.gguf"]
fact("Mistral is a llama SPM GGUF", mistral["model"] == "llama")
claim("Mistral's scores and merges", f"has {mistral['distinct_scores']:,} distinct scores and {mistral['merges']} merges")

# ---- the corpus comparison -------------------------------------------------------------------------
provenance = load("data/corpus-provenance.json")
comparison = load("data/corpus-comparison.json")
fact("the comparison covers the corpus", comparison["texts"] == provenance["texts"])
claim("the corpus size", f"The corpus is {provenance['texts']} texts")
for source, phrase in (("fixture", "the {n} cases of the HF fixture"), ("probe", "{n} hand-written probes"),
                       ("offtopic", "the {n} off-topic chat messages"), ("scifact", "{n} SciFact claims"),
                       ("fiqa", "{n} FiQA questions"), ("rust", "{n} lines each of Rust and JavaScript")):
    claim(f"the {source} count", phrase.format(n=provenance["by_source"][source]))
fact("as many Rust as JavaScript lines", provenance["by_source"]["rust"] == provenance["by_source"]["javascript"])
claim("the reference", comparison["reference"].replace(" on TinyLlama/TinyLlama-1.1B-Chat-v1.0 tokenizer.model", ""))
for label, row in comparison["tokenizers"].items():
    claim(f"{label}'s agreement", f"{row['agree']} of {comparison['texts']}")
    name, version = label.split(" ", 1)
    claim(f"{label} in the table", f"`{version}`")
fix = next(row for label, row in comparison["tokenizers"].items() if "cd81d7fb" in label)
fact("the fix's one disagreement is the literal <s> line", len(fix["first_disagreements"]) == comparison["texts"] - fix["agree"] == 1
     and "<s>" in fix["first_disagreements"][0]["text"])
main_vs_llama = next(pair for pair in comparison["identical_ids_between"] if "c7ea906a" in pair["a"] and "llama.cpp" in pair["b"])
claim("main and llama.cpp identical", f"identical ids for {main_vs_llama['texts']} of the {comparison['texts']} texts")

# ---- the edge pack ---------------------------------------------------------------------------------
for name, phrase in (("main", "agrees with those ids on 0 of 5"), ("fix", "the model's tokenizer on 5 of 5")):
    cases = load(f"data/edge-pack-{name}.json")["cases"]
    matching = sum(case["sentencepiece_vs_camelid"] for case in cases)
    fact(f"edge pack {name}", (name == "main" and matching == 0) or (name == "fix" and matching == len(cases) == 5), matching)
    fact(f"llama.cpp's ids never match the model ({name} run)", sum(case["sentencepiece_vs_llama_cpp"] for case in cases) == 0)
claim("the edge pack results", "agrees with those ids on 0 of 5")
claim("the edge pack results after", "camelid agrees with the model's tokenizer on 5 of 5")
example = load("data/edge-pack-main.json")["cases"][0]["llama_cpp_first_difference"]
fact("the camelid example", example["pieces_sentencepiece"][:3] == ["▁cam", "el", "id"] and example["pieces_llama_cpp"][:3] == ["▁came", "li", "d"], example)
claim("the camelid example", "`▁cam|el|id`, and llama.cpp as `▁came|li|d`")

# ---- tests -------------------------------------------------------------------------------------------
full = text("tests/full-suite.txt")
run = re.search(r"passed (\d+) failed (\d+) ignored (\d+) binaries (\d+)", full)
claim("the full suite", f"{run.group(1)} passed, {run.group(2)} failed, {run.group(3)} ignored, across {run.group(4)} test binaries")
fact("its one failure was the llama.cpp edge-pack test", run.group(2) == "1" and
     "encodes_tinyllama_edge_reference_pack_like_llama_cpp_when_available ... FAILED" in full)
targets = text("tests/tokenizer-targets.txt")
counts = sorted(int(n) for n in re.findall(r"test result: ok\. (\d+) passed; 0 failed", targets))
fact("both tokenizer targets pass", counts == [2, 28], counts)
claim("the tokenizer targets", "`--test tokenizer` (28) and `--test runnable_tokenizer` (2) all pass")
causal = text("tests/causal.txt")
for test in ("spm_tokenizer_matches_hf", "llama_spm_with_flat_scores_merges_by_rank",
             "encodes_tinyllama_edge_reference_pack_like_its_own_tokenizer_when_available"):
    fact(f"{test} fails with the gate reverted", f"test {test} ... FAILED" in causal)
fact("13 of 30 mismatch with the gate reverted", "encode_mismatches=13/30" in causal)
claim("the causal count", "13 of 30 cases mismatch")
fact("the edge test fails 5 of 5 with the gate reverted", "edge parity failed for 5/5" in causal)
checks = text("tests/checks.txt")
fact("fmt and clippy clean", all(f"{key}=0" in checks for key in ("FMT_EXIT", "CLIPPY_ALL_EXIT", "CLIPPY_DEFAULT_EXIT")))
fact("all gates pass", "validation gates: 47 passed, 0 failed" in checks)
claim("the gates", "all 47 validation gates pass")
fact("the generator reproduces the fixture", "fixture matches the model's own tokenizer" in checks)
smoke = {name: load(f"tests/smoke-{name}.json") for name in ("main", "fix")}
fact("both smokes ran the same request", smoke["main"]["request"] == smoke["fix"]["request"])
fact("the smoke prompt tokenizes identically", smoke["main"]["result"]["prompt_token_ids"] == smoke["fix"]["result"]["prompt_token_ids"])
fact("the smoke answers are identical", smoke["main"]["result"]["generated_token_ids"] == smoke["fix"]["result"]["generated_token_ids"])

# ---- checksums ---------------------------------------------------------------------------------------
listed = {}
for line in text("SHA256SUMS").splitlines():
    digest, name = line.split("  ", 1)
    listed[name] = digest
on_disk = sorted(os.path.relpath(os.path.join(root, f), BUNDLE) for root, _, files in os.walk(BUNDLE)
                 for f in files if f != "SHA256SUMS" and "__pycache__" not in root)
fact("SHA256SUMS lists every file", sorted(listed) == on_disk, set(on_disk) ^ set(listed))
for name, digest in listed.items():
    with open(path(name), "rb") as fh:
        fact(f"checksum of {name}", hashlib.sha256(fh.read()).hexdigest() == digest)

if failures:
    print("\n".join(failures))
    print(f"{len(failures)} of {checked} claims failed")
    sys.exit(1)
print(f"all {checked} claims hold")
