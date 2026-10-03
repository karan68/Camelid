#!/usr/bin/env python3
"""For each GGUF: tokenizer model, vocab size, merges, and the spread of its token scores (JSON lines)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spm_probe_lib import read_gguf_metadata  # noqa: E402

for path in sys.argv[1:]:
    _, meta = read_gguf_metadata(path)
    scores = meta.get("tokenizer.ggml.scores") or []
    print(json.dumps({
        "file": os.path.basename(path),
        "model": meta.get("tokenizer.ggml.model"),
        "vocab": len(meta.get("tokenizer.ggml.tokens", [])),
        "merges": len(meta.get("tokenizer.ggml.merges") or []),
        "distinct_scores": len(set(scores)),
        "score_range": [min(scores), max(scores)] if scores else None,
    }))
