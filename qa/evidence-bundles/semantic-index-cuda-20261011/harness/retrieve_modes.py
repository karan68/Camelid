"""Retrieval on the 500-PDF library in every search mode (#801).

Usage: python3 retrieve_modes.py <port> <gate out dir with state.json> <corpus dir> <out json>
For each planted question: the collection search the gate runs (default mode,
which is hybrid once the library has vectors), the same search by meaning only
and by keyword only, and a whole-library search (`library: true`, no scope).
Records the rank of the passage holding the right answer, whether the decoy's
passage came first, the mode the server reported, and the search time.
"""
import json
import os
import statistics
import sys
import time
import urllib.request

PORT, GATE_OUT, CORPUS, OUT = sys.argv[1:5]
BASE = f"http://127.0.0.1:{PORT}"
state = json.load(open(os.path.join(GATE_OUT, "state.json")))
facts = json.load(open(os.path.join(CORPUS, "planted-facts.json")))
by_file = {p["file"]: p["doc_id"] for p in state["planted"]}


def search(body):
    req = urllib.request.Request(BASE + "/api/documents/search", data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json", "Origin": BASE})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        res = json.loads(resp.read())
    return res, (time.perf_counter() - started) * 1000


VARIANTS = {
    "collection_default": lambda q: {"query": q, "collection_ids": [state["collection_id"]], "top_k": 5},
    "collection_semantic": lambda q: {"query": q, "collection_ids": [state["collection_id"]], "top_k": 5, "mode": "semantic"},
    "collection_keyword": lambda q: {"query": q, "collection_ids": [state["collection_id"]], "top_k": 5, "mode": "keyword"},
    "library": lambda q: {"query": q, "top_k": 5, "library": True},
}

report = {}
for name, body_for in VARIANTS.items():
    rows, times = [], []
    for fact in facts:
        res, ms = search(body_for(fact["question"]))
        times.append(ms)
        results = res.get("results", [])
        rank = next((i + 1 for i, r in enumerate(results)
                     if r["doc_id"] == by_file[fact["file"]] and fact["answer"] in r["excerpt"]), None)
        decoy_first = bool(results) and results[0]["doc_id"] == by_file[fact["decoy_file"]] \
            and fact["decoy_answer"] in results[0]["excerpt"]
        retrieval = res.get("retrieval", {})
        rows.append({"id": fact["id"], "rank": rank, "decoy_first": decoy_first,
                     "mode": retrieval.get("mode"), "results": len(results), "ms": round(ms, 1)})
    times_sorted = sorted(times)
    report[name] = {
        "questions": len(rows),
        "first": sum(1 for r in rows if r["rank"] == 1),
        "top5": sum(1 for r in rows if r["rank"]),
        "decoy_first": sum(1 for r in rows if r["decoy_first"]),
        "modes": sorted({r["mode"] for r in rows if r["mode"]}),
        "ms_p50": round(statistics.median(times), 1),
        "ms_p95": round(times_sorted[int(len(times_sorted) * 0.95) - 1], 1),
        "rows": rows,
    }
    print(name, json.dumps({k: v for k, v in report[name].items() if k != "rows"}), flush=True)
json.dump(report, open(OUT, "w"), indent=1)
