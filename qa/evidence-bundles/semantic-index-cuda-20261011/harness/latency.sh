# latency.sh <binary> <label> [data dir]: every search mode on the indexed library, after the auto-loaded model is ready.
R=~/r801
DATA=${3:-$R/final}
bash $R/serve.sh $1 $DATA 18847 > /dev/null 2>&1 || exit 1
for _ in $(seq 1 300); do curl -s 127.0.0.1:18847/v1/health | grep -q '"generation_ready":true' && break; sleep 1; done
sleep 10
python3 $R/retrieve_modes.py 18847 ~/f2a-500/run-799/out ~/f2a-500/run-799/corpus $R/retrieve-$2.json > $R/latency-$2.log 2>&1
python3 - >> $R/latency-$2.log <<'PY'
import json, os, time, urllib.request, statistics
state = json.load(open(os.path.expanduser("~/f2a-500/run-799/out/state.json")))
facts = json.load(open(os.path.expanduser("~/f2a-500/run-799/corpus/planted-facts.json")))
doc = state["planted"][0]["doc_id"]
def search(body):
    req = urllib.request.Request("http://127.0.0.1:18847/api/documents/search", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:18847"})
    t = time.perf_counter(); urllib.request.urlopen(req, timeout=120).read(); return (time.perf_counter() - t) * 1000
for name, extra in [("one doc, semantic", {"doc_ids": [doc], "mode": "semantic"}), ("one doc, keyword", {"doc_ids": [doc], "mode": "keyword"}),
                    ("whole collection, semantic", {"collection_ids": [state["collection_id"]], "mode": "semantic"})]:
    ms = [search({"query": f["question"], "top_k": 5, **extra}) for f in facts]
    print(f"split {name:28s} p50 {statistics.median(ms):7.1f} ms")
PY
kill $(cat $DATA/server.pid)
echo DONE >> $R/latency-$2.log
