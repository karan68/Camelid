# index-rate.sh <binary> <data dir> <port> <minutes>: serve, trigger indexing, log vectors stored per minute.
BIN=$1 DATA=$2 PORT=$3 MIN=$4
bash ~/r801/serve.sh "$BIN" "$DATA" "$PORT" || exit 1
curl -s 127.0.0.1:$PORT/api/documents/index-status | python3 -c "import json,sys; print(json.load(sys.stdin)[\"semantic\"])"
count() { python3 -c "import sqlite3; c=sqlite3.connect(\"file:$DATA/documents_rag.sqlite3?mode=ro\", uri=True); print(c.execute(\"select count(*) from document_chunk_vectors\").fetchone()[0])"; }
start=$(date +%s); prev=$(count)
for m in $(seq 1 $MIN); do sleep 60; now=$(count); echo "minute $m vectors $now (+$((now-prev)))"; prev=$now; done
echo "total $(count) in $(( $(date +%s) - start )) s"
kill $(cat "$DATA/server.pid")
