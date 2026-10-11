#!/usr/bin/env bash
# final.sh: the final build on the 550-PDF library: index, search in every mode, chat.
set -u
R=~/r801
SRC=~/f2a-500/run-799
CORPUS=$SRC/corpus
BIN=~/camelid-801-fix5
rows() { python3 -c "import sqlite3; c=sqlite3.connect('file:$1/documents_rag.sqlite3?mode=ro', uri=True); print(c.execute('select count(*) from document_chunk_vectors').fetchone()[0])"; }
vectors() { python3 -c "import sqlite3; c=sqlite3.connect('file:$1/documents_rag.sqlite3?mode=ro', uri=True); print(c.execute('select count(*) from document_chunk_vectors where vector is not null').fetchone()[0])"; }
chunks() { python3 -c "import sqlite3; c=sqlite3.connect('file:$1/documents_rag.sqlite3?mode=ro', uri=True); print(c.execute('select count(*) from document_chunks').fetchone()[0])"; }
wait_ready() { for _ in $(seq 1 300); do curl -s "127.0.0.1:$1/v1/health" | grep -q "$2" && return; sleep 1; done; }

# 1. Index a fresh copy.
rm -rf $R/final5 && mkdir -p $R/final5 && cp $SRC/data/documents_rag.sqlite3 $R/final5/
bash $R/serve.sh $BIN $R/final5 18844 > $R/final5-index.log 2>&1 || exit 1
wait_ready 18844 '"ok":true'
total=$(chunks $R/final5)
start=$(date +%s)
curl -s 127.0.0.1:18844/api/documents/index-status > /dev/null
prev=0
while :; do
  sleep 60
  now=$(rows $R/final5)
  lane=$(curl -s 127.0.0.1:18844/api/documents/index-status | python3 -c "import json,sys; s=json.load(sys.stdin)['semantic']; print(s.get('lane'), s.get('indexing'), s.get('error'))")
  gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')
  echo "t=$(( $(date +%s) - start ))s vectors $now/$total (+$((now - prev))) lane/indexing/error: $lane gpu: $gpu" >> $R/final5-index.log
  prev=$now
  [ "$now" -ge "$total" ] && break
  [ $(( $(date +%s) - start )) -gt 10800 ] && { echo "gave up after 3 h" >> $R/final5-index.log; break; }
done
echo "indexed $(rows $R/final5) of $total ($(vectors $R/final5) with a vector) in $(( $(date +%s) - start )) s" >> $R/final5-index.log
sleep 5
echo "gpu memory after indexing, idle: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')" >> $R/final5-index.log

# 2. Every search mode, twice (the second pass is the warm vector cache).
python3 $R/retrieve_modes.py 18844 $SRC/out $CORPUS $R/retrieve-final5-1.json >> $R/final5-index.log 2>&1
python3 $R/retrieve_modes.py 18844 $SRC/out $CORPUS $R/retrieve-final5-2.json >> $R/final5-index.log 2>&1
kill "$(cat $R/final5/server.pid)"; sleep 5

# 3. The gate's chat phase with hybrid retrieval, Llama 3.2 3B.
rm -rf $R/gate-out5 && cp -r $SRC/out $R/gate-out5
bash $R/serve.sh $BIN $R/final5 18845 --model ~/Camelid/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf >> $R/final5-chat.log 2>&1 || exit 1
wait_ready 18845 '"generation_ready":true'
python3 ~/f2a-500/gate.py 18845 $CORPUS $R/final5 $R/gate-out5 chat hybrid-final >> $R/final5-chat.log 2>&1
kill "$(cat $R/final5/server.pid)"
echo DONE > $R/final5.done
